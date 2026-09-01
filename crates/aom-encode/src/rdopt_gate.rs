//! The MASTER MODE/REFERENCE GATE of libaom's inter RD brain
//! (`inter_mode_search_order_independent_skip`, `av1/encoder/rdopt.c:4643`),
//! and the three small helpers around it.
//!
//! This is the predicate the inter search runs FIRST for every
//! `(mode, reference pair)` it considers: everything it rejects is never
//! searched, so a divergence here changes the candidate set before any RD is
//! computed.
//!
//! Tier 1c (all `static`; the oracle is libaom's own rdopt.c compiled into the
//! shim archive). Gate: `crates/aom-encode/tests/rdopt_gate_diff.rs`.
//!
//! | Rust | C (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | [`inter_mode_search_order_independent_skip`] | `:4643` |
//! | [`prune_ref_frame`] | `:4284` |
//! | [`record_best_compound`] | `:5440` |
//! | [`init_mbmi`] | `:4795` |
//! | [`WinnerModeSource`] | `get_winner_mode_stats` `:3835` |
//!
//! # Scope note — one dependency is NOT in this file
//!
//! `prune_ref_frame` calls `prune_ref_by_selective_ref_frame`
//! (`av1/encoder/rdopt.**h**:236`), which is a different translation unit and
//! not part of the rdopt.c surface. It is not ported here; [`prune_ref_frame`]
//! takes its result as an argument, and the differential pins
//! `sf.inter_sf.selective_ref_frame = 0` so C's own call returns 0. The
//! `cpi->prune_ref_frame_mask` half IS covered. Anyone porting rdopt.h should
//! wire the real predicate into this argument.

use crate::rdopt_mv::PredMode;
use crate::rdopt_single_state::skip_repeated_mv;
use crate::rdopt_skip::{
    ModeSkipMask, is_ref_frame_used_by_compound_ref, is_ref_frame_used_in_cache,
    match_ref_frame_pair,
};

/// `PARTITION_NONE`.
pub const PARTITION_NONE: i32 = 0;
/// `ALTREF_FRAME`.
const ALTREF_FRAME: i32 = 7;
/// `QINDEX_RANGE`.
const QINDEX_RANGE: i32 = 256;
/// `FLAG_SKIP_INTRA_LOWVAR` (`speed_features.h:193`) — bit **5** of
/// `mode_search_skip_flags`, not bit 0. The enum is sparse: bit 0 is
/// `FLAG_EARLY_TERMINATE`, bit 2 is unused, and this is `1 << 5`. Measured
/// against the C, which disagreed when this was 1.
pub const FLAG_SKIP_INTRA_LOWVAR: i32 = 1 << 5;
/// The source variance below which non-DC intra is dropped (rdopt.c:4780).
const SKIP_INTRA_VAR_THRESH: u32 = 64;

/// `prune_ref_frame` (rdopt.c:4284): is this reference type pruned outright?
///
/// `selective` is the result of `prune_ref_by_selective_ref_frame`
/// (`rdopt.h:236`), which lives outside this file — see the module note.
pub fn prune_ref_frame(ref_type: i32, prune_ref_frame_mask: i32, selective: bool) -> bool {
    if (prune_ref_frame_mask >> ref_type) & 1 != 0 {
        return true;
    }
    selective
}

/// What [`inter_mode_search_order_independent_skip`] decided.
///
/// C returns a bare `int` whose three values mean quite different things —
/// its own comment spells them out as "Case 1/2/3" — so they are an enum here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeSkipVerdict {
    /// C's 0: evaluate this candidate.
    Search,
    /// C's 1: skip the mode entirely.
    Skip,
    /// C's 2: skip the motion-mode search but still try SIMPLE_TRANSLATION.
    SkipMotionModeOnly,
}

impl ModeSkipVerdict {
    /// C's integer.
    pub const fn to_i32(self) -> i32 {
        match self {
            Self::Search => 0,
            Self::Skip => 1,
            Self::SkipMotionModeOnly => 2,
        }
    }
}

/// Everything outside the mask that
/// [`inter_mode_search_order_independent_skip`] reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModeSkipCtx {
    /// `cpi->prune_ref_frame_mask`.
    pub prune_ref_frame_mask: i32,
    /// The result of `prune_ref_by_selective_ref_frame` — see the module note.
    pub selective_prune: bool,
    /// `sf->rt_sf.use_real_time_ref_set`, which SKIPS the prune entirely.
    pub use_real_time_ref_set: bool,
    /// The caller's `skip_ref_frame_mask`.
    pub skip_ref_frame_mask: i32,
    /// `x->mb_mode_cache` — the cached mode info POINTER, independent of the
    /// flag below. C reads the FLAG for the replay block and the POINTER for
    /// `is_ref_frame_used_in_cache`; a stale non-null cache with the flag off
    /// still affects the second. Measured: collapsing the two into one option
    /// made the port disagree with C on the `SkipMotionModeOnly` verdict.
    pub mb_mode_cache: Option<(PredMode, [i32; 2])>,
    /// `x->use_mb_mode_cache`.
    pub use_mb_mode_cache: bool,
    /// `search_state->best_rd == INT64_MAX` — nothing has won yet.
    pub best_rd_is_max: bool,
    /// `mbmi->partition`.
    pub partition: i32,
    /// `x->must_find_valid_partition`.
    pub must_find_valid_partition: bool,
    /// `sf->inter_sf.prune_nearmv_using_neighbors` (0 = off, 1..3 = level).
    pub prune_nearmv_using_neighbors: i32,
    /// `x->qindex`.
    pub qindex: i32,
    /// `xd->left_available` / `up_available` and the neighbours' references.
    pub left: Option<[i32; 2]>,
    /// See [`Self::left`].
    pub above: Option<[i32; 2]>,
    /// `sf->rt_sf.mode_search_skip_flags`.
    pub mode_search_skip_flags: i32,
    /// `x->source_variance`.
    pub source_variance: u32,
}

/// `is_mode_intra` (`blockd.h`).
fn is_mode_intra(mode: PredMode) -> bool {
    mode.to_i32() < PredMode::NearestMv.to_i32()
}

/// `inter_mode_search_order_independent_skip` (rdopt.c:4643): the first gate
/// every `(mode, reference pair)` candidate passes through.
///
/// `skip_repeated_mv_args` carries what the nested
/// [`skip_repeated_mv`] call needs — including its in/out `modelled_rd`
/// column, which that call WRITES on the skipping path.
#[allow(clippy::too_many_arguments)]
pub fn inter_mode_search_order_independent_skip(
    mask: &ModeSkipMask,
    mode: PredMode,
    ref_frame: [i32; 2],
    ref_type: i32,
    ctx: &ModeSkipCtx,
    ref_mv_count: usize,
    gm_wmtype_is_translational: bool,
    mode_context: i32,
    costs: &crate::inter_costs::InterModeCosts,
    modelled_rd: &mut [i64; 25],
) -> ModeSkipVerdict {
    if mask.says_skip(ref_frame, mode) {
        return ModeSkipVerdict::Skip;
    }
    if !ctx.use_real_time_ref_set
        && prune_ref_frame(ref_type, ctx.prune_ref_frame_mask, ctx.selective_prune)
    {
        return ModeSkipVerdict::Skip;
    }
    // C's `motion_vector_unit_test` arm is a debug hook with no production
    // path and is not modelled.
    if skip_repeated_mv(
        mode,
        ref_frame[1] > 0,
        ref_mv_count,
        gm_wmtype_is_translational,
        mode_context,
        costs,
        modelled_rd,
    ) {
        return ModeSkipVerdict::Skip;
    }

    // Replaying a cached decision: anything that cannot match the cache is
    // dropped, EXCEPT a single-reference candidate a cached COMPOUND mode
    // depends on for its NEWMV start point — that one keeps its simple
    // translation and loses only the motion-mode search.
    if let Some((cached_mode, cached_frame)) = ctx.mb_mode_cache.filter(|_| ctx.use_mb_mode_cache) {
        let cached_is_single = cached_frame[1] <= 0;
        if is_mode_intra(cached_mode) && mode != cached_mode {
            return ModeSkipVerdict::Skip;
        }
        // NOTE the `cached_mode_is_single` chain below is NOT guarded on the
        // cache being inter: an INTRA cache reaches it too (its ref_frame[1]
        // is NONE, so it takes the single arm) and its `ref_frame[0] !=
        // cached_frame[0]` test then still applies. Adding the obvious
        // `!is_mode_intra` guard makes the port disagree with C on an intra
        // cache whose mode matches but whose reference does not.
        {
            if cached_is_single {
                if mode != cached_mode || ref_frame[0] != cached_frame[0] {
                    return ModeSkipVerdict::Skip;
                }
            } else if ref_frame[1] <= 0 {
                let depends = match cached_mode {
                    PredMode::NewNearMv | PredMode::NewNearestMv => ref_frame[0] == cached_frame[0],
                    PredMode::NearNewMv | PredMode::NearestNewMv => ref_frame[0] == cached_frame[1],
                    PredMode::NewNewMv => {
                        ref_frame[0] == cached_frame[0] || ref_frame[0] == cached_frame[1]
                    }
                    _ => false,
                };
                return if depends {
                    ModeSkipVerdict::SkipMotionModeOnly
                } else {
                    ModeSkipVerdict::Skip
                };
            } else if mode != cached_mode
                || ref_frame[0] != cached_frame[0]
                || ref_frame[1] != cached_frame[1]
            {
                return ModeSkipVerdict::Skip;
            }
        }
    }

    // Nothing has won yet and a valid partition is required: keep everything.
    if ctx.best_rd_is_max && ctx.partition == PARTITION_NONE && ctx.must_find_valid_partition {
        return ModeSkipVerdict::Search;
    }

    if ctx.prune_nearmv_using_neighbors > 0
        && matches!(mode, PredMode::NearNearMv | PredMode::NearMv)
        && !ctx.best_rd_is_max
        && let (Some(left), Some(above)) = (ctx.left, ctx.above)
    {
        {
            // `thresholds[level - 1][qindex_third]` (rdopt.c:4722).
            const THRESHOLDS: [[i32; 3]; 3] = [[1, 0, 0], [1, 1, 0], [2, 1, 0]];
            let qindex_sub_range = (ctx.qindex * 3 / QINDEX_RANGE) as usize;
            let thresh =
                THRESHOLDS[(ctx.prune_nearmv_using_neighbors - 1) as usize][qindex_sub_range];
            let matches = i32::from(match_ref_frame_pair(left, ref_frame))
                + i32::from(match_ref_frame_pair(above, ref_frame));
            if matches < thresh {
                return ModeSkipVerdict::Skip;
            }
        }
    }

    let mut skip_motion_mode = false;
    if ctx.partition != PARTITION_NONE {
        let mut skip_ref = ctx.skip_ref_frame_mask & (1 << ref_type) != 0;
        if ref_type <= ALTREF_FRAME && skip_ref {
            // A compound pair that survived may still need this single
            // reference's motion search as its start point.
            if is_ref_frame_used_by_compound_ref(ref_type, ctx.skip_ref_frame_mask) {
                skip_motion_mode = true;
                skip_ref = false;
            }
        }
        if is_ref_frame_used_in_cache(ref_type, ctx.mb_mode_cache.map(|(_, rf)| rf)) {
            skip_ref = false;
            skip_motion_mode =
                ref_type <= ALTREF_FRAME && ctx.mb_mode_cache.is_some_and(|(_, rf)| rf[1] > 0);
        }
        if skip_ref {
            return ModeSkipVerdict::Skip;
        }
    }

    if ref_frame[0] == 0
        && mode != PredMode::DcPred
        && ctx.mode_search_skip_flags & FLAG_SKIP_INTRA_LOWVAR != 0
        && ctx.source_variance < SKIP_INTRA_VAR_THRESH
    {
        return ModeSkipVerdict::Skip;
    }

    if skip_motion_mode {
        ModeSkipVerdict::SkipMotionModeOnly
    } else {
        ModeSkipVerdict::Search
    }
}

/// `REFERENCE_MODES` (`enums.h`): SINGLE, COMPOUND, SELECT.
pub const REFERENCE_MODES: usize = 3;
/// `SINGLE_REFERENCE`.
pub const SINGLE_REFERENCE: usize = 0;
/// `COMPOUND_REFERENCE`.
pub const COMPOUND_REFERENCE: usize = 1;
/// `REFERENCE_MODE_SELECT`.
pub const REFERENCE_MODE_SELECT: usize = 2;

/// `record_best_compound` (rdopt.c:5440): track the best RD achievable under
/// each of the three frame-level reference modes.
///
/// The point is that a candidate's rate DEPENDS on which reference mode the
/// frame header will signal: under `REFERENCE_MODE_SELECT` the per-block
/// `comp_mode` bit is already in `rd_stats->rate`, and under the other two it
/// is not. So the same candidate is scored twice, once each way.
pub fn record_best_compound(
    reference_mode: usize,
    rate: i32,
    dist: i64,
    comp_pred: bool,
    rdmult: i32,
    compmode_cost: i32,
    best_pred_rd: &mut [i64; REFERENCE_MODES],
) {
    let (single_rate, hybrid_rate) = if reference_mode == REFERENCE_MODE_SELECT {
        (rate - compmode_cost, rate)
    } else {
        (rate, rate + compmode_cost)
    };
    let single_rd = crate::rd::rdcost(rdmult, single_rate, dist);
    let hybrid_rd = crate::rd::rdcost(rdmult, hybrid_rate, dist);

    let slot = if comp_pred {
        COMPOUND_REFERENCE
    } else {
        SINGLE_REFERENCE
    };
    best_pred_rd[slot] = best_pred_rd[slot].min(single_rd);
    best_pred_rd[REFERENCE_MODE_SELECT] = best_pred_rd[REFERENCE_MODE_SELECT].min(hybrid_rd);
}

/// The ten `MB_MODE_INFO` fields `init_mbmi` (rdopt.c:4795) resets, in C's
/// write order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InitMbmi {
    /// `ref_mv_idx`.
    pub ref_mv_idx: i32,
    /// `mode`.
    pub mode: i32,
    /// `uv_mode` (`UV_DC_PRED`).
    pub uv_mode: i32,
    /// `ref_frame[0]` / `[1]`.
    pub ref_frame: [i32; 2],
    /// `palette_mode_info.palette_size[0]` / `[1]`.
    pub palette_size: [i32; 2],
    /// `filter_intra_mode_info.use_filter_intra`.
    pub use_filter_intra: i32,
    /// `motion_mode` (`SIMPLE_TRANSLATION`).
    pub motion_mode: i32,
    /// `interintra_mode`.
    ///
    /// C writes `(INTERINTRA_MODE)(II_DC_PRED - 1)`, i.e. **-1**, which is not
    /// a valid `INTERINTRA_MODE` — it is a deliberate "unset" marker stored in
    /// an unsigned 1-byte enum, so it reads back as 255. Reproduced rather
    /// than normalised, because `av1_is_interintra_wedge_used` and the
    /// interintra search both test it against real modes.
    pub interintra_mode: i32,
}

/// `init_mbmi` (rdopt.c:4795): reset the per-candidate mode info.
///
/// `set_default_interp_filters` is not modelled here — it writes
/// `mbmi->interp_filters`, which the port carries in its own candidate type.
pub fn init_mbmi(curr_mode: PredMode, ref_frames: [i32; 2]) -> InitMbmi {
    InitMbmi {
        ref_mv_idx: 0,
        mode: curr_mode.to_i32(),
        // `UV_DC_PRED` is 0.
        uv_mode: 0,
        ref_frame: ref_frames,
        palette_size: [0, 0],
        use_filter_intra: 0,
        // `SIMPLE_TRANSLATION`.
        motion_mode: 0,
        // `II_DC_PRED - 1` in a `uint8_t` enum.
        interintra_mode: 255,
    }
}

/// Which winner-mode record `get_winner_mode_stats` (rdopt.c:3835) selects.
///
/// C returns four out-parameters plus a pointer into either
/// `x->winner_mode_stats[mode_idx]` or the caller's own "best" variables. The
/// choice is the whole content of the function, so the port returns the
/// CHOICE and lets the caller do the borrow — which is also the only way to
/// express it without aliasing two mutable references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinnerModeSource {
    /// `multi_winner_mode_type == 0`: the caller's own best-mode variables.
    Best,
    /// Otherwise: `x->winner_mode_stats[mode_idx]`.
    WinnerStats(usize),
}

/// `get_winner_mode_stats` (rdopt.c:3835).
///
/// C asserts `0 <= mode_idx < x->winner_mode_count` on the multi-winner path;
/// the port returns `None` instead of indexing out of range.
pub fn get_winner_mode_stats(
    multi_winner_mode_type: i32,
    mode_idx: i32,
    winner_mode_count: i32,
) -> Option<WinnerModeSource> {
    if multi_winner_mode_type == 0 {
        return Some(WinnerModeSource::Best);
    }
    if mode_idx < 0 || mode_idx >= winner_mode_count {
        return None;
    }
    Some(WinnerModeSource::WinnerStats(mode_idx as usize))
}
