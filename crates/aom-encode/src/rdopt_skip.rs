//! The mode / reference SKIP MASK layer of libaom's inter RD brain
//! (`av1/encoder/rdopt.c`) — which `(mode, reference-pair)` combinations the
//! search is allowed to evaluate at all, plus the neighbour-reference matching
//! and RD-ordering helpers that feed it.
//!
//! Companion to [`crate::rdopt_mv`]; the oracle and the evidence tier are the
//! same (`crates/aom-sys-ref/shim/rdopt_shim.c` compiles libaom's own rdopt.c
//! into the shim archive, so the bodies under test are libaom's source). Gate:
//! `crates/aom-encode/tests/rdopt_skip_diff.rs`.
//!
//! | Rust | C (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | [`ModeSkipMask::disable_reference`] | `disable_reference` `:4018` |
//! | [`ModeSkipMask::disable_inter_references_except_altref`] | `:4026` |
//! | [`ModeSkipMask::default_for`] | `default_skip_mask` `:4050` |
//! | [`ModeSkipMask::says_skip`] | `mask_says_skip` `:4571` |
//! | [`match_ref_frame_pair`] | `:4634` |
//! | [`ref_match_found_in_nb_blocks`] | `:2465` |
//! | [`find_ref_match_in_nbs`] | `find_ref_match_in_above_nbs` `:2482` + `_left_` `:2504` |
//! | [`match_ref_frame`] | `:5048` |
//! | [`compound_skip_using_neighbor_refs`] | `:5062` |
//! | [`skip_compound_using_best_single_mode_ref`] | `:5102` |
//! | [`update_best_single_mode`] | `:5091` |
//! | [`is_ref_frame_used_by_compound_ref`] | `:4300` |
//! | [`is_ref_frame_used_in_cache`] | `:4313` |
//! | [`fetch_picked_ref_frames_mask`] | `:4613` |
//! | [`find_top_ref`] | `:5180` (+ `compare_int64` `:5134`) |
//! | [`in_single_ref_cutoff`] | `:5197` |
//! | [`inter_modes_info_sort`] | `:502` (+ `compare_rd_idx_pair` `:485`) |
//!
//! # Translation notes
//!
//! - **`ref_combo` keeps C's `+ 1` index shift as a method, not as arithmetic
//!   at every call site.** C writes `ref_combo[i][j + 1]` because the second
//!   reference can be `NONE_FRAME == -1`; [`ModeSkipMask::is_disabled`] takes
//!   the reference pair and does the shift once.
//! - **The two neighbour walks are one function.** `find_ref_match_in_above_nbs`
//!   and `find_ref_match_in_left_nbs` are the same walk over transposed axes
//!   (row stepping by `mi_size_high` versus column stepping by `mi_size_wide`),
//!   and C's copies differ only in that. [`find_ref_match_in_nbs`] takes the
//!   direction; the shared body cannot drift between the two the way the C
//!   copies can.
//! - **`find_top_ref` does not need a sort.** C `qsort`s a 7-element copy and
//!   then reads only element 0, i.e. it computes a minimum. The port takes the
//!   minimum directly. That is observationally identical (`compare_int64` is a
//!   total order on `i64`) and is checked against the C for every input the
//!   harness generates, ties included.
//! - **`inter_modes_info_sort` DOES need the tie-break.** C's comparator falls
//!   back to `idx` when the RDs are equal, deliberately (aomedia:2928), so the
//!   order is fully determined; the port uses `sort_by_key((rd, idx))`, which
//!   is the same total order and is stable regardless.

use crate::rdopt_mv::{
    ALTREF_FRAME, BWDREF_FRAME, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, LAST2_FRAME, LAST3_FRAME,
    MODE_CTX_REF_FRAMES, PredMode, REF_FRAMES, ref_frame_type,
};
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B};

/// `NONE_FRAME` (`av1/common/enums.h`) — "no second reference".
pub const NONE_FRAME: i32 = -1;

/// `REF_SET` (rdopt.c:4047): which reference combinations `default_skip_mask`
/// leaves enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefSet {
    /// Everything available.
    Full,
    /// `enable_reduced_reference_set`: 16 explicitly-enabled combinations.
    Reduced,
    /// `rt_sf.use_real_time_ref_set`: 4 combinations.
    RealTime,
}

/// `reduced_ref_combos` (rdopt.c:4037).
const REDUCED_REF_COMBOS: [[i32; 2]; 16] = [
    [LAST_FRAME, NONE_FRAME],
    [ALTREF_FRAME, NONE_FRAME],
    [LAST_FRAME, ALTREF_FRAME],
    [GOLDEN_FRAME, NONE_FRAME],
    [INTRA_FRAME, NONE_FRAME],
    [GOLDEN_FRAME, ALTREF_FRAME],
    [LAST_FRAME, GOLDEN_FRAME],
    [LAST_FRAME, INTRA_FRAME],
    [LAST_FRAME, BWDREF_FRAME],
    [LAST_FRAME, LAST3_FRAME],
    [GOLDEN_FRAME, BWDREF_FRAME],
    [GOLDEN_FRAME, INTRA_FRAME],
    [BWDREF_FRAME, NONE_FRAME],
    [BWDREF_FRAME, ALTREF_FRAME],
    [ALTREF_FRAME, INTRA_FRAME],
    [BWDREF_FRAME, INTRA_FRAME],
];

/// `real_time_ref_combos` (`av1/encoder/rd.h:71`).
const REAL_TIME_REF_COMBOS: [[i32; 2]; 4] = [
    [LAST_FRAME, NONE_FRAME],
    [ALTREF_FRAME, NONE_FRAME],
    [GOLDEN_FRAME, NONE_FRAME],
    [INTRA_FRAME, NONE_FRAME],
];

/// `mode_skip_mask_t` (rdopt.c:4006): which prediction modes and which
/// reference combinations the inter search must NOT try.
///
/// Both members are "true means skip".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeSkipMask {
    /// `pred_modes[ref]`: a bit per [`PredMode`], set to FORBID that mode for
    /// that first reference.
    pub pred_modes: [u32; REF_FRAMES],
    /// `ref_combo[ref1][ref2 + 1]`: true forbids the `(ref1, ref2)` pair. Use
    /// [`Self::is_disabled`] / [`Self::disable_pair`] rather than indexing, so
    /// the `+ 1` shift for `NONE_FRAME` happens in one place.
    pub ref_combo: [[bool; REF_FRAMES + 1]; REF_FRAMES],
}

impl Default for ModeSkipMask {
    /// C's `REF_SET_FULL` arm: `memset(mask, 0, sizeof(*mask))` — nothing
    /// skipped.
    fn default() -> Self {
        Self {
            pred_modes: [0; REF_FRAMES],
            ref_combo: [[false; REF_FRAMES + 1]; REF_FRAMES],
        }
    }
}

impl ModeSkipMask {
    /// Is the `(ref1, ref2)` combination forbidden? `ref2` may be
    /// [`NONE_FRAME`], which is what the stored `+ 1` shift exists for.
    pub fn is_disabled(&self, ref1: i32, ref2: i32) -> bool {
        self.ref_combo[ref1 as usize][(ref2 + 1) as usize]
    }

    /// Forbid the `(ref1, ref2)` combination.
    pub fn disable_pair(&mut self, ref1: i32, ref2: i32) {
        self.ref_combo[ref1 as usize][(ref2 + 1) as usize] = true;
    }

    /// Allow the `(ref1, ref2)` combination.
    pub fn enable_pair(&mut self, ref1: i32, ref2: i32) {
        self.ref_combo[ref1 as usize][(ref2 + 1) as usize] = false;
    }

    /// `disable_reference` (rdopt.c:4018): forbid `r` as a FIRST reference,
    /// against every possible second reference including `NONE_FRAME`.
    ///
    /// Note this only clears the row `r`; it does NOT forbid `r` as a second
    /// reference of some other pair. That asymmetry is C's and is load-bearing
    /// — `is_ref_frame_used_by_compound_ref` exists because of it.
    pub fn disable_reference(&mut self, r: i32) {
        self.ref_combo[r as usize].fill(true);
    }

    /// `disable_inter_references_except_altref` (rdopt.c:4026).
    pub fn disable_inter_references_except_altref(&mut self) {
        for r in [
            LAST_FRAME,
            LAST2_FRAME,
            LAST3_FRAME,
            GOLDEN_FRAME,
            BWDREF_FRAME,
            ALTREF2_FRAME,
        ] {
            self.disable_reference(r);
        }
    }

    /// `default_skip_mask` (rdopt.c:4050): the mask before any speed feature
    /// or availability has been applied.
    pub fn default_for(ref_set: RefSet) -> Self {
        let mut mask = Self::default();
        let combos: &[[i32; 2]] = match ref_set {
            RefSet::Full => return mask,
            RefSet::Reduced => &REDUCED_REF_COMBOS,
            RefSet::RealTime => &REAL_TIME_REF_COMBOS,
        };
        // All references disabled first, then the listed set re-enabled.
        for row in &mut mask.ref_combo {
            row.fill(true);
        }
        for c in combos {
            mask.enable_pair(c[0], c[1]);
        }
        mask
    }

    /// `mask_says_skip` (rdopt.c:4571): the mode bit for the first reference,
    /// or the pair being forbidden outright.
    pub fn says_skip(&self, rf: [i32; 2], this_mode: PredMode) -> bool {
        if self.pred_modes[rf[0] as usize] & (1 << this_mode.to_i32()) != 0 {
            return true;
        }
        self.is_disabled(rf[0], rf[1])
    }
}

/// `ALTREF2_FRAME`, re-exported for [`ModeSkipMask::disable_inter_references_except_altref`].
use crate::rdopt_mv::ALTREF2_FRAME;

/// `match_ref_frame_pair` (rdopt.c:4634): does a neighbour use exactly this
/// reference pair?
pub fn match_ref_frame_pair(mbmi_rf: [i32; 2], rf: [i32; 2]) -> bool {
    mbmi_rf == rf
}

/// `ref_match_found_in_nb_blocks` (rdopt.c:2465): does the neighbour use ANY
/// of the current block's references, in either of its own slots?
pub fn ref_match_found_in_nb_blocks(cur_rf: [i32; 2], nb_rf: [i32; 2]) -> bool {
    let is_cur_comp = cur_rf[1] > INTRA_FRAME;
    cur_rf
        .iter()
        .take(usize::from(is_cur_comp) + 1)
        .any(|&r| r == nb_rf[0] || r == nb_rf[1])
}

/// The direction [`find_ref_match_in_nbs`] walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NbDir {
    /// The mi row above the block (`find_ref_match_in_above_nbs`).
    Above,
    /// The mi column left of the block (`find_ref_match_in_left_nbs`).
    Left,
}

/// One neighbour mi cell, as the walk reads it.
#[derive(Clone, Copy, Debug, Default)]
pub struct NbMi {
    /// `mbmi->ref_frame`.
    pub ref_frame: [i32; 2],
    /// `mbmi->bsize` — the walk's STEP comes from this, so a wrong value skips
    /// or re-reads cells rather than just misreading one.
    pub bsize: usize,
    /// `mbmi->use_intrabc`, the other half of `is_inter_block`.
    pub use_intrabc: bool,
}

/// `find_ref_match_in_above_nbs` (rdopt.c:2482) / `find_ref_match_in_left_nbs`
/// (`:2504`), which are the same walk over transposed axes.
///
/// **An UNAVAILABLE edge returns `true`.** C returns 1 when
/// `!xd->up_available`, i.e. "assume a match" — the caller
/// (`prune_ref_frame_by_neighbours`) treats a missing neighbour as no evidence
/// to prune on. That inversion is easy to read as a bug and is not one.
///
/// `nb` is indexed by mi position along the walked axis; `available` is
/// `xd->up_available` / `xd->left_available`; `extent` is `xd->width` /
/// `xd->height` in mi units; `total_mi` is the frame's mi count along that
/// axis, which bounds the walk at the right/bottom frame edge.
pub fn find_ref_match_in_nbs(
    dir: NbDir,
    total_mi: i32,
    start: i32,
    extent: i32,
    available: bool,
    cur_rf: [i32; 2],
    nb: impl Fn(i32) -> NbMi,
) -> bool {
    if !available {
        return true;
    }
    let end = (start + extent).min(total_mi);
    let mut pos = start;
    while pos < end {
        let cell = nb(pos);
        let step = match dir {
            NbDir::Above => MI_SIZE_WIDE_B[cell.bsize],
            NbDir::Left => MI_SIZE_HIGH_B[cell.bsize],
        } as i32;
        let is_inter = cell.use_intrabc || cell.ref_frame[0] > INTRA_FRAME;
        if is_inter && ref_match_found_in_nb_blocks(cur_rf, cell.ref_frame) {
            return true;
        }
        debug_assert!(step > 0, "a zero mi step would not terminate");
        pos += step;
    }
    false
}

/// `match_ref_frame` (rdopt.c:5048): accumulate, per direction, whether a
/// neighbour uses this block's forward / backward reference.
///
/// `is_ref_match` is in/out because C's is: the caller ORs the left and the
/// above neighbour into the same pair of flags.
pub fn match_ref_frame(nb: NbMi, rf: [i32; 2], is_ref_match: &mut [bool; 2]) {
    let is_inter = nb.use_intrabc || nb.ref_frame[0] > INTRA_FRAME;
    if !is_inter {
        return;
    }
    let has_second = nb.ref_frame[1] > INTRA_FRAME;
    for (slot, flag) in is_ref_match.iter_mut().enumerate() {
        *flag |= rf[slot] == nb.ref_frame[0];
        if has_second {
            *flag |= rf[slot] == nb.ref_frame[1];
        }
    }
}

/// `compound_skip_using_neighbor_refs` (rdopt.c:5062): prune an EXTENDED
/// compound mode when too few of the two neighbours share its references.
///
/// The four non-extended compound modes (`NEAREST_NEARESTMV`, `NEAR_NEARMV`,
/// `NEW_NEWMV`, `GLOBAL_GLOBALMV`) are never pruned here.
pub fn compound_skip_using_neighbor_refs(
    this_mode: PredMode,
    rf: [i32; 2],
    prune_ext_comp_using_neighbors: i32,
    left: Option<NbMi>,
    above: Option<NbMi>,
) -> bool {
    use PredMode::*;
    if matches!(
        this_mode,
        NearestNearestMv | NearNearMv | NewNewMv | GlobalGlobalMv
    ) {
        return false;
    }
    if prune_ext_comp_using_neighbors >= 3 {
        return true;
    }
    let mut is_ref_match = [false; 2];
    if let Some(nb) = left {
        match_ref_frame(nb, rf, &mut is_ref_match);
    }
    if let Some(nb) = above {
        match_ref_frame(nb, rf, &mut is_ref_match);
    }
    let track_ref_match = i32::from(is_ref_match[0]) + i32::from(is_ref_match[1]);
    track_ref_match < prune_ext_comp_using_neighbors
}

/// `update_best_single_mode` (rdopt.c:5091): remember the cheapest single-ref
/// mode seen for a reference.
pub fn update_best_single_mode(
    best_single_rd: &mut [i64; REF_FRAMES],
    best_single_mode: &mut [Option<PredMode>; REF_FRAMES],
    this_mode: PredMode,
    ref_frame: i32,
    this_rd: i64,
) {
    let r = ref_frame as usize;
    if this_rd < best_single_rd[r] {
        best_single_rd[r] = this_rd;
        best_single_mode[r] = Some(this_mode);
    }
}

/// `skip_compound_using_best_single_mode_ref` (rdopt.c:5102): prune an
/// extended compound mode unless the reference that carries its NEWMV half
/// also won as NEWMV on its own.
///
/// `best_single_mode` is `None` where C holds `MB_MODE_COUNT` ("no best single
/// mode yet"), which `prune_comp_using_best_single_mode_ref == 1` treats as a
/// reason NOT to prune.
pub fn skip_compound_using_best_single_mode_ref(
    this_mode: PredMode,
    rf: [i32; 2],
    best_single_mode: &[Option<PredMode>; REF_FRAMES],
    prune_comp_using_best_single_mode_ref: i32,
) -> bool {
    use PredMode::*;
    if matches!(
        this_mode,
        NearestNearestMv | NearNearMv | NewNewMv | GlobalGlobalMv
    ) {
        return false;
    }
    debug_assert!(this_mode >= NearestNewMv && this_mode <= NewNearMv);
    // The direction whose half of the compound mode is NEWMV: 0 when the
    // FIRST reference is the NEWMV one, else 1.
    let newmv_dir = usize::from(crate::rdopt_mv::compound_ref0_mode(this_mode) != Some(NewMv));
    let single_mode = best_single_mode[rf[newmv_dir] as usize];
    if single_mode == Some(NewMv) {
        return false;
    }
    if prune_comp_using_best_single_mode_ref == 1 && single_mode.is_none() {
        return false;
    }
    true
}

/// `is_ref_frame_used_by_compound_ref` (rdopt.c:4300): is `ref_frame` a member
/// of any compound pair that has NOT been skipped?
pub fn is_ref_frame_used_by_compound_ref(ref_frame: i32, skip_ref_frame_mask: i32) -> bool {
    // `ref_frame_map` (mvref_common.h:129), the REF_FRAMES.. rows of the
    // ref-frame-type space, in C's order.
    for r in (ALTREF_FRAME + 1) as usize..MODE_CTX_REF_FRAMES {
        if skip_ref_frame_mask & (1 << r) != 0 {
            continue;
        }
        let pair = REF_FRAME_MAP[r - REF_FRAMES];
        if pair[0] == ref_frame || pair[1] == ref_frame {
            return true;
        }
    }
    false
}

/// `ref_frame_map[TOTAL_COMP_REFS][2]` (`mvref_common.h:129`): the reference
/// pair each compound `ref_frame_type` row denotes. The first 12 are the
/// bidirectional pairs in `BWD_RF_OFFSET`-major order, then the 9
/// unidirectional ones — the same order [`ref_frame_type`] computes.
pub const REF_FRAME_MAP: [[i32; 2]; 21] = [
    [LAST_FRAME, BWDREF_FRAME],
    [LAST2_FRAME, BWDREF_FRAME],
    [LAST3_FRAME, BWDREF_FRAME],
    [GOLDEN_FRAME, BWDREF_FRAME],
    [LAST_FRAME, ALTREF2_FRAME],
    [LAST2_FRAME, ALTREF2_FRAME],
    [LAST3_FRAME, ALTREF2_FRAME],
    [GOLDEN_FRAME, ALTREF2_FRAME],
    [LAST_FRAME, ALTREF_FRAME],
    [LAST2_FRAME, ALTREF_FRAME],
    [LAST3_FRAME, ALTREF_FRAME],
    [GOLDEN_FRAME, ALTREF_FRAME],
    [LAST_FRAME, LAST2_FRAME],
    [LAST_FRAME, LAST3_FRAME],
    [LAST_FRAME, GOLDEN_FRAME],
    [BWDREF_FRAME, ALTREF_FRAME],
    [LAST2_FRAME, LAST3_FRAME],
    [LAST2_FRAME, GOLDEN_FRAME],
    [LAST3_FRAME, GOLDEN_FRAME],
    [BWDREF_FRAME, ALTREF2_FRAME],
    [ALTREF2_FRAME, ALTREF_FRAME],
];

/// `is_ref_frame_used_in_cache` (rdopt.c:4313): does the cached mode info use
/// this reference (or, for a compound `ref_frame` index, exactly this pair)?
pub fn is_ref_frame_used_in_cache(ref_frame: i32, mi_cache: Option<[i32; 2]>) -> bool {
    let Some(cache) = mi_cache else {
        return false;
    };
    if (ref_frame as usize) < REF_FRAMES {
        return ref_frame == cache[0] || ref_frame == cache[1];
    }
    ref_frame == ref_frame_type(cache) as i32
}

/// `fetch_picked_ref_frames_mask` (rdopt.c:4613): the OR of the per-mi picked
/// reference masks over this block's footprint inside its superblock.
///
/// `picked` is C's `x->picked_ref_frames_mask`, a fixed `32 x 32` per-mi array
/// indexed `i * 32 + j` REGARDLESS of the superblock size, which is why the
/// stride is a constant here rather than derived from `mib_size`.
pub fn fetch_picked_ref_frames_mask(
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mib_size: i32,
    picked: &[i32; 32 * 32],
) -> i32 {
    let sb_size_mask = mib_size - 1;
    let row_in_sb = mi_row & sb_size_mask;
    let col_in_sb = mi_col & sb_size_mask;
    let mi_w = MI_SIZE_WIDE_B[bsize] as i32;
    let mi_h = MI_SIZE_HIGH_B[bsize] as i32;
    let mut mask = 0;
    for i in row_in_sb..row_in_sb + mi_h {
        for j in col_in_sb..col_in_sb + mi_w {
            mask |= picked[(i * 32 + j) as usize];
        }
    }
    mask
}

/// `find_top_ref` (rdopt.c:5180): write 110% of the best single-reference RD
/// into slot 0, which the search then uses as a cut-off.
///
/// C sorts a copy of slots 1..8 and reads element 0 — that is a minimum, and
/// the port takes one directly. `INT64_MAX` (no reference measured) is passed
/// through unscaled, because scaling it would overflow.
pub fn find_top_ref(ref_frame_rd: &mut [i64; REF_FRAMES]) {
    debug_assert_eq!(
        ref_frame_rd[0],
        i64::MAX,
        "slot 0 is the output, not an input"
    );
    let cutoff = ref_frame_rd[1..].iter().copied().min().unwrap_or(i64::MAX);
    ref_frame_rd[0] = if cutoff == i64::MAX {
        cutoff
    } else {
        debug_assert!(cutoff < i64::MAX / 200);
        (110 * cutoff) / 100
    };
}

/// `in_single_ref_cutoff` (rdopt.c:5197): is either reference of a compound
/// pair within [`find_top_ref`]'s cut-off?
pub fn in_single_ref_cutoff(ref_frame_rd: &[i64; REF_FRAMES], frame1: i32, frame2: i32) -> bool {
    debug_assert!(frame2 > 0);
    ref_frame_rd[frame1 as usize] <= ref_frame_rd[0]
        || ref_frame_rd[frame2 as usize] <= ref_frame_rd[0]
}

/// `inter_modes_info_sort` (rdopt.c:502) + `compare_rd_idx_pair` (`:485`):
/// rank the recorded candidates by estimated RD, ties broken by insertion
/// index.
///
/// The tie-break is deliberate upstream (aomedia:2928 — `qsort` is not stable,
/// so equal RDs otherwise gave a platform-dependent order). Returns
/// `(idx, rd)` pairs.
pub fn inter_modes_info_sort(est_rd: &[i64]) -> Vec<(usize, i64)> {
    let mut pairs: Vec<(usize, i64)> = est_rd.iter().copied().enumerate().collect();
    pairs.sort_by_key(|&(idx, rd)| (rd, idx));
    pairs
}
