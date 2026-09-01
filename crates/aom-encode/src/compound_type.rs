//! Compound-type RD search — port of `av1/encoder/compound_type.c`.
//!
//! This is the decision layer that picks, for one inter block with two
//! references, *how* the two predictors are combined: a plain average, a
//! distance-weighted average, a wedge, or a difference-weighted segmentation
//! mask. The DSP that executes those blends already lives in
//! [`aom_dsp::inter::compound`]; this module is the search that chooses among
//! them, plus the interintra (inter + intra blend) mode search that shares the
//! same wedge machinery.
//!
//! # What is here, and what is not
//! Ported so far — the decision and cost layer, every function pure over its
//! inputs:
//!
//! | Rust | C (`compound_type.c`) |
//! |---|---|
//! | [`enable_wedge_search`] | `enable_wedge_search` |
//! | [`enable_wedge_interinter_search`] | `enable_wedge_interinter_search` |
//! | [`enable_wedge_interintra_search`] | `enable_wedge_interintra_search` |
//! | [`compute_valid_comp_types`] | `compute_valid_comp_types` |
//! | [`calc_masked_type_cost`] | `calc_masked_type_cost` |
//! | [`CompoundType::comp_group_idx`] / [`CompoundType::compound_idx`] | `update_mbmi_for_compound_type` |
//! | [`get_interinter_compound_mask_rate`] | `get_interinter_compound_mask_rate` |
//! | [`save_mask_search_results`] | `save_mask_search_results` |
//! | [`push_comp_avg_est_rd`] | `push_comp_avg_est_rd` |
//! | [`prune_comp_eval_using_comp_avg_est_rd`] | `prune_comp_eval_using_comp_avg_est_rd` |
//! | [`compute_rd_thresh`] | `compute_rd_thresh` |
//!
//! # Differential coverage
//! `tests/compound_type_diff.rs`, **tier 1c** — the oracle is libaom's own
//! `compound_type.c`, compiled verbatim into `shim/compound_type_shim.c` with
//! its two exported symbols renamed out of the way, so the bodies under test
//! are C's source rather than a second transcription of it. Every function
//! above is `static` in C (`nm -g` reports only `av1_compound_type_rd` and
//! `av1_handle_inter_intra_mode` for the whole file), so this is the strongest
//! evidence the file admits short of driving the whole inter RD brain. The
//! same technique, and the same justification, as `shim/rdopt_shim.c`.

use aom_dsp::inter::interintra::is_wedge_used;

/// `block_size_wide[BLOCK_SIZES_ALL]` (`common_data.h`).
const BLOCK_SIZE_WIDE: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
/// `block_size_high[BLOCK_SIZES_ALL]` (`common_data.h`).
const BLOCK_SIZE_HIGH: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

/// `AV1_PROB_COST_SHIFT` (`av1/encoder/cost.h:28`) — `av1_cost_literal(n)` is
/// `n << AV1_PROB_COST_SHIFT`.
const AV1_PROB_COST_SHIFT: i32 = 9;

/// `av1_cost_literal(n)` (`cost.h:29`): the cost of `n` equiprobable bits.
#[inline]
const fn cost_literal(n: i32) -> i32 {
    n * (1 << AV1_PROB_COST_SHIFT)
}

/// `TOP_COMP_AVG_EST_RD_COUNT` (`block.h:880`).
pub const TOP_COMP_AVG_EST_RD_COUNT: usize = 5;

/// `INTER_INTRA_RD_THRESH_SCALE` (`rdopt_utils.h:27`).
const INTER_INTRA_RD_THRESH_SCALE: i32 = 9;
/// `INTER_INTRA_RD_THRESH_SHIFT` (`rdopt_utils.h:28`).
const INTER_INTRA_RD_THRESH_SHIFT: i32 = 4;

/// `num_comp_mode_skip_cand[3]` (`compound_type.c:30`) — how many of the top
/// compound-average estimated RD costs the `skip_cmp_using_top_cmp_avg_est_rd`
/// speed feature keeps, indexed by `level - 1`.
const NUM_COMP_MODE_SKIP_CAND: [usize; 3] = [5, 4, 2];

/// `COMPOUND_TYPE` (`av1/common/enums.h:412-419`). The discriminants are the
/// bitstream's, and every `mode_search_mask` in the encoder is a bitmask over
/// them, so they are load-bearing rather than incidental.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum CompoundType {
    /// `COMPOUND_AVERAGE`
    Average = 0,
    /// `COMPOUND_DISTWTD`
    DistWtd = 1,
    /// `COMPOUND_WEDGE`
    Wedge = 2,
    /// `COMPOUND_DIFFWTD`
    DiffWtd = 3,
}

/// `COMPOUND_TYPES` (`enums.h:417`).
pub const COMPOUND_TYPES: usize = 4;

impl CompoundType {
    /// The four types in bitstream order — C's `for (comp_type =
    /// COMPOUND_AVERAGE; comp_type < COMPOUND_TYPES; comp_type++)`.
    pub const ALL: [CompoundType; COMPOUND_TYPES] = [
        CompoundType::Average,
        CompoundType::DistWtd,
        CompoundType::Wedge,
        CompoundType::DiffWtd,
    ];

    /// The C integer, for indexing the per-type arrays the search carries.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// `mbmi->comp_group_idx = (cur_type >= COMPOUND_WEDGE)`
    /// (`update_mbmi_for_compound_type`, compound_type.c:947).
    #[inline]
    pub const fn comp_group_idx(self) -> bool {
        (self as u8) >= (CompoundType::Wedge as u8)
    }

    /// `mbmi->compound_idx = (cur_type != COMPOUND_DISTWTD)`
    /// (`update_mbmi_for_compound_type`, compound_type.c:948).
    #[inline]
    pub const fn compound_idx(self) -> bool {
        !matches!(self, CompoundType::DistWtd)
    }

    /// Whether this type carries a mask (`is_masked_compound_type`,
    /// `reconinter.h:105`).
    #[inline]
    pub const fn is_masked(self) -> bool {
        matches!(self, CompoundType::Wedge | CompoundType::DiffWtd)
    }
}

/// `is_comp_ref_allowed` (`blockd.h:65`): compound prediction needs both sides
/// of the block to be at least 8 wide.
#[inline]
pub fn is_comp_ref_allowed(bsize: usize) -> bool {
    BLOCK_SIZE_WIDE[bsize].min(BLOCK_SIZE_HIGH[bsize]) >= 8
}

/// `is_interinter_compound_used` (`reconinter.h:299`): whether `ty` is a legal
/// compound type at `bsize`. Only `COMPOUND_WEDGE` additionally requires a
/// wedge codebook.
///
/// (Declared in `common/reconinter.h` rather than `compound_type.c`, but
/// nothing else in the port needed it yet and `compute_valid_comp_types` is
/// meaningless without it.)
#[inline]
pub fn is_interinter_compound_used(ty: CompoundType, bsize: usize) -> bool {
    let comp_allowed = is_comp_ref_allowed(bsize);
    match ty {
        CompoundType::Wedge => comp_allowed && is_wedge_used(bsize),
        _ => comp_allowed,
    }
}

// ===================================================================
// Wedge-search enable predicates (compound_type.c:103-121)
// ===================================================================

/// `enable_wedge_search` (compound_type.c:103): the source variance gate.
///
/// C compares an `unsigned int` against an `unsigned int` threshold; both
/// widths are kept, so a threshold of `UINT_MAX` disables the search for every
/// block rather than wrapping.
#[inline]
pub fn enable_wedge_search(source_variance: u32, disable_wedge_var_thresh: u32) -> bool {
    source_variance > disable_wedge_var_thresh
}

/// `enable_wedge_interinter_search` (compound_type.c:110).
///
/// `disable_wedge_var_thresh` is `cpi->sf.inter_sf.disable_interinter_wedge_var_thresh`
/// and `enable_interinter_wedge` is `cpi->oxcf.comp_type_cfg.enable_interinter_wedge`.
#[inline]
pub fn enable_wedge_interinter_search(
    source_variance: u32,
    disable_wedge_var_thresh: u32,
    enable_interinter_wedge: bool,
) -> bool {
    enable_wedge_search(source_variance, disable_wedge_var_thresh) && enable_interinter_wedge
}

/// `enable_wedge_interintra_search` (compound_type.c:116) — the same shape with
/// the interintra pair of knobs.
#[inline]
pub fn enable_wedge_interintra_search(
    source_variance: u32,
    disable_wedge_var_thresh: u32,
    enable_interintra_wedge: bool,
) -> bool {
    enable_wedge_search(source_variance, disable_wedge_var_thresh) && enable_interintra_wedge
}

// ===================================================================
// compute_valid_comp_types (compound_type.c:868)
// ===================================================================

/// `DIST_WTD_COMP_FLAG` (`speed_features.h:46-50`) — the speed feature's
/// three-valued setting for distance-weighted compound.
///
/// Modelled as an enum rather than the raw `int` because
/// `compute_valid_comp_types` tests it against `DIST_WTD_COMP_DISABLED`, which
/// is **2**, not 1 — a boolean `!= 1` transcription reads `SkipMvSearch` as
/// disabled and `Disabled` as enabled, i.e. it is wrong on two of the three
/// values. (Measured: that was the first failing cell of
/// `compute_valid_comp_types_matches_c`.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum DistWtdCompFlag {
    /// `DIST_WTD_COMP_ENABLED`
    Enabled = 0,
    /// `DIST_WTD_COMP_SKIP_MV_SEARCH`
    SkipMvSearch = 1,
    /// `DIST_WTD_COMP_DISABLED`
    Disabled = 2,
}

/// The knobs `compute_valid_comp_types` reads out of `cpi`. Grouped so the
/// caller cannot silently transpose two booleans at the call site.
#[derive(Clone, Copy, Debug)]
pub struct ValidCompTypeCfg {
    /// `cm->seq_params->order_hint_info.enable_dist_wtd_comp == 1`.
    ///
    /// C compares the seq-header field against the literal 1, so any other
    /// nonzero value reads as *disabled*; the caller performs that comparison
    /// and passes the boolean.
    pub enable_dist_wtd_comp: bool,
    /// `cpi->sf.inter_sf.use_dist_wtd_comp_flag`
    pub use_dist_wtd_comp_flag: DistWtdCompFlag,
    /// `enable_wedge_interinter_search(x, cpi)`
    pub enable_interinter_wedge: bool,
    /// `cpi->oxcf.comp_type_cfg.enable_diff_wtd_comp`
    pub enable_diff_wtd_comp: bool,
}

/// `compute_valid_comp_types` (compound_type.c:868): the compound types worth
/// evaluating for this block, in C's evaluation order.
///
/// C fills a caller-owned `COMPOUND_TYPE[4]` and returns the count; Rust
/// returns the prefix directly. `mode_search_mask` is a bitmask over
/// [`CompoundType`] discriminants.
pub fn compute_valid_comp_types(
    bsize: usize,
    masked_compound_used: bool,
    mode_search_mask: u32,
    cfg: &ValidCompTypeCfg,
) -> Vec<CompoundType> {
    let mut out = Vec::with_capacity(COMPOUND_TYPES);
    let try_average = mode_search_mask & (1 << CompoundType::Average.index()) != 0;
    let try_distwtd = (mode_search_mask & (1 << CompoundType::DistWtd.index()) != 0)
        && cfg.enable_dist_wtd_comp
        && cfg.use_dist_wtd_comp_flag != DistWtdCompFlag::Disabled;

    for (ty, valid_check) in [
        (CompoundType::Average, try_average),
        (CompoundType::DistWtd, try_distwtd),
    ] {
        if valid_check && is_interinter_compound_used(ty, bsize) {
            out.push(ty);
        }
    }

    if masked_compound_used {
        for (ty, enabled) in [
            (CompoundType::Wedge, cfg.enable_interinter_wedge),
            (CompoundType::DiffWtd, cfg.enable_diff_wtd_comp),
        ] {
            if (mode_search_mask & (1 << ty.index()) != 0)
                && is_interinter_compound_used(ty, bsize)
                && enabled
            {
                out.push(ty);
            }
        }
    }
    out
}

// ===================================================================
// calc_masked_type_cost (compound_type.c:906)
// ===================================================================

/// `calc_masked_type_cost` (compound_type.c:906): the signalling cost of each
/// compound type, indexed by [`CompoundType::index`].
///
/// The four cost slices are the rows of `x->mode_costs` the function reads:
/// `comp_group_idx_cost[comp_group_idx_ctx]`, `comp_idx_cost[comp_index_ctx]`
/// and `compound_type_cost[bsize]`. Passing the already-selected ROWS rather
/// than the whole tables plus two context indices keeps the two contexts from
/// being swapped here — they index different tables and are both small ints.
pub fn calc_masked_type_cost(
    comp_group_idx_cost: [i32; 2],
    comp_idx_cost: [i32; 2],
    compound_type_cost: [i32; 2],
    masked_compound_used: bool,
) -> [i32; COMPOUND_TYPES] {
    let mut cost = [0i32; COMPOUND_TYPES];
    if masked_compound_used {
        // Group index 0 covers average + distwtd, group index 1 wedge +
        // diffwtd. C accumulates DISTWTD from AVERAGE and DIFFWTD from WEDGE,
        // which at this point are exactly the two group-index costs.
        cost[CompoundType::Average.index()] += comp_group_idx_cost[0];
        cost[CompoundType::DistWtd.index()] += cost[CompoundType::Average.index()];
        cost[CompoundType::Wedge.index()] += comp_group_idx_cost[1];
        cost[CompoundType::DiffWtd.index()] += cost[CompoundType::Wedge.index()];
    }
    cost[CompoundType::Average.index()] += comp_idx_cost[1];
    cost[CompoundType::DistWtd.index()] += comp_idx_cost[0];
    cost[CompoundType::Wedge.index()] += compound_type_cost[0];
    cost[CompoundType::DiffWtd.index()] += compound_type_cost[1];
    cost
}

// ===================================================================
// get_interinter_compound_mask_rate (compound_type.c:1026)
// ===================================================================

/// `get_interinter_compound_mask_rate` (compound_type.c:1026): the extra rate
/// for signalling a masked compound type's mask.
///
/// Called only for the two masked types; C asserts that. `wedge_idx_cost` is
/// `mode_costs->wedge_idx_cost[bsize]`.
///
/// The `av1_is_wedge_used` guard is C's, and it can only be false when the
/// caller has already picked `COMPOUND_WEDGE` at a bsize with no codebook —
/// unreachable through `compute_valid_comp_types`, but reproduced rather than
/// asserted because the cost it returns (0) is observable.
pub fn get_interinter_compound_mask_rate(
    ty: CompoundType,
    bsize: usize,
    wedge_index: usize,
    wedge_idx_cost: &[i32],
) -> i32 {
    match ty {
        CompoundType::Wedge => {
            if is_wedge_used(bsize) {
                cost_literal(1) + wedge_idx_cost[wedge_index]
            } else {
                0
            }
        }
        CompoundType::DiffWtd => cost_literal(1),
        _ => unreachable!("get_interinter_compound_mask_rate is masked-types only"),
    }
}

// ===================================================================
// The small search-state predicates (compound_type.c:1058-1090)
// ===================================================================

/// `save_mask_search_results` (compound_type.c:1058): whether the mask index
/// just chosen should be cached in `args` for reuse by the next ref-MV
/// candidate. `this_mode` is a `PREDICTION_MODE`; `NEW_NEWMV` is the mode whose
/// results are always worth keeping.
#[inline]
pub fn save_mask_search_results(this_mode_is_new_newmv: bool, reuse_level: bool) -> bool {
    reuse_level || this_mode_is_new_newmv
}

/// `push_comp_avg_est_rd` (compound_type.c:737): insert one estimated compound
/// -average RD cost into the sorted top-N list, in place.
///
/// `level` is `cpi->sf.inter_sf.skip_cmp_using_top_cmp_avg_est_rd_lvl` (0
/// disables the feature entirely, 1..=3 select the candidate count). Only the
/// first `NUM_COMP_MODE_SKIP_CAND[level - 1]` entries participate; the tail of
/// the array is deliberately untouched, as in C.
pub fn push_comp_avg_est_rd(
    top_comp_avg_est_rd: &mut [i64; TOP_COMP_AVG_EST_RD_COUNT],
    tmp_rd: i64,
    level: usize,
) {
    if level == 0 {
        return;
    }
    assert!(level <= 3, "skip_cmp_using_top_cmp_avg_est_rd_lvl <= 3");
    let num_top_cand = NUM_COMP_MODE_SKIP_CAND[level - 1];
    debug_assert!(num_top_cand <= TOP_COMP_AVG_EST_RD_COUNT);

    // Insertion sort into the first `num_top_cand` slots: the entry that falls
    // off the end of that prefix is dropped, not shifted into the tail.
    if let Some(pos) = top_comp_avg_est_rd[..num_top_cand]
        .iter()
        .position(|&v| tmp_rd < v)
    {
        top_comp_avg_est_rd[pos..num_top_cand].rotate_right(1);
        top_comp_avg_est_rd[pos] = tmp_rd;
    }
}

/// `prune_comp_eval_using_comp_avg_est_rd` (compound_type.c:761): skip the
/// masked compound types when this candidate's estimated RD is already worse
/// than every one of the top-N compound-average costs.
pub fn prune_comp_eval_using_comp_avg_est_rd(
    top_comp_avg_est_rd: &[i64; TOP_COMP_AVG_EST_RD_COUNT],
    tmp_rd: i64,
    ref_best_rd: i64,
    level: usize,
) -> bool {
    if level == 0 {
        return false;
    }
    assert!(level <= 3, "skip_cmp_using_top_cmp_avg_est_rd_lvl <= 3");
    let num_top_cand = NUM_COMP_MODE_SKIP_CAND[level - 1];
    debug_assert!(num_top_cand <= TOP_COMP_AVG_EST_RD_COUNT);

    // No pruning until the list has filled: an unset slot is INT64_MAX.
    if top_comp_avg_est_rd[num_top_cand - 1] == i64::MAX || ref_best_rd == i64::MAX {
        return false;
    }
    tmp_rd > top_comp_avg_est_rd[num_top_cand - 1]
}

// ===================================================================
// compute_rd_thresh (compound_type.c:504)
// ===================================================================

/// `get_rd_thresh_from_best_rd` (`rdopt_utils.h:260`).
///
/// The `ref_best_rd < div * (INT64_MAX / mul)` guard is C's overflow check;
/// note it is evaluated in `int64_t`, so `INT64_MAX / mul_factor` truncates
/// before the multiply. Reproduced exactly — a "cleaner" saturating form
/// picks a different threshold near the boundary.
#[inline]
pub fn get_rd_thresh_from_best_rd(ref_best_rd: i64, mul_factor: i32, div_factor: i32) -> i64 {
    if div_factor == 0 {
        return ref_best_rd;
    }
    // C computes `div_factor * (INT64_MAX / mul_factor)` in `int64_t`, which
    // overflows (UB) for factor pairs the encoder never uses. `saturating_mul`
    // keeps the two live call sites — (16, 9) here and (mul, div) from
    // `comp_type_rd_threshold_*` — bit-identical while giving the unreachable
    // pairs a defined answer instead of a panic.
    if ref_best_rd < i64::from(div_factor).saturating_mul(i64::MAX / i64::from(mul_factor)) {
        (ref_best_rd / i64::from(div_factor)) * i64::from(mul_factor)
    } else {
        i64::MAX
    }
}

/// `compute_rd_thresh` (compound_type.c:504): the RD budget left for the smooth
/// interintra search once this mode's own rate is paid for.
///
/// The result can be negative — C returns `rd_thresh - mode_rd` with no clamp,
/// and `estimate_yrd_for_sb` treats a negative threshold as "give up", so the
/// sign is load-bearing.
#[inline]
pub fn compute_rd_thresh(rdmult: i32, total_mode_rate: i32, ref_best_rd: i64) -> i64 {
    let rd_thresh = get_rd_thresh_from_best_rd(
        ref_best_rd,
        1 << INTER_INTRA_RD_THRESH_SHIFT,
        INTER_INTRA_RD_THRESH_SCALE,
    );
    let mode_rd = crate::rd::rdcost(rdmult, total_mode_rate, 0);
    rd_thresh - mode_rd
}

// ===================================================================
// The mask picks — compound_type.c:126-428.
//
// This is the search proper: given the two single-reference predictors and
// the residuals derived from them, choose the wedge (index + sign) or the
// difference-weighted mask type that minimises the modelled RD cost.
// ===================================================================

use aom_dsp::dist::{
    highbd_subtract_block, highbd_variance, subtract_block, sum_squares_i16, variance,
};
use aom_dsp::inter::compound::{
    DiffwtdMaskType, build_compound_diffwtd_mask, build_compound_diffwtd_mask_highbd,
    wedge_compute_delta_squares, wedge_sign_from_residuals, wedge_sse_from_residuals,
};
use aom_dsp::inter::interintra::wedge_mask_signed;

/// `WEDGE_WEIGHT_BITS` (`reconinter.h:44`).
const WEDGE_WEIGHT_BITS: i64 = 6;
/// `MAX_WEDGE_TYPES` (`enums.h`) — every bsize with a codebook has all 16.
pub const MAX_WEDGE_TYPES: usize = 16;

/// `ROUND_POWER_OF_TWO(value, n)` over `u64`. Written out because `n == 0` is
/// reachable (the lowbd `bd_round`) and C's macro is `(v + ((1 << n) >> 1)) >> n`,
/// i.e. it adds nothing rather than shifting by -1.
#[inline]
fn round_pow2_u64(value: u64, n: u32) -> u64 {
    (value + ((1u64 << n) >> 1)) >> n
}

/// One of the encoder's pixel buffers at the bit depth the block is coded at.
///
/// C spells this as a `uint8_t *` that the high-bit-depth arms reinterpret via
/// `CONVERT_TO_SHORTPTR`, and decides which by `is_cur_buf_hbd(xd)`. Here the
/// buffer carries its own width, so the two can never disagree — which is the
/// whole of what `is_cur_buf_hbd` is used for in this file.
#[derive(Clone, Copy, Debug)]
pub enum Pixels<'a> {
    /// 8-bit buffer (`is_cur_buf_hbd(xd) == 0`).
    Low(&'a [u8]),
    /// 16-bit buffer (`is_cur_buf_hbd(xd) == 1`), used at every bit depth
    /// including 8 when the encoder is in high-bit-depth mode.
    High(&'a [u16]),
}

impl Pixels<'_> {
    /// `is_cur_buf_hbd(xd)`.
    #[inline]
    pub const fn is_hbd(self) -> bool {
        matches!(self, Pixels::High(_))
    }
}

/// `diff[..] = src - pred`, dispatching `aom_subtract_block` /
/// `aom_highbd_subtract_block` on the buffer width. Mixing widths is a caller
/// bug, not a representable encoder state.
#[allow(clippy::too_many_arguments)]
fn subtract(
    rows: usize,
    cols: usize,
    diff: &mut [i16],
    diff_stride: usize,
    src: Pixels<'_>,
    src_stride: usize,
    pred: Pixels<'_>,
    pred_stride: usize,
) {
    match (src, pred) {
        (Pixels::Low(s), Pixels::Low(p)) => {
            subtract_block(rows, cols, diff, diff_stride, s, src_stride, p, pred_stride);
        }
        (Pixels::High(s), Pixels::High(p)) => {
            highbd_subtract_block(rows, cols, diff, diff_stride, s, src_stride, p, pred_stride);
        }
        _ => panic!("subtract: mixed 8-bit and 16-bit buffers"),
    }
}

/// The `x` / `cpi` state every mask pick reads. Grouped rather than passed
/// loose: `rdmult` and `dequant_ac` are both plain `i32` and transposing them
/// is silent.
#[derive(Clone, Copy, Debug)]
pub struct MaskSearchCtx<'a> {
    /// The luma `BLOCK_SIZE`. Also the `plane_bsize` the model RD uses, since
    /// every pick here works on plane 0.
    pub bsize: usize,
    /// `xd->bd`.
    pub bd: u8,
    /// `x->rdmult`.
    pub rdmult: i32,
    /// `x->plane[0].dequant_QTX[1]` — the AC dequant the model RD divides by.
    pub dequant_ac: i32,
    /// `x->mode_costs.wedge_idx_cost[bsize]`, `MAX_WEDGE_TYPES` entries.
    pub wedge_idx_cost: &'a [i32],
}

impl MaskSearchCtx<'_> {
    /// `model_rd_sse_fn[MODELRD_TYPE_MASKED_COMPOUND]`, which resolves to
    /// `model_rd_with_curvfit` (`MODELRD_TYPE_MASKED_COMPOUND == 1 ==
    /// MODELRD_CURVFIT`, model_rd.h:33 / :259-267).
    ///
    /// C selects `dequant_shift` on `is_cur_buf_hbd(xd)`; the port's
    /// [`crate::interp_rd::model_rd_with_curvfit`] selects on `bd > 8`. Those
    /// agree because a high-bit-depth buffer at `bd == 8` takes C's `bd - 5`
    /// arm, which is 3 — the same value the lowbd arm hard-codes.
    #[inline]
    fn model_rd(&self, sse: u64, num_samples: usize) -> (i32, i64) {
        crate::interp_rd::model_rd_with_curvfit(
            self.bsize,
            sse as i64,
            num_samples as i32,
            self.dequant_ac,
            self.bd,
            self.rdmult,
        )
    }

    /// `bd_round = hbd ? (xd->bd - 8) * 2 : 0` — the shift that brings a
    /// high-bit-depth SSE back to the 8-bit scale the RD model is fitted on.
    #[inline]
    fn bd_round(&self, hbd: bool) -> u32 {
        if hbd { (u32::from(self.bd) - 8) * 2 } else { 0 }
    }
}

/// What a wedge pick produced. C returns the RD and writes the rest through
/// out-parameters; the sse out-parameter is `UINT64_MAX` on entry and C
/// asserts it was overwritten, so it is unconditional here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WedgePick {
    /// The return value: the best RD **minus** the chosen index's signalling
    /// cost, which the caller re-adds after deciding the type.
    pub rd: i64,
    /// `mbmi->interinter_comp.wedge_index`.
    pub index: usize,
    /// `mbmi->interinter_comp.wedge_sign`.
    pub sign: usize,
    /// The masked SSE at the winning index, already `bd_round`ed.
    pub sse: u64,
}

/// `pick_wedge_fixed_sign` (compound_type.c:257): the best wedge index at a
/// sign the caller has already decided.
///
/// `residual1 = src - pred1` and `diff10 = pred1 - pred0`, both contiguous at
/// stride `bw`. `hbd` is `is_cur_buf_hbd(xd)`; it reaches only `bd_round`
/// here, since this arm reads no pixels.
pub fn pick_wedge_fixed_sign(
    ctx: &MaskSearchCtx<'_>,
    hbd: bool,
    residual1: &[i16],
    diff10: &[i16],
    wedge_sign: usize,
) -> WedgePick {
    let bw = BLOCK_SIZE_WIDE[ctx.bsize];
    let bh = BLOCK_SIZE_HIGH[ctx.bsize];
    let n = bw * bh;
    assert!(n >= 64, "pick_wedge_fixed_sign: C asserts N >= 64");
    let bd_round = ctx.bd_round(hbd);

    let mut best = WedgePick {
        rd: i64::MAX,
        index: 0,
        sign: wedge_sign,
        sse: 0,
    };
    for index in 0..MAX_WEDGE_TYPES {
        let mask = wedge_mask_signed(ctx.bsize, index, wedge_sign)
            .expect("pick_wedge_fixed_sign called at a bsize with no wedge codebook");
        let sse = round_pow2_u64(
            wedge_sse_from_residuals(residual1, diff10, &mask, n),
            bd_round,
        );
        let (rate, dist) = ctx.model_rd(sse, n);
        let rate = rate + ctx.wedge_idx_cost[index];
        let rd = crate::rd::rdcost(ctx.rdmult, rate, dist);
        if rd < best.rd {
            best = WedgePick {
                rd,
                index,
                sign: wedge_sign,
                sse,
            };
        }
    }
    best.rd -= crate::rd::rdcost(ctx.rdmult, ctx.wedge_idx_cost[best.index], 0);
    best
}

/// `pick_wedge` (compound_type.c:189): the best wedge index AND sign.
///
/// Unlike [`pick_wedge_fixed_sign`] this needs `pred0` and the source, because
/// the per-index sign is decided by `av1_wedge_sign_from_residuals` against a
/// limit derived from both residual energies.
pub fn pick_wedge(
    ctx: &MaskSearchCtx<'_>,
    src: Pixels<'_>,
    src_stride: usize,
    p0: Pixels<'_>,
    residual1: &[i16],
    diff10: &[i16],
) -> WedgePick {
    let bw = BLOCK_SIZE_WIDE[ctx.bsize];
    let bh = BLOCK_SIZE_HIGH[ctx.bsize];
    let n = bw * bh;
    assert!(n >= 64, "pick_wedge: C asserts N >= 64");
    let bd_round = ctx.bd_round(src.is_hbd());

    // residual0 = src - pred0, at stride bw (pred0 is contiguous).
    let mut residual0 = vec![0i16; n];
    subtract(bh, bw, &mut residual0, bw, src, src_stride, p0, bw);

    // C casts both `uint64_t` sums to `int64_t` BEFORE subtracting, so the
    // difference is signed and the `* 64 / 2` that follows truncates toward
    // zero on a negative value.
    let sign_limit = ((sum_squares_i16(&residual0[..n]) as i64)
        - (sum_squares_i16(&residual1[..n]) as i64))
        * (1 << WEDGE_WEIGHT_BITS)
        / 2;

    // C reuses `residual0`'s storage as `ds` and passes it as BOTH the
    // destination and the first source of `av1_wedge_compute_delta_squares`.
    // The kernel is elementwise (`d[i] = a[i]*a[i] - b[i]*b[i]`), so the
    // aliasing is benign; Rust cannot spell it, and a separate destination
    // computes the same values.
    let mut ds = vec![0i16; n];
    wedge_compute_delta_squares(&mut ds, &residual0, residual1, n);

    let mut best = WedgePick {
        rd: i64::MAX,
        index: 0,
        sign: 0,
        sse: 0,
    };
    for index in 0..MAX_WEDGE_TYPES {
        let probe = wedge_mask_signed(ctx.bsize, index, 0)
            .expect("pick_wedge called at a bsize with no wedge codebook");
        let sign = usize::from(wedge_sign_from_residuals(&ds, &probe, n, sign_limit));

        let mask = wedge_mask_signed(ctx.bsize, index, sign)
            .expect("pick_wedge called at a bsize with no wedge codebook");
        let sse = round_pow2_u64(
            wedge_sse_from_residuals(residual1, diff10, &mask, n),
            bd_round,
        );
        let (rate, dist) = ctx.model_rd(sse, n);
        let rate = rate + ctx.wedge_idx_cost[index];
        let rd = crate::rd::rdcost(ctx.rdmult, rate, dist);
        if rd < best.rd {
            best = WedgePick {
                rd,
                index,
                sign,
                sse,
            };
        }
    }
    best.rd -= crate::rd::rdcost(ctx.rdmult, ctx.wedge_idx_cost[best.index], 0);
    best
}

/// `split_qtr[BLOCK_SIZES_ALL]` (compound_type.c:127-146) — the block size of
/// one quadrant. `None` where C stores `BLOCK_INVALID`; `estimate_wedge_sign`
/// asserts it is reached only at a size that has one.
const SPLIT_QTR: [Option<usize>; 22] = [
    None,     // 4X4
    None,     // 4X8
    None,     // 8X4
    Some(0),  // 8X8   -> 4X4
    Some(1),  // 8X16  -> 4X8
    Some(2),  // 16X8  -> 8X4
    Some(3),  // 16X16 -> 8X8
    Some(4),  // 16X32 -> 8X16
    Some(5),  // 32X16 -> 16X8
    Some(6),  // 32X32 -> 16X16
    Some(7),  // 32X64 -> 16X32
    Some(8),  // 64X32 -> 32X16
    Some(9),  // 64X64 -> 32X32
    Some(10), // 64X128 -> 32X64
    Some(11), // 128X64 -> 64X32
    Some(12), // 128X128 -> 64X64
    None,     // 4X16
    None,     // 16X4
    Some(16), // 8X32  -> 4X16
    Some(17), // 32X8  -> 16X4
    Some(18), // 16X64 -> 8X32
    Some(19), // 64X16 -> 32X8
];

/// `estimate_wedge_sign` (compound_type.c:126): guess the wedge sign from the
/// two predictors' quadrant SSEs instead of searching it.
///
/// Returns C's `int8_t` 0/1 as a bool, `true` meaning sign 1.
///
/// # Two things a reader will want to "fix"
/// * The comparison is `tl + br > 0` where `tl` and `br` are built from the
///   **first** and **fourth** quadrants only. C's own comment explains why:
///   the second and third quadrants appear with opposite signs in the full sum
///   and cancel, so they are never computed.
/// * The fourth `vf` call passes **`stride0`** as `pred1`'s stride, not
///   `stride1` (compound_type.c:178). At the one call site both are `bw`, so
///   it is invisible there; it is reproduced rather than corrected because the
///   differential is against C, and a caller that ever passed different
///   strides would see C's behaviour, not the intended one.
#[allow(clippy::too_many_arguments)]
pub fn estimate_wedge_sign(
    bsize: usize,
    bd: u8,
    src: Pixels<'_>,
    src_stride: usize,
    pred0: Pixels<'_>,
    stride0: usize,
    pred1: Pixels<'_>,
    stride1: usize,
) -> bool {
    let f_index = SPLIT_QTR[bsize].expect("estimate_wedge_sign: bsize has no quarter split");
    let (qw, qh) = (BLOCK_SIZE_WIDE[f_index], BLOCK_SIZE_HIGH[f_index]);
    let bw_by2 = BLOCK_SIZE_WIDE[bsize] >> 1;
    let bh_by2 = BLOCK_SIZE_HIGH[bsize] >> 1;

    // `cpi->ppi->fn_ptr[f_index].vf(a, a_stride, b, b_stride, &sse)`: the
    // return value (the variance) is discarded; `esq` receives the SSE.
    let sse_of = |a: Pixels<'_>,
                  a_off: usize,
                  a_stride: usize,
                  b: Pixels<'_>,
                  b_off: usize,
                  b_stride: usize|
     -> u32 {
        match (a, b) {
            (Pixels::Low(x), Pixels::Low(y)) => {
                variance(&x[a_off..], a_stride, &y[b_off..], b_stride, qw, qh).1
            }
            (Pixels::High(x), Pixels::High(y)) => {
                highbd_variance(&x[a_off..], a_stride, &y[b_off..], b_stride, qw, qh, bd).1
            }
            _ => panic!("estimate_wedge_sign: mixed 8-bit and 16-bit buffers"),
        }
    };

    let esq00 = sse_of(src, 0, src_stride, pred0, 0, stride0);
    let esq01 = sse_of(
        src,
        bh_by2 * src_stride + bw_by2,
        src_stride,
        pred0,
        bh_by2 * stride0 + bw_by2,
        stride0,
    );
    let esq10 = sse_of(src, 0, src_stride, pred1, 0, stride1);
    // NOTE the stride: C passes `stride0` here for pred1. See the doc comment.
    let esq11 = sse_of(
        src,
        bh_by2 * src_stride + bw_by2,
        src_stride,
        pred1,
        bh_by2 * stride1 + bw_by2,
        stride0,
    );

    let tl = i64::from(esq00) - i64::from(esq10);
    let br = i64::from(esq11) - i64::from(esq01);
    tl + br > 0
}

/// `pick_interinter_wedge` (compound_type.c:299): the inter-inter wedge search.
///
/// `fast_wedge_sign_estimate` is `cpi->sf.inter_sf.fast_wedge_sign_estimate`;
/// when set, the sign comes from [`estimate_wedge_sign`] and only the index is
/// searched.
#[allow(clippy::too_many_arguments)]
pub fn pick_interinter_wedge(
    ctx: &MaskSearchCtx<'_>,
    fast_wedge_sign_estimate: bool,
    src: Pixels<'_>,
    src_stride: usize,
    p0: Pixels<'_>,
    p1: Pixels<'_>,
    residual1: &[i16],
    diff10: &[i16],
) -> WedgePick {
    assert!(
        is_interinter_compound_used(CompoundType::Wedge, ctx.bsize),
        "pick_interinter_wedge: COMPOUND_WEDGE is not usable at this bsize"
    );
    let bw = BLOCK_SIZE_WIDE[ctx.bsize];
    if fast_wedge_sign_estimate {
        let sign = usize::from(estimate_wedge_sign(
            ctx.bsize, ctx.bd, src, src_stride, p0, bw, p1, bw,
        ));
        pick_wedge_fixed_sign(ctx, src.is_hbd(), residual1, diff10, sign)
    } else {
        pick_wedge(ctx, src, src_stride, p0, residual1, diff10)
    }
}

/// What [`pick_interinter_seg`] produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegPick {
    /// The best RD (C returns it directly, with no cost subtracted).
    pub rd: i64,
    /// `mbmi->interinter_comp.mask_type`.
    pub mask_type: DiffwtdMaskType,
    /// The masked SSE at the winning mask type, already `bd_round`ed.
    pub sse: u64,
    /// `xd->seg_mask` as the winner leaves it: `bw * bh` entries at stride `bw`.
    ///
    /// C builds mask type 0 straight into `xd->seg_mask` and type 1 into a
    /// stack buffer, then `memcpy`s the stack one back if type 1 won — with a
    /// length of `2 * N`, of which only the first `N` were ever written or are
    /// ever read (the blend walks `h` rows at stride `bw`). This returns the
    /// `N` that mean something.
    pub seg_mask: Vec<u8>,
}

/// `pick_interinter_seg` (compound_type.c:332): choose between the two
/// difference-weighted mask types.
pub fn pick_interinter_seg(
    ctx: &MaskSearchCtx<'_>,
    p0: Pixels<'_>,
    p1: Pixels<'_>,
    residual1: &[i16],
    diff10: &[i16],
) -> SegPick {
    let bw = BLOCK_SIZE_WIDE[ctx.bsize];
    let bh = BLOCK_SIZE_HIGH[ctx.bsize];
    let n = bw * bh;
    let bd_round = ctx.bd_round(p0.is_hbd());

    let mut best = SegPick {
        rd: i64::MAX,
        mask_type: DiffwtdMaskType::Diffwtd38,
        sse: 0,
        seg_mask: vec![0u8; n],
    };
    for mask_type in [DiffwtdMaskType::Diffwtd38, DiffwtdMaskType::Diffwtd38Inv] {
        let mut mask = vec![0u8; n];
        match (p0, p1) {
            (Pixels::Low(a), Pixels::Low(b)) => {
                build_compound_diffwtd_mask(&mut mask, mask_type, a, bw, b, bw, bh, bw);
            }
            (Pixels::High(a), Pixels::High(b)) => {
                build_compound_diffwtd_mask_highbd(
                    &mut mask,
                    mask_type,
                    a,
                    bw,
                    b,
                    bw,
                    bh,
                    bw,
                    u32::from(ctx.bd),
                );
            }
            _ => panic!("pick_interinter_seg: mixed 8-bit and 16-bit buffers"),
        }
        let sse = round_pow2_u64(
            wedge_sse_from_residuals(residual1, diff10, &mask, n),
            bd_round,
        );
        let (rate, dist) = ctx.model_rd(sse, n);
        let rd = crate::rd::rdcost(ctx.rdmult, rate, dist);
        if rd < best.rd {
            best = SegPick {
                rd,
                mask_type,
                sse,
                seg_mask: mask,
            };
        }
    }
    best
}

/// `pick_interintra_wedge` (compound_type.c:394): the interintra wedge search.
///
/// `p0` is the intra predictor and `p1` the inter one, both contiguous at
/// stride `bw`. The residuals C derives here are its own — `residual1 = src -
/// p1` and `diff10 = p1 - p0` — so unlike the inter-inter picks this one takes
/// pixels rather than residuals. The sign is fixed at 0: interintra codes no
/// wedge sign.
pub fn pick_interintra_wedge(
    ctx: &MaskSearchCtx<'_>,
    src: Pixels<'_>,
    src_stride: usize,
    p0: Pixels<'_>,
    p1: Pixels<'_>,
) -> WedgePick {
    assert!(
        is_wedge_used(ctx.bsize),
        "pick_interintra_wedge: av1_is_wedge_used(bsize) must hold"
    );
    let bw = BLOCK_SIZE_WIDE[ctx.bsize];
    let bh = BLOCK_SIZE_HIGH[ctx.bsize];
    let n = bw * bh;

    let mut residual1 = vec![0i16; n];
    let mut diff10 = vec![0i16; n];
    subtract(bh, bw, &mut residual1, bw, src, src_stride, p1, bw);
    subtract(bh, bw, &mut diff10, bw, p1, bw, p0, bw);

    pick_wedge_fixed_sign(ctx, src.is_hbd(), &residual1, &diff10, 0)
}
