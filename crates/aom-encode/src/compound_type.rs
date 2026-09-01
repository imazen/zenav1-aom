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
