//! Block-level intra-mode RD evaluation — the first slice of the speed-0
//! KEY-frame mode-search decision layer: for one coding block, evaluate a
//! candidate intra mode end-to-end (predict -> subtract -> forward transform +
//! quantize + trellis -> rate + transform-domain distortion -> RDCOST) and
//! pick the minimum-RD candidate from a caller-supplied list.
//!
//! Every step composes an individually C-validated piece:
//! [`aom_intra::predict_intra_high`], [`aom_dist::highbd_subtract_block`],
//! [`crate::xform_quant_optimize`], [`aom_txb::cost_coeffs_txb`],
//! [`aom_txb::get_tx_type_cost`], [`crate::mode_costs::intra_mode_info_cost_y`],
//! [`crate::dist_block_tx_domain`], [`crate::rd::rdcost`] — and the
//! *composition* is differentially validated against the identical chain of C
//! reference steps (`intra_rd_pick_diff.rs`).
//!
//! SCOPE — this is a composition primitive, deliberately narrower than
//! libaom's `av1_rd_pick_intra_sby_mode`:
//! - single-transform-block coding blocks only (`bsize` dims == `tx_size`
//!   dims; no tx-size search / tx partition),
//! - the candidate list and its order are the caller's (none of the C search's
//!   ordering, hog/variance pruning, early termination, or adaptive
//!   angle-delta refinement),
//! - one caller-fixed `tx_type` per evaluation (no tx-type search),
//! - transform-domain distortion only (no reconstruction-domain switch, no
//!   skip-vs-coded RD alternative),
//! - plane 0 (luma), KEY-frame Y mode rate (`y_mode_costs` via the above/left
//!   `intra_mode_context` pair), `palette_size[0] == 0`.

use crate::mode_costs::{intra_mode_info_cost_y, IntraModeCosts};
use crate::{
    dist_block_tx_domain, rd, xform_quant_optimize, BlockContext, OptimizeInputs, QuantKind,
    QuantParams,
};
use aom_dist::highbd_subtract_block;
use aom_entropy::partition::get_y_mode_ctx;
use aom_intra::predict_intra_high;
use aom_txb::{cost_coeffs_txb, get_tx_type_cost, CoeffCostTables, TxTypeCosts};

/// `ANGLE_STEP` (enums.h): degrees per signaled angle-delta step.
pub const ANGLE_STEP: i32 = 3;

/// Per-block prediction environment: the reconstructed neighbourhood the
/// predictor reads, the source pixels, geometry, and edge availability
/// (`intra_avail` outputs). `bsize` must have the same dimensions as
/// `tx_size` (single-txb scope).
pub struct IntraRdEnv<'a> {
    pub recon: &'a [u16],
    /// Index of the block's top-left pixel in `recon`.
    pub ref_off: usize,
    pub ref_stride: usize,
    pub src: &'a [u16],
    /// Index of the block's top-left pixel in `src`.
    pub src_off: usize,
    pub src_stride: usize,
    pub tx_size: usize,
    /// Block size (BLOCK_SIZE discriminant), dims equal to `tx_size`.
    pub bsize: usize,
    pub n_top_px: usize,
    pub n_topright_px: i32,
    pub n_left_px: usize,
    pub n_bottomleft_px: i32,
    pub disable_edge_filter: bool,
    pub filter_type: i32,
    pub bd: u8,
}

/// Rate inputs: the derived cost tables plus the frame/neighbour state that
/// selects the mode-signaling rate terms.
pub struct IntraRdRates<'a> {
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
    pub mode_costs: &'a IntraModeCosts,
    pub rdmult: i32,
    /// Above / left neighbour Y modes (`None` = unavailable -> `DC_PRED`),
    /// selecting the KEY-frame `y_mode_costs` context pair.
    pub above_mode: Option<i32>,
    pub left_mode: Option<i32>,
    pub try_palette: bool,
    pub palette_bsize_ctx: usize,
    pub palette_mode_ctx: usize,
    pub enable_filter_intra: bool,
    pub allow_intrabc: bool,
    pub reduced_tx_set: bool,
    pub lossless: bool,
}

/// One candidate: an intra mode with its angle delta (UNscaled, in
/// `[-MAX_ANGLE_DELTA, MAX_ANGLE_DELTA]`; scaled by [`ANGLE_STEP`] for
/// prediction) or a filter-intra variant (`mode` must be `DC_PRED`).
#[derive(Clone, Copy, Debug)]
pub struct IntraCandidate {
    pub mode: usize,
    pub angle_delta: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
}

/// One candidate's RD evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntraModeRd {
    /// Total rate: coefficient bits + tx_type signaling + Y mode-info signaling.
    pub rate: i32,
    /// Transform-domain distortion (`dist_block_tx_domain`).
    pub dist: i64,
    /// `RDCOST(rdmult, rate, dist)`.
    pub rd: i64,
    /// Post-trellis end-of-block.
    pub eob: u16,
}

/// Evaluate one intra candidate for one single-txb coding block: predict from
/// the reconstructed edges, subtract, transform + quantize + trellis
/// (`xform_quant_optimize`), then combine
/// `rate = cost_coeffs_txb + get_tx_type_cost + intra_mode_info_cost_y` and
/// `dist = dist_block_tx_domain` into one `RDCOST` — the RD shape of the C
/// mode loop (`this_rd = RDCOST(rdmult, this_rate, this_distortion)`).
#[allow(clippy::too_many_arguments)]
pub fn intra_mode_rd_eval(
    env: &IntraRdEnv,
    rates: &IntraRdRates,
    cand: &IntraCandidate,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    bctx: &BlockContext,
    opt: &OptimizeInputs,
) -> IntraModeRd {
    let w = crate::TX_W[env.tx_size];
    let h = crate::TX_H[env.tx_size];
    assert_eq!(crate::BLK_W[env.bsize], w, "bsize/tx_size width mismatch (single-txb scope)");
    assert_eq!(crate::BLK_H[env.bsize], h, "bsize/tx_size height mismatch (single-txb scope)");

    // Predict into a tight w-stride buffer (av1_predict_intra_block).
    let mut pred = vec![0u16; w * h];
    predict_intra_high(
        env.recon,
        env.ref_off,
        env.ref_stride,
        &mut pred,
        w,
        cand.mode,
        cand.angle_delta * ANGLE_STEP,
        cand.use_filter_intra,
        cand.filter_intra_mode,
        env.disable_edge_filter,
        env.filter_type,
        env.tx_size,
        env.n_top_px,
        env.n_topright_px,
        env.n_left_px,
        env.n_bottomleft_px,
        env.bd as i32,
    );

    // Residual = src - pred (aom_highbd_subtract_block).
    let mut residual = vec![0i16; w * h];
    highbd_subtract_block(
        h,
        w,
        &mut residual,
        w,
        &env.src[env.src_off..],
        env.src_stride,
        &pred,
        w,
    );

    // Forward transform + quantize + trellis (the speed-0 coefficient path).
    let r = xform_quant_optimize(&residual, env.tx_size, tx_type, kind, qp, bctx, opt);

    // Rate: post-trellis coefficient bits (av1_cost_coeffs_txb) + tx_type
    // signaling + Y mode-info signaling. The real av1_cost_coeffs_txb includes
    // get_tx_type_cost inside its eob>0 body but its eob==0 branch returns the
    // txb_skip cost ALONE (an all-zero txb signals no tx_type) — so the
    // tx_type term is gated on eob != 0.
    let coeff_rate = cost_coeffs_txb(
        &r.qcoeff,
        r.eob as usize,
        env.tx_size,
        tx_type,
        r.txb_skip_ctx,
        r.dc_sign_ctx,
        rates.coeff_costs,
    );
    let tx_type_rate = if r.eob != 0 {
        get_tx_type_cost(
            rates.tx_type_costs,
            0,
            env.tx_size,
            tx_type,
            false,
            rates.reduced_tx_set,
            rates.lossless,
            cand.use_filter_intra,
            cand.filter_intra_mode,
            cand.mode,
        )
    } else {
        0
    };
    let (above_ctx, left_ctx) = get_y_mode_ctx(rates.above_mode, rates.left_mode);
    let mode_cost = rates.mode_costs.y_mode_costs[above_ctx][left_ctx][cand.mode];
    let mode_rate = intra_mode_info_cost_y(
        rates.mode_costs,
        mode_cost,
        cand.mode,
        env.bsize,
        cand.angle_delta,
        cand.use_filter_intra,
        cand.filter_intra_mode,
        false, // use_intrabc: an intrabc block would not run the intra mode loop
        rates.try_palette,
        rates.palette_bsize_ctx,
        rates.palette_mode_ctx,
        rates.enable_filter_intra,
        rates.allow_intrabc,
    );
    let rate = coeff_rate + tx_type_rate + mode_rate;

    // Transform-domain distortion, then one RDCOST over the summed rate.
    let (dist, _sse) = dist_block_tx_domain(&r.coeff, &r.dqcoeff, env.tx_size, env.bd);
    let rd = rd::rdcost(rates.rdmult, rate, dist);

    IntraModeRd { rate, dist, rd, eob: r.eob }
}

/// Evaluate every candidate and return `(argmin_index, per-candidate evals)`.
/// Ties keep the earliest candidate (strict `<` update, as the C loop's
/// `this_rd < best_rd`). The candidate order is the caller's — this does NOT
/// reproduce libaom's search order or pruning.
#[allow(clippy::too_many_arguments)]
pub fn pick_intra_mode_rd(
    env: &IntraRdEnv,
    rates: &IntraRdRates,
    candidates: &[IntraCandidate],
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    bctx: &BlockContext,
    opt: &OptimizeInputs,
) -> (usize, Vec<IntraModeRd>) {
    assert!(!candidates.is_empty());
    let evals: Vec<IntraModeRd> = candidates
        .iter()
        .map(|cand| intra_mode_rd_eval(env, rates, cand, tx_type, kind, qp, bctx, opt))
        .collect();
    let mut best = 0usize;
    let mut best_rd = i64::MAX;
    for (i, e) in evals.iter().enumerate() {
        if e.rd < best_rd {
            best_rd = e.rd;
            best = i;
        }
    }
    (best, evals)
}

// ---------------------------------------------------------------------------
// Candidate enumeration + speed-0 gating of av1_rd_pick_intra_sby_mode
// (av1/encoder/intra_mode_search.c) — the loop-head fidelity layer.
// ---------------------------------------------------------------------------

/// `INTRA_MODE_END` / `INTRA_MODES` (enums.h): 13 luma intra modes.
pub const INTRA_MODES: usize = 13;
/// `MAX_ANGLE_DELTA` (enums.h).
pub const MAX_ANGLE_DELTA: i32 = 3;
/// `LUMA_MODE_COUNT` (enums.h): `13 + 8 directional modes * 6 nonzero deltas`.
pub const LUMA_MODE_COUNT: usize = 61;

/// `intra_rd_search_mode_order` (intra_mode_search.c): the evaluation order of
/// the 13 modes at delta 0 — DC, H, V, SMOOTH, PAETH, SMOOTH_V, SMOOTH_H,
/// D135, D203, D157, D67, D113, D45 (as `PREDICTION_MODE` values).
pub const INTRA_RD_SEARCH_MODE_ORDER: [usize; INTRA_MODES] =
    [0, 2, 1, 9, 12, 10, 11, 4, 7, 6, 8, 5, 3];

/// `luma_delta_angles_order` (intra_mode_search.c): even deltas first, used
/// when `prune_luma_odd_delta_angles_in_intra` reorders the delta sweep.
pub const LUMA_DELTA_ANGLES_ORDER: [i32; 6] = [-2, 2, -3, -1, 1, 3];

/// `set_y_mode_and_delta_angle` (intra_mode_search.c, exported): map a loop
/// index `0..LUMA_MODE_COUNT` to the `(mode, angle_delta)` it evaluates.
/// Indices `< INTRA_MODES` walk [`INTRA_RD_SEARCH_MODE_ORDER`] at delta 0;
/// the rest sweep V..D67 (mode values 1..=8) x six nonzero deltas
/// (-3..-1, 1..3, or [`LUMA_DELTA_ANGLES_ORDER`] when `reorder_delta_angle_eval`).
pub fn set_y_mode_and_delta_angle(mode_idx: usize, reorder_delta_angle_eval: bool) -> (usize, i32) {
    assert!(mode_idx < LUMA_MODE_COUNT);
    if mode_idx < INTRA_MODES {
        (INTRA_RD_SEARCH_MODE_ORDER[mode_idx], 0)
    } else {
        let mode = (mode_idx - INTRA_MODES) / (MAX_ANGLE_DELTA as usize * 2) + 1; // + V_PRED
        let delta_angle_eval_idx = (mode_idx - INTRA_MODES) % (MAX_ANGLE_DELTA as usize * 2);
        let delta = if reorder_delta_angle_eval {
            LUMA_DELTA_ANGLES_ORDER[delta_angle_eval_idx]
        } else if delta_angle_eval_idx < 3 {
            delta_angle_eval_idx as i32 - 3
        } else {
            delta_angle_eval_idx as i32 - 2
        };
        (mode, delta)
    }
}

/// `av1_is_diagonal_mode` (reconintra.h): D45..D67 (mode values 3..=8).
#[inline]
pub fn is_diagonal_mode(mode: usize) -> bool {
    (3..=8).contains(&mode)
}

/// `max_txsize_lookup[BLOCK_SIZES_ALL]` (common_data.h): largest SQUARE
/// tx size for a block size (TX_4X4=0 .. TX_64X64=4) — the
/// `intra_y_mode_mask` index.
pub const MAX_TXSIZE_LOOKUP: [usize; 22] =
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0, 1, 1, 2, 2];

/// The gating inputs of the `av1_rd_pick_intra_sby_mode` candidate loop —
/// `IntraModeCfg` tool flags (all default `true` on the aomenc CLI), the
/// `intra_sf` members the loop reads, and the per-search pruning state.
///
/// Speed-0 all-intra values (speed_features.c; defaults at
/// `init_intra_mode_sf` + `set_allintra_speed_features_framesize_independent`
/// speed-0 section), each named:
/// - `disable_smooth_intra = 0` (default; never set at speed 0)
/// - `prune_filter_intra_level = 0` (default)
/// - `intra_y_mode_mask[..] = INTRA_ALL` for every tx size (default)
/// - `prune_luma_odd_delta_angles_in_intra = 0` (default)
/// - `intra_pruning_with_hog = 1` (set for allintra speed 0, thresh -1.2):
///   `directional_mode_skip_mask` is that prune's OUTPUT, an input here
///   (all-false when HOG keeps everything).
/// - `use_mb_mode_cache`: modelled OFF (MB mode cache is populated only by
///   superblock-level re-search paths, not the plain speed-0 walk).
#[derive(Clone, Debug)]
pub struct IntraSbyGates {
    /// `intra_mode_cfg.enable_diagonal_intra` (CLI default on).
    pub enable_diagonal_intra: bool,
    /// `intra_mode_cfg.enable_directional_intra`.
    pub enable_directional_intra: bool,
    /// `intra_mode_cfg.enable_smooth_intra`.
    pub enable_smooth_intra: bool,
    /// `intra_mode_cfg.enable_paeth_intra`.
    pub enable_paeth_intra: bool,
    /// `intra_mode_cfg.enable_angle_delta`.
    pub enable_angle_delta: bool,
    /// `intra_sf.disable_smooth_intra`.
    pub disable_smooth_intra: bool,
    /// `intra_sf.prune_filter_intra_level`.
    pub prune_filter_intra_level: i32,
    /// `intra_sf.intra_y_mode_mask[max_txsize_lookup[bsize]]` (bit per mode).
    pub intra_y_mode_mask: [u16; 5],
    /// `directional_mode_skip_mask` — the HOG prune output (index = mode).
    pub directional_mode_skip_mask: [bool; INTRA_MODES],
    /// `intra_sf.prune_luma_odd_delta_angles_in_intra`.
    pub prune_luma_odd_delta_angles_in_intra: bool,
}

impl IntraSbyGates {
    /// The speed-0 all-intra configuration (see struct docs), with the HOG
    /// skip mask supplied by the caller.
    pub fn speed0(directional_mode_skip_mask: [bool; INTRA_MODES]) -> Self {
        IntraSbyGates {
            enable_diagonal_intra: true,
            enable_directional_intra: true,
            enable_smooth_intra: true,
            enable_paeth_intra: true,
            enable_angle_delta: true,
            disable_smooth_intra: false,
            prune_filter_intra_level: 0,
            intra_y_mode_mask: [0x1fff; 5], // INTRA_ALL: all 13 mode bits
            directional_mode_skip_mask,
            prune_luma_odd_delta_angles_in_intra: false,
        }
    }

    /// The static skip chain of the candidate loop (intra_mode_search.c
    /// 1555-1594), for a candidate `(mode, delta)` on `bsize`: `true` =
    /// evaluate, `false` = `continue`. Excludes the model-RD prune and the
    /// odd-delta RD prune (dynamic; see
    /// [`prune_luma_odd_delta_angles_using_rd_cost`]).
    pub fn visits(&self, mode: usize, luma_delta_angle: i32, bsize: usize) -> bool {
        use aom_entropy::partition::{is_directional_mode, use_angle_delta};
        let is_directional = is_directional_mode(mode as i32);
        if is_diagonal_mode(mode) && !self.enable_diagonal_intra {
            return false;
        }
        if is_directional && !self.enable_directional_intra {
            return false;
        }
        // SMOOTH_V_PRED = 10, SMOOTH_H_PRED = 11, SMOOTH_PRED = 9, PAETH = 12.
        if (!self.enable_smooth_intra || self.disable_smooth_intra) && (mode == 11 || mode == 10) {
            return false;
        }
        if !self.enable_smooth_intra && mode == 9 {
            return false;
        }
        if self.disable_smooth_intra && self.prune_filter_intra_level == 0 && mode == 9 {
            return false;
        }
        if !self.enable_paeth_intra && mode == 12 {
            return false;
        }
        if is_directional && self.directional_mode_skip_mask[mode] {
            return false;
        }
        if is_directional
            && !(use_angle_delta(bsize) && self.enable_angle_delta)
            && luma_delta_angle != 0
        {
            return false;
        }
        (self.intra_y_mode_mask[MAX_TXSIZE_LOOKUP[bsize]] & (1 << mode)) != 0
    }

    /// The full visit sequence for one block: every `(mode, angle_delta)` the
    /// C loop evaluates (before its dynamic model-RD / odd-delta-RD prunes),
    /// in exact order.
    pub fn visit_sequence(&self, bsize: usize) -> Vec<(usize, i32)> {
        (0..LUMA_MODE_COUNT)
            .map(|idx| set_y_mode_and_delta_angle(idx, self.prune_luma_odd_delta_angles_in_intra))
            .filter(|&(mode, delta)| self.visits(mode, delta, bsize))
            .collect()
    }
}

/// `SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY` (intra_mode_search.c): per-mode RD
/// bookkeeping over deltas -4..=4 (delta `d` at index `d + MAX_ANGLE_DELTA + 1`;
/// indices 0 and 8 stay `INT64_MAX`).
pub const SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY: usize = 9;

/// `prune_luma_odd_delta_angles_using_rd_cost` (intra_mode_search.c): prune an
/// odd delta angle when both even-delta neighbours' recorded RD costs exceed
/// `best_rd + best_rd/8`. `intra_modes_rd_cost` is this mode's delta-indexed
/// RD array (see [`SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY`]). At speed 0 the
/// controlling sf is 0, so this never prunes.
pub fn prune_luma_odd_delta_angles_using_rd_cost(
    mode: usize,
    luma_delta_angle: i32,
    intra_modes_rd_cost: &[i64; SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY],
    best_rd: i64,
    prune_luma_odd_delta_angles_in_intra: bool,
) -> bool {
    if !prune_luma_odd_delta_angles_in_intra
        || !aom_entropy::partition::is_directional_mode(mode as i32)
        || (luma_delta_angle.abs() & 1) == 0
        || best_rd == i64::MAX
    {
        return false;
    }
    let rd_thresh = best_rd + (best_rd >> 3);
    intra_modes_rd_cost[(luma_delta_angle + MAX_ANGLE_DELTA) as usize] > rd_thresh
        && intra_modes_rd_cost[(luma_delta_angle + MAX_ANGLE_DELTA + 2) as usize] > rd_thresh
}

// ---------------------------------------------------------------------------
// Model-RD pruning (intra_mode_search.c) — the Hadamard-SATD gate that runs
// before each candidate's full tx search in the av1_rd_pick_intra_sby_mode
// loop. The model cost itself is [`crate::tx_search::intra_model_rd_y`].
// ---------------------------------------------------------------------------

/// `TOP_INTRA_MODEL_COUNT` (speed_features.h): the `top_intra_model_rd[]`
/// array length in the mode loop.
pub const TOP_INTRA_MODEL_COUNT: usize = 4;

/// `get_model_rd_index_for_pruning` (intra_mode_search.c): which
/// `top_intra_model_rd` slot `prune_intra_y_mode` compares against.
/// Neighbour modes: `None` = unavailable (the C's guarded
/// `xd->left_mbmi->mode` reads).
///
/// Speed-0 all-intra values (speed_features.c): `top_intra_model_count_allowed
/// = TOP_INTRA_MODEL_COUNT` (=4) and `adapt_top_model_rd_count_using_neighbors
/// = 0 (both `init_intra_mode_sf` defaults; the allintra path lowers them only
/// at speed >= 1 / speed >= 6) — so at speed 0 this is always `4 - 1 = 3`.
pub fn get_model_rd_index_for_pruning(
    cur_mode: usize,
    qindex: i32,
    top_intra_model_count_allowed: i32,
    adapt_top_model_rd_count_using_neighbors: bool,
    left_mode: Option<usize>,
    above_mode: Option<usize>,
) -> i32 {
    if !adapt_top_model_rd_count_using_neighbors {
        return top_intra_model_count_allowed - 1;
    }
    let mut model_rd_index_for_pruning = top_intra_model_count_allowed - 1;
    let is_left_mode_neq_cur_mode = left_mode.is_some_and(|m| m != cur_mode);
    let is_above_mode_neq_cur_mode = above_mode.is_some_and(|m| m != cur_mode);
    // qidx 0..=127: reduce when EITHER available neighbour mode differs;
    // qidx 128..=255: reduce only when BOTH differ.
    let reduce = if qindex <= 127 {
        is_left_mode_neq_cur_mode || is_above_mode_neq_cur_mode
    } else {
        is_left_mode_neq_cur_mode && is_above_mode_neq_cur_mode
    };
    if reduce {
        model_rd_index_for_pruning = (model_rd_index_for_pruning - 1).max(0);
    }
    model_rd_index_for_pruning
}

/// `prune_intra_y_mode` (intra_mode_search.c): sorted-insert `this_model_rd`
/// into the top-N model RDs, then prune when it exceeds
/// `1.00 * top[model_rd_index_for_pruning]` or `1.50 * best_model_rd` (both
/// DOUBLE comparisons — the i64 operands convert to f64 exactly for every
/// reachable SATD magnitude, and the reference build carries no FMA, so plain
/// Rust f64 mul + compare replicates the C bit-for-bit); otherwise lower
/// `best_model_rd`. Mutates both accumulators exactly as the C does.
pub fn prune_intra_y_mode(
    this_model_rd: i64,
    best_model_rd: &mut i64,
    top_intra_model_rd: &mut [i64],
    max_model_cnt_allowed: usize,
    model_rd_index_for_pruning: usize,
) -> bool {
    const THRESH_BEST: f64 = 1.50;
    const THRESH_TOP: f64 = 1.00;
    for i in 0..max_model_cnt_allowed {
        if this_model_rd < top_intra_model_rd[i] {
            for j in (i + 1..max_model_cnt_allowed).rev() {
                top_intra_model_rd[j] = top_intra_model_rd[j - 1];
            }
            top_intra_model_rd[i] = this_model_rd;
            break;
        }
    }
    if top_intra_model_rd[model_rd_index_for_pruning] != i64::MAX
        && (this_model_rd as f64)
            > THRESH_TOP * (top_intra_model_rd[model_rd_index_for_pruning] as f64)
    {
        return true;
    }
    if this_model_rd != i64::MAX && (this_model_rd as f64) > THRESH_BEST * (*best_model_rd as f64)
    {
        return true;
    }
    if this_model_rd < *best_model_rd {
        *best_model_rd = this_model_rd;
    }
    false
}

// ---------------------------------------------------------------------------
// intra_rd_variance_factor (intra_mode_search.c) — the ALLINTRA visual-quality
// RD scale applied to each candidate's this_rd in the mode loop.
// ---------------------------------------------------------------------------

/// `MI_SIZE` / `MI_SIZE_LOG2` (enums.h).
pub const MI_SIZE: usize = 4;

/// `mi_size_wide` / `mi_size_high` `[BLOCK_SIZES_ALL]` (common_data.h).
const MI_W_ALL: [usize; 22] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
const MI_H_ALL: [usize; 22] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];

/// `Block4x4VarInfo` (block.h): the per-4x4 source-variance cache — one entry
/// per mi position in the superblock, initialized `var = -1` /
/// `log_var = -1.0` per SB (`init_src_var_info_of_4x4_sub_blocks`, which runs
/// exactly when the variance factor is active: ALLINTRA +
/// `INTRA_RD_VAR_THRESH(speed) > 0`). The cache persists across every
/// candidate and coding block of the SB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block4x4VarInfo {
    pub var: i32,
    pub log_var: f64,
}

impl Block4x4VarInfo {
    /// One initialized (invalid) entry.
    pub fn init() -> Self {
        Block4x4VarInfo { var: -1, log_var: -1.0 }
    }
    /// A fresh per-superblock cache for `sb_size`.
    pub fn sb_cache(sb_size: usize) -> Vec<Block4x4VarInfo> {
        vec![Block4x4VarInfo::init(); MI_W_ALL[sb_size] * MI_H_ALL[sb_size]]
    }
}

/// `av1_calc_normalized_variance` for one 4x4 sub-block: the
/// `fn_ptr[BLOCK_4X4].vf` variance against an all-zero reference (= 16x the
/// raw per-pixel variance, "normalized" by the /16.0 in the log1p below).
/// fn_ptr resolution by stream depth: `aom_variance4x4` over the u8 planes
/// for 8-bit streams; `aom_highbd_<bd>_variance4x4` over the u16 planes for
/// bd > 8 (both individually C-validated in aom-dist).
fn calc_normalized_variance_4x4(buf: &[u16], off: usize, stride: usize, bd: u8) -> i32 {
    if bd > 8 {
        const ZEROS16: [u16; 4] = [0; 4];
        aom_dist::highbd_variance(&buf[off..], stride, &ZEROS16, 0, 4, 4, bd).0 as i32
    } else {
        // The production 8-bit encoder reads u8 planes; the strided window
        // holds the same 16 values, so a tight copy is kernel-identical.
        let mut w8 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                debug_assert!(buf[off + r * stride + c] <= 255);
                w8[r * 4 + c] = buf[off + r * stride + c] as u8;
            }
        }
        const ZEROS8: [u8; 4] = [0; 4];
        aom_dist::variance(&w8, 4, &ZEROS8, 0, 4, 4).0 as i32
    }
}

/// The pixel-plane / geometry inputs of [`intra_rd_variance_factor`].
/// `mb_to_right_edge` / `mb_to_bottom_edge` are the MACROBLOCKD 1/8-pel edge
/// fields (negative = the block overhangs the frame; the overhang is clipped
/// out of the variance walk).
pub struct VarFactorInputs<'a> {
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    pub recon: &'a [u16],
    pub ref_off: usize,
    pub ref_stride: usize,
    pub bsize: usize,
    pub sb_size: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub mb_to_right_edge: i32,
    pub mb_to_bottom_edge: i32,
    pub bd: u8,
}

/// `intra_rd_variance_factor` (+ `compute_avg_log_variance`),
/// intra_mode_search.c: the ALLINTRA-mode visual-quality RD scale in
/// `[1.0, 3.0]` from how the block's reconstructed variance tracks its source
/// variance. `INTRA_RD_VAR_THRESH(speed) = 1.0 - 0.25 * speed` — active
/// (positive) only for speeds 0..=3; at speed 0 the threshold is 1.0.
///
/// Per 4x4 sub-block: `log1p(var/16.0)` of source (cached in `cache`, the
/// per-SB [`Block4x4VarInfo`] array) and of the CURRENT recon plane content —
/// which, in the mode loop, is whatever `av1_pick_uniform_tx_size_type_yrd`
/// left there: the reconstruction of the LAST tx size the depth sweep
/// evaluated (the C never re-runs the winner; loop-order-sensitive state).
/// Averages accumulate in the C's exact row-major order; all arithmetic is
/// f64 with no FMA (matching the reference build), and `f64::ln_1p` resolves
/// to the same libm `log1p` the oracle calls.
pub fn intra_rd_variance_factor(
    speed: i32,
    p: &VarFactorInputs,
    cache: &mut [Block4x4VarInfo],
) -> f64 {
    let threshold = 1.0 - (0.25 * f64::from(speed)); // INTRA_RD_VAR_THRESH
    if threshold <= 0.0 {
        return 1.0;
    }

    let mut variance_rd_factor = 1.0f64;
    let mut avg_log_src_variance = 0.0f64;
    let mut avg_log_recon_variance = 0.0f64;

    // compute_avg_log_variance.
    let mi_row_in_sb = (p.mi_row as usize) & (MI_H_ALL[p.sb_size] - 1);
    let mi_col_in_sb = (p.mi_col as usize) & (MI_W_ALL[p.sb_size] - 1);
    let right_overflow =
        if p.mb_to_right_edge < 0 { ((-p.mb_to_right_edge) >> 3) as usize } else { 0 };
    let bottom_overflow =
        if p.mb_to_bottom_edge < 0 { ((-p.mb_to_bottom_edge) >> 3) as usize } else { 0 };
    let bw = MI_SIZE * MI_W_ALL[p.bsize] - right_overflow;
    let bh = MI_SIZE * MI_H_ALL[p.bsize] - bottom_overflow;

    let mut i = 0usize;
    while i < bh {
        let r = mi_row_in_sb + (i >> 2); // MI_SIZE_LOG2
        let mut j = 0usize;
        while j < bw {
            let c = mi_col_in_sb + (j >> 2);
            let mi_offset = r * MI_W_ALL[p.sb_size] + c;
            let info = &mut cache[mi_offset];
            let log_src_var;
            if info.var < 0 {
                let src_var = calc_normalized_variance_4x4(
                    p.src,
                    p.src_off + i * p.src_stride + j,
                    p.src_stride,
                    p.bd,
                );
                info.var = src_var;
                log_src_var = (f64::from(src_var) / 16.0).ln_1p();
                info.log_var = log_src_var;
            } else if info.log_var < 0.0 {
                log_src_var = (f64::from(info.var) / 16.0).ln_1p();
                info.log_var = log_src_var;
            } else {
                log_src_var = info.log_var;
            }
            avg_log_src_variance += log_src_var;

            let recon_var = calc_normalized_variance_4x4(
                p.recon,
                p.ref_off + i * p.ref_stride + j,
                p.ref_stride,
                p.bd,
            );
            avg_log_recon_variance += (f64::from(recon_var) / 16.0).ln_1p();
            j += MI_SIZE;
        }
        i += MI_SIZE;
    }

    let blocks = ((bw * bh) / 16) as f64;
    avg_log_src_variance /= blocks;
    avg_log_recon_variance /= blocks;

    // intra_rd_variance_factor tail.
    avg_log_src_variance += 0.000001;
    avg_log_recon_variance += 0.000001;

    if avg_log_src_variance >= avg_log_recon_variance {
        let var_diff = avg_log_src_variance - avg_log_recon_variance;
        if var_diff > 0.5 && avg_log_recon_variance < threshold {
            variance_rd_factor = 1.0 + ((var_diff * 2.0) / avg_log_src_variance);
        }
    } else {
        let var_diff = avg_log_recon_variance - avg_log_src_variance;
        if var_diff > 0.5 && avg_log_src_variance < threshold {
            variance_rd_factor = 1.0 + (var_diff / (2.0 * avg_log_src_variance));
        }
    }

    // AOMMIN(3.0, v).
    if 3.0 < variance_rd_factor {
        3.0
    } else {
        variance_rd_factor
    }
}

/// The mode loop's ALLINTRA application:
/// `this_rd = (int64_t)(this_rd * factor)` — i64 -> f64 conversion, one f64
/// multiply, truncation toward zero (every reachable rd is far inside the
/// exact/in-range regime).
#[inline]
pub fn apply_variance_factor(rd: i64, factor: f64) -> i64 {
    (rd as f64 * factor) as i64
}
