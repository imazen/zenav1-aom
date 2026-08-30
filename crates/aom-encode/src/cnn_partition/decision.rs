//! Port of the top-level `intra_mode_cnn_partition` decision
//! (`av1/encoder/partition_strategy.c`) — everything except the
//! partition-search side-effects: run the CNN on the 64×64 window, normalise
//! `log_q`, assemble the per-bsize DNN features, run the branch DNN, and turn
//! `logits[0]` vs the res-tier thresholds into the four prune flags.
//!
//! The CNN + DNN sub-engines are each already proven bit-exact against C
//! ([`super::cnn`], [`super::nn`]); this module adds the `log_q` term, the
//! feature assembly (`branch_*` spatial slicing via the `quad_to_linear` maps),
//! the threshold selection, and the decision — all diffed against
//! `av1/encoder/partition_strategy.c` via `aom_sys_ref::ref_intra_cnn_partition_decision`.
//!
//! Two entry points, and the difference between them is the whole of KB-PERF-1:
//! [`intra_mode_cnn_partition`] carries C's per-`BLOCK_64X64` cache
//! ([`PartitionSearchInfo`], C's `x->part_search_info`) and is what the search
//! calls; [`predict_decision`] is the uncached form the C differential gates.
//! Both share one per-node tail, so the cached path cannot drift from the
//! differentialled one.

use super::{cnn, nn, weights as w};
use aom_dsp::quant::av1_dc_quant_qtx;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

/// The four prune effects `intra_mode_cnn_partition` applies to the partition
/// search state. `none_disallowed` = `partition_none_allowed = 0` (only when
/// `logits[0] > split_thresh` AND `level != 1`); `do_square_split` +
/// `rect_disabled` = `av1_disable_rect_partitions` (when `logits[0] >
/// split_thresh`); `square_split_disabled` = `av1_disable_square_split_partition`
/// (when `logits[0] < no_split_thresh`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CnnPruneDecision {
    pub none_disallowed: bool,
    pub do_square_split: bool,
    pub rect_disabled: bool,
    pub square_split_disabled: bool,
}

impl CnnPruneDecision {
    /// True when the CNN constrains the search at all (any flag set).
    pub fn prunes(&self) -> bool {
        self.none_disallowed
            || self.do_square_split
            || self.rect_disabled
            || self.square_split_disabled
    }
}

/// `log_q` feature (partition_strategy.c:193-198): `dc_q =
/// av1_dc_quant_QTX(qindex,0,bd) >> (bd-8)`, then
/// `log_q = (log1pf(dc_q*dc_q / 256) - mean) / std`.
fn compute_log_q(qindex: i32, bd: i32) -> f32 {
    let dc_q = i32::from(av1_dc_quant_qtx(qindex, 0, bd as u8)) >> (bd - 8);
    // (float)(dc_q*dc_q) / 256.0f, then log1pf.
    let log_q = (f64_to_f32_div(dc_q * dc_q)).ln_1p();
    (log_q - w::MEAN[0]) / w::STD[0]
}

/// `(float)(dc_q*dc_q) / 256.0f` — the integer product widened to f32 then
/// divided by 256 in f32 (matching the C literal `256.0f`).
#[inline]
fn f64_to_f32_div(sq: i32) -> f32 {
    (sq as f32) / 256.0
}

/// Branch DNN weight bundle for `bsize_idx` (dnn_configs[bsize_idx]: 1→branch_0
/// .. 4→branch_3). Returns `(w0, b0, w1, b1, wlogits, blogits)`.
#[allow(clippy::type_complexity)]
fn branch_dnn(
    bsize_idx: i32,
) -> (
    &'static [f32],
    &'static [f32],
    &'static [f32],
    &'static [f32],
    &'static [f32],
    &'static [f32],
) {
    match bsize_idx {
        1 => (
            &w::BRANCH_0_DNN_LAYER_0_KERNEL,
            &w::BRANCH_0_DNN_LAYER_0_BIAS,
            &w::BRANCH_0_DNN_LAYER_1_KERNEL,
            &w::BRANCH_0_DNN_LAYER_1_BIAS,
            &w::BRANCH_0_LOGITS_KERNEL,
            &w::BRANCH_0_LOGITS_BIAS,
        ),
        2 => (
            &w::BRANCH_1_DNN_LAYER_0_KERNEL,
            &w::BRANCH_1_DNN_LAYER_0_BIAS,
            &w::BRANCH_1_DNN_LAYER_1_KERNEL,
            &w::BRANCH_1_DNN_LAYER_1_BIAS,
            &w::BRANCH_1_LOGITS_KERNEL,
            &w::BRANCH_1_LOGITS_BIAS,
        ),
        3 => (
            &w::BRANCH_2_DNN_LAYER_0_KERNEL,
            &w::BRANCH_2_DNN_LAYER_0_BIAS,
            &w::BRANCH_2_DNN_LAYER_1_KERNEL,
            &w::BRANCH_2_DNN_LAYER_1_BIAS,
            &w::BRANCH_2_LOGITS_KERNEL,
            &w::BRANCH_2_LOGITS_BIAS,
        ),
        4 => (
            &w::BRANCH_3_DNN_LAYER_0_KERNEL,
            &w::BRANCH_3_DNN_LAYER_0_BIAS,
            &w::BRANCH_3_DNN_LAYER_1_KERNEL,
            &w::BRANCH_3_DNN_LAYER_1_BIAS,
            &w::BRANCH_3_LOGITS_KERNEL,
            &w::BRANCH_3_LOGITS_BIAS,
        ),
        _ => unreachable!("intra-CNN bsize_idx in 1..=4"),
    }
}

/// Assemble the DNN input features from the CNN multi-out buffer for `bsize_idx`
/// / `quad_tree_idx`, appending `log_q` last. Returns the feature count (37 for
/// 64×64, 25 for 32×32, 25 for 16×16, 41 for 8×8). Verbatim transcription of the
/// per-bsize blocks in `intra_mode_cnn_partition` (branch spatial strides
/// 2×2 / 4×4 / 8×8, quad_to_linear spatial maps).
fn assemble_features(
    cnn_buffer: &[f32; cnn::CNN_OUT_BUF_SIZE],
    bsize_idx: i32,
    quad_tree_idx: i32,
    log_q: f32,
    out: &mut [f32; 100],
) -> usize {
    // Branch bases in the multi-out buffer (see cnn::branch_region).
    let branch_0 = &cnn_buffer[0..];
    let branch_1 = &cnn_buffer[20..];
    let branch_2 = &cnn_buffer[36..];
    let branch_3 = &cnn_buffer[356..];
    let mut f = 0usize;
    match bsize_idx {
        1 => {
            // BLOCK_64X64
            for ch in 0..20 {
                out[f] = branch_0[ch];
                f += 1;
            }
            let spa = 2 * 2;
            for lin in 0..spa {
                for ch in 0..4 {
                    out[f] = branch_1[lin + ch * spa];
                    f += 1;
                }
            }
        }
        2 => {
            // BLOCK_32X32
            for idx in 0..20 {
                out[f] = branch_0[idx];
                f += 1;
            }
            let cur_lin = w::QUAD_TO_LINEAR_1[(quad_tree_idx - 1) as usize] as usize;
            let spa = 2 * 2;
            for ch in 0..4 {
                out[f] = branch_1[cur_lin + ch * spa];
                f += 1;
            }
        }
        3 => {
            // BLOCK_16X16
            let prev_quad = (quad_tree_idx - 1) / 4;
            let prev_lin = w::QUAD_TO_LINEAR_1[(prev_quad - 1) as usize] as usize;
            let prev_spa = 2 * 2;
            for ch in 0..4 {
                out[f] = branch_1[prev_lin + ch * prev_spa];
                f += 1;
            }
            let cur_lin = w::QUAD_TO_LINEAR_2[(quad_tree_idx - 5) as usize] as usize;
            let spa = 4 * 4;
            for ch in 0..20 {
                out[f] = branch_2[cur_lin + ch * spa];
                f += 1;
            }
        }
        4 => {
            // BLOCK_8X8
            let prev_quad = (quad_tree_idx - 1) / 4;
            let prev_lin = w::QUAD_TO_LINEAR_2[(prev_quad - 5) as usize] as usize;
            let prev_spa = 4 * 4;
            for ch in 0..20 {
                out[f] = branch_2[prev_lin + ch * prev_spa];
                f += 1;
            }
            let cur_lin = w::QUAD_TO_LINEAR_3[(quad_tree_idx - 21) as usize] as usize;
            let spa = 8 * 8;
            for ch in 0..20 {
                out[f] = branch_3[cur_lin + ch * spa];
                f += 1;
            }
        }
        _ => unreachable!("intra-CNN bsize_idx in 1..=4"),
    }
    out[f] = log_q;
    f += 1;
    f
}

/// `intra_mode_cnn_partition`'s resolution-tier threshold select
/// (partition_strategy.c:311-329). `frame_w` / `frame_h` are `cm->width` /
/// `cm->height` — the TRUE CROP, never the mi-aligned extent (KB-28): the mi
/// grid rounds UP to 8 px, so a 474x480 crop reads as 480x480 there and takes
/// the midres tier where C takes lowres.
pub(crate) fn res_tier_thresholds(frame_w: i32, frame_h: i32, bsize_idx: usize) -> (f32, f32) {
    let mind = frame_w.min(frame_h);
    if mind >= 720 {
        (
            w::SPLIT_THRESH_HDRES[bsize_idx],
            w::NO_SPLIT_THRESH_HDRES[bsize_idx],
        )
    } else if mind >= 480 {
        (
            w::SPLIT_THRESH_MIDRES[bsize_idx],
            w::NO_SPLIT_THRESH_MIDRES[bsize_idx],
        )
    } else {
        (
            w::SPLIT_THRESH_LOWRES[bsize_idx],
            w::NO_SPLIT_THRESH_LOWRES[bsize_idx],
        )
    }
}

/// The intra-CNN half of C's `PartitionSearchInfo` (`av1/encoder/block.h:391-398`):
/// the cascade's multi-out buffer, its `log_q` companion, and the **validity
/// latch** that makes `intra_mode_cnn_partition` run the cascade ONCE per
/// `BLOCK_64X64` and merely READ it at every 32×32 / 16×16 / 8×8 node inside
/// that 64×64.
///
/// C's storage lives on the `MACROBLOCK` (`x->part_search_info`) and is
/// invalidated in exactly two places:
///
/// * per superblock — `encodeframe.c:692` (`x->part_search_info.cnn_output_valid = 0`);
/// * per `BLOCK_64X64` node — `init_partition_search_state_params`,
///   `partition_search.c:3339-3343`, the same two lines that re-anchor
///   `quad_tree_idx` (KB-24).
///
/// The port mirrors both: `pack_tile` builds a fresh one per superblock, and
/// [`rd_pick_partition_real`](crate::partition_pick::rd_pick_partition_real)
/// calls [`Self::invalidate_cnn`] at its `BLOCK_64X64` re-anchor.
///
/// The sibling field C keeps here, `quad_tree_idx`, stays a function parameter
/// in the port: C's SPLIT recursion advances it and then RESTORES it
/// (`partition_search.c:4571-4575` / `:4590-4592`), which a by-value parameter
/// expresses directly. Only the CNN cache genuinely needs to outlive a node.
#[derive(Clone)]
pub struct PartitionSearchInfo {
    /// `cnn_output_valid` (block.h:395).
    cnn_output_valid: bool,
    /// `cnn_buffer[CNN_OUT_BUF_SIZE]` (block.h:397).
    cnn_buffer: [f32; cnn::CNN_OUT_BUF_SIZE],
    /// `log_q` — "log of the quantization parameter of the ancestor
    /// BLOCK_64X64" (block.h:399). C computes it inside the same
    /// `bsize == BLOCK_64X64 && !cnn_output_valid` branch as the cascade
    /// (`partition_strategy.c:193-198`), so it is cached with it.
    log_q: f32,
}

impl Default for PartitionSearchInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl PartitionSearchInfo {
    /// A fresh (invalid) cache — C's `av1_zero(x->part_search_info)` state.
    pub fn new() -> Self {
        PartitionSearchInfo {
            cnn_output_valid: false,
            cnn_buffer: [0.0; cnn::CNN_OUT_BUF_SIZE],
            log_q: 0.0,
        }
    }

    /// `part_search_state->intra_part_info->cnn_output_valid = 0`
    /// (`partition_search.c:3342`, and the `must_find_valid_partition` repeat at
    /// `:5781`). Called at every `BLOCK_64X64` node on an intra frame.
    pub fn invalidate_cnn(&mut self) {
        self.cnn_output_valid = false;
    }
}

/// Opt-in self-check for the cache (KB-PERF-1): when enabled, EVERY node that
/// reads the cached cascade instead of computing it re-runs `cnn_predict` on a
/// freshly extracted window and asserts the result is bit-identical to what is
/// cached. Off by default; enabling it makes the encode roughly 2x slower
/// (every avoided cascade is run after all) and is for verification only.
static CNN_CACHE_VERIFY: AtomicBool = AtomicBool::new(false);
/// Cascades actually COMPUTED (the `bsize == BLOCK_64X64 && !valid` branch).
static CNN_CACHE_COMPUTES: AtomicU64 = AtomicU64::new(0);
/// Cache READS re-verified against a recomputation (only counted while
/// [`set_cnn_cache_verify`] is on).
static CNN_CACHE_READS_VERIFIED: AtomicU64 = AtomicU64::new(0);

/// Turn the cache self-check on/off (process-wide). See [`CNN_CACHE_VERIFY`].
pub fn set_cnn_cache_verify(on: bool) {
    CNN_CACHE_VERIFY.store(on, Relaxed);
}

/// `(cascades_computed, cache_reads_verified)` since the last
/// [`reset_cnn_cache_stats`]. `cascades_computed` is counted unconditionally
/// (one relaxed increment per 64×64); the second only while the check is on.
pub fn cnn_cache_stats() -> (u64, u64) {
    (
        CNN_CACHE_COMPUTES.load(Relaxed),
        CNN_CACHE_READS_VERIFIED.load(Relaxed),
    )
}

/// Zero both counters.
pub fn reset_cnn_cache_stats() {
    CNN_CACHE_COMPUTES.store(0, Relaxed);
    CNN_CACHE_READS_VERIFIED.store(0, Relaxed);
}

/// Port of `intra_mode_cnn_partition` (`partition_strategy.c:140-330`) **with
/// C's per-64×64 cache**, which is the whole point of this entry point:
///
/// * `:160` — the cascade runs only `if (bsize == BLOCK_64X64 &&
///   !part_info->cnn_output_valid)`, and stores into `part_info->cnn_buffer`
///   (`:224` sets the latch);
/// * `:227` — `if (!part_info->cnn_output_valid) return;` — a smaller node whose
///   ancestor 64×64 never computed (because it was not whole-in-frame, KB-23)
///   prunes nothing;
/// * `:230-330` — feature assembly + branch DNN + thresholds, per node.
///
/// `window` is called AT MOST ONCE, and only on the computing path, so the
/// caller's 65×65 extraction+copy is skipped entirely at the ~90 % of nodes that
/// hit the cache. Every node inside one 64×64 would produce the identical window
/// (`extract_intra_cnn_window` snaps its origin to the containing 64×64), which
/// is exactly why the cache cannot change a decision.
///
/// Returns `None` when C returns at `:227` (no valid cascade → no prune).
pub fn intra_mode_cnn_partition<F: Fn() -> Vec<u8>>(
    info: &mut PartitionSearchInfo,
    window: F,
    qindex: i32,
    bd: i32,
    frame_w: i32,
    frame_h: i32,
    bsize_idx: i32,
    quad_tree_idx: i32,
    level: i32,
) -> Option<([f32; 4], CnnPruneDecision)> {
    // partition_strategy.c:160-224 — compute once per 64x64, cache in part_info.
    let computed_here = if bsize_idx == 1 && !info.cnn_output_valid {
        // C computes log_q here too (:193-198) off the 64x64's own x->qindex.
        info.log_q = compute_log_q(qindex, bd);
        info.cnn_buffer = cnn::cnn_predict(&window());
        info.cnn_output_valid = true;
        CNN_CACHE_COMPUTES.fetch_add(1, Relaxed);
        true
    } else {
        false
    };

    // partition_strategy.c:227.
    if !info.cnn_output_valid {
        return None;
    }

    // Cache self-check (off by default): a READ must equal a recomputation.
    if !computed_here && CNN_CACHE_VERIFY.load(Relaxed) {
        let fresh = cnn::cnn_predict(&window());
        assert!(
            fresh == info.cnn_buffer,
            "intra-CNN cache read differs from a recomputation at bsize_idx {bsize_idx} \
             quad_tree_idx {quad_tree_idx}"
        );
        assert_eq!(
            compute_log_q(qindex, bd),
            info.log_q,
            "intra-CNN cached log_q differs from a recomputation"
        );
        CNN_CACHE_READS_VERIFIED.fetch_add(1, Relaxed);
    }

    let mut features = [0.0f32; 100];
    let nf = assemble_features(
        &info.cnn_buffer,
        bsize_idx,
        quad_tree_idx,
        info.log_q,
        &mut features,
    );
    Some(finish_decision(
        &features[..nf],
        frame_w,
        frame_h,
        bsize_idx,
        level,
    ))
}

/// Run the intra-CNN partition-prune decision for one block, **uncached** —
/// the C differential's entry point (`cnn_partition_decision_diff`), and the
/// definition of what [`intra_mode_cnn_partition`] must reproduce. `win` is the
/// parent 64×64's 65×65 luma window (replicated top/left border); `bsize_idx`
/// is `convert_bsize_to_idx` (1=64×64 .. 4=8×8); `quad_tree_idx` is the block's
/// position in the quad-tree; `level` is `intra_cnn_based_part_prune_level`
/// (1 or 2). Returns `(logits, decision)`.
pub fn predict_decision(
    win: &[u8],
    qindex: i32,
    bd: i32,
    frame_w: i32,
    frame_h: i32,
    bsize_idx: i32,
    quad_tree_idx: i32,
    level: i32,
) -> ([f32; 4], CnnPruneDecision) {
    let cnn_buffer = cnn::cnn_predict(win);
    let log_q = compute_log_q(qindex, bd);

    let mut features = [0.0f32; 100];
    let nf = assemble_features(&cnn_buffer, bsize_idx, quad_tree_idx, log_q, &mut features);
    finish_decision(&features[..nf], frame_w, frame_h, bsize_idx, level)
}

/// The per-node tail both entry points share: branch DNN
/// (`partition_strategy.c:330-341`) + res-tier thresholds (`:311-329`) + the
/// four prune flags (`:343-357`).
fn finish_decision(
    features: &[f32],
    frame_w: i32,
    frame_h: i32,
    bsize_idx: i32,
    level: i32,
) -> ([f32; 4], CnnPruneDecision) {
    let (w0, b0, w1, b1, wl, bl) = branch_dnn(bsize_idx);
    let mut logits = [0.0f32; 4];
    // num_outputs = BRANCH_*_NUM_LOGITS = 1; reduce_prec = 1 (as C calls it).
    //
    // **Deliberately the `_c` order, NOT [`nn::nn_predict_dispatched`]** —
    // KB-41 roots #26/#27. `av1_nn_predict` IS RTCD-specialized, so a real
    // encode runs its AVX2 order here (#26, ported + gated against the
    // dispatched C), but so is the CNN's own
    // `av1_cnn_convolve_no_maxpool_padding_valid`, whose `_c` variant this
    // module's `cnn::cnn_predict` transcribes (#27, NOT ported; the oracle is
    // pinned scalar by `shim/cnn_cscalar.c`). Pairing a SCALAR CNN with an
    // AVX2 DNN models neither chain: it stops matching the pinned scalar
    // oracle (`cnn_partition_decision_diff::predict_decision_matches_c`, which
    // is exactly what it broke) without matching the real dispatched one
    // either, because the branch features are already wrong upstream. The
    // switch to `nn_predict_dispatched` lands WITH root #27, not before.
    nn::nn_predict(
        features,
        &[16, 24],
        &[w0, w1, wl],
        &[b0, b1, bl],
        1,
        true,
        &mut logits,
    );

    // Res-tier thresholds (partition_strategy.c:311-329).
    let (split_thresh, no_split_thresh) = res_tier_thresholds(frame_w, frame_h, bsize_idx as usize);

    let mut d = CnnPruneDecision::default();
    if logits[0] > split_thresh {
        if level != 1 {
            d.none_disallowed = true;
        }
        d.do_square_split = true;
        d.rect_disabled = true;
    }
    if logits[0] < no_split_thresh {
        d.square_split_disabled = true;
    }
    (logits, d)
}
