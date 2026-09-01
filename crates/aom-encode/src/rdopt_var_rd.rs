//! The VARIANCE-BASED RD ADJUSTMENT arm of libaom's inter RD brain
//! (`av1/encoder/rdopt.c:624-866`), plus two more search-loop predicates.
//!
//! Tier 1c (all `static`; the oracle is libaom's own rdopt.c compiled into the
//! shim archive). Gate: `crates/aom-encode/tests/rdopt_var_rd_diff.rs`.
//!
//! | Rust | C (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | [`get_variance_stats`] | `get_variance_stats` `:709` + `_hbd` `:624` |
//! | [`adjust_cost`] | `:840` |
//! | [`adjust_rdcost`] | `:796` |
//! | [`inter_mode_compatible_skip`] | `:4581` |
//! | [`ref_mv_idx_early_breakout`] | `:2216` |
//!
//! # What the variance adjustment does
//!
//! At `--sharpness=3` on a non-key, non-GF/ARF frame, libaom measures how much
//! high-frequency energy a 3x3 Gaussian removes from the SOURCE versus from the
//! RECONSTRUCTION. If the source has more, the block's prediction has smoothed
//! detail away, and the difference is added to the RD distortion so a
//! detail-preserving candidate wins. The `AOM_TUNE_IQ` /
//! `AOM_TUNE_SSIMULACRA2` arm is unrelated and simpler: a flat 1.125x bias
//! against inter blocks.

use crate::rd::rdcost;
use crate::rdopt_mv::{MAX_REF_MV_SEARCH, PredMode, REF_FRAMES, RefMvRow, get_drl_cost};
use crate::rdopt_obmc::{BLOCK_SIZE_HIGH, BLOCK_SIZE_WIDE};

/// `MAX_SB_SIZE`.
pub const MAX_SB_SIZE: usize = 128;

/// The 3x3 Gaussian `gau_filter` (rdopt.c:626), sum 16.
const GAU_FILTER: [[i32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];

/// `get_variance_stats` (rdopt.c:709) and `get_variance_stats_hbd` (`:624`) —
/// one function, because the two C copies differ only in the sample type and
/// this port carries both depths as `u16`.
///
/// Returns `(src_var, rec_var)`: the summed squared high-frequency residual of
/// the source and of the reconstruction, each `<< 4`.
///
/// # The scratch buffer's stride is `bw`, NOT `bw + 2`, and that is deliberate
///
/// C copies a 1-pixel replicated border into `dclevel` and indexes it as
/// `pred_ptr[idy * bw + idx]` for `idy, idx` in `-1 ..= bw`. With a row stride
/// of `bw` and a column range of `bw + 2`, the rows OVERLAP: the left halo of
/// row `y` is the same storage as the right halo of row `y - 1`. The copy loop
/// runs in increasing `(idy, idx)`, so the later write wins, and the 3x3
/// filter then reads whatever that aliasing left behind.
///
/// That is not a transcription artefact to clean up — it is what the function
/// computes, and a "corrected" `bw + 2` stride gives different `src_var` and
/// `rec_var` on every block with a non-trivial border. Reproduced exactly;
/// the differential fails if the stride is widened.
pub fn get_variance_stats(
    bsize: usize,
    src: &[u16],
    src_stride: usize,
    dst: &[u16],
    dst_stride: usize,
    is_hbd: bool,
) -> (i64, i64) {
    let bw = BLOCK_SIZE_WIDE[bsize] as usize;
    let bh = BLOCK_SIZE_HIGH[bsize] as usize;
    // C's `dclevel` is (MAX_SB_SIZE + 2)^2 with `pred_ptr = &dclevel[bw + 1]`.
    let mut scratch = vec![0u16; (MAX_SB_SIZE + 2) * (MAX_SB_SIZE + 2)];
    let base = bw + 1;

    let mut pass = |plane: &[u16], stride: usize| -> i64 {
        for idy in -1i64..=bh as i64 {
            for idx in -1i64..=bw as i64 {
                let oy = idy.clamp(0, bh as i64 - 1) as usize;
                let ox = idx.clamp(0, bw as i64 - 1) as usize;
                let v = plane[oy * stride + ox];
                let at = (base as i64 + idy * bw as i64 + idx) as usize;
                // C's lowbd scratch is `uint8_t`, so its copy would truncate —
                // but a lowbd sample never exceeds 255, so the truncation is
                // unreachable. Asserting the contract beats carrying a mask
                // that no input can exercise (and that a differential
                // therefore cannot check).
                debug_assert!(is_hbd || v < 256, "a lowbd sample must fit in 8 bits");
                scratch[at] = v;
            }
        }
        let mut var = 0i64;
        for idy in 0..bh {
            for idx in 0..bw {
                let mut sum = 0i32;
                for (iy, frow) in GAU_FILTER.iter().enumerate() {
                    for (ix, &f) in frow.iter().enumerate() {
                        // The offsets go NEGATIVE at the block's first row and
                        // column; the arithmetic is done signed and only the
                        // final index is a usize, exactly as C's pointer
                        // arithmetic does it.
                        let at = base as i64
                            + (idy as i64 + iy as i64 - 1) * bw as i64
                            + (idx as i64 + ix as i64 - 1);
                        sum += i32::from(scratch[at as usize]) * f;
                    }
                }
                sum >>= 4;
                let diff = i64::from(i32::from(scratch[base + idy * bw + idx]) - sum);
                var += diff * diff;
            }
        }
        var << 4
    };

    // C runs the RECONSTRUCTION first and the SOURCE second, both through the
    // same scratch buffer. The order is invisible in the result (each pass
    // fully rewrites the region it reads) but is kept for readability.
    let rec_var = pass(dst, dst_stride);
    let src_var = pass(src, src_stride);
    (src_var, rec_var)
}

/// `aom_tune_metric` values the adjustment gates on.
pub const AOM_TUNE_IQ: i32 = 10;
/// `AOM_TUNE_SSIMULACRA2`.
pub const AOM_TUNE_SSIMULACRA2: i32 = 11;

/// The three encoder-level gates `adjust_cost` / `adjust_rdcost` read.
#[derive(Clone, Copy, Debug)]
pub struct AdjustGates {
    /// `cpi->oxcf.tune_cfg.tuning`.
    pub tuning: i32,
    /// `cpi->oxcf.algo_cfg.sharpness`.
    pub sharpness: i32,
    /// `frame_is_kf_gf_arf(cpi)` — intra-only, ARF_UPDATE or GF_UPDATE.
    pub frame_is_kf_gf_arf: bool,
    /// `x->rdmult`.
    pub rdmult: i32,
}

/// `adjust_cost` (rdopt.c:840): bias a scalar RD cost.
///
/// Returns the adjusted cost. The IQ / SSIMULACRA2 arm returns EARLY, so the
/// sharpness arm is unreachable when either tuning is selected.
#[allow(clippy::too_many_arguments)]
pub fn adjust_cost(
    rd_cost: i64,
    is_inter_pred: bool,
    gates: AdjustGates,
    bsize: usize,
    src: &[u16],
    src_stride: usize,
    dst: &[u16],
    dst_stride: usize,
    is_hbd: bool,
) -> i64 {
    if (gates.tuning == AOM_TUNE_IQ || gates.tuning == AOM_TUNE_SSIMULACRA2) && is_inter_pred {
        return rd_cost + (rd_cost >> 3);
    }
    if gates.sharpness != 3 || gates.frame_is_kf_gf_arf {
        return rd_cost;
    }
    let (src_var, rec_var) = get_variance_stats(bsize, src, src_stride, dst, dst_stride, is_hbd);
    if src_var <= rec_var {
        return rd_cost;
    }
    rd_cost + rdcost(gates.rdmult, 0, src_var - rec_var)
}

/// The three `RD_STATS` fields `adjust_rdcost` touches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdStatsCore {
    /// `rate`.
    pub rate: i32,
    /// `dist`.
    pub dist: i64,
    /// `rdcost`.
    pub rdcost: i64,
}

/// `adjust_rdcost` (rdopt.c:796): the [`adjust_cost`] logic applied to a full
/// `RD_STATS`.
///
/// The two arms differ in more than scale: the IQ / SSIMULACRA2 arm scales
/// `dist` AND `rdcost` by 1.125 independently, while the sharpness arm adds
/// the variance offset to `dist` and then RECOMPUTES `rdcost` from `rate` and
/// the new `dist`. Collapsing them loses that.
#[allow(clippy::too_many_arguments)]
pub fn adjust_rdcost(
    rd: &mut RdStatsCore,
    is_inter_pred: bool,
    gates: AdjustGates,
    bsize: usize,
    src: &[u16],
    src_stride: usize,
    dst: &[u16],
    dst_stride: usize,
    is_hbd: bool,
) {
    if (gates.tuning == AOM_TUNE_IQ || gates.tuning == AOM_TUNE_SSIMULACRA2) && is_inter_pred {
        rd.dist += rd.dist >> 3;
        rd.rdcost += rd.rdcost >> 3;
        return;
    }
    if gates.sharpness != 3 || gates.frame_is_kf_gf_arf {
        return;
    }
    let (src_var, rec_var) = get_variance_stats(bsize, src, src_stride, dst, dst_stride, is_hbd);
    if src_var <= rec_var {
        return;
    }
    rd.dist += src_var - rec_var;
    rd.rdcost = rdcost(gates.rdmult, rd.rate, rd.dist);
}

/// `REFERENCE_MODE` (`enums.h`).
pub const SINGLE_REFERENCE: i32 = 0;

/// `is_comp_ref_allowed` (`blockd.h:65`).
pub fn is_comp_ref_allowed(bsize: usize) -> bool {
    BLOCK_SIZE_WIDE[bsize].min(BLOCK_SIZE_HIGH[bsize]) >= 8
}

/// `is_interintra_allowed_bsize` (`blockd.h:1418`).
pub fn is_interintra_allowed_bsize(bsize: usize) -> bool {
    // BLOCK_8X8 (3) ..= BLOCK_32X32 (9).
    (3..=9).contains(&bsize)
}

/// `av1_ref_frame_flag_list[ref]` (`blockd.h`): the availability bit for a
/// reference frame. `LAST_FRAME` is bit 0, so the shift is `ref - 1`.
pub fn ref_frame_flag(ref_frame: i32) -> i32 {
    if ref_frame <= 0 {
        return 0;
    }
    1 << (ref_frame - 1)
}

/// `inter_mode_compatible_skip` (rdopt.c:4581): is this `(mode, reference
/// pair)` combination codeable at all?
///
/// Three families of veto: compound needs a big enough block, an available
/// second reference, an inter frame, a non-`SINGLE_REFERENCE` frame header and
/// no segment-level reference override; interintra (`ref_frames[1] ==
/// INTRA_FRAME`) needs an allowed block size and a single-reference mode.
#[allow(clippy::too_many_arguments)]
pub fn inter_mode_compatible_skip(
    bsize: usize,
    curr_mode: PredMode,
    ref_frames: [i32; 2],
    ref_frame_flags: i32,
    frame_is_intra_only: bool,
    reference_mode: i32,
    seg_ref_frame_active: bool,
) -> bool {
    let comp_pred = ref_frames[1] > 0;
    if comp_pred {
        if !is_comp_ref_allowed(bsize) {
            return true;
        }
        if ref_frame_flags & ref_frame_flag(ref_frames[1]) == 0 {
            return true;
        }
        if frame_is_intra_only {
            return true;
        }
        if reference_mode == SINGLE_REFERENCE {
            return true;
        }
        // With a segment-level reference there can only be ONE reference.
        if seg_ref_frame_active {
            return true;
        }
    }
    if ref_frames[0] > 0 && ref_frames[1] == 0 {
        // Interintra.
        if !is_interintra_allowed_bsize(bsize) {
            return true;
        }
        if !curr_mode.is_inter_singleref() {
            return true;
        }
    }
    false
}

/// `REF_CAT_LEVEL`.
const REF_CAT_LEVEL: u16 = 640;

/// The distance information `ref_mv_idx_early_breakout` prunes on.
#[derive(Clone, Copy, Debug)]
pub struct RefFrameDistanceInfo {
    /// `nearest_past_ref`.
    pub nearest_past_ref: i32,
    /// `nearest_future_ref`.
    pub nearest_future_ref: i32,
}

/// `ref_mv_idx_early_breakout` (rdopt.c:2216): should this DRL index be
/// skipped before any search runs?
///
/// **This function has a SIDE EFFECT**: partway through it writes
/// `mbmi->ref_mv_idx = ref_mv_idx`, and the caller relies on that even when
/// the answer is "break out". The port therefore returns the new index
/// alongside the verdict rather than hiding it: `(should_break, ref_mv_idx)`,
/// where the index is unchanged only on the two early-return paths that
/// precede the assignment.
#[allow(clippy::too_many_arguments)]
pub fn ref_mv_idx_early_breakout(
    reduce_inter_modes: i32,
    dist: RefFrameDistanceInfo,
    mode: PredMode,
    ref_frames: [i32; 2],
    ref_mv_idx: usize,
    qindex: i32,
    rdmult: i32,
    ref_best_rd: i64,
    row: &RefMvRow,
    drl_mode_cost0: &[[i32; 2]; 3],
    ref_frame_cost: i32,
    single_comp_cost: i32,
    single_newmv_valid: &[[bool; REF_FRAMES]; MAX_REF_MV_SEARCH],
    prev_ref_mv_idx: usize,
) -> (bool, usize) {
    let is_comp_pred = ref_frames[1] > 0;
    let has_nearmv = usize::from(mode.have_nearmv());
    if reduce_inter_modes != 0 && ref_mv_idx > 0 {
        // LAST2 / LAST3 are the references this prunes hardest on.
        if ref_frames.iter().any(|&r| r == 2 || r == 3)
            && row.weight[ref_mv_idx + has_nearmv] < REF_CAT_LEVEL
        {
            return (true, prev_ref_mv_idx);
        }
        if reduce_inter_modes >= 2 && !is_comp_pred && mode.have_newmv() {
            let closest =
                ref_frames[0] == dist.nearest_past_ref || ref_frames[0] == dist.nearest_future_ref;
            if !closest {
                let do_prune = crate::rdopt_mv::prune_ref_mv_idx_using_qindex(
                    reduce_inter_modes,
                    qindex,
                    ref_mv_idx as i32,
                );
                if do_prune && row.weight[ref_mv_idx + has_nearmv] < REF_CAT_LEVEL {
                    return (true, prev_ref_mv_idx);
                }
            }
        }
    }

    // From here on the index IS committed, whatever the verdict.
    if is_comp_pred
        && !crate::rdopt_mv::is_single_newmv_valid(
            mode,
            ref_frames,
            &single_newmv_valid[ref_mv_idx],
        )
    {
        return (true, ref_mv_idx);
    }

    // C accumulates this in a `size_t`, then compares through RDCOST's int64
    // arithmetic; the values are small rates, so i64 is exact here.
    let est_rd_rate = i64::from(ref_frame_cost)
        + i64::from(single_comp_cost)
        + i64::from(get_drl_cost(mode, ref_mv_idx, row, drl_mode_cost0));
    let over_budget = rdcost(rdmult, est_rd_rate as i32, 0) > ref_best_rd;
    // NEARESTMV and NEAREST_NEARESTMV are never dropped on rate alone — they
    // are the fallback the search needs to always have.
    if over_budget && !matches!(mode, PredMode::NearestMv | PredMode::NearestNearestMv) {
        return (true, ref_mv_idx);
    }
    (false, ref_mv_idx)
}
