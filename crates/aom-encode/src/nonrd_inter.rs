//! The mode-skip and prune cascade of `av1/encoder/nonrd_pickmode.c` — the
//! decisions the speed 8/9 INTER pickmode makes before it spends any work on a
//! candidate.
//!
//! `nonrd_pickmode.rs` holds the intra (KEY) arm of the same file. This module
//! is the inter arm's gate layer: which `(mode, ref_frame)` pairs the search
//! even looks at, and which compound candidates it drops on the strength of
//! the single-reference results it already has.
//!
//! | Rust | C |
//! |---|---|
//! | [`MODE_IDX`] | `mode_idx` (nonrd_opt.h:127) |
//! | [`skip_mode_by_threshold`] | `skip_mode_by_threshold` (:1933) |
//! | [`skip_mode_by_low_temp`] | `skip_mode_by_low_temp` (:1961) |
//! | [`skip_mode_by_bsize_and_ref_frame`] | `skip_mode_by_bsize_and_ref_frame` (:1978) |
//! | [`skip_comp_based_on_var`] | `skip_comp_based_on_var` (:2165) |
//! | [`previous_mode_performed_poorly`] | `previous_mode_performed_poorly` (:2286) |
//! | [`prune_compoundmode_with_singlemode_var`] | `prune_compoundmode_with_singlemode_var` (:2306) |
//! | [`ac_thr_factor`] | `ac_thr_factor` (:580) |
//! | [`calculate_variance`] | `calculate_variance` (:556) |
//! | [`rd_less_than_thresh`] | `rd_less_than_thresh` (`rd.h:297`) |
//!
//! # Differential coverage
//! `tests/nonrd_inter_diff.rs` — tier 1c against libaom's own
//! nonrd_pickmode.c, compiled verbatim by `shim/nonrd_pick_shim.c`. Only two
//! symbols in that file are exported, so tier 1 is not available for any of it.

use crate::rd_thresh::{
    THR_DC, THR_GLOBALA, THR_GLOBALA2, THR_GLOBALB, THR_GLOBALG, THR_GLOBALL2, THR_GLOBALL3,
    THR_GLOBALMV, THR_H_PRED, THR_NEARA, THR_NEARA2, THR_NEARB, THR_NEARESTA, THR_NEARESTA2,
    THR_NEARESTB, THR_NEARESTG, THR_NEARESTL2, THR_NEARESTL3, THR_NEARESTMV, THR_NEARG, THR_NEARL2,
    THR_NEARL3, THR_NEARMV, THR_NEWA, THR_NEWA2, THR_NEWB, THR_NEWG, THR_NEWL2, THR_NEWL3,
    THR_NEWMV, THR_SMOOTH, THR_V_PRED,
};
use crate::rdopt_mv::{PredMode, compound_ref0_mode, compound_ref1_mode};
use crate::rdopt_single_state::inter_offset;
use crate::var_part::SourceSad;

/// `RTC_MODES` (nonrd_opt.h:18) — the four modes the RT search evaluates per
/// reference: NEARESTMV, NEARMV, GLOBALMV, NEWMV (or the four intra modes for
/// the intra row).
pub const RTC_MODES: usize = 4;
/// `RTC_INTER_MODES` (nonrd_opt.h:19).
pub const RTC_INTER_MODES: usize = 4;
/// `REF_FRAMES` (`enums.h`) — INTRA_FRAME plus the seven inter references.
pub const REF_FRAMES: usize = 8;
/// `LAST_FRAME`.
pub const LAST_FRAME: usize = 1;
/// `GOLDEN_FRAME`.
pub const GOLDEN_FRAME: usize = 4;

/// `mode_idx[REF_FRAMES][RTC_MODES]` (nonrd_opt.h:127) — the `THR_MODES` slot
/// each `(ref_frame, RTC mode)` pair reads its RD threshold from.
///
/// Row 0 is INTRA_FRAME and holds the four INTRA modes, so the table is not
/// uniform in what its columns mean; only rows 1..8 are inter.
pub const MODE_IDX: [[usize; RTC_MODES]; REF_FRAMES] = [
    [THR_DC, THR_V_PRED, THR_H_PRED, THR_SMOOTH],
    [THR_NEARESTMV, THR_NEARMV, THR_GLOBALMV, THR_NEWMV],
    [THR_NEARESTL2, THR_NEARL2, THR_GLOBALL2, THR_NEWL2],
    [THR_NEARESTL3, THR_NEARL3, THR_GLOBALL3, THR_NEWL3],
    [THR_NEARESTG, THR_NEARG, THR_GLOBALG, THR_NEWG],
    [THR_NEARESTB, THR_NEARB, THR_GLOBALB, THR_NEWB],
    [THR_NEARESTA2, THR_NEARA2, THR_GLOBALA2, THR_NEWA2],
    [THR_NEARESTA, THR_NEARA, THR_GLOBALA, THR_NEWA],
];

/// `rd_less_than_thresh` (`av1/encoder/rd.h:297`).
///
/// The `thresh == INT_MAX` arm makes an unset threshold ALWAYS skip, which is
/// the opposite of what the arithmetic alone would give (a huge threshold
/// makes `best_rd < thresh * fact >> 5` true anyway, but the shift can
/// overflow first). Reproduced as written.
#[inline]
#[must_use]
pub fn rd_less_than_thresh(best_rd: i64, thresh: i64, thresh_fact: i32) -> bool {
    best_rd < ((thresh * i64::from(thresh_fact)) >> 5) || thresh == i64::from(i32::MAX)
}

/// `skip_mode_by_threshold` (nonrd_pickmode.c:1933).
///
/// The RD-threshold gate: a mode whose threshold the current best cost is
/// already under is dropped — but ONLY if its MV is non-zero. C tests
/// `mv.as_int != 0` on the PACKED union, which is why the MV crosses this
/// boundary packed rather than as a row/col pair.
///
/// `extra_shift` is applied twice for GOLDEN when the golden frame is stale
/// (`frames_since_golden > 4`), on top of the unconditional non-LAST doubling.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn skip_mode_by_threshold(
    mode: PredMode,
    ref_frame: usize,
    mv_as_int: u32,
    frames_since_golden: i32,
    rd_threshes: &[i32],
    rd_thresh_freq_fact: &[i32],
    best_cost: i64,
    best_skip: bool,
    extra_shift: u32,
) -> bool {
    let mode_index = MODE_IDX[ref_frame][inter_offset(mode)];
    let base = i64::from(rd_threshes[mode_index]);
    let mut mode_rd_thresh = if best_skip {
        base << (extra_shift + 1)
    } else {
        base << extra_shift
    };

    // Increase mode_rd_thresh for non-LAST, for encoding speed.
    if ref_frame != LAST_FRAME {
        mode_rd_thresh <<= 1;
        if ref_frame == GOLDEN_FRAME && frames_since_golden > 4 {
            mode_rd_thresh <<= extra_shift + 1;
        }
    }

    rd_less_than_thresh(best_cost, mode_rd_thresh, rd_thresh_freq_fact[mode_index])
        && mv_as_int != 0
}

/// `skip_mode_by_low_temp` (nonrd_pickmode.c:1961).
///
/// When the superblock's temporal variance is low, non-LAST references are
/// only worth a zero MV, and a 64x64-or-larger NEWMV is not worth searching at
/// all unless the source SAD is high.
#[must_use]
pub fn skip_mode_by_low_temp(
    mode: PredMode,
    ref_frame: usize,
    bsize: usize,
    source_sad_nonrd: SourceSad,
    mv_as_int: u32,
    force_skip_low_temp_var: bool,
) -> bool {
    /// `BLOCK_64X64` (`enums.h:112`).
    const BLOCK_64X64: usize = 12;
    if force_skip_low_temp_var && ref_frame != LAST_FRAME && mv_as_int != 0 {
        return true;
    }
    source_sad_nonrd != SourceSad::High
        && bsize >= BLOCK_64X64
        && force_skip_low_temp_var
        && mode == PredMode::NewMv
}

/// `skip_mode_by_bsize_and_ref_frame` (nonrd_pickmode.c:1978).
///
/// `extra_prune` is a LEVEL, not a flag: level 1 drops non-LAST NEARMV, and
/// level 2+ additionally drops non-LAST NEWMV above 16x16.
#[must_use]
pub fn skip_mode_by_bsize_and_ref_frame(
    mode: PredMode,
    ref_frame: usize,
    bsize: usize,
    extra_prune: i32,
    sse_zeromv_norm: u32,
    more_prune: bool,
    skip_nearmv: bool,
) -> bool {
    /// `BLOCK_16X16`.
    const BLOCK_16X16: usize = 6;
    /// `BLOCK_32X32`.
    const BLOCK_32X32: usize = 9;
    /// `BLOCK_128X128`.
    const BLOCK_128X128: usize = 15;
    /// C's local `thresh_skip_golden`.
    const THRESH_SKIP_GOLDEN: u32 = 500;

    if ref_frame != LAST_FRAME && sse_zeromv_norm < THRESH_SKIP_GOLDEN && mode == PredMode::NewMv {
        return true;
    }
    if (bsize == BLOCK_128X128 && mode == PredMode::NewMv)
        || (skip_nearmv && mode == PredMode::NearMv)
    {
        return true;
    }
    if extra_prune != 0 {
        if extra_prune > 1
            && ref_frame != LAST_FRAME
            && bsize > BLOCK_16X16
            && mode == PredMode::NewMv
        {
            return true;
        }
        if ref_frame != LAST_FRAME && mode == PredMode::NearMv {
            return true;
        }
        if more_prune && bsize >= BLOCK_32X32 && mode == PredMode::NearMv {
            return true;
        }
    }
    false
}

/// `skip_comp_based_on_var` (nonrd_pickmode.c:2165) — drop compound modes
/// outright when the best single-reference variance is already small.
///
/// The two thresholds are C's own float-to-int truncations of
/// `0.57356805f * 8659` and `0.23964763f * 4281`; they are computed here the
/// same way rather than being written as the integers, so the derivation stays
/// visible and the f32 rounding is the compiler's, not a transcriber's.
/// C's comment notes the 128 and 16 cases are extrapolated, not tuned.
#[must_use]
pub fn skip_comp_based_on_var(
    single_vars: &[[u32; REF_FRAMES]; RTC_INTER_MODES],
    bsize: usize,
) -> bool {
    /// `BLOCK_16X16`.
    const BLOCK_16X16: usize = 6;
    /// `BLOCK_32X32`.
    const BLOCK_32X32: usize = 9;
    /// `BLOCK_64X64`.
    const BLOCK_64X64: usize = 12;
    /// `BLOCK_128X128`.
    const BLOCK_128X128: usize = 15;

    let best_var = single_vars
        .iter()
        .flatten()
        .copied()
        .min()
        .unwrap_or(u32::MAX);
    let thresh_64 = (0.573_568_05f32 * 8659.0) as u32;
    let thresh_32 = (0.239_647_63f32 * 4281.0) as u32;
    match bsize {
        BLOCK_128X128 => best_var < 4 * thresh_64,
        BLOCK_64X64 => best_var < thresh_64,
        BLOCK_32X32 => best_var < thresh_32,
        BLOCK_16X16 => best_var < thresh_32 / 4,
        _ => false,
    }
}

/// `previous_mode_performed_poorly` (nonrd_pickmode.c:2286).
///
/// "Poorly" is `1.125 * best < this`, in FLOAT — `mult` is an `f32` and the
/// comparison promotes the `unsigned int` / `int64_t` operand to `float`, so
/// large values lose precision exactly as C's do. The port keeps the f32.
///
/// The chroma term is ANDed in only when this mode's `uv_dist` is both finite
/// and not itself the best: C's `best_uv_dist != uv_dist[...]` guard means a
/// mode that ties the best chroma distortion is judged on luma alone.
#[must_use]
pub fn previous_mode_performed_poorly(
    mode: PredMode,
    ref_frame: usize,
    vars: &[[u32; REF_FRAMES]; RTC_INTER_MODES],
    uv_dist: &[[i64; REF_FRAMES]; RTC_INTER_MODES],
) -> bool {
    let best_var = (0..RTC_INTER_MODES)
        .map(|m| vars[m][ref_frame])
        .min()
        .unwrap_or(u32::MAX);
    let best_uv_dist = (0..RTC_INTER_MODES)
        .map(|m| uv_dist[m][ref_frame])
        .min()
        .unwrap_or(i64::MAX);
    debug_assert_ne!(best_var, u32::MAX, "invalid variance data");

    let mult = 1.125f32;
    let off = inter_offset(mode);
    let this_var = vars[off][ref_frame];
    let mut var_bad = mult * (best_var as f32) < this_var as f32;

    let this_uv = uv_dist[off][ref_frame];
    if this_uv < i64::MAX && best_uv_dist != this_uv {
        var_bad &= mult * (best_uv_dist as f32) < this_uv as f32;
    }
    var_bad
}

/// `prune_compoundmode_with_singlemode_var` (nonrd_pickmode.c:2306).
///
/// A compound mode is dropped when the single-reference modes it decomposes
/// into both did poorly — or, when only one of the two is available, when that
/// one did.
///
/// `frame_mv` and `mode_checked` are indexed by the RAW `PREDICTION_MODE`
/// (`MB_MODE_COUNT` rows), while `vars` and `uv_dist` are indexed by
/// `INTER_OFFSET` (`RTC_INTER_MODES` rows). Mixing the two indexings is the
/// obvious transcription error here, so they are separate parameters with
/// separate lengths.
#[must_use]
pub fn prune_compoundmode_with_singlemode_var(
    compound_mode: PredMode,
    ref_frame: usize,
    ref_frame2: usize,
    frame_mv: &[[u32; REF_FRAMES]],
    mode_checked: &[[u8; REF_FRAMES]],
    vars: &[[u32; REF_FRAMES]; RTC_INTER_MODES],
    uv_dist: &[[i64; REF_FRAMES]; RTC_INTER_MODES],
) -> bool {
    let cm = compound_mode.to_i32() as usize;
    let mut first_valid = false;
    let mut second_valid = false;
    let mut first_bad = false;
    let mut second_bad = false;

    if let Some(single0) = compound_ref0_mode(compound_mode) {
        let s0 = single0.to_i32() as usize;
        if mode_checked[s0][ref_frame] != 0
            && frame_mv[s0][ref_frame] == frame_mv[cm][ref_frame]
            && vars[inter_offset(single0)][ref_frame] < u32::MAX
        {
            first_valid = true;
            first_bad = previous_mode_performed_poorly(single0, ref_frame, vars, uv_dist);
        }
    }
    if let Some(single1) = compound_ref1_mode(compound_mode) {
        let s1 = single1.to_i32() as usize;
        if mode_checked[s1][ref_frame2] != 0
            && frame_mv[s1][ref_frame2] == frame_mv[cm][ref_frame2]
            && vars[inter_offset(single1)][ref_frame2] < u32::MAX
        {
            second_valid = true;
            second_bad = previous_mode_performed_poorly(single1, ref_frame2, vars, uv_dist);
        }
    }

    if first_valid && second_valid {
        first_bad && second_bad
    } else if first_valid || second_valid {
        first_bad || second_bad
    } else {
        false
    }
}

/// `ac_thr_factor` (nonrd_pickmode.c:580) — the AC early-term threshold
/// multiplier, raised on small, nearly-static blocks at speed 8+.
#[must_use]
pub fn ac_thr_factor(speed: i32, width: i32, height: i32, norm_sum: i32) -> i64 {
    if speed >= 8 && norm_sum < 5 {
        if width <= 640 && height <= 480 { 4 } else { 2 }
    } else {
        1
    }
}

/// `calculate_variance` (nonrd_pickmode.c:556) — fold a grid of per-unit
/// `(sse, sum)` records up one level, 2x2 at a time, and derive each parent's
/// variance.
///
/// `bw` / `bh` are `b_width_log2` / `b_height_log2` of the BLOCK (not pixel
/// counts), and `unit_log2` is `b_width_log2_lookup[txsize_to_bsize[tx_size]]`
/// — square, so the width and height logs are equal and C adds them.
///
/// Returns `(var_o, sse_o, sum_o)`, each `(nw / 2) * (nh / 2)` entries in
/// row-major order.
///
/// The variance derivation subtracts a `uint32_t` cast of an `int64_t` shift,
/// so it WRAPS on the (unreachable) negative case rather than saturating; the
/// port matches the width.
#[must_use]
pub fn calculate_variance(
    bw: u32,
    bh: u32,
    unit_log2: u32,
    sse_i: &[u32],
    sum_i: &[i32],
) -> (Vec<u32>, Vec<u32>, Vec<i32>) {
    let nw = 1usize << (bw - unit_log2);
    let nh = 1usize << (bh - unit_log2);
    let mut var_o = Vec::with_capacity((nw / 2) * (nh / 2));
    let mut sse_o = Vec::with_capacity((nw / 2) * (nh / 2));
    let mut sum_o = Vec::with_capacity((nw / 2) * (nh / 2));
    let shift = unit_log2 + unit_log2 + 6;
    let mut row = 0usize;
    while row < nh {
        let mut col = 0usize;
        while col < nw {
            let sse = sse_i[row * nw + col]
                .wrapping_add(sse_i[row * nw + col + 1])
                .wrapping_add(sse_i[(row + 1) * nw + col])
                .wrapping_add(sse_i[(row + 1) * nw + col + 1]);
            let sum = sum_i[row * nw + col]
                .wrapping_add(sum_i[row * nw + col + 1])
                .wrapping_add(sum_i[(row + 1) * nw + col])
                .wrapping_add(sum_i[(row + 1) * nw + col + 1]);
            let mean_sq = ((i64::from(sum) * i64::from(sum)) >> shift) as u32;
            var_o.push(sse.wrapping_sub(mean_sq));
            sse_o.push(sse);
            sum_o.push(sum);
            col += 2;
        }
        row += 2;
    }
    (var_o, sse_o, sum_o)
}
