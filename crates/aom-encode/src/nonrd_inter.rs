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

// ===========================================================================
// The tx-size / subpel-precision / MV-bias cluster of nonrd_pickmode.c.
//
// | Rust | C |
// |---|---|
// | [`subpel_select`] | `subpel_select` (:99) |
// | [`use_aggressive_subpel_search_method`] | `use_aggressive_subpel_search_method` (:155) |
// | [`set_force_skip_flag`] | `set_force_skip_flag` (:423) |
// | [`calculate_tx_size`] | `calculate_tx_size` (:447) |
// | [`newmv_diff_bias`] | `newmv_diff_bias` (:988) |
// | [`update_thresh_freq_fact`] | `update_thresh_freq_fact` (:1045) |
// | [`is_same_gf_and_last_scale`] | `is_same_gf_and_last_scale` (:1752) |
// ===========================================================================

use crate::intra_rd::MAX_TXSIZE_LOOKUP;

/// `SUBPEL_FORCE_STOP` (`av1/encoder/mcomp.h`) — how far the subpel search is
/// allowed to refine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum SubpelForceStop {
    /// `EIGHTH_PEL` — no forced stop.
    EighthPel = 0,
    /// `QUARTER_PEL`.
    QuarterPel = 1,
    /// `HALF_PEL`.
    HalfPel = 2,
    /// `FULL_PEL` — do not refine at all.
    FullPel = 3,
}

impl SubpelForceStop {
    /// C's raw value.
    #[must_use]
    pub fn to_i32(self) -> i32 {
        self as i32
    }
    /// From C's raw value; `None` outside `0..=3`.
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::EighthPel),
            1 => Some(Self::QuarterPel),
            2 => Some(Self::HalfPel),
            3 => Some(Self::FullPel),
            _ => None,
        }
    }
}

/// `QINDEX_BITS` (`av1/common/enums.h`) — `qindex >> (QINDEX_BITS - 2)` is the
/// two-bit "qband" several of these helpers switch on.
pub const QINDEX_BITS: u32 = 8;

/// `x->qindex >> (QINDEX_BITS - 2)`, the qband. C asserts it is below 4, which
/// holds for any qindex in `0..=255`.
#[inline]
#[must_use]
pub fn qband(qindex: i32) -> usize {
    let b = (qindex >> (QINDEX_BITS - 2)) as usize;
    debug_assert!(b < 4, "qindex {qindex} is outside 0..=255");
    b
}

/// The speed features and block state `subpel_select` reads.
#[derive(Clone, Copy, Debug)]
pub struct SubpelSelectCtx {
    /// `cpi->rc.avg_frame_low_motion`.
    pub avg_frame_low_motion: i32,
    /// `sf.rt_sf.reduce_mv_pel_precision_highmotion`.
    pub reduce_mv_pel_precision_highmotion: i32,
    /// `sf.rt_sf.reduce_mv_pel_precision_lowcomplex`.
    pub reduce_mv_pel_precision_lowcomplex: i32,
    /// `sf.mv_sf.subpel_force_stop` — the value returned when nothing fires.
    pub subpel_force_stop: SubpelForceStop,
    /// `cm->width`.
    pub frame_width: i32,
    /// `cm->height`.
    pub frame_height: i32,
    /// `x->qindex`.
    pub qindex: i32,
    /// `x->content_state_sb.source_sad_nonrd`.
    pub source_sad_nonrd: SourceSad,
    /// `x->source_variance`.
    pub source_variance: i32,
}

/// `subpel_select` (nonrd_pickmode.c:99) — how precise a subpel refinement
/// this block gets.
///
/// The `>= 3` arm computes `mv_thresh = 4` and then IMMEDIATELY overwrites it
/// from the block size; the initial 4 is dead. Reproduced as dead rather than
/// dropped, because a reader diffing the two files should see the same lines.
///
/// Both `reduce_mv_pel_precision_*` cascades can fall through, in which case
/// the speed feature's own `subpel_force_stop` is returned unchanged.
#[must_use]
pub fn subpel_select(
    ctx: &SubpelSelectCtx,
    bsize: usize,
    mv: (i16, i16),
    ref_mv: (i16, i16),
    start_mv: (i16, i16),
    fullpel_performed_well: bool,
) -> SubpelForceStop {
    /// `BLOCK_16X16`.
    const BLOCK_16X16: usize = 6;
    /// `BLOCK_32X32`.
    const BLOCK_32X32: usize = 9;

    let (mv_row, mv_col) = (i32::from(mv.0), i32::from(mv.1));

    if ctx.reduce_mv_pel_precision_highmotion >= 3 {
        let is_low_resoln = ctx.frame_width * ctx.frame_height <= 320 * 240;
        // C's `int mv_thresh = 4;` here is overwritten on the next line.
        let mut mv_thresh = if bsize > BLOCK_32X32 {
            2
        } else if bsize > BLOCK_16X16 {
            4
        } else {
            6
        };
        if ctx.avg_frame_low_motion > 0 && ctx.avg_frame_low_motion < 40 {
            mv_thresh = 12;
        }
        if is_low_resoln {
            mv_thresh >>= 1;
        }
        if mv_row.abs() >= mv_thresh || mv_col.abs() >= mv_thresh {
            return SubpelForceStop::HalfPel;
        }
    } else if ctx.reduce_mv_pel_precision_highmotion >= 1 {
        const TH_VALS: [[i32; 3]; 2] = [[4, 8, 10], [4, 6, 8]];
        let th_idx = (ctx.reduce_mv_pel_precision_highmotion - 1) as usize;
        debug_assert!(th_idx < 2);
        let mv_thresh = if ctx.avg_frame_low_motion > 0 && ctx.avg_frame_low_motion < 40 {
            12
        } else if bsize >= BLOCK_32X32 {
            TH_VALS[th_idx][0]
        } else if bsize >= BLOCK_16X16 {
            TH_VALS[th_idx][1]
        } else {
            TH_VALS[th_idx][2]
        };
        if mv_row.abs() >= (mv_thresh << 1) || mv_col.abs() >= (mv_thresh << 1) {
            return SubpelForceStop::FullPel;
        } else if mv_row.abs() >= mv_thresh || mv_col.abs() >= mv_thresh {
            return SubpelForceStop::HalfPel;
        }
    }

    // Relatively static, low-complexity large areas get less precision.
    if ctx.reduce_mv_pel_precision_lowcomplex >= 2 {
        if ctx.source_sad_nonrd <= SourceSad::VeryLow
            && bsize > BLOCK_16X16
            && qband(ctx.qindex) != 0
        {
            if ctx.source_variance < 500 {
                return SubpelForceStop::FullPel;
            } else if ctx.source_variance < 5000 {
                return SubpelForceStop::HalfPel;
            }
        }
    } else if ctx.reduce_mv_pel_precision_lowcomplex >= 1
        && fullpel_performed_well
        && ref_mv == (0, 0)
        && start_mv == (0, 0)
    {
        return SubpelForceStop::HalfPel;
    }
    ctx.subpel_force_stop
}

/// `use_aggressive_subpel_search_method` (nonrd_pickmode.c:155).
///
/// Note it is gated on `qband > 0`, i.e. it never fires at the lowest quarter
/// of the qindex range no matter how well full-pel did.
#[must_use]
pub fn use_aggressive_subpel_search_method(
    qindex: i32,
    source_sad_nonrd: SourceSad,
    source_variance: i32,
    use_adaptive_subpel_search: bool,
    fullpel_performed_well: bool,
) -> bool {
    if !use_adaptive_subpel_search {
        return false;
    }
    qband(qindex) > 0
        && (fullpel_performed_well || source_sad_nonrd <= SourceSad::Low || source_variance < 100)
}

/// `TX_MODE` (`av1/common/enums.h`).
pub const ONLY_4X4: i32 = 0;
/// `TX_MODE_LARGEST`.
pub const TX_MODE_LARGEST: i32 = 1;
/// `TX_MODE_SELECT`.
pub const TX_MODE_SELECT: i32 = 2;

/// `tx_mode_to_biggest_tx_size` (`common_data.h:368`).
const TX_MODE_TO_BIGGEST_TX_SIZE: [usize; 3] = [
    0, // ONLY_4X4      -> TX_4X4
    4, // TX_MODE_LARGEST -> TX_64X64
    4, // TX_MODE_SELECT  -> TX_64X64
];

/// `TX_8X8`.
const TX_8X8: usize = 1;
/// `TX_16X16`.
const TX_16X16: usize = 2;

/// `CR_SEGMENT_ID_BOOST1` / `_BOOST2` (`aq_cyclicrefresh.h`).
const CR_SEGMENT_ID_BOOST1: i32 = 1;
/// See [`CR_SEGMENT_ID_BOOST1`].
const CR_SEGMENT_ID_BOOST2: i32 = 2;

/// `cyclic_refresh_segment_id_boosted` (`aq_cyclicrefresh.h:312`).
#[inline]
#[must_use]
pub fn cyclic_refresh_segment_id_boosted(segment_id: i32) -> bool {
    segment_id == CR_SEGMENT_ID_BOOST1 || segment_id == CR_SEGMENT_ID_BOOST2
}

/// `CYCLIC_REFRESH_AQ` (`aom/aomcx.h`'s `AQ_MODE`).
pub const CYCLIC_REFRESH_AQ: i32 = 3;

/// Everything `calculate_tx_size` and `set_force_skip_flag` read out of `cpi`
/// and `x` besides the per-call variance and sse.
#[derive(Clone, Copy, Debug)]
pub struct TxSizeCtx {
    /// `x->txfm_search_params.tx_mode_search_type`.
    pub tx_mode_search_type: i32,
    /// `sf.rt_sf.tx_size_level_based_on_qstep`.
    pub tx_size_level_based_on_qstep: i32,
    /// `cpi->oxcf.q_cfg.aq_mode`.
    pub aq_mode: i32,
    /// `xd->mi[0]->segment_id`.
    pub segment_id: i32,
    /// `x->qindex`.
    pub qindex: i32,
    /// `x->plane[AOM_PLANE_Y].dequant_QTX[1]` — the AC quantizer step.
    pub dequant_ac: i32,
    /// `xd->bd`.
    pub bd: u32,
    /// `x->source_variance`.
    pub source_variance: i32,
    /// `x->color_sensitivity[COLOR_SENS_IDX(AOM_PLANE_U)]`.
    pub color_sensitivity_u: u8,
    /// `x->color_sensitivity[COLOR_SENS_IDX(AOM_PLANE_V)]`.
    pub color_sensitivity_v: u8,
}

impl TxSizeCtx {
    /// `qstep = dequant_QTX[1] >> (bd - 5)`, and its square, as C computes
    /// them — `qstep * qstep` in `int` then read as `unsigned int`.
    fn qstep_sq(&self) -> u32 {
        let qstep = self.dequant_ac >> (self.bd - 5);
        (qstep * qstep) as u32
    }
}

/// `set_force_skip_flag` (nonrd_pickmode.c:423).
///
/// Marks a block transform-skip when both its sse and its source variance are
/// under the squared AC quantizer step AND neither chroma plane is
/// colour-sensitive. Only reachable at
/// `tx_size_level_based_on_qstep >= 2`; C tests `!= 0` and `>= 2` on the same
/// value, so the first test is redundant and is kept.
#[must_use]
pub fn set_force_skip_flag(ctx: &TxSizeCtx, sse: u32, force_skip_in: bool) -> bool {
    if ctx.tx_mode_search_type == TX_MODE_SELECT
        && ctx.tx_size_level_based_on_qstep != 0
        && ctx.tx_size_level_based_on_qstep >= 2
    {
        let qstep_sq = ctx.qstep_sq();
        if sse < qstep_sq
            && (ctx.source_variance as u32) < qstep_sq
            && ctx.color_sensitivity_u == 0
            && ctx.color_sensitivity_v == 0
        {
            return true;
        }
    }
    force_skip_in
}

/// `calculate_tx_size` (nonrd_pickmode.c:447).
///
/// Returns `(tx_size, force_skip)`; `force_skip` is C's in/out parameter.
///
/// The final `AOMMIN(tx_size, TX_16X16)` makes several of the branches above
/// it unobservable on their own — the port keeps them because the branch
/// STRUCTURE is what a later change would perturb.
///
/// `CAP_TX_SIZE_FOR_BSIZE_GT32` (:443) forces TX_16X16 for any block above
/// 32x32 unless the mode is ONLY_4X4, and it applies to BOTH arms of the
/// `TX_MODE_SELECT` test.
#[must_use]
pub fn calculate_tx_size(
    ctx: &TxSizeCtx,
    bsize: usize,
    var: u32,
    sse: u32,
    force_skip_in: bool,
) -> (usize, bool) {
    /// `BLOCK_32X32`.
    const BLOCK_32X32: usize = 9;

    let mut force_skip = force_skip_in;
    let mut tx_size;
    let biggest = TX_MODE_TO_BIGGEST_TX_SIZE[ctx.tx_mode_search_type as usize];

    if ctx.tx_mode_search_type == TX_MODE_SELECT {
        let mut multiplier = 8u32;
        let mut var_thresh = 0u32;
        let mut is_high_var = true;
        if ctx.tx_size_level_based_on_qstep != 0 {
            const MULT: [u32; 4] = [8, 7, 6, 5];
            multiplier = MULT[qband(ctx.qindex)];
            let qstep_sq = ctx.qstep_sq();
            var_thresh = qstep_sq * 2;
            if ctx.tx_size_level_based_on_qstep >= 2 {
                if sse < qstep_sq
                    && (ctx.source_variance as u32) < qstep_sq
                    && ctx.color_sensitivity_u == 0
                    && ctx.color_sensitivity_v == 0
                {
                    force_skip = true;
                }
                // Lower the transform size further only if the residual
                // variance is high.
                is_high_var = var >= var_thresh;
            }
        }
        // A larger transform where the DC dominates or the AC is low.
        if sse > ((var * multiplier) >> 2) || var < var_thresh {
            tx_size = MAX_TXSIZE_LOOKUP[bsize].min(biggest);
        } else {
            tx_size = TX_8X8;
        }

        if ctx.aq_mode == CYCLIC_REFRESH_AQ
            && cyclic_refresh_segment_id_boosted(ctx.segment_id)
            && is_high_var
        {
            tx_size = TX_8X8;
        } else if tx_size > TX_16X16 {
            tx_size = TX_16X16;
        }
    } else {
        tx_size = MAX_TXSIZE_LOOKUP[bsize].min(biggest);
    }

    // CAP_TX_SIZE_FOR_BSIZE_GT32 (:443).
    if ctx.tx_mode_search_type != ONLY_4X4 && bsize > BLOCK_32X32 {
        tx_size = TX_16X16;
    }
    (tx_size.min(TX_16X16), force_skip)
}

/// `INVALID_MV` (`av1/common/mv.h:26`) as the packed `as_int`.
pub const INVALID_MV: u32 = 0x8000_8000;

/// `newmv_diff_bias` (nonrd_pickmode.c:988) — penalise a NEWMV whose vector
/// disagrees with its neighbours.
///
/// Returns the adjusted `rdcost`. Three separate multipliers, and the first
/// one RETURNS EARLY, so a large-block low-variance outlier gets `<< 2` and
/// never reaches the neighbour comparison.
///
/// The neighbour average is `(above + left + 1) >> 1` when both are valid, one
/// of them when only one is, and ZERO when neither — which biases against any
/// non-zero MV at a block with no coded neighbours.
///
/// The `else` branch (a non-NEWMV mode) has its own, unrelated bias at
/// speed >= 8.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn newmv_diff_bias(
    this_mode: PredMode,
    rdcost: i64,
    bsize: usize,
    mv_row: i32,
    mv_col: i32,
    speed: i32,
    spatial_variance: u32,
    source_sad_nonrd: SourceSad,
    above: Option<(i16, i16)>,
    left: Option<(i16, i16)>,
) -> i64 {
    /// `BLOCK_32X32`.
    const BLOCK_32X32: usize = 9;
    /// `BLOCK_64X64`.
    const BLOCK_64X64: usize = 12;

    if this_mode != PredMode::NewMv {
        // Bias for speed >= 8 at low spatial variance.
        if speed >= 8
            && spatial_variance < 150
            && (mv_row > 64 || mv_row < -64 || mv_col > 64 || mv_col < -64)
        {
            return 5 * rdcost >> 2;
        }
        return rdcost;
    }

    if bsize >= BLOCK_64X64
        && source_sad_nonrd != SourceSad::High
        && spatial_variance < 300
        && (mv_row > 16 || mv_row < -16 || mv_col > 16 || mv_col < -16)
    {
        return rdcost << 2;
    }

    // C reads `above_mbmi->mv[0]` even when the MV is INVALID_MV; only the
    // VALIDITY flag comes from the as_int test, and the row/col are taken
    // unconditionally. A neighbour that exists with an invalid MV therefore
    // contributes nothing, and one that does not exist at all is the same.
    let (above_valid, above_row, above_col) = match above {
        Some((r, c)) => (pack_mv(r, c) != INVALID_MV, i32::from(r), i32::from(c)),
        None => (
            false,
            i32::from(INVALID_MV_ROW_COL),
            i32::from(INVALID_MV_ROW_COL),
        ),
    };
    let (left_valid, left_row, left_col) = match left {
        Some((r, c)) => (pack_mv(r, c) != INVALID_MV, i32::from(r), i32::from(c)),
        None => (
            false,
            i32::from(INVALID_MV_ROW_COL),
            i32::from(INVALID_MV_ROW_COL),
        ),
    };

    let (al_row, al_col) = if above_valid && left_valid {
        (
            (above_row + left_row + 1) >> 1,
            (above_col + left_col + 1) >> 1,
        )
    } else if above_valid {
        (above_row, above_col)
    } else if left_valid {
        (left_row, left_col)
    } else {
        (0, 0)
    };

    let row_diff = al_row - mv_row;
    let col_diff = al_col - mv_col;
    if row_diff > 80 || row_diff < -80 || col_diff > 80 || col_diff < -80 {
        if bsize >= BLOCK_32X32 {
            return rdcost << 1;
        }
        return 5 * rdcost >> 2;
    }
    rdcost
}

/// `INVALID_MV_ROW_COL` (`av1/common/mv.h:27`).
pub const INVALID_MV_ROW_COL: i16 = -32768;

/// `int_mv`'s `as_int` view of a `(row, col)` pair: row in the LOW half.
#[inline]
#[must_use]
pub fn pack_mv(row: i16, col: i16) -> u32 {
    (u32::from(col as u16) << 16) | u32::from(row as u16)
}

/// `RD_THRESH_INC` (`av1/encoder/rd.h:55`).
pub const RD_THRESH_INC: i32 = 1;
/// `RD_THRESH_MAX_FACT` (`rd.h:53`) — `RD_THRESH_FAC_FRAC_VAL << 1`.
pub const RD_THRESH_MAX_FACT: i32 = 64;

/// `update_thresh_freq_fact` (nonrd_pickmode.c:1045) — decay the RD threshold
/// factor for the mode that won and raise it for the ones that did not.
///
/// The `for (bs = min_size; bs <= max_size; bs += 3)` walk steps by THREE
/// through `BLOCK_SIZE`, which visits the square sizes and their two
/// rectangular siblings in turn; `min_size` and `max_size` are `bsize - 3` and
/// `bsize + 6` clamped to the enum's ends. `freq_fact` is indexed
/// `[block size][THR_MODES]` and is in/out.
pub fn update_thresh_freq_fact(
    adaptive_rd_thresh: i32,
    freq_fact: &mut [[i32; MAX_MODES]],
    bsize: usize,
    ref_frame: usize,
    best_mode_idx: usize,
    mode_offset: usize,
) {
    /// `BLOCK_4X4`.
    const BLOCK_4X4: usize = 0;
    /// `BLOCK_128X128`.
    const BLOCK_128X128: usize = 15;

    let thr_mode_idx = MODE_IDX[ref_frame][mode_offset];
    let min_size = bsize.saturating_sub(3).max(BLOCK_4X4);
    let max_size = (bsize + 6).min(BLOCK_128X128);
    let mut bs = min_size;
    while bs <= max_size {
        let cell = &mut freq_fact[bs][thr_mode_idx];
        if thr_mode_idx == best_mode_idx {
            *cell -= *cell >> 4;
        } else {
            *cell = (*cell + RD_THRESH_INC).min(adaptive_rd_thresh * RD_THRESH_MAX_FACT);
        }
        bs += 3;
    }
}

/// `MAX_MODES` (`av1/encoder/enc_enums.h`) — the width of `thresh_freq_fact`.
pub const MAX_MODES: usize = 169;

/// `is_same_gf_and_last_scale` (nonrd_pickmode.c:1752).
///
/// Compares the two references' scale factors, not their sizes: a golden frame
/// that happens to be the same size but is reached through a different scale
/// still reads as different.
#[must_use]
pub fn is_same_gf_and_last_scale(
    last_x_scale_fp: i32,
    last_y_scale_fp: i32,
    golden_x_scale_fp: i32,
    golden_y_scale_fp: i32,
) -> bool {
    last_x_scale_fp == golden_x_scale_fp && last_y_scale_fp == golden_y_scale_fp
}
