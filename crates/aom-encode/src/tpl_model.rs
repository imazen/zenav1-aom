//! Temporal dependency model (TPL) — port of `av1/encoder/tpl_model.c`.
//!
//! TPL is what makes the CRF path adaptive across a GOP: it propagates each
//! block's coding cost backwards through the reference chain, turns the result
//! into a per-frame *importance* number, and that number becomes the frame's
//! qindex and its per-superblock rdmult scaling. Nothing downstream of it is
//! byte-reproducible until this arithmetic is.
//!
//! This module holds the parts of `tpl_model.c` that are pure arithmetic over
//! scalars — the entropy model, the dependency-cost algebra, and the
//! qstep-ratio → qindex search. The parts that walk `TplDepFrame` arrays land
//! alongside them as the surrounding structs get ported.
//!
//! | Rust | C (`av1/encoder/tpl_model.c`) |
//! |---|---|
//! | [`exp_bounded`] | `exp_bounded` (:43, static) |
//! | [`exponential_entropy`] | `av1_exponential_entropy` (:2347) |
//! | [`laplace_entropy`] | `av1_laplace_entropy` (:2353) |
//! | [`estimate_coeff_entropy`] | `av1_estimate_coeff_entropy` (:2383) |
//! | [`get_overlap_area`] | `av1_get_overlap_area` (:1159) |
//! | [`tpl_ptr_pos`] | `av1_tpl_ptr_pos` (:1171) |
//! | [`round_floor`] | `round_floor` (:1149, static) |
//! | [`delta_rate_cost`] | `av1_delta_rate_cost` (:1175) |
//! | [`get_q_index_from_qstep_ratio`] | `av1_get_q_index_from_qstep_ratio` (:2427) |
//!
//! # Not ported, and why
//! - `av1_accumulate_tpl_txfm_stats`, `av1_record_tpl_txfm_block`,
//!   `av1_tpl_txfm_stats_update_abs_coeff_mean`, `av1_tpl_store_txfm_stats`,
//!   `av1_laplace_estimate_frame_rate` and the whole `av1_vbr_rc_*` family are
//!   inside `#if CONFIG_BITRATE_ACCURACY`, and `av1_read_rd_command` inside
//!   `#if CONFIG_RD_COMMAND`. Both macros are **0** in this build
//!   (`upstream/build/config/aom_config.h:26,62`), so none of those functions
//!   exists in `libaom.a` — verified with `nm -g`. They are not part of the
//!   encoder under test and have no oracle at any tier.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/tpl_model_diff.rs` — **tier 1**, against the real
//! exported C symbols out of `upstream/build/libaom.a`. `round_floor` is
//! file-static and has no exported symbol; it is gated indirectly, see the
//! test's own note.

use aom_dsp::quant::av1_dc_quant_qtx;

/// `TPL_EPSILON` (tpl_model.h:107).
const TPL_EPSILON: f64 = 0.0000001;

/// `TPL_DEP_COST_SCALE_LOG2` (tpl_model.h:105).
pub const TPL_DEP_COST_SCALE_LOG2: u32 = 4;

/// `AV1_PROB_COST_SHIFT` (encoder/cost.h:25).
const AV1_PROB_COST_SHIFT: u32 = 9;

/// `MAXQ` (common/quant_common.h:26).
const MAXQ: i32 = 255;

/// `exp_bounded` (tpl_model.c:43) — `exp` with the overflow ends pinned.
///
/// C returns `DBL_MAX` above 700 and `0` below -700 rather than letting `exp`
/// overflow to infinity or underflow to a subnormal. The bounds are exclusive
/// on both sides (`v > 700`, `v < -700`), so exactly ±700 still evaluates
/// `exp`; reproducing that is the difference between `DBL_MAX` and
/// `1.0142e304` at the positive edge.
///
/// **Neither clamp is observable through the entropy functions**, measured by
/// perturbing each arm and re-running `tests/tpl_model_diff.rs`:
/// - the positive arm is *unreachable* there — both [`exponential_entropy`]
///   and [`laplace_entropy`] evaluate `exp_bounded` at a non-positive
///   argument, so `f64::MAX` vs `f64::INFINITY` makes no difference and the
///   20k-case sweeps stay green;
/// - the negative arm is reachable but *inert* — `exp(-700)` is already
///   9.86e-305, far below the `TPL_EPSILON` (1e-7) floor applied immediately
///   after, so moving the cut to -1400 also leaves the sweeps green.
///
/// Both arms matter to `av1_tpl_rdmult_setup_sb`, which calls `exp_bounded`
/// on a `log`-domain ratio with no epsilon floor after it. Until that lands,
/// the clamp is pinned by the tier-4 vectors in
/// `exp_bounded_saturation_is_reached` rather than by a tier-1 differential —
/// see that test's note.
#[must_use]
pub fn exp_bounded(v: f64) -> f64 {
    if v > 700.0 {
        f64::MAX
    } else if v < -700.0 {
        0.0
    } else {
        v.exp()
    }
}

/// `av1_exponential_entropy` (tpl_model.c:2347) — entropy of an exponential
/// coefficient distribution with scale `b`, quantized at step `q_step`.
///
/// Both `b` and the survival probability `z` are floored (at `TPL_EPSILON`)
/// before use, so the two singularities of the closed form — `b == 0` and
/// `z == 0` — are unreachable. `z == 1` is not floored and is reachable in
/// principle (`q_step == 0`), which is what C does; the caller never passes a
/// zero quantizer step.
#[must_use]
pub fn exponential_entropy(q_step: f64, b: f64) -> f64 {
    let b = b.max(TPL_EPSILON);
    let z = exp_bounded(-q_step / b).max(TPL_EPSILON);
    -(1.0 - z).log2() - z * z.log2() / (1.0 - z)
}

/// `av1_laplace_entropy` (tpl_model.c:2353) — entropy of a Laplace
/// coefficient distribution quantized with a dead zone.
///
/// The zero bin is `zero_bin_ratio * q_step` wide, every other bin `q_step`.
/// `z` is the probability mass outside the zero bin; the tail beyond it is
/// charged as one sign bit plus the exponential entropy of the magnitude.
#[must_use]
pub fn laplace_entropy(q_step: f64, b: f64, zero_bin_ratio: f64) -> f64 {
    let b = b.max(TPL_EPSILON);
    let z = exp_bounded(-zero_bin_ratio / 2.0 * q_step / b).max(TPL_EPSILON);
    let h = exponential_entropy(q_step, b);
    -(1.0 - z) * (1.0 - z).log2() - z * z.log2() + z * (h + 1.0)
}

/// `av1_estimate_coeff_entropy` (tpl_model.c:2383) — the modelled bit cost of
/// one already-quantized coefficient under the same Laplace model.
///
/// Note this is the cost of a *specific* `qcoeff`, not an expectation: zero
/// costs `-log2(1 - z0)`, and a magnitude-`n` coefficient costs one sign bit
/// plus the geometric tail. Only `abs(qcoeff)` is read, so the sign is charged
/// as a flat bit exactly as C does.
#[must_use]
pub fn estimate_coeff_entropy(q_step: f64, b: f64, zero_bin_ratio: f64, qcoeff: i32) -> f64 {
    let b = b.max(TPL_EPSILON);
    let abs_qcoeff = f64::from(qcoeff.unsigned_abs());
    let z0 = exp_bounded(-zero_bin_ratio / 2.0 * q_step / b).max(TPL_EPSILON);
    if qcoeff == 0 {
        -(1.0 - z0).log2()
    } else {
        let z = exp_bounded(-q_step / b).max(TPL_EPSILON);
        1.0 - z0.log2() - (1.0 - z).log2() - (abs_qcoeff - 1.0) * z.log2()
    }
}

/// `av1_get_overlap_area` (tpl_model.c:1159) — the intersection area of two
/// `width` x `height` rectangles placed at `(row_a, col_a)` and `(row_b,
/// col_b)`.
///
/// Both rectangles share one size, which is why C takes six ints rather than
/// eight: this is only ever called on two TPL blocks of the same granularity.
/// Returns 0 when they do not overlap (C tests `min < max` on both axes and
/// falls through).
#[must_use]
pub fn get_overlap_area(
    row_a: i32,
    col_a: i32,
    row_b: i32,
    col_b: i32,
    width: i32,
    height: i32,
) -> i32 {
    let min_row = row_a.max(row_b);
    let max_row = (row_a + height).min(row_b + height);
    let min_col = col_a.max(col_b);
    let max_col = (col_a + width).min(col_b + width);
    if min_row < max_row && min_col < max_col {
        (max_row - min_row) * (max_col - min_col)
    } else {
        0
    }
}

/// `av1_tpl_ptr_pos` (tpl_model.c:1171) — index of the TPL stats cell covering
/// mi position `(mi_row, mi_col)`.
///
/// `right_shift` is `tpl_stats_block_mis_log2` (2, i.e. 16x16 granularity),
/// so this is a plain row-major index into the *decimated* grid. C's shifts
/// are on `int`, and both operands are non-negative at every call site.
#[must_use]
pub fn tpl_ptr_pos(mi_row: i32, mi_col: i32, stride: i32, right_shift: u8) -> i32 {
    (mi_row >> right_shift) * stride + (mi_col >> right_shift)
}

/// `round_floor` (tpl_model.c:1149, static) — divide `ref_pos` by `bsize_pix`
/// rounding towards negative infinity.
///
/// C spells the negative arm as `-(1 + (-ref_pos - 1) / bsize_pix)` because C
/// division truncates towards zero. Rust's `/` truncates identically, so this
/// is `div_euclid` for positive divisors — but it is written the C way here
/// because `bsize_pix` is only positive *by call-site convention*, and
/// `div_euclid` would round the other way if that convention were ever broken
/// with a negative divisor. Reproducing C's arithmetic keeps the two in
/// lockstep on inputs neither of them is documented for.
#[must_use]
pub fn round_floor(ref_pos: i32, bsize_pix: i32) -> i32 {
    if ref_pos < 0 {
        -(1 + (-ref_pos - 1) / bsize_pix)
    } else {
        ref_pos / bsize_pix
    }
}

/// `av1_delta_rate_cost` (tpl_model.c:1175) — the propagated rate cost of a
/// block, discounted by how much of its distortion the reference already
/// explains.
///
/// The model: `beta = srcrf_dist / recrf_dist` is how much better the *source*
/// reference is than the reconstructed one. `delta_rate` is charged in full
/// when there is essentially no source distortion to propagate
/// (`srcrf_dist <= 128`, C's early exit). Otherwise the rate is re-derived
/// from the distortion ratio; when the log-domain denominator exceeds
/// `log2(10)` the closed form saturates and C uses the limit expression
/// instead.
///
/// Integer widths are C's: `rate_cost` is `int64_t` throughout, the two
/// `f64 -> i64` conversions truncate towards zero (both C's cast and Rust's
/// `as`), and the final `<<` is on `int64_t`. `pix_num` is `int`.
#[must_use]
pub fn delta_rate_cost(delta_rate: i64, recrf_dist: i64, srcrf_dist: i64, pix_num: i32) -> i64 {
    let shift = TPL_DEP_COST_SCALE_LOG2 + AV1_PROB_COST_SHIFT;
    if srcrf_dist <= 128 {
        return delta_rate;
    }
    let beta = srcrf_dist as f64 / recrf_dist as f64;
    let pix_num = f64::from(pix_num);

    let dr = (delta_rate >> shift) as f64 / pix_num;
    let log_den = beta.ln() / 2f64.ln() + 2.0 * dr;

    if log_den > 10f64.ln() / 2f64.ln() {
        let rate_cost = (((1.0 / beta).ln() * pix_num) / 2f64.ln() / 2.0) as i64;
        return rate_cost << shift;
    }

    let num = 2f64.powf(log_den);
    let den = num * beta + (1.0 - beta) * beta;
    let rate_cost = ((pix_num * (num / den).ln()) / 2f64.ln() / 2.0) as i64;
    rate_cost << shift
}

/// `av1_get_q_index_from_qstep_ratio` (tpl_model.c:2427) — the qindex whose DC
/// quantizer step is closest to `leaf_qstep * qstep_ratio`, searched linearly
/// from `leaf_qindex`.
///
/// This is a *linear* scan, not a binary search, and the two directions stop on
/// different comparisons (`qstep <= target` walking down, `qstep >= target`
/// walking up) — the down-scan therefore stops at the first index at or below
/// the target and the up-scan at the first at or above it. `qstep_ratio == 1.0`
/// takes the up arm and returns `leaf_qindex` immediately.
///
/// The bounds are C's: down to 0 inclusive (the loop condition is `qindex > 0`,
/// so 0 is reachable only by falling out), up to `MAXQ` = 255.
#[must_use]
pub fn get_q_index_from_qstep_ratio(leaf_qindex: i32, qstep_ratio: f64, bit_depth: u8) -> i32 {
    let leaf_qstep = f64::from(av1_dc_quant_qtx(leaf_qindex, 0, bit_depth));
    let target_qstep = leaf_qstep * qstep_ratio;
    if qstep_ratio < 1.0 {
        let mut qindex = leaf_qindex;
        while qindex > 0 {
            if f64::from(av1_dc_quant_qtx(qindex, 0, bit_depth)) <= target_qstep {
                break;
            }
            qindex -= 1;
        }
        qindex
    } else {
        let mut qindex = leaf_qindex;
        while qindex < MAXQ {
            if f64::from(av1_dc_quant_qtx(qindex, 0, bit_depth)) >= target_qstep {
                break;
            }
            qindex += 1;
        }
        qindex
    }
}

// ===========================================================================
// The TPL data model, and the frame-importance -> qindex chain built on it.
// ===========================================================================

use crate::rdopt_mv::Mv;

/// `INTER_REFS_PER_FRAME` (`av1/common/enums.h`).
pub const INTER_REFS_PER_FRAME: usize = 7;

/// `REF_FRAMES` (`av1/common/enums.h`).
pub const REF_FRAMES: usize = 8;

/// `MAX_LAG_BUFFERS` (`encoder/lookahead.h:28`).
pub const MAX_LAG_BUFFERS: usize = 48;

/// `MAX_TPL_FRAME_IDX` (tpl_model.h:99) — the sub-GOP length past which TPL
/// switches itself off rather than run out of buffer.
pub const MAX_TPL_FRAME_IDX: i32 = 2 * MAX_LAG_BUFFERS as i32;

/// `RDDIV_BITS` (encoder/rd.h:29).
const RDDIV_BITS: u32 = 7;

/// `RDCOST(RM, R, D)` (encoder/rd.h:32) instantiated with an **`int64_t`
/// rate**, which is how `tpl_model.c` uses it — `TplDepStats::mc_dep_rate` is
/// `int64_t`, not the `int` that [`crate::rd::rdcost`] takes.
///
/// C's macro multiplies `(int64_t)R * RM` and shifts with
/// `ROUND_POWER_OF_TWO`, then adds `D << RDDIV_BITS`. Both products can
/// overflow `int64_t` on adversarial inputs, where C is undefined and clang
/// wraps; the wrapping ops here reproduce that rather than panicking in a
/// debug build. Real TPL state cannot reach it: `mc_dep_rate` is a sum of
/// `av1_delta_rate_cost` outputs over one frame and `base_rdmult` is bounded
/// by `av1_compute_rd_mult`.
#[inline]
fn rdcost_i64_rate(rm: i32, rate: i64, dist: i64) -> i64 {
    let scaled = rate.wrapping_mul(i64::from(rm));
    let rounded = scaled.wrapping_add(1 << (AV1_PROB_COST_SHIFT - 1)) >> AV1_PROB_COST_SHIFT;
    rounded.wrapping_add(dist.wrapping_mul(1 << RDDIV_BITS))
}

/// `TplDepStats` (tpl_model.h:145) — one cell of the TPL grid.
///
/// The grid is decimated by `tpl_stats_block_mis_log2` (2, i.e. one cell per
/// 16x16 luma block); [`tpl_ptr_pos`] maps an mi position to an index here.
///
/// Field widths are C's, and the split matters: the `_dist`/`_sse`/`mc_dep_*`
/// family is `int64_t` because it accumulates squared error over a whole
/// dependency chain, while the `_cost`/`_rate` family is `int32_t`. Narrowing
/// the first group or widening the second changes where the arithmetic
/// saturates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TplDepStats {
    /// Squared error of the source-reference prediction.
    pub srcrf_sse: i64,
    /// Distortion of the source-reference prediction.
    pub srcrf_dist: i64,
    /// Squared error of the reconstructed-reference prediction.
    pub recrf_sse: i64,
    /// Distortion of the reconstructed-reference prediction.
    pub recrf_dist: i64,
    /// Squared error of the intra prediction.
    pub intra_sse: i64,
    /// Distortion of the intra prediction.
    pub intra_dist: i64,
    /// Per-reference compound reconstruction distortion.
    pub cmp_recrf_dist: [i64; 2],
    /// Rate propagated into this block from its dependents.
    pub mc_dep_rate: i64,
    /// Distortion propagated into this block from its dependents.
    pub mc_dep_dist: i64,
    /// Per-reference prediction error, used to pick the reference pair.
    pub pred_error: [i64; INTER_REFS_PER_FRAME],
    /// Intra mode cost.
    pub intra_cost: i32,
    /// Inter mode cost.
    pub inter_cost: i32,
    /// Rate of the source-reference prediction.
    pub srcrf_rate: i32,
    /// Rate of the reconstructed-reference prediction.
    pub recrf_rate: i32,
    /// Rate of the intra prediction.
    pub intra_rate: i32,
    /// Per-reference compound reconstruction rate.
    pub cmp_recrf_rate: [i32; 2],
    /// The chosen motion vector per reference.
    pub mv: [Mv; INTER_REFS_PER_FRAME],
    /// The chosen reference pair, as indices into `mv` (`-1` = none).
    pub ref_frame_index: [i8; 2],
}

/// `TplDepFrame` (tpl_model.h:147) — the TPL state of one frame in the GOP.
///
/// # Not represented, and why
/// C's `gf_picture` / `rec_picture` are `YV12_BUFFER_CONFIG *` into the
/// encoder's frame pools, and `ref_map_index` indexes those pools. Frame
/// ownership is the mechanism this port replaces rather than translates, so
/// the pixel buffers are passed to the routines that need them instead of
/// being reachable from here. `width` / `height` (the grid extent in 16x16
/// units) are kept because the propagation walk reads them.
#[derive(Clone, Debug, Default)]
pub struct TplDepFrame {
    /// Whether this frame's stats were computed (`is_valid` in C, a `uint8_t`
    /// used only as a flag).
    pub is_valid: bool,
    /// The decimated grid, indexed by [`tpl_ptr_pos`].
    pub stats: Vec<TplDepStats>,
    /// Row stride of `stats`, in grid cells.
    pub stride: i32,
    /// Grid width, in 16x16 blocks.
    pub width: i32,
    /// Grid height, in 16x16 blocks.
    pub height: i32,
    /// Frame height in mi units — the walk bound, NOT `height << shift`.
    pub mi_rows: i32,
    /// Frame width in mi units.
    pub mi_cols: i32,
    /// The frame's RD multiplier, used to fold rate into distortion.
    pub base_rdmult: i32,
    /// Display order of this frame.
    pub frame_display_index: u32,
    /// When set, SAD replaces SSE in the intra/inter decision.
    pub use_pred_sad: bool,
    /// `ref_map_index[i]` is the GOP index of the frame in reference slot
    /// `i`, or `-1` for an empty slot. C stores it as `int[REF_FRAMES]`; the
    /// negative sentinel is load-bearing (`tpl_model_update_b` returns early
    /// on it) so it stays an `i32` rather than becoming an `Option<usize>` —
    /// the array is written wholesale by the GOP setup and read by index.
    pub ref_map_index: [i32; REF_FRAMES],
}

/// `TplParams` (tpl_model.h:168) — the GOP-level TPL state, reduced to the
/// fields the ported arithmetic reads.
///
/// # Not represented, and why
/// `tpl_stats_pool`, `tpl_rec_pool`, `txfm_stats_list`, `src_ref_frame`,
/// `ref_frame`, `prev_gop_arf_src` and `tpl_mt_sync` are allocation and
/// threading bookkeeping: C's pools exist so `tpl_stats_buffer[i].
/// tpl_stats_ptr` can be handed out and reclaimed, and `tpl_mt_sync` shards a
/// row walk this port does in one pass. Rust owns each frame's grid inside
/// [`TplDepFrame::stats`], so there is no pool to model. C's `tpl_frame` is a
/// pointer into `tpl_stats_buffer` past the `REF_FRAMES + 1` reserved slots;
/// [`Self::frames`] is that view directly.
#[derive(Clone, Debug, Default)]
pub struct TplParams {
    /// Whether the GOP's TPL pass ran to completion.
    pub ready: bool,
    /// `log2` of the grid decimation in mi units. Always 2 (16x16) — see
    /// [`Self::set_tpl_stats_block_size`].
    pub tpl_stats_block_mis_log2: u8,
    /// The 1-D block size TPL motion search uses. Always 16.
    pub tpl_bsize_1d: u8,
    /// Per-frame TPL state, indexed by GOP frame index.
    pub frames: Vec<TplDepFrame>,
    /// GOP index of the frame being processed.
    pub frame_idx: i32,
    /// Correction applied to `r0` when TPL covered only part of the GOP.
    pub r0_adjust_factor: f64,
}

impl TplParams {
    /// `set_tpl_stats_block_size` (tpl_model.c:142, static) — pins the TPL
    /// grid to 16x16.
    ///
    /// C writes through two out-pointers and asserts `tpl_bsize_1d >= 16`;
    /// both values are compile-time constants there, so this is a setter, not
    /// a computation. It is kept as a named function because
    /// `av1_init_tpl_stats` and `av1_setup_tpl_buffers` both call it and a
    /// future change to the granularity has to move through one place.
    pub fn set_tpl_stats_block_size(&mut self) {
        self.tpl_stats_block_mis_log2 = 2;
        self.tpl_bsize_1d = 16;
        debug_assert!(self.tpl_bsize_1d >= 16);
    }

    /// `av1_tpl_stats_ready` (tpl_model.c:1856) — whether frame
    /// `gf_frame_index` has usable TPL stats.
    ///
    /// Three independent gates, and the middle one is the interesting one:
    /// when the sub-GOP is longer than the TPL buffer
    /// ([`MAX_TPL_FRAME_IDX`]) C reports *not ready* rather than growing the
    /// buffer, which silently disables every TPL-driven decision for the rest
    /// of that GOP.
    #[must_use]
    pub fn tpl_stats_ready(&self, gf_frame_index: i32) -> bool {
        if !self.ready {
            return false;
        }
        if gf_frame_index >= MAX_TPL_FRAME_IDX {
            return false;
        }
        self.frames
            .get(usize::try_from(gf_frame_index).expect("gf_frame_index must be non-negative"))
            .is_some_and(|f| f.is_valid)
    }

    /// `get_frame_importance` (tpl_model.c:1942, static) — how much the rest
    /// of the GOP depends on this frame.
    ///
    /// The model: for every grid cell, compare `log(recrf_dist)` (what coding
    /// this block costs on its own) against `log(recrf_dist + mc_dep_delta)`
    /// (what it costs once everything that references it is charged too),
    /// weighting each cell by its source distortion `cbcmp`. The exponential
    /// of the weighted mean difference is the importance; 1.0 means nothing
    /// depends on this frame.
    ///
    /// Three details that a "cleaner" rewrite loses:
    /// - `cbcmp_base` starts at **1**, not 0, so a frame whose cells all have
    ///   `srcrf_dist == 0` returns `exp(0) == 1` instead of dividing by zero;
    /// - `dist_scaled` is floored at 1 *before* the log, so an all-zero
    ///   distortion cell contributes `log(1) == 0` rather than `-inf`. That
    ///   floor is **inert on encoder-produced state** (measured: deleting it
    ///   leaves the randomized differential green), because
    ///   `tpl_model_store` (tpl_model.c:1301) clamps
    ///   `recrf_dist = AOMMAX(1, recrf_dist)` before a cell is stored, making
    ///   `dist_scaled >= 128`. It is kept because C keeps it, and is pinned by
    ///   `get_frame_importance_degenerate_cells_match_c`;
    /// - the walk bound is `mi_rows`/`mi_cols` stepping by the decimation,
    ///   not the grid's own `width`/`height`.
    #[must_use]
    fn get_frame_importance(&self, gf_frame_index: i32) -> f64 {
        let tpl_frame = &self.frames[gf_frame_index as usize];
        let tpl_stride = tpl_frame.stride;
        let shift = self.tpl_stats_block_mis_log2;
        let step = 1 << shift;

        let mut intra_cost_base = 0.0f64;
        let mut mc_dep_cost_base = 0.0f64;
        let mut cbcmp_base = 1.0f64;

        let mut row = 0;
        while row < tpl_frame.mi_rows {
            let mut col = 0;
            while col < tpl_frame.mi_cols {
                let this_stats =
                    &tpl_frame.stats[tpl_ptr_pos(row, col, tpl_stride, shift) as usize];
                let cbcmp = this_stats.srcrf_dist as f64;
                let mc_dep_delta = rdcost_i64_rate(
                    tpl_frame.base_rdmult,
                    this_stats.mc_dep_rate,
                    this_stats.mc_dep_dist,
                );
                let dist_scaled = (this_stats.recrf_dist << RDDIV_BITS) as f64;
                // C's AOMMAX against the integer literal 1, not `f64::max`.
                let dist_scaled = if dist_scaled > 1.0 { dist_scaled } else { 1.0 };
                intra_cost_base += dist_scaled.ln() * cbcmp;
                mc_dep_cost_base += (dist_scaled + mc_dep_delta as f64).ln() * cbcmp;
                cbcmp_base += cbcmp;
                col += step;
            }
            row += step;
        }
        ((mc_dep_cost_base - intra_cost_base) / cbcmp_base).exp()
    }

    /// `av1_tpl_get_qstep_ratio` (tpl_model.c:2418) — the multiplier applied
    /// to this frame's quantizer step, from its importance.
    ///
    /// `sqrt(1 / importance)`: an important frame (importance > 1) gets a
    /// *smaller* step, i.e. finer quantization, because the rest of the GOP
    /// will inherit its errors. When TPL has no stats the ratio is exactly 1
    /// and the qindex is left alone.
    #[must_use]
    pub fn tpl_get_qstep_ratio(&self, gf_frame_index: i32) -> f64 {
        if !self.tpl_stats_ready(gf_frame_index) {
            return 1.0;
        }
        (1.0 / self.get_frame_importance(gf_frame_index)).sqrt()
    }

    /// `av1_tpl_get_q_index` (tpl_model.c:2446) — the frame's qindex on the
    /// CRF path: importance, to a qstep ratio, to a qindex.
    #[must_use]
    pub fn tpl_get_q_index(&self, gf_frame_index: i32, leaf_qindex: i32, bit_depth: u8) -> i32 {
        let qstep_ratio = self.tpl_get_qstep_ratio(gf_frame_index);
        get_q_index_from_qstep_ratio(leaf_qindex, qstep_ratio, bit_depth)
    }

    /// `av1_init_tpl_stats` (tpl_model.c:1839) — reset every frame's stats
    /// before a GOP.
    ///
    /// C does three things: clear `ready`, re-pin the block size, and mark all
    /// `MAX_LENGTH_TPL_FRAME_STATS` frames invalid. Its fourth act — a
    /// `memset` over each allocated `tpl_stats_pool[i]` — is pool
    /// bookkeeping: the cells are only ever read after `is_valid` is set
    /// again, and Rust owns each frame's `stats` vector, so clearing it here
    /// is equivalent and is done for the frames that exist.
    pub fn init_tpl_stats(&mut self) {
        self.ready = false;
        self.set_tpl_stats_block_size();
        for frame in &mut self.frames {
            frame.is_valid = false;
            frame.stats.fill(TplDepStats::default());
        }
    }
}

// ===========================================================================
// The backward-propagation core (all file-static in C; gated at tier 1c
// through `shim/tpl_c_shim.c`, which compiles tpl_model.c verbatim).
// ===========================================================================

use crate::rd::FrameUpdateType;
use aom_dsp::txb::SCAN_ORDERS;
use std::cmp::Ordering;

/// `MI_SIZE` (`av1/common/enums.h:40`) — mi units are 4 luma pixels.
const MI_SIZE: i32 = 4;

/// `DCT_DCT` — the only transform type TPL's rate model scans with.
const DCT_DCT: usize = 0;

/// `get_msb(n)` (`aom_ports/bitops.h`) — `floor(log2(n))`, undefined at 0.
///
/// C asserts `n != 0` and computes `31 ^ clz(n)`. The one caller here always
/// passes `abs_level + 1 >= 1`, so the contract is asserted rather than the
/// undefined behaviour reproduced.
#[inline]
fn get_msb(n: u32) -> u32 {
    assert!(n != 0, "get_msb is undefined at 0");
    31 - n.leading_zeros()
}

/// `rate_estimator` (tpl_model.c:228, static) — TPL's stand-in for a real
/// entropy coder.
///
/// Each of the first `eob` coefficients *in scan order* costs
/// `msb(|level| + 1) + 1` bits, plus one more if it is non-zero; the running
/// total starts at 1 and the result is scaled into `AV1_PROB_COST_SHIFT`
/// units. This is a magnitude-only model — it ignores context, position and
/// sign entirely, which is the whole reason TPL can afford to run it on every
/// block of every frame in the GOP.
///
/// `qcoeff` is indexed *through* the scan, so it is in raster order; `eob` is
/// a count of scan positions, not a raster index.
///
/// # Panics
/// If `eob` exceeds the transform's coefficient count, matching C's `assert`.
#[must_use]
pub fn rate_estimator(qcoeff: &[i32], eob: usize, tx_size: usize) -> i32 {
    let scan = SCAN_ORDERS[tx_size][DCT_DCT].0;
    assert!(
        eob <= scan.len(),
        "eob {eob} exceeds the {} coefficients of tx_size {tx_size}",
        scan.len()
    );
    let mut rate_cost: i32 = 1;
    for &pos in &scan[..eob] {
        let abs_level = qcoeff[pos as usize].unsigned_abs();
        rate_cost += get_msb(abs_level + 1) as i32 + 1 + i32::from(abs_level > 0);
    }
    rate_cost << AV1_PROB_COST_SHIFT
}

/// `get_gop_length` (tpl_model.c:1318, static) — the GF group size, clamped
/// to what the TPL buffer can hold.
///
/// The clamp is `MAX_TPL_FRAME_IDX - 1`, one below the bound
/// [`TplParams::tpl_stats_ready`] uses, so the last representable index is
/// never handed out as a length.
#[must_use]
pub fn get_gop_length(gf_group_size: i32) -> i32 {
    gf_group_size.min(MAX_TPL_FRAME_IDX - 1)
}

/// `eval_gop_length`'s verdict (tpl_model.c:1868, static).
///
/// C returns a bare `int` in `{0, 1, 2}` whose meaning is documented only in
/// a comment; the three outcomes are genuinely different actions, so they are
/// an enum here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GopLengthVerdict {
    /// Shorten the GF interval (C's 0).
    Shorten,
    /// Keep the GF interval (C's 1).
    Keep,
    /// Undecided — redo the TPL stats calculation (C's 2).
    Redo,
}

impl GopLengthVerdict {
    /// C's `int` encoding, for the differential.
    #[must_use]
    pub fn as_c_int(self) -> i32 {
        match self {
            Self::Shorten => 0,
            Self::Keep => 1,
            Self::Redo => 2,
        }
    }
}

/// `eval_gop_length` (tpl_model.c:1868, static) — decide whether the GF
/// interval TPL just evaluated should stand.
///
/// `beta[0]` is the base-layer ARF's dependency factor and `beta[1]` the
/// intermediate ARF's. A larger `gop_eval` means a cheaper, more approximate
/// evaluation, so the thresholds relax as it rises: mode 1 demands a 0.7
/// margin *and* `beta[0] > 3.0`, mode 2 has a middle band that refuses to
/// decide, and mode 3 is a single `beta[0] > 1.1` test. Anything else is
/// [`GopLengthVerdict::Redo`].
///
/// The `>=` in the margin tests and the strict `>` in the level tests are
/// C's, and they differ between arms (mode 2's shorten arm uses `<=` on the
/// level but `<` on the margin) — the asymmetry is load-bearing at the
/// boundaries.
#[must_use]
pub fn eval_gop_length(beta: [f64; 2], gop_eval: i32) -> GopLengthVerdict {
    match gop_eval {
        1 => {
            if beta[0] >= beta[1] + 0.7 && beta[0] > 3.0 {
                GopLengthVerdict::Keep
            } else {
                GopLengthVerdict::Shorten
            }
        }
        2 => {
            if beta[0] >= beta[1] + 0.4 && beta[0] > 1.6 {
                GopLengthVerdict::Keep
            } else if beta[0] < beta[1] + 0.1 || beta[0] <= 1.4 {
                GopLengthVerdict::Shorten
            } else {
                GopLengthVerdict::Redo
            }
        }
        3 => {
            if beta[0] > 1.1 {
                GopLengthVerdict::Keep
            } else {
                GopLengthVerdict::Shorten
            }
        }
        _ => GopLengthVerdict::Redo,
    }
}

/// `skip_tpl_for_frame` (tpl_model.c:1908, static) — whether a GOP frame is
/// skipped by the TPL pass.
///
/// Three independent skips: overlay frames never get stats (they code
/// nothing new); with `approx_gop_eval` the pass is limited to the ARF layers
/// near the base and to frames inside [`get_gop_length`]; and
/// `reduce_num_frames` additionally drops leaf frames inside that length.
///
/// Note the last two use the gop-length bound in *opposite* directions
/// (`frame_idx >= gop_length` skips, `frame_idx < gop_length` skips) — they
/// are not two spellings of one rule.
#[must_use]
pub fn skip_tpl_for_frame(
    gf_group_size: i32,
    frame_idx: i32,
    update_type: FrameUpdateType,
    layer_depth: i32,
    gop_eval: i32,
    approx_gop_eval: bool,
    reduce_num_frames: bool,
) -> bool {
    let num_arf_layers = if gop_eval == 2 { 3 } else { 2 };
    let gop_length = get_gop_length(gf_group_size);

    if matches!(
        update_type,
        FrameUpdateType::IntnlOverlay | FrameUpdateType::Overlay
    ) {
        return true;
    }
    if approx_gop_eval && (layer_depth > num_arf_layers || frame_idx >= gop_length) {
        return true;
    }
    reduce_num_frames && update_type == FrameUpdateType::Lf && frame_idx < gop_length
}

/// `is_alike_mv` (tpl_model.c:345, static) — whether `candidate` is close
/// enough to an already-queued search centre to be dropped.
///
/// The threshold is picked by `skip_alike_starting_mv` from
/// `{1, 8 << 3, 16 << 3}` in 1/8-pel units, i.e. "bit-identical", "within 8
/// full pels", "within 16 full pels". Both components must be strictly inside
/// the threshold, so at level 0 the test is exact equality.
///
/// # Panics
/// If `skip_alike_starting_mv` is not in `0..=2` — C indexes a 3-entry array
/// with it and has no bound of its own.
#[must_use]
pub fn is_alike_mv(candidate: Mv, centers: &[Mv], skip_alike_starting_mv: usize) -> bool {
    const MV_DIFF_THR: [i32; 3] = [1, 8 << 3, 16 << 3];
    let thr = MV_DIFF_THR[skip_alike_starting_mv];
    centers.iter().any(|c| {
        (i32::from(c.col) - i32::from(candidate.col)).abs() < thr
            && (i32::from(c.row) - i32::from(candidate.row)).abs() < thr
    })
}

/// `compare_sad` (tpl_model.c:336, static) — the `qsort` comparator that
/// orders TPL's motion-search start candidates by SAD.
///
/// C subtracts the two `int` SADs and returns the sign, which overflows for
/// SADs more than `INT_MAX` apart; [`Ord::cmp`] on the values themselves
/// cannot. SADs are non-negative sums bounded by `255 * 16 * 16 * 255`, so
/// the difference never approaches the overflow, and the two agree on every
/// input the encoder produces — proven over the full sign matrix in the
/// differential.
#[must_use]
pub fn compare_sad(a: i32, b: i32) -> Ordering {
    a.cmp(&b)
}

/// `convert_length_to_bsize` (tpl_model.h:43) resolved to the four block
/// dimensions `tpl_model_update_b` reads off it.
///
/// C converts `MI_SIZE << block_mis_log2` to a `BLOCK_SIZE` and then reads
/// `mi_size_wide_log2`, `mi_size_wide` and `mi_size_high` back out of it,
/// which for a square block is exactly the shift it started from. The LUT
/// round-trip is skipped here, with one exception that has to be kept: for a
/// length outside `{4, 8, 16, 32, 64}` C's `default:` arm returns
/// `BLOCK_16X16` (its `assert` is compiled out under `-DNDEBUG`), so the
/// dimensions stop tracking the shift.
///
/// Returns `(bw, bh, mi_width, mi_height)`.
fn tpl_block_dims(block_mis_log2: u8) -> (i32, i32, i32, i32) {
    if block_mis_log2 <= 4 {
        let side = MI_SIZE << block_mis_log2;
        let mi = 1 << block_mis_log2;
        (side, side, mi, mi)
    } else {
        // C's `default: return BLOCK_16X16`.
        (16, 16, 4, 4)
    }
}

/// `get_fullmv_from_mv` (`av1/common/mv.h:79`) — 1/8-pel to full-pel, via
/// `GET_MV_RAWPEL(x) = (x + 3 + (x >= 0)) >> 3`.
///
/// That is round-half-away-from-zero, not a plain `>> 3`: the `+ (x >= 0)`
/// term shifts the tie for non-negative inputs so that +4 rounds to 1 while
/// -4 rounds to 0.
#[inline]
fn get_fullmv_from_mv(mv: Mv) -> (i32, i32) {
    let raw = |x: i32| (x + 3 + i32::from(x >= 0)) >> 3;
    (raw(i32::from(mv.row)), raw(i32::from(mv.col)))
}

impl TplParams {
    /// `tpl_model_store` (tpl_model.c:1290, static) — write one block's TPL
    /// result into the grid, flooring eleven of its fields at 1.
    ///
    /// The floors are why every consumer downstream can divide by
    /// `recrf_dist` or take `log(recrf_dist)` without a guard — see the
    /// reachability note on [`Self::get_frame_importance`]. Note which fields
    /// are *not* floored: `srcrf_dist` is but `intra_dist` is not,
    /// `recrf_dist` is but `recrf_sse` is not, and `mc_dep_*` are left alone
    /// because propagation has not run yet.
    pub fn tpl_model_store(
        grid: &mut [TplDepStats],
        mi_row: i32,
        mi_col: i32,
        stride: i32,
        src: &TplDepStats,
        block_mis_log2: u8,
    ) {
        let index = tpl_ptr_pos(mi_row, mi_col, stride, block_mis_log2) as usize;
        let cell = &mut grid[index];
        *cell = *src;
        cell.intra_cost = cell.intra_cost.max(1);
        cell.inter_cost = cell.inter_cost.max(1);
        cell.srcrf_dist = cell.srcrf_dist.max(1);
        cell.srcrf_sse = cell.srcrf_sse.max(1);
        cell.recrf_dist = cell.recrf_dist.max(1);
        cell.srcrf_rate = cell.srcrf_rate.max(1);
        cell.recrf_rate = cell.recrf_rate.max(1);
        cell.cmp_recrf_dist[0] = cell.cmp_recrf_dist[0].max(1);
        cell.cmp_recrf_dist[1] = cell.cmp_recrf_dist[1].max(1);
        cell.cmp_recrf_rate[0] = cell.cmp_recrf_rate[0].max(1);
        cell.cmp_recrf_rate[1] = cell.cmp_recrf_rate[1].max(1);
    }

    /// `tpl_model_update_b` (tpl_model.c:1204, static) — push one block's
    /// dependency cost back into reference `ref_idx`.
    ///
    /// This is the propagation TPL is named for. The block at
    /// `(mi_row, mi_col)` of frame `frame_idx` predicted itself from a
    /// reference frame at a motion vector; the cost it *saved* by doing so
    /// (`recrf_dist - srcrf_dist`, plus whatever was already propagated into
    /// it) is credited back to the reference's grid cells, split by how much
    /// of each cell the motion-compensated block actually overlapped. Four
    /// cells at most, because a full-pel MV can straddle two rows and two
    /// columns of the grid.
    ///
    /// Three things C does here that a tidier rewrite silently changes:
    ///
    /// 1. **The source cell is located with frame 0's stride**, not
    ///    `frame_idx`'s: C binds `tpl_frame = tpl_data->tpl_frame` and then
    ///    indexes `tpl_ptr` (which is `frame_idx`'s grid) with
    ///    `tpl_frame->stride`. Every TPL frame in a GOP is the same video
    ///    frame size, so the two are equal in the encoder — but they are two
    ///    different reads and this reproduces the one C makes.
    /// 2. **`ref_map_index < 0` is checked AFTER the reference frame's stats
    ///    pointer has already been loaded from it.** C computes
    ///    `&tpl_frame[negative]` and dereferences it before the guard, which
    ///    is undefined behaviour; the guard is hoisted above the load here.
    ///    The two agree on every input where C is defined at all.
    /// 3. **`mc_dep_dist` is scaled in `f64` and truncated back to `i64`**,
    ///    while `cur_dep_dist` beside it stays integral. MEASURED: the
    ///    obvious integer rewrite,
    ///    `mc_dep_dist * (recrf_dist - srcrf_dist) / recrf_dist`, agrees with
    ///    C on effectively every small input — substituting it left the whole
    ///    2400-trial randomized differential green, because both forms
    ///    truncate the same real number and disagree only when float rounding
    ///    crosses an integer boundary. It stops agreeing when the integer
    ///    product overflows `i64`, which `mc_dep_dist` (a distortion summed
    ///    over a whole dependency chain, 1e12..1e15 at 4K) against a
    ///    `recrf_dist` of ~1e7 does. That is gated separately by
    ///    `tpl_model_update_mc_dep_rescale_matches_c_at_large_magnitudes`.
    ///
    /// # The rate shifts, and why they cannot be gated
    /// `srcrf_rate`/`recrf_rate` are `i32`, and C shifts them by
    /// `TPL_DEP_COST_SCALE_LOG2` *before* widening to `int64_t`. Writing
    /// `i64::from(x) << 4` instead would differ only if `x << 4` overflowed
    /// `int` — where C is undefined, so a differential there would be
    /// comparing two definitions of nothing. It cannot: every rate in a TPL
    /// cell comes from [`rate_estimator`], whose output is
    /// `(1 + eob * (msb(|level| + 1) + 2)) << AV1_PROB_COST_SHIFT` with
    /// `eob <= 256` for TPL's 16x16 transform, i.e. under 2.5e6, so `<< 4`
    /// stays under 4e7. C's spelling is kept because it is C's; the
    /// substitution is recorded here as **provably unobservable** rather than
    /// left as an untested difference.
    ///
    /// The compound arm swaps in `cmp_recrf_*[!ref]` — the *other*
    /// reference's contribution — as the "source" cost, so each reference is
    /// charged only for what it added over the other one.
    pub fn tpl_model_update_b(
        &mut self,
        mi_row: i32,
        mi_col: i32,
        block_mis_log2: u8,
        frame_idx: usize,
        ref_idx: usize,
    ) {
        // (1) C's `tpl_frame->stride` — frame 0's, not frame_idx's.
        let frame0_stride = self.frames[0].stride;
        let src_index = tpl_ptr_pos(mi_row, mi_col, frame0_stride, block_mis_log2) as usize;
        let this = self.frames[frame_idx].stats[src_index];

        let is_compound = this.ref_frame_index[1] >= 0;
        let Ok(ref_frame_index) = usize::try_from(this.ref_frame_index[ref_idx]) else {
            return; // C: `if (tpl_stats_ptr->ref_frame_index[ref] < 0) return;`
        };
        // (2) hoisted above the reference-frame load; C checks it after.
        let Ok(ref_frame) = usize::try_from(self.frames[frame_idx].ref_map_index[ref_frame_index])
        else {
            return;
        };

        let (full_mv_row, full_mv_col) = get_fullmv_from_mv(this.mv[ref_frame_index]);
        let ref_pos_row = mi_row * MI_SIZE + full_mv_row;
        let ref_pos_col = mi_col * MI_SIZE + full_mv_col;

        let (bw, bh, mi_width, mi_height) = tpl_block_dims(block_mis_log2);
        let pix_num = i64::from(bw) * i64::from(bh);

        let grid_pos_row_base = round_floor(ref_pos_row, bh) * bh;
        let grid_pos_col_base = round_floor(ref_pos_col, bw) * bw;

        let other = 1 - ref_idx;
        let srcrf_dist = if is_compound {
            this.cmp_recrf_dist[other]
        } else {
            this.srcrf_dist
        };
        // C shifts the i32 rate and only then widens, so the shift can
        // overflow `int` exactly as it does in C.
        let srcrf_rate = i64::from(if is_compound {
            this.cmp_recrf_rate[other] << TPL_DEP_COST_SCALE_LOG2
        } else {
            this.srcrf_rate << TPL_DEP_COST_SCALE_LOG2
        });

        let cur_dep_dist = this.recrf_dist - srcrf_dist;
        // (3) the fractional scale is taken in f64 and truncated, as C does.
        let mc_dep_dist = (this.mc_dep_dist as f64
            * ((this.recrf_dist - srcrf_dist) as f64 / this.recrf_dist as f64))
            as i64;
        let delta_rate = i64::from(this.recrf_rate << TPL_DEP_COST_SCALE_LOG2) - srcrf_rate;
        let mc_dep_rate = delta_rate_cost(
            this.mc_dep_rate,
            this.recrf_dist,
            srcrf_dist,
            i32::try_from(pix_num).expect("pix_num fits in an int"),
        );

        let (ref_mi_rows, ref_mi_cols, ref_stride) = {
            let f = &self.frames[ref_frame];
            (f.mi_rows, f.mi_cols, f.stride)
        };

        for block in 0..4 {
            let grid_pos_row = grid_pos_row_base + bh * (block >> 1);
            let grid_pos_col = grid_pos_col_base + bw * (block & 1);
            if grid_pos_row < 0
                || grid_pos_row >= ref_mi_rows * MI_SIZE
                || grid_pos_col < 0
                || grid_pos_col >= ref_mi_cols * MI_SIZE
            {
                continue;
            }
            let overlap_area = i64::from(get_overlap_area(
                grid_pos_row,
                grid_pos_col,
                ref_pos_row,
                ref_pos_col,
                bw,
                bh,
            ));
            let ref_mi_row = round_floor(grid_pos_row, bh) * mi_height;
            let ref_mi_col = round_floor(grid_pos_col, bw) * mi_width;
            debug_assert_eq!(1 << block_mis_log2, mi_height);
            debug_assert_eq!(1 << block_mis_log2, mi_width);
            let des = tpl_ptr_pos(ref_mi_row, ref_mi_col, ref_stride, block_mis_log2) as usize;
            let des_stats = &mut self.frames[ref_frame].stats[des];
            des_stats.mc_dep_dist += ((cur_dep_dist + mc_dep_dist) * overlap_area) / pix_num;
            des_stats.mc_dep_rate += ((delta_rate + mc_dep_rate) * overlap_area) / pix_num;
        }
    }

    /// `tpl_model_update` (tpl_model.c:1280, static) — propagate one block
    /// into both of its references.
    ///
    /// C derives the block size from `tpl_stats_block_mis_log2` on every
    /// call; that derivation lives in [`tpl_block_dims`] here, so this is
    /// just the two-reference loop.
    pub fn tpl_model_update(&mut self, mi_row: i32, mi_col: i32, frame_idx: usize) {
        let shift = self.tpl_stats_block_mis_log2;
        self.tpl_model_update_b(mi_row, mi_col, shift, frame_idx, 0);
        self.tpl_model_update_b(mi_row, mi_col, shift, frame_idx, 1);
    }
}

// ===========================================================================
// Frame-level propagation, the per-superblock rdmult, and the MV entropy.
// ===========================================================================

/// `BLOCK_16X16`'s mi dimensions — `mi_size_wide[BLOCK_16X16]` and
/// `mi_size_high[BLOCK_16X16]`, both 4. `av1_tpl_rdmult_setup` and
/// `av1_tpl_rdmult_setup_sb` pin their aggregation block to 16x16 regardless
/// of the TPL grid's own decimation, so this is not `1 << block_mis_log2`.
const RDMULT_BLOCK_MI: i32 = 4;

/// `av1_pixels_to_mi` (encoder/encoder.h:4399) — pixels to mi units,
/// rounding the width up to a multiple of 8 first.
///
/// The `ALIGN_POWER_OF_TWO(pixels, 3)` is not a rounding convenience: it
/// makes an odd-width frame occupy a whole 8-pixel mi pair, which is what the
/// chroma planes need at 4:2:0.
#[must_use]
pub fn pixels_to_mi(pixels: i32) -> i32 {
    ((pixels + 7) & !7) >> 2
}

impl TplParams {
    /// `mc_flow_synthesizer` (tpl_model.c:1611, static) — run the backward
    /// propagation over every block of one frame.
    ///
    /// Frame 0 is skipped outright (`if (!frame_idx) return;`): it is the
    /// GOP's own reference, so it has nothing further back to propagate into.
    ///
    /// **The walk step comes from `tpl_bsize_1d`, not from
    /// `tpl_stats_block_mis_log2`.** C converts `tpl_bsize_1d` to a
    /// `BLOCK_SIZE` and reads `mi_size_wide`/`mi_size_high` off *that*, then
    /// asserts (under `NDEBUG`, so not at all in the oracle build) that the
    /// two agree. They do in every libaom build, where both are pinned by
    /// `set_tpl_stats_block_size`. Reproducing the read C makes rather than
    /// the one it asserts is the difference the differential can see.
    ///
    /// `mi_rows`/`mi_cols` are the *caller's* walk bounds, which
    /// `av1_tpl_setup_stats` takes from the common frame geometry — not from
    /// the TPL frame's own `mi_rows`/`mi_cols`.
    pub fn mc_flow_synthesizer(&mut self, frame_idx: usize, mi_rows: i32, mi_cols: i32) {
        if frame_idx == 0 {
            return;
        }
        // C: convert_length_to_bsize(tpl_bsize_1d), then mi_size_{high,wide}.
        // For a square BLOCK_NxN that is N / MI_SIZE; the invalid-length arm
        // falls back to BLOCK_16X16 exactly as `tpl_block_dims` does.
        let (_, _, mi_width, mi_height) = tpl_block_dims(match self.tpl_bsize_1d {
            4 => 0,
            8 => 1,
            16 => 2,
            32 => 3,
            64 => 4,
            // C's `default: return BLOCK_16X16`.
            _ => 2,
        });
        let mut mi_row = 0;
        while mi_row < mi_rows {
            let mut mi_col = 0;
            while mi_col < mi_cols {
                self.tpl_model_update(mi_row, mi_col, frame_idx);
                mi_col += mi_width;
            }
            mi_row += mi_height;
        }
    }

    /// `av1_tpl_rdmult_setup` (tpl_model.c:2213) — the per-16x16 RD
    /// multiplier scaling this frame's TPL stats imply.
    ///
    /// For each 16x16 block, `rk` is the ratio of the block's own coding cost
    /// to that cost plus everything propagated into it. A block that many
    /// others depend on has a large `mc_dep_cost`, so a small `rk`, so a
    /// smaller rdmult — it is coded more finely. `cpi->rd.r0` normalizes
    /// against the frame average, and the `+ 1.2` keeps the factor away from
    /// zero.
    ///
    /// Returns `(num_rows, num_cols, factors)` — `factors` is row-major over
    /// the 16x16 grid. C writes into `cpi->tpl_rdmult_scaling_factors`, which
    /// the encoder allocated at exactly this size; returning the buffer keeps
    /// the sizing rule in one place instead of splitting it between allocator
    /// and writer.
    ///
    /// Returns `None` when the frame has no valid TPL stats, which is C's
    /// early `return` leaving the previous factors in place.
    ///
    /// Two geometry details: the column count comes from the **superres
    /// upscaled** width (a superres frame is coded narrow but its TPL grid is
    /// full width), while the row count comes from `mi_params.mi_rows`; and
    /// the inner accumulation skips mi positions past either bound, so a
    /// partial 16x16 block at the right or bottom edge averages only the
    /// cells that exist.
    ///
    /// That edge clip is **unreachable at the production grid decimation**.
    /// The block is fixed at 16x16 (4 mi) but the inner loop steps by
    /// `1 << tpl_stats_block_mis_log2`, which is also 4 — so there is exactly
    /// one mi position per block and `(num_rows - 1) * 4 < mi_rows` by the
    /// definition of `num_rows`. MEASURED: with shift 2 alone, deleting the
    /// clip left the differential green; it only fires once the differential
    /// also sweeps shift 0 and 1, which it now does.
    #[must_use]
    pub fn tpl_rdmult_setup(
        &self,
        gf_frame_index: usize,
        superres_upscaled_width: i32,
        mi_rows: i32,
        r0: f64,
    ) -> Option<(i32, i32, Vec<f64>)> {
        let tpl_frame = self.frames.get(gf_frame_index)?;
        if !tpl_frame.is_valid {
            return None;
        }
        let tpl_stride = tpl_frame.stride;
        let mi_cols_sr = pixels_to_mi(superres_upscaled_width);
        let num_cols = (mi_cols_sr + RDMULT_BLOCK_MI - 1) / RDMULT_BLOCK_MI;
        let num_rows = (mi_rows + RDMULT_BLOCK_MI - 1) / RDMULT_BLOCK_MI;
        const C: f64 = 1.2;
        let shift = self.tpl_stats_block_mis_log2;
        let step = 1i32 << shift;

        let mut factors = vec![0.0f64; (num_rows.max(0) * num_cols.max(0)) as usize];
        for row in 0..num_rows {
            for col in 0..num_cols {
                let mut intra_cost = 0.0f64;
                let mut mc_dep_cost = 0.0f64;
                let mut mi_row = row * RDMULT_BLOCK_MI;
                while mi_row < (row + 1) * RDMULT_BLOCK_MI {
                    let mut mi_col = col * RDMULT_BLOCK_MI;
                    while mi_col < (col + 1) * RDMULT_BLOCK_MI {
                        if mi_row >= mi_rows || mi_col >= mi_cols_sr {
                            mi_col += step;
                            continue;
                        }
                        let this_stats = &tpl_frame.stats
                            [tpl_ptr_pos(mi_row, mi_col, tpl_stride, shift) as usize];
                        let mc_dep_delta = rdcost_i64_rate(
                            tpl_frame.base_rdmult,
                            this_stats.mc_dep_rate,
                            this_stats.mc_dep_dist,
                        );
                        let scaled = (this_stats.recrf_dist << RDDIV_BITS) as f64;
                        intra_cost += scaled;
                        mc_dep_cost += scaled + mc_dep_delta as f64;
                        mi_col += step;
                    }
                    mi_row += step;
                }
                let rk = intra_cost / mc_dep_cost;
                factors[(row * num_cols + col) as usize] = rk / r0 + C;
            }
        }
        Some((num_rows, num_cols, factors))
    }

    /// `av1_compute_mv_difference` (tpl_model.c:2639) — the smallest MV
    /// residual for the cell at `(row, col)`, against its up and left
    /// neighbours.
    ///
    /// This is a *prediction*, so what it returns is a difference when a
    /// neighbour predicts better than zero, and the raw MV otherwise. The
    /// comparison is `up_error < left_error && up_error < |current|`, i.e.
    /// strict on both sides, so a tie between the two neighbours — or with
    /// coding the MV outright — falls through to the raw MV.
    ///
    /// Missing neighbours are represented by `i32::MAX` errors rather than by
    /// an `Option`, because C compares the two errors against each other and
    /// the sentinel has to lose both comparisons; an `Option` would need the
    /// same three-way logic spelled out twice.
    ///
    /// The MV read is `mv[ref_frame_index[0]]` — the *first* reference only,
    /// even for a compound cell.
    #[must_use]
    pub fn compute_mv_difference(
        tpl_frame: &TplDepFrame,
        row: i32,
        col: i32,
        step: i32,
        tpl_stride: i32,
        right_shift: u8,
    ) -> Mv {
        let cell = |r: i32, c: i32| -> Mv {
            let s = &tpl_frame.stats[tpl_ptr_pos(r, c, tpl_stride, right_shift) as usize];
            s.mv[s.ref_frame_index[0] as usize]
        };
        let current_mv = cell(row, col);
        let current_mv_magnitude =
            i32::from(current_mv.row).abs() + i32::from(current_mv.col).abs();

        let mut up_error = i32::MAX;
        let mut up_mv_diff = Mv::default();
        if row - step >= 0 {
            let up = cell(row - step, col);
            up_mv_diff = Mv::new(current_mv.row - up.row, current_mv.col - up.col);
            up_error = i32::from(up_mv_diff.row).abs() + i32::from(up_mv_diff.col).abs();
        }

        let mut left_error = i32::MAX;
        let mut left_mv_diff = Mv::default();
        if col - step >= 0 {
            let left = cell(row, col - step);
            left_mv_diff = Mv::new(current_mv.row - left.row, current_mv.col - left.col);
            left_error = i32::from(left_mv_diff.row).abs() + i32::from(left_mv_diff.col).abs();
        }

        if up_error < left_error && up_error < current_mv_magnitude {
            up_mv_diff
        } else if left_error < up_error && left_error < current_mv_magnitude {
            left_mv_diff
        } else {
            current_mv
        }
    }

    /// `av1_tpl_compute_frame_mv_entropy` (tpl_model.c:2682) — the modelled
    /// bit cost of one frame's motion field.
    ///
    /// A first-order entropy over the histogram of MV residuals: build the
    /// distribution of [`Self::compute_mv_difference`] outputs over the grid,
    /// then charge `-log2(p)` per occurrence.
    ///
    /// # Two upstream behaviours reproduced verbatim
    /// 1. **`count_col` is indexed by `mv.row`, not `mv.col`.** C writes
    ///    `count_col[clamp(mv.as_mv.row, 0, 499)] += 1`. The column histogram
    ///    is therefore a duplicate of the row histogram and the result is
    ///    exactly `2 * rate_row`. This is a copy-paste bug in libaom v3.14.1;
    ///    reproducing it is the contract, and "fixing" it fails the
    ///    differential.
    /// 2. **Negative residuals all collapse into bin 0**, because the clamp
    ///    is `[0, 499]` rather than an offset range. Half the motion field is
    ///    therefore counted as one symbol.
    ///
    /// Both are why this function is only reachable under
    /// `CONFIG_BITRATE_ACCURACY` analysis, never on the byte-exact path.
    #[must_use]
    pub fn tpl_compute_frame_mv_entropy(tpl_frame: &TplDepFrame, right_shift: u8) -> f64 {
        if !tpl_frame.is_valid {
            return 0.0;
        }
        const BINS: usize = 500;
        let mut count_row = [0i32; BINS];
        let mut count_col = [0i32; BINS];
        let mut n = 0i32;
        let tpl_stride = tpl_frame.stride;
        let step = 1i32 << right_shift;

        let mut row = 0;
        while row < tpl_frame.mi_rows {
            let mut col = 0;
            while col < tpl_frame.mi_cols {
                let mv =
                    Self::compute_mv_difference(tpl_frame, row, col, step, tpl_stride, right_shift);
                let bin = i32::from(mv.row).clamp(0, 499) as usize;
                count_row[bin] += 1;
                // (1) C indexes count_col with .row too. Not a typo here.
                count_col[bin] += 1;
                n += 1;
                col += step;
            }
            row += step;
        }

        let mut rate_row = 0.0f64;
        let mut rate_col = 0.0f64;
        for i in 0..BINS {
            if count_row[i] != 0 {
                let p = f64::from(count_row[i]) / f64::from(n);
                rate_row += f64::from(count_row[i]) * -p.log2();
            }
            if count_col[i] != 0 {
                let p = f64::from(count_col[i]) / f64::from(n);
                rate_col += f64::from(count_col[i]) * -p.log2();
            }
        }
        rate_row + rate_col
    }
}

/// The encoder scalars `av1_tpl_rdmult_setup_sb` reads, gathered so the
/// signature stays readable — C reaches into `AV1_COMP`, `AV1_COMMON`,
/// `MACROBLOCK` and `AV1EncoderConfig` for these one at a time.
#[derive(Clone, Copy, Debug)]
pub struct TplRdmultSbParams {
    /// `cpi->gf_frame_index`.
    pub gf_frame_index: usize,
    /// `gf_group->update_type[gf_frame_index]`.
    pub update_type: FrameUpdateType,
    /// `gf_group->layer_depth[gf_frame_index]`, before the clamp to 6.
    pub layer_depth: i32,
    /// `cpi->ppi->p_rc.gfu_boost`, before the clamp to 15 after /100.
    pub gfu_boost: i32,
    /// `cm->current_frame.frame_type`.
    pub frame_type: crate::rd::FrameType,
    /// `cpi->oxcf.q_cfg.aq_mode`. Non-zero disables the whole function.
    pub aq_mode: i32,
    /// `cm->superres_scale_denominator`.
    pub superres_scale_denominator: i32,
    /// `cm->superres_upscaled_width`.
    pub superres_upscaled_width: i32,
    /// `cm->mi_params.mi_rows`.
    pub mi_rows: i32,
    /// `cm->quant_params.base_qindex`.
    pub base_qindex: i32,
    /// `cm->quant_params.y_dc_delta_q`.
    pub y_dc_delta_q: i32,
    /// `x->rdmult_delta_qindex`.
    pub rdmult_delta_qindex: i32,
    /// `cm->seq_params->bit_depth`.
    pub bit_depth: u8,
    /// `cpi->oxcf.q_cfg.use_fixed_qp_offsets`.
    pub use_fixed_qp_offsets: bool,
    /// `is_stat_consumption_stage(cpi)` (encoder.h:4137).
    pub is_stat_consumption_stage: bool,
    /// `cpi->oxcf.tune_cfg.tuning`.
    pub tuning: crate::rd::TuneMetric,
    /// `cpi->oxcf.mode`.
    pub mode: crate::rd::EncMode,
    /// `mi_size_wide[sb_size]`.
    pub sb_mi_width: i32,
    /// `mi_size_high[sb_size]`.
    pub sb_mi_height: i32,
}

/// `coded_to_superres_mi` (`encoder/rdopt.h:169`) — an mi column in coded
/// space, mapped to superres-upscaled space.
///
/// `(mi_col * denom + SCALE_NUMERATOR / 2) / SCALE_NUMERATOR` with
/// `SCALE_NUMERATOR = 8`, i.e. rounded to nearest. Integer division on a
/// non-negative numerator, so this is a plain `/`.
#[must_use]
pub fn coded_to_superres_mi(mi_col: i32, denom: i32) -> i32 {
    /// `SCALE_NUMERATOR` (`av1/common/enums.h`).
    const SCALE_NUMERATOR: i32 = 8;
    (mi_col * denom + SCALE_NUMERATOR / 2) / SCALE_NUMERATOR
}

impl TplParams {
    /// `av1_tpl_rdmult_setup_sb` (tpl_model.c:2264) — turn the frame-level
    /// TPL factors into the per-superblock ones the RD search actually uses.
    ///
    /// The mechanism: over the 16x16 blocks this superblock covers, take the
    /// geometric mean of [`Self::tpl_rdmult_setup`]'s factors (via
    /// `log_sum / base_block_count`), and scale it so the superblock's own
    /// rdmult change — `new_rdmult / orig_rdmult`, the ratio the delta-q
    /// decision implies — is preserved on average. So the per-block factors
    /// keep their *relative* spread while their mean tracks the superblock's
    /// quantizer.
    ///
    /// Five early returns, in C's order, and they are not interchangeable:
    /// past the TPL buffer, no valid stats, a frame type TPL does not model
    /// (`is_frame_tpl_eligible`: only ARF/GF/KF), or any adaptive-quantization
    /// mode — because AQ writes the same output buffer.
    ///
    /// # The row/column index swap
    /// C computes the block window as `row` from `mi_row / num_mi_w` and
    /// `col` from `mi_col_sr / num_mi_h` — **width for the row, height for
    /// the column**. Both are 4 for the fixed 16x16 aggregation block, so it
    /// makes no difference and is reproduced as written rather than
    /// "corrected"; the names are C's, mismatched.
    ///
    /// Returns `None` on any early return, meaning the caller's existing
    /// `tpl_sb_rdmult_scaling_factors` are left alone. On `Some`, the result
    /// is the list of `(index, factor)` pairs written — a sparse update,
    /// because C only touches this superblock's window.
    #[must_use]
    pub fn tpl_rdmult_setup_sb(
        &self,
        p: TplRdmultSbParams,
        frame_factors: &[f64],
        mi_row: i32,
        mi_col: i32,
    ) -> Option<Vec<(usize, f64)>> {
        let boost_index = (p.gfu_boost / 100).min(15);
        let layer_depth = p.layer_depth.min(6);

        if i32::try_from(p.gf_frame_index).ok()? >= MAX_TPL_FRAME_IDX {
            return None;
        }
        if !self.frames.get(p.gf_frame_index)?.is_valid {
            return None;
        }
        // is_frame_tpl_eligible (encoder.h:4378).
        if !matches!(
            p.update_type,
            FrameUpdateType::Arf | FrameUpdateType::Gf | FrameUpdateType::Kf
        ) {
            return None;
        }
        // NO_AQ == 0.
        if p.aq_mode != 0 {
            return None;
        }

        let mi_col_sr = coded_to_superres_mi(mi_col, p.superres_scale_denominator);
        let mi_cols_sr = pixels_to_mi(p.superres_upscaled_width);
        let sb_mi_width_sr = coded_to_superres_mi(p.sb_mi_width, p.superres_scale_denominator);

        let num_mi_w = RDMULT_BLOCK_MI;
        let num_mi_h = RDMULT_BLOCK_MI;
        let num_cols = (mi_cols_sr + num_mi_w - 1) / num_mi_w;
        let num_rows = (p.mi_rows + num_mi_h - 1) / num_mi_h;
        let num_bcols = (sb_mi_width_sr + num_mi_w - 1) / num_mi_w;
        let num_brows = (p.sb_mi_height + num_mi_h - 1) / num_mi_h;

        // C's mismatched divisors: width for the row, height for the column.
        let row_start = mi_row / num_mi_w;
        let col_start = mi_col_sr / num_mi_h;

        let mut base_block_count = 0.0f64;
        let mut log_sum = 0.0f64;
        for row in row_start..num_rows.min(row_start + num_brows) {
            for col in col_start..num_cols.min(col_start + num_bcols) {
                log_sum += frame_factors[(row * num_cols + col) as usize].ln();
                base_block_count += 1.0;
            }
        }

        let orig_qindex_rdmult = p.base_qindex + p.y_dc_delta_q;
        let orig_rdmult = crate::rd::av1_compute_rd_mult(
            orig_qindex_rdmult,
            p.bit_depth,
            p.update_type,
            layer_depth,
            boost_index,
            p.frame_type,
            p.use_fixed_qp_offsets,
            p.is_stat_consumption_stage,
            p.tuning,
            p.mode,
        );
        let new_qindex_rdmult = p.base_qindex + p.rdmult_delta_qindex + p.y_dc_delta_q;
        let new_rdmult = crate::rd::av1_compute_rd_mult(
            new_qindex_rdmult,
            p.bit_depth,
            p.update_type,
            layer_depth,
            boost_index,
            p.frame_type,
            p.use_fixed_qp_offsets,
            p.is_stat_consumption_stage,
            p.tuning,
            p.mode,
        );

        let scaling_factor = f64::from(new_rdmult) / f64::from(orig_rdmult);
        // `base_block_count` is 0 when the window is empty, which makes
        // `log_sum / 0` a NaN or an infinity; C does the same division and
        // `exp_bounded` passes NaN through. Reproduced rather than guarded.
        let scale_adj = exp_bounded(scaling_factor.ln() - log_sum / base_block_count);

        let mut out = Vec::new();
        for row in row_start..num_rows.min(row_start + num_brows) {
            for col in col_start..num_cols.min(col_start + num_bcols) {
                let index = (row * num_cols + col) as usize;
                out.push((index, scale_adj * frame_factors[index]));
            }
        }
        Some(out)
    }
}

/// `TplTxfmStats` (tpl_model.h:110) — the per-frame transform-coefficient
/// histogram `CONFIG_BITRATE_ACCURACY` builds.
///
/// Only [`Self::init`] is reachable in this build: every function that
/// *fills* this (`av1_record_tpl_txfm_block`,
/// `av1_accumulate_tpl_txfm_stats`, `av1_tpl_txfm_stats_update_abs_coeff_mean`)
/// is inside `#if CONFIG_BITRATE_ACCURACY`, which is 0. The struct is modelled
/// anyway because `av1_init_tpl_txfm_stats` is exported and IS called
/// unconditionally.
#[derive(Clone, Debug)]
pub struct TplTxfmStats {
    /// Whether `abs_coeff_mean` has been computed.
    pub ready: bool,
    /// Number of coefficients tracked. Always 256 (a 16x16 transform).
    pub coeff_num: i32,
    /// Number of transform blocks folded in.
    pub txfm_block_count: i32,
    /// Per-coefficient sum of `|level| / LOSSLESS_Q_STEP`.
    pub abs_coeff_sum: Vec<f64>,
    /// `abs_coeff_sum / txfm_block_count`.
    pub abs_coeff_mean: Vec<f64>,
}

impl Default for TplTxfmStats {
    fn default() -> Self {
        Self {
            ready: false,
            coeff_num: 256,
            txfm_block_count: 0,
            abs_coeff_sum: vec![0.0; 256],
            abs_coeff_mean: vec![0.0; 256],
        }
    }
}

impl TplTxfmStats {
    /// `av1_init_tpl_txfm_stats` (tpl_model.c:55) — reset the histogram.
    ///
    /// C sets `coeff_num = 256` **before** the two `memset`s and sizes them
    /// by it, so the clear always covers 256 entries regardless of what
    /// `coeff_num` was — the assignment order is load-bearing.
    pub fn init(&mut self) {
        self.ready = false;
        self.coeff_num = 256;
        self.txfm_block_count = 0;
        self.abs_coeff_sum.resize(256, 0.0);
        self.abs_coeff_mean.resize(256, 0.0);
        self.abs_coeff_sum.fill(0.0);
        self.abs_coeff_mean.fill(0.0);
    }
}
