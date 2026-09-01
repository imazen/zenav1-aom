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
