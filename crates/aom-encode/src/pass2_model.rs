//! The error and boost model of `av1/encoder/pass2_strategy.c` — the
//! arithmetic the 2-pass bit allocator makes every decision out of.
//!
//! Reached under `--passes=2` (and, for the decay rates, under
//! `--lag-in-frames > 0` with look-ahead stats). It has no direct bitstream
//! form: it decides the per-frame bit targets and boosts, which decide the
//! qindexes, which decide every coefficient.
//!
//! | Rust | C |
//! |---|---|
//! | [`FirstpassStats`] | `FIRSTPASS_STATS` (`firstpass.h:43`) |
//! | [`double_divide_check`] | `DOUBLE_DIVIDE_CHECK` (`firstpass.h:26`) |
//! | [`calculate_active_area`] | `calculate_active_area` (:61) |
//! | [`calculate_modified_err_new`] | `calculate_modified_err_new` (:73) |
//! | [`frame_max_bits`] | `frame_max_bits` (:154) |
//! | [`calc_correction_factor`] | `calc_correction_factor` (:171) |
//! | [`qbpm_enumerator`] | `qbpm_enumerator` (:288) |
//! | [`get_sr_decay_rate`] | `get_sr_decay_rate` (:392) |
//! | [`get_zero_motion_factor`] | `get_zero_motion_factor` (:415) |
//! | [`get_prediction_decay_rate`] | `get_prediction_decay_rate` (:422) |
//! | [`baseline_err_per_mb`] | `baseline_err_per_mb` (:574) |
//! | [`calc_frame_boost`] | `calc_frame_boost` (:590) |
//! | [`calc_kf_frame_boost`] | `calc_kf_frame_boost` (:622) |
//! | [`get_projected_gfu_boost`] | `get_projected_gfu_boost` (:653) |
//! | [`calculate_boost_bits`] | `calculate_boost_bits` (:836) |
//! | [`calculate_boost_factor`] | `calculate_boost_factor` (:861) |
//! | [`is_almost_static`] | `is_almost_static` (:999) |
//! | [`gfu_boost_projection_factor`] | `av1_get_gfu_boost_projection_factor` (`rc_utils.h:156`) |
//!
//! # Floating point is the contract here
//! Almost everything in this module is `double`. `pow`, `sqrt` and `rint` go
//! to the same libm C's do; `pow(x, y)` with a non-integral `y` is the one
//! place where that matters and it is exercised directly by
//! `calc_correction_factor`'s sweep. The oracle is compiled
//! `-ffp-contract=off` so neither side fuses a multiply-add — without that the
//! comparison would mean different things on x86 and aarch64.
//!
//! # `DOUBLE_DIVIDE_CHECK` is a bias, not a guard
//! C's macro adds (or subtracts) `1e-6` from EVERY divisor it wraps, not just
//! zero ones. It changes the quotient at every magnitude, so it cannot be
//! replaced with a zero test.
//!
//! # Differential coverage
//! `tests/pass2_model_diff.rs` — tier 1c against libaom's own
//! pass2_strategy.c, compiled verbatim by `shim/pass2_shim.c`. Seven of that
//! file's 87 definitions are exported and none of these is among them.

/// `FIRSTPASS_STATS` (`av1/encoder/firstpass.h:43`) — one frame's first-pass
/// measurements. Only `is_flash` is not a `double`.
///
/// The field ORDER matters: the oracle boundary passes the 29 doubles flat, in
/// declaration order, and the shim assigns them by name.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FirstpassStats {
    /// Frame number in display order, for single-frame stats.
    pub frame: f64,
    /// Weight from the intra and brightness factors.
    pub weight: f64,
    /// Intra prediction error.
    pub intra_error: f64,
    /// Average wavelet energy (DWT).
    pub frame_avg_wavelet_energy: f64,
    /// Best of intra and inter prediction error.
    pub coded_error: f64,
    /// Error when predicting from the golden/alt reference.
    pub sr_coded_error: f64,
    /// Error when predicting from the long-term reference.
    pub lt_coded_error: f64,
    /// Fraction of blocks coded inter.
    pub pcnt_inter: f64,
    /// Fraction of blocks with non-zero motion.
    pub pcnt_motion: f64,
    /// Fraction better predicted from the second reference.
    pub pcnt_second_ref: f64,
    /// Fraction with very low intra/inter error ("neutral").
    pub pcnt_neutral: f64,
    /// Fraction of blocks skipped in intra.
    pub intra_skip_pct: f64,
    /// Inactive (letterbox) rows, in MB units.
    pub inactive_zone_rows: f64,
    /// Inactive (pillarbox) columns, in MB units.
    pub inactive_zone_cols: f64,
    /// Mean row motion vector.
    pub mvr: f64,
    /// Mean absolute row motion vector.
    pub mvr_abs: f64,
    /// Mean column motion vector.
    pub mvc: f64,
    /// Mean absolute column motion vector.
    pub mvc_abs: f64,
    /// Variance of the row motion vectors.
    pub mvrv: f64,
    /// Variance of the column motion vectors.
    pub mvcv: f64,
    /// Net motion into (positive) or out of (negative) the frame.
    pub mv_in_out_count: f64,
    /// Count of blocks that used a new MV.
    pub new_mv_count: f64,
    /// Duration of this frame / collection.
    pub duration: f64,
    /// Number of frames these stats cover.
    pub count: f64,
    /// Standard deviation of the raw per-block error.
    pub raw_error_stdev: f64,
    /// Whether this frame is a flash. C stores it as `int64_t`.
    pub is_flash: i64,
    /// Estimated noise variance.
    pub noise_var: f64,
    /// Correlation coefficient with the previous frame.
    pub cor_coeff: f64,
    /// `log1p(intra_error)`.
    pub log_intra_error: f64,
    /// `log1p(coded_error)`.
    pub log_coded_error: f64,
}

impl FirstpassStats {
    /// The 29 `double` members, in C's declaration order — the layout the
    /// oracle boundary uses. `is_flash` is excluded because it is an
    /// `int64_t`.
    #[must_use]
    pub fn to_doubles(&self) -> [f64; 29] {
        [
            self.frame,
            self.weight,
            self.intra_error,
            self.frame_avg_wavelet_energy,
            self.coded_error,
            self.sr_coded_error,
            self.lt_coded_error,
            self.pcnt_inter,
            self.pcnt_motion,
            self.pcnt_second_ref,
            self.pcnt_neutral,
            self.intra_skip_pct,
            self.inactive_zone_rows,
            self.inactive_zone_cols,
            self.mvr,
            self.mvr_abs,
            self.mvc,
            self.mvc_abs,
            self.mvrv,
            self.mvcv,
            self.mv_in_out_count,
            self.new_mv_count,
            self.duration,
            self.count,
            self.raw_error_stdev,
            self.noise_var,
            self.cor_coeff,
            self.log_intra_error,
            self.log_coded_error,
        ]
    }
}

/// What the boost model reads out of `FRAME_INFO` (`encoder.h`).
#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
    /// Frame width in pixels.
    pub frame_width: i32,
    /// Frame height in pixels.
    pub frame_height: i32,
    /// Macroblock rows.
    pub mb_rows: i32,
    /// Macroblock columns.
    pub mb_cols: i32,
    /// Total macroblocks.
    pub num_mbs: i32,
    /// `aom_bit_depth_t`.
    pub bit_depth: u8,
}

/// `MIN_ACTIVE_AREA` (:59).
pub const MIN_ACTIVE_AREA: f64 = 0.5;
/// `MAX_ACTIVE_AREA` (:60).
pub const MAX_ACTIVE_AREA: f64 = 1.0;
/// `ACT_AREA_CORRECTION` (:72).
pub const ACT_AREA_CORRECTION: f64 = 0.5;
/// `ERR_DIVISOR` (:170).
pub const ERR_DIVISOR: f64 = 96.0;
/// `INTRA_PART` (:377).
pub const INTRA_PART: f64 = 0.005;
/// `DEFAULT_DECAY_LIMIT` (:378).
pub const DEFAULT_DECAY_LIMIT: f64 = 0.75;
/// `LOW_SR_DIFF_TRHESH` (:379) — C's spelling of the name included.
pub const LOW_SR_DIFF_TRHESH: f64 = 0.01;
/// `NCOUNT_FRAME_II_THRESH` (:380).
pub const NCOUNT_FRAME_II_THRESH: f64 = 5.0;
/// `LOW_CODED_ERR_PER_MB` (:381).
pub const LOW_CODED_ERR_PER_MB: f64 = 0.01;
/// `DEFAULT_ZM_FACTOR` (:421).
pub const DEFAULT_ZM_FACTOR: f64 = 0.5;
/// `BOOST_FACTOR` (:573).
pub const BOOST_FACTOR: f64 = 12.5;
/// `MAX_GFUBOOST_FACTOR` (`encoder.h:4375`).
pub const MAX_GFUBOOST_FACTOR: f64 = 10.0;
/// `STATIC_KF_GROUP_THRESH` (`ratectrl.h:38`).
pub const STATIC_KF_GROUP_THRESH: i32 = 99;

/// `q_pow_term` (:167) — the per-qband exponent `calc_correction_factor`
/// interpolates between.
const Q_POW_TERM: [f64; 9] = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.95, 0.95];

/// `DOUBLE_DIVIDE_CHECK(x)` (`firstpass.h:26`).
///
/// A BIAS, not a guard: it moves every divisor away from zero by `1e-6` in the
/// direction of its own sign, so it changes the quotient at every magnitude.
#[inline]
#[must_use]
pub fn double_divide_check(x: f64) -> f64 {
    if x < 0.0 {
        x - 0.000_001
    } else {
        x + 0.000_001
    }
}

/// `fclamp` (`aom_dsp/aom_dsp_common.h:82`).
///
/// Written out rather than using [`f64::clamp`] because C's version returns
/// `low` for a NaN input (`NaN < low` is false, `NaN > high` is false, so it
/// falls through to `value`) — actually it returns the NaN, which
/// `f64::clamp` also does. They agree; the explicit form keeps the ordering
/// visible.
#[inline]
#[must_use]
pub fn fclamp(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

/// `calculate_active_area` (pass2_strategy.c:61) — the fraction of the frame
/// that carries picture, after letterboxing and intra-skip.
///
/// Note `intra_skip_pct / 2` and `inactive_zone_rows * 2`: the halving and
/// doubling are C's, and the second is what converts MB rows into the same
/// units as the fraction.
#[must_use]
pub fn calculate_active_area(frame_info: &FrameInfo, stats: &FirstpassStats) -> f64 {
    let active_pct = 1.0
        - ((stats.intra_skip_pct / 2.0)
            + ((stats.inactive_zone_rows * 2.0) / f64::from(frame_info.mb_rows)));
    fclamp(active_pct, MIN_ACTIVE_AREA, MAX_ACTIVE_AREA)
}

/// `calculate_modified_err_new` (pass2_strategy.c:73) — the per-frame error
/// used to split bits between easier and harder frames.
///
/// `total_stats` of `None` is C's `NULL`, which returns 0 before touching
/// anything else.
#[must_use]
pub fn calculate_modified_err_new(
    frame_info: &FrameInfo,
    total_stats: Option<&FirstpassStats>,
    this_stats: &FirstpassStats,
    vbrbias: i32,
    modified_error_min: f64,
    modified_error_max: f64,
) -> f64 {
    let Some(total) = total_stats else {
        return 0.0;
    };
    let av_weight = total.weight / total.count;
    let av_err = (total.coded_error * av_weight) / total.count;
    let mut modified_error = av_err
        * (this_stats.coded_error * this_stats.weight / double_divide_check(av_err))
            .powf(f64::from(vbrbias) / 100.0);

    // Correction for active area: a frame with letterbox bars carries a higher
    // error per remaining MB. C's comment: coding 0.5N blocks of complexity 2X
    // is a little easier than N blocks of complexity X.
    modified_error *= calculate_active_area(frame_info, this_stats).powf(ACT_AREA_CORRECTION);

    fclamp(modified_error, modified_error_min, modified_error_max)
}

/// `frame_max_bits` (pass2_strategy.c:154) — the per-frame rate ceiling.
///
/// The product is formed in `int64_t` and only narrowed on return, so a large
/// `avg_frame_bandwidth * vbrmax_section` cannot overflow before the clamp.
#[must_use]
pub fn frame_max_bits(
    avg_frame_bandwidth: i64,
    max_frame_bandwidth: i64,
    vbrmax_section: i32,
) -> i32 {
    let mut max_bits = (avg_frame_bandwidth * i64::from(vbrmax_section)) / 100;
    if max_bits < 0 {
        max_bits = 0;
    } else if max_bits > max_frame_bandwidth {
        max_bits = max_frame_bandwidth;
    }
    max_bits as i32
}

/// `calc_correction_factor` (pass2_strategy.c:171).
///
/// The exponent is a linear interpolation across the qindex's 32-wide band,
/// which is why `Q_POW_TERM` has nine entries for eight bands: index
/// `q >> 5` and `+ 1` are both read, so the last band needs a right endpoint.
///
/// # Panics
/// If `q` is outside `0..=255`; C's `q >> 5` would index past the table's
/// right endpoint above that.
#[must_use]
pub fn calc_correction_factor(err_per_mb: f64, q: i32) -> f64 {
    assert!((0..=255).contains(&q), "qindex out of range: {q}");
    let error_term = err_per_mb / ERR_DIVISOR;
    let index = (q >> 5) as usize;
    let power_term = Q_POW_TERM[index]
        + (((Q_POW_TERM[index + 1] - Q_POW_TERM[index]) * f64::from(q % 32)) / 32.0);
    debug_assert!(error_term >= 0.0);
    fclamp(error_term.powf(power_term), 0.05, 5.0)
}

/// `qbpm_enumerator` (pass2_strategy.c:288).
#[must_use]
pub fn qbpm_enumerator(rate_err_tol: i32) -> i32 {
    1_200_000 + ((300_000 * (rate_err_tol - 25).max(0).min(75)) / 75)
}

/// `get_sr_decay_rate` (pass2_strategy.c:392) — how fast the second-reference
/// prediction quality is decaying.
///
/// The `pcnt_neutral` subtraction only applies when the coded error is above
/// `LOW_CODED_ERR_PER_MB` AND the intra/inter error ratio is under
/// `NCOUNT_FRAME_II_THRESH`; both guards are C's and both are swept.
#[must_use]
pub fn get_sr_decay_rate(frame: &FirstpassStats) -> f64 {
    let sr_diff = frame.sr_coded_error - frame.coded_error;
    let mut sr_decay = 1.0;

    let mut modified_pct_inter = frame.pcnt_inter;
    if frame.coded_error > LOW_CODED_ERR_PER_MB
        && (frame.intra_error / double_divide_check(frame.coded_error)) < NCOUNT_FRAME_II_THRESH
    {
        modified_pct_inter = frame.pcnt_inter - frame.pcnt_neutral;
    }
    let modified_pcnt_intra = 100.0 * (1.0 - modified_pct_inter);

    if sr_diff > LOW_SR_DIFF_TRHESH {
        // NOTE: C divides by `intra_error` RAW here, with no
        // DOUBLE_DIVIDE_CHECK, so a zero intra_error gives an infinity that
        // propagates. Reproduced; the first pass cannot emit intra_error == 0
        // for a frame it also gives a positive sr_diff.
        let sr_diff_part = (sr_diff * 0.25) / frame.intra_error;
        sr_decay = 1.0 - sr_diff_part - (INTRA_PART * modified_pcnt_intra);
    }
    sr_decay.max(DEFAULT_DECAY_LIMIT)
}

/// `get_zero_motion_factor` (pass2_strategy.c:415).
#[must_use]
pub fn get_zero_motion_factor(frame: &FirstpassStats) -> f64 {
    let zero_motion_pct = frame.pcnt_inter - frame.pcnt_motion;
    get_sr_decay_rate(frame).min(zero_motion_pct)
}

/// `get_prediction_decay_rate` (pass2_strategy.c:422).
#[must_use]
pub fn get_prediction_decay_rate(frame: &FirstpassStats) -> f64 {
    let sr_decay_rate = get_sr_decay_rate(frame);
    let mut zero_motion_factor = DEFAULT_ZM_FACTOR * (frame.pcnt_inter - frame.pcnt_motion);

    // Clamp to [0, 1]. C's comment says this should already hold if the inputs
    // are sensible, and checks anyway.
    if zero_motion_factor > 1.0 {
        zero_motion_factor = 1.0;
    } else if zero_motion_factor < 0.0 {
        zero_motion_factor = 0.0;
    }

    zero_motion_factor.max(sr_decay_rate + ((1.0 - sr_decay_rate) * zero_motion_factor))
}

/// `baseline_err_per_mb` (pass2_strategy.c:574).
///
/// C computes `frame_height * frame_width` into an `unsigned int`, so the
/// comparison against `640 * 360` is unsigned; the port uses `u32` for the
/// same reason.
#[must_use]
pub fn baseline_err_per_mb(frame_info: &FrameInfo) -> f64 {
    let screen_area = (frame_info.frame_height as u32).wrapping_mul(frame_info.frame_width as u32);
    if screen_area <= 640 * 360 {
        500.0
    } else {
        1000.0
    }
}

/// `av1_convert_qindex_to_q` (`av1/common/quant_common.c`) — re-exported from
/// the rate model so this module's boost curves read the same one C's do.
use crate::rate_model::convert_qindex_to_q;

/// `calc_frame_boost` (pass2_strategy.c:590) — the GF/ARF boost from one
/// frame's error ratio and motion.
///
/// `scale_max_boost` is `cpi->oxcf.mode != REALTIME` at every call site except
/// the `gfu_boost_average` derivation, which passes false deliberately.
/// Note it grows `max_boost` in place only on the positive-`mv_in_out` arm.
#[must_use]
pub fn calc_frame_boost(
    avg_frame_qindex_inter: i32,
    frame_info: &FrameInfo,
    this_frame: &FirstpassStats,
    this_frame_mv_in_out: f64,
    max_boost: f64,
    scale_max_boost: bool,
) -> f64 {
    let mut max_boost = max_boost;
    let lq = convert_qindex_to_q(avg_frame_qindex_inter, frame_info.bit_depth);
    let boost_q_correction = (0.5 + (lq * 0.015)).min(1.5);
    let active_area = calculate_active_area(frame_info, this_frame);

    // The underlying factor is the inter error ratio.
    let mut frame_boost = (baseline_err_per_mb(frame_info) * active_area)
        .max(this_frame.intra_error * active_area)
        / double_divide_check(this_frame.coded_error);
    frame_boost = frame_boost * BOOST_FACTOR * boost_q_correction;

    // More boost where new data enters the frame (zoom out); slightly less on
    // a net balance of motion out of it (zoom in). Range is -1.0 .. +1.0.
    if this_frame_mv_in_out > 0.0 {
        frame_boost += frame_boost * (this_frame_mv_in_out * 2.0);
        if scale_max_boost {
            max_boost += max_boost * (this_frame_mv_in_out * 2.0);
        }
    } else {
        // In the extreme case the boost is halved.
        frame_boost += frame_boost * (this_frame_mv_in_out / 2.0);
    }

    frame_boost.min(max_boost * boost_q_correction)
}

/// `calc_kf_frame_boost` (pass2_strategy.c:622) — the key-frame twin.
///
/// `sr_accumulator` is IN/OUT: C adds this frame's `sr_coded_error -
/// coded_error` to it and floors it at zero, and the NEXT call divides by the
/// running total. The port returns the updated value alongside the boost
/// rather than taking a `&mut`, since the caller always wants both.
///
/// The `+ 40.0` before the Q correction is, in C's words, an experimentally
/// derived baseline minimum in line with the alt-ref boost's per-frame floor.
#[must_use]
pub fn calc_kf_frame_boost(
    avg_frame_qindex_inter: i32,
    frame_info: &FrameInfo,
    this_frame: &FirstpassStats,
    sr_accumulator: f64,
    max_boost: f64,
) -> (f64, f64) {
    let lq = convert_qindex_to_q(avg_frame_qindex_inter, frame_info.bit_depth);
    let boost_q_correction = (0.50 + (lq * 0.015)).min(2.00);
    let active_area = calculate_active_area(frame_info, this_frame);

    let mut frame_boost = (baseline_err_per_mb(frame_info) * active_area)
        .max(this_frame.intra_error * active_area)
        / double_divide_check((this_frame.coded_error + sr_accumulator) * active_area);

    // Update the second-reference error accumulator.
    let new_sr = (sr_accumulator + (this_frame.sr_coded_error - this_frame.coded_error)).max(0.0);

    frame_boost = (frame_boost + 40.0) * boost_q_correction;
    (frame_boost.min(max_boost * boost_q_correction), new_sr)
}

/// `av1_get_gfu_boost_projection_factor` (`rc_utils.h:156`).
#[must_use]
pub fn gfu_boost_projection_factor(min_factor: f64, max_factor: f64, frame_count: i32) -> f64 {
    let mut factor = f64::from(frame_count).sqrt();
    factor = factor.min(max_factor);
    factor = factor.max(min_factor);
    200.0 + 10.0 * factor
}

/// `get_projected_gfu_boost` (pass2_strategy.c:653).
///
/// When every stat the boost needed was available, the boost is returned
/// unchanged; otherwise it is rescaled by the ratio of the two projection
/// factors. `rint` is round-half-to-EVEN, which is what
/// [`f64::round_ties_even`] gives — `f64::round` would be half-away-from-zero
/// and differs on every exact `.5`.
#[must_use]
pub fn get_projected_gfu_boost(
    baseline_gf_interval: i32,
    gfu_boost: i32,
    frames_to_project: i32,
    num_stats_used_for_gfu_boost: i32,
) -> i32 {
    if num_stats_used_for_gfu_boost >= frames_to_project {
        return gfu_boost;
    }
    let min_boost_factor = f64::from(baseline_gf_interval).sqrt();
    let tpl_factor =
        gfu_boost_projection_factor(min_boost_factor, MAX_GFUBOOST_FACTOR, frames_to_project);
    let tpl_factor_num_stats = gfu_boost_projection_factor(
        min_boost_factor,
        MAX_GFUBOOST_FACTOR,
        num_stats_used_for_gfu_boost,
    );
    ((tpl_factor * f64::from(gfu_boost)) / tpl_factor_num_stats).round_ties_even() as i32
}

/// `calculate_boost_bits` (pass2_strategy.c:836) — the bits the boosted frames
/// of a group get.
///
/// The `boost > 1023` rescale divides BOTH `boost` and `allocation_chunks` by
/// `boost >> 10`, in integer arithmetic, so it is not an exact ratio-preserving
/// operation — it is C's overflow guard and is reproduced as written.
#[must_use]
pub fn calculate_boost_bits(frame_count: i32, boost: i32, total_group_bits: i64) -> i32 {
    // C returns 0 for inputs that could arise through rounding errors.
    if boost == 0 || total_group_bits <= 0 {
        return 0;
    }
    if frame_count <= 0 {
        return total_group_bits.min(i64::from(i32::MAX)) as i32;
    }
    let mut boost = boost;
    let mut allocation_chunks = (frame_count * 100) + boost;

    // Prevent overflow.
    if boost > 1023 {
        let divisor = boost >> 10;
        boost /= divisor;
        allocation_chunks /= divisor;
    }

    (((i64::from(boost) * total_group_bits) / i64::from(allocation_chunks)) as i32).max(0)
}

/// `calculate_boost_factor` (pass2_strategy.c:861) — the inverse of
/// [`calculate_boost_bits`].
///
/// C computes this entirely in `double` (`100.0 * frame_count * bits`) and
/// truncates on return, so the intermediate is NOT an integer division.
#[must_use]
pub fn calculate_boost_factor(frame_count: i32, bits: i32, total_group_bits: i64) -> i32 {
    (100.0 * f64::from(frame_count) * f64::from(bits) / (total_group_bits - i64::from(bits)) as f64)
        as i32
}

/// `is_almost_static` (pass2_strategy.c:999).
///
/// With look-ahead enabled `kf_zero_motion` is not trustworthy, so C tightens
/// the GF threshold and ignores the KF one entirely.
#[must_use]
pub fn is_almost_static(gf_zero_motion: f64, kf_zero_motion: i32, is_lap_enabled: bool) -> bool {
    if is_lap_enabled {
        gf_zero_motion >= 0.999
    } else {
        gf_zero_motion >= 0.995 && kf_zero_motion >= STATIC_KF_GROUP_THRESH
    }
}

// ===========================================================================
// The GF_GROUP_STATS accumulator cluster (pass2_strategy.c:490-570) — what the
// 2-pass GOP builder folds a group's first-pass records into before it decides
// the group's length, boost and bit split.
//
// | Rust | C |
// |---|---|
// | [`GfGroupStats`] | `GF_GROUP_STATS` (`pass2_strategy.h:27`) |
// | [`GfGroupStats::init`] | `init_gf_stats` (:2282) |
// | [`accumulate_frame_motion_stats`] | `accumulate_frame_motion_stats` (:490) |
// | [`accumulate_this_frame_stats`] | `accumulate_this_frame_stats` (:517) |
// | [`accumulate_next_frame_stats`] | `accumulate_next_frame_stats` (:528) |
// | [`average_gf_stats`] | `average_gf_stats` (:561) |
// | [`calculate_section_intra_ratio`] | `calculate_section_intra_ratio` (:776) |
// | [`get_second_ref_usage_thresh`] | `get_second_ref_usage_thresh` (:2866) |
// | [`read_frame_stats`] | `read_frame_stats` (:139) |
// | [`detect_flash`] | `detect_flash` (:474) |
// | [`set_baseline_gf_interval`] | `set_baseline_gf_interval` (:2277) |
// ===========================================================================

/// `GF_GROUP_STATS` (`av1/encoder/pass2_strategy.h:27`) — the running totals a
/// GF group is summarised by.
///
/// The field ORDER matters at the oracle boundary, which passes the 17 doubles
/// flat in declaration order; `non_zero_stdev_count` is an `int` and travels
/// separately.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GfGroupStats {
    /// Sum of the group's modified errors.
    pub gf_group_err: f64,
    /// Sum of the group's raw coded errors.
    pub gf_group_raw_error: f64,
    /// Sum of the group's intra-skip fractions.
    pub gf_group_skip_pct: f64,
    /// Sum of the group's inactive (letterbox) MB rows.
    pub gf_group_inactive_zone_rows: f64,
    /// How uniform the motion field is, accumulated.
    pub mv_ratio_accumulator: f64,
    /// Running product of the per-frame prediction decay rates.
    pub decay_accumulator: f64,
    /// Running minimum of the per-frame zero-motion factors.
    pub zero_motion_accumulator: f64,
    /// This frame's prediction decay rate.
    pub loop_decay_rate: f64,
    /// The previous frame's, kept for the transition-to-still test.
    pub last_loop_decay_rate: f64,
    /// This frame's motion in/out of frame.
    pub this_frame_mv_in_out: f64,
    /// Signed accumulation of the above.
    pub mv_in_out_accumulator: f64,
    /// Absolute accumulation of the above.
    pub abs_mv_in_out_accumulator: f64,
    /// Mean second-reference coded error (after [`average_gf_stats`]).
    pub avg_sr_coded_error: f64,
    /// Mean second-reference usage.
    pub avg_pcnt_second_ref: f64,
    /// Mean new-MV count.
    pub avg_new_mv_count: f64,
    /// Mean wavelet energy.
    pub avg_wavelet_energy: f64,
    /// Mean raw-error standard deviation, over the frames that had one.
    pub avg_raw_err_stdev: f64,
    /// How many frames contributed a non-zero `raw_error_stdev`.
    pub non_zero_stdev_count: i32,
}

impl GfGroupStats {
    /// `init_gf_stats` (pass2_strategy.c:2282).
    ///
    /// Note that three fields start at 1.0, not 0.0: `decay_accumulator` and
    /// `zero_motion_accumulator` because they are a running PRODUCT and a
    /// running MINIMUM, and the two decay rates because a group with no
    /// accumulated frame must read as "no decay". This is why the port has an
    /// explicit `init` rather than deriving `Default`.
    #[must_use]
    pub fn init() -> Self {
        Self {
            gf_group_err: 0.0,
            gf_group_raw_error: 0.0,
            gf_group_skip_pct: 0.0,
            gf_group_inactive_zone_rows: 0.0,
            mv_ratio_accumulator: 0.0,
            decay_accumulator: 1.0,
            zero_motion_accumulator: 1.0,
            loop_decay_rate: 1.0,
            last_loop_decay_rate: 1.0,
            this_frame_mv_in_out: 0.0,
            mv_in_out_accumulator: 0.0,
            abs_mv_in_out_accumulator: 0.0,
            avg_sr_coded_error: 0.0,
            avg_pcnt_second_ref: 0.0,
            avg_new_mv_count: 0.0,
            avg_wavelet_energy: 0.0,
            avg_raw_err_stdev: 0.0,
            non_zero_stdev_count: 0,
        }
    }

    /// The 17 `double` members in C's declaration order — the oracle
    /// boundary's layout. `non_zero_stdev_count` is excluded (it is an `int`).
    #[must_use]
    pub fn to_doubles(&self) -> [f64; 17] {
        [
            self.gf_group_err,
            self.gf_group_raw_error,
            self.gf_group_skip_pct,
            self.gf_group_inactive_zone_rows,
            self.mv_ratio_accumulator,
            self.decay_accumulator,
            self.zero_motion_accumulator,
            self.loop_decay_rate,
            self.last_loop_decay_rate,
            self.this_frame_mv_in_out,
            self.mv_in_out_accumulator,
            self.abs_mv_in_out_accumulator,
            self.avg_sr_coded_error,
            self.avg_pcnt_second_ref,
            self.avg_new_mv_count,
            self.avg_wavelet_energy,
            self.avg_raw_err_stdev,
        ]
    }

    /// Rebuild from the boundary layout produced by [`Self::to_doubles`].
    #[must_use]
    pub fn from_doubles(d: &[f64; 17], non_zero_stdev_count: i32) -> Self {
        Self {
            gf_group_err: d[0],
            gf_group_raw_error: d[1],
            gf_group_skip_pct: d[2],
            gf_group_inactive_zone_rows: d[3],
            mv_ratio_accumulator: d[4],
            decay_accumulator: d[5],
            zero_motion_accumulator: d[6],
            loop_decay_rate: d[7],
            last_loop_decay_rate: d[8],
            this_frame_mv_in_out: d[9],
            mv_in_out_accumulator: d[10],
            abs_mv_in_out_accumulator: d[11],
            avg_sr_coded_error: d[12],
            avg_pcnt_second_ref: d[13],
            avg_new_mv_count: d[14],
            avg_wavelet_energy: d[15],
            avg_raw_err_stdev: d[16],
            non_zero_stdev_count,
        }
    }
}

/// `accumulate_frame_motion_stats` (pass2_strategy.c:490).
///
/// `f_w` / `f_h` are the reciprocals of the frame dimensions the caller passes
/// (`accumulate_next_frame_stats` hands them straight through from
/// `accumulate_gop_stats`), so `mvr_abs * f_h` is a normalised motion
/// magnitude and the `min` below picks the smaller of the ratio and it.
///
/// `this_frame_mv_in_out` is OVERWRITTEN, not accumulated — only the two
/// accumulators below it accumulate.
pub fn accumulate_frame_motion_stats(
    stats: &FirstpassStats,
    gf_stats: &mut GfGroupStats,
    f_w: f64,
    f_h: f64,
) {
    let pct = stats.pcnt_motion;

    gf_stats.this_frame_mv_in_out = stats.mv_in_out_count * pct;
    gf_stats.mv_in_out_accumulator += gf_stats.this_frame_mv_in_out;
    gf_stats.abs_mv_in_out_accumulator += gf_stats.this_frame_mv_in_out.abs();

    // How uniform (conversely, how random) the motion field is: abs(mv) / mv.
    if pct > 0.05 {
        let mvr_ratio = stats.mvr_abs.abs() / double_divide_check(stats.mvr.abs());
        let mvc_ratio = stats.mvc_abs.abs() / double_divide_check(stats.mvc.abs());

        // C writes these as `a < b ? a : b`, which is NOT `f64::min` for a NaN
        // input (min returns the non-NaN operand; the ternary returns b). The
        // explicit form is kept for that reason.
        let r = stats.mvr_abs * f_h;
        gf_stats.mv_ratio_accumulator += pct * if mvr_ratio < r { mvr_ratio } else { r };
        let c = stats.mvc_abs * f_w;
        gf_stats.mv_ratio_accumulator += pct * if mvc_ratio < c { mvc_ratio } else { c };
    }
}

/// `accumulate_this_frame_stats` (pass2_strategy.c:517).
///
/// `gf_group_raw_error` is under `#if GROUP_ADAPTIVE_MAXQ`, which is `1` in
/// this build (pass2_strategy.c:49), so the arm is LIVE.
pub fn accumulate_this_frame_stats(
    stats: &FirstpassStats,
    mod_frame_err: f64,
    gf_stats: &mut GfGroupStats,
) {
    gf_stats.gf_group_err += mod_frame_err;
    gf_stats.gf_group_raw_error += stats.coded_error;
    gf_stats.gf_group_skip_pct += stats.intra_skip_pct;
    gf_stats.gf_group_inactive_zone_rows += stats.inactive_zone_rows;
}

/// `accumulate_next_frame_stats` (pass2_strategy.c:528).
///
/// The decay half is skipped entirely on a flash frame — a flash breaks
/// prediction for one frame and then recovers, so folding its decay rate into
/// the group would misreport the group as decaying.
///
/// The static-section monitor is gated on `(frames_since_key + cur_idx - 1) > 1`,
/// so the first two frames after a key frame never lower
/// `zero_motion_accumulator`.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_next_frame_stats(
    stats: &FirstpassStats,
    flash_detected: bool,
    frames_since_key: i32,
    cur_idx: i32,
    gf_stats: &mut GfGroupStats,
    f_w: i32,
    f_h: i32,
) {
    accumulate_frame_motion_stats(stats, gf_stats, f64::from(f_w), f64::from(f_h));
    gf_stats.avg_sr_coded_error += stats.sr_coded_error;
    gf_stats.avg_pcnt_second_ref += stats.pcnt_second_ref;
    gf_stats.avg_new_mv_count += stats.new_mv_count;
    gf_stats.avg_wavelet_energy += stats.frame_avg_wavelet_energy;
    if stats.raw_error_stdev.abs() > 0.000_001 {
        gf_stats.non_zero_stdev_count += 1;
        gf_stats.avg_raw_err_stdev += stats.raw_error_stdev;
    }

    if !flash_detected {
        gf_stats.last_loop_decay_rate = gf_stats.loop_decay_rate;
        gf_stats.loop_decay_rate = get_prediction_decay_rate(stats);
        gf_stats.decay_accumulator *= gf_stats.loop_decay_rate;

        // Monitor for static sections.
        if (frames_since_key + cur_idx - 1) > 1 {
            gf_stats.zero_motion_accumulator = gf_stats
                .zero_motion_accumulator
                .min(get_zero_motion_factor(stats));
        }
    }
}

/// `average_gf_stats` (pass2_strategy.c:561).
///
/// Two different divisors: the four sums use the frame count, but
/// `avg_raw_err_stdev` uses only the count of frames that HAD a non-zero
/// standard deviation. Both divisions are skipped when their divisor is zero.
pub fn average_gf_stats(total_frame: i32, gf_stats: &mut GfGroupStats) {
    if total_frame != 0 {
        let n = f64::from(total_frame);
        gf_stats.avg_sr_coded_error /= n;
        gf_stats.avg_pcnt_second_ref /= n;
        gf_stats.avg_new_mv_count /= n;
        gf_stats.avg_wavelet_energy /= n;
    }
    if gf_stats.non_zero_stdev_count != 0 {
        gf_stats.avg_raw_err_stdev /= f64::from(gf_stats.non_zero_stdev_count);
    }
}

/// `calculate_section_intra_ratio` (pass2_strategy.c:776) — the
/// intra/coded error ratio over a run of frames, used to cap the loop filter.
///
/// C walks `begin..end` and stops at `section_length`, whichever comes first;
/// the port takes the slice and the length for the same reason.
#[must_use]
pub fn calculate_section_intra_ratio(section: &[FirstpassStats], section_length: i32) -> i32 {
    let mut intra_error = 0.0f64;
    let mut coded_error = 0.0f64;
    for s in section.iter().take(section_length.max(0) as usize) {
        intra_error += s.intra_error;
        coded_error += s.coded_error;
    }
    (intra_error / double_divide_check(coded_error)) as i32
}

/// `get_second_ref_usage_thresh` (pass2_strategy.c:2866).
///
/// High second-reference usage suggests a transient (a flash, an occlusion)
/// rather than a real scene cut, so the threshold rises with how many frames
/// this key-frame group already has — up to a cap at 32 frames.
///
/// The divisor is `adapt_upto - 1`, not `adapt_upto`, so the ramp reaches its
/// maximum one frame BEFORE the cap takes over.
#[must_use]
pub fn get_second_ref_usage_thresh(frame_count_so_far: i32) -> f64 {
    const ADAPT_UPTO: i32 = 32;
    const MIN_SECOND_REF_USAGE_THRESH: f64 = 0.085;
    const SECOND_REF_USAGE_THRESH_MAX_DELTA: f64 = 0.035;
    if frame_count_so_far >= ADAPT_UPTO {
        return MIN_SECOND_REF_USAGE_THRESH + SECOND_REF_USAGE_THRESH_MAX_DELTA;
    }
    MIN_SECOND_REF_USAGE_THRESH
        + (f64::from(frame_count_so_far) / f64::from(ADAPT_UPTO - 1))
            * SECOND_REF_USAGE_THRESH_MAX_DELTA
}

/// `read_frame_stats` (pass2_strategy.c:139) — index `cur + offset` into a
/// stats buffer, or `None` when that leaves the buffer.
///
/// C's two bounds tests are written ASYMMETRICALLY — the forward one against
/// `stats_in_end`, the backward one against `stats_in_start`, each applied
/// only for its own sign of `offset`. For any `cur` inside the buffer the two
/// halves coincide with a plain `0 <= idx < len`, and that equivalence was
/// MEASURED (`read_frame_stats_bounds_match_c` sweeps every `cur` in
/// `0..=len` against C and a symmetric spelling passes it). C's shape is kept
/// anyway so a later reader comparing the two files sees the same structure.
#[must_use]
pub fn read_frame_stats(len: usize, cur: usize, offset: i32) -> Option<usize> {
    let idx = cur as i64 + i64::from(offset);
    if offset >= 0 {
        if idx >= len as i64 {
            return None;
        }
    } else if idx < 0 {
        return None;
    }
    Some(idx as usize)
}

/// `detect_flash` (pass2_strategy.c:474).
///
/// A flash shows up as the frame AFTER it being well predicted from a
/// pre-flash reference: high `pcnt_second_ref` relative to `pcnt_inter`. The
/// caller passes the offset that reaches that following frame.
#[must_use]
pub fn detect_flash(stats: &[FirstpassStats], cur: usize, offset: i32) -> bool {
    let Some(idx) = read_frame_stats(stats.len(), cur, offset) else {
        return false;
    };
    let next = &stats[idx];
    next.pcnt_second_ref > next.pcnt_inter && next.pcnt_second_ref >= 0.5
}

/// `set_baseline_gf_interval` (pass2_strategy.c:2277).
///
/// A one-line setter in C. The port keeps it as a named function so the call
/// sites read the same and the (deliberate) absence of any clamping is
/// documented in one place.
#[must_use]
pub fn set_baseline_gf_interval(arf_position: i32) -> i32 {
    arf_position
}

// ===========================================================================
// The region-analysis cluster (pass2_strategy.c:1125-1460, :3677-3730) — how
// the 2-pass GOP builder segments a shot into stable / high-variance /
// scenecut / blending regions before it places key frames and GF boundaries.
//
// | Rust | C |
// |---|---|
// | [`RegionType`] / [`Region`] | `REGION_TYPES` / `REGIONS` (`ratectrl.h:110`) |
// | [`smooth_filter_stats`] | `smooth_filter_stats` (:1125) |
// | [`get_gradient`] | `get_gradient` (:1169) |
// | [`remove_region`] | `remove_region` (:1255) |
// | [`insert_region`] | `insert_region` (:1293) |
// | [`analyze_region`] | `analyze_region` (:1324) |
// | [`get_region_stats`] | `get_region_stats` (:1360) |
// | [`find_stable_regions`] | `find_stable_regions` (:1368) |
// | [`cleanup_regions`] | `cleanup_regions` (:1423) |
// | [`remove_short_regions`] | `remove_short_regions` (:1438) |
// | [`find_regions_index`] | `find_regions_index` (:1909) |
// | [`mark_flashes`] | `mark_flashes` (:3677) |
// | [`smooth_filter_noise`] | `smooth_filter_noise` (:3699) |
// ===========================================================================

/// `SMOOTH_FILT_LEN` (:1119).
pub const SMOOTH_FILT_LEN: usize = 7;
/// `HALF_FILT_LEN` (:1120).
pub const HALF_FILT_LEN: i32 = (SMOOTH_FILT_LEN / 2) as i32;
/// `WINDOW_SIZE` (:1121).
pub const WINDOW_SIZE: usize = 7;
/// `HALF_WIN` (:1122).
pub const HALF_WIN: i32 = (WINDOW_SIZE / 2) as i32;

/// The 7-tap gaussian in `smooth_filter_stats` (:1128).
const SMOOTH_FILT: [f64; SMOOTH_FILT_LEN] = [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006];

/// `REGION_TYPES` (`av1/encoder/ratectrl.h:110`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum RegionType {
    /// `STABLE_REGION`.
    Stable = 0,
    /// `HIGH_VAR_REGION`.
    HighVar = 1,
    /// `SCENECUT_REGION`.
    Scenecut = 2,
    /// `BLENDING_REGION`.
    Blending = 3,
}

/// `REGIONS` (`ratectrl.h:119`) — one contiguous run of frames plus its
/// averaged first-pass statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Region {
    /// First frame index, inclusive.
    pub start: i32,
    /// Last frame index, inclusive.
    pub last: i32,
    /// Mean noise variance over the region.
    pub avg_noise_var: f64,
    /// Mean correlation coefficient.
    pub avg_cor_coeff: f64,
    /// Mean second-reference / frame error ratio.
    pub avg_sr_fr_ratio: f64,
    /// Mean intra error.
    pub avg_intra_err: f64,
    /// Mean coded error.
    pub avg_coded_err: f64,
    /// Which kind of region this is. C's `REGION_TYPES`, kept as its raw
    /// discriminant because `remove_region` and `insert_region` copy whole
    /// entries around including types the caller never named.
    pub kind: i32,
}

/// `smooth_filter_stats` (pass2_strategy.c:1125) — the 7-tap gaussian over
/// `intra_error` and `coded_error`.
///
/// Two facts the shape hides:
/// * the output arrays are ACCUMULATED into (`+=`), not written, so C's caller
///   has to zero them first — the port does the same rather than assigning,
///   because a caller that pre-seeded them would then differ;
/// * the two passes have DIFFERENT flash tests. The intra pass skips a tap
///   whose own frame is a flash; the coded pass also skips one whose PREVIOUS
///   frame is (C's comment: "Coded error involves idx and idx - 1"), and that
///   `idx > 0` guard is against the ARRAY's start, not `start_idx`.
///
/// When every tap in a window is skipped the filter falls back to the
/// unfiltered value, using a `total_wt > 0.01` test rather than `!= 0`.
pub fn smooth_filter_stats(
    stats: &[FirstpassStats],
    start_idx: usize,
    last_idx: usize,
    filt_intra_err: &mut [f64],
    filt_coded_err: &mut [f64],
) {
    for i in start_idx..=last_idx {
        let mut total_wt = 0.0;
        for j in -HALF_FILT_LEN..=HALF_FILT_LEN {
            let idx = (i as i32 + j).clamp(start_idx as i32, last_idx as i32) as usize;
            if stats[idx].is_flash != 0 {
                continue;
            }
            let w = SMOOTH_FILT[(j + HALF_FILT_LEN) as usize];
            filt_intra_err[i] += w * stats[idx].intra_error;
            total_wt += w;
        }
        if total_wt > 0.01 {
            filt_intra_err[i] /= total_wt;
        } else {
            filt_intra_err[i] = stats[i].intra_error;
        }
    }
    for i in start_idx..=last_idx {
        let mut total_wt = 0.0;
        for j in -HALF_FILT_LEN..=HALF_FILT_LEN {
            let idx = (i as i32 + j).clamp(start_idx as i32, last_idx as i32) as usize;
            // Coded error involves idx and idx - 1.
            if stats[idx].is_flash != 0 || (idx > 0 && stats[idx - 1].is_flash != 0) {
                continue;
            }
            let w = SMOOTH_FILT[(j + HALF_FILT_LEN) as usize];
            filt_coded_err[i] += w * stats[idx].coded_error;
            total_wt += w;
        }
        if total_wt > 0.01 {
            filt_coded_err[i] /= total_wt;
        } else {
            filt_coded_err[i] = stats[i].coded_error;
        }
    }
}

/// `get_gradient` (pass2_strategy.c:1169) — a centred difference, clamped at
/// the ends so the first and last entries are one-sided.
///
/// `start == last` writes a single zero and returns; every other length
/// divides by `next - prev`, which is 1 at the ends and 2 inside.
pub fn get_gradient(values: &[f64], start: usize, last: usize, grad: &mut [f64]) {
    if start == last {
        grad[start] = 0.0;
        return;
    }
    for i in start..=last {
        let prev = i.saturating_sub(1).max(start);
        let next = (i + 1).min(last);
        grad[i] = (values[next] - values[prev]) / (next - prev) as f64;
    }
}

/// `remove_region` (pass2_strategy.c:1255) — drop `regions[*next_region]` and
/// merge its frames into a neighbour.
///
/// `merge` is 0 (into the previous), 1 (into the next) or 2 (into both, taking
/// the previous one's type). C OVERRIDES the caller's choice at the ends: the
/// first region can only merge forward and the last only backward, which also
/// turns a `merge == 2` at either end into a single merge.
///
/// # Panics
/// If `*next_region >= *num_regions`, which C asserts.
pub fn remove_region(
    merge: i32,
    regions: &mut [Region],
    num_regions: &mut i32,
    next_region: &mut i32,
) {
    let k = *next_region;
    assert!(
        k < *num_regions,
        "next_region {k} is past num_regions {}",
        *num_regions
    );
    if *num_regions == 1 {
        *num_regions = 0;
        return;
    }
    let mut merge = merge;
    if k == 0 {
        merge = 1;
    } else if k == *num_regions - 1 {
        merge = 0;
    }
    let num_merge = if merge == 2 { 2 } else { 1 };
    match merge {
        0 => {
            regions[(k - 1) as usize].last = regions[k as usize].last;
            *next_region = k;
        }
        1 => {
            regions[(k + 1) as usize].start = regions[k as usize].start;
            *next_region = k + 1;
        }
        2 => {
            regions[(k - 1) as usize].last = regions[(k + 1) as usize].last;
            *next_region = k;
        }
        _ => unreachable!("C asserts on any other merge value"),
    }
    *num_regions -= num_merge;
    // The shift start is `*next_region - (merge == 1)`, i.e. one lower on the
    // forward-merge arm because `next_region` was advanced above.
    let mut k = *next_region - i32::from(merge == 1);
    while k < *num_regions {
        regions[k as usize] = regions[(k + num_merge) as usize];
        k += 1;
    }
}

/// `insert_region` (pass2_strategy.c:1293) — split `regions[*cur_region_idx]`
/// so that `start..=last` becomes its own region of `kind`.
///
/// Adds one entry per end that is NOT already a boundary, so the array grows
/// by 0, 1 or 2. `*cur_region_idx` comes back pointing at the LAST region the
/// split produced, which is the inserted one when it reaches the original's
/// end and the trailing remainder otherwise.
pub fn insert_region(
    start: i32,
    last: i32,
    kind: i32,
    regions: &mut [Region],
    num_regions: &mut i32,
    cur_region_idx: &mut i32,
) {
    let mut k = *cur_region_idx;
    let this_region_type = regions[k as usize].kind;
    let this_region_last = regions[k as usize].last;
    let num_add =
        i32::from(start != regions[k as usize].start) + i32::from(last != regions[k as usize].last);
    // Move the following regions further back.
    let mut r = *num_regions - 1;
    while r > k {
        regions[(r + num_add) as usize] = regions[r as usize];
        r -= 1;
    }
    *num_regions += num_add;
    if start > regions[k as usize].start {
        regions[k as usize].last = start - 1;
        k += 1;
        regions[k as usize].start = start;
    }
    regions[k as usize].kind = kind;
    if last < this_region_last {
        regions[k as usize].last = last;
        k += 1;
        regions[k as usize].start = last + 1;
        regions[k as usize].last = this_region_last;
        regions[k as usize].kind = this_region_type;
    } else {
        regions[k as usize].last = this_region_last;
    }
    *cur_region_idx = k;
}

/// `analyze_region` (pass2_strategy.c:1324) — average one region's stats.
///
/// Four of the five averages divide by the region's own length; the
/// `avg_sr_fr_ratio` divides by a DIFFERENT count (`last - start +
/// check_first_sr`) and skips the region's first frame unless the region is
/// not the first one, because the ratio needs a previous frame to compare
/// against. C reads `stats[i - 1]` there, which is why region 0's first frame
/// is excluded.
///
/// `avg_noise_var` is ACCUMULATED into without being zeroed first, unlike the
/// other four — so the caller's previous value survives. Reproduced.
pub fn analyze_region(stats: &[FirstpassStats], k: usize, regions: &mut [Region]) {
    let r = &mut regions[k];
    r.avg_cor_coeff = 0.0;
    r.avg_sr_fr_ratio = 0.0;
    r.avg_intra_err = 0.0;
    r.avg_coded_err = 0.0;
    // NOTE: avg_noise_var is NOT reset here. C leaves it alone.

    let check_first_sr = i32::from(k != 0);
    let span = f64::from(r.last - r.start + 1);

    for i in r.start..=r.last {
        let s = &stats[i as usize];
        if i > r.start || check_first_sr != 0 {
            let num_frames = f64::from(r.last - r.start + check_first_sr);
            let prev = &stats[(i - 1) as usize];
            let max_coded_error = s.coded_error.max(prev.coded_error);
            let this_ratio = s.sr_coded_error / max_coded_error.max(0.001);
            r.avg_sr_fr_ratio += this_ratio / num_frames;
        }
        r.avg_intra_err += s.intra_error / span;
        r.avg_coded_err += s.coded_error / span;
        r.avg_cor_coeff += s.cor_coeff.max(0.001) / span;
        r.avg_noise_var += s.noise_var.max(0.001) / span;
    }
}

/// `get_region_stats` (pass2_strategy.c:1360) — [`analyze_region`] over every
/// region.
pub fn get_region_stats(stats: &[FirstpassStats], regions: &mut [Region], num_regions: usize) {
    for k in 0..num_regions {
        analyze_region(stats, k, regions);
    }
}

/// `find_stable_regions` (pass2_strategy.c:1368) — the first segmentation
/// pass, marking each frame stable or high-variance from a 7-frame window and
/// then collapsing runs of the same type into regions.
///
/// Returns the number of regions written.
///
/// The three stability tests are C's, including the `0.001` seeds on all four
/// accumulators: `mean_intra` starts at `0.001`, not `0`, so the
/// variance-over-mean-squared ratios are finite even for an all-skipped
/// window. The seeds are inside the sums, so they are also divided by `count`.
#[must_use]
pub fn find_stable_regions(
    stats: &[FirstpassStats],
    grad_coded: &[f64],
    this_start: usize,
    this_last: usize,
    regions: &mut [Region],
) -> usize {
    let mut k = 0usize;
    regions[k].start = this_start as i32;
    for i in this_start..=this_last {
        let mut mean_intra = 0.001;
        let mut var_intra = 0.001;
        let mut mean_coded = 0.001;
        let mut var_coded = 0.001;
        let mut count = 0i32;
        for j in -HALF_WIN..=HALF_WIN {
            let idx = (i as i32 + j).clamp(this_start as i32, this_last as i32) as usize;
            if stats[idx].is_flash != 0 || (idx > 0 && stats[idx - 1].is_flash != 0) {
                continue;
            }
            mean_intra += stats[idx].intra_error;
            var_intra += stats[idx].intra_error * stats[idx].intra_error;
            mean_coded += stats[idx].coded_error;
            var_coded += stats[idx].coded_error * stats[idx].coded_error;
            count += 1;
        }

        let cur_type = if count > 0 {
            let n = f64::from(count);
            mean_intra /= n;
            var_intra /= n;
            mean_coded /= n;
            var_coded /= n;
            let is_intra_stable = var_intra / (mean_intra * mean_intra) < 1.03;
            let is_coded_stable = (var_coded / (mean_coded * mean_coded) < 1.04
                && grad_coded[i].abs() / mean_coded < 0.05)
                || mean_coded / mean_intra < 0.05;
            let is_coded_small = mean_coded < 0.5 * mean_intra;
            if is_intra_stable && is_coded_stable && is_coded_small {
                RegionType::Stable as i32
            } else {
                RegionType::HighVar as i32
            }
        } else {
            RegionType::HighVar as i32
        };

        // Mark a new region where the type changes.
        if i as i32 == regions[k].start {
            regions[k].kind = cur_type;
        } else if cur_type != regions[k].kind {
            regions[k].last = i as i32 - 1;
            regions[k + 1].start = i as i32;
            regions[k + 1].kind = cur_type;
            k += 1;
        }
    }
    regions[k].last = this_last as i32;
    k + 1
}

/// `cleanup_regions` (pass2_strategy.c:1423) — merge adjacent regions of the
/// same type and drop empty ones.
///
/// A SCENECUT region is never merged with an identical neighbour, because two
/// consecutive scenecuts are two events rather than one long one.
pub fn cleanup_regions(regions: &mut [Region], num_regions: &mut i32) {
    let mut k = 0i32;
    while k < *num_regions {
        let same_as_prev = k > 0
            && regions[(k - 1) as usize].kind == regions[k as usize].kind
            && regions[k as usize].kind != RegionType::Scenecut as i32;
        if same_as_prev || regions[k as usize].last < regions[k as usize].start {
            remove_region(0, regions, num_regions, &mut k);
        } else {
            k += 1;
        }
    }
}

/// `remove_short_regions` (pass2_strategy.c:1438) — drop regions of `kind`
/// shorter than `length`, merging each into BOTH neighbours, then clean up.
///
/// The loop guard is `k < *num_regions && *num_regions > 1`, so the last
/// surviving region is never removed however short it is.
pub fn remove_short_regions(regions: &mut [Region], num_regions: &mut i32, kind: i32, length: i32) {
    let mut k = 0i32;
    while k < *num_regions && *num_regions > 1 {
        let r = regions[k as usize];
        if r.last - r.start + 1 < length && r.kind == kind {
            remove_region(2, regions, num_regions, &mut k);
        } else {
            k += 1;
        }
    }
    cleanup_regions(regions, num_regions);
}

/// `find_regions_index` (pass2_strategy.c:1909) — which region contains a
/// frame, or `None` when none does.
#[must_use]
pub fn find_regions_index(regions: &[Region], num_regions: usize, frame_idx: i32) -> Option<usize> {
    (0..num_regions).find(|&k| regions[k].start <= frame_idx && regions[k].last >= frame_idx)
}

/// `mark_flashes` (pass2_strategy.c:3677) — set each frame's `is_flash` from
/// the SUCCEEDING frame's second-reference usage.
///
/// The flag lands on frame `i` when frame `i + 1` recovers from it, which is
/// why the walk stops one short and C forces the final frame to 0.
pub fn mark_flashes(stats: &mut [FirstpassStats]) {
    let len = stats.len();
    for i in 0..len.saturating_sub(1) {
        let next = stats[i + 1];
        stats[i].is_flash =
            i64::from(next.pcnt_second_ref > next.pcnt_inter && next.pcnt_second_ref >= 0.5);
    }
    // The last one is always treated as not a flash.
    if len >= 1 {
        stats[len - 1].is_flash = 0;
    }
}

/// `smooth_filter_noise` (pass2_strategy.c:3699) — a 7-tap BOX average over
/// `noise_var`, skipping flash frames.
///
/// Not the gaussian [`smooth_filter_stats`] uses: every surviving tap gets
/// weight 1.0. libaom's own TODO says a better low-pass filter would be
/// preferable. The whole array is computed before ANY of it is written back,
/// so the filter reads unfiltered input throughout.
pub fn smooth_filter_noise(stats: &mut [FirstpassStats]) {
    let len = stats.len();
    let mut smooth_noise = vec![0.0f64; len];
    for (i, out) in smooth_noise.iter_mut().enumerate() {
        let mut total_noise = 0.0;
        let mut total_wt = 0.0;
        for j in -HALF_FILT_LEN..=HALF_FILT_LEN {
            let idx = (i as i32 + j).clamp(0, len as i32 - 1) as usize;
            if stats[idx].is_flash != 0 {
                continue;
            }
            total_noise += stats[idx].noise_var;
            total_wt += 1.0;
        }
        *out = if total_wt > 0.01 {
            total_noise / total_wt
        } else {
            stats[i].noise_var
        };
    }
    for (s, n) in stats.iter_mut().zip(smooth_noise) {
        s.noise_var = n;
    }
}

// ===========================================================================
// The scenecut / noise-model helpers.
//
// | Rust | C |
// |---|---|
// | [`find_qindex_by_rate_with_correction`] | `find_qindex_by_rate_with_correction` (:294) |
// | [`slide_transition`] | `slide_transition` (:2853) |
// | [`estimate_noise`] | `estimate_noise` (:3732) |
// | [`estimate_coeff`] | `estimate_coeff` (:3827) |
// ===========================================================================

/// `VERY_LOW_II` (:2842).
pub const VERY_LOW_II: f64 = 1.5;
/// `ERROR_SPIKE` (:2844).
pub const ERROR_SPIKE: f64 = 5.0;

/// `find_qindex_by_rate_with_correction` (pass2_strategy.c:294) — the qindex
/// whose modelled bits-per-mb first drops to or below `desired_bits_per_mb`.
///
/// A binary search over `[best_qindex, worst_qindex]` that converges on the
/// LOWEST qindex satisfying the bound, because the `else` arm keeps `mid`
/// rather than moving past it. The modelled rate is computed in `uint64_t`
/// from a `double` expression, so it TRUNCATES — and a negative intermediate
/// would wrap rather than clamp. `group_weight_factor` and the correction
/// factor are both positive at every call site, so it cannot.
///
/// # Panics
/// If `best_qindex > worst_qindex`, which C asserts.
#[must_use]
pub fn find_qindex_by_rate_with_correction(
    desired_bits_per_mb: u64,
    bit_depth: u8,
    error_per_mb: f64,
    group_weight_factor: f64,
    rate_err_tol: i32,
    best_qindex: i32,
    worst_qindex: i32,
) -> i32 {
    assert!(
        best_qindex <= worst_qindex,
        "{best_qindex} > {worst_qindex}"
    );
    let mut low = best_qindex;
    let mut high = worst_qindex;
    while low < high {
        let mid = (low + high) >> 1;
        let mid_factor = calc_correction_factor(error_per_mb, mid);
        let q = convert_qindex_to_q(mid, bit_depth);
        let enumerator = qbpm_enumerator(rate_err_tol);
        let mid_bits_per_mb =
            ((f64::from(enumerator) * mid_factor * group_weight_factor) / q) as u64;
        if mid_bits_per_mb > desired_bits_per_mb {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

/// `slide_transition` (pass2_strategy.c:2853) — a hard cut between slides,
/// which is a coded-error SPIKE against both neighbours on a frame whose intra
/// error is much lower than its coded error.
///
/// All three tests must hold; the `VERY_LOW_II` one is what distinguishes a
/// slide change from ordinary motion.
#[must_use]
pub fn slide_transition(
    this_frame: &FirstpassStats,
    last_frame: &FirstpassStats,
    next_frame: &FirstpassStats,
) -> bool {
    this_frame.intra_error < (this_frame.coded_error * VERY_LOW_II)
        && this_frame.coded_error > (last_frame.coded_error * ERROR_SPIKE)
        && this_frame.coded_error > (next_frame.coded_error * ERROR_SPIKE)
}

/// `estimate_noise` (pass2_strategy.c:3732) — per-frame noise variance from a
/// three-frame innovation model, followed by three repair passes and the box
/// filter.
///
/// Writes `noise_var` on every frame from index 2 up. The four stages are C's,
/// in C's order, and none can be folded into another:
/// 1. the model, skipped for any frame within two of a flash (a flash has
///    highly correlated innovations) and for any frame whose three
///    intermediate products are not all positive;
/// 2. a repair pass that copies a trustworthy neighbour's value into any frame
///    whose own came out below 1.0, searching FORWARD first and only then
///    backward;
/// 3. a second repair pass with the same shape for the flash frames stage 1
///    skipped. It does NOT test `< 1.0` the way stage 2 does, but that is
///    unobservable and was MEASURED to be: stage 1 sets every near-flash
///    frame's `noise_var` to 0.0 before `continue`ing and stage 2 skips them,
///    so within `2..len` a near-flash frame always arrives here at 0.0.
///    Adding the missing test passes the whole differential;
/// 4. copying frame 2's value onto frames 0 and 1, then
///    [`smooth_filter_noise`].
///
/// Stage 4's guard is `(first_stats + 2) < last_stats`, so a run shorter than
/// three frames leaves frames 0 and 1 untouched.
pub fn estimate_noise(stats: &mut [FirstpassStats]) {
    let len = stats.len();
    let near_flash = |s: &[FirstpassStats], i: usize| {
        s[i].is_flash != 0 || s[i - 1].is_flash != 0 || s[i - 2].is_flash != 0
    };

    // 1. The innovation model.
    for i in 2..len {
        stats[i].noise_var = 0.0;
        if near_flash(stats, i) {
            continue;
        }
        let c1 = stats[i - 1].intra_error * (stats[i].intra_error - stats[i].coded_error);
        let c2 = stats[i - 2].intra_error * (stats[i - 1].intra_error - stats[i - 1].coded_error);
        let c3 = stats[i - 2].intra_error * (stats[i].intra_error - stats[i].sr_coded_error);
        if c1 <= 0.0 || c2 <= 0.0 || c3 <= 0.0 {
            continue;
        }
        let noise = stats[i - 1].intra_error - c1.sqrt() * c2.sqrt() / c3.sqrt();
        stats[i].noise_var = noise.max(0.01);
    }

    // 2. Copy from a neighbour where the value is not trustworthy.
    for i in 2..len {
        if near_flash(stats, i) {
            continue;
        }
        if stats[i].noise_var < 1.0 {
            let usable =
                |s: &[FirstpassStats], j: usize| !near_flash(s, j) && s[j].noise_var >= 1.0;
            if let Some(j) = (i + 1..len).find(|&j| usable(stats, j)) {
                stats[i].noise_var = stats[j].noise_var;
                continue;
            }
            if let Some(j) = (2..i).rev().find(|&j| usable(stats, j)) {
                stats[i].noise_var = stats[j].noise_var;
            }
        }
    }

    // 3. Copy into the flash frames. NOTE the missing `< 1.0` test: this pass
    // overwrites unconditionally for any frame near a flash.
    for i in 2..len {
        if !near_flash(stats, i) {
            continue;
        }
        if let Some(j) = (i + 1..len).find(|&j| !near_flash(stats, j)) {
            stats[i].noise_var = stats[j].noise_var;
            continue;
        }
        if let Some(j) = (2..i).rev().find(|&j| !near_flash(stats, j)) {
            stats[i].noise_var = stats[j].noise_var;
        }
    }

    // 4. The first two frames take frame 2's value, then the box filter.
    if len > 2 {
        let v = stats[2].noise_var;
        stats[0].noise_var = v;
        stats[1].noise_var = v;
    }
    smooth_filter_noise(stats);
}

/// `estimate_coeff` (pass2_strategy.c:3827) — each frame's correlation
/// coefficient with its predecessor, from the same innovation model.
///
/// Reads `noise_var`, which [`estimate_noise`] writes, so the two are ordered.
/// Frame 0 is set to 1.0 unconditionally, which C does after the loop; the
/// position does not matter because the loop starts at 1 and never writes
/// index 0 (measured — moving the assignment before the loop passes the whole
/// differential). It is kept where C has it.
///
/// Three separate `AOMMAX(_, 0.001)` floors, and they are NOT the same
/// expression: the first is inside a `sqrt` of a product, the second is the
/// divisor, and the third and fourth are the two halves of a ratio that is
/// itself square-rooted. Collapsing any pair changes the result.
pub fn estimate_coeff(stats: &mut [FirstpassStats]) {
    for i in 1..stats.len() {
        let prev_intra = stats[i - 1].intra_error;
        let c = (prev_intra * (stats[i].intra_error - stats[i].coded_error))
            .max(0.001)
            .sqrt();
        let cor_coeff = c / (prev_intra - stats[i].noise_var).max(0.001);
        let ratio = (prev_intra - stats[i].noise_var).max(0.001)
            / (stats[i].intra_error - stats[i].noise_var).max(0.001);
        stats[i].cor_coeff = fclamp(cor_coeff * ratio.sqrt(), 0.0, 1.0);
    }
    if !stats.is_empty() {
        stats[0].cor_coeff = 1.0;
    }
}
