//! The rate-control rate model — the qindex↔Q conversions and the
//! bits-per-macroblock estimate every fixed-Q decision is built on.
//! Port of the corresponding parts of `av1/encoder/ratectrl.c`.
//!
//! The port's existing [`crate::rc`] covers only the lone-KEY `AOM_Q` qindex
//! derivation. Multi-frame rate control needs the model underneath it, which is
//! what this module is.
//!
//! | Rust | C |
//! |---|---|
//! | [`convert_qindex_to_q`] | `av1_convert_qindex_to_q` (ratectrl.c:199) |
//! | [`find_qindex`] | `av1_find_qindex` (ratectrl.c:2619) |
//! | [`compute_qdelta`] | `av1_compute_qdelta` (ratectrl.c:2638) |
//! | [`rc_bits_per_mb`] | `av1_rc_bits_per_mb` (ratectrl.c:273) |
//! | [`rc_get_default_min_gf_interval`] | `av1_rc_get_default_min_gf_interval` |
//! | [`bpmb_enumerator`] | `get_bpmb_enumerator` (static) |
//!
//! # Scope: the non-CBR arm of `av1_rc_bits_per_mb`
//! C's `av1_rc_bits_per_mb` has two CBR-only overrides of the enumerator — an
//! `accurate_estimate` path keyed on `cpi->rec_sse`, and a real-time keyframe
//! adjustment. **Neither is ported**, because both are guarded by
//! `cpi->oxcf.rc_cfg.mode == AOM_CBR` and the encode target here is
//! `--end-usage=q`. [`rc_bits_per_mb`] therefore takes no CBR state, and the
//! differential drives the C oracle with `mode = AOM_Q` so it is comparing the
//! same arm rather than a zeroed CBR one. If CBR ever becomes a goal, those two
//! overrides are the delta.
//!
//! # Differential coverage
//! `tests/rate_model_diff.rs`, tier 1 against the real exported C.

/// `MIN_GF_INTERVAL` (encoder/ratectrl.h:44).
const MIN_GF_INTERVAL: i32 = 4;
/// `MAX_GF_INTERVAL` (ratectrl.h:45).
const MAX_GF_INTERVAL: i32 = 32;

/// `av1_convert_qindex_to_q` (ratectrl.c:199): the AC quantizer at `qindex`,
/// scaled down to the "old Q" range.
///
/// The three bit depths divide by 4, 16 and 64 — the same AC quantizer, scaled
/// by a different power of four, so a bd-8 formula reused at bd 10/12 is off by
/// a factor of 4 or 16, not by rounding.
#[must_use]
pub fn convert_qindex_to_q(qindex: i32, bit_depth: u8) -> f64 {
    let ac = f64::from(aom_dsp::quant::av1_ac_quant_qtx(qindex, 0, bit_depth));
    match bit_depth {
        8 => ac / 4.0,
        10 => ac / 16.0,
        12 => ac / 64.0,
        _ => panic!("av1_convert_qindex_to_q: bit_depth must be 8, 10 or 12, got {bit_depth}"),
    }
}

/// `av1_find_qindex` (ratectrl.c:2619): the smallest qindex in
/// `[best_qindex, worst_qindex]` whose Q is `>= desired_q`, by binary search.
///
/// The search returns `worst_qindex` when no index reaches `desired_q`, so the
/// result is a clamp, not a failure.
#[must_use]
pub fn find_qindex(desired_q: f64, bit_depth: u8, best_qindex: i32, worst_qindex: i32) -> i32 {
    assert!(best_qindex <= worst_qindex);
    let mut low = best_qindex;
    let mut high = worst_qindex;
    while low < high {
        let mid = (low + high) >> 1;
        if convert_qindex_to_q(mid, bit_depth) < desired_q {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

/// `av1_compute_qdelta` (ratectrl.c:2638): the qindex step from `qstart` to
/// `qtarget`, as a difference of two [`find_qindex`] lookups.
#[must_use]
pub fn compute_qdelta(
    qstart: f64,
    qtarget: f64,
    bit_depth: u8,
    best_quality: i32,
    worst_quality: i32,
) -> i32 {
    let start_index = find_qindex(qstart, bit_depth, best_quality, worst_quality);
    let target_index = find_qindex(qtarget, bit_depth, best_quality, worst_quality);
    target_index - start_index
}

/// `get_bpmb_enumerator` (ratectrl.c, static): the rate-model numerator, keyed
/// on frame type and whether the content was classified as screen content.
///
/// `is_key_frame` is `frame_type == KEY_FRAME`.
#[must_use]
pub fn bpmb_enumerator(is_key_frame: bool, is_screen_content_type: bool) -> i32 {
    if is_screen_content_type {
        if is_key_frame { 1_000_000 } else { 750_000 }
    } else if is_key_frame {
        2_000_000
    } else {
        1_500_000
    }
}

/// `av1_rc_bits_per_mb` (ratectrl.c:273), non-CBR arm: the projected bits per
/// macroblock at `qindex`.
///
/// `enumerator * correction_factor / q`, truncated toward zero by C's
/// `(int)` cast on a `double`. See the module note for the two CBR-only
/// enumerator overrides that are deliberately not ported.
#[must_use]
pub fn rc_bits_per_mb(
    is_key_frame: bool,
    is_screen_content_type: bool,
    qindex: i32,
    correction_factor: f64,
    bit_depth: u8,
) -> i32 {
    let q = convert_qindex_to_q(qindex, bit_depth);
    let enumerator = f64::from(bpmb_enumerator(is_key_frame, is_screen_content_type));
    (enumerator * correction_factor / q) as i32
}

/// `av1_rc_get_default_min_gf_interval` (ratectrl.c): the minimum golden-frame
/// interval for a given resolution and frame rate.
///
/// Below the "4K at 20 fps" pixel-rate threshold the answer is just the clamped
/// `framerate * 0.125`; above it, the interval scales with the pixel rate. The
/// `+ 0.5` inside the `(int)` cast is a truncating round-half-up on a positive
/// value, not a `round()`.
#[must_use]
pub fn rc_get_default_min_gf_interval(width: i32, height: i32, framerate: f64) -> i32 {
    // Assume we do not need any constraint lower than 4K 20 fps.
    const FACTOR_SAFE: f64 = 3840.0 * 2160.0 * 20.0;
    let factor = f64::from(width) * f64::from(height) * framerate;
    let default_interval = ((framerate * 0.125) as i32).clamp(MIN_GF_INTERVAL, MAX_GF_INTERVAL);
    if factor <= FACTOR_SAFE {
        default_interval
    } else {
        default_interval.max((f64::from(MIN_GF_INTERVAL) * factor / FACTOR_SAFE + 0.5) as i32)
    }
}
