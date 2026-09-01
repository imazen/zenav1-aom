//! The rate-search layer of `av1/encoder/ratectrl.c`: bits-per-frame
//! estimates, the Q↔rate correction factor, and the searches that invert the
//! rate model back into a qindex.
//!
//! [`crate::ratectrl`] owns the `AOM_Q` qindex chain, which never consults a
//! bitrate. This module is the layer beside it that does — it is what
//! `get_q` calls on any non-`AOM_Q` frame, what
//! `adjust_active_best_and_worst_quality` calls to widen the qindex window,
//! and what the recode loop's over/under-shoot limits come from.
//!
//! | Rust | C |
//! |---|---|
//! | [`get_mbs`] | `av1_get_MBs` (common/alloccommon.c:29) |
//! | [`resize_rate_factor`] | `resize_rate_factor` (ratectrl.c:124, static) |
//! | [`RateFactorLevel`] / [`rate_factor_level`] | `RATE_FACTOR_LEVEL` (ratectrl.h:75) / `get_rate_factor_level` (ratectrl.c:818, static) |
//! | [`rate_correction_factor`] | `get_rate_correction_factor` (ratectrl.c:838, static) |
//! | [`estimate_bits_at_q`] | `av1_estimate_bits_at_q` (ratectrl.c:303) |
//! | [`bits_per_mb`] | `get_bits_per_mb` (ratectrl.c:1062, static) |
//! | [`find_qindex_by_rate`] | `find_qindex_by_rate` (ratectrl.c:2653, static) |
//! | [`compute_qdelta_by_rate`] | `av1_compute_qdelta_by_rate` (ratectrl.c:2676) |
//! | [`frame_type_qdelta`] | `frame_type_qdelta` (ratectrl.c:1776, static) |
//! | [`find_closest_qindex_by_rate`] | `find_closest_qindex_by_rate` (ratectrl.c:1088, static) |
//! | [`regulate_q`] | `av1_rc_regulate_q` (ratectrl.c:1138) |
//! | [`compute_frame_size_bounds`] | `av1_rc_compute_frame_size_bounds` (ratectrl.c:2390) |
//! | [`set_frame_target`] | `av1_rc_set_frame_target` (ratectrl.c:2408) |
//!
//! # Scope
//! The `CYCLIC_REFRESH_AQ` arms of `get_bits_per_mb` and
//! `find_closest_qindex_by_rate` (which call into `aq_cyclicrefresh.c`) and
//! the `AOM_CBR` tail of `av1_rc_regulate_q` (`adjust_q_cbr`) are NOT ported:
//! the first needs the cyclic-refresh segment state, the second is CBR. Both
//! are gated by an explicit flag in the C source, and the differential drives
//! the oracle with that flag clear so it is comparing the same arm rather than
//! a zeroed one.
//!
//! Likewise the frame-parallel arms of [`rate_correction_factor`]: C selects
//! between `p_rc->rate_correction_factors` and the per-frame
//! `rc->frame_level_rate_correction_factors` on
//! `gf_group->frame_parallel_level[gf_index] > 0`, and the port is
//! single-threaded, so it takes the `p_rc` array. The port's caller passes
//! that array explicitly, which is why the selection does not appear here.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/ratectrl_rate_diff.rs`. **Tier 1** for the four
//! exported functions ([`estimate_bits_at_q`], [`compute_qdelta_by_rate`],
//! [`regulate_q`], [`compute_frame_size_bounds`], [`set_frame_target`],
//! [`get_mbs`]) — driven out of `upstream/build/libaom.a` through
//! `shim/rcarchive_shim.c`, a TU that does NOT include ratectrl.c. **Tier 1c**
//! for the file-statics, through `shim/ratectrl_shim.c`. The two shims share
//! `shim/rc_state_params.h` and build the same `AV1_COMP`, so the test can
//! also compare tier-1c-against-tier-1 for the two exported functions that
//! appear in both — which is what
//! `rate_search_shim_tu_matches_archive` does.

use crate::rate_model::rc_bits_per_mb;

/// `MIN_BPB_FACTOR` (ratectrl.c:52).
pub const MIN_BPB_FACTOR: f64 = 0.005;
/// `MAX_BPB_FACTOR` (ratectrl.c:53).
pub const MAX_BPB_FACTOR: f64 = 50.0;
/// `FRAME_OVERHEAD_BITS` (ratectrl.c:59).
pub const FRAME_OVERHEAD_BITS: i32 = 200;
/// `BPER_MB_NORMBITS` (ratectrl.h:30).
pub const BPER_MB_NORMBITS: u32 = 9;
/// `RATE_FACTOR_LEVELS` (ratectrl.h:80).
pub const RATE_FACTOR_LEVELS: usize = 4;

/// `MI_SIZE_LOG2` (common/enums.h) — the mi grid is 4 px.
const MI_SIZE_LOG2: i32 = 2;

/// `av1_get_MBs` (common/alloccommon.c:29): the 16x16-macroblock count the
/// rate model normalises by.
///
/// C aligns the frame UP to 8 px, converts to a 4-px mi grid, then rounds the
/// mi counts with `ROUND_POWER_OF_TWO(_, 2)` — a round-HALF-UP rather than a
/// ceiling. **That reduces to `ceil(w/16) * ceil(h/16)` exactly**, and the
/// reduction is worth writing down because it makes two obvious perturbations
/// of this function INERT: the align-to-8 forces `mi` even, so `(mi + 2) >> 2`
/// and `(mi + 3) >> 2` agree for every `mi` this can produce, and
/// `floor((ceil(w/8) + 1) / 2)` is `ceil(w/16)`. Verified against the C symbol
/// for every width and height in `1..=64` and a stride sweep to 4096 in
/// `get_mbs_matches_c`. The line that is NOT inert is the align-to-8 itself.
#[must_use]
pub fn get_mbs(width: i32, height: i32) -> i32 {
    // ALIGN_POWER_OF_TWO(v, 3) == (v + 7) & ~7
    let aligned_width = (width + 7) & !7;
    let aligned_height = (height + 7) & !7;
    let mi_cols = aligned_width >> MI_SIZE_LOG2;
    let mi_rows = aligned_height >> MI_SIZE_LOG2;
    // ROUND_POWER_OF_TWO(v, 2) == (v + 2) >> 2
    let mb_cols = (mi_cols + 2) >> 2;
    let mb_rows = (mi_rows + 2) >> 2;
    mb_rows * mb_cols
}

/// `resize_rate_factor` (ratectrl.c:124): how many times fewer pixels the
/// coded frame has than the configured one.
///
/// C computes both products in `int` before converting to `double`, so the
/// port does the same; at the configured maximum frame size the product still
/// fits, but the widths are deliberately not promoted early.
#[must_use]
pub fn resize_rate_factor(cfg_width: i32, cfg_height: i32, width: i32, height: i32) -> f64 {
    f64::from(cfg_width * cfg_height) / f64::from(width * height)
}

/// `RATE_FACTOR_LEVEL` (encoder/ratectrl.h:75). Discriminants match C, and are
/// the index into `p_rc->rate_correction_factors`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateFactorLevel {
    /// `INTER_NORMAL`.
    InterNormal = 0,
    /// `GF_ARF_LOW`.
    GfArfLow = 1,
    /// `GF_ARF_STD`.
    GfArfStd = 2,
    /// `KF_STD`.
    KfStd = 3,
}

/// `rate_factor_levels[FRAME_UPDATE_TYPES]` (ratectrl.c:808) — which
/// correction-factor slot each frame update type uses.
const RATE_FACTOR_LEVELS_BY_UPDATE: [RateFactorLevel; 7] = [
    RateFactorLevel::KfStd,       // KF_UPDATE
    RateFactorLevel::InterNormal, // LF_UPDATE
    RateFactorLevel::GfArfStd,    // GF_UPDATE
    RateFactorLevel::GfArfStd,    // ARF_UPDATE
    RateFactorLevel::InterNormal, // OVERLAY_UPDATE
    RateFactorLevel::InterNormal, // INTNL_OVERLAY_UPDATE
    RateFactorLevel::GfArfLow,    // INTNL_ARF_UPDATE
];

pub use crate::ref_gop::FrameUpdateType;

/// `get_rate_factor_level` (ratectrl.c:818).
#[must_use]
pub fn rate_factor_level(update_type: FrameUpdateType) -> RateFactorLevel {
    RATE_FACTOR_LEVELS_BY_UPDATE[update_type as usize]
}

/// `arf_layer_deltas[MAX_ARF_LAYERS + 1]` (ratectrl.c:1773).
const ARF_LAYER_DELTAS: [f64; 7] = [2.50, 2.00, 1.75, 1.50, 1.25, 1.15, 1.0];

/// `fclamp` (aom_dsp/aom_dsp_common.h) — C's clamp macro on doubles, which is
/// `low` when the value is below and `high` when above. Not Rust's `clamp`,
/// which panics on a reversed range; here the range is a pair of constants so
/// the distinction is inert, but the shape is kept for the reader.
fn fclamp(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

/// `get_rate_correction_factor` (ratectrl.c:838): the current Q↔bits
/// correction factor, scaled for a downscaled frame and clamped.
///
/// `factors` is `p_rc->rate_correction_factors` (the single-threaded arm; see
/// the module note). `stat_consumption` is `is_stat_consumption_stage(cpi)`.
/// `gf_cbr_boost_pct` only matters under `AOM_CBR`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn rate_correction_factor(
    factors: &[f64; RATE_FACTOR_LEVELS],
    is_key_frame: bool,
    stat_consumption: bool,
    update_type: FrameUpdateType,
    refresh_golden: bool,
    refresh_alt_ref: bool,
    is_src_frame_alt_ref: bool,
    use_svc: bool,
    is_cbr: bool,
    gf_cbr_boost_pct: i32,
    cfg_width: i32,
    cfg_height: i32,
    width: i32,
    height: i32,
) -> f64 {
    let mut rcf = if is_key_frame {
        factors[RateFactorLevel::KfStd as usize]
    } else if stat_consumption {
        factors[rate_factor_level(update_type) as usize]
    } else if (refresh_alt_ref || refresh_golden)
        && !is_src_frame_alt_ref
        && !use_svc
        && (!is_cbr || gf_cbr_boost_pct > 20)
    {
        factors[RateFactorLevel::GfArfStd as usize]
    } else {
        factors[RateFactorLevel::InterNormal as usize]
    };
    rcf *= resize_rate_factor(cfg_width, cfg_height, width, height);
    fclamp(rcf, MIN_BPB_FACTOR, MAX_BPB_FACTOR)
}

/// `av1_estimate_bits_at_q` (ratectrl.c:303): the projected frame size at
/// `q`, floored at `FRAME_OVERHEAD_BITS`.
///
/// The cast order matters and is reproduced exactly. C writes
/// `(int)((uint64_t)bpm * mbs) >> BPER_MB_NORMBITS` — the truncation to `int`
/// happens BEFORE the shift, so a product past `INT_MAX` wraps and then
/// arithmetic-shifts a possibly-negative value. Shifting first and truncating
/// after gives a different answer on large frames at low q.
#[must_use]
pub fn estimate_bits_at_q(
    is_key_frame: bool,
    is_screen_content_type: bool,
    q: i32,
    correction_factor: f64,
    bit_depth: u8,
    mbs: i32,
) -> i32 {
    let bpm = rc_bits_per_mb(
        is_key_frame,
        is_screen_content_type,
        q,
        correction_factor,
        bit_depth,
    );
    let product = (bpm as i64 as u64).wrapping_mul(mbs as i64 as u64);
    FRAME_OVERHEAD_BITS.max((product as i32) >> BPER_MB_NORMBITS)
}

/// `get_bits_per_mb` (ratectrl.c:1062) with `use_cyclic_refresh == 0`: a
/// straight call to `av1_rc_bits_per_mb`.
#[must_use]
pub fn bits_per_mb(
    is_key_frame: bool,
    is_screen_content_type: bool,
    correction_factor: f64,
    q: i32,
    bit_depth: u8,
) -> i32 {
    rc_bits_per_mb(
        is_key_frame,
        is_screen_content_type,
        q,
        correction_factor,
        bit_depth,
    )
}

/// `find_qindex_by_rate` (ratectrl.c:2653): the smallest qindex in
/// `[best, worst]` whose modelled bits-per-mb is at most `desired`.
///
/// C hard-codes `correction_factor = 1.0` and `accurate_estimate = 0` at both
/// call sites inside this function; those are not parameters.
///
/// # Panics
/// Panics if `best_qindex > worst_qindex` (C asserts the same).
#[must_use]
pub fn find_qindex_by_rate(
    desired_bits_per_mb: i32,
    is_key_frame: bool,
    is_screen_content_type: bool,
    bit_depth: u8,
    best_qindex: i32,
    worst_qindex: i32,
) -> i32 {
    assert!(best_qindex <= worst_qindex);
    let mut low = best_qindex;
    let mut high = worst_qindex;
    while low < high {
        let mid = (low + high) >> 1;
        let mid_bits = rc_bits_per_mb(is_key_frame, is_screen_content_type, mid, 1.0, bit_depth);
        if mid_bits > desired_bits_per_mb {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

/// `av1_compute_qdelta_by_rate` (ratectrl.c:2676): the qindex step that scales
/// the modelled rate by `rate_target_ratio`.
///
/// A more general form of the KEY-frame-only helper inside
/// [`crate::allintra_vis`], which fixes `frame_type = KEY_FRAME` because that
/// is all the ALLINTRA delta-q path needs.
#[must_use]
pub fn compute_qdelta_by_rate(
    is_key_frame: bool,
    is_screen_content_type: bool,
    qindex: i32,
    rate_target_ratio: f64,
    bit_depth: u8,
    best_quality: i32,
    worst_quality: i32,
) -> i32 {
    let base_bits_per_mb =
        rc_bits_per_mb(is_key_frame, is_screen_content_type, qindex, 1.0, bit_depth);
    let target_bits_per_mb = (rate_target_ratio * f64::from(base_bits_per_mb)) as i32;
    let target_index = find_qindex_by_rate(
        target_bits_per_mb,
        is_key_frame,
        is_screen_content_type,
        bit_depth,
        best_quality,
        worst_quality,
    );
    target_index - qindex
}

/// `frame_type_qdelta` (ratectrl.c:1776): the qindex delta that gives this
/// frame's pyramid layer its share of the rate.
///
/// `INTER_NORMAL` frames get ratio 1.0 (so a zero delta); everything else
/// takes `arf_layer_deltas[min(layer_depth, 6)]`. The `frame_type` used for
/// the rate model is `gf_group->frame_type[gf_index]`, NOT
/// `cm->current_frame.frame_type` — the two can differ.
// The parameter list is exactly C's read set at this call.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn frame_type_qdelta(
    update_type: FrameUpdateType,
    gf_group_is_key_frame: bool,
    layer_depth: i32,
    is_screen_content_type: bool,
    q: i32,
    bit_depth: u8,
    best_quality: i32,
    worst_quality: i32,
) -> i32 {
    let rf_lvl = rate_factor_level(update_type);
    let arf_layer = layer_depth.min(6);
    let rate_factor = if rf_lvl == RateFactorLevel::InterNormal {
        1.0
    } else {
        ARF_LAYER_DELTAS[usize::try_from(arf_layer).expect("layer_depth must be >= 0")]
    };
    compute_qdelta_by_rate(
        gf_group_is_key_frame,
        is_screen_content_type,
        q,
        rate_factor,
        bit_depth,
        best_quality,
        worst_quality,
    )
}

/// `find_closest_qindex_by_rate` (ratectrl.c:1088) with
/// `use_cyclic_refresh == 0`: like [`find_qindex_by_rate`], but returns
/// whichever of the bracketing qindices has the rate CLOSER to the target.
///
/// The tie-break `curr_bit_diff <= prev_bit_diff` favours the higher qindex.
///
/// # Panics
/// Panics if `best_qindex > worst_qindex` (C asserts the same).
#[must_use]
pub fn find_closest_qindex_by_rate(
    desired_bits_per_mb: i32,
    is_key_frame: bool,
    is_screen_content_type: bool,
    correction_factor: f64,
    bit_depth: u8,
    best_qindex: i32,
    worst_qindex: i32,
) -> i32 {
    assert!(best_qindex <= worst_qindex);
    let bpm = |q: i32| {
        bits_per_mb(
            is_key_frame,
            is_screen_content_type,
            correction_factor,
            q,
            bit_depth,
        )
    };
    let mut low = best_qindex;
    let mut high = worst_qindex;
    while low < high {
        let mid = (low + high) >> 1;
        if bpm(mid) > desired_bits_per_mb {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    let curr_q = low;
    let curr_bits = bpm(curr_q);
    // C's INT_MAX sentinel means "this side has no candidate"; Option says it.
    let curr_bit_diff = (curr_bits <= desired_bits_per_mb).then(|| desired_bits_per_mb - curr_bits);
    let prev_q = curr_q - 1;
    let prev_bit_diff = match curr_bit_diff {
        None => None,
        Some(_) if curr_q == best_qindex => None,
        Some(_) => Some(bpm(prev_q) - desired_bits_per_mb),
    };

    // C: `(curr_bit_diff <= prev_bit_diff) ? curr_q : prev_q` with INT_MAX for
    // "absent". An absent curr never loses to an absent prev (INT_MAX <=
    // INT_MAX is true), so both-absent picks curr.
    match (curr_bit_diff, prev_bit_diff) {
        (Some(c), Some(p)) => {
            if c <= p {
                curr_q
            } else {
                prev_q
            }
        }
        (Some(_), None) => curr_q,
        (None, Some(_)) => prev_q,
        (None, None) => curr_q,
    }
}

/// `av1_rc_regulate_q` (ratectrl.c:1138), non-CBR arm: the qindex in
/// `[active_best, active_worst]` whose modelled rate is closest to
/// `target_bits_per_frame`.
///
/// The `AOM_CBR && has_no_stats_stage` tail (`adjust_q_cbr`) is not ported.
///
/// `(uint64_t)target_bits_per_frame << 9` in C converts a possibly-negative
/// `int` through `uint64_t`, i.e. modulo 2^64; the port does the same rather
/// than saturating.
// The parameter list is exactly C's read set at this call.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn regulate_q(
    target_bits_per_frame: i32,
    active_best_quality: i32,
    active_worst_quality: i32,
    width: i32,
    height: i32,
    is_key_frame: bool,
    is_screen_content_type: bool,
    correction_factor: f64,
    bit_depth: u8,
) -> i32 {
    let mbs = get_mbs(width, height);
    let target_bits_per_mb =
        (((target_bits_per_frame as i64 as u64) << BPER_MB_NORMBITS) / (mbs as i64 as u64)) as i32;
    find_closest_qindex_by_rate(
        target_bits_per_mb,
        is_key_frame,
        is_screen_content_type,
        correction_factor,
        bit_depth,
        active_best_quality,
        active_worst_quality,
    )
}

/// The recode loop's over/under-shoot window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSizeBounds {
    /// `*frame_under_shoot_limit`.
    pub under_shoot: i32,
    /// `*frame_over_shoot_limit`.
    pub over_shoot: i32,
}

/// `av1_rc_compute_frame_size_bounds` (ratectrl.c:2390).
///
/// Under `AOM_Q` the window is unbounded — `[0, INT_MAX]` — which is why the
/// port's fixed-Q envelope never recodes for rate.
#[must_use]
pub fn compute_frame_size_bounds(
    is_q_mode: bool,
    frame_target: i32,
    recode_tolerance: i32,
    max_frame_bandwidth: i32,
) -> FrameSizeBounds {
    if is_q_mode {
        return FrameSizeBounds {
            under_shoot: 0,
            over_shoot: i32::MAX,
        };
    }
    // For very small rate targets the fractional adjustment can be tiny, so
    // there is a floor of 100 bits on the tolerance.
    let tolerance = (i64::from(recode_tolerance) * i64::from(frame_target) / 100).max(100) as i32;
    FrameSizeBounds {
        under_shoot: (frame_target - tolerance).max(0),
        over_shoot: (i64::from(frame_target) + i64::from(tolerance))
            .min(i64::from(max_frame_bandwidth)) as i32,
    }
}

/// The two `RATE_CONTROL` fields `av1_rc_set_frame_target` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameTarget {
    /// `rc->this_frame_target`.
    pub this_frame_target: i32,
    /// `rc->sb64_target_rate`.
    pub sb64_target_rate: i32,
}

/// `saturate_cast_double_to_int` (aom_dsp/aom_dsp_common.h:104): saturate at
/// `INT_MAX` only — the NEGATIVE side is left to C's `(int)` cast, which is
/// UB below `INT_MIN`. The port saturates both ways and says so, because the
/// only inputs that reach the low side are already nonsensical.
fn saturate_cast_double_to_int(d: f64) -> i32 {
    if d > f64::from(i32::MAX) {
        i32::MAX
    } else if d < f64::from(i32::MIN) {
        i32::MIN
    } else {
        d as i32
    }
}

/// `av1_rc_set_frame_target` (ratectrl.c:2408).
///
/// `frame_scaled` is C's `av1_frame_scaled(cm)`; the rescale of the target is
/// skipped under `AOM_CBR`.
#[must_use]
pub fn set_frame_target(
    target: i32,
    width: i32,
    height: i32,
    frame_scaled: bool,
    is_cbr: bool,
    cfg_width: i32,
    cfg_height: i32,
) -> FrameTarget {
    let mut this_frame_target = target;
    if frame_scaled && !is_cbr {
        this_frame_target = saturate_cast_double_to_int(
            f64::from(this_frame_target) * resize_rate_factor(cfg_width, cfg_height, width, height),
        );
    }
    // Target rate per SB64, including partial SB64s.
    let sb64_target_rate = ((i64::from(this_frame_target) << 12) / i64::from(width * height))
        .min(i64::from(i32::MAX)) as i32;
    FrameTarget {
        this_frame_target,
        sb64_target_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_mbs_reduces_to_a_ceiling_division() {
        // The align-to-8 + ROUND_POWER_OF_TWO(_, 2) pair is exactly
        // ceil(w/16) * ceil(h/16). Asserting the identity here is what makes
        // the two rounding perturbations of get_mbs provably inert rather
        // than an untested gap.
        for w in 1..600 {
            for h in [1, 7, 8, 9, 16, 17, 288, 1080] {
                assert_eq!(get_mbs(w, h), ((w + 15) / 16) * ((h + 15) / 16), "{w}x{h}");
            }
        }
    }

    #[test]
    fn rate_factor_level_maps_every_update_type() {
        use FrameUpdateType::*;
        let expect = [
            (Kf, RateFactorLevel::KfStd),
            (Lf, RateFactorLevel::InterNormal),
            (Gf, RateFactorLevel::GfArfStd),
            (Arf, RateFactorLevel::GfArfStd),
            (Overlay, RateFactorLevel::InterNormal),
            (IntnlOverlay, RateFactorLevel::InterNormal),
            (IntnlArf, RateFactorLevel::GfArfLow),
        ];
        for (ty, lvl) in expect {
            assert_eq!(rate_factor_level(ty), lvl, "{ty:?}");
        }
    }

    #[test]
    fn q_mode_frame_size_bounds_are_unbounded() {
        let b = compute_frame_size_bounds(true, 12345, 25, 999);
        assert_eq!(b.under_shoot, 0);
        assert_eq!(b.over_shoot, i32::MAX);
    }
}
