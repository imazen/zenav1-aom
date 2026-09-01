//! The per-frame rate-control state advance —
//! `av1_rc_update_rate_correction_factors`, `update_buffer_level` and
//! `av1_rc_postencode_update` from `av1/encoder/ratectrl.c`.
//!
//! This is what runs between frames. Everything [`crate::ratectrl`] reads to
//! pick the next frame's qindex — `avg_frame_qindex`, `last_boosted_qindex`,
//! `last_kf_qindex`, `frames_since_key`, the correction factors — is written
//! here. A GOP cannot be byte-exact past frame 1 without it.
//!
//! | Rust | C (`av1/encoder/ratectrl.c`) |
//! |---|---|
//! | [`set_rate_correction_factor`] | `set_rate_correction_factor` (:900, static) |
//! | [`update_rate_correction_factors`] | `av1_rc_update_rate_correction_factors` (:940) |
//! | [`update_buffer_level`] | `update_buffer_level` (:386, static) |
//! | [`RcUpdateState::postencode_update`] | `av1_rc_postencode_update` (:2444) |
//! | (absorbed) | `update_alt_ref_frame_stats` (:2427), `update_golden_frame_stats` (:2433) |
//! | [`set_target_rate`] | `av1_set_target_rate` (:2831) |
//!
//! # Scope
//! The arms NOT ported, each gated by a flag the differential drives clear so
//! it is comparing the same arm rather than a zeroed one:
//! * `CYCLIC_REFRESH_AQ` — `av1_cyclic_refresh_estimate_bits_at_q` and the
//!   `percent_refresh_adjustment` block need the cyclic-refresh segment state.
//! * The `FAST_DETECTION_MAXQ` scene-change early return, which is
//!   RT-overshoot detection.
//! * The `AOM_CBR && accurate_bit_estimate` `bit_est_ratio` update.
//! * `update_layer_buffer_level` (SVC) and every `use_svc` read.
//! * `CONFIG_FPMT_TEST`'s `temp_*` shadow copies, which are compiled out.
//! * `vbr_rate_correction` inside `av1_set_target_rate`, which needs the
//!   two-pass stats buffer; [`set_target_rate`] therefore covers the
//!   `AOM_Q` / `AOM_CBR` arm only and says so.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/ratectrl_update_diff.rs`. **Tier 1** — both
//! exported functions are driven out of `upstream/build/libaom.a` through
//! `shim/rcarchive_shim.c`, which copies the whole state in, runs the real C,
//! and copies the whole post-state back, so every field is compared rather
//! than the ones the port happened to care about.

use crate::rate_model::convert_qindex_to_q;
use crate::ratectrl::RcMode;
use crate::ratectrl_rate::{
    FRAME_OVERHEAD_BITS, FrameUpdateType, MAX_BPB_FACTOR, MIN_BPB_FACTOR, RATE_FACTOR_LEVELS,
    RateFactorLevel, estimate_bits_at_q, get_mbs, rate_correction_factor, rate_factor_level,
    resize_rate_factor,
};

/// `ALT_MIN_LAG` (encoder/encoder.h) — the lag below which alt-ref is off.
pub const ALT_MIN_LAG: i32 = 3;

/// `is_altref_enabled` (encoder/encoder.h:4110).
#[must_use]
pub fn is_altref_enabled(lag_in_frames: i32, enable_auto_arf: bool) -> bool {
    lag_in_frames >= ALT_MIN_LAG && enable_auto_arf
}

fn fclamp(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

/// `ROUND_POWER_OF_TWO(value, 2)` on an `i32` — `(v + 2) >> 2`, an arithmetic
/// shift, so it rounds toward negative infinity on a negative value the way
/// C's `>>` on a signed int does in practice.
fn round_power_of_two_2(value: i32) -> i32 {
    (value + 2) >> 2
}

/// `ROUND_POWER_OF_TWO_64(value, 2)`.
fn round_power_of_two_64_2(value: i64) -> i64 {
    (value + 2) >> 2
}

/// The static predicate both `get_rate_correction_factor` and
/// `set_rate_correction_factor` branch on for a non-key, non-stat-consumption
/// frame: "is this a GF/ARF frame whose factor lives in the `GF_ARF_STD` slot?"
///
/// Written once here because C spells it out twice, identically, and the two
/// copies must not drift.
#[must_use]
fn uses_gf_arf_std_slot(
    refresh_golden: bool,
    refresh_alt_ref: bool,
    is_src_frame_alt_ref: bool,
    use_svc: bool,
    is_cbr: bool,
    gf_cbr_boost_pct: i32,
) -> bool {
    (refresh_alt_ref || refresh_golden)
        && !is_src_frame_alt_ref
        && !use_svc
        && (!is_cbr || gf_cbr_boost_pct > 20)
}

/// `set_rate_correction_factor` (ratectrl.c:900), single-threaded arm.
///
/// Writes `factor`, normalised for the frame's downscale and clamped, into the
/// slot the frame's type selects — the SAME slot
/// [`crate::ratectrl_rate::rate_correction_factor`] reads. C's
/// `is_encode_stage`/`frame_parallel_level` split, which would route the write
/// to `rc->frame_level_rate_correction_factors` instead, is not ported: the
/// port is single-threaded, so `frame_parallel_level` is always 0 and
/// `update_default_rcf` stays 1.
#[allow(clippy::too_many_arguments)]
pub fn set_rate_correction_factor(
    factors: &mut [f64; RATE_FACTOR_LEVELS],
    factor: f64,
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
) {
    // Normalise out the size-dependent scaling factor that
    // get_rate_correction_factor multiplied in.
    let factor = factor / resize_rate_factor(cfg_width, cfg_height, width, height);
    let factor = fclamp(factor, MIN_BPB_FACTOR, MAX_BPB_FACTOR);

    let slot = if is_key_frame {
        RateFactorLevel::KfStd
    } else if stat_consumption {
        rate_factor_level(update_type)
    } else if uses_gf_arf_std_slot(
        refresh_golden,
        refresh_alt_ref,
        is_src_frame_alt_ref,
        use_svc,
        is_cbr,
        gf_cbr_boost_pct,
    ) {
        RateFactorLevel::GfArfStd
    } else {
        RateFactorLevel::InterNormal
    };
    factors[slot as usize] = factor;
}

/// The four `RATE_CONTROL` history fields
/// `av1_rc_update_rate_correction_factors` writes alongside the factor itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QHistory {
    /// `rc->q_1_frame` — the qindex of the frame just coded.
    pub q_1_frame: i32,
    /// `rc->q_2_frame` — the one before that.
    pub q_2_frame: i32,
    /// `rc->rc_1_frame` — `-1` overshoot, `+1` undershoot, `0` on target.
    pub rc_1_frame: i32,
    /// `rc->rc_2_frame`.
    pub rc_2_frame: i32,
}

/// `av1_rc_update_rate_correction_factors` (ratectrl.c:940), the
/// non-cyclic-refresh, non-scene-change arm.
///
/// Returns `false` (and touches nothing) on C's `is_src_frame_alt_ref` early
/// return — the rate factors are deliberately not updated for an ARF overlay.
///
/// Two details worth not "tidying":
/// * `adjustment_limit` uses `0.5 * min(0.5, |log10(cf)|)` for screen content
///   and `0.75 * ...` otherwise — a 0.25 floor either way, but a different
///   slope, so a single constant is wrong for one of the two.
/// * the `< 0.99` arm inverts the correction factor, damps, and inverts BACK.
///   Damping `cf` directly is not the same function.
#[allow(clippy::too_many_arguments)]
pub fn update_rate_correction_factors(
    factors: &mut [f64; RATE_FACTOR_LEVELS],
    history: &mut QHistory,
    is_key_frame: bool,
    is_screen_content_type: bool,
    stat_consumption: bool,
    update_type: FrameUpdateType,
    refresh_golden: bool,
    refresh_alt_ref: bool,
    is_src_frame_alt_ref: bool,
    use_svc: bool,
    is_cbr: bool,
    gf_cbr_boost_pct: i32,
    base_qindex: i32,
    projected_frame_size: i32,
    bit_depth: u8,
    cfg_width: i32,
    cfg_height: i32,
    width: i32,
    height: i32,
) -> bool {
    // Do not update the rate factors for ARF overlay frames.
    if is_src_frame_alt_ref {
        return false;
    }

    let mut rate_correction_factor = rate_correction_factor(
        factors,
        is_key_frame,
        stat_consumption,
        update_type,
        refresh_golden,
        refresh_alt_ref,
        is_src_frame_alt_ref,
        use_svc,
        is_cbr,
        gf_cbr_boost_pct,
        cfg_width,
        cfg_height,
        width,
        height,
    );

    // How big the frame would have been at this Q under the current factor.
    // Kept in double to avoid int overflow when the values are large.
    let mbs = get_mbs(width, height);
    let projected_size_based_on_q = estimate_bits_at_q(
        is_key_frame,
        is_screen_content_type,
        base_qindex,
        rate_correction_factor,
        bit_depth,
        mbs,
    );

    let mut correction_factor = 1.0f64;
    if projected_size_based_on_q > FRAME_OVERHEAD_BITS {
        correction_factor = f64::from(projected_frame_size) / f64::from(projected_size_based_on_q);
    }
    // Clamp to prevent anything too extreme.
    correction_factor = correction_factor.max(0.25);

    history.q_2_frame = history.q_1_frame;
    history.q_1_frame = base_qindex;
    history.rc_2_frame = history.rc_1_frame;
    history.rc_1_frame = if correction_factor > 1.1 {
        -1
    } else if correction_factor < 0.9 {
        1
    } else {
        0
    };

    // How heavily to dampen the adjustment.
    let adjustment_limit = if correction_factor > 0.0 {
        let slope = if is_screen_content_type { 0.5 } else { 0.75 };
        0.25 + slope * correction_factor.log10().abs().min(0.5)
    } else {
        0.75
    };

    if correction_factor > 1.01 {
        // Not already at the worst allowable quality.
        let damped = 1.0 + ((correction_factor - 1.0) * adjustment_limit);
        rate_correction_factor *= damped;
        if rate_correction_factor > MAX_BPB_FACTOR {
            rate_correction_factor = MAX_BPB_FACTOR;
        }
    } else if correction_factor < 0.99 {
        // Not already at the best allowable quality. Invert, damp, invert back.
        let inverted = 1.0 / correction_factor;
        let damped = 1.0 / (1.0 + ((inverted - 1.0) * adjustment_limit));
        rate_correction_factor *= damped;
        if rate_correction_factor < MIN_BPB_FACTOR {
            rate_correction_factor = MIN_BPB_FACTOR;
        }
    }

    set_rate_correction_factor(
        factors,
        rate_correction_factor,
        is_key_frame,
        stat_consumption,
        update_type,
        refresh_golden,
        refresh_alt_ref,
        is_src_frame_alt_ref,
        use_svc,
        is_cbr,
        gf_cbr_boost_pct,
        cfg_width,
        cfg_height,
        width,
        height,
    );
    true
}

/// The leaky-bucket buffer state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferState {
    /// `p_rc->bits_off_target`.
    pub bits_off_target: i64,
    /// `p_rc->buffer_level`.
    pub buffer_level: i64,
}

/// `update_buffer_level` (ratectrl.c:386), non-SVC arm.
///
/// A non-shown frame is pure overhead, so its size is subtracted without the
/// per-frame credit. Screen content gets a floor at
/// `-maximum_buffer_size` so the level can come back up faster after a slide
/// change with a big overshoot.
pub fn update_buffer_level(
    buffer: &mut BufferState,
    encoded_frame_size: i32,
    show_frame: bool,
    avg_frame_bandwidth: i32,
    maximum_buffer_size: i64,
    tune_content_screen: bool,
) {
    if show_frame {
        buffer.bits_off_target += i64::from(avg_frame_bandwidth) - i64::from(encoded_frame_size);
    } else {
        buffer.bits_off_target -= i64::from(encoded_frame_size);
    }
    buffer.bits_off_target = buffer.bits_off_target.min(maximum_buffer_size);
    if tune_content_screen {
        buffer.bits_off_target = buffer.bits_off_target.max(-maximum_buffer_size);
    }
    buffer.buffer_level = buffer.bits_off_target;
}

/// The full rate-control state `av1_rc_postencode_update` reads and writes.
///
/// C spreads these across `RATE_CONTROL` and `PRIMARY_RATE_CONTROL`; the port
/// gathers exactly the ones this function touches, so the type documents the
/// footprint of a single frame's state advance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RcUpdateState {
    /// `rc->projected_frame_size`.
    pub projected_frame_size: i32,
    /// `rc->q_1_frame` / `q_2_frame` / `rc_1_frame` / `rc_2_frame`.
    pub q_history: QHistory,
    /// `rc->this_frame_target`.
    pub this_frame_target: i32,
    /// `rc->avg_frame_bandwidth`.
    pub avg_frame_bandwidth: i32,
    /// `rc->prev_avg_frame_bandwidth`.
    pub prev_avg_frame_bandwidth: i32,
    /// `rc->frames_since_key`.
    pub frames_since_key: i32,
    /// `rc->frames_since_golden`.
    pub frames_since_golden: i32,
    /// `rc->frame_num_last_gf_refresh`.
    pub frame_num_last_gf_refresh: i32,
    /// `rc->frame_source_sad` (read only, in the low-sad test).
    pub frame_source_sad: i32,
    /// `rc->last_frame_low_source_sad`.
    pub last_frame_low_source_sad: i32,
    /// `rc->frame_number_encoded`.
    pub frame_number_encoded: i32,
    /// `rc->prev_coded_width` / `prev_coded_height`.
    pub prev_coded_width: i32,
    /// See [`RcUpdateState::prev_coded_width`].
    pub prev_coded_height: i32,
    /// `rc->prev_frame_is_dropped`.
    pub prev_frame_is_dropped: i32,
    /// `rc->drop_count_consec`.
    pub drop_count_consec: i32,
    /// `rc->ni_tot_qi`.
    pub ni_tot_qi: i32,
    /// `rc->ni_av_qi`.
    pub ni_av_qi: i32,
    /// `rc->is_src_frame_alt_ref` (read only here).
    pub is_src_frame_alt_ref: bool,
    /// `rc->last_encoded_size_keyframe`.
    pub last_encoded_size_keyframe: i32,
    /// `rc->last_target_size_keyframe`.
    pub last_target_size_keyframe: i32,
    /// `rc->rtc_external_ratectrl` (read only here).
    pub rtc_external_ratectrl: bool,
    /// `rc->frames_since_scene_change`.
    pub frames_since_scene_change: i32,
    /// `p_rc->last_q[KEY_FRAME]`.
    pub last_q_key: i32,
    /// `p_rc->last_q[INTER_FRAME]`.
    pub last_q_inter: i32,
    /// `p_rc->avg_frame_qindex[KEY_FRAME]`.
    pub avg_frame_qindex_key: i32,
    /// `p_rc->avg_frame_qindex[INTER_FRAME]`.
    pub avg_frame_qindex_inter: i32,
    /// `p_rc->ni_frames`.
    pub ni_frames: i32,
    /// `p_rc->tot_q`.
    pub tot_q: f64,
    /// `p_rc->avg_q`.
    pub avg_q: f64,
    /// `p_rc->last_boosted_qindex`.
    pub last_boosted_qindex: i32,
    /// `p_rc->last_kf_qindex`.
    pub last_kf_qindex: i32,
    /// `p_rc->rate_correction_factors`.
    pub rate_correction_factors: [f64; RATE_FACTOR_LEVELS],
    /// The leaky-bucket state.
    pub buffer: BufferState,
    /// `p_rc->maximum_buffer_size` (read only).
    pub maximum_buffer_size: i64,
    /// `p_rc->total_actual_bits`.
    pub total_actual_bits: i64,
    /// `p_rc->total_target_bits`.
    pub total_target_bits: i64,
    /// `p_rc->rolling_target_bits`.
    pub rolling_target_bits: i32,
    /// `p_rc->rolling_actual_bits`.
    pub rolling_actual_bits: i32,
    /// `p_rc->constrained_gf_group` (read only).
    pub constrained_gf_group: bool,
}

/// The per-frame facts `av1_rc_postencode_update` reads that are not
/// rate-control state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostencodeFrame {
    /// `bytes_used`, the coded frame size in BYTES.
    pub bytes_used: u64,
    /// `cm->quant_params.base_qindex`.
    pub base_qindex: i32,
    /// `cm->width` / `cm->height`.
    pub width: i32,
    /// See [`PostencodeFrame::width`].
    pub height: i32,
    /// `oxcf.frm_dim_cfg.width` / `.height`.
    pub cfg_width: i32,
    /// See [`PostencodeFrame::cfg_width`].
    pub cfg_height: i32,
    /// `cm->show_frame`.
    pub show_frame: bool,
    /// `cm->current_frame.frame_type == KEY_FRAME`.
    pub is_key_frame: bool,
    /// `frame_is_sframe(cm)`.
    pub is_s_frame: bool,
    /// `cm->current_frame.frame_number`.
    pub frame_number: i32,
    /// `gf_group->update_type[gf_index]`.
    pub update_type: FrameUpdateType,
    /// `cpi->refresh_frame.golden_frame`.
    pub refresh_golden: bool,
    /// `cpi->refresh_frame.alt_ref_frame`.
    pub refresh_alt_ref: bool,
    /// `oxcf.gf_cfg.lag_in_frames`.
    pub lag_in_frames: i32,
    /// `oxcf.gf_cfg.enable_auto_arf`.
    pub enable_auto_arf: bool,
    /// `cm->seq_params->bit_depth`.
    pub bit_depth: u8,
    /// `cpi->is_screen_content_type`.
    pub is_screen_content_type: bool,
    /// `oxcf.rc_cfg.mode`.
    pub rc_mode: RcMode,
    /// `is_stat_consumption_stage(cpi)`.
    pub stat_consumption: bool,
    /// `oxcf.rc_cfg.gf_cbr_boost_pct`.
    pub gf_cbr_boost_pct: i32,
    /// `oxcf.tune_cfg.content == AOM_CONTENT_SCREEN`.
    pub tune_content_screen: bool,
    /// `av1_frame_scaled(cm)`.
    pub frame_scaled: bool,
}

impl RcUpdateState {
    /// `av1_rc_postencode_update` (ratectrl.c:2444).
    ///
    /// The order of the writes matters and is preserved: the rate-correction
    /// update runs BEFORE the qindex history is recorded (it reads
    /// `projected_frame_size`, which this function has just set), and the
    /// `this_frame_target` rescale for a downscaled frame runs BEFORE the
    /// rolling-bits update that consumes it.
    pub fn postencode_update(&mut self, f: &PostencodeFrame) {
        let qindex = f.base_qindex;
        let is_intrnl_arf = f.update_type == FrameUpdateType::IntnlArf;

        self.projected_frame_size = (f.bytes_used << 3) as i32;

        // Post-encode adjustment of the Q prediction.
        update_rate_correction_factors(
            &mut self.rate_correction_factors,
            &mut self.q_history,
            f.is_key_frame,
            f.is_screen_content_type,
            f.stat_consumption,
            f.update_type,
            f.refresh_golden,
            f.refresh_alt_ref,
            self.is_src_frame_alt_ref,
            /*use_svc=*/ false,
            f.rc_mode == RcMode::Cbr,
            f.gf_cbr_boost_pct,
            qindex,
            self.projected_frame_size,
            f.bit_depth,
            f.cfg_width,
            f.cfg_height,
            f.width,
            f.height,
        );

        // Record the last Q and the ambient average Q.
        if f.is_key_frame {
            self.last_q_key = qindex;
            self.avg_frame_qindex_key =
                round_power_of_two_2(3 * self.avg_frame_qindex_key + qindex);
            // C guards this on svc.spatial_layer_id == 0, which is always true
            // for a non-SVC encode.
            self.last_encoded_size_keyframe = self.projected_frame_size;
            self.last_target_size_keyframe = self.this_frame_target;
        } else if self.rtc_external_ratectrl
            || (!self.is_src_frame_alt_ref
                && !(f.refresh_golden || is_intrnl_arf || f.refresh_alt_ref))
        {
            self.last_q_inter = qindex;
            self.avg_frame_qindex_inter =
                round_power_of_two_2(3 * self.avg_frame_qindex_inter + qindex);
            self.ni_frames += 1;
            self.tot_q += convert_qindex_to_q(qindex, f.bit_depth);
            self.avg_q = self.tot_q / f64::from(self.ni_frames);
            // The average Q over normal inter frames only (not key or GFU).
            self.ni_tot_qi += qindex;
            self.ni_av_qi = self.ni_tot_qi / self.ni_frames;
        }

        // Record the last boosted (KF/GF/ARF) Q. Also update it when the
        // current frame is coded at a lower Q, so forced key frames can match
        // quality and reduce popping.
        if qindex < self.last_boosted_qindex
            || f.is_key_frame
            || (!self.constrained_gf_group
                && (f.refresh_alt_ref
                    || is_intrnl_arf
                    || (f.refresh_golden && !self.is_src_frame_alt_ref)))
        {
            self.last_boosted_qindex = qindex;
        }
        if f.is_key_frame {
            self.last_kf_qindex = qindex;
        }

        update_buffer_level(
            &mut self.buffer,
            self.projected_frame_size,
            f.show_frame,
            self.avg_frame_bandwidth,
            self.maximum_buffer_size,
            f.tune_content_screen,
        );
        self.prev_avg_frame_bandwidth = self.avg_frame_bandwidth;

        // Rolling monitors of over/under spend, used to regulate min and max Q
        // in two pass.
        if f.frame_scaled {
            self.this_frame_target = saturate_cast_double_to_int(
                f64::from(self.this_frame_target)
                    / resize_rate_factor(f.cfg_width, f.cfg_height, f.width, f.height),
            );
        }
        if !f.is_key_frame {
            self.rolling_target_bits = round_power_of_two_64_2(
                i64::from(self.rolling_target_bits) * 3 + i64::from(self.this_frame_target),
            ) as i32;
            self.rolling_actual_bits = round_power_of_two_64_2(
                i64::from(self.rolling_actual_bits) * 3 + i64::from(self.projected_frame_size),
            ) as i32;
        }

        self.total_actual_bits += i64::from(self.projected_frame_size);
        self.total_target_bits += if f.show_frame {
            i64::from(self.avg_frame_bandwidth)
        } else {
            0
        };

        if is_altref_enabled(f.lag_in_frames, f.enable_auto_arf)
            && f.refresh_alt_ref
            && !f.is_key_frame
            && !f.is_s_frame
        {
            // update_alt_ref_frame_stats: this frame refreshes, so the next
            // ones do not unless the user says so.
            self.frames_since_golden = 0;
        } else {
            // update_golden_frame_stats.
            if f.refresh_golden || self.is_src_frame_alt_ref {
                self.frames_since_golden = 0;
            } else if f.show_frame {
                self.frames_since_golden += 1;
            }
        }

        if f.is_key_frame {
            self.frames_since_key = 0;
            self.frames_since_scene_change = 0;
        }
        if f.refresh_golden {
            self.frame_num_last_gf_refresh = f.frame_number;
        }
        if self.frame_source_sad < 10000 {
            self.last_frame_low_source_sad = self.frame_number_encoded;
        }
        self.prev_coded_width = f.width;
        self.prev_coded_height = f.height;
        self.frame_number_encoded += 1;
        self.prev_frame_is_dropped = 0;
        self.drop_count_consec = 0;
    }
}

/// `saturate_cast_double_to_int` (aom_dsp/aom_dsp_common.h:104).
fn saturate_cast_double_to_int(d: f64) -> i32 {
    if d > f64::from(i32::MAX) {
        i32::MAX
    } else if d < f64::from(i32::MIN) {
        i32::MIN
    } else {
        d as i32
    }
}

/// `av1_set_target_rate` (ratectrl.c:2831), the `AOM_Q` / `AOM_CBR` arm.
///
/// Under `AOM_VBR` / `AOM_CQ` C first runs `vbr_rate_correction`, which reads
/// the two-pass stats buffer; that arm is not ported, and this function
/// asserts the mode rather than silently taking the uncorrected path.
///
/// # Panics
/// Panics on `AOM_VBR` / `AOM_CQ`, where `vbr_rate_correction` would run.
#[must_use]
pub fn set_target_rate(
    base_frame_target: i32,
    rc_mode: RcMode,
    width: i32,
    height: i32,
    frame_scaled: bool,
    cfg_width: i32,
    cfg_height: i32,
) -> crate::ratectrl_rate::FrameTarget {
    assert!(
        matches!(rc_mode, RcMode::Q | RcMode::Cbr),
        "av1_set_target_rate runs vbr_rate_correction under AOM_VBR / AOM_CQ, \
         which needs the two-pass stats buffer and is not ported"
    );
    crate::ratectrl_rate::set_frame_target(
        base_frame_target,
        width,
        height,
        frame_scaled,
        rc_mode == RcMode::Cbr,
        cfg_width,
        cfg_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arf_overlay_frames_do_not_move_the_rate_factors() {
        let mut factors = [0.7, 0.7, 0.7, 1.0];
        let before = factors;
        let mut history = QHistory::default();
        let updated = update_rate_correction_factors(
            &mut factors,
            &mut history,
            false,
            false,
            false,
            FrameUpdateType::Overlay,
            true,
            false,
            /*is_src_frame_alt_ref=*/ true,
            false,
            false,
            0,
            128,
            10_000,
            8,
            352,
            288,
            352,
            288,
        );
        assert!(!updated);
        assert_eq!(factors, before);
        assert_eq!(
            history,
            QHistory::default(),
            "the q history must not move either"
        );
    }

    #[test]
    fn buffer_level_treats_a_hidden_frame_as_pure_overhead() {
        let mut b = BufferState {
            bits_off_target: 1000,
            buffer_level: 1000,
        };
        update_buffer_level(
            &mut b,
            400,
            /*show_frame=*/ false,
            500,
            i64::MAX,
            false,
        );
        assert_eq!(
            b.bits_off_target, 600,
            "no per-frame credit on a hidden frame"
        );
        let mut b = BufferState {
            bits_off_target: 1000,
            buffer_level: 1000,
        };
        update_buffer_level(&mut b, 400, /*show_frame=*/ true, 500, i64::MAX, false);
        assert_eq!(b.bits_off_target, 1100);
    }

    #[test]
    fn screen_content_floors_the_buffer_at_minus_max() {
        let mut b = BufferState::default();
        update_buffer_level(&mut b, 1_000_000, true, 0, 5000, /*screen=*/ true);
        assert_eq!(b.bits_off_target, -5000);
        let mut b = BufferState::default();
        update_buffer_level(&mut b, 1_000_000, true, 0, 5000, /*screen=*/ false);
        assert_eq!(b.bits_off_target, -1_000_000);
    }
}
