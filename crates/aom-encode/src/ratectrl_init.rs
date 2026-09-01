//! Rate-control state initialisation and the frame-rate-derived bandwidth
//! limits — `av1_primary_rc_init`, `av1_rc_init` and `av1_rc_update_framerate`
//! from `av1/encoder/ratectrl.c`.
//!
//! These run once per encode (and again on every frame-rate change) and set
//! the state that [`crate::ratectrl`]'s qindex chain and
//! [`crate::ratectrl_rate`]'s searches then read. Getting them wrong moves the
//! qindex on the very first frame, so they are on the byte-exactness path even
//! though nothing here touches a pixel.
//!
//! | Rust | C (`av1/encoder/ratectrl.c`) |
//! |---|---|
//! | [`PrimaryRateControlInit`] / [`primary_rc_init`] | `av1_primary_rc_init` (:460) |
//! | [`RateControlInit`] / [`rc_init`] | `av1_rc_init` (:514) |
//! | [`GfIntervalRange`] / [`gf_interval_range`] | `set_gf_interval_range` (:2692, static) |
//! | [`FramerateBandwidth`] / [`update_framerate`] | `av1_rc_update_framerate` (:2721) |
//!
//! `get_default_max_gf_interval` and `av1_rc_get_default_min_gf_interval`,
//! which all four call, live in [`crate::ratectrl`] and
//! [`crate::rate_model`] respectively.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/ratectrl_init_diff.rs`. **Tier 1** for
//! `av1_primary_rc_init`, `av1_rc_init` and `av1_rc_update_framerate` (all
//! three are exported; driven out of `upstream/build/libaom.a` through
//! `shim/rcarchive_shim.c`), **tier 1c** for the static
//! `set_gf_interval_range`.

use crate::rate_model::{convert_qindex_to_q, rc_get_default_min_gf_interval};
use crate::ratectrl::{RcMode, default_max_gf_interval};
use crate::ratectrl_rate::{FRAME_OVERHEAD_BITS, RATE_FACTOR_LEVELS, RateFactorLevel, get_mbs};

/// `MAX_STATIC_GF_GROUP_LENGTH` (encoder/ratectrl.h:42).
pub const MAX_STATIC_GF_GROUP_LENGTH: i32 = 250;
/// `MAX_MB_RATE` (ratectrl.c:49) — bits per 16x16 MB per frame, the 1080p
/// hardware-decode baseline.
pub const MAX_MB_RATE: i32 = 250;
/// `MAXRATE_1080P` (ratectrl.c:50).
pub const MAXRATE_1080P: i32 = 2_025_000;
/// `SEQ_LEVELS` (common/enums.h:459) — the number of real level indices; a
/// `target_seq_level_idx` below this means a level was actually requested.
///
/// **28**, not 24: the enum runs `SEQ_LEVEL_2_0` through `SEQ_LEVEL_8_3`, i.e.
/// SEVEN major levels of four minors each, and the 8_x block is easy to miss
/// when skimming. Counted from the C enum, not from memory — the first draft
/// of this constant said 24 and the differential caught it as a wrong
/// `avg_frame_qindex` wherever `target_seq_level_idx[0]` fell in `24..28`.
pub const SEQ_LEVELS: i32 = 28;

/// The `PRIMARY_RATE_CONTROL` fields `av1_primary_rc_init` writes.
///
/// C mutates the struct in place; the port returns the written subset, which
/// makes it explicit that everything else in `PRIMARY_RATE_CONTROL` is
/// untouched by this call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimaryRateControlInit {
    /// `p_rc->baseline_gf_interval`.
    pub baseline_gf_interval: i32,
    /// `p_rc->this_key_frame_forced`.
    pub this_key_frame_forced: bool,
    /// `p_rc->next_key_frame_forced`.
    pub next_key_frame_forced: bool,
    /// `p_rc->ni_frames`.
    pub ni_frames: i32,
    /// `p_rc->tot_q`.
    pub tot_q: f64,
    /// `p_rc->total_actual_bits`.
    pub total_actual_bits: i64,
    /// `p_rc->total_target_bits`.
    pub total_target_bits: i64,
    /// `p_rc->buffer_level`.
    pub buffer_level: i64,
    /// `p_rc->avg_frame_qindex[KEY_FRAME]`.
    pub avg_frame_qindex_key: i32,
    /// `p_rc->avg_frame_qindex[INTER_FRAME]`.
    pub avg_frame_qindex_inter: i32,
    /// `p_rc->avg_q`.
    pub avg_q: f64,
    /// `p_rc->last_q[KEY_FRAME]`.
    pub last_q_key: i32,
    /// `p_rc->last_q[INTER_FRAME]`.
    pub last_q_inter: i32,
    /// `p_rc->rate_correction_factors`, indexed by [`RateFactorLevel`].
    pub rate_correction_factors: [f64; RATE_FACTOR_LEVELS],
    /// `p_rc->bits_off_target`.
    pub bits_off_target: i64,
    /// `p_rc->rolling_target_bits`.
    pub rolling_target_bits: i32,
    /// `p_rc->rolling_actual_bits`.
    pub rolling_actual_bits: i32,
}

/// The `AV1EncoderConfig` fields the three init functions read. One struct for
/// all of them, because they read overlapping subsets and splitting it would
/// make the differential's setup diverge between them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RcInitCfg {
    /// `oxcf.rc_cfg.mode`.
    pub rc_mode: RcMode,
    /// `oxcf.rc_cfg.best_allowed_q` — already a qindex.
    pub best_allowed_q: i32,
    /// `oxcf.rc_cfg.worst_allowed_q` — already a qindex.
    pub worst_allowed_q: i32,
    /// `oxcf.rc_cfg.target_bandwidth`, bits per second.
    pub target_bandwidth: i64,
    /// `oxcf.rc_cfg.vbrmin_section`, percent.
    pub vbrmin_section: i32,
    /// `oxcf.rc_cfg.vbrmax_section`, percent.
    pub vbrmax_section: i32,
    /// `oxcf.gf_cfg.min_gf_interval`; `0` means "derive it".
    pub min_gf_interval: i32,
    /// `oxcf.gf_cfg.max_gf_interval`; `0` means "derive it".
    pub max_gf_interval: i32,
    /// `oxcf.kf_cfg.fwd_kf_dist`.
    pub fwd_kf_dist: i32,
    /// `oxcf.frm_dim_cfg.width`.
    pub width: i32,
    /// `oxcf.frm_dim_cfg.height`.
    pub height: i32,
    /// `oxcf.input_cfg.init_framerate`.
    pub init_framerate: f64,
    /// `oxcf.tool_cfg.bit_depth`.
    pub bit_depth: u8,
    /// `oxcf.pass == AOM_RC_ONE_PASS`.
    pub one_pass: bool,
    /// `oxcf.target_seq_level_idx[0]`.
    pub target_seq_level_idx0: i32,
}

/// `av1_primary_rc_init` (ratectrl.c:460).
///
/// `starting_buffer_level` is `p_rc->starting_buffer_level`, which the caller
/// has already set from the buffer config — C reads it here and copies it into
/// both `buffer_level` and `bits_off_target`.
///
/// Two details worth not "simplifying":
/// * `worst_allowed_q` is overwritten with 255 when ANY sequence level was
///   requested (`target_seq_level_idx[0] < SEQ_LEVELS`), and that override
///   feeds `avg_frame_qindex` and `last_q[INTER_FRAME]` — but NOT `avg_q`,
///   which is computed from the ORIGINAL `rc_cfg.worst_allowed_q`. The two
///   reads are deliberately different in C.
/// * `rate_correction_factors` is filled with 0.7 and then `KF_STD` is
///   overwritten with 1.0, so the KF slot is not 0.7.
#[must_use]
pub fn primary_rc_init(cfg: &RcInitCfg, starting_buffer_level: i64) -> PrimaryRateControlInit {
    let mut worst_allowed_q = cfg.worst_allowed_q;

    let min_gf_interval = if cfg.min_gf_interval == 0 {
        rc_get_default_min_gf_interval(cfg.width, cfg.height, cfg.init_framerate)
    } else {
        cfg.min_gf_interval
    };
    let max_gf_interval = if cfg.max_gf_interval == 0 {
        default_max_gf_interval(cfg.init_framerate, min_gf_interval)
    } else {
        cfg.max_gf_interval
    };

    if cfg.target_seq_level_idx0 < SEQ_LEVELS {
        worst_allowed_q = 255;
    }
    let avg_frame_qindex = if cfg.one_pass && cfg.rc_mode == RcMode::Cbr {
        worst_allowed_q
    } else {
        (worst_allowed_q + cfg.best_allowed_q) / 2
    };

    let mut rate_correction_factors = [0.7f64; RATE_FACTOR_LEVELS];
    rate_correction_factors[RateFactorLevel::KfStd as usize] = 1.0;

    // C: `AOMMAX(1, bits_per_frame > INT_MAX ? INT_MAX : (int)bits_per_frame)`.
    // The comparison is on the DOUBLE, so the truncation happens only on the
    // in-range side.
    let bits_per_frame = cfg.target_bandwidth as f64 / cfg.init_framerate;
    let rolling_target_bits = if bits_per_frame > f64::from(i32::MAX) {
        i32::MAX
    } else {
        (bits_per_frame as i32).max(1)
    };

    PrimaryRateControlInit {
        baseline_gf_interval: (min_gf_interval + max_gf_interval) / 2,
        this_key_frame_forced: false,
        next_key_frame_forced: false,
        ni_frames: 0,
        tot_q: 0.0,
        total_actual_bits: 0,
        total_target_bits: 0,
        buffer_level: starting_buffer_level,
        avg_frame_qindex_key: avg_frame_qindex,
        avg_frame_qindex_inter: avg_frame_qindex,
        // NOTE: the ORIGINAL configured worst_allowed_q, not the 255 override.
        avg_q: convert_qindex_to_q(cfg.worst_allowed_q, cfg.bit_depth),
        last_q_key: cfg.best_allowed_q,
        last_q_inter: cfg.worst_allowed_q,
        rate_correction_factors,
        bits_off_target: starting_buffer_level,
        rolling_target_bits,
        rolling_actual_bits: rolling_target_bits,
    }
}

/// The `RATE_CONTROL` fields `av1_rc_init` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RateControlInit {
    /// `rc->frames_since_key` — seeded to 8, C's "sensible default for the
    /// first frame". A zero here changes `get_active_best_quality`'s
    /// `frames_since_key > 1` test on frame 0.
    pub frames_since_key: i32,
    /// `rc->frames_to_fwd_kf`.
    pub frames_to_fwd_kf: i32,
    /// `rc->frames_till_gf_update_due`.
    pub frames_till_gf_update_due: i32,
    /// `rc->ni_av_qi`.
    pub ni_av_qi: i32,
    /// `rc->ni_tot_qi`.
    pub ni_tot_qi: i32,
    /// `rc->min_gf_interval`.
    pub min_gf_interval: i32,
    /// `rc->max_gf_interval`.
    pub max_gf_interval: i32,
    /// `rc->avg_frame_low_motion`.
    pub avg_frame_low_motion: i32,
    /// `rc->resize_avg_qp`.
    pub resize_avg_qp: i32,
    /// `rc->resize_buffer_underflow`.
    pub resize_buffer_underflow: i32,
    /// `rc->resize_count`.
    pub resize_count: i32,
    /// `rc->frames_since_scene_change`.
    pub frames_since_scene_change: i32,
}

/// `av1_rc_init` (ratectrl.c:514).
///
/// The fields C sets to plain zero and the port does not return
/// (`resize_state`, `rtc_external_ratectrl`, `frame_level_fast_extra_bits`,
/// `use_external_qp_one_pass`, `percent_blocks_inactive`, `force_max_q`,
/// `postencode_drop`, `last_frame_low_source_sad`) are RT/resize state the
/// port has no equivalent of; they are zero on a freshly built `RATE_CONTROL`
/// either way. The differential asserts C leaves them zero rather than
/// assuming it.
#[must_use]
pub fn rc_init(cfg: &RcInitCfg) -> RateControlInit {
    let min_gf_interval = if cfg.min_gf_interval == 0 {
        rc_get_default_min_gf_interval(cfg.width, cfg.height, cfg.init_framerate)
    } else {
        cfg.min_gf_interval
    };
    let max_gf_interval = if cfg.max_gf_interval == 0 {
        default_max_gf_interval(cfg.init_framerate, min_gf_interval)
    } else {
        cfg.max_gf_interval
    };
    RateControlInit {
        frames_since_key: 8,
        frames_to_fwd_kf: cfg.fwd_kf_dist,
        frames_till_gf_update_due: 0,
        ni_av_qi: cfg.worst_allowed_q,
        ni_tot_qi: 0,
        min_gf_interval,
        max_gf_interval,
        avg_frame_low_motion: 0,
        resize_avg_qp: 0,
        resize_buffer_underflow: 0,
        resize_count: 0,
        frames_since_scene_change: 0,
    }
}

/// The three GF-interval fields `set_gf_interval_range` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GfIntervalRange {
    /// `rc->min_gf_interval`.
    pub min_gf_interval: i32,
    /// `rc->max_gf_interval`.
    pub max_gf_interval: i32,
    /// `rc->static_scene_max_gf_interval`.
    pub static_scene_max_gf_interval: i32,
}

/// `set_gf_interval_range` (ratectrl.c:2692).
///
/// `framerate` is `cpi->framerate` (the RUNNING frame rate), which is not
/// `oxcf.input_cfg.init_framerate` — after the first frame-rate update the two
/// differ, and `av1_primary_rc_init` / `av1_rc_init` use the init value while
/// this uses the running one.
#[must_use]
pub fn gf_interval_range(cfg: &RcInitCfg, framerate: f64, lap_enabled: bool) -> GfIntervalRange {
    let min = if cfg.min_gf_interval == 0 {
        rc_get_default_min_gf_interval(cfg.width, cfg.height, framerate)
    } else {
        cfg.min_gf_interval
    };
    let mut max = if cfg.max_gf_interval == 0 {
        default_max_gf_interval(framerate, min)
    } else {
        cfg.max_gf_interval
    };
    // Extended max interval for genuinely static scenes like slide shows. The
    // number of stats available under LAP is limited, hence max_gf_interval.
    let static_scene_max_gf_interval = if lap_enabled {
        max + 1
    } else {
        MAX_STATIC_GF_GROUP_LENGTH
    };
    if max > static_scene_max_gf_interval {
        max = static_scene_max_gf_interval;
    }
    GfIntervalRange {
        min_gf_interval: min.min(max),
        max_gf_interval: max,
        static_scene_max_gf_interval,
    }
}

/// The bandwidth limits `av1_rc_update_framerate` writes, plus the GF range it
/// delegates to [`gf_interval_range`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramerateBandwidth {
    /// `rc->avg_frame_bandwidth`.
    pub avg_frame_bandwidth: i32,
    /// `rc->min_frame_bandwidth`.
    pub min_frame_bandwidth: i32,
    /// `rc->max_frame_bandwidth`.
    pub max_frame_bandwidth: i32,
    /// The GF interval range, written by the tail call to
    /// `set_gf_interval_range`.
    pub gf_interval: GfIntervalRange,
}

/// `saturate_cast_double_to_int` (aom_dsp/aom_dsp_common.h:104): saturates at
/// `INT_MAX` only. C's `(int)` cast is UB below `INT_MIN`; the port saturates
/// there too and says so, because a negative bandwidth is already nonsensical.
fn saturate_cast_double_to_int(d: f64) -> i32 {
    if d > f64::from(i32::MAX) {
        i32::MAX
    } else if d < f64::from(i32::MIN) {
        i32::MIN
    } else {
        d as i32
    }
}

/// `av1_rc_update_framerate` (ratectrl.c:2721).
///
/// `framerate` is `cpi->framerate`, the running rate. The `round()` on the
/// average is a real round-half-away-from-zero, not a truncation — the
/// truncating `(int)` is applied to the ALREADY-rounded double.
#[must_use]
pub fn update_framerate(
    cfg: &RcInitCfg,
    framerate: f64,
    width: i32,
    height: i32,
    lap_enabled: bool,
) -> FramerateBandwidth {
    let mbs = get_mbs(width, height);

    let avg_frame_bandwidth =
        saturate_cast_double_to_int((cfg.target_bandwidth as f64 / framerate).round());

    let vbr_min_bits = (i64::from(avg_frame_bandwidth) * i64::from(cfg.vbrmin_section) / 100)
        .min(i64::from(i32::MAX));
    let min_frame_bandwidth = (vbr_min_bits as i32).max(FRAME_OVERHEAD_BITS);

    // The frame maximum aligns with hardware that decodes 1080p at MAX_MB_RATE
    // bits per 16x16 MB, but is extended when the command line asks for more.
    let vbr_max_bits = (i64::from(avg_frame_bandwidth) * i64::from(cfg.vbrmax_section) / 100)
        .min(i64::from(i32::MAX));
    let max_frame_bandwidth = (mbs.saturating_mul(MAX_MB_RATE))
        .max(MAXRATE_1080P)
        .max(vbr_max_bits as i32);

    FramerateBandwidth {
        avg_frame_bandwidth,
        min_frame_bandwidth,
        max_frame_bandwidth,
        gf_interval: gf_interval_range(cfg, framerate, lap_enabled),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RcInitCfg {
        RcInitCfg {
            rc_mode: RcMode::Q,
            best_allowed_q: 0,
            worst_allowed_q: 255,
            target_bandwidth: 1_000_000,
            vbrmin_section: 0,
            vbrmax_section: 2000,
            min_gf_interval: 0,
            max_gf_interval: 0,
            fwd_kf_dist: 0,
            width: 352,
            height: 288,
            init_framerate: 30.0,
            bit_depth: 8,
            one_pass: true,
            target_seq_level_idx0: 31, // SEQ_LEVEL_MAX: no level requested
        }
    }

    #[test]
    fn kf_std_correction_factor_is_one_not_point_seven() {
        let p = primary_rc_init(&cfg(), 0);
        assert_eq!(
            p.rate_correction_factors[RateFactorLevel::KfStd as usize],
            1.0
        );
        for lvl in [
            RateFactorLevel::InterNormal,
            RateFactorLevel::GfArfLow,
            RateFactorLevel::GfArfStd,
        ] {
            assert_eq!(p.rate_correction_factors[lvl as usize], 0.7, "{lvl:?}");
        }
    }

    #[test]
    fn a_requested_seq_level_forces_worst_q_to_255_but_not_avg_q() {
        let mut c = cfg();
        c.worst_allowed_q = 100;
        c.best_allowed_q = 20;
        // No level requested: avg_frame_qindex is (100 + 20) / 2.
        let no_level = primary_rc_init(&c, 0);
        assert_eq!(no_level.avg_frame_qindex_key, 60);
        // A level requested: worst becomes 255, so (255 + 20) / 2.
        c.target_seq_level_idx0 = 0;
        let with_level = primary_rc_init(&c, 0);
        assert_eq!(with_level.avg_frame_qindex_key, 137);
        // ...but avg_q reads the ORIGINAL worst_allowed_q in both cases.
        assert_eq!(no_level.avg_q.to_bits(), with_level.avg_q.to_bits());
        // ...and so does last_q[INTER_FRAME].
        assert_eq!(with_level.last_q_inter, 100);
    }

    #[test]
    fn rc_init_seeds_frames_since_key_to_eight() {
        // Not zero: get_active_best_quality's `frames_since_key > 1` test
        // reads this on the very first inter frame.
        assert_eq!(rc_init(&cfg()).frames_since_key, 8);
    }
}
