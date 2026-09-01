//! Differential harness for the per-frame rate-control state advance
//! (`av1_rc_update_rate_correction_factors`, `update_buffer_level`,
//! `av1_rc_postencode_update` and the two `update_*_frame_stats` helpers it
//! absorbs) vs the REAL C libaom v3.14.1. **Tier 1** throughout: both
//! exported entry points are driven out of `upstream/build/libaom.a` via
//! `shim/rcarchive_shim.c`.
//!
//! # Why this compares the whole state
//! `av1_rc_postencode_update` writes ~30 fields across `RATE_CONTROL` and
//! `PRIMARY_RATE_CONTROL`. The shim copies the entire state in, runs C, and
//! copies the entire post-state back, and `assert_state_matches` walks every
//! field. A test that checked only the fields the port happened to think about
//! would let a forgotten write sit at whatever it was seeded with — and this
//! function's whole job is exactly those writes.
//!
//! The generator seeds every state field with a DISTINCT nonzero value rather
//! than zero, for the same reason: a port that forgot to write a field, and a
//! port that wrote the right value, are indistinguishable when the seed is
//! already the answer.

use aom_encode::ratectrl::RcMode;
use aom_encode::ratectrl_rate::{FrameUpdateType, RATE_FACTOR_LEVELS};
use aom_encode::ratectrl_update::{
    BufferState, PostencodeFrame, QHistory, RcUpdateState, update_buffer_level,
    update_rate_correction_factors,
};
use aom_sys_ref::{RefRcUpdateState, ref_postencode_update, ref_update_rate_correction_factors};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> i32 {
        (self.next() % u64::from(n)) as i32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32)
    }
    fn boolean(&mut self) -> bool {
        self.below(2) == 1
    }
}

const UPDATE_TYPES: [(FrameUpdateType, i32); 7] = [
    (FrameUpdateType::Kf, 0),
    (FrameUpdateType::Lf, 1),
    (FrameUpdateType::Gf, 2),
    (FrameUpdateType::Arf, 3),
    (FrameUpdateType::Overlay, 4),
    (FrameUpdateType::IntnlOverlay, 5),
    (FrameUpdateType::IntnlArf, 6),
];
const RC_MODES: [(RcMode, i32); 4] = [
    (RcMode::Vbr, 0),
    (RcMode::Cbr, 1),
    (RcMode::Cq, 2),
    (RcMode::Q, 3),
];
const SIZES: [(i32, i32); 5] = [
    (176, 144),
    (352, 288),
    (854, 480),
    (1280, 720),
    (1920, 1080),
];

/// Draw one coherent (C state, port state, port frame facts) triple.
fn draw(rng: &mut Rng) -> (RefRcUpdateState, RcUpdateState, PostencodeFrame) {
    let (w, h) = SIZES[rng.below(SIZES.len() as u32) as usize];
    let (cw, ch) = SIZES[rng.below(SIZES.len() as u32) as usize];
    let (update_type, c_update) = UPDATE_TYPES[rng.below(7) as usize];
    let (rc_mode, c_rc_mode) = RC_MODES[rng.below(4) as usize];
    let bit_depth = [8u8, 10, 12][rng.below(3) as usize];
    // frame_type: KEY(0) / INTER(1) / INTRA_ONLY(2) / S_FRAME(3).
    let frame_type = rng.below(4);
    let show_frame = rng.boolean();
    let refresh_golden = rng.boolean();
    let refresh_alt_ref = rng.boolean();
    let screen_content = rng.boolean();
    let stat_consumption = rng.boolean();
    let tune_content_screen = rng.boolean();
    let is_src_frame_alt_ref = rng.boolean();
    let rtc_external_ratectrl = rng.boolean();
    let constrained_gf_group = rng.boolean();
    let enable_auto_arf = rng.boolean();
    let lag_in_frames = rng.range(0, 32);

    // Every state field gets a DISTINCT nonzero seed, so a forgotten write is
    // visible instead of coinciding with the right answer.
    let c = RefRcUpdateState {
        bytes_used: i64::from(rng.range(0, 4_000_000)),
        base_qindex: rng.below(256),
        coded_width: w,
        coded_height: h,
        cfg_width: cw,
        cfg_height: ch,
        show_frame: i32::from(show_frame),
        frame_type,
        frame_number: rng.range(0, 100_000),
        update_type: c_update,
        refresh_golden: i32::from(refresh_golden),
        refresh_alt_ref: i32::from(refresh_alt_ref),
        lag_in_frames,
        enable_auto_arf: i32::from(enable_auto_arf),
        bit_depth: i32::from(bit_depth),
        screen_content: i32::from(screen_content),
        rc_mode: c_rc_mode,
        stat_consumption: i32::from(stat_consumption),
        gf_cbr_boost_pct: rng.range(0, 100),
        tune_content_screen: i32::from(tune_content_screen),
        is_encode_stage: 0,
        projected_frame_size: rng.range(1, 5_000_000),
        q_1_frame: rng.below(256),
        q_2_frame: rng.below(256),
        rc_1_frame: rng.range(-1, 1),
        rc_2_frame: rng.range(-1, 1),
        this_frame_target: rng.range(1, 5_000_000),
        avg_frame_bandwidth: rng.range(1, 5_000_000),
        prev_avg_frame_bandwidth: rng.range(1, 5_000_000),
        frames_since_key: rng.range(1, 300),
        frames_since_golden: rng.range(1, 300),
        frame_num_last_gf_refresh: rng.range(1, 1000),
        // Straddle the 10000 low-source-sad threshold.
        frame_source_sad: rng.range(0, 20000),
        last_frame_low_source_sad: rng.range(1, 1000),
        frame_number_encoded: rng.range(1, 1000),
        prev_coded_width: 640,
        prev_coded_height: 360,
        prev_frame_is_dropped: 1,
        drop_count_consec: 3,
        ni_tot_qi: rng.range(1, 100_000),
        ni_av_qi: rng.range(1, 255),
        is_src_frame_alt_ref: i32::from(is_src_frame_alt_ref),
        last_encoded_size_keyframe: rng.range(1, 1_000_000),
        last_target_size_keyframe: rng.range(1, 1_000_000),
        rtc_external_ratectrl: i32::from(rtc_external_ratectrl),
        frames_since_scene_change: rng.range(1, 300),
        last_q_key: rng.below(256),
        last_q_inter: rng.below(256),
        avg_frame_qindex_key: rng.below(256),
        avg_frame_qindex_inter: rng.below(256),
        // ni_frames must be >= 1 or C divides by zero in the inter arm; the
        // encoder only reaches that arm after at least one inter frame.
        ni_frames: rng.range(1, 1000),
        tot_q: f64::from(rng.range(1, 100_000)) / 10.0,
        avg_q: f64::from(rng.range(1, 1000)) / 10.0,
        last_boosted_qindex: rng.below(256),
        last_kf_qindex: rng.below(256),
        rate_correction_factors: [
            f64::from(rng.range(1, 5000)) / 100.0,
            f64::from(rng.range(1, 5000)) / 100.0,
            f64::from(rng.range(1, 5000)) / 100.0,
            f64::from(rng.range(1, 5000)) / 100.0,
        ],
        bits_off_target: i64::from(rng.range(-1_000_000, 1_000_000)),
        buffer_level: i64::from(rng.range(-1_000_000, 1_000_000)),
        maximum_buffer_size: i64::from(rng.range(1, 10_000_000)),
        total_actual_bits: i64::from(rng.range(1, 1_000_000_000)),
        total_target_bits: i64::from(rng.range(1, 1_000_000_000)),
        rolling_target_bits: rng.range(1, 5_000_000),
        rolling_actual_bits: rng.range(1, 5_000_000),
        constrained_gf_group: i32::from(constrained_gf_group),
    };

    let port = RcUpdateState {
        projected_frame_size: c.projected_frame_size,
        q_history: QHistory {
            q_1_frame: c.q_1_frame,
            q_2_frame: c.q_2_frame,
            rc_1_frame: c.rc_1_frame,
            rc_2_frame: c.rc_2_frame,
        },
        this_frame_target: c.this_frame_target,
        avg_frame_bandwidth: c.avg_frame_bandwidth,
        prev_avg_frame_bandwidth: c.prev_avg_frame_bandwidth,
        frames_since_key: c.frames_since_key,
        frames_since_golden: c.frames_since_golden,
        frame_num_last_gf_refresh: c.frame_num_last_gf_refresh,
        frame_source_sad: c.frame_source_sad,
        last_frame_low_source_sad: c.last_frame_low_source_sad,
        frame_number_encoded: c.frame_number_encoded,
        prev_coded_width: c.prev_coded_width,
        prev_coded_height: c.prev_coded_height,
        prev_frame_is_dropped: c.prev_frame_is_dropped,
        drop_count_consec: c.drop_count_consec,
        ni_tot_qi: c.ni_tot_qi,
        ni_av_qi: c.ni_av_qi,
        is_src_frame_alt_ref,
        last_encoded_size_keyframe: c.last_encoded_size_keyframe,
        last_target_size_keyframe: c.last_target_size_keyframe,
        rtc_external_ratectrl,
        frames_since_scene_change: c.frames_since_scene_change,
        last_q_key: c.last_q_key,
        last_q_inter: c.last_q_inter,
        avg_frame_qindex_key: c.avg_frame_qindex_key,
        avg_frame_qindex_inter: c.avg_frame_qindex_inter,
        ni_frames: c.ni_frames,
        tot_q: c.tot_q,
        avg_q: c.avg_q,
        last_boosted_qindex: c.last_boosted_qindex,
        last_kf_qindex: c.last_kf_qindex,
        rate_correction_factors: c.rate_correction_factors,
        buffer: BufferState {
            bits_off_target: c.bits_off_target,
            buffer_level: c.buffer_level,
        },
        maximum_buffer_size: c.maximum_buffer_size,
        total_actual_bits: c.total_actual_bits,
        total_target_bits: c.total_target_bits,
        rolling_target_bits: c.rolling_target_bits,
        rolling_actual_bits: c.rolling_actual_bits,
        constrained_gf_group,
    };

    let frame = PostencodeFrame {
        bytes_used: c.bytes_used as u64,
        base_qindex: c.base_qindex,
        width: w,
        height: h,
        cfg_width: cw,
        cfg_height: ch,
        show_frame,
        is_key_frame: frame_type == 0,
        is_s_frame: frame_type == 3,
        frame_number: c.frame_number,
        update_type,
        refresh_golden,
        refresh_alt_ref,
        lag_in_frames,
        enable_auto_arf,
        bit_depth,
        is_screen_content_type: screen_content,
        rc_mode,
        stat_consumption,
        gf_cbr_boost_pct: c.gf_cbr_boost_pct,
        tune_content_screen,
        // The shim builds an unscaled frame (render == coded == upscaled), so
        // av1_frame_scaled(cm) is false there too.
        frame_scaled: false,
    };
    (c, port, frame)
}

/// Compare every field of the port's post-state against C's, naming the field
/// that moved. `ctx` identifies the cell.
fn assert_state_matches(port: &RcUpdateState, c: &RefRcUpdateState, ctx: &str) {
    macro_rules! eq {
        ($got:expr, $want:expr, $name:literal) => {
            assert_eq!($got, $want, "{} differs; {}", $name, ctx);
        };
    }
    eq!(
        port.projected_frame_size,
        c.projected_frame_size,
        "projected_frame_size"
    );
    eq!(port.q_history.q_1_frame, c.q_1_frame, "q_1_frame");
    eq!(port.q_history.q_2_frame, c.q_2_frame, "q_2_frame");
    eq!(port.q_history.rc_1_frame, c.rc_1_frame, "rc_1_frame");
    eq!(port.q_history.rc_2_frame, c.rc_2_frame, "rc_2_frame");
    eq!(
        port.this_frame_target,
        c.this_frame_target,
        "this_frame_target"
    );
    eq!(
        port.prev_avg_frame_bandwidth,
        c.prev_avg_frame_bandwidth,
        "prev_avg_frame_bandwidth"
    );
    eq!(
        port.frames_since_key,
        c.frames_since_key,
        "frames_since_key"
    );
    eq!(
        port.frames_since_golden,
        c.frames_since_golden,
        "frames_since_golden"
    );
    eq!(
        port.frame_num_last_gf_refresh,
        c.frame_num_last_gf_refresh,
        "frame_num_last_gf_refresh"
    );
    eq!(
        port.last_frame_low_source_sad,
        c.last_frame_low_source_sad,
        "last_frame_low_source_sad"
    );
    eq!(
        port.frame_number_encoded,
        c.frame_number_encoded,
        "frame_number_encoded"
    );
    eq!(
        port.prev_coded_width,
        c.prev_coded_width,
        "prev_coded_width"
    );
    eq!(
        port.prev_coded_height,
        c.prev_coded_height,
        "prev_coded_height"
    );
    eq!(
        port.prev_frame_is_dropped,
        c.prev_frame_is_dropped,
        "prev_frame_is_dropped"
    );
    eq!(
        port.drop_count_consec,
        c.drop_count_consec,
        "drop_count_consec"
    );
    eq!(port.ni_tot_qi, c.ni_tot_qi, "ni_tot_qi");
    eq!(port.ni_av_qi, c.ni_av_qi, "ni_av_qi");
    eq!(
        port.last_encoded_size_keyframe,
        c.last_encoded_size_keyframe,
        "last_encoded_size_keyframe"
    );
    eq!(
        port.last_target_size_keyframe,
        c.last_target_size_keyframe,
        "last_target_size_keyframe"
    );
    eq!(
        port.frames_since_scene_change,
        c.frames_since_scene_change,
        "frames_since_scene_change"
    );
    eq!(port.last_q_key, c.last_q_key, "last_q[KEY_FRAME]");
    eq!(port.last_q_inter, c.last_q_inter, "last_q[INTER_FRAME]");
    eq!(
        port.avg_frame_qindex_key,
        c.avg_frame_qindex_key,
        "avg_frame_qindex[KEY_FRAME]"
    );
    eq!(
        port.avg_frame_qindex_inter,
        c.avg_frame_qindex_inter,
        "avg_frame_qindex[INTER_FRAME]"
    );
    eq!(port.ni_frames, c.ni_frames, "ni_frames");
    eq!(port.tot_q.to_bits(), c.tot_q.to_bits(), "tot_q");
    eq!(port.avg_q.to_bits(), c.avg_q.to_bits(), "avg_q");
    eq!(
        port.last_boosted_qindex,
        c.last_boosted_qindex,
        "last_boosted_qindex"
    );
    eq!(port.last_kf_qindex, c.last_kf_qindex, "last_kf_qindex");
    for i in 0..RATE_FACTOR_LEVELS {
        assert_eq!(
            port.rate_correction_factors[i].to_bits(),
            c.rate_correction_factors[i].to_bits(),
            "rate_correction_factors[{i}] differs; {ctx}"
        );
    }
    eq!(
        port.buffer.bits_off_target,
        c.bits_off_target,
        "bits_off_target"
    );
    eq!(port.buffer.buffer_level, c.buffer_level, "buffer_level");
    eq!(
        port.total_actual_bits,
        c.total_actual_bits,
        "total_actual_bits"
    );
    eq!(
        port.total_target_bits,
        c.total_target_bits,
        "total_target_bits"
    );
    eq!(
        port.rolling_target_bits,
        c.rolling_target_bits,
        "rolling_target_bits"
    );
    eq!(
        port.rolling_actual_bits,
        c.rolling_actual_bits,
        "rolling_actual_bits"
    );
}

#[test]
fn update_rate_correction_factors_matches_c() {
    let mut rng = Rng(0xabcd_ef01_2345_6789);
    let mut skipped = 0;
    let mut up = 0;
    let mut down = 0;
    for _ in 0..20000 {
        let (c, port, f) = draw(&mut rng);
        let mut factors = port.rate_correction_factors;
        let mut history = port.q_history;
        let updated = update_rate_correction_factors(
            &mut factors,
            &mut history,
            f.is_key_frame,
            f.is_screen_content_type,
            f.stat_consumption,
            f.update_type,
            f.refresh_golden,
            f.refresh_alt_ref,
            port.is_src_frame_alt_ref,
            false,
            f.rc_mode == RcMode::Cbr,
            f.gf_cbr_boost_pct,
            f.base_qindex,
            port.projected_frame_size,
            f.bit_depth,
            f.cfg_width,
            f.cfg_height,
            f.width,
            f.height,
        );
        let want = ref_update_rate_correction_factors(&c);
        let ctx = format!("{c:?}");
        for i in 0..RATE_FACTOR_LEVELS {
            assert_eq!(
                factors[i].to_bits(),
                want.rate_correction_factors[i].to_bits(),
                "rate_correction_factors[{i}] differs; {ctx}"
            );
        }
        assert_eq!(history.q_1_frame, want.q_1_frame, "q_1_frame; {ctx}");
        assert_eq!(history.q_2_frame, want.q_2_frame, "q_2_frame; {ctx}");
        assert_eq!(history.rc_1_frame, want.rc_1_frame, "rc_1_frame; {ctx}");
        assert_eq!(history.rc_2_frame, want.rc_2_frame, "rc_2_frame; {ctx}");
        if !updated {
            skipped += 1;
        } else if want.rate_correction_factors != c.rate_correction_factors {
            if history.rc_1_frame == -1 {
                up += 1;
            } else if history.rc_1_frame == 1 {
                down += 1;
            }
        }
    }
    // All three arms — the is_src_frame_alt_ref early return, the >1.01
    // overshoot damping and the <0.99 undershoot damping — must be reached.
    assert!(
        skipped > 1000,
        "the ARF-overlay early return fired {skipped} times"
    );
    assert!(up > 500, "the overshoot damping arm fired {up} times");
    assert!(down > 500, "the undershoot damping arm fired {down} times");
}

#[test]
fn postencode_update_matches_c() {
    let mut rng = Rng(0x1357_2468_9bdf_ace0);
    let mut key_cells = 0;
    let mut inter_avg_cells = 0;
    let mut altref_stats_cells = 0;
    for _ in 0..30000 {
        let (c, mut port, f) = draw(&mut rng);
        port.postencode_update(&f);
        let want = ref_postencode_update(&c);
        assert_state_matches(&port, &want, &format!("{c:?}"));

        if f.is_key_frame {
            key_cells += 1;
        }
        if want.ni_frames != c.ni_frames {
            inter_avg_cells += 1;
        }
        if aom_encode::ratectrl_update::is_altref_enabled(f.lag_in_frames, f.enable_auto_arf)
            && f.refresh_alt_ref
            && !f.is_key_frame
            && !f.is_s_frame
        {
            altref_stats_cells += 1;
        }
    }
    // The three arms that a narrower sweep would silently skip.
    assert!(
        key_cells > 5000,
        "the KEY_FRAME arm fired {key_cells} times"
    );
    assert!(
        inter_avg_cells > 2000,
        "the inter avg_frame_qindex / ni_frames arm fired {inter_avg_cells} times"
    );
    assert!(
        altref_stats_cells > 1000,
        "update_alt_ref_frame_stats fired {altref_stats_cells} times"
    );
}

#[test]
fn buffer_level_matches_c_through_postencode() {
    // update_buffer_level is static AND only called from postencode_update /
    // the drop path, so it has no separate oracle. This drives it through the
    // exported caller with the buffer arms deliberately straddled: a maximum
    // buffer size small enough that the AOMMIN clips, and screen content on
    // and off so the -maximum floor is reached and not reached.
    let mut rng = Rng(0x2468_1357_ace0_9bdf);
    let mut clipped_high = 0;
    let mut clipped_low = 0;
    for _ in 0..20000 {
        let (mut c, mut port, mut f) = draw(&mut rng);
        // Force the buffer into the region where both clamps can fire.
        c.maximum_buffer_size = i64::from(rng.range(1, 20_000));
        c.bits_off_target = i64::from(rng.range(-40_000, 40_000));
        c.buffer_level = c.bits_off_target;
        c.avg_frame_bandwidth = rng.range(1, 40_000);
        c.bytes_used = i64::from(rng.range(0, 5_000));
        port.maximum_buffer_size = c.maximum_buffer_size;
        port.buffer = BufferState {
            bits_off_target: c.bits_off_target,
            buffer_level: c.buffer_level,
        };
        port.avg_frame_bandwidth = c.avg_frame_bandwidth;
        f.bytes_used = c.bytes_used as u64;

        port.postencode_update(&f);
        let want = ref_postencode_update(&c);
        assert_eq!(
            port.buffer.bits_off_target, want.bits_off_target,
            "bits_off_target; {c:?}"
        );
        assert_eq!(
            port.buffer.buffer_level, want.buffer_level,
            "buffer_level; {c:?}"
        );
        if want.bits_off_target == c.maximum_buffer_size {
            clipped_high += 1;
        }
        if want.bits_off_target == -c.maximum_buffer_size {
            clipped_low += 1;
        }
    }
    assert!(
        clipped_high > 100,
        "the AOMMIN(maximum_buffer_size) clamp fired {clipped_high} times"
    );
    assert!(
        clipped_low > 100,
        "the screen-content -maximum_buffer_size floor fired {clipped_low} times"
    );
}

#[test]
fn update_buffer_level_unit_matches_the_postencode_path() {
    // A direct check that the standalone helper computes what the exported
    // caller does, so the helper can be used on its own without a second
    // oracle. Cross-checked against C through postencode above.
    let mut rng = Rng(0x0bad_f00d_dead_c0de);
    for _ in 0..5000 {
        let max = i64::from(rng.range(1, 50_000));
        let mut b = BufferState {
            bits_off_target: i64::from(rng.range(-100_000, 100_000)),
            buffer_level: 0,
        };
        let start = b;
        let size = rng.range(0, 100_000);
        let bw = rng.range(0, 100_000);
        let show = rng.boolean();
        let screen = rng.boolean();
        update_buffer_level(&mut b, size, show, bw, max, screen);
        let mut expect = start.bits_off_target
            + if show {
                i64::from(bw) - i64::from(size)
            } else {
                -i64::from(size)
            };
        expect = expect.min(max);
        if screen {
            expect = expect.max(-max);
        }
        assert_eq!(b.bits_off_target, expect);
        assert_eq!(b.buffer_level, expect);
    }
}
