//! Differential harness for rate-control initialisation
//! (`av1_primary_rc_init`, `av1_rc_init`, `set_gf_interval_range`,
//! `av1_rc_update_framerate`) vs the REAL C libaom v3.14.1.
//!
//! **Tier 1** for the three exported functions — driven out of
//! `upstream/build/libaom.a` through `shim/rcarchive_shim.c`, which does not
//! include ratectrl.c. **Tier 1c** for the static `set_gf_interval_range`.
//!
//! # Why this compares whole structs, not one field
//! Each of these functions writes a dozen or more fields. Comparing only the
//! one the port happened to care about would let a forgotten field sit at
//! whatever `calloc` left. So every field the C wrapper can reach is compared,
//! and `rc_init_leaves_the_unported_fields_zero` goes further: the shim
//! PRE-POISONS the eight fields the port does not model, so "C zeroes them" is
//! a measurement rather than an artefact of the allocation.

use aom_encode::ratectrl::RcMode;
use aom_encode::ratectrl_init::{
    RcInitCfg, gf_interval_range, primary_rc_init, rc_init, update_framerate,
};
use aom_encode::ratectrl_rate::RateFactorLevel;
use aom_sys_ref::{
    RefRcInitCfg, ref_primary_rc_init, ref_rc_init, ref_rcc_set_gf_interval_range,
    ref_update_framerate,
};

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
}

const RC_MODES: [(RcMode, i32); 4] = [
    (RcMode::Vbr, 0),
    (RcMode::Cbr, 1),
    (RcMode::Cq, 2),
    (RcMode::Q, 3),
];

/// Frame sizes that straddle `av1_rc_get_default_min_gf_interval`'s
/// "4K at 20 fps" pixel-rate threshold in both directions, since that is the
/// only branch in the default-interval derivation.
const SIZES: [(i32, i32); 7] = [
    (176, 144),
    (352, 288),
    (854, 480),
    (1280, 720),
    (1920, 1080),
    (3840, 2160),
    (7680, 4320),
];

fn draw(rng: &mut Rng) -> (RcInitCfg, RefRcInitCfg, f64, bool) {
    let (rc_mode, c_rc_mode) = RC_MODES[rng.below(4) as usize];
    let (w, h) = SIZES[rng.below(SIZES.len() as u32) as usize];
    let best_allowed_q = rng.below(64);
    let worst_allowed_q = rng.range(best_allowed_q, 255);
    // 0 for both gf intervals means "derive it", which is the interesting arm,
    // so it is drawn far more often than a specific value.
    let min_gf_interval = if rng.below(3) == 0 {
        rng.range(1, 64)
    } else {
        0
    };
    let max_gf_interval = if rng.below(3) == 0 {
        rng.range(1, 300)
    } else {
        0
    };
    // Frame rates spanning the 0.125*fps clamp at both ends, plus a
    // sub-1 fps case where (int)(framerate * 0.125) is 0.
    let init_framerate = f64::from(rng.range(1, 2400)) / 4.0;
    let framerate = f64::from(rng.range(1, 2400)) / 4.0;
    let bit_depth = [8u8, 10, 12][rng.below(3) as usize];
    let one_pass = rng.below(2) == 1;
    // Below SEQ_LEVELS (28) means a level WAS requested, which forces
    // worst_allowed_q to 255 inside av1_primary_rc_init. The draw straddles
    // the boundary tightly on both sides: the first version of this test used
    // 24 as the boundary and the cells in 24..28 are exactly what caught the
    // port's SEQ_LEVELS constant being wrong.
    let target_seq_level_idx0 = if rng.below(2) == 0 {
        rng.range(0, 27)
    } else {
        rng.range(28, 32)
    };
    let starting_buffer_level = i64::from(rng.range(0, 2_000_000_000));
    let lap_enabled = rng.below(2) == 1;
    // Log-ish spread, with a deliberate tail below one bit per frame. The
    // `AOMMAX(1, ...)` floor on rolling_target_bits is only reachable when
    // target_bandwidth / framerate < 1, which a uniform draw over
    // [1, 2e9] hits about six times in a million — i.e. never, and the floor
    // then goes untested. aomenc's own minimum (`--target-bitrate` is in kbps)
    // probably keeps a real encode out of that range, but the port is being
    // compared to the function, not to aomenc's argument validation.
    let target_bandwidth = match rng.below(4) {
        0 => i64::from(rng.range(1, 400)),
        1 => i64::from(rng.range(1, 1_000_000)),
        _ => i64::from(rng.range(1, 2_000_000_000)),
    };

    let cfg = RcInitCfg {
        rc_mode,
        best_allowed_q,
        worst_allowed_q,
        target_bandwidth,
        vbrmin_section: rng.range(0, 200),
        vbrmax_section: rng.range(0, 2000),
        min_gf_interval,
        max_gf_interval,
        fwd_kf_dist: rng.range(0, 300),
        width: w,
        height: h,
        init_framerate,
        bit_depth,
        one_pass,
        target_seq_level_idx0,
    };
    let c = RefRcInitCfg {
        rc_mode: c_rc_mode,
        best_allowed_q: cfg.best_allowed_q,
        worst_allowed_q: cfg.worst_allowed_q,
        target_bandwidth: cfg.target_bandwidth,
        vbrmin_section: cfg.vbrmin_section,
        vbrmax_section: cfg.vbrmax_section,
        min_gf_interval: cfg.min_gf_interval,
        max_gf_interval: cfg.max_gf_interval,
        fwd_kf_dist: cfg.fwd_kf_dist,
        width: cfg.width,
        height: cfg.height,
        init_framerate: cfg.init_framerate,
        bit_depth: i32::from(cfg.bit_depth),
        one_pass: i32::from(cfg.one_pass),
        target_seq_level_idx0: cfg.target_seq_level_idx0,
        starting_buffer_level,
        framerate,
        lap_enabled: i32::from(lap_enabled),
    };
    (cfg, c, framerate, lap_enabled)
}

#[test]
fn primary_rc_init_matches_c() {
    let mut rng = Rng(0x2020_2121_2222_2323);
    let mut level_requested = 0;
    let mut cbr_one_pass = 0;
    for _ in 0..20000 {
        let (cfg, c, _fr, _lap) = draw(&mut rng);
        let got = primary_rc_init(&cfg, c.starting_buffer_level);
        let (wi, wd, wl) = ref_primary_rc_init(&c);
        assert_eq!(
            got.baseline_gf_interval, wi[0],
            "baseline_gf_interval {c:?}"
        );
        assert_eq!(i32::from(got.this_key_frame_forced), wi[1], "{c:?}");
        assert_eq!(i32::from(got.next_key_frame_forced), wi[2], "{c:?}");
        assert_eq!(got.ni_frames, wi[3], "ni_frames {c:?}");
        assert_eq!(
            got.avg_frame_qindex_key, wi[4],
            "avg_frame_qindex[KEY] {c:?}"
        );
        assert_eq!(
            got.avg_frame_qindex_inter, wi[5],
            "avg_frame_qindex[INTER] {c:?}"
        );
        assert_eq!(got.last_q_key, wi[6], "last_q[KEY] {c:?}");
        assert_eq!(got.last_q_inter, wi[7], "last_q[INTER] {c:?}");
        assert_eq!(got.rolling_target_bits, wi[8], "rolling_target_bits {c:?}");
        assert_eq!(got.rolling_actual_bits, wi[9], "rolling_actual_bits {c:?}");
        assert_eq!(got.tot_q.to_bits(), wd[0].to_bits(), "tot_q {c:?}");
        assert_eq!(got.avg_q.to_bits(), wd[1].to_bits(), "avg_q {c:?}");
        for i in 0..4 {
            assert_eq!(
                got.rate_correction_factors[i].to_bits(),
                wd[2 + i].to_bits(),
                "rate_correction_factors[{i}] {c:?}"
            );
        }
        assert_eq!(got.total_actual_bits, wl[0], "{c:?}");
        assert_eq!(got.total_target_bits, wl[1], "{c:?}");
        assert_eq!(got.buffer_level, wl[2], "buffer_level {c:?}");
        assert_eq!(got.bits_off_target, wl[3], "bits_off_target {c:?}");

        if c.target_seq_level_idx0 < 28 {
            level_requested += 1;
        }
        if cfg.one_pass && cfg.rc_mode == RcMode::Cbr {
            cbr_one_pass += 1;
        }
    }
    // Both arms of the two branches inside the function must be reached, or
    // the sweep is only testing one of them.
    assert!(
        level_requested > 5000,
        "the seq-level override arm was reached {level_requested} times"
    );
    assert!(
        cbr_one_pass > 1000,
        "the one-pass CBR avg_frame_qindex arm was reached {cbr_one_pass} times"
    );
}

#[test]
fn kf_std_correction_factor_matches_c() {
    // Pinned separately because it is a single overwritten slot in an
    // otherwise uniform array: a port that filled all four with 0.7 would
    // still pass any test that only checked the array's length.
    let mut rng = Rng(0x3131_4141_5151_6161);
    let (cfg, c, _fr, _lap) = draw(&mut rng);
    let got = primary_rc_init(&cfg, c.starting_buffer_level);
    let (_wi, wd, _wl) = ref_primary_rc_init(&c);
    assert_eq!(
        got.rate_correction_factors[RateFactorLevel::KfStd as usize].to_bits(),
        wd[2 + RateFactorLevel::KfStd as usize].to_bits()
    );
    assert_ne!(
        wd[2 + RateFactorLevel::KfStd as usize],
        wd[2 + RateFactorLevel::InterNormal as usize],
        "C's KF_STD slot is 1.0 while the rest are 0.7 — if they were equal \
         here, this test could not tell a uniform fill from the real one"
    );
}

#[test]
fn rc_init_matches_c() {
    let mut rng = Rng(0x4242_5353_6464_7575);
    for _ in 0..20000 {
        let (cfg, c, _fr, _lap) = draw(&mut rng);
        let got = rc_init(&cfg);
        let want = ref_rc_init(&c);
        assert_eq!(got.frames_since_key, want[0], "frames_since_key {c:?}");
        assert_eq!(got.frames_to_fwd_kf, want[1], "frames_to_fwd_kf {c:?}");
        assert_eq!(got.frames_till_gf_update_due, want[2], "{c:?}");
        assert_eq!(got.ni_av_qi, want[3], "ni_av_qi {c:?}");
        assert_eq!(got.ni_tot_qi, want[4], "ni_tot_qi {c:?}");
        assert_eq!(got.min_gf_interval, want[5], "min_gf_interval {c:?}");
        assert_eq!(got.max_gf_interval, want[6], "max_gf_interval {c:?}");
        assert_eq!(got.avg_frame_low_motion, want[7], "{c:?}");
        assert_eq!(got.resize_avg_qp, want[8], "{c:?}");
        assert_eq!(got.resize_buffer_underflow, want[9], "{c:?}");
        assert_eq!(got.resize_count, want[10], "{c:?}");
        assert_eq!(got.frames_since_scene_change, want[11], "{c:?}");
    }
}

#[test]
fn rc_init_leaves_the_unported_fields_zero() {
    // The shim pre-poisons all eight to 1 before the call, so a zero here is
    // C actually writing one — not the calloc. If any of these ever stopped
    // being zeroed, the port's claim that it "has no equivalent" would become
    // a missing assignment instead of a non-issue.
    let mut rng = Rng(0x9a9a_8b8b_7c7c_6d6d);
    for _ in 0..2000 {
        let (_cfg, c, _fr, _lap) = draw(&mut rng);
        let want = ref_rc_init(&c);
        for (i, name) in [
            "resize_state",
            "rtc_external_ratectrl",
            "frame_level_fast_extra_bits",
            "use_external_qp_one_pass",
            "percent_blocks_inactive",
            "force_max_q",
            "postencode_drop",
            "last_frame_low_source_sad",
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(want[12 + i], 0, "C left {name} non-zero: {c:?}");
        }
    }
}

#[test]
fn gf_interval_range_matches_c() {
    let mut rng = Rng(0x5353_6464_7575_8686);
    let mut lap_cells = 0;
    let mut clamped = 0;
    for _ in 0..20000 {
        let (cfg, c, framerate, lap) = draw(&mut rng);
        let got = gf_interval_range(&cfg, framerate, lap);
        let want = ref_rcc_set_gf_interval_range(&c);
        assert_eq!(
            [
                got.min_gf_interval,
                got.max_gf_interval,
                got.static_scene_max_gf_interval
            ],
            want,
            "{c:?} framerate={framerate} lap={lap}"
        );
        if lap {
            lap_cells += 1;
        }
        if got.max_gf_interval == got.static_scene_max_gf_interval {
            clamped += 1;
        }
    }
    assert!(
        lap_cells > 5000,
        "the LAP arm was reached {lap_cells} times"
    );
    assert!(
        clamped > 100,
        "the static_scene clamp fired only {clamped} times, so the branch \
         that applies it is barely tested"
    );
}

#[test]
fn update_framerate_matches_c() {
    let mut rng = Rng(0x6464_7575_8686_9797);
    let mut maxrate_1080p_wins = 0;
    let mut mb_rate_wins = 0;
    let mut vbr_wins = 0;
    for _ in 0..20000 {
        let (cfg, c, framerate, lap) = draw(&mut rng);
        // The coded size can differ from the configured one, and only the
        // coded size feeds av1_get_MBs here.
        let (w, h) = (cfg.width, cfg.height);
        let got = update_framerate(&cfg, framerate, w, h, lap);
        let want = ref_update_framerate(&c, w, h);
        assert_eq!(
            [
                got.avg_frame_bandwidth,
                got.min_frame_bandwidth,
                got.max_frame_bandwidth,
                got.gf_interval.min_gf_interval,
                got.gf_interval.max_gf_interval,
                got.gf_interval.static_scene_max_gf_interval,
            ],
            want,
            "{c:?} framerate={framerate} {w}x{h}"
        );
        // max_frame_bandwidth is a three-way max; count which term won so the
        // sweep can prove all three are reachable.
        let mbs_term = aom_encode::ratectrl_rate::get_mbs(w, h) * 250;
        let vbr_term = (i64::from(got.avg_frame_bandwidth) * i64::from(cfg.vbrmax_section) / 100)
            .min(i64::from(i32::MAX)) as i32;
        if got.max_frame_bandwidth == vbr_term && vbr_term > mbs_term.max(2_025_000) {
            vbr_wins += 1;
        } else if mbs_term > 2_025_000 {
            mb_rate_wins += 1;
        } else {
            maxrate_1080p_wins += 1;
        }
    }
    assert!(
        maxrate_1080p_wins > 100 && mb_rate_wins > 100 && vbr_wins > 100,
        "the three-way max in max_frame_bandwidth was not exercised on all \
         terms: MAXRATE_1080P={maxrate_1080p_wins} MB_RATE={mb_rate_wins} \
         vbr={vbr_wins}"
    );
}
