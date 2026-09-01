//! Differential harness for `aom_encode::pass2_model` — the error and boost
//! arithmetic the 2-pass bit allocator makes every decision out of.
//!
//! **Tier 1c throughout.** `nm -g` reports seven exported symbols for
//! `av1/encoder/pass2_strategy.c`'s 87 definitions, and none of these is among
//! them, so tier 1 is unavailable. `shim/pass2_shim.c` compiles that .c
//! verbatim under its Release flags (including `-ffp-contract=off`, which for
//! a file of pure `double` arithmetic is what makes the oracle mean the same
//! thing on x86 and aarch64) and wraps the statics.
//!
//! Every comparison here is on the exact BIT PATTERN of the `double`, not a
//! tolerance: the whole point of the harness is that the two produce the same
//! value, and a tolerance would hide precisely the drift it exists to catch.
//!
//! | test | C oracle |
//! |---|---|
//! | `firstpass_stats_layout_matches_c` | `FIRSTPASS_STATS` (`firstpass.h:43`) |
//! | `calculate_active_area_matches_c` | `calculate_active_area` (:61) |
//! | `calculate_modified_err_new_matches_c` | `calculate_modified_err_new` (:73) |
//! | `frame_max_bits_matches_c` | `frame_max_bits` (:154) |
//! | `calc_correction_factor_matches_c` | `calc_correction_factor` (:171) |
//! | `qbpm_enumerator_matches_c` | `qbpm_enumerator` (:288) |
//! | `decay_rates_match_c` | `get_sr_decay_rate` / `_zero_motion_factor` / `_prediction_decay_rate` |
//! | `baseline_err_per_mb_matches_c` | `baseline_err_per_mb` (:574) |
//! | `calc_frame_boost_matches_c` | `calc_frame_boost` (:590) |
//! | `calc_kf_frame_boost_matches_c` | `calc_kf_frame_boost` (:622) |
//! | `boost_bits_and_factor_match_c` | `calculate_boost_bits` (:836) / `calculate_boost_factor` (:861) |
//! | `get_projected_gfu_boost_matches_c` | `get_projected_gfu_boost` (:653) |
//! | `is_almost_static_matches_c` | `is_almost_static` (:999) |
//!
//! # What bounds the generators
//! The first pass produces every input here, so the ranges come from it:
//! * the `pcnt_*` fields are FRACTIONS in `[0, 1]`;
//! * `coded_error`, `intra_error` and `sr_coded_error` are per-frame error
//!   sums, non-negative, and `sr_coded_error >= coded_error` is the normal
//!   ordering (the reverse is drawn too, because a flash inverts it);
//! * `mv_in_out_count` is documented `[-1.0, +1.0]` at `calc_frame_boost`'s
//!   own comment, and BOTH signs are swept because they take different arms;
//! * `q` is a real qindex, `0..=255`;
//! * `inactive_zone_rows` is in MB units and cannot exceed `mb_rows`.
//!
//! `calc_correction_factor`'s sweep walks every qindex, because the exponent
//! is interpolated across 32-wide bands and only a full walk crosses every
//! band boundary.

use aom_encode::pass2_model::{
    FirstpassStats, FrameInfo, baseline_err_per_mb, calc_correction_factor, calc_frame_boost,
    calc_kf_frame_boost, calculate_active_area, calculate_boost_bits, calculate_boost_factor,
    calculate_modified_err_new, frame_max_bits, get_prediction_decay_rate, get_projected_gfu_boost,
    get_sr_decay_rate, get_zero_motion_factor, is_almost_static, qbpm_enumerator,
};
use aom_sys_ref::{
    P2FrameInfo, ref_p2_baseline_err_per_mb, ref_p2_calc_correction_factor,
    ref_p2_calc_frame_boost, ref_p2_calc_kf_frame_boost, ref_p2_calculate_active_area,
    ref_p2_calculate_boost_bits, ref_p2_calculate_boost_factor, ref_p2_calculate_modified_err_new,
    ref_p2_firstpass_stats_doubles, ref_p2_frame_max_bits, ref_p2_get_prediction_decay_rate,
    ref_p2_get_projected_gfu_boost, ref_p2_get_sr_decay_rate, ref_p2_get_zero_motion_factor,
    ref_p2_is_almost_static, ref_p2_qbpm_enumerator, ref_p2_tu_qbpm_enumerator,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
    /// A uniform `f64` in `[0, 1]`.
    fn unit(&mut self) -> f64 {
        f64::from(self.below(1 << 24)) / f64::from(1u32 << 24)
    }
    /// A uniform `f64` in `[lo, hi]`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }
}

/// Compare on the exact bit pattern. Both NaNs compare equal here, which is
/// what "the two computed the same thing" means for a value that is then
/// stored and re-read.
fn bits_eq(a: f64, b: f64, what: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what}: port {a:?} ({:#x}) vs C {b:?} ({:#x})",
        a.to_bits(),
        b.to_bits()
    );
}

/// Frame geometries spanning the 640x360 corner `baseline_err_per_mb` tests.
fn frame_infos() -> Vec<(FrameInfo, P2FrameInfo)> {
    let mut out = Vec::new();
    for &(w, h) in &[
        (176i32, 144i32),
        (640, 360),
        (640, 361),
        (854, 480),
        (1920, 1080),
        (3840, 2160),
    ] {
        let mb_rows = (h + 15) / 16;
        let mb_cols = (w + 15) / 16;
        let num_mbs = mb_rows * mb_cols;
        for &bd in &[8i32, 10, 12] {
            out.push((
                FrameInfo {
                    frame_width: w,
                    frame_height: h,
                    mb_rows,
                    mb_cols,
                    num_mbs,
                    bit_depth: bd as u8,
                },
                P2FrameInfo {
                    frame_width: w,
                    frame_height: h,
                    mb_rows,
                    mb_cols,
                    num_mbs,
                    bit_depth: bd,
                },
            ));
        }
    }
    out
}

/// A first-pass stats record with the ranges the first pass really produces.
/// `mb_rows` bounds `inactive_zone_rows`, which is in MB units.
fn draw_stats(rng: &mut Rng, mb_rows: i32) -> FirstpassStats {
    let pcnt_inter = rng.unit();
    // pcnt_motion and pcnt_neutral are subsets of the inter fraction.
    let pcnt_motion = rng.range(0.0, pcnt_inter);
    let pcnt_neutral = rng.range(0.0, pcnt_inter);
    let coded_error = match rng.below(6) {
        0 => 0.0,
        1 => rng.range(0.0, 0.02), // straddles LOW_CODED_ERR_PER_MB
        _ => rng.range(1.0, 5.0e6),
    };
    // Normally sr >= coded; a flash inverts it, and get_sr_decay_rate's
    // LOW_SR_DIFF_TRHESH guard turns on exactly there.
    let sr_coded_error = if rng.below(4) == 0 {
        rng.range(0.0, coded_error)
    } else {
        coded_error + rng.range(0.0, 1.0e6)
    };
    FirstpassStats {
        frame: f64::from(rng.below(1 << 16)),
        weight: rng.range(0.1, 4.0),
        // intra/coded near NCOUNT_FRAME_II_THRESH = 5.0 keeps that guard live.
        intra_error: coded_error * rng.range(0.1, 12.0) + rng.range(0.0, 1.0),
        frame_avg_wavelet_energy: rng.range(0.0, 1.0e6),
        coded_error,
        sr_coded_error,
        lt_coded_error: rng.range(0.0, 5.0e6),
        pcnt_inter,
        pcnt_motion,
        pcnt_second_ref: rng.unit(),
        pcnt_neutral,
        intra_skip_pct: rng.unit(),
        inactive_zone_rows: rng.range(0.0, f64::from(mb_rows) / 2.0),
        inactive_zone_cols: rng.range(0.0, 4.0),
        mvr: rng.range(-64.0, 64.0),
        mvr_abs: rng.range(0.0, 64.0),
        mvc: rng.range(-64.0, 64.0),
        mvc_abs: rng.range(0.0, 64.0),
        mvrv: rng.range(0.0, 1.0e4),
        mvcv: rng.range(0.0, 1.0e4),
        mv_in_out_count: rng.range(-1.0, 1.0),
        new_mv_count: rng.range(0.0, 1.0e4),
        duration: rng.range(1.0, 1.0e6),
        count: rng.range(1.0, 300.0),
        raw_error_stdev: rng.range(0.0, 1.0e3),
        is_flash: i64::from(rng.below(2)),
        noise_var: rng.range(0.0, 1.0e3),
        cor_coeff: rng.unit(),
        log_intra_error: rng.range(0.0, 20.0),
        log_coded_error: rng.range(0.0, 20.0),
    }
}

#[test]
fn firstpass_stats_layout_matches_c() {
    // Every other test in this file sends the stats as a flat array in
    // declaration order. If C's struct ever gained or lost a double, that
    // array would silently be misaligned against it.
    assert_eq!(
        FirstpassStats::default().to_doubles().len(),
        ref_p2_firstpass_stats_doubles(),
        "FIRSTPASS_STATS double count drifted"
    );
    // The TU's own copy of a pure function, reached through a second entry
    // point, must agree with the first -- the tier-1c consistency probe.
    for tol in [-50i32, 0, 25, 26, 99, 100, 1000] {
        assert_eq!(ref_p2_qbpm_enumerator(tol), ref_p2_tu_qbpm_enumerator(tol));
    }
}

#[test]
fn calculate_active_area_matches_c() {
    let mut rng = Rng::new(0xACA1_0001);
    for (fi, cfi) in frame_infos() {
        for _ in 0..200 {
            let s = draw_stats(&mut rng, fi.mb_rows);
            bits_eq(
                calculate_active_area(&fi, &s),
                ref_p2_calculate_active_area(&cfi, &s.to_doubles(), s.is_flash),
                "active_area",
            );
        }
    }
    // Both clamp arms, constructed: a frame with no inactive zone and no intra
    // skip is fully active (clamps at MAX); one with a large letterbox clamps
    // at MIN.
    let (fi, cfi) = frame_infos()[0];
    let mut s = FirstpassStats::default();
    bits_eq(
        calculate_active_area(&fi, &s),
        ref_p2_calculate_active_area(&cfi, &s.to_doubles(), 0),
        "active_area at MAX",
    );
    assert_eq!(calculate_active_area(&fi, &s), 1.0);
    s.inactive_zone_rows = f64::from(fi.mb_rows);
    assert_eq!(
        calculate_active_area(&fi, &s),
        0.5,
        "must clamp at MIN_ACTIVE_AREA"
    );
    bits_eq(
        calculate_active_area(&fi, &s),
        ref_p2_calculate_active_area(&cfi, &s.to_doubles(), 0),
        "active_area at MIN",
    );
}

#[test]
fn calculate_modified_err_new_matches_c() {
    let mut rng = Rng::new(0xE447_0002);
    for (fi, cfi) in frame_infos() {
        for _ in 0..120 {
            let total = draw_stats(&mut rng, fi.mb_rows);
            let this = draw_stats(&mut rng, fi.mb_rows);
            // vbrbias is 0..100 in the encoder config.
            for &vbrbias in &[0i32, 25, 50, 75, 100] {
                let err_min = rng.range(0.0, 100.0);
                let err_max = err_min + rng.range(0.0, 1.0e7);
                bits_eq(
                    calculate_modified_err_new(&fi, Some(&total), &this, vbrbias, err_min, err_max),
                    ref_p2_calculate_modified_err_new(
                        &cfi,
                        Some(&total.to_doubles()),
                        &this.to_doubles(),
                        vbrbias,
                        err_min,
                        err_max,
                    ),
                    "modified_err",
                );
            }
            // The NULL total_stats arm.
            bits_eq(
                calculate_modified_err_new(&fi, None, &this, 50, 0.0, 1.0e9),
                ref_p2_calculate_modified_err_new(&cfi, None, &this.to_doubles(), 50, 0.0, 1.0e9),
                "modified_err with NULL totals",
            );
        }
    }
}

#[test]
fn frame_max_bits_matches_c() {
    let mut rng = Rng::new(0x8175_0003);
    for _ in 0..4000 {
        // avg_frame_bandwidth is bits per frame; max_frame_bandwidth is the
        // per-frame ceiling the buffer model imposes. Negative avg is drawn
        // because C explicitly clamps it to 0.
        let avg = match rng.below(5) {
            0 => -i64::from(rng.below(1 << 20)),
            1 => 0,
            _ => i64::from(rng.below(1 << 24)),
        };
        let max = i64::from(rng.below(1 << 25));
        let vbrmax = rng.below(2001) as i32;
        assert_eq!(
            frame_max_bits(avg, max, vbrmax),
            ref_p2_frame_max_bits(avg, max, vbrmax),
            "avg {avg} max {max} vbrmax {vbrmax}"
        );
    }
}

#[test]
fn calc_correction_factor_matches_c() {
    let mut rng = Rng::new(0xC0FA_0004);
    // Every qindex: the exponent is interpolated across 32-wide bands and only
    // a full walk crosses every boundary.
    for q in 0..=255i32 {
        for &err in &[0.0f64, 0.001, 1.0, 96.0, 500.0, 5000.0, 1.0e6] {
            bits_eq(
                calc_correction_factor(err, q),
                ref_p2_calc_correction_factor(err, q),
                &format!("q {q} err {err}"),
            );
        }
        for _ in 0..8 {
            let err = rng.range(0.0, 1.0e4);
            bits_eq(
                calc_correction_factor(err, q),
                ref_p2_calc_correction_factor(err, q),
                &format!("q {q} err {err}"),
            );
        }
    }
}

#[test]
fn qbpm_enumerator_matches_c() {
    for tol in -100..=300i32 {
        assert_eq!(
            qbpm_enumerator(tol),
            ref_p2_qbpm_enumerator(tol),
            "tol {tol}"
        );
    }
}

#[test]
fn decay_rates_match_c() {
    let mut rng = Rng::new(0xDECA_0005);
    let mut saw_low_diff = 0usize;
    let mut saw_high_diff = 0usize;
    for _ in 0..6000 {
        let s = draw_stats(&mut rng, 68);
        let flat = s.to_doubles();
        bits_eq(
            get_sr_decay_rate(&s),
            ref_p2_get_sr_decay_rate(&flat),
            "sr_decay",
        );
        bits_eq(
            get_zero_motion_factor(&s),
            ref_p2_get_zero_motion_factor(&flat),
            "zero_motion_factor",
        );
        bits_eq(
            get_prediction_decay_rate(&s),
            ref_p2_get_prediction_decay_rate(&flat),
            "prediction_decay_rate",
        );
        if s.sr_coded_error - s.coded_error > 0.01 {
            saw_high_diff += 1;
        } else {
            saw_low_diff += 1;
        }
    }
    // Both sides of LOW_SR_DIFF_TRHESH must have fired, or the decay branch is
    // untested.
    assert!(
        saw_low_diff > 100 && saw_high_diff > 100,
        "{saw_low_diff}/{saw_high_diff}"
    );
}

#[test]
fn baseline_err_per_mb_matches_c() {
    // Walk the 640 * 360 = 230400 corner exactly.
    for &(w, h) in &[
        (640i32, 360i32),
        (641, 360),
        (640, 361),
        (480, 480),
        (450, 512),
        (1, 1),
        (3840, 2160),
    ] {
        let fi = FrameInfo {
            frame_width: w,
            frame_height: h,
            mb_rows: (h + 15) / 16,
            mb_cols: (w + 15) / 16,
            num_mbs: 0,
            bit_depth: 8,
        };
        bits_eq(
            baseline_err_per_mb(&fi),
            ref_p2_baseline_err_per_mb(w, h),
            &format!("{w}x{h}"),
        );
    }
}

#[test]
fn calc_frame_boost_matches_c() {
    let mut rng = Rng::new(0xB005_0006);
    let (mut pos, mut neg) = (0usize, 0usize);
    for (fi, cfi) in frame_infos() {
        for _ in 0..150 {
            let mut s = draw_stats(&mut rng, fi.mb_rows);
            // Both arms of the mv_in_out sign test, which take different code.
            s.mv_in_out_count = if rng.below(2) == 0 {
                rng.range(0.0, 1.0)
            } else {
                rng.range(-1.0, 0.0)
            };
            if s.mv_in_out_count > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
            let q = rng.below(256) as i32;
            let max_boost = rng.range(10.0, 200.0);
            for &scale in &[false, true] {
                bits_eq(
                    calc_frame_boost(q, &fi, &s, s.mv_in_out_count, max_boost, scale),
                    ref_p2_calc_frame_boost(
                        q,
                        &cfi,
                        &s.to_doubles(),
                        s.mv_in_out_count,
                        max_boost,
                        scale,
                    ),
                    "frame_boost",
                );
            }
        }
    }
    assert!(
        pos > 100 && neg > 100,
        "one mv_in_out arm never fired: {pos}/{neg}"
    );
}

#[test]
fn calc_kf_frame_boost_matches_c() {
    let mut rng = Rng::new(0x4FB0_0007);
    for (fi, cfi) in frame_infos() {
        // Drive it the way find_next_key_frame does: a running accumulator
        // across a sequence of frames, not a fresh one per call. That is the
        // only way the in/out parameter is actually exercised.
        let mut sr_port = 0.0f64;
        let mut sr_c = 0.0f64;
        for _ in 0..80 {
            let s = draw_stats(&mut rng, fi.mb_rows);
            let q = rng.below(256) as i32;
            let max_boost = rng.range(10.0, 200.0);
            let (boost_p, next_p) = calc_kf_frame_boost(q, &fi, &s, sr_port, max_boost);
            let (boost_c, next_c) =
                ref_p2_calc_kf_frame_boost(q, &cfi, &s.to_doubles(), sr_c, max_boost);
            bits_eq(boost_p, boost_c, "kf_frame_boost");
            bits_eq(next_p, next_c, "sr_accumulator");
            sr_port = next_p;
            sr_c = next_c;
        }
        assert!(
            sr_port >= 0.0,
            "the accumulator's floor at 0 was not applied"
        );
    }
}

#[test]
fn boost_bits_and_factor_match_c() {
    let mut rng = Rng::new(0xB175_0008);
    let mut over_1023 = 0usize;
    for _ in 0..6000 {
        // frame_count is a GF group length; boost is a GF/ARF boost, which the
        // encoder caps around 5000; total_group_bits is the group budget.
        let frame_count = match rng.below(5) {
            0 => 0i32,
            1 => -(rng.below(4) as i32),
            _ => 1 + rng.below(32) as i32,
        };
        let boost = match rng.below(5) {
            0 => 0i32,
            1 => 1 + rng.below(1023) as i32,
            _ => 1024 + rng.below(5000) as i32,
        };
        if boost > 1023 {
            over_1023 += 1;
        }
        let total_group_bits = match rng.below(5) {
            0 => 0i64,
            1 => -i64::from(rng.below(1 << 20)),
            _ => i64::from(rng.below(1 << 28)),
        };
        assert_eq!(
            calculate_boost_bits(frame_count, boost, total_group_bits),
            ref_p2_calculate_boost_bits(frame_count, boost, total_group_bits),
            "boost_bits fc {frame_count} boost {boost} bits {total_group_bits}"
        );
        // calculate_boost_factor divides by (total_group_bits - bits), so the
        // encoder only calls it with bits < total_group_bits.
        if total_group_bits > 0 {
            let bits = rng.below((total_group_bits.min(1 << 27)) as u32 + 1) as i32;
            if i64::from(bits) < total_group_bits {
                assert_eq!(
                    calculate_boost_factor(frame_count, bits, total_group_bits),
                    ref_p2_calculate_boost_factor(frame_count, bits, total_group_bits),
                    "boost_factor fc {frame_count} bits {bits} total {total_group_bits}"
                );
            }
        }
    }
    // The `boost > 1023` overflow-guard rescale must have run.
    assert!(
        over_1023 > 500,
        "only {over_1023} draws exercised the rescale"
    );
}

#[test]
fn get_projected_gfu_boost_matches_c() {
    for baseline_gf_interval in 0..40i32 {
        for gfu_boost in [0i32, 1, 200, 1500, 5000] {
            for frames_to_project in 0..40i32 {
                for num_stats in 0..40i32 {
                    assert_eq!(
                        get_projected_gfu_boost(
                            baseline_gf_interval,
                            gfu_boost,
                            frames_to_project,
                            num_stats
                        ),
                        ref_p2_get_projected_gfu_boost(
                            baseline_gf_interval,
                            gfu_boost,
                            frames_to_project,
                            num_stats
                        ),
                        "gf {baseline_gf_interval} boost {gfu_boost} proj {frames_to_project} stats {num_stats}"
                    );
                }
            }
        }
    }
}

#[test]
fn is_almost_static_matches_c() {
    // Walk both thresholds exactly.
    for &gf in &[0.0f64, 0.9949, 0.995, 0.9951, 0.9989, 0.999, 0.9991, 1.0] {
        for kf in [0i32, 50, 98, 99, 100, 200] {
            for &lap in &[false, true] {
                assert_eq!(
                    is_almost_static(gf, kf, lap),
                    ref_p2_is_almost_static(gf, kf, lap),
                    "gf {gf} kf {kf} lap {lap}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The GF_GROUP_STATS accumulator cluster.
// ---------------------------------------------------------------------------

use aom_encode::pass2_model::{
    GfGroupStats, accumulate_frame_motion_stats, accumulate_next_frame_stats,
    accumulate_this_frame_stats, average_gf_stats, calculate_section_intra_ratio, detect_flash,
    get_second_ref_usage_thresh, read_frame_stats, set_baseline_gf_interval,
};
use aom_sys_ref::{
    ref_p2_accumulate_frame_motion_stats, ref_p2_accumulate_next_frame_stats,
    ref_p2_accumulate_this_frame_stats, ref_p2_average_gf_stats,
    ref_p2_calculate_section_intra_ratio, ref_p2_detect_flash, ref_p2_get_second_ref_usage_thresh,
    ref_p2_gf_group_stats_doubles, ref_p2_init_gf_stats, ref_p2_read_frame_stats_in_range,
};

fn gf_bits_eq(port: &GfGroupStats, c_doubles: &[f64; 17], c_nz: i32, what: &str) {
    let p = port.to_doubles();
    for (i, (a, b)) in p.iter().zip(c_doubles).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: GF_GROUP_STATS double {i}, port {a:?} vs C {b:?}"
        );
    }
    assert_eq!(
        port.non_zero_stdev_count, c_nz,
        "{what}: non_zero_stdev_count"
    );
}

#[test]
fn init_gf_stats_matches_c() {
    assert_eq!(
        GfGroupStats::init().to_doubles().len(),
        ref_p2_gf_group_stats_doubles(),
        "GF_GROUP_STATS double count drifted"
    );
    let (c, nz) = ref_p2_init_gf_stats();
    gf_bits_eq(&GfGroupStats::init(), &c, nz, "init_gf_stats");
    // Three fields start at 1.0, not 0.0 -- a Default-derived struct would be
    // wrong in exactly those three and right everywhere else.
    let init = GfGroupStats::init();
    assert_eq!(init.decay_accumulator, 1.0);
    assert_eq!(init.zero_motion_accumulator, 1.0);
    assert_eq!(init.loop_decay_rate, 1.0);
    assert_eq!(init.last_loop_decay_rate, 1.0);
}

#[test]
fn accumulate_frame_motion_stats_matches_c() {
    let mut rng = Rng::new(0x40C0_0009);
    let (mut low_pct, mut high_pct) = (0usize, 0usize);
    // Drive it as an ACCUMULATION across a group, not one call at a time --
    // three of the four fields it writes are running totals.
    for _ in 0..300 {
        let mut port = GfGroupStats::init();
        let (mut c, mut c_nz) = ref_p2_init_gf_stats();
        // f_w / f_h are the reciprocal frame dimensions the caller passes.
        let (f_w, f_h) = (1.0 / 1920.0, 1.0 / 1080.0);
        for _ in 0..16 {
            let mut s = draw_stats(&mut rng, 68);
            // Straddle the pct > 0.05 guard, which gates the whole mv-ratio
            // half of the function.
            s.pcnt_motion = if rng.below(2) == 0 {
                rng.range(0.0, 0.05)
            } else {
                rng.range(0.05, 1.0)
            };
            if s.pcnt_motion > 0.05 {
                high_pct += 1;
            } else {
                low_pct += 1;
            }
            accumulate_frame_motion_stats(&s, &mut port, f_w, f_h);
            ref_p2_accumulate_frame_motion_stats(&s.to_doubles(), &mut c, &mut c_nz, f_w, f_h);
            gf_bits_eq(&port, &c, c_nz, "accumulate_frame_motion_stats");
        }
    }
    assert!(
        low_pct > 100 && high_pct > 100,
        "one pct arm never fired: {low_pct}/{high_pct}"
    );

    // C's guard is a STRICT `>`, and a random draw never lands on 0.05
    // exactly. Constructed so `>` and `>=` disagree.
    let mut port = GfGroupStats::init();
    let (mut c, mut c_nz) = ref_p2_init_gf_stats();
    let mut s = draw_stats(&mut rng, 68);
    s.pcnt_motion = 0.05;
    // Non-zero mv magnitudes, so the mv-ratio half would move the accumulator
    // if it ran -- otherwise the boundary would be unobservable.
    s.mvr = 4.0;
    s.mvr_abs = 40.0;
    s.mvc = 3.0;
    s.mvc_abs = 30.0;
    accumulate_frame_motion_stats(&s, &mut port, 1.0 / 1920.0, 1.0 / 1080.0);
    ref_p2_accumulate_frame_motion_stats(
        &s.to_doubles(),
        &mut c,
        &mut c_nz,
        1.0 / 1920.0,
        1.0 / 1080.0,
    );
    gf_bits_eq(&port, &c, c_nz, "pct == 0.05 exactly");
    assert_eq!(
        port.mv_ratio_accumulator, 0.0,
        "pct == 0.05 must NOT enter the mv-ratio arm (C uses a strict >)"
    );
}

#[test]
fn accumulate_this_frame_stats_matches_c() {
    let mut rng = Rng::new(0x7415_000A);
    for _ in 0..300 {
        let mut port = GfGroupStats::init();
        let (mut c, mut c_nz) = ref_p2_init_gf_stats();
        for _ in 0..16 {
            let s = draw_stats(&mut rng, 68);
            let err = rng.range(0.0, 1.0e6);
            accumulate_this_frame_stats(&s, err, &mut port);
            ref_p2_accumulate_this_frame_stats(&s.to_doubles(), err, &mut c, &mut c_nz);
            gf_bits_eq(&port, &c, c_nz, "accumulate_this_frame_stats");
        }
        // GROUP_ADAPTIVE_MAXQ is 1 in this build, so the raw-error arm is LIVE.
        assert!(port.gf_group_raw_error >= 0.0);
    }
}

#[test]
fn accumulate_next_frame_stats_matches_c() {
    let mut rng = Rng::new(0x8E37_000B);
    let (mut flashes, mut normals, mut monitored, mut unmonitored) = (0usize, 0, 0usize, 0usize);
    for _ in 0..300 {
        let mut port = GfGroupStats::init();
        let (mut c, mut c_nz) = ref_p2_init_gf_stats();
        // frames_since_key near 0 keeps the (frames_since_key + cur_idx - 1) > 1
        // static-section guard on both sides.
        let frames_since_key = rng.below(4) as i32;
        for cur_idx in 0..16i32 {
            let mut s = draw_stats(&mut rng, 68);
            // raw_error_stdev straddles the 1e-6 non-zero test, which is what
            // drives non_zero_stdev_count and therefore average_gf_stats'
            // second divisor.
            s.raw_error_stdev = if rng.below(3) == 0 {
                0.0
            } else {
                rng.range(0.0, 1.0e3)
            };
            let flash = rng.below(4) == 0;
            if flash {
                flashes += 1;
            } else {
                normals += 1;
            }
            if frames_since_key + cur_idx - 1 > 1 {
                monitored += 1;
            } else {
                unmonitored += 1;
            }
            accumulate_next_frame_stats(
                &s,
                flash,
                frames_since_key,
                cur_idx,
                &mut port,
                1920,
                1080,
            );
            ref_p2_accumulate_next_frame_stats(
                &s.to_doubles(),
                flash,
                frames_since_key,
                cur_idx,
                &mut c,
                &mut c_nz,
                1920,
                1080,
            );
            gf_bits_eq(&port, &c, c_nz, "accumulate_next_frame_stats");
        }
        // And the averaging pass over what was accumulated.
        for &total in &[0i32, 1, 16] {
            let mut p2 = port;
            let mut c2 = c;
            let mut nz2 = c_nz;
            average_gf_stats(total, &mut p2);
            ref_p2_average_gf_stats(total, &mut c2, &mut nz2);
            gf_bits_eq(&p2, &c2, nz2, "average_gf_stats");
        }
    }
    // raw_error_stdev exactly at C's 1e-6 test: a random draw never lands
    // there, so the threshold's own value is otherwise untested.
    for &stdev in &[0.0f64, 1e-9, 1e-6, 1.000_000_1e-6, 1e-3] {
        let mut port = GfGroupStats::init();
        let (mut c, mut c_nz) = ref_p2_init_gf_stats();
        let mut s = draw_stats(&mut rng, 68);
        s.raw_error_stdev = stdev;
        accumulate_next_frame_stats(&s, false, 0, 0, &mut port, 1920, 1080);
        ref_p2_accumulate_next_frame_stats(
            &s.to_doubles(),
            false,
            0,
            0,
            &mut c,
            &mut c_nz,
            1920,
            1080,
        );
        gf_bits_eq(&port, &c, c_nz, &format!("raw_error_stdev {stdev}"));
    }

    assert!(flashes > 100 && normals > 100, "one flash arm never fired");
    assert!(
        monitored > 100 && unmonitored > 100,
        "the static-section guard never fired both ways: {monitored}/{unmonitored}"
    );
}

#[test]
fn calculate_section_intra_ratio_matches_c() {
    let mut rng = Rng::new(0x1417_000C);
    for count in 0..12i32 {
        for _ in 0..40 {
            let section: Vec<FirstpassStats> =
                (0..count).map(|_| draw_stats(&mut rng, 68)).collect();
            let flat: Vec<f64> = section.iter().flat_map(|s| s.to_doubles()).collect();
            // section_length below, at and above the buffer length: C stops at
            // whichever bound comes first.
            for section_length in 0..(count + 3) {
                assert_eq!(
                    calculate_section_intra_ratio(&section, section_length),
                    ref_p2_calculate_section_intra_ratio(&flat, count, section_length),
                    "count {count} section_length {section_length}"
                );
            }
        }
    }
}

#[test]
fn get_second_ref_usage_thresh_matches_c() {
    // Walk the adapt_upto = 32 cap and the (adapt_upto - 1) divisor exactly.
    for n in -4..=64i32 {
        bits_eq(
            get_second_ref_usage_thresh(n),
            ref_p2_get_second_ref_usage_thresh(n),
            &format!("frame_count_so_far {n}"),
        );
    }
}

#[test]
fn read_frame_stats_bounds_match_c() {
    // C's two bounds tests are asymmetric and each applies only for its own
    // sign of `offset`. A symmetric `0 <= i < len` is a DIFFERENT function and
    // this sweep separates them.
    for count in 0..8i32 {
        for cur in 0..=count {
            for offset in -8..=8i32 {
                let want = ref_p2_read_frame_stats_in_range(count, cur, offset);
                let got = read_frame_stats(count as usize, cur as usize, offset).is_some();
                assert_eq!(got, want, "count {count} cur {cur} offset {offset}");
            }
        }
    }
}

#[test]
fn detect_flash_matches_c() {
    let mut rng = Rng::new(0xF1A5_000D);
    let (mut yes, mut no) = (0usize, 0usize);
    for count in 1..8i32 {
        for _ in 0..60 {
            let stats: Vec<FirstpassStats> = (0..count)
                .map(|_| {
                    let mut s = draw_stats(&mut rng, 68);
                    // Straddle both of C's tests: pcnt_second_ref against
                    // pcnt_inter, and against the 0.5 floor.
                    s.pcnt_inter = rng.range(0.0, 1.0);
                    s.pcnt_second_ref = if rng.below(2) == 0 {
                        rng.range(0.0, 1.0)
                    } else {
                        rng.range(0.45, 0.55)
                    };
                    s
                })
                .collect();
            let flat: Vec<f64> = stats.iter().flat_map(|s| s.to_doubles()).collect();
            for cur in 0..count {
                for offset in -3..=3i32 {
                    let want = ref_p2_detect_flash(&flat, count, cur, offset);
                    let got = detect_flash(&stats, cur as usize, offset);
                    assert_eq!(got, want, "count {count} cur {cur} offset {offset}");
                    if want {
                        yes += 1;
                    } else {
                        no += 1;
                    }
                }
            }
        }
    }
    assert!(yes > 50 && no > 50, "one arm never fired: {yes}/{no}");

    // pcnt_second_ref EXACTLY at C's 0.5 floor, which the random draw never
    // hits. C's test is `>=`, so this must detect a flash.
    let mut s = FirstpassStats::default();
    s.pcnt_inter = 0.25;
    s.pcnt_second_ref = 0.5;
    let flat: Vec<f64> = s.to_doubles().to_vec();
    assert!(
        ref_p2_detect_flash(&flat, 1, 0, 0),
        "the construction failed: C did not report a flash at exactly 0.5"
    );
    assert_eq!(
        detect_flash(&[s], 0, 0),
        ref_p2_detect_flash(&flat, 1, 0, 0)
    );
    // And just below it, which must NOT.
    s.pcnt_second_ref = 0.499_999_999;
    let flat2: Vec<f64> = s.to_doubles().to_vec();
    assert!(!ref_p2_detect_flash(&flat2, 1, 0, 0));
    assert_eq!(
        detect_flash(&[s], 0, 0),
        ref_p2_detect_flash(&flat2, 1, 0, 0)
    );
}

#[test]
fn set_baseline_gf_interval_is_the_identity() {
    // C's setter has no clamping and no validation; recording that here means
    // a later reader does not have to go look.
    for n in [-1i32, 0, 1, 16, 32, i32::MAX] {
        assert_eq!(set_baseline_gf_interval(n), n);
    }
}

// ---------------------------------------------------------------------------
// The region-analysis cluster.
// ---------------------------------------------------------------------------

use aom_encode::pass2_model::{
    HALF_FILT_LEN, HALF_WIN, Region, RegionType, analyze_region, cleanup_regions,
    find_regions_index, find_stable_regions, get_gradient, get_region_stats, insert_region,
    mark_flashes, remove_region, remove_short_regions, smooth_filter_noise, smooth_filter_stats,
};
use aom_sys_ref::{
    P2Region, ref_p2_analyze_region, ref_p2_cleanup_regions, ref_p2_find_regions_index,
    ref_p2_find_stable_regions, ref_p2_get_gradient, ref_p2_get_region_stats, ref_p2_half_filt_len,
    ref_p2_half_win, ref_p2_insert_region, ref_p2_mark_flashes, ref_p2_region_types,
    ref_p2_remove_region, ref_p2_remove_short_regions, ref_p2_smooth_filter_noise,
    ref_p2_smooth_filter_stats,
};

/// A run of `n` first-pass records, with `is_flash` drawn separately (it is an
/// `int64_t` in C and does not ride in the 29-double array).
fn draw_run(rng: &mut Rng, n: usize, flash_prob: u32) -> Vec<FirstpassStats> {
    (0..n)
        .map(|_| {
            let mut s = draw_stats(rng, 68);
            s.is_flash = i64::from(flash_prob > 0 && rng.below(flash_prob) == 0);
            // cor_coeff and noise_var are read by analyze_region with a 0.001
            // floor, so both sides of that floor are drawn.
            s.cor_coeff = if rng.below(4) == 0 { 0.0 } else { rng.unit() };
            s.noise_var = if rng.below(4) == 0 {
                0.0
            } else {
                rng.range(0.0, 500.0)
            };
            s
        })
        .collect()
}

fn run_to_flat(run: &[FirstpassStats]) -> (Vec<f64>, Vec<i8>) {
    (
        run.iter().flat_map(|s| s.to_doubles()).collect(),
        run.iter().map(|s| s.is_flash as i8).collect(),
    )
}

fn regions_to_c(r: &[Region]) -> Vec<P2Region> {
    r.iter()
        .map(|x| P2Region {
            start: x.start,
            last: x.last,
            kind: x.kind,
            avgs: [
                x.avg_noise_var,
                x.avg_cor_coeff,
                x.avg_sr_fr_ratio,
                x.avg_intra_err,
                x.avg_coded_err,
            ],
        })
        .collect()
}

fn regions_eq(port: &[Region], c: &[P2Region], n: usize, what: &str) {
    for k in 0..n {
        assert_eq!(port[k].start, c[k].start, "{what}: region {k} start");
        assert_eq!(port[k].last, c[k].last, "{what}: region {k} last");
        assert_eq!(port[k].kind, c[k].kind, "{what}: region {k} type");
        for (i, (a, b)) in [
            port[k].avg_noise_var,
            port[k].avg_cor_coeff,
            port[k].avg_sr_fr_ratio,
            port[k].avg_intra_err,
            port[k].avg_coded_err,
        ]
        .iter()
        .zip(&c[k].avgs)
        .enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "{what}: region {k} avg {i}");
        }
    }
}

#[test]
fn region_constants_match_c() {
    let (stable, high_var, scenecut, blending) = ref_p2_region_types();
    assert_eq!(RegionType::Stable as i32, stable);
    assert_eq!(RegionType::HighVar as i32, high_var);
    assert_eq!(RegionType::Scenecut as i32, scenecut);
    assert_eq!(RegionType::Blending as i32, blending);
    assert_eq!(HALF_WIN, ref_p2_half_win());
    assert_eq!(HALF_FILT_LEN, ref_p2_half_filt_len());
}

#[test]
fn smooth_filter_stats_matches_c() {
    let mut rng = Rng::new(0x5F17_0100);
    for n in 1..14usize {
        for flash_prob in [0u32, 3, 1] {
            for _ in 0..20 {
                let run = draw_run(&mut rng, n, flash_prob);
                let (flat, flash) = run_to_flat(&run);
                for start_idx in 0..n {
                    for last_idx in start_idx..n {
                        // The outputs are ACCUMULATED into, so both sides are
                        // handed the same non-zero starting contents -- a port
                        // that ASSIGNED would agree only on a zeroed array.
                        let seed: Vec<f64> = (0..n).map(|_| rng.range(-1000.0, 1000.0)).collect();
                        let (mut ci, mut cc) = (seed.clone(), seed.clone());
                        ref_p2_smooth_filter_stats(
                            &flat,
                            &flash,
                            start_idx as i32,
                            last_idx as i32,
                            &mut ci,
                            &mut cc,
                        );
                        let (mut pi, mut pc) = (seed.clone(), seed.clone());
                        smooth_filter_stats(&run, start_idx, last_idx, &mut pi, &mut pc);
                        for k in 0..n {
                            bits_eq(
                                pi[k],
                                ci[k],
                                &format!("intra[{k}] n{n} {start_idx}..{last_idx}"),
                            );
                            bits_eq(
                                pc[k],
                                cc[k],
                                &format!("coded[{k}] n{n} {start_idx}..{last_idx}"),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn get_gradient_matches_c() {
    let mut rng = Rng::new(0x64_0101);
    for n in 1..16usize {
        for _ in 0..40 {
            let values: Vec<f64> = (0..n).map(|_| rng.range(-1.0e6, 1.0e6)).collect();
            for start in 0..n {
                for last in start..n {
                    let mut c = vec![0.0f64; n];
                    ref_p2_get_gradient(&values, start as i32, last as i32, &mut c);
                    let mut p = vec![0.0f64; n];
                    get_gradient(&values, start, last, &mut p);
                    for k in 0..n {
                        bits_eq(p[k], c[k], &format!("grad[{k}] n{n} {start}..{last}"));
                    }
                }
            }
        }
    }
}

/// A region array covering `0..n` split into `k` contiguous pieces.
fn draw_regions(rng: &mut Rng, n: usize, k: usize, cap: usize) -> (Vec<Region>, i32) {
    let mut r = vec![Region::default(); cap];
    let k = k.max(1).min(n.max(1));
    let mut bounds: Vec<usize> = (0..k).map(|i| i * n / k).collect();
    bounds.push(n);
    for i in 0..k {
        r[i].start = bounds[i] as i32;
        r[i].last = bounds[i + 1] as i32 - 1;
        r[i].kind = rng.below(4) as i32;
        r[i].avg_noise_var = rng.range(0.0, 100.0);
    }
    (r, k as i32)
}

#[test]
fn analyze_and_get_region_stats_match_c() {
    let mut rng = Rng::new(0x4E41_0102);
    let cap = 64usize;
    for n in 1..14usize {
        for _ in 0..40 {
            let run = draw_run(&mut rng, n, 4);
            let (flat, flash) = run_to_flat(&run);
            let nk = 1 + rng.below(4) as usize;
            let (regions, num) = draw_regions(&mut rng, n, nk, cap);

            // analyze_region on each index separately. avg_noise_var is NOT
            // reset by C, so the seeded value must survive into the result --
            // which is only visible because draw_regions seeds it non-zero.
            for k in 0..num as usize {
                let mut p = regions.clone();
                let mut c = regions_to_c(&regions);
                ref_p2_analyze_region(&flat, &flash, k as i32, &mut c);
                analyze_region(&run, k, &mut p);
                regions_eq(&p, &c, num as usize, &format!("analyze_region k{k} n{n}"));
            }

            // And the whole-array pass.
            let mut p = regions.clone();
            let mut c = regions_to_c(&regions);
            ref_p2_get_region_stats(&flat, &flash, num, &mut c);
            get_region_stats(&run, &mut p, num as usize);
            regions_eq(&p, &c, num as usize, &format!("get_region_stats n{n}"));
        }
    }
}

#[test]
fn find_stable_regions_matches_c() {
    let mut rng = Rng::new(0x5747_0103);
    let cap = 128usize;
    let (mut stable_seen, mut high_seen) = (0usize, 0usize);
    for n in 1..16usize {
        for flash_prob in [0u32, 3] {
            for _ in 0..40 {
                let mut run = draw_run(&mut rng, n, flash_prob);
                // A stable region needs coded_error well under intra_error and
                // low variance; a uniform draw is essentially never stable, so
                // half the runs are made flat.
                if rng.below(2) == 0 {
                    let base = rng.range(1000.0, 1.0e5);
                    for s in run.iter_mut() {
                        s.intra_error = base * rng.range(0.99, 1.01);
                        s.coded_error = base * rng.range(0.01, 0.03);
                    }
                }
                let (flat, flash) = run_to_flat(&run);
                let grad: Vec<f64> = (0..n).map(|_| rng.range(-100.0, 100.0)).collect();
                for this_start in 0..n {
                    for this_last in this_start..n {
                        let mut c = vec![P2Region::default(); cap];
                        let cn = ref_p2_find_stable_regions(
                            &flat,
                            &flash,
                            &grad,
                            this_start as i32,
                            this_last as i32,
                            &mut c,
                        );
                        let mut p = vec![Region::default(); cap];
                        let pn = find_stable_regions(&run, &grad, this_start, this_last, &mut p);
                        assert_eq!(pn as i32, cn, "count, n{n} {this_start}..{this_last}");
                        regions_eq(
                            &p,
                            &c,
                            pn,
                            &format!("find_stable n{n} {this_start}..{this_last}"),
                        );
                        for k in 0..pn {
                            if p[k].kind == RegionType::Stable as i32 {
                                stable_seen += 1;
                            } else {
                                high_seen += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        stable_seen > 100 && high_seen > 100,
        "one region type never appeared: {stable_seen} stable / {high_seen} high-var"
    );
}

#[test]
fn remove_region_matches_c() {
    let mut rng = Rng::new(0x8E4D_0104);
    let cap = 64usize;
    for num in 1..10usize {
        for _ in 0..200 {
            let (regions, _) = draw_regions(&mut rng, num * 3, num, cap);
            for merge in 0..3i32 {
                for k in 0..num {
                    let mut p = regions.clone();
                    let mut pn = num as i32;
                    let mut pk = k as i32;
                    let mut c = regions_to_c(&regions);
                    let mut cn = num as i32;
                    let mut ck = k as i32;
                    ref_p2_remove_region(merge, &mut c, &mut cn, &mut ck);
                    remove_region(merge, &mut p, &mut pn, &mut pk);
                    assert_eq!(pn, cn, "num_regions, merge {merge} k {k} num {num}");
                    assert_eq!(pk, ck, "next_region, merge {merge} k {k} num {num}");
                    regions_eq(
                        &p,
                        &c,
                        pn.max(0) as usize,
                        &format!("remove_region {merge} k{k}"),
                    );
                }
            }
        }
    }
}

#[test]
fn insert_region_matches_c() {
    let mut rng = Rng::new(0x1E5E_0105);
    let cap = 64usize;
    let mut adds = [0usize; 3];
    for num in 1..8usize {
        for _ in 0..300 {
            let n = num * 5;
            let (regions, _) = draw_regions(&mut rng, n, num, cap);
            let k = rng.below(num as u32) as usize;
            let lo = regions[k].start;
            let hi = regions[k].last;
            if hi < lo {
                continue;
            }
            // start and last must both be INSIDE region k, per C's contract.
            let start = lo + rng.below((hi - lo + 1) as u32) as i32;
            let last = start + rng.below((hi - start + 1) as u32) as i32;
            let kind = rng.below(4) as i32;

            let mut p = regions.clone();
            let mut pn = num as i32;
            let mut pk = k as i32;
            let mut c = regions_to_c(&regions);
            let mut cn = num as i32;
            let mut ck = k as i32;
            ref_p2_insert_region(start, last, kind, &mut c, &mut cn, &mut ck);
            insert_region(start, last, kind, &mut p, &mut pn, &mut pk);
            assert_eq!(pn, cn, "num_regions, insert {start}..{last} at k{k}");
            assert_eq!(pk, ck, "cur_region_idx, insert {start}..{last} at k{k}");
            regions_eq(
                &p,
                &c,
                pn as usize,
                &format!("insert_region {start}..{last} k{k}"),
            );
            adds[(pn - num as i32) as usize] += 1;
        }
    }
    // The 0, 1 and 2 growth cases are three different code paths.
    for (i, &n) in adds.iter().enumerate() {
        assert!(n > 20, "insert_region grew by {i} only {n} times");
    }
}

#[test]
fn cleanup_and_remove_short_regions_match_c() {
    let mut rng = Rng::new(0xC1EA_0106);
    let cap = 64usize;
    for num in 1..10usize {
        for _ in 0..300 {
            let (mut regions, _) = draw_regions(&mut rng, num * 3, num, cap);
            // Deliberately create the two conditions cleanup_regions looks for:
            // an adjacent same-type pair, and an EMPTY region (last < start).
            if num > 1 && rng.below(2) == 0 {
                let k = 1 + rng.below((num - 1) as u32) as usize;
                regions[k].kind = regions[k - 1].kind;
            }
            if rng.below(3) == 0 {
                let k = rng.below(num as u32) as usize;
                regions[k].last = regions[k].start - 1;
            }

            let mut p = regions.clone();
            let mut pn = num as i32;
            let mut c = regions_to_c(&regions);
            let mut cn = num as i32;
            ref_p2_cleanup_regions(&mut c, &mut cn);
            cleanup_regions(&mut p, &mut pn);
            assert_eq!(pn, cn, "cleanup num_regions, num {num}");
            regions_eq(&p, &c, pn.max(0) as usize, "cleanup_regions");

            for kind in 0..4i32 {
                for length in 0..5i32 {
                    let mut p = regions.clone();
                    let mut pn = num as i32;
                    let mut c = regions_to_c(&regions);
                    let mut cn = num as i32;
                    ref_p2_remove_short_regions(&mut c, &mut cn, kind, length);
                    remove_short_regions(&mut p, &mut pn, kind, length);
                    assert_eq!(pn, cn, "remove_short num_regions, kind {kind} len {length}");
                    regions_eq(
                        &p,
                        &c,
                        pn.max(0) as usize,
                        &format!("remove_short_regions kind {kind} len {length}"),
                    );
                }
            }
        }
    }
}

#[test]
fn find_regions_index_matches_c() {
    let mut rng = Rng::new(0xF14D_0107);
    let cap = 32usize;
    for num in 1..8usize {
        for _ in 0..100 {
            let (regions, n) = draw_regions(&mut rng, num * 4, num, cap);
            let c = regions_to_c(&regions);
            for frame_idx in -2..(num as i32 * 4 + 2) {
                let want = ref_p2_find_regions_index(&c, n, frame_idx);
                let got = find_regions_index(&regions, n as usize, frame_idx);
                assert_eq!(
                    got.map_or(-1, |k| k as i32),
                    want,
                    "num {num} frame {frame_idx}"
                );
            }
        }
    }
}

#[test]
fn mark_flashes_matches_c() {
    let mut rng = Rng::new(0x41A5_0108);
    let (mut flashes, mut clears) = (0usize, 0usize);
    for n in 0..14usize {
        for _ in 0..80 {
            let mut run = draw_run(&mut rng, n, 2);
            // Straddle both of the tests mark_flashes makes.
            for s in run.iter_mut() {
                s.pcnt_inter = rng.unit();
                s.pcnt_second_ref = if rng.below(2) == 0 {
                    rng.unit()
                } else {
                    rng.range(0.45, 0.55)
                };
            }
            let (flat, flash) = run_to_flat(&run);
            let want = ref_p2_mark_flashes(&flat, &flash);
            mark_flashes(&mut run);
            let got: Vec<i8> = run.iter().map(|s| s.is_flash as i8).collect();
            assert_eq!(got, want, "n {n}");
            flashes += got.iter().filter(|&&f| f != 0).count();
            clears += got.iter().filter(|&&f| f == 0).count();
        }
    }
    assert!(
        flashes > 100 && clears > 100,
        "one arm never fired: {flashes}/{clears}"
    );
}

#[test]
fn smooth_filter_noise_matches_c() {
    let mut rng = Rng::new(0x5401_0109);
    for n in 1..14usize {
        for flash_prob in [0u32, 3, 1] {
            for _ in 0..60 {
                let mut run = draw_run(&mut rng, n, flash_prob);
                let (flat, flash) = run_to_flat(&run);
                let want = ref_p2_smooth_filter_noise(&flat, &flash);
                smooth_filter_noise(&mut run);
                for (k, s) in run.iter().enumerate() {
                    bits_eq(s.noise_var, want[k], &format!("noise[{k}] n{n}"));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The scenecut / noise-model helpers.
// ---------------------------------------------------------------------------

use aom_encode::pass2_model::{
    ERROR_SPIKE, VERY_LOW_II, estimate_coeff, estimate_noise, find_qindex_by_rate_with_correction,
    slide_transition,
};
use aom_sys_ref::{
    ref_p2_estimate_coeff, ref_p2_estimate_noise, ref_p2_find_qindex_by_rate_with_correction,
    ref_p2_scenecut_constants, ref_p2_slide_transition,
};

#[test]
fn scenecut_constants_match_c() {
    let (ii, spike) = ref_p2_scenecut_constants();
    bits_eq(VERY_LOW_II, ii, "VERY_LOW_II");
    bits_eq(ERROR_SPIKE, spike, "ERROR_SPIKE");
}

#[test]
fn find_qindex_by_rate_with_correction_matches_c() {
    let mut rng = Rng::new(0xF19D_0200);
    let mut hit_ends = (0usize, 0usize);
    for _ in 0..4000 {
        let bit_depth = [8i32, 10, 12][rng.below(3) as usize];
        // The bounds are real qindex limits, and best <= worst is C's assert.
        let best = rng.below(200) as i32;
        let worst = best + rng.below(256 - best as u32) as i32;
        // desired_bits_per_mb spans the range the two-pass allocator produces.
        let desired = match rng.below(4) {
            0 => 0u64,
            1 => u64::from(rng.below(1 << 8)),
            2 => u64::from(rng.below(1 << 16)),
            _ => u64::from(rng.below(1 << 24)),
        };
        let error_per_mb = rng.range(0.0, 5000.0);
        let group_weight_factor = rng.range(0.1, 4.0);
        let rate_err_tol = rng.below(120) as i32;
        let want = ref_p2_find_qindex_by_rate_with_correction(
            desired,
            bit_depth,
            error_per_mb,
            group_weight_factor,
            rate_err_tol,
            best,
            worst,
        );
        let got = find_qindex_by_rate_with_correction(
            desired,
            bit_depth as u8,
            error_per_mb,
            group_weight_factor,
            rate_err_tol,
            best,
            worst,
        );
        assert_eq!(
            got, want,
            "bd{bit_depth} desired {desired} err {error_per_mb} gwf {group_weight_factor} tol {rate_err_tol} [{best},{worst}]"
        );
        if want == best {
            hit_ends.0 += 1;
        }
        if want == worst {
            hit_ends.1 += 1;
        }
    }
    // Both ends of the search must be reachable, or the binary search's two
    // arms are not both exercised at their boundaries.
    assert!(
        hit_ends.0 > 50 && hit_ends.1 > 50,
        "search never converged on an end: {hit_ends:?}"
    );
}

#[test]
fn slide_transition_matches_c() {
    let mut rng = Rng::new(0x571D_0201);
    let (mut yes, mut no) = (0usize, 0usize);
    for _ in 0..6000 {
        let mut this = draw_stats(&mut rng, 68);
        let mut last = draw_stats(&mut rng, 68);
        let mut next = draw_stats(&mut rng, 68);
        // A uniform draw essentially never satisfies the three-way
        // conjunction, so half the draws are built to straddle each factor.
        if rng.below(2) == 0 {
            this.coded_error = rng.range(1000.0, 1.0e5);
            this.intra_error = this.coded_error * rng.range(1.0, 2.0);
            last.coded_error = this.coded_error / rng.range(2.0, 8.0);
            next.coded_error = this.coded_error / rng.range(2.0, 8.0);
        }
        let want =
            ref_p2_slide_transition(&this.to_doubles(), &last.to_doubles(), &next.to_doubles());
        let got = slide_transition(&this, &last, &next);
        assert_eq!(got, want, "this {this:?} last {last:?} next {next:?}");
        if want {
            yes += 1;
        } else {
            no += 1;
        }
    }
    assert!(yes > 100 && no > 100, "one arm never fired: {yes}/{no}");
}

#[test]
fn estimate_noise_matches_c() {
    let mut rng = Rng::new(0x0157_0202);
    for n in 0..16usize {
        for flash_prob in [0u32, 5, 2] {
            for _ in 0..60 {
                let mut run = draw_run(&mut rng, n, flash_prob);
                // The model needs intra_error > coded_error and
                // intra_error > sr_coded_error for its three products to be
                // positive; a uniform draw makes them positive about an eighth
                // of the time, so half the runs are built to satisfy them and
                // the rest are left to exercise the `continue`.
                if rng.below(2) == 0 {
                    for s in run.iter_mut() {
                        s.intra_error = rng.range(1000.0, 1.0e5);
                        s.coded_error = s.intra_error * rng.range(0.1, 0.9);
                        s.sr_coded_error = s.intra_error * rng.range(0.1, 0.95);
                    }
                }
                let (flat, flash) = run_to_flat(&run);
                let want = ref_p2_estimate_noise(&flat, &flash);
                estimate_noise(&mut run);
                for (k, s) in run.iter().enumerate() {
                    bits_eq(
                        s.noise_var,
                        want[k],
                        &format!("noise[{k}] n{n} fp{flash_prob}"),
                    );
                }
            }
        }
    }
}

#[test]
fn estimate_coeff_matches_c() {
    let mut rng = Rng::new(0xC0EF_0203);
    let (mut clamped_low, mut clamped_high, mut interior) = (0usize, 0usize, 0usize);
    for n in 0..16usize {
        for _ in 0..120 {
            let mut run = draw_run(&mut rng, n, 4);
            // noise_var straddles intra_error, which is what drives the three
            // 0.001 floors -- above it they all bind and the coefficient
            // clamps.
            for s in run.iter_mut() {
                s.intra_error = rng.range(1000.0, 1.0e5);
                s.coded_error = s.intra_error * rng.range(0.05, 1.2);
                s.noise_var = match rng.below(3) {
                    0 => rng.range(0.0, 10.0),
                    1 => s.intra_error * rng.range(0.5, 0.99),
                    _ => s.intra_error * rng.range(1.0, 2.0),
                };
            }
            let (flat, flash) = run_to_flat(&run);
            let noise: Vec<f64> = run.iter().map(|s| s.noise_var).collect();
            let want = ref_p2_estimate_coeff(&flat, &flash, &noise);
            estimate_coeff(&mut run);
            for (k, s) in run.iter().enumerate() {
                bits_eq(s.cor_coeff, want[k], &format!("cor_coeff[{k}] n{n}"));
                if k > 0 {
                    if s.cor_coeff == 0.0 {
                        clamped_low += 1;
                    } else if s.cor_coeff == 1.0 {
                        clamped_high += 1;
                    } else {
                        interior += 1;
                    }
                }
            }
        }
    }
    assert!(
        clamped_high > 50 && interior > 50,
        "the clamp and the interior were not both reached: {clamped_low}/{clamped_high}/{interior}"
    );
}
