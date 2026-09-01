//! Differential harness for `av1/encoder/firstpass.c` vs the REAL C libaom
//! v3.14.1.
//!
//! **Two tiers, kept apart on purpose.** The seven `av1_firstpass_info_*`
//! entry points are **tier 1**: `shim/fp_info_shim.c` does not include
//! firstpass.c, so its calls bind to the archive's own symbols. Everything
//! else goes through `shim/fp_shim.c`, which compiles firstpass.c verbatim
//! with its 16 exported symbols renamed, to reach the 29 file-statics —
//! **tier 1c**, with `fp_shim_tu_matches_archive` closing the
//! second-compilation gap.
//!
//! Before any of that: `firstpass_stats_layout_matches_c` and
//! `frame_stats_layout_matches_c` prove the Rust `#[repr(C)]` mirrors agree
//! with C's structs field by field. Every other test here passes structs
//! across the boundary by pointer, so a layout disagreement would make the
//! whole file compare garbage against garbage and possibly pass.

use aom_encode::firstpass::{
    FIRSTPASS_INFO_STATIC_BUF_SIZE, FirstpassInfo, FirstpassInfoError, FirstpassStats, FrameStats,
    INVALID_ROW, calc_wavelet_energy, find_fp_qindex, get_num_mbs, get_search_range, get_unit_cols,
    get_unit_cols_in_tile, get_unit_rows, get_unit_rows_in_tile, raw_motion_error_stdev,
};
use aom_sys_ref::{
    RefFirstpassInfo, RefFirstpassStats, RefFrameStats, ref_exponential_entropy,
    ref_fp_accumulate_frame_stats, ref_fp_accumulate_mv_stats, ref_fp_accumulate_stats,
    ref_fp_calc_wavelet_energy, ref_fp_find_fp_qindex, ref_fp_frame_stats_layout_probe,
    ref_fp_frame_stats_size, ref_fp_get_num_mbs, ref_fp_get_search_range, ref_fp_get_unit_cols,
    ref_fp_get_unit_cols_in_tile, ref_fp_get_unit_rows, ref_fp_get_unit_rows_in_tile,
    ref_fp_normalize_firstpass_stats, ref_fp_raw_motion_error_stdev, ref_fp_stats_layout_probe,
    ref_fp_stats_size, ref_fp_twopass_zero_stats, ref_fpi_codec_ok, ref_fpi_static_buf_size,
};

/// The same LCG the other encoder differentials use.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn next_f64(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4096.0 - 500_000.0
    }
}

// ---------------------------------------------------------------------------
// 0. Layout. Nothing else in this file means anything without these.
// ---------------------------------------------------------------------------

#[test]
fn firstpass_stats_layout_matches_c() {
    assert_eq!(
        size_of::<FirstpassStats>(),
        ref_fp_stats_size(),
        "FIRSTPASS_STATS size"
    );
    assert_eq!(size_of::<RefFirstpassStats>(), ref_fp_stats_size());
    // The shim writes 1.0, 2.0, ... into C's fields in declaration order.
    let p = ref_fp_stats_layout_probe();
    let fields: [(f64, &str); 30] = [
        (p.frame, "frame"),
        (p.weight, "weight"),
        (p.intra_error, "intra_error"),
        (p.frame_avg_wavelet_energy, "frame_avg_wavelet_energy"),
        (p.coded_error, "coded_error"),
        (p.sr_coded_error, "sr_coded_error"),
        (p.lt_coded_error, "lt_coded_error"),
        (p.pcnt_inter, "pcnt_inter"),
        (p.pcnt_motion, "pcnt_motion"),
        (p.pcnt_second_ref, "pcnt_second_ref"),
        (p.pcnt_neutral, "pcnt_neutral"),
        (p.intra_skip_pct, "intra_skip_pct"),
        (p.inactive_zone_rows, "inactive_zone_rows"),
        (p.inactive_zone_cols, "inactive_zone_cols"),
        (p.mvr, "MVr"),
        (p.mvr_abs, "mvr_abs"),
        (p.mvc, "MVc"),
        (p.mvc_abs, "mvc_abs"),
        (p.mvrv, "MVrv"),
        (p.mvcv, "MVcv"),
        (p.mv_in_out_count, "mv_in_out_count"),
        (p.new_mv_count, "new_mv_count"),
        (p.duration, "duration"),
        (p.count, "count"),
        (p.raw_error_stdev, "raw_error_stdev"),
        (p.is_flash as f64, "is_flash"),
        (p.noise_var, "noise_var"),
        (p.cor_coeff, "cor_coeff"),
        (p.log_intra_error, "log_intra_error"),
        (p.log_coded_error, "log_coded_error"),
    ];
    for (i, (got, name)) in fields.iter().enumerate() {
        assert_eq!(
            *got,
            (i + 1) as f64,
            "FIRSTPASS_STATS field {name} is at the wrong offset"
        );
    }
}

#[test]
fn frame_stats_layout_matches_c() {
    assert_eq!(
        size_of::<FrameStats>(),
        ref_fp_frame_stats_size(),
        "FRAME_STATS size"
    );
    assert_eq!(size_of::<RefFrameStats>(), ref_fp_frame_stats_size());
    let p = ref_fp_frame_stats_layout_probe();
    let fields: [(i64, &str); 21] = [
        (p.intra_error, "intra_error"),
        (p.frame_avg_wavelet_energy, "frame_avg_wavelet_energy"),
        (p.coded_error, "coded_error"),
        (p.sr_coded_error, "sr_coded_error"),
        (p.lt_coded_error, "lt_coded_error"),
        (i64::from(p.mv_count), "mv_count"),
        (i64::from(p.inter_count), "inter_count"),
        (i64::from(p.second_ref_count), "second_ref_count"),
        (p.neutral_count as i64, "neutral_count"),
        (i64::from(p.intra_skip_count), "intra_skip_count"),
        (i64::from(p.image_data_start_row), "image_data_start_row"),
        (i64::from(p.new_mv_count), "new_mv_count"),
        (i64::from(p.sum_in_vectors), "sum_in_vectors"),
        (i64::from(p.sum_mvr), "sum_mvr"),
        (i64::from(p.sum_mvc), "sum_mvc"),
        (i64::from(p.sum_mvr_abs), "sum_mvr_abs"),
        (i64::from(p.sum_mvc_abs), "sum_mvc_abs"),
        (p.sum_mvrs, "sum_mvrs"),
        (p.sum_mvcs, "sum_mvcs"),
        (p.intra_factor as i64, "intra_factor"),
        (p.brightness_factor as i64, "brightness_factor"),
    ];
    for (i, (got, name)) in fields.iter().enumerate() {
        assert_eq!(
            *got,
            (i + 1) as i64,
            "FRAME_STATS field {name} is at the wrong offset"
        );
    }
}

/// The tier-1c premise for `fp_shim.c`: its second compilation of firstpass.c
/// agrees with the archive.
///
/// Two of firstpass.c's exported functions are re-exported from that TU and
/// compared here against the same functions reached through
/// `fp_info_shim.c`'s tier-1 path — `av1_firstpass_info_init` with an
/// external buffer runs `av1_accumulate_stats` over every entry, so the
/// archive's accumulator is observable through the tier-1 shim's
/// `total_stats`, and that is what the TU copy is checked against.
#[test]
fn fp_shim_tu_matches_archive() {
    let mut rng = Lcg(0x5eed_0100);
    for _ in 0..500 {
        let n = 1 + (rng.next_u32() % 8) as usize;
        let entries: Vec<RefFirstpassStats> = (0..n).map(|_| random_ref_stats(&mut rng)).collect();

        // Tier 1: the archive accumulates every entry into total_stats.
        let info = RefFirstpassInfo::new_external(&entries);
        let archive_total = info.state().total_stats;

        // Tier 1c: this TU's copy of av1_accumulate_stats, same sequence.
        let mut tu_total = RefFirstpassStats::default();
        for e in &entries {
            tu_total = ref_fp_accumulate_stats(tu_total, e);
        }
        assert_eq!(
            bits_of(&tu_total),
            bits_of(&archive_total),
            "fp_shim.c's av1_accumulate_stats disagrees with the archive's"
        );
    }
    // A cheap independent check that ref_init and the oracle are live at all.
    assert!(ref_exponential_entropy(4.0, 2.0).is_finite());
}

fn bits_of(s: &RefFirstpassStats) -> Vec<u64> {
    vec![
        s.frame.to_bits(),
        s.weight.to_bits(),
        s.intra_error.to_bits(),
        s.frame_avg_wavelet_energy.to_bits(),
        s.coded_error.to_bits(),
        s.sr_coded_error.to_bits(),
        s.lt_coded_error.to_bits(),
        s.pcnt_inter.to_bits(),
        s.pcnt_motion.to_bits(),
        s.pcnt_second_ref.to_bits(),
        s.pcnt_neutral.to_bits(),
        s.intra_skip_pct.to_bits(),
        s.inactive_zone_rows.to_bits(),
        s.inactive_zone_cols.to_bits(),
        s.mvr.to_bits(),
        s.mvr_abs.to_bits(),
        s.mvc.to_bits(),
        s.mvc_abs.to_bits(),
        s.mvrv.to_bits(),
        s.mvcv.to_bits(),
        s.mv_in_out_count.to_bits(),
        s.new_mv_count.to_bits(),
        s.duration.to_bits(),
        s.count.to_bits(),
        s.raw_error_stdev.to_bits(),
        s.is_flash as u64,
        s.noise_var.to_bits(),
        s.cor_coeff.to_bits(),
        s.log_intra_error.to_bits(),
        s.log_coded_error.to_bits(),
    ]
}

fn to_ref(s: &FirstpassStats) -> RefFirstpassStats {
    RefFirstpassStats {
        frame: s.frame,
        weight: s.weight,
        intra_error: s.intra_error,
        frame_avg_wavelet_energy: s.frame_avg_wavelet_energy,
        coded_error: s.coded_error,
        sr_coded_error: s.sr_coded_error,
        lt_coded_error: s.lt_coded_error,
        pcnt_inter: s.pcnt_inter,
        pcnt_motion: s.pcnt_motion,
        pcnt_second_ref: s.pcnt_second_ref,
        pcnt_neutral: s.pcnt_neutral,
        intra_skip_pct: s.intra_skip_pct,
        inactive_zone_rows: s.inactive_zone_rows,
        inactive_zone_cols: s.inactive_zone_cols,
        mvr: s.mvr,
        mvr_abs: s.mvr_abs,
        mvc: s.mvc,
        mvc_abs: s.mvc_abs,
        mvrv: s.mvrv,
        mvcv: s.mvcv,
        mv_in_out_count: s.mv_in_out_count,
        new_mv_count: s.new_mv_count,
        duration: s.duration,
        count: s.count,
        raw_error_stdev: s.raw_error_stdev,
        is_flash: s.is_flash,
        noise_var: s.noise_var,
        cor_coeff: s.cor_coeff,
        log_intra_error: s.log_intra_error,
        log_coded_error: s.log_coded_error,
    }
}

fn from_ref(s: &RefFirstpassStats) -> FirstpassStats {
    FirstpassStats {
        frame: s.frame,
        weight: s.weight,
        intra_error: s.intra_error,
        frame_avg_wavelet_energy: s.frame_avg_wavelet_energy,
        coded_error: s.coded_error,
        sr_coded_error: s.sr_coded_error,
        lt_coded_error: s.lt_coded_error,
        pcnt_inter: s.pcnt_inter,
        pcnt_motion: s.pcnt_motion,
        pcnt_second_ref: s.pcnt_second_ref,
        pcnt_neutral: s.pcnt_neutral,
        intra_skip_pct: s.intra_skip_pct,
        inactive_zone_rows: s.inactive_zone_rows,
        inactive_zone_cols: s.inactive_zone_cols,
        mvr: s.mvr,
        mvr_abs: s.mvr_abs,
        mvc: s.mvc,
        mvc_abs: s.mvc_abs,
        mvrv: s.mvrv,
        mvcv: s.mvcv,
        mv_in_out_count: s.mv_in_out_count,
        new_mv_count: s.new_mv_count,
        duration: s.duration,
        count: s.count,
        raw_error_stdev: s.raw_error_stdev,
        is_flash: s.is_flash,
        noise_var: s.noise_var,
        cor_coeff: s.cor_coeff,
        log_intra_error: s.log_intra_error,
        log_coded_error: s.log_coded_error,
    }
}

/// Errors and energies are non-negative sums of squared pixel differences, so
/// `log1p` of them is defined; the percentage fields are in `[0, 1]`; the MV
/// sums are signed. Values are kept large enough that the `log1p` terms are
/// not all zero, which is what makes the accumulate test non-vacuous.
fn random_ref_stats(rng: &mut Lcg) -> RefFirstpassStats {
    let pos = |r: &mut Lcg| f64::from(r.next_u32() % 100_000_000) / 64.0;
    let pct = |r: &mut Lcg| f64::from(r.next_u32() % 10001) / 10000.0;
    RefFirstpassStats {
        frame: f64::from(rng.next_u32() % 100_000),
        weight: pct(rng),
        intra_error: pos(rng),
        frame_avg_wavelet_energy: pos(rng),
        coded_error: pos(rng),
        sr_coded_error: pos(rng),
        lt_coded_error: pos(rng),
        pcnt_inter: pct(rng),
        pcnt_motion: pct(rng),
        pcnt_second_ref: pct(rng),
        pcnt_neutral: pct(rng),
        intra_skip_pct: pct(rng),
        inactive_zone_rows: pct(rng),
        inactive_zone_cols: pct(rng),
        mvr: rng.next_f64(),
        mvr_abs: pos(rng),
        mvc: rng.next_f64(),
        mvc_abs: pos(rng),
        mvrv: pos(rng),
        mvcv: pos(rng),
        mv_in_out_count: rng.next_f64(),
        new_mv_count: pos(rng),
        duration: pos(rng),
        count: f64::from(rng.next_u32() % 100),
        raw_error_stdev: pos(rng),
        is_flash: i64::from(rng.next_u32() % 2),
        noise_var: pos(rng),
        cor_coeff: pct(rng),
        log_intra_error: pos(rng),
        log_coded_error: pos(rng),
    }
}

// ---------------------------------------------------------------------------
// 1. The FIRSTPASS_STATS trio.
// ---------------------------------------------------------------------------

#[test]
fn twopass_zero_stats_matches_c() {
    let mut rng = Lcg(0x5eed_0101);
    let mut saw_preserved = false;
    for _ in 0..2_000 {
        let prior = random_ref_stats(&mut rng);
        let mut got = from_ref(&prior);
        got.zero();
        let want = ref_fp_twopass_zero_stats(prior);
        assert_eq!(bits_of(&to_ref(&got)), bits_of(&want));
        // raw_error_stdev is NOT written by C. Prove that is what is
        // happening, not that both happened to be zero.
        if prior.raw_error_stdev != 0.0 {
            assert_eq!(want.raw_error_stdev, prior.raw_error_stdev);
            saw_preserved = true;
        }
    }
    assert!(
        saw_preserved,
        "never started from a non-zero raw_error_stdev — the \"C does not \
         write this field\" claim is untested"
    );
    // duration and cor_coeff are seeded to 1.0, not 0.0.
    let mut s = FirstpassStats::default();
    s.zero();
    assert_eq!(s.duration, 1.0);
    assert_eq!(s.cor_coeff, 1.0);
    assert_eq!(s.count, 0.0);
}

#[test]
fn accumulate_stats_matches_c() {
    let mut rng = Lcg(0x5eed_0102);
    for _ in 0..3_000 {
        let section = random_ref_stats(&mut rng);
        let frame = random_ref_stats(&mut rng);
        let mut got = from_ref(&section);
        got.accumulate(&from_ref(&frame));
        let want = ref_fp_accumulate_stats(section, &frame);
        assert_eq!(bits_of(&to_ref(&got)), bits_of(&want));
    }
    // The four fields C does NOT accumulate keep the section's value.
    let mut rng = Lcg(0x5eed_0103);
    let section = random_ref_stats(&mut rng);
    let frame = random_ref_stats(&mut rng);
    let want = ref_fp_accumulate_stats(section, &frame);
    assert_eq!(want.raw_error_stdev, section.raw_error_stdev);
    assert_eq!(want.is_flash, section.is_flash);
    assert_eq!(want.noise_var, section.noise_var);
    assert_eq!(want.cor_coeff, section.cor_coeff);
    // ...and they are genuinely different in the frame, so this is a real
    // check rather than two equal values.
    assert_ne!(frame.raw_error_stdev, section.raw_error_stdev);
    assert_ne!(frame.noise_var, section.noise_var);
}

#[test]
fn normalize_firstpass_stats_matches_c() {
    let mut rng = Lcg(0x5eed_0104);
    for _ in 0..3_000 {
        let fps = random_ref_stats(&mut rng);
        // Real callers pass the 16x16 macroblock count and the frame
        // dimensions in pixels, all strictly positive.
        let num_mbs = f64::from(1 + rng.next_u32() % 100_000);
        let f_w = f64::from(16 + rng.next_u32() % 8000);
        let f_h = f64::from(16 + rng.next_u32() % 8000);
        let mut got = from_ref(&fps);
        got.normalize(num_mbs, f_w, f_h);
        let want = ref_fp_normalize_firstpass_stats(fps, num_mbs, f_w, f_h);
        assert_eq!(
            bits_of(&to_ref(&got)),
            bits_of(&want),
            "num_mbs={num_mbs} f_w={f_w} f_h={f_h}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Geometry.
// ---------------------------------------------------------------------------

/// The square BLOCK_SIZEs, plus the two the first pass actually uses
/// (BLOCK_8X8 = 3 and BLOCK_16X16 = 6).
const SQUARE_BSIZES: [i32; 6] = [0, 3, 6, 9, 12, 15];

#[test]
fn get_unit_rows_cols_match_c() {
    let mut saw_shift_down = false;
    let mut saw_shift_up = false;
    for &bsize in &SQUARE_BSIZES {
        for n in [0i32, 1, 3, 7, 15, 64, 1000] {
            let r = get_unit_rows(bsize, n);
            let c = get_unit_cols(bsize, n);
            assert_eq!(
                r,
                ref_fp_get_unit_rows(bsize, n),
                "rows bsize={bsize} n={n}"
            );
            assert_eq!(
                c,
                ref_fp_get_unit_cols(bsize, n),
                "cols bsize={bsize} n={n}"
            );
            if n > 0 {
                if r < n {
                    saw_shift_down = true;
                } else if r > n {
                    saw_shift_up = true;
                }
            }
        }
    }
    assert!(
        saw_shift_down && saw_shift_up,
        "one shift arm was never taken"
    );
    // The rectangular sizes take the two axes independently, so sweep them
    // too — get_unit_rows/cols do not assert squareness (only get_num_mbs
    // does).
    for bsize in 0..16i32 {
        for n in [1i32, 5, 100] {
            assert_eq!(get_unit_rows(bsize, n), ref_fp_get_unit_rows(bsize, n));
            assert_eq!(get_unit_cols(bsize, n), ref_fp_get_unit_cols(bsize, n));
        }
    }
}

#[test]
fn get_num_mbs_matches_c() {
    let mut saw_shift_down = false;
    let mut saw_shift_up = false;
    for &bsize in &SQUARE_BSIZES {
        for n in [0i32, 1, 4, 63, 8160, 1_000_000] {
            let got = get_num_mbs(bsize, n);
            assert_eq!(got, ref_fp_get_num_mbs(bsize, n), "bsize={bsize} n={n}");
            if n > 0 {
                if got < n {
                    saw_shift_down = true;
                } else if got > n {
                    saw_shift_up = true;
                }
            }
        }
    }
    assert!(
        saw_shift_down && saw_shift_up,
        "one shift arm was never taken"
    );
}

#[test]
#[should_panic(expected = "square first-pass block")]
fn get_num_mbs_rejects_a_rectangular_block() {
    // BLOCK_8X16 = 4. C asserts this away under a debug build and silently
    // shifts by the sum under -DNDEBUG; the port refuses rather than
    // reproducing an arm upstream calls unsupported (firstpass.c:180).
    let _ = get_num_mbs(4, 100);
}

#[test]
fn get_unit_rows_cols_in_tile_match_c() {
    let mut saw_partial = false;
    for &bsize in &SQUARE_BSIZES {
        for start in [0i32, 4, 17] {
            for len in 0..40i32 {
                let end = start + len;
                let r = get_unit_rows_in_tile(start, end, bsize);
                let c = get_unit_cols_in_tile(start, end, bsize);
                assert_eq!(
                    r,
                    ref_fp_get_unit_rows_in_tile(start, end, bsize),
                    "rows bsize={bsize} [{start},{end})"
                );
                assert_eq!(
                    c,
                    ref_fp_get_unit_cols_in_tile(start, end, bsize),
                    "cols bsize={bsize} [{start},{end})"
                );
                // The CEIL, not a floor: a partial unit still counts.
                let unit = 1 << (bsize_mi_log2(bsize));
                if len % unit != 0 && len > 0 {
                    saw_partial = true;
                    assert_eq!(r, len / unit + 1);
                }
            }
        }
    }
    assert!(saw_partial, "no partial trailing unit was exercised");
}

/// `mi_size_high_log2` for the square block sizes above.
fn bsize_mi_log2(bsize: i32) -> i32 {
    match bsize {
        0 => 0,
        3 => 1,
        6 => 2,
        9 => 3,
        12 => 4,
        15 => 5,
        _ => unreachable!(),
    }
}

#[test]
fn get_search_range_matches_c() {
    let mut saw_floor = false;
    for width in 0..300i32 {
        for &height in &[0i32, 1, 3, 4, 16, 64, 256, 1023, 1024, 4096] {
            let got = get_search_range(width, height);
            assert_eq!(
                got,
                ref_fp_get_search_range(width, height),
                "{width}x{height}"
            );
            if width.min(height) < 4 {
                saw_floor = true;
            }
        }
    }
    assert!(saw_floor, "the MI_SIZE floor was never exercised");
    // A small frame gets a LARGER range — the direction of the relation is
    // the thing a transcription is most likely to invert.
    assert!(get_search_range(64, 64) > get_search_range(1024, 1024));
}

#[test]
fn find_fp_qindex_matches_c() {
    for &bd in &[8u8, 10, 12] {
        assert_eq!(find_fp_qindex(bd), ref_fp_find_fp_qindex(bd), "bd={bd}");
    }
    // Bit depth must matter, or the sweep is one case repeated.
    let a = find_fp_qindex(8);
    let b = find_fp_qindex(10);
    let c = find_fp_qindex(12);
    assert!(a != b || b != c, "all three depths gave {a}");
}

#[test]
fn calc_wavelet_energy_matches_c() {
    let mut saw_true = false;
    for mode in -1..8i32 {
        let got = calc_wavelet_energy(mode);
        assert_eq!(got, ref_fp_calc_wavelet_energy(mode), "deltaq_mode={mode}");
        if got {
            saw_true = true;
        }
    }
    assert!(saw_true, "DELTA_Q_PERCEPTUAL was never matched");
}

// ---------------------------------------------------------------------------
// 3. The frame statistics.
// ---------------------------------------------------------------------------

#[test]
fn raw_motion_error_stdev_matches_c() {
    let mut rng = Lcg(0x5eed_0105);
    // The empty-list early exit, which returns 0 rather than NaN.
    assert_eq!(raw_motion_error_stdev(&[]), 0.0);
    assert_eq!(ref_fp_raw_motion_error_stdev(&[]), 0.0);
    let mut saw_zero_variance = false;
    for count in 1..64usize {
        for trial in 0..40 {
            // Prediction errors are non-negative sums over a 16x16 block, so
            // they top out around 255^2 * 256; a constant list (trial 0)
            // gives a zero standard deviation, which is the degenerate case.
            let list: Vec<i32> = (0..count)
                .map(|_| {
                    if trial == 0 {
                        7777
                    } else {
                        (rng.next_u32() % 16_646_400) as i32
                    }
                })
                .collect();
            let got = raw_motion_error_stdev(&list);
            let want = ref_fp_raw_motion_error_stdev(&list);
            assert_eq!(got.to_bits(), want.to_bits(), "count={count} trial={trial}");
            if trial == 0 {
                assert_eq!(got, 0.0);
                saw_zero_variance = true;
            }
        }
    }
    assert!(
        saw_zero_variance,
        "the constant-list case was never reached"
    );
}

fn to_ref_frame_stats(s: &FrameStats) -> RefFrameStats {
    RefFrameStats {
        intra_error: s.intra_error,
        frame_avg_wavelet_energy: s.frame_avg_wavelet_energy,
        coded_error: s.coded_error,
        sr_coded_error: s.sr_coded_error,
        lt_coded_error: s.lt_coded_error,
        mv_count: s.mv_count,
        inter_count: s.inter_count,
        second_ref_count: s.second_ref_count,
        neutral_count: s.neutral_count,
        intra_skip_count: s.intra_skip_count,
        image_data_start_row: s.image_data_start_row,
        new_mv_count: s.new_mv_count,
        sum_in_vectors: s.sum_in_vectors,
        sum_mvr: s.sum_mvr,
        sum_mvc: s.sum_mvc,
        sum_mvr_abs: s.sum_mvr_abs,
        sum_mvc_abs: s.sum_mvc_abs,
        sum_mvrs: s.sum_mvrs,
        sum_mvcs: s.sum_mvcs,
        intra_factor: s.intra_factor,
        brightness_factor: s.brightness_factor,
    }
}

fn random_frame_stats(rng: &mut Lcg, start_row: i32) -> FrameStats {
    FrameStats {
        intra_error: i64::from(rng.next_u32() % 16_646_400),
        frame_avg_wavelet_energy: i64::from(rng.next_u32() % 16_646_400),
        coded_error: i64::from(rng.next_u32() % 16_646_400),
        sr_coded_error: i64::from(rng.next_u32() % 16_646_400),
        lt_coded_error: i64::from(rng.next_u32() % 16_646_400),
        mv_count: (rng.next_u32() % 2) as i32,
        inter_count: (rng.next_u32() % 2) as i32,
        second_ref_count: (rng.next_u32() % 2) as i32,
        neutral_count: f64::from(rng.next_u32() % 1000) / 1000.0,
        intra_skip_count: (rng.next_u32() % 2) as i32,
        image_data_start_row: start_row,
        new_mv_count: (rng.next_u32() % 2) as i32,
        sum_in_vectors: (rng.next_u32() % 3) as i32 - 1,
        sum_mvr: (rng.next_u32() % 2048) as i32 - 1024,
        sum_mvc: (rng.next_u32() % 2048) as i32 - 1024,
        sum_mvr_abs: (rng.next_u32() % 1024) as i32,
        sum_mvc_abs: (rng.next_u32() % 1024) as i32,
        sum_mvrs: i64::from(rng.next_u32() % 1_000_000),
        sum_mvcs: i64::from(rng.next_u32() % 1_000_000),
        intra_factor: f64::from(rng.next_u32() % 4000) / 1000.0,
        brightness_factor: f64::from(rng.next_u32() % 4000) / 1000.0,
    }
}

#[test]
fn accumulate_frame_stats_matches_c() {
    let mut rng = Lcg(0x5eed_0106);
    let mut saw_first_valid_row = false;
    let mut saw_all_invalid = false;
    for mb_rows in 0..8usize {
        for mb_cols in 0..8usize {
            for trial in 0..30 {
                let n = mb_rows * mb_cols;
                // The interesting axis is image_data_start_row: it is the
                // FIRST non-INVALID_ROW in raster order, so which blocks
                // carry a valid row (and in what order) decides the answer.
                let valid_from = if trial % 3 == 0 {
                    usize::MAX // all invalid
                } else {
                    (rng.next_u32() as usize) % (n + 1)
                };
                let stats: Vec<FrameStats> = (0..n)
                    .map(|i| {
                        let row = if i >= valid_from {
                            (rng.next_u32() % 100) as i32
                        } else {
                            INVALID_ROW
                        };
                        random_frame_stats(&mut rng, row)
                    })
                    .collect();
                let got = FrameStats::accumulate_frame_stats(&stats, mb_rows, mb_cols);
                let cstats: Vec<RefFrameStats> = stats.iter().map(to_ref_frame_stats).collect();
                let want = ref_fp_accumulate_frame_stats(&cstats, mb_rows as i32, mb_cols as i32);
                assert_eq!(
                    to_ref_frame_stats(&got),
                    want,
                    "{mb_rows}x{mb_cols} trial={trial}"
                );
                if got.image_data_start_row != INVALID_ROW {
                    saw_first_valid_row = true;
                } else if n > 0 {
                    saw_all_invalid = true;
                }
            }
        }
    }
    assert!(
        saw_first_valid_row && saw_all_invalid,
        "one image_data_start_row arm was never reached"
    );
}

#[test]
fn accumulate_mv_stats_matches_c() {
    let mut rng = Lcg(0x5eed_0107);
    let mut saw_zero_mv = false;
    let mut saw_new_mv = false;
    let mut saw_repeat_mv = false;
    let mut saw_mid_row = false;
    for mb_rows in [1i32, 2, 5, 8] {
        for mb_cols in [1i32, 2, 5, 8] {
            for mb_row in 0..mb_rows {
                for mb_col in 0..mb_cols {
                    for _ in 0..30 {
                        // best_mv is sub-pel; mv is the full-pel version of a
                        // (possibly different) vector, which is exactly the
                        // pairing the caller passes. Keep zero reachable.
                        let best_mv = if rng.next_u32() % 4 == 0 {
                            (0i16, 0i16)
                        } else {
                            (
                                (rng.next_u32() % 64) as i16 - 32,
                                (rng.next_u32() % 64) as i16 - 32,
                            )
                        };
                        let mv = (best_mv.0 >> 3, best_mv.1 >> 3);
                        let last = if rng.next_u32() % 2 == 0 {
                            best_mv
                        } else {
                            (1, 1)
                        };
                        let stats = random_frame_stats(&mut rng, 0);

                        let mut got = stats;
                        let mut got_last = last;
                        got.accumulate_mv_stats(
                            best_mv,
                            mv,
                            mb_row,
                            mb_col,
                            mb_rows,
                            mb_cols,
                            &mut got_last,
                        );
                        let (want, want_last) = ref_fp_accumulate_mv_stats(
                            best_mv,
                            mv,
                            mb_row,
                            mb_col,
                            mb_rows,
                            mb_cols,
                            last,
                            to_ref_frame_stats(&stats),
                        );
                        assert_eq!(
                            to_ref_frame_stats(&got),
                            want,
                            "best_mv={best_mv:?} mv={mv:?} at ({mb_row},{mb_col}) \
                             of {mb_rows}x{mb_cols}"
                        );
                        assert_eq!(got_last, want_last, "last_non_zero_mv");

                        if best_mv == (0, 0) {
                            saw_zero_mv = true;
                        } else if best_mv == last {
                            saw_repeat_mv = true;
                        } else {
                            saw_new_mv = true;
                        }
                        if mb_row == mb_rows / 2 {
                            saw_mid_row = true;
                        }
                    }
                }
            }
        }
    }
    assert!(saw_zero_mv, "the zero-MV early return was never reached");
    assert!(saw_new_mv, "a new MV was never seen");
    assert!(saw_repeat_mv, "a repeated MV was never seen");
    assert!(
        saw_mid_row,
        "the exact mid-row (which contributes nothing) was never exercised"
    );
}

// ---------------------------------------------------------------------------
// 4. The FIRSTPASS_INFO ring buffer. **Tier 1.**
// ---------------------------------------------------------------------------

/// Compare the port's cursor state against C's after every operation.
fn assert_state(got: &FirstpassInfo, want: &RefFirstpassInfo, ctx: &str) {
    let w = want.state();
    assert_eq!(
        got.start_index() as i32,
        w.start_index,
        "{ctx}: start_index"
    );
    assert_eq!(
        got.stats_count() as i32,
        w.stats_count,
        "{ctx}: stats_count"
    );
    assert_eq!(got.cur_index() as i32, w.cur_index, "{ctx}: cur_index");
    assert_eq!(
        got.future_stats_count() as i32,
        w.future_stats_count,
        "{ctx}: future_stats_count"
    );
    assert_eq!(
        got.past_stats_count() as i32,
        w.past_stats_count,
        "{ctx}: past_stats_count"
    );
    assert_eq!(
        bits_of(&to_ref(got.total_stats())),
        bits_of(&w.total_stats),
        "{ctx}: total_stats"
    );
}

#[test]
fn firstpass_info_static_buf_size_matches_c() {
    assert_eq!(FIRSTPASS_INFO_STATIC_BUF_SIZE, ref_fpi_static_buf_size());
}

#[test]
fn firstpass_info_internal_matches_c() {
    let ok = ref_fpi_codec_ok();
    let mut rng = Lcg(0x5eed_0110);
    let mut saw_full = false;
    let mut saw_wrap = false;
    let mut saw_move_fail = false;
    let mut saw_pop_fail = false;
    for seed in 0..40 {
        let mut got = FirstpassInfo::new_internal();
        let mut want = RefFirstpassInfo::new_internal();
        assert_eq!(want.init_err, ok);
        assert_state(&got, &want, "init");

        for step in 0..400 {
            // Fill the ring outright first — a 50/50 push/pop mix hovers
            // below capacity and never reaches push's failure arm, which is
            // how an earlier version of this test left it untested.
            let op = if step < 60 { 0 } else { rng.next_u32() % 4 };
            let ctx = format!("seed={seed} step={step} op={op}");
            match op {
                0 | 4 | 5 => {
                    let s = random_ref_stats(&mut rng);
                    let g = got.push(&from_ref(&s));
                    let w = want.push(&s);
                    assert_eq!(g.is_ok(), w == ok, "{ctx}: push result");
                    if w != ok {
                        saw_full = true;
                    }
                }
                1 => {
                    let g = got.move_cur_index();
                    let w = want.move_cur_index();
                    assert_eq!(g.is_ok(), w == ok, "{ctx}: move result");
                    if w != ok {
                        saw_move_fail = true;
                    }
                }
                2 => {
                    let g = got.pop();
                    let w = want.pop();
                    assert_eq!(g.is_ok(), w == ok, "{ctx}: pop result");
                    if w != ok {
                        saw_pop_fail = true;
                    }
                }
                _ => {
                    let g = got.move_cur_index_and_pop();
                    let w = want.move_cur_index_and_pop();
                    assert_eq!(g.is_ok(), w == ok, "{ctx}: move_and_pop result");
                }
            }
            assert_state(&got, &want, &ctx);
            if got.start_index() > got.cur_index() {
                saw_wrap = true;
            }

            // peek over the whole legal window, plus one outside on each
            // side. Skip the region where C's `%` yields a NEGATIVE index and
            // reads out of bounds (see FirstpassInfo::peek's doc comment):
            // that is undefined in C, so there is nothing to compare against.
            let past = got.past_stats_count() as i32;
            let future = got.future_stats_count() as i32;
            for off in (-past - 1)..=(future + 1) {
                if got.cur_index() as i32 + off < 0 {
                    continue;
                }
                let g = got.peek(off);
                let w = want.peek(off);
                match (g, w) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(bits_of(&to_ref(a)), bits_of(&b), "{ctx}: peek({off})");
                    }
                    (a, b) => panic!(
                        "{ctx}: peek({off}) arm mismatch {} vs {}",
                        a.is_some(),
                        b.is_some()
                    ),
                }
                assert_eq!(
                    got.future_count(off),
                    want.future_count(off),
                    "{ctx}: future_count({off})"
                );
            }
        }
    }
    assert!(
        saw_full,
        "the ring never filled — push's failure arm is untested"
    );
    assert!(saw_wrap, "the ring never wrapped");
    assert!(saw_move_fail, "move_cur_index never hit its failure arm");
    assert!(saw_pop_fail, "pop never hit its failure arm");
}

#[test]
fn firstpass_info_external_matches_c() {
    let ok = ref_fpi_codec_ok();
    let mut rng = Lcg(0x5eed_0111);
    for n in [1usize, 2, 7, 49] {
        let entries: Vec<RefFirstpassStats> = (0..n).map(|_| random_ref_stats(&mut rng)).collect();
        let mut got = FirstpassInfo::new_external(entries.iter().map(from_ref).collect());
        let mut want = RefFirstpassInfo::new_external(&entries);
        assert_eq!(want.init_err, ok, "init with a non-empty external buffer");
        assert_state(&got, &want, "external init");

        for step in 0..200 {
            let ctx = format!("n={n} step={step}");
            match rng.next_u32() % 3 {
                0 => {
                    assert_eq!(
                        got.move_cur_index().is_ok(),
                        want.move_cur_index() == ok,
                        "{ctx}: move"
                    );
                }
                1 => {
                    assert_eq!(got.pop().is_ok(), want.pop() == ok, "{ctx}: pop");
                }
                _ => {
                    let s = random_ref_stats(&mut rng);
                    assert_eq!(
                        got.push(&from_ref(&s)).is_ok(),
                        want.push(&s) == ok,
                        "{ctx}: push"
                    );
                }
            }
            assert_state(&got, &want, &ctx);
        }
    }
}

#[test]
fn firstpass_info_move_cur_index_stops_one_short() {
    // The contract that makes `peek(0)` always valid: the cursor never
    // advances past the last future record. A `> 0` test instead of `> 1`
    // would let it, and would leave peek(0) reading a record that is not
    // there.
    let mut info = FirstpassInfo::new_internal();
    let s = FirstpassStats::default();
    info.push(&s).unwrap();
    info.push(&s).unwrap();
    assert_eq!(info.future_stats_count(), 2);
    assert_eq!(info.move_cur_index(), Ok(()));
    assert_eq!(info.future_stats_count(), 1);
    assert_eq!(info.move_cur_index(), Err(FirstpassInfoError::Failed));
    assert!(info.peek(0).is_some());
}

// ---------------------------------------------------------------------------
// 5. The per-block helpers. **Tier 1c.**
// ---------------------------------------------------------------------------

use aom_encode::firstpass::{
    HighbdOrLowbd, get_bsize, get_prediction_error, get_prediction_error_bitdepth,
    highbd_get_prediction_error,
};
use aom_sys_ref::{
    ref_fp_get_bsize, ref_fp_get_prediction_error, ref_fp_get_prediction_error_bitdepth,
    ref_fp_highbd_get_prediction_error,
};

#[test]
fn get_bsize_matches_c() {
    let mut saw_full = false;
    let mut saw_half_w = false;
    let mut saw_half_h = false;
    let mut saw_split = false;
    // The two sizes `get_fp_block_size` can return (firstpass.h:554), plus
    // two more square sizes so the square-index mapping is exercised past the
    // pair the encoder uses.
    for &fp_block_size in &[3i32, 6, 9, 12] {
        for mi_rows in [1i32, 2, 4, 7, 8, 16, 33] {
            for mi_cols in [1i32, 2, 4, 7, 8, 16, 33] {
                for unit_row in 0..6i32 {
                    for unit_col in 0..6i32 {
                        let got = get_bsize(mi_rows, mi_cols, fp_block_size, unit_row, unit_col);
                        let want =
                            ref_fp_get_bsize(mi_rows, mi_cols, fp_block_size, unit_row, unit_col);
                        assert_eq!(
                            got, want,
                            "bsize={fp_block_size} {mi_rows}x{mi_cols} unit=({unit_row},{unit_col})"
                        );
                        if got == fp_block_size {
                            saw_full = true;
                        }
                        // PARTITION_VERT halves the width, HORZ the height,
                        // SPLIT both — identified by comparing against the
                        // C answer, not by re-deriving the predicate.
                        let vert = ref_fp_get_bsize(1 << 30, mi_cols, fp_block_size, 0, unit_col);
                        let horz = ref_fp_get_bsize(mi_rows, 1 << 30, fp_block_size, unit_row, 0);
                        if want == vert && want != fp_block_size {
                            saw_half_w = true;
                        }
                        if want == horz && want != fp_block_size {
                            saw_half_h = true;
                        }
                        if want != fp_block_size && want != vert && want != horz {
                            saw_split = true;
                        }
                    }
                }
            }
        }
    }
    assert!(saw_full, "the full-size arm was never reached");
    assert!(saw_half_w, "the half-width arm was never reached");
    assert!(saw_half_h, "the half-height arm was never reached");
    assert!(saw_split, "the split arm was never reached");
}

#[test]
#[should_panic(expected = "is not a BLOCK_SIZE")]
fn get_bsize_rejects_an_unsupported_block_size() {
    // Every one of the 22 real BLOCK_SIZEs has a max dimension in
    // {4, 8, 16, 32, 64, 128}, so C's `default: assert(0)` arm in `get_bsize`
    // is unreachable from a valid size — the port's guard fires one level
    // earlier, in `mi_size_log2`, on an index past BLOCK_SIZES_ALL. Under
    // `-DNDEBUG` C would read past its own tables here.
    let _ = get_bsize(16, 16, 22, 0, 0);
}

/// Deterministic 8-bit plane content with a controllable amount of structure,
/// so the MSE is neither zero nor saturated.
fn fill_plane(rng: &mut Lcg, n: usize, spread: u32) -> Vec<u8> {
    (0..n).map(|_| (rng.next_u32() % spread) as u8).collect()
}

#[test]
fn get_prediction_error_matches_c() {
    let mut rng = Lcg(0x5eed_0120);
    let mut saw_nonzero = false;
    let mut saw_zero = false;
    // The four block sizes `get_bsize` can produce for a 16x16 first pass,
    // plus one that falls into C's 16x16 `default:` arm (BLOCK_4X4 = 0), so
    // the fall-through is tested rather than assumed.
    for &bsize in &[3i32, 5, 4, 6, 0] {
        for &(src_stride, ref_stride) in &[(16usize, 16usize), (32, 24), (17, 64)] {
            for trial in 0..300 {
                // 16x16 is the largest kernel; give both planes room for the
                // stride and a full 16 rows.
                let src = fill_plane(&mut rng, src_stride * 32 + 64, 256);
                let reference = if trial % 7 == 0 {
                    // An identical block, which must give sse == 0.
                    let mut r = vec![0u8; ref_stride * 32 + 64];
                    for y in 0..16 {
                        for x in 0..16 {
                            r[y * ref_stride + x] = src[y * src_stride + x];
                        }
                    }
                    r
                } else {
                    fill_plane(&mut rng, ref_stride * 32 + 64, 256)
                };
                let got = get_prediction_error(bsize, &src, src_stride, &reference, ref_stride);
                let want = ref_fp_get_prediction_error(
                    bsize,
                    &src,
                    src_stride as i32,
                    &reference,
                    ref_stride as i32,
                );
                assert_eq!(
                    got, want,
                    "bsize={bsize} strides=({src_stride},{ref_stride}) trial={trial}"
                );
                if got == 0 {
                    saw_zero = true;
                } else {
                    saw_nonzero = true;
                }
            }
        }
    }
    assert!(saw_zero, "an identical block never gave sse == 0");
    assert!(saw_nonzero, "every block gave sse == 0");
}

#[test]
fn highbd_get_prediction_error_matches_c() {
    let mut rng = Lcg(0x5eed_0121);
    let mut per_bd_saw_difference = false;
    for &bd in &[8i32, 10, 12] {
        let max = 1u32 << bd;
        for &bsize in &[3i32, 5, 4, 6] {
            for &(src_stride, ref_stride) in &[(16usize, 16usize), (40, 23)] {
                for _ in 0..200 {
                    let src: Vec<u16> = (0..src_stride * 32 + 64)
                        .map(|_| (rng.next_u32() % max) as u16)
                        .collect();
                    let reference: Vec<u16> = (0..ref_stride * 32 + 64)
                        .map(|_| (rng.next_u32() % max) as u16)
                        .collect();
                    let got = highbd_get_prediction_error(
                        bsize, &src, src_stride, &reference, ref_stride, bd as u8,
                    );
                    let want = ref_fp_highbd_get_prediction_error(
                        bsize,
                        &src,
                        src_stride as i32,
                        &reference,
                        ref_stride as i32,
                        bd,
                    );
                    assert_eq!(
                        got, want,
                        "bd={bd} bsize={bsize} strides=({src_stride},{ref_stride})"
                    );
                }
            }
        }
    }
    // The three depths must NOT be the same function: 10 shifts sse down by
    // 4 bits and 12 by 8. Feed one fixed pair to all three.
    //
    // The samples stay in 0..=255, and that bound is the CONTRACT, not
    // caution. MEASURED on this build: with samples in 0..=1023 at
    // `bd == 8`, C's `aom_highbd_8_mse16x16` returns 2_741_760 where the
    // scalar definition (and this port) give 42_944_000 — its kernel
    // accumulates in a width that assumes 8-bit samples. At `bd == 10` and
    // `bd == 12` the same inputs agree exactly, which is why the randomized
    // sweep above (which draws `0..1 << bd`) passes at every depth. A highbd
    // plane at bit depth 8 holds 8-bit samples, so the encoder cannot reach
    // the divergent input; feeding it would be testing a call C is not
    // defined for (DIFFERENTIAL_PLAYBOOK §3a(d)).
    let src: Vec<u16> = (0..16 * 16).map(|i| ((i * 37) % 256) as u16).collect();
    let reference: Vec<u16> = (0..16 * 16).map(|i| ((i * 11) % 256) as u16).collect();
    let a = highbd_get_prediction_error(6, &src, 16, &reference, 16, 8);
    let b = highbd_get_prediction_error(6, &src, 16, &reference, 16, 10);
    let c = highbd_get_prediction_error(6, &src, 16, &reference, 16, 12);
    if a != b || b != c {
        per_bd_saw_difference = true;
    }
    assert!(
        per_bd_saw_difference,
        "all three bit depths returned {a} — the sse normalisation is inert"
    );
    assert_eq!(
        a,
        ref_fp_highbd_get_prediction_error(6, &src, 16, &reference, 16, 8)
    );
    assert_eq!(
        b,
        ref_fp_highbd_get_prediction_error(6, &src, 16, &reference, 16, 10)
    );
    assert_eq!(
        c,
        ref_fp_highbd_get_prediction_error(6, &src, 16, &reference, 16, 12)
    );
    // C's `switch (bd)` puts `default:` on the 8-bit arm, so an out-of-range
    // depth is 8-bit, not an error.
    assert_eq!(
        highbd_get_prediction_error(6, &src, 16, &reference, 16, 9),
        ref_fp_highbd_get_prediction_error(6, &src, 16, &reference, 16, 9)
    );
}

#[test]
fn get_prediction_error_bitdepth_matches_c() {
    let mut rng = Lcg(0x5eed_0122);
    for &(is_hbd, bd) in &[(false, 8i32), (true, 8), (true, 10), (true, 12)] {
        for &bsize in &[3i32, 5, 4, 6] {
            for _ in 0..150 {
                let stride = 24usize;
                let max = 1u32 << bd;
                let src8 = fill_plane(&mut rng, stride * 32 + 64, 256);
                let ref8 = fill_plane(&mut rng, stride * 32 + 64, 256);
                let src16: Vec<u16> = (0..stride * 32 + 64)
                    .map(|_| (rng.next_u32() % max) as u16)
                    .collect();
                let ref16: Vec<u16> = (0..stride * 32 + 64)
                    .map(|_| (rng.next_u32() % max) as u16)
                    .collect();
                let got = if is_hbd {
                    get_prediction_error_bitdepth(
                        bd as u8,
                        bsize,
                        HighbdOrLowbd::Highbd(&src16),
                        HighbdOrLowbd::Highbd(&ref16),
                        stride,
                        stride,
                    )
                } else {
                    get_prediction_error_bitdepth(
                        bd as u8,
                        bsize,
                        HighbdOrLowbd::Lowbd(&src8),
                        HighbdOrLowbd::Lowbd(&ref8),
                        stride,
                        stride,
                    )
                };
                let want = ref_fp_get_prediction_error_bitdepth(
                    is_hbd,
                    bd,
                    bsize,
                    &src16,
                    &src8,
                    stride as i32,
                    &ref16,
                    &ref8,
                    stride as i32,
                );
                assert_eq!(
                    i64::from(got),
                    i64::from(want),
                    "is_hbd={is_hbd} bd={bd} bsize={bsize}"
                );
            }
        }
    }
}

use aom_encode::firstpass::UpdateFirstpassStatsParams;
use aom_sys_ref::ref_fp_update_firstpass_stats;

#[test]
fn update_firstpass_stats_matches_c() {
    let mut rng = Lcg(0x5eed_0130);
    let mut saw_motion = false;
    let mut saw_no_motion = false;
    let mut saw_invalid_start_row = false;
    // BLOCK_8X8 = 3 and BLOCK_16X16 = 6 are what `get_fp_block_size` returns;
    // the two counts (num_mbs vs num_mbs_16x16) differ by 4x at BLOCK_8X8,
    // so a port that used one for both fails on that arm alone.
    for &fp_block_size in &[6i32, 3] {
        for &(width, height) in &[(176i32, 144i32), (1920, 1080), (3840, 2160)] {
            let num_mbs_16x16 = ((width + 15) / 16) * ((height + 15) / 16);
            for trial in 0..400 {
                let mut stats = random_frame_stats(&mut rng, 0);
                // Frame-level sums, not per-block ones: scale them up.
                stats.coded_error = i64::from(rng.next_u32()) << 8;
                stats.sr_coded_error = i64::from(rng.next_u32()) << 8;
                stats.lt_coded_error = i64::from(rng.next_u32()) << 8;
                stats.intra_error = i64::from(rng.next_u32()) << 8;
                stats.frame_avg_wavelet_energy = i64::from(rng.next_u32());
                stats.inter_count = (rng.next_u32() % (num_mbs_16x16 as u32)) as i32;
                stats.second_ref_count = (rng.next_u32() % 1000) as i32;
                stats.intra_skip_count = (rng.next_u32() % 1000) as i32;
                stats.neutral_count = f64::from(rng.next_u32() % 100_000) / 8.0;
                stats.image_data_start_row = if trial % 5 == 0 {
                    saw_invalid_start_row = true;
                    INVALID_ROW
                } else {
                    (rng.next_u32() % 200) as i32
                };
                // The zero-motion arm has to be reached by construction, not
                // by luck: it is nine fields wide.
                if trial % 3 == 0 {
                    stats.mv_count = 0;
                    saw_no_motion = true;
                } else {
                    stats.mv_count = 1 + (rng.next_u32() % 10_000) as i32;
                    saw_motion = true;
                }
                stats.sum_mvr = (rng.next_u32() % 200_000) as i32 - 100_000;
                stats.sum_mvc = (rng.next_u32() % 200_000) as i32 - 100_000;
                stats.sum_mvr_abs = (rng.next_u32() % 200_000) as i32;
                stats.sum_mvc_abs = (rng.next_u32() % 200_000) as i32;
                stats.sum_mvrs = i64::from(rng.next_u32());
                stats.sum_mvcs = i64::from(rng.next_u32());
                stats.sum_in_vectors = (rng.next_u32() % 20_000) as i32 - 10_000;
                stats.new_mv_count = (rng.next_u32() % 10_000) as i32;
                stats.intra_factor = f64::from(rng.next_u32() % 100_000) / 1000.0;
                stats.brightness_factor = f64::from(rng.next_u32() % 100_000) / 1000.0;

                let p = UpdateFirstpassStatsParams {
                    num_mbs_16x16,
                    fp_block_size,
                    frame_number: (rng.next_u32() % 100_000) as i32,
                    ts_duration: i64::from(rng.next_u32()) * 1000,
                    raw_err_stdev: f64::from(rng.next_u32()) / 4096.0,
                    width,
                    height,
                };
                let got = FirstpassStats::from_frame_stats(&stats, p);
                let want = ref_fp_update_firstpass_stats(
                    p.num_mbs_16x16,
                    p.fp_block_size,
                    p.frame_number,
                    p.ts_duration,
                    p.raw_err_stdev,
                    p.width,
                    p.height,
                    &to_ref_frame_stats(&stats),
                );
                assert_eq!(
                    bits_of(&to_ref(&got)),
                    bits_of(&want),
                    "bsize={fp_block_size} {width}x{height} trial={trial} \
                     mv_count={}",
                    stats.mv_count
                );
            }
        }
    }
    assert!(saw_motion, "the mv_count > 0 arm was never reached");
    assert!(saw_no_motion, "the mv_count == 0 arm was never reached");
    assert!(
        saw_invalid_start_row,
        "inactive_zone_rows never carried INVALID_ROW"
    );
}

#[test]
fn update_firstpass_stats_uses_both_macroblock_counts() {
    // The percentages divide by `get_num_mbs(fp_block_size, num_mbs_16x16)`
    // and `normalize` divides the errors by `num_mbs_16x16`. At BLOCK_8X8
    // those differ by 4x, so a port that used one for both would be wrong on
    // one of the two groups. Prove the two block sizes really do give
    // different answers, or the sweep above is one case repeated.
    let mut rng = Lcg(0x5eed_0131);
    let mut stats = random_frame_stats(&mut rng, 3);
    stats.mv_count = 500;
    stats.inter_count = 900;
    stats.coded_error = 1 << 30;
    let base = UpdateFirstpassStatsParams {
        num_mbs_16x16: 8160,
        fp_block_size: 6,
        frame_number: 7,
        ts_duration: 33_000,
        raw_err_stdev: 12.5,
        width: 1920,
        height: 1080,
    };
    let a = FirstpassStats::from_frame_stats(&stats, base);
    let b = FirstpassStats::from_frame_stats(
        &stats,
        UpdateFirstpassStatsParams {
            fp_block_size: 3,
            ..base
        },
    );
    assert_ne!(a.pcnt_inter, b.pcnt_inter, "the percentages did not move");
    // The errors normalize by num_mbs_16x16, which is the SAME in both, but
    // min_err is scaled by sqrt(num_mbs), which is not — so they differ too,
    // and by a different ratio than the percentages.
    assert_ne!(a.coded_error, b.coded_error);
    assert_ne!(
        a.pcnt_inter / b.pcnt_inter,
        a.coded_error / b.coded_error,
        "the two denominators moved together — they are not distinguishable"
    );
}
