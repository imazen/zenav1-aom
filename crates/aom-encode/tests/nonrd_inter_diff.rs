//! Differential harness for `aom_encode::nonrd_inter` — the mode-skip and
//! prune cascade of the speed 8/9 inter pickmode.
//!
//! **Tier 1c throughout.** `nm -g` reports exactly two exported symbols for
//! `av1/encoder/nonrd_pickmode.c` (`av1_nonrd_pick_intra_mode` and
//! `av1_nonrd_pick_inter_mode_sb`), so none of these helpers has an address a
//! tier-1 differential could take. `shim/nonrd_pick_shim.c` compiles that .c
//! verbatim under its Release flags and wraps the statics.
//!
//! | test | C oracle |
//! |---|---|
//! | `mode_idx_table_matches_c` | `mode_idx` (nonrd_opt.h:127) |
//! | `enum_sizes_match_c` | `RTC_MODES`, `RTC_INTER_MODES`, `REF_FRAMES`, `MB_MODE_COUNT` |
//! | `skip_mode_by_threshold_matches_c` | `skip_mode_by_threshold` (:1933) |
//! | `skip_mode_by_low_temp_matches_c` | `skip_mode_by_low_temp` (:1961) |
//! | `skip_mode_by_bsize_and_ref_frame_matches_c` | `skip_mode_by_bsize_and_ref_frame` (:1978) |
//! | `skip_comp_based_on_var_matches_c` | `skip_comp_based_on_var` (:2165) |
//! | `previous_mode_performed_poorly_matches_c` | `previous_mode_performed_poorly` (:2286) |
//! | `prune_compoundmode_matches_c` | `prune_compoundmode_with_singlemode_var` (:2306) |
//! | `ac_thr_factor_matches_c` | `ac_thr_factor` (:580) |
//! | `calculate_variance_matches_c` | `calculate_variance` (:556) |
//!
//! `enum_sizes_match_c` and `mode_idx_table_matches_c` are the TU-vs-port
//! gate: every other test indexes those tables, so a size or entry drift would
//! silently move what is being compared.
//!
//! # What bounds the generators
//! * `mode` is one of the four RTC inter modes (NEARESTMV, NEARMV, GLOBALMV,
//!   NEWMV) wherever C's `INTER_OFFSET` indexes a `RTC_INTER_MODES` array;
//!   feeding an intra mode there indexes out of bounds in C.
//! * `ref_frame` is `LAST_FRAME..ALTREF_FRAME` (1..8). Row 0 of `mode_idx` is
//!   the INTRA row and the inter cascade never reaches it.
//! * `rd_threshes` is swept including `INT_MAX`, which is
//!   `rd_less_than_thresh`'s own special case and the only value that makes it
//!   true independently of the arithmetic.
//! * `vars` entries include `UINT_MAX`, which is C's "not measured" sentinel
//!   and the value `prune_compoundmode_with_singlemode_var` gates on.

use aom_encode::nonrd_inter::{
    MODE_IDX, REF_FRAMES, RTC_INTER_MODES, RTC_MODES, ac_thr_factor, calculate_variance,
    previous_mode_performed_poorly, prune_compoundmode_with_singlemode_var, skip_comp_based_on_var,
    skip_mode_by_bsize_and_ref_frame, skip_mode_by_low_temp, skip_mode_by_threshold,
};
use aom_encode::rdopt_mv::PredMode;
use aom_encode::var_part::SourceSad;
use aom_sys_ref::{
    ref_nrp_ac_thr_factor, ref_nrp_calculate_variance, ref_nrp_mb_mode_count, ref_nrp_mode_idx,
    ref_nrp_previous_mode_performed_poorly, ref_nrp_prune_compoundmode_with_singlemode_var,
    ref_nrp_ref_frames, ref_nrp_rtc_inter_modes, ref_nrp_rtc_modes, ref_nrp_skip_comp_based_on_var,
    ref_nrp_skip_mode_by_bsize_and_ref_frame, ref_nrp_skip_mode_by_low_temp,
    ref_nrp_skip_mode_by_threshold,
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
}

/// The four RTC inter modes, as `(PredMode, raw PREDICTION_MODE)`.
const RTC_MODE_SET: [(PredMode, i32); 4] = [
    (PredMode::NearestMv, 13),
    (PredMode::NearMv, 14),
    (PredMode::GlobalMv, 15),
    (PredMode::NewMv, 16),
];

/// `MAX_MODES` (`enc_enums.h`) — the length of the RD threshold arrays.
const MAX_MODES: usize = 169;

#[test]
fn enum_sizes_match_c() {
    assert_eq!(RTC_MODES, ref_nrp_rtc_modes());
    assert_eq!(RTC_INTER_MODES, ref_nrp_rtc_inter_modes());
    assert_eq!(REF_FRAMES, ref_nrp_ref_frames());
    // MB_MODE_COUNT is not mirrored as a constant here, but every test that
    // builds a frame_mv array must agree with C on it.
    assert_eq!(ref_nrp_mb_mode_count(), 25, "MB_MODE_COUNT drifted");
}

#[test]
fn mode_idx_table_matches_c() {
    let want = ref_nrp_mode_idx();
    for r in 0..REF_FRAMES {
        for m in 0..RTC_MODES {
            assert_eq!(
                MODE_IDX[r][m] as i32,
                want[r * RTC_MODES + m],
                "mode_idx[{r}][{m}]"
            );
        }
    }
}

#[test]
fn skip_mode_by_threshold_matches_c() {
    let mut rng = Rng::new(0x5417_0001);
    let mut skipped = 0usize;
    let mut kept = 0usize;
    for _ in 0..3000 {
        // Thresholds in the range av1_set_rd_speed_thresholds produces, plus
        // INT_MAX, which is rd_less_than_thresh's own special case.
        let rd_threshes: Vec<i32> = (0..MAX_MODES)
            .map(|_| match rng.below(6) {
                0 => i32::MAX,
                1 => 0,
                _ => rng.below(1 << 20) as i32,
            })
            .collect();
        let freq_fact: Vec<i32> = (0..MAX_MODES).map(|_| 32 + rng.below(96) as i32).collect();
        let (mode, raw_mode) = RTC_MODE_SET[rng.below(4) as usize];
        let ref_frame = 1 + rng.below(7) as i32;
        let mv_as_int = match rng.below(3) {
            0 => 0u32,
            _ => rng.next_u64() as u32,
        };
        let frames_since_golden = rng.below(12) as i32;
        let best_cost = match rng.below(4) {
            0 => i64::MAX,
            _ => i64::from(rng.below(1 << 24)),
        };
        let best_skip = rng.below(2) == 0;
        // extra_shift is a speed-feature-derived shift; the encoder uses 0..3.
        let extra_shift = rng.below(4) as i32;

        let want = ref_nrp_skip_mode_by_threshold(
            raw_mode,
            ref_frame,
            mv_as_int,
            frames_since_golden,
            &rd_threshes,
            &freq_fact,
            best_cost,
            best_skip,
            extra_shift,
        );
        let got = skip_mode_by_threshold(
            mode,
            ref_frame as usize,
            mv_as_int,
            frames_since_golden,
            &rd_threshes,
            &freq_fact,
            best_cost,
            best_skip,
            extra_shift as u32,
        );
        assert_eq!(
            got, want,
            "mode {raw_mode} ref {ref_frame} mv {mv_as_int:#x} fsg {frames_since_golden} \
             best {best_cost} skip {best_skip} shift {extra_shift}"
        );
        if want {
            skipped += 1;
        } else {
            kept += 1;
        }
    }
    assert!(
        skipped > 100 && kept > 100,
        "one arm never fired: {skipped}/{kept}"
    );
}

#[test]
fn skip_mode_by_low_temp_matches_c() {
    let mut skipped = 0usize;
    let mut kept = 0usize;
    for (mode, raw_mode) in RTC_MODE_SET {
        for ref_frame in 1..8i32 {
            // Every block size, including the non-square ones the RT
            // partitioner can produce.
            for bsize in 0..22i32 {
                for (sad_raw, sad) in [
                    (0i32, SourceSad::Zero),
                    (1, SourceSad::VeryLow),
                    (2, SourceSad::Low),
                    (3, SourceSad::Med),
                    (4, SourceSad::High),
                ] {
                    for &mv in &[0u32, 1, 0x0004_0004] {
                        for &force in &[false, true] {
                            let want = ref_nrp_skip_mode_by_low_temp(
                                raw_mode, ref_frame, bsize, sad_raw, mv, force,
                            );
                            let got = skip_mode_by_low_temp(
                                mode,
                                ref_frame as usize,
                                bsize as usize,
                                sad,
                                mv,
                                force,
                            );
                            assert_eq!(
                                got, want,
                                "mode {raw_mode} ref {ref_frame} bsize {bsize} sad {sad_raw} mv {mv} force {force}"
                            );
                            if want {
                                skipped += 1;
                            } else {
                                kept += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        skipped > 100 && kept > 100,
        "one arm never fired: {skipped}/{kept}"
    );
}

#[test]
fn skip_mode_by_bsize_and_ref_frame_matches_c() {
    let mut skipped = 0usize;
    let mut kept = 0usize;
    for (mode, raw_mode) in RTC_MODE_SET {
        for ref_frame in 1..8i32 {
            for bsize in 0..22i32 {
                for extra_prune in 0..3i32 {
                    // Straddle the 500 threshold C compares sse_zeromv_norm to.
                    for &sse in &[0u32, 499, 500, 501, 1 << 20] {
                        for &more in &[false, true] {
                            for &skip_near in &[false, true] {
                                let want = ref_nrp_skip_mode_by_bsize_and_ref_frame(
                                    raw_mode,
                                    ref_frame,
                                    bsize,
                                    extra_prune,
                                    sse,
                                    more,
                                    skip_near,
                                );
                                let got = skip_mode_by_bsize_and_ref_frame(
                                    mode,
                                    ref_frame as usize,
                                    bsize as usize,
                                    extra_prune,
                                    sse,
                                    more,
                                    skip_near,
                                );
                                assert_eq!(
                                    got, want,
                                    "mode {raw_mode} ref {ref_frame} bsize {bsize} prune {extra_prune} sse {sse} more {more} near {skip_near}"
                                );
                                if want {
                                    skipped += 1;
                                } else {
                                    kept += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        skipped > 100 && kept > 100,
        "one arm never fired: {skipped}/{kept}"
    );
}

#[test]
fn skip_comp_based_on_var_matches_c() {
    let mut rng = Rng::new(0x5C0B_0002);
    let mut yes = 0usize;
    let mut no = 0usize;
    // The two thresholds are ~4967 (64) and ~1025 (32); draw around them.
    // `best_var` is the MIN over all 32 entries, so a per-entry draw that
    // includes small values puts the minimum near zero almost every time and
    // the four bsize thresholds are never separated from each other. Instead
    // every entry is drawn from ONE band per iteration, and the bands are
    // placed on each threshold: thresh_32 / 4 ~= 256, thresh_32 ~= 1025,
    // thresh_64 ~= 4967, 4 * thresh_64 ~= 19868.
    for iter in 0..2000 {
        let band = iter % 6;
        let flat: Vec<u32> = (0..RTC_INTER_MODES * REF_FRAMES)
            .map(|_| match band {
                0 => 200 + rng.below(120),    // straddles thresh_32 / 4
                1 => 980 + rng.below(90),     // straddles thresh_32
                2 => 4900 + rng.below(140),   // straddles thresh_64
                3 => 19_800 + rng.below(140), // straddles 4 * thresh_64
                4 => u32::MAX,
                _ => rng.below(1 << 20),
            })
            .collect();
        let mut arr = [[0u32; REF_FRAMES]; RTC_INTER_MODES];
        for m in 0..RTC_INTER_MODES {
            arr[m].copy_from_slice(&flat[m * REF_FRAMES..(m + 1) * REF_FRAMES]);
        }
        for bsize in 0..22i32 {
            let want = ref_nrp_skip_comp_based_on_var(&flat, bsize);
            let got = skip_comp_based_on_var(&arr, bsize as usize);
            assert_eq!(got, want, "bsize {bsize}");
            if want {
                yes += 1;
            } else {
                no += 1;
            }
        }
    }
    assert!(yes > 100 && no > 100, "one arm never fired: {yes}/{no}");
}

/// Draw a `(vars, uv_dist)` pair whose values straddle the `1.125 *` margin,
/// including the `UINT_MAX` / `INT64_MAX` "not measured" sentinels.
fn draw_var_tables(rng: &mut Rng) -> (Vec<u32>, Vec<i64>) {
    let n = RTC_INTER_MODES * REF_FRAMES;
    // Two magnitude regimes. The small one keeps values within a few percent
    // of each other so the 1.125 factor is what decides. The LARGE one is
    // above 2^24, where an f32 can no longer represent every integer -- that
    // is the only place a widened (f64) spelling of C's `float mult` diverges,
    // and without it the f32-vs-f64 choice is untested.
    let large = rng.below(2) == 0;
    let vars: Vec<u32> = (0..n)
        .map(|_| match rng.below(8) {
            0 => u32::MAX,
            1 => 0,
            _ if large => 0x0400_0000 + rng.below(1 << 12),
            _ => 1000 + rng.below(300),
        })
        .collect();
    let uv: Vec<i64> = (0..n)
        .map(|_| match rng.below(8) {
            0 => i64::MAX,
            1 => 0,
            _ if large => 0x0400_0000 + i64::from(rng.below(1 << 12)),
            _ => 5000 + i64::from(rng.below(1500)),
        })
        .collect();
    (vars, uv)
}

#[test]
fn previous_mode_performed_poorly_matches_c() {
    let mut rng = Rng::new(0x9E11_0003);
    let mut bad = 0usize;
    let mut ok = 0usize;
    for _ in 0..3000 {
        let (vars, uv) = draw_var_tables(&mut rng);
        let mut varr = [[0u32; REF_FRAMES]; RTC_INTER_MODES];
        let mut uarr = [[0i64; REF_FRAMES]; RTC_INTER_MODES];
        for m in 0..RTC_INTER_MODES {
            varr[m].copy_from_slice(&vars[m * REF_FRAMES..(m + 1) * REF_FRAMES]);
            uarr[m].copy_from_slice(&uv[m * REF_FRAMES..(m + 1) * REF_FRAMES]);
        }
        for (mode, raw_mode) in RTC_MODE_SET {
            for ref_frame in 1..8i32 {
                // C asserts best_var != UINT_MAX, so a column that is all
                // sentinel is a call the encoder cannot make; skip it rather
                // than driving C into its own assert.
                if (0..RTC_INTER_MODES).all(|m| varr[m][ref_frame as usize] == u32::MAX) {
                    continue;
                }
                let want = ref_nrp_previous_mode_performed_poorly(raw_mode, ref_frame, &vars, &uv);
                let got = previous_mode_performed_poorly(mode, ref_frame as usize, &varr, &uarr);
                assert_eq!(got, want, "mode {raw_mode} ref {ref_frame}");
                if want {
                    bad += 1;
                } else {
                    ok += 1;
                }
            }
        }
    }
    assert!(bad > 100 && ok > 100, "one arm never fired: {bad}/{ok}");
}

#[test]
fn prune_compoundmode_matches_c() {
    let mut rng = Rng::new(0xC0FF_0004);
    let mb_mode_count = ref_nrp_mb_mode_count();
    let mut pruned = 0usize;
    let mut kept = 0usize;
    // The compound modes: NEAREST_NEARESTMV .. NEW_NEWMV (17..24).
    for _ in 0..1500 {
        let (vars, uv) = draw_var_tables(&mut rng);
        let mut varr = [[0u32; REF_FRAMES]; RTC_INTER_MODES];
        let mut uarr = [[0i64; REF_FRAMES]; RTC_INTER_MODES];
        for m in 0..RTC_INTER_MODES {
            varr[m].copy_from_slice(&vars[m * REF_FRAMES..(m + 1) * REF_FRAMES]);
            uarr[m].copy_from_slice(&uv[m * REF_FRAMES..(m + 1) * REF_FRAMES]);
        }
        // frame_mv / mode_checked are indexed by the RAW mode, so they have
        // MB_MODE_COUNT rows -- mixing them up with the RTC_INTER_MODES tables
        // is the obvious transcription error, and the two shapes are kept apart.
        let frame_mv_flat: Vec<u32> = (0..mb_mode_count * REF_FRAMES)
            // Repeat from a small pool so the `frame_mv[single] ==
            // frame_mv[compound]` guard actually passes sometimes.
            .map(|_| rng.below(3))
            .collect();
        let mode_checked_flat: Vec<u8> = (0..mb_mode_count * REF_FRAMES)
            .map(|_| rng.below(2) as u8)
            .collect();
        let mut frame_mv = vec![[0u32; REF_FRAMES]; mb_mode_count];
        let mut mode_checked = vec![[0u8; REF_FRAMES]; mb_mode_count];
        for m in 0..mb_mode_count {
            frame_mv[m].copy_from_slice(&frame_mv_flat[m * REF_FRAMES..(m + 1) * REF_FRAMES]);
            mode_checked[m]
                .copy_from_slice(&mode_checked_flat[m * REF_FRAMES..(m + 1) * REF_FRAMES]);
        }

        for raw_mode in 17..25i32 {
            let mode = PredMode::from_i32(raw_mode).expect("compound mode");
            for ref_frame in 1..8i32 {
                for ref_frame2 in 1..8i32 {
                    // C asserts inside previous_mode_performed_poorly, so skip
                    // the columns that would trip it.
                    let all_sentinel =
                        |r: usize| (0..RTC_INTER_MODES).all(|m| varr[m][r] == u32::MAX);
                    if all_sentinel(ref_frame as usize) || all_sentinel(ref_frame2 as usize) {
                        continue;
                    }
                    let want = ref_nrp_prune_compoundmode_with_singlemode_var(
                        raw_mode,
                        ref_frame,
                        ref_frame2,
                        &frame_mv_flat,
                        &mode_checked_flat,
                        &vars,
                        &uv,
                    );
                    let got = prune_compoundmode_with_singlemode_var(
                        mode,
                        ref_frame as usize,
                        ref_frame2 as usize,
                        &frame_mv,
                        &mode_checked,
                        &varr,
                        &uarr,
                    );
                    assert_eq!(got, want, "mode {raw_mode} refs ({ref_frame},{ref_frame2})");
                    if want {
                        pruned += 1;
                    } else {
                        kept += 1;
                    }
                }
            }
        }
    }
    assert!(
        pruned > 50 && kept > 50,
        "one arm never fired: {pruned}/{kept}"
    );
}

#[test]
fn ac_thr_factor_matches_c() {
    for speed in 0..11i32 {
        // Resolutions around the 640x480 corner the function tests.
        for &(w, h) in &[
            (16i32, 16i32),
            (640, 480),
            (641, 480),
            (640, 481),
            (1280, 720),
            (3840, 2160),
        ] {
            for norm_sum in -2..10i32 {
                assert_eq!(
                    i64::from(ref_nrp_ac_thr_factor(speed, w, h, norm_sum)),
                    ac_thr_factor(speed, w, h, norm_sum),
                    "speed {speed} {w}x{h} norm_sum {norm_sum}"
                );
            }
        }
    }
}

#[test]
fn calculate_variance_matches_c() {
    let mut rng = Rng::new(0xCA1C_0005);
    // (bw, bh) are b_width_log2 / b_height_log2 of the BLOCK; tx_size picks
    // the unit. Only the combinations the caller (block_variance_16x16_dual's
    // consumer) produces are swept: square blocks 32x32 and up over 16x16 or
    // 8x8 units, which is what calculate_variance's own asserts imply.
    for &(bw, bh, tx_size, unit_log2) in &[
        (3u32, 3u32, 2i32, 2u32), // 32x32 block, TX_16X16 units
        (4, 4, 2, 2),             // 64x64 block, TX_16X16 units
        (4, 4, 1, 1),             // 64x64 block, TX_8X8 units
        (5, 5, 2, 2),             // 128x128 block, TX_16X16 units
    ] {
        let nw = 1usize << (bw - unit_log2);
        let nh = 1usize << (bh - unit_log2);
        for _ in 0..40 {
            let sse_i: Vec<u32> = (0..nw * nh).map(|_| rng.below(1 << 22)).collect();
            let sum_i: Vec<i32> = (0..nw * nh)
                .map(|_| rng.below(1 << 15) as i32 - (1 << 14))
                .collect();
            let (want_var, want_sse, want_sum) = ref_nrp_calculate_variance(
                bw as i32, bh as i32, tx_size, unit_log2, &sse_i, &sum_i,
            );
            let (got_var, got_sse, got_sum) = calculate_variance(bw, bh, unit_log2, &sse_i, &sum_i);
            assert_eq!(got_sse, want_sse, "sse_o at bw {bw} tx {tx_size}");
            assert_eq!(got_sum, want_sum, "sum_o at bw {bw} tx {tx_size}");
            assert_eq!(got_var, want_var, "var_o at bw {bw} tx {tx_size}");
        }
    }
}

#[test]
fn previous_mode_performed_poorly_uses_c_float_width() {
    // C's `mult` is a `float`, and both comparison operands convert to float
    // too. Above 2^24 an f32 can no longer hold every integer, so a widened
    // (f64) spelling of the same expression gives a different verdict. The
    // random sweep above never lands there, so these cells are pinned.
    //
    // REACHABILITY. `vars` holds `aom_variance<W>x<H>_c`'s return, which is a
    // BLOCK total (`sse - sum^2 / n`), not a per-pixel figure: at 8 bits and
    // BLOCK_128X128 its ceiling is 128 * 128 * 255^2 = 1_065_369_600. Both
    // members of each pair below are under that, so the encoder can produce
    // them. Pairs whose values exceed it were discarded.
    //
    // Each entry is `(best_var, this_var)` with `this ~= 1.125 * best`, which
    // is where the two spellings straddle.
    const F32_WIDTH_CELLS: &[(u32, u32)] =
        &[(864_954_665, 973_074_047), (482_753_134, 543_097_277)];

    let mut separated = false;
    for &(best, this) in F32_WIDTH_CELLS {
        assert!(
            u64::from(this) <= 1_065_369_600,
            "cell ({best}, {this}) is above what aom_variance128x128 can return"
        );
        // Put `best` in every slot of the reference's column except the mode
        // under test, which gets `this`. uv_dist is pinned at the sentinel so
        // the chroma term is inert and the luma comparison alone decides.
        for (mode, raw_mode) in RTC_MODE_SET {
            let off = (raw_mode - 13) as usize;
            let mut varr = [[0u32; REF_FRAMES]; RTC_INTER_MODES];
            let uarr = [[i64::MAX; REF_FRAMES]; RTC_INTER_MODES];
            for m in 0..RTC_INTER_MODES {
                for r in 0..REF_FRAMES {
                    varr[m][r] = if m == off { this } else { best };
                }
            }
            let ref_frame = 1usize;
            let vars_flat: Vec<u32> = varr.iter().flatten().copied().collect();
            let uv_flat: Vec<i64> = uarr.iter().flatten().copied().collect();
            let want = ref_nrp_previous_mode_performed_poorly(
                raw_mode,
                ref_frame as i32,
                &vars_flat,
                &uv_flat,
            );
            let got = previous_mode_performed_poorly(mode, ref_frame, &varr, &uarr);
            assert_eq!(got, want, "mode {raw_mode} at ({best}, {this})");
            // The f64 spelling says "poorly" here; the f32 one says otherwise.
            if !want {
                separated = true;
            }
        }
    }
    assert!(
        separated,
        "no cell produced the f32-only verdict; the construction no longer separates the widths"
    );
}
