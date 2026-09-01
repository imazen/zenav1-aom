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

// ---------------------------------------------------------------------------
// The tx-size / subpel-precision / MV-bias cluster.
// ---------------------------------------------------------------------------

use aom_encode::nonrd_inter::{
    MAX_MODES as NRD_MAX_MODES, SubpelForceStop, SubpelSelectCtx, TxSizeCtx, calculate_tx_size,
    is_same_gf_and_last_scale, newmv_diff_bias, pack_mv, set_force_skip_flag, subpel_select,
    update_thresh_freq_fact, use_aggressive_subpel_search_method,
};
use aom_sys_ref::{
    ref_nrp_calculate_tx_size, ref_nrp_newmv_diff_bias, ref_nrp_set_force_skip_flag,
    ref_nrp_subpel_select, ref_nrp_thresh_freq_fact_dims, ref_nrp_update_thresh_freq_fact,
    ref_nrp_use_aggressive_subpel_search_method,
};

/// The block sizes these helpers are called at, plus the extremes.
const BSIZES: [usize; 8] = [0, 3, 6, 7, 9, 11, 12, 15];

/// `TX_MODE_SELECT` as C's raw `TX_MODE`.
const TX_MODE_SELECT_RAW: i32 = 2;

#[test]
fn subpel_select_matches_c() {
    let mut rng = Rng::new(0x5B_9001);
    let mut seen = [0usize; 4];
    for _ in 0..8000 {
        let ctx = SubpelSelectCtx {
            // avg_frame_low_motion straddles the (0, 40) window.
            avg_frame_low_motion: match rng.below(4) {
                0 => 0,
                1 => rng.below(40) as i32,
                2 => 40,
                _ => 40 + rng.below(60) as i32,
            },
            reduce_mv_pel_precision_highmotion: rng.below(5) as i32,
            reduce_mv_pel_precision_lowcomplex: rng.below(4) as i32,
            subpel_force_stop: SubpelForceStop::from_i32(rng.below(4) as i32).unwrap(),
            // Straddle the 320x240 low-resolution test.
            frame_width: [176i32, 320, 321, 1920][rng.below(4) as usize],
            frame_height: [144i32, 240, 241, 1080][rng.below(4) as usize],
            qindex: rng.below(256) as i32,
            source_sad_nonrd: [
                SourceSad::Zero,
                SourceSad::VeryLow,
                SourceSad::Low,
                SourceSad::Med,
                SourceSad::High,
            ][rng.below(5) as usize],
            // Straddle the 500 and 5000 variance thresholds.
            source_variance: match rng.below(4) {
                0 => rng.below(500) as i32,
                1 => 500 + rng.below(4500) as i32,
                2 => 5000,
                _ => 5000 + rng.below(20000) as i32,
            },
        };
        let bsize = BSIZES[rng.below(BSIZES.len() as u32) as usize];
        // MVs around the thresholds (2, 4, 6, 8, 10, 12 and their doubles).
        let draw_mv = |rng: &mut Rng| -> i16 {
            let m = rng.below(30) as i16 - 4;
            if rng.below(2) == 0 { m } else { -m }
        };
        let mv = (draw_mv(&mut rng), draw_mv(&mut rng));
        let ref_mv = if rng.below(2) == 0 {
            (0i16, 0i16)
        } else {
            (draw_mv(&mut rng), draw_mv(&mut rng))
        };
        let start_mv = if rng.below(2) == 0 {
            (0i16, 0i16)
        } else {
            (draw_mv(&mut rng), draw_mv(&mut rng))
        };
        let fpw = rng.below(2) == 0;

        let want = ref_nrp_subpel_select(
            ctx.avg_frame_low_motion,
            ctx.reduce_mv_pel_precision_highmotion,
            ctx.reduce_mv_pel_precision_lowcomplex,
            ctx.subpel_force_stop.to_i32(),
            (ctx.frame_width, ctx.frame_height),
            bsize as i32,
            mv,
            ref_mv,
            start_mv,
            ctx.qindex,
            ctx.source_sad_nonrd as i32,
            ctx.source_variance,
            fpw,
        );
        let got = subpel_select(&ctx, bsize, mv, ref_mv, start_mv, fpw);
        assert_eq!(
            got.to_i32(),
            want,
            "bsize {bsize} mv {mv:?} ref {ref_mv:?} start {start_mv:?} ctx {ctx:?} fpw {fpw}"
        );
        seen[want as usize] += 1;
    }
    // Every SUBPEL_FORCE_STOP must be reachable, or a whole arm is untested.
    for (i, &n) in seen.iter().enumerate() {
        assert!(n > 50, "SUBPEL_FORCE_STOP {i} was returned only {n} times");
    }
}

#[test]
fn use_aggressive_subpel_search_method_matches_c() {
    for qindex in [0i32, 63, 64, 127, 128, 191, 192, 255] {
        for sad in [
            SourceSad::Zero,
            SourceSad::VeryLow,
            SourceSad::Low,
            SourceSad::Med,
            SourceSad::High,
        ] {
            for &var in &[0i32, 99, 100, 101, 10000] {
                for &adaptive in &[false, true] {
                    for &fpw in &[false, true] {
                        assert_eq!(
                            use_aggressive_subpel_search_method(qindex, sad, var, adaptive, fpw),
                            ref_nrp_use_aggressive_subpel_search_method(
                                qindex, sad as i32, var, adaptive, fpw
                            ),
                            "q {qindex} sad {sad:?} var {var} adaptive {adaptive} fpw {fpw}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn set_force_skip_flag_matches_c() {
    let mut rng = Rng::new(0x5F5F_9002);
    let (mut set, mut clear) = (0usize, 0usize);
    for _ in 0..6000 {
        // dequant_ac is a real AC quantizer step; bd picks the shift.
        let bd = [8i32, 10, 12][rng.below(3) as usize];
        let dequant_ac = 4 + rng.below(2000) as i32;
        let qstep = dequant_ac >> (bd - 5);
        let qstep_sq = (qstep * qstep) as u32;
        // sse and source_variance straddle qstep_sq, which is the whole test.
        let sse = match rng.below(3) {
            0 => rng.below(qstep_sq.max(1)),
            1 => qstep_sq,
            _ => qstep_sq + rng.below(1 << 16),
        };
        let source_variance = match rng.below(3) {
            0 => rng.below(qstep_sq.max(1)) as i32,
            1 => qstep_sq as i32,
            _ => (qstep_sq + rng.below(1 << 16)) as i32,
        };
        let ctx = TxSizeCtx {
            tx_mode_search_type: rng.below(3) as i32,
            tx_size_level_based_on_qstep: rng.below(4) as i32,
            aq_mode: 0,
            segment_id: 0,
            qindex: 0,
            dequant_ac,
            bd: bd as u32,
            source_variance,
            color_sensitivity_u: rng.below(2) as u8,
            color_sensitivity_v: rng.below(2) as u8,
        };
        for &force_in in &[false, true] {
            let want = ref_nrp_set_force_skip_flag(
                ctx.tx_mode_search_type,
                ctx.tx_size_level_based_on_qstep,
                ctx.dequant_ac,
                bd,
                sse,
                ctx.source_variance,
                (
                    i32::from(ctx.color_sensitivity_u),
                    i32::from(ctx.color_sensitivity_v),
                ),
                force_in,
            );
            let got = set_force_skip_flag(&ctx, sse, force_in);
            assert_eq!(got, want, "ctx {ctx:?} sse {sse} force_in {force_in}");
            if want {
                set += 1;
            } else {
                clear += 1;
            }
        }
    }
    assert!(
        set > 200 && clear > 200,
        "one arm never fired: {set}/{clear}"
    );
}

#[test]
fn calculate_tx_size_matches_c() {
    let mut rng = Rng::new(0x7C_9003);
    let mut seen = [0usize; 5];
    let mut forced = 0usize;
    for _ in 0..12000 {
        let bd = [8i32, 10, 12][rng.below(3) as usize];
        let dequant_ac = 4 + rng.below(2000) as i32;
        let qstep = dequant_ac >> (bd - 5);
        let qstep_sq = (qstep * qstep) as u32;
        // The force_skip write-back needs a five-way conjunction
        // (TX_MODE_SELECT, level >= 2, sse and source_variance both under
        // qstep_sq, and neither chroma plane sensitive). A uniform draw over
        // each field reaches it a handful of times in twelve thousand, so the
        // three cheap terms are biased toward it and the two value terms are
        // left straddling.
        let ctx = TxSizeCtx {
            tx_mode_search_type: if rng.below(4) == 0 {
                rng.below(3) as i32
            } else {
                TX_MODE_SELECT_RAW
            },
            tx_size_level_based_on_qstep: if rng.below(4) == 0 {
                rng.below(4) as i32
            } else {
                2 + rng.below(2) as i32
            },
            // Both the CYCLIC_REFRESH_AQ arm and the others.
            aq_mode: rng.below(4) as i32,
            // Both boosted segment ids and the ones that are not.
            segment_id: rng.below(4) as i32,
            qindex: rng.below(256) as i32,
            dequant_ac,
            bd: bd as u32,
            source_variance: match rng.below(3) {
                0 => rng.below(qstep_sq.max(1)) as i32,
                _ => (qstep_sq + rng.below(1 << 16)) as i32,
            },
            color_sensitivity_u: u8::from(rng.below(4) == 0),
            color_sensitivity_v: u8::from(rng.below(4) == 0),
        };
        // var and sse straddle both `sse > (var * multiplier) >> 2` and
        // `var < var_thresh` (= 2 * qstep_sq).
        let var = match rng.below(4) {
            0 => 0u32,
            1 => rng.below(2 * qstep_sq.max(1)),
            2 => 2 * qstep_sq,
            _ => rng.below(1 << 20),
        };
        let sse = match rng.below(4) {
            0 => (var * 2) >> 2,
            1 => ((var * 8) >> 2) + 1,
            2 => rng.below(qstep_sq.max(1)),
            _ => rng.below(1 << 22),
        };
        let bsize = BSIZES[rng.below(BSIZES.len() as u32) as usize];
        for &force_in in &[false, true] {
            let want = ref_nrp_calculate_tx_size(
                ctx.tx_mode_search_type,
                ctx.tx_size_level_based_on_qstep,
                ctx.aq_mode,
                ctx.segment_id,
                bsize as i32,
                ctx.qindex,
                ctx.dequant_ac,
                bd,
                var,
                sse,
                ctx.source_variance,
                (
                    i32::from(ctx.color_sensitivity_u),
                    i32::from(ctx.color_sensitivity_v),
                ),
                force_in,
            );
            let got = calculate_tx_size(&ctx, bsize, var, sse, force_in);
            assert_eq!(
                (got.0 as i32, got.1),
                want,
                "ctx {ctx:?} bsize {bsize} var {var} sse {sse} force_in {force_in}"
            );
            seen[got.0] += 1;
            if got.1 && !force_in {
                forced += 1;
            }
        }
    }
    // TX_4X4, TX_8X8 and TX_16X16 must all be reachable; the final AOMMIN caps
    // at TX_16X16, so 3 and 4 never are.
    assert!(
        seen[0] > 100 && seen[1] > 100 && seen[2] > 100,
        "sizes seen: {seen:?}"
    );
    assert_eq!(
        seen[3], 0,
        "TX_32X32 must be unreachable through the final cap"
    );
    assert_eq!(
        seen[4], 0,
        "TX_64X64 must be unreachable through the final cap"
    );
    assert!(
        forced > 50,
        "the force_skip write-back never fired ({forced})"
    );
}

#[test]
fn newmv_diff_bias_matches_c() {
    let mut rng = Rng::new(0xB1A5_9004);
    let mut distinct = std::collections::BTreeSet::new();
    for _ in 0..12000 {
        let (mode, raw_mode) = if rng.below(2) == 0 {
            (PredMode::NewMv, 16i32)
        } else {
            RTC_MODE_SET[rng.below(4) as usize]
        };
        let rdcost = match rng.below(4) {
            0 => 0i64,
            1 => 1,
            _ => i64::from(rng.below(1 << 28)),
        };
        let bsize = BSIZES[rng.below(BSIZES.len() as u32) as usize];
        // MVs around the 16, 64 and 80 thresholds the three arms use.
        let draw = |rng: &mut Rng| -> i32 {
            let v = rng.below(200) as i32 - 100;
            if rng.below(4) == 0 { v * 2 } else { v }
        };
        let mv = (draw(&mut rng), draw(&mut rng));
        let speed = rng.below(11) as i32;
        // Straddle the 150 and 300 spatial-variance thresholds.
        let spatial_variance = match rng.below(4) {
            0 => rng.below(150),
            1 => rng.below(300),
            2 => 300,
            _ => rng.below(1 << 16),
        };
        let sad = [
            SourceSad::Zero,
            SourceSad::VeryLow,
            SourceSad::Low,
            SourceSad::Med,
            SourceSad::High,
        ][rng.below(5) as usize];
        // A neighbour that is absent, present with a valid MV, or present with
        // C's INVALID_MV -- all three take different paths.
        let draw_nb = |rng: &mut Rng| -> Option<(i16, i16)> {
            match rng.below(3) {
                0 => None,
                1 => Some((-32768i16, -32768i16)), // packs to INVALID_MV
                _ => Some((rng.below(400) as i16 - 200, rng.below(400) as i16 - 200)),
            }
        };
        let above = draw_nb(&mut rng);
        let left = draw_nb(&mut rng);

        let want = ref_nrp_newmv_diff_bias(
            raw_mode,
            rdcost,
            bsize as i32,
            mv,
            speed,
            spatial_variance,
            sad as i32,
            above.map(|(r, c)| pack_mv(r, c)),
            left.map(|(r, c)| pack_mv(r, c)),
        );
        let got = newmv_diff_bias(
            mode,
            rdcost,
            bsize,
            mv.0,
            mv.1,
            speed,
            spatial_variance,
            sad,
            above,
            left,
        );
        assert_eq!(
            got, want,
            "mode {raw_mode} rd {rdcost} bsize {bsize} mv {mv:?} speed {speed} var {spatial_variance} sad {sad:?} above {above:?} left {left:?}"
        );
        if rdcost > 1000 {
            // Which of the four multipliers fired.
            distinct.insert(if got == rdcost {
                0
            } else if got == rdcost << 2 {
                1
            } else if got == rdcost << 1 {
                2
            } else {
                3
            });
        }
    }
    assert_eq!(
        distinct.len(),
        4,
        "only {} of the four rdcost multipliers fired: {distinct:?}",
        distinct.len()
    );

    // Two boundaries the random sweep reaches only by accident, because each
    // needs a four- or five-way conjunction to be observable.
    //
    // (a) the FIRST arm's `|mv| > 16`, which needs bsize >= BLOCK_64X64, a
    //     non-High source SAD and spatial_variance < 300 as well.
    for &(mv_row, mv_col) in &[(16i32, 0i32), (17, 0), (0, 16), (0, 17), (-16, 0), (-17, 0)] {
        let want = ref_nrp_newmv_diff_bias(
            16,
            1_000_000,
            12,
            (mv_row, mv_col),
            0,
            299,
            SourceSad::Low as i32,
            None,
            None,
        );
        let got = newmv_diff_bias(
            PredMode::NewMv,
            1_000_000,
            12,
            mv_row,
            mv_col,
            0,
            299,
            SourceSad::Low,
            None,
            None,
        );
        assert_eq!(got, want, "first-arm mv boundary ({mv_row},{mv_col})");
    }

    // (b) the neighbour average's `+ 1` rounding, which is only observable
    //     when `above + left` is ODD and the resulting row_diff lands exactly
    //     on the +/-80 boundary. Constructed for both signs.
    for &(above_row, left_row, mv_row) in &[
        (81i32, 80i32, 0i32),
        (-81, -80, 0),
        (161, 160, 80),
        (-161, -160, -80),
    ] {
        let above = Some((above_row as i16, 0i16));
        let left = Some((left_row as i16, 0i16));
        let want = ref_nrp_newmv_diff_bias(
            16,
            1_000_000,
            9,
            (mv_row, 0),
            0,
            10_000,
            SourceSad::High as i32,
            above.map(|(r, c)| pack_mv(r, c)),
            left.map(|(r, c)| pack_mv(r, c)),
        );
        let got = newmv_diff_bias(
            PredMode::NewMv,
            1_000_000,
            9,
            mv_row,
            0,
            0,
            10_000,
            SourceSad::High,
            above,
            left,
        );
        assert_eq!(
            got, want,
            "neighbour-average rounding ({above_row},{left_row}) vs mv {mv_row}"
        );
    }
}

#[test]
fn update_thresh_freq_fact_matches_c() {
    let mut rng = Rng::new(0x7F_9005);
    let (nb, nm) = ref_nrp_thresh_freq_fact_dims();
    assert_eq!(nm, NRD_MAX_MODES, "MAX_MODES drifted");
    let (mut decayed, mut raised) = (0usize, 0usize);
    for _ in 0..2000 {
        let flat: Vec<i32> = (0..nb * nm).map(|_| rng.below(1 << 12) as i32).collect();
        let bsize = BSIZES[rng.below(BSIZES.len() as u32) as usize];
        let ref_frame = 1 + rng.below(7) as usize;
        let mode_offset = rng.below(4) as usize;
        let best_mode_idx = rng.below(NRD_MAX_MODES as u32) as usize;
        // adaptive_rd_thresh caps the raised value; the encoder uses 0..4.
        let adaptive_rd_thresh = rng.below(5) as i32;

        let mut want = flat.clone();
        ref_nrp_update_thresh_freq_fact(
            adaptive_rd_thresh,
            bsize as i32,
            ref_frame as i32,
            best_mode_idx as i32,
            // C takes a PREDICTION_MODE and applies mode_offset() itself;
            // NEARESTMV is 13 and mode_offset subtracts it.
            13 + mode_offset as i32,
            &mut want,
        );

        let mut got: Vec<[i32; NRD_MAX_MODES]> = flat
            .chunks_exact(nm)
            .map(|c| {
                let mut a = [0i32; NRD_MAX_MODES];
                a.copy_from_slice(c);
                a
            })
            .collect();
        update_thresh_freq_fact(
            adaptive_rd_thresh,
            &mut got,
            bsize,
            ref_frame,
            best_mode_idx,
            mode_offset,
        );
        let got_flat: Vec<i32> = got.iter().flatten().copied().collect();
        assert_eq!(
            got_flat, want,
            "bsize {bsize} ref {ref_frame} best {best_mode_idx} mode_off {mode_offset} thresh {adaptive_rd_thresh}"
        );
        if MODE_IDX[ref_frame][mode_offset] == best_mode_idx {
            decayed += 1;
        } else {
            raised += 1;
        }
    }
    assert!(raised > 100, "the raise arm never fired");
    // The decay arm needs the best mode to BE this one, which a uniform draw
    // over 169 modes almost never produces -- so it is forced.
    let _ = decayed;
    for ref_frame in 1..8usize {
        for mode_offset in 0..4usize {
            let best_mode_idx = MODE_IDX[ref_frame][mode_offset];
            let flat: Vec<i32> = (0..nb * nm).map(|i| (i % 4096) as i32).collect();
            let mut want = flat.clone();
            ref_nrp_update_thresh_freq_fact(
                4,
                9,
                ref_frame as i32,
                best_mode_idx as i32,
                13 + mode_offset as i32,
                &mut want,
            );
            let mut got: Vec<[i32; NRD_MAX_MODES]> = flat
                .chunks_exact(nm)
                .map(|c| {
                    let mut a = [0i32; NRD_MAX_MODES];
                    a.copy_from_slice(c);
                    a
                })
                .collect();
            update_thresh_freq_fact(4, &mut got, 9, ref_frame, best_mode_idx, mode_offset);
            let got_flat: Vec<i32> = got.iter().flatten().copied().collect();
            assert_eq!(
                got_flat, want,
                "decay arm, ref {ref_frame} off {mode_offset}"
            );
            assert_ne!(got_flat, flat, "the decay arm changed nothing");
        }
    }
}

#[test]
fn is_same_gf_and_last_scale_compares_both_axes() {
    // A one-line predicate; the point of pinning it is that it compares the
    // SCALE FACTORS on both axes, not the frame sizes.
    assert!(is_same_gf_and_last_scale(
        1 << 14,
        1 << 14,
        1 << 14,
        1 << 14
    ));
    assert!(!is_same_gf_and_last_scale(
        1 << 14,
        1 << 14,
        1 << 13,
        1 << 14
    ));
    assert!(!is_same_gf_and_last_scale(
        1 << 14,
        1 << 14,
        1 << 14,
        1 << 13
    ));
    assert!(!is_same_gf_and_last_scale(0, 0, 1, 1));
}
