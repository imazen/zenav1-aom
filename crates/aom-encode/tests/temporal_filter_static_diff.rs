//! Differential harness for the FILE-STATIC half of
//! `av1/encoder/temporal_filter.c`. **Tier 1c** — `shim/tf_static_shim.c`
//! compiles libaom's own temporal_filter.c verbatim (`-O3 -DNDEBUG
//! -ffp-contract=off`, its Release flags) and wraps the statics, which have no
//! exported address. Read that shim's header for the argument.
//!
//! | test | C oracle | tier |
//! |---|---|---|
//! | `tf_static_shim_tu_matches_archive` | the TU's copies vs libaom.a's | 1 vs 1c |
//! | `determine_block_partition_matches_c` | `tf_determine_block_partition` (:465) | 1c |
//! | `apply_temporal_filter_self_matches_c` | `tf_apply_temporal_filter_self` (:641) | 1c |
//! | `normalize_filtered_frame_matches_c` | `tf_normalize_filtered_frame` (:995) | 1c |
//! | `is_frame_high_bitdepth_matches_c` | `is_frame_high_bitdepth` (:520) | 1c |
//! | `check_show_filtered_frame_matches_c` | `av1_check_show_filtered_frame` (:1591) | **1** |
//! | `is_temporal_filter_on_matches_c` | `av1_is_temporal_filter_on` (:1654) | **1** |
//!
//! `tf_static_shim_tu_matches_archive` is the load-bearing one: without it,
//! "the tier-1c TU behaves like the archive" would be an assumption rather
//! than a measurement, and the whole file would drop to tier 4.
//!
//! # What bounds the generators
//! * `subblock_mses` / `midblock_mses` / `block_mse` — per-pixel squared
//!   errors out of `subblock_motion_search`, so `[0, (2^bd - 1)^2]`. The
//!   partition test additionally sweeps CLUSTERED sets (all four quadrant MSEs
//!   near each other), because the `max - min < 48` and `< 24` guards are
//!   unreachable from an unclustered uniform draw and the no-split arm would
//!   never fire.
//! * `count` for the normalizer — every block has already had
//!   `tf_apply_temporal_filter_self` add `TF_WEIGHT_SCALE = 1000`, so the
//!   count is at least 1000 and `OD_DIVU`'s `d == 0` case is unreachable. The
//!   sweep spans 1000 up past `OD_DIVU_DMAX = 1024`, which is where C switches
//!   from its reciprocal-multiply table to a real divide — the port uses a
//!   plain divide on both sides of that boundary and this is what proves it.

use aom_encode::temporal_filter::{
    NUM_16X16, TfPlane, TfPlaneMut, check_show_filtered_frame, is_temporal_filter_on,
    tf_apply_temporal_filter_self, tf_determine_block_partition, tf_normalize_filtered_frame,
};
use aom_sys_ref::{
    TfRefParams, TfRefPlane, ref_tf_estimate_noise_lowbd, ref_tf_is_temporal_filter_on,
    ref_tfs_apply_temporal_filter_self_highbd, ref_tfs_apply_temporal_filter_self_lowbd,
    ref_tfs_check_show_archive, ref_tfs_check_show_filtered_frame,
    ref_tfs_determine_block_partition, ref_tfs_is_frame_high_bitdepth,
    ref_tfs_normalize_filtered_frame_highbd, ref_tfs_normalize_filtered_frame_lowbd,
    ref_tfs_tu_estimate_noise, ref_tfs_tu_is_temporal_filter_on,
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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32) as i32
    }
}

/// `BLOCK_64X64` — `TF_BLOCK_SIZE`.
const BLOCK_64X64: i32 = 12;

fn params(bd: i32, num_planes: i32, ss: (i32, i32), mb_row: i32, mb_col: i32) -> TfRefParams {
    TfRefParams {
        block_size: BLOCK_64X64,
        mb_row,
        mb_col,
        num_planes,
        subsampling_x: [0, ss.0, ss.0],
        subsampling_y: [0, ss.1, ss.1],
        bd,
        q_factor: 0,
        filter_strength: 0,
        tf_wgt_calc_lvl: 0,
    }
}

// ---------------------------------------------------------------------------
// The gate that makes tier 1c mean something.
// ---------------------------------------------------------------------------

#[test]
fn tf_static_shim_tu_matches_archive() {
    // If the second compilation of temporal_filter.c ever stopped agreeing
    // with the copy inside libaom.a — a flag drift, a config drift — every
    // tier-1c result in this file would silently be measuring the wrong
    // binary. These two comparisons are what rule that out.
    let mut rng = Rng::new(0xC0DE_1C1C);
    for &(w, h) in &[(8usize, 8usize), (33, 21), (64, 64)] {
        let stride = w + 5;
        for _ in 0..4 {
            let plane: Vec<u8> = (0..h * stride).map(|_| rng.below(256) as u8).collect();
            for &thr in &[8i32, 50, 4096] {
                let tu = ref_tfs_tu_estimate_noise(&plane, h, w, stride, thr);
                let archive = ref_tf_estimate_noise_lowbd(&plane, h, w, stride, thr);
                assert_eq!(
                    tu.to_bits(),
                    archive.to_bits(),
                    "tier-1c TU disagrees with libaom.a at {w}x{h} thr {thr}"
                );
            }
        }
    }
    for &frames in &[0i32, 1, 7] {
        for &lag in &[0i32, 1, 2, 35] {
            assert_eq!(
                ref_tfs_tu_is_temporal_filter_on(frames, lag),
                ref_tf_is_temporal_filter_on(frames, lag),
                "tier-1c TU disagrees with libaom.a at ({frames}, {lag})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// tf_determine_block_partition.
// ---------------------------------------------------------------------------

#[test]
fn determine_block_partition_matches_c() {
    let mut rng = Rng::new(0x9A17_2B3C);
    let (mut split, mut merged) = (0usize, 0usize);
    for max_mse in [255u32 * 255, 4095 * 4095] {
        for trial in 0..600 {
            // `spread` controls how clustered the four sub-MSEs of a quadrant
            // are. C's no-split guards need `max - min < 48` (and `< 24`), so
            // an unclustered draw can never take that arm.
            let spread = match trial % 4 {
                0 => 4u32,
                1 => 20,
                2 => 60,
                _ => max_mse.min(1 << 16),
            };
            let mut sub = [(0i16, 0i16); NUM_16X16];
            let mut mses = [0i32; NUM_16X16];
            for q in 0..4 {
                let centre = rng.below(max_mse.saturating_sub(spread) + 1);
                for k in 0..4 {
                    mses[q * 4 + k] = (centre + rng.below(spread + 1)) as i32;
                }
            }
            for m in sub.iter_mut() {
                *m = (rng.range(-512, 512) as i16, rng.range(-512, 512) as i16);
            }
            let mut mid = [(0i16, 0i16); 4];
            for m in mid.iter_mut() {
                *m = (rng.range(-512, 512) as i16, rng.range(-512, 512) as i16);
            }
            // Mid-block MSEs near the quadrant sums keep both no-split guards live.
            let mut mid_mses = [0i32; 4];
            for (q, mm) in mid_mses.iter_mut().enumerate() {
                let sum: i64 = mses[q * 4..q * 4 + 4].iter().map(|&m| i64::from(m)).sum();
                let target = (sum * 4 / 15).max(0) as u32;
                *mm = rng.range(
                    (target as i32 - 32).max(0),
                    (target as i32).saturating_add(32),
                );
            }
            let block_mv = (rng.range(-512, 512) as i16, rng.range(-512, 512) as i16);
            let total: i64 = mses.iter().map(|&m| i64::from(m)).sum();
            // C's whole-block no-split guard is `block_mse * 15 <= sum`, so the
            // decision boundary is at `sum / 15`. Draw across it: a range that
            // only reaches `sum / 15` merges every time and the 16x16-split arm
            // is never observed.
            let block_mse = rng.range(0, ((total / 5).max(1).min(i64::from(i32::MAX)) as i32));

            let (mut sub_c, mut mses_c) = (sub, mses);
            let (mut sub_r, mut mses_r) = (sub, mses);
            ref_tfs_determine_block_partition(
                block_mv,
                block_mse,
                &mid,
                &mid_mses,
                &mut sub_c,
                &mut mses_c,
            );
            tf_determine_block_partition(
                block_mv,
                block_mse,
                &mid,
                &mid_mses,
                &mut sub_r,
                &mut mses_r,
            );
            assert_eq!(mses_r, mses_c, "mses, trial {trial} max {max_mse}");
            assert_eq!(sub_r, sub_c, "mvs, trial {trial} max {max_mse}");
            if sub_c == sub {
                split += 1;
            } else {
                merged += 1;
            }
        }
    }
    // Non-vacuity: both arms of the decision must have fired.
    assert!(split > 20, "only {split} draws kept the 16x16 split");
    assert!(
        merged > 20,
        "only {merged} draws merged — the no-split guards never fired"
    );
}

// ---------------------------------------------------------------------------
// tf_apply_temporal_filter_self.
// ---------------------------------------------------------------------------

#[test]
fn apply_temporal_filter_self_matches_c() {
    let mut rng = Rng::new(0x5E1F_0001);
    for &(bd, max) in &[(8i32, 255u32), (10, 1023), (12, 4095)] {
        for &num_planes in &[1i32, 3] {
            for &ss in &[(1i32, 1i32), (1, 0), (0, 0)] {
                let (yw, yh) = (128usize, 128usize);
                let (uw, uh) = (yw >> ss.0, yh >> ss.1);
                let (ys, us) = (yw + 12, uw + 6);
                let y: Vec<u16> = (0..yh * ys).map(|_| rng.below(max + 1) as u16).collect();
                let u: Vec<u16> = (0..uh * us).map(|_| rng.below(max + 1) as u16).collect();
                let v: Vec<u16> = (0..uh * us).map(|_| rng.below(max + 1) as u16).collect();
                let mut pels = 64 * 64;
                if num_planes == 3 {
                    pels += 2 * ((64 >> ss.1) * (64 >> ss.0));
                }
                let (mb_row, mb_col) = (rng.below(2) as i32, rng.below(2) as i32);
                let p = params(bd, num_planes, ss, mb_row, mb_col);
                let ssx = [0u32, ss.0 as u32, ss.0 as u32];
                let ssy = [0u32, ss.1 as u32, ss.1 as u32];

                // Pre-seed both accumulators identically so the ACCUMULATE
                // behaviour, not just the write, is compared.
                let seed_a: Vec<u32> = (0..pels).map(|_| rng.below(1 << 20)).collect();
                let seed_c: Vec<u16> = (0..pels).map(|_| rng.below(2000) as u16).collect();

                let (mut accum_c, mut count_c) = (seed_a.clone(), seed_c.clone());
                let (mut accum_r, mut count_r) = (seed_a.clone(), seed_c.clone());

                if bd == 8 {
                    let y8: Vec<u8> = y.iter().map(|&x| x as u8).collect();
                    let u8_: Vec<u8> = u.iter().map(|&x| x as u8).collect();
                    let v8: Vec<u8> = v.iter().map(|&x| x as u8).collect();
                    ref_tfs_apply_temporal_filter_self_lowbd(
                        [
                            TfRefPlane {
                                data: &y8,
                                stride: ys,
                                crop_width: yw,
                                crop_height: yh,
                            },
                            TfRefPlane {
                                data: &u8_,
                                stride: us,
                                crop_width: uw,
                                crop_height: uh,
                            },
                            TfRefPlane {
                                data: &v8,
                                stride: us,
                                crop_width: uw,
                                crop_height: uh,
                            },
                        ],
                        &p,
                        &mut accum_c,
                        &mut count_c,
                    );
                    let planes = [
                        TfPlane {
                            data: &y8[..],
                            stride: ys,
                            crop_width: yw,
                            crop_height: yh,
                        },
                        TfPlane {
                            data: &u8_[..],
                            stride: us,
                            crop_width: uw,
                            crop_height: uh,
                        },
                        TfPlane {
                            data: &v8[..],
                            stride: us,
                            crop_width: uw,
                            crop_height: uh,
                        },
                    ];
                    tf_apply_temporal_filter_self(
                        &planes[..num_planes as usize],
                        64,
                        64,
                        mb_row as usize,
                        mb_col as usize,
                        &ssx,
                        &ssy,
                        &mut accum_r,
                        &mut count_r,
                    );
                } else {
                    ref_tfs_apply_temporal_filter_self_highbd(
                        [
                            TfRefPlane {
                                data: &y,
                                stride: ys,
                                crop_width: yw,
                                crop_height: yh,
                            },
                            TfRefPlane {
                                data: &u,
                                stride: us,
                                crop_width: uw,
                                crop_height: uh,
                            },
                            TfRefPlane {
                                data: &v,
                                stride: us,
                                crop_width: uw,
                                crop_height: uh,
                            },
                        ],
                        &p,
                        &mut accum_c,
                        &mut count_c,
                    );
                    let planes = [
                        TfPlane {
                            data: &y[..],
                            stride: ys,
                            crop_width: yw,
                            crop_height: yh,
                        },
                        TfPlane {
                            data: &u[..],
                            stride: us,
                            crop_width: uw,
                            crop_height: uh,
                        },
                        TfPlane {
                            data: &v[..],
                            stride: us,
                            crop_width: uw,
                            crop_height: uh,
                        },
                    ];
                    tf_apply_temporal_filter_self(
                        &planes[..num_planes as usize],
                        64,
                        64,
                        mb_row as usize,
                        mb_col as usize,
                        &ssx,
                        &ssy,
                        &mut accum_r,
                        &mut count_r,
                    );
                }
                assert_eq!(accum_r, accum_c, "accum bd{bd} np{num_planes} ss{ss:?}");
                assert_eq!(count_r, count_c, "count bd{bd} np{num_planes} ss{ss:?}");
                assert_ne!(accum_r, seed_a, "the filter wrote nothing");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// tf_normalize_filtered_frame.
// ---------------------------------------------------------------------------

#[test]
fn normalize_filtered_frame_matches_c() {
    let mut rng = Rng::new(0x0D1F_5555);
    for &(bd, max) in &[(8i32, 255u32), (10, 1023), (12, 4095)] {
        for &num_planes in &[1i32, 3] {
            for &ss in &[(1i32, 1i32), (0, 0)] {
                let (yw, yh) = (128usize, 128usize);
                let (uw, uh) = (yw >> ss.0, yh >> ss.1);
                let (ys, us) = (yw + 12, uw + 6);
                let mut pels = 64 * 64;
                if num_planes == 3 {
                    pels += 2 * ((64 >> ss.1) * (64 >> ss.0));
                }
                let (mb_row, mb_col) = (rng.below(2) as i32, rng.below(2) as i32);
                let p = params(bd, num_planes, ss, mb_row, mb_col);
                let ssx = [0u32, ss.0 as u32, ss.0 as u32];
                let ssy = [0u32, ss.1 as u32, ss.1 as u32];

                // count spans OD_DIVU_DMAX (1024): C uses its reciprocal table
                // below it and a real divide at or above it.
                let count: Vec<u16> = (0..pels)
                    .map(|i| match i % 4 {
                        0 => 1000,
                        1 => 1023,
                        2 => 1024,
                        _ => 1000 + rng.below(30_000) as u16,
                    })
                    .collect();
                // A RESIDUAL is essential: with `accum` an exact multiple of
                // `count`, `rounding = count >> 1` can never change the
                // quotient and the rounding term is untested. The residual is
                // drawn over the full `[0, count)` so it straddles `count / 2`.
                let accum: Vec<u32> = count
                    .iter()
                    .map(|&c| u32::from(c) * rng.below(max + 1) + rng.below(u32::from(c)))
                    .collect();

                if bd == 8 {
                    let base: Vec<u8> = (0..yh * ys).map(|_| rng.below(256) as u8).collect();
                    let baseu: Vec<u8> = (0..uh * us).map(|_| rng.below(256) as u8).collect();
                    let (mut yc, mut uc, mut vc) = (base.clone(), baseu.clone(), baseu.clone());
                    let (mut yr, mut ur, mut vr) = (base.clone(), baseu.clone(), baseu.clone());
                    ref_tfs_normalize_filtered_frame_lowbd(
                        [&mut yc, &mut uc, &mut vc],
                        [ys, us],
                        &p,
                        &accum,
                        &count,
                    );
                    let mut planes = [
                        TfPlaneMut {
                            data: &mut yr[..],
                            stride: ys,
                        },
                        TfPlaneMut {
                            data: &mut ur[..],
                            stride: us,
                        },
                        TfPlaneMut {
                            data: &mut vr[..],
                            stride: us,
                        },
                    ];
                    tf_normalize_filtered_frame(
                        &mut planes[..num_planes as usize],
                        64,
                        64,
                        mb_row as usize,
                        mb_col as usize,
                        &ssx,
                        &ssy,
                        &accum,
                        &count,
                    );
                    assert_eq!(yr, yc, "y bd{bd} np{num_planes} ss{ss:?}");
                    assert_eq!(ur, uc, "u bd{bd} np{num_planes} ss{ss:?}");
                    assert_eq!(vr, vc, "v bd{bd} np{num_planes} ss{ss:?}");
                    assert_ne!(yr, base, "the normalizer wrote nothing");
                } else {
                    let base: Vec<u16> = (0..yh * ys).map(|_| rng.below(max + 1) as u16).collect();
                    let baseu: Vec<u16> = (0..uh * us).map(|_| rng.below(max + 1) as u16).collect();
                    let (mut yc, mut uc, mut vc) = (base.clone(), baseu.clone(), baseu.clone());
                    let (mut yr, mut ur, mut vr) = (base.clone(), baseu.clone(), baseu.clone());
                    ref_tfs_normalize_filtered_frame_highbd(
                        [&mut yc, &mut uc, &mut vc],
                        [ys, us],
                        &p,
                        &accum,
                        &count,
                    );
                    let mut planes = [
                        TfPlaneMut {
                            data: &mut yr[..],
                            stride: ys,
                        },
                        TfPlaneMut {
                            data: &mut ur[..],
                            stride: us,
                        },
                        TfPlaneMut {
                            data: &mut vr[..],
                            stride: us,
                        },
                    ];
                    tf_normalize_filtered_frame(
                        &mut planes[..num_planes as usize],
                        64,
                        64,
                        mb_row as usize,
                        mb_col as usize,
                        &ssx,
                        &ssy,
                        &accum,
                        &count,
                    );
                    assert_eq!(yr, yc, "y bd{bd} np{num_planes} ss{ss:?}");
                    assert_eq!(ur, uc, "u bd{bd} np{num_planes} ss{ss:?}");
                    assert_eq!(vr, vc, "v bd{bd} np{num_planes} ss{ss:?}");
                    assert_ne!(yr, base, "the normalizer wrote nothing");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The two exported predicates.
// ---------------------------------------------------------------------------

#[test]
fn is_frame_high_bitdepth_matches_c() {
    // YV12_FLAG_HIGHBITDEPTH is 8; every other flag bit must NOT select it.
    for flags in 0..64i32 {
        let want = ref_tfs_is_frame_high_bitdepth(flags);
        assert_eq!(want, flags & 8 != 0, "flags {flags}");
    }
    // And the port's type-level encoding agrees with that at both depths.
    assert!(!<u8 as aom_encode::temporal_filter::TfPixel>::HIGH_BITDEPTH);
    assert!(<u16 as aom_encode::temporal_filter::TfPixel>::HIGH_BITDEPTH);
}

#[test]
fn is_temporal_filter_on_matches_c() {
    for arnr in [-1i32, 0, 1, 2, 7, 15] {
        for lag in [-1i32, 0, 1, 2, 19, 35] {
            let want = ref_tf_is_temporal_filter_on(arnr, lag);
            assert_eq!(is_temporal_filter_on(arnr, lag), want, "({arnr}, {lag})");
        }
    }
}

#[test]
fn check_show_filtered_frame_matches_c() {
    let mut rng = Rng::new(0xC5F0_9911);
    let (mut yes, mut no) = (0usize, 0usize);
    for &bd in &[8i32, 10, 12] {
        for _ in 0..500 {
            // Frame sizes from below one TF block to 4K.
            let w = rng.range(16, 3840);
            let h = rng.range(16, 2160);
            let mb_rows = (h + 63) / 64;
            let mb_cols = (w + 63) / 64;
            let num_mbs = i64::from((mb_rows * mb_cols).max(1));
            // sum and sse are per-block accumulations of a squared pixel
            // difference, so both are non-negative and sse >= sum^2 / num_mbs
            // is what makes the variance non-negative. Draw sum first, then
            // sse at or above the value that makes std real.
            let mean = i64::from(rng.below(1 << 14));
            let sum = mean * num_mbs;
            let sse = sum / num_mbs * mean * num_mbs + i64::from(rng.below(1 << 20)) * num_mbs;
            let q_index = rng.range(0, 255);
            for &(overlay, second) in &[(true, false), (false, false), (true, true)] {
                let want = ref_tfs_check_show_archive(w, h, sum, sse, q_index, bd, overlay, second);
                let tu =
                    ref_tfs_check_show_filtered_frame(w, h, sum, sse, q_index, bd, overlay, second);
                assert_eq!(want, tu, "tier-1c TU vs archive at {w}x{h}");
                let got =
                    check_show_filtered_frame(w, h, sum, sse, q_index, bd as u8, overlay, second);
                assert_eq!(
                    got, want,
                    "{w}x{h} sum {sum} sse {sse} q {q_index} bd {bd} overlay {overlay} second {second}"
                );
                if want {
                    yes += 1;
                } else {
                    no += 1;
                }
            }
        }
    }
    assert!(
        yes > 100,
        "only {yes} shows — the accept arm is under-exercised"
    );
    assert!(
        no > 100,
        "only {no} rejects — the reject arm is under-exercised"
    );
}

#[test]
fn check_show_filtered_frame_separates_both_boundaries() {
    // The random sweep above never lands ON either comparison boundary, so
    // `mean < threshold` vs `mean <= threshold`, and an f32-vs-f64 spelling of
    // `std`, both survive it. These two constructions separate them.
    //
    // A 64x64 frame is exactly one TF block, so `num_mbs == 1` and
    // `mean == sum` / `std == sqrt(sse - sum^2)` with no division rounding.

    // (a) mean EXACTLY equal to the threshold. Sweep every q for one whose
    //     `0.7f32 * ac_q_step^2` is integral, so an integer `sum` can equal it.
    let mut hit_equal = false;
    for q in 0..=255i32 {
        let ac = f32::from(aom_dsp::quant::av1_ac_quant_qtx(q, 0, 8));
        let threshold = 0.7f32 * ac * ac;
        if threshold.fract() != 0.0 || threshold <= 0.0 || threshold > 1e9 {
            continue;
        }
        let sum = threshold as i64;
        // sse == sum^2 gives std == 0, so the second guard is satisfied and the
        // whole result turns on the first one alone.
        let sse = sum * sum;
        let want = ref_tfs_check_show_archive(64, 64, sum, sse, q, 8, true, false);
        let got = check_show_filtered_frame(64, 64, sum, sse, q, 8, true, false);
        assert_eq!(got, want, "mean == threshold at q {q} (sum {sum})");
        assert!(
            !want,
            "C uses a STRICT `<`, so mean == threshold must reject"
        );
        hit_equal = true;
    }
    assert!(
        hit_equal,
        "no q produced an integral threshold - construction failed"
    );

    // (b) std within a few units of `mean * 1.2`, which is where an f64
    //     spelling of the variance would diverge from C's f32 one.
    let mut straddled = (false, false);
    for q in [40i32, 120, 200] {
        for sum in [100i64, 1000, 12_345] {
            let t = 1.2f64 * sum as f64;
            let target = (t * t) as i64;
            for delta in -3i64..=3 {
                let sse = sum * sum + target + delta;
                let want = ref_tfs_check_show_archive(64, 64, sum, sse, q, 8, true, false);
                let got = check_show_filtered_frame(64, 64, sum, sse, q, 8, true, false);
                assert_eq!(got, want, "std boundary q {q} sum {sum} delta {delta}");
                if want {
                    straddled.0 = true;
                } else {
                    straddled.1 = true;
                }
            }
        }
    }
    assert!(
        straddled.0 && straddled.1,
        "the std boundary sweep stayed on one side: {straddled:?}"
    );

    // (c) Cells where computing the variance in f64 instead of C's f32 flips
    //     the `std < mean * 1.2` verdict. They exist only at large frames and
    //     large per-block SSEs, which is why nothing above reaches them, and
    //     they are the only thing that distinguishes C's `(float)sse / num_mbs`
    //     from a widened spelling. Found by searching the reachable
    //     (sum, sse, num_mbs) space for a sign flip; `sse` here is
    //     `sum over blocks of sse_i^2` and `sum` is `sum over blocks of sse_i`,
    //     so the implied variance is non-negative in every one, as it must be.
    //     Values are per (mb_rows, mb_cols, sum, sse).
    const F32_SPELLING_DISCRIMINATORS: &[(i32, i32, i64, i64)] = &[
        (30, 34, 1_745_771_631, 7_290_621_448_721_299),
        (6, 5, 2_428_470, 479_660_611_992),
        (14, 35, 232_727_861, 269_705_927_575_058),
        (33, 2, 87_666_121, 284_124_989_249_290),
        (18, 22, 601_286_478, 2_227_703_944_999_126),
        (19, 23, 80_153_209, 35_871_555_785_476),
    ];
    let mut reached_std = false;
    for &(mb_rows, mb_cols, sum, sse) in F32_SPELLING_DISCRIMINATORS {
        let (w, h) = (mb_cols * 64, mb_rows * 64);
        for q in [0i32, 60, 255] {
            let want = ref_tfs_check_show_archive(w, h, sum, sse, q, 8, true, false);
            let got = check_show_filtered_frame(w, h, sum, sse, q, 8, true, false);
            assert_eq!(
                got, want,
                "f32-spelling cell {w}x{h} sum {sum} sse {sse} q {q}"
            );
        }
    }
}
