//! Differential harness for `aom_encode::temporal_filter` vs the REAL exported
//! C libaom v3.14.1. **Tier 1** — every oracle call reaches an exported symbol
//! out of `upstream/build/libaom.a`; `shim/tf_shim.c` only assembles the
//! `YV12_BUFFER_CONFIG` / `MACROBLOCKD` two of them take.
//!
//! | test | C oracle |
//! |---|---|
//! | `estimate_noise_lowbd_matches_c` | `av1_estimate_noise_from_single_plane_c` |
//! | `estimate_noise_highbd_matches_c` | `av1_highbd_estimate_noise_from_single_plane_c` |
//! | `estimate_noise_unreliable_sentinel_matches_c` | ditto, forced into the `count < 16` arm |
//! | `estimate_noise_level_matches_c` | `av1_estimate_noise_level` (both depths) |
//! | `apply_temporal_filter_matches_c` | `av1_apply_temporal_filter_c` |
//! | `highbd_apply_temporal_filter_matches_c` | `av1_highbd_apply_temporal_filter_c` |
//! | `apply_temporal_filter_accumulates_across_frames` | as above, driven the way `tf_do_filtering` drives it |
//!
//! # What bounds the generators (`DIFFERENTIAL_PLAYBOOK` lesson 5)
//! Each input is bounded by what the ENCODER can hand C, not by the C type:
//! * `noise_levels` — `av1_estimate_noise_level`'s output, so either `>= 0` or
//!   the `-1.0` "unreliable" sentinel. Both are swept. A negative value below
//!   `-2.5` would make `log(2 * n + 5)` a NaN in C; the encoder cannot produce
//!   one and the sweep does not either.
//! * `subblock_mses` — `DIVIDE_AND_ROUND(error, subblock_pels)` out of
//!   `subblock_motion_search` (:132), i.e. a PER-PIXEL squared error, so it is
//!   bounded by `(2^bd - 1)^2` and never by `int`. It is drawn LOG-UNIFORMLY
//!   over that range: a uniform draw is dominated by errors so large that
//!   `scaled_error` saturates at 7 and every weight comes out 0, which is a
//!   vacuous comparison (`DIFFERENTIAL_PLAYBOOK` lesson 6). The sweeps assert
//!   that both a zero and a non-zero weight actually occurred.
//! * `filter_strength` — `arnr_strength`, documented `[0, 6]`.
//! * `q_factor` — `av1_convert_qindex_to_q` output clamped by C's own comment
//!   ("Max q_factor is 255"), so `[1, 255]`.
//! * `subblock_mvs` — full-pel MVs out of `tf_motion_search`, bounded by the
//!   `MAX_FULL_PEL_VAL`-scaled search range; swept to +/-1024 in 1/8 pel.
//!
//! # Why the accumulate test exists
//! `accum` and `count` are ADDED to, not written. A port that assigned instead
//! would pass a single-frame comparison and desync the moment the filter ran
//! over more than one reference frame, which is every real invocation.

use aom_encode::temporal_filter::{
    NOISE_ESTIMATION_EDGE_THRESHOLD, NUM_16X16, TfFilterParams, TfPlane, apply_temporal_filter,
    estimate_noise_from_single_plane, estimate_noise_level,
    highbd_estimate_noise_from_single_plane,
};
use aom_sys_ref::{
    TfRefParams, TfRefPlane, ref_tf_apply_temporal_filter_highbd,
    ref_tf_apply_temporal_filter_lowbd, ref_tf_estimate_noise_highbd,
    ref_tf_estimate_noise_level_highbd, ref_tf_estimate_noise_level_lowbd,
    ref_tf_estimate_noise_lowbd,
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
    /// A value in `[lo, hi]`.
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32) as i32
    }
    /// A log-uniform value in `[0, max]`: pick a bit width first, then a value
    /// inside it. Keeps the small end of the range from being swamped.
    fn log_uniform(&mut self, max: u32) -> u32 {
        let bits = 32 - max.leading_zeros();
        let w = self.below(bits + 1);
        self.below(1u32.checked_shl(w).unwrap_or(u32::MAX).min(max) + 1)
    }
}

/// `BLOCK_64X64` — `TF_BLOCK_SIZE`, the only block size the filter is called at.
const BLOCK_64X64: i32 = 12;

/// C returns `-1.0` where the port returns `None`.
fn as_c(v: Option<f64>) -> f64 {
    v.unwrap_or(-1.0)
}

// ---------------------------------------------------------------------------
// Noise estimation.
// ---------------------------------------------------------------------------

#[test]
fn estimate_noise_lowbd_matches_c() {
    let mut rng = Rng::new(0x7F1E_9C3A);
    // Sizes spanning the degenerate end (a plane too thin for the 3x3 window)
    // through a real 4:2:0 chroma plane and a full luma plane.
    for &(w, h) in &[
        (2usize, 2usize),
        (3, 3),
        (5, 4),
        (16, 16),
        (33, 17),
        (64, 64),
        (80, 45),
    ] {
        for &pad in &[0usize, 7] {
            let stride = w + pad;
            for trial in 0..6 {
                // trial 0/1 are flat (every pixel smooth -> the accumulate arm);
                // the rest are noisy (many pixels above edge_thresh).
                let plane: Vec<u8> = (0..h * stride)
                    .map(|i| match trial {
                        0 => 128,
                        1 => (i % 2) as u8 * 3 + 100,
                        _ => rng.below(256) as u8,
                    })
                    .collect();
                for &edge_thresh in &[0i32, 8, NOISE_ESTIMATION_EDGE_THRESHOLD, 4096] {
                    let got = estimate_noise_from_single_plane(&plane, h, w, stride, edge_thresh);
                    let want = ref_tf_estimate_noise_lowbd(&plane, h, w, stride, edge_thresh);
                    assert_eq!(
                        as_c(got).to_bits(),
                        want.to_bits(),
                        "{w}x{h} stride {stride} trial {trial} thr {edge_thresh}"
                    );
                }
            }
        }
    }
}

#[test]
fn estimate_noise_highbd_matches_c() {
    let mut rng = Rng::new(0x2B4D_10FE);
    for &bd in &[10i32, 12] {
        let max = (1u32 << bd) - 1;
        for &(w, h) in &[(3usize, 3usize), (9, 6), (32, 32), (64, 33)] {
            for &pad in &[0usize, 5] {
                let stride = w + pad;
                for trial in 0..4 {
                    let plane: Vec<u16> = (0..h * stride)
                        .map(|_| match trial {
                            0 => (max / 2) as u16,
                            1 => (max / 2 + rng.below(3)) as u16,
                            _ => rng.below(max + 1) as u16,
                        })
                        .collect();
                    for &edge_thresh in &[0i32, NOISE_ESTIMATION_EDGE_THRESHOLD, 1 << 12] {
                        let got = highbd_estimate_noise_from_single_plane(
                            &plane,
                            h,
                            w,
                            stride,
                            bd as u32,
                            edge_thresh,
                        );
                        let want =
                            ref_tf_estimate_noise_highbd(&plane, h, w, stride, bd, edge_thresh);
                        assert_eq!(
                            as_c(got).to_bits(),
                            want.to_bits(),
                            "bd{bd} {w}x{h} stride {stride} trial {trial} thr {edge_thresh}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn estimate_noise_unreliable_sentinel_matches_c() {
    // A 5x5 plane has 9 interior pixels: fewer than the 16 C demands, so the
    // `-1.0` arm is taken even when every pixel is perfectly smooth. Without
    // this the `count < 16` branch is never exercised by the sweeps above at
    // their larger sizes.
    let plane = vec![64u8; 5 * 5];
    let got = estimate_noise_from_single_plane(&plane, 5, 5, 5, 4096);
    assert_eq!(
        got, None,
        "9 interior pixels must be rejected as unreliable"
    );
    assert_eq!(ref_tf_estimate_noise_lowbd(&plane, 5, 5, 5, 4096), -1.0);

    // 6x6 gives 16 interior pixels, which is exactly the boundary C accepts.
    let plane6 = vec![64u8; 6 * 6];
    let got6 = estimate_noise_from_single_plane(&plane6, 6, 6, 6, 4096);
    assert!(got6.is_some(), "16 interior pixels is C's accept boundary");
    assert_eq!(
        as_c(got6).to_bits(),
        ref_tf_estimate_noise_lowbd(&plane6, 6, 6, 6, 4096).to_bits()
    );
}

#[test]
fn estimate_noise_level_matches_c() {
    let mut rng = Rng::new(0xA11C_E5A1);
    let (yw, yh) = (48usize, 40usize);
    let (uw, uh) = (yw / 2, yh / 2);
    let (ys, us) = (yw + 6, uw + 3);

    for trial in 0..4 {
        let y: Vec<u8> = (0..yh * ys).map(|_| rng.below(256) as u8).collect();
        let u: Vec<u8> = (0..uh * us).map(|_| rng.below(256) as u8).collect();
        let v: Vec<u8> = (0..uh * us)
            .map(|_| {
                if trial == 0 {
                    128
                } else {
                    rng.below(256) as u8
                }
            })
            .collect();

        let planes_c = [
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
        ];
        let planes_r = [
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

        // Both plane ranges the encoder asks for: Y only, and all three.
        for &(from, to) in &[(0i32, 0i32), (0, 2)] {
            let want = ref_tf_estimate_noise_level_lowbd(
                planes_c,
                from,
                to,
                8,
                NOISE_ESTIMATION_EDGE_THRESHOLD,
            );
            let got = estimate_noise_level(
                &planes_r[from as usize..=to as usize],
                8,
                NOISE_ESTIMATION_EDGE_THRESHOLD,
            );
            for (i, g) in got.iter().enumerate() {
                assert_eq!(
                    as_c(*g).to_bits(),
                    want[from as usize + i].to_bits(),
                    "trial {trial} plane {} of {from}..={to}",
                    from as usize + i
                );
            }
        }
    }

    // High bit depth, both depths.
    for &bd in &[10i32, 12] {
        let max = (1u32 << bd) - 1;
        let y: Vec<u16> = (0..yh * ys).map(|_| rng.below(max + 1) as u16).collect();
        let u: Vec<u16> = (0..uh * us).map(|_| rng.below(max + 1) as u16).collect();
        let v: Vec<u16> = (0..uh * us).map(|_| rng.below(max + 1) as u16).collect();
        let want = ref_tf_estimate_noise_level_highbd(
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
            0,
            2,
            bd,
            NOISE_ESTIMATION_EDGE_THRESHOLD,
        );
        let got = estimate_noise_level(
            &[
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
            ],
            bd as u32,
            NOISE_ESTIMATION_EDGE_THRESHOLD,
        );
        for (i, g) in got.iter().enumerate() {
            assert_eq!(as_c(*g).to_bits(), want[i].to_bits(), "bd{bd} plane {i}");
        }
    }
}

// ---------------------------------------------------------------------------
// The filter itself.
// ---------------------------------------------------------------------------

/// One randomly drawn call, in both the port's and the oracle's shapes.
struct TfCase {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
    pred: Vec<u16>,
    ys: usize,
    us: usize,
    yw: usize,
    yh: usize,
    uw: usize,
    uh: usize,
    noise: [f64; 3],
    mvs: [(i16, i16); NUM_16X16],
    mses: [i32; NUM_16X16],
    params: TfRefParams,
    pels: usize,
}

/// Draw a call the encoder could actually make: a 64x64 TF block at
/// `(mb_row, mb_col)` inside a frame large enough to contain it.
fn draw_case(rng: &mut Rng, bd: u32, num_planes: i32, ss: (u32, u32), max_pix: u32) -> TfCase {
    let (ssx, ssy) = ss;
    let (mb_w, mb_h) = (64usize, 64usize);
    // Frame big enough for a 2x2 grid of TF blocks, so mb_row/mb_col > 0 is reachable.
    let mb_row = rng.below(2) as usize;
    let mb_col = rng.below(2) as usize;
    let yw = mb_w * 2;
    let yh = mb_h * 2;
    let (uw, uh) = (yw >> ssx, yh >> ssy);
    let ys = yw + 16;
    let us = uw + 8;

    let mut px = |n: usize| -> Vec<u16> { (0..n).map(|_| rng.below(max_pix + 1) as u16).collect() };
    let y = px(yh * ys);
    let u = px(uh * us);
    let v = px(uh * us);

    // Predictor: the planes laid out back to back, exactly as tf_build_predictor
    // does. It is the frame's own block plus a bounded perturbation, because
    // that is what a motion-compensated prediction IS. Drawing it independently
    // of the frame makes every window error saturate `scaled_error` at 7, so
    // every weight comes out 0 or 1 and the comparison degenerates
    // (`DIFFERENTIAL_PLAYBOOK` lesson 6). `spread` sweeps well-matched to badly
    // matched, in units of the bit depth.
    let spread = 1u32 << rng.below(6); // 1, 2, 4, ... 32 at 8 bits
    let spread = (spread * (max_pix + 1) / 256).max(1);
    let mut pels = mb_w * mb_h;
    if num_planes == 3 {
        pels += 2 * ((mb_h >> ssy) * (mb_w >> ssx));
    }
    let mut pred = Vec::with_capacity(pels);
    {
        let push_plane = |src: &[u16],
                          stride: usize,
                          w: usize,
                          h: usize,
                          r: usize,
                          c: usize,
                          rng: &mut Rng,
                          pred: &mut Vec<u16>| {
            let base = r * h * stride + c * w;
            for i in 0..h {
                for j in 0..w {
                    let v = i32::from(src[base + i * stride + j])
                        + rng.range(-(spread as i32), spread as i32);
                    pred.push(v.clamp(0, max_pix as i32) as u16);
                }
            }
        };
        push_plane(&y, ys, mb_w, mb_h, mb_row, mb_col, rng, &mut pred);
        if num_planes == 3 {
            let (cw, ch) = (mb_w >> ssx, mb_h >> ssy);
            push_plane(&u, us, cw, ch, mb_row, mb_col, rng, &mut pred);
            push_plane(&v, us, cw, ch, mb_row, mb_col, rng, &mut pred);
        }
    }
    assert_eq!(pred.len(), pels);

    let mut noise = [0.0f64; 3];
    for n in noise.iter_mut() {
        // Either the -1.0 "unreliable" sentinel or a plausible estimate.
        *n = if rng.below(5) == 0 {
            -1.0
        } else {
            f64::from(rng.below(4000)) / 100.0
        };
    }
    let mut mvs = [(0i16, 0i16); NUM_16X16];
    for m in mvs.iter_mut() {
        *m = (rng.range(-1024, 1024) as i16, rng.range(-1024, 1024) as i16);
    }
    // Per-pixel squared error, so bounded by (2^bd - 1)^2. Log-uniform: see
    // the module header on why a uniform draw makes the comparison vacuous.
    let max_mse = max_pix * max_pix;
    let mut mses = [0i32; NUM_16X16];
    for m in mses.iter_mut() {
        *m = rng.log_uniform(max_mse) as i32;
    }

    TfCase {
        y,
        u,
        v,
        pred,
        ys,
        us,
        yw,
        yh,
        uw,
        uh,
        noise,
        mvs,
        mses,
        params: TfRefParams {
            block_size: BLOCK_64X64,
            mb_row: mb_row as i32,
            mb_col: mb_col as i32,
            num_planes,
            subsampling_x: [0, ssx as i32, ssx as i32],
            subsampling_y: [0, ssy as i32, ssy as i32],
            bd: bd as i32,
            q_factor: rng.range(1, 255),
            filter_strength: rng.range(0, 6),
            tf_wgt_calc_lvl: rng.below(2) as i32,
        },
        pels,
    }
}

impl TfCase {
    fn port_params(&self) -> TfFilterParams {
        TfFilterParams {
            block_width: 64,
            block_height: 64,
            mb_row: self.params.mb_row as usize,
            mb_col: self.params.mb_col as usize,
            subsampling_x: [
                self.params.subsampling_x[0] as u32,
                self.params.subsampling_x[1] as u32,
                self.params.subsampling_x[2] as u32,
            ],
            subsampling_y: [
                self.params.subsampling_y[0] as u32,
                self.params.subsampling_y[1] as u32,
                self.params.subsampling_y[2] as u32,
            ],
            bd: self.params.bd as u32,
            q_factor: self.params.q_factor,
            filter_strength: self.params.filter_strength,
            wgt_calc_lvl: self.params.tf_wgt_calc_lvl,
        }
    }
}

#[test]
fn apply_temporal_filter_matches_c() {
    let mut rng = Rng::new(0x5EED_7F00);
    let mut checked = 0usize;
    let (mut nonzero, mut zero, mut big) = (0usize, 0usize, 0usize);
    for &num_planes in &[1i32, 3] {
        for &ss in &[(1u32, 1u32), (1, 0), (0, 0)] {
            for _ in 0..8 {
                let c = draw_case(&mut rng, 8, num_planes, ss, 255);
                let (y8, u8_, v8, p8) = (
                    c.y.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                    c.u.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                    c.v.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                    c.pred.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                );

                let mut accum_c = vec![0u32; c.pels];
                let mut count_c = vec![0u16; c.pels];
                ref_tf_apply_temporal_filter_lowbd(
                    [
                        TfRefPlane {
                            data: &y8,
                            stride: c.ys,
                            crop_width: c.yw,
                            crop_height: c.yh,
                        },
                        TfRefPlane {
                            data: &u8_,
                            stride: c.us,
                            crop_width: c.uw,
                            crop_height: c.uh,
                        },
                        TfRefPlane {
                            data: &v8,
                            stride: c.us,
                            crop_width: c.uw,
                            crop_height: c.uh,
                        },
                    ],
                    &c.params,
                    &c.noise,
                    &c.mvs,
                    &c.mses,
                    &p8,
                    &mut accum_c,
                    &mut count_c,
                );

                let mut accum_r = vec![0u32; c.pels];
                let mut count_r = vec![0u16; c.pels];
                let planes = [
                    TfPlane {
                        data: &y8[..],
                        stride: c.ys,
                        crop_width: c.yw,
                        crop_height: c.yh,
                    },
                    TfPlane {
                        data: &u8_[..],
                        stride: c.us,
                        crop_width: c.uw,
                        crop_height: c.uh,
                    },
                    TfPlane {
                        data: &v8[..],
                        stride: c.us,
                        crop_width: c.uw,
                        crop_height: c.uh,
                    },
                ];
                apply_temporal_filter(
                    &planes[..num_planes as usize],
                    &c.port_params(),
                    &c.noise,
                    &c.mvs,
                    &c.mses,
                    &p8,
                    &mut accum_r,
                    &mut count_r,
                );

                assert_eq!(accum_r, accum_c, "accum, np{num_planes} ss{ss:?}");
                assert_eq!(count_r, count_c, "count, np{num_planes} ss{ss:?}");
                nonzero += count_c.iter().filter(|&&c| c > 0).count();
                zero += count_c.iter().filter(|&&c| c == 0).count();
                big += count_c.iter().filter(|&&c| c > 100).count();
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 48, "the sweep must actually run every cell");
    // Non-vacuity: an all-zero-weight sweep would compare nothing but zeros.
    assert!(
        nonzero > 10_000,
        "only {nonzero} non-zero weights — sweep is vacuous"
    );
    assert!(zero > 0, "the saturated (weight == 0) arm never fired");
    assert!(
        big > 0,
        "no weight ever exceeded 100: every cell saturated at scaled_error == 7, \
         so the weight curve was only compared at its floor"
    );
}

#[test]
fn highbd_apply_temporal_filter_matches_c() {
    let mut rng = Rng::new(0xB17D_EF70);
    let mut checked = 0usize;
    let (mut nonzero, mut zero, mut big) = (0usize, 0usize, 0usize);
    for &bd in &[10u32, 12] {
        let max = (1u32 << bd) - 1;
        for &num_planes in &[1i32, 3] {
            for &ss in &[(1u32, 1u32), (0, 0)] {
                for _ in 0..5 {
                    let c = draw_case(&mut rng, bd, num_planes, ss, max);
                    let mut accum_c = vec![0u32; c.pels];
                    let mut count_c = vec![0u16; c.pels];
                    ref_tf_apply_temporal_filter_highbd(
                        [
                            TfRefPlane {
                                data: &c.y,
                                stride: c.ys,
                                crop_width: c.yw,
                                crop_height: c.yh,
                            },
                            TfRefPlane {
                                data: &c.u,
                                stride: c.us,
                                crop_width: c.uw,
                                crop_height: c.uh,
                            },
                            TfRefPlane {
                                data: &c.v,
                                stride: c.us,
                                crop_width: c.uw,
                                crop_height: c.uh,
                            },
                        ],
                        &c.params,
                        &c.noise,
                        &c.mvs,
                        &c.mses,
                        &c.pred,
                        &mut accum_c,
                        &mut count_c,
                    );

                    let mut accum_r = vec![0u32; c.pels];
                    let mut count_r = vec![0u16; c.pels];
                    let planes = [
                        TfPlane {
                            data: &c.y[..],
                            stride: c.ys,
                            crop_width: c.yw,
                            crop_height: c.yh,
                        },
                        TfPlane {
                            data: &c.u[..],
                            stride: c.us,
                            crop_width: c.uw,
                            crop_height: c.uh,
                        },
                        TfPlane {
                            data: &c.v[..],
                            stride: c.us,
                            crop_width: c.uw,
                            crop_height: c.uh,
                        },
                    ];
                    apply_temporal_filter(
                        &planes[..num_planes as usize],
                        &c.port_params(),
                        &c.noise,
                        &c.mvs,
                        &c.mses,
                        &c.pred,
                        &mut accum_r,
                        &mut count_r,
                    );
                    assert_eq!(accum_r, accum_c, "accum, bd{bd} np{num_planes} ss{ss:?}");
                    assert_eq!(count_r, count_c, "count, bd{bd} np{num_planes} ss{ss:?}");
                    nonzero += count_c.iter().filter(|&&c| c > 0).count();
                    zero += count_c.iter().filter(|&&c| c == 0).count();
                    big += count_c.iter().filter(|&&c| c > 100).count();
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(
        checked, 40,
        "2 depths x 2 plane counts x 2 subsamplings x 5 draws"
    );
    assert!(
        nonzero > 10_000,
        "only {nonzero} non-zero weights — sweep is vacuous"
    );
    assert!(zero > 0, "the saturated (weight == 0) arm never fired");
    assert!(
        big > 0,
        "no weight ever exceeded 100: every cell saturated at scaled_error == 7, \
         so the weight curve was only compared at its floor"
    );
}

#[test]
fn apply_temporal_filter_accumulates_across_frames() {
    // Drive one block over several reference frames the way `tf_do_filtering`
    // does: the same accum/count pair, a fresh predictor each time. A port that
    // ASSIGNED instead of accumulating passes the single-call tests above and
    // fails here on the second frame.
    let mut rng = Rng::new(0xACC0_1234);
    let mut base = draw_case(&mut rng, 8, 3, (1, 1), 255);
    // Pin the parameters into the regime where weights are non-zero, so the
    // accumulation being tested is actually observable: a well-matched block
    // (small MSE), a mid q and the maximum filter strength.
    base.params.q_factor = 40;
    base.params.filter_strength = 6;
    base.noise = [1.0, 1.0, 1.0];
    base.mses = [12i32; NUM_16X16];
    base.mvs = [(0i16, 0i16); NUM_16X16];
    let (y8, u8_, v8) = (
        base.y.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
        base.u.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
        base.v.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
    );
    let planes_c = [
        TfRefPlane {
            data: &y8,
            stride: base.ys,
            crop_width: base.yw,
            crop_height: base.yh,
        },
        TfRefPlane {
            data: &u8_,
            stride: base.us,
            crop_width: base.uw,
            crop_height: base.uh,
        },
        TfRefPlane {
            data: &v8,
            stride: base.us,
            crop_width: base.uw,
            crop_height: base.uh,
        },
    ];
    let planes_r = [
        TfPlane {
            data: &y8[..],
            stride: base.ys,
            crop_width: base.yw,
            crop_height: base.yh,
        },
        TfPlane {
            data: &u8_[..],
            stride: base.us,
            crop_width: base.uw,
            crop_height: base.uh,
        },
        TfPlane {
            data: &v8[..],
            stride: base.us,
            crop_width: base.uw,
            crop_height: base.uh,
        },
    ];

    let mut accum_c = vec![0u32; base.pels];
    let mut count_c = vec![0u16; base.pels];
    let mut accum_r = vec![0u32; base.pels];
    let mut count_r = vec![0u16; base.pels];

    // The predictor tracks the frame's own block (see draw_case) so the weights
    // are in the interesting part of the curve rather than saturated at 0.
    let block_pred = |rng: &mut Rng| -> Vec<u8> {
        let mut out = Vec::with_capacity(base.pels);
        let (r, c) = (base.params.mb_row as usize, base.params.mb_col as usize);
        let push =
            |src: &[u8], stride: usize, w: usize, h: usize, rng: &mut Rng, o: &mut Vec<u8>| {
                let b = r * h * stride + c * w;
                for i in 0..h {
                    for j in 0..w {
                        o.push(
                            (i32::from(src[b + i * stride + j]) + rng.range(-3, 3)).clamp(0, 255)
                                as u8,
                        );
                    }
                }
            };
        push(&y8, base.ys, 64, 64, rng, &mut out);
        push(&u8_, base.us, 32, 32, rng, &mut out);
        push(&v8, base.us, 32, 32, rng, &mut out);
        out
    };
    for frame in 0..5 {
        let pred = block_pred(&mut rng);
        ref_tf_apply_temporal_filter_lowbd(
            planes_c,
            &base.params,
            &base.noise,
            &base.mvs,
            &base.mses,
            &pred,
            &mut accum_c,
            &mut count_c,
        );
        apply_temporal_filter(
            &planes_r,
            &base.port_params(),
            &base.noise,
            &base.mvs,
            &base.mses,
            &pred,
            &mut accum_r,
            &mut count_r,
        );
        assert_eq!(accum_r, accum_c, "accum after frame {frame}");
        assert_eq!(count_r, count_c, "count after frame {frame}");
    }
    // Non-vacuity: the accumulator must actually have grown.
    assert!(
        accum_c.iter().any(|&a| a > 0),
        "the oracle produced an all-zero accumulator"
    );
    assert!(count_c.iter().any(|&c| c > 0));
}
