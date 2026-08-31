//! Differential harness for the self-contained half of
//! `av1/encoder/global_motion.c` vs the REAL exported C libaom v3.14.1.
//!
//! | test | C oracle | tier |
//! |---|---|---|
//! | `is_enough_erroradvantage_matches_c` | `av1_is_enough_erroradvantage` | 1 |
//! | `get_wmtype_matches_c` | `get_wmtype` (mv.h static inline, compiled into the shim) | 1 |
//! | `convert_model_to_params_matches_c` | `av1_convert_model_to_params` | 1 |
//! | `feature_segmentation_map_matches_c` | `av1_compute_feature_segmentation_map` | 1 |
//! | `segmented_frame_error_matches_c` | `av1_segmented_frame_error` (lowbd) | 1 |
//! | `highbd_segmented_frame_error_matches_c` | `av1_segmented_frame_error` (highbd) | 1 |
//!
//! **`add_param_offset` and `force_wmtype` are NOT tested here.** They are
//! file-static in C with no linkable symbol and are reachable only through
//! `av1_refine_integerized_param`, which is not ported yet. Writing a test that
//! compared the port against a second transcription of them would prove only
//! that both were transcribed the same way, so no such test exists.

use aom_encode::global_motion::{
    compute_feature_segmentation_map, convert_model_to_params, get_wmtype,
    highbd_segmented_frame_error, is_enough_erroradvantage, segmented_frame_error,
};
use aom_sys_ref::{
    ref_compute_feature_segmentation_map, ref_convert_model_to_params, ref_get_wmtype,
    ref_highbd_segmented_frame_error, ref_is_enough_erroradvantage, ref_segmented_frame_error,
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
    /// Uniform in `[-1, 1)`.
    fn signed_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

#[test]
fn is_enough_erroradvantage_matches_c() {
    // erroradv_prod_tr is 20000, so the sweep has to straddle
    // best_erroradvantage * params_cost == 20000 as well as the ratio
    // threshold, or only one of the two conjuncts is ever decisive.
    let mut saw_true = false;
    let mut saw_false_by_ratio = false;
    let mut saw_false_by_product = false;
    for &adv in &[0.0f64, 0.001, 0.1, 0.5, 0.6, 0.65, 0.7, 0.9, 1.0, 2.0] {
        for &cost in &[0i32, 1, 10, 100, 1000, 20000, 40000, 1 << 20] {
            for &tr in &[0.5f64, 0.65, 0.7, 1.0] {
                let got = is_enough_erroradvantage(adv, cost, tr);
                let want = ref_is_enough_erroradvantage(adv, cost, tr);
                assert_eq!(got, want, "adv={adv} cost={cost} tr={tr}");
                if got {
                    saw_true = true;
                } else if adv >= tr {
                    saw_false_by_ratio = true;
                } else {
                    saw_false_by_product = true;
                }
            }
        }
    }
    assert!(saw_true, "no accepting cell");
    assert!(
        saw_false_by_ratio,
        "the ratio conjunct never decided a rejection"
    );
    assert!(
        saw_false_by_product,
        "the erroradv_prod_tr conjunct never decided a rejection — the second \
         half of the predicate is untested"
    );
}

#[test]
fn get_wmtype_matches_c() {
    const ONE: i32 = 1 << 16;
    let mut rng = Rng::new(0x7A11_0B22_9C33_1D44);
    let mut seen = [false; 4];
    // Hand-built models hitting each classification, then random ones.
    let fixed: [[i32; 6]; 6] = [
        [0, 0, ONE, 0, 0, ONE],          // IDENTITY
        [128, -64, ONE, 0, 0, ONE],      // TRANSLATION
        [0, 0, ONE + 5, 7, -7, ONE + 5], // ROTZOOM
        [0, 0, ONE + 5, 7, 7, ONE + 5],  // AFFINE (wmmat[4] != -wmmat[3])
        [0, 0, ONE, 0, 0, ONE + 1],      // AFFINE
        [1, 0, ONE, 0, 0, ONE],          // TRANSLATION (only wmmat[0] set)
    ];
    for m in &fixed {
        let wm = aom_dsp::inter::warp::WarpedMotionParams {
            wmmat: *m,
            ..Default::default()
        };
        let got = i32::from(get_wmtype(&wm));
        let want = ref_get_wmtype(m);
        assert_eq!(got, want, "wmmat={m:?}");
        seen[want as usize] = true;
    }
    for _ in 0..20_000 {
        // Bias toward the one-valued diagonal so the ROTZOOM/TRANSLATION arms
        // are reachable rather than swamped by AFFINE.
        let mut m = [0i32; 6];
        for (k, e) in m.iter_mut().enumerate() {
            let base = if k == 2 || k == 5 { ONE } else { 0 };
            *e = base + (rng.below(5) as i32 - 2);
        }
        if rng.below(2) == 0 {
            m[4] = -m[3];
            m[5] = m[2];
        }
        let wm = aom_dsp::inter::warp::WarpedMotionParams {
            wmmat: m,
            ..Default::default()
        };
        let got = i32::from(get_wmtype(&wm));
        let want = ref_get_wmtype(&m);
        assert_eq!(got, want, "wmmat={m:?}");
        seen[want as usize] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "not every transformation type was produced: {seen:?}"
    );
}

#[test]
fn convert_model_to_params_matches_c() {
    let mut rng = Rng::new(0x2B4D_6F81_A3C5_E709);
    let mut saw_clamped = false;
    // Scales chosen to straddle the GM_TRANS / GM_ALPHA clamp boundaries:
    // GM_TRANS_MAX = 4096 at GM_TRANS_PREC_BITS = 6 means |params[0..2]| = 64
    // is the boundary; GM_ALPHA_MAX = 4096 at GM_ALPHA_PREC_BITS = 15 means
    // |params[2..6] - diag| = 0.125 is the boundary.
    for &(tscale, ascale) in &[
        (1.0f64, 0.01f64),
        (16.0, 0.1),
        (63.9, 0.124),
        (64.1, 0.126),
        (1000.0, 4.0),
        (0.0, 0.0),
    ] {
        for _ in 0..3000 {
            let mut params = [0f64; 6];
            params[0] = rng.signed_unit() * tscale;
            params[1] = rng.signed_unit() * tscale;
            for (k, p) in params.iter_mut().enumerate().skip(2) {
                let diag = if k == 2 || k == 5 { 1.0 } else { 0.0 };
                *p = diag + rng.signed_unit() * ascale;
            }
            let got = convert_model_to_params(&params);
            let (wmmat, wmtype, invalid) = ref_convert_model_to_params(&params);
            assert_eq!(got.wmmat, wmmat, "params={params:?}");
            assert_eq!(i32::from(got.wmtype), wmtype, "wmtype params={params:?}");
            assert_eq!(i32::from(got.invalid), invalid);
            if tscale > 64.0 || ascale > 0.125 {
                saw_clamped = true;
            }
        }
    }
    assert!(saw_clamped, "the clamp boundaries were never crossed");
}

#[test]
fn convert_model_to_params_rounds_half_up() {
    // floor(x + 0.5) is round-half-UP, which differs from Rust's f64::round
    // (round-half-away-from-zero) at negative half-way values. Feed exact
    // half-way inputs in both signs so the difference is decisive.
    let mut params = [0f64; 6];
    for &t in &[
        -2.5f64 / 64.0,
        -1.5 / 64.0,
        -0.5 / 64.0,
        0.5 / 64.0,
        1.5 / 64.0,
    ] {
        params[0] = t;
        params[1] = -t;
        params[2] = 1.0 + 0.5 / 32768.0;
        params[3] = -0.5 / 32768.0;
        params[4] = 0.5 / 32768.0;
        params[5] = 1.0 - 0.5 / 32768.0;
        let got = convert_model_to_params(&params);
        let (wmmat, wmtype, _) = ref_convert_model_to_params(&params);
        assert_eq!(got.wmmat, wmmat, "t={t}");
        assert_eq!(i32::from(got.wmtype), wmtype);
    }
}

#[test]
fn feature_segmentation_map_matches_c() {
    let mut rng = Rng::new(0x5E1F_0A2B_3C4D_5E6F);
    let mut saw_fallback = false;
    let mut saw_segmented = false;
    for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (40, 23), (60, 34)] {
        // num_inliers spans below and far above the SEG_COUNT_TR = 48 cell
        // threshold, so both the segmented map and the all-ones fallback fire.
        for &n in &[0usize, 1, 10, 100, 500, 4000, 20000] {
            let mut inliers = Vec::with_capacity(n * 2);
            for _ in 0..n {
                inliers.push(rng.below((w << 5) as u32) as i32);
                inliers.push(rng.below((h << 5) as u32) as i32);
            }
            let got = compute_feature_segmentation_map(w, h, &inliers, n);
            let want = ref_compute_feature_segmentation_map(w, h, &inliers, n);
            assert_eq!(got, want, "{w}x{h} n={n}");
            if got.iter().all(|&v| v == 1) {
                saw_fallback = true;
            } else {
                saw_segmented = true;
            }
        }
    }
    assert!(
        saw_fallback && saw_segmented,
        "only one of the two map outcomes was produced (fallback={saw_fallback}, \
         segmented={saw_segmented}) — the SEG_COUNT_TR branch is half-tested"
    );
}

/// A segment map with a mix of set and clear cells, plus the two degenerate
/// all-clear / all-set cases.
fn seg_maps(rng: &mut Rng, cells_w: usize, cells_h: usize) -> Vec<(Vec<u8>, usize)> {
    let stride = cells_w + 2;
    let mut out = Vec::new();
    out.push((vec![0u8; stride * cells_h], stride));
    out.push((vec![1u8; stride * cells_h], stride));
    let mixed: Vec<u8> = (0..stride * cells_h).map(|_| rng.below(2) as u8).collect();
    out.push((mixed, stride));
    out
}

#[test]
fn segmented_frame_error_matches_c() {
    let mut rng = Rng::new(0x1234_ABCD_5678_EF01);
    // Sizes crossing the WARP_ERROR_BLOCK (32) boundary in both axes, including
    // partial edge blocks, which is where the patch clipping matters.
    for &(w, h) in &[
        (32usize, 32usize),
        (64, 64),
        (96, 64),
        (33, 31),
        (100, 70),
        (16, 16),
    ] {
        let cells_w = w.div_ceil(32);
        let cells_h = h.div_ceil(32);
        let ref_stride = w + 9;
        let dst_stride = w + 3;
        let refp: Vec<u8> = (0..ref_stride * h).map(|_| rng.below(256) as u8).collect();
        let dst: Vec<u8> = (0..dst_stride * h).map(|_| rng.below(256) as u8).collect();
        for (map, mstride) in seg_maps(&mut rng, cells_w, cells_h) {
            let got =
                segmented_frame_error(&refp, ref_stride, &dst, dst_stride, w, h, &map, mstride);
            let want =
                ref_segmented_frame_error(&refp, ref_stride, &dst, dst_stride, w, h, &map, mstride);
            assert_eq!(got, want, "{w}x{h}");
        }
    }
}

#[test]
fn highbd_segmented_frame_error_matches_c() {
    let mut rng = Rng::new(0xFEDC_BA98_7654_3210);
    for &bd in &[8i32, 10, 12] {
        let maxval = 1u32 << bd;
        for &(w, h) in &[(32usize, 32usize), (64, 64), (96, 64), (33, 31), (100, 70)] {
            let cells_w = w.div_ceil(32);
            let cells_h = h.div_ceil(32);
            let ref_stride = w + 5;
            let dst_stride = w + 11;
            let refp: Vec<u16> = (0..ref_stride * h)
                .map(|_| rng.below(maxval) as u16)
                .collect();
            let dst: Vec<u16> = (0..dst_stride * h)
                .map(|_| rng.below(maxval) as u16)
                .collect();
            for (map, mstride) in seg_maps(&mut rng, cells_w, cells_h) {
                let got = highbd_segmented_frame_error(
                    &refp, ref_stride, &dst, dst_stride, w, h, bd, &map, mstride,
                );
                let want = ref_highbd_segmented_frame_error(
                    &refp, ref_stride, &dst, dst_stride, w, h, bd, &map, mstride,
                );
                assert_eq!(got, want, "bd={bd} {w}x{h}");
            }
        }
    }
}

// ===================================================================
// The warp-error metric and the refinement loop.
//
// `av1_refine_integerized_param` is the ONLY exported entry that reaches
// global_motion.c's file-static `add_param_offset`, `force_wmtype`,
// `warp_error` and `get_warp_error`, so this differential is what gates all
// four of them.
// ===================================================================

use aom_dsp::inter::warp::WarpedMotionParams;
use aom_encode::global_motion::{
    AFFINE, ROTZOOM, TRANSLATION, WarpErrorParams, refine_integerized_param,
};
use aom_sys_ref::ref_refine_integerized_param;

const ONE_WM: i32 = 1 << 16;

/// A frame pair: `ref` is random content, `dst` is `ref` shifted by a small
/// translation plus noise, so a translational/affine model genuinely reduces
/// the error and the coordinate descent has somewhere to walk.
fn warp_frames(rng: &mut Rng, w: usize, h: usize, shift: (i32, i32)) -> (Vec<u8>, Vec<u8>, usize) {
    let stride = w;
    let refp: Vec<u8> = (0..stride * h).map(|_| rng.below(256) as u8).collect();
    let mut dst = vec![0u8; stride * h];
    for y in 0..h {
        for x in 0..w {
            let sy = (y as i32 + shift.0).clamp(0, h as i32 - 1) as usize;
            let sx = (x as i32 + shift.1).clamp(0, w as i32 - 1) as usize;
            let v = i32::from(refp[sy * stride + sx]) + rng.below(7) as i32 - 3;
            dst[y * stride + x] = v.clamp(0, 255) as u8;
        }
    }
    (refp, dst, stride)
}

#[test]
fn refine_integerized_param_matches_c() {
    let mut rng = Rng::new(0x4D31_7E92_C0B5_A18F);
    let mut saw_refined = false;
    let mut saw_rejected = false;
    let mut saw_zero_refinements = false;

    for &(w, h) in &[(64usize, 64usize), (96, 64), (128, 96)] {
        let cells_w = w.div_ceil(32);
        let cells_h = h.div_ceil(32);
        for &shift in &[(0i32, 0i32), (1, 2), (-2, 1)] {
            let (refp, dst, stride) = warp_frames(&mut rng, w, h, shift);
            let refp16: Vec<u16> = refp.iter().map(|&v| u16::from(v)).collect();
            let dst16: Vec<u16> = dst.iter().map(|&v| u16::from(v)).collect();
            // All-ones map, plus a mixed one so the cell-skipping arm is live.
            let maps: [(Vec<u8>, usize); 2] = [
                (vec![1u8; cells_w * cells_h], cells_w),
                (
                    (0..cells_w * cells_h)
                        .map(|i| u8::from(i % 3 != 0))
                        .collect(),
                    cells_w,
                ),
            ];
            for (map, mstride) in &maps {
                for &wmtype in &[TRANSLATION, ROTZOOM, AFFINE] {
                    for &n_ref in &[0i32, 1, 3, 5] {
                        // Start models: identity-ish plus a small perturbation,
                        // which is the shape av1_convert_model_to_params emits.
                        let start: [i32; 6] = [
                            (rng.below(512) as i32 - 256) * 1024,
                            (rng.below(512) as i32 - 256) * 1024,
                            ONE_WM + (rng.below(64) as i32 - 32) * 2,
                            (rng.below(64) as i32 - 32) * 2,
                            (rng.below(64) as i32 - 32) * 2,
                            ONE_WM + (rng.below(64) as i32 - 32) * 2,
                        ];
                        // A reference error big enough that the model is not
                        // rejected outright on every cell, and small enough
                        // that it sometimes is.
                        for &rfe in &[0i64, 1 << 14, 1 << 20] {
                            let mut wm = WarpedMotionParams {
                                wmmat: start,
                                wmtype,
                                ..Default::default()
                            };
                            let p = WarpErrorParams {
                                refp: &refp16,
                                ref_width: w,
                                ref_height: h,
                                ref_stride: stride,
                                dst: &dst16,
                                dst_stride: stride,
                                subsampling_x: 0,
                                subsampling_y: 0,
                                segment_map: map,
                                segment_map_stride: *mstride,
                            };
                            let got = refine_integerized_param(
                                &mut wm, wmtype, &p, w, h, n_ref, rfe, 0.65,
                            );
                            let want = ref_refine_integerized_param(
                                &start,
                                i32::from(wmtype),
                                &refp,
                                w,
                                h,
                                stride,
                                &dst,
                                w,
                                h,
                                stride,
                                n_ref,
                                rfe,
                                map,
                                *mstride,
                                0.65,
                            );
                            let label = format!(
                                "{w}x{h} shift={shift:?} wmtype={wmtype} n_ref={n_ref} \
                                 rfe={rfe} start={start:?}"
                            );
                            assert_eq!(got, want.best_error, "best_error: {label}");
                            assert_eq!(wm.wmmat, want.wmmat, "wmmat: {label}");
                            assert_eq!(i32::from(wm.wmtype), want.wmtype, "wmtype: {label}");
                            assert_eq!(
                                [wm.alpha, wm.beta, wm.gamma, wm.delta],
                                want.shear,
                                "shear: {label}"
                            );

                            if n_ref == 0 {
                                saw_zero_refinements = true;
                            } else if got == i64::MAX {
                                saw_rejected = true;
                            } else if wm.wmmat != start {
                                saw_refined = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Without these, the sweep could be all early-rejections and never touch
    // the coordinate descent that `add_param_offset` and `force_wmtype` live in.
    assert!(
        saw_zero_refinements,
        "the n_refinements == 0 early path never ran"
    );
    assert!(
        saw_rejected,
        "no model was rejected by the erroradv_early_tr gate — that branch is untested"
    );
    assert!(
        saw_refined,
        "the coordinate descent never changed a parameter, so add_param_offset \
         and force_wmtype are still ungated"
    );
}

#[test]
fn warp_error_clips_against_the_reference_size() {
    // `warp_error` clips each cell's warp extent against `ref_width`/
    // `ref_height`, NOT against the `p_width`/`p_height` it is walking. With a
    // reference the same size as the frame the two are identical, so the sweep
    // above cannot tell them apart (a mutation swapping them passes it). This
    // cell makes the reference SMALLER than the frame, where they differ.
    let mut rng = Rng::new(0x1F2E_3D4C_5B6A_7988);
    let (w, h) = (128usize, 96usize);
    let (r_width, r_height) = (w - 40, h - 24);
    let cells_w = w.div_ceil(32);
    let cells_h = h.div_ceil(32);
    assert!(
        r_width % 32 != 0,
        "the reference width must not land on a WARP_ERROR_BLOCK boundary, or \
         the clipped and unclipped extents coincide again"
    );

    let (refp, dst, stride) = warp_frames(&mut rng, w, h, (1, 2));
    let refp16: Vec<u16> = refp.iter().map(|&v| u16::from(v)).collect();
    let dst16: Vec<u16> = dst.iter().map(|&v| u16::from(v)).collect();
    let map = vec![1u8; cells_w * cells_h];

    for &wmtype in &[TRANSLATION, ROTZOOM, AFFINE] {
        for &n_ref in &[0i32, 2] {
            let start: [i32; 6] = [4096, -2048, ONE_WM + 8, 4, -4, ONE_WM - 8];
            let mut wm = WarpedMotionParams {
                wmmat: start,
                wmtype,
                ..Default::default()
            };
            let p = WarpErrorParams {
                refp: &refp16,
                ref_width: r_width,
                ref_height: r_height,
                ref_stride: stride,
                dst: &dst16,
                dst_stride: stride,
                subsampling_x: 0,
                subsampling_y: 0,
                segment_map: &map,
                segment_map_stride: cells_w,
            };
            let got = refine_integerized_param(&mut wm, wmtype, &p, w, h, n_ref, 1 << 22, 0.65);
            let want = ref_refine_integerized_param(
                &start,
                i32::from(wmtype),
                &refp,
                r_width,
                r_height,
                stride,
                &dst,
                w,
                h,
                stride,
                n_ref,
                1 << 22,
                &map,
                cells_w,
                0.65,
            );
            assert_eq!(got, want.best_error, "wmtype={wmtype} n_ref={n_ref}");
            assert_eq!(wm.wmmat, want.wmmat, "wmtype={wmtype} n_ref={n_ref}");
        }
    }
}
