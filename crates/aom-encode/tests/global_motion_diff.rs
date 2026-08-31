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
