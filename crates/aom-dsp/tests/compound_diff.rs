//! Differential harness for the compound (two-reference) inter-prediction DSP
//! vs the REAL exported C libaom v3.14.1. **Tier 1 throughout** — every
//! expectation below comes from calling the C function itself through
//! `aom-sys-ref`, never from a second transcription of the same logic.
//!
//! Covers `aom_dsp::inter::compound`:
//!
//! | test | C oracle |
//! |---|---|
//! | `wedge_sse_from_residuals_matches_c` | `av1_wedge_sse_from_residuals_c` |
//! | `wedge_sign_from_residuals_matches_c` | `av1_wedge_sign_from_residuals_c` |
//! | `wedge_compute_delta_squares_matches_c` | `av1_wedge_compute_delta_squares_c` |
//! | `diffwtd_mask_matches_c` | `av1_build_compound_diffwtd_mask_c` |
//! | `diffwtd_mask_d16_matches_c` | `av1_build_compound_diffwtd_mask_d16_c` |
//! | `diffwtd_mask_highbd_matches_c` | `av1_build_compound_diffwtd_mask_highbd_c` |
//! | `compound_type_mask_wedge_matches_c` | `av1_get_compound_type_mask` (COMPOUND_WEDGE) |
//! | `wedge_mask_signed_matches_c_both_signs` | ditto, sweeping `wedge_sign` |
//! | `dist_wtd_comp_weight_assign_matches_c` | `av1_dist_wtd_comp_weight_assign` |
//!
//! The residual ranges are chosen to *reach* the saturation the C kernels
//! specify rather than to avoid it: `wedge_sse_from_residuals` clamps its
//! partial term to signed 16 bits and `wedge_compute_delta_squares` saturates
//! its output, so a test that only fed small residuals would pass against a
//! port that dropped both clamps.

use aom_dsp::inter::compound::{
    CompoundType, DiffwtdMaskType, build_compound_diffwtd_mask, build_compound_diffwtd_mask_d16,
    build_compound_diffwtd_mask_highbd, dist_wtd_comp_weight_assign, get_compound_type_mask,
};
use aom_dsp::inter::interintra::wedge_mask_signed;
use aom_sys_ref::{
    ref_build_compound_diffwtd_mask, ref_build_compound_diffwtd_mask_d16,
    ref_build_compound_diffwtd_mask_highbd, ref_dist_wtd_comp_weight_assign,
    ref_get_compound_type_mask_wedge, ref_wedge_compute_delta_squares,
    ref_wedge_sign_from_residuals, ref_wedge_sse_from_residuals,
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
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }
    /// Signed value in `[-range, range]`.
    fn signed(&mut self, range: i32) -> i16 {
        (self.below((2 * range + 1) as u32) as i32 - range) as i16
    }
}

/// The compound block sizes the wedge codebook covers (`bsize`, bw, bh) —
/// BLOCK_8X8..BLOCK_32X32 plus 8X32 / 32X8.
const WEDGE_BSIZES: [(usize, usize, usize); 9] = [
    (3, 8, 8),
    (4, 8, 16),
    (5, 16, 8),
    (6, 16, 16),
    (7, 16, 32),
    (8, 32, 16),
    (9, 32, 32),
    (18, 8, 32),
    (19, 32, 8),
];

/// Block shapes the DIFFWTD masks are built for (every compound-eligible size,
/// down to 8x8 and up to 128x128, plus the non-square extremes).
const MASK_SHAPES: [(usize, usize); 12] = [
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 128),
];

// ===================================================================
// wedge_utils.c
// ===================================================================

#[test]
fn wedge_sse_from_residuals_matches_c() {
    let mut rng = Rng(0x51ed_2701_c0ff_ee11);
    // Residual magnitudes span both regimes: well inside 10 signed bits (where
    // C's documentation says the clamp never binds) and far past it (where it
    // does, and where a port that dropped the clamp diverges).
    for &range in &[8i32, 64, 255, 1023, 4096, 20000] {
        for &n in &[16usize, 64, 256, 1024, 4096] {
            let r1: Vec<i16> = (0..n).map(|_| rng.signed(range)).collect();
            let d: Vec<i16> = (0..n).map(|_| rng.signed(range)).collect();
            let m: Vec<u8> = (0..n).map(|_| rng.below(65) as u8).collect();
            let got = aom_dsp::inter::compound::wedge_sse_from_residuals(&r1, &d, &m, n);
            let want = ref_wedge_sse_from_residuals(&r1, &d, &m, n);
            assert_eq!(got, want, "range={range} n={n}");
        }
    }
}

#[test]
fn wedge_sse_from_residuals_clamp_is_reached() {
    // A guard on the guard: prove the large-range cell above actually drives
    // the partial term outside signed 16 bits, so the clamp is genuinely under
    // test rather than merely present.
    let n = 256;
    let r1 = vec![20000i16; n];
    let d = vec![20000i16; n];
    let m = vec![64u8; n];
    let unclamped = 64i64 * 20000 + 64 * 20000;
    assert!(unclamped > i64::from(i16::MAX));
    let got = aom_dsp::inter::compound::wedge_sse_from_residuals(&r1, &d, &m, n);
    let want = ref_wedge_sse_from_residuals(&r1, &d, &m, n);
    assert_eq!(got, want);
}

#[test]
fn wedge_sign_from_residuals_matches_c() {
    let mut rng = Rng(0x00c0_ffee_5eed_0001);
    for &n in &[1usize, 2, 16, 64, 255, 1024, 4096] {
        for &range in &[16i32, 1024, 32767] {
            let ds: Vec<i16> = (0..n).map(|_| rng.signed(range)).collect();
            let m: Vec<u8> = (0..n).map(|_| rng.below(65) as u8).collect();
            // Sweep the limit across the decision boundary: the exact sum, and
            // one either side of it, so both the true and false arms are hit.
            let exact: i64 = ds
                .iter()
                .zip(&m)
                .map(|(a, b)| i64::from(i32::from(*a) * i32::from(*b)))
                .sum();
            for &limit in &[i64::MIN, exact - 1, exact, exact + 1, i64::MAX] {
                let got = aom_dsp::inter::compound::wedge_sign_from_residuals(&ds, &m, n, limit);
                let want = ref_wedge_sign_from_residuals(&ds, &m, n, limit);
                assert_eq!(got, want, "n={n} range={range} limit={limit}");
            }
        }
    }
}

#[test]
fn wedge_compute_delta_squares_matches_c() {
    let mut rng = Rng(0xdead_10cc_0bad_f00d);
    for &range in &[8i32, 181, 182, 1024, 32767] {
        for &n in &[16usize, 64, 1024, 4096] {
            let a: Vec<i16> = (0..n).map(|_| rng.signed(range)).collect();
            let b: Vec<i16> = (0..n).map(|_| rng.signed(range)).collect();
            let mut got = vec![0i16; n];
            aom_dsp::inter::compound::wedge_compute_delta_squares(&mut got, &a, &b, n);
            let want = ref_wedge_compute_delta_squares(&a, &b, n);
            assert_eq!(got, want, "range={range} n={n}");
        }
    }
}

#[test]
fn wedge_compute_delta_squares_saturates_like_c() {
    // 182^2 = 33124 > i16::MAX, so a - b with a = 182, b = 0 must saturate.
    let a = vec![182i16, -182, 32767, -32768];
    let b = vec![0i16, 0, 0, 0];
    let mut got = vec![0i16; 4];
    aom_dsp::inter::compound::wedge_compute_delta_squares(&mut got, &a, &b, 4);
    let want = ref_wedge_compute_delta_squares(&a, &b, 4);
    assert_eq!(got, want);
    assert_eq!(got[0], i16::MAX, "the saturation is real, not a no-op");
}

// ===================================================================
// The DIFFWTD masks
// ===================================================================

#[test]
fn diffwtd_mask_matches_c() {
    let mut rng = Rng(0x1a2b_3c4d_5e6f_7081);
    for &(w, h) in &MASK_SHAPES {
        for (mt_idx, mt) in [DiffwtdMaskType::Diffwtd38, DiffwtdMaskType::Diffwtd38Inv]
            .into_iter()
            .enumerate()
        {
            // Non-trivial strides so a port that assumed contiguity fails.
            let s0 = w + 7;
            let s1 = w + 3;
            let src0: Vec<u8> = (0..h * s0).map(|_| rng.below(256) as u8).collect();
            let src1: Vec<u8> = (0..h * s1).map(|_| rng.below(256) as u8).collect();
            let mut got = vec![0u8; h * w];
            build_compound_diffwtd_mask(&mut got, mt, &src0, s0, &src1, s1, h, w);
            let want = ref_build_compound_diffwtd_mask(mt_idx as i32, &src0, s0, &src1, s1, h, w);
            assert_eq!(got, want, "{w}x{h} mask_type={mt_idx}");
        }
    }
}

#[test]
fn diffwtd_mask_d16_matches_c() {
    let mut rng = Rng(0x9911_7733_5522_ffee);
    // (round_0, round_1, bd) triples libaom actually produces: the lowbd
    // compound rounding (3, 7 @ bd 8) and the two highbd ones the
    // av1_get_conv_params_no_round path yields at bd 10 / 12.
    let cfgs: [(i32, i32, i32); 3] = [(3, 7, 8), (3, 7, 10), (5, 7, 12)];
    for &(r0, r1, bd) in &cfgs {
        for &(w, h) in &MASK_SHAPES {
            for (mt_idx, mt) in [DiffwtdMaskType::Diffwtd38, DiffwtdMaskType::Diffwtd38Inv]
                .into_iter()
                .enumerate()
            {
                let s0 = w + 5;
                let s1 = w + 11;
                let src0: Vec<u16> = (0..h * s0).map(|_| rng.below(1 << 16) as u16).collect();
                let src1: Vec<u16> = (0..h * s1).map(|_| rng.below(1 << 16) as u16).collect();
                let mut got = vec![0u8; h * w];
                build_compound_diffwtd_mask_d16(
                    &mut got, mt, &src0, s0, &src1, s1, h, w, r0, r1, bd,
                );
                let want = ref_build_compound_diffwtd_mask_d16(
                    mt_idx as i32,
                    &src0,
                    s0,
                    &src1,
                    s1,
                    h,
                    w,
                    r0,
                    r1,
                    bd,
                );
                assert_eq!(got, want, "{w}x{h} mt={mt_idx} r0={r0} r1={r1} bd={bd}");
            }
        }
    }
}

#[test]
fn diffwtd_mask_highbd_matches_c() {
    let mut rng = Rng(0x4242_8484_1616_3232);
    for &bd in &[8u32, 10, 12] {
        let maxval = 1u32 << bd;
        for &(w, h) in &MASK_SHAPES {
            for (mt_idx, mt) in [DiffwtdMaskType::Diffwtd38, DiffwtdMaskType::Diffwtd38Inv]
                .into_iter()
                .enumerate()
            {
                let s0 = w + 9;
                let s1 = w + 2;
                let src0: Vec<u16> = (0..h * s0).map(|_| rng.below(maxval) as u16).collect();
                let src1: Vec<u16> = (0..h * s1).map(|_| rng.below(maxval) as u16).collect();
                let mut got = vec![0u8; h * w];
                build_compound_diffwtd_mask_highbd(&mut got, mt, &src0, s0, &src1, s1, h, w, bd);
                let want = ref_build_compound_diffwtd_mask_highbd(
                    mt_idx as i32,
                    &src0,
                    s0,
                    &src1,
                    s1,
                    h,
                    w,
                    bd as i32,
                );
                assert_eq!(got, want, "{w}x{h} mt={mt_idx} bd={bd}");
            }
        }
    }
}

// ===================================================================
// The wedge mask, both signs
// ===================================================================

#[test]
fn wedge_mask_signed_matches_c_both_signs() {
    let mut any_sign_differed = false;
    for &(bsize, bw, bh) in &WEDGE_BSIZES {
        for index in 0..16usize {
            let m0 = wedge_mask_signed(bsize, index, 0).expect("wedge bsize has a codebook");
            let m1 = wedge_mask_signed(bsize, index, 1).expect("wedge bsize has a codebook");
            let c0 = ref_get_compound_type_mask_wedge(bsize, index, 0, bw, bh)
                .expect("C returned no wedge mask for sign 0");
            let c1 = ref_get_compound_type_mask_wedge(bsize, index, 1, bw, bh)
                .expect("C returned no wedge mask for sign 1");
            assert_eq!(m0, c0, "bsize={bsize} index={index} sign=0");
            assert_eq!(m1, c1, "bsize={bsize} index={index} sign=1");
            if m0 != m1 {
                any_sign_differed = true;
            }
        }
    }
    // The sign is a real index into two distinct master planes, not a no-op —
    // if this ever fires, the sign argument stopped mattering and every
    // sign-1 assertion above became vacuous.
    assert!(
        any_sign_differed,
        "wedge_sign never changed the mask: the sign argument is not load-bearing"
    );
}

#[test]
fn compound_type_mask_wedge_matches_c() {
    for &(bsize, bw, bh) in &WEDGE_BSIZES {
        for index in 0..16usize {
            for sign in 0..2usize {
                let got = get_compound_type_mask(CompoundType::Wedge { index, sign }, bsize)
                    .expect("wedge bsize has a codebook");
                let want = ref_get_compound_type_mask_wedge(bsize, index, sign, bw, bh)
                    .expect("C returned no wedge mask");
                assert_eq!(got, want, "bsize={bsize} index={index} sign={sign}");
            }
        }
    }
}

#[test]
fn compound_type_mask_segmask_defers_to_caller() {
    // The default arm of av1_get_compound_type_mask returns comp_data->seg_mask,
    // which the caller owns. The port signals that with None rather than
    // fabricating a buffer.
    assert!(get_compound_type_mask(CompoundType::SegMask, 6).is_none());
}

// ===================================================================
// Distance-weighted compound offsets
// ===================================================================

#[test]
fn dist_wtd_comp_weight_assign_matches_c() {
    // Exhaustive over the order-hint neighbourhood that matters plus the two
    // early-out flags. order_hint_bits_minus_1 = 6 (7 bits) is libaom's
    // default; 0 (1 bit) exercises the wrap in get_relative_dist.
    let mut checked = 0usize;
    let mut saw_distwtd = false;
    for &ohbm1 in &[0i32, 2, 6] {
        let modulus = 1i32 << (ohbm1 + 1);
        for &enable in &[true, false] {
            for &compound_idx in &[false, true] {
                for &is_compound in &[true, false] {
                    for &have_fwd in &[true, false] {
                        for &have_bck in &[true, false] {
                            for cur in 0..modulus.min(16) {
                                for fwd in 0..modulus.min(16) {
                                    for bck in 0..modulus.min(16) {
                                        let got = dist_wtd_comp_weight_assign(
                                            enable,
                                            ohbm1,
                                            cur,
                                            if have_fwd { fwd } else { 0 },
                                            if have_bck { bck } else { 0 },
                                            compound_idx,
                                            is_compound,
                                        );
                                        let (wf, wb, wu) = ref_dist_wtd_comp_weight_assign(
                                            enable,
                                            ohbm1,
                                            cur,
                                            fwd,
                                            bck,
                                            have_fwd,
                                            have_bck,
                                            compound_idx,
                                            is_compound,
                                        );
                                        assert_eq!(
                                            (
                                                got.fwd_offset,
                                                got.bck_offset,
                                                got.use_dist_wtd_comp_avg
                                            ),
                                            (wf, wb, wu),
                                            "ohbm1={ohbm1} enable={enable} cidx={compound_idx} \
                                             iscomp={is_compound} fwd={fwd}({have_fwd}) \
                                             bck={bck}({have_bck}) cur={cur}"
                                        );
                                        checked += 1;
                                        if wu {
                                            saw_distwtd = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 1000, "grid collapsed: only {checked} cells");
    assert!(
        saw_distwtd,
        "no cell reached the distance-weighted arm: the table walk is untested"
    );
}

// ===================================================================
// The D16 (convolve-buffer domain) mask blend.
// ===================================================================

use aom_dsp::inter::compound::{highbd_blend_a64_d16_mask, lowbd_blend_a64_d16_mask};
use aom_sys_ref::{ref_highbd_blend_a64_d16_mask, ref_lowbd_blend_a64_d16_mask};

/// The four `(subw, subh)` mask-subsampling cases C unrolls into four loops.
const SUBSAMPLINGS: [(bool, bool); 4] =
    [(false, false), (true, true), (true, false), (false, true)];

#[test]
fn lowbd_blend_a64_d16_mask_matches_c() {
    let mut rng = Rng(0x77AA_33CC_11EE_55DD);
    // libaom asserts w, h >= 4 and both powers of two on this entry point.
    for &(w, h) in &[
        (4usize, 4usize),
        (8, 8),
        (16, 8),
        (8, 16),
        (32, 32),
        (64, 64),
        (128, 32),
    ] {
        for &(subw, subh) in &SUBSAMPLINGS {
            let s0 = w + 7;
            let s1 = w + 3;
            let mask_stride = 2 * w + 5;
            let src0: Vec<u16> = (0..h * s0).map(|_| rng.below(1 << 14) as u16).collect();
            let src1: Vec<u16> = (0..h * s1).map(|_| rng.below(1 << 14) as u16).collect();
            let mask: Vec<u8> = (0..(2 * h + 2) * mask_stride)
                .map(|_| rng.below(65) as u8)
                .collect();
            let (r0, r1) = (3i32, 7i32);
            let mut got = vec![0u8; w * h];
            lowbd_blend_a64_d16_mask(
                &mut got,
                w,
                &src0,
                s0,
                &src1,
                s1,
                &mask,
                mask_stride,
                w,
                h,
                subw,
                subh,
                r0,
                r1,
            );
            let want = ref_lowbd_blend_a64_d16_mask(
                &src0,
                s0,
                &src1,
                s1,
                &mask,
                mask_stride,
                w,
                h,
                subw,
                subh,
                r0,
                r1,
            );
            assert_eq!(got, want, "{w}x{h} subw={subw} subh={subh}");
        }
    }
}

#[test]
fn highbd_blend_a64_d16_mask_matches_c() {
    let mut rng = Rng(0x2244_6688_AACC_EE00);
    for &bd in &[8u32, 10, 12] {
        let (r0, r1) = if bd == 12 { (5i32, 7i32) } else { (3i32, 7i32) };
        for &(w, h) in &[
            (4usize, 4usize),
            (8, 8),
            (16, 8),
            (8, 16),
            (32, 32),
            (64, 64),
        ] {
            for &(subw, subh) in &SUBSAMPLINGS {
                let s0 = w + 5;
                let s1 = w + 11;
                let mask_stride = 2 * w + 3;
                let src0: Vec<u16> = (0..h * s0).map(|_| rng.below(1 << 15) as u16).collect();
                let src1: Vec<u16> = (0..h * s1).map(|_| rng.below(1 << 15) as u16).collect();
                let mask: Vec<u8> = (0..(2 * h + 2) * mask_stride)
                    .map(|_| rng.below(65) as u8)
                    .collect();
                let mut got = vec![0u16; w * h];
                highbd_blend_a64_d16_mask(
                    &mut got,
                    w,
                    &src0,
                    s0,
                    &src1,
                    s1,
                    &mask,
                    mask_stride,
                    w,
                    h,
                    subw,
                    subh,
                    r0,
                    r1,
                    bd,
                );
                let want = ref_highbd_blend_a64_d16_mask(
                    &src0,
                    s0,
                    &src1,
                    s1,
                    &mask,
                    mask_stride,
                    w,
                    h,
                    subw,
                    subh,
                    r0,
                    r1,
                    bd,
                );
                assert_eq!(got, want, "bd={bd} {w}x{h} subw={subw} subh={subh}");
            }
        }
    }
}

#[test]
fn d16_mask_subsamplings_are_four_distinct_cases() {
    // C unrolls the blend into four loops that read the mask differently. If
    // they ever produced the same output the sweeps above would be one test
    // repeated four times, and a port that collapsed the 2x2 average into two
    // nested pairwise averages (a different rounding) could pass.
    let mut rng = Rng(0x9F8E_7D6C_5B4A_3928);
    let (w, h) = (16usize, 16usize);
    let s = w;
    let mask_stride = 2 * w;
    let src0: Vec<u16> = (0..h * s).map(|_| rng.below(1 << 14) as u16).collect();
    let src1: Vec<u16> = (0..h * s).map(|_| rng.below(1 << 14) as u16).collect();
    let mask: Vec<u8> = (0..2 * h * mask_stride)
        .map(|_| rng.below(65) as u8)
        .collect();
    let mut outs = Vec::new();
    for &(subw, subh) in &SUBSAMPLINGS {
        let mut got = vec![0u8; w * h];
        lowbd_blend_a64_d16_mask(
            &mut got,
            w,
            &src0,
            s,
            &src1,
            s,
            &mask,
            mask_stride,
            w,
            h,
            subw,
            subh,
            3,
            7,
        );
        outs.push(got);
    }
    for a in 0..outs.len() {
        for b in (a + 1)..outs.len() {
            assert_ne!(
                outs[a], outs[b],
                "subsampling cases {a} and {b} produced identical output"
            );
        }
    }
}
