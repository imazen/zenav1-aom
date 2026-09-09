//! Differential harness for the compound + high-bit-depth motion-compensation
//! convolutions vs the REAL exported C libaom v3.14.1. **Tier 1 throughout.**
//!
//! Covers `aom_dsp::convolve::compound` (the eight `av1_dist_wtd_convolve_*`
//! kernels, lowbd and highbd) and `aom_dsp::convolve::highbd` (the three
//! `av1_highbd_convolve_*_sr_c` single-reference kernels).
//!
//! Each compound kernel is exercised the way the encoder actually drives it —
//! **two passes**: reference 0 with `do_average = false` (which writes only the
//! 16-bit intermediate) then reference 1 with `do_average = true` (which reads
//! that intermediate back and writes the final pixel). Both the intermediate
//! and the final pixels are compared, so a port that got the intermediate right
//! and the combine wrong still fails.
//!
//! The `(round_0, round_1, bd)` triples are the ones
//! `get_conv_params_no_round` (av1/common/convolve.h:68) actually produces,
//! including the bd-12 arm where `intbufrange > 16` pushes `round_0` to 5 —
//! a cell a "3 and 7 everywhere" sweep would never reach.

use aom_dsp::convolve::compound::{
    CompoundConvolveParams, dist_wtd_convolve_2d, dist_wtd_convolve_2d_copy, dist_wtd_convolve_x,
    dist_wtd_convolve_y, highbd_dist_wtd_convolve_2d, highbd_dist_wtd_convolve_2d_copy,
    highbd_dist_wtd_convolve_x, highbd_dist_wtd_convolve_y,
};
use aom_dsp::convolve::highbd::{
    highbd_convolve_2d_sr, highbd_convolve_x_sr, highbd_convolve_y_sr,
};
use aom_dsp::convolve::{SUB_PEL_FILTERS_8, SUB_PEL_FILTERS_8SHARP, SUB_PEL_FILTERS_8SMOOTH};
use aom_sys_ref::{
    RefCompoundConvParams, ref_dist_wtd_convolve_2d, ref_dist_wtd_convolve_2d_copy,
    ref_dist_wtd_convolve_x, ref_dist_wtd_convolve_y, ref_highbd_convolve_2d_sr,
    ref_highbd_convolve_x_sr, ref_highbd_convolve_y_sr, ref_highbd_dist_wtd_convolve_2d,
    ref_highbd_dist_wtd_convolve_2d_copy, ref_highbd_dist_wtd_convolve_x,
    ref_highbd_dist_wtd_convolve_y,
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
}

/// Border kept on every side of the reference so the 8-tap kernels can reach
/// outside the block without a bounds check.
const BORDER: usize = 8;

/// Block shapes: the compound-eligible set plus the 4-wide/4-high extremes.
const SHAPES: [(usize, usize); 11] = [
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (32, 16),
    (32, 32),
    (64, 64),
    (16, 64),
];

/// The eight-tap kernel tables, indexed the way `InterpFilter` is.
fn kernel(ftype: usize, subpel: usize) -> &'static [i16; 8] {
    match ftype {
        0 => &SUB_PEL_FILTERS_8[subpel],
        1 => &SUB_PEL_FILTERS_8SMOOTH[subpel],
        _ => &SUB_PEL_FILTERS_8SHARP[subpel],
    }
}

/// `(round_0, round_1)` for a compound block at `bd`, as
/// `get_conv_params_no_round(.., is_compound = 1, bd)` computes it.
fn compound_rounds(bd: u32) -> (i32, i32) {
    let mut round_0 = 3i32;
    let round_1 = 7i32;
    let intbufrange = bd as i32 + 7 - round_0 + 2;
    if intbufrange > 16 {
        round_0 += intbufrange - 16;
    }
    (round_0, round_1)
}

/// `(round_0, round_1)` for a single-reference block at `bd`.
fn single_rounds(bd: u32) -> (i32, i32) {
    let mut round_0 = 3i32;
    let mut round_1 = 2 * 7 - round_0;
    let intbufrange = bd as i32 + 7 - round_0 + 2;
    if intbufrange > 16 {
        round_0 += intbufrange - 16;
        round_1 -= intbufrange - 16;
    }
    (round_0, round_1)
}

#[test]
fn compound_rounds_reach_the_bd12_arm() {
    // Guard on the sweep: if this ever stops holding, every bd-12 assertion
    // below silently degenerates into a second copy of the bd-8 one.
    assert_eq!(compound_rounds(8), (3, 7));
    assert_eq!(compound_rounds(10), (3, 7));
    assert_eq!(compound_rounds(12), (5, 7));
    assert_eq!(single_rounds(8), (3, 11));
    assert_eq!(single_rounds(12), (5, 9));
}

/// The `(use_dist_wtd, fwd, bck)` combos the second pass is run with: the plain
/// average, and two of the `quant_dist_lookup_table` offset pairs.
const COMBINES: [(bool, i32, i32); 4] = [(false, 8, 8), (true, 9, 7), (true, 13, 3), (true, 4, 12)];

fn params(
    round_0: i32,
    round_1: i32,
    do_average: bool,
    c: (bool, i32, i32),
) -> CompoundConvolveParams {
    CompoundConvolveParams {
        round_0,
        round_1,
        do_average,
        use_dist_wtd_comp_avg: c.0,
        fwd_offset: c.1,
        bck_offset: c.2,
    }
}

fn ref_params(
    round_0: i32,
    round_1: i32,
    do_average: bool,
    c: (bool, i32, i32),
) -> RefCompoundConvParams {
    RefCompoundConvParams {
        round_0,
        round_1,
        do_average,
        use_dist_wtd_comp_avg: c.0,
        fwd_offset: c.1,
        bck_offset: c.2,
    }
}

// ===================================================================
// lowbd
// ===================================================================

#[test]
fn dist_wtd_convolve_2d_matches_c() {
    let mut rng = Rng(0x0bad_c0de_face_1234);
    let (r0, r1) = compound_rounds(8);
    for &(w, h) in &SHAPES {
        for &ftype in &[0usize, 1, 2] {
            for &(sx, sy) in &[(0usize, 0usize), (1, 15), (7, 8), (15, 3)] {
                let stride = w + 2 * BORDER;
                let rows = h + 2 * BORDER;
                let src0: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                let src1: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                let off = BORDER * stride + BORDER;
                let xf = kernel(ftype, sx);
                let yf = kernel(ftype, sy);

                for &c in &COMBINES {
                    let (mut d_p, mut d_c) = (vec![0u8; w * h], vec![0u8; w * h]);
                    let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);

                    // Pass 1: reference 0, do_average = false.
                    dist_wtd_convolve_2d(
                        &src0,
                        off,
                        stride,
                        &mut d_p,
                        w,
                        &mut i_p,
                        w,
                        w,
                        h,
                        xf,
                        yf,
                        &params(r0, r1, false, c),
                    );
                    ref_dist_wtd_convolve_2d(
                        &src0,
                        off,
                        stride,
                        &mut d_c,
                        w,
                        &mut i_c,
                        w,
                        w,
                        h,
                        xf,
                        yf,
                        &ref_params(r0, r1, false, c),
                    );
                    assert_eq!(i_p, i_c, "pass1 dst16 {w}x{h} f={ftype} sx={sx} sy={sy}");

                    // Pass 2: reference 1, do_average = true.
                    dist_wtd_convolve_2d(
                        &src1,
                        off,
                        stride,
                        &mut d_p,
                        w,
                        &mut i_p,
                        w,
                        w,
                        h,
                        xf,
                        yf,
                        &params(r0, r1, true, c),
                    );
                    ref_dist_wtd_convolve_2d(
                        &src1,
                        off,
                        stride,
                        &mut d_c,
                        w,
                        &mut i_c,
                        w,
                        w,
                        h,
                        xf,
                        yf,
                        &ref_params(r0, r1, true, c),
                    );
                    assert_eq!(
                        d_p, d_c,
                        "pass2 dst {w}x{h} f={ftype} sx={sx} sy={sy} c={c:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn dist_wtd_convolve_x_matches_c() {
    let mut rng = Rng(0x1111_2222_3333_4444);
    let (r0, r1) = compound_rounds(8);
    for &(w, h) in &SHAPES {
        for &ftype in &[0usize, 1, 2] {
            for &sx in &[0usize, 1, 8, 15] {
                let stride = w + 2 * BORDER;
                let rows = h + 2 * BORDER;
                let src0: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                let src1: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                let off = BORDER * stride + BORDER;
                let xf = kernel(ftype, sx);
                for &c in &COMBINES {
                    let (mut d_p, mut d_c) = (vec![0u8; w * h], vec![0u8; w * h]);
                    let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    dist_wtd_convolve_x(
                        &src0,
                        off,
                        stride,
                        &mut d_p,
                        w,
                        &mut i_p,
                        w,
                        w,
                        h,
                        xf,
                        &params(r0, r1, false, c),
                    );
                    ref_dist_wtd_convolve_x(
                        &src0,
                        off,
                        stride,
                        &mut d_c,
                        w,
                        &mut i_c,
                        w,
                        w,
                        h,
                        xf,
                        &ref_params(r0, r1, false, c),
                    );
                    assert_eq!(i_p, i_c, "pass1 {w}x{h} f={ftype} sx={sx}");
                    dist_wtd_convolve_x(
                        &src1,
                        off,
                        stride,
                        &mut d_p,
                        w,
                        &mut i_p,
                        w,
                        w,
                        h,
                        xf,
                        &params(r0, r1, true, c),
                    );
                    ref_dist_wtd_convolve_x(
                        &src1,
                        off,
                        stride,
                        &mut d_c,
                        w,
                        &mut i_c,
                        w,
                        w,
                        h,
                        xf,
                        &ref_params(r0, r1, true, c),
                    );
                    assert_eq!(d_p, d_c, "pass2 {w}x{h} f={ftype} sx={sx} c={c:?}");
                }
            }
        }
    }
}

#[test]
fn dist_wtd_convolve_y_matches_c() {
    let mut rng = Rng(0x5555_6666_7777_8888);
    let (r0, r1) = compound_rounds(8);
    for &(w, h) in &SHAPES {
        for &ftype in &[0usize, 1, 2] {
            for &sy in &[0usize, 1, 8, 15] {
                let stride = w + 2 * BORDER;
                let rows = h + 2 * BORDER;
                let src0: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                let src1: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                let off = BORDER * stride + BORDER;
                let yf = kernel(ftype, sy);
                for &c in &COMBINES {
                    let (mut d_p, mut d_c) = (vec![0u8; w * h], vec![0u8; w * h]);
                    let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    dist_wtd_convolve_y(
                        &src0,
                        off,
                        stride,
                        &mut d_p,
                        w,
                        &mut i_p,
                        w,
                        w,
                        h,
                        yf,
                        &params(r0, r1, false, c),
                    );
                    ref_dist_wtd_convolve_y(
                        &src0,
                        off,
                        stride,
                        &mut d_c,
                        w,
                        &mut i_c,
                        w,
                        w,
                        h,
                        yf,
                        &ref_params(r0, r1, false, c),
                    );
                    assert_eq!(i_p, i_c, "pass1 {w}x{h} f={ftype} sy={sy}");
                    dist_wtd_convolve_y(
                        &src1,
                        off,
                        stride,
                        &mut d_p,
                        w,
                        &mut i_p,
                        w,
                        w,
                        h,
                        yf,
                        &params(r0, r1, true, c),
                    );
                    ref_dist_wtd_convolve_y(
                        &src1,
                        off,
                        stride,
                        &mut d_c,
                        w,
                        &mut i_c,
                        w,
                        w,
                        h,
                        yf,
                        &ref_params(r0, r1, true, c),
                    );
                    assert_eq!(d_p, d_c, "pass2 {w}x{h} f={ftype} sy={sy} c={c:?}");
                }
            }
        }
    }
}

#[test]
fn dist_wtd_convolve_2d_copy_matches_c() {
    let mut rng = Rng(0x9999_aaaa_bbbb_cccc);
    let (r0, r1) = compound_rounds(8);
    for &(w, h) in &SHAPES {
        let stride = w + 2 * BORDER;
        let rows = h + 2 * BORDER;
        let src0: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
        let src1: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
        let off = BORDER * stride + BORDER;
        for &c in &COMBINES {
            let (mut d_p, mut d_c) = (vec![0u8; w * h], vec![0u8; w * h]);
            let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
            dist_wtd_convolve_2d_copy(
                &src0,
                off,
                stride,
                &mut d_p,
                w,
                &mut i_p,
                w,
                w,
                h,
                &params(r0, r1, false, c),
            );
            ref_dist_wtd_convolve_2d_copy(
                &src0,
                off,
                stride,
                &mut d_c,
                w,
                &mut i_c,
                w,
                w,
                h,
                &ref_params(r0, r1, false, c),
            );
            assert_eq!(i_p, i_c, "pass1 {w}x{h}");
            dist_wtd_convolve_2d_copy(
                &src1,
                off,
                stride,
                &mut d_p,
                w,
                &mut i_p,
                w,
                w,
                h,
                &params(r0, r1, true, c),
            );
            ref_dist_wtd_convolve_2d_copy(
                &src1,
                off,
                stride,
                &mut d_c,
                w,
                &mut i_c,
                w,
                w,
                h,
                &ref_params(r0, r1, true, c),
            );
            assert_eq!(d_p, d_c, "pass2 {w}x{h} c={c:?}");
        }
    }
}

// ===================================================================
// highbd compound
// ===================================================================

#[test]
fn highbd_dist_wtd_convolve_matches_c() {
    let mut rng = Rng(0xfeed_beef_0123_4567);
    for &bd in &[8u32, 10, 12] {
        let (r0, r1) = compound_rounds(bd);
        let maxval = 1u32 << bd;
        for &(w, h) in &SHAPES {
            for &ftype in &[0usize, 2] {
                for &(sx, sy) in &[(0usize, 0usize), (5, 11), (15, 15)] {
                    let stride = w + 2 * BORDER;
                    let rows = h + 2 * BORDER;
                    let src0: Vec<u16> = (0..stride * rows)
                        .map(|_| rng.below(maxval) as u16)
                        .collect();
                    let src1: Vec<u16> = (0..stride * rows)
                        .map(|_| rng.below(maxval) as u16)
                        .collect();
                    let off = BORDER * stride + BORDER;
                    let xf = kernel(ftype, sx);
                    let yf = kernel(ftype, sy);
                    for &c in &COMBINES {
                        // 2d
                        let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        highbd_dist_wtd_convolve_2d(
                            &src0,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            xf,
                            yf,
                            &params(r0, r1, false, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_2d(
                            &src0,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            xf,
                            yf,
                            &ref_params(r0, r1, false, c),
                            bd,
                        );
                        assert_eq!(i_p, i_c, "2d pass1 bd={bd} {w}x{h}");
                        highbd_dist_wtd_convolve_2d(
                            &src1,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            xf,
                            yf,
                            &params(r0, r1, true, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_2d(
                            &src1,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            xf,
                            yf,
                            &ref_params(r0, r1, true, c),
                            bd,
                        );
                        assert_eq!(d_p, d_c, "2d pass2 bd={bd} {w}x{h} c={c:?}");

                        // x
                        let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        highbd_dist_wtd_convolve_x(
                            &src0,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            xf,
                            &params(r0, r1, false, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_x(
                            &src0,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            xf,
                            &ref_params(r0, r1, false, c),
                            bd,
                        );
                        assert_eq!(i_p, i_c, "x pass1 bd={bd} {w}x{h}");
                        highbd_dist_wtd_convolve_x(
                            &src1,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            xf,
                            &params(r0, r1, true, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_x(
                            &src1,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            xf,
                            &ref_params(r0, r1, true, c),
                            bd,
                        );
                        assert_eq!(d_p, d_c, "x pass2 bd={bd} {w}x{h} c={c:?}");

                        // y
                        let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        highbd_dist_wtd_convolve_y(
                            &src0,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            yf,
                            &params(r0, r1, false, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_y(
                            &src0,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            yf,
                            &ref_params(r0, r1, false, c),
                            bd,
                        );
                        assert_eq!(i_p, i_c, "y pass1 bd={bd} {w}x{h}");
                        highbd_dist_wtd_convolve_y(
                            &src1,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            yf,
                            &params(r0, r1, true, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_y(
                            &src1,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            yf,
                            &ref_params(r0, r1, true, c),
                            bd,
                        );
                        assert_eq!(d_p, d_c, "y pass2 bd={bd} {w}x{h} c={c:?}");

                        // 2d_copy
                        let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        highbd_dist_wtd_convolve_2d_copy(
                            &src0,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            &params(r0, r1, false, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_2d_copy(
                            &src0,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            &ref_params(r0, r1, false, c),
                            bd,
                        );
                        assert_eq!(i_p, i_c, "copy pass1 bd={bd} {w}x{h}");
                        highbd_dist_wtd_convolve_2d_copy(
                            &src1,
                            off,
                            stride,
                            &mut d_p,
                            w,
                            &mut i_p,
                            w,
                            w,
                            h,
                            &params(r0, r1, true, c),
                            bd,
                        );
                        ref_highbd_dist_wtd_convolve_2d_copy(
                            &src1,
                            off,
                            stride,
                            &mut d_c,
                            w,
                            &mut i_c,
                            w,
                            w,
                            h,
                            &ref_params(r0, r1, true, c),
                            bd,
                        );
                        assert_eq!(d_p, d_c, "copy pass2 bd={bd} {w}x{h} c={c:?}");
                    }
                }
            }
        }
    }
}

// ===================================================================
// highbd single-reference
// ===================================================================

#[test]
fn highbd_convolve_sr_matches_c() {
    let mut rng = Rng(0x0102_0304_0506_0708);
    for &bd in &[8u32, 10, 12] {
        let (r0, r1) = single_rounds(bd);
        let maxval = 1u32 << bd;
        for &(w, h) in &SHAPES {
            for &ftype in &[0usize, 1, 2] {
                for &(sx, sy) in &[(0usize, 0usize), (3, 12), (15, 1)] {
                    let stride = w + 2 * BORDER;
                    let rows = h + 2 * BORDER;
                    let src: Vec<u16> = (0..stride * rows)
                        .map(|_| rng.below(maxval) as u16)
                        .collect();
                    let off = BORDER * stride + BORDER;
                    let xf = kernel(ftype, sx);
                    let yf = kernel(ftype, sy);

                    let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    highbd_convolve_x_sr(&src, off, stride, &mut d_p, w, w, h, xf, r0, bd);
                    ref_highbd_convolve_x_sr(&src, off, stride, &mut d_c, w, w, h, xf, r0, r1, bd);
                    assert_eq!(d_p, d_c, "x_sr bd={bd} {w}x{h} f={ftype} sx={sx}");

                    let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    highbd_convolve_y_sr(&src, off, stride, &mut d_p, w, w, h, yf, bd);
                    ref_highbd_convolve_y_sr(&src, off, stride, &mut d_c, w, w, h, yf, bd);
                    assert_eq!(d_p, d_c, "y_sr bd={bd} {w}x{h} f={ftype} sy={sy}");

                    let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    highbd_convolve_2d_sr(&src, off, stride, &mut d_p, w, w, h, xf, yf, r0, r1, bd);
                    ref_highbd_convolve_2d_sr(
                        &src, off, stride, &mut d_c, w, w, h, xf, yf, r0, r1, bd,
                    );
                    assert_eq!(d_p, d_c, "2d_sr bd={bd} {w}x{h} f={ftype} sx={sx} sy={sy}");
                }
            }
        }
    }
}
