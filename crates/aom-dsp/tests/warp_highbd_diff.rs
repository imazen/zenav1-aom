//! Differential harness for the general affine warp filter
//! (`av1_highbd_warp_affine_c`) vs the REAL exported C libaom v3.14.1.
//! **Tier 1.**
//!
//! `tests/warp_diff.rs` already gates the bd-8 non-compound specialization the
//! decoder's warp path uses. This gates the full function, which the port did
//! not have at all: high bit depth, and the COMPOUND arm at any depth.
//!
//! The compound arm is driven the way the encoder drives it — two passes,
//! reference 0 with `do_average = false` (writes only the 16-bit intermediate)
//! then reference 1 with `do_average = true` (reads it back and writes pixels)
//! — and both buffers are compared.
//!
//! `bd_arms_are_not_a_widened_bd8` and `compound_and_single_arms_differ` are
//! the guards against a vacuous sweep: the bit depths and the two arms must
//! actually produce different output, or a port that folded them together
//! could pass by having each cell agree with itself.

use aom_dsp::inter::warp::{
    WarpConvolveParams, WarpedMotionParams, get_shear_params, highbd_warp_affine,
};
use aom_sys_ref::{RefWarpConvParams, ref_highbd_warp_affine};

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

const ONE_WM: i32 = 1 << 16;

/// A model with resolvable shear parameters, plus the shear values C derives.
/// Returns `None` if `av1_get_shear_params` rejects the model, which the caller
/// skips — C's warp filter is only ever called on an accepted model.
/// `(wmmat, (alpha, beta, gamma, delta))`.
type Model = ([i32; 6], (i16, i16, i16, i16));

fn model(rng: &mut Rng) -> Option<Model> {
    let mat = [
        (rng.below(4096) as i32 - 2048) * 64,
        (rng.below(4096) as i32 - 2048) * 64,
        ONE_WM + (rng.below(2048) as i32 - 1024),
        rng.below(2048) as i32 - 1024,
        rng.below(2048) as i32 - 1024,
        ONE_WM + (rng.below(2048) as i32 - 1024),
    ];
    let mut wm = WarpedMotionParams {
        wmmat: mat,
        wmtype: aom_dsp::inter::warp::AFFINE,
        ..Default::default()
    };
    if !get_shear_params(&mut wm) {
        return None;
    }
    Some((mat, (wm.alpha, wm.beta, wm.gamma, wm.delta)))
}

/// `(round_0, round_1)` for a warp block, as `get_conv_params_no_round`
/// produces them.
fn rounds(bd: u32, is_compound: bool) -> (i32, i32) {
    let mut round_0 = 3i32;
    let mut round_1 = if is_compound { 7 } else { 2 * 7 - round_0 };
    let intbufrange = bd as i32 + 7 - round_0 + 2;
    if intbufrange > 16 {
        round_0 += intbufrange - 16;
        if !is_compound {
            round_1 -= intbufrange - 16;
        }
    }
    (round_0, round_1)
}

const SHAPES: [(usize, usize); 5] = [(8, 8), (16, 16), (8, 16), (32, 32), (16, 8)];

#[test]
fn highbd_warp_affine_single_ref_matches_c() {
    let mut rng = Rng::new(0x77C1_9B3E_4D50_A26F);
    let mut cases = 0usize;
    for &bd in &[8u32, 10, 12] {
        let (r0, r1) = rounds(bd, false);
        let maxval = 1u32 << bd;
        for &(w, h) in &SHAPES {
            for &(ssx, ssy) in &[(0usize, 0usize), (1, 1), (1, 0)] {
                for _ in 0..4 {
                    let Some((mat, shear)) = model(&mut rng) else {
                        continue;
                    };
                    let stride = w + 8;
                    let refp: Vec<u16> = (0..stride * (h + 8))
                        .map(|_| rng.below(maxval) as u16)
                        .collect();
                    let mut got = vec![0u16; w * h];
                    let mut want = vec![0u16; w * h];
                    let mut d_got = vec![0u16; w * h];
                    let mut d_want = vec![0u16; w * h];
                    let cp = WarpConvolveParams {
                        round_0: r0,
                        round_1: r1,
                        is_compound: false,
                        do_average: false,
                        use_dist_wtd_comp_avg: false,
                        fwd_offset: 0,
                        bck_offset: 0,
                    };
                    highbd_warp_affine(
                        &mat,
                        &refp,
                        w + 8,
                        h + 8,
                        stride,
                        &mut got,
                        w,
                        &mut d_got,
                        w,
                        0,
                        0,
                        w,
                        h,
                        ssx,
                        ssy,
                        bd,
                        &cp,
                        shear.0,
                        shear.1,
                        shear.2,
                        shear.3,
                    );
                    ref_highbd_warp_affine(
                        &mat,
                        &refp,
                        w + 8,
                        h + 8,
                        stride,
                        &mut want,
                        w,
                        &mut d_want,
                        w,
                        0,
                        0,
                        w,
                        h,
                        ssx,
                        ssy,
                        bd,
                        &RefWarpConvParams {
                            round_0: r0,
                            round_1: r1,
                            is_compound: false,
                            do_average: false,
                            use_dist_wtd_comp_avg: false,
                            fwd_offset: 0,
                            bck_offset: 0,
                        },
                        shear,
                    );
                    assert_eq!(got, want, "bd={bd} {w}x{h} ss=({ssx},{ssy})");
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 50, "too few accepted models: {cases}");
}

#[test]
fn highbd_warp_affine_compound_matches_c() {
    let mut rng = Rng::new(0x1D2C_3B4A_5968_7776);
    // (use_dist_wtd, fwd, bck) combos, including two real
    // quant_dist_lookup_table pairs.
    let combines: [(bool, i32, i32); 3] = [(false, 8, 8), (true, 9, 7), (true, 13, 3)];
    let mut cases = 0usize;
    for &bd in &[8u32, 10, 12] {
        let (r0, r1) = rounds(bd, true);
        let maxval = 1u32 << bd;
        for &(w, h) in &SHAPES {
            for &c in &combines {
                let Some((mat, shear)) = model(&mut rng) else {
                    continue;
                };
                let stride = w + 8;
                let ref0: Vec<u16> = (0..stride * (h + 8))
                    .map(|_| rng.below(maxval) as u16)
                    .collect();
                let ref1: Vec<u16> = (0..stride * (h + 8))
                    .map(|_| rng.below(maxval) as u16)
                    .collect();
                let mut p_got = vec![0u16; w * h];
                let mut p_want = vec![0u16; w * h];
                let mut d_got = vec![0u16; w * h];
                let mut d_want = vec![0u16; w * h];

                for (pass, refp) in [(&ref0, false), (&ref1, true)]
                    .iter()
                    .map(|(r, avg)| (*avg, *r))
                {
                    let cp = WarpConvolveParams {
                        round_0: r0,
                        round_1: r1,
                        is_compound: true,
                        do_average: pass,
                        use_dist_wtd_comp_avg: c.0,
                        fwd_offset: c.1,
                        bck_offset: c.2,
                    };
                    highbd_warp_affine(
                        &mat,
                        refp,
                        w + 8,
                        h + 8,
                        stride,
                        &mut p_got,
                        w,
                        &mut d_got,
                        w,
                        0,
                        0,
                        w,
                        h,
                        0,
                        0,
                        bd,
                        &cp,
                        shear.0,
                        shear.1,
                        shear.2,
                        shear.3,
                    );
                    ref_highbd_warp_affine(
                        &mat,
                        refp,
                        w + 8,
                        h + 8,
                        stride,
                        &mut p_want,
                        w,
                        &mut d_want,
                        w,
                        0,
                        0,
                        w,
                        h,
                        0,
                        0,
                        bd,
                        &RefWarpConvParams {
                            round_0: r0,
                            round_1: r1,
                            is_compound: true,
                            do_average: pass,
                            use_dist_wtd_comp_avg: c.0,
                            fwd_offset: c.1,
                            bck_offset: c.2,
                        },
                        shear,
                    );
                    if !pass {
                        assert_eq!(d_got, d_want, "pass1 dst16 bd={bd} {w}x{h} c={c:?}");
                    } else {
                        assert_eq!(p_got, p_want, "pass2 pred bd={bd} {w}x{h} c={c:?}");
                    }
                }
                cases += 1;
            }
        }
    }
    assert!(cases > 20, "too few accepted models: {cases}");
}

#[test]
fn bd_arms_are_not_a_widened_bd8() {
    // Same model, same content values, three bit depths: the bd-dependent
    // offset_bits and the final `-(1 << (bd - 1)) - (1 << bd)` de-biasing must
    // make the outputs differ, or the bd sweeps above are three copies of one.
    let mut rng = Rng::new(0x9182_7364_5546_3728);
    let (w, h) = (16usize, 16usize);
    let stride = w + 8;
    let refp: Vec<u16> = (0..stride * (h + 8))
        .map(|_| rng.below(256) as u16)
        .collect();
    let mut mat_shear = None;
    for _ in 0..64 {
        if let Some(m) = model(&mut rng) {
            mat_shear = Some(m);
            break;
        }
    }
    let (mat, shear) = mat_shear.expect("no accepted model in 64 tries");
    let mut outs = Vec::new();
    for &bd in &[8u32, 10, 12] {
        let (r0, r1) = rounds(bd, false);
        let mut got = vec![0u16; w * h];
        let mut d = vec![0u16; w * h];
        highbd_warp_affine(
            &mat,
            &refp,
            w + 8,
            h + 8,
            stride,
            &mut got,
            w,
            &mut d,
            w,
            0,
            0,
            w,
            h,
            0,
            0,
            bd,
            &WarpConvolveParams {
                round_0: r0,
                round_1: r1,
                is_compound: false,
                do_average: false,
                use_dist_wtd_comp_avg: false,
                fwd_offset: 0,
                bck_offset: 0,
            },
            shear.0,
            shear.1,
            shear.2,
            shear.3,
        );
        outs.push(got);
    }
    assert_ne!(outs[0], outs[1], "bd 8 and bd 10 produced identical output");
    assert_ne!(
        outs[1], outs[2],
        "bd 10 and bd 12 produced identical output"
    );
}

#[test]
fn compound_and_single_arms_differ() {
    // `is_compound` changes `reduce_bits_vert` from `2*FILTER_BITS - round_0`
    // to `round_1`, so the two arms round the vertical pass differently. If
    // they agreed, the compound sweep would be a duplicate of the single one.
    let mut rng = Rng::new(0x5566_7788_99AA_BBCC);
    let (w, h) = (16usize, 16usize);
    let stride = w + 8;
    let refp: Vec<u16> = (0..stride * (h + 8))
        .map(|_| rng.below(1024) as u16)
        .collect();
    let mut mat_shear = None;
    for _ in 0..64 {
        if let Some(m) = model(&mut rng) {
            mat_shear = Some(m);
            break;
        }
    }
    let (mat, shear) = mat_shear.expect("no accepted model in 64 tries");
    let bd = 10u32;

    let (sr0, sr1) = rounds(bd, false);
    let mut single = vec![0u16; w * h];
    let mut d0 = vec![0u16; w * h];
    highbd_warp_affine(
        &mat,
        &refp,
        w + 8,
        h + 8,
        stride,
        &mut single,
        w,
        &mut d0,
        w,
        0,
        0,
        w,
        h,
        0,
        0,
        bd,
        &WarpConvolveParams {
            round_0: sr0,
            round_1: sr1,
            is_compound: false,
            do_average: false,
            use_dist_wtd_comp_avg: false,
            fwd_offset: 0,
            bck_offset: 0,
        },
        shear.0,
        shear.1,
        shear.2,
        shear.3,
    );

    let (cr0, cr1) = rounds(bd, true);
    let mut pred = vec![0u16; w * h];
    let mut d1 = vec![0u16; w * h];
    highbd_warp_affine(
        &mat,
        &refp,
        w + 8,
        h + 8,
        stride,
        &mut pred,
        w,
        &mut d1,
        w,
        0,
        0,
        w,
        h,
        0,
        0,
        bd,
        &WarpConvolveParams {
            round_0: cr0,
            round_1: cr1,
            is_compound: true,
            do_average: false,
            use_dist_wtd_comp_avg: false,
            fwd_offset: 0,
            bck_offset: 0,
        },
        shear.0,
        shear.1,
        shear.2,
        shear.3,
    );

    // The compound first pass writes only the intermediate and leaves `pred`
    // untouched; that alone distinguishes the arms.
    assert!(
        pred.iter().all(|&v| v == 0),
        "the compound first pass wrote pixels"
    );
    assert!(
        d1.iter().any(|&v| v != 0),
        "the compound first pass wrote no intermediate"
    );
    assert!(
        single.iter().any(|&v| v != 0),
        "the single-ref arm produced nothing"
    );
}
