//! Differential harness for the SCALED-reference convolves vs the REAL exported
//! C libaom v3.14.1. **Tier 1.**
//!
//! | test | C oracle |
//! |---|---|
//! | `convolve_2d_scale_matches_c` | `av1_convolve_2d_scale_c` |
//! | `highbd_convolve_2d_scale_matches_c` | `av1_highbd_convolve_2d_scale_c` |
//!
//! The step values are the real ones: `x_step_qn` / `y_step_qn` come from
//! `av1_setup_scale_factors_for_frame` for actual frame-size ratios (the same
//! `to_coarse_point_scale` the port landed in `aom_dsp::inter::scale`), not
//! from round numbers. `steps_are_not_all_unit` asserts the sweep contains a
//! genuinely non-1:1 step, so a port that ignored the stepping — which is the
//! only thing separating this kernel from the fixed-phase one — cannot pass.

use aom_dsp::convolve::scaled::{ScaleConvolveParams, convolve_2d_scale, highbd_convolve_2d_scale};
use aom_dsp::convolve::{SUB_PEL_FILTERS_8, SUB_PEL_FILTERS_8SHARP, SUB_PEL_FILTERS_8SMOOTH};
use aom_dsp::inter::scale::ScaleFactors;
use aom_sys_ref::{
    RefScaleConvParams, RefScaleSteps, ref_convolve_2d_scale, ref_highbd_convolve_2d_scale,
};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// Generous border: the stepped walk can reach far past the block.
const BORDER: usize = 96;

const SHAPES: [(usize, usize); 6] = [(4, 4), (8, 8), (16, 16), (8, 16), (32, 32), (16, 8)];

/// `(x_step_qn, y_step_qn)` for a real reference/frame size ratio, via the
/// ported scale factors. Also returns the ratio for labelling.
fn steps_for(other: i32, this: i32) -> (i32, i32) {
    let sf = ScaleFactors::for_frame(other, other, this, this);
    assert!(
        sf.is_valid(),
        "{other}->{this} is out of the AV1 size window"
    );
    (sf.x_step_q4, sf.y_step_q4)
}

/// Real frame-size ratios inside AV1's 2x-down / 16x-up window.
const RATIOS: [(i32, i32); 5] = [(640, 640), (640, 320), (320, 640), (1280, 720), (352, 704)];

fn kernel_table(ftype: usize) -> &'static [[i16; 8]; 16] {
    match ftype {
        0 => &SUB_PEL_FILTERS_8,
        1 => &SUB_PEL_FILTERS_8SMOOTH,
        _ => &SUB_PEL_FILTERS_8SHARP,
    }
}

const COMBINES: [(bool, i32, i32); 3] = [(false, 8, 8), (true, 9, 7), (true, 13, 3)];

#[test]
fn steps_are_not_all_unit() {
    // 1 << SCALE_EXTRA_BITS == 64 is the 1:1 step. If every ratio collapsed to
    // it, this whole file would be testing the fixed-phase kernel again.
    let mut non_unit = 0;
    for &(o, t) in &RATIOS {
        let (xs, ys) = steps_for(o, t);
        if xs != 64 || ys != 64 {
            non_unit += 1;
        }
    }
    assert!(
        non_unit >= 3,
        "only {non_unit} of the ratios actually scale"
    );
}

#[test]
fn convolve_2d_scale_matches_c() {
    let mut rng = Rng(0x3C5A_9E71_0FD4_2B68);
    let (r0_s, r1_s) = (3i32, 11i32); // single-reference
    let (r0_c, r1_c) = (3i32, 7i32); // compound
    for &(o, t) in &RATIOS {
        let (x_step, y_step) = steps_for(o, t);
        for &(w, h) in &SHAPES {
            for &ftype in &[0usize, 1, 2] {
                for &(sx, sy) in &[(0i32, 0i32), (64, 192), (1023, 512)] {
                    let stride = w + 2 * BORDER;
                    let rows = h + 2 * BORDER;
                    let off = BORDER * stride + BORDER;
                    let src0: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                    let src1: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
                    let xf = kernel_table(ftype);
                    let yf = kernel_table(ftype);
                    let steps = RefScaleSteps {
                        subpel_x_qn: sx,
                        x_step_qn: x_step,
                        subpel_y_qn: sy,
                        y_step_qn: y_step,
                    };

                    // Single-reference.
                    let (mut d_p, mut d_c) = (vec![0u8; w * h], vec![0u8; w * h]);
                    let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    let cp = ScaleConvolveParams {
                        round_0: r0_s,
                        round_1: r1_s,
                        is_compound: false,
                        do_average: false,
                        use_dist_wtd_comp_avg: false,
                        fwd_offset: 0,
                        bck_offset: 0,
                    };
                    convolve_2d_scale(
                        &src0, off, stride, &mut d_p, w, &mut i_p, w, w, h, xf, yf, 8, sx, x_step,
                        sy, y_step, &cp,
                    );
                    ref_convolve_2d_scale(
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
                        8,
                        &steps,
                        &RefScaleConvParams {
                            round_0: r0_s,
                            round_1: r1_s,
                            is_compound: false,
                            do_average: false,
                            use_dist_wtd_comp_avg: false,
                            fwd_offset: 0,
                            bck_offset: 0,
                        },
                    );
                    assert_eq!(
                        d_p, d_c,
                        "single {o}->{t} {w}x{h} f={ftype} phase=({sx},{sy})"
                    );

                    // Compound, two passes.
                    for &c in &COMBINES {
                        let (mut d_p, mut d_c) = (vec![0u8; w * h], vec![0u8; w * h]);
                        let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        for (pass, s) in [(false, &src0), (true, &src1)] {
                            let cp = ScaleConvolveParams {
                                round_0: r0_c,
                                round_1: r1_c,
                                is_compound: true,
                                do_average: pass,
                                use_dist_wtd_comp_avg: c.0,
                                fwd_offset: c.1,
                                bck_offset: c.2,
                            };
                            convolve_2d_scale(
                                s, off, stride, &mut d_p, w, &mut i_p, w, w, h, xf, yf, 8, sx,
                                x_step, sy, y_step, &cp,
                            );
                            ref_convolve_2d_scale(
                                s,
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
                                8,
                                &steps,
                                &RefScaleConvParams {
                                    round_0: r0_c,
                                    round_1: r1_c,
                                    is_compound: true,
                                    do_average: pass,
                                    use_dist_wtd_comp_avg: c.0,
                                    fwd_offset: c.1,
                                    bck_offset: c.2,
                                },
                            );
                            if !pass {
                                assert_eq!(i_p, i_c, "compound pass1 {o}->{t} {w}x{h}");
                            } else {
                                assert_eq!(d_p, d_c, "compound pass2 {o}->{t} {w}x{h} c={c:?}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn highbd_convolve_2d_scale_matches_c() {
    let mut rng = Rng(0x71B4_2CE9_58A3_0D6F);
    for &bd in &[8u32, 10, 12] {
        // get_conv_params_no_round: bd 12 pushes round_0 to 5.
        let (r0_s, r1_s) = if bd == 12 {
            (5i32, 9i32)
        } else {
            (3i32, 11i32)
        };
        let (r0_c, r1_c) = if bd == 12 { (5i32, 7i32) } else { (3i32, 7i32) };
        let maxval = 1u32 << bd;
        for &(o, t) in &RATIOS {
            let (x_step, y_step) = steps_for(o, t);
            for &(w, h) in &SHAPES {
                for &(sx, sy) in &[(0i32, 0i32), (256, 768)] {
                    let stride = w + 2 * BORDER;
                    let rows = h + 2 * BORDER;
                    let off = BORDER * stride + BORDER;
                    let src0: Vec<u16> = (0..stride * rows)
                        .map(|_| rng.below(maxval) as u16)
                        .collect();
                    let src1: Vec<u16> = (0..stride * rows)
                        .map(|_| rng.below(maxval) as u16)
                        .collect();
                    let xf = kernel_table(0);
                    let yf = kernel_table(2);
                    let steps = RefScaleSteps {
                        subpel_x_qn: sx,
                        x_step_qn: x_step,
                        subpel_y_qn: sy,
                        y_step_qn: y_step,
                    };

                    let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                    let cp = ScaleConvolveParams {
                        round_0: r0_s,
                        round_1: r1_s,
                        is_compound: false,
                        do_average: false,
                        use_dist_wtd_comp_avg: false,
                        fwd_offset: 0,
                        bck_offset: 0,
                    };
                    highbd_convolve_2d_scale(
                        &src0, off, stride, &mut d_p, w, &mut i_p, w, w, h, xf, yf, 8, sx, x_step,
                        sy, y_step, &cp, bd,
                    );
                    ref_highbd_convolve_2d_scale(
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
                        8,
                        &steps,
                        &RefScaleConvParams {
                            round_0: r0_s,
                            round_1: r1_s,
                            is_compound: false,
                            do_average: false,
                            use_dist_wtd_comp_avg: false,
                            fwd_offset: 0,
                            bck_offset: 0,
                        },
                        bd,
                    );
                    assert_eq!(d_p, d_c, "single bd={bd} {o}->{t} {w}x{h}");

                    for &c in &COMBINES {
                        let (mut d_p, mut d_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        let (mut i_p, mut i_c) = (vec![0u16; w * h], vec![0u16; w * h]);
                        for (pass, s) in [(false, &src0), (true, &src1)] {
                            let cp = ScaleConvolveParams {
                                round_0: r0_c,
                                round_1: r1_c,
                                is_compound: true,
                                do_average: pass,
                                use_dist_wtd_comp_avg: c.0,
                                fwd_offset: c.1,
                                bck_offset: c.2,
                            };
                            highbd_convolve_2d_scale(
                                s, off, stride, &mut d_p, w, &mut i_p, w, w, h, xf, yf, 8, sx,
                                x_step, sy, y_step, &cp, bd,
                            );
                            ref_highbd_convolve_2d_scale(
                                s,
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
                                8,
                                &steps,
                                &RefScaleConvParams {
                                    round_0: r0_c,
                                    round_1: r1_c,
                                    is_compound: true,
                                    do_average: pass,
                                    use_dist_wtd_comp_avg: c.0,
                                    fwd_offset: c.1,
                                    bck_offset: c.2,
                                },
                                bd,
                            );
                            if !pass {
                                assert_eq!(i_p, i_c, "compound pass1 bd={bd} {o}->{t} {w}x{h}");
                            } else {
                                assert_eq!(d_p, d_c, "compound pass2 bd={bd} {o}->{t} {w}x{h}");
                            }
                        }
                    }
                }
            }
        }
    }
}
