//! Differential harness for libaom v3.15's `do_border_pad` kernels
//! (`aom_encode::border_pad`) against the REAL exported `av1_subtract_block`
//! (`encodemb.c`) — which subtracts the prediction and, with `do_border_pad`,
//! overwrites the residual outside the real frame (`fill_residue_outside_frame`)
//! after clipping through `get_visible_dimensions`.
//!
//! The whole `diff` plane is compared, not just the transform block: the fill must
//! touch nothing outside it. Every `TX_TYPE` (0..=15, so all three arms of the
//! fill) x low and high bit depth x luma and subsampled chroma x frame-edge
//! distances on both sides of zero is swept, with `do_border_pad` both ways (off
//! must be a plain subtract).

use crate::common::Rng;

use aom_encode::border_pad::{fill_residue_outside_frame, get_visible_dimensions};
use aom_sys_ref as cref;

/// Plane blocks as `(width, height, BLOCK_SIZE)`.
const PLANES: [(usize, usize, i32); 10] = [
    (8, 8, 3),
    (8, 16, 4),
    (16, 8, 5),
    (16, 16, 6),
    (16, 32, 7),
    (32, 16, 8),
    (32, 32, 9),
    (32, 64, 10),
    (64, 32, 11),
    (64, 64, 12),
];

#[test]
fn subtract_block_border_pad_matches_c() {
    let mut rng = Rng(0x5eed_00b0);
    let (mut clipped, mut filled_arms, mut n) = (0usize, [0usize; 3], 0usize);
    for &(pw, ph, pbs) in &PLANES {
        for bd in [8i32, 10, 12] {
            for (plane, ss) in [(0i32, 0u32), (1, 1)] {
                for tx_type in 0..16usize {
                    for _ in 0..6 {
                        // A transform block that fits the plane block, on its grid.
                        // Only the 19 real `TX_SIZE` shapes: the SSE2 high-bit-depth
                        // subtract has no kernel for the rest (it returns NULL and
                        // libaom never asks).
                        const TXS: [(usize, usize); 19] = [
                            (4, 4),
                            (8, 8),
                            (16, 16),
                            (32, 32),
                            (64, 64),
                            (4, 8),
                            (8, 4),
                            (8, 16),
                            (16, 8),
                            (16, 32),
                            (32, 16),
                            (32, 64),
                            (64, 32),
                            (4, 16),
                            (16, 4),
                            (8, 32),
                            (32, 8),
                            (16, 64),
                            (64, 16),
                        ];
                        let fit: Vec<(usize, usize)> = TXS
                            .iter()
                            .copied()
                            .filter(|&(w, h)| w <= pw && h <= ph)
                            .collect();
                        let (tw, th) = fit[rng.range(0, fit.len() as i32) as usize];
                        let blk_col = (rng.range(0, (pw / tw) as i32) as usize * tw) / 4;
                        let blk_row = (rng.range(0, (ph / th) as i32) as usize * th) / 4;
                        // Edge distances: inside (>= 0) or overhanging, either axis.
                        let edge = |rng: &mut Rng, span: usize| -> i32 {
                            if rng.range(0, 3) == 0 {
                                rng.range(0, 9)
                            } else {
                                -rng.range(1, (span as i32) * 2 + 2)
                            }
                        };
                        let pix_bottom = edge(&mut rng, ph << ss);
                        let pix_right = edge(&mut rng, pw << ss);
                        let do_border_pad = rng.range(0, 4) != 0;

                        let maxv = 1i32 << bd;
                        let (src_stride, pred_stride, diff_stride) = (pw + 8, pw + 4, pw);
                        let src: Vec<u16> = (0..src_stride * ph)
                            .map(|_| rng.range(0, maxv) as u16)
                            .collect();
                        let pred: Vec<u16> = (0..pred_stride * ph)
                            .map(|_| rng.range(0, maxv) as u16)
                            .collect();
                        // Pre-filled so "touches nothing outside the block" is observable.
                        let init: Vec<i16> = (0..diff_stride * ph)
                            .map(|_| rng.range(-999, 999) as i16)
                            .collect();

                        // The port: subtract, then (optionally) the border fill.
                        let mut got = init.clone();
                        let (off_s, off_p, off_d) = (
                            (blk_row * src_stride + blk_col) << 2,
                            (blk_row * pred_stride + blk_col) << 2,
                            (blk_row * diff_stride + blk_col) << 2,
                        );
                        for r in 0..th {
                            for c in 0..tw {
                                got[off_d + r * diff_stride + c] =
                                    (i32::from(src[off_s + r * src_stride + c])
                                        - i32::from(pred[off_p + r * pred_stride + c]))
                                        as i16;
                            }
                        }
                        if do_border_pad {
                            let (vc, vr, was_clipped) = get_visible_dimensions(
                                pix_bottom, pix_right, ss, ss, pw, ph, blk_col, blk_row, tw, th,
                                true,
                            );
                            if was_clipped {
                                clipped += 1;
                                let arm = if tx_type <= 9 {
                                    0
                                } else if matches!(tx_type, 10 | 12 | 14) {
                                    1
                                } else {
                                    2
                                };
                                filled_arms[arm] += 1;
                                fill_residue_outside_frame(
                                    &mut got[off_d..],
                                    diff_stride,
                                    tw,
                                    th,
                                    vc,
                                    vr,
                                    tx_type,
                                );
                            }
                        }

                        let case = cref::SubtractBlockCase {
                            bd,
                            tx_rows: th as i32,
                            tx_cols: tw as i32,
                            plane_rows: ph as i32,
                            plane,
                            plane_bsize: pbs,
                            blk_col: blk_col as i32,
                            blk_row: blk_row as i32,
                            tx_type: tx_type as i32,
                            do_border_pad,
                            pix_to_bottom_edge: pix_bottom,
                            pix_to_right_edge: pix_right,
                            ss_x: ss as i32,
                            ss_y: ss as i32,
                        };
                        let mut want = init.clone();
                        cref::ref_subtract_block_border_pad(
                            &case,
                            &mut want,
                            diff_stride,
                            &src,
                            src_stride,
                            &pred,
                            pred_stride,
                        );
                        assert_eq!(
                            got, want,
                            "plane {pw}x{ph} bd{bd} plane{plane} tx {tw}x{th} @({blk_row},{blk_col}) \
                             tx_type {tx_type} edges (b {pix_bottom}, r {pix_right}) pad {do_border_pad}"
                        );
                        n += 1;
                    }
                }
            }
        }
    }
    // Non-vacuity: the clipped fill must have run, in all three arms.
    assert!(clipped > n / 8, "only {clipped}/{n} cases clipped a block");
    for (i, k) in filled_arms.iter().enumerate() {
        assert!(*k > 40, "fill arm {i} ran only {k} times");
    }
}
