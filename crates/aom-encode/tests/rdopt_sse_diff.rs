//! Differential harness for the two `av1/encoder/rdopt.c` decisions that read
//! the encoder's VARIANCE FUNCTION TABLE — the port in `aom_encode::rdopt_sse`.
//!
//! **Tier 1c** (both are `static`; the oracle is libaom's own rdopt.c compiled
//! into the shim archive).
//!
//! | test | C function |
//! |---|---|
//! | `get_sse_matches_c` | `get_sse` `:868` |
//! | `prune_zero_mv_with_sse_matches_c` | `prune_zero_mv_with_sse` `:2809` |
//!
//! # What made these hard to reach
//!
//! Both call `fn_ptr[bsize].vf`, and libaom fills that table with a `BFP()`
//! macro cascade INLINE inside `av1_create_primary_compressor` — there is no
//! separately callable initialiser. The shim rebuilds the table from the same
//! exported `aom_variance<W>x<H>` entry points libaom itself assigns, so the
//! oracle dispatches exactly as the encoder does. On x86-64 those names are
//! RTCD function POINTERS that `ref_init()` has already populated; on aarch64
//! several are bound to their NEON variant at compile time. Variance is an
//! integer sum, so every tier agrees bit-for-bit with the scalar one — which
//! is what makes the port's use of the separately-gated
//! `aom_dsp::dist::variance` legitimate rather than a second transcription.

mod common;
use common::Rng;

use aom_encode::rdopt_sse::{NO_SINGLE_SSE, SsePlane, get_sse, prune_zero_mv_with_sse};
use aom_sys_ref as cref;

const BW: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BH: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

/// `get_plane_block_size(bsize, ss_x, ss_y)` — the subsampled block size table
/// (`common_data.h`), as a `(w, h)` pair in pixels. Only the four subsampling
/// combinations AV1 allows are needed.
fn plane_dims(bsize: usize, ss_x: usize, ss_y: usize) -> (usize, usize) {
    // The subsampled dimensions are the luma ones halved, and every AV1 block
    // shape has a subsampled counterpart except where the result would be
    // below 4 — which is why C's table exists. Clamping at 4 reproduces it for
    // every shape this harness sweeps.
    ((BW[bsize] >> ss_x).max(4), (BH[bsize] >> ss_y).max(4))
}

#[test]
fn get_sse_matches_c() {
    let mut rng = Rng(0x5eed_0071);
    let mut chroma_seen = 0;
    let mut n = 0;
    // Only square-ish shapes whose 4:2:0 counterpart exists in C's table are
    // swept: the shim asks C for `get_plane_block_size(bsize, ss, ss)`, and a
    // shape with no subsampled counterpart is BLOCK_INVALID there.
    for &bsize in &[3usize, 6, 9, 12, 4, 5, 7, 8, 10, 11] {
        for &(ss_x, ss_y) in &[(0usize, 0usize), (1, 1)] {
            for is_chroma_ref in [false, true] {
                for num_planes in [1, 3] {
                    for _ in 0..4 {
                        let (yw, yh) = (BW[bsize], BH[bsize]);
                        let (cw, ch) = plane_dims(bsize, ss_x, ss_y);
                        let mk = |rng: &mut Rng, w: usize, h: usize| -> (Vec<u8>, i32) {
                            let stride = w + 7;
                            (
                                (0..stride * (h + 4))
                                    .map(|_| rng.range(0, 256) as u8)
                                    .collect(),
                                stride as i32,
                            )
                        };
                        let (ysrc, yss) = mk(&mut rng, yw, yh);
                        let (ydst, yds) = mk(&mut rng, yw, yh);
                        let (usrc, uss) = mk(&mut rng, cw, ch);
                        let (udst, uds) = mk(&mut rng, cw, ch);
                        let (vsrc, vss) = mk(&mut rng, cw, ch);
                        let (vdst, vds) = mk(&mut rng, cw, ch);
                        let (want_total, want_y) = cref::ref_rdopt_get_sse(
                            bsize as i32,
                            num_planes,
                            is_chroma_ref,
                            (ss_x as i32, ss_y as i32),
                            &[
                                cref::SsePlaneC {
                                    src: &ysrc,
                                    src_stride: yss,
                                    dst: &ydst,
                                    dst_stride: yds,
                                },
                                cref::SsePlaneC {
                                    src: &usrc,
                                    src_stride: uss,
                                    dst: &udst,
                                    dst_stride: uds,
                                },
                                cref::SsePlaneC {
                                    src: &vsrc,
                                    src_stride: vss,
                                    dst: &vdst,
                                    dst_stride: vds,
                                },
                            ],
                        );
                        let mut planes = vec![SsePlane {
                            src: &ysrc,
                            src_stride: yss as usize,
                            dst: &ydst,
                            dst_stride: yds as usize,
                            w: yw,
                            h: yh,
                        }];
                        if num_planes == 3 {
                            planes.push(SsePlane {
                                src: &usrc,
                                src_stride: uss as usize,
                                dst: &udst,
                                dst_stride: uds as usize,
                                w: cw,
                                h: ch,
                            });
                            planes.push(SsePlane {
                                src: &vsrc,
                                src_stride: vss as usize,
                                dst: &vdst,
                                dst_stride: vds as usize,
                                w: cw,
                                h: ch,
                            });
                        }
                        let (got_total, got_y) = get_sse(&planes, is_chroma_ref);
                        assert_eq!(
                            (got_total, got_y),
                            (want_total, want_y),
                            "get_sse(bsize={bsize}, planes={num_planes}, \
                             chroma_ref={is_chroma_ref}, ss=({ss_x},{ss_y}))"
                        );
                        if num_planes == 3 && is_chroma_ref {
                            chroma_seen += 1;
                            assert_ne!(
                                want_total,
                                want_y << 4,
                                "chroma contributed nothing — the plane loop is \
                                 not reaching planes 1 and 2"
                            );
                        }
                        n += 1;
                    }
                }
            }
        }
    }
    assert!(n > 100);
    assert!(chroma_seen > 20, "the chroma arm was barely exercised");
}

#[test]
fn prune_zero_mv_with_sse_matches_c() {
    let mut rng = Rng(0x5eed_0072);
    let mut trues = 0;
    let mut n = 0;
    for &bsize in &[3usize, 6, 9, 12] {
        let (w, h) = (BW[bsize], BH[bsize]);
        for _ in 0..60 {
            let stride = w + 5;
            let mk = |rng: &mut Rng| -> Vec<u8> {
                (0..stride * (h + 4))
                    .map(|_| rng.range(0, 256) as u8)
                    .collect()
            };
            let src = mk(&mut rng);
            let ref0 = mk(&mut rng);
            let ref1 = mk(&mut rng);
            // Straddle the two early-outs: a non-IDENTITY global motion for
            // some references, and the INT32_MAX "no single SSE" sentinel.
            let mut gm = [0i32; 8];
            let mut best = [0u32; 8];
            for r in 0..8 {
                gm[r] = if rng.next() % 4 == 0 {
                    rng.range(2, 4)
                } else {
                    0
                };
                best[r] = if rng.next() % 4 == 0 {
                    NO_SINGLE_SSE
                } else {
                    rng.range(0, 1 << 22) as u32
                };
            }
            for rf in [(1, -1), (4, -1), (1, 5), (2, 6)] {
                for level in 1..3 {
                    let want = cref::ref_rdopt_prune_zero_mv_with_sse(
                        bsize as i32,
                        rf,
                        &gm,
                        &best,
                        &src,
                        stride as i32,
                        &ref0,
                        stride as i32,
                        &ref1,
                        stride as i32,
                        level,
                    );
                    // The port takes the two zero-MV SSEs as inputs (the C
                    // computes them with fn_ptr[bsize].vf from the same
                    // buffers), so the harness computes them with the
                    // separately-gated port variance.
                    let sse0 = aom_dsp::dist::variance(&ref0, stride, &src, stride, w, h).1;
                    let sse1 = aom_dsp::dist::variance(&ref1, stride, &src, stride, w, h).1;
                    let got =
                        prune_zero_mv_with_sse([rf.0, rf.1], &gm, &best, &[sse0, sse1], level);
                    assert_eq!(
                        got, want,
                        "prune_zero_mv_with_sse(bsize={bsize}, rf={rf:?}, \
                         level={level}, gm={gm:?})"
                    );
                    trues += usize::from(want);
                    n += 1;
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}
