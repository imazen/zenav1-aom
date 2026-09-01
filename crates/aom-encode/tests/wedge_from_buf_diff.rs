//! Differential harness for the masked-compound assembly in
//! `av1/encoder/reconinter_enc.c` — the port in
//! `aom_encode::inter_pred_enc`.
//!
//! # Evidence tier
//!
//! **Tier 1c.** `nm -g upstream/build/libaom.a` reports
//! `av1_build_wedge_inter_predictor_from_buf` for reconinter_enc.c but neither
//! `build_wedge_inter_predictor_from_buf` (the per-plane worker) nor
//! `build_masked_compound{,_highbd}` (the blend) — both are `static`. So
//! `crates/aom-sys-ref/shim/reconinter_enc_shim.c` compiles libaom's OWN
//! reconinter_enc.c with its exports renamed out of the way and wraps the
//! statics: the bodies under test are libaom's source, compiled a second time
//! under libaom's own Release flags.
//!
//! | test | C function (`reconinter_enc.c`) |
//! |---|---|
//! | `build_masked_compound_matches_c` | `build_masked_compound` `:312` / `_highbd` `:330` |
//! | `build_wedge_from_buf_plane_matches_c` | `build_wedge_inter_predictor_from_buf` `:349` |
//! | `av1_build_wedge_from_buf_matches_c` | `av1_build_wedge_inter_predictor_from_buf` `:407` |
//!
//! # What the sweep has to reach, and why
//!
//! * **Both mask sources.** `COMPOUND_WEDGE` reads the baked codebook mask;
//!   `COMPOUND_DIFFWTD` reads `xd->seg_mask`, which plane 0 REBUILDS in place
//!   from the two scratch predictors and the chroma planes then consume. A
//!   test that only ran plane 0, or only ran wedge, would miss the ordering.
//! * **The unmasked arm.** `is_compound == false`, or a non-masked compound
//!   type, degenerates to a rectangular copy of the first predictor.
//! * **Subsampled planes.** `subw` / `subh` are DERIVED inside C from the
//!   `w`/`h` it was handed against the luma block's MI extent, so a chroma
//!   plane exercises a different arm of the blend than luma at the same
//!   `bsize`. 4:2:0, 4:2:2, 4:4:0 and 4:4:4 are all swept.
//! * **A destination WIDER than the block**, so a port that assumed a tight
//!   destination row fails.
//!
//! # Two C behaviours the sweep is bounded AWAY from, both measured here
//!
//! 1. **`build_wedge_inter_predictor_from_buf`'s unmasked arm faults at high
//!    bit depth.** Its masked arms take `ext_dst0` as a RAW `uint16_t *` and
//!    apply `CONVERT_TO_BYTEPTR` themselves (reconinter_enc.c:359-362, :375);
//!    the `else` arm instead applies `CONVERT_TO_SHORTPTR` to the same
//!    argument (:392), which shifts a real pointer LEFT and dereferences it.
//!    The arm is dead at both call sites — `av1_build_wedge_inter_predictor_from_buf`
//!    is only ever entered for a masked compound type — so this is unreachable
//!    upstream, and the sweep skips it rather than reproducing a crash. The
//!    port copies (which is what the lowbd arm does and what the name says).
//!    Measured: SIGBUS at the first `bd=10, COMPOUND_AVERAGE` cell.
//! 2. **`av1_build_compound_diffwtd_mask_neon` reads src1's second row at
//!    src0's stride** (`upstream/av1/common/arm/reconinter_neon.c:162`,
//!    the `w == 8` arm: `vld1_u8(src1 + src0_stride)`). It diverges from
//!    `av1_build_compound_diffwtd_mask_c` whenever the two strides differ.
//!    Every call site passes the same stride for both predictors, so the
//!    sweep does too — with a stride WIDER than the block, which still
//!    catches a port that assumed a tight buffer. Measured: an 8x8 DIFFWTD
//!    cell disagreed from row 1 onward when the strides were 12 and 14.

mod common;
use common::Rng;

use aom_dsp::inter::compound::DiffwtdMaskType;
use aom_encode::compound_type::{CompoundType, InterInterComp, Pixels};
use aom_encode::inter_pred_enc::{
    PixelsMut, WedgeFromBufCtx, av1_build_wedge_inter_predictor_from_buf, build_masked_compound,
    build_wedge_inter_predictor_from_buf,
};
use aom_sys_ref::{self as cref, RefPixels, RefPixelsMut};

const BLK_W: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
/// `mi_size_wide_log2` / `mi_size_high_log2` (`common_data.h`) — only used to
/// report which `(subw, subh)` arm a cell exercised.
const MI_SIZE_WIDE_LOG2: [usize; 22] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4,
];
const MI_SIZE_HIGH_LOG2: [usize; 22] = [
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2,
];
/// `MAX_SB_SQUARE`.
const MAX_SB_SQUARE: usize = 128 * 128;

/// The nine block sizes with a wedge codebook.
const WEDGE_BSIZES: [usize; 9] = [3, 4, 5, 6, 7, 8, 9, 18, 19];

/// A pixel plane in both the port's and the oracle's view.
struct Plane {
    lo: Vec<u8>,
    hi: Vec<u16>,
    hbd: bool,
}

impl Plane {
    fn random(rng: &mut Rng, n: usize, bd: u8) -> Self {
        if bd == 8 {
            Plane {
                lo: (0..n).map(|_| (rng.next() % 256) as u8).collect(),
                hi: Vec::new(),
                hbd: false,
            }
        } else {
            let maxv = (1u32 << bd) - 1;
            Plane {
                lo: Vec::new(),
                hi: (0..n)
                    .map(|_| (rng.next() as u32 % (maxv + 1)) as u16)
                    .collect(),
                hbd: true,
            }
        }
    }
    fn zeros(n: usize, bd: u8) -> Self {
        if bd == 8 {
            Plane {
                lo: vec![0u8; n],
                hi: Vec::new(),
                hbd: false,
            }
        } else {
            Plane {
                lo: Vec::new(),
                hi: vec![0u16; n],
                hbd: true,
            }
        }
    }
    fn port(&self) -> Pixels<'_> {
        if self.hbd {
            Pixels::High(&self.hi)
        } else {
            Pixels::Low(&self.lo)
        }
    }
    fn port_mut(&mut self) -> PixelsMut<'_> {
        if self.hbd {
            PixelsMut::High(&mut self.hi)
        } else {
            PixelsMut::Low(&mut self.lo)
        }
    }
    fn cref(&self) -> RefPixels<'_> {
        if self.hbd {
            RefPixels::High(&self.hi)
        } else {
            RefPixels::Low(&self.lo)
        }
    }
    fn cref_mut(&mut self) -> RefPixelsMut<'_> {
        if self.hbd {
            RefPixelsMut::High(&mut self.hi)
        } else {
            RefPixelsMut::Low(&mut self.lo)
        }
    }
    fn clone_of(&self) -> Self {
        Plane {
            lo: self.lo.clone(),
            hi: self.hi.clone(),
            hbd: self.hbd,
        }
    }
    fn same_as(&self, other: &Self) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }
}

fn mask_type_i32(m: DiffwtdMaskType) -> i32 {
    match m {
        DiffwtdMaskType::Diffwtd38 => 0,
        DiffwtdMaskType::Diffwtd38Inv => 1,
    }
}

fn comp_meta(c: &InterInterComp) -> [i32; 4] {
    [
        c.wedge_index as i32,
        c.wedge_sign as i32,
        mask_type_i32(c.mask_type),
        c.ty.index() as i32,
    ]
}

#[test]
fn shim_reports_max_sb_square() {
    assert_eq!(cref::ref_rie_max_sb_square() as usize, MAX_SB_SQUARE);
}

#[test]
fn build_masked_compound_matches_c() {
    let mut rng = Rng(0x5EED_0D01);
    let mut arms = [0usize; 4]; // (subw, subh) combinations reached
    for bsize in 0..22usize {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        // The four plane shapes the encoder produces: 4:4:4, 4:2:2, 4:4:0 and
        // 4:2:0. C derives (subw, subh) from these dimensions alone — but the
        // dimensions themselves must be ones `get_plane_block_size` can
        // return. Halving `bw` freely reaches w == 2, which no plane block
        // has: `aom_blend_a64_mask`'s SIMD tiers are written for w >= 4 and
        // diverge from the C reference below that. (Measured: bsize=4X4 with
        // w=2 was the first failing cell before this bound.)
        for (ss_x, ss_y) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let plane_bsize = aom_dsp::entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
            if plane_bsize >= 22 {
                continue; // BLOCK_INVALID
            }
            let (w, h) = (BLK_W[plane_bsize], BLK_H[plane_bsize]);
            for bd in [8u8, 10, 12] {
                let src0_stride = w + 4;
                let src1_stride = w + 8;
                let dst_stride = w + 12;
                let src0 = Plane::random(&mut rng, src0_stride * h, bd);
                let src1 = Plane::random(&mut rng, src1_stride * h, bd);
                let mask: Vec<u8> = (0..bw * bh).map(|_| (rng.next() % 65) as u8).collect();

                let mut got = Plane::zeros(dst_stride * h, bd);
                build_masked_compound(
                    &mut got.port_mut(),
                    dst_stride,
                    src0.port(),
                    src0_stride,
                    src1.port(),
                    src1_stride,
                    &mask,
                    bsize,
                    h,
                    w,
                );
                let mut want = Plane::zeros(dst_stride * h, bd);
                cref::ref_rie_build_masked_compound(
                    i32::from(bd),
                    bsize as i32,
                    h as i32,
                    w as i32,
                    src0.cref(),
                    src0_stride as i32,
                    src1.cref(),
                    src1_stride as i32,
                    &mask,
                    &mut want.cref_mut(),
                    dst_stride as i32,
                );
                assert!(
                    got.same_as(&want),
                    "build_masked_compound(bsize={bsize}, w={w}, h={h}, bd={bd})"
                );
                // `subw`/`subh` as C derives them, so the coverage counter
                // reports the arm actually taken rather than the subsampling
                // that was asked for (they differ where a plane block size
                // clamps).
                let subw = usize::from((2 << MI_SIZE_WIDE_LOG2[bsize]) == w);
                let subh = usize::from((2 << MI_SIZE_HIGH_LOG2[bsize]) == h);
                arms[subw * 2 + subh] += 1;
            }
        }
    }
    for (i, hits) in arms.iter().enumerate() {
        assert!(*hits > 0, "the (subw,subh) arm {i} was never reached");
    }
}

#[test]
fn build_wedge_from_buf_plane_matches_c() {
    let mut rng = Rng(0x5EED_0D02);
    let mut masked_cases = 0usize;
    let mut copy_cases = 0usize;
    for bsize in WEDGE_BSIZES {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        for bd in [8u8, 10, 12] {
            for ty in CompoundType::ALL {
                for is_compound in [false, true] {
                    // The unmasked (`aom_highbd_convolve_copy`) arm is not
                    // exercised at high bit depth: it faults in C. See the
                    // module header.
                    if bd > 8 && !(is_compound && ty.is_masked()) {
                        continue;
                    }
                    // Plane 0 at full size, plane 1 at 4:2:0 — the second one
                    // consumes whatever seg_mask the first left. Plane 1's
                    // dimensions come from `get_plane_block_size`, never from
                    // halving: see the note in build_masked_compound_matches_c.
                    let chroma = aom_dsp::entropy::partition::get_plane_block_size(bsize, 1, 1);
                    let mut shapes = vec![(0usize, bw, bh)];
                    if chroma < 22 {
                        shapes.push((1, BLK_W[chroma], BLK_H[chroma]));
                    }
                    for (plane, w, h) in shapes {
                        // The two ext strides are EQUAL, as every call site
                        // makes them (`strides[1] = { bw }` is passed for both
                        // preds0 and preds1 in masked_compound_type_rd and in
                        // av1_compound_type_rd's wedge arm). Not tight, so a
                        // port that assumed `bw` still fails — see the note in
                        // the module header about why they must not DIFFER.
                        let e0s = w + 4;
                        let e1s = e0s;
                        let dst_stride = w + 10;
                        let dst_rows = h + 2;
                        let ext0 = Plane::random(&mut rng, e0s * h, bd);
                        let ext1 = Plane::random(&mut rng, e1s * h, bd);
                        let comp = InterInterComp {
                            wedge_index: (rng.next() % 16) as usize,
                            wedge_sign: (rng.next() % 2) as usize,
                            mask_type: if rng.next() % 2 == 0 {
                                DiffwtdMaskType::Diffwtd38
                            } else {
                                DiffwtdMaskType::Diffwtd38Inv
                            },
                            ty,
                        };
                        let ctx = WedgeFromBufCtx {
                            bsize,
                            is_compound,
                            comp,
                            bd,
                        };

                        // Both sides start from the SAME seg_mask, so the
                        // plane-0 rebuild is compared as well as consumed.
                        let seed: Vec<u8> = (0..2 * MAX_SB_SQUARE)
                            .map(|_| (rng.next() % 65) as u8)
                            .collect();
                        let mut port_seg = seed.clone();
                        let mut c_seg = seed;

                        let mut got = Plane::zeros(dst_stride * dst_rows, bd);
                        build_wedge_inter_predictor_from_buf(
                            &ctx,
                            plane,
                            &mut got.port_mut(),
                            0,
                            dst_stride,
                            w,
                            h,
                            ext0.port(),
                            e0s,
                            ext1.port(),
                            e1s,
                            &mut port_seg,
                        );
                        let mut want = Plane::zeros(dst_stride * dst_rows, bd);
                        cref::ref_rie_build_wedge_from_buf_plane(
                            i32::from(bd),
                            bsize as i32,
                            is_compound,
                            &comp_meta(&comp),
                            plane as i32,
                            0,
                            0,
                            w as i32,
                            h as i32,
                            ext0.cref(),
                            e0s as i32,
                            ext1.cref(),
                            e1s as i32,
                            &mut want.cref_mut(),
                            dst_stride as i32,
                            dst_rows as i32,
                            &mut c_seg,
                        );
                        assert!(
                            got.same_as(&want),
                            "build_wedge_from_buf(bsize={bsize}, bd={bd}, ty={ty:?}, \
                             compound={is_compound}, plane={plane})"
                        );
                        // The port only owns the bw*bh prefix of seg_mask; C's
                        // rebuild writes exactly that much too.
                        assert_eq!(
                            port_seg[..bw * bh],
                            c_seg[..bw * bh],
                            "seg_mask after (bsize={bsize}, ty={ty:?}, plane={plane})"
                        );
                        if is_compound && ty.is_masked() {
                            masked_cases += 1;
                        } else {
                            copy_cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(masked_cases > 0, "the masked arm was never reached");
    assert!(copy_cases > 0, "the convolve-copy arm was never reached");
}

#[test]
fn av1_build_wedge_from_buf_matches_c() {
    let mut rng = Rng(0x5EED_0D03);
    let mut subsampled = 0usize;
    for bsize in WEDGE_BSIZES {
        for bd in [8u8, 10, 12] {
            for ty in [
                CompoundType::Wedge,
                CompoundType::DiffWtd,
                CompoundType::Average,
            ] {
                // Same reason as in build_wedge_from_buf_plane_matches_c: at
                // high bit depth C's unmasked arm faults, so only the masked
                // types are compared there.
                if bd > 8 && !ty.is_masked() {
                    continue;
                }
                for (ss_x, ss_y) in [(false, false), (true, true), (true, false), (false, true)] {
                    let subs = [(false, false), (ss_x, ss_y), (ss_x, ss_y)];
                    // `av1_ss_size_lookup` has genuine BLOCK_INVALID cells —
                    // e.g. 4:2:2 (ss_x=1, ss_y=0) of an 8X16 luma block
                    // (common_data.c:24). C would index `block_size_wide[255]`
                    // there, so neither side may be driven with one.
                    // (Measured: that was the SIGBUS this test opened with.)
                    if subs.iter().any(|&(x, y)| {
                        aom_dsp::entropy::partition::get_plane_block_size(
                            bsize,
                            usize::from(x),
                            usize::from(y),
                        ) >= 22
                    }) {
                        continue;
                    }
                    // Every plane gets a stride wider than its block and a
                    // slot in one packed allocation, as the shim expects.
                    let mut dims = [(0usize, 0usize); 3];
                    let mut strides = [0i32; 3];
                    let mut offs = [0i32; 3];
                    let mut bytes = [0i32; 3];
                    let mut total = 0usize;
                    for (p, item) in dims.iter_mut().enumerate() {
                        let pb = aom_dsp::entropy::partition::get_plane_block_size(
                            bsize,
                            usize::from(subs[p].0),
                            usize::from(subs[p].1),
                        );
                        let (w, h) = (BLK_W[pb], BLK_H[pb]);
                        *item = (w, h);
                        strides[p] = (w + 8) as i32;
                        offs[p] = total as i32;
                        bytes[p] = ((w + 8) * (h + 2)) as i32;
                        total += (w + 8) * (h + 2);
                    }
                    let ext0 = Plane::random(&mut rng, total, bd);
                    let ext1 = Plane::random(&mut rng, total, bd);
                    let comp = InterInterComp {
                        wedge_index: (rng.next() % 16) as usize,
                        wedge_sign: (rng.next() % 2) as usize,
                        mask_type: DiffwtdMaskType::Diffwtd38Inv,
                        ty,
                    };
                    let ctx = WedgeFromBufCtx {
                        bsize,
                        is_compound: true,
                        comp,
                        bd,
                    };
                    let seed: Vec<u8> = (0..2 * MAX_SB_SQUARE)
                        .map(|_| (rng.next() % 65) as u8)
                        .collect();
                    let mut port_seg = seed.clone();
                    let mut c_seg = seed;

                    let mut got = Plane::zeros(total, bd);
                    {
                        // Split the packed destination into three disjoint
                        // per-plane views, as C's `pd[plane].dst.buf` does.
                        let mut views: Vec<PixelsMut<'_>> = Vec::with_capacity(3);
                        if got.hbd {
                            let mut rest: &mut [u16] = &mut got.hi;
                            for p in 0..3 {
                                let (a, b) = rest.split_at_mut(bytes[p] as usize);
                                views.push(PixelsMut::High(a));
                                rest = b;
                            }
                        } else {
                            let mut rest: &mut [u8] = &mut got.lo;
                            for p in 0..3 {
                                let (a, b) = rest.split_at_mut(bytes[p] as usize);
                                views.push(PixelsMut::Low(a));
                                rest = b;
                            }
                        }
                        let e0: Vec<Pixels<'_>> = (0..3)
                            .map(|p| match ext0.port() {
                                Pixels::Low(s) => Pixels::Low(&s[offs[p] as usize..]),
                                Pixels::High(s) => Pixels::High(&s[offs[p] as usize..]),
                            })
                            .collect();
                        let e1: Vec<Pixels<'_>> = (0..3)
                            .map(|p| match ext1.port() {
                                Pixels::Low(s) => Pixels::Low(&s[offs[p] as usize..]),
                                Pixels::High(s) => Pixels::High(&s[offs[p] as usize..]),
                            })
                            .collect();
                        let su: Vec<(bool, bool)> = subs.to_vec();
                        let st: Vec<usize> = strides.iter().map(|&v| v as usize).collect();
                        av1_build_wedge_inter_predictor_from_buf(
                            &ctx,
                            0,
                            2,
                            &su,
                            &mut views,
                            &st,
                            &e0,
                            &st,
                            &e1,
                            &st,
                            &mut port_seg,
                        );
                    }

                    // The oracle destination starts from the same zeros the
                    // port did, so any byte NEITHER side wrote is compared too.
                    let mut want = Plane::zeros(total, bd);
                    let ss_flat = [
                        0,
                        0,
                        i32::from(ss_x),
                        i32::from(ss_y),
                        i32::from(ss_x),
                        i32::from(ss_y),
                    ];
                    cref::ref_rie_build_wedge_from_buf(
                        i32::from(bd),
                        bsize as i32,
                        true,
                        &comp_meta(&comp),
                        0,
                        2,
                        &ss_flat,
                        &offs,
                        &bytes,
                        ext0.cref(),
                        &strides,
                        ext1.cref(),
                        &strides,
                        &mut want.cref_mut(),
                        &strides,
                        &mut c_seg,
                    );
                    assert!(
                        got.same_as(&want),
                        "av1_build_wedge_from_buf(bsize={bsize}, bd={bd}, ty={ty:?}, \
                         ss=({ss_x},{ss_y}))"
                    );
                    assert_eq!(
                        port_seg[..BLK_W[bsize] * BLK_H[bsize]],
                        c_seg[..BLK_W[bsize] * BLK_H[bsize]],
                        "seg_mask after the plane loop"
                    );
                    if ss_x || ss_y {
                        subsampled += 1;
                    }
                }
            }
        }
    }
    assert!(
        subsampled > 0,
        "no subsampled plane configuration was reached"
    );
}
