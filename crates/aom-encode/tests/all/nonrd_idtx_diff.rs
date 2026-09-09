//! Differential harness for `aom_encode::nonrd_idtx` vs the REAL exported C
//! libaom v3.14.1. **Tier 1** — `av1_block_yrd_idtx` is exported, and
//! `shim/nonrd_idtx_shim.c` only assembles the `MACROBLOCK` it reads.
//!
//! | test | C oracle |
//! |---|---|
//! | `fast_idtx_scan_tables_match_c` | `av1_fast_idtx_{,i}scan_{4x4,8x8,16x16}` (nonrd_opt.h) |
//! | `fast_idtx_scans_are_permutations_and_inverses` | the invariant the tables' own C comments assert |
//! | `block_yrd_idtx_matches_c` | `av1_block_yrd_idtx` (nonrd_opt.c:380) |
//! | `block_yrd_idtx_skippable_path_matches_c` | ditto, forced onto the `sse < INT64_MAX && skippable` return |
//!
//! # What bounds the generators
//! * `tx_size` is `TX_4X4`, `TX_8X8` or `TX_16X16` only. C asserts on
//!   `TX_32X32` ("Not used") and `TX_64X64` ("Not implemented"), so a sweep
//!   including them would be testing a call the encoder cannot make.
//! * The quantizer rows are real ones: `round_fp_QTX` / `quant_fp_QTX` /
//!   `dequant_QTX` are built from `av1_ac_quant_qtx` / `av1_dc_quant_qtx` the
//!   way `av1_init_plane_quantizers` does, not drawn at random -- a random
//!   `quant_fp` can make every coefficient zero and the comparison vacuous.
//! * Pixels are 8-bit: C's own doc comment says this function is lowbd-only
//!   ("called in real-time mode for now, which sets high bit depth to 0").
//! * `mb_to_right_edge` / `mb_to_bottom_edge` are swept at 0 (interior) and at
//!   the negative values a block hanging off the frame edge produces, because
//!   they are the only thing that makes `max_blocks_wide/high` differ from the
//!   full block.

use aom_encode::nonrd_idtx::{
    AV1_FAST_IDTX_ISCAN_4X4, AV1_FAST_IDTX_ISCAN_8X8, AV1_FAST_IDTX_ISCAN_16X16,
    AV1_FAST_IDTX_SCAN_4X4, AV1_FAST_IDTX_SCAN_8X8, AV1_FAST_IDTX_SCAN_16X16, block_yrd_idtx,
    scale_square_buf_vals,
};
use aom_sys_ref::{ref_nrd_block_yrd_idtx, ref_nrd_fast_idtx_scan};

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

#[test]
fn fast_idtx_scan_tables_match_c() {
    for (tx_size, fwd, inv) in [
        (
            0usize,
            &AV1_FAST_IDTX_SCAN_4X4[..],
            &AV1_FAST_IDTX_ISCAN_4X4[..],
        ),
        (1, &AV1_FAST_IDTX_SCAN_8X8[..], &AV1_FAST_IDTX_ISCAN_8X8[..]),
        (
            2,
            &AV1_FAST_IDTX_SCAN_16X16[..],
            &AV1_FAST_IDTX_ISCAN_16X16[..],
        ),
    ] {
        assert_eq!(
            fwd,
            &ref_nrd_fast_idtx_scan(tx_size, false)[..],
            "scan {tx_size}"
        );
        assert_eq!(
            inv,
            &ref_nrd_fast_idtx_scan(tx_size, true)[..],
            "iscan {tx_size}"
        );
    }
}

#[test]
fn fast_idtx_scans_are_permutations_and_inverses() {
    // nonrd_opt.h says of every pair "Must be used together with <the other>".
    // That is only meaningful if they really are inverse permutations, and if
    // they were not, a table typo would still pass the equality test above
    // whenever the SAME typo were transcribed. This checks the property.
    for (fwd, inv) in [
        (&AV1_FAST_IDTX_SCAN_4X4[..], &AV1_FAST_IDTX_ISCAN_4X4[..]),
        (&AV1_FAST_IDTX_SCAN_8X8[..], &AV1_FAST_IDTX_ISCAN_8X8[..]),
        (
            &AV1_FAST_IDTX_SCAN_16X16[..],
            &AV1_FAST_IDTX_ISCAN_16X16[..],
        ),
    ] {
        let n = fwd.len();
        let mut seen = vec![false; n];
        for &v in fwd {
            let v = v as usize;
            assert!(v < n && !seen[v], "scan of {n} is not a permutation at {v}");
            seen[v] = true;
        }
        for (i, &pos) in fwd.iter().enumerate() {
            assert_eq!(
                inv[pos as usize] as usize, i,
                "iscan is not the inverse at {i} (n = {n})"
            );
        }
    }
}

#[test]
fn scale_square_buf_vals_is_a_strided_copy_times_eight() {
    // The only thing here that is not obvious is the STRIDE change: the source
    // is a row of the whole block's residual, the destination is packed at
    // tx_width. A port that reused src_stride on both sides passes at
    // tx_width == src_stride and fails everywhere else, so both are swept.
    let mut rng = Rng::new(0x5CA1_E000);
    for &tx_width in &[4usize, 8, 16] {
        for &src_stride in &[tx_width, tx_width + 4, 64] {
            let src: Vec<i16> = (0..src_stride * tx_width)
                .map(|_| rng.below(1 << 10) as i16 - 512)
                .collect();
            let mut dst = vec![0i16; tx_width * tx_width];
            scale_square_buf_vals(&mut dst, tx_width, &src, src_stride);
            for y in 0..tx_width {
                for x in 0..tx_width {
                    assert_eq!(
                        dst[y * tx_width + x],
                        src[y * src_stride + x] * 8,
                        "tx {tx_width} stride {src_stride} at ({x},{y})"
                    );
                }
            }
        }
    }
}

/// `av1_ac_quant_qtx` / `av1_dc_quant_qtx`-derived FP quantizer rows, built
/// the way `av1_init_plane_quantizers` (av1_quantize.c) does for the
/// low-precision path: lane 0 is DC, lane 1 is AC, and each row is
/// `[dc, ac, ac, ...]`.
fn quant_rows(qindex: i32) -> ([i16; 8], [i16; 8], [i16; 8]) {
    let dc = aom_dsp::quant::av1_dc_quant_qtx(qindex, 0, 8);
    let ac = aom_dsp::quant::av1_ac_quant_qtx(qindex, 0, 8);
    let mut round_fp = [0i16; 8];
    let mut quant_fp = [0i16; 8];
    let mut dequant = [0i16; 8];
    for lane in 0..2 {
        let q = if lane == 0 { dc } else { ac };
        // av1_quantize.c's invert_quant / ROUND_POWER_OF_TWO(q * 48, 7).
        let (quant, _shift) = {
            let l = 15 - (i32::from(q).leading_zeros() as i32 - 16).max(0);
            let _ = l;
            // The exact inversion is not needed: any (quant, round, dequant)
            // triple the quantizer accepts exercises the same code, and the
            // ORACLE is handed the same triple. What matters is that the
            // values are in the real range, which these are.
            (
                ((1i32 << 16) / i32::from(q).max(1)).min(i32::from(i16::MAX)) as i16,
                0,
            )
        };
        quant_fp[lane] = quant;
        round_fp[lane] = ((i32::from(q) * 48 + 64) >> 7) as i16;
        dequant[lane] = q;
    }
    for lane in 2..8 {
        quant_fp[lane] = quant_fp[1];
        round_fp[lane] = round_fp[1];
        dequant[lane] = dequant[1];
    }
    (round_fp, quant_fp, dequant)
}

/// `mi_size_wide` / `mi_size_high` for the square block sizes swept here.
fn mi_size(bsize: i32) -> (usize, usize) {
    match bsize {
        3 => (2, 2),    // BLOCK_8X8
        6 => (4, 4),    // BLOCK_16X16
        9 => (8, 8),    // BLOCK_32X32
        12 => (16, 16), // BLOCK_64X64
        _ => unreachable!("only square sizes are swept"),
    }
}

#[test]
fn block_yrd_idtx_matches_c() {
    let mut rng = Rng::new(0x1D_7000);
    let mut checked = 0usize;
    let (mut skippable_seen, mut coded_seen) = (0usize, 0usize);

    for &bsize in &[3i32, 6, 9, 12] {
        let (w4, h4) = mi_size(bsize);
        let (bw, bh) = (4 * w4, 4 * h4);
        // TX_4X4 / TX_8X8 / TX_16X16, capped by the block itself.
        for tx_size in 0..3usize {
            if (1usize << tx_size) > w4 {
                continue;
            }
            for &qindex in &[8i32, 60, 140, 220] {
                let (round_fp, quant_fp, dequant) = quant_rows(qindex);
                for trial in 0..4 {
                    let src_stride = bw + 16;
                    let pred_stride = bw + 8;
                    // A residual that spans zero (identical) through large.
                    let src: Vec<u8> = (0..src_stride * bh).map(|_| rng.below(256) as u8).collect();
                    let pred: Vec<u8> = (0..pred_stride * bh)
                        .map(|i| match trial {
                            // trial 0: identical -> every coefficient zero ->
                            // the skippable arm.
                            0 => src[(i / pred_stride) * src_stride + (i % pred_stride)],
                            1 => src[(i / pred_stride) * src_stride + (i % pred_stride)]
                                .wrapping_add(rng.below(4) as u8),
                            _ => rng.below(256) as u8,
                        })
                        .collect();

                    // Interior, and both edge clamps a block hanging off the
                    // frame produces. C: max_blocks = num_4x4 + (edge >> 5).
                    for &(right, bottom) in &[(0i32, 0i32), (-64, 0), (0, -64), (-128, -64)] {
                        for &sse_in in &[i64::MAX, 0i64, 1 << 20] {
                            let want = ref_nrd_block_yrd_idtx(
                                &src,
                                src_stride,
                                &pred,
                                pred_stride,
                                bsize,
                                tx_size as i32,
                                right,
                                bottom,
                                &round_fp,
                                &quant_fp,
                                &dequant,
                                sse_in,
                            );

                            // The port takes the residual directly; build it
                            // the way aom_subtract_block does.
                            let mut diff = vec![0i16; bw * bh];
                            for y in 0..bh {
                                for x in 0..bw {
                                    diff[y * bw + x] = i16::from(src[y * src_stride + x])
                                        - i16::from(pred[y * pred_stride + x]);
                                }
                            }
                            let max_w = (w4 as i32 + if right >= 0 { 0 } else { right >> 5 }).max(0)
                                as usize;
                            let max_h = (h4 as i32 + if bottom >= 0 { 0 } else { bottom >> 5 })
                                .max(0) as usize;
                            let got = block_yrd_idtx(
                                &diff, w4, max_w, max_h, tx_size, sse_in, &round_fp, &quant_fp,
                                &dequant,
                            );

                            assert_eq!(
                                got.skippable, want.skippable,
                                "skippable, bsize {bsize} tx {tx_size} q {qindex} trial {trial} edges ({right},{bottom}) sse {sse_in}"
                            );
                            assert_eq!(
                                got.sse, want.sse,
                                "sse, bsize {bsize} tx {tx_size} q {qindex} trial {trial}"
                            );
                            assert_eq!(
                                got.dist, want.dist,
                                "dist, bsize {bsize} tx {tx_size} q {qindex} trial {trial}"
                            );
                            assert_eq!(
                                got.rate, want.rate,
                                "rate, bsize {bsize} tx {tx_size} q {qindex} trial {trial}"
                            );
                            if want.skippable {
                                skippable_seen += 1;
                            } else {
                                coded_seen += 1;
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 400, "only {checked} cells - the sweep shrank");
    assert!(
        skippable_seen > 20,
        "only {skippable_seen} skippable blocks"
    );
    assert!(coded_seen > 100, "only {coded_seen} coded blocks");
}

#[test]
fn block_yrd_idtx_skippable_path_matches_c() {
    // The `sse < INT64_MAX && skippable` return REPLACES the accumulated
    // distortion with the rescaled sse and zeroes the rate. A port that
    // treated `sse` as an output rather than an in/out agrees everywhere else
    // and diverges here, so it is pinned on its own with an identical
    // predictor (guaranteed skippable) and a finite input sse.
    let (bsize, tx_size) = (6i32, 1usize); // BLOCK_16X16, TX_8X8
    let (w4, h4) = mi_size(bsize);
    let (bw, bh) = (4 * w4, 4 * h4);
    let (round_fp, quant_fp, dequant) = quant_rows(100);
    let src: Vec<u8> = (0..(bw + 16) * bh).map(|i| (i % 251) as u8).collect();
    let mut pred = vec![0u8; (bw + 8) * bh];
    for y in 0..bh {
        for x in 0..bw + 8 {
            pred[y * (bw + 8) + x] = src[y * (bw + 16) + x];
        }
    }
    for &sse_in in &[0i64, 1, 1 << 10, 1 << 40, i64::MAX] {
        let want = ref_nrd_block_yrd_idtx(
            &src,
            bw + 16,
            &pred,
            bw + 8,
            bsize,
            tx_size as i32,
            0,
            0,
            &round_fp,
            &quant_fp,
            &dequant,
            sse_in,
        );
        assert!(
            want.skippable,
            "the identical-predictor construction failed"
        );
        let diff = vec![0i16; bw * bh];
        let got = block_yrd_idtx(
            &diff, w4, w4, h4, tx_size, sse_in, &round_fp, &quant_fp, &dequant,
        );
        assert_eq!(got.sse, want.sse, "sse at sse_in {sse_in}");
        assert_eq!(got.dist, want.dist, "dist at sse_in {sse_in}");
        assert_eq!(got.rate, want.rate, "rate at sse_in {sse_in}");
    }
}

// ---------------------------------------------------------------------------
// av1_model_rd_for_sb_uv — TIER 1, the symbol is exported.
// ---------------------------------------------------------------------------

use aom_encode::nonrd_idtx::{UvPlane, model_rd_for_sb_uv};
use aom_sys_ref::ref_nrd_model_rd_for_sb_uv;

/// The (bsize, width, height) triples the chroma model is called at. C indexes
/// `num_pels_log2_lookup[plane_bsize]` and `fn_ptr[plane_bsize].vf`, so the
/// dimensions and the enum value have to agree.
const UV_BSIZES: [(i32, usize, usize); 8] = [
    (0, 4, 4),      // BLOCK_4X4
    (3, 8, 8),      // BLOCK_8X8
    (6, 16, 16),    // BLOCK_16X16
    (9, 32, 32),    // BLOCK_32X32
    (12, 64, 64),   // BLOCK_64X64
    (7, 16, 32),    // BLOCK_16X32
    (8, 32, 16),    // BLOCK_32X16
    (15, 128, 128), // BLOCK_128X128
];

#[test]
fn model_rd_for_sb_uv_matches_c() {
    let mut rng = Rng::new(0x0B_5B_0001);
    let (mut skipped, mut coded, mut clobbered) = (0usize, 0usize, 0usize);
    let mut checked = 0usize;

    for &(bsize, w, h) in &UV_BSIZES {
        for _ in 0..24 {
            let strides = [w + 16, w + 8, w + 24];
            let mut src_bufs: Vec<Vec<u8>> = Vec::new();
            let mut dst_bufs: Vec<Vec<u8>> = Vec::new();
            // The prediction tracks the source, with a spread that sweeps a
            // near-perfect match through a poor one -- the whole model is a
            // function of the residual, so an independent draw only ever
            // exercises the high-variance end.
            let spread = 1u32 << rng.below(8);
            for p in 0..3 {
                let s: Vec<u8> = (0..strides[p] * h).map(|_| rng.below(256) as u8).collect();
                let d: Vec<u8> = s
                    .iter()
                    .map(|&v| {
                        (i32::from(v) + rng.below(2 * spread + 1) as i32 - spread as i32)
                            .clamp(0, 255) as u8
                    })
                    .collect();
                src_bufs.push(s);
                dst_bufs.push(d);
            }

            // Real quantizer steps, and a colour-sensitivity mask that
            // includes the all-clear case (where every plane is skipped and
            // tot_sse stays 0).
            let mut dq_dc = [0i16; 3];
            let mut dq_ac = [0i16; 3];
            let mut cs = [0i32; 3];
            for p in 0..3 {
                let q = rng.below(256) as i32;
                dq_dc[p] = aom_dsp::quant::av1_dc_quant_qtx(q, 0, 8);
                dq_ac[p] = aom_dsp::quant::av1_ac_quant_qtx(q, 0, 8);
                cs[p] = i32::from(rng.below(4) != 0);
            }
            let rdmult = 1 + rng.below(1 << 14) as i32;

            for &(start, stop) in &[(1i32, 2i32), (1, 1), (2, 2)] {
                let src_refs: [&[u8]; 3] = [&src_bufs[0], &src_bufs[1], &src_bufs[2]];
                let dst_refs: [&[u8]; 3] = [&dst_bufs[0], &dst_bufs[1], &dst_bufs[2]];
                let want = ref_nrd_model_rd_for_sb_uv(
                    bsize,
                    start,
                    stop,
                    rdmult,
                    &cs,
                    &dq_dc,
                    &dq_ac,
                    &src_refs,
                    &[strides[0] as i32, strides[1] as i32, strides[2] as i32],
                    &dst_refs,
                    &[strides[0] as i32, strides[1] as i32, strides[2] as i32],
                    h as i32,
                );

                let planes: Vec<UvPlane<'_>> = (0..3)
                    .map(|p| UvPlane {
                        src: &src_bufs[p],
                        src_stride: strides[p],
                        dst: &dst_bufs[p],
                        dst_stride: strides[p],
                        dequant_dc: dq_dc[p] as u32,
                        dequant_ac: dq_ac[p] as u32,
                        color_sensitive: cs[p] != 0,
                    })
                    .collect();
                let got = model_rd_for_sb_uv(
                    &planes,
                    bsize as usize,
                    start as usize,
                    stop as usize,
                    rdmult,
                    h,
                    w,
                );

                assert_eq!(
                    got.tot_sse, want.tot_sse,
                    "tot_sse bsize {bsize} {start}..={stop}"
                );
                assert_eq!(got.rate, want.rate, "rate bsize {bsize} {start}..={stop}");
                assert_eq!(got.dist, want.dist, "dist bsize {bsize} {start}..={stop}");
                assert_eq!(
                    got.skip_txfm, want.skip_txfm,
                    "skip_txfm bsize {bsize} {start}..={stop}"
                );
                if want.skip_txfm {
                    skipped += 1;
                    // The RD clobber is the case where skip is set AND the
                    // distortion was replaced by the sse -- distinct from the
                    // zero-rate case.
                    if want.rate == 0 && want.dist == want.tot_sse << 4 && want.tot_sse != 0 {
                        clobbered += 1;
                    }
                } else {
                    coded += 1;
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 500, "only {checked} cells");
    assert!(
        skipped > 50 && coded > 50,
        "one skip_txfm arm never fired: {skipped}/{coded}"
    );
    assert!(
        clobbered > 20,
        "the RD comparison never clobbered rate and dist ({clobbered}) -- that arm is untested"
    );
}
