//! `av1_block_yrd_idtx` (`av1/encoder/nonrd_opt.c:380`) — the identity-transform
//! arm of the non-RD RD estimate, and the fast-IDTX scan orders it runs on.
//!
//! The nonrd inter search scores an IDTX candidate with this instead of the
//! Hadamard estimate ([`crate::nonrd_pickmode::block_yrd_lowbd`]): the
//! "transform" is a multiply by 8, and the quantizer then runs over a scan
//! chosen to match the coefficient ORDER that multiply produces.
//!
//! | Rust | C |
//! |---|---|
//! | [`scale_square_buf_vals`] | `scale_square_buf_vals` (:334, static) |
//! | [`block_yrd_idtx`] | `av1_block_yrd_idtx` (:380, exported) |
//! | `AV1_FAST_IDTX_*` | `av1_fast_idtx_{,i}scan_{4x4,8x8,16x16}` (nonrd_opt.h:355-427) |
//!
//! # Why IDTX gets its OWN scans
//! C says it at nonrd_opt.h:349: for entropy coding IDTX shares the ordinary
//! 2-D scan orders, but the fastest way to compute IDTX skips the transposes,
//! so its coefficients come out transposed relative to the entropy-coding
//! layout. These tables are the substitute. They are NOT interchangeable with
//! `default_scan_*` and using one in place of the other changes every eob.
//!
//! # `sse` is an INPUT here, unlike in `av1_block_yrd`
//! `av1_block_yrd_idtx` zeroes `dist` and `rate` but NOT `sse`: it reads the
//! value the caller left there and, if it is below `INT64_MAX`, rescales it
//! (`sse = (sse << 6) >> 2`) and — when the block came out skippable — returns
//! that as the distortion, discarding the accumulated one. The port takes
//! `sse` as an argument and returns the updated value for exactly that reason;
//! a port that treated `sse` as an output would agree on every non-skippable
//! block and diverge on every skippable one.
//!
//! # Differential coverage
//! `tests/nonrd_idtx_diff.rs` — tier 1 against the real exported
//! `av1_block_yrd_idtx`, plus a tier-1 equality check on all six scan tables.

use crate::nonrd_pickmode::{AV1_PROB_COST_SHIFT, block_error_lp, get_msb, quantize_lp, satd_lp};

/// `av1_fast_idtx_scan_4x4` (`av1/encoder/nonrd_opt.h:355`).
pub static AV1_FAST_IDTX_SCAN_4X4: [i16; 16] =
    [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// `av1_fast_idtx_iscan_4x4` (nonrd_opt.h:359) — the inverse of
/// [`AV1_FAST_IDTX_SCAN_4X4`]; C's comment insists the two travel together.
pub static AV1_FAST_IDTX_ISCAN_4X4: [i16; 16] =
    [0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15];

/// `av1_fast_idtx_scan_8x8` (nonrd_opt.h:369).
pub static AV1_FAST_IDTX_SCAN_8X8: [i16; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// `av1_fast_idtx_iscan_8x8` (nonrd_opt.h:376).
pub static AV1_FAST_IDTX_ISCAN_8X8: [i16; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

/// `av1_fast_idtx_scan_16x16` (nonrd_opt.h:390).
pub static AV1_FAST_IDTX_SCAN_16X16: [i16; 256] = [
    0, 1, 16, 32, 17, 2, 3, 18, 33, 48, 64, 49, 34, 19, 4, 5, 20, 35, 50, 65, 80, 96, 81, 66, 51,
    36, 21, 6, 7, 22, 37, 52, 67, 82, 97, 112, 128, 113, 98, 83, 68, 53, 38, 23, 8, 9, 24, 39, 54,
    69, 84, 99, 114, 129, 144, 160, 145, 130, 115, 100, 85, 70, 55, 40, 25, 10, 11, 26, 41, 56, 71,
    86, 101, 116, 131, 146, 161, 176, 192, 177, 162, 147, 132, 117, 102, 87, 72, 57, 42, 27, 12,
    13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 224, 209, 194, 179, 164, 149,
    134, 119, 104, 89, 74, 59, 44, 29, 14, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
    195, 210, 225, 240, 241, 226, 211, 196, 181, 166, 151, 136, 121, 106, 91, 76, 61, 46, 31, 47,
    62, 77, 92, 107, 122, 137, 152, 167, 182, 197, 212, 227, 242, 243, 228, 213, 198, 183, 168,
    153, 138, 123, 108, 93, 78, 63, 79, 94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 245,
    230, 215, 200, 185, 170, 155, 140, 125, 110, 95, 111, 126, 141, 156, 171, 186, 201, 216, 231,
    246, 247, 232, 217, 202, 187, 172, 157, 142, 127, 143, 158, 173, 188, 203, 218, 233, 248, 249,
    234, 219, 204, 189, 174, 159, 175, 190, 205, 220, 235, 250, 251, 236, 221, 206, 191, 207, 222,
    237, 252, 253, 238, 223, 239, 254, 255,
];

/// `av1_fast_idtx_iscan_16x16` (nonrd_opt.h:409).
pub static AV1_FAST_IDTX_ISCAN_16X16: [i16; 256] = [
    0, 1, 5, 6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91, 119, 120, 2, 4, 7, 13, 16, 26, 29, 43, 46,
    64, 67, 89, 92, 118, 121, 150, 3, 8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93, 117, 122, 149,
    151, 9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94, 116, 123, 148, 152, 177, 10, 19, 23, 32, 40,
    49, 61, 70, 86, 95, 115, 124, 147, 153, 176, 178, 20, 22, 33, 39, 50, 60, 71, 85, 96, 114, 125,
    146, 154, 175, 179, 200, 21, 34, 38, 51, 59, 72, 84, 97, 113, 126, 145, 155, 174, 180, 199,
    201, 35, 37, 52, 58, 73, 83, 98, 112, 127, 144, 156, 173, 181, 198, 202, 219, 36, 53, 57, 74,
    82, 99, 111, 128, 143, 157, 172, 182, 197, 203, 218, 220, 54, 56, 75, 81, 100, 110, 129, 142,
    158, 171, 183, 196, 204, 217, 221, 234, 55, 76, 80, 101, 109, 130, 141, 159, 170, 184, 195,
    205, 216, 222, 233, 235, 77, 79, 102, 108, 131, 140, 160, 169, 185, 194, 206, 215, 223, 232,
    236, 245, 78, 103, 107, 132, 139, 161, 168, 186, 193, 207, 214, 224, 231, 237, 244, 246, 104,
    106, 133, 138, 162, 167, 187, 192, 208, 213, 225, 230, 238, 243, 247, 252, 105, 134, 137, 163,
    166, 188, 191, 209, 212, 226, 229, 239, 242, 248, 251, 253, 135, 136, 164, 165, 189, 190, 210,
    211, 227, 228, 240, 241, 249, 250, 254, 255,
];

/// The `(scan, iscan)` pair `av1_block_yrd_idtx` selects for `tx_size`.
///
/// C's `switch` asserts on `TX_64X64` ("Not implemented") and `TX_32X32`
/// ("Not used") and falls through to `TX_4X4` for everything else; the port
/// returns `None` for the two asserted sizes instead of reproducing an
/// assertion that is a contract, not a computation.
#[must_use]
pub fn fast_idtx_scan_order(tx_size: usize) -> Option<(&'static [i16], usize)> {
    match tx_size {
        // TX_4X4 — C's `default:` arm, with `assert(tx_size == TX_4X4)`.
        0 => Some((&AV1_FAST_IDTX_SCAN_4X4, 4)),
        1 => Some((&AV1_FAST_IDTX_SCAN_8X8, 8)),
        2 => Some((&AV1_FAST_IDTX_SCAN_16X16, 16)),
        // TX_32X32 ("Not used") and TX_64X64 ("Not implemented").
        _ => None,
    }
}

/// `scale_square_buf_vals` (nonrd_opt.c:334) — the whole IDTX "transform":
/// copy a `tx_width` square out of a strided residual, multiplied by 8.
///
/// C spells the same loop three times behind a macro so each instance
/// specialises on a constant `tx_width`; that is a codegen trick, not
/// semantics, so the port writes it once.
///
/// # Panics
/// If `dst` is shorter than `tx_width * tx_width` or `src` cannot supply
/// `tx_width` rows at `src_stride`.
pub fn scale_square_buf_vals(dst: &mut [i16], tx_width: usize, src: &[i16], src_stride: usize) {
    debug_assert!(
        matches!(tx_width, 4 | 8 | 16),
        "C asserts on any other width (nonrd_opt.c:352)"
    );
    for idy in 0..tx_width {
        let s = &src[idy * src_stride..][..tx_width];
        let d = &mut dst[idy * tx_width..][..tx_width];
        for (o, &v) in d.iter_mut().zip(s) {
            // `int16_t * 8` in C: the multiply is done in `int` and narrowed
            // back on assignment, so it WRAPS rather than saturating. The
            // encoder cannot reach the wrap — the residual is at most 9 bits
            // at bd8, and 511 * 8 = 4088 — but the port matches the width.
            *o = v.wrapping_mul(8);
        }
    }
}

/// The RD estimate `av1_block_yrd_idtx` produces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IdtxRd {
    /// `this_rdc->rate`, after C's two final shifts.
    pub rate: i32,
    /// `this_rdc->dist`.
    pub dist: i64,
    /// `this_rdc->sse`, possibly rescaled — see the module header.
    pub sse: i64,
    /// `*skippable` / `this_rdc->skip_txfm`.
    pub skippable: bool,
}

/// `av1_block_yrd_idtx` (nonrd_opt.c:380) — the LOWBD-only IDTX RD estimate.
///
/// C's own doc comment says it: "this function is only for low bit depth
/// encoding, since it is called in real-time mode for now, which sets high bit
/// depth to 0". There is no highbd arm to port.
///
/// `diff` is the residual for the whole block, stride `4 * num_4x4_w` — the
/// caller has already run `aom_subtract_block` over `pred_buf`.
/// `max_blocks_wide` / `max_blocks_high` are C's edge clamps
/// (`num_4x4 + (mb_to_edge >> 5)` when the edge is negative).
///
/// `sse_in` is `this_rdc->sse` on entry; see the module header on why it is an
/// input.
///
/// # Panics
/// If `tx_size` is `TX_32X32` or `TX_64X64`, which C asserts on.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn block_yrd_idtx(
    diff: &[i16],
    num_4x4_w: usize,
    max_blocks_wide: usize,
    max_blocks_high: usize,
    tx_size: usize,
    sse_in: i64,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
) -> IdtxRd {
    let (scan, tx_wd) =
        fast_idtx_scan_order(tx_size).expect("av1_block_yrd_idtx asserts on TX_32X32 and TX_64X64");
    let diff_stride = 4 * num_4x4_w;
    let block_step = 1usize << tx_size;
    // 4x4 units per sub-block: C's `step = 1 << (tx_size << 1)`.
    let step = 1usize << (tx_size << 1);
    let n_coeffs = tx_wd * tx_wd;

    let mut rate: i32 = 0;
    let mut dist: i64 = 0;
    let mut eob_cost: i32 = 0;
    let mut temp_skippable = true;

    let mut coeff = [0i16; 256];
    let mut qcoeff = [0i16; 256];
    let mut dqcoeff = [0i16; 256];

    let mut r = 0usize;
    while r < max_blocks_high {
        let mut c = 0usize;
        while c < max_blocks_wide {
            let src_diff = &diff[(r * diff_stride + c) * 4..];
            scale_square_buf_vals(&mut coeff[..n_coeffs], tx_wd, src_diff, diff_stride);
            let eob = quantize_lp(
                &coeff[..n_coeffs],
                n_coeffs,
                round_fp,
                quant_fp,
                &mut qcoeff,
                &mut dqcoeff,
                dequant,
                &scan[..n_coeffs],
            );
            // update_yrd_loop_vars (nonrd_opt.c:43).
            let ncoeffs = eob as usize;
            temp_skippable &= ncoeffs == 0;
            eob_cost += get_msb(ncoeffs as u32 + 1);
            if ncoeffs == 1 {
                rate += i32::from(qcoeff[0]).abs();
            } else if ncoeffs > 1 {
                rate += satd_lp(&qcoeff, step << 4);
            }
            dist += block_error_lp(&coeff, &dqcoeff, step << 4) >> 2;
            c += block_step;
        }
        r += block_step;
    }

    let mut sse = sse_in;
    if sse < i64::MAX {
        sse = (sse << 6) >> 2;
        if temp_skippable {
            // C writes `dist = 0` then immediately `dist = sse`; the first
            // store is dead and is not reproduced.
            return IdtxRd {
                rate: 0,
                dist: sse,
                sse,
                skippable: true,
            };
        }
    }
    // C's comment: "If skippable is set, rate gets clobbered later" — so the
    // early return above leaves `rate` at whatever the caller had, which for
    // the port is 0 (C zeroes it at entry and never adds to it before the
    // return, because a skippable block contributes no rate).
    let rate = (rate << (2 + AV1_PROB_COST_SHIFT)) + (eob_cost << AV1_PROB_COST_SHIFT);
    IdtxRd {
        rate,
        dist,
        sse,
        skippable: temp_skippable,
    }
}
