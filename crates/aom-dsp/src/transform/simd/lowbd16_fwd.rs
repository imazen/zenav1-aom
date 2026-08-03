//! i16-lane FORWARD transform column and row passes — 16 columns/rows per
//! vector, 2x the lane throughput of the [`super`] `i32x8` passes, for every
//! forward kernel whose input stays inside the bound `xtask/audit_i16_fwd.py`
//! proves.
//!
//! This is the forward twin of [`super::lowbd16`], and it is gated on a
//! DIFFERENT kind of argument, because the two directions are built
//! differently:
//!
//! * the INVERSE kernels `clamp_value(_, stage_range[i])` at every stage, so
//!   `lowbd16`'s contract is a DOMAIN statement — at bd8 every kernel value is
//!   either an i16 clamp output or a 17-bit `half_btf` transient, and the
//!   iadst/identity kernels fail that statement and stay on the i32 path;
//! * the FORWARD kernels carry NO clamp at all (`av1_fwd_txfm1d.c` has no range
//!   check in the production config), so nothing bounds a value except the
//!   input. The contract here is therefore a BOUND: for each kernel there is a
//!   largest `M*` such that `|input| <= M*` keeps EVERY value of the kernel
//!   inside i16, and on that domain the i16 lane arithmetic is bit-identical to
//!   the scalar i32 kernel.
//!
//! # `M*`, and why it is a proof rather than an estimate
//!
//! Each forward kernel is a fixed sequence of adds, negations and
//! `half_btf(w0,a,w1,b,bit)`, so every value is an exact integer linear form in
//! the inputs plus an accumulated rounding error:
//!
//! ```text
//! value = sum_i c_i * input_i + e,   |e| <= E   =>   |value| <= M * sum|c_i| + E
//! ```
//!
//! `xtask/audit_i16_fwd.py` propagates `(c, E)` exactly (all denominators are
//! powers of two, so the coefficients are exact `Fraction`s) and reports the
//! largest `M` for which that bound stays `<= i16::MAX` at every value. The
//! bound is also TIGHT — the sign vertex `input_i = M * sign(c_i)` attains
//! `M * sum|c_i|` — which matters: the loose triangle-inequality bound that
//! treats each butterfly operand as independently maximal is 1.5-2x larger and
//! would reject fdct32's column pass, which is provably safe.
//!
//! `sum|c_i|` comes out at exactly `N * sqrt(2) / 2` for the whole fdct family
//! (the DC row of an AV1 `fdctN`, which is `sqrt(2)` x the orthonormal DCT-II),
//! so `M*(fdctN) = floor(46340 / N)` — 11583, 5791, 2895, 1447, 723.
//!
//! At bd8 the driver's own chain is what has to fit inside those bounds:
//! `|residual| <= 255`, the column pass enters at `255 << shift[0]` and the row
//! pass at `round_shift(column_output_bound, -shift[1])`. The audit reports
//! 169 of 193 `(tx_size, tx_type)` cells reachable in the column pass and 166
//! of 193 in the row pass. **The gate is nevertheless taken at RUNTIME, on the
//! actual block**, so this module is correct for any input, including the
//! arbitrary `i16` / `i32` a caller of the public `av1_fwd_txfm2d` may pass:
//! out of range, it declines and the caller runs the i32 pass.
//!
//! # The one rejected kernel
//!
//! `fadst4` has no i16 form at any useful bound. It works in a PRE-SHIFT
//! domain: its stage-1 values are `sinpi[j] * x` held UNSHIFTED (the `>> bit`
//! happens only at its four terminals), so `sum|c_i|` peaks at 21901 and
//! `M*` is 1-11. It stays on the i32 path, as `fwd_kernel_i16` records.

#![allow(clippy::manual_is_multiple_of)]

use archmage::prelude::*;
use magetypes::simd::generic::i16x16 as I16x16;

use crate::transform::cospi::{NEW_SQRT2, NEW_SQRT2_BITS, cospi_arr};

use super::fwd1d_v3_i16_gen::*;
use super::prims16::{
    fbtf16, max_abs_i16_strided, max_abs_i32, mulhrs16, pack_clamp16, rev16, unpk16, widen_hi,
    widen_lo,
};

/// The `mulhrs` multiplier that turns [`super::prims16::mulhrs16`] into
/// `round_shift(_, bit)` on i16 lanes: `m = 2^(15-bit)`. EXACT for every i16
/// `v` and `bit` in `1..=4` — the proof is at [`super::lowbd16::rshift_mul`],
/// and `1..=4` covers every forward `-shift[1]` (0, 1, 2, 4) and `-shift[2]`
/// (0, 2); bit 0 is the scalar early return, never instantiated as a shift.
const fn rshift_mul(bit: i32) -> i16 {
    1i16 << (15 - bit)
}

/// The forward kernels that have an i16 form, and the `M*` each was proved at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fwd1dI16 {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
    Adst8,
    Adst16,
    Idtx4,
    Idtx8,
    Idtx16,
    Idtx32,
}

/// `M*` per kernel: the largest `|input|` for which every value inside the
/// kernel provably stays in i16, hence for which the i16 lane path is
/// bit-identical to the scalar i32 kernel.
///
/// These are the MINIMUM over `cos_bit` in `10..=13` of the per-`(kernel,
/// cos_bit)` bounds `xtask/audit_i16_fwd.py` prints, so one number per kernel
/// is sound at every `cos_bit` the forward config table can select. (The
/// spread is at most 3 counts — e.g. fdct8 is 5792 at cos_bit 10-12 and 5791
/// at 13.)
pub(crate) const fn fwd_i16_max_in(k: Fwd1dI16) -> u32 {
    match k {
        // fdctN: sum|c| = N*sqrt(2)/2 exactly -> floor(46340 / N).
        Fwd1dI16::Dct4 => 11583,
        Fwd1dI16::Dct8 => 5791,
        Fwd1dI16::Dct16 => 2895,
        Fwd1dI16::Dct32 => 1447,
        Fwd1dI16::Dct64 => 723,
        Fwd1dI16::Adst8 => 6419,
        Fwd1dI16::Adst16 => 3214,
        // round_shift(v * NewSqrt2, 12) / 2v / round_shift(v * 2*NewSqrt2, 12) / 4v.
        Fwd1dI16::Idtx4 => 23167,
        Fwd1dI16::Idtx8 => 16383,
        Fwd1dI16::Idtx16 => 11583,
        Fwd1dI16::Idtx32 => 8191,
    }
}

/// TXFM_TYPE id -> i16 kernel, for the ids that have an i16 form.
pub(crate) fn fwd_kernel_i16(txfm_type: i32) -> Option<Fwd1dI16> {
    match txfm_type {
        0 => Some(Fwd1dI16::Dct4),
        1 => Some(Fwd1dI16::Dct8),
        2 => Some(Fwd1dI16::Dct16),
        3 => Some(Fwd1dI16::Dct32),
        4 => Some(Fwd1dI16::Dct64),
        // 5 => fadst4: no i16 form at any useful bound (module docs).
        6 => Some(Fwd1dI16::Adst8),
        7 => Some(Fwd1dI16::Adst16),
        8 => Some(Fwd1dI16::Idtx4),
        9 => Some(Fwd1dI16::Idtx8),
        10 => Some(Fwd1dI16::Idtx16),
        11 => Some(Fwd1dI16::Idtx32),
        _ => None,
    }
}

pub(crate) fn fwd_kernel_i16_n(k: Fwd1dI16) -> usize {
    match k {
        Fwd1dI16::Dct4 | Fwd1dI16::Idtx4 => 4,
        Fwd1dI16::Dct8 | Fwd1dI16::Adst8 | Fwd1dI16::Idtx8 => 8,
        Fwd1dI16::Dct16 | Fwd1dI16::Adst16 | Fwd1dI16::Idtx16 => 16,
        Fwd1dI16::Dct32 | Fwd1dI16::Idtx32 => 32,
        Fwd1dI16::Dct64 => 64,
    }
}

// ---- the four hand-written kernels ------------------------------------------
//
// `transpile_txfm1d.py --lanes16f` covers the six ping-pong-buffer kernels
// (fdct8/16/32/64, fadst8/16). fdct4 and the four identities are hand-ported in
// the scalar port too (`transform/fdct.rs`, `transform/special.rs`), for the
// same reason: they are not in the ping-pong form the transpiler parses.

/// 16-lane i16 twin of [`crate::transform::av1_fdct4`].
#[magetypes(define(i16x16), v3, neon, -scalar)]
fn av1_fdct4_i16_impl(t: Token, input: &[I16x16<Token>], out: &mut [I16x16<Token>], cos_bit: i32) {
    let cospi = cospi_arr(cos_bit);
    let b0 = input[0] + input[3];
    let b1 = input[1] + input[2];
    let b2 = input[1] - input[2];
    let b3 = input[0] - input[3];
    let u0 = unpk16(t, b0, b1);
    let u1 = unpk16(t, b1, b0);
    let u2 = unpk16(t, b2, b3);
    let u3 = unpk16(t, b3, b2);
    // stage 3 permutes step[] into out[]: out = [step0, step2, step1, step3].
    out[0] = fbtf16(t, u0, cospi[32], cospi[32], cos_bit);
    out[2] = fbtf16(t, u1, -cospi[32], cospi[32], cos_bit);
    out[1] = fbtf16(t, u2, cospi[48], cospi[16], cos_bit);
    out[3] = fbtf16(t, u3, cospi[48], -cospi[16], cos_bit);
}

/// 16-lane i16 twins of the four `av1_fidentityN` kernels.
///
/// `fidentity4`/`fidentity16` are `round_shift(v * C, 12)` with C = NewSqrt2 /
/// 2*NewSqrt2; that is `half_btf(C, v, 0, v, 12)`, so [`fbtf16`] serves them
/// exactly (single product `|C| * |v| <= 11586 * 11583 < 2^28`, well inside the
/// i32 accumulator). `fidentity8`/`fidentity32` are `2v` / `4v`, written as
/// repeated lane adds — exact because `M*` bounds the result inside i16.
#[magetypes(define(i16x16), v3, neon, -scalar)]
fn fidentity_i16_impl(t: Token, k: Fwd1dI16, input: &[I16x16<Token>], out: &mut [I16x16<Token>]) {
    match k {
        Fwd1dI16::Idtx4 => {
            for (o, i) in out.iter_mut().zip(input.iter()) {
                *o = fbtf16(t, unpk16(t, *i, *i), NEW_SQRT2, 0, NEW_SQRT2_BITS);
            }
        }
        Fwd1dI16::Idtx8 => {
            for (o, i) in out.iter_mut().zip(input.iter()) {
                *o = *i + *i;
            }
        }
        Fwd1dI16::Idtx16 => {
            for (o, i) in out.iter_mut().zip(input.iter()) {
                *o = fbtf16(t, unpk16(t, *i, *i), 2 * NEW_SQRT2, 0, NEW_SQRT2_BITS);
            }
        }
        Fwd1dI16::Idtx32 => {
            for (o, i) in out.iter_mut().zip(input.iter()) {
                let d = *i + *i;
                *o = d + d;
            }
        }
        _ => unreachable!(),
    }
}

/// Direct-dispatch the i16 forward kernel (same shape as [`super`]'s
/// `run_fwd1d` and [`super::lowbd16`]'s `run_inv1d_i16`).
#[magetypes(define(i16x16), v3, neon, -scalar)]
pub(crate) fn run_fwd1d_i16(
    t: Token,
    k: Fwd1dI16,
    input: &[I16x16<Token>],
    out: &mut [I16x16<Token>],
    cos_bit: i32,
) {
    let _ = t;
    match k {
        Fwd1dI16::Dct4 => incant!(av1_fdct4_i16_impl(input, out, cos_bit), [v3, neon]),
        Fwd1dI16::Dct8 => incant!(av1_fdct8_i16_impl(input, out, cos_bit), [v3, neon]),
        Fwd1dI16::Dct16 => incant!(av1_fdct16_i16_impl(input, out, cos_bit), [v3, neon]),
        Fwd1dI16::Dct32 => incant!(av1_fdct32_i16_impl(input, out, cos_bit), [v3, neon]),
        Fwd1dI16::Dct64 => incant!(av1_fdct64_i16_impl(input, out, cos_bit), [v3, neon]),
        Fwd1dI16::Adst8 => incant!(av1_fadst8_i16_impl(input, out, cos_bit), [v3, neon]),
        Fwd1dI16::Adst16 => incant!(av1_fadst16_i16_impl(input, out, cos_bit), [v3, neon]),
        Fwd1dI16::Idtx4 | Fwd1dI16::Idtx8 | Fwd1dI16::Idtx16 | Fwd1dI16::Idtx32 => {
            incant!(fidentity_i16_impl(k, input, out), [v3, neon])
        }
    }
}

// ---- i16-lane forward COLUMN pass -------------------------------------------

/// Scalar twin of [`fwd_col_pass_i16`] — declines, routing the caller to the
/// i32 pass (and from there to its own scalar loop). Same contract as
/// [`super::fwd_col_pass_scalar`]: there is deliberately no scalar
/// *implementation* here, because the driver's per-column loop IS the
/// differential's reference.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_col_pass_i16_scalar(
    _: ScalarToken,
    _kernel: Fwd1dI16,
    _input: &[i16],
    _buf: &mut [i32],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _shift0: i32,
    _shift1_bit: i32,
    _cos_bit_col: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// i16-lane forward COLUMN pass — 16 columns per lane batch.
///
/// Preconditions asserted by the caller in [`super::try_fwd_col_pass`]:
/// `col_n % 16 == 0` (a 4- or 8-wide block would run the SAME kernel
/// instruction count as the i32x8 pass — one partial batch either way — plus
/// the narrowing overhead, a structural loss, exactly as
/// [`super::lowbd16`]'s row pass argues for its own 4/8 sizes), an i16-formed
/// kernel spanning `row_n` points, and `max|input| << shift0 <= M*`.
///
/// The vector scratch is TIERED by `row_n` for the same reason every other
/// pass in this family is: a flat 64-entry array zero-init is a memset that
/// dominates the small transforms.
#[magetypes(define(i16x16), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_col_pass_i16(
    t: Token,
    kernel: Fwd1dI16,
    input: &[i16],
    buf: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
    shift1_bit: i32,
    cos_bit_col: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    debug_assert!(row_n <= 64 && col_n % 16 == 0);
    debug_assert!(shift0 == 0 || shift0 == 2);
    debug_assert!((0..=4).contains(&shift1_bit));
    if row_n <= 8 {
        let mut tin = [i16x16::zero(t); 8];
        let mut tout = [i16x16::zero(t); 8];
        incant!(
            fwd_col_pass_i16_core(
                kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col, ud_flip,
                lr_flip, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    } else if row_n <= 16 {
        let mut tin = [i16x16::zero(t); 16];
        let mut tout = [i16x16::zero(t); 16];
        incant!(
            fwd_col_pass_i16_core(
                kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col, ud_flip,
                lr_flip, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    } else {
        let mut tin = [i16x16::zero(t); 64];
        let mut tout = [i16x16::zero(t); 64];
        incant!(
            fwd_col_pass_i16_core(
                kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col, ud_flip,
                lr_flip, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    }
    true
}

#[magetypes(define(i16x16), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_col_pass_i16_core(
    t: Token,
    kernel: Fwd1dI16,
    input: &[i16],
    buf: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
    shift1_bit: i32,
    cos_bit_col: i32,
    ud_flip: bool,
    lr_flip: bool,
    tin: &mut [I16x16<Token>],
    tout: &mut [I16x16<Token>],
) {
    for cg in (0..col_n).step_by(16) {
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let src_r = if ud_flip { row_n - r - 1 } else { r };
            let base = src_r * stride + cg;
            let v = i16x16::from_slice(t, &input[base..base + 16]);
            // round_shift_array(temp_in, -shift[0]) with shift[0] in {0, 2} —
            // the NEGATIVE-bit arm, i.e. `v << shift0` (the scalar clamps to
            // i32 in i64; the gate bounds |v << shift0| <= M* < 2^15, so no
            // clamp and no i16 wrap can occur). Written as repeated lane adds
            // so no const-shift generic is needed.
            *ti = if shift0 > 0 {
                let d = v + v;
                d + d
            } else {
                v
            };
        }
        incant!(run_fwd1d_i16(kernel, &tin[..row_n], &mut tout[..row_n], cos_bit_col), [v3, neon]);
        for (r, to) in tout[..row_n].iter_mut().enumerate() {
            // round_shift_array(temp_out, -shift[1]); bit in {0, 1, 2, 4}.
            let v = if shift1_bit > 0 { mulhrs16(t, *to, rshift_mul(shift1_bit)) } else { *to };
            // Scalar: buf[r*col_n + dst_c] = temp_out[r], dst_c lr-flipped.
            let (v, base) = if lr_flip {
                (rev16(t, v), r * col_n + (col_n - cg - 16))
            } else {
                (v, r * col_n + cg)
            };
            // Sign-extend i16 -> i32 into the UNCHANGED row-major i32 `buf`: a
            // pure width extension, so the row pass (i16 or i32) reads exactly
            // what the i32 column pass would have written.
            widen_lo(t, v).store((&mut buf[base..base + 8]).try_into().unwrap());
            widen_hi(t, v).store((&mut buf[base + 8..base + 16]).try_into().unwrap());
        }
    }
}

// ---- i16-lane forward ROW pass ----------------------------------------------

/// Scalar twin of [`fwd_row_pass_i16`] — declines (see
/// [`fwd_col_pass_i16_scalar`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_row_pass_i16_scalar(
    _: ScalarToken,
    _kernel: Fwd1dI16,
    _buf: &[i32],
    _output: &mut [i32],
    _col_n: usize,
    _row_n: usize,
    _shift2_bit: i32,
    _cos_bit_row: i32,
    _rect1: bool,
) -> bool {
    false
}

/// i16-lane forward ROW pass — 16 rows per lane batch.
///
/// Preconditions asserted by the caller in [`super::try_fwd_row_pass`]:
/// `row_n % 16 == 0`, `col_n % 8 == 0` (the strided load is 8x8 i32 transpose
/// tiles, as in the i32 row pass), an i16-formed kernel spanning `col_n`
/// points, and `max|buf| <= M*`.
///
/// The rect `NewSqrt2` scale stays OUTSIDE the i16 domain, on the widened i32
/// output, exactly as [`super::lowbd16`]'s inverse row pass keeps its own rect
/// scale in i32 lanes: its 1.414x gain is not covered by `M*`, and
/// [`super::prims::mul_rshiftv`] is exact for any i32.
#[magetypes(define(i16x16), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_row_pass_i16(
    t: Token,
    kernel: Fwd1dI16,
    buf: &[i32],
    output: &mut [i32],
    col_n: usize,
    row_n: usize,
    shift2_bit: i32,
    cos_bit_row: i32,
    rect1: bool,
) -> bool {
    debug_assert!(col_n <= 64 && row_n % 16 == 0 && col_n % 8 == 0);
    debug_assert!((0..=4).contains(&shift2_bit));
    if col_n <= 8 {
        let mut tin = [i16x16::zero(t); 8];
        let mut tout = [i16x16::zero(t); 8];
        incant!(
            fwd_row_pass_i16_core(
                kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1, &mut tin,
                &mut tout,
            ),
            [v3, neon]
        );
    } else if col_n <= 16 {
        let mut tin = [i16x16::zero(t); 16];
        let mut tout = [i16x16::zero(t); 16];
        incant!(
            fwd_row_pass_i16_core(
                kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1, &mut tin,
                &mut tout,
            ),
            [v3, neon]
        );
    } else {
        let mut tin = [i16x16::zero(t); 64];
        let mut tout = [i16x16::zero(t); 64];
        incant!(
            fwd_row_pass_i16_core(
                kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1, &mut tin,
                &mut tout,
            ),
            [v3, neon]
        );
    }
    true
}

#[magetypes(define(i16x16, i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_row_pass_i16_core(
    t: Token,
    kernel: Fwd1dI16,
    buf: &[i32],
    output: &mut [i32],
    col_n: usize,
    row_n: usize,
    shift2_bit: i32,
    cos_bit_row: i32,
    rect1: bool,
    tin: &mut [I16x16<Token>],
    tout: &mut [I16x16<Token>],
) {
    for rg in (0..row_n).step_by(16) {
        // Load: tin[c].lane(k) = buf[(rg+k)*col_n + c]. Two 8x8 i32 transpose
        // tiles (contiguous row loads) — rows rg..rg+8 into lanes 0-7 and
        // rg+8..rg+16 into lanes 8-15 — then the narrowing pack. The pack
        // SATURATES, which the gate makes unreachable, so it is a pure width
        // narrowing here.
        for cg in (0..col_n).step_by(8) {
            let mut lo = [i32x8::zero(t); 8];
            for (k, rk) in lo.iter_mut().enumerate() {
                let base = (rg + k) * col_n + cg;
                *rk = i32x8::from_slice(t, &buf[base..base + 8]);
            }
            let tlo = super::prims::transpose8(t, &lo);
            let mut hi = [i32x8::zero(t); 8];
            for (k, rk) in hi.iter_mut().enumerate() {
                let base = (rg + 8 + k) * col_n + cg;
                *rk = i32x8::from_slice(t, &buf[base..base + 8]);
            }
            let thi = super::prims::transpose8(t, &hi);
            for j in 0..8 {
                tin[cg + j] = pack_clamp16(t, tlo[j], thi[j]);
            }
        }
        incant!(run_fwd1d_i16(kernel, &tin[..col_n], &mut tout[..col_n], cos_bit_row), [v3, neon]);
        for (c, to) in tout[..col_n].iter_mut().enumerate() {
            // round_shift_array(row_buffer, -shift[2]); bit in {0, 2}.
            let v = if shift2_bit > 0 { mulhrs16(t, *to, rshift_mul(shift2_bit)) } else { *to };
            let mut a = widen_lo(t, v);
            let mut b = widen_hi(t, v);
            if rect1 {
                // round_shift(v * NewSqrt2, NewSqrt2Bits) — AFTER the shift,
                // matching the scalar order, and in i32 lanes (exact for any
                // i32; its 1.414x gain is outside the i16 bound).
                a = super::prims::mul_rshiftv(t, a, NEW_SQRT2, NEW_SQRT2_BITS);
                b = super::prims::mul_rshiftv(t, b, NEW_SQRT2, NEW_SQRT2_BITS);
            }
            // Scalar: output[c*row_n + r] = row_buffer[c] — contiguous per c.
            let base = c * row_n + rg;
            a.store((&mut output[base..base + 8]).try_into().unwrap());
            b.store((&mut output[base + 8..base + 16]).try_into().unwrap());
        }
    }
}

// ---- the runtime gates ------------------------------------------------------

/// Does the i16 forward COLUMN pass apply? `max|input| << shift0 <= M*` is the
/// whole condition, and it is checked on the ACTUAL block, so the answer is
/// sound for any caller of the public `av1_fwd_txfm2d` — not only for bd8
/// residuals.
pub(crate) fn fwd_col_i16_applies(
    kernel: Fwd1dI16,
    input: &[i16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
) -> bool {
    let m = max_abs_i16_strided(input, stride, col_n, row_n);
    m <= (fwd_i16_max_in(kernel) >> shift0)
}

/// Does the i16 forward ROW pass apply? `max|buf| <= M*`.
pub(crate) fn fwd_row_i16_applies(kernel: Fwd1dI16, buf: &[i32], col_n: usize, row_n: usize) -> bool {
    max_abs_i32(&buf[..col_n * row_n]) <= fwd_i16_max_in(kernel)
}

#[cfg(test)]
mod tests {
    //! i16-lane-vs-scalar differentials for the forward kernels and for both
    //! passes, over their FULL contract domains: every kernel lane an arbitrary
    //! value in `[-M*, M*]` (exactly what the runtime gate admits), at every
    //! `cos_bit` the forward config table can select, and both passes over
    //! their full shift / flip / rect grids.
    //!
    //! Probe design follows KB-12: a hand-transcribed kernel's unit test must
    //! break the symmetries the suspected defect breaks. FLAT blocks put all
    //! the energy in the DC and are blind to a dropped permutation or a
    //! transposed store, so every case here is dense-random or an
    //! alternating/asymmetric boundary pattern, and the pass differentials use
    //! DISTINCT sentinels in the two buffers so a position missed by both
    //! still mismatches. Runs at every token permutation on both
    //! architectures, with a counter proving a vector tier actually ran.

    use super::*;
    use crate::transform::{
        av1_fadst8, av1_fadst16, av1_fdct4, av1_fdct8, av1_fdct16, av1_fdct32, av1_fdct64,
        av1_fidentity4, av1_fidentity8, av1_fidentity16, av1_fidentity32,
    };
    use crate::transform::fdct::round_shift;
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    pub(super) type ScalarKernel = fn(&[i32], &mut [i32], i32, &[i8]);

    pub(super) fn cases() -> [(&'static str, usize, ScalarKernel, Fwd1dI16); 11] {
        [
            ("fdct4", 4, av1_fdct4 as ScalarKernel, Fwd1dI16::Dct4),
            ("fdct8", 8, av1_fdct8, Fwd1dI16::Dct8),
            ("fdct16", 16, av1_fdct16, Fwd1dI16::Dct16),
            ("fdct32", 32, av1_fdct32, Fwd1dI16::Dct32),
            ("fdct64", 64, av1_fdct64, Fwd1dI16::Dct64),
            ("fadst8", 8, av1_fadst8, Fwd1dI16::Adst8),
            ("fadst16", 16, av1_fadst16, Fwd1dI16::Adst16),
            ("fidentity4", 4, av1_fidentity4, Fwd1dI16::Idtx4),
            ("fidentity8", 8, av1_fidentity8, Fwd1dI16::Idtx8),
            ("fidentity16", 16, av1_fidentity16, Fwd1dI16::Idtx16),
            ("fidentity32", 32, av1_fidentity32, Fwd1dI16::Idtx32),
        ]
    }

    pub(super) struct Rng(pub u64);
    impl Rng {
        pub(super) fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform in `[-m, m]` — the exact domain the runtime gate admits.
        pub(super) fn bounded(&mut self, m: u32) -> i16 {
            let span = 2 * m + 1;
            ((self.next() % span as u64) as i64 - m as i64) as i16
        }
    }

    const SR: [i8; 12] = [0; 12]; // forward kernels ignore stage_range

    #[magetypes(define(i16x16), v3, neon, -scalar)]
    #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
    fn assert_batch16(
        t: Token,
        name: &str,
        n: usize,
        scalar: ScalarKernel,
        kernel: Fwd1dI16,
        cos_bit: i32,
        cols: &[[i16; 16]],
        label: &str,
    ) {
        let mut vin = vec![i16x16::zero(t); n];
        for (r, c) in cols.iter().enumerate() {
            vin[r] = i16x16::from_array(t, *c);
        }
        let mut vout = vec![i16x16::zero(t); n];
        incant!(run_fwd1d_i16(kernel, &vin, &mut vout, cos_bit), [v3, neon]);

        let mut sin = vec![0i32; n];
        let mut sout = vec![0i32; n];
        for lane in 0..16 {
            for r in 0..n {
                sin[r] = cols[r][lane] as i32;
            }
            scalar(&sin, &mut sout, cos_bit, &SR);
            for r in 0..n {
                assert_eq!(
                    vout[r].to_array()[lane] as i32,
                    sout[r],
                    "{name}: {label} cos_bit={cos_bit} lane={lane} row={r} input={sin:?}"
                );
            }
        }
    }

    fn sweep_kernels_scalar(_: ScalarToken, _tier: &str) -> bool {
        false
    }

    /// Kernel differential at ONE tier. Returns true (a vector tier ran).
    #[magetypes(define(i16x16), v3, neon, -scalar)]
    fn sweep_kernels(t: Token, tier: &str) -> bool {
        let _ = t;
        let mut rng = Rng(0x_f16f_2026_0802_u64);
        for (name, n, scalar, kernel) in cases() {
            let m = fwd_i16_max_in(kernel);
            for cos_bit in 10..=13i32 {
                // (a) dense random over the FULL admitted domain.
                for rep in 0..24 {
                    let cols: Vec<[i16; 16]> =
                        (0..n).map(|_| core::array::from_fn(|_| rng.bounded(m))).collect();
                    incant!(
                        assert_batch16(
                            name,
                            n,
                            scalar,
                            kernel,
                            cos_bit,
                            &cols,
                            &format!("[{tier}] rand rep{rep}"),
                        ),
                        [v3, neon]
                    );
                }
                // (b) the exact bound, in asymmetric sign patterns — the
                // coefficient-sum vertex the audit's bound is tight at, i.e.
                // the worst case for every intermediate at once.
                let (lo, hi) = (-(m as i16), m as i16);
                let pats: [&dyn Fn(usize, usize) -> i16; 6] = [
                    &|_, _| hi,
                    &|_, _| lo,
                    &|r, l| if (r + l) % 2 == 0 { hi } else { lo },
                    &|r, l| if (r + l) % 2 == 0 { lo } else { hi },
                    &|r, l| if (r * 7 + l * 3) % 5 < 2 { hi } else { lo },
                    &|r, l| if (r * 3 + l) % 3 == 0 { lo } else { hi },
                ];
                for (pi, pat) in pats.iter().enumerate() {
                    let cols: Vec<[i16; 16]> =
                        (0..n).map(|r| core::array::from_fn(|l| pat(r, l))).collect();
                    incant!(
                        assert_batch16(
                            name,
                            n,
                            scalar,
                            kernel,
                            cos_bit,
                            &cols,
                            &format!("[{tier}] bound pat{pi}"),
                        ),
                        [v3, neon]
                    );
                }
                // (c) bound lanes mixed with zero rows — asymmetric, so a
                // dropped permutation cannot hide.
                let mut cols = vec![[0i16; 16]; n];
                cols[0] = core::array::from_fn(|l| [lo, hi, 0, -1, 1, lo + 1, hi - 1, 2][l % 8]);
                cols[n - 1] = core::array::from_fn(|l| [hi, lo, -2, 2, 0, -1, 1, lo + 1][l % 8]);
                incant!(
                    assert_batch16(
                        name,
                        n,
                        scalar,
                        kernel,
                        cos_bit,
                        &cols,
                        &format!("[{tier}] extremes"),
                    ),
                    [v3, neon]
                );
            }
        }
        true
    }

    #[test]
    fn fwd1d_i16_bit_identical_to_scalar_at_every_tier() {
        let _ = crate::dispatch::scalar_forced();
        let mut simd_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            if incant!(sweep_kernels(&tier.label), [v3, neon, scalar]) {
                simd_ran += 1;
            }
        });
        eprintln!("fwd1d i16 parity: {report}, vector permutations run: {simd_ran}");
        assert!(
            simd_ran >= 1,
            "a vector tier must run at least once (AVX2 on x86-64, NEON on aarch64)"
        );
        assert!(report.permutations_run >= 2);
    }

    // ---- pass differentials -------------------------------------------------

    fn round_shift_array(arr: &mut [i32], bit: i32) {
        if bit == 0 {
            return;
        }
        if bit > 0 {
            for v in arr.iter_mut() {
                *v = round_shift(*v as i64, bit);
            }
        } else {
            for v in arr.iter_mut() {
                let w = (1i64 << (-bit)) * (*v as i64);
                *v = w.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }
    }

    /// Scalar replica of the driver's forward COLUMN loop (`txfm2d.rs`) — the
    /// same ops in the same order.
    #[allow(clippy::too_many_arguments)]
    fn scalar_col_pass(
        scalar: ScalarKernel,
        input: &[i16],
        buf: &mut [i32],
        stride: usize,
        col_n: usize,
        row_n: usize,
        shift0: i32,
        shift1_bit: i32,
        cos_bit: i32,
        ud_flip: bool,
        lr_flip: bool,
    ) {
        let mut temp_in = vec![0i32; row_n];
        let mut temp_out = vec![0i32; row_n];
        for c in 0..col_n {
            for r in 0..row_n {
                let src_r = if ud_flip { row_n - r - 1 } else { r };
                temp_in[r] = input[src_r * stride + c] as i32;
            }
            round_shift_array(&mut temp_in, -shift0);
            scalar(&temp_in, &mut temp_out, cos_bit, &SR);
            round_shift_array(&mut temp_out, shift1_bit);
            for r in 0..row_n {
                let dst_c = if lr_flip { col_n - c - 1 } else { c };
                buf[r * col_n + dst_c] = temp_out[r];
            }
        }
    }

    /// Scalar replica of the driver's forward ROW loop (`txfm2d.rs`).
    #[allow(clippy::too_many_arguments)]
    fn scalar_row_pass(
        scalar: ScalarKernel,
        buf: &[i32],
        output: &mut [i32],
        col_n: usize,
        row_n: usize,
        shift2_bit: i32,
        cos_bit: i32,
        rect1: bool,
    ) {
        let mut rb = vec![0i32; col_n];
        for r in 0..row_n {
            scalar(&buf[r * col_n..r * col_n + col_n], &mut rb, cos_bit, &SR);
            round_shift_array(&mut rb, shift2_bit);
            if rect1 {
                for v in rb.iter_mut() {
                    *v = round_shift(*v as i64 * NEW_SQRT2 as i64, NEW_SQRT2_BITS);
                }
            }
            for c in 0..col_n {
                output[c * row_n + r] = rb[c];
            }
        }
    }

    fn sweep_passes_scalar(_: ScalarToken, _tier: &str) -> bool {
        false
    }

    /// Both pass differentials at ONE tier. Returns true (a vector tier ran).
    #[magetypes(define(i16x16), v3, neon, -scalar)]
    fn sweep_passes(t: Token, tier: &str) -> bool {
        let _ = t;
        let mut rng = Rng(0x_0802_2026_c01d);
        for (name, n, scalar, kernel) in cases() {
            let m = fwd_i16_max_in(kernel);
            for cos_bit in [10i32, 12, 13] {
                // ---- COLUMN pass: kernel spans row_n == n, col_n in {16,32,64}.
                for &col_n in &[16usize, 32, 64] {
                    let row_n = n;
                    let stride = col_n + 7; // deliberately != col_n
                    for &shift0 in &[0i32, 2] {
                        for &shift1_bit in &[0i32, 1, 2, 4] {
                            for flips in 0..4 {
                                let (ud, lr) = (flips & 1 != 0, flips & 2 != 0);
                                let mm = m >> shift0;
                                let inp: Vec<i16> = (0..row_n * stride)
                                    .map(|_| rng.bounded(mm))
                                    .collect();
                                assert!(fwd_col_i16_applies(
                                    kernel, &inp, stride, col_n, row_n, shift0
                                ));
                                let mut vbuf = vec![111_i32; col_n * row_n];
                                assert!(incant!(
                                    fwd_col_pass_i16(
                                        kernel, &inp, &mut vbuf, stride, col_n, row_n, shift0,
                                        shift1_bit, cos_bit, ud, lr
                                    ),
                                    [v3, neon]
                                ));
                                let mut sbuf = vec![-222_i32; col_n * row_n];
                                scalar_col_pass(
                                    scalar, &inp, &mut sbuf, stride, col_n, row_n, shift0,
                                    shift1_bit, cos_bit, ud, lr,
                                );
                                assert_eq!(
                                    vbuf, sbuf,
                                    "{name} col: [{tier}] col_n={col_n} row_n={row_n} \
                                     shift0={shift0} shift1={shift1_bit} cos_bit={cos_bit} \
                                     ud={ud} lr={lr}"
                                );
                            }
                        }
                    }
                }
                // ---- ROW pass: kernel spans col_n == n, row_n in {16,32,64}.
                if n >= 8 {
                    let col_n = n;
                    for &row_n in &[16usize, 32, 64] {
                        for &shift2_bit in &[0i32, 2] {
                            for &rect1 in &[false, true] {
                                let buf: Vec<i32> =
                                    (0..col_n * row_n).map(|_| rng.bounded(m) as i32).collect();
                                assert!(fwd_row_i16_applies(kernel, &buf, col_n, row_n));
                                let mut vout = vec![111_i32; col_n * row_n];
                                assert!(incant!(
                                    fwd_row_pass_i16(
                                        kernel, &buf, &mut vout, col_n, row_n, shift2_bit,
                                        cos_bit, rect1
                                    ),
                                    [v3, neon]
                                ));
                                let mut sout = vec![-222_i32; col_n * row_n];
                                scalar_row_pass(
                                    scalar, &buf, &mut sout, col_n, row_n, shift2_bit, cos_bit,
                                    rect1,
                                );
                                assert_eq!(
                                    vout, sout,
                                    "{name} row: [{tier}] col_n={col_n} row_n={row_n} \
                                     shift2={shift2_bit} cos_bit={cos_bit} rect1={rect1}"
                                );
                            }
                        }
                    }
                }
            }
        }
        true
    }

    #[test]
    fn fwd_i16_passes_bit_identical_to_scalar_at_every_tier() {
        let _ = crate::dispatch::scalar_forced();
        let mut simd_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            if incant!(sweep_passes(&tier.label), [v3, neon, scalar]) {
                simd_ran += 1;
            }
        });
        eprintln!("fwd i16 pass parity: {report}, vector permutations run: {simd_ran}");
        assert!(simd_ran >= 1, "a vector tier must run at least once");
        assert!(report.permutations_run >= 2);
    }

    /// The gate is the whole safety argument, so it gets its own pin: one
    /// value over `M*` must make it decline, and `i16::MIN` (the one value
    /// whose negation wraps) must decline at every kernel.
    #[test]
    fn the_gate_declines_above_m_star() {
        for (name, n, _scalar, kernel) in cases() {
            let m = fwd_i16_max_in(kernel);
            let (col_n, row_n, stride) = (16usize, n, 16usize);
            let mut inp = vec![0i16; row_n * stride];
            inp[row_n * stride - 1] = m as i16;
            assert!(
                fwd_col_i16_applies(kernel, &inp, stride, col_n, row_n, 0),
                "{name}: exactly M* must be admitted"
            );
            inp[row_n * stride - 1] = m as i16 + 1;
            assert!(
                !fwd_col_i16_applies(kernel, &inp, stride, col_n, row_n, 0),
                "{name}: M*+1 must be declined"
            );
            inp[row_n * stride - 1] = i16::MIN;
            assert!(
                !fwd_col_i16_applies(kernel, &inp, stride, col_n, row_n, 0),
                "{name}: i16::MIN must be declined"
            );
            // shift0 shrinks the admitted input by the same factor.
            let mut inp2 = vec![0i16; row_n * stride];
            inp2[0] = (m >> 2) as i16;
            assert!(fwd_col_i16_applies(kernel, &inp2, stride, col_n, row_n, 2), "{name}");
            inp2[0] = (m >> 2) as i16 + 1;
            assert!(!fwd_col_i16_applies(kernel, &inp2, stride, col_n, row_n, 2), "{name}");

            let mut buf = vec![0i32; n * 16];
            buf[5] = m as i32;
            assert!(fwd_row_i16_applies(kernel, &buf, n, 16), "{name}");
            buf[5] = m as i32 + 1;
            assert!(!fwd_row_i16_applies(kernel, &buf, n, 16), "{name}");
        }
    }
}

#[cfg(test)]
mod gate_bite {
    //! The gate's own non-vacuity (§2): `M*` is a SOUND bound, so bit-identity
    //! at `M*` alone would also hold for a gate that admitted far too much.
    //! This pins the other side — that outside the bound the i16 kernel and the
    //! scalar kernel genuinely diverge, i.e. the gate is load-bearing and not
    //! decorative.
    //!
    //! It also RECORDS the slack: `M*` is sound but not attained, because the
    //! rounding-error term `E` is a worst case that no single input realises.
    //! The first input magnitude at which each kernel actually diverges is
    //! printed by this test and quoted in
    //! `benchmarks/encoder_i16_fwd_2026-08-02.md`.

    use super::tests::{Rng, cases};
    use super::*;
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    fn probe_scalar(_: ScalarToken, _n: &mut usize) -> bool {
        false
    }

    /// Smallest `|input|` at which a dense random sweep finds a divergence.
    #[magetypes(define(i16x16), v3, neon, -scalar)]
    #[allow(clippy::needless_range_loop)]
    fn probe(t: Token, diverged: &mut usize) -> bool {
        let _ = t;
        let mut rng = Rng(0x_9a5e_2026_0802);
        for (name, n, scalar, kernel) in cases() {
            let m = fwd_i16_max_in(kernel);
            let mut first = None;
            let mut mag = m;
            while mag < 8 * m.max(1) && first.is_none() {
                mag += m / 16 + 1;
                'outer: for _ in 0..64 {
                    let cols: Vec<[i16; 16]> = (0..n)
                        .map(|_| core::array::from_fn(|_| rng.bounded(mag.min(32767))))
                        .collect();
                    let mut vin = vec![i16x16::zero(t); n];
                    for (r, c) in cols.iter().enumerate() {
                        vin[r] = i16x16::from_array(t, *c);
                    }
                    let mut vout = vec![i16x16::zero(t); n];
                    incant!(run_fwd1d_i16(kernel, &vin, &mut vout, 13), [v3, neon]);
                    let (mut si, mut so) = (vec![0i32; n], vec![0i32; n]);
                    for lane in 0..16 {
                        for r in 0..n {
                            si[r] = cols[r][lane] as i32;
                        }
                        scalar(&si, &mut so, 13, &[0i8; 12]);
                        for r in 0..n {
                            if vout[r].to_array()[lane] as i32 != so[r] {
                                first = Some(mag);
                                break 'outer;
                            }
                        }
                    }
                }
            }
            match first {
                Some(f) => {
                    *diverged += 1;
                    eprintln!(
                        "  {name:<12} M* = {m:>5}   first divergence at |input| ~ {f}  \
                         (slack {:.2}x)",
                        f as f64 / m as f64
                    );
                }
                None => eprintln!("  {name:<12} M* = {m:>5}   no divergence up to 8x M*"),
            }
        }
        true
    }

    #[test]
    fn the_bound_is_load_bearing() {
        let _ = crate::dispatch::scalar_forced();
        let mut diverged = 0usize;
        let mut ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_tier| {
            if ran == 0 && incant!(probe(&mut diverged), [v3, neon, scalar]) {
                ran += 1;
            }
        });
        eprintln!("gate bite: {report}");
        assert!(ran >= 1, "a vector tier must run at least once");
        // Every kernel must break somewhere above its bound; a kernel that
        // never diverged would mean the gate was costing reach for nothing.
        assert_eq!(
            diverged,
            cases().len(),
            "every kernel must diverge outside its M*, else the gate is decorative"
        );
    }
}

#[cfg(test)]
mod reach {
    //! REACH: how much of the encoder's real domain the runtime gate admits.
    //!
    //! Bit-identity at `M*` is only half the story — a gate that admitted
    //! nothing would also pass every differential above. This measures the
    //! other half on the actual driver chain at the bd8 extreme
    //! (`|residual| = 255`, `src - pred` over u8) and PINS the counts, so a
    //! future change that quietly narrows the gate fails here rather than
    //! silently costing the lever.
    //!
    //! Column reach is the gate applied to the residual block; row reach runs
    //! the real scalar column loop first, so it is measured on the buffer the
    //! row pass actually sees.

    use super::tests::Rng;
    use super::*;
    use crate::transform::txfm2d::{
        COS_BIT_COL, COS_BIT_ROW, FWD_SHIFT, HTX_TAB, TXFM_TYPE_LS, TX_SIZE_HIGH, TX_SIZE_WIDE,
        VTX_TAB, fwd_txfm_valid, log2_idx,
    };
    use crate::transform::{
        av1_fadst4, av1_fadst8, av1_fadst16, av1_fdct4, av1_fdct8, av1_fdct16, av1_fdct32,
        av1_fdct64, av1_fidentity4, av1_fidentity8, av1_fidentity16, av1_fidentity32,
    };
    use crate::transform::fdct::round_shift;

    type K = fn(&[i32], &mut [i32], i32, &[i8]);

    fn scalar_of(txfm_type: i32) -> K {
        match txfm_type {
            0 => av1_fdct4,
            1 => av1_fdct8,
            2 => av1_fdct16,
            3 => av1_fdct32,
            4 => av1_fdct64,
            5 => av1_fadst4,
            6 => av1_fadst8,
            7 => av1_fadst16,
            8 => av1_fidentity4,
            9 => av1_fidentity8,
            10 => av1_fidentity16,
            _ => av1_fidentity32,
        }
    }

    /// The two shift domains `try_fwd_col_pass` / `try_fwd_row_pass` gate on
    /// are the WHOLE of `FWD_SHIFT` today, so those gate conditions are inert
    /// — which is exactly why they are cheap insurance rather than a cost. If
    /// a future shift table leaves these ranges this test says so, and the
    /// affected cells route to the i32 pass instead of walking off the
    /// `v+v; d+d` (== `<< 2`) and `rshift_mul` (== `round_shift`, bit 1..=4)
    /// proofs.
    #[test]
    fn the_shift_table_stays_inside_the_gated_domains() {
        for (ts, sh) in FWD_SHIFT.iter().enumerate() {
            assert!(sh[0] == 0 || sh[0] == 2, "tx_size {ts}: shift[0] = {}", sh[0]);
            assert!((0..=4).contains(&-(sh[1] as i32)), "tx_size {ts}: shift[1] = {}", sh[1]);
            assert!((0..=4).contains(&-(sh[2] as i32)), "tx_size {ts}: shift[2] = {}", sh[2]);
        }
    }

    #[test]
    fn the_gate_fires_across_the_bd8_grid() {
        let mut rng = Rng(0x_0802_2026_7eac);
        let (mut col_live, mut row_live, mut cells) = (0usize, 0usize, 0usize);
        let (mut col_shape, mut row_shape) = (0usize, 0usize);
        for ts in 0..19usize {
            let (col_n, row_n) = (TX_SIZE_WIDE[ts], TX_SIZE_HIGH[ts]);
            let (wi, hi) = (log2_idx(col_n), log2_idx(row_n));
            let sh = FWD_SHIFT[ts];
            let (cbc, cbr) = (COS_BIT_COL[wi][hi] as i32, COS_BIT_ROW[wi][hi] as i32);
            for tt in 0..16usize {
                if !fwd_txfm_valid(tt, ts) {
                    continue;
                }
                cells += 1;
                let cti = TXFM_TYPE_LS[hi][VTX_TAB[tt]];
                let rti = TXFM_TYPE_LS[wi][HTX_TAB[tt]];
                // Worst-case bd8 residual: every lane at the extreme, in the
                // sign pattern that maximises a coefficient sum.
                let input: Vec<i16> = (0..col_n * row_n)
                    .map(|i| if (i * 7 + i / 5) % 3 == 0 { -255 } else { 255 })
                    .collect();
                let _ = rng.next();
                if col_n % 16 == 0 {
                    if let Some(k16) = fwd_kernel_i16(cti) {
                        col_shape += 1;
                        if fwd_col_i16_applies(k16, &input, col_n, col_n, row_n, sh[0] as i32) {
                            col_live += 1;
                        }
                    }
                }
                // Real scalar column loop -> the buffer the row pass sees.
                let f = scalar_of(cti);
                let mut buf = vec![0i32; col_n * row_n];
                let (mut ti, mut to) = (vec![0i32; row_n], vec![0i32; row_n]);
                for c in 0..col_n {
                    for r in 0..row_n {
                        ti[r] = (input[r * col_n + c] as i32) << sh[0];
                    }
                    f(&ti, &mut to, cbc, &[0i8; 12]);
                    if sh[1] < 0 {
                        for v in to.iter_mut() {
                            *v = round_shift(*v as i64, -(sh[1] as i32));
                        }
                    }
                    for r in 0..row_n {
                        buf[r * col_n + c] = to[r];
                    }
                }
                if row_n % 16 == 0 && col_n % 8 == 0 {
                    if let Some(k16) = fwd_kernel_i16(rti) {
                        row_shape += 1;
                        let _ = cbr;
                        if fwd_row_i16_applies(k16, &buf, col_n, row_n) {
                            row_live += 1;
                        } else {
                            eprintln!(
                                "  row gate declines: tx_size {ts} ({col_n}x{row_n}) \
                                 tx_type {tt} kernel {k16:?} max|buf| {} > M* {}",
                                max_abs_i32(&buf),
                                fwd_i16_max_in(k16)
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "fwd i16 reach at |residual| = 255: cells {cells}; \
             col shape-eligible {col_shape}, gate fires {col_live}; \
             row shape-eligible {row_shape}, gate fires {row_live}"
        );
        assert_eq!(cells, 193, "the (tx_size, tx_type) grid is 193 valid cells");
        // MEASURED 2026-08-02 on the worst-case bd8 residual. These are the
        // reach numbers `benchmarks/encoder_i16_fwd_2026-08-02.md` quotes; a
        // change that narrows them is a regression in the lever, not a
        // regression in correctness, and it must be re-pinned deliberately.
        assert_eq!(
            (col_shape, col_live),
            (81, 81),
            "81 of the 193 cells are column shape-eligible (col_n % 16 == 0 and an \
             i16-formed column kernel), and the gate fires on ALL of them at the bd8 \
             extreme"
        );
        assert_eq!(
            (row_shape, row_live),
            (73, 70),
            "73 cells are row shape-eligible; 3 decline at the bd8 extreme — the \
             fdct64/fdct32 row kernels whose M* the chain overshoots, exactly the \
             cells xtask/audit_i16_fwd.py predicts"
        );
    }
}
