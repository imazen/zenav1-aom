//! Hand-written lane kernels — the 1-D transforms whose scalar structure is
//! not the regular ping-pong butterfly stream (`fdct.rs` / `special.rs`):
//! `fdct4`, `iadst4` (all-i64 math), `fadst4` (sinpi, i32 wrapping), and the
//! eight identity transforms. Per-lane bit-identical to the scalar ports on
//! the full i32 domain (module docs in `super` + the `tests` differential).
//!
//! Like the transpiled kernels, each of these is ONE
//! `#[magetypes(define(i32x8), v3, neon, -scalar)]` body: the macro emits the per-tier
//! variants, and every architecture-specific helper (`hb`, `rshiftv`,
//! `mul_rshiftv`, and the i64-lane `V64` family used by `iadst4`) is supplied by
//! `super::prims` under a single cfg-selected name.

use archmage::prelude::*;
use magetypes::simd::generic::i32x8 as I32x8;

use crate::transform::cospi::{NEW_INV_SQRT2, NEW_SQRT2, NEW_SQRT2_BITS, cospi_arr, sinpi_arr};

use super::prims::{add64, hb, mul_rshiftv, mulc64, rshift64, rshiftv, sub64, widen64};

/// Lane twin of [`crate::transform::av1_fdct4`] (`fdct.rs`) — wrapping stage-1 adds,
/// four `half_btf`s, output permutation. Statement-for-statement.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_fdct4_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    cos_bit: i32,
    _stage_range: &[i8],
) {
    let cospi = cospi_arr(cos_bit);

    // stage 1 (wrapping adds; `-a + b` == `b - a` in two's complement)
    let bf1_0 = input[0] + input[3];
    let bf1_1 = input[1] + input[2];
    let bf1_2 = input[1] - input[2];
    let bf1_3 = input[0] - input[3];

    // stage 2
    let step0 = hb(t, cospi[32], bf1_0, cospi[32], bf1_1, cos_bit);
    let step1 = hb(t, -cospi[32], bf1_1, cospi[32], bf1_0, cos_bit);
    let step2 = hb(t, cospi[48], bf1_2, cospi[16], bf1_3, cos_bit);
    let step3 = hb(t, cospi[48], bf1_3, -cospi[16], bf1_2, cos_bit);

    // stage 3 (permutation)
    output[0] = step0;
    output[1] = step2;
    output[2] = step1;
    output[3] = step3;
}

/// Lane twin of [`crate::transform::av1_iadst4`] (`special.rs`) — the all-i64 sinpi
/// kernel. The scalar's all-zero-input early-out is an optimization, not a
/// semantic branch: on zero input every product/sum is 0 and
/// `round_shift(0, bit) == 0`, so computing through is bit-identical (the
/// differential mixes zero and nonzero columns to pin this).
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_iadst4_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    cos_bit: i32,
    _stage_range: &[i8],
) {
    let bit = cos_bit;
    let sinpi = sinpi_arr(bit);
    let x0 = widen64(input[0]);
    let x1 = widen64(input[1]);
    let x2 = widen64(input[2]);
    let x3 = widen64(input[3]);

    // stage 1
    let s0 = mulc64(x0, sinpi[1]);
    let s1 = mulc64(x0, sinpi[2]);
    let s2 = mulc64(x1, sinpi[3]);
    let s3 = mulc64(x2, sinpi[4]);
    let s4 = mulc64(x2, sinpi[1]);
    let s5 = mulc64(x3, sinpi[2]);
    let s6 = mulc64(x3, sinpi[4]);
    // stage 2
    let s7 = add64(sub64(x0, x2), x3);
    // stage 3 (the C reuse: s3 <- old s2, s2 <- sinpi[3]*s7)
    let s0 = add64(s0, s3);
    let s1 = sub64(s1, s4);
    let s3 = s2;
    let s2 = mulc64(s7, sinpi[3]);
    // stage 4
    let s0 = add64(s0, s5);
    let s1 = sub64(s1, s6);
    // stage 5
    let x0 = add64(s0, s3);
    let x1 = add64(s1, s3);
    let x2 = s2;
    let x3 = add64(s0, s1);
    // stage 6
    let x3 = sub64(x3, s3);

    output[0] = rshift64(t, x0, bit);
    output[1] = rshift64(t, x1, bit);
    output[2] = rshift64(t, x2, bit);
    output[3] = rshift64(t, x3, bit);
}

/// Lane twin of [`crate::transform::av1_fadst4`] (`special.rs`) — i32 wrapping sinpi
/// products/sums (lane mul/add/sub wrap identically), i64 `round_shift` at
/// the end. Same compute-through argument for the zero early-out as iadst4.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_fadst4_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    cos_bit: i32,
    _stage_range: &[i8],
) {
    let bit = cos_bit;
    let sinpi = sinpi_arr(bit);
    let sp = |k: usize| i32x8::splat(t, sinpi[k]);
    let (x0, x1, x2, x3) = (input[0], input[1], input[2], input[3]);

    // stage 1 — wrapping i32 products (lane mul == wrapping_mul)
    let s0 = sp(1) * x0;
    let s1 = sp(4) * x0;
    let s2 = sp(2) * x1;
    let s3 = sp(1) * x1;
    let s4 = sp(3) * x2;
    let s5 = sp(4) * x3;
    let s6 = sp(2) * x3;
    let s7 = x0 + x1;
    // stage 2
    let s7 = s7 - x3;
    // stage 3
    let x0 = s0 + s2;
    let x1 = sp(3) * s7;
    let x2 = s1 - s3;
    let x3 = s4;
    // stage 4
    let x0 = x0 + s5;
    let x2 = x2 + s6;
    // stage 5
    let s0 = x0 + x3;
    let s1 = x1;
    let s2 = x2 - x3;
    let s3 = x2 - x0;
    // stage 6
    let s3 = s3 + x3;

    output[0] = rshiftv(t, s0, bit);
    output[1] = rshiftv(t, s1, bit);
    output[2] = rshiftv(t, s2, bit);
    output[3] = rshiftv(t, s3, bit);
}

// ---- identity transforms ---------------------------------------------------
// iidentity8/32 and fidentity8/32 are wrapping doublings/quadruplings (the
// scalar `(x as i64 * 2) as i32` / `wrapping_mul(2)` — identical mod 2^32);
// the 4/16 variants are `round_shift(x * NewSqrt2-multiples, 12)` — the
// full-i64-product [`mul_rshiftv`] recipe.

/// `av1_iidentity4_c`: `round_shift(NewSqrt2 * x, NewSqrt2Bits)`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity4_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..4 {
        output[i] = mul_rshiftv(t, input[i], NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_iidentity8_c`: `(x as i64 * 2) as i32` == wrapping `x << 1`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity8_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..8 {
        output[i] = input[i].shl_const::<1>();
    }
}

/// `av1_iidentity16_c`: `round_shift(NewSqrt2 * 2 * x, NewSqrt2Bits)`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity16_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..16 {
        output[i] = mul_rshiftv(t, input[i], 2 * NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_iidentity32_c`: `(x as i64 * 4) as i32` == wrapping `x << 2`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity32_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..32 {
        output[i] = input[i].shl_const::<2>();
    }
}

/// `av1_fidentity4_c`: `round_shift(x * NewSqrt2, NewSqrt2Bits)`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity4_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..4 {
        output[i] = mul_rshiftv(t, input[i], NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_fidentity8_c`: `x.wrapping_mul(2)`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity8_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..8 {
        output[i] = input[i].shl_const::<1>();
    }
}

/// `av1_fidentity16_c`: `round_shift(x * 2 * NewSqrt2, NewSqrt2Bits)`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity16_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..16 {
        output[i] = mul_rshiftv(t, input[i], 2 * NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_fidentity32_c`: `x.wrapping_mul(4)`.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity32_impl(
    t: Token,
    input: &[I32x8<Token>],
    output: &mut [I32x8<Token>],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..32 {
        output[i] = input[i].shl_const::<2>();
    }
}

/// `NEW_INV_SQRT2` re-export site check: the inverse row pass scales by it.
#[allow(dead_code)]
const _ASSERT_INV_SQRT2: i32 = NEW_INV_SQRT2;
