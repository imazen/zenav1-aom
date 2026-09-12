//! **i32-lane** vector kernel for the inner loop of
//! [`super::cfl::cfl_predict_hbd`] — `dst = clip(dst + scaled_ac)` — which was
//! pure scalar while libaom dispatches `cfl_predict_highbd_avx2` on the same
//! path.
//!
//! # The identity, and why i32 lanes are exact
//!
//! The scalar kernel computes, per sample,
//!
//! ```text
//! v      = alpha_q3 * ac_q3                    (i32 product)
//! scaled = sign(v) * ((|v| + 32) >> 6)         (ROUND_POWER_OF_TWO_SIGNED(v, 6))
//! out    = clamp(scaled + dst, 0, (1 << bd) - 1)
//! ```
//!
//! The vector form keeps every step in i32 lanes:
//!
//! * `alpha_q3 * ac_q3` — `ac_q3` is `i16` and `alpha_q3` is an `i32` scalar;
//!   the lane multiply is `pmulld`-equivalent wrapping i32, **bit-identical to
//!   the scalar `i32` product for ANY inputs**, so no reach bound is needed.
//! * `sign * ((|v| + 32) >> 6)` — computed as `(v ^ sgn) - sgn` for `|v|`,
//!   then `(|v| + 32) >> 6` (non-negative, so arithmetic == logical shift),
//!   then `(r ^ sgn) - sgn` re-applies the sign. Each step is exact.
//! * `scaled + dst` — `scaled ∈ ±(i16::MAX * |alpha| + 32) >> 6` and
//!   `dst <= 4095`, so the sum fits i32 with room; the `clamp(0, max)` is the
//!   scalar's `clip_pixel_highbd` verbatim.
//!
//! Rows are contiguous (`ac` strides `CFL_BUF_LINE`, `dst` strides
//! `dst_stride`) and CfL widths are multiples of 4 (`TX_W` ∈ {4, 8, 16, 32}),
//! so each row runs `i32x8` chunks plus at most one `i32x4` chunk; the scalar
//! remainder loop exists only to keep the function total over any `width`.
//!
//! The narrowing `as u16` on the clamped value is exact (it is already in
//! `[0, 4095]`).

use archmage::prelude::*;

use super::cfl::CFL_BUF_LINE;

/// Scalar per-sample recipe — the differential reference AND the
/// scalar-tier/remainder path. Byte-identical to the loop in
/// [`super::cfl::cfl_predict_hbd`].
#[inline]
pub(crate) fn cfl_predict_row_scalar(
    ac: &[i16],
    dst: &mut [u16],
    alpha_q3: i32,
    max: i32,
    n: usize,
) {
    for i in 0..n {
        dst[i] = super::cfl::clip_pixel_highbd(
            super::cfl::scaled_luma_q0(alpha_q3, ac[i]) + i32::from(dst[i]),
            max,
        );
    }
}

/// Dispatch entry: `dst[r][i] = clip(dst[r][i] + scaled(ac[r][i]))` for
/// `r < height`, `i < width`.
pub(crate) fn cfl_predict_scaled_add(
    ac: &[i16],
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    alpha_q3: i32,
    max: i32,
    width: usize,
    height: usize,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cfl_predict_impl(ac, dst, dst_off, dst_stride, alpha_q3, max, width, height),
        [v3, neon, wasm128, scalar]
    )
}

fn cfl_predict_impl_scalar(
    _t: archmage::ScalarToken,
    ac: &[i16],
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    alpha_q3: i32,
    max: i32,
    width: usize,
    height: usize,
) {
    let mut ac_off = 0usize;
    let mut off = dst_off;
    for _ in 0..height {
        cfl_predict_row_scalar(&ac[ac_off..ac_off + width], &mut dst[off..off + width], alpha_q3, max, width);
        ac_off += CFL_BUF_LINE;
        off += dst_stride;
    }
}

#[magetypes(define(i32x8, i32x4), v3, neon, wasm128, -scalar)]
fn cfl_predict_impl(
    token: Token,
    ac: &[i16],
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    alpha_q3: i32,
    max: i32,
    width: usize,
    height: usize,
) {
    let alpha8 = i32x8::splat(token, alpha_q3);
    let c32_8 = i32x8::splat(token, 32);
    let zero8 = i32x8::zero(token);
    let max8 = i32x8::splat(token, max);
    let alpha4 = i32x4::splat(token, alpha_q3);
    let c32_4 = i32x4::splat(token, 32);
    let zero4 = i32x4::zero(token);
    let max4 = i32x4::splat(token, max);
    for r in 0..height {
        let arow = r * CFL_BUF_LINE;
        let drow = dst_off + r * dst_stride;
        let mut i = 0;
        while i + 8 <= width {
            let a: [i16; 8] = ac[arow + i..arow + i + 8].try_into().unwrap();
            let v = i32x8::from_array(token, a.map(i32::from)) * alpha8;
            let sgn = v.shr_arithmetic_const::<31>();
            let scaled = (((((v ^ sgn) - sgn) + c32_8).shr_arithmetic_const::<6>()) ^ sgn) - sgn;
            let d: [u16; 8] = dst[drow + i..drow + i + 8].try_into().unwrap();
            let sum = scaled + i32x8::from_array(token, d.map(i32::from));
            let out = sum.clamp(zero8, max8).to_array();
            dst[drow + i..drow + i + 8].copy_from_slice(&out.map(|x| x as u16));
            i += 8;
        }
        if i + 4 <= width {
            let a: [i16; 4] = ac[arow + i..arow + i + 4].try_into().unwrap();
            let v = i32x4::from_array(token, a.map(i32::from)) * alpha4;
            let sgn = v.shr_arithmetic_const::<31>();
            let scaled = (((((v ^ sgn) - sgn) + c32_4).shr_arithmetic_const::<6>()) ^ sgn) - sgn;
            let d: [u16; 4] = dst[drow + i..drow + i + 4].try_into().unwrap();
            let sum = scaled + i32x4::from_array(token, d.map(i32::from));
            let out = sum.clamp(zero4, max4).to_array();
            dst[drow + i..drow + i + 4].copy_from_slice(&out.map(|x| x as u16));
            i += 4;
        }
        if i < width {
            cfl_predict_row_scalar(
                &ac[arow + i..arow + width],
                &mut dst[drow + i..drow + width],
                alpha_q3,
                max,
                width - i,
            );
        }
    }
}
