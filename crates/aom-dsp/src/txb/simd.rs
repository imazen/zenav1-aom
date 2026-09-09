//! SIMD column kernel for `txb_init_levels` — bit-identical to the
//! scalar port on the FULL i32 domain, at every dispatch tier.
//!
//! Same aom-rs SIMD pattern as `crate::quant::simd` / `crate::cdef::simd`: the
//! magetypes kernel handles heights 8/16/32 (whole 8-lane column chunks);
//! the `_scalar` incant variant and the height-4 route call the transcribed
//! port verbatim ([`crate::txb::txb_init_levels_scalar`]).
//!
//! # Bit-exactness (full domain)
//!
//! Per coefficient the scalar port computes `unsigned_abs().min(127) as u8`.
//! Lanes compute `a = (x ^ (x>>31)) - (x>>31)` (wrapping — `i32::MIN` stays
//! `i32::MIN`), then `blend(a < 0, 127, min(a, 127))`: the only negative `a`
//! is the `i32::MIN` lane, whose `unsigned_abs() = 2^31` also clamps to 127
//! in the scalar port. Everything else is `0 <= a <= i32::MAX`, where the
//! signed lane `min` equals the scalar's unsigned min. The final `as u8`
//! narrowing writes values already in `0..=127`.

use archmage::prelude::*;

use crate::txb::{TX_PAD_BOTTOM, TX_PAD_END, TX_PAD_HOR};

/// Scalar tier = the transcribed port, verbatim.
pub(crate) fn txb_init_levels_impl_scalar(
    _t: archmage::ScalarToken,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    crate::txb::txb_init_levels_scalar(coeff, width, height, levels)
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
pub(crate) fn txb_init_levels_impl(
    token: Token,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    let stride = height + TX_PAD_HOR;
    let tail = stride * width;
    levels[tail..tail + TX_PAD_BOTTOM * stride + TX_PAD_END].fill(0);

    let zero = i32x8::zero(token);
    let cap = i32x8::splat(token, i8::MAX as i32);
    // |x|.min(127) with the i32::MIN lane mapping to 127 exactly like the
    // scalar port's unsigned_abs().min(127) — see module docs.
    let abs127 = |x: i32x8| {
        let m = x.shr_arithmetic_const::<31>();
        let a = (x ^ m) - m; // |x| (wrapping; i32::MIN stays negative)
        i32x8::blend(a.simd_lt(zero), cap, a.min(cap))
    };

    if height == 4 {
        // One 8-lane vector = TWO 4-coeff columns; each column's 4 levels +
        // 4 pad zeros are 8 output bytes, so a pair writes bytes 0..4 and
        // 8..12 of a 16-byte window. Widths are powers of two >= 4.
        debug_assert!(width % 2 == 0);
        debug_assert_eq!(stride, 8, "height 4 + TX_PAD_HOR 4");
        for p in 0..width / 2 {
            let a = abs127(i32x8::from_slice(token, &coeff[p * 8..p * 8 + 8])).to_array();
            let out = &mut levels[p * 2 * stride..p * 2 * stride + 2 * stride];
            // One 8-byte store per column instead of four byte stores plus a
            // 4-byte fill: `stride == 8` here, so each column's four levels and
            // four pad zeros are exactly one aligned run. Values are already in
            // `0..=127` (module docs), so `as u8` is exact.
            out[..8].copy_from_slice(&[a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8, 0, 0, 0, 0]);
            out[stride..stride + 8]
                .copy_from_slice(&[a[4] as u8, a[5] as u8, a[6] as u8, a[7] as u8, 0, 0, 0, 0]);
        }
        return;
    }

    assert!(height % 8 == 0);
    for i in 0..width {
        let col = &coeff[i * height..(i + 1) * height];
        let out = &mut levels[i * stride..i * stride + stride];
        for c in 0..height / 8 {
            let a = abs127(i32x8::from_slice(token, &col[c * 8..c * 8 + 8])).to_array();
            // ONE 8-byte store, not eight byte stores. magetypes 0.9.28 has no
            // i32->u8 narrowing primitive (no pack/narrow/shuffle — only
            // `blend`), so libaom's `_mm256_packs_epi32` + `_mm256_packus_epi16`
            // shape is not expressible here; building the run and storing it
            // once is what is available, and it is where this kernel's time was
            // going — it was the top `__memset`/store caller in the frame-pointer
            // profile despite already being SIMD.
            let bytes = [
                a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8,
                a[4] as u8, a[5] as u8, a[6] as u8, a[7] as u8,
            ];
            out[c * 8..c * 8 + 8].copy_from_slice(&bytes);
        }
        out[height..height + TX_PAD_HOR].fill(0);
    }
}
