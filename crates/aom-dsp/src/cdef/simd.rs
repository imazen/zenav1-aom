//! SIMD row kernel for `cdef_filter_block_16` — bit-identical to the
//! scalar core on the structural CDEF domain, at every dispatch tier
//! (`tests/cdef_filter_simd_diff.rs`).
//!
//! Same aom-rs SIMD pattern as `crate::quant::simd`: ONE magetypes generic
//! kernel (`#[magetypes(v3, neon, wasm128, -scalar)]`), a hand-written
//! `_scalar` variant that IS the transcribed scalar core, `incant!` dispatch,
//! `crate::dispatch::scalar_forced()` pin at the entry.
//!
//! Layout: one `i16x8` vector per block row (width-8 blocks — the luma 8x8
//! path, the bulk of CDEF cost). Width-4 blocks take the scalar core
//! unconditionally for now. All neighbour loads are contiguous 8-lane row
//! loads at the (per-call constant) `cdef_dir` offsets.
//!
//! # Bit-exactness argument (structural domain)
//!
//! The frame walk feeds `in_buf` values that are either real pixels
//! (`<= (1<<bd)-1 <= 4095`) or the border fill `CDEF_VERY_LARGE (16384)`,
//! and strengths from the header (`pri <= 15 << coeff_shift`,
//! `sec <= 4 << coeff_shift`, `coeff_shift = bd-8 <= 4`, damping 3..6 + cs).
//! Within that domain every intermediate fits the scalar core's OWN
//! narrowings, lane for lane:
//! * `diff = p - x` ∈ [-4095, 16384] — exact in i16.
//! * `constrain`: `a = |diff| <= 16384`; `threshold - (a >> shift)` ∈
//!   [-16384, 3840]; the clamp to `[0, a]` and the sign re-apply are lane
//!   min/max/xor-sub — exact. The scalar's `threshold == 0` early-return
//!   equals the formula's value (the clamp floor is 0), and the vector
//!   kernel skips the class entirely in that case (contribution 0) while
//!   still running the min/max updates, exactly like the scalar core.
//! * `tap * constrain` is truncated `as i16` by the scalar core; the lane
//!   `mullo` IS that truncation. The `sum` accumulator wraps i16 in both.
//! * `|sum| <= 12*pri_thr + 12*sec_thr <= 3648` in-domain, so the final
//!   `(8 + sum - (sum<0)) >> 4` fits i16 exactly; `x + adj` wraps i16 in
//!   BOTH implementations (scalar casts through i16 deliberately).
//! * min/max tracking compares nonneg values <= 16384 — exact in i16; the
//!   `p != CDEF_VERY_LARGE` max-exclusion becomes a blend to 0 (never wins
//!   a max whose floor is `x >= 0`); min is unconditional in both.
//!
//! The differential sweeps this domain densely (all bd, all strength/damping
//! combos, VERY_LARGE border mixes) plus the boundary values, at every token
//! permutation.
//!
//! # Also in this module: `cdef_find_dir`'s partial sums
//!
//! The direction search's accumulation half has its own i16-lane kernel at the
//! bottom of this file (`cdef_find_dir_partials`, SIMD_REACH_AUDIT F6). It does
//! NOT share the filter kernels' *structural-domain* bit-exactness argument —
//! its i16 fit rests on a **checked per-call value predicate** instead, so the
//! entry is bit-identical for every input. See the section header there.

use archmage::prelude::*;

use crate::cdef::{CDEF_BSTRIDE, CDEF_VERY_LARGE, PRI_TAPS, SEC_TAPS, cdef_dir, constrain, get_msb};

/// Dispatch entry used by [`crate::cdef::cdef_filter_block_16`] for width-8 blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_16_w8(
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cdef_filter_16_w8_impl(
            dst,
            dst_off,
            dstride,
            in_buf,
            in_off,
            pri_strength,
            sec_strength,
            dir,
            pri_damping,
            sec_damping,
            coeff_shift,
            block_height,
            enable_primary,
            enable_secondary
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed core, verbatim (via the width-8 store shape).
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w8_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    crate::cdef::cdef_filter_block_core(
        in_buf,
        in_off,
        pri_strength,
        sec_strength,
        dir,
        pri_damping,
        sec_damping,
        coeff_shift,
        8,
        block_height,
        enable_primary,
        enable_secondary,
        |i, j, y| dst[dst_off + i * dstride + j] = y as u16,
    );
}

/// `a >> shift` for a per-call runtime shift in 0..=15 (lane-const shifts
/// only exist as const generics; the match is perfectly predicted since the
/// shift is fixed per block).
macro_rules! shr_by {
    ($v:expr, $sh:expr) => {
        match $sh {
            0 => $v,
            1 => $v.shr_arithmetic_const::<1>(),
            2 => $v.shr_arithmetic_const::<2>(),
            3 => $v.shr_arithmetic_const::<3>(),
            4 => $v.shr_arithmetic_const::<4>(),
            5 => $v.shr_arithmetic_const::<5>(),
            6 => $v.shr_arithmetic_const::<6>(),
            7 => $v.shr_arithmetic_const::<7>(),
            8 => $v.shr_arithmetic_const::<8>(),
            9 => $v.shr_arithmetic_const::<9>(),
            10 => $v.shr_arithmetic_const::<10>(),
            11 => $v.shr_arithmetic_const::<11>(),
            12 => $v.shr_arithmetic_const::<12>(),
            13 => $v.shr_arithmetic_const::<13>(),
            14 => $v.shr_arithmetic_const::<14>(),
            _ => $v.shr_arithmetic_const::<15>(),
        }
    };
}

/// Width-4 dispatch entry (two rows per 8-lane vector; `block_height` must be
/// even — the caller routes odd heights to the scalar core).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_16_w4(
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cdef_filter_16_w4_impl(
            dst,
            dst_off,
            dstride,
            in_buf,
            in_off,
            pri_strength,
            sec_strength,
            dir,
            pri_damping,
            sec_damping,
            coeff_shift,
            block_height,
            enable_primary,
            enable_secondary
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed core, verbatim (width-4 store shape).
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w4_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    crate::cdef::cdef_filter_block_core(
        in_buf,
        in_off,
        pri_strength,
        sec_strength,
        dir,
        pri_damping,
        sec_damping,
        coeff_shift,
        4,
        block_height,
        enable_primary,
        enable_secondary,
        |i, j, y| dst[dst_off + i * dstride + j] = y as u16,
    );
}

#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w4_impl(
    token: Token,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    assert!(block_height % 2 == 0, "caller routes odd heights to scalar");
    let clipping_required = enable_primary && enable_secondary;
    let s = CDEF_BSTRIDE as i32;
    let pri_taps = &PRI_TAPS[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = &SEC_TAPS;
    let pri_shift = if pri_strength != 0 {
        (pri_damping - get_msb(pri_strength as u32)).max(0)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        (sec_damping - get_msb(sec_strength as u32)).max(0)
    } else {
        0
    };

    let zero = i16x8::zero(token);
    let eight = i16x8::splat(token, 8);
    let vl = i16x8::splat(token, CDEF_VERY_LARGE as i16);
    let pri_t = i16x8::splat(token, pri_strength as i16);
    let sec_t = i16x8::splat(token, sec_strength as i16);

    let constrain_v = |d: i16x8, thr: i16x8, shift: i32| -> i16x8 {
        let m = d.shr_arithmetic_const::<15>();
        let a = (d ^ m) - m;
        let c = (thr - shr_by!(a, shift)).clamp(zero, a);
        (c ^ m) - m
    };

    // Two-row gather: lanes [row_i .. 4px, row_i+1 .. 4px].
    let load2 = |idx: i32| -> i16x8 {
        let a = idx as usize;
        let b = (idx + s) as usize;
        let mut arr = [0u16; 8];
        arr[..4].copy_from_slice(&in_buf[a..a + 4]);
        arr[4..].copy_from_slice(&in_buf[b..b + 4]);
        u16x8::from_array(token, arr).bitcast_i16x8()
    };

    let mut i = 0i32;
    while (i as usize) < block_height {
        let base = in_off as i32 + i * s;
        let x = load2(base);
        let mut sum = zero;
        let mut maxv = x;
        let mut minv = x;
        for k in 0..2usize {
            if enable_primary {
                let off = cdef_dir(dir, k);
                let p0 = load2(base + off);
                let p1 = load2(base - off);
                if pri_strength != 0 {
                    let tap = i16x8::splat(token, pri_taps[k] as i16);
                    sum = sum + tap * constrain_v(p0 - x, pri_t, pri_shift);
                    sum = sum + tap * constrain_v(p1 - x, pri_t, pri_shift);
                }
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(p0.simd_eq(vl), zero, p0));
                    maxv = maxv.max(i16x8::blend(p1.simd_eq(vl), zero, p1));
                    minv = minv.min(p0);
                    minv = minv.min(p1);
                }
            }
            if enable_secondary {
                let o0 = cdef_dir(dir + 2, k);
                let o1 = cdef_dir(dir - 2, k);
                let s0 = load2(base + o0);
                let s1 = load2(base - o0);
                let s2 = load2(base + o1);
                let s3 = load2(base - o1);
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(s0.simd_eq(vl), zero, s0));
                    maxv = maxv.max(i16x8::blend(s1.simd_eq(vl), zero, s1));
                    maxv = maxv.max(i16x8::blend(s2.simd_eq(vl), zero, s2));
                    maxv = maxv.max(i16x8::blend(s3.simd_eq(vl), zero, s3));
                    minv = minv.min(s0).min(s1).min(s2).min(s3);
                }
                if sec_strength != 0 {
                    let tap = i16x8::splat(token, sec_taps[k] as i16);
                    sum = sum + tap * constrain_v(s0 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s1 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s2 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s3 - x, sec_t, sec_shift);
                }
            }
        }
        let m = sum.shr_arithmetic_const::<15>();
        let adj = (sum + m + eight).shr_arithmetic_const::<4>();
        let mut y = x + adj;
        if clipping_required {
            y = y.max(minv).min(maxv);
        }
        let out = y.bitcast_u16x8().to_array();
        let r0 = dst_off + i as usize * dstride;
        let r1 = dst_off + (i as usize + 1) * dstride;
        dst[r0..r0 + 4].copy_from_slice(&out[..4]);
        dst[r1..r1 + 4].copy_from_slice(&out[4..]);
        i += 2;
    }
}

#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w8_impl(
    token: Token,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let clipping_required = enable_primary && enable_secondary;
    let s = CDEF_BSTRIDE as i32;
    let pri_taps = &PRI_TAPS[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = &SEC_TAPS;
    // Per-call constants of constrain(): shift = max(0, damping - msb(thr)).
    // Guarded: msb(0) is undefined — a zero threshold contributes 0 and the
    // vector kernel skips that class's constrain entirely (see module docs).
    let pri_shift = if pri_strength != 0 {
        (pri_damping - get_msb(pri_strength as u32)).max(0)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        (sec_damping - get_msb(sec_strength as u32)).max(0)
    } else {
        0
    };

    let zero = i16x8::zero(token);
    let eight = i16x8::splat(token, 8);
    let vl = i16x8::splat(token, CDEF_VERY_LARGE as i16);
    let pri_t = i16x8::splat(token, pri_strength as i16);
    let sec_t = i16x8::splat(token, sec_strength as i16);

    // constrain() on 8 lanes: sign*(clamp(thr - (|d|>>shift), 0, |d|)).
    // (thr == 0 is handled by the caller skipping the class.)
    let constrain_v = |d: i16x8, thr: i16x8, shift: i32| -> i16x8 {
        let m = d.shr_arithmetic_const::<15>(); // -1 where negative
        let a = (d ^ m) - m; // |d| (wrapping, like the scalar core's i32 path in-domain)
        let c = (thr - shr_by!(a, shift)).clamp(zero, a);
        (c ^ m) - m
    };

    let load = |idx: i32| -> i16x8 {
        u16x8::from_slice(token, &in_buf[idx as usize..idx as usize + 8]).bitcast_i16x8()
    };

    for i in 0..block_height as i32 {
        let base = in_off as i32 + i * s;
        let x = load(base);
        let mut sum = zero;
        let mut maxv = x;
        let mut minv = x;
        for k in 0..2usize {
            if enable_primary {
                let off = cdef_dir(dir, k);
                let p0 = load(base + off);
                let p1 = load(base - off);
                if pri_strength != 0 {
                    let tap = i16x8::splat(token, pri_taps[k] as i16);
                    sum = sum + tap * constrain_v(p0 - x, pri_t, pri_shift);
                    sum = sum + tap * constrain_v(p1 - x, pri_t, pri_shift);
                }
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(p0.simd_eq(vl), zero, p0));
                    maxv = maxv.max(i16x8::blend(p1.simd_eq(vl), zero, p1));
                    minv = minv.min(p0);
                    minv = minv.min(p1);
                }
            }
            if enable_secondary {
                let o0 = cdef_dir(dir + 2, k);
                let o1 = cdef_dir(dir - 2, k);
                let s0 = load(base + o0);
                let s1 = load(base - o0);
                let s2 = load(base + o1);
                let s3 = load(base - o1);
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(s0.simd_eq(vl), zero, s0));
                    maxv = maxv.max(i16x8::blend(s1.simd_eq(vl), zero, s1));
                    maxv = maxv.max(i16x8::blend(s2.simd_eq(vl), zero, s2));
                    maxv = maxv.max(i16x8::blend(s3.simd_eq(vl), zero, s3));
                    minv = minv.min(s0).min(s1).min(s2).min(s3);
                }
                if sec_strength != 0 {
                    let tap = i16x8::splat(token, sec_taps[k] as i16);
                    sum = sum + tap * constrain_v(s0 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s1 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s2 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s3 - x, sec_t, sec_shift);
                }
            }
        }
        // y = x + ((8 + sum - (sum<0)) >> 4), computed exactly as the scalar
        // core does through i16 (in-domain |sum| <= 3648 — see module docs).
        let m = sum.shr_arithmetic_const::<15>(); // -1 where sum < 0
        let adj = (sum + m + eight).shr_arithmetic_const::<4>();
        let mut y = x + adj;
        if clipping_required {
            y = y.max(minv).min(maxv);
        }
        let row = dst_off + i as usize * dstride;
        y.bitcast_u16x8()
            .store((&mut dst[row..row + 8]).try_into().unwrap());
    }
}

// ===== bd8 lowbd (u8) direct-store variants =====
//
// Byte-for-byte the SAME i16-domain filter math as the `cdef_filter_16_*`
// kernels above (loads still come from the u16 `in_buf` work buffer); ONLY the
// destination store narrows the i16 result to `u8` — the `cdef_filter_8_*`
// path. Mirrors the exemplar `transform::simd::try_inv_col_pass_u8` (share the
// math, duplicate only the pixel store). Pinned byte-identical to the u16
// kernels AND to C lowbd by `cdef_lowbd_diff.rs`. Avoids the per-block u16
// scratch round-trip the first `cdef_filter_block_u8` used (measured 582M Ir /
// ~19% of a filter-heavy CDEF pass, benchmarks/cdef_lowbd_ir_2026-07-22.md).

/// Width-8 u8-store dispatch entry, used by [`crate::cdef::cdef_filter_block_u8`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_8_w8(
    dst: &mut [u8],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cdef_filter_8_w8_impl(
            dst,
            dst_off,
            dstride,
            in_buf,
            in_off,
            pri_strength,
            sec_strength,
            dir,
            pri_damping,
            sec_damping,
            coeff_shift,
            block_height,
            enable_primary,
            enable_secondary
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed core, verbatim (width-8 u8 store shape).
#[allow(clippy::too_many_arguments)]
fn cdef_filter_8_w8_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u8],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    crate::cdef::cdef_filter_block_core(
        in_buf,
        in_off,
        pri_strength,
        sec_strength,
        dir,
        pri_damping,
        sec_damping,
        coeff_shift,
        8,
        block_height,
        enable_primary,
        enable_secondary,
        |i, j, y| dst[dst_off + i * dstride + j] = y as u8,
    );
}

/// Width-4 u8-store dispatch entry (two rows per 8-lane vector; even height).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_8_w4(
    dst: &mut [u8],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cdef_filter_8_w4_impl(
            dst,
            dst_off,
            dstride,
            in_buf,
            in_off,
            pri_strength,
            sec_strength,
            dir,
            pri_damping,
            sec_damping,
            coeff_shift,
            block_height,
            enable_primary,
            enable_secondary
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed core, verbatim (width-4 u8 store shape).
#[allow(clippy::too_many_arguments)]
fn cdef_filter_8_w4_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u8],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    crate::cdef::cdef_filter_block_core(
        in_buf,
        in_off,
        pri_strength,
        sec_strength,
        dir,
        pri_damping,
        sec_damping,
        coeff_shift,
        4,
        block_height,
        enable_primary,
        enable_secondary,
        |i, j, y| dst[dst_off + i * dstride + j] = y as u8,
    );
}

#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn cdef_filter_8_w4_impl(
    token: Token,
    dst: &mut [u8],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    assert!(block_height % 2 == 0, "caller routes odd heights to scalar");
    let clipping_required = enable_primary && enable_secondary;
    let s = CDEF_BSTRIDE as i32;
    let pri_taps = &PRI_TAPS[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = &SEC_TAPS;
    let pri_shift = if pri_strength != 0 {
        (pri_damping - get_msb(pri_strength as u32)).max(0)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        (sec_damping - get_msb(sec_strength as u32)).max(0)
    } else {
        0
    };

    let zero = i16x8::zero(token);
    let eight = i16x8::splat(token, 8);
    let vl = i16x8::splat(token, CDEF_VERY_LARGE as i16);
    let pri_t = i16x8::splat(token, pri_strength as i16);
    let sec_t = i16x8::splat(token, sec_strength as i16);

    let constrain_v = |d: i16x8, thr: i16x8, shift: i32| -> i16x8 {
        let m = d.shr_arithmetic_const::<15>();
        let a = (d ^ m) - m;
        let c = (thr - shr_by!(a, shift)).clamp(zero, a);
        (c ^ m) - m
    };

    let load2 = |idx: i32| -> i16x8 {
        let a = idx as usize;
        let b = (idx + s) as usize;
        let mut arr = [0u16; 8];
        arr[..4].copy_from_slice(&in_buf[a..a + 4]);
        arr[4..].copy_from_slice(&in_buf[b..b + 4]);
        u16x8::from_array(token, arr).bitcast_i16x8()
    };

    let mut i = 0i32;
    while (i as usize) < block_height {
        let base = in_off as i32 + i * s;
        let x = load2(base);
        let mut sum = zero;
        let mut maxv = x;
        let mut minv = x;
        for k in 0..2usize {
            if enable_primary {
                let off = cdef_dir(dir, k);
                let p0 = load2(base + off);
                let p1 = load2(base - off);
                if pri_strength != 0 {
                    let tap = i16x8::splat(token, pri_taps[k] as i16);
                    sum = sum + tap * constrain_v(p0 - x, pri_t, pri_shift);
                    sum = sum + tap * constrain_v(p1 - x, pri_t, pri_shift);
                }
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(p0.simd_eq(vl), zero, p0));
                    maxv = maxv.max(i16x8::blend(p1.simd_eq(vl), zero, p1));
                    minv = minv.min(p0);
                    minv = minv.min(p1);
                }
            }
            if enable_secondary {
                let o0 = cdef_dir(dir + 2, k);
                let o1 = cdef_dir(dir - 2, k);
                let s0 = load2(base + o0);
                let s1 = load2(base - o0);
                let s2 = load2(base + o1);
                let s3 = load2(base - o1);
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(s0.simd_eq(vl), zero, s0));
                    maxv = maxv.max(i16x8::blend(s1.simd_eq(vl), zero, s1));
                    maxv = maxv.max(i16x8::blend(s2.simd_eq(vl), zero, s2));
                    maxv = maxv.max(i16x8::blend(s3.simd_eq(vl), zero, s3));
                    minv = minv.min(s0).min(s1).min(s2).min(s3);
                }
                if sec_strength != 0 {
                    let tap = i16x8::splat(token, sec_taps[k] as i16);
                    sum = sum + tap * constrain_v(s0 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s1 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s2 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s3 - x, sec_t, sec_shift);
                }
            }
        }
        let m = sum.shr_arithmetic_const::<15>();
        let adj = (sum + m + eight).shr_arithmetic_const::<4>();
        let mut y = x + adj;
        if clipping_required {
            y = y.max(minv).min(maxv);
        }
        // ONLY difference from cdef_filter_16_w4_impl: narrow the i16 result to
        // u8 (low byte == the C lowbd `(uint8_t)y` store for an in-domain result).
        let out = y.bitcast_u16x8().to_array();
        let r0 = dst_off + i as usize * dstride;
        let r1 = dst_off + (i as usize + 1) * dstride;
        for j in 0..4 {
            dst[r0 + j] = out[j] as u8;
            dst[r1 + j] = out[4 + j] as u8;
        }
        i += 2;
    }
}

#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn cdef_filter_8_w8_impl(
    token: Token,
    dst: &mut [u8],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let clipping_required = enable_primary && enable_secondary;
    let s = CDEF_BSTRIDE as i32;
    let pri_taps = &PRI_TAPS[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = &SEC_TAPS;
    let pri_shift = if pri_strength != 0 {
        (pri_damping - get_msb(pri_strength as u32)).max(0)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        (sec_damping - get_msb(sec_strength as u32)).max(0)
    } else {
        0
    };

    let zero = i16x8::zero(token);
    let eight = i16x8::splat(token, 8);
    let vl = i16x8::splat(token, CDEF_VERY_LARGE as i16);
    let pri_t = i16x8::splat(token, pri_strength as i16);
    let sec_t = i16x8::splat(token, sec_strength as i16);

    let constrain_v = |d: i16x8, thr: i16x8, shift: i32| -> i16x8 {
        let m = d.shr_arithmetic_const::<15>();
        let a = (d ^ m) - m;
        let c = (thr - shr_by!(a, shift)).clamp(zero, a);
        (c ^ m) - m
    };

    let load = |idx: i32| -> i16x8 {
        u16x8::from_slice(token, &in_buf[idx as usize..idx as usize + 8]).bitcast_i16x8()
    };

    for i in 0..block_height as i32 {
        let base = in_off as i32 + i * s;
        let x = load(base);
        let mut sum = zero;
        let mut maxv = x;
        let mut minv = x;
        for k in 0..2usize {
            if enable_primary {
                let off = cdef_dir(dir, k);
                let p0 = load(base + off);
                let p1 = load(base - off);
                if pri_strength != 0 {
                    let tap = i16x8::splat(token, pri_taps[k] as i16);
                    sum = sum + tap * constrain_v(p0 - x, pri_t, pri_shift);
                    sum = sum + tap * constrain_v(p1 - x, pri_t, pri_shift);
                }
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(p0.simd_eq(vl), zero, p0));
                    maxv = maxv.max(i16x8::blend(p1.simd_eq(vl), zero, p1));
                    minv = minv.min(p0);
                    minv = minv.min(p1);
                }
            }
            if enable_secondary {
                let o0 = cdef_dir(dir + 2, k);
                let o1 = cdef_dir(dir - 2, k);
                let s0 = load(base + o0);
                let s1 = load(base - o0);
                let s2 = load(base + o1);
                let s3 = load(base - o1);
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(s0.simd_eq(vl), zero, s0));
                    maxv = maxv.max(i16x8::blend(s1.simd_eq(vl), zero, s1));
                    maxv = maxv.max(i16x8::blend(s2.simd_eq(vl), zero, s2));
                    maxv = maxv.max(i16x8::blend(s3.simd_eq(vl), zero, s3));
                    minv = minv.min(s0).min(s1).min(s2).min(s3);
                }
                if sec_strength != 0 {
                    let tap = i16x8::splat(token, sec_taps[k] as i16);
                    sum = sum + tap * constrain_v(s0 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s1 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s2 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s3 - x, sec_t, sec_shift);
                }
            }
        }
        let m = sum.shr_arithmetic_const::<15>();
        let adj = (sum + m + eight).shr_arithmetic_const::<4>();
        let mut y = x + adj;
        if clipping_required {
            y = y.max(minv).min(maxv);
        }
        // ONLY difference from cdef_filter_16_w8_impl: narrow the i16 result to
        // u8 (low byte == the C lowbd `(uint8_t)y` store for an in-domain result).
        let out = y.bitcast_u16x8().to_array();
        let row = dst_off + i as usize * dstride;
        for j in 0..8 {
            dst[row + j] = out[j] as u8;
        }
    }
}


// ===== cdef_find_dir — the 8x8 direction search partial sums =====
//
// `cdef_find_dir` is ~4.7 % of the q32 decode Ir and had no SIMD tier at all
// (docs/SIMD_REACH_AUDIT_2026-07-28.md finding F6); C reaches
// `cdef_find_dir_avx2`. What this kernel ports is the PARTIAL-SUM half — the
// eight skewed accumulations over the 8x8 block. The cost fold, the argmax and
// its normative tie-break stay in ONE shared copy
// (`crate::cdef::find_dir_cost_fold`), so no bitstream decision is duplicated.
//
// # Why i16 lanes, and why that is still unconditionally bit-identical
//
// The scalar port accumulates in i32 because `cdef_find_dir_c` uses `int`.
// LLVM already vectorises its per-row slice adds, so an i32-lane rewrite buys
// nothing — the win is halving the lane width. i16 lanes are exact **iff**
// every partial fits i16, which follows from a single checked condition,
// `(px >> coeff_shift) <= 255` (`crate::cdef::cdef_find_dir_simd_eligible`,
// evaluated per call — an assumption would be a silent-wrong-pixel bug, so
// there is none):
//
// * `line = (px >> cs) - 128` is then in `[-128, 127]`.
// * `partial[0]`, `partial[4]`, `partial[5]`, `partial[6]`, `partial[7]` take
//   at most 8 line contributions per slot -> `|.| <= 1024`.
// * `partial[2][i]` is one row's sum -> `|.| <= 1024`.
// * `partial[1]`, `partial[3]` accumulate pair folds (`|pf| <= 256`), at most
//   8 per slot -> `|.| <= 2048`.
//
// All well inside i16, so each i16 partial holds the SAME INTEGER as the
// scalar port's i32 partial, and the shared fold (which widens back to i32 and
// keeps the C's `wrapping_*` ops) produces the same `(dir, var)` bit for bit.
// Outside the checked condition the entry runs `cdef_find_dir_scalar` instead;
// nothing about the frame walk's behaviour is assumed.
//
// # Two rewrites that drop work the scalar port pays for
//
// 1. **No lane reversals.** The scalar port materialises `rev` (reversed row,
//    for d4) and `pfr` (reversed pair fold, for d3). Here d3/d4 accumulate into
//    REVERSED slot order instead — `q4[m] = partial[4][14 - m]`, i.e.
//    `q4[7 - i + j] += row[j]`, and `q3[m] = partial[3][10 - m]`, i.e.
//    `q3[7 - i + k] += pf[k]` — which is a plain forward slice add at a
//    per-row offset. `cdef_find_dir` undoes the flip in its accessor.
// 2. **Pair folding for d5/d6/d7.** Their per-row offsets are `3 - i/2`, `0`
//    and `i/2`, so rows `2k` and `2k+1` share one offset: add the two rows
//    once (`tmp`) and slice-add `tmp` instead of each row (libaom's SIMD does
//    the same). Wrapping adds commute, so both rewrites are exact regroupings.
//
// Remaining shape gap vs `cdef_find_dir_avx2`: it keeps the accumulators in
// registers and shifts LANES (`v128_shl_n_byte`), where this kernel keeps them
// in a stack table and shifts the ADDRESS. magetypes 0.9.27 has no lane
// shift / permute / integer widen for i16 lanes (checked against the 0.9.28
// public-API snapshot), so register residency needs per-tier `#[rite]`
// primitives — chunk 2, see benchmarks/cdef_find_dir_simd_2026-07-28.md.

/// Dispatch entry for the [`crate::cdef::cdef_find_dir`] partial sums.
///
/// On `true`, `pa[d]` holds direction `d`'s partial sums — slots `0..=14` for
/// d0/d4, `0..=10` for d1/d3/d5/d7, `0..=7` for d2/d6 — with **d3 and d4 in
/// reversed slot order** (see the module note above). On `false` nothing was
/// computed and the caller must run the scalar search.
pub(crate) fn cdef_find_dir_partials(
    img: &[u16],
    stride: usize,
    coeff_shift: i32,
    pa: &mut [[i16; 16]; 8],
) -> bool {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cdef_find_dir_partials_impl(img, stride, coeff_shift, pa),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier: decline, so the entry runs the transcribed
/// [`crate::cdef::cdef_find_dir_scalar`] — the same "scalar tier IS the scalar
/// port" contract the filter kernels above hold, expressed as a route rather
/// than a second transcription. This is what `AOM_FORCE_SCALAR=1` reaches.
fn cdef_find_dir_partials_impl_scalar(
    _t: archmage::ScalarToken,
    _img: &[u16],
    _stride: usize,
    _coeff_shift: i32,
    _pa: &mut [[i16; 16]; 8],
) -> bool {
    false
}

/// `v >> sh` (logical) for a per-call runtime `sh` in `0..=7`; lane-const
/// shifts only exist as const generics. Same shape as `shr_by!` above.
macro_rules! shr_u16_by {
    ($v:expr, $sh:expr) => {
        match $sh {
            0 => $v,
            1 => $v.shr_logical_const::<1>(),
            2 => $v.shr_logical_const::<2>(),
            3 => $v.shr_logical_const::<3>(),
            4 => $v.shr_logical_const::<4>(),
            5 => $v.shr_logical_const::<5>(),
            6 => $v.shr_logical_const::<6>(),
            _ => $v.shr_logical_const::<7>(),
        }
    };
}

#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]
fn cdef_find_dir_partials_impl(
    token: Token,
    img: &[u16],
    stride: usize,
    coeff_shift: i32,
    pa: &mut [[i16; 16]; 8],
) -> bool {
    // ---- the checked eligibility condition (see the module note) ----
    // `(px >> cs) <= 255` for every pixel <=> `px <= (256 << cs) - 1`. The
    // shift bound also keeps that limit inside u16.
    if !(0..=7).contains(&coeff_shift) {
        return false;
    }
    let limit = u16x8::splat(token, ((256u32 << coeff_shift) - 1) as u16);
    let mut over = u16x8::zero(token);
    let mut raw = [u16x8::zero(token); 8];
    for (i, slot) in raw.iter_mut().enumerate() {
        let o = i * stride;
        let r = u16x8::from_slice(token, &img[o..o + 8]);
        over |= r.simd_gt(limit);
        *slot = r;
    }
    if over.any_true() {
        return false;
    }

    // ---- lines: (px >> cs) - 128, exactly the scalar port's `row` ----
    let c128 = i16x8::splat(token, 128);
    let mut line = [i16x8::zero(token); 8];
    for (slot, r) in line.iter_mut().zip(raw) {
        *slot = shr_u16_by!(r, coeff_shift).bitcast_i16x8() - c128;
    }

    *pa = [[0i16; 16]; 8];

    // Load / add / store one 8-lane run at `off` (every `off` here is <= 7, so
    // the fixed-size-array conversion is in bounds by construction).
    let add8 = |acc: &mut [i16; 16], off: usize, v: i16x8| {
        let dst: &mut [i16; 8] = (&mut acc[off..off + 8]).try_into().unwrap();
        (i16x8::load(token, &*dst) + v).store(dst);
    };
    // pf[k] = row[2k] + row[2k+1] in the low 4 lanes, 0 in the high 4 (the
    // zero lanes land on unread slots, and adding 0 is a no-op regardless).
    let fold = |v: i16x8| -> i16x8 {
        let a = v.to_array();
        i16x8::from_array(
            token,
            [
                a[0].wrapping_add(a[1]),
                a[2].wrapping_add(a[3]),
                a[4].wrapping_add(a[5]),
                a[6].wrapping_add(a[7]),
                0,
                0,
                0,
                0,
            ],
        )
    };

    let mut a6 = i16x8::zero(token);
    for k in 0..4usize {
        let (i0, i1) = (2 * k, 2 * k + 1);
        let (r0, r1) = (line[i0], line[i1]);

        // d0: partial[0][i + j] += row[j]
        add8(&mut pa[0], i0, r0);
        add8(&mut pa[0], i1, r1);
        // d4 reversed: q4[7 - i + j] += row[j]
        add8(&mut pa[4], 7 - i0, r0);
        add8(&mut pa[4], 7 - i1, r1);

        let (p0, p1) = (fold(r0), fold(r1));
        // d1: partial[1][i + k] += pf[k]
        add8(&mut pa[1], i0, p0);
        add8(&mut pa[1], i1, p1);
        // d3 reversed: q3[7 - i + k] += pf[k]
        add8(&mut pa[3], 7 - i0, p0);
        add8(&mut pa[3], 7 - i1, p1);

        // d2: partial[2][i] = sum(row)
        pa[2][i0] = r0.reduce_add();
        pa[2][i1] = r1.reduce_add();

        // d5/d6/d7 share one offset per ROW PAIR, so fold the pair once.
        let tmp = r0 + r1;
        add8(&mut pa[5], 3 - k, tmp); // offset 3 - i/2
        add8(&mut pa[7], k, tmp); // offset i/2
        a6 += tmp; // offset 0
    }
    a6.store((&mut pa[6][..8]).try_into().unwrap());
    true
}

