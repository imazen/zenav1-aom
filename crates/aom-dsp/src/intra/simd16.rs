//! **`u16`-lane** vector kernels for the SMOOTH family of intra predictors —
//! SMOOTH / SMOOTH_V / SMOOTH_H — plus the runtime bound that admits them.
//!
//! # PAETH was built here too, measured, and REVERTED
//!
//! The named lever was "smooth + paeth". An `i16x16` PAETH kernel of exactly
//! this shape was written, audited (`M* = 16383`, admitting bd8/10/12 —
//! `xtask/audit_nd16_lanes.py` still derives that bound), differentially gated
//! at every tier, and **measured as a dead null in four independent bands
//! (~165 interleaved rounds, two different store shapes): +0.08 / −0.01 /
//! +0.09 / +0.04 %, sign-test p = 0.65 / 1.00 / 0.29 / 0.39.** In the one band
//! whose arm ORDER was rotated so position could not confound it, the composed
//! smooth+paeth binary and the smooth-only binary were indistinguishable
//! (−0.007 %, 25/49, p = 1.00). It was reverted rather than shipped unmeasured;
//! the reasoning, the mechanism (PAETH is store-bound, not arithmetic-bound, so
//! doubling the lane width buys nothing) and the bands are in
//! `benchmarks/encoder_intra_smooth_paeth_2026-08-03.md` §4.
//!
//! # Why this exists
//!
//! [`super::simd`] already vectorizes these four predictors, but at **`i32x8`**:
//! one 32-bit lane per 8-bit sample, because that is the width the scalar core
//! computes in and nothing then needs proving. libaom runs the same predictors
//! on **narrow lanes** — `vmull_u8` / `vmlal_u8` / `vhaddq_u16` / `vrshrn_n_u16`
//! for the SMOOTH family (`aom_dsp/arm/intrapred_neon.c:2383-2520`), and the
//! `_avx2` / `_sse4_1` twins do the same — so an equal-width register carries
//! **twice** the samples. Measured at the profile cell the two `i32x8` kernels
//! were **3.18 ms of a 155 ms encode against libaom's 0.33 ms** (9.7x), of
//! which SMOOTH alone is 2.18 ms; see
//! `benchmarks/encoder_intra_smooth_paeth_2026-08-03.md`.
//!
//! This is the lane-width programme proper, and unlike
//! [`super::dir_simd`] — where neither port path was vectorized at all, so no
//! lane-width choice existed — the framing fits: the port ran `i32x8` where
//! libaom runs `u8`/`u16`.
//!
//! Each body is ONE `#[magetypes(...)]` function, so the AVX2 and NEON tiers
//! come from the same source (`docs/DIFFERENTIAL_PLAYBOOK.md` §6b: an
//! arithmetic mechanism travels across platforms; a platform call does not).
//!
//! # The bounds, and why they differ per kernel
//!
//! Derived and checked by **exhaustive enumeration** in
//! `xtask/audit_nd16_lanes.py` — the whole `(weight, sample, sample)` product
//! space, not an inequality. With weights `<= 255` by type (`SMOOTH_WEIGHTS` is
//! a `u8` table) and every sample in `[0, M]`:
//!
//! | kernel | lane | binding intermediate | `M*` | admits |
//! |---|---|---|---|---|
//! | SMOOTH | `u16` | `((A+B)>>1) + 128 = 65408` | **255** | bd8 |
//! | SMOOTH_V / SMOOTH_H | `u16` | `w*a + (256-w)*b + 128 = 65408` | **255** | bd8 |
//! | PAETH | `i16` | `base - top_left ∈ [-2M, 2M]` | **16383** | bd8, bd10, bd12 |
//!
//! Both are **tight**: at `M = 256` the term `(256-w)*b` is exactly `65536` and
//! leaves `u16`; at `M = 16384` PAETH's `base` reaches `32768` and leaves `i16`.
//! Neither was widened to admit anything.
//!
//! ## SMOOTH needs a halving add, and it must be the TRUNCATING one
//!
//! SMOOTH's full numerator `p = wh*above + (256-wh)*below + ww*left +
//! (256-ww)*right` reaches `2*256*M`, which is outside `u16` for every
//! `M >= 128`. So the two halves stay separate and combine the way libaom's
//! `vhaddq_u16` + `vrshrn_n_u16` pair does:
//!
//! ```text
//! A = wh*above + (256-wh)*below            B = ww*left + (256-ww)*right
//! out = (((A + B) >> 1) + 128) >> 8   ==   (A + B + 256) >> 9
//! ```
//!
//! The right-hand side is the scalar core's `divide_round(p, 9)`. The identity
//! holds for **every** reachable `A + B` (both sides are functions of the sum
//! alone, so the audit's sweep over `0..=130560` is complete, not a sample).
//! It is specifically the **truncating** halving add that makes it exact —
//! the rounding form (`vrhaddq_u16`) is off by one at `A + B ≡ 255 (mod 512)`,
//! which the audit demonstrates rather than asserts. magetypes has no halving
//! add, so it is written as the standard identity
//! `floor((A+B)/2) == (A & B) + ((A ^ B) >> 1)`, which cannot overflow because
//! its value IS the (in-range) result.
//!
//! # Scope — what runs here and what declines
//!
//! The gate is on the **data**, not on `bd`: it scans the block's `O(bw + bh)`
//! reference edge once per block against `O(bw * bh)` of predictor work. Out of
//! range, [`super::simd`]'s `i32x8` path runs unchanged, so bd10 and bd12 are
//! exactly as fast as they were.
//!
//! Every AV1 block width (4, 8, 16, 32, 64) is handled: the column-varying
//! inputs are staged into a `[_; 64]` array once per block, so a 16-lane load is
//! always in bounds and a `bw == 4` block simply leaves 12 lanes idle. At the
//! profile cell `bw >= 16` is **92.8 %** of SMOOTH's predicted pixels, so the
//! idle-lane cases are a rounding error either way.

// The predictors carry the reference edges as separate slices plus the block
// geometry — inherently many arguments, and the signatures mirror
// [`super::simd`]'s entry for entry on purpose so the two are diffable.
#![allow(clippy::too_many_arguments)]
// The per-row staging loops index two arrays at once by the row number; the
// indexed form is the one that reads like the scalar core it reproduces.
#![allow(clippy::needless_range_loop)]

use archmage::prelude::*;

use crate::intra::weights::SMOOTH_WEIGHT_LOG2_SCALE;

const SCALE: u16 = 1 << SMOOTH_WEIGHT_LOG2_SCALE; // 256

// The kernels bake `256`, `128` and the two shift counts as literals; lock the
// table's scale so a future weight-table edit cannot silently desync them.
const _: () = assert!(SMOOTH_WEIGHT_LOG2_SCALE == 8);

/// Largest sample for which every `u16` lane intermediate of the SMOOTH family
/// is exact. `(256 - w) * 255 = 65280` fits; `(256 - w) * 256 = 65536` does not.
/// Derived by exhaustive enumeration — `xtask/audit_nd16_lanes.py`.
pub(crate) const U16_SMOOTH_MAX: u16 = 255;

/// Widest AV1 transform block, and the column staging-array length. A multiple
/// of 16, so a 16-lane load at any `c < bw` is in bounds.
const MAX_W: usize = 64;
/// Tallest AV1 transform block, and the per-row staging-array length.
const MAX_H: usize = 64;

/// `true` if every sample in `s[..n]` is inside `m`. `O(n)` on the block's
/// reference edge — `O(bw + bh)` against `O(bw * bh)` of predictor work, so it
/// is a per-block scan and never a per-pixel one.
#[inline]
fn span_le(s: &[u16], n: usize, m: u16) -> bool {
    s.len() >= n && s[..n].iter().all(|&v| v <= m)
}

/// SMOOTH's gate: the above row and the left column both inside [`U16_SMOOTH_MAX`].
/// (`below`/`right` are members of those two spans, so they need no separate check.)
#[inline]
pub(crate) fn smooth_applies(bw: usize, bh: usize, above_row: &[u16], left: &[u16]) -> bool {
    span_le(above_row, bw, U16_SMOOTH_MAX) && span_le(left, bh, U16_SMOOTH_MAX)
}

/// SMOOTH_V's gate: the above row, plus the `below` scalar the caller extracted.
#[inline]
pub(crate) fn smooth_v_applies(bw: usize, above_row: &[u16], below: i32) -> bool {
    span_le(above_row, bw, U16_SMOOTH_MAX) && (0..=i32::from(U16_SMOOTH_MAX)).contains(&below)
}

/// SMOOTH_H's gate: the left column, plus the `right` scalar the caller extracted.
#[inline]
pub(crate) fn smooth_h_applies(bh: usize, left: &[u16], right: i32) -> bool {
    span_le(left, bh, U16_SMOOTH_MAX) && (0..=i32::from(U16_SMOOTH_MAX)).contains(&right)
}

// ===========================================================================
// SMOOTH
// ===========================================================================

/// Dispatch entry — see [`super::simd::smooth`] for the argument contract.
///
/// PRECONDITION: [`smooth_applies`] held for this block.
pub(crate) fn smooth(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    sw_w: &[u8],
    sw_h: &[u8],
) {
    incant!(
        smooth_impl(dst, stride, bw, bh, above_row, left, sw_w, sw_h),
        [v3, neon, wasm128, scalar]
    )
}

fn smooth_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    sw_w: &[u8],
    sw_h: &[u8],
) {
    super::simd::smooth_scalar(dst, stride, bw, bh, above_row, left, sw_w, sw_h);
}

#[magetypes(define(u16x16), v3, neon, wasm128, -scalar)]
fn smooth_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    sw_w: &[u8],
    sw_h: &[u8],
) {
    let below = left[bh - 1];
    let right = above_row[bw - 1];

    // Stage the column-varying inputs once per block, padded to MAX_W so a
    // 16-lane load at any c < bw is in bounds (bw is 4, 8, 16, 32 or 64), and
    // the row-varying scalar product once per row.
    let mut above_s = [0u16; MAX_W];
    let mut sww_s = [0u16; MAX_W];
    let mut below_term_s = [0u16; MAX_H];
    above_s[..bw].copy_from_slice(&above_row[..bw]);
    for (d, &w) in sww_s[..bw].iter_mut().zip(sw_w[..bw].iter()) {
        *d = u16::from(w);
    }
    for (d, &w) in below_term_s[..bh].iter_mut().zip(sw_h[..bh].iter()) {
        // (256 - wh) * below <= 256 * 255 = 65280 — inside u16 by the gate.
        *d = (SCALE - u16::from(w)) * below;
    }

    let scale_v = u16x16::splat(token, SCALE);
    let right_v = u16x16::splat(token, right);
    let round = u16x16::splat(token, 1 << (SMOOTH_WEIGHT_LOG2_SCALE - 1)); // 128
    let mut buf = [0u16; 16];

    // Column chunk OUTER, row inner: the above samples, the width weights and
    // the whole `(256 - ww) * right` term are row-invariant, so this hoists
    // them out of the row loop. libaom hoists the two loads the same way
    // (`intrapred_neon.c:2515-2585` keeps `top_v[]` / `weights_x_v[]` across
    // rows); hoisting the product as well is one step further.
    let mut c = 0;
    while c < bw {
        let n = (bw - c).min(16);
        let above_v = u16x16::from_slice(token, &above_s[c..c + 16]);
        let ww_v = u16x16::from_slice(token, &sww_s[c..c + 16]);
        let tr_term = (scale_v - ww_v) * right_v;
        for r in 0..bh {
            let wh_v = u16x16::splat(token, u16::from(sw_h[r]));
            let below_term = u16x16::splat(token, below_term_s[r]);
            let left_r_v = u16x16::splat(token, left[r]);
            // A = wh*above + (256-wh)*below   B = ww*left + (256-ww)*right
            let a = wh_v * above_v + below_term;
            let b = ww_v * left_r_v + tr_term;
            // floor((A+B)/2), which is libaom's truncating vhaddq_u16.
            let half = (a & b) + (a ^ b).shr_logical_const::<1>();
            let out = (half + round).shr_logical_const::<8>();
            out.store(&mut buf);
            let row = r * stride;
            dst[row + c..row + c + n].copy_from_slice(&buf[..n]);
        }
        c += 16;
    }
}

// ===========================================================================
// SMOOTH_V
// ===========================================================================

/// Dispatch entry — see [`super::simd::smooth_v`]. PRECONDITION:
/// [`smooth_v_applies`] held.
pub(crate) fn smooth_v(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    below: i32,
    sw_h: &[u8],
) {
    incant!(
        smooth_v_impl(dst, stride, bw, bh, above_row, below, sw_h),
        [v3, neon, wasm128, scalar]
    )
}

fn smooth_v_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    below: i32,
    sw_h: &[u8],
) {
    super::simd::smooth_v_scalar(dst, stride, bw, bh, above_row, below, sw_h);
}

#[magetypes(define(u16x16), v3, neon, wasm128, -scalar)]
fn smooth_v_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    below: i32,
    sw_h: &[u8],
) {
    let below = below as u16;
    let mut above_s = [0u16; MAX_W];
    let mut below_term_s = [0u16; MAX_H];
    above_s[..bw].copy_from_slice(&above_row[..bw]);
    for (d, &w) in below_term_s[..bh].iter_mut().zip(sw_h[..bh].iter()) {
        *d = (SCALE - u16::from(w)) * below;
    }
    let round = u16x16::splat(token, 1 << (SMOOTH_WEIGHT_LOG2_SCALE - 1)); // 128
    let mut buf = [0u16; 16];

    let mut c = 0;
    while c < bw {
        let n = (bw - c).min(16);
        let above_v = u16x16::from_slice(token, &above_s[c..c + 16]);
        for r in 0..bh {
            let w_v = u16x16::splat(token, u16::from(sw_h[r]));
            let below_term = u16x16::splat(token, below_term_s[r]);
            let p = w_v * above_v + below_term;
            let out = (p + round).shr_logical_const::<8>();
            out.store(&mut buf);
            let row = r * stride;
            dst[row + c..row + c + n].copy_from_slice(&buf[..n]);
        }
        c += 16;
    }
}

// ===========================================================================
// SMOOTH_H
// ===========================================================================

/// Dispatch entry — see [`super::simd::smooth_h`]. PRECONDITION:
/// [`smooth_h_applies`] held.
pub(crate) fn smooth_h(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &[u16],
    right: i32,
    sw_w: &[u8],
) {
    incant!(
        smooth_h_impl(dst, stride, bw, bh, left, right, sw_w),
        [v3, neon, wasm128, scalar]
    )
}

fn smooth_h_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &[u16],
    right: i32,
    sw_w: &[u8],
) {
    super::simd::smooth_h_scalar(dst, stride, bw, bh, left, right, sw_w);
}

#[magetypes(define(u16x16), v3, neon, wasm128, -scalar)]
fn smooth_h_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &[u16],
    right: i32,
    sw_w: &[u8],
) {
    let mut sww_s = [0u16; MAX_W];
    for (d, &w) in sww_s[..bw].iter_mut().zip(sw_w[..bw].iter()) {
        *d = u16::from(w);
    }
    let scale_v = u16x16::splat(token, SCALE);
    let right_v = u16x16::splat(token, right as u16);
    let round = u16x16::splat(token, 1 << (SMOOTH_WEIGHT_LOG2_SCALE - 1)); // 128
    let mut buf = [0u16; 16];

    let mut c = 0;
    while c < bw {
        let n = (bw - c).min(16);
        let w_v = u16x16::from_slice(token, &sww_s[c..c + 16]);
        let tr_term = (scale_v - w_v) * right_v; // row-invariant
        for r in 0..bh {
            let left_r_v = u16x16::splat(token, left[r]);
            let p = w_v * left_r_v + tr_term;
            let out = (p + round).shr_logical_const::<8>();
            out.store(&mut buf);
            let row = r * stride;
            dst[row + c..row + c + n].copy_from_slice(&buf[..n]);
        }
        c += 16;
    }
}

/// Does the gate FIRE on the domain the encoder actually presents? A bound that
/// is sound but never admits anything is as useless as no path at all, and only
/// a counted pin says which of the two shipped (`docs/DIFFERENTIAL_PLAYBOOK.md`
/// §2, and the counterpart to `the_lane_bounds_are_load_bearing` below).
#[cfg(test)]
mod reach {
    use super::*;

    /// `TX_SIZES_ALL` widths and heights, in the port's own order.
    const SHAPES: [(usize, usize); 19] = [
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

    /// Over the WORST-CASE edge at each bit depth (every sample `2^bd - 1`),
    /// how many of the 19 transform shapes does each gate admit? These counts
    /// are pinned, both directions, so that a future change to a bound shows up
    /// here as a number rather than as silence.
    #[test]
    fn the_gates_fire_across_the_real_grid() {
        for (bd, want_smooth) in [(8i32, 19usize), (10, 0), (12, 0)] {
            let maxv = ((1u32 << bd) - 1) as u16;
            let (mut smooth_ok, mut smooth_v_ok, mut smooth_h_ok) = (0, 0, 0);
            for &(bw, bh) in &SHAPES {
                let above = vec![maxv; bw];
                let left = vec![maxv; bh];
                let below = i32::from(left[bh - 1]);
                let right = i32::from(above[bw - 1]);
                smooth_ok += usize::from(smooth_applies(bw, bh, &above, &left));
                smooth_v_ok += usize::from(smooth_v_applies(bw, &above, below));
                smooth_h_ok += usize::from(smooth_h_applies(bh, &left, right));
            }
            assert_eq!(smooth_ok, want_smooth, "SMOOTH at bd{bd}");
            assert_eq!(smooth_v_ok, want_smooth, "SMOOTH_V at bd{bd}");
            assert_eq!(smooth_h_ok, want_smooth, "SMOOTH_H at bd{bd}");
        }
    }

    /// The two bounds are on the DATA, not on `bd` — so a bd10 or bd12 block
    /// whose samples happen to be small takes the narrow SMOOTH path too. Pin
    /// that, because it is the difference between "a bd8 kernel" (what the
    /// brief called for) and "a kernel gated on bd8 samples" (what shipped).
    #[test]
    fn the_bound_is_on_the_data_not_the_bit_depth() {
        let (bw, bh) = (16usize, 16usize);
        let dark = vec![200u16; 16]; // a legal bd12 block that never exceeds 255
        assert!(smooth_applies(bw, bh, &dark, &dark));
        let bright = vec![4095u16; 16]; // a legal bd12 block that does
        assert!(!smooth_applies(bw, bh, &bright, &bright));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intra::weights::SMOOTH_WEIGHTS;

    fn xorshift(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    }

    /// Eight edge shapes. Deliberately **asymmetric**: a flat edge makes
    /// `above[c] == below` and `left[r] == right`, under which SMOOTH's two
    /// halves are each independent of the weights — i.e. invariant under
    /// exactly the re-association being tested. A flat probe would pass against
    /// almost any wrong kernel (playbook §1 / KB-12), so it is kept only as a
    /// control and the other seven carry the test.
    fn fill(shape: usize, s: &mut u32, buf: &mut [u16], m: u16) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = match shape {
                0 => (xorshift(s) % (u32::from(m) + 1)) as u16, // dense random
                1 => {
                    if i % 2 == 0 {
                        m
                    } else {
                        0
                    }
                } // max sawtooth
                2 => (i as u16) % (m + 1),                    // ramp
                3 => m,                                       // flat (control)
                4 => (xorshift(s) % 256) as u16,              // bd8 range
                5 => m - (i as u16 % (m + 1)),                // reverse ramp
                6 => {
                    if i < 8 {
                        0
                    } else {
                        m
                    }
                } // step
                _ => (xorshift(s) % 2) as u16 * m,            // binary
            };
        }
    }

    /// Every kernel against the `simd.rs` scalar core it must reproduce, over
    /// every AV1 block shape, every dispatch tier, tight AND padded stride.
    #[test]
    fn every_kernel_matches_the_scalar_core() {
        let shapes: [(usize, usize); 19] = [
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
        let mut s = 0x2468_ace0u32;
        let mut cells = 0usize;
        for shape in 0..8 {
            for &(bw, bh) in &shapes {
                for &pad in &[0usize, 7] {
                    let stride = bw + pad;
                    let mut above = vec![0u16; bw];
                    let mut left = vec![0u16; bh];
                    fill(shape, &mut s, &mut above, U16_SMOOTH_MAX);
                    fill(shape, &mut s, &mut left, U16_SMOOTH_MAX);
                    let sw_w = &SMOOTH_WEIGHTS[bw - 4..];
                    let sw_h = &SMOOTH_WEIGHTS[bh - 4..];
                    let below = i32::from(left[bh - 1]);
                    let right = i32::from(above[bw - 1]);

                    let mut got = vec![0u16; stride * bh];
                    let mut want = vec![0u16; stride * bh];

                    assert!(smooth_applies(bw, bh, &above, &left));
                    smooth(&mut got, stride, bw, bh, &above, &left, sw_w, sw_h);
                    super::super::simd::smooth_scalar(
                        &mut want, stride, bw, bh, &above, &left, sw_w, sw_h,
                    );
                    assert_eq!(got, want, "SMOOTH {bw}x{bh} pad={pad} shape={shape}");

                    got.fill(0);
                    want.fill(0);
                    assert!(smooth_v_applies(bw, &above, below));
                    smooth_v(&mut got, stride, bw, bh, &above, below, sw_h);
                    super::super::simd::smooth_v_scalar(
                        &mut want, stride, bw, bh, &above, below, sw_h,
                    );
                    assert_eq!(got, want, "SMOOTH_V {bw}x{bh} pad={pad} shape={shape}");

                    got.fill(0);
                    want.fill(0);
                    assert!(smooth_h_applies(bh, &left, right));
                    smooth_h(&mut got, stride, bw, bh, &left, right, sw_w);
                    super::super::simd::smooth_h_scalar(
                        &mut want, stride, bw, bh, &left, right, sw_w,
                    );
                    assert_eq!(got, want, "SMOOTH_H {bw}x{bh} pad={pad} shape={shape}");
                    cells += 1;
                }
            }
        }
        // Non-vacuity: the vector body must actually have been reachable.
        assert!(cells > 250, "grid too small ({cells})");
    }

    /// Playbook §2 — the bounds must BITE. One sample over `M*` and the narrow
    /// lanes must genuinely diverge from the scalar reference, else the gates
    /// guard nothing and could be deleted without a test noticing.
    ///
    /// The divergence half is necessarily conditional on a VECTOR tier
    /// dispatching: under `AOM_FORCE_SCALAR=1` these entries route to the very
    /// scalar cores they are compared against, so they cannot diverge from
    /// themselves — the same defect `dir_simd::the_tap_bound_is_load_bearing`
    /// hit on its first scalar-pinned run. The gates' own accept/reject
    /// decisions are asserted UNconditionally: that half is arithmetic on the
    /// span and has no tier.
    #[test]
    fn the_lane_bounds_are_load_bearing() {
        let (bw, bh) = (16usize, 16usize);
        let sw_w = &SMOOTH_WEIGHTS[bw - 4..];
        let sw_h = &SMOOTH_WEIGHTS[bh - 4..];

        // --- the gates accept at M* and reject one over it -------------------
        let at = vec![U16_SMOOTH_MAX; bw.max(bh)];
        let mut over = at.clone();
        over[3] = U16_SMOOTH_MAX + 1;
        assert!(smooth_applies(bw, bh, &at, &at), "255 must be accepted");
        assert!(!smooth_applies(bw, bh, &over, &at), "256 must be rejected (above)");
        assert!(!smooth_applies(bw, bh, &at, &over), "256 must be rejected (left)");
        assert!(!smooth_v_applies(bw, &over, 0), "256 must be rejected");
        assert!(!smooth_v_applies(bw, &at, 256), "below=256 must be rejected");
        assert!(!smooth_h_applies(bh, &over, 0), "256 must be rejected");
        assert!(!smooth_h_applies(bh, &at, 256), "right=256 must be rejected");

        // At exactly the bound every kernel still agrees, at every tier.
        let mut got = vec![0u16; bw * bh];
        let mut want = vec![0u16; bw * bh];
        smooth(&mut got, bw, bw, bh, &at, &at, sw_w, sw_h);
        super::super::simd::smooth_scalar(&mut want, bw, bw, bh, &at, &at, sw_w, sw_h);
        assert_eq!(got, want, "SMOOTH at the bound");

        if crate::dispatch::scalar_forced() {
            return; // no vector tier to diverge from; the halves above still ran
        }

        // --- and one over the bound, the narrow lanes are WRONG --------------
        // A ramp, not a flat edge: with a flat `over` the wrap happens in every
        // lane identically and some kernels would still agree by accident.
        //
        // The over-bound sample goes in the SPAN (index 3), never in the
        // `below`/`right` corner scalars, which stay at 215. That is deliberate:
        // the corner scalars are folded into a per-row `(256 - w) * below`
        // BEFORE the lanes, so an out-of-range corner overflows in the kernel's
        // own host arithmetic and a debug build traps there instead of
        // diverging. Both are correct rejections, but only the span form
        // demonstrates what this test is about — that the LANES wrap.
        let mut ramp: Vec<u16> = (0..bw.max(bh)).map(|i| 200 + (i as u16 % 16)).collect();
        ramp[3] = 300; // over U16_SMOOTH_MAX
        let corner = i32::from(ramp[bw.max(bh) - 1]);
        assert!(corner <= i32::from(U16_SMOOTH_MAX));
        let mut diverged = 0;
        for (name, hit) in [
            ("SMOOTH", {
                let mut g = vec![0u16; bw * bh];
                let mut w = vec![0u16; bw * bh];
                smooth(&mut g, bw, bw, bh, &ramp, &ramp, sw_w, sw_h);
                super::super::simd::smooth_scalar(&mut w, bw, bw, bh, &ramp, &ramp, sw_w, sw_h);
                g != w
            }),
            ("SMOOTH_V", {
                let mut g = vec![0u16; bw * bh];
                let mut w = vec![0u16; bw * bh];
                smooth_v(&mut g, bw, bw, bh, &ramp, corner, sw_h);
                super::super::simd::smooth_v_scalar(&mut w, bw, bw, bh, &ramp, corner, sw_h);
                g != w
            }),
            ("SMOOTH_H", {
                let mut g = vec![0u16; bw * bh];
                let mut w = vec![0u16; bw * bh];
                smooth_h(&mut g, bw, bw, bh, &ramp, corner, sw_w);
                super::super::simd::smooth_h_scalar(&mut w, bw, bw, bh, &ramp, corner, sw_w);
                g != w
            }),
        ] {
            assert!(hit, "{name}: the u16 bound never bites — the gate is decorative");
            diverged += 1;
        }
        assert_eq!(diverged, 3);

    }
}
