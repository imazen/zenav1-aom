//! **i16-lane** vector kernel for the two-tap directional intra interpolation —
//! the inner arithmetic of `av1_dr_prediction_z{1,2,3}` — plus the runtime bound
//! that admits it.
//!
//! # Why this exists
//!
//! [`super::dir`]'s `z1_high` / `z2_high` / `z3_high` were **pure scalar at every
//! bit depth**, while libaom dispatches `av1_dr_prediction_z{1,2,3}_neon` (and
//! `_avx2` / `_sse4_1`) on the lowbd path. That is the same structural gap the
//! forward transform had (`transform::simd::lowbd16_fwd`), one stage over: the
//! port ran the wide/scalar path where libaom runs a narrow-lane vector one.
//! Measured at the profile cell, the three kernels were **3.50 ms of a 150 ms
//! encode against libaom's ~0.37 ms** — see
//! `benchmarks/encoder_intra_dir_i16_2026-08-03.md`.
//!
//! The body is ONE `#[magetypes(define(i16x16), v3, neon, -scalar)]` function,
//! so it serves NEON and AVX2 from the same source (the cross-platform half of
//! the bd8 lane-width programme — `benchmarks/winperf_windows_2026-08-02.md`
//! §CROSS-PLATFORM SCOPING).
//!
//! # The identity, and why i16 lanes are exact
//!
//! The scalar kernel computes, per output,
//!
//! ```text
//! (a0 * (32 - shift) + a1 * shift + 16) >> 5      with a0 = edge[b], a1 = edge[b+1]
//! ```
//!
//! The vector form uses libaom's re-association (`intrapred_neon.c:1307-1308`)
//!
//! ```text
//! a0 * (32 - shift) + a1 * shift  ==  (a0 << 5) + (a1 - a0) * shift
//! ```
//!
//! which is an identity over the integers, so the two agree **exactly** provided
//! no i16 lane wraps. With `shift ∈ [0, 31]` (it is `((x << up) & 0x3F) >> 1`)
//! and every tap `0 <= v <= M` the three intermediates are bounded by
//!
//! * `a0 << 5` — `<= 32 * M`,
//! * `a1 - a0` — `|.| <= M`, and `(a1 - a0) * shift` — `|.| <= 31 * M`,
//! * the sum, which equals `a0*(32-shift) + a1*shift` — a convex combination
//!   scaled by 32, so `∈ [0, 32 * M]`,
//! * `+ 16` — `<= 32 * M + 16`.
//!
//! Every one of those is inside `i16` iff `32 * M + 16 <= 32767`, i.e.
//! **`M <= 1023`**. That is [`I16_TAP_MAX`], and it is the whole audit: it is
//! **tight** (at `M = 1024`, `a0 << 5` is exactly `-32768` and the result is
//! wrong — pinned by `gate_bite::the_tap_bound_is_load_bearing`), and it is
//! taken at RUNTIME on the actual edge span, so the path is sound for any
//! caller of the public predictors and not only for bd8. In bit-depth terms it
//! admits **bd8 and bd10** (samples `<= 1023`) and declines bd12, which is the
//! honest statement of its reach — the gate is on the data, not on `bd`.
//!
//! The final `>> 5` is an arithmetic shift of a value in `[0, 32752]`, which is
//! the scalar `>>` on the same non-negative value, so the narrowing `as u16` is
//! exact.
//!
//! # Scope — what runs vector and what does not
//!
//! Only **contiguous** tap runs (`base_inc == 1`, i.e. `upsample == 0`) take the
//! vector path, because then the two operand vectors are plain unaligned loads
//! of `edge[b..b+16]` and `edge[b+1..b+17]` with no staging array at all. The
//! `upsample == 1` runs are a stride-2 gather and stay scalar; they are
//! **12.6 % of z1 and 14.9 % of z3 pixels** at the profile cell (upsampling is
//! only ever enabled for `bw + bh <= 16`, `edge::use_upsample`), and the census
//! is in the writeup. `z2`'s left-hand half is a genuine gather (`base_y` is not
//! affine in `c`) and likewise stays scalar — it is 50.2 % of z2's pixels.
//!
//! Runs shorter than [`MIN_VEC_RUN`] stay scalar: a 4-wide block cannot fill
//! enough of a 16-lane vector to pay for the round trip through the stack array.

use archmage::prelude::*;

/// The largest edge sample for which every i16 lane intermediate is exact.
/// `32 * 1023 + 16 = 32752 <= i16::MAX`; `32 * 1024 = 32768` is not.
pub(crate) const I16_TAP_MAX: u16 = 1023;

/// Shortest run given to the vector kernel. Below this the array round trip
/// costs more than the 16 scalar multiply-adds it replaces.
pub(crate) const MIN_VEC_RUN: usize = 8;

/// `true` if every sample in `edge[lo..=hi]` is inside the i16 lane bound.
/// `O(hi - lo)` — the caller's spans are `O(bw + bh)` against `O(bw * bh)` of
/// predictor work, so this is a per-block scan, never a per-pixel one.
#[inline]
pub(crate) fn span_fits_i16(edge: &[u16], lo: usize, hi: usize) -> bool {
    hi < edge.len() && lo <= hi && edge[lo..=hi].iter().all(|&v| v <= I16_TAP_MAX)
}

/// The scalar two-tap run — the differential reference AND the tail/decline
/// path. Byte-identical to the expression in [`super::dir`] by construction.
#[inline]
pub(crate) fn two_tap_run_scalar(out: &mut [u16], edge: &[u16], start: usize, shift: i32, n: usize) {
    for (i, o) in out.iter_mut().take(n).enumerate() {
        let a0 = edge[start + i] as i32;
        let a1 = edge[start + i + 1] as i32;
        *o = ((a0 * (32 - shift) + a1 * shift + 16) >> 5) as u16;
    }
}

/// Dispatch entry: write `n` outputs of the contiguous two-tap run starting at
/// `edge[start]` into `out[..n]`.
///
/// PRECONDITIONS (the caller's, and all three are what the scalar kernel already
/// requires plus the bound): `start + n < edge.len()`, `shift ∈ [0, 31]`, and
/// every sample in `edge[start ..= start + n]` `<= I16_TAP_MAX`. Callers take
/// the last one with [`span_fits_i16`] once per block.
pub(crate) fn two_tap_run(out: &mut [u16], edge: &[u16], start: usize, shift: i32, n: usize) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        two_tap_run_impl(out, edge, start, shift, n),
        [v3, neon, scalar]
    )
}

fn two_tap_run_impl_scalar(
    _t: archmage::ScalarToken,
    out: &mut [u16],
    edge: &[u16],
    start: usize,
    shift: i32,
    n: usize,
) {
    two_tap_run_scalar(out, edge, start, shift, n);
}

#[magetypes(define(i16x16, u16x16), v3, neon, -scalar)]
fn two_tap_run_impl(
    token: Token,
    out: &mut [u16],
    edge: &[u16],
    start: usize,
    shift: i32,
    n: usize,
) {
    if n < MIN_VEC_RUN {
        two_tap_run_scalar(out, edge, start, shift, n);
        return;
    }
    let sv = i16x16::splat(token, shift as i16);
    let round = i16x16::splat(token, 16);
    // KB-PERF-34: a `u16` staging buffer plus one `copy_from_slice`, NOT an
    // `i16` buffer plus a per-lane cast loop. The per-lane copy was measured to
    // be this kernel's real cost — see
    // `benchmarks/encoder_dir_pred_reach_audit_2026-09-10.md`, where doubling
    // the ARITHMETIC to avoid a gather made it 0.34 % slower because it doubled
    // the copy. The bitcast is exact rather than convenient: every output is
    // `((a0 * (32 - shift) + a1 * shift + 16) >> 5)` with taps `<= I16_TAP_MAX`,
    // hence non-negative and `<= 1023`, so its `i16` bit pattern IS its `u16`
    // value — the same equality `buf[k] as u16` relied on.
    let mut buf = [0u16; 16];
    let mut i = 0;
    // A chunk needs 17 in-range samples. For a FULL chunk that is implied by the
    // caller's `start + n < edge.len()`; the guard binds only on a partial tail
    // (n == 8 is the common one — an 8-wide block).
    while i < n && start + i + 17 <= edge.len() {
        let m = (n - i).min(16);
        let idx = start + i;
        let v0 = u16x16::from_slice(token, &edge[idx..idx + 16]).bitcast_i16x16();
        let v1 = u16x16::from_slice(token, &edge[idx + 1..idx + 17]).bitcast_i16x16();
        let res = (v0.shl_const::<5>() + (v1 - v0) * sv + round).shr_arithmetic_const::<5>();
        res.bitcast_u16x16().store(&mut buf);
        out[i..i + m].copy_from_slice(&buf[..m]);
        i += m;
    }
    if i < n {
        two_tap_run_scalar(&mut out[i..], edge, start + i, shift, n - i);
    }
}

/// Dispatch entry for the z2 LEFT-half gather, ALL ROWS in one dispatch:
/// for each row `r`, writes the `c_end(r)` columns whose `base_x` falls short
/// of the above edge (`c_end = ((y*dx - 1) >> 6).clamp(0, bw)` — see
/// `super::dir::z2_high`) into `dst[r*stride .. r*stride + c_end]`, reading the
/// left edge through `ld`/`pad` (`EdgeRef16`'s data + pad).
///
/// `base_y(c) = ((r << 6) - (c + 1) * dy) >> frac_y` is NOT affine in `c`, so
/// the taps are a genuine gather: the vector body computes the index and blend
/// arithmetic in i32 lanes, then performs the two tap loads per lane IN ORDER
/// as plain scalar reads. That keeps the panic surface byte-for-byte — the
/// scalar twin reads `&ld[idx..idx + 2]` at the same index in the same column
/// order — at the cost of a per-lane array round trip the contiguous kernel
/// never pays. The loads are irreducible (the values genuinely live at
/// non-affine offsets); what the kernel removes is the per-pixel index math
/// (~6 ops), the two-tap blend (~5 ops), the checked dst store, and the
/// per-row dispatch/setup — this is one `incant!` per block, not per row.
///
/// i32 lanes, not the i16 trick [`two_tap_run_impl`] uses: the i16 bound needs
/// `M <= I16_TAP_MAX`, and this kernel is only reachable on rows where the
/// above side's `span_fits_i16` may have declined — the gather must not inherit
/// a data bound it never checked. `res = a0*32 + (a1-a0)*s + 16` maxes at
/// `32 * 65535 + 16` in i32 — exact for every sample value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn z2_left_gather(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    ld: &[u16],
    pad: usize,
    dx: i32,
    dy: i32,
    frac_y: u32,
    up_left: u32,
) {
    let _ = crate::dispatch::scalar_forced();
    incant!(
        z2_left_gather_impl(dst, stride, bw, bh, ld, pad, dx, dy, frac_y, up_left),
        [v3, neon, wasm128, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
fn z2_left_gather_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    ld: &[u16],
    pad: usize,
    dx: i32,
    dy: i32,
    frac_y: u32,
    up_left: u32,
) {
    z2_left_gather_scalar(dst, stride, bw, bh, ld, pad, dx, dy, frac_y, up_left);
}

/// The scalar gather recipe — the differential reference and the tail path.
/// Byte-identical to the left-prefix loop in `super::dir::z2_high` (same index
/// expressions, same column order, same `&ld[i0..i0 + 2]` panic point).
#[allow(clippy::too_many_arguments)]
pub(crate) fn z2_left_gather_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    ld: &[u16],
    pad: usize,
    dx: i32,
    dy: i32,
    frac_y: u32,
    up_left: u32,
) {
    for r in 0..bh {
        let y = (r + 1) as i32;
        let c_end = ((y * dx - 1) >> 6).clamp(0, bw as i32) as usize;
        let drow = &mut dst[r * stride..r * stride + c_end];
        for (k, slot) in drow.iter_mut().enumerate() {
            let x2 = (k + 1) as i32;
            let y2 = ((r as i32) << 6) - x2 * dy;
            let base_y = y2 >> frac_y;
            let shift = ((y2 * (1 << up_left)) & 0x3F) >> 1;
            let i0 = (pad as i32 + base_y) as usize;
            let w = &ld[i0..i0 + 2];
            *slot =
                ((i32::from(w[0]) * (32 - shift) + i32::from(w[1]) * shift + 16) >> 5) as u16;
        }
    }
}

/// One row's scalar tail, starting at block column `c0` (so `x2 = c0 + k + 1`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn z2_left_gather_tail(
    drow: &mut [u16],
    ld: &[u16],
    pad: usize,
    r: usize,
    c0: usize,
    dy: i32,
    frac_y: u32,
    up_left: u32,
) {
    for (k, slot) in drow.iter_mut().enumerate() {
        let x2 = (c0 + k + 1) as i32;
        let y2 = ((r as i32) << 6) - x2 * dy;
        let base_y = y2 >> frac_y;
        let shift = ((y2 * (1 << up_left)) & 0x3F) >> 1;
        let i0 = (pad as i32 + base_y) as usize;
        let w = &ld[i0..i0 + 2];
        *slot = ((i32::from(w[0]) * (32 - shift) + i32::from(w[1]) * shift + 16) >> 5) as u16;
    }
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn z2_left_gather_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    ld: &[u16],
    pad: usize,
    dx: i32,
    dy: i32,
    frac_y: u32,
    up_left: u32,
) {
    let dyv = i32x8::splat(token, dy);
    let c63 = i32x8::splat(token, 0x3F);
    let c16 = i32x8::splat(token, 16);
    let c32 = i32x8::splat(token, 32);
    let mask_lo = i32x8::splat(token, 0xFFFF);
    let step = i32x8::splat(token, 8);
    let mut pairs = [0i32; 8];
    for r in 0..bh {
        let y = (r + 1) as i32;
        let c_end = ((y * dx - 1) >> 6).clamp(0, bw as i32) as usize;
        let n8 = c_end & !7;
        if n8 == 0 {
            if c_end > 0 {
                z2_left_gather_tail(
                    &mut dst[r * stride..r * stride + c_end],
                    ld,
                    pad,
                    r,
                    0,
                    dy,
                    frac_y,
                    up_left,
                );
            }
            continue;
        }
        let drow = &mut dst[r * stride..r * stride + c_end];
        let r6 = i32x8::splat(token, (r as i32) << 6);
        // Lane k holds x2 = c + 1 + k; each chunk advances the base by 8.
        let mut x2v = i32x8::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
        let mut c = 0usize;
        while c < n8 {
            let y2v = r6 - x2v * dyv;
            let byv = y2v.shr_arithmetic_uniform(frac_y);
            let shv = (y2v.shl_uniform(up_left) & c63).shr_logical_const::<1>();
            let by = byv.to_array();
            for (k, p) in pairs.iter_mut().enumerate() {
                // Same index expression as the scalar twin: `(pad + base_y)`
                // is computed in i32 THEN cast, so a negative base wraps to a
                // huge index and panics at the lane the scalar would.
                let i0 = (pad as i32 + by[k]) as usize;
                *p = i32::from(ld[i0]) | (i32::from(ld[i0 + 1]) << 16);
            }
            let pv = i32x8::from_array(token, pairs);
            let a0 = pv & mask_lo;
            let a1 = pv.shr_logical_const::<16>();
            let res = (a0 * c32 + (a1 - a0) * shv + c16).shr_arithmetic_const::<5>();
            let out_arr: [u16; 8] = res.to_array().map(|v| v as u16);
            drow[c..c + 8].copy_from_slice(&out_arr);
            x2v += step;
            c += 8;
        }
        if c < c_end {
            z2_left_gather_tail(&mut drow[c..], ld, pad, r, c, dy, frac_y, up_left);
        }
    }
}

/// Dispatch entry for the z3 vec path — ALL columns in ONE `incant!`.
///
/// # Shape: transposed z1, the same trick `av1_highbd_dr_prediction_z3_avx2`
/// uses
///
/// C computes z3 by running its z1 kernel down `left` into a column-major
/// scratch (`highbd_dr_prediction_z1_NxW_internal_avx2` writing `dstT`) and
/// then `highbd_transpose16x16` for the row-major stores. The port's old path
/// paid a per-column `two_tap_run` dispatch, a `col[]` stack bounce and a
/// strided `dst[r * stride + c]` scalar store per pixel — this kernel keeps
/// C's shape instead: contiguous column writes, contiguous row stores.
///
/// # Bit-exactness
///
/// The column compute is [`two_tap_run_impl`]'s i16-lane recipe verbatim —
/// `(v0 << 5) + (v1 - v0) * s + 16, >> 5` — exact because the caller's
/// `span_fits_i16` gate bounds every tap by `I16_TAP_MAX` (the convex-combo
/// bound it documents). Partial bands and the row/column tails use the scalar
/// recipe; the fill is the same `edge[pad + max_base_y]` the scalar loop
/// writes. The transpose only MOVES lanes.
///
/// # `up == 1` (upsampled left edge)
///
/// Taps step by `base_inc = 2` down the edge, so a full band's `a0`/`a1` are
/// the even/odd u16 lanes of ONE contiguous 16-element read — two `pshufb`s
/// plus an `unpacklo_epi64` each, the same trick C's `HighbdEvenOddMaskx4`
/// masks use in `highbd_dr_prediction_z1_*_internal_avx2`. The per-column
/// `shift` is `((y << up) & 0x3F) >> 1` as in the scalar.
///
/// # Bounds
///
/// A vector band reads `edge[pad + base + r*inc ..= +7*inc + 1]`; a full band
/// (`r0 + 8 <= n_act`) tops out at `pad + max_base_y`, which the gate's
/// `hi < edge.len()` already guarantees — so no per-band guard is needed and
/// the kernel adds no panic the scalar path lacks. `bw % 8 == 4` blocks (the
/// 4-wide tx sizes) take a scalar column tail.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i16x8), v3, -scalar)]
fn z3_cols_impl(
    _token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    edge: &[u16],
    pad: usize,
    dy: i32,
    up: i32,
) {
    use archmage::intrinsics::x86_64::*;

    // Standard three-stage in-register 8x8 transpose of i16 lanes
    // (`dist::hadamard` precedent): input `r[k]` is column `c0 + k`'s rows
    // `r0..r0 + 8`; output `r` is row `r0 + r` across the eight columns.
    let transpose = |r: [__m128i; 8]| -> [__m128i; 8] {
        let a0 = _mm_unpacklo_epi16(r[0], r[1]);
        let a1 = _mm_unpackhi_epi16(r[0], r[1]);
        let a2 = _mm_unpacklo_epi16(r[2], r[3]);
        let a3 = _mm_unpackhi_epi16(r[2], r[3]);
        let a4 = _mm_unpacklo_epi16(r[4], r[5]);
        let a5 = _mm_unpackhi_epi16(r[4], r[5]);
        let a6 = _mm_unpacklo_epi16(r[6], r[7]);
        let a7 = _mm_unpackhi_epi16(r[6], r[7]);
        let b0 = _mm_unpacklo_epi32(a0, a2);
        let b1 = _mm_unpackhi_epi32(a0, a2);
        let b2 = _mm_unpacklo_epi32(a1, a3);
        let b3 = _mm_unpackhi_epi32(a1, a3);
        let b4 = _mm_unpacklo_epi32(a4, a6);
        let b5 = _mm_unpackhi_epi32(a4, a6);
        let b6 = _mm_unpacklo_epi32(a5, a7);
        let b7 = _mm_unpackhi_epi32(a5, a7);
        [
            _mm_unpacklo_epi64(b0, b4),
            _mm_unpackhi_epi64(b0, b4),
            _mm_unpacklo_epi64(b1, b5),
            _mm_unpackhi_epi64(b1, b5),
            _mm_unpacklo_epi64(b2, b6),
            _mm_unpackhi_epi64(b2, b6),
            _mm_unpacklo_epi64(b3, b7),
            _mm_unpackhi_epi64(b3, b7),
        ]
    };

    let frac_bits = 6 - up;
    let base_inc = 1usize << up;
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    let fillv = edge[pad + max_base_y as usize];
    let fill = _mm_set1_epi16(fillv as i16);
    let sixteen = _mm_set1_epi16(16);
    // Stride-2 gather masks (up == 1): the low u64 of each pshufb result
    // picks the even / odd u16 lanes of one 8-lane load; the high u64 is
    // zeroed and discarded by the unpacklo_epi64 pair below.
    let ev = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let od = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    // Per 8-column group: each column's band lives in a register — no scratch
    // round trip. `base`/`shift`/`n_act` are scalar per column (they differ per
    // column and feed splats, so vectorizing their 8-lane math would not pay).
    let mut c0 = 0usize;
    while c0 + 8 <= bw {
        let mut bases = [0usize; 8];
        let mut shifts = [0i16; 8];
        let mut n_acts = [0usize; 8];
        for k in 0..8 {
            let y = dy * (c0 + k + 1) as i32;
            let base = y >> frac_bits;
            bases[k] = base as usize;
            shifts[k] = (((y << up) & 0x3F) >> 1) as i16;
            n_acts[k] = if base >= max_base_y {
                0
            } else {
                // Rows are active while base + r*base_inc < max_base_y.
                bh.min(((max_base_y - base) as usize + base_inc - 1) / base_inc)
            };
        }
        let mut r0 = 0usize;
        while r0 + 8 <= bh {
            let cols: [__m128i; 8] = core::array::from_fn(|k| {
                let n_act = n_acts[k];
                if r0 + 8 <= n_act {
                    // Full band — all eight lanes are real taps, so the loads
                    // stay <= pad + max_base_y (see Bounds above).
                    let b0 = pad + bases[k] + r0 * base_inc;
                    let (v0, v1) = if up == 0 {
                        let a: &[u16; 8] = edge[b0..b0 + 8].try_into().unwrap();
                        let b: &[u16; 8] = edge[b0 + 1..b0 + 9].try_into().unwrap();
                        (_mm_loadu_si128(a), _mm_loadu_si128(b))
                    } else {
                        // Taps step by 2 down the edge: one contiguous
                        // 16-element read + even/odd lane extraction, the
                        // same trick as C's HighbdEvenOddMaskx4 pshufb.
                        let lo: &[u16; 8] = edge[b0..b0 + 8].try_into().unwrap();
                        let hi: &[u16; 8] = edge[b0 + 8..b0 + 16].try_into().unwrap();
                        let lo = _mm_loadu_si128(lo);
                        let hi = _mm_loadu_si128(hi);
                        (
                            _mm_unpacklo_epi64(
                                _mm_shuffle_epi8(lo, ev),
                                _mm_shuffle_epi8(hi, ev),
                            ),
                            _mm_unpacklo_epi64(
                                _mm_shuffle_epi8(lo, od),
                                _mm_shuffle_epi8(hi, od),
                            ),
                        )
                    };
                    let sv = _mm_set1_epi16(shifts[k]);
                    _mm_srai_epi16::<5>(_mm_add_epi16(
                        _mm_add_epi16(
                            _mm_slli_epi16::<5>(v0),
                            _mm_mullo_epi16(_mm_sub_epi16(v1, v0), sv),
                        ),
                        sixteen,
                    ))
                } else if r0 >= n_act {
                    fill
                } else {
                    // Partial band: taps for rows < n_act, fill above — the
                    // vector load would read past `max_base_y + 1` here.
                    let mut buf = [fillv; 8];
                    let sv = shifts[k] as i32;
                    for (i, slot) in buf[..n_act - r0].iter_mut().enumerate() {
                        let t = pad + bases[k] + (r0 + i) * base_inc;
                        let a0 = edge[t] as i32;
                        let a1 = edge[t + 1] as i32;
                        *slot = ((a0 * (32 - sv) + a1 * sv + 16) >> 5) as u16;
                    }
                    _mm_loadu_si128(&buf)
                }
            });
            let rows = transpose(cols);
            for (rr, row) in rows.iter().enumerate() {
                let t: &mut [u16; 8] =
                    (&mut dst[(r0 + rr) * stride + c0..(r0 + rr) * stride + c0 + 8])
                        .try_into()
                        .unwrap();
                _mm_storeu_si128(t, *row);
            }
            r0 += 8;
        }
        for r in r0..bh {
            for k in 0..8 {
                dst[r * stride + c0 + k] = if r < n_acts[k] {
                    let t = pad + bases[k] + r * base_inc;
                    let a0 = edge[t] as i32;
                    let a1 = edge[t + 1] as i32;
                    ((a0 * (32 - shifts[k] as i32) + a1 * shifts[k] as i32 + 16) >> 5) as u16
                } else {
                    fillv
                };
            }
        }
        c0 += 8;
    }
    // Column tail: bw % 8 == 4 blocks. Scalar recipe verbatim.
    for c in c0..bw {
        let y = dy * (c + 1) as i32;
        let mut base = y >> frac_bits;
        let shift = ((y << up) & 0x3F) >> 1;
        for r in 0..bh {
            if base < max_base_y {
                let a0 = edge[pad + base as usize] as i32;
                let a1 = edge[pad + base as usize + 1] as i32;
                dst[r * stride + c] = ((a0 * (32 - shift) + a1 * shift + 16) >> 5) as u16;
                base += base_inc as i32;
            } else {
                for rr in r..bh {
                    dst[rr * stride + c] = fillv;
                }
                break;
            }
        }
    }
}

/// The scalar tier — `dir::z3_high_scalar`'s per-pixel recipe verbatim
/// (stride-`base_inc` taps for `up == 1`, contiguous for `up == 0`).
#[cfg(target_arch = "x86_64")]
fn z3_cols_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    edge: &[u16],
    pad: usize,
    dy: i32,
    up: i32,
) {
    z3_cols_body_scalar(dst, stride, bw, bh, edge, pad, dy, up);
}

/// The column loop shared by the scalar tier and the non-x86 fallback —
/// byte-identical to `dir::z3_high_scalar`'s per-pixel walk.
#[allow(clippy::too_many_arguments)]
fn z3_cols_body_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    edge: &[u16],
    pad: usize,
    dy: i32,
    up: i32,
) {
    let frac_bits = 6 - up;
    let base_inc = 1i32 << up;
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    let fillv = edge[pad + max_base_y as usize];
    let mut y = dy;
    for c in 0..bw {
        let mut base = y >> frac_bits;
        let shift = ((y << up) & 0x3F) >> 1;
        for r in 0..bh {
            if base < max_base_y {
                let a0 = edge[pad + base as usize] as i32;
                let a1 = edge[pad + base as usize + 1] as i32;
                dst[r * stride + c] = ((a0 * (32 - shift) + a1 * shift + 16) >> 5) as u16;
                base += base_inc;
            } else {
                for rr in r..bh {
                    dst[rr * stride + c] = fillv;
                }
                break;
            }
        }
        y += dy;
    }
}

/// One-`incant!` z3 entry — called from `super::dir::z3_high` only under the
/// `z3_vec_applies` gate (`up <= 1`, `bh >= MIN_VEC_RUN`, taps `<= I16_TAP_MAX`).
pub(crate) fn z3_cols(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    edge: &[u16],
    pad: usize,
    dy: i32,
    up: i32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = crate::dispatch::scalar_forced();
        archmage::incant!(
            z3_cols_impl(dst, stride, bw, bh, edge, pad, dy, up),
            [v3, scalar]
        );
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    z3_cols_body_scalar(dst, stride, bw, bh, edge, pad, dy, up);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every token permutation, against the scalar core, over the full admitted
    /// domain. Probes are asymmetric (a flat edge is invariant under the
    /// re-association being tested — playbook §1 / KB-12).
    #[test]
    fn two_tap_matches_scalar_at_every_tier() {
        let mut edge = vec![0u16; 200];
        let mut s = 0x1234_5678u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut vector_cells = 0usize;
        for rep in 0..8 {
            for (i, e) in edge.iter_mut().enumerate() {
                *e = match rep {
                    0 => (next() % (I16_TAP_MAX as u32 + 1)) as u16, // dense random
                    1 => {
                        if i % 2 == 0 {
                            I16_TAP_MAX
                        } else {
                            0
                        }
                    } // max sawtooth
                    2 => (i as u16) % 256,                        // ramp
                    3 => I16_TAP_MAX,                             // flat max
                    4 => (next() % 256) as u16,                   // bd8 range
                    5 => 255 - (i as u16 % 256),                  // reverse ramp
                    6 => {
                        if i < 100 {
                            0
                        } else {
                            I16_TAP_MAX
                        }
                    } // step
                    _ => (next() % 1024) as u16,
                };
            }
            for &n in &[1usize, 4, 7, 8, 9, 15, 16, 17, 31, 32, 64] {
                for shift in 0..32i32 {
                    for &start in &[0usize, 1, 3, 16, 100] {
                        if start + n + 1 > edge.len() {
                            continue;
                        }
                        let mut got = vec![0u16; n];
                        let mut want = vec![0u16; n];
                        two_tap_run(&mut got, &edge, start, shift, n);
                        two_tap_run_scalar(&mut want, &edge, start, shift, n);
                        assert_eq!(got, want, "n={n} shift={shift} start={start} rep={rep}");
                        if n >= MIN_VEC_RUN {
                            vector_cells += 1;
                        }
                    }
                }
            }
        }
        // Non-vacuity: the vector body must actually have been reachable.
        assert!(vector_cells > 1000, "vector arm unreached ({vector_cells})");
    }

    /// Playbook §2 — the bound must BITE. One tap over `I16_TAP_MAX` and the
    /// i16 lanes must genuinely diverge from the scalar reference, else the
    /// gate is decorative.
    ///
    /// The divergence half is necessarily conditional on a VECTOR tier actually
    /// dispatching: under `AOM_FORCE_SCALAR=1` `two_tap_run` routes to
    /// `two_tap_run_scalar`, so it cannot diverge from itself, and asserting
    /// otherwise fails the scalar-pinned CI leg (it did, on the first run). The
    /// gate's own rejection is asserted UNconditionally — that half is pure
    /// arithmetic on the span and has no tier.
    #[test]
    fn the_tap_bound_is_load_bearing() {
        let n = 16;
        let mut edge = vec![I16_TAP_MAX; 64];
        // At exactly the bound, every shift agrees. True at every tier.
        for shift in 0..32i32 {
            let (mut got, mut want) = (vec![0u16; n], vec![0u16; n]);
            two_tap_run(&mut got, &edge, 0, shift, n);
            two_tap_run_scalar(&mut want, &edge, 0, shift, n);
            assert_eq!(got, want, "at the bound, shift={shift}");
        }
        // The gate rejects one over the bound, and accepts the bound itself.
        edge[3] = I16_TAP_MAX + 1;
        assert!(!span_fits_i16(&edge, 0, 16), "1024 must be rejected");
        edge[3] = I16_TAP_MAX;
        assert!(span_fits_i16(&edge, 0, 16), "1023 must be accepted");

        if crate::dispatch::scalar_forced() {
            return; // no vector tier to diverge; the half above still ran
        }
        // One over the bound, and the vector path is wrong for at least one
        // shift — else the gate guards nothing.
        edge[3] = I16_TAP_MAX + 1;
        let mut diverged = false;
        for shift in 0..32i32 {
            let (mut got, mut want) = (vec![0u16; n], vec![0u16; n]);
            two_tap_run(&mut got, &edge, 0, shift, n);
            two_tap_run_scalar(&mut want, &edge, 0, shift, n);
            if got != want {
                diverged = true;
            }
        }
        assert!(
            diverged,
            "the i16 tap bound never bites — the gate would be decorative"
        );
    }

    /// `z2_left_gather` vs its scalar recipe across the admitted domain:
    /// block shapes 4..=64, both `up_left` values, `dx`/`dy` over the
    /// signalled z2 ranges, and samples past `I16_TAP_MAX` — the gather runs
    /// i32 lanes precisely so it needs no data bound, and bd12-range probes
    /// pin that.
    #[test]
    fn z2_left_gather_matches_scalar_recipe() {
        let mut s = 0x9E37_79B9u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut vec_chunks = 0usize;
        for rep in 0..4 {
            // pad=8, 144 usable samples (the production max is ~129), tail pad.
            let mut ld = vec![0u16; 160];
            for (i, e) in ld.iter_mut().enumerate() {
                *e = match rep {
                    0 => (next() % 4096) as u16, // bd12 dense random
                    1 => (next() % 65536) as u16, // full u16 (no data bound)
                    2 => ((i as u32 * 53) % 4096) as u16, // ramp
                    _ => 4095,                   // flat bd12 max
                };
            }
            for &up_left in &[0u32, 1] {
                let frac_y = 6 - up_left;
                for &(bw, bh) in &[(8usize, 8usize), (16, 16), (32, 32), (64, 64), (4, 8), (8, 16), (16, 8), (64, 16)] {
                    for &dx in &[4i32, 17, 32, 45, 64, 90, 121, 190, 361] {
                        for &dy in &[4i32, 17, 45, 90, 190] {
                            // Feasibility: every kept lane's `pad + base_y` /
                            // `+ 1` must be in bounds. `base_y` is
                            // non-increasing in c, so per row the extremes are
                            // lanes 0 and c_end-1.
                            let mut ok = true;
                            for r in 0..bh {
                                let y = (r + 1) as i32;
                                let ce =
                                    ((y * dx - 1) >> 6).clamp(0, bw as i32) as usize;
                                if ce == 0 {
                                    continue;
                                }
                                let r6 = (r as i32) << 6;
                                let hi = (r6 - dy) >> frac_y;
                                let lo = (r6 - (ce as i32) * dy) >> frac_y;
                                if 8 + lo < 0 || 8 + hi + 2 > ld.len() as i32 {
                                    ok = false;
                                    break;
                                }
                            }
                            if !ok {
                                continue;
                            }
                            let stride = bw;
                            let mut got = vec![0x55u16; stride * bh];
                            let mut want = vec![0x55u16; stride * bh];
                            z2_left_gather(
                                &mut got, stride, bw, bh, &ld, 8, dx, dy, frac_y, up_left,
                            );
                            z2_left_gather_scalar(
                                &mut want, stride, bw, bh, &ld, 8, dx, dy, frac_y, up_left,
                            );
                            assert_eq!(got, want, "{bw}x{bh} dx={dx} dy={dy} up_l={up_left} rep={rep}");
                            for r in 0..bh {
                                let y = (r + 1) as i32;
                                let c_end = ((y * dx - 1) >> 6).clamp(0, bw as i32) as usize;
                                vec_chunks += c_end / 8;
                            }
                        }
                    }
                }
            }
        }
        assert!(vec_chunks > 200, "vector arm unreached ({vec_chunks})");
    }
}
