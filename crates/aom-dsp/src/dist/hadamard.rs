//! Hadamard transform + SATD, bit-exact port of libaom v3.14.1 `aom_dsp/avg.c`.
//! Used for SATD-based RD cost in the encoder. Internal passes are int16
//! (wrapping), matching the C dynamic-range contract; the SSE2/AVX2 output
//! transposes are replicated.

use archmage::prelude::*;

/// `hadamard_col8` (`aom_dsp/avg.c:149`) over eight ALREADY-GATHERED values.
///
/// Taking `[i16; 8]` by value rather than `(&[i16], off, stride)` is what makes
/// this compile to twenty-four register add/subs: a slice parameter forces a
/// bounds check on each of the eight strided loads (128 per 8x8 block, since
/// the block runs this sixteen times), and the runtime slice length stops LLVM
/// keeping the intermediates in registers or vectorizing the caller's loop.
/// The arithmetic is byte-for-byte the C, `wrapping` included.
#[inline(always)]
fn hadamard_col8(s: [i16; 8]) -> [i16; 8] {
    let b0 = s[0].wrapping_add(s[1]);
    let b1 = s[0].wrapping_sub(s[1]);
    let b2 = s[2].wrapping_add(s[3]);
    let b3 = s[2].wrapping_sub(s[3]);
    let b4 = s[4].wrapping_add(s[5]);
    let b5 = s[4].wrapping_sub(s[5]);
    let b6 = s[6].wrapping_add(s[7]);
    let b7 = s[6].wrapping_sub(s[7]);
    let c0 = b0.wrapping_add(b2);
    let c1 = b1.wrapping_add(b3);
    let c2 = b0.wrapping_sub(b2);
    let c3 = b1.wrapping_sub(b3);
    let c4 = b4.wrapping_add(b6);
    let c5 = b5.wrapping_add(b7);
    let c6 = b4.wrapping_sub(b6);
    let c7 = b5.wrapping_sub(b7);
    let mut o = [0i16; 8];
    o[0] = c0.wrapping_add(c4);
    o[7] = c1.wrapping_add(c5);
    o[3] = c2.wrapping_add(c6);
    o[4] = c3.wrapping_add(c7);
    o[2] = c0.wrapping_sub(c4);
    o[6] = c1.wrapping_sub(c5);
    o[1] = c2.wrapping_sub(c6);
    o[5] = c3.wrapping_sub(c7);
    o
}

/// `hadamard_col4` (`aom_dsp/avg.c:132`) over four ALREADY-GATHERED values.
/// The first stage widens to `i32` and shifts before narrowing, exactly as C
/// does; see [`hadamard_col8`] for why the parameter is an array.
#[inline(always)]
fn hadamard_col4(s: [i16; 4]) -> [i16; 4] {
    let v = |k: usize| s[k] as i32;
    let b0 = ((v(0) + v(1)) >> 1) as i16;
    let b1 = ((v(0) - v(1)) >> 1) as i16;
    let b2 = ((v(2) + v(3)) >> 1) as i16;
    let b3 = ((v(2) - v(3)) >> 1) as i16;
    [
        b0.wrapping_add(b2),
        b1.wrapping_add(b3),
        b0.wrapping_sub(b2),
        b1.wrapping_sub(b3),
    ]
}

/// `aom_hadamard_4x4_c`. `src` row stride is `src_stride`. Returns 16 coeffs.
pub fn hadamard_4x4(src: &[i16], src_stride: usize) -> [i32; 16] {
    let mut rows = [[0i16; 4]; 4];
    for (r, row) in rows.iter_mut().enumerate() {
        row.copy_from_slice(&src[r * src_stride..r * src_stride + 4]);
    }
    // C: buffer[idx*4 + k] = A[idx][k], A[idx] = col4(source column idx).
    let a: [[i16; 4]; 4] =
        core::array::from_fn(|idx| hadamard_col4(core::array::from_fn(|k| rows[k][idx])));
    // C's second pass reads `buffer[idx + k*4]`, which under that layout is
    // A[k][idx] — see [`hadamard_8x8`] for the substitution written out.
    let b: [[i16; 4]; 4] =
        core::array::from_fn(|idx| hadamard_col4(core::array::from_fn(|k| a[k][idx])));
    let mut coeff = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            coeff[i * 4 + j] = b[j][i] as i32;
        }
    }
    coeff
}

/// `aom_hadamard_8x8_c`. Returns 64 coeffs.
///
/// Same values as the C, with C's own index algebra folded so the trailing
/// transpose is never performed. C writes `buffer[idx*8 + k] = A[idx][k]` where
/// `A[idx] = col8(source column idx)`; its second pass reads
/// `buffer[idx + k*8]`, which in that layout is `A[k][idx]`, so
/// `B[idx] = col8([A[k][idx] for k])` and it writes
/// `buffer2[idx*8 + k] = B[idx][k]`. The final
/// *"Extra transpose to match SSE2 behavior"* (`aom_dsp/avg.c:232-236`) is
/// `coeff[i*8 + j] = buffer2[j*8 + i]`, i.e. exactly `B[j][i]` — so emitting
/// `B[j][i]` in place IS the transpose, not an omission of it. KB-12 is the
/// standing warning that this transpose is load-bearing (dropping it moves the
/// eob and nothing else, which reads as a near-tie for four sessions), so it is
/// preserved by construction here rather than by a separate pass.
pub fn hadamard_8x8(src: &[i16], src_stride: usize) -> [i32; 64] {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = crate::dispatch::scalar_forced();
        if let Some(c) = archmage::incant!(hadamard_8x8_avx2(src, src_stride), [v3, scalar]) {
            return c;
        }
    }
    hadamard_8x8_scalar_core(src, src_stride)
}

/// AVX2/SSE2 `aom_hadamard_8x8_sse2`'s shape: the eight ROW loads are
/// contiguous, the butterfly runs vertically across eight `__m128i`, and the
/// pass-2 horizontal butterfly is made vertical by ONE in-register 8x8 i16
/// transpose. Returns `None` on no tier, which routes to the scalar core.
///
/// **Why raw intrinsics rather than magetypes.** magetypes 0.9.29 has no
/// integer interleave/transpose of any width (they exist for `f32` only), which
/// is what `benchmarks/encoder_simd_lane_width_audit_2026-09-09.md` recorded as
/// blocking this kernel. `archmage::intrinsics` re-exports `core::arch`, and a
/// `#[rite(v3)]` body carries `target_feature(avx2)`, so the unpack family is
/// callable here **with `#![forbid(unsafe_code)]` still in force** — loads and
/// stores go through `safe_unaligned_simd`'s reference-based forms.
/// `benchmarks/encoder_intrinsics_unblock_2026-09-09.md` is the probe.
///
/// **Bit-exactness.** `_mm_add_epi16`/`_mm_sub_epi16` wrap, which is exactly
/// `i16::wrapping_add`/`_sub`; the butterfly is the same network in the same
/// order as [`hadamard_col8`], including its output permutation. The transpose
/// only MOVES lanes. KB-12 is the standing proof that a lost transpose here
/// perturbs the `eob` alone and reads as an RD near-tie, so this is gated by
/// `hadamard_diff` against the real exported C, not by inspection.
/// The `incant!` fallback: decline, routing to the scalar core. Also what the
/// `AOM_FORCE_SCALAR` pin selects.
#[cfg(target_arch = "x86_64")]
fn hadamard_8x8_avx2_scalar(
    _t: archmage::ScalarToken,
    _src: &[i16],
    _src_stride: usize,
) -> Option<[i32; 64]> {
    None
}

#[cfg(target_arch = "x86_64")]
#[magetypes(define(i16x8), v3, -scalar)]
fn hadamard_8x8_avx2(
    _token: Token,
    src: &[i16],
    src_stride: usize,
) -> Option<[i32; 64]> {
    use archmage::intrinsics::x86_64::*;

    // The eight-point butterfly of `hadamard_col8`, lane-parallel over 8 columns.
    let col8 = |s: [__m128i; 8]| -> [__m128i; 8] {
        let b0 = _mm_add_epi16(s[0], s[1]);
        let b1 = _mm_sub_epi16(s[0], s[1]);
        let b2 = _mm_add_epi16(s[2], s[3]);
        let b3 = _mm_sub_epi16(s[2], s[3]);
        let b4 = _mm_add_epi16(s[4], s[5]);
        let b5 = _mm_sub_epi16(s[4], s[5]);
        let b6 = _mm_add_epi16(s[6], s[7]);
        let b7 = _mm_sub_epi16(s[6], s[7]);
        let c0 = _mm_add_epi16(b0, b2);
        let c1 = _mm_add_epi16(b1, b3);
        let c2 = _mm_sub_epi16(b0, b2);
        let c3 = _mm_sub_epi16(b1, b3);
        let c4 = _mm_add_epi16(b4, b6);
        let c5 = _mm_add_epi16(b5, b7);
        let c6 = _mm_sub_epi16(b4, b6);
        let c7 = _mm_sub_epi16(b5, b7);
        let mut o = [_mm_setzero_si128(); 8];
        o[0] = _mm_add_epi16(c0, c4);
        o[7] = _mm_add_epi16(c1, c5);
        o[3] = _mm_add_epi16(c2, c6);
        o[4] = _mm_add_epi16(c3, c7);
        o[2] = _mm_sub_epi16(c0, c4);
        o[6] = _mm_sub_epi16(c1, c5);
        o[1] = _mm_sub_epi16(c2, c6);
        o[5] = _mm_sub_epi16(c3, c7);
        o
    };

    // Standard three-stage in-register 8x8 transpose of i16 lanes.
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

    let mut v = [_mm_setzero_si128(); 8];
    for (k, vk) in v.iter_mut().enumerate() {
        let row: &[i16; 8] = match src[k * src_stride..k * src_stride + 8].try_into() {
            Ok(r) => r,
            Err(_) => return None,
        };
        *vk = _mm_loadu_si128(row);
    }
    // Pass 1 is vertical (lane = column). Pass 2 is horizontal in that layout,
    // so transpose first and run the identical network again.
    let u = col8(transpose(col8(v)));

    let mut coeff = [0i32; 64];
    for (i, ui) in u.iter().enumerate() {
        let wide = _mm256_cvtepi16_epi32(*ui);
        let dst: &mut [i32; 8] = (&mut coeff[i * 8..i * 8 + 8]).try_into().ok()?;
        _mm256_storeu_si256(dst, wide);
    }
    Some(coeff)
}

/// The transcribed scalar core — the differential's reference and the
/// non-x86 / `AOM_FORCE_SCALAR` path.
fn hadamard_8x8_scalar_core(src: &[i16], src_stride: usize) -> [i32; 64] {
    let mut rows = [[0i16; 8]; 8];
    for (r, row) in rows.iter_mut().enumerate() {
        row.copy_from_slice(&src[r * src_stride..r * src_stride + 8]);
    }
    let a: [[i16; 8]; 8] =
        core::array::from_fn(|idx| hadamard_col8(core::array::from_fn(|k| rows[k][idx])));
    let b: [[i16; 8]; 8] =
        core::array::from_fn(|idx| hadamard_col8(core::array::from_fn(|k| a[k][idx])));
    let mut coeff = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            coeff[i * 8 + j] = b[j][i] as i32;
        }
    }
    coeff
}

/// `aom_hadamard_16x16_c`. Returns 256 coeffs.
pub fn hadamard_16x16(src: &[i16], src_stride: usize) -> [i32; 256] {
    let mut coeff = [0i32; 256];
    for idx in 0..4 {
        let off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        let sub = hadamard_8x8(&src[off..], src_stride);
        coeff[idx * 64..idx * 64 + 64].copy_from_slice(&sub);
    }
    for idx in 0..64 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 64];
        let a2 = coeff[idx + 128];
        let a3 = coeff[idx + 192];
        let b0 = (a0.wrapping_add(a1)) >> 1;
        let b1 = (a0.wrapping_sub(a1)) >> 1;
        let b2 = (a2.wrapping_add(a3)) >> 1;
        let b3 = (a2.wrapping_sub(a3)) >> 1;
        coeff[idx] = b0.wrapping_add(b2);
        coeff[idx + 64] = b1.wrapping_add(b3);
        coeff[idx + 128] = b0.wrapping_sub(b2);
        coeff[idx + 192] = b1.wrapping_sub(b3);
    }
    // Swap columns [4..8) and [8..12) of each row (AVX2 output order).
    for i in 0..16 {
        for j in 0..4 {
            coeff.swap(i * 16 + 4 + j, i * 16 + 8 + j);
        }
    }
    coeff
}

/// `aom_hadamard_32x32_c`: four 16x16 Hadamards over the quadrants, then a 4-point
/// combine (`>>2`) across the quadrant coefficients. Returns 1024 coeffs.
pub fn hadamard_32x32(src: &[i16], src_stride: usize) -> [i32; 1024] {
    let mut coeff = [0i32; 1024];
    for idx in 0..4 {
        let off = (idx >> 1) * 16 * src_stride + (idx & 1) * 16;
        let sub = hadamard_16x16(&src[off..], src_stride);
        coeff[idx * 256..idx * 256 + 256].copy_from_slice(&sub);
    }
    for idx in 0..256 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 256];
        let a2 = coeff[idx + 512];
        let a3 = coeff[idx + 768];
        let b0 = a0.wrapping_add(a1) >> 2;
        let b1 = a0.wrapping_sub(a1) >> 2;
        let b2 = a2.wrapping_add(a3) >> 2;
        let b3 = a2.wrapping_sub(a3) >> 2;
        coeff[idx] = b0.wrapping_add(b2);
        coeff[idx + 256] = b1.wrapping_add(b3);
        coeff[idx + 512] = b0.wrapping_sub(b2);
        coeff[idx + 768] = b1.wrapping_sub(b3);
    }
    coeff
}

// highbd Hadamard 8-point column butterfly, first pass (i16, truncating like C's
// int16_t). Output permutation matches aom_dsp/avg.c.
fn highbd_col8_first_pass(src: &[i16], stride: usize, out: &mut [i16]) {
    let s = |i: usize| src[i * stride];
    let b0 = s(0).wrapping_add(s(1));
    let b1 = s(0).wrapping_sub(s(1));
    let b2 = s(2).wrapping_add(s(3));
    let b3 = s(2).wrapping_sub(s(3));
    let b4 = s(4).wrapping_add(s(5));
    let b5 = s(4).wrapping_sub(s(5));
    let b6 = s(6).wrapping_add(s(7));
    let b7 = s(6).wrapping_sub(s(7));
    let c0 = b0.wrapping_add(b2);
    let c1 = b1.wrapping_add(b3);
    let c2 = b0.wrapping_sub(b2);
    let c3 = b1.wrapping_sub(b3);
    let c4 = b4.wrapping_add(b6);
    let c5 = b5.wrapping_add(b7);
    let c6 = b4.wrapping_sub(b6);
    let c7 = b5.wrapping_sub(b7);
    out[0] = c0.wrapping_add(c4);
    out[7] = c1.wrapping_add(c5);
    out[3] = c2.wrapping_add(c6);
    out[4] = c3.wrapping_add(c7);
    out[2] = c0.wrapping_sub(c4);
    out[6] = c1.wrapping_sub(c5);
    out[1] = c2.wrapping_sub(c6);
    out[5] = c3.wrapping_sub(c7);
}

// Second pass (i32, no truncation).
fn highbd_col8_second_pass(src: &[i16], stride: usize, out: &mut [i32]) {
    let s = |i: usize| src[i * stride] as i32;
    let b0 = s(0) + s(1);
    let b1 = s(0) - s(1);
    let b2 = s(2) + s(3);
    let b3 = s(2) - s(3);
    let b4 = s(4) + s(5);
    let b5 = s(4) - s(5);
    let b6 = s(6) + s(7);
    let b7 = s(6) - s(7);
    let c0 = b0 + b2;
    let c1 = b1 + b3;
    let c2 = b0 - b2;
    let c3 = b1 - b3;
    let c4 = b4 + b6;
    let c5 = b5 + b7;
    let c6 = b4 - b6;
    let c7 = b5 - b7;
    out[0] = c0 + c4;
    out[7] = c1 + c5;
    out[3] = c2 + c6;
    out[4] = c3 + c7;
    out[2] = c0 - c4;
    out[6] = c1 - c5;
    out[1] = c2 - c6;
    out[5] = c3 - c7;
}

/// `aom_highbd_hadamard_8x8_c`: 8-point column pass (i16) then row pass (i32).
pub fn highbd_hadamard_8x8(src: &[i16], src_stride: usize) -> [i32; 64] {
    let mut buffer = [0i16; 64];
    for idx in 0..8 {
        highbd_col8_first_pass(&src[idx..], src_stride, &mut buffer[idx * 8..idx * 8 + 8]);
    }
    let mut buffer2 = [0i32; 64];
    for idx in 0..8 {
        highbd_col8_second_pass(&buffer[idx..], 8, &mut buffer2[idx * 8..idx * 8 + 8]);
    }
    buffer2
}

/// `aom_highbd_hadamard_16x16_c`: four highbd 8x8 + a 4-point `>>1` combine.
pub fn highbd_hadamard_16x16(src: &[i16], src_stride: usize) -> [i32; 256] {
    let mut coeff = [0i32; 256];
    for idx in 0..4 {
        let off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        let sub = highbd_hadamard_8x8(&src[off..], src_stride);
        coeff[idx * 64..idx * 64 + 64].copy_from_slice(&sub);
    }
    for idx in 0..64 {
        let (a0, a1, a2, a3) = (coeff[idx], coeff[idx + 64], coeff[idx + 128], coeff[idx + 192]);
        let b0 = (a0 + a1) >> 1;
        let b1 = (a0 - a1) >> 1;
        let b2 = (a2 + a3) >> 1;
        let b3 = (a2 - a3) >> 1;
        coeff[idx] = b0 + b2;
        coeff[idx + 64] = b1 + b3;
        coeff[idx + 128] = b0 - b2;
        coeff[idx + 192] = b1 - b3;
    }
    coeff
}

/// `aom_highbd_hadamard_32x32_c`: four highbd 16x16 + a 4-point `>>2` combine.
pub fn highbd_hadamard_32x32(src: &[i16], src_stride: usize) -> [i32; 1024] {
    let mut coeff = [0i32; 1024];
    for idx in 0..4 {
        let off = (idx >> 1) * 16 * src_stride + (idx & 1) * 16;
        let sub = highbd_hadamard_16x16(&src[off..], src_stride);
        coeff[idx * 256..idx * 256 + 256].copy_from_slice(&sub);
    }
    for idx in 0..256 {
        let (a0, a1, a2, a3) = (coeff[idx], coeff[idx + 256], coeff[idx + 512], coeff[idx + 768]);
        let b0 = (a0 + a1) >> 2;
        let b1 = (a0 - a1) >> 2;
        let b2 = (a2 + a3) >> 2;
        let b3 = (a2 - a3) >> 2;
        coeff[idx] = b0 + b2;
        coeff[idx + 256] = b1 + b3;
        coeff[idx + 512] = b0 - b2;
        coeff[idx + 768] = b1 - b3;
    }
    coeff
}

/// `aom_satd_c`: sum of absolute coefficients.
pub fn satd(coeff: &[i32]) -> i32 {
    let mut s: i32 = 0;
    for &c in coeff {
        s = s.wrapping_add(c.abs());
    }
    s
}
