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
    let mut coeff = [0i32; 64];
    hadamard_8x8_into(src, src_stride, &mut coeff);
    coeff
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
    _out: &mut [i32; 64],
) -> Option<()> {
    None
}

#[cfg(target_arch = "x86_64")]
#[magetypes(define(i16x8), v3, -scalar)]
fn hadamard_8x8_avx2(
    _token: Token,
    src: &[i16],
    src_stride: usize,
    out: &mut [i32; 64],
) -> Option<()> {
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

    for (i, ui) in u.iter().enumerate() {
        let wide = _mm256_cvtepi16_epi32(*ui);
        let dst: &mut [i32; 8] = (&mut out[i * 8..i * 8 + 8]).try_into().ok()?;
        _mm256_storeu_si256(dst, wide);
    }
    Some(())
}

/// `aom_hadamard_lp_8x8` — the lowbd (i16-out) Hadamard the nonrd estimate
/// arm's `av1_block_yrd` runs per 8x8. The transform is the identical network
/// to [`hadamard_8x8_into`]'s (C shares `hadamard_col8_sse2` across the fp/lp
/// variants); only the i16-narrow store differs.
///
/// The lane order written is C's own: `hadamard_col8_sse2(iter=0)`'s fused
/// transpose already emits coefficient rows in the order the scalar port
/// reaches via its trailing `coeff[i*8+j] = buffer2[j*8+i]` — the transpose
/// documented on the aom-encode scalar twin (KB-12's eob order).
pub fn hadamard_lp_8x8(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = crate::dispatch::scalar_forced();
        if let Some(()) =
            archmage::incant!(hadamard_lp_8x8_v3(src_diff, src_stride, coeff), [v3, scalar])
        {
            return;
        }
    }
    hadamard_lp_8x8_scalar(src_diff, src_stride, coeff);
}

/// `aom_hadamard_lp_8x8_dual` (avg_intrin_sse2.c) — two adjacent 8x8s.
pub fn hadamard_lp_8x8_dual(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    for i in 0..2 {
        hadamard_lp_8x8(&src_diff[i * 8..], src_stride, &mut coeff[i * 64..]);
    }
}

/// `aom_hadamard_lp_16x16` — four 8x8 stages + the `_mm_srai_epi16(.., 1)`
/// cross-combine. The combine truncates BEFORE shifting
/// (`wrapping_add(..) >> 1`), matching the note on the aom-encode scalar twin:
/// the two differ from shift-then-truncate only when `|a0+a1| > i16::MAX`,
/// unreachable on the lp arm's 9-bit inputs.
pub fn hadamard_lp_16x16(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    for idx in 0..4 {
        let off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        hadamard_lp_8x8(&src_diff[off..], src_stride, &mut coeff[idx * 64..]);
    }
    #[cfg(target_arch = "x86_64")]
    {
        let _ = crate::dispatch::scalar_forced();
        if let Some(()) = archmage::incant!(hadamard_lp_16_combine_v3(coeff), [v3, scalar]) {
            return;
        }
    }
    hadamard_lp_16_combine_scalar(coeff);
}

#[cfg(target_arch = "x86_64")]
fn hadamard_lp_8x8_v3_scalar(
    _t: archmage::ScalarToken,
    _src: &[i16],
    _src_stride: usize,
    _out: &mut [i16],
) -> Option<()> {
    None
}

/// The lp 8x8 as `aom_hadamard_lp_8x8_sse2` writes it: 8 unaligned row loads,
/// the column butterfly, the in-register transpose, the second butterfly, 8
/// i16 stores. All loads/stores are `loadu`/`storeu` — C's `_mm_load_si128`
/// assumes its own aligned scratch; the port's `diff` buffer is a `Vec<i16>`
/// with no 16-byte guarantee.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i16x8), v3, -scalar)]
fn hadamard_lp_8x8_v3(
    _token: Token,
    src: &[i16],
    src_stride: usize,
    out: &mut [i16],
) -> Option<()> {
    use archmage::intrinsics::x86_64::*;

    // Same network as hadamard_8x8_avx2's col8 — naturally-ordered output.
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
    let u = col8(transpose(col8(v)));
    for (i, ui) in u.iter().enumerate() {
        let dst: &mut [i16; 8] = (&mut out[i * 8..i * 8 + 8]).try_into().ok()?;
        _mm_storeu_si128(dst, *ui);
    }
    Some(())
}

#[cfg(target_arch = "x86_64")]
fn hadamard_lp_16_combine_v3_scalar(_t: archmage::ScalarToken, _out: &mut [i16]) -> Option<()> {
    None
}

/// `aom_hadamard_lp_16x16_sse2`'s combine pass: 8 iterations of 4 loads,
/// add/sub, `srai(.., 1)` (truncate-then-shift — see the doc above), 4 stores.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i16x8), v3, -scalar)]
fn hadamard_lp_16_combine_v3(_token: Token, coeff: &mut [i16]) -> Option<()> {
    use archmage::intrinsics::x86_64::*;
    for idx in (0..64).step_by(8) {
        let c0: &[i16; 8] = coeff[idx..idx + 8].try_into().ok()?;
        let c1: &[i16; 8] = coeff[idx + 64..idx + 72].try_into().ok()?;
        let c2: &[i16; 8] = coeff[idx + 128..idx + 136].try_into().ok()?;
        let c3: &[i16; 8] = coeff[idx + 192..idx + 200].try_into().ok()?;
        let a0 = _mm_loadu_si128(c0);
        let a1 = _mm_loadu_si128(c1);
        let a2 = _mm_loadu_si128(c2);
        let a3 = _mm_loadu_si128(c3);
        let b0 = _mm_srai_epi16(_mm_add_epi16(a0, a1), 1);
        let b1 = _mm_srai_epi16(_mm_sub_epi16(a0, a1), 1);
        let b2 = _mm_srai_epi16(_mm_add_epi16(a2, a3), 1);
        let b3 = _mm_srai_epi16(_mm_sub_epi16(a2, a3), 1);
        let d0: &mut [i16; 8] = (&mut coeff[idx..idx + 8]).try_into().ok()?;
        _mm_storeu_si128(d0, _mm_add_epi16(b0, b2));
        let d1: &mut [i16; 8] = (&mut coeff[idx + 64..idx + 72]).try_into().ok()?;
        _mm_storeu_si128(d1, _mm_add_epi16(b1, b3));
        let d2: &mut [i16; 8] = (&mut coeff[idx + 128..idx + 136]).try_into().ok()?;
        _mm_storeu_si128(d2, _mm_sub_epi16(b0, b2));
        let d3: &mut [i16; 8] = (&mut coeff[idx + 192..idx + 200]).try_into().ok()?;
        _mm_storeu_si128(d3, _mm_sub_epi16(b1, b3));
    }
    Some(())
}

/// Scalar core of [`hadamard_lp_8x8`] — the aom-encode `hadamard_lp_8x8`
/// recipe (16x [`hadamard_col8`] + the trailing transpose that matches the
/// SSE2 tier's fused-transpose output order).
fn hadamard_lp_8x8_scalar(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    let mut rows = [[0i16; 8]; 8];
    for (r, row) in rows.iter_mut().enumerate() {
        row.copy_from_slice(&src_diff[r * src_stride..r * src_stride + 8]);
    }
    let a: [[i16; 8]; 8] =
        core::array::from_fn(|idx| hadamard_col8(core::array::from_fn(|k| rows[k][idx])));
    let b: [[i16; 8]; 8] =
        core::array::from_fn(|idx| hadamard_col8(core::array::from_fn(|k| a[k][idx])));
    for i in 0..8 {
        for j in 0..8 {
            coeff[i * 8 + j] = b[j][i];
        }
    }
}

fn hadamard_lp_16_combine_scalar(coeff: &mut [i16]) {
    for idx in 0..64 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 64];
        let a2 = coeff[idx + 128];
        let a3 = coeff[idx + 192];
        let b0 = a0.wrapping_add(a1) >> 1;
        let b1 = a0.wrapping_sub(a1) >> 1;
        let b2 = a2.wrapping_add(a3) >> 1;
        let b3 = a2.wrapping_sub(a3) >> 1;
        coeff[idx] = b0.wrapping_add(b2);
        coeff[idx + 64] = b1.wrapping_add(b3);
        coeff[idx + 128] = b0.wrapping_sub(b2);
        coeff[idx + 192] = b1.wrapping_sub(b3);
    }
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

/// [`hadamard_8x8`] into a caller-provided buffer — identical values. The
/// AVX2 tier writes `out` in place so the 256-byte result never moves through
/// an `Option<[i32; 64]>` return.
pub fn hadamard_8x8_into(src: &[i16], src_stride: usize, out: &mut [i32; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = crate::dispatch::scalar_forced();
        if let Some(()) = archmage::incant!(hadamard_8x8_avx2(src, src_stride, out), [v3, scalar])
        {
            return;
        }
    }
    *out = hadamard_8x8_scalar_core(src, src_stride);
}

/// [`hadamard_16x16`] into a caller-provided buffer — identical values, but no
/// 1 KB array is initialised and moved per call. The quadrant writes go
/// straight into `out` (each [`hadamard_8x8`] returns its 256-byte block), and
/// the combine pass updates `out` element-wise in place, which is safe because
/// each position is read once and written once.
pub fn hadamard_16x16_into(src: &[i16], src_stride: usize, out: &mut [i32; 256]) {
    for idx in 0..4 {
        let off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        hadamard_8x8_into(&src[off..], src_stride, (&mut out[idx * 64..idx * 64 + 64])
            .try_into()
            .unwrap());
    }
    for idx in 0..64 {
        let a0 = out[idx];
        let a1 = out[idx + 64];
        let a2 = out[idx + 128];
        let a3 = out[idx + 192];
        let b0 = (a0.wrapping_add(a1)) >> 1;
        let b1 = (a0.wrapping_sub(a1)) >> 1;
        let b2 = (a2.wrapping_add(a3)) >> 1;
        let b3 = (a2.wrapping_sub(a3)) >> 1;
        out[idx] = b0.wrapping_add(b2);
        out[idx + 64] = b1.wrapping_add(b3);
        out[idx + 128] = b0.wrapping_sub(b2);
        out[idx + 192] = b1.wrapping_sub(b3);
    }
    for i in 0..16 {
        for j in 0..4 {
            out.swap(i * 16 + 4 + j, i * 16 + 8 + j);
        }
    }
}

/// `aom_hadamard_16x16_c`. Returns 256 coeffs.
pub fn hadamard_16x16(src: &[i16], src_stride: usize) -> [i32; 256] {
    let mut coeff = [0i32; 256];
    hadamard_16x16_into(src, src_stride, &mut coeff);
    coeff
}

/// [`hadamard_32x32`] into a caller-provided buffer — identical values, no
/// 4 KB array initialised and moved per call.
pub fn hadamard_32x32_into(src: &[i16], src_stride: usize, out: &mut [i32; 1024]) {
    for idx in 0..4 {
        let off = (idx >> 1) * 16 * src_stride + (idx & 1) * 16;
        hadamard_16x16_into(&src[off..], src_stride, (&mut out[idx * 256..idx * 256 + 256])
            .try_into()
            .unwrap());
    }
    for idx in 0..256 {
        let a0 = out[idx];
        let a1 = out[idx + 256];
        let a2 = out[idx + 512];
        let a3 = out[idx + 768];
        let b0 = a0.wrapping_add(a1) >> 2;
        let b1 = a0.wrapping_sub(a1) >> 2;
        let b2 = a2.wrapping_add(a3) >> 2;
        let b3 = a2.wrapping_sub(a3) >> 2;
        out[idx] = b0.wrapping_add(b2);
        out[idx + 256] = b1.wrapping_add(b3);
        out[idx + 512] = b0.wrapping_sub(b2);
        out[idx + 768] = b1.wrapping_sub(b3);
    }
}

/// `aom_hadamard_32x32_c`: four 16x16 Hadamards over the quadrants, then a 4-point
/// combine (`>>2`) across the quadrant coefficients. Returns 1024 coeffs.
pub fn hadamard_32x32(src: &[i16], src_stride: usize) -> [i32; 1024] {
    let mut coeff = [0i32; 1024];
    hadamard_32x32_into(src, src_stride, &mut coeff);
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

/// [`highbd_hadamard_8x8`] into a caller-provided buffer — identical values;
/// the second pass writes `out` rows directly so the 256-byte result never
/// moves through a `copy_from_slice`.
pub fn highbd_hadamard_8x8_into(src: &[i16], src_stride: usize, out: &mut [i32; 64]) {
    let mut buffer = [0i16; 64];
    for idx in 0..8 {
        highbd_col8_first_pass(&src[idx..], src_stride, &mut buffer[idx * 8..idx * 8 + 8]);
    }
    for idx in 0..8 {
        highbd_col8_second_pass(&buffer[idx..], 8, &mut out[idx * 8..idx * 8 + 8]);
    }
}

/// `aom_highbd_hadamard_8x8_c`: 8-point column pass (i16) then row pass (i32).
pub fn highbd_hadamard_8x8(src: &[i16], src_stride: usize) -> [i32; 64] {
    let mut buffer2 = [0i32; 64];
    highbd_hadamard_8x8_into(src, src_stride, &mut buffer2);
    buffer2
}

/// [`highbd_hadamard_16x16`] into a caller-provided buffer — identical values.
pub fn highbd_hadamard_16x16_into(src: &[i16], src_stride: usize, out: &mut [i32; 256]) {
    for idx in 0..4 {
        let off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        highbd_hadamard_8x8_into(&src[off..], src_stride, (&mut out
            [idx * 64..idx * 64 + 64])
            .try_into()
            .unwrap());
    }
    for idx in 0..64 {
        let (a0, a1, a2, a3) = (out[idx], out[idx + 64], out[idx + 128], out[idx + 192]);
        let b0 = (a0 + a1) >> 1;
        let b1 = (a0 - a1) >> 1;
        let b2 = (a2 + a3) >> 1;
        let b3 = (a2 - a3) >> 1;
        out[idx] = b0 + b2;
        out[idx + 64] = b1 + b3;
        out[idx + 128] = b0 - b2;
        out[idx + 192] = b1 - b3;
    }
}

/// `aom_highbd_hadamard_16x16_c`: four highbd 8x8 + a 4-point `>>1` combine.
pub fn highbd_hadamard_16x16(src: &[i16], src_stride: usize) -> [i32; 256] {
    let mut coeff = [0i32; 256];
    highbd_hadamard_16x16_into(src, src_stride, &mut coeff);
    coeff
}

/// [`highbd_hadamard_32x32`] into a caller-provided buffer — identical values.
pub fn highbd_hadamard_32x32_into(src: &[i16], src_stride: usize, out: &mut [i32; 1024]) {
    for idx in 0..4 {
        let off = (idx >> 1) * 16 * src_stride + (idx & 1) * 16;
        highbd_hadamard_16x16_into(&src[off..], src_stride, (&mut out
            [idx * 256..idx * 256 + 256])
            .try_into()
            .unwrap());
    }
    for idx in 0..256 {
        let (a0, a1, a2, a3) = (out[idx], out[idx + 256], out[idx + 512], out[idx + 768]);
        let b0 = (a0 + a1) >> 2;
        let b1 = (a0 - a1) >> 2;
        let b2 = (a2 + a3) >> 2;
        let b3 = (a2 - a3) >> 2;
        out[idx] = b0 + b2;
        out[idx + 256] = b1 + b3;
        out[idx + 512] = b0 - b2;
        out[idx + 768] = b1 - b3;
    }
}

/// `aom_highbd_hadamard_32x32_c`: four highbd 16x16 + a 4-point `>>2` combine.
pub fn highbd_hadamard_32x32(src: &[i16], src_stride: usize) -> [i32; 1024] {
    let mut coeff = [0i32; 1024];
    highbd_hadamard_32x32_into(src, src_stride, &mut coeff);
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

/// `aom_satd_lp_c` (`aom_dsp/avg.c:520`) — sum of `|coeff|` over an lp
/// Hadamard block. The i32 accumulator cannot wrap on any i16 input
/// (256 * 32768 < 2^23 << 2^31).
pub fn satd_lp(coeff: &[i16], length: usize) -> i32 {
    coeff[..length].iter().map(|&c| i32::from(c).abs()).sum()
}

/// `aom_satd_lp` with runtime SIMD dispatch. The v3 tier mirrors
/// `aom_satd_lp_avx2` (`avg_intrin_avx2.c:523`): `abs_epi16` +
/// `madd_epi16(., 1)` + i32 accumulate — the kernel RTCD actually dispatches.
/// `abs_epi16` maps -32768 back to -32768 where `_c` gives +32768: the ONE
/// divergence anywhere, probed and bounded unreachable (the lp transforms'
/// output bound is ~32654) by
/// `nonrd_block_yrd_lp_diff::lp_satd_block_error_tiers_agree_over_the_reachable_range`.
pub fn satd_lp_simd(coeff: &[i16], length: usize) -> i32 {
    let _ = crate::dispatch::scalar_forced();
    incant!(satd_lp_impl(coeff, length), [v3, neon, wasm128, scalar])
}

/// Scalar tier = the transcribed port, verbatim.
fn satd_lp_impl_scalar(_t: archmage::ScalarToken, coeff: &[i16], length: usize) -> i32 {
    satd_lp(coeff, length)
}

/// Non-x86 tiers: the `_c` sum-of-abs form — under the tier's target
/// features LLVM lowers it to the `abs`/`madd` shape C's NEON kernel uses.
/// Widening to i32 BEFORE abs keeps `_c`'s +32768 at input -32768 — a
/// deliberate `abs_epi16` vs scalar-`abs` distinction C itself makes only on
/// the unreachable bound lane.
#[magetypes(neon, wasm128, -scalar)]
fn satd_lp_impl(_t: Token, coeff: &[i16], length: usize) -> i32 {
    satd_lp(coeff, length)
}

/// v3 mirror of `aom_satd_lp_avx2`: per 16 lanes, `abs_epi16` then
/// `madd_epi16(., 1)` into a wrapping-i32 accumulator; C's cascade fold at
/// the end. The madd pair sum of two -32768 abs results is -65536 — exact
/// in i32 — so the accumulator wraps only past 2^31 of total, i.e. never on
/// the reachable domain (and C's i32 accum is equally unwrappable there).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn satd_lp_impl_v3(_t: archmage::X64V3Token, coeff: &[i16], length: usize) -> i32 {
    use archmage::intrinsics::x86_64::*;
    // C loops `i += 16` unconditionally — a length % 16 != 0 would overread.
    if length % 16 != 0 || coeff.len() < length {
        return satd_lp(coeff, length);
    }
    let one = _mm256_set1_epi16(1);
    let mut accum = _mm256_setzero_si256();
    for c in coeff[..length].as_chunks::<16>().0 {
        let src = _mm256_loadu_si256(c);
        let abs = _mm256_abs_epi16(src);
        accum = _mm256_add_epi32(accum, _mm256_madd_epi16(abs, one));
    }
    // C's horizontal add (avg_intrin_avx2.c:535-542).
    let a = _mm256_srli_si256::<8>(accum);
    let b = _mm256_add_epi32(accum, a);
    let c2 = _mm256_srli_epi64::<32>(b);
    let d = _mm256_add_epi32(b, c2);
    let acc128 = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256::<1>(d));
    _mm_cvtsi128_si32(acc128)
}
