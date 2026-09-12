//! SIMD kernel for `highbd_variance64` — bit-identical to the scalar
//! port on the pixel domain, at every dispatch tier
//! (`tests/hbd_variance_simd_diff.rs`).
//!
//! Same aom-rs SIMD pattern as `crate::quant::simd` / `crate::cdef::simd`: ONE
//! magetypes generic kernel (`#[magetypes(v3, neon, wasm128, -scalar)]`),
//! hand-written `_scalar` variant = the transcribed port verbatim,
//! `incant!` dispatch in the caller, `crate::dispatch::scalar_forced()` pin.
//!
//! # Bit-exactness (pixel domain: `a`, `b` < `1 << bd`, `bd <= 12`)
//!
//! Per pixel the scalar port computes `diff = a - b` (|diff| < 4096),
//! `lsum += diff` per row (|row sum| <= 128 * 4095 — no i32 wrap, so lane
//! sums + a horizontal reduce give the identical total), and
//! `tsse += (diff*diff as u32) as u64`. The lane square `mullo` wraps like
//! the scalar's i32 multiply (identical low-32 bits); per-lane u32 row
//! accumulation is exact (row of 128 => 16 squares/lane, each < 2^24 =>
//! per-lane sum < 2^28), and the wrapping u32 `reduce_add` cannot wrap
//! either (row total < 128 * 2^24 = 2^31). Each row's reductions land in
//! the u64/i64 totals exactly as the scalar's do. Block widths are powers
//! of two, so `w >= 8` implies `w % 8 == 0` (asserted).

use archmage::prelude::*;

/// Scalar tier = the transcribed port, verbatim.
pub(crate) fn highbd_variance64_impl_scalar(
    _t: archmage::ScalarToken,
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (u64, i64) {
    crate::dist::highbd_variance64_scalar(a, a_stride, b, b_stride, w, h)
}

// `-scalar` drops the macro's auto-appended scalar variant (the hand-written
// `_scalar` above takes that slot); `v3` is absent because a hand-written AVX2
// body below takes it — i16 lanes and `madd` pair tricks the magetypes API
// cannot express (no i16x16 type, no madd).
#[magetypes(define(i32x8), neon, wasm128, -scalar)]
pub(crate) fn highbd_variance64_impl(
    token: Token,
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (u64, i64) {
    if w == 4 {
        // The v3 body handles 4-wide blocks packed; the generic tiers take the
        // scalar walk for them (bit-identical, just slower — the hot w=4 caller
        // is `dist_block_px_domain`, an x86-heavy path).
        return crate::dist::highbd_variance64_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(w >= 8 && w % 8 == 0, "block widths are powers of two");
    let widen = |s: &[u16]| -> i32x8 {
        let arr: [i32; 8] = core::array::from_fn(|k| s[k] as i32);
        i32x8::from_array(token, arr)
    };
    let mut tsum: i64 = 0;
    let mut tsse: u64 = 0;
    for y in 0..h {
        let ra = y * a_stride;
        let rb = y * b_stride;
        let mut sum_v = i32x8::zero(token);
        // Squares accumulate in i32 lanes: two's-complement adds are
        // bit-identical to the u32 adds the scalar performs, and the final
        // `as u32` reinterpretation recovers the exact row total (< 2^31,
        // per the module-doc bound, so no reduce wrap either).
        let mut sse_v = i32x8::zero(token);
        for c in (0..w).step_by(8) {
            let d = widen(&a[ra + c..ra + c + 8]) - widen(&b[rb + c..rb + c + 8]);
            sum_v = sum_v + d;
            // (diff*diff) as u32 — the lane mullo wraps exactly like the
            // scalar's i32 multiply.
            sse_v = sse_v + d * d;
        }
        tsum += i64::from(sum_v.reduce_add());
        tsse += u64::from(sse_v.reduce_add() as u32);
    }
    (tsse, tsum)
}

/// x86-64/AVX2 body for `highbd_variance64` — C's `variance4x4_64_sse4_1`
/// shape (highbd_variance_sse4.c) generalised: **i16 lanes** (the pixel
/// domain `a,b < 1<<bd <= 4096` keeps `sub_epi16` diffs exact in i16),
/// `madd_epi16(d, ones)` for the sum and `madd_epi16(d, d)` for the square
/// pair-sums (one instruction computes `d0*d0 + d1*d1`), i32-lane
/// accumulation, one horizontal pair-reduce per row-group.
///
/// # Bit-exactness vs the scalar twin
///
/// * `diff = a - b` fits i16 (|diff| < 4096 on the pixel domain).
/// * `madd(d, ones)` pair-sums to i32: |sum| <= 2*4095 per op.
/// * `madd(d, d)` pair-sums squares: <= 2^25 per i32 lane per op. Per-row
///   accumulation caps a lane at (w/16)*2^25 <= 2^28 for w <= 128; the
///   reduce adds row totals to `tsse` as `u32 as u64` exactly like the
///   scalar's wrapping `as u32` cast (never reached, but identical anyway).
/// * `tsum` adds row totals in i64 — same value as the scalar's per-row
///   `lsum as i64` adds (addition order within a row is irrelevant in i64).
/// * w==4 packs FOUR rows per ymm (`loadu_si64` leaves zero padding, which
///   diffs to 0 and contributes nothing); w%16==8 rows take an 8-lane xmm
///   tail; `h % rows_per_vec` tails and w>256 fall to the scalar walk.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub(crate) fn highbd_variance64_impl_v3(
    _t: archmage::X64V3Token,
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (u64, i64) {
    use archmage::intrinsics::x86_64::*;
    if w > 256 {
        // Per-row i32-lane sse bound (w/16)*2^25 < 2^31 needs w < 1024;
        // 256 keeps headroom and no real caller exceeds 128.
        return crate::dist::highbd_variance64_scalar(a, a_stride, b, b_stride, w, h);
    }
    let ones = _mm256_set1_epi16(1);
    let mut tsum: i64 = 0;
    let mut tsse: u64 = 0;

    // One horizontal pair-reduce producing (sum_total, sse_total) of two i32x8
    // accumulators: hadd twice, fold halves, lane 0 = sum, lane 1 = sse.
    // The hadd tree adds lanes pairwise in i32, so it is only exact while the
    // WHOLE strip total stays < 2^31 — the strip bounds below enforce that
    // (per-lane <= 8 madds * 2*4095^2 = 268.3M; the tree's deepest node sums
    // all 8 lanes <= 2,146,435,200 < 2^31).
    macro_rules! reduce {
        ($sv:expr, $xv:expr) => {{
            let p1 = _mm256_hadd_epi32($sv, $xv);
            let p2 = _mm256_hadd_epi32(p1, p1);
            let both = _mm_add_epi32(
                _mm256_castsi256_si128(p2),
                _mm256_extracti128_si256::<1>(p2),
            );
            tsum += i64::from(_mm_extract_epi32::<0>(both));
            tsse += u64::from(_mm_extract_epi32::<1>(both) as u32);
        }};
    }

    if w == 4 {
        // Four rows per ymm: [u16;4] loads leave zeroed high lanes; the
        // unpacks/interleave order is irrelevant — every lane is summed.
        let ld = |p: &[u16], off: usize| -> __m128i {
            let r: &[u16; 4] = p[off..off + 4].try_into().unwrap();
            _mm_loadu_si64(r)
        };
        // sv/xv accumulate across row-groups in i32 lanes; a lane gains
        // <= 2*4095 (pair-sum) / <= 2*4095^2 (pair of squares) per iter, so
        // 8 iters keep the hadd-reduce tree inside i32 (see reduce! note).
        // Strip at 8 iters (32 rows) so an unbounded `h` cannot wrap.
        let mut y = 0usize;
        while y < h / 4 * 4 {
            let yend = (y + 8 * 4).min(h / 4 * 4);
            let mut sv = _mm256_setzero_si256();
            let mut xv = sv;
            for yy in (y..yend).step_by(4) {
                let (ra, rb) = (yy * a_stride, yy * b_stride);
                let av = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_unpacklo_epi64(ld(a, ra), ld(a, ra + a_stride))),
                    _mm_unpacklo_epi64(ld(a, ra + 2 * a_stride), ld(a, ra + 3 * a_stride)),
                );
                let bv = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_unpacklo_epi64(ld(b, rb), ld(b, rb + b_stride))),
                    _mm_unpacklo_epi64(ld(b, rb + 2 * b_stride), ld(b, rb + 3 * b_stride)),
                );
                let d = _mm256_sub_epi16(av, bv);
                sv = _mm256_add_epi32(sv, _mm256_madd_epi16(d, ones));
                xv = _mm256_add_epi32(xv, _mm256_madd_epi16(d, d));
            }
            reduce!(sv, xv);
            y = yend;
        }
        // h % 4 tail rows — unreachable for real block heights.
        for y in (h / 4 * 4)..h {
            for x in 0..4 {
                let diff = a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32;
                tsum += i64::from(diff);
                tsse += u64::from((diff * diff) as u32);
            }
        }
        return (tsse, tsum);
    }

    // sv/xv accumulate in i32 lanes ACROSS rows — the same structure as C's
    // `highbd_calc{8,16}x{8,16}var_avx2`, which reduces once per tile rather
    // than once per row. Per row a lane sees `ceil(w/16)` madds (the w%16==8
    // xmm tail adds a second madd into lanes 0..3), each contributing
    // <= 2*4095 / 2*4095^2. The hadd reduce sums all 8 lanes in i32, so a
    // lane may take at most 8 madds: strip = 8 / ceil(w/16) rows
    // (w<=16 -> 8, w=32 -> 4, w=64 -> 2, w>=128 -> 1, i.e. per-row again
    // only for the widest blocks where the row already fills the bound).
    let ones128 = _mm_set1_epi16(1);
    let strip = (8 / w.div_ceil(16)).max(1).min(h);
    let mut y = 0usize;
    while y < h {
        let yend = (y + strip).min(h);
        let mut sv = _mm256_setzero_si256();
        let mut xv = sv;
        for yy in y..yend {
            let (ra, rb) = (yy * a_stride, yy * b_stride);
            let mut c = 0;
            while c + 16 <= w {
                let av: &[u16; 16] = a[ra + c..ra + c + 16].try_into().unwrap();
                let bv: &[u16; 16] = b[rb + c..rb + c + 16].try_into().unwrap();
                let d = _mm256_sub_epi16(_mm256_loadu_si256(av), _mm256_loadu_si256(bv));
                sv = _mm256_add_epi32(sv, _mm256_madd_epi16(d, ones));
                xv = _mm256_add_epi32(xv, _mm256_madd_epi16(d, d));
                c += 16;
            }
            if c < w {
                // w % 16 == 8 tail (w%8==0 is a caller precondition).
                let av: &[u16; 8] = a[ra + c..ra + c + 8].try_into().unwrap();
                let bv: &[u16; 8] = b[rb + c..rb + c + 8].try_into().unwrap();
                let d = _mm_sub_epi16(_mm_loadu_si128(av), _mm_loadu_si128(bv));
                sv =
                    _mm256_add_epi32(sv, _mm256_zextsi128_si256(_mm_madd_epi16(d, ones128)));
                xv = _mm256_add_epi32(xv, _mm256_zextsi128_si256(_mm_madd_epi16(d, d)));
            }
        }
        reduce!(sv, xv);
        y = yend;
    }
    (tsse, tsum)
}
