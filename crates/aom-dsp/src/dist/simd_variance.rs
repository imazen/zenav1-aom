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

/// Scalar tier for [`crate::dist::sse_u16_u8`] — the transcribed twin.
pub(crate) fn sse_u16_u8_impl_scalar(
    _t: archmage::ScalarToken,
    a: &[u16],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    crate::dist::sse_u16_u8_scalar(a, a_stride, b, b_stride, w, h)
}

/// Mixed-width SSE (`u16` source x `u8` reconstruction) — the bd8 lowbd shape
/// of [`highbd_variance64_impl`]'s square accumulation without the sum lane
/// (the caller discards `var`): per row `d = a - b` in i32 lanes,
/// `sse_v += d*d`, one `reduce_add` per row into `i64`. Bit-identical to the
/// scalar twin on the pixel domain: |diff| <= 255 at bd8, per-lane row sum of
/// squares <= (w/8)*255^2 < 2^28 for w <= 128 — the wrapping `as u32` reduce
/// is exact.
#[magetypes(define(i32x8), neon, wasm128, -scalar)]
pub(crate) fn sse_u16_u8_impl(
    token: Token,
    a: &[u16],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    if w == 4 {
        return crate::dist::sse_u16_u8_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(w >= 8 && w % 8 == 0, "block widths are powers of two");
    let widen_a = |s: &[u16]| -> i32x8 {
        let arr: [i32; 8] = core::array::from_fn(|k| s[k] as i32);
        i32x8::from_array(token, arr)
    };
    let widen_b = |s: &[u8]| -> i32x8 {
        let arr: [i32; 8] = core::array::from_fn(|k| s[k] as i32);
        i32x8::from_array(token, arr)
    };
    let mut tsse: i64 = 0;
    for y in 0..h {
        let ra = y * a_stride;
        let rb = y * b_stride;
        let mut sse_v = i32x8::zero(token);
        for c in (0..w).step_by(8) {
            let d = widen_a(&a[ra + c..ra + c + 8]) - widen_b(&b[rb + c..rb + c + 8]);
            sse_v = sse_v + d * d;
        }
        tsse += i64::from(sse_v.reduce_add() as u32);
    }
    tsse
}

/// x86-64/AVX2 body for `sse_u16_u8` — the lowbd shape of
/// `highbd_variance64_impl_v3`'s square accumulation: 16 `u16` source lanes
/// vs 16 `u8` recon lanes widened by `cvtepu8_epi16`, `sub_epi16` (|diff|
/// <= 255 at bd8 — exact in i16), `madd_epi16(d, d)` pair-sums squares into
/// i32 lanes (<= 2*255^2 per op). Accumulation strips at 8 rows like the
/// variance kernel (lane bound (w/16)*8*2*255^2 < 2^25 for w <= 128); a
/// w%16==8 row tail takes the xmm twin; w==4 and non-multiple widths are
/// scalar-routed by the caller.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub(crate) fn sse_u16_u8_impl_v3(
    _t: archmage::X64V3Token,
    a: &[u16],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    use archmage::intrinsics::x86_64::*;
    // Two-hadd tree: hadd(x,x) pairs each 128 half; hadd again folds the two
    // pair-sums of each half into lane 0; lo+hi adds the halves — lane 0 is
    // the strip total. i32 all the way (per-lane bound <= 8*2*255^2 < 2^25).
    let reduce = |xv: __m256i| -> i64 {
        let p1 = _mm256_hadd_epi32(xv, xv);
        let p2 = _mm256_hadd_epi32(p1, p1);
        let both = _mm_add_epi32(
            _mm256_castsi256_si128(p2),
            _mm256_extracti128_si256::<1>(p2),
        );
        i64::from(_mm_extract_epi32::<0>(both) as u32)
    };
    // C's `aom_sse` accumulates one ymm over the whole block and reduces once:
    // blocks are <= 64x64 so a lane gains at most 64*4*2*255^2 < 2^25 — no
    // strip machinery needed. Every tx width is a power of two, so the hot
    // widths monomorphise into `sse_u16_u8_rows_v3` (const-W row slices fold
    // every inner bounds check); the generic loop stays for the other
    // assert-permitted multiples of 8.
    match w {
        // w==4 is ~half the call volume (4-wide txbs) — a scalar walk costs
        // ~14 Ir/px vs ~1.3 packed. The generic tiers still take the scalar
        // route for it.
        4 => return sse_u16_u8_w4_v3(_t, a, a_stride, b, b_stride, h),
        8 => return sse_u16_u8_rows_v3::<8>(_t, a, a_stride, b, b_stride, h),
        16 => return sse_u16_u8_rows_v3::<16>(_t, a, a_stride, b, b_stride, h),
        32 => return sse_u16_u8_rows_v3::<32>(_t, a, a_stride, b, b_stride, h),
        64 => return sse_u16_u8_rows_v3::<64>(_t, a, a_stride, b, b_stride, h),
        _ => {}
    }
    let mut xv = _mm256_setzero_si256();
    for yy in 0..h {
        let (ra, rb) = (yy * a_stride, yy * b_stride);
        let ar = &a[ra..ra + w];
        let br = &b[rb..rb + w];
        for c in (0..w).step_by(16) {
            if c + 16 <= w {
                let av: &[u16; 16] = ar[c..c + 16].try_into().unwrap();
                let bv: &[u8; 16] = br[c..c + 16].try_into().unwrap();
                let d = _mm256_sub_epi16(
                    _mm256_loadu_si256(av),
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(bv)),
                );
                xv = _mm256_add_epi32(xv, _mm256_madd_epi16(d, d));
            } else {
                let av: &[u16; 8] = ar[c..c + 8].try_into().unwrap();
                let bv: &[u8; 8] = br[c..c + 8].try_into().unwrap();
                let d = _mm_sub_epi16(
                    _mm_loadu_si128(av),
                    _mm_cvtepu8_epi16(_mm_loadu_si64(bv)),
                );
                let sq = _mm_madd_epi16(d, d);
                xv = _mm256_add_epi32(
                    xv,
                    _mm256_castsi128_si256(sq),
                );
            }
        }
    }
    reduce(xv)
}

/// Const-width row loop for [`sse_u16_u8_impl_v3`] — one upfront region check
/// (`(h-1)*stride + W` covers every access) replaces the per-row slice checks,
/// and `&[u16; W]`/`&[u8; W]` row views carry the len in the type so the
/// chunk guard + every `try_into` fold at compile time: W=8 emits only the
/// xmm arm, W%16==0 only ymm, fully unrolled.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn sse_u16_u8_rows_v3<const W: usize>(
    _t: archmage::X64V3Token,
    a: &[u16],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    h: usize,
) -> i64 {
    use archmage::intrinsics::x86_64::*;
    if h == 0 {
        return 0;
    }
    let (need_a, need_b) = ((h - 1) * a_stride + W, (h - 1) * b_stride + W);
    if a.len() < need_a || b.len() < need_b {
        return crate::dist::sse_u16_u8_scalar(a, a_stride, b, b_stride, W, h);
    }
    let (aa, bb) = (&a[..need_a], &b[..need_b]);
    let reduce = |xv: __m256i| -> i64 {
        let p1 = _mm256_hadd_epi32(xv, xv);
        let p2 = _mm256_hadd_epi32(p1, p1);
        let both = _mm_add_epi32(
            _mm256_castsi256_si128(p2),
            _mm256_extracti128_si256::<1>(p2),
        );
        i64::from(_mm_extract_epi32::<0>(both) as u32)
    };
    let mut xv = _mm256_setzero_si256();
    let (mut ra, mut rb) = (0usize, 0usize);
    for _ in 0..h {
        let ar: &[u16; W] = aa[ra..ra + W].try_into().unwrap();
        let br: &[u8; W] = bb[rb..rb + W].try_into().unwrap();
        for c in (0..W).step_by(16) {
            if c + 16 <= W {
                let av: &[u16; 16] = ar[c..c + 16].try_into().unwrap();
                let bv: &[u8; 16] = br[c..c + 16].try_into().unwrap();
                let d = _mm256_sub_epi16(
                    _mm256_loadu_si256(av),
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(bv)),
                );
                xv = _mm256_add_epi32(xv, _mm256_madd_epi16(d, d));
            } else {
                let av: &[u16; 8] = ar[c..c + 8].try_into().unwrap();
                let bv: &[u8; 8] = br[c..c + 8].try_into().unwrap();
                let d = _mm_sub_epi16(
                    _mm_loadu_si128(av),
                    _mm_cvtepu8_epi16(_mm_loadu_si64(bv)),
                );
                let sq = _mm_madd_epi16(d, d);
                xv = _mm256_add_epi32(xv, _mm256_castsi128_si256(sq));
            }
        }
        ra += a_stride;
        rb += b_stride;
    }
    reduce(xv)
}

/// The w==4 arm of [`sse_u16_u8_impl_v3`]: each 4-px row is too narrow for
/// the hadd tree, so four strided rows pack into one 8-lane i16 op pair
/// (`loadu_si64` x4 + `unpacklo_epi64` for the u16 src, `loadu_si32` x4 +
/// `unpacklo_epi32`/`cvtepu8_epi16` for the u8 recon) — the same row-pair
/// packing `sum_squares_2d_i16`'s 4xn kernel uses. h is a power of two >= 4
/// in every production call; the `h % 4` tail and the undersized-buffer
/// corner keep the scalar walk (semantics identical — the math is exact
/// i16/i32, no saturation anywhere). One upfront `(h-1)*stride + 4` region
/// check covers every row access.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn sse_u16_u8_w4_v3(
    _t: archmage::X64V3Token,
    a: &[u16],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    h: usize,
) -> i64 {
    use archmage::intrinsics::x86_64::*;
    if h == 0 {
        return 0;
    }
    if h % 4 != 0
        || a.len() < (h - 1) * a_stride + 4
        || b.len() < (h - 1) * b_stride + 4
    {
        return crate::dist::sse_u16_u8_scalar(a, a_stride, b, b_stride, 4, h);
    }
    let (aa, bb) = (&a[..(h - 1) * a_stride + 4], &b[..(h - 1) * b_stride + 4]);
    let la = |o: usize| -> __m128i {
        let r: &[u16; 4] = aa[o..o + 4].try_into().unwrap();
        _mm_loadu_si64(r)
    };
    let lb = |o: usize| -> __m128i {
        let r: &[u8; 4] = bb[o..o + 4].try_into().unwrap();
        _mm_loadu_si32(r)
    };
    let mut acc = _mm_setzero_si128();
    let (mut ra, mut rb) = (0usize, 0usize);
    for _ in 0..h / 4 {
        // i32 lanes gain (h/4) * 2 * 255^2 <= 2.1M at h=64 — no wrap.
        let d01 = _mm_sub_epi16(
            _mm_unpacklo_epi64(la(ra), la(ra + a_stride)),
            _mm_cvtepu8_epi16(_mm_unpacklo_epi32(lb(rb), lb(rb + b_stride))),
        );
        let d23 = _mm_sub_epi16(
            _mm_unpacklo_epi64(la(ra + 2 * a_stride), la(ra + 3 * a_stride)),
            _mm_cvtepu8_epi16(_mm_unpacklo_epi32(
                lb(rb + 2 * b_stride),
                lb(rb + 3 * b_stride),
            )),
        );
        acc = _mm_add_epi32(
            acc,
            _mm_add_epi32(_mm_madd_epi16(d01, d01), _mm_madd_epi16(d23, d23)),
        );
        ra += 4 * a_stride;
        rb += 4 * b_stride;
    }
    let s = _mm_add_epi32(acc, _mm_srli_si128::<8>(acc));
    let s = _mm_add_epi32(s, _mm_srli_si128::<4>(s));
    i64::from(_mm_cvtsi128_si32(s) as u32)
}

/// Scalar tier for [`crate::dist::sse`] — the transcribed u8 SSE walk.
pub(crate) fn sse_u8_impl_scalar(
    _t: archmage::ScalarToken,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    crate::dist::sse_scalar(a, a_stride, b, b_stride, w, h)
}

/// u8 x u8 SSE (`aom_sse`) — same square-accumulate shape as
/// [`sse_u16_u8_impl`] with both sides u8. Per-lane row bound identical
/// (|diff| <= 255, w <= 128), so the wrapping u32 reduce is exact.
#[magetypes(define(i32x8), neon, wasm128, -scalar)]
pub(crate) fn sse_u8_impl(
    token: Token,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    if w == 4 {
        return crate::dist::sse_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(w >= 8 && w % 8 == 0, "block widths are powers of two");
    let widen = |s: &[u8]| -> i32x8 {
        let arr: [i32; 8] = core::array::from_fn(|k| s[k] as i32);
        i32x8::from_array(token, arr)
    };
    let mut tsse: i64 = 0;
    for y in 0..h {
        let ra = y * a_stride;
        let rb = y * b_stride;
        let mut sse_v = i32x8::zero(token);
        for c in (0..w).step_by(8) {
            let d = widen(&a[ra + c..ra + c + 8]) - widen(&b[rb + c..rb + c + 8]);
            sse_v = sse_v + d * d;
        }
        tsse += i64::from(sse_v.reduce_add() as u32);
    }
    tsse
}

/// x86-64/AVX2 body for `sse_u8` — the lowbd `aom_sse` shape: 16 `u8` lanes
/// widened by `cvtepu8_epi16` on both sides, `sub_epi16`, `madd_epi16(d, d)`,
/// one ymm accumulate over the whole block (lane bound 64*2*2*255^2 < 2^25),
/// one hadd-tree reduce. Byte-identical to `sse_scalar`: the math is exact
/// at every width the dispatcher routes here (w % 8 == 0, w >= 8).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub(crate) fn sse_u8_impl_v3(
    _t: archmage::X64V3Token,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    use archmage::intrinsics::x86_64::*;
    if w == 4 {
        return crate::dist::sse_scalar(a, a_stride, b, b_stride, w, h);
    }
    let reduce = |xv: __m256i| -> i64 {
        let p1 = _mm256_hadd_epi32(xv, xv);
        let p2 = _mm256_hadd_epi32(p1, p1);
        let both = _mm_add_epi32(
            _mm256_castsi256_si128(p2),
            _mm256_extracti128_si256::<1>(p2),
        );
        i64::from(_mm_extract_epi32::<0>(both) as u32)
    };
    let mut xv = _mm256_setzero_si256();
    for yy in 0..h {
        let (ra, rb) = (yy * a_stride, yy * b_stride);
        let ar = &a[ra..ra + w];
        let br = &b[rb..rb + w];
        for c in (0..w).step_by(16) {
            if c + 16 <= w {
                let av: &[u8; 16] = ar[c..c + 16].try_into().unwrap();
                let bv: &[u8; 16] = br[c..c + 16].try_into().unwrap();
                let d = _mm256_sub_epi16(
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(av)),
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(bv)),
                );
                xv = _mm256_add_epi32(xv, _mm256_madd_epi16(d, d));
            } else {
                let av: &[u8; 8] = ar[c..c + 8].try_into().unwrap();
                let bv: &[u8; 8] = br[c..c + 8].try_into().unwrap();
                let d = _mm_sub_epi16(
                    _mm_cvtepu8_epi16(_mm_loadu_si64(av)),
                    _mm_cvtepu8_epi16(_mm_loadu_si64(bv)),
                );
                let sq = _mm_madd_epi16(d, d);
                xv = _mm256_add_epi32(
                    xv,
                    _mm256_castsi128_si256(sq),
                );
            }
        }
    }
    reduce(xv)
}

/// Scalar tier for [`crate::dist::variance_4x4_units`] — delegates to the
/// transcribed twin `crate::dist::variance_4x4_units_scalar`.
pub(crate) fn variance4x4_units_impl_scalar(
    _t: archmage::ScalarToken,
    a: &[u16],
    a_stride: usize,
    off: usize,
    units: usize,
    out: &mut [(i32, u32)],
) {
    crate::dist::variance_4x4_units_scalar(a, a_stride, off, units, out);
}

/// Generic tier. Per-unit work is 16 samples — too small for a generic
/// i32x8 body to beat the scalar walk once the per-unit horizontal fold is
/// counted (same shape as the w==4 delegation in `highbd_variance64_impl`),
/// so the non-v3 tiers delegate; the v3 kernel below amortises the fold
/// across FOUR units per ymm.
#[magetypes(define(i32x8), neon, wasm128, -scalar)]
pub(crate) fn variance4x4_units_impl(
    _token: Token,
    a: &[u16],
    a_stride: usize,
    off: usize,
    units: usize,
    out: &mut [(i32, u32)],
) {
    crate::dist::variance_4x4_units_scalar(a, a_stride, off, units, out);
}

/// x86-64/AVX2 body for [`crate::dist::variance_4x4_units`] — the banded
/// walk. One ymm holds a row slice of FOUR consecutive units (16 u16);
/// `madd_epi16(v, ones)` / `madd_epi16(v, v)` turn it into i32 pair-sums
/// and pair-squares, accumulated over the unit's 4 rows, so lanes
/// (2g, 2g+1) of the accumulators hold unit g's full 16-pixel sum/sumsq.
/// One `hadd_epi32` per 4-unit group folds the pairs: its lo half is
/// `[u0.sum, u1.sum, u0.sq, u1.sq]` and its hi half `[u2.sum, u3.sum,
/// u2.sq, u3.sq]`.
///
/// # Bit-exactness vs the scalar twin
///
/// * `madd(v, ones)` pair-sums i16 lanes: |pair| <= 2*4095 on the pixel
///   domain (bd <= 12); 4-row accumulation <= 8*4095 — exact in i32.
/// * `madd(v, v)` pair-squares: <= 2*4095^2 per op, <= 8*4095^2 after 4
///   rows = 134,209,800 < 2^31 — exact in i32.
/// * The hadd fold adds each unit's two lane pairs — total <= 16*4095
///   (sum) / 16*4095^2 (sq) — exact.
/// * u16 lanes are reinterpreted as i16: correct iff every sample fits a
///   positive i16 — the pixel domain (bd <= 12 => <= 4095) guarantees it,
///   same domain assumption as `highbd_variance64_impl_v3` above.
/// * Tail groups of < 4 units take the scalar walk; a 2-unit remainder
///   takes the xmm twin (same madd/hadd shape, 8 u16 per row).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub(crate) fn variance4x4_units_impl_v3(
    _t: archmage::X64V3Token,
    a: &[u16],
    a_stride: usize,
    off: usize,
    units: usize,
    out: &mut [(i32, u32)],
) {
    use archmage::intrinsics::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let ones128 = _mm_set1_epi16(1);
    let mut u = 0usize;
    while u + 4 <= units {
        let base = off + 4 * u;
        let mut sv = _mm256_setzero_si256();
        let mut xv = sv;
        for r in 0..4 {
            let row: &[u16; 16] = a[base + r * a_stride..base + r * a_stride + 16]
                .try_into()
                .unwrap();
            let v = _mm256_loadu_si256(row);
            sv = _mm256_add_epi32(sv, _mm256_madd_epi16(v, ones));
            xv = _mm256_add_epi32(xv, _mm256_madd_epi16(v, v));
        }
        let t = _mm256_hadd_epi32(sv, xv);
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(&mut tmp, t);
        // lo: [s0, s1, x0, x1]; hi: [s2, s3, x2, x3].
        for g in 0..4 {
            let b = (g & 1) + 4 * (g >> 1);
            out[u + g] = (tmp[b], tmp[2 + b] as u32);
        }
        u += 4;
    }
    while u + 2 <= units {
        let base = off + 4 * u;
        let mut sv = _mm_setzero_si128();
        let mut xv = sv;
        for r in 0..4 {
            let row: &[u16; 8] = a[base + r * a_stride..base + r * a_stride + 8]
                .try_into()
                .unwrap();
            let v = _mm_loadu_si128(row);
            sv = _mm_add_epi32(sv, _mm_madd_epi16(v, ones128));
            xv = _mm_add_epi32(xv, _mm_madd_epi16(v, v));
        }
        let t = _mm_hadd_epi32(sv, xv);
        let mut tmp = [0i32; 4];
        _mm_storeu_si128(&mut tmp, t);
        out[u] = (tmp[0], tmp[2] as u32);
        out[u + 1] = (tmp[1], tmp[3] as u32);
        u += 2;
    }
    if u < units {
        let base = off + 4 * u;
        let mut tsum = 0i32;
        let mut tsse = 0u32;
        for r in 0..4 {
            for x in 0..4 {
                let d = a[base + r * a_stride + x] as i32;
                tsum += d;
                tsse = tsse.wrapping_add((d * d) as u32);
            }
        }
        out[u] = (tsum, tsse);
    }
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
    if w == 4 && h == 4 {
        // Two-thirds of all calls at encode time (aom_variance4x4_sse2 is
        // C's most-called variance kernel at --cpu-used 3). One range check
        // per plane buys out every inner subslice: on `v` of len `3*s + 4`
        // the `v[k*s .. k*s+4]` bounds for literal k < 4 fold to `k <= 3`.
        // Flat compares, not Option chains — they stay inlined.
        if a_stride <= (usize::MAX - 4) / 3
            && b_stride <= (usize::MAX - 4) / 3
            && 3 * a_stride + 4 <= a.len()
            && 3 * b_stride + 4 <= b.len()
        {
            let (sa, sb) = (3 * a_stride + 4, 3 * b_stride + 4);
            let (av, bv) = (&a[..sa], &b[..sb]);
            {
                let ra = |k: usize| -> __m128i {
                    let r: &[u16; 4] = av[k * a_stride..k * a_stride + 4].try_into().unwrap();
                    _mm_loadu_si64(r)
                };
                let rb = |k: usize| -> __m128i {
                    let r: &[u16; 4] = bv[k * b_stride..k * b_stride + 4].try_into().unwrap();
                    _mm_loadu_si64(r)
                };
                let d01 = _mm_sub_epi16(
                    _mm_unpacklo_epi64(ra(0), ra(1)),
                    _mm_unpacklo_epi64(rb(0), rb(1)),
                );
                let d23 = _mm_sub_epi16(
                    _mm_unpacklo_epi64(ra(2), ra(3)),
                    _mm_unpacklo_epi64(rb(2), rb(3)),
                );
                let d = _mm256_set_m128i(d23, d01);
                let sv = _mm256_madd_epi16(d, _mm256_set1_epi16(1));
                let xv = _mm256_madd_epi16(d, d);
                // Fold: sv i32 lanes -> i64 (sign-extend); xv lanes are
                // non-negative and each < 2*4095^2 < 2^26 per row-pair.
                let s128 =
                    _mm_add_epi32(_mm256_castsi256_si128(sv), _mm256_extracti128_si256::<1>(sv));
                let x128 =
                    _mm_add_epi32(_mm256_castsi256_si128(xv), _mm256_extracti128_si256::<1>(xv));
                let s64 = _mm_add_epi64(
                    _mm_cvtepi32_epi64(s128),
                    _mm_cvtepi32_epi64(_mm_srli_si128::<8>(s128)),
                );
                let x64 = _mm_add_epi64(
                    _mm_cvtepi32_epi64(x128),
                    _mm_cvtepi32_epi64(_mm_srli_si128::<8>(x128)),
                );
                let tsum = _mm_cvtsi128_si64(s64) + _mm_extract_epi64::<1>(s64);
                let tsse = (_mm_cvtsi128_si64(x64) + _mm_extract_epi64::<1>(x64)) as u64;
                return (tsse, tsum);
            }
        }
        // Stride degenerate (impossible for real callers) — generic arm.
    }
    if w == 4 && h <= 128 {
        // The dominant encode-time call shape (4-wide txb tails through
        // `dist_block_px_domain_into`): C answers it with `aom_variance4x4`
        // — a ~50-instruction dedicated kernel — while the generic strip
        // machinery below costs ~260/call. This arm is C's `variance4_sse2`
        // shape: two rows per xmm, one sub + two madds per pair, i32-lane
        // accumulators.
        //
        // Exactness: the xv lane bound is h/2 * 2*4095^2 < 2^31, i.e.
        // h <= 128 — already the largest real block height; taller inputs
        // take the generic arm below (its strip loop has the same bound per
        // strip but re-folds into u64 every 32 rows). The i64 fold
        // sign-extends each lane, so sv's negative sums are exact too —
        // unlike C's i16-lane `sum` fold, which assumes u8 pixels.
        let ld = |p: &[u16], off: usize| -> __m128i {
            let r: &[u16; 4] = p[off..off + 4].try_into().unwrap();
            _mm_loadu_si64(r)
        };
        let ones128 = _mm_set1_epi16(1);
        let mut sv = _mm_setzero_si128();
        let mut xv = sv;
        let mut y = 0usize;
        while y + 2 <= h {
            let d = _mm_sub_epi16(
                _mm_unpacklo_epi64(ld(a, y * a_stride), ld(a, y * a_stride + a_stride)),
                _mm_unpacklo_epi64(ld(b, y * b_stride), ld(b, y * b_stride + b_stride)),
            );
            sv = _mm_add_epi32(sv, _mm_madd_epi16(d, ones128));
            xv = _mm_add_epi32(xv, _mm_madd_epi16(d, d));
            y += 2;
        }
        if y < h {
            // Odd tail row (unreachable for real block heights).
            let d = _mm_sub_epi16(ld(a, y * a_stride), ld(b, y * b_stride));
            sv = _mm_add_epi32(sv, _mm_madd_epi16(d, ones128));
            xv = _mm_add_epi32(xv, _mm_madd_epi16(d, d));
        }
        let s64 = _mm_add_epi64(
            _mm_cvtepi32_epi64(sv),
            _mm_cvtepi32_epi64(_mm_srli_si128::<8>(sv)),
        );
        let x64 = _mm_add_epi64(
            _mm_cvtepi32_epi64(xv),
            _mm_cvtepi32_epi64(_mm_srli_si128::<8>(xv)),
        );
        let tsum = _mm_cvtsi128_si64(s64) + _mm_extract_epi64::<1>(s64);
        let tsse = (_mm_cvtsi128_si64(x64) + _mm_extract_epi64::<1>(x64)) as u64;
        return (tsse, tsum);
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
    // Width-specialized bodies mirror C's per-size kernels
    // (`aom_highbd_8_bit_variance{8,16,32,64,128}xH_*`): with `W` literal the
    // row's chunk loop unrolls to a constant trip count and — after one
    // checked `a[ra..ra+W]` slice per row — every inner `ar[c..c+16]` index
    // is statically in range (`c + 16 <= W == ar.len()`), so the checks fold
    // away. Other multiples of 8 (only reachable via the frame-edge
    // `pixel_dist_visible_only` clip, never a real block width) take the
    // generic body.
    match w {
        8 => return var_w_v3::<8>(_t, ones, a, a_stride, b, b_stride, h),
        16 => return var_w_v3::<16>(_t, ones, a, a_stride, b, b_stride, h),
        32 => return var_w_v3::<32>(_t, ones, a, a_stride, b, b_stride, h),
        64 => return var_w_v3::<64>(_t, ones, a, a_stride, b, b_stride, h),
        128 => return var_w_v3::<128>(_t, ones, a, a_stride, b, b_stride, h),
        _ => {}
    }

    let ones128 = _mm_set1_epi16(1);
    let strip = (8 / w.div_ceil(16)).max(1).min(h);
    let mut y = 0usize;
    while y < h {
        let yend = (y + strip).min(h);
        let mut sv = _mm256_setzero_si256();
        let mut xv = sv;
        for yy in y..yend {
            let (ra, rb) = (yy * a_stride, yy * b_stride);
            let ar = &a[ra..ra + w];
            let br = &b[rb..rb + w];
            let mut c = 0;
            while c + 16 <= w {
                let av: &[u16; 16] = ar[c..c + 16].try_into().unwrap();
                let bv: &[u16; 16] = br[c..c + 16].try_into().unwrap();
                let d = _mm256_sub_epi16(_mm256_loadu_si256(av), _mm256_loadu_si256(bv));
                sv = _mm256_add_epi32(sv, _mm256_madd_epi16(d, ones));
                xv = _mm256_add_epi32(xv, _mm256_madd_epi16(d, d));
                c += 16;
            }
            if c < w {
                // w % 16 == 8 tail (w%8==0 is a caller precondition).
                let av: &[u16; 8] = ar[c..c + 8].try_into().unwrap();
                let bv: &[u16; 8] = br[c..c + 8].try_into().unwrap();
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

/// Width-specialized AVX2 body for [`highbd_variance64_impl_v3`]. `W` is a
/// compile-time block width (multiple of 8, >= 8); the per-row `&a[ra..ra+W]`
/// slice is the single checked index, after which every chunk load is
/// statically in range. Accumulation/reduction semantics are identical to the
/// generic arm's (`reduce!` shared via copy — the strip bound derivation is
/// the same formula with `W` folded in).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn var_w_v3<const W: usize>(
    _t: archmage::X64V3Token,
    ones: archmage::intrinsics::x86_64::__m256i,
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    h: usize,
) -> (u64, i64) {
    use archmage::intrinsics::x86_64::*;
    let mut tsum: i64 = 0;
    let mut tsse: u64 = 0;
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
    let strip = (8 / W.div_ceil(16)).max(1).min(h);
    let mut y = 0usize;
    while y < h {
        let yend = (y + strip).min(h);
        let mut sv = _mm256_setzero_si256();
        let mut xv = sv;
        for yy in y..yend {
            let (ra, rb) = (yy * a_stride, yy * b_stride);
            let ar = &a[ra..ra + W];
            let br = &b[rb..rb + W];
            let mut c = 0;
            while c + 16 <= W {
                let av: &[u16; 16] = ar[c..c + 16].try_into().unwrap();
                let bv: &[u16; 16] = br[c..c + 16].try_into().unwrap();
                let d = _mm256_sub_epi16(_mm256_loadu_si256(av), _mm256_loadu_si256(bv));
                sv = _mm256_add_epi32(sv, _mm256_madd_epi16(d, ones));
                xv = _mm256_add_epi32(xv, _mm256_madd_epi16(d, d));
                c += 16;
            }
            if c < W {
                let av: &[u16; 8] = ar[c..c + 8].try_into().unwrap();
                let bv: &[u16; 8] = br[c..c + 8].try_into().unwrap();
                let d = _mm_sub_epi16(_mm_loadu_si128(av), _mm_loadu_si128(bv));
                let ones128 = _mm_set1_epi16(1);
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
