//! Safe SIMD SAD via archmage `#[autoversion]` — no `unsafe`, no raw
//! `core::arch` intrinsics. `#[autoversion]` compiles one `#[target_feature]`-
//! gated variant per tier (AVX-512/AVX2/NEON/WASM/scalar) — unlocking LLVM's
//! auto-vectorizer to lower the sum-of-abs-diff loop to `psadbw` / `uabd` — plus
//! a runtime dispatcher. Result is byte-identical to scalar [`crate::dist::sad`].
//!
//! NOTE on dispatch: the generated `sad_simd` dispatcher pays a small
//! feature-check per call. In an encoder that is amortized by placing the
//! dispatch at the motion-search-loop entry (an `#[arcane]` boundary that calls
//! the SAD kernel per candidate); that entry point does not exist in this
//! kernel-only crate yet, so a per-block microbenchmark of the dispatcher is
//! dispatch-bound, not kernel-bound.

use archmage::autoversion;
use archmage::prelude::*;

/// Sum of absolute differences over a `w x h` block. Byte-identical to
/// [`crate::dist::sad`]; auto-vectorized, picks the best SIMD tier at runtime.
#[autoversion]
pub fn sad_simd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let arow = &a[y * a_stride..y * a_stride + w];
        let brow = &b[y * b_stride..y * b_stride + w];
        for x in 0..w {
            sum += (arow[x] as i32 - brow[x] as i32).unsigned_abs();
        }
    }
    sum
}

/// `aom_highbd_sad<W>x<H>` shape via `#[autoversion]` — SAD over this port's
/// u16-at-any-depth planes (samples < 4096 in every reachable bit depth).
/// The IntraBC full-pel DV search (`FullPelSearch::sad` -> `sad_wxh`) calls
/// this per diamond-search site and per hash candidate — measured 11.7 % flat
/// of a 1080p screen-content encode at `--cpu-used 6` (2026-09-13) when it was
/// a scalar i64-per-element loop.
///
/// # Bit-exactness
///
/// Identical to the scalar form: `u32` lane sums cannot overflow (|diff| <=
/// 4095 over at most 128*128 elements is under 2^24), and reassociation of an
/// unsigned sum is exact.
#[autoversion]
pub fn sad_u16_simd(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let arow = &a[y * a_stride..y * a_stride + w];
        let brow = &b[y * b_stride..y * b_stride + w];
        for x in 0..w {
            sum += (arow[x] as i32 - brow[x] as i32).unsigned_abs();
        }
    }
    sum
}

/// `av1_get_mvpred_var_cost`'s variance call shape via `#[autoversion]` — the
/// `sse - sum^2/(w*h)` integrand over u16 planes, from the IntraBC hash
/// candidate evaluation (`variance_wxh`). Same call density as the SAD above.
///
/// # Bit-exactness
///
/// Identical to the scalar form: `d` fits i32 (|d| <= 4095), `d*d` fits u32
/// (<= 2^24), `sum` fits i64 trivially, `sse` accumulates into u64 — every
/// sum reassociation is exact integer arithmetic. The `sum*sum` product can
/// reach ~(4095*16384)^2 < 2^53, inside u64.
#[autoversion]
pub fn variance_u16_simd(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sum = 0i64;
    let mut sse = 0u64;
    for y in 0..h {
        let arow = &a[y * a_stride..y * a_stride + w];
        let brow = &b[y * b_stride..y * b_stride + w];
        for x in 0..w {
            let d = arow[x] as i64 - brow[x] as i64;
            sum += d;
            sse += (d * d) as u64;
        }
    }
    (sse - ((sum * sum) as u64) / (w as u64 * h as u64)) as u32
}

/// `av1_block_error_c` (`av1/encoder/rdopt.c`) via `#[autoversion]` — the
/// transform-domain distortion at bd8, and one of the encoder's hottest loops
/// (`dist_block_tx_domain` measured **61 ms against libaom's
/// `av1_block_error_avx2` at 8.4 ms, 7.3x**, in
/// `benchmarks/encoder_x86_reprofile_1024_s3_2026-09-10.md`).
///
/// The scalar body was already bounds-check-free but stayed scalar: a 32-bit
/// `imul`, a sign-extend and a 64-bit add, two-way unrolled. `#[autoversion]`
/// compiles it once per SIMD tier with the target features enabled, which is
/// what lets LLVM lower it to 8-lane `pmulld` plus widening accumulation —
/// exactly libaom's AVX2 shape, with no raw intrinsics and no `unsafe`.
///
/// # Bit-exactness
///
/// Identical to [`crate::dist::block_error`] by construction. The per-element
/// arithmetic is unchanged (`wrapping_sub`, `wrapping_mul` in i32 — the wrap is
/// load-bearing and matches C's `int` multiply, see KB-ARM-FLOAT root #3 — then
/// sign-extended to i64). Vectorizing REASSOCIATES the two sums, and that is
/// exact here rather than merely close: i64 addition is associative, and the
/// accumulators cannot overflow (|diff * diff| <= 2^31 over at most 4096
/// coefficients is under 2^43).
///
/// `dqcoeff` is sliced to `coeff.len()` so a short `dqcoeff` panics exactly
/// where the indexed form panicked.
#[autoversion]
pub fn block_error_simd(coeff: &[i32], dqcoeff: &[i32]) -> (i64, i64) {
    let n = coeff.len();
    let dq = &dqcoeff[..n];
    let mut error = 0i64;
    let mut sqcoeff = 0i64;
    for (&c, &d) in coeff.iter().zip(dq.iter()) {
        let diff = c.wrapping_sub(d);
        error += diff.wrapping_mul(diff) as i64;
        sqcoeff += c.wrapping_mul(c) as i64;
    }
    (error, sqcoeff)
}

/// `aom_sum_squares_2d_i16` — the residual energy over a `width x height`
/// block with row stride `src_stride`. Env pin, then `incant!` dispatch.
///
/// KB-PERF-46: the scalar body carried **one bounds check per element** (a
/// runtime slice length indexed by `base + c`) plus a serial `u64` accumulator,
/// which together blocked vectorization of a trivially reducible sum. It is
/// live inside `search_tx_type_intra_into` — `perf annotate` on that symbol
/// shows its `movswl` / `imull` chain with two `cmp`s around it — and it is the
/// sibling KB-PERF-37 named and did not take.
///
/// # Bit-exactness
///
/// Scalar/neon/wasm tiers are identical to [`crate::dist::sum_squares_2d_i16`]
/// (exact `u64` accumulation — `2^30` over at most `128 * 128` elements is
/// under `2^44`). The v3 tier instead mirrors the kernel libaom actually
/// dispatches to on x86-64, `aom_sum_squares_2d_i16_avx2`, whose `madd_epi16` /
/// `add_epi32` chains accumulate in **wrapping i32** — identical to the exact
/// sum on every reachable input (residuals are `|v| <= 4095` at bd12), and
/// bug-compatible with real `aomenc` even where it does wrap (`i16::MIN`
/// pairs). See `_v3` for the per-shape semantics.
pub fn sum_squares_2d_i16_simd(src: &[i16], src_stride: usize, width: usize, height: usize) -> u64 {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        sum_squares_2d_i16_impl(src, src_stride, width, height),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed port, verbatim.
fn sum_squares_2d_i16_impl_scalar(
    _t: archmage::ScalarToken,
    src: &[i16],
    src_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let mut ss = 0u64;
    let mut off = 0usize;
    for _ in 0..height {
        let row = &src[off..off + width];
        for &v in row.iter() {
            let v = v as i32;
            ss += (v * v) as u64;
        }
        off += src_stride;
    }
    ss
}

/// Non-x86 tiers: the same row-sliced loop — the per-tier target features let
/// LLVM emit `pmaddwd`/`pmull` equivalents just as `#[autoversion]` did.
#[magetypes(neon, wasm128, -scalar)]
fn sum_squares_2d_i16_impl(
    _t: Token,
    src: &[i16],
    src_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let mut ss = 0u64;
    let mut off = 0usize;
    for _ in 0..height {
        let row = &src[off..off + width];
        for &v in row.iter() {
            let v = v as i32;
            ss += (v * v) as u64;
        }
        off += src_stride;
    }
    ss
}

/// v3 mirror of `aom_sum_squares_2d_i16_avx2` (sum_squares_avx2.c + the SSE2
/// arms it dispatches to). C's own shape dispatch is replicated — 4x4, 4xn,
/// 8-column nxn_sse2, 16-column nxn_avx2, else the C-scalar result — because
/// each arm accumulates differently: `madd_epi16` pair-products summed in
/// **wrapping i32** (per 4-row group for the nxn arms, across the whole block
/// for 4xn, sign-extended for 4x4) and only then zero-extended into `u64`.
/// Replicating the wrap is what real `aomenc` produces on the `i16::MIN`
/// adversarial domain; the differential oracle is therefore the exported
/// `aom_sum_squares_2d_i16_avx2` symbol itself.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn sum_squares_2d_i16_impl_v3(
    _t: archmage::X64V3Token,
    src: &[i16],
    src_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    use archmage::intrinsics::x86_64::*;

    // Contract preflight: degenerate dims or a short `src` take the scalar
    // path, which returns 0 / panics on the same inputs — and the chunked
    // views below are then guaranteed total.
    if height == 0
        || width == 0
        || src.len() < (height - 1).wrapping_mul(src_stride) + width
    {
        return crate::dist::sum_squares_2d_i16_scalar_ref(src, src_stride, width, height);
    }

    // zext pair mask: `_mm_set1_epi64x(~0u)` == 0x0000_0000_FFFF_FFFF.
    const MASK64: i64 = 0xFFFF_FFFF;

    if width == 4 && height == 4 {
        // aom_sum_squares_2d_i16_4x4_sse2 — loadl/loadh row pairs, i32 lanes,
        // result sign-extended from i32 (a wrapped-negative total becomes a
        // huge u64, exactly as C produces).
        let l = |o: usize| -> __m128i {
            let a: &[i16; 4] = src[o..o + 4].try_into().unwrap();
            _mm_loadu_si64(a)
        };
        let v01 = _mm_unpacklo_epi64(l(0), l(src_stride));
        let v23 = _mm_unpacklo_epi64(l(2 * src_stride), l(3 * src_stride));
        let s = _mm_add_epi32(_mm_madd_epi16(v01, v01), _mm_madd_epi16(v23, v23));
        let v = _mm_add_epi32(s, _mm_srli_epi64::<32>(s));
        let v = _mm_add_epi32(v, _mm_srli_si128::<8>(v));
        return _mm_cvtsi128_si32(v) as i64 as u64;
    }

    if width == 4 && height % 4 == 0 {
        // aom_sum_squares_2d_i16_4xn_sse2 — v_acc_q is named "q" but the op is
        // add_epi32: i32 wrapping across ALL 4-row groups, one zext fold at
        // the end.
        let l = |o: usize| -> __m128i {
            let a: &[i16; 4] = src[o..o + 4].try_into().unwrap();
            _mm_loadu_si64(a)
        };
        let mut acc = _mm_setzero_si128();
        let mut off = 0usize;
        for _ in 0..height / 4 {
            let v01 = _mm_unpacklo_epi64(l(off), l(off + src_stride));
            let v23 = _mm_unpacklo_epi64(l(off + 2 * src_stride), l(off + 3 * src_stride));
            let s = _mm_add_epi32(_mm_madd_epi16(v01, v01), _mm_madd_epi16(v23, v23));
            acc = _mm_add_epi32(acc, s);
            off += 4 * src_stride;
        }
        let a64 = _mm_add_epi64(
            _mm_and_si128(acc, _mm_set1_epi64x(MASK64)),
            _mm_srli_epi64::<32>(acc),
        );
        let a64 = _mm_add_epi64(a64, _mm_srli_si128::<8>(a64));
        return _mm_cvtsi128_si64(a64) as u64;
    }

    if width == 8 && height % 4 == 0 {
        // aom_sum_squares_2d_i16_nxn_sse2 at width 8 — i32 accumulate inside
        // each 4-row group, zext into i64 per group. C's row loads are
        // ALIGNED (`xx_load_128` == movdqa): real callers hand it a
        // 16-byte-aligned buffer with stride % 8 == 0. Anything else takes
        // the scalar path — the C dispatcher itself would fault there.
        if (src.as_ptr() as usize) % 16 != 0 || src_stride % 8 != 0 {
            return crate::dist::sum_squares_2d_i16_scalar_ref(src, src_stride, width, height);
        }
        let mask = _mm_set1_epi64x(MASK64);
        let mut acc_q = _mm_setzero_si128();
        let mut off = 0usize;
        for _ in 0..height / 4 {
            let m = |o: usize| -> __m128i {
                let a: &[i16; 8] = src[o..o + 8].try_into().unwrap();
                let v = _mm_loadu_si128(a);
                _mm_madd_epi16(v, v)
            };
            let d = _mm_add_epi32(
                _mm_add_epi32(m(off), m(off + src_stride)),
                _mm_add_epi32(m(off + 2 * src_stride), m(off + 3 * src_stride)),
            );
            acc_q = _mm_add_epi64(acc_q, _mm_and_si128(d, mask));
            acc_q = _mm_add_epi64(acc_q, _mm_srli_epi64::<32>(d));
            off += 4 * src_stride;
        }
        let acc_q = _mm_add_epi64(acc_q, _mm_srli_si128::<8>(acc_q));
        return _mm_cvtsi128_si64(acc_q) as u64;
    }

    if width % 16 == 0 && height % 4 == 0 {
        // aom_sum_squares_2d_i16_nxn_avx2 — 16 columns per iteration, i32
        // accumulate within a 4-row group, zext into i64 per group. Row
        // slices chunked to &[i16; 16] keep every load check-free.
        let mask = _mm256_set1_epi64x(MASK64);
        let mut acc_q = _mm256_setzero_si256();
        let mut off = 0usize;
        for _ in 0..height / 4 {
            let (c0, _) = src[off..off + width].as_chunks::<16>();
            let (c1, _) = src[off + src_stride..off + src_stride + width].as_chunks::<16>();
            let (c2, _) = src[off + 2 * src_stride..off + 2 * src_stride + width]
                .as_chunks::<16>();
            let (c3, _) = src[off + 3 * src_stride..off + 3 * src_stride + width]
                .as_chunks::<16>();
            let mut acc_d = _mm256_setzero_si256();
            let sq = |a: &[i16; 16]| -> __m256i {
                let v = _mm256_loadu_si256(a);
                _mm256_madd_epi16(v, v)
            };
            for (((a, b), c), d) in c0
                .iter()
                .zip(c1.iter())
                .zip(c2.iter())
                .zip(c3.iter())
            {
                let s01 = _mm256_add_epi32(sq(a), sq(b));
                let s23 = _mm256_add_epi32(sq(c), sq(d));
                acc_d = _mm256_add_epi32(acc_d, _mm256_add_epi32(s01, s23));
            }
            acc_q = _mm256_add_epi64(acc_q, _mm256_and_si256(acc_d, mask));
            acc_q = _mm256_add_epi64(acc_q, _mm256_srli_epi64::<32>(acc_d));
            off += 4 * src_stride;
        }
        let r = _mm_add_epi64(
            _mm256_castsi256_si128(acc_q),
            _mm256_extracti128_si256::<1>(acc_q),
        );
        let r = _mm_add_epi64(r, _mm_unpackhi_epi64(r, r));
        return _mm_cvtsi128_si64(r) as u64;
    }

    // Shapes C's dispatcher sends to aom_sum_squares_2d_i16_c — the exact-u64
    // scalar loop IS C-c's output, bit-identically.
    crate::dist::sum_squares_2d_i16_scalar_ref(src, src_stride, width, height)
}

/// `av1_block_error_lp_c` (`av1/encoder/rdopt.c:907`) — the lp-arm
/// transform-domain distortion. Note the per-lane arithmetic is C `int`
/// (i32) throughout: `diff` and `diff * diff` both wrap modulo 2^32 before
/// the i64 accumulate (the wrap needs |diff| > 46340 — opposite-sign lanes,
/// which quantize_lp never emits: dqcoeff carries coeff's sign on every
/// reachable input; `nonrd_block_yrd_lp_diff` measures the reachable
/// |dq - c| bound at 2308).
pub fn block_error_lp(coeff: &[i16], dqcoeff: &[i16], block_size: usize) -> i64 {
    let mut error: i64 = 0;
    for i in 0..block_size {
        let diff = i32::from(coeff[i]) - i32::from(dqcoeff[i]);
        error += diff.wrapping_mul(diff) as i64;
    }
    error
}

/// `av1_block_error_lp` with runtime SIMD dispatch. The v3 tier is an
/// instruction-level mirror of `av1_block_error_lp_avx2`
/// (`av1/encoder/x86/error_intrin_avx2.c:128`) — the kernel RTCD actually
/// dispatches on x86-64 — including its three shape arms (n==16 `hadd`,
/// n==32, and the 64-wide loop) and their i32-wrap-then-zero-extend
/// accumulation semantics. Every wrap class is unreachable on real
/// (coeff, dqcoeff) pairs — measured, see [`block_error_lp`].
///
/// C asserts `block_size % 16 == 0` and would READ PAST the buffers for a
/// %16-only size that misses its named arms; the port instead falls back to
/// the scalar transcription there (call sites pass 16/64/256).
pub fn block_error_lp_simd(coeff: &[i16], dqcoeff: &[i16], block_size: usize) -> i64 {
    let _ = crate::dispatch::scalar_forced();
    incant!(
        block_error_lp_impl(coeff, dqcoeff, block_size),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed port, verbatim.
fn block_error_lp_impl_scalar(
    _t: archmage::ScalarToken,
    coeff: &[i16],
    dqcoeff: &[i16],
    block_size: usize,
) -> i64 {
    block_error_lp(coeff, dqcoeff, block_size)
}

/// Non-x86 tiers: the same i32-diff / i64-accumulate loop — under the tier's
/// target features LLVM lowers it to the widening-multiply shape the C NEON
/// kernel (`block_error_neon`) uses. Exact-equal to `_c` on every input.
#[magetypes(neon, wasm128, -scalar)]
fn block_error_lp_impl(_t: Token, coeff: &[i16], dqcoeff: &[i16], block_size: usize) -> i64 {
    block_error_lp(coeff, dqcoeff, block_size)
}

/// v3 mirror of `av1_block_error_lp_avx2` (error_intrin_avx2.c). The three
/// named arms are replicated — `block_size16`'s `hadd_epi32` pair tree, the
/// `block_size32` two-madd fold, and the 64-wide loop — because each wraps
/// its i32 intermediate differently. All wraps are unreachable in production
/// (see [`block_error_lp`]); they are mirrored anyway so the tier is
/// bug-compatible with the dispatched C kernel on the FULL i16 domain.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn block_error_lp_impl_v3(
    _t: archmage::X64V3Token,
    coeff: &[i16],
    dqcoeff: &[i16],
    block_size: usize,
) -> i64 {
    use archmage::intrinsics::x86_64::*;

    let n = block_size;
    if n == 0 || coeff.len() < n || dqcoeff.len() < n {
        return block_error_lp(coeff, dqcoeff, block_size);
    }
    let zero = _mm256_setzero_si256();
    let mut sse = zero;

    if n == 16 {
        // av1_block_error_block_size16_avx2: one 16-lane madd, then the
        // hadd_epi32 pair tree and a single zero-extend.
        let c: &[i16; 16] = coeff[..16].try_into().unwrap();
        let d: &[i16; 16] = dqcoeff[..16].try_into().unwrap();
        let diff = _mm256_sub_epi16(_mm256_loadu_si256(d), _mm256_loadu_si256(c));
        let error = _mm256_madd_epi16(diff, diff);
        let error_hi = _mm256_hadd_epi32(error, error);
        sse = _mm256_unpacklo_epi32(error_hi, zero);
    } else if n == 32 {
        // av1_block_error_block_size32_avx2: two madds summed in i32
        // (wrapping), then zero-extended into i64 lanes.
        let c0: &[i16; 16] = coeff[..16].try_into().unwrap();
        let d0: &[i16; 16] = dqcoeff[..16].try_into().unwrap();
        let c1: &[i16; 16] = coeff[16..32].try_into().unwrap();
        let d1: &[i16; 16] = dqcoeff[16..32].try_into().unwrap();
        let diff0 = _mm256_sub_epi16(_mm256_loadu_si256(d0), _mm256_loadu_si256(c0));
        let diff1 = _mm256_sub_epi16(_mm256_loadu_si256(d1), _mm256_loadu_si256(c1));
        let err = _mm256_add_epi32(
            _mm256_madd_epi16(diff0, diff0),
            _mm256_madd_epi16(diff1, diff1),
        );
        sse = _mm256_add_epi64(
            sse,
            _mm256_add_epi64(
                _mm256_unpacklo_epi32(err, zero),
                _mm256_unpackhi_epi32(err, zero),
            ),
        );
    } else {
        // av1_block_error_block_size64_avx2: per 64-lane group, four madds
        // folded pairwise in i32 (wrapping), zero-extended into i64. C's
        // loop condition is `i < block_size` stepping 64 — a size not
        // divisible by 64 would overread there; the port declines instead.
        if n % 64 != 0 {
            return block_error_lp(coeff, dqcoeff, block_size);
        }
        let c32 = coeff[..n].as_chunks::<16>().0;
        let d32 = dqcoeff[..n].as_chunks::<16>().0;
        for i in (0..n / 16).step_by(4) {
            let diff = |k: usize| {
                _mm256_sub_epi16(
                    _mm256_loadu_si256(&d32[i + k]),
                    _mm256_loadu_si256(&c32[i + k]),
                )
            };
            let e01 = _mm256_add_epi32(
                _mm256_madd_epi16(diff(0), diff(0)),
                _mm256_madd_epi16(diff(1), diff(1)),
            );
            let e23 = _mm256_add_epi32(
                _mm256_madd_epi16(diff(2), diff(2)),
                _mm256_madd_epi16(diff(3), diff(3)),
            );
            let s01 = _mm256_add_epi64(
                _mm256_unpacklo_epi32(e01, zero),
                _mm256_unpackhi_epi32(e01, zero),
            );
            let s23 = _mm256_add_epi64(
                _mm256_unpacklo_epi32(e23, zero),
                _mm256_unpackhi_epi32(e23, zero),
            );
            sse = _mm256_add_epi64(sse, _mm256_add_epi64(s01, s23));
        }
    }

    // The caller's epilogue: fold each 128-bit half's high i64 into the low,
    // then add the halves.
    let sse_hi = _mm256_srli_si256::<8>(sse);
    let s = _mm256_add_epi64(sse, sse_hi);
    let s128 = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
    _mm_cvtsi128_si64(s128)
}
