//! The SSE2/AVX2 intrinsic vocabulary on aarch64 NEON — `x86ish` for short.
//!
//! The fused whole-block transform kernels in [`crate::transform::simd`] are verbatim
//! transcriptions of libaom's SSE2/AVX2 shapes (`av1_lowbd_fwd_txfm2d_*_sse2`,
//! `transpose_16bit_*_avx2`, `btf_16_sse2` …): they speak raw `__m128i` /
//! `__m256i` intrinsics, not the magetypes generic API. On x86-64 that
//! vocabulary is `archmage::intrinsics::x86_64`. On aarch64 the same bodies
//! compile per-tier — the `#[magetypes]` contract — only if every name they
//! use exists, which is what this module provides: one `use crate::sse_neon::*`
//! resolves to the real archmage module on x86-64 and to the NEON twins below
//! on aarch64, so each body stays a single transcription.
//!
//! # Representation choices
//!
//! * `__m128i` is `int32x4_t` — the opaque 128-bit integer container. Element
//!   width is per-op (`epi16` ops reinterpret to `int16x8_t` and back), exactly
//!   like x86 where `__m128i` is width-agnostic.
//! * `__m256i` is `[int32x4_t; 2]` — deliberately identical to
//!   `i32x8<NeonToken>`'s `Repr`, so the bodies' `i32x8::from_repr(t, …)` /
//!   `.into_repr()` sites typecheck unchanged. The ONE body that instead holds
//!   `i16x16` lanes (`fwd_16x16_fused_i16_w16`, whose Repr is
//!   `[int16x8_t; 2]`) crosses through [`i16x16_of_m256`] / [`m256_of_i16x16`],
//!   defined per-arch.
//! * AVX2's unpack/permute family operates per 128-bit lane, which is exactly
//!   "apply the NEON op to each half of the array" — the decomposition is
//!   structural, so exactness is by construction.
//!
//! # Exactness notes (each site also covered by the tier-permutation
//! differentials, which run every compiled tier against the scalar port)
//!
//! * `_mm_madd_epi16`: `vmull` + `vmull_high` + `vpaddq` — the i32 pairwise sum
//!   wraps identically on both ISAs (the `(-32768)^2 + (-32768)^2` vertex wraps
//!   to `i32::MIN` either way).
//! * `_mm_mulhrs_epi16`: `vqrdmulhq_s16` is *defined* as
//!   `sat(round(2ab / 2^16))` = `sat((ab + 2^14) >> 15)` — SSE2's `pmulhrsw`
//!   bit-for-bit, including the `a = b = -32768` saturation.
//! * `_mm_packs_epi32`: `vqmovn_s32` is the saturating signed narrow, the same
//!   edge semantics as `packssdw`.
//! * `_mm_sra_epi32` (runtime count): NEON has no right-shift instruction —
//!   `vshlq_s32` with a negative count IS the arithmetic right shift, and both
//!   ISAs saturate an out-of-range count to sign-fill.
//! * `_mm_testz_si128` / `_mm256_testz_si256`: `vmaxvq_u32(and) == 0` is
//!   `(a & b) == 0` exactly.
//! * `_mm_srli_si128::<N>`: `vextq_s8::<N>(v, zero)` emits `[v[N..16], 0…]` —
//!   the x86 byte-shift semantics for N < 16; N >= 16 yields all zeros on both.
//! * `_mm_shuffle*`: `vqtbl1q_u8` with an index vector computed from the
//!   immediate — arbitrary lane permutation, folded to a literal pool load
//!   plus one `tbl` at -O.
//! * Load/store twins are generic over per-array-type traits (`Ld128` etc.)
//!   mirroring `safe_unaligned_simd`'s `Is*BitsUnaligned` bounds — the calls
//!   sites pass `&[i32; 4]`, `&[i16; 8]`, `&mut [u16; 8]` &c. and the trait
//!   resolves per concrete type. All loads are unaligned-safe (`vld1`), as on
//!   x86.
//!
//! `forbid(unsafe_code)` holds: every body here is safe code —
//! `safe_unaligned_simd::aarch64` supplies the reference-based loads/stores
//! (re-exported through `archmage::intrinsics::aarch64`) and the NEON value
//! intrinsics are safe on the baseline-aarch64 target.

#[cfg(target_arch = "x86_64")]
mod imp {
    pub(crate) use archmage::intrinsics::x86_64::*;
    use archmage::prelude::X64V3Token;
    use magetypes::simd::generic::i16x16;

    /// `i16x16`'s Repr IS `__m256i` on this target — the conversions are
    /// `from_repr`/`into_repr` directly.
    #[inline(always)]
    pub(crate) fn i16x16_of_m256(t: X64V3Token, m: __m256i) -> i16x16<X64V3Token> {
        i16x16::from_repr(t, m)
    }

    #[inline(always)]
    pub(crate) fn m256_of_i16x16(v: i16x16<X64V3Token>) -> __m256i {
        v.into_repr()
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    pub(crate) use archmage::intrinsics::aarch64::*;
    use archmage::prelude::NeonToken;
    use magetypes::simd::generic::i16x16;

    /// SSE2 `__m128i` — the width-agnostic 128-bit integer container.
    #[allow(non_camel_case_types)]
    pub type __m128i = int32x4_t;
    /// AVX2 `__m256i` — two 128-bit halves; identical to `i32x8<NeonToken>`'s
    /// `Repr` so the magetypes boundary crossings typecheck unchanged.
    #[allow(non_camel_case_types)]
    pub type __m256i = [int32x4_t; 2];

    // ---- reinterpret helpers -------------------------------------------------
    #[inline]
    #[target_feature(enable = "neon")]
    fn s16(v: int32x4_t) -> int16x8_t {
        vreinterpretq_s16_s32(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn u16v(v: int32x4_t) -> uint16x8_t {
        vreinterpretq_u16_s32(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn u32v(v: int32x4_t) -> uint32x4_t {
        vreinterpretq_u32_s32(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn s64v(v: int32x4_t) -> int64x2_t {
        vreinterpretq_s64_s32(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn u8v(v: int32x4_t) -> uint8x16_t {
        vreinterpretq_u8_s32(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn back16(v: int16x8_t) -> int32x4_t {
        vreinterpretq_s32_s16(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn back64(v: int64x2_t) -> int32x4_t {
        vreinterpretq_s32_s64(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn backu8(v: uint8x16_t) -> int32x4_t {
        vreinterpretq_s32_u8(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn backu16(v: uint16x8_t) -> int32x4_t {
        vreinterpretq_s32_u16(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn backu32(v: uint32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(v)
    }

    // ---- loads / stores ----------------------------------------------------
    //
    // `#[target_feature]` cannot sit on a trait method, so the NEON op lives
    // in the `#[target_feature]` shim fn and the per-type work is a pure-safe
    // array <-> bytes copy through `Ld*`/`St*`. `to_ne_bytes`/`from_ne_bytes`
    // on adjacent lanes is a byte-exact copy — LLVM folds each pair into the
    // same single `vld1`/`vst1` the x86 twin emits.

    pub trait Ld128: Sized {
        fn rd128(p: &Self) -> [u8; 16];
    }
    macro_rules! ld {
        ($t:ty, $n:literal) => {
            impl Ld128 for [$t; $n] {
                #[inline]
                fn rd128(p: &Self) -> [u8; 16] {
                    p.map(<$t>::to_ne_bytes).as_flattened().try_into().unwrap()
                }
            }
        };
    }
    ld!(i32, 4);
    ld!(u32, 4);
    ld!(i16, 8);
    ld!(u16, 8);
    ld!(i64, 2);
    ld!(u64, 2);
    impl Ld128 for [u8; 16] {
        #[inline]
        fn rd128(p: &Self) -> [u8; 16] {
            *p
        }
    }
    impl Ld128 for [i8; 16] {
        #[inline]
        fn rd128(p: &Self) -> [u8; 16] {
            p.map(|v| v as u8)
        }
    }
    /// `_mm_loadu_si128` — unaligned 16-byte load.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_loadu_si128<T: Ld128>(p: &T) -> __m128i {
        backu8(vld1q_u8(&T::rd128(p)))
    }

    pub trait St128: Sized {
        fn wr128(p: &mut Self, b: [u8; 16]);
    }
    macro_rules! st {
        ($t:ty, $n:literal) => {
            impl St128 for [$t; $n] {
                #[inline]
                fn wr128(p: &mut Self, b: [u8; 16]) {
                    for (v, c) in p.iter_mut().zip(b.chunks_exact(<$t>::BITS as usize / 8)) {
                        *v = <$t>::from_ne_bytes(c.try_into().unwrap());
                    }
                }
            }
        };
    }
    st!(i32, 4);
    st!(u32, 4);
    st!(i16, 8);
    st!(u16, 8);
    st!(i64, 2);
    st!(u64, 2);
    impl St128 for [u8; 16] {
        #[inline]
        fn wr128(p: &mut Self, b: [u8; 16]) {
            *p = b;
        }
    }
    impl St128 for [i8; 16] {
        #[inline]
        fn wr128(p: &mut Self, b: [u8; 16]) {
            *p = b.map(|v| v as i8);
        }
    }
    /// `_mm_storeu_si128` — unaligned 16-byte store.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_storeu_si128<T: St128>(p: &mut T, v: __m128i) {
        let mut b = [0u8; 16];
        vst1q_u8(&mut b, u8v(v));
        T::wr128(p, b)
    }

    pub trait Ld64: Sized {
        fn rd64(p: &Self) -> [u8; 8];
    }
    macro_rules! ld64 {
        ($t:ty, $n:literal) => {
            impl Ld64 for [$t; $n] {
                #[inline]
                fn rd64(p: &Self) -> [u8; 8] {
                    p.map(<$t>::to_ne_bytes).as_flattened().try_into().unwrap()
                }
            }
        };
    }
    ld64!(i16, 4);
    ld64!(u16, 4);
    ld64!(i32, 2);
    ld64!(u32, 2);
    impl Ld64 for [u8; 8] {
        #[inline]
        fn rd64(p: &Self) -> [u8; 8] {
            *p
        }
    }
    impl Ld64 for [i8; 8] {
        #[inline]
        fn rd64(p: &Self) -> [u8; 8] {
            p.map(|v| v as u8)
        }
    }
    impl Ld64 for [i64; 1] {
        #[inline]
        fn rd64(p: &Self) -> [u8; 8] {
            p[0].to_ne_bytes()
        }
    }
    /// `_mm_loadu_si64` — unaligned 8-byte load, zero-extended.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_loadu_si64<T: Ld64>(p: &T) -> __m128i {
        backu8(vcombine_u8(vld1_u8(&T::rd64(p)), vdup_n_u8(0)))
    }

    pub trait St64: Sized {
        fn wr64(p: &mut Self, b: [u8; 8]);
    }
    macro_rules! st64 {
        ($t:ty, $n:literal) => {
            impl St64 for [$t; $n] {
                #[inline]
                fn wr64(p: &mut Self, b: [u8; 8]) {
                    for (v, c) in p.iter_mut().zip(b.chunks_exact(<$t>::BITS as usize / 8)) {
                        *v = <$t>::from_ne_bytes(c.try_into().unwrap());
                    }
                }
            }
        };
    }
    st64!(u16, 4);
    st64!(i16, 4);
    st64!(i32, 2);
    st64!(u32, 2);
    impl St64 for [u8; 8] {
        #[inline]
        fn wr64(p: &mut Self, b: [u8; 8]) {
            *p = b;
        }
    }
    /// `_mm_storeu_si64` — unaligned 8-byte store of the low half.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_storeu_si64<T: St64>(p: &mut T, v: __m128i) {
        let mut b = [0u8; 8];
        vst1_u8(&mut b, vget_low_u8(u8v(v)));
        T::wr64(p, b)
    }

    pub trait Ld256: Sized {
        fn rd256(p: &Self) -> [u8; 32];
    }
    macro_rules! ld256 {
        ($t:ty, $n:literal) => {
            impl Ld256 for [$t; $n] {
                #[inline]
                fn rd256(p: &Self) -> [u8; 32] {
                    p.map(<$t>::to_ne_bytes).as_flattened().try_into().unwrap()
                }
            }
        };
    }
    ld256!(i32, 8);
    ld256!(u32, 8);
    ld256!(i16, 16);
    ld256!(u16, 16);
    /// `_mm256_loadu_si256` — unaligned 32-byte load.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_loadu_si256<T: Ld256>(p: &T) -> __m256i {
        let b = T::rd256(p);
        [
            backu8(vld1q_u8(<&[u8; 16]>::try_from(&b[..16]).unwrap())),
            backu8(vld1q_u8(<&[u8; 16]>::try_from(&b[16..]).unwrap())),
        ]
    }

    // ---- scalar / splat / move ----------------------------------------------

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_setzero_si128() -> __m128i {
        vdupq_n_s32(0)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_setzero_si256() -> __m256i {
        [vdupq_n_s32(0), vdupq_n_s32(0)]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_set1_epi16(x: i16) -> __m128i {
        back16(vdupq_n_s16(x))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_set1_epi16(x: i16) -> __m256i {
        let v = back16(vdupq_n_s16(x));
        [v, v]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_set1_epi32(x: i32) -> __m128i {
        vdupq_n_s32(x)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_cvtsi32_si128(x: i32) -> __m128i {
        vsetq_lane_s32::<0>(x, vdupq_n_s32(0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_cvtsi128_si32(v: __m128i) -> i32 {
        vgetq_lane_s32::<0>(v)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_castsi256_si128(v: __m256i) -> __m128i {
        v[0]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_extracti128_si256<const IMM: i32>(v: __m256i) -> __m128i {
        v[(IMM & 1) as usize]
    }

    // ---- 128-bit arithmetic --------------------------------------------------

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vaddq_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_sub_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vsubq_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vqaddq_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_subs_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vqsubq_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i {
        backu16(vqsubq_u16(u16v(a), u16v(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i {
        vaddq_s32(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_sub_epi32(a: __m128i, b: __m128i) -> __m128i {
        vsubq_s32(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_abs_epi16(a: __m128i) -> __m128i {
        back16(vabsq_s16(s16(a)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_abs_epi32(a: __m128i) -> __m128i {
        vabsq_s32(a)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vminq_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vmaxq_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i {
        backu16(vmaxq_u16(u16v(a), u16v(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_max_epu32(a: __m128i, b: __m128i) -> __m128i {
        backu32(vmaxq_u32(u32v(a), u32v(b)))
    }
    /// `pmaddwd` — signed i16 products summed pairwise to i32. `vmull` +
    /// `vmull_high` cover lanes 0..4 / 4..8; `vpaddq` folds the pairs. The
    /// i32 sum wraps identically to SSE2.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_madd_epi16(a: __m128i, b: __m128i) -> __m128i {
        let (a16, b16) = (s16(a), s16(b));
        vpaddq_s32(
            vmull_s16(vget_low_s16(a16), vget_low_s16(b16)),
            vmull_high_s16(a16, b16),
        )
    }
    /// `pmulhrsw` — `vqrdmulhq_s16` is defined as `sat(round(2ab / 2^16))`,
    /// which is exactly `sat((ab + 0x4000) >> 15)`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vqrdmulhq_s16(s16(a), s16(b)))
    }
    /// `packssdw` — saturating signed i32 -> i16 narrow, `a` into the low half.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_packs_epi32(a: __m128i, b: __m128i) -> __m128i {
        back16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)))
    }
    /// `pcmpgtd` — signed i32 `a > b` -> all-ones mask; `vcgtq_s32` is the same
    /// signed comparison producing the same all-ones lane.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) -> __m128i {
        backu32(vcgtq_s32(a, b))
    }

    // ---- shifts ---------------------------------------------------------------

    /// `psrad` with a runtime count held in `count`'s low i64 — NEON's
    /// `vshlq_s32` with a negative count is the same arithmetic right shift;
    /// out-of-range counts saturate to sign-fill on both ISAs.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_sra_epi32(a: __m128i, count: __m128i) -> __m128i {
        let n = vgetq_lane_s64::<0>(s64v(count)) as i32;
        vshlq_s32(a, vdupq_n_s32(-n))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_srai_epi32<const IMM: i32>(a: __m128i) -> __m128i {
        // x86 imm > 31 -> sign-fill; vshrq_n_s32 saturates identically.
        if IMM <= 0 {
            a
        } else if IMM >= 32 {
            vshrq_n_s32::<31>(vshrq_n_s32::<1>(a))
        } else {
            vshrq_n_s32::<IMM>(a)
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_srai_epi16<const IMM: i32>(a: __m128i) -> __m128i {
        if IMM <= 0 {
            a
        } else if IMM >= 16 {
            back16(vshrq_n_s16::<15>(vshrq_n_s16::<1>(s16(a))))
        } else {
            back16(vshrq_n_s16::<IMM>(s16(a)))
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_slli_epi16<const IMM: i32>(a: __m128i) -> __m128i {
        if IMM <= 0 {
            a
        } else if IMM >= 16 {
            vdupq_n_s32(0)
        } else {
            back16(vshlq_n_s16::<IMM>(s16(a)))
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_slli_epi32<const IMM: i32>(a: __m128i) -> __m128i {
        if IMM <= 0 {
            a
        } else if IMM >= 32 {
            vdupq_n_s32(0)
        } else {
            vshlq_n_s32::<IMM>(a)
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_slli_epi16<const IMM: i32>(a: __m256i) -> __m256i {
        [_mm_slli_epi16::<IMM>(a[0]), _mm_slli_epi16::<IMM>(a[1])]
    }
    /// `psrldq` — byte-granular right shift. `vextq_s8::<N>(v, zero)` is
    /// `[v[N..16], 0…]`; N >= 16 yields all zeros (x86 clamps the imm the
    /// same way).
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_srli_si128<const IMM: i32>(a: __m128i) -> __m128i {
        let a8 = vreinterpretq_s8_s32(a);
        let z = vdupq_n_s8(0);
        let r = match IMM {
            0 => return a,
            1 => vextq_s8::<1>(a8, z),
            2 => vextq_s8::<2>(a8, z),
            3 => vextq_s8::<3>(a8, z),
            4 => vextq_s8::<4>(a8, z),
            5 => vextq_s8::<5>(a8, z),
            6 => vextq_s8::<6>(a8, z),
            7 => vextq_s8::<7>(a8, z),
            8 => vextq_s8::<8>(a8, z),
            9 => vextq_s8::<9>(a8, z),
            10 => vextq_s8::<10>(a8, z),
            11 => vextq_s8::<11>(a8, z),
            12 => vextq_s8::<12>(a8, z),
            13 => vextq_s8::<13>(a8, z),
            14 => vextq_s8::<14>(a8, z),
            15 => vextq_s8::<15>(a8, z),
            _ => return vdupq_n_s32(0),
        };
        vreinterpretq_s32_s8(r)
    }

    // ---- interleave / permute -------------------------------------------------

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_unpacklo_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vzip1q_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_unpackhi_epi16(a: __m128i, b: __m128i) -> __m128i {
        back16(vzip2q_s16(s16(a), s16(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_unpacklo_epi32(a: __m128i, b: __m128i) -> __m128i {
        vzip1q_s32(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_unpackhi_epi32(a: __m128i, b: __m128i) -> __m128i {
        vzip2q_s32(a, b)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_unpacklo_epi64(a: __m128i, b: __m128i) -> __m128i {
        back64(vzip1q_s64(s64v(a), s64v(b)))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_unpackhi_epi64(a: __m128i, b: __m128i) -> __m128i {
        back64(vzip2q_s64(s64v(a), s64v(b)))
    }

    /// Byte-table index for a 16-bit lane shuffle: `out lane i` takes input
    /// lane `sel(i)` where `sel(i)` is `IMM`'s 2-bit field for the permuted
    /// half and the identity for the untouched half.
    #[inline]
    #[target_feature(enable = "neon")]
    fn tbl_idx_epi16<const IMM: i32, const HI: bool>() -> uint8x16_t {
        let mut idx = [0u8; 16];
        let mut i = 0;
        while i < 8 {
            let permuted = (i >= 4) == HI;
            let sel = if permuted {
                ((IMM >> (2 * (i & 3))) & 3) as usize + if HI { 4 } else { 0 }
            } else {
                i
            };
            idx[2 * i] = (2 * sel) as u8;
            idx[2 * i + 1] = (2 * sel + 1) as u8;
            i += 1;
        }
        vld1q_u8(&idx)
    }
    /// `pshuflw` — permute the LOW four i16 lanes by `IMM`'s 2-bit fields.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_shufflelo_epi16<const IMM: i32>(a: __m128i) -> __m128i {
        backu8(vqtbl1q_u8(u8v(a), tbl_idx_epi16::<IMM, false>()))
    }
    /// `pshufhw` — permute the HIGH four i16 lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_shufflehi_epi16<const IMM: i32>(a: __m128i) -> __m128i {
        backu8(vqtbl1q_u8(u8v(a), tbl_idx_epi16::<IMM, true>()))
    }
    /// `pshufd` — permute all four i32 lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_shuffle_epi32<const IMM: i32>(a: __m128i) -> __m128i {
        let mut idx = [0u8; 16];
        let mut i = 0;
        while i < 4 {
            let sel = ((IMM >> (2 * i)) & 3) as usize;
            idx[4 * i] = (4 * sel) as u8;
            idx[4 * i + 1] = (4 * sel + 1) as u8;
            idx[4 * i + 2] = (4 * sel + 2) as u8;
            idx[4 * i + 3] = (4 * sel + 3) as u8;
            i += 1;
        }
        backu8(vqtbl1q_u8(u8v(a), vld1q_u8(&idx)))
    }

    // ---- testz ----------------------------------------------------------------

    /// `ptest` — `(a & b) == 0`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_testz_si128(a: __m128i, b: __m128i) -> i32 {
        (vmaxvq_u32(vandq_u32(u32v(a), u32v(b))) == 0) as i32
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_testz_si256(a: __m256i, b: __m256i) -> i32 {
        let z0 = vmaxvq_u32(vandq_u32(u32v(a[0]), u32v(b[0]))) == 0;
        let z1 = vmaxvq_u32(vandq_u32(u32v(a[1]), u32v(b[1]))) == 0;
        (z0 && z1) as i32
    }

    /// `_mm_storeu_si256`-style unaligned 32-byte store.
    pub trait St256: Sized {
        fn wr256(p: &mut Self, b: [u8; 32]);
    }
    macro_rules! st256 {
        ($t:ty, $n:literal) => {
            impl St256 for [$t; $n] {
                #[inline]
                fn wr256(p: &mut Self, b: [u8; 32]) {
                    for (v, c) in p.iter_mut().zip(b.chunks_exact(<$t>::BITS as usize / 8)) {
                        *v = <$t>::from_ne_bytes(c.try_into().unwrap());
                    }
                }
            }
        };
    }
    st256!(i32, 8);
    st256!(u32, 8);
    st256!(i16, 16);
    st256!(u16, 16);
    /// `_mm256_storeu_si256` — unaligned 32-byte store.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_storeu_si256<T: St256>(p: &mut T, v: __m256i) {
        let mut b = [0u8; 32];
        vst1q_u8(<&mut [u8; 16]>::try_from(&mut b[..16]).unwrap(), u8v(v[0]));
        vst1q_u8(<&mut [u8; 16]>::try_from(&mut b[16..]).unwrap(), u8v(v[1]));
        T::wr256(p, b)
    }

    // ---- AVX2: per-128-lane ops -------------------------------------------------

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_unpacklo_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vzip1q_s16(s16(a[0]), s16(b[0]))),
            back16(vzip1q_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_unpackhi_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vzip2q_s16(s16(a[0]), s16(b[0]))),
            back16(vzip2q_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_unpacklo_epi32(a: __m256i, b: __m256i) -> __m256i {
        [vzip1q_s32(a[0], b[0]), vzip1q_s32(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_unpackhi_epi32(a: __m256i, b: __m256i) -> __m256i {
        [vzip2q_s32(a[0], b[0]), vzip2q_s32(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_unpacklo_epi64(a: __m256i, b: __m256i) -> __m256i {
        [
            back64(vzip1q_s64(s64v(a[0]), s64v(b[0]))),
            back64(vzip1q_s64(s64v(a[1]), s64v(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_unpackhi_epi64(a: __m256i, b: __m256i) -> __m256i {
        [
            back64(vzip2q_s64(s64v(a[0]), s64v(b[0]))),
            back64(vzip2q_s64(s64v(a[1]), s64v(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_abs_epi16(a: __m256i) -> __m256i {
        [_mm_abs_epi16(a[0]), _mm_abs_epi16(a[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_abs_epi32(a: __m256i) -> __m256i {
        [vabsq_s32(a[0]), vabsq_s32(a[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_max_epu16(a: __m256i, b: __m256i) -> __m256i {
        [_mm_max_epu16(a[0], b[0]), _mm_max_epu16(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_max_epu32(a: __m256i, b: __m256i) -> __m256i {
        [_mm_max_epu32(a[0], b[0]), _mm_max_epu32(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_subs_epu16(a: __m256i, b: __m256i) -> __m256i {
        [_mm_subs_epu16(a[0], b[0]), _mm_subs_epu16(a[1], b[1])]
    }
    /// `vperm2i128` — select each 128-bit result half from `a`'s or `b`'s
    /// halves by the immediate's 2-bit fields; the two ZERO bits force a
    /// zeroed half.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_permute2x128_si256<const IMM: i32>(a: __m256i, b: __m256i) -> __m256i {
        let pick = |sel: i32| match sel & 3 {
            0 => a[0],
            1 => a[1],
            2 => b[0],
            _ => b[1],
        };
        let z = vdupq_n_s32(0);
        [
            if IMM & 0x08 != 0 { z } else { pick(IMM) },
            if IMM & 0x80 != 0 { z } else { pick(IMM >> 4) },
        ]
    }
    /// `vpermq` — 64-bit lane permute across the full 256. Each result qword
    /// `i` takes source qword `IMM[2i+1:2i]` of `a` (qwords 0,1 = `a[0]`,
    /// 2,3 = `a[1]`); folds to lane selection at -O.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_permute4x64_epi64<const IMM: i32>(a: __m256i) -> __m256i {
        let pick = |sel: i32| match sel & 3 {
            0 => vgetq_lane_s64::<0>(s64v(a[0])),
            1 => vgetq_lane_s64::<1>(s64v(a[0])),
            2 => vgetq_lane_s64::<0>(s64v(a[1])),
            _ => vgetq_lane_s64::<1>(s64v(a[1])),
        };
        let lo = vsetq_lane_s64::<1>(pick((IMM >> 2) & 3), vsetq_lane_s64::<0>(pick(IMM & 3), vdupq_n_s64(0)));
        let hi = vsetq_lane_s64::<1>(pick((IMM >> 6) & 3), vsetq_lane_s64::<0>(pick((IMM >> 4) & 3), vdupq_n_s64(0)));
        [back64(lo), back64(hi)]
    }
    /// `vpsignw` — per lane: `b < 0` -> `-a`, `b == 0` -> 0, `b > 0` -> `a`.
    /// `vnegq` + two compares + `vbslq` selects; no NEON psign exists.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_sign_epi16(a: __m256i, b: __m256i) -> __m256i {
        let half = |x: int32x4_t, y: int32x4_t| -> int32x4_t {
            let (x16, y16) = (s16(x), s16(y));
            let neg = vnegq_s16(x16);
            let pos = vcgtq_s16(y16, vdupq_n_s16(0));
            let neq = vceqq_s16(y16, vdupq_n_s16(0));
            // pos -> x; !pos & zero -> 0; else -> -x
            let nonpos = vbslq_s16(neq, vdupq_n_s16(0), neg);
            back16(vbslq_s16(pos, x16, nonpos))
        };
        [half(a[0], b[0]), half(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_mullo_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vmulq_s16(s16(a[0]), s16(b[0]))),
            back16(vmulq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    /// `vpmulhw` — signed high half of the i16 product. Rust's stdarch
    /// exposes no plain `vmulh`; `vmull` + `vshrn_n::<16>` is the same
    /// truncating high half (no rounding — that would be `vqrdmulh`).
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_mulhi_epi16(a: __m256i, b: __m256i) -> __m256i {
        let half = |x: int32x4_t, y: int32x4_t| -> int32x4_t {
            let (x16, y16) = (s16(x), s16(y));
            let lo = vmull_s16(vget_low_s16(x16), vget_low_s16(y16));
            let hi = vmull_high_s16(x16, y16);
            back16(vcombine_s16(vshrn_n_s32::<16>(lo), vshrn_n_s32::<16>(hi)))
        };
        [half(a[0], b[0]), half(a[1], b[1])]
    }
    /// `vpmulhuw` — unsigned high half, same `vmull`/`vshrn` construction.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_mulhi_epu16(a: __m256i, b: __m256i) -> __m256i {
        let half = |x: int32x4_t, y: int32x4_t| -> int32x4_t {
            let (x16, y16) = (u16v(x), u16v(y));
            let lo = vmull_u16(vget_low_u16(x16), vget_low_u16(y16));
            let hi = vmull_high_u16(x16, y16);
            backu16(vcombine_u16(vshrn_n_u32::<16>(lo), vshrn_n_u32::<16>(hi)))
        };
        [half(a[0], b[0]), half(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_cmpgt_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            backu16(vcgtq_s16(s16(a[0]), s16(b[0]))),
            backu16(vcgtq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_cmpeq_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            backu16(vceqq_s16(s16(a[0]), s16(b[0]))),
            backu16(vceqq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_adds_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vqaddq_s16(s16(a[0]), s16(b[0]))),
            back16(vqaddq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_sub_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vsubq_s16(s16(a[0]), s16(b[0]))),
            back16(vsubq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_add_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vaddq_s16(s16(a[0]), s16(b[0]))),
            back16(vaddq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_max_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vmaxq_s16(s16(a[0]), s16(b[0]))),
            back16(vmaxq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_min_epi16(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vminq_s16(s16(a[0]), s16(b[0]))),
            back16(vminq_s16(s16(a[1]), s16(b[1]))),
        ]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i {
        [vandq_s32(a[0], b[0]), vandq_s32(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_or_si256(a: __m256i, b: __m256i) -> __m256i {
        [vorrq_s32(a[0], b[0]), vorrq_s32(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_xor_si256(a: __m256i, b: __m256i) -> __m256i {
        [veorq_s32(a[0], b[0]), veorq_s32(a[1], b[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_srli_epi16<const IMM: i32>(a: __m256i) -> __m256i {
        let half = |v: int32x4_t| -> int32x4_t {
            if IMM <= 0 {
                v
            } else if IMM >= 16 {
                vdupq_n_s32(0)
            } else {
                backu16(vshrq_n_u16::<IMM>(u16v(v)))
            }
        };
        [half(a[0]), half(a[1])]
    }
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_srai_epi16<const IMM: i32>(a: __m256i) -> __m256i {
        [_mm_srai_epi16::<IMM>(a[0]), _mm_srai_epi16::<IMM>(a[1])]
    }
    /// `vpsraw` with a runtime count — `vshlq_s16` with a negative count, the
    /// same sign-fill saturation as the i32 twin.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_sra_epi16(a: __m256i, count: __m128i) -> __m256i {
        let n = vgetq_lane_s64::<0>(s64v(count)) as i32;
        let sh = vdupq_n_s16((-n) as i16);
        [
            back16(vshlq_s16(s16(a[0]), sh)),
            back16(vshlq_s16(s16(a[1]), sh)),
        ]
    }
    /// `vpackssdw` — saturating signed i32 -> i16 narrow PER 128-LANE (the
    /// AVX2 lane-local semantics): `[sat(a.lo), sat(b.lo) | sat(a.hi), sat(b.hi)]`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_packs_epi32(a: __m256i, b: __m256i) -> __m256i {
        [
            back16(vcombine_s16(vqmovn_s32(a[0]), vqmovn_s32(b[0]))),
            back16(vcombine_s16(vqmovn_s32(a[1]), vqmovn_s32(b[1]))),
        ]
    }
    /// `vpmovmskb` — the sign bit of each byte, gathered into an i32. NEON
    /// has no movemask; the standard twin is a per-byte variable right shift
    /// that lands the sign bit at bit `i % 8` of its byte, then a byte-wise
    /// horizontal add over each 8-byte half.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_movemask_epi8(a: __m256i) -> i32 {
        let shifts = [
            -7i8, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0,
        ];
        let sh = vld1q_s8(&shifts);
        let half = |v: int32x4_t| -> u32 {
            let bytes = vreinterpretq_u8_s32(v);
            let top = vandq_u8(bytes, vdupq_n_u8(0x80));
            let spread = vshlq_u8(top, sh);
            (vaddv_u8(vget_low_u8(spread)) as u32) | ((vaddv_u8(vget_high_u8(spread)) as u32) << 8)
        };
        (half(a[0]) | (half(a[1]) << 16)) as i32
    }
    /// `vpinsrw` — insert `x` into i16 lane `IMM & 15`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm256_insert_epi16<const IMM: i32>(a: __m256i, x: i16) -> __m256i {
        let lane = (IMM & 15) as usize;
        let mut r = a;
        let half = (lane / 8) as usize;
        let mut v = s16(r[half]);
        v = match lane & 7 {
            0 => vsetq_lane_s16::<0>(x, v),
            1 => vsetq_lane_s16::<1>(x, v),
            2 => vsetq_lane_s16::<2>(x, v),
            3 => vsetq_lane_s16::<3>(x, v),
            4 => vsetq_lane_s16::<4>(x, v),
            5 => vsetq_lane_s16::<5>(x, v),
            6 => vsetq_lane_s16::<6>(x, v),
            _ => vsetq_lane_s16::<7>(x, v),
        };
        r[half] = back16(v);
        r
    }
    /// `vpextrw` — extract i16 lane `IMM`, zero-extended to i32.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn _mm_extract_epi16<const IMM: i32>(a: __m128i) -> i32 {
        let v = s16(a);
        let x = match IMM & 7 {
            0 => vgetq_lane_s16::<0>(v),
            1 => vgetq_lane_s16::<1>(v),
            2 => vgetq_lane_s16::<2>(v),
            3 => vgetq_lane_s16::<3>(v),
            4 => vgetq_lane_s16::<4>(v),
            5 => vgetq_lane_s16::<5>(v),
            6 => vgetq_lane_s16::<6>(v),
            _ => vgetq_lane_s16::<7>(v),
        };
        (x as u16) as i32
    }

    // ---- i16x16 <-> __m256i repr crossings -------------------------------------
    //
    // `i16x16<NeonToken>`'s Repr is `[int16x8_t; 2]` — a different type from
    // this module's `__m256i` even though both are 256 bits. The conversion is
    // a pure reinterpret.

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn i16x16_of_m256(t: NeonToken, m: __m256i) -> i16x16<NeonToken> {
        i16x16::from_repr(t, [s16(m[0]), s16(m[1])])
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn m256_of_i16x16(v: i16x16<NeonToken>) -> __m256i {
        let r = v.into_repr();
        [back16(r[0]), back16(r[1])]
    }
}

#[allow(unused_imports)]
pub(crate) use imp::*;
