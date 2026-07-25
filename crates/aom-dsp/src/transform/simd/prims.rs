//! Lane primitives for the transform vector passes — the whole of the
//! architecture-dependent surface.
//!
//! Everything else under [`super`] (the four 2-D pass drivers and the ~4,200
//! generated lines of 1-D lane kernels) is
//! `#[magetypes(define(i32x8), v3, neon, -scalar)]`: ONE body per kernel, with
//! `Token` and the vector types substituted per tier. What makes that possible
//! is this module, which supplies the handful of operations the generic
//! magetypes API cannot express — integer WIDENING (i32 → i64; verified absent
//! at magetypes 0.9.28, whose `generic::cross_width` raising/lowering covers
//! f32 only) and cross-lane PERMUTES — as hand-written per-tier variants.
//!
//! # Which magetypes pattern each primitive uses
//!
//! * **Pattern B (extracted generic kernel)** — [`clampv`], [`negv`],
//!   [`widen16`]. Pure magetypes ops, so one generic body bounded on
//!   `I32x8Backend` serves every tier. `#[inline(always)]` is MANDATORY on these:
//!   a generic fn has no `#[target_feature]` of its own and inherits the caller's
//!   only through inlining (magetypes README: without it the path regresses
//!   ~18x). Callers invoke them directly — no `incant!` needed.
//! * **Pattern C (hand-tuned tier slotted into the family)** — everything else.
//!   Each tier module defines the SAME set of names with `#[rite(<tier>)]`, and
//!   the cfg-selected re-export at the bottom of this file publishes exactly one
//!   module's set. So callers write `hb(t, ..)` with no suffix and no cfg, and
//!   because `#[rite]` attaches `#[target_feature]` DIRECTLY (no wrapper, unlike
//!   `#[arcane]`) the call inlines straight into the caller's target-feature
//!   region. No `incant!` is needed: these are only ever called from inside a
//!   tier-annotated body, whose features are by construction a superset.
//!
//! The vector width is 8 lanes on every target. On x86-64 that is one native
//! AVX2 register; on aarch64 magetypes' `i32x8` is the 2xNEON polyfill
//! (`Repr = [int32x4_t; 2]`), so each NEON primitive works on the two halves.
//! Keeping the width identical across tiers is what lets the drivers and the
//! generated kernels stay a single shared body.
//!
//! # Bit-exactness
//!
//! Each primitive reproduces its scalar counterpart's semantics **for any i32
//! lane values** — no input-range reasoning, so SIMD == scalar unconditionally.
//! The traps, and how each tier avoids them:
//!
//! * `half_btf` ([`x86::hb`] / [`neon::hb`]) wraps each PRODUCT in i32
//!   (matching C's `int` multiply) but sums the two products + rounding in
//!   **i64**. At the driver's clamp bounds a product reaches 2^32 and the sum
//!   needs 33 bits, so summing in i32 lanes (libaom's own SSE4/AVX2 shape)
//!   diverges on crafted-but-decodable streams. Both tiers widen and sum in i64.
//! * Arithmetic shift right of an i64 lane. AVX2 has no `vpsraq`, so the x86
//!   tier uses the identity `((v >>_arith b) as i32) == low32(v >>_logical b)`
//!   for `1 <= b <= 32` (the differing sign-fill bits all land at positions >= 32
//!   and are truncated away). AArch64 has a real signed 64-bit shift, so the NEON
//!   tier just shifts — `vshlq_s64` with a NEGATIVE count is an arithmetic shift
//!   right, which also absorbs the runtime `cos_bit` without a const generic.
//! * Narrowing i64 -> i32 after clamping to the i32 range
//!   ([`x86::shl_clamp64v`] / [`neon::shl_clamp64v`]): AVX2 has no
//!   `vpmin/maxq`, so x86 does compare + blend then truncates; AArch64 has
//!   `vqmovn_s64`, whose SATURATING narrow *is* exactly "clamp to
//!   `[i32::MIN, i32::MAX]` then truncate".
//!
//! Pinned by `super::tests` and `tests/txfm2d_simd_perm_diff.rs` at every token
//! permutation — including aarch64's baseline `neon`, which enters the
//! permutation set only because `archmage/testable_dispatch` is enabled in
//! dev-dependencies (see `crate::dispatch` docs).

use magetypes::simd::backends::I32x8Backend;
use magetypes::simd::generic::i32x8;

// ---------------------------------------------------------------------------
// Pattern B — one generic body, every tier (pure magetypes ops).
// ---------------------------------------------------------------------------

/// `clamp_value(v, bit)` on lanes — identical to the scalar port for any i32
/// lanes and any `bit`: `<= 0` and `>= 32` are identities (the scalar i64 bounds
/// cover all of i32 there), else lane min/max on the i32 bounds.
#[inline(always)]
pub(crate) fn clampv<T: I32x8Backend>(t: T, v: i32x8<T>, bit: i8) -> i32x8<T> {
    if bit <= 0 || bit >= 32 {
        return v;
    }
    let hi = ((1i64 << (bit - 1)) - 1) as i32;
    let lo = (-(1i64 << (bit - 1))) as i32;
    v.clamp(i32x8::splat(t, lo), i32x8::splat(t, hi))
}

/// `wrapping_neg` on lanes (`0 - v` wraps identically).
#[inline(always)]
pub(crate) fn negv<T: I32x8Backend>(t: T, v: i32x8<T>) -> i32x8<T> {
    i32x8::zero(t) - v
}

/// Sign-extend 8 i16s (the forward transform's residual input) to i32 lanes.
/// The fixed-size array round-trip lets LLVM emit the widening load
/// (`vpmovsxwd` / `sshll`).
#[inline(always)]
pub(crate) fn widen16<T: I32x8Backend>(t: T, s: &[i16]) -> i32x8<T> {
    let a: [i16; 8] = s[..8].try_into().unwrap();
    i32x8::from_array(t, core::array::from_fn(|j| a[j] as i32))
}

// ---------------------------------------------------------------------------
// Pattern C — x86-64 / AVX2 tier.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::i32x8;
    use archmage::prelude::*;

    /// The AVX2 tier's vector type, spelled once.
    type V = i32x8<X64V3Token>;

    /// Gather the low dword of each i64 lane of (`lo`, `hi`) into one `__m256i`.
    #[rite(v3)]
    fn low32_of_i64(
        lo: core::arch::x86_64::__m256i,
        hi: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        use core::arch::x86_64::*;
        let idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let a = _mm256_permutevar8x32_epi32(lo, idx);
        let b = _mm256_permutevar8x32_epi32(hi, idx);
        _mm256_blend_epi32::<0b1111_0000>(a, b)
    }

    /// `half_btf` on 8 lanes — the exact-i64 recipe (see the module docs).
    #[rite(v3)]
    pub(crate) fn hb(t: X64V3Token, w0: i32, in0: V, w1: i32, in1: V, cos_bit: i32) -> V {
        use core::arch::x86_64::*;
        // Wrapped i32 products, exactly like the scalar port's wrapping_mul.
        let p0 = _mm256_mullo_epi32(_mm256_set1_epi32(w0), in0.raw());
        let p1 = _mm256_mullo_epi32(_mm256_set1_epi32(w1), in1.raw());
        // Widen to i64 and sum with the rounding constant — no i64 overflow:
        // |p0|,|p1| <= 2^31 and rnd <= 2^31, so |sum| <= 2^32 + 2^31 < 2^63.
        let rnd = _mm256_set1_epi64x(1i64 << (cos_bit - 1));
        let lo = _mm256_add_epi64(
            _mm256_add_epi64(
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p0)),
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p1)),
            ),
            rnd,
        );
        let hi = _mm256_add_epi64(
            _mm256_add_epi64(
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(p0)),
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(p1)),
            ),
            rnd,
        );
        let cnt = _mm_cvtsi32_si128(cos_bit);
        i32x8::from_m256i(t, low32_of_i64(_mm256_srl_epi64(lo, cnt), _mm256_srl_epi64(hi, cnt)))
    }

    /// `round_shift(v as i64, bit)` on lanes for `bit` in `1..=32`.
    #[rite(v3)]
    pub(crate) fn rshiftv(t: X64V3Token, v: V, bit: i32) -> V {
        use core::arch::x86_64::*;
        debug_assert!((1..=32).contains(&bit));
        let rnd = _mm256_set1_epi64x(1i64 << (bit - 1));
        let lo = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(v.raw())), rnd);
        let hi =
            _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(v.raw())), rnd);
        let cnt = _mm_cvtsi32_si128(bit);
        i32x8::from_m256i(t, low32_of_i64(_mm256_srl_epi64(lo, cnt), _mm256_srl_epi64(hi, cnt)))
    }

    /// Reverse the 8 lanes (for `lr_flip` column groups).
    #[rite(v3)]
    pub(crate) fn revv(t: X64V3Token, v: V) -> V {
        use core::arch::x86_64::*;
        let idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        i32x8::from_m256i(t, _mm256_permutevar8x32_epi32(v.raw(), idx))
    }

    /// `round_shift(v as i64 * mul, bit)` on lanes — the full-i64-product recipe
    /// (`vpmuldq` even/odd gives exact 32x32->64 signed products), add rounding
    /// (no overflow: |mul| <= 2^14 at the call sites → |prod| <= 2^45), LOGICAL
    /// shift + take the low dword of each i64 lane (exact for `1 <= bit <= 32`).
    #[rite(v3)]
    pub(crate) fn mul_rshiftv(t: X64V3Token, v: V, mul: i32, bit: i32) -> V {
        use core::arch::x86_64::*;
        debug_assert!((1..=32).contains(&bit) && mul.unsigned_abs() < (1 << 15));
        let m = _mm256_set1_epi32(mul);
        // vpmuldq reads the SIGNED low dword of each 64-bit lane.
        let even = _mm256_mul_epi32(v.raw(), m); // source lanes 0,2,4,6
        let odd = _mm256_mul_epi32(_mm256_srli_epi64::<32>(v.raw()), m); // lanes 1,3,5,7
        let rnd = _mm256_set1_epi64x(1i64 << (bit - 1));
        let cnt = _mm_cvtsi32_si128(bit);
        let re = _mm256_srl_epi64(_mm256_add_epi64(even, rnd), cnt);
        let ro = _mm256_srl_epi64(_mm256_add_epi64(odd, rnd), cnt);
        // Valid low dwords of `re` sit at dword positions 0,2,4,6 (source lanes
        // 0,2,4,6); shift `ro`'s up to 1,3,5,7 and blend.
        let out = _mm256_blend_epi32::<0b1010_1010>(re, _mm256_slli_epi64::<32>(ro));
        i32x8::from_m256i(t, out)
    }

    /// The NEGATIVE-bit `round_shift_array` arm on lanes: `clamp_i64(v << k)`
    /// truncated to i32 — widen to i64 halves, shift left (exact: k <= 4 →
    /// |v<<k| < 2^36), clamp to the i32 range with cmpgt/blendv min/max (AVX2 has
    /// no `vpmin/maxq`), take low dwords. Bit-identical to the scalar arm
    /// (`((1i64 << k) * v).clamp(i32::MIN, i32::MAX) as i32`) for ANY i32 v. Used
    /// by the FORWARD col pass (fwd shift[0] == 2); the inverse shifts are all
    /// positive-bit.
    #[rite(v3)]
    pub(crate) fn shl_clamp64v(t: X64V3Token, v: V, k: i32) -> V {
        use core::arch::x86_64::*;
        debug_assert!((1..=4).contains(&k));
        let cnt = _mm_cvtsi32_si128(k);
        let min_v = _mm256_set1_epi64x(i32::MIN as i64);
        let max_v = _mm256_set1_epi64x(i32::MAX as i64);
        let part = |x: __m128i| -> __m256i {
            let w = _mm256_sll_epi64(_mm256_cvtepi32_epi64(x), cnt);
            // min(w, max): if w > max take max; then max(_, min): if min > w take min.
            let w = _mm256_blendv_epi8(w, max_v, _mm256_cmpgt_epi64(w, max_v));
            _mm256_blendv_epi8(w, min_v, _mm256_cmpgt_epi64(min_v, w))
        };
        let lo = part(_mm256_castsi256_si128(v.raw()));
        let hi = part(_mm256_extracti128_si256::<1>(v.raw()));
        i32x8::from_m256i(t, low32_of_i64(lo, hi))
    }

    /// 8x8 i32 in-register transpose (unpack32 → unpack64 → permute2x128, the
    /// standard 24-op AVX2 pattern) — a pure lane permutation, so exactness is
    /// structural. Used by the row passes (strided side of the tile).
    #[rite(v3)]
    pub(crate) fn transpose8(t: X64V3Token, v: &[V]) -> [V; 8] {
        use core::arch::x86_64::*;
        let a0 = _mm256_unpacklo_epi32(v[0].raw(), v[1].raw());
        let a1 = _mm256_unpackhi_epi32(v[0].raw(), v[1].raw());
        let a2 = _mm256_unpacklo_epi32(v[2].raw(), v[3].raw());
        let a3 = _mm256_unpackhi_epi32(v[2].raw(), v[3].raw());
        let a4 = _mm256_unpacklo_epi32(v[4].raw(), v[5].raw());
        let a5 = _mm256_unpackhi_epi32(v[4].raw(), v[5].raw());
        let a6 = _mm256_unpacklo_epi32(v[6].raw(), v[7].raw());
        let a7 = _mm256_unpackhi_epi32(v[6].raw(), v[7].raw());
        let b0 = _mm256_unpacklo_epi64(a0, a2);
        let b1 = _mm256_unpackhi_epi64(a0, a2);
        let b2 = _mm256_unpacklo_epi64(a1, a3);
        let b3 = _mm256_unpackhi_epi64(a1, a3);
        let b4 = _mm256_unpacklo_epi64(a4, a6);
        let b5 = _mm256_unpackhi_epi64(a4, a6);
        let b6 = _mm256_unpacklo_epi64(a5, a7);
        let b7 = _mm256_unpackhi_epi64(a5, a7);
        [
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
            i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
        ]
    }

    /// A lane vector held as two `i64x4` halves (lanes 0..4 / 4..8) — the
    /// representation for the all-i64 `iadst4` math.
    #[derive(Clone, Copy)]
    pub(crate) struct V64 {
        lo: core::arch::x86_64::__m256i,
        hi: core::arch::x86_64::__m256i,
    }

    /// Sign-extend the 8 i32 lanes to two i64x4 halves.
    #[rite(v3)]
    pub(crate) fn widen64(v: V) -> V64 {
        use core::arch::x86_64::*;
        V64 {
            lo: _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v.raw())),
            hi: _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(v.raw())),
        }
    }

    #[rite(v3)]
    pub(crate) fn add64(a: V64, b: V64) -> V64 {
        use core::arch::x86_64::*;
        V64 { lo: _mm256_add_epi64(a.lo, b.lo), hi: _mm256_add_epi64(a.hi, b.hi) }
    }

    #[rite(v3)]
    pub(crate) fn sub64(a: V64, b: V64) -> V64 {
        use core::arch::x86_64::*;
        V64 { lo: _mm256_sub_epi64(a.lo, b.lo), hi: _mm256_sub_epi64(a.hi, b.hi) }
    }

    /// `c * v` per i64 lane for a NON-NEGATIVE constant `c < 2^31` — exact mod
    /// 2^64 (== the scalar i64 product wherever it fits, which the iadst4 bounds
    /// guarantee: |v| < 2^34, c = sinpi < 2^14 → |c*v| < 2^48). Decompose
    /// v = v_lo_u + v_hi_u*2^32 (unsigned dwords): `c*v mod 2^64 = c*v_lo_u +
    /// ((c*v_hi_u) << 32)` with wrapping adds/shifts.
    #[rite(v3)]
    pub(crate) fn mulc64(v: V64, c: i32) -> V64 {
        use core::arch::x86_64::*;
        debug_assert!(c >= 0);
        let cv = _mm256_set1_epi64x(c as i64); // low dword of each i64 lane = c
        let part = |x: __m256i| -> __m256i {
            let lo_prod = _mm256_mul_epu32(x, cv); // c * v_lo_u (exact, < 2^63)
            let hi_prod = _mm256_mul_epu32(_mm256_srli_epi64::<32>(x), cv); // c * v_hi_u
            _mm256_add_epi64(lo_prod, _mm256_slli_epi64::<32>(hi_prod))
        };
        V64 { lo: part(v.lo), hi: part(v.hi) }
    }

    /// `round_shift(v, bit)` from i64 lanes to i32 lanes — add rounding, LOGICAL
    /// shift, take the low dword (exact for `1 <= bit <= 32`, same identity as
    /// [`x86::hb`]).
    #[rite(v3)]
    pub(crate) fn rshift64(t: X64V3Token, v: V64, bit: i32) -> V {
        use core::arch::x86_64::*;
        let rnd = _mm256_set1_epi64x(1i64 << (bit - 1));
        let cnt = _mm_cvtsi32_si128(bit);
        let lo = _mm256_srl_epi64(_mm256_add_epi64(v.lo, rnd), cnt);
        let hi = _mm256_srl_epi64(_mm256_add_epi64(v.hi, rnd), cnt);
        i32x8::from_m256i(t, low32_of_i64(lo, hi))
    }
}

// ---------------------------------------------------------------------------
// Pattern C — aarch64 / NEON tier.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::i32x8;
    use archmage::prelude::*;

    /// The NEON tier's vector type, spelled once. `Repr = [int32x4_t; 2]`.
    type V = i32x8<NeonToken>;

    /// `round_shift(x as i64, bit)` over two i64 pairs: add rounding, ARITHMETIC
    /// shift right by a runtime count, narrow, recombine into 4 i32 lanes.
    /// `vshlq_s64` with a negative count is a signed right shift, so no
    /// const-generic shift is needed (AArch64, unlike AVX2, has a real 64-bit
    /// signed shift — no logical-shift-and-truncate trick).
    ///
    /// `#[rite(neon)]` (not a plain `#[inline(always)]` fn) because the body
    /// calls NEON intrinsics: outside a `#[target_feature(enable = "neon")]`
    /// region those calls are `unsafe`, and `neon` being a compile-time
    /// baseline of the target does NOT waive the requirement to list it.
    /// Mirrors the x86 tier's [`super::x86::low32_of_i64`], which is likewise
    /// `#[rite(v3)]` with no token parameter.
    #[rite(neon)]
    fn rnd_shr_narrow(lo: int64x2_t, hi: int64x2_t, rnd: int64x2_t, sh: int64x2_t) -> int32x4_t {
        vcombine_s32(
            vmovn_s64(vshlq_s64(vaddq_s64(lo, rnd), sh)),
            vmovn_s64(vshlq_s64(vaddq_s64(hi, rnd), sh)),
        )
    }

    /// `half_btf` on 8 lanes — the exact-i64 recipe (see the module docs).
    /// Products wrap in i32 (`vmulq_s32`, matching the scalar `wrapping_mul`),
    /// then each is sign-extended to i64 so the sum + rounding is exact.
    #[rite(neon)]
    pub(crate) fn hb(t: NeonToken, w0: i32, in0: V, w1: i32, in1: V, cos_bit: i32) -> V {
        let (a, b) = (in0.into_repr(), in1.into_repr());
        let vw0 = vdupq_n_s32(w0);
        let vw1 = vdupq_n_s32(w1);
        let rnd = vdupq_n_s64(1i64 << (cos_bit - 1));
        let sh = vdupq_n_s64(-(cos_bit as i64));
        let half = |x: int32x4_t, y: int32x4_t| -> int32x4_t {
            let p0 = vmulq_s32(vw0, x);
            let p1 = vmulq_s32(vw1, y);
            // |p0|,|p1| <= 2^31 and rnd <= 2^31 → |sum| < 2^63, no i64 overflow.
            let lo = vaddq_s64(vmovl_s32(vget_low_s32(p0)), vmovl_s32(vget_low_s32(p1)));
            let hi = vaddq_s64(vmovl_high_s32(p0), vmovl_high_s32(p1));
            rnd_shr_narrow(lo, hi, rnd, sh)
        };
        i32x8::from_repr(t, [half(a[0], b[0]), half(a[1], b[1])])
    }

    /// `round_shift(v as i64, bit)` on lanes for `bit` in `1..=32`.
    #[rite(neon)]
    pub(crate) fn rshiftv(t: NeonToken, v: V, bit: i32) -> V {
        debug_assert!((1..=32).contains(&bit));
        let a = v.into_repr();
        let rnd = vdupq_n_s64(1i64 << (bit - 1));
        let sh = vdupq_n_s64(-(bit as i64));
        let part = |x: int32x4_t| -> int32x4_t {
            rnd_shr_narrow(vmovl_s32(vget_low_s32(x)), vmovl_high_s32(x), rnd, sh)
        };
        i32x8::from_repr(t, [part(a[0]), part(a[1])])
    }

    /// Reverse the 8 lanes (for `lr_flip` column groups). `vrev64q_s32` reverses
    /// within each 64-bit pair and `vextq_s32::<2>` swaps the pairs, which
    /// together reverse a 4-lane vector; swapping the two halves completes the
    /// 8-lane reversal.
    #[rite(neon)]
    pub(crate) fn revv(t: NeonToken, v: V) -> V {
        let a = v.into_repr();
        let rev4 = |x: int32x4_t| -> int32x4_t {
            let r = vrev64q_s32(x);
            vextq_s32::<2>(r, r)
        };
        i32x8::from_repr(t, [rev4(a[1]), rev4(a[0])])
    }

    /// `round_shift(v as i64 * mul, bit)` on lanes. `vmull_s32` /
    /// `vmull_high_s32` give exact 32x32->64 signed products (no wrapping, unlike
    /// [`neon::hb`]'s), matching the scalar `v as i64 * mul as i64`; |mul| <=
    /// 2^14 at the call sites so |prod| <= 2^45 and adding rounding cannot
    /// overflow.
    #[rite(neon)]
    pub(crate) fn mul_rshiftv(t: NeonToken, v: V, mul: i32, bit: i32) -> V {
        debug_assert!((1..=32).contains(&bit) && mul.unsigned_abs() < (1 << 15));
        let a = v.into_repr();
        let m2 = vdup_n_s32(mul);
        let m4 = vdupq_n_s32(mul);
        let rnd = vdupq_n_s64(1i64 << (bit - 1));
        let sh = vdupq_n_s64(-(bit as i64));
        let part = |x: int32x4_t| -> int32x4_t {
            rnd_shr_narrow(vmull_s32(vget_low_s32(x), m2), vmull_high_s32(x, m4), rnd, sh)
        };
        i32x8::from_repr(t, [part(a[0]), part(a[1])])
    }

    /// The NEGATIVE-bit `round_shift_array` arm on lanes: `clamp_i64(v << k)`
    /// truncated to i32. `vqmovn_s64`'s SATURATING narrow is exactly "clamp to
    /// `[i32::MIN, i32::MAX]` then truncate", so this needs no compare/blend
    /// (contrast the x86 tier, where AVX2 lacks a 64-bit min/max). Bit-identical
    /// to the scalar `((1i64 << k) * v).clamp(i32::MIN, i32::MAX) as i32` for ANY
    /// i32 v: k <= 4, so `v << k` cannot overflow i64 before the clamp.
    #[rite(neon)]
    pub(crate) fn shl_clamp64v(t: NeonToken, v: V, k: i32) -> V {
        debug_assert!((1..=4).contains(&k));
        let a = v.into_repr();
        let sh = vdupq_n_s64(k as i64);
        let part = |x: int32x4_t| -> int32x4_t {
            let lo = vshlq_s64(vmovl_s32(vget_low_s32(x)), sh);
            let hi = vshlq_s64(vmovl_high_s32(x), sh);
            vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi))
        };
        i32x8::from_repr(t, [part(a[0]), part(a[1])])
    }

    /// 8x8 i32 in-register transpose — a pure lane permutation, so exactness is
    /// structural. With `i32x8` held as two 4-lane halves, the 8x8 transpose
    /// decomposes into four independent 4x4 transposes: output row `i` takes its
    /// low half from input rows 0..4 and its high half from rows 4..8, while
    /// output rows 0..4 read the inputs' low halves and rows 4..8 the high ones.
    #[rite(neon)]
    pub(crate) fn transpose8(t: NeonToken, v: &[V]) -> [V; 8] {
        let r: [[int32x4_t; 2]; 8] = core::array::from_fn(|i| v[i].into_repr());
        // 4x4 i32 transpose: the 32-bit trn pairs even/odd lanes, then the 64-bit
        // trn interleaves the halves.
        let t4 = |a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t| -> [int32x4_t; 4] {
            let e0 = vtrn1q_s32(a, b);
            let e1 = vtrn2q_s32(a, b);
            let e2 = vtrn1q_s32(c, d);
            let e3 = vtrn2q_s32(c, d);
            let (f0, f1) = (vreinterpretq_s64_s32(e0), vreinterpretq_s64_s32(e2));
            let (f2, f3) = (vreinterpretq_s64_s32(e1), vreinterpretq_s64_s32(e3));
            [
                vreinterpretq_s32_s64(vtrn1q_s64(f0, f1)),
                vreinterpretq_s32_s64(vtrn1q_s64(f2, f3)),
                vreinterpretq_s32_s64(vtrn2q_s64(f0, f1)),
                vreinterpretq_s32_s64(vtrn2q_s64(f2, f3)),
            ]
        };
        let lo_lo = t4(r[0][0], r[1][0], r[2][0], r[3][0]); // out 0..4, lanes 0..4
        let lo_hi = t4(r[4][0], r[5][0], r[6][0], r[7][0]); // out 0..4, lanes 4..8
        let hi_lo = t4(r[0][1], r[1][1], r[2][1], r[3][1]); // out 4..8, lanes 0..4
        let hi_hi = t4(r[4][1], r[5][1], r[6][1], r[7][1]); // out 4..8, lanes 4..8
        core::array::from_fn(|i| {
            if i < 4 {
                i32x8::from_repr(t, [lo_lo[i], lo_hi[i]])
            } else {
                i32x8::from_repr(t, [hi_lo[i - 4], hi_hi[i - 4]])
            }
        })
    }

    /// A lane vector held as four `i64x2` quarters — the representation for the
    /// all-i64 `iadst4` math. Same 8 i64 lanes as the x86 `V64`, split to NEON's
    /// register width.
    #[derive(Clone, Copy)]
    pub(crate) struct V64([int64x2_t; 4]);

    /// Sign-extend the 8 i32 lanes to four i64x2 quarters.
    #[rite(neon)]
    pub(crate) fn widen64(v: V) -> V64 {
        let a = v.into_repr();
        V64([
            vmovl_s32(vget_low_s32(a[0])),
            vmovl_high_s32(a[0]),
            vmovl_s32(vget_low_s32(a[1])),
            vmovl_high_s32(a[1]),
        ])
    }

    #[rite(neon)]
    pub(crate) fn add64(a: V64, b: V64) -> V64 {
        V64(core::array::from_fn(|i| vaddq_s64(a.0[i], b.0[i])))
    }

    #[rite(neon)]
    pub(crate) fn sub64(a: V64, b: V64) -> V64 {
        V64(core::array::from_fn(|i| vsubq_s64(a.0[i], b.0[i])))
    }

    /// `c * v` per i64 lane for a NON-NEGATIVE constant `c < 2^31` — exact mod
    /// 2^64 (== the scalar i64 product wherever it fits, which the iadst4 bounds
    /// guarantee: |v| < 2^34, c = sinpi < 2^14 → |c*v| < 2^48).
    ///
    /// AArch64 has no 64x64 multiply, so this uses the same dword decomposition
    /// as the x86 tier: v = v_lo_u + v_hi_u*2^32 (unsigned dwords), hence
    /// `c*v mod 2^64 = c*v_lo_u + ((c*v_hi_u) << 32)`, with `vmull_u32` supplying
    /// the exact 32x32->64 unsigned products.
    #[rite(neon)]
    pub(crate) fn mulc64(v: V64, c: i32) -> V64 {
        debug_assert!(c >= 0);
        let cv = vdup_n_u32(c as u32);
        V64(core::array::from_fn(|i| {
            let xu = vreinterpretq_u64_s64(v.0[i]);
            let lo_prod = vmull_u32(vmovn_u64(xu), cv); // c * v_lo_u (exact, < 2^63)
            let hi_prod = vmull_u32(vshrn_n_u64::<32>(xu), cv); // c * v_hi_u
            vreinterpretq_s64_u64(vaddq_u64(lo_prod, vshlq_n_u64::<32>(hi_prod)))
        }))
    }

    /// `round_shift(v, bit)` from i64 lanes to i32 lanes — add rounding, then a
    /// real ARITHMETIC shift right (negative-count `vshlq_s64`) and a truncating
    /// narrow. Same result as the x86 tier's logical-shift-then-low-dword
    /// identity, without needing it.
    #[rite(neon)]
    pub(crate) fn rshift64(t: NeonToken, v: V64, bit: i32) -> V {
        let rnd = vdupq_n_s64(1i64 << (bit - 1));
        let sh = vdupq_n_s64(-(bit as i64));
        let q = |i: usize| vmovn_s64(vshlq_s64(vaddq_s64(v.0[i], rnd), sh));
        i32x8::from_repr(t, [vcombine_s32(q(0), q(1)), vcombine_s32(q(2), q(3))])
    }
}

// The two tiers export the SAME names; exactly one module is compiled, so every
// caller in `super` and in the generated kernels writes `hb(t, ..)` /
// `transpose8(t, ..)` with no cfg and no suffix at the call site.
#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{
    add64, hb, mul_rshiftv, mulc64, revv, rshift64, rshiftv, shl_clamp64v, sub64, transpose8,
    widen64,
};
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{
    add64, hb, mul_rshiftv, mulc64, revv, rshift64, rshiftv, shl_clamp64v, sub64, transpose8,
    widen64,
};
