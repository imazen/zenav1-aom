//! Lane primitives for the **bd8 i16-lane** transform passes — the whole of
//! the architecture-dependent surface of [`super::lowbd16`] and the generated
//! [`super::inv1d_v3_i16_gen`] kernels.
//!
//! Same shape and the same rules as [`super::prims`], one width down: the
//! drivers and the ~1,300 generated lines of i16 lane kernels are ONE
//! `#[magetypes(define(i16x16, i32x8), v3, neon, -scalar)]` body each, and
//! everything the generic magetypes API cannot express lives here as
//! hand-written per-tier `#[rite]` variants under a single cfg-selected name.
//!
//! # What the generic API is missing (audited at magetypes 0.9.27/0.9.28)
//!
//! `i16x16<T>` exposes wrapping add/sub/mul/neg, min/max/abs/clamp, the
//! comparisons, `blend`, const shifts, the bitwise ops and load/store/array —
//! and nothing else. Specifically ABSENT, all four of which the i16 transform
//! contract is built on:
//!
//! * **saturating add/sub** — the string `saturat` appears nowhere in the
//!   crate's integer surface (only inside f32→int conversion helpers). Yet
//!   `clamp_value(a + b, 16)` on two i16-domain values IS the saturating i16
//!   add, and that identity is the entire reason the i16 pass is cheap.
//! * **integer widening / narrowing** — `backends::convert_int` is bitcast
//!   only, and `generic::cross_width` (`from_halves`/`low`/`high`/`split`)
//!   covers f32x4/f32x8 only. So i16→i32 sign-extension and the SATURATING
//!   i32→i16 narrow (which is exactly `clamp_value(_, 16)` + pack) have no
//!   generic spelling.
//! * **multiply-accumulate across lane pairs** — no `madd`/`mull`/`mlal`
//!   family at any width; `half_btf` on i16 lanes needs one.
//! * **lane interleave / reverse / transpose for i16** — `block_ops_*` exists
//!   only for f32x4/8, f64x2/4, i32x4/8, i8x16 and u32x4, and even those are
//!   array/byte views, not shuffles.
//!
//! # Bit-exactness
//!
//! Every primitive reproduces its scalar counterpart on the domain the audited
//! kernel contract guarantees (i16 values in `i16x16` lanes, 17-bit unclamped
//! `half_btf` transients as exact i32 pairs — see [`super::lowbd16`] for the
//! domain argument). Per-op proofs are on each function. The two tiers reach
//! the same values differently:
//!
//! * `half_btf` ([`x86::btf16`] / [`neon::btf16`]): AVX2 has `vpmaddwd`, which
//!   wants the two operands INTERLEAVED as (x, y) i16 pairs — hence [`Upk`],
//!   and hence the x86 [`P32`] living in "unpack order" (the permutation
//!   `vpackssdw` undoes for free). AArch64 has widening multiply-accumulate
//!   (`vmull_n_s16` / `vmlal_n_s16`) which reads its operands SEPARATELY, so
//!   the NEON tier needs no interleave at all: its `Upk` is just the operand
//!   pair, its `P32` is in natural lane order, and `unpk16` is free.
//!   Both are exact: |w| <= 4096 and |x| <= 2^15, so each product fits in
//!   2^27 and the pair sum in 2^28 — no i32 wrap, so `wrapping_mul` and an
//!   exact multiply agree, and the rounding shift by 12 stays in i32.
//! * the narrowing clamp ([`x86::pack16`] / [`neon::pack16`]): `vpackssdw`
//!   and `vqmovn_s32` are both "saturate to [-2^15, 2^15-1] then truncate" ==
//!   `clamp_value(_, 16)`; the x86 one additionally undoes the unpack
//!   permutation, the NEON one has no permutation to undo.
//! * `round_shift` on i16 lanes ([`x86::mulhrs16`] / [`neon::mulhrs16`]):
//!   `_mm256_mulhrs_epi16(v, m)` computes `((v*m >> 14) + 1) >> 1` and
//!   `vqrdmulhq_s16(v, m)` computes `(2*v*m + 2^15) >> 16` — algebraically the
//!   SAME value `floor((v*m + 2^14) / 2^15)` for every i16 pair (write
//!   `v*m = 2^14 q + r`: the first is `floor((q+1)/2)`, the second is
//!   `floor((q+1)/2 + r/2^15)` and `r/2^15 < 1/2` can never carry). Taking
//!   `m = 2^(15-bit)` makes that `round_shift(v, bit)` exactly; see the proof
//!   at [`super::lowbd16::rshift_i16`]. `vqrdmulh`'s only saturating case
//!   (`-2^15 * -2^15`) is unreachable because `m > 0`.
//!
//! Pinned by [`super::lowbd16::tests`] at every token permutation, on both
//! architectures.

use magetypes::simd::generic::{i16x16, i32x8};

// ---------------------------------------------------------------------------
// Pattern C — x86-64 / AVX2 tier.
//
// Verbatim the AVX2 sequences that landed in `lowbd16.rs` on 2026-07-17 and
// have been the shipping bd8 path since; moving them here changed no
// instruction, so the x86 bitstream path is untouched by the NEON landing.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{i16x16, i32x8};
    use archmage::prelude::*;
    use core::arch::x86_64::__m256i;
    use magetypes::simd::generic::u8x16;

    type V16 = i16x16<X64V3Token>;
    type V32 = i32x8<X64V3Token>;

    /// A pair of interleaved-i16 vectors, the shared input of the two `madd`
    /// butterflies that consume the same (x, y) operands. `lo` holds source
    /// lanes 0-3 and 8-11 as (x, y) i16 pairs, `hi` lanes 4-7 and 12-15.
    #[derive(Clone, Copy)]
    pub(crate) struct Upk {
        lo: __m256i,
        hi: __m256i,
    }

    /// A 16-lane i32 value in UNPACK ORDER: `lo` = source lanes 0-3 (low 128)
    /// and 8-11 (high 128), `hi` = lanes 4-7 and 12-15. [`pack16`]'s
    /// per-128-lane `packs_epi32` maps this back to natural i16x16 lane order
    /// exactly.
    #[derive(Clone, Copy)]
    pub(crate) struct P32 {
        lo: V32,
        hi: V32,
    }

    /// Interleave two i16x16 values into (x, y) pairs for `madd` butterflies.
    #[rite(v3)]
    pub(crate) fn unpk16(t: X64V3Token, x: V16, y: V16) -> Upk {
        use core::arch::x86_64::*;
        let _ = t;
        Upk {
            lo: _mm256_unpacklo_epi16(x.raw(), y.raw()),
            hi: _mm256_unpackhi_epi16(x.raw(), y.raw()),
        }
    }

    /// `half_btf(w0, x, w1, y, 12)` on 16 lanes, x/y in the i16 domain —
    /// EXACT: `madd` computes `w0*x + w1*y` per i32 slot with |w| <= 4095 and
    /// |x|,|y| <= 2^15, so |each product| <= 2^27 (no i32 wrap — identical to
    /// the scalar port's `wrapping_mul` which also cannot wrap here) and |pair
    /// sum| <= 2^28 (madd's internal i32 pair-add is exact; its only
    /// saturation case needs both products == -2^31, impossible with |w| <=
    /// 4095). Adding the rounding constant 2^11 cannot overflow, and the
    /// arithmetic shift by 12 equals the scalar's i64 shift because the value
    /// fits i32. Output <= ~2^16.03: a 17-bit transient, kept as exact i32
    /// pairs.
    #[rite(v3)]
    pub(crate) fn btf16(t: X64V3Token, u: Upk, w0: i32, w1: i32) -> P32 {
        use core::arch::x86_64::*;
        debug_assert!(w0.unsigned_abs() < (1 << 15) && w1.unsigned_abs() < (1 << 15));
        let cw = _mm256_set1_epi32((((w1 as u32) & 0xffff) << 16 | ((w0 as u32) & 0xffff)) as i32);
        let rnd = _mm256_set1_epi32(1 << 11);
        P32 {
            lo: i32x8::from_m256i(
                t,
                _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(u.lo, cw), rnd)),
            ),
            hi: i32x8::from_m256i(
                t,
                _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(u.hi, cw), rnd)),
            ),
        }
    }

    /// `clamp_value(v, 16)` + narrow of an unpack-order i32 pair: per-128-lane
    /// `packs_epi32` saturates each i32 to [-2^15, 2^15-1] — exactly
    /// `clamp_value(_, 16)` — and restores natural lane order (pack inverts
    /// unpack within each 128-bit lane).
    #[rite(v3)]
    pub(crate) fn pack16(t: X64V3Token, p: P32) -> V16 {
        use core::arch::x86_64::*;
        i16x16::from_m256i(t, _mm256_packs_epi32(p.lo.raw(), p.hi.raw()))
    }

    /// Sign-extend an i16x16 into an unpack-order i32 pair (for clamp adds
    /// that mix an i16 operand with a 17-bit transient): interleaving the
    /// value with its `v < 0` mask (all-ones == 0xffff) builds the exact
    /// sign-extended i32 in each slot.
    #[rite(v3)]
    pub(crate) fn ext16(t: X64V3Token, v: V16) -> P32 {
        use core::arch::x86_64::*;
        let sign = _mm256_cmpgt_epi16(_mm256_setzero_si256(), v.raw());
        P32 {
            lo: i32x8::from_m256i(t, _mm256_unpacklo_epi16(v.raw(), sign)),
            hi: i32x8::from_m256i(t, _mm256_unpackhi_epi16(v.raw(), sign)),
        }
    }

    /// `clamp_value(a + b, 16)` for two i16-domain values: the saturating i16
    /// add IS the normative clamp (a+b in [-2^16, 2^16-2] saturates to exactly
    /// `clamp_value`'s [-2^15, 2^15-1]).
    #[rite(v3)]
    pub(crate) fn sadd16(t: X64V3Token, a: V16, b: V16) -> V16 {
        use core::arch::x86_64::*;
        i16x16::from_m256i(t, _mm256_adds_epi16(a.raw(), b.raw()))
    }

    /// `clamp_value(a - b, 16)` for two i16-domain values (also serves
    /// `clamp_value(-b + a, 16)` — identical in two's complement).
    #[rite(v3)]
    pub(crate) fn ssub16(t: X64V3Token, a: V16, b: V16) -> V16 {
        use core::arch::x86_64::*;
        i16x16::from_m256i(t, _mm256_subs_epi16(a.raw(), b.raw()))
    }

    /// i32-pair add (exact: operands are <= 17-bit transients / sign-extended
    /// i16, so the wrapping lane add cannot wrap — identical to the scalar i32
    /// add).
    #[rite(v3)]
    pub(crate) fn padd32(t: X64V3Token, a: P32, b: P32) -> P32 {
        let _ = t;
        P32 { lo: a.lo + b.lo, hi: a.hi + b.hi }
    }

    /// i32-pair subtract (exact, same bound argument as [`padd32`]).
    #[rite(v3)]
    pub(crate) fn psub32(t: X64V3Token, a: P32, b: P32) -> P32 {
        let _ = t;
        P32 { lo: a.lo - b.lo, hi: a.hi - b.hi }
    }

    /// `round_shift(v, bit)` on i16 lanes as `mulhrs(v, 2^(15-bit))` — see the
    /// module docs and [`super::super::lowbd16::rshift_i16`] for the proof.
    #[rite(v3)]
    pub(crate) fn mulhrs16(t: X64V3Token, v: V16, m: i16) -> V16 {
        use core::arch::x86_64::*;
        i16x16::from_m256i(t, _mm256_mulhrs_epi16(v.raw(), _mm256_set1_epi16(m)))
    }

    /// Gather+clamp: two natural-order i32x8 values -> `clamp_value(_, 16)`
    /// per lane via the saturating pack, permuted back to natural lane order
    /// (packs_epi32 of natural-order inputs interleaves 128-lane quarters;
    /// `permute4x64(0xD8)` = [q0, q2, q1, q3] restores a0-7, b0-7).
    #[rite(v3)]
    pub(crate) fn pack_clamp16(t: X64V3Token, a: V32, b: V32) -> V16 {
        use core::arch::x86_64::*;
        let p = _mm256_packs_epi32(a.raw(), b.raw());
        i16x16::from_m256i(t, _mm256_permute4x64_epi64::<0b1101_1000>(p))
    }

    /// Reverse all 16 lanes (lr_flip on a full column group).
    #[rite(v3)]
    pub(crate) fn rev16(t: X64V3Token, v: V16) -> V16 {
        use core::arch::x86_64::*;
        let m = _mm256_setr_epi8(
            14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, //
            14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
        );
        let r = _mm256_shuffle_epi8(v.raw(), m);
        i16x16::from_m256i(t, _mm256_permute4x64_epi64::<0b0100_1110>(r))
    }

    /// Sign-extend lanes 0-7 of an i16x16 to a natural-order i32x8
    /// (`vpmovsxwd`) — a pure width extension, bit-exact.
    #[rite(v3)]
    pub(crate) fn widen_lo(t: X64V3Token, v: V16) -> V32 {
        use core::arch::x86_64::*;
        i32x8::from_m256i(t, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v.raw())))
    }

    /// Sign-extend lanes 8-15 of an i16x16 to a natural-order i32x8.
    #[rite(v3)]
    pub(crate) fn widen_hi(t: X64V3Token, v: V16) -> V32 {
        use core::arch::x86_64::*;
        i32x8::from_m256i(t, _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v.raw())))
    }

    /// `dst[j] = (dst[j] + res[j]).clamp(0, 255)` over 16 pixels. Exact: `res`
    /// is a `round_shift(_, 4)` of an i16, so |res| <= 2048 and dst <= 255 —
    /// the i16 lane add cannot wrap — and `packus_epi16`'s unsigned saturating
    /// narrow IS the [0, 255] pixel clamp.
    #[rite(v3)]
    pub(crate) fn add_store_u8(t: X64V3Token, res: V16, dst: &mut [u8; 16]) {
        use core::arch::x86_64::*;
        let d16 = _mm256_cvtepu8_epi16(u8x16::load(t, dst).raw());
        let sum = _mm256_add_epi16(res.raw(), d16);
        let packed =
            _mm_packus_epi16(_mm256_castsi256_si128(sum), _mm256_extracti128_si256::<1>(sum));
        u8x16::from_m128i(t, packed).store(dst);
    }
}

// ---------------------------------------------------------------------------
// Pattern C — aarch64 / NEON tier.
//
// `i16x16::Repr` is `[int16x8_t; 2]` and `i32x8::Repr` is `[int32x4_t; 2]`, so
// every 16-lane i16 op is two NEON instructions and every 16-lane i32 value is
// four. That is the SAME per-128-bit-register work as the AVX2 tier does per
// 256-bit register — the win over the i32x8 path is unchanged: one batch of 16
// where the i32 path needs two of 8, and (much more importantly) `half_btf`
// costs a widening multiply-accumulate instead of a widen-to-i64 round trip.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{i16x16, i32x8};
    use archmage::prelude::*;
    use magetypes::simd::generic::u8x16;

    type V16 = i16x16<NeonToken>;
    type V32 = i32x8<NeonToken>;

    /// The two `half_btf` operands, held UNinterleaved. AArch64's widening
    /// multiply-accumulate reads x and y from separate registers, so unlike
    /// the AVX2 `vpmaddwd` shape there is nothing to pre-interleave: this is a
    /// plain pair, and [`unpk16`] is free. It still exists as a type so the
    /// generated kernels — which memoize the (x, y) pair shared by two
    /// butterflies — stay ONE body across both tiers.
    #[derive(Clone, Copy)]
    pub(crate) struct Upk {
        x: [int16x8_t; 2],
        y: [int16x8_t; 2],
    }

    /// A 16-lane i32 value in NATURAL lane order: quarter `i` holds lanes
    /// `4i..4i+4`. (The AVX2 twin is in `vpunpck` order because its `madd`
    /// forces the interleave; NEON has no such constraint, so [`pack16`] here
    /// is a pure narrow with no permutation to undo.)
    #[derive(Clone, Copy)]
    pub(crate) struct P32([int32x4_t; 4]);

    /// Pair up two i16x16 values for the `madd`-equivalent butterflies. Free
    /// on this tier (see [`Upk`]).
    #[rite(neon)]
    pub(crate) fn unpk16(t: NeonToken, x: V16, y: V16) -> Upk {
        let _ = t;
        Upk { x: x.into_repr(), y: y.into_repr() }
    }

    /// `half_btf(w0, x, w1, y, 12)` on 16 lanes, x/y in the i16 domain —
    /// EXACT. `vmull_n_s16` / `vmlal_n_s16` are WIDENING 16x16->32 signed
    /// multiply(-accumulate): with |w| <= 4095 and |x|,|y| <= 2^15 each
    /// product is <= 2^27 and the sum <= 2^28, so nothing wraps and the exact
    /// products agree with the scalar port's `wrapping_mul` (which also cannot
    /// wrap here). `vrshrq_n_s32::<12>` is `(acc + 2^11) >> 12` with the add
    /// done in extended precision — exactly the scalar's
    /// `round_shift(_, 12)`, and the result (<= ~2^16.03) fits i32 so the i64
    /// shift and the i32 shift agree.
    #[rite(neon)]
    pub(crate) fn btf16(t: NeonToken, u: Upk, w0: i32, w1: i32) -> P32 {
        debug_assert!(w0.unsigned_abs() < (1 << 15) && w1.unsigned_abs() < (1 << 15));
        let (c0, c1) = (w0 as i16, w1 as i16);
        let half = |x: int16x8_t, y: int16x8_t| -> [int32x4_t; 2] {
            let lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(x), c0), vget_low_s16(y), c1);
            let hi = vmlal_high_n_s16(vmull_high_n_s16(x, c0), y, c1);
            [vrshrq_n_s32::<12>(lo), vrshrq_n_s32::<12>(hi)]
        };
        let _ = t;
        let a = half(u.x[0], u.y[0]);
        let b = half(u.x[1], u.y[1]);
        P32([a[0], a[1], b[0], b[1]])
    }

    /// `clamp_value(v, 16)` + narrow of a natural-order i32 quad:
    /// `vqmovn_s32`'s SATURATING narrow is exactly "clamp to [-2^15, 2^15-1]
    /// then truncate" == `clamp_value(_, 16)`, and the lane order is already
    /// natural.
    #[rite(neon)]
    pub(crate) fn pack16(t: NeonToken, p: P32) -> V16 {
        i16x16::from_repr(
            t,
            [
                vqmovn_high_s32(vqmovn_s32(p.0[0]), p.0[1]),
                vqmovn_high_s32(vqmovn_s32(p.0[2]), p.0[3]),
            ],
        )
    }

    /// Sign-extend an i16x16 into a natural-order i32 quad (for clamp adds
    /// that mix an i16 operand with a 17-bit transient) — `sshll`, a pure
    /// width extension.
    #[rite(neon)]
    pub(crate) fn ext16(t: NeonToken, v: V16) -> P32 {
        let _ = t;
        let a = v.into_repr();
        P32([
            vmovl_s16(vget_low_s16(a[0])),
            vmovl_high_s16(a[0]),
            vmovl_s16(vget_low_s16(a[1])),
            vmovl_high_s16(a[1]),
        ])
    }

    /// `clamp_value(a + b, 16)` for two i16-domain values: `vqaddq_s16`'s
    /// saturating add IS the normative clamp (a+b in [-2^16, 2^16-2]
    /// saturates to exactly `clamp_value`'s [-2^15, 2^15-1]).
    #[rite(neon)]
    pub(crate) fn sadd16(t: NeonToken, a: V16, b: V16) -> V16 {
        let (x, y) = (a.into_repr(), b.into_repr());
        i16x16::from_repr(t, [vqaddq_s16(x[0], y[0]), vqaddq_s16(x[1], y[1])])
    }

    /// `clamp_value(a - b, 16)` for two i16-domain values (also serves
    /// `clamp_value(-b + a, 16)` — identical in two's complement).
    #[rite(neon)]
    pub(crate) fn ssub16(t: NeonToken, a: V16, b: V16) -> V16 {
        let (x, y) = (a.into_repr(), b.into_repr());
        i16x16::from_repr(t, [vqsubq_s16(x[0], y[0]), vqsubq_s16(x[1], y[1])])
    }

    /// i32-quad add (exact: operands are <= 17-bit transients / sign-extended
    /// i16, so the lane add cannot wrap — identical to the scalar i32 add).
    #[rite(neon)]
    pub(crate) fn padd32(t: NeonToken, a: P32, b: P32) -> P32 {
        let _ = t;
        P32(core::array::from_fn(|i| vaddq_s32(a.0[i], b.0[i])))
    }

    /// i32-quad subtract (exact, same bound argument as [`padd32`]).
    #[rite(neon)]
    pub(crate) fn psub32(t: NeonToken, a: P32, b: P32) -> P32 {
        let _ = t;
        P32(core::array::from_fn(|i| vsubq_s32(a.0[i], b.0[i])))
    }

    /// `round_shift(v, bit)` on i16 lanes as `mulhrs(v, 2^(15-bit))`.
    /// `vqrdmulhq_s16` computes `(2*v*m + 2^15) >> 16` == the AVX2
    /// `_mm256_mulhrs_epi16`'s `((v*m >> 14) + 1) >> 1` for every i16 pair
    /// (proof in the module docs); its saturating case is unreachable for
    /// `m > 0`.
    #[rite(neon)]
    pub(crate) fn mulhrs16(t: NeonToken, v: V16, m: i16) -> V16 {
        let a = v.into_repr();
        let mv = vdupq_n_s16(m);
        i16x16::from_repr(t, [vqrdmulhq_s16(a[0], mv), vqrdmulhq_s16(a[1], mv)])
    }

    /// Gather+clamp: two natural-order i32x8 values -> `clamp_value(_, 16)`
    /// per lane via the saturating narrow, in natural lane order (a0-7 then
    /// b0-7).
    #[rite(neon)]
    pub(crate) fn pack_clamp16(t: NeonToken, a: V32, b: V32) -> V16 {
        let (x, y) = (a.into_repr(), b.into_repr());
        i16x16::from_repr(
            t,
            [vqmovn_high_s32(vqmovn_s32(x[0]), x[1]), vqmovn_high_s32(vqmovn_s32(y[0]), y[1])],
        )
    }

    /// Reverse all 16 lanes (lr_flip on a full column group). `vrev64q_s16`
    /// reverses within each 64-bit pair and `vextq_s16::<4>` swaps the pairs,
    /// which together reverse an 8-lane vector; swapping the two halves
    /// completes the 16-lane reversal.
    #[rite(neon)]
    pub(crate) fn rev16(t: NeonToken, v: V16) -> V16 {
        let a = v.into_repr();
        let rev8 = |x: int16x8_t| -> int16x8_t {
            let r = vrev64q_s16(x);
            vextq_s16::<4>(r, r)
        };
        i16x16::from_repr(t, [rev8(a[1]), rev8(a[0])])
    }

    /// Sign-extend lanes 0-7 of an i16x16 to a natural-order i32x8 (`sshll`) —
    /// a pure width extension, bit-exact.
    #[rite(neon)]
    pub(crate) fn widen_lo(t: NeonToken, v: V16) -> V32 {
        let a = v.into_repr();
        i32x8::from_repr(t, [vmovl_s16(vget_low_s16(a[0])), vmovl_high_s16(a[0])])
    }

    /// Sign-extend lanes 8-15 of an i16x16 to a natural-order i32x8.
    #[rite(neon)]
    pub(crate) fn widen_hi(t: NeonToken, v: V16) -> V32 {
        let a = v.into_repr();
        i32x8::from_repr(t, [vmovl_s16(vget_low_s16(a[1])), vmovl_high_s16(a[1])])
    }

    /// `dst[j] = (dst[j] + res[j]).clamp(0, 255)` over 16 pixels. Exact: `res`
    /// is a `round_shift(_, 4)` of an i16, so |res| <= 2048 and dst <= 255 —
    /// the i16 lane add cannot wrap — and `vqmovun_s16`'s signed->unsigned
    /// saturating narrow IS the [0, 255] pixel clamp.
    #[rite(neon)]
    pub(crate) fn add_store_u8(t: NeonToken, res: V16, dst: &mut [u8; 16]) {
        let d = u8x16::load(t, dst).into_repr();
        let d16 = [
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(d))),
            vreinterpretq_s16_u16(vmovl_high_u8(d)),
        ];
        let r = res.into_repr();
        let s0 = vaddq_s16(r[0], d16[0]);
        let s1 = vaddq_s16(r[1], d16[1]);
        u8x16::from_repr(t, vqmovun_high_s16(vqmovun_s16(s0), s1)).store(dst);
    }
}

// The two tiers export the SAME names; exactly one module is compiled, so every
// caller in `super::lowbd16` and in the generated i16 kernels writes
// `btf16(t, ..)` with no cfg and no suffix at the call site.
#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{
    add_store_u8, btf16, ext16, mulhrs16, pack16, pack_clamp16, padd32, psub32, rev16, sadd16,
    ssub16, unpk16, widen_hi, widen_lo,
};
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{
    add_store_u8, btf16, ext16, mulhrs16, pack16, pack_clamp16, padd32, psub32, rev16, sadd16,
    ssub16, unpk16, widen_hi, widen_lo,
};
