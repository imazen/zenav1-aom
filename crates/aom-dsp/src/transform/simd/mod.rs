//! SIMD (Gate 3) for the transform stack — lane-batched 1-D kernels + the
//! 2-D drivers' vector passes, bit-identical to the scalar port per lane.
//!
//! # One body per kernel, every tier
//!
//! Nothing in this module is written twice per architecture. Every kernel and
//! every pass driver is ONE `#[magetypes(define(i32x8), v3, neon, -scalar)]`
//! body: the macro emits the per-tier variants (`_v3` = AVX2, `_neon`) with
//! their own `#[target_feature]`, `Token` and `i32x8` are substituted per tier,
//! and `incant!` resolves each call to the tier-matching callee at compile time
//! (zero dispatcher hops once inside a tier body). `-scalar` drops the macro's
//! default scalar fallback, because the scalar twin already exists as the
//! transcribed port — which is exactly what the differentials compare against.
//!
//! The lane width is 8 on every target. On x86-64 that is one AVX2 register;
//! on aarch64 magetypes' `i32x8` is the 2×NEON polyfill (`Repr =
//! [int32x4_t; 2]`). Keeping the width identical across tiers is what lets the
//! drivers and the ~4,200 generated lines of 1-D kernels stay a single body.
//!
//! The handful of operations the generic magetypes API cannot express —
//! integer WIDENING (i32 → i64, verified absent at magetypes 0.9.28: the
//! `cross_width` raising/lowering is f32-only) and cross-lane PERMUTES — live
//! in [`prims`] as hand-written per-tier variants under one cfg-selected name.
//! That module is the WHOLE architecture-dependent surface of the transform
//! SIMD; its docs carry the per-tier exactness argument.
//!
//! # Shape (from the STATUS.md transform-SIMD design)
//!
//! Vectorize ACROSS independent 1-D transforms: the inverse 2-D driver's
//! COLUMN pass processes 8 adjacent columns as `i32x8` lanes — `buf[r*col_n +
//! c..c+8]` is a contiguous 8-lane load per row, NO transposes. The 1-D
//! kernel bodies are mechanical lane rewrites of the generated scalar
//! butterflies (`inv1d_v3_gen.rs`, emitted by `transpile_txfm1d.py --lanes`),
//! and the pass wrapper reproduces the driver's clamp / round-shift /
//! flip / clip-add stages lane-wise. Dispatch is per `func_col`: blocks
//! whose column kernel is in the ported set take the vector path, everything
//! else keeps the scalar per-column loop (byte-untouched, and the
//! `AOM_FORCE_SCALAR` pin routes everything there).
//!
//! # Bit-exactness argument (FULL i32 domain — stronger than the driver
//! clamp bounds; pinned by `tests` below at every token permutation)
//!
//! Every lane op reproduces the scalar op's exact semantics for ANY i32
//! input, so SIMD == scalar unconditionally (no domain reasoning needed):
//! * wrapping add/sub: magetypes `+`/`-` are wrapping on every backend,
//!   and `-a + b == b - a` in two's complement (the transpiler emits the
//!   latter).
//! * `clamp_value(v, bit)`: `bit <= 0` and `bit >= 32` are identities in
//!   the scalar port (the i64 bounds cover all of i32 at `bit == 32`); for
//!   `1..=31` the bounds are i32-representable → lane min/max. [`clampv`]
//! * `half_btf` — THE exactness trap: the scalar port wraps each PRODUCT in
//!   i32 (`w.wrapping_mul(in)`, matching C's int multiply) but sums the two
//!   products + rounding in **i64**. At driver clamp bounds a product
//!   reaches 2^32 and the sum needs 33 bits, so an i32-lane sum (libaom's
//!   own SSE4/AVX2 shape) diverges on crafted-but-decodable streams. [`hb`]
//!   ([`prims::hb`]) reproduces the i64 sum exactly on both tiers — the
//!   products wrap in i32, the sum and rounding happen in i64. Each tier
//!   reaches that differently (AVX2 has no `vpsraq` and needs a
//!   logical-shift + low-dword identity; AArch64 has a real 64-bit signed
//!   shift), which is precisely why `hb` is per-tier; see the [`prims`] docs.
//! * `round_shift(v as i64, bit)` (the positive-bit `round_shift_array`
//!   arm): the same widen → add rounding → shift → truncate recipe.
//!   [`prims::rshiftv`]
//! * `highbd_clip_pixel_add`: the i32 lane add wraps like the scalar
//!   `wrapping_add`; clamp to `[0, (1<<bd)-1]` is lane min/max; the `as u16`
//!   narrowing is exact after the clamp.
//! * `lr_flip` lane reversal and `ud_flip` row reversal are pure index
//!   permutations ([`prims::revv`] / loop order), identical to the scalar
//!   loops.
//!
//! Both tiers reach the inexpressible ops through raw value intrinsics inside
//! a `#[rite]` `#[target_feature]` region, so `#![forbid(unsafe_code)]` holds.

mod hand_v3;
mod inv1d_v3_gen;
// The bd8 i16-lane row/column specialization — 16 lanes per vector where the
// i32 pass gets 8. Cross-architecture since 2026-07-28: like everything else
// here it is ONE `#[magetypes]` body per kernel/driver, with the ops the
// generic magetypes API cannot express (saturating add/sub, saturating narrow,
// widening multiply-accumulate, lane reverse — audited absent, see the
// `prims16` docs) supplied per tier by `prims16`.
mod inv1d_v3_i16_gen;
mod lowbd16;
// The FORWARD half of the same lane-width programme. It is gated differently —
// the forward kernels carry no `clamp_value`, so nothing bounds their values
// except the input, and the contract is a per-kernel input BOUND proved by
// `xtask/audit_i16_fwd.py` and checked at runtime rather than a static domain
// statement. See `lowbd16_fwd`'s module docs.
mod fwd1d_v3_i16_gen;
mod lowbd16_fwd;
pub(crate) mod prims;
mod prims16;
mod txfm1d_v3_gen;

use archmage::prelude::*;
use magetypes::simd::generic::i32x8 as I32x8;

use crate::transform::cospi::{NEW_INV_SQRT2, NEW_SQRT2, NEW_SQRT2_BITS};
use prims::{clampv, mul_rshiftv, revv, rshiftv, shl_clamp64v, transpose8, widen16};

// The 1-D kernels are `#[magetypes]` FAMILIES: each name below exists once per
// tier (`av1_idct4_impl_v3`, `av1_idct4_impl_neon`, …) and is reached through
// `incant!` from inside a tier body, which rewrites to the matching variant at
// compile time. Glob-import so the per-tier names resolve without spelling all
// 25 × 2 of them; the three modules' kernel names are disjoint.
use hand_v3::*;
use inv1d_v3_gen::*;
use txfm1d_v3_gen::*;

/// Does a HALF-WIDTH lane batch (4 active lanes of 8) pay off here?
///
/// A transform whose vectorized dimension is 4 runs ONE batch with half the
/// lanes idle, and its strided side degrades from 8x8 transposes to per-lane
/// gather/scatter. Both costs are FIXED per batch, so whether they are repaid
/// depends on how much work the batch does — which is the OTHER dimension,
/// `kernel_points` (the 1-D kernel's point count). Hence a predicate, not a
/// flag.
///
/// Whether it pays is also per-architecture, because the thing it has to beat
/// is not equally fast everywhere: on aarch64 `neon` is a compile-time
/// baseline, so LLVM already auto-vectorizes the scalar driver loop.
///
/// MEASURED 2026-07-25, Apple M4 Pro, port-only `dsp_kernels` bench, before =
/// 4b92e2b (no vector path on aarch64 at all) —
/// `benchmarks/dsp_neon_transform_2026-07-25.md`:
///
/// | cell (col_n x row_n) | half batch ON | OFF |
/// |---|---|---|
/// | `inv_txfm_u8::04x04_adst` (4x4)  | **+26.6%** | +4.3% |
/// | `inv_txfm_u8::04x16_dct`  (4x16) | **-36.2%** | -15.6% |
/// | `inv_txfm_u8::04x16_adst` (4x16) | **-29.3%** | -7.8% |
/// | `inv_txfm_hbd10::04x16`   (4x16) | **-40.1%** | -20.0% |
///
/// So at 4 points the half batch loses badly and at 16 it is the single
/// biggest win in the 4-wide column — the threshold below is the boundary
/// between those two measurements.
///
/// **`kernel_points == 8` is now MEASURED, 2026-07-31** (Apple M4 Pro, same
/// port-only `dsp_kernels` bench; `benchmarks/dsp_neon_half_batch_4x8_2026-07-31.md`).
/// This rung used to be INTERPOLATED — the grid had no 4x8 cell, and the note
/// here said so and asked for one before relying on it. `TX_4X8`/`TX_8X4` cells
/// were added and the threshold A/B'd by flipping it to `>= 16`:
///
/// | cell | half batch ON (`>= 8`) | OFF (`>= 16`) | ON is |
/// |---|---|---|---|
/// | `inv_txfm_u8::04x08_dct`  | 216.1 us | 273.4 us | **21.0% faster** |
/// | `inv_txfm_u8::04x08_adst` | 248.0 us | 317.6 us | **21.9% faster** |
/// | `inv_txfm_u8::08x04_dct`  | 208.6 us | 253.0 us | **17.6% faster** |
/// | `inv_txfm_u8::08x04_adst` | 234.4 us | 296.1 us | **20.8% faster** |
///
/// The interpolation held: 8 is on the paying side, and the mechanism's
/// predicted monotonicity (4 loses, 8 pays, 16 pays more) is what the numbers
/// show. Every row is far outside the +-2% run-to-run band. Note `08x04` moves
/// too — TX_8X4's other pass is an 8-point kernel, so it sits on the same rung.
///
/// x86-64 always says yes: the 4-wide arms are the shape the 2026-07-17 AVX2
/// landing measured and kept (`benchmarks/gate3_transform_simd_2026-07-17.md`),
/// and nothing here re-measured them on an AVX2 box.
#[inline]
fn half_batch_pays(kernel_points: usize) -> bool {
    if cfg!(target_arch = "aarch64") { kernel_points >= 8 } else { true }
}

/// 1-D kernel selector — TXFM_TYPE ids 0..=11 (DCT4..64, ADST4/8/16,
/// IDTX4/8/16/32), one enum per direction. ALL 12 are ported in each
/// direction; the `Option` maps stay for unknown-id safety (→ scalar loop).
#[derive(Clone, Copy)]
enum Inv1d {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
    Adst4,
    Adst8,
    Adst16,
    Idtx4,
    Idtx8,
    Idtx16,
    Idtx32,
}

fn inv_kernel(txfm_type: i32) -> Option<Inv1d> {
    match txfm_type {
        0 => Some(Inv1d::Dct4),
        1 => Some(Inv1d::Dct8),
        2 => Some(Inv1d::Dct16),
        3 => Some(Inv1d::Dct32),
        4 => Some(Inv1d::Dct64),
        5 => Some(Inv1d::Adst4),
        6 => Some(Inv1d::Adst8),
        7 => Some(Inv1d::Adst16),
        8 => Some(Inv1d::Idtx4),
        9 => Some(Inv1d::Idtx8),
        10 => Some(Inv1d::Idtx16),
        11 => Some(Inv1d::Idtx32),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum Fwd1d {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
    Adst4,
    Adst8,
    Adst16,
    Idtx4,
    Idtx8,
    Idtx16,
    Idtx32,
}

fn fwd_kernel(txfm_type: i32) -> Option<Fwd1d> {
    match txfm_type {
        0 => Some(Fwd1d::Dct4),
        1 => Some(Fwd1d::Dct8),
        2 => Some(Fwd1d::Dct16),
        3 => Some(Fwd1d::Dct32),
        4 => Some(Fwd1d::Dct64),
        5 => Some(Fwd1d::Adst4),
        6 => Some(Fwd1d::Adst8),
        7 => Some(Fwd1d::Adst16),
        8 => Some(Fwd1d::Idtx4),
        9 => Some(Fwd1d::Idtx8),
        10 => Some(Fwd1d::Idtx16),
        11 => Some(Fwd1d::Idtx32),
        _ => None,
    }
}

/// The kernel's point count (== how many input/output vectors it consumes).
fn inv_kernel_n(k: Inv1d) -> usize {
    match k {
        Inv1d::Dct4 | Inv1d::Adst4 | Inv1d::Idtx4 => 4,
        Inv1d::Dct8 | Inv1d::Adst8 | Inv1d::Idtx8 => 8,
        Inv1d::Dct16 | Inv1d::Adst16 | Inv1d::Idtx16 => 16,
        Inv1d::Dct32 | Inv1d::Idtx32 => 32,
        Inv1d::Dct64 => 64,
    }
}

/// The forward kernel's point count (== how many input/output vectors it
/// consumes) — the symmetric twin of [`inv_kernel_n`].
fn fwd_kernel_n(k: Fwd1d) -> usize {
    match k {
        Fwd1d::Dct4 | Fwd1d::Adst4 | Fwd1d::Idtx4 => 4,
        Fwd1d::Dct8 | Fwd1d::Adst8 | Fwd1d::Idtx8 => 8,
        Fwd1d::Dct16 | Fwd1d::Adst16 | Fwd1d::Idtx16 => 16,
        Fwd1d::Dct32 | Fwd1d::Idtx32 => 32,
        Fwd1d::Dct64 => 64,
    }
}

/// Direct-dispatch the selected inverse 1-D lane kernel. `incant!` inside a
/// tier body rewrites to the tier-matching variant at COMPILE time (no
/// dispatcher branch, no cache probe — the callee inlines into this function's
/// `#[target_feature]` region), which is also why the kernels cannot be stored
/// as plain fn pointers: they are `#[target_feature]` fns.
#[magetypes(define(i32x8), v3, neon, -scalar)]
fn run_inv1d(
    t: Token,
    k: Inv1d,
    input: &[I32x8<Token>],
    out: &mut [I32x8<Token>],
    cos_bit: i32,
    stage_range: &[i8],
) {
    match k {
        Inv1d::Dct4 => incant!(av1_idct4_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Dct8 => incant!(av1_idct8_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Dct16 => incant!(av1_idct16_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Dct32 => incant!(av1_idct32_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Dct64 => incant!(av1_idct64_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Adst4 => incant!(av1_iadst4_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Adst8 => incant!(av1_iadst8_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Adst16 => incant!(av1_iadst16_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Idtx4 => incant!(av1_iidentity4_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Idtx8 => incant!(av1_iidentity8_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Idtx16 => incant!(av1_iidentity16_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Inv1d::Idtx32 => incant!(av1_iidentity32_impl(input, out, cos_bit, stage_range), [v3, neon]),
    }
}

#[magetypes(define(i32x8), v3, neon, -scalar)]
fn run_fwd1d(
    t: Token,
    k: Fwd1d,
    input: &[I32x8<Token>],
    out: &mut [I32x8<Token>],
    cos_bit: i32,
    stage_range: &[i8],
) {
    match k {
        Fwd1d::Dct4 => incant!(av1_fdct4_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Dct8 => incant!(av1_fdct8_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Dct16 => incant!(av1_fdct16_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Dct32 => incant!(av1_fdct32_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Dct64 => incant!(av1_fdct64_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Adst4 => incant!(av1_fadst4_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Adst8 => incant!(av1_fadst8_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Adst16 => incant!(av1_fadst16_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Idtx4 => incant!(av1_fidentity4_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Idtx8 => incant!(av1_fidentity8_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Idtx16 => incant!(av1_fidentity16_impl(input, out, cos_bit, stage_range), [v3, neon]),
        Fwd1d::Idtx32 => incant!(av1_fidentity32_impl(input, out, cos_bit, stage_range), [v3, neon]),
    }
}

/// Vector column pass of `av1_inv_txfm2d_add` — 8 columns per group.
/// Returns `false` (the caller runs the scalar loop) when the column kernel
/// isn't ported, the width has no full 8-column groups, or SIMD is
/// unavailable / pinned off. On `true` the pass is complete, bit-identical
/// to the scalar loop (module-docs argument + the `tests` differential).
#[allow(clippy::too_many_arguments)]
/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn inv_rect48_fused_scalar(
    _t: archmage::ScalarToken,
    _kr: Inv1d,
    _kc: Inv1d,
    _input: &[i32],
    _output: &mut [u16],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _row_clamp: i8,
    _col_clamp: i8,
    _sr_row: &[i8; 12],
    _sr_col: &[i8; 12],
    _ud_flip: bool,
    _lr_flip: bool,
    _bd: i32,
) -> bool {
    false
}

/// **The fused 4x8 / 8x4 INVERSE transforms** — KB-PERF-27's twin, **15.16 % of
/// inverse transforms** at the shipping preset.
///
/// `INV_SHIFT` is `[0, -4]` for both sizes, so **there is no row shift at all**
/// (the generic's `round_shift_array(_, -shift[0])` early-returns at 0) —
/// simpler than 8x8's, which shifts by 1. The column shift is again 4.
///
/// The rect scaling is `NEW_INV_SQRT2` and it lands in the ROW pass **before**
/// the clamp, mirroring the driver's own order:
/// `ti[c] = round_shift(input * NEW_INV_SQRT2, NEW_SQRT2_BITS)` then
/// `clamp_buf(ti, bd + 8)`.
///
/// **`lr_flip` is a LANE reverse here, not the array-order reversal
/// KB-PERF-27's forward uses** — and the asymmetry is real rather than an
/// oversight. The forward's flip permutes which column POSITION a result
/// occupies before the row pass reads positions in order; the inverse's reads
/// `buf[r][col_n-1-c]` for output column `c`, i.e. it permutes across LANES of
/// one vector. At `col_n == 4` that is a reverse of the four LIVE lanes inside
/// an eight-lane vector, which `revv` does not do, so the flipped path goes
/// through `to_array` — paid only on FLIPADST types.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_rect48_fused(
    t: Token,
    kr: Inv1d,
    kc: Inv1d,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;
    if !((col_n == 4 && row_n == 8) || (col_n == 8 && row_n == 4)) {
        return false;
    }

    // ---- row pass: lane = r, contiguous (input is column-major) ----
    let mut k = [i32x8::zero(t); 8];
    for c in 0..col_n {
        let base = c * row_n;
        // KB-PERF-31: at 4x8 a whole vector of coefficients is contiguous, so
        // this is a plain `from_slice` — the padded per-lane build is needed
        // only for the 8x4 shape, whose column is four coefficients long.
        let v = if row_n == 8 {
            match input.get(base..base + 8) {
                Some(sl) => i32x8::from_slice(t, sl),
                None => return false,
            }
        } else {
            let a: [i32; 4] = match input.get(base..base + 4).and_then(|s| s.try_into().ok()) {
                Some(a) => a,
                None => return false,
            };
            i32x8::from_array(t, core::array::from_fn(|j| if j < 4 { a[j] } else { 0 }))
        };
        // rect_type == +-1: the NEW_INV_SQRT2 scaling, BEFORE the clamp.
        let v = mul_rshiftv(t, v, NEW_INV_SQRT2, NEW_SQRT2_BITS);
        k[c] = clampv(t, v, row_clamp);
    }
    let mut w = [i32x8::zero(t); 8];
    incant!(run_inv1d(kr, &k[..col_n], &mut w[..col_n], cos_bit, sr_row), [v3]);
    // shift[0] == 0 -> `round_shift_array` early-returns; nothing to do.

    // ---- transpose -> lane = column ----
    let r8: [__m256i; 8] = core::array::from_fn(|i| w[i].into_repr());
    let a0 = _mm256_unpacklo_epi32(r8[0], r8[1]);
    let a1 = _mm256_unpackhi_epi32(r8[0], r8[1]);
    let a2 = _mm256_unpacklo_epi32(r8[2], r8[3]);
    let a3 = _mm256_unpackhi_epi32(r8[2], r8[3]);
    let a4 = _mm256_unpacklo_epi32(r8[4], r8[5]);
    let a5 = _mm256_unpackhi_epi32(r8[4], r8[5]);
    let a6 = _mm256_unpacklo_epi32(r8[6], r8[7]);
    let a7 = _mm256_unpackhi_epi32(r8[6], r8[7]);
    let b0 = _mm256_unpacklo_epi64(a0, a2);
    let b1 = _mm256_unpackhi_epi64(a0, a2);
    let b2 = _mm256_unpacklo_epi64(a1, a3);
    let b3 = _mm256_unpackhi_epi64(a1, a3);
    let b4 = _mm256_unpacklo_epi64(a4, a6);
    let b5 = _mm256_unpackhi_epi64(a4, a6);
    let b6 = _mm256_unpacklo_epi64(a5, a7);
    let b7 = _mm256_unpackhi_epi64(a5, a7);
    let tr: [i32x8; 8] = [
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
    ];

    // ---- column pass: lane = output column ----
    let mut ci = [i32x8::zero(t); 8];
    for r in 0..row_n {
        // KB-PERF-31: at 8x4 all eight lanes are live, so the flip is the same
        // whole-vector `revv` every unpadded inverse uses; only the 4-wide
        // shape needs the array round trip, because reversing four LIVE lanes
        // inside an eight-lane vector is not what `revv` does.
        let v = if lr_flip {
            if col_n == 8 {
                revv(t, tr[r])
            } else {
                let a = tr[r].to_array();
                i32x8::from_array(t, core::array::from_fn(|j| if j < col_n { a[col_n - 1 - j] } else { 0 }))
            }
        } else {
            tr[r]
        };
        ci[r] = clampv(t, v, col_clamp);
    }
    let mut co = [i32x8::zero(t); 8];
    incant!(run_inv1d(kc, &ci[..row_n], &mut co[..row_n], cos_bit, sr_col), [v3]);
    for x in co[..row_n].iter_mut() {
        *x = rshiftv(t, *x, 4); // -shift[1], shift[1] == -4
    }

    // ---- reconstruction ----
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, (1i32 << bd) - 1);
    for r in 0..row_n {
        let src = co[if ud_flip { row_n - 1 - r } else { r }];
        let idx = r * stride;
        // KB-PERF-31: at 8x4 the destination row is a whole vector — the same
        // fixed-size read the unpadded inverses use, so the bounds check is one
        // per row rather than one per lane. The 4-wide shape keeps a 4-element
        // fixed-size read for the same reason; it is a tail, not a slice walk.
        if col_n == 8 {
            let d: [u16; 8] = match output.get(idx..idx + 8).and_then(|s| s.try_into().ok()) {
                Some(d) => d,
                None => return false,
            };
            let dv = i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32));
            let s = (dv + src).clamp(zero, pix_hi).to_array();
            for j in 0..8 {
                output[idx + j] = s[j] as u16;
            }
        } else {
            let d: [u16; 4] = match output.get(idx..idx + 4).and_then(|s| s.try_into().ok()) {
                Some(d) => d,
                None => return false,
            };
            let dv = i32x8::from_array(t, core::array::from_fn(|j| if j < 4 { d[j] as i32 } else { 0 }));
            let s = (dv + src).clamp(zero, pix_hi).to_array();
            for j in 0..4 {
                output[idx + j] = s[j] as u16;
            }
        }
    }
    true
}

/// Dispatch for [`inv_rect48_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_txfm2d_rect48_fused(
    txfm_type_row: i32,
    txfm_type_col: i32,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    let (Some(kr), Some(kc)) = (inv_kernel(txfm_type_row), inv_kernel(txfm_type_col)) else {
        return false;
    };
    if inv_kernel_n(kr) != col_n || inv_kernel_n(kc) != row_n {
        return false;
    }
    incant!(
        inv_rect48_fused(
            kr, kc, input, output, stride, col_n, row_n, row_clamp, col_clamp, sr_row, sr_col,
            ud_flip, lr_flip, bd
        ),
        [v3, scalar]
    )
}

/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn inv_rect816_fused_scalar(
    _t: archmage::ScalarToken,
    _kr: Inv1d,
    _kc: Inv1d,
    _input: &[i32],
    _output: &mut [u16],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _row_clamp: i8,
    _col_clamp: i8,
    _sr_row: &[i8; 12],
    _sr_col: &[i8; 12],
    _ud_flip: bool,
    _lr_flip: bool,
    _bd: i32,
) -> bool {
    false
}

/// **The fused 8x16 / 16x8 INVERSE transforms** — KB-PERF-29's twin, **9.52 %
/// of inverse transforms** at the shipping preset.
///
/// Written over GROUPS exactly like the forward: `CG = col_n / 8` column groups
/// and `RG = row_n / 8` row groups, so one body covers both shapes and matches
/// the landed 8x8 (1,1) and 16x16 (2,2) inverses. **Both dimensions are >= 8,
/// so there is no padding and none of the per-lane `from_array`/`to_array` glue
/// KB-PERF-28 measured as the cause of its shortfall** — every coefficient load
/// is `from_slice`.
///
/// `INV_SHIFT` is `[-1, -4]` for both sizes — 8x8's recipe, not 4x8/8x4's
/// (whose row shift vanishes) and not 16x16's (whose row shift is 2). The row
/// pass therefore ends in a rounding shift by 1, the column pass by 4.
///
/// `rect_type == +-1`, so the row pass carries the `NEW_INV_SQRT2` scaling
/// **before** the clamp, mirroring the driver's own order.
///
/// `lr_flip` is the LANE reversal the inverse side uses (output column `c`
/// reads `buf` column `col_n - 1 - c`), which at `CG == 2` makes the two column
/// groups **exchange as well as reverse** — the shape the fused 16x16 inverse
/// already had, generalised over `CG`.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_rect816_fused(
    t: Token,
    kr: Inv1d,
    kc: Inv1d,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;
    if !((col_n == 8 && row_n == 16) || (col_n == 16 && row_n == 8)) {
        return false;
    }
    let cg_n = col_n / 8;
    let rg_n = row_n / 8;

    let tr8 = |m: &[i32x8]| -> [i32x8; 8] {
        let r: [__m256i; 8] = core::array::from_fn(|i| m[i].into_repr());
        let a0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let a1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let a2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let a3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let a4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let a5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let a6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let a7 = _mm256_unpackhi_epi32(r[6], r[7]);
        let b0 = _mm256_unpacklo_epi64(a0, a2);
        let b1 = _mm256_unpackhi_epi64(a0, a2);
        let b2 = _mm256_unpacklo_epi64(a1, a3);
        let b3 = _mm256_unpackhi_epi64(a1, a3);
        let b4 = _mm256_unpacklo_epi64(a4, a6);
        let b5 = _mm256_unpackhi_epi64(a4, a6);
        let b6 = _mm256_unpacklo_epi64(a5, a7);
        let b7 = _mm256_unpackhi_epi64(a5, a7);
        [
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
        ]
    };

    // ---- row pass: lane = r within a row group; the input is column-major, so
    // each (column, row group) is a contiguous 8-element run.
    let mut w = [[i32x8::zero(t); 16]; 2]; // w[rg][c], lane = r within group rg
    for rg in 0..rg_n {
        let mut k = [i32x8::zero(t); 16];
        for c in 0..col_n {
            let base = c * row_n + rg * 8;
            let sl = match input.get(base..base + 8) {
                Some(sl) => sl,
                None => return false,
            };
            let v = i32x8::from_slice(t, sl);
            // rect_type == +-1: the NEW_INV_SQRT2 scaling, BEFORE the clamp.
            let v = mul_rshiftv(t, v, NEW_INV_SQRT2, NEW_SQRT2_BITS);
            k[c] = clampv(t, v, row_clamp);
        }
        let mut o = [i32x8::zero(t); 16];
        incant!(run_inv1d(kr, &k[..col_n], &mut o[..col_n], cos_bit, sr_row), [v3]);
        for c in 0..col_n {
            w[rg][c] = rshiftv(t, o[c], 1); // -shift[0], shift[0] == -1
        }
    }

    // ---- transpose: CG * RG blocks -> lane = column.
    // `tt[rg][cg * 8 + k]` is row `rg * 8 + k`, columns `cg * 8 .. cg * 8 + 8`.
    let mut tt = [[i32x8::zero(t); 16]; 2];
    for rg in 0..rg_n {
        for cg in 0..cg_n {
            let blk = tr8(&w[rg][cg * 8..cg * 8 + 8]);
            for k in 0..8 {
                tt[rg][cg * 8 + k] = blk[k];
            }
        }
    }

    // ---- column pass: lane = output column, one call per column group ----
    let mut co = [[i32x8::zero(t); 16]; 2]; // co[cg][r]
    for cg in 0..cg_n {
        // Output column c reads buf column col_n-1-c under lr_flip, so the
        // groups exchange AND each group's lanes reverse.
        let sg = if lr_flip { cg_n - 1 - cg } else { cg };
        let mut ci = [i32x8::zero(t); 16];
        for r in 0..row_n {
            let v = tt[r / 8][sg * 8 + (r % 8)];
            let v = if lr_flip { revv(t, v) } else { v };
            ci[r] = clampv(t, v, col_clamp);
        }
        let mut o = [i32x8::zero(t); 16];
        incant!(run_inv1d(kc, &ci[..row_n], &mut o[..row_n], cos_bit, sr_col), [v3]);
        for r in 0..row_n {
            co[cg][r] = rshiftv(t, o[r], 4); // -shift[1], shift[1] == -4
        }
    }

    // ---- reconstruction ----
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, (1i32 << bd) - 1);
    for r in 0..row_n {
        let src_i = if ud_flip { row_n - 1 - r } else { r };
        let idx = r * stride;
        for cg in 0..cg_n {
            let o0 = idx + cg * 8;
            let d: [u16; 8] = match output.get(o0..o0 + 8).and_then(|s| s.try_into().ok()) {
                Some(d) => d,
                None => return false,
            };
            let dv = i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32));
            let s = (dv + co[cg][src_i]).clamp(zero, pix_hi).to_array();
            for j in 0..8 {
                output[o0 + j] = s[j] as u16;
            }
        }
    }
    true
}

/// Dispatch for [`inv_rect816_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_txfm2d_rect816_fused(
    txfm_type_row: i32,
    txfm_type_col: i32,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    let (Some(kr), Some(kc)) = (inv_kernel(txfm_type_row), inv_kernel(txfm_type_col)) else {
        return false;
    };
    if inv_kernel_n(kr) != col_n || inv_kernel_n(kc) != row_n {
        return false;
    }
    incant!(
        inv_rect816_fused(
            kr, kc, input, output, stride, col_n, row_n, row_clamp, col_clamp, sr_row, sr_col,
            ud_flip, lr_flip, bd
        ),
        [v3, scalar]
    )
}

/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn inv_16x16_fused_scalar(
    _t: archmage::ScalarToken,
    _kr: Inv1d,
    _kc: Inv1d,
    _input: &[i32],
    _output: &mut [u16],
    _stride: usize,
    _row_clamp: i8,
    _col_clamp: i8,
    _sr_row: &[i8; 12],
    _sr_col: &[i8; 12],
    _ud_flip: bool,
    _lr_flip: bool,
    _bd: i32,
) -> bool {
    false
}

/// **The fused 16x16 INVERSE transform** — KB-PERF-24 at twice the width, with
/// KB-PERF-25's two-vectors-per-row structure.
///
/// `INV_SHIFT[TX_16X16] = [-2, -4]`, so the row shift is **2** (8x8's is 1) and
/// the column shift is again 4. As at 8x8 the input is COLUMN-major, so the row
/// pass loads contiguously with lane = r and needs no transpose; the four 8x8
/// block transposes sit between the passes.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_16x16_fused(
    t: Token,
    kr: Inv1d,
    kc: Inv1d,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;

    let tr8 = |m: &[i32x8]| -> [i32x8; 8] {
        let r: [__m256i; 8] = core::array::from_fn(|i| m[i].into_repr());
        let a0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let a1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let a2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let a3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let a4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let a5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let a6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let a7 = _mm256_unpackhi_epi32(r[6], r[7]);
        let b0 = _mm256_unpacklo_epi64(a0, a2);
        let b1 = _mm256_unpackhi_epi64(a0, a2);
        let b2 = _mm256_unpacklo_epi64(a1, a3);
        let b3 = _mm256_unpackhi_epi64(a1, a3);
        let b4 = _mm256_unpacklo_epi64(a4, a6);
        let b5 = _mm256_unpackhi_epi64(a4, a6);
        let b6 = _mm256_unpacklo_epi64(a5, a7);
        let b7 = _mm256_unpackhi_epi64(a5, a7);
        [
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
        ]
    };

    // ---- row pass: lane = r, contiguous loads (input is column-major) ----
    let mut ka = [i32x8::zero(t); 16]; // rows 0-7
    let mut kb = [i32x8::zero(t); 16]; // rows 8-15
    for c in 0..16usize {
        let base = c * 16;
        let (a, b) = match (input.get(base..base + 8), input.get(base + 8..base + 16)) {
            (Some(a), Some(b)) => (a, b),
            _ => return false,
        };
        ka[c] = clampv(t, i32x8::from_slice(t, a), row_clamp);
        kb[c] = clampv(t, i32x8::from_slice(t, b), row_clamp);
    }
    let mut wa = [i32x8::zero(t); 16];
    let mut wb = [i32x8::zero(t); 16];
    incant!(run_inv1d(kr, &ka, &mut wa, cos_bit, sr_row), [v3]);
    incant!(run_inv1d(kr, &kb, &mut wb, cos_bit, sr_row), [v3]);
    for i in 0..16usize {
        wa[i] = rshiftv(t, wa[i], 2); // -shift[0], shift[0] == -2
        wb[i] = rshiftv(t, wb[i], 2);
    }

    // ---- transpose -> lane = column ----
    let pa = tr8(&wa[0..8]); //  rows 0-7,  cols 0-7
    let pb = tr8(&wa[8..16]); // rows 0-7,  cols 8-15
    let pc = tr8(&wb[0..8]); //  rows 8-15, cols 0-7
    let pd = tr8(&wb[8..16]); // rows 8-15, cols 8-15

    // Column pass input: `ci_lo[r]` holds columns 0-7 of row r, `ci_hi[r]`
    // columns 8-15. Under lr_flip output column c reads buf column 15-c, so the
    // halves exchange AND reverse — the same shape as the fused 16x16 forward.
    let mut cilo = [i32x8::zero(t); 16];
    let mut cihi = [i32x8::zero(t); 16];
    for r in 0..16usize {
        let (l, h) = if r < 8 { (pa[r], pb[r]) } else { (pc[r - 8], pd[r - 8]) };
        let (l, h) = if lr_flip { (revv(t, h), revv(t, l)) } else { (l, h) };
        cilo[r] = clampv(t, l, col_clamp);
        cihi[r] = clampv(t, h, col_clamp);
    }
    let mut colo = [i32x8::zero(t); 16];
    let mut cohi = [i32x8::zero(t); 16];
    incant!(run_inv1d(kc, &cilo, &mut colo, cos_bit, sr_col), [v3]);
    incant!(run_inv1d(kc, &cihi, &mut cohi, cos_bit, sr_col), [v3]);
    for i in 0..16usize {
        colo[i] = rshiftv(t, colo[i], 4); // -shift[1], shift[1] == -4
        cohi[i] = rshiftv(t, cohi[i], 4);
    }

    // ---- reconstruction ----
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, (1i32 << bd) - 1);
    for r in 0..16usize {
        let src_i = if ud_flip { 15 - r } else { r };
        let idx = r * stride;
        let d: [u16; 16] = match output.get(idx..idx + 16).and_then(|s| s.try_into().ok()) {
            Some(d) => d,
            None => return false,
        };
        let dl = i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32));
        let dh = i32x8::from_array(t, core::array::from_fn(|j| d[j + 8] as i32));
        let sl = (dl + colo[src_i]).clamp(zero, pix_hi).to_array();
        let sh = (dh + cohi[src_i]).clamp(zero, pix_hi).to_array();
        for j in 0..8 {
            output[idx + j] = sl[j] as u16;
            output[idx + 8 + j] = sh[j] as u16;
        }
    }
    true
}

/// Dispatch for [`inv_16x16_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_txfm2d_16x16_fused(
    txfm_type_row: i32,
    txfm_type_col: i32,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    let (Some(kr), Some(kc)) = (inv_kernel(txfm_type_row), inv_kernel(txfm_type_col)) else {
        return false;
    };
    if inv_kernel_n(kr) != 16 || inv_kernel_n(kc) != 16 {
        return false;
    }
    incant!(
        inv_16x16_fused(
            kr, kc, input, output, stride, row_clamp, col_clamp, sr_row, sr_col, ud_flip, lr_flip,
            bd
        ),
        [v3, scalar]
    )
}

/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn inv_8x8_fused_scalar(
    _t: archmage::ScalarToken,
    _kr: Inv1d,
    _kc: Inv1d,
    _input: &[i32],
    _output: &mut [u16],
    _stride: usize,
    _row_clamp: i8,
    _col_clamp: i8,
    _sr_row: &[i8; 12],
    _sr_col: &[i8; 12],
    _ud_flip: bool,
    _lr_flip: bool,
    _bd: i32,
) -> bool {
    false
}

/// **The SIMD-preserving fused 8x8 INVERSE transform** — KB-PERF-23's twin.
///
/// Its recipe does NOT collapse the way the forward's does, which is why this
/// is a separate proof rather than a copy: `INV_SHIFT[TX_8X8] = [-1, -4]`, so
/// **both** shifts are live (the forward's `[2, -1, 0]` has a no-op tail), and
/// the inverse additionally carries the two `clamp_buf` calls and the final
/// `highbd_clip_pixel_add` reconstruction.
///
/// **One structural advantage over the forward:** the inverse input is
/// COLUMN-major (`mod_input[c * row_n + r]`), so the row pass needs no
/// transpose — a contiguous 8-lane load at `input[c*8..]` already has lane = r.
/// The single transpose moves to BETWEEN the passes, where the generic path
/// pays it anyway by writing and re-reading `buf`.
///
/// # Exactness — every step is the generic driver's, in its order
///
/// * row: `clampv(_, bd + 8)` is `clamp_buf(ti, (bd+8) as i8)`; `rshiftv(_, 1)`
///   is `round_shift_array(_, -shift[0])`;
/// * `rect_type == 0` at 8x8, so the `NEW_INV_SQRT2` scaling does not apply;
/// * column: `clampv(_, col_clamp)` with `col_clamp = max(bd+6, 16)`;
///   `rshiftv(_, 4)` is `round_shift_array(_, -shift[1])`;
/// * `lr_flip` is a lane REVERSE on the TRANSPOSED vectors — the generic reads
///   buf column `col_n-1-c` for output column `c`;
/// * `ud_flip` REORDERS the eight output vectors (output row `r` takes
///   `tout[row_n-1-r]`); it is not a lane operation;
/// * reconstruction is the same wrapping add then clamp to `[0, (1<<bd)-1]`,
///   so the `as u16` narrowing is exact.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_8x8_fused(
    t: Token,
    kr: Inv1d,
    kc: Inv1d,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;

    // ---- row pass: lane = r, and the loads are already contiguous ----
    let mut k = [i32x8::zero(t); 8];
    for (c, kc_) in k.iter_mut().enumerate() {
        let v = match input.get(c * 8..c * 8 + 8) {
            Some(sl) => i32x8::from_slice(t, sl),
            None => return false,
        };
        *kc_ = clampv(t, v, row_clamp);
    }
    let mut w = [i32x8::zero(t); 8];
    incant!(run_inv1d(kr, &k, &mut w, cos_bit, sr_row), [v3]);
    for x in w.iter_mut() {
        *x = rshiftv(t, *x, 1); // -shift[0], shift[0] == -1
    }

    // ---- transpose: lane = r -> lane = output column ----
    let r: [__m256i; 8] = core::array::from_fn(|i| w[i].into_repr());
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    let tr: [i32x8; 8] = [
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u0, u4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u1, u5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u2, u6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u3, u7)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u0, u4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u1, u5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u2, u6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u3, u7)),
    ];

    // ---- column pass: lane = output column ----
    let mut ci = [i32x8::zero(t); 8];
    for (i, x) in ci.iter_mut().enumerate() {
        let v = if lr_flip { revv(t, tr[i]) } else { tr[i] };
        *x = clampv(t, v, col_clamp);
    }
    let mut co = [i32x8::zero(t); 8];
    incant!(run_inv1d(kc, &ci, &mut co, cos_bit, sr_col), [v3]);
    for x in co.iter_mut() {
        *x = rshiftv(t, *x, 4); // -shift[1], shift[1] == -4
    }

    // ---- reconstruction ----
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, (1i32 << bd) - 1);
    for r in 0..8usize {
        let src = co[if ud_flip { 7 - r } else { r }];
        let idx = r * stride;
        let d: [u16; 8] = match output.get(idx..idx + 8).and_then(|s| s.try_into().ok()) {
            Some(d) => d,
            None => return false,
        };
        let dv = i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32));
        let s = (dv + src).clamp(zero, pix_hi).to_array();
        for (j, &sv) in s.iter().enumerate() {
            output[idx + j] = sv as u16;
        }
    }
    true
}

/// Dispatch for [`inv_8x8_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_txfm2d_8x8_fused(
    txfm_type_row: i32,
    txfm_type_col: i32,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    // The i16 whole-block kernel — C's `av1_lowbd_inv_txfm2d_add_8x8` shape —
    // is exact under its per-(row, col)-kernel input bound and the bd8 clamp
    // constants; out-of-range or unmapped types take the i32 fused path
    // below, which is always correct.
    if row_clamp == 16
        && col_clamp == 16
        && sr_row.iter().all(|&b| b == 16)
        && sr_col.iter().all(|&b| b == 16)
    {
        if let (Some(kr), Some(kc)) = (inv8_kernel(txfm_type_row), inv8_kernel(txfm_type_col)) {
            if incant!(
                inv_8x8_fused_i16(
                    kr,
                    kc,
                    input,
                    output,
                    stride,
                    INV8_I16_BOUND[kr as usize][kc as usize],
                    ud_flip,
                    lr_flip,
                    bd
                ),
                [v3, scalar]
            ) {
                return true;
            }
        }
    }
    let (Some(kr), Some(kc)) = (inv_kernel(txfm_type_row), inv_kernel(txfm_type_col)) else {
        return false;
    };
    if inv_kernel_n(kr) != 8 || inv_kernel_n(kc) != 8 {
        return false;
    }
    incant!(
        inv_8x8_fused(
            kr, kc, input, output, stride, row_clamp, col_clamp, sr_row, sr_col, ud_flip, lr_flip,
            bd
        ),
        [v3, scalar]
    )
}

// ---- fused 8x8 inverse on i16 lanes: C's `av1_lowbd_inv_txfm2d_add_8x8` ----
//
// [`inv_8x8_fused`] keeps the block in i32x8 lanes; C's lowbd kernel never
// leaves i16: `packs_epi32` load (which IS `clamp_buf(16)` — both bound to the
// i16 range), `av1_idct8_sse2` / `av1_iadst8_sse2` / `iidentity8_sse2` w8
// kernels on the `btf_16_sse2` madd butterfly, one `transpose_16bit_8x8`,
// `round_shift_16bit(4)` as `mulhrs(2048)`, and a clip-add store.
//
// # The gate
//
// The contract is the port SCALAR, whose `half_btf` intermediates are
// unclamped i32 — the i16 `packs`/`adds`/`subs` are exact only while every
// butterfly output stays inside i16. Exhausting all 2^8 sign vertices per
// lane over the exact i16 dataflow (btf outputs unclamped in scalar; adds and
// the input pack saturate identically to scalar's `clamp_value(_, 16)`) gives
// per-(row, col)-kernel input bounds — the row kernel's own internals AND the
// col kernel's `row_out >> 1` input bound both apply:
//
// ```text
//               col:  Dct8  Adst8  Idtx8
//   row Dct8          2347   2431   6201
//   row Adst8         2432   2518   6423
//   row Idtx8         6202   6423  16383
// ```
//
// `iidentity8`'s `adds(v, v)` (scalar: unclamped `2 * v`) sets the 16383
// ceiling; `iadst8`'s terminal `subs(0, x)` negations must never see
// `-32768` — every bound keeps all stage values <= ~18k. Out-of-range inputs
// decline to the i32 fused path, which is always correct.

/// The three 8-point kernels of C's `lowbd_txfm_all_1d_w8_arr` row.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
enum Inv8 {
    Dct,
    Adst,
    Idtx,
}

#[cfg(target_arch = "x86_64")]
fn inv8_kernel(txfm_type: i32) -> Option<Inv8> {
    match txfm_type {
        1 => Some(Inv8::Dct),
        6 => Some(Inv8::Adst),
        9 => Some(Inv8::Idtx),
        _ => None,
    }
}

/// `INV8_BOUND[row][col]` — max `|input[i32]|` for exact i16 lanes (see the
/// block comment above for the derivation).
#[cfg(target_arch = "x86_64")]
const INV8_I16_BOUND: [[i32; 3]; 3] = [
    [2347, 2431, 6201],
    [2432, 2518, 6423],
    [6202, 6423, 16383],
];

/// The `incant!` fallback: decline to the i32 fused path.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn inv_8x8_fused_i16_scalar(
    _t: archmage::ScalarToken,
    _kr: Inv8,
    _kc: Inv8,
    _input: &[i32],
    _output: &mut [u16],
    _stride: usize,
    _bound: i32,
    _ud_flip: bool,
    _lr_flip: bool,
    _bd: i32,
) -> bool {
    false
}

/// The i16 fused 8x8 inverse transform — C's `av1_lowbd_inv_txfm2d_add_8x8`
/// shape adapted to the port's u16 output (`highbd_clip_pixel_add` store).
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_8x8_fused_i16(
    t: Token,
    kr: Inv8,
    kc: Inv8,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    bound: i32,
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let _ = t;
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;

    // `pair_set_epi16(a, b)` — i16 pair (a lo, b hi) per 32-bit group.
    let pair = |a: i32, b: i32| -> __m128i {
        _mm_set1_epi32((a as u16 as u32 | ((b as u16 as u32) << 16)) as i32)
    };

    // `btf_16_sse2`: unpack both halves, `madd` each against both weight
    // pairs, round, pack.
    let btf = |w0: __m128i, w1: __m128i, i0: __m128i, i1: __m128i| -> (__m128i, __m128i) {
        let rnd = _mm_set1_epi32(1 << (cos_bit - 1));
        let cnt = _mm_cvtsi32_si128(cos_bit);
        let t0 = _mm_unpacklo_epi16(i0, i1);
        let t1 = _mm_unpackhi_epi16(i0, i1);
        let c0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w0), rnd), cnt);
        let c1 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t1, w0), rnd), cnt);
        let d0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w1), rnd), cnt);
        let d1 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t1, w1), rnd), cnt);
        (_mm_packs_epi32(c0, c1), _mm_packs_epi32(d0, d1))
    };
    let adds_subs = |a: __m128i, b: __m128i| -> (__m128i, __m128i) {
        (_mm_adds_epi16(a, b), _mm_subs_epi16(a, b))
    };

    // `av1_idct8_sse2` verbatim.
    let idct8 = |i: &[__m128i; 8]| -> [__m128i; 8] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let mut x = [i[0], i[4], i[2], i[6], i[1], i[5], i[3], i[7]];
        let (x4, x7) = btf(pair(c[56], -c[8]), pair(c[8], c[56]), x[4], x[7]);
        x[4] = x4;
        x[7] = x7;
        let (x5, x6) = btf(pair(c[24], -c[40]), pair(c[40], c[24]), x[5], x[6]);
        x[5] = x5;
        x[6] = x6;
        let (x0, x1) = btf(pair(c[32], c[32]), pair(c[32], -c[32]), x[0], x[1]);
        x[0] = x0;
        x[1] = x1;
        let (x2, x3) = btf(pair(c[48], -c[16]), pair(c[16], c[48]), x[2], x[3]);
        x[2] = x2;
        x[3] = x3;
        let (x4, x5) = adds_subs(x[4], x[5]);
        x[4] = x4;
        x[5] = x5;
        let (x7, x6) = (_mm_adds_epi16(x[7], x[6]), _mm_subs_epi16(x[7], x[6]));
        x[7] = x7;
        x[6] = x6;
        let (x0, x3) = adds_subs(x[0], x[3]);
        let (x1, x2) = adds_subs(x[1], x[2]);
        x[0] = x0;
        x[3] = x3;
        x[1] = x1;
        x[2] = x2;
        let (x5, x6) = btf(pair(-c[32], c[32]), pair(c[32], c[32]), x[5], x[6]);
        x[5] = x5;
        x[6] = x6;
        [
            _mm_adds_epi16(x[0], x[7]),
            _mm_adds_epi16(x[1], x[6]),
            _mm_adds_epi16(x[2], x[5]),
            _mm_adds_epi16(x[3], x[4]),
            _mm_subs_epi16(x[3], x[4]),
            _mm_subs_epi16(x[2], x[5]),
            _mm_subs_epi16(x[1], x[6]),
            _mm_subs_epi16(x[0], x[7]),
        ]
    };

    // `av1_iadst8_sse2` verbatim.
    let iadst8 = |i: &[__m128i; 8]| -> [__m128i; 8] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let mut x = [i[7], i[0], i[5], i[2], i[3], i[4], i[1], i[6]];
        let (x0, x1) = btf(pair(c[4], c[60]), pair(c[60], -c[4]), x[0], x[1]);
        x[0] = x0;
        x[1] = x1;
        let (x2, x3) = btf(pair(c[20], c[44]), pair(c[44], -c[20]), x[2], x[3]);
        x[2] = x2;
        x[3] = x3;
        let (x4, x5) = btf(pair(c[36], c[28]), pair(c[28], -c[36]), x[4], x[5]);
        x[4] = x4;
        x[5] = x5;
        let (x6, x7) = btf(pair(c[52], c[12]), pair(c[12], -c[52]), x[6], x[7]);
        x[6] = x6;
        x[7] = x7;
        for (a, b) in [(0usize, 4usize), (1, 5), (2, 6), (3, 7)] {
            let (s, d) = adds_subs(x[a], x[b]);
            x[a] = s;
            x[b] = d;
        }
        let (x4, x5) = btf(pair(c[16], c[48]), pair(c[48], -c[16]), x[4], x[5]);
        x[4] = x4;
        x[5] = x5;
        let (x6, x7) = btf(pair(-c[48], c[16]), pair(c[16], c[48]), x[6], x[7]);
        x[6] = x6;
        x[7] = x7;
        for (a, b) in [(0usize, 2usize), (1, 3), (4, 6), (5, 7)] {
            let (s, d) = adds_subs(x[a], x[b]);
            x[a] = s;
            x[b] = d;
        }
        let (x2, x3) = btf(pair(c[32], c[32]), pair(c[32], -c[32]), x[2], x[3]);
        x[2] = x2;
        x[3] = x3;
        let (x6, x7) = btf(pair(c[32], c[32]), pair(c[32], -c[32]), x[6], x[7]);
        x[6] = x6;
        x[7] = x7;
        let z = _mm_setzero_si128();
        [
            x[0],
            _mm_subs_epi16(z, x[4]),
            x[6],
            _mm_subs_epi16(z, x[2]),
            x[3],
            _mm_subs_epi16(z, x[7]),
            x[5],
            _mm_subs_epi16(z, x[1]),
        ]
    };

    // `iidentity8_sse2` verbatim — `adds(v, v)` is the scalar's `2 * v`.
    let iidtx8 = |i: &[__m128i; 8]| -> [__m128i; 8] {
        [
            _mm_adds_epi16(i[0], i[0]),
            _mm_adds_epi16(i[1], i[1]),
            _mm_adds_epi16(i[2], i[2]),
            _mm_adds_epi16(i[3], i[3]),
            _mm_adds_epi16(i[4], i[4]),
            _mm_adds_epi16(i[5], i[5]),
            _mm_adds_epi16(i[6], i[6]),
            _mm_adds_epi16(i[7], i[7]),
        ]
    };

    let run8 = |k: Inv8, i: &[__m128i; 8]| -> [__m128i; 8] {
        match k {
            Inv8::Dct => idct8(i),
            Inv8::Adst => iadst8(i),
            Inv8::Idtx => iidtx8(i),
        }
    };

    // Load columns (contiguous: input[c*8+r]). `packs` saturates i32->i16,
    // which is exactly `clamp_buf(bd + 8)` at the bd8 gate this dispatcher
    // requires. The bound check folds into the loads: `abs`/`max` accumulate
    // over the block and ONE comparison decides — `abs(i32::MIN)` wraps to
    // 0x8000_0000, which a signed `cmpgt` would read as negative and slip, so
    // the test is `mx - bound > 0` per lane: `abs` outputs are <= 2^31 and
    // bound <= 16383, so the difference never wraps below -bound.
    let mut mx = _mm_setzero_si128();
    let mut b = [_mm_setzero_si128(); 8];
    for (c, v) in b.iter_mut().enumerate() {
        let col: &[i32; 8] = match input.get(c * 8..c * 8 + 8).and_then(|s| s.try_into().ok()) {
            Some(a) => a,
            None => return false,
        };
        let lo = _mm_loadu_si128(<&[i32; 4]>::try_from(&col[..4]).unwrap());
        let hi = _mm_loadu_si128(<&[i32; 4]>::try_from(&col[4..]).unwrap());
        mx = _mm_max_epu32(mx, _mm_max_epu32(_mm_abs_epi32(lo), _mm_abs_epi32(hi)));
        *v = _mm_packs_epi32(lo, hi);
    }
    let over = _mm_cmpgt_epi32(
        _mm_sub_epi32(mx, _mm_set1_epi32(bound)),
        _mm_setzero_si128(),
    );
    if _mm_testz_si128(over, over) == 0 {
        return false;
    }

    // Pass 1: register index = the c axis (input column), lanes = r.
    let mut w = run8(kr, &b);
    // `round_shift_array(., 1)` == `_mm_mulhrs_epi16(v, 1 << 14)`:
    // `(v * 16384 + 0x8000) >> 16` == `(v + 1) >> 1` for every i16.
    for v in w.iter_mut() {
        *v = _mm_mulhrs_epi16(*v, _mm_set1_epi16(16384));
    }

    // `transpose_16bit_8x8` verbatim -> register index = row, lane = column.
    let a0 = _mm_unpacklo_epi16(w[0], w[1]);
    let a1 = _mm_unpacklo_epi16(w[2], w[3]);
    let a2 = _mm_unpacklo_epi16(w[4], w[5]);
    let a3 = _mm_unpacklo_epi16(w[6], w[7]);
    let a4 = _mm_unpackhi_epi16(w[0], w[1]);
    let a5 = _mm_unpackhi_epi16(w[2], w[3]);
    let a6 = _mm_unpackhi_epi16(w[4], w[5]);
    let a7 = _mm_unpackhi_epi16(w[6], w[7]);
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpacklo_epi32(a2, a3);
    let b2 = _mm_unpacklo_epi32(a4, a5);
    let b3 = _mm_unpacklo_epi32(a6, a7);
    let b4 = _mm_unpackhi_epi32(a0, a1);
    let b5 = _mm_unpackhi_epi32(a2, a3);
    let b6 = _mm_unpackhi_epi32(a4, a5);
    let b7 = _mm_unpackhi_epi32(a6, a7);
    let mut tr = [
        _mm_unpacklo_epi64(b0, b1),
        _mm_unpackhi_epi64(b0, b1),
        _mm_unpacklo_epi64(b4, b5),
        _mm_unpackhi_epi64(b4, b5),
        _mm_unpacklo_epi64(b2, b3),
        _mm_unpackhi_epi64(b2, b3),
        _mm_unpacklo_epi64(b6, b7),
        _mm_unpackhi_epi64(b6, b7),
    ];
    // lr_flip is a LANE reverse — scalar gathers `buf[r*8 + (7-c)]` per
    // output column c: reverse i16 within each 64-bit half, then swap the
    // halves (a plain epi32 reverse would scramble the i16 pairs).
    if lr_flip {
        for v in tr.iter_mut() {
            *v = _mm_shuffle_epi32::<0x4E>(_mm_shufflehi_epi16::<0x1B>(
                _mm_shufflelo_epi16::<0x1B>(*v),
            ));
        }
    }

    // Pass 2: register index = the r axis, lanes = output column.
    let mut u = run8(kc, &tr);
    // `round_shift_array(., 4)` == `_mm_mulhrs_epi16(v, 1 << 11)`.
    for v in u.iter_mut() {
        *v = _mm_mulhrs_epi16(*v, _mm_set1_epi16(2048));
    }

    // u[r][c] — register index is the output row (ud_flip selects 7 - r),
    // lanes the output column, so each register is one contiguous dst row.
    let zero = _mm_setzero_si128();
    let hi = _mm_set1_epi16(((1i32 << bd) - 1) as i16);
    for r in 0..8usize {
        let src = u[if ud_flip { 7 - r } else { r }];
        let idx = r * stride;
        let dst: &mut [u16; 8] = match output.get_mut(idx..idx + 8).and_then(|s| s.try_into().ok()) {
            Some(d) => d,
            None => return false,
        };
        let d = _mm_loadu_si128(dst);
        let sum = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(d, src), zero), hi);
        _mm_storeu_si128(dst, sum);
    }
    true
}

pub(crate) fn try_inv_col_pass(
    txfm_type_col: i32,
    buf: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    if col_n % 8 != 0 && !(col_n == 4 && half_batch_pays(row_n)) {
        return false;
    }
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    let Some(kernel) = inv_kernel(txfm_type_col) else {
        return false;
    };
    debug_assert_eq!(inv_kernel_n(kernel), row_n);
    incant!(
        inv_col_pass(
            kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range, ud_flip,
            lr_flip, bd
        ),
        [v3, neon, scalar]
    )
}

/// Vector ROW pass of `av1_inv_txfm2d_add` — 8 rows per lane batch (or, for
/// the audited DCT kernels at the bd8 row constants with `row_n % 16 == 0`,
/// 16 rows per i16 lane batch — [`lowbd16::inv_row_pass_i16`]).
/// Contiguous loads (`mod_input[c*row_n + r..r+8]` — the input is stored
/// column-major), the optional NewInvSqrt2 rect scaling + row clamp, the
/// row kernel, `round_shift_array(-shift[0])`, then the strided store into
/// row-major `buf` via 8x8 transposes (per-lane scatter for the W=4 tail).
/// Returns `false` → caller runs the scalar loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_row_pass(
    txfm_type_row: i32,
    mod_input: &[i32],
    buf: &mut [i32],
    col_n: usize,
    row_n: usize,
    rect1: bool,
    shift0_bit: i32,
    row_clamp: i8,
    stage_range: &[i8; 12],
) -> bool {
    if row_n % 8 != 0 && !(row_n == 4 && half_batch_pays(col_n)) {
        return false;
    }
    let _ = crate::dispatch::scalar_forced();
    // Gate-3 rows lever (the Phase-C follow-up): the audited DCT kernels on
    // i16 lanes — 16 rows per vector. Fires only at the bd8 row constants
    // (row clamp 16 AND every stage_range entry 16 — the exact
    // `audit_i16_safety.py` entry conditions; bd10/12 pass 18/20 and stay
    // i32) with full 16-lane row groups. The i16 pass sign-extend-stores into
    // the same row-major i32 `buf`, so every column pass (i16 DCT or i32
    // iadst/identity) reads byte-identical values.
    //
    // Both vector tiers run it (AVX2 and NEON — see the `prims16` docs); the
    // `scalar` arm declines, which routes back to the i32 pass below and from
    // there to the driver's own loop.
    if row_clamp == 16 && stage_range.iter().all(|&b| b == 16) && row_n % 16 == 0 {
        if let Some(k16) = lowbd16::inv_kernel_i16(txfm_type_row) {
            debug_assert_eq!(lowbd16::inv_kernel_i16_n(k16), col_n);
            debug_assert!((0..=2).contains(&shift0_bit));
            if incant!(
                lowbd16::inv_row_pass_i16(
                    k16, mod_input, buf, col_n, row_n, rect1, shift0_bit
                ),
                [v3, neon, scalar]
            ) {
                return true;
            }
        }
    }
    let Some(kernel) = inv_kernel(txfm_type_row) else {
        return false;
    };
    debug_assert_eq!(inv_kernel_n(kernel), col_n);
    incant!(
        inv_row_pass(
            kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, row_clamp, stage_range
        ),
        [v3, neon, scalar]
    )
}

/// The lane-batched inverse row pass (8 rows per iteration; a 4-tall
/// transform runs as ONE group with 4 active lanes — upper lanes carry zeros
/// through the kernel and are never stored, so exactness per active lane is
/// the same module-docs argument).
///
/// The vector scratch is TIERED by `col_n` (8/16/64 lane vectors): a flat
/// `[i32x8; 64]` zero-init compiles to a 2 KiB memset per array, which
/// dominated the small transforms once they took the vector path (measured
/// +108M Ir of memset on a 4K decode). The core is `#[rite]`, so each arm
/// inlines it with its exactly-sized scratch.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_row_pass(
    t: Token,
    kernel: Inv1d,
    mod_input: &[i32],
    buf: &mut [i32],
    col_n: usize,
    row_n: usize,
    rect1: bool,
    shift0_bit: i32,
    row_clamp: i8,
    stage_range: &[i8; 12],
) -> bool {
    debug_assert!(col_n <= 64 && (row_n % 8 == 0 || row_n == 4));
    if col_n <= 8 {
        let mut tin = [i32x8::zero(t); 8];
        let mut tout = [i32x8::zero(t); 8];
        incant!(inv_row_pass_core(kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, row_clamp, stage_range,
            &mut tin, &mut tout,
        ), [v3, neon]);
    } else if col_n <= 16 {
        let mut tin = [i32x8::zero(t); 16];
        let mut tout = [i32x8::zero(t); 16];
        incant!(inv_row_pass_core(kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, row_clamp, stage_range,
            &mut tin, &mut tout,
        ), [v3, neon]);
    } else {
        let mut tin = [i32x8::zero(t); 64];
        let mut tout = [i32x8::zero(t); 64];
        incant!(inv_row_pass_core(kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, row_clamp, stage_range,
            &mut tin, &mut tout,
        ), [v3, neon]);
    }
    true
}

/// The `incant!` fallback for [`inv_row_pass`] when NO vector tier is available —
/// x86-64 without AVX2, or every token disabled by the `AOM_FORCE_SCALAR` pin.
/// Declining here is what routes the caller back to its scalar loop, so the
/// pin and the no-AVX2 path take the SAME `false` branch the pre-SIMD code
/// took. There is deliberately no scalar *implementation* of the pass: the
/// scalar twin is the driver's own per-column/row loop, which is the
/// differential's reference.
#[allow(clippy::too_many_arguments)]
fn inv_row_pass_scalar(
    _: ScalarToken,
    _kernel: Inv1d,
    _mod_input: &[i32],
    _buf: &mut [i32],
    _col_n: usize,
    _row_n: usize,
    _rect1: bool,
    _shift0_bit: i32,
    _row_clamp: i8,
    _stage_range: &[i8; 12],
) -> bool {
    false
}

/// The row-pass body over caller-sized scratch (see [`inv_row_pass`]).
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_row_pass_core(
    t: Token,
    kernel: Inv1d,
    mod_input: &[i32],
    buf: &mut [i32],
    col_n: usize,
    row_n: usize,
    rect1: bool,
    shift0_bit: i32,
    row_clamp: i8,
    stage_range: &[i8; 12],
    tin: &mut [I32x8<Token>],
    tout: &mut [I32x8<Token>],
) {
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;
    let mut rg = 0usize;
    while rg < row_n {
        let active = (row_n - rg).min(8); // 8, or 4 (row_n == 4)
        for (c, ti) in tin[..col_n].iter_mut().enumerate() {
            let mut v = if active == 8 {
                i32x8::from_slice(t, &mod_input[c * row_n + rg..c * row_n + rg + 8])
            } else {
                let a: [i32; 4] =
                    mod_input[c * row_n + rg..c * row_n + rg + 4].try_into().unwrap();
                i32x8::from_array(t, [a[0], a[1], a[2], a[3], 0, 0, 0, 0])
            };
            if rect1 {
                // round_shift(x * NewInvSqrt2, NewSqrt2Bits) — the rect scaling.
                v = mul_rshiftv(t, v, NEW_INV_SQRT2, NEW_SQRT2_BITS);
            }
            *ti = clampv(t, v, row_clamp); // the driver's clamp_buf(bd+8)
        }
        incant!(run_inv1d(kernel, &tin[..col_n], &mut tout[..col_n], cos_bit, stage_range), [v3, neon]);
        if shift0_bit > 0 {
            // round_shift_array(buf_row, -shift[0]); shift[0] in {0,-1,-2}.
            for to in tout[..col_n].iter_mut() {
                *to = rshiftv(t, *to, shift0_bit);
            }
        }
        // Store: buf[(rg+k)*col_n + c] = tout[c].lane(k), k < active —
        // transpose 8x8 tiles for the col_n%8==0 groups (only the active
        // rows of each tile are stored), per-lane scatter for the W=4 tail.
        let full = col_n & !7;
        for cg in (0..full).step_by(8) {
            let tr = transpose8(t, &tout[cg..cg + 8]);
            for (k, trk) in tr.iter().take(active).enumerate() {
                let base = (rg + k) * col_n + cg;
                trk.store((&mut buf[base..base + 8]).try_into().unwrap());
            }
        }
        for c in full..col_n {
            let a = tout[c].to_array();
            for (k, &av) in a.iter().take(active).enumerate() {
                buf[(rg + k) * col_n + c] = av;
            }
        }
        rg += active;
    }
}

/// Vector COLUMN pass of `fwd_txfm2d_core` — 8 columns per lane batch.
/// Contiguous i16 loads (`input[src_r*stride + c..c+8]`), the negative-bit
/// `round_shift_array` input stage (`v << 2` i64-clamped), the col kernel,
/// `round_shift_array(-shift[1])`, then contiguous stores into row-major
/// `buf` (lane-reversed at the mirrored position under `lr_flip`).
/// Returns `false` → caller runs the scalar loop.
#[allow(clippy::too_many_arguments)]
/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fwd_rect816_fused_scalar(
    _t: archmage::ScalarToken,
    _kc: Fwd1d,
    _kr: Fwd1d,
    _input: &[i16],
    _output: &mut [i32],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _cos_bit_col: i32,
    _cos_bit_row: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// **The fused 8x16 / 16x8 forward transforms** — 9.02 % of forward transforms
/// at the shipping preset.
///
/// Unlike KB-PERF-27's 4x8/8x4 pair, **both dimensions are >= 8, so there is no
/// padding and none of the per-lane `from_array`/`to_array` glue** that
/// KB-PERF-28 measured as the cause of its own shortfall: every load is
/// `from_slice` and every store is a whole-vector `store`.
///
/// Written over GROUPS rather than a fixed size: `CG = col_n / 8` column groups
/// and `RG = row_n / 8` row groups, so one body covers both shapes (and has the
/// same structure as the landed 8x8 (1,1) and 16x16 (2,2) kernels). The
/// transpose is `CG * RG` 8x8 blocks, each landing on exactly the (column
/// group, row group) pair the row pass consumes.
///
/// `FWD_SHIFT` is `[2, -2, 0]` for both sizes — 16x16's recipe — and
/// `rect_type == +-1`, so the row pass carries the `NEW_SQRT2` scaling.
/// `lr_flip` is the ARRAY-ORDER reversal KB-PERF-27 introduced.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_rect816_fused(
    t: Token,
    kc: Fwd1d,
    kr: Fwd1d,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let sr = [0i8; 12];
    if !((col_n == 8 && row_n == 16) || (col_n == 16 && row_n == 8)) {
        return false;
    }
    let cg_n = col_n / 8;
    let rg_n = row_n / 8;

    let tr8 = |m: &[i32x8]| -> [i32x8; 8] {
        let r: [__m256i; 8] = core::array::from_fn(|i| m[i].into_repr());
        let a0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let a1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let a2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let a3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let a4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let a5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let a6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let a7 = _mm256_unpackhi_epi32(r[6], r[7]);
        let b0 = _mm256_unpacklo_epi64(a0, a2);
        let b1 = _mm256_unpackhi_epi64(a0, a2);
        let b2 = _mm256_unpacklo_epi64(a1, a3);
        let b3 = _mm256_unpackhi_epi64(a1, a3);
        let b4 = _mm256_unpacklo_epi64(a4, a6);
        let b5 = _mm256_unpackhi_epi64(a4, a6);
        let b6 = _mm256_unpacklo_epi64(a5, a7);
        let b7 = _mm256_unpackhi_epi64(a5, a7);
        [
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
        ]
    };

    // ---- column pass: lane = column, one vector per column group ----
    let mut w = [[i32x8::zero(t); 16]; 2];
    for cg in 0..cg_n {
        let mut v = [i32x8::zero(t); 16];
        for r in 0..row_n {
            let src_r = if ud_flip { row_n - 1 - r } else { r };
            let base = src_r * stride + cg * 8;
            let a: [i16; 8] = match input.get(base..base + 8).and_then(|s| s.try_into().ok()) {
                Some(a) => a,
                None => return false,
            };
            let x = i32x8::from_array(t, core::array::from_fn(|j| a[j] as i32));
            v[r] = shl_clamp64v(t, x, 2);
        }
        let mut o = [i32x8::zero(t); 16];
        incant!(run_fwd1d(kc, &v[..row_n], &mut o[..row_n], cos_bit_col, &sr), [v3]);
        for r in 0..row_n {
            w[cg][r] = rshiftv(t, o[r], 2); // -shift[1], shift[1] == -2
        }
    }

    // ---- transpose: CG * RG blocks -> (column, row-group) ----
    let mut tt = [[i32x8::zero(t); 16]; 2]; // tt[rg][c], lane = r within group rg
    for cg in 0..cg_n {
        for rg in 0..rg_n {
            let blk = tr8(&w[cg][rg * 8..rg * 8 + 8]);
            for k in 0..8 {
                tt[rg][cg * 8 + k] = blk[k];
            }
        }
    }

    // ---- row pass: one call per row group ----
    for rg in 0..rg_n {
        let mut ri = [i32x8::zero(t); 16];
        for p in 0..col_n {
            ri[p] = if lr_flip { tt[rg][col_n - 1 - p] } else { tt[rg][p] };
        }
        let mut u = [i32x8::zero(t); 16];
        incant!(run_fwd1d(kr, &ri[..col_n], &mut u[..col_n], cos_bit_row, &sr), [v3]);
        // shift[2] == 0; rect_type == +-1 so the NEW_SQRT2 scaling applies.
        for c in 0..col_n {
            let val = mul_rshiftv(t, u[c], NEW_SQRT2, NEW_SQRT2_BITS);
            let base = c * row_n + rg * 8;
            let o: &mut [i32; 8] = match output.get_mut(base..base + 8).and_then(|s| s.try_into().ok()) {
                Some(o) => o,
                None => return false,
            };
            val.store(o);
        }
    }
    true
}

/// Dispatch for [`fwd_rect816_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_txfm2d_rect816_fused(
    txfm_type_col: i32,
    txfm_type_row: i32,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    let (Some(kc), Some(kr)) = (fwd_kernel(txfm_type_col), fwd_kernel(txfm_type_row)) else {
        return false;
    };
    if fwd_kernel_n(kc) != row_n || fwd_kernel_n(kr) != col_n {
        return false;
    }
    incant!(
        fwd_rect816_fused(
            kc, kr, input, output, stride, col_n, row_n, cos_bit_col, cos_bit_row, ud_flip, lr_flip
        ),
        [v3, scalar]
    )
}

/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fwd_rect48_fused_scalar(
    _t: archmage::ScalarToken,
    _kc: Fwd1d,
    _kr: Fwd1d,
    _input: &[i16],
    _output: &mut [i32],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _cos_bit_col: i32,
    _cos_bit_row: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// **The fused 4x8 / 8x4 forward transforms** — together **16.7 % of forward
/// transforms** at the shipping preset, three times 16x16's share and the
/// largest block left after KB-PERF-23/25.
///
/// Both fit inside eight vectors of eight lanes with some lanes or vectors
/// unused, so ONE body serves both shapes: `col_n` live lanes per row vector,
/// `row_n` row vectors, one 8x8 transpose (the unused vectors are zero and the
/// unused transpose outputs are simply not read), then `col_n` row vectors with
/// `row_n` live lanes.
///
/// # Two things differ from the square kernels
///
/// * **`rect_type == +-1`, so the row pass applies the `NEW_SQRT2` scaling**
///   that 4x4 / 8x8 / 16x16 all skip (`fwd_row_pass_core`'s `rect1` arm, after
///   the shift).
/// * **`lr_flip` is an ARRAY-ORDER reversal, not a lane reverse.** The generic
///   column pass writes lane `j`'s result to `buf[r][col_n-1-j]`, i.e. it
///   permutes which COLUMN POSITION each source column occupies before the row
///   pass reads them in order. After the transpose, `t[c]` already holds source
///   column `c`, so feeding the row pass `t[col_n-1-p]` is the same permutation
///   and costs nothing. (The square kernels reverse lanes instead, which is
///   equivalent there; this form is simply cheaper and works at any width.)
///
/// `FWD_SHIFT` is `[2, -1, 0]` for both sizes — the same recipe as 8x8 — and
/// `shift[2] == 0`, so the only tail is the rect scaling.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_rect48_fused(
    t: Token,
    kc: Fwd1d,
    kr: Fwd1d,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let sr = [0i8; 12];
    if !((col_n == 4 && row_n == 8) || (col_n == 8 && row_n == 4)) {
        return false;
    }

    // Lane = column; lanes >= col_n stay zero and are never stored.
    let mut v = [i32x8::zero(t); 8];
    for r in 0..row_n {
        let src_r = if ud_flip { row_n - 1 - r } else { r };
        let base = src_r * stride;
        let sl = match input.get(base..base + col_n) {
            Some(sl) => sl,
            None => return false,
        };
        let w = i32x8::from_array(
            t,
            core::array::from_fn(|j| if j < col_n { sl[j] as i32 } else { 0 }),
        );
        v[r] = shl_clamp64v(t, w, 2);
    }

    let mut w = [i32x8::zero(t); 8];
    incant!(run_fwd1d(kc, &v[..row_n], &mut w[..row_n], cos_bit_col, &sr), [v3]);
    for x in w[..row_n].iter_mut() {
        *x = rshiftv(t, *x, 1); // -shift[1], shift[1] == -1
    }

    // One 8x8 transpose; the padding vectors are zero and their outputs unread.
    let r8: [__m256i; 8] = core::array::from_fn(|i| w[i].into_repr());
    let a0 = _mm256_unpacklo_epi32(r8[0], r8[1]);
    let a1 = _mm256_unpackhi_epi32(r8[0], r8[1]);
    let a2 = _mm256_unpacklo_epi32(r8[2], r8[3]);
    let a3 = _mm256_unpackhi_epi32(r8[2], r8[3]);
    let a4 = _mm256_unpacklo_epi32(r8[4], r8[5]);
    let a5 = _mm256_unpackhi_epi32(r8[4], r8[5]);
    let a6 = _mm256_unpacklo_epi32(r8[6], r8[7]);
    let a7 = _mm256_unpackhi_epi32(r8[6], r8[7]);
    let b0 = _mm256_unpacklo_epi64(a0, a2);
    let b1 = _mm256_unpackhi_epi64(a0, a2);
    let b2 = _mm256_unpacklo_epi64(a1, a3);
    let b3 = _mm256_unpackhi_epi64(a1, a3);
    let b4 = _mm256_unpacklo_epi64(a4, a6);
    let b5 = _mm256_unpackhi_epi64(a4, a6);
    let b6 = _mm256_unpacklo_epi64(a5, a7);
    let b7 = _mm256_unpackhi_epi64(a5, a7);
    let tr: [i32x8; 8] = [
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
    ];

    // `lr_flip` as an array-order reversal — see the doc comment.
    let mut ri = [i32x8::zero(t); 8];
    for p in 0..col_n {
        ri[p] = if lr_flip { tr[col_n - 1 - p] } else { tr[p] };
    }

    let mut u = [i32x8::zero(t); 8];
    incant!(run_fwd1d(kr, &ri[..col_n], &mut u[..col_n], cos_bit_row, &sr), [v3]);

    // shift[2] == 0; `rect_type == +-1` so the NEW_SQRT2 scaling applies.
    for c in 0..col_n {
        let val = mul_rshiftv(t, u[c], NEW_SQRT2, NEW_SQRT2_BITS);
        let o = match output.get_mut(c * row_n..c * row_n + row_n) {
            Some(o) => o,
            None => return false,
        };
        let a = val.to_array();
        o.copy_from_slice(&a[..row_n]);
    }
    true
}

/// Dispatch for [`fwd_rect48_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_txfm2d_rect48_fused(
    txfm_type_col: i32,
    txfm_type_row: i32,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    // The i16 whole-block kernels — C's `av1_lowbd_fwd_txfm2d_{4x8,8x4}_sse2`
    // shapes — are exact under their input bound; out-of-range or unmapped
    // types take the i32 fused path below, which is always correct.
    if let (Some(kc), Some(kr)) = (fwd_r_kernel(txfm_type_col), fwd_r_kernel(txfm_type_row)) {
        if incant!(
            fwd_rect48_fused_i16(
                kc, kr, input, output, stride, col_n, row_n, cos_bit_col, cos_bit_row, ud_flip,
                lr_flip
            ),
            [v3, scalar]
        ) {
            return true;
        }
    }
    let (Some(kc), Some(kr)) = (fwd_kernel(txfm_type_col), fwd_kernel(txfm_type_row)) else {
        return false;
    };
    if fwd_kernel_n(kc) != row_n || fwd_kernel_n(kr) != col_n {
        return false;
    }
    incant!(
        fwd_rect48_fused(
            kc, kr, input, output, stride, col_n, row_n, cos_bit_col, cos_bit_row, ud_flip, lr_flip
        ),
        [v3, scalar]
    )
}

/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fwd_16x16_fused_scalar(
    _t: archmage::ScalarToken,
    _kc: Fwd1d,
    _kr: Fwd1d,
    _input: &[i16],
    _output: &mut [i32],
    _stride: usize,
    _cos_bit_col: i32,
    _cos_bit_row: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// **The SIMD-preserving fused 16x16 forward transform** — KB-PERF-23's shape
/// at twice the width.
///
/// `i32x8` holds eight lanes, so sixteen columns is TWO vectors per row and the
/// 16x16 transpose is **four 8x8 block transposes**: `w_lo[0..8]`,
/// `w_lo[8..16]`, `w_hi[0..8]`, `w_hi[8..16]` become, respectively, columns
/// 0-7 x rows 0-7, columns 0-7 x rows 8-15, columns 8-15 x rows 0-7, and
/// columns 8-15 x rows 8-15 — which is exactly the two row-groups the row pass
/// wants. No off-diagonal swap is needed because the groups are consumed
/// separately.
///
/// `FWD_SHIFT[TX_16X16] = [2, -2, 0]`, so `shift1` is a round-shift by **2**
/// (8x8's is 1) and the tail is again a no-op; `rect_type == 0`. Note
/// `COS_BIT_ROW[2][2]` is **12**, not 13 — the row and column cos bits differ
/// at this size, unlike 8x8.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_16x16_fused(
    t: Token,
    kc: Fwd1d,
    kr: Fwd1d,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let sr = [0i8; 12];

    // One 8x8 i32 transpose, written out (KB-PERF-23/24: an index-computed
    // version got a stage pairing wrong and the differentials caught it).
    let tr8 = |m: &[i32x8]| -> [i32x8; 8] {
        let r: [__m256i; 8] = core::array::from_fn(|i| m[i].into_repr());
        let a0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let a1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let a2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let a3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let a4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let a5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let a6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let a7 = _mm256_unpackhi_epi32(r[6], r[7]);
        let b0 = _mm256_unpacklo_epi64(a0, a2);
        let b1 = _mm256_unpackhi_epi64(a0, a2);
        let b2 = _mm256_unpacklo_epi64(a1, a3);
        let b3 = _mm256_unpackhi_epi64(a1, a3);
        let b4 = _mm256_unpacklo_epi64(a4, a6);
        let b5 = _mm256_unpackhi_epi64(a4, a6);
        let b6 = _mm256_unpacklo_epi64(a5, a7);
        let b7 = _mm256_unpackhi_epi64(a5, a7);
        [
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
            i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
        ]
    };

    // ---- column pass: lane = column, two vectors per row ----
    let mut vlo = [i32x8::zero(t); 16];
    let mut vhi = [i32x8::zero(t); 16];
    for r in 0..16usize {
        let src_r = if ud_flip { 15 - r } else { r };
        let base = src_r * stride;
        let a: [i16; 16] = match input.get(base..base + 16).and_then(|s| s.try_into().ok()) {
            Some(a) => a,
            None => return false,
        };
        let lo = i32x8::from_array(t, core::array::from_fn(|j| a[j] as i32));
        let hi = i32x8::from_array(t, core::array::from_fn(|j| a[j + 8] as i32));
        vlo[r] = shl_clamp64v(t, lo, 2);
        vhi[r] = shl_clamp64v(t, hi, 2);
    }
    let mut wlo = [i32x8::zero(t); 16];
    let mut whi = [i32x8::zero(t); 16];
    incant!(run_fwd1d(kc, &vlo, &mut wlo, cos_bit_col, &sr), [v3]);
    incant!(run_fwd1d(kc, &vhi, &mut whi, cos_bit_col, &sr), [v3]);

    // `round_shift_array(_, -shift[1])` with shift[1] == -2, then the lr flip:
    // source column c lands at column 15-c, i.e. the halves swap AND reverse.
    for r in 0..16usize {
        let l = rshiftv(t, wlo[r], 2);
        let h = rshiftv(t, whi[r], 2);
        if lr_flip {
            wlo[r] = revv(t, h);
            whi[r] = revv(t, l);
        } else {
            wlo[r] = l;
            whi[r] = h;
        }
    }

    // ---- transpose: four 8x8 blocks -> the two row-groups ----
    let ta = tr8(&wlo[0..8]); // cols 0-7,  rows 0-7
    let tb = tr8(&wlo[8..16]); // cols 0-7,  rows 8-15
    let tc = tr8(&whi[0..8]); // cols 8-15, rows 0-7
    let td = tr8(&whi[8..16]); // cols 8-15, rows 8-15

    let mut ga = [i32x8::zero(t); 16];
    let mut gb = [i32x8::zero(t); 16];
    for c in 0..8usize {
        ga[c] = ta[c];
        ga[c + 8] = tc[c];
        gb[c] = tb[c];
        gb[c + 8] = td[c];
    }

    // ---- row pass: lane = row, one call per 8-row group ----
    let mut ua = [i32x8::zero(t); 16];
    let mut ub = [i32x8::zero(t); 16];
    incant!(run_fwd1d(kr, &ga, &mut ua, cos_bit_row, &sr), [v3]);
    incant!(run_fwd1d(kr, &gb, &mut ub, cos_bit_row, &sr), [v3]);

    // shift[2] == 0 and rect_type == 0: `output[c*16 + r]`.
    for c in 0..16usize {
        let o = match output.get_mut(c * 16..c * 16 + 16) {
            Some(o) => o,
            None => return false,
        };
        let (lo, hi) = o.split_at_mut(8);
        match (<&mut [i32; 8]>::try_from(lo), <&mut [i32; 8]>::try_from(hi)) {
            (Ok(l), Ok(h)) => {
                ua[c].store(l);
                ub[c].store(h);
            }
            _ => return false,
        }
    }
    true
}

/// Dispatch for [`fwd_16x16_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_txfm2d_16x16_fused(
    txfm_type_col: i32,
    txfm_type_row: i32,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    let (Some(kc), Some(kr)) = (fwd_kernel(txfm_type_col), fwd_kernel(txfm_type_row)) else {
        return false;
    };
    if fwd_kernel_n(kc) != 16 || fwd_kernel_n(kr) != 16 {
        return false;
    }
    incant!(
        fwd_16x16_fused(kc, kr, input, output, stride, cos_bit_col, cos_bit_row, ud_flip, lr_flip),
        [v3, scalar]
    )
}

// ---- fused 4x4 forward: C's `av1_lowbd_fwd_txfm2d_4x4_sse2` shape ------------
//
// The generic driver runs the 4x4's row pass SCALAR (`try_fwd_row_pass`
// requires `row_n % 8 == 0`) and the pre-existing `fwd_txfm2d_4x4_fused` is a
// scalar fusion, so every 4x4 pays ~8 scalar kernel calls. C instead does the
// whole block on i16 lanes: `madd` against `pair_set_epi16` constants computes
// four columns per instruction pair, `transpose_16bit_4x4` reorders in
// registers, and the row kernel runs the same way. The three kernels below
// are verbatim transcriptions of `fdct4x4_new_sse2` / `fadst4x4_new_sse2` /
// `fidentity4x4_new_sse2`; between them they cover every TX_TYPE via the
// ud/lr flip and V_/H_ kernel-mix, exactly as C's tables do.
//
// # The gate
//
// Unlike C — whose SSE2 is bit-exact vs its own scalar only while nothing
// overflows i16 — this port's contract is the C SCALAR for every input the
// public API can reach, so the kernel is gated at runtime on
// `max|input| <= 512`. The bound is derived, not tuned: post-`<<2` values
// reach `4M` (`slli` wraps only past 8191); the butterflies sum two such
// values (`<= 8M`); each pass's outputs are bounded by `gain * 4M` with
// `gain = N*sqrt(2)/2 = 2.83` for the 4-point DCT/ADST rows and `sqrt(2)` for
// identity — so `packs_epi32` cannot saturate below `4M <= 11583`, i.e.
// `M <= 2895`, and the row pass's own input bound needs `11.32*M <= 11583`.
// `M = 512` clears every link with >= 2x margin, including `fadst4`'s `in7`
// pair sum. Real residuals (|in| <= 255 at bd8) always pass; out-of-range
// callers decline to the scalar fused path, which is always correct.

/// The three 4-point kernels of C's `col_txfm4x4_arr` / `row_txfm4x4_arr`.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
enum Fwd4 {
    Dct,
    Adst,
    Idtx,
}

#[cfg(target_arch = "x86_64")]
fn fwd4_kernel(txfm_type: i32) -> Option<Fwd4> {
    match txfm_type {
        0 => Some(Fwd4::Dct),
        5 => Some(Fwd4::Adst),
        8 => Some(Fwd4::Idtx),
        _ => None,
    }
}

/// Scalar twin — declines, routing the caller to the scalar fused path.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fwd_4x4_fused_scalar(
    _t: archmage::ScalarToken,
    _kc: Fwd4,
    _kr: Fwd4,
    _input: &[i16],
    _output: &mut [i32],
    _stride: usize,
    _cos_bit_col: i32,
    _cos_bit_row: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// The SIMD fused 4x4 forward transform — `av1_lowbd_fwd_txfm2d_4x4_sse2`.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_4x4_fused(
    t: Token,
    kc: Fwd4,
    kr: Fwd4,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let _ = t;

    // `pair_set_epi16(a, b)` — i16 pair (a lo, b hi) per 32-bit group.
    let pair = |a: i32, b: i32| -> __m128i {
        _mm_set1_epi32((a as u16 as u32 | ((b as u16 as u32) << 16)) as i32)
    };
    // `(v + (1<<(bit-1))) >> bit` on i32 lanes, runtime count.
    let sra32 = |v: __m128i, bit: i32| -> __m128i {
        let r = _mm_add_epi32(v, _mm_set1_epi32(1 << (bit - 1)));
        _mm_sra_epi32(r, _mm_cvtsi32_si128(bit))
    };

    // `fdct4x4_new_sse2` verbatim: butterflies on interleaved column pairs,
    // `madd` against the cospi pairs, round, pack. Output[k] = coefficient k
    // per column, low 4 i16 lanes live.
    let fdct4 = |i: &[__m128i; 4], cos_bit: i32| -> [__m128i; 4] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let u0 = _mm_unpacklo_epi16(i[0], i[1]);
        let u1 = _mm_unpacklo_epi16(i[3], i[2]);
        let v0 = _mm_add_epi16(u0, u1);
        let v1 = _mm_sub_epi16(u0, u1);
        let w0 = sra32(_mm_madd_epi16(v0, pair(c[32], c[32])), cos_bit);
        let w1 = sra32(_mm_madd_epi16(v0, pair(c[32], -c[32])), cos_bit);
        let w2 = sra32(_mm_madd_epi16(v1, pair(c[16], c[48])), cos_bit);
        let w3 = sra32(_mm_madd_epi16(v1, pair(c[48], -c[16])), cos_bit);
        let o0 = _mm_packs_epi32(w0, w1);
        let o1 = _mm_packs_epi32(w2, w3);
        [o0, o1, _mm_srli_si128::<8>(o0), _mm_srli_si128::<8>(o1)]
    };

    // `fadst4x4_new_sse2` verbatim.
    let fadst4 = |i: &[__m128i; 4], cos_bit: i32| -> [__m128i; 4] {
        let s = crate::transform::cospi::sinpi_arr(cos_bit);
        let z = _mm_setzero_si128();
        let in7 = _mm_add_epi16(i[0], i[1]);
        let u0 = _mm_unpacklo_epi16(i[0], i[1]);
        let u1 = _mm_unpacklo_epi16(i[2], i[3]);
        let u2 = _mm_unpacklo_epi16(in7, z);
        let u3 = _mm_unpacklo_epi16(i[2], z);
        let u4 = _mm_unpacklo_epi16(i[3], z);
        let s33 = _mm_set1_epi16(s[3] as i16);
        let v0 = _mm_madd_epi16(u0, pair(s[1], s[2]));
        let v1 = _mm_madd_epi16(u1, pair(s[3], s[4]));
        let v2 = _mm_madd_epi16(u2, s33);
        let v3 = _mm_madd_epi16(u0, pair(s[4], -s[1]));
        let v4 = _mm_madd_epi16(u1, pair(-s[3], s[2]));
        let v5 = _mm_madd_epi16(u3, s33);
        let v6 = _mm_madd_epi16(u4, s33);
        let w0 = _mm_add_epi32(v0, v1);
        let w1 = _mm_sub_epi32(v2, v6);
        let w2 = _mm_add_epi32(v3, v4);
        let w3 = _mm_sub_epi32(w2, w0);
        let w4 = _mm_slli_epi32::<2>(v5);
        let w5 = _mm_sub_epi32(w4, v5);
        let w6 = _mm_add_epi32(w3, w5);
        let o0 = sra32(w0, cos_bit);
        let o1 = sra32(w1, cos_bit);
        let o2 = sra32(w2, cos_bit);
        let o3 = sra32(w6, cos_bit);
        let p0 = _mm_packs_epi32(o0, o2);
        let p1 = _mm_packs_epi32(o1, o3);
        [p0, p1, _mm_srli_si128::<8>(p0), _mm_srli_si128::<8>(p1)]
    };

    // `fidentity4x4_new_sse2` verbatim: `round_shift(v * NewSqrt2,
    // NewSqrt2Bits)` via `madd((v,1), (NewSqrt2, 1<<11))`.
    let fidtx4 = |i: &[__m128i; 4]| -> [__m128i; 4] {
        let one = _mm_set1_epi16(1);
        let sr = pair(NEW_SQRT2, 1 << (NEW_SQRT2_BITS - 1));
        let mut o = [_mm_setzero_si128(); 4];
        for (o, i) in o.iter_mut().zip(i.iter()) {
            let a = _mm_unpacklo_epi16(*i, one);
            let b = _mm_srai_epi32::<NEW_SQRT2_BITS>(_mm_madd_epi16(a, sr));
            *o = _mm_packs_epi32(b, b);
        }
        o
    };

    let run4 = |k: Fwd4, i: &[__m128i; 4], cos_bit: i32| -> [__m128i; 4] {
        match k {
            Fwd4::Dct => fdct4(i, cos_bit),
            Fwd4::Adst => fadst4(i, cos_bit),
            Fwd4::Idtx => fidtx4(i),
        }
    };

    // load_buffer_16bit_to_16bit_w4(+_flip), then round_shift_16bit(shift[0]=2)
    // = slli by 2 — exact under the gate.
    let mut b = [_mm_setzero_si128(); 4];
    for (r, v) in b.iter_mut().enumerate() {
        let src = if ud_flip { 3 - r } else { r };
        let row: &[i16; 4] = match input.get(src * stride..src * stride + 4) {
            Some(s) => match s.try_into() {
                Ok(a) => a,
                Err(_) => return false,
            },
            None => return false,
        };
        *v = _mm_slli_epi16::<2>(_mm_loadu_si64(row));
    }

    let col = run4(kc, &b, cos_bit_col);
    // shift[1] == 0.

    // transpose_16bit_4x4 — only the low 64 bits of each output are live.
    let a0 = _mm_unpacklo_epi16(col[0], col[1]);
    let a1 = _mm_unpacklo_epi16(col[2], col[3]);
    let t0 = _mm_unpacklo_epi32(a0, a1);
    let t2 = _mm_unpackhi_epi32(a0, a1);
    let mut w = [t0, _mm_srli_si128::<8>(t0), t2, _mm_srli_si128::<8>(t2)];
    if lr_flip {
        w = [w[3], w[2], w[1], w[0]];
    }

    let row = run4(kr, &w, cos_bit_row);
    // shift[2] == 0, no rect scale at 4x4.

    // store_buffer_16bit_to_32bit_w4: sign-extend the low 4 lanes; `output` is
    // column-major, `output[c*4 + r]`.
    for (c, v) in row.iter().enumerate() {
        let ext = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(*v, *v));
        match output.get_mut(c * 4..c * 4 + 4) {
            Some(s) => match <&mut [i32; 4]>::try_from(s) {
                Ok(a) => _mm_storeu_si128(a, ext),
                Err(_) => return false,
            },
            None => return false,
        }
    }
    true
}

/// Dispatch for [`fwd_4x4_fused`]; `false` routes to the scalar fused path.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_txfm2d_4x4_fused(
    txfm_type_col: i32,
    txfm_type_row: i32,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    let (Some(kc), Some(kr)) = (fwd4_kernel(txfm_type_col), fwd4_kernel(txfm_type_row)) else {
        return false;
    };
    if prims16::max_abs_i16_strided(input, stride, 4, 4) > 512 {
        return false;
    }
    incant!(
        fwd_4x4_fused(kc, kr, input, output, stride, cos_bit_col, cos_bit_row, ud_flip, lr_flip),
        [v3, scalar]
    )
}

/// The three 4-point inverse kernels of C's `lowbd_txfm_all_1d_w4_arr` row —
/// `idct4_w4_sse2`, `iadst4_w4_sse2`, `iidentity4_ssse3`: one `__m128i` per
/// 4-point vector, i16 lanes, `madd` butterflies.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
enum Inv4 {
    Dct,
    Adst,
    Idtx,
}

#[cfg(target_arch = "x86_64")]
fn inv4_kernel(txfm_type: i32) -> Option<Inv4> {
    match txfm_type {
        0 => Some(Inv4::Dct),
        5 => Some(Inv4::Adst),
        8 => Some(Inv4::Idtx),
        _ => None,
    }
}

/// Scalar twin — declines, routing the caller to the scalar fused path.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn inv_4x4_fused_scalar(
    _t: archmage::ScalarToken,
    _kr: Inv4,
    _kc: Inv4,
    _input: &[i32],
    _output: &mut [u16],
    _stride: usize,
    _ud_flip: bool,
    _lr_flip: bool,
    _bd: i32,
) -> bool {
    false
}

/// The fused whole-block 4x4 inverse, u16 output — the port's counterpart of
/// `lowbd_inv_txfm2d_add_4x4_ssse3`. The port's coefficient input is
/// COLUMN-major (`input[c*4+r]`, the highbd/u16 convention), so the column
/// loads are contiguous and the row transform sees the transpose; C's lowbd
/// kernel loads rows contiguously because its input is row-major. The clip-add
/// is `highbd_clip_pixel_add` (u16 dest), not C's u8 `packus` write.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_4x4_fused(
    t: Token,
    kr: Inv4,
    kc: Inv4,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let _ = t;
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;

    // `pair_set_epi16(a, b)` — i16 pair (a lo, b hi) per 32-bit group.
    let pair = |a: i32, b: i32| -> __m128i {
        _mm_set1_epi32((a as u16 as u32 | ((b as u16 as u32) << 16)) as i32)
    };

    // `btf_16_4p_sse2`: interleaved-lane `madd` butterfly, round, pack.
    let btf4p = |w0: __m128i, w1: __m128i, i0: __m128i, i1: __m128i| -> (__m128i, __m128i) {
        let rnd = _mm_set1_epi32(1 << (cos_bit - 1));
        let cnt = _mm_cvtsi32_si128(cos_bit);
        let t0 = _mm_unpacklo_epi16(i0, i1);
        let u0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w0), rnd), cnt);
        let v0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w1), rnd), cnt);
        (_mm_packs_epi32(u0, u0), _mm_packs_epi32(v0, v0))
    };

    // `idct4_w4_sse2` verbatim.
    let idct4 = |i: &[__m128i; 4]| -> [__m128i; 4] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let x = [i[0], i[2], i[1], i[3]];
        let (x0, x1) = btf4p(pair(c[32], c[32]), pair(c[32], -c[32]), x[0], x[1]);
        let (x2, x3) = btf4p(pair(c[48], -c[16]), pair(c[16], c[48]), x[2], x[3]);
        [
            _mm_adds_epi16(x0, x3),
            _mm_adds_epi16(x1, x2),
            _mm_subs_epi16(x1, x2),
            _mm_subs_epi16(x0, x3),
        ]
    };

    // `iadst4_w4_sse2` verbatim.
    let iadst4 = |i: &[__m128i; 4]| -> [__m128i; 4] {
        let s = crate::transform::cospi::sinpi_arr(cos_bit);
        let u0 = _mm_unpacklo_epi16(i[0], i[2]);
        let u1 = _mm_unpacklo_epi16(i[1], i[3]);
        let x = [
            _mm_madd_epi16(u0, pair(s[1], s[4])),
            _mm_madd_epi16(u0, pair(s[2], -s[1])),
            _mm_madd_epi16(u1, pair(s[3], s[2])),
            _mm_madd_epi16(u1, pair(s[3], -s[4])),
            _mm_madd_epi16(u0, pair(s[3], -s[3])),
            _mm_madd_epi16(u1, pair(0, s[3])),
            _mm_madd_epi16(u0, pair(s[4], s[2])),
            _mm_madd_epi16(u1, pair(-s[3], -s[1])),
        ];
        let rnd = _mm_set1_epi32(1 << (cos_bit - 1));
        let cnt = _mm_cvtsi32_si128(cos_bit);
        let srai = |v: __m128i| {
            _mm_packs_epi32(_mm_sra_epi32(_mm_add_epi32(v, rnd), cnt), _mm_setzero_si128())
        };
        [
            srai(_mm_add_epi32(x[0], x[2])),
            srai(_mm_add_epi32(x[1], x[3])),
            srai(_mm_add_epi32(x[4], x[5])),
            srai(_mm_add_epi32(x[6], x[7])),
        ]
    };

    // `iidentity4_ssse3` verbatim.
    let iidtx4 = |i: &[__m128i; 4]| -> [__m128i; 4] {
        let frac = NEW_SQRT2 - (1 << NEW_SQRT2_BITS);
        let scale = _mm_set1_epi16((frac << (15 - NEW_SQRT2_BITS)) as i16);
        let mut o = [_mm_setzero_si128(); 4];
        for (o, i) in o.iter_mut().zip(i.iter()) {
            *o = _mm_adds_epi16(_mm_mulhrs_epi16(*i, scale), *i);
        }
        o
    };

    let run4 = |k: Inv4, i: &[__m128i; 4]| -> [__m128i; 4] {
        match k {
            Inv4::Dct => idct4(i),
            Inv4::Adst => iadst4(i),
            Inv4::Idtx => iidtx4(i),
        }
    };

    // transpose_16bit_4x4 — only the low 64 bits of each output are live.
    let transpose4 = |i: &[__m128i; 4]| -> [__m128i; 4] {
        let a0 = _mm_unpacklo_epi16(i[0], i[1]);
        let a1 = _mm_unpacklo_epi16(i[2], i[3]);
        let t0 = _mm_unpacklo_epi32(a0, a1);
        let t2 = _mm_unpackhi_epi32(a0, a1);
        [t0, _mm_srli_si128::<8>(t0), t2, _mm_srli_si128::<8>(t2)]
    };

    // Load columns (contiguous: input[c*4+r]). `packs` saturates i32->i16,
    // which is exactly `clamp_buf(bd + 8)` at the bd8 gate this dispatcher
    // requires — both bound to the i16 range.
    let mut cols = [_mm_setzero_si128(); 4];
    for (c, v) in cols.iter_mut().enumerate() {
        let col: &[i32; 4] = match input.get(c * 4..c * 4 + 4) {
            Some(s) => match s.try_into() {
                Ok(a) => a,
                Err(_) => return false,
            },
            None => return false,
        };
        let x = _mm_loadu_si128(col);
        *v = _mm_packs_epi32(x, x);
    }

    // Pass 1: register index = the c axis (input column), lanes = r — the
    // same batched-1-D form `run_inv1d` and C's w4 kernels use. cols[c] feeds
    // the row kernel directly; no transpose needed on the way in.
    let w = run4(kr, &cols);
    // shift[0] == 0 — no post-row shift.

    // Pass 2 input: row i of the intermediate matrix = transpose of w; lanes
    // are the output-column axis, so lr_flip is a LANE reverse (3,2,1,0)
    // inside each register — scalar gathers `buf[r*4 + (3-c)]` per output
    // column c.
    let mut tt = transpose4(&w);
    if lr_flip {
        for v in tt.iter_mut() {
            *v = _mm_shufflelo_epi16::<0x1B>(*v);
        }
    }
    let co = run4(kc, &tt);

    // `round_shift_array(., 4)` == `_mm_mulhrs_epi16(v, 1 << (15 - 4))`:
    // `(v * 2048 + 0x4000) >> 15` == `(v + 8) >> 4` for every i16.
    let mut u = co;
    for v in u.iter_mut() {
        *v = _mm_mulhrs_epi16(*v, _mm_set1_epi16(2048));
    }

    // u[j][c] = temp_out_c[j] — register index is the output row (with
    // ud_flip selecting 3 - r), lanes the output column, so each register is
    // one contiguous destination row.
    let zero = _mm_setzero_si128();
    let hi = _mm_set1_epi16(((1i32 << bd) - 1) as i16);
    for r in 0..4usize {
        let src = u[if ud_flip { 3 - r } else { r }];
        let idx = r * stride;
        let dst: &mut [u16; 4] = match output.get_mut(idx..idx + 4) {
            Some(s) => match s.try_into() {
                Ok(a) => a,
                Err(_) => return false,
            },
            None => return false,
        };
        let d = _mm_loadu_si64(dst);
        let sum = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(d, src), zero), hi);
        _mm_storeu_si64(dst, sum);
    }
    true
}

/// Dispatch for [`inv_4x4_fused`]; `false` routes to the scalar fused path.
/// The i16 lanes are exact only while every stage bound is 16 — i.e. bd 8,
/// where `opt_range` is `(16, 16)` and both `clamp_buf` bounds are 16. Above
/// that the row-pass intermediates can exceed i16 and the kernel declines.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_txfm2d_4x4_fused(
    txfm_type_row: i32,
    txfm_type_col: i32,
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    row_clamp: i8,
    col_clamp: i8,
    sr_row: &[i8; 12],
    sr_col: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    if row_clamp != 16
        || col_clamp != 16
        || !sr_row.iter().all(|&b| b == 16)
        || !sr_col.iter().all(|&b| b == 16)
    {
        return false;
    }
    let (Some(kr), Some(kc)) = (inv4_kernel(txfm_type_row), inv4_kernel(txfm_type_col)) else {
        return false;
    };
    // Runtime input bound — the i16 lanes are exact only while NO intermediate
    // saturates, which the port scalar (i32/i64-wide, no stage clamps on the
    // iadst4 path) does not guarantee. With |input| <= 4096 every lane stays
    // inside i16 through both passes: row outputs <= ~11.2k, col outputs <=
    // ~30.3k < 32767, so `packs`/`adds`/`subs` are all lossless.
    let mut max_abs = 0u32;
    for &v in &input[..16.min(input.len())] {
        max_abs = max_abs.max(v.unsigned_abs());
    }
    if input.len() < 16 || max_abs > 4096 {
        return false;
    }
    incant!(
        inv_4x4_fused(kr, kc, input, output, stride, ud_flip, lr_flip, bd),
        [v3, scalar]
    )
}

/// The `incant!` fallback: decline to the generic two-pass driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fwd_8x8_fused_scalar(
    _t: archmage::ScalarToken,
    _kc: Fwd1d,
    _kr: Fwd1d,
    _input: &[i16],
    _output: &mut [i32],
    _stride: usize,
    _cos_bit_col: i32,
    _cos_bit_row: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// **The SIMD-PRESERVING fused 8x8 forward transform.**
///
/// `encoder_txfm_size_census_2026-09-09.md`'s own correction is the brief:
/// KB-PERF-16's 8x8 fusion was SCALAR, measured **+7.07 %** and was reverted,
/// because at 8x8 both generic passes satisfy `n % 8 == 0` and run full-width —
/// so a scalar fusion trades the SIMD away. Its closing line: *"a future one
/// would have to be VECTORISED itself to beat what is already there."* This is
/// that one. **It keeps both vector passes and removes only the DRIVER**: the
/// config derivation, the two `try_*` gates, the scratch tiering, and the
/// round trip through `buf` between the passes.
///
/// `encoder_domain_split_2026-09-09.md` is why this is the ranked target:
/// **52.8 % of the shipping-preset gap is coefficient-domain** (against 24.4 %
/// for the plane representation), and transform alone is 40.5 %. 8x8 is
/// **25.74 % of forward transforms** at that preset, second only to the
/// already-fused 4x4.
///
/// # Layout
///
/// Lane = COLUMN for the first pass: `v[r]` is input row `r`, so the column
/// kernel runs vertically across the eight vectors, per lane — exactly what
/// `fwd_col_pass` does, without materialising `buf`. Then ONE in-register
/// 8x8 `i32` transpose makes lane = ROW, and the row kernel runs the same way.
/// The generic row pass already pays a transpose (it loads `buf` through 8x8
/// tiles), so this does not ADD one — it moves it out of memory.
///
/// # Exactness
///
/// Every step is the generic path's own, in its order: `shl_clamp64v(_, 2)` is
/// `round_shift_array(_, -shift[0])` with `FWD_SHIFT[TX_8X8] = [2, -1, 0]`;
/// `rshiftv(_, 1)` is `round_shift_array(_, -shift[1])`; `shift[2] == 0` and
/// `rect_type == 0` so the row pass has no tail. `lr_flip` is a LANE REVERSE
/// applied AFTER the column kernel, because the generic writes lane `j`'s
/// result to column `col_n-1-j` — reversing before the kernel would be wrong.
/// `ud_flip` is the source-row reversal at load. The transpose only moves
/// lanes.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_8x8_fused(
    t: Token,
    kc: Fwd1d,
    kr: Fwd1d,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let sr = [0i8; 12];

    // Lane = column. `shift[0] == 2` -> the NEGATIVE-bit arm of
    // `round_shift_array`, i.e. a clamped left shift, same as `fwd_col_pass`.
    let mut v = [i32x8::zero(t); 8];
    for (r, vr) in v.iter_mut().enumerate() {
        let src_r = if ud_flip { 7 - r } else { r };
        let base = src_r * stride;
        let a: [i16; 8] = match input.get(base..base + 8).and_then(|s| s.try_into().ok()) {
            Some(a) => a,
            None => return false,
        };
        let w = i32x8::from_array(
            t,
            [
                a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32,
                a[4] as i32, a[5] as i32, a[6] as i32, a[7] as i32,
            ],
        );
        *vr = shl_clamp64v(t, w, 2);
    }

    let mut w = [i32x8::zero(t); 8];
    incant!(run_fwd1d(kc, &v, &mut w, cos_bit_col, &sr), [v3]);

    // `round_shift_array(_, -shift[1])` with shift[1] == -1, then the lr flip:
    // lane j held source column j, and the generic writes it to column 7-j.
    for x in w.iter_mut() {
        let s = rshiftv(t, *x, 1);
        *x = if lr_flip { revv(t, s) } else { s };
    }

    // In-register 8x8 i32 transpose -> lane = row. Written out rather than
    // index-computed: a clever `from_fn` got stage 2's pairing wrong and the
    // differentials caught it.
    let r: [__m256i; 8] = core::array::from_fn(|i| w[i].into_repr());
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    let tr: [i32x8; 8] = [
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u0, u4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u1, u5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u2, u6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x20>(u3, u7)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u0, u4)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u1, u5)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u2, u6)),
        i32x8::from_repr(t, _mm256_permute2x128_si256::<0x31>(u3, u7)),
    ];

    let mut u = [i32x8::zero(t); 8];
    incant!(run_fwd1d(kr, &tr, &mut u, cos_bit_row, &sr), [v3]);

    // shift[2] == 0 and rect_type == 0, so no tail. `output[k*8 + r]` is
    // lane `r` of `u[k]` — one contiguous vector store per k.
    for (k, uk) in u.iter().enumerate() {
        let o: &mut [i32; 8] = match output.get_mut(k * 8..k * 8 + 8).and_then(|s| s.try_into().ok()) {
            Some(o) => o,
            None => return false,
        };
        uk.store(o);
    }
    true
}

/// Dispatch for [`fwd_8x8_fused`]; `false` routes to the generic driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_txfm2d_8x8_fused(
    txfm_type_col: i32,
    txfm_type_row: i32,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    // The i16 whole-block kernel — C's `av1_lowbd_fwd_txfm2d_8x8_sse2` shape —
    // is exact under its input bound; out-of-range or unmapped types take the
    // i32 fused path below, which is always correct.
    if let (Some(kc), Some(kr)) = (fwd8_kernel(txfm_type_col), fwd8_kernel(txfm_type_row)) {
        if incant!(
            fwd_8x8_fused_i16(
                kc, kr, input, output, stride, cos_bit_col, cos_bit_row, ud_flip, lr_flip
            ),
            [v3, scalar]
        ) {
            return true;
        }
    }
    let (Some(kc), Some(kr)) = (fwd_kernel(txfm_type_col), fwd_kernel(txfm_type_row)) else {
        return false;
    };
    if fwd_kernel_n(kc) != 8 || fwd_kernel_n(kr) != 8 {
        return false;
    }
    incant!(
        fwd_8x8_fused(kc, kr, input, output, stride, cos_bit_col, cos_bit_row, ud_flip, lr_flip),
        [v3, scalar]
    )
}

// ---- fused 8x8 forward on i16 lanes: C's `av1_lowbd_fwd_txfm2d_8x8_sse2` ---
//
// [`fwd_8x8_fused`] keeps the block in i32 lanes the whole way — exact for
// every input, but every butterfly widens and narrows. C's whole-block
// kernel never leaves i16: `fdct8x8_new_sse2` / `fadst8x8_new_sse2` /
// `fidentity8x8_new_sse2` built on the `btf_16_sse2` `madd` butterfly, one
// in-register `transpose_16bit_8x8`, and a sign-extending store. The kernels
// below are verbatim transcriptions of those three.
//
// # The gate
//
// Same discipline as the 4x4: the port's contract is the C SCALAR, whose
// intermediates never clamp, so the i16 lanes are exact only while nothing
// saturates. Bounding every stage over all 2^8 sign vertices per lane gives
// `max|input| <= 511`: post-`<<2` lanes reach 2,044; pass-1 outputs stay
// <= 11,563 (the binding kernel is `fdct8`), so `round_shift_16bit`'s
// `adds(_, 1)` cannot saturate and pass 2 sees <= 5,782 — under `fdct8`'s
// 5,792 no-saturation input bound, the tightest of the three. bd8 residuals
// (|in| <= 255) always pass; out-of-range callers decline to the i32 fused
// path, which is always correct.

/// The three 8-point kernels of C's `col_txfm8x8_arr` / `row_txfm8x8_arr`.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
enum Fwd8 {
    Dct,
    Adst,
    Idtx,
}

#[cfg(target_arch = "x86_64")]
fn fwd8_kernel(txfm_type: i32) -> Option<Fwd8> {
    match txfm_type {
        1 => Some(Fwd8::Dct),
        6 => Some(Fwd8::Adst),
        9 => Some(Fwd8::Idtx),
        _ => None,
    }
}

/// Scalar twin — declines, routing the caller to the i32 fused path.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fwd_8x8_fused_i16_scalar(
    _t: archmage::ScalarToken,
    _kc: Fwd8,
    _kr: Fwd8,
    _input: &[i16],
    _output: &mut [i32],
    _stride: usize,
    _cos_bit_col: i32,
    _cos_bit_row: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// The i16 fused 8x8 forward transform — `av1_lowbd_fwd_txfm2d_8x8_sse2`.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_8x8_fused_i16(
    t: Token,
    kc: Fwd8,
    kr: Fwd8,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let _ = t;

    // `pair_set_epi16(a, b)` — i16 pair (a lo, b hi) per 32-bit group.
    let pair = |a: i32, b: i32| -> __m128i {
        _mm_set1_epi32((a as u16 as u32 | ((b as u16 as u32) << 16)) as i32)
    };

    // `btf_16_sse2`: unpack both halves, `madd` each against both weight
    // pairs, round, pack.
    let btf =
        |w0: __m128i, w1: __m128i, i0: __m128i, i1: __m128i, cos_bit: i32| -> (__m128i, __m128i) {
            let rnd = _mm_set1_epi32(1 << (cos_bit - 1));
            let cnt = _mm_cvtsi32_si128(cos_bit);
            let t0 = _mm_unpacklo_epi16(i0, i1);
            let t1 = _mm_unpackhi_epi16(i0, i1);
            let c0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w0), rnd), cnt);
            let c1 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t1, w0), rnd), cnt);
            let d0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w1), rnd), cnt);
            let d1 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t1, w1), rnd), cnt);
            (_mm_packs_epi32(c0, c1), _mm_packs_epi32(d0, d1))
        };

    // `fdct8x8_new_sse2` verbatim.
    let fdct8 = |i: &[__m128i; 8], cos_bit: i32| -> [__m128i; 8] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let x1 = [
            _mm_adds_epi16(i[0], i[7]),
            _mm_adds_epi16(i[1], i[6]),
            _mm_adds_epi16(i[2], i[5]),
            _mm_adds_epi16(i[3], i[4]),
            _mm_subs_epi16(i[3], i[4]),
            _mm_subs_epi16(i[2], i[5]),
            _mm_subs_epi16(i[1], i[6]),
            _mm_subs_epi16(i[0], i[7]),
        ];
        let (x2_5, x2_6) = btf(pair(-c[32], c[32]), pair(c[32], c[32]), x1[5], x1[6], cos_bit);
        let x2 = [
            _mm_adds_epi16(x1[0], x1[3]),
            _mm_adds_epi16(x1[1], x1[2]),
            _mm_subs_epi16(x1[1], x1[2]),
            _mm_subs_epi16(x1[0], x1[3]),
            x1[4],
            x2_5,
            x2_6,
            x1[7],
        ];
        let (x3_0, x3_1) = btf(pair(c[32], c[32]), pair(c[32], -c[32]), x2[0], x2[1], cos_bit);
        let (x3_2, x3_3) = btf(pair(c[48], c[16]), pair(-c[16], c[48]), x2[2], x2[3], cos_bit);
        let x3 = [
            x3_0,
            x3_1,
            x3_2,
            x3_3,
            _mm_adds_epi16(x2[4], x2[5]),
            _mm_subs_epi16(x2[4], x2[5]),
            _mm_subs_epi16(x2[7], x2[6]),
            _mm_adds_epi16(x2[7], x2[6]),
        ];
        let (o1, o7) = btf(pair(c[56], c[8]), pair(-c[8], c[56]), x3[4], x3[7], cos_bit);
        let (o5, o3) = btf(pair(c[24], c[40]), pair(-c[40], c[24]), x3[5], x3[6], cos_bit);
        [x3[0], o1, x3[2], o3, x3[1], o5, x3[3], o7]
    };

    // `fadst8x8_new_sse2` verbatim.
    let fadst8 = |i: &[__m128i; 8], cos_bit: i32| -> [__m128i; 8] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let z = _mm_setzero_si128();
        let x1 = [
            i[0],
            _mm_subs_epi16(z, i[7]),
            _mm_subs_epi16(z, i[3]),
            i[4],
            _mm_subs_epi16(z, i[1]),
            i[6],
            i[2],
            _mm_subs_epi16(z, i[5]),
        ];
        let (x2_2, x2_3) = btf(pair(c[32], c[32]), pair(c[32], -c[32]), x1[2], x1[3], cos_bit);
        let (x2_6, x2_7) = btf(pair(c[32], c[32]), pair(c[32], -c[32]), x1[6], x1[7], cos_bit);
        let x2 = [x1[0], x1[1], x2_2, x2_3, x1[4], x1[5], x2_6, x2_7];
        let x3 = [
            _mm_adds_epi16(x2[0], x2[2]),
            _mm_adds_epi16(x2[1], x2[3]),
            _mm_subs_epi16(x2[0], x2[2]),
            _mm_subs_epi16(x2[1], x2[3]),
            _mm_adds_epi16(x2[4], x2[6]),
            _mm_adds_epi16(x2[5], x2[7]),
            _mm_subs_epi16(x2[4], x2[6]),
            _mm_subs_epi16(x2[5], x2[7]),
        ];
        let (x4_4, x4_5) = btf(pair(c[16], c[48]), pair(c[48], -c[16]), x3[4], x3[5], cos_bit);
        let (x4_6, x4_7) = btf(pair(-c[48], c[16]), pair(c[16], c[48]), x3[6], x3[7], cos_bit);
        let x4 = [x3[0], x3[1], x3[2], x3[3], x4_4, x4_5, x4_6, x4_7];
        let x5 = [
            _mm_adds_epi16(x4[1], x4[5]),
            _mm_subs_epi16(x4[2], x4[6]),
            _mm_adds_epi16(x4[3], x4[7]),
            _mm_subs_epi16(x4[0], x4[4]),
            _mm_subs_epi16(x4[1], x4[5]),
            _mm_adds_epi16(x4[2], x4[6]),
            _mm_subs_epi16(x4[3], x4[7]),
            _mm_adds_epi16(x4[0], x4[4]),
        ];
        let (o7, o0) = btf(pair(c[4], c[60]), pair(c[60], -c[4]), x5[7], x5[0], cos_bit);
        let (o5, o2) = btf(pair(c[20], c[44]), pair(c[44], -c[20]), x5[5], x5[2], cos_bit);
        let (o3, o4) = btf(pair(c[36], c[28]), pair(c[28], -c[36]), x5[3], x5[4], cos_bit);
        let (o1, o6) = btf(pair(c[52], c[12]), pair(c[12], -c[52]), x5[1], x5[6], cos_bit);
        [o0, o1, o2, o3, o4, o5, o6, o7]
    };

    // `fidentity8x8_new_sse2` verbatim — `adds(v, v)` is the scalar's `2 * v`.
    let fidtx8 = |i: &[__m128i; 8]| -> [__m128i; 8] {
        [
            _mm_adds_epi16(i[0], i[0]),
            _mm_adds_epi16(i[1], i[1]),
            _mm_adds_epi16(i[2], i[2]),
            _mm_adds_epi16(i[3], i[3]),
            _mm_adds_epi16(i[4], i[4]),
            _mm_adds_epi16(i[5], i[5]),
            _mm_adds_epi16(i[6], i[6]),
            _mm_adds_epi16(i[7], i[7]),
        ]
    };

    let run8 = |k: Fwd8, i: &[__m128i; 8], cos_bit: i32| -> [__m128i; 8] {
        match k {
            Fwd8::Dct => fdct8(i, cos_bit),
            Fwd8::Adst => fadst8(i, cos_bit),
            Fwd8::Idtx => fidtx8(i),
        }
    };

    // load_buffer_16bit_to_16bit(+_flip): register = source row (ud_flip
    // reversed), lane = column. `round_shift_16bit(shift[0] = 2)` = `slli`
    // by 2, exact under the gate. The gate itself folds into the loads:
    // `abs`/`max` accumulate over the block, one `cmpgt`/`movemask` decides —
    // ~16 instructions where a scalar re-scan would cost ~250.
    let mut mx = _mm_setzero_si128();
    let mut b = [_mm_setzero_si128(); 8];
    for (r, v) in b.iter_mut().enumerate() {
        let src = if ud_flip { 7 - r } else { r };
        let row: &[i16; 8] = match input
            .get(src * stride..src * stride + 8)
            .and_then(|s| s.try_into().ok())
        {
            Some(a) => a,
            None => return false,
        };
        let rv = _mm_loadu_si128(row);
        mx = _mm_max_epu16(mx, _mm_abs_epi16(rv));
        *v = _mm_slli_epi16::<2>(rv);
    }
    // `abs(-32768)` wraps to 0x8000, which a SIGNED `cmpgt` would see as
    // -32768 and slip the gate — so the test is the UNSIGNED saturating
    // subtract: lanes <= 511 give 0, everything above (including the wrapped
    // 0x8000 = 32768u) gives nonzero, and `ptest` decides the block.
    let over = _mm_subs_epu16(mx, _mm_set1_epi16(511));
    if _mm_testz_si128(over, over) == 0 {
        return false;
    }

    let mut col = run8(kc, &b, cos_bit_col);

    // `round_shift_16bit(shift[1] = -1)`: `adds(_, 1)` then `srai 1` — the
    // saturating add cannot fire under the gate (pass-1 out <= 11,563).
    for v in col.iter_mut() {
        *v = _mm_srai_epi16::<1>(_mm_adds_epi16(*v, _mm_set1_epi16(1)));
    }

    // `transpose_16bit_8x8` verbatim, then lr_flip is a REGISTER reverse —
    // the register index is now the column axis.
    let a0 = _mm_unpacklo_epi16(col[0], col[1]);
    let a1 = _mm_unpacklo_epi16(col[2], col[3]);
    let a2 = _mm_unpacklo_epi16(col[4], col[5]);
    let a3 = _mm_unpacklo_epi16(col[6], col[7]);
    let a4 = _mm_unpackhi_epi16(col[0], col[1]);
    let a5 = _mm_unpackhi_epi16(col[2], col[3]);
    let a6 = _mm_unpackhi_epi16(col[4], col[5]);
    let a7 = _mm_unpackhi_epi16(col[6], col[7]);
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpacklo_epi32(a2, a3);
    let b2 = _mm_unpacklo_epi32(a4, a5);
    let b3 = _mm_unpacklo_epi32(a6, a7);
    let b4 = _mm_unpackhi_epi32(a0, a1);
    let b5 = _mm_unpackhi_epi32(a2, a3);
    let b6 = _mm_unpackhi_epi32(a4, a5);
    let b7 = _mm_unpackhi_epi32(a6, a7);
    let mut tr = [
        _mm_unpacklo_epi64(b0, b1),
        _mm_unpackhi_epi64(b0, b1),
        _mm_unpacklo_epi64(b4, b5),
        _mm_unpackhi_epi64(b4, b5),
        _mm_unpacklo_epi64(b2, b3),
        _mm_unpackhi_epi64(b2, b3),
        _mm_unpacklo_epi64(b6, b7),
        _mm_unpackhi_epi64(b6, b7),
    ];
    if lr_flip {
        tr.reverse();
    }

    let row = run8(kr, &tr, cos_bit_row);
    // shift[2] == 0 and rect_type == 0, so no tail.

    // `store_buffer_16bit_to_32bit_w8`: sign-extend each i16x8 to two i32x4;
    // register index is the output column, `output[c*8 + r]`.
    for (c, v) in row.iter().enumerate() {
        let o: &mut [i32; 8] = match output
            .get_mut(c * 8..c * 8 + 8)
            .and_then(|s| s.try_into().ok())
        {
            Some(o) => o,
            None => return false,
        };
        let lo = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(*v, *v));
        let hi = _mm_srai_epi32::<16>(_mm_unpackhi_epi16(*v, *v));
        let (o_lo, o_hi) = o.split_at_mut(4);
        match (
            <&mut [i32; 4]>::try_from(o_lo),
            <&mut [i32; 4]>::try_from(o_hi),
        ) {
            (Ok(l), Ok(h)) => {
                _mm_storeu_si128(l, lo);
                _mm_storeu_si128(h, hi);
            }
            _ => return false,
        }
    }
    true
}

// ---- fused 4x8/8x4 forward on i16 lanes: C's
// `av1_lowbd_fwd_txfm2d_{4x8,8x4}_sse2` ------------------------------------
//
// Same construction as [`fwd_8x8_fused_i16`], mixed widths: the 4x8 runs an
// 8-pt column kernel on 4-lane registers (the `btf_16_w4_sse2` butterflies,
// `fdct4x8_new_sse2` / `fadst4x8_new_sse2` / `fidentity8x8_new_sse2`) and a
// 4-pt row kernel on full-width registers (`fdct8x4_new_sse2` /
// `fadst8x4_new_sse2` / `fidentity8x4_new_sse2`); the 8x4 mirrors it. Both
// end in the rect-scaled store (`scale_round_sse2` by `NewSqrt2` — the
// scalar path's `mul_rshiftv(_, NEW_SQRT2, NEW_SQRT2_BITS)`).
//
// # The gate
//
// `max|input| <= 1023`, derived by the same sign-vertex simulation over both
// pipeline orders: the binding case is an 8-pt pass feeding `fdct4` (or a
// 4-pt pass feeding `fdct8`) — pass-1 out <= 23,149 -> pass-2 in <= 11,575 <
// `fdct4`'s 11,584 no-saturation input bound. `fadst8x4`'s WRAPPING
// `add_epi16(in0, in1)` (its `in7` term) is covered: it diverges from the
// scalar's i32 add only past |in| ~16,383. bd8 residuals (<=255) always
// pass.

/// Kernel kind for the rect48 family — the SIZE comes from `col_n`/`row_n`.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
enum FwdR {
    Dct,
    Adst,
    Idtx,
}

#[cfg(target_arch = "x86_64")]
fn fwd_r_kernel(txfm_type: i32) -> Option<FwdR> {
    match txfm_type {
        0 | 1 => Some(FwdR::Dct),
        5 | 6 => Some(FwdR::Adst),
        8 | 9 => Some(FwdR::Idtx),
        _ => None,
    }
}

/// Scalar twin — declines, routing the caller to the i32 fused path.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fwd_rect48_fused_i16_scalar(
    _t: archmage::ScalarToken,
    _kc: FwdR,
    _kr: FwdR,
    _input: &[i16],
    _output: &mut [i32],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _cos_bit_col: i32,
    _cos_bit_row: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// The i16 fused 4x8/8x4 forward transforms.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_rect48_fused_i16(
    t: Token,
    kc: FwdR,
    kr: FwdR,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    use archmage::intrinsics::x86_64::*;
    let _ = t;
    if !((col_n == 4 && row_n == 8) || (col_n == 8 && row_n == 4)) {
        return false;
    }

    let pair = |a: i32, b: i32| -> __m128i {
        _mm_set1_epi32((a as u16 as u32 | ((b as u16 as u32) << 16)) as i32)
    };

    // `btf_16_sse2` — full 8-lane butterfly.
    let btf =
        |w0: __m128i, w1: __m128i, i0: __m128i, i1: __m128i, cos_bit: i32| -> (__m128i, __m128i) {
            let rnd = _mm_set1_epi32(1 << (cos_bit - 1));
            let cnt = _mm_cvtsi32_si128(cos_bit);
            let t0 = _mm_unpacklo_epi16(i0, i1);
            let t1 = _mm_unpackhi_epi16(i0, i1);
            let c0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w0), rnd), cnt);
            let c1 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t1, w0), rnd), cnt);
            let d0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w1), rnd), cnt);
            let d1 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t1, w1), rnd), cnt);
            (_mm_packs_epi32(c0, c1), _mm_packs_epi32(d0, d1))
        };

    // `btf_16_w4_sse2` — 4-lane butterfly; note `out1`'s high half is `c0`,
    // verbatim (those lanes are never stored).
    let btf4 =
        |w0: __m128i, w1: __m128i, i0: __m128i, i1: __m128i, cos_bit: i32| -> (__m128i, __m128i) {
            let rnd = _mm_set1_epi32(1 << (cos_bit - 1));
            let cnt = _mm_cvtsi32_si128(cos_bit);
            let t0 = _mm_unpacklo_epi16(i0, i1);
            let c0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w0), rnd), cnt);
            let d0 = _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(t0, w1), rnd), cnt);
            (_mm_packs_epi32(c0, c0), _mm_packs_epi32(d0, c0))
        };

    // `fdct4x8_new_sse2` — 8-pt DCT on eight 4-lane registers.
    let fdct8w4 = |i: &[__m128i; 8], cos_bit: i32| -> [__m128i; 8] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let x1 = [
            _mm_adds_epi16(i[0], i[7]),
            _mm_adds_epi16(i[1], i[6]),
            _mm_adds_epi16(i[2], i[5]),
            _mm_adds_epi16(i[3], i[4]),
            _mm_subs_epi16(i[3], i[4]),
            _mm_subs_epi16(i[2], i[5]),
            _mm_subs_epi16(i[1], i[6]),
            _mm_subs_epi16(i[0], i[7]),
        ];
        let (x2_5, x2_6) = btf4(pair(-c[32], c[32]), pair(c[32], c[32]), x1[5], x1[6], cos_bit);
        let x2 = [
            _mm_adds_epi16(x1[0], x1[3]),
            _mm_adds_epi16(x1[1], x1[2]),
            _mm_subs_epi16(x1[1], x1[2]),
            _mm_subs_epi16(x1[0], x1[3]),
            x1[4],
            x2_5,
            x2_6,
            x1[7],
        ];
        let (x3_0, x3_1) = btf4(pair(c[32], c[32]), pair(c[32], -c[32]), x2[0], x2[1], cos_bit);
        let (x3_2, x3_3) = btf4(pair(c[48], c[16]), pair(-c[16], c[48]), x2[2], x2[3], cos_bit);
        let x3 = [
            x3_0,
            x3_1,
            x3_2,
            x3_3,
            _mm_adds_epi16(x2[4], x2[5]),
            _mm_subs_epi16(x2[4], x2[5]),
            _mm_subs_epi16(x2[7], x2[6]),
            _mm_adds_epi16(x2[7], x2[6]),
        ];
        let (x4_4, x4_7) = btf4(pair(c[56], c[8]), pair(-c[8], c[56]), x3[4], x3[7], cos_bit);
        let (x4_5, x4_6) = btf4(pair(c[24], c[40]), pair(-c[40], c[24]), x3[5], x3[6], cos_bit);
        [x3[0], x4_4, x3[2], x4_6, x3[1], x4_5, x3[3], x4_7]
    };

    // `fadst4x8_new_sse2` — 8-pt ADST on eight 4-lane registers.
    let fadst8w4 = |i: &[__m128i; 8], cos_bit: i32| -> [__m128i; 8] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let z = _mm_setzero_si128();
        let x1 = [
            i[0],
            _mm_subs_epi16(z, i[7]),
            _mm_subs_epi16(z, i[3]),
            i[4],
            _mm_subs_epi16(z, i[1]),
            i[6],
            i[2],
            _mm_subs_epi16(z, i[5]),
        ];
        let (x2_2, x2_3) = btf4(pair(c[32], c[32]), pair(c[32], -c[32]), x1[2], x1[3], cos_bit);
        let (x2_6, x2_7) = btf4(pair(c[32], c[32]), pair(c[32], -c[32]), x1[6], x1[7], cos_bit);
        let x2 = [x1[0], x1[1], x2_2, x2_3, x1[4], x1[5], x2_6, x2_7];
        let x3 = [
            _mm_adds_epi16(x2[0], x2[2]),
            _mm_adds_epi16(x2[1], x2[3]),
            _mm_subs_epi16(x2[0], x2[2]),
            _mm_subs_epi16(x2[1], x2[3]),
            _mm_adds_epi16(x2[4], x2[6]),
            _mm_adds_epi16(x2[5], x2[7]),
            _mm_subs_epi16(x2[4], x2[6]),
            _mm_subs_epi16(x2[5], x2[7]),
        ];
        let (x4_4, x4_5) = btf4(pair(c[16], c[48]), pair(c[48], -c[16]), x3[4], x3[5], cos_bit);
        let (x4_6, x4_7) = btf4(pair(-c[48], c[16]), pair(c[16], c[48]), x3[6], x3[7], cos_bit);
        let x4 = [x3[0], x3[1], x3[2], x3[3], x4_4, x4_5, x4_6, x4_7];
        let x5 = [
            _mm_adds_epi16(x4[0], x4[4]),
            _mm_adds_epi16(x4[1], x4[5]),
            _mm_adds_epi16(x4[2], x4[6]),
            _mm_adds_epi16(x4[3], x4[7]),
            _mm_subs_epi16(x4[0], x4[4]),
            _mm_subs_epi16(x4[1], x4[5]),
            _mm_subs_epi16(x4[2], x4[6]),
            _mm_subs_epi16(x4[3], x4[7]),
        ];
        let (x6_0, x6_1) = btf4(pair(c[4], c[60]), pair(c[60], -c[4]), x5[0], x5[1], cos_bit);
        let (x6_2, x6_3) = btf4(pair(c[20], c[44]), pair(c[44], -c[20]), x5[2], x5[3], cos_bit);
        let (x6_4, x6_5) = btf4(pair(c[36], c[28]), pair(c[28], -c[36]), x5[4], x5[5], cos_bit);
        let (x6_6, x6_7) = btf4(pair(c[52], c[12]), pair(c[12], -c[52]), x5[6], x5[7], cos_bit);
        [x6_1, x6_6, x6_3, x6_4, x6_5, x6_2, x6_7, x6_0]
    };

    // `fidentity8x8_new_sse2` — `adds(v, v)` is the scalar's `2 * v`.
    let fidtx8 = |i: &[__m128i; 8]| -> [__m128i; 8] {
        [
            _mm_adds_epi16(i[0], i[0]),
            _mm_adds_epi16(i[1], i[1]),
            _mm_adds_epi16(i[2], i[2]),
            _mm_adds_epi16(i[3], i[3]),
            _mm_adds_epi16(i[4], i[4]),
            _mm_adds_epi16(i[5], i[5]),
            _mm_adds_epi16(i[6], i[6]),
            _mm_adds_epi16(i[7], i[7]),
        ]
    };

    // `fdct8x4_new_sse2` — 4-pt DCT on four full-width registers.
    let fdct4w8 = |i: &[__m128i; 4], cos_bit: i32| -> [__m128i; 4] {
        let c = crate::transform::cospi::cospi_arr(cos_bit);
        let x1 = [
            _mm_adds_epi16(i[0], i[3]),
            _mm_adds_epi16(i[1], i[2]),
            _mm_subs_epi16(i[1], i[2]),
            _mm_subs_epi16(i[0], i[3]),
        ];
        let (x2_0, x2_1) = btf(pair(c[32], c[32]), pair(c[32], -c[32]), x1[0], x1[1], cos_bit);
        let (x2_2, x2_3) = btf(pair(c[48], c[16]), pair(-c[16], c[48]), x1[2], x1[3], cos_bit);
        [x2_0, x2_2, x2_1, x2_3]
    };

    // `fadst8x4_new_sse2` — 4-pt ADST on four full-width registers; the
    // `in7` term is a WRAPPING `add_epi16`, exact under the gate.
    let fadst4w8 = |i: &[__m128i; 4], cos_bit: i32| -> [__m128i; 4] {
        let s = crate::transform::cospi::sinpi_arr(cos_bit);
        let z = _mm_setzero_si128();
        let rnd = _mm_set1_epi32(1 << (cos_bit - 1));
        let cnt = _mm_cvtsi32_si128(cos_bit);
        let s33 = _mm_set1_epi16(s[3] as i16);
        let in7 = _mm_add_epi16(i[0], i[1]);
        let u_lo = [
            _mm_unpacklo_epi16(i[0], i[1]),
            _mm_unpacklo_epi16(i[2], i[3]),
            _mm_unpacklo_epi16(in7, z),
            _mm_unpacklo_epi16(i[2], z),
            _mm_unpacklo_epi16(i[3], z),
        ];
        let u_hi = [
            _mm_unpackhi_epi16(i[0], i[1]),
            _mm_unpackhi_epi16(i[2], i[3]),
            _mm_unpackhi_epi16(in7, z),
            _mm_unpackhi_epi16(i[2], z),
            _mm_unpackhi_epi16(i[3], z),
        ];
        let v_lo = [
            _mm_madd_epi16(u_lo[0], pair(s[1], s[2])),
            _mm_madd_epi16(u_lo[1], pair(s[3], s[4])),
            _mm_madd_epi16(u_lo[2], s33),
            _mm_madd_epi16(u_lo[0], pair(s[4], -s[1])),
            _mm_madd_epi16(u_lo[1], pair(-s[3], s[2])),
            _mm_madd_epi16(u_lo[3], s33),
            _mm_madd_epi16(u_lo[4], s33),
        ];
        let v_hi = [
            _mm_madd_epi16(u_hi[0], pair(s[1], s[2])),
            _mm_madd_epi16(u_hi[1], pair(s[3], s[4])),
            _mm_madd_epi16(u_hi[2], s33),
            _mm_madd_epi16(u_hi[0], pair(s[4], -s[1])),
            _mm_madd_epi16(u_hi[1], pair(-s[3], s[2])),
            _mm_madd_epi16(u_hi[3], s33),
            _mm_madd_epi16(u_hi[4], s33),
        ];
        let w_lo = [
            _mm_add_epi32(v_lo[0], v_lo[1]),
            _mm_sub_epi32(v_lo[2], v_lo[6]),
            _mm_add_epi32(v_lo[3], v_lo[4]),
        ];
        let w_hi = [
            _mm_add_epi32(v_hi[0], v_hi[1]),
            _mm_sub_epi32(v_hi[2], v_hi[6]),
            _mm_add_epi32(v_hi[3], v_hi[4]),
        ];
        let x_lo = [
            _mm_sub_epi32(w_lo[2], w_lo[0]),
            _mm_sub_epi32(_mm_slli_epi32::<2>(v_lo[5]), v_lo[5]),
        ];
        let x_hi = [
            _mm_sub_epi32(w_hi[2], w_hi[0]),
            _mm_sub_epi32(_mm_slli_epi32::<2>(v_hi[5]), v_hi[5]),
        ];
        let y_lo = _mm_add_epi32(x_lo[0], x_lo[1]);
        let y_hi = _mm_add_epi32(x_hi[0], x_hi[1]);
        let sra = |v: __m128i| _mm_sra_epi32(_mm_add_epi32(v, rnd), cnt);
        [
            _mm_packs_epi32(sra(w_lo[0]), sra(w_hi[0])),
            _mm_packs_epi32(sra(w_lo[1]), sra(w_hi[1])),
            _mm_packs_epi32(sra(w_lo[2]), sra(w_hi[2])),
            _mm_packs_epi32(sra(y_lo), sra(y_hi)),
        ]
    };

    // `fidentity8x4_new_sse2` — `scale_round(v, NewSqrt2)` per lane, the
    // scalar's `round_shift(v * NEW_SQRT2, NEW_SQRT2_BITS)`.
    let fidtx4 = |i: &[__m128i; 4]| -> [__m128i; 4] {
        let one = _mm_set1_epi16(1);
        let sr = pair(NEW_SQRT2, 1 << (NEW_SQRT2_BITS - 1));
        let mut o = [_mm_setzero_si128(); 4];
        for (o, i) in o.iter_mut().zip(i.iter()) {
            let a_lo = _mm_unpacklo_epi16(*i, one);
            let a_hi = _mm_unpackhi_epi16(*i, one);
            let b_lo = _mm_srai_epi32::<NEW_SQRT2_BITS>(_mm_madd_epi16(a_lo, sr));
            let b_hi = _mm_srai_epi32::<NEW_SQRT2_BITS>(_mm_madd_epi16(a_hi, sr));
            *o = _mm_packs_epi32(b_lo, b_hi);
        }
        o
    };

    let run8 = |k: FwdR, i: &[__m128i; 8], cos_bit: i32| -> [__m128i; 8] {
        match k {
            FwdR::Dct => fdct8w4(i, cos_bit),
            FwdR::Adst => fadst8w4(i, cos_bit),
            FwdR::Idtx => fidtx8(i),
        }
    };
    let run4 = |k: FwdR, i: &[__m128i; 4], cos_bit: i32| -> [__m128i; 4] {
        match k {
            FwdR::Dct => fdct4w8(i, cos_bit),
            FwdR::Adst => fadst4w8(i, cos_bit),
            FwdR::Idtx => fidtx4(i),
        }
    };

    // Loads, gate, column pass, `round_shift(-1)`, transpose, lr_flip, row
    // pass — the shared skeleton. Register = source row (ud_flip reversed),
    // lane = column; after the transpose register = output column.
    let mut mx = _mm_setzero_si128();
    let mut row_pass_out = [_mm_setzero_si128(); 8];
    if col_n == 4 {
        // 4x8: 8 rows x 4 cols (`load_buffer_16bit_to_16bit_w4`), 8-pt col.
        let mut b = [_mm_setzero_si128(); 8];
        for (r, v) in b.iter_mut().enumerate() {
            let src = if ud_flip { 7 - r } else { r };
            let row: &[i16; 4] = match input
                .get(src * stride..src * stride + 4)
                .and_then(|s| s.try_into().ok())
            {
                Some(a) => a,
                None => return false,
            };
            let rv = _mm_loadu_si64(row);
            mx = _mm_max_epu16(mx, _mm_abs_epi16(rv));
            *v = _mm_slli_epi16::<2>(rv);
        }
        if _mm_testz_si128(
            _mm_subs_epu16(mx, _mm_set1_epi16(1023)),
            _mm_subs_epu16(mx, _mm_set1_epi16(1023)),
        ) == 0
        {
            return false;
        }
        let mut col = run8(kc, &b, cos_bit_col);
        for v in col.iter_mut() {
            *v = _mm_srai_epi16::<1>(_mm_adds_epi16(*v, _mm_set1_epi16(1)));
        }
        // `transpose_16bit_4x8`: 8 regs (4 live lanes) -> 4 regs (8 lanes).
        let a0 = _mm_unpacklo_epi16(col[0], col[1]);
        let a1 = _mm_unpacklo_epi16(col[2], col[3]);
        let a2 = _mm_unpacklo_epi16(col[4], col[5]);
        let a3 = _mm_unpacklo_epi16(col[6], col[7]);
        let b0 = _mm_unpacklo_epi32(a0, a1);
        let b1 = _mm_unpacklo_epi32(a2, a3);
        let b2 = _mm_unpackhi_epi32(a0, a1);
        let b3 = _mm_unpackhi_epi32(a2, a3);
        let mut tr = [
            _mm_unpacklo_epi64(b0, b1),
            _mm_unpackhi_epi64(b0, b1),
            _mm_unpacklo_epi64(b2, b3),
            _mm_unpackhi_epi64(b2, b3),
        ];
        if lr_flip {
            tr.reverse();
        }
        row_pass_out[..4].copy_from_slice(&run4(kr, &tr, cos_bit_row));
    } else {
        // 8x4: 4 rows x 8 cols, 4-pt col; two zero regs pad the 8x8
        // transpose (C reads uninitialized buf0[4..8]; only the low 4 lanes
        // of the transpose output are ever stored).
        let mut b = [_mm_setzero_si128(); 8];
        for (r, v) in b[..4].iter_mut().enumerate() {
            let src = if ud_flip { 3 - r } else { r };
            let row: &[i16; 8] = match input
                .get(src * stride..src * stride + 8)
                .and_then(|s| s.try_into().ok())
            {
                Some(a) => a,
                None => return false,
            };
            let rv = _mm_loadu_si128(row);
            mx = _mm_max_epu16(mx, _mm_abs_epi16(rv));
            *v = _mm_slli_epi16::<2>(rv);
        }
        if _mm_testz_si128(
            _mm_subs_epu16(mx, _mm_set1_epi16(1023)),
            _mm_subs_epu16(mx, _mm_set1_epi16(1023)),
        ) == 0
        {
            return false;
        }
        let mut col4 = [_mm_setzero_si128(); 4];
        col4.copy_from_slice(&b[..4]);
        let mut col = run4(kc, &col4, cos_bit_col);
        for v in col.iter_mut() {
            *v = _mm_srai_epi16::<1>(_mm_adds_epi16(*v, _mm_set1_epi16(1)));
        }
        // `transpose_16bit_8x4`: 4 regs (8 lanes) -> 8 regs (4 live lanes).
        let a0 = _mm_unpacklo_epi16(col[0], col[1]);
        let a1 = _mm_unpacklo_epi16(col[2], col[3]);
        let a4 = _mm_unpackhi_epi16(col[0], col[1]);
        let a5 = _mm_unpackhi_epi16(col[2], col[3]);
        let b0 = _mm_unpacklo_epi32(a0, a1);
        let b2 = _mm_unpacklo_epi32(a4, a5);
        let b4 = _mm_unpackhi_epi32(a0, a1);
        let b6 = _mm_unpackhi_epi32(a4, a5);
        let zz = _mm_setzero_si128();
        let mut tr = [
            _mm_unpacklo_epi64(b0, zz),
            _mm_unpackhi_epi64(b0, zz),
            _mm_unpacklo_epi64(b4, zz),
            _mm_unpackhi_epi64(b4, zz),
            _mm_unpacklo_epi64(b2, zz),
            _mm_unpackhi_epi64(b2, zz),
            _mm_unpacklo_epi64(b6, zz),
            _mm_unpackhi_epi64(b6, zz),
        ];
        if lr_flip {
            tr.reverse();
        }
        row_pass_out = run8(kr, &tr, cos_bit_row);
    }
    // shift[2] == 0; `rect_type == +-1` so the store applies NewSqrt2.

    // `store_rect_buffer_16bit_to_32bit_w{8,4}`: `scale_round(v, NewSqrt2)`
    // then sign-extend; `output[c*row_n + r]`.
    let one = _mm_set1_epi16(1);
    let sr = pair(NEW_SQRT2, 1 << (NEW_SQRT2_BITS - 1));
    for (c, v) in row_pass_out[..col_n].iter().enumerate() {
        let a_lo = _mm_unpacklo_epi16(*v, one);
        let a_hi = _mm_unpackhi_epi16(*v, one);
        let b_lo = _mm_srai_epi32::<NEW_SQRT2_BITS>(_mm_madd_epi16(a_lo, sr));
        let b_hi = _mm_srai_epi32::<NEW_SQRT2_BITS>(_mm_madd_epi16(a_hi, sr));
        let o = match output.get_mut(c * row_n..c * row_n + row_n) {
            Some(o) => o,
            None => return false,
        };
        let l: &mut [i32; 4] = match (&mut o[..4]).try_into() {
            Ok(l) => l,
            Err(_) => return false,
        };
        _mm_storeu_si128(l, b_lo);
        if row_n == 8 {
            let h: &mut [i32; 4] = match (&mut o[4..]).try_into() {
                Ok(h) => h,
                Err(_) => return false,
            };
            _mm_storeu_si128(h, b_hi);
        }
    }
    true
}

pub(crate) fn try_fwd_col_pass(
    txfm_type_col: i32,
    input: &[i16],
    buf: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
    shift1_bit: i32,
    cos_bit_col: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    // `col_n == 4` runs a HALF-FILLED 8-lane batch, exactly as
    // [`try_inv_col_pass`] already does. It is the same trade and the same
    // `half_batch_pays` policy, and it is worth taking here for the reason
    // KB-PERF-5 records: the competition is not a full-width vector pass but
    // the driver's SCALAR per-column loop, so 4 live lanes is still ~4x. Both
    // ends stay contiguous at 4-wide (an i16 run of the source row in, a
    // 4-entry run of `buf` out), so this is the inverse COLUMN pass's shape,
    // not the row pass's gather.
    if col_n % 8 != 0 && !(col_n == 4 && half_batch_pays(row_n)) {
        return false;
    }
    let _ = crate::dispatch::scalar_forced();
    // The i16-lane forward column pass — 16 columns per vector. Unlike the
    // inverse's bd8 gate (which is a static property of the stage_range /
    // clamp constants), the forward gate is a RUNTIME bound on the actual
    // block: `max|input| << shift0 <= M*`, the largest input for which
    // `xtask/audit_i16_fwd.py` proves every value of that kernel stays inside
    // i16. Sound for any caller of the public `av1_fwd_txfm2d`, and it
    // declines BEFORE touching `buf`. `col_n % 16` because a 4- or 8-wide
    // block would run the same kernel instruction count as the i32x8 pass plus
    // the narrowing overhead.
    // `shift0 in {0, 2}` and `shift1_bit in 0..=4` are the whole of
    // `FWD_SHIFT`'s column column today, and they are exactly the domains the
    // i16 pass's two shift recipes are proved on (`v+v; d+d` == `<< 2`, and
    // `rshift_mul` == `round_shift(_, bit)` for bit in 1..=4). Gating on them
    // rather than assuming them means a future shift table cannot silently
    // walk off either proof.
    if col_n % 16 == 0 && (shift0 == 0 || shift0 == 2) && (0..=4).contains(&shift1_bit) {
        if let Some(k16) = lowbd16_fwd::fwd_kernel_i16(txfm_type_col) {
            debug_assert_eq!(lowbd16_fwd::fwd_kernel_i16_n(k16), row_n);
            if lowbd16_fwd::fwd_col_i16_applies(k16, input, stride, col_n, row_n, shift0)
                && incant!(
                    lowbd16_fwd::fwd_col_pass_i16(
                        k16, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col,
                        ud_flip, lr_flip
                    ),
                    [v3, neon, scalar]
                )
            {
                return true;
            }
        }
    }
    let Some(kernel) = fwd_kernel(txfm_type_col) else {
        return false;
    };
    debug_assert_eq!(fwd_kernel_n(kernel), row_n); // col kernel spans the H points
    incant!(
        fwd_col_pass(
            kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col, ud_flip,
            lr_flip
        ),
        [v3, neon, scalar]
    )
}

/// The lane-batched forward column pass (8 columns per iteration).
///
/// The vector scratch is TIERED by `row_n` (8/16/64 lane vectors) for exactly
/// the reason [`inv_row_pass`] tiers by `col_n` and `lowbd16.rs:132` states: a
/// flat `[i32x8; 64]` zero-init compiles to a 2 KiB memset per array, which
/// dominates the small transforms. The forward passes were the only two in this
/// file that never got the treatment, and the 2026-08-02 encoder re-profile
/// measured them as the top TWO allocator/memset callers in the whole encode
/// (19.2 % of that class). The core is a separate `#[magetypes]` body, so each
/// arm inlines it with its exactly-sized scratch.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_col_pass(
    t: Token,
    kernel: Fwd1d,
    input: &[i16],
    buf: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
    shift1_bit: i32,
    cos_bit_col: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    debug_assert!(row_n <= 64 && (col_n % 8 == 0 || col_n == 4));
    if row_n <= 8 {
        let mut tin = [i32x8::zero(t); 8];
        let mut tout = [i32x8::zero(t); 8];
        incant!(fwd_col_pass_core(kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col,
            ud_flip, lr_flip, &mut tin, &mut tout,
        ), [v3, neon]);
    } else if row_n <= 16 {
        let mut tin = [i32x8::zero(t); 16];
        let mut tout = [i32x8::zero(t); 16];
        incant!(fwd_col_pass_core(kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col,
            ud_flip, lr_flip, &mut tin, &mut tout,
        ), [v3, neon]);
    } else {
        let mut tin = [i32x8::zero(t); 64];
        let mut tout = [i32x8::zero(t); 64];
        incant!(fwd_col_pass_core(kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col,
            ud_flip, lr_flip, &mut tin, &mut tout,
        ), [v3, neon]);
    }
    true
}

/// The forward column-pass body over caller-sized scratch (see [`fwd_col_pass`]).
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_col_pass_core(
    t: Token,
    kernel: Fwd1d,
    input: &[i16],
    buf: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
    shift1_bit: i32,
    cos_bit_col: i32,
    ud_flip: bool,
    lr_flip: bool,
    tin: &mut [I32x8<Token>],
    tout: &mut [I32x8<Token>],
) {
    let sr = [0i8; 12]; // fwd kernels ignore stage_range
    let mut cg = 0usize;
    while cg < col_n {
        let active = (col_n - cg).min(8); // 8, or 4 (col_n == 4)
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let src_r = if ud_flip { row_n - r - 1 } else { r };
            let off = src_r * stride + cg;
            let mut v = if active == 8 {
                widen16(t, &input[off..off + 8])
            } else {
                // The idle lanes are ZERO, and they stay inert end to end: the
                // 1-D kernels are linear, `shl_clamp64v`/`rshiftv` map 0 to 0,
                // and only `active` lanes are stored. So a half batch computes
                // the same values for its live columns as a full one.
                let a: [i16; 4] = input[off..off + 4].try_into().unwrap();
                i32x8::from_array(
                    t,
                    [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32, 0, 0, 0, 0],
                )
            };
            if shift0 > 0 {
                // round_shift_array(temp_in, -shift[0]) with shift[0]=2 →
                // the NEGATIVE-bit arm: (v << 2) clamped to i32 in i64.
                v = shl_clamp64v(t, v, shift0);
            }
            *ti = v;
        }
        incant!(run_fwd1d(kernel, &tin[..row_n], &mut tout[..row_n], cos_bit_col, &sr), [v3, neon]);
        for (r, to) in tout[..row_n].iter_mut().enumerate() {
            let v = if shift1_bit > 0 { rshiftv(t, *to, shift1_bit) } else { *to };
            // Scalar: buf[r*col_n + dst_c] = temp_out[r], dst_c lr-flipped.
            // Lane j holds source column `cg + j`, so under lr_flip it lands at
            // `col_n - 1 - (cg + j)` — a descending run based at
            // `col_n - cg - active`, i.e. the lanes reversed. That is the same
            // expression the 8-lane arm used, with `active` in place of 8.
            if lr_flip {
                let base = r * col_n + (col_n - cg - active);
                if active == 8 {
                    revv(t, v).store((&mut buf[base..base + 8]).try_into().unwrap());
                } else {
                    let a = v.to_array();
                    buf[base..base + 4].copy_from_slice(&[a[3], a[2], a[1], a[0]]);
                }
            } else {
                let base = r * col_n + cg;
                if active == 8 {
                    v.store((&mut buf[base..base + 8]).try_into().unwrap());
                } else {
                    buf[base..base + 4].copy_from_slice(&v.to_array()[..4]);
                }
            }
        }
        cg += active;
    }
}

/// The `incant!` fallback for [`fwd_col_pass`] when NO vector tier is available —
/// x86-64 without AVX2, or every token disabled by the `AOM_FORCE_SCALAR` pin.
/// Declining here is what routes the caller back to its scalar loop, so the
/// pin and the no-AVX2 path take the SAME `false` branch the pre-SIMD code
/// took. There is deliberately no scalar *implementation* of the pass: the
/// scalar twin is the driver's own per-column/row loop, which is the
/// differential's reference.
#[allow(clippy::too_many_arguments)]
fn fwd_col_pass_scalar(
    _: ScalarToken,
    _kernel: Fwd1d,
    _input: &[i16],
    _buf: &mut [i32],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _shift0: i32,
    _shift1_bit: i32,
    _cos_bit_col: i32,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// Vector ROW pass of `fwd_txfm2d_core` — 8 rows per lane batch. Strided
/// loads from row-major `buf` via 8x8 transposes (per-lane gather for the
/// W=4 tail), the row kernel, `round_shift_array(-shift[2])`, the optional
/// NewSqrt2 rect scaling (AFTER the shift, matching the scalar order), then
/// contiguous stores (`output[c*row_n + r..r+8]` — output is column-major).
/// Returns `false` → caller runs the scalar loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_row_pass(
    txfm_type_row: i32,
    buf: &[i32],
    output: &mut [i32],
    col_n: usize,
    row_n: usize,
    shift2_bit: i32,
    cos_bit_row: i32,
    rect1: bool,
) -> bool {
    // `col_n < 8` degrades this pass's LOADS to per-lane gather — no full 8x8
    // transpose tile exists — and unlike the half-batch trade above, that is
    // NOT repaid by a bigger kernel: `fwd_txfm::04x16_dct` (col_n 4, row_n 16)
    // measured +9.8% with it live, so `half_batch_pays(row_n)` would say yes
    // where the bench says no. Hence the flat gate here.
    //
    // The INVERSE row pass is deliberately NOT gated on col_n: its loads are
    // contiguous and only its STORES scatter, and it is the biggest 4-wide win
    // in the sweep (`inv_txfm_u8::04x16_dct` −36.2%). Gathers cost; scatters
    // don't.
    if row_n % 8 != 0 || (col_n < 8 && cfg!(target_arch = "aarch64")) {
        return false;
    }
    let _ = crate::dispatch::scalar_forced();
    // The i16-lane forward ROW pass — 16 rows per vector, gated the same way
    // as the column pass above: a runtime bound on the actual `buf`, so the
    // i32 column pass having produced it (or the caller having passed
    // anything at all) is irrelevant to soundness.
    if row_n % 16 == 0 && col_n % 8 == 0 && (0..=4).contains(&shift2_bit) {
        if let Some(k16) = lowbd16_fwd::fwd_kernel_i16(txfm_type_row) {
            debug_assert_eq!(lowbd16_fwd::fwd_kernel_i16_n(k16), col_n);
            if lowbd16_fwd::fwd_row_i16_applies(k16, buf, col_n, row_n)
                && incant!(
                    lowbd16_fwd::fwd_row_pass_i16(
                        k16, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1
                    ),
                    [v3, neon, scalar]
                )
            {
                return true;
            }
        }
    }
    let Some(kernel) = fwd_kernel(txfm_type_row) else {
        return false;
    };
    debug_assert_eq!(fwd_kernel_n(kernel), col_n); // row kernel spans the W points
    incant!(
        fwd_row_pass(kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1),
        [v3, neon, scalar]
    )
}

/// The lane-batched forward row pass (8 rows per iteration).
///
/// TIERED by `col_n` (8/16/64 lane vectors) — the same treatment, and for the
/// same measured reason, as [`fwd_col_pass`] above.
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_row_pass(
    t: Token,
    kernel: Fwd1d,
    buf: &[i32],
    output: &mut [i32],
    col_n: usize,
    row_n: usize,
    shift2_bit: i32,
    cos_bit_row: i32,
    rect1: bool,
) -> bool {
    debug_assert!(col_n <= 64 && row_n % 8 == 0);
    if col_n <= 8 {
        let mut tin = [i32x8::zero(t); 8];
        let mut tout = [i32x8::zero(t); 8];
        incant!(fwd_row_pass_core(kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1,
            &mut tin, &mut tout,
        ), [v3, neon]);
    } else if col_n <= 16 {
        let mut tin = [i32x8::zero(t); 16];
        let mut tout = [i32x8::zero(t); 16];
        incant!(fwd_row_pass_core(kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1,
            &mut tin, &mut tout,
        ), [v3, neon]);
    } else {
        let mut tin = [i32x8::zero(t); 64];
        let mut tout = [i32x8::zero(t); 64];
        incant!(fwd_row_pass_core(kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1,
            &mut tin, &mut tout,
        ), [v3, neon]);
    }
    true
}

/// The forward row-pass body over caller-sized scratch (see [`fwd_row_pass`]).
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn fwd_row_pass_core(
    t: Token,
    kernel: Fwd1d,
    buf: &[i32],
    output: &mut [i32],
    col_n: usize,
    row_n: usize,
    shift2_bit: i32,
    cos_bit_row: i32,
    rect1: bool,
    tin: &mut [I32x8<Token>],
    tout: &mut [I32x8<Token>],
) {
    let sr = [0i8; 12];
    for rg in (0..row_n).step_by(8) {
        // Load: tin[c].lane(k) = buf[(rg+k)*col_n + c] — transpose 8x8 tiles
        // (contiguous row loads), per-lane gather for the W=4 tail.
        let full = col_n & !7;
        for cg in (0..full).step_by(8) {
            let mut rows = [i32x8::zero(t); 8];
            for (k, rk) in rows.iter_mut().enumerate() {
                let base = (rg + k) * col_n + cg;
                *rk = i32x8::from_slice(t, &buf[base..base + 8]);
            }
            let tr = transpose8(t, &rows);
            tin[cg..cg + 8].copy_from_slice(&tr);
        }
        for c in full..col_n {
            tin[c] = i32x8::from_array(t, core::array::from_fn(|k| buf[(rg + k) * col_n + c]));
        }
        incant!(run_fwd1d(kernel, &tin[..col_n], &mut tout[..col_n], cos_bit_row, &sr), [v3, neon]);
        for (c, to) in tout[..col_n].iter_mut().enumerate() {
            let mut v = *to;
            if shift2_bit > 0 {
                v = rshiftv(t, v, shift2_bit); // round_shift_array(-shift[2])
            }
            if rect1 {
                // round_shift(v * NewSqrt2, NewSqrt2Bits) — AFTER the shift.
                v = mul_rshiftv(t, v, NEW_SQRT2, NEW_SQRT2_BITS);
            }
            // Scalar: output[c*row_n + r] = row_buffer[c] — contiguous per c.
            let base = c * row_n + rg;
            v.store((&mut output[base..base + 8]).try_into().unwrap());
        }
    }
}

/// The `incant!` fallback for [`fwd_row_pass`] when NO vector tier is available —
/// x86-64 without AVX2, or every token disabled by the `AOM_FORCE_SCALAR` pin.
/// Declining here is what routes the caller back to its scalar loop, so the
/// pin and the no-AVX2 path take the SAME `false` branch the pre-SIMD code
/// took. There is deliberately no scalar *implementation* of the pass: the
/// scalar twin is the driver's own per-column/row loop, which is the
/// differential's reference.
#[allow(clippy::too_many_arguments)]
fn fwd_row_pass_scalar(
    _: ScalarToken,
    _kernel: Fwd1d,
    _buf: &[i32],
    _output: &mut [i32],
    _col_n: usize,
    _row_n: usize,
    _shift2_bit: i32,
    _cos_bit_row: i32,
    _rect1: bool,
) -> bool {
    false
}

/// The lane-batched column pass body — the scalar per-column loop of
/// `av1_inv_txfm2d_add`, 8 columns per iteration (module docs carry the
/// per-stage exactness argument).
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass(
    t: Token,
    kernel: Inv1d,
    buf: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    debug_assert!(row_n <= 64 && (col_n % 8 == 0 || col_n == 4));
    if row_n <= 8 {
        let mut tin = [i32x8::zero(t); 8];
        let mut tout = [i32x8::zero(t); 8];
        incant!(inv_col_pass_core(kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range,
            ud_flip, lr_flip, bd, &mut tin, &mut tout,
        ), [v3, neon]);
    } else if row_n <= 16 {
        let mut tin = [i32x8::zero(t); 16];
        let mut tout = [i32x8::zero(t); 16];
        incant!(inv_col_pass_core(kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range,
            ud_flip, lr_flip, bd, &mut tin, &mut tout,
        ), [v3, neon]);
    } else {
        let mut tin = [i32x8::zero(t); 64];
        let mut tout = [i32x8::zero(t); 64];
        incant!(inv_col_pass_core(kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range,
            ud_flip, lr_flip, bd, &mut tin, &mut tout,
        ), [v3, neon]);
    }
    true
}

/// The `incant!` fallback for [`inv_col_pass`] when NO vector tier is available —
/// x86-64 without AVX2, or every token disabled by the `AOM_FORCE_SCALAR` pin.
/// Declining here is what routes the caller back to its scalar loop, so the
/// pin and the no-AVX2 path take the SAME `false` branch the pre-SIMD code
/// took. There is deliberately no scalar *implementation* of the pass: the
/// scalar twin is the driver's own per-column/row loop, which is the
/// differential's reference.
#[allow(clippy::too_many_arguments)]
fn inv_col_pass_scalar(
    _: ScalarToken,
    _kernel: Inv1d,
    _buf: &[i32],
    _output: &mut [u16],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _shift1_bit: i32,
    _col_clamp: i8,
    _stage_range: &[i8; 12],
    _ud_flip: bool,
    _lr_flip: bool,
    _bd: i32,
) -> bool {
    false
}

/// The column-pass body over caller-sized scratch (see [`inv_row_pass`] for
/// the tiering rationale).
#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass_core(
    t: Token,
    kernel: Inv1d,
    buf: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
    tin: &mut [I32x8<Token>],
    tout: &mut [I32x8<Token>],
) {
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, (1i32 << bd) - 1);
    let mut c = 0usize;
    while c < col_n {
        let active = (col_n - c).min(8); // 8, or 4 (col_n == 4)
        // Gather the column group: under lr_flip, scalar output column `c+j`
        // reads buf column `col_n-1-(c+j)` — for a full group that is the
        // ascending 8-column load at `col_n-c-8`, lanes reversed; for the
        // 4-active group it is the row's 4 entries reversed into lanes 0..4.
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let v = if active == 8 {
                if lr_flip {
                    let base = r * col_n + (col_n - c - 8);
                    revv(t, i32x8::from_slice(t, &buf[base..base + 8]))
                } else {
                    let base = r * col_n + c;
                    i32x8::from_slice(t, &buf[base..base + 8])
                }
            } else {
                let a: [i32; 4] = buf[r * col_n..r * col_n + 4].try_into().unwrap();
                if lr_flip {
                    i32x8::from_array(t, [a[3], a[2], a[1], a[0], 0, 0, 0, 0])
                } else {
                    i32x8::from_array(t, [a[0], a[1], a[2], a[3], 0, 0, 0, 0])
                }
            };
            *ti = clampv(t, v, col_clamp); // the driver's clamp_buf
        }
        let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;
        incant!(run_inv1d(kernel, &tin[..row_n], &mut tout[..row_n], cos_bit, stage_range), [v3, neon]);
        // round_shift_array(to, -shift[1]) — shift[1] is always negative for
        // the inverse sizes, so this is the positive-bit arm.
        for to in tout[..row_n].iter_mut() {
            *to = rshiftv(t, *to, shift1_bit);
        }
        // Reconstruction: output row r takes tout[row_n-1-r] under ud_flip.
        for r in 0..row_n {
            let src = tout[if ud_flip { row_n - r - 1 } else { r }];
            let idx = r * stride + c;
            let dv = if active == 8 {
                let d: [u16; 8] = output[idx..idx + 8].try_into().unwrap();
                i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32))
            } else {
                let d: [u16; 4] = output[idx..idx + 4].try_into().unwrap();
                i32x8::from_array(t, [d[0] as i32, d[1] as i32, d[2] as i32, d[3] as i32, 0, 0, 0, 0])
            };
            // (dest + trans) wraps i32 like the scalar wrapping_add, then
            // clamps to the pixel range — `as u16` is exact after the clamp.
            let s = (dv + src).clamp(zero, pix_hi).to_array();
            for (j, &sv) in s.iter().take(active).enumerate() {
                output[idx + j] = sv as u16;
            }
        }
        c += active;
    }
}

// ---- lowbd (bd8, u8 pixel) inverse column pass --------------------------------
//
// The bd8 "lowbd" decode pipeline stores reconstruction planes as `u8` instead
// of `u16`. The inverse-transform ROW pass ([`try_inv_row_pass`]) is pixel-type
// independent (it writes the i32 `buf`), so lowbd REUSES it verbatim; only the
// COLUMN pass touches pixels. This is the byte-for-byte twin of
// [`inv_col_pass_core`] with the destination loads/stores narrowed to `u8` and
// the pixel ceiling fixed at 255 (bd == 8): every i32-domain lane op — the
// column gather + clamp, the 1-D kernel, the round-shift, and the
// `(dest + trans).clamp(0, 255)` reconstruction — is identical, so a lane that
// stores value `v` here stores the SAME `v` the u16 core would (the u16 core
// also clamps to `(1<<8)-1 == 255` at bd8). The intermediate butterfly
// precision is UNNARROWED (still i32) — this is the "safe first step": only the
// destination storage changes width, which cannot move a pixel.

/// bd8/u8 counterpart of [`try_inv_col_pass`]. `bd` is fixed at 8, so the pixel
/// ceiling is 255 and the column clamp is 16 (`(8+6).max(16)`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_col_pass_u8(
    txfm_type_col: i32,
    buf: &[i32],
    output: &mut [u8],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    if col_n % 8 != 0 && !(col_n == 4 && half_batch_pays(row_n)) {
        return false;
    }
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    // Phase C: the audited DCT column kernels run on i16 lanes (16 columns per
    // vector). Preconditions are the bd8 structural constants — asserted, not
    // assumed: every caller of the u8 entry passes exactly these. Both vector
    // tiers run it (see `try_inv_row_pass`); the `scalar` arm declines.
    if let Some(k16) = lowbd16::inv_kernel_i16(txfm_type_col) {
        debug_assert_eq!(lowbd16::inv_kernel_i16_n(k16), row_n);
        debug_assert!(stage_range.iter().all(|&b| b == 16));
        if shift1_bit == 4
            && col_clamp == 16
            && incant!(
                lowbd16::inv_col_pass_u8_i16(
                    k16, buf, output, stride, col_n, row_n, ud_flip, lr_flip
                ),
                [v3, neon, scalar]
            )
        {
            return true;
        }
    }
    let Some(kernel) = inv_kernel(txfm_type_col) else {
        return false;
    };
    debug_assert_eq!(inv_kernel_n(kernel), row_n);
    incant!(
        inv_col_pass_u8(
            kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range, ud_flip,
            lr_flip
        ),
        [v3, neon, scalar]
    )
}

#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass_u8(
    t: Token,
    kernel: Inv1d,
    buf: &[i32],
    output: &mut [u8],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    debug_assert!(row_n <= 64 && (col_n % 8 == 0 || col_n == 4));
    if row_n <= 8 {
        let mut tin = [i32x8::zero(t); 8];
        let mut tout = [i32x8::zero(t); 8];
        incant!(inv_col_pass_u8_core(kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range,
            ud_flip, lr_flip, &mut tin, &mut tout,
        ), [v3, neon]);
    } else if row_n <= 16 {
        let mut tin = [i32x8::zero(t); 16];
        let mut tout = [i32x8::zero(t); 16];
        incant!(inv_col_pass_u8_core(kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range,
            ud_flip, lr_flip, &mut tin, &mut tout,
        ), [v3, neon]);
    } else {
        let mut tin = [i32x8::zero(t); 64];
        let mut tout = [i32x8::zero(t); 64];
        incant!(inv_col_pass_u8_core(kernel, buf, output, stride, col_n, row_n, shift1_bit, col_clamp, stage_range,
            ud_flip, lr_flip, &mut tin, &mut tout,
        ), [v3, neon]);
    }
    true
}

/// The `incant!` fallback for [`inv_col_pass_u8`] when NO vector tier is available —
/// x86-64 without AVX2, or every token disabled by the `AOM_FORCE_SCALAR` pin.
/// Declining here is what routes the caller back to its scalar loop, so the
/// pin and the no-AVX2 path take the SAME `false` branch the pre-SIMD code
/// took. There is deliberately no scalar *implementation* of the pass: the
/// scalar twin is the driver's own per-column/row loop, which is the
/// differential's reference.
#[allow(clippy::too_many_arguments)]
fn inv_col_pass_u8_scalar(
    _: ScalarToken,
    _kernel: Inv1d,
    _buf: &[i32],
    _output: &mut [u8],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _shift1_bit: i32,
    _col_clamp: i8,
    _stage_range: &[i8; 12],
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

#[magetypes(define(i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass_u8_core(
    t: Token,
    kernel: Inv1d,
    buf: &[i32],
    output: &mut [u8],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    tin: &mut [I32x8<Token>],
    tout: &mut [I32x8<Token>],
) {
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, 255); // (1<<8)-1
    let mut c = 0usize;
    while c < col_n {
        let active = (col_n - c).min(8);
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let v = if active == 8 {
                if lr_flip {
                    let base = r * col_n + (col_n - c - 8);
                    revv(t, i32x8::from_slice(t, &buf[base..base + 8]))
                } else {
                    let base = r * col_n + c;
                    i32x8::from_slice(t, &buf[base..base + 8])
                }
            } else {
                let a: [i32; 4] = buf[r * col_n..r * col_n + 4].try_into().unwrap();
                if lr_flip {
                    i32x8::from_array(t, [a[3], a[2], a[1], a[0], 0, 0, 0, 0])
                } else {
                    i32x8::from_array(t, [a[0], a[1], a[2], a[3], 0, 0, 0, 0])
                }
            };
            *ti = clampv(t, v, col_clamp);
        }
        let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;
        incant!(run_inv1d(kernel, &tin[..row_n], &mut tout[..row_n], cos_bit, stage_range), [v3, neon]);
        for to in tout[..row_n].iter_mut() {
            *to = rshiftv(t, *to, shift1_bit);
        }
        for r in 0..row_n {
            let src = tout[if ud_flip { row_n - r - 1 } else { r }];
            let idx = r * stride + c;
            let dv = if active == 8 {
                let d: [u8; 8] = output[idx..idx + 8].try_into().unwrap();
                i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32))
            } else {
                let d: [u8; 4] = output[idx..idx + 4].try_into().unwrap();
                i32x8::from_array(t, [d[0] as i32, d[1] as i32, d[2] as i32, d[3] as i32, 0, 0, 0, 0])
            };
            // (dest + trans) wraps i32 like the scalar wrapping_add, then clamps
            // to [0, 255] — `as u8` is exact after the clamp.
            let s = (dv + src).clamp(zero, pix_hi).to_array();
            for (j, &sv) in s.iter().take(active).enumerate() {
                output[idx + j] = sv as u8;
            }
        }
        c += active;
    }
}

#[cfg(test)]
mod tests {
    //! SIMD-vs-scalar differential for the lane kernels (Gate-3 parity rule:
    //! integer SIMD MUST be bit-identical to the scalar port) — per the
    //! STATUS.md differential plan: inputs sweep the driver clamp bounds
    //! ±2^(bd+7) for bd 8/10/12 (dense random + the exact boundary values +
    //! sign patterns engineered to maximize |p0 + p1| in half_btf), PLUS
    //! full-range i32 (the lane ops are exact on the whole domain, so the
    //! test asserts the whole domain), × cos_bit 10..=13 × the stage_range
    //! values the drivers pass (16/18/20 per `opt_range`, + the 1-D
    //! harness's 17). Every case runs at every token permutation; a counter
    //! proves the v3 arm actually ran (non-vacuous even under
    //! AOM_FORCE_SCALAR — the permutation harness owns token state).

    use super::*;
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    type ScalarKernel = fn(&[i32], &mut [i32], i32, &[i8]);

    /// One direction-erased kernel id for the test table.
    #[derive(Clone, Copy)]
    enum K {
        I(Inv1d),
        F(Fwd1d),
    }

    struct Case {
        name: &'static str,
        size: usize,
        scalar: ScalarKernel,
        simd: K,
    }

    fn cases() -> Vec<Case> {
        use K::{F, I};
        vec![
            Case { name: "idct4", size: 4, scalar: crate::transform::av1_idct4, simd: I(Inv1d::Dct4) },
            Case { name: "idct8", size: 8, scalar: crate::transform::av1_idct8, simd: I(Inv1d::Dct8) },
            Case { name: "idct16", size: 16, scalar: crate::transform::av1_idct16, simd: I(Inv1d::Dct16) },
            Case { name: "idct32", size: 32, scalar: crate::transform::av1_idct32, simd: I(Inv1d::Dct32) },
            Case { name: "idct64", size: 64, scalar: crate::transform::av1_idct64, simd: I(Inv1d::Dct64) },
            Case { name: "iadst4", size: 4, scalar: crate::transform::av1_iadst4, simd: I(Inv1d::Adst4) },
            Case { name: "iadst8", size: 8, scalar: crate::transform::av1_iadst8, simd: I(Inv1d::Adst8) },
            Case { name: "iadst16", size: 16, scalar: crate::transform::av1_iadst16, simd: I(Inv1d::Adst16) },
            Case { name: "iidentity4", size: 4, scalar: crate::transform::av1_iidentity4, simd: I(Inv1d::Idtx4) },
            Case { name: "iidentity8", size: 8, scalar: crate::transform::av1_iidentity8, simd: I(Inv1d::Idtx8) },
            Case {
                name: "iidentity16",
                size: 16,
                scalar: crate::transform::av1_iidentity16,
                simd: I(Inv1d::Idtx16),
            },
            Case {
                name: "iidentity32",
                size: 32,
                scalar: crate::transform::av1_iidentity32,
                simd: I(Inv1d::Idtx32),
            },
            Case { name: "fdct4", size: 4, scalar: crate::transform::av1_fdct4, simd: F(Fwd1d::Dct4) },
            Case { name: "fdct8", size: 8, scalar: crate::transform::av1_fdct8, simd: F(Fwd1d::Dct8) },
            Case { name: "fdct16", size: 16, scalar: crate::transform::av1_fdct16, simd: F(Fwd1d::Dct16) },
            Case { name: "fdct32", size: 32, scalar: crate::transform::av1_fdct32, simd: F(Fwd1d::Dct32) },
            Case { name: "fdct64", size: 64, scalar: crate::transform::av1_fdct64, simd: F(Fwd1d::Dct64) },
            Case { name: "fadst4", size: 4, scalar: crate::transform::av1_fadst4, simd: F(Fwd1d::Adst4) },
            Case { name: "fadst8", size: 8, scalar: crate::transform::av1_fadst8, simd: F(Fwd1d::Adst8) },
            Case { name: "fadst16", size: 16, scalar: crate::transform::av1_fadst16, simd: F(Fwd1d::Adst16) },
            Case { name: "fidentity4", size: 4, scalar: crate::transform::av1_fidentity4, simd: F(Fwd1d::Idtx4) },
            Case { name: "fidentity8", size: 8, scalar: crate::transform::av1_fidentity8, simd: F(Fwd1d::Idtx8) },
            Case {
                name: "fidentity16",
                size: 16,
                scalar: crate::transform::av1_fidentity16,
                simd: F(Fwd1d::Idtx16),
            },
            Case {
                name: "fidentity32",
                size: 32,
                scalar: crate::transform::av1_fidentity32,
                simd: F(Fwd1d::Idtx32),
            },
        ]
    }

    /// Run one lane batch through the selected kernel at the CURRENT tier
    /// (a `#[magetypes]` body, because the kernels are `#[target_feature]`
    /// fns and cannot be stored as plain fn pointers).
    #[magetypes(define(i32x8), v3, neon, -scalar)]
    fn run_kernel(
        t: Token,
        k: K,
        input: &[I32x8<Token>],
        out: &mut [I32x8<Token>],
        cos_bit: i32,
        stage_range: &[i8],
    ) {
        match k {
            K::I(k) => incant!(run_inv1d(k, input, out, cos_bit, stage_range), [v3, neon]),
            K::F(k) => incant!(run_fwd1d(k, input, out, cos_bit, stage_range), [v3, neon]),
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform in [-(1<<bits), 1<<bits] (the driver clamp domains).
        fn bounded(&mut self, bits: u32) -> i32 {
            let range = (1i64 << (bits + 1)) + 1;
            ((self.next() as i64).rem_euclid(range) - (1i64 << bits)) as i32
        }
    }

    /// Run one 8-column batch through the vector kernel and the scalar kernel
    /// per column; assert every lane matches. A tier body, so the whole
    /// comparison runs at whatever tier the enclosing permutation selected.
    #[magetypes(define(i32x8), v3, neon, -scalar)]
    fn assert_batch(
        t: Token,
        case: &Case,
        cols: &[[i32; 8]], // cols[r][lane] — row-major lane batch
        cos_bit: i32,
        stage_range: &[i8],
        label: &str,
    ) {
        let n = case.size;
        let mut vin = vec![i32x8::zero(t); n];
        for (r, c) in cols.iter().enumerate() {
            vin[r] = i32x8::from_array(t, *c);
        }
        let mut vout = vec![i32x8::zero(t); n];
        incant!(run_kernel(case.simd, &vin, &mut vout, cos_bit, stage_range), [v3, neon]);

        let mut sin = vec![0i32; n];
        let mut sout = vec![0i32; n];
        for lane in 0..8 {
            for r in 0..n {
                sin[r] = cols[r][lane];
            }
            (case.scalar)(&sin, &mut sout, cos_bit, stage_range);
            for r in 0..n {
                assert_eq!(
                    vout[r].to_array()[lane],
                    sout[r],
                    "{}: {label} lane={lane} row={r} cos_bit={cos_bit} sr={} input={sin:?}",
                    case.name,
                    stage_range[0],
                );
            }
        }
    }

    #[test]
    fn inv1d_simd_bit_identical_to_scalar_at_every_tier() {
        // Fire the AOM_FORCE_SCALAR pin (if set) BEFORE the permutation
        // harness — the harness then owns token state, so the v3 arm runs
        // in its enabled permutations in BOTH dispatch modes.
        let _ = crate::dispatch::scalar_forced();
        let mut simd_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            // `incant!` picks the live tier for THIS permutation; the scalar
            // twin returns false, which is how a scalar-only permutation is
            // counted rather than silently skipped. Gating on a named token
            // instead (the pre-2026-07-25 shape) made every aarch64
            // permutation look scalar, because `X64V3Token` is a stub there.
            if incant!(sweep_all_cases(&tier.label), [v3, neon, scalar]) {
                simd_ran += 1;
            }
        });
        eprintln!("inv1d simd parity: {report}, vector permutations run: {simd_ran}");
        assert!(
            simd_ran >= 1,
            "a vector tier must run at least once (AVX2 on x86-64 CI, NEON on aarch64); \
             on aarch64 that needs `archmage/testable_dispatch` in dev-dependencies"
        );
        assert!(report.permutations_run >= 2);
    }

    /// Scalar-only permutation: there is no vector kernel to compare against.
    fn sweep_all_cases_scalar(_: ScalarToken, _tier: &str) -> bool {
        false
    }

    /// The whole case sweep at ONE tier. Returns true (a vector tier ran).
    #[magetypes(define(i32x8), v3, neon, -scalar)]
    fn sweep_all_cases(t: Token, tier: &str) -> bool {
            let mut rng = Rng(0x_7ab5_11fe_c0de_0001);
            // Driver stage_range values (opt_range 16/18/20) + the 1-D
            // harness's 17; the drivers pass INV_COS_BIT=12, sweep 10..=13.
            for &sr in &[16i8, 17, 18, 20] {
                let stage_range = [sr; 12];
                for cos_bit in 10..=13 {
                    for case in cases() {
                        let n = case.size;
                        for &bits in &[15u32, 17, 19] {
                            // (a) driver-clamp-domain dense random: the col
                            // pass clamps input to max(bd+6,16) bits, the
                            // row pass to bd+8 — sweep ±2^15/2^17/2^19.
                            for rep in 0..24 {
                                let cols: Vec<[i32; 8]> = (0..n)
                                    .map(|_| core::array::from_fn(|_| rng.bounded(bits)))
                                    .collect();
                                incant!(assert_batch(
                                    &case,
                                    &cols,
                                    cos_bit,
                                    &stage_range,
                                    &format!("[{tier}] rand b{bits} rep{rep}"),
                                ), [v3, neon]);
                            }
                            // (b) exact clamp-bound sign patterns — the
                            // half_btf |p0 + p1| maximizers: all +B, all -B,
                            // alternating ±B (both phases), random-ish ±B.
                            let b = 1i32 << bits;
                            let pats: [&dyn Fn(usize, usize) -> i32; 5] = [
                                &|_, _| b,
                                &|_, _| -b,
                                &|r, l| if (r + l) % 2 == 0 { b } else { -b },
                                &|r, l| if (r + l) % 2 == 0 { -b } else { b },
                                &|r, l| if (r * 7 + l * 3) % 5 < 2 { b } else { -b },
                            ];
                            for (pi, pat) in pats.iter().enumerate() {
                                let cols: Vec<[i32; 8]> =
                                    (0..n).map(|r| core::array::from_fn(|l| pat(r, l))).collect();
                                incant!(assert_batch(
                                    &case,
                                    &cols,
                                    cos_bit,
                                    &stage_range,
                                    &format!("[{tier}] bound b{bits} pat{pi}"),
                                ), [v3, neon]);
                            }
                        }
                        // (c) FULL-i32 random (the lane ops are exact on the
                        // whole domain — assert it there) + extreme lanes
                        // mixed with all-zero columns.
                        for rep in 0..24 {
                            let cols: Vec<[i32; 8]> = (0..n)
                                .map(|_| core::array::from_fn(|_| rng.next() as i32))
                                .collect();
                            incant!(assert_batch(
                                &case,
                                &cols,
                                cos_bit,
                                &stage_range,
                                &format!("[{tier}] full-i32 rep{rep}"),
                            ), [v3, neon]);
                        }
                        let mut cols = vec![[0i32; 8]; n];
                        cols[0] = [
                            i32::MIN,
                            i32::MAX,
                            0,
                            -1,
                            1 << 19,
                            -(1 << 19),
                            i32::MIN + 1,
                            i32::MAX - 1,
                        ];
                        cols[n - 1] = [i32::MAX, i32::MIN, 1, 0, -(1 << 19), 1 << 19, -2, 2];
                        incant!(assert_batch(
                            &case,
                            &cols,
                            cos_bit,
                            &stage_range,
                            &format!("[{tier}] extremes+zero-cols"),
                        ), [v3, neon]);
                    }
                }
            }
        true
    }
}

