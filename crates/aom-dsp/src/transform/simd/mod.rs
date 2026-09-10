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
