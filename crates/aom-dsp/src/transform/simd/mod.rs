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
    if col_n % 8 != 0 {
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
    debug_assert!(row_n <= 64 && col_n % 8 == 0);
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
    for cg in (0..col_n).step_by(8) {
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let src_r = if ud_flip { row_n - r - 1 } else { r };
            let mut v = widen16(t, &input[src_r * stride + cg..src_r * stride + cg + 8]);
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
            if lr_flip {
                let base = r * col_n + (col_n - cg - 8);
                revv(t, v).store((&mut buf[base..base + 8]).try_into().unwrap());
            } else {
                let base = r * col_n + cg;
                v.store((&mut buf[base..base + 8]).try_into().unwrap());
            }
        }
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
