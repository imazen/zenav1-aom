//! SIMD (Gate 3) for the transform stack — lane-batched 1-D kernels + the
//! 2-D drivers' vector passes, bit-identical to the scalar port per lane.
//! x86-64 only (the module is cfg'd out elsewhere; NEON falls to scalar).
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
//!   reproduces the i64 sum exactly: `vpmulld` (wrapped products, ==
//!   scalar), widen each 128-half via `vpmovsxdq` to 2×i64x4, `vpaddq` sums
//!   (|p0|,|p1| <= 2^31, rnd <= 2^31 → no i64 overflow), then the
//!   arithmetic-shift + truncate pair via LOGICAL `vpsrlq` + low-dword
//!   gather — exact because `((v >>_arith b) as i32) == low32(v >>_logical
//!   b)` for any v when `1 <= b <= 32` (the differing sign-fill bits all
//!   land at positions >= 32 and are truncated away; cos_bit is 10..=13).
//!   AVX2 has no `vpsraq`; the logical+truncate trick dodges it.
//! * `round_shift(v as i64, bit)` (the positive-bit `round_shift_array`
//!   arm): the same widen → add rounding → logical shift → truncate recipe.
//!   [`rshiftv`]
//! * `highbd_clip_pixel_add`: the i32 lane add wraps like the scalar
//!   `wrapping_add`; clamp to `[0, (1<<bd)-1]` is lane min/max; the `as u16`
//!   narrowing is exact after the clamp.
//! * `lr_flip` lane reversal and `ud_flip` row reversal are pure index
//!   permutations ([`revv`] / loop order), identical to the scalar loops.
//!
//! magetypes has NO integer widening ops, so [`hb`]/[`rshiftv`] use the raw
//! `__m256i` escape (`i32x8::raw()`/`from_m256i`) with VALUE intrinsics —
//! safe inside `#[rite]`/`#[arcane]` `#[target_feature]` regions, keeping
//! `#![forbid(unsafe_code)]`.

mod inv1d_v3_gen;

use archmage::SimdToken;
use archmage::X64V3Token;
use archmage::prelude::*;
use magetypes::simd::i32x8;

use inv1d_v3_gen::{av1_idct4_v3, av1_idct8_v3, av1_idct16_v3, av1_idct32_v3, av1_idct64_v3};

/// `half_btf` on 8 lanes — the exact-i64 recipe (see the module docs).
/// Bit-identical to [`crate::fdct::half_btf`] per lane for ANY i32 lanes and
/// any `cos_bit` in `1..=32` (the transforms use 10..=13).
#[rite]
pub(crate) fn hb(t: X64V3Token, w0: i32, in0: i32x8, w1: i32, in1: i32x8, cos_bit: i32) -> i32x8 {
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
    // (sum >>_arith bit) as i32 == low32(sum >>_logical bit) for bit <= 32.
    let cnt = _mm_cvtsi32_si128(cos_bit);
    i32x8::from_m256i(t, low32_of_i64(_mm256_srl_epi64(lo, cnt), _mm256_srl_epi64(hi, cnt)))
}

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

/// `clamp_value(v, bit)` on lanes — identical to the scalar port for any i32
/// lanes and any `bit`: `<= 0` and `>= 32` are identities (the scalar i64
/// bounds cover all of i32 there), else lane min/max on the i32 bounds.
#[rite]
pub(crate) fn clampv(t: X64V3Token, v: i32x8, bit: i8) -> i32x8 {
    if bit <= 0 || bit >= 32 {
        return v;
    }
    let hi = ((1i64 << (bit - 1)) - 1) as i32;
    let lo = (-(1i64 << (bit - 1))) as i32;
    v.clamp(i32x8::splat(t, lo), i32x8::splat(t, hi))
}

/// `wrapping_neg` on lanes (`0 - v` wraps identically).
#[rite]
#[allow(dead_code)] // used by the iadst/fdct lane kernels (next chunks)
pub(crate) fn negv(t: X64V3Token, v: i32x8) -> i32x8 {
    i32x8::zero(t) - v
}

/// `round_shift(v as i64, bit)` on lanes for `bit` in `1..=32` — widen, add
/// rounding, logical shift, truncate (the same identity as [`hb`]).
#[rite]
fn rshiftv(t: X64V3Token, v: i32x8, bit: i32) -> i32x8 {
    use core::arch::x86_64::*;
    debug_assert!((1..=32).contains(&bit));
    let rnd = _mm256_set1_epi64x(1i64 << (bit - 1));
    let lo = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(v.raw())), rnd);
    let hi = _mm256_add_epi64(
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(v.raw())),
        rnd,
    );
    let cnt = _mm_cvtsi32_si128(bit);
    i32x8::from_m256i(t, low32_of_i64(_mm256_srl_epi64(lo, cnt), _mm256_srl_epi64(hi, cnt)))
}

/// Reverse the 8 lanes (for `lr_flip` column groups).
#[rite]
fn revv(t: X64V3Token, v: i32x8) -> i32x8 {
    use core::arch::x86_64::*;
    let idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    i32x8::from_m256i(t, _mm256_permutevar8x32_epi32(v.raw(), idx))
}

/// Inverse column kernels ported so far (TXFM_TYPE 0..=4 = DCT4..DCT64).
/// Dispatch is per `func_col` — unported kernels return `None` and the
/// driver keeps its scalar per-column loop.
#[derive(Clone, Copy)]
enum InvColKernel {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
}

fn inv_col_kernel(txfm_type_col: i32) -> Option<InvColKernel> {
    match txfm_type_col {
        0 => Some(InvColKernel::Dct4),
        1 => Some(InvColKernel::Dct8),
        2 => Some(InvColKernel::Dct16),
        3 => Some(InvColKernel::Dct32),
        4 => Some(InvColKernel::Dct64),
        _ => None,
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
    if col_n % 8 != 0 {
        return false;
    }
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    let Some(t) = X64V3Token::summon() else {
        return false;
    };
    let Some(kernel) = inv_col_kernel(txfm_type_col) else {
        return false;
    };
    inv_col_pass(
        t,
        kernel,
        buf,
        output,
        stride,
        col_n,
        row_n,
        shift1_bit,
        col_clamp,
        stage_range,
        ud_flip,
        lr_flip,
        bd,
    );
    true
}

/// The lane-batched column pass body — the scalar per-column loop of
/// `av1_inv_txfm2d_add`, 8 columns per iteration (module docs carry the
/// per-stage exactness argument).
#[arcane]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass(
    t: X64V3Token,
    kernel: InvColKernel,
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
) {
    debug_assert!(row_n <= 64 && col_n % 8 == 0);
    let mut tin = [i32x8::zero(t); 64];
    let mut tout = [i32x8::zero(t); 64];
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, (1i32 << bd) - 1);
    for c in (0..col_n).step_by(8) {
        // Gather 8 columns: under lr_flip, scalar output column `c+j` reads
        // buf column `col_n-1-(c+j)` — i.e. the ascending 8-column load at
        // `col_n-c-8`, lanes reversed.
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let v = if lr_flip {
                let base = r * col_n + (col_n - c - 8);
                revv(t, i32x8::from_slice(t, &buf[base..base + 8]))
            } else {
                let base = r * col_n + c;
                i32x8::from_slice(t, &buf[base..base + 8])
            };
            *ti = clampv(t, v, col_clamp); // the driver's clamp_buf
        }
        let cos_bit = crate::inv_txfm2d::INV_COS_BIT;
        match kernel {
            InvColKernel::Dct4 => av1_idct4_v3(t, &tin[..4], &mut tout[..4], cos_bit, stage_range),
            InvColKernel::Dct8 => av1_idct8_v3(t, &tin[..8], &mut tout[..8], cos_bit, stage_range),
            InvColKernel::Dct16 => {
                av1_idct16_v3(t, &tin[..16], &mut tout[..16], cos_bit, stage_range)
            }
            InvColKernel::Dct32 => {
                av1_idct32_v3(t, &tin[..32], &mut tout[..32], cos_bit, stage_range)
            }
            InvColKernel::Dct64 => {
                av1_idct64_v3(t, &tin[..64], &mut tout[..64], cos_bit, stage_range)
            }
        }
        // round_shift_array(to, -shift[1]) — shift[1] is always negative for
        // the inverse sizes, so this is the positive-bit arm.
        for to in tout[..row_n].iter_mut() {
            *to = rshiftv(t, *to, shift1_bit);
        }
        // Reconstruction: output row r takes tout[row_n-1-r] under ud_flip.
        for r in 0..row_n {
            let src = tout[if ud_flip { row_n - r - 1 } else { r }];
            let idx = r * stride + c;
            let d: [u16; 8] = output[idx..idx + 8].try_into().unwrap();
            let dv = i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32));
            // (dest + trans) wraps i32 like the scalar wrapping_add, then
            // clamps to the pixel range — `as u16` is exact after the clamp.
            let s = (dv + src).clamp(zero, pix_hi).to_array();
            for (j, &sv) in s.iter().enumerate() {
                output[idx + j] = sv as u16;
            }
        }
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

    struct Case {
        name: &'static str,
        size: usize,
        scalar: ScalarKernel,
        v3: InvColKernel,
    }

    fn cases() -> Vec<Case> {
        vec![
            Case { name: "idct4", size: 4, scalar: crate::av1_idct4, v3: InvColKernel::Dct4 },
            Case { name: "idct8", size: 8, scalar: crate::av1_idct8, v3: InvColKernel::Dct8 },
            Case { name: "idct16", size: 16, scalar: crate::av1_idct16, v3: InvColKernel::Dct16 },
            Case { name: "idct32", size: 32, scalar: crate::av1_idct32, v3: InvColKernel::Dct32 },
            Case { name: "idct64", size: 64, scalar: crate::av1_idct64, v3: InvColKernel::Dct64 },
        ]
    }

    /// Run one lane batch through the selected v3 kernel (the test-side
    /// arcane entry — kernels are `#[target_feature]` fns and cannot be
    /// stored as plain fn pointers).
    #[arcane]
    fn run_v3(
        t: X64V3Token,
        k: InvColKernel,
        input: &[i32x8],
        out: &mut [i32x8],
        cos_bit: i32,
        stage_range: &[i8],
    ) {
        match k {
            InvColKernel::Dct4 => av1_idct4_v3(t, input, out, cos_bit, stage_range),
            InvColKernel::Dct8 => av1_idct8_v3(t, input, out, cos_bit, stage_range),
            InvColKernel::Dct16 => av1_idct16_v3(t, input, out, cos_bit, stage_range),
            InvColKernel::Dct32 => av1_idct32_v3(t, input, out, cos_bit, stage_range),
            InvColKernel::Dct64 => av1_idct64_v3(t, input, out, cos_bit, stage_range),
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

    /// Run one 8-column batch through the v3 kernel and the scalar kernel
    /// per column; assert every lane matches.
    fn assert_batch(
        t: X64V3Token,
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
        run_v3(t, case.v3, &vin, &mut vout, cos_bit, stage_range);

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
    fn inv1d_v3_bit_identical_to_scalar_at_every_tier() {
        // Fire the AOM_FORCE_SCALAR pin (if set) BEFORE the permutation
        // harness — the harness then owns token state, so the v3 arm runs
        // in its enabled permutations in BOTH dispatch modes.
        let _ = aom_dispatch::scalar_forced();
        let mut v3_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            let Some(t) = X64V3Token::summon() else {
                return; // scalar-only permutation: nothing to compare
            };
            v3_ran += 1;
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
                                assert_batch(
                                    t,
                                    &case,
                                    &cols,
                                    cos_bit,
                                    &stage_range,
                                    &format!("[{tier}] rand b{bits} rep{rep}"),
                                );
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
                                assert_batch(
                                    t,
                                    &case,
                                    &cols,
                                    cos_bit,
                                    &stage_range,
                                    &format!("[{tier}] bound b{bits} pat{pi}"),
                                );
                            }
                        }
                        // (c) FULL-i32 random (the lane ops are exact on the
                        // whole domain — assert it there) + extreme lanes
                        // mixed with all-zero columns.
                        for rep in 0..24 {
                            let cols: Vec<[i32; 8]> = (0..n)
                                .map(|_| core::array::from_fn(|_| rng.next() as i32))
                                .collect();
                            assert_batch(
                                t,
                                &case,
                                &cols,
                                cos_bit,
                                &stage_range,
                                &format!("[{tier}] full-i32 rep{rep}"),
                            );
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
                        assert_batch(
                            t,
                            &case,
                            &cols,
                            cos_bit,
                            &stage_range,
                            &format!("[{tier}] extremes+zero-cols"),
                        );
                    }
                }
            }
        });
        eprintln!("inv1d v3 parity: {report}, v3 permutations run: {v3_ran}");
        assert!(v3_ran >= 1, "the v3 arm must run at least once (AVX2 CI)");
        assert!(report.permutations_run >= 2);
    }
}
