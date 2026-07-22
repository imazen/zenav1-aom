//! bd8 i16-lane inverse-transform COLUMN pass (Phase C of the lowbd pipeline):
//! 16 columns per AVX2 vector — 2x the lane throughput of the i32x8 pass —
//! for the audited i16-safe column kernels (idct4/8/16/32/64).
//!
//! # Exactness contract (why this is byte-identical to the scalar port)
//!
//! At bd8 the u8 column pass has fixed constants: `col_clamp == 16`,
//! `stage_range == [16; 12]`, `cos_bit == INV_COS_BIT == 12`, and the final
//! `round_shift` bit is `-shift[1] == 4` for every tx_size. The driver clamps
//! every gathered column value with `clamp_value(_, 16)` BEFORE the kernel, so
//! kernel inputs are exactly the i16 domain. Inside the audited DCT kernels
//! (`xtask/audit_i16_safety.py` over the generated scalar transcriptions):
//!
//! * every `half_btf` input is an input copy or a `clamp_value` output (i16);
//! * every `clamp_value` operand is i16 or a SINGLE unclamped `half_btf`
//!   output (|v| <= (2*4095*2^15 + 2^11)/2^12 < 2^17 — a 17-bit transient);
//! * every terminal output is a `clamp_value` output (i16).
//!
//! The two-domain representation mirrors that exactly: i16 values live in
//! `i16x16` lanes; the 17-bit butterfly transients live as exact i32 pairs in
//! unpack order ([`P32`]). Per-op proofs are on each helper below. The iadst
//! and identity kernels are NOT i16-safe (unclamped terminal negations /
//! multiplies exceed i16) and stay on the [`super`] i32x8 pass.
//!
//! x86_64/AVX2 only, like the rest of [`super`]; `AOM_FORCE_SCALAR` and
//! missing-token fall back exactly as before (the caller gates entry).

use archmage::X64V3Token;
use archmage::prelude::*;
use core::arch::x86_64::__m256i;
use magetypes::simd::{i16x16, i32x8, u8x16};

use super::inv1d_v3_i16_gen::{
    av1_idct4_v3_i16, av1_idct8_v3_i16, av1_idct16_v3_i16, av1_idct32_v3_i16, av1_idct64_v3_i16,
};

/// A pair of interleaved-i16 vectors, the shared input of the two `madd`
/// butterflies that consume the same (x, y) operands. `lo` holds source lanes
/// 0-3 and 8-11 as (x, y) i16 pairs, `hi` lanes 4-7 and 12-15.
#[derive(Clone, Copy)]
pub(crate) struct Upk {
    lo: __m256i,
    hi: __m256i,
}

/// A 16-lane i32 value in UNPACK ORDER: `lo` = source lanes 0-3 (low 128) and
/// 8-11 (high 128), `hi` = lanes 4-7 and 12-15. [`pack16`]'s per-128-lane
/// `packs_epi32` maps this back to natural i16x16 lane order exactly.
#[derive(Clone, Copy)]
pub(crate) struct P32 {
    lo: i32x8,
    hi: i32x8,
}

/// Interleave two i16x16 values into (x, y) pairs for `madd` butterflies.
#[rite]
pub(crate) fn unpk16(t: X64V3Token, x: i16x16, y: i16x16) -> Upk {
    use core::arch::x86_64::*;
    let _ = t;
    Upk {
        lo: _mm256_unpacklo_epi16(x.raw(), y.raw()),
        hi: _mm256_unpackhi_epi16(x.raw(), y.raw()),
    }
}

/// `half_btf(w0, x, w1, y, 12)` on 16 lanes, x/y in the i16 domain — EXACT:
/// `madd` computes `w0*x + w1*y` per i32 slot with |w| <= 4095 and |x|,|y| <=
/// 2^15, so |each product| <= 2^27 (no i32 wrap — identical to the scalar
/// port's `wrapping_mul` which also cannot wrap here) and |pair sum| <= 2^28
/// (madd's internal i32 pair-add is exact; its only saturation case needs both
/// products == -2^31, impossible with |w| <= 4095). Adding the rounding
/// constant 2^11 cannot overflow, and the arithmetic shift by 12 equals the
/// scalar's i64 shift because the value fits i32. Output <= ~2^16.03: a 17-bit
/// transient, kept as exact i32 pairs.
#[rite]
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
#[rite]
pub(crate) fn pack16(t: X64V3Token, p: P32) -> i16x16 {
    use core::arch::x86_64::*;
    i16x16::from_m256i(t, _mm256_packs_epi32(p.lo.raw(), p.hi.raw()))
}

/// Sign-extend an i16x16 into an unpack-order i32 pair (for clamp adds that
/// mix an i16 operand with a 17-bit transient): interleaving the value with
/// its `v < 0` mask (all-ones == 0xffff) builds the exact sign-extended i32
/// in each slot.
#[rite]
pub(crate) fn ext16(t: X64V3Token, v: i16x16) -> P32 {
    use core::arch::x86_64::*;
    let sign = _mm256_cmpgt_epi16(_mm256_setzero_si256(), v.raw());
    P32 {
        lo: i32x8::from_m256i(t, _mm256_unpacklo_epi16(v.raw(), sign)),
        hi: i32x8::from_m256i(t, _mm256_unpackhi_epi16(v.raw(), sign)),
    }
}

/// `clamp_value(a + b, 16)` for two i16-domain values: the saturating i16 add
/// IS the normative clamp (a+b in [-2^16, 2^16-2] saturates to exactly
/// `clamp_value`'s [-2^15, 2^15-1]).
#[rite]
pub(crate) fn sadd16(t: X64V3Token, a: i16x16, b: i16x16) -> i16x16 {
    use core::arch::x86_64::*;
    i16x16::from_m256i(t, _mm256_adds_epi16(a.raw(), b.raw()))
}

/// `clamp_value(a - b, 16)` for two i16-domain values (also serves
/// `clamp_value(-b + a, 16)` — identical in two's complement).
#[rite]
pub(crate) fn ssub16(t: X64V3Token, a: i16x16, b: i16x16) -> i16x16 {
    use core::arch::x86_64::*;
    i16x16::from_m256i(t, _mm256_subs_epi16(a.raw(), b.raw()))
}

/// i32-pair add (exact: operands are <= 17-bit transients / sign-extended i16,
/// so the wrapping lane add cannot wrap — identical to the scalar i32 add).
#[rite]
pub(crate) fn padd32(t: X64V3Token, a: P32, b: P32) -> P32 {
    let _ = t;
    P32 {
        lo: a.lo + b.lo,
        hi: a.hi + b.hi,
    }
}

/// i32-pair subtract (exact, same bound argument as [`padd32`]).
#[rite]
pub(crate) fn psub32(t: X64V3Token, a: P32, b: P32) -> P32 {
    let _ = t;
    P32 {
        lo: a.lo - b.lo,
        hi: a.hi - b.hi,
    }
}

/// `round_shift(v, 4)` on i16 lanes via `mulhrs(v, 2^11)` — EXACT for every
/// i16 `v`: mulhrs computes `((v * 2048) >> 14 + 1) >> 1` in full internal
/// precision; `(v * 2048) >> 14 == v >> 3` (arithmetic, exact), and
/// `((v >> 3) + 1) >> 1 == (v + 8) >> 4` for ALL integers (write `v = 8a + b`,
/// `b in [0, 8)`: both sides equal `floor((a + 1) / 2)` — for odd `a + 1` the
/// `b/16 <= 7/16` fraction can never carry past the half).
#[rite]
fn rshift4_16(t: X64V3Token, v: i16x16) -> i16x16 {
    use core::arch::x86_64::*;
    i16x16::from_m256i(t, _mm256_mulhrs_epi16(v.raw(), _mm256_set1_epi16(1 << 11)))
}

/// Gather+clamp: two natural-order i32x8 loads -> `clamp_value(_, 16)` per
/// lane via the saturating pack, permuted back to natural lane order
/// (packs_epi32 of natural-order inputs interleaves 128-lane quarters;
/// `permute4x64(0xD8)` = [q0, q2, q1, q3] restores a0-7, b0-7).
#[rite]
fn pack_clamp16(t: X64V3Token, a: i32x8, b: i32x8) -> i16x16 {
    use core::arch::x86_64::*;
    let p = _mm256_packs_epi32(a.raw(), b.raw());
    i16x16::from_m256i(t, _mm256_permute4x64_epi64::<0b1101_1000>(p))
}

/// Reverse all 16 lanes (lr_flip on a full column group).
#[rite]
fn rev16(t: X64V3Token, v: i16x16) -> i16x16 {
    use core::arch::x86_64::*;
    let m = _mm256_setr_epi8(
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, //
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
    );
    let r = _mm256_shuffle_epi8(v.raw(), m);
    i16x16::from_m256i(t, _mm256_permute4x64_epi64::<0b0100_1110>(r))
}

/// The audited i16-safe column kernels (DCT family only).
#[derive(Clone, Copy)]
pub(crate) enum Inv1dI16 {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
}

/// TXFM_TYPE id -> i16 kernel, for the ids whose kernel passed the audit.
pub(crate) fn inv_kernel_i16(txfm_type: i32) -> Option<Inv1dI16> {
    match txfm_type {
        0 => Some(Inv1dI16::Dct4),
        1 => Some(Inv1dI16::Dct8),
        2 => Some(Inv1dI16::Dct16),
        3 => Some(Inv1dI16::Dct32),
        4 => Some(Inv1dI16::Dct64),
        _ => None, // iadst4/8/16 + identity4/8/16/32: NOT i16-safe, i32 path
    }
}

pub(crate) fn inv_kernel_i16_n(k: Inv1dI16) -> usize {
    match k {
        Inv1dI16::Dct4 => 4,
        Inv1dI16::Dct8 => 8,
        Inv1dI16::Dct16 => 16,
        Inv1dI16::Dct32 => 32,
        Inv1dI16::Dct64 => 64,
    }
}

/// Direct-dispatch the i16 column kernel (rite→rite inlining, same shape as
/// [`super`]'s `run_inv1d`).
#[rite]
pub(crate) fn run_inv1d_i16(t: X64V3Token, k: Inv1dI16, input: &[i16x16], out: &mut [i16x16]) {
    match k {
        Inv1dI16::Dct4 => av1_idct4_v3_i16(t, input, out),
        Inv1dI16::Dct8 => av1_idct8_v3_i16(t, input, out),
        Inv1dI16::Dct16 => av1_idct16_v3_i16(t, input, out),
        Inv1dI16::Dct32 => av1_idct32_v3_i16(t, input, out),
        Inv1dI16::Dct64 => av1_idct64_v3_i16(t, input, out),
    }
}

/// i16-lane u8 column pass. Preconditions (asserted by the caller in
/// [`super::try_inv_col_pass_u8`]): bd8 constants — `col_clamp == 16`,
/// round-shift bit 4, `cos_bit == 12` — and an audited DCT column kernel.
#[arcane]
#[allow(clippy::too_many_arguments)]
pub(crate) fn inv_col_pass_u8_i16(
    t: X64V3Token,
    kernel: Inv1dI16,
    buf: &[i32],
    output: &mut [u8],
    stride: usize,
    col_n: usize,
    row_n: usize,
    ud_flip: bool,
    lr_flip: bool,
) {
    debug_assert!(row_n <= 64 && (col_n % 16 == 0 || col_n == 4 || col_n == 8));
    if row_n <= 8 {
        let mut tin = [i16x16::zero(t); 8];
        let mut tout = [i16x16::zero(t); 8];
        inv_col_pass_u8_i16_core(
            t, kernel, buf, output, stride, col_n, row_n, ud_flip, lr_flip, &mut tin, &mut tout,
        );
    } else if row_n <= 16 {
        let mut tin = [i16x16::zero(t); 16];
        let mut tout = [i16x16::zero(t); 16];
        inv_col_pass_u8_i16_core(
            t, kernel, buf, output, stride, col_n, row_n, ud_flip, lr_flip, &mut tin, &mut tout,
        );
    } else {
        let mut tin = [i16x16::zero(t); 64];
        let mut tout = [i16x16::zero(t); 64];
        inv_col_pass_u8_i16_core(
            t, kernel, buf, output, stride, col_n, row_n, ud_flip, lr_flip, &mut tin, &mut tout,
        );
    }
}

#[rite]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass_u8_i16_core(
    t: X64V3Token,
    kernel: Inv1dI16,
    buf: &[i32],
    output: &mut [u8],
    stride: usize,
    col_n: usize,
    row_n: usize,
    ud_flip: bool,
    lr_flip: bool,
    tin: &mut [i16x16],
    tout: &mut [i16x16],
) {
    use core::arch::x86_64::*;
    let mut c = 0usize;
    while c < col_n {
        let active = (col_n - c).min(16);
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let base = r * col_n;
            *ti = if active == 16 {
                let src = if lr_flip { col_n - c - 16 } else { c };
                let a = i32x8::from_slice(t, &buf[base + src..base + src + 8]);
                let b = i32x8::from_slice(t, &buf[base + src + 8..base + src + 16]);
                let v = pack_clamp16(t, a, b);
                if lr_flip { rev16(t, v) } else { v }
            } else if active == 8 {
                // col_n == 8: the whole row is one half-group.
                let row: &[i32; 8] = buf[base..base + 8].try_into().unwrap();
                let a = if lr_flip {
                    i32x8::from_array(
                        t,
                        [
                            row[7], row[6], row[5], row[4], row[3], row[2], row[1], row[0],
                        ],
                    )
                } else {
                    i32x8::from_array(t, *row)
                };
                pack_clamp16(t, a, i32x8::zero(t))
            } else {
                // col_n == 4.
                let row: &[i32; 4] = buf[base..base + 4].try_into().unwrap();
                let a = if lr_flip {
                    i32x8::from_array(t, [row[3], row[2], row[1], row[0], 0, 0, 0, 0])
                } else {
                    i32x8::from_array(t, [row[0], row[1], row[2], row[3], 0, 0, 0, 0])
                };
                pack_clamp16(t, a, i32x8::zero(t))
            };
        }
        run_inv1d_i16(t, kernel, &tin[..row_n], &mut tout[..row_n]);
        for r in 0..row_n {
            let src = tout[if ud_flip { row_n - r - 1 } else { r }];
            // Terminal values are clamp_value outputs (i16); round_shift(_, 4)
            // then `clamp(dest + res, 0, 255)`, all exact in i16 lanes:
            // |res| <= 2048 and dest <= 255, so the lane add cannot wrap, and
            // packus_epi16 saturation IS the [0, 255] pixel clamp.
            let res = rshift4_16(t, src);
            let idx = r * stride + c;
            if active == 16 {
                let d: &[u8; 16] = (&output[idx..idx + 16]).try_into().unwrap();
                let d16 = _mm256_cvtepu8_epi16(u8x16::load(t, d).raw());
                let sum = _mm256_add_epi16(res.raw(), d16);
                let packed = _mm_packus_epi16(
                    _mm256_castsi256_si128(sum),
                    _mm256_extracti128_si256::<1>(sum),
                );
                let out16: &mut [u8; 16] = (&mut output[idx..idx + 16]).try_into().unwrap();
                u8x16::from_m128i(t, packed).store(out16);
            } else {
                let arr = res.to_array();
                for (j, &rv) in arr.iter().take(active).enumerate() {
                    let d = &mut output[idx + j];
                    *d = ((*d as i32) + (rv as i32)).clamp(0, 255) as u8;
                }
            }
        }
        c += active;
    }
}

#[cfg(test)]
mod tests {
    //! i16-lane-vs-scalar differential for the Phase C column kernels: over
    //! the kernels' FULL input contract domain (every lane an arbitrary i16 —
    //! exactly the `clamp_value(_, 16)` image the driver feeds them at bd8),
    //! the i16 kernel must be bit-identical to the scalar transcription with
    //! `stage_range == [16; 12]`, `cos_bit == 12`. Dense random + the exact
    //! saturation boundaries (`i16::MIN`/`i16::MAX` sign patterns are the
    //! half_btf |p0 + p1| and adds/subs saturation maximizers). Runs at every
    //! token permutation; non-vacuity asserted.

    use super::*;
    use crate::transform::{av1_idct4, av1_idct8, av1_idct16, av1_idct32, av1_idct64};
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    type ScalarKernel = fn(&[i32], &mut [i32], i32, &[i8]);

    fn cases() -> [(&'static str, usize, ScalarKernel, Inv1dI16); 5] {
        [
            ("idct4", 4, av1_idct4, Inv1dI16::Dct4),
            ("idct8", 8, av1_idct8, Inv1dI16::Dct8),
            ("idct16", 16, av1_idct16, Inv1dI16::Dct16),
            ("idct32", 32, av1_idct32, Inv1dI16::Dct32),
            ("idct64", 64, av1_idct64, Inv1dI16::Dct64),
        ]
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
        fn lane(&mut self) -> i16 {
            self.next() as i16
        }
    }

    /// Test-side arcane entry (the kernels are `#[target_feature]` fns).
    #[arcane]
    fn run_i16(t: X64V3Token, k: Inv1dI16, input: &[i16x16], out: &mut [i16x16]) {
        run_inv1d_i16(t, k, input, out);
    }

    fn assert_batch16(
        t: X64V3Token,
        name: &str,
        n: usize,
        scalar: ScalarKernel,
        kernel: Inv1dI16,
        cols: &[[i16; 16]],
        label: &str,
    ) {
        let mut vin = vec![i16x16::zero(t); n];
        for (r, c) in cols.iter().enumerate() {
            vin[r] = i16x16::from_array(t, *c);
        }
        let mut vout = vec![i16x16::zero(t); n];
        run_i16(t, kernel, &vin, &mut vout);

        let stage_range = [16i8; 12];
        let mut sin = vec![0i32; n];
        let mut sout = vec![0i32; n];
        for lane in 0..16 {
            for r in 0..n {
                sin[r] = cols[r][lane] as i32;
            }
            scalar(&sin, &mut sout, 12, &stage_range);
            for r in 0..n {
                assert_eq!(
                    vout[r].to_array()[lane] as i32,
                    sout[r],
                    "{name}: {label} lane={lane} row={r} input={sin:?}"
                );
            }
        }
    }

    #[test]
    fn inv1d_i16_bit_identical_to_scalar_at_every_tier() {
        let _ = crate::dispatch::scalar_forced(); // fire the pin before the harness
        let mut v3_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            let Some(t) = X64V3Token::summon() else {
                return;
            };
            v3_ran += 1;
            let mut rng = Rng(0x_bd08_1616_2026_0722);
            for (name, n, scalar, kernel) in cases() {
                // (a) dense random over the FULL i16 lane domain.
                for rep in 0..64 {
                    let cols: Vec<[i16; 16]> = (0..n)
                        .map(|_| core::array::from_fn(|_| rng.lane()))
                        .collect();
                    assert_batch16(
                        t,
                        name,
                        n,
                        scalar,
                        kernel,
                        &cols,
                        &format!("[{tier}] rand rep{rep}"),
                    );
                }
                // (b) exact saturation-boundary sign patterns.
                let (lo, hi) = (i16::MIN, i16::MAX);
                let pats: [&dyn Fn(usize, usize) -> i16; 6] = [
                    &|_, _| hi,
                    &|_, _| lo,
                    &|r, l| if (r + l) % 2 == 0 { hi } else { lo },
                    &|r, l| if (r + l) % 2 == 0 { lo } else { hi },
                    &|r, l| if (r * 7 + l * 3) % 5 < 2 { hi } else { lo },
                    &|r, l| if (r * 3 + l) % 3 == 0 { lo } else { hi },
                ];
                for (pi, pat) in pats.iter().enumerate() {
                    let cols: Vec<[i16; 16]> = (0..n)
                        .map(|r| core::array::from_fn(|l| pat(r, l)))
                        .collect();
                    assert_batch16(
                        t,
                        name,
                        n,
                        scalar,
                        kernel,
                        &cols,
                        &format!("[{tier}] bound pat{pi}"),
                    );
                }
                // (c) boundary lanes mixed with zero columns.
                let mut cols = vec![[0i16; 16]; n];
                cols[0] = core::array::from_fn(|l| [lo, hi, 0, -1, 1, lo + 1, hi - 1, 2][l % 8]);
                cols[n - 1] = core::array::from_fn(|l| [hi, lo, -2, 2, 0, -1, 1, lo + 1][l % 8]);
                assert_batch16(
                    t,
                    name,
                    n,
                    scalar,
                    kernel,
                    &cols,
                    &format!("[{tier}] extremes"),
                );
            }
        });
        eprintln!("inv1d i16 parity: {report}, v3 permutations run: {v3_ran}");
        assert!(v3_ran >= 1, "the v3 arm must run at least once (AVX2 CI)");
        assert!(report.permutations_run >= 2);
    }
}
