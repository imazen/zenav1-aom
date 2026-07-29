//! bd8 i16-lane inverse-transform COLUMN pass (Phase C of the lowbd pipeline)
//! and ROW pass (the Phase-C follow-up lever): 16 columns/rows per vector —
//! 2x the lane throughput of the [`super`] i32x8 pass — for the audited
//! i16-safe kernels (idct4/8/16/32/64).
//!
//! Like the rest of [`super`], each driver is ONE
//! `#[magetypes(define(i16x16, i32x8), v3, neon, -scalar)]` body: the macro
//! emits a `_v3` variant on x86-64 and a `_neon` variant on aarch64, and every
//! operation the generic magetypes API cannot express (saturating add/sub,
//! saturating narrow, sign-extension, multiply-accumulate, lane reverse) comes
//! from [`super::prims16`] under one cfg-selected name. See that module for
//! the per-primitive proofs and for the audit of what magetypes does and does
//! not offer at i16 width.
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
//! `i16x16` lanes; the 17-bit butterfly transients live as exact i32 values in
//! `prims16`'s `P32`. Per-op proofs are on each primitive. The iadst and
//! identity kernels are NOT i16-safe (unclamped terminal negations /
//! multiplies exceed i16) and stay on the [`super`] i32x8 pass.

use archmage::prelude::*;
use magetypes::simd::generic::i16x16 as I16x16;

use crate::transform::cospi::{NEW_INV_SQRT2, NEW_SQRT2_BITS};

use super::inv1d_v3_i16_gen::*;
use super::prims16::{add_store_u8, mulhrs16, pack_clamp16, rev16, widen_hi, widen_lo};

/// The `mulhrs` multiplier that turns [`super::prims16::mulhrs16`] into
/// `round_shift(_, bit)` on i16 lanes: `m = 2^(15-bit)`.
///
/// EXACT for every i16 `v` and `bit` in `1..=4`. `mulhrs` computes
/// `floor((v*m + 2^14) / 2^15)` (both tiers — see the `prims16` module docs
/// for the AVX2/NEON equivalence); with `m = 2^(15-bit)` that is
/// `floor((v*2^(15-b) + 2^14) / 2^15)` = `floor((v + 2^(b-1)) / 2^b)` =
/// `round_shift(v, b)`. The intermediate product `|v*m| <= 2^29` is exact in
/// both tiers' internal precision, the result magnitude is `<= 2^14` (no
/// wrap), and the only saturating case (`-2^15 * -2^15`) is unreachable
/// because `m > 0`.
const fn rshift_mul(bit: i32) -> i16 {
    1i16 << (15 - bit)
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

/// Direct-dispatch the i16 kernel (same shape as [`super`]'s `run_inv1d`).
#[magetypes(define(i16x16), v3, neon, -scalar)]
pub(crate) fn run_inv1d_i16(
    t: Token,
    k: Inv1dI16,
    input: &[I16x16<Token>],
    out: &mut [I16x16<Token>],
) {
    let _ = t;
    match k {
        Inv1dI16::Dct4 => incant!(av1_idct4_i16_impl(input, out), [v3, neon]),
        Inv1dI16::Dct8 => incant!(av1_idct8_i16_impl(input, out), [v3, neon]),
        Inv1dI16::Dct16 => incant!(av1_idct16_i16_impl(input, out), [v3, neon]),
        Inv1dI16::Dct32 => incant!(av1_idct32_i16_impl(input, out), [v3, neon]),
        Inv1dI16::Dct64 => incant!(av1_idct64_i16_impl(input, out), [v3, neon]),
    }
}

/// Scalar twin of [`inv_col_pass_u8_i16`] — declines, routing the caller to
/// the i32 pass (and from there to its own scalar loop). Same contract as
/// [`super::inv_col_pass_u8_scalar`]: there is deliberately no scalar
/// *implementation* here, because the driver's per-column loop IS the
/// differential's reference.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inv_col_pass_u8_i16_scalar(
    _: ScalarToken,
    _kernel: Inv1dI16,
    _buf: &[i32],
    _output: &mut [u8],
    _stride: usize,
    _col_n: usize,
    _row_n: usize,
    _ud_flip: bool,
    _lr_flip: bool,
) -> bool {
    false
}

/// i16-lane u8 column pass. Preconditions (asserted by the caller in
/// [`super::try_inv_col_pass_u8`]): bd8 constants — `col_clamp == 16`,
/// round-shift bit 4, `cos_bit == 12` — and an audited DCT column kernel.
///
/// The vector scratch is TIERED by `row_n` for the same reason the i32 pass
/// tiers by `col_n`: a flat 64-entry array zero-init is a memset that
/// dominates the small transforms.
#[magetypes(define(i16x16), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn inv_col_pass_u8_i16(
    t: Token,
    kernel: Inv1dI16,
    buf: &[i32],
    output: &mut [u8],
    stride: usize,
    col_n: usize,
    row_n: usize,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    debug_assert!(row_n <= 64 && (col_n % 16 == 0 || col_n == 4 || col_n == 8));
    if row_n <= 8 {
        let mut tin = [i16x16::zero(t); 8];
        let mut tout = [i16x16::zero(t); 8];
        incant!(
            inv_col_pass_u8_i16_core(
                kernel, buf, output, stride, col_n, row_n, ud_flip, lr_flip, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    } else if row_n <= 16 {
        let mut tin = [i16x16::zero(t); 16];
        let mut tout = [i16x16::zero(t); 16];
        incant!(
            inv_col_pass_u8_i16_core(
                kernel, buf, output, stride, col_n, row_n, ud_flip, lr_flip, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    } else {
        let mut tin = [i16x16::zero(t); 64];
        let mut tout = [i16x16::zero(t); 64];
        incant!(
            inv_col_pass_u8_i16_core(
                kernel, buf, output, stride, col_n, row_n, ud_flip, lr_flip, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    }
    true
}

#[magetypes(define(i16x16, i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass_u8_i16_core(
    t: Token,
    kernel: Inv1dI16,
    buf: &[i32],
    output: &mut [u8],
    stride: usize,
    col_n: usize,
    row_n: usize,
    ud_flip: bool,
    lr_flip: bool,
    tin: &mut [I16x16<Token>],
    tout: &mut [I16x16<Token>],
) {
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
                        [row[7], row[6], row[5], row[4], row[3], row[2], row[1], row[0]],
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
        incant!(run_inv1d_i16(kernel, &tin[..row_n], &mut tout[..row_n]), [v3, neon]);
        for r in 0..row_n {
            let src = tout[if ud_flip { row_n - r - 1 } else { r }];
            // Terminal values are clamp_value outputs (i16); round_shift(_, 4)
            // then `clamp(dest + res, 0, 255)`, all exact in i16 lanes —
            // see `rshift_mul` and `prims16::add_store_u8`.
            let res = mulhrs16(t, src, rshift_mul(4));
            let idx = r * stride + c;
            if active == 16 {
                let d: &mut [u8; 16] = (&mut output[idx..idx + 16]).try_into().unwrap();
                add_store_u8(t, res, d);
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

// ---- i16-lane inverse ROW pass (bd8) ----------------------------------------
//
// The Phase-C follow-up lever: at bd8 the ROW pass has the SAME audited kernel
// contract as the column pass — the driver clamps every (possibly rect-scaled)
// input with `clamp_value(_, bd + 8 == 16)` BEFORE the kernel (exactly the i16
// domain), `stage_range == [16; 12]` (`opt_range(8)` is (16, 16) for BOTH
// directions), `cos_bit == 12` — so the SAME audited i16 DCT kernels run with
// lanes = ROWS (16 rows per vector; the input is stored column-major, so 16
// consecutive rows of one column are one contiguous 16-element load). Verified
// against `xtask/audit_i16_safety.py` (its stated domain — "every stage_range
// value == 16, driver input pre-clamped to i16" — IS the bd8 row entry
// condition): idct4/8/16/32/64 OK-i16; iadst8/16 have unclamped
// `wrapping_neg()` terminals over 17-bit transients, iadst4/iidentity* have
// unclamped 17-18-bit multiply terminals — all seven stay on the i32 pass.
//
// The row-specific stages stay OUTSIDE the kernel, each in a domain where it
// is exact:
//
// * the rect `round_shift(x * NewInvSqrt2, 12)` pre-scale runs on the RAW i32
//   input in i32 lanes ([`super::prims::mul_rshiftv`], exact for ANY i32) —
//   BEFORE the clamp, exactly like the scalar driver;
// * the driver's `clamp_buf(16)` is the saturating narrow
//   ([`super::prims16::pack_clamp16`]);
// * the post-kernel `round_shift_array(-shift[0])` (shift 1 or 2 for every
//   `row_n % 16 == 0` size; 0 is the scalar early-return identity and is
//   never instantiated as a shift) operates on kernel TERMINAL outputs —
//   `clamp_value(_, 16)` outputs, i16 — via the `mulhrs` identity proven at
//   [`rshift_mul`];
// * the store sign-extends i16 -> i32 into the UNCHANGED row-major i32 `buf`
//   (a pure width extension: the scalar row output IS that i16 value in i32
//   representation), so every column pass — the landed i16 DCT pass and the
//   i32 iadst/identity pass — reads byte-identical values to the i32 row
//   pass's output.
//
// Only `row_n % 16 == 0` (16/32/64) runs here: a 4- or 8-tall transform would
// execute the SAME kernel instruction count as the i32x8 pass (one partial
// batch either way) PLUS the pack/widen overhead — a structural loss, so
// those sizes stay on the i32 pass.

/// Scalar twin of [`inv_row_pass_i16`] — declines (see
/// [`inv_col_pass_u8_i16_scalar`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn inv_row_pass_i16_scalar(
    _: ScalarToken,
    _kernel: Inv1dI16,
    _mod_input: &[i32],
    _buf: &mut [i32],
    _col_n: usize,
    _row_n: usize,
    _rect1: bool,
    _shift0_bit: i32,
) -> bool {
    false
}

/// i16-lane inverse ROW pass. Preconditions (asserted by the caller in
/// [`super::try_inv_row_pass`]): bd8 row constants — row clamp 16, every
/// `stage_range` entry 16, `cos_bit == 12`, `shift0_bit` in 0..=2 — an
/// audited DCT row kernel spanning `col_n` points, and `row_n % 16 == 0`.
#[magetypes(define(i16x16), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn inv_row_pass_i16(
    t: Token,
    kernel: Inv1dI16,
    mod_input: &[i32],
    buf: &mut [i32],
    col_n: usize,
    row_n: usize,
    rect1: bool,
    shift0_bit: i32,
) -> bool {
    debug_assert!(row_n % 16 == 0 && row_n <= 64 && (0..=2).contains(&shift0_bit));
    if col_n <= 8 {
        let mut tin = [i16x16::zero(t); 8];
        let mut tout = [i16x16::zero(t); 8];
        incant!(
            inv_row_pass_i16_core(
                kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    } else if col_n <= 16 {
        let mut tin = [i16x16::zero(t); 16];
        let mut tout = [i16x16::zero(t); 16];
        incant!(
            inv_row_pass_i16_core(
                kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    } else {
        let mut tin = [i16x16::zero(t); 64];
        let mut tout = [i16x16::zero(t); 64];
        incant!(
            inv_row_pass_i16_core(
                kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, &mut tin, &mut tout,
            ),
            [v3, neon]
        );
    }
    true
}

/// The i16 row-pass body over caller-sized scratch (see [`inv_row_pass_i16`];
/// tiering rationale at [`super::inv_row_pass`]).
#[magetypes(define(i16x16, i32x8), v3, neon, -scalar)]
#[allow(clippy::too_many_arguments)]
fn inv_row_pass_i16_core(
    t: Token,
    kernel: Inv1dI16,
    mod_input: &[i32],
    buf: &mut [i32],
    col_n: usize,
    row_n: usize,
    rect1: bool,
    shift0_bit: i32,
    tin: &mut [I16x16<Token>],
    tout: &mut [I16x16<Token>],
) {
    let mut rg = 0usize;
    while rg < row_n {
        // Gather 16 rows of each input column c — contiguous (column-major
        // input), rect-scaled in i32 lanes (exact for any i32), then the
        // driver's clamp_buf(16) as the saturating narrow.
        for (c, ti) in tin[..col_n].iter_mut().enumerate() {
            let base = c * row_n + rg;
            let mut a = i32x8::from_slice(t, &mod_input[base..base + 8]);
            let mut b = i32x8::from_slice(t, &mod_input[base + 8..base + 16]);
            if rect1 {
                a = super::prims::mul_rshiftv(t, a, NEW_INV_SQRT2, NEW_SQRT2_BITS);
                b = super::prims::mul_rshiftv(t, b, NEW_INV_SQRT2, NEW_SQRT2_BITS);
            }
            *ti = pack_clamp16(t, a, b);
        }
        incant!(run_inv1d_i16(kernel, &tin[..col_n], &mut tout[..col_n]), [v3, neon]);
        // round_shift_array(buf_row, -shift[0]); shift[0] in {0,-1,-2}. Bit 0
        // is the scalar early return — an identity, never instantiated as a
        // shift (const-0 shift trap); 1/2 use the exact `rshift_mul` identity.
        if shift0_bit > 0 {
            let m = if shift0_bit == 1 { rshift_mul(1) } else { rshift_mul(2) };
            for to in tout[..col_n].iter_mut() {
                *to = mulhrs16(t, *to, m);
            }
        }
        // Store: buf[(rg+k)*col_n + c] = tout[c].lane(k) as i32 — widen each
        // output vector to natural-order i32 halves (rows rg..rg+8 and
        // rg+8..rg+16), 8x8 i32 transpose per half into the row-major buf;
        // per-lane scatter for the col_n == 4 tail.
        let full = col_n & !7;
        let mut cg = 0usize;
        while cg < full {
            let mut half = [i32x8::zero(t); 8];
            for (j, h) in half.iter_mut().enumerate() {
                *h = widen_lo(t, tout[cg + j]);
            }
            let tr = super::prims::transpose8(t, &half);
            for (k, trk) in tr.iter().enumerate() {
                let base = (rg + k) * col_n + cg;
                trk.store((&mut buf[base..base + 8]).try_into().unwrap());
            }
            for (j, h) in half.iter_mut().enumerate() {
                *h = widen_hi(t, tout[cg + j]);
            }
            let tr = super::prims::transpose8(t, &half);
            for (k, trk) in tr.iter().enumerate() {
                let base = (rg + 8 + k) * col_n + cg;
                trk.store((&mut buf[base..base + 8]).try_into().unwrap());
            }
            cg += 8;
        }
        for c in full..col_n {
            let a = tout[c].to_array();
            for (k, &av) in a.iter().enumerate() {
                buf[(rg + k) * col_n + c] = av as i32;
            }
        }
        rg += 16;
    }
}

#[cfg(test)]
mod tests {
    //! i16-lane-vs-scalar differentials for the Phase C kernels and the row
    //! pass: over their FULL contract domains (every kernel lane an arbitrary
    //! i16 — exactly the `clamp_value(_, 16)` image the driver feeds them at
    //! bd8; every row-pass input an arbitrary i32 — the rect scale + clamp are
    //! exact on all of i32), the vector path must be bit-identical to the
    //! scalar transcription with `stage_range == [16; 12]`, `cos_bit == 12`.
    //! Dense random + the exact saturation boundaries (`i16::MIN`/`i16::MAX`
    //! sign patterns are the half_btf |p0 + p1| and adds/subs saturation
    //! maximizers). Runs at every token permutation on BOTH architectures,
    //! with a counter proving a vector tier actually ran (non-vacuous even
    //! under AOM_FORCE_SCALAR — the permutation harness owns token state).

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

    #[magetypes(define(i16x16), v3, neon, -scalar)]
    fn assert_batch16(
        t: Token,
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
        incant!(run_inv1d_i16(kernel, &vin, &mut vout), [v3, neon]);

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

    /// Scalar replica of the driver's bd8 ROW loop (`inv_txfm2d.rs`) — the
    /// same ops in the same order: optional rect NewInvSqrt2 scale (raw i32,
    /// i64 round_shift), `clamp_buf(16)`, the scalar kernel (stage_range
    /// `[16; 12]`, cos_bit 12), `round_shift_array(-shift[0])`.
    fn scalar_row_pass(
        scalar: ScalarKernel,
        mod_input: &[i32],
        buf: &mut [i32],
        col_n: usize,
        row_n: usize,
        rect1: bool,
        shift0_bit: i32,
    ) {
        use crate::transform::cospi::{NEW_INV_SQRT2, NEW_SQRT2_BITS};
        use crate::transform::fdct::{clamp_value, round_shift};
        let stage_range = [16i8; 12];
        let mut ti = vec![0i32; col_n];
        for r in 0..row_n {
            for c in 0..col_n {
                let v = mod_input[c * row_n + r];
                let v = if rect1 {
                    round_shift(v as i64 * NEW_INV_SQRT2 as i64, NEW_SQRT2_BITS)
                } else {
                    v
                };
                ti[c] = clamp_value(v, 16);
            }
            let out = &mut buf[r * col_n..(r + 1) * col_n];
            scalar(&ti, out, 12, &stage_range);
            if shift0_bit > 0 {
                for v in out.iter_mut() {
                    *v = round_shift(*v as i64, shift0_bit);
                }
            }
        }
    }

    #[magetypes(define(i16x16), v3, neon, -scalar)]
    #[allow(clippy::too_many_arguments)]
    fn assert_row_pass(
        t: Token,
        name: &str,
        scalar: ScalarKernel,
        kernel: Inv1dI16,
        mod_input: &[i32],
        col_n: usize,
        row_n: usize,
        rect1: bool,
        shift0_bit: i32,
        label: &str,
    ) {
        let _ = t;
        // Distinct sentinels: a position missed by BOTH passes still mismatches.
        let mut vbuf = vec![111_i32; col_n * row_n];
        assert!(incant!(
            inv_row_pass_i16(kernel, mod_input, &mut vbuf, col_n, row_n, rect1, shift0_bit),
            [v3, neon]
        ));
        let mut sbuf = vec![-222_i32; col_n * row_n];
        scalar_row_pass(scalar, mod_input, &mut sbuf, col_n, row_n, rect1, shift0_bit);
        for r in 0..row_n {
            for c in 0..col_n {
                assert_eq!(
                    vbuf[r * col_n + c],
                    sbuf[r * col_n + c],
                    "{name}: {label} row={r} col={c} col_n={col_n} row_n={row_n} \
                     rect1={rect1} shift0={shift0_bit}"
                );
            }
        }
    }

    /// Scalar-only permutation: there is no vector pass to compare against.
    fn sweep_rows_scalar(_: ScalarToken, _tier: &str) -> bool {
        false
    }

    /// Row-pass differential at ONE tier. Returns true (a vector tier ran).
    #[magetypes(define(i16x16), v3, neon, -scalar)]
    fn sweep_rows(t: Token, tier: &str) -> bool {
        let _ = t;
        let mut rng = Rng(0x_bd08_1616_2026_0723);
        for (name, n, scalar, kernel) in cases() {
            let col_n = n;
            for &row_n in &[16usize, 32, 64] {
                for &rect1 in &[false, true] {
                    for &shift0 in &[0i32, 1, 2] {
                        // (a) dense random over the FULL i32 domain.
                        for rep in 0..6 {
                            let mi: Vec<i32> =
                                (0..col_n * row_n).map(|_| rng.next() as i32).collect();
                            incant!(
                                assert_row_pass(
                                    name,
                                    scalar,
                                    kernel,
                                    &mi,
                                    col_n,
                                    row_n,
                                    rect1,
                                    shift0,
                                    &format!("[{tier}] rand rep{rep}"),
                                ),
                                [v3, neon]
                            );
                        }
                        // (b) boundary cycle: i32 extremes, the i16
                        // saturation edges (the clamp boundaries), and
                        // rect-scale magnifiers (2896 * 46341 spans the
                        // i32/i16 boundary region after >>12).
                        let specials = [
                            i32::MIN,
                            i32::MAX,
                            -32769,
                            -32768,
                            -32767,
                            32766,
                            32767,
                            32768,
                            0,
                            -1,
                            1,
                            46341,
                            -46341,
                            i32::MIN + 1,
                            i32::MAX - 1,
                            -2,
                        ];
                        let mi: Vec<i32> =
                            (0..col_n * row_n).map(|i| specials[i % specials.len()]).collect();
                        incant!(
                            assert_row_pass(
                                name,
                                scalar,
                                kernel,
                                &mi,
                                col_n,
                                row_n,
                                rect1,
                                shift0,
                                &format!("[{tier}] bound cycle"),
                            ),
                            [v3, neon]
                        );
                        // (c) alternating extremes (max |half_btf| sums).
                        let mi: Vec<i32> = (0..col_n * row_n)
                            .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
                            .collect();
                        incant!(
                            assert_row_pass(
                                name,
                                scalar,
                                kernel,
                                &mi,
                                col_n,
                                row_n,
                                rect1,
                                shift0,
                                &format!("[{tier}] bound alt"),
                            ),
                            [v3, neon]
                        );
                    }
                }
            }
        }
        true
    }

    /// Row-pass differential: [`inv_row_pass_i16`] vs the scalar row loop over
    /// the pass's FULL contract domain — ANY i32 input, every kernel × row_n
    /// {16,32,64} × rect on/off × shift0 {0,1,2}. Every token permutation on
    /// every architecture; non-vacuity asserted.
    #[test]
    fn inv_row_pass_i16_bit_identical_to_scalar_at_every_tier() {
        let _ = crate::dispatch::scalar_forced(); // fire the pin before the harness
        let mut simd_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            if incant!(sweep_rows(&tier.label), [v3, neon, scalar]) {
                simd_ran += 1;
            }
        });
        eprintln!("inv row pass i16 parity: {report}, vector permutations run: {simd_ran}");
        assert!(
            simd_ran >= 1,
            "a vector tier must run at least once (AVX2 on x86-64, NEON on aarch64)"
        );
        assert!(report.permutations_run >= 2);
    }

    /// Scalar-only permutation: there is no vector kernel to compare against.
    fn sweep_kernels_scalar(_: ScalarToken, _tier: &str) -> bool {
        false
    }

    /// Kernel differential at ONE tier. Returns true (a vector tier ran).
    #[magetypes(define(i16x16), v3, neon, -scalar)]
    fn sweep_kernels(t: Token, tier: &str) -> bool {
        let _ = t;
        let mut rng = Rng(0x_bd08_1616_2026_0722);
        for (name, n, scalar, kernel) in cases() {
            // (a) dense random over the FULL i16 lane domain.
            for rep in 0..64 {
                let cols: Vec<[i16; 16]> =
                    (0..n).map(|_| core::array::from_fn(|_| rng.lane())).collect();
                incant!(
                    assert_batch16(
                        name,
                        n,
                        scalar,
                        kernel,
                        &cols,
                        &format!("[{tier}] rand rep{rep}"),
                    ),
                    [v3, neon]
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
                let cols: Vec<[i16; 16]> =
                    (0..n).map(|r| core::array::from_fn(|l| pat(r, l))).collect();
                incant!(
                    assert_batch16(
                        name,
                        n,
                        scalar,
                        kernel,
                        &cols,
                        &format!("[{tier}] bound pat{pi}"),
                    ),
                    [v3, neon]
                );
            }
            // (c) boundary lanes mixed with zero columns.
            let mut cols = vec![[0i16; 16]; n];
            cols[0] = core::array::from_fn(|l| [lo, hi, 0, -1, 1, lo + 1, hi - 1, 2][l % 8]);
            cols[n - 1] = core::array::from_fn(|l| [hi, lo, -2, 2, 0, -1, 1, lo + 1][l % 8]);
            incant!(
                assert_batch16(name, n, scalar, kernel, &cols, &format!("[{tier}] extremes")),
                [v3, neon]
            );
        }
        true
    }

    #[test]
    fn inv1d_i16_bit_identical_to_scalar_at_every_tier() {
        let _ = crate::dispatch::scalar_forced(); // fire the pin before the harness
        let mut simd_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            if incant!(sweep_kernels(&tier.label), [v3, neon, scalar]) {
                simd_ran += 1;
            }
        });
        eprintln!("inv1d i16 parity: {report}, vector permutations run: {simd_ran}");
        assert!(
            simd_ran >= 1,
            "a vector tier must run at least once (AVX2 on x86-64, NEON on aarch64)"
        );
        assert!(report.permutations_run >= 2);
    }
}
