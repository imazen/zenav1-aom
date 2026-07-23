//! SIMD deblock loop-filter kernels (Gate 3) — bit-identical to the highbd
//! scalar core, at every dispatch tier (`tests/lpf_simd_diff.rs`).
//!
//! Same aom-rs SIMD pattern as `crate::cdef` / `crate::txb`: ONE magetypes generic
//! kernel (`#[magetypes(define(i32x4), v3, neon, wasm128, -scalar)]`), a
//! hand-written `_scalar` tier that calls the untouched highbd transcription,
//! `incant!` dispatch, `crate::dispatch::scalar_forced()` pin at the entry.
//!
//! # Layout
//!
//! Each `aom_(highbd_)lpf_*` call filters **4 edge positions** — AV1's 4-px
//! edge segment (`aom_dsp/loopfilter.c`, the `for (i = 0; i < 4; ++i)` loops).
//! Those 4 positions are the 4 SIMD lanes. Tap `k` for lane `l` lives at
//! `center + l*step + k*ts`, where `ts` = tap stride and `step` = position
//! advance (horizontal: `ts` = pitch, `step` = 1; vertical: `ts` = 1,
//! `step` = pitch). Taps are gathered per lane into an `i32x4`; the filter
//! math runs once across the 4 lanes instead of the scalar core's 4 sequential
//! iterations.
//!
//! # Branchless width selection
//!
//! The scalar core branches per position: `filter6`/`filter8` apply the wide
//! (flat-region) filter iff `flat && mask`, else fall back to `filter4`; those
//! branches differ per lane. This kernel computes BOTH the `filter4` result and
//! the wide result for all 4 lanes and blends per lane on the `flat & mask`
//! lane mask (`i32x4::blend`) — the standard libaom SIMD structure. `filter4`
//! itself is already branchless (its `& mask` / `& hev` gates zero out the
//! contribution for unfiltered lanes, reproducing the scalar identity when
//! `mask == 0`). Taps that only the wide filter writes (`p2`/`q2` in the 8-tap
//! filter) blend against the ORIGINAL sample, matching the scalar core leaving
//! them untouched on the `filter4` fallback.
//!
//! # Bit-exactness (full highbd domain, bd 8/10/12)
//!
//! The highbd scalar core does the `filter4` math in `i16` (`filter`, `hev`,
//! `filter1/2`, `f`) and the WIDE (6/8/14-tap) sums in `i32`. This kernel runs
//! EVERYTHING in `i32` lanes, which reproduces the scalar result lane-for-lane:
//!
//! * `scc` (`signed_char_clamp_high`) clamps to `[-(128<<sh), (128<<sh)-1]`
//!   ⊆ `[-2048, 2047]` (bd ≤ 12) — always inside `i16`, so the `i32` clamp and
//!   the scalar `i16` clamp produce the same number.
//! * The `& hev` / `& mask` / `& !hev` gates are `0`/`-1` lane masks; ANDing a
//!   value already in the `i16` range with an all-ones/all-zero `i32` mask
//!   yields the same value the scalar `i16` AND does.
//! * `>> 3` / `>> 1` are arithmetic shifts on values in the `i16` range; the
//!   wide-sum rounding (`rpo2`) shifts a NON-NEGATIVE sum, so `>> n` is a plain
//!   unsigned divide. `i32` and `i16`/scalar agree in both cases.
//! * `iabs(a,b) = |a-b|` with `a,b` ∈ `[0, 4095]`, so `.abs()` never hits the
//!   `i32::MIN` corner; `iabs/2` is `>> 1` on a non-negative value.
//! * The wide 14-tap weighted sums reach ~`4095*16 = 65520` at bd 12 — beyond
//!   `i16`, which is exactly why the accumulation is `i32` here (and in the
//!   scalar core). Comparisons (`filter_mask*` / `flat_mask*` / `hev_mask`)
//!   use `simd_gt`, matching the scalar `iabs > threshold`.
//!
//! `bd == 8` runs this same path with `sh = 0` (the decoder keeps `u16`
//! samples at every bit depth); the differential proves it against the REAL C
//! lowbd kernels via `hbd_lpf_diff.rs` (dispatch-vs-C) plus the SIMD-vs-scalar
//! `lpf_simd_diff.rs` at every token tier.

use archmage::prelude::*;

/// Dispatch entry for one 4-position deblock edge segment (highbd `u16` path).
/// `ts` = tap stride, `step` = position advance; the axis is encoded by the
/// caller ([`crate::loopfilter::highbd::horizontal`] / [`crate::loopfilter::highbd::vertical`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lpf(
    width: u32,
    buf: &mut [u16],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        lpf_impl(width, buf, center, ts, step, bl, li, th, bd),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the untouched highbd transcription, verbatim.
#[allow(clippy::too_many_arguments)]
fn lpf_impl_scalar(
    _t: archmage::ScalarToken,
    width: u32,
    buf: &mut [u16],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
) {
    crate::loopfilter::highbd::lpf_scalar(width, buf, center, ts, step, bl, li, th, bd);
}

#[magetypes(define(i32x4), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn lpf_impl(
    token: Token,
    width: u32,
    buf: &mut [u16],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
) {
    let shift = bd - 8;
    let bias: i32 = 0x80 << shift;
    let lim: i32 = 128 << shift;
    let neg_lim = i32x4::splat(token, -lim);
    let lim_hi = i32x4::splat(token, lim - 1);
    let l = i32x4::splat(token, (li as i32) << shift); // limit
    let blv = i32x4::splat(token, (bl as i32) << shift); // blimit
    let ft = i32x4::splat(token, 1 << shift); // flat_mask thresh (== 1)

    // signed_char_clamp_high
    let scc = |v: i32x4| v.clamp(neg_lim, lim_hi);
    // |a - b| (a,b are u16 pixels widened to i32, so abs() is exact)
    let iabs = |a: i32x4, b: i32x4| (a - b).abs();
    // rpo2(v, n): round-power-of-two on a non-negative sum
    let rpo3 = |v: i32x4| (v + 4).shr_logical_const::<3>();
    let rpo4 = |v: i32x4| (v + 8).shr_logical_const::<4>();

    let c = center as isize;
    // Gather tap `k` (offset k*ts from center) across the 4 edge positions
    // (offset l*step, l in 0..4) into one i32x4 lane vector.
    let load = |k: isize| -> i32x4 {
        i32x4::from_array(
            token,
            [
                buf[(c + k * ts) as usize] as i32,
                buf[(c + step + k * ts) as usize] as i32,
                buf[(c + 2 * step + k * ts) as usize] as i32,
                buf[(c + 3 * step + k * ts) as usize] as i32,
            ],
        )
    };

    // filter4: taps p1(-2) p0(-1) q0(0) q1(1); `mask` is the filter_mask lane
    // mask (all-ones = filter). Returns (op1', op0', oq0', oq1').
    let filter4 = |op1: i32x4,
                   op0: i32x4,
                   oq0: i32x4,
                   oq1: i32x4,
                   mask: i32x4|
     -> (i32x4, i32x4, i32x4, i32x4) {
        let ps1 = op1 - bias;
        let ps0 = op0 - bias;
        let qs0 = oq0 - bias;
        let qs1 = oq1 - bias;
        let t_hev = i32x4::splat(token, (th as i32) << shift);
        let hev = iabs(op1, op0).simd_gt(t_hev) | iabs(oq1, oq0).simd_gt(t_hev);

        let mut filter = scc(ps1 - qs1) & hev;
        filter = scc(filter + (qs0 - ps0) * 3) & mask;
        let filter1 = scc(filter + 4).shr_arithmetic_const::<3>();
        let filter2 = scc(filter + 3).shr_arithmetic_const::<3>();
        let n_oq0 = scc(qs0 - filter1) + bias;
        let n_op0 = scc(ps0 + filter2) + bias;
        let f = ((filter1 + 1).shr_arithmetic_const::<1>()) & hev.not();
        let n_oq1 = scc(qs1 - f) + bias;
        let n_op1 = scc(ps1 + f) + bias;
        (n_op1, n_op0, n_oq0, n_oq1)
    };

    // filter_mask2(limit, blimit, p1, p0, q0, q1) -> filter mask (post-NOT)
    let fmask2 = |p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4| -> i32x4 {
        (iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    // filter_mask3_chroma(limit, blimit, p2,p1,p0,q0,q1,q2)
    let fmask6 = |p2: i32x4, p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4, q2: i32x4| -> i32x4 {
        (iabs(p2, p1).simd_gt(l)
            | iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | iabs(q2, q1).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    // filter_mask(limit, blimit, p3,p2,p1,p0,q0,q1,q2,q3)
    let fmask8 = |p3: i32x4,
                  p2: i32x4,
                  p1: i32x4,
                  p0: i32x4,
                  q0: i32x4,
                  q1: i32x4,
                  q2: i32x4,
                  q3: i32x4|
     -> i32x4 {
        (iabs(p3, p2).simd_gt(l)
            | iabs(p2, p1).simd_gt(l)
            | iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | iabs(q2, q1).simd_gt(l)
            | iabs(q3, q2).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    // flat_mask3_chroma(1, p2,p1,p0,q0,q1,q2)
    let flat3 = |p2: i32x4, p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4, q2: i32x4| -> i32x4 {
        (iabs(p1, p0).simd_gt(ft)
            | iabs(q1, q0).simd_gt(ft)
            | iabs(p2, p0).simd_gt(ft)
            | iabs(q2, q0).simd_gt(ft))
        .not()
    };
    // flat_mask4(1, p3,p2,p1,p0,q0,q1,q2,q3)
    let flat4 = |p3: i32x4,
                 p2: i32x4,
                 p1: i32x4,
                 p0: i32x4,
                 q0: i32x4,
                 q1: i32x4,
                 q2: i32x4,
                 q3: i32x4|
     -> i32x4 {
        (iabs(p1, p0).simd_gt(ft)
            | iabs(q1, q0).simd_gt(ft)
            | iabs(p2, p0).simd_gt(ft)
            | iabs(q2, q0).simd_gt(ft)
            | iabs(p3, p0).simd_gt(ft)
            | iabs(q3, q0).simd_gt(ft))
        .not()
    };

    // Scatter helper values into `buf` for taps `ks` — direct indexing so the
    // (immutable) `load` closure's borrow has ended by this point.
    macro_rules! store {
        ($($k:expr => $v:expr),+ $(,)?) => {{
            $(
                let a = ($v).to_array();
                buf[(c + ($k) * ts) as usize] = a[0] as u16;
                buf[(c + step + ($k) * ts) as usize] = a[1] as u16;
                buf[(c + 2 * step + ($k) * ts) as usize] = a[2] as u16;
                buf[(c + 3 * step + ($k) * ts) as usize] = a[3] as u16;
            )+
        }};
    }

    match width {
        4 => {
            // taps p1(-2) p0(-1) q0(0) q1(1)
            let op1 = load(-2);
            let op0 = load(-1);
            let oq0 = load(0);
            let oq1 = load(1);
            let mask = fmask2(op1, op0, oq0, oq1);
            let (n1, n0, m0, m1) = filter4(op1, op0, oq0, oq1, mask);
            store!(-2 => n1, -1 => n0, 0 => m0, 1 => m1);
        }
        6 => {
            // taps p2(-3) p1(-2) p0(-1) q0(0) q1(1) q2(2)
            let p2 = load(-3);
            let p1 = load(-2);
            let p0 = load(-1);
            let q0 = load(0);
            let q1 = load(1);
            let q2 = load(2);
            let mask = fmask6(p2, p1, p0, q0, q1, q2);
            let flat = flat3(p2, p1, p0, q0, q1, q2);
            let use_wide = flat & mask;
            // wide 6-tap (writes p1,p0,q0,q1)
            let w_p1 = rpo3(p2 * 3 + p1 * 2 + p0 * 2 + q0);
            let w_p0 = rpo3(p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1);
            let w_q0 = rpo3(p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2);
            let w_q1 = rpo3(p0 + q0 * 2 + q1 * 2 + q2 * 3);
            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            let o_p1 = i32x4::blend(use_wide, w_p1, f_p1);
            let o_p0 = i32x4::blend(use_wide, w_p0, f_p0);
            let o_q0 = i32x4::blend(use_wide, w_q0, f_q0);
            let o_q1 = i32x4::blend(use_wide, w_q1, f_q1);
            store!(-2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1);
        }
        8 => {
            // taps p3(-4) p2(-3) p1(-2) p0(-1) q0(0) q1(1) q2(2) q3(3)
            let p3 = load(-4);
            let p2 = load(-3);
            let p1 = load(-2);
            let p0 = load(-1);
            let q0 = load(0);
            let q1 = load(1);
            let q2 = load(2);
            let q3 = load(3);
            let mask = fmask8(p3, p2, p1, p0, q0, q1, q2, q3);
            let flat = flat4(p3, p2, p1, p0, q0, q1, q2, q3);
            let use_wide = flat & mask;
            // wide 8-tap (writes p2,p1,p0,q0,q1,q2)
            let w_p2 = rpo3(p3 * 3 + p2 * 2 + p1 + p0 + q0);
            let w_p1 = rpo3(p3 * 2 + p2 + p1 * 2 + p0 + q0 + q1);
            let w_p0 = rpo3(p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2);
            let w_q0 = rpo3(p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3);
            let w_q1 = rpo3(p1 + p0 + q0 + q1 * 2 + q2 + q3 * 2);
            let w_q2 = rpo3(p0 + q0 + q1 + q2 * 2 + q3 * 3);
            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            // p2/q2 fall back to the ORIGINAL sample (filter4 leaves them).
            let o_p2 = i32x4::blend(use_wide, w_p2, p2);
            let o_p1 = i32x4::blend(use_wide, w_p1, f_p1);
            let o_p0 = i32x4::blend(use_wide, w_p0, f_p0);
            let o_q0 = i32x4::blend(use_wide, w_q0, f_q0);
            let o_q1 = i32x4::blend(use_wide, w_q1, f_q1);
            let o_q2 = i32x4::blend(use_wide, w_q2, q2);
            store!(-3 => o_p2, -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1, 2 => o_q2);
        }
        14 => {
            // taps p6(-7)..p0(-1), q0(0)..q6(6)
            let p6 = load(-7);
            let p5 = load(-6);
            let p4 = load(-5);
            let p3 = load(-4);
            let p2 = load(-3);
            let p1 = load(-2);
            let p0 = load(-1);
            let q0 = load(0);
            let q1 = load(1);
            let q2 = load(2);
            let q3 = load(3);
            let q4 = load(4);
            let q5 = load(5);
            let q6 = load(6);

            let mask = fmask8(p3, p2, p1, p0, q0, q1, q2, q3);
            let flat = flat4(p3, p2, p1, p0, q0, q1, q2, q3);
            // flat2 = flat_mask4(1, p6,p5,p4,p0,q0,q4,q5,q6)
            let flat2 = flat4(p6, p5, p4, p0, q0, q4, q5, q6);
            let use8 = flat & mask;
            let use14 = flat2 & use8;

            // filter4 fallback (deepest else, taps p1,p0,q0,q1 with the 8-tap mask)
            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            // wide 8-tap (writes p2,p1,p0,q0,q1,q2)
            let w8_p2 = rpo3(p3 * 3 + p2 * 2 + p1 + p0 + q0);
            let w8_p1 = rpo3(p3 * 2 + p2 + p1 * 2 + p0 + q0 + q1);
            let w8_p0 = rpo3(p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2);
            let w8_q0 = rpo3(p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3);
            let w8_q1 = rpo3(p1 + p0 + q0 + q1 * 2 + q2 + q3 * 2);
            let w8_q2 = rpo3(p0 + q0 + q1 + q2 * 2 + q3 * 3);
            // wide 14-tap (writes p5,p4,p3,p2,p1,p0,q0,q1,q2,q3,q4,q5)
            let w14_p5 = rpo4(p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0);
            let w14_p4 = rpo4(p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1);
            let w14_p3 = rpo4(p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2);
            let w14_p2 =
                rpo4(p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3);
            let w14_p1 =
                rpo4(p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4);
            let w14_p0 =
                rpo4(p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5);
            let w14_q0 =
                rpo4(p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6);
            let w14_q1 =
                rpo4(p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 * 2);
            let w14_q2 = rpo4(p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 * 3);
            let w14_q3 = rpo4(p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 * 4);
            let w14_q4 = rpo4(p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 * 5);
            let w14_q5 = rpo4(p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 7);

            // 3-way nested select: use14 ? wide14 : (use8 ? wide8 : base).
            // p5,p4,p3,q3,q4,q5 are written only by wide14 (base = original).
            // p2,q2 by wide8+wide14 (base = original). p1,p0,q0,q1 by all
            // three (base = filter4).
            let o_p5 = i32x4::blend(use14, w14_p5, p5);
            let o_p4 = i32x4::blend(use14, w14_p4, p4);
            let o_p3 = i32x4::blend(use14, w14_p3, p3);
            let o_p2 = i32x4::blend(use14, w14_p2, i32x4::blend(use8, w8_p2, p2));
            let o_p1 = i32x4::blend(use14, w14_p1, i32x4::blend(use8, w8_p1, f_p1));
            let o_p0 = i32x4::blend(use14, w14_p0, i32x4::blend(use8, w8_p0, f_p0));
            let o_q0 = i32x4::blend(use14, w14_q0, i32x4::blend(use8, w8_q0, f_q0));
            let o_q1 = i32x4::blend(use14, w14_q1, i32x4::blend(use8, w8_q1, f_q1));
            let o_q2 = i32x4::blend(use14, w14_q2, i32x4::blend(use8, w8_q2, q2));
            let o_q3 = i32x4::blend(use14, w14_q3, q3);
            let o_q4 = i32x4::blend(use14, w14_q4, q4);
            let o_q5 = i32x4::blend(use14, w14_q5, q5);
            store!(
                -6 => o_p5, -5 => o_p4, -4 => o_p3, -3 => o_p2, -2 => o_p1, -1 => o_p0,
                0 => o_q0, 1 => o_q1, 2 => o_q2, 3 => o_q3, 4 => o_q4, 5 => o_q5,
            );
        }
        _ => crate::loopfilter::highbd::lpf_scalar(width, buf, center, ts, step, bl, li, th, bd),
    }
}

// ---- lowbd (bd8, u8 pixel) deblock SIMD ----------------------------------------
//
// The bd8 "lowbd" decode pipeline stores reconstruction planes as `u8` instead
// of `u16`. This is the byte-for-byte twin of [`lpf`]/[`lpf_impl`] with the
// pixel loads/stores narrowed to `u8` and `bd` fixed at 8 — so `shift = bd-8 =
// 0`, `bias = 0x80`, `lim = 128`, and every threshold is unshifted. Every
// i32-domain lane op (the tap gather, `filter4`, the `filter_mask*`/`flat*`
// predicates, the wide-tap round-shifts, the per-lane blends) is IDENTICAL to
// the u16 core, so a lane that stores value `v` here stores the SAME `v` the u16
// core stores at bd8 (a bd8 sample is `< 256`, and `u8`/`u16` agree on it). The
// i32x4 lane math is not narrowed — the loop filter's SIMD width is fixed at 4
// (the 4 edge positions of one `aom_lpf_*` call), independent of pixel width, so
// only the destination storage narrows; this is the "safe first step" the
// transform foundation established, and it cannot move a pixel (proven by
// `loopfilter_lowbd_diff` against the REAL C lowbd kernels AND the u16 port).
//
// This mirrors the transform's [`crate::transform::simd::try_inv_col_pass_u8`]
// (the u8 twin of the u16 column pass) — the sanctioned lowbd fan-out pattern:
// duplicate ONLY the pixel-touching SIMD pass, leave the u16 path byte-untouched.

/// Dispatch entry for one 4-position deblock edge segment (lowbd `u8` path).
/// `ts` = tap stride, `step` = position advance (the axis is encoded by the
/// caller [`crate::loopfilter::horizontal`] / [`crate::loopfilter::vertical`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lpf_u8(
    width: u32,
    buf: &mut [u8],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        lpf_impl_u8(width, buf, center, ts, step, bl, li, th),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the untouched u8 lowbd transcription, verbatim.
#[allow(clippy::too_many_arguments)]
fn lpf_impl_u8_scalar(
    _t: archmage::ScalarToken,
    width: u32,
    buf: &mut [u8],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
) {
    crate::loopfilter::lpf_scalar(width, buf, center, ts, step, bl, li, th);
}

#[magetypes(define(i32x4), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn lpf_impl_u8(
    token: Token,
    width: u32,
    buf: &mut [u8],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
) {
    // bd == 8 ⇒ shift == 0: bias 0x80, clamp [-128,127], thresholds unshifted.
    const BIAS: i32 = 0x80;
    let neg_lim = i32x4::splat(token, -128);
    let lim_hi = i32x4::splat(token, 127);
    let l = i32x4::splat(token, li as i32); // limit
    let blv = i32x4::splat(token, bl as i32); // blimit
    let ft = i32x4::splat(token, 1); // flat_mask thresh (== 1)

    // signed_char_clamp
    let scc = |v: i32x4| v.clamp(neg_lim, lim_hi);
    let iabs = |a: i32x4, b: i32x4| (a - b).abs();
    let rpo3 = |v: i32x4| (v + 4).shr_logical_const::<3>();
    let rpo4 = |v: i32x4| (v + 8).shr_logical_const::<4>();

    let c = center as isize;
    let load = |k: isize| -> i32x4 {
        i32x4::from_array(
            token,
            [
                buf[(c + k * ts) as usize] as i32,
                buf[(c + step + k * ts) as usize] as i32,
                buf[(c + 2 * step + k * ts) as usize] as i32,
                buf[(c + 3 * step + k * ts) as usize] as i32,
            ],
        )
    };

    let filter4 = |op1: i32x4,
                   op0: i32x4,
                   oq0: i32x4,
                   oq1: i32x4,
                   mask: i32x4|
     -> (i32x4, i32x4, i32x4, i32x4) {
        let ps1 = op1 - BIAS;
        let ps0 = op0 - BIAS;
        let qs0 = oq0 - BIAS;
        let qs1 = oq1 - BIAS;
        let t_hev = i32x4::splat(token, th as i32);
        let hev = iabs(op1, op0).simd_gt(t_hev) | iabs(oq1, oq0).simd_gt(t_hev);

        let mut filter = scc(ps1 - qs1) & hev;
        filter = scc(filter + (qs0 - ps0) * 3) & mask;
        let filter1 = scc(filter + 4).shr_arithmetic_const::<3>();
        let filter2 = scc(filter + 3).shr_arithmetic_const::<3>();
        let n_oq0 = scc(qs0 - filter1) + BIAS;
        let n_op0 = scc(ps0 + filter2) + BIAS;
        let f = ((filter1 + 1).shr_arithmetic_const::<1>()) & hev.not();
        let n_oq1 = scc(qs1 - f) + BIAS;
        let n_op1 = scc(ps1 + f) + BIAS;
        (n_op1, n_op0, n_oq0, n_oq1)
    };

    let fmask2 = |p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4| -> i32x4 {
        (iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    let fmask6 = |p2: i32x4, p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4, q2: i32x4| -> i32x4 {
        (iabs(p2, p1).simd_gt(l)
            | iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | iabs(q2, q1).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    let fmask8 = |p3: i32x4,
                  p2: i32x4,
                  p1: i32x4,
                  p0: i32x4,
                  q0: i32x4,
                  q1: i32x4,
                  q2: i32x4,
                  q3: i32x4|
     -> i32x4 {
        (iabs(p3, p2).simd_gt(l)
            | iabs(p2, p1).simd_gt(l)
            | iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | iabs(q2, q1).simd_gt(l)
            | iabs(q3, q2).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    let flat3 = |p2: i32x4, p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4, q2: i32x4| -> i32x4 {
        (iabs(p1, p0).simd_gt(ft)
            | iabs(q1, q0).simd_gt(ft)
            | iabs(p2, p0).simd_gt(ft)
            | iabs(q2, q0).simd_gt(ft))
        .not()
    };
    let flat4 = |p3: i32x4,
                 p2: i32x4,
                 p1: i32x4,
                 p0: i32x4,
                 q0: i32x4,
                 q1: i32x4,
                 q2: i32x4,
                 q3: i32x4|
     -> i32x4 {
        (iabs(p1, p0).simd_gt(ft)
            | iabs(q1, q0).simd_gt(ft)
            | iabs(p2, p0).simd_gt(ft)
            | iabs(q2, q0).simd_gt(ft)
            | iabs(p3, p0).simd_gt(ft)
            | iabs(q3, q0).simd_gt(ft))
        .not()
    };

    macro_rules! store {
        ($($k:expr => $v:expr),+ $(,)?) => {{
            $(
                let a = ($v).to_array();
                buf[(c + ($k) * ts) as usize] = a[0] as u8;
                buf[(c + step + ($k) * ts) as usize] = a[1] as u8;
                buf[(c + 2 * step + ($k) * ts) as usize] = a[2] as u8;
                buf[(c + 3 * step + ($k) * ts) as usize] = a[3] as u8;
            )+
        }};
    }

    // ---- fast-path addressing (byte-identical; only the load/store SHAPE
    // changes, the i32x4 filter arithmetic is untouched) -------------------
    //
    // The walk always calls with one of two layouts (`loopfilter::horizontal`
    // ts=pitch/step=1, `loopfilter::vertical` ts=1/step=pitch):
    //  * `hfast` (horizontal edge): the 4 lane positions are CONTIGUOUS bytes,
    //    so each tap is one `[u8; 4]` load / store (LLVM: movd + pmovzxbd)
    //    instead of 4 strided scalar accesses. Gated `ts >= 4` so distinct
    //    taps' 4-byte runs cannot alias (they are `ts` apart).
    //  * `vfast` (vertical edge): each lane row's taps are CONTIGUOUS, so the
    //    4 rows stage into fixed `[u8; W]` windows (one bounds check per row,
    //    const-index extracts) and store back as whole rows. Gated
    //    `step >= W` so the 4 row windows cannot overlap. Writing back a
    //    window's untouched columns rewrites their just-staged (current)
    //    values — byte-identical in the sequential walk.
    // Anything else (never produced by the walk, possible in a synthetic
    // harness) takes the original strided-gather path unchanged.
    let z4 = i32x4::splat(token, 0);
    macro_rules! load_taps {
        ($t:ident, $rows:ident, $kmin:expr, $vfast:expr, $hfast:expr) => {
            if $vfast {
                for (r, row) in $rows.iter_mut().enumerate() {
                    let s = (c + r as isize * step + $kmin) as usize;
                    let w = row.len();
                    *row = buf[s..s + w].try_into().unwrap();
                }
                for (i, tv) in $t.iter_mut().enumerate() {
                    *tv = i32x4::from_array(
                        token,
                        [
                            $rows[0][i] as i32,
                            $rows[1][i] as i32,
                            $rows[2][i] as i32,
                            $rows[3][i] as i32,
                        ],
                    );
                }
            } else if $hfast {
                for (i, tv) in $t.iter_mut().enumerate() {
                    let s = (c + (i as isize + $kmin) * ts) as usize;
                    let b: [u8; 4] = buf[s..s + 4].try_into().unwrap();
                    *tv = i32x4::from_array(
                        token,
                        [b[0] as i32, b[1] as i32, b[2] as i32, b[3] as i32],
                    );
                }
            } else {
                for (i, tv) in $t.iter_mut().enumerate() {
                    *tv = load(i as isize + $kmin);
                }
            }
        };
    }
    macro_rules! store_taps {
        ($out:expr, $rows:ident, $kmin:expr, $col0:expr, $vfast:expr, $hfast:expr,
         $($fb:tt)+) => {
            if $vfast {
                for (j, v) in $out.iter().enumerate() {
                    let a = v.to_array();
                    $rows[0][$col0 + j] = a[0] as u8;
                    $rows[1][$col0 + j] = a[1] as u8;
                    $rows[2][$col0 + j] = a[2] as u8;
                    $rows[3][$col0 + j] = a[3] as u8;
                }
                for (r, row) in $rows.iter().enumerate() {
                    let s = (c + r as isize * step + $kmin) as usize;
                    buf[s..s + row.len()].copy_from_slice(row);
                }
            } else if $hfast {
                for (j, v) in $out.iter().enumerate() {
                    let a = v.to_array();
                    let s = (c + ($col0 as isize + j as isize + $kmin) * ts) as usize;
                    buf[s..s + 4]
                        .copy_from_slice(&[a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8]);
                }
            } else {
                store!($($fb)+);
            }
        };
    }

    match width {
        4 => {
            let mut rows = [[0u8; 4]; 4];
            let vfast = ts == 1 && step >= 4;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 4];
            load_taps!(t, rows, -2, vfast, hfast);
            let [op1, op0, oq0, oq1] = t;
            let mask = fmask2(op1, op0, oq0, oq1);
            let (n1, n0, m0, m1) = filter4(op1, op0, oq0, oq1, mask);
            let out = [n1, n0, m0, m1];
            store_taps!(out, rows, -2, 0, vfast, hfast, -2 => n1, -1 => n0, 0 => m0, 1 => m1);
        }
        6 => {
            let mut rows = [[0u8; 6]; 4];
            let vfast = ts == 1 && step >= 6;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 6];
            load_taps!(t, rows, -3, vfast, hfast);
            let [p2, p1, p0, q0, q1, q2] = t;
            let mask = fmask6(p2, p1, p0, q0, q1, q2);
            let flat = flat3(p2, p1, p0, q0, q1, q2);
            let use_wide = flat & mask;
            let w_p1 = rpo3(p2 * 3 + p1 * 2 + p0 * 2 + q0);
            let w_p0 = rpo3(p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1);
            let w_q0 = rpo3(p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2);
            let w_q1 = rpo3(p0 + q0 * 2 + q1 * 2 + q2 * 3);
            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            let o_p1 = i32x4::blend(use_wide, w_p1, f_p1);
            let o_p0 = i32x4::blend(use_wide, w_p0, f_p0);
            let o_q0 = i32x4::blend(use_wide, w_q0, f_q0);
            let o_q1 = i32x4::blend(use_wide, w_q1, f_q1);
            let out = [o_p1, o_p0, o_q0, o_q1];
            store_taps!(out, rows, -3, 1, vfast, hfast,
                -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1);
        }
        8 => {
            let mut rows = [[0u8; 8]; 4];
            let vfast = ts == 1 && step >= 8;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 8];
            load_taps!(t, rows, -4, vfast, hfast);
            let [p3, p2, p1, p0, q0, q1, q2, q3] = t;
            let mask = fmask8(p3, p2, p1, p0, q0, q1, q2, q3);
            let flat = flat4(p3, p2, p1, p0, q0, q1, q2, q3);
            let use_wide = flat & mask;
            let w_p2 = rpo3(p3 * 3 + p2 * 2 + p1 + p0 + q0);
            let w_p1 = rpo3(p3 * 2 + p2 + p1 * 2 + p0 + q0 + q1);
            let w_p0 = rpo3(p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2);
            let w_q0 = rpo3(p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3);
            let w_q1 = rpo3(p1 + p0 + q0 + q1 * 2 + q2 + q3 * 2);
            let w_q2 = rpo3(p0 + q0 + q1 + q2 * 2 + q3 * 3);
            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            let o_p2 = i32x4::blend(use_wide, w_p2, p2);
            let o_p1 = i32x4::blend(use_wide, w_p1, f_p1);
            let o_p0 = i32x4::blend(use_wide, w_p0, f_p0);
            let o_q0 = i32x4::blend(use_wide, w_q0, f_q0);
            let o_q1 = i32x4::blend(use_wide, w_q1, f_q1);
            let o_q2 = i32x4::blend(use_wide, w_q2, q2);
            let out = [o_p2, o_p1, o_p0, o_q0, o_q1, o_q2];
            store_taps!(out, rows, -4, 1, vfast, hfast,
                -3 => o_p2, -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1, 2 => o_q2);
        }
        14 => {
            let mut rows = [[0u8; 14]; 4];
            let vfast = ts == 1 && step >= 14;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 14];
            load_taps!(t, rows, -7, vfast, hfast);
            let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = t;

            let mask = fmask8(p3, p2, p1, p0, q0, q1, q2, q3);
            let flat = flat4(p3, p2, p1, p0, q0, q1, q2, q3);
            let flat2 = flat4(p6, p5, p4, p0, q0, q4, q5, q6);
            let use8 = flat & mask;
            let use14 = flat2 & use8;

            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            let w8_p2 = rpo3(p3 * 3 + p2 * 2 + p1 + p0 + q0);
            let w8_p1 = rpo3(p3 * 2 + p2 + p1 * 2 + p0 + q0 + q1);
            let w8_p0 = rpo3(p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2);
            let w8_q0 = rpo3(p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3);
            let w8_q1 = rpo3(p1 + p0 + q0 + q1 * 2 + q2 + q3 * 2);
            let w8_q2 = rpo3(p0 + q0 + q1 + q2 * 2 + q3 * 3);
            let w14_p5 = rpo4(p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0);
            let w14_p4 = rpo4(p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1);
            let w14_p3 = rpo4(p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2);
            let w14_p2 =
                rpo4(p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3);
            let w14_p1 =
                rpo4(p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4);
            let w14_p0 =
                rpo4(p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5);
            let w14_q0 =
                rpo4(p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6);
            let w14_q1 =
                rpo4(p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 * 2);
            let w14_q2 = rpo4(p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 * 3);
            let w14_q3 = rpo4(p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 * 4);
            let w14_q4 = rpo4(p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 * 5);
            let w14_q5 = rpo4(p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 7);

            let o_p5 = i32x4::blend(use14, w14_p5, p5);
            let o_p4 = i32x4::blend(use14, w14_p4, p4);
            let o_p3 = i32x4::blend(use14, w14_p3, p3);
            let o_p2 = i32x4::blend(use14, w14_p2, i32x4::blend(use8, w8_p2, p2));
            let o_p1 = i32x4::blend(use14, w14_p1, i32x4::blend(use8, w8_p1, f_p1));
            let o_p0 = i32x4::blend(use14, w14_p0, i32x4::blend(use8, w8_p0, f_p0));
            let o_q0 = i32x4::blend(use14, w14_q0, i32x4::blend(use8, w8_q0, f_q0));
            let o_q1 = i32x4::blend(use14, w14_q1, i32x4::blend(use8, w8_q1, f_q1));
            let o_q2 = i32x4::blend(use14, w14_q2, i32x4::blend(use8, w8_q2, q2));
            let o_q3 = i32x4::blend(use14, w14_q3, q3);
            let o_q4 = i32x4::blend(use14, w14_q4, q4);
            let o_q5 = i32x4::blend(use14, w14_q5, q5);
            let out = [
                o_p5, o_p4, o_p3, o_p2, o_p1, o_p0, o_q0, o_q1, o_q2, o_q3, o_q4, o_q5,
            ];
            store_taps!(out, rows, -7, 1, vfast, hfast,
                -6 => o_p5, -5 => o_p4, -4 => o_p3, -3 => o_p2, -2 => o_p1, -1 => o_p0,
                0 => o_q0, 1 => o_q1, 2 => o_q2, 3 => o_q3, 4 => o_q4, 5 => o_q5,
            );
        }
        _ => crate::loopfilter::lpf_scalar(width, buf, center, ts, step, bl, li, th),
    }
}
