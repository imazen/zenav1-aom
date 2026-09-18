//! SIMD deblock loop-filter kernels — bit-identical to the highbd
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

/// Dispatch entry for `nseg` adjacent 4-position deblock edge segments (highbd
/// `u16` path). `ts` = tap stride, `step` = position advance; the axis is
/// encoded by the caller ([`crate::loopfilter::highbd::horizontal`] /
/// [`crate::loopfilter::highbd::vertical`]). `nseg` > 1 is C's
/// `_dual`/`_quad` batching (`av1_loopfilter.c` `use_filter_type`): the
/// segments are positions `center + s*4*step` for `s in 0..nseg`, all sharing
/// this call's limits — C applies the first segment's `lfthr` to the whole
/// batch, which the geometry criterion (same prediction block) guarantees
/// equal. Per-segment arithmetic is unchanged, so batching cannot move a
/// pixel versus `nseg` single calls.
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
    nseg: usize,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        lpf_impl(width, buf, center, ts, step, bl, li, th, bd, nseg),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the untouched highbd transcription, verbatim, per segment.
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
    nseg: usize,
) {
    for s in 0..nseg {
        let cs = (center as isize + s as isize * 4 * step) as usize;
        crate::loopfilter::highbd::lpf_scalar(width, buf, cs, ts, step, bl, li, th, bd);
    }
}

#[magetypes(define(i32x4), neon, wasm128, -scalar)]
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
    nseg: usize,
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

    let c0 = center as isize;
    // Gather tap `k` (offset k*ts from segment center `c`) across the 4 edge
    // positions (offset l*step, l in 0..4) into one i32x4 lane vector.
    // Defined per segment so `c` binds the loop-local shadow.
    macro_rules! load {
        ($c:expr, $k:expr) => {{
            let c = $c;
            let k = $k;
            i32x4::from_array(
                token,
                [
                    buf[(c + k * ts) as usize] as i32,
                    buf[(c + step + k * ts) as usize] as i32,
                    buf[(c + 2 * step + k * ts) as usize] as i32,
                    buf[(c + 3 * step + k * ts) as usize] as i32,
                ],
            )
        }};
    }

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
    // (immutable) `load` closure's borrow has ended by this point. `c` is taken
    // as an argument (like `load!`): a free `c` would resolve with def-site
    // hygiene under the `magetypes` tier expansion and not see the loop local.
    macro_rules! store {
        ($c:expr, $($k:expr => $v:expr),+ $(,)?) => {{
            let c = $c;
            $(
                let a = ($v).to_array();
                buf[(c + ($k) * ts) as usize] = a[0] as u16;
                buf[(c + step + ($k) * ts) as usize] = a[1] as u16;
                buf[(c + 2 * step + ($k) * ts) as usize] = a[2] as u16;
                buf[(c + 3 * step + ($k) * ts) as usize] = a[3] as u16;
            )+
        }};
    }

    for s in 0..nseg {
        let c = c0 + s as isize * 4 * step;
        match width {
        4 => {
            // taps p1(-2) p0(-1) q0(0) q1(1)
            let op1 = load!(c, -2);
            let op0 = load!(c, -1);
            let oq0 = load!(c, 0);
            let oq1 = load!(c, 1);
            let mask = fmask2(op1, op0, oq0, oq1);
            let (n1, n0, m0, m1) = filter4(op1, op0, oq0, oq1, mask);
            store!(c, -2 => n1, -1 => n0, 0 => m0, 1 => m1);
        }
        6 => {
            // taps p2(-3) p1(-2) p0(-1) q0(0) q1(1) q2(2)
            let p2 = load!(c, -3);
            let p1 = load!(c, -2);
            let p0 = load!(c, -1);
            let q0 = load!(c, 0);
            let q1 = load!(c, 1);
            let q2 = load!(c, 2);
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
            store!(c, -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1);
        }
        8 => {
            // taps p3(-4) p2(-3) p1(-2) p0(-1) q0(0) q1(1) q2(2) q3(3)
            let p3 = load!(c, -4);
            let p2 = load!(c, -3);
            let p1 = load!(c, -2);
            let p0 = load!(c, -1);
            let q0 = load!(c, 0);
            let q1 = load!(c, 1);
            let q2 = load!(c, 2);
            let q3 = load!(c, 3);
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
            store!(c, -3 => o_p2, -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1, 2 => o_q2);
        }
        14 => {
            // taps p6(-7)..p0(-1), q0(0)..q6(6)
            let p6 = load!(c, -7);
            let p5 = load!(c, -6);
            let p4 = load!(c, -5);
            let p3 = load!(c, -4);
            let p2 = load!(c, -3);
            let p1 = load!(c, -2);
            let p0 = load!(c, -1);
            let q0 = load!(c, 0);
            let q1 = load!(c, 1);
            let q2 = load!(c, 2);
            let q3 = load!(c, 3);
            let q4 = load!(c, 4);
            let q5 = load!(c, 5);
            let q6 = load!(c, 6);

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
                c, -6 => o_p5, -5 => o_p4, -4 => o_p3, -3 => o_p2, -2 => o_p1, -1 => o_p0,
                0 => o_q0, 1 => o_q1, 2 => o_q2, 3 => o_q3, 4 => o_q4, 5 => o_q5,
            );
        }
        _ => crate::loopfilter::highbd::lpf_scalar(width, buf, c as usize, ts, step, bl, li, th, bd),
        }
    }
}

// ---- v3: mirror of the REAL dispatched x86 kernels ---------------------------
//
// `aom_highbd_lpf_{horizontal,vertical}_{4,6,8,14}_sse2` from
// `aom_dsp/x86/highbd_loopfilter_sse2.c` + `lpf_common_sse2.h` — the kernels a
// libaom x86-64 build actually runs (the `_avx2` names are thin wrappers that
// only widen the *dual* variants; the single-edge internals are this SSE2
// code). The magetypes `i32x4` kernel above keeps the SIMD tier = one lane per
// edge position and gathers every tap scalar-wise (~577 Ir/call); this mirror
// follows C's own structure instead:
//
// * contiguous row loads — one `loadu_si128`/`loadu_si64` per row on the
//   vertical axis plus an unpack-tree transpose, versus 7–14 per-lane scalar
//   gathers;
// * taps packed `pq[i] = [p_i (4 pos) | q_i (4 pos)]` so the mask/flat/filter4
//   math evaluates BOTH sides of the edge per instruction;
// * all arithmetic in `u16`/`i16` lanes — the wide sums peak at
//   `16*4095 + 8 < 2^16` at bd 12, and `scc`'s `±(128<<sh)` bound keeps every
//   signed intermediate inside `i16`, so C's `_mm_adds`/`_mm_subs`/`_mm_srai`
//   sequence is exact (the same argument the `i32x4` tier documents lane-by-
//   lane — here it is C's own choice, so it is exact by construction).
//
// Bit-exactness vs the scalar core on the reachable domain is proven by
// `lpf_simd_diff.rs` (every tier vs the scalar transcription over the u16
// sample domain) and `hbd_lpf_sse2_diff.rs` (this body vs the real exported
// `aom_highbd_lpf_*_sse2` symbols over adversarial taps). The span check below
// keeps this panic-free on any `&mut [u16]` window: C reads full 8-u16 rows
// (taps -8..+7 for width 14, wider than the ±7 the scalar core touches), so
// out-of-span falls back to the scalar transcription rather than slice-index.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn lpf_impl_v3(
    _t: archmage::X64V3Token,
    width: u32,
    buf: &mut [u16],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
    nseg: usize,
) {
    use archmage::intrinsics::x86_64::*;

    let c = center as isize;
    let len = buf.len() as isize;

    // [lo, hi) element span this call reads and writes, per axis/width — the
    // batch covers `nseg` segments at `c + s*4*step`, so the position extent
    // grows from 4 to 4*nseg positions.
    let last_pos = (4 * nseg - 1) as isize;
    let (lo, hi) = if step == 1 {
        let (kmin, kmax) = match width {
            4 => (-2isize, 1isize),
            6 => (-3, 2),
            8 => (-4, 3),
            _ => (-7, 6), // 14
        };
        (c + kmin * ts, c + kmax * ts + 4 * nseg as isize)
    } else {
        match width {
            4 => (c - 2, c + last_pos * step + 2),
            6 => (c - 3, c + last_pos * step + 5),
            8 => (c - 4, c + last_pos * step + 4),
            _ => (c - 8, c + last_pos * step + 8), // 14
        }
    };
    if !matches!(width, 4 | 6 | 8 | 14) || lo < 0 || hi > len {
        for s in 0..nseg {
            let cs = (c + s as isize * 4 * step) as usize;
            crate::loopfilter::highbd::lpf_scalar(width, buf, cs, ts, step, bl, li, th, bd);
        }
        return;
    }

    macro_rules! ldl4 {
        ($o:expr) => {{
            let o = ($o) as usize;
            let a: &[u16; 4] = buf[o..o + 4].try_into().unwrap();
            _mm_loadu_si64(a)
        }};
    }
    macro_rules! ldu8 {
        ($o:expr) => {{
            let o = ($o) as usize;
            let a: &[u16; 8] = buf[o..o + 8].try_into().unwrap();
            _mm_loadu_si128(a)
        }};
    }
    macro_rules! stl4 {
        ($o:expr, $v:expr) => {{
            let o = ($o) as usize;
            let a: &mut [u16; 4] = (&mut buf[o..o + 4]).try_into().unwrap();
            _mm_storeu_si64(a, $v)
        }};
    }
    macro_rules! stu8 {
        ($o:expr, $v:expr) => {{
            let o = ($o) as usize;
            let a: &mut [u16; 8] = (&mut buf[o..o + 8]).try_into().unwrap();
            _mm_storeu_si128(a, $v)
        }};
    }

    // get_limit — the u8 threshold vectors broadcast and shifted by bd-8, the
    // same values the scalar core's `* << (bd - 8)` produces.
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi16(1);
    let ffff = _mm_cmpeq_epi16(one, one);
    let shift = _mm_cvtsi32_si128(bd - 8);
    let blimit = _mm_sll_epi16(_mm_set1_epi16(bl as i16), shift);
    let limit = _mm_sll_epi16(_mm_set1_epi16(li as i16), shift);
    let thresh = _mm_sll_epi16(_mm_set1_epi16(th as i16), shift);
    let t80 = _mm_set1_epi16((1i32 << (bd - 1)) as i16);
    let flat_th = _mm_sll_epi16(one, shift);
    let pmax = _mm_subs_epi16(
        _mm_subs_epi16(_mm_sll_epi16(one, _mm_cvtsi32_si128(bd)), one),
        t80,
    );
    let pmin = _mm_subs_epi16(zero, t80);
    let t3t4 = _mm_set_epi16(3, 3, 3, 3, 4, 4, 4, 4);
    let four = _mm_set1_epi16(4);
    let eight = _mm_set1_epi16(8);

    let absd = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
    let pclamp = |v: __m128i| _mm_max_epi16(_mm_min_epi16(v, pmax), pmin);
    let any_set = |v: __m128i| _mm_movemask_epi8(_mm_cmpeq_epi16(v, zero)) != 0xffff;

    // highbd_hev_filter_mask_x_sse2 → (p1p0, q1q0, abs_p1p0, hev, mask)
    let hev_fmask =
        |pq: &[__m128i], x: usize| -> (__m128i, __m128i, __m128i, __m128i, __m128i) {
            let p1p0 = _mm_unpacklo_epi64(pq[0], pq[1]);
            let q1q0 = _mm_unpackhi_epi64(pq[0], pq[1]);
            let a01 = absd(p1p0, q1q0);
            let a_p0q0 = _mm_unpacklo_epi64(_mm_adds_epu16(a01, a01), zero);
            let a_p1q1 = _mm_srli_epi16::<1>(_mm_srli_si128::<8>(a01));
            let mut max = _mm_subs_epu16(_mm_adds_epu16(a_p0q0, a_p1q1), blimit);
            max = _mm_xor_si128(_mm_cmpeq_epi16(max, zero), ffff);
            max = _mm_and_si128(max, _mm_adds_epu16(limit, one));
            let abs_p1p0 = absd(pq[0], pq[1]);
            let a_q1q0 = _mm_srli_si128::<8>(abs_p1p0);
            let max01 = _mm_max_epi16(abs_p1p0, a_q1q0);
            let h = _mm_subs_epu16(max01, thresh);
            let hev0 = _mm_xor_si128(_mm_cmpeq_epi16(h, zero), ffff);
            let hev = _mm_unpacklo_epi64(hev0, hev0);
            max = _mm_max_epi16(max, max01);
            for i in 2..x {
                max = _mm_max_epi16(max, absd(pq[i], pq[i - 1]));
            }
            max = _mm_max_epi16(max, _mm_srli_si128::<8>(max));
            max = _mm_subs_epu16(max, limit);
            let mask = _mm_cmpeq_epi16(max, zero);
            (p1p0, q1q0, abs_p1p0, hev, mask)
        };

    // highbd_filter4_sse2: in [p0|p1],[q0|q1] → out ([p0'|p1'], [q0'|q1'])
    let filter4 =
        |p1p0: __m128i, q1q0: __m128i, hev: __m128i, mask: __m128i| -> (__m128i, __m128i) {
            let ps = _mm_subs_epi16(p1p0, t80);
            let qs = _mm_subs_epi16(q1q0, t80);
            let work = pclamp(_mm_subs_epi16(ps, qs));
            let mut filt = _mm_and_si128(_mm_srli_si128::<8>(work), hev);
            filt = _mm_subs_epi16(filt, work);
            filt = _mm_subs_epi16(filt, work);
            filt = _mm_subs_epi16(filt, work);
            filt = _mm_and_si128(pclamp(filt), mask);
            filt = _mm_unpacklo_epi64(filt, filt);
            let f2f1 = _mm_srai_epi16::<3>(pclamp(_mm_adds_epi16(filt, t3t4)));
            let mut f = _mm_unpacklo_epi64(f2f1, f2f1);
            f = _mm_srai_epi16::<1>(_mm_adds_epi16(f, one));
            f = _mm_andnot_si128(hev, f);
            let f2filt = _mm_unpackhi_epi64(f2f1, f);
            let f1filt = _mm_unpacklo_epi64(f2f1, f);
            let qs_out = _mm_adds_epi16(pclamp(_mm_subs_epi16(qs, f1filt)), t80);
            let ps_out = _mm_adds_epi16(pclamp(_mm_adds_epi16(ps, f2filt)), t80);
            (ps_out, qs_out)
        };

    // flat_mask_internal
    let flat_int = |pq: &[__m128i], start: usize, end: usize| -> __m128i {
        let mut max = _mm_max_epi16(absd(pq[start], pq[0]), absd(pq[start + 1], pq[0]));
        for i in (start + 2)..end {
            max = _mm_max_epi16(max, absd(pq[i], pq[0]));
        }
        max = _mm_max_epi16(max, _mm_srli_si128::<8>(max));
        _mm_cmpeq_epi16(_mm_subs_epu16(max, flat_th), zero)
    };

    // highbd_transpose4x8_8x4_{low,high}_sse2 / highbd_transpose8x8_low_sse2
    let t4x8_low = |x0: __m128i,
                    x1: __m128i,
                    x2: __m128i,
                    x3: __m128i|
     -> (__m128i, __m128i, __m128i, __m128i) {
        let w0 = _mm_unpacklo_epi16(x0, x1);
        let w1 = _mm_unpacklo_epi16(x2, x3);
        let ww0 = _mm_unpacklo_epi32(w0, w1);
        let ww1 = _mm_unpackhi_epi32(w0, w1);
        (
            _mm_unpacklo_epi64(ww0, zero),
            _mm_unpackhi_epi64(ww0, zero),
            _mm_unpacklo_epi64(ww1, zero),
            _mm_unpackhi_epi64(ww1, zero),
        )
    };
    let t4x8_high = |x0: __m128i,
                     x1: __m128i,
                     x2: __m128i,
                     x3: __m128i|
     -> (__m128i, __m128i, __m128i, __m128i) {
        let w0 = _mm_unpackhi_epi16(x0, x1);
        let w1 = _mm_unpackhi_epi16(x2, x3);
        let ww2 = _mm_unpacklo_epi32(w0, w1);
        let ww3 = _mm_unpackhi_epi32(w0, w1);
        (
            _mm_unpacklo_epi64(ww2, zero),
            _mm_unpackhi_epi64(ww2, zero),
            _mm_unpacklo_epi64(ww3, zero),
            _mm_unpackhi_epi64(ww3, zero),
        )
    };
    let t4x8 = |x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i| -> [__m128i; 8] {
        let (a, b, c, d) = t4x8_low(x0, x1, x2, x3);
        let (e, f, g, h) = t4x8_high(x0, x1, x2, x3);
        [a, b, c, d, e, f, g, h]
    };
    let t8x8_low = |x: [__m128i; 8]| -> (__m128i, __m128i, __m128i, __m128i) {
        let w0 = _mm_unpacklo_epi16(x[0], x[1]);
        let w1 = _mm_unpacklo_epi16(x[2], x[3]);
        let w2 = _mm_unpacklo_epi16(x[4], x[5]);
        let w3 = _mm_unpacklo_epi16(x[6], x[7]);
        let ww0 = _mm_unpacklo_epi32(w0, w1);
        let ww1 = _mm_unpacklo_epi32(w2, w3);
        let d0 = _mm_unpacklo_epi64(ww0, ww1);
        let d1 = _mm_unpackhi_epi64(ww0, ww1);
        let ww0 = _mm_unpackhi_epi32(w0, w1);
        let ww1 = _mm_unpackhi_epi32(w2, w3);
        let d2 = _mm_unpacklo_epi64(ww0, ww1);
        let d3 = _mm_unpackhi_epi64(ww0, ww1);
        (d0, d1, d2, d3)
    };

    // highbd_lpf_internal_4_sse2 → (p1p0_out, q1q0_out)
    let internal4 = |p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i| -> (__m128i, __m128i) {
        let pq = [_mm_unpacklo_epi64(p0, q0), _mm_unpacklo_epi64(p1, q1)];
        let (p1p0, q1q0, _a, hev, mask) = hev_fmask(&pq, 2);
        filter4(p1p0, q1q0, hev, mask)
    };

    // highbd_lpf_internal_6_sse2 → (p1p0_out, q1q0_out)
    #[allow(clippy::too_many_arguments)]
    let internal6 = |p2: __m128i,
                     p1: __m128i,
                     p0: __m128i,
                     q0: __m128i,
                     q1: __m128i,
                     q2: __m128i|
     -> (__m128i, __m128i) {
        let pq = [
            _mm_unpacklo_epi64(p0, q0),
            _mm_unpacklo_epi64(p1, q1),
            _mm_unpacklo_epi64(p2, q2),
        ];
        let (p1p0, q1q0, abs_p1p0, hev, mask) = hev_fmask(&pq, 3);
        let (ps1ps0, qs1qs0) = filter4(p1p0, q1q0, hev, mask);
        let mut flat = _mm_max_epi16(absd(pq[2], pq[0]), abs_p1p0);
        flat = _mm_max_epi16(flat, _mm_srli_si128::<8>(flat));
        flat = _mm_subs_epu16(flat, flat_th);
        flat = _mm_cmpeq_epi16(flat, zero);
        flat = _mm_and_si128(flat, mask);
        let flat = _mm_unpacklo_epi64(flat, flat);
        let (mut p1p0_out, mut q1q0_out) = (ps1ps0, qs1qs0);
        if any_set(flat) {
            let pq0x2_pq1 = _mm_add_epi16(_mm_add_epi16(pq[0], pq[0]), pq[1]);
            let pq1_pq2 = _mm_add_epi16(pq[1], pq[2]);
            let mut wa = _mm_add_epi16(_mm_add_epi16(pq0x2_pq1, four), pq1_pq2);
            let mut wb = _mm_add_epi16(_mm_add_epi16(pq[2], pq[2]), q0);
            wb = _mm_add_epi16(wa, wb);
            let wc = _mm_srli_si128::<8>(pq0x2_pq1);
            wa = _mm_add_epi16(wa, wc);
            let flat_p1p0 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(wa, wb));
            wa = _mm_sub_epi16(_mm_sub_epi16(wa, pq[2]), pq[1]);
            wb = _mm_srli_si128::<8>(pq1_pq2);
            wa = _mm_add_epi16(wa, wb);
            let wc = _mm_sub_epi16(_mm_sub_epi16(wa, pq[1]), pq[0]);
            let wb = _mm_add_epi16(_mm_add_epi16(q2, q2), wc);
            let flat_q0q1 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(wa, wb));
            q1q0_out = _mm_or_si128(
                _mm_andnot_si128(flat, q1q0_out),
                _mm_and_si128(flat, flat_q0q1),
            );
            p1p0_out = _mm_or_si128(
                _mm_andnot_si128(flat, p1p0_out),
                _mm_and_si128(flat, flat_p1p0),
            );
        }
        (p1p0_out, q1q0_out)
    };

    // highbd_lpf_internal_8_sse2 → (p1p0_out, q1q0_out, pq2_packed)
    #[allow(clippy::too_many_arguments)]
    let internal8 = |p3: __m128i,
                     q3: __m128i,
                     p2: __m128i,
                     q2: __m128i,
                     p1: __m128i,
                     q1: __m128i,
                     p0: __m128i,
                     q0: __m128i|
     -> (__m128i, __m128i, __m128i) {
        let pq = [
            _mm_unpacklo_epi64(p0, q0),
            _mm_unpacklo_epi64(p1, q1),
            _mm_unpacklo_epi64(p2, q2),
            _mm_unpacklo_epi64(p3, q3),
        ];
        let (p1p0, q1q0, abs_p1p0, hev, mask) = hev_fmask(&pq, 4);
        let (ps1ps0, qs1qs0) = filter4(p1p0, q1q0, hev, mask);
        let mut flat = _mm_max_epi16(absd(pq[2], pq[0]), absd(pq[3], pq[0]));
        flat = _mm_max_epi16(abs_p1p0, flat);
        flat = _mm_max_epi16(flat, _mm_srli_si128::<8>(flat));
        flat = _mm_subs_epu16(flat, flat_th);
        flat = _mm_cmpeq_epi16(flat, zero);
        flat = _mm_and_si128(flat, mask);
        let flat = _mm_unpacklo_epi64(flat, flat);
        let (mut p1p0_out, mut q1q0_out, mut pq2) = (ps1ps0, qs1qs0, pq[2]);
        if any_set(flat) {
            let mut wa = _mm_add_epi16(_mm_add_epi16(p3, p3), _mm_add_epi16(p2, p1));
            wa = _mm_add_epi16(_mm_add_epi16(wa, four), p0);
            let wc = _mm_add_epi16(_mm_add_epi16(q0, p2), p3);
            let wc = _mm_add_epi16(wa, wc); // op2 pre-shift
            let mut wb = _mm_add_epi16(_mm_add_epi16(q0, q1), p1);
            let sh0 = _mm_add_epi16(wa, wb); // op1 pre-shift
            wa = _mm_add_epi16(_mm_sub_epi16(wa, p3), q2);
            wb = _mm_add_epi16(_mm_sub_epi16(wb, p1), p0);
            let sh1 = _mm_add_epi16(wa, wb); // op0 pre-shift
            let flat_p1p0 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(sh1, sh0));
            wa = _mm_add_epi16(_mm_sub_epi16(wa, p3), q3);
            wb = _mm_add_epi16(_mm_sub_epi16(wb, p0), q0);
            let sh0 = _mm_add_epi16(wa, wb); // oq0 pre-shift
            wa = _mm_add_epi16(_mm_sub_epi16(wa, p2), q3);
            wb = _mm_add_epi16(_mm_sub_epi16(wb, q0), q1);
            let sh1 = _mm_add_epi16(wa, wb); // oq1 pre-shift
            let flat_q0q1 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(sh0, sh1));
            wa = _mm_add_epi16(_mm_sub_epi16(wa, p1), q3);
            wb = _mm_add_epi16(_mm_sub_epi16(wb, q1), q2);
            wa = _mm_add_epi16(wa, wb); // oq2 pre-shift
            let opq2 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(wc, wa));
            q1q0_out = _mm_or_si128(
                _mm_andnot_si128(flat, q1q0_out),
                _mm_and_si128(flat, flat_q0q1),
            );
            p1p0_out = _mm_or_si128(
                _mm_andnot_si128(flat, p1p0_out),
                _mm_and_si128(flat, flat_p1p0),
            );
            pq2 = _mm_or_si128(_mm_andnot_si128(flat, pq[2]), _mm_and_si128(flat, opq2));
        }
        (p1p0_out, q1q0_out, pq2)
    };

    // highbd_lpf_internal_14_sse2: p[i] = tap -(i+1), q[i] = tap +i → pq[0..5]
    let internal14 = |p: [__m128i; 7], q: [__m128i; 7]| -> [__m128i; 6] {
        let mut pq = [
            _mm_unpacklo_epi64(p[0], q[0]),
            _mm_unpacklo_epi64(p[1], q[1]),
            _mm_unpacklo_epi64(p[2], q[2]),
            _mm_unpacklo_epi64(p[3], q[3]),
            _mm_unpacklo_epi64(p[4], q[4]),
            _mm_unpacklo_epi64(p[5], q[5]),
            _mm_unpacklo_epi64(p[6], q[6]),
        ];
        let (p1p0, q1q0, _a, hev, mask) = hev_fmask(&pq, 4);
        let (ps0ps1, qs0qs1) = filter4(p1p0, q1q0, hev, mask);
        let mut flat = flat_int(&pq, 1, 4);
        let mut flat2 = flat_int(&pq, 4, 7);
        flat = _mm_and_si128(flat, mask);
        flat2 = _mm_and_si128(flat2, flat);
        let flat = _mm_unpacklo_epi64(flat, flat);
        let flat2 = _mm_unpacklo_epi64(flat2, flat2);
        if any_set(flat) {
            let mut sum_p = _mm_add_epi16(pq[5], _mm_add_epi16(pq[4], pq[3]));
            let mut sum_lp = _mm_add_epi16(pq[0], _mm_add_epi16(pq[2], pq[1]));
            sum_p = _mm_add_epi16(sum_p, sum_lp);
            let mut sum_lq = _mm_srli_si128::<8>(sum_lp);
            let mut sum_q = _mm_srli_si128::<8>(sum_p);
            let sum_p_0 = _mm_add_epi16(eight, _mm_add_epi16(sum_p, sum_q));
            sum_lp = _mm_add_epi16(four, _mm_add_epi16(sum_lp, sum_lq));
            let flat_p0 = _mm_add_epi16(sum_lp, _mm_add_epi16(pq[3], pq[0]));
            let flat_q0 = _mm_add_epi16(sum_lp, _mm_add_epi16(q[3], q[0]));
            let mut sum_p6 = _mm_add_epi16(pq[6], pq[6]);
            let mut sum_p3 = _mm_add_epi16(pq[3], pq[3]);
            sum_q = _mm_sub_epi16(sum_p_0, pq[5]);
            sum_p = _mm_sub_epi16(sum_p_0, q[5]);
            let work0_0 = _mm_add_epi16(_mm_add_epi16(pq[6], pq[0]), pq[1]);
            let work0_1 =
                _mm_add_epi16(sum_p6, _mm_add_epi16(pq[1], _mm_add_epi16(pq[2], pq[0])));
            sum_lq = _mm_sub_epi16(sum_lp, pq[2]);
            sum_lp = _mm_sub_epi16(sum_lp, q[2]);
            let mut work0 = _mm_add_epi16(sum_p3, pq[1]);
            let flat_p1 = _mm_add_epi16(sum_lp, work0);
            let flat_q1 = _mm_add_epi16(sum_lq, _mm_srli_si128::<8>(work0));
            let flat_pq0 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(flat_p0, flat_q0));
            let flat_pq1 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(flat_p1, flat_q1));
            sum_lp = _mm_sub_epi16(sum_lp, q[1]);
            sum_lq = _mm_sub_epi16(sum_lq, pq[1]);
            sum_p3 = _mm_add_epi16(sum_p3, pq[3]);
            work0 = _mm_add_epi16(sum_p3, pq[2]);
            let flat_p2 = _mm_add_epi16(sum_lp, work0);
            let flat_q2 = _mm_add_epi16(sum_lq, _mm_srli_si128::<8>(work0));
            let flat_pq2 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(flat_p2, flat_q2));
            let flat_pq = [flat_pq0, flat_pq1, flat_pq2];
            let flat2_mask = any_set(flat2);
            let mut flat2_pq = [zero; 6];
            if flat2_mask {
                let f2p0 = _mm_add_epi16(sum_p_0, _mm_add_epi16(work0_0, q[0]));
                let f2q0 = _mm_add_epi16(
                    sum_p_0,
                    _mm_add_epi16(_mm_srli_si128::<8>(work0_0), pq[0]),
                );
                let f2p1 = _mm_add_epi16(sum_p, work0_1);
                let f2q1 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0_1));
                flat2_pq[0] = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(f2p0, f2q0));
                flat2_pq[1] = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(f2p1, f2q1));
                sum_p = _mm_sub_epi16(sum_p, q[4]);
                sum_q = _mm_sub_epi16(sum_q, pq[4]);
                sum_p6 = _mm_add_epi16(sum_p6, pq[6]);
                work0 = _mm_add_epi16(sum_p6, _mm_add_epi16(pq[2], _mm_add_epi16(pq[3], pq[1])));
                let f2p2 = _mm_add_epi16(sum_p, work0);
                let f2q2 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
                flat2_pq[2] = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(f2p2, f2q2));
                sum_p6 = _mm_add_epi16(sum_p6, pq[6]);
                sum_p = _mm_sub_epi16(sum_p, q[3]);
                sum_q = _mm_sub_epi16(sum_q, pq[3]);
                work0 = _mm_add_epi16(sum_p6, _mm_add_epi16(pq[3], _mm_add_epi16(pq[4], pq[2])));
                let f2p3 = _mm_add_epi16(sum_p, work0);
                let f2q3 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
                flat2_pq[3] = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(f2p3, f2q3));
                sum_p6 = _mm_add_epi16(sum_p6, pq[6]);
                sum_p = _mm_sub_epi16(sum_p, q[2]);
                sum_q = _mm_sub_epi16(sum_q, pq[2]);
                work0 = _mm_add_epi16(sum_p6, _mm_add_epi16(pq[4], _mm_add_epi16(pq[5], pq[3])));
                let f2p4 = _mm_add_epi16(sum_p, work0);
                let f2q4 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
                flat2_pq[4] = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(f2p4, f2q4));
                sum_p6 = _mm_add_epi16(sum_p6, pq[6]);
                sum_p = _mm_sub_epi16(sum_p, q[1]);
                sum_q = _mm_sub_epi16(sum_q, pq[1]);
                work0 = _mm_add_epi16(sum_p6, _mm_add_epi16(pq[5], _mm_add_epi16(pq[6], pq[4])));
                let f2p5 = _mm_add_epi16(sum_p, work0);
                let f2q5 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
                flat2_pq[5] = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(f2p5, f2q5));
            }
            pq[0] = _mm_unpacklo_epi64(ps0ps1, qs0qs1);
            pq[1] = _mm_unpackhi_epi64(ps0ps1, qs0qs1);
            for i in 0..3 {
                pq[i] = _mm_or_si128(
                    _mm_andnot_si128(flat, pq[i]),
                    _mm_and_si128(flat, flat_pq[i]),
                );
            }
            if flat2_mask {
                for i in 0..6 {
                    pq[i] = _mm_or_si128(
                        _mm_andnot_si128(flat2, pq[i]),
                        _mm_and_si128(flat2, flat2_pq[i]),
                    );
                }
            }
        } else {
            pq[0] = _mm_unpacklo_epi64(ps0ps1, qs0qs1);
            pq[1] = _mm_unpackhi_epi64(ps0ps1, qs0qs1);
        }
        [pq[0], pq[1], pq[2], pq[3], pq[4], pq[5]]
    };

    if step == 1 {
        // aom_highbd_lpf_horizontal_*_sse2 — positions contiguous along the row.
        for s in 0..nseg {
            let c = c + s as isize * 4;
            match width {
            4 => {
                let p1 = ldl4!(c - 2 * ts);
                let p0 = ldl4!(c - ts);
                let q0 = ldl4!(c);
                let q1 = ldl4!(c + ts);
                let (p1p0, q1q0) = internal4(p1, p0, q0, q1);
                stl4!(c - 2 * ts, _mm_srli_si128::<8>(p1p0));
                stl4!(c - ts, p1p0);
                stl4!(c, q1q0);
                stl4!(c + ts, _mm_srli_si128::<8>(q1q0));
            }
            6 => {
                let p2 = ldl4!(c - 3 * ts);
                let p1 = ldl4!(c - 2 * ts);
                let p0 = ldl4!(c - ts);
                let q0 = ldl4!(c);
                let q1 = ldl4!(c + ts);
                let q2 = ldl4!(c + 2 * ts);
                let (p1p0, q1q0) = internal6(p2, p1, p0, q0, q1, q2);
                stl4!(c - 2 * ts, _mm_srli_si128::<8>(p1p0));
                stl4!(c - ts, p1p0);
                stl4!(c, q1q0);
                stl4!(c + ts, _mm_srli_si128::<8>(q1q0));
            }
            8 => {
                let p3 = ldl4!(c - 4 * ts);
                let p2 = ldl4!(c - 3 * ts);
                let p1 = ldl4!(c - 2 * ts);
                let p0 = ldl4!(c - ts);
                let q0 = ldl4!(c);
                let q1 = ldl4!(c + ts);
                let q2 = ldl4!(c + 2 * ts);
                let q3 = ldl4!(c + 3 * ts);
                let (p1p0, q1q0, pq2) = internal8(p3, q3, p2, q2, p1, q1, p0, q0);
                stl4!(c - 3 * ts, pq2);
                stl4!(c - 2 * ts, _mm_srli_si128::<8>(p1p0));
                stl4!(c - ts, p1p0);
                stl4!(c, q1q0);
                stl4!(c + ts, _mm_srli_si128::<8>(q1q0));
                stl4!(c + 2 * ts, _mm_srli_si128::<8>(pq2));
            }
            _ => {
                // 14: p[i] = tap -(i+1), q[i] = tap +i
                let p = [
                    ldl4!(c - ts),
                    ldl4!(c - 2 * ts),
                    ldl4!(c - 3 * ts),
                    ldl4!(c - 4 * ts),
                    ldl4!(c - 5 * ts),
                    ldl4!(c - 6 * ts),
                    ldl4!(c - 7 * ts),
                ];
                let q = [
                    ldl4!(c),
                    ldl4!(c + ts),
                    ldl4!(c + 2 * ts),
                    ldl4!(c + 3 * ts),
                    ldl4!(c + 4 * ts),
                    ldl4!(c + 5 * ts),
                    ldl4!(c + 6 * ts),
                ];
                let pq = internal14(p, q);
                for (i, v) in pq.iter().enumerate() {
                    let i = i as isize;
                    stl4!(c - (i + 1) * ts, *v);
                    stl4!(c + i * ts, _mm_srli_si128::<8>(*v));
                }
            }
        }
        }
        return;
    }

    // aom_highbd_lpf_vertical_*_sse2 — taps contiguous; 4 positions = 4 rows.
    for s in 0..nseg {
        let c = c + s as isize * 4 * step;
        match width {
        4 => {
            let x0 = ldl4!(c - 2);
            let x1 = ldl4!(c - 2 + step);
            let x2 = ldl4!(c - 2 + 2 * step);
            let x3 = ldl4!(c - 2 + 3 * step);
            // d[j] = tap -2 + j → (p1, p0, q0, q1)
            let (d0, d1, d2, d3) = t4x8_low(x0, x1, x2, x3);
            let (p1p0, q1q0) = internal4(d0, d1, d2, d3);
            let p1v = _mm_srli_si128::<8>(p1p0);
            let q1v = _mm_srli_si128::<8>(q1q0);
            let (o0, o1, o2, o3) = t4x8_low(p1v, p1p0, q1q0, q1v);
            stl4!(c - 2, o0);
            stl4!(c - 2 + step, o1);
            stl4!(c - 2 + 2 * step, o2);
            stl4!(c - 2 + 3 * step, o3);
        }
        6 => {
            let r0 = ldu8!(c - 3);
            let r1 = ldu8!(c - 3 + step);
            let r2 = ldu8!(c - 3 + 2 * step);
            let r3 = ldu8!(c - 3 + 3 * step);
            // d[j] = tap -3 + j → (p2, p1, p0, q0, q1, q2, ..)
            let d = t4x8(r0, r1, r2, r3);
            let (p1p0, q1q0) = internal6(d[0], d[1], d[2], d[3], d[4], d[5]);
            let p0v = _mm_srli_si128::<8>(p1p0);
            let q0v = _mm_srli_si128::<8>(q1q0);
            let (o0, o1, o2, o3) = t4x8_low(p0v, p1p0, q1q0, q0v);
            stl4!(c - 2, o0);
            stl4!(c - 2 + step, o1);
            stl4!(c - 2 + 2 * step, o2);
            stl4!(c - 2 + 3 * step, o3);
        }
        8 => {
            let r0 = ldu8!(c - 4);
            let r1 = ldu8!(c - 4 + step);
            let r2 = ldu8!(c - 4 + 2 * step);
            let r3 = ldu8!(c - 4 + 3 * step);
            // d[j] = tap -4 + j → (p3, p2, p1, p0, q0, q1, q2, q3)
            let d = t4x8(r0, r1, r2, r3);
            let (p1p0, q1q0, pq2) = internal8(d[0], d[7], d[1], d[6], d[2], d[5], d[3], d[4]);
            let p0v = _mm_srli_si128::<8>(p1p0);
            let q0v = _mm_srli_si128::<8>(q1q0);
            let q2v = _mm_srli_si128::<8>(pq2);
            let (o0, o1, o2, o3) = t8x8_low([d[0], pq2, p0v, p1p0, q1q0, q0v, q2v, d[7]]);
            stu8!(c - 4, o0);
            stu8!(c - 4 + step, o1);
            stu8!(c - 4 + 2 * step, o2);
            stu8!(c - 4 + 3 * step, o3);
        }
        _ => {
            // 14: two 4x8 transposes; d[j] on the p side = tap -8 + j
            let pd = t4x8(
                ldu8!(c - 8),
                ldu8!(c - 8 + step),
                ldu8!(c - 8 + 2 * step),
                ldu8!(c - 8 + 3 * step),
            );
            let qd = t4x8(ldu8!(c), ldu8!(c + step), ldu8!(c + 2 * step), ldu8!(c + 3 * step));
            let p = [pd[7], pd[6], pd[5], pd[4], pd[3], pd[2], pd[1]];
            let q = [qd[0], qd[1], qd[2], qd[3], qd[4], qd[5], qd[6]];
            let pq = internal14(p, q);
            // p side: [p7, p6, p5', p4', p3', p2', p1', p0'] per row
            let (o0, o1, o2, o3) =
                t8x8_low([pd[0], pd[1], pq[5], pq[4], pq[3], pq[2], pq[1], pq[0]]);
            stu8!(c - 8, o0);
            stu8!(c - 8 + step, o1);
            stu8!(c - 8 + 2 * step, o2);
            stu8!(c - 8 + 3 * step, o3);
            // q side: [q0'..q5', q6, q7]
            let qv = [
                _mm_srli_si128::<8>(pq[0]),
                _mm_srli_si128::<8>(pq[1]),
                _mm_srli_si128::<8>(pq[2]),
                _mm_srli_si128::<8>(pq[3]),
                _mm_srli_si128::<8>(pq[4]),
                _mm_srli_si128::<8>(pq[5]),
                qd[6],
                qd[7],
            ];
            let (z0, z1, z2, z3) = t8x8_low(qv);
            stu8!(c, z0);
            stu8!(c + step, z1);
            stu8!(c + 2 * step, z2);
            stu8!(c + 3 * step, z3);
        }
    }
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
    nseg: usize,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        lpf_impl_u8(width, buf, center, ts, step, bl, li, th, nseg),
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
    nseg: usize,
) {
    for s in 0..nseg {
        crate::loopfilter::lpf_scalar(
            width,
            buf,
            center.wrapping_add_signed(s as isize * 4 * step),
            ts,
            step,
            bl,
            li,
            th,
        );
    }
}

#[magetypes(define(i32x4), neon, wasm128, -scalar)]
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
    nseg: usize,
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

    // `c` is taken as an argument (like the u16 twin's `store!`): a free `c`
    // would resolve with def-site hygiene under the `magetypes` tier
    // expansion and not see the loop local.
    macro_rules! store {
        ($c:expr, $($k:expr => $v:expr),+ $(,)?) => {{
            let c = $c;
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
        ($c:expr, $load:expr, $t:ident, $rows:ident, $kmin:expr, $vfast:expr, $hfast:expr) => {{
            let c = $c;
            let load = $load;
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
        }};
    }
    macro_rules! store_taps {
        ($c:expr, $out:expr, $rows:ident, $kmin:expr, $col0:expr, $vfast:expr, $hfast:expr,
         $($fb:tt)+) => {{
            let c = $c;
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
                store!(c, $($fb)+);
            }
        }};
    }

    for s in 0..nseg {
        let c = center as isize + s as isize * 4 * step;
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
        match width {
        4 => {
            let mut rows = [[0u8; 4]; 4];
            let vfast = ts == 1 && step >= 4;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 4];
            load_taps!(c, load, t, rows, -2, vfast, hfast);
            let [op1, op0, oq0, oq1] = t;
            let mask = fmask2(op1, op0, oq0, oq1);
            let (n1, n0, m0, m1) = filter4(op1, op0, oq0, oq1, mask);
            let out = [n1, n0, m0, m1];
            store_taps!(c, out, rows, -2, 0, vfast, hfast, -2 => n1, -1 => n0, 0 => m0, 1 => m1);
        }
        6 => {
            let mut rows = [[0u8; 6]; 4];
            let vfast = ts == 1 && step >= 6;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 6];
            load_taps!(c, load, t, rows, -3, vfast, hfast);
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
            store_taps!(c, out, rows, -3, 1, vfast, hfast,
                -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1);
        }
        8 => {
            let mut rows = [[0u8; 8]; 4];
            let vfast = ts == 1 && step >= 8;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 8];
            load_taps!(c, load, t, rows, -4, vfast, hfast);
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
            store_taps!(c, out, rows, -4, 1, vfast, hfast,
                -3 => o_p2, -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1, 2 => o_q2);
        }
        14 => {
            let mut rows = [[0u8; 14]; 4];
            let vfast = ts == 1 && step >= 14;
            let hfast = step == 1 && ts >= 4;
            let mut t = [z4; 14];
            load_taps!(c, load, t, rows, -7, vfast, hfast);
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
            store_taps!(c, out, rows, -7, 1, vfast, hfast,
                -6 => o_p5, -5 => o_p4, -4 => o_p3, -3 => o_p2, -2 => o_p1, -1 => o_p0,
                0 => o_q0, 1 => o_q1, 2 => o_q2, 3 => o_q3, 4 => o_q4, 5 => o_q5,
            );
        }
        _ => crate::loopfilter::lpf_scalar(width, buf, c as usize, ts, step, bl, li, th),
        }
    }
}

// ---- v3 lowbd: mirror of the REAL dispatched u8 kernels ----------------------
//
// `aom_lpf_{horizontal,vertical}_{4,6,8,14}_sse2` from
// `aom_dsp/x86/loopfilter_sse2.c` + `lpf_common_sse2.h` — the kernels a libaom
// x86-64 build runs at bd8. The magetypes `i32x4` tier above gathers every tap
// scalar-wise and widens to i32 lanes (~390 Ir/call); this mirror keeps C's own
// structure instead: packed u8 lanes, saturating byte arithmetic, and the
// unpack-tree transposes on the vertical axis. All math is the C sequence
// verbatim, so it is exact by construction (gated by `lpf_simd_diff` and
// `loopfilter_lowbd_diff` against the REAL exported `aom_lpf_*_c`).
//
// Span check mirrors the hbd v3 kernel: compute the [lo, hi) byte window the
// kernel's loads/stores touch and fall back to the scalar transcription when
// it is not fully inside `buf` — the scalar's own bounds-checked indexing then
// preserves the panic contract (it panics iff the taps are genuinely OOB).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn lpf_impl_u8_v3(
    _t: archmage::X64V3Token,
    width: u32,
    buf: &mut [u8],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    nseg: usize,
) {
    use archmage::intrinsics::x86_64::*;

    let c = center as isize;
    let len = buf.len() as isize;
    // The batch covers `nseg` 4-position segments at `c + s*4*step`.
    let dseg = (nseg.saturating_sub(1) * 4) as isize * step;

    // [lo, hi) byte span this call reads and writes.
    let (lo, hi) = if step == 1 {
        // horizontal: 4-byte position rows at c + k*ts, k in [-w/2, w/2-1].
        let (kmin, kmax) = match width {
            4 => (-2isize, 1isize),
            6 => (-3, 2),
            8 => (-4, 3),
            _ => (-7, 6), // 14
        };
        (
            c + (kmin * ts).min(kmax * ts) + dseg.min(0),
            c + (kmin * ts).max(kmax * ts) + 4 + dseg.max(0),
        )
    } else if ts == 1 {
        // vertical: `span`-byte rows at c - w/2 + k*step, k in 0..4. Widths 4/6
        // read a couple bytes past the last needed tap (C's 8-byte loads);
        // the fallback below covers the plane edge.
        let (half, span) = match width {
            4 => (2isize, 4isize),
            6 => (3, 8),
            8 => (4, 8),
            _ => (8, 16), // 14
        };
        (
            c - half + (3 * step).min(0) + dseg.min(0),
            c - half + span + (3 * step).max(0) + dseg.max(0),
        )
    } else {
        (0, -1) // neither layout — force the scalar arm
    };
    if !matches!(width, 4 | 6 | 8 | 14) || lo < 0 || hi > len {
        for s in 0..nseg {
            crate::loopfilter::lpf_scalar(
                width,
                buf,
                center.wrapping_add_signed(s as isize * 4 * step),
                ts,
                step,
                bl,
                li,
                th,
            );
        }
        return;
    }

    macro_rules! ld4 {
        ($o:expr) => {{
            let o = ($o) as usize;
            let a: &[u8; 4] = buf[o..o + 4].try_into().unwrap();
            _mm_loadu_si32(a)
        }};
    }
    macro_rules! ld8 {
        ($o:expr) => {{
            let o = ($o) as usize;
            let a: &[u8; 8] = buf[o..o + 8].try_into().unwrap();
            _mm_loadu_si64(a)
        }};
    }
    macro_rules! ld16 {
        ($o:expr) => {{
            let o = ($o) as usize;
            let a: &[u8; 16] = buf[o..o + 16].try_into().unwrap();
            _mm_loadu_si128(a)
        }};
    }
    macro_rules! st4 {
        ($o:expr, $v:expr) => {{
            let o = ($o) as usize;
            let a: &mut [u8; 4] = (&mut buf[o..o + 4]).try_into().unwrap();
            _mm_storeu_si32(a, $v)
        }};
    }
    macro_rules! st8 {
        ($o:expr, $v:expr) => {{
            let o = ($o) as usize;
            let a: &mut [u8; 8] = (&mut buf[o..o + 8]).try_into().unwrap();
            _mm_storeu_si64(a, $v)
        }};
    }
    macro_rules! st16 {
        ($o:expr, $v:expr) => {{
            let o = ($o) as usize;
            let a: &mut [u8; 16] = (&mut buf[o..o + 16]).try_into().unwrap();
            _mm_storeu_si128(a, $v)
        }};
    }

    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let fe = _mm_set1_epi8(-2i8); // 0xfe
    let ff = _mm_cmpeq_epi8(fe, fe);
    let t80 = _mm_set1_epi8(i8::MIN); // 0x80
    let t3t4 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 4, 4, 4);
    let four = _mm_set1_epi16(4);

    let absd = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    let any_set = |v: __m128i| _mm_movemask_epi8(_mm_cmpeq_epi8(v, zero)) != 0xffff;

    // filter4_sse2: in [p0 pos|p1 pos],[q0 pos|q1 pos] → out (ps1ps0, qs1qs0).
    let filter4 =
        |p1p0: __m128i, q1q0: __m128i, hev: __m128i, mask: __m128i| -> (__m128i, __m128i) {
            let ps_work = _mm_xor_si128(p1p0, t80);
            let qs_work = _mm_xor_si128(q1q0, t80);
            let work = _mm_subs_epi8(ps_work, qs_work);
            let mut filter = _mm_and_si128(_mm_srli_si128::<4>(work), hev);
            filter = _mm_subs_epi8(filter, work);
            filter = _mm_subs_epi8(filter, work);
            filter = _mm_subs_epi8(filter, work);
            filter = _mm_and_si128(filter, mask);
            filter = _mm_unpacklo_epi32(filter, filter);
            let mut f2f1 = _mm_adds_epi8(filter, t3t4);
            f2f1 = _mm_unpacklo_epi8(f2f1, f2f1);
            f2f1 = _mm_srai_epi16::<11>(f2f1);
            f2f1 = _mm_packs_epi16(f2f1, f2f1);
            filter = _mm_subs_epi8(f2f1, ff);
            filter = _mm_unpacklo_epi8(filter, filter);
            filter = _mm_srai_epi16::<9>(filter);
            filter = _mm_packs_epi16(filter, filter);
            filter = _mm_andnot_si128(hev, filter);
            filter = _mm_unpacklo_epi32(filter, filter);
            f2f1 = _mm_unpacklo_epi32(f2f1, filter);
            let hev1 = _mm_srli_si128::<8>(f2f1);
            let qs1qs0 = _mm_xor_si128(_mm_subs_epi8(qs_work, f2f1), t80);
            let ps1ps0 = _mm_xor_si128(_mm_adds_epi8(ps_work, hev1), t80);
            (ps1ps0, qs1qs0)
        };

    // transpose4x8_8x4_low_sse2 / transpose4x8_8x4_sse2 / transpose8x8_low_sse2
    let t4x8_low = |x0: __m128i,
                    x1: __m128i,
                    x2: __m128i,
                    x3: __m128i|
     -> (__m128i, __m128i, __m128i, __m128i) {
        let w0 = _mm_unpacklo_epi8(x0, x1);
        let w1 = _mm_unpacklo_epi8(x2, x3);
        let d0 = _mm_unpacklo_epi16(w0, w1);
        (
            d0,
            _mm_srli_si128::<4>(d0),
            _mm_srli_si128::<8>(d0),
            _mm_srli_si128::<12>(d0),
        )
    };
    let t4x8 = |x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i| -> [__m128i; 8] {
        let w0 = _mm_unpacklo_epi8(x0, x1);
        let w1 = _mm_unpacklo_epi8(x2, x3);
        let ww0 = _mm_unpacklo_epi16(w0, w1);
        let ww1 = _mm_unpackhi_epi16(w0, w1);
        [
            ww0,
            _mm_srli_si128::<4>(ww0),
            _mm_srli_si128::<8>(ww0),
            _mm_srli_si128::<12>(ww0),
            ww1,
            _mm_srli_si128::<4>(ww1),
            _mm_srli_si128::<8>(ww1),
            _mm_srli_si128::<12>(ww1),
        ]
    };
    let t8x8_low = |x: [__m128i; 8]| -> (__m128i, __m128i, __m128i, __m128i) {
        let w0 = _mm_unpacklo_epi8(x[0], x[1]);
        let w1 = _mm_unpacklo_epi8(x[2], x[3]);
        let w2 = _mm_unpacklo_epi8(x[4], x[5]);
        let w3 = _mm_unpacklo_epi8(x[6], x[7]);
        let w4 = _mm_unpacklo_epi16(w0, w1);
        let w5 = _mm_unpacklo_epi16(w2, w3);
        let d0 = _mm_unpacklo_epi32(w4, w5);
        let d2 = _mm_unpackhi_epi32(w4, w5);
        (d0, _mm_srli_si128::<8>(d0), d2, _mm_srli_si128::<8>(d2))
    };

    // transpose_pq_14_sse2: 4 rows of 16 taps → [q0p0 ..= q7p7] tap-pair regs.
    let tpq14 = |x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i| -> [__m128i; 8] {
        let w0 = _mm_unpacklo_epi8(x0, x1);
        let w1 = _mm_unpacklo_epi8(x2, x3);
        let w2 = _mm_unpackhi_epi8(x0, x1);
        let w3 = _mm_unpackhi_epi8(x2, x3);
        let ww0 = _mm_unpacklo_epi16(w0, w1);
        let ww1 = _mm_unpackhi_epi16(w0, w1);
        let ww2 = _mm_unpacklo_epi16(w2, w3);
        let ww3 = _mm_unpackhi_epi16(w2, w3);
        let q7p7 = _mm_unpacklo_epi32(ww0, _mm_srli_si128::<12>(ww3));
        let q6p6 = _mm_unpackhi_epi32(_mm_slli_si128::<4>(ww0), ww3);
        let q5p5 = _mm_unpackhi_epi32(ww0, _mm_slli_si128::<4>(ww3));
        let q4p4 = _mm_unpacklo_epi32(_mm_srli_si128::<12>(ww0), ww3);
        let q3p3 = _mm_unpacklo_epi32(ww1, _mm_srli_si128::<12>(ww2));
        let q2p2 = _mm_unpackhi_epi32(_mm_slli_si128::<4>(ww1), ww2);
        let q1p1 = _mm_unpackhi_epi32(ww1, _mm_slli_si128::<4>(ww2));
        let q0p0 = _mm_unpacklo_epi32(_mm_srli_si128::<12>(ww1), ww2);
        [q0p0, q1p1, q2p2, q3p3, q4p4, q5p5, q6p6, q7p7]
    };
    // transpose_pq_14_inv_sse2: x0 = q7p7 .. x7 = q0p0 → 4 rows.
    let tpq14_inv = |x: [__m128i; 8]| -> (__m128i, __m128i, __m128i, __m128i) {
        let w0 = _mm_unpacklo_epi8(x[0], x[1]);
        let w1 = _mm_unpacklo_epi8(x[2], x[3]);
        let w2 = _mm_unpacklo_epi8(x[4], x[5]);
        let w3 = _mm_unpacklo_epi8(x[6], x[7]);
        let w4 = _mm_unpacklo_epi16(w0, w1);
        let w5 = _mm_unpacklo_epi16(w2, w3);
        let d0 = _mm_unpacklo_epi32(w4, w5);
        let d2 = _mm_unpackhi_epi32(w4, w5);
        let w10 = _mm_unpacklo_epi8(x[7], x[6]);
        let w11 = _mm_unpacklo_epi8(x[5], x[4]);
        let w12 = _mm_unpacklo_epi8(x[3], x[2]);
        let w13 = _mm_unpacklo_epi8(x[1], x[0]);
        let w4 = _mm_unpackhi_epi16(w10, w11);
        let w5 = _mm_unpackhi_epi16(w12, w13);
        let d1 = _mm_unpacklo_epi32(w4, w5);
        let d3 = _mm_unpackhi_epi32(w4, w5);
        (
            _mm_unpacklo_epi64(d0, d1),
            _mm_unpackhi_epi64(d0, d1),
            _mm_unpacklo_epi64(d2, d3),
            _mm_unpackhi_epi64(d2, d3),
        )
    };

    for s in 0..nseg {
        let c = center as isize + s as isize * 4 * step;
        if step == 1 {
        // ---- horizontal: aom_lpf_horizontal_{4,6,8,14}_sse2 ------------------
        match width {
            4 => {
                // limit = unpacklo_epi32(loadl(blimit), loadl(limit)) —
                // [bl*4 | li*4 | bl*4 | li*4]; thresh = u16 lanes of `th`.
                let limit4 = _mm_unpacklo_epi32(_mm_set1_epi8(bl as i8), _mm_set1_epi8(li as i8));
                let thresh16 = _mm_set1_epi16(th as i16);

                let p1 = ld4!(c - 2 * ts);
                let p0 = ld4!(c - ts);
                let q0 = ld4!(c);
                let q1 = ld4!(c + ts);

                // lpf_internal_4_sse2
                let q1p1 = _mm_unpacklo_epi32(p1, q1);
                let q0p0 = _mm_unpacklo_epi32(p0, q0);
                let p1p0 = _mm_unpacklo_epi32(q0p0, q1p1);
                let q1q0 = _mm_srli_si128::<8>(p1p0);
                let mut flat = absd(q1p1, q0p0);
                let abs_p1q1p0q0 = absd(p1p0, q1q0);
                flat = _mm_max_epu8(flat, _mm_srli_si128::<4>(flat));
                let mut hev = _mm_unpacklo_epi8(flat, zero);
                hev = _mm_cmpgt_epi16(hev, thresh16);
                hev = _mm_packs_epi16(hev, hev);
                hev = _mm_unpacklo_epi32(hev, hev);
                let abs_p0q0 = _mm_adds_epu8(abs_p1q1p0q0, abs_p1q1p0q0);
                let mut abs_p1q1 = _mm_srli_si128::<4>(abs_p1q1p0q0);
                abs_p1q1 = _mm_unpacklo_epi8(abs_p1q1, abs_p1q1);
                abs_p1q1 = _mm_srli_epi16::<9>(abs_p1q1);
                abs_p1q1 = _mm_packs_epi16(abs_p1q1, abs_p1q1);
                let mut mask = _mm_adds_epu8(abs_p0q0, abs_p1q1);
                mask = _mm_unpacklo_epi32(mask, flat);
                mask = _mm_subs_epu8(mask, limit4);
                mask = _mm_cmpeq_epi8(mask, zero);
                mask = _mm_and_si128(mask, _mm_srli_si128::<4>(mask));
                let (ps1ps0, qs1qs0) = filter4(p1p0, q1q0, hev, mask);

                st4!(c - ts, ps1ps0);
                st4!(c - 2 * ts, _mm_srli_si128::<4>(ps1ps0));
                st4!(c, qs1qs0);
                st4!(c + ts, _mm_srli_si128::<4>(qs1qs0));
            }
            _ => {
                let blimit = _mm_set1_epi8(bl as i8);
                let limit = _mm_set1_epi8(li as i8);
                let thresh = _mm_set1_epi8(th as i8);
                match width {
                    6 => {
                        let p2 = ld4!(c - 3 * ts);
                        let p1 = ld4!(c - 2 * ts);
                        let p0 = ld4!(c - ts);
                        let q0 = ld4!(c);
                        let q1 = ld4!(c + ts);
                        let q2 = ld4!(c + 2 * ts);

                        // lpf_internal_6_sse2
                        let q2p2 = _mm_unpacklo_epi32(p2, q2);
                        let q1p1 = _mm_unpacklo_epi32(p1, q1);
                        let q0p0 = _mm_unpacklo_epi32(p0, q0);
                        let mut p1p0 = _mm_unpacklo_epi32(p0, p1);
                        let mut q1q0 = _mm_unpacklo_epi32(q0, q1);

                        let abs_p1p0 = absd(q1p1, q0p0);
                        let abs_q1q0 = _mm_srli_si128::<4>(abs_p1p0);
                        let mut abs_p0q0 = absd(p1p0, q1q0);
                        let mut abs_p1q1 = _mm_srli_si128::<4>(abs_p0q0);
                        let mut flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
                        let mut hev = _mm_subs_epu8(flat, thresh);
                        hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);
                        hev = _mm_unpacklo_epi32(hev, hev);
                        abs_p0q0 = _mm_adds_epu8(abs_p0q0, abs_p0q0);
                        abs_p1q1 = _mm_srli_epi16::<1>(_mm_and_si128(abs_p1q1, fe));
                        let mut mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
                        mask = _mm_unpacklo_epi32(mask, zero);
                        mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
                        mask = _mm_max_epu8(abs_p1p0, mask);
                        let work = absd(q2p2, q1p1);
                        mask = _mm_max_epu8(work, mask);
                        mask = _mm_max_epu8(mask, _mm_srli_si128::<4>(mask));
                        mask = _mm_subs_epu8(mask, limit);
                        mask = _mm_cmpeq_epi8(mask, zero);
                        let (ps, qs) = filter4(p1p0, q1q0, hev, mask);
                        p1p0 = ps;
                        q1q0 = qs;

                        flat = _mm_max_epu8(absd(q2p2, q0p0), abs_p1p0);
                        flat = _mm_max_epu8(flat, _mm_srli_si128::<4>(flat));
                        flat = _mm_subs_epu8(flat, one);
                        flat = _mm_cmpeq_epi8(flat, zero);
                        flat = _mm_and_si128(flat, mask);
                        flat = _mm_unpacklo_epi32(flat, flat);
                        flat = _mm_unpacklo_epi64(flat, flat);

                        if any_set(flat) {
                            // 5-tap filter16
                            let pq2_16 = _mm_unpacklo_epi8(q2p2, zero);
                            let pq1_16 = _mm_unpacklo_epi8(q1p1, zero);
                            let pq0_16 = _mm_unpacklo_epi8(q0p0, zero);
                            let q0_16 = _mm_srli_si128::<8>(pq0_16);
                            let q2_16 = _mm_srli_si128::<8>(pq2_16);
                            let pq0x2_pq1 = _mm_add_epi16(_mm_add_epi16(pq0_16, pq0_16), pq1_16);
                            let pq1_pq2 = _mm_add_epi16(pq1_16, pq2_16);
                            let mut workp_a =
                                _mm_add_epi16(_mm_add_epi16(pq0x2_pq1, four), pq1_pq2);
                            let mut workp_b = _mm_add_epi16(_mm_add_epi16(pq2_16, pq2_16), q0_16);
                            workp_b = _mm_add_epi16(workp_a, workp_b);
                            let workp_c = _mm_srli_si128::<8>(pq0x2_pq1);
                            workp_a = _mm_add_epi16(workp_a, workp_c);
                            workp_b = _mm_unpacklo_epi64(workp_a, workp_b);
                            workp_b = _mm_srli_epi16::<3>(workp_b);
                            let flat_p1p0 = _mm_packus_epi16(workp_b, workp_b);
                            workp_a = _mm_sub_epi16(_mm_sub_epi16(workp_a, pq2_16), pq1_16);
                            workp_b = _mm_srli_si128::<8>(pq1_pq2);
                            workp_a = _mm_add_epi16(workp_a, workp_b);
                            let workp_c = _mm_sub_epi16(_mm_sub_epi16(workp_a, pq1_16), pq0_16);
                            workp_b = _mm_add_epi16(q2_16, q2_16);
                            workp_b = _mm_add_epi16(workp_c, workp_b);
                            workp_a = _mm_unpacklo_epi64(workp_a, workp_b);
                            workp_a = _mm_srli_epi16::<3>(workp_a);
                            let flat_q0q1 = _mm_packus_epi16(workp_a, workp_a);
                            q1q0 = _mm_or_si128(
                                _mm_andnot_si128(flat, q1q0),
                                _mm_and_si128(flat, flat_q0q1),
                            );
                            p1p0 = _mm_or_si128(
                                _mm_andnot_si128(flat, p1p0),
                                _mm_and_si128(flat, flat_p1p0),
                            );
                        }

                        st4!(c - ts, p1p0);
                        st4!(c - 2 * ts, _mm_srli_si128::<4>(p1p0));
                        st4!(c, q1q0);
                        st4!(c + ts, _mm_srli_si128::<4>(q1q0));
                    }
                    8 => {
                        let p3 = ld4!(c - 4 * ts);
                        let p2 = ld4!(c - 3 * ts);
                        let p1 = ld4!(c - 2 * ts);
                        let p0 = ld4!(c - ts);
                        let q0 = ld4!(c);
                        let q1 = ld4!(c + ts);
                        let q2 = ld4!(c + 2 * ts);
                        let q3 = ld4!(c + 3 * ts);

                        // lpf_internal_8_sse2
                        let q3p3 = _mm_unpacklo_epi32(p3, q3);
                        let q2p2 = _mm_unpacklo_epi32(p2, q2);
                        let q1p1 = _mm_unpacklo_epi32(p1, q1);
                        let q0p0 = _mm_unpacklo_epi32(p0, q0);
                        let p1p0 = _mm_unpacklo_epi32(q0p0, q1p1);
                        let q1q0 = _mm_srli_si128::<8>(p1p0);

                        let abs_p1p0 = absd(q1p1, q0p0);
                        let abs_q1q0 = _mm_srli_si128::<4>(abs_p1p0);
                        let mut abs_p0q0 = absd(p1p0, q1q0);
                        let mut abs_p1q1 = _mm_srli_si128::<4>(abs_p0q0);
                        let mut flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
                        let mut hev = _mm_subs_epu8(flat, thresh);
                        hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);
                        hev = _mm_unpacklo_epi32(hev, hev);
                        abs_p0q0 = _mm_adds_epu8(abs_p0q0, abs_p0q0);
                        abs_p1q1 = _mm_srli_epi16::<1>(_mm_and_si128(abs_p1q1, fe));
                        let mut mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
                        mask = _mm_unpacklo_epi32(mask, zero);
                        mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
                        mask = _mm_max_epu8(abs_p1p0, mask);
                        let work = _mm_max_epu8(absd(q2p2, q1p1), absd(q3p3, q2p2));
                        mask = _mm_max_epu8(work, mask);
                        mask = _mm_max_epu8(mask, _mm_srli_si128::<4>(mask));
                        mask = _mm_subs_epu8(mask, limit);
                        mask = _mm_cmpeq_epi8(mask, zero);
                        let (mut p1p0_o, mut q1q0_o) = filter4(p1p0, q1q0, hev, mask);

                        flat = _mm_max_epu8(absd(q2p2, q0p0), absd(q3p3, q0p0));
                        flat = _mm_max_epu8(abs_p1p0, flat);
                        flat = _mm_max_epu8(flat, _mm_srli_si128::<4>(flat));
                        flat = _mm_subs_epu8(flat, one);
                        flat = _mm_cmpeq_epi8(flat, zero);
                        flat = _mm_and_si128(flat, mask);
                        flat = _mm_unpacklo_epi32(flat, flat);
                        flat = _mm_unpacklo_epi64(flat, flat);

                        let mut p2_o = p2;
                        let mut q2_o = q2;
                        if any_set(flat) {
                            // 7-tap filter16
                            let p2_16 = _mm_unpacklo_epi8(p2, zero);
                            let p1_16 = _mm_unpacklo_epi8(p1, zero);
                            let p0_16 = _mm_unpacklo_epi8(p0, zero);
                            let q0_16 = _mm_unpacklo_epi8(q0, zero);
                            let q1_16 = _mm_unpacklo_epi8(q1, zero);
                            let q2_16 = _mm_unpacklo_epi8(q2, zero);
                            let p3_16 = _mm_unpacklo_epi8(p3, zero);
                            let q3_16 = _mm_unpacklo_epi8(q3, zero);

                            let mut workp_a = _mm_add_epi16(
                                _mm_add_epi16(p3_16, p3_16),
                                _mm_add_epi16(p2_16, p1_16),
                            );
                            workp_a = _mm_add_epi16(_mm_add_epi16(workp_a, four), p0_16);
                            let mut workp_b = _mm_add_epi16(_mm_add_epi16(q0_16, p2_16), p3_16);
                            let workp_shft2 = _mm_add_epi16(workp_a, workp_b);
                            workp_b = _mm_add_epi16(_mm_add_epi16(q0_16, q1_16), p1_16);
                            let mut workp_c = _mm_add_epi16(workp_a, workp_b);
                            workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3_16), q2_16);
                            workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p1_16), p0_16);
                            let mut workp_d = _mm_add_epi16(workp_a, workp_b);
                            workp_c = _mm_unpacklo_epi64(workp_d, workp_c);
                            workp_c = _mm_srli_epi16::<3>(workp_c);
                            let flat_p1p0 = _mm_packus_epi16(workp_c, workp_c);
                            workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3_16), q3_16);
                            workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p0_16), q0_16);
                            workp_c = _mm_add_epi16(workp_a, workp_b);
                            workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p2_16), q3_16);
                            workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q0_16), q1_16);
                            workp_d = _mm_add_epi16(workp_a, workp_b);
                            workp_c = _mm_unpacklo_epi64(workp_c, workp_d);
                            workp_c = _mm_srli_epi16::<3>(workp_c);
                            let flat_q0q1 = _mm_packus_epi16(workp_c, workp_c);
                            workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p1_16), q3_16);
                            workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q1_16), q2_16);
                            let workp_shft1 = _mm_add_epi16(workp_a, workp_b);
                            workp_c = _mm_unpacklo_epi64(workp_shft2, workp_shft1);
                            workp_c = _mm_srli_epi16::<3>(workp_c);
                            let opq2 = _mm_packus_epi16(workp_c, workp_c);

                            p2_o = _mm_or_si128(
                                _mm_andnot_si128(flat, q2p2),
                                _mm_and_si128(flat, opq2),
                            );
                            q2_o = _mm_srli_si128::<4>(p2_o);
                            q1q0_o = _mm_or_si128(
                                _mm_andnot_si128(flat, q1q0_o),
                                _mm_and_si128(flat, flat_q0q1),
                            );
                            p1p0_o = _mm_or_si128(
                                _mm_andnot_si128(flat, p1p0_o),
                                _mm_and_si128(flat, flat_p1p0),
                            );
                        }

                        st4!(c - ts, p1p0_o);
                        st4!(c - 2 * ts, _mm_srli_si128::<4>(p1p0_o));
                        st4!(c, q1q0_o);
                        st4!(c + ts, _mm_srli_si128::<4>(q1q0_o));
                        st4!(c - 3 * ts, p2_o);
                        st4!(c + 2 * ts, q2_o);
                    }
                    _ => {
                        // 14
                        let q4p4 = _mm_unpacklo_epi32(ld4!(c - 5 * ts), ld4!(c + 4 * ts));
                        let q3p3 = _mm_unpacklo_epi32(ld4!(c - 4 * ts), ld4!(c + 3 * ts));
                        let q2p2 = _mm_unpacklo_epi32(ld4!(c - 3 * ts), ld4!(c + 2 * ts));
                        let q1p1 = _mm_unpacklo_epi32(ld4!(c - 2 * ts), ld4!(c + ts));
                        let q0p0 = _mm_unpacklo_epi32(ld4!(c - ts), ld4!(c));
                        let q5p5 = _mm_unpacklo_epi32(ld4!(c - 6 * ts), ld4!(c + 5 * ts));
                        let q6p6 = _mm_unpacklo_epi32(ld4!(c - 7 * ts), ld4!(c + 6 * ts));

                        let out = internal14_u8(
                            _t,
                            [q0p0, q1p1, q2p2, q3p3, q4p4, q5p5, q6p6],
                            blimit,
                            limit,
                            thresh,
                            filter4,
                        );
                        for (num, x) in out.iter().enumerate() {
                            let num = num as isize;
                            st4!(c - (num + 1) * ts, *x);
                            st4!(c + num * ts, _mm_srli_si128::<4>(*x));
                        }
                    }
                }
            }
        }
    } else {
        // ---- vertical: aom_lpf_vertical_{4,6,8,14}_sse2 --------------------
        match width {
            4 => {
                let limit4 = _mm_unpacklo_epi32(_mm_set1_epi8(bl as i8), _mm_set1_epi8(li as i8));
                let thresh16 = _mm_set1_epi16(th as i16);

                let x0 = ld4!(c - 2);
                let x1 = ld4!(c - 2 + step);
                let x2 = ld4!(c - 2 + 2 * step);
                let x3 = ld4!(c - 2 + 3 * step);
                let (p1, p0, q0, q1) = t4x8_low(x0, x1, x2, x3);

                // lpf_internal_4_sse2 (same body as the horizontal arm)
                let q1p1 = _mm_unpacklo_epi32(p1, q1);
                let q0p0 = _mm_unpacklo_epi32(p0, q0);
                let p1p0 = _mm_unpacklo_epi32(q0p0, q1p1);
                let q1q0 = _mm_srli_si128::<8>(p1p0);
                let mut flat = absd(q1p1, q0p0);
                let abs_p1q1p0q0 = absd(p1p0, q1q0);
                flat = _mm_max_epu8(flat, _mm_srli_si128::<4>(flat));
                let mut hev = _mm_unpacklo_epi8(flat, zero);
                hev = _mm_cmpgt_epi16(hev, thresh16);
                hev = _mm_packs_epi16(hev, hev);
                hev = _mm_unpacklo_epi32(hev, hev);
                let abs_p0q0 = _mm_adds_epu8(abs_p1q1p0q0, abs_p1q1p0q0);
                let mut abs_p1q1 = _mm_srli_si128::<4>(abs_p1q1p0q0);
                abs_p1q1 = _mm_unpacklo_epi8(abs_p1q1, abs_p1q1);
                abs_p1q1 = _mm_srli_epi16::<9>(abs_p1q1);
                abs_p1q1 = _mm_packs_epi16(abs_p1q1, abs_p1q1);
                let mut mask = _mm_adds_epu8(abs_p0q0, abs_p1q1);
                mask = _mm_unpacklo_epi32(mask, flat);
                mask = _mm_subs_epu8(mask, limit4);
                mask = _mm_cmpeq_epi8(mask, zero);
                mask = _mm_and_si128(mask, _mm_srli_si128::<4>(mask));
                let (ps1ps0, qs1qs0) = filter4(p1p0, q1q0, hev, mask);

                let p0_s = _mm_srli_si128::<4>(ps1ps0);
                let q0_s = _mm_srli_si128::<4>(qs1qs0);
                let (d0, d1, d2, d3) = t4x8_low(p0_s, ps1ps0, qs1qs0, q0_s);
                st4!(c - 2, d0);
                st4!(c - 2 + step, d1);
                st4!(c - 2 + 2 * step, d2);
                st4!(c - 2 + 3 * step, d3);
            }
            _ => {
                let blimit = _mm_set1_epi8(bl as i8);
                let limit = _mm_set1_epi8(li as i8);
                let thresh = _mm_set1_epi8(th as i8);
                match width {
                    6 => {
                        let x3 = ld8!(c - 3);
                        let x2 = ld8!(c - 3 + step);
                        let x1 = ld8!(c - 3 + 2 * step);
                        let x0 = ld8!(c - 3 + 3 * step);
                        let d = t4x8(x3, x2, x1, x0);
                        let (q1q0, p1p0) = internal6_u8(
                            _t,
                            [d[0], d[1], d[2], d[3], d[4], d[5]],
                            blimit,
                            limit,
                            thresh,
                            filter4,
                        );
                        let p0_s = _mm_srli_si128::<4>(p1p0);
                        let q0_s = _mm_srli_si128::<4>(q1q0);
                        let (d0, d1, d2, d3) = t4x8_low(p0_s, p1p0, q1q0, q0_s);
                        st4!(c - 2, d0);
                        st4!(c - 2 + step, d1);
                        st4!(c - 2 + 2 * step, d2);
                        st4!(c - 2 + 3 * step, d3);
                    }
                    8 => {
                        let x3 = ld8!(c - 4);
                        let x2 = ld8!(c - 4 + step);
                        let x1 = ld8!(c - 4 + 2 * step);
                        let x0 = ld8!(c - 4 + 3 * step);
                        let d = t4x8(x3, x2, x1, x0);
                        let (q1q0, p1p0, p2_o, q2_o) = internal8_u8(
                            _t,
                            [d[0], d[7], d[1], d[6], d[2], d[5], d[3], d[4]],
                            blimit,
                            limit,
                            thresh,
                            filter4,
                        );
                        let p0_s = _mm_srli_si128::<4>(p1p0);
                        let q0_s = _mm_srli_si128::<4>(q1q0);
                        let (nd0, nd1, nd2, nd3) =
                            t8x8_low([d[0], p2_o, p0_s, p1p0, q1q0, q0_s, q2_o, d[7]]);
                        st8!(c - 4, nd0);
                        st8!(c - 4 + step, nd1);
                        st8!(c - 4 + 2 * step, nd2);
                        st8!(c - 4 + 3 * step, nd3);
                    }
                    _ => {
                        // 14
                        let x6 = ld16!(c - 8);
                        let x5 = ld16!(c - 8 + step);
                        let x4 = ld16!(c - 8 + 2 * step);
                        let x3 = ld16!(c - 8 + 3 * step);
                        let mut pq = tpq14(x6, x5, x4, x3);
                        let out = internal14_u8(
                            _t,
                            [pq[0], pq[1], pq[2], pq[3], pq[4], pq[5], pq[6]],
                            blimit,
                            limit,
                            thresh,
                            filter4,
                        );
                        pq[..6].copy_from_slice(&out);
                        let (r0, r1, r2, r3) =
                            tpq14_inv([pq[7], pq[6], pq[5], pq[4], pq[3], pq[2], pq[1], pq[0]]);
                        st16!(c - 8, r0);
                        st16!(c - 8 + step, r1);
                        st16!(c - 8 + 2 * step, r2);
                        st16!(c - 8 + 3 * step, r3);
                    }
                }
            }
        }
        }
    }
}

// The shared lowbd internals (lpf_internal_{6,8,14}_sse2) are large enough that
// they live as sibling functions rather than closures so the four driver arms
// stay readable. `filter4` is passed in because it closes over the per-call
// constants (C inlines everything into one function scope).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn internal6_u8(
    _t: archmage::X64V3Token,
    d: [__m128i; 6],
    blimit: __m128i,
    limit: __m128i,
    thresh: __m128i,
    filter4: impl Fn(__m128i, __m128i, __m128i, __m128i) -> (__m128i, __m128i),
) -> (__m128i, __m128i) {
    use archmage::intrinsics::x86_64::*;
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let fe = _mm_set1_epi8(-2i8);
    let ff = _mm_cmpeq_epi8(fe, fe);
    let four = _mm_set1_epi16(4);
    let [p2, p1, p0, q0, q1, q2] = d;

    let q2p2 = _mm_unpacklo_epi32(p2, q2);
    let q1p1 = _mm_unpacklo_epi32(p1, q1);
    let q0p0 = _mm_unpacklo_epi32(p0, q0);
    let mut p1p0 = _mm_unpacklo_epi32(p0, p1);
    let mut q1q0 = _mm_unpacklo_epi32(q0, q1);

    let abs_p1p0 = _mm_or_si128(_mm_subs_epu8(q1p1, q0p0), _mm_subs_epu8(q0p0, q1p1));
    let abs_q1q0 = _mm_srli_si128::<4>(abs_p1p0);
    let mut abs_p0q0 = _mm_or_si128(_mm_subs_epu8(p1p0, q1q0), _mm_subs_epu8(q1q0, p1p0));
    let mut abs_p1q1 = _mm_srli_si128::<4>(abs_p0q0);
    let mut flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
    let mut hev = _mm_subs_epu8(flat, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);
    hev = _mm_unpacklo_epi32(hev, hev);
    abs_p0q0 = _mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16::<1>(_mm_and_si128(abs_p1q1, fe));
    let mut mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_unpacklo_epi32(mask, zero);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    mask = _mm_max_epu8(abs_p1p0, mask);
    let work = _mm_or_si128(_mm_subs_epu8(q2p2, q1p1), _mm_subs_epu8(q1p1, q2p2));
    mask = _mm_max_epu8(work, mask);
    mask = _mm_max_epu8(mask, _mm_srli_si128::<4>(mask));
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);
    let (ps, qs) = filter4(p1p0, q1q0, hev, mask);
    p1p0 = ps;
    q1q0 = qs;

    flat = _mm_max_epu8(
        _mm_or_si128(_mm_subs_epu8(q2p2, q0p0), _mm_subs_epu8(q0p0, q2p2)),
        abs_p1p0,
    );
    flat = _mm_max_epu8(flat, _mm_srli_si128::<4>(flat));
    flat = _mm_subs_epu8(flat, one);
    flat = _mm_cmpeq_epi8(flat, zero);
    flat = _mm_and_si128(flat, mask);
    flat = _mm_unpacklo_epi32(flat, flat);
    flat = _mm_unpacklo_epi64(flat, flat);

    if _mm_movemask_epi8(_mm_cmpeq_epi8(flat, zero)) != 0xffff {
        let pq2_16 = _mm_unpacklo_epi8(q2p2, zero);
        let pq1_16 = _mm_unpacklo_epi8(q1p1, zero);
        let pq0_16 = _mm_unpacklo_epi8(q0p0, zero);
        let q0_16 = _mm_srli_si128::<8>(pq0_16);
        let q2_16 = _mm_srli_si128::<8>(pq2_16);
        let pq0x2_pq1 = _mm_add_epi16(_mm_add_epi16(pq0_16, pq0_16), pq1_16);
        let pq1_pq2 = _mm_add_epi16(pq1_16, pq2_16);
        let mut workp_a = _mm_add_epi16(_mm_add_epi16(pq0x2_pq1, four), pq1_pq2);
        let mut workp_b = _mm_add_epi16(_mm_add_epi16(pq2_16, pq2_16), q0_16);
        workp_b = _mm_add_epi16(workp_a, workp_b);
        let workp_c = _mm_srli_si128::<8>(pq0x2_pq1);
        workp_a = _mm_add_epi16(workp_a, workp_c);
        workp_b = _mm_unpacklo_epi64(workp_a, workp_b);
        workp_b = _mm_srli_epi16::<3>(workp_b);
        let flat_p1p0 = _mm_packus_epi16(workp_b, workp_b);
        workp_a = _mm_sub_epi16(_mm_sub_epi16(workp_a, pq2_16), pq1_16);
        workp_b = _mm_srli_si128::<8>(pq1_pq2);
        workp_a = _mm_add_epi16(workp_a, workp_b);
        let workp_c = _mm_sub_epi16(_mm_sub_epi16(workp_a, pq1_16), pq0_16);
        workp_b = _mm_add_epi16(q2_16, q2_16);
        workp_b = _mm_add_epi16(workp_c, workp_b);
        workp_a = _mm_unpacklo_epi64(workp_a, workp_b);
        workp_a = _mm_srli_epi16::<3>(workp_a);
        let flat_q0q1 = _mm_packus_epi16(workp_a, workp_a);
        q1q0 = _mm_or_si128(_mm_andnot_si128(flat, q1q0), _mm_and_si128(flat, flat_q0q1));
        p1p0 = _mm_or_si128(_mm_andnot_si128(flat, p1p0), _mm_and_si128(flat, flat_p1p0));
    }
    (q1q0, p1p0)
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn internal8_u8(
    _t: archmage::X64V3Token,
    d: [__m128i; 8],
    blimit: __m128i,
    limit: __m128i,
    thresh: __m128i,
    filter4: impl Fn(__m128i, __m128i, __m128i, __m128i) -> (__m128i, __m128i),
) -> (__m128i, __m128i, __m128i, __m128i) {
    use archmage::intrinsics::x86_64::*;
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let fe = _mm_set1_epi8(-2i8);
    let ff = _mm_cmpeq_epi8(fe, fe);
    let four = _mm_set1_epi16(4);
    let [p3, q3, p2, q2, p1, q1, p0, q0] = d;

    let q3p3 = _mm_unpacklo_epi32(p3, q3);
    let q2p2 = _mm_unpacklo_epi32(p2, q2);
    let q1p1 = _mm_unpacklo_epi32(p1, q1);
    let q0p0 = _mm_unpacklo_epi32(p0, q0);
    let p1p0 = _mm_unpacklo_epi32(q0p0, q1p1);
    let q1q0 = _mm_srli_si128::<8>(p1p0);

    let abs_p1p0 = _mm_or_si128(_mm_subs_epu8(q1p1, q0p0), _mm_subs_epu8(q0p0, q1p1));
    let abs_q1q0 = _mm_srli_si128::<4>(abs_p1p0);
    let mut abs_p0q0 = _mm_or_si128(_mm_subs_epu8(p1p0, q1q0), _mm_subs_epu8(q1q0, p1p0));
    let mut abs_p1q1 = _mm_srli_si128::<4>(abs_p0q0);
    let mut flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
    let mut hev = _mm_subs_epu8(flat, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);
    hev = _mm_unpacklo_epi32(hev, hev);
    abs_p0q0 = _mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16::<1>(_mm_and_si128(abs_p1q1, fe));
    let mut mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_unpacklo_epi32(mask, zero);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    mask = _mm_max_epu8(abs_p1p0, mask);
    let work = _mm_max_epu8(
        _mm_or_si128(_mm_subs_epu8(q2p2, q1p1), _mm_subs_epu8(q1p1, q2p2)),
        _mm_or_si128(_mm_subs_epu8(q3p3, q2p2), _mm_subs_epu8(q2p2, q3p3)),
    );
    mask = _mm_max_epu8(work, mask);
    mask = _mm_max_epu8(mask, _mm_srli_si128::<4>(mask));
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);
    let (mut p1p0_o, mut q1q0_o) = filter4(p1p0, q1q0, hev, mask);

    flat = _mm_max_epu8(
        _mm_or_si128(_mm_subs_epu8(q2p2, q0p0), _mm_subs_epu8(q0p0, q2p2)),
        _mm_or_si128(_mm_subs_epu8(q3p3, q0p0), _mm_subs_epu8(q0p0, q3p3)),
    );
    flat = _mm_max_epu8(abs_p1p0, flat);
    flat = _mm_max_epu8(flat, _mm_srli_si128::<4>(flat));
    flat = _mm_subs_epu8(flat, one);
    flat = _mm_cmpeq_epi8(flat, zero);
    flat = _mm_and_si128(flat, mask);
    flat = _mm_unpacklo_epi32(flat, flat);
    flat = _mm_unpacklo_epi64(flat, flat);

    let mut p2_o = p2;
    let mut q2_o = q2;
    if _mm_movemask_epi8(_mm_cmpeq_epi8(flat, zero)) != 0xffff {
        let p2_16 = _mm_unpacklo_epi8(p2, zero);
        let p1_16 = _mm_unpacklo_epi8(p1, zero);
        let p0_16 = _mm_unpacklo_epi8(p0, zero);
        let q0_16 = _mm_unpacklo_epi8(q0, zero);
        let q1_16 = _mm_unpacklo_epi8(q1, zero);
        let q2_16 = _mm_unpacklo_epi8(q2, zero);
        let p3_16 = _mm_unpacklo_epi8(p3, zero);
        let q3_16 = _mm_unpacklo_epi8(q3, zero);
        let mut workp_a = _mm_add_epi16(_mm_add_epi16(p3_16, p3_16), _mm_add_epi16(p2_16, p1_16));
        workp_a = _mm_add_epi16(_mm_add_epi16(workp_a, four), p0_16);
        let mut workp_b = _mm_add_epi16(_mm_add_epi16(q0_16, p2_16), p3_16);
        let workp_shft2 = _mm_add_epi16(workp_a, workp_b);
        workp_b = _mm_add_epi16(_mm_add_epi16(q0_16, q1_16), p1_16);
        let mut workp_c = _mm_add_epi16(workp_a, workp_b);
        workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3_16), q2_16);
        workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p1_16), p0_16);
        let mut workp_d = _mm_add_epi16(workp_a, workp_b);
        workp_c = _mm_unpacklo_epi64(workp_d, workp_c);
        workp_c = _mm_srli_epi16::<3>(workp_c);
        let flat_p1p0 = _mm_packus_epi16(workp_c, workp_c);
        workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p3_16), q3_16);
        workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, p0_16), q0_16);
        workp_c = _mm_add_epi16(workp_a, workp_b);
        workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p2_16), q3_16);
        workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q0_16), q1_16);
        workp_d = _mm_add_epi16(workp_a, workp_b);
        workp_c = _mm_unpacklo_epi64(workp_c, workp_d);
        workp_c = _mm_srli_epi16::<3>(workp_c);
        let flat_q0q1 = _mm_packus_epi16(workp_c, workp_c);
        workp_a = _mm_add_epi16(_mm_sub_epi16(workp_a, p1_16), q3_16);
        workp_b = _mm_add_epi16(_mm_sub_epi16(workp_b, q1_16), q2_16);
        let workp_shft1 = _mm_add_epi16(workp_a, workp_b);
        workp_c = _mm_unpacklo_epi64(workp_shft2, workp_shft1);
        workp_c = _mm_srli_epi16::<3>(workp_c);
        let opq2 = _mm_packus_epi16(workp_c, workp_c);
        p2_o = _mm_or_si128(_mm_andnot_si128(flat, q2p2), _mm_and_si128(flat, opq2));
        q2_o = _mm_srli_si128::<4>(p2_o);
        q1q0_o = _mm_or_si128(
            _mm_andnot_si128(flat, q1q0_o),
            _mm_and_si128(flat, flat_q0q1),
        );
        p1p0_o = _mm_or_si128(
            _mm_andnot_si128(flat, p1p0_o),
            _mm_and_si128(flat, flat_p1p0),
        );
    }
    (q1q0_o, p1p0_o, p2_o, q2_o)
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn internal14_u8(
    _t: archmage::X64V3Token,
    pq: [__m128i; 7],
    blimit: __m128i,
    limit: __m128i,
    thresh: __m128i,
    filter4: impl Fn(__m128i, __m128i, __m128i, __m128i) -> (__m128i, __m128i),
) -> [__m128i; 6] {
    use archmage::intrinsics::x86_64::*;
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let fe = _mm_set1_epi8(-2i8);
    let ff = _mm_cmpeq_epi8(fe, fe);
    let four = _mm_set1_epi16(4);
    let eight = _mm_set1_epi16(8);
    let absd = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    let [q0p0, q1p1, q2p2, q3p3, q4p4, q5p5, q6p6] = pq;

    let p1p0 = _mm_unpacklo_epi32(q0p0, q1p1);
    let q1q0 = _mm_srli_si128::<8>(p1p0);

    let abs_p1p0 = absd(q1p1, q0p0);
    let abs_q1q0 = _mm_srli_si128::<4>(abs_p1p0);
    let mut abs_p0q0 = absd(p1p0, q1q0);
    let mut abs_p1q1 = _mm_srli_si128::<4>(abs_p0q0);
    let mut flat = _mm_max_epu8(abs_p1p0, abs_q1q0);
    let mut hev = _mm_subs_epu8(flat, thresh);
    hev = _mm_xor_si128(_mm_cmpeq_epi8(hev, zero), ff);
    hev = _mm_unpacklo_epi32(hev, hev);
    abs_p0q0 = _mm_adds_epu8(abs_p0q0, abs_p0q0);
    abs_p1q1 = _mm_srli_epi16::<1>(_mm_and_si128(abs_p1q1, fe));
    let mut mask = _mm_subs_epu8(_mm_adds_epu8(abs_p0q0, abs_p1q1), blimit);
    mask = _mm_unpacklo_epi32(mask, zero);
    mask = _mm_xor_si128(_mm_cmpeq_epi8(mask, zero), ff);
    mask = _mm_max_epu8(abs_p1p0, mask);
    let mut work = _mm_max_epu8(absd(q2p2, q1p1), absd(q3p3, q2p2));
    mask = _mm_max_epu8(work, mask);
    mask = _mm_max_epu8(mask, _mm_srli_si128::<4>(mask));
    mask = _mm_subs_epu8(mask, limit);
    mask = _mm_cmpeq_epi8(mask, zero);

    let (ps1ps0, qs1qs0) = filter4(p1p0, q1q0, hev, mask);
    let qs0ps0 = _mm_unpacklo_epi32(ps1ps0, qs1qs0);
    let qs1ps1 = _mm_srli_si128::<8>(qs0ps0);

    flat = _mm_max_epu8(absd(q2p2, q0p0), absd(q3p3, q0p0));
    flat = _mm_max_epu8(abs_p1p0, flat);
    flat = _mm_max_epu8(flat, _mm_srli_si128::<4>(flat));
    flat = _mm_subs_epu8(flat, one);
    flat = _mm_cmpeq_epi8(flat, zero);
    flat = _mm_and_si128(flat, mask);
    flat = _mm_unpacklo_epi32(flat, flat);
    flat = _mm_unpacklo_epi64(flat, flat);

    let mut out0 = qs0ps0;
    let mut out1 = qs1ps1;
    let mut out2 = q2p2;
    let mut out3 = q3p3;
    let mut out4 = q4p4;
    let mut out5 = q5p5;

    if _mm_movemask_epi8(_mm_cmpeq_epi8(flat, zero)) != 0xffff {
        let pq_16 = [
            _mm_unpacklo_epi8(q0p0, zero),
            _mm_unpacklo_epi8(q1p1, zero),
            _mm_unpacklo_epi8(q2p2, zero),
            _mm_unpacklo_epi8(q3p3, zero),
            _mm_unpacklo_epi8(q4p4, zero),
            _mm_unpacklo_epi8(q5p5, zero),
            _mm_unpacklo_epi8(q6p6, zero),
        ];
        let q0_16 = _mm_srli_si128::<8>(pq_16[0]);
        let q1_16 = _mm_srli_si128::<8>(pq_16[1]);
        let q2_16 = _mm_srli_si128::<8>(pq_16[2]);
        let q3_16 = _mm_srli_si128::<8>(pq_16[3]);
        let q4_16 = _mm_srli_si128::<8>(pq_16[4]);
        let q5_16 = _mm_srli_si128::<8>(pq_16[5]);

        let mut sum_p = _mm_add_epi16(pq_16[5], _mm_add_epi16(pq_16[4], pq_16[3]));
        let mut sum_lp = _mm_add_epi16(pq_16[0], _mm_add_epi16(pq_16[2], pq_16[1]));
        sum_p = _mm_add_epi16(sum_p, sum_lp);
        let mut sum_lq = _mm_srli_si128::<8>(sum_lp);
        let mut sum_q = _mm_srli_si128::<8>(sum_p);
        let sum_p_0 = _mm_add_epi16(eight, _mm_add_epi16(sum_p, sum_q));
        sum_lp = _mm_add_epi16(four, _mm_add_epi16(sum_lp, sum_lq));

        let flat_p0 = _mm_add_epi16(sum_lp, _mm_add_epi16(pq_16[3], pq_16[0]));
        let flat_q0 = _mm_add_epi16(sum_lp, _mm_add_epi16(q3_16, q0_16));

        let mut sum_p6 = _mm_add_epi16(pq_16[6], pq_16[6]);
        let mut sum_p3 = _mm_add_epi16(pq_16[3], pq_16[3]);

        sum_q = _mm_sub_epi16(sum_p_0, pq_16[5]);
        let mut sum_p = _mm_sub_epi16(sum_p_0, q5_16);

        let work0_0 = _mm_add_epi16(_mm_add_epi16(pq_16[6], pq_16[0]), pq_16[1]);
        let work0_1 = _mm_add_epi16(
            sum_p6,
            _mm_add_epi16(pq_16[1], _mm_add_epi16(pq_16[2], pq_16[0])),
        );

        sum_lq = _mm_sub_epi16(sum_lp, pq_16[2]);
        sum_lp = _mm_sub_epi16(sum_lp, q2_16);

        let mut work0 = _mm_add_epi16(sum_p3, pq_16[1]);
        let flat_p1 = _mm_add_epi16(sum_lp, work0);
        let flat_q1 = _mm_add_epi16(sum_lq, _mm_srli_si128::<8>(work0));
        let mut flat_pq0 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(flat_p0, flat_q0));
        let mut flat_pq1 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(flat_p1, flat_q1));
        flat_pq0 = _mm_packus_epi16(flat_pq0, flat_pq0);
        flat_pq1 = _mm_packus_epi16(flat_pq1, flat_pq1);

        sum_lp = _mm_sub_epi16(sum_lp, q1_16);
        sum_lq = _mm_sub_epi16(sum_lq, pq_16[1]);
        sum_p3 = _mm_add_epi16(sum_p3, pq_16[3]);
        work0 = _mm_add_epi16(sum_p3, pq_16[2]);
        let flat_p2 = _mm_add_epi16(sum_lp, work0);
        let flat_q2 = _mm_add_epi16(sum_lq, _mm_srli_si128::<8>(work0));
        let mut flat_pq2 = _mm_srli_epi16::<3>(_mm_unpacklo_epi64(flat_p2, flat_q2));
        flat_pq2 = _mm_packus_epi16(flat_pq2, flat_pq2);

        // flat2
        let mut flat2 = _mm_max_epu8(absd(q4p4, q0p0), absd(q5p5, q0p0));
        work = absd(q6p6, q0p0);
        flat2 = _mm_max_epu8(work, flat2);
        flat2 = _mm_max_epu8(flat2, _mm_srli_si128::<4>(flat2));
        flat2 = _mm_subs_epu8(flat2, one);
        flat2 = _mm_cmpeq_epi8(flat2, zero);
        flat2 = _mm_and_si128(flat2, flat);
        flat2 = _mm_unpacklo_epi32(flat2, flat2);

        // apply flat
        out0 = _mm_or_si128(
            _mm_andnot_si128(flat, qs0ps0),
            _mm_and_si128(flat, flat_pq0),
        );
        out1 = _mm_or_si128(
            _mm_andnot_si128(flat, qs1ps1),
            _mm_and_si128(flat, flat_pq1),
        );
        out2 = _mm_or_si128(_mm_andnot_si128(flat, q2p2), _mm_and_si128(flat, flat_pq2));

        if _mm_movemask_epi8(_mm_cmpeq_epi8(flat2, zero)) != 0xffff {
            let flat2_p0 = _mm_add_epi16(sum_p_0, _mm_add_epi16(work0_0, q0_16));
            let flat2_q0 = _mm_add_epi16(
                sum_p_0,
                _mm_add_epi16(_mm_srli_si128::<8>(work0_0), pq_16[0]),
            );
            let flat2_p1 = _mm_add_epi16(sum_p, work0_1);
            let flat2_q1 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0_1));
            let mut flat2_pq0 = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(flat2_p0, flat2_q0));
            let mut flat2_pq1 = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(flat2_p1, flat2_q1));
            flat2_pq0 = _mm_packus_epi16(flat2_pq0, flat2_pq0);
            flat2_pq1 = _mm_packus_epi16(flat2_pq1, flat2_pq1);

            sum_p = _mm_sub_epi16(sum_p, q4_16);
            sum_q = _mm_sub_epi16(sum_q, pq_16[4]);
            sum_p6 = _mm_add_epi16(sum_p6, pq_16[6]);
            work0 = _mm_add_epi16(
                sum_p6,
                _mm_add_epi16(pq_16[2], _mm_add_epi16(pq_16[3], pq_16[1])),
            );
            let flat2_p2 = _mm_add_epi16(sum_p, work0);
            let flat2_q2 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
            let mut flat2_pq2 = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(flat2_p2, flat2_q2));
            flat2_pq2 = _mm_packus_epi16(flat2_pq2, flat2_pq2);

            sum_p6 = _mm_add_epi16(sum_p6, pq_16[6]);
            sum_p = _mm_sub_epi16(sum_p, q3_16);
            sum_q = _mm_sub_epi16(sum_q, pq_16[3]);
            work0 = _mm_add_epi16(
                sum_p6,
                _mm_add_epi16(pq_16[3], _mm_add_epi16(pq_16[4], pq_16[2])),
            );
            let flat2_p3 = _mm_add_epi16(sum_p, work0);
            let flat2_q3 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
            let mut flat2_pq3 = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(flat2_p3, flat2_q3));
            flat2_pq3 = _mm_packus_epi16(flat2_pq3, flat2_pq3);

            sum_p6 = _mm_add_epi16(sum_p6, pq_16[6]);
            sum_p = _mm_sub_epi16(sum_p, q2_16);
            sum_q = _mm_sub_epi16(sum_q, pq_16[2]);
            work0 = _mm_add_epi16(
                sum_p6,
                _mm_add_epi16(pq_16[4], _mm_add_epi16(pq_16[5], pq_16[3])),
            );
            let flat2_p4 = _mm_add_epi16(sum_p, work0);
            let flat2_q4 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
            let mut flat2_pq4 = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(flat2_p4, flat2_q4));
            flat2_pq4 = _mm_packus_epi16(flat2_pq4, flat2_pq4);

            sum_p6 = _mm_add_epi16(sum_p6, pq_16[6]);
            sum_p = _mm_sub_epi16(sum_p, q1_16);
            sum_q = _mm_sub_epi16(sum_q, pq_16[1]);
            work0 = _mm_add_epi16(
                sum_p6,
                _mm_add_epi16(pq_16[5], _mm_add_epi16(pq_16[6], pq_16[4])),
            );
            let flat2_p5 = _mm_add_epi16(sum_p, work0);
            let flat2_q5 = _mm_add_epi16(sum_q, _mm_srli_si128::<8>(work0));
            let mut flat2_pq5 = _mm_srli_epi16::<4>(_mm_unpacklo_epi64(flat2_p5, flat2_q5));
            flat2_pq5 = _mm_packus_epi16(flat2_pq5, flat2_pq5);

            // wide flat apply
            out0 = _mm_or_si128(
                _mm_andnot_si128(flat2, out0),
                _mm_and_si128(flat2, flat2_pq0),
            );
            out1 = _mm_or_si128(
                _mm_andnot_si128(flat2, out1),
                _mm_and_si128(flat2, flat2_pq1),
            );
            out2 = _mm_or_si128(
                _mm_andnot_si128(flat2, out2),
                _mm_and_si128(flat2, flat2_pq2),
            );
            out3 = _mm_or_si128(
                _mm_andnot_si128(flat2, q3p3),
                _mm_and_si128(flat2, flat2_pq3),
            );
            out4 = _mm_or_si128(
                _mm_andnot_si128(flat2, q4p4),
                _mm_and_si128(flat2, flat2_pq4),
            );
            out5 = _mm_or_si128(
                _mm_andnot_si128(flat2, q5p5),
                _mm_and_si128(flat2, flat2_pq5),
            );
        }
    }
    [out0, out1, out2, out3, out4, out5]
}
