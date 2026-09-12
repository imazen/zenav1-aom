//! aom-quant — bit-exact AV1 quantization kernels (port of libaom v3.14.1).
//!
//! Validated byte-for-byte against the C reference by differential harnesses in
//! `tests/`. Starts with the `av1_quantize_fp` family (the encoder fast-path
//! quantizer, no quant-matrix), which is the stage directly downstream of the
//! forward transform.


mod build_quantizer;
mod qm;
mod qm_fwd_tables;
mod qm_inv_tables;
mod quant_common;
pub mod simd;
pub use build_quantizer::{
    av1_build_quantizer, av1_set_quantizer, set_q_index, Dequants, PlaneQuantRows, QuantTuning,
    QuantizerSettings, Quants, QINDEX_RANGE,
};
pub use qm::{iqmatrix, qmatrix, NUM_QM_LEVELS};
pub use quant_common::{
    aom_get_qmlevel, aom_get_qmlevel_444_chroma, aom_get_qmlevel_allintra,
    aom_get_qmlevel_luma_ssimulacra2, av1_ac_quant_qtx, av1_dc_quant_qtx, av1_get_qindex,
    Segmentation, MAX_SEGMENTS, QM_FIRST_IQ_SSIMULACRA2, QM_LAST_IQ_SSIMULACRA2, SEG_LVL_ALT_Q,
    SEG_LVL_MAX, SEG_LVL_SKIP,
};

/// `ROUND_POWER_OF_TWO(value, n)` from `aom_ports/mem.h` — bit-exact.
/// Note `(1<<n)>>1` yields 0 at n=0, so this is well-defined for `log_scale==0`.
#[inline]
pub(crate) fn round_power_of_two(value: i32, n: i32) -> i32 {
    (value + ((1 << n) >> 1)) >> n
}

/// `AOMSIGN(x)`: -1 if negative, else 0.
#[inline]
fn aomsign(x: i32) -> i32 {
    if x < 0 {
        -1
    } else {
        0
    }
}

/// Bit-exact port of `av1_quantize_fp_no_qmatrix` (`av1/encoder/av1_quantize.c`).
/// This is the body of `av1_quantize_fp_c` / `_32x32_c` / `_64x64_c` for the
/// no-quant-matrix case (`log_scale` = 0 / 1 / 2 respectively).
///
/// Writes `qcoeff` (quantized) and `dqcoeff` (dequantized) and returns the EOB.
/// `quant`, `dequant`, `round` are the `[dc, ac]` parameter pairs; `scan` is the
/// coefficient scan order (length `coeff.len()`).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_no_qmatrix(
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let rounding = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    let mut eob: u16 = 0;
    for i in 0..n {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize; // dc uses index 0, ac uses index 1
        let thresh = dequant[ac] as i64;
        let coeff_v = coeff[rc];
        let coeff_sign = aomsign(coeff_v);
        // int arithmetic then widen, as in C.
        let mut abs_coeff = (coeff_v ^ coeff_sign).wrapping_sub(coeff_sign) as i64;
        let mut tmp32: i32 = 0;
        if (abs_coeff << (1 + log_scale)) >= thresh {
            abs_coeff = (abs_coeff + rounding[ac] as i64).clamp(i16::MIN as i64, i16::MAX as i64);
            tmp32 = ((abs_coeff * quant[ac] as i64) >> (16 - log_scale)) as i32;
            if tmp32 != 0 {
                qcoeff[rc] = (tmp32 ^ coeff_sign).wrapping_sub(coeff_sign);
                let abs_dqcoeff = tmp32.wrapping_mul(dequant[ac] as i32) >> log_scale;
                dqcoeff[rc] = (abs_dqcoeff ^ coeff_sign).wrapping_sub(coeff_sign);
            }
        }
        if tmp32 != 0 {
            eob = (i + 1) as u16;
        }
    }
    eob
}

const AOM_QM_BITS: i32 = 5;

/// One coefficient of [`aom_quantize_b_no_qmatrix`], with every per-class
/// constant already resolved by the caller. Returns the quantized magnitude
/// `tmp32` (0 when the coefficient is inside the dead zone) and writes the
/// signed `qcoeff` / `dqcoeff` pair.
///
/// The arithmetic is the C body verbatim, including the `wrapping_*` forms:
/// `abs_coeff` is `i32::MIN` when `coeff == i32::MIN`, and C's `int` multiply
/// wraps there too.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn quantize_b_one(
    coeff_v: i32,
    zbin_gate: i32,
    round_rpo: i32,
    quant_v: i64,
    quant_shift_v: i64,
    dequant_v: i32,
    log_scale: i32,
    shift: i32,
    qcoeff: &mut i32,
    dqcoeff: &mut i32,
) -> i32 {
    const WT: i32 = 1 << AOM_QM_BITS; // 32; no quant matrix
    let coeff_sign = aomsign(coeff_v);
    let abs_coeff = (coeff_v ^ coeff_sign).wrapping_sub(coeff_sign);
    let tmp32 = if abs_coeff.wrapping_mul(WT) >= zbin_gate {
        let clamped =
            abs_coeff.wrapping_add(round_rpo).clamp(i16::MIN as i32, i16::MAX as i32) as i64;
        let tmp = clamped * WT as i64;
        (((((tmp * quant_v) >> 16) + tmp) * quant_shift_v) >> shift) as i32
    } else {
        0
    };
    *qcoeff = (tmp32 ^ coeff_sign).wrapping_sub(coeff_sign);
    let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
    *dqcoeff = (abs_dqcoeff ^ coeff_sign).wrapping_sub(coeff_sign);
    tmp32
}

/// Bit-exact port of `aom_quantize_b_helper_c` (`aom_dsp/quantize.c`) for the
/// no-quant-matrix case (`wt = iwt = 1<<AOM_QM_BITS`). The "b" quantizer with a
/// dead-zone (`zbin`) pre-scan and two-step `quant`/`quant_shift`.
///
/// **Walks RASTER order, where C walks scan order, and derives the EOB from
/// `iscan`** — the same restructuring `crate::quant::simd`'s `quantize_fp`
/// already uses, and the same one libaom's own `aom_quantize_b_avx2` uses (its
/// `iscan` argument exists for exactly this; the `_c` body ignores it). Three
/// facts make it value-identical, and the differential against the real
/// exported C is what asserts it:
///
/// * **The set of positions that WRITE is the same.** C's pre-scan trims
///   trailing coefficients satisfying `|coeff| < zbin[ac]` and the main loop
///   writes only where `|coeff| >= zbin[ac]` — exact complements, so a trimmed
///   position would have written nothing anyway, and `memset` had already left
///   it zero. Here every position is written unconditionally, with the same
///   zero on the dead-zone side, so **both `memset`s are gone** rather than
///   being re-done by the loop.
/// * **`ac` is the raster index test.** C computes `ac = (scan[i] != 0)`, i.e.
///   it asks whether the RASTER position is the DC. Walking raster order that
///   is `i != 0`, so the per-class constants become loop-invariant and the DC
///   peels off — no `scan[i]` load, no select, no table index per coefficient.
/// * **`eob = 1 + max(iscan[rc])` over written-nonzero positions equals C's
///   scan-order maximum**, because `iscan` is the inverse permutation of
///   `scan`. This is the ONE order-sensitive output of the family (KB-12), so
///   it carries its own bite proof in `quantize_b_diff`.
///
/// `scan` is retained for signature fidelity with `aom_quantize_b_helper_c` and
/// is used only to check the two permutations agree in length.
#[allow(clippy::too_many_arguments)]
pub fn aom_quantize_b_no_qmatrix(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    debug_assert_eq!(scan.len(), n, "scan must cover the block");
    debug_assert!(
        iscan.len() >= n && qcoeff.len() >= n && dqcoeff.len() >= n,
        "iscan/qcoeff/dqcoeff must cover the block"
    );
    if n == 0 {
        return 0;
    }
    // Below one vector width the scalar walk is strictly better (no AV1 tx
    // block area is < 8 anyway).
    if n < 8 {
        return quantize_b_scalar(
            zbin, round, quant, quant_shift, dequant, log_scale, iscan, coeff, qcoeff, dqcoeff,
        );
    }
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    archmage::incant!(
        quantize_b_impl(
            zbin, round, quant, quant_shift, dequant, log_scale, scan, iscan, coeff, qcoeff,
            dqcoeff
        ),
        [v3, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
fn quantize_b_impl_scalar(
    _t: archmage::ScalarToken,
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    _scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    quantize_b_scalar(
        zbin, round, quant, quant_shift, dequant, log_scale, iscan, coeff, qcoeff, dqcoeff,
    )
}

#[allow(clippy::too_many_arguments)]
fn quantize_b_scalar(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    const WT: i32 = 1 << AOM_QM_BITS;
    let n = coeff.len();

    // Per-class constants, hoisted out of the walk. In C every one of these is
    // re-derived per coefficient through `ac = (scan[i] != 0)`.
    let zbin_gate = [
        round_power_of_two(zbin[0] as i32, log_scale) << AOM_QM_BITS,
        round_power_of_two(zbin[1] as i32, log_scale) << AOM_QM_BITS,
    ];
    let round_rpo = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    // iwt = 32 -> dequant = (dequant[ac]*32 + 16) >> 5 == dequant[ac].
    let dequant_v = [
        (dequant[0] as i32 * WT + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
        (dequant[1] as i32 * WT + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
    ];
    let shift = 16 - log_scale + AOM_QM_BITS;

    // `eob` is 1 + the largest SCAN index that quantized nonzero; 0 = none.
    let mut eob: i32 = 0;

    // Raster position 0 is the DC — the only one taking the `ac == 0` class.
    let tmp32 = quantize_b_one(
        coeff[0],
        zbin_gate[0],
        round_rpo[0],
        quant[0] as i64,
        quant_shift[0] as i64,
        dequant_v[0],
        log_scale,
        shift,
        &mut qcoeff[0],
        &mut dqcoeff[0],
    );
    if tmp32 != 0 {
        eob = iscan[0] as i32 + 1;
    }

    // Every remaining raster position is AC, so the five constants below are
    // scalars for the whole walk and the three slices are iterated rather than
    // indexed — no bounds check, no gather, no scatter.
    let (zg, rr) = (zbin_gate[1], round_rpo[1]);
    let (qv, qsv, dqv) = (quant[1] as i64, quant_shift[1] as i64, dequant_v[1]);
    for (((&c, q), dq), &is) in coeff[1..n]
        .iter()
        .zip(qcoeff[1..n].iter_mut())
        .zip(dqcoeff[1..n].iter_mut())
        .zip(iscan[1..n].iter())
    {
        let tmp32 = quantize_b_one(c, zg, rr, qv, qsv, dqv, log_scale, shift, q, dq);
        if tmp32 != 0 {
            eob = eob.max(is as i32 + 1);
        }
    }
    eob as u16
}

/// x86-64/AVX2 body for [`aom_quantize_b_no_qmatrix`]. NOT a transcription of
/// `aom_quantize_b_avx2` — C's kernel runs the whole chain in saturating i16
/// lanes (`packs_epi32` on the coefficients, `adds_epi16` on the round), which
/// diverges from its own scalar on `|coeff| > i16::MAX` or
/// `abs + round > i16::MAX`. This port's contract is the C *scalar* on the full
/// i32 coefficient domain (that is what `quantize_b_diff` fuzzes, at |coeff| <
/// 2^19), so the kernel keeps i32 lanes and widens only where the scalar's i64
/// intermediates genuinely need it.
///
/// Per 8 lanes (i32 unless noted), mirroring [`quantize_b_one`]:
/// ```text
/// sign    = srai(c,31);  abs   = (c ^ sign) - sign          (wrapping)
/// gate    = mullo(abs,32) > zbin_gate-1                     (i32 wrap; gate<=2^20)
/// clamped = min(max(add(abs,round), i16min), i16max)        (wrapping add)
/// t       = srai(clamped*quant, 11) + clamped*32
///             -- reassociation of (((clamped*32)*quant)>>16)+clamped*32:
///             |clamped*quant| <= 2^30 stays exact in i32, and pulling the *32
///             out of the >>16 leaves >>11. |t| <= 2^21.
/// tmp32   = (t * quant_shift) >> (21-log_scale)
///             -- the ONE product that needs i64: |t*quant_shift| <= 2^36.
///             mul_epi32 on even lanes and on sign-extended odd lanes gives the
///             signed 32x32->64 products; srl_epi64 suffices because the scalar
///             keeps only the low 32 bits (`as i32`) and the arith-vs-logical
///             difference lives above bit 32.
/// qcoeff  = (tmp32 ^ sign) - sign ;  dqcoeff likewise after *dequant>>log_scale
/// eob     = max over lanes of (iscan+1) where tmp32 != 0
/// ```
///
/// Chunk 0 carries the DC lane (class 0 constants in lane 0, AC elsewhere);
/// every later chunk is all-AC. The `n % 8` remainder — unreachable for real
/// transform sizes — runs the scalar body so the function is total.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn quantize_b_impl_v3(
    _t: archmage::X64V3Token,
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    _scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    use archmage::intrinsics::x86_64::*;
    const WT: i32 = 1 << AOM_QM_BITS;
    let n = coeff.len();
    let nb = n / 8 * 8;

    // Per-class constants — same formulas as the scalar walk.
    let zg = [
        (round_power_of_two(zbin[0] as i32, log_scale) << AOM_QM_BITS) - 1,
        (round_power_of_two(zbin[1] as i32, log_scale) << AOM_QM_BITS) - 1,
    ];
    let rr = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    let dqv = [
        (dequant[0] as i32 * WT + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
        (dequant[1] as i32 * WT + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
    ];
    let shift = 16 - log_scale + AOM_QM_BITS;
    let sh_cnt = _mm_cvtsi32_si128(shift);
    let ls_cnt = _mm_cvtsi32_si128(log_scale);

    // Chunk-0 vectors carry the DC class in lane 0; the rest are all-AC.
    let lane0 = |dc: i32, ac: i32| _mm256_loadu_si256(&[dc, ac, ac, ac, ac, ac, ac, ac]);
    let zg_f = lane0(zg[0], zg[1]);
    let rr_f = lane0(rr[0], rr[1]);
    let q_f = lane0(quant[0] as i32, quant[1] as i32);
    let qs_f = lane0(quant_shift[0] as i32, quant_shift[1] as i32);
    let dq_f = lane0(dqv[0], dqv[1]);
    let (zg_a, rr_a) = (_mm256_set1_epi32(zg[1]), _mm256_set1_epi32(rr[1]));
    let (q_a, qs_a) = (_mm256_set1_epi32(quant[1] as i32), _mm256_set1_epi32(quant_shift[1] as i32));
    let dq_a = _mm256_set1_epi32(dqv[1]);

    let lo = _mm256_set1_epi32(i16::MIN as i32);
    let hi = _mm256_set1_epi32(i16::MAX as i32);
    let wt = _mm256_set1_epi32(WT);
    let one = _mm256_set1_epi32(1);
    let zero = _mm256_setzero_si256();
    let mut eob_v = zero;

    macro_rules! chunk8 {
        ($ci:expr, $zg:expr, $rr:expr, $q:expr, $qs:expr, $dq:expr) => {{
            let ci = $ci;
            let c8: &[i32; 8] = coeff[ci * 8..ci * 8 + 8].try_into().unwrap();
            let cv = _mm256_loadu_si256(c8);
            let sign = _mm256_srai_epi32::<31>(cv);
            let abs = _mm256_sub_epi32(_mm256_xor_si256(cv, sign), sign);
            let gate = _mm256_cmpgt_epi32(_mm256_mullo_epi32(abs, wt), $zg);
            let clamped =
                _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(abs, $rr), lo), hi);
            let t = _mm256_add_epi32(
                _mm256_srai_epi32::<11>(_mm256_mullo_epi32(clamped, $q)),
                _mm256_slli_epi32::<5>(clamped),
            );
            // Signed 32x32->64 products, even lanes directly, odd lanes via a
            // sign-extended i64 view (srli gets the lane, srai its sign).
            let pe = _mm256_mul_epi32(t, $qs);
            let t_odd = _mm256_blend_epi32::<0xAA>(
                _mm256_srli_epi64::<32>(t),
                _mm256_srai_epi32::<31>(t),
            );
            // Odd lanes are always AC class (the DC lane is even), so the odd
            // product always uses the all-AC `qs_a` — `mul_epi32` reads the
            // i64-lane low halves, where a per-lane vector would supply the
            // WRONG class on chunk 0.
            let po = _mm256_mul_epi32(t_odd, qs_a);
            let re = _mm256_srl_epi64(pe, sh_cnt);
            let ro = _mm256_srl_epi64(po, sh_cnt);
            // i64 results are < 2^17 in magnitude: high halves are 0, so the
            // shifted odd-lane product slots straight back into i32 position.
            let t32 = _mm256_and_si256(
                _mm256_blend_epi32::<0xAA>(re, _mm256_slli_epi64::<32>(ro)),
                gate,
            );
            let qc = _mm256_sub_epi32(_mm256_xor_si256(t32, sign), sign);
            let adq = _mm256_sra_epi32(_mm256_mullo_epi32(t32, $dq), ls_cnt);
            let dq = _mm256_sub_epi32(_mm256_xor_si256(adq, sign), sign);
            let q8: &mut [i32; 8] = (&mut qcoeff[ci * 8..ci * 8 + 8]).try_into().unwrap();
            let d8: &mut [i32; 8] = (&mut dqcoeff[ci * 8..ci * 8 + 8]).try_into().unwrap();
            _mm256_storeu_si256(q8, qc);
            _mm256_storeu_si256(d8, dq);
            let is8: &[i16; 8] = iscan[ci * 8..ci * 8 + 8].try_into().unwrap();
            let isc =
                _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm_loadu_si128(is8)), one);
            eob_v = _mm256_max_epi32(
                eob_v,
                _mm256_andnot_si256(_mm256_cmpeq_epi32(t32, zero), isc),
            );
        }};
    }

    chunk8!(0, zg_f, rr_f, q_f, qs_f, dq_f);
    for ci in 1..nb / 8 {
        chunk8!(ci, zg_a, rr_a, q_a, qs_a, dq_a);
    }

    let mut eob_lanes = [0i32; 8];
    _mm256_storeu_si256(&mut eob_lanes, eob_v);
    let mut eob = eob_lanes.into_iter().max().unwrap_or(0);

    // Raster tail (< 8 coeffs, only reachable for non-transform callers):
    // same scalar body, all-AC class.
    for i in nb..n {
        let mut qc = 0i32;
        let mut dq = 0i32;
        let t32 = quantize_b_one(
            coeff[i],
            zg[1] + 1,
            rr[1],
            quant[1] as i64,
            quant_shift[1] as i64,
            dqv[1],
            log_scale,
            shift,
            &mut qc,
            &mut dq,
        );
        qcoeff[i] = qc;
        dqcoeff[i] = dq;
        if t32 != 0 {
            eob = eob.max(iscan[i] as i32 + 1);
        }
    }
    eob as u16
}

/// Bit-exact port of `aom_quantize_b_helper_c` (`aom_dsp/quantize.c`) *with* a
/// quant matrix: `qm[rc]` weights the coefficient (`wt`), `iqm[rc]` weights the
/// dequant (`iwt`). This is the general form of [`aom_quantize_b_no_qmatrix`]
/// (`wt = iwt = 1<<AOM_QM_BITS`). `qm`/`iqm` are indexed by raster position `rc`
/// (same length as `coeff`). Integer widths mirror the C exactly (`coeff*wt`,
/// `abs_coeff*wt`, `dequant` are 32-bit; the round/quant chain is 64-bit).
// Scan/eob indexing (reverse-break pre-scan + `eob = i` forward pass) mirrors the
// C `for` loops 1:1, so the index is load-bearing — keep the explicit `scan[i]`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn aom_quantize_b_qm(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);

    // Pre-scan pass (from the end): trim trailing dead-zone coefficients.
    let mut non_zero_count = n as i32;
    for i in (0..n).rev() {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let wt = qm[rc] as i32;
        let c = coeff[rc].wrapping_mul(wt);
        if c < zbins[ac].wrapping_mul(1 << AOM_QM_BITS)
            && c > nzbins[ac].wrapping_mul(1 << AOM_QM_BITS)
        {
            non_zero_count -= 1;
        } else {
            break;
        }
    }

    let mut eob: i32 = -1;
    for i in 0..non_zero_count as usize {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let coeff_sign = aomsign(coeff_v);
        let abs_coeff = (coeff_v ^ coeff_sign).wrapping_sub(coeff_sign);
        let wt = qm[rc] as i32;
        if abs_coeff.wrapping_mul(wt) >= (zbins[ac] << AOM_QM_BITS) {
            let clamped = (abs_coeff.wrapping_add(round_power_of_two(round[ac] as i32, log_scale)))
                .clamp(i16::MIN as i32, i16::MAX as i32);
            let mut tmp = clamped as i64;
            tmp *= wt as i64;
            let tmp32 = (((((tmp * quant[ac] as i64) >> 16) + tmp) * quant_shift[ac] as i64)
                >> (16 - log_scale + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (tmp32 ^ coeff_sign).wrapping_sub(coeff_sign);
            let iwt = iqm[rc] as i32;
            let dequant_v = (dequant[ac] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
            let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ coeff_sign).wrapping_sub(coeff_sign);
            if tmp32 != 0 {
                eob = i as i32;
            }
        }
    }
    (eob + 1) as u16
}

/// `av1_quantize_fp` (log_scale 0). Signature mirrors the C entry (unused
/// `zbin`/`quant_shift`/`iscan` args omitted).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp(
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    scan: &[i16],
) -> u16 {
    av1_quantize_fp_no_qmatrix(quant, dequant, round, 0, scan, coeff, qcoeff, dqcoeff)
}

// The C's `av1_quantize_fp_32x32` / `av1_quantize_fp_64x64` are the same helper
// at log_scale 1 / 2. Call [`av1_quantize_fp_no_qmatrix`] with the log_scale
// directly — that is the entry `tests/quantize_fp_diff.rs` sweeps against C for
// all three scales; named wrappers for 1 and 2 had no callers.

/// Bit-exact port of `highbd_quantize_fp_helper_c` (`av1/encoder/av1_quantize.c`)
/// for the no-quant-matrix path. Highbd (10/12-bit) FP quantizer: like the lowbd
/// path but with 64-bit arithmetic throughout (no int16 clamp on the rounded
/// coefficient). Returns eob.
#[allow(clippy::too_many_arguments)]
pub fn av1_highbd_quantize_fp_no_qmatrix(
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let shift = 16 - log_scale;
    let lsr = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    let mut eob: i32 = -1;
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let sign = aomsign(coeff_v);
        let abs_coeff = (coeff_v ^ sign).wrapping_sub(sign);
        if ((abs_coeff as i64) << (1 + log_scale)) >= dequant[rc01] as i64 {
            let tmp = abs_coeff as i64 + lsr[rc01] as i64;
            let abs_qcoeff = ((tmp * quant[rc01] as i64) >> shift) as i32;
            qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
            let abs_dqcoeff = abs_qcoeff.wrapping_mul(dequant[rc01] as i32) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
            if abs_qcoeff != 0 {
                eob = i as i32;
            }
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `aom_highbd_quantize_b_helper_c` (`aom_dsp/quantize.c`) for
/// the no-quant-matrix case (`wt = iwt = 1<<AOM_QM_BITS`). Highbd "b" quantizer:
/// dead-zone (`zbin`) pre-scan + two-step `quant`/`quant_shift`, 64-bit.
#[allow(clippy::too_many_arguments)]
pub fn aom_highbd_quantize_b_no_qmatrix(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];
    let wt = 1i64 << AOM_QM_BITS;

    // Pre-scan pass (wt = 32): keep coeffs outside the ZBIN dead-zone.
    let mut idx_arr = Vec::with_capacity(n);
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let coeff_w = coeff[rc] as i64 * wt;
        if coeff_w >= zbins[(rc != 0) as usize] as i64 * wt
            || coeff_w <= nzbins[(rc != 0) as usize] as i64 * wt
        {
            idx_arr.push(i);
        }
    }

    let mut eob: i32 = -1;
    for &ii in &idx_arr {
        let rc = scan[ii] as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let sign = aomsign(coeff_v);
        let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        let tmp1 = abs_coeff + round_power_of_two(round[rc01] as i32, log_scale) as i64;
        let tmpw = tmp1 * wt;
        let tmp2 = ((tmpw * quant[rc01] as i64) >> 16) + tmpw;
        let abs_qcoeff =
            ((tmp2 * quant_shift[rc01] as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
        qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
        // iwt = 32: dequant = (dequant*32 + 16) >> 5 == dequant.
        let dq =
            ((dequant[rc01] as i32 * (1 << AOM_QM_BITS)) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let abs_dqcoeff = abs_qcoeff.wrapping_mul(dq) >> log_scale;
        dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
        if abs_qcoeff != 0 {
            eob = ii as i32;
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `aom_highbd_quantize_b_helper_c` (`aom_dsp/quantize.c`)
/// *with* a quant matrix: general form of [`aom_highbd_quantize_b_no_qmatrix`]
/// with per-position `wt = qm[rc]` / `iwt = iqm[rc]`. Integer widths mirror the
/// C exactly (`coeff*wt` and `dequant` are 32-bit; the quant chain is 64-bit).
#[allow(clippy::too_many_arguments)]
pub fn aom_highbd_quantize_b_qm(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];

    // Pre-scan pass (forward): keep coeffs outside the ZBIN dead-zone. `coeff*wt`
    // is 32-bit in C.
    let mut idx_arr = Vec::with_capacity(n);
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let ac = (rc != 0) as usize;
        let wt = qm[rc] as i32;
        let coeff_w = coeff[rc].wrapping_mul(wt);
        if coeff_w >= zbins[ac].wrapping_mul(1 << AOM_QM_BITS)
            || coeff_w <= nzbins[ac].wrapping_mul(1 << AOM_QM_BITS)
        {
            idx_arr.push(i);
        }
    }

    let mut eob: i32 = -1;
    for &ii in &idx_arr {
        let rc = scan[ii] as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let sign = aomsign(coeff_v);
        let wt = qm[rc] as i64;
        let iwt = iqm[rc] as i32;
        let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        let tmp1 = abs_coeff + round_power_of_two(round[rc01] as i32, log_scale) as i64;
        let tmpw = tmp1 * wt;
        let tmp2 = ((tmpw * quant[rc01] as i64) >> 16) + tmpw;
        let abs_qcoeff =
            ((tmp2 * quant_shift[rc01] as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
        qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
        let dq = (dequant[rc01] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let abs_dqcoeff = abs_qcoeff.wrapping_mul(dq) >> log_scale;
        dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
        if abs_qcoeff != 0 {
            eob = ii as i32;
        }
    }
    (eob + 1) as u16
}

/// `EOB_FACTOR` / `SKIP_EOB_FACTOR_ADJUST` (aom_dsp/quantize.h:23-24): the
/// adaptive-dead-zone widening + lone-last-coeff drop constants used ONLY by the
/// `aom_quantize_b_adaptive` family (`--quant-b-adapt`).
const EOB_FACTOR: i32 = 325;
const SKIP_EOB_FACTOR_ADJUST: i32 = 200;

/// Bit-exact port of `aom_quantize_b_adaptive_helper_c` (`aom_dsp/quantize.c`) —
/// the lowbd adaptive dead-zone "b" quantizer (`--quant-b-adapt`). Differs from
/// [`aom_quantize_b_no_qmatrix`] / [`aom_quantize_b_qm`] only in two places:
/// (1) the pre-scan dead-zone is WIDENED by
/// `prescan_add = ROUND_POWER_OF_TWO(dequant * EOB_FACTOR, 7)`, and (2) a
/// SKIP_EOB_FACTOR_ADJUST tail drops a lone trailing ±1 coefficient whose
/// factor-adjusted dead-zone would zero it. `qm`/`iqm = None` is the no-matrix
/// case (`wt = iwt = 1<<AOM_QM_BITS`); `Some` weights per raster position (the
/// `--enable-qm` + `--quant-b-adapt` corner). Integer widths mirror C: the
/// pre-scan `coeff*wt` and the zbin compares are 32-bit; the quant chain is
/// 64-bit; the rounded coeff is clamped to i16 (the lowbd difference).
#[allow(clippy::too_many_arguments)]
pub fn aom_quantize_b_adaptive_helper(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);

    let wt_at = |rc: usize| -> i32 { qm.map_or(1 << AOM_QM_BITS, |m| m[rc] as i32) };
    let prescan_add = [
        round_power_of_two(dequant[0] as i32 * EOB_FACTOR, 7),
        round_power_of_two(dequant[1] as i32 * EOB_FACTOR, 7),
    ];

    // Pre-scan pass (from the end): trim trailing (widened) dead-zone coeffs.
    let mut non_zero_count = n as i32;
    for i in (0..n).rev() {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let c = coeff[rc].wrapping_mul(wt_at(rc));
        let add = prescan_add[ac];
        if c < zbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_add(add)
            && c > nzbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_sub(add)
        {
            non_zero_count -= 1;
        } else {
            break;
        }
    }

    let mut eob: i32 = -1;
    let mut first: i32 = -1;
    for i in 0..non_zero_count as usize {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let coeff_sign = aomsign(coeff_v);
        let abs_coeff = (coeff_v ^ coeff_sign).wrapping_sub(coeff_sign);
        let wt = wt_at(rc);
        if abs_coeff.wrapping_mul(wt) >= (zbins[ac] << AOM_QM_BITS) {
            let clamped = (abs_coeff.wrapping_add(round_power_of_two(round[ac] as i32, log_scale)))
                .clamp(i16::MIN as i32, i16::MAX as i32);
            let mut tmp = clamped as i64;
            tmp *= wt as i64;
            let tmp32 = ((((tmp * quant[ac] as i64) >> 16) + tmp) * quant_shift[ac] as i64
                >> (16 - log_scale + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (tmp32 ^ coeff_sign).wrapping_sub(coeff_sign);
            let iwt = iqm.map_or(1 << AOM_QM_BITS, |m| m[rc] as i32);
            let dequant_v = (dequant[ac] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
            let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ coeff_sign).wrapping_sub(coeff_sign);
            if tmp32 != 0 {
                eob = i as i32;
                if first == -1 {
                    first = i as i32;
                }
            }
        }
    }

    // SKIP_EOB_FACTOR_ADJUST: drop a lone trailing ±1 (first == eob) whose
    // factor-adjusted dead-zone would zero it.
    if eob >= 0 && first == eob {
        let rc = scan[eob as usize] as usize;
        if qcoeff[rc] == 1 || qcoeff[rc] == -1 {
            let ac = (rc != 0) as usize;
            let c = coeff[rc].wrapping_mul(wt_at(rc));
            let factor = EOB_FACTOR + SKIP_EOB_FACTOR_ADJUST;
            let add = round_power_of_two(dequant[ac] as i32 * factor, 7);
            if c < zbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_add(add)
                && c > nzbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_sub(add)
            {
                qcoeff[rc] = 0;
                dqcoeff[rc] = 0;
                eob = -1;
            }
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `aom_highbd_quantize_b_adaptive_helper_c`
/// (`aom_dsp/quantize.c`) — the highbd (10/12-bit) adaptive dead-zone "b"
/// quantizer. Same reverse-prescan + SKIP_EOB_FACTOR_ADJUST structure as
/// [`aom_quantize_b_adaptive_helper`] but the quant chain is 64-bit with NO i16
/// clamp on the rounded coefficient (the highbd difference).
#[allow(clippy::too_many_arguments)]
pub fn aom_highbd_quantize_b_adaptive_helper(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);

    let wt_at = |rc: usize| -> i32 { qm.map_or(1 << AOM_QM_BITS, |m| m[rc] as i32) };
    let prescan_add = [
        round_power_of_two(dequant[0] as i32 * EOB_FACTOR, 7),
        round_power_of_two(dequant[1] as i32 * EOB_FACTOR, 7),
    ];

    let mut non_zero_count = n as i32;
    for i in (0..n).rev() {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let c = coeff[rc].wrapping_mul(wt_at(rc));
        let add = prescan_add[ac];
        if c < zbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_add(add)
            && c > nzbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_sub(add)
        {
            non_zero_count -= 1;
        } else {
            break;
        }
    }

    let mut eob: i32 = -1;
    let mut first: i32 = -1;
    for i in 0..non_zero_count as usize {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let sign = aomsign(coeff_v);
        let abs_coeff = (coeff_v ^ sign).wrapping_sub(sign);
        let wt = wt_at(rc);
        if abs_coeff.wrapping_mul(wt) >= (zbins[ac] << AOM_QM_BITS) {
            let tmp1 = abs_coeff as i64 + round_power_of_two(round[ac] as i32, log_scale) as i64;
            let tmpw = tmp1 * wt as i64;
            let tmp2 = ((tmpw * quant[ac] as i64) >> 16) + tmpw;
            let abs_qcoeff =
                ((tmp2 * quant_shift[ac] as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
            let iwt = iqm.map_or(1 << AOM_QM_BITS, |m| m[rc] as i32);
            let dequant_v = (dequant[ac] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
            let abs_dqcoeff = abs_qcoeff.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
            if abs_qcoeff != 0 {
                eob = i as i32;
                if first == -1 {
                    first = i as i32;
                }
            }
        }
    }

    if eob >= 0 && first == eob {
        let rc = scan[eob as usize] as usize;
        if qcoeff[rc] == 1 || qcoeff[rc] == -1 {
            let ac = (rc != 0) as usize;
            let c = coeff[rc].wrapping_mul(wt_at(rc));
            let factor = EOB_FACTOR + SKIP_EOB_FACTOR_ADJUST;
            let add = round_power_of_two(dequant[ac] as i32 * factor, 7);
            if c < zbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_add(add)
                && c > nzbins[ac].wrapping_mul(1 << AOM_QM_BITS).wrapping_sub(add)
            {
                qcoeff[rc] = 0;
                dqcoeff[rc] = 0;
                eob = -1;
            }
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of the QM branch of `quantize_fp_helper_c`
/// (`av1/encoder/av1_quantize.c`) — the lowbd VarDCT-FP quantizer with a quant
/// matrix. `wt = qm[rc]` / `iwt = iqm[rc]` per raster position. The rounded
/// coefficient is clamped to the i16 range (unlike the highbd FP variant).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_qm(
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let rounding = [
        round_power_of_two(round[0] as i32, log_scale) as i64,
        round_power_of_two(round[1] as i32, log_scale) as i64,
    ];
    let mut eob: i32 = -1;
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let wt = qm[rc] as i64;
        let iwt = iqm[rc] as i32;
        let dequant_v = (dequant[rc01] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let sign = aomsign(coeff_v);
        let mut abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        let mut tmp32 = 0i32;
        if abs_coeff * wt >= ((dequant[rc01] as i64) << (AOM_QM_BITS - (1 + log_scale))) {
            abs_coeff += rounding[rc01];
            abs_coeff = abs_coeff.clamp(i16::MIN as i64, i16::MAX as i64);
            tmp32 =
                ((abs_coeff * wt * quant[rc01] as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (tmp32 ^ sign).wrapping_sub(sign);
            let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
        }
        if tmp32 != 0 {
            eob = i as i32;
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of the QM branch of `highbd_quantize_fp_helper_c`
/// (`av1/encoder/av1_quantize.c`) — the highbd (10/12-bit) VarDCT-FP quantizer
/// with a quant matrix. No i16 clamp on the rounded coefficient; the quant chain
/// is 64-bit throughout.
#[allow(clippy::too_many_arguments)]
pub fn av1_highbd_quantize_fp_qm(
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let shift = 16 - log_scale;
    let mut eob: i32 = -1;
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let wt = qm[rc] as i64;
        let iwt = iqm[rc] as i32;
        let dequant_v = (dequant[rc01] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let sign = aomsign(coeff_v);
        let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        if abs_coeff * wt >= ((dequant[rc01] as i64) << (AOM_QM_BITS - (1 + log_scale))) {
            let tmp = abs_coeff + round_power_of_two(round[rc01] as i32, log_scale) as i64;
            let abs_qcoeff = ((tmp * quant[rc01] as i64 * wt) >> (shift + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
            let abs_dqcoeff = abs_qcoeff.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
            if abs_qcoeff != 0 {
                eob = i as i32;
            }
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `quantize_dc` (`av1/encoder/av1_quantize.c`) — the DC-only
/// quantizer (`AV1_XFORM_QUANT_DC`): quantizes coefficient 0 only, zeroing the
/// rest. `quant`/`dequant` are the DC scalars; `round[0]` is the DC round. `qm`/
/// `iqm` (when `Some`) weight position 0. Lowbd: the rounded coeff is clamped to
/// the i16 range. Returns eob (0 or 1).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_dc(
    round: &[i16; 2],
    quant: i16,
    dequant: i16,
    log_scale: i32,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let wt = qm.map_or(1 << AOM_QM_BITS, |m| m[0] as i32);
    let iwt = iqm.map_or(1 << AOM_QM_BITS, |m| m[0] as i32);
    let coeff_v = coeff[0];
    let sign = aomsign(coeff_v);
    let abs_coeff = (coeff_v ^ sign).wrapping_sub(sign);
    let clamped = (abs_coeff.wrapping_add(round_power_of_two(round[0] as i32, log_scale)))
        .clamp(i16::MIN as i32, i16::MAX as i32);
    let tmp32 =
        ((clamped as i64 * wt as i64 * quant as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
    qcoeff[0] = (tmp32 ^ sign).wrapping_sub(sign);
    let dequant_v = (dequant as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
    let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
    dqcoeff[0] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
    (tmp32 != 0) as u16
}

/// Bit-exact port of `highbd_quantize_dc` (`av1/encoder/av1_quantize.c`) — the
/// highbd (10/12-bit) DC-only quantizer. Like [`av1_quantize_dc`] but 64-bit
/// with no i16 clamp on the rounded coefficient. Returns eob (0 or 1).
#[allow(clippy::too_many_arguments)]
pub fn av1_highbd_quantize_dc(
    round: &[i16; 2],
    quant: i16,
    dequant: i16,
    log_scale: i32,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let wt = qm.map_or(1 << AOM_QM_BITS, |m| m[0] as i64);
    let iwt = iqm.map_or(1 << AOM_QM_BITS, |m| m[0] as i32);
    let coeff_v = coeff[0];
    let sign = aomsign(coeff_v);
    let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
    let tmp = abs_coeff + round_power_of_two(round[0] as i32, log_scale) as i64;
    let tmpw = tmp * wt;
    let abs_qcoeff = ((tmpw * quant as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
    qcoeff[0] = (abs_qcoeff ^ sign).wrapping_sub(sign);
    let dequant_v = (dequant as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
    let abs_dqcoeff = abs_qcoeff.wrapping_mul(dequant_v) >> log_scale;
    dqcoeff[0] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
    (abs_qcoeff != 0) as u16
}
