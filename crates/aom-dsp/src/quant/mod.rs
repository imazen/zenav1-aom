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
        [v3, neon, scalar]
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
    if n < 8 {
        return highbd_quantize_b_scalar(
            zbin, round, quant, quant_shift, dequant, log_scale, iscan, coeff, qcoeff, dqcoeff,
        );
    }
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    archmage::incant!(
        highbd_quantize_b_impl(
            zbin, round, quant, quant_shift, dequant, log_scale, iscan, coeff, qcoeff, dqcoeff
        ),
        [v3, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
fn highbd_quantize_b_impl_scalar(
    _t: archmage::ScalarToken,
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
    highbd_quantize_b_scalar(
        zbin, round, quant, quant_shift, dequant, log_scale, iscan, coeff, qcoeff, dqcoeff,
    )
}

/// The scalar core of `aom_highbd_quantize_b_helper_c` (no qmatrix) — walks
/// RASTER order where C walks scan order, exactly as [`quantize_b_scalar`]
/// does for the lowbd twin (see its doc for why the reordering is
/// value-identical: position-local arithmetic, `ac` as the raster index test,
/// eob from `iscan`). Every position is written unconditionally, which is
/// what lets the dispatch drop C's two upfront `memset`s and the
/// scan-order `idx_arr`.
#[allow(clippy::too_many_arguments)]
fn highbd_quantize_b_scalar(
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
    let n = coeff.len();
    let wt = 1i64 << AOM_QM_BITS;
    // Per-class constants (index 0 = DC, 1 = AC), hoisted out of the walk.
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale) as i64 * wt,
        round_power_of_two(zbin[1] as i32, log_scale) as i64 * wt,
    ];
    let rr = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    let dqv = [
        ((dequant[0] as i32 * (1 << AOM_QM_BITS)) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
        ((dequant[1] as i32 * (1 << AOM_QM_BITS)) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
    ];
    let shift = 16 - log_scale + AOM_QM_BITS;
    let mut eob: i32 = 0;
    for i in 0..n {
        let ac = (i != 0) as usize;
        let c = coeff[i];
        let sign = aomsign(c);
        // C's pre-scan gate: keep iff c*wt >= zbin*wt || c*wt <= -zbin*wt.
        // The compiled C treats `coeff_ptr[rc] * wt` (i32 mul, UB on
        // overflow) as unbounded — measured: |coeff| >= 2^27 lanes are
        // processed — so the compare runs in i64 and i32::MIN lands outside
        // correctly.
        let cw = c as i64 * wt;
        let tmp32 = if cw >= zbins[ac] || cw <= -zbins[ac] {
            let abs_coeff = (c ^ sign).wrapping_sub(sign);
            // C computes `abs_coeff + round` in `int` — it WRAPS for
            // |coeff| near i32::MAX, making tmp1 negative — then widens.
            let tmp1 = abs_coeff.wrapping_add(rr[ac]) as i64;
            let tmpw = tmp1 * wt;
            let tmp2 = ((tmpw * quant[ac] as i64) >> 16) + tmpw;
            ((tmp2 * quant_shift[ac] as i64) >> shift) as i32
        } else {
            0
        };
        qcoeff[i] = (tmp32 ^ sign).wrapping_sub(sign);
        let abs_dq = tmp32.wrapping_mul(dqv[ac]) >> log_scale;
        dqcoeff[i] = (abs_dq ^ sign).wrapping_sub(sign);
        if tmp32 != 0 {
            eob = eob.max(iscan[i] as i32 + 1);
        }
    }
    eob as u16
}

/// x86-64/AVX2 body for [`aom_highbd_quantize_b_no_qmatrix`], the highbd
/// sibling of `quantize_b_impl_v3` mirroring `aom_highbd_quantize_b_avx2`
/// (highbd_quantize_intrin_avx2.c): raster-order 8-lane walk, iscan max for
/// eob, dead-zone mask folded through the chain.
///
/// The highbd scalar keeps i64 intermediates where the lowbd one clamps to
/// i16, so the two multiplies can't stay in i32 lanes. On the guarded domain
/// (`tmp1 = abs + round` below `TMP1_MAX` — everything a real transform can
/// emit; 12-bit magnitudes top out near 2^23 — plus round/quant/quant_shift
/// >= 0):
/// ```text
/// tmp1  = (abs + round) & gate          i32 add, matches C's int add
/// tmp2  = ((tmp1*32)*quant)>>16 + tmp1*32
///       = (tmp1*quant)>>11 + tmp1<<5    fits i32 (<= TMP1_MAX*48 < 2^31)
/// tmp32 = (tmp2 * quant_shift) >> (21-log_scale)   low 32 kept, like scalar
/// ```
/// The products run `mul_epi32` on the even lanes and on the
/// `srli_epi64(.,32)` odd-lane view (nonneg, so no sign fixup), logical-shift
/// down, and re-narrow by `blend_epi32(prod_even, slli64(prod_odd,32), 0xAA)`.
/// A per-chunk unsigned-overflow check (`tmp1u > TMP1_MAX`, which also catches
/// the wrapped-negative sums — huge coeffs and `coeff == i32::MIN`) falls back
/// to the scalar body, so the kernel is exact on the FULL i32 domain.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn highbd_quantize_b_impl_v3(
    _t: archmage::X64V3Token,
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
    use archmage::intrinsics::x86_64::*;
    let n = coeff.len();
    let nb = n / 8 * 8;
    let wt = 1i64 << AOM_QM_BITS;
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let rr = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    let dqv = [
        ((dequant[0] as i32 * (1 << AOM_QM_BITS)) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
        ((dequant[1] as i32 * (1 << AOM_QM_BITS)) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
    ];
    let shift = 16 - log_scale + AOM_QM_BITS;

    // Preconditions for the i32-lane chain: shift counts must be in range
    // (outside, the scalar body keeps its own behaviour — including panics)
    // and the three multiplicands must be nonneg so the i64 products take
    // logical shifts.
    if !(0..=21).contains(&log_scale)
        || rr[0] < 0
        || rr[1] < 0
        || quant[0] < 0
        || quant[1] < 0
        || quant_shift[0] < 0
        || quant_shift[1] < 0
    {
        return highbd_quantize_b_scalar(
            zbin, round, quant, quant_shift, dequant, log_scale, iscan, coeff, qcoeff, dqcoeff,
        );
    }
    let sh_cnt = _mm_cvtsi32_si128(shift);
    let ls_cnt = _mm_cvtsi32_si128(log_scale);

    // tmp1 = abs + round must stay <= TMP1_MAX so tmp2 <= 48*tmp1 < 2^31
    // (see the doc). The check runs on the SUM — C computes the add in int,
    // so a wrapped-negative tmp1 must also route scalar (it does: unsigned-
    // compare of a negative i32 reads as huge).
    const TMP1_MAX: i32 = i32::MAX / 48; // 44_739_242
    let msb = _mm256_set1_epi32(i32::MIN);
    let lim = _mm256_set1_epi32(TMP1_MAX ^ i32::MIN);

    // Chunk-0 vectors carry the DC class in lane 0; the rest are all-AC.
    let lane0 = |dc: i32, ac: i32| _mm256_loadu_si256(&[dc, ac, ac, ac, ac, ac, ac, ac]);
    let zg_f = lane0(zbins[0] - 1, zbins[1] - 1);
    let rr_f = lane0(rr[0], rr[1]);
    let q_f = lane0(quant[0] as i32, quant[1] as i32);
    let qs_f = lane0(quant_shift[0] as i32, quant_shift[1] as i32);
    let dq_f = lane0(dqv[0], dqv[1]);
    let (zg_a, rr_a) = (_mm256_set1_epi32(zbins[1] - 1), _mm256_set1_epi32(rr[1]));
    let (q_a, qs_a) = (
        _mm256_set1_epi32(quant[1] as i32),
        _mm256_set1_epi32(quant_shift[1] as i32),
    );
    let dq_a = _mm256_set1_epi32(dqv[1]);
    // Odd-lane views of the (nonneg) multiplicands for `mul_epi32`.
    let q_f_o = _mm256_srli_epi64::<32>(q_f);
    let q_a_o = _mm256_srli_epi64::<32>(q_a);
    let qs_f_o = _mm256_srli_epi64::<32>(qs_f);
    let qs_a_o = _mm256_srli_epi64::<32>(qs_a);

    let one = _mm256_set1_epi32(1);
    let zero = _mm256_setzero_si256();
    let mut eob_v = zero;
    let mut eob: i32 = 0;

    /// Scalar body for one raster position — the overflow-guarded chunks and
    /// the sub-8 tail share it.
    macro_rules! one_pos {
        ($i:expr, $ac:expr) => {{
            let i = $i;
            let ac: usize = $ac;
            let c = coeff[i];
            let s = aomsign(c);
            let cw = c as i64 * wt;
            let t32 = if cw >= zbins[ac] as i64 * wt || cw <= -(zbins[ac] as i64 * wt) {
                // Same i32-wrap add as the scalar body: C's `abs_coeff +
                // round` is `int + int`, which wraps at extreme |coeff|.
                let ab = (c ^ s).wrapping_sub(s);
                let tw = (ab.wrapping_add(rr[ac]) as i64) * wt;
                let t2 = ((tw * quant[ac] as i64) >> 16) + tw;
                ((t2 * quant_shift[ac] as i64) >> shift) as i32
            } else {
                0
            };
            qcoeff[i] = (t32 ^ s).wrapping_sub(s);
            let ad = t32.wrapping_mul(dqv[ac]) >> log_scale;
            dqcoeff[i] = (ad ^ s).wrapping_sub(s);
            if t32 != 0 {
                eob = eob.max(iscan[i] as i32 + 1);
            }
        }};
    }

    macro_rules! chunk8 {
        ($ci:expr, $zg:expr, $rr:expr, $q:expr, $qo:expr, $qs:expr, $qso:expr, $dq:expr) => {{
            let ci = $ci;
            let c8: &[i32; 8] = coeff[ci * 8..ci * 8 + 8].try_into().unwrap();
            let cv = _mm256_loadu_si256(c8);
            let sign = _mm256_srai_epi32::<31>(cv);
            let abs = _mm256_sub_epi32(_mm256_xor_si256(cv, sign), sign);
            // add_epi32 IS C's `int + int` add — identical including the
            // wrap. The guard checks the SUM (unsigned tmp1 > TMP1_MAX),
            // which catches huge coeffs AND any lane whose add wrapped
            // negative (incl. the i32::MIN wrapped-abs lane): those run
            // scalar, where the i64 shifts handle the negative tmp1.
            let tmp1u = _mm256_add_epi32(abs, $rr);
            let over = _mm256_cmpgt_epi32(_mm256_xor_si256(tmp1u, msb), lim);
            if _mm256_movemask_epi8(over) != 0 {
                for i in ci * 8..ci * 8 + 8 {
                    one_pos!(i, if ci == 0 && i == 0 { 0 } else { 1 });
                }
            } else {
                let gate = _mm256_cmpgt_epi32(abs, $zg);
                let tmp1 = _mm256_and_si256(tmp1u, gate);
                // tmp1 * quant as i64 lanes, nonneg -> logical shifts.
                let pe = _mm256_mul_epi32(tmp1, $q);
                let po = _mm256_mul_epi32(_mm256_srli_epi64::<32>(tmp1), $qo);
                let prod = _mm256_blend_epi32::<0xAA>(
                    _mm256_srli_epi64::<11>(pe),
                    _mm256_slli_epi64::<32>(_mm256_srli_epi64::<11>(po)),
                );
                let tmp2 = _mm256_add_epi32(prod, _mm256_slli_epi32::<5>(tmp1));
                // tmp2 * quant_shift as i64, >> shift, keep low 32.
                let qe = _mm256_srl_epi64(_mm256_mul_epi32(tmp2, $qs), sh_cnt);
                let qo = _mm256_srl_epi64(
                    _mm256_mul_epi32(_mm256_srli_epi64::<32>(tmp2), $qso),
                    sh_cnt,
                );
                let t32 = _mm256_blend_epi32::<0xAA>(qe, _mm256_slli_epi64::<32>(qo));
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
            }
        }};
    }

    chunk8!(0, zg_f, rr_f, q_f, q_f_o, qs_f, qs_f_o, dq_f);
    for ci in 1..nb / 8 {
        chunk8!(ci, zg_a, rr_a, q_a, q_a_o, qs_a, qs_a_o, dq_a);
    }

    let mut eob_lanes = [0i32; 8];
    _mm256_storeu_si256(&mut eob_lanes, eob_v);
    eob = eob.max(eob_lanes.into_iter().max().unwrap_or(0));

    // Raster tail (< 8 coeffs, unreachable for real transform sizes): scalar.
    for i in nb..n {
        one_pos!(i, 1);
    }
    eob as u16
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

/// `av1_quantize_lp_c` (`av1/encoder/av1_quantize.c:214`): the low-precision
/// FP quantizer — i16 lanes end to end. `scan` orders the eob computation;
/// qcoeff/dqcoeff are written at RAW (`rc`) positions. round/quant/dequant
/// use row lane `[rc != 0]`. `iscan` is ignored exactly as `_c` ignores it
/// (`(void)iscan`, :219) — the SIMD tiers consume it instead; see
/// [`simd::av1_quantize_lp_dispatch`].
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_lp(
    coeff: &[i16],
    n_coeffs: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 8],
    scan: &[i16],
    iscan: &[i16],
) -> u16 {
    let _ = iscan;
    let mut eob: i32 = -1;
    qcoeff[..n_coeffs].fill(0);
    dqcoeff[..n_coeffs].fill(0);
    for (i, &sc) in scan[..n_coeffs].iter().enumerate() {
        let rc = sc as usize;
        let c = i32::from(coeff[rc]);
        let coeff_sign = c >> 31; // AOMSIGN
        let abs_coeff = (c ^ coeff_sign) - coeff_sign;
        let lane = usize::from(rc != 0);
        let mut tmp =
            (abs_coeff + i32::from(round_fp[lane])).clamp(i16::MIN as i32, i16::MAX as i32);
        tmp = (tmp * i32::from(quant_fp[lane])) >> 16;
        qcoeff[rc] = ((tmp ^ coeff_sign) - coeff_sign) as i16;
        dqcoeff[rc] = qcoeff[rc].wrapping_mul(dequant[lane]);
        if tmp != 0 {
            eob = i as i32;
        }
    }
    (eob + 1) as u16
}

/// aarch64 NEON tier for [`aom_quantize_b_no_qmatrix`] — a verbatim
/// transcription of `aom_quantize_b_neon` / `_32x32_neon` / `_64x64_neon`
/// (`av1/encoder/arm/quantize_neon.c`). C's NEON kernel is a different
/// algorithm from the AVX2 mirror above: `vqdmulh`/`vsraq` accumulation
/// instead of the i32-lane `mullo`/`mul_epi32` chain, `vmovn` truncating
/// loads, the all-AC vector broadcast with the DC lane inserted only when
/// the chunk-0 gate fires, and `eob = max(iscan[nz]) + 1` with a -1
/// sentinel. `quant_shift` is halved up-front at `log_scale == 0`
/// (`>> 1` before the `vqdmulh`, which doubles internally).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn quantize_b_impl_neon(
    t: archmage::NeonToken,
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
    let _ = t;
    use archmage::intrinsics::aarch64::*;

    let n = coeff.len();
    // C peels chunk 0 then strides 8; a %8 tail would be silently skipped
    // (production n is always a multiple of 8). Malformed shapes take the
    // scalar port, same policy as the v3 mirror.
    if n < 8 || n % 8 != 0 || iscan.len() < n || qcoeff.len() < n || dqcoeff.len() < n {
        return quantize_b_scalar(
            zbin, round, quant, quant_shift, dequant, log_scale, iscan, coeff, qcoeff, dqcoeff,
        );
    }
    let _ = scan;

    // `load_tran_low_to_s16q` — TRUNCATING vmovn narrow (mem_neon.h).
    let load8 = |c: &[i32]| -> int16x8_t {
        vcombine_s16(
            vmovn_s32(vld1q_s32(<&[i32; 4]>::try_from(&c[..4]).unwrap())),
            vmovn_s32(vld1q_s32(<&[i32; 4]>::try_from(&c[4..8]).unwrap())),
        )
    };
    // `store_s16q_to_tran_low` — sign-extending vmovl.
    let store8 = |dst: &mut [i32], v: int16x8_t| {
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut dst[..4]).unwrap(),
            vmovl_s16(vget_low_s16(v)),
        );
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut dst[4..8]).unwrap(),
            vmovl_s16(vget_high_s16(v)),
        )
    };

    let zero = vdupq_n_s16(0);
    let mut eobmax = vdupq_n_s16(-1);

    // Per-class constants: C's zbins/rounds are ROUND_POWER_OF_TWO'd by
    // log_scale (a no-op at ls=0); quant_shift is halved only at ls=0 (the
    // vqdmulh it feeds doubles internally).
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale) as i16,
        round_power_of_two(zbin[1] as i32, log_scale) as i16,
    ];
    let rounds = [
        round_power_of_two(round[0] as i32, log_scale) as i16,
        round_power_of_two(round[1] as i32, log_scale) as i16,
    ];
    let qsh = if log_scale == 0 {
        [quant_shift[0] >> 1, quant_shift[1] >> 1]
    } else {
        *quant_shift
    };
    let v_zbins_a = vdupq_n_s16(zbins[1]);
    let v_round_a = vdupq_n_s16(rounds[1]);
    let v_dequant_a = vdupq_n_s16(dequant[1]);
    let v_quant_a = vdupq_n_s16(quant[1]);
    let v_qshift_a = vdupq_n_s16(qsh[1]);
    let v_zbins0 = vsetq_lane_s16::<0>(zbins[0], v_zbins_a);

    // Chunk 0: gate on the DC-lane zbin vector; on fire, splice the DC
    // constants into lane 0 of the AC broadcasts (exactly C's vsetq order).
    {
        let c = load8(&coeff[..8]);
        let abs = vabsq_s16(c);
        let cond = vcgeq_s16(abs, v_zbins0);
        let nz_check = vget_lane_u64::<0>(vreinterpret_u64_u8(vmovn_u16(cond)));
        if nz_check != 0 {
            let r0 = vsetq_lane_s16::<0>(rounds[0], v_round_a);
            let q0 = vsetq_lane_s16::<0>(quant[0], v_quant_a);
            let dq0 = vsetq_lane_s16::<0>(dequant[0], v_dequant_a);
            let qs0 = vsetq_lane_s16::<0>(qsh[0], v_qshift_a);
            // Inline the kernel with the spliced vectors.
            let sign = vreinterpretq_s16_u16(vcltq_s16(c, zero));
            let mut tmp = vqaddq_s16(abs, r0);
            tmp = vsraq_n_s16::<1>(tmp, vqdmulhq_s16(tmp, q0));
            if log_scale == 2 {
                let ones =
                    vandq_s16(vshrq_n_s16::<14>(vmulq_s16(tmp, qs0)), vdupq_n_s16(1));
                tmp = vqdmulhq_s16(tmp, qs0);
                tmp = vaddq_s16(vshlq_s16(tmp, vdupq_n_s16(1)), ones);
            } else {
                tmp = vqdmulhq_s16(tmp, qs0);
            }
            let qc = vbslq_s16(cond, vsubq_s16(veorq_s16(tmp, sign), sign), zero);
            let dq_raw = if log_scale == 1 {
                vreinterpretq_s16_u16(vhaddq_u16(
                    vreinterpretq_u16_s16(vmulq_s16(tmp, dq0)),
                    vdupq_n_u16(0),
                ))
            } else if log_scale == 2 {
                vorrq_s16(
                    vshlq_n_s16::<13>(vqdmulhq_s16(tmp, dq0)),
                    vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_s16(
                        vmulq_s16(tmp, dq0),
                    ))),
                )
            } else {
                vmulq_s16(tmp, dq0)
            };
            let dqc = vbslq_s16(cond, vsubq_s16(veorq_s16(dq_raw, sign), sign), zero);
            store8(&mut qcoeff[..8], qc);
            store8(&mut dqcoeff[..8], dqc);
            let nz_mask = vandq_u16(vcgtq_s16(tmp, zero), cond);
            let isc = vld1q_s16(<&[i16; 8]>::try_from(&iscan[..8]).unwrap());
            let m = vmaxq_s16(isc, eobmax);
            eobmax = vbslq_s16(nz_mask, m, eobmax);
        } else {
            store8(&mut qcoeff[..8], zero);
            store8(&mut dqcoeff[..8], zero);
        }
    }

    // AC chunks: gate on the all-AC zbin broadcast.
    for i in 1..n / 8 {
        let c = load8(&coeff[i * 8..]);
        let abs = vabsq_s16(c);
        let cond = vcgeq_s16(abs, v_zbins_a);
        let nz_check = vget_lane_u64::<0>(vreinterpret_u64_u8(vmovn_u16(cond)));
        if nz_check != 0 {
            let sign = vreinterpretq_s16_u16(vcltq_s16(c, zero));
            let mut tmp = vqaddq_s16(abs, v_round_a);
            tmp = vsraq_n_s16::<1>(tmp, vqdmulhq_s16(tmp, v_quant_a));
            if log_scale == 2 {
                let ones = vandq_s16(
                    vshrq_n_s16::<14>(vmulq_s16(tmp, v_qshift_a)),
                    vdupq_n_s16(1),
                );
                tmp = vqdmulhq_s16(tmp, v_qshift_a);
                tmp = vaddq_s16(vshlq_s16(tmp, vdupq_n_s16(1)), ones);
            } else {
                tmp = vqdmulhq_s16(tmp, v_qshift_a);
            }
            let qc = vbslq_s16(cond, vsubq_s16(veorq_s16(tmp, sign), sign), zero);
            let dq_raw = if log_scale == 1 {
                vreinterpretq_s16_u16(vhaddq_u16(
                    vreinterpretq_u16_s16(vmulq_s16(tmp, v_dequant_a)),
                    vdupq_n_u16(0),
                ))
            } else if log_scale == 2 {
                vorrq_s16(
                    vshlq_n_s16::<13>(vqdmulhq_s16(tmp, v_dequant_a)),
                    vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_s16(
                        vmulq_s16(tmp, v_dequant_a),
                    ))),
                )
            } else {
                vmulq_s16(tmp, v_dequant_a)
            };
            let dqc = vbslq_s16(cond, vsubq_s16(veorq_s16(dq_raw, sign), sign), zero);
            store8(&mut qcoeff[i * 8..], qc);
            store8(&mut dqcoeff[i * 8..], dqc);
            let nz_mask = vandq_u16(vcgtq_s16(tmp, zero), cond);
            let isc =
                vld1q_s16(<&[i16; 8]>::try_from(&iscan[i * 8..i * 8 + 8]).unwrap());
            let m = vmaxq_s16(isc, eobmax);
            eobmax = vbslq_s16(nz_mask, m, eobmax);
        } else {
            store8(&mut qcoeff[i * 8..], zero);
            store8(&mut dqcoeff[i * 8..], zero);
        }
    }

    // get_max_eob: vmaxvq + 1; all-zero nz leaves the -1 sentinel -> eob 0.
    (vmaxvq_s16(eobmax) as u16).wrapping_add(1)
}
