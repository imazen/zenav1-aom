//! SIMD dispatch for the hot quantizers — bit-identical to the
//! scalar port at every tier, by construction AND by differential test
//! (`tests/quantize_fp_simd_diff.rs`).
//!
//! Structure (the aom-rs SIMD pattern):
//! * ONE generic kernel written against magetypes vector types, expanded by
//!   `#[magetypes(v3, neon, wasm128, -scalar)]` into per-tier `#[arcane]`
//!   variants (`_v3`/`_neon`/`_wasm128`).
//! * The `_scalar` variant is HAND-WRITTEN to call the transcribed scalar
//!   port verbatim ([`crate::quant::av1_quantize_fp_no_qmatrix`]) — so the
//!   `AOM_FORCE_SCALAR` pin (and non-SIMD platforms) run the exact
//!   C-differentially-validated transcription, not a re-formulation.
//! * The public entry applies the env pin, then `incant!` dispatches to the
//!   best available tier.
//!
//! # Bit-exactness argument (checked by the differential over FULL i32/i16
//! domains — adversarial inputs included, not just production tables)
//!
//! The scalar port computes per coefficient (i64 intermediates):
//! ```text
//! abs        = (coeff ^ sign) - sign                  (wrapping; i32::MIN stays i32::MIN)
//! gate       = (abs << (1+ls)) >= dequant[ac]         (i64 shift, no overflow)
//! abs_r      = clamp(abs + RP2(round[ac], ls), i16)   (i64 add)
//! tmp32      = (abs_r * quant[ac]) >> (16-ls)         (i64 mul/shift, |abs_r*quant| < 2^30)
//! qcoeff     = (tmp32 ^ sign) - sign                  (wrapping)
//! dqcoeff    = ((tmp32 *wrap dequant[ac]) >> ls ^ sign) - sign
//! eob        = 1 + max scan-index with tmp32 != 0
//! ```
//! The vector kernel reformulates only two steps, both exactly:
//! * the gate becomes `abs >= ceil(dequant / 2^(1+ls))` — the standard
//!   integer identity `(a << s) >= d  ⟺  a >= ceil(d / 2^s)` (exact for every
//!   `a` when `|d| < 2^15 << s` bounds hold, which `i16` dequant guarantees;
//!   the `ceil` is `(d + (1<<s) - 1) >> s`, exact for negative `d` too). The
//!   only negative `abs` is `coeff == i32::MIN`, and `i32::MIN < -32768 <=
//!   ceil(d/2^s)` so the gate is false in both formulations.
//! * the i64 `abs + rounding` add becomes `min(abs, 1<<17) + rounding` in i32:
//!   for `abs < 2^17` the i32 math is the i64 math; for `abs >= 2^17` both
//!   sums are `>= 2^17 - 2^15 > 32767`, so both clamp to 32767.
//! Everything else uses lane ops with the scalar port's exact semantics
//! (magetypes integer Mul/Sub are wrapping on every backend, matching the
//! port's `wrapping_mul`/`wrapping_sub`; `>>` is arithmetic).
//! `eob = 1 + max(iscan[rc])` over nonzero positions equals the scan-order
//! maximum because `iscan` is the inverse permutation of `scan`.

use archmage::prelude::*;

/// `av1_quantize_fp_no_qmatrix` with runtime SIMD dispatch. Bit-identical to
/// [`crate::quant::av1_quantize_fp_no_qmatrix`] (the differentially-validated scalar
/// port) at every dispatch tier; under `AOM_FORCE_SCALAR` it IS that function.
///
/// Same contract as the scalar port plus `iscan` (the inverse scan, from the
/// same `av1_scan_orders` row as `scan`; the SIMD tiers derive the EOB from it
/// while walking raster order). `coeff.len()` must be a multiple of 8 (every
/// AV1 transform block area is).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_no_qmatrix_dispatch(
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        quantize_fp_impl(quant, dequant, round, log_scale, scan, iscan, coeff, qcoeff, dqcoeff),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed port, verbatim (`iscan` unused there — the
/// scalar walk derives the EOB in scan order).
#[allow(clippy::too_many_arguments)]
fn quantize_fp_impl_scalar(
    _t: archmage::ScalarToken,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    _iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    crate::quant::av1_quantize_fp_no_qmatrix(
        quant, dequant, round, log_scale, scan, coeff, qcoeff, dqcoeff,
    )
}

// 256-bit kernel: the x8 generic types' backends are neon / wasm128.
// `-scalar` drops the macro's auto-appended scalar variant — the hand-written
// `_scalar` above (the transcribed port, verbatim) takes that slot instead —
// and `v3` is absent because a hand-written AVX2 body below takes it: an
// instruction-level mirror of upstream's REAL runtime kernels
// (`av1_quantize_fp_avx2`/`_32x32`/`_64x64`), which differ from the C scalar
// in i16-lane edge cases. See the `_v3` body's docs.
#[magetypes(define(i32x8), wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn quantize_fp_impl(
    token: Token,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    _scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    assert!(n % 8 == 0, "AV1 tx block areas are multiples of 8");
    assert!(iscan.len() >= n && qcoeff.len() >= n && dqcoeff.len() >= n);

    // Per-class (dc = index 0 / ac = index 1) scalar-side constants.
    let rounding = [
        crate::quant::round_power_of_two(round[0] as i32, log_scale),
        crate::quant::round_power_of_two(round[1] as i32, log_scale),
    ];
    // Gate threshold: (abs << (1+ls)) >= dequant  ⟺  abs >= ceil(dequant / 2^(1+ls)).
    let gs = 1 + log_scale;
    let thr = |d: i16| ((d as i32) + (1 << gs) - 1) >> gs;
    let thr_c = [thr(dequant[0]), thr(dequant[1])];

    // Lane-parameter vectors: chunk 0 has the DC coefficient in lane 0
    // (raster position 0); every other lane/chunk is AC.
    let mk = |dc: i32, ac: i32, first: bool| {
        if first {
            i32x8::from_array(token, [dc, ac, ac, ac, ac, ac, ac, ac])
        } else {
            i32x8::splat(token, ac)
        }
    };

    let (q_chunks, _) = i32x8::partition_slice_mut(token, &mut qcoeff[..n]);
    let (dq_chunks, _) = i32x8::partition_slice_mut(token, &mut dqcoeff[..n]);
    let zero = i32x8::zero(token);
    let abs_cap = i32x8::splat(token, 1 << 17);
    let clamp_lo = i32x8::splat(token, i16::MIN as i32);
    let clamp_hi = i32x8::splat(token, i16::MAX as i32);
    let mut eob_v = zero;

    // KB-PERF-36: only chunk 0 carries the DC lane, so the AC-only parameter
    // vectors are loop-invariant — building them once instead of four `splat`s
    // per chunk. Same values, same lanes: `mk(.., false)` IS `splat(ac)`.
    let (thr_ac, rnd_ac, qnt_ac, dqv_ac) = (
        i32x8::splat(token, thr_c[1]),
        i32x8::splat(token, rounding[1]),
        i32x8::splat(token, quant[1] as i32),
        i32x8::splat(token, dequant[1] as i32),
    );
    for ci in 0..n / 8 {
        let first = ci == 0;
        let thr_v = if first {
            mk(thr_c[0], thr_c[1], true)
        } else {
            thr_ac
        };
        let rnd_v = if first {
            mk(rounding[0], rounding[1], true)
        } else {
            rnd_ac
        };
        let qnt_v = if first {
            mk(quant[0] as i32, quant[1] as i32, true)
        } else {
            qnt_ac
        };
        let dqv_v = if first {
            mk(dequant[0] as i32, dequant[1] as i32, true)
        } else {
            dqv_ac
        };

        let c = i32x8::from_slice(token, &coeff[ci * 8..ci * 8 + 8]);
        // sign = c >> 31 (all-ones for negative); abs = (c ^ sign) - sign (wrapping).
        let sign = c.shr_arithmetic_const::<31>();
        let abs = (c ^ sign) - sign;
        let gate = abs.simd_ge(thr_v);

        // av1_quantize_fp_avx2's nzflag early-out (av1_quantize_avx2.c:207-222):
        // a chunk with no lane past the gate produces all-zero qcoeff/dqcoeff
        // and contributes nothing to the eob — identical output without the
        // multiply/shift/sign work or the iscan load. Most post-eob chunks of
        // a real block take this arm.
        if !gate.any_true() {
            zero.store(&mut q_chunks[ci]);
            zero.store(&mut dq_chunks[ci]);
            continue;
        }

        // abs_r = clamp(min(abs, 2^17) + rounding, i16::MIN, i16::MAX)
        let abs_r = (abs.min(abs_cap) + rnd_v).clamp(clamp_lo, clamp_hi);
        // tmp32 = (abs_r * quant) >> (16 - ls), gated to 0 outside the gate.
        let prod = abs_r * qnt_v; // |abs_r| <= 32768, |quant| <= 32767: exact in i32
        let tmp = match log_scale {
            0 => prod.shr_arithmetic_const::<16>(),
            1 => prod.shr_arithmetic_const::<15>(),
            2 => prod.shr_arithmetic_const::<14>(),
            _ => unreachable!("log_scale is 0/1/2"),
        };
        let tmp = i32x8::blend(gate, tmp, zero);

        // qcoeff = (tmp ^ sign) - sign; dqcoeff = ((tmp *wrap dq) >> ls ^ sign) - sign.
        let qc = (tmp ^ sign) - sign;
        let absdq = match log_scale {
            // `>> 0` is the identity; emit the value directly rather than
            // `shr_arithmetic_const::<0>()`. On x86 the const-0 shift is a legal
            // no-op, but magetypes forwards it to NEON's `vshrq_n_s32::<N>`,
            // whose `static_assert(1 <= N <= 32)` rejects N==0 at const-eval —
            // breaking the aarch64/windows-11-arm build. (Mirrors the `0 => $v`
            // guard the CDEF `shr_by!` macro already uses.) See aom-dsp #4.
            0 => tmp * dqv_v,
            1 => (tmp * dqv_v).shr_arithmetic_const::<1>(),
            2 => (tmp * dqv_v).shr_arithmetic_const::<2>(),
            _ => unreachable!(),
        };
        let dq = (absdq ^ sign) - sign;
        qc.store(&mut q_chunks[ci]);
        dq.store(&mut dq_chunks[ci]);

        // eob candidate: iscan[rc] + 1 where tmp != 0.
        let base = ci * 8;
        // KB-PERF-36: read the eight `iscan` entries as ONE fixed-size array so
        // the compiler sees a single bounds check per chunk rather than eight
        // per chunk — the same per-lane-load defect KB-PERF-34/35 measured at
        // ~0.2 pp apiece. Values and lanes are unchanged; the `unwrap_or` arm
        // is unreachable (`iscan.len() >= n` is asserted at entry) and exists
        // only to keep this panic-free.
        let raw: [i16; 8] = match iscan.get(base..base + 8).and_then(|s| s.try_into().ok()) {
            Some(a) => a,
            None => [0i16; 8],
        };
        let isc = i32x8::from_array(token, core::array::from_fn(|k| raw[k] as i32 + 1));
        let nz = tmp.simd_ne(zero);
        eob_v = eob_v.max(i32x8::blend(nz, isc, zero));
    }

    let mx = eob_v.to_array().into_iter().max().unwrap_or(0);
    mx as u16
}

/// x86-64/AVX2 tier: an instruction-level mirror of upstream's REAL runtime
/// kernels — `av1_quantize_fp_avx2`, `av1_quantize_fp_32x32_avx2` and
/// `av1_quantize_fp_64x64_avx2` (`av1/encoder/x86/av1_quantize_avx2.c`) — the
/// functions an AVX2 libaom build actually dispatches to through RTCD.
///
/// This is deliberately NOT bit-identical to the scalar port on the full i32
/// domain, because the C avx2 kernel is not bit-identical to `av1_quantize_fp_c`
/// there either — libaom's ENCODER is not arch-bit-exact, and the byte gates
/// run real `aomenc` on x86-64 where this avx2 kernel is what executes. The
/// edge cases where the two C variants differ are mirrored exactly:
///
/// * `packs_epi32` SATURATES each input coeff to i16 (scalar reads full i32);
/// * `abs_epi16` maps the saturated -32768 back to -32768;
/// * `adds_epi16` saturates `abs + round` (scalar clamps64 to the same bound);
/// * `dq` is a `mullo_epi16` product — i16-WRAPPED where the scalar keeps the
///   full i32 product (a real, if narrow, C-scalar-vs-avx2 divergence for
///   |q*dequant| >= 2^15, unreachable under the production quantizer tables
///   where |q*dequant| <= 32767 by the `quant = (1<<16)/dequant` relation);
/// * ls=1 uses `mulhi_epu16` (unsigned) and `dq = (abs_q*dequant mod 2^16)>>1`;
/// * ls=2 gates `tmp_rnd` with `& mask`, computes q/dq through the
///   mulhi<<k | mullo>>k reconstruction, and derives the eob nz mask from
///   `dq != 0` (the `psign` zeroing quirk), not `abs_q > 0`;
/// * the gate is `abs > (dequant >> (1+ls)) - 1` in i16 — which for odd
///   dequant admits the boundary `abs = dequant>>1` that the scalar's
///   `(abs << (1+ls)) >= dequant` rejects (still eob/q-inert on production
///   params: that lane quantizes to 0).
///
/// Chunk shape: 16 i16 lanes per iteration built by `packs_epi32` on two
/// i32x8 loads (giving C's [c0-3, c8-11 | c4-7, c12-15] lane order); the
/// sign-extend + unpack store dance writes i32 lanes back in natural order,
/// and `permute4x64(iscan, 0xD8)` aligns the eob gather to the same lanes.
/// `n % 16 != 0` (never on the encode path — every fp txb area is a multiple
/// of 16) falls back to the scalar transcription rather than over-reading,
/// which is what C's `while (n_coeffs > 0)` loop would do.
///
/// Differential: `tests/all/quantize_fp_avx2_diff.rs` pins this tier against
/// the exported `av1_quantize_fp{,_32x32,_64x64}_avx2` symbols over the FULL
/// i32 coeff / i16 table domain — the strongest oracle available, since the
/// mirror target is the dispatched C kernel itself.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn quantize_fp_impl_v3(
    t: archmage::X64V3Token,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    if n == 0
        || n % 16 != 0
        || log_scale < 0
        || log_scale > 2
        || iscan.len() < n
        || qcoeff.len() < n
        || dqcoeff.len() < n
    {
        return crate::quant::av1_quantize_fp_no_qmatrix(
            quant, dequant, round, log_scale, scan, coeff, qcoeff, dqcoeff,
        );
    }
    // C ships three log_scale-specialized kernels (`av1_quantize_fp_avx2`,
    // `_32x32`, `_64x64`) — `const int log_scale` per body. The const-generic
    // mirror keeps the same split: with `LS` a literal the per-chunk
    // `log_scale` selects, the `rt`/`qt` param math and the variable-count
    // threshold shift all const-fold to the one arm C compiles.
    match log_scale {
        0 => quantize_fp_v3_ls::<0>(t, quant, dequant, round, iscan, coeff, qcoeff, dqcoeff),
        1 => quantize_fp_v3_ls::<1>(t, quant, dequant, round, iscan, coeff, qcoeff, dqcoeff),
        _ => quantize_fp_v3_ls::<2>(t, quant, dequant, round, iscan, coeff, qcoeff, dqcoeff),
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn quantize_fp_v3_ls<const LS: i32>(
    _t: archmage::X64V3Token,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    use archmage::intrinsics::x86_64::*;

    let n = coeff.len();

    // init_qp (av1_quantize_avx2.c:30-55): the C param rows are
    // int16_t[8] with ac replicated through lane 7, so the 16-byte load +
    // unpackhi_epi64 broadcast yields [dc, ac x15] for chunk 0 and
    // [ac x16] after update_qp. Built here as broadcast+blend pairs: the dc
    // lane lives only in chunk 0's lane 0.
    let rt = |x: i16| -> i16 {
        // _mm_add_epi16 (wrapping) + _mm_srai_epi16 for log_scale > 0.
        if LS > 0 {
            x.wrapping_add(1i16 << (LS - 1)) >> LS
        } else {
            x
        }
    };
    let qt = |x: i16| -> i16 {
        // slli_epi16 by log_scale — only applied when log_scale == 1.
        if LS == 1 {
            x << 1
        } else {
            x
        }
    };
    // [dc, ac x15] / [ac x16] pairs. vpinsrw on a ymm touches ONLY lane 0 of
    // the low half (the imm has no half-select) — blend_epi16 would hit lane
    // 8 too, which is coeff index 4 in the packs lane order.
    let mk = |dc: i16, ac: i16| -> (__m256i, __m256i) {
        (
            _mm256_insert_epi16::<0>(_mm256_set1_epi16(ac), dc),
            _mm256_set1_epi16(ac),
        )
    };
    let (rnd0, rnd_a) = mk(rt(round[0]), rt(round[1]));
    let (qnt0, qnt_a) = mk(qt(quant[0]), qt(quant[1]));
    let (dqt0, dqt_a) = mk(dequant[0], dequant[1]);
    // threshold = (dequant >> (1+log_scale)) - 1, i16 arithmetic — C
    // computes it vector-side with the same ops.
    let sh = _mm_cvtsi32_si128(1 + LS);
    let one = _mm256_set1_epi16(1);
    let thr0 = _mm256_sub_epi16(_mm256_sra_epi16(dqt0, sh), one);
    let thr_a = _mm256_sub_epi16(_mm256_sra_epi16(dqt_a, sh), one);

    // View every buffer as fixed-width blocks over EXACTLY n elements —
    // truncating first lets LLVM see every `c8[2i]`/`q8[2i+1]`/`i16v[i]`
    // index is in-bounds for i < n/16, folding the bounds checks a
    // try_into-per-slice leaves behind. n % 16 == 0 was checked above, so
    // the as_chunks remainders are empty.
    let c8 = coeff[..n].as_chunks::<8>().0;
    let q8 = qcoeff[..n].as_chunks_mut::<8>().0;
    let d8 = dqcoeff[..n].as_chunks_mut::<8>().0;
    let i16v = iscan[..n].as_chunks::<16>().0;

    let zero = _mm256_setzero_si256();
    let mut eob_v = zero;

    // C's `quantize_fp` helper, one 16-coeff chunk. The first chunk runs the
    // [dc, ac x15] params; chunks after it run [ac x16] — C peels the same
    // way (the dc lane lives only in chunk 0).
    macro_rules! chunk {
        ($i:expr, $r_v:expr, $q_v:expr, $d_v:expr, $thr_v:expr) => {{
            let i = $i;
            // load_coefficients_avx2: packs_epi32 keeps C's permuted lane
            // order.
            let c = _mm256_packs_epi32(
                _mm256_loadu_si256(&c8[2 * i]),
                _mm256_loadu_si256(&c8[2 * i + 1]),
            );
            let abs = _mm256_abs_epi16(c);
            let mask = _mm256_cmpgt_epi16(abs, $thr_v);

            if _mm256_movemask_epi8(mask) == 0 {
                // write_zero x2 per array.
                _mm256_storeu_si256(&mut q8[2 * i], zero);
                _mm256_storeu_si256(&mut q8[2 * i + 1], zero);
                _mm256_storeu_si256(&mut d8[2 * i], zero);
                _mm256_storeu_si256(&mut d8[2 * i + 1], zero);
            } else {
                let (q16, dq16, nz);
                if LS == 0 {
                    // quantize_fp_16.
                    let tmp_rnd = _mm256_adds_epi16(abs, $r_v);
                    let abs_q = _mm256_mulhi_epi16(tmp_rnd, $q_v);
                    q16 = _mm256_sign_epi16(abs_q, c);
                    dq16 = _mm256_mullo_epi16(q16, $d_v);
                    nz = _mm256_cmpgt_epi16(abs_q, zero);
                } else if LS == 1 {
                    // quantize_fp_32x32.
                    let tmp_rnd = _mm256_adds_epi16(abs, $r_v);
                    let abs_q = _mm256_mulhi_epu16(tmp_rnd, $q_v);
                    q16 = _mm256_sign_epi16(abs_q, c);
                    let abs_dq = _mm256_srli_epi16::<1>(_mm256_mullo_epi16(abs_q, $d_v));
                    dq16 = _mm256_sign_epi16(abs_dq, c);
                    nz = _mm256_cmpgt_epi16(abs_q, zero);
                } else {
                    // quantize_fp_64x64.
                    let tmp_rnd = _mm256_and_si256(_mm256_adds_epi16(abs, $r_v), mask);
                    let qh = _mm256_slli_epi16::<2>(_mm256_mulhi_epi16(tmp_rnd, $q_v));
                    let ql = _mm256_srli_epi16::<14>(_mm256_mullo_epi16(tmp_rnd, $q_v));
                    let abs_q = _mm256_or_si256(qh, ql);
                    let dqh = _mm256_slli_epi16::<14>(_mm256_mulhi_epi16(abs_q, $d_v));
                    let dql = _mm256_srli_epi16::<2>(_mm256_mullo_epi16(abs_q, $d_v));
                    let abs_dq = _mm256_or_si256(dqh, dql);
                    q16 = _mm256_sign_epi16(abs_q, c);
                    dq16 = _mm256_sign_epi16(abs_dq, c);
                    // The z_mask quirk: eob tracks dq != 0, not abs_q > 0.
                    let z_mask = _mm256_cmpeq_epi16(dq16, zero);
                    nz = _mm256_cmpeq_epi16(z_mask, zero);
                }

                // store_coefficients_avx2: sign-extend i16 lanes to i32,
                // unpack un-permutes the packs lane order back to natural.
                let qs = _mm256_srai_epi16::<15>(q16);
                _mm256_storeu_si256(&mut q8[2 * i], _mm256_unpacklo_epi16(q16, qs));
                _mm256_storeu_si256(&mut q8[2 * i + 1], _mm256_unpackhi_epi16(q16, qs));
                let ds = _mm256_srai_epi16::<15>(dq16);
                _mm256_storeu_si256(&mut d8[2 * i], _mm256_unpacklo_epi16(dq16, ds));
                _mm256_storeu_si256(&mut d8[2 * i + 1], _mm256_unpackhi_epi16(dq16, ds));

                // get_max_lane_eob: permute4x64(0xD8) aligns iscan to the
                // packed lane order; iscan+1 is kept where the lane is
                // nonzero.
                let isc = _mm256_permute4x64_epi64::<0xD8>(_mm256_loadu_si256(&i16v[i]));
                let plus1 = _mm256_sub_epi16(isc, nz);
                eob_v = _mm256_max_epi16(eob_v, _mm256_and_si256(plus1, nz));
            }
        }};
    }

    chunk!(0, rnd0, qnt0, dqt0, thr0);
    for i in 1..n / 16 {
        chunk!(i, rnd_a, qnt_a, dqt_a, thr_a);
    }

    // quant_gather_eob's minpos fold equals a plain max: every lane holds
    // iscan+1 (>= 0) or 0, and all-zero yields 0 either way.
    let mut ea = [0i16; 16];
    let ea_ref: &mut [i16; 16] = &mut ea;
    _mm256_storeu_si256(ea_ref, eob_v);
    ea.into_iter().max().unwrap_or(0) as u16
}

/// `av1_quantize_lp` with runtime SIMD dispatch — the low-precision FP
/// quantizer `av1_block_yrd` runs on the nonrd estimate path
/// (TX_4X4/8x8/16x16 arms). Returns the eob.
///
/// `scan`/`iscan` are the C signature's pair: the scalar tier walks `scan`
/// order and ignores `iscan` (like `av1_quantize_lp_c`); the v3 tier walks
/// RASTER order like `av1_quantize_lp_avx2` and derives the eob from
/// `iscan` — equal because `iscan` is the inverse permutation.
///
/// # Tier agreement — MEASURED, not assumed
///
/// The C tiers differ in two corners, both PROVEN unreachable by
/// `nonrd_block_yrd_lp_diff::lp_quantize_tiers_agree_over_the_reachable_range`:
///
/// * every SIMD abs wraps `coeff == -32768` where `_c` computes +32768 —
///   above the lp transforms' ~32654 output bound;
/// * `_sse2`'s eob tests `dqcoeff != 0` where `_c`/`_avx2`/`_neon` test the
///   quantized magnitude — diverging only when `tmp*dequant == 0 mod 2^16`,
///   which an exhaustive hunt over all 1,536 real quantizer rows bounds at
///   `tmp*d <= 32768`, i.e. the wrap-to-zero product 65536 is unreachable.
///
/// So on every input the call site can produce, all four tiers — and this
/// dispatch — agree bit-exactly.
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_lp_dispatch(
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i16],
    n_coeffs: usize,
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
) -> u16 {
    let _ = crate::dispatch::scalar_forced();
    incant!(
        quantize_lp_impl(
            round_fp, quant_fp, dequant, scan, iscan, coeff, n_coeffs, qcoeff, dqcoeff
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed `av1_quantize_lp_c` port, verbatim.
#[allow(clippy::too_many_arguments)]
fn quantize_lp_impl_scalar(
    _t: archmage::ScalarToken,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i16],
    n_coeffs: usize,
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
) -> u16 {
    crate::quant::av1_quantize_lp(
        coeff, n_coeffs, round_fp, quant_fp, qcoeff, dqcoeff, dequant, scan, iscan,
    )
}

/// Non-x86 tiers: the `_c` semantics in raster order — i32-domain abs (no
/// i16 wrap at -32768; `_c` agrees) and the eob folded from `iscan` (the
/// inverse-permutation identity `max_{tmp!=0} iscan[rc]+1 == scan-order eob`).
/// LLVM lowers the per-lane math under the tier's target features; the
/// raster walk removes the scan-order serial dependence C's scalar has.
#[magetypes(wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn quantize_lp_impl(
    _t: Token,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
    _scan: &[i16],
    iscan: &[i16],
    coeff: &[i16],
    n_coeffs: usize,
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
) -> u16 {
    let n = n_coeffs;
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let mut eob = 0i32;
    for rc in 0..n {
        let c = i32::from(coeff[rc]);
        let sign = c >> 31;
        let abs = (c ^ sign) - sign;
        let lane = usize::from(rc != 0);
        let tmp = ((abs + i32::from(round_fp[lane])).clamp(i16::MIN as i32, i16::MAX as i32)
            * i32::from(quant_fp[lane]))
            >> 16;
        let q = ((tmp ^ sign) - sign) as i16;
        qcoeff[rc] = q;
        dqcoeff[rc] = q.wrapping_mul(dequant[lane]);
        if tmp != 0 {
            eob = eob.max(i32::from(iscan[rc]) + 1);
        }
    }
    eob as u16
}

/// v3 mirror of `av1_quantize_lp_avx2` (`av1/encoder/x86/av1_quantize_avx2.c:147`)
/// — the kernel RTCD dispatches on this host. Instruction-level:
/// `abs_epi16` / `adds_epi16` / `mulhi_epi16` / `sign_epi16` / `mullo_epi16`,
/// nz test `cmpgt(abs_q, 0)`, eob gathered from `iscan` as
/// `(iscan - nz) & nz` under `max_epi16`. Bit-compatible with the C AVX2
/// kernel on the FULL i16 domain — including the -32768 wrapping-abs corner —
/// which `quantize_lp_avx2_diff`-style gates pin against the exported symbol.
///
/// The parameter vectors replicate C's permutes exactly rather than assuming
/// the table fill: the first chunk runs `[row0..7 | row4..7, row4..7]` (the
/// `permute4x64(.., 0x54)` shape — dc at lane 0, ac at 1..7 and repeated),
/// later chunks run `[row4..7 x4]` (the `permute2x128(.., 0x31)` hi-half
/// broadcast). On the production rows lanes 1..7 all equal the ac value, so
/// this is C's own parameter layout either way.
///
/// `n < 16` or `n % 16 != 0` would overread C's fixed 16-lane chunks; those
/// shapes take the `_c` result instead (the lp path only ever calls
/// n = 16/64/256).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn quantize_lp_impl_v3(
    _t: archmage::X64V3Token,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i16],
    n_coeffs: usize,
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
) -> u16 {
    use archmage::intrinsics::x86_64::*;

    let n = n_coeffs;
    if n < 16
        || n % 16 != 0
        || coeff.len() < n
        || iscan.len() < n
        || qcoeff.len() < n
        || dqcoeff.len() < n
    {
        return crate::quant::av1_quantize_lp(
            coeff, n_coeffs, round_fp, quant_fp, qcoeff, dqcoeff, dequant, scan, iscan,
        );
    }

    // C's param construction, mirrored: chunk 0 sees [r0..r7 | r4..r7 x2],
    // AC chunks see [r4..r7 x4].
    let mk = |row: &[i16; 8]| -> (__m256i, __m256i) {
        let first: [i16; 16] = [
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[4], row[5], row[6],
            row[7], row[4], row[5], row[6], row[7],
        ];
        let ac: [i16; 16] = [
            row[4], row[5], row[6], row[7], row[4], row[5], row[6], row[7], row[4], row[5], row[6],
            row[7], row[4], row[5], row[6], row[7],
        ];
        (_mm256_loadu_si256(&first), _mm256_loadu_si256(&ac))
    };
    let (rnd0, rnd_a) = mk(round_fp);
    let (qnt0, qnt_a) = mk(quant_fp);
    let (dqt0, dqt_a) = mk(dequant);

    let zero = _mm256_setzero_si256();
    let mut eob = zero;

    let c16 = coeff[..n].as_chunks::<16>().0;
    let q16 = qcoeff[..n].as_chunks_mut::<16>().0;
    let d16 = dqcoeff[..n].as_chunks_mut::<16>().0;
    let i16v = iscan[..n].as_chunks::<16>().0;

    for i in 0..n / 16 {
        let (r_v, q_v, d_v) = if i == 0 {
            (rnd0, qnt0, dqt0)
        } else {
            (rnd_a, qnt_a, dqt_a)
        };
        // quantize_lp_16{,_first}: abs, sat-add round, mulhi, sign, mullo.
        let c = _mm256_loadu_si256(&c16[i]);
        let abs = _mm256_abs_epi16(c);
        let tmp_rnd = _mm256_adds_epi16(abs, r_v);
        let abs_q = _mm256_mulhi_epi16(tmp_rnd, q_v);
        let q = _mm256_sign_epi16(abs_q, c);
        let dq = _mm256_mullo_epi16(q, d_v);
        _mm256_storeu_si256(&mut q16[i], q);
        _mm256_storeu_si256(&mut d16[i], dq);

        // nz = abs_q > 0; eob candidate = (iscan + 1) & nz.
        let nz = _mm256_cmpgt_epi16(abs_q, zero);
        let isc = _mm256_loadu_si256(&i16v[i]);
        let nz_iscan = _mm256_and_si256(_mm256_sub_epi16(isc, nz), nz);
        eob = _mm256_max_epi16(eob, nz_iscan);
    }

    // accumulate_eob256 — the max fold, lane order as C.
    let lo = _mm256_castsi256_si128(eob);
    let hi = _mm256_extracti128_si256::<1>(eob);
    let e = _mm_max_epi16(lo, hi);
    let e = _mm_max_epi16(e, _mm_shuffle_epi32::<0xe>(e));
    let e = _mm_max_epi16(e, _mm_shufflelo_epi16::<0xe>(e));
    let e = _mm_max_epi16(e, _mm_shufflelo_epi16::<0x1>(e));
    _mm_extract_epi16::<1>(e) as u16
}

// ---------------------------------------------------------------------------
// aarch64 NEON tier — a verbatim transcription of `av1_quantize_fp_neon` /
// `_32x32_neon` / `_64x64_neon` and `av1_quantize_lp_neon`
// (av1/encoder/arm/quantize_neon.c).
//
// This is NOT the AVX2 body on NEON registers: C's NEON kernel is a
// different algorithm — `vqdmulhq` (doubling, saturating mulhi) with a
// compensating shift where AVX2 uses `mulhi`; a TRUNCATING `vmovn`
// i32->i16 narrow where AVX2's `packs` saturates; a backward zbin
// pre-scan + memset for `log_scale > 0`; `eob = max(iscan[nz]) + 1` with a
// -1 sentinel where AVX2 keeps `iscan + 1` masked; and 8-lane chunks, not
// 16. All of those differences are real on the adversarial domain, so the
// tier oracle is the exported C NEON symbol, not the C scalar port.
//
// Layout notes: `qcoeff`/`dqcoeff` are `tran_low_t` (i32) buffers — loads
// truncate pairs of i32 lanes through `vmovn`, stores sign-extend through
// `vmovl`. The `[i16; 2]` param pairs expand to C's int16_t[8] rows
// ([dc, ac x7]); the fp path's AC param vector is the lane-1 broadcast the
// C code makes after chunk 0.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn quantize_fp_impl_neon(
    t: NeonToken,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
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
    // Chunk requirements: the ls=0 body peels 8 then do-whiles (n >= 16 or
    // it would overread); the logscale body asserts n_coeffs > 16 and scans
    // backward in 16-coeff groups — n % 16 != 0 would under-read, while
    // n == 16 runs the assert-violating-but-defined path the shipped kernel
    // actually executes (the assert is NDEBUG-compiled out in the oracle).
    // Malformed shapes take the scalar port, same policy as the v3 mirror.
    if n % 8 != 0
        || n < 16
        || (log_scale > 0 && n % 16 != 0)
        || iscan.len() < n
        || qcoeff.len() < n
        || dqcoeff.len() < n
    {
        return crate::quant::av1_quantize_fp_no_qmatrix(
            quant, dequant, round, log_scale, scan, coeff, qcoeff, dqcoeff,
        );
    }

    let zero = vdupq_n_s16(0);
    let neg1 = vdupq_n_s16(-1);
    // C loads the int16_t[8] param rows — [dc, ac x7] — then broadcasts lane
    // 1 (the ac value) after chunk 0.
    let mut v_quant = vsetq_lane_s16::<0>(quant[0], vdupq_n_s16(quant[1]));
    let mut v_dequant = vsetq_lane_s16::<0>(dequant[0], vdupq_n_s16(dequant[1]));
    let mut v_round = vsetq_lane_s16::<0>(round[0], vdupq_n_s16(round[1]));
    let mut eobmax = neg1;

    // `load_tran_low_to_s16q` (mem_neon.h) — TRUNCATING vmovn narrow.
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
    // `get_max_lane_eob`: nz lanes contribute iscan, else -1.
    let lane_eob = |iscan8: &[i16; 8], eobmax: int16x8_t, mask: uint16x8_t| -> int16x8_t {
        vmaxq_s16(eobmax, vbslq_s16(mask, vld1q_s16(iscan8), neg1))
    };

    let i8v = iscan[..n].as_chunks::<8>().0;
    let mut chunks_done = 0usize;
    let mut nonzero_count = n;

    macro_rules! chunk {
        // quantize_fp_8 — the ls=0 core.
        (fp8, $i:expr) => {{
            let i = $i;
            let c = load8(&coeff[i * 8..]);
            let sign = vshrq_n_s16::<15>(c);
            let abs = vabsq_s16(c);
            let tmp = vqaddq_s16(abs, v_round);
            let tmp2 = vshrq_n_s16::<1>(vqdmulhq_s16(tmp, v_quant));
            let nz = vcgtq_s16(tmp2, zero);
            let qc = vsubq_s16(veorq_s16(tmp2, sign), sign);
            let dqc = vmulq_s16(qc, v_dequant);
            store8(&mut qcoeff[i * 8..], qc);
            store8(&mut dqcoeff[i * 8..], dqc);
            eobmax = lane_eob(&i8v[i], eobmax, nz);
        }};
        // quantize_fp_logscale_8 — the ls=1 core (also reached at ls=0 in
        // no_qmatrix form, but the dispatched ls=0 kernel is fp8 above).
        (ls8, $i:expr, $ls:expr) => {{
            let i = $i;
            let c = load8(&coeff[i * 8..]);
            let sign = vshrq_n_s16::<15>(c);
            let abs = vabsq_s16(c);
            let mask = vcgeq_s16(abs, vshlq_s16(v_dequant, vdupq_n_s16((-(1 + $ls)) as i16)));
            let tmp = vandq_s16(vqaddq_s16(abs, v_round), vreinterpretq_s16_u16(mask));
            let tmp2 = vqdmulhq_s16(vshlq_s16(tmp, vdupq_n_s16(($ls - 1) as i16)), v_quant);
            let nz = vcgtq_s16(tmp2, zero);
            let qc = vsubq_s16(veorq_s16(tmp2, sign), sign);
            let abs_dq = vshlq_u16(
                vreinterpretq_u16_s16(vmulq_s16(tmp2, v_dequant)),
                vdupq_n_s16((-$ls) as i16),
            );
            let dqc = vsubq_s16(veorq_s16(vreinterpretq_s16_u16(abs_dq), sign), sign);
            store8(&mut qcoeff[i * 8..], qc);
            store8(&mut dqcoeff[i * 8..], dqc);
            eobmax = lane_eob(&i8v[i], eobmax, nz);
        }};
        // quantize_fp_logscale2_8 — the ls=2 core (wider product kept as
        // shifted hi|lo halves instead of a vqdmulh shortcut).
        (ls2, $i:expr) => {{
            let i = $i;
            let c = load8(&coeff[i * 8..]);
            let sign = vshrq_n_s16::<15>(c);
            let abs = vabsq_s16(c);
            let mask = vcgeq_u16(
                vshlq_n_u16::<1>(vreinterpretq_u16_s16(abs)),
                vshrq_n_u16::<2>(vreinterpretq_u16_s16(v_dequant)),
            );
            let tmp = vandq_s16(vqaddq_s16(abs, v_round), vreinterpretq_s16_u16(mask));
            let tmp2 = vorrq_s16(
                vshlq_n_s16::<1>(vqdmulhq_s16(tmp, v_quant)),
                vreinterpretq_s16_u16(vshrq_n_u16::<14>(vreinterpretq_u16_s16(vmulq_s16(
                    tmp, v_quant,
                )))),
            );
            let nz = vcgtq_s16(tmp2, zero);
            let qc = vsubq_s16(veorq_s16(tmp2, sign), sign);
            let abs_dq = vorrq_s16(
                vshlq_n_s16::<13>(vqdmulhq_s16(tmp2, v_dequant)),
                vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_s16(vmulq_s16(
                    tmp2, v_dequant,
                )))),
            );
            let dqc = vsubq_s16(veorq_s16(abs_dq, sign), sign);
            store8(&mut qcoeff[i * 8..], qc);
            store8(&mut dqcoeff[i * 8..], dqc);
            eobmax = lane_eob(&i8v[i], eobmax, nz);
        }};
    }

    if log_scale == 0 {
        // av1_quantize_fp_neon: chunk 0 with the dc row, the rest on the ac
        // broadcast.
        chunk!(fp8, 0);
        v_quant = vdupq_lane_s16::<1>(vget_low_s16(v_quant));
        v_dequant = vdupq_lane_s16::<1>(vget_low_s16(v_dequant));
        v_round = vdupq_lane_s16::<1>(vget_low_s16(v_round));
        for i in 1..n / 8 {
            chunk!(fp8, i);
        }
        let _ = &mut chunks_done;
    } else {
        // quantize_fp_no_qmatrix_neon: scale the round first, then a backward
        // zbin pre-scan dropping trailing all-below-threshold 16-groups.
        v_round = vqrdmulhq_n_s16(v_round, (1i32 << (15 - log_scale)) as i16);
        let zbin = vdupq_lane_s16::<1>(vget_low_s16(vshlq_s16(
            v_dequant,
            vdupq_n_s16((-(1 + log_scale)) as i16),
        )));
        let mut i = n;
        while i > 0 {
            let a = vabsq_s16(load8(&coeff[i - 8..i]));
            let b = vabsq_s16(load8(&coeff[i - 16..i - 8]));
            let ma = vcgeq_s16(a, zbin);
            let mb = vcgeq_s16(b, zbin);
            // horizontal_long_add_u16x8 == 0: neither 8-group has a lane at
            // or above the zbin threshold.
            if vaddlvq_u16(ma) + vaddlvq_u16(mb) == 0 {
                nonzero_count -= 16;
            } else {
                break;
            }
            i -= 16;
        }
        qcoeff[nonzero_count..n].fill(0);
        dqcoeff[nonzero_count..n].fill(0);

        if log_scale == 2 {
            chunk!(ls2, 0);
        } else {
            chunk!(ls8, 0, log_scale);
        }
        v_quant = vdupq_lane_s16::<1>(vget_low_s16(v_quant));
        v_dequant = vdupq_lane_s16::<1>(vget_low_s16(v_dequant));
        v_round = vdupq_lane_s16::<1>(vget_low_s16(v_round));
        for i in 1..nonzero_count / 8 {
            if log_scale == 2 {
                chunk!(ls2, i);
            } else {
                chunk!(ls8, i, log_scale);
            }
        }
        let _ = &mut chunks_done;
    }

    // get_max_eob: vmaxvq + 1; all-zero nz leaves the -1 sentinel -> eob 0.
    (vmaxvq_s16(eobmax) as u16).wrapping_add(1)
}

/// aarch64 NEON tier for `av1_quantize_lp_dispatch` — verbatim
/// `av1_quantize_lp_neon` (i16 in, i16 out, 8-lane chunks, `qdmulh >> 1`,
/// `eob = max(iscan[nz]) + 1`). The `[i16; 8]` param rows load as-is —
/// chunk 0 runs the literal row, AC chunks run the lane-1 broadcast.
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn quantize_lp_impl_neon(
    t: NeonToken,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i16],
    n_coeffs: usize,
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
) -> u16 {
    let _ = t;
    use archmage::intrinsics::aarch64::*;

    let n = n_coeffs;
    // C peels 8 then do-whiles; n < 16 or n % 8 != 0 would overread.
    if n < 16
        || n % 8 != 0
        || coeff.len() < n
        || iscan.len() < n
        || qcoeff.len() < n
        || dqcoeff.len() < n
    {
        return crate::quant::av1_quantize_lp(
            coeff, n_coeffs, round_fp, quant_fp, qcoeff, dqcoeff, dequant, scan, iscan,
        );
    }

    let zero = vdupq_n_s16(0);
    let neg1 = vdupq_n_s16(-1);
    let mut v_quant = vld1q_s16(quant_fp);
    let mut v_dequant = vld1q_s16(dequant);
    let mut v_round = vld1q_s16(round_fp);
    let mut eobmax = neg1;

    let c8 = coeff[..n].as_chunks::<8>().0;
    let q8 = qcoeff[..n].as_chunks_mut::<8>().0;
    let d8 = dqcoeff[..n].as_chunks_mut::<8>().0;
    let i8v = iscan[..n].as_chunks::<8>().0;

    macro_rules! chunk {
        ($i:expr) => {{
            let i = $i;
            // quantize_lp_8.
            let c = vld1q_s16(&c8[i]);
            let sign = vshrq_n_s16::<15>(c);
            let abs = vabsq_s16(c);
            let tmp = vqaddq_s16(abs, v_round);
            let tmp2 = vshrq_n_s16::<1>(vqdmulhq_s16(tmp, v_quant));
            let nz = vcgtq_s16(tmp2, zero);
            let qc = vsubq_s16(veorq_s16(tmp2, sign), sign);
            let dqc = vmulq_s16(qc, v_dequant);
            vst1q_s16(&mut q8[i], qc);
            vst1q_s16(&mut d8[i], dqc);
            eobmax = vmaxq_s16(eobmax, vbslq_s16(nz, vld1q_s16(&i8v[i]), neg1));
        }};
    }

    chunk!(0);
    v_quant = vdupq_lane_s16::<1>(vget_low_s16(v_quant));
    v_dequant = vdupq_lane_s16::<1>(vget_low_s16(v_dequant));
    v_round = vdupq_lane_s16::<1>(vget_low_s16(v_round));
    for i in 1..n / 8 {
        chunk!(i);
    }

    (vmaxvq_s16(eobmax) as u16).wrapping_add(1)
}
