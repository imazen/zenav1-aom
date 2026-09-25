//! SIMD-vs-C differential for `aom_highbd_quantize_b_no_qmatrix` at every
//! dispatch tier — qcoeff, dqcoeff, AND eob — under
//! `archmage::testing::for_each_token_permutation`.
//!
//! Domain: production-shaped tables plus FULL adversarial i32 coefficients
//! (i32::MIN/MAX and lanes past the v3 kernel's `tmp1` guard), which the
//! per-chunk overflow fallback routes to the scalar body — asserting that the
//! SIMD path is exact on the whole domain, not just the vectorized interior.
//! The oracle is the REAL C `aom_highbd_quantize_b_helper_c` — note its
//! `abs_coeff + round` is `int + int` (i32 wrap at extremes), which the port
//! reproduces via `wrapping_add`.

use aom_dsp::quant::aom_highbd_quantize_b_no_qmatrix;
use aom_sys_ref as c;
use archmage::testing::{for_each_token_permutation, CompileTimePolicy};
use archmage::SimdToken;

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
    fn i16_any(&mut self) -> i16 {
        self.next() as i16
    }
    fn pos_i16(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
    }
}

/// Random permutation of 0..n (a valid scan order) + its inverse.
fn perm_pair(rng: &mut Rng, n: usize) -> (Vec<i16>, Vec<i16>) {
    let mut v: Vec<i16> = (0..n as i16).collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    let mut inv = vec![0i16; n];
    for (i, &rc) in v.iter().enumerate() {
        inv[rc as usize] = i as i16;
    }
    (v, inv)
}

#[allow(clippy::too_many_arguments)]
fn assert_case(
    label: &str,
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
) {
    let n = coeff.len();
    let mut q_got = vec![0i32; n];
    let mut dq_got = vec![0i32; n];
    let eob_got = aom_highbd_quantize_b_no_qmatrix(
        zbin,
        round,
        quant,
        quant_shift,
        dequant,
        log_scale,
        scan,
        iscan,
        coeff,
        &mut q_got,
        &mut dq_got,
    );
    let (q_ref, dq_ref, eob_ref) = c::ref_highbd_quantize_b(
        log_scale,
        coeff,
        zbin,
        round,
        quant,
        quant_shift,
        dequant,
        scan,
    );
    assert_eq!(eob_got, eob_ref, "{label}: eob");
    assert_eq!(q_got, q_ref, "{label}: qcoeff\ncoeff={coeff:?}");
    assert_eq!(dq_got, dq_ref, "{label}: dqcoeff\ncoeff={coeff:?}");
}

#[test]
fn highbd_quantize_b_simd_bit_identical_to_c_at_every_tier() {
    // Serialise: this sweep permutes PROCESS-GLOBAL dispatch state.
    let _serial = crate::dispatch_serial::dispatch_serial();
    // n < 8 hits the early scalar path; n = 10 exercises the <8 raster tail;
    // the rest are the real transform-block areas.
    let sizes = [4usize, 10, 16, 64, 256, 1024];
    let mut simd_perms = 0usize;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        if archmage::X64V3Token::summon().is_some() {
            simd_perms += 1;
        }
        let mut rng = Rng(0x_9e37_b111_7f4a_7c15);
        for &n in &sizes {
            let (scan, iscan) = perm_pair(&mut rng, n);
            for ls in 0..3i32 {
                // Production-shaped tables (positive, av1_build_quantizer-like).
                for rep in 0..4 {
                    let zbin = [rng.pos_i16(1, 1500), rng.pos_i16(1, 1500)];
                    let round = [rng.pos_i16(1, 2000), rng.pos_i16(1, 2000)];
                    let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                    let qshift = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                    let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];
                    let coeff: Vec<i32> = (0..n)
                        .map(|_| (rng.next() % (1 << 23)) as i32 - (1 << 22))
                        .collect();
                    assert_case(
                        &format!("[{tier}] prod n={n} ls={ls} rep={rep}"),
                        &zbin,
                        &round,
                        &quant,
                        &qshift,
                        &dequant,
                        ls,
                        &scan,
                        &iscan,
                        &coeff,
                    );
                }
                // Adversarial: full-range i32 coeffs — every chunk trips the
                // tmp1 guard onto the scalar path. x86-64 only: past 2^26
                // `_c` is signed-overflow UB whose x86 clang shape the port
                // mirrors (e0a1b95); aarch64 clang differs (CI 2026-09-25,
                // KB-ARM-FLOAT root #3 precedent).
                for rep in 0..if cfg!(target_arch = "x86_64") { 4 } else { 0 } {
                    let zbin = [rng.pos_i16(1, 1500), rng.pos_i16(1, 1500)];
                    let round = [rng.pos_i16(1, 2000), rng.pos_i16(1, 2000)];
                    let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                    let qshift = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                    let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];
                    let mut coeff: Vec<i32> = (0..n).map(|_| rng.next() as i32).collect();
                    coeff[0] = i32::MIN;
                    coeff[n / 3] = i32::MAX;
                    coeff[2 * n / 3] = -(1 << 27);
                    coeff[n / 2] = (1 << 26) + 7;
                    assert_case(
                        &format!("[{tier}] adv n={n} ls={ls} rep={rep}"),
                        &zbin,
                        &round,
                        &quant,
                        &qshift,
                        &dequant,
                        ls,
                        &scan,
                        &iscan,
                        &coeff,
                    );
                }
                // Threshold straddles: coeff magnitudes right at the tmp1
                // guard (44_739_242) and the zbin dead-zone edges.
                let zbin = [rng.pos_i16(1, 1500), rng.pos_i16(1, 1500)];
                let round = [rng.pos_i16(1, 2000), rng.pos_i16(1, 2000)];
                let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                let qshift = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];
                let mut edge = vec![0i32; n];
                edge[0] = i32::MAX / 48; // tmp1 at the guard bound
                edge[1 % n] = i32::MAX / 48 + 1;
                if n > 3 {
                    edge[2] = -(i32::MAX / 48);
                    edge[3] = -(i32::MAX / 48) - 1;
                }
                if n > 8 && cfg!(target_arch = "x86_64") {
                    // The two UB lanes (see the adversarial arm above).
                    edge[n - 1] = i32::MAX;
                    edge[n - 2] = i32::MIN;
                }
                assert_case(
                    &format!("[{tier}] edges n={n} ls={ls}"),
                    &zbin,
                    &round,
                    &quant,
                    &qshift,
                    &dequant,
                    ls,
                    &scan,
                    &iscan,
                    &edge,
                );
                // Full-range i16 tables (zero/negative values that
                // av1_build_quantizer never produces) trip the preconditions.
                if rep_tables(&mut rng) {
                    let zbin = [rng.i16_any(), rng.i16_any()];
                    let round = [rng.i16_any(), rng.i16_any()];
                    let quant = [rng.i16_any(), rng.i16_any()];
                    let qshift = [rng.i16_any(), rng.i16_any()];
                    let dequant = [rng.i16_any(), rng.i16_any()];
                    let coeff: Vec<i32> = (0..n)
                        .map(|_| (rng.next() % (1 << 20)) as i32 - (1 << 19))
                        .collect();
                    assert_case(
                        &format!("[{tier}] advtbl n={n} ls={ls}"),
                        &zbin,
                        &round,
                        &quant,
                        &qshift,
                        &dequant,
                        ls,
                        &scan,
                        &iscan,
                        &coeff,
                    );
                }
            }
        }
    });
    eprintln!("highbd quantize_b SIMD parity: {report}");
    assert!(
        simd_perms >= 1,
        "the v3/AVX2 permutation must run at least once — a passing run with \
         zero vector permutations compares the scalar path against itself"
    );
    assert!(
        report.permutations_run >= 2,
        "expected at least all-enabled + all-disabled tiers to run"
    );
}

fn rep_tables(rng: &mut Rng) -> bool {
    rng.next() % 3 == 0
}
