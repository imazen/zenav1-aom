//! Differential for the port's AVX2 tier of `av1_quantize_fp_no_qmatrix` vs
//! the REAL exported C kernels `av1_quantize_fp{,_32x32,_64x64}_avx2` — the
//! functions an x86-64 libaom build actually dispatches to.
//!
//! Why this exists separately from `quantize_fp_simd_diff.rs`: the C avx2
//! kernel is NOT bit-identical to `av1_quantize_fp_c` on the full i32 domain
//! (packs_epi32 input saturation, i16-wrapped dq products, the ls=2 `dq != 0`
//! eob quirk). libaom's encoder is not arch-bit-exact; the byte gates run
//! real aomenc on x86-64, so the port's v3 tier mirrors the AVX2 kernel — and
//! this test pins it against the exported symbol on the FULL adversarial
//! domain (any i32 coeff incl. MIN/MAX, any i16 tables), which is only
//! possible because the port kernel is an instruction-level mirror.

use aom_dsp::quant::simd::av1_quantize_fp_no_qmatrix_dispatch;
use aom_sys_ref as c;
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
    fn i32_any(&mut self) -> i32 {
        self.next() as i32
    }
    fn i16_any(&mut self) -> i16 {
        self.next() as i16
    }
    fn pos_i16(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
    }
}

/// Random permutation of 0..n plus its inverse (real scan-order shapes).
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

#[test]
fn quantize_fp_v3_bit_identical_to_real_avx2_full_domain() {
    if archmage::X64V3Token::summon().is_none() {
        eprintln!("no v3 token on this host; skipping");
        return;
    }
    let mut rng = Rng(0x5eed_5eed_5eed_5eed);
    // Every fp-reachable txb area, all multiples of 16.
    let sizes = [16usize, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    for &n in &sizes {
        let (scan, iscan) = perm_pair(&mut rng, n);
        for ls in 0..3i32 {
            for rep in 0..8 {
                // Production-shaped tables.
                let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];
                let round = [rng.pos_i16(0, 2000), rng.pos_i16(0, 2000)];
                let coeff: Vec<i32> = (0..n).map(|_| rng.i32_any()).collect();
                let label = format!("prod n={n} ls={ls} rep={rep}");

                let mut q_got = vec![0i32; n];
                let mut dq_got = vec![0i32; n];
                let eob_got = av1_quantize_fp_no_qmatrix_dispatch(
                    &quant,
                    &dequant,
                    &round,
                    ls,
                    &scan,
                    &iscan,
                    &coeff,
                    &mut q_got,
                    &mut dq_got,
                );
                let (q_want, dq_want, eob_want) =
                    c::ref_quantize_fp_avx2(ls, &coeff, &round, &quant, &dequant, &scan, &iscan);
                assert_eq!(eob_got, eob_want, "{label}: eob");
                assert_eq!(q_got, q_want, "{label}: qcoeff");
                assert_eq!(dq_got, dq_want, "{label}: dqcoeff");
            }
            for rep in 0..4 {
                // Full-range i16 tables — adversarial params that hit every
                // wrap/saturation edge (odd dequant, negative round, huge
                // quant where the i16 dq product wraps).
                let quant = [rng.i16_any(), rng.i16_any()];
                let dequant = [rng.i16_any(), rng.i16_any()];
                let round = [rng.i16_any(), rng.i16_any()];
                let coeff: Vec<i32> = (0..n).map(|_| rng.i32_any()).collect();
                let label = format!("adv n={n} ls={ls} rep={rep}");

                let mut q_got = vec![0i32; n];
                let mut dq_got = vec![0i32; n];
                let eob_got = av1_quantize_fp_no_qmatrix_dispatch(
                    &quant,
                    &dequant,
                    &round,
                    ls,
                    &scan,
                    &iscan,
                    &coeff,
                    &mut q_got,
                    &mut dq_got,
                );
                let (q_want, dq_want, eob_want) =
                    c::ref_quantize_fp_avx2(ls, &coeff, &round, &quant, &dequant, &scan, &iscan);
                assert_eq!(eob_got, eob_want, "{label}: eob");
                assert_eq!(q_got, q_want, "{label}: qcoeff");
                assert_eq!(dq_got, dq_want, "{label}: dqcoeff");
            }
        }
    }
}
