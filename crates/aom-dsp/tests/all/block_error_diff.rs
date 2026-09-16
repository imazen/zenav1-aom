//! Differential harness for the transform-domain distortion (av1_block_error) vs
//! C libaom: error = sum((coeff-dqcoeff)^2), ssz = sum(coeff^2). Lowbd (32-bit
//! products) + highbd (64-bit products, rounded-shift by 2*(bd-8)).

use aom_dsp::dist::{block_error, highbd_block_error};
use aom_sys_ref as c;

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
    fn coeff(&mut self, bits: u32) -> i32 {
        (self.next() % (1 << (bits + 1))) as i32 - (1 << bits)
    }
}

#[test]
fn block_error_differential() {
    let mut rng = Rng(0x_b10c_e770_9e37_79b9);
    // tx areas.
    for &n in &[16usize, 64, 256, 1024] {
        for _ in 0..5000 {
            // Lowbd (bd=8) transform coeffs: |c|^2 must stay < 2^31 (32-bit
            // products, matching C's `int` arithmetic) => ~14-bit magnitudes.
            let coeff: Vec<i32> = (0..n).map(|_| rng.coeff(14)).collect();
            let dqcoeff: Vec<i32> = (0..n).map(|_| rng.coeff(14)).collect();
            let got = block_error(&coeff, &dqcoeff);
            let want = c::ref_block_error(&coeff, &dqcoeff);
            assert_eq!(got, want, "block_error n={n}");
        }
    }
}

/// The port's v3 kernel mirrors `av1_block_error_avx2` itself — including its
/// `packs_epi32` i32->i16 saturation — so the two agree on the FULL i32 domain
/// where `_c` does not. This pins the port against the dispatched kernel.
#[test]
fn block_error_matches_c_avx2_full_domain() {
    let mut rng = Rng(0x_b10c_e770_a0c2_5ed0);
    for &n in &[16usize, 64, 256, 1024] {
        for i in 0..4000 {
            // Full i32 magnitudes, plus periodic saturation-boundary hits.
            let coeff: Vec<i32> = (0..n)
                .map(|j| {
                    if (i + j) % 97 == 0 {
                        // -2^31, -32768, 32767, +2^31-1 in rotation.
                        [i32::MIN, -32768, 32767, i32::MAX][(i / 97) & 3]
                    } else {
                        rng.next() as i32
                    }
                })
                .collect();
            let dqcoeff: Vec<i32> = (0..n).map(|_| rng.next() as i32).collect();
            let got = block_error(&coeff, &dqcoeff);
            let want = c::ref_block_error_simd(&coeff, &dqcoeff);
            assert_eq!(got, want, "block_error vs dispatched C kernel n={n} i={i}");
        }
    }
}

#[test]
fn highbd_block_error_differential() {
    let mut rng = Rng(0x_b10c_e770_c057_0b11);
    for &n in &[16usize, 64, 256, 1024] {
        for &bd in &[8u8, 10, 12] {
            for _ in 0..2000 {
                // Highbd: 64-bit products, up to ~18-bit magnitudes.
                let coeff: Vec<i32> = (0..n).map(|_| rng.coeff(18)).collect();
                let dqcoeff: Vec<i32> = (0..n).map(|_| rng.coeff(18)).collect();
                let got = highbd_block_error(&coeff, &dqcoeff, bd);
                let want = c::ref_highbd_block_error(&coeff, &dqcoeff, bd);
                assert_eq!(got, want, "highbd_block_error n={n} bd={bd}");
            }
        }
    }
}
