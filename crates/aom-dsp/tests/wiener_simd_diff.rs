//! SIMD-vs-scalar differential for `wiener_convolve_add_src` (Gate-3 parity
//! rule 1: bit-identical, no slip), at every archmage token permutation.
//!
//! The scalar reference is `wiener_convolve_add_src_scalar` (the transcribed
//! port, never SIMD-routed); the C pin is the pre-existing
//! `kernels_diff.rs::wiener_convolve_matches_c`, which drives the DISPATCHING
//! entry against the REAL C kernels (both lowbd and highbd) over odd widths
//! too. This test adds the per-tier fallback coverage on the same domain
//! (valid bd 8/10/12 pixels, real Wiener tap shapes incl. the chroma
//! tap0 == 0 form), plus width tails around the 8-lane overlap boundary.

use aom_dsp::restore::wiener::{wiener_convolve_add_src, wiener_convolve_add_src_scalar};
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u64) as i32
    }
}

const M: usize = 8;

#[test]
fn wiener_simd_bit_identical_to_scalar_at_every_tier() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    // Widths straddle the 8-lane boundary + the overlap-back tail (9, 15, 17)
    // + the sub-8 scalar route (4) + real RU shapes.
    let dims: &[(usize, usize)] = &[
        (4, 8),
        (8, 8),
        (9, 7),
        (15, 16),
        (16, 15),
        (17, 29),
        (32, 32),
        (63, 64),
        (64, 64),
        (128, 8),
    ];
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        let mut rng = Rng(0xA5A5_9e37_79b9_1234);
        for &bd in &[8, 10, 12] {
            for &(w, h) in dims {
                for case in 0..4 {
                    let (buf_w, buf_h) = (w + 2 * M, h + 2 * M);
                    let mask = (1u64 << bd) - 1;
                    let src: Vec<u16> =
                        (0..buf_w * buf_h).map(|_| (rng.next() & mask) as u16).collect();
                    let dst0: Vec<u16> =
                        (0..buf_w * buf_h).map(|_| (rng.next() & mask) as u16).collect();
                    let chroma = case % 4 == 3;
                    let mut hf = [0i16; 8];
                    let mut vf = [0i16; 8];
                    for f in [&mut hf, &mut vf] {
                        f[0] = if chroma { 0 } else { rng.range(-5, 10) as i16 };
                        f[1] = rng.range(-23, 8) as i16;
                        f[2] = rng.range(-17, 46) as i16;
                        f[3] = -2 * (f[0] + f[1] + f[2]);
                        f[4] = f[2];
                        f[5] = f[1];
                        f[6] = f[0];
                    }
                    let off = M * buf_w + M;
                    let mut got = dst0.clone();
                    wiener_convolve_add_src(
                        &src, off, buf_w, &mut got, off, buf_w, &hf, &vf, w, h, bd,
                    );
                    let mut want = dst0.clone();
                    wiener_convolve_add_src_scalar(
                        &src, off, buf_w, &mut want, off, buf_w, &hf, &vf, w, h, bd,
                    );
                    assert_eq!(
                        got, want,
                        "[{tier}] wiener {w}x{h} bd{bd} case {case} chroma={chroma}"
                    );
                }
            }
        }
    });
    eprintln!("wiener SIMD parity: {report}");
    assert!(report.permutations_run >= 2);
}
