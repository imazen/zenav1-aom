//! SIMD-vs-scalar differential for `txb_init_levels` (Gate-3 parity rule 1:
//! bit-identical, no slip), at every archmage token permutation, on the FULL
//! i32 domain (adversarial values incl. i32::MIN/MAX — the kernel's
//! exactness argument covers the whole domain, so the test asserts it there).
//!
//! The C pin is the pre-existing `txb_diff.rs`, which drives the DISPATCHING
//! `txb_init_levels` against the REAL `av1_txb_init_levels` including the
//! exact write footprint.

use aom_dsp::txb::{TX_PAD_2D, txb_init_levels, txb_init_levels_scalar};
// `summon()` comes from this trait; needed at MODULE scope because the
// non-vacuity counter below lives outside the fn-local `use` blocks.
use archmage::SimdToken;
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
}

#[test]
fn txb_init_levels_simd_bit_identical_to_scalar_at_every_tier() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    // All (adjusted) txb geometries: widths/heights 4..32.
    let dims: &[(usize, usize)] = &[
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (8, 16),
        (16, 8),
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (4, 16),
        (16, 4),
        (8, 32),
        (32, 8),
    ];
    // Counts permutations in which a VECTOR tier is actually live. Asserting
    // only `permutations_run >= 2` is satisfiable with ZERO of them, which is
    // exactly how the transform tier sat dead on aarch64 for months while its
    // differential passed (it reported simd_perms=0 — comparing the scalar
    // path against itself). See docs/SIMD_REACH_AUDIT_2026-07-28.md finding F4.
    let mut simd_perms = 0usize;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        // Per-architecture: this family's vector path is X64V3 on x86-64 and
        // Neon on aarch64. Testing only X64V3Token counts every aarch64
        // permutation as scalar (that token is a stub off x86).
        if if cfg!(target_arch = "aarch64") {
            archmage::NeonToken::summon().is_some()
        } else {
            archmage::X64V3Token::summon().is_some()
        } {
            simd_perms += 1;
        }
        let mut rng = Rng(0x_7b17_1234_5678_9abc);
        for &(w, h) in dims {
            for case in 0..12 {
                let mut coeff: Vec<i32> = (0..w * h).map(|_| rng.next() as i32).collect();
                if case == 0 {
                    coeff.fill(0);
                }
                if case == 1 {
                    coeff[0] = i32::MIN;
                    coeff[1] = i32::MAX;
                    coeff[w * h - 1] = i32::MIN;
                    coeff[w * h / 2] = -128;
                    coeff[w * h / 2 + 1] = 127;
                }
                if case == 2 {
                    for (i, c) in coeff.iter_mut().enumerate() {
                        *c = (i as i32 % 300) - 150; // realistic small levels
                    }
                }
                // Prefill both level buffers with a sentinel to also pin the
                // exact write FOOTPRINT (bytes outside it must stay 0xEE).
                let mut got = vec![0xEEu8; TX_PAD_2D];
                let mut want = vec![0xEEu8; TX_PAD_2D];
                txb_init_levels(&coeff, w, h, &mut got);
                txb_init_levels_scalar(&coeff, w, h, &mut want);
                assert_eq!(got, want, "[{tier}] {w}x{h} case {case}");
            }
        }
    });
    eprintln!("txb_init_levels SIMD parity: {report}");
    assert!(
        simd_perms >= 1,
        "the SIMD permutation ({}) must run at least once — a passing run with \
         zero vector permutations compares the scalar path against itself. On \
         aarch64 this needs archmage's `testable_dispatch` dev-feature, else \
         baseline neon is excluded from the permutation set.",
        if cfg!(target_arch = "aarch64") { "neon" } else { "v3/AVX2" }
    );
    assert!(report.permutations_run >= 2);
}
