//! SIMD-vs-scalar differential for `variance_4x4_units` — the batched
//! per-4x4-unit (sum, sumsq) walk `intra_rd_variance_factor` runs per
//! candidate-mode eval (Gate-3 parity rule 1: bit-identical, no slip), at
//! every archmage token permutation.
//!
//! There is no direct C pin for this shape — it is the `all_zeros`-folded
//! core of `av1_calc_normalized_variance`, which C calls through
//! `aom_variance4x4`. The C-side coverage of that call is the pre-existing
//! `hbd_dist_diff.rs` / `dist_diff` family; this test covers the dispatch
//! tiers and the group/tail split (4-unit ymm groups, 2-unit xmm group,
//! scalar tail) on the pixel domain (samples < 1<<bd), including all-max
//! boundary planes, strided rows and every tail residue.

use aom_dsp::dist::{variance_4x4_units, variance_4x4_units_scalar};
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
fn variance_4x4_units_bit_identical_to_scalar_at_every_tier() {
    // Serialise: this sweep permutes PROCESS-GLOBAL dispatch
    // state; see `crate::dispatch_serial`.
    let _serial = crate::dispatch_serial::dispatch_serial();
    // NOTE: no pre-flight token check — see `hbd_variance_simd_diff`'s
    // longer note; under AOM_FORCE_SCALAR the pin disables every token
    // process-wide until the permutation harness re-enables them.
    let mut simd_perms = 0usize;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        if if cfg!(target_arch = "aarch64") {
            archmage::NeonToken::summon().is_some()
        } else {
            archmage::X64V3Token::summon().is_some()
        } {
            simd_perms += 1;
        }
        let mut rng = Rng(0x_4a44_b015_5eED_2026);
        for &bd in &[8u8, 10, 12] {
            let mask = (1u64 << bd) - 1;
            // Unit counts covering: pure tails (1..3), single groups of
            // every residue, multi-group, and the largest real walk
            // (128-px leaf row = 32 units).
            for &units in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 21, 31, 32] {
                for case in 0..4 {
                    // 4 rows per unit band, strided.
                    let w_px = 4 * units;
                    let a_stride = w_px + (rng.next() % 9) as usize;
                    let h = 4usize;
                    let mut a: Vec<u16> = (0..a_stride * h)
                        .map(|_| (rng.next() & mask) as u16)
                        .collect();
                    if case == 1 {
                        a.fill(mask as u16);
                    }
                    if case == 2 {
                        let v = (rng.next() & mask) as u16;
                        a.fill(v);
                    }
                    let mut got = vec![(0i32, 0u32); units];
                    let mut want = vec![(0i32, 0u32); units];
                    variance_4x4_units(&a, a_stride, 0, units, &mut got);
                    variance_4x4_units_scalar(&a, a_stride, 0, units, &mut want);
                    assert_eq!(got, want, "[{tier}] units={units} bd{bd} case {case}");
                }
            }
        }
    });
    eprintln!("variance_4x4_units SIMD parity: {report}");
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
