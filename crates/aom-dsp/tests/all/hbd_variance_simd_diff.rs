//! SIMD-vs-scalar differential for `highbd_variance64` (Gate-3 parity rule 1:
//! bit-identical, no slip), at every archmage token permutation.
//!
//! The C pin is the pre-existing `hbd_dist_diff.rs`, which drives the
//! DISPATCHING `highbd_variance` against the REAL C kernels across all 22
//! block sizes x 3 bit depths. This test adds the per-tier fallback coverage
//! on the pixel domain (values < 1<<bd), including all-max boundary planes
//! and strided rows.

use aom_dsp::dist::{highbd_variance, highbd_variance64_scalar};
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
fn hbd_variance_simd_bit_identical_to_scalar_at_every_tier() {
    // NOTE: there is deliberately no pre-flight `X64V3Token::summon().is_some()`
    // check here. It looks like a non-vacuity guard but is an ordering trap:
    // under AOM_FORCE_SCALAR=1 the pin disables every runtime-dispatchable
    // token process-wide, so on x86-64 `summon()` correctly returns None right
    // up until `for_each_token_permutation` resets that state and re-enables
    // them. Tests that fire the pin first (the documented order) therefore
    // failed the check on the linux scalar-pin CI leg while passing on
    // aarch64, where baseline `neon` cannot be disabled at all. The real
    // non-vacuity guard is the `simd_perms >= 1` assertion below, which runs
    // INSIDE the harness and catches a genuinely vector-less box with a
    // better message.
    // All libaom block geometries (incl. the w=4 scalar route).
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
        (32, 64),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (4, 16),
        (16, 4),
        (8, 32),
        (32, 8),
        (16, 64),
        (64, 16),
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
        let mut rng = Rng(0x_d157_9e37_79b9_1111);
        for &bd in &[8u8, 10, 12] {
            let mask = (1u64 << bd) - 1;
            for &(w, h) in dims {
                for case in 0..4 {
                    let a_stride = w + (rng.next() % 9) as usize; // strided rows
                    let b_stride = w + (rng.next() % 9) as usize;
                    let mut a: Vec<u16> =
                        (0..a_stride * h).map(|_| (rng.next() & mask) as u16).collect();
                    let mut b: Vec<u16> =
                        (0..b_stride * h).map(|_| (rng.next() & mask) as u16).collect();
                    if case == 1 {
                        // Max-diff boundary: a all-max, b all-zero.
                        a.fill(mask as u16);
                        b.fill(0);
                    }
                    if case == 2 {
                        // Flat (near-tie normalisation edge in the callers).
                        let v = (rng.next() & mask) as u16;
                        a.fill(v);
                        b.fill(v);
                    }
                    let got64 = {
                        // Exercise the dispatch through the public wrapper
                        // (variance + sse), AND the raw 64-bit pair.
                        highbd_variance(&a, a_stride, &b, b_stride, w, h, bd)
                    };
                    let want_pair = highbd_variance64_scalar(&a, a_stride, &b, b_stride, w, h);
                    // Recompute the wrapper's normalisation from the scalar
                    // pair to compare the full (var, sse) result.
                    let (sse_long, sum_long) = want_pair;
                    let (sse, sum): (u32, i32) = match bd {
                        8 => (sse_long as u32, sum_long as i32),
                        10 => (
                            ((sse_long + (1 << 3)) >> 4) as u32,
                            ((sum_long + (1 << 1)) >> 2) as i32,
                        ),
                        _ => (
                            ((sse_long + (1 << 7)) >> 8) as u32,
                            ((sum_long + (1 << 3)) >> 4) as i32,
                        ),
                    };
                    let want_var = if bd == 8 {
                        sse.wrapping_sub(
                            ((i64::from(sum) * i64::from(sum)) / (w * h) as i64) as u32,
                        )
                    } else {
                        let v = i64::from(sse) - (i64::from(sum) * i64::from(sum)) / (w * h) as i64;
                        if v >= 0 { v as u32 } else { 0 }
                    };
                    assert_eq!(
                        got64,
                        (want_var, sse),
                        "[{tier}] {w}x{h} bd{bd} case {case}"
                    );
                }
            }
        }
    });
    eprintln!("highbd_variance SIMD parity: {report}");
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
