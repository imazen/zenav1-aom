//! SIMD-vs-scalar differential for **`cdef_find_dir`'s partial-sum kernel**
//! (`cdef::simd::cdef_find_dir_partials`), at every archmage token permutation.
//!
//! `cdef_find_dir` is ~4.7 % of the q32 decode Ir and until this landing had no
//! SIMD tier at all (`docs/SIMD_REACH_AUDIT_2026-07-28.md` finding **F6**). Its
//! output is a DECISION, not a pixel: `dir` selects the filter taps and `var`
//! feeds the strength derivation, so a one-slot divergence is a different
//! bitstream decision — bit-identity here is not a rounding question.
//!
//! Sides:
//! * under test — [`aom_dsp::cdef::cdef_find_dir`], the dispatching entry the
//!   frame walk (`cdef/frame.rs:620`) and `pickcdef` call. It routes to the
//!   i16-lane magetypes kernel when [`aom_dsp::cdef::cdef_find_dir_simd_eligible`]
//!   holds, else to the scalar port.
//! * reference — [`aom_dsp::cdef::cdef_find_dir_scalar`], the transcribed i32
//!   port, **never SIMD-routed**. That is the side pinned against the REAL C
//!   `cdef_find_dir_c` by `cdef_diff.rs::cdef_find_dir_matches_c` (600 k cases).
//!
//! So the chain is: C == scalar port (`cdef_diff`) == dispatching entry at
//! EVERY tier (here). `cdef_diff` alone is not enough — it exercises exactly
//! one tier (whatever the host summons) and it never leaves the eligible
//! domain, so it can see neither the intermediate x86 tiers nor the scalar
//! fallback route.
//!
//! Domain: BOTH sides of the eligibility predicate, deliberately.
//! * eligible — `coeff_shift` 0/2/4 with pixels in `0..=(256 << cs) - 1`, which
//!   is the whole domain the decoder can produce (`cs == bd - 8`, window is
//!   interior plane data). Flavours: uniform random, constant, saturated,
//!   and eight synthetic directional ramps so every `dir` value is reachable.
//! * ineligible — `CDEF_VERY_LARGE` (the border sentinel) mixed in, and full
//!   `u16` values above `0x8000`. The latter is a live check that the guard's
//!   `simd_gt` is an UNSIGNED compare: a signed one would call `0xFFFF`
//!   eligible, run i16 lanes far outside their proof, and diverge here.

use aom_dsp::cdef::{
    CDEF_VERY_LARGE, cdef_find_dir, cdef_find_dir_scalar, cdef_find_dir_simd_eligible,
    cdef_find_dir_took_simd_path,
};
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
    fn upto(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

#[derive(Default, Debug)]
struct Totals {
    cases: u32,
    eligible: u32,
    ineligible: u32,
    /// Cases in which the VECTOR kernel actually accepted the window. This is
    /// the deep non-vacuity counter — see `cdef_find_dir_took_simd_path`.
    simd_accepts: u32,
    dirs_seen: u32,
    nonzero_var: u32,
}

/// One permutation's worth of cases. `vec_live` says whether this permutation
/// leaves a vector tier summonable, which fixes what the routing MUST do.
/// Returns the tallies the liveness floors below are asserted on.
fn sweep(tier: &dyn core::fmt::Display, vec_live: bool) -> Totals {
    let mut t = Totals::default();
    let mut rng = Rng(0x_f1ed_d1cd_5eed_0001u64);

    for &(coeff_shift, maxv) in &[(0i32, 255u32), (2, 1023), (4, 4095)] {
        // Strides wider than 8 so the row walk cannot silently read a
        // contiguous 64-pixel block (the frame walk always passes 144).
        for &stride in &[8usize, 12, 17, 144] {
            for it in 0..900u32 {
                let mut img = vec![0u16; 8 * stride + 8];
                match it % 9 {
                    // --- eligible flavours ---
                    0..=3 => {
                        for v in img.iter_mut() {
                            *v = rng.upto(maxv + 1) as u16;
                        }
                    }
                    4 => {
                        // Constant block: every cost is 0 -> the argmax must
                        // fall out of the tie-break, not out of a comparison.
                        let c = rng.upto(maxv + 1) as u16;
                        for v in img.iter_mut() {
                            *v = c;
                        }
                    }
                    5 => {
                        // Saturated two-level content: the largest partials the
                        // i16 proof has to cover.
                        for v in img.iter_mut() {
                            *v = if rng.upto(2) == 0 { 0 } else { maxv as u16 };
                        }
                    }
                    6 => {
                        // Directional ramp along one of the 8 search axes, so
                        // the sweep reaches every `dir` rather than the few a
                        // uniform-random block happens to favour.
                        let d = rng.upto(8) as i32;
                        let (dy, dx) = [
                            (0i32, 1i32),
                            (1, 2),
                            (1, 1),
                            (2, 1),
                            (1, 0),
                            (2, -1),
                            (1, -1),
                            (1, -2),
                        ][d as usize];
                        let amp = 1 + rng.upto(maxv + 1);
                        for i in 0..8i32 {
                            for j in 0..8i32 {
                                let p = (i * dx - j * dy).rem_euclid(4);
                                img[i as usize * stride + j as usize] =
                                    ((p as u32 * amp / 4) % (maxv + 1)) as u16;
                            }
                        }
                    }
                    // --- ineligible flavours (must route to the scalar port) ---
                    7 => {
                        // Border sentinel mixed into the window, the shape the
                        // work buffer carries outside the frame.
                        for v in img.iter_mut() {
                            *v = if rng.upto(4) == 0 {
                                CDEF_VERY_LARGE as u16
                            } else {
                                rng.upto(maxv + 1) as u16
                            };
                        }
                    }
                    _ => {
                        // Full u16 including >= 0x8000 — the unsigned-compare
                        // check described in the module docs.
                        for v in img.iter_mut() {
                            *v = rng.upto(u16::MAX as u32 + 1) as u16;
                        }
                    }
                }

                let eligible = cdef_find_dir_simd_eligible(&img, stride, coeff_shift);
                // Routing must follow the predicate EXACTLY: in a live vector
                // tier the kernel accepts iff the window is eligible; in a
                // scalar permutation it must decline unconditionally.
                let took_simd = cdef_find_dir_took_simd_path(&img, stride, coeff_shift);
                if vec_live {
                    assert_eq!(
                        took_simd, eligible,
                        "[{tier}] routing diverged from cdef_find_dir_simd_eligible: \
                         cs={coeff_shift} stride={stride} flavour={}",
                        it % 9
                    );
                } else {
                    assert!(
                        !took_simd,
                        "[{tier}] the scalar tier must decline so the entry runs the scalar port"
                    );
                }
                let got = cdef_find_dir(&img, stride, coeff_shift);
                let want = cdef_find_dir_scalar(&img, stride, coeff_shift);
                assert_eq!(
                    got, want,
                    "[{tier}] cdef_find_dir divergence: cs={coeff_shift} stride={stride} \
                     flavour={} eligible={eligible} img={:?}",
                    it % 9,
                    &img[..8]
                );

                t.cases += 1;
                if eligible {
                    t.eligible += 1;
                } else {
                    t.ineligible += 1;
                }
                if took_simd {
                    t.simd_accepts += 1;
                }
                t.dirs_seen |= 1 << got.0;
                if got.1 != 0 {
                    t.nonzero_var += 1;
                }
            }
        }
    }
    t
}

/// Counts permutations in which a VECTOR tier is actually live. Asserting only
/// `permutations_run >= 2` is satisfiable with ZERO of them, which is exactly
/// how the transform tier sat dead on aarch64 for months while its differential
/// passed (it reported simd_perms=0 — comparing the scalar path against
/// itself). See docs/SIMD_REACH_AUDIT_2026-07-28.md findings F3 and F4.
fn vector_tier_live() -> bool {
    // Per-architecture: this family's vector path is X64V3 on x86-64 and Neon
    // on aarch64. Testing only X64V3Token counts every aarch64 permutation as
    // scalar (that token is a stub off x86).
    if cfg!(target_arch = "aarch64") {
        archmage::NeonToken::summon().is_some()
    } else {
        archmage::X64V3Token::summon().is_some()
    }
}

fn assert_non_vacuous(simd_perms: usize) {
    assert!(
        simd_perms >= 1,
        "the SIMD permutation ({}) must run at least once — a passing run with \
         zero vector permutations compares the scalar path against itself. On \
         aarch64 this needs archmage's `testable_dispatch` dev-feature, else \
         baseline neon is excluded from the permutation set.",
        if cfg!(target_arch = "aarch64") { "neon" } else { "v3/AVX2" }
    );
}

#[test]
fn cdef_find_dir_simd_bit_identical_to_scalar_at_every_tier() {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            archmage::X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for the SIMD differential to be non-vacuous"
        );
    }
    // MEASURED 2026-07-28: under `AOM_FORCE_SCALAR=1` the permutation set this
    // binary sees is NOT deterministic — the pin disables every
    // runtime-dispatchable token process-wide on its first call, and whether
    // that lands before or after `for_each_token_permutation` enumerates
    // depends on which test thread in this binary touches a kernel first (both
    // 2-permutation and 25-permutation pinned runs were observed here). So the
    // per-permutation contract below is asserted unconditionally — it is
    // well-defined either way — while the AGGREGATE non-vacuity floor is
    // asserted in the UNPINNED leg, which is a CI leg of its own
    // (`.github/workflows/ci.yml`, the `force_scalar: "0"` matrix entry).
    let pinned = aom_dsp::dispatch::scalar_forced();
    let mut simd_perms = 0usize;
    let mut simd_accepts = 0u32;
    let mut totals = Totals::default();
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        let vec_live = vector_tier_live();
        if vec_live {
            simd_perms += 1;
        }
        totals = sweep(&tier, vec_live);
        simd_accepts += totals.simd_accepts;
    });
    eprintln!(
        "cdef_find_dir SIMD parity: {report}; per-perm {totals:?}; \
         vector-kernel accepts across all permutations = {simd_accepts}"
    );

    // Liveness floors — an equality that both sides reach by taking the SAME
    // route, or that only ever compares one trivial answer, proves nothing.
    //
    // 1. The eligible (SIMD-routed) branch must dominate: 7 of the 9 flavours
    //    are eligible by construction, so the floor is a conservative half.
    assert!(
        totals.eligible * 2 > totals.cases,
        "SIMD-routed cases must dominate the sweep: {totals:?}"
    );
    // 2. The ineligible (scalar-fallback) branch must actually be exercised.
    assert!(
        totals.ineligible * 20 > totals.cases,
        "the scalar-fallback route must be exercised: {totals:?}"
    );
    // 3. All 8 directions must be reachable — a kernel that always answered
    //    dir 0 would otherwise pass on a corpus that mostly answers dir 0.
    assert_eq!(
        totals.dirs_seen, 0xFF,
        "every direction must be produced by the sweep: {totals:?}"
    );
    // 4. `var` must move too (it is the second half of the return value and
    //    feeds the strength derivation).
    assert!(
        totals.nonzero_var * 3 > totals.cases,
        "`var` must be non-zero on a real share of cases: {totals:?}"
    );

    // 5. The DEEP non-vacuity guard: whenever a vector tier WAS summonable it
    //    must actually have ACCEPTED the eligible windows. `simd_perms >= 1`
    //    alone does not give this — the entry is bit-identical whichever route
    //    it takes, so a kernel that declined everything would still pass every
    //    assertion above while comparing the scalar port against itself.
    if simd_perms >= 1 {
        assert!(
            simd_accepts >= totals.eligible,
            "the vector kernel must accept every eligible window in at least one \
             permutation: simd_accepts={simd_accepts} eligible/perm={}",
            totals.eligible
        );
    }
    // 6. ...and in the unpinned leg a vector tier MUST have been summonable.
    if !pinned {
        assert_non_vacuous(simd_perms);
    }
    assert!(report.permutations_run >= 2);
}

/// The eligibility predicate is a *checked* condition, and the routing must
/// follow it exactly: `CDEF_VERY_LARGE` anywhere in the window (the border
/// sentinel) must be ineligible at every `coeff_shift` the decoder uses, and a
/// pixel at the limit must still be eligible.
#[test]
fn cdef_find_dir_eligibility_boundaries() {
    for &cs in &[0i32, 2, 4] {
        let limit = ((256u32 << cs) - 1) as u16;
        let mut img = vec![limit; 8 * 12 + 8];
        assert!(
            cdef_find_dir_simd_eligible(&img, 12, cs),
            "the largest in-domain pixel must stay eligible (cs={cs})"
        );
        for pos in [0usize, 7, 12 * 3 + 4, 12 * 7 + 7] {
            img[pos] = limit + 1;
            assert!(
                !cdef_find_dir_simd_eligible(&img, 12, cs),
                "one over-limit pixel at {pos} must make the window ineligible (cs={cs})"
            );
            // ...and the entry must still agree with the scalar port there.
            assert_eq!(
                cdef_find_dir(&img, 12, cs),
                cdef_find_dir_scalar(&img, 12, cs)
            );
            img[pos] = limit;
        }
        let mut border = vec![100u16; 8 * 12 + 8];
        border[12 * 2 + 3] = CDEF_VERY_LARGE as u16;
        assert!(
            !cdef_find_dir_simd_eligible(&border, 12, cs),
            "the CDEF_VERY_LARGE border sentinel must never be eligible (cs={cs})"
        );
    }
    // Out-of-range coeff_shift declines rather than computing a wrong limit.
    let img = vec![7u16; 8 * 12 + 8];
    assert!(!cdef_find_dir_simd_eligible(&img, 12, -1));
    assert!(!cdef_find_dir_simd_eligible(&img, 12, 8));
}
