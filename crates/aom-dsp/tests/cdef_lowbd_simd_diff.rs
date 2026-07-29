//! SIMD-vs-scalar differential for the **bd8 lowbd (`u8`-store) CDEF kernels**
//! `cdef_filter_8_w8` and `cdef_filter_8_w4` (Gate-3 parity rule 1:
//! bit-identical, no slip), at every archmage token permutation.
//!
//! These are the kernels the PRIMARY bd8 decode configuration runs — CDEF is
//! ~27% of decode Ir on the conformance-style q32 cell — and until this file
//! landed they had **no per-permutation differential at all**
//! (`docs/SIMD_REACH_AUDIT_2026-07-28.md` finding F3). Their scalar-vs-SIMD
//! comparison existed only because CI happens to run the whole aarch64 aom-dsp
//! suite twice, once with `AOM_FORCE_SCALAR=1` — coverage that lives in the
//! workflow file, not in a test, and that never reaches the intermediate x86
//! tiers at all.
//!
//! Sides:
//! * under test — [`aom_dsp::cdef::cdef_filter_block_u8`], the DISPATCHING
//!   entry the frame walk calls (`cdef/frame.rs:152`, via `CdefPixel for u8`).
//!   It routes width-8 to `simd::cdef_filter_8_w8` and even-height width-4 to
//!   `simd::cdef_filter_8_w4`; everything else falls to the scalar core.
//! * reference — [`aom_dsp::cdef::cdef_filter_block`], the transcribed scalar
//!   core with the same `(uint8_t)y` store, **never SIMD-routed** (it calls
//!   `cdef_filter_block_core` directly). It is the u8 analogue of
//!   `cdef_filter_block_16_scalar`, and it is the side pinned against the REAL
//!   C `cdef_filter_8_{0,1,2,3}` by `cdef_filter_diff.rs::cdef_filter8_byte_identical`.
//!
//! So the chain is: C == scalar core (cdef_filter_diff) == dispatching u8 entry
//! at EVERY tier (here) — and `cdef_lowbd_diff.rs` independently pins the whole
//! frame walk against the real C lowbd walk.
//!
//! Domain mirrors `cdef_filter_simd_diff.rs` (the u16 twin) axis for axis —
//! pixel values including the `CDEF_VERY_LARGE` border sentinel, all-border and
//! all-max boundary flavours, the header's primary/secondary strength and
//! damping ranges, all 8 directions, both `pri_taps` parities, all four
//! `en_pri`/`en_sec` combinations — with two deliberate differences, both
//! WIDENING or contract-pinning rather than narrowing:
//! * `coeff_shift` is pinned to 0 and pixels to `0..=255`, because that is the
//!   whole domain this path can ever see: `cdef_frame_u8` is the bd8 walk
//!   (`frame.rs:817` asserts `bit_depth == 8`) and sets `coeff_shift: 0`
//!   (`frame.rs:872`). The u16 kernels sweep `coeff_shift` 0/2/4 only because
//!   they also serve bd10/bd12.
//! * heights sweep `{2, 4, 6, 8}` plus an odd height, wider than the u16 test's
//!   `{4, 8}`, and `dst_off`/`dstride` are non-trivial so the strided,
//!   offset store is exercised (the frame walk always stores into a plane at a
//!   row offset with stride >> block width).

use aom_dsp::cdef::{CDEF_BSTRIDE, CDEF_VERY_LARGE, cdef_filter_block, cdef_filter_block_u8};
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

/// Fill for the destination outside the written block — identical on both
/// sides, so any out-of-block write shows up as an inequality, and asserted
/// untouched on the dispatch side so it shows up as a *located* failure.
const DST_PAD: u8 = 0xAB;

/// One permutation's worth of cases for a fixed block width.
///
/// `block_width` selects which dispatch arm `cdef_filter_block_u8` takes:
/// 8 -> `cdef_filter_8_w8`, 4 -> `cdef_filter_8_w4` (even heights) / scalar
/// core (odd heights). Returns `(cases, cases_that_changed_a_pixel)`.
fn sweep_width(tier: &dyn core::fmt::Display, block_width: usize, seed: u64) -> (u32, u32) {
    let rows = 16;
    let buf_len = rows * CDEF_BSTRIDE;
    // Block origin: 4 border rows + 8 border cols (>= CDEF_VBORDER=2 and the
    // direction reach), exactly like the u16 differential.
    let in_off = 4 * CDEF_BSTRIDE + 8;
    let mut rng = Rng(seed);
    let mut n = 0u32;
    let mut n_changed = 0u32;

    for variant in 0..4i32 {
        let (en_pri, en_sec) = (variant == 0 || variant == 1, variant == 0 || variant == 2);
        for it in 0..800 {
            // Heights: the walk's 4 and 8, plus 2/6 (even -> w4 kernel) and an
            // odd height that must fall through to the scalar core.
            let bh = [8usize, 4, 2, 6, 8, 4, 5, 8][(it % 8) as usize];
            let mut inbuf = vec![0u16; buf_len];
            for v in inbuf.iter_mut() {
                *v = if rng.upto(20) == 0 {
                    CDEF_VERY_LARGE as u16
                } else {
                    rng.upto(256) as u16
                };
            }
            // Boundary flavours: all-border rows above, max-pixel rows below.
            if it % 7 == 0 {
                for v in inbuf[..2 * CDEF_BSTRIDE].iter_mut() {
                    *v = CDEF_VERY_LARGE as u16;
                }
            }
            if it % 11 == 0 {
                for v in inbuf[buf_len - 2 * CDEF_BSTRIDE..].iter_mut() {
                    *v = 255;
                }
            }
            // Header ranges at coeff_shift 0: pri 0..=15, sec 0..=4, damping
            // 3..=6 for luma and one less for chroma (frame.rs:611), so 2..=6.
            let pri = rng.upto(16) as i32;
            let sec = rng.upto(5) as i32;
            let dir = rng.upto(8) as i32;
            let prid = 2 + rng.upto(5) as i32;
            let secd = 2 + rng.upto(5) as i32;
            let cshift = 0;

            // Strided, offset destination — the shape the frame walk uses.
            let dstride = block_width + 3 + (it % 5) as usize;
            let dst_off = CDEF_BSTRIDE + 5;
            let dlen = dst_off + bh * dstride + block_width + 7;

            let mut got = vec![DST_PAD; dlen];
            cdef_filter_block_u8(
                &mut got, dst_off, dstride, &inbuf, in_off, pri, sec, dir, prid, secd, cshift,
                block_width, bh, en_pri, en_sec,
            );
            let mut want = vec![DST_PAD; dlen];
            cdef_filter_block(
                &mut want[dst_off..],
                dstride,
                &inbuf,
                in_off,
                pri,
                sec,
                dir,
                prid,
                secd,
                cshift,
                block_width,
                bh,
                en_pri,
                en_sec,
            );
            let ctx = format!(
                "[{tier}] v{variant} bw={block_width} bh={bh} dir={dir} pri={pri} sec={sec} \
                 prid={prid} secd={secd} dstride={dstride}"
            );
            assert_eq!(got, want, "{ctx}");
            // No writes outside the block footprint (the equality above already
            // covers it, but this localises the failure).
            assert!(
                got[..dst_off].iter().all(|&x| x == DST_PAD),
                "wrote before dst_off: {ctx}"
            );
            for i in 0..bh {
                let row = dst_off + i * dstride;
                assert!(
                    got[row + block_width..row + dstride].iter().all(|&x| x == DST_PAD),
                    "wrote past the block width on row {i}: {ctx}"
                );
            }
            n += 1;
            // Liveness: the filter must actually move pixels over the sweep,
            // else both sides could be agreeing on "copy the centre pixel".
            // Ceiling is 75%: variant 3 (both classes disabled) is a no-op by
            // construction. Measured ~48-50%; the floor asserted below is 1/3.
            let centre = |i: usize, j: usize| inbuf[in_off + i * CDEF_BSTRIDE + j] as u8;
            if (0..bh).any(|i| {
                (0..block_width).any(|j| got[dst_off + i * dstride + j] != centre(i, j))
            }) {
                n_changed += 1;
            }
        }
    }
    (n, n_changed)
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
fn cdef_filter8_w8_simd_bit_identical_to_scalar_at_every_tier() {
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
    let mut simd_perms = 0usize;
    let mut totals = (0u32, 0u32);
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        if vector_tier_live() {
            simd_perms += 1;
        }
        totals = sweep_width(&tier, 8, 0x_cdef_08b0_77e5_9a01);
    });
    eprintln!(
        "cdef_filter8 w8 SIMD parity: {report} ({} cases/perm, {} changed pixels)",
        totals.0, totals.1
    );
    assert!(totals.1 * 3 > totals.0, "pixel-changing floor: {totals:?}");
    assert_non_vacuous(simd_perms);
    assert!(report.permutations_run >= 2);
}

#[test]
fn cdef_filter8_w4_simd_bit_identical_to_scalar_at_every_tier() {
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
    let mut simd_perms = 0usize;
    let mut totals = (0u32, 0u32);
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        if vector_tier_live() {
            simd_perms += 1;
        }
        totals = sweep_width(&tier, 4, 0x_cdef_04b0_31c7_5e2d);
    });
    eprintln!(
        "cdef_filter8 w4 SIMD parity: {report} ({} cases/perm, {} changed pixels)",
        totals.0, totals.1
    );
    assert!(totals.1 * 3 > totals.0, "pixel-changing floor: {totals:?}");
    assert_non_vacuous(simd_perms);
    assert!(report.permutations_run >= 2);
}
