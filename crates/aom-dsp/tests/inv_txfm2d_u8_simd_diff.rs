//! SIMD-vs-scalar differential for the **bd8 lowbd (`u8`-store) inverse
//! transform column pass** — `av1_inv_txfm2d_add_u8`'s vector path — at every
//! archmage token permutation.
//!
//! This is the transform half of `docs/SIMD_REACH_AUDIT_2026-07-28.md` finding
//! **F3** (the CDEF half landed as `cdef_lowbd_simd_diff.rs`). The u8 column
//! pass is what the PRIMARY bd8 decode configuration runs — transforms are
//! ~12% of decode Ir — and until this file landed it had **no
//! `for_each_token_permutation` differential at all**:
//! `txfm2d_simd_perm_diff.rs` drives `av1_inv_txfm2d_add` (u16) and
//! `av1_fwd_txfm2d` only, and the u8 entry was reached solely by
//! `inv_txfm2d_lowbd_diff` / `recon_lowbd_diff`, i.e. at whatever tier happened
//! to be live in that test process. That made its scalar-vs-SIMD comparison
//! depend on CI running the whole aarch64 suite twice (once with
//! `AOM_FORCE_SCALAR=1`) — coverage that lives in the workflow file, not in a
//! test, and that never reaches the intermediate x86 tiers at all.
//!
//! It got MORE load-bearing, not less: `13b7c21` armed the bd8 i16-lane
//! specialization on aarch64 (`prims16.rs`, `lowbd16.rs`), so there is now live
//! NEON vector code on this path whose only guard was frame-level tests.
//!
//! # Method
//!
//! Same shape as `txfm2d_simd_perm_diff.rs`, and for the same reason: there is
//! deliberately **no scalar implementation** of the pass to call as a reference
//! (`inv_col_pass_u8_scalar` / `inv_col_pass_u8_i16_scalar` return `false`),
//! because the scalar twin IS the driver's own per-column loop in
//! `av1_inv_txfm2d_add_u8_into` — reachable only by declining the vector path.
//! So the reference is the FIRST permutation's outputs, and every later
//! permutation must byte-match it over identical (fixed-seed) inputs. The
//! harness always includes an all-off (scalar) permutation, so the equality
//! chain transitively pins vector == scalar; both counters are asserted.
//!
//! # Both dispatch arms are covered deliberately
//!
//! `try_inv_col_pass_u8` (`transform/simd/mod.rs`) has two vector arms:
//!
//! * the **i16 arm** — `lowbd16::inv_col_pass_u8_i16`, 16 columns per vector,
//!   taken when `lowbd16::inv_kernel_i16(txfm_type_col)` is `Some` (the audited
//!   DCT4/8/16/32/64 column kernels) and the bd8 constants hold
//!   (`col_clamp == 16`, `shift1_bit == 4` — both are structural at bd8:
//!   `opt_range(8) == (16, 16)` and `INV_SHIFT[_][1] == -4` for every tx_size);
//! * the **i32 arm** — `inv_col_pass_u8`, 8 columns per vector, everything else:
//!   an ADST / FLIPADST / IDTX column kernel.
//!
//! The selector is therefore exactly "is the VERTICAL 1-D transform a DCT",
//! i.e. `av1_vtx_tab[tx_type] == 0`. The two tests below partition the whole
//! valid (tx_type × tx_size) matrix on that predicate, and each asserts its own
//! side is non-empty — so a future change that silently stops reaching an arm
//! fails here instead of quietly halving the coverage.
//!
//! `VTX_TAB` is transcribed below, so it is cross-checked against the crate's
//! own public API rather than trusted: `av1_txfm_type_ls` has no ADST/FLIPADST
//! entry at 32 points and only DCT at 64, so `inv_txfm_valid` over a
//! column-constrained tx_size identifies `vtx == 0` exactly. See
//! [`vtx_tab_matches_public_validity`].
//!
//! # Domain
//!
//! Mirrors `txfm2d_simd_perm_diff`'s inverse half: coefficients across ±2^20
//! (well past `inv_txfm2d_lowbd_diff`'s ±2^16 vs-C cap, which exists only
//! because C's `half_btf` overflows an i32 there) plus the exact clamp-bound
//! spike patterns, tight AND strided destinations, random u8 predictions. Both
//! passes clamp their inputs into the i16 domain BEFORE the kernel
//! (`clamp_buf` / `pack_clamp16`), so out-of-band coefficients are in-contract
//! for both arms, not a domain violation.
//!
//! Two things `inv_txfm2d_lowbd_diff` does not do and this does: **strided u8
//! destinations** (that test only ever passes `stride == w`) and a pad-guard on
//! the bytes past the block.

use aom_dsp::transform::inv_txfm2d::{av1_inv_txfm2d_add_u8, inv_input_len, inv_txfm_valid};
use archmage::X64V3Token;
use archmage::prelude::*;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

const W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

/// `av1_vtx_tab` (`av1/common/av1_txfm.h`) — the VERTICAL (column) 1-D
/// transform class per TX_TYPE: `0 = DCT, 1 = ADST, 2 = FLIPADST, 3 = IDTX`.
/// Only the `== 0` predicate is load-bearing here, and it is verified against
/// the crate's public `inv_txfm_valid` by [`vtx_tab_matches_public_validity`].
#[rustfmt::skip]
const VTX_TAB: [usize; 16] = [0, 1, 0, 1, 2, 0, 2, 1, 2, 3, 0, 3, 1, 3, 2, 3];

/// `(ud_flip, lr_flip)` per TX_TYPE (`av1_get_flip_cfg`) — counted, not used
/// for dispatch, so the coverage report can show which flip branches of the two
/// column-pass cores actually ran.
#[rustfmt::skip]
const FLIP_CFG: [(bool, bool); 16] = [
    (false, false), (false, false), (false, false), (false, false),
    (true, false),  (false, true),  (true, true),   (false, true),
    (true, false),  (false, false), (false, false), (false, false),
    (false, false), (false, false), (true, false),  (false, true),
];

/// Which vector arm of `try_inv_col_pass_u8` a (tx_type, tx_size) cell takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// `lowbd16::inv_col_pass_u8_i16` — audited DCT column kernel, i16 lanes.
    I16,
    /// `inv_col_pass_u8` — ADST / FLIPADST / IDTX column kernel, i32 lanes.
    I32,
}

fn arm_of(tx_type: usize) -> Arm {
    if VTX_TAB[tx_type] == 0 { Arm::I16 } else { Arm::I32 }
}

const SEED: u64 = 0x_10bd_c01_5119_d1ff;

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
    /// Dequantised coefficient across ±2^20 — past the bd8 row clamp (±2^15),
    /// so a large fraction of lanes land ON the clamp, which is exactly the
    /// `pack_clamp16` / `clamp_buf` saturation both arms must reproduce.
    fn coeff(&mut self) -> i32 {
        (self.next() % (1 << 21)) as i32 - (1 << 20)
    }
    fn pixel(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

/// Clamp-bound spike patterns. `B` is the bd8 clamp bound (`clamp_value(_, 16)`
/// → `[-2^15, 2^15-1]`); the ±2^20 entries deliberately exceed it.
fn inv_spike(k: usize, i: usize, len: usize) -> i32 {
    const B: i32 = 1 << 15;
    match k {
        0 => B,
        1 => -B,
        2 => {
            if i % 2 == 0 {
                B
            } else {
                -B
            }
        }
        3 => {
            if i % 2 == 0 {
                -B
            } else {
                B
            }
        }
        4 => {
            if i == 0 {
                1 << 20
            } else {
                0
            }
        }
        5 => {
            if i + 1 == len {
                -(1 << 20)
            } else {
                0
            }
        }
        _ => B - 1,
    }
}

const RAND_REPS: usize = 4;
const SPIKES: usize = 7;
/// Byte written outside the `h * stride` block footprint; asserted untouched.
const PAD: u8 = 0x5A;

/// Coverage counters for one sweep — every one of these is asserted, because
/// "the sweep ran" and "the sweep reached the arm" are different facts.
#[derive(Default, Clone, Copy, Debug)]
struct Cov {
    /// (tx_type, tx_size) cells swept.
    cells: u32,
    /// Cells whose width makes the vector path unconditionally eligible
    /// (`col_n % 8 == 0`; the `col_n == 4` cells go through
    /// `half_batch_pays`, which is architecture-dependent, so they are not
    /// counted here).
    vector_cells: u32,
    /// Cells reaching the i16 core's 16-lane group (`col_n >= 16`) resp. the
    /// i32 core's full 8-lane group (`col_n >= 8`).
    wide_cells: u32,
    /// Cells with `lr_flip` / `ud_flip` set — the reversal branches.
    lr_cells: u32,
    ud_cells: u32,
    /// Individual transform invocations, and how many actually moved a pixel.
    cases: u32,
    changed: u32,
}

/// Run every valid cell of one ARM through `av1_inv_txfm2d_add_u8` under the
/// CURRENT token permutation. Inputs are deterministic (fixed seed), so
/// dispatch is the only thing that varies between calls.
fn all_outputs(arm: Arm) -> (Vec<(String, Vec<u8>)>, Cov) {
    let mut rng = Rng(SEED);
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    let mut cov = Cov::default();

    for tx_size in 0..19usize {
        let (w, h) = (W[tx_size], H[tx_size]);
        for tx_type in 0..16usize {
            if !inv_txfm_valid(tx_type, tx_size) || arm_of(tx_type) != arm {
                continue;
            }
            cov.cells += 1;
            if w % 8 == 0 {
                cov.vector_cells += 1;
            }
            if (arm == Arm::I16 && w >= 16) || (arm == Arm::I32 && w >= 8) {
                cov.wide_cells += 1;
            }
            let (ud, lr) = FLIP_CFG[tx_type];
            cov.ud_cells += u32::from(ud);
            cov.lr_cells += u32::from(lr);

            let ilen = inv_input_len(tx_size);
            // Tight and strided destinations. `inv_txfm2d_lowbd_diff` only ever
            // uses the tight one, so the strided store is new coverage.
            for &stride in &[w, w + 3] {
                let dlen = h * stride + 32;
                let mut run = |label: String, input: &[i32], rng: &mut Rng| {
                    let mut buf = vec![PAD; dlen];
                    for p in buf[..h * stride].iter_mut() {
                        *p = rng.pixel();
                    }
                    let pred: Vec<u8> = buf.clone();
                    av1_inv_txfm2d_add_u8(input, &mut buf, stride, tx_type, tx_size);
                    assert!(
                        buf[h * stride..].iter().all(|&x| x == PAD),
                        "wrote past the destination footprint: {label}"
                    );
                    cov.cases += 1;
                    if buf[..h * stride] != pred[..h * stride] {
                        cov.changed += 1;
                    }
                    out.push((label, buf));
                };
                for rep in 0..RAND_REPS {
                    let input: Vec<i32> = (0..ilen).map(|_| rng.coeff()).collect();
                    run(
                        format!("u8 sz{tx_size} ty{tx_type} st{stride} rand{rep}"),
                        &input,
                        &mut rng,
                    );
                }
                for k in 0..SPIKES {
                    let input: Vec<i32> = (0..ilen).map(|i| inv_spike(k, i, ilen)).collect();
                    run(
                        format!("u8 sz{tx_size} ty{tx_type} st{stride} spike{k}"),
                        &input,
                        &mut rng,
                    );
                }
            }
        }
    }
    (out, cov)
}

/// Is a VECTOR tier live in this permutation? Per-architecture: the transform's
/// vector path is `X64V3` on x86-64 and `Neon` on aarch64 — testing only
/// `X64V3Token` counts every aarch64 permutation as scalar (that token is a
/// stub off x86), which is the bug `33bb8a6` fixed across the other
/// differentials.
fn vector_tier_live() -> bool {
    if cfg!(target_arch = "aarch64") {
        archmage::NeonToken::summon().is_some()
    } else {
        X64V3Token::summon().is_some()
    }
}

/// Drive one arm's whole matrix through every token permutation and assert
/// byte-equality against the first permutation, plus the non-vacuity floors.
fn run_arm(arm: Arm, min_changed_pct: u32) {
    // Fire the AOM_FORCE_SCALAR pin (if set) BEFORE the harness takes over
    // token state, so both a vector and a scalar permutation run in either
    // dispatch mode.
    let _ = aom_dsp::dispatch::scalar_forced();

    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            X64V3Token::summon().is_some(),
            "x86-64 CI must have AVX2 for this differential to be non-vacuous"
        );
    }

    let mut reference: Option<Vec<(String, Vec<u8>)>> = None;
    let mut simd_perms = 0usize;
    let mut scalar_perms = 0usize;
    let mut cov = Cov::default();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        if vector_tier_live() {
            simd_perms += 1;
        } else {
            scalar_perms += 1;
        }
        let (cur, c) = all_outputs(arm);
        cov = c;
        match reference.as_ref() {
            None => reference = Some(cur),
            Some(r) => {
                assert_eq!(
                    cur.len(),
                    r.len(),
                    "permutation [{tier}] produced a different number of cells"
                );
                for ((cl, cd), (rl, rd)) in cur.iter().zip(r.iter()) {
                    assert_eq!(
                        cd, rd,
                        "permutation [{tier}] diverged from the reference at cell '{cl}' \
                         (reference cell '{rl}')"
                    );
                }
            }
        }
    });

    eprintln!("inv_txfm2d_add_u8 {arm:?}-arm SIMD==scalar parity: {report}; simd_perms={simd_perms} scalar_perms={scalar_perms} cov={cov:?}");

    // --- the arm was actually reached -------------------------------------
    assert!(cov.cells > 0, "{arm:?} arm swept no cells at all: {cov:?}");
    assert!(
        cov.vector_cells > 0,
        "{arm:?} arm swept no cell whose width makes the vector path \
         unconditionally eligible (col_n % 8 == 0): {cov:?}"
    );
    assert!(
        cov.wide_cells > 0,
        "{arm:?} arm never reached a full-width lane group: {cov:?}"
    );
    assert!(
        cov.lr_cells > 0,
        "{arm:?} arm never exercised the lr_flip lane reversal: {cov:?}"
    );
    // Liveness: the transform must actually move pixels, else both sides could
    // be agreeing on "the residual is zero, copy the prediction". Measured
    // ~100% on both arms (every input pattern is non-zero); the floor is set
    // well below that so a legitimate distribution shift does not trip it.
    assert!(
        cov.changed * 100 >= cov.cases * min_changed_pct,
        "pixel-changing floor ({min_changed_pct}%): {cov:?}"
    );

    // --- the permutation set was actually non-vacuous ----------------------
    assert!(
        simd_perms >= 1,
        "the SIMD permutation ({}) must run at least once — a passing run with \
         zero vector permutations compares the scalar path against itself, \
         which is exactly the state the transform differential was in before \
         d3feb5d. On aarch64 this needs archmage's `testable_dispatch` \
         dev-feature, else baseline neon is excluded from the permutation set.",
        if cfg!(target_arch = "aarch64") { "neon" } else { "v3/AVX2" }
    );
    assert!(scalar_perms >= 1, "the all-off (scalar) permutation must run at least once");
    assert!(report.permutations_run >= 2, "need >=2 permutations to compare SIMD vs scalar");
}

/// The **i16 arm** — `lowbd16::inv_col_pass_u8_i16`, 16 columns per vector,
/// the audited DCT4/8/16/32/64 column kernels. Live NEON code since `13b7c21`.
///
/// Note the `ud_flip` branch of `inv_col_pass_u8_i16_core` is DEAD BY
/// CONSTRUCTION on this arm and no sweep can reach it: `ud_flip` is set only
/// for the FLIPADST-vertical tx_types (4, 6, 8, 14), and every one of those has
/// `VTX_TAB != 0`, i.e. is on the i32 arm. That is asserted below so the fact
/// stays true or gets noticed — it is exactly the "one variant is a no-op by
/// construction" trap that a liveness floor set from the other arm's numbers
/// would walk into.
#[test]
fn inv_txfm2d_add_u8_i16_arm_simd_bit_identical_to_scalar_at_every_tier() {
    assert!(
        (0..16).all(|t| VTX_TAB[t] != 0 || !FLIP_CFG[t].0),
        "a DCT-column tx_type gained ud_flip — the i16 arm's ud_flip branch is \
         now reachable and this sweep must be widened to cover it"
    );
    run_arm(Arm::I16, 90);
}

/// The **i32 arm** — `inv_col_pass_u8`, 8 columns per vector: every ADST /
/// FLIPADST / IDTX column kernel, plus (via `FLIP_CFG`) the only cells that
/// reach the `ud_flip` row reversal at all.
#[test]
fn inv_txfm2d_add_u8_i32_arm_simd_bit_identical_to_scalar_at_every_tier() {
    run_arm(Arm::I32, 90);
}

/// Pin the transcribed [`VTX_TAB`] against the crate's own public API, so the
/// arm split above is verified rather than assumed.
///
/// `av1_txfm_type_ls` (`transform/txfm2d.rs`) has NO ADST/FLIPADST entry at 32
/// points (`[3, -1, -1, 11]`) and only DCT at 64 (`[4, -1, -1, -1]`). So over a
/// tx_size whose ROW side accepts everything, `inv_txfm_valid` reads out the
/// column class directly:
///   * tx_size 17 = 16x64 (row idx 2, all four classes valid) → valid iff DCT;
///   * tx_size 15 = 8x32  (row idx 1, all four classes valid) → valid iff
///     DCT or IDTX.
/// Together those identify `vtx == 0` exactly — which is the whole predicate
/// the arm split rests on.
#[test]
fn vtx_tab_matches_public_validity() {
    assert_eq!((W[17], H[17]), (16, 64));
    assert_eq!((W[15], H[15]), (8, 32));
    for tx_type in 0..16usize {
        assert_eq!(
            inv_txfm_valid(tx_type, 17),
            VTX_TAB[tx_type] == 0,
            "VTX_TAB[{tx_type}] disagrees with inv_txfm_valid at 16x64 (DCT-only columns)"
        );
        assert_eq!(
            inv_txfm_valid(tx_type, 15),
            VTX_TAB[tx_type] == 0 || VTX_TAB[tx_type] == 3,
            "VTX_TAB[{tx_type}] disagrees with inv_txfm_valid at 8x32 (DCT/IDTX columns)"
        );
    }
    // And the split is a real partition: both sides non-empty.
    assert!((0..16).any(|t| arm_of(t) == Arm::I16));
    assert!((0..16).any(|t| arm_of(t) == Arm::I32));
}
