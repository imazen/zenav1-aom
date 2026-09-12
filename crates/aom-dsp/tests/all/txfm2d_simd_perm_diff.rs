//! SIMD-vs-scalar permutation-equality integration test for the 2-D transform
//! drivers (`av1_inv_txfm2d_add` / `av1_fwd_txfm2d`).
//!
//! The per-kernel unit differential (`simd::tests` in the lib) pins each 1-D
//! lane kernel == scalar; THIS test pins the 2-D PASS PLUMBING that lives in
//! the drivers, not the kernels — the lr/ud flips, the `clamp_buf` clamps, the
//! `round_shift_array` shifts, the NewSqrt2/NewInvSqrt2 rect scalings, the
//! transpose loads/stores (incl. the W=4 per-lane scatter/gather tail) and the
//! final `highbd_clip_pixel_add` — SIMD == scalar, over the FULL driver clamp
//! domain the vs-C harness cannot reach.
//!
//! Why a separate SIMD-vs-scalar test (not just the vs-C harnesses): the vs-C
//! differentials (`inv_txfm2d_diff`, `txfm2d_diff`) cap inverse coefficients at
//! ±2^16 because C's `half_btf` sums two i32 products in an i32 and OVERFLOWS
//! (undefined behaviour) at the true bd12 clamp bound ±2^19. This test compares
//! the port's SIMD path to the port's SCALAR path — neither has that UB — so it
//! drives inverse coefficients across ±2^20 (they clamp to ±2^19, the |p0 + p1|
//! maximiser that the exact-i64 `hb` recipe exists to survive) and forward
//! residuals across the full i16 range. This is the zero-tolerance guarantee:
//! on crafted-but-decodable streams that push dequantised coefficients to the
//! clamp bounds, the vector path must reproduce the scalar path bit-for-bit.
//!
//! Method: `for_each_token_permutation` runs the whole (tx_size × tx_type × bd
//! × input) matrix once per token permutation, feeding IDENTICAL inputs (the
//! RNG is re-seeded with the same constant at the top of every permutation, so
//! dispatch is the only thing that varies). Every permutation's complete output
//! set must byte-match the first permutation's. The harness always includes an
//! all-off (scalar) permutation and — on AVX2 CI — a v3 (SIMD) permutation, so
//! the equality chain transitively pins SIMD == scalar. Non-vacuous: asserts a
//! SIMD permutation and a scalar permutation both ran.

use aom_dsp::transform::inv_txfm2d::{av1_inv_txfm2d_add, inv_input_len, inv_txfm_valid};
use aom_dsp::transform::txfm2d::{av1_fwd_txfm2d, fwd_txfm_valid};
use archmage::prelude::*;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use archmage::X64V3Token;

const W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

const SEED: u64 = 0x_5119_d1ff_2d00_0001;

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
    /// Inverse dequantised coefficient pushed to ±2^20 — well past the vs-C
    /// harness's ±2^16, so a large fraction land on the driver's ±2^19 clamp
    /// (the bd12 row `clamp_buf`), which is exactly the `half_btf` i64-sum
    /// stress the vector recipe must survive. Both paths clamp identically, so
    /// there is no UB (unlike C).
    fn coeff(&mut self) -> i32 {
        (self.next() % (1 << 21)) as i32 - (1 << 20)
    }
    /// Full-range i16 forward residual (far beyond the [-255, 255] encoder
    /// range) — stresses the forward col shl-clamp + butterfly plumbing.
    fn residual(&mut self) -> i16 {
        self.next() as i16
    }
    /// bd8 forward residual: `src - pred` over u8, i.e. [-255, 255]. This is
    /// the ONLY range the i16-lane forward passes accept, so without this arm
    /// the forward half of this test would exercise the i32 path exclusively
    /// (the full-range arm above is over `M*` for every kernel and every one
    /// of its blocks declines). Playbook 1: a test that cannot reach the code
    /// it is supposed to guard is not a test of it.
    fn residual_bd8(&mut self) -> i16 {
        (self.next() % 511) as i16 - 255
    }
    /// Residuals across ±512 — the fused 4x4 SIMD kernel's runtime gate bound.
    /// The bd8 arm (±255) is comfortably inside it and the full-range i16 arm
    /// is comfortably outside, so without this band the accept-side edge of the
    /// gate is never driven: a block must have EVERY lane at |in| <= 512 to
    /// take the vector path, which full-range randoms essentially never do.
    fn residual_gate(&mut self) -> i16 {
        (self.next() % 1025) as i16 - 512
    }
    /// Inverse coefficients across ±4096 — the fused 4x4 INVERSE kernel's
    /// runtime gate bound (|input| <= 4096 keeps every i16 intermediate
    /// unsaturated through both passes). The ±2^20 `coeff` arm declines every
    /// 4x4 block, so without this band the accept side of the inverse gate is
    /// never driven.
    fn coeff_gate(&mut self) -> i32 {
        (self.next() % 8193) as i32 - 4096
    }
    /// Inverse coefficients across ±737 — inside the TIGHTEST fused 8x8
    /// inverse gate (`Adst8 -> Adst8` allows |input| <= 737; every other pair
    /// is looser), so this arm accepts all nine (row, col) kernel pairs.
    fn coeff_gate8(&mut self) -> i32 {
        (self.next() % 1475) as i32 - 737
    }
    fn pixel(&mut self, bd: i32) -> u16 {
        (self.next() % (1u64 << bd)) as u16
    }
}

/// Exact clamp-bound spike patterns for the inverse coefficient buffer — the
/// `|p0 + p1|` maximisers plus a DC-only spike. `k` selects the pattern.
fn inv_spike(k: usize, i: usize, len: usize) -> i32 {
    const B: i32 = 1 << 19; // the bd12 row clamp bound
    match k {
        0 => B,                                             // all +bound
        1 => -B,                                            // all -bound
        2 => {
            if i % 2 == 0 {
                B
            } else {
                -B
            }
        } // alternating
        3 => {
            if i % 2 == 0 {
                -B
            } else {
                B
            }
        } // alternating (other phase)
        4 => {
            if i == 0 {
                1 << 20
            } else {
                0
            }
        } // DC spike beyond the clamp
        5 => {
            if i + 1 == len {
                -(1 << 20)
            } else {
                0
            }
        } // last-coeff spike
        _ => B - 1,                                         // ±(2^19 - 1), the exact hi bound
    }
}

/// Gate-edge spikes for the fused 4x4 INVERSE SIMD kernel — the exact accept
/// bound (±4096) and just past it (±4097), plus a partial block where only
/// some lanes are over, so both sides of the runtime decline are driven.
fn inv_spike_gate(k: usize, i: usize) -> i32 {
    match k {
        0 => 4096,
        1 => -4096,
        2 => 4097, // one over -> decline
        _ => {
            if i % 4 == 0 {
                4096
            } else {
                -3500
            }
        }
    }
}

/// Gate-edge spikes for the fused 8x8 INVERSE SIMD kernel — the tightest
/// accept bound (±737, `Adst8 -> Adst8`) and the `Dct8 -> Dct8` edge
/// (±2347/±2348), plus a partial block where only some lanes are over.
fn inv_spike_gate8(k: usize, i: usize) -> i32 {
    match k {
        0 => 737,
        1 => -737,
        2 => 738,  // over Adst8->Adst8's bound
        3 => 2347, // Dct8->Dct8's exact bound
        4 => 2348, // one over Dct8->Dct8
        _ => {
            if i % 8 == 0 {
                2348
            } else {
                -700
            }
        }
    }
}

/// Gate-edge spikes for the fused 4x8/8x4 INVERSE SIMD kernel — the tightest
/// accept bound across both bound tables (±3282, `*8 -> Adst4` at 8x4) and
/// just past it (±3283), plus ±6204/±8000 which only the looser pairs accept
/// (drives per-pair accept/decline splits) and a partial block over the bound.
fn inv_spike_gate48(k: usize, i: usize) -> i32 {
    match k {
        0 => 3282,
        1 => -3282,
        2 => 3283, // over the tightest bound -> decline at 8x4 adst4 cols
        3 => 8000, // accepted by some pairs only
        _ => {
            if i % 4 == 0 {
                6205
            } else {
                -6000
            }
        }
    }
}

/// bd8-range spike patterns for the forward residual buffer — the
/// coefficient-sum sign vertices at the exact ±255 bd8 extreme, which is where
/// the i16-lane forward passes are closest to their proven bound.
fn fwd_spike_bd8(k: usize, i: usize) -> i16 {
    match k {
        0 => 255,
        1 => -255,
        2 => {
            if i % 2 == 0 {
                255
            } else {
                -255
            }
        }
        _ => {
            if (i * 7 + i / 5) % 3 == 0 {
                -255
            } else {
                255
            }
        }
    }
}

/// Full-range i16 spike patterns for the forward residual buffer.
fn fwd_spike(k: usize, i: usize) -> i16 {
    match k {
        0 => i16::MAX,
        1 => i16::MIN,
        2 => {
            if i % 2 == 0 {
                i16::MAX
            } else {
                i16::MIN
            }
        }
        _ => {
            if i == 0 {
                i16::MAX
            } else {
                0
            }
        }
    }
}

/// Gate-edge spikes for the fused forward SIMD kernels — the exact accept
/// bounds (±512 for the 4x4, ±511 for the 8x8, ±1023 for the 4x8/8x4) and
/// just past them (±513 / ±1024 / the partial-block patterns), so both sides
/// of each runtime decline are driven.
fn fwd_spike_gate(k: usize, i: usize) -> i16 {
    match k {
        0 => 512,
        1 => -512,
        2 => 513, // one over both bounds -> decline
        4 => {
            if i % 4 == 0 {
                511 // inside both bounds -> accept at the 8x8 edge
            } else {
                -300
            }
        }
        5 => 1023,  // the rect48 accept edge
        6 => -1023,
        7 => 1024, // just over -> decline
        8 => -1024,
        9 => {
            if i % 8 == 0 {
                1023 // inside the rect48 bound -> accept there
            } else {
                -513 // over the 4x4/8x8 bounds -> decline there
            }
        }
        10 => 285,  // over 16x16 DCT->ADST/ADST->DCT (284) -> decline those pairs
        11 => -285,
        12 => 316,  // over ADST->ADST (315)
        13 => -724, // over DCT->IDTX (723)
        14 => 804,  // over ADST->IDTX (803)
        15 => -1137, // over IDTX->ADST (1136)
        16 => 2895, // IDTX->IDTX's exact edge -> accept
        _ => {
            if i % 4 == 0 {
                512 // over the 8x8 bound -> decline there, accept at the 4x4 edge
            } else {
                -300
            }
        }
    }
}

const RAND_REPS: usize = 5;
const INV_SPIKES: usize = 7;
const FWD_SPIKES: usize = 5;

/// Run the entire (tx_size × tx_type × bd × input) matrix through the public
/// 2-D entries under the CURRENT token permutation, collecting every output
/// (tagged with a human-readable label). Inputs are deterministic (fixed seed),
/// so only the SIMD/scalar dispatch differs between calls.
fn all_outputs() -> Vec<(String, Vec<i64>)> {
    let mut rng = Rng(SEED);
    let mut out: Vec<(String, Vec<i64>)> = Vec::new();

    for tx_size in 0..19usize {
        let (w, h) = (W[tx_size], H[tx_size]);

        // ---------- inverse: coeffs ±2^20, tight + strided dest ----------
        for tx_type in 0..16usize {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            let ilen = inv_input_len(tx_size);
            for bd in [8i32, 10, 12] {
                // strides: tight (w) and an odd strided dest (w + 3) to drive
                // the per-lane scatter tail of the transpose store.
                for &stride in &[w, w + 3] {
                    let mut push_inv = |label: String, input: &[i32], rng: &mut Rng| {
                        let dest: Vec<u16> = (0..h * stride).map(|_| rng.pixel(bd)).collect();
                        let mut got = dest.clone();
                        av1_inv_txfm2d_add(input, &mut got, stride, tx_type, tx_size, bd);
                        out.push((label, got.iter().map(|&x| x as i64).collect()));
                    };
                    for rep in 0..RAND_REPS {
                        let input: Vec<i32> = (0..ilen).map(|_| rng.coeff()).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} rand{rep}"),
                            &input,
                            &mut rng,
                        );
                    }
                    // Accepted-side band of the inv-4x4 i16 gate (all other
                    // sizes/types just take their normal paths on these).
                    for rep in 0..2 {
                        let input: Vec<i32> =
                            (0..ilen).map(|_| rng.coeff_gate()).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} randg{rep}"),
                            &input,
                            &mut rng,
                        );
                    }
                    // Accepted side of the fused 8x8 i16 gate: ±737 is inside
                    // the tightest (row, col) pair bound.
                    for rep in 0..2 {
                        let input: Vec<i32> =
                            (0..ilen).map(|_| rng.coeff_gate8()).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} randg8_{rep}"),
                            &input,
                            &mut rng,
                        );
                    }
                    for k in 0..4usize {
                        let input: Vec<i32> =
                            (0..ilen).map(|i| inv_spike_gate(k, i)).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} gspike{k}"),
                            &input,
                            &mut rng,
                        );
                    }
                    for k in 0..6usize {
                        let input: Vec<i32> =
                            (0..ilen).map(|i| inv_spike_gate8(k, i)).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} gspike8_{k}"),
                            &input,
                            &mut rng,
                        );
                    }
                    // Gate-edge spikes for the fused 4x8/8x4 i16 kernel.
                    for k in 0..5usize {
                        let input: Vec<i32> =
                            (0..ilen).map(|i| inv_spike_gate48(k, i)).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} gspike48_{k}"),
                            &input,
                            &mut rng,
                        );
                    }
                    for k in 0..INV_SPIKES {
                        let input: Vec<i32> =
                            (0..ilen).map(|i| inv_spike(k, i, ilen)).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} spike{k}"),
                            &input,
                            &mut rng,
                        );
                    }
                }
            }
        }

        // ---------- forward: full-range i16 residuals ----------
        for tx_type in 0..16usize {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            let mut push_fwd = |label: String, input: &[i16]| {
                let mut got = vec![0i32; w * h];
                av1_fwd_txfm2d(input, &mut got, w, tx_type, tx_size);
                out.push((label, got.iter().map(|&x| x as i64).collect()));
            };
            for rep in 0..RAND_REPS {
                let input: Vec<i16> = (0..w * h).map(|_| rng.residual()).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} rand{rep}"), &input);
            }
            for k in 0..FWD_SPIKES {
                let input: Vec<i16> = (0..w * h).map(|i| fwd_spike(k, i)).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} spike{k}"), &input);
            }
            // bd8 arm — the domain the i16-lane forward passes accept.
            for rep in 0..RAND_REPS {
                let input: Vec<i16> = (0..w * h).map(|_| rng.residual_bd8()).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} bd8rand{rep}"), &input);
            }
            for k in 0..FWD_SPIKES {
                let input: Vec<i16> = (0..w * h).map(|i| fwd_spike_bd8(k, i)).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} bd8spike{k}"), &input);
            }
            // Gate-edge arm — ±512 randoms (inside the fused 4x4 kernel's
            // bound) plus spikes at and just past each bound (±512/±511/
            // ±1023 edges for the 4x4/8x8/rect48 kernels, ±513/±1024 on the
            // decline side).
            for rep in 0..RAND_REPS {
                let input: Vec<i16> = (0..w * h).map(|_| rng.residual_gate()).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} gaterand{rep}"), &input);
            }
            for k in 0..17usize {
                let input: Vec<i16> = (0..w * h).map(|i| fwd_spike_gate(k, i)).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} gatespike{k}"), &input);
            }
        }
    }
    out
}

#[test]
fn txfm2d_simd_equals_scalar_at_every_permutation() {
    // Serialise: this sweep permutes PROCESS-GLOBAL dispatch
    // state; see `crate::dispatch_serial`.
    let _serial = crate::dispatch_serial::dispatch_serial();
    // Fire the AOM_FORCE_SCALAR pin (if set) BEFORE the permutation harness —
    // the harness then owns token state, so both a SIMD and a scalar
    // permutation run in either dispatch mode.
    let _ = aom_dsp::dispatch::scalar_forced();

    let mut reference: Option<Vec<(String, Vec<i64>)>> = None;
    let mut simd_perms = 0usize;
    let mut scalar_perms = 0usize;

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        // "Is a vector tier live in this permutation" is per-architecture: the
        // transform's vector path is `X64V3` on x86-64 and `Neon` on aarch64.
        // Testing only `X64V3Token` made every aarch64 permutation count as
        // scalar (that token is a stub off x86), which is why the non-vacuity
        // assert below failed on ARM.
        let simd_live = if cfg!(target_arch = "aarch64") {
            archmage::NeonToken::summon().is_some()
        } else {
            X64V3Token::summon().is_some()
        };
        if simd_live {
            simd_perms += 1;
        } else {
            scalar_perms += 1;
        }
        let cur = all_outputs();
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

    eprintln!(
        "txfm2d SIMD==scalar parity: {report}; simd_perms={simd_perms} scalar_perms={scalar_perms}"
    );
    // Non-vacuity: a SIMD permutation must have run, and both a SIMD and a
    // scalar permutation must have been compared so the equality chain actually
    // pins the vector path against the scalar path. The tier named here is
    // per-architecture (v3 on x86-64, neon on aarch64); reaching the neon arm
    // requires archmage's `testable_dispatch` dev-feature, because baseline
    // `neon` is otherwise excluded from the permutation set — see this crate's
    // Cargo.toml.
    assert!(
        simd_perms >= 1,
        "the SIMD permutation ({}) must run at least once — if this fails on \
         aarch64, archmage's `testable_dispatch` dev-feature is not enabled and \
         baseline neon was excluded from the permutations",
        if cfg!(target_arch = "aarch64") { "neon" } else { "v3/AVX2" }
    );
    assert!(scalar_perms >= 1, "the all-off (scalar) permutation must run at least once");
    assert!(report.permutations_run >= 2, "need >=2 permutations to compare SIMD vs scalar");
}

