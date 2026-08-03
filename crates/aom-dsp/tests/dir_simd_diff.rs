//! i16-lane-vs-scalar differential for the **directional** highbd intra
//! predictors `z1_high` / `z2_high` / `z3_high`, at every archmage token
//! permutation, over the full directional domain: every AV1 transform shape,
//! every signalled angle (through the real `dr_intra_derivative` table), both
//! `upsample` values, bd 8 / 10 / 12 sample ranges, tight AND padded stride.
//!
//! The dispatching entries run `intra::dir_simd`'s i16 kernel wherever the
//! runtime tap bound admits it; the `*_scalar` entries are the fixed,
//! never-dispatched C transcriptions. They must agree byte-for-byte at EVERY
//! tier.
//!
//! Playbook §1 — the probes are deliberately asymmetric. The vector form is a
//! re-association (`a0*(32-s) + a1*s == (a0<<5) + (a1-a0)*s`), and a FLAT edge
//! makes `a1 - a0 == 0`, i.e. it is invariant under exactly the term this
//! change introduces. A constant-edge probe would pass against a completely
//! broken `shift` handling. So every probe below has structure, and the
//! `edge_kind` sweep includes ramps and sawtooths that put energy in the
//! difference term.
//!
//! Non-vacuity is asserted twice: `simd_perms >= 1` (a vector tier really ran —
//! `docs/SIMD_REACH_AUDIT_2026-07-28.md` F4) and `vec_cells >= 1` per predictor
//! (the runtime gate really fired, so the comparison is not scalar-vs-scalar).

use aom_dsp::intra::dir::{
    DR_INTRA_DERIVATIVE, EdgeRef16, get_dx, get_dy, z1_high, z1_high_scalar, z2_high,
    z2_high_scalar, z3_high, z3_high_scalar,
};
use aom_dsp::intra::edge::use_upsample;
use archmage::SimdToken;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

const PAD: usize = 16;
const BUF: usize = 160;

/// AV1 `TX_SIZE` dims (`tx_size_wide` / `tx_size_high`) — the exact shapes the
/// predictors are ever called with.
const TX_DIMS: [(usize, usize); 19] = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
];

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

/// Fill an edge buffer with a probe of the given shape, clamped to `maxv`.
/// None of these is flat except `kind == 5`, which is kept only as a control.
fn fill_edge(buf: &mut [u16], kind: usize, maxv: u32, rng: &mut Rng) {
    for (i, e) in buf.iter_mut().enumerate() {
        *e = match kind {
            0 => rng.upto(maxv + 1) as u16,                                  // dense random
            1 => ((i as u32 * 37) % (maxv + 1)) as u16,                      // ramp
            2 => {
                if i % 2 == 0 {
                    maxv as u16
                } else {
                    0
                }
            } // max sawtooth
            3 => (maxv - (i as u32 * 11) % (maxv + 1)) as u16,               // reverse ramp
            4 => {
                if i < BUF / 2 {
                    0
                } else {
                    maxv as u16
                }
            } // step
            _ => maxv as u16,                                                 // flat (control)
        };
    }
}

/// The signalled angles: `mode_to_angle_map` bases plus `angle_delta * 3`.
fn angles() -> Vec<i32> {
    let mut v = Vec::new();
    for base in [45, 67, 90, 113, 135, 157, 180, 203] {
        for d in -3..=3 {
            v.push(base + d * 3);
        }
    }
    v
}

#[test]
fn dir_highbd_simd_bit_identical_to_scalar_at_every_tier() {
    let mut simd_perms = 0usize;
    let (mut z1_vec, mut z2_vec, mut z3_vec) = (0usize, 0usize, 0usize);
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_tier| {
        if if cfg!(target_arch = "aarch64") {
            archmage::NeonToken::summon().is_some()
        } else {
            archmage::X64V3Token::summon().is_some()
        } {
            simd_perms += 1;
        }
        let mut rng = Rng(0x_d12_5117_0803_2026);
        let mut above = vec![0u16; BUF];
        let mut left = vec![0u16; BUF];
        for &bd in &[8i32, 10, 12] {
            let maxv = (1u32 << bd) - 1;
            for kind in 0..6usize {
                fill_edge(&mut above, kind, maxv, &mut rng);
                fill_edge(&mut left, (kind + 3) % 6, maxv, &mut rng);
                let a = EdgeRef16::new(&above, PAD);
                let l = EdgeRef16::new(&left, PAD);
                for &(bw, bh) in &TX_DIMS {
                    for &stride in &[bw, bw + 5] {
                        // `up` is DERIVED exactly as `build_directional_intra_high`
                        // derives it, so the grid only contains reachable
                        // (shape, angle, upsample) triples. Sweeping `up` freely
                        // walks the edge buffers off their ends — in the SCALAR
                        // kernel too, which is how this was caught.
                        for &filter_type in &[0i32, 1] {
                            for angle in angles() {
                                let (dx, dy) = (get_dx(angle), get_dy(angle));
                                let up = use_upsample(
                                    bw as i32, bh as i32, angle - 90, filter_type,
                                );
                                let upl = use_upsample(
                                    bh as i32, bw as i32, angle - 180, filter_type,
                                );
                                let n = bh * stride;
                                let (mut got, mut want) = (vec![0u16; n], vec![0u16; n]);
                                if angle > 0 && angle < 90 {
                                    z1_high(&mut got, stride, bw, bh, &a, up, dx);
                                    z1_high_scalar(&mut want, stride, bw, bh, &a, up, dx);
                                    if up == 0 && bw >= 8 && bd == 8 {
                                        z1_vec += 1;
                                    }
                                    assert_eq!(
                                        got, want,
                                        "z1 {bw}x{bh} stride={stride} up={up} angle={angle} \
                                         bd={bd} kind={kind}"
                                    );
                                } else if angle > 90 && angle < 180 {
                                    z2_high(&mut got, stride, bw, bh, &a, &l, up, upl, dx, dy);
                                    z2_high_scalar(
                                        &mut want, stride, bw, bh, &a, &l, up, upl, dx, dy,
                                    );
                                    if up == 0 && bd == 8 {
                                        z2_vec += 1;
                                    }
                                    assert_eq!(
                                        got, want,
                                        "z2 {bw}x{bh} stride={stride} up_a={up} up_l={upl} \
                                         angle={angle} bd={bd} kind={kind}"
                                    );
                                } else if angle > 180 && angle < 270 {
                                    z3_high(&mut got, stride, bw, bh, &l, upl, dy);
                                    z3_high_scalar(&mut want, stride, bw, bh, &l, upl, dy);
                                    if upl == 0 && bh >= 8 && bd == 8 {
                                        z3_vec += 1;
                                    }
                                    assert_eq!(
                                        got, want,
                                        "z3 {bw}x{bh} stride={stride} up={upl} angle={angle} \
                                         bd={bd} kind={kind}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    eprintln!(
        "dir highbd i16 parity: {report}  (shape-eligible bd8 cells z1={z1_vec} z2={z2_vec} z3={z3_vec})"
    );
    assert!(
        simd_perms >= 1,
        "the SIMD permutation must run at least once — a passing run with zero \
         vector permutations compares the scalar path against itself"
    );
    assert!(report.permutations_run >= 2);
    // REACH (playbook §1): the gate must actually admit cells, or this whole
    // file is scalar-vs-scalar. The forward-transform lever's `txfm2d_simd_perm_diff`
    // hit exactly that — its inputs were outside the new path's domain.
    assert!(
        z1_vec > 0 && z2_vec > 0 && z3_vec > 0,
        "no shape-eligible bd8 cells in the grid: z1={z1_vec} z2={z2_vec} z3={z3_vec}"
    );
}

/// The angle table is what makes `up`/`dx`/`dy` reachable; pin that the probe
/// grid above is not silently degenerate.
#[test]
fn the_angle_grid_reaches_all_three_predictors() {
    let (mut n1, mut n2, mut n3) = (0, 0, 0);
    for angle in angles() {
        if angle > 0 && angle < 90 {
            n1 += 1;
            assert!(get_dx(angle) > 0);
        } else if angle > 90 && angle < 180 {
            n2 += 1;
            assert!(get_dx(angle) > 0 && get_dy(angle) > 0);
        } else if angle > 180 && angle < 270 {
            n3 += 1;
            assert!(get_dy(angle) > 0);
        }
    }
    assert!(n1 >= 8 && n2 >= 8 && n3 >= 4, "{n1}/{n2}/{n3}");
    assert_ne!(DR_INTRA_DERIVATIVE[3], 0);
}
