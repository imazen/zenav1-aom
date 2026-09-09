//! Differential harness for the superres denom-SELECTION math vs the faithful
//! C facade (`superres_shim.c`, which calls the real exported
//! `av1_fwd_txfm2d_16x4_c` + `av1_convert_qindex_to_q`).
//!
//! Covers `analyze_hor_freq` (bit-exact f64 energy vector), the
//! `get_superres_denom_from_qindex_energy` threshold walk, and the full KEY
//! QTHRESH arm. The end-to-end gate (`encoder_gate_superres_e2e.rs`) validates
//! the same selection against real `aomenc`; this isolates the arithmetic so a
//! denom mismatch localizes to the analysis vs the parse/pack wiring.

use aom_encode::superres_select::{
    analyze_hor_freq, get_superres_denom_from_qindex_energy, superres_denom_qthresh_key,
};
use aom_sys_ref as c;

/// Deterministic content generators, each stressing a different horizontal
/// frequency profile (so the energy vector — and the derived denom — is
/// non-trivial across the k bins).
fn gen_content(kind: u32, w: usize, h: usize, bd: u8) -> Vec<u16> {
    let maxv = (1u32 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let v: u32 = match kind {
                0 => ((r * 37 + col * 23) as u32) & maxv, // smooth-ish gradient
                1 => {
                    // vertical stripes — strong high horizontal frequency
                    if (col & 3) < 2 { maxv } else { 0 }
                }
                2 => {
                    // diagonal ramp + speckle
                    let base = ((r * 3 + col * 5) as u32) & maxv;
                    let hf = if (r ^ col) & 1 == 1 { maxv / 8 } else { 0 };
                    (base ^ hf) & maxv
                }
                3 => {
                    // mid-frequency sinusoid-like blocks
                    let phase = (col / 5) & 1;
                    if phase == 0 { (maxv * 3) / 4 } else { maxv / 4 }
                }
                _ => ((r.wrapping_mul(101).wrapping_add(col.wrapping_mul(197))) as u32) & maxv,
            };
            y[r * w + col] = v.min(maxv) as u16;
        }
    }
    y
}

fn assert_energy_bit_exact(label: &str, port: &[f64; 16], reference: &[f64; 16]) {
    for k in 1..16 {
        assert_eq!(
            port[k].to_bits(),
            reference[k].to_bits(),
            "{label}: energy[{k}] port={} ref={} (bits {:#x} vs {:#x})",
            port[k],
            reference[k],
            port[k].to_bits(),
            reference[k].to_bits(),
        );
    }
}

#[test]
fn analyze_hor_freq_matches_c_facade() {
    c::ref_init();
    for &bd in &[8u8, 10, 12] {
        for &(w, h) in &[
            (196usize, 196usize),
            (128, 128),
            (256, 256),
            (64, 96),
            (200, 60),
        ] {
            for kind in 0..5u32 {
                let src = gen_content(kind, w, h, bd);
                let port = analyze_hor_freq(&src, w, h, w, bd);
                let reference = c::ref_superres_analyze_hor_freq(&src, w, h, w, bd);
                assert_energy_bit_exact(
                    &format!("analyze bd{bd} {w}x{h} kind{kind}"),
                    &port,
                    &reference,
                );
            }
        }
    }
}

#[test]
fn analyze_hor_freq_matches_c_facade_strided() {
    // A strided luma plane (padding to the right of the visible width) must not
    // change the analysis — the window reads only the first `width` cols.
    c::ref_init();
    let (w, h, stride, bd) = (196usize, 196usize, 224usize, 8u8);
    for kind in 0..5u32 {
        let tight = gen_content(kind, w, h, bd);
        let mut strided = vec![0u16; stride * h];
        for r in 0..h {
            strided[r * stride..r * stride + w].copy_from_slice(&tight[r * w..r * w + w]);
            // fill the padding with an out-of-window sentinel
            for c_ in w..stride {
                strided[r * stride + c_] = 123;
            }
        }
        let port = analyze_hor_freq(&strided, w, h, stride, bd);
        let reference = c::ref_superres_analyze_hor_freq(&strided, w, h, stride, bd);
        assert_energy_bit_exact(&format!("strided kind{kind}"), &port, &reference);
    }
}

#[test]
fn denom_from_qindex_energy_matches_c_facade() {
    c::ref_init();
    // Build real energy vectors from content, then sweep qindex/thresholds.
    for &bd in &[8u8, 10, 12] {
        for kind in 0..5u32 {
            let src = gen_content(kind, 196, 196, bd);
            let energy = analyze_hor_freq(&src, 196, 196, 196, bd);
            let ref_energy = c::ref_superres_analyze_hor_freq(&src, 196, 196, 196, bd);
            assert_energy_bit_exact("denom-setup", &energy, &ref_energy);
            for &q in &[0i32, 16, 40, 96, 128, 180, 224, 255] {
                for &(tq, tp) in &[(0.012f64, 0.2f64), (0.008, 0.2), (0.02, 0.1)] {
                    let port = get_superres_denom_from_qindex_energy(q, &energy, tq, tp);
                    let reference = c::ref_superres_denom_from_qindex_energy(q, &energy, tq, tp);
                    assert_eq!(
                        port, reference,
                        "denom bd{bd} kind{kind} q{q} tq{tq} tp{tp}: port {port} ref {reference}"
                    );
                }
            }
        }
    }
}

#[test]
fn qthresh_key_full_selection_matches_c_facade() {
    c::ref_init();
    // Full KEY QTHRESH arm (single-frame AOM_Q envelope: frames_to_key<=1, no
    // screen content). Sweep q vs kf_qthresh so both branches (q<=thresh -> 8,
    // and q>thresh -> energy derivation) are exercised.
    for &bd in &[8u8, 10, 12] {
        for kind in 0..5u32 {
            let src = gen_content(kind, 196, 196, bd);
            for &q in &[24i32, 96, 128, 180, 240] {
                for &kf_qthresh in &[16i32, 128, 200] {
                    let port = superres_denom_qthresh_key(
                        &src, 196, 196, 196, bd, q, kf_qthresh, false, true,
                    );
                    let reference = c::ref_superres_denom_qthresh_key(
                        &src, 196, 196, 196, bd, q, kf_qthresh, false, true,
                    );
                    assert_eq!(
                        port, reference,
                        "qthresh bd{bd} kind{kind} q{q} kfqt{kf_qthresh}: port {port} ref {reference}"
                    );
                }
            }
        }
    }
}
