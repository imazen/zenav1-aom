//! Differential harness for the bd8 LOWBD (u8 pixel) inverse 2-D transform +
//! reconstruction vs C libaom v3.14.1 AND vs the port's own highbd (u16) bd8
//! path. This is the byte-identity PROOF for the lowbd decode pipeline's
//! transform lever: the narrower u8 destination must reconstruct the SAME pixel
//! every C (and the u16 port) does at bit depth 8.
//!
//! Two independent oracles, both asserted:
//!   1. `u8_out[i] as u16 == ref_inv_txfm2d_add(bd=8)[i]`  — vs the real
//!      exported C `av1_inv_txfm2d_add_*_c`.
//!   2. `u8_out[i] as u16 == av1_inv_txfm2d_add(bd=8)[i]`  — vs the port's
//!      already-C-verified highbd path (guards against the two ever drifting).
//!
//! Coverage: every supported (tx_type x tx_size), the all-zero-coeff pass, a
//! large randomized fuzz over the full dequant coefficient range, strided
//! destinations, and the lossless 4x4 Walsh–Hadamard (both eob arms). The
//! coefficient range spans the row-pass `clamp_buf(bd+8)` domain so the whole
//! butterfly (incl. the per-stage normative clamps) is exercised.

use aom_dsp::transform::inv_txfm2d::{
    av1_inv_txfm2d_add, av1_inv_txfm2d_add_u8, av1_iwht4x4_add_u8, inv_input_len, inv_txfm_valid,
};
use aom_sys_ref as c;

const W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

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
    fn coeff(&mut self) -> i32 {
        (self.next() % (1 << 17)) as i32 - (1 << 16)
    }
    fn pixel_u8(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

/// One (tx_size, tx_type) cell: random coeffs + random u8 prediction; the lowbd
/// u8 reconstruction must equal BOTH the C bd8 reconstruction and the port's
/// highbd bd8 reconstruction, pixel for pixel.
fn check(rng: &mut Rng, tx_size: usize, tx_type: usize) {
    let (w, h) = (W[tx_size], H[tx_size]);
    let input: Vec<i32> = (0..inv_input_len(tx_size)).map(|_| rng.coeff()).collect();
    let pred: Vec<u8> = (0..w * h).map(|_| rng.pixel_u8()).collect();

    // lowbd u8 path (the thing under test)
    let mut got_u8 = pred.clone();
    av1_inv_txfm2d_add_u8(&input, &mut got_u8, w, tx_type, tx_size);

    // oracle 1 — real exported C at bd8
    let mut want_c: Vec<u16> = pred.iter().map(|&p| p as u16).collect();
    c::ref_inv_txfm2d_add(tx_size, &input, &mut want_c, w, tx_type, 8);

    // oracle 2 — port highbd path at bd8
    let mut want_hi: Vec<u16> = pred.iter().map(|&p| p as u16).collect();
    av1_inv_txfm2d_add(&input, &mut want_hi, w, tx_type, tx_size, 8);

    for i in 0..w * h {
        assert_eq!(
            got_u8[i] as u16, want_c[i],
            "lowbd vs C: tx_size={tx_size} ({w}x{h}) tx_type={tx_type} px={i}"
        );
        assert_eq!(
            got_u8[i] as u16, want_hi[i],
            "lowbd vs highbd port: tx_size={tx_size} ({w}x{h}) tx_type={tx_type} px={i}"
        );
    }
}

#[test]
fn inv_txfm2d_lowbd_zero_coeff() {
    // All-zero coeffs: the destination must pass through unchanged (residual 0).
    let mut rng = Rng(7);
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            let (w, h) = (W[tx_size], H[tx_size]);
            let input = vec![0i32; inv_input_len(tx_size)];
            let pred: Vec<u8> = (0..w * h).map(|_| rng.pixel_u8()).collect();
            let mut got = pred.clone();
            av1_inv_txfm2d_add_u8(&input, &mut got, w, tx_type, tx_size);
            assert_eq!(got, pred, "zero-coeff must pass through: tx_size={tx_size}");
        }
    }
}

#[test]
fn inv_txfm2d_lowbd_differential_fuzz() {
    let mut rng = Rng(0x_10bd_5eed_2026);
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for _ in 0..700 {
                check(&mut rng, tx_size, tx_type);
            }
        }
    }
}

/// The lowbd lossless 4x4 Walsh–Hadamard inverse-add vs the real exported C
/// (`av1_highbd_iwht4x4_add`, bd=8) AND the port's highbd WHT. Both eob arms,
/// strided destinations, full dequant clamp range.
#[test]
fn iwht4x4_lowbd_add_matches_c() {
    use aom_dsp::transform::inv_txfm2d::av1_highbd_iwht4x4_add;
    let mut rng = Rng(0x1D7_4A17_B8);
    let (mut full_cases, mut dc_cases) = (0usize, 0usize);
    let bound = 1i64 << (7 + 8); // dequant_txb clamp bound at bd8
    let span = (2 * bound) as u64;
    for stride in [4usize, 7, 16, 33] {
        for _ in 0..4000 {
            let full = rng.next() & 1 == 0;
            let eob = if full { 2 + (rng.next() % 15) as usize } else { 1 };
            let mut input = [0i32; 16];
            if full {
                for v in input.iter_mut() {
                    *v = ((rng.next() % span) as i64 - bound) as i32;
                }
                full_cases += 1;
            } else {
                input[0] = ((rng.next() % span) as i64 - bound) as i32;
                dc_cases += 1;
            }
            let pred: Vec<u8> = (0..4 * stride).map(|_| rng.pixel_u8()).collect();

            let mut got = pred.clone();
            av1_iwht4x4_add_u8(&input, &mut got, stride, eob);

            let mut want_c: Vec<u16> = pred.iter().map(|&p| p as u16).collect();
            c::ref_highbd_iwht4x4_add(&input, &mut want_c, stride, eob, 8);

            let mut want_hi: Vec<u16> = pred.iter().map(|&p| p as u16).collect();
            av1_highbd_iwht4x4_add(&input, &mut want_hi, stride, eob, 8);

            for i in 0..4 * stride {
                assert_eq!(got[i] as u16, want_c[i], "wht lowbd vs C: stride={stride} eob={eob} i={i}");
                assert_eq!(got[i] as u16, want_hi[i], "wht lowbd vs highbd: stride={stride} i={i}");
            }
        }
    }
    assert!(full_cases > 100 && dc_cases > 100, "both WHT arms exercised");
}
