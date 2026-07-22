//! Callgrind microbench for the bd8 inverse-transform lowbd lever: run a fixed
//! representative transform workload N times through either the LOWBD (u8) path
//! or the HIGHBD (u16, bd=8) path, on IDENTICAL randomized inputs, so a
//! callgrind Ir profile compares the two entry points directly.
//!
//! Usage: lowbd_txfm_profile <u8|u16> <iters>
//!
//! The first iteration cross-checks u8 vs u16 byte-identity (a corrupt build
//! must never be profiled). Compare inclusive Ir of
//! `av1_inv_txfm2d_add_u8_into` (u8 side) vs `av1_inv_txfm2d_add_into`
//! (u16 side) across the two runs.

use aom_dsp::transform::inv_txfm2d::{
    av1_inv_txfm2d_add_into, av1_inv_txfm2d_add_u8_into, inv_input_len, inv_txfm_valid, InvTxfmScratch,
};

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
}

struct Cell {
    tx_size: usize,
    tx_type: usize,
    input: Vec<i32>,
    pred: Vec<u8>,
    w: usize,
    h: usize,
}

/// A fixed, deterministic workload weighted toward the sizes/types a real bd8
/// decode spends its transform time in (small DCT-heavy blocks dominate; a few
/// larger and non-DCT cells for coverage). Each valid (tx_type, tx_size) is
/// included once with random coeffs + a random u8 prediction.
fn workload() -> Vec<Cell> {
    let mut rng = Rng(0x_bd87_f114_2026);
    let mut cells = Vec::new();
    // Repeat the whole grid a few times with fresh randomness so no single cell
    // dominates and the small blocks (the real hot set) get proportional weight.
    for _ in 0..4 {
        for tx_size in 0..19 {
            for tx_type in 0..16 {
                if !inv_txfm_valid(tx_type, tx_size) {
                    continue;
                }
                let (w, h) = (W[tx_size], H[tx_size]);
                let input: Vec<i32> = (0..inv_input_len(tx_size))
                    .map(|_| (rng.next() % (1 << 17)) as i32 - (1 << 16))
                    .collect();
                let pred: Vec<u8> = (0..w * h).map(|_| (rng.next() & 0xff) as u8).collect();
                cells.push(Cell { tx_size, tx_type, input, pred, w, h });
            }
        }
    }
    cells
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: lowbd_txfm_profile <u8|u16> <iters>");
        std::process::exit(2);
    }
    let side = args[1].as_str();
    let iters: usize = args[2].parse().expect("iters must be a number");
    let cells = workload();

    // Byte-identity cross-check on cell 0 of every cell before profiling.
    let mut txfm = InvTxfmScratch::default();
    for c in &cells {
        let mut got_u8 = c.pred.clone();
        av1_inv_txfm2d_add_u8_into(&c.input, &mut got_u8, c.w, c.tx_type, c.tx_size, &mut txfm);
        let mut got_hi: Vec<u16> = c.pred.iter().map(|&p| p as u16).collect();
        av1_inv_txfm2d_add_into(&c.input, &mut got_hi, c.w, c.tx_type, c.tx_size, 8, &mut txfm);
        for i in 0..c.w * c.h {
            assert_eq!(
                got_u8[i] as u16, got_hi[i],
                "u8 vs u16 divergence tx_size={} tx_type={} px={i}",
                c.tx_size, c.tx_type
            );
        }
    }

    let mut sink = 0u64;
    match side {
        "u8" => {
            // one reusable dst per cell size, mirroring the decoder's plane reuse
            let mut dsts: Vec<Vec<u8>> = cells.iter().map(|c| c.pred.clone()).collect();
            for _ in 0..iters {
                for (c, dst) in cells.iter().zip(dsts.iter_mut()) {
                    dst.copy_from_slice(&c.pred);
                    av1_inv_txfm2d_add_u8_into(&c.input, dst, c.w, c.tx_type, c.tx_size, &mut txfm);
                    sink = sink.wrapping_add(dst[0] as u64);
                }
            }
        }
        "u16" => {
            let mut dsts: Vec<Vec<u16>> = cells
                .iter()
                .map(|c| c.pred.iter().map(|&p| p as u16).collect())
                .collect();
            let preds16: Vec<Vec<u16>> = cells
                .iter()
                .map(|c| c.pred.iter().map(|&p| p as u16).collect())
                .collect();
            for _ in 0..iters {
                for ((c, dst), pred) in cells.iter().zip(dsts.iter_mut()).zip(preds16.iter()) {
                    dst.copy_from_slice(pred);
                    av1_inv_txfm2d_add_into(&c.input, dst, c.w, c.tx_type, c.tx_size, 8, &mut txfm);
                    sink = sink.wrapping_add(dst[0] as u64);
                }
            }
        }
        other => panic!("side must be u8|u16, got {other}"),
    }
    eprintln!("{side} x{iters}: sink={sink}");
}
