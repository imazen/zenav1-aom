//! Differential harness for the per-block entropy-context propagation vs C
//! libaom: `get_txb_ctx` (neighbours -> txb_skip_ctx/dc_sign_ctx) and
//! `av1_get_txb_entropy_context` (block -> packed neighbour context).

use aom_sys_ref as c;
use aom_txb::{get_txb_ctx, scan, txb_high, txb_wide, txb_entropy_context};

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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
}

// Plane block sizes that can co-occur with each tx (superset is fine — both
// sides get the same value). Uses a few representative BlockSize discriminants.
const PLANE_BSIZES: [usize; 8] = [0, 3, 6, 9, 12, 4, 7, 10];

#[test]
fn get_txb_ctx_matches_c() {
    let mut rng = Rng(0x_e17c_0000_c057_9911);
    for tx_size in 0..19usize {
        // enough context entries for the largest tx unit (16).
        for _ in 0..4000 {
            // ENTROPY_CONTEXT bytes: cul_level (0..7) | dc_sign (0..2)<<3.
            let mk = |rng: &mut Rng| -> Vec<i8> {
                (0..16)
                    .map(|_| {
                        let cul = rng.range(0, 8) as i32;
                        let sign = rng.range(0, 3) as i32;
                        (cul | (sign << 3)) as i8
                    })
                    .collect()
            };
            let a = mk(&mut rng);
            let l = mk(&mut rng);
            for &plane in &[0usize, 1, 2] {
                for &pb in &PLANE_BSIZES {
                    let got = get_txb_ctx(pb, tx_size, plane, &a, &l);
                    let want = c::ref_get_txb_ctx(pb, tx_size, plane, &a, &l);
                    assert_eq!(got, want, "get_txb_ctx pb={pb} ts={tx_size} plane={plane}");
                }
            }
        }
    }
}

#[test]
fn txb_entropy_context_matches_c() {
    let mut rng = Rng(0x_e17c_beef_0000_2222);
    const TX_TYPES: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];
    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &tx_type in &TX_TYPES {
            for _ in 0..300 {
                let sc = scan(tx_size, tx_type);
                let eob = rng.range(0, area as u32 + 1) as usize;
                let mut qcoeff = vec![0i32; area];
                for i in 0..eob {
                    if rng.range(0, 10) >= 4 {
                        let mag = rng.range(1, 500) as i32;
                        let sign = if rng.next() & 1 == 1 { -1 } else { 1 };
                        qcoeff[sc[i] as usize] = sign * mag;
                    }
                }
                let got = txb_entropy_context(&qcoeff, tx_size, tx_type, eob);
                let want = c::ref_txb_entropy_context(&qcoeff, tx_size, tx_type, eob);
                assert_eq!(got, want, "txb_entropy_context ts={tx_size} tt={tx_type} eob={eob}");
            }
        }
    }
}
