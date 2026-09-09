//! Differential harness for `dequant_txb` — the decoder's `qcoeff → dqcoeff`
//! step fused into `av1_read_coeffs_txb` (av1/decoder/decodetxb.c) — against C
//! libaom v3.14.1 (`ref_dequant_txb`, a verbatim transcription of that math).
//! Swept over all 19 transform sizes, bitdepths {8,10,12}, and quant-matrix
//! on/off, with `qcoeff` biased to trip the 20-bit level mask, the 24-bit
//! product mask, and the `±(1<<(7+bd))` clamp.

use aom_sys_ref as c;
use aom_dsp::txb::{dequant_txb, txb_high, txb_wide};

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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    /// A signed qcoeff biased to hit every dequant branch: mostly small, some
    /// large enough to trip the 20-bit level mask / 24-bit product mask / clamp.
    fn qcoeff(&mut self) -> i32 {
        let mag = match self.next() % 10 {
            0..=5 => self.range(0, 64),
            6..=7 => self.range(0, 4096),
            8 => self.range(0, 1 << 20),
            _ => self.range(0, 1 << 23), // trips product mask + clamp
        };
        if self.next() & 1 == 1 {
            -mag
        } else {
            mag
        }
    }
}

#[test]
fn dequant_txb_matches_c_decoder() {
    let mut rng = Rng(0xde0d_ea11_c0ff_ee42);
    // Coverage flags: prove the exotic branches actually fired.
    let mut saw_clamp = false;
    let mut saw_level_mask = false;
    let mut saw_qm = false;
    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &bd in &[8i32, 10, 12] {
            let max_value = (1i32 << (7 + bd)) - 1;
            let min_value = -(1i32 << (7 + bd));
            for qm_on in [false, true] {
                for _ in 0..400 {
                    let qcoeff: Vec<i32> = (0..area).map(|_| rng.qcoeff()).collect();
                    let dequant = [rng.range(1, 8000) as i16, rng.range(1, 8000) as i16];
                    let iqm: Option<Vec<u8>> = if qm_on {
                        Some((0..area).map(|_| rng.range(1, 256) as u8).collect())
                    } else {
                        None
                    };
                    let iqm_ref = iqm.as_deref();

                    let mut got = vec![0i32; area];
                    dequant_txb(&qcoeff, &mut got, tx_size, dequant, iqm_ref, bd);
                    let want = c::ref_dequant_txb(&qcoeff, tx_size, dequant, iqm_ref, bd);

                    assert_eq!(
                        got, want,
                        "dqcoeff mismatch ts={tx_size} bd={bd} qm={qm_on}\nqcoeff={qcoeff:?}\ndequant={dequant:?}"
                    );

                    saw_clamp |= want.iter().any(|&v| v == max_value || v == min_value);
                    saw_level_mask |= qcoeff.iter().any(|&q| q.unsigned_abs() >= (1 << 20));
                    saw_qm |= qm_on && want.iter().any(|&v| v != 0);
                }
            }
        }
    }
    assert!(saw_clamp, "test never exercised the bitdepth clamp");
    assert!(saw_level_mask, "test never exercised the 20-bit level mask");
    assert!(saw_qm, "test never exercised the quant-matrix path");
}
