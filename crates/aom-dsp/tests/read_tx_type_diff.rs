//! Roundtrip harness for `read_tx_type`: the decoder must invert `write_tx_type`
//! exactly. `write_tx_type`'s derivation is exhaustively C-validated
//! (`ext_tx_diff`), so a clean roundtrip — recovered tx_type (or the inferred
//! DCT_DCT when the block doesn't signal) plus lockstep CDF adaptation — pins
//! `read_tx_type` to libaom's `av1_read_tx_type`.

use aom_dsp::entropy::dec::OdEcDec;
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::txb::{ext_tx_derive, read_tx_type, write_tx_type};

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

/// A valid `n`-symbol inverse-CDF into `out` (`out[0..n-1]` strictly decreasing,
/// `out[n-1]=0`, `out[n]=count=0`).
fn mk_cdf(rng: &mut Rng, n: usize, out: &mut [u16; 17]) {
    *out = [0u16; 17];
    let mut acc: u32 = 0;
    for e in out[..n - 1].iter_mut() {
        acc += 1 + (rng.next() % (32000 / n as u64).max(2)) as u32;
        *e = 32768u32.saturating_sub(acc).max(1) as u16;
    }
    out[n - 1] = 0;
    out[n] = 0;
}

#[test]
fn read_tx_type_roundtrips_write() {
    let mut rng = Rng(0x1e_7787_c0de_0090);
    for tx_size in 0..19usize {
        for &is_inter in &[false, true] {
            for &reduced in &[false, true] {
                for tx_type in 0..16usize {
                    let d = ext_tx_derive(tx_size, is_inter, reduced, tx_type, false, 0, 0);
                    if d.used == 0 {
                        continue; // tx_type not in this block's ext-tx set
                    }
                    for &signal_gate in &[true, false] {
                        for _ in 0..40 {
                            let mut cdf0 = [0u16; 17];
                            mk_cdf(&mut rng, d.num.max(2) as usize, &mut cdf0);
                            let mut ce = cdf0;
                            let mut cdd = cdf0;

                            let mut enc = OdEcEnc::new();
                            write_tx_type(
                                &mut enc, &mut ce, tx_size, is_inter, reduced, tx_type,
                                false, 0, 0, signal_gate,
                            );
                            let bytes = enc.done().to_vec();

                            let mut dec = OdEcDec::new(&bytes);
                            let got = read_tx_type(
                                &mut dec, &mut cdd, tx_size, is_inter, reduced, signal_gate,
                            );
                            let expected = if d.num > 1 && signal_gate { tx_type } else { 0 };
                            assert_eq!(
                                got, expected,
                                "tx_type ts={tx_size} inter={is_inter} red={reduced} \
                                 t={tx_type} gate={signal_gate} num={}",
                                d.num
                            );
                            assert_eq!(
                                ce, cdd,
                                "cdf adaptation ts={tx_size} t={tx_type} gate={signal_gate}"
                            );
                        }
                    }
                }
            }
        }
    }
}
