//! Differential harness for `av1_write_coeffs_txb`: the Rust writer must produce
//! byte-identical bitstream output to the transcribed C harness (which calls the
//! pristine `od_ec`, `update_cdf`, `av1_txb_init_levels_c`,
//! `av1_get_nz_map_contexts_c`, and libaom's scan orders).
//!
//! Both sides start from an identical random CDF arena (writer bit-exactness is
//! CDF-agnostic, so random-but-valid CDFs exercise every symbol path), and we
//! test both `allow_update_cdf` polarities — with adaptation on, the arena is
//! mutated in lockstep, so a single diverging update would surface in the bytes.
//! `av1_write_tx_type` (plane-0 tx_type signaling) is out of scope on both sides.

use aom_dsp::entropy::enc::OdEcEnc;
use aom_sys_ref as c;
use aom_dsp::txb::{scan, txb_high, txb_wide, write_coeffs_txb, CDF_ARENA_LEN};

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

/// The arena's region layout: (offset, slot_count, nsymbs). Each slot occupies
/// `nsymbs + 1` u16 — an `nsymbs`-entry inverse-CDF (`icdf[nsymbs-1] == 0`) plus
/// a 1-u16 adaptation counter. Mirrors `write.rs` / the shim exactly.
const REGIONS: [(usize, usize, usize); 13] = [
    (0, 5 * 13, 2),   // TXB_SKIP
    (195, 4, 5),      // EOB16
    (219, 4, 6),      // EOB32
    (247, 4, 7),      // EOB64
    (279, 4, 8),      // EOB128
    (315, 4, 9),      // EOB256
    (355, 4, 10),     // EOB512
    (399, 4, 11),     // EOB1024
    (447, 5 * 2 * 9, 2),  // EOB_EXTRA
    (717, 5 * 2 * 4, 3),  // BASE_EOB
    (877, 5 * 2 * 42, 4), // BASE
    (2977, 5 * 2 * 21, 4), // BR
    (4027, 2 * 3, 2), // DC_SIGN
];

/// A valid random CDF arena, filled slot-by-slot per the real layout: each slot
/// is a strictly-decreasing inverse-CDF ending in 0 with its adaptation counter
/// zeroed. Random-but-valid CDFs exercise every symbol probability while keeping
/// the od_ec invariants (and so `update_cdf`'s `rate` bounded).
fn gen_arena(rng: &mut Rng) -> Vec<u16> {
    let mut a = vec![0u16; CDF_ARENA_LEN];
    for &(off, count, n) in &REGIONS {
        for slot in 0..count {
            let base = off + slot * (n + 1);
            let mut acc: u32 = 0;
            for e in a[base..base + n - 1].iter_mut() {
                acc += rng.range(1, (32000 / n as u32).max(2));
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            a[base + n - 1] = 0; // icdf sentinel
            a[base + n] = 0; // adaptation counter
        }
    }
    a
}

/// Sparse quantized coefficients in transposed layout, consistent with libaom's
/// invariant: `eob` is the last-nonzero scan index + 1, so `scan[eob-1]` is
/// nonzero and every scan position `>= eob` is zero. Magnitudes are biased
/// toward the values that exercise the base / base-range / golomb paths.
fn gen_coeffs(rng: &mut Rng, scan: &[i16], area: usize) -> (Vec<i32>, usize) {
    let mut coeff = vec![0i32; area];
    let eob = rng.range(1, area as u32 + 1) as usize;
    let nz = |rng: &mut Rng| -> i32 {
        // strictly nonzero magnitude
        let mag = match rng.range(0, 10) {
            0..=4 => rng.range(1, 3) as i32, // base (1..2)
            5..=7 => rng.range(1, 20) as i32, // base-range
            _ => rng.range(1, 3000) as i32,  // golomb
        };
        if rng.next() & 1 == 1 { -mag } else { mag }
    };
    #[allow(clippy::needless_range_loop)]
    for i in 0..eob {
        let pos = scan[i] as usize;
        if i == eob - 1 {
            coeff[pos] = nz(rng); // eob position: always nonzero
        } else if rng.range(0, 10) >= 4 {
            coeff[pos] = nz(rng); // ~60% of interior scan positions nonzero
        }
    }
    (coeff, eob)
}

#[test]
fn write_coeffs_txb_byte_identical() {
    let mut rng = Rng(0x_00c0_ffee_1234_5678);
    // tx_types across all three classes and both 2D scan families.
    const TX_TYPES: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];

    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &tx_type in &TX_TYPES {
            for &plane_type in &[0usize, 1] {
                for &upd in &[true, false] {
                    for _ in 0..40 {
                        let sc = scan(tx_size, tx_type);
                        let (coeff, eob) = gen_coeffs(&mut rng, sc, area);
                        let txb_skip_ctx = rng.range(0, 13) as usize;
                        let dc_sign_ctx = rng.range(0, 3) as usize;

                        let arena0 = gen_arena(&mut rng);
                        let mut arena_c = arena0.clone();
                        let mut arena_r = arena0.clone();

                        let want = c::ref_write_coeffs_txb(
                            &coeff, eob, tx_size, tx_type, plane_type, txb_skip_ctx,
                            dc_sign_ctx, upd, &mut arena_c,
                        );

                        let mut enc = OdEcEnc::new();
                        write_coeffs_txb(
                            &mut enc, &mut arena_r, &coeff, eob, tx_size, tx_type,
                            plane_type, txb_skip_ctx, dc_sign_ctx, upd,
                        );
                        let got = enc.done().to_vec();

                        assert_eq!(
                            got, want,
                            "bytes tx_size={tx_size} tx_type={tx_type} plane={plane_type} \
                             upd={upd} eob={eob}"
                        );
                        if upd {
                            assert_eq!(
                                arena_r, arena_c,
                                "cdf adaptation diverged tx_size={tx_size} tx_type={tx_type}"
                            );
                        }
                    }
                }
            }
        }
    }
}
