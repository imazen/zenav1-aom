//! Is the inverse transform's `assert!(cfg.valid, ..)` reachable from a
//! bitstream?
//!
//! `av1_inv_txfm2d_add_{into,u8_into}` assert on `(tx_type, tx_size)` pairs the
//! inverse transform has no kernel for — `TXFM_TYPE_LS` has `-1` holes because
//! only DCT is defined at 64 points. The decoder feeds those two arguments
//! straight from the bitstream (`read_tx_type` on a CDF, `tx_size` from the
//! tx-size read), so "can a crafted stream reach the assert?" is a real
//! question about the untrusted-input surface, not a style question. A panic
//! there is a denial of service on the AVIF decode path.
//!
//! The answer is supposed to be no, by construction: `av1_get_ext_tx_set_type`
//! picks the tx-set from `tx_size` alone (plus `is_inter` / `reduced_tx_set`,
//! both frame-level), and the coded symbol can only name a type the set marks
//! `used`. At `TXSIZE_SQR_UP > TX_32X32` — every 64-point shape — the set is
//! `EXT_TX_SET_DCTONLY`, so `DCT_DCT` is the only decodable type there.
//!
//! "Supposed to be" is an argument across two tables in two crates. This test
//! is the argument made executable: enumerate EVERY `(tx_size, is_inter,
//! reduced)` combination the decoder can be in, enumerate every `tx_type` that
//! combination's set marks decodable, and require the inverse transform to
//! have a kernel for it. If a table edit ever opens a hole, this fails here
//! rather than as a decoder panic on a crafted file.

use aom_dsp::transform::inv_txfm2d::inv_txfm_valid;
use aom_dsp::txb::{ext_tx_derive, TX_TYPES};

/// `TX_SIZES_ALL`.
const TX_SIZES_ALL: usize = 19;

#[test]
fn every_bitstream_decodable_tx_type_and_size_has_an_inverse_kernel() {
    let mut decodable = 0usize;
    let mut holes: Vec<(usize, usize, bool, bool)> = Vec::new();

    for tx_size in 0..TX_SIZES_ALL {
        for is_inter in [false, true] {
            for reduced in [false, true] {
                for tx_type in 0..TX_TYPES {
                    // `used` is `av1_ext_tx_used[set_type][tx_type]`: whether
                    // this type is in the set the decoder selected, i.e.
                    // whether any coded symbol can name it. The mode /
                    // filter-intra arguments only pick which CDF row is read,
                    // never which types the set contains, so they are pinned.
                    let d = ext_tx_derive(tx_size, is_inter, reduced, tx_type, false, 0, 0);
                    if d.used == 0 {
                        continue;
                    }
                    decodable += 1;
                    if !inv_txfm_valid(tx_type, tx_size) {
                        holes.push((tx_size, tx_type, is_inter, reduced));
                    }
                }
            }
        }
    }

    assert!(
        holes.is_empty(),
        "a crafted bitstream can select {} (tx_size, tx_type) pair(s) the inverse transform \
         has no kernel for — each is a reachable panic in av1_inv_txfm2d_add: {holes:?}",
        holes.len()
    );

    // Anti-vacuity, both directions. The sweep must actually enumerate
    // something, and `inv_txfm_valid` must actually be capable of returning
    // false — otherwise this passes for the wrong reason.
    assert!(
        decodable > 100,
        "only {decodable} decodable pairs enumerated — the set derivation returned `used == 0` \
         almost everywhere, so this test proves nothing"
    );
    let invalid_pairs = (0..TX_SIZES_ALL)
        .flat_map(|s| (0..TX_TYPES).map(move |t| (s, t)))
        .filter(|&(s, t)| !inv_txfm_valid(t, s))
        .count();
    assert!(
        invalid_pairs > 0,
        "inv_txfm_valid never returns false, so requiring it above is not a constraint"
    );
    eprintln!(
        "{decodable} decodable (tx_size, tx_type, is_inter, reduced) selections, all with \
         kernels; {invalid_pairs} of the {} (tx_size, tx_type) pairs have NO kernel and none \
         of them is selectable from a bitstream",
        TX_SIZES_ALL * TX_TYPES
    );
}
