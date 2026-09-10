//! Differential harness for `predict_intra_high_in_place` — the form that
//! predicts INTO the reconstruction plane it reads its neighbours from, which is
//! what libaom's `av1_predict_intra_block_facade` does (it hands the predictor
//! `pd->dst`).
//!
//! The claim under test is aliasing-freedom, so the assertion is on the WHOLE
//! PLANE, not on the block: for every mode / angle-delta / filter-intra mode /
//! tx size / availability combination / bit depth, predicting in place must
//! leave the plane byte-identical to predicting into a tight scratch and copying
//! the block back — the exact sequence every encoder call site used to spell out
//! — and the block itself must still equal the REAL exported C predictor. A
//! stray write outside the block, or a read of a neighbour that the write half
//! had already clobbered, moves the plane and fails here.
//!
//! Two shapes are swept deliberately: the block at an interior offset with a
//! large row stride (the encoder's recon plane) and, through `combos`, the
//! degenerate availability cases that reach the directional early-out — the one
//! arm that reads `recon` outside `assemble_dir_edges`.

use aom_dsp::intra::{predict_intra_high, predict_intra_high_in_place};
use aom_sys_ref as c;

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
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
}

const STRIDE: usize = 256;
const ROWS: usize = 256;
const ROW0: usize = 4;
const COL0: usize = 4;

#[allow(clippy::too_many_arguments)]
fn check(
    recon: &[u16],
    tx_size: usize,
    mode: usize,
    delta: i32,
    use_fi: bool,
    fi_mode: usize,
    disable: bool,
    filt: i32,
    combo: (i32, i32, i32, i32),
    bd: i32,
) {
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let ref_off = ROW0 * STRIDE + COL0;
    let (nt, ntr, nl, nbl) = combo;

    // (a) the two-slice form: predict into a tight scratch, then copy the block
    // back row by row — the sequence the encoder call sites used to spell out.
    let mut pred = vec![0u16; txw * txh];
    predict_intra_high(
        recon, ref_off, STRIDE, &mut pred, txw, mode, delta, use_fi, fi_mode, disable, filt,
        tx_size, nt as usize, ntr, nl as usize, nbl, bd,
    );
    let mut want_plane = recon.to_vec();
    for r in 0..txh {
        want_plane[ref_off + r * STRIDE..ref_off + r * STRIDE + txw]
            .copy_from_slice(&pred[r * txw..r * txw + txw]);
    }

    // (b) the in-place form, on its own copy of the same plane.
    let mut got_plane = recon.to_vec();
    predict_intra_high_in_place(
        &mut got_plane,
        ref_off,
        STRIDE,
        mode,
        delta,
        use_fi,
        fi_mode,
        disable,
        filt,
        tx_size,
        nt as usize,
        ntr,
        nl as usize,
        nbl,
        bd,
    );

    // The whole plane, so a write outside the block is caught too.
    if got_plane != want_plane {
        let first = got_plane
            .iter()
            .zip(&want_plane)
            .position(|(a, b)| a != b)
            .expect("planes differ but no differing element");
        panic!(
            "in-place prediction moved the plane at index {first} (row {}, col {}) \
             ts={tx_size} ({txw}x{txh}) mode={mode} delta={delta} use_fi={use_fi} \
             fi_mode={fi_mode} disable={disable} filt={filt} combo={combo:?} bd={bd}: \
             got {} want {}",
            first / STRIDE,
            first % STRIDE,
            got_plane[first],
            want_plane[first]
        );
    }

    // And the block is still the REAL C predictor's output — so this is a
    // differential against libaom, not merely a self-consistency check.
    let want = c::ref_hbd_predict_intra(
        recon, ref_off, STRIDE, mode, delta, use_fi, fi_mode, disable, filt, tx_size, txw, txh, nt,
        ntr, nl, nbl, bd,
    );
    let block: Vec<u16> = (0..txh)
        .flat_map(|r| got_plane[ref_off + r * STRIDE..ref_off + r * STRIDE + txw].to_vec())
        .collect();
    assert_eq!(
        block, want,
        "in-place block differs from C ts={tx_size} ({txw}x{txh}) mode={mode} delta={delta} \
         use_fi={use_fi} fi_mode={fi_mode} disable={disable} filt={filt} combo={combo:?} bd={bd}"
    );
}

#[test]
fn predict_intra_in_place_matches_predict_then_copy() {
    let mut rng = Rng(0x51a7_c00d_9ec7_4000);
    for &bd in &[8i32, 10, 12] {
        let recon: Vec<u16> = (0..STRIDE * ROWS)
            .map(|_| (rng.next() % (1u64 << bd)) as u16)
            .collect();
        for tx_size in 0..19usize {
            let (tw, th) = (TX_W[tx_size] as i32, TX_H[tx_size] as i32);
            // The last two reach the directional degenerate early-out (one side
            // needed, none available) — the only arm that reads `recon` outside
            // `assemble_dir_edges`.
            let combos: [(i32, i32, i32, i32); 6] = [
                (tw, th, th, tw), // full + above-right + below-left
                (tw, -1, th, -1), // full top+left, no extension
                (tw, 0, th, 0),   // extension considered, 0 px
                (0, -1, th, -1),  // left only
                (tw, -1, 0, -1),  // above only
                (0, -1, 0, -1),   // neither
            ];
            for mode in 0..13usize {
                let deltas: &[i32] = if (1..=8).contains(&mode) {
                    &[-9, -6, -3, 0, 3, 6, 9]
                } else {
                    &[0]
                };
                for &delta in deltas {
                    for &combo in &combos {
                        check(&recon, tx_size, mode, delta, false, 0, false, 1, combo, bd);
                    }
                }
            }
            if TX_W[tx_size] <= 32 && TX_H[tx_size] <= 32 {
                for fi_mode in 0..5usize {
                    for &combo in &combos[..4] {
                        check(&recon, tx_size, 0, 0, true, fi_mode, false, 1, combo, bd);
                    }
                }
            }
            for &mode in &[0usize, 1, 3, 9, 12] {
                let delta = if (1..=8).contains(&mode) { 3 } else { 0 };
                for disable in [false, true] {
                    for filt in [0i32, 1] {
                        check(
                            &recon, tx_size, mode, delta, false, 0, disable, filt, combos[0], bd,
                        );
                    }
                }
            }
        }
    }
}
