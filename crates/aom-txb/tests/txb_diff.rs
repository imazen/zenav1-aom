//! Differential harness for the txb coefficient-coding kernels vs C libaom
//! v3.14.1: `av1_txb_init_levels_c`, `av1_get_nz_map_contexts_c`,
//! `av1_get_eob_pos_token`, and the `av1_nz_map_ctx_offset` context tables.
//!
//! The scan orders are taken from libaom's own `av1_scan_orders` (through the
//! oracle shim) and fed identically to both sides, isolating the kernels under
//! test. `tx_class` is likewise a parameter on both sides; the Rust
//! `TX_TYPE_TO_CLASS` transcription selects it (the mapping itself is verbatim
//! from txb_common.h and will be end-to-end validated by the future
//! `av1_write_coeffs_txb` full diff).

use aom_sys_ref as c;
use aom_txb::{
    get_eob_pos_token, get_nz_map_contexts, nz_map_ctx_offset, txb_high, txb_init_levels,
    txb_wide, TX_PAD_2D, TX_TYPE_TO_CLASS,
};

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

/// Sparse, level-map-realistic coefficient: mostly zero, mostly small when
/// nonzero, with occasional huge values to exercise the |coeff| clamp to 127.
fn gen_coeff(rng: &mut Rng) -> i32 {
    let r = rng.next();
    let mag = match r % 10 {
        0..=5 => 0,
        6..=7 => (r >> 8) as i32 % 4,          // 0..3 — the ctx-sensitive range
        8 => (r >> 8) as i32 % 200,            // mid magnitudes (clamp boundary)
        _ => 128 + ((r >> 8) as i32 % 100_000), // clamps to 127 in the level map
    };
    if (r >> 40) & 1 == 1 { -mag } else { mag }
}

#[test]
fn nz_ctx_offset_tables_match_c() {
    for t in 0..19usize {
        let n = txb_wide(t) * txb_high(t);
        let ours = &nz_map_ctx_offset(t)[..n];
        let theirs = c::ref_nz_ctx_offset(t, n);
        assert_eq!(ours, &theirs[..], "nz_map_ctx_offset tx_size={t}");
    }
}

#[test]
fn eob_pos_token_matches_c() {
    for eob in 1..=1024i32 {
        assert_eq!(get_eob_pos_token(eob), c::ref_eob_pos_token(eob), "eob={eob}");
    }
}

#[test]
fn txb_kernels_byte_identical() {
    let mut rng = Rng(0x_7b_1234_5678_9abc_u64);
    // TX_TYPE sample covering all three TX_CLASSes and several 2D scans:
    // DCT_DCT(0), ADST_ADST(3), IDTX(9) — 2D; V_DCT(10), V_FLIPADST(14) — VERT;
    // H_DCT(11), H_FLIPADST(15) — HORIZ.
    const TX_TYPES_SAMPLE: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];

    for tx_size in 0..19usize {
        let w = txb_wide(tx_size);
        let h = txb_high(tx_size);
        let area = w * h;

        for iter in 0..300 {
            // --- av1_txb_init_levels ---
            let coeff: Vec<i32> = (0..area).map(|_| gen_coeff(&mut rng)).collect();
            let mut lv_ours = vec![0xAAu8; TX_PAD_2D];
            let mut lv_c = vec![0xAAu8; TX_PAD_2D];
            txb_init_levels(&coeff, w, h, &mut lv_ours);
            c::ref_txb_init_levels(&coeff, w, h, &mut lv_c);
            assert_eq!(lv_ours, lv_c, "init_levels tx_size={tx_size} iter={iter}");

            // --- av1_get_nz_map_contexts, all classes/scans on these levels ---
            for &tx_type in &TX_TYPES_SAMPLE {
                let tx_class = TX_TYPE_TO_CLASS[tx_type];
                let scan = c::ref_scan_order(tx_size, tx_type, area);
                for eob in [0usize, 1, area, 1 + (rng.next() as usize % area)] {
                    let mut cc_ours = vec![0x7Fi8; 32 * 32];
                    let mut cc_c = vec![0x7Fi8; 32 * 32];
                    get_nz_map_contexts(&lv_ours, &scan, eob, tx_size, tx_class, &mut cc_ours);
                    c::ref_get_nz_map_contexts(
                        &lv_c,
                        &scan,
                        eob,
                        tx_size,
                        tx_class as i32,
                        &mut cc_c,
                    );
                    assert_eq!(
                        cc_ours, cc_c,
                        "nz_map_contexts tx_size={tx_size} tx_type={tx_type} eob={eob}"
                    );
                }
            }
        }
    }
}
