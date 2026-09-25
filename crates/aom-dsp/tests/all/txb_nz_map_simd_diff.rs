//! SIMD-vs-C differential for `get_nz_map_contexts` at every archmage token
//! permutation: the v3 raster-tile port (`av1_get_nz_map_contexts_sse2`,
//! encodetxb_sse2.c) against the REAL `av1_get_nz_map_contexts_c`, including
//! the exact write footprint (positions past `eob` must stay untouched).
//!
//! The same default-tier C pin already runs in `txb_diff.rs`; this file adds
//! the dispatch sweep so `AOM_FORCE_SCALAR` / tier permutations cannot ship a
//! kernel that only ever compared scalar against itself (docs/
//! SIMD_REACH_AUDIT_2026-07-28.md finding F4).

use aom_dsp::txb::{
    get_nz_map_contexts, txb_high, txb_init_levels, txb_wide, TX_PAD_2D, TX_PAD_HOR,
    TX_TYPE_TO_CLASS,
};
use aom_sys_ref as c;
// `summon()` comes from this trait; needed at MODULE scope because the
// non-vacuity counter below lives outside the fn-local `use` blocks.
use archmage::testing::{for_each_token_permutation, CompileTimePolicy};
use archmage::SimdToken;

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

fn gen_coeff(rng: &mut Rng) -> i32 {
    let r = rng.next();
    // Mostly small levels (typical post-quantization), occasionally large.
    if r & 0x3f == 0 {
        ((r >> 8) as i32 & 0x3fff) - 0x2000
    } else {
        ((r >> 8) as i32 % 37) - 18
    }
}

#[test]
fn nz_map_contexts_simd_bit_identical_to_c_at_every_tier() {
    // Serialise: this sweep permutes PROCESS-GLOBAL dispatch
    // state; see `crate::dispatch_serial`.
    let _serial = crate::dispatch_serial::dispatch_serial();
    // See txb_init_levels_simd_diff.rs for why the non-vacuity guard lives
    // INSIDE the harness (the AOM_FORCE_SCALAR pin ordering trap).
    let mut simd_perms = 0usize;
    // All three TX_CLASSes; every tx_size on the 2D class, a representative
    // spread on HORIZ/VERT.
    const TX_TYPES_SAMPLE: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        if if cfg!(target_arch = "aarch64") {
            archmage::NeonToken::summon().is_some()
        } else {
            archmage::X64V3Token::summon().is_some()
        } {
            simd_perms += 1;
        }
        let mut rng = Rng(0x_0217_9abc_def0_1234);
        for tx_size in 0..19usize {
            let w = txb_wide(tx_size);
            let h = txb_high(tx_size);
            let area = w * h;
            for case in 0..8u32 {
                let coeff: Vec<i32> = (0..area).map(|_| gen_coeff(&mut rng)).collect();
                let mut levels = vec![0xAAu8; TX_PAD_2D];
                txb_init_levels(&coeff, w, h, &mut levels);
                if case == 0 {
                    levels[..w * (h + TX_PAD_HOR)].fill(0);
                }
                for &tx_type in &TX_TYPES_SAMPLE {
                    let tx_class = TX_TYPE_TO_CLASS[tx_type];
                    let scan = c::ref_scan_order(tx_size, tx_type, area);
                    for eob in [0usize, 1, area, 1 + (rng.next() as usize % area)] {
                        let mut got = vec![0x7Fi8; 32 * 32];
                        let mut want = vec![0x7Fi8; 32 * 32];
                        get_nz_map_contexts(&levels, &scan, eob, tx_size, tx_class, &mut got);
                        c::ref_get_nz_map_contexts(
                            &levels,
                            &scan,
                            eob,
                            tx_size,
                            tx_class as i32,
                            &mut want,
                        );
                        assert_eq!(
                            got, want,
                            "[{tier}] tx_size={tx_size} tx_type={tx_type} eob={eob} case={case}"
                        );
                    }
                }
            }
        }
    });
    eprintln!("nz_map_contexts SIMD parity: {report}");
    assert!(
        simd_perms >= 1,
        "the SIMD permutation ({}) must run at least once — a passing run with \
         zero vector permutations compares the scalar path against itself.",
        if cfg!(target_arch = "aarch64") {
            "neon"
        } else {
            "v3/AVX2"
        }
    );
    assert!(report.permutations_run >= 2);
}
