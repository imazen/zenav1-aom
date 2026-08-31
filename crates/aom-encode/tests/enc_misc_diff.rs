//! Differential harness for the small exported encoder helpers in
//! `aom_encode::enc_misc` vs the REAL exported C libaom v3.14.1. **Tier 1.**
//!
//! | test | C oracle |
//! |---|---|
//! | `get_intra_cost_penalty_matches_c` | `av1_get_intra_cost_penalty` |
//! | `hash_perfect_predicates_match_c` | `av1_hash_is_{horizontal,vertical}_perfect` |
//! | `dropout_qcoeff_num_matches_c` | `av1_dropout_qcoeff_num` |
//!
//! `hash_perfect_predicates_are_not_the_same_function` exists because the two
//! hash predicates are trivially confusable: C writes the horizontal one with a
//! per-row re-based pointer and the vertical one indexed off a single origin
//! with the outer variable as the COLUMN. A block that is row-flat but not
//! column-flat separates them, and without such a block a port that ran the
//! same loop twice would pass.

use aom_dsp::txb::scan;
use aom_encode::enc_misc::{
    dropout_qcoeff_num, get_intra_cost_penalty, hash_is_horizontal_perfect,
    hash_is_vertical_perfect,
};
use aom_sys_ref::{
    ref_dropout_qcoeff_num, ref_get_intra_cost_penalty, ref_hash_is_horizontal_perfect,
    ref_hash_is_vertical_perfect,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

#[test]
fn get_intra_cost_penalty_matches_c() {
    // The whole qindex range, both DC deltas the spec allows, all three depths.
    for &bd in &[8u8, 10, 12] {
        for qindex in 0..=255i32 {
            for &qdelta in &[-63i32, -16, 0, 1, 16, 63] {
                let got = get_intra_cost_penalty(qindex, qdelta, bd);
                let want = ref_get_intra_cost_penalty(qindex, qdelta, bd);
                assert_eq!(got, want, "bd={bd} qindex={qindex} qdelta={qdelta}");
            }
        }
    }
}

/// The `(plane, stride)` pair for a `block_size` square at `(x, y)` inside a
/// larger plane.
fn plane_with(rng: &mut Rng, w: usize, h: usize) -> (Vec<u16>, usize) {
    let stride = w + 7;
    let p: Vec<u16> = (0..stride * h).map(|_| rng.below(256) as u16).collect();
    (p, stride)
}

#[test]
fn hash_perfect_predicates_match_c() {
    let mut rng = Rng::new(0x33AA_55CC_77EE_9911);
    let (pw, ph) = (64usize, 64usize);
    for &block_size in &[4usize, 8, 16, 32] {
        for kind in 0..4 {
            let (mut plane, stride) = plane_with(&mut rng, pw, ph);
            // Overwrite one block with content of a known shape.
            let (x0, y0) = (8usize, 8usize);
            for i in 0..block_size {
                for j in 0..block_size {
                    let v = match kind {
                        0 => rng.below(256) as u16, // random
                        1 => 77,                    // fully flat
                        2 => (i * 3 + 1) as u16,    // row-flat
                        _ => (j * 5 + 2) as u16,    // column-flat
                    };
                    plane[(y0 + i) * stride + x0 + j] = v;
                }
            }
            let gh = hash_is_horizontal_perfect(&plane, stride, block_size, x0, y0);
            let wh = ref_hash_is_horizontal_perfect(&plane, stride, block_size, x0, y0);
            let gv = hash_is_vertical_perfect(&plane, stride, block_size, x0, y0);
            let wv = ref_hash_is_vertical_perfect(&plane, stride, block_size, x0, y0);
            assert_eq!(gh, wh, "horizontal bs={block_size} kind={kind}");
            assert_eq!(gv, wv, "vertical bs={block_size} kind={kind}");
        }
    }
}

#[test]
fn hash_perfect_predicates_are_not_the_same_function() {
    let mut rng = Rng::new(0x1029_3847_5647_3829);
    let (pw, ph) = (32usize, 32usize);
    let block_size = 8usize;
    let (mut plane, stride) = plane_with(&mut rng, pw, ph);
    let (x0, y0) = (4usize, 4usize);
    // Row-flat but NOT column-flat: value depends only on the row.
    for i in 0..block_size {
        for j in 0..block_size {
            plane[(y0 + i) * stride + x0 + j] = (i * 9 + 1) as u16;
        }
    }
    assert!(hash_is_horizontal_perfect(
        &plane, stride, block_size, x0, y0
    ));
    assert!(
        !hash_is_vertical_perfect(&plane, stride, block_size, x0, y0),
        "a row-flat block must not read as column-flat"
    );
    // And the mirror image.
    for i in 0..block_size {
        for j in 0..block_size {
            plane[(y0 + i) * stride + x0 + j] = (j * 9 + 1) as u16;
        }
    }
    assert!(!hash_is_horizontal_perfect(
        &plane, stride, block_size, x0, y0
    ));
    assert!(hash_is_vertical_perfect(&plane, stride, block_size, x0, y0));
}

/// The `TX_SIZE` values swept. `max_eob` is taken from the scan-order length
/// rather than hard-coded, and `max_eob_matches_c_get_max_eob` checks a few
/// against `av1_get_max_eob`'s own values so a wrong table cannot go unnoticed.
const TX_SIZES: [usize; 6] = [
    0, // TX_4X4
    1, // TX_8X8
    2, // TX_16X16
    3, // TX_32X32
    4, // TX_64X64 (av1_get_max_eob caps this at 1024)
    7, // TX_8X16
];

#[test]
fn max_eob_matches_c_get_max_eob() {
    // av1_get_max_eob(TX_SIZE) for the sizes swept below.
    for &(tx_size, expect) in &[
        (0usize, 16usize),
        (1, 64),
        (2, 256),
        (3, 1024),
        (4, 1024),
        (7, 128),
    ] {
        assert_eq!(
            scan(tx_size, 0).len(),
            expect,
            "scan length for tx_size {tx_size} disagrees with av1_get_max_eob"
        );
    }
}

#[test]
fn dropout_qcoeff_num_matches_c() {
    let mut rng = Rng::new(0x6F5E_4D3C_2B1A_0918);
    let mut saw_shrink = false;
    let mut saw_unchanged = false;
    for &tx_size in &TX_SIZES {
        for &tx_type in &[0usize, 1, 9] {
            let sc = scan(tx_size, tx_type);
            let max_eob = sc.len();
            // `dropout_num_after == 0` is excluded deliberately: C indexes
            // scan[-1] there (see the contract note on `dropout_qcoeff_num`),
            // so the two sides are not comparable and the port asserts instead.
            // The (64, 64) and (256, 256) pairs are what the real caller
            // `av1_dropout_qcoeff` produces (multiplier in [2,8] times
            // CLIP(max(tx_w, tx_h), 16, 32)); the smaller ones are inside this
            // function's own defined behaviour and reach the state machine on
            // blocks the realistic values would early-out on.
            for &(before, after) in &[
                (1i32, 1i32),
                (2, 2),
                (4, 4),
                (8, 8),
                (16, 32),
                (64, 64),
                (256, 256),
            ] {
                for trial in 0..8 {
                    // Coefficients biased toward zeros and small magnitudes, so
                    // the dropout state machine actually has runs to work on.
                    let mut q = vec![0i32; max_eob];
                    let mut dq = vec![0i32; max_eob];
                    let eob = 1 + rng.below(max_eob as u32) as usize;
                    for i in 0..eob {
                        let s = sc[i] as usize;
                        let r = rng.below(10);
                        q[s] = match r {
                            0..=5 => 0,
                            6..=7 => 1 - 2 * (rng.below(2) as i32),
                            8 => 2 - 4 * (rng.below(2) as i32),
                            _ => (rng.below(64) as i32) - 32,
                        };
                        dq[s] = q[s] * 8;
                    }
                    let (mut q_p, mut dq_p) = (q.clone(), dq.clone());
                    let (mut q_c, mut dq_c) = (q.clone(), dq.clone());
                    let got =
                        dropout_qcoeff_num(&mut q_p, &mut dq_p, eob, max_eob, sc, before, after);
                    let want = ref_dropout_qcoeff_num(
                        &mut q_c, &mut dq_c, eob, tx_size, tx_type, before, after,
                    );
                    let label = format!(
                        "tx_size={tx_size} tx_type={tx_type} eob={eob} \
                         before={before} after={after} trial={trial}"
                    );
                    assert_eq!(got, want, "eob: {label}");
                    assert_eq!(q_p, q_c, "qcoeff: {label}");
                    assert_eq!(dq_p, dq_c, "dqcoeff: {label}");
                    if want < eob {
                        saw_shrink = true;
                    } else {
                        saw_unchanged = true;
                    }
                }
            }
        }
    }
    assert!(
        saw_shrink,
        "the dropout never shrank an eob — the whole state machine is untested"
    );
    assert!(
        saw_unchanged,
        "the dropout always fired; the early-out is untested"
    );
}
