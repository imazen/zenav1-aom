//! Differential harness for the RD mode-threshold machinery vs the REAL
//! exported C libaom v3.14.1. **Tier 1.**
//!
//! | test | C oracle |
//! |---|---|
//! | `thresh_mult_matches_c` | `av1_set_rd_speed_thresholds` |
//! | `update_rd_thresh_fact_matches_c` | `av1_update_rd_thresh_fact` (+ the static `update_thr_fact`) |
//!
//! `thresh_mult_matches_c` is the whole point of this file. `THR_MODES` is a
//! 169-entry enum whose ORDER is load-bearing, and
//! `av1_set_rd_speed_thresholds` is 169 assignments keyed by it. Comparing the
//! entire array covers the ordering and the constants at once: a wrong enum
//! index puts a right value in a wrong slot, and the array comparison sees it.
//! Nothing here is checked by reading.

use aom_encode::rd_thresh::{
    BLOCK_SIZES_ALL, MAX_MODES, THR_D45_PRED, THR_DC, THR_NEARESTMV, set_rd_speed_thresholds,
    update_rd_thresh_fact,
};
use aom_sys_ref::{ref_set_rd_speed_thresholds, ref_update_rd_thresh_fact};

#[test]
fn thresh_mult_matches_c() {
    let want = ref_set_rd_speed_thresholds();
    assert_eq!(
        want.len(),
        MAX_MODES,
        "C's MAX_MODES is {} but the port's is {MAX_MODES} — the THR_MODES enum \
         has a different number of entries, so every index below it is suspect",
        want.len()
    );
    let got = set_rd_speed_thresholds();
    // Report the first divergence with its index, since an ordering error shows
    // up as a run of shifted values and the index is what localises it.
    for i in 0..MAX_MODES {
        assert_eq!(
            got[i], want[i],
            "thresh_mult[{i}] differs: an enum-ordering error shows up here as a \
             value that belongs to a neighbouring mode"
        );
    }
    // Non-vacuity: C assigns every mode, so a zero would mean a missed
    // assignment reading as a legitimate value.
    assert!(
        want.iter().all(|&v| v != 0),
        "C left a thresh_mult entry at 0 — the table is not what this test assumes"
    );
    // And the table must not be constant, or an all-one-value port would pass.
    let first = want[0];
    assert!(
        want.iter().any(|&v| v != first),
        "every thresh_mult entry is {first} — the comparison proves nothing"
    );
}

#[test]
fn thr_mode_anchors_have_the_c_values() {
    // A second, independent check on three enum anchors the RD driver uses as
    // range bounds. If `thresh_mult_matches_c` ever became vacuous these would
    // still pin the ordering at its ends and at the inter/intra boundary.
    assert_eq!(THR_NEARESTMV, 0, "THR_MODE_START must be 0");
    assert_eq!(
        THR_D45_PRED,
        MAX_MODES - 1,
        "THR_D45_PRED must be the last entry before MAX_MODES"
    );
    assert!(
        THR_DC > THR_NEARESTMV && THR_DC < THR_D45_PRED,
        "THR_DC (= THR_INTER_MODE_END) must separate the inter and intra ranges"
    );
}

/// `BLOCK_64X64` and `BLOCK_128X128` in `BLOCK_SIZE` order — the two superblock
/// sizes AV1 allows.
const SB_SIZES: [usize; 2] = [12, 15];

#[test]
fn update_rd_thresh_fact_matches_c() {
    let mut checked = 0usize;
    let mut saw_1_to_4 = false;
    let mut saw_square = false;
    for &sb_size in &SB_SIZES {
        for bsize in 0..BLOCK_SIZES_ALL {
            for &use_adaptive in &[1i32, 2, 4] {
                for &best in &[0usize, 5, 100, MAX_MODES - 1] {
                    // A deterministic, non-uniform starting buffer: a uniform
                    // one would hide an update that touched the wrong rows.
                    let mut flat: Vec<i32> = (0..BLOCK_SIZES_ALL * MAX_MODES)
                        .map(|i| ((i * 37) % 97) as i32)
                        .collect();
                    let mut buf = vec![[0i32; MAX_MODES]; BLOCK_SIZES_ALL];
                    for (b, row) in buf.iter_mut().enumerate() {
                        row.copy_from_slice(&flat[b * MAX_MODES..(b + 1) * MAX_MODES]);
                    }

                    update_rd_thresh_fact(
                        sb_size,
                        &mut buf,
                        use_adaptive,
                        bsize,
                        best,
                        0,
                        THR_DC,
                        THR_DC,
                        MAX_MODES,
                    );
                    ref_update_rd_thresh_fact(
                        sb_size as i32,
                        &mut flat,
                        use_adaptive,
                        bsize as i32,
                        best as i32,
                        0,
                        THR_DC as i32,
                        THR_DC as i32,
                        MAX_MODES as i32,
                    );

                    for b in 0..BLOCK_SIZES_ALL {
                        assert_eq!(
                            &buf[b][..],
                            &flat[b * MAX_MODES..(b + 1) * MAX_MODES],
                            "row {b}: sb_size={sb_size} bsize={bsize} \
                             adaptive={use_adaptive} best={best}"
                        );
                    }
                    checked += 1;
                    if bsize > sb_size {
                        saw_1_to_4 = true;
                    } else {
                        saw_square = true;
                    }
                }
            }
        }
    }
    assert!(checked > 300, "grid collapsed: {checked} cells");
    // Both arms of the asymmetric block-size window must be reached: the
    // `bsize > sb_size` arm (1:4 and 4:1 shapes, which update only their own
    // row) and the +/-2 window arm.
    assert!(
        saw_1_to_4 && saw_square,
        "only one arm of the block-size window was exercised \
         (1_to_4={saw_1_to_4}, square={saw_square})"
    );
}
