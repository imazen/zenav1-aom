//! Differential: the full ported intra-CNN partition-prune decision
//! (`cnn_partition::decision::predict_decision`) vs the REAL
//! `intra_mode_cnn_partition` (`aom_sys_ref::ref_intra_cnn_partition_decision`).
//!
//! Two bars, one single-threaded test (the C-scalar oracle toggles a global):
//!   1. **Flag parity vs the dispatched (AVX2) path** — what the encoder runs.
//!      This is the byte-exactness-relevant guarantee: the CNN's only bitstream
//!      effect is these four flags, so matching them = matching the partition
//!      search constraints. MUST hold for every case.
//!   2. **Bit-exact logits vs the pure C-scalar path** — validates the new code
//!      (log_q, feature assembly, thresholds, decision) as a faithful
//!      transcription, on top of the already-bit-exact CNN + DNN engines.
//!
//! Sweeps all four bsizes over their full quad_tree_idx ranges, the real
//! qindex band, and all three res tiers (lowres/midres/hdres via frame size).

use aom_encode::cnn_partition::decision::{CnnPruneDecision, predict_decision};
use aom_sys_ref as c;

struct XorShift(u64);
impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn u8(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
}

fn window(content: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let mut win = vec![0u8; 65 * 65];
    for i in 0..65 {
        for j in 0..65 {
            let fr = (i as i32 - 1).max(0) as usize;
            let fc = (j as i32 - 1).max(0) as usize;
            win[i * 65 + j] = content(fr, fc);
        }
    }
    win
}

fn flags_of(d: CnnPruneDecision) -> [i32; 4] {
    [
        i32::from(d.none_disallowed),
        i32::from(d.do_square_split),
        i32::from(d.rect_disabled),
        i32::from(d.square_split_disabled),
    ]
}

/// (bsize_idx, quad_tree_idx range).
const BLOCKS: &[(i32, std::ops::RangeInclusive<i32>)] =
    &[(1, 0..=0), (2, 1..=4), (3, 5..=20), (4, 21..=84)];

#[test]
fn predict_decision_matches_c() {
    c::ref_init();
    let mut rng = XorShift(0xcafe_f00d_1234_9999);

    let mut windows: Vec<Vec<u8>> = vec![
        window(|_, col| (32 + col * 190 / 256) as u8), // vgrad-256
        window(|_, _| 128),                            // flat
        window(|_, col| if col < 32 { 40 } else { 200 }), // two-tone
        window(|r, col| (16 + (r + col) * 200 / 128) as u8), // diagonal
    ];
    for _ in 0..8 {
        windows.push((0..65 * 65).map(|_| rng.u8()).collect());
    }

    // (frame_w, frame_h) picking each res tier: lowres <480, 480<=midres<720,
    // hdres >=720.
    let frames = [(256i32, 256i32), (512, 512), (1280, 720)];
    // qindex band incl. the real cq32/cq48 values (128/192) + the extremes.
    let qindices = [8i32, 128, 192, 255];
    let level = 2i32; // non-screen-content speed-1.

    let mut n = 0usize;
    let mut n_prune = 0usize;
    for win in &windows {
        for &(fw, fh) in &frames {
            for &qindex in &qindices {
                for (bsize_idx, qt_range) in BLOCKS {
                    for qt in qt_range.clone() {
                        let (logits, dec) =
                            predict_decision(win, qindex, 8, fw, fh, *bsize_idx, qt, level);
                        let got_flags = flags_of(dec);

                        // Bar 1: flag parity vs AVX2 (encoder path).
                        let (_la, flags_avx2) = c::ref_intra_cnn_partition_decision(
                            win, qindex, 8, fw, fh, *bsize_idx, qt, level, false,
                        );
                        assert_eq!(
                            got_flags, flags_avx2,
                            "FLAG MISMATCH vs AVX2: bsize_idx={bsize_idx} qt={qt} qindex={qindex} \
                             frame=({fw},{fh}) rust_logit0={} rust_flags={got_flags:?} \
                             c_flags={flags_avx2:?}",
                            logits[0]
                        );

                        // Bar 2: bit-exact logits + flags vs C-scalar.
                        let (lc, flags_c) = c::ref_intra_cnn_partition_decision(
                            win, qindex, 8, fw, fh, *bsize_idx, qt, level, true,
                        );
                        assert_eq!(
                            logits[0].to_bits(),
                            lc[0].to_bits(),
                            "LOGIT MISMATCH vs C-scalar: bsize_idx={bsize_idx} qt={qt} \
                             qindex={qindex} frame=({fw},{fh}) rust={} ({:#010x}) c={} ({:#010x})",
                            logits[0],
                            logits[0].to_bits(),
                            lc[0],
                            lc[0].to_bits()
                        );
                        assert_eq!(got_flags, flags_c, "flag mismatch vs C-scalar");

                        n += 1;
                        if dec.prunes() {
                            n_prune += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "predict_decision_matches_c: {n} cases, flag-parity vs AVX2 + bit-exact logits vs \
         C-scalar; {n_prune} of them prune"
    );
    assert!(n_prune > 0, "sweep must exercise the pruning path");
}
