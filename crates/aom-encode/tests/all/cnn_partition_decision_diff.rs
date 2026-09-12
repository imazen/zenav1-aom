//! Differential: the full ported intra-CNN partition-prune decision
//! (`cnn_partition::decision::predict_decision`) vs the REAL
//! `intra_mode_cnn_partition` (`aom_sys_ref::ref_intra_cnn_partition_decision`).
//!
//! One bar, resolved per tier: **bit-exact logits + flags vs whichever C
//! engine this process's dispatch runs** — the dispatched (AVX2) path under
//! the v3 tier (`aom_dsp::cnn::conv_valid` reproduces libaom's AVX2 convolve
//! accumulation order — KB-41 root #27 — and `decision::finish_decision`
//! pairs it with `nn::nn_predict_avx2_order`, closing #26's remaining half:
//! the whole chain now models a real aomenc on AVX2), or the pure C-scalar
//! oracle under `AOM_FORCE_SCALAR` / off x86-64 (`shim/cnn_cscalar.c`, a
//! scalar-bound copy of libaom's engine — CLAUDE.md KB-ARM-FLOAT #2 — plus
//! `av1_nn_predict_c`; the oracle runs the dispatched `av1_nn_predict` when
//! `force_cscalar` is off, matching the port's per-tier chain).
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
    // The port runs whichever convolve engine this process's dispatch resolves
    // to: the v3 AVX2 kernels (bit-exact against libaom's dispatched convolve —
    // KB-41 root #27 closed) or the `_c` transcription under the pin. The
    // oracle's `force_cscalar` flag selects the matching C engine, and the
    // comparison is bit-exact logits + flags against it — under v3 this is
    // STRONGER than the old bar, which could only assert flag parity vs AVX2.
    let simd_tier = aom_dsp::cnn::v3_tier_active();
    let mut n_prune = 0usize;
    for win in &windows {
        for &(fw, fh) in &frames {
            for &qindex in &qindices {
                for (bsize_idx, qt_range) in BLOCKS {
                    for qt in qt_range.clone() {
                        let (logits, dec) =
                            predict_decision(win, qindex, 8, fw, fh, *bsize_idx, qt, level);
                        let got_flags = flags_of(dec);

                        let (lc, flags_c) = c::ref_intra_cnn_partition_decision(
                            win, qindex, 8, fw, fh, *bsize_idx, qt, level, !simd_tier,
                        );
                        assert_eq!(
                            logits[0].to_bits(),
                            lc[0].to_bits(),
                            "LOGIT MISMATCH (simd_tier={simd_tier}): bsize_idx={bsize_idx} qt={qt} \
                             qindex={qindex} frame=({fw},{fh}) rust={} ({:#010x}) c={} ({:#010x})",
                            logits[0],
                            logits[0].to_bits(),
                            lc[0],
                            lc[0].to_bits()
                        );
                        assert_eq!(
                            got_flags, flags_c,
                            "flag mismatch (simd_tier={simd_tier}): bsize_idx={bsize_idx} qt={qt} \
                             qindex={qindex} frame=({fw},{fh}) rust={got_flags:?} c={flags_c:?}"
                        );

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
        "predict_decision_matches_c: {n} cases, bit-exact logits + flags vs {} engine; \
         {n_prune} of them prune",
        if simd_tier { "dispatched-AVX2" } else { "C-scalar" },
    );
    assert!(n_prune > 0, "sweep must exercise the pruning path");
}
