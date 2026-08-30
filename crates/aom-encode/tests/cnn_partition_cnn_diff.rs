//! Differential: the ported intra-CNN conv cascade
//! (`cnn_partition::cnn::cnn_predict`) vs the REAL libaom CNN engine
//! (`av1_cnn_predict_img_multi_out`, `aom_sys_ref::ref_intra_cnn_run`).
//!
//! Two comparisons:
//!   1. vs the pure **C-scalar** engine (`force_cscalar = true`) — must be
//!      BIT-EXACT: proves the Rust cascade is a faithful transcription of
//!      `av1_cnn_convolve_no_maxpool_padding_valid_c` + the layer wiring.
//!      That oracle is `shim/cnn_cscalar.c`: libaom's own `av1/encoder/cnn.c`
//!      compiled into the shim archive with the one RTCD-dispatched convolve
//!      rebound to `_c` and its exports renamed. It is scalar on EVERY target,
//!      unlike the runtime-pointer swap it replaced, which existed only on
//!      x86-64 and left this comparison silently NEON-backed on aarch64
//!      (CLAUDE.md KB-ARM-FLOAT root #2).
//!   2. vs the **dispatched** (AVX2) engine (`force_cscalar = false`, what the
//!      encoder runs) — reported as a max-abs gap, and NOT bit-exact.
//!
//! **The rationale this file used to give for (2) was wrong, and is now
//! measured wrong — KB-41 root #27.** It said the gap "only has to stay far
//! inside the DNN prec-reduce bucket so the downstream split/no-split FLAGS
//! agree (that flag-parity is asserted in the full-model diff)". Neither half
//! holds:
//!
//!   * A gap does not have to approach the bucket WIDTH to change the bucket.
//!     `av1_nn_output_prec_reduce` rounds the branch logit to 1/512, so an
//!     arbitrarily small gap moves the quantum whenever the logit sits near a
//!     boundary — and the prune compares exactly that quantum against
//!     `no_split_thresh` (`partition_strategy.c:341`). MEASURED on
//!     `2765x4096 cq6 --cpu-used 6`, mi(0,352): the port's branch features
//!     match this oracle under `force_cscalar` to the bit and differ from the
//!     DISPATCHED oracle in the 7th digit; raw logits −3.86037111 vs
//!     −3.8603348731994629 land on the ADJACENT quanta −3.859375 and
//!     −3.857421875, either side of `no_split_thresh = −3.858222961`. C splits
//!     the 32x32; the port codes it NONE.
//!   * `cnn_partition_decision_diff` asserts flag parity against the C-SCALAR
//!     oracle ("flag mismatch vs C-scalar"), never against the dispatched one —
//!     so no gate has ever covered the claim.
//!
//! This test's own printed number is the corroboration: over its 205 windows
//! the worst `|rust − AVX2|` is **7.87e-6**, i.e. the 7th digit — the same
//! magnitude by which the branch features differ at mi(0,352). The gap is
//! genuinely tiny AND it flips partitions; those are not in tension, because
//! what matters is the boundary, not the width.
//!
//! The assertion below is kept (it is a real ceiling on the convolve gap) but
//! it pins ONLY that: it is not, and cannot be, evidence of flag parity with a
//! real encoder. Closing that requires porting the dispatched convolve — root
//! #27, queued in CLAUDE.md's coverage table.

use aom_encode::cnn_partition::cnn::{CNN_OUT_BUF_SIZE, cnn_predict};
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

/// Build a 65×65 window (stride 65) from a content closure over frame coords,
/// applying the replicated top/left border (`src(max(i-1,0), max(j-1,0))`).
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

#[test]
fn cnn_predict_matches_c_scalar_bit_exact_and_reports_avx2_gap() {
    c::ref_init();
    let mut rng = XorShift(0x51ed_c0de_1234_5678);

    // A representative mix: uniform random, the real vgrad-256 content, flats,
    // two-tone, gradients, and impulse-ish patterns.
    let mut windows: Vec<Vec<u8>> = Vec::new();
    windows.push(window(|_, c| (32 + c * 190 / 256) as u8)); // vgrad-256 SB(0,0)
    windows.push(window(|_, _| 128)); // flat
    windows.push(window(|_, c| if c < 32 { 40 } else { 200 })); // two-tone
    windows.push(window(|r, c| (16 + (r + c) * 200 / 128) as u8)); // diagonal
    windows.push(window(|r, c| if (r + c) % 2 == 0 { 0 } else { 255 })); // checker
    for _ in 0..200 {
        // Pure random windows.
        let w: Vec<u8> = (0..65 * 65).map(|_| rng.u8()).collect();
        windows.push(w);
    }

    let mut worst_avx2_gap = 0.0f32;
    for (wi, win) in windows.iter().enumerate() {
        let got = cnn_predict(win);
        assert_eq!(got.len(), CNN_OUT_BUF_SIZE);

        // 1. C-scalar: BIT-EXACT.
        let want_c = c::ref_intra_cnn_run(win, true);
        for (idx, (&g, &wc)) in got.iter().zip(want_c.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                wc.to_bits(),
                "window {wi} cnn_buffer[{idx}]: rust={g} ({:#010x}) c_scalar={wc} ({:#010x})",
                g.to_bits(),
                wc.to_bits()
            );
        }

        // 2. AVX2 (encoder path): report the gap, keep it tiny.
        let want_avx2 = c::ref_intra_cnn_run(win, false);
        for (&g, &wa) in got.iter().zip(want_avx2.iter()) {
            worst_avx2_gap = worst_avx2_gap.max((g - wa).abs());
        }
    }

    eprintln!(
        "cnn_predict: {} windows BIT-EXACT vs C-scalar; worst |rust - AVX2| = {worst_avx2_gap:e}",
        windows.len()
    );
    // A CEILING on the convolve gap, and nothing more (see the module docs):
    // any nonzero gap can still move a branch logit across a 1/512 prec-reduce
    // boundary and flip `do_square_split`, which is KB-41 root #27. Kept at the
    // value it has always had so a REGRESSION in the transcription still trips
    // it. (libaom's own CNN C-vs-SIMD MSE tolerance is 1e-6.)
    assert!(
        worst_avx2_gap < 1e-2,
        "AVX2 gap {worst_avx2_gap:e} unexpectedly large — the convolve \
         transcription regressed (this bound does NOT imply flag parity)"
    );
}
