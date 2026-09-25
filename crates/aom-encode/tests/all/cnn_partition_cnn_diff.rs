//! Differential: the ported intra-CNN conv cascade
//! (`cnn_partition::cnn::cnn_predict`) vs the REAL libaom CNN engine
//! (`av1_cnn_predict_img_multi_out`, `aom_sys_ref::ref_intra_cnn_run`).
//!
//! One comparison, resolved per tier — **BIT-EXACT either way**:
//!   * Under the v3 tier the port runs `aom_dsp::cnn::conv_valid` — a 1:1 port
//!     of `av1_cnn_convolve_no_maxpool_padding_valid_avx2`, every mul/add/hadd
//!     in C's order — and is compared BIT-EXACT against the **dispatched**
//!     engine (`force_cscalar = false`, what the encoder runs).
//!   * Under `AOM_FORCE_SCALAR` (or off x86-64) the port runs its `_c`
//!     transcription and is compared BIT-EXACT against the **C-scalar** engine
//!     (`force_cscalar = true`): `shim/cnn_cscalar.c`, libaom's own
//!     `av1/encoder/cnn.c` compiled with the one RTCD-dispatched convolve
//!     rebound to `_c` — scalar on every target (CLAUDE.md KB-ARM-FLOAT #2).
//!
//! **KB-41 root #27 is CLOSED by this port.** The gap it recorded — the port
//! matched C-scalar to the bit while the dispatched AVX2 engine differed in
//! the 7th digit (worst `|rust − AVX2|` = 7.87e-6 over this window set), and
//! `av1_nn_output_prec_reduce`'s 1/512 quantum turned that into a flipped
//! `do_square_split` at `2765x4096 cq6 --cpu-used 6`, mi(0,352) — was a
//! *missing kernel*, not a tolerance question. The dispatched order is now
//! what runs, so the comparison is bit-exact against the engine a real
//! aomenc uses — and the decision-level gate
//! (`cnn_partition_decision_diff`) asserts bit-exact logits + flags against
//! that same engine.

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
    /// Masked word in `0..=(1<<bd)-1`.
    fn u16_bd(&mut self, bd: i32) -> u16 {
        ((self.next_u64() >> 33) as u16) & (((1u32 << bd) - 1) as u16)
    }
}

/// Build a 65×65 window (stride 65) from a content closure over frame coords,
/// applying the replicated top/left border (`src(max(i-1,0), max(j-1,0))`).
fn window(content: impl Fn(usize, usize) -> u16) -> Vec<u16> {
    let mut win = vec![0u16; 65 * 65];
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
fn cnn_predict_matches_resolved_engine_bit_exact() {
    c::ref_init();
    let mut rng = XorShift(0x51ed_c0de_1234_5678);

    // A representative mix: uniform random, the real vgrad-256 content, flats,
    // two-tone, gradients, and impulse-ish patterns — at each bit depth (the
    // layer-0 normalisation is `pixel / ((1<<bd)-1)`, so bd10/bd12 exercise the
    // `_highbd` arm — KB-61).
    let mut windows: Vec<(i32, Vec<u16>)> = Vec::new();
    let mut push_bd = |bd: i32, n_rand: usize, rng: &mut XorShift| {
        let maxv = ((1u32 << bd) - 1) as u16;
        let s = |v: usize| -> u16 { ((v as u32 * maxv as u32) / 255) as u16 };
        windows.push((bd, window(|_, c| s(32 + c * 190 / 256)))); // vgrad-256 SB(0,0)
        windows.push((bd, window(|_, _| s(128)))); // flat
        windows.push((bd, window(|_, c| s(if c < 32 { 40 } else { 200 })))); // two-tone
        windows.push((bd, window(|r, c| s(16 + (r + c) * 200 / 128)))); // diagonal
        windows.push((bd, window(|r, c| s(if (r + c) % 2 == 0 { 0 } else { 255 })))); // checker
        for _ in 0..n_rand {
            // Pure random windows.
            let w: Vec<u16> = if bd == 8 {
                (0..65 * 65).map(|_| rng.u8() as u16).collect()
            } else {
                (0..65 * 65).map(|_| rng.u16_bd(bd)).collect()
            };
            windows.push((bd, w));
        }
    };
    push_bd(8, 200, &mut rng);
    push_bd(10, 50, &mut rng);
    push_bd(12, 50, &mut rng);

    // The port runs whichever engine this process's dispatch resolves to:
    // `aom_dsp::cnn::conv_valid`'s v3 kernels (bit-exact against libaom's OWN
    // AVX2 convolve, closing KB-41 root #27) when the v3 tier is live, the
    // scalar `_c` transcription (bit-exact against C-scalar) under the pin or
    // off x86-64. The oracle flag selects the matching C engine — the
    // comparison is BIT-EXACT in both arms.
    let simd_tier = aom_dsp::cnn::v3_tier_active();
    for (wi, (bd, win)) in windows.iter().enumerate() {
        let got = cnn_predict(win, *bd);
        assert_eq!(got.len(), CNN_OUT_BUF_SIZE);

        let want = c::ref_intra_cnn_run(win, *bd, !simd_tier);
        for (idx, (&g, &wc)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                wc.to_bits(),
                "window {wi} bd={bd} cnn_buffer[{idx}] (simd_tier={simd_tier}): \
                 rust={g} ({:#010x}) c={wc} ({:#010x})",
                g.to_bits(),
                wc.to_bits()
            );
        }
    }

    eprintln!(
        "cnn_predict: {} windows BIT-EXACT vs {} engine",
        windows.len(),
        if simd_tier {
            "dispatched-AVX2"
        } else {
            "C-scalar"
        },
    );
}
