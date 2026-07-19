//! INTER-ENCODE chunk 2d differential — `aom_upsampled_pred` (lowbd, USE_8_TAPS)
//! vs the REAL exported `aom_upsampled_pred_c`.
//!
//! `aom_upsampled_pred` is the subpel-predictor cost primitive of the speed-0
//! subpel motion search (`av1_find_best_sub_pixel_tree` → `upsampled_pref_error`
//! → `check_better` / `upsampled_setup_center_error`): it builds an 8-tap
//! (EIGHTTAP_REGULAR) subpel prediction of the reference at a 1/8-pel offset,
//! which the search then scores with the plain variance. This gate locks the
//! port's [`aom_encode::inter_me::upsampled_pred`] byte-for-byte against
//! `aom_upsampled_pred_c` across every subpel phase, all 4 dispatch arms
//! (copy / horiz / vert / 2-D), and a sweep of square + rectangular block
//! sizes.

use aom_encode::inter_me::upsampled_pred;
use aom_sys_ref::ref_upsampled_pred;

/// Deterministic xorshift64* PRNG (the port's standard differential-fuzz RNG).
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
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
}

/// Build a `(w+7)×(h+7)` reference buffer (u8) with the block origin at (3, 3)
/// — 3 samples of border before + 4 after in each direction, exactly the margin
/// the 8-tap subpel filter reads. Returns `(buf_u8, ref_off, ref_stride)`.
fn ref_block(rng: &mut Rng, w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let stride = w + 7;
    let rows = h + 7;
    let mut buf = vec![0u8; stride * rows];
    for b in buf.iter_mut() {
        *b = rng.byte();
    }
    (buf, 3 * stride + 3, stride)
}

#[test]
fn upsampled_pred_matches_real_c() {
    let mut rng = Rng::new(0xA0B1_C2D3_E4F5_0617);
    // Inter block sizes (square + rectangular) the subpel search runs on.
    let sizes = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (8, 4),
        (4, 8),
        (16, 8),
        (8, 16),
        (32, 16),
        (16, 64),
        (64, 16),
    ];
    let mut cases = 0usize;
    for &(w, h) in &sizes {
        for sx in 0..8usize {
            for sy in 0..8usize {
                // A few independent reference draws per (size, phase).
                for _ in 0..3 {
                    let (buf8, ref_off, ref_stride) = ref_block(&mut rng, w, h);
                    let buf16: Vec<u16> = buf8.iter().map(|&b| b as u16).collect();

                    let got = upsampled_pred(&buf16, ref_off, ref_stride, w, h, sx, sy);
                    let want = ref_upsampled_pred(&buf8, ref_off, ref_stride, w, h, sx, sy);

                    assert_eq!(got.len(), w * h);
                    assert_eq!(want.len(), w * h);
                    for i in 0..w * h {
                        assert_eq!(
                            got[i], want[i] as u16,
                            "mismatch w={w} h={h} sx={sx} sy={sy} at ({}, {}): port {} != C {}",
                            i / w,
                            i % w,
                            got[i],
                            want[i]
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    // Sanity: the sweep actually exercised all four dispatch arms.
    assert_eq!(cases, sizes.len() * 8 * 8 * 3);
}
