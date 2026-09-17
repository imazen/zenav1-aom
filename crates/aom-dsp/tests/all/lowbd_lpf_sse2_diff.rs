//! Differential: the v3 lowbd loop-filter mirror vs the REAL dispatched x86-64
//! kernels — `aom_lpf_{horizontal,vertical}_{4,6,8,14}_sse2` — the functions
//! libaom itself runs on v3 hardware at bd8 (the `_avx2` single-edge names
//! wrap these same SSE2 internals). Stronger than `loopfilter_lowbd_diff.rs`
//! (which oracles the `_c` scalar references): this proves bit-identity on the
//! kernels' OWN domain — including the packed-u8 saturating arithmetic and the
//! transposed tap layout the vertical kernels use.
//!
//! C's `_sse2` and `_c` lowbd kernels are NOT bit-identical to each other:
//! the mask sums run in saturating `adds_epu8`, so at `blimit == 255` a capped
//! sum (255) passes the blimit criterion where the scalar's unclamped sum
//! (>255) fails it. The sweep deliberately includes `blimit = 255` so the
//! gate covers exactly the corner where the two C oracles disagree — the port
//! must follow `_sse2`, the kernel actually dispatched.
//!
//! Positions are given PER-LANE structure: each of the 4 edge positions gets
//! an independently drawn profile (flat / soft-edge / hard-edge / hev-heavy)
//! so the lane masks diverge inside one call — the case where a broken
//! per-lane blend or a swapped transpose lane would show.

#![cfg(target_arch = "x86_64")]

use aom_dsp::loopfilter;
use aom_sys_ref as c;
use archmage::SimdToken;

const PITCH: usize = 32;
const ROWS: usize = 32;
const CENTER: usize = 12 * PITCH + 12;

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
    fn upto(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

/// Per-position profiles. `off` is the tap index relative to the edge
/// (negative = p side); returns the sample for position `pos`.
fn profile(kind: u32, pos: usize, off: i32, rng: &mut Rng) -> i32 {
    match kind {
        // flat: every tap near one level — drives flat/flat2 masks on
        0 => 110 + pos as i32 * 20 + rng.upto(3) as i32 - 1,
        // soft edge: small step across the edge — mask on, filter4 path
        1 => 90 + pos as i32 * 15 + if off >= 0 { 10 } else { 0 },
        // hard edge: big step — mask off for that lane
        2 => {
            if off >= 0 {
                200 + pos as i32 * 10
            } else {
                30 + pos as i32 * 12
            }
        }
        // hev-heavy: p1-p0 / q1-q0 contrast above most thresholds
        _ => {
            let base = 80 + pos as i32 * 20;
            if off == -2 || off == 1 {
                base + 90
            } else {
                base
            }
        }
    }
}

#[test]
fn lowbd_lpf_v3_matches_real_sse2_kernels() {
    // Sibling tests drive `for_each_token_permutation`, which disables tokens
    // process-wide between permutations — without this lock `summon()` can
    // observe a transiently-disabled v3 and the assert below flakes.
    let _token_guard = archmage::testing::lock_token_testing();
    assert!(
        archmage::X64V3Token::summon().is_some(),
        "test requires a v3-capable host"
    );
    let mut rng = Rng(0x_10bd_5e2d_ba5e_ca11);
    let mut saturation_corner = 0u32;
    for &dir in b"hv" {
        for &width in &[4u32, 6, 8, 14] {
            for _ in 0..12_000 {
                let mut buf = vec![0u8; PITCH * ROWS];
                // Structured positions: for each of the 4 positions of the
                // edge segment, draw a profile and fill its tap range.
                let kinds: [u32; 4] = [rng.upto(4), rng.upto(4), rng.upto(4), rng.upto(4)];
                for pos in 0..4usize {
                    for k in -8i32..8 {
                        let idx = if dir == b'h' {
                            (CENTER + pos) as i32 + k * PITCH as i32
                        } else {
                            (CENTER as i32) + pos as i32 * PITCH as i32 + k
                        };
                        if idx >= 0 && (idx as usize) < buf.len() {
                            buf[idx as usize] =
                                profile(kinds[pos], pos, k, &mut rng).clamp(0, 255) as u8;
                        }
                    }
                }
                // Sprinkle outliers across the rest so non-tap pixels stay
                // non-uniform (catches stray writes).
                for _ in 0..40 {
                    let i = rng.upto((PITCH * ROWS) as u32) as usize;
                    buf[i] = rng.upto(256) as u8;
                }

                let bl = match rng.upto(3) {
                    0 => rng.upto(256) as u8,
                    // 255: the saturating-mask corner where `_sse2` and `_c`
                    // genuinely differ — hit it often enough to pin it.
                    1 => 255,
                    _ => (16 + rng.upto(200)) as u8,
                };
                if bl == 255 {
                    saturation_corner += 1;
                }
                let li = if rng.upto(2) == 0 {
                    rng.upto(256) as u8
                } else {
                    (1 + rng.upto(64)) as u8
                };
                let th = rng.upto(256) as u8;

                let mut got = buf.clone();
                let mut want = buf.clone();
                if dir == b'h' {
                    loopfilter::horizontal(width, &mut got, CENTER, PITCH, bl, li, th);
                } else {
                    loopfilter::vertical(width, &mut got, CENTER, PITCH, bl, li, th);
                }
                c::ref_lpf_sse2(dir, width, &mut want, CENTER, PITCH, bl, li, th);
                assert_eq!(
                    got, want,
                    "lowbd lpf sse2 dir={} width={width} bl={bl} li={li} th={th} kinds={kinds:?}",
                    dir as char
                );
            }
        }
    }
    assert!(
        saturation_corner > 1000,
        "the blimit=255 saturating-mask corner must actually be exercised \
         ({saturation_corner} cells)"
    );
}
