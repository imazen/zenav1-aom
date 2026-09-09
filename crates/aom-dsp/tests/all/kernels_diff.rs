//! Wiener + self-guided restoration kernels vs the REAL C `_c` functions
//! (`av1_wiener_convolve_add_src_c` on u8 for bd 8 — the production lowbd
//! path — / `av1_highbd_wiener_convolve_add_src_c`;
//! `av1_apply_selfguided_restoration_c` both arms), over random padded
//! buffers: every filtered sample byte-identical, and the untouched dst
//! margin byte-identical too (no over-write).

use aom_dsp::restore::{sgr, wiener};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u64) as i32
    }
}

/// Margin around the filtered block: the kernels read up to 4 samples out
/// (wiener tap 7 loads row/col +4; SGR stages ±3).
const M: usize = 8;

fn rand_plane(rng: &mut Rng, len: usize, bd: i32) -> Vec<u16> {
    let mask = (1u64 << bd) - 1;
    (0..len).map(|_| (rng.next() & mask) as u16).collect()
}

#[test]
fn wiener_convolve_matches_c() {
    let mut rng = Rng(0xA5A5_1111_2222_4444);
    let dims: &[(usize, usize)] = &[
        (1, 1),
        (4, 4),
        (8, 3),
        (15, 16),
        (16, 15),
        (17, 29),
        (32, 32),
        (48, 56),
        (63, 64),
        (64, 64),
        (80, 28),
        (128, 8),
    ];
    for &bd in &[8, 10, 12] {
        for &(w, h) in dims {
            for case in 0..24 {
                let (buf_w, buf_h) = (w + 2 * M, h + 2 * M);
                let src = rand_plane(&mut rng, buf_w * buf_h, bd);
                let dst0 = rand_plane(&mut rng, buf_w * buf_h, bd);
                // Random valid taps; every 4th case is the chroma shape
                // (tap 0 zero).
                let chroma = case % 4 == 3;
                let mut hf = [0i16; 8];
                let mut vf = [0i16; 8];
                for f in [&mut hf, &mut vf] {
                    f[0] = if chroma { 0 } else { rng.range(-5, 10) as i16 };
                    f[1] = rng.range(-23, 8) as i16;
                    f[2] = rng.range(-17, 46) as i16;
                    f[3] = -2 * (f[0] + f[1] + f[2]);
                    f[4] = f[2];
                    f[5] = f[1];
                    f[6] = f[0];
                }
                let off = M * buf_w + M;
                let mut dst_c = dst0.clone();
                c::ref_wiener_convolve(&src, &mut dst_c, buf_w, buf_h, M, M, w, h, &hf, &vf, bd);
                let mut dst_r = dst0.clone();
                wiener::wiener_convolve_add_src(
                    &src, off, buf_w, &mut dst_r, off, buf_w, &hf, &vf, w, h, bd,
                );
                assert_eq!(
                    dst_r, dst_c,
                    "wiener {w}x{h} bd{bd} case {case} chroma={chroma}"
                );
            }
        }
    }
}

#[test]
fn apply_selfguided_matches_c() {
    let mut rng = Rng(0x5613_D00D_CAFE_F00D);
    let dims: &[(usize, usize)] = &[
        (1, 1),
        (4, 4),
        (8, 3),
        (15, 16),
        (17, 29),
        (32, 32),
        (33, 56),
        (56, 48),
        (64, 64),
        (64, 1),
        (1, 64),
    ];
    for &bd in &[8, 10, 12] {
        for &(w, h) in dims {
            for ep in 0..16usize {
                let (buf_w, buf_h) = (w + 2 * M, h + 2 * M);
                let src = rand_plane(&mut rng, buf_w * buf_h, bd);
                let dst0 = rand_plane(&mut rng, buf_w * buf_h, bd);
                let r = sgr::SGR_PARAMS[ep].0;
                let xqd = if r[0] == 0 {
                    [0, rng.range(-32, 95)]
                } else if r[1] == 0 {
                    let x0 = rng.range(-96, 31);
                    [x0, (128 - x0).clamp(-32, 95)]
                } else {
                    [rng.range(-96, 31), rng.range(-32, 95)]
                };
                let off = M * buf_w + M;
                let mut dst_c = dst0.clone();
                c::ref_apply_sgr(&src, &mut dst_c, buf_w, buf_h, M, M, w, h, ep, xqd, bd);
                let mut dst_r = dst0.clone();
                sgr::apply_selfguided_restoration(
                    &src, off, buf_w, w, h, ep, &xqd, &mut dst_r, off, buf_w, bd,
                );
                assert_eq!(dst_r, dst_c, "sgr {w}x{h} bd{bd} ep{ep} xqd{xqd:?}");
            }
        }
    }
}
