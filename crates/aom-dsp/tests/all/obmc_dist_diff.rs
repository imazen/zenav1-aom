//! Differential harness for the OBMC distortion kernels vs the REAL exported C
//! libaom v3.14.1. **Tier 1 throughout.**
//!
//! | test | C oracle |
//! |---|---|
//! | `obmc_variance_matches_c` | `aom_obmc_variance{W}x{H}_c` |
//! | `obmc_sub_pixel_variance_matches_c` | `aom_obmc_sub_pixel_variance{W}x{H}_c` |
//! | `highbd_obmc_variance_matches_c` | `aom_highbd_{8,10,12}_obmc_variance{W}x{H}_c` |
//! | `highbd_obmc_sub_pixel_variance_matches_c` | `aom_highbd_{8,10,12}_obmc_sub_pixel_variance{W}x{H}_c` |
//!
//! `wsrc` and `mask` are generated the way `calc_target_weighted_pred` builds
//! them — `mask` is an A64 weight scaled to 1/4096 and `wsrc` is a weighted
//! target at the same precision — rather than as unstructured noise, so the
//! per-pixel `round_signed(wsrc - pre*mask, 12)` lands in its real range.
//! `highbd_bd_arms_differ` then proves the bd 8 / 10 / 12 arms are three
//! different functions, so a port that folded them into one shift cannot pass
//! by having each arm agree with itself.
//!
//! **Known gap, stated rather than hidden:** the negative-variance clamp in the
//! bd-10 / bd-12 arms is not exercised. Deleting it from the port leaves these
//! tests green, because `sse >= sum^2/n` holds exactly before rounding and the
//! two roundings cannot flip that sign. It appears to be defensive code in C.
//! See the note on `aom_dsp::dist::obmc::highbd_obmc_variance`.

use aom_dsp::dist::obmc::{
    highbd_obmc_sub_pixel_variance, highbd_obmc_variance, obmc_sub_pixel_variance, obmc_variance,
};
use aom_sys_ref::{
    ref_highbd_obmc_sub_pixel_variance, ref_highbd_obmc_variance, ref_obmc_sub_pixel_variance,
    ref_obmc_variance,
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

/// The 22 block shapes libaom generates OBMC kernels for.
const SHAPES: [(usize, usize); 22] = [
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
];

/// `(wsrc, mask)` shaped like `calc_target_weighted_pred`'s output: `mask` is an
/// A64 weight (0..=64) scaled by 1/64 to 1/4096 precision, and `wsrc` is a
/// weighted target at the same precision.
fn wsrc_mask(rng: &mut Rng, w: usize, h: usize, maxval: u32) -> (Vec<i32>, Vec<i32>) {
    let n = w * h;
    let mut mask = vec![0i32; n];
    let mut wsrc = vec![0i32; n];
    for i in 0..n {
        let a64 = rng.below(65) as i32; // 0..=64
        mask[i] = a64 * 64; // AOM_BLEND_A64 weight raised to 1/4096
        let target = rng.below(maxval) as i32;
        wsrc[i] = target * mask[i];
    }
    // Force the extremes so the round-to-zero and full-weight cases are covered.
    mask[0] = 0;
    wsrc[0] = 0;
    if n > 1 {
        mask[1] = 64 * 64;
        wsrc[1] = (maxval as i32 - 1) * mask[1];
    }
    (wsrc, mask)
}

const BORDER: usize = 4;

#[test]
fn obmc_variance_matches_c() {
    let mut rng = Rng::new(0x1357_9BDF_2468_ACE0);
    for &(w, h) in &SHAPES {
        let stride = w + BORDER;
        let pre: Vec<u8> = (0..stride * (h + BORDER))
            .map(|_| rng.below(256) as u8)
            .collect();
        let (wsrc, mask) = wsrc_mask(&mut rng, w, h, 256);
        let got = obmc_variance(&pre, 0, stride, &wsrc, &mask, w, h);
        let want = ref_obmc_variance(&pre, 0, stride, &wsrc, &mask, w, h);
        assert_eq!(got, want, "{w}x{h}");
    }
}

#[test]
fn obmc_sub_pixel_variance_matches_c() {
    let mut rng = Rng::new(0x2468_ACE0_1357_9BDF);
    for &(w, h) in &SHAPES {
        for &(xo, yo) in &[(0usize, 0usize), (0, 4), (4, 0), (3, 5), (7, 7), (1, 6)] {
            let stride = w + BORDER;
            let pre: Vec<u8> = (0..stride * (h + BORDER))
                .map(|_| rng.below(256) as u8)
                .collect();
            let (wsrc, mask) = wsrc_mask(&mut rng, w, h, 256);
            let got = obmc_sub_pixel_variance(&pre, 0, stride, xo, yo, &wsrc, &mask, w, h);
            let want = ref_obmc_sub_pixel_variance(&pre, 0, stride, xo, yo, &wsrc, &mask, w, h);
            assert_eq!(got, want, "{w}x{h} phase=({xo},{yo})");
        }
    }
}

#[test]
fn highbd_obmc_variance_matches_c() {
    let mut rng = Rng::new(0xFACE_B00C_DEAD_BEEF);
    for &bd in &[8u32, 10, 12] {
        let maxval = 1u32 << bd;
        for &(w, h) in &SHAPES {
            let stride = w + BORDER;
            let pre: Vec<u16> = (0..stride * (h + BORDER))
                .map(|_| rng.below(maxval) as u16)
                .collect();
            let (wsrc, mask) = wsrc_mask(&mut rng, w, h, maxval);
            let got = highbd_obmc_variance(&pre, 0, stride, &wsrc, &mask, w, h, bd);
            let want = ref_highbd_obmc_variance(&pre, 0, stride, &wsrc, &mask, w, h, bd);
            assert_eq!(got, want, "bd={bd} {w}x{h}");
        }
    }
}

#[test]
fn highbd_obmc_sub_pixel_variance_matches_c() {
    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);
    for &bd in &[8u32, 10, 12] {
        let maxval = 1u32 << bd;
        for &(w, h) in &SHAPES {
            for &(xo, yo) in &[(0usize, 0usize), (2, 6), (7, 1)] {
                let stride = w + BORDER;
                let pre: Vec<u16> = (0..stride * (h + BORDER))
                    .map(|_| rng.below(maxval) as u16)
                    .collect();
                let (wsrc, mask) = wsrc_mask(&mut rng, w, h, maxval);
                let got =
                    highbd_obmc_sub_pixel_variance(&pre, 0, stride, xo, yo, &wsrc, &mask, w, h, bd);
                let want = ref_highbd_obmc_sub_pixel_variance(
                    &pre, 0, stride, xo, yo, &wsrc, &mask, w, h, bd,
                );
                assert_eq!(got, want, "bd={bd} {w}x{h} phase=({xo},{yo})");
            }
        }
    }
}

#[test]
fn highbd_bd_arms_differ() {
    // The bd 8 / 10 / 12 arms round `sum` and `sse` by different amounts and
    // differ in whether a negative variance is clamped to zero. Run all three
    // on IDENTICAL inputs: if they ever agreed everywhere, the bd sweeps above
    // would be three copies of one test and a single-shift port would pass.
    let mut rng = Rng::new(0x1111_3333_5555_7777);
    let (w, h) = (16usize, 16usize);
    let stride = w + BORDER;
    let pre: Vec<u16> = (0..stride * (h + BORDER))
        .map(|_| rng.below(4096) as u16)
        .collect();
    let (wsrc, mask) = wsrc_mask(&mut rng, w, h, 4096);
    let a = highbd_obmc_variance(&pre, 0, stride, &wsrc, &mask, w, h, 8);
    let b = highbd_obmc_variance(&pre, 0, stride, &wsrc, &mask, w, h, 10);
    let c = highbd_obmc_variance(&pre, 0, stride, &wsrc, &mask, w, h, 12);
    assert!(
        a != b && b != c && a != c,
        "the three bit-depth arms agreed: a={a:?} b={b:?} c={c:?}"
    );
}
