//! Differential harness for the encoder-side compound / high-bit-depth
//! predictor construction vs the REAL exported C libaom v3.14.1.
//! **Tier 1 throughout.**
//!
//! Covers `aom_encode::inter_pred_enc`:
//!
//! | test | C oracle |
//! |---|---|
//! | `comp_avg_pred_matches_c` | `aom_comp_avg_pred_c` |
//! | `comp_mask_pred_matches_c` | `aom_comp_mask_pred_c` |
//! | `highbd_comp_avg_pred_matches_c` | `aom_highbd_comp_avg_pred_c` |
//! | `highbd_comp_mask_pred_matches_c` | `aom_highbd_comp_mask_pred_c` |
//! | `comp_avg_upsampled_pred_matches_c` | `aom_comp_avg_upsampled_pred_c` |
//! | `comp_mask_upsampled_pred_matches_c` | `aom_comp_mask_upsampled_pred` |
//! | `highbd_upsampled_pred_matches_c` | `aom_highbd_upsampled_pred_c` |
//! | `highbd_comp_avg_upsampled_pred_matches_c` | `aom_highbd_comp_avg_upsampled_pred_c` |
//! | `highbd_comp_mask_upsampled_pred_matches_c` | `aom_highbd_comp_mask_upsampled_pred` |
//!
//! Every masked entry is swept at `invert_mask` **both** 0 and 1, and
//! `mask_selects_both_sources` proves the two settings actually produce
//! different output — otherwise the `invert_mask = 1` half of each sweep would
//! be a second copy of the `invert_mask = 0` half and an inverted port would
//! pass.

use aom_encode::inter_pred_enc::{
    comp_avg_pred, comp_avg_upsampled_pred, comp_mask_pred, comp_mask_upsampled_pred,
    highbd_comp_avg_pred, highbd_comp_avg_upsampled_pred, highbd_comp_mask_pred,
    highbd_comp_mask_upsampled_pred, highbd_upsampled_pred,
};
use aom_sys_ref::{
    ref_comp_avg_pred, ref_comp_avg_upsampled_pred, ref_comp_mask_pred,
    ref_comp_mask_upsampled_pred, ref_highbd_comp_avg_pred, ref_highbd_comp_avg_upsampled_pred,
    ref_highbd_comp_mask_pred, ref_highbd_comp_mask_upsampled_pred, ref_highbd_upsampled_pred,
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

/// Border on every side of the reference plane so the 8-tap upsampled
/// predictor can reach outside the block.
const BORDER: usize = 16;

const SHAPES: [(usize, usize); 10] = [
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (16, 64),
];

/// Shapes for the MASKED variants. Width 4 is excluded, and the reason is a
/// contract, not a convenience.
///
/// `aom_comp_mask_pred` and `aom_highbd_comp_mask_pred` are RTCD-dispatched. On
/// x86 their SSE2/SSSE3/AVX2 implementations branch on `width == 8`,
/// `width == 16`, else a 32-wide loop — **width 4 falls off the end of the
/// chain and nothing is written**, leaving the caller's buffer holding the
/// unblended prediction. That is not a libaom bug: masked compound (wedge and
/// DIFFWTD) is undefined below BLOCK_8X8 — `av1_is_wedge_used` has no codebook
/// there, and `WEDGE_BSIZES` in `aom-dsp/tests/compound_diff.rs` starts at 8x8
/// for the same reason. So a width-4 masked call is a size the encoder never
/// makes.
///
/// Measured 2026-08-31 by running this file on `x86_64-apple-darwin` under
/// Rosetta: `highbd_comp_mask_upsampled_pred_matches_c` failed at
/// `bd=8 4x4 phase=(0,0)` with the first row matching and the rest not — the
/// signature of a blend that never ran. On aarch64 the same names are
/// `#define`d to NEON kernels that do handle width 4, so the whole thing is
/// invisible there.
const MASKED_SHAPES: [(usize, usize); 7] = [
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (16, 64),
];

/// The subpel phase pairs swept: full-pel, x-only, y-only and both, which are
/// the four dispatch arms of `aom_upsampled_pred`.
const PHASES: [(usize, usize); 6] = [(0, 0), (3, 0), (0, 5), (3, 5), (7, 7), (1, 2)];

fn plane_u8(rng: &mut Rng, w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let buf: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
    (buf, BORDER * stride + BORDER, stride)
}

fn plane_u16(rng: &mut Rng, w: usize, h: usize, bd: u32) -> (Vec<u16>, usize, usize) {
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let maxval = 1u32 << bd;
    let buf: Vec<u16> = (0..stride * rows)
        .map(|_| rng.below(maxval) as u16)
        .collect();
    (buf, BORDER * stride + BORDER, stride)
}

/// A mask with values across the whole 0..=64 A64 range, including both
/// saturating ends, at a stride wider than the block.
fn mask_plane(rng: &mut Rng, w: usize, h: usize) -> (Vec<u8>, usize) {
    let stride = w + 5;
    let mut m: Vec<u8> = (0..stride * h).map(|_| rng.below(65) as u8).collect();
    // Guarantee both extremes appear so the blend's two limbs are both reached.
    m[0] = 0;
    if m.len() > 1 {
        m[1] = 64;
    }
    (m, stride)
}

#[test]
fn comp_avg_pred_matches_c() {
    let mut rng = Rng::new(0x11AA_22BB_33CC_44DD);
    for &(w, h) in &SHAPES {
        let (refb, off, stride) = plane_u8(&mut rng, w, h);
        let pred: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
        let got = comp_avg_pred(&pred, &refb, off, stride, w, h);
        let want = ref_comp_avg_pred(&pred, &refb, off, stride, w, h);
        assert_eq!(got, want, "{w}x{h}");
    }
}

#[test]
fn comp_mask_pred_matches_c() {
    let mut rng = Rng::new(0x55EE_66FF_7788_99AA);
    for &(w, h) in &MASKED_SHAPES {
        for &invert in &[false, true] {
            let (refb, off, stride) = plane_u8(&mut rng, w, h);
            let pred: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
            let (mask, mstride) = mask_plane(&mut rng, w, h);
            let got = comp_mask_pred(&pred, &refb, off, stride, &mask, mstride, invert, w, h);
            let want = ref_comp_mask_pred(&pred, &refb, off, stride, &mask, mstride, invert, w, h);
            assert_eq!(got, want, "{w}x{h} invert={invert}");
        }
    }
}

#[test]
fn highbd_comp_avg_pred_matches_c() {
    let mut rng = Rng::new(0xABCD_1234_5678_EF01);
    for &bd in &[8u32, 10, 12] {
        for &(w, h) in &SHAPES {
            let (refb, off, stride) = plane_u16(&mut rng, w, h, bd);
            let pred: Vec<u16> = (0..w * h).map(|_| rng.below(1 << bd) as u16).collect();
            let got = highbd_comp_avg_pred(&pred, &refb, off, stride, w, h);
            let want = ref_highbd_comp_avg_pred(&pred, &refb, off, stride, w, h);
            assert_eq!(got, want, "bd={bd} {w}x{h}");
        }
    }
}

#[test]
fn highbd_comp_mask_pred_matches_c() {
    let mut rng = Rng::new(0x0F0E_0D0C_0B0A_0908);
    for &bd in &[8u32, 10, 12] {
        for &(w, h) in &MASKED_SHAPES {
            for &invert in &[false, true] {
                let (refb, off, stride) = plane_u16(&mut rng, w, h, bd);
                let pred: Vec<u16> = (0..w * h).map(|_| rng.below(1 << bd) as u16).collect();
                let (mask, mstride) = mask_plane(&mut rng, w, h);
                let got =
                    highbd_comp_mask_pred(&pred, &refb, off, stride, &mask, mstride, invert, w, h);
                let want = ref_highbd_comp_mask_pred(
                    &pred, &refb, off, stride, &mask, mstride, invert, w, h,
                );
                assert_eq!(got, want, "bd={bd} {w}x{h} invert={invert}");
            }
        }
    }
}

#[test]
fn comp_avg_upsampled_pred_matches_c() {
    let mut rng = Rng::new(0x2020_3030_4040_5050);
    for &(w, h) in &SHAPES {
        for &(sx, sy) in &PHASES {
            let (refb, off, stride) = plane_u8(&mut rng, w, h);
            let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
            let pred: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
            let pred16: Vec<u16> = pred.iter().map(|&b| u16::from(b)).collect();
            let got = comp_avg_upsampled_pred(&pred16, &refb16, off, stride, w, h, sx, sy);
            let want = ref_comp_avg_upsampled_pred(&pred, &refb, off, stride, w, h, sx, sy);
            let got8: Vec<u8> = got.iter().map(|&v| v as u8).collect();
            assert_eq!(got8, want, "{w}x{h} phase=({sx},{sy})");
        }
    }
}

#[test]
fn comp_mask_upsampled_pred_matches_c() {
    let mut rng = Rng::new(0x6070_8090_A0B0_C0D0);
    for &(w, h) in &MASKED_SHAPES {
        for &(sx, sy) in &PHASES {
            for &invert in &[false, true] {
                let (refb, off, stride) = plane_u8(&mut rng, w, h);
                let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
                let pred: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
                let pred16: Vec<u16> = pred.iter().map(|&b| u16::from(b)).collect();
                let (mask, mstride) = mask_plane(&mut rng, w, h);
                let got = comp_mask_upsampled_pred(
                    &pred16, &refb16, off, stride, &mask, mstride, invert, w, h, sx, sy,
                );
                let want = ref_comp_mask_upsampled_pred(
                    &pred, &refb, off, stride, &mask, mstride, invert, w, h, sx, sy,
                );
                let got8: Vec<u8> = got.iter().map(|&v| v as u8).collect();
                assert_eq!(got8, want, "{w}x{h} phase=({sx},{sy}) invert={invert}");
            }
        }
    }
}

#[test]
fn highbd_upsampled_pred_matches_c() {
    let mut rng = Rng::new(0xDEAD_C0DE_FACE_B00C);
    for &bd in &[8u32, 10, 12] {
        for &(w, h) in &SHAPES {
            for &(sx, sy) in &PHASES {
                let (refb, off, stride) = plane_u16(&mut rng, w, h, bd);
                let got = highbd_upsampled_pred(&refb, off, stride, w, h, sx, sy, bd);
                let want = ref_highbd_upsampled_pred(&refb, off, stride, w, h, sx, sy, bd);
                assert_eq!(got, want, "bd={bd} {w}x{h} phase=({sx},{sy})");
            }
        }
    }
}

#[test]
fn highbd_comp_avg_upsampled_pred_matches_c() {
    let mut rng = Rng::new(0x1122_3344_5566_7788);
    for &bd in &[8u32, 10, 12] {
        for &(w, h) in &SHAPES {
            for &(sx, sy) in &PHASES {
                let (refb, off, stride) = plane_u16(&mut rng, w, h, bd);
                let pred: Vec<u16> = (0..w * h).map(|_| rng.below(1 << bd) as u16).collect();
                let got =
                    highbd_comp_avg_upsampled_pred(&pred, &refb, off, stride, w, h, sx, sy, bd);
                let want =
                    ref_highbd_comp_avg_upsampled_pred(&pred, &refb, off, stride, w, h, sx, sy, bd);
                assert_eq!(got, want, "bd={bd} {w}x{h} phase=({sx},{sy})");
            }
        }
    }
}

#[test]
fn highbd_comp_mask_upsampled_pred_matches_c() {
    let mut rng = Rng::new(0x99AA_BBCC_DDEE_FF00);
    for &bd in &[8u32, 10, 12] {
        for &(w, h) in &MASKED_SHAPES {
            for &(sx, sy) in &PHASES {
                for &invert in &[false, true] {
                    let (refb, off, stride) = plane_u16(&mut rng, w, h, bd);
                    let pred: Vec<u16> = (0..w * h).map(|_| rng.below(1 << bd) as u16).collect();
                    let (mask, mstride) = mask_plane(&mut rng, w, h);
                    let got = highbd_comp_mask_upsampled_pred(
                        &pred, &refb, off, stride, &mask, mstride, invert, w, h, sx, sy, bd,
                    );
                    let want = ref_highbd_comp_mask_upsampled_pred(
                        &pred, &refb, off, stride, &mask, mstride, invert, w, h, sx, sy, bd,
                    );
                    assert_eq!(
                        got, want,
                        "bd={bd} {w}x{h} phase=({sx},{sy}) invert={invert}"
                    );
                }
            }
        }
    }
}

#[test]
fn mask_selects_both_sources() {
    // Without this, every `invert_mask = true` assertion above could be a
    // duplicate of its `false` twin and an inverted port would still pass.
    let mut rng = Rng::new(0x4444_5555_6666_7777);
    let (w, h) = (16usize, 16usize);
    let (refb, off, stride) = plane_u8(&mut rng, w, h);
    let pred: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
    let (mask, mstride) = mask_plane(&mut rng, w, h);
    let a = comp_mask_pred(&pred, &refb, off, stride, &mask, mstride, false, w, h);
    let b = comp_mask_pred(&pred, &refb, off, stride, &mask, mstride, true, w, h);
    assert_ne!(
        a, b,
        "invert_mask made no difference — the sweep is vacuous"
    );

    let refb16: Vec<u16> = refb.iter().map(|&v| u16::from(v)).collect();
    let pred16: Vec<u16> = pred.iter().map(|&v| u16::from(v)).collect();
    let ha = highbd_comp_mask_pred(&pred16, &refb16, off, stride, &mask, mstride, false, w, h);
    let hb = highbd_comp_mask_pred(&pred16, &refb16, off, stride, &mask, mstride, true, w, h);
    assert_ne!(ha, hb, "highbd invert_mask made no difference");
}
