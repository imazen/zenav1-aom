//! Differential harness for the standalone full-pel inter motion-search
//! kernels vs the REAL exported C libaom v3.14.1. **Tier 1 throughout.**
//!
//! | test | C oracle |
//! |---|---|
//! | `refining_search_8p_matches_c` | `av1_refining_search_8p_c` (mcomp.c:1696) |
//! | `vector_match_matches_c` | `av1_vector_match` (mcomp.c:2276) |
//!
//! Both C functions have shapes that a "clean up while porting" transcription
//! silently changes, so the sweeps are chosen to reach them rather than to be
//! large: `refining_search_8p` de-duplicates on a moving 7x7 *grid* rather than
//! on the MV and applies its improvement test twice (before and after the MV
//! cost), and `vector_match`'s non-full cascade recentres each round on a value
//! that is only updated at the round boundary.

use aom_encode::inter_fullpel::{RefineSearch8pParams, refining_search_8p, vector_match};
use aom_encode::intrabc_search::{DvCosts, FullMvLimits, MV_MAX, MV_VALS};
use aom_sys_ref::{ref_refining_search_8p, ref_vector_match};

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
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// A monotone per-component MV cost table (value `v` at index `MV_MAX + v`).
/// The exact numbers do not matter — both sides read the same table — only that
/// they grow with `|v|` so the MV cost actually changes decisions.
fn mvcost_table(scale: i32) -> Vec<i32> {
    let mut t = vec![0i32; MV_VALS];
    for (i, e) in t.iter_mut().enumerate() {
        let v = i as i32 - MV_MAX;
        *e = v.abs() * scale + 96;
    }
    t
}

const BORDER: usize = 32;

#[test]
fn refining_search_8p_matches_c() {
    let mut rng = Rng::new(0x3C1D_77A0_9B2E_4451);
    let mvjcost = [0i32, 240, 240, 480];
    let sizes = [
        (4usize, 4usize),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (8, 4),
        (16, 8),
        (16, 64),
    ];
    let mut moved_at_least_once = false;
    for &(w, h) in &sizes {
        for &start in &[(0i32, 0i32), (2, -3), (-5, 4), (7, 7), (-9, 1)] {
            for &full_ref in &[(0i32, 0i32), (1, 1), (-4, 6)] {
                for &sad_per_bit in &[0i32, 32, 512] {
                    // A reference plane with a planted better match a few
                    // pixels away, so the refinement actually walks.
                    let stride = w + 2 * BORDER;
                    let rows = h + 2 * BORDER;
                    let mut refb = vec![0u8; stride * rows];
                    for b in refb.iter_mut() {
                        *b = rng.byte();
                    }
                    let ref_origin = BORDER * stride + BORDER;
                    // Source = a crop of the reference at a nearby full-pel
                    // offset, so a real minimum exists inside the 8p range.
                    let planted = (
                        rng.below(5) as i32 - 2 + start.0,
                        rng.below(5) as i32 - 2 + start.1,
                    );
                    let base = (ref_origin as isize
                        + planted.0 as isize * stride as isize
                        + planted.1 as isize) as usize;
                    let mut src = vec![0u8; w * h];
                    for y in 0..h {
                        for x in 0..w {
                            src[y * w + x] = refb[base + y * stride + x];
                        }
                    }

                    let mvcost0 = mvcost_table(48);
                    let mvcost1 = mvcost_table(64);
                    let dv = DvCosts {
                        joint_mv: mvjcost,
                        dv_costs: [mvcost0.clone(), mvcost1.clone()],
                    };
                    let limits = (-20i32, 20i32, -20i32, 20i32);

                    let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
                    let src16: Vec<u16> = src.iter().map(|&b| u16::from(b)).collect();

                    let got = refining_search_8p(
                        &RefineSearch8pParams {
                            src: &src16,
                            src_off: 0,
                            src_stride: w,
                            refb: &refb16,
                            ref_origin,
                            ref_stride: stride,
                            w,
                            h,
                            limits: FullMvLimits {
                                col_min: limits.2,
                                col_max: limits.3,
                                row_min: limits.0,
                                row_max: limits.1,
                            },
                            dv: &dv,
                            full_ref_mv: full_ref,
                            sad_per_bit,
                        },
                        start,
                    );

                    let (want_mv, want_sad) = ref_refining_search_8p(
                        &src,
                        w,
                        &refb,
                        ref_origin,
                        stride,
                        w,
                        h,
                        start,
                        full_ref,
                        &mvjcost,
                        &mvcost0,
                        &mvcost1,
                        sad_per_bit,
                        limits,
                    );

                    let label = format!(
                        "{w}x{h} start={start:?} full_ref={full_ref:?} spb={sad_per_bit} \
                         planted={planted:?}"
                    );
                    assert_eq!(got.best_mv, want_mv, "best_mv: {label}");
                    assert_eq!(got.best_sad, want_sad, "best_sad: {label}");
                    if got.best_mv != start {
                        moved_at_least_once = true;
                    }
                }
            }
        }
    }
    assert!(
        moved_at_least_once,
        "the refinement never left its start MV — the 8-neighbour walk, the \
         moving visited-grid and the double improvement test are all untested"
    );
}

#[test]
fn refining_search_8p_respects_tight_limits() {
    // A separate cell where the limits clamp the START MV and then forbid every
    // neighbour, which is the path that exercises clamp_fullmv + the
    // in-range rejection rather than the walk.
    let mut rng = Rng::new(0xA10C_4D33_0011_2299);
    let (w, h) = (8usize, 8usize);
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let mut refb = vec![0u8; stride * rows];
    for b in refb.iter_mut() {
        *b = rng.byte();
    }
    let ref_origin = BORDER * stride + BORDER;
    let src: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
    let mvjcost = [0i32, 240, 240, 480];
    let mvcost0 = mvcost_table(48);
    let mvcost1 = mvcost_table(64);
    let dv = DvCosts {
        joint_mv: mvjcost,
        dv_costs: [mvcost0.clone(), mvcost1.clone()],
    };
    let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
    let src16: Vec<u16> = src.iter().map(|&b| u16::from(b)).collect();
    // A single legal MV.
    let limits = (3i32, 3i32, -2i32, -2i32);
    for &start in &[(0i32, 0i32), (99, -99), (3, -2)] {
        let got = refining_search_8p(
            &RefineSearch8pParams {
                src: &src16,
                src_off: 0,
                src_stride: w,
                refb: &refb16,
                ref_origin,
                ref_stride: stride,
                w,
                h,
                limits: FullMvLimits {
                    col_min: limits.2,
                    col_max: limits.3,
                    row_min: limits.0,
                    row_max: limits.1,
                },
                dv: &dv,
                full_ref_mv: (0, 0),
                sad_per_bit: 64,
            },
            start,
        );
        let (want_mv, want_sad) = ref_refining_search_8p(
            &src,
            w,
            &refb,
            ref_origin,
            stride,
            w,
            h,
            start,
            (0, 0),
            &mvjcost,
            &mvcost0,
            &mvcost1,
            64,
            limits,
        );
        assert_eq!(got.best_mv, want_mv, "start={start:?}");
        assert_eq!(got.best_sad, want_sad, "start={start:?}");
        assert_eq!(got.best_mv, (3, -2), "the clamp did not bind");
    }
}

#[test]
fn vector_match_matches_c() {
    let mut rng = Rng::new(0x6BB1_0E22_47FC_A390);
    // bwl is contractually {2, 3, 4, 5} — `aom_vector_var_c`'s own header
    // comment says so and `aom_vector_var_neon` asserts it. Below 2 the NEON
    // kernel's `width -= 8` underflows its `while (width != 0)` loop and reads
    // off the end of the buffer (measured: SIGBUS on aarch64 at bwl 0/1). This
    // sweep stays inside the contract rather than pinning undefined behaviour.
    for &bwl in &[2i32, 3, 4, 5] {
        let width = 4usize << bwl;
        for &(top, bottom) in &[(0i32, 0i32), (8, 8), (16, 16), (32, 32), (7, 23), (64, 64)] {
            let bw = (top + bottom) as usize;
            for &full_search in &[true, false] {
                for trial in 0..6 {
                    let src: Vec<i16> = (0..width)
                        .map(|_| (rng.below(4096) as i32 - 2048) as i16)
                        .collect();
                    let mut reff: Vec<i16> = (0..bw + width)
                        .map(|_| (rng.below(4096) as i32 - 2048) as i16)
                        .collect();
                    // On half the trials plant an exact match at a random
                    // position, so the search has a true minimum to find rather
                    // than only noise; a noise-only sweep would agree with a
                    // wrong cascade far too often.
                    if trial % 2 == 0 && bw > 0 {
                        let at = rng.below(bw as u32 + 1) as usize;
                        reff[at..at + width].copy_from_slice(&src);
                    }
                    let got = vector_match(&reff, &src, bwl, top, bottom, full_search);
                    let want = ref_vector_match(&reff, &src, bwl, top, bottom, full_search);
                    assert_eq!(
                        got, want,
                        "bwl={bwl} top={top} bottom={bottom} full={full_search} trial={trial}"
                    );
                }
            }
        }
    }
}

#[test]
fn vector_match_full_and_cascade_differ_somewhere() {
    // The non-full cascade is not guaranteed to find the exhaustive minimum.
    // If it always agreed with `full_search`, the cascade's recentring shape
    // would be untested by `vector_match_matches_c` on the cheap cells.
    let mut rng = Rng::new(0x1357_9BDF_2468_ACE0);
    let bwl = 2i32;
    let width = 4usize << bwl;
    let (top, bottom) = (32i32, 32i32);
    let bw = (top + bottom) as usize;
    let mut differed = false;
    for _ in 0..200 {
        let src: Vec<i16> = (0..width)
            .map(|_| (rng.below(4096) as i32 - 2048) as i16)
            .collect();
        let reff: Vec<i16> = (0..bw + width)
            .map(|_| (rng.below(4096) as i32 - 2048) as i16)
            .collect();
        if vector_match(&reff, &src, bwl, top, bottom, true)
            != vector_match(&reff, &src, bwl, top, bottom, false)
        {
            differed = true;
            break;
        }
    }
    assert!(
        differed,
        "the coarse-to-fine cascade agreed with the exhaustive search on every \
         trial — the cascade path is not being distinguished"
    );
}

#[test]
fn refining_search_8p_breaks_ties_like_c() {
    // Random content almost never produces an exact tie between two
    // neighbours, so the sweep above cannot see the probe ORDER at all (with a
    // strict `<`, a unique minimum is order-independent). This cell forces the
    // tie: a horizontally period-2 reference makes every odd-column window
    // identical, so (0,-1) and (0,+1) have the same SAD *and* the same MV cost.
    // C's neighbour order (cardinals up/left/right/down, then diagonals) then
    // decides the winner.
    let (w, h) = (8usize, 8usize);
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let mut refb = vec![0u8; stride * rows];
    for y in 0..rows {
        for x in 0..stride {
            refb[y * stride + x] = if x % 2 == 0 { 10 } else { 200 };
        }
    }
    let ref_origin = BORDER * stride + BORDER;
    // Source = the window one column to the right of the origin (odd start).
    let mut src = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            src[y * w + x] = refb[ref_origin + 1 + y * stride + x];
        }
    }
    let mvjcost = [0i32, 240, 240, 480];
    let mvcost0 = mvcost_table(48);
    let mvcost1 = mvcost_table(64);
    let dv = DvCosts {
        joint_mv: mvjcost,
        dv_costs: [mvcost0.clone(), mvcost1.clone()],
    };
    let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
    let src16: Vec<u16> = src.iter().map(|&b| u16::from(b)).collect();
    let limits = (-20i32, 20i32, -20i32, 20i32);
    for &sad_per_bit in &[0i32, 64, 512] {
        let got = refining_search_8p(
            &RefineSearch8pParams {
                src: &src16,
                src_off: 0,
                src_stride: w,
                refb: &refb16,
                ref_origin,
                ref_stride: stride,
                w,
                h,
                limits: FullMvLimits {
                    col_min: limits.2,
                    col_max: limits.3,
                    row_min: limits.0,
                    row_max: limits.1,
                },
                dv: &dv,
                full_ref_mv: (0, 0),
                sad_per_bit,
            },
            (0, 0),
        );
        let (want_mv, want_sad) = ref_refining_search_8p(
            &src,
            w,
            &refb,
            ref_origin,
            stride,
            w,
            h,
            (0, 0),
            (0, 0),
            &mvjcost,
            &mvcost0,
            &mvcost1,
            sad_per_bit,
            limits,
        );
        assert_eq!(got.best_mv, want_mv, "spb={sad_per_bit}");
        assert_eq!(got.best_sad, want_sad, "spb={sad_per_bit}");
    }
}

// ===================================================================
// The OBMC full-pel motion search.
// ===================================================================

use aom_encode::inter_fullpel::{ObmcFullPelParams, get_obmc_mvpred_var, obmc_full_pixel_search};
use aom_sys_ref::ref_obmc_full_pixel_search;

/// `(wsrc, mask)` shaped like `calc_target_weighted_pred`'s output: the mask is
/// an A64 weight raised to 1/4096 and `wsrc` is a target weighted by it, built
/// from a planted reference window so a real minimum exists.
fn obmc_wsrc_mask(
    rng: &mut Rng,
    refb: &[u8],
    ref_origin: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    planted: (i32, i32),
) -> (Vec<i32>, Vec<i32>) {
    let base = (ref_origin as isize + planted.0 as isize * ref_stride as isize + planted.1 as isize)
        as usize;
    let n = w * h;
    let mut mask = vec![0i32; n];
    let mut wsrc = vec![0i32; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a64 = (rng.next_u64() % 65) as i32;
            mask[i] = a64 * 64;
            // Target = the planted window plus a little noise, weighted.
            let t = i32::from(refb[base + y * ref_stride + x]) + (rng.next_u64() % 9) as i32 - 4;
            wsrc[i] = t.clamp(0, 255) * mask[i];
        }
    }
    (wsrc, mask)
}

/// The block shapes the shim has both an `aom_obmc_sad` and an
/// `aom_obmc_variance` kernel for.
const OBMC_SIZES: [(usize, usize); 9] = [
    (4, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (64, 64),
];

#[test]
fn obmc_full_pixel_search_matches_c() {
    let mut rng = Rng::new(0x0B3C_1D2E_3F40_5162);
    let mut moved = false;
    for &(w, h) in &OBMC_SIZES {
        for &fast in &[false, true] {
            for &step_param in &[0usize, 2, 4, 6] {
                for &start in &[(0i32, 0i32), (3, -2), (-6, 5)] {
                    let stride = w + 2 * BORDER;
                    let rows = h + 2 * BORDER;
                    let mut refb = vec![0u8; stride * rows];
                    for b in refb.iter_mut() {
                        *b = rng.byte();
                    }
                    let ref_origin = BORDER * stride + BORDER;
                    let planted = (
                        start.0 + (rng.next_u64() % 7) as i32 - 3,
                        start.1 + (rng.next_u64() % 7) as i32 - 3,
                    );
                    let (wsrc, mask) =
                        obmc_wsrc_mask(&mut rng, &refb, ref_origin, stride, w, h, planted);

                    let mvjcost = [0i32, 240, 240, 480];
                    let mvcost0 = mvcost_table(48);
                    let mvcost1 = mvcost_table(64);
                    let dv = DvCosts {
                        joint_mv: mvjcost,
                        dv_costs: [mvcost0.clone(), mvcost1.clone()],
                    };
                    let limits = (-16i32, 16i32, -16i32, 16i32);
                    let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();

                    let p = ObmcFullPelParams {
                        refb: &refb16,
                        ref_origin,
                        ref_stride: stride,
                        w,
                        h,
                        wsrc: &wsrc,
                        obmc_mask: &mask,
                        limits: FullMvLimits {
                            col_min: limits.2,
                            col_max: limits.3,
                            row_min: limits.0,
                            row_max: limits.1,
                        },
                        dv: &dv,
                        full_ref_mv: (0, 0),
                        sad_per_bit: 64,
                        error_per_bit: 256,
                    };
                    let got = obmc_full_pixel_search(&p, start, step_param, fast);
                    let want = ref_obmc_full_pixel_search(
                        &refb,
                        ref_origin,
                        stride,
                        w,
                        h,
                        &wsrc,
                        &mask,
                        start,
                        (0, 0),
                        &mvjcost,
                        &mvcost0,
                        &mvcost1,
                        256,
                        64,
                        step_param,
                        fast,
                        limits,
                    );
                    let label = format!(
                        "{w}x{h} fast={fast} step={step_param} start={start:?} \
                         planted={planted:?}"
                    );
                    assert_eq!(got.1, want.1, "best_mv: {label}");
                    assert_eq!(got.0, want.0, "bestsme: {label}");
                    if got.1 != start {
                        moved = true;
                    }
                }
            }
        }
    }
    assert!(
        moved,
        "the OBMC search never left its start MV — neither the diamond walk nor \
         the refinement is actually exercised"
    );
}

#[test]
fn obmc_fast_and_diamond_arms_differ() {
    // `fast_obmc_search` selects two structurally different searches (a diamond
    // vs a clamped 4-neighbour refinement re-scored on the variance metric).
    // If they agreed everywhere, the `fast = true` half of the sweep above
    // would be a duplicate of the `fast = false` half.
    let mut rng = Rng::new(0x7788_99AA_BBCC_DDEE);
    let (w, h) = (16usize, 16usize);
    let mut differed = false;
    for _ in 0..40 {
        let stride = w + 2 * BORDER;
        let rows = h + 2 * BORDER;
        let mut refb = vec![0u8; stride * rows];
        for b in refb.iter_mut() {
            *b = rng.byte();
        }
        let ref_origin = BORDER * stride + BORDER;
        let planted = (
            (rng.next_u64() % 13) as i32 - 6,
            (rng.next_u64() % 13) as i32 - 6,
        );
        let (wsrc, mask) = obmc_wsrc_mask(&mut rng, &refb, ref_origin, stride, w, h, planted);
        let mvjcost = [0i32, 240, 240, 480];
        let mvcost0 = mvcost_table(48);
        let mvcost1 = mvcost_table(64);
        let dv = DvCosts {
            joint_mv: mvjcost,
            dv_costs: [mvcost0, mvcost1],
        };
        let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
        let p = ObmcFullPelParams {
            refb: &refb16,
            ref_origin,
            ref_stride: stride,
            w,
            h,
            wsrc: &wsrc,
            obmc_mask: &mask,
            limits: FullMvLimits {
                col_min: -16,
                col_max: 16,
                row_min: -16,
                row_max: 16,
            },
            dv: &dv,
            full_ref_mv: (0, 0),
            sad_per_bit: 64,
            error_per_bit: 256,
        };
        if obmc_full_pixel_search(&p, (0, 0), 4, false)
            != obmc_full_pixel_search(&p, (0, 0), 4, true)
        {
            differed = true;
            break;
        }
    }
    assert!(
        differed,
        "the diamond and fast-refinement arms agreed on every trial — one of \
         them is not being distinguished"
    );
}

#[test]
fn get_obmc_mvpred_var_is_the_reported_score() {
    // The search returns `get_obmc_mvpred_var(best_mv)` on the diamond arm, so
    // the two must agree; this pins the scorer independently of the walk.
    let mut rng = Rng::new(0x3141_5926_5358_9793);
    let (w, h) = (16usize, 16usize);
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let mut refb = vec![0u8; stride * rows];
    for b in refb.iter_mut() {
        *b = rng.byte();
    }
    let ref_origin = BORDER * stride + BORDER;
    let (wsrc, mask) = obmc_wsrc_mask(&mut rng, &refb, ref_origin, stride, w, h, (2, -1));
    let mvjcost = [0i32, 240, 240, 480];
    let mvcost0 = mvcost_table(48);
    let mvcost1 = mvcost_table(64);
    let dv = DvCosts {
        joint_mv: mvjcost,
        dv_costs: [mvcost0, mvcost1],
    };
    let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
    let p = ObmcFullPelParams {
        refb: &refb16,
        ref_origin,
        ref_stride: stride,
        w,
        h,
        wsrc: &wsrc,
        obmc_mask: &mask,
        limits: FullMvLimits {
            col_min: -16,
            col_max: 16,
            row_min: -16,
            row_max: 16,
        },
        dv: &dv,
        full_ref_mv: (0, 0),
        sad_per_bit: 64,
        error_per_bit: 256,
    };
    let (sme, mv) = obmc_full_pixel_search(&p, (0, 0), 4, false);
    assert_eq!(sme, get_obmc_mvpred_var(&p, mv));
}

// ===================================================================
// The OBMC sub-pel motion search.
// ===================================================================

use aom_encode::inter_fullpel::{
    ObmcSubpelParams, ObmcSubpelSearchType, find_best_obmc_sub_pixel_tree_up,
};
use aom_encode::inter_me::SubpelMvLimits;
use aom_sys_ref::ref_find_best_obmc_sub_pixel_tree_up;

#[test]
fn obmc_sub_pixel_tree_up_matches_c() {
    let mut rng = Rng::new(0x5A5A_C3C3_9696_0F0F);
    let mut moved = false;
    for &(w, h) in &OBMC_SIZES {
        for &use_2_taps_orig in &[false, true] {
            for &(allow_hp, forced_stop, iters) in &[
                (true, 0, 2),
                (false, 0, 2),
                (true, 1, 2),
                (true, 2, 2),
                (true, 3, 2),
                (true, 0, 1),
            ] {
                for &start in &[(0i32, 0i32), (8, 0), (0, 8), (8, 8), (-8, -8)] {
                    let stride = w + 2 * BORDER;
                    let rows = h + 2 * BORDER;
                    let mut refb = vec![0u8; stride * rows];
                    for b in refb.iter_mut() {
                        *b = rng.byte();
                    }
                    let ref_origin = BORDER * stride + BORDER;
                    let planted = (start.0 >> 3, start.1 >> 3);
                    let (wsrc, mask) =
                        obmc_wsrc_mask(&mut rng, &refb, ref_origin, stride, w, h, planted);

                    let mvjcost = [0i32, 240, 240, 480];
                    let mvcost0 = mvcost_table(48);
                    let mvcost1 = mvcost_table(64);
                    let dv = DvCosts {
                        joint_mv: mvjcost,
                        dv_costs: [mvcost0.clone(), mvcost1.clone()],
                    };
                    let limits = (-256i32, 256i32, -256i32, 256i32);
                    let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();

                    let p = ObmcSubpelParams {
                        refb: &refb16,
                        ref_origin,
                        ref_stride: stride,
                        w,
                        h,
                        wsrc: &wsrc,
                        obmc_mask: &mask,
                        start_mv: start,
                        ref_mv: start,
                        dv: &dv,
                        error_per_bit: 256,
                        allow_hp,
                        forced_stop,
                        iters_per_step: iters,
                        limits: SubpelMvLimits {
                            row_min: limits.0,
                            row_max: limits.1,
                            col_min: limits.2,
                            col_max: limits.3,
                        },
                        search_type: if use_2_taps_orig {
                            ObmcSubpelSearchType::Use2TapsOrig
                        } else {
                            ObmcSubpelSearchType::Upsampled
                        },
                    };
                    let got = find_best_obmc_sub_pixel_tree_up(&p);
                    let want = ref_find_best_obmc_sub_pixel_tree_up(
                        &refb,
                        ref_origin,
                        stride,
                        w,
                        h,
                        &wsrc,
                        &mask,
                        start,
                        start,
                        &mvjcost,
                        &mvcost0,
                        &mvcost1,
                        256,
                        allow_hp,
                        forced_stop,
                        iters,
                        use_2_taps_orig,
                        limits,
                    );
                    let label = format!(
                        "{w}x{h} 2taps={use_2_taps_orig} hp={allow_hp} stop={forced_stop} \
                         iters={iters} start={start:?}"
                    );
                    assert_eq!(got.best_mv, want.best_mv, "best_mv: {label}");
                    assert_eq!(got.distortion, want.distortion, "distortion: {label}");
                    assert_eq!(got.sse, want.sse, "sse: {label}");
                    assert_eq!(got.besterr, want.besterr, "besterr: {label}");
                    if got.best_mv != start {
                        moved = true;
                    }
                }
            }
        }
    }
    assert!(
        moved,
        "the OBMC subpel tree never refined away from its start MV — the \
         cardinal/diagonal walk is untested"
    );
}

#[test]
fn obmc_subpel_search_types_differ() {
    // USE_2_TAPS_ORIG and the upsampled arm use different predictors AND
    // different MV-rate formulas (`estimate_obmc_mvcost` shifts by 13 with a
    // spurious x8 on the diff, `mv_err_cost_` by 14 without it). If they ever
    // agreed everywhere, the `use_2_taps_orig = true` half of the sweep above
    // would be a duplicate.
    let mut rng = Rng::new(0x0F1E_2D3C_4B5A_6978);
    let (w, h) = (16usize, 16usize);
    let mut differed = false;
    for _ in 0..40 {
        let stride = w + 2 * BORDER;
        let rows = h + 2 * BORDER;
        let mut refb = vec![0u8; stride * rows];
        for b in refb.iter_mut() {
            *b = rng.byte();
        }
        let ref_origin = BORDER * stride + BORDER;
        // A NON-ZERO start MV is required here: at start (0, 0) the two arms'
        // centre errors coincide, because `setup_obmc_center_error`'s
        // origin-instead-of-MV read (upstream's own acknowledged bug) happens
        // to be the same window the upsampled arm builds. That degeneracy is
        // exactly what this test exists to rule out.
        let start = (16i32, -8i32);
        let (wsrc, mask) = obmc_wsrc_mask(&mut rng, &refb, ref_origin, stride, w, h, (2, -1));
        let mvjcost = [0i32, 240, 240, 480];
        let dv = DvCosts {
            joint_mv: mvjcost,
            dv_costs: [mvcost_table(48), mvcost_table(64)],
        };
        let refb16: Vec<u16> = refb.iter().map(|&b| u16::from(b)).collect();
        let mk = |t| ObmcSubpelParams {
            refb: &refb16,
            ref_origin,
            ref_stride: stride,
            w,
            h,
            wsrc: &wsrc,
            obmc_mask: &mask,
            start_mv: start,
            ref_mv: start,
            dv: &dv,
            error_per_bit: 256,
            allow_hp: true,
            forced_stop: 0,
            iters_per_step: 2,
            limits: SubpelMvLimits {
                row_min: -256,
                row_max: 256,
                col_min: -256,
                col_max: 256,
            },
            search_type: t,
        };
        let a = find_best_obmc_sub_pixel_tree_up(&mk(ObmcSubpelSearchType::Upsampled));
        let b = find_best_obmc_sub_pixel_tree_up(&mk(ObmcSubpelSearchType::Use2TapsOrig));
        if a != b {
            differed = true;
            break;
        }
    }
    assert!(
        differed,
        "the two OBMC subpel search types agreed on every trial — one of them \
         is not being distinguished"
    );
}
