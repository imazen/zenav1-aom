//! Differential harness for the compound-type decision layer of
//! `av1/encoder/compound_type.c` — the port in `aom_encode::compound_type`.
//!
//! # Evidence tier — read this before trusting a green run
//!
//! Every function under test is `static` in C. `nm -g upstream/build/libaom.a`
//! reports exactly TWO exported symbols for the whole file
//! (`av1_compound_type_rd`, `av1_handle_inter_intra_mode`), and neither is a
//! decision helper. There is therefore no exported symbol to take the address
//! of, and the alternative to this harness is hand-derived vectors read off
//! the C source — which CLAUDE.md ranks last ("transcribed oracles can carry
//! shared bugs") and this repo labels tier 4.
//!
//! So `crates/aom-sys-ref/shim/compound_type_shim.c` compiles libaom's OWN
//! compound_type.c into the shim archive, with its two exports renamed out of
//! the way, and exposes flat wrappers around the statics. The bodies under
//! test are libaom's source. **Tier 1c**: real C source, compiled verbatim, as
//! against tier 1's real symbol out of the archive.
//!
//! The one gap between 1c and 1 is that this is a SECOND COMPILATION of the
//! same source, which could in principle diverge from the archive's copy
//! through flags. [`shim_tu_constants_match_the_port`] closes the part of that
//! gap this file depends on: the shim TU reports `COMPOUND_TYPES`,
//! `MAX_WEDGE_TYPES` and `TOP_COMP_AVG_EST_RD_COUNT` as it sees them, and the
//! port asserts against those rather than against its own copies. (It does not
//! prove the two compilations agree on arithmetic; no function in this file is
//! also exported, so there is nothing here to cross-check the way
//! `rdopt_mv_diff.rs` cross-checks `av1_block_error_c`.)
//!
//! | test | C function (`av1/encoder/compound_type.c`) |
//! |---|---|
//! | `enable_wedge_search_matches_c` | `enable_wedge_search` `:103` |
//! | `enable_wedge_interinter_search_matches_c` | `enable_wedge_interinter_search` `:110` |
//! | `enable_wedge_interintra_search_matches_c` | `enable_wedge_interintra_search` `:116` |
//! | `compute_valid_comp_types_matches_c` | `compute_valid_comp_types` `:868` |
//! | `calc_masked_type_cost_matches_c` | `calc_masked_type_cost` `:906` |
//! | `update_mbmi_for_compound_type_matches_c` | `update_mbmi_for_compound_type` `:945` |
//! | `get_interinter_compound_mask_rate_matches_c` | `get_interinter_compound_mask_rate` `:1026` |
//! | `save_mask_search_results_matches_c` | `save_mask_search_results` `:1058` |
//! | `push_comp_avg_est_rd_matches_c` | `push_comp_avg_est_rd` `:737` |
//! | `prune_comp_eval_using_comp_avg_est_rd_matches_c` | `prune_comp_eval_using_comp_avg_est_rd` `:761` |
//! | `compute_rd_thresh_matches_c` | `compute_rd_thresh` `:504` |
//!
//! # Non-vacuity
//!
//! Predicates are cheap to pass by accident: a port that returns `false`
//! always agrees with C wherever C returns false. So each boolean test counts
//! how often the oracle said `true` and asserts the count is strictly inside
//! `0..n` — a constant-answer port fails even when it agrees everywhere else.

mod common;
use common::Rng;

use aom_encode::compound_type::{
    COMPOUND_TYPES, CompoundType, DistWtdCompFlag, TOP_COMP_AVG_EST_RD_COUNT, ValidCompTypeCfg,
    calc_masked_type_cost, compute_rd_thresh, compute_valid_comp_types,
    enable_wedge_interinter_search, enable_wedge_interintra_search, enable_wedge_search,
    get_interinter_compound_mask_rate, prune_comp_eval_using_comp_avg_est_rd, push_comp_avg_est_rd,
    save_mask_search_results,
};
use aom_sys_ref as cref;

/// `MAX_WEDGE_TYPES` (`av1/common/enums.h`).
const MAX_WEDGE_TYPES: usize = 16;
/// `BLOCK_SIZES_ALL`.
const BLOCK_SIZES_ALL: usize = 22;
/// `NEW_NEWMV` — the one `PREDICTION_MODE` `save_mask_search_results` names.
/// (`MB_MODE_COUNT` is 25; `NEW_NEWMV` is the last inter mode.)
const NEW_NEWMV: i32 = 24;
/// `MB_MODE_COUNT`.
const MB_MODE_COUNT: i32 = 25;

/// The constants the port carries as its own copies, checked against the
/// oracle TU's rather than assumed equal.
#[test]
fn shim_tu_constants_match_the_port() {
    let (top_cnt, comp_types, max_wedge) = cref::ref_ct_constants();
    assert_eq!(top_cnt as usize, TOP_COMP_AVG_EST_RD_COUNT);
    assert_eq!(comp_types as usize, COMPOUND_TYPES);
    assert_eq!(max_wedge as usize, MAX_WEDGE_TYPES);
    // And the port's enum discriminants are the bitstream's.
    for (i, ty) in CompoundType::ALL.iter().enumerate() {
        assert_eq!(ty.index(), i);
    }
}

#[test]
fn enable_wedge_search_matches_c() {
    let mut rng = Rng(0x5EED_0C01);
    let mut trues = 0usize;
    let mut n = 0usize;
    // Sweep the boundary explicitly (var == thresh, ±1) as well as at random:
    // the predicate is a single `>`, so an off-by-one is only visible there.
    let mut cases: Vec<(u32, u32)> = Vec::new();
    for t in [0u32, 1, 7, 100, 1000, u32::MAX - 1, u32::MAX] {
        for d in [0i64, -1, 1] {
            let v = (i64::from(t) + d).clamp(0, i64::from(u32::MAX)) as u32;
            cases.push((v, t));
        }
    }
    for _ in 0..2000 {
        cases.push((rng.next() as u32, rng.next() as u32));
    }
    for (var, thresh) in cases {
        let got = enable_wedge_search(var, thresh);
        let want = cref::ref_ct_enable_wedge_search(var, thresh);
        assert_eq!(got, want, "enable_wedge_search(var={var}, thresh={thresh})");
        trues += usize::from(want);
        n += 1;
    }
    assert!(trues > 0 && trues < n, "vacuous: {trues}/{n} true");
}

#[test]
fn enable_wedge_interinter_search_matches_c() {
    let mut rng = Rng(0x5EED_0C02);
    let (mut trues, mut n) = (0usize, 0usize);
    for _ in 0..3000 {
        let var = rng.next() as u32;
        let thresh = if rng.next() % 2 == 0 {
            var.wrapping_add((rng.next() % 3) as u32).wrapping_sub(1)
        } else {
            rng.next() as u32
        };
        let en = rng.next() % 2 == 0;
        let got = enable_wedge_interinter_search(var, thresh, en);
        let want = cref::ref_ct_enable_wedge_interinter_search(var, thresh, en);
        assert_eq!(got, want, "interinter(var={var}, thresh={thresh}, en={en})");
        trues += usize::from(want);
        n += 1;
    }
    assert!(trues > 0 && trues < n, "vacuous: {trues}/{n} true");
}

#[test]
fn enable_wedge_interintra_search_matches_c() {
    let mut rng = Rng(0x5EED_0C03);
    let (mut trues, mut n) = (0usize, 0usize);
    for _ in 0..3000 {
        let var = rng.next() as u32;
        let thresh = if rng.next() % 2 == 0 {
            var.wrapping_add((rng.next() % 3) as u32).wrapping_sub(1)
        } else {
            rng.next() as u32
        };
        let en = rng.next() % 2 == 0;
        let got = enable_wedge_interintra_search(var, thresh, en);
        let want = cref::ref_ct_enable_wedge_interintra_search(var, thresh, en);
        assert_eq!(got, want, "interintra(var={var}, thresh={thresh}, en={en})");
        trues += usize::from(want);
        n += 1;
    }
    assert!(trues > 0 && trues < n, "vacuous: {trues}/{n} true");
}

#[test]
fn compute_valid_comp_types_matches_c() {
    let mut rng = Rng(0x5EED_0C04);
    let mut seen_counts = [0usize; COMPOUND_TYPES + 1];
    // Every bsize, every mode_search_mask, and the knob cross-product; the
    // random draw only picks the source variance / threshold pair that decides
    // the wedge gate.
    for bsize in 0..BLOCK_SIZES_ALL {
        for mask in 0..16u32 {
            for masked_used in [false, true] {
                for dwc in [0i32, 1, 2] {
                    for (use_dwc, use_dwc_flag) in [
                        (0i32, DistWtdCompFlag::Enabled),
                        (1, DistWtdCompFlag::SkipMvSearch),
                        (2, DistWtdCompFlag::Disabled),
                    ] {
                        for diff_wtd in [false, true] {
                            let var = rng.next() as u32 % 4096;
                            let thresh = rng.next() as u32 % 4096;
                            let en_wedge = rng.next() % 2 == 0;
                            let cfg = ValidCompTypeCfg {
                                enable_dist_wtd_comp: dwc == 1,
                                // The raw C integer goes to the oracle and the
                                // port's enum to the port, so the enum's
                                // discriminants are what is under test.
                                use_dist_wtd_comp_flag: use_dwc_flag,
                                enable_interinter_wedge: enable_wedge_interinter_search(
                                    var, thresh, en_wedge,
                                ),
                                enable_diff_wtd_comp: diff_wtd,
                            };
                            let got: Vec<i32> =
                                compute_valid_comp_types(bsize, masked_used, mask, &cfg)
                                    .iter()
                                    .map(|t| t.index() as i32)
                                    .collect();
                            let want = cref::ref_ct_compute_valid_comp_types(
                                bsize as i32,
                                masked_used,
                                mask as i32,
                                var,
                                thresh,
                                en_wedge,
                                dwc,
                                use_dwc,
                                diff_wtd,
                            );
                            assert_eq!(
                                got, want,
                                "compute_valid_comp_types(bsize={bsize}, mask={mask}, \
                                 masked_used={masked_used}, dwc={dwc}, use_dwc={use_dwc}, \
                                 diff_wtd={diff_wtd}, var={var}, thresh={thresh})"
                            );
                            seen_counts[got.len()] += 1;
                        }
                    }
                }
            }
        }
    }
    // Non-vacuity: the sweep must have produced every possible count, not just
    // "empty" (which a port returning `vec![]` would also pass).
    for (count, hits) in seen_counts.iter().enumerate() {
        assert!(*hits > 0, "no case produced {count} valid compound types");
    }
}

#[test]
fn calc_masked_type_cost_matches_c() {
    let mut rng = Rng(0x5EED_0C05);
    for _ in 0..4000 {
        let g = [rng.cost(), rng.cost()];
        let i = [rng.cost(), rng.cost()];
        let c = [rng.cost(), rng.cost()];
        for masked_used in [false, true] {
            let got = calc_masked_type_cost(g, i, c, masked_used);
            let want = cref::ref_ct_calc_masked_type_cost(g, i, c, masked_used);
            assert_eq!(
                got, want,
                "calc_masked_type_cost(g={g:?}, i={i:?}, c={c:?}, m={masked_used})"
            );
        }
    }
}

#[test]
fn update_mbmi_for_compound_type_matches_c() {
    for ty in CompoundType::ALL {
        let (want_ty, want_group, want_idx) =
            cref::ref_ct_update_mbmi_for_compound_type(ty.index() as i32);
        assert_eq!(want_ty, ty.index() as i32, "the shim round-trips the type");
        assert_eq!(
            i32::from(ty.comp_group_idx()),
            want_group,
            "comp_group_idx for {ty:?}"
        );
        assert_eq!(
            i32::from(ty.compound_idx()),
            want_idx,
            "compound_idx for {ty:?}"
        );
    }
    // Non-vacuity: both derived flags must actually take both values.
    assert!(CompoundType::ALL.iter().any(|t| t.comp_group_idx()));
    assert!(CompoundType::ALL.iter().any(|t| !t.comp_group_idx()));
    assert!(CompoundType::ALL.iter().any(|t| t.compound_idx()));
    assert!(CompoundType::ALL.iter().any(|t| !t.compound_idx()));
}

#[test]
fn get_interinter_compound_mask_rate_matches_c() {
    let mut rng = Rng(0x5EED_0C06);
    let mut wedge_bsizes = 0usize;
    for bsize in 0..BLOCK_SIZES_ALL {
        let costs: Vec<i32> = (0..MAX_WEDGE_TYPES).map(|_| rng.cost()).collect();
        for ty in [CompoundType::Wedge, CompoundType::DiffWtd] {
            for widx in 0..MAX_WEDGE_TYPES {
                let got = get_interinter_compound_mask_rate(ty, bsize, widx, &costs);
                let want = cref::ref_ct_get_interinter_compound_mask_rate(
                    ty.index() as i32,
                    bsize as i32,
                    widx as i32,
                    &costs,
                );
                assert_eq!(
                    got, want,
                    "mask_rate(ty={ty:?}, bsize={bsize}, wedge_index={widx})"
                );
            }
        }
        // Count the bsizes where the wedge arm is live, so the sweep is known
        // to have exercised BOTH sides of `av1_is_wedge_used`.
        if get_interinter_compound_mask_rate(CompoundType::Wedge, bsize, 0, &costs) != 0 {
            wedge_bsizes += 1;
        }
    }
    assert!(
        wedge_bsizes > 0 && wedge_bsizes < BLOCK_SIZES_ALL,
        "vacuous: {wedge_bsizes}/{BLOCK_SIZES_ALL} bsizes took the wedge arm"
    );
}

#[test]
fn save_mask_search_results_matches_c() {
    let (mut trues, mut n) = (0usize, 0usize);
    for mode in 0..MB_MODE_COUNT {
        for reuse in [false, true] {
            let got = save_mask_search_results(mode == NEW_NEWMV, reuse);
            let want = cref::ref_ct_save_mask_search_results(mode, reuse);
            assert_eq!(
                got, want,
                "save_mask_search_results(mode={mode}, reuse={reuse})"
            );
            trues += usize::from(want);
            n += 1;
        }
    }
    assert!(trues > 0 && trues < n, "vacuous: {trues}/{n} true");
}

#[test]
fn push_comp_avg_est_rd_matches_c() {
    let mut rng = Rng(0x5EED_0C07);
    for level in 0..=3usize {
        for _ in 0..2000 {
            // Start from a list that is sometimes fully unset (INT64_MAX),
            // sometimes partly filled and sorted, sometimes filled at random —
            // C's insertion scan behaves differently in each.
            let mut top = [i64::MAX; TOP_COMP_AVG_EST_RD_COUNT];
            let fill = (rng.next() % (TOP_COMP_AVG_EST_RD_COUNT as u64 + 1)) as usize;
            let mut vals: Vec<i64> = (0..fill)
                .map(|_| i64::from(rng.range(0, 1 << 20)))
                .collect();
            vals.sort_unstable();
            top[..fill].copy_from_slice(&vals);
            let mut want = top;
            // As above: land exactly on an existing entry sometimes, so the
            // insertion scan's strict `<` is exercised at its boundary.
            let tmp_rd = if fill > 0 && rng.next() % 4 == 0 {
                top[(rng.next() as usize) % fill]
            } else {
                i64::from(rng.range(-16, 1 << 20))
            };

            push_comp_avg_est_rd(&mut top, tmp_rd, level);
            cref::ref_ct_push_comp_avg_est_rd(&mut want, tmp_rd, level as i32);
            assert_eq!(
                top, want,
                "push_comp_avg_est_rd(level={level}, tmp_rd={tmp_rd})"
            );
        }
    }
}

#[test]
fn prune_comp_eval_using_comp_avg_est_rd_matches_c() {
    let mut rng = Rng(0x5EED_0C08);
    let (mut trues, mut n) = (0usize, 0usize);
    for level in 0..=3usize {
        for _ in 0..3000 {
            let mut top = [i64::MAX; TOP_COMP_AVG_EST_RD_COUNT];
            let fill = (rng.next() % (TOP_COMP_AVG_EST_RD_COUNT as u64 + 1)) as usize;
            let mut vals: Vec<i64> = (0..fill)
                .map(|_| i64::from(rng.range(0, 1 << 20)))
                .collect();
            vals.sort_unstable();
            top[..fill].copy_from_slice(&vals);
            // One draw in four sits EXACTLY on the last kept candidate: the
            // comparison is a strict `>`, and with random values only, a
            // `>=` transcription passes (measured — that perturbation was
            // inert until this case was added).
            let tmp_rd = if fill > 0 && rng.next() % 4 == 0 {
                top[(rng.next() as usize) % fill]
            } else {
                i64::from(rng.range(0, 1 << 20))
            };
            // Exercise the `ref_best_rd == INT64_MAX` early-out too.
            let ref_best_rd = if rng.next() % 4 == 0 {
                i64::MAX
            } else {
                i64::from(rng.range(0, 1 << 20))
            };
            let got = prune_comp_eval_using_comp_avg_est_rd(&top, tmp_rd, ref_best_rd, level);
            let want = cref::ref_ct_prune_comp_eval_using_comp_avg_est_rd(
                &top,
                tmp_rd,
                ref_best_rd,
                level as i32,
            );
            assert_eq!(
                got, want,
                "prune(level={level}, tmp_rd={tmp_rd}, ref_best_rd={ref_best_rd}, top={top:?})"
            );
            trues += usize::from(want);
            n += 1;
        }
    }
    assert!(trues > 0 && trues < n, "vacuous: {trues}/{n} true");
}

#[test]
fn compute_rd_thresh_matches_c() {
    let mut rng = Rng(0x5EED_0C09);
    let mut negatives = 0usize;
    let mut saturated = 0usize;
    let mut cases: Vec<(i32, i32, i64)> = vec![
        // The overflow boundary of `get_rd_thresh_from_best_rd`: the guard is
        // `ref_best_rd < 9 * (INT64_MAX / 16)`, so straddle that exactly.
        (100, 0, 9 * (i64::MAX / 16) - 1),
        (100, 0, 9 * (i64::MAX / 16)),
        (100, 0, 9 * (i64::MAX / 16) + 1),
        (100, 0, i64::MAX),
        (1, 0, 0),
    ];
    for _ in 0..5000 {
        let rdmult = rng.range(1, 1 << 16);
        let rate = rng.range(0, 1 << 18);
        let ref_best_rd = if rng.next() % 8 == 0 {
            i64::MAX
        } else {
            i64::from(rng.range(0, 1 << 24)) << (rng.next() % 24)
        };
        cases.push((rdmult, rate, ref_best_rd));
    }
    for (rdmult, rate, ref_best_rd) in cases {
        let got = compute_rd_thresh(rdmult, rate, ref_best_rd);
        let want = cref::ref_ct_compute_rd_thresh(rdmult, rate, ref_best_rd);
        assert_eq!(
            got, want,
            "compute_rd_thresh(rdmult={rdmult}, rate={rate}, ref_best_rd={ref_best_rd})"
        );
        if got < 0 {
            negatives += 1;
        }
        if ref_best_rd == i64::MAX {
            saturated += 1;
        }
    }
    // Both branches of `get_rd_thresh_from_best_rd` and the negative-result
    // case must all have been reached, or the test proves less than it looks.
    assert!(negatives > 0, "no case produced a negative threshold");
    assert!(saturated > 0, "the INT64_MAX arm was never reached");
}

// ===================================================================
// The mask picks (compound_type.c:126-428).
//
// | test | C function |
// |---|---|
// | `estimate_wedge_sign_matches_c` | `estimate_wedge_sign` `:126` |
// | `pick_wedge_matches_c` | `pick_wedge` `:189` |
// | `pick_wedge_fixed_sign_matches_c` | `pick_wedge_fixed_sign` `:257` |
// | `pick_interinter_wedge_matches_c` | `pick_interinter_wedge` `:299` |
// | `pick_interinter_seg_matches_c` | `pick_interinter_seg` `:332` |
// | `pick_interintra_wedge_matches_c` | `pick_interintra_wedge` `:394` |
// ===================================================================

use aom_dsp::inter::compound::DiffwtdMaskType;
use aom_encode::compound_type::{
    MaskSearchCtx, Pixels, estimate_wedge_sign, pick_interinter_seg, pick_interinter_wedge,
    pick_interintra_wedge, pick_wedge, pick_wedge_fixed_sign,
};
use aom_sys_ref::RefPixels;

/// `block_size_wide` / `block_size_high`.
const BLK_W: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

/// The nine block sizes with a wedge codebook — every bsize any of these picks
/// can legally be called at (`av1_wedge_params_lookup`, reconinter.c:236-268).
const WEDGE_BSIZES: [usize; 9] = [3, 4, 5, 6, 7, 8, 9, 18, 19];

/// A pixel plane plus the two views the port and the oracle each want.
struct Plane {
    lo: Vec<u8>,
    hi: Vec<u16>,
    hbd: bool,
}

impl Plane {
    /// `bd == 8` produces the lowbd (`u8`) buffer, higher depths the `u16`
    /// one — the two arms of `is_cur_buf_hbd`.
    fn random(rng: &mut Rng, n: usize, bd: u8) -> Self {
        let maxv = (1u32 << bd) - 1;
        if bd == 8 {
            let lo: Vec<u8> = (0..n).map(|_| (rng.next() % 256) as u8).collect();
            Plane {
                lo,
                hi: Vec::new(),
                hbd: false,
            }
        } else {
            let hi: Vec<u16> = (0..n)
                .map(|_| (rng.next() as u32 % (maxv + 1)) as u16)
                .collect();
            Plane {
                lo: Vec::new(),
                hi,
                hbd: true,
            }
        }
    }
    fn port(&self) -> Pixels<'_> {
        if self.hbd {
            Pixels::High(&self.hi)
        } else {
            Pixels::Low(&self.lo)
        }
    }
    fn cref(&self) -> RefPixels<'_> {
        if self.hbd {
            RefPixels::High(&self.hi)
        } else {
            RefPixels::Low(&self.lo)
        }
    }
}

/// Residual pair with the magnitudes the encoder can actually produce.
///
/// `residual1 = src - pred1` and `diff10 = pred1 - pred0` are both differences
/// of two same-bit-depth pixels, so at `bd` they live in
/// `-(2^bd - 1) ..= 2^bd - 1` and NOTHING wider is reachable. Bounding them
/// that way is the §5-of-the-brief discipline: `av1_wedge_sse_from_residuals`
/// accumulates `(m*d + (64-m)*r)` through `_mm_madd_epi16` on x86, so a
/// generator that drew the full `i16` range would report a divergence that the
/// producer cannot create.
fn residuals(rng: &mut Rng, n: usize, bd: u8) -> (Vec<i16>, Vec<i16>) {
    let lim = (1i32 << bd) - 1;
    let mut r1 = vec![0i16; n];
    let mut d10 = vec![0i16; n];
    for i in 0..n {
        r1[i] = rng.range(-lim, lim + 1) as i16;
        d10[i] = rng.range(-lim, lim + 1) as i16;
    }
    (r1, d10)
}

/// The context both sides use, plus the cost row.
fn ctx_for<'a>(
    bsize: usize,
    bd: u8,
    rdmult: i32,
    dequant_ac: i32,
    wedge_idx_cost: &'a [i32],
) -> MaskSearchCtx<'a> {
    MaskSearchCtx {
        bsize,
        bd,
        rdmult,
        dequant_ac,
        wedge_idx_cost,
    }
}

/// Every (bsize, bd) cell the picks are reachable at. `bd == 8` is the lowbd
/// buffer world; 8/10/12 with `hbd` are the three high-bit-depth arms — note
/// **hbd at bd 8** is a real encoder configuration and takes a different code
/// path (`CONVERT_TO_SHORTPTR`) with the SAME `bd_round == 0`.
fn cells() -> Vec<(usize, u8)> {
    let mut v = Vec::new();
    for b in WEDGE_BSIZES {
        for bd in [8u8, 10, 12] {
            v.push((b, bd));
        }
    }
    v
}

#[test]
fn estimate_wedge_sign_matches_c() {
    let mut rng = Rng(0x5EED_0C10);
    let (mut trues, mut n_cases) = (0usize, 0usize);
    for (bsize, bd) in cells() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        for case in 0..9 {
            // A source stride WIDER than the block: C reads src at
            // `p->src.stride`, and a port that assumed `bw` would pass at
            // stride == bw and fail here.
            let src_stride = bw + 8;
            let src = Plane::random(&mut rng, src_stride * bh, bd);
            let p0 = Plane::random(&mut rng, bw * bh, bd);
            // One case in nine gives the two predictors IDENTICAL content —
            // a state the encoder reaches whenever both references agree.
            // Then every quadrant SSE cancels and `tl + br` is exactly 0, the
            // only input that separates C's `> 0` from `>= 0`. (Measured: with
            // random predictors only, that perturbation was inert.)
            let p1 = if case == 8 {
                Plane {
                    lo: p0.lo.clone(),
                    hi: p0.hi.clone(),
                    hbd: p0.hbd,
                }
            } else {
                Plane::random(&mut rng, bw * bh, bd)
            };
            let got = estimate_wedge_sign(
                bsize,
                bd,
                src.port(),
                src_stride,
                p0.port(),
                bw,
                p1.port(),
                bw,
            );
            let want = cref::ref_ct_estimate_wedge_sign(
                bsize as i32,
                i32::from(bd),
                src.cref(),
                src_stride as i32,
                p0.cref(),
                bw as i32,
                p1.cref(),
                bw as i32,
            );
            assert_eq!(got, want, "estimate_wedge_sign(bsize={bsize}, bd={bd})");
            trues += usize::from(want);
            n_cases += 1;
        }
    }
    assert!(
        trues > 0 && trues < n_cases,
        "vacuous: {trues}/{n_cases} true"
    );
}

#[test]
fn pick_wedge_fixed_sign_matches_c() {
    let mut rng = Rng(0x5EED_0C11);
    let mut distinct_indices = std::collections::BTreeSet::new();
    for (bsize, bd) in cells() {
        let n = BLK_W[bsize] * BLK_H[bsize];
        for sign in 0..2usize {
            for _ in 0..4 {
                let costs: Vec<i32> = (0..MAX_WEDGE_TYPES).map(|_| rng.cost()).collect();
                let (r1, d10) = residuals(&mut rng, n, bd);
                let rdmult = rng.range(1, 1 << 14);
                let dequant_ac = rng.range(4, 1 << 11);
                let hbd = bd > 8;
                let ctx = ctx_for(bsize, bd, rdmult, dequant_ac, &costs);
                let got = pick_wedge_fixed_sign(&ctx, hbd, &r1, &d10, sign);
                let (rd, index, sse) = cref::ref_ct_pick_wedge_fixed_sign(
                    bsize as i32,
                    hbd,
                    i32::from(bd),
                    rdmult,
                    dequant_ac,
                    &costs,
                    &r1,
                    &d10,
                    sign as i32,
                );
                assert_eq!(
                    (got.rd, got.index as i32, got.sse),
                    (rd, index, sse),
                    "pick_wedge_fixed_sign(bsize={bsize}, bd={bd}, sign={sign})"
                );
                distinct_indices.insert(index);
            }
        }
    }
    // Non-vacuity: a port that always answered index 0 would agree on any cell
    // where C also picked 0.
    assert!(
        distinct_indices.len() > 4,
        "vacuous: only {} distinct winning indices",
        distinct_indices.len()
    );
}

#[test]
fn pick_wedge_matches_c() {
    let mut rng = Rng(0x5EED_0C12);
    let mut signs = [0usize; 2];
    let mut distinct_indices = std::collections::BTreeSet::new();
    for (bsize, bd) in cells() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let n = bw * bh;
        for iter in 0..48 {
            let src_stride = bw + 8;
            let src = Plane::random(&mut rng, src_stride * bh, bd);
            let p0 = Plane::random(&mut rng, n, bd);
            let (r1, d10) = residuals(&mut rng, n, bd);
            let costs: Vec<i32> = (0..MAX_WEDGE_TYPES).map(|_| rng.cost()).collect();
            let rdmult = rng.range(1, 1 << 14);
            let dequant_ac = rng.range(4, 1 << 11);
            let ctx = ctx_for(bsize, bd, rdmult, dequant_ac, &costs);
            let got = pick_wedge(&ctx, src.port(), src_stride, p0.port(), &r1, &d10);
            let (rd, sign, index, sse) = cref::ref_ct_pick_wedge(
                bsize as i32,
                i32::from(bd),
                rdmult,
                dequant_ac,
                &costs,
                src.cref(),
                src_stride as i32,
                p0.cref(),
                &r1,
                &d10,
            );
            assert_eq!(
                (got.rd, got.sign as i32, got.index as i32, got.sse),
                (rd, sign, index, sse),
                "pick_wedge(bsize={bsize}, bd={bd})"
            );
            signs[sign as usize] += 1;
            distinct_indices.insert(index);
        }
    }
    assert!(
        signs[0] > 0 && signs[1] > 0,
        "vacuous: the sign search never produced both values ({signs:?})"
    );
    assert!(
        distinct_indices.len() > 4,
        "vacuous: {} indices",
        distinct_indices.len()
    );
}

#[test]
fn pick_interinter_wedge_matches_c() {
    let mut rng = Rng(0x5EED_0C13);
    let mut fast_signs = [0usize; 2];
    for (bsize, bd) in cells() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let n = bw * bh;
        for fast in [false, true] {
            for _ in 0..4 {
                let src_stride = bw + 8;
                let src = Plane::random(&mut rng, src_stride * bh, bd);
                let p0 = Plane::random(&mut rng, n, bd);
                let p1 = Plane::random(&mut rng, n, bd);
                let (r1, d10) = residuals(&mut rng, n, bd);
                let costs: Vec<i32> = (0..MAX_WEDGE_TYPES).map(|_| rng.cost()).collect();
                let rdmult = rng.range(1, 1 << 14);
                let dequant_ac = rng.range(4, 1 << 11);
                let ctx = ctx_for(bsize, bd, rdmult, dequant_ac, &costs);
                let got = pick_interinter_wedge(
                    &ctx,
                    fast,
                    src.port(),
                    src_stride,
                    p0.port(),
                    p1.port(),
                    &r1,
                    &d10,
                );
                let (rd, sign, index, sse) = cref::ref_ct_pick_interinter_wedge(
                    bsize as i32,
                    i32::from(bd),
                    rdmult,
                    dequant_ac,
                    &costs,
                    fast,
                    src.cref(),
                    src_stride as i32,
                    p0.cref(),
                    p1.cref(),
                    &r1,
                    &d10,
                );
                assert_eq!(
                    (got.rd, got.sign as i32, got.index as i32, got.sse),
                    (rd, sign, index, sse),
                    "pick_interinter_wedge(bsize={bsize}, bd={bd}, fast={fast})"
                );
                if fast {
                    fast_signs[sign as usize] += 1;
                }
            }
        }
    }
    assert!(
        fast_signs[0] > 0 && fast_signs[1] > 0,
        "vacuous: the fast sign estimate never produced both values ({fast_signs:?})"
    );
}

#[test]
fn pick_interinter_seg_matches_c() {
    let mut rng = Rng(0x5EED_0C14);
    let mut types = [0usize; 2];
    // Not restricted to wedge bsizes: COMPOUND_DIFFWTD is legal at every
    // compound-capable size (`is_interinter_compound_used`), so sweep those.
    let bsizes: Vec<usize> = (0..22).filter(|&b| BLK_W[b].min(BLK_H[b]) >= 8).collect();
    for bsize in bsizes {
        for bd in [8u8, 10, 12] {
            let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
            let n = bw * bh;
            for case in 0..4 {
                let p0 = Plane::random(&mut rng, n, bd);
                // The last case makes the two predictors identical, so
                // `diff10 = pred1 - pred0` is all zero. `av1_wedge_sse_from_
                // residuals` is then `64 * r1[i]` with the mask term gone, so
                // BOTH mask types score identically and C's strict `<` keeps
                // the first. That tie is the only input separating `<` from
                // `<=`, and it is exactly what two agreeing references give.
                let p1 = if case == 3 {
                    Plane {
                        lo: p0.lo.clone(),
                        hi: p0.hi.clone(),
                        hbd: p0.hbd,
                    }
                } else {
                    Plane::random(&mut rng, n, bd)
                };
                let (r1, mut d10) = residuals(&mut rng, n, bd);
                if case == 3 {
                    d10.iter_mut().for_each(|v| *v = 0);
                }
                let rdmult = rng.range(1, 1 << 14);
                let dequant_ac = rng.range(4, 1 << 11);
                let costs = [0i32; MAX_WEDGE_TYPES];
                let ctx = ctx_for(bsize, bd, rdmult, dequant_ac, &costs);
                let got = pick_interinter_seg(&ctx, p0.port(), p1.port(), &r1, &d10);
                let (rd, mask_type, sse, seg) = cref::ref_ct_pick_interinter_seg(
                    bsize as i32,
                    i32::from(bd),
                    rdmult,
                    dequant_ac,
                    p0.cref(),
                    p1.cref(),
                    &r1,
                    &d10,
                    n,
                );
                let got_type = match got.mask_type {
                    DiffwtdMaskType::Diffwtd38 => 0,
                    DiffwtdMaskType::Diffwtd38Inv => 1,
                };
                assert_eq!(
                    (got.rd, got_type, got.sse),
                    (rd, mask_type, sse),
                    "pick_interinter_seg(bsize={bsize}, bd={bd})"
                );
                assert_eq!(
                    got.seg_mask, seg,
                    "pick_interinter_seg seg_mask (bsize={bsize}, bd={bd})"
                );
                types[mask_type as usize] += 1;
            }
        }
    }
    assert!(
        types[0] > 0 && types[1] > 0,
        "vacuous: only one mask type ever won ({types:?})"
    );
}

#[test]
fn pick_interintra_wedge_matches_c() {
    let mut rng = Rng(0x5EED_0C15);
    let mut distinct_indices = std::collections::BTreeSet::new();
    for (bsize, bd) in cells() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let n = bw * bh;
        for _ in 0..6 {
            let src_stride = bw + 8;
            let src = Plane::random(&mut rng, src_stride * bh, bd);
            let p0 = Plane::random(&mut rng, n, bd);
            let p1 = Plane::random(&mut rng, n, bd);
            let costs: Vec<i32> = (0..MAX_WEDGE_TYPES).map(|_| rng.cost()).collect();
            let rdmult = rng.range(1, 1 << 14);
            let dequant_ac = rng.range(4, 1 << 11);
            let ctx = ctx_for(bsize, bd, rdmult, dequant_ac, &costs);
            let got = pick_interintra_wedge(&ctx, src.port(), src_stride, p0.port(), p1.port());
            let (rd, index) = cref::ref_ct_pick_interintra_wedge(
                bsize as i32,
                i32::from(bd),
                rdmult,
                dequant_ac,
                &costs,
                src.cref(),
                src_stride as i32,
                p0.cref(),
                p1.cref(),
            );
            assert_eq!(
                (got.rd, got.index as i32),
                (rd, index),
                "pick_interintra_wedge(bsize={bsize}, bd={bd})"
            );
            assert_eq!(got.sign, 0, "interintra codes no wedge sign");
            distinct_indices.insert(index);
        }
    }
    assert!(
        distinct_indices.len() > 4,
        "vacuous: {} indices",
        distinct_indices.len()
    );
}

// ===================================================================
// The compound-RD reuse cache (compound_type.c:32-101, :955-1057).
//
// | test | C function |
// |---|---|
// | `is_comp_rd_match_matches_c` | `is_comp_rd_match` `:32` |
// | `find_comp_rd_in_stats_matches_c` | `find_comp_rd_in_stats` `:85` |
// | `save_comp_rd_search_stat_matches_c` | `save_comp_rd_search_stat` `:997` |
// | `backup_stats_matches_c` | `backup_stats` `:1044` |
// | `update_best_info_matches_c` | `update_best_info` `:1005` |
// | `update_mask_best_mv_matches_c` | `update_mask_best_mv` `:1016` |
// | `populate_reuse_comp_type_data_matches_c` | `populate_reuse_comp_type_data` `:962` |
// ===================================================================

use aom_encode::compound_type::{
    BestCompTypeStats, CompRdBlock, CompRdReuseCfg, CompRdStats, CompTypeCosts, InterInterComp,
    MAX_COMP_RD_STATS, TransformationType, find_comp_rd_in_stats, is_comp_rd_match,
    populate_reuse_comp_type_data, save_comp_rd_search_stat, update_mask_best_mv,
};
use aom_encode::rdopt_mv::{Mv, PredMode};

const WMTYPES: [TransformationType; 4] = [
    TransformationType::Identity,
    TransformationType::Translation,
    TransformationType::RotZoom,
    TransformationType::Affine,
];

/// Pack a [`CompTypeCosts`] the way the shim expects: `rate ++ model_rate ++
/// rs2` and `dist ++ model_dist`.
fn pack_costs(c: &CompTypeCosts) -> ([i32; 12], [i64; 8]) {
    let mut a = [0i32; 12];
    let mut b = [0i64; 8];
    for i in 0..COMPOUND_TYPES {
        a[i] = c.rate[i];
        a[COMPOUND_TYPES + i] = c.model_rate[i];
        a[2 * COMPOUND_TYPES + i] = c.rs2[i];
        b[i] = c.dist[i];
        b[COMPOUND_TYPES + i] = c.model_dist[i];
    }
    (a, b)
}

fn unpack_costs(a: &[i32; 12], b: &[i64; 8]) -> CompTypeCosts {
    let mut c = CompTypeCosts::default();
    for i in 0..COMPOUND_TYPES {
        c.rate[i] = a[i];
        c.model_rate[i] = a[COMPOUND_TYPES + i];
        c.rs2[i] = a[2 * COMPOUND_TYPES + i];
        c.dist[i] = b[i];
        c.model_dist[i] = b[COMPOUND_TYPES + i];
    }
    c
}

/// A cost set with values in every slot (never a sentinel), so a reuse that
/// copies the wrong slot is visible.
fn rand_costs(rng: &mut Rng) -> CompTypeCosts {
    let mut c = CompTypeCosts::default();
    for i in 0..COMPOUND_TYPES {
        c.rate[i] = rng.range(0, 1 << 16);
        c.model_rate[i] = rng.range(0, 1 << 16);
        c.rs2[i] = rng.range(0, 1 << 14);
        c.dist[i] = i64::from(rng.range(0, 1 << 20));
        c.model_dist[i] = i64::from(rng.range(0, 1 << 20));
    }
    c
}

fn rand_mv(rng: &mut Rng) -> Mv {
    Mv {
        row: rng.range(-64, 64) as i16,
        col: rng.range(-64, 64) as i16,
    }
}

fn pack_stats(st: &CompRdStats) -> ([i32; 12], [i64; 8], [i16; 4], [i32; 11]) {
    let (a, b) = pack_costs(&st.costs);
    let mv = [st.mv[0].row, st.mv[0].col, st.mv[1].row, st.mv[1].col];
    let meta = [
        i32::from(st.ref_frames[0]),
        i32::from(st.ref_frames[1]),
        st.mode.to_i32(),
        st.filter as i32,
        st.ref_mv_idx,
        i32::from(st.is_global[0]),
        i32::from(st.is_global[1]),
        st.interinter_comp.wedge_index as i32,
        st.interinter_comp.wedge_sign as i32,
        match st.interinter_comp.mask_type {
            DiffwtdMaskType::Diffwtd38 => 0,
            DiffwtdMaskType::Diffwtd38Inv => 1,
        },
        st.interinter_comp.ty.index() as i32,
    ];
    (a, b, mv, meta)
}

fn pack_mi(mi: &CompRdBlock) -> ([i16; 4], [i32; 7]) {
    let mv = [mi.mv[0].row, mi.mv[0].col, mi.mv[1].row, mi.mv[1].col];
    let meta = [
        i32::from(mi.ref_frames[0]),
        i32::from(mi.ref_frames[1]),
        mi.mode.to_i32(),
        mi.filter as i32,
        mi.bsize as i32,
        mi.wmtype[0] as i32,
        mi.wmtype[1] as i32,
    ];
    (mv, meta)
}

/// A (cached entry, current block) pair that agrees on `n_agree` of the four
/// things `is_comp_rd_match` compares, so both the match and every distinct
/// mismatch are reachable.
#[allow(clippy::type_complexity)]
fn rand_pair(rng: &mut Rng) -> (CompRdStats, CompRdBlock) {
    // Compound blocks always have two real references; bsize is drawn from
    // both sides of `is_global_mv_block`'s `min(w,h) >= 8` gate.
    let rf = [rng.range(1, 5) as i8, rng.range(5, 8) as i8];
    let mv = [rand_mv(rng), rand_mv(rng)];
    let filter = rng.range(0, 3) as u32;
    let bsize = rng.range(0, 22) as usize;
    let mode = PredMode::from_i32(rng.range(13, MB_MODE_COUNT)).expect("inter mode");
    let wmtype = [
        WMTYPES[(rng.next() % 4) as usize],
        WMTYPES[(rng.next() % 4) as usize],
    ];
    let mi = CompRdBlock {
        filter,
        ref_frames: rf,
        mv,
        mode,
        bsize,
        wmtype,
    };

    // The stored entry starts as an exact match and is then perturbed in one
    // of the five ways the comparison can fail (or not at all).
    let mut st = CompRdStats {
        costs: rand_costs(rng),
        mv,
        ref_frames: rf,
        mode: PredMode::from_i32(rng.range(13, MB_MODE_COUNT)).expect("inter mode"),
        filter,
        ref_mv_idx: rng.range(0, 3),
        is_global: [
            aom_encode::compound_type::is_global_mv_block(mode, bsize, wmtype[0]),
            aom_encode::compound_type::is_global_mv_block(mode, bsize, wmtype[1]),
        ],
        interinter_comp: InterInterComp::default(),
    };
    match rng.next() % 6 {
        0 => st.filter = st.filter.wrapping_add(1),
        1 => st.ref_frames[0] = st.ref_frames[0].wrapping_add(1),
        2 => st.mv[1].row = st.mv[1].row.wrapping_add(1),
        3 => st.is_global[0] = !st.is_global[0],
        4 => st.is_global[1] = !st.is_global[1],
        _ => {}
    }
    (st, mi)
}

#[test]
fn is_comp_rd_match_matches_c() {
    let mut rng = Rng(0x5EED_0C20);
    let (mut matches, mut n) = (0usize, 0usize);
    for _ in 0..4000 {
        let (st, mi) = rand_pair(&mut rng);
        for dis_wedge in [false, true] {
            for fast in [false, true] {
                let cfg = CompRdReuseCfg {
                    disable_interinter_wedge_newmv_search: dis_wedge,
                    enable_fast_compound_mode_search: fast,
                };
                // Both sides start from the SAME partly-filled cost arrays, so
                // a reuse that copies too much or too little shows up.
                let seed = rand_costs(&mut rng);
                let mut port_out = seed;
                let (mut c_i32, mut c_i64) = pack_costs(&seed);
                let (st_i32, st_i64, st_mv, st_meta) = pack_stats(&st);
                let (mi_mv, mi_meta) = pack_mi(&mi);

                let got = is_comp_rd_match(&cfg, &st, &mi, &mut port_out);
                let want = cref::ref_ct_is_comp_rd_match(
                    dis_wedge, fast, &st_i32, &st_i64, &st_mv, &st_meta, &mi_mv, &mi_meta,
                    &mut c_i32, &mut c_i64,
                );
                assert_eq!(got, want, "is_comp_rd_match verdict");
                assert_eq!(
                    port_out,
                    unpack_costs(&c_i32, &c_i64),
                    "is_comp_rd_match reuse mask (dis_wedge={dis_wedge}, fast={fast})"
                );
                matches += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(matches > 0 && matches < n, "vacuous: {matches}/{n} matched");
}

#[test]
fn find_comp_rd_in_stats_matches_c() {
    let mut rng = Rng(0x5EED_0C21);
    let (mut hits, mut n) = (0usize, 0usize);
    for _ in 0..600 {
        let cfg = CompRdReuseCfg {
            disable_interinter_wedge_newmv_search: rng.next() % 2 == 0,
            enable_fast_compound_mode_search: rng.next() % 2 == 0,
        };
        let (probe_st, mi) = rand_pair(&mut rng);
        // A short cache with the probe entry somewhere in it, so the FIRST
        // match rather than any match is what the position test checks.
        let len = (rng.next() % 5) as usize;
        let mut stats: Vec<CompRdStats> = (0..len).map(|_| rand_pair(&mut rng).0).collect();
        let at = if len == 0 {
            0
        } else {
            (rng.next() as usize) % (len + 1)
        };
        stats.insert(at.min(stats.len()), probe_st);

        let seed = rand_costs(&mut rng);
        let mut port_out = seed;
        let got = find_comp_rd_in_stats(&cfg, &stats, &mi, &mut port_out);

        // The oracle has no whole-cache entry point (`find_comp_rd_in_stats`
        // walks `x->comp_rd_stats`, which a shim would have to fill entry by
        // entry anyway), so the reference scan is `is_comp_rd_match` — the
        // REAL C function — applied in the same order.
        let mut want: Option<usize> = None;
        let mut want_costs = seed;
        for (j, st) in stats.iter().enumerate() {
            let (a, b) = pack_costs(&want_costs);
            let (mut c_i32, mut c_i64) = (a, b);
            let (st_i32, st_i64, st_mv, st_meta) = pack_stats(st);
            let (mi_mv, mi_meta) = pack_mi(&mi);
            let m = cref::ref_ct_is_comp_rd_match(
                cfg.disable_interinter_wedge_newmv_search,
                cfg.enable_fast_compound_mode_search,
                &st_i32,
                &st_i64,
                &st_mv,
                &st_meta,
                &mi_mv,
                &mi_meta,
                &mut c_i32,
                &mut c_i64,
            );
            want_costs = unpack_costs(&c_i32, &c_i64);
            if m {
                want = Some(j);
                break;
            }
        }
        assert_eq!(got, want, "find_comp_rd_in_stats index");
        assert_eq!(port_out, want_costs, "find_comp_rd_in_stats costs");
        hits += usize::from(want.is_some());
        n += 1;
    }
    assert!(hits > 0 && hits < n, "vacuous: {hits}/{n} found a match");
}

#[test]
fn save_comp_rd_search_stat_matches_c() {
    let mut rng = Rng(0x5EED_0C22);
    assert_eq!(cref::ref_ct_max_comp_rd_stats() as usize, MAX_COMP_RD_STATS);
    let mut dropped = 0usize;
    for start_idx in [0i32, 1, 17, 62, 63, 64, 65] {
        for _ in 0..40 {
            let (probe, mi) = rand_pair(&mut rng);
            let costs = rand_costs(&mut rng);
            let mv = [rand_mv(&mut rng), rand_mv(&mut rng)];
            let comp = InterInterComp {
                wedge_index: (rng.next() % 16) as usize,
                wedge_sign: (rng.next() % 2) as usize,
                mask_type: if rng.next() % 2 == 0 {
                    DiffwtdMaskType::Diffwtd38
                } else {
                    DiffwtdMaskType::Diffwtd38Inv
                },
                ty: CompoundType::ALL[(rng.next() % 4) as usize],
            };
            let ref_mv_idx = rng.range(0, 3);
            let _ = probe;

            // The port's cache is a Vec whose LENGTH is C's index, so a
            // `start_idx` past capacity is modelled by a full Vec.
            let filler = CompRdStats {
                costs: CompTypeCosts::default(),
                mv: [Mv { row: 0, col: 0 }; 2],
                ref_frames: [1, 5],
                mode: PredMode::NewNewMv,
                filter: 0,
                ref_mv_idx: 0,
                is_global: [false; 2],
                interinter_comp: InterInterComp::default(),
            };
            let mut stats: Vec<CompRdStats> =
                vec![filler; (start_idx.max(0) as usize).min(MAX_COMP_RD_STATS)];
            let before = stats.len();
            let stored = save_comp_rd_search_stat(
                &mut stats,
                &costs,
                mv,
                mi.ref_frames,
                mi.mode,
                mi.filter,
                ref_mv_idx,
                mi.bsize,
                mi.wmtype,
                comp,
            );

            let (a, b) = pack_costs(&costs);
            let mvp = [mv[0].row, mv[0].col, mv[1].row, mv[1].col];
            let (mi_mv, mi_meta) = pack_mi(&mi);
            let comp_meta = [
                comp.wedge_index as i32,
                comp.wedge_sign as i32,
                match comp.mask_type {
                    DiffwtdMaskType::Diffwtd38 => 0,
                    DiffwtdMaskType::Diffwtd38Inv => 1,
                },
                comp.ty.index() as i32,
            ];
            let (new_idx, entry) = cref::ref_ct_save_comp_rd_search_stat(
                start_idx, &a, &b, &mvp, &mi_mv, &mi_meta, &comp_meta, ref_mv_idx,
            );
            assert_eq!(
                stored,
                entry.is_some(),
                "save_comp_rd_search_stat stored? (start_idx={start_idx})"
            );
            assert_eq!(
                stats.len() as i32,
                new_idx
                    .max(0)
                    .min(MAX_COMP_RD_STATS as i32 + 1)
                    .min(if entry.is_some() {
                        new_idx
                    } else {
                        before as i32
                    }),
                "cache length tracks C's comp_rd_stats_idx (start_idx={start_idx})"
            );
            if let Some((meta, ei32, ei64, emv)) = entry {
                let got = stats.last().expect("stored entry");
                let (g32, g64, gmv, gmeta) = pack_stats(got);
                assert_eq!(
                    (g32, g64, gmv, gmeta),
                    (ei32, ei64, emv, meta),
                    "stored COMP_RD_STATS (start_idx={start_idx})"
                );
            } else {
                dropped += 1;
            }
        }
    }
    assert!(dropped > 0, "the cache-full arm was never reached");
}

#[test]
fn backup_stats_matches_c() {
    let mut rng = Rng(0x5EED_0C23);
    for _ in 0..3000 {
        for ty in CompoundType::ALL {
            let mut port = rand_costs(&mut rng);
            let (mut a, mut b) = pack_costs(&port);
            let rate_sum = rng.range(0, 1 << 16);
            let dist_sum = i64::from(rng.range(0, 1 << 20));
            let rd_rate = rng.range(0, 1 << 16);
            let rd_dist = i64::from(rng.range(0, 1 << 20));
            let rs2 = rng.range(0, 1 << 14);
            port.backup(ty, rd_rate, rd_dist, rate_sum, dist_sum, rs2);
            cref::ref_ct_backup_stats(
                ty.index() as i32,
                &mut a,
                &mut b,
                rate_sum,
                dist_sum,
                rd_rate,
                rd_dist,
                rs2,
            );
            assert_eq!(port, unpack_costs(&a, &b), "backup_stats({ty:?})");
        }
    }
}

#[test]
fn update_best_info_matches_c() {
    let mut rng = Rng(0x5EED_0C24);
    for _ in 0..3000 {
        let mbmi_comp = InterInterComp {
            wedge_index: (rng.next() % 16) as usize,
            wedge_sign: (rng.next() % 2) as usize,
            mask_type: if rng.next() % 2 == 0 {
                DiffwtdMaskType::Diffwtd38
            } else {
                DiffwtdMaskType::Diffwtd38Inv
            },
            ty: CompoundType::ALL[(rng.next() % 4) as usize],
        };
        let mut best = BestCompTypeStats::default();
        best.comp_best_model_rd = i64::from(rng.range(0, 1 << 20));
        best.best_compmode_interinter_cost = rng.range(0, 1 << 14);
        let mut rd = i64::from(rng.range(0, 1 << 24));
        let best_rd_cur = i64::from(rng.range(0, 1 << 24));
        let model_rd_cur = i64::from(rng.range(0, 1 << 24));
        let rs2 = rng.range(0, 1 << 14);

        let mbmi_meta = [
            mbmi_comp.wedge_index as i32,
            mbmi_comp.wedge_sign as i32,
            match mbmi_comp.mask_type {
                DiffwtdMaskType::Diffwtd38 => 0,
                DiffwtdMaskType::Diffwtd38Inv => 1,
            },
            mbmi_comp.ty.index() as i32,
        ];
        let best_meta = [
            best.best_compound_data.wedge_index as i32,
            best.best_compound_data.wedge_sign as i32,
            0,
            best.best_compound_data.ty.index() as i32,
        ];
        let (c_rd, c_model, c_meta, c_cost) = cref::ref_ct_update_best_info(
            &mbmi_meta,
            rd,
            best.comp_best_model_rd,
            &best_meta,
            best.best_compmode_interinter_cost,
            best_rd_cur,
            model_rd_cur,
            rs2,
        );
        best.update(&mut rd, mbmi_comp, best_rd_cur, model_rd_cur, rs2);
        let got_meta = [
            best.best_compound_data.wedge_index as i32,
            best.best_compound_data.wedge_sign as i32,
            match best.best_compound_data.mask_type {
                DiffwtdMaskType::Diffwtd38 => 0,
                DiffwtdMaskType::Diffwtd38Inv => 1,
            },
            best.best_compound_data.ty.index() as i32,
        ];
        assert_eq!(
            (
                rd,
                best.comp_best_model_rd,
                got_meta,
                best.best_compmode_interinter_cost
            ),
            (c_rd, c_model, c_meta, c_cost),
            "update_best_info"
        );
    }
}

#[test]
fn update_mask_best_mv_matches_c() {
    let mut rng = Rng(0x5EED_0C25);
    for _ in 0..3000 {
        let mbmi_mv = [rand_mv(&mut rng), rand_mv(&mut rng)];
        let mut best_mv = [rand_mv(&mut rng), rand_mv(&mut rng)];
        let mut best_rate = rng.range(0, 1 << 14);
        let tmp_rate = rng.range(0, 1 << 14);

        let c_in_mv = [
            best_mv[0].row,
            best_mv[0].col,
            best_mv[1].row,
            best_mv[1].col,
        ];
        let (c_mv, c_rate) = cref::ref_ct_update_mask_best_mv(
            &[
                mbmi_mv[0].row,
                mbmi_mv[0].col,
                mbmi_mv[1].row,
                mbmi_mv[1].col,
            ],
            &c_in_mv,
            best_rate,
            tmp_rate,
        );
        update_mask_best_mv(mbmi_mv, &mut best_mv, &mut best_rate, tmp_rate);
        assert_eq!(
            (
                [
                    best_mv[0].row,
                    best_mv[0].col,
                    best_mv[1].row,
                    best_mv[1].col
                ],
                best_rate
            ),
            (c_mv, c_rate),
            "update_mask_best_mv"
        );
    }
}

#[test]
fn populate_reuse_comp_type_data_matches_c() {
    let mut rng = Rng(0x5EED_0C26);
    let (mut applied, mut n) = (0usize, 0usize);
    for _ in 0..4000 {
        let winner = CompoundType::ALL[(rng.next() % 4) as usize];
        let mut costs = rand_costs(&mut rng);
        // One case in three leaves the winner's rate at its INT_MAX sentinel,
        // which is the "reuse produced nothing" arm.
        if rng.next() % 3 == 0 {
            costs.rate[winner.index()] = i32::MAX;
        }
        let st = CompRdStats {
            costs,
            mv: [rand_mv(&mut rng), rand_mv(&mut rng)],
            ref_frames: [1, 5],
            mode: PredMode::NewNewMv,
            filter: 0,
            ref_mv_idx: 0,
            is_global: [false; 2],
            interinter_comp: InterInterComp {
                wedge_index: (rng.next() % 16) as usize,
                wedge_sign: (rng.next() % 2) as usize,
                mask_type: if rng.next() % 2 == 0 {
                    DiffwtdMaskType::Diffwtd38
                } else {
                    DiffwtdMaskType::Diffwtd38Inv
                },
                ty: winner,
            },
        };
        let cur_mv = [rand_mv(&mut rng), rand_mv(&mut rng)];
        let rate_mv = rng.range(0, 1 << 14);
        let best = BestCompTypeStats {
            best_compmode_interinter_cost: rng.range(0, 1 << 14),
            ..BestCompTypeStats::default()
        };
        let rdmult = rng.range(1, 1 << 14);

        let got = populate_reuse_comp_type_data(rdmult, &st, &costs, cur_mv, rate_mv, &best);

        let (_, _, _, st_meta) = pack_stats(&st);
        let (a, b) = pack_costs(&costs);
        let cur = [cur_mv[0].row, cur_mv[0].col, cur_mv[1].row, cur_mv[1].col];
        let (c_ret, c_rd, c_meta, c_mv, c_flags) = cref::ref_ct_populate_reuse_comp_type_data(
            rdmult,
            &st_meta,
            &a,
            &b,
            &cur,
            rate_mv,
            best.best_compmode_interinter_cost,
            i64::MAX,
        );
        assert_eq!(
            got.compmode_interinter_cost, c_ret,
            "populate_reuse_comp_type_data return"
        );
        match got.applied {
            None => {
                // C leaves *rd and mbmi untouched on this arm.
                assert_eq!(c_rd, i64::MAX, "the no-reuse arm must not set *rd");
            }
            Some(app) => {
                assert_eq!(app.rd, c_rd, "reused rd");
                assert_eq!(app.winner.index() as i32, c_meta[3], "winner type");
                assert_eq!(
                    [
                        app.interinter_comp.wedge_index as i32,
                        app.interinter_comp.wedge_sign as i32,
                        match app.interinter_comp.mask_type {
                            DiffwtdMaskType::Diffwtd38 => 0,
                            DiffwtdMaskType::Diffwtd38Inv => 1,
                        },
                        app.interinter_comp.ty.index() as i32,
                    ],
                    c_meta,
                    "mbmi->interinter_comp after reuse"
                );
                assert_eq!(
                    [app.mv[0].row, app.mv[0].col, app.mv[1].row, app.mv[1].col],
                    c_mv,
                    "mbmi->mv after reuse"
                );
                assert_eq!(
                    [
                        i32::from(app.winner.comp_group_idx()),
                        i32::from(app.winner.compound_idx())
                    ],
                    c_flags,
                    "comp_group_idx / compound_idx after reuse"
                );
                applied += 1;
            }
        }
        n += 1;
    }
    assert!(applied > 0 && applied < n, "vacuous: {applied}/{n} reused");
}

// ===================================================================
// The transform-search gate (compound_type.c:1069 + rdopt_utils.h:347, :778
// + model_rd.h:69).
//
// | test | C function |
// |---|---|
// | `get_txfm_rd_gate_level_matches_c` | `get_txfm_rd_gate_level` (rdopt_utils.h:778) |
// | `check_txfm_eval_matches_c` | `check_txfm_eval` (rdopt_utils.h:347) |
// | `compute_sse_plane_matches_c` | `compute_sse_plane` (model_rd.h:69) + `calculate_sse` (:49) |
// | `prune_mode_by_skip_rd_matches_c` | `prune_mode_by_skip_rd` (compound_type.c:1069) |
// ===================================================================

use aom_encode::compound_type::{
    MAX_TX_RD_GATE_LEVEL, TX_SEARCH_CASES, TxSearchCase, check_txfm_eval, compute_sse_plane,
    get_txfm_rd_gate_level, prune_mode_by_skip_rd,
};
use aom_encode::tx_search::get_txb_visible_dimensions;

#[test]
fn tx_gate_constants_match_the_shim() {
    let (cases, max_level) = cref::ref_ct_tx_gate_constants();
    assert_eq!(cases as usize, TX_SEARCH_CASES);
    assert_eq!(max_level as usize, MAX_TX_RD_GATE_LEVEL);
}

#[test]
fn get_txfm_rd_gate_level_matches_c() {
    let mut rng = Rng(0x5EED_0C30);
    let mut distinct = std::collections::BTreeSet::new();
    for bsize in 0..BLOCK_SIZES_ALL {
        for case in [
            TxSearchCase::Default,
            TxSearchCase::MotionMode,
            TxSearchCase::CompTypeMode,
        ] {
            for masked in [false, true] {
                for eval_mm in [false, true] {
                    // Distinct level per slot, so returning the wrong slot is
                    // visible rather than a coincidence.
                    let levels = [rng.range(0, 6), rng.range(0, 6), rng.range(0, 6)];
                    let got = get_txfm_rd_gate_level(masked, levels, bsize, case, eval_mm);
                    let want = cref::ref_ct_get_txfm_rd_gate_level(
                        masked,
                        &levels,
                        bsize as i32,
                        case as i32,
                        eval_mm,
                    );
                    assert_eq!(
                        got, want,
                        "get_txfm_rd_gate_level(bsize={bsize}, case={case:?}, \
                         masked={masked}, eval_mm={eval_mm})"
                    );
                    distinct.insert((case as u8, got == levels[case as usize]));
                }
            }
        }
    }
    // The MOTION_MODE case must have been seen taking BOTH its own slot and
    // falling through to DEFAULT (the `num_pels_log2 > 8` size gate).
    assert!(distinct.contains(&(TxSearchCase::MotionMode as u8, true)));
    assert!(distinct.contains(&(TxSearchCase::CompTypeMode as u8, true)));
}

#[test]
fn check_txfm_eval_matches_c() {
    let mut rng = Rng(0x5EED_0C31);
    let (mut trues, mut n) = (0usize, 0usize);
    for level in 1..=MAX_TX_RD_GATE_LEVEL {
        for is_luma_only in [false, true] {
            for _ in 0..400 {
                let bsize = rng.range(0, BLOCK_SIZES_ALL as i32) as usize;
                let source_variance = (rng.next() % (1 << 16)) as u32;
                let qindex = rng.range(0, 256);
                // One in eight uses the INT64_MAX sentinel, which is the
                // "no best skip RD yet" arm and skips the threshold entirely.
                let best_skip_rd = if rng.next() % 8 == 0 {
                    i64::MAX
                } else {
                    i64::from(rng.range(0, 1 << 24)) << (rng.next() % 20)
                };
                let skip_rd = i64::from(rng.range(0, 1 << 24)) << (rng.next() % 20);
                let got = check_txfm_eval(
                    source_variance,
                    qindex,
                    bsize,
                    best_skip_rd,
                    skip_rd,
                    level,
                    is_luma_only,
                );
                let want = cref::ref_ct_check_txfm_eval(
                    source_variance,
                    qindex,
                    bsize as i32,
                    best_skip_rd,
                    skip_rd,
                    level as i32,
                    is_luma_only,
                );
                assert_eq!(
                    got, want,
                    "check_txfm_eval(level={level}, luma_only={is_luma_only}, \
                     var={source_variance}, q={qindex}, bsize={bsize}, \
                     best={best_skip_rd}, skip={skip_rd})"
                );
                trues += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(trues > 0 && trues < n, "vacuous: {trues}/{n} evaluated");

    // The verdict is `skip_rd > rd_thresh`, a STRICT comparison, and a random
    // sweep never lands on `skip_rd == rd_thresh` — a `>=` transcription
    // passes it. So find the boundary with the ORACLE (binary search for the
    // largest `skip_rd` C still evaluates) and check the port on both sides.
    // The threshold is never recomputed here, so this stays a differential
    // rather than a re-derivation of C's formula.
    let mut boundaries = 0usize;
    for level in 1..=MAX_TX_RD_GATE_LEVEL {
        for is_luma_only in [false, true] {
            for _ in 0..40 {
                let bsize = rng.range(0, BLOCK_SIZES_ALL as i32) as usize;
                let source_variance = (rng.next() % (1 << 16)) as u32;
                let qindex = rng.range(0, 256);
                let best_skip_rd = i64::from(rng.range(1, 1 << 24)) << (rng.next() % 16);
                let probe = |skip: i64| {
                    cref::ref_ct_check_txfm_eval(
                        source_variance,
                        qindex,
                        bsize as i32,
                        best_skip_rd,
                        skip,
                        level as i32,
                        is_luma_only,
                    )
                };
                if !probe(0) {
                    continue; // the gate rejects everything at this cell
                }
                let (mut lo, mut hi) = (0i64, i64::MAX / 4);
                if probe(hi) {
                    continue; // it accepts everything; no boundary to find
                }
                while lo + 1 < hi {
                    let mid = lo + (hi - lo) / 2;
                    if probe(mid) { lo = mid } else { hi = mid }
                }
                // `lo` is the last accepted value, `hi == lo + 1` the first
                // rejected one — exactly the pair `<=` vs `<` disagree on.
                for skip in [lo, hi] {
                    let got = check_txfm_eval(
                        source_variance,
                        qindex,
                        bsize,
                        best_skip_rd,
                        skip,
                        level,
                        is_luma_only,
                    );
                    assert_eq!(
                        got,
                        probe(skip),
                        "check_txfm_eval at the boundary (level={level}, \
                         luma_only={is_luma_only}, skip={skip}, best={best_skip_rd})"
                    );
                }
                boundaries += 1;
            }
        }
    }
    assert!(
        boundaries > 0,
        "no cell had a findable accept/reject boundary"
    );
}

/// Both the visible extent (a block clipped by the frame edge) and the full
/// one, since `compute_sse_plane` measures only what is visible.
#[test]
fn compute_sse_plane_matches_c() {
    let mut rng = Rng(0x5EED_0C32);
    let mut clipped = 0usize;
    for bsize in 0..BLOCK_SIZES_ALL {
        for bd in [8u8, 10, 12] {
            // LUMA only. `prune_mode_by_skip_rd` — this file's one caller —
            // passes `PLANE_TYPE_Y`, and that matters for more than coverage:
            // a subsampled plane's visible width can be an ODD multiple of 2
            // (the edge is a multiple of `MI_SIZE * 8` and chroma shifts it by
            // 4, not 3), and `aom_sse_neon`'s width-2 path reads FOUR columns
            // (`sse_wxh_neon`, aom_dsp/arm/sse_neon.c:57-67, takes the
            // `sse_4x2_neon` remainder branch). Luma visible widths are always
            // multiples of 4, which every `aom_sse` tier handles.
            for (ss_x, ss_y) in [(0usize, 0usize)] {
                let plane_bsize =
                    aom_dsp::entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
                if plane_bsize >= BLOCK_SIZES_ALL {
                    continue;
                }
                let (pw, ph) = (BLK_W[plane_bsize], BLK_H[plane_bsize]);
                // `mb_to_*_edge` are in 1/8-pel LUMA units and negative when
                // the block overhangs the frame. The encoder only ever
                // produces MULTIPLES OF `MI_SIZE * 8 == 32` there, because the
                // edge is `(mi_cols - mi_col - bw_mi) * MI_SIZE * 8`. That
                // matters: a free choice reaches a visible width of 3, and
                // `aom_sse`'s dispatched kernels are written for the widths
                // the encoder can produce (measured: SIGBUS at 3x3).
                for k in [0i32, 1, 2] {
                    let mb_to_right_edge = -32 * k;
                    let mb_to_bottom_edge = -32 * k;
                    let (vw, vh) = get_txb_visible_dimensions(
                        pw,
                        ph,
                        pw,
                        ph,
                        0,
                        0,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                        ss_x as u32,
                        ss_y as u32,
                    );
                    if vw == 0 || vh == 0 {
                        continue; // the block would be entirely off-frame
                    }
                    let src_stride = pw + 8;
                    let dst_stride = pw + 4;
                    let rows = ph + 2;
                    let src = Plane::random(&mut rng, src_stride * rows, bd);
                    let dst = Plane::random(&mut rng, dst_stride * rows, bd);
                    let got = compute_sse_plane(
                        src.port(),
                        src_stride,
                        dst.port(),
                        dst_stride,
                        bd,
                        vw,
                        vh,
                    );
                    let want = cref::ref_ct_compute_sse_plane(
                        i32::from(bd),
                        bsize as i32,
                        ss_x as i32,
                        ss_y as i32,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                        src.cref(),
                        src_stride as i32,
                        dst.cref(),
                        dst_stride as i32,
                        rows as i32,
                    );
                    assert_eq!(
                        got, want,
                        "compute_sse_plane(bsize={bsize}, bd={bd}, ss=({ss_x},{ss_y}), \
                         k={k}, visible={vw}x{vh})"
                    );
                    if vw < pw || vh < ph {
                        clipped += 1;
                    }
                }
            }
        }
    }
    assert!(clipped > 0, "no frame-edge-clipped cell was reached");
}

#[test]
fn prune_mode_by_skip_rd_matches_c() {
    let mut rng = Rng(0x5EED_0C33);
    let (mut trues, mut n, mut gated_off) = (0usize, 0usize, 0usize);
    for bsize in 0..BLOCK_SIZES_ALL {
        for bd in [8u8, 10, 12] {
            for masked in [false, true] {
                for _ in 0..4 {
                    // level 0 in some slots so the "gate disabled" early-out is
                    // reached as well as the live gate.
                    let levels = [rng.range(0, 6), rng.range(0, 6), rng.range(0, 6)];
                    let (pw, ph) = (BLK_W[bsize], BLK_H[bsize]);
                    let src_stride = pw + 8;
                    let dst_stride = pw + 4;
                    let rows = ph + 2;
                    let src = Plane::random(&mut rng, src_stride * rows, bd);
                    let dst = Plane::random(&mut rng, dst_stride * rows, bd);
                    let source_variance = (rng.next() % (1 << 16)) as u32;
                    let qindex = rng.range(0, 256);
                    let rdmult = rng.range(1, 1 << 14);
                    let ref_skip_rd = if rng.next() % 8 == 0 {
                        i64::MAX
                    } else {
                        i64::from(rng.range(0, 1 << 24)) << (rng.next() % 16)
                    };
                    let mode_rate = rng.range(0, 1 << 16);
                    // A MULTIPLE of `MI_SIZE * 8 == 32`, as the encoder's
                    // `(mi_cols - mi_col - bw_mi) * MI_SIZE * 8` always is.
                    // A free choice reaches a visible width of 3, which the
                    // `aom_sse` tiers handle differently per ISA — measured on
                    // x86 (Rosetta) after this test passed on aarch64.
                    let mb_to_right_edge = -32 * i32::from(rng.next() % 2 == 0);
                    let mb_to_bottom_edge = 0;
                    let (vw, vh) = get_txb_visible_dimensions(
                        pw,
                        ph,
                        pw,
                        ph,
                        0,
                        0,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                        0,
                        0,
                    );
                    if vw == 0 || vh == 0 {
                        // Entirely off-frame — a block the encoder never codes,
                        // and `aom_sse`'s do-while loops still read one
                        // iteration at width 0.
                        continue;
                    }

                    let got = prune_mode_by_skip_rd(
                        masked,
                        levels,
                        bsize,
                        source_variance,
                        qindex,
                        rdmult,
                        ref_skip_rd,
                        mode_rate,
                        || {
                            compute_sse_plane(
                                src.port(),
                                src_stride,
                                dst.port(),
                                dst_stride,
                                bd,
                                vw,
                                vh,
                            )
                        },
                    );
                    let want = cref::ref_ct_prune_mode_by_skip_rd(
                        i32::from(bd),
                        bsize as i32,
                        masked,
                        &levels,
                        source_variance,
                        qindex,
                        rdmult,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                        src.cref(),
                        src_stride as i32,
                        dst.cref(),
                        dst_stride as i32,
                        rows as i32,
                        ref_skip_rd,
                        mode_rate,
                    );
                    assert_eq!(
                        got, want,
                        "prune_mode_by_skip_rd(bsize={bsize}, bd={bd}, masked={masked}, \
                         levels={levels:?})"
                    );
                    if get_txfm_rd_gate_level(
                        masked,
                        levels,
                        bsize,
                        TxSearchCase::CompTypeMode,
                        false,
                    ) == 0
                    {
                        gated_off += 1;
                    }
                    trues += usize::from(want);
                    n += 1;
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "vacuous: {trues}/{n} evaluated");
    assert!(
        gated_off > 0,
        "the gate-disabled early-out was never reached"
    );
}

// ===================================================================
// The interintra mode search (compound_type.c:520).
//
// | test | C function |
// |---|---|
// | `compute_best_wedge_interintra_matches_c` | `compute_best_wedge_interintra` `:520` |
//
// **One perturbation of this function is INERT, and measurably so.** Dropping
// `wedge_idx_cost[wedge_index]` from `rate_overhead` changes no decision:
// `pick_interintra_wedge` already SUBTRACTS that same cost from the RD it
// returns (compound_type.c:280 via `pick_wedge_fixed_sign`), so re-adding it
// here shifts every mode by nearly the same amount, and a flip needs the two
// leading modes to have picked wedge indices priced far enough apart to cross
// their RD gap. Measured over 3,600 cells spanning all nine wedge block sizes,
// four cost regimes and `rdmult` log-swept over 13 binades: **zero flips.**
// The term is kept because it is what C computes and nothing forbids it
// firing; it is recorded here so a later reader does not read the surviving
// perturbation as a hole in the gate.
//
// The intra predictors are NOT built twice. C rebuilds them per mode inside
// the loop with `av1_build_intra_predictors_for_interintra`
// (`common/reconinter.c:1115`, a different file); the shim reports the four it
// built and the port is driven from those, so what is under test here is the
// SEARCH — the mode loop, the rate-overhead accumulation and the argmin — and
// not a second intra prediction.
// ===================================================================

use aom_encode::compound_type::{INTERINTRA_MODES, compute_best_wedge_interintra};

#[test]
fn compute_best_wedge_interintra_matches_c() {
    let mut rng = Rng(0x5EED_0C40);
    assert_eq!(cref::ref_ct_interintra_modes() as usize, INTERINTRA_MODES);
    let mut winners = [0usize; INTERINTRA_MODES];
    let mut distinct_wedges = std::collections::BTreeSet::new();
    let mut ties = 0usize;

    // Interintra is allowed at 8x8..32x32 (`av1_is_interintra_allowed_bsize`),
    // which is exactly the wedge codebook set minus the two 8x32/32x8 shapes;
    // sweep all nine anyway, since the search itself does not gate on that.
    for bsize in WEDGE_BSIZES {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let n = bw * bh;
        for iter in 0..12 {
            // The context plane the intra prediction reads its neighbours
            // from: a real 2-D buffer with the block placed well inside it,
            // so the above row, the top-right run, the left column and the
            // bottom-left run are all in bounds.
            let ctx_stride = 2 * bw + 64;
            let ctx_rows = 2 * bh + 16;
            let ctx_origin = 8 * ctx_stride + 32;
            // Every fourth cell uses a CONSTANT neighbour context. All four
            // intra modes then predict the same flat block, every mode's RD is
            // identical, and with equal mode costs so is every `total_rd` —
            // the exact tie that separates C's strict `<` from `<=`. It is a
            // real encoder state (a flat region), and without it that
            // perturbation is inert.
            let flat_ctx = iter % 4 == 3;
            let ctx_plane: Vec<u8> = if flat_ctx {
                vec![(rng.next() % 256) as u8; ctx_stride * ctx_rows]
            } else {
                (0..ctx_stride * ctx_rows)
                    .map(|_| (rng.next() % 256) as u8)
                    .collect()
            };

            let src_stride = bw + 8;
            let inter: Vec<u8> = (0..n).map(|_| (rng.next() % 256) as u8).collect();

            // With a random source the DC predictor wins EVERY cell — it is
            // the flat prediction, and a random block's best flat fit is its
            // mean, so the mode loop would be compared on one arm only.
            // (Measured: winners [54, 0, 0, 0].) A first oracle call harvests
            // the four predictors C builds from THIS neighbour context, and
            // the source is then drawn around one of them — which is what
            // content this search is meant to find looks like. The comparison
            // below is a second, independent call.
            let target = iter % INTERINTRA_MODES;
            let probe = vec![128u8; src_stride * bh];
            let (_, _, _, seed_intra) = cref::ref_ct_compute_best_wedge_interintra(
                bsize as i32,
                1,
                8,
                &vec![0i32; MAX_WEDGE_TYPES],
                &[0; 4],
                &probe,
                src_stride as i32,
                &inter,
                &ctx_plane,
                ctx_stride as i32,
                ctx_rows as i32,
                ctx_origin as i32,
                8,
                8,
                32 * 8,
                32 * 8,
                n,
            );
            // On one cell in four the source sits HALFWAY between two modes'
            // predictors instead of on one of them. That is what puts two
            // modes within a rate term of each other, and it is the only
            // regime in which the WEDGE-index half of `rate_overhead` can
            // decide the winner — without it, dropping that half is inert
            // (measured: the seeded mode wins by orders of magnitude).
            let straddle = iter % 4 == 2;
            let target2 = (target + 1) % INTERINTRA_MODES;
            let noise = 1 + (rng.next() % if straddle { 4 } else { 48 }) as i32;
            let mut src = vec![0u8; src_stride * bh];
            for r in 0..bh {
                for c in 0..bw {
                    let a = i32::from(seed_intra[target * n + r * bw + c]);
                    let base = if straddle {
                        let b = i32::from(seed_intra[target2 * n + r * bw + c]);
                        (a + b + 1) >> 1
                    } else {
                        a
                    };
                    let d = rng.range(-noise, noise + 1);
                    src[r * src_stride + c] = (base + d).clamp(0, 255) as u8;
                }
            }
            let wedge_costs: Vec<i32> = (0..MAX_WEDGE_TYPES).map(|_| rng.cost()).collect();
            // LOG-SCALED mode costs. With costs all in one narrow band the
            // mode that minimises `rd` also minimises `rd + RDCOST(overhead)`
            // in every cell, and both "argmin on rd instead of total_rd" and
            // "drop the wedge cost from the overhead" pass. Spreading the
            // costs over 22 binades makes the rate term decide some cells.
            //
            // THREE regimes, because no single one exercises the whole
            // decision (each of the first two leaves a different perturbation
            // inert, measured):
            //   * flat  — equal costs on a constant context: the exact tie.
            //   * wide  — costs log-spread over 23 binades, so the rate term,
            //             not the RD, decides which mode wins. This is what
            //             separates "argmin on total_rd" from "argmin on rd".
            //   * tight — costs in the band `av1_cost_*` actually produces
            //             (0..~10k). Only here are two modes close enough for
            //             the WEDGE-index cost to move the decision, which is
            //             what makes dropping it from `rate_overhead` visible.
            let mode_costs = if flat_ctx {
                let c = rng.cost();
                [c; INTERINTRA_MODES]
            } else if iter % 4 == 0 {
                let mut mc = [0i32; INTERINTRA_MODES];
                for m in mc.iter_mut() {
                    let hi = 1i32 << (rng.next() % 23) as i32;
                    *m = rng.range(0, hi.max(1));
                }
                mc
            } else {
                [rng.cost(), rng.cost(), rng.cost(), rng.cost()]
            };
            // `rdmult` is LOG-SWEPT, not drawn from one band. The rate terms
            // scale with it and the distortion term does not, so only a large
            // rdmult lets the WEDGE-index cost decide a cell — and without
            // such a cell, dropping `wedge_idx_cost` from `rate_overhead`
            // entirely is inert (measured). av1_compute_rd_mult spans this
            // range across the qindex sweep.
            let rd_hi = 1i32 << (8 + (rng.next() % 13) as i32);
            let rdmult = rng.range(1, rd_hi);
            let dequant_ac = rng.range(4, 1 << 11);
            // Far enough from the frame edge that the intra predictor's
            // top-right / bottom-left reaches are all available.
            let (mi_row, mi_col) = (8, 8);
            let mb_to_right_edge = 32 * 8;
            let mb_to_bottom_edge = 32 * 8;

            let (want_rd, want_mode, want_widx, intra_flat) =
                cref::ref_ct_compute_best_wedge_interintra(
                    bsize as i32,
                    rdmult,
                    dequant_ac,
                    &wedge_costs,
                    &mode_costs,
                    &src,
                    src_stride as i32,
                    &inter,
                    &ctx_plane,
                    ctx_stride as i32,
                    ctx_rows as i32,
                    ctx_origin as i32,
                    mi_row,
                    mi_col,
                    mb_to_right_edge,
                    mb_to_bottom_edge,
                    n,
                );

            let intra_planes: Vec<Pixels<'_>> = (0..INTERINTRA_MODES)
                .map(|m| Pixels::Low(&intra_flat[m * n..(m + 1) * n]))
                .collect();
            let intra_arr: [Pixels<'_>; INTERINTRA_MODES] = [
                intra_planes[0],
                intra_planes[1],
                intra_planes[2],
                intra_planes[3],
            ];
            let ctx = ctx_for(bsize, 8, rdmult, dequant_ac, &wedge_costs);
            let got = compute_best_wedge_interintra(
                &ctx,
                &mode_costs,
                Pixels::Low(&src),
                src_stride,
                Pixels::Low(&inter),
                &intra_arr,
            );
            assert_eq!(
                (got.rd, got.mode as i32, got.wedge_index as i32),
                (want_rd, want_mode, want_widx),
                "compute_best_wedge_interintra(bsize={bsize})"
            );

            // On a non-flat context the four predictors must actually
            // DIFFER, or the mode loop is being compared on four copies of one
            // input. On a flat one they must all AGREE, or the tie the flavour
            // exists to produce is not being produced.
            let distinct: std::collections::BTreeSet<&[u8]> = (0..INTERINTRA_MODES)
                .map(|m| &intra_flat[m * n..(m + 1) * n])
                .collect();
            if flat_ctx {
                assert_eq!(
                    distinct.len(),
                    1,
                    "a constant neighbour context must give one predictor"
                );
                ties += 1;
            } else {
                assert!(
                    distinct.len() >= 3,
                    "only {} distinct intra predictors at bsize={bsize} — the \
                     neighbour context is not being read",
                    distinct.len()
                );
            }
            winners[want_mode as usize] += 1;
            distinct_wedges.insert(want_widx);
        }
    }
    // Every intra mode must have won somewhere, and the wedge index must not
    // be constant: a port that returned mode 0 / index 0 would otherwise agree
    // on every cell where C happened to as well.
    for (m, hits) in winners.iter().enumerate() {
        assert!(*hits > 0, "interintra mode {m} never won");
    }
    assert!(
        distinct_wedges.len() > 2,
        "vacuous: only {} distinct winning wedge indices",
        distinct_wedges.len()
    );
    assert!(ties > 0, "the flat-context tie flavour never ran");
}

// ===================================================================
// compute_best_interintra_mode (compound_type.c:459).
//
// This one has a SIDE EFFECT the caller depends on: it combines the inter and
// intra predictors into `xd->plane[0].dst` before measuring, and leaves that
// buffer holding the combination for the mode just scored. The test compares
// the destination buffer as well as the decision, because a port that scored
// correctly and left the wrong pixels behind would break
// `handle_smooth_inter_intra_mode`, which reads them.
//
// As with the wedge search, the intra predictors come from C — the shim
// reports the four it built with `av1_build_intra_predictors_for_interintra`
// (`common/reconinter.c`, not this file) and the port is driven from those.
// ===================================================================

use aom_encode::compound_type::compute_best_interintra_mode;

#[test]
fn compute_best_interintra_mode_matches_c() {
    let mut rng = Rng(0x5EED_0C50);
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut wedge_cells = 0usize;

    for bsize in WEDGE_BSIZES {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let n = bw * bh;
        for iter in 0..16 {
            let ctx_stride = 2 * bw + 64;
            let ctx_rows = 2 * bh + 16;
            let ctx_origin = 8 * ctx_stride + 32;
            let ctx_plane: Vec<u8> = (0..ctx_stride * ctx_rows)
                .map(|_| (rng.next() % 256) as u8)
                .collect();

            let src_stride = bw + 8;
            let src: Vec<u8> = (0..src_stride * bh)
                .map(|_| (rng.next() % 256) as u8)
                .collect();
            let inter: Vec<u8> = (0..n).map(|_| (rng.next() % 256) as u8).collect();
            let mode = iter % INTERINTRA_MODES;
            // Both blend shapes: the SMOOTH interintra mask (built per mode at
            // plane resolution) and the WEDGE one (the baked codebook mask,
            // read at luma stride). They are different code paths inside
            // `av1_combine_interintra` and only one of them depends on `mode`.
            let use_wedge = iter % 4 == 3;
            let wedge_index = (rng.next() % 16) as usize;
            let mode_costs = [rng.cost(), rng.cost(), rng.cost(), rng.cost()];
            let rdmult = rng.range(1, 1 << 14);
            let dequant_ac = rng.range(4, 1 << 11);
            let dst_stride = bw + 12;
            let dst_rows = bh + 2;
            // The running best the caller threads in: sometimes INT64_MAX (the
            // first mode of a loop), sometimes a value this mode may or may not
            // beat, so both arms of the accept test are reached.
            let (best_rd_in, best_mode_in) = if rng.next() % 3 == 0 {
                (i64::MAX, INTERINTRA_MODES as i32)
            } else {
                (i64::from(rng.range(1, 1 << 30)) << (rng.next() % 8), 0)
            };

            let mut c_dst: Vec<u8> = (0..dst_stride * dst_rows)
                .map(|_| (rng.next() % 256) as u8)
                .collect();
            let mut port_dst = c_dst.clone();

            let (want_rd, want_mode, intra_flat) = cref::ref_ct_compute_best_interintra_mode(
                bsize as i32,
                rdmult,
                dequant_ac,
                &mode_costs,
                mode as i32,
                use_wedge,
                wedge_index as i32,
                &src,
                src_stride as i32,
                &inter,
                &ctx_plane,
                ctx_stride as i32,
                ctx_rows as i32,
                ctx_origin as i32,
                8,
                8,
                32 * 8,
                32 * 8,
                best_rd_in,
                best_mode_in,
                &mut c_dst,
                dst_stride as i32,
                dst_rows as i32,
                n,
            );

            // The block is well inside the frame here, so the visible extent
            // IS the plane block — `get_txb_dimensions` clamps nothing.
            let ctx = ctx_for(bsize, 8, rdmult, dequant_ac, &[0i32; MAX_WEDGE_TYPES]);
            let mut got_mode = best_mode_in as usize;
            let mut got_rd = best_rd_in;
            compute_best_interintra_mode(
                &ctx,
                &mode_costs,
                mode,
                use_wedge,
                wedge_index,
                &src,
                src_stride,
                &inter,
                &intra_flat[mode * n..(mode + 1) * n],
                &mut port_dst,
                dst_stride,
                bw,
                bh,
                &mut got_mode,
                &mut got_rd,
            );

            assert_eq!(
                (got_rd, got_mode as i32),
                (want_rd, want_mode),
                "compute_best_interintra_mode(bsize={bsize}, mode={mode}, \
                 wedge={use_wedge}, best_rd_in={best_rd_in})"
            );
            assert_eq!(
                port_dst, c_dst,
                "the COMBINED destination (bsize={bsize}, mode={mode}, wedge={use_wedge})"
            );

            if want_rd != best_rd_in {
                accepted += 1;
            } else {
                rejected += 1;
            }
            if use_wedge {
                wedge_cells += 1;
            }
        }
    }
    // The accept test is a STRICT `<`, and no random `best_rd_in` lands on the
    // rd this mode produces — a `<=` transcription passes the sweep above
    // (measured). So run each cell twice: once to learn the rd from the
    // ORACLE, then again with `best_rd_in` set to exactly that. C must decline
    // and keep the incoming mode; a `<=` port accepts.
    let mut ties = 0usize;
    for bsize in WEDGE_BSIZES {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let n = bw * bh;
        for mode in 0..INTERINTRA_MODES {
            let ctx_stride = 2 * bw + 64;
            let ctx_rows = 2 * bh + 16;
            let ctx_origin = 8 * ctx_stride + 32;
            let ctx_plane: Vec<u8> = (0..ctx_stride * ctx_rows)
                .map(|_| (rng.next() % 256) as u8)
                .collect();
            let src_stride = bw + 8;
            let src: Vec<u8> = (0..src_stride * bh)
                .map(|_| (rng.next() % 256) as u8)
                .collect();
            let inter: Vec<u8> = (0..n).map(|_| (rng.next() % 256) as u8).collect();
            let mode_costs = [rng.cost(), rng.cost(), rng.cost(), rng.cost()];
            let rdmult = rng.range(1, 1 << 14);
            let dequant_ac = rng.range(4, 1 << 11);
            let dst_stride = bw + 12;
            let dst_rows = bh + 2;
            let seed_dst: Vec<u8> = (0..dst_stride * dst_rows)
                .map(|_| (rng.next() % 256) as u8)
                .collect();

            let mut probe_dst = seed_dst.clone();
            let (rd_here, _, _) = cref::ref_ct_compute_best_interintra_mode(
                bsize as i32,
                rdmult,
                dequant_ac,
                &mode_costs,
                mode as i32,
                false,
                0,
                &src,
                src_stride as i32,
                &inter,
                &ctx_plane,
                ctx_stride as i32,
                ctx_rows as i32,
                ctx_origin as i32,
                8,
                8,
                32 * 8,
                32 * 8,
                i64::MAX,
                INTERINTRA_MODES as i32,
                &mut probe_dst,
                dst_stride as i32,
                dst_rows as i32,
                n,
            );

            // Feed that rd back in as the running best, under a DIFFERENT
            // incoming mode so the two verdicts are distinguishable.
            let incoming = (mode + 1) % INTERINTRA_MODES;
            let mut c_dst = seed_dst.clone();
            let mut port_dst = seed_dst.clone();
            let (want_rd, want_mode, intra_flat) = cref::ref_ct_compute_best_interintra_mode(
                bsize as i32,
                rdmult,
                dequant_ac,
                &mode_costs,
                mode as i32,
                false,
                0,
                &src,
                src_stride as i32,
                &inter,
                &ctx_plane,
                ctx_stride as i32,
                ctx_rows as i32,
                ctx_origin as i32,
                8,
                8,
                32 * 8,
                32 * 8,
                rd_here,
                incoming as i32,
                &mut c_dst,
                dst_stride as i32,
                dst_rows as i32,
                n,
            );
            let ctx = ctx_for(bsize, 8, rdmult, dequant_ac, &[0i32; MAX_WEDGE_TYPES]);
            let mut got_mode = incoming;
            let mut got_rd = rd_here;
            compute_best_interintra_mode(
                &ctx,
                &mode_costs,
                mode,
                false,
                0,
                &src,
                src_stride,
                &inter,
                &intra_flat[mode * n..(mode + 1) * n],
                &mut port_dst,
                dst_stride,
                bw,
                bh,
                &mut got_mode,
                &mut got_rd,
            );
            assert_eq!(
                (got_rd, got_mode as i32),
                (want_rd, want_mode),
                "compute_best_interintra_mode at an exact tie (bsize={bsize}, mode={mode})"
            );
            assert_eq!(port_dst, c_dst, "the combined destination at a tie");
            assert_eq!(want_mode, incoming as i32, "C must DECLINE an exact tie");
            ties += 1;
        }
    }
    assert!(ties > 0, "the exact-tie pass never ran");

    // Both arms of the accept test, and both mask shapes, must have run — a
    // port that never accepted (or never combined) would otherwise agree
    // wherever C also declined.
    assert!(accepted > 0, "the candidate was never accepted");
    assert!(rejected > 0, "the candidate was never rejected");
    assert!(wedge_cells > 0, "the wedge mask shape was never used");
}
