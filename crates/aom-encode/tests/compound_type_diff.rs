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
