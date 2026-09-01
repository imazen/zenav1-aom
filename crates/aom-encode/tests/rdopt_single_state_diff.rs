//! Differential harness for the SINGLE-REFERENCE STATE table of libaom's
//! inter RD brain (`av1/encoder/rdopt.c`) — the port in
//! `aom_encode::rdopt_single_state`.
//!
//! **Tier 1c** throughout: every function under test is `static`, so the
//! oracle is libaom's own rdopt.c compiled into the shim archive
//! (`crates/aom-sys-ref/shim/rdopt_shim.c`). See `rdopt_mv_diff.rs`'s header
//! for the argument and `rdopt_mv_diff::rdopt_shim_tu_agrees_with_archive` for
//! the measurement that backs it.
//!
//! | test | C function (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | `init_single_inter_mode_search_state_matches_c` | `:4465` |
//! | `collect_single_states_matches_c` | `:4813` |
//! | `analyze_single_states_matches_c` | `:4859` |
//! | `compound_skip_get_candidates_matches_c` | `:4948` |
//! | `compound_skip_by_single_states_matches_c` | `:4982` |
//! | `skip_repeated_mv_matches_c` | `:1238` |
//! | `init_comp_avg_est_rd_matches_c` | `:516` |
//! | `init_top_tx_no_split_rd_matches_c` | `:5940` |
//! | `inter_modes_info_push_matches_c` | `:468` |
//! | `increase_motion_mode_rd_matches_c` | `:1442` |
//! | `skip_interp_filter_search_matches_c` | `:6060` |
//!
//! The interesting one is `collect` + `analyze`: they are stateful, so the
//! harness drives a RANDOM SEQUENCE of collects into both implementations and
//! compares the whole table after each step, rather than testing one call in
//! isolation. An insertion sort that is right for one element and wrong for
//! the fourth only shows up that way.

mod common;
use common::Rng;

use aom_encode::inter_costs::{
    DRL_MODE_CONTEXTS, GLOBALMV_MODE_CONTEXTS, INTRA_INTER_CONTEXTS, InterModeCosts,
    NEWMV_MODE_CONTEXTS, REF_CONTEXTS, REFMV_MODE_CONTEXTS, SINGLE_REF_BITS,
};
use aom_encode::rdopt_mv::{MAX_REF_MV_SEARCH, Mv, PredMode, RefMvRow};
use aom_encode::rdopt_single_state::{
    CompoundRows, FWD_REFS, OBMC_CAUSAL, SIMPLE_TRANSLATION, SingleState, SingleStates,
    WARPED_CAUSAL, compound_skip_by_single_states, increase_motion_mode_rd, init_comp_avg_est_rd,
    init_top_tx_no_split_rd_for_inter_modes, inter_offset, skip_interp_filter_search,
    skip_repeated_mv,
};
use aom_sys_ref as cref;

/// Draw a set of DISTINCT `(mode, ref_frame)` pairs.
///
/// The encoder collects each (direction, mode, reference) triple at most once,
/// and `single_state[dir][mode]` has only `FWD_REFS` slots with no bound check
/// in C — repeating a triple makes libaom itself write past the array. So the
/// harness draws without replacement rather than testing an input the encoder
/// cannot produce.
fn distinct_candidates(rng: &mut Rng, n: usize) -> Vec<(i32, i32)> {
    let mut all: Vec<(i32, i32)> = (13..17).flat_map(|m| (1..8).map(move |r| (m, r))).collect();
    // Fisher-Yates over the 28 pairs.
    for i in (1..all.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        all.swap(i, j);
    }
    all.truncate(n);
    all
}

/// A `mode_context` whose three sub-fields are all IN RANGE.
///
/// `REFMV_CTX_MASK` is 4 bits but `REFMV_MODE_CONTEXTS` is 6, so a uniformly
/// random 12-bit mode_context indexes `refmv_mode_cost` out of bounds — in C
/// as well as in the port. The real value is assembled from three bounded
/// counters by `av1_mode_context_analyzer`, so the harness assembles it the
/// same way.
fn rand_mode_context(rng: &mut Rng) -> i32 {
    let newmv = rng.range(0, NEWMV_MODE_CONTEXTS as i32);
    let globalmv = rng.range(0, GLOBALMV_MODE_CONTEXTS as i32);
    let refmv = rng.range(0, REFMV_MODE_CONTEXTS as i32);
    newmv | (globalmv << 3) | (refmv << 4)
}

fn to_c(s: &SingleStates) -> cref::SingleStatesFlat {
    let mut f = cref::SingleStatesFlat::default();
    for d in 0..2 {
        for m in 0..4 {
            f.ss_cnt[d][m] = s.simple[d][m].count as i32;
            f.sm_cnt[d][m] = s.modelled[d][m].count as i32;
            for r in 0..FWD_REFS {
                let e = s.simple[d][m].entries[r];
                f.ss_rd[d][m][r] = e.rd;
                f.ss_ref[d][m][r] = e.ref_frame.unwrap_or(-1);
                f.ss_valid[d][m][r] = i32::from(e.valid);
                let e = s.modelled[d][m].entries[r];
                f.sm_rd[d][m][r] = e.rd;
                f.sm_ref[d][m][r] = e.ref_frame.unwrap_or(-1);
                f.sm_valid[d][m][r] = i32::from(e.valid);
                f.order[d][m][r] = s.order[d][m][r].unwrap_or(-1);
            }
        }
    }
    f
}

fn from_c(f: &cref::SingleStatesFlat) -> SingleStates {
    let mut s = SingleStates::default();
    for d in 0..2 {
        for m in 0..4 {
            s.simple[d][m].count = f.ss_cnt[d][m] as usize;
            s.modelled[d][m].count = f.sm_cnt[d][m] as usize;
            for r in 0..FWD_REFS {
                s.simple[d][m].entries[r] = SingleState {
                    rd: f.ss_rd[d][m][r],
                    ref_frame: (f.ss_ref[d][m][r] >= 0).then_some(f.ss_ref[d][m][r]),
                    valid: f.ss_valid[d][m][r] != 0,
                };
                s.modelled[d][m].entries[r] = SingleState {
                    rd: f.sm_rd[d][m][r],
                    ref_frame: (f.sm_ref[d][m][r] >= 0).then_some(f.sm_ref[d][m][r]),
                    valid: f.sm_valid[d][m][r] != 0,
                };
                s.order[d][m][r] = (f.order[d][m][r] >= 0).then_some(f.order[d][m][r]);
            }
        }
    }
    s
}

fn assert_states_eq(port: &SingleStates, c: &cref::SingleStatesFlat, what: &str) {
    let got = to_c(port);
    for d in 0..2 {
        for m in 0..4 {
            assert_eq!(got.ss_cnt[d][m], c.ss_cnt[d][m], "{what}: ss_cnt[{d}][{m}]");
            assert_eq!(got.sm_cnt[d][m], c.sm_cnt[d][m], "{what}: sm_cnt[{d}][{m}]");
            // Compare only the POPULATED prefix of each list. C leaves the
            // tail at whatever init left, which for the modelled list is a
            // different sentinel than the port's default, and neither is read.
            let n = c.ss_cnt[d][m].max(0) as usize;
            for r in 0..n.min(FWD_REFS) {
                assert_eq!(
                    got.ss_rd[d][m][r], c.ss_rd[d][m][r],
                    "{what}: ss_rd[{d}][{m}][{r}]"
                );
                assert_eq!(
                    got.ss_ref[d][m][r], c.ss_ref[d][m][r],
                    "{what}: ss_ref[{d}][{m}][{r}]"
                );
                assert_eq!(
                    got.ss_valid[d][m][r], c.ss_valid[d][m][r],
                    "{what}: ss_valid[{d}][{m}][{r}]"
                );
            }
            let n = c.sm_cnt[d][m].max(0) as usize;
            for r in 0..n.min(FWD_REFS) {
                assert_eq!(
                    got.sm_rd[d][m][r], c.sm_rd[d][m][r],
                    "{what}: sm_rd[{d}][{m}][{r}]"
                );
                assert_eq!(
                    got.sm_ref[d][m][r], c.sm_ref[d][m][r],
                    "{what}: sm_ref[{d}][{m}][{r}]"
                );
                assert_eq!(
                    got.sm_valid[d][m][r], c.sm_valid[d][m][r],
                    "{what}: sm_valid[{d}][{m}][{r}]"
                );
            }
            for r in 0..FWD_REFS {
                assert_eq!(
                    got.order[d][m][r], c.order[d][m][r],
                    "{what}: order[{d}][{m}][{r}]"
                );
            }
        }
    }
}

#[test]
fn init_single_inter_mode_search_state_matches_c() {
    let want = cref::ref_rdopt_init_single_inter_mode_search_state();
    let got = SingleStates::init();
    assert_states_eq(&got, &want, "init_single_inter_mode_search_state");
    // The shim poisons the state with 0x33 first, so these assertions prove C
    // actually writes the fields rather than that both sides start zeroed.
    for d in 0..2 {
        for m in 0..4 {
            assert_eq!(want.ss_cnt[d][m], 0);
            for r in 0..FWD_REFS {
                assert_eq!(want.ss_rd[d][m][r], i64::MAX, "C left ss_rd poisoned");
                assert_eq!(want.ss_ref[d][m][r], -1, "C left ss_ref poisoned");
                assert_eq!(want.order[d][m][r], -1, "C left single_rd_order poisoned");
            }
        }
    }
}

/// Drive a random SEQUENCE of collects into both implementations, comparing
/// the whole table after every step.
#[test]
fn collect_single_states_matches_c() {
    let mut rng = Rng(0x5eed_0031);
    let mut inserted = 0;
    let mut equal_rds = 0;
    for _ in 0..300 {
        let mut c = cref::ref_rdopt_init_single_inter_mode_search_state();
        let mut p = SingleStates::init();
        for (m, ref_frame) in distinct_candidates(&mut rng, 8) {
            let mode = PredMode::from_i32(m).unwrap();
            let ref_mv_count = (rng.next() % 9) as usize;
            // Draw RDs from a SMALL set so equal values are common: the
            // insertion sort's strictly-greater shift condition is what keeps
            // equal RDs in insertion order, and that is invisible otherwise.
            let simple: [i64; MAX_REF_MV_SEARCH] = std::array::from_fn(|_| rng.range(0, 5) as i64);
            let modelled: [i64; MAX_REF_MV_SEARCH] =
                std::array::from_fn(|_| rng.range(0, 5) as i64);
            let row = RefMvRow {
                count: ref_mv_count,
                ..RefMvRow::default()
            };
            cref::ref_rdopt_collect_single_states(
                &mut c,
                mode.to_i32(),
                ref_frame,
                ref_mv_count as i32,
                &simple,
                &modelled,
            );
            p.collect(mode, ref_frame, &row, &simple, &modelled);
            assert_states_eq(
                &p,
                &c,
                &format!(
                    "collect_single_states(mode={:?}, ref={ref_frame}, count={ref_mv_count})",
                    mode
                ),
            );
            inserted += 1;
            let d = usize::from(ref_frame > 4);
            let mo = inter_offset(mode);
            let n = c.ss_cnt[d][mo] as usize;
            if n >= 2 && c.ss_rd[d][mo][n - 1] == c.ss_rd[d][mo][n - 2] {
                equal_rds += 1;
            }
        }
    }
    assert!(inserted > 2000);
    assert!(
        equal_rds > 50,
        "only {equal_rds} inserts landed next to an equal RD — the insertion \
         sort's tie behaviour is then untested"
    );
}

#[test]
fn analyze_single_states_matches_c() {
    let mut rng = Rng(0x5eed_0032);
    let mut ordered = 0;
    let mut invalidated = 0;
    for _ in 0..600 {
        let mut c = cref::ref_rdopt_init_single_inter_mode_search_state();
        let mut p = SingleStates::init();
        // Populate with a spread of RDs wide enough that the prune_factor/8
        // cut-off actually fires on some of them.
        let n_cand = rng.range(1, 13) as usize;
        // Two magnitude regimes. The WIDE one makes the prune cut-off fire at
        // all; the TINY one (RDs in 1..40) is what separates C's
        // `(rd >> 3) * factor` from the transposed `(rd * factor) >> 3` —
        // measured: with only the wide regime, that perturbation left this
        // test green, because the two expressions agreed on every verdict.
        let tiny = rng.next() % 2 == 0;
        let scale: i64 = if tiny { 1 } else { 1 << rng.range(0, 20) };
        let hi = if tiny { 40 } else { 1000 };
        for (m, ref_frame) in distinct_candidates(&mut rng, n_cand) {
            let mode = PredMode::from_i32(m).unwrap();
            let simple: [i64; MAX_REF_MV_SEARCH] =
                std::array::from_fn(|_| rng.range(1, hi) as i64 * scale);
            let modelled: [i64; MAX_REF_MV_SEARCH] =
                std::array::from_fn(|_| rng.range(1, hi) as i64 * scale);
            let row = RefMvRow {
                count: (rng.next() % 9) as usize,
                ..RefMvRow::default()
            };
            cref::ref_rdopt_collect_single_states(
                &mut c,
                mode.to_i32(),
                ref_frame,
                row.count as i32,
                &simple,
                &modelled,
            );
            p.collect(mode, ref_frame, &row, &simple, &modelled);
        }
        for prune_level in 1..5 {
            let mut cc = c;
            let mut pp = p;
            cref::ref_rdopt_analyze_single_states(&mut cc, prune_level);
            pp.analyze(prune_level);
            assert_states_eq(&pp, &cc, &format!("analyze_single_states({prune_level})"));
            for d in 0..2 {
                for m in 0..4 {
                    if cc.order[d][m][0] >= 0 {
                        ordered += 1;
                    }
                    for r in 1..FWD_REFS {
                        if cc.ss_valid[d][m][r] == 0 && c.ss_cnt[d][m] as usize > r {
                            invalidated += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(ordered > 0, "single_rd_order was never populated");
    assert!(
        invalidated > 0,
        "the prune_factor cut-off never invalidated an entry — the RD spread \
         is too narrow for this test to exercise it"
    );
}

#[test]
fn compound_skip_get_candidates_matches_c() {
    let mut rng = Rng(0x5eed_0033);
    let mut distinct = std::collections::BTreeSet::new();
    for _ in 0..800 {
        let mut c = cref::ref_rdopt_init_single_inter_mode_search_state();
        let n_cand = rng.range(1, 13) as usize;
        for (m, ref_frame) in distinct_candidates(&mut rng, n_cand) {
            let mode = PredMode::from_i32(m).unwrap();
            let simple: [i64; MAX_REF_MV_SEARCH] =
                std::array::from_fn(|_| rng.range(1, 1 << 22) as i64);
            let modelled: [i64; MAX_REF_MV_SEARCH] =
                std::array::from_fn(|_| rng.range(1, 1 << 22) as i64);
            cref::ref_rdopt_collect_single_states(
                &mut c,
                mode.to_i32(),
                ref_frame,
                rng.range(0, 9),
                &simple,
                &modelled,
            );
        }
        let prune_level = rng.range(1, 5);
        cref::ref_rdopt_analyze_single_states(&mut c, prune_level);
        let p = from_c(&c);
        for dir in 0..2 {
            for m in 13..17 {
                let mode = PredMode::from_i32(m).unwrap();
                let want =
                    cref::ref_rdopt_compound_skip_get_candidates(&c, prune_level, dir as i32, m);
                let got = p.candidates(prune_level, dir, mode);
                assert_eq!(
                    got as i32, want,
                    "compound_skip_get_candidates(prune={prune_level}, dir={dir}, mode={m})"
                );
                distinct.insert(want);
            }
        }
    }
    assert!(
        distinct.len() >= 3,
        "only {distinct:?} candidate counts were produced — the prune_level \
         arms are not separated"
    );
}

fn rand_single_row(rng: &mut Rng) -> (RefMvRow, cref::RefMvRow) {
    let mut port = RefMvRow {
        count: (rng.next() % 9) as usize,
        ..RefMvRow::default()
    };
    let mut c = cref::RefMvRow {
        count: port.count as i32,
        ..cref::RefMvRow::default()
    };
    for i in 0..8 {
        // A narrow MV range so single and compound MVs collide often, which is
        // what makes `ref_mv_match` interesting.
        let (r, col) = (rng.range(-2, 3) as i16, rng.range(-2, 3) as i16);
        let (cr, cc) = (rng.range(-2, 3) as i16, rng.range(-2, 3) as i16);
        port.this_mv[i] = Mv::new(r, col);
        port.comp_mv[i] = Mv::new(cr, cc);
        c.this_mv[i] = (r, col);
        c.comp_mv[i] = (cr, cc);
    }
    (port, c)
}

#[test]
fn compound_skip_by_single_states_matches_c() {
    let mut rng = Rng(0x5eed_0034);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..500 {
        let mut c = cref::ref_rdopt_init_single_inter_mode_search_state();
        let n_cand = rng.range(1, 10) as usize;
        for (m, ref_frame) in distinct_candidates(&mut rng, n_cand) {
            let mode = PredMode::from_i32(m).unwrap();
            let simple: [i64; MAX_REF_MV_SEARCH] =
                std::array::from_fn(|_| rng.range(1, 1 << 22) as i64);
            let modelled: [i64; MAX_REF_MV_SEARCH] =
                std::array::from_fn(|_| rng.range(1, 1 << 22) as i64);
            cref::ref_rdopt_collect_single_states(
                &mut c,
                mode.to_i32(),
                ref_frame,
                rng.range(0, 9),
                &simple,
                &modelled,
            );
        }
        let prune_level = rng.range(1, 5);
        cref::ref_rdopt_analyze_single_states(&mut c, prune_level);
        let p_states = from_c(&c);

        let (comp_port, mut comp_c) = rand_single_row(&mut rng);
        let (s0_port, s0_c) = rand_single_row(&mut rng);
        let (s1_port, s1_c) = rand_single_row(&mut rng);
        let mut gp = [Mv::default(); 8];
        for r in 0..8 {
            let g = (rng.range(-2, 3) as i16, rng.range(-2, 3) as i16);
            comp_c.global_mvs[r] = g;
            gp[r] = Mv::new(g.0, g.1);
        }
        for rf in [(1, 5), (1, 7), (4, 6), (2, 5), (3, 7)] {
            for m in 17..25 {
                let mode = PredMode::from_i32(m).unwrap();
                let want = cref::ref_rdopt_compound_skip_by_single_states(
                    &c,
                    prune_level,
                    m,
                    rf,
                    &comp_c,
                    &s0_c,
                    &s1_c,
                );
                let got = compound_skip_by_single_states(
                    &p_states,
                    prune_level,
                    mode,
                    [rf.0, rf.1],
                    &CompoundRows {
                        compound: &comp_port,
                        single: [&s0_port, &s1_port],
                    },
                    &gp,
                );
                assert_eq!(
                    got, want,
                    "compound_skip_by_single_states(mode={m}, rf={rf:?}, \
                     prune={prune_level}, comp_count={}, s0={}, s1={})",
                    comp_port.count, s0_port.count, s1_port.count
                );
                trues += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

fn rand_costs(rng: &mut Rng) -> InterModeCosts {
    let mut c = InterModeCosts {
        intra_inter_cost: [[0; 2]; INTRA_INTER_CONTEXTS],
        single_ref_cost: [[[0; 2]; SINGLE_REF_BITS]; REF_CONTEXTS],
        newmv_mode_cost: [[0; 2]; NEWMV_MODE_CONTEXTS],
        zeromv_mode_cost: [[0; 2]; GLOBALMV_MODE_CONTEXTS],
        refmv_mode_cost: [[0; 2]; REFMV_MODE_CONTEXTS],
        drl_mode_cost0: [[0; 2]; DRL_MODE_CONTEXTS],
    };
    for row in &mut c.newmv_mode_cost {
        for v in row {
            *v = rng.cost();
        }
    }
    for row in &mut c.zeromv_mode_cost {
        for v in row {
            *v = rng.cost();
        }
    }
    for row in &mut c.refmv_mode_cost {
        for v in row {
            *v = rng.cost();
        }
    }
    c
}

#[test]
fn skip_repeated_mv_matches_c() {
    let mut rng = Rng(0x5eed_0035);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..1500 {
        let costs = rand_costs(&mut rng);
        // Fill the modelled_rd column with a mix of INT64_MAX (mode not
        // searched) and real values, so both the early-out and the carry-across
        // are reached.
        let mut c_rd = [i64::MAX; 25];
        for slot in c_rd.iter_mut() {
            if rng.next() % 2 == 0 {
                *slot = rng.range(1, 1 << 24) as i64;
            }
        }
        let mode_context = rand_mode_context(&mut rng);
        for m in 13..17 {
            let mode = PredMode::from_i32(m).unwrap();
            for ref_mv_count in 0..4 {
                for wmtype in [0, 1, 2, 3] {
                    for is_comp in [false, true] {
                        let mut want_rd = c_rd;
                        let want = cref::ref_rdopt_skip_repeated_mv(
                            m,
                            if is_comp { (1, 5) } else { (1, -1) },
                            ref_mv_count,
                            wmtype,
                            mode_context,
                            &costs.newmv_mode_cost,
                            &costs.zeromv_mode_cost,
                            &costs.refmv_mode_cost,
                            &mut want_rd,
                        );
                        let mut got_rd = c_rd;
                        let got = skip_repeated_mv(
                            mode,
                            is_comp,
                            ref_mv_count as usize,
                            // TRANSLATION is wmtype 1, so "<= TRANSLATION" is
                            // wmtype 0 or 1.
                            wmtype <= 1,
                            mode_context,
                            &costs,
                            &mut got_rd,
                        );
                        assert_eq!(
                            got, want,
                            "skip_repeated_mv(mode={m}, count={ref_mv_count}, \
                             wmtype={wmtype}, comp={is_comp})"
                        );
                        assert_eq!(
                            got_rd, want_rd,
                            "skip_repeated_mv modelled_rd carry-across \
                             (mode={m}, count={ref_mv_count}, wmtype={wmtype})"
                        );
                        trues += usize::from(want);
                        n += 1;
                    }
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn init_comp_avg_est_rd_matches_c() {
    let mut rng = Rng(0x5eed_0036);
    let len = cref::ref_top_comp_avg_est_rd_count();
    assert!(len > 0);
    for level in 0..3 {
        for _ in 0..20 {
            let seed: Vec<i64> = (0..len).map(|_| rng.range(0, 1 << 20) as i64).collect();
            let mut want = seed.clone();
            cref::ref_rdopt_init_comp_avg_est_rd(level, &mut want);
            let mut got = seed.clone();
            init_comp_avg_est_rd(&mut got, level);
            assert_eq!(got, want, "init_comp_avg_est_rd(level={level})");
            if level == 0 {
                assert_eq!(want, seed, "level 0 must leave the buffer alone");
            } else {
                assert!(want.iter().all(|&v| v == i64::MAX));
            }
        }
    }
}

#[test]
fn init_top_tx_no_split_rd_matches_c() {
    let mut rng = Rng(0x5eed_0037);
    let (n_blocks, n_top) = cref::ref_top_inter_tx_no_split_dims();
    assert!(n_blocks > 0 && n_top > 0);
    for level in 0..3 {
        let seed: Vec<i64> = (0..n_blocks * n_top)
            .map(|_| rng.range(0, 1 << 20) as i64)
            .collect();
        let mut want = seed.clone();
        cref::ref_rdopt_init_top_tx_no_split_rd(level, &mut want);
        let mut got = seed.clone();
        init_top_tx_no_split_rd_for_inter_modes(&mut got, level);
        assert_eq!(got, want, "init_top_tx_no_split_rd(level={level})");
        if level == 0 {
            assert_eq!(want, seed, "level 0 must leave the buffer alone");
        }
    }
}

#[test]
fn inter_modes_info_push_matches_c() {
    let mut rng = Rng(0x5eed_0038);
    let mut list = Vec::new();
    for i in 0..500 {
        let mode_rate = rng.cost();
        let sse = rng.range(0, 1 << 26) as i64;
        let rd = rng.range(0, 1 << 26) as i64;
        let (num, mr, s, e) = cref::ref_rdopt_inter_modes_info_push(i, mode_rate, sse, rd);
        assert_eq!(num, i + 1, "C's num did not advance");
        assert!(aom_encode::rdopt_single_state::inter_modes_info_push(
            &mut list,
            aom_encode::rdopt_single_state::InterModeInfoEntry {
                mode_rate,
                sse,
                est_rd: rd,
            }
        ));
        assert_eq!(list.len() as i32, num, "the port's count diverged");
        let last = *list.last().unwrap();
        assert_eq!((last.mode_rate, last.sse, last.est_rd), (mr, s, e));
    }
}

#[test]
fn increase_motion_mode_rd_matches_c() {
    let mut rng = Rng(0x5eed_0039);
    let mut moved = 0;
    let mut n = 0;
    for _ in 0..2000 {
        for best_mm in [SIMPLE_TRANSLATION, OBMC_CAUSAL, WARPED_CAUSAL] {
            for this_mm in [SIMPLE_TRANSLATION, OBMC_CAUSAL, WARPED_CAUSAL] {
                let a = if rng.next() % 8 == 0 {
                    i64::MAX
                } else {
                    rng.range(0, 1 << 28) as i64
                };
                let b = if rng.next() % 8 == 0 {
                    i64::MAX
                } else {
                    rng.range(0, 1 << 28) as i64
                };
                let warp_pct = rng.range(0, 60);
                // A float percentage with a fractional part, so the f32 -> f64
                // promotion is exercised rather than only whole numbers.
                let obmc_pct = rng.range(0, 6000) as f32 / 100.0;
                let (mut wa, mut wb) = (a, b);
                cref::ref_rdopt_increase_motion_mode_rd(
                    best_mm, this_mm, &mut wa, &mut wb, warp_pct, obmc_pct,
                );
                let (mut ga, mut gb) = (a, b);
                increase_motion_mode_rd(best_mm, this_mm, &mut ga, &mut gb, warp_pct, obmc_pct);
                assert_eq!(
                    (ga, gb),
                    (wa, wb),
                    "increase_motion_mode_rd(best={best_mm}, this={this_mm}, \
                     rd=({a},{b}), warp={warp_pct}%, obmc={obmc_pct}%)"
                );
                if (wa, wb) != (a, b) {
                    moved += 1;
                }
                n += 1;
            }
        }
    }
    assert!(
        moved > 0 && moved < n,
        "constant answer ({moved}/{n} moved)"
    );
}

#[test]
fn skip_interp_filter_search_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    // MODE: GOOD 0, REALTIME 1, ALLINTRA 2. REFERENCE_MODE: SINGLE 0,
    // COMPOUND 1, SELECT 2.
    for encoding_mode in 0..3 {
        for reference_mode in 0..3 {
            for sf in 0..2 {
                for ifs in 0..2 {
                    for single in [false, true] {
                        let want = cref::ref_rdopt_skip_interp_filter_search(
                            encoding_mode,
                            reference_mode,
                            sf,
                            ifs,
                            single,
                        );
                        let got = skip_interp_filter_search(
                            encoding_mode,
                            reference_mode,
                            sf != 0,
                            ifs != 0,
                            single,
                        );
                        assert_eq!(
                            got, want,
                            "skip_interp_filter_search(mode={encoding_mode}, \
                             ref_mode={reference_mode}, sf={sf}, ifs={ifs}, \
                             single={single})"
                        );
                        trues += usize::from(want);
                        n += 1;
                    }
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}
