//! Differential harness for libaom's inter-mode RD MODEL and the NEWMV
//! assembly around it (`av1/encoder/rdopt.c`) — the ports in
//! `aom_encode::rdopt_model` and the NEWMV half of `aom_encode::rdopt_mv`.
//!
//! # Evidence tier — MIXED, per function
//!
//! - **Tier 1 (the real exported symbol out of `libaom.a`)**:
//!   `av1_inter_mode_data_init`, `av1_inter_mode_data_fit`,
//!   `av1_get_ref_mv_from_stack`, `av1_get_ref_mv`. The shim `#undef`s the
//!   renames at the top of `rdopt_shim.c` for the first two so the call lands
//!   in the archive, not in the shim TU's copy.
//! - **Tier 1c (libaom's C source compiled into the shim archive)**:
//!   `get_est_rate_dist`, `inter_mode_data_push`, `inter_mode_data_block_idx`,
//!   `clamp_mv_in_range`, `prune_ref_mv_idx_search`. These are `static`; see
//!   `rdopt_mv_diff.rs`'s header and `rdopt_shim.c`'s for the argument, and
//!   `rdopt_mv_diff::rdopt_shim_tu_agrees_with_archive` for the measurement
//!   that backs it.
//!
//! # Float exactness
//!
//! `av1_inter_mode_data_fit` and `get_est_rate_dist` are `f64` arithmetic with
//! a `sqrt`, a division and a `round`. Every comparison here is on `to_bits()`
//! — an epsilon comparison would hide exactly the divergence this exists to
//! catch (a fused multiply-add, or `round_ties_even` instead of C's
//! round-half-away-from-zero).

mod common;
use common::Rng;

use aom_encode::rdopt_model::{InterModeRdModel, block_uses_rd_model};
use aom_encode::rdopt_mv::{
    FullMvLimits, MAX_REF_MV_SEARCH, Mv, PredMode, RefMvRow, SingleNewMvRow, clamp_mv_in_range,
    get_ref_mv, get_ref_mv_from_stack, newmv_reduced_search_range, prune_ref_mv_idx_search,
    ref_frame_type,
};
use aom_sys_ref as cref;

const BLOCK_SIZES_ALL: usize = 22;

fn to_c_model(m: &InterModeRdModel) -> cref::InterModeRdModelFlat {
    cref::InterModeRdModelFlat {
        ready: i32::from(m.ready),
        num: m.num,
        d: [
            m.a,
            m.b,
            m.dist_mean,
            m.ld_mean,
            m.sse_mean,
            m.sse_sse_mean,
            m.sse_ld_mean,
            m.dist_sum,
            m.ld_sum,
            m.sse_sum,
            m.sse_sse_sum,
            m.sse_ld_sum,
        ],
    }
}

fn assert_model_eq(port: &InterModeRdModel, c: &cref::InterModeRdModelFlat, what: &str) {
    let got = to_c_model(port);
    assert_eq!(got.ready, c.ready, "{what}: ready");
    assert_eq!(got.num, c.num, "{what}: num");
    const NAMES: [&str; 12] = [
        "a",
        "b",
        "dist_mean",
        "ld_mean",
        "sse_mean",
        "sse_sse_mean",
        "sse_ld_mean",
        "dist_sum",
        "ld_sum",
        "sse_sum",
        "sse_sse_sum",
        "sse_ld_sum",
    ];
    for i in 0..12 {
        assert_eq!(
            got.d[i].to_bits(),
            c.d[i].to_bits(),
            "{what}: {} differs — got {:?}, C {:?} (compared as bits: an \
             epsilon comparison would hide a contracted multiply-add, which is \
             the divergence this test exists for)",
            NAMES[i],
            got.d[i],
            c.d[i]
        );
    }
}

fn rand_model(rng: &mut Rng) -> InterModeRdModel {
    let d = |rng: &mut Rng| (rng.range(-1 << 20, 1 << 20) as f64) / 7.0;
    InterModeRdModel {
        ready: rng.next() % 2 == 0,
        num: rng.range(0, 400),
        a: d(rng),
        b: d(rng),
        dist_mean: d(rng).abs(),
        ld_mean: d(rng),
        sse_mean: d(rng).abs(),
        sse_sse_mean: d(rng).abs(),
        sse_ld_mean: d(rng),
        dist_sum: d(rng).abs(),
        ld_sum: d(rng),
        sse_sum: d(rng).abs(),
        sse_sse_sum: d(rng).abs(),
        sse_ld_sum: d(rng),
    }
}

#[test]
fn block_uses_rd_model_matches_c() {
    let mut excluded = 0;
    for bsize in 0..BLOCK_SIZES_ALL {
        let want = cref::ref_rdopt_inter_mode_data_block_idx(bsize as i32) != -1;
        let got = block_uses_rd_model(bsize);
        assert_eq!(got, want, "inter_mode_data_block_idx({bsize})");
        excluded += usize::from(!want);
    }
    assert_eq!(
        excluded, 5,
        "C excludes exactly the five sub-8x8-ish shapes"
    );
}

#[test]
fn inter_mode_data_init_matches_c() {
    let mut rng = Rng(0x5eed_0021);
    for _ in 0..200 {
        let start = rand_model(&mut rng);
        let bsize = rng.range(0, BLOCK_SIZES_ALL as i32);
        let mut want = to_c_model(&start);
        cref::ref_rdopt_inter_mode_data_init(bsize, &mut want);
        let mut got = start;
        got.init();
        assert_model_eq(&got, &want, &format!("av1_inter_mode_data_init({bsize})"));
    }
    // Non-vacuity: init must leave the means ALONE. If it zeroed them the port
    // and C would still agree only if both did, so assert C's behaviour
    // directly.
    let mut m = cref::InterModeRdModelFlat {
        ready: 1,
        num: 42,
        d: [1.5; 12],
    };
    cref::ref_rdopt_inter_mode_data_init(3, &mut m);
    assert_eq!(m.ready, 0);
    assert_eq!(m.num, 0);
    assert_eq!(
        m.d[2], 1.5,
        "C's init cleared dist_mean — the port's `init` must be changed to match"
    );
    assert_eq!(m.d[7], 0.0, "C's init did NOT clear dist_sum");
}

#[test]
fn inter_mode_data_push_matches_c() {
    let mut rng = Rng(0x5eed_0022);
    let mut accepted = 0;
    let mut n = 0;
    for _ in 0..3000 {
        let start = rand_model(&mut rng);
        let bsize = rng.range(0, BLOCK_SIZES_ALL as i32);
        let sse = rng.range(0, 1 << 26) as i64;
        // Half the time make dist == sse, which is one of C's two drop paths.
        let dist = if rng.next() % 2 == 0 {
            sse
        } else {
            rng.range(0, 1 << 26) as i64
        };
        let residue_cost = if rng.next() % 4 == 0 {
            0
        } else {
            rng.range(1, 1 << 16)
        };
        let mut want = to_c_model(&start);
        cref::ref_rdopt_inter_mode_data_push(bsize, sse, dist, residue_cost, &mut want);
        let mut got = start;
        got.push(bsize as usize, sse, dist, residue_cost);
        assert_model_eq(
            &got,
            &want,
            &format!(
                "inter_mode_data_push(bsize={bsize}, sse={sse}, dist={dist}, rc={residue_cost})"
            ),
        );
        if want.num != start.num {
            accepted += 1;
        }
        n += 1;
    }
    assert!(
        accepted > 0 && accepted < n,
        "push either always or never accepted ({accepted}/{n}) — the two drop \
         paths (residue_cost == 0, sse == dist) were not both exercised"
    );
}

#[test]
fn inter_mode_data_fit_matches_c() {
    let mut rng = Rng(0x5eed_0023);
    let mut fitted = 0;
    let mut refreshed = 0;
    let mut n = 0;
    for _ in 0..3000 {
        let mut start = rand_model(&mut rng);
        // Push num across the 200 / 64 thresholds from both sides.
        start.num = match rng.next() % 4 {
            0 => rng.range(0, 64),
            1 => rng.range(64, 200),
            2 => rng.range(200, 400),
            _ => rng.range(0, 400),
        };
        let bsize = rng.range(0, BLOCK_SIZES_ALL as i32);
        let rdmult = rng.range(0, 1 << 12);
        let mut want = to_c_model(&start);
        cref::ref_rdopt_inter_mode_data_fit(bsize, rdmult, &mut want);
        let mut got = start;
        got.fit(bsize as usize);
        assert_model_eq(
            &got,
            &want,
            &format!(
                "av1_inter_mode_data_fit(bsize={bsize}, ready={}, num={})",
                start.ready, start.num
            ),
        );
        if want.num != start.num {
            fitted += 1;
            if start.ready {
                refreshed += 1;
            }
        }
        n += 1;
    }
    assert!(
        fitted > 0 && fitted < n,
        "fit never ran, or always ran ({fitted}/{n}) — the sample-count gates \
         were not straddled"
    );
    assert!(
        refreshed > 0,
        "the ready==1 REFRESH arm (which blends the old means at weight 3) was \
         never reached; only the first-fit arm was tested"
    );
}

#[test]
fn get_est_rate_dist_matches_c() {
    let mut rng = Rng(0x5eed_0024);
    let mut ready_hits = 0;
    let mut zero_cost = 0;
    let mut clamped = 0;
    let mut halves = 0;
    let mut cost_halves = 0;
    let mut n = 0;
    for _ in 0..6000 {
        let mut m = rand_model(&mut rng);
        // Drive est_ld near zero often, so the |est_ld| < 1e-2 clamp arm and
        // the negative-cost arm are both reached rather than assumed dead.
        if rng.next() % 3 == 0 {
            m.a = 0.0;
            m.b = (rng.range(-100, 100) as f64) / 10000.0;
        }
        let bsize = rng.range(0, BLOCK_SIZES_ALL as i32);
        let mut sse = rng.range(0, 1 << 26) as i64;
        // EXACT half-integers, deliberately. Both `round`s in this function are
        // ties-away-from-zero in C, and `f64::round_ties_even` — the obvious
        // Rust spelling — agrees everywhere EXCEPT at a tie. Measured: with a
        // generator that never produces one, swapping the port to
        // `round_ties_even` left this test GREEN. So a fifth of the draws are
        // built to land on .5 in `dist_mean.round()` and in
        // `((sse - dist_mean) / est_ld).round()`.
        if rng.next() % 5 == 0 {
            m.ready = true;
            m.dist_mean = rng.range(0, 1 << 20) as f64 + 0.5;
            m.a = 0.0;
            m.b = 2.0;
            // (sse - dist_mean) / 2 is a half-integer iff sse - dist_mean is
            // an odd multiple of 1, which it is by construction here.
            sse = m.dist_mean as i64 + 1 + 2 * rng.range(0, 1 << 16) as i64;
        } else if rng.next() % 5 == 0 {
            // The OTHER round: `((sse - dist_mean) / est_ld).round()`. With an
            // integer dist_mean, est_ld == 2 and an ODD gap, the quotient is an
            // exact .5. (Measured: without this family, swapping that call to
            // `round_ties_even` also left the test green.)
            m.ready = true;
            m.dist_mean = rng.range(0, 1 << 20) as f64;
            m.a = 0.0;
            m.b = 2.0;
            sse = m.dist_mean as i64 + 1 + 2 * rng.range(0, 1 << 14) as i64;
        }
        let c_model = to_c_model(&m);
        let want = cref::ref_rdopt_get_est_rate_dist(bsize, &c_model, sse);
        let got = m.est_rate_dist(sse);
        assert_eq!(
            got, want,
            "get_est_rate_dist(bsize={bsize}, sse={sse}, ready={}, a={}, b={}, \
             dist_mean={})",
            m.ready, m.a, m.b, m.dist_mean
        );
        if let Some((cost, _)) = want {
            ready_hits += 1;
            if m.dist_mean.fract() == 0.5 {
                halves += 1;
            }
            if m.ready && m.a == 0.0 && m.b == 2.0 && m.dist_mean.fract() == 0.0 {
                let v = (sse as f64 - m.dist_mean) / 2.0;
                if v.fract().abs() == 0.5 {
                    cost_halves += 1;
                }
            }
            if cost == 0 {
                zero_cost += 1;
            }
            if cost == i32::MAX / 2 {
                clamped += 1;
            }
        }
        n += 1;
    }
    assert!(
        ready_hits > 0 && ready_hits < n,
        "the ready gate was constant"
    );
    assert!(zero_cost > 0, "the zero-residue-cost arm was never reached");
    assert!(
        clamped > 0,
        "the |est_ld| < 1e-2 -> INT_MAX/2 clamp arm was never reached; that arm \
         is the one C's own TODO flags as a stopgap"
    );
    assert!(
        halves > 100,
        "only {halves} draws put dist_mean on an exact .5 — without those, \
         `round` and `round_ties_even` are indistinguishable and this test \
         does not pin C's rounding mode"
    );
    assert!(
        cost_halves > 100,
        "only {cost_halves} draws put the residue-cost quotient on an exact .5 \
         — the SECOND `round` in this function is then unpinned"
    );
}

// ---------------------------------------------------------------------------
// NEWMV assembly.
// ---------------------------------------------------------------------------

fn ref_pairs() -> Vec<(i32, i32)> {
    const UNIDIR: [(i32, i32); 9] = [
        (1, 2),
        (1, 3),
        (1, 4),
        (5, 7),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (6, 7),
    ];
    let mut v: Vec<(i32, i32)> = (1..8).map(|r| (r, -1)).collect();
    for a in 1..5 {
        for b in 5..8 {
            v.push((a, b));
        }
    }
    v.extend_from_slice(&UNIDIR);
    v
}

fn rand_row(rng: &mut Rng) -> (RefMvRow, cref::RefMvRow) {
    let mut port = RefMvRow {
        count: (rng.next() % 9) as usize,
        ..RefMvRow::default()
    };
    let mut c = cref::RefMvRow {
        count: port.count as i32,
        ..cref::RefMvRow::default()
    };
    for i in 0..8 {
        let (tr, tc) = (rng.range(-800, 800) as i16, rng.range(-800, 800) as i16);
        let (cr, cc) = (rng.range(-800, 800) as i16, rng.range(-800, 800) as i16);
        port.this_mv[i] = Mv::new(tr, tc);
        port.comp_mv[i] = Mv::new(cr, cc);
        c.this_mv[i] = (tr, tc);
        c.comp_mv[i] = (cr, cc);
    }
    for r in 0..8 {
        let g = (rng.range(-40, 40) as i16, rng.range(-40, 40) as i16);
        c.global_mvs[r] = g;
    }
    (port, c)
}

/// The port's helpers take `global_mvs[ref_frame_type(rf)]` directly, so the
/// harness resolves the row exactly as C's accessor does.
///
/// **A COMPOUND pair has no resolvable row here.** C's
/// `av1_get_ref_mv_from_stack` fallback reads
/// `mbmi_ext->global_mvs[ref_frame_type]`, and `global_mvs` is declared
/// `int_mv global_mvs[REF_FRAMES]` — 8 entries, while a compound
/// `ref_frame_type` is 8..29. That would be an out-of-bounds read in C, and it
/// is unreachable: the fallback sits inside the `rf[1] <= INTRA_FRAME` arm, so
/// only single-reference rows ever get there (and for those,
/// `av1_ref_frame_type` returns `rf[0]`, which is in range). The harness
/// returns a dummy for compound rather than indexing, and the differential
/// still covers the compound arm — it just never consults the value, exactly
/// as C does not.
fn global_at_row(c: &cref::RefMvRow, rf: (i32, i32)) -> Mv {
    if rf.1 > 0 {
        return Mv::default();
    }
    let row = ref_frame_type([rf.0, rf.1]);
    Mv::new(c.global_mvs[row].0, c.global_mvs[row].1)
}

#[test]
fn get_ref_mv_from_stack_matches_c() {
    let mut rng = Rng(0x5eed_0025);
    let mut fallbacks = 0;
    let mut n = 0;
    for _ in 0..400 {
        let (port_row, c_row) = rand_row(&mut rng);
        for &rf in &ref_pairs() {
            let g = global_at_row(&c_row, rf);
            let max_ref_idx = if rf.1 > 0 { 2 } else { 1 };
            for ref_idx in 0..max_ref_idx {
                for ref_mv_idx in 0..4 {
                    let want =
                        cref::ref_rdopt_get_ref_mv_from_stack(ref_idx, rf, ref_mv_idx, &c_row);
                    let got = get_ref_mv_from_stack(
                        ref_idx as usize,
                        [rf.0, rf.1],
                        ref_mv_idx as usize,
                        &port_row,
                        g,
                    );
                    assert_eq!(
                        (got.row, got.col),
                        want,
                        "av1_get_ref_mv_from_stack(ref_idx={ref_idx}, rf={rf:?}, \
                         ref_mv_idx={ref_mv_idx}, count={})",
                        port_row.count
                    );
                    if rf.1 <= 0 && ref_mv_idx as usize >= port_row.count {
                        fallbacks += 1;
                    }
                    n += 1;
                }
            }
        }
    }
    assert!(
        fallbacks > 0 && fallbacks < n,
        "the global-MV fallback arm was never taken ({fallbacks}/{n})"
    );
}

#[test]
fn get_ref_mv_matches_c() {
    let mut rng = Rng(0x5eed_0026);
    let mut shifted = 0;
    for _ in 0..300 {
        let (port_row, c_row) = rand_row(&mut rng);
        for &rf in &ref_pairs() {
            let g = global_at_row(&c_row, rf);
            let max_ref_idx = if rf.1 > 0 { 2 } else { 1 };
            for m in 13..25 {
                let mode = PredMode::from_i32(m).unwrap();
                // C asserts has_second_ref for the two shifting modes.
                if matches!(mode, PredMode::NearNewMv | PredMode::NewNearMv) && rf.1 <= 0 {
                    continue;
                }
                for ref_idx in 0..max_ref_idx {
                    for ref_mv_idx in 0..3 {
                        let want = cref::ref_rdopt_get_ref_mv(ref_idx, m, rf, ref_mv_idx, &c_row);
                        let got = get_ref_mv(
                            ref_idx as usize,
                            mode,
                            [rf.0, rf.1],
                            ref_mv_idx as usize,
                            &port_row,
                            g,
                        );
                        assert_eq!(
                            (got.row, got.col),
                            want,
                            "av1_get_ref_mv(ref_idx={ref_idx}, mode={m}, rf={rf:?}, \
                             ref_mv_idx={ref_mv_idx})"
                        );
                        if matches!(mode, PredMode::NearNewMv | PredMode::NewNearMv) {
                            shifted += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        shifted > 0,
        "the NEAR_NEWMV / NEW_NEARMV `ref_mv_idx + 1` shift was never exercised"
    );
}

#[test]
fn clamp_mv_in_range_matches_c() {
    let mut rng = Rng(0x5eed_0027);
    let mut moved = 0;
    let mut n = 0;
    for _ in 0..300 {
        let (port_row, c_row) = rand_row(&mut rng);
        let col_min = rng.range(-400, 0);
        let row_min = rng.range(-400, 0);
        let limits = FullMvLimits {
            col_min,
            col_max: col_min + rng.range(0, 800),
            row_min,
            row_max: row_min + rng.range(0, 800),
        };
        for &rf in &ref_pairs() {
            let g = global_at_row(&c_row, rf);
            let max_ref_idx = if rf.1 > 0 { 2 } else { 1 };
            for m in 13..25 {
                let mode = PredMode::from_i32(m).unwrap();
                if matches!(mode, PredMode::NearNewMv | PredMode::NewNearMv) && rf.1 <= 0 {
                    continue;
                }
                for ref_idx in 0..max_ref_idx {
                    let mv = (
                        rng.range(-20000, 20000) as i16,
                        rng.range(-20000, 20000) as i16,
                    );
                    let want = cref::ref_rdopt_clamp_mv_in_range(
                        mv,
                        ref_idx,
                        m,
                        rf,
                        1,
                        &c_row,
                        [
                            limits.col_min,
                            limits.col_max,
                            limits.row_min,
                            limits.row_max,
                        ],
                    );
                    let got = clamp_mv_in_range(
                        Mv::new(mv.0, mv.1),
                        ref_idx as usize,
                        mode,
                        [rf.0, rf.1],
                        1,
                        &port_row,
                        g,
                        limits,
                    );
                    assert_eq!(
                        (got.row, got.col),
                        want,
                        "clamp_mv_in_range(mv={mv:?}, ref_idx={ref_idx}, mode={m}, \
                         rf={rf:?}, limits={limits:?})"
                    );
                    if want != mv {
                        moved += 1;
                    }
                    n += 1;
                }
            }
        }
    }
    assert!(
        moved > 0 && moved < n,
        "clamp_mv_in_range never changed an MV, or always did ({moved}/{n})"
    );
}

#[test]
fn prune_ref_mv_idx_search_matches_c() {
    let mut rng = Rng(0x5eed_0028);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..4000 {
        // Seed the save array with a mix of INVALID_MV and near-duplicates of
        // the MV under test, so both the `continue` and the `mv_diff <= thr`
        // arms fire.
        let base = (rng.range(-200, 200) as i16, rng.range(-200, 200) as i16);
        let mut c_save = [(0i16, 0i16); 4];
        let mut p_save = [[Mv::default(); 2]; MAX_REF_MV_SEARCH - 1];
        for i in 0..4 {
            let v = if rng.next() % 3 == 0 {
                (i16::MIN, i16::MIN)
            } else {
                (
                    base.0 + rng.range(-6, 7) as i16,
                    base.1 + rng.range(-6, 7) as i16,
                )
            };
            c_save[i] = v;
            p_save[i / 2][i % 2] = Mv::new(v.0, v.1);
        }
        let mv = [
            (
                base.0 + rng.range(-6, 7) as i16,
                base.1 + rng.range(-6, 7) as i16,
            ),
            (
                base.0 + rng.range(-6, 7) as i16,
                base.1 + rng.range(-6, 7) as i16,
            ),
        ];
        let ref_mv_idx = rng.range(0, 3);
        let best = rng.range(-1, 3);
        let pruning_factor = rng.range(0, 3);
        let rf = if rng.next() % 2 == 0 { (1, -1) } else { (1, 5) };
        let want_ok = cref::ref_rdopt_prune_ref_mv_idx_search(
            ref_mv_idx,
            best,
            &mut c_save,
            rf,
            mv,
            pruning_factor,
        );
        let got_ok = prune_ref_mv_idx_search(
            ref_mv_idx as usize,
            best,
            &mut p_save,
            rf.1 > 0,
            [Mv::new(mv[0].0, mv[0].1), Mv::new(mv[1].0, mv[1].1)],
            pruning_factor,
        );
        assert_eq!(
            got_ok, want_ok,
            "prune_ref_mv_idx_search return (idx={ref_mv_idx}, best={best}, \
             factor={pruning_factor}, rf={rf:?})"
        );
        let flat: Vec<(i16, i16)> = p_save.iter().flatten().map(|m| (m.row, m.col)).collect();
        assert_eq!(
            flat.as_slice(),
            c_save.as_slice(),
            "prune_ref_mv_idx_search save_mv (idx={ref_mv_idx}, rf={rf:?}) — this \
             compares the WRITE-BACK, which is what makes the next index's \
             decision"
        );
        trues += usize::from(want_ok);
        n += 1;
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn newmv_reduced_search_range_is_self_consistent() {
    // TIER 4 — deliberately, and this is the only tier-4 assertion in the
    // rdopt port so far. `handle_newmv`'s reduce_search_range block cannot be
    // driven through the C: it sits inside `handle_newmv`, whose other half is
    // a call to `av1_single_motion_search` needing a full AV1_COMP, a source
    // frame and a reference frame. What IS checked here is the block's
    // contract against the C source at rdopt.c:1372-1403, plus the invariants
    // that make it safe to call.
    let mut rng = Rng(0x5eed_0029);
    let mut reduced = 0;
    for _ in 0..2000 {
        let (row, _) = rand_row(&mut rng);
        let single: Vec<SingleNewMvRow> = (0..MAX_REF_MV_SEARCH)
            .map(|_| {
                let mut s = SingleNewMvRow::default();
                for r in 0..8 {
                    s.valid[r] = rng.next() % 3 != 0;
                    s.mv[r] = Mv::new(rng.range(-300, 300) as i16, rng.range(-300, 300) as i16);
                }
                s
            })
            .collect();
        let ref_mv = Mv::new(rng.range(-300, 300) as i16, rng.range(-300, 300) as i16);
        for ref_mv_idx in 0..3 {
            let got = newmv_reduced_search_range(
                [1, -1],
                ref_mv_idx,
                &row,
                Mv::default(),
                &single,
                ref_mv,
            );
            // idx 0 has nothing to compare against, so C leaves search_range
            // at INT_MAX.
            if ref_mv_idx == 0 {
                assert_eq!(got, None, "index 0 must never reduce the search range");
            }
            if let Some(r) = got {
                assert!(r >= 0, "a negative full-pel search radius is meaningless");
                // C: min_mv_diff < 16*8 is the gate, and the extra term is a
                // component distance, so the sum is bounded before the >> 3.
                assert!(r <= (128 + 2 * 32768 + 4) >> 3);
                reduced += 1;
            }
        }
    }
    assert!(
        reduced > 0,
        "the reduction never fired in 2000 draws — the test asserts nothing"
    );
}

#[test]
fn update_mode_start_end_index_matches_c() {
    let mut distinct = std::collections::BTreeSet::new();
    for wm in 0..2 {
        for warp in 0..2 {
            // BLOCK_16X16 is index **6** (enums.h:106), not 8 — an earlier
            // version of this sweep used 8 and the differential caught it.
            // 0/3/6 are at or below it, 7/9/21 above.
            for bsize in [0, 3, 6, 7, 9, 21] {
                for last in 0..4 {
                    for ii in 0..2 {
                        for eval in [false, true] {
                            let want = cref::ref_rdopt_update_mode_start_end_index(
                                wm, warp, bsize, last, ii, eval,
                            );
                            let got = aom_encode::rdopt_mv::update_mode_start_end_index(
                                wm != 0,
                                warp != 0,
                                bsize > 6,
                                last,
                                ii,
                                eval,
                            );
                            assert_eq!(
                                got, want,
                                "update_mode_start_end_index(winner_cand={wm}, \
                                 extra_prune_warped={warp}, bsize={bsize}, last={last}, \
                                 interintra={ii}, eval={eval})"
                            );
                            distinct.insert(want);
                        }
                    }
                }
            }
        }
    }
    assert!(
        distinct.len() >= 5,
        "only {} distinct (start, end) pairs were produced — the arms are not \
         separated by this sweep",
        distinct.len()
    );
}

#[test]
fn handle_newmv_compound_matches_c() {
    let mut rng = Rng(0x5eed_002a);
    let mv_vals = cref::ref_mv_vals();
    let mv_max = cref::ref_mv_max();
    // One cost table pair for the whole run: they are ~256 KB each and the
    // shim copies them on every call.
    let mvjcost: [i32; 4] = [rng.cost(), rng.cost(), rng.cost(), rng.cost()];
    let mvcost0: Vec<i32> = (0..mv_vals).map(|_| rng.cost()).collect();
    let mvcost1: Vec<i32> = (0..mv_vals).map(|_| rng.cost()).collect();
    assert_eq!(
        mv_max * 2 + 1,
        mv_vals,
        "MV_VALS is not 2*MV_MAX+1 — the port's cost-table centring is wrong"
    );

    let mut seeded = 0;
    let mut nonzero_rate = 0;
    let mut n = 0;
    for _ in 0..150 {
        let (port_row, c_row) = rand_row(&mut rng);
        let col_min = rng.range(-400, 0);
        let row_min = rng.range(-400, 0);
        let limits = FullMvLimits {
            col_min,
            col_max: col_min + rng.range(0, 800),
            row_min,
            row_max: row_min + rng.range(0, 800),
        };
        let mut c_single = cref::SingleNewMvTable::default();
        let mut p_single = vec![SingleNewMvRow::default(); MAX_REF_MV_SEARCH];
        for i in 0..MAX_REF_MV_SEARCH {
            for r in 0..8 {
                let v = (rng.range(-500, 500) as i16, rng.range(-500, 500) as i16);
                let ok = rng.next() % 3 != 0;
                c_single.mv[i][r] = v;
                c_single.valid[i][r] = ok;
                p_single[i].mv[r] = Mv::new(v.0, v.1);
                p_single[i].valid[r] = ok;
            }
        }
        // Compound pairs only: the single arm calls av1_single_motion_search.
        for rf in [(1, 5), (1, 7), (4, 6), (2, 5), (1, 2), (5, 7)] {
            for m in 17..25 {
                let mode = PredMode::from_i32(m).unwrap();
                for ref_mv_idx in 0..2 {
                    let seed = [
                        (rng.range(-60, 60) as i16, rng.range(-60, 60) as i16),
                        (rng.range(-60, 60) as i16, rng.range(-60, 60) as i16),
                    ];
                    let mut c_mv = seed;
                    let (want_rate, ret) = cref::ref_rdopt_handle_newmv_compound(
                        &mut c_mv,
                        m,
                        rf,
                        ref_mv_idx,
                        &c_row,
                        &c_single,
                        [
                            limits.col_min,
                            limits.col_max,
                            limits.row_min,
                            limits.row_max,
                        ],
                        &mvjcost,
                        &mvcost0,
                        &mvcost1,
                    );
                    assert_eq!(ret, 0, "the compound arm must return 0");
                    let mut p_mv = [Mv::new(seed[0].0, seed[0].1), Mv::new(seed[1].0, seed[1].1)];
                    let g = global_at_row(&c_row, rf);
                    let got_rate = aom_encode::rdopt_mv::handle_newmv_compound(
                        &mut p_mv,
                        mode,
                        [rf.0, rf.1],
                        ref_mv_idx as usize,
                        &p_single[ref_mv_idx as usize],
                        &port_row,
                        g,
                        limits,
                        |mv, ref_mv| {
                            aom_encode::inter_me::mv_bit_cost(
                                (i32::from(mv.row), i32::from(mv.col)),
                                (i32::from(ref_mv.row), i32::from(ref_mv.col)),
                                &mvjcost,
                                &mvcost0,
                                &mvcost1,
                                aom_encode::inter_me::MV_COST_WEIGHT,
                            )
                        },
                    );
                    assert_eq!(
                        [(p_mv[0].row, p_mv[0].col), (p_mv[1].row, p_mv[1].col)],
                        c_mv,
                        "handle_newmv cur_mv (mode={m}, rf={rf:?}, idx={ref_mv_idx}, \
                         seed={seed:?})"
                    );
                    assert_eq!(
                        got_rate, want_rate,
                        "handle_newmv rate_mv (mode={m}, rf={rf:?}, idx={ref_mv_idx})"
                    );
                    if c_mv != seed {
                        seeded += 1;
                    }
                    if want_rate != 0 {
                        nonzero_rate += 1;
                    }
                    n += 1;
                }
            }
        }
    }
    assert!(
        seeded > 0 && seeded < n,
        "handle_newmv never re-seeded cur_mv from the single search, or always \
         did ({seeded}/{n}) — the `valid` gate was not straddled"
    );
    assert!(
        nonzero_rate > 0,
        "rate_mv was always zero — vacuous cost tables"
    );
}
