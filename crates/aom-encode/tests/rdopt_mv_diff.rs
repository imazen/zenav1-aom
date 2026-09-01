//! Differential harness for the ref-MV / DRL layer of libaom's inter RD brain
//! (`av1/encoder/rdopt.c`) — the port in `aom_encode::rdopt_mv`.
//!
//! # Evidence tier — read this before trusting a green run
//!
//! Every function under test is `static` in C. `nm -g upstream/build/libaom.a`
//! reports TEN exported symbols for the whole of rdopt.c and none of them is
//! `get_this_mv`, `build_cur_mv`, `get_drl_cost` or any other decision helper
//! the inter search is made of. There is therefore no exported symbol to take
//! the address of, and the alternative to this harness is hand-derived vectors
//! read off the C source — which CLAUDE.md ranks last ("transcribed oracles
//! can carry shared bugs") and this repo labels tier 4.
//!
//! So `crates/aom-sys-ref/shim/rdopt_shim.c` compiles libaom's OWN rdopt.c
//! into the shim archive, with its ten exports renamed out of the way, and
//! exposes flat wrappers around the statics. The bodies under test are
//! libaom's source. Call it **tier 1c**: real C source, compiled verbatim, as
//! against tier 1's real symbol out of the archive.
//!
//! The one gap between 1c and 1 is that this is a SECOND COMPILATION of the
//! same source, which could in principle diverge from the archive's copy
//! through flags. [`rdopt_shim_tu_agrees_with_archive`] closes that by
//! measurement: it drives the shim TU's copies of `av1_block_error_c` and
//! `av1_get_horver_correlation_full_c` (the two of rdopt.c's ten exports that
//! are pure functions of their arguments) against the ARCHIVE's exported
//! symbols on random inputs. If the second compilation ever stopped meaning
//! the same thing, that test fails and every other test in this file is
//! suspect.
//!
//! | test | C function (`av1/encoder/rdopt.c` unless noted) |
//! |---|---|
//! | `ref_frame_type_matches_c` | `av1_ref_frame_type` (`mvref_common.h:113`) |
//! | `get_single_mode_matches_c` | `get_single_mode` `:989` (+ `compound_ref{0,1}_mode`) |
//! | `check_repeat_ref_mv_matches_c` | `check_repeat_ref_mv` `:1993` |
//! | `get_this_mv_matches_c` | `get_this_mv` `:2030` |
//! | `build_cur_mv_matches_c` | `build_cur_mv` `:2110` (+ `clamp_and_check_mv` `:1293`) |
//! | `clamp_mv2_matches_c` | `clamp_mv2` `:1227` |
//! | `clamp_and_check_mv_matches_c` | `clamp_and_check_mv` `:1293` |
//! | `get_drl_cost_matches_c` | `get_drl_cost` `:2139` (+ `av1_drl_ctx`) |
//! | `get_drl_refmv_count_matches_c` | `get_drl_refmv_count` `:2182` |
//! | `is_single_newmv_valid_matches_c` | `is_single_newmv_valid` `:2168` |
//! | `prune_ref_mv_idx_using_qindex_matches_c` | `prune_ref_mv_idx_using_qindex` `:2199` |
//! | `skip_nearest_near_matches_c` | `skip_nearest_near_mv_using_refmv_weight` `:2069` |
//! | `conditional_skipintra_matches_c` | `conditional_skipintra` `:941` |
//! | `idx_mask_matches_c` | `mask_set_bit` `:2347` / `mask_check_bit` `:2349` |
//!
//! # Non-vacuity
//!
//! Predicates are cheap to pass by accident: a port that returns `false`
//! always agrees with C on every input where C returns false. So each boolean
//! test counts how often the oracle said `true` and asserts the count is in
//! `1..n` — a constant-answer port fails even when it agrees everywhere else.

mod common;
use common::Rng;

use aom_encode::rdopt_mv::{
    BlockEdges, FullMvLimits, IdxMask, MAX_REF_MV_STACK_SIZE, MB_MODE_COUNT, Mv, PredMode,
    REF_FRAMES, RefMvRow, build_cur_mv, check_repeat_ref_mv, clamp_and_check_mv, clamp_mv2,
    conditional_skipintra, get_drl_cost, get_drl_refmv_count, get_single_mode, get_this_mv,
    is_single_newmv_valid, prune_ref_mv_idx_using_qindex, ref_frame_type,
    skip_nearest_near_mv_using_refmv_weight,
};
use aom_sys_ref as cref;

/// Every reference pair `av1_ref_frame_type` has a row for: the 7 single
/// references (`rf1 == NONE_FRAME == -1`), the 12 bidirectional pairs
/// (a forward ref with a backward one) and the 9 unidirectional pairs.
///
/// **Not "every ordered pair".** `av1_ref_frame_type` computes
/// `REF_FRAMES + FWD_RF_OFFSET(rf[0]) + BWD_RF_OFFSET(rf[1]) * FWD_REFS` for
/// anything that is neither, and for e.g. `(LAST2, LAST)` that is `-7` — C
/// then indexes `ref_mv_count[-7]`. The pair is not one the encoder can build,
/// so the sweep must not build it either; an earlier version of this file did,
/// and the oracle read out of bounds rather than disagreeing.
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

/// The twelve inter modes plus the thirteen intra ones, as C's integers.
fn all_modes() -> Vec<i32> {
    (0..MB_MODE_COUNT).collect()
}

/// `mv_span`: the half-range the stack MVs are drawn from, in 1/8-pel.
/// A NARROW span (±3) makes `check_repeat_ref_mv`'s global-MV scan fire often;
/// a WIDE one (±700, i.e. ±87 full pel) is what pushes `clamp_and_check_mv`'s
/// full-pel range test across its limits. Measured: with the narrow span alone,
/// perturbing `get_mv_rawpel` to a plain `>> 3` left `build_cur_mv_matches_c`
/// GREEN — the MVs never reached a limit, so the test could not see it.
fn rand_row_span(rng: &mut Rng, mv_span: i32) -> (RefMvRow, cref::RefMvRow) {
    let mut port = RefMvRow {
        count: (rng.next() % 9) as usize,
        ..RefMvRow::default()
    };
    let mut c = cref::RefMvRow {
        count: port.count as i32,
        ..cref::RefMvRow::default()
    };
    for i in 0..MAX_REF_MV_STACK_SIZE {
        // A narrow MV range so the stack collides with `global_mv` often
        // enough for `check_repeat_ref_mv`'s scan to actually fire.
        let (tr, tc) = (
            rng.range(-mv_span, mv_span + 1) as i16,
            rng.range(-mv_span, mv_span + 1) as i16,
        );
        let (cr, cc) = (
            rng.range(-mv_span, mv_span + 1) as i16,
            rng.range(-mv_span, mv_span + 1) as i16,
        );
        let w = rng.range(0, 1300) as u16;
        port.this_mv[i] = Mv::new(tr, tc);
        port.comp_mv[i] = Mv::new(cr, cc);
        port.weight[i] = w;
        c.this_mv[i] = (tr, tc);
        c.comp_mv[i] = (cr, cc);
        c.weight[i] = w;
    }
    (port, c)
}

fn rand_row(rng: &mut Rng) -> (RefMvRow, cref::RefMvRow) {
    rand_row_span(rng, 3)
}

fn rand_global_mvs(rng: &mut Rng) -> ([Mv; REF_FRAMES], [(i16, i16); REF_FRAMES]) {
    let mut p = [Mv::default(); REF_FRAMES];
    let mut c = [(0i16, 0i16); REF_FRAMES];
    for i in 0..REF_FRAMES {
        let (r, col) = (rng.range(-3, 4) as i16, rng.range(-3, 4) as i16);
        p[i] = Mv::new(r, col);
        c[i] = (r, col);
    }
    (p, c)
}

fn rand_edges(rng: &mut Rng) -> BlockEdges {
    BlockEdges {
        left: -(rng.range(0, 4000) * 8),
        right: rng.range(0, 4000) * 8,
        top: -(rng.range(0, 4000) * 8),
        bottom: rng.range(0, 4000) * 8,
    }
}

fn rand_limits(rng: &mut Rng) -> FullMvLimits {
    let col_min = rng.range(-600, 0);
    let row_min = rng.range(-600, 0);
    FullMvLimits {
        col_min,
        col_max: col_min + rng.range(0, 1200),
        row_min,
        row_max: row_min + rng.range(0, 1200),
    }
}

// ---------------------------------------------------------------------------
// The tier claim itself.
// ---------------------------------------------------------------------------

/// The shim TU's copy of rdopt.c must agree with the ARCHIVE's copy. This is
/// what licenses reading every other test in this file as evidence about
/// libaom rather than about a second build of libaom.
#[test]
fn rdopt_shim_tu_agrees_with_archive() {
    let mut rng = Rng(0x5eed_0001);
    let mut moved = 0usize;
    for _ in 0..64 {
        let n = 16 * (1 + (rng.next() % 4) as usize);
        let coeff: Vec<i32> = (0..n).map(|_| rng.range(-4000, 4000)).collect();
        let dq: Vec<i32> = (0..n).map(|_| rng.range(-4000, 4000)).collect();
        let (tu_err, tu_ssz) = cref::ref_rdopt_tu_block_error(&coeff, &dq);
        let (a_err, a_ssz) = cref::ref_block_error(&coeff, &dq);
        assert_eq!(
            (tu_err, tu_ssz),
            (a_err, a_ssz),
            "the shim TU's av1_block_error_c disagrees with the archive's — \
             the second compilation of rdopt.c is NOT the same function, so \
             every other test in this file is measuring the wrong thing"
        );
        if tu_err != 0 {
            moved += 1;
        }
    }
    assert!(moved > 32, "block_error was ~always zero: vacuous inputs");

    for _ in 0..32 {
        let (w, h) = (8usize, 8usize);
        let stride = w + 4;
        let diff: Vec<i16> = (0..stride * h)
            .map(|_| rng.range(-500, 500) as i16)
            .collect();
        let tu = cref::ref_rdopt_tu_horver(&diff, stride, w, h);
        let arch = cref::ref_horver_correlation_full(&diff, stride, w, h);
        assert_eq!(
            tu.0.to_bits(),
            arch.0.to_bits(),
            "shim-TU hcorr differs from the archive's"
        );
        assert_eq!(
            tu.1.to_bits(),
            arch.1.to_bits(),
            "shim-TU vcorr differs from the archive's"
        );
    }
}

// ---------------------------------------------------------------------------
// Exhaustive tests — small enough domains that random sampling would be worse.
// ---------------------------------------------------------------------------

#[test]
fn ref_frame_type_matches_c() {
    let mut seen = std::collections::BTreeSet::new();
    for (a, b) in ref_pairs() {
        let want = cref::ref_rdopt_ref_frame_type(a, b);
        let got = ref_frame_type([a, b]);
        assert_eq!(
            got as i32, want,
            "av1_ref_frame_type([{a}, {b}]) — the mbmi_ext row a reference \
             pair selects; a wrong row silently reads another pair's stack"
        );
        seen.insert(want);
    }
    // 7 single rows + 21 compound rows = every row a real pair can address.
    assert_eq!(
        seen.len(),
        28,
        "the pair sweep addressed {} distinct rows, not 28 — the sweep is \
         narrower than it looks",
        seen.len()
    );
}

#[test]
fn get_single_mode_matches_c() {
    let mut compound_seen = 0;
    for m in all_modes() {
        for ref_idx in 0..2 {
            let want = cref::ref_rdopt_get_single_mode(m, ref_idx);
            // C's LUT maps the 13 intra modes to themselves at ref_idx 0 and
            // to MB_MODE_COUNT at ref_idx 1; its own assert forbids calling it
            // with them, and the port answers `None`. Only compare where a
            // caller can legally land.
            let Some(mode) = PredMode::from_i32(m) else {
                continue;
            };
            if !mode.is_inter() {
                continue;
            }
            let got =
                get_single_mode(mode, ref_idx as usize).map_or(MB_MODE_COUNT, PredMode::to_i32);
            assert_eq!(got, want, "get_single_mode(mode={m}, ref_idx={ref_idx})");
            if mode.is_inter_compound() && ref_idx == 1 {
                compound_seen += 1;
            }
        }
    }
    assert_eq!(
        compound_seen, 8,
        "only {compound_seen} compound modes reached the ref_idx=1 arm — that \
         arm is what distinguishes compound_ref1_mode from compound_ref0_mode"
    );
}

#[test]
fn conditional_skipintra_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    for m in 0..13 {
        for best in 0..13 {
            let want = cref::ref_rdopt_conditional_skipintra(m, best);
            let got = conditional_skipintra(
                PredMode::from_i32(m).unwrap(),
                PredMode::from_i32(best).unwrap(),
            );
            assert_eq!(got, want, "conditional_skipintra(mode={m}, best={best})");
            trues += usize::from(want);
            n += 1;
        }
    }
    assert!(
        trues > 0 && trues < n,
        "conditional_skipintra was constant over the whole intra grid \
         ({trues}/{n} true) — a constant port would pass this test"
    );
}

#[test]
fn prune_ref_mv_idx_using_qindex_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    // C asserts `reduce_inter_modes == 2` below 3, so 2 and 3 are the domain.
    for reduce in 2..=4 {
        for qindex in 0..256 {
            for idx in 0..3 {
                let want = cref::ref_rdopt_prune_ref_mv_idx_using_qindex(reduce, qindex, idx);
                let got = prune_ref_mv_idx_using_qindex(reduce, qindex, idx);
                assert_eq!(
                    got, want,
                    "prune_ref_mv_idx_using_qindex(reduce={reduce}, q={qindex}, idx={idx})"
                );
                trues += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn get_drl_refmv_count_matches_c() {
    let mut seen = std::collections::BTreeSet::new();
    for m in all_modes() {
        let Some(mode) = PredMode::from_i32(m) else {
            continue;
        };
        for count in 0..9 {
            let want = cref::ref_rdopt_get_drl_refmv_count((1, -1), m, count);
            let got = get_drl_refmv_count(mode, count as usize);
            assert_eq!(
                got as i32, want,
                "get_drl_refmv_count(mode={m}, ref_mv_count={count})"
            );
            seen.insert(want);
        }
    }
    assert!(
        seen.len() >= 3,
        "get_drl_refmv_count only ever returned {seen:?} — the DRL search set \
         never varied, so the test proves nothing about the has_drl arm"
    );
}

#[test]
fn idx_mask_matches_c() {
    for start in [0i32, 1, 5, -1, 0x2a] {
        for index in 0..8 {
            let mut m = IdxMask(start);
            m.set(index);
            assert_eq!(
                m.0,
                cref::ref_rdopt_mask_set_bit(start, index as i32),
                "mask_set_bit({start}, {index})"
            );
            assert_eq!(
                IdxMask(start).get(index),
                cref::ref_rdopt_mask_check_bit(start, index as i32),
                "mask_check_bit({start}, {index})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Randomised tests over the ref-MV stack.
// ---------------------------------------------------------------------------

#[test]
fn check_repeat_ref_mv_matches_c() {
    let mut rng = Rng(0x5eed_0002);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..400 {
        let (port_row, mut c_row) = rand_row(&mut rng);
        let (gp, gc) = rand_global_mvs(&mut rng);
        c_row.global_mvs = gc;
        for &(a, b) in &ref_pairs() {
            for ref_idx in 0..2 {
                // C asserts single_mode != NEWMV.
                for sm in [PredMode::NearestMv, PredMode::NearMv, PredMode::GlobalMv] {
                    let want =
                        cref::ref_rdopt_check_repeat_ref_mv((a, b), ref_idx, sm.to_i32(), &c_row);
                    let slot = if ref_idx == 0 { a } else { b };
                    // C reads global_mvs[ref_frame[ref_idx]]; NONE_FRAME (-1)
                    // in the second slot is only reachable when the caller
                    // asked about a single reference's second half, which
                    // build_cur_mv never does.
                    if slot < 0 {
                        continue;
                    }
                    let got =
                        check_repeat_ref_mv(&port_row, ref_idx as usize, sm, gp[slot as usize]);
                    assert_eq!(
                        got, want,
                        "check_repeat_ref_mv(rf=({a},{b}), ref_idx={ref_idx}, \
                         single_mode={sm:?}, count={})",
                        port_row.count
                    );
                    trues += usize::from(want);
                    n += 1;
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn get_this_mv_matches_c() {
    let mut rng = Rng(0x5eed_0003);
    let mut skipped = 0;
    let mut invalid = 0;
    let mut n = 0;
    for _ in 0..300 {
        let (port_row, mut c_row) = rand_row(&mut rng);
        let (gp, gc) = rand_global_mvs(&mut rng);
        c_row.global_mvs = gc;
        for &(a, b) in &ref_pairs() {
            for ref_idx in 0..2 {
                if (if ref_idx == 0 { a } else { b }) < 0 {
                    continue;
                }
                for m in all_modes() {
                    let Some(mode) = PredMode::from_i32(m) else {
                        continue;
                    };
                    if !mode.is_inter() {
                        continue;
                    }
                    if get_single_mode(mode, ref_idx as usize).is_none() {
                        continue; // C would assert.
                    }
                    for ref_mv_idx in 0..3 {
                        for skip in [false, true] {
                            let want = cref::ref_rdopt_get_this_mv(
                                m,
                                ref_idx,
                                ref_mv_idx,
                                skip,
                                (a, b),
                                &c_row,
                            );
                            let slot = if ref_idx == 0 { a } else { b };
                            let got = get_this_mv(
                                mode,
                                ref_idx as usize,
                                ref_mv_idx as usize,
                                skip,
                                &port_row,
                                gp[slot as usize],
                            );
                            let got_pair = got.map(|mv| (mv.row, mv.col));
                            assert_eq!(
                                got_pair, want,
                                "get_this_mv(mode={m}, ref_idx={ref_idx}, \
                                 ref_mv_idx={ref_mv_idx}, skip={skip}, \
                                 rf=({a},{b}), count={})",
                                port_row.count
                            );
                            n += 1;
                            if want.is_none() {
                                skipped += 1;
                            } else if want == Some((i16::MIN, i16::MIN)) {
                                invalid += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        skipped > 0,
        "the `return 0` (candidate is a repeat) arm was never taken in {n} cases"
    );
    assert!(
        invalid > 0,
        "the NEWMV INVALID_MV arm was never taken in {n} cases — that arm is \
         the one whose sentinel this port deliberately does NOT model as None"
    );
}

#[test]
fn clamp_mv2_matches_c() {
    let mut rng = Rng(0x5eed_0004);
    let mut clamped = 0;
    for _ in 0..4000 {
        let edges = rand_edges(&mut rng);
        let mv = Mv::new(
            rng.range(-30000, 30000) as i16,
            rng.range(-30000, 30000) as i16,
        );
        let want = cref::ref_rdopt_clamp_mv2(
            (mv.row, mv.col),
            cref::BlockEdges {
                left: edges.left,
                right: edges.right,
                top: edges.top,
                bottom: edges.bottom,
            },
        );
        let got = clamp_mv2(mv, edges);
        assert_eq!((got.row, got.col), want, "clamp_mv2({mv:?}, {edges:?})");
        if want != (mv.row, mv.col) {
            clamped += 1;
        }
    }
    assert!(
        clamped > 100,
        "only {clamped}/4000 MVs were actually clamped — the generator is not \
         reaching the limits, so this is close to an identity test"
    );
}

#[test]
fn clamp_and_check_mv_matches_c() {
    let mut rng = Rng(0x5eed_0005);
    let mut in_range = 0;
    let mut lowered = 0;
    for i in 0..6000 {
        let edges = rand_edges(&mut rng);
        let limits = rand_limits(&mut rng);
        let mv = Mv::new(rng.range(-8000, 8000) as i16, rng.range(-8000, 8000) as i16);
        let allow_hp = i % 2 == 0;
        let force_int = i % 5 == 0;
        let want = cref::ref_rdopt_clamp_and_check_mv(
            (mv.row, mv.col),
            allow_hp,
            force_int,
            cref::BlockEdges {
                left: edges.left,
                right: edges.right,
                top: edges.top,
                bottom: edges.bottom,
            },
            [
                limits.col_min,
                limits.col_max,
                limits.row_min,
                limits.row_max,
            ],
        );
        let got = clamp_and_check_mv(mv, allow_hp, force_int, edges, limits);
        assert_eq!(
            ((got.0.row, got.0.col), got.1),
            want,
            "clamp_and_check_mv({mv:?}, allow_hp={allow_hp}, force_int={force_int}, \
             {edges:?}, {limits:?})"
        );
        if want.1 {
            in_range += 1;
        }
        if want.0 != (mv.row, mv.col) {
            lowered += 1;
        }
    }
    assert!(
        in_range > 0 && in_range < 6000,
        "the in-range predicate was constant ({in_range}/6000)"
    );
    assert!(lowered > 0, "no MV was ever modified — vacuous");
}

#[test]
fn build_cur_mv_matches_c() {
    let mut rng = Rng(0x5eed_0006);
    let mut ok_count = 0;
    let mut n = 0;
    for iter in 0..200 {
        // Alternate a narrow span (exercises the global-MV / repeat arms) with
        // a wide one (pushes the MVs past the full-pel search limits, which is
        // the only way the `clamp_and_check_mv` verdict inside this function
        // becomes observable).
        let (port_row, mut c_row) = rand_row_span(&mut rng, if iter % 2 == 0 { 3 } else { 700 });
        let (gp, gc) = rand_global_mvs(&mut rng);
        c_row.global_mvs = gc;
        let edges = rand_edges(&mut rng);
        let limits = if iter % 2 == 0 {
            rand_limits(&mut rng)
        } else {
            let col_min = rng.range(-40, 0);
            let row_min = rng.range(-40, 0);
            FullMvLimits {
                col_min,
                col_max: col_min + rng.range(0, 80),
                row_min,
                row_max: row_min + rng.range(0, 80),
            }
        };
        let allow_hp = rng.next() % 2 == 0;
        let force_int = rng.next() % 4 == 0;
        for &(a, b) in &ref_pairs() {
            for m in all_modes() {
                let Some(mode) = PredMode::from_i32(m) else {
                    continue;
                };
                let is_comp = b > 0;
                // C's `has_second_ref` decides the loop count from mbmi, and
                // `get_single_mode(mode, 1)` must exist whenever it runs.
                if is_comp != mode.is_inter_compound() || !mode.is_inter() {
                    continue;
                }
                for ref_mv_idx in 0..3 {
                    for skip in [false, true] {
                        let seed = [
                            rng.range(-40, 40) as i16,
                            rng.range(-40, 40) as i16,
                            rng.range(-40, 40) as i16,
                            rng.range(-40, 40) as i16,
                        ];
                        let mut c_mv = [(seed[0], seed[1]), (seed[2], seed[3])];
                        let want_ok = cref::ref_rdopt_build_cur_mv(
                            &mut c_mv,
                            m,
                            (a, b),
                            ref_mv_idx,
                            skip,
                            &c_row,
                            allow_hp,
                            force_int,
                            cref::BlockEdges {
                                left: edges.left,
                                right: edges.right,
                                top: edges.top,
                                bottom: edges.bottom,
                            },
                            [
                                limits.col_min,
                                limits.col_max,
                                limits.row_min,
                                limits.row_max,
                            ],
                        );
                        let mut p_mv = [Mv::new(seed[0], seed[1]), Mv::new(seed[2], seed[3])];
                        let g0 = gp[a as usize];
                        let g1 = if b > 0 { gp[b as usize] } else { Mv::default() };
                        let got_ok = build_cur_mv(
                            &mut p_mv,
                            mode,
                            is_comp,
                            ref_mv_idx as usize,
                            skip,
                            &port_row,
                            [g0, g1],
                            allow_hp,
                            force_int,
                            edges,
                            limits,
                        );
                        assert_eq!(
                            got_ok, want_ok,
                            "build_cur_mv return (mode={m}, rf=({a},{b}), \
                             ref_mv_idx={ref_mv_idx}, skip={skip})"
                        );
                        assert_eq!(
                            [(p_mv[0].row, p_mv[0].col), (p_mv[1].row, p_mv[1].col)],
                            c_mv,
                            "build_cur_mv cur_mv (mode={m}, rf=({a},{b}), \
                             ref_mv_idx={ref_mv_idx}, skip={skip}) — this \
                             compares the PARTIAL writes on the failing path too"
                        );
                        n += 1;
                        ok_count += usize::from(want_ok);
                    }
                }
            }
        }
    }
    assert!(
        ok_count > 0 && ok_count < n,
        "build_cur_mv was constant ({ok_count}/{n} succeeded)"
    );
}

#[test]
fn get_drl_cost_matches_c() {
    let mut rng = Rng(0x5eed_0007);
    let mut nonzero = 0;
    let mut n = 0;
    for _ in 0..300 {
        let (port_row, c_row) = rand_row(&mut rng);
        let mut costs = [[0i32; 2]; 3];
        for row in &mut costs {
            for v in row {
                *v = rng.cost();
            }
        }
        for m in all_modes() {
            let Some(mode) = PredMode::from_i32(m) else {
                continue;
            };
            for ref_mv_idx in 0..3 {
                let want = cref::ref_rdopt_get_drl_cost(m, ref_mv_idx, (1, -1), &c_row, &costs);
                let got = get_drl_cost(mode, ref_mv_idx as usize, &port_row, &costs);
                assert_eq!(
                    got, want,
                    "get_drl_cost(mode={m}, ref_mv_idx={ref_mv_idx}, count={})",
                    port_row.count
                );
                nonzero += usize::from(want != 0);
                n += 1;
            }
        }
    }
    assert!(
        nonzero > 0 && nonzero < n,
        "get_drl_cost was always {} — a `return 0` stub would pass",
        if nonzero == 0 { "zero" } else { "nonzero" }
    );
}

#[test]
fn is_single_newmv_valid_matches_c() {
    let mut rng = Rng(0x5eed_0008);
    let mut falses = 0;
    let mut n = 0;
    // 40 rather than a few hundred: each C call allocates and zeroes a
    // `HandleInterModeArgs`, which is tens of KB, so the oracle dominates.
    for _ in 0..40 {
        let mut valid = [[0u8; 8]; 3];
        for row in &mut valid {
            for v in row.iter_mut() {
                *v = u8::from(rng.next() % 3 != 0);
            }
        }
        for &(a, b) in &ref_pairs() {
            for m in all_modes() {
                let Some(mode) = PredMode::from_i32(m) else {
                    continue;
                };
                if !mode.is_inter() {
                    continue;
                }
                // C reads single_newmv_valid[..][ref_frame[ref_idx]] only when
                // that slot's single mode is NEWMV, which for a single
                // reference never happens at ref_idx 1.
                if b <= 0 && mode.is_inter_compound() {
                    continue;
                }
                for ref_mv_idx in 0..3usize {
                    let want =
                        cref::ref_rdopt_is_single_newmv_valid(m, (a, b), ref_mv_idx as i32, &valid);
                    let row: [bool; 8] = std::array::from_fn(|i| valid[ref_mv_idx][i] != 0);
                    let got = is_single_newmv_valid(mode, [a, b], &row);
                    assert_eq!(
                        got, want,
                        "is_single_newmv_valid(mode={m}, rf=({a},{b}), idx={ref_mv_idx})"
                    );
                    falses += usize::from(!want);
                    n += 1;
                }
            }
        }
    }
    assert!(
        falses > 0 && falses < n,
        "constant answer ({falses}/{n} false)"
    );
}

#[test]
fn skip_nearest_near_matches_c() {
    let mut rng = Rng(0x5eed_0009);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..800 {
        let (port_row, c_row) = rand_row(&mut rng);
        let left = rng.next() % 4 != 0;
        let up = rng.next() % 4 != 0;
        for m in all_modes() {
            let Some(mode) = PredMode::from_i32(m) else {
                continue;
            };
            for best in all_modes() {
                let best_mode = PredMode::from_i32(best);
                let want = cref::ref_rdopt_skip_nearest_near_mv_using_refmv_weight(
                    m,
                    1,
                    best,
                    left,
                    up,
                    c_row.count,
                    &c_row.weight,
                );
                let got =
                    skip_nearest_near_mv_using_refmv_weight(mode, best_mode, left, up, &port_row);
                assert_eq!(
                    got, want,
                    "skip_nearest_near_mv_using_refmv_weight(mode={m}, best={best}, \
                     left={left}, up={up}, count={})",
                    port_row.count
                );
                trues += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}
