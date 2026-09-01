//! Differential harness for the MASTER MODE/REFERENCE GATE of libaom's inter
//! RD brain (`inter_mode_search_order_independent_skip`,
//! `av1/encoder/rdopt.c:4643`) and three helpers around it — the port in
//! `aom_encode::rdopt_gate`.
//!
//! **Tier 1c** (all `static`; libaom's own rdopt.c compiled into the shim
//! archive). This is the predicate that decides which `(mode, reference pair)`
//! candidates are searched at all, so a divergence here changes the candidate
//! set before any RD is computed — which is exactly the class of bug a byte
//! gate finds ten thousand blocks later.
//!
//! | test | C function (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | `inter_mode_search_order_independent_skip_matches_c` | `:4643` |
//! | `prune_ref_frame_matches_c` | `:4284` |
//! | `record_best_compound_matches_c` | `:5440` |
//! | `init_mbmi_matches_c` | `:4795` |
//!
//! # One dependency is deliberately pinned off
//!
//! `prune_ref_frame` calls `prune_ref_by_selective_ref_frame`, which lives in
//! `av1/encoder/rdopt.**h**` — a different translation unit, not part of the
//! rdopt.c surface, and not ported. The shim pins
//! `sf.inter_sf.selective_ref_frame = 0` and `prune_comp_ref_frames = 0` so
//! C's own call returns 0, and the port takes that result as an argument. The
//! `cpi->prune_ref_frame_mask` half IS covered. **That arm of `prune_ref_frame`
//! is therefore UNTESTED here and is named as missing.**

mod common;
use common::Rng;

use aom_encode::inter_costs::{
    DRL_MODE_CONTEXTS, GLOBALMV_MODE_CONTEXTS, INTRA_INTER_CONTEXTS, InterModeCosts,
    NEWMV_MODE_CONTEXTS, REF_CONTEXTS, REFMV_MODE_CONTEXTS, SINGLE_REF_BITS,
};
use aom_encode::rdopt_gate::{
    FLAG_SKIP_INTRA_LOWVAR, ModeSkipCtx, ModeSkipVerdict, REFERENCE_MODES, init_mbmi,
    inter_mode_search_order_independent_skip, prune_ref_frame, record_best_compound,
};
use aom_encode::rdopt_mv::{PredMode, ref_frame_type};
use aom_encode::rdopt_skip::{ModeSkipMask, RefSet};
use aom_sys_ref as cref;

fn to_c_mask(mask: &ModeSkipMask) -> cref::ModeSkipMaskFlat {
    let mut f = cref::ModeSkipMaskFlat {
        pred_modes: mask.pred_modes,
        ..cref::ModeSkipMaskFlat::default()
    };
    for (i, row) in mask.ref_combo.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            f.ref_combo[i * 9 + j] = u8::from(v);
        }
    }
    f
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

/// A `mode_context` whose three sub-fields are in range — see
/// `rdopt_single_state_diff::rand_mode_context` for why a raw random value is
/// out of bounds in C as well as in the port.
fn rand_mode_context(rng: &mut Rng) -> i32 {
    let newmv = rng.range(0, NEWMV_MODE_CONTEXTS as i32);
    let globalmv = rng.range(0, GLOBALMV_MODE_CONTEXTS as i32);
    let refmv = rng.range(0, REFMV_MODE_CONTEXTS as i32);
    newmv | (globalmv << 3) | (refmv << 4)
}

/// The pairs `av1_ref_frame_type` has a row for, plus the intra and interintra
/// shapes the gate treats specially.
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
    let mut v: Vec<(i32, i32)> = (0..8).map(|r| (r, -1)).collect();
    for a in 1..5 {
        for b in 5..8 {
            v.push((a, b));
        }
    }
    v.extend_from_slice(&UNIDIR);
    v
}

#[test]
fn prune_ref_frame_matches_c() {
    let mut rng = Rng(0x5eed_0061);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..400 {
        let mask = (rng.next() & 0x1fff_ffff) as i32;
        for ref_type in 0..29 {
            let want = cref::ref_rdopt_prune_ref_frame(ref_type, mask);
            let got = prune_ref_frame(ref_type, mask, false);
            assert_eq!(got, want, "prune_ref_frame({ref_type}, mask={mask:#x})");
            trues += usize::from(want);
            n += 1;
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn record_best_compound_matches_c() {
    let mut rng = Rng(0x5eed_0062);
    assert_eq!(cref::ref_reference_modes(), REFERENCE_MODES);
    let mut moved = 0;
    let mut n = 0;
    for _ in 0..4000 {
        let seed: Vec<i64> = (0..REFERENCE_MODES)
            .map(|_| {
                if rng.next() % 4 == 0 {
                    i64::MAX
                } else {
                    rng.range(0, 1 << 28) as i64
                }
            })
            .collect();
        for reference_mode in 0..3 {
            for comp_pred in [false, true] {
                let rate = rng.range(0, 1 << 18);
                let dist = rng.range(0, 1 << 24) as i64;
                let rdmult = rng.range(1, 1 << 13);
                let compmode_cost = rng.range(0, 1 << 12);
                let mut want = seed.clone();
                cref::ref_rdopt_record_best_compound(
                    reference_mode,
                    rate,
                    dist,
                    comp_pred,
                    rdmult,
                    compmode_cost,
                    &mut want,
                );
                let mut got: [i64; REFERENCE_MODES] = seed.clone().try_into().unwrap();
                record_best_compound(
                    reference_mode as usize,
                    rate,
                    dist,
                    comp_pred,
                    rdmult,
                    compmode_cost,
                    &mut got,
                );
                assert_eq!(
                    got.to_vec(),
                    want,
                    "record_best_compound(ref_mode={reference_mode}, rate={rate}, \
                     dist={dist}, comp={comp_pred}, rdmult={rdmult}, \
                     compmode_cost={compmode_cost})"
                );
                if want != seed {
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
fn init_mbmi_matches_c() {
    for m in 0..25 {
        let mode = PredMode::from_i32(m).unwrap();
        for rf in ref_pairs() {
            // `set_default_interp_filters` reads cm->features.interp_filter;
            // the port does not model interp_filters, so only the ten fields
            // C's init writes directly are compared.
            let want = cref::ref_rdopt_init_mbmi(m, rf, 0);
            let got = init_mbmi(mode, [rf.0, rf.1]);
            assert_eq!(got.ref_mv_idx, want[0], "init_mbmi ref_mv_idx");
            assert_eq!(got.mode, want[1], "init_mbmi mode");
            assert_eq!(got.uv_mode, want[2], "init_mbmi uv_mode");
            assert_eq!(got.ref_frame[0], want[3], "init_mbmi ref_frame[0]");
            assert_eq!(got.ref_frame[1], want[4], "init_mbmi ref_frame[1]");
            assert_eq!(got.palette_size[0], want[5], "init_mbmi palette_size[0]");
            assert_eq!(got.palette_size[1], want[6], "init_mbmi palette_size[1]");
            assert_eq!(got.use_filter_intra, want[7], "init_mbmi use_filter_intra");
            assert_eq!(got.motion_mode, want[8], "init_mbmi motion_mode");
            assert_eq!(
                got.interintra_mode, want[9],
                "init_mbmi interintra_mode — C stores II_DC_PRED - 1 in an \
                 unsigned 1-byte enum, so it reads back as 255"
            );
        }
    }
    // The shim poisons the struct with 0x5a first, so these prove C WRITES
    // these fields rather than that both sides happened to start zeroed.
    let poisoned = cref::ref_rdopt_init_mbmi(16, (1, -1), 0);
    assert_eq!(poisoned[0], 0, "C left ref_mv_idx poisoned");
    assert_eq!(poisoned[8], 0, "C left motion_mode poisoned");
    assert_eq!(poisoned[9], 255, "C's interintra_mode sentinel is not 255");
}

#[test]
fn inter_mode_search_order_independent_skip_matches_c() {
    let mut rng = Rng(0x5eed_0063);
    let mut verdicts = std::collections::BTreeMap::new();
    let mut n = 0usize;
    let mut rd_written = 0usize;

    for iter in 0..500 {
        let costs = rand_costs(&mut rng);
        let mut mask = ModeSkipMask::default_for(if iter % 3 == 0 {
            RefSet::Reduced
        } else {
            RefSet::Full
        });
        for pm in &mut mask.pred_modes {
            *pm = if rng.next() % 3 == 0 {
                (rng.next() % (1 << 25)) as u32
            } else {
                0
            };
        }
        for row in &mut mask.ref_combo {
            for v in row.iter_mut() {
                *v = rng.next() % 5 == 0;
            }
        }
        let c_mask = to_c_mask(&mask);

        let use_cache = iter % 3 == 0;
        // The POINTER and the FLAG are separate: sweep a stale non-null cache
        // with the flag off, which only `is_ref_frame_used_in_cache` sees.
        let cache_ptr = iter % 2 == 0 || use_cache;
        let cache_mode = rng.range(0, 25);
        let cache_rf = if rng.next() % 2 == 0 {
            (rng.range(1, 8), -1)
        } else {
            (rng.range(1, 5), rng.range(5, 8))
        };
        let ctx_common = ModeSkipCtx {
            prune_ref_frame_mask: if rng.next() % 2 == 0 {
                (rng.next() & 0xffff) as i32
            } else {
                0
            },
            selective_prune: false,
            use_real_time_ref_set: false,
            skip_ref_frame_mask: (rng.next() & 0x1fff_ffff) as i32,
            mb_mode_cache: cache_ptr.then(|| {
                (
                    PredMode::from_i32(cache_mode).unwrap(),
                    [cache_rf.0, cache_rf.1],
                )
            }),
            use_mb_mode_cache: use_cache,
            best_rd_is_max: iter % 4 == 0,
            partition: rng.range(0, 4),
            must_find_valid_partition: iter % 5 == 0,
            prune_nearmv_using_neighbors: rng.range(0, 4),
            qindex: rng.range(0, 256),
            left: (iter % 7 != 0).then(|| [rng.range(-1, 8), rng.range(-1, 8)]),
            above: (iter % 8 != 0).then(|| [rng.range(-1, 8), rng.range(-1, 8)]),
            mode_search_skip_flags: if rng.next() % 2 == 0 {
                FLAG_SKIP_INTRA_LOWVAR
            } else {
                0
            },
            source_variance: rng.range(0, 200) as u32,
        };

        let mut base_rd = [i64::MAX; 25];
        for slot in base_rd.iter_mut() {
            if rng.next() % 2 == 0 {
                *slot = rng.range(1, 1 << 24) as i64;
            }
        }

        for rf in ref_pairs() {
            let ref_mv_count = rng.range(0, 4);
            let gm_wmtype = rng.range(0, 4);
            let mode_context = rand_mode_context(&mut rng);
            for m in 0..25 {
                let mode = PredMode::from_i32(m).unwrap();
                let c_ctx = cref::ModeSkipCtx {
                    prune_ref_frame_mask: ctx_common.prune_ref_frame_mask,
                    skip_ref_frame_mask: ctx_common.skip_ref_frame_mask,
                    use_mb_mode_cache: i32::from(use_cache),
                    cache_mode,
                    cache_rf0: cache_rf.0,
                    cache_rf1: cache_rf.1,
                    best_rd_is_max: i32::from(ctx_common.best_rd_is_max),
                    partition: ctx_common.partition,
                    must_find_valid_partition: i32::from(ctx_common.must_find_valid_partition),
                    prune_nearmv_using_neighbors: ctx_common.prune_nearmv_using_neighbors,
                    qindex: ctx_common.qindex,
                    left_available: i32::from(ctx_common.left.is_some()),
                    up_available: i32::from(ctx_common.above.is_some()),
                    left_rf0: ctx_common.left.map_or(0, |v| v[0]),
                    left_rf1: ctx_common.left.map_or(0, |v| v[1]),
                    above_rf0: ctx_common.above.map_or(0, |v| v[0]),
                    above_rf1: ctx_common.above.map_or(0, |v| v[1]),
                    mode_search_skip_flags: ctx_common.mode_search_skip_flags,
                    source_variance: ctx_common.source_variance as i32,
                    ref_mv_count,
                    gm_wmtype,
                    mode_context,
                    cache_ptr_nonnull: i32::from(cache_ptr),
                };
                let mut want_rd = base_rd;
                let want = cref::ref_rdopt_inter_mode_search_order_independent_skip(
                    &c_mask,
                    m,
                    rf,
                    &c_ctx,
                    &costs.newmv_mode_cost,
                    &costs.zeromv_mode_cost,
                    &costs.refmv_mode_cost,
                    &mut want_rd,
                );
                let mut got_rd = base_rd;
                let got = inter_mode_search_order_independent_skip(
                    &mask,
                    mode,
                    [rf.0, rf.1],
                    ref_frame_type([rf.0, rf.1]) as i32,
                    &ctx_common,
                    ref_mv_count as usize,
                    // TRANSLATION is wmtype 1.
                    gm_wmtype <= 1,
                    mode_context,
                    &costs,
                    &mut got_rd,
                );
                assert_eq!(
                    got.to_i32(),
                    want,
                    "inter_mode_search_order_independent_skip(mode={m}, rf={rf:?}, \
                     partition={}, cache={use_cache}/{cache_ptr}, best_rd_max={}, \
                     prune_nearmv={}, q={})",
                    ctx_common.partition,
                    ctx_common.best_rd_is_max,
                    ctx_common.prune_nearmv_using_neighbors,
                    ctx_common.qindex
                );
                assert_eq!(
                    got_rd, want_rd,
                    "the nested skip_repeated_mv's modelled_rd carry-across \
                     differs (mode={m}, rf={rf:?})"
                );
                if want_rd != base_rd {
                    rd_written += 1;
                }
                *verdicts.entry(want).or_insert(0usize) += 1;
                n += 1;
            }
        }
    }

    // All three verdicts must occur, and none may dominate: C's own comment
    // calls them "Case 1/2/3" and the 2 case (skip motion modes only) is the
    // one a port is most likely to collapse into 1.
    for v in [0, 1, 2] {
        let c = verdicts.get(&v).copied().unwrap_or(0);
        assert!(
            c > 0,
            "verdict {v} never occurred in {n} cases (saw {verdicts:?}) — the \
             tri-state return is not being exercised"
        );
    }
    assert!(
        rd_written > 0,
        "the nested skip_repeated_mv never wrote modelled_rd — its carry-across \
         side effect is untested"
    );
}
