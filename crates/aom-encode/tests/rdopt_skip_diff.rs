//! Differential harness for the mode / reference SKIP MASK layer of libaom's
//! inter RD brain (`av1/encoder/rdopt.c`) — the port in
//! `aom_encode::rdopt_skip`.
//!
//! **Tier 1c**, the same as `rdopt_mv_diff.rs`: every function here is
//! `static` in C, so the oracle is `crates/aom-sys-ref/shim/rdopt_shim.c`,
//! which compiles libaom's own rdopt.c into the shim archive rather than
//! transcribing it. `rdopt_mv_diff::rdopt_shim_tu_agrees_with_archive` is the
//! measurement backing that tier for both files; read the shim header for the
//! argument.
//!
//! | test | C function (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | `default_skip_mask_matches_c` | `default_skip_mask` `:4050` |
//! | `disable_reference_matches_c` | `disable_reference` `:4018` + `_except_altref` `:4026` |
//! | `mask_says_skip_matches_c` | `mask_says_skip` `:4571` |
//! | `match_ref_frame_pair_matches_c` | `:4634` |
//! | `ref_match_found_in_nb_blocks_matches_c` | `:2465` |
//! | `find_ref_match_in_nbs_matches_c` | `:2482` + `:2504` |
//! | `match_ref_frame_matches_c` | `:5048` |
//! | `compound_skip_using_neighbor_refs_matches_c` | `:5062` |
//! | `skip_compound_using_best_single_mode_ref_matches_c` | `:5102` |
//! | `is_ref_frame_used_by_compound_ref_matches_c` | `:4300` |
//! | `is_ref_frame_used_in_cache_matches_c` | `:4313` |
//! | `fetch_picked_ref_frames_mask_matches_c` | `:4613` |
//! | `find_top_ref_matches_c` | `:5180` + `compare_int64` `:5134` |
//! | `in_single_ref_cutoff_matches_c` | `:5197` |
//! | `inter_modes_info_sort_matches_c` | `:502` + `compare_rd_idx_pair` `:485` |
//!
//! Every boolean test counts how often the ORACLE answered `true` and asserts
//! the count is strictly between 0 and n, so a port that returns a constant
//! fails even where it happens to agree.

mod common;
use common::Rng;

use aom_encode::rdopt_mv::{PredMode, REF_FRAMES};
use aom_encode::rdopt_skip::{
    ModeSkipMask, NONE_FRAME, NbDir, NbMi, RefSet, compound_skip_using_neighbor_refs,
    fetch_picked_ref_frames_mask, find_ref_match_in_nbs, find_top_ref, in_single_ref_cutoff,
    inter_modes_info_sort, is_ref_frame_used_by_compound_ref, is_ref_frame_used_in_cache,
    match_ref_frame, match_ref_frame_pair, ref_match_found_in_nb_blocks,
    skip_compound_using_best_single_mode_ref,
};
use aom_sys_ref as cref;

/// `BLOCK_SIZES_ALL`, minus the 1:4 / 4:1 shapes' indices being irrelevant —
/// the walk only reads `mi_size_{wide,high}`, so every bsize is in scope.
const N_BSIZE: usize = 22;

fn to_c(mask: &ModeSkipMask) -> cref::ModeSkipMaskFlat {
    let mut f = cref::ModeSkipMaskFlat {
        pred_modes: mask.pred_modes,
        ..cref::ModeSkipMaskFlat::default()
    };
    for (i, row) in mask.ref_combo.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            f.ref_combo[i * (REF_FRAMES + 1) + j] = u8::from(v);
        }
    }
    f
}

fn assert_same(port: &ModeSkipMask, c: &cref::ModeSkipMaskFlat, what: &str) {
    let got = to_c(port);
    assert_eq!(
        got.pred_modes, c.pred_modes,
        "{what}: pred_modes differ (C 0xa5-poisons the struct first, so an \
         unwritten field reads 0xa5a5a5a5 rather than a plausible 0)"
    );
    assert_eq!(got.ref_combo, c.ref_combo, "{what}: ref_combo differs");
}

/// Every reference pair the mask is indexed by: `ref1` in `0..REF_FRAMES`
/// (INTRA_FRAME included — it has a mask row) and `ref2` in
/// `NONE_FRAME..REF_FRAMES`.
fn mask_pairs() -> Vec<(i32, i32)> {
    let mut v = Vec::new();
    for a in 0..REF_FRAMES as i32 {
        for b in NONE_FRAME..REF_FRAMES as i32 {
            v.push((a, b));
        }
    }
    v
}

#[test]
fn default_skip_mask_matches_c() {
    // C's REF_SET enum order: FULL, REDUCED, REALTIME.
    for (i, set) in [RefSet::Full, RefSet::Reduced, RefSet::RealTime]
        .into_iter()
        .enumerate()
    {
        let want = cref::ref_rdopt_default_skip_mask(i as i32);
        let got = ModeSkipMask::default_for(set);
        assert_same(&got, &want, &format!("default_skip_mask({set:?})"));
    }
    // Non-vacuity: the three arms must not all produce the same mask, or the
    // test would pass against a port that ignored `ref_set`.
    let full = cref::ref_rdopt_default_skip_mask(0);
    let reduced = cref::ref_rdopt_default_skip_mask(1);
    let realtime = cref::ref_rdopt_default_skip_mask(2);
    assert_ne!(full.ref_combo, reduced.ref_combo);
    assert_ne!(reduced.ref_combo, realtime.ref_combo);
}

#[test]
fn disable_reference_matches_c() {
    for r in 0..REF_FRAMES as i32 {
        for base in [RefSet::Full, RefSet::Reduced] {
            let mut got = ModeSkipMask::default_for(base);
            let mut want = cref::ref_rdopt_default_skip_mask(match base {
                RefSet::Full => 0,
                RefSet::Reduced => 1,
                RefSet::RealTime => 2,
            });
            got.disable_reference(r);
            cref::ref_rdopt_disable_reference(r, &mut want);
            assert_same(&got, &want, &format!("disable_reference({r}) on {base:?}"));
        }
    }
    for base in [RefSet::Full, RefSet::Reduced, RefSet::RealTime] {
        let mut got = ModeSkipMask::default_for(base);
        let mut want = cref::ref_rdopt_default_skip_mask(match base {
            RefSet::Full => 0,
            RefSet::Reduced => 1,
            RefSet::RealTime => 2,
        });
        got.disable_inter_references_except_altref();
        cref::ref_rdopt_disable_inter_references_except_altref(&mut want);
        assert_same(
            &got,
            &want,
            &format!("disable_inter_references_except_altref on {base:?}"),
        );
        // ALTREF's own row must survive — that is the whole point of the name.
        assert!(
            !got.ref_combo[7].iter().all(|&v| v),
            "ALTREF_FRAME's row was disabled too"
        );
    }
}

#[test]
fn mask_says_skip_matches_c() {
    let mut rng = Rng(0x5eed_0011);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..40 {
        let mut mask = ModeSkipMask::default_for(if rng.next() % 2 == 0 {
            RefSet::Full
        } else {
            RefSet::Reduced
        });
        for pm in &mut mask.pred_modes {
            *pm = (rng.next() % (1 << 25)) as u32;
        }
        for row in &mut mask.ref_combo {
            for v in row.iter_mut() {
                *v = rng.next() % 3 == 0;
            }
        }
        let c = to_c(&mask);
        for (a, b) in mask_pairs() {
            for m in 0..25 {
                let mode = PredMode::from_i32(m).unwrap();
                let want = cref::ref_rdopt_mask_says_skip(&c, (a, b), m);
                let got = mask.says_skip([a, b], mode);
                assert_eq!(got, want, "mask_says_skip(rf=({a},{b}), mode={m})");
                trues += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn match_ref_frame_pair_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    for a0 in -1..8 {
        for a1 in -1..8 {
            for b0 in -1..8 {
                for b1 in -1..8 {
                    let want = cref::ref_rdopt_match_ref_frame_pair((a0, a1), (b0, b1));
                    let got = match_ref_frame_pair([a0, a1], [b0, b1]);
                    assert_eq!(got, want, "match_ref_frame_pair(({a0},{a1}), ({b0},{b1}))");
                    trues += usize::from(want);
                    n += 1;
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn ref_match_found_in_nb_blocks_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    for c0 in -1..8 {
        for c1 in -1..8 {
            for n0 in -1..8 {
                for n1 in -1..8 {
                    let want = cref::ref_rdopt_ref_match_found_in_nb_blocks((c0, c1), (n0, n1));
                    let got = ref_match_found_in_nb_blocks([c0, c1], [n0, n1]);
                    assert_eq!(
                        got, want,
                        "ref_match_found_in_nb_blocks(cur=({c0},{c1}), nb=({n0},{n1}))"
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
fn match_ref_frame_matches_c() {
    let mut moved = 0;
    let mut n = 0;
    for nb0 in -1..8 {
        for nb1 in -1..8 {
            for intrabc in [false, true] {
                for rf0 in -1..8 {
                    for rf1 in -1..8 {
                        // The accumulator is in/out; seed it non-empty half the
                        // time so a port that ASSIGNS instead of ORing fails.
                        for seed in [[0, 0], [1, 0], [0, 1]] {
                            let mut want = seed;
                            cref::ref_rdopt_match_ref_frame(
                                (nb0, nb1),
                                intrabc,
                                (rf0, rf1),
                                &mut want,
                            );
                            let mut got = [seed[0] != 0, seed[1] != 0];
                            match_ref_frame(
                                NbMi {
                                    ref_frame: [nb0, nb1],
                                    bsize: 0,
                                    use_intrabc: intrabc,
                                },
                                [rf0, rf1],
                                &mut got,
                            );
                            assert_eq!(
                                [i32::from(got[0]), i32::from(got[1])],
                                want,
                                "match_ref_frame(nb=({nb0},{nb1}), intrabc={intrabc}, \
                                 rf=({rf0},{rf1}), seed={seed:?})"
                            );
                            if want != seed {
                                moved += 1;
                            }
                            n += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        moved > 0 && moved < n,
        "match_ref_frame never fired ({moved}/{n})"
    );
}

#[test]
fn compound_skip_using_neighbor_refs_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    for m in 17..25 {
        let mode = PredMode::from_i32(m).unwrap();
        for prune in 0..4 {
            for left_av in [false, true] {
                for up_av in [false, true] {
                    for rf in [(1, 5), (1, 7), (4, 6), (1, 2)] {
                        for lrf in [(1, -1), (5, -1), (1, 5), (0, -1)] {
                            for arf in [(7, -1), (1, -1), (4, 6), (0, -1)] {
                                let want = cref::ref_rdopt_compound_skip_using_neighbor_refs(
                                    m,
                                    rf,
                                    prune,
                                    left_av,
                                    up_av,
                                    lrf,
                                    arf,
                                    (false, false),
                                );
                                let nb = |r: (i32, i32)| NbMi {
                                    ref_frame: [r.0, r.1],
                                    bsize: 0,
                                    use_intrabc: false,
                                };
                                let got = compound_skip_using_neighbor_refs(
                                    mode,
                                    [rf.0, rf.1],
                                    prune,
                                    left_av.then(|| nb(lrf)),
                                    up_av.then(|| nb(arf)),
                                );
                                assert_eq!(
                                    got, want,
                                    "compound_skip_using_neighbor_refs(mode={m}, rf={rf:?}, \
                                     prune={prune}, left={left_av}/{lrf:?}, up={up_av}/{arf:?})"
                                );
                                trues += usize::from(want);
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn skip_compound_using_best_single_mode_ref_matches_c() {
    let mut rng = Rng(0x5eed_0012);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..200 {
        let mut c_best = [0i32; 8];
        let mut p_best: [Option<PredMode>; 8] = [None; 8];
        for i in 0..8 {
            // 25 == MB_MODE_COUNT, C's "no best single mode yet".
            let v = rng.range(13, 26);
            c_best[i] = v;
            p_best[i] = PredMode::from_i32(v);
        }
        for m in 17..25 {
            let mode = PredMode::from_i32(m).unwrap();
            for prune in 0..3 {
                for rf in [(1, 5), (1, 7), (4, 6), (2, 5)] {
                    let want = cref::ref_rdopt_skip_compound_using_best_single_mode_ref(
                        m, rf, &c_best, prune,
                    );
                    let got = skip_compound_using_best_single_mode_ref(
                        mode,
                        [rf.0, rf.1],
                        &p_best,
                        prune,
                    );
                    assert_eq!(
                        got, want,
                        "skip_compound_using_best_single_mode_ref(mode={m}, rf={rf:?}, \
                         prune={prune}, best={c_best:?})"
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
fn is_ref_frame_used_by_compound_ref_matches_c() {
    let mut rng = Rng(0x5eed_0013);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..600 {
        let skip_mask = (rng.next() & 0x1fff_ffff) as i32;
        for r in 0..8 {
            let want = cref::ref_rdopt_is_ref_frame_used_by_compound_ref(r, skip_mask);
            let got = is_ref_frame_used_by_compound_ref(r, skip_mask);
            assert_eq!(
                got, want,
                "is_ref_frame_used_by_compound_ref({r}, mask={skip_mask:#x}) — a wrong \
                 ref_frame_map row makes this answer about a different pair"
            );
            trues += usize::from(want);
            n += 1;
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn is_ref_frame_used_in_cache_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    // 0..8 is a single reference; 8..29 is a compound ref_frame_type.
    for r in 0..29 {
        for cache in [
            None,
            Some((1, -1)),
            Some((1, 5)),
            Some((4, 7)),
            Some((0, -1)),
        ] {
            let want = cref::ref_rdopt_is_ref_frame_used_in_cache(r, cache);
            let got = is_ref_frame_used_in_cache(r, cache.map(|c| [c.0, c.1]));
            assert_eq!(got, want, "is_ref_frame_used_in_cache({r}, {cache:?})");
            trues += usize::from(want);
            n += 1;
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn fetch_picked_ref_frames_mask_matches_c() {
    let mut rng = Rng(0x5eed_0014);
    let mut nonzero = 0;
    let mut n = 0;
    for _ in 0..40 {
        let picked: Vec<i32> = (0..32 * 32).map(|_| (rng.next() & 0xff) as i32).collect();
        let arr: [i32; 32 * 32] = picked.clone().try_into().unwrap();
        for mib_size in [16, 32] {
            for bsize in 0..N_BSIZE {
                let mi_row = rng.range(0, 64);
                let mi_col = rng.range(0, 64);
                // Only sweep placements whose footprint stays inside the 32x32
                // per-mi array; C reads out of bounds otherwise and the
                // encoder never places a block there.
                let want_fits = {
                    let m = mib_size - 1;
                    let (r, c) = (mi_row & m, mi_col & m);
                    r + mi_h(bsize) <= 32 && c + mi_w(bsize) <= 32
                };
                if !want_fits {
                    continue;
                }
                let want = cref::ref_rdopt_fetch_picked_ref_frames_mask(
                    mi_row,
                    mi_col,
                    bsize as i32,
                    mib_size,
                    &picked,
                );
                let got = fetch_picked_ref_frames_mask(mi_row, mi_col, bsize, mib_size, &arr);
                assert_eq!(
                    got, want,
                    "fetch_picked_ref_frames_mask(row={mi_row}, col={mi_col}, \
                     bsize={bsize}, mib={mib_size})"
                );
                nonzero += usize::from(want != 0);
                n += 1;
            }
        }
    }
    assert!(n > 200, "only {n} placements were swept");
    assert!(nonzero > 0 && nonzero <= n, "always zero — vacuous inputs");
}

/// `mi_size_wide` / `mi_size_high` (`common_data.h`), duplicated here because
/// the port's copies are `pub(crate)`. Only used to keep the sweep inside the
/// bounds C reads.
fn mi_w(bsize: usize) -> i32 {
    const W: [i32; 22] = [
        1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
    ];
    W[bsize]
}
fn mi_h(bsize: usize) -> i32 {
    const H: [i32; 22] = [
        1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
    ];
    H[bsize]
}

#[test]
fn find_ref_match_in_nbs_matches_c() {
    let mut rng = Rng(0x5eed_0015);
    let (rows, cols) = (24usize, 24usize);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..120 {
        // A grid of same-size blocks in each cell, so `mi_size_*` stepping is
        // exercised with a real (and varying) stride.
        let grid_bsize: Vec<i32> = (0..rows * cols).map(|_| rng.range(0, 16)).collect();
        let grid_rf: Vec<i32> = (0..rows * cols * 2)
            .map(|i| {
                if i % 2 == 0 {
                    rng.range(-1, 8)
                } else {
                    rng.range(-1, 8)
                }
            })
            .collect();
        for above in [true, false] {
            for avail in [true, false] {
                let mi_row = rng.range(4, 12) as usize;
                let mi_col = rng.range(4, 12) as usize;
                let width = rng.range(1, 9);
                let height = rng.range(1, 9);
                let cur_rf = (rng.range(-1, 8), rng.range(-1, 8));
                let total_mi = rng.range(8, 24);
                let want = cref::ref_rdopt_find_ref_match_in_nbs(
                    above,
                    total_mi,
                    rows,
                    cols,
                    mi_row,
                    mi_col,
                    width,
                    height,
                    if above { avail } else { true },
                    if above { true } else { avail },
                    &grid_rf,
                    &grid_bsize,
                    cur_rf,
                );
                // C's above walk reads `xd->mi - mi_col - mi_stride` + col, and
                // the left walk `xd->mi - 1 - mi_row * mi_stride` + row *
                // stride; both land in the row above / column left of the
                // block. Mirror that indexing here.
                let cell = |pos: i32| -> NbMi {
                    let idx = if above {
                        (mi_row - 1) * cols + pos as usize
                    } else {
                        pos as usize * cols + (mi_col - 1)
                    };
                    NbMi {
                        ref_frame: [grid_rf[2 * idx], grid_rf[2 * idx + 1]],
                        bsize: grid_bsize[idx] as usize,
                        use_intrabc: false,
                    }
                };
                let got = find_ref_match_in_nbs(
                    if above { NbDir::Above } else { NbDir::Left },
                    total_mi,
                    if above { mi_col as i32 } else { mi_row as i32 },
                    if above { width } else { height },
                    avail,
                    [cur_rf.0, cur_rf.1],
                    cell,
                );
                assert_eq!(
                    got,
                    want,
                    "find_ref_match_in_{}_nbs(total_mi={total_mi}, mi=({mi_row},{mi_col}), \
                     w={width}, h={height}, avail={avail}, cur={cur_rf:?})",
                    if above { "above" } else { "left" }
                );
                trues += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn find_top_ref_matches_c() {
    let mut rng = Rng(0x5eed_0016);
    let mut finite = 0;
    for i in 0..4000 {
        let mut c = [i64::MAX; 8];
        for slot in c.iter_mut().skip(1) {
            // A quarter of the slots stay INT64_MAX (reference not measured),
            // and every so often the WHOLE row does — which is the arm where
            // C skips the 110% scaling entirely.
            *slot = if rng.next() % 4 == 0 || i % 97 == 0 {
                i64::MAX
            } else {
                rng.range(1, 1 << 30) as i64
            };
        }
        let mut want = c;
        cref::ref_rdopt_find_top_ref(&mut want);
        let mut got = c;
        find_top_ref(&mut got);
        assert_eq!(got, want, "find_top_ref({c:?})");
        if want[0] != i64::MAX {
            finite += 1;
        }
    }
    assert!(
        finite > 100 && finite < 4000,
        "the all-INT64_MAX arm and the scaled arm were not both reached ({finite}/4000)"
    );
}

#[test]
fn in_single_ref_cutoff_matches_c() {
    let mut rng = Rng(0x5eed_0017);
    let mut trues = 0;
    let mut n = 0;
    for _ in 0..500 {
        let mut rd = [i64::MAX; 8];
        for slot in rd.iter_mut() {
            *slot = if rng.next() % 5 == 0 {
                i64::MAX
            } else {
                rng.range(1, 1 << 20) as i64
            };
        }
        for f1 in 0..8 {
            for f2 in 1..8 {
                let want = cref::ref_rdopt_in_single_ref_cutoff(&rd, f1, f2);
                let got = in_single_ref_cutoff(&rd, f1, f2);
                assert_eq!(got, want, "in_single_ref_cutoff({rd:?}, {f1}, {f2})");
                trues += usize::from(want);
                n += 1;
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
}

#[test]
fn inter_modes_info_sort_matches_c() {
    let mut rng = Rng(0x5eed_0018);
    let mut ties = 0;
    for _ in 0..400 {
        let num = (rng.next() % 40) as usize;
        // Draw from a SMALL value set so equal RDs are common: the tie-break on
        // `idx` (aomedia:2928) is the only part of this a naive port gets
        // wrong, and it is invisible without duplicates.
        let est: Vec<i64> = (0..num).map(|_| rng.range(0, 6) as i64).collect();
        let want = cref::ref_rdopt_inter_modes_info_sort(&est);
        let got = inter_modes_info_sort(&est);
        assert_eq!(
            got.iter()
                .map(|&(i, rd)| (i as i32, rd))
                .collect::<Vec<_>>(),
            want,
            "inter_modes_info_sort({est:?})"
        );
        let mut sorted = est.clone();
        sorted.sort_unstable();
        ties += sorted.windows(2).filter(|w| w[0] == w[1]).count();
    }
    assert!(
        ties > 100,
        "only {ties} equal-RD adjacencies were generated — the tie-break the \
         sort exists for was barely exercised"
    );
}

#[test]
fn compare_int64_matches_c() {
    let vals = [
        i64::MIN,
        -1 << 40,
        -1,
        0,
        1,
        1 << 40,
        i64::MAX - 1,
        i64::MAX,
    ];
    for &a in &vals {
        for &b in &vals {
            let want = cref::ref_rdopt_compare_int64(a, b);
            let got = match a.cmp(&b) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            };
            assert_eq!(
                got, want,
                "compare_int64({a}, {b}) — C compares the values, NOT their \
                 difference, so it is safe at the i64 extremes and `Ord` agrees"
            );
        }
    }
}
