//! Differential harness for the variance-based RD adjustment and two more
//! search-loop predicates (`av1/encoder/rdopt.c`) — the port in
//! `aom_encode::rdopt_var_rd`.
//!
//! **Tier 1c** (all `static`; the oracle is libaom's own rdopt.c compiled into
//! the shim archive — see `rdopt_mv_diff.rs`'s header).
//!
//! | test | C function (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | `get_variance_stats_matches_c` | `get_variance_stats` `:709` + `_hbd` `:624` |
//! | `adjust_cost_matches_c` | `:840` |
//! | `adjust_rdcost_matches_c` | `:796` |
//! | `inter_mode_compatible_skip_matches_c` | `:4581` |
//! | `ref_mv_idx_early_breakout_matches_c` | `:2216` |
//!
//! `get_variance_stats` is the one worth staring at: C's scratch buffer has a
//! row stride of `bw` while the copy loop writes `bw + 2` columns per row, so
//! the halo columns ALIAS the neighbouring rows. The obvious "clean" port with
//! a `bw + 2` stride disagrees on every block, and the bite proof below
//! confirms this harness sees that.

mod common;
use common::Rng;

use aom_encode::rdopt_mv::{MAX_REF_MV_SEARCH, PredMode, REF_FRAMES, RefMvRow};
use aom_encode::rdopt_var_rd::{
    AOM_TUNE_IQ, AOM_TUNE_SSIMULACRA2, AdjustGates, RdStatsCore, RefFrameDistanceInfo, adjust_cost,
    adjust_rdcost, get_variance_stats, inter_mode_compatible_skip, ref_mv_idx_early_breakout,
};
use aom_sys_ref as cref;

/// Block sizes to sweep. `get_variance_stats` reads `block_size_{wide,high}`
/// so every shape is in scope; 128-wide ones are included because that is
/// where the scratch buffer is nearly full.
const BSIZES: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
];
const BW: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BH: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

fn planes(rng: &mut Rng, stride: usize, bh: usize, maxval: i32) -> Vec<u16> {
    (0..stride * (bh + 8))
        .map(|_| rng.range(0, maxval) as u16)
        .collect()
}

#[test]
fn get_variance_stats_matches_c() {
    let mut rng = Rng(0x5eed_0051);
    let mut nonzero = 0;
    let mut n = 0;
    for &bsize in &BSIZES {
        for hbd in [false, true] {
            for _ in 0..6 {
                let maxval = if hbd { 1 << 12 } else { 1 << 8 };
                let (bw, bh) = (BW[bsize], BH[bsize]);
                let src_stride = bw + 7;
                let dst_stride = bw + 3;
                let src = planes(&mut rng, src_stride, bh, maxval);
                let dst = planes(&mut rng, dst_stride, bh, maxval);
                let want = cref::ref_rdopt_get_variance_stats(
                    bsize as i32,
                    hbd,
                    &src,
                    src_stride,
                    &dst,
                    dst_stride,
                );
                let got = get_variance_stats(bsize, &src, src_stride, &dst, dst_stride, hbd);
                assert_eq!(
                    got,
                    want,
                    "get_variance_stats{}(bsize={bsize} = {bw}x{bh})",
                    if hbd { "_hbd" } else { "" }
                );
                if want.0 != 0 || want.1 != 0 {
                    nonzero += 1;
                }
                n += 1;
            }
        }
    }
    assert!(
        nonzero > n / 2,
        "only {nonzero}/{n} cases produced a nonzero variance — the filter is \
         not seeing real detail"
    );
}

fn rand_gates(rng: &mut Rng, i: usize) -> (AdjustGates, cref::AdjustGates) {
    // Sweep all three arms: the IQ/SSIMULACRA2 bias, the sharpness-3 variance
    // path, and the two no-op paths.
    let tuning = match i % 4 {
        0 => AOM_TUNE_IQ,
        1 => AOM_TUNE_SSIMULACRA2,
        _ => 0,
    };
    let sharpness = if i % 3 == 0 { 3 } else { rng.range(0, 3) };
    let frame_is_intra = i % 5 == 0;
    // ARF_UPDATE / GF_UPDATE are the two update types frame_is_kf_gf_arf also
    // matches; C's FRAME_UPDATE_TYPE has KF=0, LF=1, GF=2, ARF=3, OVERLAY=4,
    // INTNL_OVERLAY=5, INTNL_ARF=6.
    let update_type = rng.range(0, 7);
    let rdmult = rng.range(1, 1 << 14);
    let kf_gf_arf = frame_is_intra || update_type == 2 || update_type == 3;
    (
        AdjustGates {
            tuning,
            sharpness,
            frame_is_kf_gf_arf: kf_gf_arf,
            rdmult,
        },
        cref::AdjustGates {
            tuning,
            sharpness,
            frame_is_intra,
            update_type,
            rdmult,
        },
    )
}

#[test]
fn adjust_cost_matches_c() {
    let mut rng = Rng(0x5eed_0052);
    let mut moved = 0;
    let mut var_arm = 0;
    let mut n = 0;
    for i in 0..800 {
        let bsize = BSIZES[(rng.next() as usize) % BSIZES.len()];
        let (bw, bh) = (BW[bsize], BH[bsize]);
        let hbd = i % 3 == 0;
        let maxval = if hbd { 1 << 12 } else { 1 << 8 };
        let src_stride = bw + 5;
        let dst_stride = bw + 2;
        let src = planes(&mut rng, src_stride, bh, maxval);
        let dst = planes(&mut rng, dst_stride, bh, maxval);
        let (pg, cg) = rand_gates(&mut rng, i);
        let is_inter = i % 2 == 0;
        let rd = rng.range(0, 1 << 28) as i64;
        let want = cref::ref_rdopt_adjust_cost(
            rd,
            is_inter,
            cg,
            bsize as i32,
            hbd,
            &src,
            src_stride,
            &dst,
            dst_stride,
        );
        let got = adjust_cost(
            rd, is_inter, pg, bsize, &src, src_stride, &dst, dst_stride, hbd,
        );
        assert_eq!(
            got, want,
            "adjust_cost(rd={rd}, inter={is_inter}, tuning={}, sharpness={}, \
             kf_gf_arf={}, bsize={bsize}, hbd={hbd})",
            pg.tuning, pg.sharpness, pg.frame_is_kf_gf_arf
        );
        if want != rd {
            moved += 1;
        }
        if pg.sharpness == 3
            && !pg.frame_is_kf_gf_arf
            && pg.tuning != AOM_TUNE_IQ
            && pg.tuning != AOM_TUNE_SSIMULACRA2
        {
            var_arm += 1;
        }
        n += 1;
    }
    assert!(
        moved > 0 && moved < n,
        "constant answer ({moved}/{n} moved)"
    );
    assert!(
        var_arm > 20,
        "only {var_arm} draws reached the sharpness-3 variance arm — the \
         IQ/SSIMULACRA2 arm would be the only thing tested"
    );
}

#[test]
fn adjust_rdcost_matches_c() {
    let mut rng = Rng(0x5eed_0053);
    let mut moved = 0;
    let mut n = 0;
    for i in 0..800 {
        let bsize = BSIZES[(rng.next() as usize) % BSIZES.len()];
        let (bw, bh) = (BW[bsize], BH[bsize]);
        let hbd = i % 4 == 0;
        let maxval = if hbd { 1 << 12 } else { 1 << 8 };
        let src_stride = bw + 6;
        let dst_stride = bw + 1;
        let src = planes(&mut rng, src_stride, bh, maxval);
        let dst = planes(&mut rng, dst_stride, bh, maxval);
        let (pg, cg) = rand_gates(&mut rng, i);
        let is_inter = i % 2 == 1;
        let seed = RdStatsCore {
            rate: rng.range(0, 1 << 20),
            dist: rng.range(0, 1 << 26) as i64,
            rdcost: rng.range(0, 1 << 28) as i64,
        };
        let mut want = [i64::from(seed.rate), seed.dist, seed.rdcost];
        cref::ref_rdopt_adjust_rdcost(
            &mut want,
            is_inter,
            cg,
            bsize as i32,
            hbd,
            &src,
            src_stride,
            &dst,
            dst_stride,
        );
        let mut got = seed;
        adjust_rdcost(
            &mut got, is_inter, pg, bsize, &src, src_stride, &dst, dst_stride, hbd,
        );
        assert_eq!(
            [i64::from(got.rate), got.dist, got.rdcost],
            want,
            "adjust_rdcost({seed:?}, inter={is_inter}, tuning={}, sharpness={}, \
             kf_gf_arf={}, bsize={bsize}, hbd={hbd})",
            pg.tuning,
            pg.sharpness,
            pg.frame_is_kf_gf_arf
        );
        if want != [i64::from(seed.rate), seed.dist, seed.rdcost] {
            moved += 1;
        }
        n += 1;
    }
    assert!(
        moved > 0 && moved < n,
        "constant answer ({moved}/{n} moved)"
    );
}

#[test]
fn inter_mode_compatible_skip_matches_c() {
    let mut trues = 0;
    let mut n = 0;
    // ref_frames[1] == INTRA_FRAME (0) is the interintra case; -1 is single.
    let pairs = [
        (1, -1),
        (4, -1),
        (1, 0),
        (4, 0),
        (7, 0),
        (1, 5),
        (1, 7),
        (4, 6),
        (2, 5),
    ];
    for bsize in BSIZES {
        for &rf in &pairs {
            for m in 13..25 {
                let mode = PredMode::from_i32(m).unwrap();
                for flags in [0i32, 0x7f, 0x10, 0x41] {
                    for frame_is_intra in [false, true] {
                        for reference_mode in 0..3 {
                            for seg in [false, true] {
                                let want = cref::ref_rdopt_inter_mode_compatible_skip(
                                    bsize as i32,
                                    m,
                                    rf,
                                    flags,
                                    frame_is_intra,
                                    reference_mode,
                                    seg,
                                    seg,
                                );
                                let got = inter_mode_compatible_skip(
                                    bsize,
                                    mode,
                                    [rf.0, rf.1],
                                    flags,
                                    frame_is_intra,
                                    reference_mode,
                                    seg,
                                );
                                assert_eq!(
                                    got, want,
                                    "inter_mode_compatible_skip(bsize={bsize}, mode={m}, \
                                     rf={rf:?}, flags={flags:#x}, intra={frame_is_intra}, \
                                     ref_mode={reference_mode}, seg={seg})"
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
fn ref_mv_idx_early_breakout_matches_c() {
    let mut rng = Rng(0x5eed_0054);
    let mut trues = 0;
    let mut n = 0;
    let mut idx_written = 0;
    for _ in 0..600 {
        let mut row = RefMvRow {
            count: (rng.next() % 9) as usize,
            ..RefMvRow::default()
        };
        let mut c_row = cref::RefMvRow {
            count: row.count as i32,
            ..cref::RefMvRow::default()
        };
        for i in 0..8 {
            // Straddle REF_CAT_LEVEL so the weight gate is not constant.
            let w = rng.range(400, 900) as u16;
            row.weight[i] = w;
            c_row.weight[i] = w;
        }
        let mut costs = [[0i32; 2]; 3];
        for r in &mut costs {
            for v in r {
                *v = rng.cost();
            }
        }
        let mut c_valid = [[0u8; 8]; MAX_REF_MV_SEARCH];
        let mut p_valid = [[false; REF_FRAMES]; MAX_REF_MV_SEARCH];
        for i in 0..MAX_REF_MV_SEARCH {
            for r in 0..8 {
                let v = rng.next() % 3 != 0;
                c_valid[i][r] = u8::from(v);
                p_valid[i][r] = v;
            }
        }
        let dist = RefFrameDistanceInfo {
            nearest_past_ref: rng.range(1, 8),
            nearest_future_ref: rng.range(1, 8),
        };
        let ref_frame_cost = rng.cost();
        let single_comp_cost = rng.cost();
        let rdmult = rng.range(1, 1 << 14);
        let qindex = rng.range(0, 256);
        let ref_best_rd = rng.range(0, 1 << 28) as i64;
        for reduce in 0..4 {
            for rf in [(1, -1), (2, -1), (3, -1), (1, 5), (2, 5), (3, 7)] {
                for m in 13..25 {
                    let mode = PredMode::from_i32(m).unwrap();
                    let is_comp = rf.1 > 0;
                    if is_comp != mode.is_inter_compound() {
                        continue;
                    }
                    for ref_mv_idx in 0..3 {
                        // C reads weight[ref_mv_idx + has_nearmv], up to
                        // index 3, which is inside the 8-entry row.
                        let (want, want_idx) = cref::ref_rdopt_ref_mv_idx_early_breakout(
                            reduce,
                            m,
                            rf,
                            ref_mv_idx,
                            qindex,
                            rdmult,
                            ref_best_rd,
                            dist.nearest_past_ref,
                            dist.nearest_future_ref,
                            &c_row,
                            &costs,
                            ref_frame_cost,
                            single_comp_cost,
                            &c_valid,
                        );
                        // C's mbmi->ref_mv_idx starts at 0 in the shim.
                        let (got, got_idx) = ref_mv_idx_early_breakout(
                            reduce,
                            dist,
                            mode,
                            [rf.0, rf.1],
                            ref_mv_idx as usize,
                            qindex,
                            rdmult,
                            ref_best_rd,
                            &row,
                            &costs,
                            ref_frame_cost,
                            single_comp_cost,
                            &p_valid,
                            0,
                        );
                        assert_eq!(
                            got, want,
                            "ref_mv_idx_early_breakout(reduce={reduce}, mode={m}, \
                             rf={rf:?}, idx={ref_mv_idx}, q={qindex})"
                        );
                        assert_eq!(
                            got_idx as i32, want_idx,
                            "ref_mv_idx_early_breakout SIDE EFFECT on \
                             mbmi->ref_mv_idx (reduce={reduce}, mode={m}, \
                             rf={rf:?}, idx={ref_mv_idx})"
                        );
                        if want_idx == ref_mv_idx && ref_mv_idx != 0 {
                            idx_written += 1;
                        }
                        trues += usize::from(want);
                        n += 1;
                    }
                }
            }
        }
    }
    assert!(trues > 0 && trues < n, "constant answer ({trues}/{n})");
    assert!(
        idx_written > 0,
        "the ref_mv_idx side effect was never observed — the early-return \
         paths that PRECEDE it dominated the sweep"
    );
}
