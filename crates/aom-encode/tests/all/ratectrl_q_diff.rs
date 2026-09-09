//! Differential harness for the fixed-Q rate controller
//! (`av1/encoder/ratectrl.c`) vs the REAL C libaom v3.14.1.
//!
//! **Tier 1c** — none of ratectrl.c's 31 exported symbols is on the qindex
//! decision path below `av1_rc_pick_q_and_bounds`, and the twelve minq lookup
//! tables are file-static, so the oracle is libaom's own `ratectrl.c` compiled
//! verbatim into `crates/aom-sys-ref/shim/ratectrl_shim.c`. The one thing that
//! separates 1c from 1 — that a second compilation could differ from the
//! archive's copy — is measured, not asserted, by
//! `ratectrl_shim_tu_matches_archive`.
//!
//! | test | C oracle |
//! |---|---|
//! | `ratectrl_shim_tu_matches_archive` | the TU-vs-archive gap closer |
//! | `minq_index_matches_c` | `get_minq_index` |
//! | `minq_luts_match_c` | `init_minq_luts` / `rc_init_minq_luts` / `av1_rc_init_minq_luts` |
//! | `active_quality_matches_c` | `get_active_quality` |
//! | `kf_active_quality_matches_c` | `get_kf_active_quality` |
//! | `gf_active_quality_matches_c` | `get_gf_active_quality` (+ `_no_rc`) |
//! | `gf_high_motion_quality_matches_c` | `get_gf_high_motion_quality` |
//! | `default_max_gf_interval_matches_c` | `get_default_max_gf_interval` |
//! | `gf_group_pyramid_level_matches_c` | `gf_group_pyramid_level` |
//! | `active_cq_level_matches_c` | `get_active_cq_level` |
//! | `intra_q_and_bounds_matches_c` | `get_intra_q_and_bounds` |
//! | `active_best_quality_matches_c` | `get_active_best_quality` |
//! | `pick_q_and_bounds_q_mode_matches_c` | `rc_pick_q_and_bounds_q_mode` |
//!
//! # Why the sweeps are shaped this way
//! `minq_luts_match_c` compares all **256 entries of all 6 curves × 3 bit
//! depths × 2 rtc modes × 2 resolution cells = 18,432 integers**, not a
//! sampled subset. The tables are the input to every other function here, and
//! a wrong polynomial coefficient typically moves only a short run of qindices
//! — a spot check at q=0/128/255 would miss it. It also pins the
//! `ASSIGN_MINQ_TABLE_2` subscript ORDER (`[rtc_mode][res]`, not
//! `[res][rtc_mode]`), which the macro's declaration and body disagree about.
//!
//! The `pick_q_and_bounds_q_mode` sweep drives all four `aom_rc_mode`s even
//! though the port targets `AOM_Q`, because the function's arms are selected
//! by mode and comparing only `AOM_Q` would leave three of them unmeasured.

use aom_encode::ratectrl::{
    FrameQParams, FrameUpdateType, MinqLuts, QINDEX_RANGE, RcMode, RcState, SCALE_NUMERATOR,
    SuperresMode, active_best_quality, active_cq_level, active_quality, default_max_gf_interval,
    gf_active_quality, gf_group_pyramid_level, gf_high_motion_quality, intra_q_and_bounds,
    kf_active_quality, minq_index, pick_q_and_bounds_q_mode,
};
use aom_sys_ref::{
    RefRcQParams, ref_compute_qdelta, ref_convert_qindex_to_q, ref_find_qindex,
    ref_rc_get_default_min_gf_interval, ref_rcc_get_active_best_quality,
    ref_rcc_get_active_cq_level, ref_rcc_get_active_quality, ref_rcc_get_default_max_gf_interval,
    ref_rcc_get_gf_active_quality, ref_rcc_get_gf_high_motion_quality,
    ref_rcc_get_intra_q_and_bounds, ref_rcc_get_kf_active_quality, ref_rcc_get_minq_index,
    ref_rcc_gf_group_pyramid_level, ref_rcc_minq_lut, ref_rcc_pick_q_and_bounds_q_mode,
    ref_rcc_probe_compute_qdelta, ref_rcc_probe_convert_qindex_to_q, ref_rcc_probe_find_qindex,
    ref_rcc_probe_min_gf_interval,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> i32 {
        (self.next() % u64::from(n)) as i32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32)
    }
}

const BIT_DEPTHS: [u8; 3] = [8, 10, 12];

/// The gap between tier 1c and tier 1: this TU is a SECOND compilation of
/// ratectrl.c, so it could in principle differ from the copy inside libaom.a.
/// Five of ratectrl.c's exported functions are re-exported from the shim TU;
/// this compares them against the archive's own symbols. If the second
/// compilation ever stopped meaning the same thing, this fails and every other
/// test in the file is downgraded to "agrees with a second build".
#[test]
fn ratectrl_shim_tu_matches_archive() {
    for &bd in &BIT_DEPTHS {
        for qindex in 0..=255 {
            assert_eq!(
                ref_rcc_probe_convert_qindex_to_q(qindex, bd).to_bits(),
                ref_convert_qindex_to_q(qindex, bd).to_bits(),
                "av1_convert_qindex_to_q: shim TU vs archive at bd={bd} q={qindex}"
            );
        }
    }
    let mut rng = Rng(0x1111_2222_3333_4444);
    for _ in 0..2000 {
        let bd = BIT_DEPTHS[(rng.below(3)) as usize];
        let desired = f64::from(rng.range(0, 6000)) / 8.0;
        let best = rng.range(0, 255);
        let worst = rng.range(best, 255);
        assert_eq!(
            ref_rcc_probe_find_qindex(desired, bd, best, worst),
            ref_find_qindex(desired, bd, best, worst),
            "av1_find_qindex: shim TU vs archive"
        );
        let qstart = f64::from(rng.range(1, 4000)) / 8.0;
        let qtarget = f64::from(rng.range(1, 4000)) / 8.0;
        assert_eq!(
            ref_rcc_probe_compute_qdelta(qstart, qtarget, bd, best, worst),
            ref_compute_qdelta(qstart, qtarget, bd, best, worst),
            "av1_compute_qdelta: shim TU vs archive"
        );
        let w = rng.range(16, 4096);
        let h = rng.range(16, 4096);
        let fps = f64::from(rng.range(1, 240));
        assert_eq!(
            ref_rcc_probe_min_gf_interval(w, h, fps),
            ref_rc_get_default_min_gf_interval(w, h, fps),
            "av1_rc_get_default_min_gf_interval: shim TU vs archive"
        );
    }
}

#[test]
fn minq_index_matches_c() {
    // The five real coefficient triples plus the rtc curve, over the whole
    // qindex-derived maxq range, at every bit depth.
    let coeffs: [(f64, f64); 5] = [
        (0.000001, -0.0004),
        (0.0000021, -0.00125),
        (0.0000015, -0.0009),
        (0.0000021, -0.00125),
        (0.00000271, -0.00113),
    ];
    let x1s = [
        0.1771, 0.379, 0.3279, 0.6634, 1.385, 0.15, 0.45, 0.70, 1.1482,
    ];
    let mut cells = 0;
    for &bd in &BIT_DEPTHS {
        for qindex in 0..256 {
            let maxq = ref_convert_qindex_to_q(qindex, bd);
            for &(x3, x2) in &coeffs {
                for &x1 in &x1s {
                    assert_eq!(
                        minq_index(maxq, x3, x2, x1, bd),
                        ref_rcc_get_minq_index(maxq, x3, x2, x1, bd),
                        "bd={bd} qindex={qindex} x3={x3} x2={x2} x1={x1}"
                    );
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 3 * 256 * 5 * 9);
}

#[test]
fn minq_luts_match_c() {
    // All 256 entries of all 6 curves x 3 bit depths x 2 rtc x 2 res cells.
    // Not a sample: these tables are every other function's input, and a wrong
    // coefficient moves only a short run of qindices.
    let mut compared = 0;
    for &bd in &BIT_DEPTHS {
        for rtc in [false, true] {
            for hi_res in [false, true] {
                let got = MinqLuts::new(bd, rtc, hi_res);
                let mode_idx = i32::from(rtc);
                let res_idx = i32::from(hi_res);
                for (which, ours) in [
                    (0, &got.kf_low),
                    (1, &got.kf_high),
                    (2, &got.arfgf_low),
                    (3, &got.arfgf_high),
                    (4, &got.inter),
                    (5, &got.rtc),
                ] {
                    let want = ref_rcc_minq_lut(which, bd, mode_idx, res_idx);
                    for q in 0..QINDEX_RANGE {
                        assert_eq!(
                            ours[q], want[q],
                            "minq table {which} bd={bd} rtc={rtc} hi_res={hi_res} q={q} \
                             (a table subscript swap shows up as a whole-curve mismatch, \
                             a wrong coefficient as a short run)"
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert_eq!(compared, 3 * 2 * 2 * 6 * 256);
}

#[test]
fn active_quality_matches_c() {
    let mut rng = Rng(0xaaaa_bbbb_cccc_dddd);
    let mut cells = 0;
    for &bd in &BIT_DEPTHS {
        for rtc in [false, true] {
            for hi_res in [false, true] {
                let l = MinqLuts::new(bd, rtc, hi_res);
                // Both real endpoint pairs plus adversarial ones: a boost far
                // outside the window (both saturating arms) and a boost exactly
                // on each endpoint (which C resolves through the interpolating
                // branch, not the saturating one).
                for (low, high) in [(553, 8000), (562, 2875), (100, 4994), (300, 2400), (1, 2)] {
                    for _ in 0..200 {
                        let q = rng.below(256);
                        let boost = rng.range(low - 500, high + 500);
                        assert_eq!(
                            active_quality(q, boost, low, high, &l.arfgf_low, &l.arfgf_high),
                            ref_rcc_get_active_quality(
                                q,
                                boost,
                                low,
                                high,
                                &l.arfgf_low,
                                &l.arfgf_high
                            ),
                            "bd={bd} rtc={rtc} hi={hi_res} q={q} boost={boost} \
                             window=[{low},{high}]"
                        );
                        cells += 1;
                    }
                    // The exact endpoints.
                    for boost in [low, high] {
                        for q in [0, 1, 127, 254, 255] {
                            assert_eq!(
                                active_quality(q, boost, low, high, &l.kf_low, &l.kf_high),
                                ref_rcc_get_active_quality(
                                    q, boost, low, high, &l.kf_low, &l.kf_high
                                ),
                                "endpoint boost={boost} q={q}"
                            );
                            cells += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cells >= 5000, "sweep too small: {cells}");

    // The rounding arm the real tables cannot reach. `get_active_quality`
    // rounds with `(offset * qdiff + gap/2) / gap`, C integer division, which
    // truncates TOWARD ZERO — so a NEGATIVE `qdiff` rounds the other way from
    // Euclidean division. Measured on the twelve real minq tables, `qdiff` is
    // >= 0 in all 6,144 (bd x rtc x res x curve x qindex) cells, so a
    // Euclidean transcription passes every table-driven sweep above. This arm
    // feeds C and the port synthetic arrays with the curves swapped, which is
    // legal for the function's signature and makes the rounding observable.
    let mut low = [0i32; QINDEX_RANGE];
    let mut high = [0i32; QINDEX_RANGE];
    let mut neg_cells = 0;
    for q in 0..QINDEX_RANGE {
        low[q] = (q as i32) * 2 + 40;
        high[q] = q as i32; // deliberately BELOW low => qdiff < 0
    }
    for (lo_b, hi_b) in [(553, 8000), (562, 2875), (100, 4994), (1, 9)] {
        for boost in (lo_b..=hi_b).step_by(((hi_b - lo_b) / 97).max(1) as usize) {
            for q in [0, 1, 3, 17, 64, 127, 200, 254, 255] {
                assert_eq!(
                    active_quality(q, boost, lo_b, hi_b, &low, &high),
                    ref_rcc_get_active_quality(q, boost, lo_b, hi_b, &low, &high),
                    "negative-qdiff rounding: q={q} boost={boost} window=[{lo_b},{hi_b}]"
                );
                neg_cells += 1;
            }
        }
    }
    assert!(
        neg_cells >= 1000,
        "negative-qdiff arm too small: {neg_cells}"
    );
}

#[test]
fn kf_active_quality_matches_c() {
    let mut rng = Rng(0x1357_9bdf_0246_8ace);
    for &bd in &BIT_DEPTHS {
        for rtc in [false, true] {
            // res_idx is the THREE-valued index at the call site, but only
            // `res_idx > 1` reaches the table, so sweep all three.
            for res_idx in 0..3 {
                let l = MinqLuts::new(bd, rtc, res_idx > 1);
                for _ in 0..400 {
                    let q = rng.below(256);
                    let kf_boost = rng.range(0, 12000);
                    assert_eq!(
                        kf_active_quality(&l, kf_boost, q, rtc),
                        ref_rcc_get_kf_active_quality(kf_boost, q, bd, res_idx, rtc),
                        "bd={bd} rtc={rtc} res_idx={res_idx} q={q} kf_boost={kf_boost}"
                    );
                }
            }
        }
    }
}

#[test]
fn gf_active_quality_matches_c() {
    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    for &bd in &BIT_DEPTHS {
        for rtc in [false, true] {
            for res_idx in 0..3 {
                let l = MinqLuts::new(bd, rtc, res_idx > 1);
                for _ in 0..500 {
                    let q = rng.below(256);
                    let gfu_boost = rng.range(0, 8000);
                    // Straddle gfboost_thresh (4000 / 4000 / 3000) so both
                    // (gf_low_1, gf_high_1) and (gf_low_2, gf_high_2) windows
                    // are reached.
                    let gfu_boost_average = rng.range(0, 8000);
                    assert_eq!(
                        gf_active_quality(
                            &l,
                            gfu_boost,
                            gfu_boost_average,
                            q,
                            res_idx as usize,
                            rtc
                        ),
                        ref_rcc_get_gf_active_quality(
                            gfu_boost,
                            gfu_boost_average,
                            q,
                            bd,
                            res_idx,
                            rtc
                        ),
                        "bd={bd} rtc={rtc} res_idx={res_idx} q={q} boost={gfu_boost} \
                         avg={gfu_boost_average}"
                    );
                }
            }
        }
    }
}

#[test]
fn gf_high_motion_quality_matches_c() {
    for &bd in &BIT_DEPTHS {
        for rtc in [false, true] {
            for res_idx in 0..3 {
                let l = MinqLuts::new(bd, rtc, res_idx > 1);
                for q in 0..256 {
                    assert_eq!(
                        gf_high_motion_quality(&l, q),
                        ref_rcc_get_gf_high_motion_quality(q, bd, res_idx, rtc),
                        "bd={bd} rtc={rtc} res_idx={res_idx} q={q}"
                    );
                }
            }
        }
    }
}

#[test]
fn default_max_gf_interval_matches_c() {
    for fps_x8 in 0..2000 {
        let fps = f64::from(fps_x8) / 8.0;
        for min_gf in [0, 1, 4, 16, 32, 33, 64] {
            assert_eq!(
                default_max_gf_interval(fps, min_gf),
                ref_rcc_get_default_max_gf_interval(fps, min_gf),
                "framerate={fps} min_gf_interval={min_gf}"
            );
        }
    }
}

#[test]
fn gf_group_pyramid_level_matches_c() {
    for depth in 0..=8 {
        assert_eq!(
            gf_group_pyramid_level(depth),
            ref_rcc_gf_group_pyramid_level(depth)
        );
    }
}

const RC_MODES: [(RcMode, i32); 4] = [
    (RcMode::Vbr, 0),
    (RcMode::Cbr, 1),
    (RcMode::Cq, 2),
    (RcMode::Q, 3),
];

const SUPERRES_MODES: [(SuperresMode, i32); 5] = [
    (SuperresMode::None, 0),
    (SuperresMode::Fixed, 1),
    (SuperresMode::Random, 2),
    (SuperresMode::QThresh, 3),
    (SuperresMode::Auto, 4),
];

const UPDATE_TYPES: [(FrameUpdateType, i32); 7] = [
    (FrameUpdateType::Kf, 0),
    (FrameUpdateType::Lf, 1),
    (FrameUpdateType::Gf, 2),
    (FrameUpdateType::Arf, 3),
    (FrameUpdateType::Overlay, 4),
    (FrameUpdateType::IntnlOverlay, 5),
    (FrameUpdateType::IntnlArf, 6),
];

/// Draw one coherent (port params, C params) pair.
#[allow(clippy::type_complexity)]
fn draw(
    rng: &mut Rng,
) -> (
    FrameQParams,
    RcState,
    RefRcQParams,
    RcMode,
    i32,
    FrameUpdateType,
    i32,
    bool,
    i32,
    i64,
    i64,
) {
    let bd = BIT_DEPTHS[rng.below(3) as usize];
    // Frame sizes straddling both res_idx steps (480 and 608) on the shorter
    // side, plus one below 352x288 to reach the small-format q_adj_factor arm.
    let (w, h) = match rng.below(6) {
        0 => (176, 144),
        1 => (352, 288),
        2 => (640, 360),
        3 => (854, 480),
        4 => (1280, 720),
        _ => (1920, 1080),
    };
    let (rc_mode, c_rc_mode) = RC_MODES[rng.below(4) as usize];
    let (superres_mode, c_superres) = SUPERRES_MODES[rng.below(5) as usize];
    let superres_denom = rng.range(8, 16);
    let (update_type, c_update) = UPDATE_TYPES[rng.below(7) as usize];
    let rtc_mode = rng.below(2) == 1;
    let screen_content = rng.below(2) == 1;
    let large_scale = rng.below(8) == 0;
    let refresh_golden = rng.below(2) == 1;
    let refresh_alt_ref = rng.below(2) == 1;
    let intra_only = rng.below(2) == 1;
    let two_pass = rng.below(2) == 1;
    let cq_level = rng.below(256);
    let active_worst_in = rng.below(256);
    let layer_depth = rng.range(0, 7);
    let best_quality = rng.below(64);
    let worst_quality = rng.range(best_quality, 255);
    let total_target_bits = i64::from(rng.below(1_000_000));
    let total_actual_bits = i64::from(rng.below(1_000_000));

    let p = FrameQParams {
        bit_depth: bd,
        coded_width: w,
        coded_height: h,
        width: w,
        height: h,
        rtc_mode,
        screen_content,
        superres_mode,
        superres_denom,
        large_scale,
        refresh_golden,
        refresh_alt_ref,
    };
    let rc = RcState {
        kf_boost: rng.range(0, 12000),
        gfu_boost: rng.range(0, 8000),
        gfu_boost_average: rng.range(0, 8000),
        arf_boost_factor: rng.range(0, 400) as f32 / 100.0,
        arf_q: rng.below(256),
        avg_frame_qindex_inter: rng.below(256),
        this_key_frame_forced: rng.below(2) == 1,
        last_boosted_qindex: rng.below(256),
        last_kf_qindex: rng.below(256),
        frames_to_key: rng.range(0, 100),
        frames_since_key: rng.range(0, 100),
        best_quality,
        worst_quality,
        kf_zeromotion_pct: rng.range(0, 100),
        last_kfgroup_zeromotion_pct: rng.range(0, 100),
        two_pass,
    };
    let c = RefRcQParams {
        bit_depth: i32::from(bd),
        coded_width: w,
        coded_height: h,
        width: w,
        height: h,
        rtc_mode: i32::from(rtc_mode),
        screen_content: i32::from(screen_content),
        superres_mode: c_superres,
        superres_denom,
        large_scale: i32::from(large_scale),
        refresh_golden: i32::from(refresh_golden),
        refresh_alt_ref: i32::from(refresh_alt_ref),
        rc_mode: c_rc_mode,
        cq_level,
        intra_only: i32::from(intra_only),
        active_worst_in,
        update_type: c_update,
        layer_depth,
        kf_boost: rc.kf_boost,
        gfu_boost: rc.gfu_boost,
        gfu_boost_average: rc.gfu_boost_average,
        arf_boost_factor: rc.arf_boost_factor,
        arf_q: rc.arf_q,
        avg_frame_qindex_inter: rc.avg_frame_qindex_inter,
        this_key_frame_forced: i32::from(rc.this_key_frame_forced),
        last_boosted_qindex: rc.last_boosted_qindex,
        last_kf_qindex: rc.last_kf_qindex,
        frames_to_key: rc.frames_to_key,
        frames_since_key: rc.frames_since_key,
        best_quality: rc.best_quality,
        worst_quality: rc.worst_quality,
        kf_zeromotion_pct: rc.kf_zeromotion_pct,
        last_kfgroup_zeromotion_pct: rc.last_kfgroup_zeromotion_pct,
        two_pass: i32::from(two_pass),
        total_actual_bits,
        total_target_bits,
    };
    (
        p,
        rc,
        c,
        rc_mode,
        cq_level,
        update_type,
        layer_depth,
        intra_only,
        active_worst_in,
        total_actual_bits,
        total_target_bits,
    )
}

#[test]
fn active_cq_level_matches_c() {
    let mut rng = Rng(0xdead_beef_cafe_0001);
    for _ in 0..20000 {
        let (_p, rc, c, rc_mode, cq_level, _ut, _ld, intra_only, _aw, actual, target) =
            draw(&mut rng);
        let got = active_cq_level(
            cq_level,
            rc_mode,
            intra_only,
            rc.frames_to_key,
            _p.superres_mode,
            _p.superres_denom,
            actual,
            target,
        );
        assert_eq!(got, ref_rcc_get_active_cq_level(&c), "params = {c:?}");
    }
}

#[test]
fn intra_q_and_bounds_matches_c() {
    let mut rng = Rng(0xdead_beef_cafe_0002);
    for _ in 0..20000 {
        let (p, rc, c, rc_mode, cq_level, _ut, _ld, _io, active_worst_in, _a, _t) = draw(&mut rng);
        let luts = p.minq_luts();
        let got = intra_q_and_bounds(&p, &rc, &luts, rc_mode, cq_level, active_worst_in);
        let want = ref_rcc_get_intra_q_and_bounds(&c, cq_level, active_worst_in);
        assert_eq!(
            (got.active_best, got.active_worst),
            want,
            "params = {c:?} cq_level={cq_level} active_worst_in={active_worst_in}"
        );
    }
}

#[test]
fn active_best_quality_matches_c() {
    let mut rng = Rng(0xdead_beef_cafe_0003);
    for _ in 0..20000 {
        let (p, rc, c, rc_mode, cq_level, update_type, layer_depth, _io, active_worst_in, _a, _t) =
            draw(&mut rng);
        let luts = p.minq_luts();
        let got = active_best_quality(
            &p,
            &rc,
            &luts,
            rc_mode,
            active_worst_in,
            cq_level,
            update_type,
            layer_depth,
        );
        let want = ref_rcc_get_active_best_quality(&c, active_worst_in, cq_level);
        assert_eq!(got, want, "params = {c:?} cq_level={cq_level}");
    }
}

#[test]
fn pick_q_and_bounds_q_mode_matches_c() {
    let mut rng = Rng(0xdead_beef_cafe_0004);
    let mut intra_cells = 0;
    let mut inter_cells = 0;
    for _ in 0..30000 {
        let (
            p,
            rc,
            c,
            rc_mode,
            cq_level,
            update_type,
            layer_depth,
            intra_only,
            active_worst_in,
            actual,
            target,
        ) = draw(&mut rng);
        let got = pick_q_and_bounds_q_mode(
            &p,
            &rc,
            rc_mode,
            cq_level,
            active_worst_in,
            intra_only,
            update_type,
            layer_depth,
            actual,
            target,
        );
        let (q, bottom, top) = ref_rcc_pick_q_and_bounds_q_mode(&c);
        assert_eq!(
            (got.q, got.bottom_index, got.top_index),
            (q, bottom, top),
            "params = {c:?}"
        );
        if intra_only {
            intra_cells += 1;
        } else {
            inter_cells += 1;
        }
    }
    // Both arms must actually be exercised — the intra arm is a whole separate
    // function (get_intra_q_and_bounds) and a generator that only produced
    // inter frames would leave it untested while still passing.
    assert!(
        intra_cells > 5000,
        "intra arm barely reached: {intra_cells}"
    );
    assert!(
        inter_cells > 5000,
        "inter arm barely reached: {inter_cells}"
    );
}

/// The `--lag-in-frames=0 --end-usage=q` low-delay envelope, end to end: a KEY
/// frame followed by LF_UPDATE leaves, at every cq level. This is the config
/// `crate::rc::base_qindex_lowdelay_p_from_cq` documents, checked here against
/// the real C decision rather than against the port's own reasoning.
#[test]
fn lowdelay_q_envelope_matches_c() {
    for &bd in &BIT_DEPTHS {
        for cq_level in 0..=255 {
            for (intra_only, frames_to_key, update_type, c_update) in [
                (true, 1, FrameUpdateType::Kf, 0),
                (false, 0, FrameUpdateType::Lf, 1),
            ] {
                let p = FrameQParams {
                    bit_depth: bd,
                    coded_width: 352,
                    coded_height: 288,
                    width: 352,
                    height: 288,
                    rtc_mode: false,
                    screen_content: false,
                    superres_mode: SuperresMode::None,
                    superres_denom: SCALE_NUMERATOR,
                    large_scale: false,
                    refresh_golden: false,
                    refresh_alt_ref: false,
                };
                let rc = RcState {
                    frames_to_key,
                    best_quality: 0,
                    worst_quality: 255,
                    ..Default::default()
                };
                let c = RefRcQParams {
                    bit_depth: i32::from(bd),
                    coded_width: 352,
                    coded_height: 288,
                    width: 352,
                    height: 288,
                    superres_mode: 0,
                    superres_denom: SCALE_NUMERATOR,
                    rc_mode: 3, // AOM_Q
                    cq_level,
                    intra_only: i32::from(intra_only),
                    active_worst_in: 255,
                    update_type: c_update,
                    frames_to_key,
                    best_quality: 0,
                    worst_quality: 255,
                    ..Default::default()
                };
                let got = pick_q_and_bounds_q_mode(
                    &p,
                    &rc,
                    RcMode::Q,
                    cq_level,
                    255,
                    intra_only,
                    update_type,
                    0,
                    0,
                    0,
                );
                assert_eq!(
                    (got.q, got.bottom_index, got.top_index),
                    ref_rcc_pick_q_and_bounds_q_mode(&c),
                    "bd={bd} cq_level={cq_level} intra={intra_only}"
                );
            }
        }
    }
}
