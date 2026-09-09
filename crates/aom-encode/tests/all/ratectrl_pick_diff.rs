//! Differential harness for the qindex decision's outer layers
//! (`av1_rc_pick_q_and_bounds` and the four statics under it) vs the REAL C
//! libaom v3.14.1.
//!
//! **Tier 1** for the exported `av1_rc_pick_q_and_bounds`, driven out of
//! `upstream/build/libaom.a` through `shim/rcarchive_shim.c`.
//! **Tier 1c** for `calc_active_worst_quality_no_stats_vbr`,
//! `adjust_active_best_and_worst_quality`, `get_q`,
//! `rc_pick_q_and_bounds_no_stats` and `rc_pick_q_and_bounds`, which are
//! file-static. `dispatcher_shim_tu_matches_archive` compares the two
//! compilations on the dispatcher itself.

use aom_encode::ratectrl::{
    FrameQParams, FrameUpdateType, PickedQ, QBounds, RcMode, RcState, SuperresMode,
};
use aom_encode::ratectrl_pick::{
    FIXED_GF_INTERVAL, PickQRoute, TwoPassExtend, active_worst_quality_no_stats_vbr,
    adjust_active_best_and_worst_quality, arf_q_after_pick, get_q, pick_q_and_bounds,
    pick_q_and_bounds_no_stats,
};
use aom_encode::ratectrl_rate::rate_correction_factor;
use aom_sys_ref::{
    RefRcPickParams, ref_rc_pick_q_and_bounds, ref_rcc_active_worst_quality_no_stats_vbr,
    ref_rcc_adjust_active_best_and_worst_quality, ref_rcc_get_q, ref_rcc_pick_q_and_bounds,
    ref_rcc_pick_q_and_bounds_no_stats, ref_rcc_probe_rc_pick_q_and_bounds,
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
    fn boolean(&mut self) -> bool {
        self.below(2) == 1
    }
}

const RC_MODES: [(RcMode, i32); 4] = [
    (RcMode::Vbr, 0),
    (RcMode::Cbr, 1),
    (RcMode::Cq, 2),
    (RcMode::Q, 3),
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
const SUPERRES: [(SuperresMode, i32); 5] = [
    (SuperresMode::None, 0),
    (SuperresMode::Fixed, 1),
    (SuperresMode::Random, 2),
    (SuperresMode::QThresh, 3),
    (SuperresMode::Auto, 4),
];
const SIZES: [(i32, i32); 6] = [
    (176, 144),
    (352, 288),
    (640, 360),
    (854, 480),
    (1280, 720),
    (1920, 1080),
];

/// One coherent cell: the C parameter block plus the port-side views of it.
struct Cell {
    c: RefRcPickParams,
    p: FrameQParams,
    rc: RcState,
    rc_mode: RcMode,
    update_type: FrameUpdateType,
    intra_only: bool,
    frame_type_is_key: bool,
    two_pass_extend: TwoPassExtend,
    correction_factor: f64,
    has_no_stats_stage: bool,
}

/// `av1_rc_regulate_q`'s tail is `adjust_q_cbr` when
/// `mode == AOM_CBR && has_no_stats_stage`, and that arm is NOT ported (see
/// `crate::ratectrl_rate`'s scope note). Every function below `get_q` inherits
/// it, so those cells are skipped for VALUE comparison — and counted, so the
/// share of the sweep they take is visible rather than silent.
fn reaches_unported_adjust_q_cbr(cell: &Cell) -> bool {
    cell.rc_mode == RcMode::Cbr && cell.has_no_stats_stage
}

fn draw(rng: &mut Rng) -> Cell {
    let bd = [8u8, 10, 12][rng.below(3) as usize];
    let (w, h) = SIZES[rng.below(SIZES.len() as u32) as usize];
    let (cw, ch) = SIZES[rng.below(SIZES.len() as u32) as usize];
    let (rc_mode, c_rc_mode) = RC_MODES[rng.below(4) as usize];
    let (update_type, c_update) = UPDATE_TYPES[rng.below(7) as usize];
    let (superres_mode, c_superres) = SUPERRES[rng.below(5) as usize];
    let frame_type = rng.below(4);
    let intra_only = frame_type == 0 || frame_type == 2;
    let best_quality = rng.below(64);
    let worst_quality = rng.range(best_quality, 255);
    // frame_number spread across the whole 16-entry delta_rate cycle,
    // including the eight zero-initialised tail entries.
    let frame_number = rng.range(0, 4 * FIXED_GF_INTERVAL as i32);

    let c = RefRcPickParams {
        bit_depth: i32::from(bd),
        coded_width: w,
        coded_height: h,
        width: w,
        height: h,
        cfg_width: cw,
        cfg_height: ch,
        rtc_mode: i32::from(rng.boolean()),
        screen_content: i32::from(rng.boolean()),
        superres_mode: c_superres,
        superres_denom: rng.range(8, 16),
        large_scale: i32::from(rng.below(8) == 0),
        refresh_golden: i32::from(rng.boolean()),
        refresh_bwd_ref: i32::from(rng.boolean()),
        refresh_alt_ref: i32::from(rng.boolean()),
        rc_mode: c_rc_mode,
        cq_level: rng.below(256),
        frame_type,
        frame_number,
        has_no_stats_stage: i32::from(rng.boolean()),
        two_pass: i32::from(rng.boolean()),
        update_type: c_update,
        layer_depth: rng.range(0, 8),
        gf_index_frame_type: rng.below(2),
        active_worst_quality: rng.below(256),
        best_quality,
        worst_quality,
        frames_to_key: rng.range(0, 100),
        frames_since_key: rng.range(0, 100),
        is_src_frame_alt_ref: i32::from(rng.boolean()),
        this_frame_target: rng.range(0, 5_000_000),
        max_frame_bandwidth: rng.range(1000, 100_000_000),
        kf_boost: rng.range(0, 12000),
        gfu_boost: rng.range(0, 8000),
        gfu_boost_average: rng.range(0, 8000),
        arf_boost_factor: rng.range(0, 400) as f32 / 100.0,
        arf_q: rng.below(256),
        avg_frame_qindex_key: rng.below(256),
        avg_frame_qindex_inter: rng.below(256),
        this_key_frame_forced: i32::from(rng.boolean()),
        last_boosted_qindex: rng.below(256),
        last_kf_qindex: rng.below(256),
        last_q_key: rng.below(256),
        last_q_inter: rng.below(256),
        active_best_quality_by_layer: core::array::from_fn(|_| rng.below(256)),
        total_actual_bits: i64::from(rng.below(1_000_000)),
        total_target_bits: i64::from(rng.below(1_000_000)),
        rate_correction_factors: core::array::from_fn(|_| f64::from(rng.range(1, 5000)) / 100.0),
        kf_zeromotion_pct: rng.range(0, 100),
        last_kfgroup_zeromotion_pct: rng.range(0, 100),
        extend_minq: rng.range(0, 64),
        extend_maxq: rng.range(0, 64),
    };

    let p = FrameQParams {
        bit_depth: bd,
        coded_width: w,
        coded_height: h,
        width: w,
        height: h,
        rtc_mode: c.rtc_mode != 0,
        screen_content: c.screen_content != 0,
        superres_mode,
        superres_denom: c.superres_denom,
        large_scale: c.large_scale != 0,
        refresh_golden: c.refresh_golden != 0,
        refresh_alt_ref: c.refresh_alt_ref != 0,
    };
    let rc = RcState {
        kf_boost: c.kf_boost,
        gfu_boost: c.gfu_boost,
        gfu_boost_average: c.gfu_boost_average,
        arf_boost_factor: c.arf_boost_factor,
        arf_q: c.arf_q,
        avg_frame_qindex_inter: c.avg_frame_qindex_inter,
        this_key_frame_forced: c.this_key_frame_forced != 0,
        last_boosted_qindex: c.last_boosted_qindex,
        last_kf_qindex: c.last_kf_qindex,
        frames_to_key: c.frames_to_key,
        frames_since_key: c.frames_since_key,
        best_quality: c.best_quality,
        worst_quality: c.worst_quality,
        kf_zeromotion_pct: c.kf_zeromotion_pct,
        last_kfgroup_zeromotion_pct: c.last_kfgroup_zeromotion_pct,
        two_pass: c.two_pass != 0,
    };
    // The two C stage predicates, spelled out exactly as encoder.h does,
    // because the shim derives oxcf.pass and lap_enabled from the two request
    // flags and the resulting predicates are NOT those flags:
    //   has_no_stats_stage       = pass == ONE_PASS && (!lap || mode == REALTIME)
    //   is_stat_consumption_stage = twopass || (ONE_PASS && lap && mode != REALTIME)
    // An earlier draft of this test wrote the second as `twopass || lap` and
    // dropped the `mode != REALTIME` clause, which put the port on a different
    // rate-correction slot from C on every realtime+lap cell.
    let pass_one = c.two_pass == 0;
    let lap = c.has_no_stats_stage == 0 && c.two_pass == 0;
    let realtime = c.rtc_mode != 0;
    let has_no_stats_stage = pass_one && (!lap || realtime);
    let stat_consumption = c.two_pass != 0 || (pass_one && lap && !realtime);

    // get_rate_correction_factor is what the C functions derive internally;
    // the port takes it as a parameter, so compute the same value.
    let correction_factor = rate_correction_factor(
        &c.rate_correction_factors,
        frame_type == 0,
        stat_consumption,
        update_type,
        c.refresh_golden != 0,
        c.refresh_alt_ref != 0,
        c.is_src_frame_alt_ref != 0,
        false,
        rc_mode == RcMode::Cbr,
        0,
        cw,
        ch,
        w,
        h,
    );

    Cell {
        c,
        p,
        rc,
        rc_mode,
        update_type,
        intra_only,
        frame_type_is_key: frame_type == 0,
        two_pass_extend: TwoPassExtend {
            extend_minq: c.extend_minq,
            extend_maxq: c.extend_maxq,
        },
        correction_factor,
        has_no_stats_stage,
    }
}

#[test]
fn dispatcher_shim_tu_matches_archive() {
    // The tier-1c gap closer, measured on the dispatcher itself.
    let mut rng = Rng(0x7070_8080_9090_a0a0);
    for _ in 0..8000 {
        let cell = draw(&mut rng);
        assert_eq!(
            ref_rcc_probe_rc_pick_q_and_bounds(&cell.c),
            ref_rc_pick_q_and_bounds(&cell.c),
            "av1_rc_pick_q_and_bounds: shim TU vs archive, params = {:?}",
            cell.c
        );
    }
}

#[test]
fn active_worst_quality_no_stats_vbr_matches_c() {
    let mut rng = Rng(0x1212_3434_5656_7878);
    let mut frame0 = 0;
    let mut frame1 = 0;
    for _ in 0..20000 {
        let mut cell = draw(&mut rng);
        // The function branches on frame_number == 0 and == 1, so force those
        // often enough to test both.
        cell.c.frame_number = match rng.below(4) {
            0 => 0,
            1 => 1,
            _ => cell.c.frame_number,
        };
        let got = active_worst_quality_no_stats_vbr(
            cell.frame_type_is_key,
            cell.c.frame_number as u32,
            cell.c.last_q_key,
            cell.c.last_q_inter,
            cell.c.is_src_frame_alt_ref != 0,
            cell.c.refresh_golden != 0,
            cell.c.refresh_bwd_ref != 0,
            cell.c.refresh_alt_ref != 0,
            cell.c.worst_quality,
        );
        assert_eq!(
            got,
            ref_rcc_active_worst_quality_no_stats_vbr(&cell.c),
            "params = {:?}",
            cell.c
        );
        if cell.c.frame_number == 0 {
            frame0 += 1;
        }
        if cell.c.frame_number == 1 {
            frame1 += 1;
        }
    }
    assert!(
        frame0 > 1000 && frame1 > 1000,
        "the frame 0/1 arms went untested"
    );
}

#[test]
fn adjust_active_best_and_worst_quality_matches_c() {
    let mut rng = Rng(0x2323_4545_6767_8989);
    for _ in 0..20000 {
        let cell = draw(&mut rng);
        let active_best = rng.range(cell.c.best_quality, cell.c.worst_quality);
        let active_worst = rng.range(active_best, 255);
        let is_intrl_arf_boost = cell.update_type == FrameUpdateType::IntnlArf;
        let got = adjust_active_best_and_worst_quality(
            QBounds {
                active_best,
                active_worst,
            },
            cell.rc_mode,
            cell.two_pass_extend,
            cell.intra_only,
            cell.rc.this_key_frame_forced,
            cell.rc.last_kfgroup_zeromotion_pct,
            cell.update_type,
            cell.c.gf_index_frame_type == 0,
            cell.c.layer_depth,
            cell.p.screen_content,
            cell.p.bit_depth,
            cell.rc.best_quality,
            cell.rc.worst_quality,
            // The shim builds an unscaled frame.
            false,
            cell.intra_only
                || matches!(cell.update_type, FrameUpdateType::Arf | FrameUpdateType::Gf),
            cell.frame_type_is_key,
        );
        let want = ref_rcc_adjust_active_best_and_worst_quality(
            &cell.c,
            active_best,
            active_worst,
            is_intrl_arf_boost,
        );
        assert_eq!(
            (got.active_best, got.active_worst),
            want,
            "params = {:?} in=({active_best},{active_worst})",
            cell.c
        );
    }
}

#[test]
fn get_q_matches_c() {
    let mut rng = Rng(0x3434_5656_7878_9a9a);
    let mut cbr_skipped = 0;
    for _ in 0..20000 {
        let cell = draw(&mut rng);
        if reaches_unported_adjust_q_cbr(&cell) {
            cbr_skipped += 1;
            continue;
        }
        let active_best = rng.range(cell.c.best_quality, cell.c.worst_quality);
        let active_worst = rng.range(active_best, 255);
        let got = get_q(
            cell.rc_mode,
            cell.intra_only,
            cell.rc.this_key_frame_forced,
            cell.rc.kf_zeromotion_pct,
            cell.rc.last_kfgroup_zeromotion_pct,
            cell.rc.frames_to_key,
            cell.rc.last_boosted_qindex,
            cell.rc.last_kf_qindex,
            active_worst,
            active_best,
            cell.c.this_frame_target,
            cell.c.max_frame_bandwidth,
            cell.c.coded_width,
            cell.c.coded_height,
            cell.frame_type_is_key,
            cell.p.screen_content,
            cell.correction_factor,
            cell.p.bit_depth,
        );
        assert_eq!(
            got,
            ref_rcc_get_q(&cell.c, active_worst, active_best),
            "params = {:?} bounds=({active_best},{active_worst})",
            cell.c
        );
    }
    assert!(
        cbr_skipped * 4 < 20000,
        "{cbr_skipped} of 20,000 cells were skipped for the unported \
         adjust_q_cbr tail — that is most of the sweep, not an exclusion"
    );
}

#[test]
fn pick_q_and_bounds_no_stats_matches_c() {
    let mut rng = Rng(0x4545_6767_8989_abab);
    let mut delta_rate_tail = 0;
    let mut cbr_skipped = 0;
    for _ in 0..20000 {
        let cell = draw(&mut rng);
        if reaches_unported_adjust_q_cbr(&cell) {
            cbr_skipped += 1;
            continue;
        }
        let got = pick_q_and_bounds_no_stats(
            &cell.p,
            &cell.rc,
            cell.rc_mode,
            cell.c.cq_level,
            cell.intra_only,
            cell.c.frame_number as u32,
            cell.c.width,
            cell.c.height,
            cell.c.last_q_key,
            cell.c.last_q_inter,
            cell.c.avg_frame_qindex_key,
            cell.c.is_src_frame_alt_ref != 0,
            cell.c.refresh_bwd_ref != 0,
            cell.c.this_frame_target,
            cell.c.max_frame_bandwidth,
            cell.correction_factor,
            cell.c.total_actual_bits,
            cell.c.total_target_bits,
            cell.frame_type_is_key,
        );
        let want = ref_rcc_pick_q_and_bounds_no_stats(&cell.c);
        assert_eq!(
            (got.q, got.bottom_index, got.top_index),
            want,
            "params = {:?}",
            cell.c
        );
        if cell.rc_mode == RcMode::Q
            && !cell.intra_only
            && (cell.c.frame_number as usize % FIXED_GF_INTERVAL) >= 8
        {
            delta_rate_tail += 1;
        }
    }
    // The eight zero-initialised delta_rate entries must be reached, or the
    // half of the table C never fills goes untested.
    let _ = cbr_skipped;
    assert!(
        delta_rate_tail > 200,
        "the zero tail of delta_rate was reached {delta_rate_tail} times"
    );
}

#[test]
fn pick_q_and_bounds_matches_c() {
    let mut rng = Rng(0x5656_7878_9a9a_bcbc);
    let mut q_mode_cells = 0;
    let mut layered_cells = 0;
    let mut cbr_skipped = 0;
    for _ in 0..20000 {
        let cell = draw(&mut rng);
        if reaches_unported_adjust_q_cbr(&cell) {
            cbr_skipped += 1;
            continue;
        }
        let got = pick_q_and_bounds(
            &cell.p,
            &cell.rc,
            cell.rc_mode,
            cell.c.cq_level,
            cell.c.active_worst_quality,
            cell.intra_only,
            cell.update_type,
            cell.c.layer_depth,
            cell.c.total_actual_bits,
            cell.c.total_target_bits,
            &cell.c.active_best_quality_by_layer,
            cell.two_pass_extend,
            cell.c.is_src_frame_alt_ref != 0,
            cell.c.this_frame_target,
            cell.c.max_frame_bandwidth,
            cell.correction_factor,
            false,
            cell.frame_type_is_key,
            cell.c.gf_index_frame_type == 0,
        );
        let want = ref_rcc_pick_q_and_bounds(&cell.c);
        assert_eq!(
            (got.q, got.bottom_index, got.top_index),
            want,
            "params = {:?}",
            cell.c
        );
        if cell.rc_mode == RcMode::Q {
            q_mode_cells += 1;
        } else if !cell.intra_only && cell.c.layer_depth > 1 && cell.c.layer_depth <= 6 {
            layered_cells += 1;
        }
    }
    let _ = cbr_skipped;
    assert!(
        q_mode_cells > 3000,
        "the AOM_Q route fired {q_mode_cells} times"
    );
    assert!(
        layered_cells > 1000,
        "the pyramid-layer active_best_quality[] arm fired {layered_cells} times"
    );
}

#[test]
fn dispatcher_route_and_arf_q_match_c() {
    // The dispatcher itself: the route it picks, the q it returns, and the
    // arf_q it writes on an ARF_UPDATE.
    let mut rng = Rng(0x6767_8989_abab_cdcd);
    let mut routes = [0usize; 3];
    for _ in 0..20000 {
        let cell = draw(&mut rng);
        let route = PickQRoute::of(cell.rc_mode, cell.update_type, cell.has_no_stats_stage);
        let (want_q, want_bottom, want_top, want_arf_q) = ref_rc_pick_q_and_bounds(&cell.c);

        let got: PickedQ = match route {
            PickQRoute::NoStatsCbr => {
                // Not ported; skip the value comparison but still count it, so
                // the CBR share of the sweep is visible rather than silent.
                routes[0] += 1;
                continue;
            }
            PickQRoute::NoStats => {
                routes[1] += 1;
                if reaches_unported_adjust_q_cbr(&cell) {
                    continue;
                }
                pick_q_and_bounds_no_stats(
                    &cell.p,
                    &cell.rc,
                    cell.rc_mode,
                    cell.c.cq_level,
                    cell.intra_only,
                    cell.c.frame_number as u32,
                    cell.c.width,
                    cell.c.height,
                    cell.c.last_q_key,
                    cell.c.last_q_inter,
                    cell.c.avg_frame_qindex_key,
                    cell.c.is_src_frame_alt_ref != 0,
                    cell.c.refresh_bwd_ref != 0,
                    cell.c.this_frame_target,
                    cell.c.max_frame_bandwidth,
                    cell.correction_factor,
                    cell.c.total_actual_bits,
                    cell.c.total_target_bits,
                    cell.frame_type_is_key,
                )
            }
            PickQRoute::General => {
                routes[2] += 1;
                if reaches_unported_adjust_q_cbr(&cell) {
                    continue;
                }
                pick_q_and_bounds(
                    &cell.p,
                    &cell.rc,
                    cell.rc_mode,
                    cell.c.cq_level,
                    cell.c.active_worst_quality,
                    cell.intra_only,
                    cell.update_type,
                    cell.c.layer_depth,
                    cell.c.total_actual_bits,
                    cell.c.total_target_bits,
                    &cell.c.active_best_quality_by_layer,
                    cell.two_pass_extend,
                    cell.c.is_src_frame_alt_ref != 0,
                    cell.c.this_frame_target,
                    cell.c.max_frame_bandwidth,
                    cell.correction_factor,
                    false,
                    cell.frame_type_is_key,
                    cell.c.gf_index_frame_type == 0,
                )
            }
        };
        assert_eq!(
            (got.q, got.bottom_index, got.top_index),
            (want_q, want_bottom, want_top),
            "route={route:?} params = {:?}",
            cell.c
        );
        assert_eq!(
            arf_q_after_pick(cell.update_type, got.q, cell.c.arf_q),
            want_arf_q,
            "arf_q after the dispatcher; route={route:?} params = {:?}",
            cell.c
        );
    }
    assert!(
        routes[0] > 500 && routes[1] > 500 && routes[2] > 500,
        "the three dispatcher routes were not all exercised: \
         NoStatsCbr={} NoStats={} General={}",
        routes[0],
        routes[1],
        routes[2]
    );
}
