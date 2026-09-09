//! Differential harness for the rate-search layer of `av1/encoder/ratectrl.c`
//! vs the REAL C libaom v3.14.1.
//!
//! Two evidence tiers, deliberately mixed so each function gets the strongest
//! one available:
//!
//! * **Tier 1** — the exported functions, driven out of
//!   `upstream/build/libaom.a` through `shim/rcarchive_shim.c`, a TU that does
//!   NOT include ratectrl.c: `av1_get_MBs`, `av1_estimate_bits_at_q`,
//!   `av1_compute_qdelta_by_rate`, `av1_rc_regulate_q`,
//!   `av1_rc_compute_frame_size_bounds`, `av1_rc_set_frame_target`.
//! * **Tier 1c** — the file-statics, which have no exported symbol, through
//!   `shim/ratectrl_shim.c`'s verbatim-compiled copy: `resize_rate_factor`,
//!   `get_rate_factor_level`, `get_rate_correction_factor`,
//!   `get_bits_per_mb`, `find_qindex_by_rate`,
//!   `find_closest_qindex_by_rate`, `frame_type_qdelta`.
//!
//! `rate_search_shim_tu_matches_archive` closes the gap between them directly:
//! the two shims share `shim/rc_state_params.h` and build the same
//! `AV1_COMP`, so driving `av1_estimate_bits_at_q` and `av1_rc_regulate_q`
//! through BOTH and comparing measures exactly the "second compilation" risk
//! that separates 1c from 1 — on the functions this file actually tests, not
//! on a proxy.

use aom_encode::ratectrl_rate::{
    FrameUpdateType, RateFactorLevel, bits_per_mb, compute_frame_size_bounds,
    compute_qdelta_by_rate, estimate_bits_at_q, find_closest_qindex_by_rate, find_qindex_by_rate,
    frame_type_qdelta, get_mbs, rate_correction_factor, rate_factor_level, regulate_q,
    resize_rate_factor, set_frame_target,
};
use aom_sys_ref::{
    RefRcStateParams, ref_compute_frame_size_bounds, ref_compute_qdelta_by_rate,
    ref_estimate_bits_at_q, ref_get_mbs, ref_rcc_find_closest_qindex_by_rate,
    ref_rcc_find_qindex_by_rate, ref_rcc_frame_type_qdelta, ref_rcc_get_bits_per_mb,
    ref_rcc_get_rate_correction_factor, ref_rcc_get_rate_factor_level,
    ref_rcc_probe_estimate_bits_at_q, ref_rcc_probe_regulate_q, ref_rcc_resize_rate_factor,
    ref_regulate_q, ref_set_frame_target,
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

const BIT_DEPTHS: [u8; 3] = [8, 10, 12];
const UPDATE_TYPES: [(FrameUpdateType, i32); 7] = [
    (FrameUpdateType::Kf, 0),
    (FrameUpdateType::Lf, 1),
    (FrameUpdateType::Gf, 2),
    (FrameUpdateType::Arf, 3),
    (FrameUpdateType::Overlay, 4),
    (FrameUpdateType::IntnlOverlay, 5),
    (FrameUpdateType::IntnlArf, 6),
];

/// Frame sizes chosen to separate `av1_get_MBs`'s round-half-up from a
/// ceiling: sizes just above and just below a multiple of 8 and of 16.
const SIZES: [(i32, i32); 8] = [
    (16, 16),
    (33, 17),
    (176, 144),
    (352, 288),
    (639, 359),
    (1280, 720),
    (1921, 1081),
    (3840, 2160),
];

fn draw(rng: &mut Rng) -> (RefRcStateParams, u8, bool, bool, FrameUpdateType) {
    let bd = BIT_DEPTHS[rng.below(3) as usize];
    let (w, h) = SIZES[rng.below(SIZES.len() as u32) as usize];
    let (cfg_w, cfg_h) = SIZES[rng.below(SIZES.len() as u32) as usize];
    let frame_type = rng.below(4); // KEY / INTER / INTRA_ONLY / S_FRAME
    let screen = rng.boolean();
    let (update_type, c_update) = UPDATE_TYPES[rng.below(7) as usize];
    let best = rng.below(64);
    let worst = rng.range(best, 255);
    let p = RefRcStateParams {
        bit_depth: i32::from(bd),
        coded_width: w,
        coded_height: h,
        cfg_width: cfg_w,
        cfg_height: cfg_h,
        frame_type,
        screen_content: i32::from(screen),
        rc_mode: rng.below(4),
        rtc_mode: i32::from(rng.boolean()),
        stat_consumption: i32::from(rng.boolean()),
        refresh_golden: i32::from(rng.boolean()),
        refresh_alt_ref: i32::from(rng.boolean()),
        is_src_frame_alt_ref: i32::from(rng.boolean()),
        gf_cbr_boost_pct: rng.range(0, 100),
        update_type: c_update,
        layer_depth: rng.range(0, 8),
        best_quality: best,
        worst_quality: worst,
        base_qindex: rng.below(256),
        max_frame_bandwidth: rng.range(1000, 100_000_000),
        recode_tolerance: rng.range(0, 100),
        rate_correction_factors: [
            f64::from(rng.range(1, 5000)) / 100.0,
            f64::from(rng.range(1, 5000)) / 100.0,
            f64::from(rng.range(1, 5000)) / 100.0,
            f64::from(rng.range(1, 5000)) / 100.0,
        ],
    };
    // C's `frame_type == KEY_FRAME` test is on cm->current_frame.frame_type.
    let is_key = frame_type == 0;
    (p, bd, is_key, screen, update_type)
}

#[test]
fn rate_search_shim_tu_matches_archive() {
    // The tier-1c gap closer, measured on the functions this file tests: drive
    // the same exported function through the archive AND through the
    // verbatim-compiled copy, from the same parameter block.
    let mut rng = Rng(0x5151_2626_3737_4848);
    for _ in 0..4000 {
        let (p, _bd, _k, _s, _u) = draw(&mut rng);
        let q = rng.below(256);
        let cf = f64::from(rng.range(1, 5000)) / 100.0;
        assert_eq!(
            ref_rcc_probe_estimate_bits_at_q(&p, q, cf),
            ref_estimate_bits_at_q(&p, q, cf),
            "av1_estimate_bits_at_q: shim TU vs archive, params = {p:?}"
        );
        let target = rng.range(0, 5_000_000);
        let best = rng.below(200);
        let worst = rng.range(best, 255);
        assert_eq!(
            ref_rcc_probe_regulate_q(&p, target, best, worst, p.coded_width, p.coded_height),
            ref_regulate_q(&p, target, best, worst, p.coded_width, p.coded_height),
            "av1_rc_regulate_q: shim TU vs archive, params = {p:?}"
        );
    }
}

#[test]
fn get_mbs_matches_c() {
    // Every size 1..=64 in both axes plus a coarse sweep to 4096: the
    // round-half-up in ROUND_POWER_OF_TWO differs from a ceiling exactly at
    // the small sizes, so a sampled sweep would miss it.
    for w in 1..=64 {
        for h in 1..=64 {
            assert_eq!(get_mbs(w, h), ref_get_mbs(w, h), "{w}x{h}");
        }
    }
    for w in (1..=4096).step_by(7) {
        for h in (1..=4096).step_by(101) {
            assert_eq!(get_mbs(w, h), ref_get_mbs(w, h), "{w}x{h}");
        }
    }
}

#[test]
fn resize_rate_factor_matches_c() {
    for &(cw, ch) in &SIZES {
        for &(w, h) in &SIZES {
            assert_eq!(
                resize_rate_factor(cw, ch, w, h).to_bits(),
                ref_rcc_resize_rate_factor(cw, ch, w, h).to_bits(),
                "cfg {cw}x{ch} coded {w}x{h}"
            );
        }
    }
}

#[test]
fn rate_factor_level_matches_c() {
    for (ty, c_ty) in UPDATE_TYPES {
        assert_eq!(
            rate_factor_level(ty) as i32,
            ref_rcc_get_rate_factor_level(c_ty),
            "{ty:?}"
        );
    }
    // The four levels must also be distinct — a mapping that collapsed two of
    // them would still index a valid slot.
    let seen: Vec<i32> = UPDATE_TYPES
        .iter()
        .map(|&(t, _)| rate_factor_level(t) as i32)
        .collect();
    assert!(seen.contains(&(RateFactorLevel::KfStd as i32)));
    assert!(seen.contains(&(RateFactorLevel::InterNormal as i32)));
    assert!(seen.contains(&(RateFactorLevel::GfArfStd as i32)));
    assert!(seen.contains(&(RateFactorLevel::GfArfLow as i32)));
}

#[test]
fn rate_correction_factor_matches_c() {
    let mut rng = Rng(0x9999_8888_7777_6666);
    for _ in 0..20000 {
        let (p, _bd, is_key, _s, update_type) = draw(&mut rng);
        let got = rate_correction_factor(
            &p.rate_correction_factors,
            is_key,
            p.stat_consumption != 0,
            update_type,
            p.refresh_golden != 0,
            p.refresh_alt_ref != 0,
            p.is_src_frame_alt_ref != 0,
            /*use_svc=*/ false,
            p.rc_mode == 1,
            p.gf_cbr_boost_pct,
            p.cfg_width,
            p.cfg_height,
            p.coded_width,
            p.coded_height,
        );
        let want = ref_rcc_get_rate_correction_factor(&p, p.coded_width, p.coded_height);
        assert_eq!(got.to_bits(), want.to_bits(), "params = {p:?}");
    }
}

#[test]
fn bits_per_mb_matches_c() {
    let mut rng = Rng(0x1a2b_3c4d_5e6f_7080);
    for _ in 0..8000 {
        let (p, bd, is_key, screen, _u) = draw(&mut rng);
        let q = rng.below(256);
        let cf = f64::from(rng.range(1, 5000)) / 100.0;
        assert_eq!(
            bits_per_mb(is_key, screen, cf, q, bd),
            ref_rcc_get_bits_per_mb(&p, cf, q),
            "params = {p:?} q={q} cf={cf}"
        );
    }
}

#[test]
fn estimate_bits_at_q_matches_c() {
    let mut rng = Rng(0x0f1e_2d3c_4b5a_6978);
    let mut overhead_floor_hits = 0;
    for _ in 0..20000 {
        let (p, bd, is_key, screen, _u) = draw(&mut rng);
        let q = rng.below(256);
        let cf = f64::from(rng.range(1, 5000)) / 100.0;
        let mbs = get_mbs(p.coded_width, p.coded_height);
        let got = estimate_bits_at_q(is_key, screen, q, cf, bd, mbs);
        let want = ref_estimate_bits_at_q(&p, q, cf);
        assert_eq!(got, want, "params = {p:?} q={q} cf={cf} mbs={mbs}");
        if got == 200 {
            overhead_floor_hits += 1;
        }
    }
    // The FRAME_OVERHEAD_BITS floor must actually be reached, or the max() is
    // untested.
    assert!(
        overhead_floor_hits > 0,
        "the FRAME_OVERHEAD_BITS floor was never reached"
    );
}

#[test]
fn find_qindex_by_rate_matches_c() {
    let mut rng = Rng(0x7766_5544_3322_1100);
    for _ in 0..20000 {
        let (p, bd, _k, screen, _u) = draw(&mut rng);
        // C's find_qindex_by_rate takes frame_type as an ARGUMENT, separate
        // from cm->current_frame.frame_type, so sweep it independently.
        let frame_type = rng.below(4);
        let is_key = frame_type == 0;
        let desired = rng.range(0, 400000);
        let best = rng.below(200);
        let worst = rng.range(best, 255);
        assert_eq!(
            find_qindex_by_rate(desired, is_key, screen, bd, best, worst),
            ref_rcc_find_qindex_by_rate(&p, desired, frame_type, best, worst),
            "params = {p:?} desired={desired} ft={frame_type} [{best},{worst}]"
        );
    }
}

#[test]
fn compute_qdelta_by_rate_matches_c() {
    let mut rng = Rng(0x1029_3847_5666_7788);
    for _ in 0..20000 {
        let (p, bd, _k, screen, _u) = draw(&mut rng);
        let frame_type = rng.below(4);
        let is_key = frame_type == 0;
        let qindex = rng.below(256);
        let ratio = f64::from(rng.range(1, 400)) / 100.0;
        assert_eq!(
            compute_qdelta_by_rate(
                is_key,
                screen,
                qindex,
                ratio,
                bd,
                p.best_quality,
                p.worst_quality
            ),
            ref_compute_qdelta_by_rate(&p, frame_type, qindex, ratio),
            "params = {p:?} ft={frame_type} qindex={qindex} ratio={ratio}"
        );
    }
}

#[test]
fn frame_type_qdelta_matches_c() {
    let mut rng = Rng(0x4433_2211_8877_6655);
    let mut nonzero = 0;
    for _ in 0..20000 {
        let (p, bd, _k, screen, update_type) = draw(&mut rng);
        let q = rng.below(256);
        // frame_type_qdelta reads gf_group->frame_type[gf_index], which the
        // shim sets from the same `frame_type` field.
        let got = frame_type_qdelta(
            update_type,
            p.frame_type == 0,
            p.layer_depth,
            screen,
            q,
            bd,
            p.best_quality,
            p.worst_quality,
        );
        let want = ref_rcc_frame_type_qdelta(&p, q);
        assert_eq!(got, want, "params = {p:?} q={q}");
        if got != 0 {
            nonzero += 1;
        }
    }
    // INTER_NORMAL frames take ratio 1.0 and mostly yield 0; if EVERY cell
    // were 0 the arf_layer_deltas table would be untested.
    assert!(nonzero > 1000, "arf_layer_deltas barely reached: {nonzero}");
}

#[test]
fn find_closest_qindex_by_rate_matches_c() {
    let mut rng = Rng(0xcafe_d00d_f00d_babe);
    let mut picked_prev = 0;
    for _ in 0..20000 {
        let (p, bd, is_key, screen, _u) = draw(&mut rng);
        let cf = f64::from(rng.range(1, 5000)) / 100.0;
        let best = rng.below(200);
        let worst = rng.range(best, 255);
        // Aim `desired` AT a real modelled rate, jittered by a few percent.
        // Drawing it uniformly from a wide range puts the answer at an
        // endpoint almost every time, and the "previous qindex was closer"
        // branch — half the function — then goes essentially untested (it
        // fired 4 times in 20,000 with a uniform draw).
        let anchor_q = rng.range(best, worst);
        let anchor = bits_per_mb(is_key, screen, cf, anchor_q, bd);
        let jitter = rng.range(-8, 8);
        let desired = anchor.saturating_add((i64::from(anchor) * i64::from(jitter) / 100) as i32);
        let got = find_closest_qindex_by_rate(desired, is_key, screen, cf, bd, best, worst);
        let want = ref_rcc_find_closest_qindex_by_rate(&p, desired, cf, best, worst);
        assert_eq!(
            got, want,
            "params = {p:?} desired={desired} cf={cf} [{best},{worst}]"
        );
        // Track how often the "previous qindex was closer" branch wins; if it
        // never did, half the function would be untested. NOTE: the reference
        // point has to be the binary search run with THIS correction factor —
        // `find_qindex_by_rate` hardcodes `correction_factor = 1.0`, so using
        // it here compares two different rate models and reports ~0 hits
        // regardless of what the generator does.
        let mut lo = best;
        let mut hi = worst;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if bits_per_mb(is_key, screen, cf, mid, bd) > desired {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if got == lo - 1 {
            picked_prev += 1;
        }
    }
    assert!(
        picked_prev > 1000,
        "the prev-qindex branch — half of find_closest_qindex_by_rate — was \
         taken only {picked_prev} times in 20,000, so it is barely tested"
    );
}

#[test]
fn regulate_q_matches_c() {
    let mut rng = Rng(0xfeed_beef_1234_5678);
    for _ in 0..20000 {
        let (mut p, bd, is_key, screen, update_type) = draw(&mut rng);
        // The AOM_CBR tail (adjust_q_cbr) is not ported, so drive the oracle
        // off that arm rather than comparing against a zeroed one. This MUST
        // happen before the correction factor is derived: rc_mode also
        // selects an arm inside get_rate_correction_factor, so substituting
        // afterwards would hand the port a factor C never used.
        if p.rc_mode == 1 {
            p.rc_mode = 3; // AOM_Q
        }
        // regulate_q derives its own correction factor from cpi; the port
        // takes it as a parameter, so compute the same one.
        let cf = rate_correction_factor(
            &p.rate_correction_factors,
            is_key,
            p.stat_consumption != 0,
            update_type,
            p.refresh_golden != 0,
            p.refresh_alt_ref != 0,
            p.is_src_frame_alt_ref != 0,
            false,
            p.rc_mode == 1,
            p.gf_cbr_boost_pct,
            p.cfg_width,
            p.cfg_height,
            p.coded_width,
            p.coded_height,
        );
        let target = rng.range(0, 5_000_000);
        let best = rng.below(200);
        let worst = rng.range(best, 255);
        let got = regulate_q(
            target,
            best,
            worst,
            p.coded_width,
            p.coded_height,
            is_key,
            screen,
            cf,
            bd,
        );
        let want = ref_regulate_q(&p, target, best, worst, p.coded_width, p.coded_height);
        assert_eq!(got, want, "params = {p:?} target={target} [{best},{worst}]");
    }
}

#[test]
fn compute_frame_size_bounds_matches_c() {
    let mut rng = Rng(0x0102_0304_0506_0708);
    let mut q_cells = 0;
    let mut other_cells = 0;
    for _ in 0..20000 {
        let (p, _bd, _k, _s, _u) = draw(&mut rng);
        // Both the tiny targets (where the 100-bit tolerance floor bites) and
        // large ones (where max_frame_bandwidth clips the over-shoot limit).
        let target = if rng.boolean() {
            rng.range(0, 400)
        } else {
            rng.range(0, 200_000_000)
        };
        let got = compute_frame_size_bounds(
            p.rc_mode == 3,
            target,
            p.recode_tolerance,
            p.max_frame_bandwidth,
        );
        let want = ref_compute_frame_size_bounds(&p, target);
        assert_eq!(
            (got.under_shoot, got.over_shoot),
            want,
            "params = {p:?} target={target}"
        );
        if p.rc_mode == 3 {
            q_cells += 1;
        } else {
            other_cells += 1;
        }
    }
    assert!(
        q_cells > 1000 && other_cells > 1000,
        "one arm went untested"
    );
}

#[test]
fn set_frame_target_matches_c() {
    let mut rng = Rng(0x1111_0000_2222_3333);
    for _ in 0..20000 {
        let (p, _bd, _k, _s, _u) = draw(&mut rng);
        let target = rng.range(0, 200_000_000);
        // The shim builds an UNSCALED frame (render == coded == upscaled), so
        // av1_frame_scaled(cm) is false and the rescale is not taken. That is
        // the frame the fixed-Q envelope codes.
        let got = set_frame_target(
            target,
            p.coded_width,
            p.coded_height,
            /*frame_scaled=*/ false,
            p.rc_mode == 1,
            p.cfg_width,
            p.cfg_height,
        );
        let want = ref_set_frame_target(&p, target, p.coded_width, p.coded_height);
        assert_eq!(
            (got.this_frame_target, got.sb64_target_rate),
            want,
            "params = {p:?} target={target}"
        );
    }
}
