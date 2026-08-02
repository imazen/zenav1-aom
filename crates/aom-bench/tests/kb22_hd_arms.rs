//! KB-22 follow-ups — the two gaps its close named, both on the frame-size axis
//! that no pre-existing encoder gate reaches (every other cell in this harness
//! encodes at most 640x640).
//!
//! 1. **The 720p..2159p isolation cell.** KB-22's only e2e evidence is at
//!    2160x2160, where the KB-19 `is_4k_or_larger` arm is ALSO live, so the two
//!    arms are only jointly proven. `qindex_arm_720p_isolation_e2e_byte_match`
//!    encodes 1280x720 cq32 (base_qindex 128) at speed 0, where the KB-22
//!    `av1_set_speed_features_qindex_dependent` arm (speed_features.c:2914) IS
//!    live and the KB-19 arm (speed_features.c:187-189, `min(w,h) >= 2160`) is
//!    NOT. It also reaches a `perform_coeff_opt` value no other cell produces:
//!    `2 + is_1080p_or_larger` = **2** at 720p, versus 3 at the 2160p cell.
//!
//! 2. **The loop-restoration unit-size bounds** (`lpf_sf.min_lr_unit_size` /
//!    `max_lr_unit_size`, speed_features.c:3080-3108). KB-22 recorded these as
//!    "still unmodelled by the port ... a `--enable-restoration=1` cell at
//!    >=720p and speed>=1 is expected to diverge". **That prediction is
//!    wrong, and this file is the measurement that says so** — see
//!    `lr_unit_size_bounds_track_c` (the derivation, default tier) and
//!    `lr_unit_size_hd_speed1_e2e` (the encode, on-demand tier). The fields are
//!    unmodelled in `aom_encode::SpeedFeatures` only; the port's loop-restoration
//!    search takes its unit-size range from its caller, and that caller
//!    (`aom_bench::lr_search_sf_allintra` / `..._good`) already transcribes the
//!    whole block including both framesize arms.
//!
//! The two e2e tests are `#[ignore]`d — an HD speed-0 encode pair costs tens of
//! seconds — and belong to the same nightly / on-demand tier as
//! `kb19_min_partition_4k`. Run them with:
//!
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb22_hd_arms -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, lr_search_sf_allintra, lr_search_sf_good, parse_restoration_decision};
use aom_sys_ref as c;

/// Mirror-tile a decoded cell up to `w x h` at quality `cq`. Mirroring (rather
/// than wrapping) keeps the seam continuous, so the enlarged frame stays
/// photographic instead of acquiring a synthetic edge grid every tile period.
/// Same recipe as `kb19_min_partition_4k` and the size axis of the
/// config-permutation gate.
fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mir = |i: usize, n: usize| {
        let m = i % (2 * n);
        if m < n { m } else { 2 * n - 1 - m }
    };
    let (bw, bh) = (base.w, base.h);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = base.y[mir(r, bh) * bw + mir(col, bw)];
        }
    }
    let (bcw, bch) = ((bw + base.ss_x) >> base.ss_x, (bh + base.ss_y) >> base.ss_y);
    let (cw, ch) = ((w + base.ss_x) >> base.ss_x, (h + base.ss_y) >> base.ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = base.u[mir(r, bch) * bcw + mir(col, bcw)];
            v[r * cw + col] = base.v[mir(r, bch) * bcw + mir(col, bcw)];
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono: base.mono,
        ss_x: base.ss_x,
        ss_y: base.ss_y,
        usage: base.usage,
        cq_level: cq,
        speed,
        bd: base.bd,
        y,
        u,
        v,
    }
}

// ---------------------------------------------------------------------------
// 1. The 720p..2159p isolation cell for the KB-22 qindex arm.
// ---------------------------------------------------------------------------

/// 1280x720 bd8 4:2:0 real content, speed-0 ALLINTRA KEY, stock knobs — a hard
/// byte-identity gate vs real aomenc, in the band where the KB-22 arm is live
/// ALONE.
///
/// Why this shape:
/// - `min(w, h) = 720` → `is_720p_or_larger` holds, so
///   `av1_set_speed_features_qindex_dependent`'s speed-0 arm
///   (speed_features.c:2914) fires;
/// - `min(w, h) < 1080` → `is_1080p_or_larger` is FALSE, so
///   `rd_sf.perform_coeff_opt = 2 + is_1080p_or_larger` resolves to **2** — a
///   `coeff_opt_thresholds` row (:88-98, DEFAULT_EVAL dist gate 1600) that no
///   other cell in the suite reaches. The 2160p cell resolves 3 (gate 864) and
///   every sub-720p cell resolves 1 (gate 3200);
/// - `min(w, h) < 2160` → KB-19's `is_4k_or_larger` arm
///   (speed_features.c:187-189) does NOT fire, so `default_min_partition_size`
///   stays `BLOCK_4X4` exactly as it does on the small cells. This is what makes
///   the cell an isolation of the KB-22 arm rather than a second joint test;
/// - `--cq-level 32` → `av1_quantizer_to_qindex` (av1_quantize.c:1033) maps to
///   `base_qindex` 128, sitting exactly ON the arm's `<= 128` boundary.
///
/// **MEASURED 2026-07-31, reference box aarch64-apple-darwin,
/// `--profile test-fast`** (`benchmarks/kb23_partial_sb_2026-07-31.tsv`): port
/// **85,441 B == C 85,441 B**, C 8.6 s / port 25.6 s. Bite proof — stubbing the
/// arm body to `if false && is_720p_or_larger && base_qindex <= 128` fails this
/// cell at 85,667 B (**+226**), while 61 of the 62 `aom-encode` lib tests stay
/// green (the one failure being that arm's own derivation test).
#[test]
#[ignore = "1280x720 speed-0 encode pair costs tens of seconds; nightly / on-demand tier"]
fn qindex_arm_720p_isolation_e2e_byte_match() {
    c::ref_init();
    let base = EncodeCell::real_content("kb22base", "av1-1-b8-00-quantizer-00", None, 32, 0);
    let cell = mirror_tile(&base, "kb22_720p", 1280, 720, 32, 0);

    // Reach assertions — the whole point of the cell is WHICH arms it lights up.
    assert_eq!((cell.w, cell.h), (1280, 720));
    assert!(
        cell.w.min(cell.h) >= 720,
        "the cell must reach the KB-22 is_720p_or_larger arm"
    );
    assert!(
        cell.w.min(cell.h) < 1080,
        "the cell must sit BELOW is_1080p_or_larger so perform_coeff_opt resolves 2, \
         not the 3 the 2160p cell already covers"
    );
    assert!(
        cell.w.min(cell.h) < 2160,
        "the cell must sit BELOW KB-19's is_4k_or_larger arm — otherwise this is not \
         an isolation of the KB-22 arm"
    );
    assert_eq!(
        cell.cq_level, 32,
        "cq32 -> base_qindex 128, exactly on the arm's `base_qindex <= 128` boundary"
    );
    assert_eq!(cell.speed, 0, "the modelled arm is the `speed == 0` block");

    let t0 = std::time::Instant::now();
    let tu = cell.c_encode();
    let c_ms = t0.elapsed().as_millis();
    let real = EncodeCell::frame_obu_payload(&tu);

    let t1 = std::time::Instant::now();
    let port = cell.port_encode(&tu);
    let port_ms = t1.elapsed().as_millis();

    println!(
        "  kb22 1280x720 cq32 speed0 (720p<=q128 arm ALONE): port {} B ({port_ms} ms) vs \
         C {} B ({c_ms} ms) -> {}",
        port.len(),
        real.len(),
        if port == real { "MATCH" } else { "DIVERGE" }
    );

    assert!(
        port == real,
        "the 720p isolation cell is not byte-identical to real aomenc (port {} B vs C {} B, \
         delta {:+}). This cell is live for exactly ONE modelled arm that the sub-720p gates \
         are not: `SpeedFeatures::apply_allintra_qindex_dependent`'s \
         `is_720p_or_larger && base_qindex <= 128` block (speed_features.c:2914), at its \
         `perform_coeff_opt = 2` / `intra_tx_size_search_init_depth_rect = 1` resolution. \
         KB-19's `is_4k_or_larger` arm is NOT live here.",
        port.len(),
        real.len(),
        port.len() as i64 - real.len() as i64
    );
}

// ---------------------------------------------------------------------------
// 2. The loop-restoration unit-size bounds — the KB-22 "live gap" prediction.
// ---------------------------------------------------------------------------

/// `min_lr_unit_size` / `max_lr_unit_size` (speed_features.c:3080-3108) as the
/// port's loop-restoration search actually receives them, pinned against
/// hand-derived expectations at every boundary of the C block, in BOTH
/// directions.
///
/// The C block, verbatim (v3.14.1, `av1_set_speed_features_qindex_dependent`):
///
/// ```text
/// sf->lpf_sf.min_lr_unit_size = RESTORATION_PROC_UNIT_SIZE;   //  64   :3082
/// sf->lpf_sf.max_lr_unit_size = RESTORATION_UNITSIZE_MAX;     // 256   :3083
/// if (speed >= 1) {                                           //       :3085
///   if (is_1440p_or_larger)      min = RESTORATION_UNITSIZE_MAX;      // :3089
///   else if (is_720p_or_larger)  min = RESTORATION_UNITSIZE_MAX >> 1; // :3091
/// }
/// if (speed >= 3 || (mode == ALLINTRA && speed >= 1)) {        //       :3095
///   if (base_qindex <= 96 && !is_1440p_or_larger) min = max = 128;    // :3102
///   else                                          min = max = 256;    // :3105
/// }
/// ```
///
/// Two consequences worth stating, because they are what makes the KB-22
/// prediction wrong on TWO independent counts:
///
/// - for ALLINTRA at `speed >= 1` the second block is **entirely overwritten**
///   by the third, so the only framesize term that survives is
///   `is_1440p_or_larger`. A 720p or 1080p allintra cell therefore resolves the
///   SAME `(min, max)` as a 64x64 one — 720p is not a frame size at which the
///   allintra bounds change at all;
/// - the bounds are not unmodelled. `SpeedFeatures` carries no field for them
///   because the port's LR search takes its `LrSearchSf` from its caller, and
///   `aom_bench::lr_search_sf_allintra` / `..._good` transcribe the block
///   above — including both framesize arms — already.
///
/// Expected values below are hand-derived from that listing, not recomputed by
/// a second copy of the algorithm.
#[test]
fn lr_unit_size_bounds_track_c() {
    // (speed, w, h, qindex, expected_min, expected_max)
    const ALLINTRA: &[(i32, usize, usize, i32, i32, i32)] = &[
        // speed 0: the full 64..256 descent, at every size and qindex.
        (0, 64, 64, 96, 64, 256),
        (0, 1280, 720, 96, 64, 256),
        (0, 2560, 1440, 96, 64, 256),
        (0, 2560, 1440, 255, 64, 256),
        // speed 1: the ALLINTRA arm of the third block fires, overwriting the
        // second. Below 1440p the qindex 96 boundary is the only live term.
        (1, 64, 64, 96, 128, 128),
        (1, 64, 64, 97, 256, 256),
        (1, 640, 640, 96, 128, 128),
        // 719 vs 720: NOT a boundary for allintra (the >=720p arm is overwritten).
        (1, 1280, 719, 96, 128, 128),
        (1, 1280, 720, 96, 128, 128),
        (1, 1280, 720, 97, 256, 256),
        (1, 1920, 1080, 96, 128, 128),
        // 1439 vs 1440: THE live framesize boundary, and only at qindex <= 96.
        (1, 2560, 1439, 96, 128, 128),
        (1, 2560, 1440, 96, 256, 256),
        (1, 2560, 1439, 97, 256, 256),
        (1, 2560, 1440, 97, 256, 256),
        // Vertical framing: the predicates are on min(w, h), so the transpose
        // must resolve identically.
        (1, 1440, 2560, 96, 256, 256),
        (1, 1439, 2560, 96, 128, 128),
        // speeds 2..9 take the same arm (restoration is switched off elsewhere
        // at speed >= 5 — that is the seq bit, not these bounds).
        (2, 1280, 720, 96, 128, 128),
        (3, 1280, 720, 96, 128, 128),
        (4, 2560, 1440, 96, 256, 256),
        (9, 64, 64, 97, 256, 256),
    ];

    // GOOD: the third block needs speed >= 3, so speeds 1-2 expose the SECOND
    // block's framesize arms un-overwritten — the only place where 720p really
    // is a live unit-size boundary.
    const GOOD: &[(i32, usize, usize, i32, i32, i32)] = &[
        (0, 2560, 1440, 96, 64, 256),
        (1, 640, 640, 96, 64, 256),
        (1, 1280, 719, 96, 64, 256),
        (1, 1280, 720, 96, 128, 256),
        (1, 1280, 720, 255, 128, 256),
        (1, 2560, 1439, 96, 128, 256),
        (1, 2560, 1440, 96, 256, 256),
        (2, 1280, 720, 96, 128, 256),
        // speed >= 3 joins the allintra rule.
        (3, 640, 640, 96, 128, 128),
        (3, 640, 640, 97, 256, 256),
        (3, 1280, 720, 96, 128, 128),
        (3, 2560, 1440, 96, 256, 256),
        (5, 2560, 1439, 97, 256, 256),
    ];

    let mut checked = 0usize;
    for &(speed, w, h, q, emin, emax) in ALLINTRA {
        let sf = lr_search_sf_allintra(speed, q, w, h, false);
        assert_eq!(
            (sf.min_lr_unit_size, sf.max_lr_unit_size),
            (emin, emax),
            "ALLINTRA speed {speed} {w}x{h} qindex {q}: LR unit-size bounds \
             (speed_features.c:3080-3108)"
        );
        checked += 1;
    }
    for &(speed, w, h, q, emin, emax) in GOOD {
        let sf = lr_search_sf_good(speed, q, w, h, false);
        assert_eq!(
            (sf.min_lr_unit_size, sf.max_lr_unit_size),
            (emin, emax),
            "GOOD speed {speed} {w}x{h} qindex {q}: LR unit-size bounds \
             (speed_features.c:3080-3108)"
        );
        checked += 1;
    }
    assert_eq!(checked, ALLINTRA.len() + GOOD.len());

    // Anti-vacuity: the table must actually contain rows that DISAGREE on each
    // live term, otherwise it could pass against a constant.
    let distinct: std::collections::BTreeSet<(i32, i32)> = ALLINTRA
        .iter()
        .chain(GOOD.iter())
        .map(|&(_, _, _, _, mi, ma)| (mi, ma))
        .collect();
    assert!(
        distinct.len() >= 4,
        "the expectation table collapses to {} distinct outcomes — it cannot be \
         distinguishing the arms",
        distinct.len()
    );

    // `allow_screen_content_tools` must not touch the unit-size bounds (it only
    // feeds `prune_sgr_based_on_wiener` at speed >= 3).
    for &sc in &[false, true] {
        let sf = lr_search_sf_allintra(1, 96, 1280, 720, sc);
        assert_eq!((sf.min_lr_unit_size, sf.max_lr_unit_size), (128, 128));
        let sf = lr_search_sf_good(3, 97, 2560, 1440, sc);
        assert_eq!((sf.min_lr_unit_size, sf.max_lr_unit_size), (256, 256));
    }
}

/// The encode KB-22 predicted would diverge: `--enable-restoration=1` at
/// >=720p and speed >= 1, with the frame's `base_qindex` placed on both sides
/// of the `<= 96` threshold that selects the unit size.
///
/// Each quality point runs TWO pairs at the same size and speed:
/// - the restoration-ON pair (`c_encode_lr` / `port_encode_lr`), which is the
///   prediction's cell;
/// - a restoration-OFF control (`c_encode` / `port_encode`) at the identical
///   cell, which separates "the LR path is wrong" from "the base encode at this
///   size and speed is wrong". Without the control a divergence in the ON pair
///   proves nothing about loop restoration.
///
/// The parsed frame-header restoration decision (per-plane
/// `frame_restoration_type` and the CODED unit sizes) is printed for both
/// streams: if the unit-size bounds really were unmodelled, the coded unit size
/// is where it would show, and it is a bitstream fact rather than a C internal.
///
/// **MEASURED 2026-07-31, reference box aarch64-apple-darwin,
/// `--profile test-fast`, mirror-tiled `av1-1-b8-00-quantizer-00` at 1280x720
/// speed 1** (`benchmarks/kb23_partial_sb_2026-07-31.tsv`):
///
/// | cq | base_qindex | derived (min,max) | C coded unit size | port coded unit size | LR-ON delta | LR-OFF control delta |
/// |---|---|---|---|---|---|---|
/// | 24 | 96 (`<= 96`) | (128, 128) | 128,128,128 | 128,128,128 | -3 B | **-3 B** |
/// | 25 | 100 (`> 96`) | (256, 256) | 256,256,256 | 256,256,256 | +31 B | **+31 B** |
///
/// **The KB-22 prediction does not hold.** The port codes exactly C's
/// restoration unit size at BOTH sides of the threshold, and the coded size IS
/// the derived bound — so `min_lr_unit_size` / `max_lr_unit_size` reach the
/// bitstream correctly, at every cell, before and after the fix below. The
/// residual byte delta was identical with restoration ON and OFF, i.e. entirely
/// in the base encode: **KB-23**, the frame-edge intra-CNN prune that
/// `kb23_partial_sb_size_and_speed_axis` mapped and
/// `partition_pick.rs`'s `cnn_root_whole_in_frame` closed. With that landed all
/// four pairs are byte-identical (both deltas 0), and the assertions below are
/// hard byte gates.
#[test]
#[ignore = "four 1280x720 encode pairs; nightly / on-demand tier"]
fn lr_unit_size_hd_speed1_e2e() {
    c::ref_init();
    let base = EncodeCell::real_content("kb22lrbase", "av1-1-b8-00-quantizer-00", None, 32, 0);

    // cq24 -> base_qindex 96 (ON the `<= 96` threshold -> unit size 128),
    // cq25 -> base_qindex 100 (OFF it -> unit size 256).
    // (`quantizer_to_qindex`, av1_quantize.c:1033.)
    let mut exact_on = 0usize;
    let mut exact_off = 0usize;
    let mut rows = Vec::new();

    // Real aomenc's payload sizes for these cells, pinned so that an oracle or
    // cell-recipe change is reported as such instead of silently re-baselining
    // the deltas below (the `kb19_min_partition_4k` convention).
    // (cq, derived unit-size bound, C LR-ON bytes, C LR-OFF bytes)
    const PINNED: &[(i32, i32, usize, usize)] =
        &[(24, 128, 137_600, 137_380), (25, 256, 129_451, 129_403)];

    for &(cq, expect_bound, c_on_len, c_off_len) in PINNED {
        let cell = mirror_tile(&base, &format!("lr720_s1_cq{cq}"), 1280, 720, cq, 1);
        assert!(cell.w.min(cell.h) >= 720);
        assert!(cell.w.min(cell.h) < 1440);
        let sf = lr_search_sf_allintra(cell.speed, if cq == 24 { 96 } else { 100 }, cell.w, cell.h, false);
        assert_eq!(
            (sf.min_lr_unit_size, sf.max_lr_unit_size),
            (expect_bound, expect_bound),
            "cq{cq}: the port's derived LR unit-size bounds"
        );

        // --- restoration ON: the predicted-divergent pair.
        let t0 = std::time::Instant::now();
        let c_on = cell.c_encode_lr();
        let c_on_ms = t0.elapsed().as_millis();
        assert!(!c_on.is_empty(), "cq{cq}: real LR encode failed");
        let t1 = std::time::Instant::now();
        let p_on = cell.port_encode_lr(&c_on);
        let p_on_ms = t1.elapsed().as_millis();
        let real_on = EncodeCell::frame_obu_payload(&c_on);
        assert_eq!(
            real_on.len(),
            c_on_len,
            "cq{cq}: the C restoration-ON reference itself moved — the oracle build or the \
             cell recipe changed, so nothing below is comparable to the pinned measurement"
        );
        let on_match = p_on == real_on;
        if on_match {
            exact_on += 1;
        }

        let (c_frt, c_us) = parse_restoration_decision(&c_on);
        let port_tu = aom_bench::rd_close::splice_frame_obu(&c_on, &p_on);
        let (p_frt, p_us) = parse_restoration_decision(&port_tu);

        // --- restoration OFF control at the identical cell.
        let c_off = cell.c_encode();
        let real_off = EncodeCell::frame_obu_payload(&c_off);
        assert_eq!(
            real_off.len(),
            c_off_len,
            "cq{cq}: the C restoration-OFF control reference itself moved — see above"
        );
        let p_off = cell.port_encode(&c_off);
        let off_match = p_off == real_off;
        if off_match {
            exact_off += 1;
        }

        println!(
            "  cq{cq} (qindex {}) 1280x720 speed1: LR-ON port {} B vs C {} B ({}) \
             [C {c_on_ms} ms, port {p_on_ms} ms] | decision C frt={c_frt:?} us={c_us:?} \
             port frt={p_frt:?} us={p_us:?} | LR-OFF control port {} B vs C {} B ({})",
            if cq == 24 { 96 } else { 100 },
            p_on.len(),
            real_on.len(),
            if on_match { "MATCH" } else { "DIVERGE" },
            p_off.len(),
            real_off.len(),
            if off_match { "MATCH" } else { "DIVERGE" },
        );
        rows.push((
            cq,
            on_match,
            off_match,
            c_frt,
            c_us,
            p_frt,
            p_us,
            expect_bound,
            p_on.len() as i64 - real_on.len() as i64,
            p_off.len() as i64 - real_off.len() as i64,
        ));
    }

    // Anti-vacuity: the reference must actually restore somewhere on this grid,
    // otherwise an all-NONE header would let the unit-size question go
    // unasked. (The coded unit size is only meaningful when a plane restores.)
    let real_restores = rows.iter().filter(|r| r.3.iter().any(|&t| t != 0)).count();
    assert_eq!(
        real_restores,
        rows.len(),
        "a reference stream coded an all-NONE restoration header, so its coded unit size \
         carries no information — the unit-size question would go unasked on that cell"
    );
    println!(
        "  LR-HD summary: {}/{} restoration-ON pairs byte-exact, {}/{} controls byte-exact, \
         {real_restores}/{} reference streams actually restore a plane",
        exact_on,
        rows.len(),
        exact_off,
        rows.len(),
        rows.len(),
    );

    // ---- The LOOP-RESTORATION statement (hard, and it PASSES). --------------
    // This is the direct test of the KB-22 prediction: `min_lr_unit_size` /
    // `max_lr_unit_size` are exactly the quantities that select the coded
    // restoration unit size, and the coded size is a bitstream fact rather than
    // a C internal. If the bounds were unmodelled, this is where it would show.
    for (cq, _, _, c_frt, c_us, p_frt, p_us, bound, _, _) in &rows {
        assert_eq!(
            c_frt, p_frt,
            "cq{cq}: the port's frame_restoration_type differs from real aomenc's"
        );
        assert_eq!(
            c_us, p_us,
            "cq{cq}: the port coded a DIFFERENT loop-restoration unit size than real \
             aomenc — this is what an unmodelled `min_lr_unit_size`/`max_lr_unit_size` \
             (speed_features.c:3080-3108) would look like"
        );
        // And the coded size IS the derived bound, on BOTH sides of the
        // qindex-96 threshold — so the bound is not merely equal by accident of
        // the search landing on the same answer from a wider range.
        assert_eq!(
            *c_us,
            [*bound; 3],
            "cq{cq}: real aomenc coded a unit size other than the bound the C block derives"
        );
    }

    // ---- The residual divergence is NOT the loop-restoration path. ----------
    // The ON and OFF deltas being equal is the load-bearing claim: whatever is
    // wrong is already wrong before restoration runs, and restoration adds
    // nothing on top of it.
    for (cq, _, _, _, _, _, _, _, on_delta, off_delta) in &rows {
        assert_eq!(
            on_delta, off_delta,
            "cq{cq}: the restoration-ON byte delta ({on_delta:+}) differs from the \
             restoration-OFF control's ({off_delta:+}) — the loop-restoration path IS \
             contributing to the divergence after all, which contradicts the KB-23 \
             attribution and needs re-localizing"
        );
    }

    // ---- Hard byte gates (KB-23 closed 2026-07-31). -------------------------
    // Stated precisely, because the difference matters: these were written as
    // self-promoting pins over the pre-KB-23 divergence and held that form for
    // exactly one measurement round, then were converted BY HAND when the
    // `cnn_output_valid` latch landed in the same session. They were never
    // committed in the pinned form, so no pin "fired" here — the fired pin was
    // `encoder_gate_real_content_speed1to4_e2e`'s, which promoted 6 cells
    // unprompted (playbook §5, and it is what surfaced the KB-13
    // misattribution). A regression on either side now fails here.
    assert_eq!(
        exact_off,
        rows.len(),
        "the restoration-OFF control at 1280x720 speed 1 is no longer byte-identical to \
         real aomenc. That is a KB-23 regression (the frame-edge intra-CNN latch in \
         `partition_pick.rs`), NOT a loop-restoration failure — see \
         `kb23_partial_sb_size_and_speed_axis`."
    );
    assert_eq!(
        exact_on,
        rows.len(),
        "a restoration-ON cell at >=720p speed 1 is no longer byte-identical while its \
         restoration-OFF control is — the divergence IS in the loop-restoration path"
    );
}

/// **KB-23 — the frame-size x speed gate that found and now guards the
/// intra-CNN `cnn_output_valid` bug.**
///
/// `lr_unit_size_hd_speed1_e2e`'s restoration-OFF control diverged at 1280x720,
/// and no pre-existing gate encodes above 640x640 at speed >= 1 (the
/// config-permutation speed axis runs 64x64 and 128x128 only —
/// `benchmarks/config_perm_speed_axis_2026-07-30.tsv`, both SB-exact sizes).
/// This walks the same content across the two framesize predicates
/// `set_allintra_speed_feature_framesize_dependent` uses at speed 1
/// (`is_480p_or_larger`, `is_720p_or_larger`, speed_features.c:175-233), both
/// sides of each, INTERLEAVED with sizes that are and are not exact multiples
/// of the 64-px superblock — so "a framesize bucket" and "frame-edge partial
/// superblocks" are separable by the result pattern alone rather than guessed
/// at. Phase 2 adds the speed axis on the four cheapest sizes.
///
/// **MEASURED 2026-07-31** (`benchmarks/kb23_partial_sb_2026-07-31.tsv`). The
/// split was total, and it was NOT a framesize bucket — 480x480 diverged while
/// 512/640/704 all matched, so no `is_480p_or_larger` arm can explain it:
///
/// Three states, kept distinct because the attribution depends on it — (A) base
/// `999d295`, (B) base + the KB-23 latch, (C) rebased onto `8a0faa7` (KB-21
/// root #3, the quant matrix in the SATD trellis-skip arm) with the latch:
///
/// | | (A) base | (B) +KB-23 | (C) +KB-21 #3 |
/// |---|---|---|---|
/// | speed 1, partial-SB sizes | **5/5 divergent** | 0/6 | 0/6 |
/// | speed 1, SB-exact sizes | 0/6 | 0/6 | 0/6 |
/// | speeds 0..3, partial-SB (132/196) | 6/8 divergent (speed 0 clean) | 0/8 | 0/8 |
/// | speed 4, all four sizes | 4/4 divergent | **4/4 divergent** | 0/4 |
///
/// A→B is KB-23's: it moves every partial-SB cell at speeds 1..3 and moves
/// NOTHING at speed 4 — the speed-4 SB-exact deltas are byte-identical in A and
/// B (-1 at 192x192, -25 at 256x256), which is the evidence that KB-23's fix
/// does not reach into KB-21's band. B→C is KB-21 root #3's, and this file
/// neither owns nor claims it.
#[test]
#[ignore = "twelve speed-1 encode pairs up to 1280x720 plus a small speed sweep; nightly / on-demand tier"]
fn kb23_partial_sb_size_and_speed_axis() {
    c::ref_init();
    let base = EncodeCell::real_content("kb22sizebase", "av1-1-b8-00-quantizer-00", None, 24, 0);
    // Square sizes so `min(w, h)` IS the label, plus the real 1280x720 shape.
    const SIZES: &[(usize, usize)] = &[
        // Exact multiples of 64 (no partial superblock) vs non-multiples,
        // interleaved across both framesize predicates so the two explanations
        // — "a framesize bucket" and "frame-edge partial superblocks" — are
        // separable by the result pattern alone.
        (132, 132), // 2.06 SB  partial   sub-480p
        (192, 192), // 3    SB  exact     sub-480p
        (196, 196), // 3.06 SB  partial   sub-480p
        (256, 256), // 4    SB  exact     sub-480p
        (448, 448), // 7    SB  exact     sub-480p
        (480, 480), // 7.5  SB  partial   is_480p_or_larger boundary
        (512, 512), // 8    SB  exact     >=480p
        (640, 640), // 10   SB  exact     >=480p (largest any other gate reaches)
        (704, 704), // 11   SB  exact     >=480p
        (720, 720), // 11.25 SB partial   is_720p_or_larger boundary
        (1280, 720), // partial rows      >=720p
        // A size where the MI-ALIGNED dims and the TRUE crop disagree ACROSS a
        // 64-px superblock boundary: mi_dim(250) * 4 = 256, so the mi extent
        // says 256 while the frame is 250 tall, and the last whole 64x64
        // (origin 192, window rows 191..256) reads 6 rows past the crop.
        // **KB-28 (2026-08-02) settled what that row does and does not show.**
        // It exercises the CNN *window* only, which is inert either way: C
        // reads the border-extended source with no clamp at all
        // (partition_strategy.c:205-220) and every read past the crop returns
        // the replicated edge pixel, so a clamp to the crop and a clamp to the
        // mi extent produce identical windows. It does NOT reach the CNN
        // *res-tier threshold* (:311-312), which was the real crop-dependent
        // consumer, because min(250,250) and min(256,256) are both below 480.
        // The rows that DO reach it are 474x480 and 714x720 in
        // `kb28_crop_dims::rd_band_min_dim_tiers_byte_match`; this one stays
        // here as the window control.
        (250, 250),
    ];
    let mut verdicts = Vec::new();
    for &(w, h) in SIZES {
        let cell = mirror_tile(&base, &format!("s1_{w}x{h}"), w, h, 24, 1);
        let tu = cell.c_encode();
        let real = EncodeCell::frame_obu_payload(&tu);
        let t = std::time::Instant::now();
        let port = cell.port_encode(&tu);
        let ms = t.elapsed().as_millis();
        let ok = port == real;
        let partial = w % 64 != 0 || h % 64 != 0;
        println!(
            "  speed1 {w}x{h} cq24 (min_dim {}, {}): port {} B vs C {} B delta {:+} -> {} [{ms} ms]",
            w.min(h),
            if partial { "PARTIAL-SB" } else { "SB-exact " },
            port.len(),
            real.len(),
            port.len() as i64 - real.len() as i64,
            if ok { "MATCH" } else { "DIVERGE" }
        );
        verdicts.push((w, h, ok));
    }
    let part_div = verdicts
        .iter()
        .filter(|(w, h, ok)| (w % 64 != 0 || h % 64 != 0) && !ok)
        .count();
    let part_n = verdicts
        .iter()
        .filter(|(w, h, _)| w % 64 != 0 || h % 64 != 0)
        .count();
    let exact_div = verdicts
        .iter()
        .filter(|(w, h, ok)| w % 64 == 0 && h % 64 == 0 && !ok)
        .count();
    let exact_n = verdicts
        .iter()
        .filter(|(w, h, _)| w % 64 == 0 && h % 64 == 0)
        .count();
    println!(
        "  speed1 size-axis: {}/{} byte-exact | PARTIAL-SB sizes {part_div}/{part_n} divergent, \
         SB-exact sizes {exact_div}/{exact_n} divergent",
        verdicts.iter().filter(|v| v.2).count(),
        verdicts.len()
    );
    // Phase 2 — the speed axis on the two cheapest partial-SB sizes and their
    // SB-exact neighbours. If the divergence were a frame-GEOMETRY bug it would
    // show at speed 0 too (where KB-6's partial-SB series is byte-exact); if it
    // is speed-gated it appears only from speed 1 up.
    let mut matrix = Vec::new();
    for &(w, h) in &[(132usize, 132usize), (192, 192), (196, 196), (256, 256)] {
        for speed in 0..=4 {
            let cell = mirror_tile(&base, &format!("s{speed}_{w}x{h}"), w, h, 24, speed);
            let tu = cell.c_encode();
            let real = EncodeCell::frame_obu_payload(&tu);
            let port = cell.port_encode(&tu);
            let ok = port == real;
            matrix.push((w, h, speed, ok, port.len() as i64 - real.len() as i64));
        }
    }
    for &(w, h, speed, ok, d) in &matrix {
        println!(
            "  speed{speed} {w}x{h} cq24 ({}): delta {d:+} -> {}",
            if w % 64 != 0 || h % 64 != 0 {
                "PARTIAL-SB"
            } else {
                "SB-exact "
            },
            if ok { "MATCH" } else { "DIVERGE" }
        );
    }
    let s0_partial_div = matrix
        .iter()
        .filter(|(w, h, s, ok, _)| *s == 0 && (w % 64 != 0 || h % 64 != 0) && !ok)
        .count();
    let sge1_partial_div = matrix
        .iter()
        .filter(|(w, h, s, ok, _)| *s >= 1 && (w % 64 != 0 || h % 64 != 0) && !ok)
        .count();
    let sge1_partial_n = matrix
        .iter()
        .filter(|(w, h, s, _, _)| *s >= 1 && (w % 64 != 0 || h % 64 != 0))
        .count();
    let sge1_exact_div = matrix
        .iter()
        .filter(|(w, h, s, ok, _)| *s >= 1 && w % 64 == 0 && h % 64 == 0 && !ok)
        .count();
    println!(
        "  KB-23 speed axis: speed0 partial-SB divergent {s0_partial_div}, \
         speed>=1 partial-SB divergent {sge1_partial_div}/{sge1_partial_n}, \
         speed>=1 SB-exact divergent {sge1_exact_div}"
    );

    // ---- The KB-23 regression gate. -----------------------------------------
    // Before the fix this read 5/5 partial-SB sizes divergent and 0/6 SB-exact;
    // after it, every size at speed 1 is byte-exact and the partial-SB sizes are
    // byte-exact at speeds 0..=3. Both halves are asserted, so a regression is
    // reported as one whichever way it lands.
    assert_eq!(
        part_div, 0,
        "PARTIAL-SB frames diverged at speed 1 — this is the KB-23 signature \
         (`cnn_output_valid`, partition_strategy.c:160/227 + partition_search.c:3340-3343): \
         the intra-CNN partition prune must not fire anywhere inside a 64x64 that is not \
         whole-in-frame. See `partition_pick.rs`'s `cnn_root_whole_in_frame`."
    );
    assert_eq!(
        exact_div, 0,
        "an SB-exact size diverged at speed 1 — that is NOT the KB-23 shape (which is \
         frame-EDGE partial superblocks only), so it needs its own localization"
    );
    assert_eq!(
        s0_partial_div, 0,
        "a partial-SB size diverged at SPEED 0 — that is a KB-6 partial-SB regression, \
         not KB-23 (which is speed >= 1 only)"
    );
    let sle3_div = matrix.iter().filter(|(_, _, s, ok, _)| *s <= 3 && !ok).count();
    assert_eq!(
        sle3_div, 0,
        "a speed 0..=3 cell diverged in the speed sweep — KB-23 covers the partial-SB \
         cells there and the SB-exact ones were already exact"
    );
    // Speed 4 is DELIBERATELY not asserted, even though it currently passes on
    // all four sizes (KB-21 root #3, `8a0faa7`, closed it — NOT this file's
    // fix: the speed-4 deltas are byte-identical before and after the KB-23
    // latch, which is exactly how we know KB-23 does not reach that band).
    // KB-21 root #2 is still open and owned elsewhere; pinning its band from
    // here would fire on that owner's work rather than on a real regression.
    // Speeds 0..=3 above carry this test's teeth.
    let _ = sge1_partial_div;
}
