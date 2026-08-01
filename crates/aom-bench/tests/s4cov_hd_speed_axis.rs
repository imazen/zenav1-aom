//! **S4 coverage extension — the RD speeds at >= 720p.**
//!
//! Nothing had run `--cpu-used >= 1` above 640x640 except KB-22's two 1280x720
//! **speed-0/1** cells and KB-19's 2160p speed-0 cell. That matters because the
//! framesize-dependent speed-feature pass has arms at EVERY speed, not just
//! zero, and the two whose thresholds sit inside this band are exactly the ones
//! a speed sweep at 64x64 / 128x128 can never see:
//!
//! * `set_allintra_speed_feature_framesize_dependent`'s
//!   `use_square_partition_only_threshold` tiers (`speed_features.c:175-316`):
//!   at speed 1 it is `BLOCK_128X128` for `is_720p_or_larger`, `BLOCK_64X64`
//!   for `is_480p_or_larger`, `BLOCK_32X32` below — so 1280x720 and 640x640 take
//!   DIFFERENT arms of the KB-3 rect-kill; at speed >= 2 it is `BLOCK_64X64`
//!   (720p+) vs `BLOCK_32X32`.
//! * `rd_sf.perform_coeff_opt = 2 + is_1080p_or_larger` — 720p resolves the
//!   value **2**, a `coeff_opt_thresholds` row (`:88-98`) that no sub-720p cell
//!   produces (they resolve 1) and the 2160p cell does not either (it resolves
//!   3). At speed >= 4 that row's SATD column is what KB-21 roots #2/#3 live
//!   in, so "720p x speed 4" crosses two independently-established rows.
//! * `av1_set_speed_features_qindex_dependent` (KB-22): the speed-0 arm needs
//!   `is_720p_or_larger`; the speed-1..3 arms are qindex-keyed too. Both cq
//!   points below straddle its `base_qindex <= 128` boundary.
//!
//! The grid is the two framesize tiers on either side of the 720p predicate
//! (640x640 and 1280x720), crossed with speeds and with a qindex on each side
//! of 128 — so a divergence that is really a framesize arm is separable from
//! one that is really a speed arm.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test s4cov_hd_speed_axis -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// Mirror-tile (same recipe as `kb22_hd_arms::mirror_tile`).
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

/// `av1_quantizer_to_qindex` is `4 * cq` over this range (`av1_quantize.c:1033`),
/// so cq24 -> 96 (<= 128, inside the KB-22 arm) and cq40 -> 160 (> 128, outside).
const CQS: [i32; 2] = [24, 40];

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Ok,
    Diverge,
    Panic,
}

/// Encode one cell and classify. A PANIC is a distinct outcome, not a test
/// abort — the port carries deliberate "refuse loudly" guards for configurations
/// it does not model (see `LARGE_FRAME_OPEN`), and aborting on the first one
/// would hide the rest of the map.
fn measure(cell: &EncodeCell) -> (Verdict, i64, String) {
    let c_tu = cell.c_encode();
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let msg = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = std::sync::Arc::clone(&msg);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let s = info.to_string();
        *sink.lock().unwrap() = s
            .lines()
            .find(|l| l.contains("assertion") || l.contains("not implemented"))
            .unwrap_or_else(|| s.lines().last().unwrap_or(""))
            .to_string();
    }));
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(&c_tu, &ToggleKnobs::default())
    }));
    std::panic::set_hook(hook);
    match got {
        Ok(p) if p == real => (Verdict::Ok, 0, String::new()),
        Ok(p) => (
            Verdict::Diverge,
            p.len() as i64 - real.len() as i64,
            String::new(),
        ),
        Err(_) => (Verdict::Panic, 0, msg.lock().unwrap().clone()),
    }
}

/// **The gate.** bd8 4:2:0 real content at 640x640 (below the 720p predicate)
/// and 1280x720 (on it), at `--cpu-used` 1..7, both quality points.
///
/// 640x640 is the largest frame any pre-existing gate encodes and it is
/// deliberately included: it is the negative control for the framesize
/// predicate. If a row diverges at 1280x720 and its 640x640 twin does not, the
/// difference is the `is_720p_or_larger` arm; if both diverge, it is not.
///
/// Speed 0 is covered by `kb22_hd_arms::qindex_arm_720p_isolation_e2e_byte_match`
/// and is skipped here because it costs ~35 s per cell — this file is the
/// speed >= 1 half that was missing.
///
/// **MEASURED 2026-08-01** (`benchmarks/s4cov_axes_2026-08-01.tsv`,
/// aarch64-apple-darwin, `--profile test-fast`):
///
/// | speed | 640x640 | 1280x720 |
/// |---|---|---|
/// | 1, 2, 3 | MATCH (both cq) | MATCH (both cq) |
/// | 4, 5, 6 | **DIVERGE (both cq)** | **DIVERGE** |
/// | 7 | DIVERGE at cq24, MATCH at cq40 | **PANIC** (see below) |
///
/// **The 640x640 twins diverge exactly as the 1280x720 rows do, so this is NOT
/// an `is_720p_or_larger` arm** — the control the grid exists to provide says
/// the framesize predicate is not what is moving. It is a LARGE-FRAME x
/// `--cpu-used >= 4` divergence, and `large_frame_speed4_size_ladder` below
/// bounds where it starts. All of it is pinned in `LARGE_FRAME_OPEN`, in both
/// directions.
///
/// The 1280x720 speed-7 PANIC is a different thing again and is a deliberate
/// port guard, not a crash: `pack.rs`'s VBP-threshold arm needs `cm->width *
/// cm->height` to pick `set_vbp_thresholds`' <720p bucket, `pack_tile` is not
/// given the true crop dims, and rather than guess it refuses whenever the
/// mi-aligned area and the "up to 3px smaller per axis" crop could land on
/// opposite sides of `1280*720`. An EXACTLY 1280x720 frame is inside that
/// window, so the first HD frame anyone would reach for at speed 7 refuses to
/// encode. Fix is to thread the true crop dims into `SbEncodeEnv` (it carries
/// only mi-aligned extents today — the same gap KB-23's 250x250 row names).
#[test]
#[ignore = "28 encode pairs up to 1280x720; nightly / on-demand tier"]
fn hd_speed_axis_byte_matches() {
    c::ref_init();
    let base = EncodeCell::real_content("s4hd", "av1-1-b8-00-quantizer-00", None, 24, 0);
    const SIZES: &[(usize, usize)] = &[(640, 640), (1280, 720)];
    // (w, h, cq, speed, verdict) — the exact current state, pinned.
    const LARGE_FRAME_OPEN: &[(usize, usize, i32, i32, Verdict)] = &[
        (640, 640, 24, 4, Verdict::Diverge),
        (640, 640, 24, 5, Verdict::Diverge),
        (640, 640, 24, 6, Verdict::Diverge),
        (640, 640, 24, 7, Verdict::Diverge),
        (640, 640, 40, 4, Verdict::Diverge),
        (640, 640, 40, 5, Verdict::Diverge),
        (640, 640, 40, 6, Verdict::Diverge),
        (1280, 720, 24, 4, Verdict::Diverge),
        (1280, 720, 24, 5, Verdict::Diverge),
        (1280, 720, 24, 6, Verdict::Diverge),
        (1280, 720, 24, 7, Verdict::Panic),
        (1280, 720, 40, 4, Verdict::Diverge),
        (1280, 720, 40, 5, Verdict::Diverge),
        (1280, 720, 40, 6, Verdict::Diverge),
        (1280, 720, 40, 7, Verdict::Panic),
    ];
    let mut observed: Vec<(usize, usize, i32, i32, Verdict)> = Vec::new();
    let mut rows = 0usize;
    for &(w, h) in SIZES {
        // Reach assertion, per size (playbook §8): the point of the pair is
        // which side of `is_720p_or_larger` it sits on.
        let is_720p = w.min(h) >= 720;
        assert_eq!(
            is_720p,
            (w, h) == (1280, 720),
            "the 640x640 cell must sit BELOW is_720p_or_larger and the 1280x720 cell ON it"
        );
        for &cq in &CQS {
            for speed in 1..=7 {
                let cell = mirror_tile(&base, &format!("hd_{w}x{h}_cq{cq}_s{speed}"), w, h, cq, speed);
                let t0 = std::time::Instant::now();
                let (v, delta, note) = measure(&cell);
                let ms = t0.elapsed().as_millis();
                rows += 1;
                println!(
                    "  {w}x{h} ({}) cq{cq} cpu{speed}: delta {delta:+} -> {:?} [{ms} ms]{}",
                    if is_720p { ">=720p" } else { "<720p " },
                    v,
                    if note.is_empty() {
                        String::new()
                    } else {
                        format!("  [{note}]")
                    }
                );
                if v != Verdict::Ok {
                    observed.push((w, h, cq, speed, v));
                }
            }
        }
    }
    println!(
        "  s4cov HD speed axis: {}/{rows} byte-exact",
        rows - observed.len()
    );

    // Speeds 1..3 must be clean at BOTH framesizes and BOTH qindex sides —
    // that is this grid's positive result and the teeth on the framesize arms
    // (`use_square_partition_only_threshold`'s 720p/480p tiers at speeds 1-2,
    // `perform_coeff_opt = 2 + is_1080p_or_larger` at 720p) it reaches first.
    let low_speed_bad: Vec<String> = observed
        .iter()
        .filter(|(_, _, _, s, _)| *s <= 3)
        .map(|(w, h, cq, s, v)| format!("{w}x{h} cq{cq} cpu{s} {v:?}"))
        .collect();
    assert!(
        low_speed_bad.is_empty(),
        "an HD cell stopped byte-matching at speed 1..3. Compare each 1280x720 row against \
         its 640x640 twin at the same (cq, speed): only-720p means an `is_720p_or_larger` \
         framesize arm (speed_features.c:175-316 / :88-98); both means it is not a \
         framesize arm: {low_speed_bad:?}"
    );

    let pinned: Vec<(usize, usize, i32, i32, Verdict)> = LARGE_FRAME_OPEN.to_vec();
    assert_eq!(
        observed, pinned,
        "the large-frame speed>=4 map moved. A row that started MATCHING means the \
         large-frame x cpu-4..7 divergence closed (re-pin here, and re-run \
         `large_frame_speed4_size_ladder` — its threshold will have moved too). A row that \
         started DIVERGING or PANICKING is a regression."
    );
}

/// **Where does the large-frame `--cpu-used >= 4` divergence start?**
///
/// The speed axis of the config-permutation gate runs 64x64 and 128x128
/// (`benchmarks/config_perm_speed_axis_2026-07-30.tsv`) and KB-23's grid tops
/// out at 256x256 — all byte-exact at speed 4 after KB-21 root #3. The gate
/// above shows 640x640 is not. This walks SB-exact sizes between them at
/// `--cpu-used=4`, cq24, so whoever localizes it starts from the smallest
/// reproducing frame instead of a 640x640 one.
///
/// SB-exact sizes only: mixing in partial-SB frames would confound the
/// frame-edge axis with the size axis, which is the mistake KB-23 was found by
/// avoiding.
///
/// **MEASURED 2026-08-01**: see the printed ladder. The smallest reproducing
/// size is asserted below so a change in it is reported rather than absorbed.
#[test]
#[ignore = "7 encode pairs up to 640x640 at speed 4; diagnostic, run explicitly"]
fn large_frame_speed4_size_ladder() {
    c::ref_init();
    let base = EncodeCell::real_content("s4ladder", "av1-1-b8-00-quantizer-00", None, 24, 0);
    const SIZES: &[usize] = &[256, 320, 384, 448, 512, 576, 640];
    let mut first_bad: Option<usize> = None;
    for &n in SIZES {
        assert_eq!(n % 64, 0, "SB-exact sizes only — see the doc comment");
        let cell = mirror_tile(&base, &format!("ladder{n}"), n, n, 24, 4);
        let (v, delta, _) = measure(&cell);
        println!("  {n}x{n} cq24 cpu4: delta {delta:+} -> {v:?}");
        if v != Verdict::Ok && first_bad.is_none() {
            first_bad = Some(n);
        }
    }
    println!("  large-frame cpu4 ladder: smallest reproducing SB-exact size {first_bad:?}");
    assert!(
        first_bad.is_some(),
        "no size on the ladder reproduces the large-frame cpu4 divergence any more — it \
         closed; re-pin `LARGE_FRAME_OPEN` in `hd_speed_axis_byte_matches` and delete this \
         ladder"
    );
}

/// **Handoff diagnostic for the large-frame `cpu >= 4` divergence: the ONE
/// `is_480p_or_larger` x `speed >= 4` arm C has, A/B'd.**
///
/// `set_allintra_speed_feature_framesize_dependent` (`speed_features.c:166-320`)
/// contains exactly one setting keyed on BOTH `is_480p_or_larger` and
/// `speed >= 4`: `tx_sf.tx_type_search.prune_tx_type_using_stats = 2`
/// (`:299-301`; the same predicate sets it to **1** at `speed >= 2`, `:261-263`).
/// Every other 480p-keyed arm in that function is either speed-independent or
/// `is_720p_or_larger`-keyed, and 640x640 sits below 720p — so this is where a
/// framesize-keyed explanation has to live.
///
/// `ToggleKnobs::disable_tx_stats_prune` forces the port's derived value to 0
/// while leaving the C reference untouched, which turns the prune into a
/// falsifiable A/B. **MEASURED 2026-08-01, and the result is an anomaly worth
/// handing off rather than a fix:**
///
/// | cell | prune on | prune forced off |
/// |---|---|---|
/// | 448x448 (sub-480p) cpu2..5 | MATCH | MATCH — inert, correctly |
/// | 512x512 cpu2 | MATCH | DIVERGE -100 — **load-bearing** |
/// | 512x512 cpu3 | MATCH | DIVERGE -61 — **load-bearing** |
/// | 512x512 cpu4 | DIVERGE -28 | DIVERGE -28 — **byte-identical, i.e. INERT** |
/// | 512x512 cpu5 | DIVERGE +106 | DIVERGE +106 — **byte-identical, INERT** |
///
/// So the port's stats prune demonstrably changes the bitstream at speeds 2-3
/// and demonstrably does not at speeds 4-5, on the same frame, where C's only
/// change across that boundary is the prune's own level going 1 -> 2. On a lone
/// KEY frame `update_type == KF_UPDATE == 0` and `thresh_arr[0][0] ==
/// thresh_arr[1][0] == 10` (`tx_search.c:1887-1891`), so the level change is
/// expected to be a NO-OP — which makes "the prune stopped mattering at all"
/// the thing to explain, not the level. The next step is the per-txb dump of
/// KB-21 root #2 (playbook §10) around `get_tx_mask_intra`'s multi-type arm at
/// speed 4, checking whether it is reached at all: at speed >= 4 the MODE_EVAL
/// stage takes the single-type `use_default_intra_tx_type` arm
/// (`tx_search.c:1871`), which never reaches the stats prune, so the question
/// is whether the WINNER_MODE_EVAL pass is reaching it.
#[test]
#[ignore = "16 encode pairs at 448/512; diagnostic, run explicitly"]
fn tx_stats_prune_ab_across_the_480p_boundary() {
    c::ref_init();
    let base = EncodeCell::real_content("s4ab", "av1-1-b8-00-quantizer-00", None, 24, 0);
    let mut inert_at_4_5 = 0usize;
    let mut load_bearing_at_2_3 = 0usize;
    for n in [448usize, 512] {
        for speed in [2i32, 3, 4, 5] {
            let cell = mirror_tile(&base, &format!("ab{n}_{speed}"), n, n, 24, speed);
            let c_tu = cell.c_encode();
            let real = EncodeCell::frame_obu_payload(&c_tu);
            let on = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
            let off = cell.port_encode_with(
                &c_tu,
                &ToggleKnobs {
                    disable_tx_stats_prune: true,
                    ..Default::default()
                },
            );
            let bites = on != off;
            println!(
                "  {n}x{n} cq24 cpu{speed}: prune-on {} ({:+}) | prune-off {} ({:+}) | \
                 prune {} the bitstream",
                if on == real { "MATCH  " } else { "DIVERGE" },
                on.len() as i64 - real.len() as i64,
                if off == real { "MATCH  " } else { "DIVERGE" },
                off.len() as i64 - real.len() as i64,
                if bites { "CHANGES" } else { "does NOT change" }
            );
            if n == 512 && speed <= 3 && bites {
                load_bearing_at_2_3 += 1;
            }
            if n == 512 && speed >= 4 && !bites {
                inert_at_4_5 += 1;
            }
            if n == 448 {
                assert!(
                    !bites,
                    "the stats prune changed a SUB-480p bitstream — it must be derived as 0 \
                     below `is_480p_or_larger` (speed_features.c:261/299)"
                );
            }
        }
    }
    println!(
        "  tx-stats A/B: 512x512 load-bearing at {load_bearing_at_2_3}/2 of speeds 2-3, \
         INERT at {inert_at_4_5}/2 of speeds 4-5"
    );
    assert_eq!(
        load_bearing_at_2_3, 2,
        "the stats prune stopped changing the >=480p bitstream at speeds 2-3, so this A/B \
         no longer isolates anything — re-derive it before reading the speed 4-5 row"
    );
    assert_eq!(
        inert_at_4_5, 2,
        "the stats prune now CHANGES the >=480p bitstream at speeds 4-5. That is the \
         anomaly this diagnostic recorded closing — re-read `LARGE_FRAME_OPEN` in \
         `hd_speed_axis_byte_matches`, which may have moved with it"
    );
}
