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
/// aarch64-apple-darwin, `--profile test-fast`) — the ORIGINAL map, kept so a
/// regression is recognisable by shape:
///
/// | speed | 640x640 | 1280x720 |
/// |---|---|---|
/// | 1, 2, 3 | MATCH (both cq) | MATCH (both cq) |
/// | 4, 5, 6 | DIVERGE (both cq) | DIVERGE |
/// | 7 | DIVERGE at cq24, MATCH at cq40 | PANIC (see below) |
///
/// The 640x640 twins diverged exactly as the 1280x720 rows did, so it was never
/// an `is_720p_or_larger` arm — it was a LARGE-FRAME x `--cpu-used >= 4`
/// divergence (KB-26).
///
/// **RE-MEASURED 2026-08-01 after the KB-26 fix: 26/28 byte-exact.** All 13
/// divergent rows closed. Root: the speed>=4 winner-mode two-pass derived its
/// MODE_EVAL / WINNER_MODE_EVAL tx policies from a fresh, FRAMESIZE-BLIND
/// `SpeedFeatures::set_allintra`, dropping the framesize-derived
/// `tx_sf.tx_type_search.prune_tx_type_using_stats` (`is_480p_or_larger`,
/// `speed_features.c:261/299`) for the entire luma tx search on every >=480p
/// frame — while speeds 0..3, which use the caller's resolved policy directly,
/// kept it. Fixed by `TxTypeSearchPolicy::carry_frame_level_tx_sf`
/// (`tx_search.rs`), called from `partition_pick`'s `wm_parts`.
/// `LARGE_FRAME_OPEN` then held ONLY the two KB-28 speed-7 panics.
///
/// **RE-MEASURED 2026-08-02 after the KB-28 fix: 28/28 byte-exact, and
/// `LARGE_FRAME_OPEN` is now EMPTY** — this pin fired unprompted, which is
/// what it was for. The two speed-7 rows were never a crash: `pack.rs`'s
/// VBP-threshold arm needs `cm->width * cm->height` to pick
/// `set_vbp_thresholds`' <720p bucket (var_based_part.c:667 -> :547),
/// `pack_tile` was given only mi-ALIGNED extents, and rather than guess it
/// refused across the window where the two could straddle `1280*720`. The fix
/// threads the true crop dims through `SbEncodeEnv::frame_{width,height}` to
/// all six framesize consumers; the refusal window, its `mi - 3`-vs-`mi - 7`
/// error and the 8,776 crops it silently missed are mapped in
/// `kb28_crop_dims::refusal_window_is_characterised`.
#[test]
#[ignore = "28 encode pairs up to 1280x720; nightly / on-demand tier"]
fn hd_speed_axis_byte_matches() {
    c::ref_init();
    let base = EncodeCell::real_content("s4hd", "av1-1-b8-00-quantizer-00", None, 24, 0);
    const SIZES: &[(usize, usize)] = &[(640, 640), (1280, 720)];
    // (w, h, cq, speed, verdict) — the exact current state, pinned.
    // KB-26 closed every DIVERGE row on 2026-08-01, leaving only KB-28's two
    // speed-7 crop-ambiguity refusals; **KB-28 closed on 2026-08-02 and this
    // pin fired**, so the list is now EMPTY: 28/28 byte-exact. Anything
    // appearing here again is a regression, not a known-open cell.
    const LARGE_FRAME_OPEN: &[(usize, usize, i32, i32, Verdict)] = &[];
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

    // Speeds 1..6 must be clean at BOTH framesizes and BOTH qindex sides — that
    // is this grid's positive result and the teeth on the framesize arms
    // (`use_square_partition_only_threshold`'s 720p/480p tiers at speeds 1-2,
    // `perform_coeff_opt = 2 + is_1080p_or_larger` at 720p) it reaches first.
    // Speeds 1..3 were clean from the start; 4..6 were the KB-26 divergence and
    // became clean when the framesize-derived `prune_tx_type_using_stats`
    // stopped being dropped by the winner-mode stage derivation. Speed 7 was
    // excluded until 2026-08-02 because of KB-28's 1280x720 refusal; it is now
    // inside the range, which is why this filter reads `<= 7`.
    let low_speed_bad: Vec<String> = observed
        .iter()
        .filter(|(_, _, _, s, _)| *s <= 7)
        .map(|(w, h, cq, s, v)| format!("{w}x{h} cq{cq} cpu{s} {v:?}"))
        .collect();
    assert!(
        low_speed_bad.is_empty(),
        "an HD cell stopped byte-matching at speed 1..6. If the row is at speed >= 4 and \
         >= 480p on its short side, suspect KB-26 first: the winner-mode two-pass in \
         `partition_pick` re-derives its stage tx policies from a FRAMESIZE-BLIND \
         `SpeedFeatures::set_allintra`, so anything framesize-derived must be carried \
         across by `TxTypeSearchPolicy::carry_frame_level_tx_sf`. Otherwise compare each \
         1280x720 row against its 640x640 twin at the same (cq, speed): only-720p means \
         an `is_720p_or_larger` framesize arm (speed_features.c:175-316 / :88-98); both \
         means it is not a framesize arm: {low_speed_bad:?}"
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

/// **The size ladder across `is_480p_or_larger`, at `--cpu-used=4` — now a
/// positive byte-match gate.**
///
/// The speed axis of the config-permutation gate runs 64x64 and 128x128
/// (`benchmarks/config_perm_speed_axis_2026-07-30.tsv`) and KB-23's grid tops
/// out at 256x256; this walks SB-exact sizes from there to 640x640 at
/// `--cpu-used=4`, cq24. It was written as a bisector — "where does the
/// large-frame cpu4 divergence start?" — and it did its job: **448 MATCH,
/// 512 DIVERGE (-28 B)** put the boundary exactly on `is_480p_or_larger`
/// (`speed_features.c:169`), which is what named KB-26's root.
///
/// SB-exact sizes only: mixing in partial-SB frames would confound the
/// frame-edge axis with the size axis, which is the mistake KB-23 was found by
/// avoiding.
///
/// **MEASURED 2026-08-01, after the KB-26 fix: 7/7 byte-exact** (256, 320, 384,
/// 448, 512, 576, 640 — all `+0`). It is kept and PROMOTED to a hard gate
/// rather than deleted, because it is the only thing in the suite that crosses
/// the 480p predicate at a speed where the winner-mode two-pass runs: the
/// 448/512 pair is exactly the A/B that would go divergent again if a
/// framesize-derived speed feature were once more dropped by
/// `partition_pick`'s stage-policy derivation.
#[test]
#[ignore = "7 encode pairs up to 640x640 at speed 4; nightly / on-demand tier"]
fn large_frame_speed4_size_ladder() {
    c::ref_init();
    let base = EncodeCell::real_content("s4ladder", "av1-1-b8-00-quantizer-00", None, 24, 0);
    const SIZES: &[usize] = &[256, 320, 384, 448, 512, 576, 640];
    // Reach assertion (playbook §2): the ladder is worthless unless it actually
    // straddles `is_480p_or_larger` — 448 below it, 512 above.
    assert!(
        SIZES.iter().any(|&n| n < 480) && SIZES.iter().any(|&n| n >= 480),
        "the ladder must cross AOMMIN(w,h) >= 480 to gate the KB-26 root"
    );
    let mut bad: Vec<String> = Vec::new();
    for &n in SIZES {
        assert_eq!(n % 64, 0, "SB-exact sizes only — see the doc comment");
        let cell = mirror_tile(&base, &format!("ladder{n}"), n, n, 24, 4);
        let (v, delta, _) = measure(&cell);
        println!(
            "  {n}x{n} ({}) cq24 cpu4: delta {delta:+} -> {v:?}",
            if n >= 480 { ">=480p" } else { "<480p " }
        );
        if v != Verdict::Ok {
            bad.push(format!("{n}x{n} {v:?} ({delta:+})"));
        }
    }
    assert!(
        bad.is_empty(),
        "the cpu4 SB-exact size ladder stopped byte-matching. If the failing sizes are \
         >= 480 and the sub-480 ones are clean, this is KB-26 regressing: something \
         framesize-derived is being dropped when `partition_pick`'s winner-mode two-pass \
         re-derives its stage tx policies from a framesize-blind \
         `SpeedFeatures::set_allintra` (see `TxTypeSearchPolicy::carry_frame_level_tx_sf`). \
         Offenders: {bad:?}"
    );
}

/// **The KB-26 root, A/B'd: the ONE `is_480p_or_larger` x `speed >= 4` arm C
/// has, asserted LIVE on both sides of the winner-mode boundary.**
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
/// falsifiable A/B.
///
/// **MEASURED 2026-08-01 (PRE-FIX) — this table is what localized KB-26:**
///
/// | cell | prune on | prune forced off |
/// |---|---|---|
/// | 448x448 (sub-480p) cpu2..5 | MATCH | MATCH — inert, correctly |
/// | 512x512 cpu2 | MATCH | DIVERGE -100 — load-bearing |
/// | 512x512 cpu3 | MATCH | DIVERGE -61 — load-bearing |
/// | 512x512 cpu4 | DIVERGE -28 | DIVERGE -28 — byte-identical, i.e. INERT |
/// | 512x512 cpu5 | DIVERGE +106 | DIVERGE +106 — byte-identical, INERT |
///
/// The prune demonstrably changed the bitstream at speeds 2-3 and demonstrably
/// did not at speeds 4-5, on the same frame — while C's only change across that
/// boundary is the prune's own level going 1 -> 2, which on a lone KEY frame is
/// a NO-OP (`update_type == KF_UPDATE == 0`, `thresh_arr[0][0] ==
/// thresh_arr[1][0] == 10`, `tx_search.c:1887-1891`). So "the prune stopped
/// mattering AT ALL" was the thing to explain, not the level.
///
/// **ROOT (2026-08-01, closed).** An instrumented reach count over
/// `get_tx_mask_intra` on the 512x512 cpu4 cell answered it directly: the
/// multi-type arm IS reached (140,154 times), but the stats-prune body inside
/// it ran **0 times** — the sf arrived as 0. At speed >= 4 `partition_pick`'s
/// winner-mode two-pass builds its MODE_EVAL / WINNER_MODE_EVAL policies from a
/// FRESH `SpeedFeatures::set_allintra`, which is framesize-blind, so the
/// framesize-derived `prune_tx_type_using_stats` was silently 0 for the whole
/// luma tx search on every >=480p frame. Speeds 0..3 use the caller's resolved
/// policy directly and so were unaffected — precisely the observed split.
/// Fixed by `TxTypeSearchPolicy::carry_frame_level_tx_sf`.
///
/// **RE-MEASURED after the fix: the prune is load-bearing at 4/4 of speeds
/// 2,3,4,5 on 512x512 and still inert at 448x448**, and every prune-on cell
/// byte-matches real aomenc. This test is now that gate: `inert_at_4_5`
/// returning to nonzero means the framesize-derived sf is being dropped again.
#[test]
#[ignore = "16 encode pairs at 448/512; nightly / on-demand tier"]
fn tx_stats_prune_ab_across_the_480p_boundary() {
    c::ref_init();
    let base = EncodeCell::real_content("s4ab", "av1-1-b8-00-quantizer-00", None, 24, 0);
    let mut inert_at_4_5 = 0usize;
    let mut load_bearing_at_2_3 = 0usize;
    let mut load_bearing_at_4_5 = 0usize;
    let mut on_mismatch: Vec<String> = Vec::new();
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
            if on != real {
                on_mismatch.push(format!("{n}x{n} cpu{speed}"));
            }
            if n == 512 && speed <= 3 && bites {
                load_bearing_at_2_3 += 1;
            }
            if n == 512 && speed >= 4 && !bites {
                inert_at_4_5 += 1;
            }
            if n == 512 && speed >= 4 && bites {
                load_bearing_at_4_5 += 1;
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
        "  tx-stats A/B: 512x512 load-bearing at {load_bearing_at_2_3}/2 of speeds 2-3 and \
         {load_bearing_at_4_5}/2 of speeds 4-5 (INERT at {inert_at_4_5}/2)"
    );
    assert_eq!(
        load_bearing_at_2_3, 2,
        "the stats prune stopped changing the >=480p bitstream at speeds 2-3, so this A/B \
         no longer isolates anything — re-derive it before reading the speed 4-5 row"
    );
    // The KB-26 teeth. Before the fix this was 0/2 and `inert_at_4_5` was 2/2:
    // the winner-mode stage derivation dropped the framesize-resolved sf, so the
    // knob that is supposed to control the prune controlled nothing at speed >= 4.
    assert_eq!(
        (load_bearing_at_4_5, inert_at_4_5),
        (2, 0),
        "KB-26 REGRESSED: the tx-type stats prune has gone inert on a >=480p frame at \
         speed >= 4, which means `partition_pick`'s winner-mode two-pass is again \
         deriving its MODE_EVAL / WINNER_MODE_EVAL policies from a framesize-BLIND \
         `SpeedFeatures::set_allintra` instead of carrying the frame-resolved \
         `prune_tx_type_using_stats` across (`TxTypeSearchPolicy::carry_frame_level_tx_sf`, \
         speed_features.c:261/299)"
    );
    assert!(
        on_mismatch.is_empty(),
        "a stats-prune A/B cell stopped byte-matching real aomenc with the prune ON: \
         {on_mismatch:?}"
    );
}
