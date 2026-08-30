//! **KB-37 — `av1_search_palette_mode_luma` is PORTED for the nonrd estimate
//! arm.**
//!
//! KB-35 narrowed the estimate arm's palette refusal to C's own `try_palette`
//! (nonrd_pickmode.c:1698-1710) and left the refusal itself standing: the
//! SEARCH — `av1_search_palette_mode_luma` (intra_mode_search.c:1122) — was
//! unported, so any leaf where C's predicate holds was a hard stop rather than
//! a wrong encode. This file is the byte gate for the port of that search.
//!
//! # The isolation problem this file solves, and how
//!
//! At `--cpu-used 8` the palette search is reachable from **two** arms:
//! `hybrid_intra_mode_search` (partition_search.c:756) sends
//! `bsize < BLOCK_16X16 && source_variance >= 101` to the full-RD
//! `av1_rd_pick_intra_mode_sb` (whose own palette search is
//! `av1_rd_pick_palette_intra_sby`) and everything else to the estimate arm.
//! The speed-8 crossing of the FULL-RD palette search is an open divergence
//! class (`PALETTE_ON_SPEED8_OPEN`, pinned in `kb35_nonrd_palette_arm.rs`), so
//! a speed-8 palette cell cannot attribute a byte delta to either arm.
//!
//! **`--cpu-used 9` removes the ambiguity structurally.** At speed 9
//! `hybrid_intra_pickmode = 0` (speed_features.c:1795), so `hybrid_use_rdopt`
//! is false for EVERY leaf and the full-RD arm — hence the full-RD palette
//! search — is unreachable. What normally makes speed 9 useless for palette is
//! that `av1_set_screen_content_options` (encoder.c:2466-2470) turns
//! screen-content DETECTION off when `use_nonrd_pick_mode &&
//! !hybrid_intra_pickmode`. But that arm is reached only after the
//! `tune_cfg.content == AOM_CONTENT_SCREEN` arm above it (encoder.c:2448-2454),
//! which sets `allow_screen_content_tools = 1` and returns. So
//! `--tune-content=screen --cpu-used=9` gives a frame where the screen flag is
//! on, palette is enabled, and **every** leaf is the estimate arm.
//!
//! Every row asserts that isolation from the instruments rather than assuming
//! it: `nonrd_leaf_arms()[0]` must be **0** (no full-RD leaf ran at all).
//!
//! Speed 9 also turns on `prune_intra_mode_using_best_sad_so_far`
//! (speed_features.c:1799), which is what makes `args.prune_mode_based_on_sad`
//! — the fourth term of `try_palette`, and the one that is dead at speed 8 —
//! live here.
//!
//! # Non-vacuity
//!
//! Three independent conditions, all asserted:
//!
//! * `palette_gate_reach()[2] > 0` — leaves satisfying C's predicate exist;
//! * `nonrd_palette_search_stats()[1] > 0` — the search returned a WINNER, so
//!   the winner writeback (mode/tx_size/tx_type_map/palette) is exercised and
//!   not just the "no palette" exit;
//! * on the many-colour family, the C reference with `--enable-palette=1`
//!   DIFFERS from the same encode with palette off, so the palette search is
//!   changing the reference bytes this gate matches.
//!
//! `[2]` of the search stats additionally records how many searches ran with
//! `color_palette_thresh == 32` (`best_sad_norm < 500`, nonrd_pickmode.c:1713)
//! versus 64 — the per-leaf field that is NOT the per-superblock 64
//! `encode_sb_row` resets to.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb37_nonrd_palette_search -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_encode::nonrd_pickmode::{
    nonrd_palette_search_stats, palette_gate_reach, reset_nonrd_palette_search_stats,
    reset_palette_gate_reach,
};
use aom_encode::partition_pick::{nonrd_leaf_arms, reset_nonrd_leaf_arms};
use aom_sys_ref as c;

/// `AV1E_SET_TUNE_CONTENT` (aomcx.h `aome_enc_control_id`, verified value 43)
/// with `AOM_CONTENT_SCREEN = 1` (`aom_tune_content`, aomcx.h).
const AV1E_SET_TUNE_CONTENT: i32 = 43;
const AOM_CONTENT_SCREEN: i32 = 1;

/// Few-colour "terminal text" luma — the `rd_close_palette` /
/// `kb35_nonrd_palette_arm` recipe, verbatim.
fn text_luma(r: usize, c: usize) -> u16 {
    let row_in_line = r % 10;
    if row_in_line >= 7 {
        return 235;
    }
    let glyph = (c / 8 + (r / 10) * 5) % 4;
    let col_in_glyph = c % 8;
    match glyph {
        0 => {
            if col_in_glyph < 5 && row_in_line % 2 == 0 {
                32
            } else {
                235
            }
        }
        1 => {
            if col_in_glyph % 3 == 0 || row_in_line == 3 {
                32
            } else {
                235
            }
        }
        2 => {
            if col_in_glyph < 2 || col_in_glyph >= 6 {
                96
            } else {
                235
            }
        }
        _ => 235,
    }
}

/// Flat few-colour chroma panels (period-16 bands, zero gradients).
fn ui_chroma(r: usize, c: usize) -> u16 {
    match (c / 16 + r / 24) % 3 {
        0 => 84,
        1 => 128,
        _ => 170,
    }
}

fn cell_from(label: &str, w: usize, h: usize, cq: i32, speed: i32, luma: impl Fn(usize, usize) -> u16) -> EncodeCell {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for cc in 0..w {
            y[r * w + cc] = luma(r, cc);
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for cc in 0..cw {
            u[r * cw + cc] = ui_chroma(r, cc);
            v[r * cw + cc] = ui_chroma(r + 5, cc + 7);
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2, // ALLINTRA
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

fn screen_cell(label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    cell_from(label, w, h, cq, speed, text_luma)
}

/// `ncol` distinct luma levels in a 4x4-cell tiling, to straddle
/// `color_palette_thresh` (32 / 64) and `PALETTE_MAX_SIZE`. Levels stay inside
/// `[16, 16 + 4*(ncol-1)]`, which is <= 255 for every `ncol` used here.
fn many_color_cell(label: &str, w: usize, h: usize, cq: i32, speed: i32, ncol: u16) -> EncodeCell {
    assert!(16 + 4 * (ncol - 1) < 256, "many_color_cell would clip");
    cell_from(label, w, h, cq, speed, move |r, cc| {
        16 + ((cc / 4 + (r / 4) * 7) as u16 % ncol) * 4
    })
}

/// `ncol` distinct luma levels on a **2x2** cell grid, so a single 16x16 leaf
/// sees up to 64 distinct colours — the range where `color_palette_thresh`
/// (32 vs 64, nonrd_pickmode.c:1713) decides whether
/// `av1_rd_pick_palette_intra_sby` runs its search at all
/// (`colors_threshold > 1 && colors_threshold <= color_thresh_palette`,
/// palette.c:592). The coarse `many_color_cell` above cannot reach that range:
/// its 4x4 cells give at most 16 distinct values per 16x16 leaf.
fn fine_color_cell(label: &str, w: usize, h: usize, cq: i32, speed: i32, ncol: u16) -> EncodeCell {
    assert!(16 + 3 * (ncol - 1) < 256, "fine_color_cell would clip");
    cell_from(label, w, h, cq, speed, move |r, cc| {
        16 + ((cc / 2 + (r / 2) * 8) as u16 % ncol) * 3
    })
}

/// The `--tune-content=screen --enable-palette=1` reference. `c_encode_ctrls`
/// applies extra pairs AFTER the base control set (dec_shim.c:419-424), so
/// `AV1E_SET_ENABLE_PALETTE` here overrides the base `enable_palette = 0`
/// `shim_encode_av1_kf_ctrls` passes — a combination
/// `shim_encode_av1_kf_screen_content` cannot express (it has no
/// tune-content argument).
fn c_ref(cell: &EncodeCell) -> Vec<u8> {
    cell.c_encode_ctrls(&[
        (AV1E_SET_TUNE_CONTENT, AOM_CONTENT_SCREEN),
        (c::cx_ctrl::AV1E_SET_ENABLE_PALETTE, 1),
    ])
}

/// The same encode with palette OFF — the control that proves the palette
/// search moves the reference bytes.
fn c_ref_palette_off(cell: &EncodeCell) -> Vec<u8> {
    cell.c_encode_ctrls(&[(AV1E_SET_TUNE_CONTENT, AOM_CONTENT_SCREEN)])
}

struct Row {
    label: String,
    matched: bool,
    delta: i64,
    reach: [u64; 3],
    arms: [u64; 2],
    stats: [u64; 3],
}

fn measure(label: &str, cell: &EncodeCell, c_tu: &[u8]) -> Row {
    let real = EncodeCell::frame_obu_payload(c_tu);
    let knobs = ToggleKnobs {
        enable_palette: true,
        // The reference is `--tune-content=screen` (see `c_ref`):
        // the port's screen-content decision must take that arm, not the detector.
        tune_content_screen: true,
        ..Default::default()
    };
    reset_palette_gate_reach();
    reset_nonrd_leaf_arms();
    reset_nonrd_palette_search_stats();
    let got = cell.port_encode_with(c_tu, &knobs);
    Row {
        label: label.to_string(),
        matched: got == real,
        delta: got.len() as i64 - real.len() as i64,
        reach: palette_gate_reach(),
        arms: nonrd_leaf_arms(),
        stats: nonrd_palette_search_stats(),
    }
}

/// **The KB-37 byte gate.** 27 cells, every one with the full-RD arm provably
/// unreachable, must be byte-identical to real aomenc.
///
/// On the pre-port tree every cell here PANICs with *"HANDOFF:
/// av1_search_palette_mode_luma (intra_mode_search.c:1122) is not ported"* —
/// that is the teeth, and it is why the assertions below are about the
/// instruments (isolation, reach, winners) and not merely about equality.
#[test]
#[ignore = "27 encode pairs + 12 palette-off controls; nightly / on-demand tier"]
fn nonrd_estimate_arm_palette_search_is_byte_identical() {
    c::ref_init();
    let mut rows: Vec<Row> = Vec::new();

    // (a) few-colour text screen content, three sizes x five quantizers.
    for &(w, h) in &[(128usize, 128usize), (256, 256), (512, 512)] {
        for cq in [12, 32, 50, 60, 63] {
            let cell = screen_cell(&format!("scr{w}"), w, h, cq, 9);
            let c_tu = c_ref(&cell);
            assert!(!c_tu.is_empty(), "{w}x{h} cq{cq}: C encode failed");
            rows.push(measure(&format!("scr {w}x{h} cq{cq}"), &cell, &c_tu));
        }
    }

    // (b) many-colour content: `av1_count_colors` lands near
    // `color_palette_thresh`, and each cell carries its palette-OFF control.
    let mut palette_moved_bytes = 0usize;
    for ncol in [20u16, 34, 48, 60] {
        for cq in [12, 40, 63] {
            let cell = many_color_cell("mc256", 256, 256, cq, 9, ncol);
            let c_tu = c_ref(&cell);
            assert!(!c_tu.is_empty(), "mc n{ncol} cq{cq}: C encode failed");
            if c_ref_palette_off(&cell) != c_tu {
                palette_moved_bytes += 1;
            }
            rows.push(measure(&format!("mc256 n{ncol} cq{cq}"), &cell, &c_tu));
        }
    }

    for r in &rows {
        println!(
            "  {}: {} {:+} reach {:?} arms {:?} stats {:?}",
            r.label,
            if r.matched { "MATCH" } else { "DIVERGE" },
            r.delta,
            r.reach,
            r.arms,
            r.stats
        );
    }

    // (1) ISOLATION — asserted, not assumed. A single full-RD leaf would make
    // every delta below un-attributable (that is exactly the state
    // `PALETTE_ON_SPEED8_OPEN` leaves the speed-8 cells in).
    let leaky: Vec<&str> = rows
        .iter()
        .filter(|r| r.arms[0] != 0)
        .map(|r| r.label.as_str())
        .collect();
    assert!(
        leaky.is_empty(),
        "these cells reached the FULL-RD leaf arm, so a divergence could belong \
         to `av1_rd_pick_palette_intra_sby` instead of the estimate arm — the \
         gate is no longer isolating: {leaky:?}"
    );

    // (2) the estimate arm's palette gate fired, and the SEARCH ran.
    let searched: u64 = rows.iter().map(|r| r.stats[0]).sum();
    let reached: u64 = rows.iter().map(|r| r.reach[2]).sum();
    assert_eq!(
        searched, reached,
        "the palette-gate reach counter and the search counter disagree — one \
         of `nonrd_palette_arm_is_live` and the `if pick.palette_arm_live` \
         dispatch is reading a different predicate than the other"
    );
    assert!(
        searched > 1000,
        "only {searched} palette searches ran across the whole grid; the cells \
         no longer exercise `av1_search_palette_mode_luma`"
    );

    // (3) the search WON leaves — so the winner writeback (DC_PRED + the
    // palette tx size + the tx_type_map copy gate + `palette_y`) is covered,
    // not just the `rdcost = INT64_MAX` exit.
    let won: u64 = rows.iter().map(|r| r.stats[1]).sum();
    assert!(
        won > 0,
        "every one of the {searched} palette searches returned NO palette, so \
         this grid proves nothing about the winner path"
    );

    // (4) both sides of `color_palette_thresh` are exercised.
    let t32: u64 = rows.iter().map(|r| r.stats[2]).sum();
    assert!(
        t32 > 0 && t32 < searched,
        "`color_palette_thresh` took only one value across the grid ({t32} of \
         {searched} searches at 32) — the `best_sad_norm < 500` branch \
         (nonrd_pickmode.c:1713) is untested"
    );

    // (5) the palette search moves the REFERENCE bytes on the many-colour
    // family, so matching it is a statement about the palette search.
    assert!(
        palette_moved_bytes >= 6,
        "only {palette_moved_bytes} of the 12 many-colour cells changed when \
         aomenc's own `--enable-palette` was toggled; the reference this gate \
         matches is nearly palette-free"
    );

    // (6) byte identity.
    let bad: Vec<String> = rows
        .iter()
        .filter(|r| !r.matched)
        .map(|r| format!("{} ({:+})", r.label, r.delta))
        .collect();
    assert!(
        bad.is_empty(),
        "the nonrd estimate arm's palette search diverged from aomenc. With the \
         full-RD arm provably unreachable (arms[0] == 0 above), the cause is in \
         `nonrd_pickmode::search_palette_mode_luma` or in what it passes to \
         `palette_search::rd_pick_palette_intra_sby`: {bad:?}"
    );
}

// ---------------------------------------------------------------------------
// What this landing FOUND and did not cause: `PALETTE_MANY_COLORS_OPEN`.
// ---------------------------------------------------------------------------

/// **A divergence class in the SHARED palette machinery, reachable at every
/// speed, found by this landing's content and pinned rather than attributed
/// to the arm this landing ports.**
///
/// `fine_color_cell` puts 33..64 distinct luma values inside one 16x16 leaf.
/// That is the band where `av1_rd_pick_palette_intra_sby`'s own gate
/// (`colors_threshold > 1 && colors_threshold <= color_thresh_palette`,
/// palette.c:592) is decision-bearing, and — more importantly — the band where
/// `colors > PALETTE_MAX_SIZE`, so the k-means arm and the descending-order
/// re-search of `av1_rd_pick_palette_intra_sby` do real work. Nothing in the
/// tree had ever encoded such a block: `rd_close_palette.rs`'s text/UI content
/// has <= 8 colours, which takes the `colors == PALETTE_MIN_SIZE` /
/// `max_n == colors` shortcuts instead.
///
/// **The attribution is asserted, not argued.** The same content is encoded
/// through the FULL-RD path at `--cpu-used` 0, 2 and 6, where
/// `nonrd_leaf_arms()` reads `[0, 0]` — the nonrd walk does not run at all, so
/// `search_palette_mode_luma` is not merely uninvolved, it is unreachable.
/// Divergences there are the shared machinery's. The test fails if that control
/// ever goes clean while the speed-9 rows still diverge, because then the class
/// WOULD be this arm's.
///
/// **CLOSED 2026-08-30 by KB-41 root #23 — the doc above is the history, not the
/// current state.** The class was never in the k-means/descending-order arms: it was
/// the DC_PRED term of every nonrd palette candidate's header. `mbmode_cost` was
/// filled from an all-zero placeholder CDF in `real_costs.rs`, so
/// `av1_search_palette_mode_luma`'s `mbmode_cost[size_group_lookup[bsize]][DC_PRED]`
/// (intra_mode_search.c:1139-1140, :1152) read 3 where C reads 375 at BLOCK_16X16 —
/// a 372/512-bit constant on every candidate, which is the size of these near-ties.
/// `fc->y_mode_cdf` is `default_if_y_mode_cdf` and never adapts on an intra frame, so
/// the KEY value is exactly that default table. The full-RD control (a) and the
/// speed-9 rows (b) are now BOTH byte-identical, so this is a hard byte gate.
///
/// Per playbook §10 the next step WOULD have been a sibling-C dump of
/// `av1_rd_pick_palette_intra_sby`'s per-candidate RD on the smallest divergent
/// cell (64x64 n40 cq40 at `--cpu-used 0`); the actual localization went through the
/// paired C/port `intra_mode_info_cost_y` breakdown instead, which showed every term
/// but `mode_cost` matching to the unit.
#[test]
#[ignore = "24 encode pairs; a hard byte gate since KB-41 root #23 closed this class"]
fn many_colour_palette_blocks_are_pinned_and_are_not_this_arm() {
    c::ref_init();
    let knobs = ToggleKnobs {
        enable_palette: true,
        // The reference is `--tune-content=screen` (see `c_ref`):
        // the port's screen-content decision must take that arm, not the detector.
        tune_content_screen: true,
        ..Default::default()
    };

    // (a) the FULL-RD control: no nonrd leaf exists at these speeds.
    let mut full_rd_diverged = 0usize;
    let mut full_rd_total = 0usize;
    for speed in [0i32, 2, 6] {
        for ncol in [38u16, 40, 42] {
            let cell = fine_color_cell("fc64", 64, 64, 40, speed, ncol);
            let c_tu = c_ref(&cell);
            let real = EncodeCell::frame_obu_payload(&c_tu);
            reset_nonrd_leaf_arms();
            let got = cell.port_encode_with(&c_tu, &knobs);
            let arms = nonrd_leaf_arms();
            assert_eq!(
                arms,
                [0, 0],
                "fc64 n{ncol} cq40 s{speed}: the NONRD walk ran at an RD speed, so this \
                 row is no longer a control for the estimate arm"
            );
            full_rd_total += 1;
            if got != real {
                full_rd_diverged += 1;
                println!(
                    "  FULL-RD s{speed} n{ncol} cq40: DIVERGE {:+}",
                    got.len() as i64 - real.len() as i64
                );
            }
        }
    }
    // KB-41 (2026-08-30): the full-RD control IS clean now — 9/9 many-colour
    // cells byte-identical at speeds 0/2/6 once the palette cost tables followed
    // the per-SB refresh (+ the UV palette flag cost). The "shared palette
    // machinery is inexact in the (32, 64] band" reading was those stale costs.
    // Pinned as a hard byte gate; any remaining speed-9 divergence below is
    // therefore attributable to the speed-9 path itself.
    assert_eq!(
        full_rd_diverged, 0,
        "the full-RD many-colour palette control REGRESSED ({full_rd_diverged} of \
         {full_rd_total} cells diverge; KB-41 closed all 9) — the palette cost tables \
         stopped following the per-SB refresh, or the UV palette flag cost is gone"
    );

    // (b) the speed-9 rows of the same family, pinned.
    let mut diverged: Vec<String> = Vec::new();
    for ncol in [40u16, 56, 64] {
        for cq in [12, 40, 63] {
            let cell = fine_color_cell("fc256", 256, 256, cq, 9, ncol);
            let c_tu = c_ref(&cell);
            let row = measure(&format!("fc256 n{ncol} cq{cq}"), &cell, &c_tu);
            println!(
                "  {}: {} {:+} reach {:?} arms {:?} stats {:?}",
                row.label,
                if row.matched { "MATCH" } else { "DIVERGE" },
                row.delta,
                row.reach,
                row.arms,
                row.stats
            );
            assert_eq!(
                row.arms[0], 0,
                "{}: a full-RD leaf ran at speed 9",
                row.label
            );
            if !row.matched {
                diverged.push(format!("{} ({:+})", row.label, row.delta));
            }
        }
    }
    // Measured 2026-08-03 as `["fc256 n40 cq40 (-1)"]`; **CLOSED 2026-08-30 by KB-41
    // root #23** — `mbmode_cost` was filled from an all-zero placeholder CDF, so the
    // DC_PRED term every nonrd palette candidate's header carries
    // (`av1_search_palette_mode_luma` -> `mbmode_cost[size_group_lookup[bsize]][DC_PRED]`,
    // intra_mode_search.c:1139-1140) was ~2 orders of magnitude cheap, which is exactly
    // the near-tie that flipped this cell. Now pinned EMPTY: this whole family is
    // byte-identical at speed 9, so `PALETTE_MANY_COLORS_OPEN` is closed.
    assert_eq!(
        diverged,
        Vec::<String>::new(),
        "the pinned many-colour set moved. It was closed by KB-41 root #23 (mbmode_cost \
         now carries default_if_y_mode_cdf, not a zeroed placeholder); ANY entry here is \
         a regression of that root or of the shared palette search"
    );
}

/// **`color_palette_thresh` is implemented but is NOT byte-gate-protected,
/// and this test says so with a number instead of a comment.**
///
/// `x->color_palette_thresh = (best_sad_norm < 500) ? 32 : 64`
/// (nonrd_pickmode.c:1713) can only change an encode when a block's colour
/// count lands in `(32, 64]` — and that is exactly the
/// `PALETTE_MANY_COLORS_OPEN` band above, which used to be inexact, so the byte
/// gate could not witness the formula. **That changed 2026-08-30 (KB-41 root #23):
/// the band is byte-exact on the shipped formula (0 of 75), so this IS now a byte
/// gate on it.**
///
/// What CAN be measured, and is: over a 75-cell grid in that band, the C
/// formula diverges on strictly FEWER cells than either constant. That is
/// evidence for the formula without pretending it is proof.
///
/// | `color_palette_thresh` | divergent cells of 75 (2026-08-03, pre-root-#23) |
/// |---|---|
/// | `(best_sad_norm < 500) ? 32 : 64` (C) | **48** |
/// | hardcoded 32 | 58 |
/// | hardcoded 64 | 66 |
///
/// Measured 2026-08-03 by replacing the field at its construction site in
/// `partition_pick.rs` and re-running the grid below. **The shipped-formula row is
/// now 0 of 75 (KB-41 root #23); the two hardcoded-constant rows have NOT been
/// re-measured on the corrected costs, so treat the ordering above as the
/// pre-#23 evidence it was.** The gate here re-runs only the shipped arm and pins
/// its count, so a change to the formula (or to the shared machinery) shows up as a
/// moved number.
#[test]
#[ignore = "75 encode pairs at 64x64..256x256; the color_palette_thresh evidence"]
fn color_palette_thresh_band_divergence_count_is_pinned() {
    c::ref_init();
    let knobs = ToggleKnobs {
        enable_palette: true,
        // The reference is `--tune-content=screen` (see `c_ref`):
        // the port's screen-content decision must take that arm, not the detector.
        tune_content_screen: true,
        ..Default::default()
    };
    let mut diverged = 0usize;
    let mut total = 0usize;
    let mut t32 = 0u64;
    let mut searched = 0u64;
    for &(w, h) in &[(64usize, 64usize), (128, 128), (256, 256)] {
        for ncol in [38u16, 39, 40, 41, 42] {
            for cq in [36, 38, 40, 42, 44] {
                let cell = fine_color_cell("fc", w, h, cq, 9, ncol);
                let c_tu = c_ref(&cell);
                let row = measure(&format!("fc {w}x{h} n{ncol} cq{cq}"), &cell, &c_tu);
                assert_eq!(row.arms[0], 0, "{}: a full-RD leaf ran", row.label);
                total += 1;
                searched += row.stats[0];
                t32 += row.stats[2];
                if !row.matched {
                    diverged += 1;
                }
            }
        }
    }
    println!("  band: {diverged} of {total} diverge; {t32} of {searched} searches at thresh 32");
    assert!(
        t32 > 0 && t32 < searched,
        "the band no longer exercises both sides of `color_palette_thresh` \
         ({t32} of {searched} at 32)"
    );
    // Measured 2026-08-03 as (48, 75); **(0, 75) since 2026-08-30, KB-41 root #23** —
    // the band's divergences were all `PALETTE_MANY_COLORS_OPEN`, and that class closed
    // when `mbmode_cost` stopped being a zeroed placeholder (see the test above). The
    // band still exercises both sides of `color_palette_thresh` (asserted above), so it
    // remains a live gate on the formula — now as a hard byte gate.
    assert_eq!(
        (diverged, total),
        (0, 75),
        "the `color_palette_thresh` band moved. It was closed by KB-41 root #23; ANY \
         divergence here is a regression of that root, of the shared palette search, or \
         of `nonrd_color_palette_thresh` itself"
    );
}
