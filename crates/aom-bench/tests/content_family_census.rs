//! **The census as a GATE.** Every coding-tool family the harness claims to
//! reach is pinned here, so a content change, an encoder change or a speed-
//! feature change that silently stops exercising a family fails loudly instead
//! of turning a future band into a structural zero nobody notices.
//!
//! # Why this exists
//!
//! `benchmarks/winperf_content_census_2026-08-03.md` §4 shipped a LIST of what
//! the harness could not see. A list is a document; it goes stale the day
//! someone changes a generator parameter or a speed feature. This is the same
//! statement as an executable artefact (`docs/DIFFERENTIAL_PLAYBOOK.md` §8:
//! derive coverage from artefacts, not from names).
//!
//! The failure it prevents is real and already happened once: KB-PERF-4's
//! Windows band was read off content where `z1` fires **six times in a 1 MP
//! frame**, and the null it reported was the code never running.
//!
//! # Both directions (playbook §5)
//!
//! Each pin carries the family's CURRENT measured share and fails if the share
//! drops below it — and separately if it rises far enough above it that the pin
//! has stopped describing reality. The second half is the point: when a change
//! makes a family MORE reachable, that is news the record should carry, not a
//! silently-passing test.
//!
//! # Cost
//!
//! Eight `port_encode`s (four contents, warm-up subtracted), no C oracle, no
//! fit: ~21 s, of which the screen row's palette + intraBC searches are ~20.
//! It needs `--features census` (the counters are default-off and a census
//! build is never a timing build), which is why it is not in the default
//! `cargo test` tier — `just census-gate` runs it, and CI runs that.

#![cfg(feature = "census")]

use aom_bench::winperf::{self, Content};
use aom_bench::{ToggleKnobs, stream_allows_screen_content_tools};
use aom_dsp::census::{self, Counts};

/// A family's reach on one content, as measured on 2026-08-03 at the study
/// cell (1024x1024 / cq44 / cpu-used 6 / ALLINTRA / 8-bit 4:2:0) — except the
/// `Screen` rows, which are at [`winperf::SCREEN_GATE_CELL`] (512x384).
struct Pin {
    content: Content,
    /// Which knobs the number was measured under. Palette and intraBC are
    /// default-OFF (`--enable-palette` / `--enable-intrabc`), so a pin taken
    /// with them is a statement about a knob-gated encoder and must say so.
    screen_knobs: bool,
    family: &'static str,
    /// Percent. The measured value, floored: the gate fails below this.
    floor: f64,
    /// Percent. The gate ALSO fails at or above this, with a re-pin message —
    /// a family that became much more reachable is a finding.
    ceiling: f64,
}

/// The share of `family` on a census, as a percent of that family's natural
/// denominator. Kept in one place so the gate and the census tool cannot drift.
fn share(c: &Counts, family: &str) -> f64 {
    let p = |n: u64, d: u64| if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 };
    match family {
        "directional_px" => p(c.directional_px(), c.intra_total_px()),
        "chroma_pred_calls" => p(c.plane_calls[1] + c.plane_calls[2], c.plane_total()),
        "cfl_pred_px" => p(c.cfl_px, c.intra_total_px() + c.cfl_px),
        "cfl_leaves" => p(c.cfl_leaves(), c.leaf_chroma_ref),
        "rect_leaves" => p(c.rect_leaves(), c.leaves()),
        "leaves_le_8px" => p(c.small_leaves(), c.leaves()),
        "fwd_tx_4pt" => p(c.small_fwd_tx(), c.fwd_tx.iter().flatten().sum()),
        "fwd_tx_rect" => p(c.rect_fwd_tx(), c.fwd_tx.iter().flatten().sum()),
        "fwd_tx_non_dct" => p(c.non_dct_fwd_tx(), c.fwd_tx.iter().flatten().sum()),
        "nonzero_angle_delta" => p(c.nonzero_angle_delta_leaves(), c.leaves()),
        "palette_y" => p(c.palette_y_leaves(), c.leaves()),
        "palette_uv" => p(c.palette_uv_leaves(), c.leaves()),
        "intrabc" => p(c.leaf_intrabc, c.leaves()),
        "filter_intra" => p(c.filter_intra_leaves(), c.leaves()),
        other => panic!("unknown family {other:?}"),
    }
}

/// One `port_encode`, warm-up subtracted.
///
/// The screen row runs at [`winperf::SCREEN_GATE_CELL`] (512x384) rather than
/// [`winperf::CELL`]: the intraBC displacement search on 1 MP of screen content
/// runs for minutes, and a gate nobody runs is not a gate. Perf bands still use
/// the 1 MP cell; this is a coverage assertion.
fn census(content: Content, screen_knobs: bool) -> Box<Counts> {
    let ((w, h, q, s), boot) = if content == Content::Screen {
        (winperf::SCREEN_GATE_CELL, winperf::bootstrap_screen_gate())
    } else {
        (winperf::CELL, winperf::bootstrap(content))
    };
    let cell = winperf::cell(w, h, q, s, content);
    let knobs = ToggleKnobs {
        enable_palette: screen_knobs,
        enable_intrabc: screen_knobs,
        ..Default::default()
    };
    census::reset();
    let _ = cell.port_encode_with(&boot, &knobs);
    let base = census::snapshot();
    let _ = cell.port_encode_with(&boot, &knobs);
    let c = census::snapshot().since(&base);
    assert!(!c.is_empty(), "{content:?}: empty census");
    // Non-vacuity for the plane annotation (playbook §2): the DSP-side hook
    // inside `predict_intra_high` cannot be missed, the encoder-side plane tags
    // can be, and a missed tag would deflate `chroma_pred_calls` silently.
    assert_eq!(
        c.plane_total(),
        c.intra_total_calls(),
        "{content:?}: an encoder call site is missing census::note_plane_intra_pred"
    );
    c
}

/// The pinned coverage picture, measured 2026-08-03 on commit `HEAD` at the
/// study cell. Full table + provenance:
/// `benchmarks/winperf_family_census_2026-08-03.md`.
///
/// Floors are set a little under the measured value and ceilings a little over
/// it; both bounds are stated per row in the record so a re-pin is a diff, not
/// a judgement call.
const PINS: &[Pin] = &[
    // ---- photo: the mode-scoped content ---------------------------------
    Pin { content: Content::Photo, screen_knobs: false, family: "directional_px", floor: 15.0, ceiling: 21.0 },
    Pin { content: Content::Photo, screen_knobs: false, family: "nonzero_angle_delta", floor: 9.0, ceiling: 14.0 },
    Pin { content: Content::Photo, screen_knobs: false, family: "rect_leaves", floor: 4.0, ceiling: 9.0 },
    Pin { content: Content::Photo, screen_knobs: false, family: "fwd_tx_4pt", floor: 5.0, ceiling: 10.0 },
    Pin { content: Content::Photo, screen_knobs: false, family: "chroma_pred_calls", floor: 33.0, ceiling: 40.0 },
    // ---- detail: the allocation content ---------------------------------
    Pin { content: Content::Detail, screen_knobs: false, family: "fwd_tx_non_dct", floor: 40.0, ceiling: 49.0 },
    Pin { content: Content::Detail, screen_knobs: false, family: "chroma_pred_calls", floor: 27.0, ceiling: 35.0 },
    // ---- smooth: the low-work end ---------------------------------------
    Pin { content: Content::Smooth, screen_knobs: false, family: "directional_px", floor: 10.0, ceiling: 17.0 },
    // ---- screen: the ONLY content that reaches the screen tools ----------
    // Measured 21.61 / 24.04 / 75.19 at `winperf::SCREEN_GATE_CELL` on
    // 2026-08-03; RE-PINNED 2026-08-30 (KB-42) to 22.75 / 33.63 / 80.84 after
    // KB-41 roots #3-#6 ported the speed-dependent IntraBC search
    // (`intrabc_search_level` / hash-8x8 cap / DIAMOND site configs, `735a0a6d`,
    // 30/30 datagen cells byte-identical against the C oracle). More leaves win
    // IntraBC, and IntraBC winners are small, so `intrabc` and `leaves_le_8px`
    // both rose — the ceiling half of the gate firing is exactly the "became
    // MORE reachable" news it exists to report, and the named root explains it.
    // Both FLOORS rise (18 -> 25, 68 -> 73): this is a tightening, not a
    // relaxation. Each band keeps its own historic relative shape
    // (intrabc 0.75x/1.25x, leaves_le_8px 0.90x/1.09x of the measurement).
    // `palette_y` moved 21.61 -> 22.75, still well inside its band, so its
    // bounds are left alone and only the measurement is recorded.
    Pin { content: Content::Screen, screen_knobs: true, family: "palette_y", floor: 16.0, ceiling: 28.0 },
    Pin { content: Content::Screen, screen_knobs: true, family: "intrabc", floor: 25.0, ceiling: 42.0 },
    Pin { content: Content::Screen, screen_knobs: true, family: "leaves_le_8px", floor: 73.0, ceiling: 88.0 },
];

/// Every pinned family is still reached, and none has moved so far that the pin
/// stopped describing the harness.
#[test]
fn every_pinned_family_is_still_reached() {
    let mut fails: Vec<String> = Vec::new();
    // Census each (content, knobs) pair once; the pins index into it.
    let mut cache: Vec<((Content, bool), Box<Counts>)> = Vec::new();
    for pin in PINS {
        let key = (pin.content, pin.screen_knobs);
        if !cache.iter().any(|(k, _)| *k == key) {
            cache.push((key, census(pin.content, pin.screen_knobs)));
        }
        let c = &cache.iter().find(|(k, _)| *k == key).unwrap().1;
        let got = share(c, pin.family);
        // Printed unconditionally (visible under `--nocapture`) so a re-pin is
        // a copy rather than a re-derivation.
        println!(
            "{:?}{}\t{}\t{got:.2}\t[{:.2}, {:.2})",
            pin.content,
            if pin.screen_knobs { "+screen-knobs" } else { "" },
            pin.family,
            pin.floor,
            pin.ceiling,
        );
        if got < pin.floor {
            fails.push(format!(
                "{:?}{} {}: {got:.2} % < pinned floor {:.2} % — this family is no \
                 longer reached; a band read against it would be a structural zero",
                pin.content,
                if pin.screen_knobs { "+screen-knobs" } else { "" },
                pin.family,
                pin.floor,
            ));
        } else if got >= pin.ceiling {
            fails.push(format!(
                "{:?}{} {}: {got:.2} % >= pinned ceiling {:.2} % — the family became \
                 MORE reachable; RE-PIN this row (floor {:.2} -> ~{:.2}) and update \
                 benchmarks/winperf_family_census_2026-08-03.md",
                pin.content,
                if pin.screen_knobs { "+screen-knobs" } else { "" },
                pin.family,
                pin.ceiling,
                pin.floor,
                got * 0.9,
            ));
        }
    }
    assert!(fails.is_empty(), "content family coverage moved:\n  {}", fails.join("\n  "));
}

/// The census must be able to tell a family that is UNREACHED from one that is
/// merely rare — so the gate also pins the families the harness is known to
/// miss, and fails if one of them silently starts firing (which would mean the
/// coverage record is out of date in the good direction).
///
/// * **filter-intra is a SPEED zero, not a content zero.** At the study cell's
///   `--cpu-used 6` the port sets `prune_filter_intra_level = 2`
///   (`speed_features.rs`, libaom `speed_features.c:529`), i.e.
///   `rd_pick_filter_intra_sby` is never called. No source can reach it here;
///   only a slower cell can.
/// * **palette / intraBC without the knobs.** Both are default-off, so the
///   DEFAULT encoder codes neither, on any content, including the screen one.
#[test]
fn the_known_zeros_are_still_zero_and_still_for_the_stated_reason() {
    for content in Content::ALL {
        let c = census(content, false);
        assert_eq!(
            c.filter_intra_leaves(),
            0,
            "{content:?}: filter-intra fired at cpu-used 6, where \
             prune_filter_intra_level == 2 should make the search unreachable. \
             Either the speed feature changed or the pin is wrong — do not \
             delete this assertion, re-derive it."
        );
        assert_eq!(
            c.palette_y_leaves() + c.leaf_intrabc,
            0,
            "{content:?}: a screen tool fired with --enable-palette / \
             --enable-intrabc OFF. Those knobs are default-off; if that changed, \
             every default-knob band in benchmarks/ needs re-reading."
        );
    }
}

/// The screen content is the one source whose bootstrap carries
/// `allow_screen_content_tools`, and the other three do not.
///
/// Without that header bit the palette and intraBC searches decline exactly as
/// C's do, so this is the precondition the whole screen row rests on. Asserting
/// the three photographic contents do NOT carry it is what stops this passing
/// vacuously against a harness that force-set the bit everywhere.
#[test]
fn only_the_screen_bootstrap_signals_screen_content_tools() {
    for content in Content::ALL {
        let got = stream_allows_screen_content_tools(&winperf::bootstrap(content));
        let want = content == Content::Screen;
        if want {
            assert!(
                stream_allows_screen_content_tools(&winperf::bootstrap_screen_gate()),
                "the 256x256 gate fixture lost allow_screen_content_tools"
            );
        }
        assert_eq!(
            got, want,
            "{content:?}: allow_screen_content_tools = {got}, expected {want}. \
             The screen fixture is bootstrapped through c_encode_screen and real \
             aomenc's own detection decides; nothing here forces the bit."
        );
    }
}

/// The screen generator's two structural properties, measured on the SOURCE
/// PIXELS rather than inferred from the encoder's decisions — so this still
/// says something if a future encoder change stops picking the tools.
///
/// 1. **Bounded colour count.** Every 16x16 luma block of `screen` contains at
///    most 8 distinct values (AV1's palette maximum). The comparator contents
///    are measured in the same test so it cannot pass vacuously: `photo` and
///    `detail` blow past that bound everywhere.
/// 2. **Exact repetition.** A large share of `screen`'s 8x8 blocks have an
///    exact duplicate earlier in raster order — the causal region intraBC
///    copies from. `photo` and `detail` have essentially none.
#[test]
fn screen_source_is_few_coloured_and_repetitive_and_the_others_are_not() {
    use std::collections::{HashMap, HashSet};
    let (w, h) = (256usize, 256usize);
    let mut report = String::new();
    let mut screen: Option<(usize, f64)> = None;
    let mut worst_other = (0usize, 0.0f64);
    for content in Content::ALL {
        let buf = winperf::synth_i420(w, h, content);
        // Max distinct luma values over 16x16 blocks.
        let mut max_colors = 0usize;
        for by in (0..h).step_by(16) {
            for bx in (0..w).step_by(16) {
                let mut set = HashSet::new();
                for y in by..by + 16 {
                    for x in bx..bx + 16 {
                        set.insert(buf[y * w + x]);
                    }
                }
                max_colors = max_colors.max(set.len());
            }
        }
        // Share of 8x8 blocks (on an 8-aligned grid) that repeat an earlier one.
        let mut seen: HashMap<Vec<u8>, ()> = HashMap::new();
        let (mut total, mut dup) = (0usize, 0usize);
        for by in (0..h).step_by(8) {
            for bx in (0..w).step_by(8) {
                let mut blk = Vec::with_capacity(64);
                for y in by..by + 8 {
                    blk.extend_from_slice(&buf[y * w + bx..y * w + bx + 8]);
                }
                total += 1;
                if seen.insert(blk, ()).is_some() {
                    dup += 1;
                }
            }
        }
        let dup_pct = 100.0 * dup as f64 / total as f64;
        report.push_str(&format!(
            "{content:?}: max distinct colours per 16x16 = {max_colors}, \
             repeated 8x8 blocks = {dup_pct:.1} %\n"
        ));
        if content == Content::Screen {
            screen = Some((max_colors, dup_pct));
        } else {
            worst_other = (worst_other.0.max(max_colors), worst_other.1.max(dup_pct));
        }
    }
    let (screen_colors, screen_dup) = screen.expect("Content::Screen censused");
    assert!(
        screen_colors <= 8,
        "screen source has {screen_colors} distinct colours in some 16x16 block; \
         AV1's palette carries at most 8, so palette could not win there.\n{report}"
    );
    assert!(
        screen_dup >= 20.0,
        "screen source repeats only {screen_dup:.1} % of its 8x8 blocks; intraBC \
         needs an exact earlier copy to displace to.\n{report}"
    );
    // The comparator half — this is what makes the two assertions above
    // statements about SCREEN rather than about 16x16 blocks in general.
    assert!(
        worst_other.0 > 64,
        "the photographic contents were expected to blow past the palette bound \
         and did not ({} colours) — this test would now pass vacuously.\n{report}",
        worst_other.0
    );
    assert!(
        worst_other.1 < 1.0,
        "a photographic content repeats {:.1} % of its 8x8 blocks — the \
         repetition assertion above is no longer discriminating.\n{report}",
        worst_other.1
    );
}
