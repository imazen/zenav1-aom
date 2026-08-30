//! **KB-13's two remaining real-content cells** — `av1-1-b8-00-quantizer-00`
//! 128x128@64,64 and `av1-1-b8-23-film_grain-50` 64x64@96,64, both at
//! `--cpu-used 3 --cq-level 63` (CLAUDE.md "KB-13 … REMAINING, current
//! 2026-08-03: TWO cells, both `cpu3 cq63`").
//!
//! Replayed here the way `kb41_screen_detected_defaults` replays a cell: the
//! oracle's flagless ALLINTRA defaults stream (`c_encode_defaults`) against the
//! port driven with the oracle's OWN screen decision (palette + IntraBC on when
//! the header says `allow_screen_content_tools`, off otherwise), so a
//! screen-detected frame is not misread as a search divergence (that misread
//! was the whole of KB-30). Self-promoting in both directions: the pinned open
//! set below is asserted EQUAL to what is observed, so a cell that closes (or
//! a new one that opens) fails loudly and gets re-pinned with its KB.

use aom_bench::{stream_allows_screen_content_tools, EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// (label, vector, crop (w, h, x, y), cq, speed) — the two KB-13 cells.
const CELLS: &[(&str, &str, (usize, usize, usize, usize), i32, i32)] = &[
    ("kb13_q00_128", "av1-1-b8-00-quantizer-00", (128, 128, 64, 64), 63, 3),
    ("kb13_grain_64", "av1-1-b8-23-film_grain-50", (64, 64, 96, 64), 63, 3),
];

/// Cells still diverging, as (label, byte delta port − oracle).
/// **CLOSED 2026-08-30 by KB-41 roots #7-#13 (zenav1-aom `38a92657`)**: both
/// byte-exact on replay (see the test's println for the per-cell line).
const KB13_OPEN: &[(&str, i64)] = &[];

#[test]
#[ignore = "on-demand: two conformance-vector crops at cpu 3 (needs the AV1 vector corpus)"]
fn kb13_cpu3_cq63_cells() {
    c::ref_init();
    let mut open: Vec<(&str, i64)> = Vec::new();
    for &(label, vector, crop, cq, speed) in CELLS {
        let cell = EncodeCell::real_content(label, vector, Some(crop), cq, speed);
        let c_tu = cell.c_encode_defaults();
        assert!(!c_tu.is_empty(), "{label}: oracle encode failed");
        let real = EncodeCell::frame_obu_payload(&c_tu);
        let sct = stream_allows_screen_content_tools(&c_tu);
        let knobs = ToggleKnobs {
            enable_palette: sct,
            enable_intrabc: sct,
            ..Default::default()
        };
        let ours = cell.port_encode_with(&c_tu, &knobs);
        let d = ours.len() as i64 - real.len() as i64;
        println!(
            "  {label:<14} {}x{} cq{cq} cpu{speed} sct={} port {:>6} vs oracle {:>6} delta {d:+} {}",
            crop.0,
            crop.1,
            u8::from(sct),
            ours.len(),
            real.len(),
            if ours == real { "MATCH" } else { "DIVERGE" }
        );
        if ours != real {
            open.push((label, d));
        }
    }
    assert_eq!(
        open,
        KB13_OPEN.to_vec(),
        "KB-13's open set moved. FEWER entries => something closed it: re-pin KB13_OPEN and \
         say which KB. MORE => a regression on a real-content cell at cpu 3"
    );
}
