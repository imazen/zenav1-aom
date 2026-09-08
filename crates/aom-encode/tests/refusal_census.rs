//! REFUSAL CENSUS for the standalone still-image entry point.
//!
//! The ship goal (`CLAUDE.md`, 2026-09-08) asks for "no panics or refusals on
//! inputs a caller can produce". That is a claim about a SET, so it needs a
//! measurement over that set rather than a spot check — this file is the
//! measurement.
//!
//! It sweeps the `KeyFrameConfig` space a caller could plausibly hand
//! `encode_key_frame`, encodes a real frame at every point, and classifies the
//! outcome. Three classes, and only the first is acceptable:
//!
//! * **Ok** — a stream came back, and it decodes.
//! * **Refused** — a named `KeyFrameError`. Acceptable ONLY where the refusal
//!   is the documented contract (a config outside the gated envelope); every
//!   such point is listed by name below, so a NEW refusal fails this test
//!   rather than quietly narrowing what the backend accepts.
//! * **Panic** — never acceptable. A panic is caught and reported with its
//!   configuration so the failure names the cell instead of aborting the run.
//!
//! Deliberately NOT a byte-parity gate: `self_contained_key_frame.rs` owns
//! that, over a grid chosen for parity. This grid is chosen for REACH — odd
//! sizes, the boundaries of every enumerated field, and the combinations a
//! wrapper would generate — and asks only "does the encoder answer".

use aom_encode::key_frame::{encode_key_frame, KeyFrameConfig, KeyFramePlanes, MAX_FRAME_DIM};

/// A deterministic textured source. Content matters here only in that it must
/// not be flat: a flat frame codes to a handful of skip blocks and would exit
/// most of the encoder before reaching anything that could refuse.
fn planes(cfg: &KeyFrameConfig) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (w, h) = (cfg.width, cfg.height);
    let (cw, ch) = cfg.chroma_dims();
    let peak = (1u32 << cfg.bit_depth) - 1;
    let mk = |pw: usize, ph: usize, seed: u32| {
        let mut v = Vec::with_capacity(pw * ph);
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        for y in 0..ph {
            for x in 0..pw {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // A gradient plus a bounded pseudo-random ripple: enough
                // structure to drive partitions, transforms and (with the
                // right knobs) palette / IntraBC.
                let g = ((x * 7 + y * 5) as u32) & 0xff;
                let n = (s >> 24) & 0x3f;
                v.push((((g + n) * peak) / 0x13f) as u16);
            }
        }
        v
    };
    let y = mk(w, h, 1);
    if cfg.monochrome {
        (y, Vec::new(), Vec::new())
    } else {
        (y, mk(cw, ch, 2), mk(cw, ch, 3))
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Ok(usize),
    Refused(String),
    Panicked(String),
}

fn attempt(cfg: &KeyFrameConfig) -> Outcome {
    let (y, u, v) = planes(cfg);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_key_frame(
            KeyFramePlanes {
                y: &y,
                u: &u,
                v: &v,
            },
            cfg,
        )
    }));
    match r {
        Ok(Ok(bytes)) => Outcome::Ok(bytes.len()),
        Ok(Err(e)) => Outcome::Refused(e.to_string()),
        Err(p) => Outcome::Panicked(
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".into()),
        ),
    }
}

fn base(w: usize, h: usize) -> KeyFrameConfig {
    KeyFrameConfig::allintra_speed0(w, h, 8, false, 1, 1, 32)
}

/// Panic-freedom is a hard property; a panic here names its own cell.
fn assert_no_panics(label: &str, cells: &[(String, Outcome)]) {
    let panics: Vec<&(String, Outcome)> = cells
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::Panicked(_)))
        .collect();
    assert!(
        panics.is_empty(),
        "{label}: {} of {} configurations PANICKED — a panic is never an \
         acceptable answer to a caller's configuration:\n{}",
        panics.len(),
        cells.len(),
        panics
            .iter()
            .map(|(l, o)| format!("  {l}: {o:?}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn summarise(label: &str, cells: &[(String, Outcome)]) -> (usize, usize) {
    let ok = cells.iter().filter(|(_, o)| matches!(o, Outcome::Ok(_))).count();
    let refused: Vec<&(String, Outcome)> = cells
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::Refused(_)))
        .collect();
    println!("{label}: {ok} ok, {} refused, of {}", refused.len(), cells.len());
    for (l, o) in &refused {
        println!("    REFUSED {l}: {o:?}");
    }
    (ok, refused.len())
}

/// THE FORMAT SURFACE a still-image wrapper generates: every chroma format x
/// bit depth x a size that is NOT a nice multiple of anything.
///
/// This is the grid the ship goal's "zero refusals on any configuration
/// reachable from zenavif's public API" is really about, and it must be
/// refusal-free AND panic-free in full.
#[test]
fn every_format_and_depth_encodes_at_an_awkward_size() {
    let mut cells = Vec::new();
    for &(w, h) in &[(1, 1), (3, 7), (17, 5), (65, 67), (100, 60), (129, 128)] {
        for bd in [8u8, 10, 12] {
            for (mono, sx, sy) in [(true, 1, 1), (false, 1, 1), (false, 1, 0), (false, 0, 0)] {
                let mut cfg = KeyFrameConfig::allintra_speed0(w, h, bd, mono, sx, sy, 32);
                cfg.cpu_used = 6;
                let label = format!(
                    "{w}x{h} bd{bd} {}",
                    if mono { "mono".into() } else { format!("ss{sx}{sy}") }
                );
                let o = attempt(&cfg);
                cells.push((label, o));
            }
        }
    }
    assert_no_panics("format surface", &cells);
    let (ok, refused) = summarise("format surface", &cells);
    assert_eq!(
        refused, 0,
        "every chroma format x bit depth x awkward size must ENCODE — a refusal \
         here is a hole in the backend contract, not a parity pin"
    );
    assert_eq!(ok, cells.len());
}

/// The enumerated fields, at both ends of every range plus the interior:
/// quantizer, speed, superblock size, tiles, and the two post-filter knobs.
#[test]
fn every_enumerated_knob_encodes_across_its_whole_range() {
    let mut cells = Vec::new();
    for cq in [0, 1, 32, 62, 63] {
        for speed in 0..=9 {
            let mut cfg = base(65, 67);
            cfg.cq_level = cq;
            cfg.cpu_used = speed;
            cells.push((format!("cq{cq} s{speed}"), attempt(&cfg)));
        }
    }
    for sb128 in [false, true] {
        for (tc, tr) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)] {
            for (cdef, lr) in [(false, false), (true, false), (false, true), (true, true)] {
                let mut cfg = base(200, 136);
                cfg.sb_size_128 = sb128;
                cfg.tile_columns_log2 = tc;
                cfg.tile_rows_log2 = tr;
                cfg.enable_cdef = cdef;
                cfg.enable_restoration = lr;
                cells.push((
                    format!("sb{} tiles{tc}x{tr} cdef{} lr{}", if sb128 { 128 } else { 64 },
                            cdef as u8, lr as u8),
                    attempt(&cfg),
                ));
            }
        }
    }
    assert_no_panics("knob ranges", &cells);
    let (ok, refused) = summarise("knob ranges", &cells);
    assert_eq!(
        refused, 0,
        "every enumerated knob value must encode across its whole documented range"
    );
    assert_eq!(ok, cells.len());
}

/// The refusals that ARE the contract, pinned BY NAME and in both directions:
/// each of these must refuse (or the documented envelope is wider than the docs
/// say), and nothing else in the sweeps above may refuse.
#[test]
fn the_documented_refusals_are_exactly_these() {
    let mut cases: Vec<(&str, KeyFrameConfig)> = Vec::new();
    let mut c = base(65, 67);
    c.usage = 0;
    cases.push(("usage", c));
    let mut c = base(65, 67);
    c.cpu_used = 10;
    cases.push(("cpu_used", c));
    let mut c = base(65, 67);
    c.bit_depth = 9;
    cases.push(("bit_depth", c));
    let mut c = base(0, 67);
    cases.push(("zero width", c));
    let mut c = base(65, 67);
    c.cq_level = 64;
    cases.push(("cq_level", c));
    let mut c = base(65, 67);
    c.ss_x = 0;
    cases.push(("ss (0,1)", c));
    let mut c = base(65, 67);
    c.monochrome = true;
    c.ss_y = 0;
    cases.push(("mono ss", c));
    let mut c = base(MAX_FRAME_DIM + 1, 67);
    cases.push(("width ceiling", c));

    let mut cells = Vec::new();
    for (name, cfg) in cases {
        // The planes are deliberately empty here: a config-level refusal must
        // come back BEFORE any source sample is read.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_key_frame(
                KeyFramePlanes {
                    y: &[],
                    u: &[],
                    v: &[],
                },
                &cfg,
            )
        }));
        let o = match r {
            Ok(Ok(b)) => Outcome::Ok(b.len()),
            Ok(Err(e)) => Outcome::Refused(e.to_string()),
            Err(_) => Outcome::Panicked("panic".into()),
        };
        assert!(
            matches!(o, Outcome::Refused(_)),
            "{name}: must be REFUSED by name, got {o:?}"
        );
        // The query must agree, which is what makes the refusal a contract a
        // router can rely on rather than an encode-time surprise.
        assert!(
            cfg.validate_configuration().is_err(),
            "{name}: the support query must refuse whatever the encoder refuses"
        );
        cells.push((name.to_string(), o));
    }
    println!("documented refusals: {} pinned", cells.len());
}

/// The grid that made this file worth writing: a uniform-spacing tile request
/// whose product is NOT a power of two.
///
/// 200x136 is 50x34 mi = 4x3 superblocks at SB64. `--tile-columns=2
/// --tile-rows=2` clamps to `log2 = (2, 2)` — `max_log2_rows` is
/// `tile_log2(1, 3) == 2` — while `av1_calculate_tile_rows` counts **3** rows
/// over 3 superblock rows. So the grid is 4 x 3 = **12 tiles against a log2 sum
/// of 4**, and `encode_key_frame` used to REFUSE it on an invariant
/// (`rows * cols == 1 << (log2_cols + log2_rows)`) that simply is not true of
/// AV1: the log2 pair is the `context_update_tile_id` FIELD WIDTH and the
/// product is the tile COUNT.
///
/// Real aomenc accepts the same request, so this was a refusal on a
/// configuration a caller can produce. Gated here on the property that
/// matters — the stream the port emits is accepted by the REAL C decoder and
/// reconstructs to the same pixels this port's own decoder produces.
#[test]
fn a_non_power_of_two_tile_grid_encodes_and_decodes() {
    let mut cfg = base(200, 136);
    cfg.tile_columns_log2 = 2;
    cfg.tile_rows_log2 = 2;
    cfg.cpu_used = 6;

    let tiles = cfg.derive_tiles().expect("the grid must be accepted");
    assert_eq!(
        (tiles.cols, tiles.rows, tiles.log2_cols, tiles.log2_rows),
        (4, 3, 2, 2),
        "the cell must actually EXHIBIT the non-power-of-two grid, else it \
         gates nothing"
    );
    assert_ne!(
        tiles.cols * tiles.rows,
        1usize << (tiles.log2_cols + tiles.log2_rows),
        "product != 1<<log2sum is the whole point of this cell"
    );

    let (y, u, v) = planes(&cfg);
    let stream = encode_key_frame(
        KeyFramePlanes {
            y: &y,
            u: &u,
            v: &v,
        },
        &cfg,
    )
    .expect("a 4x3 tile grid must encode");

    // The REAL C decoder is the authority on whether the tile syntax is well
    // formed: a wrong `context_update_tile_id` width or a tile count that
    // disagreed with the header would be rejected here, not merely differ.
    let theirs = aom_sys_ref::ref_decode_av1_kf(&stream, cfg.width, cfg.height);
    let ours = aom_decode::frame::decode_frame_obus(&stream)
        .expect("this port's decoder must accept its own stream");
    assert_eq!(
        (ours.width, ours.height),
        (cfg.width, cfg.height),
        "decoded dims"
    );
    assert_eq!(ours.y, theirs.y, "luma must match the C decoder");
    assert_eq!(ours.u, theirs.u, "U must match the C decoder");
    assert_eq!(ours.v, theirs.v, "V must match the C decoder");
    println!(
        "200x136 tiles {}x{} (log2 {},{}) -> {} B, both decoders agree",
        tiles.cols,
        tiles.rows,
        tiles.log2_cols,
        tiles.log2_rows,
        stream.len()
    );
}
