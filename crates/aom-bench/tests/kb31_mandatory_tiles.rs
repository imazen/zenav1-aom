//! **KB-31 — frames whose SIZE makes multi-tile MANDATORY (issue #6).**
//!
//! `EncodeCell::port_encode` asserted `tiles_log2 == 0` ("single-tile envelope
//! only") and therefore PANICKED, at every `--cpu-used`, on any frame libaom
//! must split into more than one tile. Two independent size predicates force
//! that split (`av1_get_tile_limits`, `av1/common/tile_common.c:31-50`, plus
//! `set_tile_info`'s stricter encoder-side column bound,
//! `av1/encoder/encoder.c:385-390`):
//!
//! * **width** — `sb_cols >= MAX_TILE_WIDTH >> sb_size_log2` (== 64 SB64s, i.e.
//!   `mi_cols >= 1009`, **width >= 4033 px**). Note the encoder's own loop uses
//!   `<=`, one column MORE than the limits function's `<`, so 64 SB columns is
//!   already a split; the grid below measures 4032 (1 tile) against 4096 (2);
//! * **area** — `sb_cols * sb_rows > MAX_TILE_AREA >> 2*sb_size_log2` (== 2304
//!   SB64s, i.e. roughly **9.44 MP**).
//!
//! Neither is the "64k+32 partial-superblock column" the issue guessed at: the
//! cases below include SB-EXACT frames on both sides of each predicate, so the
//! governing property is separable from frame alignment by the result pattern.
//!
//! The fix is composition, not new machinery: the per-tile `pack_tile` walk with
//! tile-bound isolation was already byte-proven by
//! `aom-encode/tests/encoder_gate_multitile.rs`, and the derived multi-tile
//! header + tile-group assembly by
//! `aom-encode/tests/obu_assemble_multitile_diff.rs`. Composing them exposed a
//! SECOND, deeper root — a real frame-header PARSE defect, see
//! `read_tile_info_max_tile`'s doc comment + `rb_diff::read_tile_info_inverts_write`.
//!
//! Run the on-demand tier:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb31_mandatory_tiles -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// Mirror-tile (the `kb22_hd_arms::mirror_tile` / `s4cov_hd_speed_axis` recipe).
fn mirror_tile(
    base: &EncodeCell,
    label: &str,
    w: usize,
    h: usize,
    cq: i32,
    speed: i32,
) -> EncodeCell {
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

/// The tile grid REAL aomenc coded, read back off the reference stream itself
/// (playbook §8: coverage from artefacts, not from names/derivations). Returns
/// `(tile_cols, tile_rows)`.
fn coded_tile_grid(stream: &[u8]) -> (usize, usize) {
    let (_t, _cfg, p) = aom_decode::frame::decode_frame_obus_prefilter(stream)
        .expect("port prefilter decode of the reference stream");
    (p.tile_info.cols, p.tile_info.rows)
}

/// `(w, h, expected tile cols, expected tile rows, why)`.
///
/// MEASURED 2026-08-01 against real `aomenc --allintra` (`c_encode_defaults`),
/// aarch64-apple-darwin. The single-tile rows are the negative controls that make
/// the predicate separable from frame alignment: 4032x64 and 4096x64 are BOTH
/// exact multiples of 64 px, and only the second one splits.
const GRID: &[(usize, usize, usize, usize, &str)] = &[
    (4032, 64, 1, 1, "63 SB cols — below the width predicate"),
    (
        4096,
        64,
        2,
        1,
        "64 SB cols == max_width_sb — set_tile_info's `<=` loop splits here, where\n         av1_get_tile_limits' `<` bound alone would not",
    ),
    (4160, 64, 2, 1, "65 SB cols — av1_get_tile_limits splits too"),
    (4160, 128, 2, 1, "the same split with more than one SB row"),
];

/// **The gate, default tier.** Every cell must (a) really be the tile grid the
/// table claims, read off the reference stream, and (b) encode BYTE-IDENTICALLY
/// to real aomenc through `EncodeCell::port_encode`.
///
/// These are the SMALLEST frames that reproduce issue #6 — 4096x64 is 0.26 MP, so
/// the whole gate runs in well under a second and stays in the default tier
/// rather than the `--ignored` one. `--cpu-used=9` keeps each cell cheap and is
/// also one of the three speeds the issue reported; the speed sweep below covers
/// the rest.
#[test]
fn mandatory_tile_split_encodes_byte_identical() {
    c::ref_init();
    let base = EncodeCell::real_content("kb31", "av1-1-b8-00-quantizer-00", None, 30, 9);
    let mut multi = 0usize;
    let mut single = 0usize;
    for &(w, h, want_cols, want_rows, why) in GRID {
        let cell = mirror_tile(&base, &format!("kb31_{w}x{h}"), w, h, 30, 9);
        let c_tu = cell.c_encode_defaults();
        assert!(!c_tu.is_empty(), "{w}x{h}: C encode failed");
        let grid = coded_tile_grid(&c_tu);
        assert_eq!(
            grid,
            (want_cols, want_rows),
            "{w}x{h} ({why}): real aomenc coded a {}x{} tile grid, not the expected {want_cols}x{want_rows}",
            grid.0,
            grid.1
        );
        if want_cols * want_rows > 1 {
            multi += 1;
        } else {
            single += 1;
        }
        let ours = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
        let real = EncodeCell::frame_obu_payload(&c_tu);
        assert_eq!(
            ours.len() as i64 - real.len() as i64,
            0,
            "{w}x{h} ({why}, {}x{} tiles): port payload {} B vs real {} B",
            grid.0,
            grid.1,
            ours.len(),
            real.len()
        );
        assert_eq!(ours, real, "{w}x{h} ({why}): payload bytes differ");
    }
    // Non-vacuity (playbook §2): the gate must contain BOTH a frame that forces a
    // tile split and one that does not, or it proves nothing about the predicate.
    assert!(
        multi >= 3 && single >= 1,
        "the grid must straddle the width predicate (multi={multi}, single={single})"
    );
}

/// **The AREA predicate — and the only cells that give the frame-header PARSE fix
/// its teeth.** The width-predicate grid above cannot see the second KB-31 root:
/// those frames have `min_log2 == 0`, so `min_log2_rows` is 0 whether or not
/// `av1_calculate_tile_cols`' re-derivation runs. It bites only where BOTH
/// `min_log2 > 0` (area) AND `log2_cols > min_log2_cols` (width), and the cheapest
/// such frame is 4096x2368 — 64 SB columns (so `set_tile_info`'s `<=` loop raises
/// `log2_cols` to 1 while `av1_get_tile_limits` says `min_log2_cols == 0`) and 2368
/// superblocks (so `min_log2 == 1`).
///
/// Its 4032x2368 sibling is one SB column narrower, which keeps `log2_cols == 0`
/// and pushes the whole split onto the ROW axis instead — the only 1x2 grid in this
/// file, and the only cell that exercises a tile boundary running horizontally.
///
/// `--cpu-used=7` because these frames are ~9.6 MP and KB-32 (below) is open at
/// speeds 8/9; at speed 7 both are byte-identical to real aomenc, which is the
/// strongest available statement.
///
/// **MEASURED 2026-08-01**: 4096x2368 -> 2x1 tiles, 987,642 B both sides;
/// 4032x2368 -> 1x2 tiles, 970,415 B both sides; ~3.6 s per cell.
///
/// Bite proof for the parse root: reverting `read_tile_info_max_tile`'s
/// `t.min_log2_rows = (min_log2 - t.log2_cols).max(0)` makes the 4096x2368 cell
/// read a 2x2 grid and desync `context_update_tile_id` by one bit, coding ~6% of
/// the expected payload.
#[test]
#[ignore = "2 x ~9.6 MP encode pairs at cpu7 (~8 s); on-demand tier"]
fn area_forced_tile_split_byte_identical() {
    c::ref_init();
    let base = EncodeCell::real_content("kb31a", "av1-1-b8-00-quantizer-00", None, 30, 7);
    // (w, h, tile cols, tile rows, why)
    const CELLS: &[(usize, usize, usize, usize, &str)] = &[
        (
            4096,
            2368,
            2,
            1,
            "area AND width predicates both fire: log2_cols(1) > min_log2_cols(0) with \
             min_log2 == 1 — the parse re-derivation's only reachable cell",
        ),
        (
            4032,
            2368,
            1,
            2,
            "area predicate alone (63 SB cols): the split lands on the ROW axis",
        ),
    ];
    let mut saw_row_split = false;
    for &(w, h, want_cols, want_rows, why) in CELLS {
        let cell = mirror_tile(&base, &format!("kb31a_{w}x{h}"), w, h, 30, 7);
        let c_tu = cell.c_encode_defaults();
        let grid = coded_tile_grid(&c_tu);
        assert_eq!(
            grid,
            (want_cols, want_rows),
            "{w}x{h} ({why}): real aomenc coded {}x{} tiles",
            grid.0,
            grid.1
        );
        saw_row_split |= want_rows > 1;
        let ours = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
        let real = EncodeCell::frame_obu_payload(&c_tu);
        println!(
            "  {w}x{h} ({}x{} tiles): {} B vs {} B, delta {:+}",
            grid.0,
            grid.1,
            ours.len(),
            real.len(),
            ours.len() as i64 - real.len() as i64
        );
        assert_eq!(ours, real, "{w}x{h} ({why}): payload bytes differ");
    }
    assert!(
        saw_row_split,
        "this gate must contain a tile-ROW split or the row axis is untested"
    );
}

/// **The speed axis, with a paired single-tile control at every speed.** The panic
/// fired at every `--cpu-used` (the refused assert sat before any speed-dependent
/// code), so a fix that only worked at one speed would be a false close. 4160x64
/// (2x1 tiles) at every speed 0..=9, and 4032x64 (1x1, the same content, 1 SB
/// column narrower) as the control — **a divergence that shows up on BOTH is not
/// the tile machinery's**, and that is what separates the two roots here.
///
/// **MEASURED 2026-08-01** (aarch64-apple-darwin, `--profile test-fast`,
/// `c_encode_defaults` bootstrap, cq30): byte-identical on **18 of 20** cells —
/// every speed 0..7 and 9, tiled and control alike. The two open cells are
/// `cpu8`, and they are open **on the control too** (-59 B single-tile, -134 B
/// tiled), so they belong to KB-12's pinned speed-8 estimate-arm class, not to
/// tiles. **Re-verified unchanged after KB-32 closed both size-band roots** —
/// as predicted, because these cells are 64 px on the short side and KB-32's
/// speed feature needs 720. What survives here is the leaf-mode class
/// `kb32_nonrd_size_bands::estimate_arm_residual_is_a_leaf_mode_near_tie`
/// localizes. Independently confirmed by a same-binary A/B: the single-tile control's
/// cpu8 payload is BYTE-IDENTICAL (13,224 B, -59) before and after the KB-31
/// landing, as are 1024², 2048² and 3072² at cpu8/cpu9 — see KB-32 for that
/// separate, pre-existing large-frame speed-8/9 divergence.
///
/// Pinned in BOTH directions (playbook §5): if a byte-exact cell regresses this
/// fails, and if `cpu8` silently starts matching it also fails and asks for a
/// re-pin.
#[test]
#[ignore = "20 encode pairs incl. speed 0; on-demand tier (~97 s)"]
fn mandatory_tile_split_byte_identical_across_speeds() {
    c::ref_init();
    let base = EncodeCell::real_content("kb31s", "av1-1-b8-00-quantizer-00", None, 30, 0);
    /// `(w, speed)` of the cells that are NOT byte-identical today. Both are
    /// `cpu8`, on the tiled cell AND its single-tile control — KB-12's class.
    const OPEN: &[(usize, i32)] = &[(4032, 8), (4160, 8)];
    let mut observed: Vec<(usize, i32)> = Vec::new();
    for speed in 0..=9 {
        for &(w, h, tiles) in &[(4032usize, 64usize, "1x1"), (4160, 64, "2x1")] {
            let cell = mirror_tile(&base, &format!("kb31_{w}x{h}_s{speed}"), w, h, 30, speed);
            let c_tu = cell.c_encode_defaults();
            let grid = coded_tile_grid(&c_tu);
            let got = format!("{}x{}", grid.0, grid.1);
            assert_eq!(got, tiles, "{w}x{h} cpu{speed}: coded tile grid");
            let ours = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
            let real = EncodeCell::frame_obu_payload(&c_tu);
            let delta = ours.len() as i64 - real.len() as i64;
            println!(
                "  {w}x{h} ({tiles} tiles) cpu{speed}: delta {delta:+} {}",
                if ours == real { "MATCH" } else { "DIVERGE" }
            );
            if ours != real {
                observed.push((w, speed));
            }
        }
    }
    assert_eq!(
        observed,
        OPEN.to_vec(),
        "the open set moved. Fewer entries => a root closed: re-pin OPEN (and say which \
         KB closed it). More entries => a REGRESSION. Note both members of OPEN are cpu8 \
         and appear on the SINGLE-TILE control as well, so they are KB-12's, not KB-31's."
    );
    // Non-vacuity: cpu8 aside, both the tiled cell and its control must have been
    // byte-exact somewhere, or the pin is asserting nothing about tiles.
    assert!(
        observed.len() < 20,
        "every cell diverged — the gate proves nothing"
    );
}

/// **The two frame sizes issue #6 actually reported**, at the cheapest of its three
/// speeds. Both used to exit 101; the contract asserted here is that they ENCODE —
/// producing the tile grid real aomenc produced, at the payload length real aomenc
/// produced ±the KB-32 residual — not that they are byte-identical, because they
/// are not yet: KB-32 (large frames at `--cpu-used` 8/9) is open and is measurably
/// NOT a tile effect (see the speed-axis test above).
///
/// **MEASURED 2026-08-01** (aarch64-apple-darwin, `--profile test-fast`, cq30,
/// `--cpu-used=9`, `c_encode_defaults`), before and after KB-32's two roots:
///
/// | cell | tiles | real aomenc | port BEFORE KB-32 | port AFTER |
/// |---|---|---|---|---|
/// | 5472x3648 (20 MP) | 2x2 | 2,121,452 B | 2,124,645 (+3,193, +0.15%) | 2,121,791 (**+339, +0.016%**) |
/// | 12000x9000 (108 MP) | 4x4 | 11,520,317 B | 11,548,497 (+28,180, +0.24%) | **refuses — see below** |
///
/// Wall/RSS for the pair: 1.6 s / 0.79 GB and 8.5 s / 3.11 GB. Neither OOMs on a
/// 64 GB box; 108 MP is nonetheless the memory ceiling anyone should expect from
/// this harness (source + SB-aligned strided copy + full reconstruction, all at
/// u16, plus libaom's own frame buffers).
///
/// **The 108 MP cell now REFUSES rather than encoding, and that is a real
/// consequence of KB-32, recorded honestly rather than smoothed over.** Closing
/// the variance-partition thresholds legitimately ENLARGES them, which lets
/// `set_vt_partitioning`'s HORZ/VERT pair arms win on this extremely smooth
/// mirror-tiled content — and the nonrd estimate arm cannot code a non-square
/// leaf yet (`nonrd_pickmode::nonrd_leaf_tx_size`'s HANDOFF has the precise
/// scope and the fix). The pre-KB-32 stream for this cell was 0.24% wrong; it is
/// now a loud, named refusal instead. **Measured reachability: of 18 large cells
/// probed at speeds 8 and 9 (768² through 5472x3648) NONE reach a non-square
/// leaf — this cell is the only one in the tree that does.**
///
/// Pinned as a verdict plus bounds, not as byte counts: byte counts would fire on
/// every unrelated encoder landing, while these fire exactly when the tile panic
/// comes back, when the 20 MP cell drifts, or when the non-square arm lands.
#[test]
#[ignore = "108 MP encode: ~11 s and ~3.1 GB peak RSS; on-demand tier"]
fn issue6_reported_sizes_encode() {
    c::ref_init();
    let base = EncodeCell::real_content("kb31i", "av1-1-b8-00-quantizer-00", None, 30, 9);
    // (w, h, tile cols, tile rows) — the geometry real aomenc codes.
    const CELLS: &[(usize, usize, usize, usize)] = &[(5472, 3648, 2, 2), (12000, 9000, 4, 4)];
    for &(w, h, want_cols, want_rows) in CELLS {
        let cell = mirror_tile(&base, &format!("kb31_issue6_{w}x{h}"), w, h, 30, 9);
        let c_tu = cell.c_encode_defaults();
        assert_eq!(
            coded_tile_grid(&c_tu),
            (want_cols, want_rows),
            "{w}x{h}: real aomenc tile grid"
        );
        let real = EncodeCell::frame_obu_payload(&c_tu);
        // The panic this file exists for was an unwind, so catching it here is how
        // the test reports "still broken" instead of aborting the run.
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cell.port_encode_with(&c_tu, &ToggleKnobs::default())
        }));
        let ours = match got {
            Ok(p) => p,
            Err(e) => {
                // Exactly ONE refusal is expected, and only for the reason named
                // above. Anything else — including issue #6's tile panic — fails.
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_default();
                assert!(
                    (w, h) == (12000, 9000) && msg.contains("non-square leaf bsize"),
                    "{w}x{h} ({want_cols}x{want_rows} tiles): port_encode PANICKED \
                     with {msg:?}. The only sanctioned refusal here is 12000x9000's \
                     non-square nonrd leaf (KB-32); a tile panic means issue #6 is back."
                );
                println!(
                    "  {w}x{h} ({want_cols}x{want_rows} tiles): REFUSED (KB-32 \
                     non-square nonrd leaf) — {msg}"
                );
                continue;
            }
        };
        assert_ne!(
            (w, h),
            (12000, 9000),
            "12000x9000 now ENCODES — the non-square nonrd estimate arm landed. \
             Re-pin: assert its byte delta instead of this refusal."
        );
        let delta = ours.len() as i64 - real.len() as i64;
        println!("  {w}x{h} ({want_cols}x{want_rows} tiles): {} B vs {} B, delta {delta:+}", ours.len(), real.len());
        // KB-32's two roots took this from +3,193 (0.15%) to +339 (0.016%); what
        // is left is the estimate-arm leaf-mode class
        // (`kb32_nonrd_size_bands::estimate_arm_residual_is_a_leaf_mode_near_tie`).
        // The bound is one order of magnitude tighter than the pre-KB-32 0.01,
        // so a re-opened size band cannot hide under it — while the pre-KB-31
        // header desync (6% of the expected payload) is still caught.
        let frac = (delta.abs() as f64) / (real.len() as f64);
        assert!(
            frac < 0.001,
            "{w}x{h}: payload is {:.3}% off real aomenc — that is above KB-32's \
             post-fix residual (0.016%); a size-scaling root or the tile walk is wrong",
            frac * 100.0
        );
        assert_ne!(
            ours, real,
            "{w}x{h} is now BYTE-IDENTICAL — the KB-32 estimate-arm residual \
             closed. Promote this to a hard byte gate and re-pin the table in \
             this doc comment."
        );
    }
}
