//! **KB-31 residual (a) — per-SB delta-q (`--deltaq-mode` 2/3, and the
//! `--delta-lf-mode=1` that rides on it) on a MULTI-TILE frame.**
//!
//! `EncodeCell::port_encode` REFUSED the combination outright:
//!
//! ```text
//! multi-tile x per-SB delta-q is unmodelled (the running qindex base resets
//! per tile in pack_tile but not in this harness's frame-raster replay) — see KB-31
//! ```
//!
//! # What C actually does (the refusal's diagnosis had it backwards)
//!
//! The refusal named the right two walks and drew the wrong conclusion about
//! which one was wrong. `pack_tile`'s per-tile restart is exactly C's behaviour;
//! `xd->current_base_qindex` is re-seeded from the frame `base_qindex` at the top
//! of **every tile**, on every side of the codec:
//!
//! | side | site | reset scope |
//! |---|---|---|
//! | search | `encode_sb_row`, *"Reset delta for quantizer and loof filters at the beginning of every tile"*, `av1/encoder/encodeframe.c:1232-1239` | per tile (`mi_row == tile_info->mi_row_start`) |
//! | pack | `write_modes`, `av1/encoder/bitstream.c:1745-1751` | per tile |
//! | decode | `decodeframe.c:2948` (serial tile loop) / `:3023` (`tile_worker_hook_init`) | per tile |
//!
//! and then advances one superblock at a time, in TILE raster, via
//! `xd->current_base_qindex = mbmi->current_qindex` (`partition_search.c:1476`
//! search side, `bitstream.c:979` write side). The base matters because
//! `av1_adjust_q_from_delta_q_res(res, prev, curr)` deadzone-rounds each SB's own
//! qindex against it (`av1/encoder/rd.c:494-505`), so tile order changes the
//! coded qindexes, not merely how they are signalled.
//!
//! What was actually wrong was the HARNESS's own two **frame-raster** replays —
//! the ones that DERIVE `delta_q_present` (`td->deltaq_used`, `encodeframe.c:375`,
//! OR-reduced over tiles at `:1593`, folded into the header at
//! `bitstream.c:4286-4289`) and the per-SB `delta_lf_from_base`
//! (`encodeframe.c:380-398`) instead of reading either off the bootstrap. They
//! carried ONE base across the whole frame. They now walk tiles, with the same
//! per-tile reset: `replay_sb_qindex_tile_order` in `aom-bench/src/lib.rs`.
//!
//! **No decoder-side counterpart existed** (contrast KB-31 root #2, which was a
//! decoder bug found while fixing an encoder refusal): `aom-decode`'s `start_tile`
//! already re-seeds `current_base_qindex` and zeroes the delta-lf carries per
//! tile (`aom-decode/src/lib.rs:2153-2157`), mirroring `decodeframe.c:2948`. The
//! `multi_tile_deltaq_round_trips` leg below is the positive control for that.
//!
//! # Reachability (a predicate, not a cell count — playbook §9)
//!
//! Two independent halves, both measured by `deltaq_multitile_axis_is_reachable`:
//! the frame must be one libaom REQUIRES to split (`width >= 4033 px` via
//! `set_tile_info`'s `<=` column loop, `av1/encoder/encoder.c:385-390`, or
//! `> 2304 SB64s` ~ 9.44 MP via `av1_get_tile_limits`' area bound,
//! `av1/common/tile_common.c:31-50`), **or** the caller asks for tiles explicitly
//! (`AV1E_SET_TILE_COLUMNS` / `_ROWS`, which needs no size at all); AND
//! `--deltaq-mode` 2/3 must be set and must actually modulate. So: one or two
//! non-default flags, no size requirement in the explicit case — reachable, but
//! not by default.
//!
//! # Scope — MEASURED, and what is still open
//!
//! Every cell here is **speed 0**, bd8 4:2:0 SB64. That is not timidity, it is
//! where the delta-q modes themselves are byte-exact: the two pre-existing
//! delta-q gates (`deltaq_mode2_e2e`, `deltaq_mode3_e2e`) and `delta_lf_mode_e2e`
//! are speed-0-only, and sweeping speed here found delta-q diverging from real
//! aomenc **at one tile as well as many** from speed 1 up on some content — see
//! `DELTAQ_SPEED_OPEN` in CLAUDE.md. Gating the tile axis at a speed where the
//! single-tile baseline is itself divergent would prove nothing about tiles.

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

const AV1E_SET_TILE_COLUMNS: i32 = 33;
const AV1E_SET_TILE_ROWS: i32 = 34;
const AV1E_SET_DELTAQ_MODE: i32 = 107;
const AV1E_SET_DELTALF_MODE: i32 = 108;

/// Mirror-tile a small real-content cell up to `(w, h)` — the
/// `kb31_mandatory_tiles::mirror_tile` / `kb22_hd_arms::mirror_tile` recipe.
/// Mirroring (rather than tiling) keeps the content continuous across the seam,
/// so the per-SB variance the delta-q modes read is real image statistics rather
/// than a periodic step function.
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

/// What the REFERENCE stream actually coded, read back off it rather than
/// derived (playbook §8): `(tile_cols, tile_rows, delta_q_present, delta_q_res)`.
fn coded_facts(stream: &[u8]) -> (usize, usize, bool, i32) {
    let (_t, _cfg, p) = aom_decode::frame::decode_frame_obus_prefilter(stream)
        .expect("port prefilter decode of the reference stream");
    (
        p.tile_info.cols,
        p.tile_info.rows,
        p.delta_q.delta_q_present,
        p.delta_q.delta_q_res,
    )
}

/// `mode` is the `--deltaq-mode` value (0 = the delta-q-OFF control); `dlf` adds
/// `--delta-lf-mode=1`, which only does anything when a delta-q mode is firing
/// (`enable_deltalf_mode`, av1_cx_iface.c).
fn knobs_for(mode: i32, dlf: bool) -> ToggleKnobs {
    ToggleKnobs {
        deltaq_mode3: mode == 3,
        deltaq_mode2: mode == 2,
        delta_lf_mode: dlf && mode != 0,
        ..Default::default()
    }
}

struct Row {
    label: String,
    tiles: (usize, usize),
    dq_present: bool,
    port_len: usize,
    real_len: usize,
    first_diff: Option<usize>,
}

impl Row {
    fn is_multi(&self) -> bool {
        self.tiles.0 * self.tiles.1 > 1
    }
}

/// Encode one cell both ways and compare the assembled frame-OBU payloads.
/// `grid` is `Some((log2_cols, log2_rows))` for an explicitly-tiled encode, or
/// `None` to let the frame SIZE decide (the mandatory-split path).
fn run_cell(cell: &EncodeCell, mode: i32, dlf: bool, grid: Option<(i32, i32)>) -> Row {
    let mut ctrls: Vec<(i32, i32)> = Vec::new();
    if let Some((lc, lr)) = grid {
        ctrls.push((AV1E_SET_TILE_COLUMNS, lc));
        ctrls.push((AV1E_SET_TILE_ROWS, lr));
    }
    if mode != 0 {
        ctrls.push((AV1E_SET_DELTAQ_MODE, mode));
        if dlf {
            ctrls.push((AV1E_SET_DELTALF_MODE, 1));
        }
    }
    let c_tu = cell.c_encode_ctrls(&ctrls);
    assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
    let (tc, tr, dqp, _res) = coded_facts(&c_tu);
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let ours = cell.port_encode_with(&c_tu, &knobs_for(mode, dlf));
    let first_diff = if ours == real {
        None
    } else {
        Some(
            ours.iter()
                .zip(real.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(ours.len().min(real.len())),
        )
    };
    Row {
        label: cell.label.clone(),
        tiles: (tc, tr),
        dq_present: dqp,
        port_len: ours.len(),
        real_len: real.len(),
        first_diff,
    }
}

fn report(rows: &[Row]) -> String {
    let mut s = String::new();
    for r in rows {
        s.push_str(&format!(
            "  {:<26} {}x{} tiles  dq_present={:<5} port {:>9} B  real {:>9} B  {}\n",
            r.label,
            r.tiles.0,
            r.tiles.1,
            r.dq_present,
            r.port_len,
            r.real_len,
            match r.first_diff {
                None => "MATCH".to_string(),
                Some(i) => format!("MISMATCH @ byte {i}"),
            }
        ));
    }
    s
}

/// Shared tail: print, assert the modes really fired, assert the grid straddles
/// single/multi tile, then assert byte-identity.
fn finish(what: &str, rows: &[Row], min_multi: usize, min_single: usize) {
    eprintln!("{what}:\n{}", report(rows));
    for r in rows {
        if !r.label.starts_with("dq0") {
            assert!(
                r.dq_present,
                "{}: the reference header carries no delta_q_present — the mode never \
                 modulated on this cell, so it tests nothing",
                r.label
            );
        }
    }
    let multi = rows.iter().filter(|r| r.is_multi()).count();
    let single = rows.len() - multi;
    assert!(
        multi >= min_multi && single >= min_single,
        "{what}: the cell set must straddle single- and multi-tile \
         (multi={multi}, single={single})"
    );
    assert!(
        rows.iter().all(|r| r.first_diff.is_none()),
        "{what}: not every cell byte-matches real aomenc:\n{}",
        report(rows)
    );
}

/// **The refusal's own cells.** `--deltaq-mode` 2 AND 3 on the SMALLEST frame
/// whose SIZE forces a tile split (4096x64 = 0.26 MP, 64 SB columns -> 2x1
/// tiles), plus both controls that make the two axes separable:
///
/// * a **single-tile control at the same deltaq mode** (4032x64, one SB column
///   narrower, 1x1 tiles) — if this passed while the 4096 cell failed, the tile
///   axis is what moved;
/// * a **multi-tile control at deltaq 0** (4096x64, no `--deltaq-mode`) — if this
///   passed while the mode-2/3 cells failed, the deltaq axis is what moved.
#[test]
fn deltaq_multitile_byte_identical() {
    c::ref_init();
    const SPEED: i32 = 0;
    const CQ: i32 = 32;
    let base = EncodeCell::real_content("kb31dq", "av1-1-b8-00-quantizer-00", None, CQ, SPEED);
    let mut rows = Vec::new();
    for mode in [3, 2] {
        // 4032 = 63 SB columns (1 tile); 4096 = 64 (2 tiles).
        for w in [4096usize, 4032] {
            let cell = mirror_tile(&base, &format!("dq{mode}_{w}x64"), w, 64, CQ, SPEED);
            rows.push(run_cell(&cell, mode, false, None));
        }
    }
    let plain = mirror_tile(&base, "dq0_4096x64", 4096, 64, CQ, SPEED);
    rows.push(run_cell(&plain, 0, false, None));
    finish("size-forced tile split x --deltaq-mode", &rows, 3, 2);
}

/// **The tile GRID matrix**, driven by `AV1E_SET_TILE_COLUMNS`/`_ROWS` so a tile
/// ROW boundary — the reset position a column split never produces (tile 1 starts
/// at `mi_row > 0, mi_col == 0`) — and a 2x2 grid are reachable at a cheap size
/// instead of at the 9.44 MP the AREA predicate would demand.
///
/// The `dq0` rows are the tile-axis control: delta-q OFF at the same four grids.
/// If those pass and a `dq2`/`dq3` row fails, the failure is delta-q's, not the
/// tile walk's.
#[test]
fn deltaq_tile_grid_matrix_byte_identical() {
    c::ref_init();
    const SPEED: i32 = 0;
    // 192x192 = 3x3 SB64s: asking for 2 tile columns yields an UNEVEN 2+1 split,
    // which is the interesting case (a tile whose SB count differs from its
    // neighbour's). 4 tile columns would exceed the 3 available, which libaom
    // clamps to a 3x3 grid the harness's uniform-spacing header check rejects —
    // out of scope here, and unrelated to delta-q.
    const GRIDS: &[(i32, i32)] = &[(0, 0), (1, 0), (0, 1), (1, 1)];
    let mut rows = Vec::new();
    for cq in [12i32, 32] {
        for mode in [0, 3, 2] {
            for &(lc, lr) in GRIDS {
                let cell = EncodeCell::real_content(
                    &format!("dq{mode}_cq{cq}_t{lc}{lr}"),
                    "av1-1-b8-00-quantizer-00",
                    Some((192, 192, 0, 0)),
                    cq,
                    SPEED,
                );
                rows.push(run_cell(&cell, mode, false, Some((lc, lr))));
            }
        }
    }
    finish("explicit tile grid x --deltaq-mode", &rows, 18, 6);
}

/// **`--delta-lf-mode=1` across tile grids** — the only leg where the replay's
/// output can reach the BYTES at all. `delta_q_present` is one frame bit that the
/// harness cross-checks against the real header, so a mis-ordered replay fails
/// loudly there rather than mis-coding; the per-SB `delta_lf_from_base` values it
/// also feeds are per-superblock and reach the stream through
/// `stamp_lf_delta_lf` -> the LF grid -> `get_filter_level` -> the picked frame
/// filter level.
///
/// **Honest negative, MEASURED 2026-08-04:** restoring the pre-fix frame-raster
/// replay (one running base for the whole frame) does NOT fail these cells — all
/// 24 still byte-match, as do `delta_lf_mode_e2e` / `deltaq_mode2_e2e` /
/// `deltaq_mode3_e2e`. The filter-level pick absorbs the delta_lf differences at
/// these sizes. The bite for the reordering is at unit level instead
/// (`aom-bench/src/lib.rs`'s `replay_resets_the_running_base_at_every_tile`);
/// this file's job is to hold the axis the refusal used to block. See KB-39.
#[test]
fn delta_lf_multitile_byte_identical() {
    c::ref_init();
    const SPEED: i32 = 0;
    const GRIDS: &[(i32, i32)] = &[(0, 0), (1, 0), (0, 1), (1, 1)];
    let mut rows = Vec::new();
    for cq in [12i32, 32, 48] {
        for mode in [3, 2] {
            for &(lc, lr) in GRIDS {
                let cell = EncodeCell::real_content(
                    &format!("dlf{mode}_cq{cq}_t{lc}{lr}"),
                    "av1-1-b8-00-quantizer-00",
                    Some((192, 192, 0, 0)),
                    cq,
                    SPEED,
                );
                let mut row = run_cell(&cell, mode, true, Some((lc, lr)));
                // cq48 mode-3 is below this content's modulation threshold: the
                // reference header drops delta_q_present, so delta-lf cannot ride
                // and the row is a (still useful) negative control.
                if !row.dq_present {
                    row.label = format!("dq0-{}", row.label);
                }
                rows.push(row);
            }
        }
    }
    finish("explicit tile grid x --delta-lf-mode=1", &rows, 18, 6);
}

/// **The port's own decoder on the multi-tile delta-q stream.** Byte-identity
/// above proves the port ENCODES what libaom encodes; this proves the port
/// DECODES it, which is the half KB-31 root #2 turned out to live in. No
/// pre-existing decode fixture reaches it: `armed_tools_decode_gate`'s delta-q
/// arms are 196x196 single-tile, and every multi-tile decode fixture in the tree
/// has delta-q off.
#[test]
fn multi_tile_deltaq_round_trips() {
    c::ref_init();
    const SPEED: i32 = 0;
    let mut checked = 0usize;
    for mode in [3, 2] {
        for &(lc, lr) in &[(1i32, 0i32), (0, 1), (1, 1)] {
            let cell = EncodeCell::real_content(
                &format!("rt{mode}_t{lc}{lr}"),
                "av1-1-b8-00-quantizer-00",
                Some((192, 192, 0, 0)),
                32,
                SPEED,
            );
            let c_tu = cell.c_encode_ctrls(&[
                (AV1E_SET_TILE_COLUMNS, lc),
                (AV1E_SET_TILE_ROWS, lr),
                (AV1E_SET_DELTAQ_MODE, mode),
                (AV1E_SET_DELTALF_MODE, 1),
            ]);
            let (tc, tr, dqp, _) = coded_facts(&c_tu);
            assert!(tc * tr > 1 && dqp, "{}: not a multi-tile delta-q stream", cell.label);
            // The port's frame OBU is byte-identical to the reference (the gates
            // above), so decoding the reference stream decodes the port's bytes.
            let dec = aom_decode::frame::decode_frame_obus(&c_tu)
                .unwrap_or_else(|e| panic!("{}: port decode of a {tc}x{tr}-tile delta-q stream failed: {e:?}", cell.label));
            assert_eq!(
                (dec.width, dec.height),
                (cell.w, cell.h),
                "{}: decoded dimensions",
                cell.label
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 6, "every multi-tile delta-q round-trip cell must run");
}

/// **The AREA predicate, at the size that forces it.** 4032x2368 is 63 SB columns
/// (so the width predicate does NOT fire) and 2,368 SB64s (so the AREA one does),
/// which puts the whole split on the ROW axis — a 1x2 grid at 9.55 MP, and the
/// only delta-q cell in the tree where `min_log2 > 0`, i.e. the header shape
/// KB-31 root #2 lived in. The `dq0` row is the tile-axis control at the same
/// size and speed.
///
/// Speed 0 because that is where delta-q is byte-exact at all (see the module
/// header); a ~9.55 MP speed-0 encode is ~225 s port-side, hence the on-demand
/// tier. MEASURED 2026-08-04 (aarch64-apple-darwin, `test-fast`): 3/3
/// byte-identical — dq0 1,005,925 B, dq3 1,018,101 B, dq2 1,739,509 B — 701.6 s
/// for the test.
#[test]
#[ignore = "3 x ~9.55 MP speed-0 encode pairs (~225 s each); on-demand tier"]
fn deltaq_area_forced_row_split_byte_identical() {
    c::ref_init();
    const SPEED: i32 = 0;
    const CQ: i32 = 30;
    let base = EncodeCell::real_content("kb31dqa", "av1-1-b8-00-quantizer-00", None, CQ, SPEED);
    let mut rows = Vec::new();
    for mode in [0, 3, 2] {
        let cell = mirror_tile(&base, &format!("dq{mode}_4032x2368"), 4032, 2368, CQ, SPEED);
        rows.push(run_cell(&cell, mode, false, None));
    }
    assert!(
        rows.iter().all(|r| r.tiles == (1, 2)),
        "every cell must be the 1x2 (ROW-split) grid the area predicate forces:\n{}",
        report(&rows)
    );
    finish("area-forced tile ROW split x --deltaq-mode", &rows, 3, 0);
}

/// **Reachability, stated as a predicate rather than a cell count** (playbook §9),
/// for the SIZE-forced half. Passes before the fix as well as after — it asks the
/// reference encoder what it does, not the port.
#[test]
fn deltaq_multitile_axis_is_reachable() {
    c::ref_init();
    let base = EncodeCell::real_content("kb31dqr", "av1-1-b8-00-quantizer-00", None, 32, 0);
    let split = mirror_tile(&base, "reach_4096x64", 4096, 64, 32, 0);
    let (tc, tr, dqp, res) = coded_facts(&split.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, 3)]));
    assert!(
        tc * tr > 1,
        "4096x64 must force a tile split (got {tc}x{tr}) — the width predicate moved"
    );
    assert!(
        dqp,
        "--deltaq-mode=3 must modulate on 4096x64 mirror content, else the axis is \
         unreachable from this corpus"
    );
    assert_eq!(res, 4, "DELTA_Q_RES_PERCEPTUAL");
    // One SB column narrower is a single tile with the same flag firing — the two
    // halves of the predicate are independent.
    let no_split = mirror_tile(&base, "reach_4032x64", 4032, 64, 32, 0);
    let (tc2, tr2, dqp2, _) = coded_facts(&no_split.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, 3)]));
    assert_eq!((tc2, tr2), (1, 1), "4032x64 must stay single-tile");
    assert!(dqp2, "--deltaq-mode=3 must modulate on 4032x64 too");
}
