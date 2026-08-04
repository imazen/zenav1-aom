//! **KB-35 — the nonrd estimate arm's palette refusal was the FRAME flag, not
//! C's `try_palette`, so `--cpu-used 8` refused ordinary images.**
//!
//! > **Header correction 2026-08-03.** This paragraph opened *"`av1_search_palette_mode_luma`
//! > (intra_mode_search.c:1122) is not ported for the nonrd estimate arm, so the port refuses
//! > rather than silently coding a different winner."* **KB-37 ported it the same day** —
//! > there is no refusal on this arm any more, and this file's own §2
//! > (`nonrd_palette_arm_reaching_set`) says so. What KB-35 fixed is described below and is
//! > still accurate; only the "not ported / refuses" framing was overtaken. The live
//! > residual is a byte divergence, `PALETTE_ON_SPEED8_OPEN`, not a refusal.
//!
//! Historically: the refusal (when it existed) fired on
//! `cm->features.allow_screen_content_tools` alone — **one of four terms** of
//! C's `try_palette` (nonrd_pickmode.c:1698-1710). The other three are
//! `cpi->oxcf.tool_cfg.enable_palette`, the ordinal size bounds of
//! `av1_allow_palette`, and (at `prune_palette_search_nonrd > 0`, which is 1 at
//! every speed that dispatches this arm) `bsize <= BLOCK_16X16 &&
//! source_variance > 200`.
//!
//! The consequence was a hard failure on a default-reachable configuration:
//! **136 of the 2,012 rows of `benchmarks/nonsquare_leaf_reach_2026-08-02.tsv`
//! PANICked** — a plain smooth gradient at `--cpu-used 8`, at every size from
//! 1024x1024 up and every quantizer from cq2 to cq63 — on cells where the C
//! oracle passes `--enable-palette=0` (`shim_encode_av1_kf`, dec_shim.c:614)
//! and therefore provably never enters the palette search at all.
//!
//! **Why it looked like a speed axis and was not.** Speed 9 never refused, so
//! the shape read as "speed 8 only". The cause is unrelated to the palette:
//! `av1_set_screen_content_options` (encoder.c:2466-2470) turns screen-content
//! DETECTION off when `use_nonrd_pick_mode && !hybrid_intra_pickmode`, which is
//! the speed-9 combination (`hybrid_intra_pickmode = 0`, speed_features.c:1795)
//! but not the speed-8 one (`= 2`, :578). So at speed 9 the frame flag is 0 and
//! the old guard was vacuously satisfied.
//!
//! **What this file measures** (record `benchmarks/kb35_palette_arm_2026-08-03.tsv`):
//!
//! | class | rows | pristine | fixed |
//! |---|---|---|---|
//! | smooth gradient, cpu8, screen flag ON, palette OFF | 8 | PANIC | **MATCH** |
//! | screen content, cpu8, palette OFF both sides | 9 | PANIC | **MATCH** |
//! | smooth gradient <= 512x512, cpu8 (flag off) | 8 | MATCH | MATCH |
//! | screen content, cpu8, palette ON both sides | 9 | PANIC | **DIVERGE** (pinned) |
//! | screen content 1024x1024, cq >= 60, cpu8, palette ON | 3 | PANIC | PANIC (**genuine**) |
//!
//! The last row is the teeth on the narrowing: the refusal still fires, on C's
//! own predicate, at `bsize == BLOCK_16X16` with `source_variance = 3140`. The
//! fourth row is a divergence class the overbroad refusal had been HIDING — it
//! is not this arm (`palette_gate_reach()[2]` reads 0 on every one of those
//! cells, i.e. the estimate arm never wanted a palette there), so it is pinned
//! rather than attributed.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb35_nonrd_palette_arm -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_dsp::entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader,
};
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_encode::nonrd_pickmode::{palette_gate_reach, reset_palette_gate_reach};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;

/// Parse `allow_screen_content_tools` out of a reference stream's headers
/// (sequence + uncompressed frame header), so the gate can assert the screen
/// detection actually fired on its cells.
fn stream_allow_screen_content(stream: &[u8]) -> bool {
    let mut pos = 0usize;
    let mut seq_payload: Option<&[u8]> = None;
    let mut frame_payload: Option<&[u8]> = None;
    while pos < stream.len() {
        let hdr = read_obu_header(&stream[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_dsp::entropy::leb128::uleb_decode(&stream[after_header..]).expect("leb128");
        let start = after_header + size_bytes;
        let end = start + size as usize;
        match hdr.obu_type {
            t if t == OBU_SEQUENCE_HEADER => seq_payload = Some(&stream[start..end]),
            t if t == OBU_FRAME => frame_payload = Some(&stream[start..end]),
            _ => {}
        }
        pos = end;
    }
    let seq_payload = seq_payload.expect("no sequence header OBU");
    let frame_payload = frame_payload.expect("no frame OBU");
    let mut rb = ReadBitBuffer::new(seq_payload);
    let seq = aom_dsp::entropy::header::read_sequence_header_obu(&mut rb);
    let s = &seq.seq_header;
    let cfg = FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length
                as u32,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            frame_id_length: s.frame_id_length as u32,
            force_screen_content_tools: s.force_screen_content_tools,
            force_integer_mv: s.force_integer_mv,
            max_frame_width: s.max_frame_width,
            max_frame_height: s.max_frame_height,
            enable_order_hint: s.enable_order_hint,
            order_hint_bits_minus_1: s.order_hint_bits_minus_1,
            operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
            operating_point_idc: seq.operating_point_idc,
            op_decoder_model_param_present: seq.op_decoder_model_param_present,
            buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length as u32,
            temporal_layer_id: 0,
            spatial_layer_id: 0,
            ..Default::default()
        },
        frame_size: FrameSizeHeader {
            num_bits_width: s.num_bits_width,
            num_bits_height: s.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            ..Default::default()
        },
        num_planes: if seq.color_config.monochrome { 1 } else { 3 },
        separate_uv_delta_q: seq.color_config.separate_uv_delta_q,
        loopfilter: LoopfilterHeader::default(),
        cdef: CdefHeader {
            enable_cdef: s.enable_cdef,
            ..Default::default()
        },
        restoration: RestorationHeader {
            enable_restoration: s.enable_restoration,
            sb_size_128: s.sb_size_128,
            subsampling_x: seq.color_config.subsampling_x,
            subsampling_y: seq.color_config.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    let mut rb = ReadBitBuffer::new(frame_payload);
    let p = aom_dsp::entropy::header::read_uncompressed_header(&mut rb, &cfg);
    p.prefix.allow_screen_content_tools
}


/// Few-colour "terminal text" luma (the `rd_close_palette` recipe, verbatim):
/// period-8 exact repeats + large flat runs, so libaom's ANTIALIASING_AWARE
/// screen-content detection fires.
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

fn screen_cell(label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for cc in 0..w {
            y[r * w + cc] = text_luma(r, cc);
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

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Match,
    Diverge,
    Panic,
}

/// Encode `cell` on both sides and classify, returning the verdict, the byte
/// delta, the panic's last line (empty when none), and this thread's
/// `palette_gate_reach()` for the run. A PANIC is a distinct outcome, not a
/// test abort — the whole point of this file is the shape of the refusal set.
fn measure(cell: &EncodeCell, c_tu: &[u8], knobs: &ToggleKnobs) -> (Verdict, i64, String, [u64; 3]) {
    let real = EncodeCell::frame_obu_payload(c_tu);
    let msg = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = std::sync::Arc::clone(&msg);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        *sink.lock().unwrap() = info.to_string().lines().last().unwrap_or("").to_string();
    }));
    reset_palette_gate_reach();
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(c_tu, knobs)
    }));
    std::panic::set_hook(hook);
    let reach = palette_gate_reach();
    let note = msg.lock().unwrap().clone();
    match got {
        Ok(p) if p == real => (Verdict::Match, 0, String::new(), reach),
        Ok(p) => (
            Verdict::Diverge,
            p.len() as i64 - real.len() as i64,
            String::new(),
            reach,
        ),
        Err(_) => (Verdict::Panic, 0, note, reach),
    }
}

// ---------------------------------------------------------------------------
// 1. The class the overbroad refusal was blocking.
// ---------------------------------------------------------------------------

/// **The KB-35 byte gate.** Every cell here PANICS on the pre-fix tree with
/// *"HANDOFF: av1_search_palette_mode_luma (palette.c) not ported — required
/// before any screen-content (allow_screen_content_tools=1) speed-8 cell"*,
/// and is byte-identical to real aomenc now.
///
/// Three sub-classes, deliberately interleaved so the result pattern decides
/// (playbook §1's asymmetry, and §9's "state a predicate, not a cell count"):
///
/// * **gradient, screen flag ON** — `synthetic_diag` at 1024x1024 and 1280x720.
///   libaom's own detector fires on a smooth gradient at these sizes. Palette
///   is OFF on both sides (`shim_encode_av1_kf` passes `--enable-palette=0`),
///   so C's `try_palette` is FALSE by its first term and the port must encode.
/// * **gradient, screen flag OFF** — the same content at 256x256 and 512x512,
///   where the detector does NOT fire. These MATCHED before the fix too; they
///   are the negative control proving the fix is not "everything passes now".
/// * **screen content, palette OFF** — text/UI content the detector certainly
///   flags, at three sizes and three quantizers, with `--enable-palette=0`.
///
/// Non-vacuity is derived from the artefacts, not from the labels: the
/// `flag_on` column is read out of the REAL stream's frame header, and the
/// grid must contain rows on both sides of it.
#[test]
#[ignore = "26 encode pairs up to 1280x720; nightly / on-demand tier"]
fn speed8_screen_detected_cells_byte_match() {
    c::ref_init();
    let mut rows: Vec<(String, bool, Verdict, i64)> = Vec::new();

    // (a) + (b): the smooth gradient, above and below the detector's reach.
    for &(w, h) in &[(256usize, 256usize), (512, 512), (1024, 1024), (1280, 720)] {
        for cq in [2, 24, 44, 63] {
            let cell = EncodeCell::synthetic_diag(&format!("diag{w}x{h}"), w, h, cq, 8);
            let c_tu = cell.c_encode();
            assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
            let flag_on = stream_allow_screen_content(&c_tu);
            let (v, d, note, reach) = measure(&cell, &c_tu, &ToggleKnobs::default());
            println!(
                "  diag {w}x{h} cq{cq} cpu8 (screen flag {}): {v:?} {d:+} reach {reach:?}{}",
                if flag_on { "ON " } else { "off" },
                if note.is_empty() {
                    String::new()
                } else {
                    format!("  [{note}]")
                }
            );
            rows.push((format!("diag {w}x{h} cq{cq}"), flag_on, v, d));
        }
    }

    // (c): real screen content, palette OFF on both sides.
    for &(w, h) in &[(128usize, 128usize), (256, 256), (512, 512)] {
        for cq in [12, 32, 50] {
            let cell = screen_cell(&format!("scr{w}x{h}"), w, h, cq, 8);
            let c_tu = cell.c_encode_screen(false, false);
            assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
            let flag_on = stream_allow_screen_content(&c_tu);
            let (v, d, note, reach) = measure(&cell, &c_tu, &ToggleKnobs::default());
            println!(
                "  scr  {w}x{h} cq{cq} cpu8 palette-OFF (screen flag {}): {v:?} {d:+} \
                 reach {reach:?}{}",
                if flag_on { "ON " } else { "off" },
                if note.is_empty() {
                    String::new()
                } else {
                    format!("  [{note}]")
                }
            );
            rows.push((format!("scr {w}x{h} cq{cq}"), flag_on, v, d));
        }
    }

    // Reach assertions (playbook §2/§8): the grid is worthless unless it
    // straddles the detector. Both sides must be present, and the ON side is
    // what the pre-fix tree refused.
    let on = rows.iter().filter(|r| r.1).count();
    let off = rows.len() - on;
    assert!(
        on >= 8 && off >= 4,
        "the grid must contain rows with the screen-content flag ON (the cells the old guard \
         refused) and rows with it off (the control): {on} on / {off} off"
    );
    let bad: Vec<String> = rows
        .iter()
        .filter(|r| r.2 != Verdict::Match)
        .map(|r| format!("{} {:?} ({:+})", r.0, r.2, r.3))
        .collect();
    assert!(
        bad.is_empty(),
        "a KB-35 cell stopped byte-matching. A PANIC naming `try_palette` means the refusal \
         widened again — check `nonrd_palette_arm_is_live` against nonrd_pickmode.c:1698-1710 \
         (the frame flag is ONE of four terms). A DIVERGE means the estimate arm's leaf pick \
         moved: {bad:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. The refusal that is REAL, and still loud.
// ---------------------------------------------------------------------------

/// **The refusal is GONE — `av1_search_palette_mode_luma` is ported (KB-37) —
/// so what this test now pins is the REACHING SET plus what happens there.**
///
/// KB-35's own closing sentence said: *"If `av1_search_palette_mode_luma` has
/// been PORTED for this arm, delete this test and promote the cells to byte
/// gates"*. Only half of that is possible, and the half that is not is worth
/// keeping visible.
///
/// * **Promoted**: the estimate arm's palette search is byte-identical to
///   aomenc on a 27-cell grid where the full-RD arm is provably unreachable —
///   `kb37_nonrd_palette_search.rs`. That is where the port is gated.
/// * **NOT promoted**: these cells. At `--cpu-used 8` the same frame reaches
///   BOTH palette searches, and the full-RD one at speed 8 is the open
///   divergence class `PALETTE_ON_SPEED8_OPEN` pinned below. A byte delta here
///   cannot be attributed to either arm, so promoting them would be pinning
///   someone else's bug to this arm's name.
///
/// What is still asserted, because it is what KB-35 established and it must
/// not silently move: the reaching set is **exactly `{cq60, cq62, cq63}`** at
/// 1024x1024 screen content (`set_vbp_thresholds` scales with qindex, so only
/// at a high quantizer does a 16x16 with `source_variance = 3140` survive the
/// variance split undivided — cq58 on the same content reaches ZERO such
/// leaves), the reach counter and the predicate agree on every row, and NO row
/// refuses any more.
#[test]
#[ignore = "8 encode pairs at 1024x1024; nightly / on-demand tier"]
fn estimate_arm_palette_search_reaching_set_is_pinned() {
    c::ref_init();
    let knobs = ToggleKnobs {
        enable_palette: true,
        ..Default::default()
    };
    let mut reaching: Vec<i32> = Vec::new();
    for cq in [40, 50, 55, 58, 60, 62, 63] {
        let cell = screen_cell("scr1024", 1024, 1024, cq, 8);
        let c_tu = cell.c_encode_screen(true, false);
        let (v, d, note, reach) = measure(&cell, &c_tu, &knobs);
        println!("  scr 1024x1024 cq{cq} cpu8 palette-ON: {v:?} {d:+} reach {reach:?}");
        assert_ne!(
            v,
            Verdict::Panic,
            "cq{cq} REFUSED. `av1_search_palette_mode_luma` is ported (KB-37), so a \
             refusal here is a NEW handoff, not this one: {note}"
        );
        assert!(
            reach[0] > 0,
            "cq{cq}: no estimate-arm leaf even reached `av1_allow_palette` — the frame's \
             screen-content flag is off or the arm was not dispatched, so this row says \
             nothing about the palette gate"
        );
        if reach[2] > 0 {
            reaching.push(cq);
        }
    }
    assert_eq!(
        reaching,
        vec![60, 62, 63],
        "the reaching set moved. FEWER entries means `nonrd_palette_arm_is_live` has been \
         narrowed past C's predicate (or the VBP thresholds moved); MORE means it widened. \
         Either way the KB-37 byte gate's coverage claim needs re-checking"
    );
}

// ---------------------------------------------------------------------------
// 3. What the refusal was hiding.
// ---------------------------------------------------------------------------

/// **`PALETTE_ON_SPEED8_OPEN` — a divergence class the overbroad refusal had
/// been masking, pinned exactly rather than smoothed over.**
///
/// With `--enable-palette=1` on both sides, screen-detected content at
/// `--cpu-used 8` diverges at every size and quantizer tried. It is **not**
/// the nonrd estimate arm: `palette_gate_reach()[2]` reads **0** on every one
/// of these cells, i.e. no estimate-arm leaf satisfies C's `try_palette`, and
/// the port never refuses. That leaves the FULL-RD leaf's palette search
/// (`av1_rd_pick_palette_intra_sby`, which `hybrid_use_rdopt` dispatches for
/// every `bsize < BLOCK_16X16` with `source_variance >= 101` at speed 8) as
/// the location — a speed-8 crossing of the palette search that no gate in
/// the tree covers (`rd_close_palette.rs` is speed 0 throughout).
///
/// **NARROWED 2026-08-03 by KB-37, with a positive control rather than an
/// absence.** The attribution above was "reach[2] reads 0, so not the estimate
/// arm" — an argument from a counter reading zero. `kb37_nonrd_palette_search.rs`
/// now encodes THE SAME text/UI content, at the same three sizes, with palette
/// on and the screen flag on, at `--cpu-used 9`, where `hybrid_intra_pickmode
/// = 0` makes the full-RD arm structurally unreachable — and it is
/// BYTE-IDENTICAL on all 15 of those rows.
///
/// **What that RULES OUT as a cause of the deltas below** — the durable result,
/// rather than "the class did not move":
///
/// * the CONTENT: the same pixels encode byte-exactly;
/// * the SHARED `palette_search` machinery: the same
///   `av1_rd_pick_palette_intra_sby`, the same colour cache / k-means / cost
///   tables, on the same blocks;
/// * the nonrd ESTIMATE arm's palette search: at speed 9 it is the only
///   palette producer there is;
/// * the screen-content frame flag and everything gated on it: it is ON in
///   both arms of the comparison.
///
/// What is left is the one thing `--cpu-used 8` adds over 9: the full-RD leaf
/// (`hybrid_use_rdopt` -> `av1_rd_pick_intra_mode_sb`) and its own palette
/// search. Caveat kept explicit: speed 9 needs `--tune-content=screen` to hold
/// the flag on, which also sets `allow_intrabc`, so the two configs are not
/// bit-for-bit matched. This narrows the suspect list; it does not identify
/// the line.
///
/// Pinned self-promoting in both directions. Localizing it is NOT this
/// landing's; per playbook §10 the next step is the sibling-C per-block dump
/// on the smallest divergent cell (128x128 cq12, delta -1), not reasoning
/// from the byte deltas — which range from -1 to +34 here and, per KB-22's
/// lesson, say nothing about whether the cause is a near-tie or a whole
/// unmodelled pass.
#[test]
#[ignore = "18 encode pairs; pins an open divergence class"]
fn palette_on_speed8_screen_content_is_pinned() {
    c::ref_init();
    // (w, h, cq) -> byte delta. Measured 2026-08-03, aarch64-apple-darwin,
    // --profile test-fast.
    const PALETTE_ON_SPEED8_OPEN: &[(usize, usize, i32, i64)] = &[
        (128, 128, 12, -1),
        (128, 128, 32, 3),
        (128, 128, 50, 1),
        (256, 256, 12, 0),
        (256, 256, 32, 3),
        (256, 256, 50, -5),
        (512, 512, 12, 34),
        (512, 512, 32, 0),
        (512, 512, 50, -17),
    ];
    let knobs = ToggleKnobs {
        enable_palette: true,
        ..Default::default()
    };
    let mut observed: Vec<(usize, usize, i32, i64)> = Vec::new();
    for &(w, h, cq) in &[
        (128usize, 128usize, 12i32),
        (128, 128, 32),
        (128, 128, 50),
        (256, 256, 12),
        (256, 256, 32),
        (256, 256, 50),
        (512, 512, 12),
        (512, 512, 32),
        (512, 512, 50),
    ] {
        let cell = screen_cell(&format!("scr{w}"), w, h, cq, 8);
        let c_tu = cell.c_encode_screen(true, false);
        let (v, d, _, reach) = measure(&cell, &c_tu, &knobs);
        println!("  scr {w}x{h} cq{cq} cpu8 palette-ON: {v:?} {d:+} reach {reach:?}");
        assert_ne!(
            v,
            Verdict::Panic,
            "{w}x{h} cq{cq} refused. reach {reach:?} — if reach[2] > 0 this is the genuine \
             estimate-arm palette arm and belongs in \
             `estimate_arm_palette_refusal_is_reachable_and_loud`"
        );
        assert_eq!(
            reach[2], 0,
            "{w}x{h} cq{cq}: an estimate-arm leaf satisfied C's `try_palette` here, so this \
             cell can no longer be attributed to the full-RD palette search"
        );
        // The speed-9 twin must MATCH: at speed 9 `hybrid_intra_pickmode == 0`
        // turns screen detection off entirely (encoder.c:2466-2470), so the
        // palette search does not run on either side. That is the control
        // proving the divergence is the speed-8 palette crossing and not the
        // cell's content.
        let c9 = screen_cell(&format!("scr{w}"), w, h, cq, 9);
        let c9_tu = c9.c_encode_screen(true, false);
        let (v9, d9, _, reach9) = measure(&c9, &c9_tu, &knobs);
        assert_eq!(
            (v9, reach9[0]),
            (Verdict::Match, 0),
            "the speed-9 control diverged ({d9:+}) or reached the palette gate, so the \
             speed-8 divergence below is not isolated to the speed-8 palette crossing"
        );
        if v != Verdict::Match {
            observed.push((w, h, cq, d));
        }
    }
    assert_eq!(
        observed,
        PALETTE_ON_SPEED8_OPEN.to_vec(),
        "the palette x speed-8 open map moved. FEWER/smaller entries => something closed it \
         (re-pin, and say which KB). MORE => a regression. Either way this is the FULL-RD \
         palette leaf at speed 8, not the nonrd estimate arm (reach[2] is asserted 0 above)"
    );
}
