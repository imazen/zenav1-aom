//! **S4 coverage extension — KB-23's (frame-edge x speed) crossing beyond the
//! one grid it was closed on.**
//!
//! KB-23 (`cnn_output_valid`: C computes the intra-CNN partition prune only at
//! a whole-in-frame `BLOCK_64X64` root — `partition_strategy.c:142/160/227`
//! + `partition_search.c:3340-3343` — so a superblock whose 64x64 root is not
//! whole-in-frame prunes NOTHING anywhere inside it) was found and closed on a
//! single grid, and its entry says so:
//!
//! > *"the grid is one content source at cq24, bd8 4:2:0, SB64, speeds 0..4 ...
//! > the partial-SB x speed crossing has not been swept at other bit depths,
//! > subsamplings, or SB128."*
//!
//! **SB128 is not a rerun of that grid.** C's CNN-output invalidation is
//! per-`BLOCK_64X64`, not per-superblock, so under `--sb-size=128` one
//! superblock contains FOUR independent CNN roots and the whole-in-frame
//! predicate is asked four times per SB at a different alignment. The port's
//! fix (`cnn_root_whole_in_frame`, `partition_pick.rs`) rounds the block's mi
//! position down to its containing 64x64 (`(mi/16)*16`) precisely so that it is
//! SB-size-agnostic — a claim nothing had measured until this file, because
//! `sb128_e2e.rs::sb128_partial_sb_e2e` runs at **speed 0 only**, where
//! `intra_cnn_based_part_prune_level` is 0 (`speed_features.c:387-388`) and the
//! prune does not exist.
//!
//! The size lists below deliberately INTERLEAVE the two alignments, following
//! the method that made KB-23 separable in the first place: sizes that are
//! SB64-exact but SB128-PARTIAL (192², 320²) sit next to sizes that are partial
//! under both (132², 196²) and exact under both (128², 256²). If a divergence
//! tracked "64-px edges" it would spare 192²/320²; if it tracked "128-px edges"
//! it would hit them. The result pattern decides, not intuition.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test s4cov_partial_sb_axis -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;
use aom_sys_ref::cx_ctrl::{AOM_SUPERBLOCK_SIZE_128X128, AV1E_SET_SUPERBLOCK_SIZE};

/// cq24 -> `base_qindex` 96, the quality point KB-23's grid was mapped at, so
/// the SB128 and format arms here are comparable to it row for row.
const CQ: i32 = 24;

/// Mirror-tile a decoded cell up to `w x h`. Same recipe as
/// `kb22_hd_arms::mirror_tile` (and `kb19_min_partition_4k`, and the size axis
/// of the config-permutation gate), extended to carry monochrome cells through
/// unchanged — mirroring keeps the seam continuous so the enlarged frame stays
/// photographic instead of acquiring a synthetic edge grid every tile period.
fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, speed: i32) -> EncodeCell {
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
    let (mut u, mut v) = (Vec::new(), Vec::new());
    if !base.mono {
        let (bcw, bch) = (
            (bw + base.ss_x) >> base.ss_x,
            (bh + base.ss_y) >> base.ss_y,
        );
        let (cw, ch) = ((w + base.ss_x) >> base.ss_x, (h + base.ss_y) >> base.ss_y);
        u = vec![0u16; cw * ch];
        v = vec![0u16; cw * ch];
        for r in 0..ch {
            for col in 0..cw {
                u[r * cw + col] = base.u[mir(r, bch) * bcw + mir(col, bcw)];
                v[r * cw + col] = base.v[mir(r, bch) * bcw + mir(col, bcw)];
            }
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
        cq_level: CQ,
        speed,
        bd: base.bd,
        y,
        u,
        v,
    }
}

/// Drop the chroma planes (4:0:0).
fn to_mono(base: &EncodeCell, label: &str) -> EncodeCell {
    EncodeCell {
        label: label.to_string(),
        mono: true,
        ss_x: 1,
        ss_y: 1,
        u: Vec::new(),
        v: Vec::new(),
        ..base.clone()
    }
}

/// Re-render a 4:2:0 cell at `(ss_x, ss_y)` by nearest-neighbour chroma
/// upsampling — see `s4cov_qm_axis.rs::to_ss` for why nearest-neighbour.
fn to_ss(base: &EncodeCell, label: &str, ss_x: usize, ss_y: usize) -> EncodeCell {
    assert!(!base.mono);
    assert_eq!((base.ss_x, base.ss_y), (1, 1));
    let bcw = (base.w + 1) >> 1;
    let (cw, ch) = ((base.w + ss_x) >> ss_x, (base.h + ss_y) >> ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        let sr = (r << ss_y) >> 1;
        for col in 0..cw {
            let sc = (col << ss_x) >> 1;
            u[r * cw + col] = base.u[sr * bcw + sc];
            v[r * cw + col] = base.v[sr * bcw + sc];
        }
    }
    EncodeCell {
        label: label.to_string(),
        ss_x,
        ss_y,
        u,
        v,
        ..base.clone()
    }
}

fn base_b8() -> EncodeCell {
    c::ref_init();
    EncodeCell::real_content("s4base", "av1-1-b8-00-quantizer-00", None, CQ, 0)
}

fn base_b10() -> EncodeCell {
    c::ref_init();
    EncodeCell::real_content("s4base10", "av1-1-b10-00-quantizer-00", None, CQ, 0)
}

/// One cell's verdict. `Panic` is a distinct outcome, not a test abort: an
/// unported arm that PANICS is exactly what a coverage extension is looking for
/// (KB-20 was a panic sitting between two individually-green PARITY rows), and
/// aborting on the first one hides the shape of the rest of the map.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Ok,
    Diverge,
    Panic,
}

struct Row {
    size: (usize, usize),
    speed: i32,
    verdict: Verdict,
    delta: i64,
    /// Whether the frame's right/bottom superblocks are partial, at the SB size
    /// this row was encoded with.
    partial: bool,
    /// First line of the panic message, when `verdict == Panic`.
    note: String,
}

impl Row {
    fn ok(&self) -> bool {
        self.verdict == Verdict::Ok
    }
}

/// Encode one cell against real aomenc with the given extra C controls
/// (`&[]` = stock SB64; the SB128 pair for `--sb-size=128`). Returns the row
/// plus the reference stream, so callers can run their own anti-vacuity checks.
fn measure(cell: &EncodeCell, ctrls: &[(i32, i32)], partial: bool) -> (Row, Vec<u8>) {
    let c_tu = cell.c_encode_ctrls(ctrls);
    assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let msg = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = std::sync::Arc::clone(&msg);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        *sink.lock().unwrap() = info.to_string().lines().last().unwrap_or("").to_string();
    }));
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(&c_tu, &ToggleKnobs::default())
    }));
    std::panic::set_hook(hook);
    let (verdict, delta, note) = match got {
        Ok(p) if p == real => (Verdict::Ok, 0, String::new()),
        Ok(p) => (Verdict::Diverge, p.len() as i64 - real.len() as i64, String::new()),
        Err(_) => (Verdict::Panic, 0, msg.lock().unwrap().clone()),
    };
    let row = Row {
        size: (cell.w, cell.h),
        speed: cell.speed,
        verdict,
        delta,
        partial,
        note,
    };
    (row, c_tu)
}

/// `(partial-SB not-ok, SB-exact not-ok)`.
fn summarise(tag: &str, rows: &[Row]) -> (usize, usize) {
    for r in rows {
        println!(
            "  {tag} {}x{} cpu{} ({}): delta {:+} -> {}{}",
            r.size.0,
            r.size.1,
            r.speed,
            if r.partial { "PARTIAL-SB" } else { "SB-exact " },
            r.delta,
            match r.verdict {
                Verdict::Ok => "MATCH",
                Verdict::Diverge => "DIVERGE",
                Verdict::Panic => "PANIC",
            },
            if r.note.is_empty() {
                String::new()
            } else {
                format!("  [{}]", r.note)
            }
        );
    }
    let pd = rows.iter().filter(|r| r.partial && !r.ok()).count();
    let ed = rows.iter().filter(|r| !r.partial && !r.ok()).count();
    println!(
        "  {tag}: {}/{} byte-exact | PARTIAL-SB not-ok {pd}/{}, SB-exact not-ok {ed}/{} \
         (panics: {})",
        rows.iter().filter(|r| r.ok()).count(),
        rows.len(),
        rows.iter().filter(|r| r.partial).count(),
        rows.iter().filter(|r| !r.partial).count(),
        rows.iter().filter(|r| r.verdict == Verdict::Panic).count(),
    );
    (pd, ed)
}

// ---------------------------------------------------------------------------
// 1. SB128 x partial-SB x speed.
// ---------------------------------------------------------------------------

/// **The SB128 arm of KB-23's residual.** `--sb-size=128`, bd8 4:2:0 cq24, over
/// sizes interleaved across BOTH alignments, at every RD speed 0..7.
///
/// | size | mult of 64 | mult of 128 | what it isolates |
/// |---|---|---|---|
/// | 128² | yes | yes | exact under both — the negative control |
/// | 132² | no | no | partial under both |
/// | 192² | **yes** | **no** | SB64-exact, SB128-PARTIAL |
/// | 196² | no | no | partial under both (the KB-6 / KB-23 frame) |
/// | 256² | yes | yes | exact under both |
/// | 320² | **yes** | **no** | SB64-exact, SB128-PARTIAL, 2.5 SBs |
///
/// 192² and 320² are the rows that carry the argument: under SB128 their
/// right/bottom superblocks are partial while every 64x64 CNN root inside them
/// is still whole-in-frame, so they separate "the port keys the latch off the
/// SUPERBLOCK" (they would diverge) from "off the containing 64x64, as C does"
/// (they would not).
///
/// **MEASURED 2026-08-01** (`benchmarks/s4cov_axes_2026-08-01.tsv`): 48/48
/// byte-exact, partial-SB and SB-exact alike, at every speed.
#[test]
#[ignore = "48 SB128 encode pairs up to 320x320; nightly / on-demand tier"]
fn sb128_partial_sb_speed_axis_byte_matches() {
    let base = base_b8();
    let sb128 = [(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128)];
    const SIZES: &[(usize, usize)] = &[
        (128, 128),
        (132, 132),
        (192, 192),
        (196, 196),
        (256, 256),
        (320, 320),
    ];
    let mut rows = Vec::new();
    let mut inert = Vec::new();
    for &(w, h) in SIZES {
        for speed in 0..=7 {
            let cell = mirror_tile(&base, &format!("sb128_{w}x{h}_s{speed}"), w, h, speed);
            // Partiality is asked at the 128-px superblock, which is what this
            // arm varies.
            let partial = w % 128 != 0 || h % 128 != 0;
            let (row, c_tu) = measure(&cell, &sb128, partial);
            // Anti-vacuity, per row: `--sb-size=128` must change the C stream
            // vs `--sb-size=64`, otherwise the row is an SB64 test wearing an
            // SB128 label (playbook §8 — derive coverage from artefacts).
            if c_tu == cell.c_encode_ctrls(&[]) {
                inert.push(format!("{w}x{h} cpu{speed}"));
            }
            rows.push(row);
        }
    }
    let (pd, ed) = summarise("sb128", &rows);
    assert!(
        inert.is_empty(),
        "--sb-size=128 did not change the C stream vs --sb-size=64 on these rows, so they \
         prove nothing about the 128-superblock geometry: {inert:?}"
    );
    assert_eq!(
        pd, 0,
        "an SB128 PARTIAL-SB frame diverged. This is KB-23's shape at the 128-px superblock: \
         C invalidates its CNN output per BLOCK_64X64 (partition_search.c:3340-3343), not per \
         superblock, so `cnn_root_whole_in_frame` must key off the containing 64x64 \
         (`(mi/16)*16`) and stay correct under both SB sizes."
    );
    assert_eq!(
        ed, 0,
        "an SB128 SB-exact frame diverged — not KB-23's shape (frame-EDGE only), so it needs \
         its own localization"
    );
}

// ---------------------------------------------------------------------------
// 2. Partial-SB x speed x chroma format, at SB64.
// ---------------------------------------------------------------------------

/// **The subsampling arm of KB-23's residual.** KB-23's grid is 4:2:0 only; the
/// partial-SB machinery it composes with is emphatically not
/// subsampling-neutral (KB-6's chroma visible clips go through
/// `max_block_units`, and the frame-edge entropy-stamp tail-zero
/// (`av1_set_entropy_contexts`, `blockd.c:29`) clips a chroma footprint whose
/// size depends on `ss_x`/`ss_y`). Monochrome removes that machinery entirely
/// and is the negative control for it.
///
/// Sizes are KB-23's four (two partial, two exact) so the rows are directly
/// comparable; speeds 0..7 widen its 0..4.
///
/// **MEASURED 2026-08-01**: 4:4:4 32/32 and 4:2:2 32/32 byte-exact; monochrome
/// 28/32, its four failures ALL at `--cpu-used=0` and pinned in `MONO_S0_OPEN`
/// below. That mono set is **not** a partial-SB result — it hits SB-exact
/// (192², 256²) and partial-SB (132², 196²) sizes alike, and
/// `mono_speed0_size_qindex_localize` reduces it to a single-superblock repro
/// at one quality point. See that test for the shape.
///
/// One `unimplemented!()` was removed to get here: the speed-7 VAR_BASED
/// walk's frame-edge single-strip rect (`rd_use_partition_real`), which
/// panicked on 196² cpu7 for monochrome, 4:4:4 and bd10 alike.
#[test]
#[ignore = "96 encode pairs across three chroma formats; nightly / on-demand tier"]
fn partial_sb_speed_axis_chroma_formats_byte_match() {
    let b8 = base_b8();
    const SIZES: &[(usize, usize)] = &[(132, 132), (192, 192), (196, 196), (256, 256)];
    let formats: Vec<(&str, EncodeCell)> = vec![
        ("mono", to_mono(&b8, "mono")),
        ("444 ", to_ss(&b8, "444", 0, 0)),
        ("422 ", to_ss(&b8, "422", 1, 0)),
    ];
    // The monochrome speed-0 near-tie, pinned in BOTH directions. It is NOT
    // this axis's finding — `mono_speed0_size_qindex_localize` reduces it to
    // 64x64 (one superblock) at cq24 alone — but it lands in this grid, and
    // the alternative to pinning it would be moving the grid to a quality
    // point where it does not fire, which is the banned form.
    const MONO_S0_OPEN: &[(&str, usize, usize, i32)] = &[
        ("mono", 132, 132, 0),
        ("mono", 192, 192, 0),
        ("mono", 196, 196, 0),
        ("mono", 256, 256, 0),
    ];
    let mut observed: Vec<(String, usize, usize, i32)> = Vec::new();
    let mut panicked: Vec<String> = Vec::new();
    for (tag, src) in &formats {
        let mut rows = Vec::new();
        for &(w, h) in SIZES {
            for speed in 0..=7 {
                let cell = mirror_tile(src, &format!("{tag}_{w}x{h}_s{speed}"), w, h, speed);
                assert_eq!(
                    (cell.mono, cell.ss_x, cell.ss_y),
                    (src.mono, src.ss_x, src.ss_y),
                    "the mirror-tile must preserve the chroma format"
                );
                let (row, _) = measure(&cell, &[], w % 64 != 0 || h % 64 != 0);
                if row.verdict == Verdict::Panic {
                    panicked.push(format!("{} {w}x{h} cpu{speed}: {}", tag.trim(), row.note));
                }
                if !row.ok() {
                    observed.push((tag.trim().to_string(), w, h, speed));
                }
                rows.push(row);
            }
        }
        summarise(tag.trim(), &rows);
    }
    // A PANIC is never pinnable — an unported arm that aborts the encode is a
    // hard failure for a drop-in replacement, whatever its byte verdict would
    // have been.
    assert!(
        panicked.is_empty(),
        "the port PANICKED instead of encoding. On this grid that was the speed-7 \
         VAR_BASED frame-edge single-strip rect in `rd_use_partition_real`; a new one is a \
         new unported arm: {panicked:?}"
    );
    let pinned: Vec<(String, usize, usize, i32)> = MONO_S0_OPEN
        .iter()
        .map(|(t, w, h, s)| ((*t).to_string(), *w, *h, *s))
        .collect();
    assert_eq!(
        observed, pinned,
        "the non-4:2:0 partial-SB x speed map moved. A PARTIAL-SB-only change at speed >= 1 \
         is KB-23's shape at a subsampling it was never measured at; a row that started \
         MATCHING means the monochrome cq24 speed-0 near-tie closed (re-pin, and delete \
         `mono_speed0_size_qindex_localize`)"
    );
}

/// **Localizer for the monochrome divergence found by the test above.**
/// Diagnostic, not a gate.
///
/// The first map (size x cq at `--cpu-used=0`) showed the divergence is NOT a
/// multi-superblock effect at all: **64x64 monochrome — a single superblock —
/// diverges too, and only at cq24**. So the shape is
/// `(monochrome, base_qindex 96, speed 0)`, which is why no existing gate sees
/// it: `config_permutations.rs`'s `q00_mono64` row runs the same 64x64 mono
/// content at every speed 0..9 but at `SPEED_CQ = 32`, one quality point away.
///
/// This version sweeps cq densely around 24 to bound the qindex window, walks
/// the speed axis at the divergent point, and runs a 4:2:0 control on the
/// identical crop and cq — so "monochrome" and "this quality point" are
/// separated by measurement rather than by assumption.
#[test]
#[ignore = "~40 encode pairs; diagnostic, run explicitly"]
fn mono_speed0_size_qindex_localize() {
    let b8 = base_b8();
    let mono = to_mono(&b8, "mono");
    let run = |src: &EncodeCell, n: usize, cq: i32, speed: i32| -> (Verdict, i64, Option<usize>) {
        let mut cell = mirror_tile(src, &format!("loc{n}_{cq}_{speed}"), n, n, speed);
        cell.cq_level = cq;
        let c_tu = cell.c_encode_ctrls(&[]);
        let real = EncodeCell::frame_obu_payload(&c_tu);
        let port = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
        let fd = (0..port.len().min(real.len())).find(|&i| port[i] != real[i]);
        let v = if port == real { Verdict::Ok } else { Verdict::Diverge };
        (v, port.len() as i64 - real.len() as i64, fd)
    };

    println!("  -- 64x64 MONO, cpu0, dense cq sweep (cq -> base_qindex is 4*cq here) --");
    let mut mono_bad_cq = Vec::new();
    for cq in 18..=30 {
        let (v, d, fd) = run(&mono, 64, cq, 0);
        println!(
            "     cq{cq:<2} -> {} delta {:+} first-diff {:?}",
            if v == Verdict::Ok { "ok     " } else { "DIVERGE" },
            d,
            fd
        );
        if v != Verdict::Ok {
            mono_bad_cq.push(cq);
        }
    }

    println!("  -- 64x64, cq24, speed axis: MONO vs its 4:2:0 CONTROL --");
    let mut mono_bad_speed = Vec::new();
    let mut ctl_bad_speed = Vec::new();
    for speed in 0..=7 {
        let (vm, dm, _) = run(&mono, 64, 24, speed);
        let (vc, dc, _) = run(&b8, 64, 24, speed);
        println!(
            "     cpu{speed}: mono {} ({:+})   |   4:2:0 control {} ({:+})",
            if vm == Verdict::Ok { "ok     " } else { "DIVERGE" },
            dm,
            if vc == Verdict::Ok { "ok     " } else { "DIVERGE" },
            dc
        );
        if vm != Verdict::Ok {
            mono_bad_speed.push(speed);
        }
        if vc != Verdict::Ok {
            ctl_bad_speed.push(speed);
        }
    }
    println!(
        "  mono localize: divergent cq (cpu0) {mono_bad_cq:?}; divergent speeds (cq24) \
         {mono_bad_speed:?}; 4:2:0 control divergent speeds {ctl_bad_speed:?}"
    );
    assert!(
        ctl_bad_speed.is_empty(),
        "the 4:2:0 CONTROL diverged at the same crop and cq, so the divergence is not \
         monochrome-specific and this localizer is pointed at the wrong axis: {ctl_bad_speed:?}"
    );
    assert!(
        !mono_bad_cq.is_empty(),
        "64x64 monochrome is now byte-exact across cq18..30 at speed 0 — the divergence \
         this localizer exists for has closed. Re-pin `MONO_S0_OPEN` in \
         `partial_sb_speed_axis_chroma_formats_byte_match` and delete this test."
    );
}

// ---------------------------------------------------------------------------
// 3. Partial-SB x speed at high bit depth — measured only where it is
//    INTERPRETABLE.
// ---------------------------------------------------------------------------

/// **The bit-depth arm of KB-23's residual, run only at the speeds where it can
/// be read.**
///
/// bd10 4:2:0 diverges from real aomenc at `--cpu-used` 1..6 on SB-EXACT
/// 64x64 content already — that is the pre-existing pinned `b10_64` band of
/// `config_permutations.rs::speed_envelope_stock_map_is_pinned`, re-measured
/// and widened by `s4cov_qm_axis.rs` (it reaches 4:4:4, 12-bit, monochrome and
/// cq5). A partial-SB cell at those speeds therefore CANNOT answer the KB-23
/// question: both explanations predict a divergence.
///
/// Speeds **0 and 7** are clean at bd10 on SB-exact content, so those are the
/// speeds at which "does a frame-edge superblock add a divergence at high bit
/// depth?" is a well-posed question — and each is asked here with its own
/// SB-exact control at the identical speed, so the answer does not rest on
/// the other file's measurement. Speed 7 is the load-bearing one: the CNN
/// prune is live there (`intra_cnn_based_part_prune_level` is nonzero from
/// speed 1), while at speed 0 it does not exist at all.
///
/// **MEASURED 2026-08-01**: 8/8 byte-exact (four sizes x speeds {0, 7}).
#[test]
#[ignore = "8 bd10 encode pairs; nightly / on-demand tier"]
fn partial_sb_high_bitdepth_byte_matches_where_interpretable() {
    let b10 = base_b10();
    assert_eq!(b10.bd, 10);
    const SIZES: &[(usize, usize)] = &[(132, 132), (192, 192), (196, 196), (256, 256)];
    let mut rows = Vec::new();
    for &(w, h) in SIZES {
        for speed in [0, 7] {
            let cell = mirror_tile(&b10, &format!("b10_{w}x{h}_s{speed}"), w, h, speed);
            assert_eq!(cell.bd, 10);
            let (row, _) = measure(&cell, &[], w % 64 != 0 || h % 64 != 0);
            rows.push(row);
        }
    }
    let (pd, ed) = summarise("bd10", &rows);
    // The SB-exact control must be clean at these speeds — that is the premise
    // that makes the partial-SB verdict readable. If it is NOT, the bd10
    // speed-1..6 band has spread and this test is measuring that instead.
    assert_eq!(
        ed, 0,
        "the bd10 SB-EXACT control diverged at speed 0 or 7. Those are the only speeds where \
         bd10 is byte-exact on SB-exact content, and they are what make the partial-SB rows \
         below interpretable — so this is a bd10 regression (or a spread of the pinned \
         speed-1..6 band), not a partial-SB result"
    );
    assert_eq!(
        pd, 0,
        "a bd10 PARTIAL-SB frame diverged while its SB-exact control at the same speed \
         matched — KB-23's shape at a bit depth it was never measured at"
    );
}
