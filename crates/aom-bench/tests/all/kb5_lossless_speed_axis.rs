//! **KB-5 / coverage-queue T1 — lossless (`--cq-level 0`) at every speed.**
//!
//! Until 2026-08-03 the e2e lossless path was closed at EVERY `--cpu-used`, and
//! the refusal was in the HARNESS, not the encoder: `aom-bench/src/lib.rs`
//! carried `assert!(p.quant.base_qindex > 0, "lossless cells are out of this
//! harness's scope")` immediately after a SINGLE-pass `read_uncompressed_header`
//! — which is also why it had to: `read_uncompressed_header` gates its
//! loop-filter / CDEF / restoration / tx-mode tail reads on
//! `cfg.coded_lossless`, so without the decoder's two-pass probe the parsed
//! header is wrong at qindex 0. KB-5's parity therefore rested entirely on its
//! own driver (`aom-encode/tests/kb5_lossless_localize.rs` and
//! `encoder_gate_chroma_ss_e2e::encoder_gate_lossless_cq0_e2e_kb5_repro`), both
//! of which hardcode `let speed = 0i32`.
//!
//! Behind that harness refusal sat two real encoder holes, both
//! `unimplemented!()` in `nonrd_pickmode.rs`: the TX_4X4 arms of
//! `block_yrd_lowbd` / `block_yrd_hbd`. They are the coded-lossless arms —
//! `select_tx_mode` returns `ONLY_4X4` when `cm->features.coded_lossless`
//! (rdopt_utils.h:392), so `mi->tx_size` is TX_4X4 at every leaf and the nonrd
//! estimate arm (speeds 8/9) walks nothing else.
//!
//! What this file measures:
//!
//! | class | cells | harness reverted | encoder arms reverted | fixed |
//! |---|---|---|---|---|
//! | cq0 bd8 x {4:2:0, mono} x {textured 64², smooth 128²}, `--cpu-used` 0..9 | 40 | PANIC 40 | PANIC 6 | **MATCH 40** |
//! | cq0 bd10/bd12 x the same two contents, `--cpu-used` {0, 8, 9} | 12 | PANIC 12 | PANIC 6 | **MATCH 12** |
//! | cq1 controls, `--cpu-used` 0..9 | 10 | MATCH | MATCH | MATCH |
//!
//! Reverting the two roots ONE AT A TIME gives **different** failing sets — 52
//! of 52 for the harness, 12 of 52 for the encoder arms (exactly the cells
//! whose partitions reach a TX_4X4 estimate leaf) — which is what makes them
//! two roots rather than two spellings of one (playbook §1). The cq1 rows are
//! the negative control the coverage-queue entry named ("cq1 is byte-exact at
//! all four" speeds it probed): they must stay green, so a failure here is
//! attributable to the lossless axis and nothing else.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb5_lossless_speed_axis -- --ignored --nocapture
//! ```

use aom_bench::EncodeCell;
use aom_encode::nonrd_pickmode::{multi_txb_leaf_counts, reset_multi_txb_leaf_counts};
use aom_sys_ref as c;

/// A textured 4:2:0 / mono cell. Lossless coding is content-sensitive in a way
/// the smooth `synthetic_diag` gradient is not: at qindex 0 a gradient
/// quantizes to a handful of coefficients per block and the leaf pick is a
/// near-tie everywhere, so a grid built only from it would exercise the TX_4X4
/// arm without ever loading it. This adds high-frequency texture on top.
fn textured_cell(label: &str, w: usize, h: usize, mono: bool, bd: u8, cq: i32, speed: i32) -> EncodeCell {
    let peak = (1u32 << bd) - 1;
    let scale = |v: u32| -> u16 { ((v * peak) / 255) as u16 };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            // A diagonal ramp plus a period-3/5 texture and a deterministic
            // pseudo-noise term — enough coefficient energy that qindex 0 codes
            // real EOBs rather than all-skip.
            let ramp = 32 + (r + col) * 160 / (w + h);
            let tex = ((r % 3) * 17 + (col % 5) * 11) as usize;
            let noise = ((r * 2654435761 + col * 40503) >> 7) % 23;
            y[r * w + col] = scale(((ramp + tex + noise) % 256) as u32);
        }
    }
    let (cw, ch) = ((w + 1) >> 1, (h + 1) >> 1);
    let (mut u, mut v) = (vec![0u16; cw * ch], vec![0u16; cw * ch]);
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                u[r * cw + col] = scale((60 + (r * 7 + col * 3) % 80) as u32);
                v[r * cw + col] = scale((70 + (r * 5 + col * 9) % 70) as u32);
            }
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x: 1,
        ss_y: 1,
        usage: 2, // ALLINTRA
        cq_level: cq,
        speed,
        bd,
        y,
        u,
        v,
    }
}

/// A SMOOTH cell — the diag gradient of `EncodeCell::synthetic_diag`, scaled to
/// `bd`. This exists for a measured reason: at `--cpu-used 8`
/// `hybrid_intra_pickmode` is 2, so `hybrid_use_rdopt` sends every leaf BELOW
/// BLOCK_16X16 with `source_variance >= 101` to the full-RD leaf
/// (partition_search.c:755) and only leaves at 16x16 or larger reach the nonrd
/// ESTIMATE arm. On [`textured_cell`] the variance partitioner splits a 64x64
/// SB all the way to 8x8, so a textured-only grid reports `tx4x4-leaves 0` at
/// cpu8 — byte-identical, and vacuous with respect to the arms this landing
/// wrote. Measured on this grid: textured 64x64 gives 0 estimate leaves at
/// cpu8; the smooth 128x128 gives 256.
fn smooth_cell(label: &str, w: usize, h: usize, mono: bool, bd: u8, cq: i32, speed: i32) -> EncodeCell {
    let peak = (1u32 << bd) - 1;
    let scale = |v: u32| -> u16 { ((v * peak) / 255) as u16 };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = scale((32 + (r + col) * 190 / (w + h)) as u32);
        }
    }
    let (cw, ch) = ((w + 1) >> 1, (h + 1) >> 1);
    let (mut u, mut v) = (vec![0u16; cw * ch], vec![0u16; cw * ch]);
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                let val = scale((60 + (r * 7 + col * 3) % 80) as u32);
                u[r * cw + col] = val;
                v[r * cw + col] = val;
            }
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: cq,
        speed,
        bd,
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

/// Encode on both sides and classify, plus the number of NONRD-ESTIMATE-ARM
/// leaves that walked more than one txb.
///
/// That counter is this file's reach witness for the TX_4X4 arms. At
/// coded-lossless `nonrd_leaf_tx_size` returns TX_4X4 at every bsize, so an
/// estimate-arm leaf is single-txb only at BLOCK_4X4 — which the KEY variance
/// partitioner never stamps. A nonzero count therefore means "the estimate arm
/// ran, and it ran the TX_4X4 walk"; a zero one at cpu 8/9 would mean this row
/// says nothing about the arms this landing wrote.
///
/// A PANIC is a distinct outcome, not a test abort — the point of this file is
/// the shape of the refusal set.
fn measure(cell: &EncodeCell) -> (Verdict, i64, String, u64) {
    let c_tu = cell.c_encode();
    assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let msg = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = std::sync::Arc::clone(&msg);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        *sink.lock().unwrap() = info.to_string().lines().last().unwrap_or("").to_string();
    }));
    reset_multi_txb_leaf_counts();
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cell.port_encode(&c_tu)));
    std::panic::set_hook(hook);
    let multi_txb: u64 = multi_txb_leaf_counts().iter().sum();
    let note = msg.lock().unwrap().clone();
    match got {
        Ok(p) if p == real => (Verdict::Match, 0, String::new(), multi_txb),
        Ok(p) => (
            Verdict::Diverge,
            p.len() as i64 - real.len() as i64,
            String::new(),
            multi_txb,
        ),
        Err(_) => (Verdict::Panic, 0, note, multi_txb),
    }
}

/// Assert the reference stream really is coded-lossless, from the REAL stream's
/// own frame header — the reach assertion this whole file rests on (playbook
/// §2: derive coverage from the artefact, not from the `--cq-level` we asked
/// for). `av1_set_quantizer` is free to refuse qindex 0; if it ever does, every
/// row below silently stops testing the lossless path.
fn stream_is_coded_lossless(cell: &EncodeCell) -> bool {
    let c_tu = cell.c_encode();
    aom_bench::stream_base_qindex(&c_tu) == 0
}

// ---------------------------------------------------------------------------
// 1. The axis the harness was closing.
// ---------------------------------------------------------------------------

/// **The KB-5 speed-axis byte gate.** Every cq0 row PANICKED on the pre-fix
/// tree — at speeds 0..7 with *"lossless cells are out of this harness's
/// scope"*, and at 8/9 with that same assert (it fires before the encoder runs)
/// which, once removed, becomes *"TX_4X4 block_yrd (lossless) — out of canon
/// envelope"*. Every row is byte-identical to real aomenc now.
#[test]
#[ignore = "40 encode pairs across cpu-used 0..9; nightly / on-demand tier"]
fn lossless_cq0_byte_matches_at_every_speed() {
    c::ref_init();
    let mut rows: Vec<(String, Verdict, i64)> = Vec::new();
    let mut lossless_confirmed = 0usize;
    // Estimate-arm reach, per speed: cpu 8 and cpu 9 are different dispatches
    // (`hybrid_intra_pickmode` 2 vs 0) and each must be witnessed on its own.
    let mut estimate_leaves = [0u64; 10];

    // (a) TEXTURED, 64x64, 4:2:0 and mono — the full-RD lossless path at every
    //     speed, and cpu9's estimate arm (which takes every leaf).
    // (b) SMOOTH gradient, 128x128 — the only class that reaches the estimate
    //     arm at cpu8 (see `smooth_cell`).
    for &(mono, tag) in &[(false, "420"), (true, "mono")] {
        for speed in 0..=9i32 {
            for &(smooth, w, h) in &[(false, 64usize, 64usize), (true, 128, 128)] {
                let kind = if smooth { "smooth" } else { "textur" };
                let label = format!("cq0-{tag}-{kind}-s{speed}");
                let cell = if smooth {
                    smooth_cell(&label, w, h, mono, 8, 0, speed)
                } else {
                    textured_cell(&label, w, h, mono, 8, 0, speed)
                };
                if stream_is_coded_lossless(&cell) {
                    lossless_confirmed += 1;
                }
                let (v, d, note, multi_txb) = measure(&cell);
                println!(
                    "  cq0 {tag} {kind} {w}x{h} cpu{speed}: {v:?} {d:+}  tx4x4-leaves \
                     {multi_txb}{}",
                    if note.is_empty() {
                        String::new()
                    } else {
                        format!("  [{note}]")
                    }
                );
                estimate_leaves[speed as usize] += multi_txb;
                rows.push((format!("cq0 {tag} {kind} {w}x{h} cpu{speed}"), v, d));
            }
        }
    }

    // Reach 1 (playbook §2): every row must genuinely be coded-lossless, or the
    // grid is testing the ordinary quantizer under a lossless-sounding label.
    assert_eq!(
        lossless_confirmed,
        rows.len(),
        "only {lossless_confirmed} of {} reference streams carried base_qindex 0",
        rows.len()
    );
    // Reach 2: the two nonrd dispatches must each have walked a TX_4X4 grid,
    // else the cpu8/cpu9 rows say nothing about `block_yrd_lowbd`'s TX_4X4 arm.
    assert!(
        estimate_leaves[8] > 0 && estimate_leaves[9] > 0,
        "the estimate arm's TX_4X4 walk was not reached: cpu8 {} leaves, cpu9 {} \
         leaves. Speeds 0..7 are expected to be 0 (full-RD path); a 0 at 8 or 9 \
         means this grid's partitions never left a leaf large enough for \
         `hybrid_use_rdopt` to decline",
        estimate_leaves[8],
        estimate_leaves[9]
    );
    assert_eq!(
        &estimate_leaves[..8],
        &[0u64; 8],
        "a speed below 8 dispatched the nonrd estimate arm — the arm is gated on \
         `allintra && speed >= 8` (pack.rs), so this is a dispatch regression"
    );

    let bad: Vec<String> = rows
        .iter()
        .filter(|r| r.1 != Verdict::Match)
        .map(|r| format!("{} {:?} ({:+})", r.0, r.1, r.2))
        .collect();
    assert!(
        bad.is_empty(),
        "a lossless speed-axis cell stopped byte-matching. A PANIC naming \
         `TX_4X4 block_yrd` means the coded-lossless ONLY_4X4 arm regressed \
         (`nonrd_leaf_tx_size`'s lossless term, nonrd_pickmode.c:1591-1594). A \
         PANIC naming `base_qindex` means the two-pass coded-lossless probe was \
         removed from `port_encode_full`. A DIVERGE at cpu 8/9 is the estimate \
         arm's TX_4X4 kernels (`aom_fdct4x4` / `aom_fdct4x4_lp` + the 4x4 \
         scan): {bad:?}"
    );
}

/// The negative control the coverage-queue entry named. cq1 is one quantizer
/// step off lossless and was byte-exact before this landing; it must stay so,
/// which is what makes a cq0 failure attributable to the lossless axis.
#[test]
#[ignore = "10 encode pairs; nightly / on-demand tier"]
fn cq1_control_stays_byte_exact_at_every_speed() {
    c::ref_init();
    let mut bad: Vec<String> = Vec::new();
    for speed in 0..=9i32 {
        let cell = textured_cell(&format!("cq1-420-s{speed}"), 64, 64, false, 8, 1, speed);
        assert!(
            !stream_is_coded_lossless(&cell),
            "cq1 cpu{speed}: the reference stream came back coded-lossless — \
             this row is then a duplicate of the cq0 grid, not a control"
        );
        let (v, d, note, multi_txb) = measure(&cell);
        println!("  cq1 420 64x64 cpu{speed}: {v:?} {d:+}  multi-txb {multi_txb}  {note}");
        if v != Verdict::Match {
            bad.push(format!("cq1 cpu{speed} {v:?} ({d:+})"));
        }
    }
    assert!(
        bad.is_empty(),
        "the cq1 CONTROL regressed — this landing broke something outside the \
         lossless axis: {bad:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. The hbd arm.
// ---------------------------------------------------------------------------

/// `block_yrd_hbd`'s TX_4X4 arm. Its forward transform is `aom_fdct4x4`, which
/// — unlike the lowbd `aom_fdct4x4_lp` — is genuinely ISA-conditional at bd10 /
/// bd12: the NEON and SSE2 tiers hold every intermediate in `int16` where
/// `aom_fdct4x4_c` uses `tran_high_t`, and the first pass already reaches 46296
/// at bd10. `nonrd_block_yrd_hbd_diff::fdct4x4_dispatched_matches_the_real_
/// specialised_symbol` locks the model against the exported tier; this locks
/// the whole encode.
///
/// Speeds {0, 8, 9}: 0 is the full-RD path (KB-5's own envelope, at a bit depth
/// its driver never ran), 8 and 9 are the two nonrd estimate dispatches.
#[test]
#[ignore = "12 encode pairs at bd10/bd12; nightly / on-demand tier"]
fn lossless_cq0_byte_matches_at_high_bit_depth() {
    c::ref_init();
    let mut rows: Vec<(String, Verdict, i64)> = Vec::new();
    let mut estimate_leaves = [0u64; 10];
    for &bd in &[10u8, 12] {
        for &speed in &[0i32, 8, 9] {
            for &(smooth, w, h) in &[(false, 64usize, 64usize), (true, 128, 128)] {
                let kind = if smooth { "smooth" } else { "textur" };
                let label = format!("cq0-bd{bd}-{kind}-s{speed}");
                let cell = if smooth {
                    smooth_cell(&label, w, h, false, bd, 0, speed)
                } else {
                    textured_cell(&label, w, h, false, bd, 0, speed)
                };
                assert!(
                    stream_is_coded_lossless(&cell),
                    "bd{bd} {kind} cpu{speed}: the reference stream is not coded-lossless"
                );
                let (v, d, note, multi_txb) = measure(&cell);
                println!(
                    "  cq0 bd{bd} {kind} {w}x{h} cpu{speed}: {v:?} {d:+}  tx4x4-leaves \
                     {multi_txb}{}",
                    if note.is_empty() {
                        String::new()
                    } else {
                        format!("  [{note}]")
                    }
                );
                estimate_leaves[speed as usize] += multi_txb;
                rows.push((format!("cq0 bd{bd} {kind} cpu{speed}"), v, d));
            }
        }
    }
    assert!(
        estimate_leaves[8] > 0 && estimate_leaves[9] > 0,
        "the hbd TX_4X4 walk was not reached: cpu8 {} leaves, cpu9 {} leaves",
        estimate_leaves[8],
        estimate_leaves[9]
    );
    let bad: Vec<String> = rows
        .iter()
        .filter(|r| r.1 != Verdict::Match)
        .map(|r| format!("{} {:?} ({:+})", r.0, r.1, r.2))
        .collect();
    assert!(
        bad.is_empty(),
        "an hbd lossless cell stopped byte-matching. At cpu 8/9 the first thing \
         to check is `fdct4x4_dispatched`: modelling `aom_fdct4x4_c` instead of \
         the linked tier is a silent defect at every hbd lossless block: {bad:?}"
    );
}
