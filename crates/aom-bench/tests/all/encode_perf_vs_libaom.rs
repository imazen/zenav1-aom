//! ENCODE TIME vs real libaom, on the standalone entry point — and the RD claim
//! that goes with it.
//!
//! The standing goal asks for "encode time within libaom" and for "matching the
//! RD of C". Both were unmeasured IN THIS REPO: the only numbers were two
//! retained fleet witnesses quoted in GitHub #16 (2.49x and 2.65x), taken on a
//! different machine with a different harness. This file measures both here, so
//! the claim has a reproducible in-repo record.
//!
//! **RD is measured as BYTE IDENTITY, which is stronger than any RD metric.**
//! If the port emits the same bytes as `aomenc` for the same source and
//! settings, it made every partition, mode, transform and coefficient decision
//! C made — the rate is identical and the reconstruction is identical, so the
//! rate-distortion point is the same point, not a nearby one. No SSIM or
//! BD-rate comparison can say more than that. Every cell here asserts it, so
//! the timing ratio below is a like-for-like comparison of two encoders
//! producing IDENTICAL output rather than a comparison of different work.
//!
//! **Method** (the repo's own perf discipline, `DIFFERENTIAL_PLAYBOOK` §6):
//! arms are INTERLEAVED within each repetition rather than run in separate
//! phases, so a drift in machine state hits both equally; the reported figure
//! is the per-cell MINIMUM over repetitions, which is the right reduction for
//! wall time (contention can only ADD); and a same-arm null is reported so the
//! reader can see the floor the ratio is resolved against.
//!
//! On-demand (`#[ignore]`): it is a timing measurement, and
//! `CLAUDE.md` requires those to be isolated. Run with
//! `just gate-encode-perf`.

use aom_bench::EncodeCell;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;
use std::time::{Duration, Instant};

/// Repetitions per cell. The reduction is the MINIMUM, so this is a noise
/// floor, not an averaging window: 3 is enough to drop a single deschedule.
const REPS: usize = 3;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The port's standalone entry, configured exactly as the C arm below.
///
/// `enable_cdef = false` / `enable_restoration = true` is real aomenc's
/// ALLINTRA default (`av1_cx_iface.c:3067` for CDEF; restoration stays 1), so
/// this is the configuration a still-image caller actually gets.
fn cfg_for(cell: &EncodeCell) -> KeyFrameConfig {
    let mut cfg = KeyFrameConfig::allintra_speed0(
        cell.w,
        cell.h,
        cell.bd,
        cell.mono,
        cell.ss_x,
        cell.ss_y,
        cell.cq_level,
    );
    cfg.cpu_used = cell.speed;
    cfg.enable_cdef = false;
    cfg.enable_restoration = true;
    cfg
}

fn c_encode(cell: &EncodeCell) -> Vec<u8> {
    c::ref_encode_av1_kf(
        &cell.y,
        &cell.u,
        &cell.v,
        cell.w,
        cell.h,
        i32::from(cell.bd),
        cell.mono,
        cell.ss_x as i32,
        cell.ss_y as i32,
        cell.cq_level,
        cell.speed,
        false, // enable_cdef      — the ALLINTRA default
        true,  // enable_restoration
        cell.usage,
        0,
        false,
    )
}

fn port_encode(cell: &EncodeCell, cfg: &KeyFrameConfig) -> Vec<u8> {
    encode_key_frame(
        KeyFramePlanes {
            y: &cell.y,
            u: &cell.u,
            v: &cell.v,
        },
        cfg,
    )
    .expect("the port must encode this cell")
}

struct Row {
    label: String,
    px: usize,
    port_ms: f64,
    c_ms: f64,
    null_ms: f64,
    bytes: usize,
    identical: bool,
}

/// THE measurement. Prints a table and asserts the RD claim on every cell; the
/// timing is REPORTED rather than gated, because a wall-time bar on hardware
/// this project does not choose is the mistake KB-48 already paid for.
#[test]
#[ignore = "TIMING GATE — run isolated via `just gate-encode-perf`"]
fn standalone_encode_time_against_libaom() {
    c::ref_init();

    // Real photographic content, cropped from the conformance corpus, at the
    // sizes a still-image backend actually sees. Speeds span the range a caller
    // would pick: 0 (the parity anchor), 6 (the fleet witnesses' neighbourhood)
    // and 9 (the fast path).
    let mut cells = Vec::new();
    for &(w, h) in &[(128usize, 128usize), (192, 192)] {
        for &speed in &[0i32, 6, 9] {
            for &cq in &[27i32, 45] {
                cells.push(EncodeCell::real_content(
                    &format!("photo_{w}x{h}_cq{cq}_s{speed}"),
                    "av1-1-b8-01-size-196x196",
                    Some((w, h, 0, 0)),
                    cq,
                    speed,
                ));
            }
        }
    }

    // `--cpu-used` >= 7 above roughly 3x3 superblocks is an ALREADY-PINNED
    // divergence, not a finding of this file: `self_contained_key_frame.rs`'s
    // pin table records *"at speed 9, 192x192 is not [byte-exact] either"* for
    // the unlocalized VAR_BASED_PARTITION / nonrd arm (`PIN_256x256_speed7`).
    // Those rows are TIMED and REPORTED — speed 9 is the fastest preset and the
    // most interesting throughput number — but their ratios compare different
    // work, so they are excluded from the RD assertion and labelled as such.
    let rd_pinned = |label: &str| label.contains("_s9");

    let mut rows = Vec::new();
    for cell in &cells {
        let cfg = cfg_for(cell);
        // Warm both arms once, untimed: the first encode pays page faults and
        // any one-time table init, which is not what a caller experiences.
        let warm_port = port_encode(cell, &cfg);
        let warm_c = c_encode(cell);
        let identical = warm_port == warm_c;

        let (mut best_port, mut best_c, mut best_null) = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..REPS {
            // INTERLEAVED, and with a same-arm null so the ratio has a visible
            // floor: port, C, port-again. The two port timings differ only by
            // machine noise.
            let t = Instant::now();
            let a = port_encode(cell, &cfg);
            let p1 = ms(t.elapsed());

            let t = Instant::now();
            let _ = c_encode(cell);
            let cm = ms(t.elapsed());

            let t = Instant::now();
            let b = port_encode(cell, &cfg);
            let p2 = ms(t.elapsed());
            assert_eq!(a, b, "{}: the port is not deterministic", cell.label);

            best_port = best_port.min(p1);
            best_c = best_c.min(cm);
            best_null = best_null.min((p1 - p2).abs());
        }
        rows.push(Row {
            label: cell.label.clone(),
            px: cell.w * cell.h,
            port_ms: best_port,
            c_ms: best_c,
            null_ms: best_null,
            bytes: warm_c.len(),
            identical,
        });
    }

    println!(
        "\n{:<26} {:>9} {:>10} {:>10} {:>7} {:>8} {:>6}",
        "cell", "pixels", "port ms", "libaom ms", "ratio", "null ms", "bytes"
    );
    let mut worst = 0.0f64;
    let mut worst_label = String::new();
    let mut diverged = Vec::new();
    for r in &rows {
        let ratio = r.port_ms / r.c_ms.max(1e-9);
        if ratio > worst {
            worst = ratio;
            worst_label = r.label.clone();
        }
        if !r.identical {
            diverged.push(r.label.clone());
        }
        println!(
            "{:<26} {:>9} {:>10.2} {:>10.2} {:>6.2}x {:>8.2} {:>6}{}",
            r.label,
            r.px,
            r.port_ms,
            r.c_ms,
            ratio,
            r.null_ms,
            r.bytes,
            if r.identical { "" } else { "  BYTES DIFFER" }
        );
    }
    let worst_rd: (f64, &str) = rows
        .iter()
        .filter(|r| r.identical)
        .fold((0.0, ""), |acc, r| {
            let x = r.port_ms / r.c_ms.max(1e-9);
            if x > acc.0 { (x, &r.label) } else { acc }
        });
    println!(
        "\nworst ratio over BYTE-IDENTICAL cells: {:.2}x at {}\n\
         worst ratio over all cells:            {worst:.2}x at {worst_label}\n\
         {} of {} cells byte-identical ({} excluded by the `--cpu-used >= 7` pin)",
        worst_rd.0,
        worst_rd.1,
        rows.len() - diverged.len(),
        rows.len(),
        diverged.len()
    );

    // THE RD CLAIM, asserted on every cell that is not already pinned. Byte
    // identity means the port reached the same rate-distortion point C did, not
    // a nearby one.
    let unexpected: Vec<&String> = diverged.iter().filter(|l| !rd_pinned(l)).collect();
    assert!(
        unexpected.is_empty(),
        "these cells are NOT byte-identical to libaom and are NOT covered by the \
         `--cpu-used >= 7` pin, so their timing rows compare different work and \
         the RD claim does not hold: {unexpected:?}"
    );
    // ...and the pinned rows must STILL diverge, so the exclusion cannot go
    // stale: if they start matching, they belong in the asserted set.
    let pinned_matching: Vec<&String> = rows
        .iter()
        .filter(|r| rd_pinned(&r.label) && r.identical)
        .map(|r| &r.label)
        .collect();
    assert!(
        pinned_matching.is_empty(),
        "these `--cpu-used 9` cells are now byte-identical — the pin closed, so \
         move them into the asserted set: {pinned_matching:?}"
    );

    // The timing is REPORTED, not gated — see the module doc. What IS asserted
    // is that the measurement is meaningful: the null must be small beside the
    // gap being measured, else the ratio is noise.
    for r in rows.iter().filter(|r| r.identical) {
        let gap = (r.port_ms - r.c_ms).abs();
        assert!(
            r.null_ms <= gap.max(r.c_ms * 0.5),
            "{}: the same-arm null ({:.2} ms) is not small beside the port/C gap \
             ({:.2} ms) — this cell's ratio is not resolvable on this machine",
            r.label,
            r.null_ms,
            gap
        );
    }
}
