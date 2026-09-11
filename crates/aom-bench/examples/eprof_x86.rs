//! Profiling driver for the ENCODER on Linux/x86-64 — the arm a `perf record`
//! wraps.
//!
//! Every encoder perf number and every ranked lever in this repo
//! (`benchmarks/encoder_hotspot_reprofile_2026-08-02.md`, KB-PERF-1..5) was
//! measured on **aarch64-apple-darwin** with macOS `sample`, which does not run
//! on Linux — `scripts/eprof_sample.sh` is Darwin-only by construction. Yet the
//! ratio the standing goal is judged against
//! (`benchmarks/encode_perf_vs_libaom_2026-09-08.md`, 3.24x-4.03x) is an x86-64
//! number. KB-PERF-2 measured that a lever's RANK does not survive a platform
//! change (its allocation lever was 21 % of the win on Darwin and 86-99 % on
//! Windows), so the Darwin ranking must not be spent here without a local
//! profile.
//!
//! This binary is that profile's subject: one cell, one arm, N encodes, no
//! table formatting or I/O inside the timed loop.
//!
//! ```text
//! cargo build --release -p zenav1-aom-bench --example eprof_x86
//! perf record -F 999 -g -- ./target/release/examples/eprof_x86 port 192 192 27 0 20
//! ```
//!
//! The `c` arm runs real libaom through the same shim the byte gates use, so a
//! single recording can carry both and their symbols stay distinguishable.

use aom_bench::EncodeCell;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;
use std::time::Instant;

/// Mirror-tile a small cell up to `w`x`h` — the same recipe as
/// `kb28_crop_dims::mirror_tile` / `s4cov_hd_speed_axis::mirror_tile`.
fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!("usage: eprof_x86 <port|c> <w> <h> <cq> <speed> <reps>");
        std::process::exit(2);
    }
    let (arm, w, h, cq, speed, reps) = (
        a[1].clone(),
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<usize>().unwrap(),
        a[4].parse::<i32>().unwrap(),
        a[5].parse::<i32>().unwrap(),
        a[6].parse::<usize>().unwrap(),
    );

    c::ref_init();
    // The source vector is 196x196, so anything larger is MIRROR-TILED from it
    // (the recipe every HD gate in this repo uses). Real sizes matter for a
    // perf measurement and not only for coverage: at 192x192 the bd8 u16 planes
    // are ~110 KiB and sit in L2, so memory traffic is free and a measurement
    // there sees only op count. A lever that halves traffic — any lane-width
    // change — cannot show its win in that regime.
    let base = EncodeCell::real_content(
        "eprof_base_196",
        "av1-1-b8-01-size-196x196",
        Some((196, 196, 0, 0)),
        cq,
        speed,
    );
    let cell = if w <= 196 && h <= 196 {
        EncodeCell::real_content(
            &format!("photo_{w}x{h}_cq{cq}_s{speed}"),
            "av1-1-b8-01-size-196x196",
            Some((w, h, 0, 0)),
            cq,
            speed,
        )
    } else {
        mirror_tile(&base, &format!("photo_{w}x{h}_cq{cq}_s{speed}"), w, h, cq, speed)
    };

    let mut cfg = KeyFrameConfig::allintra_speed0(
        cell.w, cell.h, cell.bd, cell.mono, cell.ss_x, cell.ss_y, cell.cq_level,
    );
    cfg.cpu_used = cell.speed;
    // Optional 7th arg: the tile grid, as `N` (square: cols = rows = N) or
    // `C,R` (independent log2s), so the tile-parallel ceiling and its bitstream
    // cost can be measured. The two spellings are NOT interchangeable for a
    // threading study: ROW tiles own contiguous row bands of the recon plane
    // and so are reachable with `split_at_mut` under `forbid(unsafe_code)`,
    // while COLUMN tiles interleave within every row and are not.
    if let Some(t) = std::env::args().nth(7) {
        let (c, r) = match t.split_once(',') {
            Some((c, r)) => (c.parse::<i32>().unwrap(), r.parse::<i32>().unwrap()),
            None => {
                let n = t.parse::<i32>().unwrap();
                (n, n)
            }
        };
        cfg.tile_columns_log2 = c;
        cfg.tile_rows_log2 = r;
    }
    cfg.enable_cdef = false;
    cfg.enable_restoration = true;

    let run = |arm: &str| -> Vec<u8> {
        match arm {
            "port" => encode_key_frame(
                KeyFramePlanes { y: &cell.y, u: &cell.u, v: &cell.v },
                &cfg,
            )
            .expect("the port must encode this cell"),
            "c" => c::ref_encode_av1_kf(
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
                false, // enable_cdef        — the ALLINTRA default
                true,  // enable_restoration — likewise
                cell.usage,
                0,
                false,
            ),
            other => panic!("unknown arm {other}"),
        }
    };

    // Warm once, untimed: the first encode pays page faults and one-time table
    // init, which is not what a caller experiences and not what we want to rank.
    let warm = run(&arm);

    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(run(&arm));
    }
    let el = t.elapsed().as_secs_f64() * 1e3;

    println!(
        "arm={arm} {w}x{h} cq{cq} s{speed} reps={reps} bytes={} total={el:.1}ms per={:.2}ms",
        warm.len(),
        el / reps as f64
    );
}
