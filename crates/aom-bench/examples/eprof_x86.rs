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
    let cell = EncodeCell::real_content(
        &format!("photo_{w}x{h}_cq{cq}_s{speed}"),
        "av1-1-b8-01-size-196x196",
        Some((w, h, 0, 0)),
        cq,
        speed,
    );

    let mut cfg = KeyFrameConfig::allintra_speed0(
        cell.w, cell.h, cell.bd, cell.mono, cell.ss_x, cell.ss_y, cell.cq_level,
    );
    cfg.cpu_used = cell.speed;
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
