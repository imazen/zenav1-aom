//! `eprof_x86` for a REAL image: reads an 8-bit I420 `.yuv` (as written by
//! `xtool prep`) instead of the mirror-tiled conformance cell, and times
//! `encode_key_frame` vs the real-C shim identically. Optional 8th arg dumps
//! both streams to `<prefix>.port.obu` / `<prefix>.c.obu`.
//!
//! ```text
//! eprof_yuv <port|c> <w> <h> <cq> <speed> <reps> <in.yuv> [dump_prefix]
//! ```

use aom_bench::EncodeCell;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;
use std::time::Instant;

fn load_i420(path: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let need = w * h + 2 * cw * ch;
    assert_eq!(raw.len(), need, "{path}: expected {need} bytes for {w}x{h} i420");
    let y: Vec<u16> = raw[..w * h].iter().map(|&b| u16::from(b)).collect();
    let u: Vec<u16> = raw[w * h..w * h + cw * ch]
        .iter()
        .map(|&b| u16::from(b))
        .collect();
    let v: Vec<u16> = raw[w * h + cw * ch..].iter().map(|&b| u16::from(b)).collect();
    EncodeCell {
        label: path.to_string(),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2, // ALLINTRA — same as EncodeCell::real_content
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 8 {
        eprintln!("usage: eprof_yuv <port|c> <w> <h> <cq> <speed> <reps> <in.yuv> [dump_prefix]");
        std::process::exit(2);
    }
    let (arm, w, h, cq, speed, reps, yuv) = (
        a[1].clone(),
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<usize>().unwrap(),
        a[4].parse::<i32>().unwrap(),
        a[5].parse::<i32>().unwrap(),
        a[6].parse::<usize>().unwrap(),
        a[7].clone(),
    );

    c::ref_init();
    let cell = load_i420(&yuv, w, h, cq, speed);

    let mut cfg = KeyFrameConfig::allintra_speed0(
        cell.w, cell.h, cell.bd, cell.mono, cell.ss_x, cell.ss_y, cell.cq_level,
    );
    cfg.cpu_used = cell.speed;
    cfg.enable_cdef = false;
    cfg.enable_restoration = true;

    let run = |arm: &str| -> Vec<u8> {
        match arm {
            "port" => encode_key_frame(
                KeyFramePlanes::new(&cell.y, &cell.u, &cell.v),
                &cfg,
            )
            .expect("the port must encode this cell"),
            "c" => c::ref_encode_av1_kf_screen_content(
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
                cfg.enable_palette,
                cfg.enable_intrabc,
            ),
            other => panic!("unknown arm {other}"),
        }
    };

    let warm = run(&arm);
    if let Some(prefix) = a.get(8) {
        std::fs::write(format!("{prefix}.{arm}.obu"), &warm).unwrap();
    }

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
