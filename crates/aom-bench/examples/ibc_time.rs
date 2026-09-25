//! IntraBC-path timing: `encode_key_frame` vs the C screen-tools oracle
//! (`c_encode_screen(palette, intrabc)`) on `Content::Screen` at rising frame
//! sizes, `--cpu-used 6`. KB-41's perf note recorded ~80 s for the port on a
//! 1080p screenshot vs ~1 s for the oracle; that figure predated its own
//! speed-feature fixes. At 2026-09-13 HEAD the port is ~2.5x C on the 1080p
//! cell and byte-identical on every size — the flat-arena `IntrabcHashTable`
//! and the `sad_u16_simd`/`variance_u16_simd` autoversioned kernels took it
//! 3.4x -> 2.5x.
//!
//! Usage: `cargo run --release -p zenav1-aom-bench --example ibc_time`

use aom_bench::EncodeCell;
use aom_bench::winperf::{Content, synth_i420};
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use std::time::Instant;

fn cell(w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let i420 = synth_i420(w, h, Content::Screen);
    let (cw, ch) = (w / 2, h / 2);
    let y: Vec<u16> = i420[..w * h].iter().map(|&b| u16::from(b)).collect();
    let u: Vec<u16> = i420[w * h..w * h + cw * ch]
        .iter()
        .map(|&b| u16::from(b))
        .collect();
    let v: Vec<u16> = i420[w * h + cw * ch..]
        .iter()
        .map(|&b| u16::from(b))
        .collect();
    EncodeCell {
        label: format!("screen_{w}x{h}_cq{cq}_s{speed}"),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

fn main() {
    println!("cell\tc_ms\tport_ms\tratio");
    for &(w, h) in &[
        (320usize, 180usize),
        (640, 360),
        (960, 540),
        (1280, 720),
        (1920, 1080),
    ] {
        for &cq in &[32i32] {
            for &speed in &[6i32] {
                let c = cell(w, h, cq, speed);
                let t0 = Instant::now();
                let cs = c.c_encode_screen(true, true).len();
                let c_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut cfg = KeyFrameConfig::allintra_speed0(w, h, 8, false, 1, 1, cq);
                cfg.cpu_used = speed;
                cfg.enable_restoration = true;
                cfg.enable_palette = true;
                cfg.enable_intrabc = true;
                let t0 = Instant::now();
                let ps = encode_key_frame(KeyFramePlanes::new(&c.y, &c.u, &c.v), &cfg)
                    .expect("encode")
                    .len();
                let p_ms = t0.elapsed().as_secs_f64() * 1e3;
                println!(
                    "{w}x{h} cq{cq} s{speed}\t{c_ms:.0}\t{p_ms:.0}\t{:.1}\tcs={cs} ps={ps}",
                    p_ms / c_ms.max(0.001)
                );
            }
        }
    }
}
