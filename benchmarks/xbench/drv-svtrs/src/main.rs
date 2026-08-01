//! drv-svtrs — timed still-picture encode driver for **zenav1-svt**, the
//! pure-Rust SVT-AV1 still-picture port (presets 0-13).
//!
//! Uniform driver contract:
//!   drv <w> <h> <qp 0..63> <preset 0..13> <in.yuv> <out.obu> <warmup> <reps>
//!   stdout: `NS=<n> ... BYTES=<m>`
//!
//! Config mirrors the port's own byte-identity harness
//! (`zenav1-svt/rust/svtav1/examples/perf_encode.rs`): CQP, 8-bit, 4:2:0,
//! single tile, `hierarchical_levels = 0`, `intra_period = 1` (still/allintra),
//! SB size derived by C's own rule. Timed region = `encode_frame_420` on a
//! FRESH pipeline; `EncodePipeline::new` (the port's one-time setup, the
//! analogue of C `svt_av1_enc_init`) is excluded — matching the C driver, which
//! also excludes init.

use std::time::Instant;
use svtav1_encoder::pipeline::EncodePipeline;
use svtav1_encoder::rate_control::{RcConfig, RcMode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 9 {
        eprintln!("usage: drv-svtrs <w> <h> <qp 0..63> <preset 0..13> <in.yuv> <out.obu> <warmup> <reps>");
        std::process::exit(2);
    }
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let qp: u8 = a[3].parse().unwrap();
    let preset: u8 = a[4].parse().unwrap();
    let warmup: usize = a[7].parse().unwrap();
    let reps: usize = a[8].parse().unwrap();
    assert!(w % 2 == 0 && h % 2 == 0, "even dims only");

    let buf = std::fs::read(&a[5]).expect("read .yuv");
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(buf.len(), w * h + 2 * cw * ch, "I420 size mismatch");
    let y = buf[..w * h].to_vec();
    let u = buf[w * h..w * h + cw * ch].to_vec();
    let v = buf[w * h + cw * ch..].to_vec();

    let build = || {
        let rc = RcConfig {
            mode: RcMode::Cqp,
            qp,
            ..RcConfig::default()
        };
        EncodePipeline::new(w as u32, h as u32, preset, rc, 0, 1)
            .with_bit_depth(8)
            .with_tile_rows_log2(0)
            .with_tile_cols_log2(0)
            .with_sb_size(None)
            .with_chroma_420(true)
            .with_thread_count(1)
    };

    for _ in 0..warmup {
        let mut p = build();
        let _ = p.encode_frame_420(&y, &u, &v, w);
    }
    let mut samples = Vec::with_capacity(reps);
    let mut last = Vec::new();
    for _ in 0..reps {
        let mut p = build();
        let t = Instant::now();
        let obu = p.encode_frame_420(&y, &u, &v, w);
        samples.push(t.elapsed().as_nanos());
        last = obu;
    }
    std::fs::write(&a[6], &last).expect("write .obu");
    let mut line = String::new();
    for s in &samples {
        line.push_str(&format!("NS={s} "));
    }
    println!("{line}BYTES={}", last.len());
}
