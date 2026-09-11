//! **encbench** — the zenbench cross-encoder comparison for AV1 still images.
//!
//! Four encoders, one process, interleaved round-robin:
//!
//! | arm | what it is |
//! |---|---|
//! | `zenav1-aom` | this repo's `encode_key_frame` — the SHIPPING path zenavif calls |
//! | `zenav1-svt` | the pure-Rust SVT-AV1 port, sibling repo at `main` |
//! | `ravif` | **crates.io** `ravif` (the `cavif` engine over upstream rav1e) |
//! | `zenrav1e` | the in-house rav1e fork, for continuity with the older xbench tables |
//!
//! **Why zenbench rather than four sequential timing loops.** The arms differ
//! by more than an order of magnitude in cost, so run back to back each one
//! sees a differently-warmed CPU and a different turbo state — the exact
//! confound zenbench's interleaved round-robin removes by construction. Every
//! round runs all four in shuffled order, and the statistics are paired on the
//! round-by-round differences.
//!
//! **Every arm is single-threaded.** `zenav1-aom` spawns no threads at all;
//! the other three are pinned to one.
//!
//! **What this chart does NOT say.** It is a SPEED comparison at each
//! encoder's own quality setting, and those settings are not interchangeable:
//! `ravif` takes a 1..100 quality, rav1e a 0..255 quantizer, the other two a
//! 0..63 index. Speed without rate and quality beside it is half an answer —
//! the RD half is `scripts/xbench.py stage2`, which scores every arm through
//! ONE decoder (`xtool decode` + `score-rgb`) precisely because the arms do
//! not agree on chroma format or bit depth either (`ravif` codes 4:4:4 and
//! defaults to 10-bit; the rest code 8-bit 4:2:0).
//!
//! Usage: `encbench <in.yuv> <w> <h> [--md out.md]`

use ravif::{BitDepth, ColorModel, Encoder as RavifEncoder};
use std::time::Duration;
use svtav1_encoder::pipeline::EncodePipeline;
use svtav1_encoder::rate_control::{RcConfig, RcMode};
use zenrav1e::prelude::*;

/// The quality points each arm is run at. Chosen so the coded sizes land in
/// the same neighbourhood on the study image — NOT so the numbers match, which
/// they cannot (four different scales). Re-fit them if the corpus changes, and
/// say so wherever the chart is published.
const AOM_CQ: i32 = 32;
const SVT_QP: u8 = 32;
const RAV1E_QUANTIZER: usize = 130;
const RAVIF_QUALITY: f32 = 70.0;
const SPEED_AOM: i32 = 3;
const SPEED_SVT: u8 = 6;
const SPEED_RAV1E: u8 = 6;
const SPEED_RAVIF: u8 = 6;

struct Source {
    w: usize,
    h: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// BT.709 limited-range RGB — the SAME conversion `xtool yuv2rgb` performs
    /// to build the scoring reference, so `ravif` is handed the pixels every
    /// other arm's stream decodes to.
    rgba: Vec<rgb::RGBA8>,
}

fn load(path: &str, w: usize, h: usize) -> Source {
    let buf = std::fs::read(path).expect("read .yuv");
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(buf.len(), w * h + 2 * cw * ch, "{path}: not an {w}x{h} I420");
    let y = buf[..w * h].to_vec();
    let u = buf[w * h..w * h + cw * ch].to_vec();
    let v = buf[w * h + cw * ch..].to_vec();
    let mut rgba = vec![rgb::RGBA8::new(0, 0, 0, 255); w * h];
    for r in 0..h {
        for c in 0..w {
            let yy = (f64::from(y[r * w + c]) - 16.0) * 255.0 / 219.0;
            let ci = (r / 2) * cw + (c / 2);
            let cb = (f64::from(u[ci]) - 128.0) * 255.0 / 224.0;
            let cr = (f64::from(v[ci]) - 128.0) * 255.0 / 224.0;
            let rr = yy + 1.574_8 * cr;
            let bb = yy + 1.855_6 * cb;
            let gg = (yy - 0.212_6 * rr - 0.072_2 * bb) / 0.715_2;
            rgba[r * w + c] = rgb::RGBA8::new(
                rr.round().clamp(0.0, 255.0) as u8,
                gg.round().clamp(0.0, 255.0) as u8,
                bb.round().clamp(0.0, 255.0) as u8,
                255,
            );
        }
    }
    Source { w, h, y, u, v, rgba }
}

fn enc_aom(s: &Source) -> usize {
    use aom_encode::key_frame::{encode_key_frame, KeyFrameConfig, KeyFramePlanes};
    let y: Vec<u16> = s.y.iter().map(|&b| u16::from(b)).collect();
    let u: Vec<u16> = s.u.iter().map(|&b| u16::from(b)).collect();
    let v: Vec<u16> = s.v.iter().map(|&b| u16::from(b)).collect();
    let mut cfg = KeyFrameConfig::allintra_speed0(s.w, s.h, 8, false, 1, 1, AOM_CQ);
    cfg.cpu_used = SPEED_AOM;
    cfg.enable_restoration = true;
    encode_key_frame(KeyFramePlanes { y: &y, u: &u, v: &v }, &cfg)
        .expect("zenav1-aom encode")
        .len()
}

fn enc_svt(s: &Source) -> usize {
    let rc = RcConfig { mode: RcMode::Cqp, qp: SVT_QP, ..RcConfig::default() };
    let mut p = EncodePipeline::new(s.w as u32, s.h as u32, SPEED_SVT, rc, 0, 1)
        .with_bit_depth(8)
        .with_tile_rows_log2(0)
        .with_tile_cols_log2(0)
        .with_sb_size(None)
        .with_chroma_420(true)
        .with_thread_count(1);
    p.encode_frame_420(&s.y, &s.u, &s.v, s.w).len()
}

fn enc_rav1e(s: &Source) -> usize {
    let enc = EncoderConfig {
        width: s.w,
        height: s.h,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        pixel_range: PixelRange::Limited,
        still_picture: true,
        min_key_frame_interval: 0,
        max_key_frame_interval: 1,
        low_latency: true,
        quantizer: RAV1E_QUANTIZER,
        tiles: 1,
        speed_settings: SpeedSettings::from_preset(SPEED_RAV1E),
        ..Default::default()
    };
    let cfg = Config::new().with_encoder_config(enc).with_threads(1);
    let mut ctx: Context<u8> = cfg.new_context().expect("rav1e context");
    let (cw, _) = (s.w / 2, s.h / 2);
    let mut f = ctx.new_frame();
    for (pi, p) in f.planes.iter_mut().enumerate() {
        let (src, pw): (&[u8], usize) = match pi {
            0 => (&s.y, s.w),
            1 => (&s.u, cw),
            _ => (&s.v, cw),
        };
        p.copy_from_raw_u8(src, pw, 1);
    }
    ctx.send_frame(f).expect("send_frame");
    ctx.flush();
    let mut n = 0usize;
    loop {
        match ctx.receive_packet() {
            Ok(pkt) => n += pkt.data.len(),
            Err(EncoderStatus::LimitReached) | Err(EncoderStatus::NeedMoreData) => break,
            Err(EncoderStatus::Encoded) => {}
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
    n
}

fn enc_ravif(s: &Source) -> usize {
    let e = RavifEncoder::new()
        .with_quality(RAVIF_QUALITY)
        .with_speed(SPEED_RAVIF)
        .with_num_threads(Some(1))
        .with_bit_depth(BitDepth::Auto) // ravif's own default (10-bit)
        .with_internal_color_model(ColorModel::YCbCr);
    e.encode_rgba(imgref::Img::new(&s.rgba[..], s.w, s.h))
        .expect("ravif encode")
        .color_byte_size
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: encbench <in.yuv> <w> <h> [--md <out.md>]");
        std::process::exit(2);
    }
    let (path, w, h) = (a[1].clone(), a[2].parse().unwrap(), a[3].parse().unwrap());
    let md_out = a.iter().position(|x| x == "--md").and_then(|i| a.get(i + 1)).cloned();

    // Sizes first, printed OUTSIDE the benchmark: a speed chart with no rate
    // beside it invites exactly the misreading this harness exists to avoid.
    // Leaked deliberately: zenbench's closures are `'static`, and this binary
    // is one measurement run that exits immediately after.
    let src: &'static Source = Box::leak(Box::new(load(&path, w, h)));
    let sizes = [
        ("zenav1-aom", enc_aom(src)),
        ("zenav1-svt", enc_svt(src)),
        ("ravif", enc_ravif(src)),
        ("zenrav1e", enc_rav1e(src)),
    ];
    println!("# coded AV1 payload bytes at the configured quality points");
    for (n, b) in &sizes {
        println!("#   {n:<12} {b:>9}");
    }
    println!("#   source {w}x{h} = {} px\n", w * h);

    let group_name = format!("av1_still_encode_{w}x{h}");
    let result = zenbench::run(|suite| {
        suite.compare(&group_name, |group| {
            // Each arm is 50 ms - 3 s per call, so a handful of rounds is
            // already minutes. zenbench stops early once the estimate is
            // precise enough; this is the ceiling, not the target.
            group.config().max_rounds(12).max_time(Duration::from_secs(900));
            group.bench("zenav1-aom", |b| b.iter(|| zenbench::black_box(enc_aom(src))));
            group.bench("zenav1-svt", |b| b.iter(|| zenbench::black_box(enc_svt(src))));
            group.bench("ravif", |b| b.iter(|| zenbench::black_box(enc_ravif(src))));
            group.bench("zenrav1e", |b| b.iter(|| zenbench::black_box(enc_rav1e(src))));
        });
    });

    if let Some(p) = md_out {
        use zenbench::quickchart::QuickChartConfig;
        let md = result.to_quickchart_markdown(&QuickChartConfig::default());
        std::fs::write(&p, md).expect("write markdown");
        eprintln!("wrote {p}");
    }
}
