//! drv-rav1e — timed still-picture encode driver for **zenrav1e** (the rav1e
//! fork).
//!
//! Uniform driver contract:
//!   drv <w> <h> <quantizer 0..255> <speed 0..10> <in.yuv> <out.obu> <warmup> <reps>
//!   stdout: `NS=<n> ... BYTES=<m>`
//!
//! Config: `still_picture = true`, one KEY frame, 8-bit, 4:2:0, limited range,
//! `tiles = 1`, `Config::with_threads(1)` — single-threaded, matching every
//! other encoder in this harness. Timed region = `send_frame` + `flush` +
//! `receive_packet` (the whole encode of the one frame). Context construction
//! (`new_context`, the analogue of the other encoders' init) is excluded, and
//! so is reading the `.yuv` and building the `Frame`.

use std::time::Instant;
use zenrav1e::config::SpeedSettings;
use zenrav1e::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 9 {
        eprintln!("usage: drv-rav1e <w> <h> <quantizer 0..255> <speed 0..10> <in.yuv> <out.obu> <warmup> <reps>");
        std::process::exit(2);
    }
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let quantizer: usize = a[3].parse().unwrap();
    let speed: u8 = a[4].parse().unwrap();
    let warmup: usize = a[7].parse().unwrap();
    let reps: usize = a[8].parse().unwrap();
    assert!(w % 2 == 0 && h % 2 == 0, "even dims only");

    let buf = std::fs::read(&a[5]).expect("read .yuv");
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(buf.len(), w * h + 2 * cw * ch, "I420 size mismatch");
    let planes: [&[u8]; 3] = [
        &buf[..w * h],
        &buf[w * h..w * h + cw * ch],
        &buf[w * h + cw * ch..],
    ];

    let enc = EncoderConfig {
        width: w,
        height: h,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        pixel_range: PixelRange::Limited,
        still_picture: true,
        min_key_frame_interval: 0,
        max_key_frame_interval: 1,
        low_latency: true,
        quantizer,
        tiles: 1,
        speed_settings: SpeedSettings::from_preset(speed),
        ..Default::default()
    };
    let cfg = Config::new().with_encoder_config(enc).with_threads(1);

    let mk_frame = |ctx: &Context<u8>| {
        let mut f = ctx.new_frame();
        for (pi, p) in f.planes.iter_mut().enumerate() {
            let pw = if pi == 0 { w } else { cw };
            p.copy_from_raw_u8(planes[pi], pw, 1);
        }
        f
    };

    let encode_once = |ctx: &mut Context<u8>, f: Frame<u8>| -> Vec<u8> {
        ctx.send_frame(f).expect("send_frame");
        ctx.flush();
        let mut out = Vec::new();
        loop {
            match ctx.receive_packet() {
                Ok(pkt) => out.extend_from_slice(&pkt.data),
                Err(EncoderStatus::LimitReached) => break,
                Err(EncoderStatus::Encoded) => {}
                Err(EncoderStatus::NeedMoreData) => break,
                Err(e) => panic!("receive_packet: {e:?}"),
            }
        }
        out
    };

    for _ in 0..warmup {
        let mut ctx: Context<u8> = cfg.new_context().expect("new_context");
        let f = mk_frame(&ctx);
        let _ = encode_once(&mut ctx, f);
    }
    let mut samples = Vec::with_capacity(reps);
    let mut last = Vec::new();
    for _ in 0..reps {
        let mut ctx: Context<u8> = cfg.new_context().expect("new_context");
        let f = mk_frame(&ctx);
        let t = Instant::now();
        let out = encode_once(&mut ctx, f);
        samples.push(t.elapsed().as_nanos());
        last = out;
    }
    std::fs::write(&a[6], &last).expect("write .obu");
    let mut line = String::new();
    for s in &samples {
        line.push_str(&format!("NS={s} "));
    }
    println!("{line}BYTES={}", last.len());
}
