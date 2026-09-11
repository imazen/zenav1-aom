//! drv-ravif — timed still-picture encode driver for the **crates.io `ravif`**
//! crate (the `cavif` engine, built on upstream `rav1e`).
//!
//! Deliberately NOT `~/work/zen/ravif` and NOT the `zenrav1e` fork that
//! `drv-rav1e` measures: this arm answers "what does a Rust developer get today
//! from `cargo add ravif`", which is a different question from "how does our
//! fork compare".
//!
//! Uniform driver contract, identical to every other driver in this harness:
//!
//!   drv <w> <h> <q> <speed> <in.yuv> <out.obu> <warmup> <reps>
//!   stdout: `NS=<n> NS=<n> ... BYTES=<m>`   (one NS per timed rep)
//!
//! **Three things differ from the AV1-payload drivers, and all three are
//! ravif's own design choices rather than harness artifacts. State them
//! wherever this arm is charted:**
//!
//! 1. **ravif always encodes 4:4:4** (`ChromaSampling::Cs444`,
//!    av1encoder.rs:414) — its docs call subsampling "a bad idea for AVIF
//!    anyway". Every other arm here codes 4:2:0. So ravif carries twice the
//!    chroma samples, which costs bytes it cannot recover here: the SOURCE is
//!    an I420 `.yuv`, so there is no extra chroma detail for 4:4:4 to keep.
//! 2. **ravif defaults to 10-bit output** (`BitDepth::Auto` -> `Ten`), again
//!    a quality-first default. `RAVIF_DEPTH=8` switches it, so the chart can
//!    carry both and the reader can see what the default costs.
//! 3. **`q` is ravif's 1..100 QUALITY scale**, not a quantizer index. The
//!    ladders are therefore not aligned by number and only an RD curve
//!    (size against a measured quality score) compares them honestly — which
//!    is why this harness scores every arm rather than charting bytes alone.
//!
//! Single-threaded by contract (`with_num_threads(Some(1))`), matching every
//! other driver. The timed region is the `encode_*` call only — never the
//! `.yuv` read, never the YUV->RGB conversion, never the file write.
//!
//! `BYTES` is `EncodedImage::color_byte_size`, i.e. the **AV1 payload**, not
//! the `.avif` file: the container is not what is under comparison. The full
//! AVIF file is still written to `out.obu` so a decode/score step can read it.

use ravif::{BitDepth, ColorModel, Encoder};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 9 {
        eprintln!(
            "usage: drv-ravif <w> <h> <quality 1..100> <speed 1..10> <in.yuv> <out.avif> \
             <warmup> <reps>"
        );
        eprintln!("env: RAVIF_DEPTH=8|10 (default 10 = ravif's own default)");
        std::process::exit(2);
    }
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let quality: f32 = a[3].parse().unwrap();
    let speed: u8 = a[4].parse().unwrap();
    let warmup: usize = a[7].parse().unwrap();
    let reps: usize = a[8].parse().unwrap();
    assert!(w % 2 == 0 && h % 2 == 0, "even dims only");

    // I420 in, exactly the byte layout every other driver reads.
    let buf = std::fs::read(&a[5]).expect("read .yuv");
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(
        buf.len(),
        w * h + 2 * cw * ch,
        "{}: expected an {w}x{h} I420 frame",
        a[5]
    );
    let (yp, rest) = buf.split_at(w * h);
    let (up, vp) = rest.split_at(cw * ch);

    // BT.709 limited-range YUV -> RGB, byte-for-byte the conversion
    // `xtool yuv2rgb` performs to build the scoring REFERENCE. That identity
    // is load-bearing, not cosmetic: it is what makes ravif's input the same
    // pixels every other arm's input decodes to, so the RD curves are
    // comparable. Change one and you must change the other.
    // Untimed — the harness times the encode call only.
    let mut rgb = vec![rgb::RGBA8::new(0, 0, 0, 255); w * h];
    for r in 0..h {
        for c in 0..w {
            let yy = (f64::from(yp[r * w + c]) - 16.0) * 255.0 / 219.0;
            let ci = (r / 2) * cw + (c / 2);
            let cb = (f64::from(up[ci]) - 128.0) * 255.0 / 224.0;
            let cr = (f64::from(vp[ci]) - 128.0) * 255.0 / 224.0;
            let rr = yy + 1.574_8 * cr;
            let bb = yy + 1.855_6 * cb;
            let gg = (yy - 0.212_6 * rr - 0.072_2 * bb) / 0.715_2;
            rgb[r * w + c] = rgb::RGBA8::new(
                rr.round().clamp(0.0, 255.0) as u8,
                gg.round().clamp(0.0, 255.0) as u8,
                bb.round().clamp(0.0, 255.0) as u8,
                255,
            );
        }
    }
    let img = imgref::Img::new(&rgb[..], w, h);

    let depth = match std::env::var("RAVIF_DEPTH").as_deref() {
        Ok("8") => BitDepth::Eight,
        Ok("10") => BitDepth::Ten,
        _ => BitDepth::Auto, // ravif's own default
    };
    let enc = Encoder::new()
        .with_quality(quality)
        .with_speed(speed)
        .with_num_threads(Some(1))
        .with_bit_depth(depth)
        .with_internal_color_model(ColorModel::YCbCr);

    for _ in 0..warmup {
        let _ = enc.encode_rgba(img).expect("ravif encode");
    }
    let mut ns = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        let out = enc.encode_rgba(img).expect("ravif encode");
        ns.push(t.elapsed().as_nanos());
        last = Some(out);
    }
    let out = last.expect("reps >= 1");
    std::fs::write(&a[6], &out.avif_file).expect("write .avif");

    let mut s = String::new();
    for n in &ns {
        s.push_str(&format!("NS={n} "));
    }
    s.push_str(&format!("BYTES={}", out.color_byte_size));
    println!("{s}");
}
