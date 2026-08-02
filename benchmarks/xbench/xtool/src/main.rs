//! xtool — the shared, encoder-agnostic half of the cross-encoder AV1
//! still-picture benchmark (`benchmarks/xbench`).
//!
//! Everything here runs OUTSIDE any encoder's timed region, and every encoder
//! sees the identical bytes it produces:
//!
//!   * `prep`  — PNG -> (optionally Lanczos3-downscaled) 8-bit I420 `.yuv`.
//!               ONE converter, ONE downscaler, for all four encoders.
//!   * `ivf`   — wrap a section-5 OBU stream in an IVF container so `aomdec` /
//!               `dav1d` will take it, whichever encoder emitted it.
//!   * `score` — decode-side metrics. Both sides of the comparison are
//!               I420 -> RGB through the SAME inverse converter, so the score
//!               isolates CODEC loss (a lossless coder scores 100 / 0.0).
//!
//! Colour handling (identical for every encoder, stated because it changes the
//! numbers): BT.709, LIMITED range, 8-bit. RGB->YUV444 then a 2x2 box average
//! for chroma; YUV420->RGB replicates chroma (nearest). The encoders are given
//! no colour signalling and none is read back — the same matrix is applied on
//! both sides of every codec, so signalling cannot bias the comparison.

use std::path::Path;

fn die(msg: &str) -> ! {
    eprintln!("xtool: {msg}");
    std::process::exit(1)
}

// ---------------------------------------------------------------- colour ---

/// BT.709 limited-range RGB8 -> Y'CbCr 8-bit (444, before chroma decimation).
fn rgb_to_yuv444(rgb: &[u8], n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; n];
    let mut u = vec![0u8; n];
    let mut v = vec![0u8; n];
    for i in 0..n {
        let r = f64::from(rgb[i * 3]);
        let g = f64::from(rgb[i * 3 + 1]);
        let b = f64::from(rgb[i * 3 + 2]);
        // BT.709 luma coefficients, limited range (16..235 / 16..240).
        let yf = 0.212_6 * r + 0.715_2 * g + 0.072_2 * b;
        let cb = (b - yf) / 1.855_6;
        let cr = (r - yf) / 1.574_8;
        y[i] = (16.0 + yf * 219.0 / 255.0).round().clamp(0.0, 255.0) as u8;
        u[i] = (128.0 + cb * 224.0 / 255.0).round().clamp(0.0, 255.0) as u8;
        v[i] = (128.0 + cr * 224.0 / 255.0).round().clamp(0.0, 255.0) as u8;
    }
    (y, u, v)
}

/// I420 (8-bit, BT.709 limited) -> packed RGB8, chroma replicated (nearest).
fn yuv420_to_rgb(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let mut out = vec![0u8; w * h * 3];
    for r in 0..h {
        for c in 0..w {
            let yy = (f64::from(y[r * w + c]) - 16.0) * 255.0 / 219.0;
            let ci = (r / 2) * cw + (c / 2);
            let cb = (f64::from(u[ci]) - 128.0) * 255.0 / 224.0;
            let cr = (f64::from(v[ci]) - 128.0) * 255.0 / 224.0;
            let rr = yy + 1.574_8 * cr;
            let bb = yy + 1.855_6 * cb;
            let gg = (yy - 0.212_6 * rr - 0.072_2 * bb) / 0.715_2;
            let o = (r * w + c) * 3;
            out[o] = rr.round().clamp(0.0, 255.0) as u8;
            out[o + 1] = gg.round().clamp(0.0, 255.0) as u8;
            out[o + 2] = bb.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn read_i420(path: &Path, w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let buf = std::fs::read(path).unwrap_or_else(|e| die(&format!("read {path:?}: {e}")));
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let want = w * h + 2 * cw * ch;
    if buf.len() < want {
        die(&format!(
            "{path:?}: {} bytes, need {want} for {w}x{h} I420",
            buf.len()
        ));
    }
    let y = buf[..w * h].to_vec();
    let u = buf[w * h..w * h + cw * ch].to_vec();
    let v = buf[w * h + cw * ch..want].to_vec();
    (y, u, v)
}

// ------------------------------------------------------------------ prep ---

fn cmd_prep(args: &[String]) {
    // prep <in.png> <out.yuv> <mode>
    //   mode = native            — even-cropped source, no resample
    //        | crop:WxH          — CENTER crop to WxH, no resample
    //        | at:WxH+X+Y        — crop WxH at the EXPLICIT offset (X, Y), no
    //                              resample. Added for the issue-#5 SVT interop
    //                              corpus, which needs many distinct tiles per
    //                              source image rather than one centre crop.
    //        | square:N          — center square crop, then Lanczos3 to NxN
    //                              (DOWNSCALE ONLY — errors out on upscale)
    if args.len() != 3 {
        die("usage: prep <in.png> <out.yuv> <native|crop:WxH|at:WxH+X+Y|square:N>");
    }
    let img = image::open(&args[0]).unwrap_or_else(|e| die(&format!("open {}: {e}", args[0])));
    let mut rgb = img.to_rgb8();
    let mode = args[2].as_str();
    if let Some(n) = mode.strip_prefix("square:") {
        let n: u32 = n.parse().unwrap_or_else(|_| die("square:N"));
        let side = rgb.width().min(rgb.height());
        if n > side {
            die(&format!(
                "{}: square:{n} would UPSCALE a {}x{} source — refused",
                args[0],
                rgb.width(),
                rgb.height()
            ));
        }
        let (ox, oy) = ((rgb.width() - side) / 2, (rgb.height() - side) / 2);
        let sq = image::imageops::crop_imm(&rgb, ox, oy, side, side).to_image();
        rgb = if side == n {
            sq
        } else {
            image::imageops::resize(&sq, n, n, image::imageops::FilterType::Lanczos3)
        };
    } else if let Some(wh) = mode.strip_prefix("crop:") {
        let (cw, chh) = wh.split_once('x').unwrap_or_else(|| die("crop:WxH"));
        let cw: u32 = cw.parse().unwrap_or_else(|_| die("crop W"));
        let chh: u32 = chh.parse().unwrap_or_else(|_| die("crop H"));
        if cw > rgb.width() || chh > rgb.height() {
            die(&format!("{}: crop {cw}x{chh} exceeds source", args[0]));
        }
        let (ox, oy) = ((rgb.width() - cw) / 2, (rgb.height() - chh) / 2);
        rgb = image::imageops::crop_imm(&rgb, ox, oy, cw, chh).to_image();
    } else if let Some(spec) = mode.strip_prefix("at:") {
        // at:WxH+X+Y
        let (wh, off) = spec.split_once('+').unwrap_or_else(|| die("at:WxH+X+Y"));
        let (ox, oy) = off.split_once('+').unwrap_or_else(|| die("at:WxH+X+Y"));
        let (cw, chh) = wh.split_once('x').unwrap_or_else(|| die("at:WxH+X+Y"));
        let cw: u32 = cw.parse().unwrap_or_else(|_| die("at W"));
        let chh: u32 = chh.parse().unwrap_or_else(|_| die("at H"));
        let ox: u32 = ox.parse().unwrap_or_else(|_| die("at X"));
        let oy: u32 = oy.parse().unwrap_or_else(|_| die("at Y"));
        if ox + cw > rgb.width() || oy + chh > rgb.height() {
            die(&format!(
                "{}: crop {cw}x{chh}+{ox}+{oy} exceeds the {}x{} source",
                args[0],
                rgb.width(),
                rgb.height()
            ));
        }
        rgb = image::imageops::crop_imm(&rgb, ox, oy, cw, chh).to_image();
    } else if mode != "native" {
        die(&format!("unknown prep mode {mode}"));
    }
    // AV1 4:2:0 wants even dims here (the harness never exercises odd-dim
    // partial chroma) — crop the last row/col rather than pad.
    let (mut w, mut h) = (rgb.width() as usize, rgb.height() as usize);
    w -= w % 2;
    h -= h % 2;
    let src = rgb.as_raw();
    let sw = rgb.width() as usize;
    let mut packed = vec![0u8; w * h * 3];
    for r in 0..h {
        packed[r * w * 3..(r + 1) * w * 3].copy_from_slice(&src[r * sw * 3..r * sw * 3 + w * 3]);
    }
    let (y, u4, v4) = rgb_to_yuv444(&packed, w * h);
    let (cw, ch) = (w / 2, h / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            let idx = [
                (2 * r) * w + 2 * c,
                (2 * r) * w + 2 * c + 1,
                (2 * r + 1) * w + 2 * c,
                (2 * r + 1) * w + 2 * c + 1,
            ];
            let su: u32 = idx.iter().map(|&i| u32::from(u4[i])).sum();
            let sv: u32 = idx.iter().map(|&i| u32::from(v4[i])).sum();
            u[r * cw + c] = ((su + 2) / 4) as u8;
            v[r * cw + c] = ((sv + 2) / 4) as u8;
        }
    }
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    out.extend_from_slice(&y);
    out.extend_from_slice(&u);
    out.extend_from_slice(&v);
    std::fs::write(&args[1], &out).unwrap_or_else(|e| die(&format!("write {}: {e}", args[1])));
    println!("W={w} H={h} BYTES={}", out.len());
}

// ------------------------------------------------------------------- ivf ---

fn cmd_ivf(args: &[String]) {
    // ivf <in.obu> <out.ivf> <w> <h>
    if args.len() != 4 {
        die("usage: ivf <in.obu> <out.ivf> <w> <h>");
    }
    let data = std::fs::read(&args[0]).unwrap_or_else(|e| die(&format!("read: {e}")));
    let w: u16 = args[2].parse().unwrap_or_else(|_| die("w"));
    let h: u16 = args[3].parse().unwrap_or_else(|_| die("h"));
    let mut o = Vec::with_capacity(data.len() + 44);
    o.extend_from_slice(b"DKIF");
    o.extend_from_slice(&0u16.to_le_bytes()); // version
    o.extend_from_slice(&32u16.to_le_bytes()); // header length
    o.extend_from_slice(b"AV01");
    o.extend_from_slice(&w.to_le_bytes());
    o.extend_from_slice(&h.to_le_bytes());
    o.extend_from_slice(&30u32.to_le_bytes()); // rate
    o.extend_from_slice(&1u32.to_le_bytes()); // scale
    o.extend_from_slice(&1u32.to_le_bytes()); // frame count
    o.extend_from_slice(&0u32.to_le_bytes()); // unused
    o.extend_from_slice(&(data.len() as u32).to_le_bytes());
    o.extend_from_slice(&0u64.to_le_bytes()); // pts
    o.extend_from_slice(&data);
    std::fs::write(&args[1], &o).unwrap_or_else(|e| die(&format!("write: {e}")));
    println!("BYTES={}", data.len());
}

// ----------------------------------------------------------------- score ---

fn cmd_score(args: &[String]) {
    // score <ref.yuv> <dist.yuv> <w> <h>
    if args.len() != 4 {
        die("usage: score <ref.yuv> <dist.yuv> <w> <h>");
    }
    let w: usize = args[2].parse().unwrap_or_else(|_| die("w"));
    let h: usize = args[3].parse().unwrap_or_else(|_| die("h"));
    let (ry, ru, rv) = read_i420(Path::new(&args[0]), w, h);
    let (dy, du, dv) = read_i420(Path::new(&args[1]), w, h);
    let rrgb = yuv420_to_rgb(&ry, &ru, &rv, w, h);
    let drgb = yuv420_to_rgb(&dy, &du, &dv, w, h);

    let to_ss = |b: &[u8]| -> imgref::ImgVec<[u8; 3]> {
        imgref::ImgVec::new(b.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(), w, h)
    };
    let a = to_ss(&rrgb);
    let b = to_ss(&drgb);
    let ss2 = fast_ssim2::compute_ssimulacra2(a.as_ref(), b.as_ref())
        .unwrap_or_else(|e| die(&format!("ssimulacra2: {e:?}")));

    let to_ba = |bytes: &[u8]| -> butteraugli::ImgVec<butteraugli::RGB8> {
        butteraugli::ImgVec::new(
            bytes
                .chunks_exact(3)
                .map(|c| butteraugli::RGB8 {
                    r: c[0],
                    g: c[1],
                    b: c[2],
                })
                .collect(),
            w,
            h,
        )
    };
    let ba_r = to_ba(&rrgb);
    let ba_d = to_ba(&drgb);
    let bar = butteraugli::butteraugli(
        ba_r.as_ref(),
        ba_d.as_ref(),
        &butteraugli::ButteraugliParams::default(),
    )
    .unwrap_or_else(|e| die(&format!("butteraugli: {e:?}")));

    println!(
        "SSIM2={ss2:.6} BA_MAX={:.6} BA_3N={:.6}",
        bar.score, bar.pnorm_3
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        die("usage: xtool <prep|ivf|score> ...");
    }
    match a[1].as_str() {
        "prep" => cmd_prep(&a[2..]),
        "ivf" => cmd_ivf(&a[2..]),
        "score" => cmd_score(&a[2..]),
        other => die(&format!("unknown subcommand {other}")),
    }
}
