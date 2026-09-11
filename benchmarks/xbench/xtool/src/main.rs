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
        die("usage: xtool <prep|ivf|score|yuv2rgb|decode|score-rgb> ...");
    }
    match a[1].as_str() {
        "prep" => cmd_prep(&a[2..]),
        "ivf" => cmd_ivf(&a[2..]),
        "score" => cmd_score(&a[2..]),
        "yuv2rgb" => cmd_yuv2rgb(&a[2..]),
        "decode" => cmd_decode(&a[2..]),
        "score-rgb" => cmd_score_rgb(&a[2..]),
        other => die(&format!("unknown subcommand {other}")),
    }
}

// ------------------------------------------------- format-neutral scoring ---
//
// The arms in this harness do NOT agree on chroma format or bit depth —
// `ravif` codes 4:4:4 and defaults to 10-bit while every AV1-payload driver
// codes 8-bit 4:2:0 — so a YUV-plane comparison cannot be written at all, let
// alone written fairly. These three subcommands put every arm through ONE
// decoder and score in RGB, which is the only space they share.
//
//   yuv2rgb  <in.yuv> <out.rgb> <w> <h>   the REFERENCE (what each encoder saw)
//   decode   <in.obu|in.avif> <out.rgb>   any arm's output -> 8-bit RGB
//   score-rgb <ref.rgb> <dist.rgb> <w> <h>
//
// `decode` reads the matrix coefficients and range the STREAM signals rather
// than assuming them, because the arms disagree there too: ravif writes BT.601
// for its YCbCr model, and an Identity/GBR stream is not YCbCr at all.

/// Extract the AV1 payload from an AVIF file's `mdat`, or pass a raw OBU
/// stream through unchanged. A minimal ISOBMFF top-level box walk — enough for
/// the single-item, no-alpha files this harness produces, and it fails loudly
/// rather than guessing on anything else.
fn av1_payload(bytes: Vec<u8>) -> Vec<u8> {
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return bytes; // already a bare OBU stream
    }
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let size = u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        let kind = &bytes[i + 4..i + 8];
        let (body, next) = match size {
            0 => (i + 8, bytes.len()),              // to end of file
            1 => die("64-bit box sizes are not handled"),
            n if n < 8 => die("corrupt box size"),
            n => (i + 8, i + n),
        };
        if kind == b"mdat" {
            return bytes[body..next.min(bytes.len())].to_vec();
        }
        if next <= i {
            die("box walk made no progress");
        }
        i = next;
    }
    die("no mdat box found in the AVIF file")
}

/// Convert a decoded frame to 8-bit RGB using the CICP the stream signals.
fn frame_to_rgb(f: &aom_decode::frame::FrameDecode) -> Vec<u8> {
    let (w, h) = (f.width, f.height);
    let maxv = f64::from((1i32 << f.bit_depth) - 1);
    let scale = 255.0 / maxv;
    let mut out = vec![0u8; w * h * 3];
    // MC 0 = Identity (GBR), 1 = BT.709, 5/6 = BT.601, 2 = unspecified.
    // `xtool score`'s own reference conversion is BT.709 limited-range, so an
    // unspecified stream is read that way and the two agree by construction.
    let (kr, kb) = match f.matrix_coefficients {
        5 | 6 => (0.299_f64, 0.114_f64),  // BT.601
        _ => (0.212_6_f64, 0.072_2_f64),  // BT.709 / unspecified
    };
    let identity = f.matrix_coefficients == 0;
    for r in 0..h {
        for c in 0..w {
            let o = (r * w + c) * 3;
            if f.monochrome {
                let g = (f64::from(f.y[r * w + c]) * scale).round().clamp(0.0, 255.0) as u8;
                out[o] = g;
                out[o + 1] = g;
                out[o + 2] = g;
                continue;
            }
            let ci = (r >> f.subsampling_y) * f.width_uv + (c >> f.subsampling_x);
            let (yv, uv, vv) = (
                f64::from(f.y[r * w + c]) * scale,
                f64::from(f.u[ci]) * scale,
                f64::from(f.v[ci]) * scale,
            );
            let (rr, gg, bb) = if identity {
                // GBR: plane order is G, B, R (AV1's identity matrix).
                (vv, yv, uv)
            } else {
                let yy = if f.full_range { yv } else { (yv - 16.0) * 255.0 / 219.0 };
                let (cb, cr) = if f.full_range {
                    (uv - 128.0, vv - 128.0)
                } else {
                    ((uv - 128.0) * 255.0 / 224.0, (vv - 128.0) * 255.0 / 224.0)
                };
                let rr = yy + 2.0 * (1.0 - kr) * cr;
                let bb = yy + 2.0 * (1.0 - kb) * cb;
                let gg = (yy - kr * rr - kb * bb) / (1.0 - kr - kb);
                (rr, gg, bb)
            };
            out[o] = rr.round().clamp(0.0, 255.0) as u8;
            out[o + 1] = gg.round().clamp(0.0, 255.0) as u8;
            out[o + 2] = bb.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn cmd_yuv2rgb(args: &[String]) {
    if args.len() != 4 {
        die("usage: yuv2rgb <in.yuv> <out.rgb> <w> <h>");
    }
    let w: usize = args[2].parse().unwrap_or_else(|_| die("w"));
    let h: usize = args[3].parse().unwrap_or_else(|_| die("h"));
    let (y, u, v) = read_i420(Path::new(&args[0]), w, h);
    std::fs::write(&args[1], yuv420_to_rgb(&y, &u, &v, w, h))
        .unwrap_or_else(|e| die(&format!("write: {e}")));
    println!("W={w} H={h}");
}

fn cmd_decode(args: &[String]) {
    if args.len() != 2 {
        die("usage: decode <in.obu|in.avif> <out.rgb>");
    }
    let bytes = std::fs::read(&args[0]).unwrap_or_else(|e| die(&format!("read: {e}")));
    let payload = av1_payload(bytes);
    let f = aom_decode::frame::decode_frame_obus(&payload)
        .unwrap_or_else(|e| die(&format!("decode {}: {e}", args[0])));
    let rgb = frame_to_rgb(&f);
    std::fs::write(&args[1], rgb).unwrap_or_else(|e| die(&format!("write: {e}")));
    println!(
        "W={} H={} BD={} SS={}{} MC={} RANGE={}",
        f.width,
        f.height,
        f.bit_depth,
        f.subsampling_x,
        f.subsampling_y,
        f.matrix_coefficients,
        u8::from(f.full_range)
    );
}

fn cmd_score_rgb(args: &[String]) {
    if args.len() != 4 {
        die("usage: score-rgb <ref.rgb> <dist.rgb> <w> <h>");
    }
    let w: usize = args[2].parse().unwrap_or_else(|_| die("w"));
    let h: usize = args[3].parse().unwrap_or_else(|_| die("h"));
    let rd = std::fs::read(&args[0]).unwrap_or_else(|e| die(&format!("read ref: {e}")));
    let dd = std::fs::read(&args[1]).unwrap_or_else(|e| die(&format!("read dist: {e}")));
    if rd.len() != w * h * 3 || dd.len() != w * h * 3 {
        die(&format!(
            "expected {} bytes of RGB each, got {} / {}",
            w * h * 3,
            rd.len(),
            dd.len()
        ));
    }
    let to_ss = |b: &[u8]| -> imgref::ImgVec<[u8; 3]> {
        imgref::ImgVec::new(b.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(), w, h)
    };
    let ss2 = fast_ssim2::compute_ssimulacra2(to_ss(&rd).as_ref(), to_ss(&dd).as_ref())
        .unwrap_or_else(|e| die(&format!("ssimulacra2: {e:?}")));
    let to_ba = |b: &[u8]| -> butteraugli::ImgVec<butteraugli::RGB8> {
        butteraugli::ImgVec::new(
            b.chunks_exact(3)
                .map(|c| butteraugli::RGB8 { r: c[0], g: c[1], b: c[2] })
                .collect(),
            w,
            h,
        )
    };
    let bar = butteraugli::butteraugli(
        to_ba(&rd).as_ref(),
        to_ba(&dd).as_ref(),
        &butteraugli::ButteraugliParams::default(),
    )
    .unwrap_or_else(|e| die(&format!("butteraugli: {e:?}")));
    println!("SSIM2={ss2:.6} BA_MAX={:.6} BA_3N={:.6}", bar.score, bar.pnorm_3);
}
