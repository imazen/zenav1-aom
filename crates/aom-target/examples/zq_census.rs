//! zq_census — the registered phase-A census harness for the aom backend's
//! dependency-injected Zq target loop (benchmarks/zensim_zq_target_wave_2026-08-29.md).
//!
//! Injections: encode = `aomenc` CLI (libaom; quantizer pinned via
//! --min-q/--max-q on the CLI's 0..63 scale — the pure-Rust encoder later
//! exposes full qindex resolution), decode = `aomdec` CLI (y4m both ways,
//! one BT.601 FULL-RANGE matrix pair in-harness, gated by a near-lossless
//! roundtrip check that fails loud on matrix drift), judge = zensim
//! **Profile C** (folded-944, the frozen north-anchor bake — same judge
//! family as the avif census).
//!
//! Usage: zq_census <corpus.tsv path\tname\tclass> <targets csv> <max_encodes> <out.tsv>
use std::io::Write;
use std::process::Command;
use zenav1_aom_target::{search_target_qindex, TargetOptions};
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn rgb_to_y4m(px: &[u8], w: usize, h: usize, path: &str) {
    let mut buf = Vec::with_capacity(w * h * 3 + 128);
    buf.extend_from_slice(
        format!("YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C444 XCOLORRANGE=FULL\nFRAME\n").as_bytes(),
    );
    let (mut yp, mut up, mut vp) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
    for i in 0..w * h {
        let (r, g, b) = (px[3 * i] as f32, px[3 * i + 1] as f32, px[3 * i + 2] as f32);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        yp[i] = y.round().clamp(0.0, 255.0) as u8;
        up[i] = (128.0 + (b - y) / 1.772).round().clamp(0.0, 255.0) as u8;
        vp[i] = (128.0 + (r - y) / 1.402).round().clamp(0.0, 255.0) as u8;
    }
    buf.extend_from_slice(&yp);
    buf.extend_from_slice(&up);
    buf.extend_from_slice(&vp);
    std::fs::write(path, buf).expect("y4m write");
}

fn y4m_to_rgb(path: &str) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).expect("y4m read");
    let hdr_end = data.iter().position(|&b| b == b'\n').expect("y4m header");
    let hdr = std::str::from_utf8(&data[..hdr_end]).expect("y4m header utf8");
    let mut w = 0usize;
    let mut h = 0usize;
    for tok in hdr.split_whitespace() {
        if let Some(v) = tok.strip_prefix('W') { w = v.parse().unwrap_or(0); }
        if let Some(v) = tok.strip_prefix('H') { h = v.parse().unwrap_or(0); }
    }
    assert!(w > 0 && h > 0, "y4m dims");
    assert!(hdr.contains("C444"), "expected C444 y4m from aomdec, got: {hdr}");
    let frame_hdr = data[hdr_end + 1..].iter().position(|&b| b == b'\n').expect("FRAME") + hdr_end + 2;
    let n = w * h;
    let (yp, up, vp) = (
        &data[frame_hdr..frame_hdr + n],
        &data[frame_hdr + n..frame_hdr + 2 * n],
        &data[frame_hdr + 2 * n..frame_hdr + 3 * n],
    );
    let mut px = vec![0u8; n * 3];
    for i in 0..n {
        let (y, u, v) = (yp[i] as f32, up[i] as f32 - 128.0, vp[i] as f32 - 128.0);
        px[3 * i] = (y + 1.402 * v).round().clamp(0.0, 255.0) as u8;
        px[3 * i + 2] = (y + 1.772 * u).round().clamp(0.0, 255.0) as u8;
        px[3 * i + 1] = ((y - 0.299 * (y + 1.402 * v) - 0.114 * (y + 1.772 * u)) / 0.587)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    (px, w, h)
}

fn read_png(path: &str) -> (Vec<u8>, usize, usize) {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).expect("png open")));
    let mut reader = dec.read_info().expect("png info");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    let (w, h) = (info.width as usize, info.height as usize);
    let px = match info.color_type {
        png::ColorType::Rgb => buf[..w * h * 3].to_vec(),
        png::ColorType::Rgba => {
            let mut o = Vec::with_capacity(w * h * 3);
            for c in buf[..w * h * 4].chunks_exact(4) { o.extend_from_slice(&c[..3]); }
            o
        }
        other => panic!("unsupported png color type {other:?}"),
    };
    (px, w, h)
}

fn bytemuck_cast(v: &[u8]) -> &[[u8; 3]] {
    let (chunks, rest) = v.as_chunks::<3>();
    assert!(rest.is_empty(), "rgb byte length not divisible by 3");
    chunks
}

static BAKE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
fn bake_bytes() -> &'static [u8] { BAKE.get().expect("bake").as_slice() }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (corpus, targets_s, k_s, out_path) = (&a[1], &a[2], &a[3], &a[4]);
    let k: u8 = k_s.parse().expect("max_encodes");
    let targets: Vec<f64> = targets_s.split(',').map(|t| t.parse().expect("target")).collect();
    let bake_path = std::env::var("ZQ_BAKE").unwrap_or_else(|_| {
        "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin".into()
    });
    BAKE.set(std::fs::read(&bake_path).expect("bake read")).unwrap();
    let params = zensim::profile::ProfileParams::builder()
        .mlp(bake_bytes)
        .skip_score_mapping(true)
        .extrapolate_score(true)
        .extended_features(true)
        .compute_iw_features(true)
        .build();
    let params: &'static zensim::profile::ProfileParams = Box::leak(Box::new(params));
    let profile = ZensimProfile::Custom { params, name: "aom-zq-census" };
    let z = Zensim::new(profile);
    let tmp = std::env::var("ZQ_TMP").unwrap_or_else(|_| "/home/lilith/tmp/aomzq".into());
    std::fs::create_dir_all(&tmp).expect("tmp dir");
    let mut out = std::fs::File::create(out_path).expect("out tsv");
    writeln!(out, "image\tclass\ttarget\tqindex\tachieved\tencodes\tbytes").unwrap();

    for line in std::fs::read_to_string(corpus).expect("corpus").lines() {
        let mut f = line.split('\t');
        let (path, name, class) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        let (px, w, h) = read_png(path);
        let src_y4m = format!("{tmp}/{name}.y4m");
        rgb_to_y4m(&px, w, h, &src_y4m);
        let px3: &[[u8; 3]] = bytemuck_cast(&px);
        let ref_slice = RgbSlice::new(px3, w, h);

        // Roundtrip matrix gate at min-q (near-lossless): fails loud on drift.
        let (rt, gate_bytes) = trial_encode(&src_y4m, 0, &tmp, name);
        assert!(gate_bytes > 0, "gate encode produced no bytes");
        let maxd = px.iter().zip(rt.0.iter()).map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs()).max().unwrap();
        assert!(maxd <= 12, "{name}: roundtrip max diff {maxd} > 12 — matrix drift or encode fault");

        for &t in &targets {
            let mut last_bytes = 0u64;
            let opts = TargetOptions { min_qindex: 0, max_qindex: 63, tolerance: 0.0, max_encodes: k, qindex_start: None };
            let mut best_seen: Option<(f64, u8, u64)> = None;
            let r = search_target_qindex::<_, String>(t, &opts, |qi| {
                let (dec, nbytes) = trial_encode(&src_y4m, qi, &tmp, name);
                last_bytes = nbytes;
                let dec3: &[[u8; 3]] = bytemuck_cast(&dec.0);
                let dec_slice = RgbSlice::new(dec3, dec.1, dec.2);
                let v2 = z
                    .compute_folded720_append2_features(&ref_slice, &dec_slice)
                    .map_err(|e| format!("folded-944 failed: {e:?}"))?;
                let feats = v2.features();
                let sc = zensim::score_features_with_profile(profile, feats, w as u32, h as u32)
                    .map_err(|e| format!("forward failed: {e:?}"))?;
                let better = best_seen.map(|(bs, _, _)| (sc - t).abs() < (bs - t).abs()).unwrap_or(true);
                if better { best_seen = Some((sc, qi, nbytes)); }
                Ok(sc)
            })
            .expect("search failed");
            let (bs, bq, bb) = best_seen.expect("at least one trial");
            let _ = (r, last_bytes);
            writeln!(out, "{name}\t{class}\t{t}\t{bq}\t{bs:.3}\t{k}\t{bb}").unwrap();
            eprintln!("[zq_census] {name} t={t}: qindex {bq} achieved {bs:.2} bytes {bb}");
        }
    }
}

/// One aomenc→aomdec cycle at a pinned CLI quantizer; returns decoded RGB + bytes.
fn trial_encode(src_y4m: &str, q: u8, tmp: &str, name: &str) -> ((Vec<u8>, usize, usize), u64) {
    let ivf = format!("{tmp}/{name}_q{q}.ivf");
    let dec_y4m = format!("{tmp}/{name}_q{q}.y4m");
    let st = Command::new("aomenc")
        .args([
            // cq-level alone pins the operating point (min/max-q equal is
            // refused by aomenc); the knob only needs monotonicity.
            "--passes=1", "--end-usage=q", &format!("--cq-level={q}"),
            "--cpu-used=6", "--threads=1", "--limit=1", "--ivf",
            "-o", &ivf, src_y4m,
        ])
        .output()
        .expect("aomenc spawn");
    assert!(st.status.success(), "aomenc failed: {}", String::from_utf8_lossy(&st.stderr));
    let nbytes = std::fs::metadata(&ivf).expect("ivf meta").len();
    let st = Command::new("aomdec")
        .args(["-o", &dec_y4m, &ivf])
        .output()
        .expect("aomdec spawn");
    assert!(st.status.success(), "aomdec failed: {}", String::from_utf8_lossy(&st.stderr));
    let dec = y4m_to_rgb(&dec_y4m);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&dec_y4m);
    (dec, nbytes)
}
