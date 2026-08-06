//! HIGH-BIT-DEPTH INTER DECODE ENVELOPE — the live C-oracle gate (issue #8).
//!
//! `crates/aom-decode/tests/animated_avif.rs` already gates the animated-AVIF
//! tracks, but against md5 STRINGS committed in 2026-07-23 — a golden, not an
//! oracle. Issue #8 reported that
//! `colors-animated-12bpc-keyframes-0-2-3.avif` frame 1 (an inter frame; the
//! file's keyframes are 0, 2, 3) decodes differently through zenav1-aom than
//! through rav1d-safe. Deciding which side is wrong needs the REAL C decoder,
//! decoding the SAME bytes in the SAME process — that is what this file does.
//!
//! Three parts:
//!
//! 1. [`animated_tracks_match_c_decoder`] — every committed animated track,
//!    every shown frame, port `decode_frames` vs `aom_codec_av1_dx`, sample by
//!    sample, plus a shown-frame COUNT agreement check. The issue-#8 vector's
//!    12-bit 4:2:2 color track and its 12-bit monochrome alpha track are in the
//!    table.
//! 2. [`highbd_key_p_envelope_vs_c`] — the bit-depth x subsampling sweep the
//!    8-bit-only `inter_harness_chunk0` envelope map never covered: real
//!    `aomenc` `[KEY, P]` clips at bd 8/10/12 x {4:2:0, 4:2:2, 4:4:4, mono},
//!    port-decoded and diffed against C frame by frame. Reports the whole grid
//!    and hard-asserts the measured state.
//! 3. [`highbd_nonzero_mv_fails_loud_not_wrong`] — the honest-boundary pin: at
//!    bd > 8 a nonzero MV is OUTSIDE the ported envelope (the sub-pel filter
//!    chain is still 8-bit), and the decoder must REFUSE such a stream rather
//!    than reconstruct it wrong. A silent wrong-pixel path here would be the
//!    exact defect issue #8 suspected.

use aom_bench::inter_localize::{Divergence, FrameView, SB64_PX, first_frameset_divergence};
use aom_bench::{EncodeCell, MultiFrameEncodeCell};
use aom_decode::frame::FrameDecode;

const ANIMATED: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../aom-decode/tests/data/animated"
);

/// Decode a stream with the port's multi-frame path, or return the error text
/// (a panic is caught and reported as one — an unimplemented feature must not
/// take the harness down).
fn port_frames_result(stream: &[u8]) -> Result<Vec<FrameDecode>, String> {
    aom_bench::inter_localize::try_decode_frames(stream)
}

/// C-decode every shown frame of `stream` at `(w, h)`, probing upward until the
/// shim reports "fewer shown frames" — so the returned length IS C's shown-frame
/// count, independently of what the port thinks.
fn c_frames(stream: &[u8], w: usize, h: usize) -> Vec<aom_sys_ref::RefDecodedFrame> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(f) = aom_sys_ref::ref_decode_av1_stream_frame_opt(stream, i, w, h) {
        out.push(f);
        i += 1;
        assert!(i < 64, "runaway frame probe");
    }
    out
}

// ---------------------------------------------------------------------------
// 1. The issue-#8 corpus: port vs the REAL C decoder, not vs a stored md5
// ---------------------------------------------------------------------------

/// `(fixture stem, coded width, coded height)`. Dims are pinned here rather
/// than taken from the port, so a port geometry error cannot hide: the C shim
/// returns rc 4 (panic) when the decoded frame does not match `expect_w/h`.
const ANIMATED_TRACKS: &[(&str, usize, usize)] = &[
    ("colors-animated-8bpc.color", 150, 150),
    ("colors-animated-8bpc-audio.color", 150, 150),
    ("colors-animated-8bpc-alpha-exif-xmp.color", 150, 150),
    ("colors-animated-8bpc-alpha-exif-xmp.alpha", 150, 150),
    ("colors-animated-8bpc-depth-exif-xmp.color", 150, 150),
    ("colors-animated-8bpc-depth-exif-xmp.alpha", 150, 150),
    // The issue-#8 vector: 12-bit 4:2:2 color + 12-bit monochrome alpha.
    ("colors-animated-12bpc-keyframes-0-2-3.color", 64, 64),
    ("colors-animated-12bpc-keyframes-0-2-3.alpha", 64, 64),
];

#[test]
fn animated_tracks_match_c_decoder() {
    aom_sys_ref::ref_init();
    println!("\n=== animated-AVIF tracks: port decode_frames vs aom_codec_av1_dx ===");
    println!("track | bd | ss | frames | verdict");
    let mut failures: Vec<String> = Vec::new();
    for &(track, w, h) in ANIMATED_TRACKS {
        let stream = std::fs::read(format!("{ANIMATED}/{track}.obu"))
            .unwrap_or_else(|e| panic!("fixture {track}.obu missing: {e}"));
        let pf = match port_frames_result(&stream) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{track}: port decode failed: {e}"));
                println!("{track} | - | - | - | PORT DECODE ERROR: {e}");
                continue;
            }
        };
        let cf = c_frames(&stream, w, h);
        let pv: Vec<FrameView> = pf.iter().map(FrameView::of_decode).collect();
        let cv: Vec<FrameView> = cf.iter().map(FrameView::of_ref_decoded).collect();
        let div = first_frameset_divergence(&pv, &cv, SB64_PX);
        let (bd, ss) = pf
            .first()
            .map(|f| {
                (
                    f.bit_depth,
                    if f.monochrome {
                        "mono".to_string()
                    } else {
                        format!("{}{}", 4 - 2 * f.subsampling_x, 2 - 2 * f.subsampling_y)
                    },
                )
            })
            .unwrap_or((0, "-".into()));
        let verdict = match &div {
            None => format!("byte-exact ({} shown frames)", pf.len()),
            Some(d) => d.to_string(),
        };
        println!("{track} | {bd} | {ss} | {} | {verdict}", pf.len());
        // The coded CICP + geometry, for the record: these are the inputs a
        // caller's YUV->RGB conversion runs on, and they are identical for
        // every decoder that parses the same sequence header.
        if let Some(f) = pf.first() {
            println!(
                "    coded: {}x{} luma, {}x{} chroma, mc={} cp={} tc={} full_range={} csp={}",
                f.width,
                f.height,
                f.width_uv,
                f.height_uv,
                f.matrix_coefficients,
                f.color_primaries,
                f.transfer_characteristics,
                f.full_range,
                f.chroma_sample_position,
            );
        }
        if let Some(d) = div {
            failures.push(format!("{track}: {d}"));
        }
    }
    assert!(
        failures.is_empty(),
        "animated tracks diverged from the C decoder: {failures:#?}"
    );
    println!(
        "FINDING (issue #8): every shown frame of every animated track — including the 12-bit\n\
         4:2:2 color track and the 12-bit monochrome alpha track of\n\
         colors-animated-12bpc-keyframes-0-2-3.avif — reconstructs BYTE-IDENTICALLY to the real\n\
         libaom C decoder, on the SAME bytes, in-process. The port's plane output for that vector\n\
         is not the divergent side."
    );
}

// ---------------------------------------------------------------------------
// 2. bit-depth x subsampling envelope on real aomenc [KEY, P] clips
// ---------------------------------------------------------------------------

/// Depth-scaled textured content: a smooth ramp plus a fine checker so the
/// transform has real work, spanning most of `bd`'s range.
#[allow(clippy::too_many_arguments)]
fn hbd_base(
    label: &str,
    w: usize,
    h: usize,
    bd: u8,
    ss_x: usize,
    ss_y: usize,
    mono: bool,
    cq: i32,
) -> EncodeCell {
    let max = (1u32 << bd) - 1;
    let scale = |v: u32| -> u16 { ((v * max) / 255).min(max) as u16 };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = scale((16 + ((r * 3 + c * 5) % 200)) as u32);
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
    };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = scale((100 + ((r * 2 + c) % 50)) as u32);
            v[r * cw + c] = scale((130 + ((r + c * 3) % 50)) as u32);
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x,
        ss_y,
        usage: 0, // GOOD_QUALITY — the inter context
        cq_level: cq,
        speed: 0,
        bd,
        y,
        u,
        v,
    }
}

/// The quality axis of the sweep. cq20 keeps most residual (deep coefficient
/// coding at 12 bits), cq60 is the near-skip end.
const CQ_LEVELS: &[i32] = &[20, 60];

/// `(label, ss_x, ss_y, mono)`.
const CHROMA_SHAPES: &[(&str, usize, usize, bool)] = &[
    ("420", 1, 1, false),
    ("422", 1, 0, false),
    ("444", 0, 0, false),
    ("mono", 1, 1, true),
];

/// Per-frame verdict of a port-vs-C decode.
fn frame_verdict(
    pf: &[FrameDecode],
    cf: &[aom_sys_ref::RefDecodedFrame],
    i: usize,
) -> Option<Divergence> {
    let pv = [FrameView::of_decode(&pf[i])];
    let cv = [FrameView::of_ref_decoded(&cf[i])];
    first_frameset_divergence(&pv, &cv, SB64_PX)
}

#[test]
fn highbd_key_p_envelope_vs_c() {
    aom_sys_ref::ref_init();
    println!(
        "\n=== bd x subsampling x cq envelope: aomenc [KEY, P] (64x64, cpu0, zero-MV P, cdef/lr off) ===\n\
         port decode_frames vs aom_codec_av1_dx, per shown frame"
    );
    println!("bd | chroma | cq | frame0 KEY | frame1 P");
    let mut key_failures: Vec<String> = Vec::new();
    let mut p_results: Vec<(u8, &str, i32, String)> = Vec::new();
    for bd in [8u8, 10, 12] {
        for &(cname, ss_x, ss_y, mono) in CHROMA_SHAPES {
            for cq in CQ_LEVELS {
                let label = format!("hbd{bd}_{cname}_cq{cq}");
                let base = hbd_base(&label, 64, 64, bd, ss_x, ss_y, mono, *cq);
                // dx = dy = 0: the degenerate zero-MV P — inside the port's
                // documented bd>8 inter envelope (integer-pel convolve-copy).
                let cell = MultiFrameEncodeCell::translational(&base, 0, 0);
                let stream = cell.c_encode_inter(/*cdef=*/ false, /*lr=*/ false);
                let cf = c_frames(&stream, cell.w, cell.h);
                assert_eq!(cf.len(), 2, "{label}: C decoded {} frames", cf.len());
                assert_eq!(
                    cf[0].info[0], bd as i32,
                    "{label}: C decoded bit depth {} (wanted {bd})",
                    cf[0].info[0]
                );
                let (k, p) = match port_frames_result(&stream) {
                    Err(e) => (format!("PORT ERROR: {e}"), format!("PORT ERROR: {e}")),
                    Ok(pf) if pf.len() != 2 => {
                        let m = format!("port decoded {} frames, C decoded 2", pf.len());
                        (m.clone(), m)
                    }
                    Ok(pf) => {
                        let k = match frame_verdict(&pf, &cf, 0) {
                            None => "byte-exact".to_string(),
                            Some(d) => d.to_string(),
                        };
                        let p = match frame_verdict(&pf, &cf, 1) {
                            None => "byte-exact".to_string(),
                            Some(d) => d.to_string(),
                        };
                        (k, p)
                    }
                };
                println!("{bd:>2} | {cname:<6} | {cq:>2} | {k} | {p}");
                if k != "byte-exact" {
                    key_failures.push(format!("bd{bd} {cname} cq{cq} KEY: {k}"));
                }
                p_results.push((bd, cname, *cq, p));
            }
        }
    }

    // The KEY (intra) half must be byte-exact at EVERY bit depth and chroma
    // shape — including 12-bit, which the intra conformance corpus does not
    // cover (README: zero 12-bit, 4:2:2 or 4:4:4 vectors in that corpus).
    assert!(
        key_failures.is_empty(),
        "intra (KEY) frames diverged from C: {key_failures:#?}"
    );

    // The P half: every cell must be either byte-exact or an HONEST refusal
    // (a decode error naming the unsupported feature). A cell that decodes
    // WITHOUT error and diverges is the wrong-pixel class and fails here.
    let mut silent_wrong: Vec<String> = Vec::new();
    for (bd, cname, cq, p) in &p_results {
        let honest = p == "byte-exact" || p.starts_with("PORT ERROR:");
        if !honest {
            silent_wrong.push(format!("bd{bd} {cname} cq{cq} P: {p}"));
        }
    }
    assert!(
        silent_wrong.is_empty(),
        "inter (P) frames decoded without error but diverged from C — wrong pixels: {silent_wrong:#?}"
    );

    // Ratchet: EVERY cell in the grid is byte-exact today, so the ratchet is
    // the whole grid. A cell turning into an error (or a divergence) is a
    // regression and fails here.
    let not_exact: Vec<String> = p_results
        .iter()
        .filter(|(_, _, _, p)| p != "byte-exact")
        .map(|(bd, cname, cq, p)| format!("bd{bd} {cname} cq{cq} P: {p}"))
        .collect();
    assert!(
        not_exact.is_empty(),
        "ratcheted P cells regressed (the whole grid is byte-exact today): {not_exact:#?}"
    );
    println!(
        "FINDING: {} of {} [KEY, P] cells byte-exact vs C on BOTH frames, across bd 8/10/12 x\n\
         {{4:2:0, 4:2:2, 4:4:4, mono}} x cq {CQ_LEVELS:?}. 12-bit intra and 12-bit zero-MV inter are\n\
         both inside the byte-exact envelope.",
        p_results.len(),
        p_results.len()
    );
}

// ---------------------------------------------------------------------------
// 3. The honest boundary: highbd nonzero-MV refuses instead of guessing
// ---------------------------------------------------------------------------

#[test]
fn highbd_nonzero_mv_fails_loud_not_wrong() {
    aom_sys_ref::ref_init();
    println!("\n=== highbd nonzero-MV P: refusal vs wrong pixels ===");
    println!("bd | chroma | port verdict on the P frame");
    for bd in [8u8, 10, 12] {
        for &(cname, ss_x, ss_y, mono) in CHROMA_SHAPES {
            let label = format!("mv3_{bd}_{cname}");
            let base = hbd_base(&label, 64, 64, bd, ss_x, ss_y, mono, 60);
            // dx = 3: a real translational MV, so aomenc codes nonzero MVs.
            let cell = MultiFrameEncodeCell::translational(&base, 3, 0);
            let stream = cell.c_encode_inter(false, false);
            let cf = c_frames(&stream, cell.w, cell.h);
            assert_eq!(cf.len(), 2, "{label}: C decoded {} frames", cf.len());
            let verdict = match port_frames_result(&stream) {
                Err(e) => format!("refused: {e}"),
                Ok(pf) if pf.len() != 2 => format!("port decoded {} frames", pf.len()),
                Ok(pf) => match frame_verdict(&pf, &cf, 1) {
                    None => "byte-exact".to_string(),
                    Some(d) => format!("WRONG PIXELS: {d}"),
                },
            };
            println!("{bd:>2} | {cname:<6} | {verdict}");
            assert!(
                !verdict.starts_with("WRONG PIXELS"),
                "{label}: the port decoded a nonzero-MV highbd P without error and got different \
                 pixels than C — {verdict}"
            );
        }
    }
    println!(
        "FINDING: outside the ported envelope the decoder REFUSES (mark_corrupt) rather than \n\
         reconstructing wrong samples — every cell above is byte-exact or an explicit error."
    );
}
