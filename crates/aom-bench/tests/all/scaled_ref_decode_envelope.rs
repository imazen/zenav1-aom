//! SCALED-REFERENCE DECODE ENVELOPE — `experimental-video` step 3
//! (docs/HANDOFF-EXPERIMENTAL-VIDEO.md).
//!
//! A scaled reference is an inter frame whose bound refs have different LUMA
//! crop dims than the frame being coded — C routes their MC through
//! `av1_convolve_2d_scale` (the per-output-sample stepped convolve in
//! `aom-dsp/src/convolve/scaled.rs`, itself differentially locked at kernel
//! level by `convolve_scale_diff.rs`). This file gates the END-TO-END decoder
//! path: `frame_size_override` acceptance, `setup_frame_size_with_refs`
//! (`found_ref`), per-ref `ScaleFactors` (`av1_setup_scale_factors_for_frame`),
//! the scaled `dec_calc_subpel_params` branch, border extension, and the
//! WARPED_CAUSAL motion-mode disable — against the REAL C decoder on the same
//! bytes.
//!
//! Fixtures (`crates/aom-decode/tests/data/inter/`), each verified conformant
//! (the pinned `aomdec` decodes them):
//!
//! - `scaled-ref-down.obu` — 128x128 KEY + five 64x64 inter frames referencing
//!   the 2x-LARGER ref (`av1_is_scaled`, x/y step 2048 q10):
//!   `aomenc --obu --limit=6 --cpu-used=6 --end-usage=q --cq-level=30
//!   --resize-mode=1 --resize-denominator=16 --resize-kf-denominator=8
//!   --enable-order-hint=0 --lag-in-frames=0 --auto-alt-ref=0
//!   --enable-fwd-kf=0 --fps=30/1` over perturbed-photo y4m.
//! - `scaled-ref-up.obu` — 64x64 KEY + four 128x128 inter frames referencing the
//!   2x-SMALLER ref (the reciprocal direction, step 512 q10): same recipe with
//!   denominators swapped and `--cq-level=50 --limit=5`.
//! - `frame-size-override.obu` — a resized KEY frame (51x51 < seq max): the
//!   `frame_size_override` mechanism on an intra frame, no inter MC.
//!
//! With `experimental-video` ON every shown frame must be byte-identical to
//! `aom_codec_av1_dx`; with it OFF each stream must be refused by name
//! (`frame_size_override`). The per-frame dims are PINNED in the table below —
//! the C shim panics when the decoded frame's dims differ from the expectation,
//! so a port geometry error cannot hide behind `is_scaled`.

use aom_bench::inter_localize::{FrameView, SB64_PX, first_frameset_divergence};
use aom_decode::DecodeError;

const INTER_FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../aom-decode/tests/data/inter"
);

/// `(fixture stem, pinned per-frame (w, h) in shown order)`. Dims are pinned
/// rather than read back, so a port geometry error cannot hide.
const SCALED_STREAMS: &[(&str, &[(usize, usize)])] = &[
    (
        "scaled-ref-down",
        &[(128, 128), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64)],
    ),
    (
        "scaled-ref-up",
        &[(64, 64), (128, 128), (128, 128), (128, 128), (128, 128)],
    ),
    ("frame-size-override", &[(51, 51)]),
];

/// C-decode the pinned `dims.len()` shown frames of `stream`, asserting each
/// against its pinned `(w, h)` — the shim panics when a decoded frame's dims
/// differ from the expectation, and `None` mid-list (C produced fewer shown
/// frames than pinned) fails loudly rather than silently truncating the
/// comparison.
fn c_frames_pinned(
    stream: &[u8],
    dims: &[(usize, usize)],
    label: &str,
) -> Vec<aom_sys_ref::RefDecodedFrame> {
    let mut out = Vec::with_capacity(dims.len());
    for (i, &(w, h)) in dims.iter().enumerate() {
        match aom_sys_ref::ref_decode_av1_stream_frame_opt(stream, i, w, h) {
            Some(f) => out.push(f),
            None => panic!("{label}: C produced fewer than {} shown frames", dims.len()),
        }
    }
    out
}

#[test]
fn scaled_reference_streams_match_c_decoder() {
    aom_sys_ref::ref_init();
    println!("\n=== scaled-reference streams: port decode_frames vs aom_codec_av1_dx ===");
    println!("stream | frames | dims | scaled ref | verdict");

    if !aom_decode::EXPERIMENTAL_VIDEO {
        // Feature off: each conformant stream must refuse BY NAME at the
        // frame_size_override gate — the scaled-reference family's front door.
        for &(name, _) in SCALED_STREAMS {
            let stream = std::fs::read(format!("{INTER_FIXTURES}/{name}.obu"))
                .unwrap_or_else(|e| panic!("fixture {name}.obu missing: {e}"));
            match aom_decode::frame::decode_frames(&stream) {
                Err(DecodeError::UnsupportedFeature(n)) => {
                    assert!(
                        n.contains("frame_size_override"),
                        "{name}: refusal lost its name — {n:?}"
                    );
                    println!("{name} | - | - | - | refused by name ({n})");
                }
                other => panic!(
                    "{name}: experimental-video OFF must refuse frame_size_override by name, got {other:?}"
                ),
            }
        }
        return;
    }

    let mut failures: Vec<String> = Vec::new();
    for &(name, dims) in SCALED_STREAMS {
        let stream = std::fs::read(format!("{INTER_FIXTURES}/{name}.obu"))
            .unwrap_or_else(|e| panic!("fixture {name}.obu missing: {e}"));
        let pf = match aom_bench::inter_localize::try_decode_frames(&stream) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{name}: port decode failed: {e}"));
                println!("{name} | - | - | - | PORT DECODE ERROR: {e}");
                continue;
            }
        };
        assert_eq!(
            pf.len(),
            dims.len(),
            "{name}: port shown-frame count {} != pinned {}",
            pf.len(),
            dims.len()
        );
        for (i, (f, &(w, h))) in pf.iter().zip(dims.iter()).enumerate() {
            assert_eq!(
                (f.width, f.height),
                (w, h),
                "{name} frame {i}: port dims {}x{} != pinned {w}x{h}",
                f.width,
                f.height
            );
        }
        // Non-vacuity: a scaled stream must contain a frame whose dims differ
        // from its bound ref's — i.e. SOME pair of shown frames differs in size
        // (the first inter frame's only ref here is the KEY frame).
        let scaled = pf
            .iter()
            .any(|f| (f.width, f.height) != (pf[0].width, pf[0].height));
        if name != "frame-size-override" {
            assert!(scaled, "{name}: no frame dim change — nothing was scaled");
        }
        let cf = c_frames_pinned(&stream, dims, name);
        let pv: Vec<FrameView> = pf.iter().map(FrameView::of_decode).collect();
        let cv: Vec<FrameView> = cf.iter().map(FrameView::of_ref_decoded).collect();
        let div = first_frameset_divergence(&pv, &cv, SB64_PX);
        let verdict = match &div {
            None => format!("byte-exact ({} shown frames)", pf.len()),
            Some(d) => d.to_string(),
        };
        println!(
            "{name} | {} | {:?} | {scaled} | {verdict}",
            pf.len(),
            pf.iter().map(|f| (f.width, f.height)).collect::<Vec<_>>()
        );
        if let Some(d) = div {
            failures.push(format!("{name}: {d}"));
        }
    }
    assert!(
        failures.is_empty(),
        "scaled-reference decode diverged from the C oracle: {failures:#?}"
    );
    println!(
        "FINDING (experimental-video step 3): scaled-reference streams decode\n\
         BYTE-IDENTICALLY to the real libaom C decoder in both scale directions,\n\
         including the inter frame whose only ref is 2x its size."
    );
}
