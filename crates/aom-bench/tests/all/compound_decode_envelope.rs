//! COMPOUND-REFERENCE DECODE ENVELOPE — `experimental-video` step 4a
//! (docs/HANDOFF-EXPERIMENTAL-VIDEO.md).
//!
//! A compound inter block predicts from TWO bound references and blends them —
//! group 0 (`comp_group_idx = 0`) is plain averaging or distance-weighted
//! averaging (`compound_idx`), which this envelope routes end-to-end; group 1
//! (`comp_group_idx = 1`, wedge / diff-weighted masked compound) stays refused
//! BY NAME until step 4b lands the mask builders.
//!
//! This file gates the END-TO-END decoder path against the REAL C decoder on
//! the same bytes: the compound reference-pair syntax (slots 1..9 of
//! `ref_frame_cdfs`), the compound MV-ref stack (`find_inter_mv_refs` rf-pair),
//! `read_inter_compound_mode`, the per-submode `assign_mv` resolution, the
//! group-0 compound-type read (`compound_idx` + dist-wtd weights from the
//! bound refs' order hints), both refs' `build_compound_inter_predictor`, and
//! the neighbour/frame-MV stamps that let the NEXT block see this block as a
//! compound neighbour.
//!
//! Fixture (`crates/aom-decode/tests/data/inter/`), verified conformant (the
//! pinned `aomdec` decodes it):
//!
//! - `compound-refs.obu` — 4-frame 64x64 aomenc stream whose last (blend
//!   content) frame codes compound-reference blocks:
//!   `aomenc --ivf --obu --codec=av1 --end-usage=q --cq-level=24
//!   --cpu-used=1 --lag-in-frames=4 --enable-global-motion=0` over photo-crop
//!   y4m frames where frame 3 is the per-pixel average of two translations —
//!   the shape where compound averaging wins RD.
//!
//! With `experimental-video` ON every shown frame must be byte-identical to
//! `aom_codec_av1_dx`; with it OFF the stream must be refused by name
//! (`compound`). The per-frame dims are PINNED — the C shim panics when the
//! decoded frame's dims differ from the expectation, so a port geometry error
//! cannot hide behind the compound path.

use aom_bench::inter_localize::{FrameView, SB64_PX, first_frameset_divergence};
use aom_decode::DecodeError;

const INTER_FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../aom-decode/tests/data/inter"
);

/// `(fixture stem, pinned per-frame (w, h) in shown order)`. Dims are pinned
/// rather than read back, so a port geometry error cannot hide.
const COMPOUND_STREAMS: &[(&str, &[(usize, usize)])] = &[("compound-refs", &[(64, 64); 4])];

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
fn compound_reference_streams_match_c_decoder() {
    aom_sys_ref::ref_init();
    println!("\n=== compound-reference streams: port decode_frames vs aom_codec_av1_dx ===");
    println!("stream | frames | dims | verdict");

    if !aom_decode::EXPERIMENTAL_VIDEO {
        // Feature off: the conformant stream must refuse BY NAME at the
        // compound-reference gate.
        for &(name, _) in COMPOUND_STREAMS {
            let stream = std::fs::read(format!("{INTER_FIXTURES}/{name}.obu"))
                .unwrap_or_else(|e| panic!("fixture {name}.obu missing: {e}"));
            match aom_decode::frame::decode_frames(&stream) {
                Err(DecodeError::UnsupportedFeature(n)) => {
                    assert!(
                        n.contains("compound"),
                        "{name}: refusal lost its name — {n:?}"
                    );
                    println!("{name} | - | - | refused by name ({n})");
                }
                other => panic!(
                    "{name}: experimental-video OFF must refuse compound by name, got {other:?}"
                ),
            }
        }
        return;
    }

    let mut failures: Vec<String> = Vec::new();
    for &(name, dims) in COMPOUND_STREAMS {
        let stream = std::fs::read(format!("{INTER_FIXTURES}/{name}.obu"))
            .unwrap_or_else(|e| panic!("fixture {name}.obu missing: {e}"));
        let pf = match aom_bench::inter_localize::try_decode_frames(&stream) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{name}: port decode failed: {e}"));
                println!("{name} | - | - | PORT DECODE ERROR: {e}");
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
        let cf = c_frames_pinned(&stream, dims, name);
        let pv: Vec<FrameView> = pf.iter().map(FrameView::of_decode).collect();
        let cv: Vec<FrameView> = cf.iter().map(FrameView::of_ref_decoded).collect();
        let div = first_frameset_divergence(&pv, &cv, SB64_PX);
        let verdict = match &div {
            None => format!("byte-exact ({} shown frames)", pf.len()),
            Some(d) => d.to_string(),
        };
        println!(
            "{name} | {} | {:?} | {verdict}",
            pf.len(),
            pf.iter().map(|f| (f.width, f.height)).collect::<Vec<_>>()
        );
        if let Some(d) = div {
            failures.push(format!("{name}: {d}"));
        }
    }
    assert!(
        failures.is_empty(),
        "compound-reference decode diverged from the C oracle: {failures:#?}"
    );
    println!(
        "FINDING (experimental-video step 4a): the compound-reference stream\n\
         decodes BYTE-IDENTICALLY to the real libaom C decoder — group-0\n\
         (average / dist-weighted) compound blocks route end-to-end."
    );
}
