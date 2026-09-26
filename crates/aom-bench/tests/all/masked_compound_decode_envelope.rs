//! MASKED-COMPOUND DECODE ENVELOPE — `experimental-video` step 4b
//! (docs/HANDOFF-EXPERIMENTAL-VIDEO.md).
//!
//! Group-1 compound (`comp_group_idx = 1`) predicts from TWO bound references
//! and blends them with a **spatial mask** rather than a uniform/distance
//! weight: `COMPOUND_WEDGE` fetches a codebook mask (`wedge_index`/`wedge_sign`),
//! `COMPOUND_DIFFWTD` builds `seg_mask` from the two luma convolve intermediates
//! (`mask_type` picks the sign). Each reference convolves into its OWN `d16`
//! buffer (C's `tmp_conv_dst`/`tmp_buf16`), then `aom_*_blend_a64_d16_mask`
//! combines them through the luma-resolution mask — chroma subsamples it via
//! `d16_mask_at`.
//!
//! This file gates the END-TO-END decoder path against the REAL C decoder on
//! the same bytes: the masked-compound syntax (comp_group_idx, compound_type,
//! wedge_index/sign or mask_type), the per-ref `convolve_one_compound_ref`
//! gathers into the two d16 buffers, the wedge/diffwtd mask selection, the luma
//! diffwtd `seg_mask` build + chroma reuse, and the d16 masked blend at both
//! lowbd and the chroma subsampling.
//!
//! Fixture (`crates/aom-decode/tests/data/inter/`), verified conformant (the
//! pinned `aomdec` decodes it):
//!
//! - `masked-compound.obu` — 4-frame 64x64 aomenc stream whose blend-content
//!   frames code masked-compound blocks:
//!   `aomenc --ivf --obu --codec=av1 --end-usage=q --cq-level=24
//!   --cpu-used=1 --lag-in-frames=4` over photo-crop y4m frames with an
//!   occlusion/disocclusion boundary — the shape where a wedge or diffwtd mask
//!   wins RD over uniform averaging.
//!
//! With `experimental-video` ON every shown frame must be byte-identical to
//! `aom_codec_av1_dx`; with it OFF the stream must be refused by name
//! (`compound` — the compound-reference gate fires before the masked read).
//! The per-frame dims are PINNED — the C shim panics when the decoded frame's
//! dims differ from the expectation, so a port geometry error cannot hide
//! behind the masked path.

use aom_bench::inter_localize::{FrameView, SB64_PX, first_frameset_divergence};
use aom_decode::DecodeError;

const INTER_FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../aom-decode/tests/data/inter"
);

/// `(fixture stem, pinned per-frame (w, h) in shown order)`. Dims are pinned
/// rather than read back, so a port geometry error cannot hide.
const MASKED_STREAMS: &[(&str, &[(usize, usize)])] = &[("masked-compound", &[(64, 64); 4])];

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
fn masked_compound_streams_match_c_decoder() {
    aom_sys_ref::ref_init();
    println!("\n=== masked-compound streams: port decode_frames vs aom_codec_av1_dx ===");
    println!("stream | frames | dims | verdict");

    if !aom_decode::EXPERIMENTAL_VIDEO {
        // Feature off: the conformant stream must refuse BY NAME at the
        // compound-reference gate (compound is refused before the masked read).
        for &(name, _) in MASKED_STREAMS {
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
    for &(name, dims) in MASKED_STREAMS {
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
        "masked-compound decode diverged from the C oracle: {failures:#?}"
    );
    println!(
        "FINDING (experimental-video step 4b): the masked-compound stream\n\
         decodes BYTE-IDENTICALLY to the real libaom C decoder — wedge and\n\
         diff-weighted masked compound route end-to-end through the two-buffer\n\
         d16 mask blend."
    );
}
