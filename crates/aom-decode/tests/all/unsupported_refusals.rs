//! Refusal-by-name pins at the `experimental-video` envelope boundary
//! (docs/HANDOFF-EXPERIMENTAL-VIDEO.md).
//!
//! A conformant stream that uses an AV1 decode tool outside this build's
//! envelope is refused BY NAME — `DecodeError::UnsupportedFeature` — never
//! `Malformed` (the stream is not corrupt), never a panic, never silently
//! wrong pixels. These pins run in BOTH build states: feature OFF the
//! refusals are the whole contract; feature ON they must still fire for the
//! families not yet routed, and each routing landing flips its arm here to
//! "decodes" (the byte-exact differential lives in the aom-bench gates).
//!
//! Fixtures under `tests/data/inter/` (each verified conformant — the pinned
//! `aomdec` decodes them; the refusal is OUR envelope, not a bad stream):
//!
//! - `compound-refs.obu` — 4-frame 64x64 aomenc stream whose last (blend
//!   content) frame codes compound-reference blocks:
//!   `aomenc --ivf --obu --codec=av1 --end-usage=q --cq-level=24
//!   --cpu-used=1 --lag-in-frames=4 --enable-global-motion=0` over
//!   photo-crop y4m frames where frame 3 is the per-pixel average of two
//!   translations — the shape where compound averaging wins RD.
//! - `frame-size-override.obu` — aomenc `--resize-mode` stream: the coded
//!   frame size differs from the sequence max, tripping the
//!   `frame_size_override` gate (the scaled-reference family's front door).
//! - `highbd-nonzero-mv.obu` — 3-frame 64x64 bd10 stream,
//!   `aomenc --ivf --obu --codec=av1 --profile=0 --bit-depth=10
//!   --input-bit-depth=8 --end-usage=q --cq-level=30 --cpu-used=4
//!   --lag-in-frames=0`, translated content so the inter frames carry
//!   nonzero MVs.

use aom_decode::DecodeError;
use aom_decode::frame::decode_frames;

const COMPOUND_STREAM: &[u8] = include_bytes!("../data/inter/compound-refs.obu");
const SCALED_STREAM: &[u8] = include_bytes!("../data/inter/frame-size-override.obu");
const HIGHBD_MV_STREAM: &[u8] = include_bytes!("../data/inter/highbd-nonzero-mv.obu");

/// Evidence for the pin's premise: `stream` is CONFORMANT — the in-process
/// C oracle (`aom_codec_av1_dx`, pinned to the same upstream rev) decodes
/// every frame listed in `dims`, each asserted against its pinned `(w, h)`
/// (the shim panics on a dims mismatch and on any mid-stream decode error).
/// The named refusal that follows is therefore OUR envelope boundary, not a
/// corrupt stream.
#[track_caller]
fn assert_conformant_to_c_oracle(stream: &[u8], dims: &[(usize, usize)], label: &str) {
    aom_sys_ref::ref_init();
    assert!(
        !dims.is_empty(),
        "{label}: pin must list at least one shown frame"
    );
    for (i, &(w, h)) in dims.iter().enumerate() {
        aom_sys_ref::ref_decode_av1_stream_frame_opt(stream, i, w, h).unwrap_or_else(|| {
            panic!(
                "{label}: C oracle produced fewer than {} shown frames",
                dims.len()
            )
        });
    }
}

/// The contract every envelope-boundary refusal keeps, pinned verbatim:
/// `UnsupportedFeature` naming the tool family — not `Malformed` (the stream
/// is conformant — proven against the C oracle first), not a panic, not
/// silent output.
#[track_caller]
fn assert_refused_by_name(stream: &[u8], family: &str, dims: &[(usize, usize)]) {
    assert_conformant_to_c_oracle(stream, dims, family);
    match decode_frames(stream) {
        Err(DecodeError::UnsupportedFeature(name)) => {
            assert!(
                name.contains(family),
                "refusal for the {family:?} family lost its name — got {name:?}"
            );
        }
        Err(e) => panic!(
            "{family:?}: refused with the wrong category — {e:?} maps to {:?}; \
             the stream is conformant, so the category must be unsupported-feature",
            e.category()
        ),
        Ok(frames) => panic!(
            "{family:?}: decoded {} shown frames — the named refusal is gone \
             (if this family just landed behind experimental-video, flip this \
             pin's on-arm to decode-success)",
            frames.len()
        ),
    }
}

#[test]
fn compound_reference_blocks_refused_by_name() {
    // OFF: named refusal. ON: identical until step 4 routes compound — that
    // landing replaces this arm with decode-success.
    assert_refused_by_name(COMPOUND_STREAM, "compound", &[(64, 64); 4]);
}

#[test]
fn scaled_reference_frame_refused_by_name() {
    // OFF: named refusal at the frame_size_override gate. ON: identical until
    // step 3 routes scaled references.
    assert_refused_by_name(SCALED_STREAM, "frame_size_override", &[(51, 51)]);
}

#[test]
fn highbd_nonzero_mv_refused_by_name() {
    if aom_decode::EXPERIMENTAL_VIDEO {
        // Step 2 routed highbd sub-pel MC: the fixture now decodes end-to-end.
        // Byte-exactness vs the C oracle is pinned by aom-bench's
        // highbd_inter_decode_envelope.
        let frames = decode_frames(HIGHBD_MV_STREAM)
            .expect("experimental-video on: a nonzero-MV bd10 stream must decode");
        assert_eq!(frames.len(), 2, "expected KEY + P, got {}", frames.len());
    } else {
        assert_refused_by_name(HIGHBD_MV_STREAM, "above bd8", &[(64, 64); 2]);
    }
}
