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

/// The contract every envelope-boundary refusal keeps, pinned verbatim:
/// `UnsupportedFeature` naming the tool family — not `Malformed` (the stream
/// is conformant), not a panic, not silent output.
#[track_caller]
fn assert_refused_by_name(stream: &[u8], family: &str) {
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
    assert_refused_by_name(COMPOUND_STREAM, "compound");
}

#[test]
fn scaled_reference_frame_refused_by_name() {
    // OFF: named refusal at the frame_size_override gate. ON: identical until
    // step 3 routes scaled references.
    assert_refused_by_name(SCALED_STREAM, "frame_size_override");
}

#[test]
fn highbd_nonzero_mv_refused_by_name() {
    // OFF: named refusal. ON: identical until step 2 routes highbd sub-pel MC
    // — that landing replaces this arm with decode-success (byte-exactness is
    // pinned by aom-bench's highbd_inter_decode_envelope against the oracle).
    assert_refused_by_name(HIGHBD_MV_STREAM, "above bd8");
}
