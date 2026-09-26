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
//! - `scaled-ref-down.obu` / `scaled-ref-up.obu` — aomenc `--resize-mode=1`
//!   streams (respectively `--resize-denominator=16 --resize-kf-denominator=8`
//!   and the inverse, `--enable-order-hint=0 --lag-in-frames=0
//!   --auto-alt-ref=0 --enable-fwd-kf=0` to keep refs single): the first inter
//!   frame references a 2x-larger / 2x-smaller ref — the scaled-MC path.
//! - `highbd-nonzero-mv.obu` — 3-frame 64x64 bd10 stream,
//!   `aomenc --ivf --obu --codec=av1 --profile=0 --bit-depth=10
//!   --input-bit-depth=8 --end-usage=q --cq-level=30 --cpu-used=4
//!   --lag-in-frames=0`, translated content so the inter frames carry
//!   nonzero MVs.

use aom_decode::DecodeError;
use aom_decode::frame::decode_frames;

const COMPOUND_STREAM: &[u8] = include_bytes!("../data/inter/compound-refs.obu");
const MASKED_COMPOUND_STREAM: &[u8] = include_bytes!("../data/inter/masked-compound.obu");
const SCALED_STREAM: &[u8] = include_bytes!("../data/inter/frame-size-override.obu");
const SCALED_DOWN_STREAM: &[u8] = include_bytes!("../data/inter/scaled-ref-down.obu");
const SCALED_UP_STREAM: &[u8] = include_bytes!("../data/inter/scaled-ref-up.obu");
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
    if aom_decode::EXPERIMENTAL_VIDEO {
        // Step 4a routed group-0 compound (average / dist-weighted): the
        // fixture now decodes end-to-end. Masked compound (wedge/diffwtd,
        // `comp_group_idx = 1`) stays refused by name until step 4b — the
        // byte-exactness vs the C oracle is pinned by aom-bench's
        // compound_decode_envelope.
        let frames = decode_frames(COMPOUND_STREAM)
            .expect("experimental-video on: a group-0 compound stream must decode");
        assert_eq!(frames.len(), 4, "expected 4 shown frames");
    } else {
        assert_refused_by_name(COMPOUND_STREAM, "compound", &[(64, 64); 4]);
    }
}

#[test]
fn masked_compound_blocks_refused_by_name() {
    // `masked-compound.obu` — 4-frame 64x64 stream whose inter frames code
    // `comp_group_idx = 1` (wedge / diff-weighted masked compound) blocks:
    // aomenc `--end-usage=q --cq-level=20 --cpu-used=1 --lag-in-frames=4
    // --enable-global-motion=0 --enable-dist-wtd-comp=0 --enable-masked-comp=1
    // --enable-interinter-wedge=1 --enable-diff-wtd-comp=1 --limit=4` over a
    // vertical-boundary clip whose halves translate opposite directions — the
    // shape where a spatial split (wedge) between two refs wins RD.
    //
    // Feature OFF the whole compound gate refuses by name (`compound`);
    // feature ON step 4b routed the masked (wedge/diffwtd) path — the fixture
    // decodes end-to-end, byte-pinned vs the C oracle by aom-bench's
    // masked_compound_decode_envelope.
    if aom_decode::EXPERIMENTAL_VIDEO {
        let frames = decode_frames(MASKED_COMPOUND_STREAM)
            .expect("experimental-video on: a masked-compound stream must decode");
        assert_eq!(frames.len(), 4, "expected 4 shown frames");
    } else {
        assert_refused_by_name(MASKED_COMPOUND_STREAM, "compound", &[(64, 64); 4]);
    }
}

#[test]
fn scaled_reference_frame_refused_by_name() {
    if aom_decode::EXPERIMENTAL_VIDEO {
        // Step 3 routed scaled references: all three fixtures now decode
        // end-to-end. `SCALED_STREAM` is a resized KEY frame (KEY + the
        // frame_size_override mechanism alone); the up/down streams each code
        // an inter frame referencing a 2x-larger / 2x-smaller ref —
        // byte-exactness vs the C oracle is pinned by aom-bench's
        // scaled_ref_decode_envelope.
        for (name, stream, n) in [
            ("frame-size-override", SCALED_STREAM, 1usize),
            ("scaled-ref-down", SCALED_DOWN_STREAM, 6),
            ("scaled-ref-up", SCALED_UP_STREAM, 5),
        ] {
            let frames = decode_frames(stream)
                .unwrap_or_else(|e| panic!("experimental-video on: {name} must decode, got {e:?}"));
            assert_eq!(frames.len(), n, "{name}: shown-frame count");
        }
    } else {
        for (stream, dims) in [
            (SCALED_STREAM, &[(51, 51)][..]),
            (
                SCALED_DOWN_STREAM,
                &[(128, 128), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64)][..],
            ),
            (
                SCALED_UP_STREAM,
                &[(64, 64), (128, 128), (128, 128), (128, 128), (128, 128)][..],
            ),
        ] {
            assert_refused_by_name(stream, "frame_size_override", dims);
        }
    }
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
