//! The `whereat` feature's located-error entry points (`decode_frame_obus_at`,
//! `decode_frames_at`).
//!
//! **Why this file exists.** Both entries are `#[cfg(feature = "whereat")]` and
//! the feature is default-off, so before this test nothing in the workspace —
//! no test, no CI job, no bench — ever *compiled* them, let alone ran them. A
//! feature that is offered in `Cargo.toml` and never built is a feature that
//! can be broken for months in silence. The whole file is gated the same way,
//! so the skip decision belongs to the caller (`--features whereat`, wired as
//! `just test-whereat` and a CI step), not to a runtime `if` inside a test.
//!
//! **What it pins.** Each `_at` entry is a one-line `map_err(at!)` over a
//! specific inner function. That shape has exactly one interesting failure
//! mode, and it is a silent one: wire the wrapper to the WRONG inner call and
//! it still compiles, still returns `Ok`, still returns plausible pixels — a
//! `decode_frames_at` delegating to `decode_frame_obus_with` just quietly
//! yields the first frame of an animation. So the success tests compare each
//! `_at` entry against its named non-`_at` sibling on a real multi-frame
//! stream, and the error tests require the `At` payload to carry the sibling's
//! exact error plus a non-empty trace.

#![cfg(feature = "whereat")]

use aom_decode::DecodeConfig;
use aom_decode::frame::{
    FrameDecode, decode_frame_obus_at, decode_frame_obus_with, decode_frames_at, decode_frames_with,
};

/// A committed multi-frame AV1 temporal-unit stream (5 shown frames): the same
/// fixture `animated_avif.rs` gates byte-exactness on. Multi-frame is the
/// point — a single-frame input could not tell `decode_frames_at` apart from
/// `decode_frame_obus_at`.
const STREAM: &[u8] = include_bytes!("data/animated/colors-animated-8bpc.color.obu");

/// Byte length of [`STREAM`]'s FIRST temporal unit — up to the second
/// `OBU_TEMPORAL_DELIMITER` (`0x12 0x00`: obu_type 2, has_size_field, size 0).
/// `decode_frame_obus*` accepts exactly one temporal unit and rejects the rest
/// of the animation, so the single-frame entries need this prefix. The constant
/// is self-checking: the tests below require the prefix to decode AND to equal
/// frame 0 of the full multi-frame decode, so a wrong boundary fails loudly.
const FIRST_TEMPORAL_UNIT: usize = 39;

fn same_frame(a: &FrameDecode, b: &FrameDecode, what: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}: dims");
    assert_eq!(a.bit_depth, b.bit_depth, "{what}: bit depth");
    assert_eq!(a.y, b.y, "{what}: luma");
    assert_eq!(a.u, b.u, "{what}: cb");
    assert_eq!(a.v, b.v, "{what}: cr");
}

/// `decode_frames_at` must be `decode_frames_with` plus a location — same frame
/// COUNT and same pixels. The count is what catches a wrapper delegating to the
/// single-frame entry.
#[test]
fn decode_frames_at_matches_decode_frames_with() {
    let cfg = DecodeConfig::default();
    let plain = decode_frames_with(STREAM, &cfg).expect("plain multi-frame decode");
    let located = decode_frames_at(STREAM, &cfg).expect("located multi-frame decode");
    assert!(
        plain.len() > 1,
        "fixture must be multi-frame for this test to discriminate (got {})",
        plain.len()
    );
    assert_eq!(
        located.len(),
        plain.len(),
        "decode_frames_at returned a different frame count than decode_frames_with"
    );
    for (i, (a, b)) in located.iter().zip(plain.iter()).enumerate() {
        same_frame(a, b, &format!("frame {i}"));
    }
}

/// `decode_frame_obus_at` must be `decode_frame_obus_with` plus a location, on
/// the single temporal unit that entry accepts.
#[test]
fn decode_frame_obus_at_matches_decode_frame_obus_with() {
    let cfg = DecodeConfig::default();
    let one_tu = &STREAM[..FIRST_TEMPORAL_UNIT];
    let plain = decode_frame_obus_with(one_tu, &cfg).expect("plain single-frame decode");
    let located = decode_frame_obus_at(one_tu, &cfg).expect("located single-frame decode");
    same_frame(&located, &plain, "first frame");
    // Independently validates FIRST_TEMPORAL_UNIT: the prefix must reconstruct
    // the same pixels the multi-frame walk produces for frame 0.
    let all = decode_frames_with(STREAM, &cfg).expect("plain multi-frame decode");
    same_frame(&located, &all[0], "first temporal unit vs frame 0");
}

/// The two entries have genuinely DIFFERENT envelopes — `decode_frame_obus*`
/// accepts one temporal unit and rejects an animation, `decode_frames*` walks
/// it. That difference is the sharpest available cross-wiring probe: feed both
/// located entries the full animation and require the single-frame one to
/// reject with its sibling's exact error while the multi-frame one succeeds. A
/// `decode_frame_obus_at` accidentally delegating to `decode_frames_with` would
/// return `Ok` here.
#[test]
fn located_entries_keep_their_own_envelopes() {
    let cfg = DecodeConfig::default();
    let plain_err = decode_frame_obus_with(STREAM, &cfg)
        .err()
        .expect("the single-frame entry must reject a multi-frame stream");
    let located_err = decode_frame_obus_at(STREAM, &cfg)
        .err()
        .expect("decode_frame_obus_at must reject a multi-frame stream too");
    assert_eq!(
        located_err.error(),
        &plain_err,
        "decode_frame_obus_at must reject with its sibling's error"
    );
    assert!(
        decode_frames_at(STREAM, &cfg).is_ok(),
        "decode_frames_at must accept the same stream"
    );
}

/// On failure the located entries must preserve the sibling's exact
/// `DecodeError` (no category laundering) AND actually attach a trace — an
/// `At` with an empty trace and no crate info is the silent regression here,
/// because it still type-checks and still carries the error.
#[test]
fn located_errors_keep_the_plain_error_and_carry_a_trace() {
    let cfg = DecodeConfig::default();
    // Truncated mid-stream: enough bytes to start parsing, not enough to finish.
    let truncated = &STREAM[..STREAM.len() / 3];

    let plain_one = decode_frame_obus_with(truncated, &cfg)
        .err()
        .expect("truncated input must fail the single-frame entry");
    let located_one = decode_frame_obus_at(truncated, &cfg)
        .err()
        .expect("truncated input must fail the located single-frame entry");
    assert_eq!(
        located_one.error(),
        &plain_one,
        "decode_frame_obus_at changed the error"
    );
    assert!(
        located_one.frame_count() >= 1,
        "at! recorded no location frame"
    );
    assert_eq!(
        located_one.crate_info().map(|i| i.name()),
        Some("zenav1-aom-decode"),
        "located error must name this crate"
    );

    let plain_many = decode_frames_with(truncated, &cfg)
        .err()
        .expect("truncated input must fail the multi-frame entry");
    let located_many = decode_frames_at(truncated, &cfg)
        .err()
        .expect("truncated input must fail the located multi-frame entry");
    assert_eq!(
        located_many.error(),
        &plain_many,
        "decode_frames_at changed the error"
    );
    assert!(
        located_many.frame_count() >= 1,
        "at! recorded no location frame"
    );
    assert_eq!(
        located_many.crate_info().map(|i| i.name()),
        Some("zenav1-aom-decode"),
        "located error must name this crate"
    );
}
