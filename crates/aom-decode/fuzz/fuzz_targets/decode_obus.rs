#![no_main]
//! Fuzz the single-KEY-frame OBU decode entry point.
//!
//! `decode_frame_obus` decodes one AV1 temporal unit (temporal delimiter +
//! sequence header + frame) to cropped planes — the exact function the Gate-1
//! conformance harness drives and the natural entry for a still AVIF image.
//! On ANY malformed input it must return `Err`, never panic and never allocate
//! without bound.
//!
//! Per `CLAUDE.md` §5 the target drives `decode_frame_obus_with` under a **low
//! `max_pixels`** limit: a crafted header may legitimately declare a frame up
//! to the `1<<28` default ceiling (≈268 Mpx), whose in-bounds recon/mi
//! allocation exceeds libFuzzer's 2 GiB malloc limit and reports a (spurious,
//! for fuzzing) OOM. The low cap makes the decoder reject an oversized header
//! with `LimitExceeded` — exercising that reject path while keeping every
//! accepted decode's allocation bounded well under the OOM limit.
use aom_decode::{DecodeConfig, DecodeLimits};
use libfuzzer_sys::fuzz_target;

/// 4 Mpx (2048×2048) — a realistic web-image ceiling that bounds the peak
/// per-frame allocation to a few tens of MiB, far under libFuzzer's OOM limit.
const FUZZ_MAX_PIXELS: u64 = 1 << 22;

fuzz_target!(|data: &[u8]| {
    let mut limits = DecodeLimits::default();
    limits.max_pixels = Some(FUZZ_MAX_PIXELS);
    let config = DecodeConfig::default().with_limits(limits);
    let _ = aom_decode::frame::decode_frame_obus_with(data, &config);
});
