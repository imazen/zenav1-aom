#![no_main]
//! Fuzz the multi-frame OBU decode entry point.
//!
//! `decode_frames` parses a stream of raw AV1 OBU temporal units (a KEY frame
//! optionally followed by inter frames — the exact bytes an AVIF `mdat` /
//! animated-AVIF track carries, and what zenavif hands the decoder). On ANY
//! malformed input it must return `Err`, never panic (unwrap / expect /
//! out-of-bounds slice index / `assert!` / arithmetic overflow) and never
//! allocate without bound. This target holds that contract.
//!
//! Per `CLAUDE.md` §5 the target drives `decode_frames_with` under a **low
//! `max_pixels`** limit — see the `decode_obus` target for the rationale (an
//! in-bounds but very large declared frame would otherwise report a spurious
//! OOM against libFuzzer's 2 GiB malloc limit).
use aom_decode::{DecodeConfig, DecodeLimits};
use libfuzzer_sys::fuzz_target;

/// 4 Mpx (2048×2048) — a realistic web-image ceiling that bounds the peak
/// per-frame allocation to a few tens of MiB, far under libFuzzer's OOM limit.
const FUZZ_MAX_PIXELS: u64 = 1 << 22;

fuzz_target!(|data: &[u8]| {
    let mut limits = DecodeLimits::default();
    limits.max_pixels = Some(FUZZ_MAX_PIXELS);
    let config = DecodeConfig::default().with_limits(limits);
    let _ = aom_decode::frame::decode_frames_with(data, &config);
});
