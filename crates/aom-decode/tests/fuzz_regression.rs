//! Fuzz crash regression suite (stable toolchain — no nightly / cargo-fuzz).
//!
//! Runs every file in `fuzz/regression/` through both decoder entry points the
//! cargo-fuzz targets exercise (`decode_frames` and `decode_frame_obus`). Each
//! seed is a previously-found crash on untrusted input that has since been
//! fixed to return `Err` instead of panicking; this test guards against any of
//! them re-introducing a panic (unwrap/expect/OOB index/assert/overflow).
//!
//! The decoder ships into zenavif and decodes untrusted AVIF OBU payloads, so a
//! reachable panic on a malformed bitstream is a denial-of-service. A seed may
//! decode (`Ok`) or be rejected (`Err`) — either is fine; the contract is only
//! that neither entry point panics.
//!
//! To add a seed: drop the (preferably `cargo fuzz tmin`-minimized, <8 KB,
//! target <1 KB) crash file into `fuzz/regression/`. No other change needed.

use aom_decode::frame::{decode_frame_obus_with, decode_frames_with};
use aom_decode::{DecodeConfig, DecodeLimits};
use std::fs;
use std::path::PathBuf;

fn regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

/// 4 Mpx (2048×2048) — the same low `max_pixels` the cargo-fuzz targets pin
/// (see `fuzz/fuzz_targets/decode_obus.rs`). It bounds the peak per-frame
/// allocation to a few tens of MiB so an in-bounds-but-huge declared frame is
/// rejected with `LimitExceeded` instead of driving a multi-GiB allocation.
const FUZZ_MAX_PIXELS: u64 = 1 << 22;

fn fuzz_config() -> DecodeConfig<'static> {
    let mut limits = DecodeLimits::default();
    limits.max_pixels = Some(FUZZ_MAX_PIXELS);
    DecodeConfig::default().with_limits(limits)
}

/// Feed one seed through both untrusted-input entry points. A panic here
/// unwinds with the seed name in the failure message (`#[test]` catches it).
fn run_all_entry_points(input: &[u8]) {
    let config = fuzz_config();
    // Multi-frame OBU stream (KEY + inter) — the superset entry.
    let _ = decode_frames_with(input, &config);
    // Single KEY-frame temporal unit — the still-AVIF entry.
    let _ = decode_frame_obus_with(input, &config);
}

#[test]
fn fuzz_regression_seeds_do_not_panic() {
    let dir = regression_dir();
    let entries: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .collect();

    assert!(
        !entries.is_empty(),
        "fuzz/regression/ is empty — the committed crash POCs should be present at {}",
        dir.display()
    );

    for entry in entries {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>")
            .to_owned();
        let input = fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));

        // Each entry point may return Err but MUST NOT panic. If it does, the
        // test fails with this seed identified.
        run_all_entry_points(&input);

        eprintln!("ok: {name} ({} bytes)", input.len());
    }
}

/// "Does not panic" is the floor, not the contract. Every POC is UNTRUSTED
/// input, so whatever it produces must be a caller-actionable outcome:
/// `Ok`, or an `Err` in a category that describes the INPUT. In particular it
/// must never be [`DecodeError::Internal`] — that variant means "a bug in this
/// decoder, not attacker input", so an attacker-reachable `Internal` is by
/// definition a defect (and the seam maps it onto an internal-error category
/// the consumer cannot act on).
#[test]
fn fuzz_regression_seeds_never_report_an_internal_error() {
    let config = fuzz_config();
    let mut checked = 0usize;
    for entry in fs::read_dir(regression_dir()).expect("read fuzz/regression") {
        let path = entry.expect("dir entry").path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_owned();
        let input = fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
        for (entry_name, res) in [
            ("decode_frame_obus", decode_frame_obus_with(&input, &config).map(|_| ())),
            ("decode_frames", decode_frames_with(&input, &config).map(|_| ())),
        ] {
            if let Err(e) = res {
                assert_ne!(
                    e.category(),
                    "internal",
                    "{name} via {entry_name}: untrusted input reached an INTERNAL error \
                     ({e}) — that variant is reserved for decoder bugs, so this is one"
                );
            }
        }
        checked += 1;
    }
    assert!(checked > 0, "no POCs checked — fuzz/regression/ is empty");
    eprintln!("{checked} POCs: no internal errors");
}

/// The chroma `BLOCK_INVALID` rejection is LIVE, and it is an `Err` rather than
/// a panic. `av1_ss_size_lookup` has no valid chroma size for some luma shapes
/// at 4:2:2; `decode_mbmi_block` (decodeframe.c:393-401) calls that
/// `AOM_CODEC_CORRUPT_FRAME`. This POC is the minimized crafted stream that
/// reaches it — anti-vacuous teeth for the guards in `decode_partition` /
/// `decode_block` / the chroma txb loop.
#[test]
fn invalid_422_chroma_subsize_is_a_typed_corrupt_frame_error() {
    let path = regression_dir().join("decode_obus_422_invalid_chroma_subsize.obu");
    let input = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let config = fuzz_config();
    let err = decode_frame_obus_with(&input, &config)
        .err()
        .expect("the 4:2:2 invalid-chroma-subsize POC must be REJECTED, not decoded");
    assert_eq!(err.category(), "malformed", "got {err}");
    let msg = err.to_string();
    assert!(
        msg.contains("invalid with subsampling") || msg.contains("invalid chroma block size"),
        "the rejection must name the chroma-size condition, got: {msg}"
    );
}

/// A readable-but-out-of-spec-range syntax value and a bit reader that ran off
/// the end of the payload are DIFFERENT failures for a consumer (corrupt file
/// vs short file). They used to share one hedged message
/// ("bit-reader error / out-of-range syntax value") and one category; this pins
/// the split — the malformed side must name the field that was out of range.
#[test]
fn out_of_range_header_syntax_is_malformed_and_names_the_field() {
    let path = regression_dir().join("decode_obus_filmgrain_num_points_oob.obu");
    let input = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let err = decode_frame_obus_with(&input, &fuzz_config())
        .err()
        .expect("the film-grain num_points POC must be rejected");
    assert_eq!(err.category(), "malformed", "got {err}");
    let msg = err.to_string();
    assert!(
        msg.contains("num_y_points") || msg.contains("num_cb_points") || msg.contains("num_cr_points"),
        "the rejection must name the out-of-range field, got: {msg}"
    );
}

/// The truncated side of the same split: a valid stream cut short mid-header
/// must report `truncated`, not `malformed`. Built by truncating a committed
/// seed, so it cannot drift out of sync with the real header layout. The seed
/// itself must decode, otherwise the test proves nothing about truncation.
#[test]
fn truncated_header_is_reported_as_truncated_not_malformed() {
    let seed = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/seeds/decode_obus/av1-1-b8-01-size-16x16-key.obu");
    let full = fs::read(&seed).unwrap_or_else(|e| panic!("read {}: {e}", seed.display()));
    let config = fuzz_config();
    assert!(
        decode_frame_obus_with(&full, &config).is_ok(),
        "the seed must decode, else truncating it proves nothing"
    );

    // Walk prefixes and require that EVERY rejection is a truncation-shaped
    // one. A prefix can also be rejected by the OBU/leb128 layer (also
    // `truncated`) or, once the OBU size field itself is cut, by a size
    // mismatch — all of which are input-ended failures, never `internal`.
    let mut saw_truncated = 0usize;
    for keep in 1..full.len() {
        let err = match decode_frame_obus_with(&full[..keep], &config) {
            Ok(_) => continue,
            Err(e) => e,
        };
        assert_ne!(err.category(), "internal", "prefix of {keep} bytes: {err}");
        if err.category() == "truncated" {
            saw_truncated += 1;
        }
    }
    assert!(
        saw_truncated > 0,
        "no prefix of the seed was reported as `truncated` — the truncated/malformed \
         split is not reachable, so the distinction is decorative"
    );
    eprintln!("{saw_truncated} of {} prefixes reported truncated", full.len() - 1);
}

// ---- the invariant the chroma guards rest on -------------------------------
//
// Both new rejections (and `max_uv_txsize`'s named panic) are justified by the
// shape of `av1_ss_size_lookup` (common_data.c:17): BLOCK_INVALID appears ONLY
// in the ss=(0,1) and ss=(1,0) columns. 4:4:0 (0,1) cannot be coded by a
// sequence header, so 4:2:2 is the only reachable hole — which is why the
// decoder's 4:2:2-scoped `decode_partition` guard suffices for conformant
// streams. Pin that here so a table edit cannot silently invalidate the
// reasoning in those comments.

/// `BLOCK_SIZES_ALL`.
const BLOCK_SIZES_ALL: usize = 22;

#[test]
fn block_invalid_chroma_sizes_occur_only_at_422_and_440() {
    use aom_dsp::entropy::partition::get_plane_block_size;
    let mut holes = Vec::new();
    for bsize in 0..BLOCK_SIZES_ALL {
        for ss_x in 0..2 {
            for ss_y in 0..2 {
                if get_plane_block_size(bsize, ss_x, ss_y) == 255 {
                    holes.push((bsize, ss_x, ss_y));
                }
            }
        }
    }
    assert!(!holes.is_empty(), "the table has no BLOCK_INVALID entries at all — the guards \
        that reject them, and this test, are then testing nothing");
    for &(bsize, ss_x, ss_y) in &holes {
        assert!(
            (ss_x, ss_y) == (1, 0) || (ss_x, ss_y) == (0, 1),
            "bsize {bsize} has no chroma size at subsampling ({ss_x},{ss_y}) — outside the \
             4:2:2 / 4:4:0 columns the decoder's 4:2:2-scoped guard covers"
        );
    }
    // 4:2:0 and 4:4:4, the two configurations the decoder is exercised on
    // everywhere else, must be hole-free.
    for bsize in 0..BLOCK_SIZES_ALL {
        assert_ne!(get_plane_block_size(bsize, 1, 1), 255, "4:2:0 bsize {bsize}");
        assert_ne!(get_plane_block_size(bsize, 0, 0), 255, "4:4:4 bsize {bsize}");
    }
    eprintln!("{} BLOCK_INVALID entries, all at (1,0)/(0,1)", holes.len());
}

/// `max_uv_txsize` on a BLOCK_INVALID pair must fail with a message that says
/// what the CALLER did wrong. It used to be a `debug_assert_ne!` — compiled out
/// in release, where the same input then died as a bare
/// `MAX_TXSIZE_RECT_LOOKUP` "index out of bounds". This test runs in whatever
/// profile CI uses, so it pins the named form in both.
#[test]
#[should_panic(expected = "no valid chroma plane size")]
fn max_uv_txsize_names_the_broken_contract() {
    // BLOCK_32X64 (10) at 4:2:2 — the exact shape the committed 4:2:2 POC
    // drives the decoder to, per that POC's rejection message.
    let _ = aom_decode::max_uv_txsize(10, 1, 0);
}
