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

use aom_decode::frame::{decode_frame_obus, decode_frames};
use std::fs;
use std::path::PathBuf;

fn regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

/// Feed one seed through both untrusted-input entry points. A panic here
/// unwinds with the seed name in the failure message (`#[test]` catches it).
fn run_all_entry_points(input: &[u8]) {
    // Multi-frame OBU stream (KEY + inter) — the superset entry.
    let _ = decode_frames(input);
    // Single KEY-frame temporal unit — the still-AVIF entry.
    let _ = decode_frame_obus(input);
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
