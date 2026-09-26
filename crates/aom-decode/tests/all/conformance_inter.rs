//! GATE-1 CONFORMANCE — INTER scope under `experimental-video`.
//!
//! The intra gate (`conformance_corpus.rs`) covers the caller-visible
//! KEY-frame surface. This is the INTER complement: the five inter-scope
//! vectors (`05-mv`, `06-mfmv`, `22-svc-L{1,2}T{1,2}`) are multi-frame streams
//! that exercise motion compensation, multi-reference, compound prediction,
//! scaled references and SVC temporal layering.
//!
//! # The pin
//!
//! [`expected_for`] records each fetched vector's pinned outcome:
//!   - [`Outcome::ByteExact`]: the port must decode every shown frame
//!     byte-identical to the REAL C decoder AND to the shipped golden MD5.
//!   - [`Outcome::Refused`]: the port must refuse the stream by a named
//!     envelope boundary whose message contains the pinned substring.
//!
//! The pin is the byte-exact SUBSET, stated explicitly so a regression is
//! loud: any present vector whose outcome changed — a `ByteExact` vector that
//! now refuses or diverges, OR a `Refused` vector that started decoding —
//! FAILS LOUD and the pin must be updated deliberately. A byte DIVERGENCE is
//! never a pinned outcome; it always fails (a diverged vector is a finding
//! recorded in `STATUS.md` / `docs/HANDOFF-EXPERIMENTAL-VIDEO.md`, not a
//! stable expectation). A PANIC is likewise never a pinned refusal — a
//! conformant official vector that panics is a bug, not an envelope boundary.
//!
//! # Skip-by-name
//!
//! The inter corpus is fetched with `python3 xtask/conformance.py --fetch
//! --scope inter`. CI provisions only the `--scope intra` corpus, so when no
//! inter vector is present this test SKIPS BY NAME (logs and returns) rather
//! than failing — unlike the intra gate, whose vectors are always provisioned
//! in CI. It is also a feature gate: the inter envelope is only claimed under
//! `experimental-video`, so it skips-by-name feature-off too.

use aom_decode::frame::{FrameDecode, decode_frames};
use aom_sys_ref as c;
use std::path::PathBuf;

use crate::common::md5::Md5;

// libaom `md5_helper.h::Add(aom_image_t*)`: per-plane cropped rows, 1
// byte/sample at bd8 else 2 (LE). Chroma dims round up. `mono` synthesizes the
// neutral 4:2:0 chroma the shipped golden hashes.
fn image_md5(f: &FrameDecode) -> String {
    let (w, h) = (f.width, f.height);
    let hi = f.bit_depth > 8;
    let (ss_x, ss_y) = (f.subsampling_x, f.subsampling_y);
    let mut m = Md5::new();
    let push = |plane: &[u16], pw: usize, ph: usize, m: &mut Md5| {
        let mut row = Vec::with_capacity(pw * if hi { 2 } else { 1 });
        for r in 0..ph {
            row.clear();
            for &s in &plane[r * pw..r * pw + pw] {
                if hi {
                    row.extend_from_slice(&s.to_le_bytes());
                } else {
                    row.push(s as u8);
                }
            }
            m.update(&row);
        }
    };
    push(&f.y, w, h, &mut m);
    let (cw, ch) = ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y);
    if f.monochrome {
        let neutral = vec![1u16 << (f.bit_depth - 1); cw * ch];
        push(&neutral, cw, ch, &mut m);
        push(&neutral, cw, ch, &mut m);
    } else {
        push(&f.u, cw, ch, &mut m);
        push(&f.v, cw, ch, &mut m);
    }
    m.finish()
}

fn corpus_dir() -> PathBuf {
    if let Ok(d) = std::env::var("AOM_CONFORMANCE_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("data")
}

fn ivf_temporal_units(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 12; // 4-byte size + 8-byte timestamp
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

fn ivf_hdr_dims(data: &[u8]) -> (usize, usize) {
    (
        u16::from_le_bytes([data[12], data[13]]) as usize,
        u16::from_le_bytes([data[14], data[15]]) as usize,
    )
}

fn parse_golden(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().next().unwrap().to_ascii_lowercase())
        .collect()
}

/// The pinned outcome for a fetched inter vector.
#[allow(dead_code)] // `ByteExact` is the passing arm — un-constructed while the subset is empty
enum Outcome {
    /// Every shown frame must decode byte-identical to the C decoder AND to
    /// the shipped golden MD5. This is the byte-exact subset.
    ByteExact,
    /// The port must refuse the stream by a named envelope boundary whose
    /// error message contains this substring.
    Refused(&'static str),
}

/// True when `name` is one of the inter-scope families (`--scope inter` in
/// `xtask/conformance.py`). Intra-scope vectors present in the same corpus dir
/// belong to `conformance_corpus.rs`, not this gate.
fn is_inter_vector(name: &str) -> bool {
    name.contains("-05-mv") || name.contains("-06-mfmv") || name.contains("-22-svc")
}

/// The caller-visible outcome pin. Every fetched inter vector is listed
/// EXPLICITLY; a present vector not listed here fails loud as an unclassified
/// family. As of step-4 close-out all five inter vectors refuse by name — the
/// byte-exact subset is empty, and the two reachable boundaries are recorded
/// in `STATUS.md` / `docs/HANDOFF-EXPERIMENTAL-VIDEO.md` / `COVERAGE_QUEUE` T3.
fn expected_for(name: &str) -> Option<Outcome> {
    if name.contains("-05-mv") || name.contains("-06-mfmv") {
        // Both hit `inter.gm_wmtype[ref] != 0` — a reference carrying a
        // non-identity global-motion model (lib.rs "non-identity global
        // motion" refusal). Reachable inter tool → COVERAGE_QUEUE T3.
        Some(Outcome::Refused("non-identity global motion"))
    } else if name.contains("-22-svc") {
        // The SVC vectors split each frame's tiles across >1 tile-group OBU
        // (`--num-tile-groups>1`), refused at `read_full_tile_group`
        // (frame.rs). Reachable bitstream feature → COVERAGE_QUEUE T3.
        Some(Outcome::Refused("partial tile group"))
    } else {
        None
    }
}

/// Enumerate `av1-1-*.ivf` names present in `dir`, sorted.
fn present_vectors(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter_map(|n| n.strip_suffix(".ivf").map(str::to_owned))
            .filter(|n| n.starts_with("av1-1-"))
            .collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    names
}

/// What the port actually did on a vector, classified for the pin compare.
enum Actual {
    /// `decode_frames` produced frames; byte-exactness is checked separately.
    Decoded(Vec<FrameDecode>),
    /// `decode_frames` returned a `DecodeError` — a named envelope refusal
    /// (or a malformed-bitstream rejection); carries the display message.
    Refused(String),
    /// `decode_frames` panicked — always a hard failure, never a pin.
    Panicked,
}

fn try_decode(stream: &[u8]) -> Actual {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let res = std::panic::catch_unwind(|| decode_frames(stream));
    std::panic::set_hook(hook);
    match res {
        Ok(Ok(f)) => Actual::Decoded(f),
        Ok(Err(e)) => Actual::Refused(e.to_string()),
        Err(_) => Actual::Panicked,
    }
}

/// First diverging (frame, plane, sb) between the port and C frames, or None.
fn first_divergence(pf: &[FrameDecode], cf: &[c::RefDecodedFrame]) -> Option<String> {
    if pf.len() != cf.len() {
        return Some(format!("frame count port={} c={}", pf.len(), cf.len()));
    }
    for (i, (p, cfr)) in pf.iter().zip(cf.iter()).enumerate() {
        let cw = cfr.info[4] as usize;
        if p.y != cfr.y {
            let n = p.y.iter().zip(&cfr.y).position(|(a, b)| a != b).unwrap();
            let (row, col) = (n / cw, n % cw);
            return Some(format!(
                "f{i} Y @(r{row},c{col}) sb({},{})",
                row / 64,
                col / 64
            ));
        }
        if p.u != cfr.u {
            return Some(format!("f{i} U differs"));
        }
        if p.v != cfr.v {
            return Some(format!("f{i} V differs"));
        }
    }
    None
}

#[test]
fn inter_conformance_vectors_pinned_outcomes() {
    // Feature gate: the inter decode envelope is only claimed under
    // `experimental-video`. Feature-off these streams are not a supported
    // surface, so skip-by-name rather than pin a different refusal set.
    if !aom_decode::EXPERIMENTAL_VIDEO {
        eprintln!(
            "inter_conformance_vectors_pinned_outcomes: SKIPPED — inter envelope is \
             `experimental-video`-only; run the feature-on leg (just test-next-video)"
        );
        return;
    }

    c::ref_init();
    let dir = corpus_dir();
    let inter_present: Vec<String> = present_vectors(&dir)
        .into_iter()
        .filter(|n| is_inter_vector(n))
        .collect();

    // Skip-by-name when the inter corpus is not fetched (CI provisions only
    // the `--scope intra` corpus). Unlike the intra gate's fail-loud-on-empty,
    // this is a deliberate skip so the intra-only CI legs stay green.
    if inter_present.is_empty() {
        eprintln!(
            "inter_conformance_vectors_pinned_outcomes: SKIPPED — no inter vectors in \
             {}; fetch via `python3 xtask/conformance.py --fetch --scope inter`",
            dir.display()
        );
        return;
    }

    let mut failures: Vec<String> = Vec::new();
    let mut report = String::new();
    let mut byte_exact: Vec<String> = Vec::new();
    let mut refused: Vec<String> = Vec::new();

    for name in &inter_present {
        let expected = match expected_for(name) {
            Some(e) => e,
            None => {
                failures.push(format!(
                    "{name}: unclassified inter vector — add it to expected_for()"
                ));
                continue;
            }
        };

        let ivf = std::fs::read(dir.join(format!("{name}.ivf")))
            .unwrap_or_else(|e| panic!("{name}: present vector unreadable: {e}"));
        let (w, h) = ivf_hdr_dims(&ivf);
        // The port consumes a raw OBU stream: concatenate every temporal
        // unit's OBU payload (IVF is libaom framing, not AV1 OBUs).
        let obu_stream: Vec<u8> = ivf_temporal_units(&ivf).concat();

        match try_decode(&obu_stream) {
            Actual::Panicked => {
                failures.push(format!(
                    "{name}: PANICKED — a conformant vector must never panic"
                ));
            }
            Actual::Refused(msg) => {
                refused.push(format!("{name} ({msg})"));
                match expected {
                    Outcome::Refused(sub) => {
                        if !msg.contains(sub) {
                            failures.push(format!(
                                "{name}: refused by '{msg}' but the pin expects a refusal \
                                 containing '{sub}' — the refusal name changed; update the pin"
                            ));
                        }
                    }
                    Outcome::ByteExact => {
                        failures.push(format!(
                            "{name}: pinned BYTE-EXACT but now REFUSES ('{msg}') — the \
                             passing subset shrank; investigate or update the pin"
                        ));
                    }
                }
            }
            Actual::Decoded(pf) => {
                // The port decoded — check byte-exactness vs C + golden MD5.
                let golden = parse_golden(
                    &std::fs::read_to_string(dir.join(format!("{name}.ivf.md5")))
                        .expect("ivf present but no .md5 golden"),
                );
                let mut cf = Vec::new();
                for i in 0..golden.len() {
                    match c::ref_decode_av1_stream_frame_opt(&ivf, i, w, h) {
                        Some(f) => cf.push(f),
                        None => break,
                    }
                }
                if let Some(d) = first_divergence(&pf, &cf) {
                    // A byte divergence is a finding, never a pinned outcome.
                    failures.push(format!("{name}: DIVERGED — {d}"));
                    continue;
                }
                for (i, f) in pf.iter().enumerate() {
                    if i < golden.len() && image_md5(f) != golden[i] {
                        failures.push(format!(
                            "{name}: port==C but frame {i} MD5 {} != golden {}",
                            image_md5(f),
                            golden[i]
                        ));
                    }
                }
                match expected {
                    Outcome::ByteExact => byte_exact.push(name.clone()),
                    Outcome::Refused(sub) => {
                        failures.push(format!(
                            "{name}: pinned Refused('{sub}') but now DECODES byte-exact — the \
                             byte-exact subset grew; update the pin to Outcome::ByteExact"
                        ));
                    }
                }
            }
        }
    }

    report.push_str(&format!(
        "\ninter conformance tally ({} vectors, feature-on):\n  byte-exact: {}\n  refused:    {}\n",
        inter_present.len(),
        byte_exact.len(),
        refused.len()
    ));
    for b in &byte_exact {
        report.push_str(&format!("    OK       {b}\n"));
    }
    for r in &refused {
        report.push_str(&format!("    REFUSED  {r}\n"));
    }
    eprint!("{report}");

    assert!(
        failures.is_empty(),
        "inter conformance gate: {} vector(s) FAILED:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
