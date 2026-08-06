//! STABLE-PATH DECODER FUZZ SWEEP (no nightly / cargo-fuzz required).
//!
//! Complements `fuzz_regression.rs` (which REPLAYS the committed crash POCs): this
//! adds a seeded structured-random MUTATION sweep over the committed seeds — the
//! discovery mechanism that finds NEW escaping panics a replay-only gate cannot.
//!
//! The `crates/aom-decode/fuzz/` targets need a nightly toolchain + `cargo-fuzz`
//! + libFuzzer. This test gives the same robustness contract a home on **stable**
//! `cargo test`, so CI enforces it on every platform without nightly:
//!
//!   for ANY input, `decode_frame_obus` / `decode_frames` return `Ok` / `Err(String)`
//!   — never a panic (unwrap / expect / out-of-bounds slice / `assert!` / debug
//!   arithmetic overflow) and never an unbounded allocation.
//!
//! Two parts, both on stable:
//!   1. REPLAY — every committed `fuzz/regression/*` reproducer (minimized POCs of
//!      fixed crashes) and every `fuzz/seeds/**` seed is run through both public
//!      OBU entry points; a panic fails the test. This is the
//!      `tests/fuzz_regression.rs` gate that keeps fixed crashes fixed.
//!   2. STRUCTURED-RANDOM SWEEP — a seeded XorShift mutates the committed seeds
//!      (bit flips, truncation, length-field corruption, header splicing, insert /
//!      delete, HOSTILE TILE PAYLOAD, payload extension) and asserts no entry
//!      panics. Self-contained (mutates only committed seeds — no external
//!      corpus), so it always runs in CI. The frame-dimension ceiling
//!      (`frame.rs`) keeps a mutated giant-dimension header from OOMing this
//!      in-process sweep.
//!
//! The last two mutation ops exist because few-bit flips leave a mostly-valid
//! arithmetic stream: the symbol decoder stays near the states a real encoder
//! produced. Replacing the whole tile payload with PRNG bytes (and letting the
//! range decoder read past the real tile end) is what drives the partition /
//! tx-size / palette / mode / coefficient machinery through the states a
//! CRAFTED bitstream reaches. The sweep reports a reach histogram and FAILS if
//! the deep-reach fraction collapses (see `MIN_DEEP_REACH_PPM`) — a no-panic
//! result over inputs the OBU parser rejected is not evidence about the decoder.
//!
//! Every distinct panic found in the sweep is collected (deduped by message), its
//! reproducer written to `$FUZZ_CRASH_DIR` (default
//! `/root/fuzz-corpus/aom-rs/stable-crashes/`), and the test fails with the full
//! list — RED while any panic exists, GREEN once fixed.
//!
//! Knobs (env): `FUZZ_SMOKE_ITERS` sweep iterations (default 60000),
//! `FUZZ_SMOKE_SEED` PRNG seed, `FUZZ_CRASH_DIR`.

use aom_decode::{DecodeConfig, DecodeLimits};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

/// 4 Mpx (2048×2048) — the same low `max_pixels` the cargo-fuzz targets pin.
/// It bounds the peak per-frame allocation so an in-bounds-but-huge declared
/// frame is rejected with `LimitExceeded` instead of a multi-GiB allocation.
const FUZZ_MAX_PIXELS: u64 = 1 << 22;

/// Floor on the fraction of mutated inputs that must reach frame/tile-level
/// work (parts per million). MEASURED on this corpus + mutator, see the run
/// recorded in `benchmarks/decoder_panic_surface_2026-08-06.md`; the floor is
/// set well under the measured value so ordinary corpus churn cannot trip it,
/// but a mutator or seed change that collapses reach to header-parse noise
/// will. Not a quality threshold to be relaxed — if it fires, the sweep stopped
/// testing the decoder.
const MIN_DEEP_REACH_PPM: u64 = 100_000; // 10%

fn fuzz_config() -> DecodeConfig<'static> {
    let mut limits = DecodeLimits::default();
    limits.max_pixels = Some(FUZZ_MAX_PIXELS);
    DecodeConfig::default().with_limits(limits)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

fn fuzz_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/aom-decode
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz")
}

/// Recursively load every regular file under `sub` (skipping dotfiles).
fn load_tree(sub: &str) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let root = fuzz_root().join(sub);
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut ents: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        ents.sort_by_key(|e| e.file_name());
        for e in ents {
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(bytes) = std::fs::read(&p) {
                out.push((name.to_string(), bytes));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// How far a mutated input actually got. A no-panic sweep is only evidence if
/// the inputs REACHED the code under test: an input rejected by the OBU header
/// parse never exercises the tile walk, so a sweep made entirely of those is
/// green for the wrong reason. The counters are reported at the end of the run
/// (`--nocapture`) and floored by [`MIN_DEEP_REACH_PPM`].
#[derive(Default)]
struct Reach {
    /// Decoded a full frame (deepest reach: tile walk + post-filters ran).
    decoded: u64,
    /// Rejected AFTER the frame header parsed — the tile walk was entered or
    /// an in-envelope check fired, i.e. the input was structurally a frame.
    deep_err: u64,
    /// Rejected at/ before the header parse — shallow, header-shaped noise.
    shallow_err: u64,
}

/// A decode error whose text proves the input got past OBU/sequence-header
/// parsing into frame-level work. Kept as substrings (not variants) so the
/// classification cannot silently drift when a message is reworded — a
/// misclassification only makes the floor HARDER to meet, never easier.
fn is_deep_err(msg: &str) -> bool {
    msg.contains("corrupt frame")
        // A frame-header rejection means the OBU layer and the sequence header
        // both parsed — the input was frame-shaped, not header-parse noise.
        || msg.contains("frame header")
        || msg.contains("tile")
        || msg.contains("partition")
        || msg.contains("segment")
        || msg.contains("intrabc")
        || msg.contains("interp filter")
        || msg.contains("film grain")
        || msg.contains("unsupported feature")
        || msg.contains("limit exceeded")
}

/// Run one input through both public decode entries under `catch_unwind`.
/// Returns the panic message of the first entry that panics, else `None`.
/// Accumulates how deep the input got into `reach`.
fn probe(data: &[u8], reach: &mut Reach) -> Option<String> {
    let config = fuzz_config();
    let outcome = catch_unwind(AssertUnwindSafe(
        || match aom_decode::frame::decode_frame_obus_with(data, &config) {
            Ok(_) => (true, String::new()),
            Err(e) => (false, e.to_string()),
        },
    ));
    match outcome {
        Err(p) => return Some(format!("[decode_frame_obus] {}", panic_msg(&p))),
        Ok((true, _)) => reach.decoded += 1,
        Ok((false, msg)) => {
            if is_deep_err(&msg) {
                reach.deep_err += 1;
            } else {
                reach.shallow_err += 1;
            }
        }
    }
    if let Err(p) = catch_unwind(AssertUnwindSafe(|| {
        let _ = aom_decode::frame::decode_frames_with(data, &config);
    })) {
        return Some(format!("[decode_frames] {}", panic_msg(&p)));
    }
    None
}

fn crash_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("FUZZ_CRASH_DIR")
            .unwrap_or_else(|_| "/root/fuzz-corpus/aom-rs/stable-crashes".to_string()),
    )
}

fn mutate(rng: &mut Rng, base: &[u8], others: &[Vec<u8>]) -> Vec<u8> {
    let mut v = base.to_vec();
    match rng.below(9) {
        7 => {
            // HOSTILE TILE PAYLOAD: keep the leading bytes (OBU headers +
            // sequence/frame header live there) and replace the whole tail with
            // PRNG bytes. The few-bit flips above leave a mostly-valid arithmetic
            // stream, so the symbol decoder stays near the states a real encoder
            // produced; a random payload drives `OdEcDec` through arbitrary
            // symbol sequences instead — the partition / tx-size / palette /
            // mode / coefficient states a crafted bitstream reaches and a
            // mutated-real-stream almost never does. Offsets are biased into
            // 8..=56 so the seq/frame header usually survives and the decoder
            // actually gets as far as the tile walk.
            if v.len() > 8 {
                let at = 8 + rng.below(v.len().min(56) - 7);
                for b in &mut v[at..] {
                    *b = (rng.next() & 0xff) as u8;
                }
            }
        }
        8 => {
            // PAYLOAD EXTENSION: append PRNG bytes so the range decoder keeps
            // finding readable bytes past the real tile end instead of latching
            // `OD_EC_LOTS_OF_BITS` early. Combined with a leb128 size bump this
            // is what lets a crafted stream keep the tile walk running long
            // after a conformant one would have stopped.
            let extra = 1 + rng.below(2048);
            for _ in 0..extra {
                v.push((rng.next() & 0xff) as u8);
            }
        }
        0 => {
            // bit flips (1..=6)
            for _ in 0..1 + rng.below(6) {
                if v.is_empty() {
                    break;
                }
                let i = rng.below(v.len());
                v[i] ^= 1u8 << rng.below(8);
            }
        }
        1 => {
            // truncate to a random prefix
            if !v.is_empty() {
                let keep = rng.below(v.len() + 1);
                v.truncate(keep);
            }
        }
        2 => {
            // length-field corruption: hammer the early bytes (OBU header + size
            // leb128 + frame-size fields all live in the first ~48 bytes).
            let span = v.len().min(48);
            for _ in 0..1 + rng.below(6) {
                if span == 0 {
                    break;
                }
                let i = rng.below(span);
                v[i] = (rng.next() & 0xff) as u8;
            }
        }
        3 => {
            // splice: overwrite a random middle run with another seed's bytes
            if !others.is_empty() && !v.is_empty() {
                let src = &others[rng.below(others.len())];
                if !src.is_empty() {
                    let at = rng.below(v.len());
                    let take = 1 + rng.below(src.len());
                    let so = rng.below(src.len());
                    for k in 0..take {
                        if at + k >= v.len() || so + k >= src.len() {
                            break;
                        }
                        v[at + k] = src[so + k];
                    }
                }
            }
        }
        4 => {
            // concatenate two seeds (multi-OBU / multi-frame splicing)
            if !others.is_empty() {
                v.extend_from_slice(&others[rng.below(others.len())]);
            }
        }
        5 => {
            // random byte insertion
            for _ in 0..1 + rng.below(8) {
                let at = rng.below(v.len() + 1);
                v.insert(at, (rng.next() & 0xff) as u8);
            }
        }
        _ => {
            // random byte deletion
            for _ in 0..1 + rng.below(8) {
                if v.is_empty() {
                    break;
                }
                let at = rng.below(v.len());
                v.remove(at);
            }
        }
    }
    // Cap mutated size so splice/concat chains cannot balloon (real in-scope
    // temporal units are tiny); keeps the sweep fast.
    v.truncate(8192);
    v
}

#[test]
fn decoder_sweep_never_panics_on_mutated_input() {
    let regressions = load_tree("regression");
    let seeds = load_tree("seeds");
    assert!(
        !seeds.is_empty(),
        "no fuzz seeds found under {} — the seed corpus must be committed",
        fuzz_root().join("seeds").display()
    );

    // Silence the default panic hook during replay+sweep so a would-be panic
    // (caught by probe) does not flood stderr; restore before the final report.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    // ---- part 1: replay every committed reproducer + seed (hard gate) -------
    let mut replay_reach = Reach::default();
    let mut replay_failures: Vec<(String, String)> = Vec::new();
    for (name, bytes) in regressions.iter().chain(seeds.iter()) {
        if let Some(msg) = probe(bytes, &mut replay_reach) {
            replay_failures.push((name.clone(), msg));
        }
    }

    // ---- part 2: structured-random sweep over the seeds ---------------------
    let iters: u64 = std::env::var("FUZZ_SMOKE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60_000);
    let seed: u64 = std::env::var("FUZZ_SMOKE_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let mut rng = Rng(seed | 1);

    let corpus: Vec<Vec<u8>> = seeds
        .iter()
        .chain(regressions.iter())
        .map(|(_, b)| b.clone())
        .collect();

    let mut reach = Reach::default();
    let mut distinct: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    for _ in 0..iters {
        let base = &corpus[rng.below(corpus.len())];
        let input = mutate(&mut rng, base, &corpus);
        if let Some(msg) = probe(&input, &mut reach) {
            distinct.entry(msg).or_insert(input);
        }
    }

    std::panic::set_hook(prev_hook);

    // ---- reach report + floor ----------------------------------------------
    // A green sweep only means something if the inputs got past the header
    // parse. Report the histogram, and FAIL if the deep fraction collapses —
    // that is the signal that a mutation op or a seed change quietly turned the
    // sweep into a header-parser test.
    let deep = reach.decoded + reach.deep_err;
    let total = deep + reach.shallow_err;
    let deep_ppm = if total == 0 {
        0
    } else {
        deep.saturating_mul(1_000_000) / total
    };
    println!(
        "fuzz sweep reach: {} decoded, {} deep-err, {} shallow-err of {total} \
         ({deep_ppm} ppm deep); replay {} decoded / {} deep-err / {} shallow-err",
        reach.decoded,
        reach.deep_err,
        reach.shallow_err,
        replay_reach.decoded,
        replay_reach.deep_err,
        replay_reach.shallow_err,
    );
    assert!(
        deep_ppm >= MIN_DEEP_REACH_PPM,
        "fuzz sweep no longer reaches the frame/tile decoder: only {deep_ppm} ppm of \
         {total} mutated inputs got past the header parse (floor {MIN_DEEP_REACH_PPM} ppm). \
         A sweep that only exercises the OBU parser cannot certify the tile walk."
    );

    // ---- report -------------------------------------------------------------
    if replay_failures.is_empty() && distinct.is_empty() {
        return; // GREEN: no panic escaped any public entry.
    }

    let dir = crash_dir();
    let _ = std::fs::create_dir_all(&dir);
    let mut report = String::new();
    if !replay_failures.is_empty() {
        report.push_str(&format!(
            "\n{} committed seed/regression input(s) PANIC (must never happen):\n",
            replay_failures.len()
        ));
        for (name, msg) in &replay_failures {
            report.push_str(&format!("  - {name}: {msg}\n"));
        }
    }
    if !distinct.is_empty() {
        report.push_str(&format!(
            "\n{} DISTINCT panic(s) found by the structured-random sweep:\n",
            distinct.len()
        ));
        for (i, (msg, input)) in distinct.iter().enumerate() {
            let path = dir.join(format!("stable-crash-{i:03}.obu"));
            let _ = std::fs::write(&path, input);
            report.push_str(&format!(
                "  - {msg}\n      ({} bytes) reproducer: {}\n",
                input.len(),
                path.display()
            ));
        }
    }
    panic!("{report}\nseed={seed:#x} iters={iters}");
}
