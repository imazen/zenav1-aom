//! `winperf_bootstrap_gen` — regenerate the committed `winperf` bootstrap.
//!
//! Requires the `c-oracle` feature (it runs a real `aomenc --allintra` encode),
//! which is exactly why the harness itself does NOT: this runs once, on a box
//! that can build libaom, and commits its output as
//! `crates/aom-bench/fixtures/winperf_bootstrap_<w>x<h>_cq<q>_s<speed>.hex`.
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example winperf_bootstrap_gen
//! ```
//!
//! Re-run it if [`aom_bench::winperf::synth_i420`] or the study cell changes —
//! the bootstrap's frame header is derived from the same `(w, h, cq, speed,
//! format)` the harness encodes at, and the pinned source checksum in
//! `winperf.rs`'s tests is what makes a silent drift impossible.

use aom_bench::winperf;

fn main() {
    let (w, h, q, s) = winperf::CELL;
    // Optional filter: regenerating ONE content leaves the others' committed
    // bytes untouched, which matters because every recorded winperf band was
    // taken against the fixture in the tree at the time.
    let only: Vec<String> = std::env::args().skip(1).collect();
    for content in winperf::Content::ALL {
        if !only.is_empty() && !only.iter().any(|a| a == content.label()) {
            continue;
        }
        gen_one(content, w, h, q, s);
    }
    // The gate cell: `Content::Screen` only, and only because a 1 MP
    // palette+intraBC encode is minutes of search — see
    // `winperf::SCREEN_GATE_CELL`.
    if only.is_empty() || only.iter().any(|a| a == "screen") {
        let (gw, gh, gq, gs) = winperf::SCREEN_GATE_CELL;
        gen_one(winperf::Content::Screen, gw, gh, gq, gs);
    }
}

fn gen_one(content: winperf::Content, w: usize, h: usize, q: i32, s: i32) {
    {
        let cell = winperf::cell(w, h, q, s, content);
        // `Screen` bootstraps through the SCREEN config so its frame header can
        // carry `allow_screen_content_tools` — aomenc's own detection decides,
        // and the assertion below is what proves it fired. Without that bit the
        // palette / intraBC searches decline and the content measures nothing.
        let boot = if content == winperf::Content::Screen {
            let b = cell.c_encode_screen(true, true);
            assert!(
                aom_bench::stream_allows_screen_content_tools(&b),
                "winperf::Content::Screen: real aomenc did NOT signal \
                 allow_screen_content_tools on this source — the generator \
                 parameters need rework, not the fixture"
            );
            b
        } else {
            cell.c_encode_defaults()
        };
        assert!(!boot.is_empty(), "C bootstrap encode failed");
        // The harness consumes only the SEQUENCE HEADER and the FRAME OBU's
        // uncompressed header; every coded byte after that is dead weight in a
        // committed fixture. `screen` codes 41 KB (83 KB as hex) because
        // screen content at cq44 is expensive, which is well past what belongs
        // in git, so its frame OBU payload is trimmed to the header and the
        // OBU size field rewritten. The equality assertion below is what makes
        // that safe: the port must emit the SAME bytes from the trimmed
        // bootstrap as from the whole one.
        //
        // The three older fixtures are deliberately NOT trimmed — every
        // recorded winperf band was taken against the bytes in the tree, and
        // rewriting them would invalidate that provenance for no benefit.
        let boot = if content == winperf::Content::Screen {
            let trimmed = trim_to_headers(&boot, 128);
            assert_eq!(
                cell.port_encode(&trimmed),
                cell.port_encode(&boot),
                "{}: trimming the bootstrap to its headers changed the port's \
                 output — the trim is not safe, ship the whole stream",
                content.label()
            );
            assert_eq!(
                aom_bench::stream_allows_screen_content_tools(&trimmed),
                aom_bench::stream_allows_screen_content_tools(&boot),
                "trimming lost allow_screen_content_tools"
            );
            println!("  trimmed {} -> {} bytes", boot.len(), trimmed.len());
            trimmed
        } else {
            boot
        };
        let path = format!(
            "crates/aom-bench/fixtures/winperf_bootstrap_{w}x{h}_cq{q}_s{s}_{}.hex",
            content.label()
        );
        let mut out = String::new();
        for (i, b) in boot.iter().enumerate() {
            if i > 0 && i % 32 == 0 {
                out.push('\n');
            }
            out.push_str(&format!("{b:02x}"));
        }
        out.push('\n');
        std::fs::write(&path, &out).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("wrote {path}: {} bootstrap bytes -> {} chars", boot.len(), out.len());

        // Also print the source-plane checksums the winperf unit test pins, so
        // a regeneration and a re-pin are one step rather than two.
        let buf = winperf::synth_i420(w, h, content);
        let (cw, ch) = (w / 2, h / 2);
        let sum = |x: &[u8]| x.iter().map(|&b| u64::from(b)).sum::<u64>();
        println!(
            "  synth_i420({w},{h},{content:?}) plane sums: y={} u={} v={}",
            sum(&buf[..w * h]),
            sum(&buf[w * h..w * h + cw * ch]),
            sum(&buf[w * h + cw * ch..])
        );
    }
}

/// Copy `stream`, truncating the payload of the combined `OBU_FRAME` to its
/// first `keep` bytes and rewriting that OBU's leb128 size field.
///
/// The AV1 uncompressed frame header is tens of bits, so `keep = 128` is far
/// more than it can read; the caller asserts byte-equality of the port's output
/// with and without the trim rather than relying on that.
fn trim_to_headers(stream: &[u8], keep: usize) -> Vec<u8> {
    const OBU_FRAME: u32 = 6;
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < stream.len() {
        let hdr = aom_dsp::entropy::obu::read_obu_header(&stream[pos..]).expect("obu header");
        let after = pos + hdr.header_len;
        let (size, nb) = aom_dsp::entropy::leb128::uleb_decode(&stream[after..]).expect("leb128");
        let (start, end) = (after + nb, after + nb + size as usize);
        out.extend_from_slice(&stream[pos..after]); // the OBU header byte(s)
        let payload = if hdr.obu_type == OBU_FRAME && size as usize > keep {
            &stream[start..start + keep]
        } else {
            &stream[start..end]
        };
        // leb128, minimal encoding (what the C encoder emits for these sizes).
        let mut n = payload.len() as u64;
        loop {
            let mut b = (n & 0x7f) as u8;
            n >>= 7;
            if n != 0 {
                b |= 0x80;
            }
            out.push(b);
            if n == 0 {
                break;
            }
        }
        out.extend_from_slice(payload);
        pos = end;
    }
    out
}
