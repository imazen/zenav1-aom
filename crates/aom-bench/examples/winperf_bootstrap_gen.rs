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
    for content in winperf::Content::ALL {
        let cell = winperf::cell(w, h, q, s, content);
        let boot = cell.c_encode_defaults();
        assert!(!boot.is_empty(), "C bootstrap encode failed");
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
