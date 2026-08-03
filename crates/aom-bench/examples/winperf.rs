//! `winperf` — timed encode driver with **no C oracle**, for the Windows
//! allocation A/B (`.github/workflows/winperf.yml`).
//!
//! Same driver contract as `benchmarks/xbench/drv-aom`, minus the `.yuv`
//! argument (the source is generated — [`aom_bench::winperf::synth_i420`]) and
//! minus the live C bootstrap (it is a committed fixture):
//!
//! ```text
//! winperf <detail|smooth> <warmup> <reps> [out.obu]
//! stdout: `NS=<n> NS=<n> ... BYTES=<m> FRAMEBYTES=<m>`
//! ```
//!
//! The timed region is exactly `drv-aom`'s: one `EncodeCell::port_encode`, the
//! port's whole frame encode (header-field bootstrap parse, quantizer + cost
//! tables, source copy + border extension, the SB search+pack walk, loop-filter
//! level search, OBU assembly). Reading the fixture and synthesising the source
//! happen once, untimed.
//!
//! Cell is fixed at [`aom_bench::winperf::CELL`] on purpose: the bootstrap
//! fixture's frame header belongs to that `(size, cq, cpu-used, format)` and
//! nothing else, so accepting the cell on the command line would only let a
//! caller silently mismatch them.
//!
//! `FRAMEBYTES` is printed every invocation and is the harness's
//! did-both-arms-do-the-same-work check: an allocation change that altered a
//! coding decision would move it. It is NOT a byte-exactness proof — no C
//! encoder runs here (see the module docs of `aom_bench::winperf`).

use aom_bench::winperf;
use std::time::Instant;

const OBU_FRAME: u8 = 6;

/// Minimal section-5 OBU walk: `(type, header_bytes, payload)` per OBU.
/// Verbatim from `drv-aom`, so the reported `BYTES` is the same quantity.
fn walk(mut s: &[u8]) -> Vec<(u8, Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    while !s.is_empty() {
        let mut hdr = vec![s[0]];
        let ty = (s[0] >> 3) & 0xf;
        let ext = (s[0] >> 2) & 1;
        let has_size = (s[0] >> 1) & 1;
        assert_eq!(has_size, 1, "obu_has_size_field must be set");
        let mut i = 1;
        if ext == 1 {
            hdr.push(s[1]);
            i = 2;
        }
        let mut size = 0usize;
        let mut shift = 0;
        loop {
            let b = s[i];
            i += 1;
            size |= ((b & 0x7f) as usize) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        out.push((ty, hdr, s[i..i + size].to_vec()));
        s = &s[i + size..];
    }
    out
}

fn leb128(mut v: usize) -> Vec<u8> {
    let mut o = Vec::new();
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        o.push(b);
        if v == 0 {
            return o;
        }
    }
}

fn reassemble(bootstrap: &[u8], frame_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (ty, hdr, payload) in walk(bootstrap) {
        let p: &[u8] = if ty == OBU_FRAME { frame_payload } else { &payload };
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&leb128(p.len()));
        out.extend_from_slice(p);
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 || a.len() > 5 {
        eprintln!("usage: winperf <detail|smooth> <warmup> <reps> [out.obu]");
        std::process::exit(2);
    }
    let content = winperf::Content::parse(&a[1]);
    let warmup: usize = a[2].parse().expect("warmup");
    let reps: usize = a[3].parse().expect("reps");

    let (w, h, q, s) = winperf::CELL;
    let cell = winperf::cell(w, h, q, s, content);
    let bootstrap = winperf::bootstrap(content);

    for _ in 0..warmup {
        let _ = cell.port_encode(&bootstrap);
    }
    let mut samples = Vec::with_capacity(reps);
    let mut last = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let payload = cell.port_encode(&bootstrap);
        samples.push(t.elapsed().as_nanos());
        last = payload;
    }

    let stream = reassemble(&bootstrap, &last);
    if let Some(p) = a.get(4) {
        std::fs::write(p, &stream).unwrap_or_else(|e| panic!("write {p}: {e}"));
    }
    let mut line = String::new();
    for n in &samples {
        line.push_str(&format!("NS={n} "));
    }
    println!("{line}BYTES={} FRAMEBYTES={}", stream.len(), last.len());
}
