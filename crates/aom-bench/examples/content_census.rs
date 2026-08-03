//! `content_census` — **what the encoder actually does to a source**, printed
//! as a TSV, so a harness content can be checked against a reference instead of
//! assumed to resemble one.
//!
//! # Why this exists
//!
//! `winperf`'s two synthetic sources were tuned until their **allocator call
//! count** bracketed the dev box's study photograph
//! (`crates/aom-bench/src/winperf.rs`). That is one axis. On a second axis —
//! which intra mode family the encoder picks — `detail` turns out to reach
//! `z1` **six times in a whole 1 MP frame** against the photograph's 8 520,
//! which is why KB-PERF-4's Windows band could not resolve a directional-intra
//! lever: the code under test never ran
//! (`benchmarks/encoder_intra_dir_i16_2026-08-03.md` §7,
//! `docs/DIFFERENTIAL_PLAYBOOK.md` §6b).
//!
//! That census was hand-applied throwaway instrumentation. This is the
//! committed, re-runnable version, and it is the artefact that makes the next
//! content choice checkable rather than a guess.
//!
//! # Running it
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --features census \
//!     --example content_census -- <source> [<source> ...]
//! ```
//!
//! Sources:
//!
//! * `winperf:detail` / `winperf:smooth` / `winperf:photo` — the committed
//!   generator plus its committed bootstrap fixture. Needs **no** C oracle, so
//!   this runs anywhere `winperf` itself runs.
//! * `yuv:<path>:<w>x<h>` — a raw 8-bit I420 file, bootstrapped by a real
//!   `aomenc` encode. Needs `--features c-oracle,census`. This is how the study
//!   photograph (the REFERENCE distribution) is censused.
//!
//! The **first** source is the reference: every later source additionally gets
//! an `L1` row, the sum over the intra-class axis of `|pct_px - ref_pct_px|`.
//! That single number is what a content is fitted on — see the `FIT` block.
//!
//! # This binary is not a timing binary
//!
//! The `census` feature puts thread-local counters on the intra predictor and
//! the forward transform. It is default-off, `winperf`/`winperf_alloc` are
//! built without it, and this example **fails loud** if the census comes back
//! empty — which is what a build without the feature produces.

use aom_bench::{EncodeCell, winperf};
use aom_dsp::census::{
    self, BSIZE_NAME, Counts, INTRA_CLASS, MODE_NAME, N_BSIZE, N_MODE, N_TX_SIZE, N_TX_TYPE,
    TX_SIZE_NAME, TX_TYPE_NAME,
};

/// A censused source: its label, its counts, and the bytes it coded.
struct Row {
    label: String,
    counts: Box<Counts>,
    coded_bytes: usize,
    /// `None` for the `winperf` sources (whose dimensions are [`winperf::CELL`]).
    px: usize,
}

fn main() {
    assert!(
        census::enabled(),
        "built without --features census: every counter is a no-op and this \
         tool would print a table of zeros"
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: content_census <source> [<source> ...]");

    let rows: Vec<Row> = args.iter().map(|s| census_one(s)).collect();

    // ---- per-source detail ------------------------------------------------
    for r in &rows {
        println!("########## {}", r.label);
        print_source(r);
        println!();
    }

    // ---- the comparison the content choice is made on ---------------------
    println!("########## COMPARE (reference = {})", rows[0].label);
    print!("axis\tkey");
    for r in &rows {
        print!("\t{}", r.label);
    }
    println!();
    let classpx = |c: &Counts, i: usize| -> f64 {
        let t = c.intra_total_px();
        if t == 0 { 0.0 } else { 100.0 * c.intra_px[i].iter().sum::<u64>() as f64 / t as f64 }
    };
    for (i, name) in INTRA_CLASS.iter().enumerate() {
        print!("intra_class_pct_px\t{name}");
        for r in &rows {
            print!("\t{:.2}", classpx(&r.counts, i));
        }
        println!();
    }
    print!("intra_dir_pct_px\tz1+z2+z3");
    for r in &rows {
        let t = r.counts.intra_total_px();
        print!("\t{:.2}", if t == 0 { 0.0 } else { 100.0 * r.counts.directional_px() as f64 / t as f64 });
    }
    println!();
    print!("intra_dir_pct_calls\tz1+z2+z3");
    for r in &rows {
        let t = r.counts.intra_total_calls();
        print!("\t{:.2}", if t == 0 { 0.0 } else { 100.0 * r.counts.directional_calls() as f64 / t as f64 });
    }
    println!();
    print!("intra_calls_total\t-");
    for r in &rows {
        print!("\t{}", r.counts.intra_total_calls());
    }
    println!();
    print!("intra_px_per_frame_px\t-");
    for r in &rows {
        print!("\t{:.2}", r.counts.intra_total_px() as f64 / r.px as f64);
    }
    println!();
    print!("fwd_tx_total\t-");
    for r in &rows {
        print!("\t{}", r.counts.fwd_tx.iter().flatten().sum::<u64>());
    }
    println!();
    print!("leaves_total\t-");
    for r in &rows {
        print!("\t{}", r.counts.leaf_bsize.iter().sum::<u64>());
    }
    println!();
    print!("coded_bytes\t-");
    for r in &rows {
        print!("\t{}", r.coded_bytes);
    }
    println!();

    // ---- the fit number ---------------------------------------------------
    //
    // L1 over the intra-class share vector, in percentage points. 0 = identical
    // distribution; the maximum is 200 (two disjoint distributions). It is a
    // deliberately blunt statistic: the point is to be able to SAY which
    // candidate is closer, not to model anything.
    println!();
    println!("########## FIT (L1 over intra-class pct_px, vs {})", rows[0].label);
    println!("source\tL1_intra_class_pp\tL1_leaf_bsize_pp\tL1_fwd_tx_pp");
    for r in &rows {
        let l1_class: f64 =
            (0..INTRA_CLASS.len()).map(|i| (classpx(&r.counts, i) - classpx(&rows[0].counts, i)).abs()).sum();
        let l1_bsize = l1_over(
            &(0..N_BSIZE).map(|b| r.counts.leaf_bsize[b]).collect::<Vec<_>>(),
            &(0..N_BSIZE).map(|b| rows[0].counts.leaf_bsize[b]).collect::<Vec<_>>(),
        );
        let l1_tx = l1_over(
            &(0..N_TX_TYPE * N_TX_SIZE)
                .map(|i| r.counts.fwd_tx[i / N_TX_SIZE][i % N_TX_SIZE])
                .collect::<Vec<_>>(),
            &(0..N_TX_TYPE * N_TX_SIZE)
                .map(|i| rows[0].counts.fwd_tx[i / N_TX_SIZE][i % N_TX_SIZE])
                .collect::<Vec<_>>(),
        );
        println!("{}\t{:.2}\t{:.2}\t{:.2}", r.label, l1_class, l1_bsize, l1_tx);
    }
}

/// L1 distance between two count vectors after normalising each to percent.
fn l1_over(a: &[u64], b: &[u64]) -> f64 {
    let (sa, sb) = (a.iter().sum::<u64>() as f64, b.iter().sum::<u64>() as f64);
    if sa == 0.0 || sb == 0.0 {
        return f64::NAN;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (100.0 * *x as f64 / sa - 100.0 * *y as f64 / sb).abs())
        .sum()
}

fn print_source(r: &Row) {
    let c = &r.counts;
    let (tn, tp) = (c.intra_total_calls(), c.intra_total_px());
    println!("class\tcalls\tpixels\tpct_calls\tpct_px");
    for (i, name) in INTRA_CLASS.iter().enumerate() {
        let n: u64 = c.intra_calls[i].iter().sum();
        let p: u64 = c.intra_px[i].iter().sum();
        if n > 0 {
            println!(
                "{name}\t{n}\t{p}\t{:.2}\t{:.2}",
                100.0 * n as f64 / tn as f64,
                100.0 * p as f64 / tp as f64
            );
        }
    }
    println!();
    println!("class_x_tx\tcalls\tpixels\tpct_calls\tpct_px");
    for (i, name) in INTRA_CLASS.iter().enumerate() {
        for t in 0..N_TX_SIZE {
            let n = c.intra_calls[i][t];
            if n > 0 {
                println!(
                    "{name}:{}\t{n}\t{}\t{:.2}\t{:.2}",
                    TX_SIZE_NAME[t],
                    c.intra_px[i][t],
                    100.0 * n as f64 / tn as f64,
                    100.0 * c.intra_px[i][t] as f64 / tp as f64
                );
            }
        }
    }
    println!();
    println!("nd_mode\tcalls\tpixels");
    for m in 0..N_MODE {
        if c.nd_mode_calls[m] > 0 {
            println!("{}\t{}\t{}", MODE_NAME[m], c.nd_mode_calls[m], c.nd_mode_px[m]);
        }
    }
    println!();
    let fwd_total: u64 = c.fwd_tx.iter().flatten().sum();
    println!("fwd_tx_type\tcount\tpct");
    for ty in 0..N_TX_TYPE {
        let n: u64 = c.fwd_tx[ty].iter().sum();
        if n > 0 {
            println!("{}\t{n}\t{:.2}", TX_TYPE_NAME[ty], 100.0 * n as f64 / fwd_total as f64);
        }
    }
    println!();
    println!("fwd_tx_size\tcount\tpct");
    for t in 0..N_TX_SIZE {
        let n: u64 = (0..N_TX_TYPE).map(|ty| c.fwd_tx[ty][t]).sum();
        if n > 0 {
            println!("{}\t{n}\t{:.2}", TX_SIZE_NAME[t], 100.0 * n as f64 / fwd_total as f64);
        }
    }
    println!();
    let leaves: u64 = c.leaf_bsize.iter().sum();
    println!("leaf_bsize\tcount\tpct");
    for b in 0..N_BSIZE {
        if c.leaf_bsize[b] > 0 {
            println!("{}\t{}\t{:.2}", BSIZE_NAME[b], c.leaf_bsize[b], 100.0 * c.leaf_bsize[b] as f64 / leaves as f64);
        }
    }
    println!();
    println!("leaf_mode\tcount\tpct");
    for m in 0..N_MODE {
        if c.leaf_mode[m] > 0 {
            println!("{}\t{}\t{:.2}", MODE_NAME[m], c.leaf_mode[m], 100.0 * c.leaf_mode[m] as f64 / leaves as f64);
        }
    }
    println!();
    println!("total_intra_calls\t{tn}");
    println!("total_intra_px\t{tp}");
    println!("frame_px\t{}", r.px);
    println!("coded_bytes\t{}", r.coded_bytes);
}

/// Encode `spec` once (after one warm-up encode whose counts are subtracted, so
/// any lazily-built table that only allocates on first use cannot land in the
/// census) and return its counts.
fn census_one(spec: &str) -> Row {
    let (cell, bootstrap) = build_cell(spec);
    census::reset();
    let _ = cell.port_encode(&bootstrap); // warm
    let base = census::snapshot();
    let out = cell.port_encode(&bootstrap);
    let counts = census::snapshot().since(&base);
    assert!(!counts.is_empty(), "{spec}: empty census — is the `census` feature on?");
    Row { label: spec.to_string(), counts, coded_bytes: out.len(), px: cell.w * cell.h }
}

fn build_cell(spec: &str) -> (EncodeCell, Vec<u8>) {
    if let Some(name) = spec.strip_prefix("winperf:") {
        let (w, h, q, s) = winperf::CELL;
        let content = winperf::Content::parse(name);
        return (winperf::cell(w, h, q, s, content), winperf::bootstrap(content));
    }
    if let Some(rest) = spec.strip_prefix("yuv:") {
        #[cfg(not(feature = "c-oracle"))]
        {
            let _ = rest;
            panic!("{spec}: a raw .yuv source needs a bootstrap from real aomenc — rebuild with --features c-oracle,census");
        }
        #[cfg(feature = "c-oracle")]
        {
            let (path, dims) = rest.rsplit_once(':').expect("yuv:<path>:<w>x<h>");
            let (w, h) = dims.split_once('x').expect("yuv:<path>:<w>x<h>");
            let (w, h): (usize, usize) = (w.parse().unwrap(), h.parse().unwrap());
            let (_, _, q, s) = winperf::CELL;
            let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            let (cw, ch) = (w / 2, h / 2);
            assert_eq!(buf.len(), w * h + 2 * cw * ch, "{path}: not {w}x{h} 8-bit I420");
            let up = |s: &[u8]| s.iter().map(|&b| u16::from(b)).collect::<Vec<u16>>();
            let cell = EncodeCell {
                label: spec.to_string(),
                w,
                h,
                mono: false,
                ss_x: 1,
                ss_y: 1,
                usage: 2, // ALLINTRA, the study cell
                cq_level: q,
                speed: s,
                bd: 8,
                y: up(&buf[..w * h]),
                u: up(&buf[w * h..w * h + cw * ch]),
                v: up(&buf[w * h + cw * ch..]),
            };
            let boot = cell.c_encode_defaults();
            assert!(!boot.is_empty(), "{spec}: the C bootstrap encode produced nothing");
            return (cell, boot);
        }
    }
    panic!("unknown source {spec:?}; want winperf:<name> or yuv:<path>:<w>x<h>");
}
