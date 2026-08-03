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
//! * `winperf:detail` / `winperf:smooth` / `winperf:photo` / `winperf:screen` —
//!   the committed generator plus its committed bootstrap fixture. Needs **no**
//!   C oracle, so this runs anywhere `winperf` itself runs.
//! * `yuv:<path>:<w>x<h>` — a raw 8-bit I420 file, bootstrapped by a real
//!   `aomenc` encode. Needs `--features c-oracle,census`. This is how the study
//!   photograph (the REFERENCE distribution) is censused.
//! * `scr:<path>:<w>x<h>` — the same, bootstrapped by `c_encode_screen` so the
//!   frame header can carry `allow_screen_content_tools` (aomenc's own
//!   ANTIALIASING_AWARE detection decides; the census REPORTS whether it fired
//!   rather than assuming it did). Without that header bit the palette and
//!   intraBC searches are unreachable no matter what the knobs say.
//!
//! Options:
//!
//! * `--speed N` — `--cpu-used`. The default is [`winperf::CELL`]'s 6, and
//!   several families are speed-gated rather than content-gated: at speed 6
//!   `prune_filter_intra_level == 2`, i.e. `rd_pick_filter_intra_sby` is not
//!   called at all, so filter-intra is a structural zero on EVERY source.
//! * `--cq N` — `--cq-level`, default [`winperf::CELL`]'s 44.
//! * `--knobs palette,intrabc` — turn the port's screen-content RD searches on.
//!   Both are default-OFF knobs (`--enable-palette` / `--enable-intrabc`), so a
//!   census taken without them is a census of the DEFAULT encoder and a census
//!   taken with them is a census of a knob-gated one. The two are reported
//!   separately and never merged.
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

use aom_bench::{EncodeCell, ToggleKnobs, winperf};
use aom_dsp::census::{
    self, BSIZE_NAME, Counts, FILTER_INTRA_MODE_NAME, INTRA_CLASS, MODE_NAME, N_ANGLE_DELTA,
    N_BSIZE, N_FILTER_INTRA_MODE, N_MODE, N_PALETTE_SIZE, N_PLANE, N_TX_SIZE, N_TX_TYPE,
    N_UV_MODE, PLANE_NAME, TX_SIZE_NAME, TX_TYPE_NAME, UV_MODE_NAME,
};

/// How the census drives the encoder, so a row can say which encoder it is a
/// census OF. Palette and intraBC are default-off knobs and a family reachable
/// only under one of them is a different statement from a family reachable by
/// default (playbook §8: derive coverage from artefacts, and name the artefact).
#[derive(Clone, Copy)]
struct Opts {
    speed: i32,
    cq: i32,
    palette: bool,
    intrabc: bool,
}

impl Opts {
    fn knobs(&self) -> ToggleKnobs {
        ToggleKnobs {
            enable_palette: self.palette,
            enable_intrabc: self.intrabc,
            ..Default::default()
        }
    }

    fn tag(&self) -> String {
        let mut t = format!("s{}q{}", self.speed, self.cq);
        if self.palette {
            t.push_str("+pal");
        }
        if self.intrabc {
            t.push_str("+ibc");
        }
        t
    }
}

/// A censused source: its label, its counts, and the bytes it coded.
struct Row {
    label: String,
    counts: Box<Counts>,
    coded_bytes: usize,
    /// `None` for the `winperf` sources (whose dimensions are [`winperf::CELL`]).
    px: usize,
    /// Whether the bootstrap's frame header signalled
    /// `allow_screen_content_tools`. Palette and intraBC are unreachable
    /// without it regardless of the knobs, so a census that does not report it
    /// cannot tell "the content has no palette in it" from "the tool was never
    /// legal here".
    screen_tools: bool,
}

fn main() {
    assert!(
        census::enabled(),
        "built without --features census: every counter is a no-op and this \
         tool would print a table of zeros"
    );
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (w, h, cq, speed) = winperf::CELL;
    let _ = (w, h);
    let mut opts = Opts { speed, cq, palette: false, intrabc: false };
    let mut args: Vec<String> = Vec::new();
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--speed" => opts.speed = it.next().expect("--speed N").parse().expect("int"),
            "--cq" => opts.cq = it.next().expect("--cq N").parse().expect("int"),
            "--knobs" => {
                for k in it.next().expect("--knobs a,b").split(',') {
                    match k {
                        "palette" => opts.palette = true,
                        "intrabc" => opts.intrabc = true,
                        "" => {}
                        other => panic!("unknown knob {other:?}; want palette / intrabc"),
                    }
                }
            }
            other if other.starts_with("--") => panic!("unknown option {other:?}"),
            other => args.push(other.to_string()),
        }
    }
    assert!(!args.is_empty(), "usage: content_census [--speed N] [--cq N] [--knobs palette,intrabc] <source> [<source> ...]");
    println!("# config\t{}", opts.tag());

    let rows: Vec<Row> = args.iter().map(|s| census_one(s, &opts)).collect();

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

    // ---- the FAMILY table: does this source reach the tool at all? --------
    //
    // One row per coding-tool family, as a share of the natural denominator
    // for that family (leaves, coded luma pixels, chroma-reference leaves, or
    // forward transforms). This is the table a lever author looks their family
    // up in before quoting a band — a 0.00 here means the code under test does
    // not run on this source and any null measured against it is structural.
    println!();
    println!("########## FAMILY REACH (percent; 0.00 = the tool never fires)");
    print!("family\tdenominator");
    for r in &rows {
        print!("\t{}", r.label);
    }
    println!();
    let fam: [(&str, &str, fn(&Counts) -> (u64, u64)); 16] = [
        ("filter_intra", "leaves", |c| (c.filter_intra_leaves(), c.leaves())),
        ("filter_intra_px", "coded_px", |c| (c.leaf_filter_intra_px, c.leaf_total_px())),
        ("palette_y", "leaves", |c| (c.palette_y_leaves(), c.leaves())),
        ("palette_y_px", "coded_px", |c| (c.leaf_palette_y_px, c.leaf_total_px())),
        ("palette_uv", "leaves", |c| (c.palette_uv_leaves(), c.leaves())),
        ("intrabc", "leaves", |c| (c.leaf_intrabc, c.leaves())),
        ("intrabc_px", "coded_px", |c| (c.leaf_intrabc_px, c.leaf_total_px())),
        ("cfl", "chroma_ref_leaves", |c| (c.cfl_leaves(), c.leaf_chroma_ref)),
        ("cfl_pred_px", "pred_px", |c| (c.cfl_px, c.intra_total_px() + c.cfl_px)),
        ("chroma_pred_calls", "pred_calls", |c| {
            (c.plane_calls[1] + c.plane_calls[2], c.plane_total())
        }),
        ("directional_px", "pred_px", |c| (c.directional_px(), c.intra_total_px())),
        ("nonzero_angle_delta", "leaves", |c| (c.nonzero_angle_delta_leaves(), c.leaves())),
        ("rect_leaves", "leaves", |c| (c.rect_leaves(), c.leaves())),
        ("leaves_le_8px", "leaves", |c| (c.small_leaves(), c.leaves())),
        ("fwd_tx_4pt", "fwd_tx", |c| (c.small_fwd_tx(), c.fwd_tx.iter().flatten().sum())),
        ("fwd_tx_non_dct", "fwd_tx", |c| {
            (c.non_dct_fwd_tx(), c.fwd_tx.iter().flatten().sum())
        }),
    ];
    for (name, den, f) in fam {
        print!("{name}\t{den}");
        for r in &rows {
            let (n, d) = f(&r.counts);
            print!("\t{:.2}", if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 });
        }
        println!();
    }
    // Absolute counts for the families whose share rounds to 0.00 — "6 calls in
    // a 1 MP frame" and "zero" are different findings and must not print alike.
    print!("filter_intra_leaves_abs\t-");
    for r in &rows {
        print!("\t{}", r.counts.filter_intra_leaves());
    }
    println!();
    print!("palette_y_leaves_abs\t-");
    for r in &rows {
        print!("\t{}", r.counts.palette_y_leaves());
    }
    println!();
    print!("intrabc_leaves_abs\t-");
    for r in &rows {
        print!("\t{}", r.counts.leaf_intrabc);
    }
    println!();
    print!("cfl_leaves_abs\t-");
    for r in &rows {
        print!("\t{}", r.counts.cfl_leaves());
    }
    println!();
    print!("allow_screen_content_tools\t-");
    for r in &rows {
        print!("\t{}", u8::from(r.screen_tools));
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
    println!("leaf_uv_mode\tcount\tpct");
    for m in 0..N_UV_MODE {
        if c.leaf_uv_mode[m] > 0 {
            println!(
                "{}\t{}\t{:.2}",
                UV_MODE_NAME[m],
                c.leaf_uv_mode[m],
                100.0 * c.leaf_uv_mode[m] as f64 / c.leaf_chroma_ref.max(1) as f64
            );
        }
    }
    println!();
    println!("leaf_tx_size\tcount\tpct");
    for t in 0..N_TX_SIZE {
        if c.leaf_tx_size[t] > 0 {
            println!(
                "{}\t{}\t{:.2}",
                TX_SIZE_NAME[t],
                c.leaf_tx_size[t],
                100.0 * c.leaf_tx_size[t] as f64 / leaves.max(1) as f64
            );
        }
    }
    println!();
    println!("filter_intra_mode\tcount");
    for m in 0..N_FILTER_INTRA_MODE {
        if c.leaf_filter_intra[m] > 0 {
            println!("{}\t{}", FILTER_INTRA_MODE_NAME[m], c.leaf_filter_intra[m]);
        }
    }
    println!();
    println!("palette_size\ty_leaves\tuv_leaves");
    for s in 0..N_PALETTE_SIZE {
        if c.leaf_palette_y[s] > 0 || c.leaf_palette_uv[s] > 0 {
            println!("{s}\t{}\t{}", c.leaf_palette_y[s], c.leaf_palette_uv[s]);
        }
    }
    println!();
    println!("angle_delta\ty_leaves\tuv_leaves");
    for d in 0..N_ANGLE_DELTA {
        if c.leaf_angle_delta_y[d] > 0 || c.leaf_angle_delta_uv[d] > 0 {
            println!("{}\t{}\t{}", d as i32 - 3, c.leaf_angle_delta_y[d], c.leaf_angle_delta_uv[d]);
        }
    }
    println!();
    println!("plane\tpred_calls\tpred_px");
    for p in 0..N_PLANE {
        println!("{}\t{}\t{}", PLANE_NAME[p], c.plane_calls[p], c.plane_px[p]);
    }
    println!();
    println!("cfl_calls\t{}", c.cfl_calls.iter().sum::<u64>());
    println!("cfl_px\t{}", c.cfl_px);
    println!("leaf_chroma_ref\t{}", c.leaf_chroma_ref);
    println!("leaf_intrabc\t{}", c.leaf_intrabc);
    println!("leaf_skip_txfm\t{}", c.leaf_skip_txfm);
    println!("leaf_inter\t{}", c.leaf_inter);
    println!("allow_screen_content_tools\t{}", u8::from(r.screen_tools));
    println!("total_intra_calls\t{tn}");
    println!("total_intra_px\t{tp}");
    println!("frame_px\t{}", r.px);
    println!("coded_bytes\t{}", r.coded_bytes);
}

/// Encode `spec` once (after one warm-up encode whose counts are subtracted, so
/// any lazily-built table that only allocates on first use cannot land in the
/// census) and return its counts.
fn census_one(spec: &str, opts: &Opts) -> Row {
    // Progress to stderr as each source lands. A palette+intraBC census of a
    // 1 MP screen source takes minutes, and a tool that prints nothing until
    // every row is computed loses the whole run to one bad path argument — as
    // one did on 2026-08-03, 35 minutes in.
    let t0 = std::time::Instant::now();
    eprintln!("[census] {spec} ...");
    let (cell, bootstrap) = build_cell(spec, opts);
    let knobs = opts.knobs();
    let screen_tools = bootstrap_allows_screen_tools(&bootstrap);
    census::reset();
    let _ = cell.port_encode_with(&bootstrap, &knobs); // warm
    let base = census::snapshot();
    let out = cell.port_encode_with(&bootstrap, &knobs);
    let counts = census::snapshot().since(&base);
    assert!(!counts.is_empty(), "{spec}: empty census — is the `census` feature on?");
    // Non-vacuity for the plane split (playbook §2): the per-plane annotation
    // is a hand-placed hook next to each `predict_intra_high` call, and a call
    // site nobody annotated would silently deflate the chroma share. The DSP
    // hook inside `predict_intra_high` cannot be missed, so the two totals
    // agreeing is what proves the annotation is complete.
    assert_eq!(
        counts.plane_total(),
        counts.intra_total_calls(),
        "{spec}: {} intra predictions but only {} carry a plane tag — an \
         encoder call site is missing `census::note_plane_intra_pred`",
        counts.intra_total_calls(),
        counts.plane_total()
    );
    eprintln!(
        "[census] {spec}: {:.1}s leaves {} palette_y {} intrabc {} cfl {} scdet {} coded {}",
        t0.elapsed().as_secs_f64(),
        counts.leaves(),
        counts.palette_y_leaves(),
        counts.leaf_intrabc,
        counts.cfl_leaves(),
        u8::from(screen_tools),
        out.len(),
    );
    Row {
        label: spec.to_string(),
        counts,
        coded_bytes: out.len(),
        px: cell.w * cell.h,
        screen_tools,
    }
}

/// `allow_screen_content_tools`, read out of the bootstrap's own headers. The
/// port's palette / intraBC searches are gated on this bit exactly as C's are,
/// so a census that reports "palette: 0.00 %" without it cannot say whether the
/// content had no palette in it or the tool was never legal.
fn bootstrap_allows_screen_tools(bootstrap: &[u8]) -> bool {
    use aom_dsp::entropy::header::{
        CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
        RestorationHeader, read_sequence_header_obu, read_uncompressed_header,
    };
    use aom_dsp::entropy::rb::ReadBitBuffer;
    const OBU_SEQUENCE_HEADER: u32 = 1;
    const OBU_FRAME: u32 = 6;
    let mut pos = 0usize;
    let (mut seqp, mut framep): (Option<&[u8]>, Option<&[u8]>) = (None, None);
    while pos < bootstrap.len() {
        let hdr = aom_dsp::entropy::obu::read_obu_header(&bootstrap[pos..]).expect("obu header");
        let after = pos + hdr.header_len;
        let (size, nb) =
            aom_dsp::entropy::leb128::uleb_decode(&bootstrap[after..]).expect("leb128");
        let (start, end) = (after + nb, after + nb + size as usize);
        match hdr.obu_type {
            t if t == OBU_SEQUENCE_HEADER => seqp = Some(&bootstrap[start..end]),
            t if t == OBU_FRAME => framep = Some(&bootstrap[start..end]),
            _ => {}
        }
        pos = end;
    }
    let seq = read_sequence_header_obu(&mut ReadBitBuffer::new(seqp.expect("seq OBU")));
    let s = &seq.seq_header;
    let cc = &seq.color_config;
    let cfg = FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length
                as u32,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            frame_id_length: s.frame_id_length as u32,
            force_screen_content_tools: s.force_screen_content_tools,
            force_integer_mv: s.force_integer_mv,
            max_frame_width: s.max_frame_width,
            max_frame_height: s.max_frame_height,
            enable_order_hint: s.enable_order_hint,
            order_hint_bits_minus_1: s.order_hint_bits_minus_1,
            operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
            operating_point_idc: seq.operating_point_idc,
            op_decoder_model_param_present: seq.op_decoder_model_param_present,
            buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length as u32,
            temporal_layer_id: 0,
            spatial_layer_id: 0,
            ..Default::default()
        },
        frame_size: FrameSizeHeader {
            num_bits_width: s.num_bits_width,
            num_bits_height: s.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            ..Default::default()
        },
        num_planes: if cc.monochrome { 1 } else { 3 },
        separate_uv_delta_q: cc.separate_uv_delta_q,
        loopfilter: LoopfilterHeader::default(),
        cdef: CdefHeader { enable_cdef: s.enable_cdef, ..Default::default() },
        restoration: RestorationHeader {
            enable_restoration: s.enable_restoration,
            sb_size_128: s.sb_size_128,
            subsampling_x: cc.subsampling_x,
            subsampling_y: cc.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    let p = read_uncompressed_header(&mut ReadBitBuffer::new(framep.expect("frame OBU")), &cfg);
    p.allow_screen_content_tools
}

fn build_cell(spec: &str, opts: &Opts) -> (EncodeCell, Vec<u8>) {
    if let Some(name) = spec.strip_prefix("winperf:") {
        let (w, h, _, _) = winperf::CELL;
        let content = winperf::Content::parse(name);
        let cell = winperf::cell(w, h, opts.cq, opts.speed, content);
        // The committed bootstrap fixture is per (content, CELL): its frame
        // header carries THIS cell's base_qindex. Re-censusing at another q or
        // speed would read a header that does not describe the encode, so the
        // fixture path is pinned to CELL and any other (q, speed) needs the
        // oracle to bootstrap afresh.
        let (_, _, cq0, s0) = winperf::CELL;
        if (opts.cq, opts.speed) == (cq0, s0) {
            return (cell, winperf::bootstrap(content));
        }
        #[cfg(not(feature = "c-oracle"))]
        panic!(
            "{spec}: --speed/--cq off winperf::CELL needs a fresh bootstrap — \
             rebuild with --features c-oracle,census"
        );
        #[cfg(feature = "c-oracle")]
        {
            let boot = cell.c_encode_defaults();
            assert!(!boot.is_empty(), "{spec}: the C bootstrap encode produced nothing");
            return (cell, boot);
        }
    }
    if let Some(rest) = spec.strip_prefix("real:") {
        // The DIFFERENTIAL corpus (`EncodeCell::real_content`): a conformance
        // vector decoded back to pixels. This is what KB-13's byte-parity map
        // is measured on, so "which families does the byte gate exercise?" is
        // the same question as "what does this census say?".
        #[cfg(not(feature = "c-oracle"))]
        {
            let _ = rest;
            panic!("{spec}: a conformance vector needs the C decoder — rebuild with --features c-oracle,census");
        }
        #[cfg(feature = "c-oracle")]
        {
            let (vector, crop) = match rest.split_once(':') {
                None => (rest, None),
                Some((v, c)) => {
                    let (wh, off) = c.split_once('+').unwrap_or((c, "0+0"));
                    let (w, h) = wh.split_once('x').expect("real:<vector>:<w>x<h>[+x+y]");
                    let (ox, oy) = off.split_once('+').unwrap_or(("0", "0"));
                    (
                        v,
                        Some((
                            w.parse().unwrap(),
                            h.parse().unwrap(),
                            ox.parse().unwrap(),
                            oy.parse().unwrap(),
                        )),
                    )
                }
            };
            let cell = EncodeCell::real_content(spec, vector, crop, opts.cq, opts.speed);
            let boot = cell.c_encode_defaults();
            assert!(!boot.is_empty(), "{spec}: the C bootstrap encode produced nothing");
            return (cell, boot);
        }
    }
    let screen = spec.starts_with("scr:");
    if let Some(rest) = spec.strip_prefix("yuv:").or_else(|| spec.strip_prefix("scr:")) {
        #[cfg(not(feature = "c-oracle"))]
        {
            let _ = (rest, screen);
            panic!("{spec}: a raw .yuv source needs a bootstrap from real aomenc — rebuild with --features c-oracle,census");
        }
        #[cfg(feature = "c-oracle")]
        {
            let (path, dims) = rest.rsplit_once(':').expect("<path>:<w>x<h>");
            let (w, h) = dims.split_once('x').expect("<path>:<w>x<h>");
            let (w, h): (usize, usize) = (w.parse().unwrap(), h.parse().unwrap());
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
                cq_level: opts.cq,
                speed: opts.speed,
                bd: 8,
                y: up(&buf[..w * h]),
                u: up(&buf[w * h..w * h + cw * ch]),
                v: up(&buf[w * h + cw * ch..]),
            };
            // `c_encode_screen` passes `--enable-palette` / `--enable-intrabc`
            // to the REFERENCE encode; aomenc's own screen-content detection
            // then decides whether to signal `allow_screen_content_tools`. We
            // do not force the bit — forcing it would make every source look
            // palette-capable and the census would stop measuring content.
            let boot = if screen {
                cell.c_encode_screen(opts.palette, opts.intrabc)
            } else {
                cell.c_encode_defaults()
            };
            assert!(!boot.is_empty(), "{spec}: the C bootstrap encode produced nothing");
            return (cell, boot);
        }
    }
    panic!("unknown source {spec:?}; want winperf:<name>, yuv:<path>:<w>x<h> or scr:<path>:<w>x<h>");
}
