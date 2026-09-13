//! Diff two key-frame streams' uncompressed headers field-by-field — the
//! first-cut localizer for "bytes differ" cells: if a HEADER field differs the
//! mechanism is upstream of the payload; if headers are identical the
//! divergence is pure tile data and the decode-both localizer takes over.
//!
//! ```text
//! diff_stream_headers <a.obu> <b.obu>
//! ```

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: diff_stream_headers <a.obu> <b.obu>");
        std::process::exit(2);
    }
    let fa = std::fs::read(&a[1]).unwrap();
    let fb = std::fs::read(&a[2]).unwrap();
    let ha = aom_bench::stream_frame_header(&fa);
    let hb = aom_bench::stream_frame_header(&fb);
    let ta = format!("{ha:#?}");
    let tb = format!("{hb:#?}");
    let mut any = false;
    let va: Vec<&str> = ta.lines().collect();
    let vb: Vec<&str> = tb.lines().collect();
    for (i, (la, lb)) in va.iter().zip(vb.iter()).enumerate() {
        if la != lb {
            for c in va.iter().take(i).skip(i.saturating_sub(6)) {
                println!("    ctx: {c}");
            }
            println!("  a: {la}\n  b: {lb}");
            any = true;
        }
    }
    if !any {
        println!("headers IDENTICAL — divergence is in tile payload");
    }
    // Also: first payload byte after the shared header is more informative than
    // the OBU size field — report the first content difference too.
    match fa.iter().zip(fb.iter()).position(|(x, y)| x != y) {
        Some(i) => println!("first byte diff at offset {i} (a={} b={})", fa[i], fb[i]),
        None if fa.len() != fb.len() => println!("prefix-identical; len {} vs {}", fa.len(), fb.len()),
        None => println!("byte-identical"),
    }
}
