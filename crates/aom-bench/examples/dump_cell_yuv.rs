//! Dump a photographic ENCODE CELL as an 8-bit I420 `.yuv`, so the xbench
//! cross-encoder drivers (`drv-aom`, `drv-rav1e`) can be pointed at exactly the
//! content this repo's own clause-(4) bands use.
//!
//! WHY THIS EXISTS: the deployment question "can zenavif default to this?" is a
//! comparison against **zenrav1e** (zenavif's `#[default]` backend), not against
//! libaom — and the only cross-encoder table in the tree
//! (`benchmarks/xbench_2026-08-01.md`) predates the whole perf programme, so its
//! zenav1-aom rows are stale by roughly 5x at the shipping preset. Re-running it
//! needs a `.yuv`, and the committed corpus holds only screen content
//! (`codec-corpus/gb82-sc`), which is the ONE class where the port is
//! pathologically slow (KB-41: the IntraBC DV search is ~80 s per 1 MP) and so
//! would answer a different question. This writes the in-repo PHOTOGRAPHIC cell
//! instead: the mirror-tiled `av1-1-b8-01-size-196x196` decode that
//! `eprof_x86` and every clause-(4) band already use.
//!
//! usage: dump_cell_yuv <w> <h> <out.yuv>
use aom_bench::EncodeCell;
use aom_sys_ref as c;

fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mir = |i: usize, n: usize| {
        let m = i % (2 * n);
        if m < n { m } else { 2 * n - 1 - m }
    };
    let (bw, bh) = (base.w, base.h);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = base.y[mir(r, bh) * bw + mir(col, bw)];
        }
    }
    let (bcw, bch) = ((bw + base.ss_x) >> base.ss_x, (bh + base.ss_y) >> base.ss_y);
    let (cw, ch) = ((w + base.ss_x) >> base.ss_x, (h + base.ss_y) >> base.ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = base.u[mir(r, bch) * bcw + mir(col, bcw)];
            v[r * cw + col] = base.v[mir(r, bch) * bcw + mir(col, bcw)];
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono: base.mono,
        ss_x: base.ss_x,
        ss_y: base.ss_y,
        usage: base.usage,
        cq_level: cq,
        speed,
        bd: base.bd,
        y,
        u,
        v,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: dump_cell_yuv <w> <h> <out.yuv>");
        std::process::exit(2);
    }
    let (w, h) = (a[1].parse::<usize>().unwrap(), a[2].parse::<usize>().unwrap());
    c::ref_init();
    let base = EncodeCell::real_content("dump_base_196", "av1-1-b8-01-size-196x196", None, 27, 3);
    assert_eq!(base.bd, 8, "this dumper writes 8-bit I420 only");
    assert_eq!((base.ss_x, base.ss_y), (1, 1), "4:2:0 only");
    let cell = mirror_tile(&base, "dump", w, h, 27, 3);
    let (cw, ch) = ((w + 1) >> 1, (h + 1) >> 1);
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    // The cell holds u16 samples at bd8, so every value is already <= 255; assert
    // rather than truncate, so a bd10 base would fail loudly instead of aliasing.
    for p in [&cell.y[..], &cell.u[..], &cell.v[..]] {
        for &s in p {
            assert!(s <= 255, "sample {s} out of 8-bit range");
            out.push(s as u8);
        }
    }
    assert_eq!(out.len(), w * h + 2 * cw * ch, "I420 size mismatch");
    std::fs::write(&a[3], &out).expect("write .yuv");
    println!("wrote {} ({}x{}, {} bytes)", a[3], w, h, out.len());
}
