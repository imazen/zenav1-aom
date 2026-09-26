//! Forward/inverse transform size census for the eprof cell.
//! Needs `--features census`.
use aom_bench::EncodeCell;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;

fn mirror_tile(
    base: &EncodeCell,
    label: &str,
    w: usize,
    h: usize,
    _cq: i32,
    _speed: i32,
) -> EncodeCell {
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
        label: label.into(),
        w,
        h,
        mono: base.mono,
        ss_x: base.ss_x,
        ss_y: base.ss_y,
        bd: base.bd,
        cq_level: base.cq_level,
        speed: base.speed,
        usage: base.usage,
        y,
        u,
        v,
    }
}

fn main() {
    c::ref_init();
    let base = EncodeCell::real_content(
        "b",
        "av1-1-b8-01-size-196x196",
        Some((196, 196, 0, 0)),
        27,
        3,
    );
    let cell = mirror_tile(&base, "photo_1024", 1024, 1024, 27, 3);
    let mut cfg = KeyFrameConfig::allintra_speed0(
        cell.w,
        cell.h,
        cell.bd,
        cell.mono,
        cell.ss_x,
        cell.ss_y,
        cell.cq_level,
    );
    cfg.cpu_used = cell.speed;
    cfg.enable_cdef = false;
    cfg.enable_restoration = true;
    aom_dsp::census::reset();
    let _ = encode_key_frame(KeyFramePlanes::new(&cell.y, &cell.u, &cell.v), &cfg).unwrap();
    let cs = aom_dsp::census::snapshot();
    let names = [
        "4x4", "8x8", "16x16", "32x32", "64x64", "4x8", "8x4", "8x16", "16x8", "16x32", "32x16",
        "32x64", "64x32", "4x16", "16x4", "8x32", "32x8", "16x64", "64x16",
    ];
    let tot: u64 = cs.fwd_tx.iter().flatten().sum();
    println!("fwd_tx_size (total {tot}):");
    let mut rows: Vec<(usize, u64)> = (0..19)
        .map(|i| (i, cs.fwd_tx.iter().map(|t| t[i]).sum()))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (i, n) in rows {
        if n > 0 {
            println!(
                "  {:>6} {:>9} {:5.1}%",
                names[i],
                n,
                100.0 * n as f64 / tot as f64
            );
        }
    }
    let itot: u64 = cs.inv_tx.iter().flatten().sum();
    println!("inv_tx_size (total {itot}):");
    let mut rows: Vec<(usize, u64)> = (0..19)
        .map(|i| (i, cs.inv_tx.iter().map(|t| t[i]).sum()))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (i, n) in rows {
        if n > 0 {
            println!(
                "  {:>6} {:>9} {:5.1}%",
                names[i],
                n,
                100.0 * n as f64 / itot as f64
            );
        }
    }
}
