//! Encode ONE mirror-tiled cell with BOTH `encode_key_frame` and the real-C
//! shim (`ref_encode_av1_kf`), and write each stream to disk — the byte /
//! quality-diff companion to `eprof_x86`, which times but keeps nothing.
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example dump_kf_stream -- \
//!     3840 2160 27 3 /tmp/cell4k_s3
//! # writes /tmp/cell4k_s3.port.obu and /tmp/cell4k_s3.c.obu
//! ```

use aom_bench::EncodeCell;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;

/// Same recipe as `eprof_x86::mirror_tile` / `dump_cell_yuv`.
fn mirror_tile(
    base: &EncodeCell,
    label: &str,
    w: usize,
    h: usize,
    cq: i32,
    speed: i32,
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
    if a.len() < 6 {
        eprintln!("usage: dump_kf_stream <w> <h> <cq> <speed> <out_prefix>");
        std::process::exit(2);
    }
    let (w, h, cq, speed, prefix) = (
        a[1].parse::<usize>().unwrap(),
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<i32>().unwrap(),
        a[4].parse::<i32>().unwrap(),
        a[5].clone(),
    );

    c::ref_init();
    let base = EncodeCell::real_content(
        "eprof_base_196",
        "av1-1-b8-01-size-196x196",
        Some((196, 196, 0, 0)),
        cq,
        speed,
    );
    let cell = if w <= 196 && h <= 196 {
        EncodeCell::real_content(
            &format!("photo_{w}x{h}_cq{cq}_s{speed}"),
            "av1-1-b8-01-size-196x196",
            Some((w, h, 0, 0)),
            cq,
            speed,
        )
    } else {
        mirror_tile(
            &base,
            &format!("photo_{w}x{h}_cq{cq}_s{speed}"),
            w,
            h,
            cq,
            speed,
        )
    };

    // Optional 6th arg: CDEF mode — `1` enables plain CDEF on both arms (the
    // pin cells' `with_postfilter(true, false)` shape — CDEF on, LR off), `3`
    // selects adaptive CDEF (`quality.cdef_adaptive` / AV1E_SET_ENABLE_CDEF=3,
    // routed through the cfg shim).
    let cdef_mode: i32 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(0);
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
    // Optional 7th/8th args (after cdef_mode): tile grid `C,R` and port worker
    // count — the same spellings eprof_x86 uses.
    if let Some(t) = std::env::args().nth(7) {
        let (c, r) = match t.split_once(',') {
            Some((c, r)) => (c.parse::<i32>().unwrap(), r.parse::<i32>().unwrap()),
            None => {
                let n = t.parse::<i32>().unwrap();
                (n, n)
            }
        };
        cfg.tile_columns_log2 = c;
        cfg.tile_rows_log2 = r;
    }
    if let Some(t) = std::env::args().nth(8) {
        cfg.threads = t.parse::<usize>().unwrap();
    }
    cfg.enable_cdef = cdef_mode != 0;
    cfg.enable_restoration = cdef_mode == 0;
    cfg.quality.cdef_adaptive = cdef_mode == 3;

    let port = encode_key_frame(KeyFramePlanes::new(&cell.y, &cell.u, &cell.v), &cfg)
        .expect("the port must encode this cell");
    let cref = if cdef_mode == 3 {
        c::ref_encode_av1_kf_cfg(
            &cell.y,
            &cell.u,
            &cell.v,
            cell.w,
            cell.h,
            i32::from(cell.bd),
            cell.mono,
            cell.ss_x as i32,
            cell.ss_y as i32,
            cell.cq_level,
            cell.speed,
            cell.usage,
            &c::RefKfCfg {
                enable_cdef: cdef_mode,
                enable_restoration: false,
                sb_size_128: cfg.sb_size_128,
                tile_columns_log2: cfg.tile_columns_log2,
                tile_rows_log2: cfg.tile_rows_log2,
                enable_palette: cfg.enable_palette,
                enable_intrabc: cfg.enable_intrabc,
                tuning: -1,
                sharpness: 0,
                enable_adaptive_sharpness: 0,
                dist_metric: -1,
                enable_chroma_deltaq: 0,
                deltaq_mode: -1,
                deltaq_strength: 100,
                enable_deltalf_mode: 0,
                enable_qm: 0,
                qm_min: -1,
                qm_max: -1,
                superres_denom: 0,
                film_grain_table: None,
                ctrls: Vec::new(),
            },
        )
    } else {
        c::ref_encode_av1_kf_screen_content(
            &cell.y,
            &cell.u,
            &cell.v,
            cell.w,
            cell.h,
            i32::from(cell.bd),
            cell.mono,
            cell.ss_x as i32,
            cell.ss_y as i32,
            cell.cq_level,
            cell.speed,
            cdef_mode != 0,
            cdef_mode == 0,
            cell.usage,
            0,
            false,
            cfg.enable_palette,
            cfg.enable_intrabc,
        )
    };

    let (pp, cp) = (format!("{prefix}.port.obu"), format!("{prefix}.c.obu"));
    std::fs::write(&pp, &port).unwrap();
    std::fs::write(&cp, &cref).unwrap();
    let first_diff = port
        .iter()
        .zip(cref.iter())
        .position(|(a, b)| a != b)
        .map(|i| i as isize)
        .unwrap_or(-1);
    println!(
        "{w}x{h} cq{cq} s{speed} port_bytes={} c_bytes={} first_diff={first_diff}",
        port.len(),
        cref.len()
    );
}
