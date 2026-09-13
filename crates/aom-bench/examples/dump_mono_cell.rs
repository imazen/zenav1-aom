//! Dump the KB-27 `MONO_S0_OPEN` repro cell (`av1-1-b8-00-quantizer-00` ->
//! mono -> 64x64, cq24, cpu0) through BOTH port paths, against real aomenc.
//! The pin closed 2026-09-13 (KB-62 — the AB-reuse clone's stale
//! `tx_type_map`); this stays as the divergence-localization tool if the
//! shape ever regresses:
//!   * bootstrap: `port_encode_with(c_encode_ctrls(&[]))` — the s4cov path
//!   * standalone: `encode_key_frame` + `ref_encode_av1_kf_cfg` — full TU dumps
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example dump_mono_cell -- /tmp/mono_cq24
//! ```

use aom_bench::EncodeCell;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;

/// s4cov_partial_sb_axis.rs's mirror_tile + to_mono, verbatim.
fn mirror_tile_mono(base: &EncodeCell, w: usize, h: usize, speed: i32) -> EncodeCell {
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
    EncodeCell {
        label: format!("mono_{w}x{h}_s{speed}"),
        w,
        h,
        mono: true,
        ss_x: 1,
        ss_y: 1,
        usage: base.usage,
        cq_level: base.cq_level,
        speed,
        bd: base.bd,
        y,
        u: Vec::new(),
        v: Vec::new(),
    }
}

fn first_diff(a: &[u8], b: &[u8]) -> isize {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .map(|i| i as isize)
        .unwrap_or(-1)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: dump_mono_cell <out_prefix> [w h cq speed]");
        std::process::exit(2);
    }
    let prefix = a[1].clone();
    let w = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let h = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let cq = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(24i32);
    let speed = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0i32);

    c::ref_init();
    let mut base = EncodeCell::real_content("mono_base", "av1-1-b8-00-quantizer-00", None, cq, 0);
    base.cq_level = cq;
    let cell = mirror_tile_mono(&base, w, h, speed);
    assert!(cell.mono && cell.u.is_empty());

    // ---- bootstrap path (the s4cov pin's own route) ----
    let c_stream = cell.c_encode_ctrls(&[]);
    let real_payload = EncodeCell::frame_obu_payload(&c_stream);
    let port_payload = cell.port_encode_with(&c_stream, &aom_bench::ToggleKnobs::default());
    println!(
        "bootstrap path: port {} B vs C {} B  first_diff={}",
        port_payload.len(),
        real_payload.len(),
        first_diff(&port_payload, &real_payload)
    );
    std::fs::write(format!("{prefix}.boot.c.obu"), &c_stream).unwrap();
    std::fs::write(format!("{prefix}.boot.port.payload"), &port_payload).unwrap();

    // ---- standalone path ----
    let cfg = KeyFrameConfig::allintra_speed0(w, h, cell.bd, true, 1, 1, cq);
    let mut cfg = cfg;
    cfg.cpu_used = speed;
    let port2 = encode_key_frame(KeyFramePlanes { y: &cell.y, u: &cell.u, v: &cell.v }, &cfg)
        .expect("standalone encode");
    let cref = c::ref_encode_av1_kf_cfg(
        &cell.y, &cell.u, &cell.v, w, h, i32::from(cell.bd), true, 1, 1, cq, speed, cell.usage,
        &c::RefKfCfg {
            enable_cdef: 0,
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
    );
    println!(
        "standalone path: port {} B vs C {} B  first_diff={}",
        port2.len(),
        cref.len(),
        first_diff(&port2, &cref)
    );
    std::fs::write(format!("{prefix}.kf.port.obu"), &port2).unwrap();
    std::fs::write(format!("{prefix}.kf.c.obu"), &cref).unwrap();
}
