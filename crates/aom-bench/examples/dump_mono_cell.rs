//! Dump a mirror-tiled mono cell through BOTH port paths, against real
//! aomenc. Built for the KB-27 `MONO_S0_OPEN` repro (`av1-1-b8-00-quantizer-00`
//! -> mono -> 64x64, cq24, cpu0 — closed 2026-09-13 by KB-62); the optional
//! knob suffixes generalize it to the KB-63 finding-B cells (e.g.
//! `480 480 63 0 sb128 p140`). Both are closed; this stays as the
//! divergence-localization tool if either shape ever regresses:
//!   * bootstrap: `port_encode_with(c_encode_ctrls(..))` — the s4cov path
//!   * standalone: `encode_key_frame` + `ref_encode_av1_kf_cfg` — full TU dumps
//!     (skipped when the knob set isn't expressible in `KeyFrameConfig`)
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example dump_mono_cell -- /tmp/mono_cq24
//! cargo run --release -p zenav1-aom-bench --example dump_mono_cell -- /tmp/fb 480 480 63 0 sb128 p140
//! ```

use aom_bench::EncodeCell;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;

/// sb128_e2e.rs's `diag256_fmt` content, verbatim (mono arm): the synthetic
/// diagonal ramp `y = 32 + (r+c)*190/(w+h)` that codes a 128-level leaf at
/// high cq — the `mono_cq63` pinned near-tie's source.
fn diag_mono(w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = (32 + (r + col) * 190 / (w + h)) as u16;
        }
    }
    EncodeCell {
        label: format!("diag_mono_{w}x{h}_s{speed}"),
        w,
        h,
        mono: true,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u: Vec::new(),
        v: Vec::new(),
    }
}

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
        eprintln!(
            "usage: dump_mono_cell <out_prefix> [w h cq speed] [k=v ...]\n  \
             knobs: ab0 p140 sb128 diag"
        );
        std::process::exit(2);
    }
    let prefix = a[1].clone();
    let w = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let h = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let cq = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(24i32);
    let speed = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0i32);
    let mut knobs = aom_bench::ToggleKnobs::default();
    let mut sb128 = false;
    let mut diag = false;
    for kv in &a[6.min(a.len())..] {
        match kv.as_str() {
            "ab0" => knobs.enable_ab_partitions = false,
            "p140" => knobs.enable_1to4_partitions = false,
            "sb128" => sb128 = true,
            "diag" => diag = true,
            other => {
                eprintln!("unknown knob {other}");
                std::process::exit(2);
            }
        }
    }

    c::ref_init();
    let cell = if diag {
        diag_mono(w, h, cq, speed)
    } else {
        let mut base =
            EncodeCell::real_content("mono_base", "av1-1-b8-00-quantizer-00", None, cq, 0);
        base.cq_level = cq;
        mirror_tile_mono(&base, w, h, speed)
    };
    assert!(cell.mono && cell.u.is_empty());

    // ---- bootstrap path (the s4cov pin's own route) ----
    let mut ctrls = knobs.c_ctrls();
    if sb128 {
        ctrls.push((
            c::cx_ctrl::AV1E_SET_SUPERBLOCK_SIZE,
            c::cx_ctrl::AOM_SUPERBLOCK_SIZE_128X128,
        ));
    }
    let c_stream = cell.c_encode_ctrls(&ctrls);
    let real_payload = EncodeCell::frame_obu_payload(&c_stream);
    let port_payload = cell.port_encode_with(&c_stream, &knobs);
    println!(
        "bootstrap path: port {} B vs C {} B  first_diff={}",
        port_payload.len(),
        real_payload.len(),
        first_diff(&port_payload, &real_payload)
    );
    std::fs::write(format!("{prefix}.boot.c.obu"), &c_stream).unwrap();
    std::fs::write(format!("{prefix}.boot.port.payload"), &port_payload).unwrap();
    // A decodable TU for the decode-diff localizer: port payload spliced into
    // the C stream's headers.
    let spliced = aom_bench::rd_close::splice_frame_obu(&c_stream, &port_payload);
    std::fs::write(format!("{prefix}.boot.port.tu"), &spliced).unwrap();

    // ---- standalone path — only where the knob set maps onto
    // KeyFrameConfig (default tools; sb128 carries through) ----
    if knobs != aom_bench::ToggleKnobs::default() {
        println!("standalone path: skipped (knobs not expressible in KeyFrameConfig)");
        return;
    }
    let cfg = KeyFrameConfig::allintra_speed0(w, h, cell.bd, true, 1, 1, cq);
    let mut cfg = cfg;
    cfg.cpu_used = speed;
    cfg.sb_size_128 = sb128;
    let port2 = encode_key_frame(KeyFramePlanes::new(&cell.y, &cell.u, &cell.v), &cfg)
        .expect("standalone encode");
    let cref = c::ref_encode_av1_kf_cfg(
        &cell.y,
        &cell.u,
        &cell.v,
        w,
        h,
        i32::from(cell.bd),
        true,
        1,
        1,
        cq,
        speed,
        cell.usage,
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
