//! Encode ONE `self_contained_tools`-shaped cell (the same deterministic
//! textured `planes()` generator and `ref_encode_av1_kf_cfg` oracle) with both
//! arms and write each stream to disk — the byte-diff companion for the
//! quality-knob pin cells, which `dump_kf_stream` cannot reproduce (it feeds
//! the mirror-tiled photo through the screen-content shim).
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example dump_tools_cell -- \
//!     128 8 20 /tmp/cdefa 3
//! # w/h=128 bd8 cq20 mono=0 ss=420 speed=0 cdef=3 -> /tmp/cdefa.{port,c}.obu
//! ```

use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;

/// Verbatim copy of `self_contained_tools::planes` — the byte gate depends on
/// feeding both arms the identical synthetic image.
fn planes(w: usize, h: usize, bd: u8, mono: bool, ss_x: usize, ss_y: usize, seed: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let maxv = (1u32 << bd) - 1;
    let sample = |r: usize, col: usize, phase: u32| -> u16 {
        let mut x = (r as u32).wrapping_mul(0x9E37_79B9) ^ (col as u32).wrapping_mul(0x85EB_CA6B) ^ seed ^ phase;
        x ^= x >> 15;
        x = x.wrapping_mul(0x2C1B_3C6D);
        x ^= x >> 12;
        let region = ((r / 16) + (col / 16) + phase as usize) % 3;
        let amp = [6u32, 24, 72][region];
        let grad = (64 + (r * 96) / h.max(1) + (col * 64) / w.max(1)) as u32;
        let tex = (x % (2 * amp + 1)) as i64 - amp as i64;
        let v8 = (grad as i64 + tex).clamp(0, 255) as u32;
        ((v8 * maxv) / 255) as u16
    };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = sample(r, col, 0);
        }
    }
    if mono {
        return (y, Vec::new(), Vec::new());
    }
    let (cw, ch) = ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = sample(r, col, 1);
            v[r * cw + col] = sample(r, col, 2);
        }
    }
    (y, u, v)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: dump_tools_cell <w> <cq> <speed> <out_prefix> [cdef_mode] [mono]");
        std::process::exit(2);
    }
    let (w, cq, speed, prefix) = (
        a[1].parse::<usize>().unwrap(),
        a[2].parse::<i32>().unwrap(),
        a[3].parse::<i32>().unwrap(),
        a[4].clone(),
    );
    let cdef_mode: i32 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(0);
    let mono = a.get(6).map(|s| s == "1").unwrap_or(false);
    let (ss_x, ss_y) = (1, 1);
    let (y, u, v) = planes(w, w, 8, mono, ss_x, ss_y, 7);

    c::ref_init();
    let mut cfg = KeyFrameConfig::allintra_speed0(w, w, 8, mono, ss_x, ss_y, cq);
    cfg.cpu_used = speed;
    cfg.enable_cdef = cdef_mode != 0;
    cfg.enable_restoration = false;
    cfg.quality.cdef_adaptive = cdef_mode == 3;

    let port = encode_key_frame(KeyFramePlanes { y: &y, u: &u, v: &v }, &cfg)
        .expect("the port must encode this cell");
    let cref = c::ref_encode_av1_kf_cfg(
        &y,
        &u,
        &v,
        w,
        w,
        8,
        mono,
        ss_x as i32,
        ss_y as i32,
        cq,
        speed,
        cfg.usage,
        &c::RefKfCfg {
            enable_cdef: cdef_mode,
            enable_restoration: false,
            sb_size_128: cfg.sb_size_128,
            tile_columns_log2: cfg.tile_columns_log2,
            tile_rows_log2: cfg.tile_rows_log2,
            enable_palette: cfg.enable_palette,
            enable_intrabc: cfg.enable_intrabc,
            tuning: cfg.quality.tune.aomenc_value(),
            sharpness: 0,
            enable_adaptive_sharpness: 0,
            dist_metric: 0,
            enable_chroma_deltaq: 0,
            deltaq_mode: 0,
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

    std::fs::write(format!("{prefix}.port.obu"), &port).unwrap();
    std::fs::write(format!("{prefix}.c.obu"), &cref).unwrap();
    let first_diff = port
        .iter()
        .zip(cref.iter())
        .position(|(a, b)| a != b)
        .map(|i| i as isize)
        .unwrap_or(-1);
    println!(
        "{w}x{w} cq{cq} s{speed} cdef{cdef_mode} mono={} port_bytes={} c_bytes={} first_diff={first_diff}",
        mono as u8,
        port.len(),
        cref.len()
    );
}
