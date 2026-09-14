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

use aom_encode::key_frame::{DeltaQMode, KeyFrameConfig, KeyFramePlanes, Tune, encode_key_frame};
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

/// Verbatim copy of `self_contained_key_frame`'s `Content::Texture` +
/// `cell_planes` — the recipe the `PIN_bd*` HBD cells use (gradient + bars +
/// ripple luma, flat mid chroma).
fn tex_planes(w: usize, h: usize, bd: u8, mono: bool, ss_x: usize, ss_y: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let maxv = (1u32 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let grad = 32i64 + ((r + col) * 150 / 256) as i64;
            let bar: i64 = if (col / 16) % 2 == 0 { 0 } else { 45 };
            let ripple: i64 = if (r + col) % 2 == 0 { 14 } else { -14 };
            let v8 = (grad + bar + ripple).clamp(0, 255) as u32;
            y[r * w + col] = ((v8 * maxv) / 255) as u16;
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
    };
    let mid = (maxv / 2 + 1) as u16;
    (y, vec![mid; cw * ch], vec![mid; cw * ch])
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
    // Optional key=value knobs matching self_contained_tools' quality cells:
    //   deltaq=2|3|6 dlf=0|1 strength=N chromadq=0|1 tune=iq|ssim2
    //   dctonly=0|1 bd=8|10|12 tex=0|1 tools0=0|1
    let mut deltaq = 0i32;
    let mut dlf = false;
    let mut strength = 100u32;
    let mut chroma_dq = false;
    let mut dctonly = false;
    let mut bd = 8u8;
    let mut tex = false;
    let mut tools0 = false;
    let mut tune = Tune::default();
    for kv in &a[7.min(a.len())..] {
        let (k, v) = kv.split_once('=').expect("knob args are key=value");
        match k {
            "deltaq" => deltaq = v.parse().unwrap(),
            "dlf" => dlf = v == "1",
            "strength" => strength = v.parse().unwrap(),
            "chromadq" => chroma_dq = v == "1",
            "dctonly" => dctonly = v == "1",
            "bd" => bd = v.parse().unwrap(),
            "tex" => tex = v == "1",
            "tools0" => tools0 = v == "1",
            "tune" => {
                tune = match v {
                    "iq" => Tune::Iq,
                    "ssim2" => Tune::Ssimulacra2,
                    "psnr" => Tune::Psnr,
                    other => panic!("unknown tune {other}"),
                };
            }
            other => panic!("unknown knob {other}"),
        }
    }
    let (ss_x, ss_y) = (1, 1);
    let (y, u, v) = if tex {
        tex_planes(w, w, bd, mono, ss_x, ss_y)
    } else {
        planes(w, w, bd, mono, ss_x, ss_y, 7)
    };

    c::ref_init();
    let mut cfg = KeyFrameConfig::allintra_speed0(w, w, bd, mono, ss_x, ss_y, cq);
    cfg.cpu_used = speed;
    cfg.enable_cdef = cdef_mode != 0;
    cfg.enable_restoration = false;
    if tools0 {
        cfg.enable_palette = false;
        cfg.enable_intrabc = false;
    }
    cfg.quality.tune = tune;
    cfg.quality.cdef_adaptive = cdef_mode == 3;
    cfg.quality.deltaq_mode = match deltaq {
        0 => DeltaQMode::Off,
        2 => DeltaQMode::Perceptual,
        3 => DeltaQMode::PerceptualAi,
        6 => DeltaQMode::VarianceBoost,
        other => panic!("unknown deltaq mode {other}"),
    };
    cfg.quality.deltaq_strength = strength;
    cfg.quality.delta_lf = dlf;
    cfg.quality.chroma_deltaq = chroma_dq;
    cfg.tools.use_intra_dct_only = dctonly;

    let port = encode_key_frame(KeyFramePlanes::new(&y, &u, &v), &cfg)
        .expect("the port must encode this cell");
    let cref = c::ref_encode_av1_kf_cfg(
        &y,
        &u,
        &v,
        w,
        w,
        i32::from(bd),
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
            enable_chroma_deltaq: chroma_dq as i32,
            deltaq_mode: deltaq,
            deltaq_strength: strength as i32,
            enable_deltalf_mode: dlf as i32,
            enable_qm: 0,
            qm_min: -1,
            qm_max: -1,
            superres_denom: 0,
            film_grain_table: None,
            ctrls: if dctonly {
                vec![(c::cx_ctrl::AV1E_SET_INTRA_DCT_ONLY, 1)]
            } else {
                Vec::new()
            },
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
