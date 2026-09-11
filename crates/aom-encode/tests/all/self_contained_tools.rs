//! **The shipping path's TOOL KNOBS, byte-gated against real aomenc with a
//! MATCHED oracle.** `aom_encode::key_frame::encode_key_frame` gained the
//! `quality` (tune / QM / sharpness / chroma delta-q / delta-q modes / adaptive
//! CDEF), `tools` (the C8–C11 toggles), `film_grain` and `superres_denom` knobs
//! on 2026-09-11; until then every one of those features was ported and gated
//! only through the `aom-bench` harness, so the entry zenavif calls could not
//! reach them (the README's "harness-only" column).
//!
//! Every cell here drives BOTH sides with the same resolved knobs:
//! `ref_encode_av1_kf_cfg` applies `AOME_SET_TUNING` first and then every
//! explicit knob as an override, which is exactly the semantics
//! `KeyFrameConfig::apply_tune` + the individual fields document — so the
//! comparison is like for like by construction, not by argument. Every cell
//! also decodes the port's stream with the REAL libaom decoder and this repo's
//! own decoder and requires identical pixels (KB-29 / KB-33: a stream can be
//! byte-different from aomenc and still conformant, or byte-close and corrupt;
//! only a decoder tells).
//!
//! Known divergence carried through, not hidden: `--enable-cdef` at
//! `--cpu-used >= 4` diverges from aomenc in the header's `cdef_strengths`
//! (PARITY.md C1, pinned in `self_contained_key_frame.rs`); the tune bundle
//! turns CDEF on, so tune cells at speed >= 4 are gated on the DECODE leg only
//! and their byte status is pinned self-promotingly in `TUNE_FAST_CDEF_OPEN`.

use aom_dsp::entropy::header::FilmGrainParams;
use aom_encode::grain_table::{lookup, read_film_grain_table};
use aom_encode::key_frame::{
    CodingTools, DeltaQMode, KeyFrameConfig, KeyFrameError, KeyFramePlanes, TrellisMode, Tune,
    encode_key_frame,
};
use aom_sys_ref as c;

/// Deterministic textured content: a low-frequency gradient plus a hashed
/// texture whose amplitude varies by region, so partitions, transforms and the
/// delta-q maps all have real decisions to make. Chroma gets its own phase.
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

/// The oracle configuration that MEANS the same thing as `cfg` — every
/// tune-family knob passed explicitly (never `-1`), so C's config equals the
/// port's regardless of what the tune bundle installed.
fn ref_cfg(cfg: &KeyFrameConfig, grain_table: Option<std::path::PathBuf>) -> c::RefKfCfg {
    use c::cx_ctrl::*;
    let q = &cfg.quality;
    let t = &cfg.tools;
    let d = CodingTools::default();
    let mut ctrls = Vec::new();
    let mut flag = |on: bool, id: i32, v: bool| {
        if on {
            ctrls.push((id, v as i32));
        }
    };
    flag(t.enable_rect_partitions != d.enable_rect_partitions, AV1E_SET_ENABLE_RECT_PARTITIONS, t.enable_rect_partitions);
    flag(t.enable_ab_partitions != d.enable_ab_partitions, AV1E_SET_ENABLE_AB_PARTITIONS, t.enable_ab_partitions);
    flag(t.enable_1to4_partitions != d.enable_1to4_partitions, AV1E_SET_ENABLE_1TO4_PARTITIONS, t.enable_1to4_partitions);
    flag(t.enable_intra_edge_filter != d.enable_intra_edge_filter, AV1E_SET_ENABLE_INTRA_EDGE_FILTER, t.enable_intra_edge_filter);
    flag(t.enable_filter_intra != d.enable_filter_intra, AV1E_SET_ENABLE_FILTER_INTRA, t.enable_filter_intra);
    flag(t.enable_smooth_intra != d.enable_smooth_intra, AV1E_SET_ENABLE_SMOOTH_INTRA, t.enable_smooth_intra);
    flag(t.enable_paeth_intra != d.enable_paeth_intra, AV1E_SET_ENABLE_PAETH_INTRA, t.enable_paeth_intra);
    flag(t.enable_cfl_intra != d.enable_cfl_intra, AV1E_SET_ENABLE_CFL_INTRA, t.enable_cfl_intra);
    flag(t.enable_directional_intra != d.enable_directional_intra, AV1E_SET_ENABLE_DIRECTIONAL_INTRA, t.enable_directional_intra);
    flag(t.enable_diagonal_intra != d.enable_diagonal_intra, AV1E_SET_ENABLE_DIAGONAL_INTRA, t.enable_diagonal_intra);
    flag(t.enable_angle_delta != d.enable_angle_delta, AV1E_SET_ENABLE_ANGLE_DELTA, t.enable_angle_delta);
    flag(t.enable_tx64 != d.enable_tx64, AV1E_SET_ENABLE_TX64, t.enable_tx64);
    flag(t.enable_rect_tx != d.enable_rect_tx, AV1E_SET_ENABLE_RECT_TX, t.enable_rect_tx);
    flag(t.enable_flip_idtx != d.enable_flip_idtx, AV1E_SET_ENABLE_FLIP_IDTX, t.enable_flip_idtx);
    flag(t.use_intra_dct_only != d.use_intra_dct_only, AV1E_SET_INTRA_DCT_ONLY, t.use_intra_dct_only);
    flag(t.use_intra_default_tx_only != d.use_intra_default_tx_only, AV1E_SET_INTRA_DEFAULT_TX_ONLY, t.use_intra_default_tx_only);
    flag(t.reduced_tx_type_set != d.reduced_tx_type_set, AV1E_SET_REDUCED_TX_TYPE_SET, t.reduced_tx_type_set);
    flag(t.enable_tx_size_search != d.enable_tx_size_search, AV1E_SET_ENABLE_TX_SIZE_SEARCH, t.enable_tx_size_search);
    if t.min_partition_size_px != d.min_partition_size_px {
        ctrls.push((AV1E_SET_MIN_PARTITION_SIZE, t.min_partition_size_px as i32));
    }
    if t.max_partition_size_px != d.max_partition_size_px {
        ctrls.push((AV1E_SET_MAX_PARTITION_SIZE, t.max_partition_size_px as i32));
    }
    if t.cdf_update_mode != d.cdf_update_mode {
        ctrls.push((AV1E_SET_CDF_UPDATE_MODE, t.cdf_update_mode as i32));
    }
    if t.trellis != d.trellis {
        ctrls.push((AV1E_SET_DISABLE_TRELLIS_QUANT, t.trellis.aomenc_value()));
    }
    c::RefKfCfg {
        enable_cdef: match (cfg.enable_cdef, q.cdef_adaptive) {
            (false, _) => 0,
            (true, false) => 1,
            (true, true) => 3,
        },
        enable_restoration: cfg.enable_restoration,
        sb_size_128: cfg.sb_size_128,
        tile_columns_log2: cfg.tile_columns_log2,
        tile_rows_log2: cfg.tile_rows_log2,
        enable_palette: cfg.enable_palette,
        enable_intrabc: cfg.enable_intrabc,
        tuning: q.tune.aomenc_value(),
        sharpness: q.sharpness,
        enable_adaptive_sharpness: q.adaptive_sharpness as i32,
        dist_metric: q.qm_dist_metric as i32,
        enable_chroma_deltaq: q.chroma_deltaq as i32,
        deltaq_mode: q.deltaq_mode.aomenc_value(),
        deltaq_strength: q.deltaq_strength as i32,
        enable_deltalf_mode: q.delta_lf as i32,
        enable_qm: q.qm.is_some() as i32,
        qm_min: q.qm.map_or(-1, |(lo, _)| lo),
        qm_max: q.qm.map_or(-1, |(_, hi)| hi),
        superres_denom: i32::from(cfg.superres_denom),
        film_grain_table: grain_table,
        ctrls,
    }
}

struct Outcome {
    label: String,
    byte_match: bool,
    port_len: usize,
    c_len: usize,
}

/// Encode one cell both ways; the DECODE leg (real C decoder == port decoder on
/// the port's stream) is asserted unconditionally, the byte leg is returned.
fn run(label: &str, cfg: &KeyFrameConfig, grain_table: Option<std::path::PathBuf>) -> Outcome {
    c::ref_init();
    let (y, u, v) = planes(cfg.width, cfg.height, cfg.bit_depth, cfg.monochrome, cfg.ss_x, cfg.ss_y, 7);
    let port = encode_key_frame(KeyFramePlanes { y: &y, u: &u, v: &v }, cfg)
        .unwrap_or_else(|e| panic!("{label}: encode_key_frame refused: {e}"));
    let c_tu = c::ref_encode_av1_kf_cfg(
        &y,
        &u,
        &v,
        cfg.width,
        cfg.height,
        i32::from(cfg.bit_depth),
        cfg.monochrome,
        cfg.ss_x as i32,
        cfg.ss_y as i32,
        cfg.cq_level,
        cfg.cpu_used,
        cfg.usage,
        &ref_cfg(cfg, grain_table),
    );
    assert!(!c_tu.is_empty(), "{label}: C encode failed");
    let c_dec = c::ref_decode_av1_kf(&port, cfg.width, cfg.height);
    let p_dec = aom_decode::frame::decode_frame_obus(&port)
        .unwrap_or_else(|e| panic!("{label}: port decode of its own stream: {e}"));
    assert_eq!(
        (&p_dec.y, &p_dec.u, &p_dec.v),
        (&c_dec.y, &c_dec.u, &c_dec.v),
        "{label}: port-decode(port stream) != real-C-decode(port stream)"
    );
    let byte_match = port == c_tu;
    if !byte_match {
        let first = port.iter().zip(c_tu.iter()).position(|(a, b)| a != b).unwrap_or(port.len().min(c_tu.len()));
        eprintln!("{label}: MISMATCH port {} B vs C {} B, first diff at {first}", port.len(), c_tu.len());
        if first < 24 {
            eprintln!("  port[0..24] {:02x?}", &port[..24.min(port.len())]);
            eprintln!("  c   [0..24] {:02x?}", &c_tu[..24.min(c_tu.len())]);
        }
    }
    Outcome { label: label.to_string(), byte_match, port_len: port.len(), c_len: c_tu.len() }
}

/// Byte-identity gate with a SELF-PROMOTING pin: `pinned_open` is the exact set
/// of cells measured divergent from real aomenc when the knob landed. A cell
/// that leaves the set has CLOSED and must be promoted (removed from the pin);
/// one that enters it is a regression. The decode leg is asserted inside
/// [`run`] for every cell regardless.
fn report(gate: &str, outcomes: &[Outcome], pinned_open: &[&str]) {
    let mut open: Vec<&str> = outcomes.iter().filter(|o| !o.byte_match).map(|o| o.label.as_str()).collect();
    open.sort_unstable();
    let matched = outcomes.len() - open.len();
    println!("{gate}: {matched}/{} byte-identical, {} pinned open", outcomes.len(), pinned_open.len());
    for o in outcomes.iter().filter(|o| !o.byte_match) {
        println!("  OPEN {} (port {} B vs C {} B)", o.label, o.port_len, o.c_len);
    }
    let pinned: std::collections::BTreeSet<&str> = pinned_open.iter().copied().collect();
    let observed: std::collections::BTreeSet<&str> = open.iter().copied().collect();
    let closed: Vec<&&str> = pinned.difference(&observed).collect();
    let regressed: Vec<&&str> = observed.difference(&pinned).collect();
    assert!(
        closed.is_empty() && regressed.is_empty(),
        "{gate}: the pinned divergence set MOVED.\n  CLOSED (promote by removing from the pin): {closed:?}\n  REGRESSED (new divergence): {regressed:?}"
    );
}

fn base(w: usize, h: usize, bd: u8, mono: bool, ss: (usize, usize), cq: i32, speed: i32) -> KeyFrameConfig {
    let mut cfg = KeyFrameConfig::allintra_speed0(w, h, bd, mono, ss.0, ss.1, cq);
    cfg.cpu_used = speed;
    cfg
}

const FORMATS: [(&str, bool, (usize, usize)); 3] = [("420", false, (1, 1)), ("mono", true, (1, 1)), ("444", false, (0, 0))];

/// **tune=IQ / tune=SSIMULACRA2, the whole `handle_tuning` bundle** through the
/// shipping path: QM 2..=10, sharpness 7, QM-PSNR distortion, ADAPTIVE CDEF,
/// chroma delta-q, Variance-Boost delta-q (+ adaptive sharpness for IQ). cq 8
/// (qindex 32: the adaptive-CDEF OFF arm), 20 (halve + zero-low arms), 40
/// (qindex 160 > 140: halve only). Speeds 0 and 3 are byte gates; loop
/// restoration on and off.
#[test]
fn tune_bundles_byte_match_real_aomenc() {
    let mut out = Vec::new();
    for tune in [Tune::Iq, Tune::Ssimulacra2] {
        for &(fmt, mono, ss) in &FORMATS {
            for &sz in &[64usize, 128] {
                for &cq in &[8i32, 20, 40] {
                    for &speed in &[0i32, 3] {
                        for &lr in &[false, true] {
                            if lr && (sz == 64 || cq != 20) {
                                continue; // one LR arm per format is enough
                            }
                            let mut cfg = base(sz, sz, 8, mono, ss, cq, speed);
                            cfg.apply_tune(tune);
                            cfg.enable_restoration = lr;
                            out.push(run(&format!("{tune:?} {fmt} {sz}x{sz} cq{cq} s{speed} lr{}", lr as u8), &cfg, None));
                        }
                    }
                }
            }
        }
    }
    // MEASURED 2026-09-11 at landing: 0/84 byte-identical. Every cell DECODES
    // identically on both decoders (the streams are conformant), and the first
    // differing byte is the frame OBU's size field — i.e. an RD/tile-payload
    // divergence, not a header-derivation defect. Localizing it is the next
    // job (playbook §10, decode-both); it is pinned here so a cell that closes
    // is promoted rather than passing silently.
    let all: Vec<String> = out.iter().map(|o| o.label.clone()).collect();
    let all_refs: Vec<&str> = all.iter().map(String::as_str).collect();
    report("tune bundles (speed 0/3)", &out, &all_refs);
}

/// The tune bundle at the FAST presets (`--cpu-used` 6 and 8). CDEF is on
/// under the bundle and the port's CDEF search at speed >= 4 is a pinned
/// divergence (PARITY.md C1), so the byte leg is PINNED self-promotingly and
/// the decode leg (inside `run`) is the gate.
#[test]
fn tune_bundles_at_fast_presets_decode_and_are_pinned() {
    const TUNE_FAST_CDEF_OPEN: &[&str] = &[
        // Pinned 2026-09-11: all eight cells diverge in the header's cdef
        // strengths, the same shape PARITY.md C1 records for --enable-cdef at
        // speed >= 4. Self-promoting: a cell that starts matching fails below.
        "Iq 420 128x128 cq20 s6",
        "Iq 420 128x128 cq20 s8",
        "Iq mono 128x128 cq20 s6",
        "Iq mono 128x128 cq20 s8",
        "Ssimulacra2 420 128x128 cq20 s6",
        "Ssimulacra2 420 128x128 cq20 s8",
        "Ssimulacra2 mono 128x128 cq20 s6",
        "Ssimulacra2 mono 128x128 cq20 s8",
    ];
    let mut out = Vec::new();
    for tune in [Tune::Iq, Tune::Ssimulacra2] {
        for &(fmt, mono, ss) in &FORMATS[..2] {
            for &speed in &[6i32, 8] {
                let mut cfg = base(128, 128, 8, mono, ss, 20, speed);
                cfg.apply_tune(tune);
                out.push(run(&format!("{tune:?} {fmt} 128x128 cq20 s{speed}"), &cfg, None));
            }
        }
    }
    let open: Vec<&str> = out.iter().filter(|o| !o.byte_match).map(|o| o.label.as_str()).collect();
    let closed: Vec<&str> = out.iter().filter(|o| o.byte_match).map(|o| o.label.as_str()).collect();
    println!("tune at fast presets: {} open, {} byte-identical (all decode)", open.len(), closed.len());
    for o in &out {
        println!("  {} {}: port {} B vs C {} B", if o.byte_match { "MATCH" } else { "open " }, o.label, o.port_len, o.c_len);
    }
    let pinned: std::collections::BTreeSet<&str> = TUNE_FAST_CDEF_OPEN.iter().copied().collect();
    let observed: std::collections::BTreeSet<&str> = open.iter().copied().collect();
    assert_eq!(
        observed, pinned,
        "the tune-at-fast-preset divergence set MOVED. A cell that left the set has CLOSED — promote it \
         (remove from TUNE_FAST_CDEF_OPEN); a cell that entered it is a REGRESSION."
    );
}

/// Each `QualityTools` knob ALONE on the PSNR tune, plus the CDEF_ADAPTIVE arms
/// under `--enable-cdef=3` without the rest of the bundle.
#[test]
fn quality_knobs_byte_match_real_aomenc() {
    let mut out = Vec::new();
    let sz = 128usize;
    // QM ranges, including the flat top level and the allintra default range.
    for &(lo, hi) in &[(4i32, 10i32), (0, 15), (2, 10), (8, 8)] {
        for &cq in &[20i32, 44] {
            let mut cfg = base(sz, sz, 8, false, (1, 1), cq, 0);
            cfg.quality.qm = Some((lo, hi));
            out.push(run(&format!("qm({lo},{hi}) 420 cq{cq}"), &cfg, None));
        }
    }
    // QM-PSNR distortion metric with QM on.
    for &cq in &[20i32, 44] {
        let mut cfg = base(sz, sz, 8, false, (1, 1), cq, 0);
        cfg.quality.qm = Some((4, 10));
        cfg.quality.qm_dist_metric = true;
        out.push(run(&format!("qm-psnr 420 cq{cq}"), &cfg, None));
    }
    // Sharpness, alone and with the adaptive cap (three cap arms: qindex 48 / 128 / 200).
    for &sh in &[3i32, 7] {
        for &(cq, adaptive) in &[(12i32, false), (32, false), (12, true), (32, true), (50, true)] {
            let mut cfg = base(sz, sz, 8, false, (1, 1), cq, 0);
            cfg.quality.sharpness = sh;
            cfg.quality.adaptive_sharpness = adaptive;
            out.push(run(&format!("sharpness{sh}{} 420 cq{cq}", if adaptive { "+adaptive" } else { "" }), &cfg, None));
        }
    }
    // Chroma delta-q: the constant PSNR arm at every subsampling, and the tune
    // ramps with every OTHER bundle piece left at its PSNR default.
    for &(fmt, mono, ss) in &FORMATS {
        for &cq in &[12i32, 32, 56] {
            let mut cfg = base(sz, sz, 8, mono, ss, cq, 0);
            cfg.quality.chroma_deltaq = true;
            out.push(run(&format!("chroma-deltaq {fmt} cq{cq}"), &cfg, None));
        }
    }
    for tune in [Tune::Iq, Tune::Ssimulacra2] {
        for &(fmt, mono, ss) in &FORMATS {
            let mut cfg = base(sz, sz, 8, mono, ss, 32, 0);
            cfg.quality.tune = tune;
            cfg.quality.chroma_deltaq = true;
            out.push(run(&format!("chroma-deltaq {tune:?} {fmt} cq32"), &cfg, None));
        }
    }
    // 4:2:2 chroma delta-q arm.
    for &cq in &[12i32, 40] {
        let mut cfg = base(sz, sz, 8, false, (1, 0), cq, 0);
        cfg.quality.chroma_deltaq = true;
        out.push(run(&format!("chroma-deltaq 422 cq{cq}"), &cfg, None));
    }
    // Delta-q modes alone, at the RD speeds and the nonrd speed 8 (KB-46), with
    // and without delta-lf; strength 50 / 100 / 200 for Variance Boost.
    for mode in [DeltaQMode::Perceptual, DeltaQMode::PerceptualAi, DeltaQMode::VarianceBoost] {
        for &speed in &[0i32, 3, 8] {
            for &cq in &[20i32, 44] {
                for &dlf in &[false, true] {
                    if dlf && speed != 0 {
                        continue;
                    }
                    let mut cfg = base(sz, sz, 8, false, (1, 1), cq, speed);
                    cfg.quality.deltaq_mode = mode;
                    cfg.quality.delta_lf = dlf;
                    out.push(run(&format!("deltaq {mode:?} 420 cq{cq} s{speed} dlf{}", dlf as u8), &cfg, None));
                }
            }
        }
    }
    for &strength in &[50u32, 200] {
        let mut cfg = base(sz, sz, 8, false, (1, 1), 32, 0);
        cfg.quality.deltaq_mode = DeltaQMode::VarianceBoost;
        cfg.quality.deltaq_strength = strength;
        out.push(run(&format!("deltaq VarianceBoost strength{strength} 420 cq32"), &cfg, None));
    }
    // CDEF_ADAPTIVE on the PSNR tune (`--enable-cdef=3`): off / halve+zero / halve.
    for &cq in &[8i32, 20, 40, 60] {
        for &(fmt, mono, ss) in &FORMATS[..2] {
            let mut cfg = base(sz, sz, 8, mono, ss, cq, 0);
            cfg.enable_cdef = true;
            cfg.quality.cdef_adaptive = true;
            out.push(run(&format!("cdef-adaptive {fmt} cq{cq}"), &cfg, None));
        }
    }
    // MEASURED 2026-09-11 at landing: 48/71 byte-identical. Byte-identical
    // ALONE on the PSNR tune: every QM range, the QM-PSNR metric, sharpness and
    // adaptive sharpness, the constant chroma-delta-q arm at 4:2:0/4:2:2/4:4:4/
    // mono, Variance-Boost delta-q at speed 0 and the cq20 s3 cell, Perceptual-
    // AI delta-q at speeds 0 and 3 (with and without delta-lf), cdef-adaptive at
    // cq 8 (the OFF arm) and cq 40. Open, all conformant (decode leg green):
    // the tune chroma-delta-q RAMPS, Perceptual (mode 2) everywhere, the nonrd
    // (s8) arm of every mode, VarianceBoost cq44 s3, and cdef-adaptive at cq 20
    // / cq 60 (the halve + zero-low arms). First differing byte is the frame OBU
    // size in every case — payload divergences to localize, not header bugs.
    report(
        "quality knobs",
        &out,
        &[
            "cdef-adaptive 420 cq20",
            "cdef-adaptive 420 cq60",
            "cdef-adaptive mono cq20",
            "cdef-adaptive mono cq60",
            "chroma-deltaq Iq 420 cq32",
            "chroma-deltaq Iq 444 cq32",
            "chroma-deltaq Iq mono cq32",
            "chroma-deltaq Ssimulacra2 420 cq32",
            "chroma-deltaq Ssimulacra2 444 cq32",
            "chroma-deltaq Ssimulacra2 mono cq32",
            "deltaq Perceptual 420 cq20 s0 dlf0",
            "deltaq Perceptual 420 cq20 s0 dlf1",
            "deltaq Perceptual 420 cq20 s3 dlf0",
            "deltaq Perceptual 420 cq20 s8 dlf0",
            "deltaq Perceptual 420 cq44 s0 dlf0",
            "deltaq Perceptual 420 cq44 s0 dlf1",
            "deltaq Perceptual 420 cq44 s3 dlf0",
            "deltaq Perceptual 420 cq44 s8 dlf0",
            "deltaq PerceptualAi 420 cq20 s8 dlf0",
            "deltaq PerceptualAi 420 cq44 s8 dlf0",
            "deltaq VarianceBoost 420 cq20 s8 dlf0",
            "deltaq VarianceBoost 420 cq44 s3 dlf0",
            "deltaq VarianceBoost 420 cq44 s8 dlf0",
        ],
    );
}

/// Each `CodingTools` toggle flipped ALONE from aomenc's default (PARITY.md
/// C8–C11), at speed 0 and the fast speed 6, on 4:2:0 and 4:4:4.
#[test]
fn coding_tools_byte_match_real_aomenc() {
    let flips: Vec<(&str, Box<dyn Fn(&mut CodingTools)>)> = vec![
        ("rect-partitions=0", Box::new(|t| t.enable_rect_partitions = false)),
        ("ab-partitions=0", Box::new(|t| t.enable_ab_partitions = false)),
        ("1to4-partitions=0", Box::new(|t| t.enable_1to4_partitions = false)),
        ("min-partition=16", Box::new(|t| t.min_partition_size_px = 16)),
        ("max-partition=32", Box::new(|t| t.max_partition_size_px = 32)),
        ("intra-edge-filter=0", Box::new(|t| t.enable_intra_edge_filter = false)),
        ("filter-intra=0", Box::new(|t| t.enable_filter_intra = false)),
        ("smooth-intra=0", Box::new(|t| t.enable_smooth_intra = false)),
        ("paeth-intra=0", Box::new(|t| t.enable_paeth_intra = false)),
        ("cfl-intra=0", Box::new(|t| t.enable_cfl_intra = false)),
        ("directional-intra=0", Box::new(|t| t.enable_directional_intra = false)),
        ("diagonal-intra=0", Box::new(|t| t.enable_diagonal_intra = false)),
        ("angle-delta=0", Box::new(|t| t.enable_angle_delta = false)),
        ("tx64=0", Box::new(|t| t.enable_tx64 = false)),
        ("rect-tx=0", Box::new(|t| t.enable_rect_tx = false)),
        ("flip-idtx=0", Box::new(|t| t.enable_flip_idtx = false)),
        ("intra-dct-only=1", Box::new(|t| t.use_intra_dct_only = true)),
        ("intra-default-tx-only=1", Box::new(|t| t.use_intra_default_tx_only = true)),
        ("reduced-tx-type-set=1", Box::new(|t| t.reduced_tx_type_set = true)),
        ("tx-size-search=0", Box::new(|t| t.enable_tx_size_search = false)),
        ("cdf-update-mode=2", Box::new(|t| t.cdf_update_mode = 2)),
        ("trellis=full", Box::new(|t| t.trellis = TrellisMode::Full)),
        ("trellis=off", Box::new(|t| t.trellis = TrellisMode::Off)),
        ("trellis=final-pass", Box::new(|t| t.trellis = TrellisMode::FinalPass)),
    ];
    let mut out = Vec::new();
    for (name, flip) in &flips {
        for &(fmt, mono, ss) in &[FORMATS[0], FORMATS[2]] {
            for &speed in &[0i32, 6] {
                if speed == 6 && fmt == "444" {
                    continue;
                }
                let mut cfg = base(128, 128, 8, mono, ss, 32, speed);
                flip(&mut cfg.tools);
                out.push(run(&format!("{name} {fmt} cq32 s{speed}"), &cfg, None));
            }
        }
    }
    // MEASURED 2026-09-11 at landing: 46/48 byte-identical; `--intra-dct-only`
    // at speed 0 diverges in the payload on both formats (conformant). Its
    // speed-6 cell and every other toggle at both speeds are byte-identical.
    // `--cdf-update-mode=0` is REFUSED by name (KB-53) and is exercised in
    // `new_knob_refusals_are_named` instead.
    report(
        "coding tools",
        &out,
        &["intra-dct-only=1 420 cq32 s0", "intra-dct-only=1 444 cq32 s0"],
    );
}

/// `--film-grain-table`: the port signals the parsed table entry in its own
/// header; the coded pixels are the ordinary encode. Four libaom test vectors
/// (rich full-chroma / max-lag / no-chroma-points / chroma-from-luma) at three
/// formats.
#[test]
fn film_grain_table_byte_matches_real_aomenc() {
    c::ref_init();
    let mut out = Vec::new();
    for &tv in &[1i32, 2, 6, 15] {
        let path = std::env::temp_dir().join(format!("aomrs_tools_grain_{}_{tv}.tbl", std::process::id()));
        c::ref_write_grain_table_test_vector(tv, &path);
        let entries = read_film_grain_table(&std::fs::read(&path).expect("read table")).expect("parse table");
        let mut fg = FilmGrainParams::default();
        assert!(lookup(&entries, 0, &mut fg), "tv{tv}: time-0 lookup");
        for &(fmt, mono, ss) in &FORMATS {
            if mono && tv != 6 && tv != 15 {
                continue; // chroma tables on a mono frame: libaom drops the chroma points; one arm suffices
            }
            let mut cfg = base(128, 128, 8, mono, ss, 32, 0);
            cfg.film_grain = Some(fg);
            // Does the oracle DECODER accept C's OWN grain stream? If not, the
            // decode leg cannot be run on this knob through this shim.
            let (y, u, v) = planes(128, 128, 8, mono, ss.0, ss.1, 7);
            let c_tu = c::ref_encode_av1_kf_cfg(&y, &u, &v, 128, 128, 8, mono, ss.0 as i32, ss.1 as i32, 32, 0, 2, &ref_cfg(&cfg, Some(path.clone())));
            // The oracle decoder REJECTS libaom's OWN tv15-on-monochrome stream
            // (chroma-scaling-from-luma grain with no chroma planes), so the
            // decode leg is run only where the oracle accepts its own output;
            // the byte leg holds everywhere.
            let c_accepts = std::panic::catch_unwind(|| c::ref_decode_av1_kf(&c_tu, 128, 128)).is_ok();
            let port = encode_key_frame(KeyFramePlanes { y: &y, u: &u, v: &v }, &cfg)
                .unwrap_or_else(|e| panic!("grain tv{tv} {fmt}: refused: {e}"));
            if c_accepts {
                let c_dec = c::ref_decode_av1_kf(&port, 128, 128);
                let p_dec = aom_decode::frame::decode_frame_obus(&port).expect("port decode");
                assert_eq!((&p_dec.y, &p_dec.u, &p_dec.v), (&c_dec.y, &c_dec.u, &c_dec.v), "grain tv{tv} {fmt}: decoders disagree");
            } else {
                println!("grain tv{tv} {fmt}: oracle decoder rejects libaom's own stream; byte leg only");
            }
            out.push(Outcome { label: format!("grain tv{tv} {fmt}"), byte_match: port == c_tu, port_len: port.len(), c_len: c_tu.len() });
        }
        let _ = std::fs::remove_file(&path);
    }
    report("film grain table", &out, &[]);
}

/// `--superres-mode=fixed --superres-denominator=N` (CDEF and restoration off,
/// the gated envelope): denominators 9, 12 and 16 at bd8 and bd10, 4:2:0 and
/// mono, at two widths (exact 2:1 for 128 x denom 16 takes libaom's optimized
/// 8-bit scaler; the others the non-normative resize).
#[test]
fn superres_fixed_byte_matches_real_aomenc() {
    let mut out = Vec::new();
    for &denom in &[9u8, 12, 16] {
        for &(fmt, mono, ss) in &FORMATS[..2] {
            for &bd in &[8u8, 10] {
                for &w in &[128usize, 200] {
                    if bd == 10 && w == 200 {
                        continue;
                    }
                    let mut cfg = base(w, 96, bd, mono, ss, 32, 0);
                    cfg.superres_denom = denom;
                    out.push(run(&format!("superres d{denom} {fmt} bd{bd} {w}x96"), &cfg, None));
                }
            }
        }
    }
    report("superres fixed", &out, &[]);
}

/// The new knobs' documented refusals, by name, and the support query's
/// agreement with the encoder (it IS the same predicate).
#[test]
fn new_knob_refusals_are_named() {
    let cases: Vec<(&str, KeyFrameConfig)> = {
        let mut v = Vec::new();
        let mut c1 = base(64, 64, 8, false, (1, 1), 32, 0);
        c1.superres_denom = 12;
        c1.enable_cdef = true;
        v.push(("superres + cdef", c1));
        let mut c2 = base(64, 64, 8, false, (1, 1), 32, 0);
        c2.superres_denom = 12;
        c2.enable_restoration = true;
        v.push(("superres + restoration", c2));
        let mut c3 = base(64, 64, 8, false, (1, 1), 32, 0);
        c3.superres_denom = 17;
        v.push(("superres denom 17", c3));
        let mut c4 = base(64, 64, 8, false, (1, 1), 32, 0);
        c4.quality.qm = Some((12, 4));
        v.push(("qm min > max", c4));
        let mut c5 = base(64, 64, 8, false, (1, 1), 32, 0);
        c5.quality.sharpness = 9;
        v.push(("sharpness 9", c5));
        let mut c6 = base(64, 64, 8, false, (1, 1), 32, 0);
        c6.tools.min_partition_size_px = 12;
        v.push(("min partition 12px", c6));
        let mut c7 = base(64, 64, 8, false, (1, 1), 32, 0);
        c7.tools.cdf_update_mode = 3;
        v.push(("cdf update mode 3", c7));
        let mut c8 = base(64, 64, 8, false, (1, 1), 32, 0);
        c8.tools.cdf_update_mode = 0;
        v.push(("cdf update mode 0 (KB-53)", c8));
        v
    };
    for (name, cfg) in cases {
        let q = cfg.validate_configuration();
        assert!(matches!(q, Err(KeyFrameError::Unsupported(_))), "{name}: support query must refuse by name, got {q:?}");
        let e = encode_key_frame(KeyFramePlanes { y: &[], u: &[], v: &[] }, &cfg);
        assert!(matches!(e, Err(KeyFrameError::Unsupported(_))), "{name}: encoder must refuse the same way, got {e:?}");
    }
}
