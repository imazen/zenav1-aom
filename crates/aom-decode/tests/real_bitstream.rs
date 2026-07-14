//! THE REAL-BITSTREAM GATE TEST: decode bitstreams produced by the REAL
//! libaom v3.14.1 encoder (`aom_codec_av1_cx`, the same library+path the
//! aomenc CLI drives) with `decode_frame_obus`, and compare every plane
//! BYTE-IDENTICALLY against the REAL C decoder (`aom_codec_av1_dx`).
//!
//! ENVELOPE (every constraint, honestly):
//! - one shown KEY frame per stream (`g_limit=1`, forced KF) — all-intra.
//! - encoder flags: `--cpu-used=0 --end-usage=q --cq-level=<q>
//!   --enable-cdef=0 --enable-restoration=0 --sb-size=64 --deltaq-mode=0
//!   --aq-mode=0 --enable-palette=0 --enable-intrabc=0` (usage GOOD).
//! - loop-filter levels must come out 0 — VERIFIED FROM OUR OWN HEADER PARSE
//!   per stream (the C decoder skips deblocking then; the driver has none).
//!   The cq values below were probed to yield level 0 on this content; a
//!   config that stops satisfying it FAILS (no silent skip) — see the
//!   `high_q_deblocked_stream_is_rejected` companion, which pins the first
//!   out-of-envelope behavior.
//! - single tile, no superres / film grain / screen-content tools / qm /
//!   segmentation / lossless / 128x128 SBs (decode_frame_obus hard-errors).
//! - sizes include non-multiple-of-SB (96x80) and non-multiple-of-8 (100x76,
//!   the decoder's 8px-aligned mi grid cropped back).
//! - 4:2:0 + 4:4:4 + monochrome, bit depths 8 and 10.

use aom_decode::frame::decode_frame_obus;
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Photographic-ish content: smooth gradients + sinusoidal structure + noise
/// (deliberately NOT flat/synthetic-few-colors, which could trip the
/// encoder's screen-content detection — screen-content tools are out of the
/// decode envelope and would hard-error).
fn gen_plane(w: usize, h: usize, bd: i32, seed: u64, chroma: bool) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 0.25 + 0.5 * (0.6 * fx + 0.4 * fy);
            let wave = 0.12 * ((fx * 9.0).sin() * (fy * 7.0).cos());
            let noise = ((rng.next() >> 40) as i64 % 33 - 16) as f64 / maxv as f64;
            let mut v = base + wave + noise * if chroma { 2.0 } else { 4.0 };
            v = v.clamp(0.0, 1.0);
            p[r * w + col] = (v * maxv as f64).round() as u16;
        }
    }
    p
}

struct Cfg {
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss: (i32, i32),
    cq: i32,
}

fn run_config(cfg: &Cfg) -> (usize, bool, i32) {
    let (cw, ch) = if cfg.mono {
        (0, 0)
    } else {
        (
            (cfg.w + cfg.ss.0 as usize) >> cfg.ss.0,
            (cfg.h + cfg.ss.1 as usize) >> cfg.ss.1,
        )
    };
    let seed =
        ((cfg.w as u64) << 32) ^ ((cfg.h as u64) << 16) ^ ((cfg.bd as u64) << 8) ^ cfg.cq as u64;
    let y = gen_plane(cfg.w, cfg.h, cfg.bd, seed ^ 0x1111, false);
    let u = gen_plane(cw, ch, cfg.bd, seed ^ 0x2222, true);
    let v = gen_plane(cw, ch, cfg.bd, seed ^ 0x3333, true);

    // REAL encoder bytes (cpu-used=0).
    let bytes = c::ref_encode_av1_kf(
        &y, &u, &v, cfg.w, cfg.h, cfg.bd, cfg.mono, cfg.ss.0, cfg.ss.1, cfg.cq, 0, false,
    );

    // Rust decode (hard-errors outside the envelope — that FAILS the test).
    let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
        panic!(
            "decode_frame_obus rejected {}x{} bd{} mono={} ss={:?} cq={}: {e}",
            cfg.w, cfg.h, cfg.bd, cfg.mono, cfg.ss, cfg.cq
        )
    });

    // Envelope facts verified from OUR OWN parse.
    assert_eq!(rust.filter_level, [0; 4], "loop-filter must be level 0");
    assert_eq!(rust.base_qindex > 0, true);
    assert_eq!((rust.width, rust.height), (cfg.w, cfg.h));
    assert_eq!(rust.bit_depth, cfg.bd);
    assert_eq!(rust.monochrome, cfg.mono);
    if !cfg.mono {
        assert_eq!(
            (rust.subsampling_x as i32, rust.subsampling_y as i32),
            cfg.ss
        );
    }

    // Gold oracle: the REAL C decoder on the same bytes.
    let cref = c::ref_decode_av1_kf(&bytes, cfg.w, cfg.h);
    assert_eq!(cref.info[0], cfg.bd);
    assert_eq!(cref.info[1] != 0, cfg.mono);

    assert_eq!(
        rust.y, cref.y,
        "LUMA mismatch {}x{} bd{} ss={:?} cq={}",
        cfg.w, cfg.h, cfg.bd, cfg.ss, cfg.cq
    );
    if cfg.mono {
        assert!(rust.u.is_empty() && rust.v.is_empty());
    } else {
        assert_eq!(
            rust.u, cref.u,
            "U mismatch {}x{} bd{} ss={:?} cq={}",
            cfg.w, cfg.h, cfg.bd, cfg.ss, cfg.cq
        );
        assert_eq!(
            rust.v, cref.v,
            "V mismatch {}x{} bd{} ss={:?} cq={}",
            cfg.w, cfg.h, cfg.bd, cfg.ss, cfg.cq
        );
    }
    (bytes.len(), rust.tx_mode_select, rust.base_qindex)
}

#[test]
fn real_bitstreams_decode_byte_identical_to_c() {
    // cq grid PROBED 2026-07-14 (this content, libaom v3.14.1 build,
    // cpu-used=0): the encoder picks loop-filter level 0 for every config
    // below; a level>0 pick FAILS the run (envelope check inside run_config).
    // The bd10 arm stops at cq 6 (cq>=8 starts picking nonzero levels there);
    // the bd8 arm reaches into every coefficient-CDF qindex band:
    // cq 2 -> q 8 (band 0), 6 -> 24 (band 1), 16 -> 64 (band 2),
    // 28 -> 112 (band 2), 36 -> 144 (band 3).
    let sizes = [(64usize, 64usize), (96, 80), (100, 76)];
    let combos = [
        (8i32, (1i32, 1i32), false),
        (8, (0, 0), false),
        (10, (1, 1), false),
        (10, (0, 0), false),
        (8, (1, 1), true), // monochrome
    ];
    let mut n = 0u32;
    let mut select_seen = 0u32;
    let mut bands = [0u32; 4];
    let mut run = |w: usize, h: usize, bd: i32, ss: (i32, i32), mono: bool, cq: i32| {
        let (len, sel, q) = run_config(&Cfg {
            w,
            h,
            bd,
            mono,
            ss,
            cq,
        });
        assert!(len > 50, "suspiciously small stream ({len} bytes)");
        select_seen += sel as u32;
        bands[if q <= 20 {
            0
        } else if q <= 60 {
            1
        } else if q <= 120 {
            2
        } else {
            3
        }] += 1;
        n += 1;
    };
    for &(w, h) in &sizes {
        for &(bd, ss, mono) in &combos {
            for &cq in &[2i32, 6] {
                run(w, h, bd, ss, mono, cq);
            }
            if bd == 8 {
                for &cq in &[16i32, 28] {
                    run(w, h, bd, ss, mono, cq);
                }
            }
        }
    }
    // Band-3 (q>120) arm: every bd8 combo that probes to level 0 at cq 36
    // ((100x76, 4:4:4) does not — it picks [0,2,5,0]).
    for &(w, h, ss, mono) in &[
        (64usize, 64usize, (1i32, 1i32), false),
        (64, 64, (0, 0), false),
        (64, 64, (1, 1), true),
        (96, 80, (1, 1), false),
        (96, 80, (0, 0), false),
        (96, 80, (1, 1), true),
        (100, 76, (1, 1), false),
        (100, 76, (1, 1), true),
    ] {
        run(w, h, 8, ss, mono, 36);
    }
    assert_eq!(
        n,
        30 + 18 + 8,
        "15 combos x cq{{2,6}} + 9 bd8 x cq{{16,28}} + 8 band-3"
    );
    // Speed-0 allintra codes TX_MODE_SELECT — the multi-txb path must be live.
    assert!(select_seen > 0, "no TX_MODE_SELECT stream decoded");
    // All four coefficient-CDF qindex bands exercised on real streams.
    assert!(bands.iter().all(|&b| b > 0), "band coverage {bands:?}");
}

#[test]
fn high_q_deblocked_stream_is_rejected_not_misdecoded() {
    // At aggressive q the encoder picks nonzero deblock levels; the driver
    // has no deblocker, so decode_frame_obus must REFUSE (the honest
    // envelope boundary), never return unfiltered pixels.
    let (w, h, bd) = (64usize, 64usize, 8);
    let y = gen_plane(w, h, bd, 0x9999, false);
    let u = gen_plane(w / 2, h / 2, bd, 0xAAAA, true);
    let v = gen_plane(w / 2, h / 2, bd, 0xBBBB, true);
    let bytes = c::ref_encode_av1_kf(&y, &u, &v, w, h, bd, false, 1, 1, 60, 0, false);
    match decode_frame_obus(&bytes) {
        Err(e) => assert!(
            e.contains("loop-filter"),
            "expected the loop-filter envelope error, got: {e}"
        ),
        Ok(f) => {
            // If the encoder ever picks level 0 even at cq 60 the stream is
            // legitimately in-envelope — then it must match the C decoder.
            assert_eq!(f.filter_level, [0; 4]);
            let cref = c::ref_decode_av1_kf(&bytes, w, h);
            assert_eq!(f.y, cref.y);
        }
    }
}

#[test]
fn cdef_enabled_stream_is_rejected() {
    // A REAL stream encoded with CDEF on: out of envelope by construction —
    // the driver reads the per-SB strength literals but applies no CDEF
    // filtering, so it must refuse rather than return unfiltered pixels.
    let (w, h, bd) = (64usize, 64usize, 8);
    let y = gen_plane(w, h, bd, 0xCCCC, false);
    let u = gen_plane(w / 2, h / 2, bd, 0xDDDD, true);
    let v = gen_plane(w / 2, h / 2, bd, 0xEEEE, true);
    let bytes = c::ref_encode_av1_kf(&y, &u, &v, w, h, bd, false, 1, 1, 8, 0, true);
    let e = decode_frame_obus(&bytes).expect_err("CDEF stream must be rejected");
    assert!(
        e.contains("CDEF"),
        "expected the CDEF envelope error, got: {e}"
    );
}

#[test]
fn garbage_input_errors_cleanly() {
    assert!(decode_frame_obus(&[0u8; 4]).is_err());
    assert!(decode_frame_obus(&[]).is_err());
}
