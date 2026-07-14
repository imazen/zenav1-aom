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
//! - DEBLOCKING IS IN THE ENVELOPE: whatever loop-filter levels /
//!   sharpness / mode-ref deltas the encoder picks are applied
//!   (aom-loopfilter::frame, C-diffed in lf_apply_diff.rs). The low-cq arms
//!   still come out level 0 (deblock is a gated no-op there — same gate as
//!   the C decoder); the high-cq arms pick NONZERO levels and pin the real
//!   deblock application byte-for-byte. The run asserts both populations
//!   are present (`nonzero_lf` / luma+chroma coverage guards).
//! - single tile, no superres / film grain / screen-content tools / qm /
//!   segmentation / lossless / 128x128 SBs (decode_frame_obus hard-errors).
//! - sizes include non-multiple-of-SB (96x80) and non-multiple-of-8 (100x76,
//!   the decoder's 8px-aligned mi grid cropped back).
//! - 4:2:0 + 4:4:4 + monochrome, bit depths 8 and 10. (4:2:2 with nonzero
//!   CHROMA deblock levels is the one rejected combination — libaom's
//!   4:2:2 chroma path reads max_txsize_rect_lookup[BLOCK_INVALID] out of
//!   bounds; see `deblocked_422_chroma_is_rejected`.)

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

fn run_config(cfg: &Cfg) -> (usize, bool, i32, [i32; 4]) {
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
    assert!(rust.base_qindex > 0);
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
    (
        bytes.len(),
        rust.tx_mode_select,
        rust.base_qindex,
        rust.filter_level,
    )
}

#[test]
fn real_bitstreams_decode_byte_identical_to_c() {
    // cq grid PROBED 2026-07-14 (this content, libaom v3.14.1 build,
    // cpu-used=0). Low-cq arms come out loop-filter level 0 (bd10 picks
    // nonzero from cq>=8; bd8 from around cq>=40 on this content); the added
    // high-cq arms (bd8 cq 44/52, bd10 cq 16/36, plus the previously-excluded
    // (100x76,444) cq 36 which picks [0,2,5,0]) pick NONZERO levels — the
    // deblock application is pinned byte-identically on those. bd8 bands:
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
    let mut zero_lf = 0u32;
    let mut nonzero_luma_lf = 0u32;
    let mut nonzero_chroma_lf = 0u32;
    let mut run = |w: usize, h: usize, bd: i32, ss: (i32, i32), mono: bool, cq: i32| {
        let (len, sel, q, lf) = run_config(&Cfg {
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
        if lf == [0; 4] {
            zero_lf += 1;
        }
        if lf[0] != 0 || lf[1] != 0 {
            nonzero_luma_lf += 1;
        }
        if lf[2] != 0 || lf[3] != 0 {
            nonzero_chroma_lf += 1;
        }
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
                // Deblocked arms: aggressive q picks nonzero filter levels.
                for &cq in &[44i32, 52] {
                    run(w, h, bd, ss, mono, cq);
                }
            } else {
                // bd10 picks nonzero levels from cq>=8 — previously excluded,
                // now the deblocked bd10 arm.
                for &cq in &[16i32, 36] {
                    run(w, h, bd, ss, mono, cq);
                }
            }
        }
    }
    // Band-3 (q>120) arm at cq 36, incl. the (100x76, 4:4:4) combo that
    // picks [0,2,5,0] (was excluded when deblocking was out of envelope).
    for &(w, h, ss, mono) in &[
        (64usize, 64usize, (1i32, 1i32), false),
        (64, 64, (0, 0), false),
        (64, 64, (1, 1), true),
        (96, 80, (1, 1), false),
        (96, 80, (0, 0), false),
        (96, 80, (1, 1), true),
        (100, 76, (1, 1), false),
        (100, 76, (0, 0), false),
        (100, 76, (1, 1), true),
    ] {
        run(w, h, 8, ss, mono, 36);
    }
    assert_eq!(
        n,
        30 + 18 + 18 + 12 + 9,
        "15 combos x cq{{2,6}} + 9 bd8 x cq{{16,28}} + 9 bd8 x cq{{44,52}} + 6 bd10 x cq{{16,36}} + 9 band-3"
    );
    // Speed-0 allintra codes TX_MODE_SELECT — the multi-txb path must be live.
    assert!(select_seen > 0, "no TX_MODE_SELECT stream decoded");
    // All four coefficient-CDF qindex bands exercised on real streams.
    assert!(bands.iter().all(|&b| b > 0), "band coverage {bands:?}");
    // BOTH loop-filter populations must be present: level-0 streams (the
    // application is a gated no-op — the original 56-stream envelope stays
    // green) and genuinely deblocked streams, luma AND chroma.
    println!(
        "lf coverage: zero={zero_lf} nonzero_luma={nonzero_luma_lf} nonzero_chroma={nonzero_chroma_lf} of {n}"
    );
    // Observed on this deterministic content (2026-07-14 probe): 76 zero,
    // 11 nonzero-luma of which 6 nonzero-chroma — the floors keep both
    // populations from silently vanishing.
    assert!(zero_lf >= 40, "level-0 population collapsed ({zero_lf})");
    assert!(
        nonzero_luma_lf >= 10,
        "deblocked-luma population too small ({nonzero_luma_lf})"
    );
    assert!(
        nonzero_chroma_lf >= 5,
        "deblocked-chroma population too small ({nonzero_chroma_lf})"
    );
}

#[test]
fn high_q_deblocked_stream_decodes_byte_identical() {
    // The old envelope boundary, now INSIDE the envelope: this config
    // (100x76 4:2:0 cq 52 on the deterministic run_config content) picks
    // filter_level [4, 0, 0, 6] — a genuinely deblocked stream (luma vert +
    // chroma V) whose output must match the C decoder byte-for-byte.
    // (64x64 cq 60 — the old rejection-test config — actually picks level 0
    // on its content; its old Err-arm never fired.)
    let (len, _, _, lf) = run_config(&Cfg {
        w: 100,
        h: 76,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 52,
    });
    println!("deblocked companion: {len} bytes, filter_level = {lf:?}");
    // This companion pins REAL deblocking: if the encoder ever stops picking
    // nonzero levels here the assertion below flags the lost coverage.
    assert_ne!(lf, [0; 4], "cq 52 no longer picks deblocking");
}

#[test]
fn deblocked_422_chroma_is_rejected_not_misdecoded() {
    // 4:2:2 with nonzero CHROMA deblock levels is the one deblock combination
    // out of envelope (libaom's 4:2:2 chroma path indexes
    // max_txsize_rect_lookup[BLOCK_INVALID] out of bounds for tall blocks —
    // not portable). Luma-only or level-0 4:2:2 must still decode identical.
    // This config picks u=16 v=17 (probed 2026-07-14) — the rejection arm.
    let (w, h, bd) = (96usize, 80usize, 10);
    let cq = 44;
    let seed = ((w as u64) << 32) ^ ((h as u64) << 16) ^ ((bd as u64) << 8) ^ cq as u64;
    let y = gen_plane(w, h, bd, seed ^ 0x1111, false);
    let u = gen_plane(w / 2, h, bd, seed ^ 0x2222, true);
    let v = gen_plane(w / 2, h, bd, seed ^ 0x3333, true);
    let bytes = c::ref_encode_av1_kf(&y, &u, &v, w, h, bd, false, 1, 0, cq, 0, false);
    match decode_frame_obus(&bytes) {
        Err(e) => {
            println!("422 arm: REJECTED ({e})");
            assert!(
                e.contains("4:2:2 chroma deblocking"),
                "expected the 4:2:2 chroma-deblock envelope error, got: {e}"
            );
        }
        Ok(f) => {
            // Chroma levels 0 on this content: legitimately in envelope
            // (luma-only 4:2:2 deblocking is supported) — must match C.
            println!("422 arm: decoded, filter_level = {:?}", f.filter_level);
            assert_eq!(
                (f.filter_level[2], f.filter_level[3]),
                (0, 0),
                "Ok decode implies zero chroma levels"
            );
            let cref = c::ref_decode_av1_kf(&bytes, w, h);
            assert_eq!(f.y, cref.y);
            assert_eq!(f.u, cref.u);
            assert_eq!(f.v, cref.v);
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
