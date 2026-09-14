//! The zenaom-mode SCM trial (`av1_determine_sc_tools_with_encoding`)
//! against both C oracles: one-pass aomenc (which never runs the trial —
//! KB-66) and the two-pass oracle (which does), plus the gating contract —
//! `LibaomExact` must never pay for or observe the trial.
//!
//! What is asserted here, and what is deliberately not: `LibaomExact`
//! stays byte-identical to C's ONE-PASS stream on every cell; `Zenaom`
//! flips `allow_screen_content_tools` on the detector-negative-but-
//! profitable cells and produces a conformant stream. The trial DECISION
//! agrees with C's own (`AOM_SCT_TRIAL_DBG` / `AOM_SCT_C_DBG` prints show
//! the same win/lose on every cell of the sweep). Byte-identity to C's
//! TWO-PASS stream is NOT the gate: the last pass encodes at a
//! stats-derived qindex, so only the decision (and, on 256x256 cq32 s0,
//! coincidentally the bytes) can match.

use aom_encode::key_frame::{KeyFrameConfig, KeyFrameMode, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;

/// Deterministic hashed noise over a gentle gradient — genuinely
/// photographic per the screen detector (many colours per 16x16 block, so
/// the block lands in `count_photo`, not `count_palette`).
fn photo_sample(r: usize, col: usize) -> i32 {
    let mut x = (r as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add((col as u32).wrapping_mul(40503));
    x ^= x >> 13;
    x = x.wrapping_mul(1_274_126_177);
    x ^= x >> 16;
    (x & 0xff) as i32 / 4 + (32 + (r + col) * 100 / 512) as i32
}

/// Few exact colours with 8px block structure — palette material.
fn ui_sample(r: usize, col: usize) -> i32 {
    let band = (r / 12) % 4;
    let cell = ((r / 8) + (col / 8)) % 2;
    match (band, cell) {
        (0, 0) => 16,
        (0, _) => 235,
        (1, 0) => 60,
        (1, _) => 200,
        (2, 0) => 120,
        (2, _) => 16,
        (_, 0) => 235,
        (_, _) => 60,
    }
}

/// A photo frame with a `pw x ph` UI patch centred in it — the
/// detector-negative-but-profitable class: small enough that the colour-
/// count detector stays under its ~10%-of-16x16-blocks threshold, large
/// enough that palette wins whole `BLOCK_32X32` trial leaves.
fn patched_sample(w: usize, h: usize, pw: usize, ph: usize) -> impl Fn(usize, usize) -> i32 {
    let (x0, y0) = ((w - pw) / 2, (h - ph) / 2);
    move |r, col| {
        if r >= y0 && r < y0 + ph && col >= x0 && col < x0 + pw {
            ui_sample(r - y0, col - x0)
        } else {
            photo_sample(r, col)
        }
    }
}

fn planes_of(w: usize, h: usize, f: impl Fn(usize, usize) -> i32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = f(r, col).clamp(0, 255) as u16;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    (y, vec![128u16; cw * ch], vec![128u16; cw * ch])
}

/// Config matching the screen-content oracle: palette + IntraBC enabled,
/// restoration on, CDEF off.
fn cfg_for(w: usize, h: usize, cq: i32, speed: i32, mode: KeyFrameMode) -> KeyFrameConfig {
    let mut cfg = KeyFrameConfig::allintra_speed0(w, h, 8, false, 1, 1, cq);
    cfg.cpu_used = speed;
    cfg.enable_restoration = true;
    cfg.enable_palette = true;
    cfg.enable_intrabc = true;
    cfg.mode = mode;
    cfg
}

fn port(y: &[u16], u: &[u16], v: &[u16], cfg: &KeyFrameConfig) -> Vec<u8> {
    encode_key_frame(KeyFramePlanes { y, u, v }, cfg).expect("encode_key_frame refused")
}

/// C's ONE-PASS stream (aomenc allintra default envelope — the trial is
/// unreachable, KB-66).
fn c_one_pass(y: &[u16], u: &[u16], v: &[u16], w: usize, h: usize, cq: i32, speed: i32) -> Vec<u8> {
    c::ref_encode_av1_kf_screen_content(
        y, u, v, w, h, 8, false, 1, 1, cq, speed, false, true, 2, 0,
        /*two_pass=*/ false, /*enable_palette=*/ true, /*enable_intrabc=*/ true,
    )
}

/// C's TWO-PASS stream — `AOM_RC_LAST_PASS` reaches `encode_with_recode_loop`
/// and so DOES run `av1_determine_sc_tools_with_encoding`.
fn c_two_pass(y: &[u16], u: &[u16], v: &[u16], w: usize, h: usize, cq: i32, speed: i32) -> Vec<u8> {
    c::ref_encode_av1_kf_screen_content(
        y, u, v, w, h, 8, false, 1, 1, cq, speed, false, true, 2, 0,
        /*two_pass=*/ true, /*enable_palette=*/ true, /*enable_intrabc=*/ true,
    )
}

/// The zenaom opt-in: on a detector-negative frame carrying a 64x64 UI
/// patch, the margin gate fires, the trial's tools-on pass wins
/// (`diff 0.76 dB / palette_ratio 0.125` at 256x128 s0 — C's own trial
/// independently prints `win=1` on the same cell), and the main encode
/// flips to screen tools. Asserted: the flip changes the stream, the
/// flipped stream is CONFORMANT through the real C decoder, and the
/// one-pass stream is untouched (`LibaomExact` == C1p).
#[test]
fn zenaom_trial_flips_detector_negative_screen_content() {
    c::ref_init();
    for &(w, h) in &[(256usize, 128usize), (256, 256)] {
        for &speed in &[0i32, 3] {
            let label = format!("{w}x{h} patch64 s{speed}");
            let (y, u, v) = planes_of(w, h, patched_sample(w, h, 64, 64));
            let exact = port(&y, &u, &v, &cfg_for(w, h, 32, speed, KeyFrameMode::LibaomExact));
            let zen = port(&y, &u, &v, &cfg_for(w, h, 32, speed, KeyFrameMode::Zenaom));
            assert_eq!(
                exact,
                c_one_pass(&y, &u, &v, w, h, 32, speed),
                "{label}: LibaomExact diverged from C one-pass"
            );
            assert_ne!(
                zen, exact,
                "{label}: the trial did not flip a detector-negative cell it \
                 is built to rescue"
            );
            // Anti-vacuity: C's own trial also flipped this cell — its
            // two-pass stream must differ from its one-pass stream.
            let c1 = c_one_pass(&y, &u, &v, w, h, 32, speed);
            let c2 = c_two_pass(&y, &u, &v, w, h, 32, speed);
            assert_ne!(c1, c2, "{label}: C's trial did not flip either");
            // Conformance: the flipped stream must decode identically
            // through the real C decoder and the port decoder.
            let c_dec = c::ref_decode_av1_kf(&zen, w, h);
            let p_dec = aom_decode::frame::decode_frame_obus(&zen)
                .unwrap_or_else(|e| panic!("{label}: port decode of flipped stream: {e}"));
            assert_eq!(
                (&p_dec.y, &p_dec.u, &p_dec.v),
                (&c_dec.y, &c_dec.u, &c_dec.v),
                "{label}: port-decode(flipped stream) != C-decode(flipped stream)"
            );
        }
    }
}

/// The cost contract: pure photographic content never pays for the trial
/// — the detector's net score is negative, the margin gate stays shut,
/// and the zenaom stream is byte-identical to `LibaomExact`. (If this ever
/// fails with the trial merely DECLINING, the stream is still identical —
/// a failure here means either a flip on photo content, which is a
/// quality regression to investigate, or a trial-side leak into the main
/// encode.)
#[test]
fn zenaom_photo_content_does_not_flip() {
    c::ref_init();
    for &(w, h) in &[(256usize, 128usize), (256, 256)] {
        let (y, u, v) = planes_of(w, h, photo_sample);
        let exact = port(&y, &u, &v, &cfg_for(w, h, 32, 3, KeyFrameMode::LibaomExact));
        let zen = port(&y, &u, &v, &cfg_for(w, h, 32, 3, KeyFrameMode::Zenaom));
        assert_eq!(zen, exact, "{w}x{h} PHOTO: zenaom != exact");
        // With a caller hint at/above SCM_TRIAL_HINT_MIN the trial RUNS on
        // the same photo — it must still decline (and must not corrupt or
        // panic).
        let mut hinted = cfg_for(w, h, 32, 3, KeyFrameMode::Zenaom);
        hinted.screen_likelihood = Some(0.5);
        let zen_hinted = port(&y, &u, &v, &hinted);
        let c_dec = c::ref_decode_av1_kf(&zen_hinted, w, h);
        let p_dec = aom_decode::frame::decode_frame_obus(&zen_hinted)
            .unwrap_or_else(|e| panic!("{w}x{h} hinted decode: {e}"));
        assert_eq!((&p_dec.y, &p_dec.u, &p_dec.v), (&c_dec.y, &c_dec.u, &c_dec.v));
    }
}

/// The mode contract, hard: `LibaomExact` (the default) must produce the
/// same bytes as C's one-pass encode on every cell, regardless of whether
/// the trial would have flipped the frame.
#[test]
fn libaom_exact_ignores_the_trial() {
    c::ref_init();
    let (w, h) = (256usize, 128usize);
    let (y, u, v) = planes_of(w, h, patched_sample(w, h, 64, 64));
    for &cq in &[20i32, 32, 50] {
        for &speed in &[0i32, 3, 6] {
            let ours = port(&y, &u, &v, &cfg_for(w, h, cq, speed, KeyFrameMode::LibaomExact));
            let theirs = c_one_pass(&y, &u, &v, w, h, cq, speed);
            assert_eq!(
                ours, theirs,
                "LibaomExact {w}x{h} cq{cq} s{speed}: port != C one-pass"
            );
        }
    }
}

/// Diagnostic sweep across frame size / patch size / speed — prints the
/// four-way comparison plus both sides' trial decisions. `#[ignore]`d:
/// ~24 cells of multi-pass debug-profile encodes. Run with
/// `cargo test -p zenav1-aom-encode --test all --features __internals
/// zenaom_scm_trial::probe_zenaom_trial_matrix -- --ignored --nocapture`
/// and `AOM_SCT_DBG=1 AOM_SCT_TRIAL_DBG=1` to see the decisions.
#[test]
#[ignore]
fn probe_zenaom_trial_matrix() {
    c::ref_init();
    for &(w, h) in &[(256usize, 128usize), (256, 256), (512, 256)] {
        for &(pw, ph) in &[(48usize, 48usize), (64, 64), (96, 64), (128, 96)] {
            for &speed in &[0i32, 3] {
                let (y, u, v) = planes_of(w, h, patched_sample(w, h, pw, ph));
                let exact = port(&y, &u, &v, &cfg_for(w, h, 32, speed, KeyFrameMode::LibaomExact));
                let zen = port(&y, &u, &v, &cfg_for(w, h, 32, speed, KeyFrameMode::Zenaom));
                let c1 = c_one_pass(&y, &u, &v, w, h, 32, speed);
                let c2 = c_two_pass(&y, &u, &v, w, h, 32, speed);
                eprintln!(
                    "{w}x{h} patch{pw}x{ph} s{speed}: \
                     exact==C1p:{} zen==C1p:{} zen==C2p:{} zen==exact:{} \
                     sizes exact={} zen={} c1p={} c2p={}",
                    exact == c1,
                    zen == c1,
                    zen == c2,
                    zen == exact,
                    exact.len(),
                    zen.len(),
                    c1.len(),
                    c2.len()
                );
            }
        }
        let (y, u, v) = planes_of(w, h, photo_sample);
        let exact = port(&y, &u, &v, &cfg_for(w, h, 32, 3, KeyFrameMode::LibaomExact));
        let zen = port(&y, &u, &v, &cfg_for(w, h, 32, 3, KeyFrameMode::Zenaom));
        let c1 = c_one_pass(&y, &u, &v, w, h, 32, 3);
        eprintln!(
            "{w}x{h} PHOTO s3: exact==C1p:{} zen==exact:{} sizes {} {} {}",
            exact == c1,
            zen == exact,
            exact.len(),
            zen.len(),
            c1.len()
        );
    }
}
