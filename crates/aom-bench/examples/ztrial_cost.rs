//! Zenaom SCM-trial cost measurement — `encode_key_frame` wall time in
//! `LibaomExact` vs `Zenaom` on (a) a detector-negative photo frame carrying
//! a UI patch large enough to nominate the trial, and (b) pure photographic
//! noise where the margin gate must keep the trial off. Measured 2026-09-13
//! (release, single-threaded): firing cells 1.07x–1.19x, gate-off and photo
//! cells 1.00x — the fixed-32x32 trial passes at q=244 carry no partition
//! search, so a fire costs a small fraction of an encode, not 2x.
//!
//! Usage: `cargo run --release -p zenav1-aom-bench --example ztrial_cost`
use aom_encode::key_frame::{KeyFrameConfig, KeyFrameMode, KeyFramePlanes, encode_key_frame};
use std::time::Instant;

fn photo_sample(r: usize, col: usize) -> i32 {
    let mut x = (r as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add((col as u32).wrapping_mul(40503));
    x ^= x >> 13;
    x = x.wrapping_mul(1_274_126_177);
    x ^= x >> 16;
    (x & 0xff) as i32 / 4 + (32 + (r + col) * 100 / 512) as i32
}
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
fn patched(w: usize, h: usize, pw: usize, ph: usize) -> impl Fn(usize, usize) -> i32 {
    let (x0, y0) = ((w - pw) / 2, (h - ph) / 2);
    move |r, col| {
        if r >= y0 && r < y0 + ph && col >= x0 && col < x0 + pw {
            ui_sample(r - y0, col - x0)
        } else {
            photo_sample(r, col)
        }
    }
}
fn planes(w: usize, h: usize, f: impl Fn(usize, usize) -> i32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = f(r, c).clamp(0, 255) as u16;
        }
    }
    (
        y,
        vec![128u16; (w / 2) * (h / 2)],
        vec![128u16; (w / 2) * (h / 2)],
    )
}
fn cfg(w: usize, h: usize, cq: i32, speed: i32, mode: KeyFrameMode) -> KeyFrameConfig {
    let mut c = KeyFrameConfig::allintra_speed0(w, h, 8, false, 1, 1, cq);
    c.cpu_used = speed;
    c.enable_restoration = true;
    c.enable_palette = true;
    c.enable_intrabc = true;
    c.mode = mode;
    c
}
fn timeit(y: &[u16], u: &[u16], v: &[u16], c: &KeyFrameConfig, reps: usize) -> f64 {
    // warm-up
    let _ = encode_key_frame(KeyFramePlanes::new(y, u, v), c).unwrap();
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = encode_key_frame(KeyFramePlanes::new(y, u, v), c).unwrap();
    }
    t0.elapsed().as_secs_f64() / reps as f64 * 1000.0
}
fn main() {
    for &(w, h, pw, ph, name) in &[
        (256usize, 128usize, 64usize, 64usize, "flip64"),
        (512, 256, 64, 64, "gate-off64"),
        (512, 256, 128, 128, "flip128"),
    ] {
        let (y, u, v) = planes(w, h, patched(w, h, pw, ph));
        let t_e = timeit(&y, &u, &v, &cfg(w, h, 32, 3, KeyFrameMode::LibaomExact), 3);
        let t_z = timeit(&y, &u, &v, &cfg(w, h, 32, 3, KeyFrameMode::Zenaom), 3);
        println!(
            "{name} {w}x{h}: exact={t_e:.1}ms zen={t_z:.1}ms ratio={:.2}x",
            t_z / t_e
        );
    }
    // photo control — gate should keep this at 1.00x
    for &(w, h) in &[(256usize, 128usize), (512, 256)] {
        let (y, u, v) = planes(w, h, photo_sample);
        let t_e = timeit(&y, &u, &v, &cfg(w, h, 32, 3, KeyFrameMode::LibaomExact), 3);
        let t_z = timeit(&y, &u, &v, &cfg(w, h, 32, 3, KeyFrameMode::Zenaom), 3);
        println!(
            "PHOTO {w}x{h}: exact={t_e:.1}ms zen={t_z:.1}ms ratio={:.2}x",
            t_z / t_e
        );
    }
}
