//! C7 grain-estimator — END-TO-END DIFFERENTIAL gate for the full
//! `--denoise-noise-level` path (`aom_denoise_and_model_run`).
//!
//! Runs the Rust one-shot entry ([`aom_encode::denoise::estimate_film_grain`])
//! and the REAL `aom_denoise_and_model_run` (via a YV12 buffer) over an
//! IDENTICAL synthetic frame, and asserts:
//!   * the same `apply_grain` outcome,
//!   * BYTE-IDENTICAL quantized grain params (serialized `filmgrn1`), AND
//!   * BYTE-IDENTICAL denoised planes.
//!
//! This closes the loop: source → flat-block find → FFT Wiener denoise → AR
//! model fit → grain-param quantize, all composed exactly as libaom does.
//! Covered: 8-bit + 10-bit, 4:4:4 + 4:2:0, 32-aligned dims.

use aom_encode::denoise::estimate_film_grain;
use aom_encode::grain_table::{write_film_grain_table, GrainTableEntry};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn noise(&mut self, amp: i32) -> i32 {
        (self.next() % (2 * amp as u64 + 1)) as i32 - amp
    }
}

/// Smooth ramp + moderate noise (mostly-flat content → the estimator finds
/// enough flat blocks and produces a valid grain estimate).
fn make_plane(rng: &mut Rng, w: usize, h: usize, maxv: i32, base: i32) -> Vec<u16> {
    (0..w * h)
        .map(|i| {
            let x = (i % w) as i32;
            let y = (i / w) as i32;
            let d = base + (x * 3 + y * 2) % 16 - 8 + rng.noise(10);
            d.clamp(0, maxv) as u16
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case(w: usize, h: usize, ss_x: i32, ss_y: i32, bit_depth: i32, seed: u64) -> bool {
    assert_eq!(w % 32, 0, "dims must be 32-aligned for the YV12 tight-buffer path");
    let maxv = (1i32 << bit_depth) - 1;
    let block_size = 32usize;
    let noise_level = 2.5f32;
    let random_seed = 0x51_23i32;
    let mut rng = Rng::new(seed);
    let cw = w >> ss_x;
    let ch = h >> ss_y;
    let planes = [
        make_plane(&mut rng, w, h, maxv, maxv / 2),
        make_plane(&mut rng, cw, ch, maxv, maxv / 2 + 12),
        make_plane(&mut rng, cw, ch, maxv, maxv / 2 - 12),
    ];
    let data = [&planes[0][..], &planes[1][..], &planes[2][..]];

    // ---- Port ----
    let port = estimate_film_grain(
        data, w, h, w, cw, ch, ss_x, ss_y, false, bit_depth, block_size, noise_level, random_seed,
    );

    // ---- C oracle ----
    let cref = c::ref_denoise_and_model_run(
        data, w, h, ss_x, ss_y, bit_depth, block_size, noise_level, random_seed,
    );

    assert_eq!(
        port.is_some(),
        cref.apply_grain,
        "apply_grain mismatch ({w}x{h} ss{ss_x}{ss_y} bd{bit_depth})"
    );
    if let Some((fg, port_den)) = port {
        // Grain params: serialize and byte-compare.
        let c_table = cref.grain_table.expect("C grain table");
        let port_table = write_film_grain_table(&[GrainTableEntry {
            params: fg,
            start_time: 0,
            end_time: i64::MAX,
        }]);
        assert_eq!(
            port_table,
            c_table,
            "grain params differ ({w}x{h} ss{ss_x}{ss_y} bd{bit_depth})\n--- C ---\n{}\n--- port ---\n{}",
            String::from_utf8_lossy(&c_table),
            String::from_utf8_lossy(&port_table)
        );
        // Denoised planes: byte-identical.
        let names = ["Y", "U", "V"];
        for cc in 0..3 {
            assert_eq!(
                port_den[cc],
                cref.denoised[cc],
                "denoised {} differs ({w}x{h} ss{ss_x}{ss_y} bd{bit_depth}): first diff {:?}",
                names[cc],
                port_den[cc].iter().zip(&cref.denoised[cc]).position(|(a, b)| a != b)
            );
        }
        true
    } else {
        false
    }
}

#[test]
fn denoise_and_model_matches_c() {
    c::ref_init();
    // (w, h, ss_x, ss_y, bit_depth)
    let configs = [
        (128usize, 128usize, 1i32, 1i32, 8i32),
        (128, 128, 0, 0, 8),
        (128, 128, 1, 1, 10),
        (96, 96, 0, 0, 10),
        (160, 128, 1, 1, 8),
    ];
    let mut estimated = 0;
    for (i, &(w, h, sx, sy, bd)) in configs.iter().enumerate() {
        if run_case(w, h, sx, sy, bd, 0xE2E_0011 ^ (i as u64) << 16) {
            estimated += 1;
        }
    }
    // Anti-vacuity: the estimator actually produced grain on most cases (not a
    // vacuous "both returned no-estimate" pass).
    assert!(
        estimated >= configs.len() - 1,
        "estimator produced grain on too few cases ({estimated}/{})",
        configs.len()
    );
    println!(
        "denoise_and_model_diff: {} configs (8/10-bit x 444/420) byte-identical to aom_denoise_and_model_run ({estimated} grain estimates)",
        configs.len()
    );
}
