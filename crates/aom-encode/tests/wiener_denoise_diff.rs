//! C7 grain-estimator — DIFFERENTIAL gate for the Wiener DENOISE.
//!
//! Runs the Rust port ([`aom_encode::denoise::wiener_denoise_2d`]) and the REAL
//! exported `aom_wiener_denoise_2d` over an IDENTICAL synthetic frame + flat
//! noise PSD, and asserts the denoised planes are BYTE-IDENTICAL. This is the
//! last FFT-dependent stage of the estimator: it exercises the overlapped-block
//! FFT Wiener filter (forward/filter/inverse), the half-cosine window (`cos`),
//! the planar flat-block extraction, and the Floyd–Steinberg dither/quantize —
//! all end-to-end. Covered: 8-bit (lowbd) + 10-bit (highbd), 4:4:4 + 4:2:0.

use aom_encode::denoise::wiener_denoise_2d;
use aom_encode::noise_fft::noise_psd_get_default_value;
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

fn make_plane(rng: &mut Rng, w: usize, h: usize, maxv: i32, base: i32) -> Vec<u16> {
    (0..w * h)
        .map(|i| {
            let x = (i % w) as i32;
            let y = (i / w) as i32;
            let d = base + (x * 11 + y * 5) % 20 - 10 + rng.noise(12);
            d.clamp(0, maxv) as u16
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case(w: usize, h: usize, csx: i32, csy: i32, bit_depth: i32, use_highbd: bool, seed: u64) {
    let maxv = (1i32 << bit_depth) - 1;
    let block_size = 32usize;
    let noise_level = 2.5f32;
    let mut rng = Rng::new(seed);
    let cw = w >> csx;
    let ch = h >> csy;
    let planes = [
        make_plane(&mut rng, w, h, maxv, maxv / 2),
        make_plane(&mut rng, cw, ch, maxv, maxv / 2 + 15),
        make_plane(&mut rng, cw, ch, maxv, maxv / 2 - 15),
    ];
    let data = [&planes[0][..], &planes[1][..], &planes[2][..]];
    let strides = [w, cw, cw];
    let plane_lens = [w * h, cw * ch, cw * ch];

    // Flat noise PSD, luma + chroma (as aom_denoise_and_model does).
    let y_noise = noise_psd_get_default_value(block_size, noise_level);
    let uv_noise = noise_psd_get_default_value(block_size >> csx, noise_level);
    let psd_y = vec![y_noise; block_size * block_size];
    let psd_uv = vec![uv_noise; block_size * block_size];
    let psd = [&psd_y[..], &psd_uv[..], &psd_uv[..]];

    // ---- Port ----
    let mut d0 = vec![0u16; plane_lens[0]];
    let mut d1 = vec![0u16; plane_lens[1]];
    let mut d2 = vec![0u16; plane_lens[2]];
    let ok = wiener_denoise_2d(
        data,
        [&mut d0, &mut d1, &mut d2],
        w,
        h,
        strides,
        [csx, csy],
        psd,
        block_size,
        bit_depth,
    );
    assert!(ok, "port wiener_denoise_2d failed");

    // ---- C oracle ----
    let cout = c::ref_wiener_denoise_2d(
        data, w, h, strides, csx, csy, psd, block_size, bit_depth, use_highbd, plane_lens,
    )
    .expect("C wiener_denoise_2d");

    let port = [d0, d1, d2];
    let names = ["Y", "U", "V"];
    for cc in 0..3 {
        assert_eq!(
            port[cc], cout[cc],
            "plane {} mismatch ({}x{} cs{}{} bd{}): first diff at {:?}",
            names[cc],
            w,
            h,
            csx,
            csy,
            bit_depth,
            port[cc].iter().zip(&cout[cc]).position(|(a, b)| a != b)
        );
    }
}

#[test]
fn wiener_denoise_matches_c() {
    c::ref_init();
    // (w, h, csx, csy, bit_depth, use_highbd)
    let configs = [
        (128usize, 128usize, 0i32, 0i32, 8i32, false),
        (128, 128, 1, 1, 8, false),
        (128, 128, 0, 0, 10, true),
        (128, 128, 1, 1, 10, true),
        (96, 160, 1, 1, 8, false),
        (160, 96, 0, 0, 10, true),
    ];
    for (i, &(w, h, csx, csy, bd, hbd)) in configs.iter().enumerate() {
        run_case(w, h, csx, csy, bd, hbd, 0xD0_1234 ^ (i as u64) << 12);
    }
    println!("wiener_denoise_diff: 6 configs (8/10-bit x 444/420, incl. partial-block dims) byte-identical to C");
}
