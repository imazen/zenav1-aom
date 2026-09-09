//! Lowbd (bd8, u8 pixel) deblock differential — the loopfilter family's entry
//! in the parallel bd8 pipeline (see `aom_dsp::lowbd`). Asserts, over the
//! family's full input space (both edge directions, all 4 widths, the
//! mask/flat/flat2-triggering pixel windows + threshold ranges), that the u8
//! lowbd kernel is byte-identical to BOTH:
//!   * the REAL C lowbd kernels (`aom_lpf_*_c`, via `c::ref_lpf`) — the strongest
//!     oracle; and
//!   * the port's own u16 highbd path run at `bd = 8` (`highbd::{horizontal,
//!     vertical}`) — the path bd8 frames take today, which this replaces.
//!
//! Both the SIMD-dispatched entry (`loopfilter::{horizontal, vertical}`) and the
//! never-dispatched pure-scalar reference (`loopfilter::{horizontal_scalar,
//! vertical_scalar}`) are checked, so the scalar tier is proven directly (no
//! `AOM_FORCE_SCALAR=1` needed for coverage) AND under `AOM_FORCE_SCALAR=1` the
//! SIMD entry collapses onto that same scalar core. Model: `inv_txfm2d_lowbd_diff`.

use aom_dsp::loopfilter::{self, highbd};
use aom_sys_ref as c;

const PITCH: usize = 32;
const ROWS: usize = 32;
const CENTER: usize = 12 * PITCH + 12; // row 12, col 12 — >=7 margin all sides

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn u8(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    fn upto(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

fn gen_buf(rng: &mut Rng, strategy: u32) -> Vec<u8> {
    match strategy {
        0 => (0..PITCH * ROWS).map(|_| rng.u8()).collect(), // fully random
        _ => {
            // near-flat: base + small noise (triggers flat/flat2 paths)
            let base = rng.u8() as i32;
            let amp = 1 + rng.upto(8) as i32;
            (0..PITCH * ROWS)
                .map(|_| (base + rng.upto((2 * amp + 1) as u32) as i32 - amp).clamp(0, 255) as u8)
                .collect()
        }
    }
}

/// One shared body: assert the `u8` kernel `apply` matches the REAL C lowbd
/// kernel AND the port's u16 bd8 highbd kernel, over the full input space.
fn kernel_matches(mut apply: impl FnMut(u8, u32, &mut [u8], usize, usize, u8, u8, u8), label: &str) {
    c::ref_init();
    let mut rng = Rng(0x_10fd_ead0_1234_5678);
    let mut checks = 0u64;
    for &dir in b"hv" {
        for &width in &[4u32, 6, 8, 14] {
            for _ in 0..30_000 {
                let strategy = rng.upto(3);
                let base = gen_buf(&mut rng, strategy);
                let blimit = if rng.upto(2) == 0 { rng.u8() } else { (16 + rng.upto(200)) as u8 };
                let limit = if rng.upto(2) == 0 { rng.u8() } else { (1 + rng.upto(64)) as u8 };
                let thresh = rng.u8();

                // u8 lowbd kernel under test.
                let mut got_u8 = base.clone();
                apply(dir, width, &mut got_u8, CENTER, PITCH, blimit, limit, thresh);

                // REAL C lowbd (aom_lpf_*_c).
                let mut want_c = base.clone();
                c::ref_lpf(dir, width, &mut want_c, CENTER, PITCH, blimit, limit, thresh);

                // Port's u16 highbd path at bd = 8 (what bd8 frames run today).
                let mut got_hbd: Vec<u16> = base.iter().map(|&x| x as u16).collect();
                if dir == b'h' {
                    highbd::horizontal(width, &mut got_hbd, CENTER, PITCH, blimit, limit, thresh, 8);
                } else {
                    highbd::vertical(width, &mut got_hbd, CENTER, PITCH, blimit, limit, thresh, 8);
                }

                assert_eq!(
                    got_u8, want_c,
                    "{label}: u8 vs C lowbd dir={} width={width} bl={blimit} li={limit} th={thresh}",
                    dir as char
                );
                for (i, (&g8, &gh)) in got_u8.iter().zip(got_hbd.iter()).enumerate() {
                    assert_eq!(
                        g8 as u16, gh,
                        "{label}: u8 vs u16-highbd @ {i} dir={} width={width} bl={blimit} li={limit} th={thresh}",
                        dir as char
                    );
                }
                checks += 1;
            }
        }
    }
    assert!(checks > 200_000, "{label}: {checks}");
}

/// The SIMD-dispatched u8 entry (the production path) == C lowbd == u16 highbd.
#[test]
fn lowbd_kernel_byte_identical() {
    kernel_matches(
        |dir, w, buf, c0, p, bl, li, th| {
            if dir == b'h' {
                loopfilter::horizontal(w, buf, c0, p, bl, li, th);
            } else {
                loopfilter::vertical(w, buf, c0, p, bl, li, th);
            }
        },
        "SIMD",
    );
}

/// The never-dispatched pure-scalar u8 reference == C lowbd == u16 highbd.
/// Guarantees scalar-tier coverage without `AOM_FORCE_SCALAR=1`.
#[test]
fn lowbd_kernel_scalar_byte_identical() {
    kernel_matches(
        |dir, w, buf, c0, p, bl, li, th| {
            if dir == b'h' {
                loopfilter::horizontal_scalar(w, buf, c0, p, bl, li, th);
            } else {
                loopfilter::vertical_scalar(w, buf, c0, p, bl, li, th);
            }
        },
        "scalar",
    );
}
