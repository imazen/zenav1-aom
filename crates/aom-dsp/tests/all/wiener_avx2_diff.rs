//! Differential: the port's dispatching `wiener_convolve_add_src` vs the REAL
//! exported `av1_highbd_wiener_convolve_add_src_avx2` — the kernel
//! `wiener_impl_v3` mirrors instruction-for-instruction (shifted-window
//! `madd_epi16` horizontal, row-pair-unpack vertical, `1<<FILTER_BITS` folded
//! into tap 3, `packs_epi32` intermediate order).
//!
//! Domain notes:
//! * C-avx2 asserts `w % 8 == 0`; narrower/odd widths are a port extension
//!   (they take the scalar/i32x8 bodies).
//! * For `w % 16 == 8` C's 16-column stores write up to 8 columns PAST `w`
//!   into the dst margin; the port deliberately keeps the margin intact
//!   (`kernels_diff` asserts the whole buffer), so this test compares only
//!   the `w x h` output block.
//! * `madd_epi16` reads u16 inputs as SIGNED i16: for samples >= 32768 — off
//!   every valid bit depth — C-avx2 and the v3 mirror agree, while the
//!   zero-extending scalar/i32x8 bodies do not. The full-u16 adversarial salt
//!   below therefore only runs on `w >= 16` shapes (v3's domain); `w == 8`
//!   pins the scalar fallback against C-avx2's valid-column outputs instead.
//! * Same for taps: arbitrary i16 taps wrap `tap[3] + 128` differently from
//!   the scalar body's separate `src[3] << 7` term, so adversarial taps are
//!   also gated to `w >= 16`.

use aom_dsp::restore::wiener::wiener_convolve_add_src;
use aom_sys_ref as c;
use archmage::SimdToken;

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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u64) as i32
    }
}

const M: usize = 8;

/// True when the dispatching entry takes the v3 (C-avx2 mirror) tier.
fn v3_live() -> bool {
    archmage::X64V3Token::summon().is_some() && std::env::var_os("AOM_FORCE_SCALAR").is_none()
}

#[test]
fn wiener_v3_matches_real_highbd_avx2() {
    // Held for the test body: the v3 observation must not race a sibling
    // test's token permutation.
    let _token_guard = archmage::testing::lock_token_testing();
    if !v3_live() {
        eprintln!("v3 not dispatched on this host/pin; skipping");
        return;
    }
    let mut rng = Rng(0xB1E9_0000_0000_A11C);
    // w % 8 == 0 throughout (the kernel's own assert). w % 16 == 8 cases
    // (8, 24, 40) exercise C's overrun vs the port's overlap-back tail.
    let dims: &[(usize, usize)] = &[
        (8, 8),
        (16, 8),
        (24, 16),
        (32, 32),
        (40, 24),
        (48, 56),
        (64, 64),
        (128, 8),
        (128, 64),
    ];
    for &bd in &[8, 10, 12] {
        for &(w, h) in dims {
            for case in 0..10 {
                // Right/top slack beyond M: C reads/writes a few elements past
                // the documented +4 margin on its last tile.
                let (buf_w, buf_h) = (w + 2 * M + 16, h + 2 * M + 8);
                let adversarial = w >= 16 && case % 3 == 2;
                let src: Vec<u16> = (0..buf_w * buf_h)
                    .map(|_| {
                        if adversarial {
                            rng.next() as u16 // full u16, incl. >= 32768 (i16-negative to madd)
                        } else {
                            (rng.next() & ((1 << bd) - 1)) as u16
                        }
                    })
                    .collect();
                let dst0: Vec<u16> = (0..buf_w * buf_h)
                    .map(|_| (rng.next() & ((1 << bd) - 1)) as u16)
                    .collect();
                let mut hf = [0i16; 8];
                let mut vf = [0i16; 8];
                for f in [&mut hf, &mut vf] {
                    if adversarial {
                        // Full i16 taps — incl. tap3 values that wrap the
                        // folded +1<<FILTER_BITS.
                        for v in f.iter_mut().take(7) {
                            *v = rng.next() as i16;
                        }
                    } else {
                        let chroma = case % 5 == 4;
                        f[0] = if chroma { 0 } else { rng.range(-5, 10) as i16 };
                        f[1] = rng.range(-23, 8) as i16;
                        f[2] = rng.range(-17, 46) as i16;
                        f[3] = -2 * (f[0] + f[1] + f[2]);
                        f[4] = f[2];
                        f[5] = f[1];
                        f[6] = f[0];
                    }
                }
                let off = M * buf_w + M;
                let mut dst_c = dst0.clone();
                c::ref_wiener_convolve_hbd_avx2(
                    &src, &mut dst_c, buf_w, buf_h, M, M, w, h, &hf, &vf, bd,
                );
                let mut dst_r = dst0.clone();
                wiener_convolve_add_src(
                    &src, off, buf_w, &mut dst_r, off, buf_w, &hf, &vf, w, h, bd,
                );
                for r in 0..h {
                    let row = (M + r) * buf_w + M;
                    assert_eq!(
                        &dst_r[row..row + w],
                        &dst_c[row..row + w],
                        "wiener {w}x{h} bd{bd} case {case} row {r} (adversarial={adversarial})"
                    );
                }
            }
        }
    }
}
