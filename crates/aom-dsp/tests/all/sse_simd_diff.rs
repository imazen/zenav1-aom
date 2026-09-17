//! SIMD-vs-scalar differential for `sse_u16_u8` and `sse` (Gate-3 parity
//! rule 1: bit-identical, no slip), at every archmage token permutation.
//!
//! `sse_u16_u8` is a port-internal kernel (C has no mixed-width export — at
//! bd8 C's source plane is already u8, so `aom_sse` is the C twin of `sse`),
//! so the oracle here is the scalar transcription, with the C pin carried by
//! `dist_diff.rs` driving `sse` against `aom_sse_c`. Widths cover every
//! const-W arm (8/16/32/64), the generic fallback (24/48/128), the w==4
//! scalar route and a non-multiple-of-8 guard, on strided planes with
//! max-diff and flat boundary cases.

use aom_dsp::dist::{sse, sse_scalar, sse_u16_u8, sse_u16_u8_scalar};
// `summon()` comes from this trait; needed at MODULE scope because the
// non-vacuity counter below lives outside the fn-local `use` blocks.
use archmage::SimdToken;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

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
}

// Every const-W arm, the generic w%16==8 / wider-than-64 fallbacks, the w==4
// scalar route, and w=12 (non-multiple of 8 — public guard's scalar route).
const DIMS: &[(usize, usize)] = &[
    (4, 4),
    (4, 16),
    (4, 64),
    (4, 2),  // w4 arm's h % 4 != 0 scalar-tail route
    (4, 6),
    (8, 4),
    (8, 8),
    (8, 32),
    (16, 8),
    (16, 16),
    (16, 64),
    (24, 8),
    (32, 16),
    (32, 64),
    (48, 16),
    (64, 32),
    (64, 64),
    (128, 16),
    (12, 8),
];

#[test]
fn sse_family_simd_bit_identical_to_scalar_at_every_tier() {
    // Serialise: this sweep permutes PROCESS-GLOBAL dispatch
    // state; see `crate::dispatch_serial`.
    let _serial = crate::dispatch_serial::dispatch_serial();
    // No pre-flight token check — under AOM_FORCE_SCALAR=1 `summon()`
    // correctly returns None until `for_each_token_permutation` resets that
    // state. The real non-vacuity guard is `simd_perms >= 1` inside.
    let mut simd_perms = 0usize;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_tier| {
        if if cfg!(target_arch = "aarch64") {
            archmage::NeonToken::summon().is_some()
        } else {
            archmage::X64V3Token::summon().is_some()
        } {
            simd_perms += 1;
        }
        let mut rng = Rng(0x_55e_16a8_bd8_0001);
        for &(w, h) in DIMS {
            for case in 0..3 {
                let a_stride = w + (rng.next() % 9) as usize; // strided rows
                let b_stride = w + (rng.next() % 9) as usize;
                let mut a16: Vec<u16> =
                    (0..a_stride * h).map(|_| (rng.next() & 0xff) as u16).collect();
                let mut a8: Vec<u8> = (0..a_stride * h).map(|_| (rng.next() & 0xff) as u8).collect();
                let mut b8: Vec<u8> = (0..b_stride * h).map(|_| (rng.next() & 0xff) as u8).collect();
                if case == 1 {
                    // Max-diff boundary: a all-max, b all-zero.
                    a16.fill(255);
                    a8.fill(255);
                    b8.fill(0);
                }
                if case == 2 {
                    // Flat.
                    let v = (rng.next() & 0xff) as u8;
                    a16.fill(v as u16);
                    a8.fill(v);
                    b8.fill(v);
                }
                assert_eq!(
                    sse_u16_u8(&a16, a_stride, &b8, b_stride, w, h),
                    sse_u16_u8_scalar(&a16, a_stride, &b8, b_stride, w, h),
                    "sse_u16_u8 {w}x{h} case {case}"
                );
                assert_eq!(
                    sse(&a8, a_stride, &b8, b_stride, w, h),
                    sse_scalar(&a8, a_stride, &b8, b_stride, w, h),
                    "sse {w}x{h} case {case}"
                );
            }
        }
    });
    eprintln!("sse family SIMD parity: {report}");
    assert!(report.permutations_run >= 2);
    assert!(
        simd_perms >= 1,
        "no vector tier was ever live — the sweep compared scalar to itself"
    );
}
