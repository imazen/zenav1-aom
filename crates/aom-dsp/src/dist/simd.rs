//! Safe SIMD SAD via archmage `#[autoversion]` — no `unsafe`, no raw
//! `core::arch` intrinsics. `#[autoversion]` compiles one `#[target_feature]`-
//! gated variant per tier (AVX-512/AVX2/NEON/WASM/scalar) — unlocking LLVM's
//! auto-vectorizer to lower the sum-of-abs-diff loop to `psadbw` / `uabd` — plus
//! a runtime dispatcher. Result is byte-identical to scalar [`crate::dist::sad`].
//!
//! NOTE on dispatch: the generated `sad_simd` dispatcher pays a small
//! feature-check per call. In an encoder that is amortized by placing the
//! dispatch at the motion-search-loop entry (an `#[arcane]` boundary that calls
//! the SAD kernel per candidate); that entry point does not exist in this
//! kernel-only crate yet, so a per-block microbenchmark of the dispatcher is
//! dispatch-bound, not kernel-bound.

use archmage::autoversion;

/// Sum of absolute differences over a `w x h` block. Byte-identical to
/// [`crate::dist::sad`]; auto-vectorized, picks the best SIMD tier at runtime.
#[autoversion]
pub fn sad_simd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let arow = &a[y * a_stride..y * a_stride + w];
        let brow = &b[y * b_stride..y * b_stride + w];
        for x in 0..w {
            sum += (arow[x] as i32 - brow[x] as i32).unsigned_abs();
        }
    }
    sum
}

/// `av1_block_error_c` (`av1/encoder/rdopt.c`) via `#[autoversion]` — the
/// transform-domain distortion at bd8, and one of the encoder's hottest loops
/// (`dist_block_tx_domain` measured **61 ms against libaom's
/// `av1_block_error_avx2` at 8.4 ms, 7.3x**, in
/// `benchmarks/encoder_x86_reprofile_1024_s3_2026-09-10.md`).
///
/// The scalar body was already bounds-check-free but stayed scalar: a 32-bit
/// `imul`, a sign-extend and a 64-bit add, two-way unrolled. `#[autoversion]`
/// compiles it once per SIMD tier with the target features enabled, which is
/// what lets LLVM lower it to 8-lane `pmulld` plus widening accumulation —
/// exactly libaom's AVX2 shape, with no raw intrinsics and no `unsafe`.
///
/// # Bit-exactness
///
/// Identical to [`crate::dist::block_error`] by construction. The per-element
/// arithmetic is unchanged (`wrapping_sub`, `wrapping_mul` in i32 — the wrap is
/// load-bearing and matches C's `int` multiply, see KB-ARM-FLOAT root #3 — then
/// sign-extended to i64). Vectorizing REASSOCIATES the two sums, and that is
/// exact here rather than merely close: i64 addition is associative, and the
/// accumulators cannot overflow (|diff * diff| <= 2^31 over at most 4096
/// coefficients is under 2^43).
///
/// `dqcoeff` is sliced to `coeff.len()` so a short `dqcoeff` panics exactly
/// where the indexed form panicked.
#[autoversion]
pub fn block_error_simd(coeff: &[i32], dqcoeff: &[i32]) -> (i64, i64) {
    let n = coeff.len();
    let dq = &dqcoeff[..n];
    let mut error = 0i64;
    let mut sqcoeff = 0i64;
    for (&c, &d) in coeff.iter().zip(dq.iter()) {
        let diff = c.wrapping_sub(d);
        error += diff.wrapping_mul(diff) as i64;
        sqcoeff += c.wrapping_mul(c) as i64;
    }
    (error, sqcoeff)
}
