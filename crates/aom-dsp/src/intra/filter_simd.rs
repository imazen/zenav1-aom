//! Vector filter-intra prediction — the port's counterpart to libaom's
//! `av1_filter_intra_predictor_sse4_1`.
//!
//! **Why this exists.** `benchmarks/encoder_simd_lane_width_audit_2026-09-09.md`
//! compared every operation where port and libaom are both vectorised and found
//! filter-intra to be one of only two where the port was still SCALAR against a
//! real C SIMD kernel: **31.1 ms against `av1_filter_intra_predictor_sse4_1`'s
//! 7.3 ms, 4.26x**, at the preset zenavif ships (`--cpu-used 3`). KB-PERF-11
//! removed that function's 2178-byte scratch memset and said in as many words
//! that *"the taps themselves are still scalar — 8 outputs x 7 taps per 4x2
//! block is the natural 8-lane dot product libaom's SSE4.1 vectorises — its own
//! landing."* This is that landing.
//!
//! **The shape is exact, with no idle lanes.** One 4x2 sub-block produces
//! EIGHT outputs and the vector is EIGHT i32 lanes, so lane `k` is output `k`.
//! Each output is a 7-tap dot product over the SAME seven neighbour samples,
//! so the neighbours splat and the taps are per-lane constants:
//!
//! ```text
//! acc = sum_j  splat(p[j]) * TAPS_T[mode][j]      (j = 0..6)
//! ```
//!
//! 56 scalar multiplies and 56 adds become **7 vector multiply-adds**.
//!
//! **Bit-exact by construction, not by a bound.** Every lane runs the identical
//! `i32` expression the scalar loop runs — same products, same order of
//! accumulation (`j` ascending), same `(acc + 8) >> 4` arithmetic shift, same
//! `clamp(0, max_v)`. Nothing is reassociated and no value is narrowed, so
//! there is no range precondition and no runtime gate: the kernel is correct at
//! every bit depth. `TAPS_T` is a compile-time transpose of the same
//! `FILTER_INTRA_TAPS` the scalar path reads, checked against it by
//! `taps_transpose_matches_the_scalar_table`.

use archmage::prelude::*;

use super::{FILTER_INTRA_TAPS, TX_H, TX_W};

/// `FILTER_INTRA_TAPS` transposed to `[mode][tap][output]`, so tap `j` is one
/// contiguous 8-lane vector. Built by a `const fn` from the scalar table, so
/// the two cannot drift.
const fn transpose_taps() -> [[[i32; 8]; 7]; 5] {
    let mut out = [[[0i32; 8]; 7]; 5];
    let mut m = 0;
    while m < 5 {
        let mut j = 0;
        while j < 7 {
            let mut k = 0;
            while k < 8 {
                out[m][j][k] = FILTER_INTRA_TAPS[m][k][j] as i32;
                k += 1;
            }
            j += 1;
        }
        m += 1;
    }
    out
}
const TAPS_T: [[[i32; 8]; 7]; 5] = transpose_taps();

/// The `incant!` fallback: decline, which routes the caller back to its own
/// scalar loop — the same `false` the pre-SIMD code took, and the differential's
/// reference.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_intra_predict_high_impl_scalar(
    _t: archmage::ScalarToken,
    _dst: &mut [u16],
    _dst_stride: usize,
    _tx_size: usize,
    _above: &[u16],
    _left: &[u16],
    _mode: usize,
    _bd: i32,
) -> bool {
    false
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_intra_predict_high_impl(
    t: Token,
    dst: &mut [u16],
    dst_stride: usize,
    tx_size: usize,
    above: &[u16],
    left: &[u16],
    mode: usize,
    bd: i32,
) -> bool {
    let (bw, bh) = (TX_W[tx_size], TX_H[tx_size]);
    debug_assert!(bw <= 32 && bh <= 32);

    let zero = i32x8::zero(t);
    let maxv = i32x8::splat(t, (1i32 << bd) - 1);
    let eight = i32x8::splat(t, 8);
    // The seven tap vectors for this mode, hoisted out of BOTH loops: `mode` is
    // fixed for the whole predict, so this is 7 loads per call rather than per
    // 4x2 block.
    let tv: [i32x8; 7] = core::array::from_fn(|j| i32x8::from_slice(t, &TAPS_T[mode][j]));

    // Three rows, exactly as the scalar path (KB-PERF-11): the recursion reads
    // only `r-1`, `r`, `r+1`.
    let mut prev = [0u16; 33];
    let mut row0 = [0u16; 33];
    let mut row1 = [0u16; 33];
    prev[..bw + 1].copy_from_slice(&above[..bw + 1]);

    let mut r = 1;
    while r < bh + 1 {
        row0[0] = left[r - 1];
        row1[0] = left[r];
        let mut c = 1;
        while c < bw + 1 {
            let p = [
                prev[c - 1] as i32,
                prev[c] as i32,
                prev[c + 1] as i32,
                prev[c + 2] as i32,
                prev[c + 3] as i32,
                row0[c - 1] as i32,
                row1[c - 1] as i32,
            ];
            // lane k = output k; ascending j is the scalar loop's own order.
            let mut acc = i32x8::splat(t, p[0]) * tv[0];
            for j in 1..7 {
                acc = acc + i32x8::splat(t, p[j]) * tv[j];
            }
            let v = ((acc + eight).shr_arithmetic_const::<4>()).max(zero).min(maxv).to_array();
            for k in 0..4 {
                row0[c + k] = v[k] as u16;
                row1[c + k] = v[k + 4] as u16;
            }
            c += 4;
        }
        dst[(r - 1) * dst_stride..(r - 1) * dst_stride + bw].copy_from_slice(&row0[1..bw + 1]);
        dst[r * dst_stride..r * dst_stride + bw].copy_from_slice(&row1[1..bw + 1]);
        prev = row1;
        r += 2;
    }
    true
}

/// Dispatch once per predict call (not per 4x2 block); `false` routes the
/// caller to its scalar loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_filter_intra_predict_high(
    dst: &mut [u16],
    dst_stride: usize,
    tx_size: usize,
    above: &[u16],
    left: &[u16],
    mode: usize,
    bd: i32,
) -> bool {
    let _ = crate::dispatch::scalar_forced();
    incant!(
        filter_intra_predict_high_impl(dst, dst_stride, tx_size, above, left, mode, bd),
        [v3, neon, wasm128, scalar]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_transpose_matches_the_scalar_table() {
        for m in 0..5 {
            for k in 0..8 {
                for j in 0..7 {
                    assert_eq!(TAPS_T[m][j][k], FILTER_INTRA_TAPS[m][k][j] as i32, "m{m} k{k} j{j}");
                }
            }
        }
    }
}
