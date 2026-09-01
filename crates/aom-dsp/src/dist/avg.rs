//! Block averages and min/max spreads — the `aom_dsp/avg.c` kernels the
//! variance-based partitioner's INTER leaf fill is built on.
//!
//! | Rust | C |
//! |---|---|
//! | [`avg_4x4`] | `aom_avg_4x4_c` (avg.c:32) |
//! | [`avg_8x8`] | `aom_avg_8x8_c` (avg.c:42) |
//! | [`avg_8x8_quad`] | `aom_avg_8x8_quad_c` (avg.c:52) |
//! | [`minmax_8x8`] | `aom_minmax_8x8_c` (avg.c:18) |
//! | [`highbd_avg_4x4`] | `aom_highbd_avg_4x4_c` (avg.c:74) |
//! | [`highbd_avg_8x8`] | `aom_highbd_avg_8x8_c` (avg.c:63) |
//! | [`highbd_minmax_8x8`] | `aom_highbd_minmax_8x8_c` (avg.c:85) |
//!
//! The averages are rounded, not truncated: `(sum + half) >> log2(n)`. The
//! rounding constant is part of the contract and differs by block size.
//!
//! # The two min/max seeds are NOT the same
//! `aom_minmax_8x8_c` seeds `*min = 255` and `aom_highbd_minmax_8x8_c` seeds
//! `*min = 65535`. Both are "larger than any diff this arm can see", so the
//! seed is only observable if a caller could pass a zero-iteration window —
//! neither can, both loops are a fixed 8x8. The values are still reproduced as
//! written rather than unified, because unifying them would make the lowbd arm
//! silently wrong if it were ever handed 16-bit data.
//!
//! # Differential coverage
//! `tests/avg_diff.rs`, tier 1 against the real exported C.

/// `aom_avg_4x4_c` (avg.c:32) — mean of a 4x4 window, rounded.
#[must_use]
pub fn avg_4x4<P: Into<u32> + Copy>(src: &[P], stride: usize) -> u32 {
    let sum: u32 = (0..4)
        .flat_map(|i| src[i * stride..][..4].iter())
        .map(|&p| p.into())
        .sum();
    (sum + 8) >> 4
}

/// `aom_avg_8x8_c` (avg.c:42) — mean of an 8x8 window, rounded.
#[must_use]
pub fn avg_8x8<P: Into<u32> + Copy>(src: &[P], stride: usize) -> u32 {
    let sum: u32 = (0..8)
        .flat_map(|i| src[i * stride..][..8].iter())
        .map(|&p| p.into())
        .sum();
    (sum + 32) >> 6
}

/// `aom_highbd_avg_4x4_c` (avg.c:74) — the 10/12-bit arm of [`avg_4x4`].
///
/// C reaches the same arithmetic through `CONVERT_TO_SHORTPTR`; here the
/// element type carries it, so the two share a body.
#[must_use]
pub fn highbd_avg_4x4(src: &[u16], stride: usize) -> u32 {
    avg_4x4(src, stride)
}

/// `aom_highbd_avg_8x8_c` (avg.c:63) — the 10/12-bit arm of [`avg_8x8`].
#[must_use]
pub fn highbd_avg_8x8(src: &[u16], stride: usize) -> u32 {
    avg_8x8(src, stride)
}

/// `aom_avg_8x8_quad_c` (avg.c:52) — the four 8x8 averages of one 16x16 block.
///
/// C indexes `s` by the ABSOLUTE `(x16_idx, y16_idx)` rather than taking a
/// pre-offset pointer, so the port keeps that shape: `src` is the plane and
/// the two indices are pixel coordinates in it.
///
/// The sub-block order is raster within the 16x16: `k` splits as
/// `x8 = (k & 1) << 3`, `y8 = (k >> 1) << 3`.
#[must_use]
pub fn avg_8x8_quad<P: Into<u32> + Copy>(
    src: &[P],
    stride: usize,
    x16_idx: usize,
    y16_idx: usize,
) -> [u32; 4] {
    let mut avg = [0u32; 4];
    for (k, out) in avg.iter_mut().enumerate() {
        let x8_idx = x16_idx + ((k & 1) << 3);
        let y8_idx = y16_idx + ((k >> 1) << 3);
        *out = avg_8x8(&src[y8_idx * stride + x8_idx..], stride);
    }
    avg
}

/// `aom_minmax_8x8_c` (avg.c:18) — the smallest and largest absolute
/// difference over an 8x8 window, as `(min, max)`.
#[must_use]
pub fn minmax_8x8(s: &[u8], s_stride: usize, d: &[u8], d_stride: usize) -> (i32, i32) {
    // C seeds min at 255, the largest 8-bit difference.
    let mut min = 255i32;
    let mut max = 0i32;
    for i in 0..8 {
        let srow = &s[i * s_stride..][..8];
        let drow = &d[i * d_stride..][..8];
        for (&sv, &dv) in srow.iter().zip(drow) {
            let diff = i32::from(sv.abs_diff(dv));
            min = min.min(diff);
            max = max.max(diff);
        }
    }
    (min, max)
}

/// `aom_highbd_minmax_8x8_c` (avg.c:85) — the 10/12-bit arm of [`minmax_8x8`].
///
/// Note the different `min` seed: 65535, not 255. See the module header.
#[must_use]
pub fn highbd_minmax_8x8(s: &[u16], s_stride: usize, d: &[u16], d_stride: usize) -> (i32, i32) {
    let mut min = 65535i32;
    let mut max = 0i32;
    for i in 0..8 {
        let srow = &s[i * s_stride..][..8];
        let drow = &d[i * d_stride..][..8];
        for (&sv, &dv) in srow.iter().zip(drow) {
            let diff = i32::from(sv.abs_diff(dv));
            min = min.min(diff);
            max = max.max(diff);
        }
    }
    (min, max)
}
