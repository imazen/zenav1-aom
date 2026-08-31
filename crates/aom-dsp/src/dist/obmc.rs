//! OBMC distortion kernels — port of the `aom_obmc_variance*` /
//! `aom_obmc_sub_pixel_variance*` families (`aom_dsp/variance.c`), lowbd and
//! high bit depth.
//!
//! These are the metrics the OBMC (overlapped block motion compensation)
//! motion search scores candidates with. They differ from the plain variance
//! in that the "source" is a per-pixel **weighted** target
//! (`wsrc`) and each predictor sample is scaled by a per-pixel `mask` before
//! the difference, both at 1/4096 precision:
//!
//! ```text
//! diff = round_signed(wsrc[j] - pre[j] * mask[j], 12)
//! ```
//!
//! | Rust | C |
//! |---|---|
//! | [`obmc_variance`] | `aom_obmc_variance{W}x{H}_c` (variance.c:794 macro) |
//! | [`obmc_sub_pixel_variance`] | `aom_obmc_sub_pixel_variance{W}x{H}_c` (:803) |
//! | [`highbd_obmc_variance`] | `aom_highbd_{8,10,12}_obmc_variance{W}x{H}_c` (:937) |
//! | [`highbd_obmc_sub_pixel_variance`] | `aom_highbd_{8,10,12}_obmc_sub_pixel_variance{W}x{H}_c` (:966) |
//!
//! # The three high-bit-depth arms are not one function with a shift
//! C generates *three* differently-rounded variants per size. The bd-10 and
//! bd-12 arms round `sum` and `sse` down by different amounts (2/4 and 4/8),
//! and they clamp a negative variance to zero, which the bd-8 arm does not.
//! [`highbd_obmc_variance`] takes `bd` and selects among them; folding them
//! into a single shift would be wrong in both directions.
//!
//! Note the bd-10/12 rounding uses C's **unsigned-style** `ROUND_POWER_OF_TWO`
//! on a value that can be negative — an arithmetic shift of `sum + half` — not
//! `ROUND_POWER_OF_TWO_SIGNED`, which is what the per-pixel `diff` above uses.
//! The two macros differ on negative inputs and C really does use both, one
//! line apart.
//!
//! # Differential coverage
//! `tests/obmc_dist_diff.rs`, tier 1 against the real exported C.

use super::{BILINEAR_FILTERS_2T, FILTER_BITS};

/// `ROUND_POWER_OF_TWO(value, n)` — the unsigned-style macro, as an arithmetic
/// shift so a negative input behaves the way C's does on a signed type.
#[inline]
fn round_pow2_i64(value: i64, n: i32) -> i64 {
    (value + (1i64 << (n - 1))) >> n
}

/// `ROUND_POWER_OF_TWO_SIGNED(value, n)` — rounds the magnitude, restores the
/// sign.
#[inline]
fn round_pow2_signed_i32(value: i32, n: i32) -> i32 {
    if value < 0 {
        -(((-value) + (1 << (n - 1))) >> n)
    } else {
        (value + (1 << (n - 1))) >> n
    }
}

/// `obmc_variance` (variance.c:772): the raw `(sse, sum)` pair, lowbd.
///
/// `tsse` is a C `unsigned int` and `tsum` a C `int`, both of which wrap on
/// overflow; the port uses the same widths and wrapping so a pathological
/// `wsrc`/`mask` pair produces the same numbers rather than panicking.
fn obmc_variance_raw(
    pre: &[u8],
    pre_off: usize,
    pre_stride: usize,
    wsrc: &[i32],
    mask: &[i32],
    w: usize,
    h: usize,
) -> (u32, i32) {
    let mut tsse: u32 = 0;
    let mut tsum: i32 = 0;
    for i in 0..h {
        for j in 0..w {
            let diff = round_pow2_signed_i32(
                wsrc[i * w + j] - i32::from(pre[pre_off + i * pre_stride + j]) * mask[i * w + j],
                12,
            );
            tsum = tsum.wrapping_add(diff);
            tsse = tsse.wrapping_add((diff.wrapping_mul(diff)) as u32);
        }
    }
    (tsse, tsum)
}

/// `aom_obmc_variance{W}x{H}_c` (variance.c:794). Returns `(variance, sse)`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn obmc_variance(
    pre: &[u8],
    pre_off: usize,
    pre_stride: usize,
    wsrc: &[i32],
    mask: &[i32],
    w: usize,
    h: usize,
) -> (u32, u32) {
    let (sse, sum) = obmc_variance_raw(pre, pre_off, pre_stride, wsrc, mask, w, h);
    let n = (w * h) as i64;
    let var = sse.wrapping_sub(((i64::from(sum) * i64::from(sum)) / n) as u32);
    (var, sse)
}

/// `aom_obmc_sub_pixel_variance{W}x{H}_c` (variance.c:803): bilinear-interpolate
/// `pre` at the 1/8-pel phase `(xoffset, yoffset)`, then take the OBMC variance.
///
/// `pre` must carry one extra column and row past the block for the taps.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn obmc_sub_pixel_variance(
    pre: &[u8],
    pre_off: usize,
    pre_stride: usize,
    xoffset: usize,
    yoffset: usize,
    wsrc: &[i32],
    mask: &[i32],
    w: usize,
    h: usize,
) -> (u32, u32) {
    let fx = BILINEAR_FILTERS_2T[xoffset];
    let mut fdata3 = vec![0u16; (h + 1) * w];
    for i in 0..(h + 1) {
        for j in 0..w {
            let a0 = i32::from(pre[pre_off + i * pre_stride + j]);
            let a1 = i32::from(pre[pre_off + i * pre_stride + j + 1]);
            fdata3[i * w + j] = round_pow2_i64(
                i64::from(a0 * i32::from(fx[0]) + a1 * i32::from(fx[1])),
                FILTER_BITS,
            ) as u16;
        }
    }
    let fy = BILINEAR_FILTERS_2T[yoffset];
    let mut temp2 = vec![0u8; h * w];
    for i in 0..h {
        for j in 0..w {
            let v0 = i32::from(fdata3[i * w + j]);
            let v1 = i32::from(fdata3[(i + 1) * w + j]);
            temp2[i * w + j] = round_pow2_i64(
                i64::from(v0 * i32::from(fy[0]) + v1 * i32::from(fy[1])),
                FILTER_BITS,
            ) as u8;
        }
    }
    obmc_variance(&temp2, 0, w, wsrc, mask, w, h)
}

/// `highbd_obmc_variance64` (variance.c:880): the 64-bit `(sse, sum)` pair.
fn highbd_obmc_variance64(
    pre: &[u16],
    pre_off: usize,
    pre_stride: usize,
    wsrc: &[i32],
    mask: &[i32],
    w: usize,
    h: usize,
) -> (u64, i64) {
    let mut tsse: u64 = 0;
    let mut tsum: i64 = 0;
    for i in 0..h {
        for j in 0..w {
            let diff = round_pow2_signed_i32(
                wsrc[i * w + j] - i32::from(pre[pre_off + i * pre_stride + j]) * mask[i * w + j],
                12,
            );
            tsum += i64::from(diff);
            tsse += (i64::from(diff) * i64::from(diff)) as u64;
        }
    }
    (tsse, tsum)
}

/// `aom_highbd_{8,10,12}_obmc_variance{W}x{H}_c` (variance.c:937), selected by
/// `bd`. Returns `(variance, sse)`.
///
/// The bd-8 arm truncates the 64-bit accumulators to `int`/`unsigned` and does a
/// **wrapping** subtract; the bd-10 and bd-12 arms round first (by 2/4 and 4/8
/// respectively) and then clamp a negative variance to zero. Those are three
/// different functions in C, not one with a parameter.
///
/// The negative-variance clamp in the bd-10/12 arms is **defensive and, as far
/// as this port can tell, unreachable**: `sse >= sum^2 / n` holds exactly
/// before rounding, and the two roundings scale by the same factor (2 bits on
/// `sum`, 4 on `sse`, squared vs linear), so they cannot flip the sign. A
/// mutation deleting the clamp does NOT fail `tests/obmc_dist_diff.rs`, and
/// that is reported rather than papered over — the clamp is kept because C has
/// it, not because the differential pins it.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn highbd_obmc_variance(
    pre: &[u16],
    pre_off: usize,
    pre_stride: usize,
    wsrc: &[i32],
    mask: &[i32],
    w: usize,
    h: usize,
    bd: u32,
) -> (u32, u32) {
    let (sse64, sum64) = highbd_obmc_variance64(pre, pre_off, pre_stride, wsrc, mask, w, h);
    let n = (w * h) as i64;
    match bd {
        8 => {
            let sum = sum64 as i32;
            let sse = sse64 as u32;
            let var = sse.wrapping_sub(((i64::from(sum) * i64::from(sum)) / n) as u32);
            (var, sse)
        }
        10 => {
            let sum = round_pow2_i64(sum64, 2) as i32;
            let sse = round_pow2_i64(sse64 as i64, 4) as u32;
            let var = i64::from(sse) - ((i64::from(sum) * i64::from(sum)) / n);
            (if var >= 0 { var as u32 } else { 0 }, sse)
        }
        12 => {
            let sum = round_pow2_i64(sum64, 4) as i32;
            let sse = round_pow2_i64(sse64 as i64, 8) as u32;
            let var = i64::from(sse) - ((i64::from(sum) * i64::from(sum)) / n);
            (if var >= 0 { var as u32 } else { 0 }, sse)
        }
        _ => panic!("aom_dsp::dist::obmc: unsupported bit depth {bd} (8/10/12 only)"),
    }
}

/// `aom_highbd_{8,10,12}_obmc_sub_pixel_variance{W}x{H}_c` (variance.c:966).
///
/// The bilinear intermediate stays 16-bit through **both** passes here (the
/// lowbd twin narrows to `uint8_t` between them), which is the substantive
/// difference between the two bit depths in this kernel.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn highbd_obmc_sub_pixel_variance(
    pre: &[u16],
    pre_off: usize,
    pre_stride: usize,
    xoffset: usize,
    yoffset: usize,
    wsrc: &[i32],
    mask: &[i32],
    w: usize,
    h: usize,
    bd: u32,
) -> (u32, u32) {
    let fx = BILINEAR_FILTERS_2T[xoffset];
    let mut fdata3 = vec![0u16; (h + 1) * w];
    for i in 0..(h + 1) {
        for j in 0..w {
            let a0 = i32::from(pre[pre_off + i * pre_stride + j]);
            let a1 = i32::from(pre[pre_off + i * pre_stride + j + 1]);
            fdata3[i * w + j] = round_pow2_i64(
                i64::from(a0 * i32::from(fx[0]) + a1 * i32::from(fx[1])),
                FILTER_BITS,
            ) as u16;
        }
    }
    let fy = BILINEAR_FILTERS_2T[yoffset];
    let mut temp2 = vec![0u16; h * w];
    for i in 0..h {
        for j in 0..w {
            let v0 = i32::from(fdata3[i * w + j]);
            let v1 = i32::from(fdata3[(i + 1) * w + j]);
            temp2[i * w + j] = round_pow2_i64(
                i64::from(v0 * i32::from(fy[0]) + v1 * i32::from(fy[1])),
                FILTER_BITS,
            ) as u16;
        }
    }
    highbd_obmc_variance(&temp2, 0, w, wsrc, mask, w, h, bd)
}
