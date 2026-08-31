//! High-bit-depth single-reference motion-compensation convolution — port of
//! libaom v3.14.1's `av1_highbd_convolve_{x,y,2d}_sr_c`
//! (`av1/common/convolve.c:689/717/737`).
//!
//! These are the 10/12-bit twins of the parent module's lowbd
//! `convolve_{x,y,2d}_sr`. The port stores highbd planes as `u16`, so the
//! signatures take `&[u16]` on both sides.
//!
//! Three shape differences from the lowbd kernels, all faithful to C:
//!
//! * `av1_highbd_convolve_y_sr_c` takes **no `ConvolveParams`** — its rounding
//!   is the fixed `FILTER_BITS`, with no `round_0`/`round_1` to vary.
//! * The 2-D kernel's `offset_bits` and the horizontal pass's pre-bias both
//!   scale with `bd`, so the intermediate is *not* the lowbd one widened.
//! * The horizontal intermediate is still `int16_t`, so it truncates at 16 bits
//!   even at bd 12.
//!
//! # Differential coverage
//! `tests/compound_convolve_diff.rs`, tier 1 against the real exported C.

use super::{FILTER_BITS, rpo2};

#[inline]
fn clip_pixel_highbd(v: i32, bd: u32) -> u16 {
    v.clamp(0, (1i32 << bd) - 1) as u16
}

/// `av1_highbd_convolve_x_sr_c` (convolve.c:689).
///
/// `src_off` is the interior origin; the reference needs `taps/2 - 1` samples
/// of border to the left and `taps/2` to the right. `x_filter` is the
/// already-subpel-selected kernel row.
#[allow(clippy::too_many_arguments)]
pub fn highbd_convolve_x_sr(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[i16],
    round_0: i32,
    bd: u32,
) {
    let fo_horiz = x_filter.len() / 2 - 1;
    let bits = FILTER_BITS - round_0;
    for y in 0..h {
        for x in 0..w {
            let base =
                src_off as isize + (y * src_stride) as isize + x as isize - fo_horiz as isize;
            let mut res = 0i32;
            for (k, f) in x_filter.iter().enumerate() {
                res += i32::from(*f) * i32::from(src[(base + k as isize) as usize]);
            }
            res = rpo2(res, round_0);
            dst[y * dst_stride + x] = clip_pixel_highbd(rpo2(res, bits), bd);
        }
    }
}

/// `av1_highbd_convolve_y_sr_c` (convolve.c:717).
///
/// Takes no `ConvolveParams` in C: the rounding is exactly `FILTER_BITS`.
#[allow(clippy::too_many_arguments)]
pub fn highbd_convolve_y_sr(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    y_filter: &[i16],
    bd: u32,
) {
    let fo_vert = y_filter.len() / 2 - 1;
    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize
                + ((y as isize - fo_vert as isize) * src_stride as isize)
                + x as isize;
            let mut res = 0i32;
            for (k, f) in y_filter.iter().enumerate() {
                res += i32::from(*f) * i32::from(src[(base + (k * src_stride) as isize) as usize]);
            }
            dst[y * dst_stride + x] = clip_pixel_highbd(rpo2(res, FILTER_BITS), bd);
        }
    }
}

/// `av1_highbd_convolve_2d_sr_c` (convolve.c:737).
#[allow(clippy::too_many_arguments)]
pub fn highbd_convolve_2d_sr(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[i16],
    y_filter: &[i16],
    round_0: i32,
    round_1: i32,
    bd: u32,
) {
    let taps_x = x_filter.len();
    let taps_y = y_filter.len();
    let im_h = h + taps_y - 1;
    let im_stride = w;
    let fo_vert = taps_y / 2 - 1;
    let fo_horiz = taps_x / 2 - 1;
    let bits = FILTER_BITS * 2 - round_0 - round_1;

    let mut im = vec![0i16; im_h * im_stride];
    let src_horiz = src_off as isize - (fo_vert * src_stride) as isize;
    for y in 0..im_h {
        for x in 0..w {
            let base = src_horiz + (y * src_stride) as isize + x as isize - fo_horiz as isize;
            let mut sum = 1i32 << (bd as i32 + FILTER_BITS - 1);
            for (k, f) in x_filter.iter().enumerate() {
                sum += i32::from(*f) * i32::from(src[(base + k as isize) as usize]);
            }
            im[y * im_stride + x] = rpo2(sum, round_0) as i16;
        }
    }

    let offset_bits = bd as i32 + 2 * FILTER_BITS - round_0;
    let round_offset = (1i32 << (offset_bits - round_1)) + (1i32 << (offset_bits - round_1 - 1));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 1i32 << offset_bits;
            for (k, f) in y_filter.iter().enumerate() {
                sum += i32::from(*f) * i32::from(im[(y + k) * im_stride + x]);
            }
            let res = rpo2(sum, round_1) - round_offset;
            dst[y * dst_stride + x] = clip_pixel_highbd(rpo2(res, bits), bd);
        }
    }
}
