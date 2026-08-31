//! Compound (two-reference) motion-compensation convolution — port of libaom
//! v3.14.1's `av1_dist_wtd_convolve_*` family (`av1/common/convolve.c`).
//!
//! These are the kernels the **second** reference of a compound block goes
//! through. The distinction from the single-reference `av1_convolve_*_sr_c`
//! kernels in the parent module is the `dst16` intermediate buffer:
//!
//! * `do_average == false` (the first reference) writes the unrounded 16-bit
//!   intermediate to `dst16` and does **not** touch `dst`;
//! * `do_average == true` (the second reference) reads that intermediate back,
//!   combines it with this reference's, and writes the final pixel to `dst`.
//!
//! The combine is either a plain average (`use_dist_wtd_comp_avg == false`) or
//! the distance-weighted `(fwd*a + bck*b) >> DIST_PRECISION_BITS` whose two
//! offsets come from
//! [`crate::inter::compound::dist_wtd_comp_weight_assign`].
//!
//! | Rust | C |
//! |---|---|
//! | [`dist_wtd_convolve_2d`] | `av1_dist_wtd_convolve_2d_c` (convolve.c:293) |
//! | [`dist_wtd_convolve_y`] | `av1_dist_wtd_convolve_y_c` (:361) |
//! | [`dist_wtd_convolve_x`] | `av1_dist_wtd_convolve_x_c` (:408) |
//! | [`dist_wtd_convolve_2d_copy`] | `av1_dist_wtd_convolve_2d_copy_c` (:455) |
//! | [`highbd_dist_wtd_convolve_2d`] | `av1_highbd_dist_wtd_convolve_2d_c` (:906) |
//! | [`highbd_dist_wtd_convolve_x`] | `av1_highbd_dist_wtd_convolve_x_c` (:975) |
//! | [`highbd_dist_wtd_convolve_y`] | `av1_highbd_dist_wtd_convolve_y_c` (:1023) |
//! | [`highbd_dist_wtd_convolve_2d_copy`] | `av1_highbd_dist_wtd_convolve_2d_copy_c` (:1071) |
//!
//! # Two traps this port reproduces deliberately
//!
//! 1. **`bits` is not symmetric between the x and y kernels.** The x kernels use
//!    `FILTER_BITS - round_1` and the y kernels use `FILTER_BITS - round_0`.
//!    That is what libaom does, in both bit depths; it is not a typo to
//!    "correct" here.
//! 2. **`CONV_BUF_TYPE` is `uint16_t`, and the truncation to it is
//!    load-bearing.** In the 2-D kernels the vertical result is narrowed to 16
//!    bits *before* it is fed to the distance-weighted product, and in the
//!    `_2d_copy` kernels both the shift and the `+= round_offset` happen in 16
//!    bits. The port narrows at exactly those points.
//!
//! # Differential coverage
//! `tests/compound_convolve_diff.rs`, tier 1 against the real exported C.

use super::{FILTER_BITS, clip_pixel, rpo2};

/// `DIST_PRECISION_BITS` (`av1/common/enums.h:76`).
const DIST_PRECISION_BITS: i32 = 4;

/// The subset of libaom's `ConvolveParams` the compound kernels read. `dst` /
/// `dst_stride` (the 16-bit intermediate) are passed alongside rather than
/// inside, because Rust cannot hold a `&mut` to it in a `Copy` params struct.
#[derive(Clone, Copy, Debug)]
pub struct CompoundConvolveParams {
    /// `conv_params->round_0`
    pub round_0: i32,
    /// `conv_params->round_1`
    pub round_1: i32,
    /// `conv_params->do_average` — false for the first reference (write
    /// `dst16`), true for the second (combine and write `dst`).
    pub do_average: bool,
    /// `conv_params->use_dist_wtd_comp_avg`
    pub use_dist_wtd_comp_avg: bool,
    /// `conv_params->fwd_offset`
    pub fwd_offset: i32,
    /// `conv_params->bck_offset`
    pub bck_offset: i32,
}

#[inline]
fn clip_pixel_highbd(v: i32, bd: u32) -> u16 {
    v.clamp(0, (1i32 << bd) - 1) as u16
}

/// The shared tail of every compound kernel: given this reference's 16-bit
/// intermediate `res`, either store it (first reference) or combine it with the
/// stored one and return the rounded, offset-corrected pixel value.
///
/// `res` is passed as `i32` because the x/y kernels keep it at full width
/// through the combine and only narrow when storing; the 2-D kernels narrow
/// first and pass the narrowed value in.
#[inline]
#[allow(clippy::too_many_arguments)]
fn combine(
    cp: &CompoundConvolveParams,
    dst16: &mut [u16],
    dst16_idx: usize,
    res: i32,
    round_offset: i32,
    round_bits: i32,
) -> Option<i32> {
    if !cp.do_average {
        dst16[dst16_idx] = res as u16;
        return None;
    }
    let mut tmp = i32::from(dst16[dst16_idx]);
    if cp.use_dist_wtd_comp_avg {
        tmp = tmp * cp.fwd_offset + res * cp.bck_offset;
        tmp >>= DIST_PRECISION_BITS;
    } else {
        tmp += res;
        tmp >>= 1;
    }
    tmp -= round_offset;
    Some(rpo2(tmp, round_bits))
}

// ===================================================================
// lowbd (bd = 8)
// ===================================================================

/// `av1_dist_wtd_convolve_2d_c` (convolve.c:293).
///
/// `src_off` is the interior origin; the reference must carry `fo_vert` rows of
/// border above (`taps_y/2 - 1`) and `fo_horiz` columns to the left, plus the
/// matching trailing border. `x_filter` / `y_filter` are the already-subpel-
/// selected kernel rows, `taps` long each.
#[allow(clippy::too_many_arguments)]
pub fn dist_wtd_convolve_2d(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[i16],
    y_filter: &[i16],
    cp: &CompoundConvolveParams,
) {
    const BD: i32 = 8;
    let taps_x = x_filter.len();
    let taps_y = y_filter.len();
    let im_h = h + taps_y - 1;
    let im_stride = w;
    let fo_vert = taps_y / 2 - 1;
    let fo_horiz = taps_x / 2 - 1;
    let round_bits = 2 * FILTER_BITS - cp.round_0 - cp.round_1;

    // Horizontal pass into the int16 intermediate.
    let mut im = vec![0i16; im_h * im_stride];
    let src_horiz = src_off as isize - (fo_vert * src_stride) as isize;
    for y in 0..im_h {
        for x in 0..w {
            let base = src_horiz + (y * src_stride) as isize + x as isize - fo_horiz as isize;
            let mut sum = 1i32 << (BD + FILTER_BITS - 1);
            for (k, f) in x_filter.iter().enumerate() {
                sum += i32::from(*f) * i32::from(src[(base + k as isize) as usize]);
            }
            im[y * im_stride + x] = rpo2(sum, cp.round_0) as i16;
        }
    }

    // Vertical pass.
    let offset_bits = BD + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 1i32 << offset_bits;
            for (k, f) in y_filter.iter().enumerate() {
                sum += i32::from(*f) * i32::from(im[(y + k) * im_stride + x]);
            }
            // CONV_BUF_TYPE res — narrowed to 16 bits BEFORE the combine.
            let res = i32::from(rpo2(sum, cp.round_1) as u16);
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                res,
                round_offset,
                round_bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel(px);
            }
        }
    }
}

/// `av1_dist_wtd_convolve_y_c` (convolve.c:361). Note `bits = FILTER_BITS -
/// round_0` here, against `round_1` in the x sibling — see the module note.
#[allow(clippy::too_many_arguments)]
pub fn dist_wtd_convolve_y(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    y_filter: &[i16],
    cp: &CompoundConvolveParams,
) {
    const BD: i32 = 8;
    let fo_vert = y_filter.len() / 2 - 1;
    let bits = FILTER_BITS - cp.round_0;
    let offset_bits = BD + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    let round_bits = 2 * FILTER_BITS - cp.round_0 - cp.round_1;

    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize
                + ((y as isize - fo_vert as isize) * src_stride as isize)
                + x as isize;
            let mut res = 0i32;
            for (k, f) in y_filter.iter().enumerate() {
                res += i32::from(*f) * i32::from(src[(base + (k * src_stride) as isize) as usize]);
            }
            res *= 1 << bits;
            res = rpo2(res, cp.round_1) + round_offset;
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                res,
                round_offset,
                round_bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel(px);
            }
        }
    }
}

/// `av1_dist_wtd_convolve_x_c` (convolve.c:408). Note `bits = FILTER_BITS -
/// round_1` here — see the module note.
#[allow(clippy::too_many_arguments)]
pub fn dist_wtd_convolve_x(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[i16],
    cp: &CompoundConvolveParams,
) {
    const BD: i32 = 8;
    let fo_horiz = x_filter.len() / 2 - 1;
    let bits = FILTER_BITS - cp.round_1;
    let offset_bits = BD + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    let round_bits = 2 * FILTER_BITS - cp.round_0 - cp.round_1;

    for y in 0..h {
        for x in 0..w {
            let base =
                src_off as isize + (y * src_stride) as isize + x as isize - fo_horiz as isize;
            let mut res = 0i32;
            for (k, f) in x_filter.iter().enumerate() {
                res += i32::from(*f) * i32::from(src[(base + k as isize) as usize]);
            }
            res = (1 << bits) * rpo2(res, cp.round_0);
            res += round_offset;
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                res,
                round_offset,
                round_bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel(px);
            }
        }
    }
}

/// `av1_dist_wtd_convolve_2d_copy_c` (convolve.c:455) — the full-pel compound
/// case. Both the shift and the `+= round_offset` happen in `uint16_t`, and the
/// final rounding uses `bits`, not `round_bits`.
#[allow(clippy::too_many_arguments)]
pub fn dist_wtd_convolve_2d_copy(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    cp: &CompoundConvolveParams,
) {
    const BD: i32 = 8;
    let bits = FILTER_BITS * 2 - cp.round_1 - cp.round_0;
    let offset_bits = BD + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));

    for y in 0..h {
        for x in 0..w {
            let s = i32::from(src[src_off + y * src_stride + x]);
            // CONV_BUF_TYPE res: both steps are 16-bit.
            let res = (s << bits) as u16;
            let res = res.wrapping_add(round_offset as u16);
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                i32::from(res),
                round_offset,
                bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel(px);
            }
        }
    }
}

// ===================================================================
// highbd (bd 10 / 12; the C is also correct at bd 8)
// ===================================================================

/// `av1_highbd_dist_wtd_convolve_2d_c` (convolve.c:906).
#[allow(clippy::too_many_arguments)]
pub fn highbd_dist_wtd_convolve_2d(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[i16],
    y_filter: &[i16],
    cp: &CompoundConvolveParams,
    bd: u32,
) {
    let taps_x = x_filter.len();
    let taps_y = y_filter.len();
    let im_h = h + taps_y - 1;
    let im_stride = w;
    let fo_vert = taps_y / 2 - 1;
    let fo_horiz = taps_x / 2 - 1;
    let round_bits = 2 * FILTER_BITS - cp.round_0 - cp.round_1;

    let mut im = vec![0i16; im_h * im_stride];
    let src_horiz = src_off as isize - (fo_vert * src_stride) as isize;
    for y in 0..im_h {
        for x in 0..w {
            let base = src_horiz + (y * src_stride) as isize + x as isize - fo_horiz as isize;
            let mut sum = 1i32 << (bd as i32 + FILTER_BITS - 1);
            for (k, f) in x_filter.iter().enumerate() {
                sum += i32::from(*f) * i32::from(src[(base + k as isize) as usize]);
            }
            im[y * im_stride + x] = rpo2(sum, cp.round_0) as i16;
        }
    }

    let offset_bits = bd as i32 + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 1i32 << offset_bits;
            for (k, f) in y_filter.iter().enumerate() {
                sum += i32::from(*f) * i32::from(im[(y + k) * im_stride + x]);
            }
            let res = i32::from(rpo2(sum, cp.round_1) as u16);
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                res,
                round_offset,
                round_bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel_highbd(px, bd);
            }
        }
    }
}

/// `av1_highbd_dist_wtd_convolve_x_c` (convolve.c:975). `bits = FILTER_BITS -
/// round_1`.
#[allow(clippy::too_many_arguments)]
pub fn highbd_dist_wtd_convolve_x(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[i16],
    cp: &CompoundConvolveParams,
    bd: u32,
) {
    let fo_horiz = x_filter.len() / 2 - 1;
    let bits = FILTER_BITS - cp.round_1;
    let offset_bits = bd as i32 + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    let round_bits = 2 * FILTER_BITS - cp.round_0 - cp.round_1;

    for y in 0..h {
        for x in 0..w {
            let base =
                src_off as isize + (y * src_stride) as isize + x as isize - fo_horiz as isize;
            let mut res = 0i32;
            for (k, f) in x_filter.iter().enumerate() {
                res += i32::from(*f) * i32::from(src[(base + k as isize) as usize]);
            }
            res = (1 << bits) * rpo2(res, cp.round_0);
            res += round_offset;
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                res,
                round_offset,
                round_bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel_highbd(px, bd);
            }
        }
    }
}

/// `av1_highbd_dist_wtd_convolve_y_c` (convolve.c:1023). `bits = FILTER_BITS -
/// round_0`.
#[allow(clippy::too_many_arguments)]
pub fn highbd_dist_wtd_convolve_y(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    y_filter: &[i16],
    cp: &CompoundConvolveParams,
    bd: u32,
) {
    let fo_vert = y_filter.len() / 2 - 1;
    let bits = FILTER_BITS - cp.round_0;
    let offset_bits = bd as i32 + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    let round_bits = 2 * FILTER_BITS - cp.round_0 - cp.round_1;

    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize
                + ((y as isize - fo_vert as isize) * src_stride as isize)
                + x as isize;
            let mut res = 0i32;
            for (k, f) in y_filter.iter().enumerate() {
                res += i32::from(*f) * i32::from(src[(base + (k * src_stride) as isize) as usize]);
            }
            res *= 1 << bits;
            res = rpo2(res, cp.round_1) + round_offset;
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                res,
                round_offset,
                round_bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel_highbd(px, bd);
            }
        }
    }
}

/// `av1_highbd_dist_wtd_convolve_2d_copy_c` (convolve.c:1071).
#[allow(clippy::too_many_arguments)]
pub fn highbd_dist_wtd_convolve_2d_copy(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    cp: &CompoundConvolveParams,
    bd: u32,
) {
    let bits = FILTER_BITS * 2 - cp.round_1 - cp.round_0;
    let offset_bits = bd as i32 + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));

    for y in 0..h {
        for x in 0..w {
            let s = i32::from(src[src_off + y * src_stride + x]);
            let res = (s << bits) as u16;
            let res = res.wrapping_add(round_offset as u16);
            if let Some(px) = combine(
                cp,
                dst16,
                y * dst16_stride + x,
                i32::from(res),
                round_offset,
                bits,
            ) {
                dst[y * dst_stride + x] = clip_pixel_highbd(px, bd);
            }
        }
    }
}
