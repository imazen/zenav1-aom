//! Scaled-reference motion-compensation convolution — port of
//! `av1_convolve_2d_scale_c` and `av1_highbd_convolve_2d_scale_c`
//! (`av1/common/convolve.c`).
//!
//! AV1 lets a reference frame differ in size from the frame being coded
//! (`av1_is_scaled`, see [`crate::inter::scale`]). When it does, the predictor
//! is not built by the fixed-phase `convolve_*_sr` kernels: the sub-pel phase
//! **advances per output sample** by `x_step_qn` / `y_step_qn`, so every column
//! and every row can use a different filter row, and the source position walks
//! at a non-unit rate.
//!
//! | Rust | C |
//! |---|---|
//! | [`convolve_2d_scale`] | `av1_convolve_2d_scale_c` (convolve.c:494) |
//! | [`highbd_convolve_2d_scale`] | `av1_highbd_convolve_2d_scale_c` |
//!
//! Both handle the single-reference and the compound arms, since the scaled
//! kernel is the one entry point for both (unlike the unscaled path, which has
//! separate `_sr` and `dist_wtd_` families).
//!
//! # The vertical pass walks columns, not rows
//! C's vertical loop is `for x { for y { ... } src_vert++; }` — the OUTER
//! variable is the column, and the intermediate pointer advances by one sample
//! per column. Transposing it to row-major changes nothing numerically but does
//! change which `y_qn` accumulator each output sees, because `y_qn` is reset
//! per column. This port keeps C's loop order for that reason.
//!
//! # Differential coverage
//! `tests/convolve_scale_diff.rs`, tier 1 against the real exported C.
//!
//! **Known gap, stated rather than hidden:** C stores the vertical result in a
//! `CONV_BUF_TYPE` (`uint16_t`) before the compound combine, and this port
//! narrows at the same point — but the narrowing is NOT distinguished by the
//! differential. Removing it leaves the tests green, because at the
//! `(round_0, round_1)` pairs `get_conv_params_no_round` actually produces the
//! value stays inside 16 bits. The narrowing is kept because C has it, not
//! because the differential pins it.

use super::{FILTER_BITS, clip_pixel, rpo2};

/// `SCALE_SUBPEL_BITS` (`aom_dsp/aom_filter.h:28`).
const SCALE_SUBPEL_BITS: i32 = 10;
/// `SCALE_SUBPEL_MASK` (`aom_filter.h:30`).
const SCALE_SUBPEL_MASK: i32 = (1 << SCALE_SUBPEL_BITS) - 1;
/// `SCALE_EXTRA_BITS` (`aom_filter.h:31`).
const SCALE_EXTRA_BITS: i32 = SCALE_SUBPEL_BITS - 4;
/// `DIST_PRECISION_BITS` (`av1/common/enums.h:76`).
const DIST_PRECISION_BITS: i32 = 4;

/// The `ConvolveParams` fields the scaled kernels read. `dst16` is passed
/// alongside, as in [`super::compound`].
#[derive(Clone, Copy, Debug)]
pub struct ScaleConvolveParams {
    /// `conv_params->round_0`
    pub round_0: i32,
    /// `conv_params->round_1`
    pub round_1: i32,
    /// `conv_params->is_compound`
    pub is_compound: bool,
    /// `conv_params->do_average`
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

/// `av1_convolve_2d_scale_c` (convolve.c:494), lowbd.
///
/// `x_filter` / `y_filter` are the FULL `[16][taps]` kernel tables (not a
/// pre-selected row): the scaled path re-selects a row per output sample from
/// `(qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS`.
///
/// `src_off` is the interior origin; the reference needs `taps/2 - 1` samples
/// of border before it in each direction, and enough after it to cover the
/// stepped walk.
#[allow(clippy::too_many_arguments)]
pub fn convolve_2d_scale(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[[i16; 8]; 16],
    y_filter: &[[i16; 8]; 16],
    taps: usize,
    subpel_x_qn: i32,
    x_step_qn: i32,
    subpel_y_qn: i32,
    y_step_qn: i32,
    cp: &ScaleConvolveParams,
) {
    const BD: i32 = 8;
    let bits = FILTER_BITS * 2 - cp.round_0 - cp.round_1;
    let im_h =
        ((((h as i32 - 1) * y_step_qn + subpel_y_qn) >> SCALE_SUBPEL_BITS) + taps as i32) as usize;
    let im_stride = w;
    let fo_vert = taps / 2 - 1;
    let fo_horiz = taps / 2 - 1;

    // Horizontal pass.
    let mut im = vec![0i16; im_h * im_stride];
    let mut src_horiz = src_off as isize - (fo_vert * src_stride) as isize;
    for y in 0..im_h {
        let mut x_qn = subpel_x_qn;
        for x in 0..w {
            let src_x = src_horiz + (x_qn >> SCALE_SUBPEL_BITS) as isize;
            let idx = ((x_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS) as usize;
            let f = &x_filter[idx];
            let mut sum = 1i32 << (BD + FILTER_BITS - 1);
            for k in 0..taps {
                sum += i32::from(f[k])
                    * i32::from(src[(src_x + k as isize - fo_horiz as isize) as usize]);
            }
            im[y * im_stride + x] = rpo2(sum, cp.round_0) as i16;
            x_qn += x_step_qn;
        }
        src_horiz += src_stride as isize;
    }

    // Vertical pass — column-major, per the module note.
    let offset_bits = BD + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    for x in 0..w {
        let mut y_qn = subpel_y_qn;
        for y in 0..h {
            let base = (fo_vert * im_stride) as isize
                + ((y_qn >> SCALE_SUBPEL_BITS) as isize) * im_stride as isize
                + x as isize;
            let idx = ((y_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS) as usize;
            let f = &y_filter[idx];
            let mut sum = 1i32 << offset_bits;
            // `k` indexes both the kernel and a strided source offset, so the
            // iterator form would need a zip over a computed range; C's shape
            // is clearer here.
            #[allow(clippy::needless_range_loop)]
            for k in 0..taps {
                let off = base + (k as isize - fo_vert as isize) * im_stride as isize;
                sum += i32::from(f[k]) * i32::from(im[off as usize]);
            }
            // CONV_BUF_TYPE res — narrowed to 16 bits.
            let res = i32::from(rpo2(sum, cp.round_1) as u16);
            if cp.is_compound {
                if cp.do_average {
                    let mut tmp = i32::from(dst16[y * dst16_stride + x]);
                    if cp.use_dist_wtd_comp_avg {
                        tmp = tmp * cp.fwd_offset + res * cp.bck_offset;
                        tmp >>= DIST_PRECISION_BITS;
                    } else {
                        tmp += res;
                        tmp >>= 1;
                    }
                    tmp -= round_offset;
                    dst[y * dst_stride + x] = clip_pixel(rpo2(tmp, bits));
                } else {
                    dst16[y * dst16_stride + x] = res as u16;
                }
            } else {
                let tmp = res - round_offset;
                dst[y * dst_stride + x] = clip_pixel(rpo2(tmp, bits));
            }
            y_qn += y_step_qn;
        }
    }
}

/// `av1_highbd_convolve_2d_scale_c` — the high-bit-depth twin.
#[allow(clippy::too_many_arguments)]
pub fn highbd_convolve_2d_scale(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    dst16: &mut [u16],
    dst16_stride: usize,
    w: usize,
    h: usize,
    x_filter: &[[i16; 8]; 16],
    y_filter: &[[i16; 8]; 16],
    taps: usize,
    subpel_x_qn: i32,
    x_step_qn: i32,
    subpel_y_qn: i32,
    y_step_qn: i32,
    cp: &ScaleConvolveParams,
    bd: u32,
) {
    let bits = FILTER_BITS * 2 - cp.round_0 - cp.round_1;
    let im_h =
        ((((h as i32 - 1) * y_step_qn + subpel_y_qn) >> SCALE_SUBPEL_BITS) + taps as i32) as usize;
    let im_stride = w;
    let fo_vert = taps / 2 - 1;
    let fo_horiz = taps / 2 - 1;

    let mut im = vec![0i16; im_h * im_stride];
    let mut src_horiz = src_off as isize - (fo_vert * src_stride) as isize;
    for y in 0..im_h {
        let mut x_qn = subpel_x_qn;
        for x in 0..w {
            let src_x = src_horiz + (x_qn >> SCALE_SUBPEL_BITS) as isize;
            let idx = ((x_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS) as usize;
            let f = &x_filter[idx];
            let mut sum = 1i32 << (bd as i32 + FILTER_BITS - 1);
            for k in 0..taps {
                sum += i32::from(f[k])
                    * i32::from(src[(src_x + k as isize - fo_horiz as isize) as usize]);
            }
            im[y * im_stride + x] = rpo2(sum, cp.round_0) as i16;
            x_qn += x_step_qn;
        }
        src_horiz += src_stride as isize;
    }

    let offset_bits = bd as i32 + 2 * FILTER_BITS - cp.round_0;
    let round_offset =
        (1i32 << (offset_bits - cp.round_1)) + (1i32 << (offset_bits - cp.round_1 - 1));
    for x in 0..w {
        let mut y_qn = subpel_y_qn;
        for y in 0..h {
            let base = (fo_vert * im_stride) as isize
                + ((y_qn >> SCALE_SUBPEL_BITS) as isize) * im_stride as isize
                + x as isize;
            let idx = ((y_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS) as usize;
            let f = &y_filter[idx];
            let mut sum = 1i32 << offset_bits;
            // `k` indexes both the kernel and a strided source offset, so the
            // iterator form would need a zip over a computed range; C's shape
            // is clearer here.
            #[allow(clippy::needless_range_loop)]
            for k in 0..taps {
                let off = base + (k as isize - fo_vert as isize) * im_stride as isize;
                sum += i32::from(f[k]) * i32::from(im[off as usize]);
            }
            let res = i32::from(rpo2(sum, cp.round_1) as u16);
            if cp.is_compound {
                if cp.do_average {
                    let mut tmp = i32::from(dst16[y * dst16_stride + x]);
                    if cp.use_dist_wtd_comp_avg {
                        tmp = tmp * cp.fwd_offset + res * cp.bck_offset;
                        tmp >>= DIST_PRECISION_BITS;
                    } else {
                        tmp += res;
                        tmp >>= 1;
                    }
                    tmp -= round_offset;
                    dst[y * dst_stride + x] = clip_pixel_highbd(rpo2(tmp, bits), bd);
                } else {
                    dst16[y * dst16_stride + x] = res as u16;
                }
            } else {
                let tmp = res - round_offset;
                dst[y * dst_stride + x] = clip_pixel_highbd(rpo2(tmp, bits), bd);
            }
            y_qn += y_step_qn;
        }
    }
}
