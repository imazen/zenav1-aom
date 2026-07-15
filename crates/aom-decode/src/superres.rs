//! Byte-exact normative superres upscaling (`av1/common/resize.c`).
//!
//! AV1 superres codes a frame at a reduced (downscaled) width and the decoder
//! upscales it back to the full `UpscaledWidth` **horizontally only**, as a
//! normative post-CDEF stage (decodeframe.c:5451 `superres_post_decode`). The
//! upscale is an 8-tap polyphase horizontal convolution
//! (`av1_convolve_horiz_rs` / `av1_highbd_convolve_horiz_rs`) driven by the
//! normative filter table `av1_resize_filter_normative`, with fixed-point
//! subpel accumulation (`RS_SCALE_SUBPEL_BITS`) and edge-pixel extension.
//!
//! This port covers the SINGLE-tile-column case (`cm->tiles.cols == 1`, the
//! decode envelope for superres KEY/AVIF-still streams): `downscaled_x0 == 0`
//! and one convolve pass per plane. The frame edges are handled by clamping the
//! sample index to `[0, src_mi_width)` — byte-identical to libaom's
//! save/`memset`/restore edge replication (`upscale_normative_rect` pads
//! `border_cols` columns with the edge pixel; a clamp reproduces exactly that
//! for the single tile column, where both edges are frame edges).
//!
//! The decoder stores every plane as `u16` regardless of bit depth, so one
//! implementation with a `bd` clamp covers 8/10/12-bit: `clip_pixel` (lowbd,
//! `[0,255]`) and `clip_pixel_highbd` (`[0,(1<<bd)-1]`) are the same clamp on
//! the shared `u16` storage, and the integer `sum`/round math is bit-depth
//! independent.

// aom_dsp/aom_filter.h
const RS_SUBPEL_BITS: i32 = 6;
const RS_SUBPEL_MASK: i32 = (1 << RS_SUBPEL_BITS) - 1;
const RS_SCALE_SUBPEL_BITS: i32 = 14;
const RS_SCALE_SUBPEL_MASK: i32 = (1 << RS_SCALE_SUBPEL_BITS) - 1;
const RS_SCALE_EXTRA_BITS: i32 = RS_SCALE_SUBPEL_BITS - RS_SUBPEL_BITS; // 8
const RS_SCALE_EXTRA_OFF: i32 = 1 << (RS_SCALE_EXTRA_BITS - 1); // 128

// av1/common/resize.h, aom_dsp/aom_dsp_common.h
const UPSCALE_NORMATIVE_TAPS: usize = 8;
const FILTER_BITS: i32 = 7;

/// `SCALE_NUMERATOR` (av1/common/scale.h): superres numerator (always 8).
pub const SCALE_NUMERATOR: i32 = 8;

/// `av1_superres_scaled` (resize.h): the frame was coded downscaled iff the
/// denominator exceeds the numerator (range `[9, 16]`).
#[inline]
pub fn superres_scaled(scale_denominator: i32) -> bool {
    scale_denominator > SCALE_NUMERATOR
}

/// The coded (downscaled) `FrameWidth` for a given full `UpscaledWidth` and
/// superres denominator (`av1_superres_params` / `frame_size`):
/// `FrameWidth = (UpscaledWidth * SCALE_NUMERATOR + SuperresDenom/2) / SuperresDenom`.
#[inline]
pub fn coded_frame_width(upscaled_width: i32, scale_denominator: i32) -> i32 {
    if !superres_scaled(scale_denominator) {
        return upscaled_width;
    }
    (upscaled_width * SCALE_NUMERATOR + scale_denominator / 2) / scale_denominator
}

/// `av1_get_upscale_convolve_step` (resize.c): the `RS_SCALE_SUBPEL_BITS`
/// fixed-point sampling step to walk `in_length` source pixels across
/// `out_length` output columns.
#[inline]
pub fn get_upscale_convolve_step(in_length: i32, out_length: i32) -> i32 {
    ((in_length << RS_SCALE_SUBPEL_BITS) + out_length / 2) / out_length
}

/// `get_upscale_convolve_x0` (resize.c, static): the initial fixed-point subpel
/// offset (masked to `RS_SCALE_SUBPEL_MASK`) for the first output column.
#[inline]
pub fn get_upscale_convolve_x0(in_length: i32, out_length: i32, x_step_qn: i32) -> i32 {
    let err = out_length * x_step_qn - (in_length << RS_SCALE_SUBPEL_BITS);
    let x0 = (-((out_length - in_length) << (RS_SCALE_SUBPEL_BITS - 1)) + out_length / 2)
        / out_length
        + RS_SCALE_EXTRA_OFF
        - err / 2;
    // C: (int32_t)((uint32_t)x0 & RS_SCALE_SUBPEL_MASK)
    ((x0 as u32) & (RS_SCALE_SUBPEL_MASK as u32)) as i32
}

#[inline]
fn round_power_of_two(value: i32, n: i32) -> i32 {
    (value + (1 << (n - 1))) >> n
}

/// One plane's horizontal upscale, single tile column (`downscaled_x0 == 0`).
///
/// `src` holds the downscaled plane at `src_stride`, with valid reconstructed
/// content out to `src_mi_width` columns (the mi-aligned width — libaom's
/// `mi_col_end << (MI_SIZE_LOG2 - ss)` border-extension bound; reads past it
/// replicate the last mi-aligned pixel). `dst` receives `upscaled_plane_width`
/// columns per row at `dst_stride`. `downscaled_plane_width`/
/// `upscaled_plane_width` are the ACTUAL (crop, subsampled) widths that drive
/// the subpel step/offset. `rows` rows are processed. `bd` is the bit depth.
#[allow(clippy::too_many_arguments)]
pub fn upscale_plane(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    downscaled_plane_width: i32,
    upscaled_plane_width: i32,
    src_mi_width: i32,
    rows: usize,
    bd: i32,
) {
    debug_assert!(downscaled_plane_width > 0 && upscaled_plane_width > 0);
    debug_assert!(src_mi_width > 0);
    let x_step_qn = get_upscale_convolve_step(downscaled_plane_width, upscaled_plane_width);
    let x0_qn = get_upscale_convolve_x0(downscaled_plane_width, upscaled_plane_width, x_step_qn);
    let maxval = (1i32 << bd) - 1;
    let out_w = upscaled_plane_width as usize;
    let clamp_hi = src_mi_width - 1;

    for y in 0..rows {
        let srow = &src[y * src_stride..y * src_stride + src_mi_width as usize];
        let drow = &mut dst[y * dst_stride..y * dst_stride + out_w];
        let mut x_qn = x0_qn;
        for d in drow.iter_mut().take(out_w) {
            let int_pel = x_qn >> RS_SCALE_SUBPEL_BITS;
            let filter_idx = ((x_qn & RS_SCALE_SUBPEL_MASK) >> RS_SCALE_EXTRA_BITS) as usize;
            debug_assert!(filter_idx <= RS_SUBPEL_MASK as usize);
            let filt = &RESIZE_FILTER_NORMATIVE[filter_idx];
            // Sampling base relative to the plane origin (downscaled_x0 == 0):
            // input - 1 (rect) - (TAPS/2 - 1) (convolve) + int_pel = int_pel - 4.
            let base = int_pel - (UPSCALE_NORMATIVE_TAPS as i32 / 2 - 1) - 1;
            let mut sum = 0i32;
            for (k, &tap) in filt.iter().enumerate() {
                let idx = (base + k as i32).clamp(0, clamp_hi) as usize;
                sum += srow[idx] as i32 * tap as i32;
            }
            *d = round_power_of_two(sum, FILTER_BITS).clamp(0, maxval) as u16;
            x_qn += x_step_qn;
        }
    }
}

/// `av1_resize_filter_normative[1 << RS_SUBPEL_BITS][UPSCALE_NORMATIVE_TAPS]`
/// (resize.c): the 64-phase 8-tap normative upscale filter. Each row sums to
/// `1 << FILTER_BITS` (128).
#[rustfmt::skip]
pub static RESIZE_FILTER_NORMATIVE: [[i16; UPSCALE_NORMATIVE_TAPS]; 1 << RS_SUBPEL_BITS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],        [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],      [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],      [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],  [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],  [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],  [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1], [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1], [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1], [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1], [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1], [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],  [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],  [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],  [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],  [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],  [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],  [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],  [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],  [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],  [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1], [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1], [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1], [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1], [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1], [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],  [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],  [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],  [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],      [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],      [0, 0, -1, 2, 128, -1, 0, 0],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_rows_sum_to_128() {
        for (i, row) in RESIZE_FILTER_NORMATIVE.iter().enumerate() {
            let s: i32 = row.iter().map(|&t| t as i32).sum();
            assert_eq!(s, 1 << FILTER_BITS, "filter phase {i} does not sum to 128");
        }
    }

    #[test]
    fn coded_width_matches_spec() {
        // denom 8 => unscaled.
        assert_eq!(coded_frame_width(100, 8), 100);
        assert!(!superres_scaled(8));
        // denom in [9,16] downscales.
        for denom in 9..=16 {
            let up = 256;
            let coded = coded_frame_width(up, denom);
            assert!(superres_scaled(denom));
            assert!(coded < up, "denom {denom}: coded {coded} !< upscaled {up}");
            // (UpscaledWidth * 8 + denom/2) / denom
            assert_eq!(coded, (up * 8 + denom / 2) / denom);
        }
    }
}
