//! Encoder-side inter predictor construction — port of the
//! `aom_*_upsampled_pred` family (`av1/encoder/reconinter_enc.c`) and the two
//! `aom_comp_*_pred` blends (`aom_dsp/variance.c`) the motion search scores
//! compound candidates with.
//!
//! [`crate::inter_me::upsampled_pred`] already holds the single-reference lowbd
//! `aom_upsampled_pred_c`. Everything here is what the COMPOUND and
//! high-bit-depth arms of the subpel search additionally need: the second
//! predictor blended in (average or wedge/diffwtd mask), and the 10/12-bit
//! twins of the whole chain.
//!
//! | Rust | C |
//! |---|---|
//! | [`comp_avg_pred`] | `aom_comp_avg_pred_c` (aom_dsp/variance.c) |
//! | [`comp_mask_pred`] | `aom_comp_mask_pred_c` (variance.c) |
//! | [`highbd_comp_avg_pred`] | `aom_highbd_comp_avg_pred_c` (variance.c) |
//! | [`highbd_comp_mask_pred`] | `aom_highbd_comp_mask_pred_c` (variance.c) |
//! | [`comp_avg_upsampled_pred`] | `aom_comp_avg_upsampled_pred_c` (reconinter_enc.c) |
//! | [`comp_mask_upsampled_pred`] | `aom_comp_mask_upsampled_pred` (reconinter_enc.c) |
//! | [`highbd_upsampled_pred`] | `aom_highbd_upsampled_pred_c` (reconinter_enc.c) |
//! | [`highbd_comp_avg_upsampled_pred`] | `aom_highbd_comp_avg_upsampled_pred_c` (reconinter_enc.c) |
//! | [`highbd_comp_mask_upsampled_pred`] | `aom_highbd_comp_mask_upsampled_pred` (reconinter_enc.c) |
//!
//! # `invert_mask` is a swap, and C spells it two different ways
//! `aom_comp_mask_pred_c` swaps the two source POINTERS *and* their strides;
//! `aom_highbd_comp_mask_pred_c` instead branches inside the inner loop and
//! leaves the pointer advance fixed. The two are equivalent — the strides
//! travel with their buffers either way — but the lowbd spelling is the one
//! that inverts easily during transcription, so both are written here in the
//! shape C uses and both are differentially gated at `invert_mask` 0 and 1.
//!
//! # Differential coverage
//! `tests/inter_pred_enc_diff.rs`, tier 1 against the real exported C.

use aom_dsp::convolve::SUB_PEL_FILTERS_8;

/// `FILTER_BITS` (`aom_dsp/aom_filter.h`).
const FILTER_BITS: i32 = 7;
/// `SUBPEL_TAPS` (`aom_dsp/aom_filter.h`).
const SUBPEL_TAPS: usize = 8;
/// `SUBPEL_TAPS / 2 - 1` — the leading-tap offset both convolve passes apply.
const FILTER_OFF: usize = SUBPEL_TAPS / 2 - 1;
/// `AOM_BLEND_A64_ROUND_BITS` (`aom_dsp/blend.h:23`).
const AOM_BLEND_A64_ROUND_BITS: i32 = 6;
/// `AOM_BLEND_A64_MAX_ALPHA` (`blend.h:24`).
const AOM_BLEND_A64_MAX_ALPHA: i32 = 1 << AOM_BLEND_A64_ROUND_BITS;

#[inline]
fn round_pow2(v: i32, n: i32) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

#[inline]
fn clip_pixel_highbd(v: i32, bd: u32) -> u16 {
    v.clamp(0, (1i32 << bd) - 1) as u16
}

/// `AOM_BLEND_A64(a, v0, v1)` (`blend.h:26`).
#[inline]
fn blend_a64(a: i32, v0: i32, v1: i32) -> i32 {
    round_pow2(
        a * v0 + (AOM_BLEND_A64_MAX_ALPHA - a) * v1,
        AOM_BLEND_A64_ROUND_BITS,
    )
}

// ===================================================================
// aom_dsp/variance.c — the two compound blends
// ===================================================================

/// `aom_comp_avg_pred_c`: `comp_pred = round((pred + ref) / 2)`.
/// `comp_pred` and `pred` are tight at stride `width`; `ref` carries its own
/// stride.
pub fn comp_avg_pred(
    pred: &[u8],
    refb: &[u8],
    ref_off: usize,
    ref_stride: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut comp = vec![0u8; width * height];
    for i in 0..height {
        for j in 0..width {
            let tmp =
                i32::from(pred[i * width + j]) + i32::from(refb[ref_off + i * ref_stride + j]);
            comp[i * width + j] = round_pow2(tmp, 1) as u8;
        }
    }
    comp
}

/// `aom_comp_mask_pred_c`: the A64 mask blend of `ref` and `pred`.
///
/// `invert_mask` swaps which buffer the mask weights — see the module note.
#[allow(clippy::too_many_arguments)]
pub fn comp_mask_pred(
    pred: &[u8],
    refb: &[u8],
    ref_off: usize,
    ref_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    invert_mask: bool,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut comp = vec![0u8; width * height];
    for i in 0..height {
        for j in 0..width {
            let r = i32::from(refb[ref_off + i * ref_stride + j]);
            let p = i32::from(pred[i * width + j]);
            let (v0, v1) = if invert_mask { (p, r) } else { (r, p) };
            comp[i * width + j] = blend_a64(i32::from(mask[i * mask_stride + j]), v0, v1) as u8;
        }
    }
    comp
}

/// `aom_highbd_comp_avg_pred_c`.
pub fn highbd_comp_avg_pred(
    pred: &[u16],
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    width: usize,
    height: usize,
) -> Vec<u16> {
    let mut comp = vec![0u16; width * height];
    for i in 0..height {
        for j in 0..width {
            let tmp =
                i32::from(pred[i * width + j]) + i32::from(refb[ref_off + i * ref_stride + j]);
            comp[i * width + j] = round_pow2(tmp, 1) as u16;
        }
    }
    comp
}

/// `aom_highbd_comp_mask_pred_c`.
#[allow(clippy::too_many_arguments)]
pub fn highbd_comp_mask_pred(
    pred: &[u16],
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    invert_mask: bool,
    width: usize,
    height: usize,
) -> Vec<u16> {
    let mut comp = vec![0u16; width * height];
    for i in 0..height {
        for j in 0..width {
            let r = i32::from(refb[ref_off + i * ref_stride + j]);
            let p = i32::from(pred[i * width + j]);
            let (v0, v1) = if invert_mask { (p, r) } else { (r, p) };
            comp[i * width + j] = blend_a64(i32::from(mask[i * mask_stride + j]), v0, v1) as u16;
        }
    }
    comp
}

// ===================================================================
// reconinter_enc.c — the highbd upsampled predictor
// ===================================================================

/// `aom_highbd_convolve8_horiz_c` at a fixed phase (`x_step_q4 == 16`).
#[allow(clippy::too_many_arguments)]
fn highbd_convolve8_horiz(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    kernel: &[i16; 8],
    bd: u32,
) {
    for y in 0..h {
        let row = src_off as isize + (y * src_stride) as isize - FILTER_OFF as isize;
        for x in 0..w {
            let base = row + x as isize;
            let mut sum = 0i32;
            for (k, kv) in kernel.iter().enumerate() {
                sum += i32::from(*kv) * i32::from(src[(base + k as isize) as usize]);
            }
            dst[y * dst_stride + x] = clip_pixel_highbd(round_pow2(sum, FILTER_BITS), bd);
        }
    }
}

/// `aom_highbd_convolve8_vert_c` at a fixed phase (`y_step_q4 == 16`).
#[allow(clippy::too_many_arguments)]
fn highbd_convolve8_vert(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    kernel: &[i16; 8],
    bd: u32,
) {
    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize
                + (y as isize - FILTER_OFF as isize) * src_stride as isize
                + x as isize;
            let mut sum = 0i32;
            for (k, kv) in kernel.iter().enumerate() {
                sum += i32::from(*kv)
                    * i32::from(src[(base + (k as isize) * src_stride as isize) as usize]);
            }
            dst[y * dst_stride + x] = clip_pixel_highbd(round_pow2(sum, FILTER_BITS), bd);
        }
    }
}

/// `aom_highbd_upsampled_pred_c` (reconinter_enc.c), unscaled,
/// `subpel_search == USE_8_TAPS`.
///
/// The 10/12-bit twin of [`crate::inter_me::upsampled_pred`]: same four-way
/// dispatch on `(subpel_x_q3, subpel_y_q3)` and the same doubled kernel index
/// (`SUB_PEL_FILTERS_8[subpel_q3 << 1]`), with `clip_pixel_highbd(.., bd)` in
/// place of the byte clamp — including **between** the two passes of the 2-D
/// case, which is where the two bit depths genuinely diverge rather than merely
/// widening.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn highbd_upsampled_pred(
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    subpel_x_q3: usize,
    subpel_y_q3: usize,
    bd: u32,
) -> Vec<u16> {
    debug_assert!(subpel_x_q3 <= 7 && subpel_y_q3 <= 7);
    let mut dst = vec![0u16; w * h];
    let need_x = subpel_x_q3 != 0;
    let need_y = subpel_y_q3 != 0;

    if !need_x && !need_y {
        for y in 0..h {
            let s = ref_off + y * ref_stride;
            dst[y * w..y * w + w].copy_from_slice(&refb[s..s + w]);
        }
    } else if !need_y {
        let kx = &SUB_PEL_FILTERS_8[subpel_x_q3 << 1];
        highbd_convolve8_horiz(refb, ref_off, ref_stride, &mut dst, w, w, h, kx, bd);
    } else if !need_x {
        let ky = &SUB_PEL_FILTERS_8[subpel_y_q3 << 1];
        highbd_convolve8_vert(refb, ref_off, ref_stride, &mut dst, w, w, h, ky, bd);
    } else {
        let kx = &SUB_PEL_FILTERS_8[subpel_x_q3 << 1];
        let ky = &SUB_PEL_FILTERS_8[subpel_y_q3 << 1];
        // intermediate_height = (((h - 1) * 8 + subpel_y_q3) >> 3) + taps,
        // which is h + 7 for every subpel_y_q3 in 1..=7.
        let inter_h = h + SUBPEL_TAPS - 1;
        let mut temp = vec![0u16; inter_h * w];
        let horiz_off = ref_off - FILTER_OFF * ref_stride;
        highbd_convolve8_horiz(
            refb, horiz_off, ref_stride, &mut temp, w, w, inter_h, kx, bd,
        );
        highbd_convolve8_vert(&temp, FILTER_OFF * w, w, &mut dst, w, w, h, ky, bd);
    }
    dst
}

// ===================================================================
// reconinter_enc.c — the compound upsampled predictors
// ===================================================================

/// `aom_comp_avg_upsampled_pred_c`: build the lowbd upsampled predictor at the
/// sub-pel phase, then average it with the already-built first predictor.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn comp_avg_upsampled_pred(
    pred: &[u16],
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    subpel_x_q3: usize,
    subpel_y_q3: usize,
) -> Vec<u16> {
    let mut comp =
        crate::inter_me::upsampled_pred(refb, ref_off, ref_stride, w, h, subpel_x_q3, subpel_y_q3);
    for i in 0..h * w {
        comp[i] = round_pow2(i32::from(comp[i]) + i32::from(pred[i]), 1) as u16;
    }
    comp
}

/// `aom_comp_mask_upsampled_pred`: the lowbd upsampled predictor, then the A64
/// mask blend against the first predictor.
///
/// C reuses `comp_pred` as both the `ref` input and the output of
/// `aom_comp_mask_pred`, at stride `width` — so the mask blend's "reference"
/// is the freshly built predictor, not the original reference plane.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn comp_mask_upsampled_pred(
    pred: &[u16],
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    invert_mask: bool,
    w: usize,
    h: usize,
    subpel_x_q3: usize,
    subpel_y_q3: usize,
) -> Vec<u16> {
    let up =
        crate::inter_me::upsampled_pred(refb, ref_off, ref_stride, w, h, subpel_x_q3, subpel_y_q3);
    let mut comp = vec![0u16; w * h];
    for i in 0..h {
        for j in 0..w {
            let r = i32::from(up[i * w + j]);
            let p = i32::from(pred[i * w + j]);
            let (v0, v1) = if invert_mask { (p, r) } else { (r, p) };
            comp[i * w + j] = blend_a64(i32::from(mask[i * mask_stride + j]), v0, v1) as u16;
        }
    }
    comp
}

/// `aom_highbd_comp_avg_upsampled_pred_c`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn highbd_comp_avg_upsampled_pred(
    pred: &[u16],
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    subpel_x_q3: usize,
    subpel_y_q3: usize,
    bd: u32,
) -> Vec<u16> {
    let mut comp = highbd_upsampled_pred(
        refb,
        ref_off,
        ref_stride,
        w,
        h,
        subpel_x_q3,
        subpel_y_q3,
        bd,
    );
    for i in 0..h * w {
        comp[i] = round_pow2(i32::from(pred[i]) + i32::from(comp[i]), 1) as u16;
    }
    comp
}

/// `aom_highbd_comp_mask_upsampled_pred`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn highbd_comp_mask_upsampled_pred(
    pred: &[u16],
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    invert_mask: bool,
    w: usize,
    h: usize,
    subpel_x_q3: usize,
    subpel_y_q3: usize,
    bd: u32,
) -> Vec<u16> {
    let up = highbd_upsampled_pred(
        refb,
        ref_off,
        ref_stride,
        w,
        h,
        subpel_x_q3,
        subpel_y_q3,
        bd,
    );
    let mut comp = vec![0u16; w * h];
    for i in 0..h {
        for j in 0..w {
            let r = i32::from(up[i * w + j]);
            let p = i32::from(pred[i * w + j]);
            let (v0, v1) = if invert_mask { (p, r) } else { (r, p) };
            comp[i * w + j] = blend_a64(i32::from(mask[i * mask_stride + j]), v0, v1) as u16;
        }
    }
    comp
}
