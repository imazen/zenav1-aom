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

// ===================================================================
// reconinter_enc.c — assembling a masked compound predictor from the two
// single-reference predictors the RD search already built.
//
// `av1_compound_type_rd`'s wedge and diffwtd arms build each reference's
// predictor ONCE into a scratch buffer and then, for every candidate mask,
// blend those two buffers rather than re-running motion compensation. These
// three functions are that blend.
//
// | Rust | C (`av1/encoder/reconinter_enc.c`) |
// |---|---|
// | [`build_masked_compound`] | `build_masked_compound` :312 + `build_masked_compound_highbd` :330 |
// | [`build_wedge_inter_predictor_from_buf`] | `build_wedge_inter_predictor_from_buf` :349 |
// | [`av1_build_wedge_inter_predictor_from_buf`] | `av1_build_wedge_inter_predictor_from_buf` :407 |
//
// Differential coverage: `tests/wedge_from_buf_diff.rs`, tier 1c (the oracle
// is libaom's own reconinter_enc.c compiled verbatim into
// `shim/reconinter_enc_shim.c`; three of these four C definitions are
// file-static).
// ===================================================================

use crate::compound_type::{CompoundType, InterInterComp, Pixels};
use aom_dsp::inter::compound::{build_compound_diffwtd_mask, build_compound_diffwtd_mask_highbd};
use aom_dsp::inter::interintra::{blend_a64_mask, blend_a64_mask_lowbd, wedge_mask_signed};

/// `block_size_wide` / `block_size_high` / `mi_size_wide_log2` /
/// `mi_size_high_log2` (`common_data.h`).
const BLK_W: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
const MI_SIZE_WIDE_LOG2: [usize; 22] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4,
];
const MI_SIZE_HIGH_LOG2: [usize; 22] = [
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2,
];

/// A writable pixel plane, the destination twin of
/// [`crate::compound_type::Pixels`].
pub enum PixelsMut<'a> {
    /// 8-bit destination.
    Low(&'a mut [u8]),
    /// 16-bit destination.
    High(&'a mut [u16]),
}

// # Two upstream behaviours a later reader will want to "fix"
//
// 1. **The unmasked arm of `build_wedge_inter_predictor_from_buf` faults at
//    high bit depth.** Its masked arms take `ext_dst0` as a RAW `uint16_t *`
//    and apply `CONVERT_TO_BYTEPTR` themselves (:359-362, :375); the `else`
//    arm applies `CONVERT_TO_SHORTPTR` to that same argument (:392), which
//    shifts a real pointer LEFT and dereferences it. Both call sites enter
//    this function only for a masked compound type, so the arm is dead
//    upstream. The port copies — which is what the name says and what the
//    lowbd arm does — and the differential does not drive C there.
// 2. **`av1_build_compound_diffwtd_mask_neon` reads src1's second row at
//    src0's stride** (`av1/common/arm/reconinter_neon.c:162`, the `w == 8`
//    arm). It agrees with the C reference only when the two predictor strides
//    are equal, which every call site makes them. The differential therefore
//    passes equal (but not tight) strides.

/// `build_masked_compound` (reconinter_enc.c:312) and its
/// `build_masked_compound_highbd` twin (:330).
///
/// # `subw` / `subh` are DERIVED, not passed
/// Both C functions recover the plane's subsampling by comparing the block
/// dimensions they were handed against the luma block's:
/// `subh = (2 << mi_size_high_log2[sb_type]) == h`. The comment there
/// ("May be refactored to pass in subsampling factors directly") is C's own.
/// The comparison is with `2 << log2`, i.e. **twice** the block's MI extent —
/// `mi_size_high_log2` counts 4-pixel units, so `2 << log2` is the block's
/// height in pixels divided by two. So `subh` is true exactly when this plane
/// is half the luma height.
///
/// # The mask is at LUMA stride
/// `mask_stride` is `block_size_wide[sb_type]` even for a chroma plane; the
/// blend box-averages it down. Passing the plane's own width instead reads the
/// wrong rows and is silent.
#[allow(clippy::too_many_arguments)]
pub fn build_masked_compound(
    dst: &mut PixelsMut<'_>,
    dst_stride: usize,
    src0: Pixels<'_>,
    src0_stride: usize,
    src1: Pixels<'_>,
    src1_stride: usize,
    mask: &[u8],
    sb_type: usize,
    h: usize,
    w: usize,
) {
    let subh = (2 << MI_SIZE_HIGH_LOG2[sb_type]) == h;
    let subw = (2 << MI_SIZE_WIDE_LOG2[sb_type]) == w;
    let mask_stride = BLK_W[sb_type];
    match (dst, src0, src1) {
        (PixelsMut::Low(d), Pixels::Low(a), Pixels::Low(b)) => blend_a64_mask_lowbd(
            d,
            dst_stride,
            a,
            src0_stride,
            b,
            src1_stride,
            mask,
            mask_stride,
            w,
            h,
            subw,
            subh,
        ),
        (PixelsMut::High(d), Pixels::High(a), Pixels::High(b)) => blend_a64_mask(
            d,
            dst_stride,
            a,
            src0_stride,
            b,
            src1_stride,
            mask,
            mask_stride,
            w,
            h,
            subw,
            subh,
        ),
        _ => panic!("build_masked_compound: mixed 8-bit and 16-bit buffers"),
    }
}

/// The `xd` state [`build_wedge_inter_predictor_from_buf`] reads.
#[derive(Clone, Copy, Debug)]
pub struct WedgeFromBufCtx {
    /// `mbmi->bsize` — the LUMA block size, which the mask is sized by.
    pub bsize: usize,
    /// `has_second_ref(mbmi)`.
    pub is_compound: bool,
    /// `mbmi->interinter_comp`.
    pub comp: InterInterComp,
    /// `xd->bd`.
    pub bd: u8,
}

/// `build_wedge_inter_predictor_from_buf` (reconinter_enc.c:349): blend one
/// plane of a masked compound block out of the two scratch predictors, or copy
/// the first one through when the block is not masked.
///
/// `seg_mask` is `xd->seg_mask`, `bw * bh` entries at luma stride. It is an
/// **in/out** parameter: on plane 0 of a `COMPOUND_DIFFWTD` block C REBUILDS
/// it here from the two scratch predictors, and the chroma planes then read
/// what luma left. That ordering is load-bearing — calling this for plane 1
/// before plane 0 blends against a stale mask, silently.
///
/// C spells the non-masked arm as `aom_convolve_copy`, which for these
/// arguments is a plain rectangular copy.
#[allow(clippy::too_many_arguments)]
pub fn build_wedge_inter_predictor_from_buf(
    ctx: &WedgeFromBufCtx,
    plane: usize,
    dst: &mut PixelsMut<'_>,
    dst_offset: usize,
    dst_stride: usize,
    w: usize,
    h: usize,
    ext0: Pixels<'_>,
    ext0_stride: usize,
    ext1: Pixels<'_>,
    ext1_stride: usize,
    seg_mask: &mut [u8],
) {
    let masked = ctx.is_compound && ctx.comp.ty.is_masked();
    if !masked {
        copy_rect(dst, dst_offset, dst_stride, ext0, ext0_stride, w, h);
        return;
    }

    if plane == 0 && ctx.comp.ty == CompoundType::DiffWtd {
        match (ext0, ext1) {
            (Pixels::Low(a), Pixels::Low(b)) => build_compound_diffwtd_mask(
                seg_mask,
                ctx.comp.mask_type,
                a,
                ext0_stride,
                b,
                ext1_stride,
                h,
                w,
            ),
            (Pixels::High(a), Pixels::High(b)) => build_compound_diffwtd_mask_highbd(
                seg_mask,
                ctx.comp.mask_type,
                a,
                ext0_stride,
                b,
                ext1_stride,
                h,
                w,
                u32::from(ctx.bd),
            ),
            _ => panic!("build_wedge_inter_predictor_from_buf: mixed pixel widths"),
        }
    }

    // `av1_get_compound_type_mask` (reconinter.c:290): the baked wedge mask for
    // COMPOUND_WEDGE, `comp_data->seg_mask` for COMPOUND_DIFFWTD.
    let wedge;
    let mask: &[u8] = match ctx.comp.ty {
        CompoundType::Wedge => {
            wedge = wedge_mask_signed(ctx.bsize, ctx.comp.wedge_index, ctx.comp.wedge_sign)
                .expect("COMPOUND_WEDGE at a bsize with no wedge codebook");
            &wedge
        }
        _ => seg_mask,
    };

    // The destination is a sub-rectangle of the plane; the blend writes at
    // `dst_stride` from `dst_offset`, which is what C's
    // `dst_buf->buf + dst_buf->stride * y + x` spells.
    match dst {
        PixelsMut::Low(d) => build_masked_compound(
            &mut PixelsMut::Low(&mut d[dst_offset..]),
            dst_stride,
            ext0,
            ext0_stride,
            ext1,
            ext1_stride,
            mask,
            ctx.bsize,
            h,
            w,
        ),
        PixelsMut::High(d) => build_masked_compound(
            &mut PixelsMut::High(&mut d[dst_offset..]),
            dst_stride,
            ext0,
            ext0_stride,
            ext1,
            ext1_stride,
            mask,
            ctx.bsize,
            h,
            w,
        ),
    }
}

/// `aom_convolve_copy` / `aom_highbd_convolve_copy` as this call site uses
/// them: a `w * h` rectangular copy between two strided planes.
fn copy_rect(
    dst: &mut PixelsMut<'_>,
    dst_offset: usize,
    dst_stride: usize,
    src: Pixels<'_>,
    src_stride: usize,
    w: usize,
    h: usize,
) {
    match (dst, src) {
        (PixelsMut::Low(d), Pixels::Low(s)) => {
            for r in 0..h {
                let (o, i) = (dst_offset + r * dst_stride, r * src_stride);
                d[o..o + w].copy_from_slice(&s[i..i + w]);
            }
        }
        (PixelsMut::High(d), Pixels::High(s)) => {
            for r in 0..h {
                let (o, i) = (dst_offset + r * dst_stride, r * src_stride);
                d[o..o + w].copy_from_slice(&s[i..i + w]);
            }
        }
        _ => panic!("copy_rect: mixed 8-bit and 16-bit buffers"),
    }
}

/// `get_plane_block_size` (`common_data.h`), reusing the decoder-side table
/// rather than re-transcribing `ss_size_lookup`.
#[inline]
fn plane_block_size(bsize: usize, ss_x: bool, ss_y: bool) -> usize {
    aom_dsp::entropy::partition::get_plane_block_size(bsize, usize::from(ss_x), usize::from(ss_y))
}

/// `av1_build_wedge_inter_predictor_from_buf` (reconinter_enc.c:407): the
/// plane loop around [`build_wedge_inter_predictor_from_buf`].
///
/// Each plane is blended over its FULL plane block size (`x == y == 0`,
/// `w`/`h` from `get_plane_block_size`), which is why the inner function's
/// `x`/`y` arguments are always zero at this one call site.
#[allow(clippy::too_many_arguments)]
pub fn av1_build_wedge_inter_predictor_from_buf(
    ctx: &WedgeFromBufCtx,
    plane_from: usize,
    plane_to: usize,
    subsampling: &[(bool, bool)],
    dst: &mut [PixelsMut<'_>],
    dst_stride: &[usize],
    ext0: &[Pixels<'_>],
    ext0_stride: &[usize],
    ext1: &[Pixels<'_>],
    ext1_stride: &[usize],
    seg_mask: &mut [u8],
) {
    for plane in plane_from..=plane_to {
        let (ss_x, ss_y) = subsampling[plane];
        let plane_bsize = plane_block_size(ctx.bsize, ss_x, ss_y);
        build_wedge_inter_predictor_from_buf(
            ctx,
            plane,
            &mut dst[plane],
            0,
            dst_stride[plane],
            BLK_W[plane_bsize],
            BLK_H[plane_bsize],
            ext0[plane],
            ext0_stride[plane],
            ext1[plane],
            ext1_stride[plane],
            seg_mask,
        );
    }
}

// ===================================================================
// reconinter_enc.c — the subpel-parameter derivation every encoder-side
// predictor build starts from.
//
// `av1_enc_build_one_inter_predictor` is `build_one_inter_predictor`
// (`common/reconinter_template.inc:23`) with `IS_DEC == 0`, and its first act
// is `enc_calc_subpel_params`: turn the block's position and MV into a source
// pointer plus the fractional offsets the convolve is driven with. That step
// is pure arithmetic and is ported here; the convolve dispatch above it
// (`av1_make_inter_predictor` / `av1_make_masked_inter_predictor`) still is
// not.
//
// | Rust | C |
// |---|---|
// | [`SubpelParams`] | `SubpelParams` (`common/blockd.h`) |
// | [`init_subpel_params`] | `init_subpel_params` (`common/reconinter.h:131`) |
// | [`enc_calc_subpel_params`] | `enc_calc_subpel_params` (`reconinter_enc.c:32`) |
// | [`InterBlockParams::new`] | `init_inter_block_params` (`reconinter.h:194`), the `top`/`left` half |
//
// Differential coverage: `tests/subpel_params_diff.rs`, tier 1c.
// ===================================================================

use aom_dsp::inter::scale::ScaleFactors;

/// `SUBPEL_BITS` (`aom_dsp/aom_filter.h:23`).
const SUBPEL_BITS: i32 = 4;
/// `SCALE_SUBPEL_BITS` (`aom_filter.h:28`).
const SCALE_SUBPEL_BITS: i32 = 10;
/// `SCALE_SUBPEL_MASK` (`aom_filter.h:30`).
const SCALE_SUBPEL_MASK: i32 = (1 << SCALE_SUBPEL_BITS) - 1;
/// `SCALE_EXTRA_BITS` (`aom_filter.h:31`).
const SCALE_EXTRA_BITS: i32 = SCALE_SUBPEL_BITS - SUBPEL_BITS;
/// `SCALE_EXTRA_OFF` (`aom_filter.h:32`).
const SCALE_EXTRA_OFF: i32 = (1 << SCALE_EXTRA_BITS) / 2;
/// `AOM_INTERP_EXTEND` (`aom_scale/yv12config.h:31`).
const AOM_INTERP_EXTEND: i32 = 4;
/// `AOM_BORDER_IN_PIXELS` (`yv12config.h:32`).
const AOM_BORDER_IN_PIXELS: i32 = 288;

/// `AOM_LEFT_TOP_MARGIN_SCALED(subsampling)` (`reconinter.h:31`) — how far
/// above/left of the reference plane a prediction may legally reach, in
/// `SCALE_SUBPEL_BITS` units. `init_inter_block_params` stores it NEGATED as
/// `inter_pred_params->{top,left}`.
#[inline]
const fn left_top_margin_scaled(subsampling: u32) -> i32 {
    ((AOM_BORDER_IN_PIXELS >> subsampling) - AOM_INTERP_EXTEND) << SCALE_SUBPEL_BITS
}

/// The slice of `InterPredParams` that [`init_subpel_params`] reads.
///
/// C fills a 30-field struct through `av1_init_inter_params`, of which the
/// subpel derivation touches six. Naming just those six keeps the port honest
/// about what it does and does not model — everything else in
/// `InterPredParams` belongs to the convolve, which is not ported yet.
#[derive(Clone, Copy, Debug)]
pub struct InterBlockParams {
    /// `pix_row` — the block's top edge in the PLANE's pixel grid.
    pub pix_row: i32,
    /// `pix_col` — the block's left edge in the plane's pixel grid.
    pub pix_col: i32,
    /// `subsampling_x`.
    pub subsampling_x: u32,
    /// `subsampling_y`.
    pub subsampling_y: u32,
    /// `top` = `-AOM_LEFT_TOP_MARGIN_SCALED(subsampling_y)`.
    pub top: i32,
    /// `left` = `-AOM_LEFT_TOP_MARGIN_SCALED(subsampling_x)`.
    pub left: i32,
}

impl InterBlockParams {
    /// `init_inter_block_params` (`reconinter.h:194`), restricted to the
    /// fields the subpel derivation uses. `top` and `left` are DERIVED from
    /// the subsampling, so they cannot disagree with it.
    #[inline]
    pub const fn new(pix_row: i32, pix_col: i32, subsampling_x: u32, subsampling_y: u32) -> Self {
        Self {
            pix_row,
            pix_col,
            subsampling_x,
            subsampling_y,
            top: -left_top_margin_scaled(subsampling_y),
            left: -left_top_margin_scaled(subsampling_x),
        }
    }
}

/// `SubpelParams` (`common/blockd.h`) — what the convolve is steered with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubpelParams {
    /// `xs` — the horizontal step, `sf->x_step_q4`.
    pub xs: i32,
    /// `ys` — the vertical step, `sf->y_step_q4`.
    pub ys: i32,
    /// `subpel_x` — `pos_x & SCALE_SUBPEL_MASK`.
    pub subpel_x: i32,
    /// `subpel_y` — `pos_y & SCALE_SUBPEL_MASK`.
    pub subpel_y: i32,
    /// `pos_x` — the horizontal source position, clamped.
    pub pos_x: i32,
    /// `pos_y` — the vertical source position, clamped.
    pub pos_y: i32,
}

/// `init_subpel_params` (`reconinter.h:131`): the block position plus the MV,
/// mapped through the reference's scale factors and clamped to the legal
/// reach of the reference plane.
///
/// `width` / `height` are the REFERENCE buffer's dimensions
/// (`pre_buf->width`, `pre_buf->height`), not the block's — they set how far
/// down and right the prediction may reach.
///
/// # The MV is scaled by the plane's subsampling before anything else
/// `src_mv->row * (1 << (1 - ssy))` doubles a luma MV (ssy == 0) and leaves a
/// chroma one alone (ssy == 1). An MV arrives in 1/8-pel luma units; the
/// position it is added to is in `SUBPEL_BITS` (1/16) plane units, so the
/// doubling is the unit conversion, not a scaling choice.
pub fn init_subpel_params(
    src_mv: (i16, i16),
    params: &InterBlockParams,
    sf: &ScaleFactors,
    width: i32,
    height: i32,
) -> SubpelParams {
    let (mv_row, mv_col) = src_mv;
    let orig_pos_y =
        (params.pix_row << SUBPEL_BITS) + i32::from(mv_row) * (1 << (1 - params.subsampling_y));
    let orig_pos_x =
        (params.pix_col << SUBPEL_BITS) + i32::from(mv_col) * (1 << (1 - params.subsampling_x));

    // `av1_unscaled_value` (scale.h:54) ignores `sf` entirely and is just the
    // q4 -> q10 shift; the scaled arm additionally applies the ratio.
    let (mut pos_x, mut pos_y) = if sf.is_scaled() {
        (sf.scaled_x(orig_pos_x), sf.scaled_y(orig_pos_y))
    } else {
        (
            orig_pos_x * (1 << SCALE_EXTRA_BITS),
            orig_pos_y * (1 << SCALE_EXTRA_BITS),
        )
    };
    pos_x += SCALE_EXTRA_OFF;
    pos_y += SCALE_EXTRA_OFF;

    let bottom = (height + AOM_INTERP_EXTEND) << SCALE_SUBPEL_BITS;
    let right = (width + AOM_INTERP_EXTEND) << SCALE_SUBPEL_BITS;
    pos_y = pos_y.clamp(params.top, bottom);
    pos_x = pos_x.clamp(params.left, right);

    SubpelParams {
        xs: sf.x_step_q4,
        ys: sf.y_step_q4,
        subpel_x: pos_x & SCALE_SUBPEL_MASK,
        subpel_y: pos_y & SCALE_SUBPEL_MASK,
        pos_x,
        pos_y,
    }
}

/// `enc_calc_subpel_params` (reconinter_enc.c:32): [`init_subpel_params`] plus
/// the source pointer it implies.
///
/// C returns `pre_buf->buf0 + (pos_y >> SCALE_SUBPEL_BITS) * stride +
/// (pos_x >> SCALE_SUBPEL_BITS)`. Rust returns that as an OFFSET from `buf0`,
/// which is what the caller needs and what can be checked; note it is a signed
/// offset — `pos_y` and `pos_x` are clamped to `top`/`left`, which are
/// NEGATIVE, so a block predicting from above or left of the reference plane
/// legitimately produces a negative offset into the frame's border.
///
/// The `>>` is arithmetic on a signed `int` in C, i.e. it rounds toward
/// negative infinity, not toward zero.
pub fn enc_calc_subpel_params(
    src_mv: (i16, i16),
    params: &InterBlockParams,
    sf: &ScaleFactors,
    ref_width: i32,
    ref_height: i32,
    ref_stride: i32,
) -> (SubpelParams, i32) {
    let subpel = init_subpel_params(src_mv, params, sf, ref_width, ref_height);
    let offset =
        (subpel.pos_y >> SCALE_SUBPEL_BITS) * ref_stride + (subpel.pos_x >> SCALE_SUBPEL_BITS);
    (subpel, offset)
}
