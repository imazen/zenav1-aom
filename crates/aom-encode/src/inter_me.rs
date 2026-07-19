//! Inter-frame motion estimation — the net-new subpel search machinery
//! (INTER-ENCODE-ROADMAP.md chunk 2d).
//!
//! The full-pel diamond/mesh search is the shared intrabc/inter core in
//! [`crate::intrabc_search`] (retargeted to a reference frame). This module
//! holds the pieces that are net-new for inter: the **upsampled subpel
//! predictor** ([`upsampled_pred`], the cost primitive of
//! `av1_find_best_sub_pixel_tree`) and — as they land — the subpel tree search
//! itself.
//!
//! All lowbd (bd = 8). The port stores planes as `u16` (bd8 values `0..=255`),
//! matching the rest of the codebase; the arithmetic is byte-identical to
//! libaom's `u8` kernels since every value fits in a byte.

use aom_convolve::SUB_PEL_FILTERS_8;

const FILTER_BITS: i32 = 7;
const SUBPEL_TAPS: usize = 8;
/// `SUBPEL_TAPS / 2 - 1` — the 8-tap filter's left/top origin offset.
const FILTER_OFF: usize = SUBPEL_TAPS / 2 - 1; // 3

#[inline]
fn round_pow2(v: i32, n: i32) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

#[inline]
fn clip_pixel(v: i32) -> u16 {
    v.clamp(0, 255) as u16
}

/// One horizontal 8-tap pass (`aom_convolve8_horiz_c` with `x_step_q4 ==
/// SUBPEL_SHIFTS`, i.e. the fixed-phase `aom_upsampled_pred` use): for each
/// output `(y, x)`, `dst = clip(round(Σ_k kernel[k]·src[y·stride + x - 3 + k],
/// FILTER_BITS))`. `src_off` is the block origin; the tap reads `x-3 .. x+4`, so
/// `src` needs `>= 3` samples of left border and `>= 4` of right.
#[allow(clippy::too_many_arguments)]
fn convolve8_horiz(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    kernel: &[i16; 8],
) {
    for y in 0..h {
        let row = src_off as isize + (y * src_stride) as isize - FILTER_OFF as isize;
        for x in 0..w {
            let base = row + x as isize;
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                sum += kernel[k] as i32 * src[(base + k as isize) as usize] as i32;
            }
            dst[y * dst_stride + x] = clip_pixel(round_pow2(sum, FILTER_BITS));
        }
    }
}

/// One vertical 8-tap pass (`aom_convolve8_vert_c`, fixed-phase): for each
/// output `(y, x)`, `dst = clip(round(Σ_k kernel[k]·src[(y - 3 + k)·stride + x],
/// FILTER_BITS))`. `src` needs `>= 3` samples of top border and `>= 4` of bottom.
#[allow(clippy::too_many_arguments)]
fn convolve8_vert(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    kernel: &[i16; 8],
) {
    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize
                + (y as isize - FILTER_OFF as isize) * src_stride as isize
                + x as isize;
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                sum += kernel[k] as i32 * src[(base + (k as isize) * src_stride as isize) as usize] as i32;
            }
            dst[y * dst_stride + x] = clip_pixel(round_pow2(sum, FILTER_BITS));
        }
    }
}

/// `aom_upsampled_pred_c` (av1/encoder/reconinter_enc.c:462), lowbd, unscaled,
/// `subpel_search == USE_8_TAPS` (`av1_get_filter(USE_8_TAPS)` =
/// `EIGHTTAP_REGULAR`). The fixed-phase 8-tap subpel predictor the speed-0
/// subpel motion search builds (`upsampled_pref_error` ->
/// `check_better`/`upsampled_setup_center_error`).
///
/// The C kernel selects `av1_get_interp_filter_subpel_kernel(filter,
/// subpel_q3 << 1)` — the `EIGHTTAP_REGULAR` row at the doubled 1/16-pel phase,
/// which is [`SUB_PEL_FILTERS_8`]`[subpel_q3 << 1]`. Dispatch matches C:
/// - `(0, 0)` → block copy;
/// - `(x, 0)` → single horizontal pass;
/// - `(0, y)` → single vertical pass;
/// - `(x, y)` → horizontal into a `(h + 7)`-row intermediate (u8-clipped, as the
///   C 2-D path clips between passes), then vertical.
///
/// `refb`/`ref_off`/`ref_stride` describe the reference plane; `ref_off` is the
/// fullpel block origin with `>= 3` samples of border before and `>= 4` after in
/// every subpel-filtered direction (the caller's `get_buf_from_mv` position on a
/// border-extended reference frame). `subpel_x_q3`/`subpel_y_q3` are 1/8-pel
/// phases in `0..=7`. Returns the `w`×`h` predictor (u16 bd8, tight stride `w`).
///
/// Differentially locked vs the REAL `aom_upsampled_pred_c` in
/// `tests/upsampled_pred_diff.rs`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn upsampled_pred(
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    subpel_x_q3: usize,
    subpel_y_q3: usize,
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
        convolve8_horiz(refb, ref_off, ref_stride, &mut dst, w, w, h, kx);
    } else if !need_x {
        let ky = &SUB_PEL_FILTERS_8[subpel_y_q3 << 1];
        convolve8_vert(refb, ref_off, ref_stride, &mut dst, w, w, h, ky);
    } else {
        // 2-D separable: horizontal into an (h + 7)-row intermediate starting 3
        // rows above the block origin, then vertical. The intermediate is
        // u8-clipped per pass (round to FILTER_BITS + clip), byte-identical to
        // aom_convolve8_horiz_c writing its uint8_t temp.
        let kx = &SUB_PEL_FILTERS_8[subpel_x_q3 << 1];
        let ky = &SUB_PEL_FILTERS_8[subpel_y_q3 << 1];
        let inter_h = h + SUBPEL_TAPS - 1; // h + 7
        let mut temp = vec![0u16; inter_h * w];
        let horiz_off = ref_off - FILTER_OFF * ref_stride;
        convolve8_horiz(refb, horiz_off, ref_stride, &mut temp, w, w, inter_h, kx);
        // The block origin sits at intermediate row FILTER_OFF (= 3); the
        // vertical pass reads temp[(y - 3 + k) + 3] = temp[y + k].
        convolve8_vert(&temp, FILTER_OFF * w, w, &mut dst, w, w, h, ky);
    }
    dst
}
