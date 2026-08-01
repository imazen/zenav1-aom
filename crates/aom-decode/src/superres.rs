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
//! The upscale runs ONE CONVOLVE PASS PER TILE COLUMN
//! ([`upscale_plane_tiles`], `av1_upscale_normative_rows`'s `j` loop): each pass
//! covers the destination columns `[upscaled_x0, upscaled_x1)` that the tile's
//! source range `[downscaled_x0, downscaled_x1)` maps to, and the fractional
//! sampling offset `x0_qn` CARRIES across tile columns so the sampling grid stays
//! continuous over the frame. [`upscale_plane`] is the single-tile-column wrapper.
//!
//! **Edge replication vs real neighbours.** `upscale_normative_rect` pads
//! `border_cols = UPSCALE_NORMATIVE_TAPS/2 + 1 = 5` columns just outside the tile
//! with a replica of the tile's edge pixel, but only when `pad_left`
//! (`j == 0`) / `pad_right` (`j == cols - 1`) — i.e. only at FRAME edges. At an
//! INTERIOR tile boundary neither pad applies and the convolve reads the real
//! neighbouring tile column's reconstructed pixels.
//!
//! Both are reproduced by one clamp of the frame-global sample index to
//! `[0, src_mi_width - 1]`, because the pads only ever occur at those two
//! bounds: `pad_left` implies `downscaled_x0 == 0`, and `pad_right` implies
//! `downscaled_x1 == src_mi_width`. Interior reads stay strictly inside the
//! range (the convolve reaches at most `downscaled_x1 + 2` / `downscaled_x0 - 5`,
//! and every non-edge tile has a neighbour supplying those pixels), so the clamp
//! is inert there. The one residual case — an interior read reaching
//! `src_mi_width` when the LAST tile column is under 3 plane pixels wide — also
//! agrees: libaom upscales from a `copy_buffer` whose crop width is
//! `ALIGN_POWER_OF_TWO(cm->width, 3) == mi_cols * MI_SIZE` and whose columns
//! beyond that are `aom_extend_frame_borders` replicas of `src_mi_width - 1`
//! (resize.c:1329-1339 → yv12extend.c:200-237).
//!
//! **The tile walk is bit-identical to one continuous convolve** (derived from
//! C, then property-tested by `tile_walk_matches_the_continuous_walk` over
//! denominators 9..=16 × 2/3/4-column splits × aligned and over-aligned widths,
//! and confirmed end-to-end against the C decoder by
//! `tests/superres_tiles_diff.rs`). Since interior boundaries take no pad, the
//! only thing tiling could change is the sampling grid — and the carry cancels
//! the per-tile origin exactly: the absolute sample position at output column
//! `X` in tile `j` is
//! `downscaled_x0*2^14 + x0_qn_j + (X - upscaled_x0)*x_step_qn`, and substituting
//! `x0_qn_j = x0_qn_0 + upscaled_x0*x_step_qn - downscaled_x0*2^14` (the carry,
//! telescoped) collapses it to `x0_qn_0 + X*x_step_qn` for every tile. No
//! division enters the cancellation, so it is exact, not approximate.
//! Consequently the tile loop is a structural/parallelism artifact of libaom's
//! rect-based API rather than a behavioural mode — which is why lifting the
//! former `multi-tile superres` reject needed no change to the pixel results,
//! only a faithful walk and the evidence that it is faithful. It is ported in
//! C's shape anyway so the correspondence stays auditable.
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
/// Thin wrapper over [`upscale_plane_tiles`] with the one-tile boundary list
/// `[0, src_mi_width]`. With `cols == 1` the single pass is BOTH the first and
/// the last tile column, so `upscaled_x0 == 0`, `upscaled_x1` takes the
/// last-tile rule (`= upscaled_plane_width`) and the `x0_qn` carry is dead —
/// the denominator never enters the arithmetic, which is why passing the
/// unscaled [`SCALE_NUMERATOR`] here is exact rather than a placeholder.
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
    upscale_plane_tiles(
        src,
        src_stride,
        dst,
        dst_stride,
        downscaled_plane_width,
        upscaled_plane_width,
        src_mi_width,
        &[0, src_mi_width],
        SCALE_NUMERATOR,
        rows,
        bd,
    );
}

/// One plane's horizontal upscale across every tile column
/// (`av1_upscale_normative_rows`, resize.c:1119).
///
/// `tile_x` is the tile-column boundary list in DOWNSCALED plane pixels:
/// length `cols + 1`, strictly increasing, `tile_x[0] == 0` and
/// `tile_x[cols] == src_mi_width`. Entry `j` is
/// `av1_tile_set_col`'s `mi_col_start << (MI_SIZE_LOG2 - ss_x)` for tile `j`
/// (equivalently tile `j-1`'s `mi_col_end`, with the final entry clamped to
/// `mi_cols`). `scale_denominator` is `SuperresDenom`; every other parameter
/// matches [`upscale_plane`].
///
/// Two details of C's tile walk are load-bearing and easy to drop:
/// the LAST tile column takes `upscaled_x1 = upscaled_plane_width` directly
/// rather than the rounded `(downscaled_x1 * denom) / SCALE_NUMERATOR` (which
/// can land short), and `x0_qn` carries from one tile column to the next by
/// `dst_width * x_step_qn - (src_width << RS_SCALE_SUBPEL_BITS)`.
#[allow(clippy::too_many_arguments)]
pub fn upscale_plane_tiles(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    downscaled_plane_width: i32,
    upscaled_plane_width: i32,
    src_mi_width: i32,
    tile_x: &[i32],
    scale_denominator: i32,
    rows: usize,
    bd: i32,
) {
    debug_assert!(downscaled_plane_width > 0 && upscaled_plane_width > 0);
    debug_assert!(src_mi_width > 0);
    debug_assert!(tile_x.len() >= 2, "tile_x needs cols+1 >= 2 boundaries");
    debug_assert_eq!(tile_x[0], 0, "tile_x must start at the plane origin");
    debug_assert_eq!(
        tile_x[tile_x.len() - 1],
        src_mi_width,
        "tile_x must end at the mi-aligned plane width"
    );
    let cols = tile_x.len() - 1;
    // x_step_qn / x0_qn are frame-level: derived ONCE from the plane crop widths,
    // before the tile loop (resize.c:1130-1134).
    let x_step_qn = get_upscale_convolve_step(downscaled_plane_width, upscaled_plane_width);
    let mut x0_qn =
        get_upscale_convolve_x0(downscaled_plane_width, upscaled_plane_width, x_step_qn);
    let maxval = (1i32 << bd) - 1;
    let clamp_hi = src_mi_width - 1;

    for j in 0..cols {
        let downscaled_x0 = tile_x[j];
        let downscaled_x1 = tile_x[j + 1];
        debug_assert!(downscaled_x1 > downscaled_x0, "empty tile column {j}");
        let src_width = downscaled_x1 - downscaled_x0;
        let upscaled_x0 = (downscaled_x0 * scale_denominator) / SCALE_NUMERATOR;
        // Last tile column: rounding can leave (downscaled_x1 * denom) /
        // SCALE_NUMERATOR BELOW upscaled_plane_width, so C uses the plane width
        // itself rather than AOMMIN (resize.c:1148-1155).
        let upscaled_x1 = if j == cols - 1 {
            upscaled_plane_width
        } else {
            (downscaled_x1 * scale_denominator) / SCALE_NUMERATOR
        };
        let dst_width = upscaled_x1 - upscaled_x0;
        debug_assert!(dst_width > 0, "empty destination range for tile column {j}");

        for y in 0..rows {
            let srow = &src[y * src_stride..y * src_stride + src_mi_width as usize];
            let row_base = y * dst_stride;
            let drow = &mut dst[row_base + upscaled_x0 as usize..row_base + upscaled_x1 as usize];
            let mut x_qn = x0_qn;
            for d in drow.iter_mut() {
                let int_pel = x_qn >> RS_SCALE_SUBPEL_BITS;
                let filter_idx = ((x_qn & RS_SCALE_SUBPEL_MASK) >> RS_SCALE_EXTRA_BITS) as usize;
                debug_assert!(filter_idx <= RS_SUBPEL_MASK as usize);
                let filt = &RESIZE_FILTER_NORMATIVE[filter_idx];
                // Sampling base, frame-global: downscaled_x0 (the rect's origin)
                // - 1 (rect passes `input - 1`) - (TAPS/2 - 1) (convolve) + int_pel.
                let base = downscaled_x0 + int_pel - (UPSCALE_NORMATIVE_TAPS as i32 / 2 - 1) - 1;
                let mut sum = 0i32;
                for (k, &tap) in filt.iter().enumerate() {
                    // Clamping to the frame-global [0, src_mi_width-1] reproduces
                    // BOTH of C's cases: the pad_left/pad_right edge replication
                    // (which only ever occurs at exactly these two bounds) and the
                    // unpadded interior read of the neighbouring tile column
                    // (where the clamp is inert). See the module docs.
                    let idx = (base + k as i32).clamp(0, clamp_hi) as usize;
                    sum += srow[idx] as i32 * tap as i32;
                }
                *d = round_power_of_two(sum, FILTER_BITS).clamp(0, maxval) as u16;
                x_qn += x_step_qn;
            }
        }
        // Carry the fractional pixel offset into the next tile column
        // (resize.c:1183). Without this only tile column 0 samples correctly.
        x0_qn += (dst_width * x_step_qn) - (src_width << RS_SCALE_SUBPEL_BITS);
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

    /// A ramp plane whose horizontal detail survives the 8-tap upscale, so a
    /// sampling-grid difference shows up as a pixel difference.
    fn ramp(w: usize, rows: usize) -> Vec<u16> {
        (0..w * rows)
            .map(|i| {
                let x = (i % w) as i32;
                let y = (i / w) as i32;
                (((x * 7 + y * 3) % 251) + ((x * x / 5) % 37)) as u16
            })
            .collect()
    }

    /// The single-tile wrapper is EXACTLY `upscale_plane_tiles` over the one-tile
    /// boundary list — the structural guarantee that generalising the kernel left
    /// `cols == 1` byte-identical.
    #[test]
    fn single_tile_wrapper_matches_explicit_one_column() {
        let (src_w, up_w, rows) = (128i32, 256i32, 9usize);
        let src = ramp(src_w as usize, rows);
        let mut a = vec![0u16; up_w as usize * rows];
        let mut b = vec![0u16; up_w as usize * rows];
        upscale_plane(
            &src, src_w as usize, &mut a, up_w as usize, 125, up_w, src_w, rows, 8,
        );
        upscale_plane_tiles(
            &src,
            src_w as usize,
            &mut b,
            up_w as usize,
            125,
            up_w,
            src_w,
            &[0, src_w],
            16,
            rows,
            8,
        );
        assert_eq!(a, b, "one-column tiles walk must equal the wrapper");
    }

    /// THE TILE-WALK INVARIANT (see the module docs): splitting a plane into any
    /// number of tile columns must leave the pixels UNCHANGED. Interior tile
    /// boundaries take no pad, and the `x0_qn` carry cancels the per-tile origin
    /// exactly, so C's tile loop reproduces one continuous convolve.
    ///
    /// This is the sharpest available check on the three pieces that are easy to
    /// get wrong, because each of them breaks it:
    /// * dropping the `x0_qn` carry shifts every column past the first;
    /// * padding (clamping) at interior boundaries replicates an edge pixel where
    ///   the neighbouring tile's samples belong;
    /// * omitting `downscaled_x0` from the sampling base reads from the frame
    ///   origin for every column.
    ///
    /// Swept over denominators 9..=16, both a mi-aligned and an over-aligned
    /// source width, and 2/3/4-column splits at superblock-multiple boundaries.
    #[test]
    fn tile_walk_matches_the_continuous_walk() {
        let rows = 7usize;
        let mut checked = 0u32;
        // (src_mi_width, downscaled_plane_width): the second case has the crop
        // width BELOW the mi-aligned width, the border-clamp regime.
        for &(src_w, crop_w) in &[(256i32, 256i32), (256, 251)] {
            let src = ramp(src_w as usize, rows);
            for denom in 9i32..=16 {
                // The upscaled width whose coded width is `crop_w` (frame_size).
                let up_w = (crop_w * denom + SCALE_NUMERATOR / 2) / SCALE_NUMERATOR;
                let mut one = vec![0u16; up_w as usize * rows];
                upscale_plane(
                    &src,
                    src_w as usize,
                    &mut one,
                    up_w as usize,
                    crop_w,
                    up_w,
                    src_w,
                    rows,
                    8,
                );
                for splits in [
                    vec![0, 128, src_w],
                    vec![0, 64, src_w],
                    vec![0, 64, 128, src_w],
                    vec![0, 64, 128, 192, src_w],
                ] {
                    let mut many = vec![0u16; up_w as usize * rows];
                    upscale_plane_tiles(
                        &src,
                        src_w as usize,
                        &mut many,
                        up_w as usize,
                        crop_w,
                        up_w,
                        src_w,
                        &splits,
                        denom,
                        rows,
                        8,
                    );
                    assert_eq!(
                        one,
                        many,
                        "denom {denom} src_w {src_w} crop {crop_w} splits {splits:?}: \
                         the tile walk diverged from the continuous walk"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 2 * 8 * 4, "tile-walk invariant arm count");
        // Anti-vacuity: the plane genuinely has horizontal detail, so a shifted
        // sampling grid WOULD show up (a flat plane upscales to itself).
        let src = ramp(256, rows);
        assert!(
            src.windows(2).filter(|w| w[0] != w[1]).count() > 200,
            "ramp is too flat to discriminate a sampling-grid shift"
        );
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
