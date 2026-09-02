//! `estimate_screen_content_antialiasing_aware` (av1/encoder/encoder.c:2222) —
//! the `AOM_SCREEN_DETECTION_ANTIALIASING_AWARE` screen-content detector
//! libaom runs on the UNFILTERED source luma BEFORE the frame search
//! (`av1_set_screen_content_options`, :2439; the default mode).
//!
//! It walks the luma plane in 16x16 blocks (partial edge blocks skipped; a
//! checkerboard of every other block, each weighed twice, under
//! `hl_sf.screen_detection_mode2_fast_detection`), classifies each block by
//! its colour count (8-bit domain — HBD sources are down-converted by
//! `>> (bd - 8)`) and its per-pixel variance (always on the un-converted
//! source), and derives two frame flags from the counts:
//!
//! * `allow_screen_content_tools = (palette - photo/16) * 256 * 10 > area`
//! * `allow_intrabc = allow_screen_content_tools && (intrabc - photo/16) * 256 * 12 > area`
//!
//! **Why the port needs it (KB-41 root #7).** `allow_intrabc` is decided
//! HERE, before the search, and the whole frame is searched with it: every
//! intra candidate pays `intrabc_cost[0]` (`intra_mode_info_cost_y`) and
//! `rd_pick_intrabc_mode_sb` runs. Only after the frame, if no block chose
//! IntraBC, does `encode_frame_internal` flip the header bit to 0
//! (encodeframe.c:2442 `if (features->allow_intrabc && !cpi->intrabc_used)
//! features->allow_intrabc = 0`) — WITHOUT re-searching. So the oracle
//! stream's `allow_intrabc` is the FINAL flag and cannot tell "searched with
//! IntraBC, none chosen" (every intra rate +51 at the default CDF) from
//! "never searched". The datagen census cells where IntraBC won somewhere
//! were byte-exact with the header bit; the ones where it lost everywhere
//! (e.g. 128x128 cq19 cpu4) carried a uniform −51 on every intra candidate
//! from mi(0,0) on. This module re-derives the search-time decision.

use crate::partition_pick::perpixel_variance_y;

/// `kBlockWidth` / `kBlockHeight` (:2225-2226).
const BLOCK: usize = 16;
/// `kBlockArea`.
const BLOCK_AREA: i64 = (BLOCK * BLOCK) as i64;
/// `kSimpleColorThresh` — text/glyphs without anti-aliasing, 4-colour graphics.
const SIMPLE_COLOR_THRESH: i32 = 4;
/// `kComplexInitialColorThresh` — the first pass of a potentially anti-aliased block.
const COMPLEX_INITIAL_COLOR_THRESH: i32 = 40;
/// `kComplexFinalColorThresh` — the colour count after one dilation round.
const COMPLEX_FINAL_COLOR_THRESH: i32 = 6;
/// `kVarThresh` — low- vs high-variance blocks (per-pixel variance).
const VAR_THRESH: u32 = 5;
/// `BLOCK_16X16` in the port's block-size enum.
const BLOCK_16X16: usize = 6;

/// The detector's frame-level output (`FeatureFlags` + the `cpi` mirrors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenContentDecision {
    /// `features->allow_screen_content_tools`.
    pub allow_screen_content_tools: bool,
    /// `features->allow_intrabc` — the SEARCH-TIME decision (see the module doc).
    pub allow_intrabc: bool,
    /// `cpi->is_screen_content_type`.
    pub is_screen_content_type: bool,
    /// The (multiplier-normalized) block counts, for diagnostics.
    pub count_palette: i64,
    pub count_intrabc: i64,
    pub count_photo: i64,
}

impl ScreenContentDecision {
    /// `av1_set_screen_content_options`' no-detection arm (encoder.c:2466-2470):
    /// `rt_sf.use_nonrd_pick_mode && !rt_sf.hybrid_intra_pickmode` (= allintra
    /// speed 9) sets `allow_screen_content_tools = allow_intrabc = 0` and
    /// returns before either estimator runs. KB-41 root #13.
    pub fn detection_disabled() -> Self {
        Self {
            allow_screen_content_tools: false,
            allow_intrabc: false,
            is_screen_content_type: false,
            count_palette: 0,
            count_intrabc: 0,
            count_photo: 0,
        }
    }

    /// `av1_set_screen_content_options`' sequence-forced arm (encoder.c:
    /// 2443-2447): `seq_params->force_screen_content_tools != SELECT` copies
    /// the forced bit into BOTH `allow_screen_content_tools` and
    /// `allow_intrabc` (`is_screen_content_type` untouched) before any
    /// detector runs.
    pub fn forced(on: bool) -> Self {
        Self {
            allow_screen_content_tools: on,
            allow_intrabc: on,
            ..Self::detection_disabled()
        }
    }

    /// `av1_set_screen_content_options`' `--tune-content=screen` arm
    /// (encoder.c:2449-2455): both tools on unconditionally (allintra:
    /// `allow_intrabc = oxcf->mode == REALTIME ? 0 : 1` = 1), the frame is
    /// typed screen content, and no detector (or trial encode) runs.
    pub fn tuned_screen() -> Self {
        Self {
            allow_screen_content_tools: true,
            allow_intrabc: true,
            is_screen_content_type: true,
            ..Self::detection_disabled()
        }
    }
}

/// `av1_count_colors_with_threshold` (intra_mode_search.c:383): counts the
/// distinct 8-bit values of a block, bailing (returns `(false, thresh + 1)`)
/// the moment the count exceeds `threshold`.
fn count_colors_with_threshold(
    blk: &[u8],
    stride: usize,
    rows: usize,
    cols: usize,
    threshold: i32,
) -> (bool, i32) {
    let mut has = [false; 256];
    let mut n = 0i32;
    for r in 0..rows {
        for c in 0..cols {
            let v = blk[r * stride + c] as usize;
            if !has[v] {
                has[v] = true;
                n += 1;
                if n > threshold {
                    return (false, n);
                }
            }
        }
    }
    (true, n)
}

/// `av1_find_dominant_value` (encoder.c:2110): the most frequent value; ties
/// resolve to the FIRST value to reach the winning count (strict `>`).
fn find_dominant_value(blk: &[u8], stride: usize, rows: usize, cols: usize) -> u8 {
    let mut count = [0u32; 256];
    let mut dom_count = 0u32;
    let mut dom = 0u8;
    for r in 0..rows {
        for c in 0..cols {
            let v = blk[r * stride + c];
            count[v as usize] += 1;
            if count[v as usize] > dom_count {
                dom = v;
                dom_count = count[v as usize];
            }
        }
    }
    dom
}

/// `av1_dilate_block` (encoder.c:2152): one round of 8-neighbour dilation of
/// the dominant value — every source pixel equal to it stamps its 4 sides and
/// 4 corners in `dilated` (bounds-checked), on top of a plain copy.
fn dilate_block(
    src: &[u8],
    src_stride: usize,
    dilated: &mut [u8],
    dilated_stride: usize,
    rows: usize,
    cols: usize,
) {
    let dom = find_dominant_value(src, src_stride, rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            dilated[r * dilated_stride + c] = src[r * src_stride + c];
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            let value = src[r * src_stride + c];
            if value != dom {
                continue;
            }
            if r != 0 {
                dilated[(r - 1) * dilated_stride + c] = value;
            }
            if r != rows - 1 {
                dilated[(r + 1) * dilated_stride + c] = value;
            }
            if c != 0 {
                dilated[r * dilated_stride + (c - 1)] = value;
            }
            if c != cols - 1 {
                dilated[r * dilated_stride + (c + 1)] = value;
            }
            if r != 0 && c != 0 {
                dilated[(r - 1) * dilated_stride + (c - 1)] = value;
            }
            if r != 0 && c != cols - 1 {
                dilated[(r - 1) * dilated_stride + (c + 1)] = value;
            }
            if r != rows - 1 && c != 0 {
                dilated[(r + 1) * dilated_stride + (c - 1)] = value;
            }
            if r != rows - 1 && c != cols - 1 {
                dilated[(r + 1) * dilated_stride + (c + 1)] = value;
            }
        }
    }
}

/// The detector (module doc). `src` is the luma plane (u16 samples, `bd`-bit
/// range; 8-bit sources carry 8-bit values) at `off` with `stride`;
/// `fast_detection` is `hl_sf.screen_detection_mode2_fast_detection` (allintra
/// speed >= 3).
///
/// # `width` / `height` are `y_width` / `y_height` — NOT the crop
///
/// C reads `cpi->unfiltered_source->y_width` / `->y_height`
/// (`encoder.c`, `estimate_screen_content_antialiasing_aware`), which for a
/// `YV12_BUFFER_CONFIG` are the **8-ALIGNED** dimensions
/// (`aom_realloc_frame_buffer`: `y_width = (width + 7) & ~7`), not
/// `y_crop_width` / `y_crop_height`. Callers must pass `(crop + 7) & !7`.
///
/// This is load-bearing twice over: `area` is the denominator of BOTH frame
/// decisions (`(palette - photo/16) * 256 * 10 > area`), and the 16x16 block
/// loop bound is `c + 16 <= width`. On a crop that is not 8-aligned the two
/// readings differ by up to ~3% of area, which flips borderline frames.
/// MEASURED 2026-09-02 on the bootstrap-free encoder gate: passing the crop
/// instead of the aligned size made 258x258, 260x260 and 262x262 (textured,
/// 4:2:0, cq 32) code `allow_screen_content_tools = 1` where real aomenc codes
/// 0; with the aligned size all three agree and 258/262 become byte-identical
/// end to end.
pub fn estimate_screen_content_antialiasing_aware(
    src: &[u16],
    off: usize,
    stride: usize,
    width: usize,
    height: usize,
    bd: u8,
    fast_detection: bool,
) -> ScreenContentDecision {
    let area = (width as i64) * (height as i64);
    let multiplier = if fast_detection { 2usize } else { 1 };
    let mut count_palette = 0i64;
    let mut count_intrabc = 0i64;
    let mut count_photo = 0i64;
    let mut blk8 = [0u8; BLOCK * BLOCK];
    let mut dilated = [0u8; BLOCK * BLOCK];
    let shift = u32::from(bd) - 8;
    let mut r = 0usize;
    while r + BLOCK <= height {
        // Alternate skipping in a "checkerboard" pattern under fast detection.
        let initial_col = if fast_detection && (r / BLOCK) % 2 == 1 {
            BLOCK
        } else {
            0
        };
        let mut c = initial_col;
        while c + BLOCK <= width {
            let blk_off = off + r * stride + c;
            // Down-convert to the 8-bit domain (identity at bd 8).
            for br in 0..BLOCK {
                for bc in 0..BLOCK {
                    let v = src[blk_off + br * stride + bc] >> shift;
                    debug_assert!(v < 256);
                    blk8[br * BLOCK + bc] = v as u8;
                }
            }
            let (under, n) = count_colors_with_threshold(
                &blk8,
                BLOCK,
                BLOCK,
                BLOCK,
                COMPLEX_INITIAL_COLOR_THRESH,
            );
            if n > 1 && under {
                if n <= SIMPLE_COLOR_THRESH {
                    // Simple block: palettizable; IntraBC candidate when textured.
                    count_palette += 1;
                    // Variance always comes from the source with no down-conversion.
                    let var = perpixel_variance_y(src, blk_off, stride, BLOCK_16X16, bd);
                    if var > VAR_THRESH {
                        count_intrabc += 1;
                    }
                } else {
                    // Complex block: dilate with the dominant colour to drop the
                    // anti-aliased pixels from the final palette count.
                    dilate_block(&blk8, BLOCK, &mut dilated, BLOCK, BLOCK, BLOCK);
                    let (under2, _) = count_colors_with_threshold(
                        &dilated,
                        BLOCK,
                        BLOCK,
                        BLOCK,
                        COMPLEX_FINAL_COLOR_THRESH,
                    );
                    if under2 {
                        let var = perpixel_variance_y(src, blk_off, stride, BLOCK_16X16, bd);
                        if var > VAR_THRESH {
                            count_palette += 1;
                            count_intrabc += 1;
                        }
                    }
                    // else: non-palettizable — counted nowhere.
                }
            } else if n > COMPLEX_INITIAL_COLOR_THRESH {
                // Photo-like block (the count bailed past the threshold).
                count_photo += 1;
            }
            // else: solid block (1 colour) — counted nowhere.
            c += BLOCK * multiplier;
        }
        r += BLOCK;
    }
    if fast_detection {
        count_palette *= 2;
        count_intrabc *= 2;
        count_photo *= 2;
    }
    let allow_screen_content_tools = (count_palette - count_photo / 16) * BLOCK_AREA * 10 > area;
    let allow_intrabc =
        allow_screen_content_tools && (count_intrabc - count_photo / 16) * BLOCK_AREA * 12 > area;
    let is_screen_content_type = allow_intrabc
        || (count_palette * BLOCK_AREA * 15 > area * 4 && count_intrabc * BLOCK_AREA * 30 > area);
    ScreenContentDecision {
        allow_screen_content_tools,
        allow_intrabc,
        is_screen_content_type,
        count_palette,
        count_intrabc,
        count_photo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominant_value_tie_keeps_first_to_reach_the_count() {
        // 3 x 0 then 3 x 7: 0 reaches count 3 first.
        let blk = [0u8, 0, 0, 7, 7, 7];
        assert_eq!(find_dominant_value(&blk, 6, 1, 6), 0);
        let blk = [7u8, 7, 7, 0, 0, 0];
        assert_eq!(find_dominant_value(&blk, 6, 1, 6), 7);
    }

    #[test]
    fn threshold_count_bails_at_threshold_plus_one() {
        let blk: Vec<u8> = (0..16u8).collect();
        assert_eq!(count_colors_with_threshold(&blk, 16, 1, 16, 40), (true, 16));
        assert_eq!(count_colors_with_threshold(&blk, 16, 1, 16, 4), (false, 5));
    }

    #[test]
    fn dilation_extends_the_dominant_value_to_all_eight_neighbours() {
        // A 4x4 block: dominant 9 in the middle 2x2, the rest 1..=12 distinct.
        let mut src = [0u8; 16];
        let mut k = 1u8;
        for (i, v) in src.iter_mut().enumerate() {
            if matches!(i, 5 | 6 | 9 | 10) {
                *v = 9;
            } else {
                *v = k;
                k += 1;
                if k == 9 {
                    k = 13;
                }
            }
        }
        let mut out = [0u8; 16];
        dilate_block(&src, 4, &mut out, 4, 4, 4);
        assert!(out.iter().all(|&v| v == 9), "{out:?}");
    }

    #[test]
    fn flat_frame_is_not_screen_content() {
        // 64x64 solid grey: every block is a 1-colour "solid" block → no counts.
        let src = vec![128u16; 64 * 64];
        let d = estimate_screen_content_antialiasing_aware(&src, 0, 64, 64, 64, 8, false);
        assert_eq!((d.count_palette, d.count_intrabc, d.count_photo), (0, 0, 0));
        assert!(!d.allow_screen_content_tools && !d.allow_intrabc);
    }

    #[test]
    fn two_colour_textured_frame_enables_both_tools() {
        // 64x64 checkerboard of 0/255 at 2px: 2 colours (simple), high variance.
        let mut src = vec![0u16; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                if ((r / 2) + (c / 2)) % 2 == 0 {
                    src[r * 64 + c] = 255;
                }
            }
        }
        let d = estimate_screen_content_antialiasing_aware(&src, 0, 64, 64, 64, 8, false);
        assert_eq!(
            (d.count_palette, d.count_intrabc, d.count_photo),
            (16, 16, 0)
        );
        assert!(d.allow_screen_content_tools && d.allow_intrabc);
        // fast detection: 8 of the 16 blocks visited, each counted twice.
        let f = estimate_screen_content_antialiasing_aware(&src, 0, 64, 64, 64, 8, true);
        assert_eq!((f.count_palette, f.count_intrabc), (16, 16));
    }
}
