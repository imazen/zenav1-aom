//! Global-motion model conversion and error metrics — port of the
//! self-contained half of libaom v3.14.1 `av1/encoder/global_motion.c`.
//!
//! Global motion is a whole encoder subsystem the port did not have. This
//! module lands the parts that do **not** need the warp predictor in the loop:
//! the double→fixed-point model conversion the estimator's output goes
//! through, the parameter-offset arithmetic the refinement walks on, the
//! inlier segmentation map, and the segmented frame-error metric.
//!
//! | Rust | C |
//! |---|---|
//! | [`is_enough_erroradvantage`] | `av1_is_enough_erroradvantage` (global_motion.c:33) |
//! | [`convert_model_to_params`] | `av1_convert_model_to_params` (:57) via `convert_to_params` (:39) |
//! | [`add_param_offset`] | `add_param_offset` (:67) |
//! | [`force_wmtype`] | `force_wmtype` (:93) |
//! | [`get_wmtype`] | `get_wmtype` (av1/common/mv.h:299) |
//! | [`compute_feature_segmentation_map`] | `av1_compute_feature_segmentation_map` (:483) |
//! | [`segmented_frame_error`] / [`highbd_segmented_frame_error`] | `av1_segmented_frame_error` (:321) |
//!
//! | [`warp_error`] | `warp_error` (:276) |
//! | [`get_warp_error`] | `get_warp_error` (:341) |
//! | [`refine_integerized_param`] | `av1_refine_integerized_param` (:364) |
//!
//! **Tier note.** Everything above is gated at tier 1: the exported entries
//! against their own C symbol, and the `static` helpers (`add_param_offset`,
//! `force_wmtype`, `warp_error`, `get_warp_error`) through the exported
//! `av1_refine_integerized_param`, which drives all four.
//!
//! **NOT here:** the high-bit-depth error path (`highbd_warp_error`,
//! `highbd_segmented_frame_error` is present but `highbd_warp_error` is not),
//! and the `global_motion_facade.c` driver that decides WHICH models to try.
//! [`refine_integerized_param`] is lowbd-only and says so at its signature.
//!
//! # Differential coverage
//! `tests/global_motion_diff.rs`, tier 1 against the real exported C.

use aom_dsp::inter::warp::WarpedMotionParams;

/// `WARPEDMODEL_PREC_BITS` (av1/common/mv.h).
const WARPEDMODEL_PREC_BITS: i32 = 16;
/// `GM_TRANS_PREC_BITS` (mv.h:164).
const GM_TRANS_PREC_BITS: i32 = 6;
/// `GM_ABS_TRANS_BITS` (mv.h:165).
const GM_ABS_TRANS_BITS: i32 = 12;
/// `GM_TRANS_PREC_DIFF` (mv.h:167).
const GM_TRANS_PREC_DIFF: i32 = WARPEDMODEL_PREC_BITS - GM_TRANS_PREC_BITS;
/// `GM_TRANS_DECODE_FACTOR` (mv.h:169).
const GM_TRANS_DECODE_FACTOR: i32 = 1 << GM_TRANS_PREC_DIFF;
/// `GM_ALPHA_PREC_BITS` (mv.h:172).
const GM_ALPHA_PREC_BITS: i32 = 15;
/// `GM_ABS_ALPHA_BITS` (mv.h:173).
const GM_ABS_ALPHA_BITS: i32 = 12;
/// `GM_ALPHA_PREC_DIFF` (mv.h:174).
const GM_ALPHA_PREC_DIFF: i32 = WARPEDMODEL_PREC_BITS - GM_ALPHA_PREC_BITS;
/// `GM_ALPHA_DECODE_FACTOR` (mv.h:175).
const GM_ALPHA_DECODE_FACTOR: i32 = 1 << GM_ALPHA_PREC_DIFF;
/// `GM_TRANS_MAX` (mv.h:177).
const GM_TRANS_MAX: i32 = 1 << GM_ABS_TRANS_BITS;
/// `GM_ALPHA_MAX` (mv.h:178).
const GM_ALPHA_MAX: i32 = 1 << GM_ABS_ALPHA_BITS;
/// `GM_TRANS_MIN` (mv.h:180).
const GM_TRANS_MIN: i32 = -GM_TRANS_MAX;
/// `GM_ALPHA_MIN` (mv.h:181).
const GM_ALPHA_MIN: i32 = -GM_ALPHA_MAX;
/// `WARP_ERROR_BLOCK_LOG` (common/warped_motion.h:33).
pub const WARP_ERROR_BLOCK_LOG: usize = 5;
/// `WARP_ERROR_BLOCK` (warped_motion.h:34).
pub const WARP_ERROR_BLOCK: usize = 1 << WARP_ERROR_BLOCK_LOG;
/// `erroradv_prod_tr` (encoder/global_motion.h:81).
const ERRORADV_PROD_TR: f64 = 20000.0;
/// `FEAT_COUNT_TR` (global_motion.c:481).
const FEAT_COUNT_TR: u8 = 3;
/// `SEG_COUNT_TR` (global_motion.c:482).
const SEG_COUNT_TR: i32 = 48;

/// `TransformationType` (av1/common/mv.h).
pub const IDENTITY: u8 = 0;
/// `TRANSLATION`
pub const TRANSLATION: u8 = 1;
/// `ROTZOOM`
pub const ROTZOOM: u8 = 2;
/// `AFFINE`
pub const AFFINE: u8 = 3;

/// `max_trans_model_params[TRANS_TYPES]` (global_motion.c:365): how many of the
/// six `wmmat` entries each transformation type actually codes.
pub const MAX_TRANS_MODEL_PARAMS: [usize; 4] = [0, 2, 4, 6];

/// `av1_is_enough_erroradvantage` (global_motion.c:33): whether a global-motion
/// model's error advantage justifies the bits its parameters cost.
///
/// Both halves are needed — the model must beat the ratio threshold **and**
/// its advantage-times-cost product must stay under `erroradv_prod_tr`.
#[must_use]
pub fn is_enough_erroradvantage(
    best_erroradvantage: f64,
    params_cost: i32,
    gm_erroradv_tr: f64,
) -> bool {
    best_erroradvantage < gm_erroradv_tr
        && best_erroradvantage * f64::from(params_cost) < ERRORADV_PROD_TR
}

/// `get_wmtype` (av1/common/mv.h:299): classify a model by its `wmmat`.
#[must_use]
pub fn get_wmtype(wm: &WarpedMotionParams) -> u8 {
    let one = 1 << WARPEDMODEL_PREC_BITS;
    if wm.wmmat[5] == one && wm.wmmat[4] == 0 && wm.wmmat[2] == one && wm.wmmat[3] == 0 {
        return if wm.wmmat[1] == 0 && wm.wmmat[0] == 0 {
            IDENTITY
        } else {
            TRANSLATION
        };
    }
    if wm.wmmat[2] == wm.wmmat[5] && wm.wmmat[3] == -wm.wmmat[4] {
        ROTZOOM
    } else {
        AFFINE
    }
}

/// `convert_to_params` (global_motion.c:39) + `av1_convert_model_to_params`
/// (:57): the double model the estimator produces, rounded into the
/// fixed-point `wmmat` the bitstream carries.
///
/// Two details:
/// * The translation entries clamp **then** multiply by
///   `GM_TRANS_DECODE_FACTOR`; the alpha entries subtract the diagonal one,
///   clamp, add it back, and only then multiply. Reordering either changes the
///   result at the clamp boundary.
/// * The rounding is `floor(x + 0.5)`, which is round-half-UP, not
///   round-half-to-even and not Rust's `f64::round` (round-half-away-from-zero)
///   — they differ at negative half-way values.
#[must_use]
pub fn convert_model_to_params(params: &[f64; 6]) -> WarpedMotionParams {
    let mut wm = WarpedMotionParams::default();
    let m = &mut wm.wmmat;

    for i in 0..2 {
        let v = (params[i] * f64::from(1 << GM_TRANS_PREC_BITS) + 0.5).floor() as i32;
        m[i] = v.clamp(GM_TRANS_MIN, GM_TRANS_MAX) * GM_TRANS_DECODE_FACTOR;
    }
    for i in 2..6 {
        let diag_value = if i == 2 || i == 5 {
            1 << GM_ALPHA_PREC_BITS
        } else {
            0
        };
        let v = (params[i] * f64::from(1 << GM_ALPHA_PREC_BITS) + 0.5).floor() as i32;
        let v = (v - diag_value).clamp(GM_ALPHA_MIN, GM_ALPHA_MAX);
        m[i] = (v + diag_value) * GM_ALPHA_DECODE_FACTOR;
    }

    wm.wmtype = get_wmtype(&wm);
    wm.invalid = 0;
    wm
}

/// `add_param_offset` (global_motion.c:67): step one model parameter, handling
/// the precision shift, the clamp, and the one-centering of the two diagonal
/// entries.
///
/// `param_index` selects both the precision (`< 2` is translation) and whether
/// the parameter is one-centered (`== 2 || == 5`). Those are two *different*
/// predicates over the same index and are easy to conflate.
///
/// This function is `static` in C with no exported symbol; it is gated at
/// tier 1 indirectly, through [`refine_integerized_param`]'s differential
/// against the exported `av1_refine_integerized_param`, which is the only
/// caller.
#[must_use]
pub fn add_param_offset(param_index: usize, param_value: i32, offset: i32) -> i32 {
    let scale_vals = [GM_TRANS_PREC_DIFF, GM_ALPHA_PREC_DIFF];
    let clamp_vals = [GM_TRANS_MAX, GM_ALPHA_MAX];
    let param_type = usize::from(param_index >= 2);
    let is_one_centered = i32::from(param_index == 2 || param_index == 5);

    let mut v =
        (param_value - (is_one_centered << WARPEDMODEL_PREC_BITS)) >> scale_vals[param_type];
    v += offset;
    v = v.clamp(-clamp_vals[param_type], clamp_vals[param_type]);
    v *= 1 << scale_vals[param_type];
    v + (is_one_centered << WARPEDMODEL_PREC_BITS)
}

/// `force_wmtype` (global_motion.c:93): coerce a model to the given
/// transformation type.
///
/// The C switch **falls through** — `IDENTITY` runs the `TRANSLATION` and
/// `ROTZOOM` bodies too, and `TRANSLATION` runs `ROTZOOM`'s. A Rust `match`
/// does not fall through, so this port spells the cascade out; writing it as
/// four independent arms would leave `IDENTITY` with a stale `wmmat[2..6]`.
///
/// Like [`add_param_offset`], this is `static` in C with no exported symbol;
/// it is gated indirectly through [`refine_integerized_param`].
pub fn force_wmtype(wm: &mut WarpedMotionParams, wmtype: u8) {
    let one = 1 << WARPEDMODEL_PREC_BITS;
    if wmtype == IDENTITY {
        wm.wmmat[0] = 0;
        wm.wmmat[1] = 0;
    }
    if wmtype == IDENTITY || wmtype == TRANSLATION {
        wm.wmmat[2] = one;
        wm.wmmat[3] = 0;
    }
    if wmtype == IDENTITY || wmtype == TRANSLATION || wmtype == ROTZOOM {
        wm.wmmat[4] = -wm.wmmat[3];
        wm.wmmat[5] = wm.wmmat[2];
    }
    wm.wmtype = wmtype;
}

/// `av1_compute_feature_segmentation_map` (global_motion.c:483): mark the
/// `WARP_ERROR_BLOCK`-sized cells that contain at least `FEAT_COUNT_TR`
/// inliers of the motion model.
///
/// `inliers` is a flat `[x0, y0, x1, y1, ...]` list in PIXEL coordinates;
/// `width`/`height` are the map dimensions in cells.
///
/// If fewer than `SEG_COUNT_TR` cells survive, C **discards the whole map** and
/// fills it with 1s, so the error metric falls back to the unsegmented form.
/// That fallback is the interesting arm: a port that only implemented the
/// counting would score the wrong pixels on every low-inlier model.
#[must_use]
pub fn compute_feature_segmentation_map(
    width: usize,
    height: usize,
    inliers: &[i32],
    num_inliers: usize,
) -> Vec<u8> {
    let mut map = vec![0u8; width * height];
    for i in 0..num_inliers {
        let x = inliers[i * 2];
        let y = inliers[i * 2 + 1];
        let seg_x = (x >> WARP_ERROR_BLOCK_LOG) as usize;
        let seg_y = (y >> WARP_ERROR_BLOCK_LOG) as usize;
        // C accumulates into a uint8_t, so a cell with >255 inliers wraps.
        let idx = seg_y * width + seg_x;
        map[idx] = map[idx].wrapping_add(1);
    }
    let mut seg_count = 0i32;
    for v in map.iter_mut() {
        *v = u8::from(*v >= FEAT_COUNT_TR);
        seg_count += i32::from(*v);
    }
    if seg_count < SEG_COUNT_TR {
        map.fill(1);
    }
    map
}

/// `generic_sad` (global_motion.c:219) / `generic_sad_highbd` (:114) — the
/// partial-block SAD used at the frame edge. Generic over the pixel type so
/// both bit depths share the loop, as C's two copies do.
#[allow(clippy::too_many_arguments)]
fn generic_sad<T: Copy + Into<i32>>(
    refp: &[T],
    ref_off: usize,
    ref_stride: usize,
    dst: &[T],
    dst_off: usize,
    dst_stride: usize,
    p_width: usize,
    p_height: usize,
) -> i32 {
    let mut sad = 0i32;
    for i in 0..p_height {
        for j in 0..p_width {
            let d: i32 = dst[dst_off + j + i * dst_stride].into();
            let r: i32 = refp[ref_off + j + i * ref_stride].into();
            sad += (d - r).abs();
        }
    }
    sad
}

/// `av1_segmented_frame_error` (global_motion.c:321), lowbd arm, via
/// `segmented_frame_error` (:239): the SAD between `ref` and `dst` over only
/// the `WARP_ERROR_BLOCK` cells the segmentation map marks.
///
/// The `segment_map` index is derived from the ABSOLUTE pixel offset
/// (`j >> WARP_ERROR_BLOCK_LOG`), not from a running cell counter, and the
/// partial blocks at the right/bottom edge are clipped to `p_width - j` /
/// `p_height - i`.
///
/// C splits the inner call in two — `aom_sad32x32` for a full block and
/// `generic_sad` for a partial one — purely so the full case can take a SIMD
/// path. Both compute the same plain SAD, so this port runs one loop; the
/// differential covers both block shapes and confirms the values agree.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn segmented_frame_error(
    refp: &[u8],
    ref_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    p_width: usize,
    p_height: usize,
    segment_map: &[u8],
    segment_map_stride: usize,
) -> i64 {
    let error_bsize_w = p_width.min(WARP_ERROR_BLOCK);
    let error_bsize_h = p_height.min(WARP_ERROR_BLOCK);
    let mut sum_error = 0i64;
    let mut i = 0;
    while i < p_height {
        let mut j = 0;
        while j < p_width {
            let seg_x = j >> WARP_ERROR_BLOCK_LOG;
            let seg_y = i >> WARP_ERROR_BLOCK_LOG;
            if segment_map[seg_y * segment_map_stride + seg_x] != 0 {
                let patch_w = error_bsize_w.min(p_width - j);
                let patch_h = error_bsize_h.min(p_height - i);
                sum_error += i64::from(generic_sad(
                    refp,
                    j + i * ref_stride,
                    ref_stride,
                    dst,
                    j + i * dst_stride,
                    dst_stride,
                    patch_w,
                    patch_h,
                ));
            }
            j += WARP_ERROR_BLOCK;
        }
        i += WARP_ERROR_BLOCK;
    }
    sum_error
}

/// `av1_segmented_frame_error` (global_motion.c:321), highbd arm, via
/// `highbd_segmented_frame_error` (:134). `bd` is accepted and ignored, exactly
/// as C does (`(void)bd;`) — the metric is a plain SAD at any depth.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn highbd_segmented_frame_error(
    refp: &[u16],
    ref_stride: usize,
    dst: &[u16],
    dst_stride: usize,
    p_width: usize,
    p_height: usize,
    _bd: i32,
    segment_map: &[u8],
    segment_map_stride: usize,
) -> i64 {
    let error_bsize_w = p_width.min(WARP_ERROR_BLOCK);
    let error_bsize_h = p_height.min(WARP_ERROR_BLOCK);
    let mut sum_error = 0i64;
    let mut i = 0;
    while i < p_height {
        let mut j = 0;
        while j < p_width {
            let seg_x = j >> WARP_ERROR_BLOCK_LOG;
            let seg_y = i >> WARP_ERROR_BLOCK_LOG;
            if segment_map[seg_y * segment_map_stride + seg_x] != 0 {
                let patch_w = error_bsize_w.min(p_width - j);
                let patch_h = error_bsize_h.min(p_height - i);
                sum_error += i64::from(generic_sad(
                    refp,
                    j + i * ref_stride,
                    ref_stride,
                    dst,
                    j + i * dst_stride,
                    dst_stride,
                    patch_w,
                    patch_h,
                ));
            }
            j += WARP_ERROR_BLOCK;
        }
        i += WARP_ERROR_BLOCK;
    }
    sum_error
}

// ===================================================================
// The warp-error metric and the parameter refinement loop.
// ===================================================================

/// `erroradv_early_tr` (encoder/global_motion.h:92) — the looser threshold the
/// refinement's FIRST error computation is allowed, before any stepping.
const ERRORADV_EARLY_TR: f64 = 0.70;
/// `ERRORADV_BORDER` (global_motion.c:31).
const ERRORADV_BORDER: usize = 0;

/// The frame geometry [`warp_error`] and [`refine_integerized_param`] walk.
/// Planes are `u16` bd8 (`0..=255`), matching the rest of the port.
pub struct WarpErrorParams<'a> {
    /// The reference plane, edge-clamped internally by the warp filter.
    pub refp: &'a [u16],
    pub ref_width: usize,
    pub ref_height: usize,
    pub ref_stride: usize,
    /// The frame being coded.
    pub dst: &'a [u16],
    pub dst_stride: usize,
    pub subsampling_x: usize,
    pub subsampling_y: usize,
    /// The inlier segmentation map from [`compute_feature_segmentation_map`].
    pub segment_map: &'a [u8],
    pub segment_map_stride: usize,
}

/// `warp_error` (global_motion.c:276), lowbd: warp the reference through `wm`
/// one `WARP_ERROR_BLOCK` cell at a time and accumulate the SAD against `dst`,
/// skipping cells the segmentation map clears.
///
/// Two details the differential pins:
/// * the per-cell warp extent is clipped against **`ref_width`/`ref_height`**
///   (`p_col + ref_width - j`), not against `p_width`/`p_height` — which is
///   the opposite of what the sibling [`segmented_frame_error`] does;
/// * the accumulator short-circuits to `i64::MAX` as soon as it exceeds
///   `best_error`, so a caller that passes a tight bound gets a sentinel rather
///   than a total. That early exit is load-bearing: it is how the refinement
///   rejects a model.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn warp_error(
    wm: &WarpedMotionParams,
    p: &WarpErrorParams,
    p_col: usize,
    p_row: usize,
    p_width: usize,
    p_height: usize,
    best_error: i64,
) -> i64 {
    let error_bsize_w = p_width.min(WARP_ERROR_BLOCK);
    let error_bsize_h = p_height.min(WARP_ERROR_BLOCK);
    let mut gm_sumerr = 0i64;
    let mut tmp = vec![0u16; WARP_ERROR_BLOCK * WARP_ERROR_BLOCK];

    let mut i = p_row;
    while i < p_row + p_height {
        let mut j = p_col;
        while j < p_col + p_width {
            let seg_x = j >> WARP_ERROR_BLOCK_LOG;
            let seg_y = i >> WARP_ERROR_BLOCK_LOG;
            if p.segment_map[seg_y * p.segment_map_stride + seg_x] == 0 {
                j += WARP_ERROR_BLOCK;
                continue;
            }
            // C computes these in `int`, so a cell that starts past the
            // reference's right/bottom edge yields a NEGATIVE extent. Both the
            // warp and the SAD are `for (x = start; x < start + extent; ...)`
            // loops, so a negative extent makes each of them empty and the cell
            // contributes zero — it is not skipped by the segment map, it is
            // walked and adds nothing. The port reproduces that outcome
            // explicitly rather than underflowing a `usize`.
            let warp_w_i = (error_bsize_w as i64).min(p_col as i64 + p.ref_width as i64 - j as i64);
            let warp_h_i =
                (error_bsize_h as i64).min(p_row as i64 + p.ref_height as i64 - i as i64);
            if warp_w_i <= 0 || warp_h_i <= 0 {
                j += WARP_ERROR_BLOCK;
                continue;
            }
            let warp_w = warp_w_i as usize;
            let warp_h = warp_h_i as usize;
            aom_dsp::inter::warp::warp_affine(
                &wm.wmmat,
                p.refp,
                p.ref_width,
                p.ref_height,
                p.ref_stride,
                &mut tmp,
                0,
                WARP_ERROR_BLOCK,
                j as i32,
                i as i32,
                warp_w,
                warp_h,
                p.subsampling_x,
                p.subsampling_y,
                wm.alpha,
                wm.beta,
                wm.gamma,
                wm.delta,
            );
            gm_sumerr += i64::from(generic_sad(
                &tmp,
                0,
                WARP_ERROR_BLOCK,
                p.dst,
                j + i * p.dst_stride,
                p.dst_stride,
                warp_w,
                warp_h,
            ));
            if gm_sumerr > best_error {
                return i64::MAX;
            }
            j += WARP_ERROR_BLOCK;
        }
        i += WARP_ERROR_BLOCK;
    }
    gm_sumerr
}

/// `get_warp_error` (global_motion.c:341), lowbd: recompute the model's shear
/// parameters and, if they are representable, take the warp error.
///
/// `av1_get_shear_params` **mutates** `wm` (it writes `alpha`/`beta`/`gamma`/
/// `delta` and can set `invalid`), so this takes `&mut`. A model whose shear
/// params do not resolve scores `i64::MAX`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn get_warp_error(
    wm: &mut WarpedMotionParams,
    p: &WarpErrorParams,
    p_col: usize,
    p_row: usize,
    p_width: usize,
    p_height: usize,
    best_error: i64,
) -> i64 {
    if !aom_dsp::inter::warp::get_shear_params(wm) {
        return i64::MAX;
    }
    warp_error(wm, p, p_col, p_row, p_width, p_height, best_error)
}

/// `av1_refine_integerized_param` (global_motion.c:364), lowbd.
///
/// Coordinate descent over the model parameters: `n_refinements` rounds, each
/// halving the step, and within a round each of the `wmtype`'s coded parameters
/// is probed left, then right, then repeatedly in whichever direction won until
/// the error stops falling.
///
/// `n_refinements == 0` is a distinct early path: it scores the model ONCE
/// against a threshold derived from `gm_erroradv_tr`, and never steps.
/// With refinement, the initial score uses the looser `erroradv_early_tr`
/// (0.70) instead, and a model that fails it is rejected outright with
/// `i64::MAX` — those are two different thresholds and swapping them changes
/// which models survive.
///
/// `wm` is updated in place with the refined model. Returns the best error, or
/// `i64::MAX` for a rejected model.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn refine_integerized_param(
    wm: &mut WarpedMotionParams,
    wmtype: u8,
    p: &WarpErrorParams,
    d_width: usize,
    d_height: usize,
    n_refinements: i32,
    ref_frame_error: i64,
    gm_erroradv_tr: f64,
) -> i64 {
    let border = ERRORADV_BORDER;
    let n_params = MAX_TRANS_MODEL_PARAMS[wmtype as usize];
    let p_width = d_width - 2 * border;
    let p_height = d_height - 2 * border;

    force_wmtype(wm, wmtype);
    wm.wmtype = get_wmtype(wm);

    if n_refinements == 0 {
        // C: (int64_t)lrint(ref_frame_error * gm_erroradv_tr). `lrint` rounds
        // half to EVEN under the default rounding mode, which is neither
        // `f64::round` nor a truncation.
        let selection_threshold =
            (ref_frame_error as f64 * gm_erroradv_tr).round_ties_even() as i64;
        return get_warp_error(
            wm,
            p,
            border,
            border,
            p_width,
            p_height,
            selection_threshold,
        );
    }

    let selection_threshold = (ref_frame_error as f64 * ERRORADV_EARLY_TR).round_ties_even() as i64;
    let mut best_error = get_warp_error(
        wm,
        p,
        border,
        border,
        p_width,
        p_height,
        selection_threshold,
    );
    if best_error > selection_threshold {
        return i64::MAX;
    }

    let mut step = 1i32 << (n_refinements - 1);
    for _ in 0..n_refinements {
        for prm in 0..n_params {
            let mut step_dir = 0i32;
            let curr_param = wm.wmmat[prm];
            let mut best_param = curr_param;

            // Left.
            wm.wmmat[prm] = add_param_offset(prm, curr_param, -step);
            force_wmtype(wm, wmtype);
            let step_error = get_warp_error(wm, p, border, border, p_width, p_height, best_error);
            if step_error < best_error {
                best_error = step_error;
                best_param = wm.wmmat[prm];
                step_dir = -1;
            }

            // Right.
            wm.wmmat[prm] = add_param_offset(prm, curr_param, step);
            force_wmtype(wm, wmtype);
            let step_error = get_warp_error(wm, p, border, border, p_width, p_height, best_error);
            if step_error < best_error {
                best_error = step_error;
                best_param = wm.wmmat[prm];
                step_dir = 1;
            }

            // Keep going in the winning direction while it keeps improving.
            // Note the offset is applied to `best_param`, not to the running
            // value, so the step is always measured from the current best.
            while step_dir != 0 {
                wm.wmmat[prm] = add_param_offset(prm, best_param, step * step_dir);
                force_wmtype(wm, wmtype);
                let step_error =
                    get_warp_error(wm, p, border, border, p_width, p_height, best_error);
                if step_error < best_error {
                    best_error = step_error;
                    best_param = wm.wmmat[prm];
                } else {
                    step_dir = 0;
                }
            }

            wm.wmmat[prm] = best_param;
            force_wmtype(wm, wmtype);
        }
        step >>= 1;
    }

    wm.wmtype = get_wmtype(wm);
    // C asserts av1_get_shear_params succeeds here; the port recomputes and
    // leaves `invalid` set rather than aborting, since only warp-able models
    // reach this point.
    let _ = aom_dsp::inter::warp::get_shear_params(wm);
    best_error
}
