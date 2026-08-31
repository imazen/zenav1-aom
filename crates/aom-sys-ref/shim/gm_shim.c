/* Oracle shims for the self-contained half of av1/encoder/global_motion.c
 * (crate aom-encode, module global_motion). Oracle use only.
 *
 * All four entries drive REAL exported C functions. `get_wmtype` is a static
 * inline in av1/common/mv.h; the shim calls it from a translation unit that
 * includes the same header under the same oracle flags, so it is still the C
 * definition rather than a second transcription.
 *
 * NOT shimmed, because C gives them no linkable symbol: `add_param_offset` and
 * `force_wmtype` are file-static in global_motion.c and are reachable only
 * through av1_refine_integerized_param, which the port does not yet have.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"
#include "av1/common/mv.h"
#include "av1/common/warped_motion.h"
#include "av1/encoder/global_motion.h"

int shim_is_enough_erroradvantage(double best_erroradvantage, int params_cost,
                                  double gm_erroradv_tr) {
  return av1_is_enough_erroradvantage(best_erroradvantage, params_cost,
                                      gm_erroradv_tr);
}

/* Returns the six wmmat entries plus the derived wmtype. */
void shim_convert_model_to_params(const double *params, int32_t *out_wmmat,
                                  int *out_wmtype, int *out_invalid) {
  WarpedMotionParams wm;
  memset(&wm, 0, sizeof(wm));
  av1_convert_model_to_params(params, &wm);
  memcpy(out_wmmat, wm.wmmat, 6 * sizeof(int32_t));
  *out_wmtype = (int)wm.wmtype;
  *out_invalid = (int)wm.invalid;
}

int shim_get_wmtype(const int32_t *wmmat) {
  WarpedMotionParams wm;
  memset(&wm, 0, sizeof(wm));
  memcpy(wm.wmmat, wmmat, 6 * sizeof(int32_t));
  return (int)get_wmtype(&wm);
}

void shim_compute_feature_segmentation_map(uint8_t *segment_map, int width,
                                           int height, const int *inliers,
                                           int num_inliers) {
  /* C takes a mutable int* it never writes; copy to satisfy the signature
   * without letting the oracle mutate the caller's slice. */
  int *tmp = NULL;
  if (num_inliers > 0) {
    tmp = (int *)malloc((size_t)num_inliers * 2 * sizeof(int));
    if (!tmp) return;
    memcpy(tmp, inliers, (size_t)num_inliers * 2 * sizeof(int));
  }
  av1_compute_feature_segmentation_map(segment_map, width, height, tmp,
                                       num_inliers);
  free(tmp);
}

int64_t shim_segmented_frame_error(int use_hbd, int bd, const void *ref,
                                   int ref_stride, void *dst, int dst_stride,
                                   int p_width, int p_height,
                                   uint8_t *segment_map,
                                   int segment_map_stride) {
  if (use_hbd) {
    return av1_segmented_frame_error(
        1, bd, CONVERT_TO_BYTEPTR((const uint16_t *)ref), ref_stride,
        CONVERT_TO_BYTEPTR((uint16_t *)dst), dst_stride, p_width, p_height,
        segment_map, segment_map_stride);
  }
  return av1_segmented_frame_error(0, bd, (const uint8_t *)ref, ref_stride,
                                   (uint8_t *)dst, dst_stride, p_width,
                                   p_height, segment_map, segment_map_stride);
}

/* ---- shim_refine_integerized_param -----------------------------------
 * Drives the REAL exported av1_refine_integerized_param (global_motion.c:364)
 * on the lowbd path. This is the only entry point that reaches the file-static
 * add_param_offset / force_wmtype / warp_error / get_warp_error, so it is what
 * gates them.
 *
 * `wmmat` is in/out: the caller's starting model goes in and the refined model
 * comes back. `alpha/beta/gamma/delta` come back too, since
 * av1_get_shear_params writes them.
 */
#include "av1/encoder/encoder.h"

int64_t shim_refine_integerized_param(
    int32_t *wmmat, int wmtype, const uint8_t *ref, int r_width, int r_height,
    int r_stride, uint8_t *dst, int d_width, int d_height, int d_stride,
    int n_refinements, int64_t ref_frame_error, uint8_t *segment_map,
    int segment_map_stride, double gm_erroradv_tr, int *out_wmtype,
    int16_t *out_shear, int *out_invalid) {
  WarpedMotionParams wm;
  memset(&wm, 0, sizeof(wm));
  memcpy(wm.wmmat, wmmat, 6 * sizeof(int32_t));
  wm.wmtype = (TransformationType)wmtype;

  int64_t err = av1_refine_integerized_param(
      &wm, (TransformationType)wmtype, /*use_hbd=*/0, /*bd=*/8, (uint8_t *)ref,
      r_width, r_height, r_stride, dst, d_width, d_height, d_stride,
      n_refinements, ref_frame_error, segment_map, segment_map_stride,
      gm_erroradv_tr);

  memcpy(wmmat, wm.wmmat, 6 * sizeof(int32_t));
  *out_wmtype = (int)wm.wmtype;
  out_shear[0] = wm.alpha;
  out_shear[1] = wm.beta;
  out_shear[2] = wm.gamma;
  out_shear[3] = wm.delta;
  *out_invalid = (int)wm.invalid;
  return err;
}
