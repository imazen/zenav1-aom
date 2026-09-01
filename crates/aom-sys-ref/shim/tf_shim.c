/* Oracle shims for av1/encoder/temporal_filter.c — the alt-ref temporal
 * filter's pixel kernels and its noise estimator.
 *
 * EVIDENCE TIER 1 throughout. Every entry below drives the REAL exported C
 * symbol out of libaom.a; nothing here re-compiles libaom source. The four
 * functions the port needs are all exported:
 *
 *   av1_estimate_noise_from_single_plane_c        (temporal_filter.c:1426)
 *   av1_highbd_estimate_noise_from_single_plane_c (:1465)
 *   av1_estimate_noise_level                      (:1505)
 *   av1_apply_temporal_filter_c                   (:795)
 *   av1_highbd_apply_temporal_filter_c            (:964)
 *
 * The first two take plain pointers and could be declared straight from Rust;
 * they are wrapped here anyway so that the `_c` suffix (the scalar arm, which
 * is what the port reproduces) is pinned in one place rather than at each call
 * site, and so that the two arms share this file's header contract.
 *
 * The last three need a YV12_BUFFER_CONFIG and a MACROBLOCKD, which is the
 * only reason a shim exists at all. Both are built here from flat arguments.
 *
 * CONTRACTS THE CALLER MUST HAND US (DIFFERENTIAL_PLAYBOOK §3a):
 *  - `mbd->error_info` is dereferenced by av1_apply_temporal_filter_c only on
 *    an allocation failure, but it is set to a real object anyway so that a
 *    failure reports rather than faulting.
 *  - av1_apply_temporal_filter_c is a pure C function (no RTCD dispatch in its
 *    body), but ref_init() is called unconditionally on the Rust side per the
 *    playbook's structural rule.
 *  - No buffer here is read by a SIMD kernel, so the alignment/size contract of
 *    §3a(c) does not bind; the caller's plane slices are used as given and the
 *    shim allocates its own YV12/MACROBLOCKD on the heap (both are large).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/encoder/block.h"
/* temporal_filter.h needs MAX_LAG_BUFFERS (lookahead.h) and GF_GROUP
 * (firstpass.h) from its own TU's include order; pull them first. */
#include "av1/encoder/lookahead.h"
#include "av1/encoder/firstpass.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/temporal_filter.h"

double shim_tf_estimate_noise_lowbd(const uint8_t *src, int height, int width,
                                    int stride, int edge_thresh) {
  return av1_estimate_noise_from_single_plane_c(src, height, width, stride,
                                                edge_thresh);
}

#if CONFIG_AV1_HIGHBITDEPTH
double shim_tf_estimate_noise_highbd(const uint16_t *src, int height, int width,
                                     int stride, int bit_depth,
                                     int edge_thresh) {
  return av1_highbd_estimate_noise_from_single_plane_c(
      src, height, width, stride, bit_depth, edge_thresh);
}
#endif

/* ---- av1_estimate_noise_level ---------------------------------------------
 * Fills noise_level[plane_from..plane_to] from a frame the shim assembles over
 * the caller's three plane pointers. `highbd` selects the uint16 arm, which is
 * the YV12_FLAG_HIGHBITDEPTH the real encoder sets.
 */
void shim_tf_estimate_noise_level(const void *y, const void *u, const void *v,
                                  int y_stride, int uv_stride, int y_w,
                                  int y_h, int uv_w, int uv_h, int highbd,
                                  int plane_from, int plane_to, int bit_depth,
                                  int edge_thresh, double *noise_level) {
  YV12_BUFFER_CONFIG buf;
  memset(&buf, 0, sizeof(buf));
  buf.flags = highbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  buf.crop_widths[0] = y_w;
  buf.crop_widths[1] = uv_w;
  buf.crop_heights[0] = y_h;
  buf.crop_heights[1] = uv_h;
  buf.strides[0] = y_stride;
  buf.strides[1] = uv_stride;
  buf.buffers[0] = highbd ? CONVERT_TO_BYTEPTR(y) : (uint8_t *)y;
  buf.buffers[1] = highbd ? CONVERT_TO_BYTEPTR(u) : (uint8_t *)u;
  buf.buffers[2] = highbd ? CONVERT_TO_BYTEPTR(v) : (uint8_t *)v;
  av1_estimate_noise_level(&buf, noise_level, plane_from, plane_to, bit_depth,
                           edge_thresh);
}

/* ---- av1_apply_temporal_filter_c / av1_highbd_apply_temporal_filter_c ------
 * `subblock_mvs` arrives as NUM_16X16 (row, col) int16 pairs — never a packed
 * as_int — per the rdopt_shim convention.
 *
 * `use_highbd_entry` selects which of the two exported symbols is called. They
 * are not interchangeable at the API level even though the highbd one only
 * forwards: driving both is what proves the forward is still a forward.
 */
void shim_tf_apply_temporal_filter(
    const void *y, const void *u, const void *v, int y_stride, int uv_stride,
    int y_crop_w, int y_crop_h, int uv_crop_w, int uv_crop_h, int highbd,
    int block_size, int mb_row, int mb_col, int num_planes,
    const int *subsampling_x, const int *subsampling_y, int bd,
    const double *noise_levels, const int16_t *subblock_mvs,
    const int *subblock_mses, int q_factor, int filter_strength,
    int tf_wgt_calc_lvl, const void *pred, uint32_t *accum, uint16_t *count,
    int use_highbd_entry) {
  YV12_BUFFER_CONFIG *buf = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(*buf));
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  struct aom_internal_error_info *err =
      (struct aom_internal_error_info *)calloc(1, sizeof(*err));
  MV mvs[NUM_16X16];
  if (!buf || !xd || !err) {
    free(buf);
    free(xd);
    free(err);
    return;
  }

  buf->flags = highbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  buf->crop_widths[0] = y_crop_w;
  buf->crop_widths[1] = uv_crop_w;
  buf->crop_heights[0] = y_crop_h;
  buf->crop_heights[1] = uv_crop_h;
  buf->y_crop_width = y_crop_w;
  buf->y_crop_height = y_crop_h;
  buf->strides[0] = y_stride;
  buf->strides[1] = uv_stride;
  buf->buffers[0] = highbd ? CONVERT_TO_BYTEPTR(y) : (uint8_t *)y;
  buf->buffers[1] = highbd ? CONVERT_TO_BYTEPTR(u) : (uint8_t *)u;
  buf->buffers[2] = highbd ? CONVERT_TO_BYTEPTR(v) : (uint8_t *)v;

  xd->bd = bd;
  xd->error_info = err;
  for (int p = 0; p < MAX_MB_PLANE; ++p) {
    xd->plane[p].subsampling_x = p < num_planes ? subsampling_x[p] : 0;
    xd->plane[p].subsampling_y = p < num_planes ? subsampling_y[p] : 0;
  }

  for (int i = 0; i < NUM_16X16; ++i) {
    mvs[i].row = subblock_mvs[2 * i];
    mvs[i].col = subblock_mvs[2 * i + 1];
  }

  const uint8_t *pred8 =
      highbd ? CONVERT_TO_BYTEPTR((const uint16_t *)pred) : (const uint8_t *)pred;

#if CONFIG_AV1_HIGHBITDEPTH
  if (use_highbd_entry) {
    av1_highbd_apply_temporal_filter_c(
        buf, xd, (BLOCK_SIZE)block_size, mb_row, mb_col, num_planes,
        noise_levels, mvs, subblock_mses, q_factor, filter_strength,
        tf_wgt_calc_lvl, pred8, accum, count);
  } else
#else
  (void)use_highbd_entry;
#endif
  {
    av1_apply_temporal_filter_c(buf, xd, (BLOCK_SIZE)block_size, mb_row, mb_col,
                                num_planes, noise_levels, mvs, subblock_mses,
                                q_factor, filter_strength, tf_wgt_calc_lvl,
                                pred8, accum, count);
  }

  free(buf);
  free(xd);
  free(err);
}

/* ---- av1_check_show_filtered_frame / av1_is_temporal_filter_on -------------
 * Both exported, both taking a struct, so both wrapped. TIER 1: these call
 * the ARCHIVE's copies. `tf_static_shim.c` exposes the same two functions as
 * compiled into ITS TU, and the test compares the pair — that is what keeps
 * the tier-1c shim's second compilation honest.
 */
int shim_tf_check_show_archive(int y_crop_width, int y_crop_height,
                               int64_t diff_sum, int64_t diff_sse, int q_index,
                               int bit_depth, int enable_overlay,
                               int is_second_arf) {
  YV12_BUFFER_CONFIG buf;
  FRAME_DIFF fd;
  memset(&buf, 0, sizeof(buf));
  buf.y_crop_width = y_crop_width;
  buf.y_crop_height = y_crop_height;
  fd.sum = diff_sum;
  fd.sse = diff_sse;
  return av1_check_show_filtered_frame(&buf, &fd, q_index,
                                       (aom_bit_depth_t)bit_depth,
                                       enable_overlay, is_second_arf);
}

int shim_tf_is_temporal_filter_on_archive(int arnr_max_frames,
                                          int lag_in_frames) {
  AV1EncoderConfig *oxcf = (AV1EncoderConfig *)calloc(1, sizeof(*oxcf));
  if (!oxcf) return -1;
  oxcf->algo_cfg.arnr_max_frames = arnr_max_frames;
  oxcf->gf_cfg.lag_in_frames = lag_in_frames;
  const int r = av1_is_temporal_filter_on(oxcf);
  free(oxcf);
  return r;
}
