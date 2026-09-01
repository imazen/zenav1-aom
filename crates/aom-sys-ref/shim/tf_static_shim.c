/* Oracle shims for the FILE-STATIC helpers of av1/encoder/temporal_filter.c.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * Fourteen of temporal_filter.c's definitions are exported; the rest — the
 * per-block filter body, the normalizer, the partition decision, the self
 * filter — are `static` and have no address a differential can take. The
 * alternative is hand-derived vectors traced from the C source, which this
 * repo labels tier 4 and ranks last. So this TU compiles libaom's OWN
 * temporal_filter.c, unmodified, with its fourteen exported symbols renamed
 * out of the way, and exposes flat wrappers around the statics.
 *
 * EVIDENCE TIER 1c — the real C source compiled verbatim, as opposed to
 * tier 1's real exported symbol out of the archive. Same technique and same
 * justification as shim/rdopt_shim.c and shim/compound_type_shim.c; read
 * rdopt_shim.c's header for the full argument. The second-compilation gap is
 * closed by measurement: `shim_tfs_*` re-exports temporal_filter.c's own
 * exported functions from THIS TU and
 * `tf_static_shim_tu_matches_archive` in tests/temporal_filter_static_diff.rs
 * asserts they agree with the archive's `av1_*` symbols on random inputs.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`, plus the oracle-wide `-ffp-contract=off`). `-DNDEBUG` is
 * separately mandatory for ABI agreement (DIFFERENTIAL_PLAYBOOK §3a(a)).
 *
 * CONTRACTS.
 *  - No wrapper here reaches a dispatched RTCD kernel: tf_build_predictor and
 *    tf_motion_search (which do) are deliberately NOT wrapped. The Rust side
 *    still calls ref_init() unconditionally, per the playbook's structural rule.
 *  - Buffers are used as the caller gives them; nothing in this file's call
 *    tree is SIMD, so §3a(c)'s alignment contract does not bind. If a wrapper
 *    for tf_build_predictor is ever added, that stops being true.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename temporal_filter.c's fourteen exported symbols so this TU links
 * beside libaom.a. `nm -g` on the object file gives exactly this list. */
#define av1_apply_temporal_filter_c shim_tfs_apply_temporal_filter_c
#define av1_highbd_apply_temporal_filter_c shim_tfs_highbd_apply_temporal_filter_c
#define av1_estimate_noise_from_single_plane_c shim_tfs_estimate_noise_c
#define av1_highbd_estimate_noise_from_single_plane_c shim_tfs_highbd_estimate_noise_c
#define av1_estimate_noise_level shim_tfs_estimate_noise_level
#define av1_check_show_filtered_frame shim_tfs_check_show_filtered_frame
#define av1_is_temporal_filter_on shim_tfs_is_temporal_filter_on
#define av1_temporal_filter shim_tfs_temporal_filter
#define av1_tf_do_filtering_row shim_tfs_tf_do_filtering_row
#define av1_tf_info_alloc shim_tfs_tf_info_alloc
#define av1_tf_info_free shim_tfs_tf_info_free
#define av1_tf_info_reset shim_tfs_tf_info_reset
#define av1_tf_info_filtering shim_tfs_tf_info_filtering
#define av1_tf_info_get_filtered_buf shim_tfs_tf_info_get_filtered_buf

/* --- libaom's own temporal filter, unmodified. --- */
#include "av1/encoder/temporal_filter.c"

/* ======================================================================== *
 * 1. tf_determine_block_partition (:465) — pure integer decision.
 *
 * subblock_mvs / subblock_mses are IN/OUT: C overwrites the entries it decides
 * not to split. MVs cross as int16_t[2] = {row, col} pairs.
 * ======================================================================== */
void shim_tfs_determine_block_partition(int16_t block_mv_row,
                                        int16_t block_mv_col, int block_mse,
                                        const int16_t *midblock_mvs,
                                        const int *midblock_mses,
                                        int16_t *subblock_mvs,
                                        int *subblock_mses) {
  MV bmv = { block_mv_row, block_mv_col };
  MV mid[4];
  MV sub[NUM_16X16];
  for (int i = 0; i < 4; ++i) {
    mid[i].row = midblock_mvs[2 * i];
    mid[i].col = midblock_mvs[2 * i + 1];
  }
  for (int i = 0; i < NUM_16X16; ++i) {
    sub[i].row = subblock_mvs[2 * i];
    sub[i].col = subblock_mvs[2 * i + 1];
  }
  tf_determine_block_partition(bmv, block_mse, mid, midblock_mses, sub,
                              subblock_mses);
  for (int i = 0; i < NUM_16X16; ++i) {
    subblock_mvs[2 * i] = sub[i].row;
    subblock_mvs[2 * i + 1] = sub[i].col;
  }
}

/* ======================================================================== *
 * 2. tf_apply_temporal_filter_self (:641) — the reference frame filtering
 *    itself, at a fixed weight of TF_WEIGHT_SCALE.
 *
 * Note it reads `is_cur_buf_hbd(mbd)` (xd->cur_buf), NOT the ref_frame flags
 * the other kernels use, so the shim sets BOTH from the same `highbd`.
 * ======================================================================== */
void shim_tfs_apply_temporal_filter_self(const void *y, const void *u,
                                         const void *v, int y_stride,
                                         int uv_stride, int highbd,
                                         int block_size, int mb_row, int mb_col,
                                         int num_planes,
                                         const int *subsampling_x,
                                         const int *subsampling_y, int bd,
                                         uint32_t *accum, uint16_t *count) {
  YV12_BUFFER_CONFIG *buf = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(*buf));
  YV12_BUFFER_CONFIG *cur = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(*cur));
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  if (!buf || !cur || !xd) {
    free(buf);
    free(cur);
    free(xd);
    return;
  }
  buf->flags = highbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  buf->strides[0] = y_stride;
  buf->strides[1] = uv_stride;
  buf->buffers[0] = highbd ? CONVERT_TO_BYTEPTR(y) : (uint8_t *)y;
  buf->buffers[1] = highbd ? CONVERT_TO_BYTEPTR(u) : (uint8_t *)u;
  buf->buffers[2] = highbd ? CONVERT_TO_BYTEPTR(v) : (uint8_t *)v;
  cur->flags = buf->flags;
  xd->cur_buf = cur;
  xd->bd = bd;
  for (int p = 0; p < MAX_MB_PLANE; ++p) {
    xd->plane[p].subsampling_x = p < num_planes ? subsampling_x[p] : 0;
    xd->plane[p].subsampling_y = p < num_planes ? subsampling_y[p] : 0;
  }
  tf_apply_temporal_filter_self(buf, xd, (BLOCK_SIZE)block_size, mb_row, mb_col,
                                num_planes, accum, count);
  free(buf);
  free(cur);
  free(xd);
}

/* ======================================================================== *
 * 3. tf_normalize_filtered_frame (:995) — accum/count -> pixels, via OD_DIVU.
 *
 * The result buffer is written IN PLACE, at the block's frame offset, so the
 * caller passes whole planes and reads them back.
 * ======================================================================== */
void shim_tfs_normalize_filtered_frame(void *y, void *u, void *v, int y_stride,
                                       int uv_stride, int highbd,
                                       int block_size, int mb_row, int mb_col,
                                       int num_planes,
                                       const int *subsampling_x,
                                       const int *subsampling_y, int bd,
                                       const uint32_t *accum,
                                       const uint16_t *count) {
  YV12_BUFFER_CONFIG *buf = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(*buf));
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  if (!buf || !xd) {
    free(buf);
    free(xd);
    return;
  }
  buf->flags = highbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  buf->strides[0] = y_stride;
  buf->strides[1] = uv_stride;
  buf->buffers[0] = highbd ? CONVERT_TO_BYTEPTR(y) : (uint8_t *)y;
  buf->buffers[1] = highbd ? CONVERT_TO_BYTEPTR(u) : (uint8_t *)u;
  buf->buffers[2] = highbd ? CONVERT_TO_BYTEPTR(v) : (uint8_t *)v;
  xd->bd = bd;
  for (int p = 0; p < MAX_MB_PLANE; ++p) {
    xd->plane[p].subsampling_x = p < num_planes ? subsampling_x[p] : 0;
    xd->plane[p].subsampling_y = p < num_planes ? subsampling_y[p] : 0;
  }
  tf_normalize_filtered_frame(xd, (BLOCK_SIZE)block_size, mb_row, mb_col,
                              num_planes, accum, count, buf);
  free(buf);
  free(xd);
}

/* ======================================================================== *
 * 4. is_frame_high_bitdepth (:520) — one flag test, wrapped so the port's
 *    type-level version is measured rather than assumed.
 * ======================================================================== */
int shim_tfs_is_frame_high_bitdepth(int flags) {
  YV12_BUFFER_CONFIG buf;
  memset(&buf, 0, sizeof(buf));
  buf.flags = flags;
  return is_frame_high_bitdepth(&buf);
}

/* ======================================================================== *
 * 5. The TU-vs-archive gate: these forward to THIS TU's copies of two
 *    exported functions, so the test can assert the second compilation still
 *    means what the archive's copy means.
 * ======================================================================== */
double shim_tfs_tu_estimate_noise(const uint8_t *src, int height, int width,
                                  int stride, int edge_thresh) {
  return shim_tfs_estimate_noise_c(src, height, width, stride, edge_thresh);
}

int shim_tfs_tu_is_temporal_filter_on(int arnr_max_frames, int lag_in_frames) {
  AV1EncoderConfig oxcf;
  memset(&oxcf, 0, sizeof(oxcf));
  oxcf.algo_cfg.arnr_max_frames = arnr_max_frames;
  oxcf.gf_cfg.lag_in_frames = lag_in_frames;
  return shim_tfs_is_temporal_filter_on(&oxcf);
}

/* ======================================================================== *
 * 6. av1_check_show_filtered_frame (:1591) — exported, but takes a
 *    YV12_BUFFER_CONFIG and a FRAME_DIFF, so it is wrapped here too. This
 *    wrapper calls THIS TU's copy; the archive's copy is reached separately
 *    from tf_shim.c, and the two are compared.
 * ======================================================================== */
int shim_tfs_check_show(int y_crop_width, int y_crop_height, int64_t diff_sum,
                        int64_t diff_sse, int q_index, int bit_depth,
                        int enable_overlay, int is_second_arf) {
  YV12_BUFFER_CONFIG buf;
  FRAME_DIFF fd;
  memset(&buf, 0, sizeof(buf));
  buf.y_crop_width = y_crop_width;
  buf.y_crop_height = y_crop_height;
  fd.sum = diff_sum;
  fd.sse = diff_sse;
  return shim_tfs_check_show_filtered_frame(&buf, &fd, q_index,
                                            (aom_bit_depth_t)bit_depth,
                                            enable_overlay, is_second_arf);
}
