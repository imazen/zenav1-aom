/* Oracle shims for the FILE-STATIC helpers of av1/encoder/var_based_part.c.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * `nm -g` on the object file reports exactly four exported symbols for the
 * whole file (av1_choose_var_based_partitioning,
 * av1_set_variance_partition_thresholds and the two
 * av1_get_force_skip_low_temp_var lookups). The INTER leaf fill, the min/max
 * spread, the content threshold scale, the MV distance and the zero-MV skip
 * gate are all static and have no address a differential can take. The
 * alternative is hand-derived vectors, which this repo labels tier 4 and ranks
 * last -- and worse, re-deriving the expected value in the TEST is comparing
 * the port against a second transcription of the same logic, which proves
 * nothing about either.
 *
 * EVIDENCE TIER 1c -- the real C source compiled verbatim. Same technique and
 * same justification as shim/rdopt_shim.c; read that file's header for the
 * full argument. The second-compilation gap is closed by measurement:
 * `shim_vbps_tu_force_skip_*` re-export var_based_part.c's own exported
 * lookups from THIS TU, and `vbp_static_shim_tu_matches_archive` in
 * tests/var_part_inter_diff.rs asserts they agree with the archive's.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`, plus the oracle-wide `-ffp-contract=off`). `-DNDEBUG` is
 * separately mandatory for ABI agreement (DIFFERENTIAL_PLAYBOOK §3a(a)) and
 * doubly so here: av1_get_force_skip_low_temp_var asserts on its own mi
 * alignment for two of the block sizes.
 *
 * RTCD. fill_variance_8x8avg reaches aom_avg_8x8_quad / aom_avg_8x8, and
 * compute_minmax_8x8 reaches aom_minmax_8x8 -- all DISPATCHED, i.e. real
 * RTCD_EXTERN pointers on x86 that are NULL until aom_dsp_rtcd() runs
 * (DIFFERENTIAL_PLAYBOOK §3a(b)). Every ref_* wrapper on the Rust side calls
 * ref_init() unconditionally, which is what makes that safe.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename var_based_part.c's four exported symbols. --- */
#define av1_choose_var_based_partitioning shim_vbps_choose_var_based_partitioning
#define av1_set_variance_partition_thresholds shim_vbps_set_variance_partition_thresholds
#define av1_get_force_skip_low_temp_var shim_vbps_force_skip_low_temp_var
#define av1_get_force_skip_low_temp_var_small_sb shim_vbps_force_skip_low_temp_var_small_sb

/* --- libaom's own variance-based partitioner, unmodified. --- */
#include "av1/encoder/var_based_part.c"

int shim_vbps_all_blks_inside(int x16_idx, int y16_idx, int pixels_wide,
                              int pixels_high) {
  return all_blks_inside(x16_idx, y16_idx, pixels_wide, pixels_high);
}

/* fill_variance_8x8avg writes four VPartVar records; the shim returns the two
 * fields fill_variance() actually sets (log2_count is always 0 here). */
void shim_vbps_fill_variance_8x8avg(const void *src, int src_stride,
                                    const void *dst, int dst_stride,
                                    int x16_idx, int y16_idx, int highbd,
                                    int pixels_wide, int pixels_high,
                                    uint32_t *sse_out, int32_t *sum_out) {
  VP16x16 *vst = (VP16x16 *)calloc(1, sizeof(*vst));
  if (!vst) return;
  const uint8_t *s = highbd ? CONVERT_TO_BYTEPTR(src) : (const uint8_t *)src;
  const uint8_t *d = highbd ? CONVERT_TO_BYTEPTR(dst) : (const uint8_t *)dst;
  fill_variance_8x8avg(s, src_stride, d, dst_stride, x16_idx, y16_idx, vst,
                       highbd ? YV12_FLAG_HIGHBITDEPTH : 0, pixels_wide,
                       pixels_high);
  for (int i = 0; i < 4; ++i) {
    sse_out[i] = vst->split[i].part_variances.none.sum_square_error;
    sum_out[i] = vst->split[i].part_variances.none.sum_error;
  }
  free(vst);
}

int shim_vbps_compute_minmax_8x8(const void *src, int src_stride,
                                 const void *dst, int dst_stride, int x16_idx,
                                 int y16_idx, int highbd, int pixels_wide,
                                 int pixels_high) {
  const uint8_t *s = highbd ? CONVERT_TO_BYTEPTR(src) : (const uint8_t *)src;
  const uint8_t *d = highbd ? CONVERT_TO_BYTEPTR(dst) : (const uint8_t *)dst;
  return compute_minmax_8x8(s, src_stride, d, dst_stride, x16_idx, y16_idx,
#if CONFIG_AV1_HIGHBITDEPTH
                            highbd ? YV12_FLAG_HIGHBITDEPTH : 0,
#endif
                            pixels_wide, pixels_high);
}

int64_t shim_vbps_scale_part_thresh_content(int64_t threshold_base, int speed,
                                            int non_reference_frame,
                                            int is_static) {
  return scale_part_thresh_content(threshold_base, speed, non_reference_frame,
                                   is_static);
}

int shim_vbps_mv_distance(int16_t r0, int16_t c0, int16_t r1, int16_t c1) {
  FULLPEL_MV a = { r0, c0 };
  FULLPEL_MV b = { r1, c1 };
  return mv_distance(&a, &b);
}

int shim_vbps_is_set_force_zeromv_skip_based_on_src_sad(int level,
                                                        int source_sad_nonrd) {
  return is_set_force_zeromv_skip_based_on_src_sad(
             level, (SOURCE_SAD)source_sad_nonrd)
             ? 1
             : 0;
}

/* The TU-vs-archive gate: these forward to THIS TU's copies. */
int shim_vbps_tu_force_skip_low_temp_var(const uint8_t *variance_low,
                                         int mi_row, int mi_col, int bsize) {
  return shim_vbps_force_skip_low_temp_var(variance_low, mi_row, mi_col,
                                           (BLOCK_SIZE)bsize);
}

int shim_vbps_tu_force_skip_low_temp_var_small_sb(const uint8_t *variance_low,
                                                  int mi_row, int mi_col,
                                                  int bsize) {
  return shim_vbps_force_skip_low_temp_var_small_sb(variance_low, mi_row,
                                                    mi_col, (BLOCK_SIZE)bsize);
}
