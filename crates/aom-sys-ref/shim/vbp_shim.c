/* Oracle shims for the variance-based partitioner's INTER surface.
 *
 * TWO EVIDENCE TIERS, kept apart on purpose:
 *
 * TIER 1 (this file's first half) — the aom_dsp/avg.c block-average and
 * min/max kernels, and av1/encoder/var_based_part.c's two exported
 * force-skip lookups. All seven are exported from libaom.a and are called
 * here by their real names. The `_c` suffix is used deliberately: the port
 * reproduces the scalar arm, and the dispatched name would select NEON/SSE2
 * (see DIFFERENTIAL_PLAYBOOK §3a(b) on why "the dispatched one is the same
 * function" is not safe to assume).
 *
 *   aom_avg_4x4_c, aom_avg_8x8_c, aom_avg_8x8_quad_c, aom_minmax_8x8_c,
 *   aom_highbd_avg_4x4_c, aom_highbd_avg_8x8_c, aom_highbd_minmax_8x8_c
 *   av1_get_force_skip_low_temp_var, av1_get_force_skip_low_temp_var_small_sb
 *
 * TIER 1c (vbp_static_shim.c) — the file-statics, reached by compiling
 * var_based_part.c into that TU.
 *
 * CONTRACT. aom_avg_8x8_quad's x16/y16 indices are ABSOLUTE plane
 * coordinates, not an offset applied by the caller; the shim passes them
 * through unchanged so the port's identical shape is what is measured.
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
#include "av1/encoder/var_based_part.h"

unsigned int shim_vbp_avg_4x4(const uint8_t *s, int p) {
  return aom_avg_4x4_c(s, p);
}

unsigned int shim_vbp_avg_8x8(const uint8_t *s, int p) {
  return aom_avg_8x8_c(s, p);
}

void shim_vbp_avg_8x8_quad(const uint8_t *s, int p, int x16_idx, int y16_idx,
                           int *avg) {
  aom_avg_8x8_quad_c(s, p, x16_idx, y16_idx, avg);
}

void shim_vbp_minmax_8x8(const uint8_t *s, int p, const uint8_t *d, int dp,
                         int *min, int *max) {
  aom_minmax_8x8_c(s, p, d, dp, min, max);
}

#if CONFIG_AV1_HIGHBITDEPTH
unsigned int shim_vbp_highbd_avg_4x4(const uint16_t *s, int p) {
  return aom_highbd_avg_4x4_c(CONVERT_TO_BYTEPTR(s), p);
}

unsigned int shim_vbp_highbd_avg_8x8(const uint16_t *s, int p) {
  return aom_highbd_avg_8x8_c(CONVERT_TO_BYTEPTR(s), p);
}

void shim_vbp_highbd_minmax_8x8(const uint16_t *s, int p, const uint16_t *d,
                                int dp, int *min, int *max) {
  aom_highbd_minmax_8x8_c(CONVERT_TO_BYTEPTR(s), p, CONVERT_TO_BYTEPTR(d), dp,
                          min, max);
}
#endif

int shim_vbp_force_skip_low_temp_var(const uint8_t *variance_low, int mi_row,
                                     int mi_col, int bsize) {
  return av1_get_force_skip_low_temp_var(variance_low, mi_row, mi_col,
                                         (BLOCK_SIZE)bsize);
}

int shim_vbp_force_skip_low_temp_var_small_sb(const uint8_t *variance_low,
                                              int mi_row, int mi_col,
                                              int bsize) {
  return av1_get_force_skip_low_temp_var_small_sb(variance_low, mi_row, mi_col,
                                                  (BLOCK_SIZE)bsize);
}
