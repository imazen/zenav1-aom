/* Oracle shims for the inter-encoder motion search (crate aom-encode,
 * INTER-ENCODE chunk 2d). Oracle use only.
 *
 *  - shim_upsampled_pred wraps the REAL libaom `aom_upsampled_pred_c`
 *    (av1/encoder/reconinter_enc.c:462) for the lowbd, unscaled,
 *    USE_8_TAPS (EIGHTTAP_REGULAR) subpel-predictor path used by the speed-0
 *    subpel motion search (`av1_find_best_sub_pixel_tree` ->
 *    `upsampled_pref_error`). `xd == NULL` takes the unscaled branch directly
 *    (`aom_upsampled_pred_scaled` returns false for a NULL xd — "expect xd ==
 *    NULL only in tests", reconinter_enc.c:427-428), so cm/mi_row/mi_col/mv
 *    are unused and the output is purely the fixed-phase 8-tap convolution of
 *    the reference at (subpel_x_q3, subpel_y_q3).
 */
#include <stdint.h>
#include "config/aom_dsp_rtcd.h"
#include "av1/common/filter.h"
#include "av1/encoder/reconinter_enc.h"

void shim_upsampled_pred(const uint8_t *ref, int ref_stride, int width,
                         int height, int subpel_x_q3, int subpel_y_q3,
                         uint8_t *dst) {
  MV mv = { 0, 0 };
  aom_upsampled_pred_c(NULL, NULL, 0, 0, &mv, dst, width, height, subpel_x_q3,
                       subpel_y_q3, ref, ref_stride, USE_8_TAPS);
}
