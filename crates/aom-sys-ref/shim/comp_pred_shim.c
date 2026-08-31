/* Oracle shims for the encoder-side compound / high-bit-depth predictor
 * construction (crate aom-encode, module inter_pred_enc). Oracle use only.
 *
 * Every entry drives the REAL exported C function. `xd == NULL` takes the
 * unscaled branch of aom_upsampled_pred_scaled ("expect xd == NULL only in
 * tests", reconinter_enc.c:427-428), which is why cm/mi_row/mi_col/mv are
 * unused here — the same construction shim_upsampled_pred already relies on.
 */
#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"
#include "av1/common/reconinter.h"
#include "av1/common/entropymv.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/reconinter_enc.h"

/* ---- aom_dsp/variance.c: the two compound blends ------------------- */

void shim_comp_avg_pred(uint8_t *comp_pred, const uint8_t *pred, int width,
                        int height, const uint8_t *ref, int ref_stride) {
  aom_comp_avg_pred_c(comp_pred, pred, width, height, ref, ref_stride);
}

void shim_comp_mask_pred(uint8_t *comp_pred, const uint8_t *pred, int width,
                         int height, const uint8_t *ref, int ref_stride,
                         const uint8_t *mask, int mask_stride,
                         int invert_mask) {
  aom_comp_mask_pred_c(comp_pred, pred, width, height, ref, ref_stride, mask,
                       mask_stride, invert_mask);
}

void shim_highbd_comp_avg_pred(uint16_t *comp_pred, const uint16_t *pred,
                               int width, int height, const uint16_t *ref,
                               int ref_stride) {
  aom_highbd_comp_avg_pred_c(CONVERT_TO_BYTEPTR(comp_pred),
                             CONVERT_TO_BYTEPTR(pred), width, height,
                             CONVERT_TO_BYTEPTR(ref), ref_stride);
}

void shim_highbd_comp_mask_pred(uint16_t *comp_pred, const uint16_t *pred,
                                int width, int height, const uint16_t *ref,
                                int ref_stride, const uint8_t *mask,
                                int mask_stride, int invert_mask) {
  aom_highbd_comp_mask_pred_c(CONVERT_TO_BYTEPTR(comp_pred),
                              CONVERT_TO_BYTEPTR(pred), width, height,
                              CONVERT_TO_BYTEPTR(ref), ref_stride, mask,
                              mask_stride, invert_mask);
}

/* ---- reconinter_enc.c: the upsampled predictors -------------------- */

void shim_comp_avg_upsampled_pred(uint8_t *comp_pred, const uint8_t *pred,
                                  int width, int height, int subpel_x_q3,
                                  int subpel_y_q3, const uint8_t *ref,
                                  int ref_stride) {
  MV mv = { 0, 0 };
  aom_comp_avg_upsampled_pred_c(NULL, NULL, 0, 0, &mv, comp_pred, pred, width,
                                height, subpel_x_q3, subpel_y_q3, ref,
                                ref_stride, USE_8_TAPS);
}

void shim_comp_mask_upsampled_pred(uint8_t *comp_pred, const uint8_t *pred,
                                   int width, int height, int subpel_x_q3,
                                   int subpel_y_q3, const uint8_t *ref,
                                   int ref_stride, const uint8_t *mask,
                                   int mask_stride, int invert_mask) {
  MV mv = { 0, 0 };
  aom_comp_mask_upsampled_pred(NULL, NULL, 0, 0, &mv, comp_pred, pred, width,
                               height, subpel_x_q3, subpel_y_q3, ref,
                               ref_stride, mask, mask_stride, invert_mask,
                               USE_8_TAPS);
}

void shim_highbd_upsampled_pred(uint16_t *comp_pred, int width, int height,
                                int subpel_x_q3, int subpel_y_q3,
                                const uint16_t *ref, int ref_stride, int bd) {
  MV mv = { 0, 0 };
  aom_highbd_upsampled_pred_c(NULL, NULL, 0, 0, &mv,
                              CONVERT_TO_BYTEPTR(comp_pred), width, height,
                              subpel_x_q3, subpel_y_q3, CONVERT_TO_BYTEPTR(ref),
                              ref_stride, bd, USE_8_TAPS);
}

void shim_highbd_comp_avg_upsampled_pred(uint16_t *comp_pred,
                                         const uint16_t *pred, int width,
                                         int height, int subpel_x_q3,
                                         int subpel_y_q3, const uint16_t *ref,
                                         int ref_stride, int bd) {
  MV mv = { 0, 0 };
  aom_highbd_comp_avg_upsampled_pred_c(
      NULL, NULL, 0, 0, &mv, CONVERT_TO_BYTEPTR(comp_pred),
      CONVERT_TO_BYTEPTR(pred), width, height, subpel_x_q3, subpel_y_q3,
      CONVERT_TO_BYTEPTR(ref), ref_stride, bd, USE_8_TAPS);
}

void shim_highbd_comp_mask_upsampled_pred(uint16_t *comp_pred,
                                          const uint16_t *pred, int width,
                                          int height, int subpel_x_q3,
                                          int subpel_y_q3, const uint16_t *ref,
                                          int ref_stride, const uint8_t *mask,
                                          int mask_stride, int invert_mask,
                                          int bd) {
  MV mv = { 0, 0 };
  aom_highbd_comp_mask_upsampled_pred(
      NULL, NULL, 0, 0, &mv, CONVERT_TO_BYTEPTR(comp_pred),
      CONVERT_TO_BYTEPTR(pred), width, height, subpel_x_q3, subpel_y_q3,
      CONVERT_TO_BYTEPTR(ref), ref_stride, mask, mask_stride, invert_mask, bd,
      USE_8_TAPS);
}
