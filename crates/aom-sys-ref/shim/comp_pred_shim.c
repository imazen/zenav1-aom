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
#include "aom_mem/aom_mem.h"


/* ---- the alignment contract -----------------------------------------
 * Several entries below reach RTCD-dispatched kernels (`aom_upsampled_pred`,
 * `aom_highbd_upsampled_pred`, `aom_comp_mask_pred`, `aom_highbd_comp_mask_pred`).
 * On this aarch64 oracle those names are `#define`d straight to their NEON
 * implementations, which use unaligned accesses — so a misaligned buffer is
 * invisible here. On x86 they are real function pointers into SSE/AVX kernels,
 * and libaom's own callers always hand them `DECLARE_ALIGNED(16, ...)` storage
 * (mcomp.c:2923/2933/2944). A Rust `Vec<u8>` is 1-byte aligned.
 *
 * So: every buffer a dispatched kernel reads or writes is bounced through
 * 64-byte-aligned scratch here (64 covers AVX-512, strictly stronger than the
 * encoder's 16 and never weaker). The copies are the price of handing C the
 * same contract the encoder does. */
static void *shim_align_in(const void *src, size_t bytes) {
  void *p = aom_memalign(64, bytes ? bytes : 64);
  if (!p) return NULL;
  if (src) memcpy(p, src, bytes);
  else memset(p, 0, bytes);
  return p;
}

/* The DESTINATION buffer of `aom_upsampled_pred` / `aom_highbd_upsampled_pred`
 * must be sized as the encoder sizes it, not as `width * height`.
 *
 * Their x86 SIMD implementations use `comp_pred8` ITSELF as the horizontal
 * intermediate for the 2-D sub-pel case, writing `intermediate_height` rows at
 * stride MAX_SB_SIZE (reconinter_enc_sse2.c: `temp = CONVERT_TO_SHORTPTR(comp_pred8)`,
 * `temp_start_vert = temp + MAX_SB_SIZE * ((taps >> 1) - 1)`). A `w * h`
 * allocation is scribbled far past its end. libaom's own buffer is
 *   aom_memalign(16, (1 + is_hbd) * ((MAX_SB_SIZE + 16) + 16) * MAX_SB_SIZE)
 * (encoder.c:981) for exactly this reason.
 *
 * Measured 2026-08-31 on `x86_64-apple-darwin` under Rosetta: with a `w * h`
 * scratch, `comp_mask_upsampled_pred`, `highbd_comp_mask_upsampled_pred` and
 * `highbd_comp_avg_upsampled_pred` all diverged at sub-pel phase (3,5). On
 * aarch64 the NEON kernels use their own scratch, so the undersize is
 * invisible — the alignment fix alone did not catch this. */
#define SHIM_UPSAMPLE_DST_BYTES   ((size_t)2 * ((MAX_SB_SIZE + 16) + 16) * MAX_SB_SIZE)

/* ---- aom_dsp/variance.c: the two compound blends ------------------- */

void shim_comp_avg_pred(uint8_t *comp_pred, const uint8_t *pred, int width,
                        int height, const uint8_t *ref, int ref_stride) {
  aom_comp_avg_pred_c(comp_pred, pred, width, height, ref, ref_stride);
}

void shim_comp_mask_pred(uint8_t *comp_pred, const uint8_t *pred, int width,
                         int height, const uint8_t *ref, int ref_stride,
                         const uint8_t *mask, int mask_stride,
                         int invert_mask) {
  const size_t n = (size_t)width * height;
  uint8_t *acomp = (uint8_t *)shim_align_in(NULL, SHIM_UPSAMPLE_DST_BYTES);
  uint8_t *apred = (uint8_t *)shim_align_in(pred, n);
  if (!acomp || !apred) {
    aom_free(acomp);
    aom_free(apred);
    return;
  }
  aom_comp_mask_pred_c(acomp, apred, width, height, ref, ref_stride, mask,
                       mask_stride, invert_mask);
  memcpy(comp_pred, acomp, n);
  aom_free(acomp);
  aom_free(apred);
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
  const size_t n = (size_t)width * height;
  uint8_t *acomp = (uint8_t *)shim_align_in(NULL, SHIM_UPSAMPLE_DST_BYTES);
  uint8_t *apred = (uint8_t *)shim_align_in(pred, n);
  if (!acomp || !apred) {
    aom_free(acomp);
    aom_free(apred);
    return;
  }
  aom_comp_avg_upsampled_pred_c(NULL, NULL, 0, 0, &mv, acomp, apred, width,
                                height, subpel_x_q3, subpel_y_q3, ref,
                                ref_stride, USE_8_TAPS);
  memcpy(comp_pred, acomp, n);
  aom_free(acomp);
  aom_free(apred);
}

void shim_comp_mask_upsampled_pred(uint8_t *comp_pred, const uint8_t *pred,
                                   int width, int height, int subpel_x_q3,
                                   int subpel_y_q3, const uint8_t *ref,
                                   int ref_stride, const uint8_t *mask,
                                   int mask_stride, int invert_mask) {
  MV mv = { 0, 0 };
  const size_t n = (size_t)width * height;
  uint8_t *acomp = (uint8_t *)shim_align_in(NULL, SHIM_UPSAMPLE_DST_BYTES);
  uint8_t *apred = (uint8_t *)shim_align_in(pred, n);
  if (!acomp || !apred) {
    aom_free(acomp);
    aom_free(apred);
    return;
  }
  aom_comp_mask_upsampled_pred(NULL, NULL, 0, 0, &mv, acomp, apred, width,
                               height, subpel_x_q3, subpel_y_q3, ref,
                               ref_stride, mask, mask_stride, invert_mask,
                               USE_8_TAPS);
  memcpy(comp_pred, acomp, n);
  aom_free(acomp);
  aom_free(apred);
}

void shim_highbd_upsampled_pred(uint16_t *comp_pred, int width, int height,
                                int subpel_x_q3, int subpel_y_q3,
                                const uint16_t *ref, int ref_stride, int bd) {
  MV mv = { 0, 0 };
  const size_t n = (size_t)width * height;
  uint16_t *acomp = (uint16_t *)shim_align_in(NULL, SHIM_UPSAMPLE_DST_BYTES);
  if (!acomp) return;
  aom_highbd_upsampled_pred_c(NULL, NULL, 0, 0, &mv,
                              CONVERT_TO_BYTEPTR(acomp), width, height,
                              subpel_x_q3, subpel_y_q3, CONVERT_TO_BYTEPTR(ref),
                              ref_stride, bd, USE_8_TAPS);
  memcpy(comp_pred, acomp, n * sizeof(uint16_t));
  aom_free(acomp);
}

void shim_highbd_comp_avg_upsampled_pred(uint16_t *comp_pred,
                                         const uint16_t *pred, int width,
                                         int height, int subpel_x_q3,
                                         int subpel_y_q3, const uint16_t *ref,
                                         int ref_stride, int bd) {
  MV mv = { 0, 0 };
  const size_t n = (size_t)width * height;
  uint16_t *acomp = (uint16_t *)shim_align_in(NULL, SHIM_UPSAMPLE_DST_BYTES);
  uint16_t *apred = (uint16_t *)shim_align_in(pred, n * sizeof(uint16_t));
  if (!acomp || !apred) {
    aom_free(acomp);
    aom_free(apred);
    return;
  }
  aom_highbd_comp_avg_upsampled_pred_c(
      NULL, NULL, 0, 0, &mv, CONVERT_TO_BYTEPTR(acomp),
      CONVERT_TO_BYTEPTR(apred), width, height, subpel_x_q3, subpel_y_q3,
      CONVERT_TO_BYTEPTR(ref), ref_stride, bd, USE_8_TAPS);
  memcpy(comp_pred, acomp, n * sizeof(uint16_t));
  aom_free(acomp);
  aom_free(apred);
}

void shim_highbd_comp_mask_upsampled_pred(uint16_t *comp_pred,
                                          const uint16_t *pred, int width,
                                          int height, int subpel_x_q3,
                                          int subpel_y_q3, const uint16_t *ref,
                                          int ref_stride, const uint8_t *mask,
                                          int mask_stride, int invert_mask,
                                          int bd) {
  MV mv = { 0, 0 };
  const size_t n = (size_t)width * height;
  uint16_t *acomp = (uint16_t *)shim_align_in(NULL, SHIM_UPSAMPLE_DST_BYTES);
  uint16_t *apred = (uint16_t *)shim_align_in(pred, n * sizeof(uint16_t));
  if (!acomp || !apred) {
    aom_free(acomp);
    aom_free(apred);
    return;
  }
  aom_highbd_comp_mask_upsampled_pred(
      NULL, NULL, 0, 0, &mv, CONVERT_TO_BYTEPTR(acomp),
      CONVERT_TO_BYTEPTR(apred), width, height, subpel_x_q3, subpel_y_q3,
      CONVERT_TO_BYTEPTR(ref), ref_stride, mask, mask_stride, invert_mask, bd,
      USE_8_TAPS);
  memcpy(comp_pred, acomp, n * sizeof(uint16_t));
  aom_free(acomp);
  aom_free(apred);
}
