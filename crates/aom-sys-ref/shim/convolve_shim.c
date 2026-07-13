/* Shim over av1_convolve_{x,y,2d}_sr_c using libaom's real 8-tap kernels
 * (regular/smooth/sharp) + SR ConvolveParams. Oracle use only. */
#include <string.h>
#include "av1/common/convolve.h"
#include "av1/common/filter.h"

void av1_convolve_x_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                         int dst_stride, int w, int h,
                         const InterpFilterParams *filter_params_x,
                         const int subpel_x_qn, ConvolveParams *conv_params);
void av1_convolve_y_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                         int dst_stride, int w, int h,
                         const InterpFilterParams *filter_params_y,
                         const int subpel_y_qn);
void av1_convolve_2d_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                          int dst_stride, int w, int h,
                          const InterpFilterParams *filter_params_x,
                          const InterpFilterParams *filter_params_y,
                          const int subpel_x_qn, const int subpel_y_qn,
                          ConvolveParams *conv_params);

static const int16_t *kernel_for(int ftype) {
  switch (ftype) {
    case 1: return (const int16_t *)av1_sub_pel_filters_8smooth;
    case 2: return (const int16_t *)av1_sub_pel_filters_8sharp;
    default: return (const int16_t *)av1_sub_pel_filters_8;
  }
}

void shim_convolve_x_sr(const uint8_t *src, int ss, uint8_t *dst, int ds,
                        int w, int h, int subpel, int ftype) {
  InterpFilterParams fp = { kernel_for(ftype), SUBPEL_TAPS, EIGHTTAP_REGULAR };
  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.round_0 = 3;
  cp.round_1 = 2 * FILTER_BITS - 3;
  av1_convolve_x_sr_c(src, ss, dst, ds, w, h, &fp, subpel, &cp);
}

void shim_convolve_y_sr(const uint8_t *src, int ss, uint8_t *dst, int ds,
                        int w, int h, int subpel, int ftype) {
  InterpFilterParams fp = { kernel_for(ftype), SUBPEL_TAPS, EIGHTTAP_REGULAR };
  av1_convolve_y_sr_c(src, ss, dst, ds, w, h, &fp, subpel);
}

void shim_convolve_2d_sr(const uint8_t *src, int ss, uint8_t *dst, int ds,
                         int w, int h, int subpel_x, int subpel_y, int ftype) {
  InterpFilterParams fp = { kernel_for(ftype), SUBPEL_TAPS, EIGHTTAP_REGULAR };
  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.round_0 = 3;
  cp.round_1 = 2 * FILTER_BITS - 3;
  av1_convolve_2d_sr_c(src, ss, dst, ds, w, h, &fp, &fp, subpel_x, subpel_y, &cp);
}
