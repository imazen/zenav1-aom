/* Oracle shims for the compound + high-bit-depth motion-compensation
 * convolutions (crate aom-dsp, modules convolve::compound / convolve::highbd).
 * Oracle use only.
 *
 * Each shim drives the REAL exported C kernel. The only work done here is
 * building the InterpFilterParams / ConvolveParams structs the C signatures
 * take, from the flat kernel row + scalars the Rust side passes.
 *
 * `x_filter` / `y_filter` are the ALREADY-SUBPEL-SELECTED kernel rows the port
 * uses. C selects internally with
 *   av1_get_interp_filter_subpel_kernel(p, subpel) = p->filter_ptr + p->taps*subpel
 * so the shim hands it a one-row table and subpel 0, which makes that lookup
 * the identity and keeps the two sides comparing the same coefficients.
 */
#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "av1/common/convolve.h"
#include "av1/common/filter.h"

static void shim_fill_conv_params(ConvolveParams *cp, uint16_t *dst16,
                                  int dst16_stride, int round_0, int round_1,
                                  int do_average, int use_dist_wtd, int fwd,
                                  int bck) {
  memset(cp, 0, sizeof(*cp));
  cp->dst = dst16;
  cp->dst_stride = dst16_stride;
  cp->round_0 = round_0;
  cp->round_1 = round_1;
  cp->do_average = do_average;
  cp->use_dist_wtd_comp_avg = use_dist_wtd;
  cp->fwd_offset = fwd;
  cp->bck_offset = bck;
  cp->is_compound = 1;
}

static void shim_fill_filter(InterpFilterParams *p, const int16_t *row,
                             int taps) {
  memset(p, 0, sizeof(*p));
  p->filter_ptr = row;
  p->taps = (uint16_t)taps;
  p->interp_filter = EIGHTTAP_REGULAR;
}

/* ---- lowbd av1_dist_wtd_convolve_* -------------------------------- */

void shim_dist_wtd_convolve_2d(const uint8_t *src, int src_stride, uint8_t *dst,
                               int dst_stride, uint16_t *dst16,
                               int dst16_stride, int w, int h,
                               const int16_t *x_filter, int taps_x,
                               const int16_t *y_filter, int taps_y, int round_0,
                               int round_1, int do_average, int use_dist_wtd,
                               int fwd, int bck) {
  InterpFilterParams fx, fy;
  shim_fill_filter(&fx, x_filter, taps_x);
  shim_fill_filter(&fy, y_filter, taps_y);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_dist_wtd_convolve_2d_c(src, src_stride, dst, dst_stride, w, h, &fx, &fy, 0,
                             0, &cp);
}

void shim_dist_wtd_convolve_x(const uint8_t *src, int src_stride, uint8_t *dst,
                              int dst_stride, uint16_t *dst16, int dst16_stride,
                              int w, int h, const int16_t *x_filter, int taps_x,
                              int round_0, int round_1, int do_average,
                              int use_dist_wtd, int fwd, int bck) {
  InterpFilterParams fx;
  shim_fill_filter(&fx, x_filter, taps_x);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_dist_wtd_convolve_x_c(src, src_stride, dst, dst_stride, w, h, &fx, 0, &cp);
}

void shim_dist_wtd_convolve_y(const uint8_t *src, int src_stride, uint8_t *dst,
                              int dst_stride, uint16_t *dst16, int dst16_stride,
                              int w, int h, const int16_t *y_filter, int taps_y,
                              int round_0, int round_1, int do_average,
                              int use_dist_wtd, int fwd, int bck) {
  InterpFilterParams fy;
  shim_fill_filter(&fy, y_filter, taps_y);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_dist_wtd_convolve_y_c(src, src_stride, dst, dst_stride, w, h, &fy, 0, &cp);
}

void shim_dist_wtd_convolve_2d_copy(const uint8_t *src, int src_stride,
                                    uint8_t *dst, int dst_stride,
                                    uint16_t *dst16, int dst16_stride, int w,
                                    int h, int round_0, int round_1,
                                    int do_average, int use_dist_wtd, int fwd,
                                    int bck) {
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_dist_wtd_convolve_2d_copy_c(src, src_stride, dst, dst_stride, w, h, &cp);
}

/* ---- highbd av1_highbd_dist_wtd_convolve_* ------------------------- */

void shim_highbd_dist_wtd_convolve_2d(const uint16_t *src, int src_stride,
                                      uint16_t *dst, int dst_stride,
                                      uint16_t *dst16, int dst16_stride, int w,
                                      int h, const int16_t *x_filter,
                                      int taps_x, const int16_t *y_filter,
                                      int taps_y, int round_0, int round_1,
                                      int do_average, int use_dist_wtd, int fwd,
                                      int bck, int bd) {
  InterpFilterParams fx, fy;
  shim_fill_filter(&fx, x_filter, taps_x);
  shim_fill_filter(&fy, y_filter, taps_y);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_highbd_dist_wtd_convolve_2d_c(src, src_stride, dst, dst_stride, w, h, &fx,
                                    &fy, 0, 0, &cp, bd);
}

void shim_highbd_dist_wtd_convolve_x(const uint16_t *src, int src_stride,
                                     uint16_t *dst, int dst_stride,
                                     uint16_t *dst16, int dst16_stride, int w,
                                     int h, const int16_t *x_filter, int taps_x,
                                     int round_0, int round_1, int do_average,
                                     int use_dist_wtd, int fwd, int bck,
                                     int bd) {
  InterpFilterParams fx;
  shim_fill_filter(&fx, x_filter, taps_x);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_highbd_dist_wtd_convolve_x_c(src, src_stride, dst, dst_stride, w, h, &fx,
                                   0, &cp, bd);
}

void shim_highbd_dist_wtd_convolve_y(const uint16_t *src, int src_stride,
                                     uint16_t *dst, int dst_stride,
                                     uint16_t *dst16, int dst16_stride, int w,
                                     int h, const int16_t *y_filter, int taps_y,
                                     int round_0, int round_1, int do_average,
                                     int use_dist_wtd, int fwd, int bck,
                                     int bd) {
  InterpFilterParams fy;
  shim_fill_filter(&fy, y_filter, taps_y);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_highbd_dist_wtd_convolve_y_c(src, src_stride, dst, dst_stride, w, h, &fy,
                                   0, &cp, bd);
}

void shim_highbd_dist_wtd_convolve_2d_copy(const uint16_t *src, int src_stride,
                                           uint16_t *dst, int dst_stride,
                                           uint16_t *dst16, int dst16_stride,
                                           int w, int h, int round_0,
                                           int round_1, int do_average,
                                           int use_dist_wtd, int fwd, int bck,
                                           int bd) {
  ConvolveParams cp;
  shim_fill_conv_params(&cp, dst16, dst16_stride, round_0, round_1, do_average,
                        use_dist_wtd, fwd, bck);
  av1_highbd_dist_wtd_convolve_2d_copy_c(src, src_stride, dst, dst_stride, w, h,
                                         &cp, bd);
}

/* ---- highbd single-reference av1_highbd_convolve_*_sr_c ------------ */

void shim_highbd_convolve_x_sr(const uint16_t *src, int src_stride,
                               uint16_t *dst, int dst_stride, int w, int h,
                               const int16_t *x_filter, int taps_x, int round_0,
                               int round_1, int bd) {
  InterpFilterParams fx;
  shim_fill_filter(&fx, x_filter, taps_x);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, NULL, 0, round_0, round_1, 0, 0, 0, 0);
  cp.is_compound = 0;
  av1_highbd_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h, &fx, 0, &cp,
                             bd);
}

/* av1_highbd_convolve_y_sr_c takes NO ConvolveParams — its rounding is the
 * fixed FILTER_BITS. Passing one would be a lie about the C signature. */
void shim_highbd_convolve_y_sr(const uint16_t *src, int src_stride,
                               uint16_t *dst, int dst_stride, int w, int h,
                               const int16_t *y_filter, int taps_y, int bd) {
  InterpFilterParams fy;
  shim_fill_filter(&fy, y_filter, taps_y);
  av1_highbd_convolve_y_sr_c(src, src_stride, dst, dst_stride, w, h, &fy, 0, bd);
}

void shim_highbd_convolve_2d_sr(const uint16_t *src, int src_stride,
                                uint16_t *dst, int dst_stride, int w, int h,
                                const int16_t *x_filter, int taps_x,
                                const int16_t *y_filter, int taps_y,
                                int round_0, int round_1, int bd) {
  InterpFilterParams fx, fy;
  shim_fill_filter(&fx, x_filter, taps_x);
  shim_fill_filter(&fy, y_filter, taps_y);
  ConvolveParams cp;
  shim_fill_conv_params(&cp, NULL, 0, round_0, round_1, 0, 0, 0, 0);
  cp.is_compound = 0;
  av1_highbd_convolve_2d_sr_c(src, src_stride, dst, dst_stride, w, h, &fx, &fy,
                              0, 0, &cp, bd);
}
