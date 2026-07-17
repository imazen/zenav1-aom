/* ------------------------------------------------------------------ */
/* pickrst_shim.c — loop-restoration ENCODER-SEARCH oracles            */
/*                                                                     */
/* Thin wrappers over the EXPORTED `_c` reference functions of         */
/* av1/encoder/pickrst.c (the per-RU search numeric core):             */
/*   (a) shim_compute_stats          — av1_compute_stats_c (lowbd      */
/*       Wiener autocorrelation M/H, incl. downsampled-stats mode)     */
/*   (b) shim_compute_stats_highbd   — av1_compute_stats_highbd_c      */
/*       (u16 planes via CONVERT_TO_BYTEPTR, bd 10/12 divider)         */
/*   (c) shim_pixel_proj_error       — av1_lowbd_pixel_proj_error_c /  */
/*       av1_highbd_pixel_proj_error_c (SGR projection SSE)            */
/*   (d) shim_calc_proj_params       — av1_calc_proj_params_c /        */
/*       av1_calc_proj_params_high_bd_c (SGR least-squares H/C)        */
/*   (e) shim_selfguided_restoration — av1_selfguided_restoration_c    */
/*       (the flt0/flt1 producer the SGR search projects against)      */
/* No transcription: every entry point calls the real compiled code.   */
/* ------------------------------------------------------------------ */

#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "aom/aom_integer.h"
#include "aom_ports/mem.h"
#include "av1/common/restoration.h"

/* Exported reference functions (av1/av1_rtcd.h / restoration.h). */
void av1_compute_stats_c(int wiener_win, const uint8_t *dgd, const uint8_t *src,
                         int16_t *dgd_avg, int16_t *src_avg, int h_start,
                         int h_end, int v_start, int v_end, int dgd_stride,
                         int src_stride, int64_t *M, int64_t *H,
                         int use_downsampled_wiener_stats);
void av1_compute_stats_highbd_c(int wiener_win, const uint8_t *dgd8,
                                const uint8_t *src8, int16_t *dgd_avg,
                                int16_t *src_avg, int h_start, int h_end,
                                int v_start, int v_end, int dgd_stride,
                                int src_stride, int64_t *M, int64_t *H,
                                aom_bit_depth_t bit_depth);
int64_t av1_lowbd_pixel_proj_error_c(const uint8_t *src8, int width,
                                     int height, int src_stride,
                                     const uint8_t *dat8, int dat_stride,
                                     int32_t *flt0, int flt0_stride,
                                     int32_t *flt1, int flt1_stride, int xq[2],
                                     const sgr_params_type *params);
int64_t av1_highbd_pixel_proj_error_c(const uint8_t *src8, int width,
                                      int height, int src_stride,
                                      const uint8_t *dat8, int dat_stride,
                                      int32_t *flt0, int flt0_stride,
                                      int32_t *flt1, int flt1_stride,
                                      int xq[2],
                                      const sgr_params_type *params);
void av1_calc_proj_params_c(const uint8_t *src8, int width, int height,
                            int src_stride, const uint8_t *dat8,
                            int dat_stride, int32_t *flt0, int flt0_stride,
                            int32_t *flt1, int flt1_stride, int64_t H[2][2],
                            int64_t C[2], const sgr_params_type *params);
void av1_calc_proj_params_high_bd_c(const uint8_t *src8, int width, int height,
                                    int src_stride, const uint8_t *dat8,
                                    int dat_stride, int32_t *flt0,
                                    int flt0_stride, int32_t *flt1,
                                    int flt1_stride, int64_t H[2][2],
                                    int64_t C[2],
                                    const sgr_params_type *params);
int av1_selfguided_restoration_c(const uint8_t *dgd8, int width, int height,
                                 int dgd_stride, int32_t *flt0, int32_t *flt1,
                                 int flt_stride, int sgr_params_idx,
                                 int bit_depth, int highbd);

/* (a) Lowbd Wiener stats: u8 buffers, M[win2] / H[win2*win2] out. */
void shim_compute_stats(int wiener_win, const uint8_t *dgd, const uint8_t *src,
                        int h_start, int h_end, int v_start, int v_end,
                        int dgd_stride, int src_stride, int64_t *M, int64_t *H,
                        int use_downsampled_wiener_stats) {
  av1_compute_stats_c(wiener_win, dgd, src, NULL, NULL, h_start, h_end,
                      v_start, v_end, dgd_stride, src_stride, M, H,
                      use_downsampled_wiener_stats);
}

/* (b) Highbd Wiener stats: u16 buffers (CONVERT_TO_BYTEPTR), bd 8/10/12. */
void shim_compute_stats_highbd(int wiener_win, const uint16_t *dgd,
                               const uint16_t *src, int h_start, int h_end,
                               int v_start, int v_end, int dgd_stride,
                               int src_stride, int64_t *M, int64_t *H,
                               int bit_depth) {
  av1_compute_stats_highbd_c(wiener_win, CONVERT_TO_BYTEPTR(dgd),
                             CONVERT_TO_BYTEPTR(src), NULL, NULL, h_start,
                             h_end, v_start, v_end, dgd_stride, src_stride, M,
                             H, (aom_bit_depth_t)bit_depth);
}

/* (c) SGR projection SSE. highbd=0: u8 src/dat (u16 inputs narrowed by the
 * caller into the low bytes of the u16 arrays is NOT done here — the caller
 * passes real u8 buffers via src8/dat8 when highbd=0 and u16 buffers when
 * highbd=1). ep selects av1_sgr_params[ep]. */
int64_t shim_pixel_proj_error(const uint8_t *src8, const uint16_t *src16,
                              int width, int height, int src_stride,
                              const uint8_t *dat8, const uint16_t *dat16,
                              int dat_stride, int32_t *flt0, int flt0_stride,
                              int32_t *flt1, int flt1_stride, int *xq, int ep,
                              int highbd) {
  const sgr_params_type *params = &av1_sgr_params[ep];
  if (highbd) {
    return av1_highbd_pixel_proj_error_c(
        CONVERT_TO_BYTEPTR(src16), width, height, src_stride,
        CONVERT_TO_BYTEPTR(dat16), dat_stride, flt0, flt0_stride, flt1,
        flt1_stride, xq, params);
  }
  return av1_lowbd_pixel_proj_error_c(src8, width, height, src_stride, dat8,
                                      dat_stride, flt0, flt0_stride, flt1,
                                      flt1_stride, xq, params);
}

/* (d) SGR least-squares H (2x2) / C (2) accumulation. Layout: h_out[4] row
 * major, c_out[2]. */
void shim_calc_proj_params(const uint8_t *src8, const uint16_t *src16,
                           int width, int height, int src_stride,
                           const uint8_t *dat8, const uint16_t *dat16,
                           int dat_stride, int32_t *flt0, int flt0_stride,
                           int32_t *flt1, int flt1_stride, int64_t *h_out,
                           int64_t *c_out, int ep, int highbd) {
  const sgr_params_type *params = &av1_sgr_params[ep];
  int64_t H[2][2] = { { 0, 0 }, { 0, 0 } };
  int64_t C[2] = { 0, 0 };
  if (highbd) {
    av1_calc_proj_params_high_bd_c(CONVERT_TO_BYTEPTR(src16), width, height,
                                   src_stride, CONVERT_TO_BYTEPTR(dat16),
                                   dat_stride, flt0, flt0_stride, flt1,
                                   flt1_stride, H, C, params);
  } else {
    av1_calc_proj_params_c(src8, width, height, src_stride, dat8, dat_stride,
                           flt0, flt0_stride, flt1, flt1_stride, H, C, params);
  }
  h_out[0] = H[0][0];
  h_out[1] = H[0][1];
  h_out[2] = H[1][0];
  h_out[3] = H[1][1];
  c_out[0] = C[0];
  c_out[1] = C[1];
}

/* (e) The flt0/flt1 producer (search-side apply_sgr building block).
 * highbd=0: dgd8 is a real u8 buffer; highbd=1: dgd16 via CONVERT. */
int shim_selfguided_restoration(const uint8_t *dgd8, const uint16_t *dgd16,
                                int width, int height, int dgd_stride,
                                int32_t *flt0, int32_t *flt1, int flt_stride,
                                int ep, int bit_depth, int highbd) {
  const uint8_t *p = highbd ? CONVERT_TO_BYTEPTR(dgd16) : dgd8;
  return av1_selfguided_restoration_c(p, width, height, dgd_stride, flt0, flt1,
                                      flt_stride, ep, bit_depth, highbd);
}
