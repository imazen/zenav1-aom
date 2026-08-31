/* Oracle shims for OBMC (overlapped block motion compensation) kernels
 * (crate aom-inter, chunk 4). Oracle use only.
 *
 *  - shim_get_obmc_mask wraps the REAL libaom `av1_get_obmc_mask`
 *    (av1/common/reconinter.c:774) — the raised-cosine feather mask table.
 *  - shim_blend_a64_vmask / shim_blend_a64_hmask wrap the REAL exported
 *    `aom_blend_a64_vmask_c` / `aom_blend_a64_hmask_c`
 *    (aom_dsp/blend_a64_{v,h}mask.c) — the per-row / per-column A64 blends used
 *    by `build_obmc_inter_pred_{above,left}` (reconinter.c:852/:891).
 */
#include "av1/common/reconinter.h"
#include "aom_dsp/blend.h"
#include "config/aom_dsp_rtcd.h"

const unsigned char *shim_get_obmc_mask(int length) {
  return av1_get_obmc_mask(length);
}

void shim_blend_a64_vmask(uint8_t *dst, uint32_t dst_stride,
                          const uint8_t *src0, uint32_t src0_stride,
                          const uint8_t *src1, uint32_t src1_stride,
                          const uint8_t *mask, int w, int h) {
  aom_blend_a64_vmask_c(dst, dst_stride, src0, src0_stride, src1, src1_stride,
                        mask, w, h);
}

void shim_blend_a64_hmask(uint8_t *dst, uint32_t dst_stride,
                          const uint8_t *src0, uint32_t src0_stride,
                          const uint8_t *src1, uint32_t src1_stride,
                          const uint8_t *mask, int w, int h) {
  aom_blend_a64_hmask_c(dst, dst_stride, src0, src0_stride, src1, src1_stride,
                        mask, w, h);
}

/* ---- OBMC distortion kernels --------------------------------------
 * Drives the REAL exported aom_obmc_variance{W}x{H}_c /
 * aom_obmc_sub_pixel_variance{W}x{H}_c and their highbd 8/10/12 twins,
 * dispatched by (w, h) through a size table. Returns the variance and writes
 * the sse out-param; -1 means "no C kernel for this block shape", which the
 * Rust wrapper turns into a named panic rather than a silent pass.
 */
#include <stdint.h>
#include <string.h>
#include "aom_dsp/variance.h"

#define OBMC_SIZES(F)                                                     \
  F(4, 4) F(4, 8) F(8, 4) F(8, 8) F(8, 16) F(16, 8) F(16, 16) F(16, 32)   \
  F(32, 16) F(32, 32) F(32, 64) F(64, 32) F(64, 64) F(64, 128)            \
  F(128, 64) F(128, 128) F(4, 16) F(16, 4) F(8, 32) F(32, 8) F(16, 64)    \
  F(64, 16)

#define DECL_OBMC(W, H)                                                       \
  extern unsigned int aom_obmc_variance##W##x##H##_c(                         \
      const uint8_t *, int, const int32_t *, const int32_t *, unsigned int *);\
  extern unsigned int aom_obmc_sub_pixel_variance##W##x##H##_c(               \
      const uint8_t *, int, int, int, const int32_t *, const int32_t *,       \
      unsigned int *);                                                        \
  extern unsigned int aom_highbd_8_obmc_variance##W##x##H##_c(                \
      const uint8_t *, int, const int32_t *, const int32_t *, unsigned int *);\
  extern unsigned int aom_highbd_10_obmc_variance##W##x##H##_c(               \
      const uint8_t *, int, const int32_t *, const int32_t *, unsigned int *);\
  extern unsigned int aom_highbd_12_obmc_variance##W##x##H##_c(               \
      const uint8_t *, int, const int32_t *, const int32_t *, unsigned int *);\
  extern unsigned int aom_highbd_8_obmc_sub_pixel_variance##W##x##H##_c(      \
      const uint8_t *, int, int, int, const int32_t *, const int32_t *,       \
      unsigned int *);                                                        \
  extern unsigned int aom_highbd_10_obmc_sub_pixel_variance##W##x##H##_c(     \
      const uint8_t *, int, int, int, const int32_t *, const int32_t *,       \
      unsigned int *);                                                        \
  extern unsigned int aom_highbd_12_obmc_sub_pixel_variance##W##x##H##_c(     \
      const uint8_t *, int, int, int, const int32_t *, const int32_t *,       \
      unsigned int *);
OBMC_SIZES(DECL_OBMC)
#undef DECL_OBMC

int64_t shim_obmc_variance(const uint8_t *pre, int pre_stride,
                           const int32_t *wsrc, const int32_t *mask, int w,
                           int h, unsigned int *out_sse) {
#define PICK(W, H)                                                       \
  if (w == W && h == H)                                                  \
    return (int64_t)aom_obmc_variance##W##x##H##_c(pre, pre_stride, wsrc, \
                                                   mask, out_sse);
  OBMC_SIZES(PICK)
#undef PICK
  return -1;
}

int64_t shim_obmc_sub_pixel_variance(const uint8_t *pre, int pre_stride,
                                     int xoffset, int yoffset,
                                     const int32_t *wsrc, const int32_t *mask,
                                     int w, int h, unsigned int *out_sse) {
#define PICK(W, H)                                                            \
  if (w == W && h == H)                                                       \
    return (int64_t)aom_obmc_sub_pixel_variance##W##x##H##_c(                 \
        pre, pre_stride, xoffset, yoffset, wsrc, mask, out_sse);
  OBMC_SIZES(PICK)
#undef PICK
  return -1;
}

int64_t shim_highbd_obmc_variance(const uint16_t *pre, int pre_stride,
                                  const int32_t *wsrc, const int32_t *mask,
                                  int w, int h, int bd,
                                  unsigned int *out_sse) {
  const uint8_t *p8 = CONVERT_TO_BYTEPTR(pre);
#define PICK(W, H)                                                            \
  if (w == W && h == H) {                                                     \
    if (bd == 8)                                                              \
      return (int64_t)aom_highbd_8_obmc_variance##W##x##H##_c(                \
          p8, pre_stride, wsrc, mask, out_sse);                               \
    if (bd == 10)                                                             \
      return (int64_t)aom_highbd_10_obmc_variance##W##x##H##_c(               \
          p8, pre_stride, wsrc, mask, out_sse);                               \
    if (bd == 12)                                                             \
      return (int64_t)aom_highbd_12_obmc_variance##W##x##H##_c(               \
          p8, pre_stride, wsrc, mask, out_sse);                               \
    return -1;                                                                \
  }
  OBMC_SIZES(PICK)
#undef PICK
  return -1;
}

int64_t shim_highbd_obmc_sub_pixel_variance(const uint16_t *pre, int pre_stride,
                                            int xoffset, int yoffset,
                                            const int32_t *wsrc,
                                            const int32_t *mask, int w, int h,
                                            int bd, unsigned int *out_sse) {
  const uint8_t *p8 = CONVERT_TO_BYTEPTR(pre);
#define PICK(W, H)                                                            \
  if (w == W && h == H) {                                                     \
    if (bd == 8)                                                              \
      return (int64_t)aom_highbd_8_obmc_sub_pixel_variance##W##x##H##_c(      \
          p8, pre_stride, xoffset, yoffset, wsrc, mask, out_sse);             \
    if (bd == 10)                                                             \
      return (int64_t)aom_highbd_10_obmc_sub_pixel_variance##W##x##H##_c(     \
          p8, pre_stride, xoffset, yoffset, wsrc, mask, out_sse);             \
    if (bd == 12)                                                             \
      return (int64_t)aom_highbd_12_obmc_sub_pixel_variance##W##x##H##_c(     \
          p8, pre_stride, xoffset, yoffset, wsrc, mask, out_sse);             \
    return -1;                                                                \
  }
  OBMC_SIZES(PICK)
#undef PICK
  return -1;
}
