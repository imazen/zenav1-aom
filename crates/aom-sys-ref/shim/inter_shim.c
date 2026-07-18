/* Oracle shims for the decoder single-ref translational inter predictor
 * (crate aom-inter, chunk 1d). Oracle use only.
 *
 *  - shim_inter_predictor wraps the REAL libaom `inter_predictor`
 *    (av1/common/reconinter.h:255, static inline) for the unscaled lowbd SR
 *    facade path: it constructs the SubpelParams / ConvolveParams / interp
 *    filter params exactly as the decoder does and calls through to the real
 *    `av1_convolve_2d_facade`. Sub-pel phases are passed in 0..15 (they are
 *    left-shifted by SCALE_EXTRA_BITS here, then `revert_scale_extra_bits`
 *    shifts them back — the same round-trip the decoder performs).
 *
 *  - shim_build_mc_border is a verbatim copy of libaom's `build_mc_border`
 *    (av1/decoder/decodeframe.c:455) — a `static inline` with no exported
 *    symbol to wrap, so it is transcribed here (compiled as the exact C body by
 *    clang) to serve as the border oracle. `ref_row` is started at the plane
 *    origin instead of forming `buf_ptr` then subtracting, which is
 *    mathematically identical (C: `ref_row = src - x - y*stride`, with
 *    `src = plane + y*stride + x`) and avoids an out-of-bounds intermediate
 *    pointer.
 */
#include <string.h>
#include "av1/common/reconinter.h"

void shim_inter_predictor(const uint8_t *src, int src_stride, uint8_t *dst,
                          int dst_stride, int w, int h, int subpel_x,
                          int subpel_y, int filter_x, int filter_y) {
  SubpelParams sp;
  memset(&sp, 0, sizeof(sp));
  sp.xs = SCALE_SUBPEL_SHIFTS;
  sp.ys = SCALE_SUBPEL_SHIFTS;
  sp.subpel_x = subpel_x << SCALE_EXTRA_BITS;
  sp.subpel_y = subpel_y << SCALE_EXTRA_BITS;

  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.round_0 = 3;
  cp.round_1 = 2 * FILTER_BITS - 3;
  cp.is_compound = 0;
  cp.do_average = 0;

  const InterpFilterParams *ifp[2];
  ifp[0] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_x, w);
  ifp[1] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_y, h);

  inter_predictor(src, src_stride, dst, dst_stride, &sp, w, h, &cp, ifp);
}

/* Verbatim libaom build_mc_border (decodeframe.c:455), plane-origin form. */
void shim_build_mc_border(const uint8_t *plane, int src_stride, int w, int h,
                          int x, int y, int b_w, int b_h, uint8_t *dst) {
  const int dst_stride = b_w;
  const uint8_t *ref_row = plane;

  if (y >= h)
    ref_row += (h - 1) * src_stride;
  else if (y > 0)
    ref_row += y * src_stride;

  do {
    int right = 0, copy;
    int left = x < 0 ? -x : 0;

    if (left > b_w) left = b_w;

    if (x + b_w > w) right = x + b_w - w;

    if (right > b_w) right = b_w;

    copy = b_w - left - right;

    if (left) memset(dst, ref_row[0], left);

    if (copy) memcpy(dst + left, ref_row + x + left, copy);

    if (right) memset(dst + left + copy, ref_row[w - 1], right);

    dst += dst_stride;
    ++y;

    if (y > 0 && y < h) ref_row += src_stride;
  } while (--b_h);
}
