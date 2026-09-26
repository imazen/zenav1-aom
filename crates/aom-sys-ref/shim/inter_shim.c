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
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "av1/common/reconinter.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"

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

/* ============================ compound predictors ===========================
 * Oracle for the two-reference compound predictor (aom-inter, step 4a). Wraps
 * the SAME REAL `inter_predictor` / `highbd_inter_predictor` static inlines as
 * `shim_inter_predictor` (av1/common/reconinter.h:255/275), but drives the
 * two-ref compound loop the decoder runs in `build_inter_predictors`
 * (reconinter_template.inc): per ref it builds ConvolveParams via the REAL
 * `get_conv_params_no_round(cmp_index=ref, .., is_compound=1, bd)` — which sets
 * `do_average = ref`, `round_0 = ROUND0_BITS(+bd12 adj)`, `round_1 =
 * COMPOUND_ROUND1_BITS`, `dst = tmp_conv_dst` — then stamps the block's
 * dist-weighted blend (`use_dist_wtd_comp_avg`/`fwd_offset`/`bck_offset`, from
 * `av1_dist_wtd_comp_weight_assign`) onto it. ref 0 fills the shared dst16,
 * ref 1 combines into `dst`. The two refs can carry DIFFERENT subpel phases
 * (each has its own MV); the interp filters are shared (one per block).
 *
 * ALIGNMENT / DISPATCH: `inter_predictor` -> `av1_convolve_2d_facade` reaches
 * the RTCD-dispatched `av1_dist_wtd_convolve_*` / `av1_highbd_dist_wtd_convolve_*`
 * family, whose AVX2 (and SSE4_1 `build_compound_diffwtd`-style) tiers do
 * ALIGNED 32-byte stores to `dst`/`conv_params->dst` at caller-chosen strides —
 * the same latent-fault class as KB-43 (reconinter_enc_shim.c). libaom's own
 * callers hand those kernels `aom_memalign`'d block buffers; this oracle is fed
 * ordinary heap slices. So the four dist-wtd RTCD entries are rebound to their
 * scalar `_c` references below. On x86 RTCD binds them through assignable
 * function pointers; on aarch64 the generated header `#define`s each name to a
 * NEON tier that uses unaligned loads (no fault, nothing to rebind).
 */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
static void shim_pin_dist_wtd_scalar(void) {
  av1_dist_wtd_convolve_2d_copy = av1_dist_wtd_convolve_2d_copy_c;
  av1_dist_wtd_convolve_x = av1_dist_wtd_convolve_x_c;
  av1_dist_wtd_convolve_y = av1_dist_wtd_convolve_y_c;
  av1_dist_wtd_convolve_2d = av1_dist_wtd_convolve_2d_c;
#if CONFIG_AV1_HIGHBITDEPTH
  av1_highbd_dist_wtd_convolve_2d_copy = av1_highbd_dist_wtd_convolve_2d_copy_c;
  av1_highbd_dist_wtd_convolve_x = av1_highbd_dist_wtd_convolve_x_c;
  av1_highbd_dist_wtd_convolve_y = av1_highbd_dist_wtd_convolve_y_c;
  av1_highbd_dist_wtd_convolve_2d = av1_highbd_dist_wtd_convolve_2d_c;
#endif
}
#else
static void shim_pin_dist_wtd_scalar(void) {}
#endif

void shim_compound_inter_predictor(
    const uint8_t *src0, int src_stride0, const uint8_t *src1, int src_stride1,
    uint8_t *dst, int dst_stride, CONV_BUF_TYPE *dst16, int w, int h,
    int subpel_x0, int subpel_y0, int subpel_x1, int subpel_y1, int filter_x,
    int filter_y, int use_dist_wtd, int fwd_offset, int bck_offset) {
  shim_pin_dist_wtd_scalar();
  const uint8_t *srcs[2] = { src0, src1 };
  const int strs[2] = { src_stride0, src_stride1 };
  const int subpx[2] = { subpel_x0, subpel_x1 };
  const int subpy[2] = { subpel_y0, subpel_y1 };

  const InterpFilterParams *ifp[2];
  ifp[0] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_x, w);
  ifp[1] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_y, h);

  for (int ref = 0; ref < 2; ref++) {
    SubpelParams sp;
    memset(&sp, 0, sizeof(sp));
    sp.xs = SCALE_SUBPEL_SHIFTS;
    sp.ys = SCALE_SUBPEL_SHIFTS;
    sp.subpel_x = subpx[ref] << SCALE_EXTRA_BITS;
    sp.subpel_y = subpy[ref] << SCALE_EXTRA_BITS;

    ConvolveParams cp = get_conv_params_no_round(ref, 0, dst16, w, 1, 8);
    cp.use_dist_wtd_comp_avg = use_dist_wtd;
    cp.fwd_offset = fwd_offset;
    cp.bck_offset = bck_offset;

    inter_predictor(srcs[ref], strs[ref], dst, dst_stride, &sp, w, h, &cp, ifp);
  }
}

void shim_highbd_compound_inter_predictor(
    const uint16_t *src0, int src_stride0, const uint16_t *src1, int src_stride1,
    uint16_t *dst, int dst_stride, CONV_BUF_TYPE *dst16, int w, int h,
    int subpel_x0, int subpel_y0, int subpel_x1, int subpel_y1, int filter_x,
    int filter_y, int use_dist_wtd, int fwd_offset, int bck_offset, int bd) {
  shim_pin_dist_wtd_scalar();
  const uint16_t *srcs[2] = { src0, src1 };
  const int strs[2] = { src_stride0, src_stride1 };
  const int subpx[2] = { subpel_x0, subpel_x1 };
  const int subpy[2] = { subpel_y0, subpel_y1 };

  const InterpFilterParams *ifp[2];
  ifp[0] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_x, w);
  ifp[1] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_y, h);

  for (int ref = 0; ref < 2; ref++) {
    SubpelParams sp;
    memset(&sp, 0, sizeof(sp));
    sp.xs = SCALE_SUBPEL_SHIFTS;
    sp.ys = SCALE_SUBPEL_SHIFTS;
    sp.subpel_x = subpx[ref] << SCALE_EXTRA_BITS;
    sp.subpel_y = subpy[ref] << SCALE_EXTRA_BITS;

    ConvolveParams cp = get_conv_params_no_round(ref, 0, dst16, w, 1, bd);
    cp.use_dist_wtd_comp_avg = use_dist_wtd;
    cp.fwd_offset = fwd_offset;
    cp.bck_offset = bck_offset;

    /* The facade stores u16 planes behind `uint8_t*` handles via
     * CONVERT_TO_BYTEPTR (aom_ports/mem.h: `u16addr >> 1`, round-tripped by
     * CONVERT_TO_SHORTPTR's `<< 1`) — pass the packed byte pointer, not a raw
     * `(uint8_t *)` cast, or the facade reads at twice the real address. */
    highbd_inter_predictor(CONVERT_TO_BYTEPTR(srcs[ref]), strs[ref],
                           CONVERT_TO_BYTEPTR(dst), dst_stride, &sp, w, h, &cp,
                           ifp, bd);
  }
}

/* ==================== masked-compound predictors =============================
 * Oracle for `build_masked_compound_inter_predictor` (aom-inter, step 4b) — the
 * wedge / diff-weighted masked compound path of `av1_make_masked_inter_predictor`
 * + `build_masked_compound_no_round` (reconinter.c:629/:602). Unlike the group-0
 * compound shim, each reference convolves into its OWN `CONV_BUF` (ref 0 ->
 * `org_dst`/`tmp_conv_dst`, ref 1 -> `tmp_buf16`) with `do_average = 0`, then a
 * REAL `aom_*_blend_a64_d16_mask` blends them through the luma-resolution mask:
 * `av1_get_compound_type_mask` returns the wedge codebook mask, or — for
 * COMPOUND_DIFFWTD — `comp_data->seg_mask`, which `av1_build_compound_
 * diffwtd_mask_d16` builds from the two d16 buffers on luma (`!conv_params.plane`)
 * and which the chroma calls reuse. `mask_stride = block_size_wide[sb_type]`.
 *
 * `seg_mask` is caller-owned (it plays `xd->seg_mask`): for COMPOUND_DIFFWTD the
 * shim builds it when `build_seg_mask` (the luma call) and reads it as-is
 * otherwise (the chroma calls) — the same in/out contract the Rust predictor
 * exposes through `is_luma`. For COMPOUND_WEDGE `seg_mask` is ignored.
 *
 * ALIGNMENT / DISPATCH: same RTCD aligned-store hazard as the compound shim —
 * pin the dist-wtd family to `_c`.
 */
void shim_masked_compound_inter_predictor(
    const uint8_t *src0, int src_stride0, const uint8_t *src1, int src_stride1,
    uint8_t *dst, int dst_stride, int w, int h, int subpel_x0, int subpel_y0,
    int subpel_x1, int subpel_y1, int filter_x, int filter_y, int comp_type,
    int wedge_index, int wedge_sign, int mask_type, int bsize, int ssx,
    int ssy, uint8_t *seg_mask, int build_seg_mask) {
  shim_pin_dist_wtd_scalar();
  av1_init_wedge_masks();
  const uint8_t *srcs[2] = { src0, src1 };
  const int strs[2] = { src_stride0, src_stride1 };
  const int subpx[2] = { subpel_x0, subpel_x1 };
  const int subpy[2] = { subpel_y0, subpel_y1 };
  const InterpFilterParams *ifp[2];
  ifp[0] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_x, w);
  ifp[1] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_y, h);

  /* ref 0 -> conv_a (org_dst), ref 1 -> conv_b (tmp_buf16); both do_average=0.
   * get_conv_params_no_round(cmp_index=0, .., is_compound=1, bd) gives the
   * compound rounds (round_1 = COMPOUND_ROUND1_BITS) with do_average = 0. */
  CONV_BUF_TYPE conv_a[MAX_SB_SQUARE], conv_b[MAX_SB_SQUARE];
  CONV_BUF_TYPE *convs[2] = { conv_a, conv_b };
  for (int ref = 0; ref < 2; ref++) {
    SubpelParams sp;
    memset(&sp, 0, sizeof(sp));
    sp.xs = SCALE_SUBPEL_SHIFTS;
    sp.ys = SCALE_SUBPEL_SHIFTS;
    sp.subpel_x = subpx[ref] << SCALE_EXTRA_BITS;
    sp.subpel_y = subpy[ref] << SCALE_EXTRA_BITS;
    ConvolveParams cp = get_conv_params_no_round(0, 0, convs[ref], w, 1, 8);
    inter_predictor(srcs[ref], strs[ref], dst, dst_stride, &sp, w, h, &cp, ifp);
  }

  INTERINTER_COMPOUND_DATA cd;
  memset(&cd, 0, sizeof(cd));
  cd.seg_mask = seg_mask;
  cd.type = (COMPOUND_TYPE)comp_type;
  cd.wedge_index = (int8_t)wedge_index;
  cd.wedge_sign = (int8_t)wedge_sign;
  cd.mask_type = (DIFFWTD_MASK_TYPE)mask_type;
  const int mask_stride = block_size_wide[bsize];
  /* The blend's rounding params are the compound no-round pair. */
  ConvolveParams cpb = get_conv_params_no_round(0, 0, conv_a, w, 1, 8);
  const uint8_t *mask;
  if (comp_type == COMPOUND_DIFFWTD) {
    if (build_seg_mask) {
      av1_build_compound_diffwtd_mask_d16(cd.seg_mask, cd.mask_type, conv_a, w,
                                          conv_b, w, h, w, &cpb, 8);
    }
    mask = cd.seg_mask;
  } else {
    mask = av1_get_compound_type_mask(&cd, (BLOCK_SIZE)bsize);
  }
  /* The `_c` tier — the RTCD AVX2 blend does aligned 32-byte stores that fault
   * on ordinary heap oracle buffers (same latent class as the dist-wtd pin). */
  aom_lowbd_blend_a64_d16_mask_c(dst, dst_stride, conv_a, w, conv_b, w, mask,
                                 mask_stride, w, h, ssx, ssy, &cpb);
}

void shim_highbd_masked_compound_inter_predictor(
    const uint16_t *src0, int src_stride0, const uint16_t *src1, int src_stride1,
    uint16_t *dst, int dst_stride, int w, int h, int subpel_x0, int subpel_y0,
    int subpel_x1, int subpel_y1, int filter_x, int filter_y, int comp_type,
    int wedge_index, int wedge_sign, int mask_type, int bsize, int ssx,
    int ssy, uint8_t *seg_mask, int build_seg_mask, int bd) {
  shim_pin_dist_wtd_scalar();
  av1_init_wedge_masks();
  const uint16_t *srcs[2] = { src0, src1 };
  const int strs[2] = { src_stride0, src_stride1 };
  const int subpx[2] = { subpel_x0, subpel_x1 };
  const int subpy[2] = { subpel_y0, subpel_y1 };
  const InterpFilterParams *ifp[2];
  ifp[0] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_x, w);
  ifp[1] =
      av1_get_interp_filter_params_with_block_size((InterpFilter)filter_y, h);

  CONV_BUF_TYPE conv_a[MAX_SB_SQUARE], conv_b[MAX_SB_SQUARE];
  CONV_BUF_TYPE *convs[2] = { conv_a, conv_b };
  for (int ref = 0; ref < 2; ref++) {
    SubpelParams sp;
    memset(&sp, 0, sizeof(sp));
    sp.xs = SCALE_SUBPEL_SHIFTS;
    sp.ys = SCALE_SUBPEL_SHIFTS;
    sp.subpel_x = subpx[ref] << SCALE_EXTRA_BITS;
    sp.subpel_y = subpy[ref] << SCALE_EXTRA_BITS;
    ConvolveParams cp = get_conv_params_no_round(0, 0, convs[ref], w, 1, bd);
    highbd_inter_predictor(CONVERT_TO_BYTEPTR(srcs[ref]), strs[ref],
                           CONVERT_TO_BYTEPTR(dst), dst_stride, &sp, w, h, &cp,
                           ifp, bd);
  }

  INTERINTER_COMPOUND_DATA cd;
  memset(&cd, 0, sizeof(cd));
  cd.seg_mask = seg_mask;
  cd.type = (COMPOUND_TYPE)comp_type;
  cd.wedge_index = (int8_t)wedge_index;
  cd.wedge_sign = (int8_t)wedge_sign;
  cd.mask_type = (DIFFWTD_MASK_TYPE)mask_type;
  const int mask_stride = block_size_wide[bsize];
  ConvolveParams cpb = get_conv_params_no_round(0, 0, conv_a, w, 1, bd);
  const uint8_t *mask;
  if (comp_type == COMPOUND_DIFFWTD) {
    if (build_seg_mask) {
      av1_build_compound_diffwtd_mask_d16(cd.seg_mask, cd.mask_type, conv_a, w,
                                          conv_b, w, h, w, &cpb, bd);
    }
    mask = cd.seg_mask;
  } else {
    mask = av1_get_compound_type_mask(&cd, (BLOCK_SIZE)bsize);
  }
  aom_highbd_blend_a64_d16_mask_c(CONVERT_TO_BYTEPTR(dst), dst_stride, conv_a, w,
                                  conv_b, w, mask, mask_stride, w, h, ssx, ssy,
                                  &cpb, bd);
}
