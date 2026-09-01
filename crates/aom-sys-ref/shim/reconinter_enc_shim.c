/* Oracle shims for av1/encoder/reconinter_enc.c — the encoder-side inter
 * predictor builders.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * The masked-compound assembly this file exposes is three functions, and
 * `nm -g upstream/build/libaom.a` reports only the outermost one
 * (`av1_build_wedge_inter_predictor_from_buf`) for reconinter_enc.c. The two
 * that do the work — `build_wedge_inter_predictor_from_buf` and
 * `build_masked_compound{,_highbd}` — are `static`, and driving them only
 * through the exported plane loop would leave the per-plane arm untested at
 * the sub-rectangle offsets the C signature admits.
 *
 * So this TU compiles libaom's OWN reconinter_enc.c, unmodified, with its
 * exported symbols renamed out of the way, and wraps the statics. Same
 * technique and same justification as `shim/rdopt_shim.c` and
 * `shim/compound_type_shim.c`. EVIDENCE TIER 1c: the real C source, compiled
 * verbatim, as against tier 1's real symbol out of the archive.
 *
 * FLAGS: `-O3 -DNDEBUG` plus the oracle-wide `-ffp-contract=off`, i.e.
 * libaom's own Release flags. `-DNDEBUG` is separately mandatory for ABI
 * agreement (DIFFERENTIAL_PLAYBOOK §3a(a)).
 *
 * ---- the alignment contract ----------------------------------------
 * `aom_blend_a64_mask` / `aom_highbd_blend_a64_mask` /
 * `av1_build_compound_diffwtd_mask{,_highbd}` / `aom_convolve_copy` are all
 * RTCD-dispatched. On aarch64 they are `#define`d straight to NEON (unaligned
 * loads, so misalignment is invisible); on x86 they are SSE/AVX kernels and
 * libaom's own callers hand them `aom_memalign`'d storage. Every buffer that
 * crosses into one is therefore bounced through 64-byte-aligned scratch here.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

#include "aom_mem/aom_mem.h"

/* --- Rename reconinter_enc.c's exported symbols so this TU links beside
 * libaom.a. The `aom_*_upsampled_pred` family is already gated through
 * shim/comp_pred_shim.c against the ARCHIVE's copies, so these renamed
 * duplicates are unused here — they exist only to keep the link clean. */
#define aom_upsampled_pred_c shim_rie_upsampled_pred_c
#define aom_upsampled_pred_scaled shim_rie_upsampled_pred_scaled
#define aom_comp_avg_upsampled_pred_c shim_rie_comp_avg_upsampled_pred_c
#define aom_highbd_upsampled_pred_c shim_rie_highbd_upsampled_pred_c
#define aom_highbd_comp_avg_upsampled_pred_c \
  shim_rie_highbd_comp_avg_upsampled_pred_c
#define aom_comp_mask_upsampled_pred shim_rie_comp_mask_upsampled_pred
#define aom_highbd_comp_mask_upsampled_pred \
  shim_rie_highbd_comp_mask_upsampled_pred
#define av1_enc_build_one_inter_predictor shim_rie_enc_build_one_inter_predictor
#define av1_enc_build_inter_predictor_y shim_rie_enc_build_inter_predictor_y
#define av1_enc_build_inter_predictor_y_nonrd \
  shim_rie_enc_build_inter_predictor_y_nonrd
#define av1_enc_build_inter_predictor shim_rie_enc_build_inter_predictor
#define av1_build_prediction_by_above_preds \
  shim_rie_build_prediction_by_above_preds
#define av1_build_prediction_by_left_preds shim_rie_build_prediction_by_left_preds
#define av1_build_obmc_inter_predictors_sb shim_rie_build_obmc_inter_predictors_sb
#define av1_build_inter_predictors_for_planes_single_buf \
  shim_rie_build_inter_predictors_for_planes_single_buf
#define av1_build_wedge_inter_predictor_from_buf \
  shim_rie_build_wedge_inter_predictor_from_buf

/* --- libaom's own encoder-side predictor builders, unmodified. --- */
#include "av1/encoder/reconinter_enc.c"

static void *shim_rie_align(const void *src, size_t bytes) {
  void *p = aom_memalign(64, bytes ? bytes : 64);
  if (!p) return NULL;
  if (src)
    memcpy(p, src, bytes);
  else
    memset(p, 0, bytes);
  return p;
}

/* ======================================================================== *
 * 1. build_masked_compound (:312) / build_masked_compound_highbd (:330).
 *
 * The mask is the caller's, at LUMA stride `block_size_wide[sb_type]`, so the
 * subsampled arms are reachable by passing a `w`/`h` smaller than the luma
 * block's.
 * ======================================================================== */

void shim_rie_build_masked_compound(int hbd, int bd, int sb_type, int h, int w,
                                    const void *src0, int src0_stride,
                                    const void *src1, int src1_stride,
                                    const uint8_t *mask, void *dst,
                                    int dst_stride) {
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  const int mask_stride = block_size_wide[sb_type];
  const int mask_h = block_size_high[sb_type];
  INTERINTER_COMPOUND_DATA comp;
  memset(&comp, 0, sizeof(comp));
  /* COMPOUND_DIFFWTD makes `av1_get_compound_type_mask` return
   * `comp_data->seg_mask`, i.e. exactly the buffer handed in — which is what
   * makes this entry a test of the BLEND rather than of the mask lookup. */
  comp.type = COMPOUND_DIFFWTD;

  void *a0 = shim_rie_align(src0, (size_t)src0_stride * h * px);
  void *a1 = shim_rie_align(src1, (size_t)src1_stride * h * px);
  void *ad = shim_rie_align(NULL, (size_t)dst_stride * h * px);
  uint8_t *am = (uint8_t *)shim_rie_align(mask, (size_t)mask_stride * mask_h);
  comp.seg_mask = am;

#if CONFIG_AV1_HIGHBITDEPTH
  if (hbd) {
    build_masked_compound_highbd(
        (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)ad), dst_stride,
        (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)a0), src0_stride,
        (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)a1), src1_stride, &comp,
        (BLOCK_SIZE)sb_type, h, w, bd);
  } else {
    build_masked_compound((uint8_t *)ad, dst_stride, (uint8_t *)a0,
                          src0_stride, (uint8_t *)a1, src1_stride, &comp,
                          (BLOCK_SIZE)sb_type, h, w);
  }
#else
  (void)bd;
  build_masked_compound((uint8_t *)ad, dst_stride, (uint8_t *)a0, src0_stride,
                        (uint8_t *)a1, src1_stride, &comp, (BLOCK_SIZE)sb_type,
                        h, w);
#endif
  memcpy(dst, ad, (size_t)dst_stride * h * px);

  aom_free(am);
  aom_free(ad);
  aom_free(a1);
  aom_free(a0);
}

/* ======================================================================== *
 * 2. build_wedge_inter_predictor_from_buf (:349), one plane.
 *
 * `comp_meta` = {wedge_index, wedge_sign, mask_type, type}. `seg_mask` is
 * in/out: the plane-0 DIFFWTD arm REBUILDS it from the two ext buffers.
 * ======================================================================== */

void shim_rie_build_wedge_from_buf_plane(
    int hbd, int bd, int bsize, int is_compound, const int32_t *comp_meta,
    int plane, int x, int y, int w, int h, const void *ext0, int ext0_stride,
    const void *ext1, int ext1_stride, void *dst, int dst_stride, int dst_rows,
    uint8_t *seg_mask /* in/out, 2 * MAX_SB_SQUARE */) {
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  YV12_BUFFER_CONFIG *cb =
      (YV12_BUFFER_CONFIG *)calloc(1, sizeof(YV12_BUFFER_CONFIG));
  uint8_t *aseg = (uint8_t *)shim_rie_align(seg_mask, 2 * MAX_SB_SQUARE);
  void *a0 = shim_rie_align(ext0, (size_t)ext0_stride * h * px);
  void *a1 = shim_rie_align(ext1, (size_t)ext1_stride * h * px);
  void *ad = shim_rie_align(dst, (size_t)dst_stride * dst_rows * px);
  if (!xd || !mbmi || !cb || !aseg || !a0 || !a1 || !ad) goto done;

  cb->flags = hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  xd->cur_buf = cb;
  xd->bd = bd;
  xd->seg_mask = aseg;
  MB_MODE_INFO *mi_ptr = mbmi;
  xd->mi = &mi_ptr;
  mbmi->bsize = (BLOCK_SIZE)bsize;
  /* `has_second_ref(mbmi)` is `ref_frame[1] > INTRA_FRAME`. */
  mbmi->ref_frame[0] = LAST_FRAME;
  mbmi->ref_frame[1] = is_compound ? ALTREF_FRAME : NONE_FRAME;
  mbmi->interinter_comp.wedge_index = (int8_t)comp_meta[0];
  mbmi->interinter_comp.wedge_sign = (int8_t)comp_meta[1];
  mbmi->interinter_comp.mask_type = (DIFFWTD_MASK_TYPE)comp_meta[2];
  mbmi->interinter_comp.type = (COMPOUND_TYPE)comp_meta[3];
  xd->plane[plane].dst.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)ad) : (uint8_t *)ad;
  xd->plane[plane].dst.stride = dst_stride;

  build_wedge_inter_predictor_from_buf(xd, plane, x, y, w, h, (uint8_t *)a0,
                                       ext0_stride, (uint8_t *)a1,
                                       ext1_stride);

  memcpy(dst, ad, (size_t)dst_stride * dst_rows * px);
  memcpy(seg_mask, aseg, 2 * MAX_SB_SQUARE);

done:
  aom_free(ad);
  aom_free(a1);
  aom_free(a0);
  aom_free(aseg);
  free(cb);
  free(mbmi);
  free(xd);
}

/* ======================================================================== *
 * 3. av1_build_wedge_inter_predictor_from_buf (:407), the plane loop.
 *
 * Every plane's buffers are packed back-to-back in the caller's allocation at
 * the per-plane strides given; `plane_off` names each plane's start.
 * ======================================================================== */

void shim_rie_build_wedge_from_buf(
    int hbd, int bd, int bsize, int is_compound, const int32_t *comp_meta,
    int plane_from, int plane_to, const int32_t *ss /* [3][2] flattened */,
    const int32_t *plane_off, const int32_t *plane_bytes, const void *ext0,
    const int32_t *ext0_stride, const void *ext1, const int32_t *ext1_stride,
    void *dst, const int32_t *dst_stride, uint8_t *seg_mask) {
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  YV12_BUFFER_CONFIG *cb =
      (YV12_BUFFER_CONFIG *)calloc(1, sizeof(YV12_BUFFER_CONFIG));
  uint8_t *aseg = (uint8_t *)shim_rie_align(seg_mask, 2 * MAX_SB_SQUARE);
  uint8_t *e0[MAX_MB_PLANE] = { NULL, NULL, NULL };
  uint8_t *e1[MAX_MB_PLANE] = { NULL, NULL, NULL };
  uint8_t *ad[MAX_MB_PLANE] = { NULL, NULL, NULL };
  int s0[MAX_MB_PLANE], s1[MAX_MB_PLANE];
  if (!xd || !mbmi || !cb || !aseg) goto done;

  cb->flags = hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  xd->cur_buf = cb;
  xd->bd = bd;
  xd->seg_mask = aseg;
  MB_MODE_INFO *mi_ptr = mbmi;
  xd->mi = &mi_ptr;
  mbmi->bsize = (BLOCK_SIZE)bsize;
  mbmi->ref_frame[0] = LAST_FRAME;
  mbmi->ref_frame[1] = is_compound ? ALTREF_FRAME : NONE_FRAME;
  mbmi->interinter_comp.wedge_index = (int8_t)comp_meta[0];
  mbmi->interinter_comp.wedge_sign = (int8_t)comp_meta[1];
  mbmi->interinter_comp.mask_type = (DIFFWTD_MASK_TYPE)comp_meta[2];
  mbmi->interinter_comp.type = (COMPOUND_TYPE)comp_meta[3];

  for (int p = plane_from; p <= plane_to; ++p) {
    xd->plane[p].subsampling_x = ss[2 * p];
    xd->plane[p].subsampling_y = ss[2 * p + 1];
    const size_t bytes = (size_t)plane_bytes[p] * px;
    e0[p] = (uint8_t *)shim_rie_align((const uint8_t *)ext0 + plane_off[p] * px,
                                      bytes);
    e1[p] = (uint8_t *)shim_rie_align((const uint8_t *)ext1 + plane_off[p] * px,
                                      bytes);
    ad[p] = (uint8_t *)shim_rie_align((const uint8_t *)dst + plane_off[p] * px,
                                      bytes);
    s0[p] = ext0_stride[p];
    s1[p] = ext1_stride[p];
    xd->plane[p].dst.buf = hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)ad[p])
                               : ad[p];
    xd->plane[p].dst.stride = dst_stride[p];
  }

  av1_build_wedge_inter_predictor_from_buf(xd, (BLOCK_SIZE)bsize, plane_from,
                                           plane_to, e0, s0, e1, s1);

  for (int p = plane_from; p <= plane_to; ++p) {
    memcpy((uint8_t *)dst + plane_off[p] * px, ad[p],
           (size_t)plane_bytes[p] * px);
  }
  memcpy(seg_mask, aseg, 2 * MAX_SB_SQUARE);

done:
  for (int p = 0; p < MAX_MB_PLANE; ++p) {
    aom_free(ad[p]);
    aom_free(e1[p]);
    aom_free(e0[p]);
  }
  aom_free(aseg);
  free(cb);
  free(mbmi);
  free(xd);
}

int shim_rie_max_sb_square(void) { return MAX_SB_SQUARE; }

/* ======================================================================== *
 * 4. enc_calc_subpel_params (reconinter_enc.c:32) and, through it,
 *    init_subpel_params (common/reconinter.h:131).
 *
 * Both are `static inline`, so neither has an address; this TU has the real
 * source of both. The source POINTER crosses back as a signed offset from
 * `pre_buf->buf0` rather than as a pointer, so the Rust side compares an
 * arithmetic result instead of an address — and so that the negative offsets
 * a block predicting into the frame border legitimately produces are visible
 * rather than being a wild pointer.
 *
 * `sf` is built by `av1_setup_scale_factors_for_frame`, the REAL exported
 * function (already gated by aom-dsp/tests/scale_diff.rs), so the scaled arm
 * is driven with scale factors the encoder can actually construct.
 * ======================================================================== */

void shim_rie_enc_calc_subpel_params(int mv_row, int mv_col, int pix_row,
                                     int pix_col, int ssx, int ssy,
                                     int ref_w, int ref_h, int this_w,
                                     int this_h, int pre_width, int pre_height,
                                     int pre_stride, int *out /* 7 */) {
  InterPredParams ipp;
  struct scale_factors sf;
  SubpelParams sp;
  uint8_t *pre = NULL;
  int src_stride = 0;
  MV mv;

  memset(&ipp, 0, sizeof(ipp));
  memset(&sp, 0, sizeof(sp));
  av1_setup_scale_factors_for_frame(&sf, ref_w, ref_h, this_w, this_h);

  /* The `top` / `left` reach limits come from init_inter_block_params
   * (reconinter.h:211-212); everything else the derivation reads is set
   * explicitly here. */
  ipp.pix_row = pix_row;
  ipp.pix_col = pix_col;
  ipp.subsampling_x = ssx;
  ipp.subsampling_y = ssy;
  ipp.top = -AOM_LEFT_TOP_MARGIN_SCALED(ssy);
  ipp.left = -AOM_LEFT_TOP_MARGIN_SCALED(ssx);
  ipp.scale_factors = &sf;
  ipp.ref_frame_buf.buf0 = NULL;
  ipp.ref_frame_buf.width = pre_width;
  ipp.ref_frame_buf.height = pre_height;
  ipp.ref_frame_buf.stride = pre_stride;

  mv.row = (int16_t)mv_row;
  mv.col = (int16_t)mv_col;
  enc_calc_subpel_params(&mv, &ipp, &pre, &sp, &src_stride);

  out[0] = sp.xs;
  out[1] = sp.ys;
  out[2] = sp.subpel_x;
  out[3] = sp.subpel_y;
  out[4] = sp.pos_x;
  out[5] = sp.pos_y;
  /* `buf0` is NULL, so `pre` IS the offset — in bytes, which for a lowbd
   * plane is the same as in pixels. That is the whole reason this driver
   * passes NULL rather than a real buffer. */
  out[6] = (int)(intptr_t)pre;
  (void)src_stride;
}

/* The scale factors the same call produced, so the Rust side can build the
 * identical `ScaleFactors` rather than assuming its own constructor agrees. */
void shim_rie_scale_factors(int ref_w, int ref_h, int this_w, int this_h,
                            int *out /* 4 */) {
  struct scale_factors sf;
  av1_setup_scale_factors_for_frame(&sf, ref_w, ref_h, this_w, this_h);
  out[0] = sf.x_scale_fp;
  out[1] = sf.y_scale_fp;
  out[2] = sf.x_step_q4;
  out[3] = sf.y_step_q4;
}
