/* Oracle shims for the compound (two-reference) inter-prediction DSP
 * (crate aom-dsp, module inter::compound). Oracle use only.
 *
 * Every shim here drives the REAL exported C function; none of them
 * re-implements anything. The wrappers exist only to flatten C's pointer /
 * struct arguments into a plain scalar+slice ABI the Rust side can call.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/reconinter.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"

/* ---- av1/encoder/wedge_utils.c ------------------------------------- */

uint64_t shim_wedge_sse_from_residuals(const int16_t *r1, const int16_t *d,
                                       const uint8_t *m, int N) {
  return av1_wedge_sse_from_residuals_c(r1, d, m, N);
}

int shim_wedge_sign_from_residuals(const int16_t *ds, const uint8_t *m, int N,
                                   int64_t limit) {
  return (int)av1_wedge_sign_from_residuals_c(ds, m, N, limit);
}

void shim_wedge_compute_delta_squares(int16_t *d, const int16_t *a,
                                      const int16_t *b, int N) {
  av1_wedge_compute_delta_squares_c(d, a, b, N);
}

/* ---- av1/common/reconinter.c — the DIFFWTD masks ------------------- */

void shim_build_compound_diffwtd_mask(uint8_t *mask, int mask_type,
                                      const uint8_t *src0, int src0_stride,
                                      const uint8_t *src1, int src1_stride,
                                      int h, int w) {
  av1_build_compound_diffwtd_mask_c(mask, (DIFFWTD_MASK_TYPE)mask_type, src0,
                                    src0_stride, src1, src1_stride, h, w);
}

/* ConvolveParams is only read for round_0 / round_1 by diffwtd_mask_d16, so the
 * shim zeroes the struct and sets just those two (plus bd, a separate arg). */
void shim_build_compound_diffwtd_mask_d16(uint8_t *mask, int mask_type,
                                          const uint16_t *src0, int src0_stride,
                                          const uint16_t *src1, int src1_stride,
                                          int h, int w, int round_0,
                                          int round_1, int bd) {
  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.round_0 = round_0;
  cp.round_1 = round_1;
  av1_build_compound_diffwtd_mask_d16_c(mask, (DIFFWTD_MASK_TYPE)mask_type,
                                        src0, src0_stride, src1, src1_stride, h,
                                        w, &cp, bd);
}

/* The highbd entry takes uint8_t* that are really CONVERT_TO_BYTEPTR'd
 * uint16_t*, so the shim converts back the way every libaom caller does. */
void shim_build_compound_diffwtd_mask_highbd(uint8_t *mask, int mask_type,
                                             const uint16_t *src0,
                                             int src0_stride,
                                             const uint16_t *src1,
                                             int src1_stride, int h, int w,
                                             int bd) {
  av1_build_compound_diffwtd_mask_highbd_c(
      mask, (DIFFWTD_MASK_TYPE)mask_type, CONVERT_TO_BYTEPTR(src0), src0_stride,
      CONVERT_TO_BYTEPTR(src1), src1_stride, h, w, bd);
}

/* ---- av1_get_compound_type_mask ------------------------------------
 * COMPOUND_WEDGE arm only (the default arm just returns comp_data->seg_mask,
 * which is the caller's own buffer and needs no oracle). Copies the returned
 * bw*bh mask out at stride bw. av1_init_wedge_masks() must have run; the Rust
 * wrapper calls it.
 */
int shim_get_compound_type_mask_wedge(int bsize, int wedge_index,
                                      int wedge_sign, uint8_t *out) {
  INTERINTER_COMPOUND_DATA cd;
  memset(&cd, 0, sizeof(cd));
  cd.type = COMPOUND_WEDGE;
  cd.wedge_index = (int8_t)wedge_index;
  cd.wedge_sign = (int8_t)wedge_sign;
  const uint8_t *m = av1_get_compound_type_mask(&cd, (BLOCK_SIZE)bsize);
  if (!m) return -1;
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  memcpy(out, m, (size_t)bw * bh);
  return 0;
}

/* ---- av1_dist_wtd_comp_weight_assign --------------------------------
 * Builds the minimal AV1_COMMON + MB_MODE_INFO the function reads:
 *   cm->cur_frame->order_hint
 *   cm->seq_params->order_hint_info
 *   get_ref_frame_buf(cm, mbmi->ref_frame[i])
 *     == cm->ref_frame_map[cm->remapped_ref_idx[ref - LAST_FRAME]]
 * Slot 0 of ref_frame_map carries the backward (ref_frame[0]) order hint and
 * slot 1 the forward (ref_frame[1]) one; `have_bck`/`have_fwd` == 0 leaves the
 * map entry NULL, which C treats as order hint 0.
 */
int shim_dist_wtd_comp_weight_assign(int enable_order_hint,
                                     int order_hint_bits_minus_1,
                                     int cur_order_hint, int fwd_order_hint,
                                     int bck_order_hint, int have_fwd,
                                     int have_bck, int compound_idx,
                                     int is_compound, int *out_fwd,
                                     int *out_bck, int *out_use) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(AV1_COMMON));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(SequenceHeader));
  RefCntBuffer *cur = (RefCntBuffer *)calloc(1, sizeof(RefCntBuffer));
  RefCntBuffer *bck = (RefCntBuffer *)calloc(1, sizeof(RefCntBuffer));
  RefCntBuffer *fwd = (RefCntBuffer *)calloc(1, sizeof(RefCntBuffer));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(MB_MODE_INFO));
  if (!cm || !seq || !cur || !bck || !fwd || !mbmi) {
    free(cm); free(seq); free(cur); free(bck); free(fwd); free(mbmi);
    return -1;
  }

  seq->order_hint_info.enable_order_hint = enable_order_hint;
  seq->order_hint_info.order_hint_bits_minus_1 = order_hint_bits_minus_1;
  cm->seq_params = seq;

  cur->order_hint = cur_order_hint;
  cm->cur_frame = cur;

  bck->order_hint = bck_order_hint;
  fwd->order_hint = fwd_order_hint;
  cm->ref_frame_map[0] = have_bck ? bck : NULL;
  cm->ref_frame_map[1] = have_fwd ? fwd : NULL;
  cm->remapped_ref_idx[LAST_FRAME - LAST_FRAME] = 0;   /* ref_frame[0] */
  cm->remapped_ref_idx[ALTREF_FRAME - LAST_FRAME] = 1; /* ref_frame[1] */

  mbmi->ref_frame[0] = LAST_FRAME;
  mbmi->ref_frame[1] = ALTREF_FRAME;
  mbmi->compound_idx = (uint8_t)compound_idx;

  av1_dist_wtd_comp_weight_assign(cm, mbmi, out_fwd, out_bck, out_use,
                                  is_compound);

  free(cm); free(seq); free(cur); free(bck); free(fwd); free(mbmi);
  return 0;
}

/* ---- av1/common/scale.c --------------------------------------------
 * The scale-factor setup and MV scaling. `av1_setup_scale_factors_for_frame`
 * and `av1_scale_mv` are exported; `av1_scaled_x`/`av1_scaled_y` are static
 * inlines in scale.h, so the shim exposes them by calling them directly out of
 * this translation unit (which is the same source, compiled with the same
 * flags, not a re-implementation).
 */
#include "av1/common/scale.h"

void shim_setup_scale_factors_for_frame(int other_w, int other_h, int this_w,
                                        int this_h, int *out_x_scale_fp,
                                        int *out_y_scale_fp, int *out_x_step_q4,
                                        int *out_y_step_q4) {
  struct scale_factors sf;
  memset(&sf, 0, sizeof(sf));
  av1_setup_scale_factors_for_frame(&sf, other_w, other_h, this_w, this_h);
  *out_x_scale_fp = sf.x_scale_fp;
  *out_y_scale_fp = sf.y_scale_fp;
  *out_x_step_q4 = sf.x_step_q4;
  *out_y_step_q4 = sf.y_step_q4;
}

void shim_scale_mv(int mv_row, int mv_col, int x, int y, int x_scale_fp,
                   int y_scale_fp, int *out_row, int *out_col) {
  struct scale_factors sf;
  memset(&sf, 0, sizeof(sf));
  sf.x_scale_fp = x_scale_fp;
  sf.y_scale_fp = y_scale_fp;
  MV mv = { (int16_t)mv_row, (int16_t)mv_col };
  MV32 res = av1_scale_mv(&mv, x, y, &sf);
  *out_row = res.row;
  *out_col = res.col;
}

int shim_scaled_x(int val, int x_scale_fp) {
  struct scale_factors sf;
  memset(&sf, 0, sizeof(sf));
  sf.x_scale_fp = x_scale_fp;
  return av1_scaled_x(val, &sf);
}

int shim_scaled_y(int val, int y_scale_fp) {
  struct scale_factors sf;
  memset(&sf, 0, sizeof(sf));
  sf.y_scale_fp = y_scale_fp;
  return av1_scaled_y(val, &sf);
}

int shim_valid_ref_frame_size(int ref_w, int ref_h, int this_w, int this_h) {
  return valid_ref_frame_size(ref_w, ref_h, this_w, this_h);
}

int shim_is_scaled(int x_scale_fp, int y_scale_fp) {
  struct scale_factors sf;
  memset(&sf, 0, sizeof(sf));
  sf.x_scale_fp = x_scale_fp;
  sf.y_scale_fp = y_scale_fp;
  return av1_is_scaled(&sf);
}

/* ---- aom_dsp/blend_a64_mask.c — the D16 mask blend ------------------ */
#include "aom_dsp/blend.h"

void shim_lowbd_blend_a64_d16_mask(uint8_t *dst, uint32_t dst_stride,
                                   const uint16_t *src0, uint32_t src0_stride,
                                   const uint16_t *src1, uint32_t src1_stride,
                                   const uint8_t *mask, uint32_t mask_stride,
                                   int w, int h, int subw, int subh,
                                   int round_0, int round_1) {
  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.round_0 = round_0;
  cp.round_1 = round_1;
  aom_lowbd_blend_a64_d16_mask_c(dst, dst_stride, src0, src0_stride, src1,
                                 src1_stride, mask, mask_stride, w, h, subw,
                                 subh, &cp);
}

void shim_highbd_blend_a64_d16_mask(uint16_t *dst, uint32_t dst_stride,
                                    const uint16_t *src0, uint32_t src0_stride,
                                    const uint16_t *src1, uint32_t src1_stride,
                                    const uint8_t *mask, uint32_t mask_stride,
                                    int w, int h, int subw, int subh,
                                    int round_0, int round_1, int bd) {
  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.round_0 = round_0;
  cp.round_1 = round_1;
  aom_highbd_blend_a64_d16_mask_c(CONVERT_TO_BYTEPTR(dst), dst_stride, src0,
                                  src0_stride, src1, src1_stride, mask,
                                  mask_stride, w, h, subw, subh, &cp, bd);
}
