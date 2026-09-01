/* Oracle shims for the FILE-STATIC decision helpers of
 * av1/encoder/nonrd_pickmode.c -- the speed 8/9 inter pickmode.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * `nm -g` on the object file reports exactly TWO exported symbols for the
 * whole file: av1_nonrd_pick_intra_mode and av1_nonrd_pick_inter_mode_sb.
 * Everything the inter search decides with -- the mode-skip cascade, the
 * compound prunes, the tx-size and early-term heuristics -- is static and has
 * no address a differential can take. Hand-derived vectors are tier 4 here,
 * and re-deriving the expected value inside the test would only compare the
 * port against a second transcription of the same logic.
 *
 * EVIDENCE TIER 1c -- the real C source compiled verbatim, with its two
 * exported symbols renamed out of the way. Same technique and same
 * justification as shim/rdopt_shim.c; read that file's header for the full
 * argument. The second-compilation gap is measured, not assumed:
 * `shim_nrp_tu_*` re-exports this TU's `mode_idx` table and
 * `nonrd_pick_shim_tu_matches_headers` compares it against the port's.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`). `-DNDEBUG` is separately mandatory for ABI agreement
 * (DIFFERENTIAL_PLAYBOOK §3a(a)) and doubly so here:
 * previous_mode_performed_poorly asserts on its own input.
 *
 * CONVENTIONS. The 2-D arrays these helpers take are indexed
 * `[mode][REF_FRAMES]`, so they cross the boundary FLAT in row-major order and
 * the shim re-forms them. int_mv crosses as a packed `as_int`, because that is
 * the form C compares (`mv.as_int != 0`, `frame_mv[a].as_int == frame_mv[b]`)
 * and reproducing the union's row/col split would add a conversion the C code
 * never performs.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename nonrd_pickmode.c's two exported symbols. --- */
#define av1_nonrd_pick_intra_mode shim_nrp_nonrd_pick_intra_mode
#define av1_nonrd_pick_inter_mode_sb shim_nrp_nonrd_pick_inter_mode_sb

/* --- libaom's own nonrd pickmode, unmodified. --- */
#include "av1/encoder/nonrd_pickmode.c"

/* ---- the mode_idx table (nonrd_opt.h:127), for the TU-vs-header gate. ---- */
void shim_nrp_mode_idx(int *out) {
  for (int r = 0; r < REF_FRAMES; ++r)
    for (int m = 0; m < RTC_MODES; ++m) out[r * RTC_MODES + m] = mode_idx[r][m];
}

int shim_nrp_rtc_modes(void) { return RTC_MODES; }
int shim_nrp_rtc_inter_modes(void) { return RTC_INTER_MODES; }
int shim_nrp_ref_frames(void) { return REF_FRAMES; }

/* ---- skip_mode_by_threshold (nonrd_pickmode.c:1933) --------------------- */
int shim_nrp_skip_mode_by_threshold(int mode, int ref_frame, uint32_t mv_as_int,
                                    int frames_since_golden,
                                    const int *rd_threshes,
                                    const int *rd_thresh_freq_fact,
                                    int64_t best_cost, int best_skip,
                                    int extra_shift) {
  int_mv mv;
  mv.as_int = mv_as_int;
  return skip_mode_by_threshold((PREDICTION_MODE)mode,
                                (MV_REFERENCE_FRAME)ref_frame, mv,
                                frames_since_golden, rd_threshes,
                                rd_thresh_freq_fact, best_cost, best_skip,
                                extra_shift);
}

/* ---- skip_mode_by_low_temp (:1961) -------------------------------------- */
int shim_nrp_skip_mode_by_low_temp(int mode, int ref_frame, int bsize,
                                   int source_sad_nonrd, uint32_t mv_as_int,
                                   int force_skip_low_temp_var) {
  int_mv mv;
  mv.as_int = mv_as_int;
  CONTENT_STATE_SB content_state_sb;
  memset(&content_state_sb, 0, sizeof(content_state_sb));
  content_state_sb.source_sad_nonrd = (SOURCE_SAD)source_sad_nonrd;
  return skip_mode_by_low_temp((PREDICTION_MODE)mode,
                               (MV_REFERENCE_FRAME)ref_frame,
                               (BLOCK_SIZE)bsize, content_state_sb, mv,
                               force_skip_low_temp_var);
}

/* ---- skip_mode_by_bsize_and_ref_frame (:1978) --------------------------- */
int shim_nrp_skip_mode_by_bsize_and_ref_frame(int mode, int ref_frame,
                                              int bsize, int extra_prune,
                                              unsigned int sse_zeromv_norm,
                                              int more_prune, int skip_nearmv) {
  return skip_mode_by_bsize_and_ref_frame(
      (PREDICTION_MODE)mode, (MV_REFERENCE_FRAME)ref_frame, (BLOCK_SIZE)bsize,
      extra_prune, sse_zeromv_norm, more_prune, skip_nearmv);
}

/* ---- skip_comp_based_on_var (:2165) ------------------------------------- */
int shim_nrp_skip_comp_based_on_var(const unsigned int *single_vars_flat,
                                    int bsize) {
  unsigned int vars[RTC_INTER_MODES][REF_FRAMES];
  memcpy(vars, single_vars_flat, sizeof(vars));
  return skip_comp_based_on_var(vars, (BLOCK_SIZE)bsize) ? 1 : 0;
}

/* ---- previous_mode_performed_poorly (:2286) ----------------------------- */
int shim_nrp_previous_mode_performed_poorly(int mode, int ref_frame,
                                            const unsigned int *vars_flat,
                                            const int64_t *uv_dist_flat) {
  unsigned int vars[RTC_INTER_MODES][REF_FRAMES];
  int64_t uv_dist[RTC_INTER_MODES][REF_FRAMES];
  memcpy(vars, vars_flat, sizeof(vars));
  memcpy(uv_dist, uv_dist_flat, sizeof(uv_dist));
  return previous_mode_performed_poorly((PREDICTION_MODE)mode,
                                        (MV_REFERENCE_FRAME)ref_frame, vars,
                                        uv_dist)
             ? 1
             : 0;
}

/* ---- prune_compoundmode_with_singlemode_var (:2306) ---------------------
 * frame_mv and mode_checked are [MB_MODE_COUNT][REF_FRAMES], NOT
 * [RTC_INTER_MODES][...]: C indexes them by the raw PREDICTION_MODE.
 */
int shim_nrp_prune_compoundmode_with_singlemode_var(
    int compound_mode, int ref_frame, int ref_frame2,
    const uint32_t *frame_mv_flat, const uint8_t *mode_checked_flat,
    const unsigned int *vars_flat, const int64_t *uv_dist_flat) {
  static int_mv frame_mv[MB_MODE_COUNT][REF_FRAMES];
  static uint8_t mode_checked[MB_MODE_COUNT][REF_FRAMES];
  unsigned int vars[RTC_INTER_MODES][REF_FRAMES];
  int64_t uv_dist[RTC_INTER_MODES][REF_FRAMES];
  for (int m = 0; m < MB_MODE_COUNT; ++m)
    for (int r = 0; r < REF_FRAMES; ++r)
      frame_mv[m][r].as_int = frame_mv_flat[m * REF_FRAMES + r];
  memcpy(mode_checked, mode_checked_flat, sizeof(mode_checked));
  memcpy(vars, vars_flat, sizeof(vars));
  memcpy(uv_dist, uv_dist_flat, sizeof(uv_dist));
  return prune_compoundmode_with_singlemode_var(
             (PREDICTION_MODE)compound_mode, (MV_REFERENCE_FRAME)ref_frame,
             (MV_REFERENCE_FRAME)ref_frame2, frame_mv, mode_checked, vars,
             uv_dist)
             ? 1
             : 0;
}

int shim_nrp_mb_mode_count(void) { return MB_MODE_COUNT; }

/* ---- ac_thr_factor (:580) ----------------------------------------------- */
int shim_nrp_ac_thr_factor(int speed, int width, int height, int norm_sum) {
  return ac_thr_factor(speed, width, height, norm_sum);
}

/* ---- calculate_variance (:556) ------------------------------------------
 * bw / bh are b_width_log2 / b_height_log2 of the BLOCK, not pixel counts.
 * The input arrays are nw*nh entries; the output arrays are (nw/2)*(nh/2).
 */
void shim_nrp_calculate_variance(int bw, int bh, int tx_size,
                                 const unsigned int *sse_i, const int *sum_i,
                                 unsigned int *var_o, unsigned int *sse_o,
                                 int *sum_o) {
  /* calculate_variance takes non-const pointers but only reads them. */
  calculate_variance(bw, bh, (TX_SIZE)tx_size, (unsigned int *)sse_i,
                     (int *)sum_i, var_o, sse_o, sum_o);
}

/* ======================================================================== *
 * The tx-size / subpel-precision / MV-bias cluster.
 * ======================================================================== */

/* ---- subpel_select (:99) ------------------------------------------------
 * Returns a SUBPEL_FORCE_STOP. `mv` crosses as a packed as_fullmv pair, and
 * `ref_mv` / `start_mv` as row/col pairs, since C compares their components.
 */
int shim_nrp_subpel_select(int avg_frame_low_motion,
                           int reduce_mv_pel_precision_highmotion,
                           int reduce_mv_pel_precision_lowcomplex,
                           int subpel_force_stop, int cm_width, int cm_height,
                           int bsize, int16_t mv_row, int16_t mv_col,
                           int16_t ref_mv_row, int16_t ref_mv_col,
                           int16_t start_mv_row, int16_t start_mv_col,
                           int qindex, int source_sad_nonrd,
                           int source_variance, int fullpel_performed_well) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  if (!cpi || !x) {
    free(cpi);
    free(x);
    return -1;
  }
  cpi->rc.avg_frame_low_motion = avg_frame_low_motion;
  cpi->sf.rt_sf.reduce_mv_pel_precision_highmotion =
      reduce_mv_pel_precision_highmotion;
  cpi->sf.rt_sf.reduce_mv_pel_precision_lowcomplex =
      reduce_mv_pel_precision_lowcomplex;
  cpi->sf.mv_sf.subpel_force_stop = (SUBPEL_FORCE_STOP)subpel_force_stop;
  cpi->common.width = cm_width;
  cpi->common.height = cm_height;
  x->qindex = qindex;
  x->content_state_sb.source_sad_nonrd = (SOURCE_SAD)source_sad_nonrd;
  x->source_variance = source_variance;

  int_mv mv;
  mv.as_fullmv.row = mv_row;
  mv.as_fullmv.col = mv_col;
  MV ref_mv = { ref_mv_row, ref_mv_col };
  FULLPEL_MV start_mv = { start_mv_row, start_mv_col };
  const int r = subpel_select(cpi, x, (BLOCK_SIZE)bsize, &mv, ref_mv, start_mv,
                              fullpel_performed_well != 0);
  free(cpi);
  free(x);
  return r;
}

/* ---- use_aggressive_subpel_search_method (:155) ------------------------- */
int shim_nrp_use_aggressive_subpel_search_method(int qindex,
                                                 int source_sad_nonrd,
                                                 int source_variance,
                                                 int use_adaptive,
                                                 int fullpel_performed_well) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  if (!x) return -1;
  x->qindex = qindex;
  x->content_state_sb.source_sad_nonrd = (SOURCE_SAD)source_sad_nonrd;
  x->source_variance = source_variance;
  const int r = use_aggressive_subpel_search_method(
                    x, use_adaptive != 0, fullpel_performed_well != 0)
                    ? 1
                    : 0;
  free(x);
  return r;
}

/* ---- set_force_skip_flag (:423) ----------------------------------------- */
int shim_nrp_set_force_skip_flag(int tx_mode_search_type,
                                 int tx_size_level_based_on_qstep,
                                 int dequant_ac, int bd, unsigned int sse,
                                 int source_variance, int color_sens_u,
                                 int color_sens_v, int force_skip_in) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  int16_t *dq = (int16_t *)calloc(8, sizeof(int16_t));
  if (!cpi || !x || !dq) {
    free(cpi);
    free(x);
    free(dq);
    return -1;
  }
  dq[1] = (int16_t)dequant_ac;
  x->txfm_search_params.tx_mode_search_type = (TX_MODE)tx_mode_search_type;
  cpi->sf.rt_sf.tx_size_level_based_on_qstep = tx_size_level_based_on_qstep;
  x->plane[AOM_PLANE_Y].dequant_QTX = dq;
  x->e_mbd.bd = bd;
  x->source_variance = source_variance;
  x->color_sensitivity[COLOR_SENS_IDX(AOM_PLANE_U)] = (uint8_t)color_sens_u;
  x->color_sensitivity[COLOR_SENS_IDX(AOM_PLANE_V)] = (uint8_t)color_sens_v;
  int force_skip = force_skip_in;
  set_force_skip_flag(cpi, x, sse, &force_skip);
  free(cpi);
  free(x);
  free(dq);
  return force_skip;
}

/* ---- calculate_tx_size (:447) -------------------------------------------
 * Returns the TX_SIZE; `force_skip` is in/out and comes back in *force_skip_io.
 */
int shim_nrp_calculate_tx_size(int tx_mode_search_type,
                               int tx_size_level_based_on_qstep, int aq_mode,
                               int segment_id, int bsize, int qindex,
                               int dequant_ac, int bd, unsigned int var,
                               unsigned int sse, int source_variance,
                               int color_sens_u, int color_sens_v,
                               int *force_skip_io) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mi = (MB_MODE_INFO *)calloc(1, sizeof(*mi));
  MB_MODE_INFO **slot = (MB_MODE_INFO **)calloc(1, sizeof(*slot));
  int16_t *dq = (int16_t *)calloc(8, sizeof(int16_t));
  if (!cpi || !x || !mi || !slot || !dq) {
    free(cpi); free(x); free(mi); free(slot); free(dq);
    return -1;
  }
  dq[1] = (int16_t)dequant_ac;
  x->txfm_search_params.tx_mode_search_type = (TX_MODE)tx_mode_search_type;
  cpi->sf.rt_sf.tx_size_level_based_on_qstep = tx_size_level_based_on_qstep;
  cpi->oxcf.q_cfg.aq_mode = (AQ_MODE)aq_mode;
  x->qindex = qindex;
  x->plane[AOM_PLANE_Y].dequant_QTX = dq;
  x->e_mbd.bd = bd;
  x->source_variance = source_variance;
  x->color_sensitivity[COLOR_SENS_IDX(AOM_PLANE_U)] = (uint8_t)color_sens_u;
  x->color_sensitivity[COLOR_SENS_IDX(AOM_PLANE_V)] = (uint8_t)color_sens_v;
  mi->segment_id = (int8_t)segment_id;
  slot[0] = mi;
  x->e_mbd.mi = slot;

  const int r = (int)calculate_tx_size(cpi, (BLOCK_SIZE)bsize, x, var, sse,
                                       force_skip_io);
  free(cpi); free(x); free(mi); free(slot); free(dq);
  return r;
}

/* ---- newmv_diff_bias (:988) ---------------------------------------------
 * `above_valid` / `left_valid` select whether xd->above_mbmi / left_mbmi are
 * non-NULL at all; their MVs cross packed, because C tests `as_int !=
 * INVALID_MV` on the union.
 */
int64_t shim_nrp_newmv_diff_bias(int this_mode, int64_t rdcost_in, int bsize,
                                 int mv_row, int mv_col, int speed,
                                 uint32_t spatial_variance,
                                 int source_sad_nonrd, int above_valid,
                                 uint32_t above_mv_as_int, int left_valid,
                                 uint32_t left_mv_as_int) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  MB_MODE_INFO *above = (MB_MODE_INFO *)calloc(1, sizeof(*above));
  MB_MODE_INFO *left = (MB_MODE_INFO *)calloc(1, sizeof(*left));
  if (!xd || !above || !left) {
    free(xd); free(above); free(left);
    return -1;
  }
  above->mv[0].as_int = above_mv_as_int;
  left->mv[0].as_int = left_mv_as_int;
  xd->above_mbmi = above_valid ? above : NULL;
  xd->left_mbmi = left_valid ? left : NULL;

  RD_STATS rdc;
  memset(&rdc, 0, sizeof(rdc));
  rdc.rdcost = rdcost_in;
  CONTENT_STATE_SB cs;
  memset(&cs, 0, sizeof(cs));
  cs.source_sad_nonrd = (SOURCE_SAD)source_sad_nonrd;
  newmv_diff_bias(xd, (PREDICTION_MODE)this_mode, &rdc, (BLOCK_SIZE)bsize,
                  mv_row, mv_col, speed, spatial_variance, cs);
  const int64_t r = rdc.rdcost;
  free(xd); free(above); free(left);
  return r;
}

/* ---- update_thresh_freq_fact (:1045) ------------------------------------
 * thresh_freq_fact is [BLOCK_SIZES_ALL][MAX_MODES]; it crosses flat and comes
 * back modified.
 */
int shim_nrp_thresh_freq_fact_dims(int *bsizes, int *modes) {
  MACROBLOCK x;
  *bsizes = (int)(sizeof(x.thresh_freq_fact) / sizeof(x.thresh_freq_fact[0]));
  *modes = (int)(sizeof(x.thresh_freq_fact[0]) /
                 sizeof(x.thresh_freq_fact[0][0]));
  return 0;
}

int shim_nrp_update_thresh_freq_fact(int adaptive_rd_thresh, int bsize,
                                     int ref_frame, int best_mode_idx,
                                     int mode, int *freq_fact_flat) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  if (!cpi || !x) {
    free(cpi);
    free(x);
    return -1;
  }
  cpi->sf.inter_sf.adaptive_rd_thresh = adaptive_rd_thresh;
  const int nb = (int)(sizeof(x->thresh_freq_fact) /
                       sizeof(x->thresh_freq_fact[0]));
  const int nm = (int)(sizeof(x->thresh_freq_fact[0]) /
                       sizeof(x->thresh_freq_fact[0][0]));
  memcpy(x->thresh_freq_fact, freq_fact_flat,
         (size_t)nb * (size_t)nm * sizeof(int));
  update_thresh_freq_fact(cpi, x, (BLOCK_SIZE)bsize,
                          (MV_REFERENCE_FRAME)ref_frame,
                          (THR_MODES)best_mode_idx, (PREDICTION_MODE)mode);
  memcpy(freq_fact_flat, x->thresh_freq_fact,
         (size_t)nb * (size_t)nm * sizeof(int));
  free(cpi);
  free(x);
  return 0;
}
