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
