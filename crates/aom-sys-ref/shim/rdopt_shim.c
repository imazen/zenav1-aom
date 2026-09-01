/* Oracle shims for av1/encoder/rdopt.c — THE INTER RD BRAIN.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * 105 of rdopt.c's function definitions are `static`: `nm -g upstream/build/
 * libaom.a` reports exactly ten exported symbols for the whole file
 * (`av1_block_error{,_lp,_highbd}_c`, `av1_get_horver_correlation_full_c`,
 * `av1_inter_mode_data_{init,fit}`, `av1_rd_pick_{intra,inter}_mode_sb*`,
 * `av1_rd_pick_inter_mode`). Every decision helper the inter search is built
 * out of — `get_this_mv`, `build_cur_mv`, `get_drl_cost`, `handle_newmv`,
 * `motion_mode_rd`, `skip_mode_rd`, `set_params_rd_pick_inter_mode` — has
 * internal linkage and no address a differential can take.
 *
 * The alternative to this file is hand-derived vectors traced from the C
 * source, which CLAUDE.md ranks last in the evidence hierarchy ("transcribed
 * oracles can carry shared bugs") and which this repo labels tier 4. So
 * instead this TU compiles libaom's OWN rdopt.c, unmodified, with its ten
 * exported symbols renamed out of the way, and exposes flat wrappers around
 * the statics. The bodies under test are libaom's source, not a transcription
 * of it — the same technique, and the same justification, as
 * `shim/cnn_cscalar.c` (the only other shim that includes a libaom .c).
 *
 * EVIDENCE TIER. Call this **tier 1c**: the real C source, compiled verbatim,
 * as opposed to tier 1's real exported symbol out of the archive. The gap
 * between the two is that this is a SECOND COMPILATION — a different TU could
 * in principle differ from the archive's copy through flags. That gap is
 * closed by measurement rather than assertion: `shim_rdc_*` re-exports all ten
 * of rdopt.c's real exported functions from THIS TU, and
 * `rdopt_shim_tu_matches_archive` in tests/rdopt_mv_diff.rs asserts they agree
 * with the archive's `av1_*` symbols on random inputs. If the second
 * compilation ever stopped meaning the same thing, that gate fails.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`, plus the oracle-wide `-ffp-contract=off`) so it is the same
 * source under the same settings as the copy inside libaom.a. `-DNDEBUG` is
 * separately mandatory for ABI agreement (DIFFERENTIAL_PLAYBOOK §3a(a)):
 * `MACROBLOCK` has an `#ifndef NDEBUG` member, so without it every struct that
 * embeds one disagrees with the archive about its own field offsets.
 *
 * CONVENTIONS in the wrappers below.
 * - MVs cross the boundary as `int16_t[2]` = {row, col}, never as a packed
 *   `as_int`, so the Rust side never has to reproduce the union's layout.
 *   The one exception is where C's own sentinel is the packed value
 *   (`INVALID_MV == 0x80008000`, i.e. row == col == -32768), which is
 *   representable as a row/col pair and is passed as one.
 * - Only the ONE `mbmi_ext` row the function reads is filled — the row is
 *   selected by `av1_ref_frame_type(ref_frame)`, so the caller passes the
 *   reference pair and the shim derives the index the same way C does.
 * - `MACROBLOCK` / `MACROBLOCKD` are heap-allocated and zeroed. They are large
 *   (tens of KB) and stack-allocating them in a test process is a real risk.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename rdopt.c's ten exported symbols so this TU links beside libaom.a. */
#define av1_inter_mode_data_init shim_rdc_inter_mode_data_init
#define av1_inter_mode_data_fit shim_rdc_inter_mode_data_fit
#define av1_get_horver_correlation_full_c shim_rdc_get_horver_correlation_full_c
#define av1_block_error_c shim_rdc_block_error_c
#define av1_block_error_lp_c shim_rdc_block_error_lp_c
#define av1_highbd_block_error_c shim_rdc_highbd_block_error_c
#define av1_rd_pick_intra_mode_sb shim_rdc_rd_pick_intra_mode_sb
#define av1_rd_pick_inter_mode shim_rdc_rd_pick_inter_mode
#define av1_rd_pick_inter_mode_sb_seg_skip shim_rdc_rd_pick_inter_mode_sb_seg_skip

/* --- libaom's own inter RD brain, unmodified. --- */
#include "av1/encoder/rdopt.c"

/* ======================================================================== *
 * 0. Helpers shared by the wrappers.
 * ======================================================================== */

static void shim_rd_fill_ext_row(MB_MODE_INFO_EXT *ext, int8_t row,
                                 int ref_mv_count, const int16_t *stack_this,
                                 const int16_t *stack_comp,
                                 const uint16_t *weight) {
  ext->ref_mv_count[row] = (uint8_t)ref_mv_count;
  for (int i = 0; i < MAX_REF_MV_STACK_SIZE; ++i) {
    if (stack_this) {
      ext->ref_mv_stack[row][i].this_mv.as_mv.row = stack_this[2 * i];
      ext->ref_mv_stack[row][i].this_mv.as_mv.col = stack_this[2 * i + 1];
    }
    if (stack_comp) {
      ext->ref_mv_stack[row][i].comp_mv.as_mv.row = stack_comp[2 * i];
      ext->ref_mv_stack[row][i].comp_mv.as_mv.col = stack_comp[2 * i + 1];
    }
    if (weight) ext->weight[row][i] = weight[i];
  }
}

static void shim_rd_fill_global_mvs(MB_MODE_INFO_EXT *ext,
                                    const int16_t *global_mvs) {
  if (!global_mvs) return;
  for (int r = 0; r < REF_FRAMES; ++r) {
    ext->global_mvs[r].as_mv.row = global_mvs[2 * r];
    ext->global_mvs[r].as_mv.col = global_mvs[2 * r + 1];
  }
}

/* ======================================================================== *
 * 1. Pure predicates — no encoder state at all.
 * ======================================================================== */

int shim_rdopt_get_single_mode(int this_mode, int ref_idx) {
  return (int)get_single_mode((PREDICTION_MODE)this_mode, ref_idx);
}

int shim_rdopt_conditional_skipintra(int mode, int best_intra_mode) {
  return conditional_skipintra((PREDICTION_MODE)mode,
                               (PREDICTION_MODE)best_intra_mode);
}

int shim_rdopt_prune_ref_mv_idx_using_qindex(int reduce_inter_modes, int qindex,
                                             int ref_mv_idx) {
  return prune_ref_mv_idx_using_qindex(reduce_inter_modes, qindex, ref_mv_idx);
}

int shim_rdopt_mask_set_bit(int mask, int index) {
  mask_set_bit(&mask, index);
  return mask;
}

int shim_rdopt_mask_check_bit(int mask, int index) {
  return (int)mask_check_bit(mask, index);
}

int shim_rdopt_ref_frame_type(int rf0, int rf1) {
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  return (int)av1_ref_frame_type(rf);
}

/* ======================================================================== *
 * 2. The ref-MV / DRL family — reads one `mbmi_ext` row.
 * ======================================================================== */

int shim_rdopt_check_repeat_ref_mv(int rf0, int rf1, int ref_idx,
                                   int single_mode, int ref_mv_count,
                                   const int16_t *stack_this,
                                   const int16_t *stack_comp,
                                   const int16_t *global_mvs) {
  MB_MODE_INFO_EXT *ext = (MB_MODE_INFO_EXT *)calloc(1, sizeof(*ext));
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  shim_rd_fill_ext_row(ext, av1_ref_frame_type(rf), ref_mv_count, stack_this,
                       stack_comp, NULL);
  shim_rd_fill_global_mvs(ext, global_mvs);
  const int r = check_repeat_ref_mv(ext, ref_idx, rf,
                                    (PREDICTION_MODE)single_mode);
  free(ext);
  return r;
}

int shim_rdopt_get_this_mv(int this_mode, int ref_idx, int ref_mv_idx,
                           int skip_repeated_ref_mv, int rf0, int rf1,
                           int ref_mv_count, const int16_t *stack_this,
                           const int16_t *stack_comp, const int16_t *global_mvs,
                           int16_t *out_mv) {
  MB_MODE_INFO_EXT *ext = (MB_MODE_INFO_EXT *)calloc(1, sizeof(*ext));
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  shim_rd_fill_ext_row(ext, av1_ref_frame_type(rf), ref_mv_count, stack_this,
                       stack_comp, NULL);
  shim_rd_fill_global_mvs(ext, global_mvs);
  int_mv mv;
  mv.as_int = 0;
  const int r = get_this_mv(&mv, (PREDICTION_MODE)this_mode, ref_idx,
                            ref_mv_idx, skip_repeated_ref_mv, rf, ext);
  out_mv[0] = mv.as_mv.row;
  out_mv[1] = mv.as_mv.col;
  free(ext);
  return r;
}

int shim_rdopt_get_drl_cost(int mode, int ref_mv_idx, int rf0, int rf1,
                            int ref_mv_count, const uint16_t *weight,
                            const int *drl_mode_cost0 /* [3][2] flattened */) {
  MB_MODE_INFO_EXT *ext = (MB_MODE_INFO_EXT *)calloc(1, sizeof(*ext));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  const int8_t row = av1_ref_frame_type(rf);
  shim_rd_fill_ext_row(ext, row, ref_mv_count, NULL, NULL, weight);
  mbmi->mode = (PREDICTION_MODE)mode;
  mbmi->ref_mv_idx = (uint8_t)ref_mv_idx;
  int costs[DRL_MODE_CONTEXTS][2];
  for (int c = 0; c < DRL_MODE_CONTEXTS; ++c) {
    costs[c][0] = drl_mode_cost0[2 * c];
    costs[c][1] = drl_mode_cost0[2 * c + 1];
  }
  const int r = get_drl_cost(mbmi, ext, costs, row);
  free(mbmi);
  free(ext);
  return r;
}

int shim_rdopt_get_drl_refmv_count(int rf0, int rf1, int mode,
                                   int ref_mv_count) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  x->mbmi_ext.ref_mv_count[av1_ref_frame_type(rf)] = (uint8_t)ref_mv_count;
  const int r = get_drl_refmv_count(x, rf, (PREDICTION_MODE)mode);
  free(x);
  return r;
}

int shim_rdopt_is_single_newmv_valid(int this_mode, int rf0, int rf1,
                                     int ref_mv_idx,
                                     const uint8_t *single_newmv_valid) {
  /* `single_newmv_valid` is a POINTER to `int[REF_FRAMES]` rows, not an inline
   * array — a calloc'd HandleInterModeArgs leaves it NULL and the first read
   * traps. (Measured: exit 133 / SIGTRAP before this backing store existed.) */
  HandleInterModeArgs *args =
      (HandleInterModeArgs *)calloc(1, sizeof(*args));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  int (*valid)[REF_FRAMES] =
      (int (*)[REF_FRAMES])calloc(MAX_REF_MV_SEARCH, sizeof(*valid));
  mbmi->ref_frame[0] = (MV_REFERENCE_FRAME)rf0;
  mbmi->ref_frame[1] = (MV_REFERENCE_FRAME)rf1;
  mbmi->ref_mv_idx = (uint8_t)ref_mv_idx;
  for (int i = 0; i < MAX_REF_MV_SEARCH; ++i)
    for (int r = 0; r < REF_FRAMES; ++r)
      valid[i][r] = single_newmv_valid[i * REF_FRAMES + r];
  args->single_newmv_valid = valid;
  const int r = is_single_newmv_valid(args, mbmi, (PREDICTION_MODE)this_mode);
  free(valid);
  free(mbmi);
  free(args);
  return r;
}

int shim_rdopt_skip_nearest_near_mv_using_refmv_weight(
    int this_mode, int ref_frame_type, int best_mode, int left_available,
    int up_available, int ref_mv_count, const uint16_t *weight) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  x->e_mbd.left_available = left_available;
  x->e_mbd.up_available = up_available;
  x->mbmi_ext.ref_mv_count[ref_frame_type] = (uint8_t)ref_mv_count;
  for (int i = 0; i < MAX_REF_MV_STACK_SIZE; ++i)
    x->mbmi_ext.weight[ref_frame_type][i] = weight[i];
  const int r = skip_nearest_near_mv_using_refmv_weight(
      x, (PREDICTION_MODE)this_mode, (int8_t)ref_frame_type,
      (PREDICTION_MODE)best_mode);
  free(x);
  return r;
}

/* ======================================================================== *
 * 3. MV clamping — reads the block's edge distances and the fullpel limits.
 * ======================================================================== */

void shim_rdopt_clamp_mv2(int16_t *mv, int mb_to_left_edge, int mb_to_right_edge,
                          int mb_to_top_edge, int mb_to_bottom_edge) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  xd->mb_to_left_edge = mb_to_left_edge;
  xd->mb_to_right_edge = mb_to_right_edge;
  xd->mb_to_top_edge = mb_to_top_edge;
  xd->mb_to_bottom_edge = mb_to_bottom_edge;
  MV m = { mv[0], mv[1] };
  clamp_mv2(&m, xd);
  mv[0] = m.row;
  mv[1] = m.col;
  free(xd);
}

int shim_rdopt_clamp_and_check_mv(const int16_t *in_mv, int16_t *out_mv,
                                  int allow_high_precision_mv,
                                  int cur_frame_force_integer_mv,
                                  int mb_to_left_edge, int mb_to_right_edge,
                                  int mb_to_top_edge, int mb_to_bottom_edge,
                                  const int *fullmv_limits /* col_min, col_max,
                                                              row_min, row_max */) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  cm->features.allow_high_precision_mv = allow_high_precision_mv;
  cm->features.cur_frame_force_integer_mv = cur_frame_force_integer_mv;
  x->e_mbd.mb_to_left_edge = mb_to_left_edge;
  x->e_mbd.mb_to_right_edge = mb_to_right_edge;
  x->e_mbd.mb_to_top_edge = mb_to_top_edge;
  x->e_mbd.mb_to_bottom_edge = mb_to_bottom_edge;
  x->mv_limits.col_min = fullmv_limits[0];
  x->mv_limits.col_max = fullmv_limits[1];
  x->mv_limits.row_min = fullmv_limits[2];
  x->mv_limits.row_max = fullmv_limits[3];
  int_mv in;
  in.as_mv.row = in_mv[0];
  in.as_mv.col = in_mv[1];
  int_mv out;
  out.as_int = 0;
  const int r = clamp_and_check_mv(&out, in, cm, x);
  out_mv[0] = out.as_mv.row;
  out_mv[1] = out.as_mv.col;
  free(x);
  free(cm);
  return r;
}

/* `build_cur_mv` (rdopt.c:2110): the non-NEWMV MV assembly for one candidate.
 * Drives get_this_mv + clamp_and_check_mv over both reference slots.
 * `out_mv` is 4 int16_t = {mv0.row, mv0.col, mv1.row, mv1.col} and carries C's
 * PARTIAL-WRITE semantics: on an early `return 0` the later slot is whatever
 * the caller passed in, which is why it is an in/out buffer.
 */
int shim_rdopt_build_cur_mv(int16_t *out_mv, int this_mode, int rf0, int rf1,
                            int ref_mv_idx, int skip_repeated_ref_mv,
                            int ref_mv_count, const int16_t *stack_this,
                            const int16_t *stack_comp, const int16_t *global_mvs,
                            int allow_high_precision_mv,
                            int cur_frame_force_integer_mv, int mb_to_left_edge,
                            int mb_to_right_edge, int mb_to_top_edge,
                            int mb_to_bottom_edge, const int *fullmv_limits) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  cm->features.allow_high_precision_mv = allow_high_precision_mv;
  cm->features.cur_frame_force_integer_mv = cur_frame_force_integer_mv;
  x->e_mbd.mb_to_left_edge = mb_to_left_edge;
  x->e_mbd.mb_to_right_edge = mb_to_right_edge;
  x->e_mbd.mb_to_top_edge = mb_to_top_edge;
  x->e_mbd.mb_to_bottom_edge = mb_to_bottom_edge;
  x->mv_limits.col_min = fullmv_limits[0];
  x->mv_limits.col_max = fullmv_limits[1];
  x->mv_limits.row_min = fullmv_limits[2];
  x->mv_limits.row_max = fullmv_limits[3];
  mbmi->ref_frame[0] = (MV_REFERENCE_FRAME)rf0;
  mbmi->ref_frame[1] = (MV_REFERENCE_FRAME)rf1;
  mbmi->ref_mv_idx = (uint8_t)ref_mv_idx;
  mbmi->mode = (PREDICTION_MODE)this_mode;
  x->e_mbd.mi = &mbmi;
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  shim_rd_fill_ext_row(&x->mbmi_ext, av1_ref_frame_type(rf), ref_mv_count,
                       stack_this, stack_comp, NULL);
  shim_rd_fill_global_mvs(&x->mbmi_ext, global_mvs);
  int_mv cur_mv[2];
  cur_mv[0].as_mv.row = out_mv[0];
  cur_mv[0].as_mv.col = out_mv[1];
  cur_mv[1].as_mv.row = out_mv[2];
  cur_mv[1].as_mv.col = out_mv[3];
  const int r = build_cur_mv(cur_mv, (PREDICTION_MODE)this_mode, cm, x,
                             skip_repeated_ref_mv);
  out_mv[0] = cur_mv[0].as_mv.row;
  out_mv[1] = cur_mv[0].as_mv.col;
  out_mv[2] = cur_mv[1].as_mv.row;
  out_mv[3] = cur_mv[1].as_mv.col;
  free(mbmi);
  free(x);
  free(cm);
  return r;
}

/* ======================================================================== *
 * 4. Re-exported real exported functions, so the differential can prove this
 *    second compilation agrees with the archive's copy (see header).
 * ======================================================================== */

int64_t shim_rdc_block_error_via_tu(const tran_low_t *coeff,
                                    const tran_low_t *dqcoeff, intptr_t bsz,
                                    int64_t *ssz) {
  return shim_rdc_block_error_c(coeff, dqcoeff, bsz, ssz);
}

void shim_rdc_horver_via_tu(const int16_t *diff, int stride, int w, int h,
                            float *hcorr, float *vcorr) {
  shim_rdc_get_horver_correlation_full_c(diff, stride, w, h, hcorr, vcorr);
}
