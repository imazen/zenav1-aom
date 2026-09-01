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

/* ======================================================================== *
 * 5. The mode / reference skip mask (rdopt.c:4006-4290).
 *
 * `mode_skip_mask_t` crosses the boundary flattened: `pred_modes[REF_FRAMES]`
 * as uint32 and `ref_combo[REF_FRAMES][REF_FRAMES + 1]` as uint8, because
 * `bool` has no guaranteed C ABI width worth relying on across the FFI.
 * ======================================================================== */

static void shim_rd_load_mask(mode_skip_mask_t *mask, const uint32_t *pred_modes,
                              const uint8_t *ref_combo) {
  for (int i = 0; i < REF_FRAMES; ++i) {
    mask->pred_modes[i] = pred_modes[i];
    for (int j = 0; j < REF_FRAMES + 1; ++j)
      mask->ref_combo[i][j] = (bool)ref_combo[i * (REF_FRAMES + 1) + j];
  }
}

static void shim_rd_store_mask(const mode_skip_mask_t *mask, uint32_t *pred_modes,
                               uint8_t *ref_combo) {
  for (int i = 0; i < REF_FRAMES; ++i) {
    pred_modes[i] = mask->pred_modes[i];
    for (int j = 0; j < REF_FRAMES + 1; ++j)
      ref_combo[i * (REF_FRAMES + 1) + j] = (uint8_t)mask->ref_combo[i][j];
  }
}

static void shim_rd_load_combo(bool combo[REF_FRAMES][REF_FRAMES + 1],
                               const uint8_t *ref_combo) {
  for (int i = 0; i < REF_FRAMES; ++i)
    for (int j = 0; j < REF_FRAMES + 1; ++j)
      combo[i][j] = (bool)ref_combo[i * (REF_FRAMES + 1) + j];
}

static void shim_rd_store_combo(const bool combo[REF_FRAMES][REF_FRAMES + 1],
                                uint8_t *ref_combo) {
  for (int i = 0; i < REF_FRAMES; ++i)
    for (int j = 0; j < REF_FRAMES + 1; ++j)
      ref_combo[i * (REF_FRAMES + 1) + j] = (uint8_t)combo[i][j];
}

void shim_rdopt_disable_reference(int ref, uint8_t *ref_combo) {
  bool combo[REF_FRAMES][REF_FRAMES + 1];
  shim_rd_load_combo(combo, ref_combo);
  disable_reference((MV_REFERENCE_FRAME)ref, combo);
  shim_rd_store_combo(combo, ref_combo);
}

void shim_rdopt_disable_inter_references_except_altref(uint8_t *ref_combo) {
  bool combo[REF_FRAMES][REF_FRAMES + 1];
  shim_rd_load_combo(combo, ref_combo);
  disable_inter_references_except_altref(combo);
  shim_rd_store_combo(combo, ref_combo);
}

void shim_rdopt_default_skip_mask(int ref_set, uint32_t *pred_modes,
                                  uint8_t *ref_combo) {
  mode_skip_mask_t mask;
  /* Deliberately NOT zeroed: default_skip_mask's REF_SET_FULL arm memsets the
   * whole struct and its other arms memset only pred_modes, so a pre-poisoned
   * buffer is what proves each arm writes everything it claims to. */
  memset(&mask, 0xa5, sizeof(mask));
  default_skip_mask(&mask, (REF_SET)ref_set);
  shim_rd_store_mask(&mask, pred_modes, ref_combo);
}

int shim_rdopt_mask_says_skip(const uint32_t *pred_modes,
                              const uint8_t *ref_combo, int rf0, int rf1,
                              int this_mode) {
  mode_skip_mask_t mask;
  memset(&mask, 0, sizeof(mask));
  shim_rd_load_mask(&mask, pred_modes, ref_combo);
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  return (int)mask_says_skip(&mask, rf, (PREDICTION_MODE)this_mode);
}

/* ======================================================================== *
 * 6. Neighbour reference matching (rdopt.c:2465-2525, :5048-:5090).
 * ======================================================================== */

int shim_rdopt_match_ref_frame_pair(int mbmi_rf0, int mbmi_rf1, int rf0,
                                    int rf1) {
  MB_MODE_INFO mbmi;
  memset(&mbmi, 0, sizeof(mbmi));
  mbmi.ref_frame[0] = (MV_REFERENCE_FRAME)mbmi_rf0;
  mbmi.ref_frame[1] = (MV_REFERENCE_FRAME)mbmi_rf1;
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  return match_ref_frame_pair(&mbmi, rf);
}

int shim_rdopt_ref_match_found_in_nb_blocks(int cur0, int cur1, int nb0,
                                            int nb1) {
  MB_MODE_INFO cur, nb;
  memset(&cur, 0, sizeof(cur));
  memset(&nb, 0, sizeof(nb));
  cur.ref_frame[0] = (MV_REFERENCE_FRAME)cur0;
  cur.ref_frame[1] = (MV_REFERENCE_FRAME)cur1;
  nb.ref_frame[0] = (MV_REFERENCE_FRAME)nb0;
  nb.ref_frame[1] = (MV_REFERENCE_FRAME)nb1;
  return ref_match_found_in_nb_blocks(&cur, &nb);
}

void shim_rdopt_match_ref_frame(int mbmi_rf0, int mbmi_rf1, int use_intrabc,
                                int rf0, int rf1, int *is_ref_match) {
  MB_MODE_INFO mbmi;
  memset(&mbmi, 0, sizeof(mbmi));
  mbmi.ref_frame[0] = (MV_REFERENCE_FRAME)mbmi_rf0;
  mbmi.ref_frame[1] = (MV_REFERENCE_FRAME)mbmi_rf1;
  mbmi.use_intrabc = (uint8_t)use_intrabc;
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  match_ref_frame(&mbmi, rf, is_ref_match);
}

int shim_rdopt_compound_skip_using_neighbor_refs(
    int this_mode, int rf0, int rf1, int prune_ext_comp_using_neighbors,
    int left_available, int up_available, const int *left_rf, const int *above_rf,
    const int *nb_intrabc) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  MB_MODE_INFO *left = (MB_MODE_INFO *)calloc(1, sizeof(*left));
  MB_MODE_INFO *above = (MB_MODE_INFO *)calloc(1, sizeof(*above));
  left->ref_frame[0] = (MV_REFERENCE_FRAME)left_rf[0];
  left->ref_frame[1] = (MV_REFERENCE_FRAME)left_rf[1];
  left->use_intrabc = (uint8_t)nb_intrabc[0];
  above->ref_frame[0] = (MV_REFERENCE_FRAME)above_rf[0];
  above->ref_frame[1] = (MV_REFERENCE_FRAME)above_rf[1];
  above->use_intrabc = (uint8_t)nb_intrabc[1];
  xd->left_available = left_available;
  xd->up_available = up_available;
  xd->left_mbmi = left;
  xd->above_mbmi = above;
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  const int r = compound_skip_using_neighbor_refs(
      xd, (PREDICTION_MODE)this_mode, rf, prune_ext_comp_using_neighbors);
  free(above);
  free(left);
  free(xd);
  return r;
}

/* `find_ref_match_in_{above,left}_nbs` walk the mi grid through `xd->mi`,
 * `xd->mi_stride`, `xd->width` / `xd->height`. The shim builds a real grid of
 * MB_MODE_INFO pointers so the walk (including its `mi_size_wide[bsize]`
 * stepping, which is the part a port is most likely to get wrong) is the C
 * one. `grid_rf` is `rows * cols * 2` reference indices in raster order and
 * `grid_bsize` the matching per-MI block sizes. */
int shim_rdopt_find_ref_match_in_nbs(int above, int total_mi, int rows,
                                     int cols, int mi_row, int mi_col,
                                     int width, int height, int up_available,
                                     int left_available, const int *grid_rf,
                                     const int *grid_bsize, const int *cur_rf) {
  const int n = rows * cols;
  MB_MODE_INFO *store = (MB_MODE_INFO *)calloc(n, sizeof(*store));
  MB_MODE_INFO **grid = (MB_MODE_INFO **)calloc(n, sizeof(*grid));
  for (int i = 0; i < n; ++i) {
    store[i].ref_frame[0] = (MV_REFERENCE_FRAME)grid_rf[2 * i];
    store[i].ref_frame[1] = (MV_REFERENCE_FRAME)grid_rf[2 * i + 1];
    store[i].bsize = (BLOCK_SIZE)grid_bsize[i];
    grid[i] = &store[i];
  }
  MB_MODE_INFO cur;
  memset(&cur, 0, sizeof(cur));
  cur.ref_frame[0] = (MV_REFERENCE_FRAME)cur_rf[0];
  cur.ref_frame[1] = (MV_REFERENCE_FRAME)cur_rf[1];
  const int cur_idx = mi_row * cols + mi_col;
  grid[cur_idx] = &cur;
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  xd->mi = grid + cur_idx;
  xd->mi_stride = cols;
  xd->mi_row = mi_row;
  xd->mi_col = mi_col;
  xd->width = width;
  xd->height = height;
  xd->up_available = up_available;
  xd->left_available = left_available;
  const int r = above ? find_ref_match_in_above_nbs(total_mi, xd)
                      : find_ref_match_in_left_nbs(total_mi, xd);
  free(xd);
  free(grid);
  free(store);
  return r;
}

/* ======================================================================== *
 * 7. Reference-availability predicates and the RD-order sorts.
 * ======================================================================== */

int shim_rdopt_is_ref_frame_used_by_compound_ref(int ref_frame,
                                                 int skip_ref_frame_mask) {
  return is_ref_frame_used_by_compound_ref(ref_frame, skip_ref_frame_mask);
}

int shim_rdopt_is_ref_frame_used_in_cache(int ref_frame, int have_cache,
                                          int cache_rf0, int cache_rf1) {
  MB_MODE_INFO cache;
  memset(&cache, 0, sizeof(cache));
  cache.ref_frame[0] = (MV_REFERENCE_FRAME)cache_rf0;
  cache.ref_frame[1] = (MV_REFERENCE_FRAME)cache_rf1;
  return is_ref_frame_used_in_cache((MV_REFERENCE_FRAME)ref_frame,
                                    have_cache ? &cache : NULL);
}

int shim_rdopt_fetch_picked_ref_frames_mask(int mi_row, int mi_col, int bsize,
                                            int mib_size, const int *picked) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  x->e_mbd.mi_row = mi_row;
  x->e_mbd.mi_col = mi_col;
  for (int i = 0; i < 32 * 32; ++i) x->picked_ref_frames_mask[i] = picked[i];
  const int r = fetch_picked_ref_frames_mask(x, (BLOCK_SIZE)bsize, mib_size);
  free(x);
  return r;
}

int shim_rdopt_skip_compound_using_best_single_mode_ref(
    int this_mode, int rf0, int rf1, const int *best_single_mode,
    int prune_comp_using_best_single_mode_ref) {
  PREDICTION_MODE best[REF_FRAMES];
  for (int i = 0; i < REF_FRAMES; ++i)
    best[i] = (PREDICTION_MODE)best_single_mode[i];
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  return skip_compound_using_best_single_mode_ref(
      (PREDICTION_MODE)this_mode, rf, best,
      prune_comp_using_best_single_mode_ref);
}

/* `find_top_ref` sorts a copy of ref_frame_rd[1..] with qsort/compare_int64 and
 * writes the 110%-of-best cut-off back into slot 0. Both the sort comparator
 * and the cut-off arithmetic are under test. */
void shim_rdopt_find_top_ref(int64_t *ref_frame_rd) {
  find_top_ref(ref_frame_rd);
}

int shim_rdopt_in_single_ref_cutoff(const int64_t *ref_frame_rd, int f1,
                                    int f2) {
  int64_t copy[REF_FRAMES];
  memcpy(copy, ref_frame_rd, sizeof(copy));
  return (int)in_single_ref_cutoff(copy, (MV_REFERENCE_FRAME)f1,
                                   (MV_REFERENCE_FRAME)f2);
}

/* `inter_modes_info_sort` + `compare_rd_idx_pair`: the est-rd ranking that
 * decides which candidates reach the real transform search. The tie-break on
 * `idx` (aomedia:2928) is the part a port gets wrong, and it is only visible
 * when equal RDs are present, so the harness feeds duplicates deliberately. */
void shim_rdopt_inter_modes_info_sort(int num, const int64_t *est_rd,
                                      int *out_idx, int64_t *out_rd) {
  InterModesInfo *info = (InterModesInfo *)calloc(1, sizeof(*info));
  RdIdxPair *pairs = (RdIdxPair *)calloc(MAX_INTER_MODES, sizeof(*pairs));
  info->num = num;
  for (int i = 0; i < num; ++i) info->est_rd_arr[i] = est_rd[i];
  inter_modes_info_sort(info, pairs);
  for (int i = 0; i < num; ++i) {
    out_idx[i] = pairs[i].idx;
    out_rd[i] = pairs[i].rd;
  }
  free(pairs);
  free(info);
}

int shim_rdopt_compare_int64(int64_t a, int64_t b) {
  return compare_int64(&a, &b);
}

/* ======================================================================== *
 * 8. The inter-mode RD MODEL (rdopt.c:353-467).
 *
 * `av1_inter_mode_data_init` and `av1_inter_mode_data_fit` are two of
 * rdopt.c's ten EXPORTED symbols, so these two are tier 1 proper: the
 * `#undef`s below reach the ARCHIVE's copies rather than this TU's, and the
 * prototypes are restated because the header's were rewritten by the renames
 * at the top of this file. `get_est_rate_dist` and `inter_mode_data_push` are
 * static and come from this TU.
 *
 * The model crosses the boundary as a flat `double[13]` plus `ready`/`num`,
 * in InterModeRdModel declaration order (encoder.h:1248).
 * ======================================================================== */

#undef av1_inter_mode_data_init
#undef av1_inter_mode_data_fit
void av1_inter_mode_data_init(TileDataEnc *tile_data);
void av1_inter_mode_data_fit(TileDataEnc *tile_data, int rdmult);

enum { SHIM_RD_MODEL_DOUBLES = 12 };

static void shim_rd_model_out(const InterModeRdModel *md, int *ready, int *num,
                              double *d) {
  *ready = md->ready;
  *num = md->num;
  d[0] = md->a;
  d[1] = md->b;
  d[2] = md->dist_mean;
  d[3] = md->ld_mean;
  d[4] = md->sse_mean;
  d[5] = md->sse_sse_mean;
  d[6] = md->sse_ld_mean;
  d[7] = md->dist_sum;
  d[8] = md->ld_sum;
  d[9] = md->sse_sum;
  d[10] = md->sse_sse_sum;
  d[11] = md->sse_ld_sum;
}

static void shim_rd_model_in(InterModeRdModel *md, int ready, int num,
                             const double *d) {
  md->ready = ready;
  md->num = num;
  md->a = d[0];
  md->b = d[1];
  md->dist_mean = d[2];
  md->ld_mean = d[3];
  md->sse_mean = d[4];
  md->sse_sse_mean = d[5];
  md->sse_ld_mean = d[6];
  md->dist_sum = d[7];
  md->ld_sum = d[8];
  md->sse_sum = d[9];
  md->sse_sse_sum = d[10];
  md->sse_ld_sum = d[11];
}

/* Real exported `av1_inter_mode_data_init`, driven at one bsize.
 *
 * IN/OUT rather than OUT: `av1_inter_mode_data_init` resets only seven of the
 * fourteen fields (ready, num, dist_sum, ld_sum, sse_sum, sse_sse_sum,
 * sse_ld_sum) and leaves the five means plus `a` and `b` at whatever the
 * caller's allocation held. That is not an oversight to "fix" in the port — a
 * model with `ready == 0` never reads them — but it does mean the differential
 * has to start both sides from the SAME values to compare all fourteen. */
void shim_rdopt_inter_mode_data_init(int bsize, int *ready, int *num,
                                     double *inout) {
  TileDataEnc *td = (TileDataEnc *)calloc(1, sizeof(*td));
  shim_rd_model_in(&td->inter_mode_rd_models[bsize], *ready, *num, inout);
  av1_inter_mode_data_init(td);
  shim_rd_model_out(&td->inter_mode_rd_models[bsize], ready, num, inout);
  free(td);
}

/* Real exported `av1_inter_mode_data_fit`, driven at one bsize. */
void shim_rdopt_inter_mode_data_fit(int bsize, int rdmult, int *ready, int *num,
                                    double *inout) {
  TileDataEnc *td = (TileDataEnc *)calloc(1, sizeof(*td));
  shim_rd_model_in(&td->inter_mode_rd_models[bsize], *ready, *num, inout);
  av1_inter_mode_data_fit(td, rdmult);
  shim_rd_model_out(&td->inter_mode_rd_models[bsize], ready, num, inout);
  free(td);
}

int shim_rdopt_get_est_rate_dist(int bsize, int ready, int num,
                                 const double *model, int64_t sse,
                                 int *est_residue_cost, int64_t *est_dist) {
  TileDataEnc *td = (TileDataEnc *)calloc(1, sizeof(*td));
  shim_rd_model_in(&td->inter_mode_rd_models[bsize], ready, num, model);
  const int r = get_est_rate_dist(td, (BLOCK_SIZE)bsize, sse, est_residue_cost,
                                  est_dist);
  free(td);
  return r;
}

void shim_rdopt_inter_mode_data_push(int bsize, int64_t sse, int64_t dist,
                                     int residue_cost, int *ready, int *num,
                                     double *inout) {
  TileDataEnc *td = (TileDataEnc *)calloc(1, sizeof(*td));
  shim_rd_model_in(&td->inter_mode_rd_models[bsize], *ready, *num, inout);
  inter_mode_data_push(td, (BLOCK_SIZE)bsize, sse, dist, residue_cost);
  shim_rd_model_out(&td->inter_mode_rd_models[bsize], ready, num, inout);
  free(td);
}

int shim_rdopt_inter_mode_data_block_idx(int bsize) {
  return inter_mode_data_block_idx((BLOCK_SIZE)bsize);
}

/* ======================================================================== *
 * 9. NEWMV assembly (rdopt.c:1308-1420) and the two encodemv.c accessors it
 *    is built on (both EXPORTED, so those two are tier 1 proper).
 * ======================================================================== */

void shim_rdopt_clamp_mv_in_range(int16_t *mv, int ref_idx, int this_mode,
                                  int rf0, int rf1, int ref_mv_idx,
                                  int ref_mv_count, const int16_t *stack_this,
                                  const int16_t *stack_comp,
                                  const int16_t *global_mvs,
                                  const int *fullmv_limits) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  mbmi->ref_frame[0] = (MV_REFERENCE_FRAME)rf0;
  mbmi->ref_frame[1] = (MV_REFERENCE_FRAME)rf1;
  mbmi->ref_mv_idx = (uint8_t)ref_mv_idx;
  mbmi->mode = (PREDICTION_MODE)this_mode;
  x->e_mbd.mi = &mbmi;
  x->mv_limits.col_min = fullmv_limits[0];
  x->mv_limits.col_max = fullmv_limits[1];
  x->mv_limits.row_min = fullmv_limits[2];
  x->mv_limits.row_max = fullmv_limits[3];
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  shim_rd_fill_ext_row(&x->mbmi_ext, av1_ref_frame_type(rf), ref_mv_count,
                       stack_this, stack_comp, NULL);
  shim_rd_fill_global_mvs(&x->mbmi_ext, global_mvs);
  int_mv m;
  m.as_mv.row = mv[0];
  m.as_mv.col = mv[1];
  clamp_mv_in_range(x, &m, ref_idx);
  mv[0] = m.as_mv.row;
  mv[1] = m.as_mv.col;
  free(mbmi);
  free(x);
}

/* `av1_get_ref_mv_from_stack` (encodemv.c:302) — real exported symbol.
 * NOTE its single-reference arm falls back to `global_mvs[ref_frame_type]`,
 * which is indexed by the ROW, not by the reference frame. For a single
 * reference those coincide (`av1_ref_frame_type` returns rf[0]); for a
 * compound row they would not, but that arm is unreachable there. */
void shim_rdopt_get_ref_mv_from_stack(int ref_idx, int rf0, int rf1,
                                      int ref_mv_idx, int ref_mv_count,
                                      const int16_t *stack_this,
                                      const int16_t *stack_comp,
                                      const int16_t *global_mvs,
                                      int16_t *out_mv) {
  MB_MODE_INFO_EXT *ext = (MB_MODE_INFO_EXT *)calloc(1, sizeof(*ext));
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  shim_rd_fill_ext_row(ext, av1_ref_frame_type(rf), ref_mv_count, stack_this,
                       stack_comp, NULL);
  shim_rd_fill_global_mvs(ext, global_mvs);
  const int_mv m = av1_get_ref_mv_from_stack(ref_idx, rf, ref_mv_idx, ext);
  out_mv[0] = m.as_mv.row;
  out_mv[1] = m.as_mv.col;
  free(ext);
}

/* `av1_get_ref_mv` (encodemv.c:322) — real exported symbol. Adds the NEAR_NEWMV
 * / NEW_NEARMV `ref_mv_idx + 1` shift on top of the stack accessor. */
void shim_rdopt_get_ref_mv(int ref_idx, int this_mode, int rf0, int rf1,
                           int ref_mv_idx, int ref_mv_count,
                           const int16_t *stack_this, const int16_t *stack_comp,
                           const int16_t *global_mvs, int16_t *out_mv) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
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
  const int_mv m = av1_get_ref_mv(x, ref_idx);
  out_mv[0] = m.as_mv.row;
  out_mv[1] = m.as_mv.col;
  free(mbmi);
  free(x);
}

int shim_rdopt_prune_ref_mv_idx_search(int ref_mv_idx, int best_ref_mv_idx,
                                       int16_t *save_mv /* [2][2][2] */,
                                       int rf0, int rf1, const int16_t *mbmi_mv,
                                       int pruning_factor) {
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  mbmi->ref_frame[0] = (MV_REFERENCE_FRAME)rf0;
  mbmi->ref_frame[1] = (MV_REFERENCE_FRAME)rf1;
  mbmi->mv[0].as_mv.row = mbmi_mv[0];
  mbmi->mv[0].as_mv.col = mbmi_mv[1];
  mbmi->mv[1].as_mv.row = mbmi_mv[2];
  mbmi->mv[1].as_mv.col = mbmi_mv[3];
  int_mv save[MAX_REF_MV_SEARCH - 1][2];
  for (int i = 0; i < MAX_REF_MV_SEARCH - 1; ++i)
    for (int j = 0; j < 2; ++j) {
      save[i][j].as_mv.row = save_mv[(i * 2 + j) * 2];
      save[i][j].as_mv.col = save_mv[(i * 2 + j) * 2 + 1];
    }
  const int r = prune_ref_mv_idx_search(ref_mv_idx, best_ref_mv_idx, save, mbmi,
                                        pruning_factor);
  for (int i = 0; i < MAX_REF_MV_SEARCH - 1; ++i)
    for (int j = 0; j < 2; ++j) {
      save_mv[(i * 2 + j) * 2] = save[i][j].as_mv.row;
      save_mv[(i * 2 + j) * 2 + 1] = save[i][j].as_mv.col;
    }
  free(mbmi);
  return r;
}

/* ======================================================================== *
 * 10. `handle_newmv` (rdopt.c:1317), COMPOUND arm.
 *
 * The single-reference arm calls `av1_single_motion_search`, which needs a
 * whole AV1_COMP plus a source and a reference frame; the compound arm needs
 * none of that and does not read `cpi` at all, so this driver passes NULL for
 * it deliberately (and only ever calls with a compound mode, which is what
 * keeps the single arm unreachable). `mode_info` is likewise single-arm-only.
 *
 * `mvjcost` is MV_JOINTS ints; `mvcost0` / `mvcost1` are MV_VALS ints each,
 * indexed from the START of the allocation (C centres them at MV_MAX with the
 * `nmv_cost_hp` pointers, and this driver reproduces that centring).
 * ======================================================================== */

int shim_rdopt_handle_newmv_compound(
    int16_t *cur_mv /* 4: {mv0.row, mv0.col, mv1.row, mv1.col}, in/out */,
    int *rate_mv, int this_mode, int rf0, int rf1, int ref_mv_idx,
    int ref_mv_count, const int16_t *stack_this, const int16_t *stack_comp,
    const int16_t *global_mvs, const int16_t *single_newmv /* [3][8][2] */,
    const uint8_t *single_newmv_valid /* [3][8] */, const int *fullmv_limits,
    const int *mvjcost, const int *mvcost0, const int *mvcost1) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  MvCosts *mv_costs = (MvCosts *)calloc(1, sizeof(*mv_costs));
  HandleInterModeArgs *args = (HandleInterModeArgs *)calloc(1, sizeof(*args));
  int_mv (*snmv)[REF_FRAMES] =
      (int_mv (*)[REF_FRAMES])calloc(MAX_REF_MV_SEARCH, sizeof(*snmv));
  int (*snmv_valid)[REF_FRAMES] =
      (int (*)[REF_FRAMES])calloc(MAX_REF_MV_SEARCH, sizeof(*snmv_valid));
  int (*snmv_rate)[REF_FRAMES] =
      (int (*)[REF_FRAMES])calloc(MAX_REF_MV_SEARCH, sizeof(*snmv_rate));

  for (int j = 0; j < MV_JOINTS; ++j) mv_costs->nmv_joint_cost[j] = mvjcost[j];
  for (int v = 0; v < MV_VALS; ++v) {
    mv_costs->nmv_cost_hp_alloc[0][v] = mvcost0[v];
    mv_costs->nmv_cost_hp_alloc[1][v] = mvcost1[v];
  }
  mv_costs->nmv_cost_hp[0] = &mv_costs->nmv_cost_hp_alloc[0][MV_MAX];
  mv_costs->nmv_cost_hp[1] = &mv_costs->nmv_cost_hp_alloc[1][MV_MAX];
  mv_costs->mv_cost_stack = mv_costs->nmv_cost_hp;
  x->mv_costs = mv_costs;

  mbmi->ref_frame[0] = (MV_REFERENCE_FRAME)rf0;
  mbmi->ref_frame[1] = (MV_REFERENCE_FRAME)rf1;
  mbmi->ref_mv_idx = (uint8_t)ref_mv_idx;
  mbmi->mode = (PREDICTION_MODE)this_mode;
  x->e_mbd.mi = &mbmi;
  x->mv_limits.col_min = fullmv_limits[0];
  x->mv_limits.col_max = fullmv_limits[1];
  x->mv_limits.row_min = fullmv_limits[2];
  x->mv_limits.row_max = fullmv_limits[3];

  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  shim_rd_fill_ext_row(&x->mbmi_ext, av1_ref_frame_type(rf), ref_mv_count,
                       stack_this, stack_comp, NULL);
  shim_rd_fill_global_mvs(&x->mbmi_ext, global_mvs);

  for (int i = 0; i < MAX_REF_MV_SEARCH; ++i) {
    for (int r = 0; r < REF_FRAMES; ++r) {
      snmv[i][r].as_mv.row = single_newmv[(i * REF_FRAMES + r) * 2];
      snmv[i][r].as_mv.col = single_newmv[(i * REF_FRAMES + r) * 2 + 1];
      snmv_valid[i][r] = single_newmv_valid[i * REF_FRAMES + r];
      snmv_rate[i][r] = 0;
    }
  }
  args->single_newmv = snmv;
  args->single_newmv_valid = snmv_valid;
  args->single_newmv_rate = snmv_rate;

  int_mv mv[2];
  mv[0].as_mv.row = cur_mv[0];
  mv[0].as_mv.col = cur_mv[1];
  mv[1].as_mv.row = cur_mv[2];
  mv[1].as_mv.col = cur_mv[3];

  *rate_mv = 0;
  const int64_t r = handle_newmv(NULL, x, BLOCK_16X16, mv, rate_mv, args, NULL);

  cur_mv[0] = mv[0].as_mv.row;
  cur_mv[1] = mv[0].as_mv.col;
  cur_mv[2] = mv[1].as_mv.row;
  cur_mv[3] = mv[1].as_mv.col;

  free(snmv_rate);
  free(snmv_valid);
  free(snmv);
  free(args);
  free(mv_costs);
  free(mbmi);
  free(x);
  return (int)r;
}

int shim_rdopt_mv_vals(void) { return MV_VALS; }
int shim_rdopt_mv_max(void) { return MV_MAX; }

void shim_rdopt_update_mode_start_end_index(int motion_mode_for_winner_cand,
                                            int extra_prune_warped, int bsize,
                                            int last_motion_mode_allowed,
                                            int interintra_allowed,
                                            int eval_motion_mode, int *start,
                                            int *end) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  cpi->sf.winner_mode_sf.motion_mode_for_winner_cand =
      motion_mode_for_winner_cand;
  cpi->sf.inter_sf.extra_prune_warped = extra_prune_warped;
  mbmi->bsize = (BLOCK_SIZE)bsize;
  update_mode_start_end_index(cpi, mbmi, start, end, last_motion_mode_allowed,
                              interintra_allowed, eval_motion_mode);
  free(mbmi);
  free(cpi);
}

/* ======================================================================== *
 * 11. The SINGLE-REFERENCE STATE table (rdopt.c:4465, :4813-:5046).
 *
 * `InterModeSearchState` is declared inside rdopt.c, so it exists in this TU
 * and nowhere a normal shim could reach it. Only the single-state half of it
 * is exchanged, flattened into `ShimSingleStates`, which the Rust side
 * mirrors as a `#[repr(C)]` struct of the same shape.
 * ======================================================================== */

typedef struct {
  int64_t ss_rd[2][SINGLE_INTER_MODE_NUM][FWD_REFS];
  int32_t ss_ref[2][SINGLE_INTER_MODE_NUM][FWD_REFS];
  int32_t ss_valid[2][SINGLE_INTER_MODE_NUM][FWD_REFS];
  int32_t ss_cnt[2][SINGLE_INTER_MODE_NUM];
  int64_t sm_rd[2][SINGLE_INTER_MODE_NUM][FWD_REFS];
  int32_t sm_ref[2][SINGLE_INTER_MODE_NUM][FWD_REFS];
  int32_t sm_valid[2][SINGLE_INTER_MODE_NUM][FWD_REFS];
  int32_t sm_cnt[2][SINGLE_INTER_MODE_NUM];
  int32_t order[2][SINGLE_INTER_MODE_NUM][FWD_REFS];
} ShimSingleStates;

static void shim_ss_in(InterModeSearchState *st, const ShimSingleStates *s) {
  for (int d = 0; d < 2; ++d)
    for (int m = 0; m < SINGLE_INTER_MODE_NUM; ++m) {
      st->single_state_cnt[d][m] = s->ss_cnt[d][m];
      st->single_state_modelled_cnt[d][m] = s->sm_cnt[d][m];
      for (int r = 0; r < FWD_REFS; ++r) {
        st->single_state[d][m][r].rd = s->ss_rd[d][m][r];
        st->single_state[d][m][r].ref_frame = (MV_REFERENCE_FRAME)s->ss_ref[d][m][r];
        st->single_state[d][m][r].valid = s->ss_valid[d][m][r];
        st->single_state_modelled[d][m][r].rd = s->sm_rd[d][m][r];
        st->single_state_modelled[d][m][r].ref_frame =
            (MV_REFERENCE_FRAME)s->sm_ref[d][m][r];
        st->single_state_modelled[d][m][r].valid = s->sm_valid[d][m][r];
        st->single_rd_order[d][m][r] = (MV_REFERENCE_FRAME)s->order[d][m][r];
      }
    }
}

static void shim_ss_out(const InterModeSearchState *st, ShimSingleStates *s) {
  for (int d = 0; d < 2; ++d)
    for (int m = 0; m < SINGLE_INTER_MODE_NUM; ++m) {
      s->ss_cnt[d][m] = st->single_state_cnt[d][m];
      s->sm_cnt[d][m] = st->single_state_modelled_cnt[d][m];
      for (int r = 0; r < FWD_REFS; ++r) {
        s->ss_rd[d][m][r] = st->single_state[d][m][r].rd;
        s->ss_ref[d][m][r] = st->single_state[d][m][r].ref_frame;
        s->ss_valid[d][m][r] = st->single_state[d][m][r].valid;
        s->sm_rd[d][m][r] = st->single_state_modelled[d][m][r].rd;
        s->sm_ref[d][m][r] = st->single_state_modelled[d][m][r].ref_frame;
        s->sm_valid[d][m][r] = st->single_state_modelled[d][m][r].valid;
        s->order[d][m][r] = st->single_rd_order[d][m][r];
      }
    }
}

void shim_rdopt_init_single_inter_mode_search_state(ShimSingleStates *s) {
  InterModeSearchState *st =
      (InterModeSearchState *)calloc(1, sizeof(*st));
  /* Poison so a field the function fails to reset is visible. */
  memset(st, 0x33, sizeof(*st));
  init_single_inter_mode_search_state(st);
  shim_ss_out(st, s);
  free(st);
}

/* `collect_single_states` also reads `simple_rd[mode][idx][ref]` and
 * `modelled_rd[mode][idx][ref]` for the ref_mv indices this mode allows;
 * only those `MAX_REF_MV_SEARCH` values are exchanged. */
void shim_rdopt_collect_single_states(ShimSingleStates *s, int this_mode,
                                      int ref_frame, int ref_mv_count,
                                      const int64_t *simple_rd,
                                      const int64_t *modelled_rd) {
  InterModeSearchState *st = (InterModeSearchState *)calloc(1, sizeof(*st));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  shim_ss_in(st, s);
  mbmi->ref_frame[0] = (MV_REFERENCE_FRAME)ref_frame;
  mbmi->ref_frame[1] = NONE_FRAME;
  mbmi->mode = (PREDICTION_MODE)this_mode;
  x->mbmi_ext.ref_mv_count[ref_frame] = (uint8_t)ref_mv_count;
  for (int i = 0; i < MAX_REF_MV_SEARCH; ++i) {
    st->simple_rd[this_mode][i][ref_frame] = simple_rd[i];
    st->modelled_rd[this_mode][i][ref_frame] = modelled_rd[i];
  }
  collect_single_states(x, st, mbmi);
  shim_ss_out(st, s);
  free(mbmi);
  free(x);
  free(st);
}

void shim_rdopt_analyze_single_states(ShimSingleStates *s, int prune_level) {
  InterModeSearchState *st = (InterModeSearchState *)calloc(1, sizeof(*st));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  shim_ss_in(st, s);
  cpi->sf.inter_sf.prune_comp_search_by_single_result = prune_level;
  analyze_single_states(cpi, st);
  shim_ss_out(st, s);
  free(cpi);
  free(st);
}

int shim_rdopt_compound_skip_get_candidates(const ShimSingleStates *s,
                                            int prune_level, int dir,
                                            int mode) {
  InterModeSearchState *st = (InterModeSearchState *)calloc(1, sizeof(*st));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  shim_ss_in(st, s);
  cpi->sf.inter_sf.prune_comp_search_by_single_result = prune_level;
  const int r = compound_skip_get_candidates(cpi, st, dir,
                                             (PREDICTION_MODE)mode);
  free(cpi);
  free(st);
  return r;
}

int shim_rdopt_compound_skip_by_single_states(
    const ShimSingleStates *s, int prune_level, int this_mode, int rf0, int rf1,
    int ref_mv_count, const int16_t *stack_this, const int16_t *stack_comp,
    const int16_t *global_mvs, int single0_ref_mv_count,
    const int16_t *single0_stack, int single1_ref_mv_count,
    const int16_t *single1_stack) {
  InterModeSearchState *st = (InterModeSearchState *)calloc(1, sizeof(*st));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  shim_ss_in(st, s);
  cpi->sf.inter_sf.prune_comp_search_by_single_result = prune_level;
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  /* This function reads THREE mbmi_ext rows: the compound pair's, and the two
   * single-reference rows the `single_refs` lookups inside it use. */
  shim_rd_fill_ext_row(&x->mbmi_ext, av1_ref_frame_type(rf), ref_mv_count,
                       stack_this, stack_comp, NULL);
  shim_rd_fill_ext_row(&x->mbmi_ext, (int8_t)rf0, single0_ref_mv_count,
                       single0_stack, NULL, NULL);
  shim_rd_fill_ext_row(&x->mbmi_ext, (int8_t)rf1, single1_ref_mv_count,
                       single1_stack, NULL, NULL);
  shim_rd_fill_global_mvs(&x->mbmi_ext, global_mvs);
  const int r = compound_skip_by_single_states(
      cpi, st, (PREDICTION_MODE)this_mode, (MV_REFERENCE_FRAME)rf0,
      (MV_REFERENCE_FRAME)rf1, x);
  free(x);
  free(cpi);
  free(st);
  return r;
}

/* `skip_repeated_mv` (rdopt.c:1238) reads and WRITES search_state->modelled_rd,
 * so the three-entry row it touches is in/out. */
int shim_rdopt_skip_repeated_mv(int this_mode, int rf0, int rf1,
                                int ref_mv_count, int gm_wmtype,
                                int mode_context, const int *newmv_cost,
                                const int *zeromv_cost, const int *refmv_cost,
                                int64_t *modelled_rd /* MB_MODE_COUNT */) {
  InterModeSearchState *st = (InterModeSearchState *)calloc(1, sizeof(*st));
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  x->mbmi_ext.ref_mv_count[av1_ref_frame_type(rf)] = (uint8_t)ref_mv_count;
  x->mbmi_ext.mode_context[av1_ref_frame_type(rf)] = (int16_t)mode_context;
  cm->global_motion[rf0].wmtype = (TransformationType)gm_wmtype;
  for (int i = 0; i < NEWMV_MODE_CONTEXTS; ++i) {
    x->mode_costs.newmv_mode_cost[i][0] = newmv_cost[2 * i];
    x->mode_costs.newmv_mode_cost[i][1] = newmv_cost[2 * i + 1];
  }
  for (int i = 0; i < GLOBALMV_MODE_CONTEXTS; ++i) {
    x->mode_costs.zeromv_mode_cost[i][0] = zeromv_cost[2 * i];
    x->mode_costs.zeromv_mode_cost[i][1] = zeromv_cost[2 * i + 1];
  }
  for (int i = 0; i < REFMV_MODE_CONTEXTS; ++i) {
    x->mode_costs.refmv_mode_cost[i][0] = refmv_cost[2 * i];
    x->mode_costs.refmv_mode_cost[i][1] = refmv_cost[2 * i + 1];
  }
  for (int m = 0; m < MB_MODE_COUNT; ++m)
    st->modelled_rd[m][0][rf0] = modelled_rd[m];
  const int r = skip_repeated_mv(cm, x, (PREDICTION_MODE)this_mode, rf, st);
  for (int m = 0; m < MB_MODE_COUNT; ++m)
    modelled_rd[m] = st->modelled_rd[m][0][rf0];
  free(x);
  free(cm);
  free(st);
  return r;
}

/* ======================================================================== *
 * 12. Small initialisers and the winner-candidate push.
 * ======================================================================== */

void shim_rdopt_init_comp_avg_est_rd(int level, int64_t *out) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  for (int j = 0; j < TOP_COMP_AVG_EST_RD_COUNT; ++j)
    x->top_comp_avg_est_rd[j] = out[j];
  init_comp_avg_est_rd(x, level);
  for (int j = 0; j < TOP_COMP_AVG_EST_RD_COUNT; ++j)
    out[j] = x->top_comp_avg_est_rd[j];
  free(x);
}

int shim_rdopt_top_comp_avg_est_rd_count(void) {
  return TOP_COMP_AVG_EST_RD_COUNT;
}

void shim_rdopt_init_top_tx_no_split_rd(int level, int64_t *out, int n_blocks,
                                        int n_top) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  for (int i = 0; i < n_blocks; ++i)
    for (int j = 0; j < n_top; ++j)
      x->top_inter_tx_no_split_rd[i][j] = out[i * n_top + j];
  init_top_tx_no_split_rd_for_inter_modes(x, level);
  for (int i = 0; i < n_blocks; ++i)
    for (int j = 0; j < n_top; ++j)
      out[i * n_top + j] = x->top_inter_tx_no_split_rd[i][j];
  free(x);
}

int shim_rdopt_max_tx_blocks_in_max_sb(void) { return MAX_TX_BLOCKS_IN_MAX_SB; }
int shim_rdopt_top_inter_tx_no_split_count(void) {
  return TOP_INTER_TX_NO_SPLIT_COUNT;
}

/* `inter_modes_info_push` appends one candidate; the differential checks the
 * scalar columns and `num`, which are the ones a port must reproduce (the
 * mbmi / RD_STATS copies are memcpys of structs the port models differently). */
void shim_rdopt_inter_modes_info_push(int num_in, int mode_rate, int64_t sse,
                                      int64_t rd, int *num_out,
                                      int *mode_rate_out, int64_t *sse_out,
                                      int64_t *est_rd_out) {
  InterModesInfo *info = (InterModesInfo *)calloc(1, sizeof(*info));
  MB_MODE_INFO mbmi;
  RD_STATS c, cy, cuv;
  memset(&mbmi, 0, sizeof(mbmi));
  memset(&c, 0, sizeof(c));
  memset(&cy, 0, sizeof(cy));
  memset(&cuv, 0, sizeof(cuv));
  info->num = num_in;
  inter_modes_info_push(info, mode_rate, sse, rd, &c, &cy, &cuv, &mbmi);
  *num_out = info->num;
  *mode_rate_out = info->mode_rate_arr[num_in];
  *sse_out = info->sse_arr[num_in];
  *est_rd_out = info->est_rd_arr[num_in];
  free(info);
}

void shim_rdopt_increase_motion_mode_rd(int best_motion_mode,
                                        int this_motion_mode,
                                        int64_t *best_scaled_rd,
                                        int64_t *this_scaled_rd,
                                        int rd_warp_bias_scale_pct,
                                        float rd_obmc_bias_scale_pct) {
  MB_MODE_INFO best, cur;
  memset(&best, 0, sizeof(best));
  memset(&cur, 0, sizeof(cur));
  best.motion_mode = (MOTION_MODE)best_motion_mode;
  cur.motion_mode = (MOTION_MODE)this_motion_mode;
  increase_motion_mode_rd(&best, &cur, best_scaled_rd, this_scaled_rd,
                          rd_warp_bias_scale_pct, rd_obmc_bias_scale_pct);
}

int shim_rdopt_skip_interp_filter_search(int encoding_mode, int reference_mode,
                                         int sf_skip_interp_filter_search,
                                         int winner_mode_ifs,
                                         int is_single_pred) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  cpi->oxcf.mode = (MODE)encoding_mode;
  cpi->common.current_frame.reference_mode = (REFERENCE_MODE)reference_mode;
  cpi->sf.interp_sf.skip_interp_filter_search = sf_skip_interp_filter_search;
  cpi->sf.winner_mode_sf.winner_mode_ifs = winner_mode_ifs;
  const int r = (int)skip_interp_filter_search(cpi, is_single_pred);
  free(cpi);
  return r;
}

/* ======================================================================== *
 * 13. `calc_target_weighted_pred` (rdopt.c:6888) — the OBMC TARGET.
 *
 * One driver covers all three functions: the two per-neighbour visitors
 * (`calc_target_weighted_pred_above` :6752, `_left` :6800) are only reachable
 * through the `foreach_overlappable_nb_{above,left}` walks that this function
 * runs, so driving it exercises the walks, the visitors and the surrounding
 * scaling in one differential.
 *
 * The mi grid is REAL: `rows * cols` MB_MODE_INFO cells with per-cell bsize
 * and reference frame, so the walk's `mi_size_wide[bsize]` stepping, its
 * 4-wide "move to the chroma half of the pair" fixup, and
 * `is_neighbor_overlappable` all run on C's own code path rather than on an
 * assumption about it.
 *
 * Pixels cross as uint16 for both bit depths; the lowbd arm narrows to uint8
 * inside, because C reads `uint8_t *` there and `CONVERT_TO_SHORTPTR` in the
 * hbd arm.
 * ======================================================================== */

int shim_rdopt_calc_target_weighted_pred(
    int bsize, int mi_row, int mi_col, int xd_width, int xd_height,
    int up_available, int left_available, int rows, int cols, int mi_rows,
    int mi_cols, const int *grid_bsize, const int *grid_ref0, int is_hbd,
    const uint16_t *above, int above_stride, const uint16_t *left,
    int left_stride, const uint16_t *src, int src_stride, int32_t *wsrc_out,
    int32_t *mask_out) {
  const int bw = xd_width << MI_SIZE_LOG2;
  const int bh = xd_height << MI_SIZE_LOG2;
  const int n = rows * cols;

  MB_MODE_INFO *store = (MB_MODE_INFO *)calloc(n, sizeof(*store));
  MB_MODE_INFO **grid = (MB_MODE_INFO **)calloc(n, sizeof(*grid));
  for (int i = 0; i < n; ++i) {
    store[i].bsize = (BLOCK_SIZE)grid_bsize[i];
    store[i].ref_frame[0] = (MV_REFERENCE_FRAME)grid_ref0[i];
    store[i].ref_frame[1] = NONE_FRAME;
    grid[i] = &store[i];
  }
  MB_MODE_INFO *cur = (MB_MODE_INFO *)calloc(1, sizeof(*cur));
  cur->bsize = (BLOCK_SIZE)bsize;
  const int cur_idx = mi_row * cols + mi_col;
  grid[cur_idx] = cur;

  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  cm->mi_params.mi_rows = mi_rows;
  cm->mi_params.mi_cols = mi_cols;
  /* `av1_num_planes(cm)` reads seq_params->monochrome; the walk only passes
   * that count through to a visitor that ignores it, but it must not
   * dereference NULL. */
  SequenceHeader seq;
  memset(&seq, 0, sizeof(seq));
  cm->seq_params = &seq;

  MACROBLOCKD *xd = &x->e_mbd;
  xd->mi = grid + cur_idx;
  xd->mi_stride = cols;
  xd->mi_row = mi_row;
  xd->mi_col = mi_col;
  xd->width = xd_width;
  xd->height = xd_height;
  xd->up_available = up_available;
  xd->left_available = left_available;
  xd->plane[0].subsampling_x = 0;
  xd->plane[0].subsampling_y = 0;

  /* `is_cur_buf_hbd(xd)` reads xd->cur_buf->flags. cur_frame is NULL on a
   * calloc'd AV1_COMMON, so a standalone YV12 buffer is used instead. */
  YV12_BUFFER_CONFIG buf;
  memset(&buf, 0, sizeof(buf));
  buf.flags = is_hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  xd->cur_buf = &buf;

  /* Plane / neighbour buffers. For lowbd C dereferences uint8_t*, so the
   * uint16 inputs are narrowed into scratch buffers first. */
  uint8_t *above8 = NULL, *left8 = NULL, *src8 = NULL;
  const int above_n = above_stride * (bh + 64);
  const int left_n = left_stride * (bh + 64);
  const int src_n = src_stride * (bh + 64);
  if (!is_hbd) {
    above8 = (uint8_t *)malloc(above_n);
    left8 = (uint8_t *)malloc(left_n);
    src8 = (uint8_t *)malloc(src_n);
    for (int i = 0; i < above_n; ++i) above8[i] = (uint8_t)above[i];
    for (int i = 0; i < left_n; ++i) left8[i] = (uint8_t)left[i];
    for (int i = 0; i < src_n; ++i) src8[i] = (uint8_t)src[i];
    x->plane[0].src.buf = src8;
  } else {
    x->plane[0].src.buf = CONVERT_TO_BYTEPTR(src);
  }
  x->plane[0].src.stride = src_stride;

  int32_t *wsrc = (int32_t *)calloc(bw * bh, sizeof(*wsrc));
  int32_t *mask = (int32_t *)calloc(bw * bh, sizeof(*mask));
  x->obmc_buffer.wsrc = wsrc;
  x->obmc_buffer.mask = mask;

  calc_target_weighted_pred(
      cm, x, xd, is_hbd ? CONVERT_TO_BYTEPTR(above) : above8, above_stride,
      is_hbd ? CONVERT_TO_BYTEPTR(left) : left8, left_stride);

  memcpy(wsrc_out, wsrc, (size_t)bw * bh * sizeof(*wsrc));
  memcpy(mask_out, mask, (size_t)bw * bh * sizeof(*mask));

  free(mask);
  free(wsrc);
  free(src8);
  free(left8);
  free(above8);
  free(x);
  free(cm);
  free(cur);
  free(grid);
  free(store);
  return bw * bh;
}

/* ======================================================================== *
 * 14. Variance-based RD adjustment (rdopt.c:624-866).
 *
 * `get_variance_stats` / `_hbd` read the block's source and its reconstructed
 * prediction, each through a 1-pixel replicated border, and measure how much
 * high-frequency energy a 3x3 Gaussian removes. `adjust_cost` / `adjust_rdcost`
 * turn the difference into an RD penalty.
 * ======================================================================== */

void shim_rdopt_get_variance_stats(int bsize, int is_hbd, const uint16_t *src,
                                   int src_stride, const uint16_t *dst,
                                   int dst_stride, int64_t *src_var,
                                   int64_t *rec_var) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  mbmi->bsize = (BLOCK_SIZE)bsize;
  x->e_mbd.mi = &mbmi;
  const int bh = block_size_high[bsize];
  uint8_t *src8 = NULL, *dst8 = NULL;
  if (is_hbd) {
    x->plane[0].src.buf = CONVERT_TO_BYTEPTR(src);
    x->e_mbd.plane[0].dst.buf = CONVERT_TO_BYTEPTR(dst);
  } else {
    src8 = (uint8_t *)malloc((size_t)src_stride * (bh + 8));
    dst8 = (uint8_t *)malloc((size_t)dst_stride * (bh + 8));
    for (int i = 0; i < src_stride * (bh + 8); ++i) src8[i] = (uint8_t)src[i];
    for (int i = 0; i < dst_stride * (bh + 8); ++i) dst8[i] = (uint8_t)dst[i];
    x->plane[0].src.buf = src8;
    x->e_mbd.plane[0].dst.buf = dst8;
  }
  x->plane[0].src.stride = src_stride;
  x->e_mbd.plane[0].dst.stride = dst_stride;
  if (is_hbd)
    get_variance_stats_hbd(x, src_var, rec_var);
  else
    get_variance_stats(x, src_var, rec_var);
  free(dst8);
  free(src8);
  free(mbmi);
  free(x);
}

/* `adjust_cost` / `adjust_rdcost` gate on three cpi fields: the tuning mode,
 * the sharpness level, and `frame_is_kf_gf_arf`. The last reads
 * `cpi->ppi->gf_group.update_type[cpi->gf_frame_index]` plus
 * `frame_is_intra_only(cm)`, so both are set explicitly here. */
static void shim_rd_set_adjust_gates(AV1_COMP *cpi, AV1_PRIMARY *ppi,
                                     int tuning, int sharpness,
                                     int frame_is_intra, int update_type) {
  cpi->ppi = ppi;
  cpi->oxcf.tune_cfg.tuning = (aom_tune_metric)tuning;
  cpi->oxcf.algo_cfg.sharpness = sharpness;
  cpi->common.current_frame.frame_type = frame_is_intra ? KEY_FRAME : INTER_FRAME;
  cpi->gf_frame_index = 0;
  ppi->gf_group.update_type[0] = (FRAME_UPDATE_TYPE)update_type;
}

int64_t shim_rdopt_adjust_cost(int64_t rd_cost, int is_inter_pred, int tuning,
                               int sharpness, int frame_is_intra,
                               int update_type, int rdmult, int bsize,
                               int is_hbd, const uint16_t *src, int src_stride,
                               const uint16_t *dst, int dst_stride) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  AV1_PRIMARY *ppi = (AV1_PRIMARY *)calloc(1, sizeof(*ppi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  shim_rd_set_adjust_gates(cpi, ppi, tuning, sharpness, frame_is_intra,
                           update_type);
  mbmi->bsize = (BLOCK_SIZE)bsize;
  x->e_mbd.mi = &mbmi;
  x->rdmult = rdmult;
  YV12_BUFFER_CONFIG buf;
  memset(&buf, 0, sizeof(buf));
  buf.flags = is_hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  x->e_mbd.cur_buf = &buf;
  const int bh = block_size_high[bsize];
  uint8_t *src8 = NULL, *dst8 = NULL;
  if (is_hbd) {
    x->plane[0].src.buf = CONVERT_TO_BYTEPTR(src);
    x->e_mbd.plane[0].dst.buf = CONVERT_TO_BYTEPTR(dst);
  } else {
    src8 = (uint8_t *)malloc((size_t)src_stride * (bh + 8));
    dst8 = (uint8_t *)malloc((size_t)dst_stride * (bh + 8));
    for (int i = 0; i < src_stride * (bh + 8); ++i) src8[i] = (uint8_t)src[i];
    for (int i = 0; i < dst_stride * (bh + 8); ++i) dst8[i] = (uint8_t)dst[i];
    x->plane[0].src.buf = src8;
    x->e_mbd.plane[0].dst.buf = dst8;
  }
  x->plane[0].src.stride = src_stride;
  x->e_mbd.plane[0].dst.stride = dst_stride;
  adjust_cost(cpi, x, &rd_cost, (bool)is_inter_pred);
  free(dst8);
  free(src8);
  free(mbmi);
  free(x);
  free(ppi);
  free(cpi);
  return rd_cost;
}

void shim_rdopt_adjust_rdcost(int64_t *rate_dist_rdcost /* 3 in/out */,
                              int is_inter_pred, int tuning, int sharpness,
                              int frame_is_intra, int update_type, int rdmult,
                              int bsize, int is_hbd, const uint16_t *src,
                              int src_stride, const uint16_t *dst,
                              int dst_stride) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  AV1_PRIMARY *ppi = (AV1_PRIMARY *)calloc(1, sizeof(*ppi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  shim_rd_set_adjust_gates(cpi, ppi, tuning, sharpness, frame_is_intra,
                           update_type);
  mbmi->bsize = (BLOCK_SIZE)bsize;
  x->e_mbd.mi = &mbmi;
  x->rdmult = rdmult;
  YV12_BUFFER_CONFIG buf;
  memset(&buf, 0, sizeof(buf));
  buf.flags = is_hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  x->e_mbd.cur_buf = &buf;
  const int bh = block_size_high[bsize];
  uint8_t *src8 = NULL, *dst8 = NULL;
  if (is_hbd) {
    x->plane[0].src.buf = CONVERT_TO_BYTEPTR(src);
    x->e_mbd.plane[0].dst.buf = CONVERT_TO_BYTEPTR(dst);
  } else {
    src8 = (uint8_t *)malloc((size_t)src_stride * (bh + 8));
    dst8 = (uint8_t *)malloc((size_t)dst_stride * (bh + 8));
    for (int i = 0; i < src_stride * (bh + 8); ++i) src8[i] = (uint8_t)src[i];
    for (int i = 0; i < dst_stride * (bh + 8); ++i) dst8[i] = (uint8_t)dst[i];
    x->plane[0].src.buf = src8;
    x->e_mbd.plane[0].dst.buf = dst8;
  }
  x->plane[0].src.stride = src_stride;
  x->e_mbd.plane[0].dst.stride = dst_stride;
  RD_STATS rd;
  av1_init_rd_stats(&rd);
  rd.rate = (int)rate_dist_rdcost[0];
  rd.dist = rate_dist_rdcost[1];
  rd.rdcost = rate_dist_rdcost[2];
  adjust_rdcost(cpi, x, &rd, (bool)is_inter_pred);
  rate_dist_rdcost[0] = rd.rate;
  rate_dist_rdcost[1] = rd.dist;
  rate_dist_rdcost[2] = rd.rdcost;
  free(dst8);
  free(src8);
  free(mbmi);
  free(x);
  free(ppi);
  free(cpi);
}

/* ======================================================================== *
 * 15. Two more search-loop predicates.
 * ======================================================================== */

int shim_rdopt_inter_mode_compatible_skip(int bsize, int curr_mode, int rf0,
                                          int rf1, int ref_frame_flags,
                                          int frame_is_intra, int reference_mode,
                                          int seg_enabled, int seg_ref_active) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  x->e_mbd.mi = &mbmi;
  mbmi->segment_id = 0;
  cpi->ref_frame_flags = ref_frame_flags;
  cpi->common.current_frame.frame_type = frame_is_intra ? KEY_FRAME : INTER_FRAME;
  cpi->common.current_frame.reference_mode = (REFERENCE_MODE)reference_mode;
  cpi->common.seg.enabled = seg_enabled;
  if (seg_ref_active) cpi->common.seg.feature_mask[0] |= 1 << SEG_LVL_REF_FRAME;
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  const int r = inter_mode_compatible_skip(cpi, x, (BLOCK_SIZE)bsize,
                                           (PREDICTION_MODE)curr_mode, rf);
  free(mbmi);
  free(x);
  free(cpi);
  return r;
}

int shim_rdopt_ref_mv_idx_early_breakout(
    int reduce_inter_modes, int prune_comp, int this_mode, int rf0, int rf1,
    int ref_mv_idx, int qindex, int rdmult, int64_t ref_best_rd,
    int nearest_past_ref, int nearest_future_ref, int ref_mv_count,
    const uint16_t *weight, const int *drl_mode_cost0, int ref_frame_cost,
    int single_comp_cost, const uint8_t *single_newmv_valid,
    int *out_ref_mv_idx) {
  SPEED_FEATURES *sf = (SPEED_FEATURES *)calloc(1, sizeof(*sf));
  RefFrameDistanceInfo *dist =
      (RefFrameDistanceInfo *)calloc(1, sizeof(*dist));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  HandleInterModeArgs *args = (HandleInterModeArgs *)calloc(1, sizeof(*args));
  int (*valid)[REF_FRAMES] =
      (int (*)[REF_FRAMES])calloc(MAX_REF_MV_SEARCH, sizeof(*valid));
  (void)prune_comp;
  sf->inter_sf.reduce_inter_modes = reduce_inter_modes;
  dist->nearest_past_ref = (MV_REFERENCE_FRAME)nearest_past_ref;
  dist->nearest_future_ref = (MV_REFERENCE_FRAME)nearest_future_ref;
  mbmi->ref_frame[0] = (MV_REFERENCE_FRAME)rf0;
  mbmi->ref_frame[1] = (MV_REFERENCE_FRAME)rf1;
  mbmi->mode = (PREDICTION_MODE)this_mode;
  x->e_mbd.mi = &mbmi;
  x->qindex = qindex;
  x->rdmult = rdmult;
  const MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0,
                                     (MV_REFERENCE_FRAME)rf1 };
  const int8_t row = av1_ref_frame_type(rf);
  x->mbmi_ext.ref_mv_count[row] = (uint8_t)ref_mv_count;
  for (int i = 0; i < MAX_REF_MV_STACK_SIZE; ++i)
    x->mbmi_ext.weight[row][i] = weight[i];
  for (int c = 0; c < DRL_MODE_CONTEXTS; ++c) {
    x->mode_costs.drl_mode_cost0[c][0] = drl_mode_cost0[2 * c];
    x->mode_costs.drl_mode_cost0[c][1] = drl_mode_cost0[2 * c + 1];
  }
  for (int i = 0; i < MAX_REF_MV_SEARCH; ++i)
    for (int r = 0; r < REF_FRAMES; ++r)
      valid[i][r] = single_newmv_valid[i * REF_FRAMES + r];
  args->single_newmv_valid = valid;
  args->ref_frame_cost = ref_frame_cost;
  args->single_comp_cost = single_comp_cost;
  const int r =
      (int)ref_mv_idx_early_breakout(sf, dist, x, args, ref_best_rd, ref_mv_idx);
  /* mbmi->ref_mv_idx is WRITTEN by the function partway through, and the
   * caller relies on that side effect. */
  *out_ref_mv_idx = mbmi->ref_mv_idx;
  free(valid);
  free(args);
  free(mbmi);
  free(x);
  free(dist);
  free(sf);
  return r;
}
