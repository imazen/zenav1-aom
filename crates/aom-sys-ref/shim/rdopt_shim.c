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
