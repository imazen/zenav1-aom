/* Oracle shims for av1/encoder/compound_type.c — the compound / interintra
 * mask RD search.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * `nm -g upstream/build/libaom.a` reports exactly TWO exported symbols for
 * compound_type.c: `av1_compound_type_rd` and `av1_handle_inter_intra_mode`.
 * The other 32 definitions — the wedge and diffwtd mask picks, the type-cost
 * table, the reuse-stats matcher, every enable/prune predicate — are `static`
 * and have no address a differential can take. Driving the two exported
 * entries instead would require a fully built AV1_COMP with real source and
 * reference frames, a motion-search facade and a transform search, i.e. the
 * whole inter RD brain, which is not what is being tested here.
 *
 * So this TU compiles libaom's OWN compound_type.c, unmodified, with its two
 * exported symbols renamed out of the way, and exposes flat wrappers around
 * the statics. The bodies under test are libaom's source, not a transcription
 * of it — the same technique, and the same justification, as
 * `shim/rdopt_shim.c` and `shim/cnn_cscalar.c`.
 *
 * EVIDENCE TIER. **Tier 1c**: the real C source, compiled verbatim, as opposed
 * to tier 1's real exported symbol out of the archive. The gap between the two
 * is that this is a SECOND COMPILATION which could in principle differ from
 * the archive's copy through flags. That gap is closed by measurement:
 * `shim_ct_compound_type_rd` / `shim_ct_handle_inter_intra_mode` re-export
 * BOTH of compound_type.c's real exported functions from this TU, and
 * `compound_type_shim_tu_matches_archive` in tests/compound_type_diff.rs
 * asserts the two addresses' TU agrees with the archive on the shared statics
 * it can reach. See that test for what it does and does not prove.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`, plus the oracle-wide `-ffp-contract=off`). `-DNDEBUG` is
 * separately mandatory for ABI agreement (DIFFERENTIAL_PLAYBOOK §3a(a)):
 * `MACROBLOCK` has an `#ifndef NDEBUG` member, so without it every struct that
 * embeds one — `AV1_COMP` above all — disagrees with the archive about its own
 * field offsets.
 *
 * CONVENTIONS in the wrappers below.
 * - `MACROBLOCK` and `AV1_COMP` are heap-allocated and zeroed; they are large
 *   (tens of KB and megabytes respectively) and stack-allocating them in a
 *   test process is a real risk.
 * - Cost tables cross the boundary as flat `const int *` in C's own row-major
 *   order, so the Rust side never has to reproduce a 2-D array's layout.
 * - Compound types cross as plain `int`, matching COMPOUND_TYPE's
 *   discriminants.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename compound_type.c's two exported symbols so this TU links beside
 * libaom.a. */
#define av1_compound_type_rd shim_ct_compound_type_rd
#define av1_handle_inter_intra_mode shim_ct_handle_inter_intra_mode

/* --- libaom's own compound-type RD search, unmodified. --- */
#include "av1/encoder/compound_type.c"

/* ======================================================================== *
 * 1. Wedge-search enable predicates (compound_type.c:103-121).
 * ======================================================================== */

int shim_ct_enable_wedge_search(unsigned int source_variance,
                                unsigned int disable_wedge_var_thresh) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  x->source_variance = source_variance;
  const int r = (int)enable_wedge_search(x, disable_wedge_var_thresh);
  free(x);
  return r;
}

int shim_ct_enable_wedge_interinter_search(unsigned int source_variance,
                                           unsigned int disable_wedge_var_thresh,
                                           int enable_interinter_wedge) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  x->source_variance = source_variance;
  cpi->sf.inter_sf.disable_interinter_wedge_var_thresh =
      disable_wedge_var_thresh;
  cpi->oxcf.comp_type_cfg.enable_interinter_wedge = enable_interinter_wedge;
  const int r = (int)enable_wedge_interinter_search(x, cpi);
  free(cpi);
  free(x);
  return r;
}

int shim_ct_enable_wedge_interintra_search(unsigned int source_variance,
                                           unsigned int disable_wedge_var_thresh,
                                           int enable_interintra_wedge) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  x->source_variance = source_variance;
  cpi->sf.inter_sf.disable_interintra_wedge_var_thresh =
      disable_wedge_var_thresh;
  cpi->oxcf.comp_type_cfg.enable_interintra_wedge = enable_interintra_wedge;
  const int r = (int)enable_wedge_interintra_search(x, cpi);
  free(cpi);
  free(x);
  return r;
}

/* ======================================================================== *
 * 2. compute_valid_comp_types (compound_type.c:868).
 *
 * `out_types` receives COMPOUND_TYPES ints; the return value is the count of
 * meaningful leading entries.
 * ======================================================================== */

int shim_ct_compute_valid_comp_types(int bsize, int masked_compound_used,
                                     int mode_search_mask,
                                     unsigned int source_variance,
                                     unsigned int disable_wedge_var_thresh,
                                     int enable_interinter_wedge,
                                     int enable_dist_wtd_comp,
                                     int use_dist_wtd_comp_flag,
                                     int enable_diff_wtd_comp,
                                     int *out_types) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));

  x->source_variance = source_variance;
  cpi->sf.inter_sf.disable_interinter_wedge_var_thresh =
      disable_wedge_var_thresh;
  cpi->oxcf.comp_type_cfg.enable_interinter_wedge = enable_interinter_wedge;
  cpi->oxcf.comp_type_cfg.enable_diff_wtd_comp = enable_diff_wtd_comp;
  cpi->sf.inter_sf.use_dist_wtd_comp_flag = use_dist_wtd_comp_flag;
  seq->order_hint_info.enable_dist_wtd_comp = enable_dist_wtd_comp;
  cpi->common.seq_params = seq;

  COMPOUND_TYPE types[COMPOUND_TYPES] = { COMPOUND_AVERAGE, COMPOUND_DISTWTD,
                                          COMPOUND_WEDGE, COMPOUND_DIFFWTD };
  const int n = compute_valid_comp_types(x, cpi, (BLOCK_SIZE)bsize,
                                         masked_compound_used, mode_search_mask,
                                         types);
  for (int i = 0; i < COMPOUND_TYPES; ++i) out_types[i] = (int)types[i];
  free(seq);
  free(cpi);
  free(x);
  return n;
}

/* ======================================================================== *
 * 3. calc_masked_type_cost (compound_type.c:906).
 *
 * The three cost rows arrive already selected by their contexts; this wrapper
 * plants them at context 0 / bsize 0 and calls with those indices, so the
 * function under test still performs its own lookups.
 * ======================================================================== */

void shim_ct_calc_masked_type_cost(const int *comp_group_idx_cost /* [2] */,
                                   const int *comp_idx_cost /* [2] */,
                                   const int *compound_type_cost /* [2] */,
                                   int masked_compound_used,
                                   int *out_cost /* [COMPOUND_TYPES] */) {
  ModeCosts *mc = (ModeCosts *)calloc(1, sizeof(*mc));
  for (int i = 0; i < 2; ++i) {
    mc->comp_group_idx_cost[0][i] = comp_group_idx_cost[i];
    mc->comp_idx_cost[0][i] = comp_idx_cost[i];
    mc->compound_type_cost[0][i] = compound_type_cost[i];
  }
  calc_masked_type_cost(mc, /*bsize=*/(BLOCK_SIZE)0, /*comp_group_idx_ctx=*/0,
                        /*comp_index_ctx=*/0, masked_compound_used, out_cost);
  free(mc);
}

/* ======================================================================== *
 * 4. update_mbmi_for_compound_type (compound_type.c:945).
 * ======================================================================== */

void shim_ct_update_mbmi_for_compound_type(int cur_type, int *out_type,
                                           int *out_comp_group_idx,
                                           int *out_compound_idx) {
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  update_mbmi_for_compound_type(mbmi, (COMPOUND_TYPE)cur_type);
  *out_type = (int)mbmi->interinter_comp.type;
  *out_comp_group_idx = (int)mbmi->comp_group_idx;
  *out_compound_idx = (int)mbmi->compound_idx;
  free(mbmi);
}

/* ======================================================================== *
 * 5. get_interinter_compound_mask_rate (compound_type.c:1026).
 * ======================================================================== */

int shim_ct_get_interinter_compound_mask_rate(int comp_type, int bsize,
                                              int wedge_index,
                                              const int *wedge_idx_cost
                                              /* [MAX_WEDGE_TYPES] */) {
  ModeCosts *mc = (ModeCosts *)calloc(1, sizeof(*mc));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  for (int i = 0; i < MAX_WEDGE_TYPES; ++i)
    mc->wedge_idx_cost[bsize][i] = wedge_idx_cost[i];
  mbmi->bsize = (BLOCK_SIZE)bsize;
  mbmi->interinter_comp.type = (COMPOUND_TYPE)comp_type;
  mbmi->interinter_comp.wedge_index = (int8_t)wedge_index;
  const int r = get_interinter_compound_mask_rate(mc, mbmi);
  free(mbmi);
  free(mc);
  return r;
}

/* ======================================================================== *
 * 6. save_mask_search_results (compound_type.c:1058).
 * ======================================================================== */

int shim_ct_save_mask_search_results(int this_mode, int reuse_level) {
  return save_mask_search_results((PREDICTION_MODE)this_mode, reuse_level);
}

/* ======================================================================== *
 * 7. The top compound-average estimated-RD list (compound_type.c:737, :761).
 * ======================================================================== */

void shim_ct_push_comp_avg_est_rd(int64_t *top /* [TOP_COMP_AVG_EST_RD_COUNT] */,
                                  int64_t tmp_rd, int level) {
  push_comp_avg_est_rd(top, tmp_rd, level);
}

int shim_ct_prune_comp_eval_using_comp_avg_est_rd(
    const int64_t *top /* [TOP_COMP_AVG_EST_RD_COUNT] */, int64_t tmp_rd,
    int64_t ref_best_rd, int level) {
  return (int)prune_comp_eval_using_comp_avg_est_rd(top, tmp_rd, ref_best_rd,
                                                    level);
}

/* ======================================================================== *
 * 8. compute_rd_thresh (compound_type.c:504).
 * ======================================================================== */

int64_t shim_ct_compute_rd_thresh(int rdmult, int total_mode_rate,
                                  int64_t ref_best_rd) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  x->rdmult = rdmult;
  const int64_t r = compute_rd_thresh(x, total_mode_rate, ref_best_rd);
  free(x);
  return r;
}

/* Constants the Rust side asserts it agrees with, rather than re-deriving. */
int shim_ct_top_comp_avg_est_rd_count(void) {
  return TOP_COMP_AVG_EST_RD_COUNT;
}
int shim_ct_compound_types(void) { return COMPOUND_TYPES; }
int shim_ct_max_wedge_types(void) { return MAX_WEDGE_TYPES; }
