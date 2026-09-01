/* Oracle shims for the FILE-STATIC half of av1/encoder/tpl_model.c.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * `nm -g` on `upstream/build/libaom.a`'s `tpl_model.c.o` reports 21 exported
 * symbols out of 63 definitions. The backward-propagation core — the whole
 * point of TPL — is in the other 42: `tpl_model_update_b` (the per-reference
 * propagation), `tpl_model_update`, `tpl_model_store`, `round_floor`,
 * `rate_estimator`, `get_gop_length`, `eval_gop_length`, `skip_tpl_for_frame`,
 * `is_alike_mv` and `compare_sad` all have internal linkage, and none of them
 * has an exported caller a differential can drive without building a complete
 * AV1_COMP with real source and reference frames.
 *
 * So this TU compiles libaom's OWN tpl_model.c, unmodified, with its 21
 * exported symbols renamed out of the way, and exposes flat wrappers around
 * the statics. The bodies under test are libaom's source, not a transcription
 * of it — the same technique, and the same justification, as
 * `shim/rdopt_shim.c` and `shim/cnn_cscalar.c`.
 *
 * EVIDENCE TIER. **Tier 1c**: the real C source, compiled verbatim, as
 * opposed to tier 1's real exported symbol out of the archive. The gap is
 * that this is a SECOND COMPILATION, which could in principle differ from the
 * archive's copy through flags. That gap is closed by measurement, not
 * assertion: the renames below give this TU its own copy of all 21 exported
 * functions under `shim_tplc_*` names, and
 * `tpl_c_shim_tu_matches_archive` in tests/tpl_model_diff.rs asserts a sample
 * of them agrees with the archive's `av1_*` symbols on random inputs. If the
 * second compilation ever stopped meaning the same thing, that gate fails.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`, plus the oracle-wide `-ffp-contract=off`) so it is the same
 * source under the same settings as the copy inside libaom.a. `-DNDEBUG` is
 * separately mandatory for ABI agreement (DIFFERENTIAL_PLAYBOOK §3a(a)).
 *
 * CONVENTIONS in the wrappers below.
 * - `TplParams` is heap-allocated and zeroed; it embeds 105 `TplDepFrame`s
 *   and 48 `YV12_BUFFER_CONFIG`s and must not go on a test stack.
 * - Per-frame TPL grids arrive as ONE concatenated array plus a per-frame
 *   offset array, so a caller can give each frame a different stride and the
 *   differential can actually see which frame's stride C reads.
 * - MVs cross as `int16_t[2]` = {row, col}, never as a packed `as_int`.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename tpl_model.c's 21 exported symbols so this TU links beside
 * libaom.a. The `shim_tplc_` copies are what the TU-agreement gate compares
 * against the archive. */
#define av1_compute_mv_difference shim_tplc_compute_mv_difference
#define av1_delta_rate_cost shim_tplc_delta_rate_cost
#define av1_estimate_coeff_entropy shim_tplc_estimate_coeff_entropy
#define av1_exponential_entropy shim_tplc_exponential_entropy
#define av1_free_tpl_gop_stats shim_tplc_free_tpl_gop_stats
#define av1_get_overlap_area shim_tplc_get_overlap_area
#define av1_get_q_index_from_qstep_ratio shim_tplc_get_q_index_from_qstep_ratio
#define av1_init_tpl_stats shim_tplc_init_tpl_stats
#define av1_init_tpl_txfm_stats shim_tplc_init_tpl_txfm_stats
#define av1_laplace_entropy shim_tplc_laplace_entropy
#define av1_mc_flow_dispenser_row shim_tplc_mc_flow_dispenser_row
#define av1_setup_tpl_buffers shim_tplc_setup_tpl_buffers
#define av1_tpl_compute_frame_mv_entropy shim_tplc_tpl_compute_frame_mv_entropy
#define av1_tpl_get_q_index shim_tplc_tpl_get_q_index
#define av1_tpl_get_qstep_ratio shim_tplc_tpl_get_qstep_ratio
#define av1_tpl_preload_rc_estimate shim_tplc_tpl_preload_rc_estimate
#define av1_tpl_ptr_pos shim_tplc_tpl_ptr_pos
#define av1_tpl_rdmult_setup shim_tplc_tpl_rdmult_setup
#define av1_tpl_rdmult_setup_sb shim_tplc_tpl_rdmult_setup_sb
#define av1_tpl_setup_stats shim_tplc_tpl_setup_stats
#define av1_tpl_stats_ready shim_tplc_tpl_stats_ready

/* --- libaom's own temporal dependency model, unmodified. --- */
#include "av1/encoder/tpl_model.c"

/* ======================================================================== *
 * 1. Scalar statics.
 * ======================================================================== */

/* round_floor (tpl_model.c:1149) — floor division, spelled C's way. */
int shim_tplc_round_floor(int ref_pos, int bsize_pix) {
  return round_floor(ref_pos, bsize_pix);
}

/* rate_estimator (tpl_model.c:228) — the modelled coefficient rate TPL uses
 * in place of a real entropy coder. `qcoeff` is in raster order; the function
 * walks it through the DCT_DCT scan for `tx_size`. */
int shim_tplc_rate_estimator(const int32_t *qcoeff, int eob, int tx_size) {
  return rate_estimator((const tran_low_t *)qcoeff, eob, (TX_SIZE)tx_size);
}

/* get_gop_length (tpl_model.c:1318) — the GF group size clamped to the TPL
 * buffer. Only `gf_group->size` is read. */
int shim_tplc_get_gop_length(int gf_group_size) {
  GF_GROUP *gf = (GF_GROUP *)calloc(1, sizeof(*gf));
  if (!gf) return INT_MIN;
  gf->size = gf_group_size;
  int r = get_gop_length(gf);
  free(gf);
  return r;
}

/* eval_gop_length (tpl_model.c:1868) — the three-way GOP-length verdict. */
int shim_tplc_eval_gop_length(double beta0, double beta1, int gop_eval) {
  double beta[2] = { beta0, beta1 };
  return eval_gop_length(beta, gop_eval);
}

/* skip_tpl_for_frame (tpl_model.c:1908) — whether a GOP frame gets TPL stats.
 * Reads `update_type` and `layer_depth` at `frame_idx` plus `size`. */
int shim_tplc_skip_tpl_for_frame(int gf_group_size, int frame_idx,
                                 int update_type, int layer_depth, int gop_eval,
                                 int approx_gop_eval, int reduce_num_frames) {
  GF_GROUP *gf = (GF_GROUP *)calloc(1, sizeof(*gf));
  if (!gf) return INT_MIN;
  gf->size = gf_group_size;
  if (frame_idx >= 0 && frame_idx < MAX_STATIC_GF_GROUP_LENGTH) {
    gf->update_type[frame_idx] = (unsigned char)update_type;
    gf->layer_depth[frame_idx] = (unsigned char)layer_depth;
  }
  int r = skip_tpl_for_frame(gf, frame_idx, gop_eval, approx_gop_eval,
                             reduce_num_frames);
  free(gf);
  return r;
}

/* is_alike_mv (tpl_model.c:345) — the near-duplicate test that prunes TPL's
 * motion-search start candidates. `center_mvs` arrives as {row, col} pairs. */
int shim_tplc_is_alike_mv(int16_t cand_row, int16_t cand_col,
                          const int16_t *center_mvs, int center_mvs_count,
                          int skip_alike_starting_mv) {
  center_mv_t *list =
      (center_mv_t *)calloc((size_t)(center_mvs_count > 0 ? center_mvs_count : 1),
                            sizeof(*list));
  if (!list) return -1;
  for (int i = 0; i < center_mvs_count; ++i) {
    list[i].mv.as_mv.row = center_mvs[2 * i];
    list[i].mv.as_mv.col = center_mvs[2 * i + 1];
    list[i].sad = 0;
  }
  int_mv cand;
  cand.as_mv.row = cand_row;
  cand.as_mv.col = cand_col;
  int r = is_alike_mv(cand, list, center_mvs_count, skip_alike_starting_mv);
  free(list);
  return r;
}

/* compare_sad (tpl_model.c:336) — the qsort comparator over center_mv_t. */
int shim_tplc_compare_sad(int sad_a, int sad_b) {
  center_mv_t a, b;
  memset(&a, 0, sizeof(a));
  memset(&b, 0, sizeof(b));
  a.sad = sad_a;
  b.sad = sad_b;
  return compare_sad(&a, &b);
}

/* ======================================================================== *
 * 2. tpl_model_store (tpl_model.c:1290).
 * ======================================================================== */

/* The whole TplDepStats record crosses the boundary, because the function is
 * a struct copy followed by eleven AOMMAX(1, .) floors. Layout is flat, in
 * the order the Rust side reads it back. */
typedef struct {
  int64_t srcrf_sse, srcrf_dist, recrf_sse, recrf_dist, intra_sse, intra_dist;
  int64_t cmp_recrf_dist[2];
  int64_t mc_dep_rate, mc_dep_dist;
  int64_t pred_error[INTER_REFS_PER_FRAME];
  int32_t intra_cost, inter_cost, srcrf_rate, recrf_rate, intra_rate;
  int32_t cmp_recrf_rate[2];
  int16_t mv[2 * INTER_REFS_PER_FRAME];
  int8_t ref_frame_index[2];
} ShimTplDepStats;

static void shim_tplc_from_flat(TplDepStats *dst, const ShimTplDepStats *src) {
  memset(dst, 0, sizeof(*dst));
  dst->srcrf_sse = src->srcrf_sse;
  dst->srcrf_dist = src->srcrf_dist;
  dst->recrf_sse = src->recrf_sse;
  dst->recrf_dist = src->recrf_dist;
  dst->intra_sse = src->intra_sse;
  dst->intra_dist = src->intra_dist;
  dst->cmp_recrf_dist[0] = src->cmp_recrf_dist[0];
  dst->cmp_recrf_dist[1] = src->cmp_recrf_dist[1];
  dst->mc_dep_rate = src->mc_dep_rate;
  dst->mc_dep_dist = src->mc_dep_dist;
  for (int i = 0; i < INTER_REFS_PER_FRAME; ++i) {
    dst->pred_error[i] = src->pred_error[i];
    dst->mv[i].as_mv.row = src->mv[2 * i];
    dst->mv[i].as_mv.col = src->mv[2 * i + 1];
  }
  dst->intra_cost = src->intra_cost;
  dst->inter_cost = src->inter_cost;
  dst->srcrf_rate = src->srcrf_rate;
  dst->recrf_rate = src->recrf_rate;
  dst->intra_rate = src->intra_rate;
  dst->cmp_recrf_rate[0] = src->cmp_recrf_rate[0];
  dst->cmp_recrf_rate[1] = src->cmp_recrf_rate[1];
  dst->ref_frame_index[0] = src->ref_frame_index[0];
  dst->ref_frame_index[1] = src->ref_frame_index[1];
}

static void shim_tplc_to_flat(ShimTplDepStats *dst, const TplDepStats *src) {
  memset(dst, 0, sizeof(*dst));
  dst->srcrf_sse = src->srcrf_sse;
  dst->srcrf_dist = src->srcrf_dist;
  dst->recrf_sse = src->recrf_sse;
  dst->recrf_dist = src->recrf_dist;
  dst->intra_sse = src->intra_sse;
  dst->intra_dist = src->intra_dist;
  dst->cmp_recrf_dist[0] = src->cmp_recrf_dist[0];
  dst->cmp_recrf_dist[1] = src->cmp_recrf_dist[1];
  dst->mc_dep_rate = src->mc_dep_rate;
  dst->mc_dep_dist = src->mc_dep_dist;
  for (int i = 0; i < INTER_REFS_PER_FRAME; ++i) {
    dst->pred_error[i] = src->pred_error[i];
    dst->mv[2 * i] = src->mv[i].as_mv.row;
    dst->mv[2 * i + 1] = src->mv[i].as_mv.col;
  }
  dst->intra_cost = src->intra_cost;
  dst->inter_cost = src->inter_cost;
  dst->srcrf_rate = src->srcrf_rate;
  dst->recrf_rate = src->recrf_rate;
  dst->intra_rate = src->intra_rate;
  dst->cmp_recrf_rate[0] = src->cmp_recrf_rate[0];
  dst->cmp_recrf_rate[1] = src->cmp_recrf_rate[1];
  dst->ref_frame_index[0] = src->ref_frame_index[0];
  dst->ref_frame_index[1] = src->ref_frame_index[1];
}

/* Writes `src` into a grid of `n_stats` cells at (mi_row, mi_col) and returns
 * the stored cell, so the caller sees both the index arithmetic and the
 * eleven floors. */
int shim_tplc_tpl_model_store(int mi_row, int mi_col, int stride,
                              uint8_t block_mis_log2, int n_stats,
                              const ShimTplDepStats *src,
                              ShimTplDepStats *out_cell, int *out_index) {
  if (n_stats <= 0) return -1;
  TplDepStats *grid = (TplDepStats *)calloc((size_t)n_stats, sizeof(*grid));
  if (!grid) return -1;
  TplDepStats in;
  shim_tplc_from_flat(&in, src);
  int index = shim_tplc_tpl_ptr_pos(mi_row, mi_col, stride, block_mis_log2);
  if (index < 0 || index >= n_stats) {
    free(grid);
    return -2;
  }
  tpl_model_store(grid, mi_row, mi_col, stride, &in, block_mis_log2);
  shim_tplc_to_flat(out_cell, &grid[index]);
  *out_index = index;
  free(grid);
  return 0;
}

/* ======================================================================== *
 * 3. tpl_model_update / tpl_model_update_b (tpl_model.c:1204, :1280).
 *
 * The backward propagation: the cell at (mi_row, mi_col) of frame `frame_idx`
 * pushes a share of its dependency cost into each of its up-to-two reference
 * frames, split across the up-to-four grid cells its motion vector overlaps.
 *
 * Every frame's grid is a slice of ONE concatenated array (`offsets[i]` gives
 * frame i's first cell), and each frame carries its own mi_rows / mi_cols /
 * stride — so a differential can see which frame's stride C actually reads
 * when it locates the SOURCE cell (spoiler: `tpl_frame->stride`, i.e. frame
 * 0's, not `frame_idx`'s).
 * ======================================================================== */
int shim_tplc_tpl_model_update(int n_frames, int frame_idx, int mi_row,
                               int mi_col, uint8_t block_mis_log2,
                               const int32_t *mi_rows, const int32_t *mi_cols,
                               const int32_t *strides, const int32_t *offsets,
                               int total_cells, const int32_t *ref_map_index,
                               const ShimTplDepStats *src_cell,
                               int64_t *mc_dep_dist, int64_t *mc_dep_rate) {
  if (n_frames <= 0 || n_frames > MAX_LENGTH_TPL_FRAME_STATS) return -1;
  if (frame_idx < 0 || frame_idx >= n_frames) return -1;
  if (total_cells <= 0) return -1;

  TplParams *tpl = (TplParams *)calloc(1, sizeof(*tpl));
  if (!tpl) return -1;
  TplDepStats *cells =
      (TplDepStats *)calloc((size_t)total_cells, sizeof(*cells));
  if (!cells) {
    free(tpl);
    return -1;
  }

  tpl->tpl_stats_block_mis_log2 = block_mis_log2;
  tpl->tpl_bsize_1d = 16;
  tpl->tpl_frame = &tpl->tpl_stats_buffer[0];
  for (int i = 0; i < n_frames; ++i) {
    TplDepFrame *f = &tpl->tpl_frame[i];
    f->is_valid = 1;
    f->mi_rows = mi_rows[i];
    f->mi_cols = mi_cols[i];
    f->stride = strides[i];
    f->tpl_stats_ptr = &cells[offsets[i]];
    for (int r = 0; r < REF_FRAMES; ++r) {
      f->ref_map_index[r] = ref_map_index[i * REF_FRAMES + r];
    }
  }
  /* Seed the in-out propagation state. */
  for (int i = 0; i < total_cells; ++i) {
    cells[i].mc_dep_dist = mc_dep_dist[i];
    cells[i].mc_dep_rate = mc_dep_rate[i];
  }
  /* Write the source cell exactly where tpl_model_update_b will read it:
   * frame_idx's grid, indexed with tpl_frame[0].stride. */
  const int src_index =
      shim_tplc_tpl_ptr_pos(mi_row, mi_col, strides[0], block_mis_log2);
  const int frame_cells = (frame_idx + 1 < n_frames)
                              ? offsets[frame_idx + 1] - offsets[frame_idx]
                              : total_cells - offsets[frame_idx];
  if (src_index < 0 || src_index >= frame_cells) {
    free(cells);
    free(tpl);
    return -2;
  }
  TplDepStats in;
  shim_tplc_from_flat(&in, src_cell);
  /* Keep the seeded mc_dep_* of the source cell: C reads them. */
  tpl->tpl_frame[frame_idx].tpl_stats_ptr[src_index] = in;

  tpl_model_update(tpl, mi_row, mi_col, frame_idx);

  for (int i = 0; i < total_cells; ++i) {
    mc_dep_dist[i] = cells[i].mc_dep_dist;
    mc_dep_rate[i] = cells[i].mc_dep_rate;
  }
  free(cells);
  free(tpl);
  return 0;
}

/* ======================================================================== *
 * 4. TU-agreement gate: this TU's copy of the exported functions vs the
 *    archive's. See the header comment's EVIDENCE TIER paragraph.
 * ======================================================================== */
double shim_tplc_exp_entropy(double q_step, double b) {
  return shim_tplc_exponential_entropy(q_step, b);
}

int shim_tplc_overlap_area(int row_a, int col_a, int row_b, int col_b, int w,
                           int h) {
  return shim_tplc_get_overlap_area(row_a, col_a, row_b, col_b, w, h);
}

int64_t shim_tplc_drate_cost(int64_t delta_rate, int64_t recrf_dist,
                             int64_t srcrf_dist, int pix_num) {
  return shim_tplc_delta_rate_cost(delta_rate, recrf_dist, srcrf_dist, pix_num);
}

int shim_tplc_ptr_pos(int mi_row, int mi_col, int stride, uint8_t rs) {
  return shim_tplc_tpl_ptr_pos(mi_row, mi_col, stride, rs);
}
