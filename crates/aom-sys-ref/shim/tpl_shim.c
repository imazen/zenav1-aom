/* Oracle shims for av1/encoder/tpl_model.c — the temporal dependency model.
 *
 * WHAT IS BEING DRIVEN
 * --------------------
 * `av1_tpl_stats_ready`, `av1_tpl_get_qstep_ratio` and `av1_tpl_get_q_index`
 * are all EXPORTED (`nm -g upstream/build/libaom.a` reports `T` for each), so
 * every wrapper here calls the archive's own symbol — **tier 1**. They are
 * shimmed rather than bound directly only because their first argument is a
 * `TplParams`, which is 100+ KB of frame bookkeeping the Rust side has no
 * business reproducing byte-for-byte.
 *
 * The point of the middle one is `get_frame_importance` (tpl_model.c:1942),
 * which is file-static: `av1_tpl_get_qstep_ratio` is its ONLY caller and is
 * exported, so driving the caller gates the static at tier 1 too. That is the
 * whole reason this shim exists in the shape it does.
 *
 * WHAT THE CALLER SUPPLIES
 * ------------------------
 * `get_frame_importance` reads exactly five things off the TPL state:
 * `tpl_frame->{mi_rows, mi_cols, stride, base_rdmult, tpl_stats_ptr}` and
 * `tpl_data->tpl_stats_block_mis_log2`. Per grid cell it reads four int64
 * fields: `srcrf_dist`, `recrf_dist`, `mc_dep_rate`, `mc_dep_dist`. Those four
 * arrive as four parallel arrays indexed EXACTLY as `av1_tpl_ptr_pos` indexes
 * the real `tpl_stats_ptr` buffer, so the shim's fill loop is a copy, not a
 * re-derivation of the addressing under test.
 *
 * `TplParams` is heap-allocated and zeroed: it embeds
 * `TplDepFrame tpl_stats_buffer[105]` plus `YV12_BUFFER_CONFIG
 * tpl_rec_pool[48]`, which is far too large to put on a test process's stack.
 * (Same reasoning as rdopt_shim.c's `MACROBLOCK`.)
 *
 * `tpl_data->tpl_frame` points at `tpl_stats_buffer[0]` here, not at
 * `[REF_FRAMES + 1]` as the encoder sets it. That offset is pure allocation
 * bookkeeping — every read in the functions under test is relative to
 * `tpl_frame`, and `av1_tpl_stats_ready` bounds `gf_frame_index` by
 * `MAX_TPL_FRAME_IDX` (96), which is inside the 105-entry buffer either way.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"

#include "av1/common/av1_common_int.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/tpl_model.h"

/* Build a TplParams carrying one populated TPL frame at `gf_frame_index`.
 * Returns NULL on allocation failure; the caller frees with shim_tpl_free.
 */
static TplParams *shim_tpl_build(int ready, int gf_frame_index, int is_valid,
                                 int mi_rows, int mi_cols, int stride,
                                 int base_rdmult, uint8_t block_mis_log2,
                                 const int64_t *srcrf_dist,
                                 const int64_t *recrf_dist,
                                 const int64_t *mc_dep_rate,
                                 const int64_t *mc_dep_dist, int n_stats) {
  TplParams *tpl = (TplParams *)calloc(1, sizeof(*tpl));
  if (!tpl) return NULL;
  TplDepStats *stats = NULL;
  if (n_stats > 0) {
    stats = (TplDepStats *)calloc((size_t)n_stats, sizeof(*stats));
    if (!stats) {
      free(tpl);
      return NULL;
    }
    for (int i = 0; i < n_stats; ++i) {
      stats[i].srcrf_dist = srcrf_dist[i];
      stats[i].recrf_dist = recrf_dist[i];
      stats[i].mc_dep_rate = mc_dep_rate[i];
      stats[i].mc_dep_dist = mc_dep_dist[i];
    }
  }
  tpl->ready = ready;
  tpl->tpl_stats_block_mis_log2 = block_mis_log2;
  tpl->tpl_bsize_1d = 16;
  tpl->tpl_frame = &tpl->tpl_stats_buffer[0];
  if (gf_frame_index >= 0 && gf_frame_index < MAX_LENGTH_TPL_FRAME_STATS) {
    TplDepFrame *f = &tpl->tpl_frame[gf_frame_index];
    f->is_valid = (uint8_t)is_valid;
    f->mi_rows = mi_rows;
    f->mi_cols = mi_cols;
    f->stride = stride;
    f->base_rdmult = base_rdmult;
    f->tpl_stats_ptr = stats;
  } else if (stats) {
    free(stats);
    stats = NULL;
  }
  return tpl;
}

static void shim_tpl_free(TplParams *tpl, int gf_frame_index) {
  if (!tpl) return;
  if (gf_frame_index >= 0 && gf_frame_index < MAX_LENGTH_TPL_FRAME_STATS) {
    free(tpl->tpl_frame[gf_frame_index].tpl_stats_ptr);
  }
  free(tpl);
}

/* ---- av1_tpl_stats_ready (tpl_model.c:1856) ------------------------------
 * Three gates: the global `ready` flag, the MAX_TPL_FRAME_IDX bound on the
 * sub-GOP index, and the per-frame `is_valid`. No stats buffer is read, so
 * none is built.
 */
int shim_tpl_stats_ready(int ready, int gf_frame_index, int is_valid) {
  TplParams *tpl = shim_tpl_build(ready, gf_frame_index, is_valid, 0, 0, 0, 0,
                                  2, NULL, NULL, NULL, NULL, 0);
  if (!tpl) return -1;
  int r = av1_tpl_stats_ready(tpl, gf_frame_index);
  shim_tpl_free(tpl, gf_frame_index);
  return r;
}

/* ---- av1_tpl_get_qstep_ratio (tpl_model.c:2418) --------------------------
 * Drives the exported entry point, and through it the file-static
 * `get_frame_importance`. `out` receives the ratio; the return value is 0 on
 * success and -1 if the shim could not allocate (so an allocation failure can
 * never be mistaken for a ratio of 0).
 */
int shim_tpl_get_qstep_ratio(int ready, int gf_frame_index, int is_valid,
                             int mi_rows, int mi_cols, int stride,
                             int base_rdmult, uint8_t block_mis_log2,
                             const int64_t *srcrf_dist,
                             const int64_t *recrf_dist,
                             const int64_t *mc_dep_rate,
                             const int64_t *mc_dep_dist, int n_stats,
                             double *out) {
  TplParams *tpl = shim_tpl_build(ready, gf_frame_index, is_valid, mi_rows,
                                  mi_cols, stride, base_rdmult, block_mis_log2,
                                  srcrf_dist, recrf_dist, mc_dep_rate,
                                  mc_dep_dist, n_stats);
  if (!tpl) return -1;
  *out = av1_tpl_get_qstep_ratio(tpl, gf_frame_index);
  shim_tpl_free(tpl, gf_frame_index);
  return 0;
}

/* ---- av1_tpl_get_q_index (tpl_model.c:2446) ------------------------------
 * The composition the CRF path actually calls: qstep ratio -> qindex.
 */
int shim_tpl_get_q_index(int ready, int gf_frame_index, int is_valid,
                         int mi_rows, int mi_cols, int stride, int base_rdmult,
                         uint8_t block_mis_log2, const int64_t *srcrf_dist,
                         const int64_t *recrf_dist, const int64_t *mc_dep_rate,
                         const int64_t *mc_dep_dist, int n_stats,
                         int leaf_qindex, int bit_depth, int *out) {
  TplParams *tpl = shim_tpl_build(ready, gf_frame_index, is_valid, mi_rows,
                                  mi_cols, stride, base_rdmult, block_mis_log2,
                                  srcrf_dist, recrf_dist, mc_dep_rate,
                                  mc_dep_dist, n_stats);
  if (!tpl) return -1;
  *out = av1_tpl_get_q_index(tpl, gf_frame_index, leaf_qindex,
                             (aom_bit_depth_t)bit_depth);
  shim_tpl_free(tpl, gf_frame_index);
  return 0;
}

/* ======================================================================== *
 * 4. The MV-entropy pair and the rdmult setup — all three EXPORTED.
 * ======================================================================== */

/* Build a standalone TplDepFrame whose grid carries only what
 * `av1_compute_mv_difference` / `av1_tpl_compute_frame_mv_entropy` read:
 * `mv[]` and `ref_frame_index[0]`. `mvs` is {row, col} pairs, laid out
 * n_stats * INTER_REFS_PER_FRAME entries. */
static TplDepFrame *shim_tpl_mv_frame(int is_valid, int mi_rows, int mi_cols,
                                      int stride, int n_stats,
                                      const int16_t *mvs,
                                      const int8_t *ref_frame_index0) {
  TplDepFrame *f = (TplDepFrame *)calloc(1, sizeof(*f));
  if (!f) return NULL;
  TplDepStats *stats = NULL;
  if (n_stats > 0) {
    stats = (TplDepStats *)calloc((size_t)n_stats, sizeof(*stats));
    if (!stats) {
      free(f);
      return NULL;
    }
    for (int i = 0; i < n_stats; ++i) {
      for (int r = 0; r < INTER_REFS_PER_FRAME; ++r) {
        const int base = 2 * (i * INTER_REFS_PER_FRAME + r);
        stats[i].mv[r].as_mv.row = mvs[base];
        stats[i].mv[r].as_mv.col = mvs[base + 1];
      }
      stats[i].ref_frame_index[0] = ref_frame_index0[i];
      stats[i].ref_frame_index[1] = -1;
    }
  }
  f->is_valid = (uint8_t)is_valid;
  f->mi_rows = mi_rows;
  f->mi_cols = mi_cols;
  f->stride = stride;
  f->tpl_stats_ptr = stats;
  return f;
}

static void shim_tpl_mv_frame_free(TplDepFrame *f) {
  if (!f) return;
  free(f->tpl_stats_ptr);
  free(f);
}

/* ---- av1_compute_mv_difference (tpl_model.c:2639) ---------------------- */
int shim_tpl_compute_mv_difference(int mi_rows, int mi_cols, int stride,
                                   int n_stats, const int16_t *mvs,
                                   const int8_t *ref_frame_index0, int row,
                                   int col, int step, int tpl_stride,
                                   int right_shift, int16_t *out_mv) {
  TplDepFrame *f = shim_tpl_mv_frame(1, mi_rows, mi_cols, stride, n_stats, mvs,
                                     ref_frame_index0);
  if (!f) return -1;
  int_mv r = av1_compute_mv_difference(f, row, col, step, tpl_stride,
                                       right_shift);
  out_mv[0] = r.as_mv.row;
  out_mv[1] = r.as_mv.col;
  shim_tpl_mv_frame_free(f);
  return 0;
}

/* ---- av1_tpl_compute_frame_mv_entropy (tpl_model.c:2681) --------------- */
int shim_tpl_compute_frame_mv_entropy(int is_valid, int mi_rows, int mi_cols,
                                      int stride, int n_stats,
                                      const int16_t *mvs,
                                      const int8_t *ref_frame_index0,
                                      uint8_t right_shift, double *out) {
  TplDepFrame *f = shim_tpl_mv_frame(is_valid, mi_rows, mi_cols, stride,
                                     n_stats, mvs, ref_frame_index0);
  if (!f) return -1;
  *out = av1_tpl_compute_frame_mv_entropy(f, right_shift);
  shim_tpl_mv_frame_free(f);
  return 0;
}

/* ---- av1_tpl_rdmult_setup (tpl_model.c:2213) --------------------------- *
 * Fills `cpi->tpl_rdmult_scaling_factors` from the TPL grid. The function
 * reads the superres-upscaled width (not the coded width) for its column
 * count, so that is a separate parameter here rather than derived.
 * `out_factors` must hold num_rows * num_cols doubles; `out_num_cols` and
 * `out_num_rows` report the grid the function actually walked. */
int shim_tpl_rdmult_setup(int gf_frame_index, int gf_group_size, int is_valid,
                          int superres_upscaled_width, int mi_rows,
                          int stride, int base_rdmult, uint8_t block_mis_log2,
                          double r0, const int64_t *recrf_dist,
                          const int64_t *mc_dep_rate,
                          const int64_t *mc_dep_dist, int n_stats,
                          double *out_factors, int out_capacity,
                          int *out_num_rows, int *out_num_cols) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  AV1_PRIMARY *ppi = (AV1_PRIMARY *)calloc(1, sizeof(*ppi));
  TplDepStats *stats = NULL;
  if (!cpi || !ppi) goto fail;
  if (n_stats > 0) {
    stats = (TplDepStats *)calloc((size_t)n_stats, sizeof(*stats));
    if (!stats) goto fail;
    for (int i = 0; i < n_stats; ++i) {
      stats[i].recrf_dist = recrf_dist[i];
      stats[i].mc_dep_rate = mc_dep_rate[i];
      stats[i].mc_dep_dist = mc_dep_dist[i];
    }
  }

  cpi->ppi = ppi;
  cpi->gf_frame_index = gf_frame_index;
  ppi->gf_group.size = gf_group_size;
  cpi->common.superres_upscaled_width = superres_upscaled_width;
  cpi->common.mi_params.mi_rows = mi_rows;
  cpi->rd.r0 = r0;

  TplParams *tpl = &ppi->tpl_data;
  tpl->tpl_stats_block_mis_log2 = block_mis_log2;
  tpl->tpl_bsize_1d = 16;
  tpl->tpl_frame = &tpl->tpl_stats_buffer[0];
  if (gf_frame_index < 0 || gf_frame_index >= MAX_LENGTH_TPL_FRAME_STATS)
    goto fail;
  TplDepFrame *f = &tpl->tpl_frame[gf_frame_index];
  f->is_valid = (uint8_t)is_valid;
  f->mi_rows = mi_rows;
  f->mi_cols = av1_pixels_to_mi(superres_upscaled_width);
  f->stride = stride;
  f->base_rdmult = base_rdmult;
  f->tpl_stats_ptr = stats;

  const int mi_cols_sr = av1_pixels_to_mi(superres_upscaled_width);
  const int num_mi_w = mi_size_wide[BLOCK_16X16];
  const int num_mi_h = mi_size_high[BLOCK_16X16];
  const int num_cols = (mi_cols_sr + num_mi_w - 1) / num_mi_w;
  const int num_rows = (mi_rows + num_mi_h - 1) / num_mi_h;
  if (num_cols <= 0 || num_rows <= 0 || num_rows * num_cols > out_capacity)
    goto fail;
  *out_num_rows = num_rows;
  *out_num_cols = num_cols;

  cpi->tpl_rdmult_scaling_factors = out_factors;
  for (int i = 0; i < num_rows * num_cols; ++i) out_factors[i] = 0.0;

  av1_tpl_rdmult_setup(cpi);

  free(stats);
  free(ppi);
  free(cpi);
  return 0;

fail:
  free(stats);
  free(ppi);
  free(cpi);
  return -1;
}
