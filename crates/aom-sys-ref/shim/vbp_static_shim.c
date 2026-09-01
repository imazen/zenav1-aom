/* Oracle shims for the FILE-STATIC helpers of av1/encoder/var_based_part.c.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * `nm -g` on the object file reports exactly four exported symbols for the
 * whole file (av1_choose_var_based_partitioning,
 * av1_set_variance_partition_thresholds and the two
 * av1_get_force_skip_low_temp_var lookups). The INTER leaf fill, the min/max
 * spread, the content threshold scale, the MV distance and the zero-MV skip
 * gate are all static and have no address a differential can take. The
 * alternative is hand-derived vectors, which this repo labels tier 4 and ranks
 * last -- and worse, re-deriving the expected value in the TEST is comparing
 * the port against a second transcription of the same logic, which proves
 * nothing about either.
 *
 * EVIDENCE TIER 1c -- the real C source compiled verbatim. Same technique and
 * same justification as shim/rdopt_shim.c; read that file's header for the
 * full argument. The second-compilation gap is closed by measurement:
 * `shim_vbps_tu_force_skip_*` re-export var_based_part.c's own exported
 * lookups from THIS TU, and `vbp_static_shim_tu_matches_archive` in
 * tests/var_part_inter_diff.rs asserts they agree with the archive's.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`, plus the oracle-wide `-ffp-contract=off`). `-DNDEBUG` is
 * separately mandatory for ABI agreement (DIFFERENTIAL_PLAYBOOK §3a(a)) and
 * doubly so here: av1_get_force_skip_low_temp_var asserts on its own mi
 * alignment for two of the block sizes.
 *
 * RTCD. fill_variance_8x8avg reaches aom_avg_8x8_quad / aom_avg_8x8, and
 * compute_minmax_8x8 reaches aom_minmax_8x8 -- all DISPATCHED, i.e. real
 * RTCD_EXTERN pointers on x86 that are NULL until aom_dsp_rtcd() runs
 * (DIFFERENTIAL_PLAYBOOK §3a(b)). Every ref_* wrapper on the Rust side calls
 * ref_init() unconditionally, which is what makes that safe.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename var_based_part.c's four exported symbols. --- */
#define av1_choose_var_based_partitioning shim_vbps_choose_var_based_partitioning
#define av1_set_variance_partition_thresholds shim_vbps_set_variance_partition_thresholds
#define av1_get_force_skip_low_temp_var shim_vbps_force_skip_low_temp_var
#define av1_get_force_skip_low_temp_var_small_sb shim_vbps_force_skip_low_temp_var_small_sb

/* --- libaom's own variance-based partitioner, unmodified. --- */
#include "av1/encoder/var_based_part.c"

int shim_vbps_all_blks_inside(int x16_idx, int y16_idx, int pixels_wide,
                              int pixels_high) {
  return all_blks_inside(x16_idx, y16_idx, pixels_wide, pixels_high);
}

/* fill_variance_8x8avg writes four VPartVar records; the shim returns the two
 * fields fill_variance() actually sets (log2_count is always 0 here). */
void shim_vbps_fill_variance_8x8avg(const void *src, int src_stride,
                                    const void *dst, int dst_stride,
                                    int x16_idx, int y16_idx, int highbd,
                                    int pixels_wide, int pixels_high,
                                    uint32_t *sse_out, int32_t *sum_out) {
  VP16x16 *vst = (VP16x16 *)calloc(1, sizeof(*vst));
  if (!vst) return;
  const uint8_t *s = highbd ? CONVERT_TO_BYTEPTR(src) : (const uint8_t *)src;
  const uint8_t *d = highbd ? CONVERT_TO_BYTEPTR(dst) : (const uint8_t *)dst;
  fill_variance_8x8avg(s, src_stride, d, dst_stride, x16_idx, y16_idx, vst,
                       highbd ? YV12_FLAG_HIGHBITDEPTH : 0, pixels_wide,
                       pixels_high);
  for (int i = 0; i < 4; ++i) {
    sse_out[i] = vst->split[i].part_variances.none.sum_square_error;
    sum_out[i] = vst->split[i].part_variances.none.sum_error;
  }
  free(vst);
}

int shim_vbps_compute_minmax_8x8(const void *src, int src_stride,
                                 const void *dst, int dst_stride, int x16_idx,
                                 int y16_idx, int highbd, int pixels_wide,
                                 int pixels_high) {
  const uint8_t *s = highbd ? CONVERT_TO_BYTEPTR(src) : (const uint8_t *)src;
  const uint8_t *d = highbd ? CONVERT_TO_BYTEPTR(dst) : (const uint8_t *)dst;
  return compute_minmax_8x8(s, src_stride, d, dst_stride, x16_idx, y16_idx,
#if CONFIG_AV1_HIGHBITDEPTH
                            highbd ? YV12_FLAG_HIGHBITDEPTH : 0,
#endif
                            pixels_wide, pixels_high);
}

int64_t shim_vbps_scale_part_thresh_content(int64_t threshold_base, int speed,
                                            int non_reference_frame,
                                            int is_static) {
  return scale_part_thresh_content(threshold_base, speed, non_reference_frame,
                                   is_static);
}

int shim_vbps_mv_distance(int16_t r0, int16_t c0, int16_t r1, int16_t c1) {
  FULLPEL_MV a = { r0, c0 };
  FULLPEL_MV b = { r1, c1 };
  return mv_distance(&a, &b);
}

int shim_vbps_is_set_force_zeromv_skip_based_on_src_sad(int level,
                                                        int source_sad_nonrd) {
  return is_set_force_zeromv_skip_based_on_src_sad(
             level, (SOURCE_SAD)source_sad_nonrd)
             ? 1
             : 0;
}

/* The TU-vs-archive gate: these forward to THIS TU's copies. */
int shim_vbps_tu_force_skip_low_temp_var(const uint8_t *variance_low,
                                         int mi_row, int mi_col, int bsize) {
  return shim_vbps_force_skip_low_temp_var(variance_low, mi_row, mi_col,
                                           (BLOCK_SIZE)bsize);
}

int shim_vbps_tu_force_skip_low_temp_var_small_sb(const uint8_t *variance_low,
                                                  int mi_row, int mi_col,
                                                  int bsize) {
  return shim_vbps_force_skip_low_temp_var_small_sb(variance_low, mi_row,
                                                    mi_col, (BLOCK_SIZE)bsize);
}

/* ======================================================================== *
 * set_low_temp_var_flag (:829) and the two size-specific setters it
 * dispatches to (:691 for SB64, :744 for SB128).
 *
 * These are what WRITE the `variance_low` array that
 * av1_get_force_skip_low_temp_var reads back, so the pair only means
 * something once both halves are gated.
 *
 * THE VARIANCE TREE crosses as 105 int values in this fixed layout, chosen
 * because it is exactly the set the three functions read:
 *
 *   [0  .. 5)    the 128x128 node:  none, horz[0], horz[1], vert[0], vert[1]
 *   [5  .. 25)   four 64x64 nodes,  same five fields each
 *   [25 .. 41)   sixteen 32x32 nodes, `none` only
 *   [41 ..105)   sixty-four 16x16 nodes, `none` only
 *
 * On the SB64 path C passes `&vt->split[0]`, so only the first 64x64's slice
 * and its descendants are read; the rest of the array is still filled, which
 * is what the encoder's own tree looks like.
 *
 * THE MI GRID crosses as a flat array of BLOCK_SIZE values, one per mi cell,
 * with -1 meaning C's NULL pointer -- both setters check for NULL before
 * dereferencing and the port has to reproduce that. The array is indexed the
 * way C indexes mi_grid_base: `mi_stride * row + col`.
 */
static void shim_vbps_load_vt(VP128x128 *vt, VP64x64 *l1, const int *v) {
  memset(vt, 0, sizeof(*vt));
  memset(l1, 0, 4 * sizeof(*l1));
  vt->split = l1;
  vt->part_variances.none.variance = v[0];
  vt->part_variances.horz[0].variance = v[1];
  vt->part_variances.horz[1].variance = v[2];
  vt->part_variances.vert[0].variance = v[3];
  vt->part_variances.vert[1].variance = v[4];
  for (int a = 0; a < 4; ++a) {
    const int *b = v + 5 + a * 5;
    l1[a].part_variances.none.variance = b[0];
    l1[a].part_variances.horz[0].variance = b[1];
    l1[a].part_variances.horz[1].variance = b[2];
    l1[a].part_variances.vert[0].variance = b[3];
    l1[a].part_variances.vert[1].variance = b[4];
    for (int c = 0; c < 4; ++c) {
      l1[a].split[c].part_variances.none.variance = v[25 + a * 4 + c];
      for (int d = 0; d < 4; ++d) {
        l1[a].split[c].split[d].part_variances.none.variance =
            v[41 + (a * 16) + (c * 4) + d];
      }
    }
  }
}

int shim_vbps_set_low_temp_var_flag(int is_small_sb, int ref_frame_partition,
                                    int cur_bsize, const int *variances,
                                    const int *mi_bsize, int mi_grid_len,
                                    int mi_stride, int mi_rows, int mi_cols,
                                    int mi_row, int mi_col,
                                    const int64_t *thresholds,
                                    uint8_t *variance_low_out) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  PartitionSearchInfo *part_info =
      (PartitionSearchInfo *)calloc(1, sizeof(*part_info));
  VP64x64 *l1 = (VP64x64 *)calloc(4, sizeof(*l1));
  MB_MODE_INFO *pool =
      (MB_MODE_INFO *)calloc((size_t)(mi_grid_len > 0 ? mi_grid_len : 1),
                             sizeof(MB_MODE_INFO));
  MB_MODE_INFO **grid =
      (MB_MODE_INFO **)calloc((size_t)(mi_grid_len > 0 ? mi_grid_len : 1),
                              sizeof(MB_MODE_INFO *));
  MB_MODE_INFO *cur = (MB_MODE_INFO *)calloc(1, sizeof(*cur));
  MB_MODE_INFO **cur_slot = (MB_MODE_INFO **)calloc(1, sizeof(*cur_slot));
  VP128x128 vt;
  if (!cpi || !xd || !part_info || !l1 || !pool || !grid || !cur || !cur_slot) {
    free(cpi); free(xd); free(part_info); free(l1);
    free(pool); free(grid); free(cur); free(cur_slot);
    return -1;
  }

  for (int i = 0; i < mi_grid_len; ++i) {
    if (mi_bsize[i] < 0) {
      grid[i] = NULL;
    } else {
      pool[i].bsize = (BLOCK_SIZE)mi_bsize[i];
      grid[i] = &pool[i];
    }
  }

  cpi->common.mi_params.mi_stride = mi_stride;
  cpi->common.mi_params.mi_rows = mi_rows;
  cpi->common.mi_params.mi_cols = mi_cols;
  cpi->common.mi_params.mi_grid_base = grid;

  cur->bsize = (BLOCK_SIZE)cur_bsize;
  cur_slot[0] = cur;
  xd->mi = cur_slot;

  shim_vbps_load_vt(&vt, l1, variances);

  int64_t thr[5];
  memcpy(thr, thresholds, sizeof(thr));

  set_low_temp_var_flag(cpi, part_info, xd, &vt, thr,
                        (MV_REFERENCE_FRAME)ref_frame_partition, mi_col, mi_row,
                        is_small_sb != 0);

  memcpy(variance_low_out, part_info->variance_low,
         sizeof(part_info->variance_low));

  free(cpi); free(xd); free(part_info); free(l1);
  free(pool); free(grid); free(cur); free(cur_slot);
  return 0;
}

int shim_vbps_variance_low_len(void) {
  PartitionSearchInfo p;
  return (int)sizeof(p.variance_low);
}

/* ======================================================================== *
 * set_force_zeromv_skip_for_sb (:1563) and set_ref_frame_for_partition
 * (:1219) -- their DECISION cores.
 *
 * Both functions mix a decision with buffer plumbing the port replaces rather
 * than translates (`set_block_size` writes the mi grid, `aom_free(vt)` frees
 * the variance tree, `av1_setup_pre_planes` re-points the predictor planes).
 * The shim drives the REAL C functions and reports only the decision outputs;
 * the plumbing runs but is not compared, because there is nothing in the port
 * for it to be compared against.
 *
 * For set_force_zeromv_skip_for_sb that means a live mi grid has to exist for
 * set_block_size to write into, and `vt` has to be a real aom_malloc'd
 * VP128x128 because C frees it on the exit path.
 * ======================================================================== */
int shim_vbps_set_force_zeromv_skip_for_sb(
    int set_zeromv_skip_based_on_source_sad, int source_sad_nonrd,
    int increase_source_sad_thresh, int part_early_exit_zeromv,
    int sb_size, int bsize, unsigned int thresh_exit_part_y_cfg,
    int mi_row, int mi_col, int tile_mi_row_end, int tile_mi_col_end,
    unsigned int y_sad, unsigned int uv_sad0, unsigned int uv_sad1,
    int mi_stride, int mi_rows, int mi_cols, int *force_zeromv_skip_out) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  const int grid_len = mi_stride * (mi_rows + 64);
  MB_MODE_INFO *pool = (MB_MODE_INFO *)calloc((size_t)grid_len, sizeof(*pool));
  MB_MODE_INFO **grid =
      (MB_MODE_INFO **)calloc((size_t)grid_len, sizeof(*grid));
  VP128x128 *vt = (VP128x128 *)aom_calloc(1, sizeof(*vt));
  TileInfo tile;
  unsigned int uv_sad[2];
  if (!cpi || !x || !seq || !pool || !grid || !vt) {
    free(cpi); free(x); free(seq); free(pool); free(grid); aom_free(vt);
    return -1;
  }
  for (int i = 0; i < grid_len; ++i) grid[i] = &pool[i];

  memset(&tile, 0, sizeof(tile));
  tile.mi_row_end = tile_mi_row_end;
  tile.mi_col_end = tile_mi_col_end;

  seq->sb_size = (BLOCK_SIZE)sb_size;
  cpi->common.seq_params = seq;
  cpi->common.mi_params.mi_stride = mi_stride;
  cpi->common.mi_params.mi_rows = mi_rows;
  cpi->common.mi_params.mi_cols = mi_cols;
  cpi->common.mi_params.mi_grid_base = grid;
  /* set_block_size (:136) writes through mi_alloc via get_alloc_mi_idx, so the
   * allocation view has to exist too: BLOCK_4X4 units make mi_alloc_stride the
   * same as mi_stride and the two indices coincide, which keeps the shim's
   * bookkeeping to one array. */
  cpi->common.mi_params.mi_alloc = pool;
  cpi->common.mi_params.mi_alloc_bsize = BLOCK_4X4;
  cpi->common.mi_params.mi_alloc_stride = mi_stride;
  cpi->common.mi_params.mi_alloc_size = grid_len;
  cpi->sf.rt_sf.set_zeromv_skip_based_on_source_sad =
      set_zeromv_skip_based_on_source_sad;
  cpi->sf.rt_sf.increase_source_sad_thresh = increase_source_sad_thresh;
  cpi->sf.rt_sf.part_early_exit_zeromv = part_early_exit_zeromv;
  cpi->zeromv_skip_thresh_exit_part[bsize] = thresh_exit_part_y_cfg;
  x->content_state_sb.source_sad_nonrd = (SOURCE_SAD)source_sad_nonrd;
  x->force_zeromv_skip_for_sb = 0;

  uv_sad[0] = uv_sad0;
  uv_sad[1] = uv_sad1;

  const int r = set_force_zeromv_skip_for_sb(cpi, x, &tile, vt, uv_sad, mi_row,
                                             mi_col, y_sad, (BLOCK_SIZE)bsize)
                    ? 1
                    : 0;
  *force_zeromv_skip_out = x->force_zeromv_skip_for_sb;

  /* C frees vt only on the exit path; free it here otherwise. */
  if (!r) aom_free(vt);
  free(cpi); free(x); free(seq); free(pool); free(grid);
  return r;
}

/* set_ref_frame_for_partition's decision half. av1_setup_pre_planes needs a
 * real reference buffer and scale factors, which the shim supplies as an
 * all-zero YV12 -- the call is made for fidelity but only the decision
 * outputs are reported. */
int shim_vbps_set_ref_frame_for_partition(
    int spatial_layer_id, int has_lower_quality_layer,
    unsigned int y_sad_in, unsigned int y_sad_g, unsigned int y_sad_alt,
    int nonrd_prune_ref_frame_search_cfg, unsigned int *y_sad_out,
    int *ref_frame_partition_out, int *prune_out, int *sb_me_out) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mi = (MB_MODE_INFO *)calloc(1, sizeof(*mi));
  YV12_BUFFER_CONFIG *g = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(*g));
  YV12_BUFFER_CONFIG *a = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(*a));
  uint8_t *plane = (uint8_t *)aom_calloc(1, 256 * 256);
  if (!cpi || !x || !mi || !g || !a || !plane) {
    free(cpi); free(x); free(mi); free(g); free(a); aom_free(plane);
    return -1;
  }
  for (int p = 0; p < MAX_MB_PLANE; ++p) {
    g->buffers[p] = plane;
    a->buffers[p] = plane;
  }
  g->strides[0] = g->strides[1] = 256;
  a->strides[0] = a->strides[1] = 256;
  g->crop_widths[0] = g->crop_widths[1] = 128;
  g->crop_heights[0] = g->crop_heights[1] = 128;
  a->crop_widths[0] = a->crop_widths[1] = 128;
  a->crop_heights[0] = a->crop_heights[1] = 128;

  /* av1_setup_pre_planes reads xd->mi[0]->bsize, so the mi slot has to be
   * wired up -- without it the plumbing half faults before the decision half
   * runs. */
  MB_MODE_INFO **slot = (MB_MODE_INFO **)calloc(1, sizeof(*slot));
  if (!slot) {
    free(cpi); free(x); free(mi); free(g); free(a); aom_free(plane);
    return -1;
  }
  mi->bsize = BLOCK_64X64;
  slot[0] = mi;
  x->e_mbd.mi = slot;
  /* Unscaled references: REF_NO_SCALE with unit steps, which is what
   * av1_setup_scale_factors_for_frame produces for a same-size reference. */
  for (int i = 0; i < REF_FRAMES; ++i) {
    cpi->common.ref_scale_factors[i].x_scale_fp = REF_NO_SCALE;
    cpi->common.ref_scale_factors[i].y_scale_fp = REF_NO_SCALE;
    cpi->common.ref_scale_factors[i].x_step_q4 = 16;
    cpi->common.ref_scale_factors[i].y_step_q4 = 16;
  }

  cpi->svc.spatial_layer_id = spatial_layer_id;
  cpi->svc.has_lower_quality_layer = has_lower_quality_layer;
  cpi->sf.rt_sf.nonrd_prune_ref_frame_search =
      nonrd_prune_ref_frame_search_cfg;
  x->nonrd_prune_ref_frame_search = -1;
  x->sb_me_partition = -1;

  MV_REFERENCE_FRAME ref_frame_partition = NONE_FRAME;
  unsigned int y_sad = y_sad_in, ysg = y_sad_g, ysa = y_sad_alt;
  set_ref_frame_for_partition(cpi, x, &x->e_mbd, &ref_frame_partition, mi,
                              &y_sad, &ysg, &ysa, g, a, 0, 0, 1);

  *y_sad_out = y_sad;
  *ref_frame_partition_out = (int)ref_frame_partition;
  *prune_out = x->nonrd_prune_ref_frame_search;
  *sb_me_out = x->sb_me_partition;

  free(cpi); free(x); free(mi); free(g); free(a); free(slot); aom_free(plane);
  return 0;
}
