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

#include "aom_mem/aom_mem.h"

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

/* ======================================================================== *
 * 9. The mask picks (compound_type.c:126-428).
 *
 * ---- the alignment / sizing contract ----------------------------------
 * These reach RTCD-DISPATCHED kernels: `aom_subtract_block`,
 * `av1_wedge_sse_from_residuals`, `av1_wedge_sign_from_residuals`,
 * `av1_wedge_compute_delta_squares`, `av1_build_compound_diffwtd_mask{,_highbd}`
 * and (via `estimate_wedge_sign`) `ppi->fn_ptr[].vf`. On this aarch64 oracle
 * most are `#define`d straight to NEON, which loads unaligned, so a misaligned
 * buffer is invisible; on x86 they are AVX2/SSE kernels and libaom's own
 * callers always hand them `DECLARE_ALIGNED(32, ...)` storage
 * (compound_type.c:210 `residual0`, :396-397 `residual1`/`diff10`). A Rust
 * `Vec` is 1- or 2-byte aligned. So every buffer that crosses into a
 * dispatched kernel is bounced through 64-byte-aligned scratch (64 covers
 * AVX-512, strictly stronger than the encoder's 32 and never weaker), exactly
 * as shim/comp_pred_shim.c does.
 *
 * `xd->seg_mask` is a POINTER (blockd.h:889), NULL in a calloc'd MACROBLOCKD;
 * `pick_interinter_seg` writes through it on its first iteration. It is
 * allocated here at the encoder's own size, `2 * MAX_SB_SQUARE`.
 * ======================================================================== */

static void *shim_ct_align(const void *src, size_t bytes) {
  void *p = aom_memalign(64, bytes ? bytes : 64);
  if (!p) return NULL;
  if (src)
    memcpy(p, src, bytes);
  else
    memset(p, 0, bytes);
  return p;
}

/* `cpi->ppi->fn_ptr[bsize].vf` for the ONE quarter-block size
 * `estimate_wedge_sign` looks up. libaom installs the whole table in
 * `av1_create_primary_compressor` (encoder.c:1171 `BFP`) / `highbd_set_var_fns`
 * (encoder.c:865); constructing a real AV1_PRIMARY here would drag in the
 * entire encoder, so just the one entry under test is installed, from the same
 * `aom_variance` / `aom_highbd_<bd>_variance` families those macros use.
 *
 * These names are RTCD symbols, so `ref_init()` (which runs `aom_dsp_rtcd()`)
 * must have happened before the pointer is taken — every Rust wrapper below
 * calls it. */
static void shim_ct_install_vf(AV1_PRIMARY *ppi, BLOCK_SIZE b, int hbd,
                               int bd) {
#define V(BT, W, H)                                                          \
  case BT:                                                                   \
    ppi->fn_ptr[BT].vf = hbd ? (bd == 12   ? aom_highbd_12_variance##W##x##H \
                                : bd == 10 ? aom_highbd_10_variance##W##x##H \
                                           : aom_highbd_8_variance##W##x##H) \
                             : aom_variance##W##x##H;                        \
    break;
  switch (b) {
    V(BLOCK_4X4, 4, 4)
    V(BLOCK_4X8, 4, 8)
    V(BLOCK_8X4, 8, 4)
    V(BLOCK_8X8, 8, 8)
    V(BLOCK_8X16, 8, 16)
    V(BLOCK_16X8, 16, 8)
    V(BLOCK_16X16, 16, 16)
    V(BLOCK_16X32, 16, 32)
    V(BLOCK_32X16, 32, 16)
    V(BLOCK_32X32, 32, 32)
    V(BLOCK_32X64, 32, 64)
    V(BLOCK_64X32, 64, 32)
    V(BLOCK_64X64, 64, 64)
    V(BLOCK_4X16, 4, 16)
    V(BLOCK_16X4, 16, 4)
    V(BLOCK_8X32, 8, 32)
    V(BLOCK_32X8, 32, 8)
    default: break;
  }
#undef V
}

/* The per-call state every pick needs, built and torn down together so no
 * wrapper can forget one of the six pieces. */
typedef struct {
  MACROBLOCK *x;
  AV1_COMP *cpi;
  AV1_PRIMARY *ppi;
  YV12_BUFFER_CONFIG *cb;
  MB_MODE_INFO *mbmi;
  MB_MODE_INFO *mi_ptr;
  uint8_t *seg_mask;
  /* `plane[].dequant_QTX` is a `const int16_t *` (block.h:168), so the AC
   * dequant the model RD divides by needs real backing storage. */
  int16_t *dq;
} shim_ct_env;

static int shim_ct_env_init(shim_ct_env *e, int bsize, int hbd, int bd,
                            int rdmult, int dequant_ac,
                            const int *wedge_idx_cost) {
  memset(e, 0, sizeof(*e));
  e->x = (MACROBLOCK *)calloc(1, sizeof(*e->x));
  e->cpi = (AV1_COMP *)calloc(1, sizeof(*e->cpi));
  e->ppi = (AV1_PRIMARY *)calloc(1, sizeof(*e->ppi));
  e->cb = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(*e->cb));
  e->mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*e->mbmi));
  e->seg_mask = (uint8_t *)shim_ct_align(NULL, 2 * MAX_SB_SQUARE);
  e->dq = (int16_t *)calloc(2, sizeof(int16_t));
  if (!e->x || !e->cpi || !e->ppi || !e->cb || !e->mbmi || !e->seg_mask ||
      !e->dq)
    return 0;

  e->cb->flags = hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  e->x->e_mbd.cur_buf = e->cb;
  e->x->e_mbd.bd = bd;
  e->x->e_mbd.seg_mask = e->seg_mask;
  e->mi_ptr = e->mbmi;
  e->x->e_mbd.mi = &e->mi_ptr;
  e->mbmi->bsize = (BLOCK_SIZE)bsize;
  e->x->rdmult = rdmult;
  e->dq[0] = (int16_t)dequant_ac;
  e->dq[1] = (int16_t)dequant_ac;
  e->x->plane[0].dequant_QTX = e->dq;
  if (wedge_idx_cost)
    for (int i = 0; i < MAX_WEDGE_TYPES; ++i)
      e->x->mode_costs.wedge_idx_cost[bsize][i] = wedge_idx_cost[i];
  e->cpi->ppi = e->ppi;
  return 1;
}

static void shim_ct_env_free(shim_ct_env *e) {
  free(e->dq);
  aom_free(e->seg_mask);
  free(e->mbmi);
  free(e->cb);
  free(e->ppi);
  free(e->cpi);
  free(e->x);
}

int64_t shim_ct_pick_wedge(int bsize, int hbd, int bd, int rdmult,
                           int dequant_ac, const int *wedge_idx_cost,
                           const void *src, int src_stride, const void *p0,
                           const int16_t *residual1, const int16_t *diff10,
                           int *out_sign, int *out_index, uint64_t *out_sse) {
  const int bw = block_size_wide[bsize], bh = block_size_high[bsize];
  const int n = bw * bh;
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  shim_ct_env e;
  if (!shim_ct_env_init(&e, bsize, hbd, bd, rdmult, dequant_ac,
                        wedge_idx_cost))
    return 0;

  void *asrc = shim_ct_align(src, (size_t)src_stride * bh * px);
  void *ap0 = shim_ct_align(p0, (size_t)n * px);
  int16_t *ar1 = (int16_t *)shim_ct_align(residual1, (size_t)n * 2);
  int16_t *ad10 = (int16_t *)shim_ct_align(diff10, (size_t)n * 2);

  e.x->plane[0].src.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)asrc) : (uint8_t *)asrc;
  e.x->plane[0].src.stride = src_stride;
  /* The PREDICTORS go in RAW, at every bit depth: compound_type.c applies
   * `CONVERT_TO_BYTEPTR` to them itself on the hbd arms (:214 pick_wedge,
   * :351-354 pick_interinter_seg, :400-405 pick_interintra_wedge, :160-163
   * estimate_wedge_sign). Only `x->plane[0].src.buf` arrives pre-converted,
   * because that is how the encoder stores it. Converting here as well
   * double-shifts the pointer and segfaults — measured at bd=10, the first
   * cell of pick_interinter_seg_matches_c. */
  const uint8_t *p0_arg = (const uint8_t *)ap0;

  int8_t sign = 0, index = -1;
  uint64_t sse = UINT64_MAX;
  const int64_t rd = pick_wedge(e.cpi, e.x, (BLOCK_SIZE)bsize, p0_arg, ar1,
                                ad10, &sign, &index, &sse);
  *out_sign = sign;
  *out_index = index;
  *out_sse = sse;

  aom_free(ad10);
  aom_free(ar1);
  aom_free(ap0);
  aom_free(asrc);
  shim_ct_env_free(&e);
  return rd;
}

int64_t shim_ct_pick_wedge_fixed_sign(int bsize, int hbd, int bd, int rdmult,
                                      int dequant_ac,
                                      const int *wedge_idx_cost,
                                      const int16_t *residual1,
                                      const int16_t *diff10, int wedge_sign,
                                      int *out_index, uint64_t *out_sse) {
  const int n = block_size_wide[bsize] * block_size_high[bsize];
  shim_ct_env e;
  if (!shim_ct_env_init(&e, bsize, hbd, bd, rdmult, dequant_ac,
                        wedge_idx_cost))
    return 0;

  int16_t *ar1 = (int16_t *)shim_ct_align(residual1, (size_t)n * 2);
  int16_t *ad10 = (int16_t *)shim_ct_align(diff10, (size_t)n * 2);

  int8_t index = -1;
  uint64_t sse = UINT64_MAX;
  const int64_t rd =
      pick_wedge_fixed_sign(e.cpi, e.x, (BLOCK_SIZE)bsize, ar1, ad10,
                            (int8_t)wedge_sign, &index, &sse);
  *out_index = index;
  *out_sse = sse;

  aom_free(ad10);
  aom_free(ar1);
  shim_ct_env_free(&e);
  return rd;
}

int shim_ct_estimate_wedge_sign(int bsize, int hbd, int bd, const void *src,
                                int src_stride, const void *pred0, int stride0,
                                const void *pred1, int stride1) {
  const int bh = block_size_high[bsize];
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  shim_ct_env e;
  if (!shim_ct_env_init(&e, bsize, hbd, bd, 0, 0, NULL)) return 0;

  /* `split_qtr[bsize]` is compound_type.c's own table; the wrapper reproduces
   * only the lookup, so the ONE vf entry the function will read is installed.
   * A bsize with no quarter split trips C's assert, which -DNDEBUG removes —
   * the Rust wrapper refuses those instead. */
  static const BLOCK_SIZE split_qtr[BLOCK_SIZES_ALL] = {
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X4,
    BLOCK_4X8,     BLOCK_8X4,     BLOCK_8X8,     BLOCK_8X16,
    BLOCK_16X8,    BLOCK_16X16,   BLOCK_16X32,   BLOCK_32X16,
    BLOCK_32X32,   BLOCK_32X64,   BLOCK_64X32,   BLOCK_64X64,
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X16,    BLOCK_16X4,
    BLOCK_8X32,    BLOCK_32X8
  };
  shim_ct_install_vf(e.ppi, split_qtr[bsize], hbd, bd);

  void *asrc = shim_ct_align(src, (size_t)src_stride * bh * px);
  void *ap0 = shim_ct_align(pred0, (size_t)stride0 * bh * px);
  void *ap1 = shim_ct_align(pred1, (size_t)stride1 * bh * px);

  e.x->plane[0].src.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)asrc) : (uint8_t *)asrc;
  e.x->plane[0].src.stride = src_stride;
  /* estimate_wedge_sign does its own CONVERT_TO_BYTEPTR on pred0/pred1 when
   * the block is hbd, so these go in as raw pointers either way. */
  const int r =
      estimate_wedge_sign(e.cpi, e.x, (BLOCK_SIZE)bsize, (const uint8_t *)ap0,
                          stride0, (const uint8_t *)ap1, stride1);

  aom_free(ap1);
  aom_free(ap0);
  aom_free(asrc);
  shim_ct_env_free(&e);
  return r;
}

int64_t shim_ct_pick_interinter_wedge(int bsize, int hbd, int bd, int rdmult,
                                      int dequant_ac,
                                      const int *wedge_idx_cost,
                                      int fast_wedge_sign_estimate,
                                      const void *src, int src_stride,
                                      const void *p0, const void *p1,
                                      const int16_t *residual1,
                                      const int16_t *diff10, int *out_sign,
                                      int *out_index, uint64_t *out_sse) {
  const int bw = block_size_wide[bsize], bh = block_size_high[bsize];
  const int n = bw * bh;
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  shim_ct_env e;
  if (!shim_ct_env_init(&e, bsize, hbd, bd, rdmult, dequant_ac,
                        wedge_idx_cost))
    return 0;

  static const BLOCK_SIZE split_qtr[BLOCK_SIZES_ALL] = {
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X4,
    BLOCK_4X8,     BLOCK_8X4,     BLOCK_8X8,     BLOCK_8X16,
    BLOCK_16X8,    BLOCK_16X16,   BLOCK_16X32,   BLOCK_32X16,
    BLOCK_32X32,   BLOCK_32X64,   BLOCK_64X32,   BLOCK_64X64,
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X16,    BLOCK_16X4,
    BLOCK_8X32,    BLOCK_32X8
  };
  if (split_qtr[bsize] != BLOCK_INVALID)
    shim_ct_install_vf(e.ppi, split_qtr[bsize], hbd, bd);

  /* `assert(cpi->common.seq_params->enable_masked_compound)` — compiled out
   * under -DNDEBUG, but seq_params is dereferenced nowhere else here. */
  e.cpi->sf.inter_sf.fast_wedge_sign_estimate = fast_wedge_sign_estimate;

  void *asrc = shim_ct_align(src, (size_t)src_stride * bh * px);
  void *ap0 = shim_ct_align(p0, (size_t)n * px);
  void *ap1 = shim_ct_align(p1, (size_t)n * px);
  int16_t *ar1 = (int16_t *)shim_ct_align(residual1, (size_t)n * 2);
  int16_t *ad10 = (int16_t *)shim_ct_align(diff10, (size_t)n * 2);

  e.x->plane[0].src.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)asrc) : (uint8_t *)asrc;
  e.x->plane[0].src.stride = src_stride;
  /* The PREDICTORS go in RAW, at every bit depth: compound_type.c applies
   * `CONVERT_TO_BYTEPTR` to them itself on the hbd arms (:214 pick_wedge,
   * :351-354 pick_interinter_seg, :400-405 pick_interintra_wedge, :160-163
   * estimate_wedge_sign). Only `x->plane[0].src.buf` arrives pre-converted,
   * because that is how the encoder stores it. Converting here as well
   * double-shifts the pointer and segfaults — measured at bd=10, the first
   * cell of pick_interinter_seg_matches_c. */
  const uint8_t *p0_arg = (const uint8_t *)ap0;
  const uint8_t *p1_arg = (const uint8_t *)ap1;

  uint64_t sse = UINT64_MAX;
  const int64_t rd =
      pick_interinter_wedge(e.cpi, e.x, (BLOCK_SIZE)bsize, p0_arg, p1_arg, ar1,
                            ad10, &sse);
  *out_sign = e.mbmi->interinter_comp.wedge_sign;
  *out_index = e.mbmi->interinter_comp.wedge_index;
  *out_sse = sse;

  aom_free(ad10);
  aom_free(ar1);
  aom_free(ap1);
  aom_free(ap0);
  aom_free(asrc);
  shim_ct_env_free(&e);
  return rd;
}

int64_t shim_ct_pick_interinter_seg(int bsize, int hbd, int bd, int rdmult,
                                    int dequant_ac, const void *p0,
                                    const void *p1, const int16_t *residual1,
                                    const int16_t *diff10, int *out_mask_type,
                                    uint64_t *out_sse,
                                    uint8_t *out_seg_mask /* n bytes */) {
  const int n = block_size_wide[bsize] * block_size_high[bsize];
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  shim_ct_env e;
  if (!shim_ct_env_init(&e, bsize, hbd, bd, rdmult, dequant_ac, NULL)) return 0;

  void *ap0 = shim_ct_align(p0, (size_t)n * px);
  void *ap1 = shim_ct_align(p1, (size_t)n * px);
  int16_t *ar1 = (int16_t *)shim_ct_align(residual1, (size_t)n * 2);
  int16_t *ad10 = (int16_t *)shim_ct_align(diff10, (size_t)n * 2);

  /* The PREDICTORS go in RAW, at every bit depth: compound_type.c applies
   * `CONVERT_TO_BYTEPTR` to them itself on the hbd arms (:214 pick_wedge,
   * :351-354 pick_interinter_seg, :400-405 pick_interintra_wedge, :160-163
   * estimate_wedge_sign). Only `x->plane[0].src.buf` arrives pre-converted,
   * because that is how the encoder stores it. Converting here as well
   * double-shifts the pointer and segfaults — measured at bd=10, the first
   * cell of pick_interinter_seg_matches_c. */
  const uint8_t *p0_arg = (const uint8_t *)ap0;
  const uint8_t *p1_arg = (const uint8_t *)ap1;

  uint64_t sse = UINT64_MAX;
  const int64_t rd = pick_interinter_seg(e.cpi, e.x, (BLOCK_SIZE)bsize, p0_arg,
                                         p1_arg, ar1, ad10, &sse);
  *out_mask_type = e.mbmi->interinter_comp.mask_type;
  *out_sse = sse;
  memcpy(out_seg_mask, e.x->e_mbd.seg_mask, (size_t)n);

  aom_free(ad10);
  aom_free(ar1);
  aom_free(ap1);
  aom_free(ap0);
  shim_ct_env_free(&e);
  return rd;
}

int64_t shim_ct_pick_interintra_wedge(int bsize, int hbd, int bd, int rdmult,
                                      int dequant_ac,
                                      const int *wedge_idx_cost,
                                      const void *src, int src_stride,
                                      const void *p0, const void *p1,
                                      int *out_index, uint64_t *out_sse) {
  const int bw = block_size_wide[bsize], bh = block_size_high[bsize];
  const int n = bw * bh;
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  shim_ct_env e;
  if (!shim_ct_env_init(&e, bsize, hbd, bd, rdmult, dequant_ac,
                        wedge_idx_cost))
    return 0;

  void *asrc = shim_ct_align(src, (size_t)src_stride * bh * px);
  void *ap0 = shim_ct_align(p0, (size_t)n * px);
  void *ap1 = shim_ct_align(p1, (size_t)n * px);

  e.x->plane[0].src.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)asrc) : (uint8_t *)asrc;
  e.x->plane[0].src.stride = src_stride;
  /* The PREDICTORS go in RAW, at every bit depth: compound_type.c applies
   * `CONVERT_TO_BYTEPTR` to them itself on the hbd arms (:214 pick_wedge,
   * :351-354 pick_interinter_seg, :400-405 pick_interintra_wedge, :160-163
   * estimate_wedge_sign). Only `x->plane[0].src.buf` arrives pre-converted,
   * because that is how the encoder stores it. Converting here as well
   * double-shifts the pointer and segfaults — measured at bd=10, the first
   * cell of pick_interinter_seg_matches_c. */
  const uint8_t *p0_arg = (const uint8_t *)ap0;
  const uint8_t *p1_arg = (const uint8_t *)ap1;

  const int64_t rd = pick_interintra_wedge(e.cpi, e.x, (BLOCK_SIZE)bsize,
                                           p0_arg, p1_arg);
  *out_index = e.mbmi->interintra_wedge_index;
  /* `pick_interintra_wedge` keeps its own `sse` local and does not return it;
   * the port's shape does, so the value is not compared. Reported as 0. */
  *out_sse = 0;

  aom_free(ap1);
  aom_free(ap0);
  aom_free(asrc);
  shim_ct_env_free(&e);
  return rd;
}
