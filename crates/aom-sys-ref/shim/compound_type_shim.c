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

/* ======================================================================== *
 * 10. The compound-RD reuse cache (compound_type.c:32-101, :955-1057).
 *
 * `COMP_RD_STATS` crosses the boundary as flat arrays rather than as a struct,
 * so the Rust side never has to reproduce a C layout that `-DNDEBUG` or a
 * libaom bump could move under it. Layout, per entry:
 *   rate[4] model_rate[4] rs2[4]            -> `const int32_t *i32`  (12)
 *   dist[4] model_dist[4]                   -> `const int64_t *i64`  (8)
 *   mv[2] as {row,col}                      -> `const int16_t *mv`   (4)
 *   ref_frames[2], mode, filter, ref_mv_idx,
 *   is_global[2], wedge_index, wedge_sign,
 *   mask_type, comp type                    -> `const int32_t *meta` (11)
 * ======================================================================== */

static void shim_ct_fill_stats(COMP_RD_STATS *st, const int32_t *i32,
                               const int64_t *i64v, const int16_t *mv,
                               const int32_t *meta) {
  for (int i = 0; i < COMPOUND_TYPES; ++i) {
    st->rate[i] = i32[i];
    st->model_rate[i] = i32[COMPOUND_TYPES + i];
    st->comp_rs2[i] = i32[2 * COMPOUND_TYPES + i];
    st->dist[i] = i64v[i];
    st->model_dist[i] = i64v[COMPOUND_TYPES + i];
  }
  for (int i = 0; i < 2; ++i) {
    st->mv[i].as_mv.row = mv[2 * i];
    st->mv[i].as_mv.col = mv[2 * i + 1];
    st->ref_frames[i] = (MV_REFERENCE_FRAME)meta[i];
    st->is_global[i] = meta[5 + i];
  }
  st->mode = (PREDICTION_MODE)meta[2];
  st->filter.as_int = (uint32_t)meta[3];
  st->ref_mv_idx = meta[4];
  st->interinter_comp.wedge_index = (int8_t)meta[7];
  st->interinter_comp.wedge_sign = (int8_t)meta[8];
  st->interinter_comp.mask_type = (DIFFWTD_MASK_TYPE)meta[9];
  st->interinter_comp.type = (COMPOUND_TYPE)meta[10];
}

/* `mi_meta`: ref_frames[2], mode, filter, bsize, wmtype[2]. */
static void shim_ct_fill_mi(MB_MODE_INFO *mi, const int16_t *mi_mv,
                            const int32_t *mi_meta, WarpedMotionParams *gm) {
  for (int i = 0; i < 2; ++i) {
    mi->mv[i].as_mv.row = mi_mv[2 * i];
    mi->mv[i].as_mv.col = mi_mv[2 * i + 1];
    mi->ref_frame[i] = (MV_REFERENCE_FRAME)mi_meta[i];
  }
  mi->mode = (PREDICTION_MODE)mi_meta[2];
  mi->interp_filters.as_int = (uint32_t)mi_meta[3];
  mi->bsize = (BLOCK_SIZE)mi_meta[4];
  /* `xd->global_motion` is indexed by mi->ref_frame[i], so the two wmtypes go
   * into the rows those references name. A reference of NONE_FRAME (-1) has
   * no row; C would read global_motion[-1], which the encoder never does
   * because a compound block always has two real references. */
  for (int i = 0; i < 2; ++i) {
    const int rf = mi_meta[i];
    if (rf >= 0 && rf < REF_FRAMES) gm[rf].wmtype = (TransformationType)mi_meta[5 + i];
  }
}

int shim_ct_is_comp_rd_match(int disable_interinter_wedge_newmv_search,
                             int enable_fast_compound_mode_search,
                             const int32_t *st_i32, const int64_t *st_i64,
                             const int16_t *st_mv, const int32_t *st_meta,
                             const int16_t *mi_mv, const int32_t *mi_meta,
                             int32_t *io_i32 /* 12, in/out */,
                             int64_t *io_i64 /* 8, in/out */) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mi = (MB_MODE_INFO *)calloc(1, sizeof(*mi));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  COMP_RD_STATS *st = (COMP_RD_STATS *)calloc(1, sizeof(*st));
  /* `xd->global_motion` is a POINTER to the frame's warp models (blockd.h),
   * NULL in a calloc'd MACROBLOCKD; `is_comp_rd_match` dereferences it once
   * per reference. */
  WarpedMotionParams *gm =
      (WarpedMotionParams *)calloc(REF_FRAMES, sizeof(*gm));
  if (!x || !mi || !cpi || !st || !gm) {
    free(gm); free(st); free(cpi); free(mi); free(x);
    return 0;
  }
  x->e_mbd.global_motion = gm;
  cpi->sf.inter_sf.disable_interinter_wedge_newmv_search =
      disable_interinter_wedge_newmv_search;
  cpi->sf.inter_sf.enable_fast_compound_mode_search =
      enable_fast_compound_mode_search;
  shim_ct_fill_stats(st, st_i32, st_i64, st_mv, st_meta);
  shim_ct_fill_mi(mi, mi_mv, mi_meta, gm);

  int32_t comp_rate[COMPOUND_TYPES], comp_model_rate[COMPOUND_TYPES];
  int comp_rs2[COMPOUND_TYPES];
  int64_t comp_dist[COMPOUND_TYPES], comp_model_dist[COMPOUND_TYPES];
  for (int i = 0; i < COMPOUND_TYPES; ++i) {
    comp_rate[i] = io_i32[i];
    comp_model_rate[i] = io_i32[COMPOUND_TYPES + i];
    comp_rs2[i] = io_i32[2 * COMPOUND_TYPES + i];
    comp_dist[i] = io_i64[i];
    comp_model_dist[i] = io_i64[COMPOUND_TYPES + i];
  }
  const int r = is_comp_rd_match(cpi, x, st, mi, comp_rate, comp_dist,
                                 comp_model_rate, comp_model_dist, comp_rs2);
  for (int i = 0; i < COMPOUND_TYPES; ++i) {
    io_i32[i] = comp_rate[i];
    io_i32[COMPOUND_TYPES + i] = comp_model_rate[i];
    io_i32[2 * COMPOUND_TYPES + i] = comp_rs2[i];
    io_i64[i] = comp_dist[i];
    io_i64[COMPOUND_TYPES + i] = comp_model_dist[i];
  }
  free(gm); free(st); free(cpi); free(mi); free(x);
  return r;
}

/* `save_comp_rd_search_stat` writes into `x->comp_rd_stats[x->comp_rd_stats_idx]`.
 * The wrapper seeds the index, runs the append, and reports the entry it
 * produced plus the new index — so the cache-full behaviour (silently drop,
 * never evict) is observable. */
int shim_ct_save_comp_rd_search_stat(int start_idx, const int32_t *i32,
                                     const int64_t *i64v, const int16_t *mv,
                                     const int16_t *mi_mv,
                                     const int32_t *mi_meta,
                                     const int32_t *comp_meta /* 4 */,
                                     int ref_mv_idx, int32_t *out_meta /* 11 */,
                                     int32_t *out_i32, int64_t *out_i64,
                                     int16_t *out_mv) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mi = (MB_MODE_INFO *)calloc(1, sizeof(*mi));
  WarpedMotionParams *gm =
      (WarpedMotionParams *)calloc(REF_FRAMES, sizeof(*gm));
  if (!x || !mi || !gm) {
    free(gm); free(mi); free(x);
    return -1;
  }
  x->e_mbd.global_motion = gm;
  x->comp_rd_stats_idx = start_idx;
  shim_ct_fill_mi(mi, mi_mv, mi_meta, gm);
  mi->ref_mv_idx = (uint8_t)ref_mv_idx;
  mi->interinter_comp.wedge_index = (int8_t)comp_meta[0];
  mi->interinter_comp.wedge_sign = (int8_t)comp_meta[1];
  mi->interinter_comp.mask_type = (DIFFWTD_MASK_TYPE)comp_meta[2];
  mi->interinter_comp.type = (COMPOUND_TYPE)comp_meta[3];

  int32_t comp_rate[COMPOUND_TYPES], comp_model_rate[COMPOUND_TYPES];
  int comp_rs2[COMPOUND_TYPES];
  int64_t comp_dist[COMPOUND_TYPES], comp_model_dist[COMPOUND_TYPES];
  int_mv cur_mv[2];
  for (int i = 0; i < COMPOUND_TYPES; ++i) {
    comp_rate[i] = i32[i];
    comp_model_rate[i] = i32[COMPOUND_TYPES + i];
    comp_rs2[i] = i32[2 * COMPOUND_TYPES + i];
    comp_dist[i] = i64v[i];
    comp_model_dist[i] = i64v[COMPOUND_TYPES + i];
  }
  for (int i = 0; i < 2; ++i) {
    cur_mv[i].as_mv.row = mv[2 * i];
    cur_mv[i].as_mv.col = mv[2 * i + 1];
  }

  save_comp_rd_search_stat(x, mi, comp_rate, comp_dist, comp_model_rate,
                           comp_model_dist, cur_mv, comp_rs2);

  const int new_idx = x->comp_rd_stats_idx;
  if (new_idx > start_idx) {
    const COMP_RD_STATS *st = &x->comp_rd_stats[start_idx];
    for (int i = 0; i < COMPOUND_TYPES; ++i) {
      out_i32[i] = st->rate[i];
      out_i32[COMPOUND_TYPES + i] = st->model_rate[i];
      out_i32[2 * COMPOUND_TYPES + i] = st->comp_rs2[i];
      out_i64[i] = st->dist[i];
      out_i64[COMPOUND_TYPES + i] = st->model_dist[i];
    }
    for (int i = 0; i < 2; ++i) {
      out_mv[2 * i] = st->mv[i].as_mv.row;
      out_mv[2 * i + 1] = st->mv[i].as_mv.col;
      out_meta[i] = st->ref_frames[i];
      out_meta[5 + i] = st->is_global[i];
    }
    out_meta[2] = st->mode;
    out_meta[3] = (int32_t)st->filter.as_int;
    out_meta[4] = st->ref_mv_idx;
    out_meta[7] = st->interinter_comp.wedge_index;
    out_meta[8] = st->interinter_comp.wedge_sign;
    out_meta[9] = st->interinter_comp.mask_type;
    out_meta[10] = st->interinter_comp.type;
  }
  free(gm);
  free(mi);
  free(x);
  return new_idx;
}

void shim_ct_backup_stats(int cur_type, int32_t *io_i32, int64_t *io_i64,
                          int rate_sum, int64_t dist_sum, int rd_rate,
                          int64_t rd_dist, int rs2) {
  int32_t comp_rate[COMPOUND_TYPES], comp_model_rate[COMPOUND_TYPES];
  int comp_rs2[COMPOUND_TYPES];
  int64_t comp_dist[COMPOUND_TYPES], comp_model_dist[COMPOUND_TYPES];
  for (int i = 0; i < COMPOUND_TYPES; ++i) {
    comp_rate[i] = io_i32[i];
    comp_model_rate[i] = io_i32[COMPOUND_TYPES + i];
    comp_rs2[i] = io_i32[2 * COMPOUND_TYPES + i];
    comp_dist[i] = io_i64[i];
    comp_model_dist[i] = io_i64[COMPOUND_TYPES + i];
  }
  RD_STATS rd_stats;
  memset(&rd_stats, 0, sizeof(rd_stats));
  rd_stats.rate = rd_rate;
  rd_stats.dist = rd_dist;
  backup_stats((COMPOUND_TYPE)cur_type, comp_rate, comp_dist, comp_model_rate,
               comp_model_dist, rate_sum, dist_sum, &rd_stats, comp_rs2, rs2);
  for (int i = 0; i < COMPOUND_TYPES; ++i) {
    io_i32[i] = comp_rate[i];
    io_i32[COMPOUND_TYPES + i] = comp_model_rate[i];
    io_i32[2 * COMPOUND_TYPES + i] = comp_rs2[i];
    io_i64[i] = comp_dist[i];
    io_i64[COMPOUND_TYPES + i] = comp_model_dist[i];
  }
}

/* `update_best_info` + `update_mask_best_mv` (compound_type.c:1005, :1016). */
void shim_ct_update_best_info(const int32_t *comp_meta /* 4 */, int64_t *io_rd,
                              int64_t *io_model_rd, int32_t *io_comp_meta /* 4 */,
                              int32_t *io_cost, int64_t best_rd_cur,
                              int64_t comp_model_rd_cur, int rs2) {
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  BEST_COMP_TYPE_STATS bts;
  memset(&bts, 0, sizeof(bts));
  if (!mbmi) return;
  mbmi->interinter_comp.wedge_index = (int8_t)comp_meta[0];
  mbmi->interinter_comp.wedge_sign = (int8_t)comp_meta[1];
  mbmi->interinter_comp.mask_type = (DIFFWTD_MASK_TYPE)comp_meta[2];
  mbmi->interinter_comp.type = (COMPOUND_TYPE)comp_meta[3];
  bts.comp_best_model_rd = *io_model_rd;
  bts.best_compmode_interinter_cost = *io_cost;
  bts.best_compound_data.wedge_index = (int8_t)io_comp_meta[0];
  bts.best_compound_data.wedge_sign = (int8_t)io_comp_meta[1];
  bts.best_compound_data.mask_type = (DIFFWTD_MASK_TYPE)io_comp_meta[2];
  bts.best_compound_data.type = (COMPOUND_TYPE)io_comp_meta[3];

  update_best_info(mbmi, io_rd, &bts, best_rd_cur, comp_model_rd_cur, rs2);

  *io_model_rd = bts.comp_best_model_rd;
  *io_cost = bts.best_compmode_interinter_cost;
  io_comp_meta[0] = bts.best_compound_data.wedge_index;
  io_comp_meta[1] = bts.best_compound_data.wedge_sign;
  io_comp_meta[2] = bts.best_compound_data.mask_type;
  io_comp_meta[3] = bts.best_compound_data.type;
  free(mbmi);
}

void shim_ct_update_mask_best_mv(const int16_t *mbmi_mv, int16_t *best_mv,
                                 int *best_tmp_rate_mv, int tmp_rate_mv) {
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  int_mv bmv[2];
  if (!mbmi) return;
  for (int i = 0; i < 2; ++i) {
    mbmi->mv[i].as_mv.row = mbmi_mv[2 * i];
    mbmi->mv[i].as_mv.col = mbmi_mv[2 * i + 1];
    bmv[i].as_mv.row = best_mv[2 * i];
    bmv[i].as_mv.col = best_mv[2 * i + 1];
  }
  update_mask_best_mv(mbmi, bmv, best_tmp_rate_mv, tmp_rate_mv);
  for (int i = 0; i < 2; ++i) {
    best_mv[2 * i] = bmv[i].as_mv.row;
    best_mv[2 * i + 1] = bmv[i].as_mv.col;
  }
  free(mbmi);
}

/* `populate_reuse_comp_type_data` (compound_type.c:962). Returns the
 * function's own return value; `*io_rd`, the mbmi fields and the winner are
 * reported through the out-parameters so the "reuse produced nothing" arm is
 * distinguishable from a successful reuse of a zero-cost type. */
int shim_ct_populate_reuse_comp_type_data(
    int rdmult, const int32_t *st_meta, const int32_t *io_i32,
    const int64_t *io_i64, const int16_t *cur_mv, int rate_mv,
    int best_compmode_interinter_cost, int64_t *io_rd, int32_t *out_comp_meta,
    int16_t *out_mv, int32_t *out_flags /* comp_group_idx, compound_idx */) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(*mbmi));
  if (!x || !mbmi) {
    free(mbmi); free(x);
    return 0;
  }
  x->rdmult = rdmult;
  /* The matched entry lives at index 0 of the cache for this driver. */
  x->comp_rd_stats[0].interinter_comp.wedge_index = (int8_t)st_meta[7];
  x->comp_rd_stats[0].interinter_comp.wedge_sign = (int8_t)st_meta[8];
  x->comp_rd_stats[0].interinter_comp.mask_type = (DIFFWTD_MASK_TYPE)st_meta[9];
  x->comp_rd_stats[0].interinter_comp.type = (COMPOUND_TYPE)st_meta[10];

  BEST_COMP_TYPE_STATS bts;
  memset(&bts, 0, sizeof(bts));
  bts.best_compmode_interinter_cost = best_compmode_interinter_cost;

  int32_t comp_rate[COMPOUND_TYPES];
  int comp_rs2[COMPOUND_TYPES];
  int64_t comp_dist[COMPOUND_TYPES];
  for (int i = 0; i < COMPOUND_TYPES; ++i) {
    comp_rate[i] = io_i32[i];
    comp_rs2[i] = io_i32[2 * COMPOUND_TYPES + i];
    comp_dist[i] = io_i64[i];
  }
  int_mv mv[2];
  for (int i = 0; i < 2; ++i) {
    mv[i].as_mv.row = cur_mv[2 * i];
    mv[i].as_mv.col = cur_mv[2 * i + 1];
  }
  int rate_mv_io = rate_mv;
  const int r = populate_reuse_comp_type_data(x, mbmi, &bts, mv, comp_rate,
                                              comp_dist, comp_rs2, &rate_mv_io,
                                              io_rd, /*match_index=*/0);
  out_comp_meta[0] = mbmi->interinter_comp.wedge_index;
  out_comp_meta[1] = mbmi->interinter_comp.wedge_sign;
  out_comp_meta[2] = mbmi->interinter_comp.mask_type;
  out_comp_meta[3] = mbmi->interinter_comp.type;
  out_flags[0] = mbmi->comp_group_idx;
  out_flags[1] = mbmi->compound_idx;
  for (int i = 0; i < 2; ++i) {
    out_mv[2 * i] = mbmi->mv[i].as_mv.row;
    out_mv[2 * i + 1] = mbmi->mv[i].as_mv.col;
  }
  free(mbmi);
  free(x);
  return r;
}

int shim_ct_max_comp_rd_stats(void) { return MAX_COMP_RD_STATS; }

/* ======================================================================== *
 * 11. The transform-search gate (compound_type.c:1069 + rdopt_utils.h:347,
 *     :778 + model_rd.h:69).
 *
 * `prune_mode_by_skip_rd` reads the prediction through
 * `x->plane[0].src` / `xd->plane[0].dst`, so the wrapper hands it two real
 * planes; the visible extent comes from `xd->mb_to_{right,bottom}_edge`,
 * which the caller sets so the frame-edge clip is exercised rather than
 * assumed away.
 * ======================================================================== */

int shim_ct_get_txfm_rd_gate_level(int is_masked_compound_enabled,
                                   const int *levels /* [TX_SEARCH_CASES] */,
                                   int bsize, int tx_search_case,
                                   int eval_motion_mode) {
  int lv[TX_SEARCH_CASES];
  for (int i = 0; i < TX_SEARCH_CASES; ++i) lv[i] = levels[i];
  return get_txfm_rd_gate_level(is_masked_compound_enabled, lv,
                                (BLOCK_SIZE)bsize,
                                (TX_SEARCH_CASE)tx_search_case,
                                eval_motion_mode);
}

int shim_ct_check_txfm_eval(unsigned int source_variance, int qindex,
                            int bsize, int64_t best_skip_rd, int64_t skip_rd,
                            int level, int is_luma_only) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  if (!x) return 0;
  x->source_variance = source_variance;
  x->qindex = qindex;
  const int r = check_txfm_eval(x, (BLOCK_SIZE)bsize, best_skip_rd, skip_rd,
                                level, is_luma_only);
  free(x);
  return r;
}

int64_t shim_ct_compute_sse_plane(int hbd, int bd, int bsize, int ss_x,
                                  int ss_y, int mb_to_right_edge,
                                  int mb_to_bottom_edge, const void *src,
                                  int src_stride, const void *dst,
                                  int dst_stride, int rows) {
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  YV12_BUFFER_CONFIG *cb =
      (YV12_BUFFER_CONFIG *)calloc(1, sizeof(YV12_BUFFER_CONFIG));
  void *asrc = shim_ct_align(src, (size_t)src_stride * rows * px);
  void *adst = shim_ct_align(dst, (size_t)dst_stride * rows * px);
  if (!x || !cb || !asrc || !adst) {
    aom_free(adst); aom_free(asrc); free(cb); free(x);
    return 0;
  }
  cb->flags = hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  x->e_mbd.cur_buf = cb;
  x->e_mbd.bd = bd;
  x->e_mbd.mb_to_right_edge = mb_to_right_edge;
  x->e_mbd.mb_to_bottom_edge = mb_to_bottom_edge;
  x->e_mbd.plane[0].subsampling_x = ss_x;
  x->e_mbd.plane[0].subsampling_y = ss_y;
  x->plane[0].src.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)asrc) : (uint8_t *)asrc;
  x->plane[0].src.stride = src_stride;
  x->e_mbd.plane[0].dst.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)adst) : (uint8_t *)adst;
  x->e_mbd.plane[0].dst.stride = dst_stride;

  const int64_t r =
      compute_sse_plane(x, &x->e_mbd, AOM_PLANE_Y, (BLOCK_SIZE)bsize);

  aom_free(adst);
  aom_free(asrc);
  free(cb);
  free(x);
  return r;
}

int shim_ct_prune_mode_by_skip_rd(int hbd, int bd, int bsize,
                                  int is_masked_compound_enabled,
                                  const int *levels, unsigned int source_variance,
                                  int qindex, int rdmult, int mb_to_right_edge,
                                  int mb_to_bottom_edge, const void *src,
                                  int src_stride, const void *dst,
                                  int dst_stride, int rows, int64_t ref_skip_rd,
                                  int mode_rate) {
  const size_t px = hbd ? sizeof(uint16_t) : sizeof(uint8_t);
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  YV12_BUFFER_CONFIG *cb =
      (YV12_BUFFER_CONFIG *)calloc(1, sizeof(YV12_BUFFER_CONFIG));
  void *asrc = shim_ct_align(src, (size_t)src_stride * rows * px);
  void *adst = shim_ct_align(dst, (size_t)dst_stride * rows * px);
  if (!x || !cpi || !seq || !cb || !asrc || !adst) {
    aom_free(adst); aom_free(asrc); free(cb); free(seq); free(cpi); free(x);
    return 0;
  }
  seq->enable_masked_compound = is_masked_compound_enabled;
  cpi->common.seq_params = seq;
  for (int i = 0; i < TX_SEARCH_CASES; ++i)
    cpi->sf.inter_sf.txfm_rd_gate_level[i] = levels[i];

  cb->flags = hbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  x->e_mbd.cur_buf = cb;
  x->e_mbd.bd = bd;
  x->rdmult = rdmult;
  x->source_variance = source_variance;
  x->qindex = qindex;
  x->e_mbd.mb_to_right_edge = mb_to_right_edge;
  x->e_mbd.mb_to_bottom_edge = mb_to_bottom_edge;
  x->plane[0].src.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)asrc) : (uint8_t *)asrc;
  x->plane[0].src.stride = src_stride;
  x->e_mbd.plane[0].dst.buf =
      hbd ? (uint8_t *)CONVERT_TO_BYTEPTR((uint16_t *)adst) : (uint8_t *)adst;
  x->e_mbd.plane[0].dst.stride = dst_stride;

  const int r = prune_mode_by_skip_rd(cpi, x, &x->e_mbd, (BLOCK_SIZE)bsize,
                                      ref_skip_rd, mode_rate);
  aom_free(adst);
  aom_free(asrc);
  free(cb);
  free(seq);
  free(cpi);
  free(x);
  return r;
}

int shim_ct_tx_search_cases(void) { return TX_SEARCH_CASES; }
int shim_ct_max_tx_rd_gate_level(void) { return MAX_TX_RD_GATE_LEVEL; }

/* ======================================================================== *
 * 12. compute_best_wedge_interintra (compound_type.c:520).
 *
 * The function rebuilds the intra predictor per mode with the REAL exported
 * `av1_build_intra_predictors_for_interintra`, so this driver cannot hand the
 * predictors in — it has to stand up enough MACROBLOCKD for intra prediction
 * to be well defined, and then report the four predictors it built so the
 * Rust port can be driven from C's own.
 *
 * The context plane is a real 2-D buffer with a row above and a column to the
 * left of the block origin: `ctx->plane[0]` points INTO it, and
 * `build_intra_predictors` reads `ref - ref_stride` (above, up to
 * `n_top_px + n_topright_px`) and `ref - 1` (left). The block sits well inside
 * the buffer so both reaches stay in bounds, and `up_available` /
 * `left_available` are set so the neighbours are used rather than the
 * 127/129 defaults — those would make all four modes flat and the mode search
 * a formality.
 *
 * LOWBD ONLY. `pick_interintra_wedge`, which does the work per mode, is
 * already gated at bd 8/10/12 by section 9; what is new here is the mode loop
 * and its cost accumulation, which is bit-depth independent.
 * ======================================================================== */

int64_t shim_ct_compute_best_wedge_interintra(
    int bsize, int rdmult, int dequant_ac, const int *wedge_idx_cost,
    const int *interintra_mode_cost /* [INTERINTRA_MODES] */,
    const uint8_t *src, int src_stride, const uint8_t *inter_pred,
    const uint8_t *ctx_plane, int ctx_stride, int ctx_rows, int ctx_origin,
    int mi_row, int mi_col, int mb_to_right_edge, int mb_to_bottom_edge,
    int *out_mode, int *out_wedge_index,
    uint8_t *out_intrapred /* INTERINTRA_MODES * bw * bh */) {
  const int bw = block_size_wide[bsize], bh = block_size_high[bsize];
  const int n = bw * bh;
  shim_ct_env e;
  AV1_COMMON *cm_unused = NULL;
  (void)cm_unused;
  if (!shim_ct_env_init(&e, bsize, /*hbd=*/0, /*bd=*/8, rdmult, dequant_ac,
                        wedge_idx_cost))
    return 0;

  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  uint8_t *actx = (uint8_t *)shim_ct_align(ctx_plane, (size_t)ctx_stride * ctx_rows);
  uint8_t *asrc = (uint8_t *)shim_ct_align(src, (size_t)src_stride * bh);
  uint8_t *ainter = (uint8_t *)shim_ct_align(inter_pred, (size_t)n);
  uint8_t *aintra = (uint8_t *)shim_ct_align(NULL, (size_t)n);
  if (!seq || !actx || !asrc || !ainter || !aintra) {
    aom_free(aintra); aom_free(ainter); aom_free(asrc); aom_free(actx);
    free(seq);
    shim_ct_env_free(&e);
    return 0;
  }

  seq->sb_size = BLOCK_64X64;
  seq->enable_intra_edge_filter = 1;
  e.cpi->common.seq_params = seq;

  MACROBLOCKD *xd = &e.x->e_mbd;
  xd->mi_row = mi_row;
  xd->mi_col = mi_col;
  xd->up_available = 1;
  xd->left_available = 1;
  xd->chroma_up_available = 1;
  xd->chroma_left_available = 1;
  xd->mb_to_right_edge = mb_to_right_edge;
  xd->mb_to_bottom_edge = mb_to_bottom_edge;
  xd->mb_to_left_edge = -(mi_col * MI_SIZE * 8);
  xd->mb_to_top_edge = -(mi_row * MI_SIZE * 8);
  xd->plane[0].width = bw;
  xd->plane[0].height = bh;
  xd->plane[0].subsampling_x = 0;
  xd->plane[0].subsampling_y = 0;
  e.mbmi->partition = PARTITION_NONE;
  e.mbmi->angle_delta[PLANE_TYPE_Y] = 0;
  e.mbmi->angle_delta[PLANE_TYPE_UV] = 0;
  e.mbmi->filter_intra_mode_info.use_filter_intra = 0;
  e.mbmi->use_intrabc = 0;
  e.mbmi->ref_frame[0] = LAST_FRAME;
  e.mbmi->ref_frame[1] = INTRA_FRAME;

  e.x->plane[0].src.buf = asrc;
  e.x->plane[0].src.stride = src_stride;

  BUFFER_SET ctx_set;
  memset(&ctx_set, 0, sizeof(ctx_set));
  ctx_set.plane[0] = actx + ctx_origin;
  ctx_set.stride[0] = ctx_stride;

  /* Report the four predictors BEFORE the search runs, so the Rust side is
   * fed exactly what C will build (the search mutates only
   * `mbmi->interintra_{mode,wedge_index}`, both restored below). */
  for (int m = 0; m < INTERINTRA_MODES; ++m) {
    e.mbmi->interintra_mode = (INTERINTRA_MODE)m;
    av1_build_intra_predictors_for_interintra(&e.cpi->common, xd,
                                              (BLOCK_SIZE)bsize, 0, &ctx_set,
                                              aintra, bw);
    memcpy(out_intrapred + (size_t)m * n, aintra, (size_t)n);
  }
  e.mbmi->interintra_mode = II_DC_PRED;
  e.mbmi->interintra_wedge_index = 0;

  for (int i = 0; i < INTERINTRA_MODES; ++i)
    e.x->mode_costs.interintra_mode_cost[size_group_lookup[bsize]][i] =
        interintra_mode_cost[i];

  int best_mode = 0, best_wedge_index = 0;
  const int64_t rd = compute_best_wedge_interintra(
      e.cpi, e.mbmi, xd, e.x,
      e.x->mode_costs.interintra_mode_cost[size_group_lookup[bsize]], &ctx_set,
      aintra, (uint8_t *)ainter, &best_mode, &best_wedge_index,
      (BLOCK_SIZE)bsize);
  *out_mode = best_mode;
  *out_wedge_index = best_wedge_index;

  aom_free(aintra);
  aom_free(ainter);
  aom_free(asrc);
  aom_free(actx);
  free(seq);
  shim_ct_env_free(&e);
  return rd;
}

int shim_ct_interintra_modes(void) { return INTERINTRA_MODES; }

/* ======================================================================== *
 * 13. compute_best_interintra_mode (compound_type.c:459).
 *
 * Same MACROBLOCKD standing-up as section 12, plus a destination plane: the
 * function COMBINES into `xd->plane[0].dst` and then measures src against it,
 * so the buffer's contents on return are part of what is under test.
 *
 * The four intra predictors C builds from this neighbour context are reported
 * (as in section 12) so the Rust port can be driven from them rather than
 * from a second intra prediction.
 * ======================================================================== */

int64_t shim_ct_compute_best_interintra_mode(
    int bsize, int rdmult, int dequant_ac,
    const int *interintra_mode_cost /* [INTERINTRA_MODES] */, int mode,
    int use_wedge_interintra, int wedge_index, const uint8_t *src,
    int src_stride, const uint8_t *inter_pred, const uint8_t *ctx_plane,
    int ctx_stride, int ctx_rows, int ctx_origin, int mi_row, int mi_col,
    int mb_to_right_edge, int mb_to_bottom_edge, int64_t best_rd_in,
    int best_mode_in, int *out_best_mode, uint8_t *out_dst, int dst_stride,
    int dst_rows, uint8_t *out_intrapred /* INTERINTRA_MODES * bw * bh */) {
  const int bw = block_size_wide[bsize], bh = block_size_high[bsize];
  const int n = bw * bh;
  shim_ct_env e;
  if (!shim_ct_env_init(&e, bsize, /*hbd=*/0, /*bd=*/8, rdmult, dequant_ac,
                        NULL))
    return 0;

  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  uint8_t *actx = (uint8_t *)shim_ct_align(ctx_plane, (size_t)ctx_stride * ctx_rows);
  uint8_t *asrc = (uint8_t *)shim_ct_align(src, (size_t)src_stride * bh);
  uint8_t *ainter = (uint8_t *)shim_ct_align(inter_pred, (size_t)n);
  uint8_t *aintra = (uint8_t *)shim_ct_align(NULL, (size_t)n);
  uint8_t *adst = (uint8_t *)shim_ct_align(out_dst, (size_t)dst_stride * dst_rows);
  if (!seq || !actx || !asrc || !ainter || !aintra || !adst) {
    aom_free(adst); aom_free(aintra); aom_free(ainter); aom_free(asrc);
    aom_free(actx); free(seq);
    shim_ct_env_free(&e);
    return 0;
  }

  seq->sb_size = BLOCK_64X64;
  seq->enable_intra_edge_filter = 1;
  e.cpi->common.seq_params = seq;

  MACROBLOCKD *xd = &e.x->e_mbd;
  xd->mi_row = mi_row;
  xd->mi_col = mi_col;
  xd->up_available = 1;
  xd->left_available = 1;
  xd->chroma_up_available = 1;
  xd->chroma_left_available = 1;
  xd->is_chroma_ref = 1;
  xd->mb_to_right_edge = mb_to_right_edge;
  xd->mb_to_bottom_edge = mb_to_bottom_edge;
  xd->mb_to_left_edge = -(mi_col * MI_SIZE * 8);
  xd->mb_to_top_edge = -(mi_row * MI_SIZE * 8);
  xd->plane[0].width = bw;
  xd->plane[0].height = bh;
  xd->plane[0].subsampling_x = 0;
  xd->plane[0].subsampling_y = 0;
  xd->plane[0].dst.buf = adst;
  xd->plane[0].dst.stride = dst_stride;
  e.mbmi->partition = PARTITION_NONE;
  e.mbmi->angle_delta[PLANE_TYPE_Y] = 0;
  e.mbmi->angle_delta[PLANE_TYPE_UV] = 0;
  e.mbmi->filter_intra_mode_info.use_filter_intra = 0;
  e.mbmi->use_intrabc = 0;
  e.mbmi->ref_frame[0] = LAST_FRAME;
  e.mbmi->ref_frame[1] = INTRA_FRAME;
  e.mbmi->use_wedge_interintra = use_wedge_interintra;
  e.mbmi->interintra_wedge_index = (int8_t)wedge_index;
  e.x->plane[0].src.buf = asrc;
  e.x->plane[0].src.stride = src_stride;

  BUFFER_SET ctx_set;
  memset(&ctx_set, 0, sizeof(ctx_set));
  ctx_set.plane[0] = actx + ctx_origin;
  ctx_set.stride[0] = ctx_stride;

  for (int m = 0; m < INTERINTRA_MODES; ++m) {
    e.mbmi->interintra_mode = (INTERINTRA_MODE)m;
    av1_build_intra_predictors_for_interintra(&e.cpi->common, xd,
                                              (BLOCK_SIZE)bsize, 0, &ctx_set,
                                              aintra, bw);
    memcpy(out_intrapred + (size_t)m * n, aintra, (size_t)n);
  }

  for (int i = 0; i < INTERINTRA_MODES; ++i)
    e.x->mode_costs.interintra_mode_cost[size_group_lookup[bsize]][i] =
        interintra_mode_cost[i];

  INTERINTRA_MODE best_mode = (INTERINTRA_MODE)best_mode_in;
  int64_t best_rd = best_rd_in;
  compute_best_interintra_mode(
      e.cpi, e.mbmi, xd, e.x,
      e.x->mode_costs.interintra_mode_cost[size_group_lookup[bsize]], &ctx_set,
      aintra, ainter, &best_mode, &best_rd, (INTERINTRA_MODE)mode,
      (BLOCK_SIZE)bsize);

  *out_best_mode = (int)best_mode;
  memcpy(out_dst, adst, (size_t)dst_stride * dst_rows);

  aom_free(adst);
  aom_free(aintra);
  aom_free(ainter);
  aom_free(asrc);
  aom_free(actx);
  free(seq);
  shim_ct_env_free(&e);
  return best_rd;
}
