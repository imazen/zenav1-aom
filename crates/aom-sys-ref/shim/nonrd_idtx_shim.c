/* Oracle shim for av1_block_yrd_idtx (av1/encoder/nonrd_opt.c:380) and the
 * fast-IDTX scan tables it runs on.
 *
 * EVIDENCE TIER 1. av1_block_yrd_idtx is EXPORTED; this file calls the real
 * symbol out of libaom.a and only assembles the MACROBLOCK it reads:
 *
 *   x->plane[0].src.{buf,stride}     the source block
 *   x->plane[0].src_diff             the residual scratch aom_subtract_block fills
 *   x->plane[0].{round,quant}_fp_QTX the low-precision quantizer rows
 *   x->plane[0].dequant_QTX
 *   x->e_mbd.mb_to_{right,bottom}_edge  the two edge clamps
 *   x->txfm_search_info.blk_skip     written per sub-block, read back out
 *
 * The scan tables are `static const` in nonrd_opt.h, so they have no exported
 * address either; this TU includes that header and copies them out, which is
 * how the port's copies are checked against the ones the encoder really uses
 * rather than against a transcription of the same header.
 *
 * CONTRACTS (DIFFERENTIAL_PLAYBOOK §3a):
 *  - `-DNDEBUG` is mandatory: MACROBLOCK has an #ifndef NDEBUG member, so
 *    without it every field below sits at a different offset than the
 *    archive's. build.rs passes it to every shim.
 *  - av1_block_yrd_idtx calls aom_subtract_block and av1_quantize_lp, both
 *    DISPATCHED. Every ref_* wrapper calls ref_init() unconditionally.
 *  - src / pred / src_diff are bounced through 64-byte aom_memalign scratch
 *    because the dispatched aom_subtract_block writes src_diff with vector
 *    stores; a 1-byte-aligned Rust Vec faults there on x86.
 *  - src_diff is sized for the whole 128x128 block, not w*h, matching
 *    av1_alloc_src_diff_buf's own allocation.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"
#include "aom_mem/aom_mem.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/encoder/block.h"
/* nonrd_opt.h needs the full AV1_COMP definition (rdopt_utils.h) plus the
 * declarations nonrd_opt.c itself pulls in ahead of it. */
#include "av1/encoder/encoder.h"
#include "av1/common/reconinter.h"
#include "av1/common/mvref_common.h"
#include "av1/encoder/encodemv.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/nonrd_opt.h"

/* --- the fast-IDTX scan tables, copied out of nonrd_opt.h's own definitions. */
void shim_nrd_fast_idtx_scan(int tx_size, int inverse, int16_t *out) {
  const int16_t *src = NULL;
  int n = 0;
  switch (tx_size) {
    case 0:
      src = inverse ? av1_fast_idtx_iscan_4x4 : av1_fast_idtx_scan_4x4;
      n = 16;
      break;
    case 1:
      src = inverse ? av1_fast_idtx_iscan_8x8 : av1_fast_idtx_scan_8x8;
      n = 64;
      break;
    default:
      src = inverse ? av1_fast_idtx_iscan_16x16 : av1_fast_idtx_scan_16x16;
      n = 256;
      break;
  }
  memcpy(out, src, n * sizeof(*out));
}

/* --- av1_block_yrd_idtx --------------------------------------------------- */
int shim_nrd_block_yrd_idtx(const uint8_t *src, int src_stride,
                            const uint8_t *pred, int pred_stride, int bsize,
                            int tx_size, int mb_to_right_edge,
                            int mb_to_bottom_edge, const int16_t *round_fp,
                            const int16_t *quant_fp, const int16_t *dequant,
                            int64_t sse_in, int32_t *rate_out,
                            int64_t *dist_out, int64_t *sse_out,
                            uint8_t *blk_skip_out) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(*x));
  if (!x) return -1;
  /* MAX_SB_SIZE^2 residual, the same shape av1_alloc_src_diff_buf gives it. */
  int16_t *src_diff = (int16_t *)aom_memalign(64, 128 * 128 * sizeof(int16_t));
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  uint8_t *src_a = (uint8_t *)aom_memalign(64, (size_t)src_stride * bh + 64);
  uint8_t *pred_a = (uint8_t *)aom_memalign(64, (size_t)pred_stride * bh + 64);
  if (!src_diff || !src_a || !pred_a) {
    aom_free(src_diff);
    aom_free(src_a);
    aom_free(pred_a);
    free(x);
    return -1;
  }
  memcpy(src_a, src, (size_t)src_stride * bh);
  memcpy(pred_a, pred, (size_t)pred_stride * bh);
  memset(src_diff, 0, 128 * 128 * sizeof(int16_t));

  x->plane[AOM_PLANE_Y].src.buf = src_a;
  x->plane[AOM_PLANE_Y].src.stride = src_stride;
  x->plane[AOM_PLANE_Y].src_diff = src_diff;
  x->plane[AOM_PLANE_Y].round_fp_QTX = round_fp;
  x->plane[AOM_PLANE_Y].quant_fp_QTX = quant_fp;
  x->plane[AOM_PLANE_Y].dequant_QTX = dequant;
  x->e_mbd.mb_to_right_edge = mb_to_right_edge;
  x->e_mbd.mb_to_bottom_edge = mb_to_bottom_edge;

  RD_STATS rdc;
  memset(&rdc, 0, sizeof(rdc));
  rdc.sse = sse_in;
  int skippable = 0;
  av1_block_yrd_idtx(x, pred_a, pred_stride, &rdc, &skippable,
                     (BLOCK_SIZE)bsize, (TX_SIZE)tx_size);

  *rate_out = rdc.rate;
  *dist_out = rdc.dist;
  *sse_out = rdc.sse;
  memcpy(blk_skip_out, x->txfm_search_info.blk_skip,
         sizeof(x->txfm_search_info.blk_skip));

  (void)bw;
  aom_free(src_diff);
  aom_free(src_a);
  aom_free(pred_a);
  free(x);
  return skippable;
}
