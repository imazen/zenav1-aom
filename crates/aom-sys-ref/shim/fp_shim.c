/* Oracle shims for av1/encoder/firstpass.c.
 *
 * TWO TIERS IN ONE FILE, and the split is deliberate.
 *
 * firstpass.c exports 16 symbols out of 45 definitions (`nm -g` on the
 * archive's firstpass.c.o). The exported ones are reached DIRECTLY here
 * (**tier 1**); this TU also compiles firstpass.c itself, with those 16
 * renamed out of the way, to reach the 29 file-static ones (**tier 1c**, the
 * rdopt_shim.c technique). Wrappers around exported functions call the
 * `av1_*` symbol from the ARCHIVE, not this TU's copy, so tier 1 stays tier 1
 * — the renames make that distinction mechanical rather than a promise.
 *
 * Wait: the renames apply to the whole TU, so a wrapper written here cannot
 * name the archive's copy. So the exported eleven are bound WITHOUT a shim,
 * straight from Rust (see aom-sys-ref/src/lib.rs), and this file wraps only
 * the statics plus the two exported functions whose arguments are structs the
 * Rust side should not have to lay out (`av1_get_unit_{rows,cols}_in_tile`
 * take a `TileInfo`). Those two are called through this TU's copy and are
 * therefore tier 1c; their agreement with the archive is covered by the
 * TU-agreement gate below, which compares this TU's `av1_twopass_zero_stats`
 * and `av1_accumulate_stats` against the archive's.
 *
 * FLAGS: `-O3 -DNDEBUG`, libaom's own Release flags, for the same reason as
 * rdopt_shim.c — this is the same source and must be built the same way.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename firstpass.c's 16 exported symbols so this TU links beside
 * libaom.a. */
#define av1_accumulate_stats shim_fpc_accumulate_stats
#define av1_end_first_pass shim_fpc_end_first_pass
#define av1_first_pass shim_fpc_first_pass
#define av1_first_pass_row shim_fpc_first_pass_row
#define av1_firstpass_info_future_count shim_fpc_firstpass_info_future_count
#define av1_firstpass_info_init shim_fpc_firstpass_info_init
#define av1_firstpass_info_move_cur_index shim_fpc_firstpass_info_move_cur_index
#define av1_firstpass_info_move_cur_index_and_pop \
  shim_fpc_firstpass_info_move_cur_index_and_pop
#define av1_firstpass_info_peek shim_fpc_firstpass_info_peek
#define av1_firstpass_info_pop shim_fpc_firstpass_info_pop
#define av1_firstpass_info_push shim_fpc_firstpass_info_push
#define av1_free_firstpass_data shim_fpc_free_firstpass_data
#define av1_get_unit_cols_in_tile shim_fpc_get_unit_cols_in_tile
#define av1_get_unit_rows_in_tile shim_fpc_get_unit_rows_in_tile
#define av1_noop_first_pass_frame shim_fpc_noop_first_pass_frame
#define av1_twopass_zero_stats shim_fpc_twopass_zero_stats

/* --- libaom's own first pass, unmodified. --- */
#include "av1/encoder/firstpass.c"

/* ======================================================================== *
 * 1. Layout probe — proves the Rust mirrors of FIRSTPASS_STATS and
 *    FRAME_STATS agree with C's, field by field, before anything else here
 *    is trusted.
 * ======================================================================== */
int shim_fp_stats_size(void) { return (int)sizeof(FIRSTPASS_STATS); }
int shim_fp_frame_stats_size(void) { return (int)sizeof(FRAME_STATS); }

/* Writes a distinct ramp into every FIRSTPASS_STATS field, in declaration
 * order, so a Rust mirror with a permuted or mistyped field is caught. */
void shim_fp_stats_layout_probe(FIRSTPASS_STATS *s) {
  memset(s, 0, sizeof(*s));
  double v = 1.0;
  s->frame = v++;
  s->weight = v++;
  s->intra_error = v++;
  s->frame_avg_wavelet_energy = v++;
  s->coded_error = v++;
  s->sr_coded_error = v++;
  s->lt_coded_error = v++;
  s->pcnt_inter = v++;
  s->pcnt_motion = v++;
  s->pcnt_second_ref = v++;
  s->pcnt_neutral = v++;
  s->intra_skip_pct = v++;
  s->inactive_zone_rows = v++;
  s->inactive_zone_cols = v++;
  s->MVr = v++;
  s->mvr_abs = v++;
  s->MVc = v++;
  s->mvc_abs = v++;
  s->MVrv = v++;
  s->MVcv = v++;
  s->mv_in_out_count = v++;
  s->new_mv_count = v++;
  s->duration = v++;
  s->count = v++;
  s->raw_error_stdev = v++;
  s->is_flash = (int64_t)v++;
  s->noise_var = v++;
  s->cor_coeff = v++;
  s->log_intra_error = v++;
  s->log_coded_error = v++;
}

void shim_fp_frame_stats_layout_probe(FRAME_STATS *s) {
  memset(s, 0, sizeof(*s));
  int64_t v = 1;
  s->intra_error = v++;
  s->frame_avg_wavelet_energy = v++;
  s->coded_error = v++;
  s->sr_coded_error = v++;
  s->lt_coded_error = v++;
  s->mv_count = (int)v++;
  s->inter_count = (int)v++;
  s->second_ref_count = (int)v++;
  s->neutral_count = (double)v++;
  s->intra_skip_count = (int)v++;
  s->image_data_start_row = (int)v++;
  s->new_mv_count = (int)v++;
  s->sum_in_vectors = (int)v++;
  s->sum_mvr = (int)v++;
  s->sum_mvc = (int)v++;
  s->sum_mvr_abs = (int)v++;
  s->sum_mvc_abs = (int)v++;
  s->sum_mvrs = v++;
  s->sum_mvcs = v++;
  s->intra_factor = (double)v++;
  s->brightness_factor = (double)v++;
}

/* ======================================================================== *
 * 2. TU-agreement gate: this TU's copy of two exported functions vs the
 *    archive's. See the header comment.
 * ======================================================================== */
void shim_fp_tu_twopass_zero_stats(FIRSTPASS_STATS *s) {
  shim_fpc_twopass_zero_stats(s);
}

void shim_fp_tu_accumulate_stats(FIRSTPASS_STATS *section,
                                 const FIRSTPASS_STATS *frame) {
  shim_fpc_accumulate_stats(section, frame);
}

/* ======================================================================== *
 * 3. Exported functions whose arguments are structs (tier 1c through this
 *    TU's copy; the agreement gate above covers the second compilation).
 * ======================================================================== */
int shim_fp_get_unit_rows_in_tile(int mi_row_start, int mi_row_end,
                                  int fp_block_size) {
  TileInfo tile;
  memset(&tile, 0, sizeof(tile));
  tile.mi_row_start = mi_row_start;
  tile.mi_row_end = mi_row_end;
  return shim_fpc_get_unit_rows_in_tile(&tile, (BLOCK_SIZE)fp_block_size);
}

int shim_fp_get_unit_cols_in_tile(int mi_col_start, int mi_col_end,
                                  int fp_block_size) {
  TileInfo tile;
  memset(&tile, 0, sizeof(tile));
  tile.mi_col_start = mi_col_start;
  tile.mi_col_end = mi_col_end;
  return shim_fpc_get_unit_cols_in_tile(&tile, (BLOCK_SIZE)fp_block_size);
}

/* ======================================================================== *
 * 4. The file-static helpers. **Tier 1c.**
 * ======================================================================== */
int shim_fp_get_unit_rows(int fp_block_size, int mb_rows) {
  return get_unit_rows((BLOCK_SIZE)fp_block_size, mb_rows);
}

int shim_fp_get_unit_cols(int fp_block_size, int mb_cols) {
  return get_unit_cols((BLOCK_SIZE)fp_block_size, mb_cols);
}

int shim_fp_get_num_mbs(int fp_block_size, int num_mbs_16x16) {
  return get_num_mbs((BLOCK_SIZE)fp_block_size, num_mbs_16x16);
}

int shim_fp_get_search_range(int width, int height) {
  return get_search_range(width, height);
}

int shim_fp_find_fp_qindex(int bit_depth) {
  return find_fp_qindex((aom_bit_depth_t)bit_depth);
}

double shim_fp_raw_motion_error_stdev(int *list, int count) {
  return raw_motion_error_stdev(list, count);
}

void shim_fp_normalize_firstpass_stats(FIRSTPASS_STATS *fps,
                                       double num_mbs_16x16, double f_w,
                                       double f_h) {
  normalize_firstpass_stats(fps, num_mbs_16x16, f_w, f_h);
}

int shim_fp_get_bsize(int mi_rows, int mi_cols, int fp_block_size,
                      int unit_row, int unit_col) {
  CommonModeInfoParams mi_params;
  memset(&mi_params, 0, sizeof(mi_params));
  mi_params.mi_rows = mi_rows;
  mi_params.mi_cols = mi_cols;
  return (int)get_bsize(&mi_params, (BLOCK_SIZE)fp_block_size, unit_row,
                        unit_col);
}

int shim_fp_calc_wavelet_energy(int deltaq_mode) {
  AV1EncoderConfig *oxcf = (AV1EncoderConfig *)calloc(1, sizeof(*oxcf));
  if (!oxcf) return -1;
  oxcf->q_cfg.deltaq_mode = (DELTAQ_MODE)deltaq_mode;
  int r = calc_wavelet_energy(oxcf);
  free(oxcf);
  return r;
}

void shim_fp_accumulate_frame_stats(FRAME_STATS *mb_stats, int mb_rows,
                                    int mb_cols, FRAME_STATS *out) {
  *out = accumulate_frame_stats(mb_stats, mb_rows, mb_cols);
}

void shim_fp_accumulate_mv_stats(int16_t best_mv_row, int16_t best_mv_col,
                                 int16_t mv_row, int16_t mv_col, int mb_row,
                                 int mb_col, int mb_rows, int mb_cols,
                                 int16_t *last_non_zero_mv, FRAME_STATS *stats) {
  MV best_mv = { best_mv_row, best_mv_col };
  FULLPEL_MV mv = { mv_row, mv_col };
  MV last = { last_non_zero_mv[0], last_non_zero_mv[1] };
  accumulate_mv_stats(best_mv, mv, mb_row, mb_col, mb_rows, mb_cols, &last,
                      stats);
  last_non_zero_mv[0] = last.row;
  last_non_zero_mv[1] = last.col;
}

/* ---- get_prediction_error / highbd_get_prediction_error /
 *      get_prediction_error_bitdepth (firstpass.c:207, :244, :618) ---------
 * All three take `struct buf_2d`s, which the shim builds over the caller's
 * plane. The highbd arm goes through CONVERT_TO_BYTEPTR, exactly as the
 * encoder's buffers do.
 */
unsigned int shim_fp_get_prediction_error(int bsize, const uint8_t *src,
                                          int src_stride, const uint8_t *ref,
                                          int ref_stride) {
  struct buf_2d s, r;
  memset(&s, 0, sizeof(s));
  memset(&r, 0, sizeof(r));
  s.buf = (uint8_t *)src;
  s.stride = src_stride;
  r.buf = (uint8_t *)ref;
  r.stride = ref_stride;
  return get_prediction_error((BLOCK_SIZE)bsize, &s, &r);
}

unsigned int shim_fp_highbd_get_prediction_error(int bsize,
                                                 const uint16_t *src,
                                                 int src_stride,
                                                 const uint16_t *ref,
                                                 int ref_stride, int bd) {
#if CONFIG_AV1_HIGHBITDEPTH
  struct buf_2d s, r;
  memset(&s, 0, sizeof(s));
  memset(&r, 0, sizeof(r));
  s.buf = CONVERT_TO_BYTEPTR(src);
  s.stride = src_stride;
  r.buf = CONVERT_TO_BYTEPTR(ref);
  r.stride = ref_stride;
  return highbd_get_prediction_error((BLOCK_SIZE)bsize, &s, &r, bd);
#else
  (void)bsize; (void)src; (void)src_stride; (void)ref; (void)ref_stride;
  (void)bd;
  return 0;
#endif
}

int shim_fp_get_prediction_error_bitdepth(int is_high_bitdepth, int bitdepth,
                                          int bsize, const uint16_t *src16,
                                          const uint8_t *src8, int src_stride,
                                          const uint16_t *ref16,
                                          const uint8_t *ref8,
                                          int ref_stride) {
  struct buf_2d s, r;
  memset(&s, 0, sizeof(s));
  memset(&r, 0, sizeof(r));
  s.buf = is_high_bitdepth ? CONVERT_TO_BYTEPTR(src16) : (uint8_t *)src8;
  s.stride = src_stride;
  r.buf = is_high_bitdepth ? CONVERT_TO_BYTEPTR(ref16) : (uint8_t *)ref8;
  r.stride = ref_stride;
  return get_prediction_error_bitdepth(is_high_bitdepth, bitdepth,
                                       (BLOCK_SIZE)bsize, &s, &r);
}

/* ---- get_bsize (firstpass.c:335) -------------------------------------- */
int shim_fp_get_bsize2(int mi_rows, int mi_cols, int fp_block_size,
                       int unit_row, int unit_col) {
  CommonModeInfoParams mi_params;
  memset(&mi_params, 0, sizeof(mi_params));
  mi_params.mi_rows = mi_rows;
  mi_params.mi_cols = mi_cols;
  return (int)get_bsize(&mi_params, (BLOCK_SIZE)fp_block_size, unit_row,
                        unit_col);
}

/* ---- update_firstpass_stats (firstpass.c:907) ------------------------- *
 * Drives the whole static, and returns the FIRSTPASS_STATS it wrote into
 * `twopass->stats_buf_ctx->stats_in_end`.
 *
 * Three shim choices, each of which suppresses a side effect the port does
 * not model, rather than changing the arithmetic:
 *   - `lap_enabled = 1` sends the record to av1_firstpass_info_push instead
 *     of output_stats (which would push an aom_codec_cx_pkt onto a list);
 *     firstpass_info is left zeroed, so push returns AOM_CODEC_ERROR and
 *     writes nothing.
 *   - `total_stats = NULL` skips the av1_accumulate_stats fold.
 *   - `use_ducky_encode = 1` skips the circular/linear buffer wrap entirely.
 * `resize_mode` is left RESIZE_NONE so `num_mbs_16X16` comes from
 * `mi_params->MBs`, which the caller sets directly.
 */
int shim_fp_update_firstpass_stats(int num_mbs_16x16, int fp_block_size,
                                   int frame_number, int64_t ts_duration,
                                   double raw_err_stdev, int width, int height,
                                   const FRAME_STATS *stats,
                                   FIRSTPASS_STATS *out) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(*cpi));
  AV1_PRIMARY *ppi = (AV1_PRIMARY *)calloc(1, sizeof(*ppi));
  FIRSTPASS_STATS *buf = (FIRSTPASS_STATS *)calloc(8, sizeof(*buf));
  STATS_BUFFER_CTX *ctx = (STATS_BUFFER_CTX *)calloc(1, sizeof(*ctx));
  if (!cpi || !ppi || !buf || !ctx) {
    free(ctx);
    free(buf);
    free(ppi);
    free(cpi);
    return -1;
  }
  cpi->ppi = ppi;
  cpi->common.mi_params.MBs = num_mbs_16x16;
  cpi->common.width = width;
  cpi->common.height = height;
  cpi->oxcf.resize_cfg.resize_mode = RESIZE_NONE;
  cpi->use_ducky_encode = 1;
  ppi->lap_enabled = 1;
  ppi->twopass.stats_buf_ctx = ctx;
  ctx->stats_in_start = buf;
  ctx->stats_in_end = buf;
  ctx->stats_in_buf_end = buf + 8;
  ctx->total_stats = NULL;

  update_firstpass_stats(cpi, stats, raw_err_stdev, frame_number, ts_duration,
                         (BLOCK_SIZE)fp_block_size);

  *out = buf[0];
  free(ctx);
  free(buf);
  free(ppi);
  free(cpi);
  return 0;
}
