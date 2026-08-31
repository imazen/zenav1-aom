/* Oracle shims for the small exported encoder helpers ported into
 * crate aom-encode, module enc_misc. Oracle use only. Every entry drives the
 * REAL exported C function.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/scan.h"
#include "av1/encoder/block.h"
#include "av1/encoder/encodemb.h"
#include "av1/encoder/hash_motion.h"
#include "av1/encoder/rd.h"

int shim_get_intra_cost_penalty(int qindex, int qdelta, int bit_depth) {
  return av1_get_intra_cost_penalty(qindex, qdelta,
                                    (aom_bit_depth_t)bit_depth);
}

/* ---- av1_hash_is_{horizontal,vertical}_perfect ----------------------
 * Both take a YV12_BUFFER_CONFIG; the shim builds one over the caller's plane.
 * `highbd` selects the uint16 arm (the buffer's YV12_FLAG_HIGHBITDEPTH), which
 * is what the port's u16 planes correspond to.
 */
int shim_hash_is_horizontal_perfect(const uint16_t *plane, int stride,
                                    int block_size, int x_start, int y_start,
                                    int highbd) {
  YV12_BUFFER_CONFIG buf;
  memset(&buf, 0, sizeof(buf));
  buf.y_stride = stride;
  buf.flags = highbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  buf.y_buffer = highbd ? CONVERT_TO_BYTEPTR(plane) : (uint8_t *)plane;
  return av1_hash_is_horizontal_perfect(&buf, block_size, x_start, y_start);
}

int shim_hash_is_vertical_perfect(const uint16_t *plane, int stride,
                                  int block_size, int x_start, int y_start,
                                  int highbd) {
  YV12_BUFFER_CONFIG buf;
  memset(&buf, 0, sizeof(buf));
  buf.y_stride = stride;
  buf.flags = highbd ? YV12_FLAG_HIGHBITDEPTH : 0;
  buf.y_buffer = highbd ? CONVERT_TO_BYTEPTR(plane) : (uint8_t *)plane;
  return av1_hash_is_vertical_perfect(&buf, block_size, x_start, y_start);
}

/* ---- av1_dropout_qcoeff_num -----------------------------------------
 * Builds the MACROBLOCK plane the function reads: qcoeff / dqcoeff / eobs /
 * txb_entropy_ctx at block 0. `tx_size` and `tx_type` pick both the scan order
 * and `av1_get_max_eob`, so they are passed through rather than derived.
 * qcoeff/dqcoeff are edited in place and the new eob is returned.
 */
int shim_dropout_qcoeff_num(int32_t *qcoeff, int32_t *dqcoeff, int eob,
                            int tx_size, int tx_type, int dropout_num_before,
                            int dropout_num_after) {
  MACROBLOCK *mb = (MACROBLOCK *)calloc(1, sizeof(MACROBLOCK));
  if (!mb) return -1;
  const int max_eob = av1_get_max_eob((TX_SIZE)tx_size);
  uint16_t *eobs = (uint16_t *)calloc(1, sizeof(uint16_t));
  uint8_t *ectx = (uint8_t *)calloc(1, sizeof(uint8_t));
  if (!eobs || !ectx) {
    free(mb); free(eobs); free(ectx);
    return -1;
  }
  eobs[0] = (uint16_t)eob;
  mb->plane[0].qcoeff = qcoeff;
  mb->plane[0].dqcoeff = dqcoeff;
  mb->plane[0].eobs = eobs;
  mb->plane[0].txb_entropy_ctx = ectx;

  av1_dropout_qcoeff_num(mb, 0, 0, (TX_SIZE)tx_size, (TX_TYPE)tx_type,
                         dropout_num_before, dropout_num_after);

  const int out = (int)eobs[0];
  (void)max_eob;
  free(mb); free(eobs); free(ectx);
  return out;
}

/* ---- av1/encoder/ratectrl.c: the rate model -------------------------
 * Drives the REAL exported av1_convert_qindex_to_q / av1_find_qindex /
 * av1_compute_qdelta / av1_rc_bits_per_mb / av1_rc_get_default_min_gf_interval.
 *
 * The two that take AV1_COMP / RATE_CONTROL get a calloc'd one with ONLY the
 * fields they read populated, and — for av1_rc_bits_per_mb — with
 * `oxcf.rc_cfg.mode` set to AOM_Q so both CBR-only enumerator overrides are
 * provably not taken. Driving it with a zeroed mode would silently pick
 * AOM_VBR (0) instead, which happens to take the same arm today but is not the
 * arm the port claims to implement.
 */
#include "av1/encoder/ratectrl.h"
#include "av1/encoder/encoder.h"

double shim_convert_qindex_to_q(int qindex, int bit_depth) {
  return av1_convert_qindex_to_q(qindex, (aom_bit_depth_t)bit_depth);
}

int shim_find_qindex(double desired_q, int bit_depth, int best_qindex,
                     int worst_qindex) {
  return av1_find_qindex(desired_q, (aom_bit_depth_t)bit_depth, best_qindex,
                         worst_qindex);
}

int shim_compute_qdelta(double qstart, double qtarget, int bit_depth,
                        int best_quality, int worst_quality) {
  RATE_CONTROL rc;
  memset(&rc, 0, sizeof(rc));
  rc.best_quality = best_quality;
  rc.worst_quality = worst_quality;
  return av1_compute_qdelta(&rc, qstart, qtarget, (aom_bit_depth_t)bit_depth);
}

int shim_rc_bits_per_mb(int is_key_frame, int is_screen_content_type,
                        int qindex, double correction_factor, int bit_depth) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(AV1_COMP));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(SequenceHeader));
  if (!cpi || !seq) {
    free(cpi);
    free(seq);
    return -1;
  }
  seq->bit_depth = (aom_bit_depth_t)bit_depth;
  cpi->common.seq_params = seq;
  cpi->is_screen_content_type = is_screen_content_type;
  /* AOM_Q: neither CBR override can be taken. */
  cpi->oxcf.rc_cfg.mode = AOM_Q;
  const int r = av1_rc_bits_per_mb(
      cpi, is_key_frame ? KEY_FRAME : INTER_FRAME, qindex, correction_factor,
      /*accurate_estimate=*/0);
  free(cpi);
  free(seq);
  return r;
}

int shim_rc_get_default_min_gf_interval(int width, int height,
                                        double framerate) {
  return av1_rc_get_default_min_gf_interval(width, height, framerate);
}

/* ---- av1/encoder/rd.c: the RD mode-threshold machinery ---------------
 * `av1_set_rd_speed_thresholds` writes `cpi->rd.thresh_mult[MAX_MODES]`; the
 * shim returns the whole array plus MAX_MODES, so one comparison covers both
 * the THR_MODES ordering and the ~169 constants (a wrong index puts a right
 * value in a wrong slot).
 *
 * `av1_update_rd_thresh_fact` reads only `cm->seq_params->sb_size` out of
 * AV1_COMMON, and edits the caller's [BLOCK_SIZES_ALL][MAX_MODES] buffer.
 */
int shim_set_rd_speed_thresholds(int32_t *out, int out_len) {
  if (out_len < MAX_MODES) return -MAX_MODES;
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(AV1_COMP));
  if (!cpi) return -1;
  av1_set_rd_speed_thresholds(cpi);
  for (int i = 0; i < MAX_MODES; ++i) out[i] = cpi->rd.thresh_mult[i];
  free(cpi);
  return MAX_MODES;
}

int shim_update_rd_thresh_fact(int sb_size, int32_t *factor_buf,
                               int use_adaptive_rd_thresh, int bsize,
                               int best_mode_index, int inter_mode_start,
                               int inter_mode_end, int intra_mode_start,
                               int intra_mode_end) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(AV1_COMMON));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(SequenceHeader));
  if (!cm || !seq) {
    free(cm);
    free(seq);
    return -1;
  }
  seq->sb_size = (BLOCK_SIZE)sb_size;
  cm->seq_params = seq;

  av1_update_rd_thresh_fact(cm, (int(*)[MAX_MODES])factor_buf,
                            use_adaptive_rd_thresh, (BLOCK_SIZE)bsize,
                            (THR_MODES)best_mode_index,
                            (THR_MODES)inter_mode_start,
                            (THR_MODES)inter_mode_end,
                            (THR_MODES)intra_mode_start,
                            (THR_MODES)intra_mode_end);
  free(cm);
  free(seq);
  return 0;
}
