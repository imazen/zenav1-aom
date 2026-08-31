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
