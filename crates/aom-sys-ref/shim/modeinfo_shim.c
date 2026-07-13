/* Oracles for the mode-info partition CDF primitives — the real static-inline
 * partition_cdf_length / partition_gather_{vert,horz}_alike from av1_common_int.h. */
#include <stdint.h>
#include "av1/common/av1_common_int.h"

int shim_partition_cdf_length(int bsize) {
  return partition_cdf_length((BLOCK_SIZE)bsize);
}

void shim_partition_gather_vert(uint16_t *out, const uint16_t *in, int bsize) {
  partition_gather_vert_alike(out, in, (BLOCK_SIZE)bsize);
}

void shim_partition_gather_horz(uint16_t *out, const uint16_t *in, int bsize) {
  partition_gather_horz_alike(out, in, (BLOCK_SIZE)bsize);
}

/* Facade: set the two partition-context pointers on a stack MACROBLOCKD and call
 * the real partition_plane_context (it reads only those two fields). */
#include "av1/common/blockd.h"
int shim_partition_plane_context(const signed char *above, const signed char *left,
                                 int mi_row, int mi_col, int bsize) {
  MACROBLOCKD xd;
  xd.above_partition_context = (PARTITION_CONTEXT *)above; /* pointer field */
  for (int i = 0; i < MAX_MIB_SIZE; i++)
    xd.left_partition_context[i] = left[i]; /* inline array field */
  return partition_plane_context(&xd, mi_row, mi_col, (BLOCK_SIZE)bsize);
}

/* Transcribed body of write_partition over the pristine C od_ec + update_cdf: every
 * symbol write is aom_write_symbol (encode + adapt) or aom_write_cdf (encode only, on
 * the gathered edge CDF). Returns the coded bytes + the adapted partition CDF. */
#include "aom_dsp/entenc.h"
#include "aom_dsp/prob.h"
uint32_t shim_write_partition(uint16_t *partition_cdf, int cdf_len, int p, int has_rows,
                              int has_cols, int bsize, uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 1024);
  if (bsize >= BLOCK_8X8) {
    if (has_rows && has_cols) {
      od_ec_encode_cdf_q15(&ec, p, partition_cdf, cdf_len);
      update_cdf(partition_cdf, p, cdf_len);
    } else if (!has_rows && has_cols) {
      aom_cdf_prob cdf[2];
      partition_gather_vert_alike(cdf, partition_cdf, (BLOCK_SIZE)bsize);
      od_ec_encode_cdf_q15(&ec, p == PARTITION_SPLIT, cdf, 2);
    } else if (has_rows && !has_cols) {
      aom_cdf_prob cdf[2];
      partition_gather_horz_alike(cdf, partition_cdf, (BLOCK_SIZE)bsize);
      od_ec_encode_cdf_q15(&ec, p == PARTITION_SPLIT, cdf, 2);
    }
  }
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < cdf_len + 1; i++) out_cdf[i] = partition_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}
