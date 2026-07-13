/* Shim over av1 txb coefficient-coding kernels + scan/ctx-offset data.
 * Oracle use only. */
#include <stdint.h>

typedef int32_t tran_low_t;

void av1_txb_init_levels_c(const tran_low_t *coeff, int width, int height,
                           uint8_t *levels);
void av1_get_nz_map_contexts_c(const uint8_t *levels, const int16_t *scan,
                               int eob, int tx_size, int tx_class,
                               int8_t *coeff_contexts);
int av1_get_eob_pos_token(int eob, int *extra);

extern const int8_t *av1_nz_map_ctx_offset[19];

/* Layout mirror of libaom SCAN_ORDER (av1/common/scan.h): two pointers. */
typedef struct {
  const int16_t *scan;
  const int16_t *iscan;
} SHIM_SCAN_ORDER;
extern const SHIM_SCAN_ORDER av1_scan_orders[19][16];

void shim_txb_init_levels(const int32_t *coeff, int width, int height,
                          uint8_t *levels) {
  av1_txb_init_levels_c(coeff, width, height, levels);
}

void shim_get_nz_map_contexts(const uint8_t *levels, const int16_t *scan,
                              int eob, int tx_size, int tx_class,
                              int8_t *out) {
  av1_get_nz_map_contexts_c(levels, scan, eob, tx_size, tx_class, out);
}

int shim_eob_pos_token(int eob, int *extra) {
  return av1_get_eob_pos_token(eob, extra);
}

const int8_t *shim_nz_ctx_offset(int tx_size) {
  return av1_nz_map_ctx_offset[tx_size];
}

const int16_t *shim_scan(int tx_size, int tx_type) {
  return av1_scan_orders[tx_size][tx_type].scan;
}

const int16_t *shim_iscan(int tx_size, int tx_type) {
  return av1_scan_orders[tx_size][tx_type].iscan;
}
