/* Tier-1 oracle for prune_tx_2D (tx_search.c:1541).
 *
 * The static helpers (get_energy_distribution_finer, get_adaptive_thresholds +
 * its table) and the driver body are copied VERBATIM from tx_search.c; the
 * exported av1_get_horver_correlation_full / av1_nn_predict /
 * av1_nn_fast_softmax_16 and the real non-V2 av1_tx_type_nnconfig_map_{hor,ver}
 * (+ weights) come from the linked libaom via the includes below. So this is the
 * real algorithm, same C compiler, byte-identical to the encoder's prune_tx_2D
 * — it just takes the txb residual `diff` (stride `diff_stride`) directly
 * instead of marshalling a MACROBLOCK. */

#include <stdint.h>
#include <string.h>
#include "config/av1_rtcd.h"        /* av1_nn_predict, av1_nn_fast_softmax_16, av1_get_horver_correlation_full (rtcd dispatch) */
#include "av1/common/enums.h"       /* TX_SIZE, TxSetType, TX_TYPE(_INVALID), TX_TYPES */
#include "av1/common/common_data.h" /* tx_size_wide, tx_size_high */
#include "av1/encoder/ml.h"         /* NN_CONFIG */
#include "av1/encoder/sorting_network.h" /* av1_sort_fi32_8 / _16 */

/* TX_TYPE_PRUNE_MODE enum values (speed_features.h — not includable here: its
 * context_tree.h dep needs PARTITION_SEARCH_TYPE). Only these two are used. */
#define TX_TYPE_PRUNE_1 1
#define TX_TYPE_PRUNE_4 4

/* the real non-V2 hor/ver nnconfig maps + weights (CONFIG_NN_V2 == 0) */
#include "av1/encoder/tx_prune_model_weights.h"

/* ---- copied verbatim from tx_search.c:1390-1527 ---- */
static const float *prune_2D_adaptive_thresholds_shim[] = {
  (float[]){ 0.00549f, 0.01306f, 0.02039f, 0.02747f, 0.03406f, 0.04065f,
             0.04724f, 0.05383f, 0.06067f, 0.06799f, 0.07605f, 0.08533f,
             0.09778f, 0.11780f },
  (float[]){ 0.00037f, 0.00183f, 0.00525f, 0.01038f, 0.01697f, 0.02502f,
             0.03381f, 0.04333f, 0.05286f, 0.06287f, 0.07434f, 0.08850f,
             0.10803f, 0.14124f },
  (float[]){ 0.01404f, 0.02000f, 0.04211f, 0.05164f, 0.05798f, 0.06335f,
             0.06897f, 0.07629f, 0.08875f, 0.11169f },
  NULL,
  NULL,
  (float[]){ 0.00183f, 0.00745f, 0.01428f, 0.02185f, 0.02966f, 0.03723f,
             0.04456f, 0.05188f, 0.05920f, 0.06702f, 0.07605f, 0.08704f,
             0.10168f, 0.12585f },
  (float[]){ 0.00085f, 0.00476f, 0.01135f, 0.01892f, 0.02698f, 0.03528f,
             0.04358f, 0.05164f, 0.05994f, 0.06848f, 0.07849f, 0.09021f,
             0.10583f, 0.13123f },
  (float[]){ 0.00037f, 0.00232f, 0.00671f, 0.01257f, 0.01965f, 0.02722f,
             0.03552f, 0.04382f, 0.05237f, 0.06189f, 0.07336f, 0.08728f,
             0.10730f, 0.14221f },
  (float[]){ 0.00061f, 0.00330f, 0.00818f, 0.01453f, 0.02185f, 0.02966f,
             0.03772f, 0.04578f, 0.05383f, 0.06262f, 0.07288f, 0.08582f,
             0.10339f, 0.13464f },
  NULL,
  NULL,
  NULL,
  NULL,
  (float[]){ 0.00232f, 0.00671f, 0.01257f, 0.01941f, 0.02673f, 0.03430f,
             0.04211f, 0.04968f, 0.05750f, 0.06580f, 0.07507f, 0.08655f,
             0.10242f, 0.12878f },
  (float[]){ 0.00110f, 0.00525f, 0.01208f, 0.01990f, 0.02795f, 0.03601f,
             0.04358f, 0.05115f, 0.05896f, 0.06702f, 0.07629f, 0.08752f,
             0.10217f, 0.12610f },
  NULL,
  NULL,
  NULL,
  NULL,
};

static inline float get_adaptive_thresholds_shim(int tx_size, int tx_set_type,
                                                 int prune_2d_txfm_mode) {
  const int prune_aggr_table[5][2] = {
    { 4, 1 }, { 6, 3 }, { 9, 6 }, { 9, 6 }, { 12, 9 }
  };
  int pruning_aggressiveness = 0;
  if (tx_set_type == EXT_TX_SET_ALL16)
    pruning_aggressiveness = prune_aggr_table[prune_2d_txfm_mode - TX_TYPE_PRUNE_1][0];
  else if (tx_set_type == EXT_TX_SET_DTT9_IDTX_1DDCT)
    pruning_aggressiveness = prune_aggr_table[prune_2d_txfm_mode - TX_TYPE_PRUNE_1][1];
  return prune_2D_adaptive_thresholds_shim[tx_size][pruning_aggressiveness];
}

static inline void get_energy_distribution_finer_shim(const int16_t *diff,
                                                      int stride, int bw, int bh,
                                                      float *hordist,
                                                      float *verdist) {
  unsigned int esq[256];
  const int w_shift = bw <= 8 ? 0 : 1;
  const int h_shift = bh <= 8 ? 0 : 1;
  const int esq_w = bw >> w_shift;
  const int esq_h = bh >> h_shift;
  const int esq_sz = esq_w * esq_h;
  int i, j;
  memset(esq, 0, esq_sz * sizeof(esq[0]));
  if (w_shift) {
    for (i = 0; i < bh; i++) {
      unsigned int *cur_esq_row = esq + (i >> h_shift) * esq_w;
      const int16_t *cur_diff_row = diff + i * stride;
      for (j = 0; j < bw; j += 2) {
        cur_esq_row[j >> 1] += (cur_diff_row[j] * cur_diff_row[j] +
                                cur_diff_row[j + 1] * cur_diff_row[j + 1]);
      }
    }
  } else {
    for (i = 0; i < bh; i++) {
      unsigned int *cur_esq_row = esq + (i >> h_shift) * esq_w;
      const int16_t *cur_diff_row = diff + i * stride;
      for (j = 0; j < bw; j++) {
        cur_esq_row[j] += cur_diff_row[j] * cur_diff_row[j];
      }
    }
  }

  uint64_t total = 0;
  for (i = 0; i < esq_sz; i++) total += esq[i];

  if (total == 0) {
    float hor_val = 1.0f / esq_w;
    for (j = 0; j < esq_w - 1; j++) hordist[j] = hor_val;
    float ver_val = 1.0f / esq_h;
    for (i = 0; i < esq_h - 1; i++) verdist[i] = ver_val;
    return;
  }

  const float e_recip = 1.0f / (float)total;
  memset(hordist, 0, (esq_w - 1) * sizeof(hordist[0]));
  memset(verdist, 0, (esq_h - 1) * sizeof(verdist[0]));
  const unsigned int *cur_esq_row;
  for (i = 0; i < esq_h - 1; i++) {
    cur_esq_row = esq + i * esq_w;
    for (j = 0; j < esq_w - 1; j++) {
      hordist[j] += (float)cur_esq_row[j];
      verdist[i] += (float)cur_esq_row[j];
    }
    verdist[i] += (float)cur_esq_row[j];
  }
  cur_esq_row = esq + i * esq_w;
  for (j = 0; j < esq_w - 1; j++) hordist[j] += (float)cur_esq_row[j];

  for (j = 0; j < esq_w - 1; j++) hordist[j] *= e_recip;
  for (i = 0; i < esq_h - 1; i++) verdist[i] *= e_recip;
}

static inline int check_bit_mask_shim(uint16_t mask, int val) {
  return mask & (1 << val);
}
static inline void set_bit_mask_shim(uint16_t *mask, int val) {
  *mask |= (1 << val);
}
static inline void unset_bit_mask_shim(uint16_t *mask, int val) {
  *mask &= ~(1 << val);
}

/* The driver body copied from tx_search.c:1547-1694, taking `diff`/`diff_stride`
 * directly (the caller's `4*blk_row*diff_stride + 4*blk_col` offset is baked in).
 * The `prune_2d_txfm_mode >= TX_TYPE_PRUNE_4` block is kept for fidelity. */
void shim_prune_tx_2D(const int16_t *diff, int diff_stride, int tx_size,
                      int tx_set_type, int prune_2d_txfm_mode, uint16_t in_mask,
                      uint16_t *out_mask, int *out_txk_map) {
  static const int tx_type_table_2D[16] = {
    DCT_DCT,      DCT_ADST,      DCT_FLIPADST,      V_DCT,
    ADST_DCT,     ADST_ADST,     ADST_FLIPADST,     V_ADST,
    FLIPADST_DCT, FLIPADST_ADST, FLIPADST_FLIPADST, V_FLIPADST,
    H_DCT,        H_ADST,        H_FLIPADST,        IDTX
  };
  uint16_t allowed = in_mask;
  int *txk_map = out_txk_map;

  if (tx_set_type != EXT_TX_SET_ALL16 &&
      tx_set_type != EXT_TX_SET_DTT9_IDTX_1DDCT) {
    *out_mask = allowed;
    for (int k = 0; k < 16; k++) txk_map[k] = tx_type_table_2D[k];
    return;
  }
  const NN_CONFIG *nn_config_hor = av1_tx_type_nnconfig_map_hor[tx_size];
  const NN_CONFIG *nn_config_ver = av1_tx_type_nnconfig_map_ver[tx_size];
  if (!nn_config_hor || !nn_config_ver) {
    *out_mask = allowed;
    for (int k = 0; k < 16; k++) txk_map[k] = tx_type_table_2D[k];
    return;
  }

  float hfeatures[16], vfeatures[16];
  float hscores[4], vscores[4];
  float scores_2D_raw[16];
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];
  const int hfeatures_num = bw <= 8 ? bw : bw / 2;
  const int vfeatures_num = bh <= 8 ? bh : bh / 2;

  get_energy_distribution_finer_shim(diff, diff_stride, bw, bh, hfeatures,
                                     vfeatures);
  av1_get_horver_correlation_full_c(diff, diff_stride, bw, bh,
                                  &hfeatures[hfeatures_num - 1],
                                  &vfeatures[vfeatures_num - 1]);

  av1_nn_predict_c(hfeatures, nn_config_hor, 1, hscores);
  av1_nn_predict_c(vfeatures, nn_config_ver, 1, vscores);

  for (int i = 0; i < 4; i++) {
    float *cur_scores_2D = scores_2D_raw + i * 4;
    cur_scores_2D[0] = vscores[i] * hscores[0];
    cur_scores_2D[1] = vscores[i] * hscores[1];
    cur_scores_2D[2] = vscores[i] * hscores[2];
    cur_scores_2D[3] = vscores[i] * hscores[3];
  }
  av1_nn_fast_softmax_16_c(scores_2D_raw, scores_2D_raw);

  const float score_thresh =
      get_adaptive_thresholds_shim(tx_size, tx_set_type, prune_2d_txfm_mode);

  int max_score_i = 0;
  float max_score = 0.0f;
  uint16_t allow_bitmask = 0;
  float sum_score = 0.0;
  int allow_count = 0;
  int tx_type_allowed[16];
  for (int k = 0; k < 16; k++) tx_type_allowed[k] = TX_TYPE_INVALID;
  float scores_2D[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  };
  for (int tx_idx = 0; tx_idx < 16; tx_idx++) {
    const int allow_tx_type = check_bit_mask_shim(allowed, tx_type_table_2D[tx_idx]);
    if (!allow_tx_type) continue;
    if (scores_2D_raw[tx_idx] > max_score) {
      max_score = scores_2D_raw[tx_idx];
      max_score_i = tx_idx;
    }
    if (scores_2D_raw[tx_idx] >= score_thresh) {
      set_bit_mask_shim(&allow_bitmask, tx_type_table_2D[tx_idx]);
      sum_score += scores_2D_raw[tx_idx];
      scores_2D[allow_count] = scores_2D_raw[tx_idx];
      tx_type_allowed[allow_count] = tx_type_table_2D[tx_idx];
      allow_count += 1;
    }
  }
  if (!check_bit_mask_shim(allow_bitmask, tx_type_table_2D[max_score_i])) {
    set_bit_mask_shim(&allow_bitmask, tx_type_table_2D[max_score_i]);
    memcpy(txk_map, tx_type_table_2D, sizeof(tx_type_table_2D));
    *out_mask = allow_bitmask;
    return;
  }

  if (allow_count <= 8) {
    av1_sort_fi32_8(scores_2D, tx_type_allowed);
  } else {
    av1_sort_fi32_16(scores_2D, tx_type_allowed);
  }

  if (prune_2d_txfm_mode >= TX_TYPE_PRUNE_4) {
    float temp_score = 0.0;
    float score_ratio = 0.0;
    int tx_idx, tx_count = 0;
    const float inv_sum_score = 100 / sum_score;
    for (tx_idx = 0; tx_idx < allow_count; tx_idx++) {
      if (score_ratio > 30.0 && tx_count >= 2) break;
      temp_score += scores_2D[tx_idx];
      score_ratio = temp_score * inv_sum_score;
      tx_count++;
    }
    for (; tx_idx < allow_count; tx_idx++)
      unset_bit_mask_shim(&allow_bitmask, tx_type_allowed[tx_idx]);
  }

  memcpy(txk_map, tx_type_allowed, sizeof(tx_type_table_2D));
  *out_mask = allow_bitmask;
}
