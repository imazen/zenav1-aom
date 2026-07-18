/*
 * Superres denom-selection reference shim.
 *
 * `analyze_hor_freq`, `get_superres_denom_from_qindex_energy` and the QTHRESH
 * arm of `calculate_next_superres_scale` are `static` in
 * av1/encoder/superres_scale.c, so they cannot be linked directly. This shim is
 * a FAITHFUL transcription of those functions that calls the REAL exported leaf
 * math it depends on — `av1_fwd_txfm2d_16x4_c` (the scalar C 16x4 forward
 * transform) and `av1_convert_qindex_to_q` (ratectrl.c) — so the port's
 * `aom_encode::superres_select` can be differentially validated against C's
 * exact energy/threshold arithmetic (a "facade-over-real-fn" reference). The
 * end-to-end gate against real `aomenc` (encoder_gate_superres_e2e.rs) remains
 * the top-tier evidence; this shim isolates the arithmetic from the parse/pack
 * wiring so a denom mismatch localizes fast.
 *
 * Append-only oracle addition; touches no other shim.
 */
#include <stdint.h>
#include <string.h>
#include "av1/common/enums.h"        /* TX_TYPE (H_DCT), TX_SIZE (TX_16X4) */
#include "aom/aom_codec.h"           /* aom_bit_depth_t (AOM_BITS_8) */
#include "aom_ports/mem.h"           /* ROUND_POWER_OF_TWO */

/* Real exported leaf math (scalar C reference — no RTCD needed). */
void av1_fwd_txfm2d_16x4_c(const int16_t *input, int32_t *output, int stride,
                           TX_TYPE tx_type, int bd);
double av1_convert_qindex_to_q(int qindex, aom_bit_depth_t bit_depth);

#define SR_SCALE_NUMERATOR 8

/* Faithful transcription of superres_scale.c `analyze_hor_freq`, generalized to
 * take a raw luma buffer (`src`, tight or strided by `stride`, uint16 samples;
 * for bd==8 the values are 0..255) instead of `cpi->source`. Writes energy[0]
 * unused, energy[1..15] set. */
void shim_superres_analyze_hor_freq(const uint16_t *src, int width, int height,
                                    int stride, int bd, double *energy) {
  uint64_t freq_energy[16] = { 0 };
  int32_t coeff[16 * 4];
  int n = 0;
  memset(freq_energy, 0, sizeof(freq_energy));
  if (bd > 8) {
    /* Highbd path: transform reads the source directly at `stride`. The samples
     * are int16 (values fit for bd 10/12). */
    for (int i = 0; i < height - 4; i += 4) {
      for (int j = 0; j < width - 16; j += 16) {
        av1_fwd_txfm2d_16x4_c((const int16_t *)(src + (size_t)i * stride + j),
                              coeff, stride, H_DCT, bd);
        for (int k = 1; k < 16; ++k) {
          const uint64_t this_energy = ((int64_t)coeff[k] * coeff[k]) +
                                       ((int64_t)coeff[k + 16] * coeff[k + 16]) +
                                       ((int64_t)coeff[k + 32] * coeff[k + 32]) +
                                       ((int64_t)coeff[k + 48] * coeff[k + 48]);
          freq_energy[k] += ROUND_POWER_OF_TWO(this_energy, 2 + 2 * (bd - 8));
        }
        n++;
      }
    }
  } else {
    int16_t src16[16 * 4];
    for (int i = 0; i < height - 4; i += 4) {
      for (int j = 0; j < width - 16; j += 16) {
        for (int ii = 0; ii < 4; ++ii)
          for (int jj = 0; jj < 16; ++jj)
            src16[ii * 16 + jj] =
                (int16_t)src[(size_t)(i + ii) * stride + (j + jj)];
        av1_fwd_txfm2d_16x4_c(src16, coeff, 16, H_DCT, bd);
        for (int k = 1; k < 16; ++k) {
          const uint64_t this_energy = ((int64_t)coeff[k] * coeff[k]) +
                                       ((int64_t)coeff[k + 16] * coeff[k + 16]) +
                                       ((int64_t)coeff[k + 32] * coeff[k + 32]) +
                                       ((int64_t)coeff[k + 48] * coeff[k + 48]);
          freq_energy[k] += ROUND_POWER_OF_TWO(this_energy, 2);
        }
        n++;
      }
    }
  }
  if (n) {
    for (int k = 1; k < 16; ++k) energy[k] = (double)freq_energy[k] / n;
    for (int k = 14; k > 0; --k) energy[k] += energy[k + 1];
  } else {
    for (int k = 1; k < 16; ++k) energy[k] = 1e+20;
  }
}

/* Faithful transcription of superres_scale.c
 * `get_superres_denom_from_qindex_energy`. */
uint8_t shim_superres_denom_from_qindex_energy(int qindex, const double *energy,
                                               double threshq, double threshp) {
  const double q = av1_convert_qindex_to_q(qindex, AOM_BITS_8);
  const double tq = threshq * q * q;
  const double tp = threshp * energy[1];
  const double thresh = tq < tp ? tq : tp; /* AOMMIN */
  int k;
  for (k = SR_SCALE_NUMERATOR * 2; k > SR_SCALE_NUMERATOR; --k) {
    if (energy[k - 1] > thresh) break;
  }
  return (uint8_t)(3 * SR_SCALE_NUMERATOR - k);
}

#define SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME_SOLO 0.012
#define SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME 0.008
#define SUPERRES_ENERGY_BY_AC_THRESH 0.2

/* The KEY-frame QTHRESH arm of `calculate_next_superres_scale` +
 * `get_superres_denom_for_qindex` (sr_kf=1, KF_UPDATE), restricted to the
 * single-KEY AOM_Q envelope: `av1_set_target_rate` is not called (Q mode), the
 * recode bump is AUTO-only (never fires for QTHRESH), and `q` is passed in (the
 * qindex the rate controller picked). `frames_to_key_le_1` selects the SOLO vs
 * non-SOLO energy_by_q2 threshold. Returns the superres denominator (8..16;
 * 8 == no superres). */
uint8_t shim_superres_denom_qthresh_key(const uint16_t *src, int w, int h,
                                        int stride, int bd, int q,
                                        int kf_qthresh_qindex, int allow_scc,
                                        int frames_to_key_le_1) {
  if (allow_scc) return SR_SCALE_NUMERATOR; /* screen content: no superres */
  if (q <= kf_qthresh_qindex) return SR_SCALE_NUMERATOR;
  double energy[16];
  shim_superres_analyze_hor_freq(src, w, h, stride, bd, energy);
  const double energy_by_q2_thresh =
      frames_to_key_le_1 ? SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME_SOLO
                         : SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME;
  return shim_superres_denom_from_qindex_energy(q, energy, energy_by_q2_thresh,
                                                SUPERRES_ENERGY_BY_AC_THRESH);
}
