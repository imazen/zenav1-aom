/* Oracle shims for the EXPORTED av1/encoder/ratectrl.c functions on the
 * fixed-Q rate-search path. Oracle use only.
 *
 * This TU deliberately does NOT include ratectrl.c, so every `av1_*` name
 * below binds to the copy inside `upstream/build/libaom.a`. That makes these
 * **tier 1** (the real exported symbol) rather than the tier 1c that
 * ratectrl_shim.c gives for the file-statics. The two TUs share
 * `rc_state_params.h`, so they build the same AV1_COMP from the same fields —
 * a difference between them is a difference between the compilations, not
 * between two hand-written setups.
 *
 * `-DNDEBUG` is passed by build.rs to every shim: `MACROBLOCK` has an
 * `#ifndef NDEBUG` member, so without it every field offset of `AV1_COMP` in
 * this TU disagrees with the archive's (DIFFERENTIAL_PLAYBOOK 3a(a)) — which
 * matters here more than anywhere, because the functions below read a dozen
 * fields scattered across the struct.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/alloccommon.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/ratectrl.h"

#include "rc_state_params.h"

#define SHIM_RCA_GF_INDEX 2

typedef struct {
  AV1_COMP *cpi;
  AV1_PRIMARY *ppi;
  SequenceHeader *seq;
} ShimRca;

static int shim_rca_alloc(ShimRca *s, const ShimRcStateParams *p) {
  s->cpi = (AV1_COMP *)calloc(1, sizeof(AV1_COMP));
  s->ppi = (AV1_PRIMARY *)calloc(1, sizeof(AV1_PRIMARY));
  s->seq = (SequenceHeader *)calloc(1, sizeof(SequenceHeader));
  if (!s->cpi || !s->ppi || !s->seq) {
    free(s->cpi); free(s->ppi); free(s->seq);
    s->cpi = NULL; s->ppi = NULL; s->seq = NULL;
    return 0;
  }
  AV1_COMP *cpi = s->cpi;
  cpi->ppi = s->ppi;
  s->seq->bit_depth = (aom_bit_depth_t)p->bit_depth;
  cpi->common.seq_params = s->seq;

  cpi->common.width = p->coded_width;
  cpi->common.height = p->coded_height;
  /* Unscaled: av1_frame_scaled() is superres_scaled || resize_scaled, both of
   * which compare the upscaled/render sizes against the coded ones. */
  cpi->common.superres_upscaled_width = p->coded_width;
  cpi->common.superres_upscaled_height = p->coded_height;
  cpi->common.render_width = p->coded_width;
  cpi->common.render_height = p->coded_height;
  cpi->common.superres_scale_denominator = SCALE_NUMERATOR;
  cpi->common.current_frame.frame_type = (FRAME_TYPE)p->frame_type;
  cpi->common.quant_params.base_qindex = p->base_qindex;
  cpi->common.mi_params.MBs = av1_get_MBs(p->coded_width, p->coded_height);

  cpi->is_screen_content_type = p->screen_content;
  cpi->refresh_frame.golden_frame = (bool)p->refresh_golden;
  cpi->refresh_frame.alt_ref_frame = (bool)p->refresh_alt_ref;

  cpi->oxcf.mode = p->rtc_mode ? REALTIME : GOOD;
  /* is_stat_consumption_stage() is (pass >= AOM_RC_SECOND_PASS) || lap. */
  cpi->oxcf.pass = p->stat_consumption ? AOM_RC_SECOND_PASS : AOM_RC_ONE_PASS;
  cpi->oxcf.rc_cfg.mode = (enum aom_rc_mode)p->rc_mode;
  cpi->oxcf.rc_cfg.gf_cbr_boost_pct = p->gf_cbr_boost_pct;
  cpi->oxcf.frm_dim_cfg.width = p->cfg_width;
  cpi->oxcf.frm_dim_cfg.height = p->cfg_height;

  cpi->sf.hl_sf.recode_tolerance = p->recode_tolerance;
  cpi->sf.hl_sf.accurate_bit_estimate = 0;

  cpi->rc.is_src_frame_alt_ref = p->is_src_frame_alt_ref;
  cpi->rc.best_quality = p->best_quality;
  cpi->rc.worst_quality = p->worst_quality;
  cpi->rc.max_frame_bandwidth = p->max_frame_bandwidth;

  for (int i = 0; i < RATE_FACTOR_LEVELS; ++i)
    cpi->ppi->p_rc.rate_correction_factors[i] = p->rate_correction_factors[i];

  cpi->gf_frame_index = SHIM_RCA_GF_INDEX;
  cpi->ppi->gf_group.update_type[SHIM_RCA_GF_INDEX] =
      (FRAME_UPDATE_TYPE)p->update_type;
  cpi->ppi->gf_group.layer_depth[SHIM_RCA_GF_INDEX] = p->layer_depth;
  cpi->ppi->gf_group.frame_type[SHIM_RCA_GF_INDEX] = (FRAME_TYPE)p->frame_type;
  /* frame_parallel_level 0 selects p_rc->rate_correction_factors (not the
   * per-frame copies), which is the arm the port implements. */
  cpi->ppi->gf_group.frame_parallel_level[SHIM_RCA_GF_INDEX] = 0;

  av1_rc_init_minq_luts();
  return 1;
}

static void shim_rca_free(ShimRca *s) {
  free(s->cpi); free(s->ppi); free(s->seq);
}

int shim_rca_get_MBs(int width, int height) {
  return av1_get_MBs(width, height);
}

int shim_rca_estimate_bits_at_q(const ShimRcStateParams *p, int q,
                                double correction_factor) {
  ShimRca s;
  if (!shim_rca_alloc(&s, p)) return INT_MIN;
  const int r = av1_estimate_bits_at_q(s.cpi, q, correction_factor);
  shim_rca_free(&s);
  return r;
}

int shim_rca_compute_qdelta_by_rate(const ShimRcStateParams *p, int frame_type,
                                    int qindex, double rate_target_ratio) {
  ShimRca s;
  if (!shim_rca_alloc(&s, p)) return INT_MIN;
  const int r = av1_compute_qdelta_by_rate(s.cpi, (FRAME_TYPE)frame_type,
                                           qindex, rate_target_ratio);
  shim_rca_free(&s);
  return r;
}

int shim_rca_regulate_q(const ShimRcStateParams *p, int target_bits_per_frame,
                        int active_best_quality, int active_worst_quality,
                        int width, int height) {
  ShimRca s;
  if (!shim_rca_alloc(&s, p)) return INT_MIN;
  const int r =
      av1_rc_regulate_q(s.cpi, target_bits_per_frame, active_best_quality,
                        active_worst_quality, width, height);
  shim_rca_free(&s);
  return r;
}

/* out[0] = under-shoot limit, out[1] = over-shoot limit. */
int shim_rca_compute_frame_size_bounds(const ShimRcStateParams *p,
                                       int frame_target, int32_t *out) {
  ShimRca s;
  if (!shim_rca_alloc(&s, p)) return -1;
  int under = 0, over = 0;
  av1_rc_compute_frame_size_bounds(s.cpi, frame_target, &under, &over);
  out[0] = under;
  out[1] = over;
  shim_rca_free(&s);
  return 0;
}

/* out[0] = rc->this_frame_target, out[1] = rc->sb64_target_rate. */
int shim_rca_set_frame_target(const ShimRcStateParams *p, int target, int width,
                              int height, int32_t *out) {
  ShimRca s;
  if (!shim_rca_alloc(&s, p)) return -1;
  av1_rc_set_frame_target(s.cpi, target, width, height);
  out[0] = s.cpi->rc.this_frame_target;
  out[1] = s.cpi->rc.sb64_target_rate;
  shim_rca_free(&s);
  return 0;
}
