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

/* ======================================================================== *
 * av1_primary_rc_init / av1_rc_init / av1_rc_update_framerate, out of the
 * archive. Each builds the AV1EncoderConfig from ShimRcInitCfg and copies the
 * written fields out; nothing else in the struct is read, so a field the port
 * forgot shows up as a mismatch rather than as a silent zero.
 * ======================================================================== */

static void shim_rca_fill_oxcf(AV1EncoderConfig *oxcf,
                               const ShimRcInitCfg *c) {
  memset(oxcf, 0, sizeof(*oxcf));
  oxcf->rc_cfg.mode = (enum aom_rc_mode)c->rc_mode;
  oxcf->rc_cfg.best_allowed_q = c->best_allowed_q;
  oxcf->rc_cfg.worst_allowed_q = c->worst_allowed_q;
  oxcf->rc_cfg.target_bandwidth = c->target_bandwidth;
  oxcf->rc_cfg.vbrmin_section = c->vbrmin_section;
  oxcf->rc_cfg.vbrmax_section = c->vbrmax_section;
  oxcf->gf_cfg.min_gf_interval = c->min_gf_interval;
  oxcf->gf_cfg.max_gf_interval = c->max_gf_interval;
  oxcf->kf_cfg.fwd_kf_dist = c->fwd_kf_dist;
  oxcf->frm_dim_cfg.width = c->width;
  oxcf->frm_dim_cfg.height = c->height;
  oxcf->input_cfg.init_framerate = c->init_framerate;
  oxcf->tool_cfg.bit_depth = (aom_bit_depth_t)c->bit_depth;
  oxcf->pass = c->one_pass ? AOM_RC_ONE_PASS : AOM_RC_SECOND_PASS;
  oxcf->target_seq_level_idx[0] = (AV1_LEVEL)c->target_seq_level_idx0;
}

/* out_i: 0 baseline_gf_interval, 1 this_key_frame_forced,
 * 2 next_key_frame_forced, 3 ni_frames, 4 avg_frame_qindex[KEY],
 * 5 avg_frame_qindex[INTER], 6 last_q[KEY], 7 last_q[INTER],
 * 8 rolling_target_bits, 9 rolling_actual_bits.
 * out_d: 0 tot_q, 1 avg_q, 2..5 rate_correction_factors.
 * out_l: 0 total_actual_bits, 1 total_target_bits, 2 buffer_level,
 *        3 bits_off_target. */
int shim_rca_primary_rc_init(const ShimRcInitCfg *c, int32_t *out_i,
                             double *out_d, int64_t *out_l) {
  AV1EncoderConfig oxcf;
  shim_rca_fill_oxcf(&oxcf, c);
  PRIMARY_RATE_CONTROL *p_rc =
      (PRIMARY_RATE_CONTROL *)calloc(1, sizeof(PRIMARY_RATE_CONTROL));
  if (!p_rc) return -1;
  p_rc->starting_buffer_level = c->starting_buffer_level;
  av1_primary_rc_init(&oxcf, p_rc);
  out_i[0] = p_rc->baseline_gf_interval;
  out_i[1] = p_rc->this_key_frame_forced;
  out_i[2] = p_rc->next_key_frame_forced;
  out_i[3] = p_rc->ni_frames;
  out_i[4] = p_rc->avg_frame_qindex[KEY_FRAME];
  out_i[5] = p_rc->avg_frame_qindex[INTER_FRAME];
  out_i[6] = p_rc->last_q[KEY_FRAME];
  out_i[7] = p_rc->last_q[INTER_FRAME];
  out_i[8] = p_rc->rolling_target_bits;
  out_i[9] = p_rc->rolling_actual_bits;
  out_d[0] = p_rc->tot_q;
  out_d[1] = p_rc->avg_q;
  for (int i = 0; i < RATE_FACTOR_LEVELS; ++i)
    out_d[2 + i] = p_rc->rate_correction_factors[i];
  out_l[0] = p_rc->total_actual_bits;
  out_l[1] = p_rc->total_target_bits;
  out_l[2] = p_rc->buffer_level;
  out_l[3] = p_rc->bits_off_target;
  free(p_rc);
  return 0;
}

/* out: 0 frames_since_key, 1 frames_to_fwd_kf, 2 frames_till_gf_update_due,
 * 3 ni_av_qi, 4 ni_tot_qi, 5 min_gf_interval, 6 max_gf_interval,
 * 7 avg_frame_low_motion, 8 resize_avg_qp, 9 resize_buffer_underflow,
 * 10 resize_count, 11 frames_since_scene_change, then the fields the port
 * does NOT return, so the test can assert C leaves them zero:
 * 12 resize_state, 13 rtc_external_ratectrl, 14 frame_level_fast_extra_bits,
 * 15 use_external_qp_one_pass, 16 percent_blocks_inactive, 17 force_max_q,
 * 18 postencode_drop, 19 last_frame_low_source_sad. */
int shim_rca_rc_init(const ShimRcInitCfg *c, int32_t *out) {
  AV1EncoderConfig oxcf;
  shim_rca_fill_oxcf(&oxcf, c);
  RATE_CONTROL *rc = (RATE_CONTROL *)calloc(1, sizeof(RATE_CONTROL));
  if (!rc) return -1;
  /* Pre-poison every field the port claims C zeroes, so "C left it zero" is a
   * measurement rather than an artefact of the calloc. */
  rc->resize_state = 1;
  rc->rtc_external_ratectrl = 1;
  rc->frame_level_fast_extra_bits = 1;
  rc->use_external_qp_one_pass = 1;
  rc->percent_blocks_inactive = 1;
  rc->force_max_q = 1;
  rc->postencode_drop = 1;
  rc->last_frame_low_source_sad = 1;
  av1_rc_init(&oxcf, rc);
  out[0] = rc->frames_since_key;
  out[1] = rc->frames_to_fwd_kf;
  out[2] = rc->frames_till_gf_update_due;
  out[3] = rc->ni_av_qi;
  out[4] = rc->ni_tot_qi;
  out[5] = rc->min_gf_interval;
  out[6] = rc->max_gf_interval;
  out[7] = rc->avg_frame_low_motion;
  out[8] = rc->resize_avg_qp;
  out[9] = rc->resize_buffer_underflow;
  out[10] = rc->resize_count;
  out[11] = rc->frames_since_scene_change;
  out[12] = (int32_t)rc->resize_state;
  out[13] = rc->rtc_external_ratectrl;
  out[14] = rc->frame_level_fast_extra_bits;
  out[15] = rc->use_external_qp_one_pass;
  out[16] = rc->percent_blocks_inactive;
  out[17] = rc->force_max_q;
  out[18] = rc->postencode_drop;
  out[19] = rc->last_frame_low_source_sad;
  free(rc);
  return 0;
}

/* out: 0 avg_frame_bandwidth, 1 min_frame_bandwidth, 2 max_frame_bandwidth,
 * 3 min_gf_interval, 4 max_gf_interval, 5 static_scene_max_gf_interval. */
int shim_rca_update_framerate(const ShimRcInitCfg *c, int width, int height,
                              int32_t *out) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(AV1_COMP));
  AV1_PRIMARY *ppi = (AV1_PRIMARY *)calloc(1, sizeof(AV1_PRIMARY));
  if (!cpi || !ppi) {
    free(cpi); free(ppi);
    return -1;
  }
  cpi->ppi = ppi;
  shim_rca_fill_oxcf(&cpi->oxcf, c);
  cpi->framerate = c->framerate;
  ppi->lap_enabled = c->lap_enabled;
  av1_rc_update_framerate(cpi, width, height);
  out[0] = cpi->rc.avg_frame_bandwidth;
  out[1] = cpi->rc.min_frame_bandwidth;
  out[2] = cpi->rc.max_frame_bandwidth;
  out[3] = cpi->rc.min_gf_interval;
  out[4] = cpi->rc.max_gf_interval;
  out[5] = cpi->rc.static_scene_max_gf_interval;
  free(cpi); free(ppi);
  return 0;
}

/* ======================================================================== *
 * av1_rc_postencode_update and av1_rc_update_rate_correction_factors, out of
 * the archive. Both mutate a real AV1_COMP; the shim copies ShimRcUpdateState
 * in, runs the function, and copies the mutated fields back over the same
 * struct, so the caller compares the whole post-state at once.
 * ======================================================================== */

static int shim_rcu_build(ShimRca *s, const ShimRcUpdateState *u) {
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
  s->seq->bit_depth = (aom_bit_depth_t)u->bit_depth;
  cpi->common.seq_params = s->seq;

  cpi->common.width = u->coded_width;
  cpi->common.height = u->coded_height;
  cpi->common.superres_upscaled_width = u->coded_width;
  cpi->common.superres_upscaled_height = u->coded_height;
  cpi->common.render_width = u->coded_width;
  cpi->common.render_height = u->coded_height;
  cpi->common.superres_scale_denominator = SCALE_NUMERATOR;
  cpi->common.show_frame = u->show_frame;
  cpi->common.current_frame.frame_type = (FRAME_TYPE)u->frame_type;
  cpi->common.current_frame.frame_number = (unsigned int)u->frame_number;
  cpi->common.quant_params.base_qindex = u->base_qindex;
  cpi->common.mi_params.MBs = av1_get_MBs(u->coded_width, u->coded_height);

  cpi->is_screen_content_type = u->screen_content;
  cpi->refresh_frame.golden_frame = (bool)u->refresh_golden;
  cpi->refresh_frame.alt_ref_frame = (bool)u->refresh_alt_ref;

  cpi->oxcf.mode = GOOD;
  cpi->oxcf.pass = u->stat_consumption ? AOM_RC_SECOND_PASS : AOM_RC_ONE_PASS;
  cpi->oxcf.rc_cfg.mode = (enum aom_rc_mode)u->rc_mode;
  cpi->oxcf.rc_cfg.gf_cbr_boost_pct = u->gf_cbr_boost_pct;
  cpi->oxcf.frm_dim_cfg.width = u->cfg_width;
  cpi->oxcf.frm_dim_cfg.height = u->cfg_height;
  cpi->oxcf.gf_cfg.lag_in_frames = u->lag_in_frames;
  cpi->oxcf.gf_cfg.enable_auto_arf = (bool)u->enable_auto_arf;
  cpi->oxcf.tune_cfg.content =
      u->tune_content_screen ? AOM_CONTENT_SCREEN : AOM_CONTENT_DEFAULT;
  cpi->oxcf.q_cfg.aq_mode = NO_AQ;
  cpi->sf.hl_sf.accurate_bit_estimate = 0;

  cpi->gf_frame_index = SHIM_RCA_GF_INDEX;
  cpi->ppi->gf_group.update_type[SHIM_RCA_GF_INDEX] =
      (FRAME_UPDATE_TYPE)u->update_type;
  cpi->ppi->gf_group.frame_type[SHIM_RCA_GF_INDEX] = (FRAME_TYPE)u->frame_type;
  cpi->ppi->gf_group.frame_parallel_level[SHIM_RCA_GF_INDEX] = 0;

  RATE_CONTROL *rc = &cpi->rc;
  rc->projected_frame_size = u->projected_frame_size;
  rc->q_1_frame = u->q_1_frame;
  rc->q_2_frame = u->q_2_frame;
  rc->rc_1_frame = u->rc_1_frame;
  rc->rc_2_frame = u->rc_2_frame;
  rc->this_frame_target = u->this_frame_target;
  rc->avg_frame_bandwidth = u->avg_frame_bandwidth;
  rc->prev_avg_frame_bandwidth = u->prev_avg_frame_bandwidth;
  rc->frames_since_key = u->frames_since_key;
  rc->frames_since_golden = u->frames_since_golden;
  rc->frame_num_last_gf_refresh = u->frame_num_last_gf_refresh;
  rc->frame_source_sad = (uint64_t)u->frame_source_sad;
  rc->last_frame_low_source_sad = (unsigned int)u->last_frame_low_source_sad;
  rc->frame_number_encoded = (unsigned int)u->frame_number_encoded;
  rc->prev_coded_width = u->prev_coded_width;
  rc->prev_coded_height = u->prev_coded_height;
  rc->prev_frame_is_dropped = u->prev_frame_is_dropped;
  rc->drop_count_consec = u->drop_count_consec;
  rc->ni_tot_qi = u->ni_tot_qi;
  rc->ni_av_qi = u->ni_av_qi;
  rc->is_src_frame_alt_ref = u->is_src_frame_alt_ref;
  rc->last_encoded_size_keyframe = u->last_encoded_size_keyframe;
  rc->last_target_size_keyframe = u->last_target_size_keyframe;
  rc->rtc_external_ratectrl = u->rtc_external_ratectrl;
  rc->frames_since_scene_change = u->frames_since_scene_change;

  PRIMARY_RATE_CONTROL *p_rc = &cpi->ppi->p_rc;
  p_rc->last_q[KEY_FRAME] = u->last_q_key;
  p_rc->last_q[INTER_FRAME] = u->last_q_inter;
  p_rc->avg_frame_qindex[KEY_FRAME] = u->avg_frame_qindex_key;
  p_rc->avg_frame_qindex[INTER_FRAME] = u->avg_frame_qindex_inter;
  p_rc->ni_frames = u->ni_frames;
  p_rc->tot_q = u->tot_q;
  p_rc->avg_q = u->avg_q;
  p_rc->last_boosted_qindex = u->last_boosted_qindex;
  p_rc->last_kf_qindex = u->last_kf_qindex;
  for (int i = 0; i < RATE_FACTOR_LEVELS; ++i)
    p_rc->rate_correction_factors[i] = u->rate_correction_factors[i];
  p_rc->bits_off_target = u->bits_off_target;
  p_rc->buffer_level = u->buffer_level;
  p_rc->maximum_buffer_size = u->maximum_buffer_size;
  p_rc->total_actual_bits = u->total_actual_bits;
  p_rc->total_target_bits = u->total_target_bits;
  p_rc->rolling_target_bits = u->rolling_target_bits;
  p_rc->rolling_actual_bits = u->rolling_actual_bits;
  p_rc->constrained_gf_group = u->constrained_gf_group;

  av1_rc_init_minq_luts();
  return 1;
}

static void shim_rcu_read_back(const ShimRca *s, ShimRcUpdateState *u) {
  const RATE_CONTROL *rc = &s->cpi->rc;
  const PRIMARY_RATE_CONTROL *p_rc = &s->cpi->ppi->p_rc;
  u->projected_frame_size = rc->projected_frame_size;
  u->q_1_frame = rc->q_1_frame;
  u->q_2_frame = rc->q_2_frame;
  u->rc_1_frame = rc->rc_1_frame;
  u->rc_2_frame = rc->rc_2_frame;
  u->this_frame_target = rc->this_frame_target;
  u->prev_avg_frame_bandwidth = rc->prev_avg_frame_bandwidth;
  u->frames_since_key = rc->frames_since_key;
  u->frames_since_golden = rc->frames_since_golden;
  u->frame_num_last_gf_refresh = rc->frame_num_last_gf_refresh;
  u->last_frame_low_source_sad = (int32_t)rc->last_frame_low_source_sad;
  u->frame_number_encoded = (int32_t)rc->frame_number_encoded;
  u->prev_coded_width = rc->prev_coded_width;
  u->prev_coded_height = rc->prev_coded_height;
  u->prev_frame_is_dropped = rc->prev_frame_is_dropped;
  u->drop_count_consec = rc->drop_count_consec;
  u->ni_tot_qi = rc->ni_tot_qi;
  u->ni_av_qi = rc->ni_av_qi;
  u->last_encoded_size_keyframe = rc->last_encoded_size_keyframe;
  u->last_target_size_keyframe = rc->last_target_size_keyframe;
  u->frames_since_scene_change = rc->frames_since_scene_change;
  u->last_q_key = p_rc->last_q[KEY_FRAME];
  u->last_q_inter = p_rc->last_q[INTER_FRAME];
  u->avg_frame_qindex_key = p_rc->avg_frame_qindex[KEY_FRAME];
  u->avg_frame_qindex_inter = p_rc->avg_frame_qindex[INTER_FRAME];
  u->ni_frames = p_rc->ni_frames;
  u->tot_q = p_rc->tot_q;
  u->avg_q = p_rc->avg_q;
  u->last_boosted_qindex = p_rc->last_boosted_qindex;
  u->last_kf_qindex = p_rc->last_kf_qindex;
  for (int i = 0; i < RATE_FACTOR_LEVELS; ++i)
    u->rate_correction_factors[i] = p_rc->rate_correction_factors[i];
  u->bits_off_target = p_rc->bits_off_target;
  u->buffer_level = p_rc->buffer_level;
  u->total_actual_bits = p_rc->total_actual_bits;
  u->total_target_bits = p_rc->total_target_bits;
  u->rolling_target_bits = p_rc->rolling_target_bits;
  u->rolling_actual_bits = p_rc->rolling_actual_bits;
}

int shim_rca_update_rate_correction_factors(ShimRcUpdateState *u) {
  ShimRca s;
  if (!shim_rcu_build(&s, u)) return -1;
  av1_rc_update_rate_correction_factors(s.cpi, u->is_encode_stage,
                                        u->coded_width, u->coded_height);
  shim_rcu_read_back(&s, u);
  shim_rca_free(&s);
  return 0;
}

int shim_rca_postencode_update(ShimRcUpdateState *u) {
  ShimRca s;
  if (!shim_rcu_build(&s, u)) return -1;
  av1_rc_postencode_update(s.cpi, (uint64_t)u->bytes_used);
  shim_rcu_read_back(&s, u);
  shim_rca_free(&s);
  return 0;
}

/* av1_rc_pick_q_and_bounds out of the ARCHIVE (tier 1). Shares
 * ShimRcPickParams with ratectrl_shim.c, so the two builds are set up
 * identically and a difference between them is a difference between the
 * compilations. Duplicated setup rather than shared, because this TU must not
 * pull ratectrl.c in — that is the whole point of the file.
 * out[0] = q, out[1] = bottom, out[2] = top, out[3] = p_rc->arf_q after. */
int shim_rca_rc_pick_q_and_bounds(const ShimRcPickParams *p, int32_t *out) {
  AV1_COMP *cpi = (AV1_COMP *)calloc(1, sizeof(AV1_COMP));
  AV1_PRIMARY *ppi = (AV1_PRIMARY *)calloc(1, sizeof(AV1_PRIMARY));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(SequenceHeader));
  if (!cpi || !ppi || !seq) {
    free(cpi); free(ppi); free(seq);
    return -1;
  }
  cpi->ppi = ppi;
  seq->bit_depth = (aom_bit_depth_t)p->bit_depth;
  cpi->common.seq_params = seq;

  cpi->common.width = p->coded_width;
  cpi->common.height = p->coded_height;
  cpi->common.superres_upscaled_width = p->coded_width;
  cpi->common.superres_upscaled_height = p->coded_height;
  cpi->common.render_width = p->coded_width;
  cpi->common.render_height = p->coded_height;
  cpi->common.superres_scale_denominator = (uint8_t)p->superres_denom;
  cpi->common.tiles.large_scale = p->large_scale;
  cpi->common.current_frame.frame_type = (FRAME_TYPE)p->frame_type;
  cpi->common.current_frame.frame_number = (unsigned int)p->frame_number;
  cpi->common.mi_params.MBs = av1_get_MBs(p->coded_width, p->coded_height);

  cpi->superres_mode = (aom_superres_mode)p->superres_mode;
  cpi->is_screen_content_type = p->screen_content;
  cpi->refresh_frame.golden_frame = (bool)p->refresh_golden;
  cpi->refresh_frame.bwd_ref_frame = (bool)p->refresh_bwd_ref;
  cpi->refresh_frame.alt_ref_frame = (bool)p->refresh_alt_ref;

  cpi->oxcf.mode = p->rtc_mode ? REALTIME : GOOD;
  cpi->oxcf.pass = p->two_pass ? AOM_RC_SECOND_PASS : AOM_RC_ONE_PASS;
  ppi->lap_enabled = (!p->has_no_stats_stage && !p->two_pass) ? 1 : 0;
  cpi->oxcf.rc_cfg.mode = (enum aom_rc_mode)p->rc_mode;
  cpi->oxcf.rc_cfg.cq_level = p->cq_level;
  cpi->oxcf.frm_dim_cfg.width = p->cfg_width;
  cpi->oxcf.frm_dim_cfg.height = p->cfg_height;
  cpi->oxcf.q_cfg.aq_mode = NO_AQ;
  cpi->sf.hl_sf.accurate_bit_estimate = 0;
  cpi->sf.hl_sf.recode_tolerance = 25;

  cpi->rc.active_worst_quality = p->active_worst_quality;
  cpi->rc.best_quality = p->best_quality;
  cpi->rc.worst_quality = p->worst_quality;
  cpi->rc.frames_to_key = p->frames_to_key;
  cpi->rc.frames_since_key = p->frames_since_key;
  cpi->rc.is_src_frame_alt_ref = p->is_src_frame_alt_ref;
  cpi->rc.this_frame_target = p->this_frame_target;
  cpi->rc.max_frame_bandwidth = p->max_frame_bandwidth;

  PRIMARY_RATE_CONTROL *p_rc = &ppi->p_rc;
  p_rc->kf_boost = p->kf_boost;
  p_rc->gfu_boost = p->gfu_boost;
  p_rc->gfu_boost_average = p->gfu_boost_average;
  p_rc->arf_boost_factor = p->arf_boost_factor;
  p_rc->arf_q = p->arf_q;
  p_rc->avg_frame_qindex[KEY_FRAME] = p->avg_frame_qindex_key;
  p_rc->avg_frame_qindex[INTER_FRAME] = p->avg_frame_qindex_inter;
  p_rc->this_key_frame_forced = p->this_key_frame_forced;
  p_rc->last_boosted_qindex = p->last_boosted_qindex;
  p_rc->last_kf_qindex = p->last_kf_qindex;
  p_rc->last_q[KEY_FRAME] = p->last_q_key;
  p_rc->last_q[INTER_FRAME] = p->last_q_inter;
  for (int i = 0; i < MAX_ARF_LAYERS; ++i)
    p_rc->active_best_quality[i] = p->active_best_quality_by_layer[i];
  p_rc->total_actual_bits = p->total_actual_bits;
  p_rc->total_target_bits = p->total_target_bits;
  for (int i = 0; i < RATE_FACTOR_LEVELS; ++i)
    p_rc->rate_correction_factors[i] = p->rate_correction_factors[i];

  ppi->twopass.kf_zeromotion_pct = p->kf_zeromotion_pct;
  ppi->twopass.last_kfgroup_zeromotion_pct = p->last_kfgroup_zeromotion_pct;
  ppi->twopass.extend_minq = p->extend_minq;
  ppi->twopass.extend_maxq = p->extend_maxq;

  cpi->gf_frame_index = SHIM_RCA_GF_INDEX;
  ppi->gf_group.update_type[SHIM_RCA_GF_INDEX] =
      (FRAME_UPDATE_TYPE)p->update_type;
  ppi->gf_group.layer_depth[SHIM_RCA_GF_INDEX] = p->layer_depth;
  ppi->gf_group.frame_type[SHIM_RCA_GF_INDEX] =
      (FRAME_TYPE)p->gf_index_frame_type;
  ppi->gf_group.frame_parallel_level[SHIM_RCA_GF_INDEX] = 0;

  av1_rc_init_minq_luts();

  int bottom = 0, top = 0;
  out[0] = av1_rc_pick_q_and_bounds(cpi, p->width, p->height,
                                    SHIM_RCA_GF_INDEX, &bottom, &top);
  out[1] = bottom;
  out[2] = top;
  out[3] = p_rc->arf_q;
  free(cpi); free(ppi); free(seq);
  return 0;
}
