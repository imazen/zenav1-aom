/* Oracle shims for av1/encoder/ratectrl.c — the fixed-Q rate controller.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * `nm -g upstream/build/CMakeFiles/aom_av1_encoder.dir/av1/encoder/ratectrl.c.o`
 * reports 31 exported symbols, and none of them is a function on the qindex
 * decision path below `av1_rc_pick_q_and_bounds`. The twelve minq lookup
 * TABLES are file-static too, so even `get_minq_index`'s output is
 * unaddressable from outside. Every one of `get_minq_index`, `init_minq_luts`,
 * `get_active_quality`, `get_kf_active_quality`, `get_gf_active_quality`,
 * `get_gf_high_motion_quality`, `get_active_cq_level`, `get_intra_q_and_bounds`,
 * `get_active_best_quality`, `rc_pick_q_and_bounds_q_mode`,
 * `gf_group_pyramid_level` and `get_default_max_gf_interval` has internal
 * linkage.
 *
 * So this TU compiles libaom's OWN ratectrl.c, unmodified, with its 31
 * exported symbols renamed out of the way, and exposes flat wrappers around
 * the statics. The bodies under test are libaom's source, not a transcription
 * of it — the same technique, and the same justification, as
 * shim/rdopt_shim.c and shim/compound_type_shim.c.
 *
 * EVIDENCE TIER: **tier 1c** — the real C source compiled verbatim, as opposed
 * to tier 1's real exported symbol out of the archive. The gap between the two
 * is that this is a SECOND COMPILATION, which could in principle differ from
 * the archive's through flags. That gap is closed by measurement:
 * `shim_rcc_*` re-exports five of ratectrl.c's real exported functions from
 * THIS TU, and `ratectrl_shim_tu_matches_archive` in tests/ratectrl_q_diff.rs
 * asserts they agree with the archive's `av1_*` symbols over a sweep.
 *
 * FLAGS: build.rs compiles this TU with libaom's own Release flags
 * (-O3 -DNDEBUG plus the oracle-wide -ffp-contract=off), so it is the same
 * source under the same settings as the copy inside libaom.a. -DNDEBUG is
 * separately mandatory for ABI agreement (DIFFERENTIAL_PLAYBOOK §3a(a)) AND
 * for reachability here: rc_pick_q_and_bounds_q_mode ends in three asserts on
 * the qindex bounds, so a Debug build would abort on any cell where the port
 * is being asked what C does outside them.
 *
 * CONTRACT NOTE: the wrappers below calloc an AV1_COMP + AV1_PRIMARY and fill
 * only the fields the function under test reads. `is_one_pass_rt_params` and
 * `has_no_stats_stage` are reached through oxcf.{pass,mode,gf_cfg}, so those
 * three are always set explicitly rather than left zeroed — a zeroed
 * `oxcf.rc_cfg.mode` is AOM_VBR (0), not AOM_Q, and that is a different arm.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename ratectrl.c's 31 exported symbols so this TU links beside
 * libaom.a. The five re-exported below (convert_qindex_to_q, find_qindex,
 * compute_qdelta, rc_bits_per_mb, rc_get_default_min_gf_interval) are the
 * TU-vs-archive agreement probes. */
#define av1_adjust_gf_refresh_qp_one_pass_rt shim_rcc_adjust_gf_refresh_qp_one_pass_rt
#define av1_calc_iframe_target_size_one_pass_cbr shim_rcc_calc_iframe_target_size_one_pass_cbr
#define av1_calc_iframe_target_size_one_pass_vbr shim_rcc_calc_iframe_target_size_one_pass_vbr
#define av1_calc_pframe_target_size_one_pass_cbr shim_rcc_calc_pframe_target_size_one_pass_cbr
#define av1_calc_pframe_target_size_one_pass_vbr shim_rcc_calc_pframe_target_size_one_pass_vbr
#define av1_compute_qdelta shim_rcc_compute_qdelta
#define av1_compute_qdelta_by_rate shim_rcc_compute_qdelta_by_rate
#define av1_convert_q_to_qindex shim_rcc_convert_q_to_qindex
#define av1_convert_qindex_to_q shim_rcc_convert_qindex_to_q
#define av1_encodedframe_overshoot_cbr shim_rcc_encodedframe_overshoot_cbr
#define av1_estimate_bits_at_q shim_rcc_estimate_bits_at_q
#define av1_find_qindex shim_rcc_find_qindex
#define av1_get_one_pass_rt_params shim_rcc_get_one_pass_rt_params
#define av1_postencode_drop_cbr shim_rcc_postencode_drop_cbr
#define av1_primary_rc_init shim_rcc_primary_rc_init
#define av1_rc_bits_per_mb shim_rcc_rc_bits_per_mb
#define av1_rc_compute_frame_size_bounds shim_rcc_rc_compute_frame_size_bounds
#define av1_rc_drop_frame shim_rcc_rc_drop_frame
#define av1_rc_get_default_min_gf_interval shim_rcc_rc_get_default_min_gf_interval
#define av1_rc_init shim_rcc_rc_init
#define av1_rc_init_minq_luts shim_rcc_rc_init_minq_luts
#define av1_rc_pick_q_and_bounds shim_rcc_rc_pick_q_and_bounds
#define av1_rc_postencode_update shim_rcc_rc_postencode_update
#define av1_rc_postencode_update_drop_frame shim_rcc_rc_postencode_update_drop_frame
#define av1_rc_regulate_q shim_rcc_rc_regulate_q
#define av1_rc_scene_detection_onepass_rt shim_rcc_rc_scene_detection_onepass_rt
#define av1_rc_set_frame_target shim_rcc_rc_set_frame_target
#define av1_rc_update_framerate shim_rcc_rc_update_framerate
#define av1_rc_update_rate_correction_factors shim_rcc_rc_update_rate_correction_factors
#define av1_set_rtc_reference_structure_one_layer shim_rcc_set_rtc_reference_structure_one_layer
#define av1_set_target_rate shim_rcc_set_target_rate

/* --- libaom's own rate controller, unmodified. --- */
#include "av1/encoder/ratectrl.c"

#include "rc_state_params.h"

/* ======================================================================== *
 * 0. TU-vs-archive agreement probes (the tier-1c gap closer).
 * ======================================================================== */

double shim_rcc_probe_convert_qindex_to_q(int qindex, int bit_depth) {
  return shim_rcc_convert_qindex_to_q(qindex, (aom_bit_depth_t)bit_depth);
}

int shim_rcc_probe_find_qindex(double desired_q, int bit_depth, int best_qindex,
                               int worst_qindex) {
  return shim_rcc_find_qindex(desired_q, (aom_bit_depth_t)bit_depth,
                              best_qindex, worst_qindex);
}

int shim_rcc_probe_compute_qdelta(double qstart, double qtarget, int bit_depth,
                                  int best_quality, int worst_quality) {
  RATE_CONTROL rc;
  memset(&rc, 0, sizeof(rc));
  rc.best_quality = best_quality;
  rc.worst_quality = worst_quality;
  return shim_rcc_compute_qdelta(&rc, qstart, qtarget,
                                 (aom_bit_depth_t)bit_depth);
}

int shim_rcc_probe_min_gf_interval(int width, int height, double framerate) {
  return shim_rcc_rc_get_default_min_gf_interval(width, height, framerate);
}

/* ======================================================================== *
 * 1. The minq machinery.
 * ======================================================================== */

int shim_rcc_get_minq_index(double maxq, double x3, double x2, double x1v,
                            int bit_depth) {
  return get_minq_index(maxq, x3, x2, x1v, (aom_bit_depth_t)bit_depth);
}

/* Dump one minq lookup table cell. `which` selects the curve:
 * 0 kf_low, 1 kf_high, 2 arfgf_low, 3 arfgf_high, 4 inter, 5 rtc.
 * `mode_idx` is rtc_mode, `res_idx` is C's `res_idx > 1` — the two subscripts
 * ASSIGN_MINQ_TABLE_2 applies, in the order its BODY applies them. */
int shim_rcc_minq_lut(int which, int bit_depth, int mode_idx, int res_idx,
                      int32_t *out, int out_len) {
  if (out_len < QINDEX_RANGE) return -1;
  if (mode_idx < 0 || mode_idx >= MODE_NUM) return -2;
  if (res_idx < 0 || res_idx >= RES_NUM) return -2;
  shim_rcc_rc_init_minq_luts();
  const int *src = NULL;
  switch (bit_depth) {
    case 8:
      switch (which) {
        case 0: src = kf_low_motion_minq_8[mode_idx][res_idx]; break;
        case 1: src = kf_high_motion_minq_8[mode_idx][res_idx]; break;
        case 2: src = arfgf_low_motion_minq_8[mode_idx][res_idx]; break;
        case 3: src = arfgf_high_motion_minq_8[mode_idx][res_idx]; break;
        case 4: src = inter_minq_8[mode_idx][res_idx]; break;
        case 5: src = rtc_minq_8; break;
        default: return -3;
      }
      break;
    case 10:
      switch (which) {
        case 0: src = kf_low_motion_minq_10[mode_idx][res_idx]; break;
        case 1: src = kf_high_motion_minq_10[mode_idx][res_idx]; break;
        case 2: src = arfgf_low_motion_minq_10[mode_idx][res_idx]; break;
        case 3: src = arfgf_high_motion_minq_10[mode_idx][res_idx]; break;
        case 4: src = inter_minq_10[mode_idx][res_idx]; break;
        case 5: src = rtc_minq_10; break;
        default: return -3;
      }
      break;
    case 12:
      switch (which) {
        case 0: src = kf_low_motion_minq_12[mode_idx][res_idx]; break;
        case 1: src = kf_high_motion_minq_12[mode_idx][res_idx]; break;
        case 2: src = arfgf_low_motion_minq_12[mode_idx][res_idx]; break;
        case 3: src = arfgf_high_motion_minq_12[mode_idx][res_idx]; break;
        case 4: src = inter_minq_12[mode_idx][res_idx]; break;
        case 5: src = rtc_minq_12; break;
        default: return -3;
      }
      break;
    default: return -4;
  }
  for (int i = 0; i < QINDEX_RANGE; ++i) out[i] = src[i];
  return QINDEX_RANGE;
}

/* ======================================================================== *
 * 2. The boost-interpolated active-quality curves.
 * ======================================================================== */

int shim_rcc_get_active_quality(int q, int gfu_boost, int low, int high,
                                const int32_t *low_motion,
                                const int32_t *high_motion) {
  /* get_active_quality takes non-const int*; the arrays are only read. */
  return get_active_quality(q, gfu_boost, low, high, (int *)low_motion,
                            (int *)high_motion);
}

int shim_rcc_get_kf_active_quality(int kf_boost, int q, int bit_depth,
                                   int res_idx, int rtc_mode) {
  PRIMARY_RATE_CONTROL p_rc;
  memset(&p_rc, 0, sizeof(p_rc));
  p_rc.kf_boost = kf_boost;
  shim_rcc_rc_init_minq_luts();
  return get_kf_active_quality(&p_rc, q, (aom_bit_depth_t)bit_depth, res_idx,
                               (bool)rtc_mode);
}

int shim_rcc_get_gf_active_quality(int gfu_boost, int gfu_boost_average, int q,
                                   int bit_depth, int res_idx, int rtc_mode) {
  PRIMARY_RATE_CONTROL p_rc;
  memset(&p_rc, 0, sizeof(p_rc));
  p_rc.gfu_boost = gfu_boost;
  p_rc.gfu_boost_average = gfu_boost_average;
  shim_rcc_rc_init_minq_luts();
  return get_gf_active_quality(&p_rc, q, (aom_bit_depth_t)bit_depth, res_idx,
                               (bool)rtc_mode);
}

int shim_rcc_get_gf_high_motion_quality(int q, int bit_depth, int res_idx,
                                        int rtc_mode) {
  shim_rcc_rc_init_minq_luts();
  return get_gf_high_motion_quality(q, (aom_bit_depth_t)bit_depth, res_idx,
                                    (bool)rtc_mode);
}

int shim_rcc_get_default_max_gf_interval(double framerate,
                                         int min_gf_interval) {
  return get_default_max_gf_interval(framerate, min_gf_interval);
}

int shim_rcc_gf_group_pyramid_level(int layer_depth) {
  GF_GROUP gf;
  memset(&gf, 0, sizeof(gf));
  gf.layer_depth[3] = layer_depth;
  return gf_group_pyramid_level(&gf, 3);
}

/* ======================================================================== *
 * 3. The q-mode decision chain.
 * ======================================================================== */

/* Mirrors the Rust `ShimRcQParams`. Every field is exactly one C read site;
 * see the Rust doc comment for the mapping. */
typedef struct {
  int32_t bit_depth;
  int32_t coded_width, coded_height;
  int32_t width, height;
  int32_t rtc_mode;          /* oxcf.mode == REALTIME */
  int32_t screen_content;    /* cpi->is_screen_content_type */
  int32_t superres_mode;     /* cpi->superres_mode */
  int32_t superres_denom;    /* cm->superres_scale_denominator */
  int32_t large_scale;       /* cm->tiles.large_scale */
  int32_t refresh_golden;
  int32_t refresh_alt_ref;
  int32_t rc_mode;           /* oxcf.rc_cfg.mode */
  int32_t cq_level;          /* oxcf.rc_cfg.cq_level, already a qindex */
  int32_t intra_only;        /* frame_is_intra_only(cm) */
  int32_t active_worst_in;   /* rc->active_worst_quality */
  int32_t update_type;
  int32_t layer_depth;
  int32_t kf_boost;
  int32_t gfu_boost;
  int32_t gfu_boost_average;
  float arf_boost_factor; /* p_rc->arf_boost_factor is float_t == float */
  int32_t arf_q;
  int32_t avg_frame_qindex_inter;
  int32_t this_key_frame_forced;
  int32_t last_boosted_qindex;
  int32_t last_kf_qindex;
  int32_t frames_to_key;
  int32_t frames_since_key;
  int32_t best_quality;
  int32_t worst_quality;
  int32_t kf_zeromotion_pct;
  int32_t last_kfgroup_zeromotion_pct;
  int32_t two_pass;          /* is_stat_consumption_stage_twopass */
  int64_t total_actual_bits;
  int64_t total_target_bits;
} ShimRcQParams;

/* The gf group slot the wrappers use. */
#define SHIM_RC_GF_INDEX 2

typedef struct {
  AV1_COMP *cpi;
  AV1_PRIMARY *ppi;
  SequenceHeader *seq;
} ShimRc;

static int shim_rc_alloc(ShimRc *s, const ShimRcQParams *p) {
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
  /* av1_frame_scaled(cm) is superres_scaled || resize_scaled; both compare the
   * upscaled/render sizes against the coded ones, so setting them equal makes
   * the frame unscaled — which is what the q-mode path is being asked about. */
  cpi->common.superres_upscaled_width = p->coded_width;
  cpi->common.superres_upscaled_height = p->coded_height;
  cpi->common.render_width = p->coded_width;
  cpi->common.render_height = p->coded_height;
  cpi->common.superres_scale_denominator = (uint8_t)p->superres_denom;
  cpi->common.tiles.large_scale = p->large_scale;
  cpi->common.current_frame.frame_type =
      p->intra_only ? KEY_FRAME : INTER_FRAME;

  cpi->superres_mode = (aom_superres_mode)p->superres_mode;
  cpi->is_screen_content_type = p->screen_content;
  cpi->refresh_frame.golden_frame = (bool)p->refresh_golden;
  cpi->refresh_frame.alt_ref_frame = (bool)p->refresh_alt_ref;

  cpi->oxcf.mode = p->rtc_mode ? REALTIME : GOOD;
  cpi->oxcf.pass = p->two_pass ? AOM_RC_SECOND_PASS : AOM_RC_ONE_PASS;
  cpi->oxcf.rc_cfg.mode = (enum aom_rc_mode)p->rc_mode;
  cpi->oxcf.rc_cfg.cq_level = p->cq_level;

  cpi->rc.active_worst_quality = p->active_worst_in;
  cpi->rc.best_quality = p->best_quality;
  cpi->rc.worst_quality = p->worst_quality;
  cpi->rc.frames_to_key = p->frames_to_key;
  cpi->rc.frames_since_key = p->frames_since_key;

  cpi->ppi->p_rc.kf_boost = p->kf_boost;
  cpi->ppi->p_rc.gfu_boost = p->gfu_boost;
  cpi->ppi->p_rc.gfu_boost_average = p->gfu_boost_average;
  cpi->ppi->p_rc.arf_boost_factor = p->arf_boost_factor;
  cpi->ppi->p_rc.arf_q = p->arf_q;
  cpi->ppi->p_rc.avg_frame_qindex[INTER_FRAME] = p->avg_frame_qindex_inter;
  cpi->ppi->p_rc.this_key_frame_forced = p->this_key_frame_forced;
  cpi->ppi->p_rc.last_boosted_qindex = p->last_boosted_qindex;
  cpi->ppi->p_rc.last_kf_qindex = p->last_kf_qindex;
  cpi->ppi->p_rc.total_actual_bits = p->total_actual_bits;
  cpi->ppi->p_rc.total_target_bits = p->total_target_bits;

  cpi->ppi->twopass.kf_zeromotion_pct = p->kf_zeromotion_pct;
  cpi->ppi->twopass.last_kfgroup_zeromotion_pct =
      p->last_kfgroup_zeromotion_pct;

  cpi->gf_frame_index = SHIM_RC_GF_INDEX;
  cpi->ppi->gf_group.update_type[SHIM_RC_GF_INDEX] =
      (FRAME_UPDATE_TYPE)p->update_type;
  cpi->ppi->gf_group.layer_depth[SHIM_RC_GF_INDEX] = p->layer_depth;
  cpi->ppi->gf_group.frame_type[SHIM_RC_GF_INDEX] =
      p->intra_only ? KEY_FRAME : INTER_FRAME;

  shim_rcc_rc_init_minq_luts();
  return 1;
}

static void shim_rc_free(ShimRc *s) {
  free(s->cpi); free(s->ppi); free(s->seq);
}

int shim_rcc_get_active_cq_level(const ShimRcQParams *p) {
  ShimRc s;
  if (!shim_rc_alloc(&s, p)) return INT_MIN;
  const int r = get_active_cq_level(
      &s.cpi->rc, &s.cpi->ppi->p_rc, &s.cpi->oxcf, p->intra_only,
      (aom_superres_mode)p->superres_mode, p->superres_denom);
  shim_rc_free(&s);
  return r;
}

/* out[0] = active_best, out[1] = active_worst. */
int shim_rcc_get_intra_q_and_bounds(const ShimRcQParams *p, int cq_level,
                                    int active_worst_in, int32_t *out) {
  ShimRc s;
  if (!shim_rc_alloc(&s, p)) return -1;
  int active_best = 0;
  int active_worst = active_worst_in;
  get_intra_q_and_bounds(s.cpi, p->width, p->height, &active_best,
                         &active_worst, cq_level);
  out[0] = active_best;
  out[1] = active_worst;
  shim_rc_free(&s);
  return 0;
}

int shim_rcc_get_active_best_quality(const ShimRcQParams *p,
                                     int active_worst_quality, int cq_level) {
  ShimRc s;
  if (!shim_rc_alloc(&s, p)) return INT_MIN;
  const int r = get_active_best_quality(s.cpi, active_worst_quality, cq_level,
                                        SHIM_RC_GF_INDEX);
  shim_rc_free(&s);
  return r;
}

/* out[0] = q, out[1] = bottom_index, out[2] = top_index. */
int shim_rcc_pick_q_and_bounds_q_mode(const ShimRcQParams *p, int32_t *out) {
  ShimRc s;
  if (!shim_rc_alloc(&s, p)) return -1;
  int bottom = 0, top = 0;
  const int q = rc_pick_q_and_bounds_q_mode(s.cpi, p->width, p->height,
                                            SHIM_RC_GF_INDEX, &bottom, &top);
  out[0] = q;
  out[1] = bottom;
  out[2] = top;
  shim_rc_free(&s);
  return 0;
}


/* ======================================================================== *
 * 4. The rate-search statics (get_rate_correction_factor and friends).
 *
 * The AV1_COMP these build shares `rc_state_params.h` with rcarchive_shim.c,
 * which drives the same file's EXPORTED functions out of the archive — so the
 * two TUs are set up identically and a difference between them is a
 * difference between the compilations.
 * ======================================================================== */

#define SHIM_RCS_GF_INDEX 2

static int shim_rcs_alloc(ShimRc *s, const ShimRcStateParams *p) {
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

  cpi->gf_frame_index = SHIM_RCS_GF_INDEX;
  cpi->ppi->gf_group.update_type[SHIM_RCS_GF_INDEX] =
      (FRAME_UPDATE_TYPE)p->update_type;
  cpi->ppi->gf_group.layer_depth[SHIM_RCS_GF_INDEX] = p->layer_depth;
  cpi->ppi->gf_group.frame_type[SHIM_RCS_GF_INDEX] = (FRAME_TYPE)p->frame_type;
  cpi->ppi->gf_group.frame_parallel_level[SHIM_RCS_GF_INDEX] = 0;

  shim_rcc_rc_init_minq_luts();
  return 1;
}

double shim_rcc_resize_rate_factor(int cfg_width, int cfg_height, int width,
                                   int height) {
  FrameDimensionCfg cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.width = cfg_width;
  cfg.height = cfg_height;
  return resize_rate_factor(&cfg, width, height);
}

int shim_rcc_get_rate_factor_level(int update_type) {
  GF_GROUP gf;
  memset(&gf, 0, sizeof(gf));
  gf.update_type[SHIM_RCS_GF_INDEX] = (FRAME_UPDATE_TYPE)update_type;
  return (int)get_rate_factor_level(&gf, SHIM_RCS_GF_INDEX);
}

double shim_rcc_get_rate_correction_factor(const ShimRcStateParams *p,
                                           int width, int height) {
  ShimRc s;
  if (!shim_rcs_alloc(&s, p)) return -1.0;
  const double r = get_rate_correction_factor(s.cpi, width, height);
  shim_rc_free(&s);
  return r;
}

int shim_rcc_get_bits_per_mb(const ShimRcStateParams *p,
                             double correction_factor, int q) {
  ShimRc s;
  if (!shim_rcs_alloc(&s, p)) return INT_MIN;
  const int r = get_bits_per_mb(s.cpi, /*use_cyclic_refresh=*/0,
                                correction_factor, q);
  shim_rc_free(&s);
  return r;
}

int shim_rcc_find_qindex_by_rate(const ShimRcStateParams *p,
                                 int desired_bits_per_mb, int frame_type,
                                 int best_qindex, int worst_qindex) {
  ShimRc s;
  if (!shim_rcs_alloc(&s, p)) return INT_MIN;
  const int r = find_qindex_by_rate(s.cpi, desired_bits_per_mb,
                                    (FRAME_TYPE)frame_type, best_qindex,
                                    worst_qindex);
  shim_rc_free(&s);
  return r;
}

int shim_rcc_find_closest_qindex_by_rate(const ShimRcStateParams *p,
                                         int desired_bits_per_mb,
                                         double correction_factor,
                                         int best_qindex, int worst_qindex) {
  ShimRc s;
  if (!shim_rcs_alloc(&s, p)) return INT_MIN;
  const int r = find_closest_qindex_by_rate(desired_bits_per_mb, s.cpi,
                                            correction_factor, best_qindex,
                                            worst_qindex);
  shim_rc_free(&s);
  return r;
}

int shim_rcc_frame_type_qdelta(const ShimRcStateParams *p, int q) {
  ShimRc s;
  if (!shim_rcs_alloc(&s, p)) return INT_MIN;
  const int r = frame_type_qdelta(s.cpi, q);
  shim_rc_free(&s);
  return r;
}

/* The TU-vs-archive probes for the EXPORTED rate-search functions: the same
 * four rcarchive_shim.c drives out of the archive, driven here out of this
 * TU's copy. */
int shim_rcc_probe_estimate_bits_at_q(const ShimRcStateParams *p, int q,
                                      double correction_factor) {
  ShimRc s;
  if (!shim_rcs_alloc(&s, p)) return INT_MIN;
  const int r = shim_rcc_estimate_bits_at_q(s.cpi, q, correction_factor);
  shim_rc_free(&s);
  return r;
}

int shim_rcc_probe_regulate_q(const ShimRcStateParams *p,
                              int target_bits_per_frame,
                              int active_best_quality, int active_worst_quality,
                              int width, int height) {
  ShimRc s;
  if (!shim_rcs_alloc(&s, p)) return INT_MIN;
  const int r = shim_rcc_rc_regulate_q(s.cpi, target_bits_per_frame,
                                       active_best_quality,
                                       active_worst_quality, width, height);
  shim_rc_free(&s);
  return r;
}
