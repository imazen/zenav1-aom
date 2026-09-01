/* Shared parameter block for the two ratectrl oracle shims.
 *
 *   ratectrl_shim.c   — includes libaom's ratectrl.c (tier 1c), reaching the
 *                       file-static functions.
 *   rcarchive_shim.c  — includes NOTHING of ratectrl.c, so the `av1_*` names
 *                       it calls bind to the ARCHIVE's copies (tier 1).
 *
 * Both build the same AV1_COMP out of this struct, so a divergence between
 * them is a divergence between the two compilations, not between two setups.
 * Keep the field order in lockstep with Rust's `RefRcStateParams`.
 */
#ifndef AOM_RS_SHIM_RC_STATE_PARAMS_H_
#define AOM_RS_SHIM_RC_STATE_PARAMS_H_

#include <stdint.h>

typedef struct {
  int32_t bit_depth;
  int32_t coded_width, coded_height;   /* cm->width / cm->height */
  int32_t cfg_width, cfg_height;       /* oxcf.frm_dim_cfg.width / .height */
  int32_t frame_type;                  /* cm->current_frame.frame_type */
  int32_t screen_content;              /* cpi->is_screen_content_type */
  int32_t rc_mode;                     /* oxcf.rc_cfg.mode */
  int32_t rtc_mode;                    /* oxcf.mode == REALTIME */
  int32_t stat_consumption;            /* is_stat_consumption_stage(cpi) */
  int32_t refresh_golden, refresh_alt_ref;
  int32_t is_src_frame_alt_ref;        /* rc->is_src_frame_alt_ref */
  int32_t gf_cbr_boost_pct;            /* oxcf.rc_cfg.gf_cbr_boost_pct */
  int32_t update_type;                 /* gf_group->update_type[gf_index] */
  int32_t layer_depth;                 /* gf_group->layer_depth[gf_index] */
  int32_t best_quality, worst_quality; /* rc->best_quality / worst_quality */
  int32_t base_qindex;                 /* cm->quant_params.base_qindex */
  int32_t max_frame_bandwidth;         /* rc->max_frame_bandwidth */
  int32_t recode_tolerance;            /* sf.hl_sf.recode_tolerance */
  double rate_correction_factors[4];   /* p_rc->rate_correction_factors */
} ShimRcStateParams;

/* The AV1EncoderConfig fields av1_primary_rc_init / av1_rc_init /
 * av1_rc_update_framerate read. One struct for all three, because they read
 * overlapping subsets and splitting it would let the differential's setup
 * diverge between them. Keep in lockstep with Rust's `RefRcInitCfg`. */
typedef struct {
  int32_t rc_mode;               /* oxcf.rc_cfg.mode */
  int32_t best_allowed_q;        /* oxcf.rc_cfg.best_allowed_q (a qindex) */
  int32_t worst_allowed_q;       /* oxcf.rc_cfg.worst_allowed_q (a qindex) */
  int64_t target_bandwidth;      /* oxcf.rc_cfg.target_bandwidth */
  int32_t vbrmin_section;
  int32_t vbrmax_section;
  int32_t min_gf_interval;       /* oxcf.gf_cfg.min_gf_interval */
  int32_t max_gf_interval;       /* oxcf.gf_cfg.max_gf_interval */
  int32_t fwd_kf_dist;           /* oxcf.kf_cfg.fwd_kf_dist */
  int32_t width, height;         /* oxcf.frm_dim_cfg */
  double init_framerate;         /* oxcf.input_cfg.init_framerate */
  int32_t bit_depth;             /* oxcf.tool_cfg.bit_depth */
  int32_t one_pass;              /* oxcf.pass == AOM_RC_ONE_PASS */
  int32_t target_seq_level_idx0; /* oxcf.target_seq_level_idx[0] */
  int64_t starting_buffer_level; /* p_rc->starting_buffer_level */
  double framerate;              /* cpi->framerate (the RUNNING rate) */
  int32_t lap_enabled;           /* cpi->ppi->lap_enabled */
} ShimRcInitCfg;

#endif  /* AOM_RS_SHIM_RC_STATE_PARAMS_H_ */
