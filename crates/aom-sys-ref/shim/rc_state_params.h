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

/* Everything av1_rc_postencode_update (and the two functions it calls,
 * av1_rc_update_rate_correction_factors and update_buffer_level) reads or
 * writes. The shim copies this INTO a real AV1_COMP, runs the function, and
 * copies the mutated fields back out over the same struct — so the Rust side
 * sees the post-state directly and a field the port forgot to update shows up
 * as a mismatch rather than as a value that happened to already be right.
 * Keep in lockstep with Rust's `RefRcUpdateState`. */
typedef struct {
  /* ---- inputs that are not state ---------------------------------- */
  int64_t bytes_used;
  int32_t base_qindex;            /* cm->quant_params.base_qindex */
  int32_t coded_width, coded_height;
  int32_t cfg_width, cfg_height;  /* oxcf.frm_dim_cfg */
  int32_t show_frame;             /* cm->show_frame */
  int32_t frame_type;             /* cm->current_frame.frame_type */
  int32_t frame_number;           /* cm->current_frame.frame_number */
  int32_t update_type;            /* gf_group->update_type[gf_index] */
  int32_t refresh_golden, refresh_alt_ref;
  int32_t lag_in_frames;          /* oxcf.gf_cfg.lag_in_frames */
  int32_t enable_auto_arf;        /* oxcf.gf_cfg.enable_auto_arf */
  int32_t bit_depth;
  int32_t screen_content;         /* cpi->is_screen_content_type */
  int32_t rc_mode;                /* oxcf.rc_cfg.mode */
  int32_t stat_consumption;       /* is_stat_consumption_stage(cpi) */
  int32_t gf_cbr_boost_pct;
  int32_t tune_content_screen;    /* oxcf.tune_cfg.content == AOM_CONTENT_SCREEN */
  int32_t is_encode_stage;        /* the av1_rc_update_rate_correction_factors arg */

  /* ---- RATE_CONTROL state, in AND out ------------------------------ */
  int32_t projected_frame_size;
  int32_t q_1_frame, q_2_frame;
  int32_t rc_1_frame, rc_2_frame;
  int32_t this_frame_target;
  int32_t avg_frame_bandwidth;
  int32_t prev_avg_frame_bandwidth;
  int32_t frames_since_key;
  int32_t frames_since_golden;
  int32_t frame_num_last_gf_refresh;
  int32_t frame_source_sad;
  int32_t last_frame_low_source_sad;
  int32_t frame_number_encoded;
  int32_t prev_coded_width, prev_coded_height;
  int32_t prev_frame_is_dropped;
  int32_t drop_count_consec;
  int32_t ni_tot_qi, ni_av_qi;
  int32_t is_src_frame_alt_ref;
  int32_t last_encoded_size_keyframe, last_target_size_keyframe;
  int32_t rtc_external_ratectrl;
  int32_t frames_since_scene_change;

  /* ---- PRIMARY_RATE_CONTROL state, in AND out ---------------------- */
  int32_t last_q_key, last_q_inter;
  int32_t avg_frame_qindex_key, avg_frame_qindex_inter;
  int32_t ni_frames;
  double tot_q, avg_q;
  int32_t last_boosted_qindex, last_kf_qindex;
  double rate_correction_factors[4];
  int64_t bits_off_target, buffer_level, maximum_buffer_size;
  int64_t total_actual_bits, total_target_bits;
  int32_t rolling_target_bits, rolling_actual_bits;
  int32_t constrained_gf_group;
} ShimRcUpdateState;

#endif  /* AOM_RS_SHIM_RC_STATE_PARAMS_H_ */
