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

#endif  /* AOM_RS_SHIM_RC_STATE_PARAMS_H_ */
