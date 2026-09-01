/* Oracle shims for av1/encoder/encode_strategy.c's reference / GOP management
 * layer, ported into crate aom-encode, module ref_gop. Oracle use only. Every
 * entry drives the REAL exported C function.
 *
 * The functions here take `AV1_COMP *` and reach `cpi->ppi->gf_group`, so each
 * shim calloc's an AV1_COMP plus an AV1_PRIMARY and populates ONLY the fields
 * the C function reads. Two contracts that matter (DIFFERENTIAL_PLAYBOOK §3a):
 *
 *  - build.rs passes -DNDEBUG to every shim, matching the Release archive.
 *    Without it MACROBLOCK gains its `last_set_offsets_loc` member and every
 *    AV1_COMP field offset in this TU disagrees with the archive's.
 *  - `is_one_pass_rt_params(cpi)` is `has_no_stats_stage && lag_in_frames == 0
 *    && (mode == REALTIME || svc.number_spatial_layers > 1)`. A calloc'd cpi
 *    has mode == GOOD (0) and 0 spatial layers, so it reads FALSE; the
 *    `one_pass_rt` argument below sets oxcf.mode = REALTIME (with pass =
 *    AOM_RC_ONE_PASS, lag 0) to reach the other arm. Driving these with a
 *    zeroed cpi would silently exercise only enable_refresh_skip = 1.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "av1/common/av1_common_int.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/encode_strategy.h"
#include "av1/encoder/firstpass.h"
#include "av1/encoder/ratectrl.h"

/* Slot 0 of the gf group is used throughout; the frame-parallel arm of
 * av1_get_ref_frames reads gf_index - 1, so the shims place the current frame
 * at gf_index 1 and keep entry 0 as the previous frame.
 */
#define SHIM_GF_INDEX 1

typedef struct {
  AV1_COMP *cpi;
  AV1_PRIMARY *ppi;
} ShimEnc;

static int shim_enc_alloc(ShimEnc *e, int one_pass_rt) {
  e->cpi = (AV1_COMP *)calloc(1, sizeof(AV1_COMP));
  e->ppi = (AV1_PRIMARY *)calloc(1, sizeof(AV1_PRIMARY));
  if (!e->cpi || !e->ppi) {
    free(e->cpi);
    free(e->ppi);
    e->cpi = NULL;
    e->ppi = NULL;
    return 0;
  }
  e->cpi->ppi = e->ppi;
  e->cpi->oxcf.pass = AOM_RC_ONE_PASS;
  e->cpi->oxcf.gf_cfg.lag_in_frames = 0;
  e->cpi->oxcf.mode = one_pass_rt ? REALTIME : GOOD;
  return 1;
}

static void shim_enc_free(ShimEnc *e) {
  free(e->cpi);
  free(e->ppi);
}

/* ---- av1_get_refresh_ref_frame_map ---------------------------------- */
int shim_get_refresh_ref_frame_map(int refresh_frame_flags) {
  return av1_get_refresh_ref_frame_map(refresh_frame_flags);
}

/* ---- av1_configure_buffer_updates ------------------------------------
 * Returns the three refresh flags packed as bits 0/1/2 (golden/bwd/alt),
 * `is_src_frame_alt_ref` in bit 3, and the (possibly rewritten)
 * gf_group->update_type in bits 8..15.
 */
int shim_configure_buffer_updates(int update_type, int refbuf_state,
                                  int force_refresh_all, int ext_pending,
                                  int ext_golden, int ext_bwd, int ext_alt) {
  ShimEnc e;
  if (!shim_enc_alloc(&e, 0)) return -1;
  e.cpi->gf_frame_index = SHIM_GF_INDEX;
  e.ppi->gf_group.update_type[SHIM_GF_INDEX] = (FRAME_UPDATE_TYPE)update_type;
  e.cpi->ext_flags.refresh_frame.update_pending = ext_pending;
  e.cpi->ext_flags.refresh_frame.golden_frame = ext_golden;
  e.cpi->ext_flags.refresh_frame.bwd_ref_frame = ext_bwd;
  e.cpi->ext_flags.refresh_frame.alt_ref_frame = ext_alt;

  RefreshFrameInfo refresh;
  memset(&refresh, 0, sizeof(refresh));
  av1_configure_buffer_updates(e.cpi, &refresh,
                               (FRAME_UPDATE_TYPE)update_type,
                               (REFBUF_STATE)refbuf_state, force_refresh_all);

  int out = (refresh.golden_frame ? 1 : 0) | (refresh.bwd_ref_frame ? 2 : 0) |
            (refresh.alt_ref_frame ? 4 : 0) |
            (e.cpi->rc.is_src_frame_alt_ref ? 8 : 0);
  out |= ((int)e.ppi->gf_group.update_type[SHIM_GF_INDEX]) << 8;
  shim_enc_free(&e);
  return out;
}

/* ---- av1_calc_refresh_idx_for_intnl_arf ------------------------------
 * `pairs` is 2 * REF_FRAMES ints: [pyr_level, disp_order] per slot.
 * `skip_frame_refresh` is REF_FRAMES ints (-1 terminated).
 */
static void shim_fill_pairs(RefFrameMapPair *dst, const int32_t *src) {
  for (int i = 0; i < REF_FRAMES; ++i) {
    dst[i].pyr_level = src[2 * i];
    dst[i].disp_order = src[2 * i + 1];
  }
}

int shim_calc_refresh_idx_for_intnl_arf(const int32_t *pairs,
                                        const int32_t *skip_frame_refresh,
                                        int one_pass_rt, int cur_frame_disp) {
  ShimEnc e;
  if (!shim_enc_alloc(&e, one_pass_rt)) return -2;
  RefFrameMapPair map_pairs[REF_FRAMES];
  shim_fill_pairs(map_pairs, pairs);
  for (int i = 0; i < REF_FRAMES; ++i)
    e.ppi->gf_group.skip_frame_refresh[SHIM_GF_INDEX][i] =
        skip_frame_refresh[i];
  e.ppi->gf_group.display_idx[SHIM_GF_INDEX] = cur_frame_disp;
  const int r =
      av1_calc_refresh_idx_for_intnl_arf(e.cpi, map_pairs, SHIM_GF_INDEX);
  shim_enc_free(&e);
  return r;
}

/* ---- av1_get_refresh_frame_flags ------------------------------------- */
int shim_get_refresh_frame_flags(const int32_t *pairs, int refbuf_state,
                                 int frame_type, int show_existing_frame,
                                 int update_type,
                                 const int32_t *skip_frame_refresh,
                                 int one_pass_rt, int cur_disp_order,
                                 int ext_pending, int ext_last, int ext_golden,
                                 int ext_bwd, int ext_alt, int ext_alt2,
                                 const int32_t *remapped_ref_idx) {
  ShimEnc e;
  if (!shim_enc_alloc(&e, one_pass_rt)) return -2;
  RefFrameMapPair map_pairs[REF_FRAMES];
  shim_fill_pairs(map_pairs, pairs);

  e.ppi->gf_group.refbuf_state[SHIM_GF_INDEX] = (REFBUF_STATE)refbuf_state;
  for (int i = 0; i < REF_FRAMES; ++i) {
    e.ppi->gf_group.skip_frame_refresh[SHIM_GF_INDEX][i] =
        skip_frame_refresh[i];
    e.cpi->common.remapped_ref_idx[i] = remapped_ref_idx[i];
  }
  e.cpi->ext_flags.refresh_frame.update_pending = ext_pending;
  e.cpi->ext_flags.refresh_frame.last_frame = ext_last;
  e.cpi->ext_flags.refresh_frame.golden_frame = ext_golden;
  e.cpi->ext_flags.refresh_frame.bwd_ref_frame = ext_bwd;
  e.cpi->ext_flags.refresh_frame.alt_ref_frame = ext_alt;
  e.cpi->ext_flags.refresh_frame.alt2_ref_frame = ext_alt2;

  EncodeFrameParams frame_params;
  memset(&frame_params, 0, sizeof(frame_params));
  frame_params.frame_type = (FRAME_TYPE)frame_type;
  frame_params.show_existing_frame = show_existing_frame;

  const int r = av1_get_refresh_frame_flags(
      e.cpi, &frame_params, (FRAME_UPDATE_TYPE)update_type, SHIM_GF_INDEX,
      cur_disp_order, map_pairs);
  shim_enc_free(&e);
  return r;
}

/* ---- av1_get_ref_frames ----------------------------------------------
 * `parallel_kind`: 0 = the frame-parallel exclusion is not reached (the common
 * case), 1 = reached with is_parallel_encode (skip by map index), 2 = reached
 * without it (skip by display order). `skip_value` carries the map index or
 * display order respectively.
 */
int shim_get_ref_frames(const int32_t *pairs, int cur_frame_disp,
                        int one_pass_rt, int parallel_kind, int skip_value,
                        int use_ext_ref_frame_map, const int32_t *ref_frame_list,
                        int32_t *out_remapped) {
  ShimEnc e;
  if (!shim_enc_alloc(&e, one_pass_rt)) return -2;
  RefFrameMapPair map_pairs[REF_FRAMES];
  shim_fill_pairs(map_pairs, pairs);

  e.ppi->gf_group.use_ext_ref_frame_map[SHIM_GF_INDEX] = use_ext_ref_frame_map;
  for (int i = 0; i < REF_FRAMES; ++i)
    e.ppi->gf_group.ref_frame_list[SHIM_GF_INDEX][i] =
        (int8_t)ref_frame_list[i];

  int is_parallel_encode = 0;
  if (parallel_kind != 0) {
    e.ppi->gf_group.frame_parallel_level[SHIM_GF_INDEX] = 2;
    e.ppi->gf_group.frame_parallel_level[SHIM_GF_INDEX - 1] = 1;
    e.ppi->gf_group.update_type[SHIM_GF_INDEX - 1] = INTNL_ARF_UPDATE;
    e.ppi->gf_group.update_type[SHIM_GF_INDEX] = INTNL_ARF_UPDATE;
    if (parallel_kind == 1) {
      is_parallel_encode = 1;
      e.cpi->ref_idx_to_skip = skip_value;
      e.ppi->gf_group.skip_frame_as_ref[SHIM_GF_INDEX] = INVALID_IDX;
    } else {
      e.cpi->ref_idx_to_skip = INVALID_IDX;
      e.ppi->gf_group.skip_frame_as_ref[SHIM_GF_INDEX] = skip_value;
    }
  }

  int remapped[REF_FRAMES];
  av1_get_ref_frames(map_pairs, cur_frame_disp, e.cpi, SHIM_GF_INDEX,
                     is_parallel_encode, remapped);
  for (int i = 0; i < REF_FRAMES; ++i) out_remapped[i] = remapped[i];
  shim_enc_free(&e);
  return 0;
}
