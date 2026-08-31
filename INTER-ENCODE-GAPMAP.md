# Inter-ENCODE C surface — triaged gap map

**Measured 2026-08-31** on `upstream/` libaom v3.14.1 (`03087864`) with
`tools/c_surface_inventory.py`. Raw data:
[`benchmarks/inter_encode_c_surface_2026-08-31.tsv`](benchmarks/inter_encode_c_surface_2026-08-31.tsv)
(one row per C function: file, name, ported/MISSING, and whether an exported
symbol exists in `upstream/build/libaom.a`).

This is a **work queue, not a coverage claim.** Two limits, both structural:

1. **The match is by NAME.** The port deliberately names its functions after
   C's, so a name hit is good evidence of a port and a name miss is good
   evidence of a gap — but a hit can be a doc comment mentioning the C function
   rather than a port of it, and a miss can be a port that renamed. Every
   "ported" row below has been treated as unverified.
2. **The definition regex only sees single-line, column-0 definitions.** libaom
   wraps long signatures across lines constantly. **Every total here is a LOWER
   BOUND on the real surface.** A file reported with `gap 0` is not proven
   complete.

The `sym` column is *not* a heuristic — it is `nm -g` on the built oracle
archive. `Y` means a **tier-1 differential against the real exported C function
is possible** through `crates/aom-sys-ref`. `n` means the function is `static`;
its oracle must be a shim that drives its exported caller, or hand-derived
vectors **labelled tier 4**.

## 1. The numbers

| | count |
|---|---|
| C function definitions found in the 43-file inter-encode scope | 1491 |
| name-matched somewhere under `crates/` | 517 |
| **unmatched** | **974** |
| unmatched **with an exported symbol** (tier-1 differential available) | **268** |
| unmatched, `static` (tier-1 only via an exported caller) | 706 |

## 2. Triage — what is in scope for a byte-exact inter encoder

### 2.1 OUT OF SCOPE — the port replaces the mechanism, not the math

These are not untranslated *semantics*; they are C-specific plumbing that a Rust
port structurally does not have. Listing them so the 974 is not read as 974
things to write.

| C file / family | unmatched | why out |
|---|---|---|
| `encoder/lookahead.c` (all 8 exported + statics) | 8 | Frame-queue allocation and ring-buffer bookkeeping (`av1_lookahead_init/destroy/push/pop/peek/full/depth`). The port feeds frames through Rust ownership; there is no `struct lookahead_ctx` to reproduce. No bitstream consequence. |
| `encoder/context_tree.c` (all 10) | 10 | `PC_TREE` / `PICK_MODE_CONTEXT` malloc/free and the shared coeff-buffer pool. Rust owns these by value; `av1_copy_tree_context` is a memcpy of a struct the port does not have. |
| alloc/free pairs elsewhere: `av1_alloc_txb_buf`, `av1_free_txb_buf`, `av1_get_cb_coeff_buffer`, `av1_alloc_src_diff_buf`, `av1_dealloc_src_diff_buf`, `av1_init_obmc_buffer`, `av1_setup_shared_coeff_buffer`, `av1_free_shared_coeff_buffer`, `av1_setup_sms_tree`, `av1_free_sms_tree`, `av1_alloc_pmc`, `av1_free_pmc`, `av1_alloc_pc_tree_node`, `av1_free_pc_tree_recursive`, `av1_tf_info_alloc`, `av1_tf_info_free`, `av1_free_firstpass_data`, `av1_free_tpl_gop_stats`, `av1_setup_tpl_buffers` | 19 | Allocation. |
| threading: `av1_accumulate_pack_bs_thread_data`, `av1_reset_pack_bs_thread_data`, `av1_init_rtc_counters`, `av1_accumulate_rtc_counters`, `av1_alloc_tile_data`, `calc_pack_bs_mt_workers`, `av1_first_pass_row`, `av1_mc_flow_dispenser_row`, `av1_tf_do_filtering_row`, `wait_for_top_right_sb`, `delay_wait_for_top_right_sb`, `encode_tiles` | 12 | The oracle is built `CONFIG_MULTITHREAD=0`; the port is single-threaded by construction. Row/tile workers exist only to shard work that the port does in one pass. |
| debug / stats dump: `dump_mode_info`, `enc_dump_logs`, `dump_one_image`, `dump_ref_frame_images`, `av1_dump_filtered_recon_frames`, `print_reconstruction_frame`, `output_stats`, `write_features_to_file`, `write_motion_feature_to_file`, `get_feature_file_name`, `print_stage_time`, `print_time`, `print_partition_timing_stats*`, `start/end_partition_block_timer`, `accumulate_partition_timing_stats`, `init_partition_block_timing_stats`, `rd_token_stats_mismatch`, `av1_read_rd_command` | 20 | Compiled out or debug-only; no bitstream effect. |
| external-partition / ML-experiment hooks: `ext_ml_model_decision_*` (8), `external_partition.c` API, `read_partition_tree`, `write_partition_tree`, `verify_write_partition_tree`, `av1_reset_sf_for_ext_part`, `ml_partition_search_whole_tree`, `ml_partition_search_partial`, `av1_rd_partition_search` | ~15 | Reachable only through the external-partition API / `CONFIG_PARTITION_SEARCH_ORDER`, which the port does not expose. |
| large-scale-tile (VR) OBU path: `pack_large_scale_tiles_in_tg_obus`, `write_large_scale_tile_obu`, `write_large_scale_tile_obu_size`, `init_large_scale_tile_obu_header`, `find_identical_tile` | 5 | `large_scale_tile` is not a supported config. |
| SVC / real-time-only rate control: `av1_set_rtc_reference_structure_one_layer`, `av1_adjust_gf_refresh_qp_one_pass_rt`, `av1_rc_scene_detection_onepass_rt`, `av1_get_one_pass_rt_params`, `av1_encodedframe_overshoot_cbr`, `av1_postencode_drop_cbr`, `av1_rc_drop_frame`, `av1_rc_postencode_update_drop_frame`, `dynamic_resize_one_pass_cbr`, `resize_reset_rc`, `rc_compute_variance_onepass_rt`, `rc_spatial_act_onepass_rt`, `adjust_rtc_keyframe`, `update_layer_buffer_level`, `set_flag_rps_bias_recovery_frame` | 15 | CBR/RT/SVC. The encode target is `--end-usage=q`; these arms are unreachable there. In scope only if CBR is ever a goal. |
| denoiser / active-map / ROI: `av1_pickmode_ctx_den_update`, `recheck_zeromv_after_denoising`, `av1_apply_active_map`, `av1_apply_roi_map` | 4 | Not in the supported config set. |
| `av1_write_metadata_obu` / `av1_write_metadata_array` / `write_tu_pts_info` / `write_profile` / `mem_put_varsize` / `obu_memmove*` / `remux_tiles` | 8 | Already covered by the ported OBU assembler (`obu_assemble.rs`) under different names — **verify before dismissing**, these are name-misses not proven absences. |

### 2.2 IN SCOPE, DEFERRED — needed only outside the first envelope

The first byte-exact target (INTER-ENCODE-ROADMAP §3) is `--lag-in-frames=0
--end-usage=q`, which structurally removes alt-ref, TPL, temporal filtering and
2-pass. These are real inter-encode semantics, they are just not on the
critical path to the first byte-exact P-frame.

| family | unmatched | gate that makes it reachable |
|---|---|---|
| `encoder/firstpass.c` | 43 | 2-pass (`--passes=2`) |
| `encoder/pass2_strategy.c` | 85 | 2-pass |
| `encoder/tpl_model.c` | 62 | `lag>0` + TPL enabled |
| `encoder/temporal_filter.c` | 27 | `lag>=ALT_MIN_LAG` + alt-ref |
| `encoder/gop_structure.c` | 17 | `lag>0` (a `lag=0` GF group is trivial) |
| `encoder/nonrd_pickmode.c` + `nonrd_opt.c` + `var_based_part.c` | 71 | `--cpu-used 8/9` inter (the nonrd pickmode) |
| `encoder/ratectrl.c` CBR/VBR arms | ~40 | `--end-usage` other than `q` |

### 2.3 IN SCOPE, ON THE CRITICAL PATH

Ranked. `sym Y` throughout unless noted; the static helpers listed with each
family are ported as part of their exported parent and verified through it.

**W1 — compound / masked prediction DSP.** Pure kernels, all exported, no
encoder state. Every compound, wedge, diff-weighted, interintra and OBMC inter
mode is built on these, and none of them exist in the port.
- `av1_wedge_sse_from_residuals_c`, `av1_wedge_sign_from_residuals_c`,
  `av1_wedge_compute_delta_squares_c` (`wedge_utils.c`)
- `av1_build_compound_diffwtd_mask_c`, `_d16_c`, `_highbd_c` +
  statics `diffwtd_mask`, `diffwtd_mask_d16`, `diffwtd_mask_highbd`
  (`common/reconinter.c`)
- `av1_dist_wtd_comp_weight_assign`, `av1_get_compound_type_mask`
- `av1_dist_wtd_convolve_2d_c`, `_x_c`, `_y_c`, `_2d_copy_c` and the four
  `av1_highbd_dist_wtd_convolve_*_c` (`common/convolve.c`)
- `av1_highbd_convolve_2d_sr_c`, `_x_sr_c`, `_y_sr_c` (10/12-bit inter MC)

**W2 — encoder-side inter predictor build** (`reconinter_enc.c`)
- `av1_enc_build_one_inter_predictor`, `av1_enc_build_inter_predictor_y`,
  `av1_build_inter_predictors_for_planes_single_buf`,
  `av1_build_wedge_inter_predictor_from_buf`, `av1_build_obmc_inter_predictors_sb`
- `aom_comp_avg_upsampled_pred_c`, `aom_highbd_upsampled_pred_c`,
  `aom_comp_mask_upsampled_pred`
- `av1_make_masked_inter_predictor` (`common/reconinter.c`)

**W3 — motion search beyond speed-0 single-ref** (`mcomp.c`,
`motion_search_facade.c`)
- `av1_find_best_sub_pixel_tree_pruned`, `_pruned_more` (speeds 1+)
- `av1_return_min_sub_pixel_mv`, `av1_return_max_sub_pixel_mv` (forced-stop arms)
- `av1_refining_search_8p_c`, `av1_make_default_subpel_ms_params`
- `av1_obmc_full_pixel_search`, `av1_find_best_obmc_sub_pixel_tree_up`
- `av1_refine_warped_mv` (WARPED_CAUSAL RD)
- `av1_joint_motion_search`, `av1_compound_single_motion_search`,
  `av1_interinter_compound_motion_search`
- statics: the search-method family (`hex_search`, `bigdia_search`,
  `fast_dia_search`, `vfast_dia_search`, `fast_bigdia_search`, their
  `init_motion_compensation_*`), the `calc_sad*_update_bestmv` inner loops,
  `check_better_fast` / `first_level_check_fast` / `second_level_check_fast` /
  `two_level_checks_fast`, and the whole `obmc_*` subpel chain

**W4 — compound-ref prediction contexts** (`common/pred_common.c`) — 6 small
exported functions the compound `write_ref_frames` cascade needs:
`av1_get_pred_context_comp_ref_p`, `_p1`, `_p2`,
`av1_get_pred_context_comp_bwdref_p`, `_p1`,
`av1_get_pred_context_uni_comp_ref_p2`

**W5 — MV prediction / search seeding** (`rd.c`, `encodemv.c`, `scale.c`)
- `av1_mv_pred` (search start MV), `av1_fill_mv_costs`, `av1_setup_pred_block`
- `av1_get_ref_mv_from_stack`, `av1_find_best_ref_mvs_from_stack`
- `av1_setup_scale_factors_for_frame`, `av1_scale_mv` (scaled references)
- `av1_get_scaled_ref_frame`, `av1_update_rd_thresh_fact`,
  `av1_set_rd_speed_thresholds`, `av1_get_adaptive_rdmult`,
  `av1_get_intra_cost_penalty`

**W6 — the inter RD brain** (`rdopt.c`, 89 unmatched; `compound_type.c`, 33;
`interp_search.c`, 8; `tx_search.c` inter arm, 20). Almost entirely `static`,
so tier-1 requires driving `av1_rd_pick_inter_mode_sb` itself. Named heads:
`set_params_rd_pick_inter_mode`, `handle_newmv`, `motion_mode_rd`,
`build_cur_mv`, `get_drl_cost`, `skip_mode_rd`, `rd_pick_skip_mode`,
`calc_target_weighted_pred*` (the OBMC target), `av1_compound_type_rd`,
`av1_handle_inter_intra_mode`, `pick_interinter_wedge`, `pick_interinter_seg`,
`inter_block_yrd` / `tx_block_yrd` (the var-tx coeff arm).

**W7 — reference / frame management** (`encode_strategy.c`,
`encodeframe_utils.c`): `av1_get_ref_frames`, `av1_get_refresh_ref_frame_map`,
`av1_configure_buffer_updates`, `choose_primary_ref_frame`,
`set_refresh_frame_flags`, `av1_update_inter_mode_stats`,
`av1_update_picked_ref_frames_mask`, `av1_is_leaf_split_partition`,
`av1_active_h_edge` / `_v_edge`, `av1_get_rdmult_delta`, `av1_get_cb_rdmult`.

**W8 — multi-frame fixed-Q rate control** (`ratectrl.c`, `q` arms only):
`av1_rc_regulate_q`, `av1_rc_postencode_update`,
`av1_rc_update_rate_correction_factors`, `av1_estimate_bits_at_q`,
`av1_rc_compute_frame_size_bounds`, `av1_rc_update_framerate`,
`av1_primary_rc_init`, `av1_rc_get_default_min_gf_interval`,
`av1_rc_set_frame_target`, plus the statics `get_active_cq_level`,
`get_minq_index`, `frame_type_qdelta`, `adjust_active_best_and_worst_quality`.

**W9 — global motion estimation** (`global_motion.c`, `global_motion_facade.c`):
`av1_refine_integerized_param`, `av1_convert_model_to_params`,
`av1_is_enough_erroradvantage`, `av1_segmented_frame_error`,
`av1_compute_feature_segmentation_map`, `av1_compute_global_motion_facade`,
`av1_compute_gm_for_valid_ref_frames`, plus the `warp_error` /
`highbd_warp_error` / `generic_sad` statics.

## 3. Landed against this map

Each entry names the wave, the C function, the tier of its gate, and the test.

| wave | C function | tier | gate |
|---|---|---|---|
| — | *(updated as work lands)* | | |
