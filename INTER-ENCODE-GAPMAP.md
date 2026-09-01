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
| ~~`encoder/firstpass.c`~~ | ~~43~~ **14 left** | 2-pass (`--passes=2`) — see §3.1 |
| `encoder/pass2_strategy.c` | 85 | 2-pass |
| ~~`encoder/tpl_model.c`~~ | ~~62~~ **15 left** | `lag>0` + TPL enabled — see §3.1 |
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
so tier-1 requires driving `av1_rd_pick_inter_mode_sb` itself.

> **rdopt.c status, 2026-09-01: 67 of its 105 functions ported and gated, 29
> NOT ported.** The missing ones are named individually with a reason in
> [`docs/RDOPT_C_COVERAGE_2026-09-01.md`](docs/RDOPT_C_COVERAGE_2026-09-01.md);
> read that rather than re-running the inventory, which now reports rdopt.c as
> nearly complete because every ported module cites its C function by name in a
> doc comment. The oracle for the `static` functions is a new technique —
> `shim/rdopt_shim.c` compiles libaom's own rdopt.c into the shim archive
> ("tier 1c"), with a measurement that it agrees with the archive's copy.

Named heads:
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

## 3.1 tpl_model.c + firstpass.c — full accounting (wave `wx-tpl`, 2026-08-31)

**51 C functions ported and gated.** Both files were "IN SCOPE, DEFERRED"
above; the owner then asked for a full translation, so this is the accounting
for them, by NAME rather than by the inventory's count (see the caveat at the
end of this section).

Ports: `crates/aom-encode/src/tpl_model.rs`, `crates/aom-encode/src/firstpass.rs`.
Gates: `crates/aom-encode/tests/tpl_model_diff.rs` (36/36),
`crates/aom-encode/tests/firstpass_diff.rs` (20/20).
Shims: `tpl_shim.c` + `fp_info_shim.c` (tier 1, no libaom `.c` included),
`tpl_c_shim.c` + `fp_shim.c` (tier 1c, the file's own source compiled verbatim
with its exported symbols renamed, each with a TU-agreement gate).

### `tpl_model.c` — 63 definitions

| | count | |
|---|---|---|
| ported + gated | **30** | listed below |
| compiled OUT of this build | 14 | `CONFIG_BITRATE_ACCURACY` / `CONFIG_RD_COMMAND` are **0** (`upstream/build/config/aom_config.h:26,62`), confirmed absent from `libaom.a` with `nm -g`: `av1_accumulate_tpl_txfm_stats`, `av1_record_tpl_txfm_block`, `av1_tpl_txfm_stats_update_abs_coeff_mean`, `av1_tpl_store_txfm_stats`, `av1_laplace_estimate_frame_rate`, `av1_read_rd_command`, and the eight `av1_vbr_rc_*`. **No oracle at any tier**, so they are counted out rather than reported missing. |
| mechanism the port replaces | 4 | `av1_setup_tpl_buffers`, `av1_free_tpl_gop_stats` (allocation), `av1_mc_flow_dispenser_row` (row-worker sharding; the oracle is `CONFIG_MULTITHREAD=0`), `tpl_reset_src_ref_frames` (nulls two pointer arrays the port does not have) |
| **still missing, in scope** | **15** | named below |

**Ported (30).** `av1_exponential_entropy`, `av1_laplace_entropy`,
`av1_estimate_coeff_entropy`, `av1_get_overlap_area`, `av1_tpl_ptr_pos`,
`av1_delta_rate_cost`, `av1_get_q_index_from_qstep_ratio`,
`av1_tpl_stats_ready`, `av1_tpl_get_qstep_ratio`, `av1_tpl_get_q_index`,
`av1_init_tpl_stats`, `av1_tpl_rdmult_setup`, `av1_tpl_rdmult_setup_sb`,
`av1_compute_mv_difference`, `av1_tpl_compute_frame_mv_entropy`,
`av1_init_tpl_txfm_stats` + the statics `exp_bounded`, `round_floor`,
`rate_estimator`, `set_tpl_stats_block_size`, `get_frame_importance`,
`tpl_model_update_b`, `tpl_model_update`, `tpl_model_store`,
`mc_flow_synthesizer`, `get_gop_length`, `eval_gop_length`,
`skip_tpl_for_frame`, `is_alike_mv`, `compare_sad`.

**Still missing (15)**, all `static`, all one connected subsystem — the
per-block TPL pass and the GOP driver that runs it. They need the encoder's
frame buffers, motion search and quantizer wired up, which is the next chunk:
`mode_estimation`, `motion_estimation`, `get_rate_distortion`,
`get_inter_cost`, `get_quantize_error`, `txfm_quant_rdcost`,
`tpl_get_satd_cost`, `init_mc_flow_dispenser`, `mc_flow_dispenser`,
`init_gop_frames_for_tpl`, `tpl_store_before_propagation`,
`init_tpl_stats_before_propagation`, `trim_tpl_stats`, `av1_tpl_setup_stats`,
`av1_tpl_preload_rc_estimate`.

### `firstpass.c` — 45 definitions

| | count | |
|---|---|---|
| ported + gated | **21** | |
| mechanism the port replaces | 10 | `output_stats` + `av1_end_first_pass` (packet plumbing), `print_reconstruction_frame` (debug), `setup_firstpass_data` + `av1_free_firstpass_data` (allocation), `av1_first_pass_row` + `first_pass_tile` + `first_pass_tiles` (row/tile sharding), `av1_get_first_pass_search_site_config` (picks a preallocated config out of a table), `copy_rect` (a `memcpy` helper) |
| **still missing, in scope** | **14** | named below |

**Ported (21).** `av1_twopass_zero_stats`, `av1_accumulate_stats`,
`av1_get_unit_rows_in_tile`, `av1_get_unit_cols_in_tile`, the seven
`av1_firstpass_info_*` entry points + the statics `get_unit_rows`,
`get_unit_cols`, `get_num_mbs`, `get_search_range`, `find_fp_qindex`,
`raw_motion_error_stdev`, `normalize_firstpass_stats`, `calc_wavelet_energy`,
`accumulate_frame_stats`, `accumulate_mv_stats`.

**Still missing (14)** — the per-block first pass itself:
`firstpass_intra_prediction`, `firstpass_inter_prediction`,
`first_pass_motion_search`, `first_pass_intra_pred_and_calc_diff`,
`first_pass_predict_intra_block_for_luma_plane`, `get_prediction_error`,
`highbd_get_prediction_error`, `get_prediction_error_bitdepth`,
`get_block_variance_fn`, `highbd_get_block_variance_fn`, `get_bsize`,
`update_firstpass_stats`, `av1_first_pass`, `av1_noop_first_pass_frame`.

### The inventory tool misses a renamed port — read its number as a LOWER bound here

`tools/c_surface_inventory.py` matches `fn <name>` in the port's own Rust
source. After the 2026-08-31 fix that stopped it crediting doc comments and
shims, it now reports **34 unmatched for tpl_model.c and 33 for firstpass.c**,
against the 15 and 14 above. The difference is entirely functions ported under
an idiomatic Rust name: `av1_twopass_zero_stats` is `FirstpassStats::zero`,
`av1_accumulate_stats` is `FirstpassStats::accumulate`, the seven
`av1_firstpass_info_*` are `FirstpassInfo::{push, pop, peek, ...}`,
`normalize_firstpass_stats` is `FirstpassStats::normalize`,
`av1_init_tpl_txfm_stats` is `TplTxfmStats::init`, and so on. The tool's own
header says a miss can be a rename; on these two files that is most of the
misses, so the counts above are the ones to use.

## 3. Landed against this map

**68 exported C functions and ~30 of their file-static helpers** through the
`rd_thresh_diff.rs` row, all gated at tier 1 against the real C symbol (the
statics through the exported caller that is their only entry point), plus the
six **tier 1c** rows below them: 25 of `compound_type.c`'s 34 functions and 6
of `reconinter_enc.c`'s. Tier 1c is libaom's own `.c` compiled verbatim into a
shim with its exports renamed — the technique `shim/rdopt_shim.c` introduced —
and it is what a file whose functions are almost all `static` admits.
Ordered by the commit that landed them.

| wave | C functions | gate |
|---|---|---|
| W1 | `av1_wedge_sse_from_residuals_c`, `av1_wedge_sign_from_residuals_c`, `av1_wedge_compute_delta_squares_c`, `av1_build_compound_diffwtd_mask_c` / `_d16_c` / `_highbd_c`, `av1_get_compound_type_mask`, `av1_dist_wtd_comp_weight_assign` (+ `av1_get_contiguous_soft_mask` generalized to a signed wedge) | `aom-dsp/tests/compound_diff.rs` |
| W1 | `av1_dist_wtd_convolve_2d_c` / `_x_c` / `_y_c` / `_2d_copy_c`, the four `av1_highbd_dist_wtd_convolve_*_c`, `av1_highbd_convolve_2d_sr_c` / `_x_sr_c` / `_y_sr_c` | `aom-dsp/tests/compound_convolve_diff.rs` |
| W3 | `av1_find_best_sub_pixel_tree_pruned`, `_pruned_more`, `av1_return_min_sub_pixel_mv`, `av1_return_max_sub_pixel_mv` (+ 8 statics: `estimated_pref_error`, `check_better_fast`, `first_level_check_fast`, `second_level_check_fast`, `two_level_checks_fast`, `setup_center_error`, `divide_and_round`, `is_cost_list_wellbehaved`, `get_cost_surf_min`) | `aom-encode/tests/subpel_tree_diff.rs` |
| W3 | `av1_refining_search_8p_c`, `av1_vector_match` | `aom-encode/tests/inter_fullpel_diff.rs` |
| W2 | `aom_comp_avg_pred_c`, `aom_comp_mask_pred_c`, `aom_highbd_comp_avg_pred_c`, `aom_highbd_comp_mask_pred_c`, `aom_comp_avg_upsampled_pred_c`, `aom_comp_mask_upsampled_pred`, `aom_highbd_upsampled_pred_c`, `aom_highbd_comp_avg_upsampled_pred_c`, `aom_highbd_comp_mask_upsampled_pred` | `aom-encode/tests/inter_pred_enc_diff.rs` |
| W5 | `av1_setup_scale_factors_for_frame`, `av1_scale_mv`, `av1_scaled_x`, `av1_scaled_y`, `av1_is_scaled`, `valid_ref_frame_size` | `aom-dsp/tests/scale_diff.rs` |
| W9 | `av1_is_enough_erroradvantage`, `av1_convert_model_to_params`, `get_wmtype`, `av1_compute_feature_segmentation_map`, `av1_segmented_frame_error` (lowbd + highbd) | `aom-encode/tests/global_motion_diff.rs` |
| — | the `aom_obmc_variance` / `aom_obmc_sub_pixel_variance` families, lowbd + the three highbd bit-depth arms | `aom-dsp/tests/obmc_dist_diff.rs` |
| W3 | `av1_obmc_full_pixel_search` (+ `obmc_full_pixel_diamond`, `obmc_diamond_search_sad`, `obmc_refining_search_sad`, `get_obmc_mvpred_var`) | `aom-encode/tests/inter_fullpel_diff.rs` |
| W3 | `av1_find_best_obmc_sub_pixel_tree_up` (+ `upsampled_obmc_pref_error`, `setup_obmc_center_error`, `upsampled_setup_obmc_center_error`, `estimate_obmc_mvcost`, `estimate_obmc_pref_error`, `obmc_check_better` / `_fast`, `obmc_first_level_check`, `obmc_second_level_check_v2`) | `aom-encode/tests/inter_fullpel_diff.rs` |
| W9 | `av1_refine_integerized_param` (+ `warp_error`, `get_warp_error`, `add_param_offset`, `force_wmtype`) | `aom-encode/tests/global_motion_diff.rs` |
| W1 | `aom_lowbd_blend_a64_d16_mask_c`, `aom_highbd_blend_a64_d16_mask_c` | `aom-dsp/tests/compound_diff.rs` |
| — | `av1_highbd_warp_affine_c` (the general warp filter: any bd, compound + single) | `aom-dsp/tests/warp_highbd_diff.rs` |
| — | `av1_convolve_2d_scale_c`, `av1_highbd_convolve_2d_scale_c` | `aom-dsp/tests/convolve_scale_diff.rs` |
| — | `av1_dropout_qcoeff_num`, `av1_get_intra_cost_penalty`, `av1_hash_is_horizontal_perfect`, `av1_hash_is_vertical_perfect` | `aom-encode/tests/enc_misc_diff.rs` |
| W8 | `av1_convert_qindex_to_q`, `av1_find_qindex`, `av1_compute_qdelta`, `av1_rc_bits_per_mb` (non-CBR arm), `av1_rc_get_default_min_gf_interval`, `get_bpmb_enumerator` | `aom-encode/tests/rate_model_diff.rs` |
| W6 | the `THR_MODES` enum (169 entries), `av1_set_rd_speed_thresholds`, `av1_update_rd_thresh_fact`, `update_thr_fact` | `aom-encode/tests/rd_thresh_diff.rs` |
| W6 | `compound_type.c`'s decision layer: `enable_wedge_search` / `_interinter_` / `_interintra_`, `compute_valid_comp_types`, `calc_masked_type_cost`, `update_mbmi_for_compound_type`, `get_interinter_compound_mask_rate`, `save_mask_search_results`, `push_comp_avg_est_rd`, `prune_comp_eval_using_comp_avg_est_rd`, `compute_rd_thresh` (+ `get_rd_thresh_from_best_rd`, `is_interinter_compound_used`) | `aom-encode/tests/compound_type_diff.rs` (**tier 1c**) |
| W6 | the mask picks: `estimate_wedge_sign`, `pick_wedge`, `pick_wedge_fixed_sign`, `pick_interinter_wedge`, `pick_interinter_seg`, `pick_interintra_wedge` | `compound_type_diff.rs` (tier 1c) |
| W6 | the compound-RD reuse cache: `is_comp_rd_match`, `find_comp_rd_in_stats`, `save_comp_rd_search_stat`, `backup_stats`, `update_best_info`, `update_mask_best_mv`, `populate_reuse_comp_type_data` (+ `is_global_mv_block`) | `compound_type_diff.rs` (tier 1c) |
| W6 | the transform-search gate: `prune_mode_by_skip_rd` (+ `get_txfm_rd_gate_level`, `check_txfm_eval`, `compute_sse_plane` / `calculate_sse`) | `compound_type_diff.rs` (tier 1c) |
| W2 | the masked-compound assembly: `build_masked_compound` (+ `_highbd`), `build_wedge_inter_predictor_from_buf`, `av1_build_wedge_inter_predictor_from_buf` | `aom-encode/tests/wedge_from_buf_diff.rs` (tier 1c) |
| W2 | the subpel derivation: `enc_calc_subpel_params`, `init_subpel_params`, the `top`/`left` half of `init_inter_block_params` | `aom-encode/tests/subpel_params_diff.rs` (tier 1c) |
| W7 | `encode_strategy.c`'s reference/GOP management: `av1_configure_buffer_updates`, `av1_get_refresh_ref_frame_map`, `av1_calc_refresh_idx_for_intnl_arf`, `av1_get_refresh_frame_flags` (default + external-override arms), `av1_get_ref_frames` (default + frame-parallel + `use_ext_ref_frame_map` arms) (+ statics `set_refresh_frame_flags`, `get_free_ref_map_index`, `get_refresh_idx`, `get_new_fb_map_idx_rc`, `is_in_ref_map`, `add_ref_to_slot`, `set_unmapped_ref`, `compare_map_idx_pair_asc`) | `aom-encode/tests/ref_gop_diff.rs` |
| W7 | tier 4 (C is `static`, only caller is `av1_encode_strategy`): `set_additional_frame_flags`, `update_frame_flags`, `set_ext_overrides`, `choose_primary_ref_frame` | unit tests in `aom-encode/src/ref_gop.rs` |
| W7 | the lookahead DECISIONS: `is_forced_keyframe_pending`, `av1_new_framerate` (clamp) | `aom-encode/tests/frame_source_diff.rs` |
| W7 | tier 4: `allow_show_existing`, `adjust_frame_rate` | unit tests in `aom-encode/src/frame_source.rs` |
| W8 | the fixed-Q qindex chain: `get_minq_index`, `init_minq_luts` / `rc_init_minq_luts` / `av1_rc_init_minq_luts`, `get_active_quality`, `get_kf_active_quality`, `get_gf_active_quality` (+ `_no_rc`), `get_gf_high_motion_quality`, `get_default_max_gf_interval`, `gf_group_pyramid_level`, `get_active_cq_level`, `get_intra_q_and_bounds`, `get_active_best_quality`, `rc_pick_q_and_bounds_q_mode` | `aom-encode/tests/ratectrl_q_diff.rs` (**tier 1c**) |
| W8 | the rate-search layer: `av1_get_MBs`, `av1_estimate_bits_at_q`, `av1_compute_qdelta_by_rate`, `av1_rc_regulate_q`, `av1_rc_compute_frame_size_bounds`, `av1_rc_set_frame_target` (tier 1) + `resize_rate_factor`, `get_rate_factor_level`, `get_rate_correction_factor`, `get_bits_per_mb`, `find_qindex_by_rate`, `find_closest_qindex_by_rate`, `frame_type_qdelta` (tier 1c) | `aom-encode/tests/ratectrl_rate_diff.rs` |
| W8 | RC initialisation: `av1_primary_rc_init`, `av1_rc_init`, `av1_rc_update_framerate` (tier 1) + `set_gf_interval_range` (tier 1c) | `aom-encode/tests/ratectrl_init_diff.rs` |
| W8 | the per-frame state advance: `av1_rc_update_rate_correction_factors`, `av1_rc_postencode_update` (+ `set_rate_correction_factor`, `update_buffer_level`, `update_alt_ref_frame_stats`, `update_golden_frame_stats`), `av1_set_target_rate` (Q/CBR arm) | `aom-encode/tests/ratectrl_update_diff.rs` |
| W8 | the q-and-bounds dispatcher: `av1_rc_pick_q_and_bounds` (tier 1) + `rc_pick_q_and_bounds`, `rc_pick_q_and_bounds_no_stats`, `adjust_active_best_and_worst_quality`, `get_q`, `calc_active_worst_quality_no_stats_vbr` (tier 1c) | `aom-encode/tests/ratectrl_pick_diff.rs` |

**None of it is wired into the encoder yet.** These are the kernels, searches
and decisions the inter RD brain (W6) will call; the brain itself is still
absent.

**W7 and W8 are now COMPLETE at the function level**, which the sentence above
used to deny. Measured 2026-08-31 against the two files' full definition lists:

| C file | definitions | ported | out of scope | missing |
|---|---|---|---|---|
| `encoder/encode_strategy.c` | 25 | 20 | 5 | **0** |
| `encoder/ratectrl.c` | 87 | 51 | 36 | **0** |

"Out of scope" is per-function and reasoned, not a bucket: for
`encode_strategy.c` it is `av1_encode_strategy`, `denoise_and_encode` and
`choose_frame_source` (pipeline and lookahead-ring orchestration the port
replaces rather than translates) plus `dump_one_image` /
`dump_ref_frame_images`, which are behind `#define DUMP_REF_FRAME_IMAGES 0`
and are not compiled at all. For `ratectrl.c` it is the CBR, VBR, real-time,
SVC and drop-frame arms — the encode target is `--end-usage=q` — plus three
functions behind macros that evaluate to 0 in this build
(`rc_pick_q_and_bounds_no_stats_cq` under `USE_UNRESTRICTED_Q_IN_CQ_MODE`,
`get_q_passive_strategy` under `RT_PASSIVE_STRATEGY`, and the
`CONFIG_FPMT_TEST` shadow-state blocks). Every ported row's module names its C
source and states its evidence tier; every unported one carries a one-line
reason in the module docs.

Caveat on the count, stated because the tool that produced the denominator
says so: `tools/c_surface_inventory.py`'s regex only sees single-line,
column-0 definitions, so 25 and 87 are LOWER BOUNDS on the two files' real
surface. "0 missing" means "none of the definitions the tool can see", not
"the file is provably exhausted".

### Verified on two ISAs, and what CI shows

Every differential above passes on **both** `aarch64-apple-darwin` and
`x86_64-apple-darwin` (the latter via the target-aware `build.rs` plus Rosetta;
see `docs/DIFFERENTIAL_PLAYBOOK.md` §3). That second ISA earned its keep
immediately: it found two shim defects that had been green on aarch64 for a day
(§"Findings" below), and one of them is the SAME defect CI's x86-64 job was
failing on. Every CI run between the `comp_pred_shim` commit and the
`make the oracle cross-buildable` fix fails x86-64 with a single
`inter_pred_enc_diff` SIGABRT — one cause, not a run of independent failures.
CI is roughly two hours backlogged and has not yet reached the fix commit, so
that confirmation is still pending; the fix is verified locally on both ISAs.

### What the re-run inventory says — and the tool it says it with

`tools/c_surface_inventory.py` matched a C name **anywhere in the concatenated
Rust tree** until 2026-08-31, doc comments and oracle shims included. That is
not a small bias: a module whose docs list every function in its C file — the
ported ones AND the ones explicitly named as gaps — scored as fully ported.
Measured on `compound_type.c` the day this was found: **34/34 matched, with 9
of those 34 named in the same file as NOT ported.** `av1_build_prediction_by_above_preds`
and `_left_preds` were likewise credited, from a doc comment in `aom-decode`
naming the DECODER's `dec_build_prediction_by_*`.

The tool now matches a Rust `fn <name>` **defined in the port's own source**,
excluding `crates/aom-sys-ref` (the oracle — every `ref_*` wrapper and
`extern` block names the C function it drives), `tests/`, `benches/`,
`examples/`, and every `.c` / `.h` file. It also strips libaom's `_c`
reference-implementation suffix, since `aom_upsampled_pred_c` is ported as
`upsampled_pred`.

**Post-fix, the whole inter-encode scope reads 441 matched of 1491, 1050
unmatched, 296 of those with an exported symbol.** That is a much smaller
number than the 605 the old matcher reported, and it is the honest one. Read
it as a floor in the other direction now: a port that RENAMED a function no
longer matches, and several deliberately did (`backup_stats` is
`CompTypeCosts::backup`, `update_best_info` is `BestCompTypeStats::update`,
`update_mbmi_for_compound_type` is `CompoundType::comp_group_idx` /
`compound_idx`), so `compound_type.c` reads 22/34 where the true count is
25/34. A MISSING row is a work item to triage, not a proven absence — which
is what this map always claimed, and now is.

### Findings worth carrying forward

Three C behaviours found while porting that a later reader is likely to
"correct" back:

1. **`estimate_obmc_mvcost` (mcomp.c:3714) does not agree with `mv_err_cost_`.**
   It shifts by 13 with a `+4096` bias instead of 14 with `+8192`, and applies
   `GET_MV_SUBPEL` (a x8 multiply) to a difference already in 1/8-pel units.
   libaom's own TODO flags it. Rewriting it fails the differential.
2. **`setup_obmc_center_error` (mcomp.c:3683) scores the reference at its
   buffer ORIGIN, ignoring `start_mv`.** libaom's own TODO flags this too.
   "Fixing" it fails the differential.
3. **`warp_error` (global_motion.c:276) clips each cell against the REFERENCE
   size, not the walked extent**, and computes it in `int` — a cell past the
   reference edge gets a negative extent that makes both its loops empty, so it
   contributes zero rather than being skipped. A `usize` transcription
   underflows there.
4. **`av1_dropout_qcoeff_num` is undefined at `dropout_num_after == 0`** — C
   indexes `scan[-1]` there. Its only caller never passes less than 32. The
   port asserts the contract rather than reproducing the UB.
5. **`estimate_obmc_mvcost` and `setup_obmc_center_error` both carry upstream
   TODO-flagged bugs** (a mismatched cost shift with a spurious `GET_MV_SUBPEL`,
   and a centre error read at the buffer origin instead of at `start_mv`).
   Both are reproduced verbatim and both fail the differential if "fixed".

Three more live in `docs/DIFFERENTIAL_PLAYBOOK.md` §3a rather than here,
because they are properties of the ORACLE rather than of any ported kernel:
the `-DNDEBUG` ABI requirement, RTCD pointers that are null one frame below a
`T` symbol, and the alignment+size contract for buffers a dispatched kernel
writes.
