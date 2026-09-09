//! Consolidated integration-test harness for `aom-encode`.
//!
//! Every `tests/*.rs` here is a MODULE of one binary rather than a binary of
//! its own. Cargo makes one test target per `tests/*.rs`, and each target
//! statically links the whole port plus libaom.a (~22 MB a piece, measured), so
//! N files cost N near-identical link steps. Folding them into one target
//! replaces that with one link, and lets `cargo test`'s intra-binary thread
//! pool run ALL of these concurrently -- it runs separate binaries one after
//! another, which is why the suite left this 24-core box ~84 % idle.
//!
//! To run one module's tests: `cargo test --test all -- <module>::`.
//!
//! Files kept OUT of here, each for a stated reason, are listed in the crate's
//! tests/ directory alongside this one.

mod common;
mod avg_4x4_diff;
mod avif_parity;
mod cfl_alpha_search_diff;
mod cnn_partition_cnn_diff;
mod cnn_partition_decision_diff;
mod cnn_partition_nn_diff;
mod coeff_costs_fill_diff;
mod color_description;
mod compound_type_diff;
mod configuration_support;
mod curvfit_diff;
mod decode_diff_ab_probe;
mod decode_diff_multisb;
mod decode_diff_noise_case;
mod deltaq_cast_semantics_diff;
mod deltaq_perceptual_ai_diff;
mod deltaq_perceptual_wavelet_diff;
mod denoise_and_model_diff;
mod dist_tx_domain_diff;
mod enc_misc_diff;
mod encode_block_coeffs_diff;
mod encode_block_full_diff;
mod encode_cancel;
mod encode_coding_block_diff;
mod encode_fuzz_sweep;
mod encode_intra_plane_diff;
mod encode_intra_plane_uv_diff;
mod encoder_gate_bd10_diff;
mod encoder_gate_chroma_ss_e2e;
mod encoder_gate_e2e_byte_match;
mod encoder_gate_multitile;
mod encoder_gate_tune_iq_e2e;
mod encode_sb_diff;
mod firstpass_diff;
mod flat_block_finder_diff;
mod frame_header_matches_real_encoder;
mod frame_source_diff;
mod full_pixel_search_diff;
mod fwht4x4_diff;
mod global_motion_diff;
mod grain_table_diff;
mod hog_prune_diff;
mod inter_fullpel_diff;
mod inter_pred_enc_diff;
mod intra_avail_partition_diff;
mod intra_mode_cost_diff;
mod intra_model_rd_diff;
mod intra_prune_diff;
mod intra_rd_pick_diff;
mod intra_sbuv_mode_loop_diff;
mod intra_sby_candidates_diff;
mod intra_sby_mode_loop_diff;
mod intra_tx_nn_diff;
mod intra_variance_factor_diff;
mod kb11_speed7_noise_localize;
mod kb21_qm_satd_arm_lock;
mod kb4_bd10_rd_localize;
mod kb4_txb2_probe;
mod kb4_txb_tie_probe;
mod kb5_lossless_block_roundtrip;
mod kb5_lossless_localize;
mod kb6_real_rd_localize;
mod kb7_rd_localize;
mod min_max_q_diff;
mod model_rd_diff;
mod nmv_cost_table_diff;
mod noise_fft_diff;
mod noise_model_diff;
mod noise_strength_solver_diff;
mod nonrd_block_yrd_hbd_diff;
mod nonrd_block_yrd_lp_diff;
mod nonrd_idtx_diff;
mod nonrd_inter_diff;
mod obu_assemble_multitile_diff;
mod oracle_contract;
mod pack_tile_roundtrip;
mod palette_kmeans_diff;
mod part4_old_nn_diff;
mod partition_none_split_diff;
mod partition_pick_diff;
mod pass2_model_diff;
mod pixel_distortion_diff;
mod prune_tx_2d_diff;
mod qindex_from_cq_diff;
mod qm_encode_witness;
mod qm_forward_block_diff;
mod quant_setup_diff;
mod ratectrl_init_diff;
mod ratectrl_pick_diff;
mod ratectrl_q_diff;
mod ratectrl_rate_diff;
mod ratectrl_update_diff;
mod rate_model_diff;
mod rdcost_diff;
mod rd_mult_diff;
mod rdopt_gate_diff;
mod rdopt_model_diff;
mod rdopt_mv_diff;
mod rdopt_obmc_diff;
mod rdopt_single_state_diff;
mod rdopt_skip_diff;
mod rdopt_sse_diff;
mod rdopt_var_rd_diff;
mod rd_pick_intra_sb_diff;
mod rd_thresh_diff;
mod read_coding_block_diff;
mod reconstruct_txb_diff;
mod ref_gop_diff;
mod refusal_census;
mod resize_opt_scaler_diff;
mod resize_plane_diff;
mod resize_plane_highbd_diff;
mod search_tx_type_diff;
mod self_contained_key_frame;
mod seq_header_matches_real_encoder;
mod seq_level_idx_diff;
mod single_motion_search_composition;
mod standalone_avif_parse_readback;
mod subpel_params_diff;
mod subpel_tree_diff;
mod superres_select_diff;
mod temporal_filter_diff;
mod temporal_filter_static_diff;
mod tile_group_obu_matches_real_encoder;
mod tpl_model_diff;
mod txb_rd_cost_diff;
mod txfm_uvrd_diff;
mod tx_mask_diff;
mod tx_size_cost_diff;
mod tx_split_nn_diff;
mod uniform_txfm_yrd_diff;
mod upsampled_pred_diff;
mod uv_cost_mask_diff;
mod var_part_inter_diff;
mod var_tx_leaf_diff;
mod var_tx_recursion_diff;
mod wedge_from_buf_diff;
mod wiener_denoise_diff;
mod xform_quant_diff;
mod xform_quant_optimize_diff;
mod xform_quant_optimize_highbd_diff;
