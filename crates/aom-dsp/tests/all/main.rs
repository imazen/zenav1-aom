//! Consolidated integration-test harness for `aom-dsp`.
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

mod dispatch_serial;
mod avail_diff;
mod block_error_diff;
mod block_error_qm_diff;
mod build_dir_diff;
mod build_filter_intra_diff;
mod build_nd_diff;
mod build_quantizer_diff;
mod cdef_diff;
mod cdef_filter_diff;
mod cdef_filter_simd_diff;
mod cdef_find_dir_simd_diff;
mod cdef_frame_diff;
mod cdef_lowbd_diff;
mod cdef_lowbd_simd_diff;
mod cdf_diff;
mod cfl_cdiff;
mod cfl_vectors;
mod compound_convolve_diff;
mod compound_diff;
mod convolve_diff;
mod convolve_scale_diff;
mod cost_coeffs_diff;
mod dc_quant_diff;
mod dec_facades_cdiff;
mod default_cdfs_diff;
mod dequant_txb_diff;
mod dir_diff;
mod dir_highbd_diff;
mod dir_simd_diff;
mod dist_diff;
mod dr_predict_high_diff;
mod dv_ref_diff;
mod edge_diff;
mod entropy_ctx_diff;
mod entropy_diff;
mod ext_tx_diff;
mod fdct_diff;
mod fill_diff;
mod filter_intra_diff;
mod frame_walk_diff;
mod hadamard_diff;
mod hbd_dist_diff;
mod hbd_lpf_diff;
mod hbd_variance_simd_diff;
mod header_diff;
mod highbd_diff;
mod highbd_hadamard_diff;
mod highbd_quant_diff;
mod interintra_diff;
mod interp_filter_params_diff;
mod inter_pred_diff;
mod intra_avail_diff;
mod intra_diff;
mod intra_lowbd_diff;
mod intra_simd_diff;
mod inv_txfm1d_diff;
mod inv_txfm2d_diff;
mod inv_txfm2d_lowbd_diff;
mod inv_txfm2d_u8_simd_diff;
mod inv_txfm_decodable_pairs;
mod kernels_diff;
mod leb128_diff;
mod lf_apply_diff;
mod loopfilter_lowbd_diff;
mod lpf_diff;
mod lpf_simd_diff;
mod lr_read_diff;
mod lr_write_diff;
mod obmc_dist_diff;
mod obu_diff;
mod optimize_diff;
mod optimize_qm_diff;
mod partition_diff;
mod pick_diff;
mod pick_search;
mod predict_intra_diff;
mod predict_intra_in_place_diff;
mod prob_cost_diff;
mod qm_fwd_select_diff;
mod qm_inv_select_diff;
mod qm_level_diff;
mod quantize_b_adaptive_diff;
mod quantize_b_diff;
mod quantize_dc_diff;
mod quantize_fp_diff;
mod quantize_fp_simd_diff;
mod quantize_qm_diff;
mod rb_diff;
mod read_coeffs_diff;
mod read_txb_full_diff;
mod read_tx_type_diff;
mod recon_lowbd_diff;
mod sad_simd;
mod scale_diff;
mod set_q_index_diff;
mod subtract_diff;
mod sum_squares_diff;
mod trellis_cost_diff;
mod txb_diff;
mod txb_init_levels_simd_diff;
mod txfm1d_diff;
mod txfm2d_diff;
mod txfm2d_simd_perm_diff;
mod tx_size_ctx;
mod tx_type_cost_diff;
mod vector_var_diff;
mod warp_diff;
mod warp_highbd_diff;
mod wb_diff;
mod wiener_simd_diff;
mod write_coeffs_diff;
mod write_txb_full_diff;
