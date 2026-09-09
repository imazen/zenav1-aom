//! Consolidated integration-test harness for `aom-bench`.
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

mod armed_tools_decode_gate;
mod cancel_latency;
mod cnn_cache_identity;
mod config_permutations;
mod delta_lf_mode_e2e;
mod deltaq_mode2_e2e;
mod deltaq_mode3_e2e;
mod deltaq_nonrd_speed;
mod encode_perf_vs_libaom;
mod encoder_gate_cdef_e2e;
mod encoder_gate_superres_e2e;
mod film_grain_gate;
mod highbd_inter_decode_envelope;
mod inter_e2e_search;
mod inter_harness_chunk0;
mod inter_header_derive_diff;
mod inter_pack_tile_diff;
mod inter_rc_qindex_diff;
mod kb13_cpu3_cq63;
mod kb19_min_partition_4k;
mod kb21_qm_speed4;
mod kb22_hd_arms;
mod kb28_crop_dims;
mod kb31_deltaq_multitile;
mod kb31_mandatory_tiles;
mod kb32_nonrd_size_bands;
mod kb34_nonsquare_nonrd_leaf;
mod kb35_nonrd_palette_arm;
mod kb36_above_720p_speed_axis;
mod kb37_nonrd_palette_search;
mod kb41_screen_detected_defaults;
mod kb5_lossless_speed_axis;
mod lr_default_parity;
mod lr_restoration_gate;
mod rd_close_harness;
mod rd_close_intrabc;
mod rd_close_palette;
mod s4cov_crop_format_axis;
mod s4cov_hd_format_axis;
mod s4cov_hd_speed_axis;
mod s4cov_partial_sb_axis;
mod s4cov_qm_axis;
mod sb128_e2e;
mod svt_interop_decode_gate;
mod toggles_rd_close;
mod tx_stats_prune_e2e;
