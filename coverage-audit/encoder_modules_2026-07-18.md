# aom-rs encoder module coverage audit — single-frame / ALLINTRA / KEY

**Reference:** libaom v3.14.1 (`03087864`). **Audit date:** 2026-07-18. **Author:** aom-rs.
**Mission scope:** single-frame / ALLINTRA (usage=2) / KEY-frame encode. Inter/motion/TPL/GOP/rate-control-beyond-fixed-Q is **OUT-OF-SCOPE-inter** (enumerated for completeness, not counted as a gap).

This is a REPORT-ONLY, evidence-based map of the libaom **encoder algorithm/module surface** (the internal encode tech, not CLI knobs) versus the aom-rs port. Every row was checked against source — the C in `reference/libaom/av1/…` and the Rust in `/root/aom-rs/crates/…`. Where the in-repo docs (STATUS.md "Next candidates"/"Gate posture", checklist.json "TODO" notes) are stale relative to landed work, the source + PARITY.md §A + the dated STATUS sections + the actual differential/gate tests were used as ground truth.

## Class legend

- **BYTE-EXACT** — ported AND proven byte-identical vs the C oracle (differential test or e2e byte-match gate cited).
- **PARTIAL** — core ported; some arms / speeds / bit-depths / sizes absent (named in the row).
- **ABSENT** — no port for the single-frame path.
- **OUT-OF-SCOPE-inter** — inter/motion/TPL/GOP-group/RC — not a single-frame gap.
- **N/A-decode** — decode-side only (not encoder).

## Gate posture (context)

The port's own search+pack pipeline produces a **byte-identical AV1 tile-group payload** vs real `aomenc` across a large envelope (PARITY.md §A): ALLINTRA KEY speed **cpu-used 0-9** (synthetic grids 64/64 each; speed-0 also on **real decoded content 30/30**), bit depths **8/10/12**, chroma **mono / 4:2:0 / 4:2:2 / 4:4:4**, **multi-tile**, **coded-lossless** (mono+420), **QM-on**, **CDEF-strength search**, **loop-restoration search**, **tune=IQ/SSIMULACRA2**, **superres FIXED**, **film-grain table-inject**, **`--deltaq-mode=3/6`**, and the **C8/C9/C10/C11 CLI-toggle** disable arms. The auto-derived CLI-feature coverage ledger still reads **0/349** because the frame HEADER remains partly bootstrapped from the real parse (see §F) — no *complete* CLI feature yet flows end-to-end through a fully self-derived header. The four project gates (full-corpus correctness, ≤1.20× perf, full coverage, zenavif parity) are not yet satisfied as whole-corpus gates.

---

## §A — Low-level DSP kernel layer (encode-critical primitives)

These are the shared DSP primitives the encode search/recon composes. All BYTE-EXACT with a differential-fuzz test vs the exported C `_c` function. (Inverse transform + deblock/CDEF/LR apply are decode-shared but are used by the encoder's reconstruction + two-pass filter search, so they are encode-critical here.)

| Subsystem | C file:fn | CLASS | Evidence / gap |
|---|---|---|---|
| Forward 1-D transforms (fdct{4,8,16,32,64}, fadst{4,8,16}, fidentity{4,8,16,32}) | av1_fwd_txfm1d.c | BYTE-EXACT | `aom-transform/src/txfm1d_gen.rs`,`fdct.rs`; `txfm1d_diff.rs` (4.8M cmp) |
| Forward 2-D transform, all 19 sizes × 193 (type×size) combos incl. 64-pt repack + rect Sqrt2 | av1_fwd_txfm2d.c `av1_fwd_txfm2d_*` | BYTE-EXACT | `aom-transform/src/txfm2d.rs`; `txfm2d_diff.rs` (~386k) + SIMD `txfm2d_simd_perm_diff.rs` |
| Forward WHT (coded-lossless TX_4X4) | hybrid_fwd_txfm.c / `av1_fwht4x4_c` | BYTE-EXACT | `aom-transform`; `fwht4x4_diff` (KB-5) |
| Inverse 1-D/2-D transform + reconstruction (bd 8/10/12) — encoder recon | av1_inv_txfm1d.c / av1_inv_txfm2d.c | BYTE-EXACT | `aom-transform/src/inv_txfm2d.rs`; `inv_txfm1d_diff.rs`,`inv_txfm2d_diff.rs` (~405k) |
| Quantizers FP / B / DC × lowbd/highbd × flat/QM | av1_quantize.c, aom_dsp/quantize.c | BYTE-EXACT | `aom-quant/src/{lib,qm}.rs`; `quantize_{fp,b,dc}_diff`,`highbd_quant_diff`,`quantize_qm_diff`. **Adaptive `_adaptive`/`quant_b_adapt`: ABSENT** |
| Quantizer build / dequant / set_q_index / sharpness bias | av1_build_quantizer, av1_set_quantizer | BYTE-EXACT | `aom-quant/src/build_quantizer.rs`; `build_quantizer_diff`,`set_q_index_diff` |
| QM fwd/inv weight tables + level select | aom_qm tables, `aom_get_qmlevel*` | BYTE-EXACT | `aom-quant/src/{qm_fwd_tables,qm_inv_tables}.rs`; `qm_fwd_select_diff`,`qm_inv_select_diff`,`qm_level_diff` |
| Daala range coder od_ec enc+dec | aom_dsp/entenc.c, entdec.c | BYTE-EXACT | `aom-entropy/src/{enc,dec}.rs`; `entropy_diff.rs` (~40k seqs) |
| CDF adaptation + aom_write/read_symbol | entropymode / `update_cdf` | BYTE-EXACT | `aom-entropy/src/cdf.rs`; `cdf_diff.rs` (1M updates) |
| Default CDF tables (per-qindex bands) | av1/common/entropy* defaults | BYTE-EXACT | `aom-entropy/src/default_cdfs.rs`; `default_cdfs_diff.rs` |
| Intra predictors non-directional lowbd+highbd (DC*/V/H/Paeth/Smooth*) | aom_dsp/intrapred.c | BYTE-EXACT | `aom-intra/src/{lib,weights}.rs`; `intra_diff.rs`,`highbd_diff.rs` (380k) |
| Intra predictors directional z1/z2/z3 lowbd+highbd | reconintra.c `av1_dr_prediction_z{1,2,3}` | BYTE-EXACT | `aom-intra/src/dir.rs`; `dir_diff.rs`,`dir_highbd_diff.rs`,`dr_predict_high_diff.rs` |
| Intra edge filter/upsample + strength lowbd+highbd | reconintra.c `av1_filter_intra_edge`/`_upsample` | BYTE-EXACT | `aom-intra/src/edge.rs`; `edge_diff.rs` |
| Filter-intra predictor (5 modes) | reconintra.c `av1_filter_intra_predictor` | BYTE-EXACT | `aom-intra`; `filter_intra_diff.rs`,`build_filter_intra_diff.rs` |
| CfL predictor + luma subsample | cfl.c `av1_cfl_predict`/`cfl_subsample` | BYTE-EXACT | `aom-intra/src/cfl.rs`; `cfl_cdiff.rs`,`cfl_vectors.rs` |
| Distortion: SAD / variance / sub-pel var, lowbd+highbd, 22 sizes (+ avg/masked/obmc SAD) | aom_dsp/sad.c, variance.c, sad_av1.c | BYTE-EXACT | `aom-dist/src/{lib,simd,simd_variance}.rs`; `dist_diff.rs`,`hbd_dist_diff.rs` (~495k) |
| Hadamard 4/8/16/32 + SATD, lowbd+highbd | aom_dsp/avg.c | BYTE-EXACT | `aom-dist/src/hadamard.rs`; `hadamard_diff.rs`,`highbd_hadamard_diff.rs` |
| Transform-domain distortion: block_error(+qm) / subtract / sum_squares | av1_block_error_c, aom_subtract_block_c | BYTE-EXACT | `aom-dist`; `block_error_diff`,`block_error_qm_diff`,`subtract_diff`,`sum_squares_diff` |
| CDEF find_dir + filter block, lowbd+highbd | cdef_block.c | BYTE-EXACT | `aom-cdef/src/{lib,frame,simd}.rs`; `cdef_diff`,`cdef_filter_diff`,`cdef_frame_diff` |
| Deblock filter apply, lowbd+highbd (4/6/8/14 h+v) | aom_dsp/loopfilter.c | BYTE-EXACT | `aom-loopfilter/src/{frame,highbd}.rs`; `lpf_diff`,`hbd_lpf_diff`,`lf_apply_diff` |
| Coeff-coding kernels: init_levels / nz_map_contexts / scan+iscan / get_txb_ctx | encodetxb.c, txb_common.*, scan.c | BYTE-EXACT | `aom-txb/src/{fill,entropy_ctx,scan}.rs`; `txb_diff`,`entropy_ctx_diff` |
| `av1_write_coeffs_txb` + `av1_write_tx_type` (full txb bitstream) | encodetxb.c | BYTE-EXACT | `aom-txb/src/write.rs`; `write_coeffs_diff`,`write_txb_full_diff` |
| `av1_cost_coeffs_txb`(+laplacian) + fill_coeff_costs + prob_cost | txb_rdopt.c | BYTE-EXACT | `aom-txb/src/{cost,prob_cost}.rs`; `cost_coeffs_diff`,`fill_diff`,`prob_cost_diff` |
| `av1_optimize_txb` coefficient trellis (+`_qm`) | txb_rdopt.c | BYTE-EXACT | `aom-txb/src/{optimize,trellis_cost}.rs`; `optimize_diff`,`optimize_qm_diff`,`trellis_cost_diff` |
| ext-tx set type/index derivation | av1_get_ext_tx_set*, av1_ext_tx_ind | BYTE-EXACT | `aom-txb/src/ext_tx.rs`; `ext_tx_diff.rs` |

## §B — Partition search + partition-level speed-features / ML-prunes

Port = `crates/aom-encode/src/partition_pick.rs` unless noted. C = `reference/libaom/av1/encoder/`.

| Subsystem | C file:fn | CLASS | Evidence / gap |
|---|---|---|---|
| DRIVER `av1_rd_pick_partition` (full RD, speed 0-6) | partition_search.c:5688 | BYTE-EXACT (synthetic) / PARTIAL (real) | `rd_pick_partition_real` (:2047). `encoder_gate_speed{0..6}_textured_allintra` 64/64. Real content diverges at speed≥1 (KB-13) |
| DRIVER `av1_rd_use_partition` (fixed VBP tree, speed 7) | partition_search.c:1764 | BYTE-EXACT | `rd_use_partition_real` (:3572). `encoder_gate_speed7_textured_allintra` 64/64 |
| DRIVER `av1_nonrd_use_partition` (single-pass, speed 8-9) | partition_search.c:2960 | PARTIAL | `nonrd_use_partition_real` (:4040). Speed-9 64/64; speed-8 60/64 (4 pinned, KB-12). 8-bit/4:2:0/mono/non-screen only |
| DRIVER `av1_choose_var_based_partitioning` KEY arm | var_based_part.c:1601 | BYTE-EXACT | `choose_var_based_partitioning_key` (var_part.rs:416); `avg_4x4_matches_real_c` |
| PARTITION_NONE @ all sizes | ps.c:4399 | BYTE-EXACT | :2353 |
| PARTITION_SPLIT @ all sizes | ps.c:4512 | BYTE-EXACT | :2470 (recursive) |
| PARTITION_HORZ / VERT @ all sizes | ps.c:3520 rectangular_partition_search | BYTE-EXACT (interior) / PARTIAL (edge) | :2599; `rd_pick_rect_partition`:1392. Sub-1 at true frame edge = `off_frame_placeholder` (interior-only) |
| PARTITION_HORZ_A/B, VERT_A/B (AB) | ps.c:3762 ab_partitions_search | BYTE-EXACT (interior) / PARTIAL (edge) | AB stage :2779; `rd_pick_ab_part`:1786. All 4 AB + nested sub-block reuse. Frame-edge partial-AB not modeled |
| PARTITION_HORZ_4 / VERT_4 (4-way) | ps.c:3919 rd_pick_4partition | BYTE-EXACT (interior) / PARTIAL (edge) | :2962/3112. Attempted only if all-4-strips-in-frame (:2983); edge partial-4-way out of scope |
| **SB128 encode path** (128×128 SB root) | ps.c:5688 (sb_size=128) | **ABSENT** (encode) | Harnesses SB-64-only (PARITY:313). Recursion structurally supports 128 but no e2e gate; decoder/entropy are SB-generic. Biggest structural hole in cluster |
| Intra CNN prune `av1_intra_mode_cnn_partition` | partition_strategy.c:142; ps.c:4278 | BYTE-EXACT | `cnn_partition/` `predict_decision`, wired :2197. `cnn_partition_decision_diff` vs real AVX2. Level 0@sp0 / `screen?0:2`@sp≥1 / `screen?1:2`@sp≥5 |
| `av1_ml_prune_rect_partition` | partition_strategy.c:1124; ps.c:4344 | OUT-OF-SCOPE-inter | Gated `!frame_is_intra_only` (ps.c:4336). Correctly not ported |
| `av1_ml_prune_4_partition` (BOTH branches) | partition_strategy.c:1326; ps.c:4194 | BYTE-EXACT | `part4_prune::predict_4partition_prune` (:3060). new-model (lvl0-2) AND old-model (lvl3=sp≥3, KB-7) + `nn_output_prec_reduce`. `part4_old_nn_decision_matches_c` |
| `ml_prune_ab_partition` (AB DNN) | partition_strategy.c:1223 | BYTE-EXACT | `ab_nn_prune::predict_ab_partition_prune` via `prune_ab_partitions` (:1755). Intra-live |
| `evaluate_ab_partition_based_on_split` | partition_strategy.c:1870/2009 | BYTE-EXACT | :1661, wired :2825 (`prune_ext_part_using_split_info>=2` = sp≥4) |
| `av1_ml_early_term_after_split` | partition_strategy.c:1017; ps.c:4328 | OUT-OF-SCOPE-inter | Gated `!frame_is_intra_only`. Correctly not ported |
| `av1_ml_predict_breakout` | partition_strategy.c:1529; ps.c:4268 | OUT-OF-SCOPE-inter | In `prune_partitions_after_none`, gated `!frame_is_intra_only`. Correctly not ported |
| `prune_ext_partition_types_search_level` (AB RD-ratio) | partition_strategy.c:1923 | BYTE-EXACT | `prune_ab_partitions` (:1709, level 1 LIVE @sp0) |
| `prune_rect_part_using_4x4_var_deviation` | speed_features.c:539; ps.c:5793 | BYTE-EXACT | :2319/2332 (sp≥6; `var_max-var_min < 3.0`) |
| `prune_rect_part_using_none_pred_mode` | ps.c:4488 | BYTE-EXACT | :2416 (sp≥6, post-NONE mode-class rect kill) |
| `prune_rectangular_split_based_on_qidx` | partition_strategy.c:1739 | BYTE-EXACT | :2164 (sp≥6, qindex-thirds) |
| `prune_sub_8x8_partition_level` | partition_strategy.c:1760 | BYTE-EXACT | :2181 (sp≥6, BLOCK_8X8 both-neighbour-larger) |
| `prune_part4_search` | speed_features.c:355/543; ps.c:4152 | BYTE-EXACT | :2973 (base 2, sp≥6→3) |
| `use_square_partition_only_threshold` | speed_features.c framesize-dep; ps.c:5749 | BYTE-EXACT | `use_square_partition_only_threshold_allintra` (:1989). 64X64@sp0 / 32X32@sp1-5 / 16X16@sp6 (KB-3 root) |
| `ext_partition_eval_thresh` | speed_features.c:510/2947 | BYTE-EXACT | `ext_partition_eval_thresh_allintra_key` (:1603). Gates AB+4-way; 128X128 (disable) @sp≥6 / sub-480p sp5 |
| `prune_ext_part_using_split_info` (4-way arm) | ps.c:4023; speed_features.c:446 | BYTE-EXACT | :3094 (sp≥3; `split_part_rect_win` threading) |
| `av1_prune_partitions_before_search` motion-rect arm | partition_strategy.c:1801 | OUT-OF-SCOPE-inter | Gated `!frame_is_intra_only && !superres_scaled`. Correctly not ported |
| `rect_partition_eval_thresh` | speed_features.c:2980 | N/A (KEY-dead) | Only setter is `!boosted`; KEY is boosted → arm never fires. Documented no-op (:2160) |
| `av1_prune_partitions_by_max_min_bsize` | partition_strategy.c:1837 | BYTE-EXACT | :2252 (`--min/max-partition-size`, PARITY C8) |
| VBP `set_vbp_thresholds` KEY | var_based_part.c:535 | BYTE-EXACT | `set_vbp_thresholds_key` (var_part.rs:247) |
| VBP `fill_variance_4x4avg` | var_based_part.c:390 | BYTE-EXACT | `avg_4x4`+`fill_variance` (var_part.rs:153) |
| VBP `set_vt_partitioning` + edge-fit stamps | var_based_part.c:149 | BYTE-EXACT | var_part.rs:292 + `get_partition_from_stamps`:686 (= C `get_partition`) |
| `setup_block_rdmult` + `intra_sb_rdmult_modifier` fold | ps.c:596/5710 | BYTE-EXACT | :289/310, folded at SB root pack.rs:1207/1614; identity @sp≥7 |
| `av1_get_cb_rdmult` (per-block AQ rdmult) | encodeframe_utils.c:93 | OUT-OF-SCOPE-inter | Only under AQ; allintra fixed-Q passes NO_AQ → frame-constant rdmult |
| `write_partition` + `partition_plane_context` | bitstream.c write_modes | BYTE-EXACT | aom-entropy/partition.rs:116/37; gather_vert/horz_alike. Proven via real_bitstream/multi-tile/KB-6 |
| `set_partition_cost_for_edge_blk` | ps.c:3411 | BYTE-EXACT | :2025 (frame-init CDF gather; KB-6 CHUNK-3) |



## §C — Intra mode search (luma + chroma), angle-delta, filter-intra, CfL, palette, IntraBC

Port = `crates/aom-encode/src/{intra_rd,intra_uv_rd,palette_search,intrabc_search,nonrd_pickmode,hog}.rs`. C = `reference/libaom/av1/encoder/`.

| Subsystem | C file:fn | CLASS | Evidence / gap |
|---|---|---|---|
| Luma DC / V / H / Paeth / Smooth / Smooth_V / Smooth_H (predict + RD-search) | intrapred.c + intra_mode_search.c | BYTE-EXACT | Predictors `aom-intra`; flat 61-candidate loop `intra_rd.rs:1038` `rd_pick_intra_sby_mode_y`; `intra_sby_mode_loop_diff.rs`; e2e cpu0-9; C10 disable pins |
| Luma D45/D135/D113/D157/D203/D67 (directional) | reconintra.c `av1_dr_prediction_z{1,2,3}` | BYTE-EXACT | `dir.rs` lowbd+highbd, `MODE_TO_ANGLE`; `dir_diff`,`dir_highbd_diff`; all 8 in the candidate loop |
| Luma mode-loop driver (61-candidate) | intra_mode_search.c:1468 `av1_rd_pick_intra_sby_mode` | BYTE-EXACT | `intra_rd.rs:1038` (flat loop, not legacy `rd_pick_intra_angle_sby`); `rd_pick_intra_sb_diff.rs` |
| Angle-delta search (±3, MAX_ANGLE_DELTA) | intra_mode_search.c:403 `set_y_mode_and_delta_angle` | BYTE-EXACT | `intra_rd.rs:299` (ANGLE_STEP=3); `use_angle_delta` gate :393; C10 `--enable-angle-delta=0` pin |
| Luma HOG prune (sad-hog) | intra_mode_search_utils.h `prune_intra_mode_with_hog` | BYTE-EXACT | `hog.rs:257` (Sobel HOG + AVX2-exact NN), wired :687 @`intra_pruning_with_hog=1`; `hog_prune_diff` |
| Filter-intra search | intra_mode_search.c:231 `rd_pick_filter_intra_sby` | BYTE-EXACT | `intra_rd.rs:1274/1478`, FILTER_INTRA_MODES=5, `prune_filter_intra_level` 0→2; `filter_intra_diff`; C10 pin. (`handle_filter_intra_mode` C:1238 is inter → OUT-OF-SCOPE) |
| CfL alpha / joint-sign search | intra_mode_search.c:745 `cfl_rd_pick_alpha` | BYTE-EXACT | `intra_uv_rd.rs:1220`, `cfl_search_range` (1@sp6/full else), full joint-sign scan; `cfl_alpha_search_diff`,`cfl_cdiff`; C10 pin |
| `is_cfl_allowed` (incl. lossless) | blockd.h | BYTE-EXACT | aom-entropy/partition.rs:2168; lossless-CfL fixed+gated (KB-5) |
| Chroma UV mode loop (all UV modes incl UV_CFL_PRED) | intra_mode_search.c:864 `av1_rd_pick_intra_sbuv_mode` | BYTE-EXACT | `intra_uv_rd.rs:1594`; `intra_sbuv_mode_loop_diff`. **Exception `--use-intra-dct-only=1` chroma PINNED-OPEN** (PARITY §B) |
| Chroma angle-delta search | intra_mode_search.c:531 `rd_pick_intra_angle_sbuv` | BYTE-EXACT | `intra_uv_rd.rs:1500` (even-pass then odd-pass rd_thresh prune) |
| Chroma HOG prune | intra_mode_search.c:959 | BYTE-EXACT | `hog.rs:312`, wired :1168; force-disabled @sp≥4 (KB-7 tail) |
| `prune_chroma_modes_using_luma_winner` | intra_mode_search.c:939 | BYTE-EXACT | `intra_uv_rd.rs:1660`; live sp≥4 (KB-7/8 64/64) |
| `prune_smooth_intra_mode_for_chroma` | intra_mode_search.c:850 | PARTIAL | Consumer wired `intra_uv_rd.rs:1723` but sf OFF sp0-5; @sp6 body transcription-faithful yet UNREACHED on all gate grids (KB-10) — never gate-exercised |
| Palette Y search (k-means dim1/2, top-colors, cache, cost, recon) | palette.c `av1_rd_pick_palette_intra_sby` | BYTE-EXACT (5/7) + 2 PARTIAL | `palette_search.rs`; 5/7 hard byte pins `rd_close_palette::palette_y_rd_close_gate`; 2 cells PINNED (KB-P29 128² AB/4-way near-ties) |
| Palette UV search | palette.c `av1_rd_pick_palette_intra_sbuv` | BYTE-EXACT | `palette_search.rs:1353`, wired `intra_uv_rd.rs:1775`; same Section-B gate |
| Palette prune `prune_palette_search_level` / `prune_luma_palette_size_search_level` | intra_sf | BYTE-EXACT | `palette_search.rs:583/586`; speed-0 levels gated; sp1-5 levels wired untested-by-gate |
| Palette prune `early_term_chroma_palette_size_search` | intra_sf | PARTIAL | No named field in `palette_search.rs`; UV `do_header_rd_based_gating` may subsume but not independently verified |
| `av1_search_palette_mode` / `_luma` | intra_mode_search.c:1032/1122 | OUT-OF-SCOPE-inter | Inter-frame callers (`!frame_is_intra_only`) |
| IntraBC leaf search | rdopt.c:3427 `rd_pick_intrabc_mode_sb` | PARTIAL (skeleton, UNWIRED) | `intrabc_search.rs:1098` exists but UNWIRED (`rd_pick.rs:407` no-op → zero byte effect). Present: CRC hash, DV cost tables, predict luma/chroma (chroma byte-exact vs decoder), dv_ref (`dv_ref_diff`). ABSENT: coeff arm (SKIP-only :1325), `min(skip,coeff)`, NSTEP diamond + mesh DV search, bd>8 sse, full integration |
| IntraBC DV hash / `av1_find_ref_dv` | hash_motion.c / mvref_common.c | BYTE-EXACT (unit) | `intrabc_search.rs:278/319`; `dv_ref.rs` locked by `dv_ref_diff` — reachable only once leaf is wired |
| nonrd intra estimate arm | nonrd_pickmode.c:1582 `av1_nonrd_pick_intra_mode` | PARTIAL | `nonrd_pickmode.rs:592` DC/V/H/SMOOTH + `block_yrd` LP-Hadamard + 3 est prunes; sp9 64/64, sp8 60/64 (4 diag pinned, KB-12). ABSENT: bd10/12 arm (:602 assert-dead), lossless TX_4X4 (:467 `unimplemented!`), screen palette arm |
| `hybrid_intra_mode_search` | partition_search.c:756 | BYTE-EXACT | dispatch in `nonrd_use_partition_real`: full-RD for `bsize<16×16 && src_var≥101`, else estimate; sp8 `=2`/sp9 `=0` |
| `intra_y_mode_bsize_mask_nrd` | nonrd_pickmode.c:583 | N/A (inert on KEY) | Masks nonrd INTER path; intra fn loops directly → inert (KB-12) |
| nonrd CHROMA | nonrd_pickmode.c:1735 | BYTE-EXACT | Y-only; hard-sets `uv_mode=UV_DC_PRED`; mono+420 gate agreement confirms |
| Intra edge filter type `get_intra_edge_filter_type` | reconintra.c:974 | BYTE-EXACT | Per-block recompute (luma :744, chroma :1028, re-encode encode_sb.rs:534/643) — KB-2/KB-6/#26 fixes; `encoder_gate_444_bd8_chroma_edge_filter_witness` |
| Winner-mode `store_winner_mode_stats` | rdopt_utils.h:679 | BYTE-EXACT | `intra_rd.rs:932`; `store_winner_mode_stats_matches_c_semantics` (KB-10) |
| `multi_winner_mode_type` / `winner_mode_count_allowed` | speed_features.h | BYTE-EXACT | `WinnerModeCfg::max_winner_count`; DEFAULT=3(sp4)/FAST=2(sp5)/OFF=1(sp6) |
| `prune_winner_mode_eval_level` / `bypass_winner_mode_processing` | rdopt_utils.h:403 | BYTE-EXACT | `intra_rd.rs:890/1312`; src-var gate live @sp6 (KB-10) |
| `top_intra_model_count_allowed` / `prune_intra_y_mode` | intra_mode_search.c:459 | BYTE-EXACT | `intra_rd.rs:486/520` (4@sp0→2@sp6); `intra_prune_diff`,`intra_model_rd_diff` |
| `adapt_top_model_rd_count_using_neighbors` | intra_mode_search.c:459 | BYTE-EXACT | `intra_rd.rs:486`; live @sp6 (KB-10) |
| `prune_luma_odd_delta_angles_in_intra` | intra_mode_search.c | BYTE-EXACT | `intra_rd.rs:448` + even-first delta reorder; @sp6 (KB-10) |
| `dc_blk_pred_level` (predict_dc skip) | tx_search.c:2011 `predict_dc_only_block` | BYTE-EXACT | @sp6 `=1`, threaded `predict_skip_zero_blk_rate` (KB-10) |
| `prune_intra_tx_depths_using_nn` (8×8 NN) | tx_search.c:2823 | BYTE-EXACT | `intra_tx_nn_weights.rs`; `intra_tx_nn_diff` (4000/4000 vs real NN); live @sp6 |



## §D — Transform search + trellis + quantization + coefficient coding

Port = `crates/aom-encode/src/tx_search.rs` + `crates/aom-txb/` + `crates/aom-quant/` + `crates/aom-transform/`. C = `reference/libaom/av1/encoder/`.

| Subsystem | C file:fn | CLASS | Evidence / gap |
|---|---|---|---|
| **FORWARD TRANSFORM** | | | |
| `av1_fwd_txfm2d_*` all 19 sizes × 16 tx_types | av1_fwd_txfm2d.c `fwd_txfm2d_c` | BYTE-EXACT | `txfm2d.rs:287` (+64-pt repack); `txfm2d_diff` all valid size×type |
| Bit-depth independence of fwd txfm | bd feeds only disabled range checker | BYTE-EXACT | `txfm2d.rs:7`; e2e byte-match bd8/10/12 (`kb4_gate_bd10_bd12_*`) |
| 1-D kernels fdct/fadst/fidentity | av1_fwd_txfm1d.c | BYTE-EXACT | `fdct.rs`,`txfm1d_gen.rs`; `fdct_diff`,`txfm1d_diff` |
| `av1_fwht4x4` (lossless WHT, shared hi/lo bd) | hybrid_fwd_txfm.c `av1_fwht4x4_c` | BYTE-EXACT | `inv_txfm2d.rs:414`, routed `lib.rs:299`; `fwht4x4_diff` |
| `av1_lowbd/highbd_fwd_txfm` hybrid dispatch | hybrid_fwd_txfm.c, av1_fwd_txfm.c | BYTE-EXACT | Folded into `xform_quant` (`lib.rs:296`); e2e bd8/10/12 |
| Forward SIMD (AVX2 lane passes) | (rtcd) | BYTE-EXACT | `aom-transform/src/simd/`; `txfm2d_simd_perm_diff` (Gate-3) |
| **TX-TYPE SEARCH** | | | |
| `search_tx_type` (luma intra) | tx_search.c:2079 | BYTE-EXACT | `tx_search.rs:1047`; `search_tx_type_diff`,`uniform_txfm_yrd_diff`; e2e sp0-9 |
| `get_tx_mask` candidate set (luma+chroma intra) | tx_search.c:1776 | BYTE-EXACT | `tx_search.rs:143/274`; `tx_mask_diff` |
| `reduced_tx_set` (`--reduced-tx-type-set=1`) | get_tx_mask:1830 | BYTE-EXACT | `tx_search.rs:169`; C9 e2e `toggles_c9_*` |
| `prune_tx_type_est_rd` + `cost_coeffs_txb_laplacian` + `prune_txk_type` | tx_search.c:1317; txb_rdopt.c:718 | BYTE-EXACT | `tx_search.rs:611` + `cost.rs:227`; `cost_coeffs_diff`; live intra WINNER sp≥4 (KB-8) |
| `prune_2d_txfm_mode` PRUNE_1/2/3 factor table | tx_search.c:1071 | BYTE-EXACT | `tx_search.rs:573 PRUNE_FACTORS` |
| `winner_mode_tx_type_pruning` (=2 sp4/=3 sp6) | speed_features.c:488/551 | BYTE-EXACT | winner two-pass; sp4-9 gates 64/64 |
| `fast_intra_tx_type_search`/`use_intra_default_tx_only`/`get_default_tx_type` | blockd.h:1175 | BYTE-EXACT | `tx_search.rs:118`; C9 `--use-intra-default-tx-only=1` gate |
| `use_intra_dct_only` (`--use-intra-dct-only=1`) | get_tx_mask:1863 | PARTIAL | Luma byte-faithful; **chroma UV-loop winner diverges out-of-band 64²cq32**; PINNED-OPEN (PARITY §B). Non-default knob |
| **`prune_tx_type_using_stats`** (luma-intra tx-type stats prune) | tx_search.c:1876 (in `assert(plane==0)` arm, NOT is_inter-gated) | **ABSENT** | No port. C enables it ALLINTRA sp≥2(lvl1)/sp≥4(lvl2) but only `is_480p_or_larger` (speed_features.c:261/299). All gate frames sub-480p → never exercised. **Real hole: ≥480p KEY at cpu-used≥2** |
| `prune_tx_2D` (ML 2D-txfm prune) | tx_search.c:1541, called :1934 | OUT-OF-SCOPE-inter | Explicit `is_inter &&` gate; documented `tx_search.rs:197` |
| `prune_txk_type_separ` (num_allowed>7) | tx_search.c:1173 | OUT-OF-SCOPE-inter | Only via EXT_TX_SET_ALL16 (inter); intra caps at 7 |
| **TX-SIZE SEARCH** | | | |
| `av1_pick_uniform_tx_size_type_yrd` (intra) | tx_search.c:3628 | BYTE-EXACT | `tx_search.rs:2458`; `uniform_txfm_yrd_diff` |
| `choose_tx_size_type_from_rd` (USE_FULL_RD depth sweep) | tx_search.c:2967 | BYTE-EXACT | `tx_search.rs:2359` |
| `choose_largest_tx_size` (USE_LARGESTALL) + `--enable-tx-size-search=0` | tx_search.c:2715 | BYTE-EXACT | `tx_search.rs:2277` (KB-8 chunk 4); C9 e2e gate |
| `choose_smallest_tx_size` (ONLY_4X4 lossless) | tx_search.c:3683 | BYTE-EXACT | `tx_search.rs:2475`; KB-5 gate |
| `intra_tx_size_search_init_depth_{rect,sqr}` | speed_features.c:367/409 | BYTE-EXACT | `tx_search.rs:2317`; #25 rect-init fix |
| `av1_ml_predict_intra_tx_depth_prune` (8×8 NN) | tx_search.c:2823 | BYTE-EXACT | `tx_search.rs:2211` + `intra_tx_nn_weights.rs`; `intra_tx_nn_diff` 4000/4000 (KB-10) |
| `av1_pick_recursive_tx_size_type_yrd` (recursive tx partition) | tx_search.c:3553 (`assert(is_inter_block)`) | OUT-OF-SCOPE-inter | Intra KEY = uniform only |
| `model_based_prune_tx_search_level` | inter consumer | OUT-OF-SCOPE-inter | #27 confirmed inert on intra |
| **TRELLIS / COEFF OPTIMIZE** | | | |
| `av1_optimize_txb` (+ `_qm`), all update rules | txb_rdopt.c:333 | BYTE-EXACT | `optimize.rs:85/131`; `optimize_diff`,`optimize_qm_diff` |
| `av1_optimize_b` wrapper | encodemb.c:87 | BYTE-EXACT | `lib.rs:495 xform_quant_optimize` |
| `perform_coeff_opt` levels 0..7 | speed_features.c tables | BYTE-EXACT | stage policy fields (KB-8/9/10); `search_tx_type_intra` |
| `NO_TRELLIS_OPT`/`FINAL_PASS_TRELLIS_OPT` (`--disable-trellis-quant=1/2`) | encodemb.h:153 | BYTE-EXACT | PARITY §A `toggles_c9_trellis_quant_off`/`_final_pass_only` |
| `av1_cost_coeffs_txb` | txb_rdopt.c:682 | BYTE-EXACT | `cost.rs:107`; `cost_coeffs_diff` |
| tx-domain vs pixel-domain distortion (`tx_domain_dist_level`/`_thres_level`) | tx_search.c:1131 | BYTE-EXACT | `lib.rs:832` + `tx_search.rs:1517`; `dist_tx_domain_diff` |
| QM-weighted distortion (`use_qm_dist_metric`, tune=IQ) | txb_rdopt.c:346; tx_search.c:1150 | BYTE-EXACT | `lib.rs:855`; C4 tune=IQ gate; `block_error_qm_diff` |
| `skip_trellis_opt_based_on_satd` | tx_search.c SATD gate | BYTE-EXACT | `tx_search.rs:531` (KB-8 chunk 2) |
| **QUANTIZERS** | | | |
| `av1_quantize_fp`/_32x32/_64x64 (lowbd) + highbd | av1_quantize.c:203/567 | BYTE-EXACT | `lib.rs:54/287`; `quantize_fp_diff`,`highbd_quant_diff` |
| `aom_quantize_b` (+32/64, lowbd+highbd) | quantize.c:109/262 | BYTE-EXACT | `lib.rs:103/330`; `quantize_b_diff` |
| QM variants fp_qm / b_qm / highbd_qm | av1_quantize.c QM branches | BYTE-EXACT | `lib.rs:174/392/459/509`; `quantize_qm_diff`; #23 QM-on e2e |
| `av1_quantize_dc` (+highbd/qm) | av1_quantize.c | BYTE-EXACT | `lib.rs:555/588`; `quantize_dc_diff`,`dc_quant_diff` |
| `av1_quantize_lp` (nonrd LP estimate) | av1_quantize.c:214 | BYTE-EXACT | `nonrd_pickmode.rs:302`; sp9 e2e 64/64 (KB-12) |
| `av1_build/init/set_quantizer` + dequant + sharpness bias | av1_quantize.c:604/696/878 | BYTE-EXACT | `build_quantizer.rs`; `build_quantizer_diff`,`set_q_index_diff`,`dequant_txb_diff`; C4 `--sharpness` gate |
| **`aom_quantize_b_adaptive`** (`--quant-b-adapt`) | av1_quantize.c:311/455; quantize.c:17/174 | **ABSENT** | No port. Non-default (default 0). PARITY C9 remaining (S–M), lowbd+highbd+32/64 |
| lossless coded quant (WHT + B-no-trellis + entropy ctx) | av1_quantize.c / encodemb.c | BYTE-EXACT | KB-5 (mono+420) `encoder_gate_lossless_cq0_e2e_kb5_repro` |
| **COEFF CODING / ENTROPY CTX** | | | |
| `av1_write_coeffs_txb` + `av1_write_tx_type` | encodetxb.c:596 | BYTE-EXACT | `write.rs:60/239` + `ext_tx.rs`; `write_coeffs_diff`,`write_txb_full_diff` |
| `av1_get_txb_entropy_context` / `get_txb_ctx` | encodetxb.c / txb_common.h | BYTE-EXACT | `entropy_ctx.rs:58/107`; `entropy_ctx_diff` |
| nz_map/br context + base_eob + scan/iscan tables | txb_common.*, scan.c | BYTE-EXACT | `scan.rs`,`lib.rs`; `txb_diff` |
| `av1_set_entropy_contexts` (edge tail-zero, KB-6) | blockd.c:29 | BYTE-EXACT | `encode_sb.rs`; real-content map 30/30 |
| `av1_fill_coeff_costs` | rd.c | BYTE-EXACT | `fill.rs`; `fill_diff` |
| tokenize record-ctx (`av1_update_and_record_txb_context`) | tokenize.c / encodetxb.c | BYTE-EXACT | `encode_sb.rs:786/876` (KB-6 pack write-ctx fix) |
| `predict_dc_only_block`/`dc_blk_pred_level` (level-1 skip) | tx_search.c:2014 | BYTE-EXACT | `tx_search.rs:1068` (KB-10); level>1 is RT-only (asserted dead) |
| coeff READER `av1_read_coeffs_txb` | decodetxb.c | N/A-decode | `read.rs`; roundtrip tests only |
| **SATD FAST PATH** | | | |
| `aom_satd` | avg.c | BYTE-EXACT | `hadamard.rs:268`; `hadamard_diff` |
| `av1_quick_txfm` (WHT+SATD, intra model_rd) | intra_mode_search_utils.h | BYTE-EXACT | `tx_search.rs:2521/2554`; `hadamard_diff`,`highbd_hadamard_diff` |
| `av1_block_yrd` + LP Hadamard/SATD (nonrd sp8/9) | nonrd_opt.c:126, avg.c | BYTE-EXACT (8-bit) | `nonrd_pickmode.rs:402`; sp9 e2e 64/64. **HBD arm ABSENT** (asserted dead, `:594`) |



## §E — In-loop post-filter SEARCH (LF/CDEF/LR) + delta-Q / AQ / segmentation (decision side)

Verified allintra defaults (av1_cx_iface.c, first-hand): **CDEF OFF** (:3067 override), **QM OFF** (`enable_qm` stays 0), **Loop-Restoration ON** (`default_extra_cfg.enable_restoration=1` :286, NOT touched by the :3065 override; runtime :1273-74 keeps it on for non-realtime — this contradicts CLAUDE.md's "LR OFF by default in allintra"; PARITY C2 is correct).

| Subsystem | C file:fn | CLASS | Evidence / gap |
|---|---|---|---|
| **LOOP-FILTER LEVEL SEARCH** | | | |
| `av1_pick_filter_level` (LPF_PICK_FROM_FULL_IMAGE recon search) | picklpf.c:211 | BYTE-EXACT | `lf_search.rs:411`; header LF bytes byte-match `encoder_gate_e2e_*` (sp0-5). Dir 2→0→1 Y + U/V |
| `search_filter_level` (step-halving, bias, ss_err memo) | picklpf.c:102 | BYTE-EXACT | `lf_search.rs:304` |
| NON_DUAL vs DUAL (4 levels Y-V/Y-H/U/V) | picklpf.c:376 | BYTE-EXACT | `non_dual` flag: DUAL sp0-3, NON_DUAL sp4-5 |
| `pick_filter_level_from_q` (LPF_PICK_FROM_Q) | picklpf.c:266 | BYTE-EXACT | `lf_search.rs:483`; `speed6_prep_lf_from_q_matches_real_aomenc`; live sp6/7 |
| Adaptive-sharpness qindex cap {7,1,0} | picklpf.c:220 | BYTE-EXACT | `lf_search.rs:386`; witnessed in `encoder_gate_tune_iq_e2e` |
| twopass `get_max_filter_level`/rating bias | picklpf.c:53,158 | OUT-OF-SCOPE-inter | Two-pass stat consumption; one-pass envelope always max=63 |
| **CDEF STRENGTH SEARCH** (default OFF in allintra) | | | |
| `av1_cdef_search` FULL (speed 0) | pickcdef.c:838 | BYTE-EXACT | `pickcdef.rs:953`; `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close` 14/14 |
| Joint strength-set select + RD bits loop + per-64-unit stamp | pickcdef.c:86-224 | BYTE-EXACT | `pickcdef.rs:175-289`; mono(single)+color(dual Y+UV) both gated |
| `get_cdef_filter_strengths` FAST LVL1..5 | pickcdef.c:29 | PARTIAL | All 6 methods ported `pickcdef.rs:149` + table unit-tested; **only FULL e2e-gated** (FAST 1..5 = speeds 1-6, unexercised since CDEF off by default) |
| SB128 CDEF-on (>64 filter-block arms) | pickcdef.c:696 | PARTIAL | Search's >64-fb arms present; SB128 CDEF-on blocked on pack's SB64 envelope — not e2e-gated |
| `av1_pick_cdef_from_qp` (CDEF_PICK_FROM_Q, sp≥7 rt) | pickcdef.c:747 | ABSENT | `pickcdef.rs:954` asserts method 0..5; documented dead for cpu 0-6 |
| CDEF_ADAPTIVE arms (tune=IQ) | pickcdef.c:841-1095 | ABSENT | `--enable-cdef=1`→CDEF_ALL; composite tune gate overrides enable_cdef=0. Documented dead |
| **LOOP-RESTORATION SEARCH** (default ON in allintra) | | | |
| `av1_pick_filter_restoration` (frame walk, 256→128→64 descent, per-plane RD) | pickrst.c:2040 | BYTE-EXACT (under `--enable-restoration=1`) | `pick.rs:1922`; `lr_restoration_gate` 8/8 byte-identical + decisions==C; format axis mono/444/bd12 3/3 |
| Wiener: compute_stats[_highbd], finer_search ±{4,2,1}, decompose_sep_sym, linsolve | pickrst.c:1004,1498,1353,1145 | BYTE-EXACT | `pick.rs:101,1149,419,273`; diffed vs exported `_c` |
| SGR: search_selfguided (16 eps), calc_proj_params/get_proj_subspace, pixel_proj_error | pickrst.c:818,634,673,231 | BYTE-EXACT | `pick.rs:1598,619,692,542` |
| LR unit-info syntax writer (`write_lr_unit`) | bitstream.c | BYTE-EXACT | `aom-entropy/src/lr.rs`; byte-identical + count parity |
| **LR in the DEFAULT allintra envelope** | av1_cx_iface.c:286,1273 | **PARTIAL (default-parity gap)** | Search byte-exact but only under explicit knob. C allintra default = restoration **ON**; default `encoder_gate_e2e_*` use a restoration-**OFF** reference. A true default `aomenc --allintra` stream (restoration on, even all-NONE → different seq/frame header bits) is NOT byte-matched by the port's default path |
| LR speed 1..4 / GOOD-mode arms | speed_features.c:1164,1352 | PARTIAL | Source-verified, PINNED not gated (base encode not byte-exact at sp≥1 real content, KB-6/13); sp≥5 allintra structurally LR-off in C |
| **DELTA-Q MODES (decision side)** | | | |
| `--deltaq-mode=3` PERCEPTUAL_AI (`av1_set_mb_wiener_variance` per-8×8 intra-SATD+Weber+2-iter norm; `av1_get_sbq_perceptual_ai`; `av1_get_deltaq_offset`) | allintra_vis.c:592,743 / rd.c:466 | BYTE-EXACT (bd8) | `allintra_vis.rs:292-544`; `deltaq_mode3_e2e` 7/7 + `get_deltaq_offset_matches_c` (18432/18432) |
| ↳ deltaq-mode=3 highbd (bd10/12 FP-quant) + partial-SB + multi-tile | allintra_vis.c:592 | PARTIAL/ABSENT sub-arms | `allintra_vis.rs:546` scope: bd8/single-tile/dims×8px only. Highbd FP-quant arm + partial-edge source-border extension unported |
| `--deltaq-mode=6` VARIANCE_BOOST (`av1_get_sbq_variance_boost`, block variance, `--deltaq-strength`) | allintra_vis.c:1072 / aq_variance.c:184 | BYTE-EXACT | `allintra_vis.rs:112,155,182`; witnessed `encoder_gate_tune_iq_e2e` |
| `av1_adjust_q_from_delta_q_res` (deadzone quant to grid) | rd.c:494 | BYTE-EXACT | `allintra_vis.rs:198`; shared by mode 3 & 6 |
| `--deltaq-mode=1` OBJECTIVE | encodeframe.c:343 | N/A (inert) | TPL-gated; no TPL for a lone still → never fires |
| **`--deltaq-mode=2` PERCEPTUAL (wavelet)** (`av1_compute_q_from_energy_level_deltaq_mode`, `log_block_wavelet_energy`, `haar_ac_energy`) | encodeframe.c:330 / aq_variance.c:138 | **ABSENT** | No wavelet/haar in port. Single-frame-applicable, unported |
| `--deltaq-mode=4/5` (user-rating / HDR) | allintra_vis.c:1045 | ABSENT | Needs external rating map / HDR avg; niche |
| **`--enable-rate-guide-deltaq`** (`get_rate_guided_quantizer`) | allintra_vis.c:688 | ABSENT | Needs external rate-file plumbing |
| **`--auto-intra-tools-off`** (`automatic_intra_tools_off`, `model_rd_sse`) | allintra_vis.c:515 | ABSENT | Disables smooth/paeth/cfl/diagonal on high-Q low-q frames; unported |
| **AQ MODES (segmentation decision side)** | | | |
| **`--aq-mode=1` VARIANCE_AQ** (`av1_vaq_frame_setup`, `av1_log_block_var`) | aq_variance.c:43,251 | **ABSENT** | Only in the C oracle shim, not the port. Single-frame-applicable segmentation gap |
| **`--aq-mode=2` COMPLEXITY_AQ** (`av1_setup_in_frame_q_adj`, `av1_caq_select_segment`) | aq_complexity.c:62,130 | **ABSENT** | Unported |
| `--aq-mode=3` CYCLIC_REFRESH_AQ | aq_cyclicrefresh.c | OUT-OF-SCOPE-inter | RT/inter (`!frame_is_intra_only`, source_sad) |
| Segmentation encode `av1_choose_segmap_coding_method` + segfeature setup | segmentation.c / segmentation.h:30 | ABSENT | Only under AQ/cyclic-refresh; seg off in still-no-AQ envelope → not reached |
| `write_segment_id` (+ neg_interleave) writer | bitstream.c | BYTE-EXACT (writer) | `aom-entropy/partition.rs:1176`; present, unexercised without seg |
| **PER-SB DELTA-Q/DELTA-LF SIGNALING (writer)** | | | |
| `write_delta_qindex` (exp-Golomb per-SB) | bitstream.c | BYTE-EXACT | `partition.rs:178`; drives byte-exact deltaq 3/6 streams |
| `write_delta_q_params` per-SB dispatch | bitstream.c | PARTIAL | `partition.rs:2353`: delta-q arm byte-exact; delta-lf arm present but never exercised |
| `--delta-lf-mode` (per-SB `delta_lf_from_base` DECISION) | bitstream.c / picklpf | PARTIAL (writer only) | Writer `partition.rs:201`; decision never enabled (`delta_lf_present=false`, `lib.rs:1402`). Decision ABSENT |
| **QUANTIZER / CHROMA-DELTAQ / QMATRIX** | | | |
| `av1_set_quantizer` base_qindex (cq→qindex) | av1_quantize.c:884 | BYTE-EXACT | `rc.rs:62`; `qindex_from_cq_diff` (cq0..63 × GOOD/ALLINTRA × bd8/10/12) |
| `--enable-chroma-deltaq` (u/v dc/ac delta_q) | av1_set_quantizer | BYTE-EXACT | `header.rs:26,54`; witnessed `encoder_gate_tune_iq_e2e` |
| QM-level select (`aom_get_qmlevel_allintra`/`_luma_ssimulacra2`/`_444_chroma`) | quant_common.h | BYTE-EXACT | `quant_common.rs:264,299,331`; `qm_level_diff`+`encoder_gate_qm_on_e2e` (40 cells). QM OFF by default → default envelope inert |



## §F — Bitstream / OBU / header framing (write) + encode-frame composition + speed features + RD/mode-cost tables

Port = `crates/aom-entropy/src/{header,obu,leb128,partition,cdf,enc,lr}.rs` + `crates/aom-encode/src/{pack,obu_assemble,encode_sb,speed_features,rd,mode_costs,real_costs,lib}.rs`.

| Subsystem | C file:fn | CLASS | Evidence / gap |
|---|---|---|---|
| **SEQUENCE HEADER OBU** | | | |
| Seq-header OBU top assembly | bitstream.c `av1_write_sequence_header_obu` | BYTE-EXACT | `header.rs:1046`; `seq_header_matches_real_encoder.rs` (parse-real→re-emit→identical) + `write_sequence_header_obu_matches_real_c` |
| profile / still-picture / reduced-hdr / bitstream level | `write_profile`/`write_bitstream_level` | BYTE-EXACT | `header.rs:1037-1051` |
| timing / decoder-model info | `write_timing_info_header`/`write_decoder_model_info` | BYTE-EXACT (writer) | `header.rs:963/983`; values echoed from parse (default still: absent) |
| operating points loop (idc/level/tier/op-model) | seq OBU op loop | BYTE-EXACT (writer) | `header.rs:1062-1088`; **field values bootstrapped** (parsed), not self-derived |
| `write_sequence_header` body (tool flags, sb_size, order-hint, scc/int-mv) | `write_sequence_header` | BYTE-EXACT | `header.rs:815` |
| color_config / bitdepth / CICP / subsampling / separate_uv_delta_q | `write_color_config`/`write_bitdepth` | BYTE-EXACT (writer) | `header.rs:915/903`. **CICP fields echoed from parse** (no encoder-side derivation) |
| **FRAME HEADER OBU (KEY)** | | | |
| Frame-header top assembly | bitstream.c `write_uncompressed_header_obu` | BYTE-EXACT (writer) | `header.rs:1469`; body order matches C field-for-field; proven byte-identical in `encoder_gate_e2e_byte_match` (full frame-OBU payload) |
| prefix (frame_type/show/err-res/disable_cdf/scc/int-mv/frame-size-override/order-hint/refresh) | prefix of write_uncompressed_header_obu | BYTE-EXACT (writer) | `header.rs:1151`; show-existing early-return + KEY arms |
| frame_size / render_size / superres scale | `write_frame_size`/`write_render_size`/`write_superres_scale` | BYTE-EXACT | `header.rs:305/275/254`; superres FIXED gated (PARITY C6) |
| KEY intrabc-present bit | `allow_intrabc` | BYTE-EXACT | `header.rs:1483` |
| `write_tile_info` (max-tile + ctx-update-id + tile-size-bytes) | `write_tile_info`/`wb_write_uniform` | BYTE-EXACT | `header.rs:390/423`; multi-tile derived `obu_assemble.rs:215`; `encoder_gate_multitile_e2e` |
| quantization params | `encode_quantization` | BYTE-EXACT (writer) | `header.rs:39`; QM-on gated. **base_qindex value bootstrapped** (see gaps) |
| segmentation params | `encode_segmentation` | BYTE-EXACT (writer, off-case) | `header.rs:203`; seg off → single bit proven. Seg-map derivation ABSENT (§E) |
| delta_q / delta_lf params | `write_delta_q_params` | BYTE-EXACT | `header.rs:530`; deltaq3/6 port-derived+gated |
| loop-filter params | `encode_loopfilter` | BYTE-EXACT | `header.rs:92`; LF levels **port-derived** (`lf_search`) — the documented self-derived exception |
| CDEF params | `encode_cdef` | BYTE-EXACT | `header.rs:151`; CDEF-search gated |
| LR params | `encode_restoration_mode` | BYTE-EXACT | `header.rs:456`; LR-search gated (but default-on gap, §E) |
| tx_mode / reduced_tx_set | `write_tx_mode` | BYTE-EXACT | `header.rs:549/1384`; C9 gated |
| cdf-update flag | prefix `disable_cdf_update` | BYTE-EXACT | `header.rs:1185`; C11 gated (caught real pack bug) |
| film-grain params | `write_film_grain_params` | BYTE-EXACT | `header.rs:597`; table-inject gated. Estimation ABSENT (§G) |
| global-motion / interp-filter / ref-mvs / skip-mode | header inter fields | OUT-OF-SCOPE-inter | Gated `!intra_only` (`header.rs:1545`); not written on KEY |
| **TILE-GROUP OBU + pack_tile** | | | |
| aom range encoder (od_ec_enc) | entenc.c | BYTE-EXACT | `enc.rs` (carry-propagate + BE flush) |
| `aom_write_symbol` + allow_update_cdf gate | bitwriter.h | BYTE-EXACT | `cdf.rs:39` (gates update_cdf on `allow_update_cdf`), set per-tile `pack.rs:1039` |
| tile walk / write_modes (left-ctx + LR reset) | `write_modes`/`pack_tile` | BYTE-EXACT | `pack.rs:1018/1113` |
| write_modes_sb recursion (all 10 partition arms) | `write_modes_sb` | BYTE-EXACT | `pack.rs:572`; `partition.rs:3422` |
| write_modes_b block writer | `write_modes_b` | BYTE-EXACT | `pack.rs:275` (exact write_modes_b order) |
| tile-group header + multi-tile assembly | `write_tile_group_header`/`write_tile_obu_size`/`choose_size_bytes` | BYTE-EXACT | `header.rs:1565`; `obu_assemble.rs:143/215`; `encoder_gate_multitile_e2e` |
| **MODE-INFO SYMBOL WRITERS** | | | |
| partition / skip / intra Y kf / intra UV / angle_delta / filter_intra / cfl_alphas / tx_size(sel) / delta_qindex / delta_lflevel / cdef strength / tx_type | bitstream.c write_modes_b symbols | BYTE-EXACT | `partition.rs` (116/151/279/305/533/751/244/567/178/201/2412) + tx_type via pack ext-tx CDF; edge CDF-source quirk handled (KB-6) |
| KEY block driver `write_mb_modes_kf` (+prefix/tail) | bitstream.c | BYTE-EXACT | `partition.rs:5622/2455/2562` (seg→skip→cdef→dq/dlf→intrabc→Y→adY→UV→cfl→adUV→palette) |
| palette mode info + color-map tokens | `write_palette_mode_info`/`pack_map_tokens` | BYTE-EXACT (RD-close) | `partition.rs:1825/3567`; PARITY §B 5/7 EXACT, 2 pinned (KB-P29) |
| intrabc info writer | `write_intrabc_info` | BYTE-EXACT (writer) | `partition.rs:1198`; DV **search** skeleton unwired (§C) |
| var-tx size / inter mode / mv / ref-frames / motion-mode / interp / drl / compound | inter symbol writers | OUT-OF-SCOPE-inter | Present in `partition.rs` but inter-only |
| segment_id writer | `write_segment_id` | BYTE-EXACT (off) | `partition.rs:1176` |
| **OBU ASSEMBLY / FRAMING** | | | |
| OBU header (type/ext/size) + leb128 obu_size | `av1_write_obu_header` / `aom_uleb_encode` | BYTE-EXACT | `obu.rs:9`; `leb128.rs:11/26` (32-bit cap) |
| OBU_FRAME payload assembly (hdr+align+tg+tiles) + trailing bits | spec frame_obu() / `add_trailing_bits` | BYTE-EXACT | `obu_assemble.rs:51/80/143/215`; `header.rs:1093` |
| temporal-delimiter OBU + full TU self-assembly | encoder TU write | PARTIAL | No dedicated TD writer; encode path never self-emits a full TU — harness splices port frame-OBU into real TU (`splice_frame_obu`). All wrapping primitives exist + byte-exact |
| `--full-still-picture-hdr` / annexb / large-scale-tile | large-scale/annexb framing | ABSENT | PARITY C11 follow-up; `write_ext_tile_info` present but path ungated |
| **ENCODE COMPOSITION** | | | |
| `av1_xform_quant` + `xform_quant_optimize` | encodemb.c | BYTE-EXACT | `lib.rs:276/495`; lossless WHT arm (KB-5) |
| tx-block loop / dry-run encode / update_state / tx_type_map COPY semantics | `av1_encode_sb`/`av1_foreach_transformed_block`/`av1_update_state` | BYTE-EXACT | `encode_sb.rs:475` (`output_enabled` COPY-vs-alias, KB-4) |
| SB search+pack two-pass walk | `encode_sb_row`/`av1_encode_frame` | BYTE-EXACT (sp0-9 synthetic) / PARTIAL (real) | `pack.rs`+`partition_pick.rs`; PARITY §A cpu 0..9. **Real content sp≥1 = 24/60** (KB-13) |
| entropy-context threading (ta/tl, cul stamps, edge tail-zero) | `av1_set_entropy_contexts`/`av1_set_txb_context` | BYTE-EXACT | `encode_sb.rs` (KB-5/KB-6 roots) |
| **SPEED FEATURES** | | | |
| set_allintra framesize-independent + dependent + qindex-dependent (cpu 0-9) | `set_allintra_speed_features_*` / `av1_set_speed_features_qindex_dependent` | BYTE-EXACT (KEY-relevant arms) | `speed_features.rs:448`; each field source-attributed with inert/inter notes; inter-only/boosted arms documented inert (#27). (Doc string `:445` "only speed 0/1 modeled" is STALE — code+gates cover 0-9) |
| speed 8/9 nonrd pickmode | `use_nonrd_pick_mode`/`av1_nonrd_use_partition`/`hybrid_intra_mode_search` | BYTE-EXACT (sp9 64/64; sp8 60/64) | `nonrd_pickmode.rs`; 4 diag near-ties pinned (KB-12) |
| stage-aware tx-type/tx-size policies | `set_mode_eval_params` | BYTE-EXACT | `speed_features.rs:914/949/1040` |
| **RD INFRASTRUCTURE** | | | |
| Laplacian rate/dist model + RDCOST macros | `av1_model_rd_from_var_lapndz` / rd.h | BYTE-EXACT | `rd.rs:55/41/91/99` |
| rdmult from qindex + error/sad-per-bit + qindex→q | `av1_compute_rd_mult*` / `av1_set_error_per_bit` | BYTE-EXACT | `rd.rs:191/237/273/294/280` (kf/arf mult, bd10/12 shifts, tune-IQ weight) |
| plane quantizer / block-rdmult setup | `av1_init_plane_quantizers` | BYTE-EXACT | `rd.rs:327` |
| `av1_get_deltaq_offset` | rd.c | BYTE-EXACT | `allintra_vis.rs:37` (18432/18432 vs C) |
| **MODE-COST TABLES** | | | |
| intra mode / palette-flag / angle / filter-intra / intrabc / CfL / tx-size / partition / skip / coeff(LV_MAP) / palette-color costs | `av1_fill_mode_rates` + `av1_fill_coeff_costs` | BYTE-EXACT | `mode_costs.rs` + `real_costs.rs` (built from live frame CDFs) |
| inter/nonkf-Y mode costs | `av1_fill_mode_rates` inter slices | OUT-OF-SCOPE-inter | Degenerate fillers, unread on KEY |
| **LEVEL / TIER** | | | |
| seq level index / tier computation | level.c `av1_get_seq_level_idx` | **ABSENT (bootstrapped)** | No port; `seq_level_idx[]`/`tier[]` echoed from parsed real seq header. The one true missing header ALGORITHM (others are wiring) |



## §G — Single-frame estimation / analysis / superres / film-grain / tune / NN engine

| Subsystem | C file:fn | CLASS | Evidence / gap |
|---|---|---|---|
| Superres FIXED — source downscale | resize.c `av1_resize_plane` / `highbd_resize_plane` | BYTE-EXACT | `resize.rs` (`resize_plane`:377, `highbd_resize_plane`:669); `resize_plane_diff`,`resize_plane_highbd_diff`; `encoder_gate_superres_e2e` 13/13 bd8 + 16/16 bd10/12 |
| Superres FIXED — coded-width encode + header signal | superres_scale.c `calculate_next_superres_scale`(FIXED); bitstream.c `write_superres_scale` | BYTE-EXACT | Same superres gate; header writer bit-exact |
| **Superres AUTO/QTHRESH denom selection** | superres_scale.c `calculate_next_superres_scale`:184, `analyze_hor_freq`, `get_superres_denom_from_qindex_energy` | **ABSENT** | Zero port refs. Needs `av1_fwd_txfm2d_16x4` H_DCT energy (not wired) |
| **Superres RANDOM denom** | superres_scale.c RANDOM arm + `validate_size_scales` | **ABSENT** | No port ref |
| **Superres denom-16 optimized scaler** | resize.c `av1_resize_and_extend_frame` (8-bit even-width) | **ABSENT** | Gate asserts-OUT of this corner (`encoder_gate_superres_e2e.rs:242`) |
| Superres recode loop | superres_scale.c `av1_superres_in_recode_allowed`, `SUPERRES_AUTO_DUAL` | ABSENT (OUT-OF-SCOPE for FIXED still) | AUTO+non-SOLO+frames_to_key>1 only → never fires for FIXED KEY still |
| Film grain table-inject | aom_dsp/grain_table.c `aom_film_grain_table_read`/`_lookup` + `write_film_grain_params` | BYTE-EXACT | `grain_table.rs`; `film_grain_gate.rs` (vectors 1/2/6/15 × 420/mono/444/bd10) + no-bootstrap-leak witness |
| `--film-grain-test` vectors | grain_test_vectors.h | BYTE-EXACT (transitive) | Fixture params from real vectors; covered via table-inject gate. No dedicated knob gate (trivial follow-up) |
| Noise-model: strength solver + LUT | noise_model.c `aom_noise_strength_solver_*`, `linsolve`, `_lut_eval` | BYTE-EXACT | `noise_model.rs`; `noise_strength_solver_diff` 300/300 (all f64) |
| Noise-model: flat-block finder | noise_model.c `aom_flat_block_finder_init/extract_block/run` | PARTIAL | `noise_model.rs` `FlatBlockFinder`; `flat_block_finder_diff`. `is_flat`+features exact; top-10% sigmoid `exp` is the sole libm-sensitive step (count bit-exact) |
| **Noise-model: AR estimate + grain-param quantize** | noise_model.c `aom_noise_model_init/_update`, `ar_equation_system_solve`, `aom_noise_model_get_grain_parameters` | **ABSENT** | No port (doc-noted "remaining") |
| **Noise-model: Wiener FFT denoise** | noise_model.c `aom_wiener_denoise_2d`, `get_half_cos_window` | **ABSENT** | FFT float denoise unported |
| **Noise-model: orchestrator + wiring** | noise_model.c `aom_denoise_and_model_run` | **ABSENT** | `--denoise-noise-level`/`--enable-dnl-denoising` unported (float/FFT determinism-gated) |
| Noise estimate | av1_noise_estimate.c `av1_noise_estimate` | ABSENT (not on allintra path) | Realtime/temporal-denoiser only; not invoked on ALLINTRA KEY still |
| **tune=butteraugli** | tune_butteraugli.c `av1_setup_butteraugli_rdmult`/`_rdo` | **ABSENT** | Zero port refs |
| **tune=vmaf** | tune_vmaf.c `av1_set_mb_vmaf_rdmult_scaling`/`av1_set_vmaf_rdmult` | **ABSENT** | Zero port refs |
| Saliency map | saliency_map.c `av1_setup_saliency_map` | ABSENT | `CONFIG_SALIENCY_MAP` (non-default build) + tune=vmaf only; off in reference build |
| Blockiness | blockiness.c `av1_get_blockiness` | ABSENT (byte-inert, stats-only) | Only consumer is output-metric logging (encoder.c:4880); no bitstream impact |
| CNN (intra partition prune) | cnn.c `av1_cnn_predict_img_multi_out` | BYTE-EXACT | `cnn_partition/{cnn,nn,decision,weights}.rs`; `cnn_partition_{cnn,nn,decision}_diff`. Wired in `rd_pick_partition_real` |
| **ml.c NN inference engine** | ml.c `av1_nn_predict_c`, `av1_nn_output_prec_reduce`, `av1_nn_softmax` | BYTE-EXACT | `cnn_partition/nn.rs`, `ab_nn_prune.rs`, `part4_prune.rs`; `cnn_partition_nn_diff`,`part4_old_nn_diff`,`intra_tx_nn_diff`,`hog_prune_diff` (all vs real `av1_nn_predict_c`). Drives every wired intra prune |
| **dwt.c (Haar AC / fwd DWT)** | dwt.c `av1_fdwt8x8`, `av1_haar_ac_sad_8x8_uint8_input` | **ABSENT** | No port. Blocks the Wiener-variance perceptual deltaq (`av1_set_mb_wiener_variance` via aq_variance) — but note **`--deltaq-mode=3` IS byte-exact** (see §E; that path uses intra-SATD, not dwt). dwt-Haar is single-frame-relevant to the un-ported Wiener-variance AQ arm |
| hash.c (CRC-32C) | hash.c `av1_get_crc32c_value` | BYTE-EXACT | `intrabc_search.rs` `Crc32c` — intrabc DV block hash. (tx_search mb_rd_record cache is byte-inert, not ported) |
| external_partition.c | external_partition.c `av1_ext_part_*` | ABSENT | Research/test plug-in (`AV1E_SET_EXTERNAL_PARTITION`); not in default encode |
| sparse_linear_solver.c | conjugate-gradient sparse | OUT-OF-SCOPE-inter | Sole consumer optical_flow.c (inter) |
| wedge_utils.c | `av1_wedge_sse_from_residuals` | OUT-OF-SCOPE-inter | Sole consumer compound_type.c (inter) |



## §H — Inter / motion / TPL / GOP / rate-control file enumeration (OUT-OF-SCOPE-inter)

Enumerated for completeness. Each file is OUT-OF-SCOPE-inter; the note flags any sub-piece that touches the single-frame KEY path (mostly intrabc, which reuses motion/hash/mv machinery).

| C file | CLASS | Single-frame-relevant sub-piece (if any) |
|---|---|---|
| mcomp.c | OUT-OF-SCOPE-inter | `av1_intrabc_hash_search`:1908 IS single-frame-relevant → ported in `intrabc_search.rs` (intrabc DV search) |
| motion_search_facade.c | OUT-OF-SCOPE-inter | Inter motion orchestration; nothing on KEY path |
| interp_search.c | OUT-OF-SCOPE-inter | Inter interpolation-filter search |
| reconinter_enc.c | OUT-OF-SCOPE-inter | Inter prediction recon (build_inter_predictors) |
| tpl_model.c | OUT-OF-SCOPE-inter | TPL lookahead RD; no KEY-still use |
| gop_structure.c | OUT-OF-SCOPE-inter | GOP/pyramid structure |
| pass2_strategy.c | OUT-OF-SCOPE-inter | 2-pass rate strategy |
| firstpass.c | OUT-OF-SCOPE-inter | First-pass stats (uses dwt) |
| ratectrl.c | OUT-OF-SCOPE-inter | Fixed-Q `av1_rc_pick_q_and_bounds`/qindex-from-cq IS single-frame-relevant → ported (`rc.rs`, `qindex_from_cq_diff`). RC-beyond-fixed-Q OUT-OF-SCOPE |
| av1_ext_ratectrl.c | OUT-OF-SCOPE-inter | External RC callback API |
| thirdpass.c | OUT-OF-SCOPE-inter | 3-pass/tpl-from-file |
| lookahead.c | OUT-OF-SCOPE-inter | Lookahead source buffer; trivial single-frame passthrough |
| temporal_filter.c | OUT-OF-SCOPE-inter | Alt-ref temporal filtering |
| global_motion.c / global_motion_facade.c | OUT-OF-SCOPE-inter | Global-motion estimation (inter) |
| encodemv.c | OUT-OF-SCOPE-inter | `av1_encode_dv`:276 + `av1_build_nmv_cost_table`:294 ARE single-frame-relevant → intrabc DV coding via pack + `fill_dv_costs` in `intrabc_search.rs` |
| mv_prec.c | OUT-OF-SCOPE-inter | intrabc forces integer MV (`cur_frame_force_integer_mv`) — trivial touch |
| compound_type.c | OUT-OF-SCOPE-inter | Compound/wedge (uses wedge_utils) |
| nonrd_opt.c | OUT-OF-SCOPE-inter (core) | nonrd shared infra IS single-frame-relevant → `av1_nonrd_pick_intra_mode`/`av1_nonrd_use_partition`/`hybrid_intra_mode_search` ported for sp8/9 allintra (KB-12) |
| optical_flow.c | OUT-OF-SCOPE-inter | OPFL (uses sparse_linear_solver) |
| svc_layercontext.c | OUT-OF-SCOPE-inter | SVC layer context |
| av1_temporal_denoiser.c | OUT-OF-SCOPE-inter | Realtime temporal denoiser |
| hash_motion.c | OUT-OF-SCOPE-inter | intrabc block hash IS single-frame-relevant → `av1_hash_table_init`/`av1_get_block_hash_value` ported in `intrabc_search.rs` |
| aom-convolve (av1_convolve_{x,y,2d}_sr EIGHTTAP, lowbd) | OUT-OF-SCOPE-inter | Inter-pred motion-comp DSP; not on KEY intra path (kernel byte-exact `convolve_diff`, but only SR/lowbd; dist_wtd/4-tap/highbd absent) |



---

## Rollup — actionable single-frame gaps

### ABSENT (single-frame encode-tech with no port)

1. **SB128 encode path** (§B) — the RD partition search + pack are SB-64-only; no e2e byte gate for a 128×128 SB root / 128→64 split / >64-block recon interleave. Decoder + entropy are already SB-generic. **Biggest structural hole.** (`--sb-size=128`)
2. **`prune_tx_type_using_stats`** (§D) — luma-intra tx-type stats prune, enabled by C for ALLINTRA at cpu-used≥2 but only `is_480p_or_larger`. All gate frames are sub-480p, so it is un-ported AND un-exercised — a real hole for a **≥480p KEY frame at cpu-used≥2**.
3. **`aom_quantize_b_adaptive` / `--quant-b-adapt`** (§D) — adaptive dead-zone quantizer family (lowbd+highbd+32/64), no port. Non-default.
4. **IntraBC leaf search** (§C) — `rd_pick_intrabc_mode_sb` is a PARTIAL skeleton but currently **UNWIRED** (zero byte effect); the coeff arm, `min(skip,coeff)`, NSTEP+mesh full-pel DV search, and bd>8 sse are ABSENT. Under `--enable-intrabc` (screen-content default-on). Hash/DV-cost/predictors/dv_ref are unit-byte-exact.
5. **nonrd (cpu-used 8/9) HBD estimate arm + lossless TX_4X4 + screen-palette arm** (§C/§D) — `block_yrd`/`av1_nonrd_pick_intra_mode` are 8-bit-non-lossless-non-screen only; bd10/12 + lossless + palette assert-dead.
6. **Superres AUTO/QTHRESH/RANDOM denom selection + denom-16 optimized scaler** (§G) — only FIXED denom is byte-exact. AUTO needs `analyze_hor_freq` (16×4 H_DCT, not wired) + `get_superres_denom_from_qindex_energy`.
7. **Film-grain noise-model ESTIMATION** (§G, `--denoise-noise-level`) — AR noise model, grain-param quantize, Wiener FFT denoise, and the `aom_denoise_and_model_run` orchestrator are ABSENT (float/FFT-determinism-gated). The noise-strength solver + flat-block finder ARE present/differential.
8. **tune=butteraugli, tune=vmaf** (§G) — zero port refs.
9. **`--deltaq-mode=2` PERCEPTUAL (wavelet)** (§E) — `av1_compute_q_from_energy_level_deltaq_mode` + `log_block_wavelet_energy`/`haar_ac_energy`; single-frame-applicable, needs dwt.c (Haar AC, also ABSENT).
10. **`--aq-mode=1` VARIANCE_AQ + `--aq-mode=2` COMPLEXITY_AQ + segment-map coding-method** (§E) — decision/frame-setup side unported (the `write_segment_id` writer exists). Single-frame-applicable segmentation.
11. **`--enable-rate-guide-deltaq`, `--auto-intra-tools-off`, `--deltaq-mode=4/5`** (§E) — niche deltaq companions, all ABSENT.
12. **`--delta-lf-mode` decision side** (§E) — only the `write_delta_lflevel` writer is ported.
13. **level/tier computation `av1_get_seq_level_idx`** (§F, level.c) — the one true missing HEADER algorithm; `seq_level_idx[]`/`tier[]` are bootstrapped from the parsed real seq header.
14. **`--full-still-picture-hdr` / annexb / large-scale-tile framing** (§F); **saliency map / blockiness / external_partition** (§G) — non-default / byte-inert / research-plugin (blockiness is stats-only, never affects bytes).

### PARTIAL (core ported; arms / speeds / bit-depths / config missing)

1. **DEFAULT-PARITY GAP — loop-restoration is ON by default in allintra, but the (byte-exact) LR search is wired only behind `--enable-restoration=1`, not the default path** (§E). Verified first-hand: `av1_cx_iface.c:286` default `enable_restoration=1`, not cleared by the :3065 allintra override; runtime :1273 keeps it on for non-realtime. The port's default `encoder_gate_e2e_*` compare against a restoration-**OFF** reference, so a true default `aomenc --allintra` stream (restoration on, even when all RU resolve NONE → different seq/frame header bits) is **not** byte-matched by the port's default path. **CLAUDE.md's "Loop-restoration: OFF by default in allintra" is factually wrong** (PARITY C2 is correct). Highest-value single-frame fix: wire the existing byte-exact LR search into the default path (and fix the doc).
2. **Real-content byte-parity at cpu-used≥1** (§B/§F, KB-13) — synthetic gates are 64/64 at every speed, but decoded-conformance content is **24/60** byte-exact at cpu 1-4 (interior BLOCK_16X16/8X8 AB/rect/split-prune near-ties; the port under-prunes). Speed-0 real content is 30/30.
3. **deltaq-mode=3 PERCEPTUAL_AI: highbd + partial-SB sub-arms** (§E) — bd8/single-tile/dims-×8px only; the bd10/12 FP-quantize arm + partial-SB source-border extension are unported.
4. **Header self-derivation still bootstrapped** (§F) — base_qindex (byte-exact #8 mapping exists but not wired into the encode composition; harness parses it), tile-count/config choice, CICP echo, and full temporal-unit / TD-OBU self-assembly. These are wiring, not missing algorithms (except level/tier, above).
5. **CDEF FAST search levels 1..5 + SB128-CDEF-on not e2e-gated** (§E) — ported + table-unit-tested; only FULL (speed 0) is e2e byte-gated (CDEF off by default → unexercised).
6. **Frame-edge partial-SB AB & 4-way** (§B) — the RD search attempts 4-way/AB only when all strips are in-frame; nonrd rect at a frame edge is `unimplemented!()`. Not biting at speed 0 (KB-6 30/30) but a risk for edge SBs coding partial AB/4-way at speed≥1.
7. **Pinned self-promoting near-ties** (not coverage gaps — open byte-parity residuals, each a sibling-C RD dump away): `--use-intra-dct-only=1` chroma (§C/§D); palette 2/7 128² AB/4-way (KB-P29); speed-6/7 noise-cq63 (mi 8,0) TX_16X16-vs-32X32 (KB-10/11); speed-8 4 diag est-arm V/H (KB-12).
8. **flat_block_finder percentile** (§G) — `is_flat` + features exact; the top-10% sigmoid `exp` is the sole libm-sensitive step (count is bit-exact).



---

## SUMMARY

For the single-frame / ALLINTRA / KEY mission, the aom-rs port covers the libaom v3.14.1 encoder algorithm surface **deeply**: the entire low-level DSP layer (transform fwd+inv all sizes/types + WHT, quant FP/B/DC ×lowbd/highbd ×flat/QM, range coder + CDF adaptation, intra predictors incl. highbd/directional/edge/filter/CfL, distortion + Hadamard/SATD, CDEF + deblock kernels, the full txb coeff-coding + trellis + cost stack) is **BYTE-EXACT**, and the whole speed-0..9 ALLINTRA KEY search→pack→coeff→post-filter-search→header pipeline is **byte-identical to real `aomenc`** on synthetic content across bit depths 8/10/12, mono/4:2:0/4:2:2/4:4:4, multi-tile, coded-lossless, QM-on, CDEF-strength search, loop-restoration search, tune=IQ/SSIMULACRA2, superres-FIXED, film-grain table-inject, and deltaq-mode 3/6 — every partition type (incl. AB + 4-way), every intra mode + angle-delta + filter-intra + CfL + palette, every tx-type/tx-size search level + trellis, and the full winner-mode two-pass all proven with differential and/or e2e byte gates. **The single most impactful gap is a default-config mismatch, not a missing kernel: loop-restoration is ON by default in allintra, but the byte-exact LR search is wired only behind `--enable-restoration=1`, so the port's default path does not yet byte-match a plain `aomenc --allintra` stream.** The genuinely-ABSENT single-frame encode-tech is: SB128 encode, `prune_tx_type_using_stats` (≥480p cpu≥2), `--quant-b-adapt`, the IntraBC leaf coeff arm + full-pel DV search (skeleton unwired), the nonrd sp8/9 HBD/lossless/palette arms, superres AUTO/QTHRESH/RANDOM, film-grain/noise ESTIMATION (`--denoise-noise-level`), tune=butteraugli/vmaf, deltaq-mode=2 (wavelet), aq-mode=1/2 + segmap coding, rate-guide/auto-intra-tools-off deltaq, and level/tier computation (the one missing header algorithm — other header fields are bootstrap-wiring, not missing algorithms). Real-content byte-parity holds at speed 0 (30/30) but is PARTIAL at cpu≥1 (24/60, interior partition-prune near-ties), with a handful of self-promoting pinned mode/tx near-ties. Inter/motion/TPL/GOP/RC-beyond-fixed-Q are OUT-OF-SCOPE-inter (fully enumerated); the only inter files with single-frame relevance are those intrabc reuses (mcomp/hash_motion/encodemv) and the shared nonrd infra.

