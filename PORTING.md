# PORTING.md — the C→Rust map

This is the auditability index for `zenav1-aom`: for each Rust module, the
`upstream/` libaom v3.14.1 source it ports and the differential test that gates
it. Browse the Rust next to the C it reimplements, then run the gate that proves
they agree byte-for-byte.

> **Coverage note, added 2026-08-03.** Until this revision only `zenav1-aom-dsp`
> had a module→C table; the decoder and encoder sections listed *gates* only, so
> the file did not actually map the encoder's ~45 modules to their upstream C —
> `cnn_partition` (the intra CNN partition prune, and the subject of the largest
> encoder performance landing to date), `palette_search`, `var_part`,
> `nonrd_pickmode`, `intrabc_search` and the rest were absent. Module tables for
> both crates are below. Each module's own source-file doc comment remains the
> authority on its exact C line refs; these tables are the index.
>
> The list is derived from the module doc comments as of `4d341c7`. If you add a
> module, add its row in the same commit.

## How the differential harness works

Every kernel is validated against the **real exported C function**, not a
transcription of it. A `*_diff` test feeds identical inputs to the Rust port and
to the linked C libaom function and asserts byte-identity across randomized fuzz
inputs plus hand-picked edge cases. Priority of evidence, highest first: real
exported C function > synthetic facade over a real function > verbatim
transcription (a transcribed oracle can carry a shared bug, so it is the weakest
witness).

The C oracle is the pinned `upstream/` git submodule (libaom v3.14.1,
`03087864`), built once from source by `crates/aom-sys-ref/build.rs` in the
deterministic single-thread config — see [`reference/BUILD_CONFIG.md`](reference/BUILD_CONFIG.md).
`cargo test` drives that build automatically; a fresh box needs only
`cmake`, `nasm`, and a C compiler on `PATH`.

Run any gate with `cargo test -p <crate> --test <name>`. Run everything with
`just test`; run it again under the scalar pin with `just test-scalar`
(`AOM_FORCE_SCALAR=1`), which forces every SIMD kernel through its scalar twin so
the full suite proves SIMD work left the transcribed scalar path untouched. The
`dispatch` module is validated precisely by that scalar-pin job — it is the
archmage RTCD-equivalent dispatch layer, not a port of a specific C file.

The conformance-corpus and real-content e2e tests need the AV1 test vectors;
provision them first with `python3 xtask/conformance.py --fetch --scope intra`
(the same command CI runs) or point `AOM_CONFORMANCE_DIR` at a populated corpus.
These tests fail loud when the corpus is absent — they never silently skip.

---

## `zenav1-aom-dsp` — the kernels

`crates/aom-dsp/src/<module>/`. Each module's source-file doc comment names its
exact upstream provenance; this table is the index. Gate commands are
`cargo test -p zenav1-aom-dsp --test <name>`.

| Module | `upstream/` libaom source | Differential gate(s) |
|---|---|---|
| `transform` | `av1/common/av1_inv_txfm1d.c`, `av1/common/av1_inv_txfm2d.c`, `av1/encoder/av1_fwd_txfm1d.c`, `av1/encoder/av1_fwd_txfm2d.c` | `txfm1d_diff`, `txfm2d_diff`, `inv_txfm1d_diff`, `inv_txfm2d_diff`, `fdct_diff`, `txfm2d_simd_perm_diff` |
| `quant` | `av1/encoder/av1_quantize.c`, `av1/common/quant_common.c` | `quantize_fp_diff`, `quantize_b_diff`, `quantize_b_adaptive_diff`, `quantize_qm_diff`, `dc_quant_diff`, `build_quantizer_diff` |
| `txb` | `av1/encoder/encodetxb.c`, `av1/common/txb_common.{c,h}` | `txb_diff`, `write_txb_full_diff`, `read_txb_full_diff`, `cost_coeffs_diff`, `txb_init_levels_simd_diff` |
| `cdef` | `av1/common/cdef_block.c`, `av1/common/cdef.c` | `cdef_diff`, `cdef_filter_diff`, `cdef_frame_diff`, `cdef_filter_simd_diff` |
| `restore` | `av1/common/restoration.c` | `lr_read_diff`, `lr_write_diff`, `wiener_simd_diff` |
| `intra` | `aom_dsp/intrapred.c`, `av1/common/reconintra.c` | `intra_diff`, `predict_intra_diff`, `dr_predict_high_diff`, `edge_diff`, `filter_intra_diff`, `build_filter_intra_diff`, `intra_simd_diff` |
| `loopfilter` | `aom_dsp/loopfilter.c` | `lpf_diff`, `hbd_lpf_diff`, `lf_apply_diff`, `lpf_simd_diff` |
| `dist` | `aom_dsp/sad.c`, `aom_dsp/variance.c` | `dist_diff`, `sad_simd`, `sum_squares_diff`, `hbd_dist_diff`, `vector_var_diff`, `hbd_variance_simd_diff` |
| `inter` | `av1/decoder/decodeframe.c` (`dec_build_inter_predictor`), `av1/common/reconinter.c` | `inter_pred_diff`, `interintra_diff`, `warp_diff` |
| `convolve` | `av1/common/convolve.c`, `av1/common/filter.h` | `convolve_diff` |
| `recon` | composition: dequant (`quant`) + inverse transform (`transform`, `av1_inverse_transform_block`) + residual add | `dequant_txb_diff` (dsp); `reconstruct_txb_diff` (encode) |
| `dispatch` | archmage RTCD-equivalent SIMD/scalar dispatch (infrastructure, not a C-file port) | the `AOM_FORCE_SCALAR` scalar-pin CI job / `just test-scalar` |
| `entropy` | `aom_dsp/entdec.c`, `aom_dsp/entenc.c` (Daala/MSAC range coder); default CDFs from `av1/common/entropy.c` + `token_cdfs.h`/`entropymode.c` | `entropy_diff`, `cdf_diff`, `default_cdfs_diff`, `entropy_ctx_diff`, `prob_cost_diff`, `leb128_diff`, `obu_diff` |

---

## `zenav1-aom-decode` — the decoder

`crates/aom-decode/src/`. The tile-reconstruction driver (partition walk +
per-leaf mode-info/coeff decode + intra predict + inverse transform + post-filter
frame walk) over the `aom-dsp` kernels. Ports the decode path of
`av1/decoder/decodeframe.c` + `av1/decoder/decodemv.c` + `av1/decoder/decodetxb.c`
and the common frame walks (`av1/common/cdef.c`, `av1/common/restoration.c`,
`av1/common/av1_loopfilter.c`). Gate commands are
`cargo test -p zenav1-aom-decode --test <name>`.

| Gate (test) | What it proves |
|---|---|
| `conformance_corpus` | **Gate 1.** Byte-identity + golden per-plane MD5 vs the C decoder across the AV1 intra conformance scope. Needs the corpus (see above). |
| `real_bitstream` | Real coded streams decode byte-identical, including 128×128 superblocks (`--sb-size=128`) and multi-tile. |
| `superres_diff` | Superres KEY frames (`AOM_SUPERRES_FIXED`, several denominators) decode byte-identical. |
| `tile_roundtrip` | Encode→decode roundtrip: the port's own coded tiles decode back to the source recon. |
| `film_grain_diff` | Film-grain synthesis matches the C decoder. |
| `disable_cdf_update_diff` | `disable_cdf_update` frames decode byte-identical. |
| `chroma_facades_cdiff` | The chroma reconstruction facades match their C counterparts. |
| `config_permutations_decode` | The *combination* half of Gate 1: bitstream-feature crossings the single-axis gates leave unproven (delta-lf × deblock, per-tile context resets × delta-q/segment-id, QM × per-segment dequant, 4:2:2 × CDEF/LR). Every cell's stream comes from the real C encoder and its realized tuple is read back from the port's own parse — see `docs/DECODER_CONFIG_COVERAGE_2026-07-30.md`. |
| `superres_tiles_diff` | Superres × multi-tile-column. Refused outright until 2026-07-31 and covered by nothing else — the conformance corpus contains zero superres and zero multi-tile vectors. |
| `fuzz_regression`, `fuzz_sweep` | The no-panic property on malformed input, plus the committed crash corpus. |
| `inter_walking_skeleton`, `inter_ratchet`, `inter_real_frame` | The in-progress inter-frame path, ratcheted byte-exact through a single-reference feature ladder and several real frames. |

### Decoder modules

| Module (`crates/aom-decode/src/`) | `upstream/` libaom source |
|---|---|
| `lib.rs` | tile-reconstruction driver: `av1/decoder/decodeframe.c` (`decode_tiles`/`decode_partition`/`decode_block`), `decodemv.c`, `decodetxb.c` |
| `frame.rs` | `decode_frame_obus`: the OBU walk (`av1/decoder/obu.c`) + `read_uncompressed_header` (`decodeframe.c`), then drives the post-filter frame order — deblock, CDEF, superres upscale, loop restoration — over the `aom-dsp` `*::frame` walks (`decodeframe.c:5422` ordering) |
| `superres.rs` | `av1/common/resize.c` — normative horizontal superres upscale |
| `film_grain.rs`, `film_grain_gaussian.rs` | `av1/decoder/grain_synthesis.c` (`av1_add_film_grain` / `add_film_grain_run`) + its gaussian table |
| `qm.rs`, `qm_tables.rs` | `av1/common/quant_common.c` (`av1_get_iqmatrix` / `av1_qm_init`) + the QM tables |
| `plane.rs` | reconstruction-plane storage; the bd8 `u8` / bd10-12 `u16` split of `aom_dsp::lowbd` (port infrastructure, not a C-file port) |
| `config.rs`, `error.rs` | the zen cross-cutting decode contract — resource limits, allocation mode, stop token, categorized errors (port infrastructure; see CLAUDE.md "Zen codec cross-cutting compliance") |

---

## `zenav1-aom-encode` — the encoder

`crates/aom-encode/src/`. The RD partition/mode/tx search + forward
transform/quantize/entropy-coding + bitstream pack, over the `aom-dsp` kernels.
Ports `av1/encoder/` (`encodeframe.c`, `partition_search.c`, `rdopt.c`,
`tx_search.c`, `encodetxb.c`, `bitstream.c`, and the speed-feature machinery in
`speed_features.c`). Gate commands are
`cargo test -p zenav1-aom-encode --test <name>`.

| Gate (test) | What it proves |
|---|---|
| `xform_quant_diff` | `av1_xform_quant` (forward transform + quantize + entropy context) is byte-exact vs C. |
| `partition_pick_diff`, `rd_pick_intra_sb_diff` | The RD partition/mode search matches C's recursion, decision for decision. |
| `search_tx_type_diff`, `uniform_txfm_yrd_diff`, `txfm_uvrd_diff` | Tx-type / tx-size RD search matches C. |
| `encoder_gate_e2e_byte_match` | **Gate 2.** ALLINTRA KEY encode byte-matches real `aomenc` across `--cpu-used 0..9` on synthetic grids. |
| `encoder_gate_chroma_ss_e2e` | Real conformance-decoded content (KB-6 recipe) byte-matches real `aomenc` at speed 0, across chroma subsampling. |
| `encoder_gate_bd10_diff` | 10-bit encode byte-matches. |
| `encoder_gate_multitile` | Multi-tile encode byte-matches across tile grids. |
| `encoder_gate_tune_iq_e2e`, `qm_encode_witness` | `tune=IQ` and quantization-matrix (`--enable-qm`) encodes byte-match. |
| `var_tx_leaf_diff`, `var_tx_recursion_diff`, `tx_split_nn_diff`, `prune_tx_2d_diff` | The inter/intrabc variable-transform coefficient arm (recursion + leaf + the tx-split and prune-tx-2D NN prunes) is differential-locked vs C — the in-progress IntraBC/inter coeff path. |

The **whole-frame** encoder gates live in the `zenav1-aom-bench` crate (`cargo test
-p zenav1-aom-bench --test <name>`), because they need the C oracle end to end:

| Gate (test) | What it proves |
|---|---|
| `config_permutations` | The knob-combination half of Gate 2: 10 speeds × 26 axis levels, singletons and covering-array combinations, against real `aomenc`. Its two open-cell lists have been **empty since 2026-08-02** — every cell byte-identical. Design: `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md`. |
| `armed_tools_decode_gate` | Round-trips any encode with a non-default tool armed through the C decoder *and* dav1d. Byte-identity to a reference proves nothing for configurations the reference never encodes; its coverage is a compile error rather than a list. |
| `s4cov_crop_format_axis`, `s4cov_partial_sb_axis`, `s4cov_hd_speed_axis`, `s4cov_qm_axis` | The configuration-axis sweeps: crop/format, partial superblocks, the HD speed band, and QM × speed. Open cells here are the T4 tier of CLAUDE.md's coverage queue. |
| `kb5_lossless_speed_axis`, `kb19_min_partition_4k`, `kb21_qm_speed4`, `kb22_hd_arms`, `kb28_crop_dims`, `kb31_mandatory_tiles`, `kb32_nonrd_size_bands`, `kb34_nonsquare_nonrd_leaf`, `kb35_nonrd_palette_arm`, `kb36_above_720p_speed_axis`, `kb37_nonrd_palette_search` | One per closed (or pinned) Known Bug — each pins the axis its bug was found on, so the bug cannot silently return. See CLAUDE.md's Known Bugs ledger. |
| `rd_close_harness`, `rd_close_palette`, `rd_close_intrabc`, `toggles_rd_close` | The RD-closeness ledger for features whose byte-identity gate has not landed yet (PARITY.md section B): size + zensim bands vs real aomenc rather than byte-identity. |
| `sb128_e2e`, `lr_restoration_gate`, `lr_default_parity`, `encoder_gate_cdef_e2e`, `encoder_gate_superres_e2e`, `film_grain_gate`, `delta_lf_mode_e2e`, `deltaq_mode2_e2e`, `deltaq_mode3_e2e` | Per-feature whole-frame byte-identity gates (PARITY.md section A). |

The kernel-level encoder differentials (CNN partition prune, intra-mode cost,
noise model, denoise, resize, and the rest) live alongside these in
`crates/aom-encode/tests/` — each named `*_diff` and gated the same way.

### Encoder modules

`crates/aom-encode/src/`. Every module's own doc comment carries the exact C
line refs; this is the index.

**Partition / mode / transform search**

| Module | `upstream/` libaom source |
|---|---|
| `partition.rs`, `partition_pick.rs` | `av1_rd_pick_partition` (`av1/encoder/partition_search.c:5653`) — the survey and the real recursion with live contexts |
| `encode_sb.rs` | `encode_sb` / `encode_b` (`partition_search.c:1581`/`:1419`) — the winner-subtree `DRY_RUN_NORMAL` re-encode |
| `rd_pick.rs` | `av1_rd_pick_intra_mode_sb` (`av1/encoder/rdopt.c:3636-3698`) |
| `intra_rd.rs` | block-level intra-mode RD — a composition primitive deliberately narrower than `av1_rd_pick_intra_sby_mode` (single-transform-block blocks, caller-supplied candidate order); gated as a composition by `intra_rd_pick_diff` |
| `intra_uv_rd.rs` | chroma intra RD — `av1_txfm_rd_in_plane` (`tx_search.c`) + `intra_mode_search.c` |
| `tx_search.rs` | `search_tx_type` / `get_tx_mask` / `av1_pixel_diff_dist` (`av1/encoder/tx_search.c`) |
| `var_tx.rs` | `av1_pick_recursive_tx_size_type_yrd` (`tx_search.c:3553`) + its recursion — the inter/IntraBC variable-transform coefficient arm (KB-15) |
| `rd.rs` | `av1/encoder/rd.c` + `rd.h` — the Laplacian model, RD-cost macros, qindex→λ |
| `mode_costs.rs`, `real_costs.rs` | `av1_fill_mode_rates` / `av1_fill_coeff_costs` (`rd.c`) + `intra_mode_info_cost_y` (`intra_mode_search_utils.h`) |
| `curvfit_tables.rs` | GENERATED — the curvefit model-rd grids from `rd.c` |
| `interp_rd.rs` | the switchable interp-filter rate model (inter track) |
| `lib.rs` | `av1_xform_quant` (`av1/encoder/encodemb.c`) — the composition layer |
| `encode_intra.rs` | `av1_encode_intra_block_plane` (`encodemb.c:801`) — the winner re-encode pass |

**Speed features and the fast tiers**

| Module | `upstream/` libaom source |
|---|---|
| `speed_features.rs` | `set_allintra_speed_features_framesize_independent` + `_framesize_dependent` + `av1_set_speed_features_qindex_dependent` (`av1/encoder/speed_features.c`) |
| `cnn_partition/` | `intra_mode_cnn_partition` (`av1/encoder/partition_strategy.c`) — the speed≥1 learned SPLIT/no-split prune. Also the subject of KB-PERF-1, the largest encoder performance landing to date |
| `var_part.rs` | `av1_choose_var_based_partitioning` (`av1/encoder/var_based_part.c`), KEY-frame arm — speed 7's fixed partition tree |
| `nonrd_pickmode.rs` | `av1_nonrd_pick_intra_mode` (`av1/encoder/nonrd_pickmode.c:1582`) + `av1_block_yrd` (`nonrd_opt.c:126`) — speeds 8/9 |

**Learned prunes (NN weights are transcribed by `xtask/transcribe_*.py`)**

| Module | `upstream/` libaom source |
|---|---|
| `part4_prune.rs`, `part4_nn_weights.rs` | `av1_ml_prune_4_partition` (`partition_strategy.c:1326-1523`) + `partition_model_weights.h` |
| `ab_nn_prune.rs`, `ab_nn_weights.rs` | `ml_prune_ab_partition` (`partition_strategy.c:1223-1320`) + `partition_model_weights.h` |
| `prune_tx_2d.rs`, `prune_tx_2d_nn_weights.rs` | `prune_tx_2D` (`tx_search.c:1541`) |
| `tx_split_nn_weights.rs` | `ml_predict_tx_split` (`tx_search.c:1755`) |
| `intra_tx_nn_weights.rs` | `ml_predict_intra_tx_depth_prune` (`tx_search.c:2823`) |
| `hog.rs`, `hog/` | `generate_hog` + `av1_intra_hog_model_nnconfig` (`intra_mode_search_utils.h`) — directional-mode pruning |

**Screen content**

| Module | `upstream/` libaom source |
|---|---|
| `palette_search.rs` | `av1_rd_pick_palette_intra_sby` (`av1/encoder/palette.c`) + the k-means kernels (`k_means_template.h`), `av1_count_colors[_highbd]`, `find_top_colors`, `optimize_palette_colors` |
| `intrabc_search.rs` | `rd_pick_intrabc_mode_sb` (`rdopt.c:3427`) + the source hash (`hash_motion.c`, `hash.c`) + `av1_intrabc_hash_search` (`mcomp.c:1908`) |

**Frame-level decisions and filters**

| Module | `upstream/` libaom source |
|---|---|
| `rc.rs` | `av1_quantizer_to_qindex` (`av1/encoder/av1_quantize.c:1041`) + the single-KEY-frame `AOM_Q` branch of `av1_rc_pick_q_and_bounds` (`ratectrl.c:1832-1837`) — the `--cq-level` → `base_qindex` mapping |
| `lf_search.rs` | `av1_pick_filter_level` (`av1/encoder/picklpf.c`) |
| `pickcdef.rs` | `av1_cdef_search` (`av1/encoder/pickcdef.c`) — reached only with `--enable-cdef=1`; CDEF is off by default in allintra |
| `allintra_vis.rs` | `av1_get_variance_boost_block_variance` (`av1/encoder/aq_variance.c:184`) — `--deltaq-mode=6`, the tune=IQ/SSIMULACRA2 default |
| `resize.rs` | `av1/common/resize.c` — the non-normative source downscale for superres |
| `superres_select.rs` | `av1/encoder/superres_scale.c` — QTHRESH/AUTO/RANDOM denominator derivation |

**Film grain (the `--denoise-noise-level` estimation path)**

| Module | `upstream/` libaom source |
|---|---|
| `denoise.rs` | `aom_wiener_denoise_2d` + `aom_denoise_and_model_run` (`aom_dsp/noise_model.c`) |
| `noise_model.rs` | `aom_noise_strength_solver_*` + `linsolve` (`aom_dsp/noise_model.c`) |
| `noise_fft.rs`, `noise_fft_gen.rs` | `aom_dsp/fft.c` + the `aom_noise_tx_*` wrapper in `aom_dsp/noise_util.c` |
| `grain_table.rs` | `aom_dsp/grain_table.c` — the `--film-grain-table` reader/lookup |

**Bitstream**

| Module | `upstream/` libaom source |
|---|---|
| `pack.rs` | the `OUTPUT_ENABLED` partition/mode-info/coefficient write walk (`av1/encoder/bitstream.c` + `encodetxb.c`) |
| `obu_assemble.rs` | `OBU_FRAME` assembly (frame header + tile group, the `num_tg == 1` combined form) |

**Inter encode — an early skeleton, not a landed path** (see `INTER-ENCODE-ROADMAP.md`)

| Module | `upstream/` libaom source |
|---|---|
| `inter_frame.rs` | encode-side reference management + low-delay P frame-header derivation |
| `inter_rd.rs` | `av1_rd_pick_inter_mode_sb` (`rdopt.c` ~6180) → `handle_inter_mode` (`:3063`), reduced to the roadmap §3 envelope |
| `inter_me.rs` | `av1_find_best_sub_pixel_tree` (`av1/encoder/mcomp.c:3266`) — the subpel search machinery net-new for inter; the full-pel core is shared with `intrabc_search` |
| `inter_costs.rs` | the inter mode / reference / MV-mode cost tables |
| `inter_pack.rs` | `pack_inter_mode_mvs` (`bitstream.c:1092`) |

---

## Dev-only harness crates

None of them ship (`publish = false`); they exist to measure the port.

- **`zenav1-aom-sys-ref`** (`crates/aom-sys-ref/`) — the FFI oracle. Its sole
  `build.rs` builds the pinned libaom from `upstream/` and links it; every
  `*_diff` test above is a dev-dependent of this crate. This is the only crate in
  the workspace that touches C.
- **`zenav1-aom-bench`** (`crates/aom-bench/`) — the Gate-3 port-vs-C performance
  harness (`cargo bench -p zenav1-aom-bench --bench gate3`, paired zenbench
  rounds) plus every whole-frame encoder gate, tabled in the encoder section
  above (`cargo test -p zenav1-aom-bench --test <name>`). The encoder-performance
  A/B tooling lives here too: `scripts/eprof_ab.sh` (`ROTATE=1` is now the
  default) and the `eprof_*` / census binaries.
- **`zenav1-aom-dsp-bench`** (`crates/aom-dsp-bench/`) — port-only DSP kernel
  benchmarks with **no** C-oracle dependency, for timing the port's own dispatch
  entry points (`cargo bench -p zenav1-aom-dsp-bench`).
