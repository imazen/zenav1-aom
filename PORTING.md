# PORTING.md — the C→Rust map

This is the auditability index for `zenav1-aom`: for each Rust module, the
`upstream/` libaom v3.14.1 source it ports and the differential test that gates
it. Browse the Rust next to the C it reimplements, then run the gate that proves
they agree byte-for-byte.

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
| `inter_walking_skeleton`, `inter_ratchet`, `inter_real_frame` | The in-progress inter-frame path, ratcheted byte-exact through a single-reference feature ladder and several real frames. |

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

The kernel-level encoder differentials (CNN partition prune, intra-mode cost,
noise model, denoise, resize, and the rest) live alongside these in
`crates/aom-encode/tests/` — each named `*_diff` and gated the same way.

---

## Dev-only harness crates

Neither ships (`publish = false`); they exist to measure the port.

- **`zenav1-aom-sys-ref`** (`crates/aom-sys-ref/`) — the FFI oracle. Its sole
  `build.rs` builds the pinned libaom from `upstream/` and links it; every
  `*_diff` test above is a dev-dependent of this crate. This is the only crate in
  the workspace that touches C.
- **`zenav1-aom-bench`** (`crates/aom-bench/`) — the Gate-3 performance harness
  (`cargo bench -p zenav1-aom-bench --bench gate3`, port vs C oracle, paired
  zenbench rounds) plus the whole-frame stills-parity gates: `rd_close_harness`,
  `sb128_e2e`, `lr_restoration_gate`, `lr_default_parity`, `encoder_gate_cdef_e2e`,
  `encoder_gate_superres_e2e`, `film_grain_gate`, and the toggle/deltaq e2e gates
  (`cargo test -p zenav1-aom-bench --test <name>`).
