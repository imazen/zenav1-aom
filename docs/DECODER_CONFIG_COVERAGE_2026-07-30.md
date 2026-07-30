# Decoder config-permutation coverage — 2026-07-30

What the decoder-track gates actually cover across the **bitstream feature
combination** space, established from bitstream CONTENT (not test names), the
holes that were open, which of them are now closed, and which remain (with the
reason).

Companion gate: `crates/aom-decode/tests/config_permutations_decode.rs`
(31 cells, **4.0 s** debug / **2.2 s** release, all byte-identical to
`aom_codec_av1_dx`).
Companion raw data: `benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`
(the realized feature tuple of frame 0 of all 235 conformance vectors).

---

## 1. Method — coverage measured from bitstreams, never from names

Every claim below about what a stream exercises comes from the **port's own
header parse**: `aom_decode::frame::decode_frame_obus_prefilter` returns
`(KfTileDecode, KfTileConfig, FrameHeaderObu)`, and those two structs carry
every axis of interest (`sb_size_128`, `bd`, `monochrome`, `subsampling_*`,
`prefix.reduced_still_picture_hdr`, `superres_scaled`, `coded_lossless`,
`tx_mode_select`, `reduced_tx_set_used`, `disable_cdf_update`,
`delta_q_present`, `delta_lf_present`, `allow_screen_content_tools`,
`allow_intrabc`, `tile_info.{cols,rows}`, `loopfilter.filter_level*`, `cdef.*`,
`lr.frame_restoration_type`, `using_qmatrix` + `qm_{y,u,v}`, `seg.enabled`,
`film_grain_params_present`).

Two inventories were run:

1. **Conformance corpus (direct evidence).** All 235 `.ivf` vectors in
   `conformance/data` were parsed; 0 failures. Raw table committed as
   `benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`.
2. **Port-generated gates (construction + spot evidence).** The
   `real_bitstream` family's config grids were read out of the test sources,
   and the encoder-knob → realized-tuple mapping for every knob those grids use
   was confirmed by re-encoding representative cells through the SAME
   `aom_sys_ref` shims and parsing the result (that probing is what produced
   the new gate's cell table; each cell's realized tuple is printed on every
   run).

This method already corrected one documentation claim (see §6).

---

## 2. What the conformance corpus covers — measured

235 vectors, families: `b8-01-size` (100), `b8-00-quantizer` (64),
`b10-00-quantizer` (64), plus one each of `b8-02-allintra`, `b8-04-cdfupdate`,
`b8-16-intra_only-intrabc`, `b8/b10-23-film_grain`, `b8/b10-24-monochrome`.

| axis | value distribution over frame 0 of all 235 vectors |
|---|---|
| `sb_size_128` | **1 in 230**, 0 in 5 |
| bit depth | 8 in 169, 10 in 66, **12 in 0** |
| subsampling | **4:2:0 in all 235** (4:2:2 and 4:4:4 in **0**) |
| monochrome | 2 |
| `reduced_still_picture_hdr` | **0 in all 235** |
| `film_grain_params_present` | 2 |
| `superres_scaled` | **0 in all 235**, denom == 8 everywhere |
| `base_qindex` | 96 distinct values, 0..255 |
| `coded_lossless` | 2 (`b8/b10-00-quantizer-00`, both SB128 4:2:0) |
| `tx_mode_select` | 226 SELECT / 9 LARGEST |
| `reduced_tx_set_used` | **0 in all 235** |
| `disable_cdf_update` | **0 in all 235** |
| `delta_q_present` | 2 (the film-grain pair only; SB64, 4:2:0, TX LARGEST) |
| `delta_lf_present` | **0 in all 235** |
| `allow_screen_content_tools` | 3 (2 monochrome + the intrabc vector) |
| `allow_intrabc` | 1 |
| tile grid | **1x1 in all 235** |
| CDEF `cdef_bits` | 0/1/2/3 all present (37/31/76/91) |
| LR `frame_restoration_type` | 12 distinct per-plane triples, all four kernels |
| `using_qmatrix` | **0 in all 235** |
| `seg.enabled` | **0 in all 235** |

So the corpus is a deep sweep of **one sequence shape** (SB128, 4:2:0, 8/10-bit)
across quantizer/size/CDEF/LR, and contributes essentially nothing on
4:2:2 / 4:4:4 / 12-bit / superres / tiles / QM / segmentation /
reduced_tx_set / disable_cdf_update / delta-lf.

---

## 3. What the port-generated gates cover

| gate | axes swept | held at default |
|---|---|---|
| `real_bitstream::real_bitstreams_decode_byte_identical_to_c` (336 arms) | {64x64, 96x80, 100x76} x {bd8,bd10} x {4:2:0, 4:4:4} + mono-bd8, cq bands 0–3, CDEF on/off, LR on/off, segmentation (aq 1/2, two-pass), GOOD + ALL_INTRA | SB64, 1 tile, deltaq-mode 0, no QM/palette/intrabc/lossless/superres/4:2:2 |
| `::sb128_streams_...` | SB128 x {4:2:0, 4:4:4, mono} x {bd8,bd10} x 4 sizes x CDEF/LR | 1 tile, no 4:2:2, no bd12, no QM/seg/lossless/delta |
| `::multi_tile_streams_...` | tiles (col-only, row-only, 2D, 4-col) x {4:2:0,4:4:4,mono} x {bd8,bd10} x CDEF/LR/seg, one SB128 cross-check | no 4:2:2, no bd12, no QM/lossless/nocdf/screen tools/delta |
| `::composition_422_...`, `::deblocked_422_chroma_*` | 4:2:2 x bd{8,10,12} x CDEF+LR x sizes incl. odd chroma half-width | SB64, 1 tile, no QM/seg/lossless/nocdf/delta |
| `::bd12_composition_...` | 12-bit | SB64, 1 tile |
| `::qm_streams_...` | QM levels {0,5,8,12} x {4:2:0,4:4:4,mono} x {bd8,bd10} x 3 sizes | **CDEF off, LR off**, SB64, 1 tile, aq 0, no 4:2:2 |
| `::lossless_streams_...` | lossless x {4:2:0,4:4:4,mono} x bit depths | SB64, 1 tile, no 4:2:2 |
| `::palette_streams_...`, `::intrabc_{mono,colour}_streams_...` | screen-content tools | SB64, 1 tile |
| `superres_diff` (4 tests, 90 streams) | superres denom {9,12,16} x {mono, 4:2:0, 4:4:4} x bd{8,10,12} x CDEF on/off x LR composed, + the KB-14 single-SB-column corner | **SB64 only, 1 tile only, no QM/seg/4:2:2/lossless/nocdf/delta** (the shim takes no control list — see §5) |
| `disable_cdf_update_diff` (54 streams) | `disable_cdf_update` x {4:2:0,4:4:4,mono} x bd{8,10,12} x 3 sizes x 2 cq | **1 tile**, SB64, no CDEF/LR forced, no 4:2:2 |
| `film_grain_diff` (30 streams + unit-level C-diff) | grain vectors {1,2,15} x {4:2:0,4:4:4,mono} x bd{8,10} | SB64, 1 tile |

Read together with §2, **every individual axis is covered somewhere.** The gap
was entirely in the **crossings**.

---

## 4. Holes found, and what closed them

Ranked by shared-code risk. "Closed by" names cells in
`config_permutations_decode.rs`.

| # | hole (crossing) | why it shares code | prior coverage | status |
|---|---|---|---|---|
| H1 | `delta_lf_present` **at all**, end-to-end | it is the ONLY frame-header flag that moves the per-superblock DEBLOCK LEVEL (`build_lf_inputs` -> `LfMi::delta_lf_from_base`) | **none**: 0/235 corpus vectors, and `real_bitstream` pins `--deltaq-mode=0`. Only symbol-level coverage existed (the synthetic mirror encoder in `tile_roundtrip.rs`) | **CLOSED** — D1, D2, D3, D4, D5, Q7 (6 cells; 3 of them with genuinely nonzero luma+chroma filter levels). Teeth-verified: see §7 |
| H2 | `delta_q_present` x tiles / SB128 / 4:2:2 / QM | the per-SB delta-q carry (`current_base_qindex`) restarts at each tile; it feeds `av1_get_qindex`, which QM then weights | corpus had `delta_q` on 2 vectors only, both SB64 / 4:2:0 / single-tile | **CLOSED** — D2 (4:2:2), D3 (tiles 2x2), D4 (SB128 4:4:4), D5 (SB128 + tiles 2x2), Q7 (x QM) |
| H3 | `reduced_tx_set_used = 1` **at all**, end-to-end | switches the ext-tx SET TYPE, i.e. the tx-type symbol alphabet and its CDF, for every coded block | **none**: 0/235 corpus, never set by any port gate. Symbol-level only (`tile_roundtrip.rs`) | **CLOSED** — R1, R2 (4:2:2 + live deblock), R3 (SB128 + mono), R4 (tiles 2x2 + 4:4:4) |
| H4 | QM x CDEF / LR | the QM gate pins `cdef=false, restoration=false`; QM changes the residual the filters then consume | none | **CLOSED** — Q1, Q2, Q4, Q5, Q6, Q7 |
| H5 | QM x segmentation | two stacked dequant modifiers: `av1_get_qindex(seg, ...)` picks the qindex, `av1_get_iqmatrix` weights it | none | **CLOSED** — Q4, Q5 |
| H6 | QM x SB128, QM x multi-tile, QM x 4:2:2 | `qm_v` is only coded under `separate_uv_delta_q`; the U/V matrices index the 4:2:2 chroma tx shapes | none | **CLOSED** — Q2 (SB128 + tiles 2x2), Q3 (4:2:2), Q5 (4:2:2 + seg), Q6 (mono + SB128) |
| H7 | `disable_cdf_update` x multi-tile | the header only codes `context_update_tile_id` + `tile_size_bytes` when tiles > 1, and the non-adapting reader must survive the per-tile `KfFrameContext` reset | none (the nocdf gate is single-tile) | **CLOSED** — T1 (tiles 2x2 + CDEF + LR), T9 (SB128 + 4:2:2) |
| H8 | segmentation x multi-tile | the segment-id **spatial predictor** reads above/left segment ids, which must restart at each tile edge | partially: `multi_tile_streams_...` carries seg arms | **REINFORCED** — T7 (tiles 2x2), T8 (SB128 + tiles + 4:4:4) |
| H9 | 4:2:2 x SB128, 4:2:2 x multi-tile | the 4:2:2 chroma grid under a `mib_size = 32` SB walk / per-tile chroma context resets | none | **CLOSED** — S1, T3, T9 |
| H10 | lossless x 4:2:2, lossless x SB128 x tiles | `all_lossless` deletes the loop-filter/CDEF/restoration header sections, so the tile split lands on a differently-SHAPED header | lossless gate is SB64 single-tile; corpus adds SB128 4:2:0 | **CLOSED** — L1 (4:2:2), L2 (mono + tiles), T2 (SB128 + tiles) |
| H11 | 12-bit x multi-tile, 12-bit x SB128, 12-bit x 4:4:4 filters | bit-depth-generic code, but the SB128/tile walks were never exercised at bd12 | none | **CLOSED** — T4, S2, S4 |
| H12 | monochrome x multi-tile | no-chroma path through the per-tile resets | covered by `multi_tile_streams_...` mono arms | **REINFORCED** — T5, L2 |
| H13 | screen-content tools x multi-tile / SB128 | the palette colour cache is seeded from above/left neighbours, which reset at tile boundaries | none | **CLOSED** — T6 (tiles + 4:4:4), S3 (SB128 + 4:4:4) |
| H14 | superres x {SB128, multi-tile, QM, segmentation, 4:2:2, delta-q, nocdf} | `superres_scaled` is read inside the LR RU-grid derivation (`aom-dsp/src/entropy/lr.rs:438`) — the RU grid is upscaled while the SB walk is downscaled, so SB128 (`lr_corners_in_sb` at `mib_size = 32`) and tiles genuinely interact with it | superres gate is SB64 / single-tile / no QM / no seg / no 4:2:2 | **OPEN — BLOCKED**, see §5 |
| H15 | `reduced_still_picture_hdr = 0` (the FULL sequence header) x port-generated tools | different sequence-header parse shape | **already covered**: 235/235 corpus vectors are `redhdr = 0`, and 31/31 of the new cells (like every port-generated stream) are `redhdr = 1`. Both branches are exercised | **NOT A HOLE** (see §6) |
| H16 | `delta_lf_multi = 1` | per-plane delta-lf instead of from-base | none anywhere | **OPEN — unreachable**: libaom's `--delta-lf-mode=1` only ever emits the from-base form (`setup_delta_q`, `encodeframe.c`). Symbol-level coverage exists in `tile_roundtrip.rs`. Needs a hand-built or non-libaom stream |

---

## 5. Collapse table — what is NOT crossed, and why

The gate composes rather than permutes: the decoder is a pipeline
(header parse -> tile split -> per-block symbols -> reconstruct -> deblock ->
CDEF -> superres -> LR), and byte-identity is all-or-nothing over the whole
pipeline, so **one cell with N features live at once exercises every pairwise
interaction among those N in a single decode**. Permuting the same N axes costs
2^N cells and proves nothing extra. 31 cells therefore cover ~120 crossings.

Pairs deliberately left uncrossed:

| pair | class | evidence / reason |
|---|---|---|
| QM x {deblock, CDEF, superres, LR} | (a) proven-disjoint — *but crossed anyway* | `using_qmatrix` / `qm_{y,u,v}` have exactly 5 read sites in the port: the frame-header parse (`aom-dsp/entropy/header.rs`), the `KfTileConfig` bridge (`frame.rs:1985`), `block_qm_level` derivation (`lib.rs:723-737`, `lib.rs:5052`) and the three `qm::iqmatrix` dequant call sites (`lib.rs:4310, 4416, 5268, 5582`). ZERO reads in the deblock / CDEF / superres / LR stages — QM can only reach them through pixel VALUES. The claim is corroborated, not relied on: Q1/Q2/Q4/Q5/Q6/Q7 cross it regardless |
| film grain x {tiles, SB128, QM, delta-q, 4:2:2, reduced_tx_set} | (a) proven-disjoint | film-grain synthesis runs on the FINAL upscaled+restored frame and reads only `(bd, mono, subsampling, matrix_coefficients)` plus its own params (`aom-decode/src/film_grain.rs`); it has no read of any frame-header tool flag. Covered in parallel by `film_grain_diff` (30 streams x 3 grain vectors x 5 formats) and by the 2 corpus film-grain vectors (which additionally carry `delta_q_present = 1` + CDEF + LR) |
| CICP colour description x everything | (a) proven-disjoint | display metadata; the only reconstruction consumer is film grain's MC-identity range gate, already covered above |
| `still_picture` sequence shape x tools | (a) covered in parallel | both branches are exercised in bulk — corpus is 235/235 `redhdr=0`, port gates are `redhdr=1` (§6) |
| **superres x {SB128, tiles, QM, seg, 4:2:2, delta-q, nocdf}** | **(b) UNREACHABLE with the current shim** | superres is an `aom_codec_enc_cfg_t` field (`rc_superres_mode` / `rc_superres_kf_denominator`), NOT an `aome_enc_control_id`. `AV1E_SET_ENABLE_SUPERRES` (102) only toggles the sequence-header bit. `shim_encode_av1_kf_superres` takes no extra-control list, so no superres stream can currently be co-emitted with any other tool. **This is an open hole, not a collapse** — see chunk 2 |
| `delta_lf_multi = 1` x anything | (b) unreachable | libaom never emits it (H16) |
| inter-frame tools (motion, ref frames, warp, OBMC, skip mode) | out of scope | the decoder is KEY-frame/intra-only per `CLAUDE.md` |

---

## 6. Documentation correction

`crates/aom-decode/tests/conformance_corpus.rs:29` states the in-scope corpus
frames are `reduced_still_picture=0`. That is **correct and now measured**
(235/235). What was *undocumented* is the complement: **every** stream produced
by the `aom_sys_ref` encode shims comes out with
`reduced_still_picture_hdr = 1` (they all set `cfg.g_limit = 1`, which makes
libaom pick `seq->still_picture` and therefore the reduced header). So the two
sequence-header shapes are split cleanly between the corpus gate and the
port-generated gates, and `reduced_still_picture_hdr` — which reading the prose
alone would flag as an untested axis — is in fact covered on both branches by
1000+ streams. Recorded here so a future session does not re-open it as a hole.

---

## 7. Teeth — the gate is load-bearing, and the hole it closes was real

Perturbation: in `crates/aom-decode/src/frame.rs:2123` (`build_lf_inputs`), the
per-block `delta_lf_from_base` copied into `LfMi` was replaced by a constant
`0`, i.e. the per-superblock delta-lf contribution to the deblock level was
dropped.

* **New gate FAILS**, in 0.02 s, on its first delta cell:

  ```
  D1_deltaq_deltalf_deblock_cdef_lr_420_bd10 [sb128=0 bd=10 mono=0 ss=(1, 1)
  stillhdr=1 sr=0 q=208 lossless=0 txsel=0 rtx=0 nocdf=0 dq=1 dlf=1 screen=0
  ibc=0 tiles=1x1 lf_y=1 lf_uv=1 cdef=1 lr=[0, 0, 0] qm=0(L0) seg=0 bytes=85]:
  LUMA differs from the C decoder at pixel 5829 (x=69, y=60): port=642 c=641
  ```

* **Every pre-existing decoder gate still PASSES** under the same perturbation:
  `conformance_corpus` (3 tests, all 235 vectors, 1.8 s), `real_bitstream`
  (15 tests, 192 s), `superres_diff` (5 tests, 22 s),
  `disable_cdf_update_diff` (2 tests, 20 s) — 25/25 green.

That asymmetry is the proof that H1 was a genuine hole and that the new cells,
not redundancy, are what close it. Perturbation reverted; `git diff` on
`crates/aom-decode/src/` is empty.

---

## 8. Timing

| build | wall |
|---|---|
| `--release` | **2.2 s** |
| `--profile test-fast` | 2.3 s |
| debug (stock `cargo test`) | **4.0 s** |

31 real libaom encodes + 31 port decodes + 31 C decodes. No corpus fetch is
involved (this gate generates its own streams), so there is no excluded
provisioning cost. Well under the 120 s budget; the headroom was spent on cells
rather than on slower `--cpu-used` settings.

Per-cell `--cpu-used` (0..6) is chosen purely to reach the target feature tuple
cheaply. The decoder does not care how a conformant stream was produced, and
every cell asserts its realized tuple, so speed is not a coverage variable —
but several tuples (nonzero filter levels at moderate q; `delta_q_present` from
`--deltaq-mode=2/3`) only appear at particular speeds on this content, which is
why the values are per-cell and probed rather than uniform.

---

## 9. Coverage arithmetic

* Axes considered: **21** (SB size, bit depth, monochrome, subsampling,
  reduced-still-picture header, film grain, superres, lossless, tx mode,
  reduced_tx_set, disable_cdf_update, delta-q, delta-lf, screen tools, intrabc,
  tile grid, deblock levels, CDEF, LR, QM, segmentation).
* Distinct crossings **enumerated** as interesting (a pair steering shared
  decoder code, both reachable): **16** (H1–H16 above).
* Already covered before this work: **2** (H8 segmentation x tiles partially,
  H12 mono x tiles; plus H15 which turned out never to have been a hole).
* **Closed here: 11** (H1–H7, H9, H10, H11, H13), reinforced: 2 (H8, H12).
* **Left open: 2** — H14 (superres crossings, blocked by the shim) and
  H16 (`delta_lf_multi`, unreachable from libaom).
* Run-level axis witnesses from the gate (printed every run, floored in the
  test): sb64 20 / sb128 11 / multi_tile 13 / tiles_both_axes 9 / delta_q 6 /
  delta_lf 6 / delta_lf_with_live_deblock 3 / reduced_tx_set 4 /
  disable_cdf_update 2 / qm 7 / qm_x_seg 2 / seg 4 / seg_x_tiles 2 / lossless 3
  / screen_tools 2 / cdef_live 25 / lr_live 11 / deblock_luma 12 /
  deblock_chroma 9 / monochrome 4 / yuv420 13 / yuv422 8 / yuv444 6 / bd8 20 /
  bd10 8 / bd12 3.

---

## 10. Chunk 2 — the next slice

1. **Unblock the superres crossings (H14).** Add an `extra_ctrl_ids /
   extra_ctrl_vals / n_extra_ctrls` passthrough to `shim_encode_av1_kf_superres`
   in `crates/aom-sys-ref/shim/dec_shim.c` (append-only; every existing caller
   passes `NULL, NULL, 0` and stays byte-inert — the exact pattern
   `encode_kf_pass` already uses) plus the matching
   `ref_encode_av1_kf_superres_ctrls` wrapper. That single change makes
   superres x {SB128, tiles, QM, segmentation, 4:2:2, delta-q, nocdf} reachable,
   and those are the highest-risk remaining crossings because
   `superres_scaled` is read *inside* the LR RU-grid derivation
   (`aom-dsp/src/entropy/lr.rs:438`) while the SB/tile walk runs on the
   downscaled grid. Owner: whoever owns `aom-sys-ref` (not this agent).
   Expected new cells: ~6, ~0.5 s.
2. **`delta_lf_multi = 1` (H16).** Needs a stream libaom will not produce.
   Options: a third-party encoder, or splicing a re-written frame header onto
   real tile data (the `EncodeCell::frame_obu_payload` / `splice_frame_obu`
   machinery in `aom-bench` already does exactly this shape of thing). Low
   frequency in the wild, but it is the last never-exercised frame-header flag.
3. **Widen the corpus scope.** The corpus fetch currently pulls the `intra`
   scope. `av1-1-b8-02-allintra` exists only at 8-bit; libaom's test-data set
   also ships 4:4:4 / 4:2:2 profile vectors (`av1-1-b8-*-profile*`) that this
   corpus does not contain and that would give independently-produced 4:2:2 /
   4:4:4 evidence to sit alongside our own encoder's.
4. **Fold the axis-witness map into a machine-readable artifact.** The gate
   already prints the witness `BTreeMap`; emitting it as a TSV under
   `benchmarks/` on every run would let CI diff coverage over time instead of
   only floor-checking it.
