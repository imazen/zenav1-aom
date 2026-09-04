# PARITY.md — the stills-parity ledger

Single source of truth for the **stills bulk-port pivot** (2026-07-16): port ALL absent
stills-relevant encoder features with **RD-closeness** validation first (quality + size vs
real aomenc), byte-exactness deferred per feature. Stills scope = single-frame / ALLINTRA
(usage=2) / KEY-frame encode; inter-frame/video-only features (motion, TPL, GOP, RC beyond
fixed-Q, S-frames, temporal filtering) are out of scope.

## Rules

1. **New features land OFF-by-default.** The proven byte-exact envelope (section A) must
   stay byte-exact — every landing runs the full suite and perturbs nothing. A feature is
   reached only by its explicit knob until it graduates.
2. **Every landing updates its row.** A bulk-ported feature appends a section-B row with
   its measured deltas the same commit it lands. Absent features live in section C; a
   feature moves C → B when its RD-close gate lands, and B → A when a byte-identity gate
   lands (cite the new gate + commit in the row you move).
3. **The RD-close gate is `aom_bench::rd_close`** (landed 3c5235e):
   encode the SAME input with the port (knob on) and real aomenc (same knobs), decode BOTH
   with the port decoder, score BOTH recons against the source with zensim (single-threaded,
   deterministic), record sizes. Acceptance bands (`RdBands::default()`):
   **|size_delta| <= 5% AND zensim_drop <= 0.5** (bit-identical cells fast-path as EXACT).
   Bands were sanity-anchored on first real data: byte-exact cells report 0/0 with zensim
   79.5–91.7 at web cq; a genuine cq20→cq63 divergence measures −94% size / −171 zensim —
   the bands discriminate near-ties from regressions with wide margin. Tightening per
   family is fine; widening is a test relaxation (user sign-off required).
   Usage: `cargo test -p aom-bench --test rd_close_harness -- --nocapture` (the harness's
   own gate); bulk agents call `compare_cell` / `run_stock_cell` / `splice_frame_obu` from
   their own tests with their knob wired.
4. **No bootstrap leaks.** The port's stock encode bootstraps some frame-header FIELDS from
   the C stream (qindex mapping, tile limits — the documented Gate-3 caveat). The feature
   under test must NOT flow through that bootstrap: a CDEF-search port derives its own
   strengths, an LR port its own RU params. Copying the feature's decisions from the C
   header fakes parity.
5. Cheap cells (64²/128², a few cq) so gates run often; always include at least one
   real-content cell (`EncodeCell::real_content`, the KB-6 conformance-decoded-YUV recipe)
   — synthetic-only validation has already missed real divergences once (KB-6).
6. **A landing's gate list must name INTEGRATION targets, not a unit-test count.**
   `-p <crate> --lib` runs none of the byte-identity gates in this ledger — they all live
   in `tests/` targets — and the coverage census (`content_family_census`) needs a
   non-default feature, so neither is reached by `--lib` or by a default `cargo test -p`.
   `just gate-encode` runs both. Added 2026-08-30 after KB-42: four consecutive landings
   gated on `--lib` plus named diff tests while 23 byte/RD gates and the census sat broken
   across six red CI runs.

## Section A — BIT-IDENTICAL (proven)

Byte-identity gates landed and green on origin/main. Any regression here is a shipping bug.

### Encoder (vs real `aomenc` path, `aom_codec_av1_cx`)

| Component / envelope | Gate (test name) | Landed |
|---|---|---|
| ALLINTRA speed-0 e2e, synthetic grids (mono+4:2:0, multi-SB 16/16, cq5..63) | `encoder_gate_e2e_byte_match` | 76b1ffb |
| speed-0 low-qindex web range (cq8–30, 12 cells) | `encoder_gate_e2e_low_qindex_speed0` | ec5905c |
| speed-0 rich-content strong-LF incl. screen-content cq62 (6/6) | `encoder_gate_e2e_rich_content_strong_lf` | 74fb582 (KB-2) |
| **REAL-content map 30/30** (bd8 4:2:0, 64²/128²/196² × cq5..63, incl. partial-SB frame edge) | `encoder_gate_real_image_e2e_kb6_repro` | ca2826f → 57d5ce0 (KB-6 series) |
| `--cpu-used=1` (14/14) | `encoder_gate_speed1_textured_allintra` | 7e2391d, ad734e4, a128655 |
| `--cpu-used=2` | `encoder_gate_speed2_textured_allintra` | a8a3992 |
| `--cpu-used=3` (64/64) | `encoder_gate_speed3_textured_allintra` | e18772c, 652423e (KB-7) |
| `--cpu-used=4` (64/64) | `encoder_gate_speed4_textured_allintra` | e8c662f → 35fdce8, 652423e (KB-8) |
| `--cpu-used=5` (64/64) | `encoder_gate_speed5_textured_allintra` (+ `encoder_gate_speed5_vs_speed4_sf_witness`) | 9aeb0ee |
| `--cpu-used=6` (64/64 canon; noise ext 6/6 asserted — the cq63 near-tie CLOSED 2026-07-31 by KB-21 root #2) | `encoder_gate_speed6_textured_allintra` (+ `encoder_gate_speed6_vs_speed5_sf_witness`, `encoder_gate_speed6_noise_flatuv_allintra`) | 90e69e8 |
| `--cpu-used=7` (64/64 canon; VAR_BASED_PARTITION fixed-tree + rd_use_partition; noise ext 8/8 asserted — the cq63 KB-10-twin near-tie CLOSED 2026-07-31 by KB-21 root #2) | `encoder_gate_speed7_textured_allintra` (+ `encoder_gate_speed7_vs_speed6_sf_witness`, `encoder_gate_speed7_noise_flatuv_allintra`, `kb11_speed7_noise_localize`) | a9dc5f1 |
| `--cpu-used=8` (**64/64 canon**; nonrd PICKMODE — `nonrd_use_partition` single-pass walk + `av1_nonrd_pick_intra_mode` estimate/hybrid arm; noise ext cq12/32/48/63 asserted. **Real content >= 720 px on the short side: the `force_large_partition_blocks_intra` threshold arms landed 2026-08-01 (KB-32 / issue #7); the last residual — 4 `diag` cells read as an estimate-arm near-tie since 2026-07-17 — closed 2026-08-02 with `aom_hadamard_lp_8x8`'s missing trailing transpose, KB-12**) | `encoder_gate_speed8_textured_allintra` (+ `encoder_gate_speed8_vs_speed7_sf_witness`, `encoder_gate_speed8_noise_flatuv_allintra`) | 9b57803 |
| `--cpu-used=9` (64/64 canon; all-estimate `hybrid_intra_pickmode=0` + the 3 speed-9 mode prunes + INTERNAL_COST_UPD_OFF **below 4k / SBROW at 4k+**; noise ext cq12/32/48/63 asserted — KB-12) — GATE 2 (cpu 0-9) COMPLETE. **Real content: byte-exact across the `RESOLUTION_720P` area threshold and up to 2112² since 2026-08-01 (KB-32 / issue #7)** | `encoder_gate_speed9_textured_allintra` (+ `encoder_gate_speed9_vs_speed8_sf_witness`, `encoder_gate_speed9_noise_flatuv_allintra`) | 9b57803 |
| bd10/bd12 mono+4:2:0 aggressive-HF (12/12) | `kb4_gate_bd10_bd12_mono_hf_byte_match` | a2dd28e (KB-4) |
| bd10 non-4:2:0 (444/422 × 64²/128²) | `encoder_gate_bd10_non420_e2e_kb4_repro` | 1ecfafb |
| bd10/bd12 full-frame mono+4:2:0 | `encoder_gate_bd10_diff` | 20f1e70, 800e6fc |
| 4:2:2 / 4:4:4 bd8 full-frame | `encoder_gate_chroma_ss_e2e` | 2ee900d, 0eb42eb (#26) |
| Coded-lossless cq0 **mono + 4:2:0** bd8 (both hard-asserted; KB-5 closed / #32) | `encoder_gate_lossless_cq0_e2e_kb5_repro` | ba560eb (mono) + KB-5 420 fix |
| Coded-lossless cq0 across **`--cpu-used` 0..9** x {4:2:0, mono} x {textured 64², smooth 128²} bd8, **plus bd10/bd12** x {0, 8, 9} — 52 lossless cells + 10 cq1 controls, estimate-arm TX_4X4 reach asserted | `kb5_lossless_speed_axis` | (this landing) |
| QM-on forward-quant (`--enable-qm`, 40 cells bd8+bd10) | `qm_encode_witness` | 5b512bf (parts 624e91d/a066cf8/abb68d9) |
| Multi-tile encode (2×1/1×2/2×2, 4:4:4 128²) | `encoder_gate_multitile_e2e` | f6e6319 |
| **C8 partition-control disable arms** (`--enable-rect-partitions=0`, `--enable-ab-partitions=0`, `--enable-1to4-partitions=0`, `--min-partition-size=16`, `--max-partition-size=32`, square-only 8..32 band) × real-content 64²(cq32/63)+128²(cq12), each knob anti-vacuity-witnessed (must change the C stream) | `toggles_rd_close::toggles_c8_*` (hard `bit_identical` pins) | (this landing) |
| **C10 intra-tool disable arms** (`--enable-smooth-intra=0`, `--enable-paeth-intra=0`, `--enable-cfl-intra=0`, `--enable-directional-intra=0`, `--enable-diagonal-intra=0`, `--enable-angle-delta=0`, `--enable-filter-intra=0`, `--enable-intra-edge-filter=0`) × the same witnessed grid; seq-header knobs assert the C stream's seq bits == the knob (no bootstrap flow) | `toggles_rd_close::toggles_c10_*` (hard `bit_identical` pins) | (this landing) |
| **C9 tx-control arms** (`--enable-tx64=0`, `--enable-rect-tx=0`, `--enable-flip-idtx=0`, `--use-intra-default-tx-only=1`, `--reduced-tx-type-set=1`, `--enable-tx-size-search=0` — frame-header bits/tx_mode asserted == knob) × the same witnessed grid | `toggles_rd_close::toggles_c9_*` (hard `bit_identical` pins) | (this landing) |
| **C11 `--cdf-update-mode=0` encoder e2e** × the same witnessed grid (header `disable_cdf_update` asserted == knob). Landing FIXED a real pack bug: only the coeff writer was gated — partition/mode/tx symbol writers adapted CDFs unconditionally, desyncing the stream vs the non-adapting decoder (zensim −264 vs C's +79 pre-fix). Fix = C's architecture: `allow_update_cdf` on `OdEcEnc`, gated in `write_symbol` (aom_write_symbol), set per tile in `pack_tile` (write_modes) | `toggles_rd_close::toggles_c11_cdf_update_mode_0` (hard `bit_identical` pins) | (this landing) |
| **C9 `--disable-trellis-quant` arms** (`=1` NO_TRELLIS_OPT, `=2` FINAL_PASS_TRELLIS_OPT) × the same witnessed grid. `=2` landing FIXED a real pack bug: `encode_b_intra_dry` hardcoded `dry_run_output_enabled: false`, so the OUTPUT_ENABLED pack pass did not apply FINAL_PASS trellis (search=no-trellis, pack must trellis, encodemb.h:153) → recon divergence (Δzensim 1.855 pre-fix). Fix threads the `output_enabled` arg; byte-inert for every non-FINAL_PASS gate | `toggles_rd_close::toggles_c9_trellis_quant_off` / `_final_pass_only` (hard `bit_identical` pins) | 2026-07-17 (5a644c6) |
| **`prune_tx_type_using_stats` (luma-intra tx-type stats prune)** — the ABSENT-and-UNEXERCISED sf: C enables it ALLINTRA at cpu-used>=2 (level 1) / >=4 (level 2) but ONLY `is_480p_or_larger` (speed_features.c:262/300), so every sub-480p gate frame missed it. Ported the multi-type-arm prune in `get_tx_mask_intra` (drops tx types whose KF frame-prob < the threshold, keeping the max-prob type; `update_type = KF_UPDATE` for a lone still, `default_tx_type_probs[0]`) + the framesize+speed derivation (`SpeedFeatures::prune_tx_type_using_stats`, set in port_encode_full from `min(w,h) >= 480`; the KB-3 `use_square_partition_only_threshold` framesize analog). Byte-inert on every sub-480p gate. | `tx_mask_diff` (port `get_tx_mask_intra` == the C oracle — REAL exported `default_tx_type_probs` + the prune — across all tx_size×mode×config × prune-level 0/1/2; `default_tx_type_probs_kf_matches_c`; `stats_prune_shrinks_the_mask` 120 cases) + `tx_stats_prune_e2e` (`_knob_bites`: >=480p 512² cpu-2 noise, prune LOAD-BEARING — port-without diverges from real aomenc, port-with byte-matches; `_sub480p_unchanged` regression) | (this landing) |
| qindex-from-cq derivation (#8) | `qindex_from_cq_diff` | (landed pre-pivot) |
| **SELF-CONTAINED KEY-FRAME ENCODE — no C bootstrap anywhere in the path** (`aom_encode::key_frame::encode_key_frame`): the port AUTHORS its own TD OBU + sequence-header OBU + frame header and packs its own tile. **372/372 cells BYTE-IDENTICAL to real aomenc's whole temporal unit** (186/186 on 2026-09-02; +186 coded-lossless cells on 2026-09-03) — cq 0..63 step 5 (+63), a 1..19 step-2 low-q arm, a dense **cq-0 / coded-lossless J arm** ({mono, 4:2:0, 4:2:2, 4:4:4} x bd {8,10,12} x 5 contents x `--cpu-used` {0,9}, + bd8 at {3,6}, + a 1x1..258x258 size ladder), {mono, 4:2:0, 4:2:2, 4:4:4} x bd {8, 10, 12} (profiles 0/1/2), sizes 16x16..512x512, 12 crop / partial-SB sizes incl. 1x1, 5 content classes (flat / gradient / texture / noise / checkerboard), **all four (CDEF, loop-restoration) combinations including BOTH ON** (27 post-filter cells across mono/4:2:0/4:4:4 x cq 5..63 x 64x64..256x256), **`--cpu-used` 0..=9** (68 speed cells), and **MANDATORY multi-tile** (frames wider than `MAX_TILE_WIDTH`: 4160x64 / 4224x128 / 4160x192 two-tile and 8320x64 three-tile, 8192x64 four-tile, speeds 0..6 -- each tile packed independently with a fresh frame context, assembled through `assemble_multitile_frame_obu_payload_derived`). The tile-count derivation models `set_tile_info`'s STRICTER column minimum (`encoder.c:386-390` uses `(max_width_sb << k) <= sb_cols`, one more than `av1_get_tile_limits`' own `tile_log2`, which differ exactly at a frame whose `sb_cols` is an exact multiple-by-power-of-two of `max_width_sb`); bite-proved by weakening it to `<`, which makes 4033x64, 4096x64 and 8192x64 diverge while 4032x64 / 4097x64 / 4160x64 / 2048x64 stay exact. Real aomenc's ALLINTRA default is CDEF OFF with restoration ON (`av1_cx_iface.c:3067`) and is byte-gated at EVERY speed; `--enable-cdef=1` is byte-gated at speeds 0..3 and pinned-divergent at 4..9 (the FAST search levels, PARITY C1's never-e2e-gated fraction, measured divergent here in the header's `cdef_strengths` only). The port also models `speed_features.c:2753`, which CLEARS the sequence header's `enable_restoration` bit at allintra speed >= 5 -- without it every `--enable-restoration=1` cell at speed 5..9 diverged. Both the real C decoder AND `aom-decode` decode the PORT's own stream to the pixels real aomenc's stream decodes to. Derived, not replayed: `seq_level_idx`/`tier` (`set_bitstream_level_tier`), profile / bit depth / subsampling / `num_bits_*` / the reduced-still-picture framing, `base_qindex`, `allow_screen_content_tools` (the ported detector DRIVES it now), the tile grid, the loop-filter levels, and the `tx_mode` SELECT->LARGEST flip via a new `txb_split_count` over the port's own winner trees. Post-filter order is C's `cdef_restoration_frame`: deblock -> `av1_cdef_search` -> `av1_cdef_frame` (apply) -> `av1_pick_filter_restoration` on the POST-CDEF recon, packed through the new additive `pack::pack_tile_from_trees_lr` (neither predecessor carried CDEF literals AND per-RU restoration params, so the default config had no pack entry point). Envelope: ALLINTRA, `--cpu-used` 0..=9, SB64 — everything else REFUSED by name (`KeyFrameError`). **The same landing found and fixed a real shell bug the gate had mis-attributed:** the pack env carried a past-the-end sentinel for `tile_row_end`/`tile_col_end` instead of C's `AOMMIN(.., mi_rows/mi_cols)` clamp (`av1_tile_set_row`/`_col`), which changes the search's frame-edge decisions. Reverting the clamp makes 131x131, 132x64, 132x128, 132x132, 196x64, 196x196, 260x260 and 261x261 all diverge and the clamp makes all eight byte-identical — bite-proved. Four of those had been pinned as "RD near-ties"; they were this. Also fixed a real bug in BOTH screen-detector callers: C reads the 8-ALIGNED `y_width`/`y_height`, not the crop. | `self_contained_key_frame` (`byte_matches_real_aomenc` 372/372, `decodes_to_the_same_pixels` 25 cells x 2 decoders, `coded_lossless_reconstructs_the_source_exactly` 248/248 x 2 decoders, `no_seq_header_stream_is_rejected_by_the_c_decoder`, `mutated_sequence_header_is_caught`, `refuses_configurations_it_has_no_gate_for`, `open_divergences_are_pinned`) + `seq_level_idx_diff` | (this landing, 2026-09-02) |
| Gate-3 perf cells byte-verified before timing | `aom-bench` `EncodeCell::assert_byte_exact` | 057bde2 |
| **CDEF-strength RD search** (`--enable-cdef=1`, #7 / family C1): 14/14 cells — real content 196²/64² cq5..63 (cdef_bits=2 four-strength joint sets, per-unit literals) + mono/4:4:4/4:2:0/bd10 axes; speed-0 FULL search; two-pass encode→LF→search→pack | `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close` (aom-bench; rd_close report + full byte-identity asserts) | 016d4dd + 9850da6 + c9ebf83 |
| **Loop-restoration RD search** (`--enable-restoration=1`, family C2): 8/8 cells BYTE-IDENTICAL + 8/8 decisions equal C's — real content 64² cq{12,32,48}, 196² cq{20,48} (partial-SB edges), 352×288 cq{32,55} (multi-unit size-descent grids), b10 352×288 cq32; decision shapes covered: all-NONE, WIENER-luma, SGRPROJ-luma, WIENER-all-3-planes, mixed SGR-luma+WIENER-chroma (b10), unit-size descent picking 128; allintra speed-0 full search (all 16 SGR eps, ±{4,2,1} Wiener tap refine, 256→128→64 size loop) | `lr_restoration_gate.rs::lr_restoration_search_rd_close_vs_real_aomenc` (aom-bench; rd_close report + full byte-identity + decision-equality asserts) | e24cf09 + 96d3464 + dfd757e + 96534c4 |
| **tune=IQ / tune=SSIMULACRA2 family** (`--tune=iq` / `--tune=ssimulacra2`, family C4): each bundle piece e2e byte-identical — QM-level formulas (`aom_get_qmlevel_luma_ssimulacra2` + `_444_chroma`), QM-PSNR dist metric (trellis + tx-search transform-domain distortion QM-weighted), `--sharpness` 0..7, `--enable-chroma-deltaq`, `--enable-adaptive-sharpness`, Variance-Boost `--deltaq-mode=6` — PLUS the **full composite bundle** (54/54 cells: mono/420/444 × 64/128/192 × cq12/32/50, CDEF overridden off = the separate C1 track, symbol-inert). All OFF by default (`TuneKnobs::default()` = PSNR). Anti-vacuous witnesses for sharpness/adaptive/variance-boost + `tune_shim_smoke` | `encoder_gate_tune_iq_e2e` (9 tests) + `qm_level_diff` + `tune_shim_smoke` | 2026-07-17 |
| **Superres encoder-side, FIXED denom, bd8 + bd10/12** (`--superres-mode=fixed --superres-denominator=D`, family C6): **13/13 bd8 + 16/16 bd10/12 cells BYTE-IDENTICAL** — bd8 real-content 196² 4:2:0 (denoms 9/12/14 × cq{20,32,48}) + mono (denoms 9/12 × cq{20,48}); bd10/12 textured-synthetic 128² 4:2:0 (denoms 9/12/14 × cq{20,48}) + mono (denoms 9/12 × cq32), 8 cells/bit-depth. The source is downscaled horizontally to the coded `FrameWidth` via the ported non-normative `av1_resize_plane` (bd8) / `highbd_resize_plane` (bd10/12) (`aom_encode::resize`, differentially bit-exact vs the exported C symbols — interpolate 5-band + down2_symeven/symodd + resize_multistep + `coded_superres_width`), encoded at the reduced width (existing speed-0 KEY machinery, mi grid sized to coded_w), superres denom + upscaled width signalled in the header (`write_superres_scale`); port+C decoders agree on the upscaled recon. Superres OFF by default. **Anti-vacuity**: `scale_denominator == D`, `coded_w < w`. **Follow-ups (Section C6)**: the 8-bit denom-16-even-width optimized-scaler corner (`av1_resize_and_extend_frame`), and AUTO/QTHRESH/RANDOM denom selection + the recode loop. | `encoder_gate_superres_{fixed_real_content,fixed_mono,fixed_highbd}_rd_close` (aom-bench; rd_close report + full byte-identity asserts) | 2505b49f (kernel) + 68703b1 (bd8) + (this bd10/12 landing) |
| **C7 film-grain table-inject** (`--film-grain-table` / `AV1E_SET_FILM_GRAIN_TABLE`): the port's OWN grain-table reader + lookup (`aom-encode/src/grain_table.rs`, port of `aom_dsp/grain_table.c` `aom_film_grain_table_read`/`_lookup`) → `FilmGrainParams` → the already-bit-exact `write_film_grain_params` header writer. Byte-identical vs real aomenc on 4:2:0 bd8 REAL content (64² cq20/32, 128² cq12) + mono/444/bd10 synthetic × built-in test vectors 1/2/6/15 (rich full-chroma / max-lag / chroma-points-absent / chroma-scaling-from-luma). Grain is decode-side synthesis → coded tiles UNCHANGED (the C shim replicates the plain `encode_kf_pass` control set so only the seq present bit + frame grain block are added). No-bootstrap-leak witness: injecting a different vector's params DIVERGES. | `film_grain_gate.rs::film_grain_table_inject_{420_real,format_axes}` + `film_grain_no_bootstrap_leak_witness` (aom-bench) | (this landing) |
| **`--deltaq-mode=3` DELTA_Q_PERCEPTUAL_AI** (family C5, the stills-specific perceptual-AI arm): 7/7 cells BYTE-IDENTICAL to real aomenc `--deltaq-mode=3` — real content 192²/192×128/128×192 4:2:0 cq12..63 (9/6 SBs each get a distinct wiener qindex; the delta fires + the port reproduces it). Full port of `av1_set_mb_wiener_variance` (per-8x8 intra-SATD search + FP-quantize + Weber stats + the `norm_wiener_variance` `estimate`+2-iter refinement), the map reductions (`get_{satd,sse,max_scale,window_wiener_var,var_perceptual_ai}`), `av1_get_sbq_perceptual_ai` + `av1_get_deltaq_offset` (`av1_get_deltaq_offset` differentially locked 18432/18432 vs the exported C fn), and the per-SB pack threading (`setup_delta_q_perceptual_ai` → the shared `av1_adjust_q_from_delta_q_res`, reusing the mode-6 `DeltaQFrameCtx`). OFF by default; anti-vacuous knob-bites witness. **Highbd (bd10/12) DONE** (this landing): the FP-quantize arm dispatches `av1_highbd_quantize_fp` for bd>8 in `av1_set_mb_wiener_variance` (the only bd8-specific step; predict/subtract/DCT/inverse/Weber were already bd-parameterized); 5 bd10 + 1 bd10 non-square + 3 bd12 (bd10-content ×4 promoted) cells added to the gate, all byte-identical. Scope: bd8/10/12 4:2:0, dims a multiple of 64/8px (196²-partial-SB is the remaining follow-up). | `encoder_gate_deltaq_mode3_e2e` (`deltaq_mode3_e2e.rs`: `deltaq_mode3_perceptual_ai_e2e` 16/16 hard byte-match incl. bd10/12 + `deltaq_mode3_knob_bites`) + `deltaq_perceptual_ai_diff` (`get_deltaq_offset_matches_c`) | 2026-07-17 + (this bd10/12 landing) |
| **`--delta-lf-mode=1` DELTA_LF** (family C5): 7/7 cells BYTE-IDENTICAL to real aomenc `--delta-lf-mode=1 --deltaq-mode=2` — real content 192²/192×128/128×192 4:2:0 cq12..63. Per-SB `delta_lf_from_base = ((delta_qindex/4 + res/2) & ~(res-1))` clamped (setup_delta_q, encodeframe.c:380-383, `DEFAULT_DELTA_LF_RES=2`, single/`DEFAULT_DELTA_LF_MULTI=0`), derived from each SB's `delta_qindex` (reuses the mode-2/3 delta-q) in `pack_leaf` + coded via the already-plumbed `write_delta_q_params_sb` delta-lf arm. The frame `filter_level` DEPENDS on delta-lf: the LF pick's trial deblock reads `mbmi->delta_lf_from_base` via `get_filter_level` (av1_loopfilter.c:73-88), so the port stamps the per-SB delta-lf into the LF mi grid (`stamp_lf_delta_lf`) + sets `LfSearchFrame::delta_lf_present`. OFF by default (rides on a firing delta-q mode); anti-vacuous knob-bites witness. Scope: bd8 4:2:0, dims a multiple of the 64px SB. | `delta_lf_mode_e2e` (`delta_lf_mode_e2e.rs`: 7/7 hard byte-match + `delta_lf_mode_knob_bites`) | 2026-07-18 |
| **`--deltaq-mode=2` DELTA_Q_PERCEPTUAL** (family C5, the wavelet-AC-energy arm; `DELTA_Q_PERCEPTUAL_MODULATION==1`): 7/7 cells BYTE-IDENTICAL to real aomenc `--deltaq-mode=2` — real content 192²/192×128/128×192 4:2:0 cq12..63 (per-SB wavelet energy modulates the qindex; the delta fires + the port reproduces it, decode-verified: C's per-SB `current_qindex` == the port's). Full port of the 5/3 dyadic dwt (`av1_fdwt8x8_uint8_input` + `haar_ac_sad` + `av1_haar_ac_sad_mxn_uint8_input`, dwt.c — a pure-C RTCD entry, no SIMD, so bit-exact to real aomenc; differentially locked vs the exported C fn), `haar_ac_energy`/`av1_block_wavelet_energy_level`/`av1_compute_q_from_energy_level_deltaq_mode` (aq_variance.c, single-frame `energy_midpoint=10.0`), and the `av1_rc_bits_per_mb`(KEY/AOM_Q)/`find_qindex_by_rate`/`av1_compute_qdelta_by_rate` rate model (ratectrl.c), per-SB pack threading (`setup_delta_q_perceptual` → the shared `av1_adjust_q_from_delta_q_res`, reusing the mode-3/6 `DeltaQFrameCtx`). OFF by default; anti-vacuous knob-bites witness. Scope: bd8 4:2:0, dims a multiple of the 64px SB (highbd + partial-SB are follow-ups). | `deltaq_mode2_perceptual_wavelet_e2e` (`deltaq_mode2_e2e.rs`: 7/7 hard byte-match + `deltaq_mode2_knob_bites`) + `deltaq_perceptual_wavelet_diff` (`haar_ac_sad_mxn_matches_c`) | 2026-07-18 |

### Decoder (vs real `aom_codec_av1_dx`)

| Component / envelope | Gate | Landed |
|---|---|---|
| Gate-1 conformance corpus, intra scope, **incl. q62/q63** (KB-1 fixed) + film-grain-synthesis / monochrome / cdf-update frame-0 breadth | `conformance_corpus` (byte-identity + golden MD5, CI `xtask/conformance.py --fetch --scope intra`) | 386c24f → 463f49f → 134c43c → ae0e6a1 |
| Real-bitstream KEY envelope (deblock, CDEF, LR, superres, SB128, lossless, QM, multi-tile, palette, intrabc, disable-cdf-update, 4:2:2 chroma deblock) | `real_bitstream` gate family | b8d79b2 → 3380a91, 798ec25, a90b0e7, 8502e13, 6899bea, 1dfbcc3, 42423ab, 351a160 |
| Superres **crossed with multi-tile-column** (the `UnsupportedFeature("multi-tile superres")` reject is GONE): 44 real libaom streams byte-identical, 2/3/4 tile columns × denoms {9,12,16} × bd8/bd10 × mono/4:2:0/4:4:4, composed with CDEF, LR and tile rows. Per-tile-column convolve walk (`av1_upscale_normative_rows`) in `superres::upscale_plane_tiles`. Also enforces `av1_is_min_tile_width_satisfied`, which superres tightens to 128 coded px per inner column and libaom's Release encoder violates (see `docs/LIBAOM_UPSTREAM_NOTES.md` C5). | `superres_tiles_diff` (4 byte-identity gates + the min-tile-width reject pair), `superres::tests::tile_walk_matches_the_continuous_walk` | (this landing) |

## Section B — PORTED, RD-CLOSE (not yet bit-exact)

Bulk agents append rows here as features land (rule 2). Empty at pivot start.

| Component | Knobs | Cells | size_delta | zensim_drop | Harness ref (test) | Date | Notes |
|---|---|---|---|---|---|---|---|
| Palette RD search (Y `av1_rd_pick_palette_intra_sby` + UV `_sbuv`: dim-1/2 k-means, top-colours, colour/map costs, header-rd gating + chroma early-term, palette recon + pack syntax/map tokens, neighbour cache/ctx grids) | `PickFrameCfg::palette_costs = Some` (= `--enable-palette=1`; OFF everywhere else) | 6 screen (text/UI, mono+420, 64²/128², cq12..63) + 1 real-content control | **5/7 EXACT (byte-identical)**; worst +2.55% | worst +0.190 (one cell −1.041 = port better) | `rd_close_palette::palette_y_rd_close_gate` (aom-bench) | 2026-07-17 | speed-0 sf levels (search 0 / size-search 1 / chroma early-term 1); speeds 1–5 levels wired untested-by-gate. Fixed latent UV no-palette-flag under-cost on screen frames (per-leaf `try_palette`). **The 5 EXACT cells are now HARD byte-identity asserts (Section-A-grade regression guards) inside the gate** (2026-07-17 pickup). **The 2 CLOSE 128² cells (`ui_420_128_cq32`, `text_420_128_cq20`) are PINNED** — decode-both localized to genuine palette-induced AB/4-way partition near-ties (`ui`: (mi 0,0) BLOCK_32X32 real HORZ_B vs port HORZ_4; `text`: (mi 8,20) BLOCK_16X16 real VERT vs port VERT_A); both are byte-exact with palette OFF and the palette machinery (`av1_allow_palette` / `av1_get_palette_bsize_ctx`/`_mode_ctx` / k-means / neighbour cache+ctx stamping) is verified C-faithful — same class as the KB-10/KB-11 pinned near-ties (closing needs a sibling-C per-candidate partition-RD dump). Regression-guarded by `decode_diff_palette_close_cells` (asserts the divergence PRESENT → self-promotes on any fix). (CDEF search + loop-restoration search, the first two bulk families, went straight to section A — 14/14 and 8/8 EXACT.) |
| **KB-41 (2026-08-30)**: palette cost plumbing — `palette_uv_mode_cost` filled + palette size/colour-index tables follow the per-SB mode-cost refresh; closed `PALETTE_ON_SPEED8_OPEN` (9/9), the full-RD half of `PALETTE_MANY_COLORS_OPEN` (9/9), `SCREEN_ARRAY_OPEN_ROWS` (1/1) and promoted `text_420_128_cq20` (`ui_420_128_cq32` followed with roots #7-#12 below) | `rd_close_palette`, `kb35_nonrd_palette_arm`, `kb37_nonrd_palette_search`, `config_permutations` | KB-41 |
| **KB-41 roots #3-#6 (2026-08-30)**: the speed-dependent IntraBC search (`intrabc_search_level` 1 / hash-8x8 cap / 64-candidate prune / DIAMOND + CLAMPED_DIAMOND site configs / NSTEP-only mesh / speed<=2 qindex bands / >=720p skip-row SAD), DEFAULT_EVAL transform-domain distortion + `predict_dc_level` in the intrabc coeff arm, the speed-dependent inter var-tx knobs (init depth 1, ml split 4000, PRUNE_2/3, `skip_tx_search`, stats prune), and the encode-time skip re-derivation for intrabc coeff blocks — 30/30 datagen cells byte-identical; promoted the KB-15 cell `scc_480x180_196_cq48` | `kb41_screen_detected_defaults` (on-demand planes), `rd_close_intrabc`, the palette/screen gates | KB-41 |
| **KB-41 roots #7-#13 (2026-08-30)**: (#7) the SEARCH-time `allow_intrabc` is the ported `estimate_screen_content_antialiasing_aware` decision (`screen_detect.rs`; C flips the header to 0 only after the tiles when no block used IntraBC, encodeframe.c:2443); (#8) the search-ctx `intrabc_cdf` adapts under that search-time allow even when the header dropped the flag; (#9) the search-ctx palette-Y flag/size CDFs adapt only for chroma-reference intra blocks (`av1_sum_intra_stats` early return; IntraBC winners are `is_inter_block` and never reach it); (#10) `rt_sf.use_nonrd_pick_mode` (speed >= 8) + `mv_sf.use_intrabc` switch the frame-wide DV search off (`rd_pick_intrabc_mode_sb`, rdopt.c:3432-3434) so the header's `allow_intrabc` ends 0; (#11) frame-edge HORZ_4/VERT_4 code fewer than 4 strips (`rd_pick_4partition` breaks at the first out-of-frame strip, partition_search.c:3948; the port's all-4-strips envelope guard is gone, `SbTree::{Horz4,Vert4}` carry `Option` strips); (#12) the search-ctx `tx_size_cdf` adapts for every coded intra block at the search-time TX_MODE_SELECT (`tx_size_search_methods[0][DEFAULT_EVAL]`) even when the header ends TX_MODE_LARGEST via the `txb_split_count == 0` flip (encodeframe.c:2797). (#13) `av1_set_screen_content_options` (encoder.c:2440-2480) is ported arm-by-arm: seq-forced → `--tune-content=screen` (declared per cell via `ToggleKnobs::tune_content_screen`; kb37's reference) → detection OFF under `use_nonrd_pick_mode && !hybrid_intra_pickmode` (allintra speed 9; kb35's control) → the antialiasing-aware detector. **NOT ported: `av1_determine_sc_tools_with_encoding`** (encoder.c:3312 — two q>=244 fixed-32x32 trial encodes that can turn screen tools ON with `allow_intrabc=0` when the detector said off; live on allintra below speed 8) — the bench asserts the decision against the oracle header and names that arm when it fires. All three shadows (`TileCtxState::search_*`) sit behind `allow_update_cdf`. Census **57/57** byte-identical (aomplanes 24 incl. 1920x1080 s6/s8, band 6, repro dirs 27 incl. the 85x128 partial-edge cells); promoted `ui_420_128_cq32`, the last pinned palette near-tie; **KB-30 closed the same day** — `cid22_6292444` is screen-detected content, byte-exact 11/11 (cq 1..60, cpu 6) once the tools are mirrored; **`RD_BAND_OPEN` closed** (1272x724 cq24 byte-exact at cpu 0..7, `kb34` re-pinned empty) and **KB-13's last two cells closed** (`cpu3 cq63` pair, `kb13_cpu3_cq63.rs`, sct=0) | `kb41_screen_detected_defaults` (on-demand planes), `rd_close_palette`, `rd_close_intrabc`, `config_permutations` | KB-41 |
| **KB-41 roots #14-#17 (2026-08-30, speed 4 on a screen frame)**: probing the 1280x800 screen cell at cpu 4 (the wave's slowest arm; the census had only cpu 6/8 there) found a NON-CONFORMANT stream (libaom and the port decoder both rejected it: "intrabc DV failed validity") and, once decodable, three more RD roots. (#14) `allow_intrabc` disables every post-filter stage in C — `if (!allow_intrabc) loopfilter_frame()` (encoder.c:3780) wraps the deblock pick, CDEF AND `av1_pick_filter_restoration`; the header codes no lr_params (the reader sets RESTORE_NONE) but the port's restoration search still ran at speed ≤5 and its re-pack wrote 54 LR units the header never announced — the whole LR stage is now gated on the final `allow_intrabc`. (#15) `try_tx_block_split`'s first child is bounded by `ref_best_rd - 0`: C sets the partition-flag rate on the init'd stats WITHOUT recomputing `rdcost` (tx_search.c:2470-2477), the port subtracted RDCOST(flag) and its child bailed through the `adaptive_txb_search_level` prune. (#16) the speed-4/5 est-rd tx-type prune (`prune_txk_type` ≤7 types / `prune_txk_type_separ` >7, tx_search.c:1912-1928) REPLACES the 2D-NN prune on inter/IntraBC blocks — ported (`var_tx.rs`); the 16x4 IntraBC block's mask was {V_DCT,H_DCT} in C vs the NN's {DCT_ADST,ADST_ADST}. (#17) `calc_pixel_domain_distortion_final`'s single-type downgrade reads the mask AFTER `get_tx_mask` (:2182); the port tested the pre-prune mask, kept the pixel re-measure on a DCT-only 8x8 IntraBC block (dist 9344 vs C's tx-domain 13503) and won a block C gave to SMOOTH. Cell byte-exact after the four. Widened census on the same screen frames (the s6 planes replayed at s4/s5 against a fresh oracle): **s4 6/9** exact (open: 1280x800 cq44 +10 B, 1920x1080 cq57 +10 B, 1920x1080 cq6 −33 B), **s5 1/3** (open: 1280x800 cq25 −5 B, 1920x1080 cq32 −123 B); every prior census cell still exact (68/68) and every gate green | `kb41_screen_detected_defaults` (+ `ZENAV1_DUMP_ORACLE_DIR`), `var_tx_*_diff`, the palette/intrabc/toggle gates | KB-41 |
| **KB-41 roots #18-#21 (2026-08-30, the cpu-4/5 screen residuals)**: the five cells left open after #14-#17 (s4 1280x800 cq44 +10 B, 1920x1080 cq57 +10 B, 1920x1080 cq6 −33 B; s5 1280x800 cq25 −5 B, 1920x1080 cq32 −123 B) were four more C conventions, each localized by first-syntax-diff + paired C/port per-txb probes. (#18) AB-partition split-ctx REUSE (`is_split_ctx_is_ready`, partition_search.c:4611-4617) requires the SPLIT child leaf to carry no palette and no CfL — the port reused any leaf. (#19) the SEARCH-side txfm-partition contexts are stamped on the OUTPUT run too: C's `tx_partition_count_update` → `update_txfm_count` → `txfm_partition_update` (partition_search.c:511-516) writes the var-tx leaf sizes for a non-skip inter block at `!dry_run`, the dry run via `tx_partition_set_contexts`; the port stamped only on dry runs, so the search arrays kept the value restored at SB start (the row-ABOVE SB's stamp) and the SB row below costed its tx split at ctx 18 where C had 19 (mi(32,90) under the 8x8 IntraBC at mi(30,90), leaf TX_4X4 eobs [0,0,5,0]). (#20) the IntraBC candidate's predict-skip SSE (`predict_skip_flag` → `pixel_diff_dist(x, 0, 0, 0, bsize, bsize, NULL)`, tx_search.c:194-214) sums the VISIBLE block only; the port summed the whole block, so a frame-bottom 64x64's skip candidate carried its 8 off-frame rows (847872 vs 716800) and lost to PAETH (mi(256,16), dv (-512,0)). (#21) a pick-skip'd var-tx txb hands its SIBLINGS the SEARCHED entropy context (`no_split->txb_entropy_ctx = p->txb_entropy_ctx[block]`, tx_search.c:2447 — pick_skip zeroes only eob + tx type; the encode pass re-derives 0 later via `is_blk_skip`); the port zeroed it, so the (1,1) child of an 8x8 IntraBC got txb_skip_ctx 3 where C had 5 after the (0,1) child searched eob 13 and was pick-skip'd (mi(20,154), dv (-368,-824)). Census after the four: **102/102 cells byte-identical across every plane dir** (s4 9/9, s5 3/3, s6/s8 24/24, band 6, 1080s6/one/s6res 4, the repro dirs 56 incl. the 85x128 partial-edge and cid22 512² screen cells), every gate green (the `var_tx_recursion_diff` facade modelled the OLD pick-skip ctx and was corrected to tx_search.c:2447), all TEMP probes stripped; the s4 census (three 1920x1080 cells among nine) runs in 81.5 s with oracle + both decodes — ~20 s per 1080p cell at cpu 4 — so the wave's screen cap is lifted (zenmetrics `ZEN_AOMRS_MAX_SCREEN_MP` default 16 MP) | `kb41_screen_detected_defaults` on `~/tmp/aomplanes{,_s4,_s5}`, `var_tx_*_diff`, the palette/intrabc/toggle gates | KB-41 |
| **KB-41 roots #22-#23 (2026-08-30, the >1 MP screen cells the aom-rs datagen wave refused)**: the fleet run `avifaom-enc-20260830` refused **277 cells on 56 screen-detected renditions above 1 MP** (counts per the wave report, not re-derived here) at low q (cq >= ~40), cpu 4 and 8 — the executor byte-verifies every cell, so a frame-OBU mismatch is a refusal. Localized on `2091x3072_cq62` (qindex 249) at cpu 4 and cpu 8; both roots are cost/reuse conventions only screen content reaches. **Method** (cheap, reusable): two new env gates on the instrumented oracle (`~/tmp/libaom-instr`) instead of the GB-scale `AOM_KB41_DUMP` — `AOM_KB41_PART=1` prints the partition-gate trace only (`PGATE` / `ABGATE` / `ABENTRY` / `ABALLOW` / `ABNN` / `ABSPLITINFO` / `P4ENTRY` / `P4PREML` / `P4POSTML` / `P4SPLITINFO`; 1.2M lines for a 6.4 MP frame, ~30 s), and `AOM_KB41_YHDR=1` prints `av1_rd_pick_palette_intra_sby`'s entry `dc_mode_cost`/`best_rd` (`YENTRY`) plus the per-candidate `intra_mode_info_cost_y` term breakdown (`YHDR`, whose hardcoded mi filter became this env). Matching port-side probes under `AOM_RS_PART_DUMP` made both roots a term-by-term diff. (#22) **`is_rect_ctx_is_ready[i]` is the RECT twin of root #18's SPLIT gate** (partition_search.c:3613-3619): C sets it only when the rect sub-block-0 winner carries **no luma AND no chroma palette** *and* `uv_mode != UV_CFL_PRED`; the port tested only the CfL half, under a comment asserting palette was off "in this envelope" — false on every screen-detected frame. So HORZ_B/VERT_B's sub-block 0 was COPIED from the rect winner where C re-searches it under the AB mode cache. cpu 4, mi(124,328) BLOCK_16X16: the two sides agree EXACTLY on the gate (`ABALLOW` = HORZ_B only, `best_rdc` 880,017,878, `rect_part_rd` 409,693,321/MAX + 347,285,816/MAX, `split_rd` 214,568,028/303,650,004/75,913,680/354,715,735, `pb_source_variance` 10,877, `x->source_variance` 11,853), but C's re-searched HORZ_B sub-0 accumulates 494,626,860 (rate 34,532) against the port's reused palette-4 leaf at 427,208,793 (rate 63,246), so C runs out of budget on sub-block 2 (310,011,550 remaining vs that 8x8's own 357,159,754) and keeps SPLIT, while the port's HORZ_B totals 852,710,591 < 880,017,878 and wins the 16x16 (frame +445 B). (#23) **`mbmode_cost` was filled from an all-zero placeholder CDF** under a "confirmed dead in the KEY-frame search" note (`real_costs.rs`) — it is NOT dead at speed >= 8: the nonrd palette shell `av1_search_palette_mode_luma` costs its DC_PRED with `mbmode_cost[size_group_lookup[bsize]][DC_PRED]` (intra_mode_search.c:1139-1140, :1152). `fc->y_mode_cdf` is seeded from `default_if_y_mode_cdf` (entropymode.c:1006) and NEVER adapts on an intra frame (the Y-mode `update_cdf` there runs on `kf_y_cdf`), so the KEY-frame value is exactly that default table — a frame constant, which is also how the dump identifies the arm: ONE distinct `dc_mode_cost` across all 1,513 BLOCK_16X16 palette entries in C **and** in the port, vs 314 (C) / 325 (port) distinct values at BLOCK_8X8 where both take the full-RD `bmode_costs[DC_PRED]`. Measured at BLOCK_16X16 (size group 2): **C 375 vs port 3**, i.e. every nonrd palette candidate's header was 372 (1/512-bit units) cheap. cpu 8, mi(80,280): every other header term matches to the unit (flag 566 / size 307 / uniform 512 / colour 5,120 / map 20,260 / cache {33,252}), C's 2-colour candidate scores 354,174,826 against best 353,694,615 (loses by 480,211) while the port's scores 350,168,236 and wins — the port coded PALETTE(2)+TX_8X8 where the oracle codes no palette + TX_16X16 (frame +200 B). Both cells byte-exact after the two roots. Census **104/104** cells byte-identical across 14 plane dirs (the 102 of roots #18-#21 unchanged + the new `2091x3072_cq62` s4/s8 pair), every gate green, all TEMP probes stripped. **Root #23 also CLOSED `PALETTE_MANY_COLORS_OPEN`** — its two `kb37_nonrd_palette_search` pins were the gates that caught the fix and are RE-PINNED to zero divergences (`["fc256 n40 cq40 (-1)"]` -> `[]`, and the `color_palette_thresh` band `(48, 75)` -> `(0, 75)`), so both are now hard byte gates; the missing DC_PRED term was exactly the size of those near-ties, and the band's byte gate can now witness the `color_palette_thresh` formula (the two hardcoded-constant comparison rows in that test's doc are pre-#23 and NOT re-measured). **Still open in this family:** the tiny-cell class — see C3's `av1_determine_sc_tools_with_encoding` bullet | `kb41_screen_detected_defaults` on `~/tmp/aom_poison_repro/planes_6012.scale2091x3072.png` + the 13 prior plane dirs, `rd_close_palette`, `rd_close_intrabc`, `kb35_nonrd_palette_arm`, `kb37_nonrd_palette_search`, `config_permutations`, `toggles_rd_close`, `kb22_hd_arms`, the s4cov/kb13/kb28/kb31/kb32/kb34 arms | KB-41 |
| **KB-42 (2026-08-30, the six red CI runs)**: `main` CI had been red since `735a0a6d` on **TWO independent roots**, split apart by reading the per-JOB conclusions instead of the run conclusions — run `33302025668` (roots #3-#6) has all four DIFFERENTIAL legs green and fails only the two `portability` legs. **(A)** `content_family_census`'s ceiling half fired: `Screen+screen-knobs intrabc` **33.63 %** against a pinned `[18, 30)`, i.e. the gate reporting the reachability GAIN that roots #3-#6's speed-dependent IntraBC search (30/30 datagen cells byte-identical) legitimately produced; `leaves_le_8px` rose with it (75.19 -> 80.84, passing on a 1.16-point margin). Both re-pinned keeping their own relative shape — intrabc `[25, 42)`, leaves≤8px `[73, 88)`, **both floors RISE** — and `benchmarks/winperf_family_census_2026-08-03.md` re-measured. That gate runs ONLY on the two `portability` legs. **(B)** the 23 byte/RD gates start one commit later at `38a92657` (roots #7-#13) — BISECTED (`encoder_gate_e2e_low_qindex_speed0` passes at `735a0a6d`, fails at `38a92657` with `main`'s exact 5-cell signature); the three runs in between are ~2.5-minute COMPILE failures (`armed_tools_decode_gate.rs` E0027, a pattern missing the new `ToggleKnobs::tune_content_screen`), so no test ran until `4e0229e`. Mechanism: root #12 added `PackCfg::search_tx_mode_is_select` (C's `select_tx_mode`, rdopt_utils.h:390-400, consumed by `update_stats` at partition_search.c:509-527) and switched `derive_real_costs` from `kf.tx_size` to the search shadow `TileCtxState::search_tx_size` (real_costs.rs:188). The shadow is seeded from `kf` at tile start and adapts ONLY under that flag (pack.rs:817) — and **19 of 19 call sites carried the placeholder `false`** (every `zenav1-aom-encode` integration harness + aom-bench's CDEF, superres and inter-P harnesses), so on every non-lossless TX_MODE_SELECT frame the SEARCH's tx-size costs froze at their frame-init snapshot instead of tracking the writer's adapting CDF, flipping near-tie tx-size decisions (worst at low qindex: the failing cells are cq8/cq16). Only `EncodeCell::port_encode` (aom-bench/src/lib.rs:2068) derived it, which is why the KB-41 census stayed 104/104 and the wave never saw it. Fixed by deriving it at all 19 sites (`!coded_lossless`; no harness passes `--enable-tx-size-search=0`, and the header's own TX_MODE_LARGEST can still come from the `txb_split_count == 0` flip, encodeframe.c:2797 — which is exactly why the header mode must not be used here). `zenav1-aom-encode` **316 passed / 0 failed** (was 300/16); CDEF + superres gates green; census 104/104 unchanged. Rule 6 above + `just gate-encode` close the `--lib`-only process hole. **CI run `33325340898` (`cb76cda9`) is green on all seven legs** — linux x86-64, linux forced-scalar, aarch64 default + forced-scalar, windows-11-arm, macos-15-intel, i686; locally aom-encode + aom-bench are 513/0/54 under CI's `--profile test-fast` and identical under `AOM_FORCE_SCALAR=1`, and the KB-41 plane-dir census re-runs at **104/104 `screen=OK (+0)`**. **(C)** the last red test, `nonrd_estimate_arm_palette_round_trips_through_the_c_decoder`, was a test that lied to the port about its own config: it bootstraps with `AV1E_SET_TUNE_CONTENT = AOM_CONTENT_SCREEN` but drove `port_encode_with` without declaring `ToggleKnobs::tune_content_screen`, so root #13's ported `av1_set_screen_content_options` fell through to the `use_nonrd_pick_mode && !hybrid_intra_pickmode` arm (encoder.c:2466-2470, allintra speed 9) and decided screen tools OFF where C took the AOM_CONTENT_SCREEN arm at :2449-2455 — the arm ORDER that test's own doc comment says it exists to check. `38a92657` added the knob and the assertion (aom-bench/src/lib.rs:1914) but not the declaration; the assertion named this as one of its two candidates and it was the right one. 4/4 green. The other candidate — unported `av1_determine_sc_tools_with_encoding` — remains open for the tiny-cell class below, unchanged | `encoder_gate_e2e_byte_match`, `encoder_gate_chroma_ss_e2e`, `encoder_gate_bd10_diff`, `encoder_gate_tune_iq_e2e`, `kb6_real_rd_localize`, `avif_parity`, `encoder_gate_cdef_e2e`, `encoder_gate_superres_e2e`, `content_family_census`, `armed_tools_decode_gate`, `kb41_screen_detected_defaults` | KB-42 |
| **KB-41 roots #24-#26 (2026-08-30, the 41 cells the aom-rs requeue round left)**: the `avifaom-enc-20260830` requeue left **41 real port divergences — 31 screen-detected + 10 photo (`screen_tools=false`, the wave's first non-screen class)**; planes + params + the executor's own verdict per cell are staged at `/mnt/v/output/avifaom-2026-08-30/poison-planes-2026-08-30` (pointer: `zenmetrics/benchmarks/avifaom_poison_planes_2026-08-30.pointer.md`). Payload deltas are TINY (median 26 B, max 1,620 B; 3 cells byte-length-EQUAL but content-different), so all three roots are near-tie flips. The localizer (`kb41_screen_detected_defaults`) was extended to diff the arm each cell's own `sct` PREDICTS — the screen knobs on an `sct=0` photo cell describe nothing — which is what made the photo class readable at all. **(#24) `get_tx_mask`'s mandatory "need at least one transform type allowed" tail was never ported** (tx_search.c:1948-1952). Both est-rd prune arms can legitimately clear the INTER mask: `prune_txk_type_separ` returns `0xFFFF` when its best horizontal candidate is skipped (:1875), and — the case that bit — its combine loop can end with `num_cand == 0`, after which `prune = ~(1 << txk_map[0])` reads the tail slot the skip loop last wrote, a type that need not be in `allowed_tx_mask` at all (:1918-1934). C then pins DCT_DCT; the port evaluated NOTHING, `search_tx_type_inter` returned None, and the CALLER read that as "no valid transform" — dropping an IntraBC candidate whole or invalidating a var-tx block. Localized on the cheapest reproducer in the set, `256x256_cq47_s4` (2,615 B, byte-length-equal, 0.7 s per run): mi(48,60) BLOCK_16X16, TX_16X16, mask `0x0fff`, C's `PSEP-F num_cand=0 map=13,…` ⇒ `~(1 << H_ADST)` ⇒ mask 0 ⇒ DCT_DCT. Both sides search the SAME DV — `IBC-SEARCH bestsme=15494 best_mv=(-65,-80) dv_ref=(-520,-640)` on each — but C's 16x16 IntraBC candidate scores rd 38,375,433 (rate 14,354 / dist 235,456, `y_skip=0`, i.e. the COEFF arm) against the intra winner's 131,342,197, while the port dropped it (`IBC-BAIL yrd`, 238 empty-mask bails in that one tiny frame) and split the 16x16 VERT. Closed **13 of the 31 screen cells — exactly the `--cpu-used 4` subset**, which is the reach: `prune_tx_type_est_rd` is the speed-4/5 arm. **(#25) `av1_allow_intrabc(cm)` was conflated with "the DV search runs".** Root #10 turns the frame-wide DV search off at speed >= 8 (`rt_sf.use_nonrd_pick_mode`, rdopt.c:3432-3434) by leaving `PickFrameCfg::intrabc` `None`; the intra COST path then read `allow_intrabc: cfg.intrabc.is_some()` and dropped `intra_mode_info_cost_y`'s `intrabc_cost[0]` term (:563-564) — but C zeroes `features->allow_intrabc` only AFTER the tiles (encodeframe.c:2444, root #7), so at speed 8 it searches nothing and still charges every intra luma candidate the flag. MEASURED on `256x256_cq14_s8` mi(40,18) BLOCK_8X8: every C `YMODE` rate is exactly **3 units (= `ibc0`) above the port's** (45500/45497, 52319/52316), so the port's running `best_rd` was 58 tighter (1,463,735 vs 1,463,793) — enough for the port's V_PRED ad=-2 uniform tx search to hit `av1_txfm_rd_in_plane`'s `current_rd > ref_best_rd` bail where C completed it, after which `prune_luma_odd_delta_angles_using_rd_cost` dropped ad=-3 as well; C's ad=-3 (rd 1,402,880) was the WINNER and the port coded SMOOTH. Fixed with an explicit `PickFrameCfg::search_allow_intrabc` (the twin of `PackCfg::search_allow_intrabc`, root #8), derived as the detector decision masked with the CLI knob; every other construction site is a non-screen envelope where it is genuinely 0. Closed **15 more screen cells**: after #24+#25 the poison screen set is **26 of 29 stems / 28 of 31 cells byte-exact**. **(#26) `av1_nn_predict` is RTCD-SPECIALIZED and the port transcribed `av1_nn_predict_c`.** `av1_rtcd_defs.pl:467` specializes it sse3/avx2/neon, so a real encode runs `av1_nn_predict_avx2`, which reassociates the dot product into pairwise `hadd` trees and adds the bias LAST. The intra-CNN prune's `av1_nn_output_prec_reduce` 1/512 quantisation hides that *except* at a boundary — and the boundary is exactly where `logits[0]` is compared with `no_split_thresh` (partition_strategy.c:341). Ported as scalar f32 reproducing the AVX2 add tree (`cnn_partition::nn::nn_predict_dispatched`, selected on the HOST's CPU capability the way libaom's own RTCD does — deliberately NOT on the port's `AOM_FORCE_SCALAR`, since that env forces only the PORT's kernels while the linked C still runs AVX2). New gate `cnn_partition_nn_diff::nn_predict_dispatched_matches_dispatched_c` pins it against the DISPATCHED C (new `shim_nn_predict_dispatched` / `ref_nn_predict_dispatched`; RTCD pointers need `ref_init()` or the call segfaults) — **204/204 exact**, including all four real branch geometries. Honest scope: only the AVX2 order is ported (a non-AVX2 host keeps the `_c` order), **and it is deliberately NOT WIRED into `finish_decision` yet** — pairing an AVX2 DNN with the SCALAR CNN below it models neither chain, and doing so broke `cnn_partition_decision_diff::predict_decision_matches_c` against its pinned scalar oracle while still not matching the real encoder (the branch features are already wrong upstream). The switch lands WITH root #27; until then `nn_predict_dispatched` ships as the gated, ready half. **STILL OPEN — root #27, and it is the carrier for the 10 photo cells + the last 3 screen cells (all `--cpu-used 6`)**: `av1_cnn_convolve_no_maxpool_padding_valid` is the CNN engine's one RTCD-dispatched primitive, and the port's transcription target is DELIBERATELY the `_c` variant (`shim/cnn_cscalar.c` pins it so `cnn_partition_cnn_diff` has a stable oracle) — while a real encode runs `_avx2` (x86-64) / `_neon` (aarch64). PROVEN on `2765x4096_cq6_s6` mi(0,352) BLOCK_32X32: the port's 25 DNN features are **bit-identical to the oracle re-run under `AOM_SIMD_CAPS=0`** (0.586125314 / 0.429137081 / 0.158744156 / 0.0466946848 / 0.373701125 / 0.841030896 / 0.0613204911 / 0.0411717147) and differ from the DISPATCHED oracle's in the 7th digit (0.586126685 / 0.429137766 / …); that propagates to raw logits -3.86037111 (port) vs -3.8603348731994629 (C), which `prec_reduce` puts on ADJACENT 1/512 quanta, -3.859375 vs -3.857421875, straddling `no_split_thresh = -3.858222961`. So C keeps `do_square_split = 1` and splits the 32x32 where the port takes `av1_disable_square_split_partition` and codes 32x32 NONE — the first syntax diff at block #298 (`PGATE mi(0,352) bsize=9 sq=1` vs the port's `sq=0`, everything else on the line identical). Corroborated by the family's own gate: `cnn_partition_cnn_diff` prints `205 windows BIT-EXACT vs C-scalar; worst |rust - AVX2| = 7.867813e-6` — the same 7th digit. That file's rationale for tolerating the gap was WRONG and is corrected in place: it claimed the gap only had to stay "far inside the DNN prec-reduce bucket so the downstream split/no-split FLAGS agree (that flag-parity is asserted in the full-model diff)", but (a) a gap need not approach the bucket WIDTH to change the bucket — only to straddle a boundary — and (b) `cnn_partition_decision_diff` asserts flag parity against the C-SCALAR oracle only, so no gate ever covered the claim. The `< 1e-2` assertion is KEPT (unchanged value; it is a real regression ceiling on the transcription) but now says what it actually pins. Cost to close: a bit-exact transcription of `cnn_avx2.c` — only TWO of its specializations are reachable here (5x5/skip-4 for layer 0, 2x2/skip-2 for layers 1-3; everything else falls through to `_c`) — plus its NEON twin for the aarch64 leg. Its own landing, NOT this one. **Round-3 outcome, MEASURED on the fleet** (`zenfleet-ctl requeue --classes encoder_panic` over all 59 still-failing cells, executor rebuilt from a CLEAN tree at this commit as `:exec-zensim944hdr-8662064f`): the ledger moved **125,941 -> 125,973 distinct done, 27 still failing**, and those 27 are cell-for-cell the two registered open classes — **14 tiny** (C3's `av1_determine_sc_tools_with_encoding`; four of the original 18 closed as a side effect of #24/#25) and **13 large** (root #27: the 3 screen cpu-6 cells + all 10 photo). Table: `zenmetrics/benchmarks/avifaom_round3_2026-08-30_open.tsv`; record: `zenmetrics/benchmarks/avifaom_poison_planes_2026-08-30.pointer.md`. Census **104/104** across the 14 plane dirs, `just gate-encode` green | `kb41_screen_detected_defaults` on `~/tmp/kb43/{screen,photo}` + the 14 census dirs, `cnn_partition_nn_diff`, `just gate-encode` (aom-encode + aom-bench integration targets + `content_family_census`) | KB-41 |
| C9 `--use-intra-dct-only=1` (PINNED-OPEN: luma byte-faithful, chroma UV-mode-loop divergence) | `AV1E_SET_INTRA_DCT_ONLY=1` | 64²cq32 / 64²cq63 / 128²cq12 (real content) | +2.23% / 0 (EXACT) / −1.40% | +3.588 (OUT of band) / 0 / +0.333 | `toggles_rd_close::toggles_c9_intra_dct_only_pinned_open` | 2026-07-17 | Y recon identical; first divergent leaf mi(0,0) 32×32: real uv=D45/aduv2 (eob 1) vs port uv=V (eob 78); real winners are derived-type==DCT modes (DCT-forced-search signature). Port UV txb eval + UV mode loop both match the C-pieces oracles under the knob (txfm_uvrd_diff / intra_sbuv_mode_loop_diff sweep green; mask verified vs the REAL facade incl. the PAETH reduced-set reset) ⇒ shared port+oracle mis-model of the REAL UV loop. **Sibling-C dump DONE 2026-07-17** (throwaway ar-swapped libaom, intra_mode_search.c + tx_search.c instrumented, cq32 mi(0,0) 32×32): C evaluates only DC (this_rd 2157931) and D45 (aduv2, this_rd 1985157 — wins); C REJECTS V/H/directionals via `rd_pick_intra_angle_sbuv` anglefail (its inner `av1_txfm_rd_in_plane` returns INT_MAX) and SMOOTH/PAETH via txfmfail. The port instead ACCEPTS V (uv_mode=1, aduv0, DCT-forced tx_type=0, eob=1, **dist=0**, rate 20508 → this_rd 1872917) and V wins. Decisive: C's V prediction `block_sse`=1048576 == the port's V sse=1048576 ⇒ **the prediction MATCHES; NOT a pred bug**. Root = the port's `txfm_rd_in_plane_uv_p` computes V's DCT dist=0 / accepts where C's `av1_txfm_rd_in_plane` rejects V (same pred, same DCT) — a tx-search RD-eval / early-out mis-model shared by the port AND the txfm_uvrd_diff oracle (which is why the differential is green). NEXT: dump C's per-txb V DCT dist/coeffs inside `av1_txfm_rd_in_plane`/`search_txk_type` (the INT_MAX path fires before av1_txfm_uvrd's merge) vs the port's `search_tx_type_intra` V winner, to find why the same DCT residual yields dist=0 in the port and INT_MAX-rd in C. |

## Section C — ABSENT (to port), by family

Status legend: **ABSENT** = no port; **PARTIAL** = kernels/plumbing exist, search/threading/
validation missing. Size: S (≤1 day), M (1–3 days), L (multi-day → decompose). C entry
points are libaom v3.14.1 (`reference/libaom`). Defaults verified in
`av1/av1_cx_iface.c` (allintra override block :3065–3078 sets ONLY `enable_cdef=0`,
`screen_detection_mode=ANTIALIASING_AWARE`, `qm_min=4`, `qm_max=10`).

### C1 — CDEF strength search — **PORTED, BIT-IDENTICAL → section A** (2026-07-17)
- Landed 016d4dd (`aom-encode/src/pickcdef.rs`, the full `av1_cdef_search` + FAST-level
  tables) + 9850da6 (`pack_tile_from_trees` two-pass pack + `write_cdef` literal wiring) +
  c9ebf83 (the byte-identity gate, 14/14 EXACT). See the section A row + STATUS.md
  2026-07-17 for the full inventory.
- Remaining sub-scope (honest fractions): e2e-gated = speed-0 `CDEF_FULL_SEARCH` only;
  `CDEF_FAST_SEARCH_LVL1..5` are ported + table-unit-tested but not yet e2e-gated
  (cheap extension: CDEF-on cells at `--cpu-used=1..6`); `CDEF_PICK_FROM_Q`
  (speed≥7 rt) + `CDEF_ADAPTIVE` (`tune=IQ/SSIMULACRA2`, off at cq≤32) NOT ported
  (documented-dead for `--enable-cdef=1`); SB128 CDEF-on blocked on the pack's SB64
  envelope (the search's >64-fb arms are already in place).

### C2 — Loop-restoration search (Wiener/SGR) — **PORTED, BIT-IDENTICAL → section A** (2026-07-17)
- `--enable-restoration` / `AV1E_SET_ENABLE_RESTORATION`. **Allintra config default is ON
  (=1)** — verified: `default_extra_cfg.enable_restoration = 1` (av1_cx_iface.c:286),
  threaded non-realtime at :1273, NOT touched by the allintra override block. A DEFAULT
  allintra aomenc encode RUNS `av1_pick_filter_restoration` (sometimes resolving all-NONE,
  but the seq/frame header bits differ from `=0`) — this family was the highest-priority
  default-parity gap, now closed at the knob level.
- **Landed (4 chunks):** e24cf09 (write-side syntax: binary-codes writer primitives +
  `write_lr_unit`, byte-identical to the REAL C writer + exhaustive count parity);
  96d3464 (search numeric core: `compute_stats[_highbd]`, `pixel_proj_error`,
  `calc_proj_params`/`get_proj_subspace`, SGR flt producer — all diffed vs EXPORTED `_c`
  fns; Wiener solve chain transcribed, no C export exists); dfd757e (decision layer:
  per-unit RD searches, SB-coding-order walk, unit-size descent, `pick_filter_restoration`);
  96534c4 (`pack_tile_lr` RU-interleaved SB-root writes + `port_encode_lr` pipeline:
  LF apply → search → repack → derived restoration header; gate). Gate hardened to full
  byte-identity + decision-equality asserts after measuring 8/8 EXACT.
- Chunk-5 outcome (2026-07-17): the **format axis is now byte-exact** — `mono / 4:4:4 /
  bd12` speed-0 cells assert **3/3 BYTE-IDENTICAL** (`lr_restoration_gate.rs::
  lr_restoration_format_axis`), extending the proven LR-search coverage to 1-plane LR,
  full-res chroma LR, and the highbd-12 path (compute_stats divider 16, SGR 12-bit clamps).
  The allintra speed-1..4 arms (`lr_search_sf_allintra`) and GOOD-mode cells staged in the
  chunk-5 WIP are **LR-orthogonal near-ties, PINNED (not gated)**: a base-vs-LR split
  (throwaway `lr_localize` harness) showed the speed>=1 cells diverge in the BASE encode
  itself — the LR-OFF stream already differs (s1 real content, first byte 3, both off and
  on) — because the port's real-content speed>=1 base encode is not yet byte-exact (KB-6
  proved real content only at speed 0; the KB-8..11 speed gates are synthetic). The GOOD
  cells derive `set_allintra` base speed-features (the harness has no `set_good`), so their
  base search mismatches C's GOOD encode. `lr_search_sf_good` is now source-verified vs
  speed_features.c (:1164 `reduce_wiener_window_size` is UNCONDITIONAL — unlike allintra's
  speed>=3 gate; :1352-1358 is at speed>=3 — the WIP's `speed>=4` was an off-by-one,
  corrected) and ready for a future GOOD gate. speed>=5 allintra is structurally LR-off in
  C (sf disable + seq-bit clear). `pack_tile_from_trees` unification (reuse the CDEF
  two-pass pack instead of the re-search repack) queued as an optimization.
- Decoder-side LR (apply path) was already complete + gated pre-pivot (section A decoder
  rows).
- **DEFAULT-PATH WIRING (2026-07-18): default-config parity closed.** The byte-exact search is
  now wired into the port's DEFAULT allintra path — `aom-bench::EncodeCell::port_encode` derives
  the LR stage from the frame's `enable_restoration && !coded_lossless` (C's `is_restoration_used`,
  encoder.h:4431; the parsed seq bit already encodes C's speed>=5 clear). New gate
  `lr_default_parity.rs` asserts the port's default `port_encode` frame-OBU is **BYTE-IDENTICAL to
  a plain `aomenc --allintra`** (no tool flags) on 8/8 real-content cells, where the reference is
  the new `shim_encode_av1_kf_defaults` (every coding-tool control at its allintra default:
  cdef OFF, restoration ON, qm OFF). It also asserts `c_encode_defaults() == c_encode_lr()`
  (restoration's default IS on; palette/intrabc/deltaq inert on non-screen stills) and that the
  default stream differs from `--enable-restoration=0` even on the all-RU-NONE cell (header bits).
  The explicit-off `encoder_gate_e2e_*` gates stay valid as `--enable-restoration=0` config tests.

### C3 — Screen-content tools — ABSENT (L, decompose) — bulk agent live (#29)
- Palette search: `--enable-palette` (default ON, gated on `allow_screen_content_tools`).
  C: `av1/encoder/palette.c` `av1_rd_pick_palette_intra_sby/_sbuv` (k-means),
  `intra_mode_search.c` `av1_search_palette_mode_luma`;
  `intra_sf.{prune_palette_search_level, prune_luma_palette_size_search_level,
  early_term_chroma_palette_size_search}`. **MOVED to section B (2026-07-17)** — the Y+UV
  searches + palette recon + pack syntax/map tokens landed RD-close (5/7 cells byte-exact).
  Remaining inside the family: `av1_search_palette_mode[_luma]` (inter-frame callers, out of
  stills scope).
- IntraBC: `--enable-intrabc` (default ON, screen-gated). C: `av1/encoder/rdopt.c`
  `rd_pick_intrabc_mode_sb`, DV hash `av1/encoder/hash_motion.c`,
  `mv_sf.intrabc_search_level`. **SEARCH + skip-arm + full wiring LANDED 2026-07-18** —
  `rd_pick_intrabc_mode_sb` (`aom-encode/src/intrabc_search.rs`) is WIRED (rd_pick.rs step 6 →
  real) and runs under the screen-content gate (`p.allow_intrabc`):
  - **Full-pel DV search: hash + NSTEP diamond + mesh.** The source-frame hash (chunk 3a)
    + the ported `full_pixel_diamond` (NSTEP site config, `diamond_search_sad`
    coarse→fine with the `UPDATE_SEARCH_STEP` num00 collapse) + the `full_pixel_exhaustive`
    mesh (screen `exhaustive_searches_thresh = 1<<20`), SAD-metric walk / variance-cost
    result; the pixel search ALWAYS runs at `intrabc_search_level 0` (rdopt.c:3570).
    Geometry unit-locked (`nstep_config_matches_c`, `mv_step_param_matches_c`,
    `diamond_finds_exact_repeat`). The HASH is square-gated (mcomp.c:1918); the diamond
    runs for every bsize (non-square intrabc supported).
  - **`predict_skip_txfm` (tx_search.c:183)** + `set_skip_txfm` hbd sse scaling: the port
    offers an intrabc candidate ONLY in the skip regime (luma predict_skip fires AND chroma
    is an exact match), where `av1_txfm_search` forces `skip_txfm=1` and BYPASSES the coeff
    arm — there the skip RD (`rate=mode+mv+skip1`, `dist=sse`) is byte-exact.
  - **Wiring:** `LeafWinner`/`RdPickIntraBest` use_intrabc/dv/dv_ref/skip; `ModeGrid` DV grid
    (`dc_screen`, 25 stamp sites) for `find_dv_ref_mvs`; `PickFrameCfg::intrabc`
    (`IntrabcFrameCfg`: hash/dv_costs/txfm_partition_costs/error+sad_per_bit/mv_step_param);
    `encode_b_intra_dry` intrabc arm (predict-from-recon + skip entropy reset + skip txfm ctx);
    `pack_leaf` intrabc arm (use_intrabc + DV diff via `write_mb_modes_kf_fc`, skip tx/coeff);
    harness (`build_intrabc_hash_table` from source luma, `PackCfg::allow_intrabc`, LF forced
    0 for intrabc frames, `ToggleKnobs::enable_intrabc`).
  - **PINNED — real-content byte-exactness blocked on the inter var-tx COEFF ARM (L).** Real
    screen content codes the MAJORITY of intrabc blocks via the COEFF arm (nonzero quantized
    residual) and as NON-SQUARE shapes — measured on a 196² conformance crop: C uses 49
    intrabc blocks, **39 coeff-arm + 42 non-square**, only 10 skip. The
    `av1_pick_recursive_tx_size_type_yrd` inter var-tx quadtree + `prune_tx_2D` /
    `ml_predict_tx_split` NN prunes + the var-tx pack are NOT ported, so the port codes those
    blocks as intra and the frame diverges. Gate `rd_close_intrabc::intrabc_dv_search_pinned`
    (aom-bench) asserts the content is anti-vacuous (C genuinely codes intrabc) and PINS the
    divergence self-promotingly (a byte-match → promote). The 420 skip subset additionally
    needs a chroma-eob-0 check (currently `chroma_sse==0`, exact-only). Envelope untouched:
    non-screen frames (`allow_intrabc=0`) are byte-inert — palette gate + partition_pick_diff
    + rd_pick_intra_sb_diff green.
- Screen detection: `--screen-detection-mode` (allintra default ANTIALIASING_AWARE=2).
  C: `av1/encoder/encoder.c` `av1_set_screen_content_options`. **PORTED 2026-08-30 (KB-41
  root #13)** arm-by-arm — `screen_detect.rs`; the port no longer takes
  `allow_screen_content_tools` purely as an input.
- **`av1_determine_sc_tools_with_encoding` (encoder_utils.c:1214) — ABSENT (M), the NEXT OPEN
  CLASS in this family.** The second, *encoding-based* screen decision: on a KEY frame, when
  the detector said "not screen" and `!rt_sf.use_nonrd_pick_mode` (so allintra **below speed
  8**), C runs **two extra whole-frame trial encodes** and can turn screen tools ON anyway
  (with `allow_intrabc = cpi->intrabc_used`). Reproducer: 35 tiny datagen cells, e.g.
  `~/tmp/aom_poison_repro/refs/8468.scale59x128.png`, staged as `59x128_cq44_s4` and
  `59x128_cq50_s6` (both fail identically: detector says 0, the oracle header says 1, with
  `palette=0 intrabc=0 photo=6 fast=true`); the bench asserts
  the ported decision against the oracle header and names this arm when it fires
  (`kb41_screen_detected_defaults`, planes dir
  `~/tmp/aom_poison_repro/planes_8468.scale59x128.png` — **that dir is INCOMPLETE as
  of 2026-08-30: it holds `59x128_cq44_s4.{json,u,v}` with the `.y` plane MISSING and
  no `59x128_cq50_s6` at all, so the census binary panics on it; re-dump from the
  intact `~/tmp/aom_poison_repro/refs/8468.scale59x128.png` before using it**).
  After the 2026-08-30 round-3 requeue the class is **14 cells, not 18** — four closed
  as a side effect of roots #24/#25 — and the survivors are 12 renditions of
  `8468.scale59x128` plus `5052.scale78x128` and `8020.scale115x128`
  (`zenmetrics/benchmarks/avifaom_round3_2026-08-30_open.tsv`). What it needs, in order:
  1. `set_encoding_params_for_screen_content` (encoder_utils.c): pass 0 = tools OFF, pass 1 =
     `allow_screen_content_tools = 1` (intrabc stays 0 — the C's own TODO), both with
     `part_sf.partition_search_type = FIXED_PARTITION` + `fixed_partition_size = BLOCK_32X32`.
     The port has `rd_use_partition_real`, but no FIXED_PARTITION driver wired to it.
  2. `q_for_screen_content_quick_run = AOMMAX(q_orig, 244)` (or `q_orig` when lossless), then
     `av1_set_quantizer` + `av1_set_speed_features_qindex_dependent` + `av1_init_quantizer`
     re-run at that q for the trials, and RESTORED afterwards.
  3. `cpi->rc.projected_frame_size` — i.e. each trial must be **packed**, not just searched.
  4. `aom_calc_psnr` / `aom_calc_highbd_psnr` of source vs the trial reconstruction, at the
     STREAM bit depth (encoder_utils.c:1245-1253), both passes.
  5. `cpi->palette_pixel_num` accumulated during the pass-1 encode, and `cpi->intrabc_used`.
  6. The decision (`screen_content_tools_determination`): `psnr_diff = psnr[1] - psnr[0]`,
     `palette_ratio = palette_pixel_num / (w*h)`, ON iff
     `psnr_diff > STRICT_PSNR_DIFF_THRESH` (0.9, encoder_utils.c:1123) **or** (`palette_ratio >= 0.0001` and
     `psnr_diff / palette_ratio > 4`); otherwise the detector's original decision stands.
  Cost is dominated by items 1 and 3 (a fixed-partition encode + pack that the port has never
  driven), not by the arithmetic — this is NOT a one-sitting port, which is why roots #22/#23
  landed without it.
  **NEW EVIDENCE 2026-09-03 (issue #15, `self_contained_key_frame.rs`, commit `65ffb75d`):**
  the reproducer geometry above (`8468.scale59x128.png`'s 59x128, plus 78x128/115x128) does
  NOT panic on `encode_key_frame` — the standalone shell this session's landing built, which
  did not exist when the 14-tiny poison class was discovered (that class ran through the
  bootstrap-driven `port_encode` path). 25/25 byte-exact on the same GEOMETRIES (source pixels
  weren't preserved in the poison dump, so this is a geometry-class regression probe, not a
  literal replay) across the two poisoned speeds (4, 6) + a control, texture and checker
  content, and the full cq ladder on the worst-hit dimension. Separately, TWO adversarial
  differential probes (`probe_sc_tools_trial_gap_on_detector_negative_content`,
  `probe_sc_tools_trial_gap_flat_patch_on_small_noisy_frame`; 105 cells total) tried
  purpose-built content designed to sit near or below the base detector's own threshold while
  still being genuinely palette-friendly, including a checker-patch size sweep that BRACKETS
  the detector's threshold crossover from both sides on a small (64x64) frame — exactly where
  a second-opinion trial encode would most plausibly disagree with the block-counting
  heuristic. Zero divergences found, on either side of the boundary. This does not close this
  bullet or shrink the port estimate above — it is evidence that `encode_key_frame`'s specific
  ALLINTRA/single-KEY-frame envelope has not yet been shown to need it, tested with content
  designed to find a counterexample rather than content designed to avoid one. Kept as
  permanent regression probes in that file.
- `--tune-content` screen/film forcing (gates the above). (S)

### C4 — tune=IQ / tune=SSIMULACRA2 family — **PORTED, BIT-IDENTICAL → section A** (2026-07-17)
The tune bundle (`handle_tuning`, av1_cx_iface.c:1938–1978): `enable_qm=1, qm_min=2,
qm_max=10, sharpness=7, dist_metric=QM_PSNR, enable_cdef=ADAPTIVE, enable_chroma_deltaq=1,
deltaq_mode=6 (VARIANCE_BOOST)`; IQ adds `enable_adaptive_sharpness=1`.
- **DONE — every piece e2e byte-identical to real aomenc + the full composite bundle** (all OFF
  by default; `TuneKnobs::default()` = PSNR, the proven envelope is untouched). Gate:
  `encoder_gate_tune_iq_e2e` (9 tests) + `qm_level_diff` + `tune_shim_smoke`:
  1. QM-level formulas `aom_get_qmlevel_luma_ssimulacra2` + `aom_get_qmlevel_444_chroma`
     (quant_common) — `qm_level_diff`, byte-exact vs the real C static inlines.
  2. QM-PSNR dist metric (`dist_block_tx_domain_qm`; trellis `optimize_txb_qm` +
     tx-search transform-domain distortion weighted by the forward QM only under
     `use_qm_dist_metric`; txb_rdopt.c:346-351/:378-386) + `tune_shim_smoke` anti-vacuous.
  3. `--sharpness` 0..7 (`av1_build_quantizer` rounding bias + trellis + LF level) + witness.
  4. `--enable-chroma-deltaq` (`av1_set_quantizer` chroma delta-q arms, port-derived +
     cross-checked vs the real header).
  5. `--enable-adaptive-sharpness` (qindex-adaptive LF sharpness cap, picklpf.c) + witness.
  6. `--deltaq-mode=6` Variance Boost (`allintra_vis.rs` per-SB source-variance qindex
     modulation; `pack_tile`/`pack_tile_from_trees` per-SB delta-q threading) + witness.
  - **Composite** (`encoder_gate_tune_composite_full_e2e`): the whole `--tune=iq` /
    `--tune=ssimulacra2` bundle live at once, 54/54 cells byte-match (mono/420/444 ×
    64/128/192 × cq12/32/50), proving the knobs compose.
- **One bundle member NOT in the tune port: CDEF_ADAPTIVE.** The composite gate overrides
  `enable_cdef=0` — CDEF is the separate, already-bit-exact C1 track (`av1_cdef_search`), applied
  post-reconstruction so it is symbol-inert on the coded tile bytes. The tune-family port
  deliberately does not own it; a full tune=IQ *with* CDEF-adaptive needs the C1 CDEF search wired
  under the per-SB tune qindex (deferred, cross-track).

### C5 — aq-mode / deltaq-mode variants — PARTIALLY DONE (mode 2 + mode 3 + mode 6 bit-exact → section A)
- `--deltaq-mode=2` DELTA_Q_PERCEPTUAL (wavelet AC energy) — **PORTED, BIT-IDENTICAL → section A** (2026-07-18):
  the 5/3 dwt Haar-AC energy (dwt.c) → `av1_block_wavelet_energy_level` → the rate-ratio segment
  qindex (`av1_compute_q_from_energy_level_deltaq_mode` + `av1_compute_qdelta_by_rate` / the
  `av1_rc_bits_per_mb` KEY/AOM_Q arm / `find_qindex_by_rate`), per-SB threaded through the shared
  `DeltaQFrameCtx` + `av1_adjust_q_from_delta_q_res` (`setup_delta_q_perceptual`). 7/7 real-content
  cells byte-match real aomenc `--deltaq-mode=2` (`deltaq_mode2_perceptual_wavelet_e2e`); the dwt
  kernel is differentially locked vs the exported C (`deltaq_perceptual_wavelet_diff`).
  **Follow-ups (NOT done):** the highbd (bd10/12) dwt arm; the partial-SB source-border extension
  (dims not a multiple of 64px). Note `is_screen_content_type` (the rate enumerator 2M/1M) cancels
  in the ratio, so the non-screen envelope is faithful; a screen-detection port would be needed for
  screen content. (M)
- `--deltaq-mode=3` DELTA_Q_PERCEPTUAL_AI — **PORTED, BIT-IDENTICAL → section A** (2026-07-17):
  `av1_set_mb_wiener_variance` (per-8x8 intra-SATD search + FP-quantize + Weber stats + the
  2-iteration `norm_wiener_variance`) + `av1_get_sbq_perceptual_ai` + `av1_get_deltaq_offset`,
  per-SB pack threading reusing the mode-6 `DeltaQFrameCtx`; 7/7 real-content cells byte-match
  real aomenc `--deltaq-mode=3` (`encoder_gate_deltaq_mode3_e2e`). **Highbd (bd10/12)
  FP-quantize arm — DONE (this landing):** `av1_set_mb_wiener_variance` dispatches
  `av1_highbd_quantize_fp` for bd>8 (the sole bd8-specific step); bd10 (real b10 content) +
  bd12 (bd10 content ×4-promoted) cells added to the gate (16/16 byte-identical). **Follow-ups
  (NOT done):** the partial-SB source-border extension (frames whose
  dims aren't a multiple of 8px — the KB-6 partial-SB analogue); `--enable-rate-guide-deltaq`
  (the `get_rate_guided_quantizer` arm reading an external rate file — needs the file plumbing +
  `ext_rate_guided_quantization`); `--auto-intra-tools-off` (`automatic_intra_tools_off` +
  `model_rd_sse` accumulation, which disables smooth/paeth/cfl/diagonal intra on high-quality
  low-q frames — a separate intra-tool gate).
- `--deltaq-mode=6` DELTA_Q_VARIANCE_BOOST (tune=IQ default): **DONE, BIT-IDENTICAL** (landed
  with C4, fed362b) — `allintra_vis.c`
  `av1_get_sbq_variance_boost`, `aq_variance.c` `av1_get_variance_boost_block_variance`,
  `--deltaq-strength`. (M)
- `--deltaq-mode=1` OBJECTIVE (base default) is TPL-gated (encodeframe.c:343) — **inert
  for a lone still**; document-only.
- `--delta-lf-mode=1` DELTA_LF — **PORTED, BIT-IDENTICAL → section A** (2026-07-18): per-SB
  `delta_lf_from_base` derived from `delta_qindex` in `pack_leaf` + the LF-pick delta-lf
  application (`stamp_lf_delta_lf` + `LfSearchFrame::delta_lf_present`). Rides on a firing
  delta-q mode (gated tested with `--deltaq-mode=2`). **Follow-ups:** `delta_lf_multi=1`
  (per-plane-type deltas — `DEFAULT_DELTA_LF_MULTI=0` so untested), highbd, partial-SB.
- `--aq-mode=1/2` (variance/complexity segmentation): `aq_variance.c` `av1_vaq_frame_setup`
  (VARIANCE_AQ) / `aq_complexity.c` `av1_setup_in_frame_q_adj`+`av1_caq_select_segment`
  (COMPLEXITY_AQ). **Fires single-pass** (encoder.c:3494, NOT two-pass-gated; single-pass
  uses degenerate `avg_energy=0` → `avg_ratio=rate_ratio[0]=2.2`), but is the **first
  single-frame use of SEGMENTATION ENCODE**: it enables `cm->seg`, sets 8 per-segment
  `SEG_LVL_ALT_Q` deltas (`av1_compute_qdelta_by_rate(rate_ratio[i]/avg_ratio)` — the same
  rate model as mode 2, already ported), selects `mbmi->segment_id` per block from the block
  energy (VARIANCE_AQ: `mbmi->segment_id = av1_log_block_var(...)`-mapped, partition_search.c:
  603-608; COMPLEXITY_AQ: `av1_caq_select_segment`, :963), codes `write_segment_id` (writer
  proven), and re-selects per-segment quantizers in the RD search + pack. The seg-map decision
  + per-segment quantizer threading through the search/pack is the remaining work. (M–L,
  segmentation-encode plumbing). NOTE: `av1_log_block_var` is used but a SEPARATE fn from the
  mode-2 wavelet path.
- `--deltaq-mode=4` USER_RATING_BASED: `av1_get_sbq_user_rating_based` reads an EXTERNAL per-SB
  rating map `cpi->mb_delta_q` (from `--rate-distribution-info` / `AV1E_SET_RATE_DISTRIBUTION_INFO`)
  — needs the external-file plumbing; ABSENT.
- `--deltaq-mode=5` HDR / `enable_hdr_deltaq`: `av1_get_q_for_hdr` **asserts bd10** AND is a
  NO-OP under `DISABLE_HDR_LUMA_DELTAQ=1` (encoder.h:101 — returns `base_qindex`, so
  `deltaq_used=0` → `delta_q_present` resets to 0). **INERT** in the shipped build; document-only.
- `--enable-rate-guide-deltaq` / `--rate-distribution-info` (`get_rate_guided_quantizer`,
  allintra_vis.c:688) — needs the external rate-file plumbing (`ext_rate_guided_quantization`);
  ABSENT.
- `--auto-intra-tools-off` (`automatic_intra_tools_off` + `model_rd_sse`, allintra_vis.c:515) —
  needs `--deltaq-mode=3`; disables smooth/paeth/cfl/diagonal intra on high-Q low-q frames via
  a `model_rd_sse` frame accumulation gate. Self-contained (no seg/LF/external), a moderate
  intra-search-space arm; ABSENT.
- Encoder-side per-SB delta-q/delta-lf tile signaling (writer side): **DONE** (delta-q via mode
  2/3/6, delta-lf via `--delta-lf-mode=1`).

### C6 — Superres (encode side) — FIXED + QTHRESH/AUTO/RANDOM + denom-16 scaler DONE; only the AUTO recode loop remains (inter follow-up)
- `--superres-mode/-denominator/-kf-denominator/-qthresh/-kf-qthresh`. Default NONE.
- **DONE (Section A): FIXED mode — 8-bit 13/13 + 10/12-bit 16/16, byte-identical** — source downscale
  (`av1_resize_plane`, `aom_encode::resize`, differentially bit-exact vs exported C:
  `resize_plane_diff` 5 tests) + coded-width encode + `write_superres_scale` header
  signalling. Gate `encoder_gate_superres_{fixed_real_content,fixed_mono}_rd_close`. The
  source downscale for an ALLINTRA KEY still takes the `DISALLOW_RECODE` `encode_without_recode`
  path → `av1_resize_and_extend_frame_nonnormative` → `av1_resize_plane` (verified in
  `reference/libaom`), so there is NO recode loop for FIXED stills.
- **Remaining:**
  1. **Highbd (10/12-bit) downscale — DONE (this landing)** — `highbd_resize_plane`
     (resize.c:771+, u16 `highbd_interpolate`/`highbd_down2_*`) wired as the bd>8 arm in the
     gate's `downscale_plane`; `encoder_gate_superres_fixed_highbd_rd_close` 16/16 byte-identical
     (bd10/12 × 4:2:0 denoms 9/12/14 cq{20,48} + mono denoms 9/12 cq32). Also fixed a
     byte-neutral aom-dist `highbd_variance64` SIMD edge panic (non-mult-of-8 visible-only SSE
     → scalar twin; verified 0-failed in both dispatch modes).
  2. **8-bit denom-16-even-width corner — DONE (2026-07-18, 6b77342)** — the exact-1/2 horizontal
     ratio trips libaom's OPTIMIZED `av1_resize_and_extend_frame` (`aom_scaled_2d`, `EIGHTTAP_SMOOTH`
     / phase 8), ported as `optimized_downscale_plane_8bit` (edge-extend + 16×16-block separable
     8-tap convolve). Differentially bit-exact vs the exported `av1_resize_and_extend_frame_c`
     (`resize_opt_scaler_diff`, 5 content × luma/chroma/other dims); wired into the gate's bd8
     downscale (all-planes-or-none per C). bd8 denom-16 QTHRESH/AUTO cells now emit BYTE-IDENTICAL
     streams to real aomenc.
  3. **AUTO / QTHRESH / RANDOM denom selection — DONE (2026-07-18, 3c8a8c2)** —
     `calculate_next_superres_scale` (superres_scale.c) ported as `aom_encode::superres_select` for
     the single-frame KEY/AOM_Q envelope: `analyze_hor_freq` (16×4 H_DCT energy, bit-exact vs the
     exported `av1_fwd_txfm2d_16x4` — `superres_select_diff` facade + e2e) +
     `get_superres_denom_from_qindex_energy` + the QTHRESH gate (q vs `--superres-kf-qthresh`); AUTO =
     allintra `Dual` (qthresh 0, same energy derivation); RANDOM = the process-global static-seed lcg
     (34567 → 11,14,15,9). `q` is the AOM_Q cq-qindex (`rc::base_qindex_from_cq`, #8). The port
     re-derives the denom the real encoder chose (embedded in the stream) and matches it for EVERY
     cell (QTHRESH 21/21, AUTO 11/11 bd8/10/12; RANDOM 4 draws), then reproduces the coded bytes
     (RANDOM 4/4 byte-identical real content; QTHRESH/AUTO bd8 engaged denoms 9/10/16 byte-identical).
     Gates: `encoder_gate_superres_{qthresh,auto,random}_e2e`. Superres stays OFF by default.
  4. **AUTO recode loop — remaining (inter/GOP follow-up).** `av1_superres_in_recode_allowed` is
     AUTO && !SOLO && `frames_to_key>1`; a single-frame KEY still has `frames_to_key<=1`, so the
     recode loop NEVER fires for it (confirmed by the AUTO e2e denom match — the non-recode denom is
     exact). The `SUPERRES_AUTO_DUAL` multi-pass recode search is only reachable with a multi-frame
     GOP, so it is out of the single-frame KEY scope. **Decoder note, CORRECTED 2026-07-31:** this line
     used to say the port DECODER's superres denom-16 (exact-2:1) upscale diverges from C. It does
     not, and the cited KB was already recording that: KB-14 is FIXED, and its root was a header
     coded-lossless false-positive — explicitly **NOT the upscale**. The gate
     `superres_denom16_single_sb_column_byte_identical_kb14` (`aom-decode/tests/superres_diff.rs:315`)
     is green in-tree (re-run 2026-07-31: 5/5 in that file). Superres decode is byte-identical to C,
     now including the multi-tile-column walk (`superres_tiles_diff.rs`, 44 streams). The encoder is
     byte-exact as this row already said.

### C7 — Film grain / denoise estimation — table-inject DONE (byte-exact → section A); estimation ABSENT (L)
- **`--film-grain-table` — DONE (this landing, → section A).** Ported `aom_dsp/grain_table.c`
  (`aom_film_grain_table_read` + `grain_table_entry_read` + `aom_film_grain_table_lookup`, plus
  `_write` for fixtures/round-trip) as `aom-encode/src/grain_table.rs`; wired
  `EncodeCell::port_encode_film_grain` to inject the port's own table-derived `FilmGrainParams`
  into the frame header (context fields from the cell), written by the already-bit-exact
  `write_film_grain_params`. Gate `film_grain_gate.rs` byte-matches real aomenc
  `--film-grain-table` (C shims `shim_write_grain_table_test_vector` +
  `shim_encode_av1_kf_film_grain_table`, the latter replicating the plain `encode_kf_pass`
  control set so grain adds ONLY header bytes). See the section-A row.
  - `--film-grain-test` (`AV1E_SET_FILM_GRAIN_TEST_VECTOR`, built-in `grain_test_vectors.h`)
    shares the identical param-plumbing + writer; the table-inject gate uses those 16 vectors as
    the shared fixture source (via the C `aom_film_grain_table_write`), so the test-vector param
    set is transitively covered. A direct `--film-grain-test` e2e gate (reusing the existing
    `ref_encode_av1_kf_film_grain` shim) is a trivial follow-up if a distinct knob is wanted.
- `--denoise-noise-level/-block-size`, `--enable-dnl-denoising`: noise-model ESTIMATION +
  source denoise + grain fit. C: `aom_dsp/noise_model.c` (`aom_denoise_and_model_run` →
  `aom_flat_block_finder_*` → `aom_wiener_denoise_2d` (FFT) → `aom_noise_model_*` →
  `aom_noise_model_get_grain_parameters`). (L — **all `double`/FFT float math**, so a byte-exact
  `--denoise-noise-level` stream is float-determinism-gated; the realistic deliverable is
  per-kernel DIFFERENTIAL validation against the exported `aom_noise_*`/`aom_flat_block_finder_*`
  `_c` functions. Decompose: (2) `noise_strength_solver` (linear system) — **DONE (this landing):**
  `aom-encode/src/noise_model.rs` (`linsolve` + `NoiseStrengthSolver` add/solve/get_center/get_value
  + `NoiseStrengthLut` eval/fit_piecewise), differential `noise_strength_solver_diff.rs` 300/300
  trials bit-identical to the exported `aom_noise_strength_solver_*` / `_lut_eval` /
  `_fit_piecewise` across bd 8/10/12, varying bins/obs + a near-singular case (C-quirk faithful: the
  greedy LUT reduction leaves the `residual` array un-shifted on point removal). (3) `flat_block_finder`
  (planar-model + threshold) — **DONE (this landing):** `FlatBlockFinder` (init lazy-inverse of the
  3-param planar `AᵀA`, `extract_block` planar-fit residual, `run` gradient-covariance eigen features
  + hard `is_flat` threshold + 10th-percentile sigmoid ranking), differential `flat_block_finder_diff.rs`
  48/48 trials bit-identical to the exported `aom_flat_block_finder_init`/`_run` (flat_blocks map +
  num_flat) across bs 16/32 × bd 8/10/12 × exact+partial-edge sizes, anti-vacuous (356 flat / 3060
  non-flat cells). NB the `exp` sigmoid (percentile arm) matches C's glibc `exp` bit-exactly on this
  host; `is_flat` + all features are exact `f64`/`sqrt`. (4) `noise_model` (AR estimate +
  `get_grain_parameters` quantize), (5) `wiener_denoise_2d` + FFT, (6) `denoise_and_model_run`
  orchestrator + encoder wiring — all still ABSENT.)

### C8 — Partition controls — disable arms DONE (byte-exact); SB128 encode DONE (byte-exact)
- **DONE (this landing, → section A):** `--enable-rect-partitions=0`, `--enable-ab-partitions=0`,
  `--enable-1to4-partitions=0`, `--min-partition-size`, `--max-partition-size` + the
  square-only 8..32 interaction arm — all BYTE-IDENTICAL vs real aomenc (same ctrl) on the
  real-content grid, hard-pinned in `toggles_rd_close` (aom-bench). Infra: generic ctrl-pair
  shim `shim_encode_av1_kf_ctrls` + `ToggleKnobs`/`port_encode_with`; ctrl-id constants
  header-cross-checked (`cx_ctrl_ids_match_reference_headers`). C mapping verified:
  `set_max_min_partition_size` (partition_strategy.h:214) `min(sf_default, dim_to_size(px),
  sb)` / `min(max(BLOCK_4X4, dim), sb)`; the auto-max ML arm is inter-only.
- **`--sb-size=128` ENCODE side — DONE ✅ (byte-exact, this landing):** the encoder search+pack
  now walk the frame in 128×128 superblocks and byte-match real `aomenc --sb-size=128` on
  real-image content ≥128px. Three gates in `sb128_e2e.rs` (aom-bench), each with a
  sb128-vs-sb64 anti-vacuity witness: (1) `sb128_forced_split_e2e`
  (`--sb-size=128 --max-partition-size=64`, forced SPLIT at the 128 root, all ≤64 leaves —
  isolates the 128-SB geometry + the 8-way BLOCK_128X128 partition symbol/context/cost + the
  pack walk over 128-SBs); (2) `sb128_natural_e2e` (plain `--sb-size=128` on real `quantizer-00`
  crops — the RD search EVALUATES the 128×128 NONE candidate, exercising the mu-64 SEARCH tx
  walks, then resolves to SPLIT/≤64 on this textured content); (3) `sb128_coded_128_leaf_e2e` (a
  smooth diagonal ramp at 256² cq55/cq63 — the content real aomenc actually resolves to a coded
  128-level partition [anti-vacuity-checked natural≠forced-split], so THIS gate exercises the pack
  `av1_write_intra_coeffs_mb` L/U/V 64-chunk interleave + the >64 re-encode; the photographic
  `quantizer-00` crops split to ≤64 even at cq63 and never reach it). Real-image cells:
  128² (1 SB) + 256² (2×2 SBs) × cq{12,32,63}.
  Harness threads the live SB geometry off the bootstrap seq header's `use_128x128_superblock`
  bit (aom-bench `port_encode_full`: `sb_block`/`sb_mi`/`sb_px`); the 4-way stage gets the C
  `bsize != BLOCK_128X128` gate (partition_search.c:4166). The `av1_foreach_transformed_block
  _in_plane` mu-64 chunk walk (encodemb.c:560-582) is ported into every >64 predict/reconstruct
  site — the two search tx walks (`txfm_rd_in_plane_intra`, `intra_model_rd_y`), the chroma RD
  walk, the two re-encode plane walks (`encode_intra_block_plane_y`/`_uv`) + `encode_b_intra_dry`
  Step 4 — and the pack coeff write is the `av1_write_intra_coeffs_mb` 64-chunk **L/U/V
  interleave** (encodetxb.c:431-472). This closes the **KB-1 encoder cross-check** (the >64-block
  txb order had never actually been exercised — the cited "256² cq63" evidence is an SB64 frame
  with no >64 blocks). `AV1E_SET_SUPERBLOCK_SIZE`(56)/`AOM_SUPERBLOCK_SIZE_128X128`(1) added to
  `cx_ctrl` (header-verified). **Partial-SB-at-128 (frames not a multiple of 128px) also
  byte-exact** — `sb128_partial_sb_e2e` (192² + the KB-6 196² conformance frame × cq{32,63}): the
  KB-6 partial-SB machinery (distortion visible-clips, `set_partition_cost_for_edge_blk`, the
  frame-edge entropy-stamp tail-zero) combines cleanly with the 128-SB geometry + the mu-64
  edge-clip. **Chroma formats — mono (4:0:0) + 4:4:4 coded-128-leaf also byte-exact**
  (`sb128_chroma_format_e2e`): 4:4:4 exercises the chroma mu-64 interleave at ss=0 (chroma chunks ==
  luma chunks), mono the luma-only interleave — proving the mu-64 machinery is ss-generic-correct.
  **One PINNED near-tie: `mono 256² cq63`** (port codes 1 fewer byte, the KB-2/KB-10/KB-12 "cheaper
  RD decision" signature) — NOT a mu-64 bug (4:2:0 + 4:4:4 128-leaves at cq63 AND mono cq55 all
  byte-match; only mono-cq63, with no chroma RD to break a qindex-252 tie, flips), same class as the
  pinned KB-10/KB-11 high-qindex near-ties; the gate asserts the divergence PRESENT
  (self-promoting), closing needs a sibling-C RD dump. **KB-12's member of this "class" turned
  out to be a dropped kernel transpose, not a tie (2026-08-02) — treat the signature as a lead,
  not a diagnosis.** Deferred: a coded 128-LEAF at a frame edge
  (the partial-SB cells split to ≤64, so the 128-leaf mu-64 edge-clip itself is still untested);
  non-default knob × sb128 combos; speed≥1 × sb128.
- External partition / `--partition-info-path` / `--sb-qp-sweep`: diagnostic, lowest
  priority. (M, defer)

### C9 — Transform controls — mostly DONE (byte-exact); dct-only pinned-open
- **DONE (this landing, → section A):** `--enable-tx64=0`, `--enable-rect-tx=0`,
  `--enable-flip-idtx=0`, `--use-intra-default-tx-only=1`, `--reduced-tx-type-set=1` — all
  BYTE-IDENTICAL vs real aomenc on the witnessed grid (`toggles_rd_close::toggles_c9_*`).
  Threading landed: `TxTypeSearchPolicy.{enable_flip_idtx, use_intra_dct_only}` →
  `TxMaskParams` (tx_search.rs; C reads oxcf directly in `get_tx_mask`, stage-independent);
  partition_pick's derived winner-mode stage policies copy the CLI toggles from `cfg.pol`
  (+ the MODE_EVAL `use_default_intra_tx_type` OR, rdopt_utils.h:579). The five layer
  differentials (`uniform_txfm_yrd_diff`, `intra_sby_mode_loop_diff`, `rd_pick_intra_sb_diff`,
  `txfm_uvrd_diff`, `intra_sbuv_mode_loop_diff`) now SWEEP `use_intra_dct_only` (oracle chain
  threads it into the REAL `get_tx_mask` facades) — all green.
- **`--use-intra-dct-only=1` — PINNED-OPEN** (section B row): luma byte-faithful; chroma
  UV-mode-loop winner divergence vs real aomenc, out of band at 64²cq32. Full localization
  trail in the section-B row + the pinned test's doc comment.
- **`--enable-tx-size-search=0` DONE (this landing, → section A):** knob route landed —
  `TxTypeSearchPolicy.enable_tx_size_search` (the port's oxcf.txfm_cfg carrier): the speed-0
  single-pass method pick goes USE_FULL_RD → USE_LARGESTALL (intra_rd.rs), the winner-mode sf
  derivation forces `tx_size_search_level = 3` post-speed (speed_features.c:2726 shape,
  partition_pick.rs), and the leaf `tx_mode_is_select` init ANDs the knob (select_tx_mode →
  TX_MODE_LARGEST; the existing KB-10 LARGESTALL⇒not-select coupling handles the pass level).
  C forbids combining with `--enable-tx64=0` (encodeframe.c:2461 assert) — not celled.
- Remaining: `--disable-trellis-quant` values 1/2 as explicit knob states (stage-aware
  policies exist from KB-8; default is 3) (S).
- **`--quant-b-adapt` (the `aom_quantize_b_adaptive` family) — KERNEL PORTED + FUNNEL WIRED
  (this landing):** `aom_quantize_b_adaptive_helper` / `aom_highbd_quantize_b_adaptive_helper`
  (aom-quant; lowbd+highbd, qm+no-qm, the prescan-widened dead-zone `EOB_FACTOR=325` +
  `SKIP_EOB_FACTOR_ADJUST=200` tail) are **bit-identical vs the exported C
  `aom_quantize_b_adaptive_helper_c` / `_highbd_` across 192k random cells** (`quantize_b_adaptive_diff`,
  log_scale 0/1/2 × bd 8/10/12 × qm/no-qm × large+sparse regimes, both eob==0 and eob>0
  exercised). `QuantParams::with_adaptive` routes `QuantKind::B` through the adaptive helper
  in the `xform_quant` funnel. **NOT DONE:** the frame-flag threading for an e2e byte gate —
  `quant_b_adapt` only affects the B quantizer, used with trellis OFF (`USE_B_QUANT_NO_TRELLIS`,
  so the gate needs `--quant-b-adapt=1 --disable-trellis-quant=1`); threading the flag through
  the TxfmYrdEnv/partition_pick/UV-env pipeline (the same deep carriers as
  `disable_trellis_quant`) is deferred as a mechanical follow-up on the proven kernel+funnel. (S–M)

### C10 — Intra mode toggles — DONE (byte-exact)
- **DONE (this landing, → section A):** all 8 toggles — `--enable-smooth-intra=0`,
  `--enable-paeth-intra=0`, `--enable-cfl-intra=0`, `--enable-directional-intra=0`,
  `--enable-diagonal-intra=0`, `--enable-angle-delta=0`, `--enable-filter-intra=0`,
  `--enable-intra-edge-filter=0` — BYTE-IDENTICAL vs real aomenc on the witnessed grid
  (`toggles_rd_close::toggles_c10_*`). Threading landed: `IntraToolCfg` on `PickFrameCfg`
  (partition_pick.rs; the 5 luma flags applied onto `IntraSbyGates` after the sf
  derivation — C keeps CLI + sf gates separate and the diffed visit chain reads both);
  chroma copies ride the existing `UvLoopPolicy` fields (the speed>=3 chroma rebuild
  spreads `..cfg.uv_lp.clone()`, so they survive at all speeds). The seq-level pair
  (filter-intra / intra-edge-filter) is knob-driven on the port side with the bootstrap
  seq bits ASSERTED equal (no bootstrap flow).

### C11 — Bitstream / global — mostly PRESENT
- PRESENT: bd 8/10/12, mono, 4:2:0/4:2:2/4:4:4, tiles (multi-tile e2e), lossless-mono,
  QM signaling, header/OBU writers (seq + frame, all components bit-exact).
  `--reduced-tx-type-set=1` e2e byte gate landed with C9 (this landing).
- **`--cdf-update-mode=0` encoder e2e DONE (this landing, → section A)** — and it caught a
  REAL pack bug (see the section-A row: symbol writers adapted CDFs unconditionally; the
  writer-side `allow_update_cdf` gate now mirrors C's aom_write_symbol).
- PARTIAL: cost-upd-freq knobs
  (`--coeff/mode/dv-cost-upd-freq` non-default arms; default arm proven byte-exact via
  the multi-SB e2e gates) (S–M); self-derived seq/frame header fields (drop the Gate-3
  bootstrap caveat: qindex mapping done #8; tile limits, CICP echo, level/tier remain)
  (S–M); `--full-still-picture-hdr` / annexb framing arms (S).
- **`--min-q` / `--max-q` — DONE (this landing):** the qindex clamp bounds. For a lone KEY
  still under AOM_Q, `base_qindex = clamp(quantizer_to_qindex(cq), quantizer_to_qindex(min_q),
  quantizer_to_qindex(max_q))` (rc_pick_q_and_bounds_q_mode, ratectrl.c:2158; best/worst_quality
  from rc_cfg, encoder.c:1003). `aom_encode::rc::base_qindex_from_cq_clamped` reproduces the real
  encoder's parsed `base_qindex` across a `(cq, min_q, max_q)` sweep — clamp-down, clamp-up, and
  inert cases (`min_max_q_diff`, via the new `shim_encode_av1_kf_minmaxq` / `ref_encode_av1_kf_minmaxq`
  that set `cfg.rc_min/max_quantizer`; the -1 sentinel leaves every existing caller inert). `--min-cr`
  clamp remains (S).

### C12 — Lossless tail — DONE (→ Section A) — mono + 4:2:0 both byte-exact
- **DONE (KB-5 closed, #32):** coded-lossless cq0 KEY is byte-exact for BOTH mono AND 4:2:0 bd8,
  hard-asserted in `encoder_gate_lossless_cq0_e2e_kb5_repro`. The former "4:2:0 cq0 chroma RD
  near-tie" was a search-SPACE gap (CfL was banned at coded-lossless in the SEARCH), not RD math:
  fixed by routing the leaf `cfl_allowed` through
  `aom_entropy::partition::is_cfl_allowed(bsize, lossless, ss_x, ss_y)` (C allows CfL at lossless
  when the partition size == the transform size). See CLAUDE.md KB-5.
- **DONE 2026-08-03 — the SPEED and BIT-DEPTH axes.** The above was proven only at
  `speed = 0` and bd8: the e2e harness refused qindex 0 at every speed (a single-pass
  header parse + `assert!(base_qindex > 0)`), and behind it the nonrd estimate arm's two
  TX_4X4 `block_yrd` arms were `unimplemented!()`. Both closed; 52 lossless cells byte-identical
  across cpu-used 0..9 and bd8/bd10/bd12 (`kb5_lossless_speed_axis`). `aom_fdct4x4` is
  ISA-conditional at hbd — LIBAOM_UPSTREAM_NOTES A6. See CLAUDE.md KB-5.
- **DONE 2026-09-03 — the CHROMA-FORMAT and CONTENT axes, plus the zenavif#45 root.**
  `self_contained_key_frame`'s new J arm sweeps cq 0 over {mono, 4:2:0, **4:2:2**,
  **4:4:4**} x bd {8, 10, 12} x 5 content classes x `--cpu-used` {0, 9} (+ bd8 at
  {3, 6}) and a 13-point size ladder 1x1..258x258 — **186 new cells, all
  byte-identical to real aomenc** (gate 186/186 -> **372/372**). A second test,
  `coded_lossless_reconstructs_the_source_exactly`, asserts the property C cannot
  arbitrate inside the `HBD_OPEN` band: **248/248 cells** decode — on the real C
  decoder AND on `aom-decode` — to the encoder's own input, exactly, on every plane.
  The landing's root fix: `key_frame::count_leaf` computed C's
  `mbmi->tx_size != max_txsize_rect_lookup[bsize]` (`partition_search.c:517`, `:555`)
  as `tx_size_to_depth(..) != 0`, and that paraphrase is only valid where the
  tx-size SYMBOL exists — i.e. NOT at coded-lossless, where `select_tx_mode`
  (`rdopt_utils.h:391-393`) forces ONLY_4X4 and every leaf is `TX_4X4` at depths up
  to 4. See CLAUDE.md KB-44.
- **Remaining (follow-up, S):** lossless x SB128 (SB128 is REFUSED by name in the
  standalone shell), and bd10/bd12 lossless at `--cpu-used` 1..6, which is the
  pre-existing `HBD_OPEN` band and pinned as such (`PIN_cq0_bd10_grad`,
  `PIN_cq0_bd12_tex`), NOT a lossless finding — measured 2026-09-03 to be the same
  bd x speed band at cq 32.

### C13 — Speed levels 6–9 — DONE (→ Section A) — Gate-2 (cpu 0–9) byte-complete
- **DONE (KB-10 / KB-11 / KB-12):** speeds 6, 7, 8, 9 all landed byte-identical on the synthetic
  canon grids (Section-A cpu-used rows). Speed 6 = new machinery (LPF_PICK_FROM_Q + partition prunes
  + predict_dc skip + 8×8 NN tx-depth prune + winner-mode restructure), 64/64; speed 7 =
  VAR_BASED_PARTITION fixed-tree + `av1_rd_use_partition`, 64/64; speeds 8/9 = the nonrd PICKMODE
  (`av1_nonrd_use_partition` single-pass walk + `av1_nonrd_pick_intra_mode`), speed 9 64/64 +
  speed 8 **64/64** (60/64 until 2026-08-02, when KB-12's `aom_hadamard_lp_8x8` transpose closed
  the last four). `top_intra_model_count_allowed=2` lands at speed≥6. See CLAUDE.md KB-10/11/12.
- **Remaining (pinned near-ties, self-promoting — NOT coverage gaps):** speed-6/7 noise-cq63
  (mi 8,0) TX_16X16-vs-TX_32X32 (KB-10/11); the nonrd bd10/12 + lossless + screen-palette arms
  (asserted dead on the 8-bit canon grid). Real content at cpu≥1 is a separate residual (KB-13,
  24/60 at cpu 1–4). **The 4 speed-8 `diag` cells left this list 2026-08-02 — and they were
  never a near-tie** (a dropped Hadamard transpose, KB-12), which is a standing caution about
  the two entries above: the "cheaper RD decision" signature is what a small unmodelled rate
  term looks like, not evidence of a tie.

### Priority order (proposed)
~~1. **C2 LR search**~~ DONE (section A, 2026-07-17) → ~~2. **C1 CDEF search**~~ DONE
(section A, 2026-07-17) → 3. **C3 screen content** (web stills) → 4. **C4
tune=IQ/SSIMULACRA2 tail** (image-quality tuning, small pieces) → 5. **C5 deltaq 3/6** →
6. C8/C9/C10 toggle threading (cheap wins, many S) → 7. C6 superres, C7 film grain →
8. C11/C12 tails → ~~C13 speeds 7–9~~ **DONE (KB-10/11/12, Gate-2 complete)**. (C2/C1
leftovers — LR speed-1..4 e2e arms, CDEF FAST levels e2e — are follow-ups within their families,
below the C3+ fronts.)
