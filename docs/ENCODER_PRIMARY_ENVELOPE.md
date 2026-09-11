> Moved VERBATIM out of `CLAUDE.md` on 2026-09-11 so the per-turn instruction file stays small (it had reached 744 KB / ~186k tokens). Nothing below was edited in the move; "above"/"below" refer to positions in the original file. Keep appending here, not in `CLAUDE.md`.

## Encoder single-frame primary envelope (VERIFIED against reference/libaom)

Primary config = ALLINTRA (usage=2), speed-0 KEY frame. libaom's own allintra tuning
(`av1/av1_cx_iface.c:3065`) sets these **defaults** — so matching them, NOT the base defaults,
is what "single-frame exact" means:

- **CDEF: OFF** by default in allintra ("CDEF has been found to blur images, so it's disabled
  in all-intra mode"). Only `--enable-cdef` turns it on.
- **Loop-restoration: ON** by default in allintra (speeds 0-4). CORRECTED 2026-07-18 (the prior
  "OFF by default" claim was WRONG — verified first-hand against reference/libaom):
  `default_extra_cfg.enable_restoration = 1` (`av1_cx_iface.c:286`, the `!CONFIG_REALTIME_ONLY`
  build), NOT cleared by the `:3065` allintra override (which only touches CDEF / screen-mode /
  qm_min/max), and kept for non-realtime at `:1273-74` (`usage != REALTIME` → stays 1). So a
  plain `aomenc --allintra` runs `av1_pick_filter_restoration` and emits the seq/frame
  restoration syntax (even when every unit resolves RESTORE_NONE → different header bits from
  `--enable-restoration=0`). At **speed >= 5** C disables both Wiener+SGR
  (`speed_features.c:519-520`) → `enable_restoration &= 0` (`:2754`) → restoration OFF (the seq
  bit is 0). PARITY C2 was correct all along. The port's byte-exact LR search is now wired into
  the DEFAULT path (`aom-bench::EncodeCell::port_encode` derives the LR stage from the frame's
  `enable_restoration` = C's `is_restoration_used`); default parity is gated by
  `lr_default_parity::port_default_matches_plain_aomenc_allintra` (port default byte-matches a
  no-flags `aomenc --allintra`).
- **QM: OFF** by default in allintra. CORRECTED 2026-07-15 (the prior "QM: ON" claim was WRONG —
  it conflated the qm_min/max override with `enable_qm`). The allintra override at
  `av1_cx_iface.c:3065` sets `qm_min=4`/`qm_max=10` but does NOT assign `enable_qm`, which stays
  at its base default `0` (`:290/447`); `using_qm = enable_qm` (`:1310`). qm_min/max are INERT
  unless QM is turned on by `--enable-qm` (`:2076`) or `tune=IQ`/`SSIMULACRA2` (`:1946`).
  Empirical proof: the passing `encoder_gate_e2e_*` gates byte-match the port with `qm=None` —
  impossible if the reference allintra encodes were QM-on.
- screen_detection_mode = ANTIALIASING_AWARE.

**What the encoder track has byte-matched (`encoder_gate_e2e_*`):** own-search partition / mode /
tx / coefficients + LF-level derivation, in a **CDEF-off + restoration-off + QM-off** reference
encode (`shim encode_av1_kf`, cdef/restoration/qm passed as explicit params). This envelope
matches the allintra defaults for CDEF and QM (both off) but codes restoration **off** — a
NON-default config for **speeds 0-4** (the true default is restoration ON; see the corrected note
above). Those `encoder_gate_e2e_*` gates stay valid as `--enable-restoration=0` config tests
(and at speed >= 5 restoration-off IS the true default). The DEFAULT (restoration-on) path is
now separately wired + gated: `lr_restoration_gate` (the LR search, 8/8 byte-exact) +
`lr_default_parity` (port default == plain no-flags `aomenc --allintra`, restoration on). The
frame HEADER is still bootstrapped from the real parse (qindex, tile info, cdf-update, ...) —
only LF-level (and, on the default path, the restoration decision) is port-derived.

**Remaining for single-frame-PRIMARY exactness (blocks "all single frame exactly"):**
- **KB-2 (#22) cq62 speed-0 — FIXED ✅ (74fb582)**: per-block `get_intra_edge_filter_type`
  recompute in `partition_pick.rs` (a SMOOTH neighbour was not raising the angled-prediction edge
  filter → model-RD over-pruned V_PRED adj=−1 → flipped SB(32,32) partition). cq62 byte-matches +
  asserted in `encoder_gate_e2e_rich_content_strong_lf`. See the KB-2 FIXED block above.
- **#25 two latent speed-1 bugs — DONE ✅** (verified 2026-07-15): both are fixed in source
  (parameterized, no longer hardcoded 0) — `part4_prune.rs` takes a `level_index` param
  (`min(speed,3)`, with the `>=3` alternate-branch guard) and `tx_search.rs` takes an
  `intra_tx_size_init_depth_rect` field — and the asserted per-feature-revert witness
  `encoder_gate_speed1_rect_and_4way_25` (in `encoder_gate_e2e_byte_match.rs`) re-diverges if either
  fix is reverted. (Earlier "need test cells to validate" note was stale.)
- **#10 cpu-used 0..9 speed-feature sweep** (Gate 2) — **DONE ✅ (all speeds 0-9, ZERO pinned
  cells since 2026-08-02)**: speeds 0-7 (KB-8/KB-9/KB-10/KB-11; 6/7 = 64/64 canon each), speed 9
  = 64/64 canon + noise, speed 8 = **64/64** canon + noise (was 60/64 — the 4 diag
  "estimate-arm near-tie" cells closed with KB-12's `aom_hadamard_lp_8x8` transpose) — the nonrd
  PICKMODE (`use_nonrd_pick_mode`, `av1_nonrd_use_partition` single-pass walk,
  `av1_nonrd_pick_intra_mode` + `hybrid_intra_mode_search`). See KB-12. The speed sweep above is
  SYNTHETIC content; **REAL content at speed>=1 is a SEPARATE residual (KB-13, task #39)** — the
  synthetic gates are 64/64 but decoded-conformance content diverges at 36/60 speed-1..4 cells (all
  interior BLOCK_16X16/8X8 partition RD near-ties, port over-picks AB/SPLIT), pinned self-promoting
  in `encoder_gate_real_content_speed1to4_e2e`. Remaining Gate-2 byte-exactness is the
  KB-10/KB-11 speed-6/7 noise-cq63 near-tie + the KB-13 real-content set (the 4 speed-8 diag
  cells CLOSED 2026-08-02 — and they were never a tie: see KB-12's transpose root, which is a
  standing warning against reading this shape as a tie in the two that remain). (#8 qindex-from-cq and #21 decoder q62/q63 also DONE + CI-green.)

**Confirmed NON-divergences (ruled out — do not re-chase):**
- **#27 `model_based_prune_tx_search_level`.** `av1_set_speed_features_qindex_dependent` sets it
  to 0 for `{<720p, base_qindex ≤ thresh}` while the port keeps 1, but the field is **inter-only**:
  the C consumer gate lives in `av1_pick_recursive_tx_size_type_yrd` behind `is_inter_block`, so it
  is inert on the all-intra KEY path and the port never reads it. `prune_tx_size_level` is inter-only
  the same way. Coordinator independently confirmed both. Empirical guard: the new asserted
  `encoder_gate_e2e_low_qindex_speed0` (cq8–30 → qindex 32–120, 12 cells) byte-matches end-to-end
  with the field left at 1 — the previously-untested aggressive-web low-q regime is now covered.

**NOT blocking single-frame-primary (non-default single-frame knobs — these ARE single-frame work
to be done before "the rest"=inter-frame, but lower priority than the primary default config):**
- **#23 QM-on encode — DONE ✅ (2026-07-16)**: `--enable-qm=1` allintra KEY byte-matches real
  aomenc — `encoder_gate_qm_on_e2e` (40 cells, bd8+bd10, qm ranges (5,9)+(4,10), mono+420) +
  anti-vacuous witness. QM selection runs inside the RD search (`resolve_qm` per tx in
  `xform_quant`), levels via `aom_get_qmlevel_allintra`. KEY subtlety (root-caused via
  sibling-libaom dump): C's trellis weights its DISTORTION by the forward matrix ONLY under
  `dist_metric == QM_PSNR` (tune=IQ) — with default PSNR the trellis runs `qmatrix = NULL`
  while dequant still folds `iqmatrix` (`optimize_txb_qm` now takes `Option` for the dist qm).
  tune=IQ / tune=SSIMULACRA2 (QM_PSNR dist, 444-chroma level formula, chroma deltaq,
  sharpness=7) remain out of envelope. See STATUS.md 2026-07-16.
- **#7 CDEF-strength RD search — DONE ✅ (2026-07-17), BIT-IDENTICAL**: full `av1_cdef_search`
  port (`aom-encode/src/pickcdef.rs`) + the two-pass encode→LF→search→pack architecture
  (`pack_tile_from_trees`, pack.rs) — 14/14 cells byte-match real aomenc `--enable-cdef=1`
  (real content 196²/64² cq5..63 with cdef_bits=2 per-unit literals; mono/444/420/bd10
  synthetic axes). Gate: `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close`
  (aom-bench, via the rd_close harness + full byte-identity asserts). CDEF stays off by
  default — the default envelope is untouched. FAST search levels 1..5 ported (table-level
  unit tests); only FULL (speed 0) is e2e-gated so far. See STATUS.md 2026-07-17.
- **Loop-restoration (Wiener/SGR) search — DONE ✅ (2026-07-18), BIT-IDENTICAL + DEFAULT-WIRED**:
  loop-restoration is **ON by default** in allintra (speeds 0-4; NOT a non-default knob — the
  prior "off by default" note here was WRONG, see the corrected primary-envelope note above). The
  byte-exact `av1_pick_filter_restoration` search (`crates/aom-dsp/src/restore/pick.rs`; PARITY C2) is now
  wired into the port's DEFAULT path (`aom-bench::EncodeCell::port_encode` derives the LR stage
  from the frame's `enable_restoration` = C's `is_restoration_used`). Gates: `lr_restoration_gate`
  (the search, 8/8 real content + 3/3 mono/444/bd12 format axis) + **`lr_default_parity`** (the
  port's default encode byte-matches a genuinely no-flags `aomenc --allintra` — the reference is
  the new `shim_encode_av1_kf_defaults`, tools at allintra defaults). Restoration-off remains a
  valid non-default config tested by the `encoder_gate_e2e_*` gates. Speeds 1-4 LR search arms +
  GOOD-mode arms are source-verified but PINNED (real-content base encode not yet byte-exact at
  speed >= 1, KB-13); speed >= 5 is structurally LR-off in C. See STATUS.md 2026-07-18.

