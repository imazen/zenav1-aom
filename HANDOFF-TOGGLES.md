# HANDOFF — TOGGLE SWEEP (C8/C9/C10/C11) — remaining items

**LANDED 2026-07-17.** The C8-C11 toggle sweep is on `origin/main`: 23 knob arms
BYTE-IDENTICAL vs real aomenc (hard `bit_identical` pins in
`crates/aom-bench/tests/toggles_rd_close.rs`, 25 tests) + 1 pinned-open
(`--use-intra-dct-only`). All landed toggles are rows in PARITY.md §A; the
mechanism notes live in STATUS.md's TOGGLE-SWEEP sections. This file now tracks
ONLY the genuinely-remaining items.

The two `--disable-trellis-quant=1/2` arms landed EXACT this pickup; `=2`
(FINAL_PASS) needed a real encoder fix (5a644c6): `encode_b_intra_dry` had
hardcoded `dry_run_output_enabled: false`, so the OUTPUT_ENABLED pack pass did
not apply FINAL_PASS trellis. Byte-inert for every other trellis mode (the flag
is dead outside FINAL_PASS).

## Genuinely-remaining toggle work

| Toggle (ctrl) | State | What's needed |
|---|---|---|
| `--use-intra-dct-only=1` (119) | **PINNED-OPEN** (1 cell: 64²cq32) | Deep near-tie; see the sibling-C localization below. Cell stays pinned (fails on movement either way). |
| `--sb-size=128` encode | **DONE — corrected 2026-08-03** (this row read "**UNSTARTED (M)**; encoder walk/harness are SB-64-only") | Landed. `crates/aom-bench/tests/sb128_e2e.rs` byte-matches real `aomenc --sb-size=128` with sb128-vs-sb64 anti-vacuity witnesses per gate, including a genuinely coded 128 leaf, partial-SB frames, and 4:4:4/4:2:2 — see PARITY.md section A. One residual: a mono 256² cq63 near-tie is PINNED there. |
| `--coeff/mode-cost-upd-freq` (126/127) | **C-SIDE ONLY** (6e32167) | C ctrls emitted from the knobs; the port-side gate is UNWIRED. Design (in the ToggleKnobs doc comment): split pack.rs's per-SB `derive_real_costs` rebuild per table set — SB=every SB (current), SBROW=only at `c==0`, TILE/OFF=never (single-tile ⇒ identical). Add a `cost_upd: CostUpdCfg` (Default) field in one sweep across PackCfg literal sites. |
| `--quant-b-adapt` | **INERT standalone; the kernel now EXISTS** (`crates/aom-dsp/tests/quantize_b_adaptive_diff.rs` gates it — corrected 2026-08-03; this said "needs kernel for the combo"). What remains is the policy plumb for the `--disable-trellis-quant=1/2` combo, not the kernel. | VERIFIED against C: at the default speed-0 allintra envelope trellis is ON, so the encode uses `AV1_XFORM_QUANT_FP` (encodemb.c:406-412, `USE_B_QUANT_NO_TRELLIS ? QUANT_B : QUANT_FP` gated on `!use_trellis`) — `quant_b_adapt` only feeds `AV1_XFORM_QUANT_B`, so it is INERT on the primary config (a standalone cell is vacuous — the witness refuses it). It is LIVE only combined with `--disable-trellis-quant=1/2` (QUANT_B path), and porting it then needs the `aom_quantize_b_adaptive` kernel family [PORTED — `crates/aom-dsp/src/quant/`, gated by `crates/aom-dsp/tests/quantize_b_adaptive_diff.rs`; note `aom-quant` is a dead crate path since the 2026-07 consolidation] (av1_quantize.c:311 `use_quant_b_adapt` arm) + policy plumb. |

## Verified-INERT on this envelope (documented, NOT remaining work)

| Toggle | Why inert / vacuous (verified vs C / harness) |
|---|---|
| `--disable-trellis-quant=0` (FULL) | vs the default (3, NO_ESTIMATE_YRD) differs only in `estimate_yrd_for_sb`, which is inter-only → the C stream never changes → the anti-vacuity witness (correctly) refuses the cell. |
| `--min-q/--max-q/--min-cr` | The toggle harness BOOTSTRAPS qindex from the C parse (`aom-bench/src/lib.rs` `qindex = p.quant.base_qindex`), so a clamp changes C's qindex AND the port follows it — the cell would byte-match but exercises NO port-specific min/max-q logic (the port doesn't self-derive qindex). Vacuous port-side until the #8 self-derived-qindex path drives the harness. |
| `--full-still-picture-hdr` / annexb | OBU-framing only. The port emits the frame-OBU PAYLOAD (the byte-compared unit); annexb changes container framing / temporal delimiters / size fields, not the frame-OBU payload — nothing for the port to reproduce here (seq spliced from C). |
| `--dv-cost-upd-freq` (142) | DV costs are intrabc-only → INERT on the KEY allintra envelope (intrabc off) → not cellable (witness refuses). The ctrl id is present for completeness. |

## The `--use-intra-dct-only` pinned-open — sibling-C dump DONE (2026-07-17)

Cell: `av1-1-b8-01-size-64x64`, full 64×64, cq32 (qindex 128), allintra speed-0,
`AV1E_SET_INTRA_DCT_ONLY=1`. 64²cq63 EXACT, 128²cq12 CLOSE; only 64²cq32 diverges
(+2.23% size, +3.588 zensim OUT of band). Y recon IDENTICAL; the divergence is
the mi(0,0) 32×32 chroma UV mode.

Sibling-C dump method (used + REVERTED this session — do not re-derive from
scratch): ar-swap a throwaway `libaom.a` (compile ONE instrumented TU with the
cmake flags from `build/CMakeFiles/aom_av1_encoder.dir/flags.make`, `ar r` it
into a copy of `reference/libaom/build/libaom.a`) and temporarily repoint
`aom-sys-ref/build.rs`'s `build_dir` at it, then run the pinned test so
`c_encode_ctrls` drives the instrumented C. Byte-inert `fprintf`s in
`intra_mode_search.c` (`av1_rd_pick_intra_sbuv_mode` loop) + `tx_search.c`
(`av1_txfm_uvrd` / the txb `block_sse`).

**Measured (cq32 mi(0,0) 32×32):**
- C evaluates ONLY DC (this_rd 2157931) and **D45/aduv2 (this_rd 1985157 — WINS)**.
  Every other UV mode is rejected: V/H/directionals at `rd_pick_intra_angle_sbuv`
  anglefail (its inner `av1_txfm_rd_in_plane` returns INT_MAX for plane U);
  SMOOTH/PAETH at txfmfail. (CFL: no eval, cfl branch drop.)
- The PORT instead ACCEPTS V (uv_mode=1, aduv0, DCT-forced tx_type=0, eob=1,
  **dist=0**, rate 20508 → this_rd 1872917) and V WINS.
- **Decisive:** C's V prediction `block_sse`=1048576 == the port's V sse=1048576
  ⇒ the PREDICTION MATCHES. This is NOT a prediction bug (the earlier "port CfL /
  angle-gate" suspects are ruled out).
- **Root:** the port's `txfm_rd_in_plane_uv_p` computes V's DCT dist=0 and ACCEPTS
  V, where C's `av1_txfm_rd_in_plane` REJECTS the identical V (same prediction,
  same DCT tx). This is a tx-search RD-eval / early-out mis-model shared by the
  port AND the `txfm_uvrd_diff` oracle — which is exactly why the five layer
  differentials are all green yet disagree with the full encoder.

**Next step:** dump C's per-txb V DCT dist + quantized coeffs inside
`av1_txfm_rd_in_plane` / `search_txk_type` (the INT_MAX path fires BEFORE
`av1_txfm_uvrd`'s merge, so a dump at the merge misses it — instrument the txb
level), and the port's `search_tx_type_intra` V winner, to find why the SAME DCT
residual (same prediction) yields dist=0 + accept in the port but INT_MAX-rd +
reject in C. The fix must be applied to BOTH the port and the differential oracle
(shared mis-model), then the pinned test graduates to a `bit_identical` assert.

## Validation recipe (how everything was proven)

1. Cell = `run_toggle_cell` (`toggles_rd_close.rs`): C encode via
   `EncodeCell::c_encode_ctrls(knobs.c_ctrls())` (real `aom_codec_av1_cx` + ctrl
   pairs), port via `port_encode_with(&c_tu, &knobs)`, compare via
   `rd_close::compare_cell`. Grid = 64²cq32 + 64²cq63 + 128²cq12 real content.
2. **Anti-vacuity witness is mandatory**: `run_grid_and_gate` panics unless the
   knob CHANGED the C stream on ≥1 cell. Never trust an EXACT verdict without it.
3. EXACT cells get `expect_exact=true` (hard `bit_identical` pin). Divergent
   knobs get a pinned-open test (fails on movement either way) + a PARITY §B row.
4. Ctrl ids: `aom_sys_ref::cx_ctrl` — a new id needs a `shim_cx_ctrl_id_by_probe`
   arm + PROBE_TABLE entry (`cx_ctrl_ids_match_reference_headers` cross-checks).
5. C defaults verified in `av1_cx_iface.c` `default_extra_cfg`: enable_* = 1,
   min/max part 4/128, dct/default-tx/reduced = 0, tx-size-search = 1,
   cdf-update = 1, trellis-quant = 3, cost-upd = SB. The allintra override block
   touches NONE of them.
6. Seq/frame-header knobs (filter-intra, edge-filter, reduced-tx-set, tx_mode,
   disable_cdf_update): port side is KNOB-driven with the bootstrap header bit
   ASSERTED equal. The tx_mode assert stays ONE-directional (C's post-hoc
   SELECT→LARGEST demotion on zero-split frames).

## Gotchas (do not re-lose)

- `write_coeffs_txb_full` keeps its explicit `allow_update_cdf` param — redundant
  with the `OdEcEnc` flag but consistent (both from the same header bit).
- C FORBIDS `--enable-tx64=0` + `--enable-tx-size-search=0` together
  (encodeframe.c:2461 assert) — never grid that combo.
- aom-bench's `max_partition_size` is C-derived `min(sf_default, dim, sb)`=12 at
  SB64 — outcome-identical (consumers OR with `bsize == sb_size`).
- Sibling-C encoder instrumentation must NOT touch `reference/libaom` in place
  (worktree isolation blocks it, and it is the shared differential oracle). Use
  the ar-swap-into-a-throwaway + build.rs-repoint pattern above; revert both
  (build.rs edit + `cargo clean -p aom-sys-ref`) before landing.
