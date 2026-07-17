# HANDOFF — tune=IQ / tune=SSIMULACRA2 family (PARITY.md C4), job 3651b35b

Written under a shutdown directive. State captured so the next agent can finish without
re-deriving. Branch `worktree-agent-a07347614b295cb09`, based on `origin/main` (`4e70236`).
Everything is **OFF by default** — `TuneKnobs::default()` = PSNR envelope, byte-identical to the
proven pre-tune path. The whole point of the family: nothing changes unless a tune knob is set.

## Provenance / adoption

The predecessor agent (`worktree-agent-a6595a97d6d5ead8b`, its branch tip `3703f1d`, plus a large
uncommitted working tree) did ~90% of this. Its base was `27ca089` (pre-palette). Current
`origin/main` (`4e70236`) has the palette/intrabc bulk-port landed on top, touching the same
`aom-encode` files. I rebased the predecessor's full delta (`27ca089`..working-tree) onto
`4e70236` with `git apply --3way`:

- Combined patch (committed 3703f1d + uncommitted, minus data/docs): was at
  `/root/.claude/jobs/3651b35b/tmp/combined_tune.patch` (EPHEMERAL — regenerate from the
  predecessor worktree with `git -C <pred> diff 27ca089 -- crates` if needed; the predecessor
  worktree is READ-ONLY, do not modify it).
- **CANONICAL SALVAGE — predecessor worktree `agent-a6595a97d6d5ead8b` commit `5442f88`**
  ("wip: coordinator salvage — spend-limit shutdown checkpoint") = the predecessor's FULL
  uncommitted WIP committed on top of `3703f1d` (incl. PARITY.md/STATUS.md + the 1147-line
  `encoder_gate_tune_iq_e2e.rs`). My adoption == `git diff 27ca089 5442f88` minus docs. Prefer
  `5442f88` as the source of truth over the ephemeral patch. The doc (PARITY.md/STATUS.md) changes
  I did NOT apply — mine them from `git -C <pred> show 5442f88 -- PARITY.md STATUS.md` after
  re-verifying results.
- 31/37 files applied cleanly; **6 conflicted** (all trivial additive overlaps — see below).
- `crates/aom-encode/src/allintra_vis.rs` is UNTRACKED in the predecessor; copied in verbatim.
- Committed as `wip:` (`64edeb4`) WITH conflict markers first (preservation), then resolved.

## The 6 conflicts and their resolutions (all "keep both", additive)

Palette work (ours/main) and tune work (theirs/predecessor) each added a field/param to the same
struct/call. Resolution is mechanical — **keep both sides**, except pack.rs which threads a value:

1. `encode_intra.rs` `EncodeIntraYEnv` — keep `palette: Option<..>` AND add `tune: crate::TuneKnobs`.
2. `encode_sb.rs` `EncodeIntraYEnv` init — keep `palette: winner.palette_y…` AND `tune: env.tune`.
3. `pack.rs` mbmi init (~322) — keep main's `cdef_strength,` (shorthand local) BUT take theirs'
   `current_qindex: sb_current_qindex,` (the variance-boost per-SB qindex).
4. `pack.rs` `pack_tile` head (~1004) — keep BOTH blocks: main's VAR_BASED_PARTITION (speed≥7)
   `use_var_based_partition`/`vbp_stamps`/`vbp_frame` AND theirs' `search_base_qindex` init.
5. `rd_pick.rs` `EncodeIntraYEnv` init — keep `palette: y.palette_y…` AND `tune: re.tune`.
6. `tests/encode_intra_plane_diff.rs` — keep `palette: None,` AND `tune: Default::default(),`.
7. `tests/kb4_txb2_probe.rs` — take theirs' `trellis_rdmult_intra_y(rdmult, 0, bd, false)` (the
   new `iq_tuning` arg; main's call lacked it).

## Pieces state — N/6 + composite

Spec: PARITY.md §C4 + CLAUDE.md "#23 QM-on DONE" note. The tune bundle (`handle_tuning`,
av1_cx_iface.c:1938-1978): `enable_qm=1, qm_min=2, qm_max=10, sharpness=7, dist_metric=QM_PSNR,
enable_cdef=ADAPTIVE, enable_chroma_deltaq=1, deltaq_mode=6 (VARIANCE_BOOST)`; IQ adds
`enable_adaptive_sharpness=1`.

1. **QM-level formulas — CODE DONE + gate.** `aom_get_qmlevel_luma_ssimulacra2` +
   `aom_get_qmlevel_444_chroma` (quant_common.h:111/:150) + `QM_FIRST/LAST_IQ_SSIMULACRA2`=2/10 in
   `aom-quant/src/quant_common.rs`; C oracle `qm_shim.c`; gate `qm_level_diff` (4 formulas × 6
   ranges × qindex 0..=255 vs REAL C static inlines). Was commit `3703f1d`. Predecessor claimed
   byte-exact. VERIFY: `cargo test -p aom-quant --test qm_level_diff`.

2. **QM_PSNR dist metric — CODE PRESENT, needs compile+test.** `TuneKnobs { use_qm_dist_metric,
   iq_tuning }` (aom-encode/src/lib.rs) threaded through `SbEncodeEnv`/`EncodeIntraYEnv`/
   `UvEncodeParams`/`ReencodeParams`/`TxTypeSearchPolicy`. New: `dist_block_tx_domain_qm` +
   `dist_qmatrix` (lib.rs), `QuantParams::with_qm_dist_metric`/`QmCtx::use_qm_dist_metric` (the
   trellis forward-matrix arm, txb_rdopt.c:346-351), `trellis_rdmult_intra{,_y}` gain `iq_tuning`
   (rshift 7 vs 5, txb_rdopt.c:378-386), `TxTypeSearchPolicy::with_tune_knobs` (forces tx-domain
   dist on, rdopt_utils.h:516-522). Wired in `search_tx_type_intra`'s tx-domain arm + 64pt/high-
   energy hybrid + `prune_txk_type_intra` (est-rd, whose B-quant now folds QM). C ref:
   `shim_encode_av1_kf_tune`/`ref_encode_av1_kf_tune` (dec_shim.c, AOME_SET_TUNING first then
   per-knob overrides; dist_metric via `aom_codec_set_option("dist-metric",…)`). Gate:
   `encoder_gate_qm_psnr_dist_e2e` (encoder_gate_tune_iq_e2e.rs — 27/27 claimed: bd8 mono/420/444 ×
   64²/128²/192² × cq{12,32,50}) + anti-vacuous `tune_shim_smoke` (aom-sys-ref). VERIFY:
   `cargo test -p aom-encode --test encoder_gate_tune_iq_e2e` + `cargo test -p aom-sys-ref --test tune_shim_smoke`.

3. **--sharpness e2e — CODE mostly present, CELLS + witness PENDING.** Trellis takes sharpness 0..7
   (pre-existing). Quantizer rounding bias `sharpness_adjustment` (av1_quantize.c:607) — check
   `aom-quant/src/build_quantizer.rs` (modified in the patch). `frame_lf_sharpness` (lf_search.rs)
   = picklpf.c:220-247 sharpness gate. HANDOFF: e2e byte cells (`--sharpness=N`) + anti-vacuous
   witness NOT written. Add cells to encoder_gate_tune_iq_e2e.rs mirroring the QM_PSNR gate shape.

4. **chroma deltaq (--enable-chroma-deltaq) — ABSENT (HANDOFF).** Not started. C ref:
   `av1_get_deltaq_offset` chroma path + av1_cx_iface enable_chroma_deltaq gate. `// HANDOFF:` stub
   to add.

5. **adaptive sharpness (picklpf.c:232) — helper landed, CELLS PENDING.** `frame_lf_sharpness`'s
   `enable_adaptive_sharpness` arm (qindex cap: ≤112→7, ≤160→1, else 0). HANDOFF: byte cells.

6. **deltaq_mode=6 VARIANCE_BOOST — CODE PRESENT, gate cells + anti-vacuous witness PENDING (THIS
   WAS THE NEXT STEP).** `allintra_vis.rs` (full port): `variance_boost_block_variance`
   (aq_variance.c:184), `av1_get_sbq_variance_boost` (allintra_vis.c:1072), `variance_boost_delta_q_res`
   (encodeframe.c:1920), `av1_adjust_q_from_delta_q_res` (rd.c:494), `av1_convert_qindex_to_q`/
   `_q_to_qindex` (ratectrl.c:199/:211), `setup_delta_q_variance_boost` (encodeframe.c:341). Threading:
   `SbEncodeEnv::deltaq: Option<DeltaQFrameCtx>`, `PackCfg::delta_q_present/delta_q_res`, `pack_tile`
   per-SB qindex derivation + `set_q_index` row re-select + SB rdmult from adjusted qindex,
   `pack_leaf`/`pack_sb` gain `sb_current_qindex`. **HANDOFF — remaining:** (a) gate cells running
   the port with `deltaq=Some(..)` vs real `aomenc --deltaq-mode=6 [--deltaq-strength=N]`; (b) the
   anti-vacuous witness (deltaq-off == base byte-identical; deltaq-on CHANGES the stream). The C shim
   `ref_encode_av1_kf_tune` needs a `--deltaq-mode=6` path (AOME_SET_ or set_option) — verify it
   plumbs deltaq_mode/strength. FP note (allintra_vis.rs:19): `av1_get_sbq_variance_boost` uses f64
   log2/round in C's exact order; log2 → platform libm (same glibc both builds) so byte gates hold
   locally.

**composite tune=IQ/SSIMULACRA2 arm — PENDING.** The config arm that installs the whole bundle at
once (`--tune=iq` / `--tune=ssimulacra2`). cdef ADAPTIVE at cq>32 depends on C1's CDEF search
(separate track); cq≤32 = zero-strength early-out. Needs: one composite gate that sets ALL knobs
(qm on + qm-level formulas + QM_PSNR + sharpness=7 + chroma deltaq + deltaq=6 + adaptive sharpness
for IQ) and byte-matches `aomenc --tune=iq` / `--tune=ssimulacra2`.

## Validation recipe (frugal: cheap targeted while developing, full suite ONCE at end)

1. Compile: `cargo build -p aom-encode -p aom-quant -p aom-sys-ref` (fix conflict fallout first).
2. Piece gates (targeted): `cargo test -p aom-quant --test qm_level_diff`;
   `cargo test -p aom-encode --test encoder_gate_tune_iq_e2e`;
   `cargo test -p aom-sys-ref --test tune_shim_smoke`.
3. Envelope guard (must stay green — everything OFF by default): the existing `encoder_gate_e2e_*`
   / KB gates must be byte-unchanged.
4. RD-closeness: `aom_bench::rd_close` — port(tune=iq) vs real `aomenc --tune=iq` per cell (+
   `--tune=ssimulacra2`); bit-identical cells recorded EXACT. (crates/aom-bench/src/rd_close.rs.)
5. FULL suite ONCE at the end: `cargo test --workspace --no-fail-fast`.

## Landing (when green)

Consolidated 1-2 commits, pathspec-scoped (`crates/` + PARITY.md/STATUS.md; never `.claude`/
`conformance/data`), author `aom-rs <lilith@imazen.io>`, trailer
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`, rebase-additive, `git push origin
HEAD:main`, verify `git merge-base --is-ancestor HEAD origin/main`. Then PARITY.md §C4 → move the
landed pieces to Section A/B rows with measured deltas; update STATUS.md.

## Risks / watch-outs

- **pack.rs in-loop variance-boost insertions applied via 3-way against a heavily-rewritten
  `pack_tile` (palette + VBP speed-7 landed on main).** The 2 marked conflicts are resolved, but
  VERIFY the cleanly-applied hunks (setup_delta_q_variance_boost call site, `sb_base_rdmult`,
  `dq_rows` row re-select, the `sb_pick_cfg`/`sb_env` construction) landed at the RIGHT place in the
  current loop body — grep `setup_delta_q_variance_boost`, `sb_current_qindex`, `search_base_qindex`
  and read the surrounding loop. This is the highest-risk merge area. **PARTIALLY VERIFIED
  2026-07-17:** the deltaq `sb_base_rdmult` (pack.rs:1101) IS correctly threaded into main's
  VBP-gated `sb_rdmult` (pack.rs:1132 — `if allintra && !use_var_based_partition {
  fold_intra_sb_rdmult(sb_base_rdmult, modifier) } else { sb_base_rdmult }`); the 3-way combined
  both features coherently, and the `setup_delta_q_variance_boost`/`dq_rows` derivation (:1075-1100)
  reads right. STILL to confirm on first compile: the `sb_pick_cfg`/`sb_env` `rows_y/u/v` deltaq
  override (`..*env` spread) and the `pack_leaf`/`pack_sb` `sb_current_qindex` param plumbing.
- The predecessor's PARITY.md/STATUS.md diffs were NOT applied (would conflict + carry unverified
  claims). Their prose is worth mining from the predecessor worktree
  (`git -C <pred> diff 27ca089 -- PARITY.md STATUS.md`) once results are re-verified.
- CLAUDE.md provenance is messy: the predecessor worktree CLAUDE.md has KB-9/KB-10 + "#23 QM-on
  DONE" + "KB-5 420 FIXED" that origin/main's CLAUDE.md lacks (parallel-agent doc divergence). Trust
  code + tests, not the doc, and reconcile CLAUDE.md at the end.
