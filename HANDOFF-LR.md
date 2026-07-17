# HANDOFF-LR.md — loop-restoration encoder-search family (bulk-port agent, 2026-07-17)

Written at forced shutdown. Branch: `worktree-agent-a907d1a3b045ddf64`
(worktree of /root/aom-rs). Everything below is COMMITTED on this branch.

## 1. LANDED on origin/main (pushed, ancestor-verified)

- **e24cf09** — LR write-side syntax (`aom_entropy::lr`): forward
  `recenter_finite_nonneg`, `write/count_primitive_{quniform,subexpfin,refsubexpfin}`,
  `write_wiener_filter` / `write_sgrproj_filter` / `write_lr_unit`,
  `count_wiener_bits` / `count_sgrproj_bits`. Gate `lr_write_diff.rs`:
  400 sequences byte-identical to the REAL C writer (`shim_lr_units_roundtrip`),
  CDF lockstep, exhaustive count parity (21,760 cells) vs EXPORTED
  `aom_count_primitive_refsubexpfin` (new binding `ref_count_primitive_refsubexpfin`).

## 2. COMMITTED on this branch, VALIDATED, NOT PUSHED (push was queued when shutdown hit)

Stack `96d3464..41894f2` (5 commits, already REBASED onto origin @ 0730a68):

- **96d3464** — search numeric core (`aom-restore/src/pick.rs` part 1):
  `find_average(_highbd)`, `compute_stats` (lowbd, downsampled-stats mode, `dgd_origin`
  base for C pointer semantics at plane edges), `compute_stats_highbd` (divider 4/16),
  `linsolve_wiener` (b/278065963 overflow scaling), `update_sep_sym` (a/b dirs),
  `wiener_decompose_sep_sym` (4 iters), `compute_score`, `finalize_sym_filter`,
  `pixel_proj_error` (lowbd+highbd forms), `calc_proj_params`, `get_proj_subspace`
  (overflow guards), `encode_xq`. Oracle shim `pickrst_shim.c` +
  `ref_compute_stats(_highbd)/ref_pixel_proj_error/ref_calc_proj_params/
  ref_selfguided_restoration` bindings. Gate `pick_diff.rs` (6 tests) — all green.
  HONEST GAP: Wiener solve chain has NO C export (static) — transcribed; validated e2e.
- **dfd757e** — decision layer (`pick.rs` part 2): `LrSearchSf` (lpf_sf slice),
  `LrSearchInput/LrPlanePixels/LrSearchOutcome`, `PlaneCtx` staging (padded dgd +
  extend_frame + the encoder's TWO save_boundary_lines passes + trial dst; frame.rs
  internals opened pub(crate)), `try_restoration_unit`, `search_norestore/wiener/
  sgrproj/switchable`, `finer_search_wiener` (±{4,2,1}), `search_selfguided_restoration`
  (ep ladder incl. pruning), `restoration_search` (tile→SB→corners walk, CODING order),
  `pick_filter_restoration` (size descent + early breaks). Gate `pick_search.rs`.
  NOTE: fixed a +s-revert mirror-tap sign bug in finer_search_wiener during review —
  if a future tap-refinement mismatch appears, that area is proven correct now.
- **96534c4** — encoder wiring + gate: `pack_tile_lr` (aom-encode/pack.rs — SB-root
  RU writes BEFORE the partition symbol, per-tile `LrRefState`, adapts `kf` LR CDFs,
  gated on `pack_cfg.allow_update_cdf`; `pack_tile` delegates lr=None), aom-bench
  `EncodeCell::c_encode_lr` / `port_encode_lr` (LF APPLY via `loop_filter_frame` with
  KF deltas → search with `av1_fill_lr_rates` costs from FRAME-INIT LR CDFs (they never
  adapt pre-search in C) + frame RDMULT + `av1_dc_quant_qtx(qindex,0,bd)` → derived
  `p.restoration` header fields → repack when any plane restores),
  `parse_restoration_decision`, `lr_search_sf_allintra`. Gate `lr_restoration_gate.rs`.
- **3d115a6** — rebase reconciliation (deduped aom-loopfilter dep; CDEF landings).
- **41894f2** — PARITY C2 → **Section A** + STATUS entry + gate HARDENED to full
  byte-identity + decision-equality asserts.

**Measured (twice: pre-rebase detail run + post-rebase suite): the hardened gate is
GREEN — 8/8 cells BYTE-IDENTICAL to real `aomenc --enable-restoration=1`, 8/8 decisions
equal.** Cells: 64² cq{12,32,48}, 196² cq{20,48}, 352×288 cq{32,55}, b10 352×288 cq32.
Decision shapes covered: all-NONE, WIENER-luma, SGRPROJ-luma, WIENER-all-3-planes,
mixed SGR-luma+WIENER-chroma (b10), size-descent picking 128.

**NEXT STEP (was queued): push.** The landing suite (post-rebase tree) was mid-run at
shutdown with 0 failures and the LR + CDEF gates already green
(`/root/.claude/jobs/3651b35b/tmp/lr_land_suite.log`). Recipe:
```
cargo test --workspace                      # must be 0 failed
git push origin HEAD:main                   # rebase-additive if origin moved:
                                            #   git fetch && git rebase origin/main
git merge-base --is-ancestor 41894f2 origin/main && echo LANDED   # (hash moves if rebased)
```

## 3. COMMITTED-UNVALIDATED (this final wip commit — written at shutdown, compiles, NEVER RUN)

- **GOOD-mode sf routing** (aom-bench lib): `port_encode_lr` now routes
  allintra→`lr_search_sf_allintra`, GOOD→`lr_search_sf_good`. GOOD **speed 0 is
  verified-equal to defaults** (good framesize-independent arm sets no LR fields at
  speed 0; the single-size qindex rule requires speed>=3 for GOOD). GOOD speed>=1
  brackets are TRANSCRIBED FROM PARTIAL NOTES — `// HANDOFF:` comments carry the
  speed_features.c line anchors to verify (:1220, :1272, :1352, :1452, :648).
  `EncodeCell` gained `#[derive(Clone)]`.
- **Chunk-5 gate arms** (`lr_restoration_gate.rs::lr_restoration_speed_and_format_arms`):
  allintra speed 1–4 cells (quantizer-00; first e2e exercise of the ep-prune ladder,
  src-var prune, sgr-from-wiener prune, reduced 5-tap window, single-size rule),
  mono / 4:4:4 (nn-upsampled chroma) / bd12 (<<4) format cells, GOOD speed-0 cells.
  Asserts rd_close bands; byte/decision tallies printed. `// HANDOFF:` banner in the
  file gives the exact run + hardening recipe (if 12/12 EXACT → assert bit_identical
  like the main gate; else record honest fractions in PARITY C2).

## 4. Validation recipe for the unvalidated parts

```
# 1. chunk-5 arms (~15-20 min; speed>=1 cells are single-size, faster than speed-0):
cargo test -p aom-bench --test lr_restoration_gate -- --nocapture 2>&1 | tee /tmp/lr5.log
# 2. per-cell: decision_EQUAL + EXACT expected for mono/444/bd12/GOOD-s0.
#    Speed 1-4: if a cell DIFFERS, check printed decisions first:
#    - wrong unit size  => qindex rule bracket (lr_search_sf_allintra) or qindex read
#    - wrong ep         => SGRPROJ_EP_GRP tables / ladder flow (pick.rs search_selfguided)
#    - wrong taps       => reduced-window (WIENER_WIN_REDUCED) plumbing
#    - equal decisions, bytes differ => pack placement (unlikely; main grid proves it)
# 3. Harden per the HANDOFF banner, update PARITY C2 honest fractions, full suite once,
#    push everything.
```

## 5. Every C reference I hold (all v3.14.1, paths relative to reference/libaom)

- `av1/encoder/pickrst.c` — the whole search. Key facts: NUM_WIENER_ITERS=5;
  WIENER_TAP_SCALE_FACTOR=1<<16; DUAL_SGR_PENALTY_MULT=0.01 (ep<10, ×(1+0.01·lvl));
  WIENER_SGR_PENALTY_MULT=0.005 (switchable bias); start_step 4 (wiener) / 2 (sgr xqd);
  `count_wiener_bits` uses the NOMINAL win (7 luma) even under the reduced window;
  ep ladder tables `sgproj_ep_grp1_seed={0,3,6,9}`, `sgproj_ep_grp2_3[2][14]`;
  rusi reuse-across-sizes is safe because every stale read is guarded by sse==MAX.
- `av1/encoder/rd.h` — `RDCOST_DBL_WITH_NATIVE_BD_DIST(RM,R,D,BD) =
  R·RM/512 + (D>>(2(BD-8)))·128` (f64); bits are AV1_PROB_COST (<<9), call sites >>4.
- `av1/encoder/rd.c:802` — `RDMULT = av1_compute_rd_mult(base_qindex + y_dc_delta_q,…)`
  (y_dc_delta_q=0 in this envelope; if delta-q lands, thread it).
- `av1/encoder/encoder.c:2765` `cdef_restoration_frame` — ordering: save_boundaries(0)
  on DEBLOCKED → cdef → superres → save_boundaries(1) on CURRENT →
  `av1_pick_filter_restoration(cpi->source, cpi)` → apply with optimized_lr=0.
  `is_restoration_used = seq.enable_restoration && !all_lossless` (encoder.h:4430).
- `av1/encoder/bitstream.c:1625-1645` — LR RU writes at SB root BEFORE write_partition;
  corners fn self-gates bsize==sb_size; refs reset per tile (av1_reset_loop_restoration).
- `av1/encoder/speed_features.c` — allintra LR: s1 dual_sgr=1+ep_prune=1 (:420);
  s2 src_var=1+sgr_from_wiener=1 (:435); s3 sgr=screen?1:2, chroma-keep, reduce_window,
  src_var=2 (:463); s5 disable both (:519) AND seq-bit clear (:2754, !seq_params_locked
  ⇒ speed>=5 allintra streams NEVER carry LR — no gate possible). qindex-dependent
  (:3080-3108): s0 min=64,max=256 (descent 256→128→64 after sb-width clamp);
  `speed>=3 || (ALLINTRA && speed>=1)` ⇒ single size (128 iff qindex<=96 && <1440p
  else 256); s>=1 720p/1440p min-size floors. GOOD brackets: see HANDOFF comments.
- `aom_dsp/binary_codes_writer.c` + `recenter.h` — forward recenter else-branch is
  `recenter_nonneg(n-1-r, n-1-v)` with NO outer n-1- (the read side differs!).
- `aom_dsp/psnr.c` / `sum_squares.c` — sse_part = plain Σ(a-b)²;
  `aom_var_2d = (ss - s·s/(w·h))`, then caller /(w·h). u64, order exact.
- `pickrst.h` — `find_average` = u8/u16-truncating mean.
- restoration.h: RESTORATION_PROC_UNIT_SIZE=**64** (not 32!), UNITSIZE_MAX=256,
  UNIT_OFFSET=8, BORDER=3, WIENER_FILT_BITS=30, UNITPELS_MAX=(406·398).
- EXPORTED C oracles (nm-verified in the static lib): `av1_compute_stats_c`,
  `av1_compute_stats_highbd_c`, `av1_{lowbd,highbd}_pixel_proj_error_c`,
  `av1_calc_proj_params(_high_bd)_c`, `av1_selfguided_restoration_c`,
  `av1_loop_restoration_filter_unit`, `av1_pick_filter_restoration` (needs full
  AV1_COMP — not shimmed), `aom_write/count_primitive_refsubexpfin`, `av1_fill_lr_rates`,
  `av1_cost_tokens_from_cdf`.

## 6. Known follow-ups beyond chunk 5

- `pack_tile_from_trees` unification: reuse the CDEF two-pass pack (origin 9850da6)
  for the LR repack instead of the second full search — add an `lr` param via a
  `pack_tile_from_trees_lr` delegator (keeps their call sites untouched); wire
  `port_encode_lr` to pass pass-1 `trees`. Byte-neutral (deterministic search),
  roughly halves gate runtime.
- SIMD `compute_stats` (the search hot spot) on the Gate-3 track — the wiener apply
  kernel is already SIMD (origin 198235c) and feeds try_restoration_unit.
- Multi-tile LR search (`restoration_search` already takes tile SB bounds vec;
  the harness only wires single-tile).
- The DEFAULT flip (enable_restoration=1 as the port's allintra default) is the
  coordinator/foundation call — the knob path is fully proven; flipping = threading
  the seq bit + running the LR stage by default in whatever product encode API lands.
- Speed-6+ allintra: LR structurally off in C (no work).

## 7. Files owned/touched by this family

- `crates/aom-entropy/src/lr.rs` (read+WRITE side), `crates/aom-entropy/tests/lr_write_diff.rs`
- `crates/aom-restore/src/pick.rs` (the whole search), `src/frame.rs` (pub(crate) opens),
  `src/lib.rs`, `tests/pick_diff.rs`, `tests/pick_search.rs`, `Cargo.toml` (aom-txb dev-dep)
- `crates/aom-sys-ref/shim/pickrst_shim.c`, `build.rs` (+pickrst_shim), `src/lib.rs` (bindings)
- `crates/aom-encode/src/pack.rs` (pack_tile_lr + LrPackParams)
- `crates/aom-bench/src/lib.rs` (c_encode_lr/port_encode_lr/parse_restoration_decision/
  lr_search_sf_allintra/lr_search_sf_good/Clone), `Cargo.toml`, `tests/lr_restoration_gate.rs`
- `PARITY.md` (A row + C2), `STATUS.md` (milestone section)
