# s9 pack_leaf retained-output replay — 2026-09-15

Profile cell: `port 192x192 cq27 s9 reps=5` (callgrind `Ir`). Same protocol as
`encoder_s9_alloc_class_2026-09-15.md` / `encoder_s9_lp_kernels_2026-09-15.md`.

Shipping witness `port 1024x1024 cq27 s3`: **byte-identical 40,237 B**.
192x192 s9 witness: **byte-identical 1,894 B**. Full encode `all` suite
**654/654** incl. every e2e byte gate; `encode_sb_diff` 43/43 nonrd+walk
tests green.

## What landed

The structural item the lp-kernel record attributed and deferred: `pack_leaf`
re-ran `encode_b_intra_dry` on every committed leaf (3,312 calls vs the pick's
1,656 at this cell — one per pack pass) where C's `write_modes_b` reads the
`cb_coef_buff` coefficients the OUTPUT_ENABLED encode already produced. The
re-encode was ~44 % of the cell.

- `LeafWinner::replay: Option<LeafEncodeOut>` — the output-enabled walk's
  own result, retained on the winner. `Some` only for an ordinary intra leaf
  (`!is_inter && !use_intrabc`) encoded with `output_enabled = true`.
- `encode_sb_dry`'s 19 leaf sites funnel through `finish_leaf_out`:
  `retain` → `w.replay = Some(out)`; `!retain` (DRY_RUN context-propagation
  walks) → `leaves.push(out)` unchanged, so the `encode_sb_diff` gate keeps
  its full output list.
- `nonrd_leaf_pick_and_encode`'s two sites (full-RD + estimate arms) retain
  the same way.
- `stamp_leaf_ctx` — `encode_b_intra_dry`'s steps 4+6 (the tokenize
  `(txb_skip_ctx, dc_sign_ctx)` derive, edge-clipped `cul` above/left stamps,
  `set_txfm_ctxs`) extracted verbatim into a shared `pub(crate)` fn; the
  encode calls it on the fresh `LeafEncodeOut`, the replay arm calls it on
  the retained one against the PACK pass's `TileCtxState`. The per-txb ctx
  pairs come out recomputed-identical; the `cul == txb_entropy_ctx`
  debug-asserts stay live on retained data.
- `pack_leaf` replays: `winner.replay.take()` → geometry/kind assert (a
  retained payload must name this leaf's mi/bsize — a mismatch means the
  tree was rebuilt after the encode walk) → `stamp_leaf_ctx` → the existing
  coeff writers consume `retained.y/u/v.txbs` unchanged. `None` falls back
  to the re-encode (inter/intrabc early-return arms stamp reset-to-zero
  contexts — a different pattern — and carry no replay). The payload is
  stored back so the phase-2 `pack_tile_from_trees_lr` re-walk over the same
  trees replays too; a `None` leaf whose re-encode ran is likewise stored,
  making its second pass replayable.

## Why it is byte-identical

- The pick walk that fills `replay` runs `encode_b_intra_dry` with
  `output_enabled = true` already (the SB-root winner walk /
  `nonrd_leaf_pick_and_encode`), so the retained qcoeff/eob/tx_type/dqcoeff
  ARE the OUTPUT_ENABLED payload — `FINAL_PASS_TRELLIS_OPT` included.
- `encode_b_intra_dry`'s only persistent-`TileCtxState` writes on the
  ordinary-intra arm live inside the extracted block (verified: every other
  `state.` access in the body is a read; the early-return arms are
  inter/intrabc only). `stamp_leaf_ctx` reproduces them on pack's ctx.
- The coeff writers consume only retained fields (`qcoeff`, `eob`,
  `tx_type`, `txb_entropy_ctx`, `txb_skip_ctx`, `dc_sign_ctx`); mode,
  partition, palette, tx-size and CDF syntax still run off `winner`.
- eob-0 map resets land in the transient frame map under OUTPUT_ENABLED, so
  `winner.tx_type_map` is untouched — the retained `tx_type` equals what a
  re-encode would re-derive. (eob-0 txbs' cul is 0 under any tx type, so even
  a hypothetical reset keeps the `cul == txb_entropy_ctx` assert true.)

## Measured (192x192 cq27 s9 reps=5, same protocol)

| arm | total Ir | vs base |
|---|---:|---:|
| base (post lp kernels `4c8bb3a`) | 229.85M | — |
| + pack replay | **166.93M** | **-27.4 %** |

Cumulative session delta at this cell: **271.1M -> 166.93M Ir (-38.4 %)**.

Call-shape check: `encode_b_intra_dry` **4,968 -> 1,524 calls** (the pack
re-walks are gone; the pick's winner walk remains); `pack_leaf` self Ir
0.77M. `stamp_leaf_ctx` shows 3,312 calls — the two pack passes' replay
stamps (the pick-side calls ride inside `encode_b_intra_dry`).

## Perf gate (real photo cells, wall, interleaved)

`encode_perf_vs_libaom::standalone_encode_time_against_libaom`, 12 cells,
12/12 byte-identical:

| cell | port ms | C ms | ratio | ratio before this batch |
|---|---:|---:|---:|---:|
| 128² cq27 s0 | 174.97 | 95.76 | 1.83x | ~2.4x |
| 128² cq45 s0 | 123.71 | 61.14 | 2.02x | ~2.6x |
| 128² cq27 s6 | 6.84 | 3.61 | 1.89x | ~1.8x |
| 128² cq45 s6 | 6.26 | 3.06 | 2.04x | ~2.0x |
| 128² cq27 s9 | 0.90 | 0.53 | 1.68x | ~2.5x |
| 128² cq45 s9 | 0.67 | 0.44 | 1.55x | ~2.3x |
| 192² cq27 s0 | 354.85 | 187.06 | 1.90x | ~1.9x |
| 192² cq45 s0 | 236.10 | 115.49 | 2.04x | ~2.0x |
| 192² cq27 s6 | 12.98 | 6.59 | 1.97x | ~2.0x |
| 192² cq45 s6 | 11.71 | 5.53 | **2.12x** | ~2.1x |
| 192² cq27 s9 | 1.65 | 0.81 | 2.03x | **3.42x** |
| 192² cq45 s9 | 1.23 | 0.63 | 1.96x | ~3.3x |

Worst ratio over byte-identical cells: **2.12x** (192² cq45 s6), down from
3.42x at the s9 cells. ("Before" column = the 2026-09-15 pre-batch run on the
same protocol; small-cell ratios carry ±~0.1x machine noise.)

## Residual attribution (named)

- `memset`/`memcpy` (14.6M + 11.2M self Ir at this cell): CDF table copies,
  plane clones, arena fills — mostly semantic — plus the
  `forbid(unsafe_code)` safe-Rust init floor.
- `OdEcEnc::normalize`/`encode_cdf_q15` (~5.9M): the entropy coder itself.
- `stamp_leaf_ctx` 1.48M/4,968 calls — the replay's remaining per-leaf cost.
- `optimize_txb_core` still trellises chroma where C's `optimize_b` is
  luma-only at s9 (count now 1x the pick — the pack-side calls went away
  with the re-encode).
