# Winner-mode two-pass mode/tx evaluation — porting design (task #10, the speed-4 wall)

**Scope.** The `--cpu-used` sweep worklist (`docs/cpu_used_allintra_sweep_plan.md`)
flags the **winner-mode subsystem** as the "MAJOR structural chunk / first wall": at
`speed >= 4` on the all-intra KEY path, libaom switches luma intra-mode + tx search from
a single RD pass to a **two-pass** scheme (a cheap `MODE_EVAL` first pass that collects
the top-N "winner" candidates, then a full-RD `WINNER_MODE_EVAL` second pass over only
those winners). The port (`crates/aom-encode`) implements only the single pass and reads
the `DEFAULT_EVAL` table column, which is a **verified no-op equivalent** through speed 3
and **breaks at speed 4**. This doc turns the C machinery into a concrete porting design.

**Reference:** libaom v3.14.1 (git 03087864), `reference/libaom/av1/encoder/`. Every line
number below was read directly from source on 2026-07-15. Claims are tagged
**[CONFIRMED]** (read in source) or **[INFERRED]** (logical consequence, not a single
cite).

---

## 0. TL;DR

- **Activation:** all four winner-mode flags flip together at **speed 4**
  (`speed_features.c:502-505`) for allintra: `enable_winner_mode_for_coeff_opt=1`,
  `enable_winner_mode_for_use_tx_domain_dist=1`, `multi_winner_mode_type=MULTI_WINNER_MODE_DEFAULT`
  (3 winners), `enable_winner_mode_for_tx_size_srch=1`. They are a **package** — you cannot
  byte-match speed 4 by porting one sub-feature.
- **The two-pass is entirely inside the luma function** `av1_rd_pick_intra_sby_mode`
  (`intra_mode_search.c:1468-1741`). Chroma runs at `DEFAULT_EVAL` (no winner two-pass on
  the KEY path). **[CONFIRMED]**
- **Net effect at speed 4:** the first pass RD-ranks all 61 luma candidates with a *cheap*
  tx search (largest tx only, looser coeff-opt/skip thresholds, DCT-only tx-type); the top
  3 by that cheap RD are then re-evaluated with the *full* tx search (recursive tx-size RD,
  full coeff trellis). The final winner can differ from the single-pass winner because the
  ranking that selects the top-3 uses the cheap metric. **[INFERRED]**
- **The port's job:** add (a) a `MODE_EVAL`-column policy for pass 1 + a `WINNER_MODE_EVAL`-column
  policy for pass 2, (b) the `USE_LARGESTALL` tx-search arm (new), (c) an insertion-sorted
  winner list, (d) the second-pass re-eval loop, (e) the `is_winner_mode_processing_enabled`
  gate — all gated so speed 0-3 output stays **byte-identical**.

---

## 1. What activates at which speed (allintra KEY)

### 1.1 The winner-mode speed-feature timeline

All in `set_allintra_speed_features_framesize_independent` (`speed_features.c:345-616`).
Fields live in `sf->winner_mode_sf` (`WINNER_MODE_SPEED_FEATURES`, `speed_features.h`),
`sf->rd_sf`, and `sf->tx_sf.tx_type_search`. Base init: `init_winner_mode_sf`
(`speed_features.c:2506-2515`) zeroes everything (`tx_size_search_level=0`,
all `enable_winner_mode_for_*=0`, `multi_winner_mode_type=0`, `dc_blk_pred_level=0`). **[CONFIRMED]**

| speed | field | value | line |
|---|---|---|---|
| **4** | `winner_mode_sf.enable_winner_mode_for_coeff_opt` | `1` | `speed_features.c:502` |
| **4** | `winner_mode_sf.enable_winner_mode_for_use_tx_domain_dist` | `1` | `:503` |
| **4** | `winner_mode_sf.multi_winner_mode_type` | `MULTI_WINNER_MODE_DEFAULT` (=2 → 3 winners) | `:504` |
| **4** | `winner_mode_sf.enable_winner_mode_for_tx_size_srch` | `1` | `:505` |
| 5 | `winner_mode_sf.multi_winner_mode_type` | `MULTI_WINNER_MODE_FAST` (=1 → 2 winners) | `:524` |
| 6 | `winner_mode_sf.multi_winner_mode_type` | `MULTI_WINNER_MODE_OFF` (=0 → 1 winner; per-tool flags STAY on) | `:561` |
| 6 | `winner_mode_sf.prune_winner_mode_eval_level` | `1` | `:562` |
| 6 | `winner_mode_sf.dc_blk_pred_level` | `1` | `:563` |

Supporting scalars that only *mean something* once the two-pass exists (they select which
table row the eval columns read — see §1.3):

| speed | field | value | line |
|---|---|---|---|
| 4 | `tx_sf.tx_type_search.winner_mode_tx_type_pruning` | `2` | `:488` |
| 4 | `tx_sf.tx_type_search.fast_intra_tx_type_search` | `2` | `:489` |
| 4 | `tx_sf.tx_type_search.prune_2d_txfm_mode` | `TX_TYPE_PRUNE_3` | `:490` |
| 4 | `tx_sf.tx_type_search.prune_tx_type_est_rd` | `1` | `:491` |
| 4 | `rd_sf.perform_coeff_opt` | `5` | `:493` |
| 4 | `rd_sf.tx_domain_dist_thres_level` | `3` | `:494` |
| 6 | `tx_sf.tx_type_search.winner_mode_tx_type_pruning` | `3` | `:551` |
| 6 | `tx_sf.tx_type_search.prune_tx_type_est_rd` | `0` | `:552` |
| 6 | `rd_sf.perform_coeff_opt` | `6` | `:555` |
| 6 | `rd_sf.tx_domain_dist_level` | `3` | `:556` |

**Interacting field that already differs at speed 3** (independent of winner-mode, but it
drives the same `MODE_EVAL`-column plumbing): `tx_sf.tx_type_search.use_skip_flag_prediction=2`
at speed 3 (`:459`). It feeds `predict_skip_levels` (§1.3) — see §5.4.

### 1.2 `tx_size_search_level` stays 0 for allintra — the tx-size two-pass uses row 0

`winner_mode_params->tx_size_search_methods` is filled from
`tx_size_search_methods[cpi->sf.winner_mode_sf.tx_size_search_level]`
(`speed_features.c:2822-2823`). For allintra:

- base init `tx_size_search_level = 0` (`:2510`); **the allintra setter never assigns it**
  (verified absent across `:345-616`). **[CONFIRMED — by absence]**
- The only override is `if (!oxcf->txfm_cfg.enable_tx_size_search && !use_nonrd) level=3`
  (`:2726-2728`). Tx-size search is ON by default → **level stays 0** for the primary
  config. **[CONFIRMED]** (`:2047`'s `level=1` is inside `set_rt_speed_features_framesize_independent`
  = REALTIME, not allintra.)

So the tx-size eval columns at speed 4 come from `tx_size_search_methods[0]`
(`speed_features.c:106-111`):

```
tx_size_search_methods[0] = { USE_FULL_RD,   USE_LARGESTALL, USE_FULL_RD }
                              // DEFAULT_EVAL   MODE_EVAL       WINNER_MODE_EVAL
```

⇒ **first pass (MODE_EVAL) = `USE_LARGESTALL`** (evaluate only the largest tx size, no
recursive depth RD), **winner re-eval (WINNER_MODE_EVAL) = `USE_FULL_RD`** (the recursive
depth sweep the port already implements). At speed ≤3 (`enable_winner_mode_for_tx_size_srch=0`)
both columns collapse to `USE_FULL_RD` (see §1.4). **[CONFIRMED]**

### 1.3 The eval-column tables and which speed-4 values matter for intra KEY

The tables (`speed_features.c:40-138`) are `[level][MODE_EVAL_TYPES]` where the second index
is `{DEFAULT_EVAL=0, MODE_EVAL=1, WINNER_MODE_EVAL=2}` (enum `rd.h:92-102`). At speed 4
allintra the selected rows are:

| table (line) | selected by | row @ speed 4 | `{DEFAULT, MODE_EVAL, WINNER}` |
|---|---|---|---|
| `tx_size_search_methods` (`:106`) | `tx_size_search_level=0` | 0 | `{FULL_RD, LARGESTALL, FULL_RD}` |
| `coeff_opt_thresholds` (`:88`) | `perform_coeff_opt=5` | 5 | `{{864,97}, {142,16}, {UINT_MAX,UINT_MAX}}` |
| `tx_domain_dist_thresholds` (`:54`) | `tx_domain_dist_thres_level=3` | 3 | `{0, 0, 0}` |
| `tx_domain_dist_types` (`:71`) | `tx_domain_dist_level` (unchanged from speed 1 = `1`) | 1 | `{1, 2, 0}` |
| `predict_skip_levels` (`:120`) | `use_skip_flag_prediction=2` | 2 | `{1, 2, 1}` |
| `predict_dc_levels` (`:136`) | `dc_blk_pred_level=0` (until speed 6) | 0 | `{0, 0, 0}` |

Reading the eval columns at speed 4 tells you exactly how the two passes differ:

- **tx-size:** pass1 `LARGESTALL` (cheap) vs pass2 `FULL_RD` (full). **[CONFIRMED]**
- **coeff-opt (trellis) gate:** pass1 `{dist 142, satd 16}` (aggressive skip of trellis) vs
  pass2 `{UINT_MAX, UINT_MAX}` (always trellis). DEFAULT was `{864,97}`. **[CONFIRMED]**
- **tx-domain distortion type:** pass1 `use_transform_domain_distortion=2`, pass2 `=0`
  (`tx_domain_dist_types[1] = {1,2,0}`). Threshold `0` on all columns
  (`tx_domain_dist_thresholds[3]`). **[CONFIRMED]**
- **skip-txfm prediction:** pass1 level `2`, pass2 level `1` (`predict_skip_levels[2]={1,2,1}`).
  **[CONFIRMED]**
- **DC-block prediction:** `0` everywhere until speed 6 → no effect at speed 4-5. **[CONFIRMED]**

Additionally the `MODE_EVAL` case in `set_mode_eval_params` sets, from the tx_sf scalars
(not from the eval-column tables), pass-1-only cheapenings **[CONFIRMED]** (`rdopt_utils.h:578-609`):
- `use_default_intra_tx_type = (fast_intra_tx_type_search==2)` → **DCT_DCT-only tx-type in pass 1**;
- `prune_2d_txfm_mode = prune_mode[winner_mode_tx_type_pruning-1][is_winner_mode=0]`
  = `prune_mode[1][0] = TX_TYPE_PRUNE_4` (`set_tx_type_prune`, `rdopt_utils.h:497-511`).
- `WINNER_MODE_EVAL` resets `use_default_intra_tx_type=0` (full tx-type set) and uses
  `prune_mode[1][1]=TX_TYPE_PRUNE_0` (`rdopt_utils.h:610-637`).

### 1.4 Why speed 0-3 is a no-op (and where the equivalence first cracks)

The port is correct at speed 0-3 **not** because C skips the eval machinery — C calls
`set_mode_eval_params(MODE_EVAL)` at the top of the loop at *every* speed
(`intra_mode_search.c:1515`) — but because every per-column selector **falls back to the
`DEFAULT_EVAL` value when its `enable_winner_mode_for_*` flag is off** (all off at speed ≤3):

- `set_tx_size_search_method` (`rdopt_utils.h:478-495`): initialises to
  `tx_size_search_methods[DEFAULT_EVAL]`, overrides with MODE_EVAL/WINNER only
  `if (enable_winner_mode_for_tx_size_srch)`. **[CONFIRMED]**
- `set_tx_domain_dist_params` (`:513-543`): returns the `DEFAULT_EVAL` pair
  `if (!enable_winner_mode_for_tx_domain_dist)`. **[CONFIRMED]**
- `get_rd_opt_coeff_thresh` (`rd.h:313-340`): returns `DEFAULT_EVAL` pair
  `if (!enable_winner_mode_for_coeff_opt)`. **[CONFIRMED]**
- `set_tx_type_prune` (`:497-511`): returns after setting the default `prune_2d_txfm_mode`
  `if (!winner_mode_tx_type_pruning)`. **[CONFIRMED]**
- **The second pass is gated off:** `is_winner_mode_processing_enabled`
  (`rdopt_utils.h:444-476`) returns 1 for intra **only if** `fast_intra_tx_type_search` (`:462`)
  **or** `enable_winner_mode_for_coeff_opt` (`:469`) **or** `enable_winner_mode_for_tx_size_srch`
  (`:473`) — all 0 at speed ≤3 ⇒ returns 0 ⇒ the `WINNER_MODE_EVAL` re-eval is **skipped**
  (`intra_mode_search.c:1730`). And `multi_winner_mode_type==OFF` makes
  `store_winner_mode_stats` return immediately (`rdopt_utils.h:688`), so the multi-winner
  branch is dead too. **[CONFIRMED]**

**Two fields DO read the raw `MODE_EVAL`/`WINNER_MODE_EVAL` column unconditionally**
(not gated by an `enable_` flag): `skip_txfm_level = winner_mode_params->skip_txfm_level[eval]`
(`rdopt_utils.h:586-589`) and `predict_dc_level` (`:588-589`). They are still no-ops at
speed 0-2 because their tables have identical columns there
(`predict_skip_levels[0]={0,0,0}`, `[1]={1,1,1}`; `predict_dc_levels[0]={0,0,0}`). **The
equivalence first cracks at speed 3**: `use_skip_flag_prediction=2` selects
`predict_skip_levels[2]={1,2,1}`, so the single MODE_EVAL pass at speed 3 wants
`skip_txfm_level=2` where DEFAULT is 1 — a speed-3 delta the port must handle *before*
the winner two-pass lands (it is a separate `[KEY][HIGH]` worklist row, but it exercises
the same plumbing). **[CONFIRMED]** See §5.4.

---

## 2. The C two-pass flow (precise call-flow)

**Entry (KEY):** `av1_rd_pick_intra_mode_sb` (`rdopt.c:3636`) →
`av1_rd_pick_intra_sby_mode` (luma, `intra_mode_search.c:1468`) then
`set_mode_eval_params(DEFAULT_EVAL)` (`rdopt.c:3659`) → `av1_rd_pick_intra_sbuv_mode`
(chroma, `rdopt.c:3670`, runs at DEFAULT_EVAL). **The winner two-pass is contained in the
luma function.** **[CONFIRMED]** (The other `WINNER_MODE_EVAL` sites — `rdopt.c:3883`,
`:4456` — are in the inter-mode path, past `rd_pick_skip_mode` at `:3705`; not on the KEY
intra path. **[INFERRED from function boundaries]**)

### 2.1 `av1_rd_pick_intra_sby_mode` (`intra_mode_search.c:1468-1741`)

**Setup (`:1499-1543`):**
- `mbmi->angle_delta[Y]=0`; HOG directional prune (`intra_pruning_with_hog`, `:1501-1510`).
- **`set_mode_eval_params(cpi, x, MODE_EVAL)` (`:1515`)** — installs the pass-1 params.
- `max_winner_mode_count = winner_mode_count_allowed[multi_winner_mode_type]` (`:1518-1519`;
  table `{1,2,3}` at `rdopt_utils.h:236-239`). `zero_winner_mode_stats(...)` + `winner_mode_count=0`
  (`:1520-1521`).
- `top_intra_model_rd[TOP_INTRA_MODEL_COUNT]` and `intra_modes_rd_cost[][]` init to `INT64_MAX`.

**Pass 1 — MODE_EVAL loop over all luma modes (`:1545-1661`):** for each of `LUMA_MODE_COUNT`
`(mode, angle_delta)` visits:
1. `set_y_mode_and_delta_angle` (`:1547`) + the static gate chain (smooth/paeth/directional
   enables, `intra_y_mode_mask`, odd-angle prune) — `continue` on rejection.
2. `this_model_rd = intra_model_rd(...)` (`:1602-1603`) — the Hadamard-SATD model estimate at
   `min(TX_32X32, max_txsize)`.
3. `prune_intra_y_mode(this_model_rd, &best_model_rd, top_intra_model_rd, top_intra_model_count_allowed, ...)`
   (`:1608-1611`) — top-N model prune; `continue` if pruned. **Runs identically regardless of
   eval column** (upstream of the tx search). **[CONFIRMED]**
4. **`av1_pick_uniform_tx_size_type_yrd(cpi, x, &this_rd_stats, bsize, best_rd)` (`:1617`)** —
   the tx search, **using the MODE_EVAL params** (`USE_LARGESTALL` + looser thresholds at
   speed 4).
5. `this_rate = stats.rate + intra_mode_info_cost_y(...)`, `this_rd = RDCOST(...)`, then the
   ALLINTRA `intra_rd_variance_factor` multiply (`:1631-1639`).
6. `intra_modes_rd_cost[mode][angle+MAX_ANGLE_DELTA+1] = this_rd` (`:1641`).
7. **`store_winner_mode_stats(..., this_rd, multi_winner_mode_type, txfm_search_done=1)`
   (`:1646-1648`)** — insert into the winner list (no-op if OFF).
8. `if (this_rd < best_rd)` update `best_mbmi/best_rd/beat_best_rd/*rate/...` and snapshot
   `ctx->tx_type_map` (`:1649-1660`). Strict `<` ⇒ ties keep the earlier mode.

**Post-loop (`:1663-1684`):** palette search (`try_palette`), filter-intra search
(`beat_best_rd && av1_filter_intra_allowed_bsize`), and `if (!beat_best_rd) return INT64_MAX`.

**Pass 2 — winner re-evaluation (`:1689-1737`):**
- **Multi-winner (`multi_winner_mode_type != OFF`, `:1689-1725`):** for `mode_idx` in
  `0..winner_mode_count` (winners are stored ascending-RD): `*mbmi = winner_mode_stats[mode_idx].mbmi`;
  if `is_winner_mode_processing_enabled(cpi, x, mbmi, 0)` (`:1698`): restore palette color map;
  **`set_mode_eval_params(cpi, x, WINNER_MODE_EVAL)` (`:1707`)**; `intra_block_yrd(...)`
  (`:1713`) — full-RD re-eval, updates `best_mbmi/best_rd` if strictly better; track
  `best_mode_idx`. Finally copy the winning palette color map (`:1719-1725`).
- **Single-winner (`OFF` but a per-tool flag on, i.e. speed 6, `:1726-1737`):** if
  `is_winner_mode_processing_enabled`: `set_mode_eval_params(WINNER_MODE_EVAL)`; `*mbmi = best_mbmi`;
  one `intra_block_yrd`.
- `*mbmi = best_mbmi`; `av1_copy_array(xd->tx_type_map, ctx->tx_type_map, ...)`; `return best_rd`.

### 2.2 `store_winner_mode_stats` (`rdopt_utils.h:679-718`) — the winner list

Insertion-sorted, ascending by `rd`, capped at `max_winner_mode_count`:
- return if `multi_winner_mode_type==OFF` (`:688`) or `this_rd==INT64_MAX` (`:690`).
- find first slot with `winner_mode_stats[mode_idx].rd > this_rd` (**strict `>`**, `:701-702`);
  if none and list full (`mode_idx==max_count`) → drop (`:704-706`); else `memmove` to open a
  slot (`:707-712`) and write `{mbmi, rd=this_rd, mode_index}` (`:714-717`).
- **Tie-break:** because the compare is `>` (not `>=`), an incoming mode with `rd` equal to an
  existing entry sorts **after** it — first-seen wins the slot. **[CONFIRMED]** Must be
  replicated exactly (see §5.1).

### 2.3 `intra_block_yrd` (`intra_mode_search.c:1188-1228`) — the winner re-eval kernel

`ref_best_rd = use_rd_based_breakout_for_intra_tx_search ? *best_rd : INT64_MAX`
(`:1200-1202`; the flag is `true` at speed ≥3, `set_allintra...:460`). Calls
`av1_pick_uniform_tx_size_type_yrd(cpi, x, &rd_stats, bsize, ref_best_rd)` (`:1203`) — now
under WINNER_MODE_EVAL params (`USE_FULL_RD`, full trellis, full tx-type set) — recomputes
`this_rd`; `if (this_rd < *best_rd)` updates best + snapshots `ctx->tx_type_map`, returns 1
(`:1217-1226`). Because pass-2 params differ from pass-1, the re-eval RD is generally *lower*
than the stored winner RD, and the post-re-eval ordering can flip which winner is best. **[INFERRED]**

---

## 3. The port's current structure and the gap

### 3.1 Active call chain (the encoder's live files)

`partition_pick.rs::leaf_pick_sb_modes` (`:445`, the `pick_sb_modes` leaf) builds
`IntraSbySearchCfg` (`:581`) with a single `pol: cfg.pol` (`:603`) → `rd_pick.rs::rd_pick_intra_mode_sb`
(`:239`, the `av1_rd_pick_intra_mode_sb` equivalent) → `intra_rd.rs::rd_pick_intra_sby_mode_y`
(`:888`, the luma search) → `tx_search.rs::pick_uniform_tx_size_type_yrd_intra` (`:1410`).
Chroma via `intra_uv_rd.rs`. Speed features + policy in `speed_features.rs`. **[CONFIRMED]**
These five files (`partition_pick.rs`, `rd_pick.rs`, `intra_rd.rs`, `tx_search.rs`,
`speed_features.rs`) are the active RD-path files; the winner-mode work lives in them.

### 3.2 What exists

- **Tables already ported verbatim, all three columns present** (`speed_features.rs:49-138`):
  `TX_DOMAIN_DIST_THRESHOLDS`, `TX_DOMAIN_DIST_TYPES`, `COEFF_OPT_THRESHOLDS`. **Only the
  `DEFAULT_EVAL` (index 0) column is read** — `SpeedFeatures::tx_type_search_policy`
  (`speed_features.rs:277`) hardcodes `[...][DEFAULT_EVAL]` (`:278,284,286`). `const DEFAULT_EVAL=0`
  (`:45`). `tx_size_search_methods`, `predict_skip_levels`, `predict_dc_levels` are **not**
  ported. **[CONFIRMED]**
- **The model-RD prune is faithful and eval-column-independent:** `prune_intra_y_mode`
  (`intra_rd.rs:518`), `get_model_rd_index_for_pruning` (`:484`) match C. No change needed.
- **The single-pass loop** `rd_pick_intra_sby_mode_y` (`intra_rd.rs:888-1069`): one pass with
  `cfg.pol`, `pick_uniform_tx_size_type_yrd_intra` (`:956`), strict-`<` best tracking
  (`:1027`), a `// store_winner_mode_stats: hard no-op at speed 0` marker (`:1025`), no second
  pass. Outcome type `IntraSbyBest`/`IntraSbyOutcome` (`:836-854`). **[CONFIRMED]**
- **The tx search** `pick_uniform_tx_size_type_yrd_intra` (`tx_search.rs:1410`) →
  `choose_tx_size_type_from_rd_intra` (`:1341`, the `USE_FULL_RD` depth sweep). The doc string
  explicitly marks `USE_LARGESTALL` / winner-mode arms out of scope (`:1407`). `TxfmYrdEnv`
  (`:919`) carries **no** eval-type/speed field — the policy is passed separately as
  `pol: &TxTypeSearchPolicy` (`:413-444`). **[CONFIRMED]**

### 3.3 The gap (what must be added)

1. **Eval-column policy resolution.** `TxTypeSearchPolicy` currently = the DEFAULT_EVAL slice.
   Need a way to build the **MODE_EVAL** and **WINNER_MODE_EVAL** policies (read column 1 / 2
   of the three tables + `use_default_intra_tx_type` from `fast_intra_tx_type_search` +
   `prune_2d_txfm_mode` from `set_tx_type_prune`). The fall-back-to-DEFAULT logic of
   `set_tx_size_search_method` / `set_tx_domain_dist_params` / `get_rd_opt_coeff_thresh` /
   `set_tx_type_prune` (§1.4) must be reproduced so that with all flags off the three policies
   are byte-identical (the no-op invariant).
2. **The `USE_LARGESTALL` tx-search arm** in `pick_uniform_tx_size_type_yrd_intra` — evaluate
   only the largest tx (`av1_pick_uniform_tx_size_type_yrd`'s LARGESTALL branch: set
   `mbmi->tx_size` to the block's largest, one `uniform_txfm_yrd` call, no depth recursion).
   This is genuinely new code; the port has only the FULL_RD sweep. Also `USE_FAST_RD` exists
   in the tables but is unreachable for allintra (row 0 has no FAST_RD), so it can be deferred.
3. **The winner list** — an insertion-sorted `Vec<WinnerModeStat>` capped at
   `winner_mode_count_allowed[multi_winner_mode_type]`, matching `store_winner_mode_stats`
   (§2.2) exactly, including the strict-`>` tie-break.
4. **The second pass** — after the pass-1 loop, iterate the winner list (or the single best),
   gated by an `is_winner_mode_processing_enabled` port, calling the tx search again with the
   WINNER_MODE_EVAL policy (`USE_FULL_RD`) and updating best on strict improvement (the
   `intra_block_yrd` semantics, §2.3), then snapshotting the winning `tx_type_map`.
5. **`SpeedFeatures` fields** — add `enable_winner_mode_for_{coeff_opt,use_tx_domain_dist,
   tx_size_srch}`, `multi_winner_mode_type`, `winner_mode_tx_type_pruning`,
   `fast_intra_tx_type_search`, `use_skip_flag_prediction`, `prune_winner_mode_eval_level`,
   `dc_blk_pred_level`, `tx_size_search_level`, `use_rd_based_breakout_for_intra_tx_search`
   with their speed gates (§1.1). All default such that speed 0-3 = current behaviour.

---

## 4. Porting plan (smallest-demoable-chunk first; no-op-when-off invariant)

**Guiding invariant (must hold after every chunk):** with all `enable_winner_mode_for_*=0`,
`multi_winner_mode_type=OFF`, `winner_mode_tx_type_pruning=0`, `fast_intra_tx_type_search=0`,
`use_skip_flag_prediction<2` (speed 0-2 baseline), the new code path must produce
**byte-identical** output to today. Guard with the existing speed-0/1 e2e gates
(`encoder_gate_e2e_*`, `encoder_gate_speed1_textured_allintra`) — they must stay green after
each chunk. **Because all four flags flip together at speed 4, no single chunk byte-matches a
speed-4 cell alone**; the demoable unit for chunks 1-4 is the **differential harness** (per-block
RD dumps vs the sibling C instrument, the KB-2/KB-3 method), not a standalone speed cell. The
speed-4 byte-match is the acceptance gate for chunk 5.

**Chunk 0 — speed-feature scaffolding (no behavior change).** Add the `winner_mode_sf` fields
to `SpeedFeatures` (`speed_features.rs`) with their speed gates from §1.1, plus
`tx_size_search_level` (always 0 for allintra) and `use_rd_based_breakout_for_intra_tx_search`.
Port `predict_skip_levels`/`predict_dc_levels`/`tx_size_search_methods` tables. Add unit tests
asserting the field values per speed (mirrors the existing `speed_features.rs` tests at
`:308-393`). *Files:* `speed_features.rs`. *Gate:* existing SF tests + no e2e change.

**Chunk 1 — eval-column policy builders (no behavior change while flags off).** Generalize
`tx_type_search_policy` into three builders — `policy_default_eval`, `policy_mode_eval`,
`policy_winner_mode_eval` — each reproducing the exact fall-back logic of
`set_tx_size_search_method`/`set_tx_domain_dist_params`/`get_rd_opt_coeff_thresh`/`set_tx_type_prune`
(§1.4). Add a `tx_size_search_method: {FullRd, LargestAll}` field to the policy (default
`FullRd`). **Assert the three policies are equal when all flags are off** (the invariant, as a
unit test). *Files:* `speed_features.rs`, `tx_search.rs` (policy struct). *Gate:* invariant
unit test + e2e unchanged.

**Chunk 2 — the `USE_LARGESTALL` tx-search arm.** Implement the LARGESTALL branch in
`pick_uniform_tx_size_type_yrd_intra` (`tx_search.rs:1410`): when
`pol.tx_size_search_method == LargestAll`, set the block's largest allowed tx size and run one
`uniform_txfm_yrd_intra` (no depth sweep). Validate the LARGESTALL RD against the C instrument
on a handful of blocks (differential). No production path selects it yet (speed ≤3 always
FULL_RD), so e2e stays green. *Files:* `tx_search.rs`. *Gate:* differential per-block +
e2e unchanged.

**Chunk 3 — winner list + first-pass wiring (still gated off).** Add `WinnerModeStat` +
`store_winner_mode_stats` (insertion sort, §2.2) to `intra_rd.rs`. Wire the pass-1 loop to (a)
use the MODE_EVAL policy when winner-mode is active, and (b) collect winners. Keep the
production caller passing the DEFAULT policy + `multi_winner_mode_type=OFF` so the collection is
a no-op (`store` returns immediately) and pass 1 == today. *Files:* `intra_rd.rs`. *Gate:*
`store` no-op invariant unit test + e2e unchanged.

**Chunk 4 — the second pass + gate.** Port `is_winner_mode_processing_enabled`
(+ `bypass_winner_mode_processing`, though its `prune_winner_mode_eval_level` arm is speed-6;
implement it correctly anyway) and add the winner re-eval loop to `rd_pick_intra_sby_mode_y`
(the multi-winner and single-winner branches, §2.1), calling the tx search with the
WINNER_MODE_EVAL policy and `intra_block_yrd` semantics (§2.3). With the gate returning 0 at
speed ≤3, this is dead code there. *Files:* `intra_rd.rs`, `rd_pick.rs` (thread the extra
policies + winner cfg through `IntraSbySearchCfg`). *Gate:* e2e unchanged at speed 0-3.

**Chunk 5 — activate speed 4 + byte-match.** Flip the speed-4 gates on in `SpeedFeatures`
(chunk 0 already added them; here make the leaf actually pass the MODE_EVAL/WINNER policies and
`multi_winner_mode_type=DEFAULT` when `speed>=4`). Also land the speed-4 tx-type/coeff scalar
deltas that the two-pass consumes (`winner_mode_tx_type_pruning=2`, `fast_intra_tx_type_search=2`,
`prune_2d_txfm_mode=TX_TYPE_PRUNE_3`, `prune_tx_type_est_rd=1`, `perform_coeff_opt=5`,
`tx_domain_dist_thres_level=3`) — these only make sense once the two passes exist. Build a
speed-4 allintra e2e cell (new gate, following the speed-1 gate pattern) and root-cause any
divergence with the sibling-C RD-dump method. *Files:* `speed_features.rs`, `partition_pick.rs`
(leaf policy selection), `tx_search.rs` (tx-type prune levels), tests. *Gate:* **speed-4
allintra byte-match** (the real acceptance criterion for the whole subsystem).

**Chunk 6 — speeds 5-6 retune (rides on the machinery).** Speed 5: `multi_winner_mode_type=FAST`
(2 winners). Speed 6: `multi_winner_mode_type=OFF` + `prune_winner_mode_eval_level=1` +
`dc_blk_pred_level=1` + `winner_mode_tx_type_pruning=3` + `perform_coeff_opt=6` +
`tx_domain_dist_level=3`. Small once chunks 1-5 land. Note speed 6 uses the **single-winner**
branch (OFF but per-tool flags stay on) — validates that path. *Files:* `speed_features.rs`,
tests. *Gate:* speed-5 and speed-6 allintra byte-match.

*Dependency note:* the speed-3 `use_skip_flag_prediction=2` delta (§5.4) is a **prerequisite**
for the speed-4 match (it changes the MODE_EVAL `skip_txfm_level`) and is independently a
speed-3 worklist row — land it with chunk 0/1's plumbing.

*Chroma:* out of scope for the primary luma envelope. Chroma runs at DEFAULT_EVAL
(`rdopt.c:3659`), so `intra_uv_rd.rs` needs **no** winner two-pass. The one chroma winner
interaction, `prune_chroma_modes_using_luma_winner=1` (speed 4, `:480`), is a separate
`{chroma}` worklist item for a 4:2:0 cell — do not fold it in here.

---

## 5. Risks / subtleties (where bit-exactness can slip)

**5.1 Winner-list tie-break (`store_winner_mode_stats`).** The compare is strict `>`
(`rdopt_utils.h:702`): equal-RD candidates keep first-seen order, and a full list drops an
incoming equal-or-worse candidate. A port using `>=`, a stable-vs-unstable sort, or a different
scan direction will pick a different top-N and diverge. Replicate the `memmove` insertion
literally. The visit order (`set_y_mode_and_delta_angle` sequence) must also match, since it
determines "first-seen." **[CONFIRMED]**

**5.2 `best_rd` carries across passes.** Pass 1's `best_rd` is the starting bound for pass 2
(`intra_mode_search.c:1689` reuses it) and, with `use_rd_based_breakout_for_intra_tx_search=true`
at speed ≥3, becomes the `ref_best_rd` early-exit threshold inside `intra_block_yrd`
(`:1200-1202`). A wrong pass-1 `best_rd` doesn't just mis-rank — it changes the pass-2 tx
search's breakout, cascading. Thread the running `best_rd` exactly as C does. **[CONFIRMED]**

**5.3 Pass-2 re-eval mutates the winner.** In multi-winner mode, `intra_block_yrd` overwrites
`best_mbmi/best_rd/*rate/tx_type_map` whenever a re-evaluated winner beats the running best
(`:1217-1225`). The final `tx_type_map` (and thus the coded tx types) comes from the
**post-re-eval** winner, not the pass-1 winner. The port's `IntraSbyBest.winners`
(`intra_rd.rs`) must be refreshed from the pass-2 tx search, and `winner_tx_type_map`
(`rd_pick.rs:203`) must read the re-eval result. Getting pass 1 right but forgetting to
overwrite from pass 2 will silently code the wrong tx types. **[CONFIRMED]**

**5.4 The MODE_EVAL first pass already diverges from DEFAULT at speed 3** via
`skip_txfm_level` (`predict_skip_levels[2]={1,2,1}`, §1.4). This means "single pass ==
DEFAULT" is **only** true through speed 2. If the port models the pass-1 policy as "DEFAULT
until speed 4", speed 3 will be wrong. The skip-txfm prediction (`predict_skip_levels`) must be
wired into the tx search's skip decision — verify whether the port models txfm-skip prediction
at all; if not, it is a latent speed-3 gap the winner-mode work surfaces. **[CONFIRMED the
table divergence; INFERRED that the port doesn't yet model skip_txfm_level.]**

**5.5 `USE_LARGESTALL` correctness.** The largest-tx-only pass must select the *same* "largest"
tx size C does (`av1_get_max_uv_txsize`/`max_txsize_rect_lookup` semantics for luma) and run the
*same* single `uniform_txfm_yrd` (same trellis/dist policy from the MODE_EVAL columns). An
off-by-one in "largest" (e.g. rect vs square max, or the TX64 disable interaction at
`tx_search.rs:1357-1359`) changes the pass-1 RD of *every* mode and re-ranks the winners. **[INFERRED]**

**5.6 `intra_rd_variance_factor` reads shared recon state.** The ALLINTRA variance multiply
(`intra_mode_search.c:1637-1639`; port `intra_rd.rs:1000-1021`) reads the recon buffer the last
prediction wrote. In the two-pass world the winner re-eval re-predicts into `recon`, so if the
variance factor is (re)applied in pass 2 it must read the pass-2 recon. C applies the variance
factor **only in pass 1** (`:1637`, inside the loop) — `intra_block_yrd` does **not** re-apply
it (`:1216`). The port must **not** re-apply the variance factor during the winner re-eval, or
RD will differ. **[CONFIRMED — the factor is absent from `intra_block_yrd`.]**

**5.7 `reset_mb_rd_record` on eval-stage change.** `set_mode_eval_params` resets the MB RD hash
record whenever the eval stage changes (`rdopt_utils.h:645-648`) because tx params differ per
stage. The port doesn't use that hash record on the intra path (residue hashing is inter-only,
`tx_search.rs:1403-1404`), so this is likely inert — but confirm no cross-pass caching of tx
results leaks stale RD between MODE_EVAL and WINNER_MODE_EVAL. **[INFERRED]**

**5.8 Determinism of the model-RD prune is unchanged** — `prune_intra_y_mode` runs before the
tx search and is eval-column-independent, so the *set* of modes reaching pass 1 is identical to
today. The winner-mode change is confined to the tx-search policy + collection + re-eval. This
bounds the blast radius: if a speed-4 cell diverges, the cause is in the two-pass tx evaluation,
not the mode enumeration/prune. **[CONFIRMED]**

---

*Provenance: line-cites read directly from `reference/libaom/av1/encoder/`
(`speed_features.c`, `intra_mode_search.c`, `rdopt_utils.h`, `rd.h`, `rdopt.c`,
`speed_features.h`) and the port (`crates/aom-encode/src/`: `speed_features.rs`, `intra_rd.rs`,
`tx_search.rs`, `rd_pick.rs`, `partition_pick.rs`) on 2026-07-15, libaom v3.14.1 git 03087864.
**[CONFIRMED]** = read in source; **[INFERRED]** = logical consequence. Companion worklist:
`docs/cpu_used_allintra_sweep_plan.md`.*
