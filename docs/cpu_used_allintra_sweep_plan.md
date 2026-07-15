# `--cpu-used` 0..9 allintra speed-feature sweep — per-level porting worklist (task #10, Gate 2)

**Scope.** Source-verified worklist of every `SPEED_FEATURES` field the C reference
changes at each `if (speed >= N)` boundary on the **all-intra KEY-frame** path, for
`speed = 2..9`. The port (`crates/aom-encode/src/speed_features.rs`,
`SpeedFeatures::set_allintra`) currently models **only speed 0 + speed 1** faithfully;
for `speed >= 2` it reuses the speed-1 deltas (module doc lines 28-34). This doc is the
map of the real per-level deltas the encoder must port.

**Reference:** libaom v3.14.1 (git 03087864), `av1/encoder/speed_features.c`. Three
allintra setters (dispatch confirmed for `AOM_USAGE_ALL_INTRA` at
`av1_set_speed_features_framesize_dependent`:2661 and
`av1_set_speed_features_framesize_independent`:2716):

- **INDEP** = `set_allintra_speed_features_framesize_independent` (345-616) — the main
  `if (speed >= N)` cascade.
- **DEP** = `set_allintra_speed_feature_framesize_dependent` (166-343) — resolution-gated
  deltas (the KB-3 class).
- **QIDX** = `av1_set_speed_features_qindex_dependent` (2873-3116) — qindex-gated deltas
  (the sweep spans qindex, so these matter).

Every line number below was read directly from the source. Values quoted verbatim.

---

## Classification legend

**Reachability tag:**

- **[KEY]** — affects the all-intra KEY-frame bitstream (partition / intra-mode / tx-size /
  tx-type / coeff-opt / winner-mode / deblock-level decisions on a single intra frame).
- **[INTER]** — only reachable on inter frames / not exercised in single-frame allintra.
  Verified inter-gates found while classifying:
  - **All `simple_motion_search_*` + `ml_early_term_after_part_split_level`** — the SMS
    subsystem is initialized only when `!frame_is_intra_only(cm)` (`encodeframe.c:681`,
    the `use_simple_motion_search` guard). Never fires on a KEY frame.
  - **`auto_max_partition_based_on_simple_motion`** — consumed only inside
    `use_auto_max_partition(...)`, which returns
    `!frame_is_intra_only(cm) && !use_screen_content_tools && ... != NOT_IN_USE`
    (`partition_strategy.h:191`). Inert on KEY frames. (Contrast: the *unconditional*
    `default_max/min_partition_size` set alongside it in
    `set_max_min_partition_size`, `partition_strategy.h:221-230`, DO apply on KEY.)
  - **All `mv_sf.*`, `tpl_sf.*`, `inter_sf.*` (except cost-upd, see below), `gm_sf.*`,
    `interp_sf.*`, `hl_sf.high_precision_mv_usage`, `inter_sf.reuse_mask_search_results`**
    — motion / temporal-lookahead / inter-mode machinery, no reference frame on a lone KEY.

**Byte-exactness impact tag** (for the PRIMARY config: sub-720p, 8-bit, non-screen,
single KEY frame, CDEF+restoration+QM OFF per `CLAUDE.md` envelope):

- **[HIGH]** — changes partition / tx / coeff / intra-mode / deblock-level decisions →
  alters output bytes.
- **[MED]** — changes a search bound / prune that *usually* leaves the winner unchanged but
  can tip a near-tie (KB-2 class).
- **[LOW]** — inert for the primary config (CDEF-off / restoration-off / HBD-only /
  chroma-only-on-monochrome / dead field). Listed for provenance; skip for the primary
  byte-match, revisit for the corresponding non-default cell.

**Conditional markers:** `{chroma}` = only bites a 4:2:0 cell (monochrome/luma path inert —
see the KB-2 latent chroma follow-up); `{480p+}`/`{720p+}`/`{1080p+}` = framesize-gated,
inert at the 256×256 primary but live in a full-resolution sweep; `{hbd}` = 10-bit only;
`{scc}` = screen-content only; `{qidx≤N}` = qindex-gated.

---

## Winner-mode system — the structural wall at speed 4 (flagged: MAJOR chunk)

`enable_winner_mode_for_*` / `multi_winner_mode_type` first activate for allintra at
**speed 4** (INDEP:502-505). This is not a scalar flip — it switches mode/tx/coeff
evaluation to a **two-pass** scheme: a first `MODE_EVAL` pass (cheap thresholds) narrows
to the top candidates, then a `WINNER_MODE_EVAL` pass re-evaluates them with different
`coeff_opt` / `tx_domain_dist` / `tx_size` thresholds and (for `multi_winner_mode`) keeps
several winners for a full second RD pass.

- **Port impact:** `SpeedFeatures::tx_type_search_policy` (`speed_features.rs:277`) hard-reads
  the `DEFAULT_EVAL` column and its doc states "winner-mode off through speed 3" — that
  assumption **breaks at speed 4**. The winner-mode column of `TX_DOMAIN_DIST_THRESHOLDS`
  (`speed_features.rs:49`), `TX_DOMAIN_DIST_TYPES` (:58), `COEFF_OPT_THRESHOLDS` (:63) must be
  wired, and `winner_mode_params` (`tx_size_search_methods`, `predict_dc_level`, the coeff/dist
  thresholds copied at `speed_features.c:2820-2828`) resolved.
- **Consumers to port:** `intra_mode_search.c:295,1519,1648,1689` (multi-winner luma mode
  eval), `partition_search.c:414` + `encodeframe.c:2456` (`enable_winner_mode_for_tx_size_srch`),
  `rdopt_utils.h:408` (`prune_winner_mode_eval_level`), winner-mode DC-block prediction
  (`predict_dc_level` ← `dc_blk_pred_level`).
- **Everything at speed >= 4 assumes this exists.** Port the winner-mode subsystem before
  the speed-4 tx/coeff scalar deltas, or none of speed 4-6 will byte-match.

Winner-mode field timeline: `MULTI_WINNER_MODE_DEFAULT` @4 (:504) → `MULTI_WINNER_MODE_FAST`
@5 (:524) → `MULTI_WINNER_MODE_OFF` + `prune_winner_mode_eval_level=1` + `dc_blk_pred_level=1`
@6 (:561-563). Note `multi_winner_mode_type` returns to OFF at 6 but the per-tool
`enable_winner_mode_for_*` flags set at 4 stay on.

---

## Two more structural walls: VAR-partition @7, nonrd @8

- **Speed 7 — `part_sf.partition_search_type = VAR_BASED_PARTITION` (INDEP:571).** Replaces
  the exhaustive RD partition search with a variance-threshold partitioner
  (`encodeframe.c:874`, `set_max_min_partition_size` no longer drives it). A completely
  different partition algorithm → a second large chunk, independent of winner-mode.
- **Speed 8 — `rt_sf.use_nonrd_pick_mode = 1` (INDEP:579).** Switches mode selection to the
  non-RD pick-mode path (`encodeframe.c:1223,2425`). Third large chunk; speed 9 only refines
  it. The `rt_sf.*` intra fields (`hybrid_intra_pickmode`, `intra_y_mode_bsize_mask_nrd`,
  the speed-9 nonrd prunes) are all part of this path.

---

## Per-level deltas

Legend for the "modeled?" column: ✓ = the field exists in the port's `SpeedFeatures` struct
(may hold the wrong value for speed≥2); ✗ = not present in the port at all.

### speed >= 2  (INDEP:424-437, DEP:236-267, QIDX:2939-2973 / 2987-3030)

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `intra_sf.disable_smooth_intra` | `1` | INDEP:429 | intra-mode: drop SMOOTH_PRED from luma search | ✗ | [KEY][HIGH] |
| `intra_sf.intra_pruning_with_hog` | `2` | INDEP:430 | intra-mode: luma HOG dir-prune aggressiveness | ✓(=1) | [KEY][HIGH] |
| `intra_sf.prune_filter_intra_level` | `1` | INDEP:431 | intra-mode: filter-intra prune | ✗ | [KEY][HIGH] |
| `rd_sf.perform_coeff_opt` | `3` | INDEP:433 | coeff-opt threshold (`COEFF_OPT_THRESHOLDS[3]`) | ✓(=2) | [KEY][HIGH] |
| `part_sf.use_square_partition_only_threshold` | 720p:`64X64` else `32X32` | DEP:238-242 | partition: rect-kill threshold (KB-3 field) | ✓ (KB-3) | [KEY][HIGH] |
| `tx_sf.tx_type_search.prune_tx_type_using_stats` | `1` `{480p+}` | DEP:262 | tx-type: stats-based prune | ✗ | [KEY][HIGH]{480p+} |
| `part_sf.partition_search_breakout_dist_thr`/`rate_thr` | 720p:`1<<24`/120 else `1<<22`/100 | DEP:254-258 | partition: NONE-breakout thresholds | ✗ | [KEY][MED] |
| `part_sf.ml_4_partition_search_level_index` | `2` | DEP:237 | partition: HORZ4/VERT4 ML-prune row | ✓(=1) | [KEY][MED] |
| `tx_sf.prune_tx_size_level` | `2`/`3` `{hbd}` | DEP:263,265 | tx-size prune (HBD only) | ✗ | [KEY][LOW]{hbd} |
| `part_sf.ext_partition_eval_thresh` | `BLOCK_128X128` `{qidx≤thr,!boosted}` | QIDX:2951-2952 | partition: ext-partition eval gate | ✗ | [KEY][HIGH]{qidx} |
| `mv_sf.auto_mv_step_size` | `1` | INDEP:425 | motion step size | — | [INTER] |
| `part_sf.simple_motion_search_prune_agg` | `SIMPLE_AGG_LVL2` | INDEP:427 | SMS partition prune | — | [INTER] |
| `mv_sf`/`tpl_sf.search_method` by qidx | (various) | QIDX:2995-3026 | motion search method | — | [INTER] |
| `lpf_sf.prune_wiener_based_on_src_var`=1, `prune_sgr_based_on_wiener`=1 | | INDEP:435-436 | LR search prune | ✗ | [KEY][LOW] (restoration off) |
| `part_sf.less_rectangular_check_level` | `1` (unconditional @ speed≤2) | QIDX:3029 | partition rect check | ✓(=1) | [KEY][LOW] (=base) |

**speed-2 [KEY] deltas (primary): 4** (`disable_smooth_intra`, `intra_pruning_with_hog=2`,
`prune_filter_intra_level`, `perform_coeff_opt=3`) + the DEP `use_square_partition_only_threshold`
extension (KB-3 field, already wired — just add the speed-2 row).

### speed >= 3  (INDEP:439-469, DEP:269-290, QIDX:3032-3034)

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `part_sf.less_rectangular_check_level` | `2` (qidx≥170 → `1`) | INDEP:444, QIDX:3032-3034 | partition: SPLIT-stage rect kill | ✓(=1) | [KEY][HIGH] + [qidx] |
| `intra_sf.intra_pruning_with_hog` | `3` | INDEP:455 | intra-mode: luma HOG prune | ✓ | [KEY][HIGH] |
| `intra_sf.chroma_intra_pruning_with_hog` | `2` | INDEP:454 | chroma-mode HOG prune | ✗ | [KEY][HIGH]{chroma} |
| `intra_sf.prune_palette_search_level` | `2` | INDEP:456 | palette search prune | ✓(=1) | [KEY][MED] |
| `tx_sf.tx_type_search.use_skip_flag_prediction` | `2` | INDEP:459 | tx: skip-flag prediction | ✗ | [KEY][HIGH] |
| `tx_sf.use_rd_based_breakout_for_intra_tx_search` | `true` | INDEP:460 | intra tx-search RD breakout | ✗ | [KEY][HIGH] |
| `part_sf.prune_ext_part_using_split_info` | `1` | INDEP:446 | partition: ext-part prune via split info | ✗ | [KEY][MED] |
| `hl_sf.recode_loop` | `ALLOW_RECODE_KFARFGF` | INDEP:441 | frame recode on KF (verify: fixed-Q may not recode) | ✗ | [KEY][MED] |
| `hl_sf.screen_detection_mode2_fast_detection` | `1` | INDEP:442 | screen-content detection speed | ✗ | [KEY][MED] |
| `tx_sf.adaptive_txb_search_level` | `2` (no change; already 2 @1) | INDEP:458 | txb search | ✓ | [KEY][LOW] (=speed1) |
| `part_sf.max_intra_bsize` | `BLOCK_32X32` `{<720p}` | DEP:285 | partition: max intra block size | ✗ | [KEY][HIGH]{<720p} |
| `part_sf.partition_search_breakout_dist_thr`/`rate_thr` | 720p:`1<<25`/200 else `1<<23`/120 | DEP:282-287 | partition NONE-breakout | ✗ | [KEY][MED] |
| `part_sf.ml_4_partition_search_level_index` | `3` | DEP:271 | HORZ4/VERT4 ML row | ✓ | [KEY][MED] |
| `part_sf.rect_partition_eval_thresh` | `BLOCK_8X8` `{qidx≤65,!boosted,480p+}` | QIDX:2975-2984 | partition: rect eval gate | ✗ | [KEY][HIGH]{qidx,480p+} |
| `part_sf.ml_early_term_after_part_split_level` | `0` | DEP:270 | SMS split early-term | — | [INTER] |
| `hl_sf.high_precision_mv_usage`, `mv_sf.full_pixel_search_level`, `mv_sf.search_method`, `simple_motion_search_prune_agg`, `prune_ext_part_using_split_info` MV path | | INDEP:440,445,448-449 | motion / SMS | — | [INTER] |
| `lpf_sf.prune_sgr_based_on_wiener`/`disable_loop_restoration_chroma`/`reduce_wiener_window_size`/`prune_wiener_based_on_src_var` | | INDEP:465-468 | LR search | ✗ | [KEY][LOW] (restoration off) |

**speed-3 [KEY] deltas (primary): 8** — `less_rectangular_check_level=2` (with the qidx≥170→1
override), `intra_pruning_with_hog=3`, `prune_palette_search_level=2`,
`use_skip_flag_prediction=2`, `use_rd_based_breakout_for_intra_tx_search`,
`prune_ext_part_using_split_info=1`, `recode_loop` (verify), `screen_detection_mode2`.
Plus `{chroma}` `chroma_intra_pruning_with_hog=2` and `{<720p}` `max_intra_bsize`.

### speed >= 4  (INDEP:471-506) — WINNER MODE ACTIVATES

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `winner_mode_sf.enable_winner_mode_for_coeff_opt` | `1` | INDEP:502 | ★ winner-mode coeff-opt pass | ✗ | [KEY][HIGH] |
| `winner_mode_sf.enable_winner_mode_for_use_tx_domain_dist` | `1` | INDEP:503 | ★ winner-mode tx-domain-dist pass | ✗ | [KEY][HIGH] |
| `winner_mode_sf.multi_winner_mode_type` | `MULTI_WINNER_MODE_DEFAULT` | INDEP:504 | ★ multi-winner mode eval | ✗ | [KEY][HIGH] |
| `winner_mode_sf.enable_winner_mode_for_tx_size_srch` | `1` | INDEP:505 | ★ winner-mode tx-size search | ✗ | [KEY][HIGH] |
| `tx_sf.tx_type_search.winner_mode_tx_type_pruning` | `2` | INDEP:488 | tx-type prune in winner pass | ✗ | [KEY][HIGH] |
| `tx_sf.tx_type_search.fast_intra_tx_type_search` | `2` | INDEP:489 | intra tx-type fast search | ✗ | [KEY][HIGH] |
| `tx_sf.tx_type_search.prune_2d_txfm_mode` | `TX_TYPE_PRUNE_3` | INDEP:490 | tx-type 2D prune | ✓(=PRUNE_2) | [KEY][HIGH] |
| `tx_sf.tx_type_search.prune_tx_type_est_rd` | `1` | INDEP:491 | tx-type est-RD prune (intra path) | ✗ | [KEY][HIGH] |
| `rd_sf.perform_coeff_opt` | `5` | INDEP:493 | coeff-opt threshold (`[5]`, adds SATD col) | ✓(=2) | [KEY][HIGH] |
| `rd_sf.tx_domain_dist_thres_level` | `3` | INDEP:494 | tx-domain-dist threshold (`[3]`=0) | ✓(=1) | [KEY][HIGH] |
| `intra_sf.prune_chroma_modes_using_luma_winner` | `1` | INDEP:480 | chroma-mode prune (also forces `chroma_intra_pruning_with_hog=0`, :614) | ✗ | [KEY][HIGH]{chroma} |
| `part_sf.early_term_after_none_split` | `1` | INDEP:477 | partition: early term after NONE+SPLIT | ✗ | [KEY][MED] |
| `part_sf.prune_ext_part_using_split_info` | `2` | INDEP:476 | partition ext-part prune | ✗ | [KEY][MED] |
| `lpf_sf.lpf_pick` | `LPF_PICK_FROM_FULL_IMAGE_NON_DUAL` | INDEP:496 | deblock-level pick (non-dual search) — changes LF-level in header (`picklpf.c:376`) | ✗ | [KEY][HIGH] |
| `part_sf.ml_predict_breakout_level` | `3` | INDEP:478 | partition ML breakout (8-bit: =base 3) | ✗ | [KEY][LOW] (8-bit no-op) |
| `tx_sf.tx_type_search.prune_tx_type_using_stats` | `2` `{480p+}` | DEP:300 | tx-type stats prune | ✗ | [KEY][HIGH]{480p+} |
| `part_sf.partition_search_breakout_dist_thr` | 720p:`1<<26` else `1<<24` | DEP:294-296 | partition NONE-breakout | ✗ | [KEY][MED] |
| `part_sf.less_rectangular_check_level` | `2` (unconditional @ speed≥4) | QIDX:3048 | partition rect check | ✓ | [KEY] (=indep) |
| `mv_sf.*` (subpel_search_method, simple_motion_subpel_force_stop, reduce_search_range, hash_max_8x8_intrabc), `tpl_sf.*`, `simple_motion_search_*` | | INDEP:472-486,499-500 | motion / SMS / TPL / intrabc | — | [INTER] |
| `lpf_sf.cdef_pick_method` | `CDEF_FAST_SEARCH_LVL3` | INDEP:497 | CDEF search | ✗ | [KEY][LOW] (CDEF off) |

**speed-4 [KEY] deltas (primary): 14** — of which **4 are the winner-mode activation ★**
(the MAJOR chunk). The rest are tx-type/coeff/deblock scalar flips that only make sense once
winner-mode exists.

### speed >= 5  (INDEP:508-525)

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `part_sf.ext_partition_eval_thresh` | non-scc:`BLOCK_16X16` scc:`BLOCK_8X8` | INDEP:510-511 | partition: ext-partition eval gate | ✗ | [KEY][HIGH] |
| `winner_mode_sf.multi_winner_mode_type` | `MULTI_WINNER_MODE_FAST` | INDEP:524 | multi-winner eval (fewer winners) | ✗ | [KEY][HIGH] |
| `lpf_sf.use_coarse_filter_level_search` | `0` | INDEP:517 | deblock-level search granularity | ✗ | [KEY][MED] |
| `intra_sf.chroma_intra_pruning_with_hog` | `3` | INDEP:515 | chroma HOG prune — **inert @≥4** (forced 0 by `prune_chroma_modes_using_luma_winner`, :614) | ✗ | [KEY][LOW]{chroma} |
| `part_sf.intra_cnn_based_part_prune_level` | scc:`1` non-scc:`2` (=base) | INDEP:512-513 | CNN partition prune (non-scc no-op) | ✓ | [KEY][LOW] (non-scc =speed1) |
| `part_sf.prune_sub_8x8_partition_level` | `0` `{scc,qidx<128,≤480p}` | QIDX:3071-3077 | partition sub-8x8 prune | ✗ | [KEY][LOW]{scc} |
| `part_sf.simple_motion_search_prune_agg`, `mv_sf.prune_mesh_search` | `SIMPLE_AGG_LVL5`, `PRUNE_MESH_SEARCH_LVL_2` | INDEP:509,522 | SMS / mesh | — | [INTER] |
| `lpf_sf.disable_wiener_filter`/`disable_sgr_filter` | `true` | INDEP:519-520 | LR disable | ✗ | [KEY][LOW] (restoration off) |

**speed-5 [KEY] deltas (primary): 3** — `ext_partition_eval_thresh=BLOCK_16X16`,
`multi_winner_mode_type=FAST`, `use_coarse_filter_level_search=0`. (The chroma HOG delta is a
verified no-op at speed≥4.) Small level once winner-mode exists.

### speed >= 6  (INDEP:527-564, DEP:304-316)

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `intra_sf.prune_filter_intra_level` | `2` | INDEP:529 | filter-intra prune | ✗ | [KEY][HIGH] |
| `intra_sf.intra_pruning_with_hog` | `4` | INDEP:531 | luma HOG prune | ✓ | [KEY][HIGH] |
| `intra_sf.top_intra_model_count_allowed` | `2` | INDEP:533 | intra top-N modes to full RD | ✓(=3) | [KEY][HIGH] |
| `intra_sf.prune_luma_odd_delta_angles_in_intra` | `1` | INDEP:535 | intra: prune odd angle-deltas (KB-2 area) | ✗ | [KEY][HIGH] |
| `intra_sf.cfl_search_range` | `1` | INDEP:532 | CfL (chroma-from-luma) search range | ✗ | [KEY][HIGH]{chroma} |
| `intra_sf.adapt_top_model_rd_count_using_neighbors` | `1` | INDEP:534 | intra top-model adapt | ✗ | [KEY][MED] |
| `intra_sf.prune_smooth_intra_mode_for_chroma` | `1` | INDEP:528 | chroma SMOOTH prune | ✗ | [KEY][MED]{chroma} |
| `part_sf.prune_rectangular_split_based_on_qidx` | non-scc:`2` scc:`0` | INDEP:537-538 | partition: qidx rect-split prune | ✗ | [KEY][HIGH][qidx] |
| `part_sf.prune_rect_part_using_4x4_var_deviation` | `true` | INDEP:539 | partition: 4x4-var rect prune | ✗ | [KEY][HIGH] |
| `part_sf.prune_rect_part_using_none_pred_mode` | `true` | INDEP:540 | partition: NONE-pred-mode rect prune | ✗ | [KEY][HIGH] |
| `part_sf.default_max_partition_size` | `BLOCK_32X32` | INDEP:546 | partition: max start size (always applied) | ✗ | [KEY][HIGH] |
| `part_sf.prune_part4_search` | `3` | INDEP:543 | partition: 4-way search prune (base 2, :355) | ✗ | [KEY][MED] |
| `part_sf.prune_sub_8x8_partition_level` | non-scc:`1` scc:`0` | INDEP:541-542 | partition sub-8x8 prune | ✗ | [KEY][MED] |
| `tx_sf.tx_type_search.winner_mode_tx_type_pruning` | `3` | INDEP:551 | tx-type prune (winner) | ✗ | [KEY][HIGH] |
| `tx_sf.prune_intra_tx_depths_using_nn` | `true` | INDEP:553 | intra tx-depth NN prune | ✗ | [KEY][HIGH] |
| `tx_sf.tx_type_search.prune_tx_type_est_rd` | `0` | INDEP:552 | reverts speed-4's 1→0 | ✗ | [KEY][MED] |
| `rd_sf.perform_coeff_opt` | `6` | INDEP:555 | coeff-opt threshold (`[6]`) | ✓ | [KEY][HIGH] |
| `rd_sf.tx_domain_dist_level` | `3` | INDEP:556 | tx-domain-dist type (`[3]`) | ✓(=1) | [KEY][HIGH] |
| `lpf_sf.lpf_pick` | `LPF_PICK_FROM_Q` | INDEP:559 | deblock-level from Q (not searched) — big LF-level change (`picklpf.c:266`) | ✗ | [KEY][HIGH] |
| `winner_mode_sf.multi_winner_mode_type` | `MULTI_WINNER_MODE_OFF` | INDEP:561 | multi-winner off (per-tool flags stay) | ✗ | [KEY][HIGH] |
| `winner_mode_sf.prune_winner_mode_eval_level` | `1` | INDEP:562 | winner-eval prune (`rdopt_utils.h:414`) | ✗ | [KEY][MED] |
| `winner_mode_sf.dc_blk_pred_level` | `1` | INDEP:563 | winner DC-block pred threshold (`speed_features.c:2826`) | ✗ | [KEY][LOW] |
| `intra_sf.chroma_intra_pruning_with_hog` | `4` | INDEP:530 | chroma HOG — inert @≥4 (forced 0, :614) | ✗ | [KEY][LOW]{chroma} |
| `part_sf.use_square_partition_only_threshold` | `BLOCK_16X16` | DEP:315 | partition rect-kill (KB-3 field) | ✓ | [KEY][HIGH] |
| `part_sf.default_min_partition_size` | `BLOCK_8X8` `{1080p+}` | DEP:312 | partition min size | ✗ | [KEY][HIGH]{1080p+} |
| `auto_max_partition_based_on_simple_motion`, `mv_sf.use_bsize_dependent_search_method`, `mv_sf.intrabc_search_level` | | INDEP:548-549, DEP:304-309 | SMS / motion / intrabc | — | [INTER] |
| `lpf_sf.cdef_pick_method` | `CDEF_FAST_SEARCH_LVL4` | INDEP:558 | CDEF search | ✗ | [KEY][LOW] (CDEF off) |

**speed-6 [KEY] deltas (primary): ~19** — the heaviest "normal-RD" level (last before the
nonrd walls). Big blocks: intra-mode prunes (odd-angle-delta, filter-intra, top-N), the four
rect-partition prunes, `default_max_partition_size`, tx-type/coeff prunes, `lpf_pick=FROM_Q`,
winner-mode retune. Plus the DEP `use_square_partition_only_threshold=BLOCK_16X16` (KB-3 field).

### speed >= 7  (INDEP:569-575) — VAR-BASED PARTITION WALL

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `part_sf.partition_search_type` | `VAR_BASED_PARTITION` | INDEP:571 | ★★ variance-based partitioner (replaces RD partition search, `encodeframe.c:874`) | ✗ | [KEY][HIGH] |
| `rt_sf.var_part_split_threshold_shift` | `7` | INDEP:574 | var-partition split threshold | ✗ | [KEY][HIGH] |
| `part_sf.default_min_partition_size` | `BLOCK_8X8` | INDEP:570 | partition min size (always applied) | ✗ | [KEY][HIGH] |
| `rt_sf.mode_search_skip_flags` | `\|= FLAG_SKIP_INTRA_DIRMISMATCH` | INDEP:573 | intra dir-mismatch skip | ✗ | [KEY][MED] |
| `lpf_sf.cdef_pick_method` | `CDEF_PICK_FROM_Q` | INDEP:572 | CDEF from Q | ✗ | [KEY][LOW] (CDEF off) |

**speed-7 [KEY] deltas (primary): 4** — dominated by the `VAR_BASED_PARTITION` structural
switch ★★. (Speed 7 still uses the RD mode path — `use_nonrd_pick_mode` is 0 until speed 8 —
but partitions come from the variance partitioner.)

### speed >= 8  (INDEP:577-590, DEP:322-329) — NONRD PICK-MODE WALL

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `rt_sf.use_nonrd_pick_mode` | `1` | INDEP:579 | ★★ non-RD mode selection path (`encodeframe.c:1223,2425`) | ✗ | [KEY][HIGH] |
| `rt_sf.hybrid_intra_pickmode` | `2` | INDEP:578 | nonrd intra pick mode | ✗ | [KEY][HIGH] |
| `rt_sf.intra_y_mode_bsize_mask_nrd[]` | `INTRA_DC` (≥32x32) / `INTRA_DC_H_V` (<32x32) | INDEP:584-589 | nonrd intra Y-mode mask per bsize | ✗ | [KEY][HIGH] |
| `rt_sf.nonrd_check_partition_merge_mode` | `1` (`2` if `{<480p}`, DEP:324) | INDEP:580, DEP:323-324 | nonrd partition merge | ✗ | [KEY][HIGH] |
| `rt_sf.var_part_split_threshold_shift` | `8` | INDEP:581 | var-partition split threshold | ✗ | [KEY][HIGH] |
| `rt_sf.prune_palette_search_nonrd` | `1` | INDEP:582 | nonrd palette prune | ✗ | [KEY][MED] |
| `rt_sf.force_large_partition_blocks_intra` | `1` `{720p+}` | DEP:326-327 | nonrd force large intra blocks | ✗ | [KEY][MED]{720p+} |

**speed-8 [KEY] deltas (primary): 6** — dominated by `use_nonrd_pick_mode` ★★ (the whole
nonrd intra pipeline). This is a distinct pipeline from the RD path used at speed 0-7.

### speed >= 9  (INDEP:592-606, DEP:331-342)

| Field | New value | line | stage | modeled? | tags |
|---|---|---|---|---|---|
| `rt_sf.nonrd_check_partition_merge_mode` | `0` | INDEP:596 | nonrd partition merge off | ✗ | [KEY][HIGH] |
| `rt_sf.hybrid_intra_pickmode` | `0` | INDEP:597 | nonrd intra pick off | ✗ | [KEY][HIGH] |
| `rt_sf.var_part_split_threshold_shift` | `7` | INDEP:601 | var-partition threshold (lower than speed 8) | ✗ | [KEY][HIGH] |
| `rt_sf.vbp_prune_16x16_split_using_min_max_sub_blk_var` | `true` | INDEP:602 | var-partition 16x16 split prune | ✗ | [KEY][HIGH] |
| `rt_sf.prune_h_pred_using_best_mode_so_far` | `true` | INDEP:603 | nonrd H_PRED prune | ✗ | [KEY][HIGH] |
| `rt_sf.enable_intra_mode_pruning_using_neighbors` | `true` | INDEP:604 | nonrd neighbor intra prune | ✗ | [KEY][HIGH] |
| `rt_sf.prune_intra_mode_using_best_sad_so_far` | `true` | INDEP:605 | nonrd SAD intra prune | ✗ | [KEY][HIGH] |
| `inter_sf.coeff_cost_upd_level` | `INTERNAL_COST_UPD_SBROW` (`OFF` if `{<4k}`, DEP:339) | INDEP:593, DEP:333-340 | coeff-cost update cadence (`encodeframe_utils.c:1633`) — affects RD rate model | ✗ | [KEY][MED] |
| `inter_sf.mode_cost_upd_level` | `INTERNAL_COST_UPD_SBROW` (`OFF` if `{<4k}`, DEP:340) | INDEP:594, DEP:333-340 | mode-cost update cadence | ✗ | [KEY][MED] |

**speed-9 [KEY] deltas (primary): 9** — all nonrd-path refinements + the two cost-update
cadence flips. Depends entirely on the speed-8 nonrd pipeline existing first.

---

## Framesize-DEPENDENT deltas — need per-resolution handling (KB-3 class)

The DEP setter (166-343) gates on `AOMMIN(width,height)` buckets (480/720/1080/2160). The
256×256 primary is **sub-480p**, so many are inert there but live in a full-size sweep. Per
`CLAUDE.md` KB-3, the port already wires `use_square_partition_only_threshold` (framesize+speed
dependent) into `rd_pick_partition_real` — extend the same mechanism to:

- `use_square_partition_only_threshold` — speed 2 (DEP:238-242) and speed 6 (`BLOCK_16X16`,
  DEP:315). Already the KB-3 field; add the speed-2/6 rows. **[KEY][HIGH]**
- `max_intra_bsize=BLOCK_32X32` `{<720p}` — speed 3 (DEP:285). **[KEY][HIGH]** — bites the
  sub-720p primary at speed 3.
- `prune_tx_type_using_stats` — speed 2 `{480p+}`→1 (DEP:262), speed 4 `{480p+}`→2 (DEP:300).
  **[KEY][HIGH]{480p+}**
- `partition_search_breakout_dist_thr`/`rate_thr` — speeds 2/3/4 (DEP:254-258,282-287,294-296),
  720p-vs-below values. **[KEY][MED]**
- `default_min_partition_size=BLOCK_8X8` — speed 6 `{1080p+}` (DEP:312), speed 7 unconditional
  (INDEP:570). **[KEY][HIGH]**
- `ml_4_partition_search_level_index` = `min(speed,3)` — speeds 2/3 (DEP:237,271). **[KEY][MED]**
- `prune_tx_size_level` — speed 2 `{hbd}` (DEP:263,265), speed 3 `{hbd}` (DEP:289). **[KEY][LOW]{hbd}**
- `nonrd_check_partition_merge_mode=2` `{<480p}` (DEP:324) / `force_large_partition_blocks_intra=1`
  `{720p+}` (DEP:327) — speed 8. **[KEY][MED]** (nonrd path)
- `coeff/mode_cost_upd_level=INTERNAL_COST_UPD_OFF` `{<4k}` — speed 9 (DEP:339-340), **overrides**
  the INDEP:593-594 SBROW for sub-4k. **[KEY][MED]**
- INTER/inert here: `auto_max_partition_based_on_simple_motion` (INTER, all speeds),
  `ml_partition_search_breakout_thresh[]`/`_model_index` (only change at the speed-0→1 boundary
  for sub-720p; also feed `av1_ml_predict_breakout`, no new sub-720p delta at speed≥2),
  `mv_sf.use_downsampled_sad` `{720p+}` (INTER).

---

## Qindex-DEPENDENT deltas — the sweep spans qindex, so these are live

From `av1_set_speed_features_qindex_dependent` (2873-3116). All apply AFTER the INDEP/DEP
cascade, so they OVERRIDE it. KEY-relevant, ordered by speed:

- **speed 0** (QIDX:2907-2912): for `{<720p, base_qindex ≤ (boosted?70:is_arf2?110:140)}` sets
  `model_based_prune_tx_search_level=0` (+ inter-only SMS fields). The port's speed-0 keeps this
  at `1` (allintra base, :368) — a **latent speed-0 qindex gap** at low qindex. **[KEY][MED][qidx]**
  — the high-qindex byte-matched cells (cq62/cq63 = qindex 249) are >140 so it never fired;
  a low-qindex speed-0 cell would diverge. Verify `frame_is_boosted` for a lone allintra KEY
  (sets the 70-vs-140 threshold).
- **speed ≥ 2** (QIDX:2939-2973): `ext_partition_eval_thresh=BLOCK_128X128`, qindex-gated. For
  KEY frames the `aggr≤1` branch (speed 2-3, :2951, `qidx≤thr && !boosted`) and the `else`
  branch (`aggr>3` = speed ≥ 6, :2971, unconditional) fire; the `aggr≤2/≤3` branches are
  `!frame_is_intra_only`-guarded (inter only). **[KEY][HIGH][qidx]** — interacts with the INDEP
  speed-5 `ext_partition_eval_thresh=BLOCK_16X16`.
- **speed ≥ 3** (QIDX:2975-2984): `rect_partition_eval_thresh=BLOCK_8X8` for
  `{qidx≤(speed≤4?65:80), !boosted, 480p+}`. **[KEY][HIGH][qidx]{480p+}**
- **speed == 3** (QIDX:3032-3034): `less_rectangular_check_level = (qidx≥170)?1:2` — overrides
  the INDEP:444 value of 2. **[KEY][MED][qidx]** — must model the 170 threshold.
- **speed ≥ 4** (QIDX:3048): `less_rectangular_check_level=2` unconditional (matches INDEP).
- **speed ≥ 5** (QIDX:3071-3077): `prune_sub_8x8_partition_level=0` for
  `{scc, qidx<128, ≤480p}`. **[KEY][LOW]{scc}[qidx]**
- ALLINTRA (QIDX:2888-2890): `zero_low_cdef_strengths=1` for `{qidx≤140}` — CDEF off →
  **[KEY][LOW]** (inert unless `--enable-cdef`).
- LR unit-size (QIDX:3080-3108, incl. the `ALLINTRA && speed≥1` branch at :3095) — restoration
  off → **[KEY][LOW]** (inert unless `--enable-restoration`).
- INTER/excluded on KEY: speed≤2 motion-search-method-by-qindex (QIDX:2987-3027), speed==1
  `reuse_mask_search_results` (QIDX:3051-3057), speed==5 `winner_mode_tx_type_pruning`
  (QIDX:3060-3068, `!(intra||scc)`-guarded).

---

## Summary

### Per-level [KEY] delta count (primary config: sub-720p, 8-bit, non-screen, single KEY)

Counts the framesize-INDEPENDENT [KEY] deltas that are NEW and non-inert at each level
(the primary byte-match surface). DEP/QIDX/chroma/hbd/larger-size conditionals are additive
on top — listed per level above.

| speed | new [KEY] INDEP deltas | of which structural | notes |
|---|---|---|---|
| 2 | 4 | — | pure scalar flips; + DEP `use_square_partition_only_threshold` (KB-3, wired) |
| 3 | 8 | — | partition/intra/tx prunes; + qidx `less_rect(170)`, `{chroma}` hog, `{<720p}` `max_intra_bsize` |
| 4 | 14 | **4 (winner-mode ★)** | winner-mode activation is the big chunk |
| 5 | 3 | — | small once winner-mode exists |
| 6 | ~19 | — | heaviest normal-RD level; + DEP `use_square_partition_only_threshold=16X16` |
| 7 | 4 | **1 (VAR_BASED_PARTITION ★★)** | variance partitioner replaces RD partition search |
| 8 | 6 | **1 (nonrd pick-mode ★★)** | whole nonrd intra pipeline |
| 9 | 9 | — | nonrd refinements; depends on speed-8 pipeline |

**Total unported [KEY] gap, speed 2..9 (framesize-independent, primary): ≈ 67 field-deltas**,
structured around **three structural walls**: winner-mode @4, VAR_BASED_PARTITION @7, nonrd
pick-mode @8. Plus the framesize-dependent set (extend the KB-3 mechanism: ~8 additional [KEY]
deltas, mostly `{480p+}`/`{720p+}` for a full-resolution sweep + `{<720p}` `max_intra_bsize`@3)
and the qindex-dependent set (~5 [KEY] overrides, of which `ext_partition_eval_thresh` and the
speed-3 `less_rect(170)` matter at the primary size).

### Suggested porting ORDER (lowest-risk / smallest first, respecting the 3 walls)

1. **Speed 2** — smallest, zero structural change. Four scalar intra/coeff flips
   (`disable_smooth_intra`, `intra_pruning_with_hog=2`, `prune_filter_intra_level`,
   `perform_coeff_opt=3`) + extend the already-wired KB-3 `use_square_partition_only_threshold`
   with the speed-2 row. Add the QIDX `ext_partition_eval_thresh` gate. Land + byte-match a
   speed-2 cell before touching anything else.
2. **Speed 3** — moderate, still no winner-mode. Partition/intra/tx prunes; the notable new
   dependencies are the QIDX `less_rectangular_check_level=(qidx≥170)?1:2` interaction and (for
   sub-720p) DEP `max_intra_bsize`. Verify `recode_loop=ALLOW_RECODE_KFARFGF` is inert under
   fixed-Q allintra (likely no recode) before spending effort on it.
3. **Speed 4 — WINNER-MODE subsystem (MAJOR chunk).** Port the two-pass MODE_EVAL /
   WINNER_MODE_EVAL machinery + `winner_mode_params` threshold resolution + the winner-mode
   columns of the coeff/dist tables. This unblocks speeds 4-6. Do the winner-mode subsystem
   FIRST, then the speed-4 tx-type/coeff/deblock scalar deltas on top.
4. **Speed 5** — small; rides on winner-mode (`multi_winner_mode_type=FAST`,
   `ext_partition_eval_thresh=16X16`, `use_coarse_filter_level_search`).
5. **Speed 6** — large but all normal-RD (last RD-path level). ~19 deltas: intra-mode prunes
   (incl. `prune_luma_odd_delta_angles_in_intra` — directly the KB-2 angle-delta area),
   four rect-partition prunes, `default_max_partition_size`, tx prunes, `lpf_pick=FROM_Q`,
   winner-mode retune, DEP `use_square_partition_only_threshold=16X16`.
6. **Speed 7 — VAR_BASED_PARTITION (2nd structural wall).** Separate chunk: the variance
   partitioner. Independent of winner-mode; can be scheduled in parallel with 4-6 by a
   different agent if desired.
7. **Speed 8 — nonrd pick-mode (3rd structural wall).** The whole nonrd intra pipeline.
8. **Speed 9** — nonrd refinements; only after 8.

**Dependency-first framing:** levels 2, 3 are cheap and cover the aggressive-quality web range
(q5-q40) where most product traffic lives — ship those first. Winner-mode @4 is the single
biggest gate (unblocks 4-6). VAR-part @7 and nonrd @8 are a **separate sub-project** (the
real-time-derived path) and can be deferred until the RD-path levels 2-6 all byte-match, since
speeds 7-9 target "approximately real-time speed 6/7/8" quality (INDEP:565-568) rather than the
archival/primary regime.

---

*Provenance: every line-cite read directly from `reference/libaom/av1/encoder/speed_features.c`
(v3.14.1, git 03087864) on 2026-07-15. Reachability gates verified in `encodeframe.c:681`,
`partition_strategy.h:191`, `intra_mode_search.c`, `tx_search.c`, `partition_search.c`,
`picklpf.c`, `rdopt_utils.h`. Port baseline: `crates/aom-encode/src/speed_features.rs`
models speed 0 + speed 1 only.*
