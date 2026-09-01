# `av1/encoder/rdopt.c` — the inter RD brain: what is ported, what is not

**Measured 2026-09-01.** This file is the honest accounting for the rdopt.c
half of INTER-ENCODE-GAPMAP §2.3 W6. It is written as a fraction with the
MISSING side first, because the inventory tool cannot produce that number:
`tools/c_surface_inventory.py` matches a NAME anywhere in the Rust tree, and
every module below cites its C source function by name in a doc comment, so
re-running it now reports rdopt.c as nearly complete. It is not.

## 1. The fraction

| | count |
|---|---|
| function definitions the inventory sees in rdopt.c | 105 |
| **NOT ported** | **29** |
| ported before 2026-09-01 | 9 |
| ported on 2026-09-01 (this wave) | 67 |

"Ported" below means a Rust function exists that implements the C function's
behaviour AND is gated against libaom. It does not mean a doc comment mentions
the name.

## 2. NOT PORTED — named first, with the reason

### 2.1 Blocked on other files' ports (11)

Each of these is orchestration whose body is mostly calls into machinery that
lives outside rdopt.c and is not ported yet. Porting the shell without the
callees would be a stub with the right name, which CLAUDE.md forbids reporting
as done.

| C function | blocked on |
|---|---|
| `motion_mode_rd` (`:1539`) | `av1_txfm_search`, `av1_interpolation_filter_search`, `av1_build_obmc_inter_prediction`, `av1_refine_warped_mv` — the OBMC/warp predictors and the full tx search |
| `set_params_rd_pick_inter_mode` (`:4331`) | `av1_setup_pred_block` (buffer plumbing), `av1_mv_pred` (rd.c, W5), the compound arm of `estimate_ref_frame_costs` |
| `skip_mode_rd` (`:1945`) | `av1_enc_build_inter_predictor` over all planes + `cpi->ppi->fn_ptr[bs].vf` |
| `rd_pick_skip_mode` (`:3705`) | `skip_mode_rd` |
| `tx_search_best_inter_candidates` (`:5471`) | `av1_txfm_search` |
| `refine_winner_mode_tx` (`:3866`) | `av1_txfm_search` |
| `handle_winner_cand` (`:5670`) | `refine_winner_mode_tx` |
| `evaluate_motion_mode_for_winner_candidates` (`:5205`) | `motion_mode_rd` |
| `process_compound_inter_mode` (`:2698`) | `av1_compound_type_rd` (compound_type.c — a sibling lane) |
| `fast_interp_search` (`:2886`) | `av1_interpolation_filter_search` |
| `simple_translation_pred_rd` (`:2270`) | `av1_enc_build_inter_predictor` + `motion_mode_rd`'s model-rd path |

`ref_mv_idx_to_search` (`:2357`) is blocked on `simple_translation_pred_rd`, so
call it 12.

### 2.2 Blocked on the encoder-side variance function table (3)

`cpi->ppi->fn_ptr[bs].vf` is filled inline inside libaom's
`av1_create_primary_compressor` and is not separately callable, so a shim
cannot populate it without transcribing that table. The three functions that
read it are:

- `get_sse` (`:868`) — **the inventory calls this "ported"; it is not.** The
  name collides with an unrelated method in `allintra_vis.rs`.
- `prune_zero_mv_with_sse` (`:2809`)
- `get_block_temp_var` (`:6074`) — also needs `av1_get_force_skip_low_temp_var`

### 2.3 Deferred by the roadmap: TPL (3)

INTER-ENCODE-ROADMAP §3's first envelope is `--lag-in-frames=0`, which removes
TPL entirely. These are real semantics, just not on the critical path:

- `get_block_level_tpl_stats` (`:2543`)
- `prune_modes_based_on_tpl_stats` (`:2598`)
- `calculate_cost_from_tpl_data` (`:5954`)

### 2.4 The two top-level drivers and their state (5)

- `av1_rd_pick_inter_mode` (`:6100`) — 500 lines calling essentially everything
  above. The inventory calls it "ported"; what exists is
  `aom_encode::inter_rd`, a deliberately reduced NEARESTMV/NEARMV/GLOBALMV
  skip-arm envelope, not this function.
- `av1_rd_pick_inter_mode_sb_seg_skip` (`:6611`)
- `init_inter_mode_search_state` (`:4493`) — the half this port does not hold
  (`intra_search_state`, `mode_threshold` from `cpi->rd.threshes` and
  `x->thresh_freq_fact`, `best_y_rdcost`). Its single-reference half IS ported
  as `SingleStates::init`.
- `update_search_state` (`:5146`) — copies `RD_STATS` / `MB_MODE_INFO` /
  `tx_type_map` wholesale; the port carries those in its own candidate type,
  so this is buffer plumbing rather than a decision.
- `store_coding_context` (`:1157`) — same, and likewise mis-marked "ported".

### 2.5 Out of scope: the port replaces the mechanism (1)

- `init_neighbor_pred_buf` (`:4259`) — hands three raw `uint8_t *` slices of one
  OBMC scratch allocation to `HandleInterModeArgs`, with a `CONVERT_TO_BYTEPTR`
  hbd variant. There is no pointer arithmetic to reproduce: the Rust port owns
  those buffers as slices.

### 2.6 Intra-in-inter-frame, and one partial (4)

- `init_mode_skip_mask` (`:4088`) — reads ~20 speed-feature / rate-control
  fields. Its callee `default_skip_mask` and both `disable_reference` variants
  ARE ported and gated; the driver is not. Mis-marked "ported".
- `search_intra_modes_in_interframe` (`:5747`), `skip_intra_modes_in_interframe`
  (`:5993`) — both mis-marked "ported".
- `estimate_ref_frame_costs` (`:995`) — the SINGLE-reference arm is ported
  (`inter_costs::ref_cost_single_last`); the compound arm is not.

## 3. PORTED on 2026-09-01 — 67 functions, 7 landings

All tier 1c unless noted (see §4). Every landing carries bite proofs in its
commit message.

| module | C functions |
|---|---|
| `rdopt_mv.rs` | `get_single_mode`, `check_repeat_ref_mv`, `get_this_mv`, `build_cur_mv`, `clamp_mv2`, `clamp_and_check_mv`, `clamp_mv_in_range`, `get_drl_cost`, `get_drl_refmv_count`, `is_single_newmv_valid`, `prune_ref_mv_idx_using_qindex`, `prune_ref_mv_idx_search`, `skip_nearest_near_mv_using_refmv_weight`, `conditional_skipintra`, `mask_set_bit`, `mask_check_bit`, `handle_newmv` (compound arm), `update_mode_start_end_index` (+ `av1_ref_frame_type`, `compound_ref0_mode`, `compound_ref1_mode`, `av1_drl_ctx`, and **tier 1** `av1_get_ref_mv` / `av1_get_ref_mv_from_stack` from encodemv.c) |
| `rdopt_skip.rs` | `disable_reference`, `disable_inter_references_except_altref`, `default_skip_mask`, `mask_says_skip`, `match_ref_frame_pair`, `ref_match_found_in_nb_blocks`, `find_ref_match_in_above_nbs`, `find_ref_match_in_left_nbs`, `match_ref_frame`, `compound_skip_using_neighbor_refs`, `update_best_single_mode`, `skip_compound_using_best_single_mode_ref`, `is_ref_frame_used_by_compound_ref`, `is_ref_frame_used_in_cache`, `fetch_picked_ref_frames_mask`, `find_top_ref`, `in_single_ref_cutoff`, `inter_modes_info_sort`, `compare_rd_idx_pair`, `compare_int64` |
| `rdopt_model.rs` | **tier 1** `av1_inter_mode_data_init`, `av1_inter_mode_data_fit`; tier 1c `get_est_rate_dist`, `inter_mode_data_push`, `inter_mode_data_block_idx` |
| `rdopt_single_state.rs` | `init_single_inter_mode_search_state` (single-ref half), `collect_single_states`, `analyze_single_states`, `compound_skip_get_candidates`, `compound_skip_by_single_states`, `skip_repeated_mv`, `init_comp_avg_est_rd`, `init_top_tx_no_split_rd_for_inter_modes`, `inter_modes_info_push`, `increase_motion_mode_rd`, `skip_interp_filter_search` |
| `rdopt_obmc.rs` | `calc_target_weighted_pred`, `calc_target_weighted_pred_above`, `calc_target_weighted_pred_left` (+ `foreach_overlappable_nb_above` / `_left`) |
| `rdopt_var_rd.rs` | `get_variance_stats`, `get_variance_stats_hbd`, `adjust_cost`, `adjust_rdcost`, `inter_mode_compatible_skip`, `ref_mv_idx_early_breakout` |
| `rdopt_gate.rs` | `inter_mode_search_order_independent_skip`, `prune_ref_frame` (mask half), `record_best_compound`, `init_mbmi`, `get_winner_mode_stats` |

**None of it is wired into the encoder yet.** These are the decision layer the
top-level driver (§2.4) will call.

## 4. Evidence tier — what "tier 1c" means and what backs it

`nm -g upstream/build/libaom.a` reports TEN exported symbols for the whole of
rdopt.c; every decision helper is `static`. So
`crates/aom-sys-ref/shim/rdopt_shim.c` compiles libaom's OWN rdopt.c into the
shim archive, with its ten exports renamed out of the way and built with
libaom's Release flags, and exposes flat wrappers around the statics. The
bodies under test are libaom's source, not a transcription of it — the same
technique, and the same justification, as the pre-existing `shim/cnn_cscalar.c`.

Call that **tier 1c**: real C source, compiled verbatim, versus tier 1's real
symbol out of the archive. The one gap is that it is a SECOND COMPILATION, and
that gap is closed by measurement, not assertion:
`rdopt_mv_diff::rdopt_shim_tu_agrees_with_archive` drives the shim TU's
`av1_block_error_c` and `av1_get_horver_correlation_full_c` against the
ARCHIVE's exported symbols and asserts bit equality. If the second compilation
ever stopped meaning the same thing, that test fails and every tier-1c claim
here is void.

Two entries are **tier 1 proper** (the shim `#undef`s the renames so the call
lands in the archive): `av1_inter_mode_data_init` and
`av1_inter_mode_data_fit`, plus the two encodemv.c accessors.

**One assertion is tier 4 and is labelled so in the test**:
`newmv_reduced_search_range` (`handle_newmv`'s `reduce_search_range` block,
`:1372-1403`) sits next to a call to `av1_single_motion_search` and cannot be
driven through the C without a full `AV1_COMP`; the test asserts its
invariants, not C agreement.

## 5. Gates

```
cargo test -p zenav1-aom-encode --test rdopt_mv_diff           # 15
cargo test -p zenav1-aom-encode --test rdopt_skip_diff         # 16
cargo test -p zenav1-aom-encode --test rdopt_model_diff        # 12
cargo test -p zenav1-aom-encode --test rdopt_single_state_diff # 11
cargo test -p zenav1-aom-encode --test rdopt_obmc_diff         #  1 (600 blocks)
cargo test -p zenav1-aom-encode --test rdopt_var_rd_diff       #  5
cargo test -p zenav1-aom-encode --test rdopt_gate_diff         #  4
```

## 6. C behaviours reproduced deliberately — do not "fix" these

Each was found by the differential and each fails it if corrected.

1. **`build_cur_mv` ASSIGNS `ret = get_this_mv(...)` per iteration** rather than
   accumulating, so a clamp failure on reference 0 is DISCARDED when reference
   1 is NEWMV.
2. **`GET_MV_RAWPEL` is `((x) + 3 + ((x) >= 0)) >> 3`**, not `>> 3`.
3. **`find_ref_match_in_*_nbs` returns 1 for an UNAVAILABLE edge** — "no
   evidence to prune on", not "a match was found".
4. **`av1_inter_mode_data_init` resets 7 of 14 fields**, leaving the five means
   and `a`/`b` at whatever the allocation held.
5. **`get_variance_stats`' scratch buffer has row stride `bw` with `bw + 2`
   columns**, so the halo columns alias neighbouring rows. A clean `bw + 2`
   stride disagrees on every block.
6. **`fit` computes `dx = sqrt(sse_sse_mean)` then uses `dx * dx`** — not the
   identity in binary64.
7. **C's `round()` is ties-AWAY-from-zero**; `f64::round_ties_even` is the
   wrong Rust spelling and differs only at an exact `.5`.
8. **`inter_mode_search_order_independent_skip`'s cache chain is not guarded on
   the cache being inter** — an INTRA cache reaches the single-reference arm.
9. **`x->mb_mode_cache` (pointer) and `x->use_mb_mode_cache` (flag) are
   separate inputs**; a stale non-null cache with the flag off still changes
   the verdict.
10. **`init_mbmi` writes `interintra_mode = II_DC_PRED - 1`**, i.e. -1 in an
    unsigned 1-byte enum, so it reads back as 255.
11. **`FLAG_SKIP_INTRA_LOWVAR` is `1 << 5`** (the enum is sparse), and
    **`AOM_TUNE_IQ`/`AOM_TUNE_SSIMULACRA2` are 10/11**, and **`RDCOST` rounds at
    shift 9**, and **`BLOCK_16X16` is index 6**. All four were wrong on the
    first attempt and all four were caught by the differential, not by reading.

## 7. C preconditions the port ASSERTS rather than reproduces

These are inputs on which libaom itself reads or writes out of bounds. The
encoder cannot produce them; the port asserts the contract and the harnesses
generate only reachable inputs.

- `av1_ref_frame_type` computes **-7** for `(LAST2, LAST)` and C then indexes
  `ref_mv_count[-7]`. Only 7 single + 12 bidirectional + 9 unidirectional pairs
  are addressable.
- `StateList::insert` has no bound check: a fifth candidate filed into one
  (direction, mode) writes past `single_state[dir][mode]`.
- A uniformly random `mode_context` indexes `refmv_mode_cost` out of bounds —
  `REFMV_CTX_MASK` is 4 bits while `REFMV_MODE_CONTEXTS` is 6.
- `calc_target_weighted_pred` needs an 8x8-or-larger block (so `mi_row`/`mi_col`
  are even; from an odd start the 4-wide pair fixup gives a negative
  `rel_mi_col` and C writes before `wsrc`) and an ALIGNED neighbour tiling (an
  unaligned one makes `rel + op_mi_size` exceed the block and C writes past
  `wsrc`).
- `av1_get_ref_mv_from_stack`'s single-reference fallback indexes
  `global_mvs[ref_frame_type]`, which has only `REF_FRAMES` entries — out of
  bounds for a compound row, and unreachable because the fallback is inside the
  single-reference arm.
