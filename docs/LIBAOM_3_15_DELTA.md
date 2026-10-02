# libaom v3.14.1 → v3.15.1 — what changed, what the port did about it

**Status 2026-10-02.** The pinned oracle (`upstream/` submodule) is **libaom v3.15.1**
(`44d0a577`, tag `v3.15.1`, 2026-09-21; v3.15.0 is `de4c1d1e`, 2026-08-25). It was
`03087864` (v3.14.1). Historical `benchmarks/*.meta` keep the old SHA on purpose — they
record what each measurement ran against.

This file started as a *prediction* (read from the diff) and is now the *record* (the oracle
was rebuilt and the whole workspace run against it). Where the prediction was wrong it says so.

Reproduce the delta:

```
git clone --bare https://aomedia.googlesource.com/aom ~/tmp/aom-scope && cd ~/tmp/aom-scope
git log --oneline v3.14.1..v3.15.1                       # 168 commits
git diff --stat v3.14.1 v3.15.1                          # 276 files, +56,379 / -6,460
git log --grep=STATS_CHANGED --oneline v3.14.1..v3.15.1  # 24: libaom's own "output moved" tag
```

`third_party/highway` (118 files, 36.6 k lines) and `wedge_masks_data.inc` (14.1 k) are
most of the line count and none of the risk. `av1/decoder` has **no changed files**.

## Measured result

| Run | Result |
|---|---|
| Workspace, default dispatch, v3.15.1 oracle, **before** any port change | 1552 / 1562 passed — **10 red** |
| Same, after the changes below | **1562 / 1562 passed** (8 slow) |
| Forced-scalar dispatch and the rest of the landing gate | recorded in the landing commit message, not here, so this file cannot go stale |

The 10 red, with what each one turned out to be:

| Red test | Real cause | Fix |
|---|---|---|
| `encoder_gate_speed8/9_textured_allintra` (**stills byte gates**) | libaom `137bcff61e` — `av1_nonrd_pick_intra_mode` caps `tx_size` to TX_16X16 for a flat block (`source_variance == 0`) at `base_qindex > 150` on the top/left frame edge. **Not tagged `STATS_CHANGED`, and the first draft of this file had filed it under "RTC intrabc" — wrong.** Flat 64x64, cq48/cq63, `--cpu-used` 8/9. | `nonrd_flat_edge_tx_cap` (`nonrd_pickmode.rs`) |
| `rdopt_var_rd_diff` × 4 | `get_variance_stats` moved to `aom_calc_variance_stat` with `pstride = bw + 2` — the port had reproduced v3.14.1's row-aliasing stride *on purpose*; v3.15 fixed it. Plus hbd rounding (`7a210280be`). | stride + `bd` parameter (`rdopt_var_rd.rs`) |
| `tx_mask_diff::pixel_diff_dist` | shim: `av1_pixel_diff_dist` clips on `x->pix_to_*_edge` now. Port: empty visible area returns `0`, not `UINT_MAX` | shim + `tx_search.rs` |
| `wiener_denoise_diff` | `20e8e3df1b`: error-diffusion error snaps to 0 below 1e-6 | `denoise.rs` |
| `compound_type_diff` × 2, `upsampled_pred_diff` | shims only: a NULL `ppi->fn_ptr[].vf`, and `aom_upsampled_pred_c` now uses `comp_pred` as its scratch | shims |

Also found while verifying, **not** caught by any failing test: `aom-sys-ref` declared
`av1_model_rd_curvfit` with its v3.14.1 five-argument shape. A C symbol carries no signature, so it
linked, and by stack-layout luck still passed — the C wrote `rate_dist_f[1]` over whatever followed
`rate_f`. Fixed (`ref_model_rd_curvfit`).

## Item by item

| # | Item | Outcome |
|---|---|---|
| 1 | Bump the oracle, fix the shims | **Done.** Gitlink `44d0a577`; pins in `build.rs`, `build.sh`, `BUILD_CONFIG.md`, `ci.yml`, `Cargo.toml`; `-DENABLE_APPS=1` (v3.15 redefined `ENABLE_EXAMPLES`). Five shims failed to compile (`dec`, `rdopt`, `compound_type`, `pass2`, `tpl_c`), four more compiled and failed at run time (`me`, `rd`, two `compound_type` entry points), and `interintra_shim.c` compiled but described machinery that no longer exists (comment only). |
| 2 | Re-run the byte-identity envelope, attribute every move | **Done** — table above. Exactly one stills-envelope move: the nonrd cap. |
| 3 | `get_variance_stats` | **Done**, 4 differentials green at bd {8,10,12}. |
| 4 | Noise model (`20e8e3df1b`) | **Done**: NaN-fail in `EquationSystem::solve`, `FlatBlockFinder::new -> Option`, `y_corr` guard, error snap. `noise_model_diff`, `flat_block_finder_diff`, `denoise_and_model_diff`, `wiener_denoise_diff` green. |
| 5 | `do_border_pad` | **Kernels done and gated; encode integration blocked, with a measured reason.** `border_pad.rs` ports `set_pixels_to_frame_edge`, `get_visible_dimensions`, `fill_residue_outside_frame`; `border_pad_diff` compares the whole `diff` plane against the REAL exported `av1_subtract_block` (5,760 cases; all 16 `TX_TYPE`s, bd 8/10/12, luma and chroma, edges either side of zero; all three fill arms asserted reached; a rounding mutation fails it). **Reach:** in C the flag is set for `mode == GOOD && deltaq_mode == DELTA_Q_OBJECTIVE && enable_tpl_model && aq == NO_AQ && …`; GOOD defaults are `deltaq_mode = DELTA_Q_OBJECTIVE` and `enable_tpl_model = 1` (`av1_cx_iface.c:305`), so a GOOD-usage frame with non-multiple-of-8 dimensions reaches it. `encode_key_frame` **refuses** every usage but ALLINTRA (`key_frame.rs:783`), so no encode this port produces can set it. Wiring `cpi->do_border_pad` into the search lands with GOOD usage. The flag-OFF behaviour changes (empty-visible `block_mse_q8`, `pixel_diff_stats`) were ported and are covered by `tx_mask_diff`. |
| 6 | Zenaom SCM trial (`577e360c9a`) | **Done and gated end to end.** `ratio_is_large_2` added; the detector's `allow_intrabc` survives a win. `zenaom_trial_decision_matches_c_on_the_ratio_arm_cells`: on 6 cells where that arm is the ONLY one that fires (11 measured, all agree) the port's trial and the v3.15.1 oracle's two-pass trial both flip `allow_screen_content_tools`, and on 2 declining cells both stay off; the one-pass `LibaomExact` stream never flips. Disabling the arm fails the first cell. **Measured limit:** C's two-pass trial PSNRs are not the port's to the last digit (256x128 patch80x64 amp25: port 24.7455 / 24.9816, C 24.7149 / 24.9699), so on cells within ~7% of the `diff/ratio = 2` line the decisions can differ (port no at 1.89, C yes) — 3 such cells found, deliberately not gated; cause of the PSNR gap not isolated. |
| 7 | Inter / 2-pass / RC rows | Below. |
| — | `increase_motion_mode_rd` (removed upstream) | **Replaced** by `scale_rdstats` / `increase_motion_mode_rdstats` / `increase_motion_mode_rate` / `get_global_mv_mode_bias`, 20,000 differential cases each. lc-dec only, but the port had a differential on the old function, so it follows. |
| — | `calc_correction_factor` / `qbpm_enumerator` / `find_qindex_by_rate_with_correction` | **Done**: new parameters; `qbpm_enumerator` constants `1_050_000` (first GOP) / `1_125_750` (was `1_200_000`). |

### Verified to need no port change (measured, not assumed)

- **`av1_optimize_txb`** (`f55fa5ee4e`, `f8f91ecd56`): `update_coeff_{general,simple,eob}` were rewritten to track
  `dist - dist0` directly. The existing trellis differential — port against the v3.15.1 oracle — stays green, so
  the rewrite is bit-exact on every input it draws. (The first draft flagged the `abs_qc == 1` rate derivation as
  needing a look; the differential is the look.)
- **`av1_interp_cubic_rate_dist`** (new RTCD function, SSE2): the curve-fit differential against the dispatched
  `av1_model_rd_curvfit` stays green with the corrected FFI declaration above.
- **Wedge masks** are `.rdata` offsets now (`8695b3ebac`): the compound / interintra differentials are green.
- **Decoder**: no source change; conformance and `real_bitstream` decode gates are green against the new oracle.
- `get_txb_ctx_*` table hoist, `av1_inv_txfm2d.h`, `get_log_range*`, the picklpf comment, the `fopen`/bounds checks,
  the Highway / AVX2 / AVX-512 / Arm SIMD work, the `aom_start_encode` size argument, `force_max_q` and
  `mode_ref_delta_enabled` plumbing. (`mode_ref_delta_enabled` defaults to 1 and leaves streams unchanged; the
  port's `header.rs:103` writer would only be wrong if something passed `false`, which nothing does.)

### §7 — inter / 2-pass / RC rows, each checked against the port

"No counterpart" = the C function is not in the port (grep over `crates/*/src`, 2026-10-02), so there is nothing
to diff. The standing non-goal in `CLAUDE.md` (inter-frame parity after ship) is why these stay unported.

| Upstream change | Port counterpart | Outcome |
|---|---|---|
| `tf_motion_search` mv limits (`28ea9e6b7a`) | `temporal_filter.rs` ports the **kernels only** (noise estimate, apply filter) | no counterpart |
| `encode_strategy` qindex for filtered frames (`2f67d6b97d`) | `ref_gop.rs` / `frame_source.rs` mention it; no function | no counterpart |
| TPL: VBR QP clip/reduce, `MV_COST_NONE`, prev-GOP ARF match + recon, full-pel `tpl_block_stats` MVs (`95cf99e94d`, `431ff8b214`, `5b0036d480`, `de45b6e71c`, `d50b000ac8`, `136038baa1`, `9acf475fe0`) | `tpl_model.rs` ports the **scalar arithmetic only** (entropy model, dependency cost, qstep→qindex); none of those helpers changed | no counterpart |
| `get_twopass_worst_quality`, `twopass_update_bpm_factor` (`883c1ab25b`, `4ff7ae6260`) | not ported; only the three helpers they call (done above) | helpers done |
| `cb_delta_rdmult_enabled`, `enable_delta_q`/`rdmult`, `disable_deltaq_for_intl_arfs` (`6c7f40b404`) | `encode_sb.rs`, `allintra_vis.rs` | For ALLINTRA/KEY `cb_delta_rdmult_enabled == delta_q_present_flag` — identical; the INTNL_ARF logic is GOOD two-pass only |
| Second-MV eval (`6f75fec9dc`), `masked_compound_type_rd` for `COMP_WEDGE` (`da3a5a6209`), speed-feature extensions (`prune_one_sided_comp`, `prune_inter_modes_based_on_tpl`, `skip_model_rd_uv`, `disable_extensive_joint_motion_search`, `prune_h_or_v_4part_using_sms_info`, `enable_comp_wedge_search_using_model_rd`) | those speed features and `av1_joint_motion_search` are **absent** from the port | no counterpart |
| `direct_partition_merging` `num_proj_ref` recompute (`15d987ce53`) | `partition_pick.rs` notes it is gated to `!frame_is_intra_only` | inter only |
| RTC screen IntraBC (`4a527cabbb` and 11 more), lc-dec (`72030e86dd`, `ea30b63756`, `c8db5b2007`, `608ad413b1`, `673071a141`, `2b9ffa6a12`) | `rt_use_intrabc`, `rt_prune_intrabc_nonrd`, `rt_intrabc_miss_mode`, `weighted_chroma_distortion`, `bias_gm_mode_rd_scale_pct` are absent | realtime / video only |

## Process lessons (kept because they cost time)

1. **Read the untagged commits.** `STATS_CHANGED` is libaom's own "output moved" tag; `137bcff61e` — the one that
   actually broke a stills gate — did not carry it. The prediction triaged by tag and by title; the *measurement*
   found it.
2. **A stale FFI signature links.** A C symbol has no signature, so a changed prototype in a directly-bound function
   (`av1_model_rd_curvfit`) produces no error and may even pass. Audit every directly-bound symbol on an oracle bump.
3. **`#include "….c"` shims are the fragile layer**, and a shim that "compiles" can still describe machinery that no
   longer exists (`interintra_shim.c`) or hand a dispatched kernel an unset function pointer (the `vf` NULLs).
4. **`-p <crate>` runs none of the integration tests** (`__internals` is off): `nextest -p zenav1-aom-encode -E …`
   reports "0 run, N skipped". Filter with `--workspace`.
5. **jj cannot record a submodule pointer change.** The `upstream/` gitlink commit is built with git plumbing
   (`git read-tree` + `update-index --cacheinfo 160000,…` + `commit-tree`, then `jj git import`).
