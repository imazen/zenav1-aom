# libaom v3.14.1 → v3.15.1 — what the port has to absorb

**Scope of this file.** The pinned oracle (`upstream/` submodule, `reference/BUILD_CONFIG.md`,
`crates/aom-sys-ref/build.rs::PINNED_SHA`) is libaom **v3.14.1** (`03087864`). Upstream is now at
**v3.15.1** (`07a36fde`, tag object; commit `44d0a577`, 2026-09-21). v3.15.0 is `bd16ba82`
(commit `de4c1d1e`, 2026-08-25). All three tags are present on the submodule's mirror
(`imazen/libaom-mirror`), so the bump is a gitlink change, not a re-point.

**Provenance.** Everything below is **SOURCE** (read from the `v3.14.1..v3.15.1` diff and the
port's current tree, 2026-10-01). Nothing here has been *measured* — no oracle was rebuilt at
v3.15.1, no gate was re-run. "Affects the port" means "the C changed on a path the port
transcribes"; whether a given cell's bytes move is for the differential to say.

Reproduce the delta:

```
git clone --bare https://aomedia.googlesource.com/aom ~/tmp/aom-scope
cd ~/tmp/aom-scope
git log --oneline v3.14.1..v3.15.1                       # 168 commits
git diff --stat v3.14.1 v3.15.1                          # 276 files, +56,379 / -6,460
git log --grep=STATS_CHANGED --oneline v3.14.1..v3.15.1  # 24 commits: libaom's own "encoder output moved" tag
```

Size, so the diff is read in proportion: `third_party/highway` is 118 files / 36.6 k changed lines
(vendored SIMD library — no port impact); `av1/common/wedge_masks_data.inc` is 14.1 k lines (one
generated table). Excluding those two, the real surface is ~5 k lines, mostly `av1/encoder`.

## Verdict

| Area | Port impact |
|---|---|
| **Decoder** (`av1/decoder`, `aom_dsp` entropy/recon) | **None.** `av1/decoder` has zero changed files. `av1/common` changes are refactors (see §5). The decoder port needs no source change; it needs the oracle bump + a conformance re-run. |
| **Stills / KEY-frame encoder** (the proven envelope, PARITY.md §A) | **Small.** Three items can move bytes (§1). None touches the `LibaomExact` ALLINTRA envelope as currently reachable: §1.1 is `Zenaom`-only (KB-66/67), §1.2 is gated off ALLINTRA by construction, §1.3 is the denoise path. All three need an oracle measurement before that sentence can be trusted. |
| **Inter / 2-pass / RC encoder** (partial in the port) | **Large.** ~15 output-changing commits (§2). |
| **Oracle harness** (`aom-sys-ref` shims, `reference/build.sh`) | **Will not compile as-is.** 8 `#include "…c"` shims + 2 API-shape breaks (§4). This is the first blocker for everything else. |

## 1. Stills / KEY-frame encode — can move bytes

### 1.1 Screen-content trial decision (`577e360c9a` STATS_CHANGED, `b9e4140065`, `1ae0d47fac`)
`screen_content_tools_determination` (`av1/encoder/encoder_utils.c`), the decision half of
`av1_determine_sc_tools_with_encoding`. **The port has this**: `scm_trial_determine`
(`aom-encode/src/key_frame.rs:2444-2660`), reachable **only** in `KeyFrameMode::Zenaom` — C reaches the
trial through the recode loop, which one-pass/no-lookahead encoding disables, so `LibaomExact` never runs it
(KB-66, KB-67 in `CLAUDE.md`). v3.15.1 changes the decision, and the port's copy encodes the v3.14.1 one:

| | v3.14.1 (what `key_frame.rs` does) | v3.15.1 |
|---|---|---|
| win condition | `psnr_diff > 0.9 \|\| (ratio >= 1e-4 && psnr_diff/ratio > 4)` (`key_frame.rs:2633`) | adds `ratio_is_large_2 = psnr_diff > 0.1 && palette_ratio >= 0.05 && psnr_diff/palette_ratio > 2` as a third OR arm |
| `allow_intrabc` on a win | `cpi->intrabc_used`, which is always 0 here, so the port sets `false` (`key_frame.rs:2647-2650`) | `allow_intrabc_orig_decision \|\| cpi->intrabc_used` — i.e. the detector's own decision survives |
| bookkeeping | `projected_size_pass[]` + `CONFIG_FPMT_TEST` block; `psnr[3]` | removed; `psnr[2]` (no output effect) |

Two concrete Rust edits (`win`, and `sct.allow_intrabc = false` → the pre-trial value), plus the doc blocks at
`key_frame.rs:523-531` / `:2444-2452` that quote the old thresholds. **What it moves:** only Zenaom decisions —
`aom-encode/tests/all/zenaom_scm_trial.rs` (decision agreement with C's own two-pass trial) and the
`refusal_census.rs` screen-shaped tiny cells will see the new arm; the `LibaomExact` byte gates should not.
Also `av1_set_screen_content_options` (`encoder.c:2449`, ported in `screen_detect.rs`): the `AOM_CONTENT_SCREEN`
arm now reads `mode == REALTIME && !rt_use_intrabc ? 0 : 1` — identical for every non-REALTIME usage.

### 1.2 `do_border_pad` — residual fill + visible-dimension SSE (`1c1c4abc43`, `f1e7f4e023`, `ec404abb2c`)
The history is enable/disable/enable churn (`f311099b0a`, `b973895c4c`, `998a3fcda4`, `ec404abb2c`);
**the final state at v3.15.1 is enabled**, in `av1_encode` (`encoder.c`):

```
do_border_pad = mode == GOOD && deltaq_mode == DELTA_Q_OBJECTIVE && enable_tpl_model
             && aq_mode == NO_AQ && !seg.enabled && !roi.enabled && !sb_qp_sweep
             && !use_ducky_encode && sharpness != 3
```

- **ALLINTRA (`mode == ALLINTRA`) never sets it** — the proven ALLINTRA envelope is unaffected by the flag.
- **GOOD-quality usage with frame dims not a multiple of 8 does.** Whether a *single-frame* GOOD
  encode actually reaches it depends on `enable_tpl_model` surviving config for a 1-frame / lag-0
  encode. **Not checked — measure on the oracle before deciding this is in or out of scope.**
- What it does when on: transform-block residual outside the real frame is overwritten (zero for
  IDTX, in-frame mean for the other 2-D types, 1-D variants per `htx_tab`/`vtx_tab`) in
  `fill_residue_outside_frame` (`encodemb.c`); distortion/SSE are clipped to the *actual* frame
  via the new `x->pix_to_bottom_edge/right_edge` (`set_pixels_to_frame_edge`, set from
  `av1_set_offsets_without_segment_id`); `search_tx_type` re-subtracts per candidate tx type.
  Signatures changed: `av1_subtract_block/txb/plane`, `intra_model_rd`, `compute_sse_plane`,
  `store_winner_mode_stats(cpi,…)`.
- **Also changes behavior with the flag OFF — these need an equivalence check, not an assumption:**
  `av1_pixel_diff_dist`, `pixel_diff_stats`, `pixel_dist` now call `get_visible_dimensions(…, clip_dims=true)`
  instead of `get_txb_dimensions`; for a tx block with zero visible rows/cols `block_mse_q8` is now
  `0` (was `UINT_MAX`) and `block_var`/`per_px_mean` are zeroed; `aom_sum_sse_2d_i16` is no longer
  called with a zero dimension. In-frame blocks should be identical; fully-outside blocks are the
  case to differential.
- Port: no `do_border_pad` / `pix_to_*_edge` / `get_visible_dimensions` anywhere (grep, 2026-10-01).
  Touch points: `aom-encode/src/{tx_search,rdopt_sse,inter_rd,intrabc_search,var_tx}.rs`,
  `aom-dsp/src/dist/mod.rs`, `aom-dsp/src/txb/`.

### 1.3 Film-grain noise estimation (`20e8e3df1b`, untagged but behavior-changing)
`aom_dsp/noise_model.c`: `equation_system_solve` now rejects NaN solutions; `aom_flat_block_finder_init`
propagates that failure; `y_corr[c-1]` is `0` when `average_strength <= 1e-6` (was a divide);
the error-diffusion `err` is snapped to `0` when `fabsf(err) < 1e-6f`. The snap is on the hot
denoise path, so it can move output for ordinary inputs, not only degenerate ones.
Port: `aom-encode/src/noise_model.rs` (has `equation_system_solve`, `y_corr`) and `denoise.rs`.

### 1.4 Bitstream writer — loop-filter deltas (`3712e6af6f`)
`encode_loopfilter` now nests the `mode_ref_delta_update` bit (and the delta lists) **inside**
`if (mode_ref_delta_enabled)`. Default `mode_ref_delta_enabled = 1` (new `extra_cfg` field,
control `AV1E_SET_MODE_REF_DELTA_ENABLED = 176`, documented "only used in loopfilter control unit
test"), so default streams are unchanged. Port: `aom-dsp/src/entropy/header.rs:103-105` writes the
`meaningful` bit unconditionally after the enabled bit, and every `key_frame.rs` call site passes
`mode_ref_delta_enabled: true`. **No divergence today**; the writer is wrong the moment anything
passes `false`. Fix is a 4-line nest plus a `false` cell.

## 2. Inter / 2-pass / RC — output-changing, port is partial here

`STATS_CHANGED` commits, with the port's nearest module (`crates/aom-encode/src/`). "absent" = no
symbol found by grep, i.e. there is nothing to diff, the feature itself is unported.

| Commit | Change | Port |
|---|---|---|
| `6c7f40b404` | no SB-level qp offsets for `INTNL_ARF_UPDATE`; new `cb_delta_rdmult_enabled`, `enable_delta_q/rdmult`, `disable_deltaq_for_intl_arfs`; `setup_delta_q` runs on `cb_delta_rdmult_enabled`, `x->delta_qindex` forced 0 when `!delta_q_present_flag` while `rdmult_delta_qindex` keeps the real value; `av1_init_plane_quantizers` derives `qindex_rd` from `rdmult_delta_qindex` | `encode_sb.rs`, `allintra_vis.rs`, `tpl_model.rs`. `cb_delta_rdmult` absent. For KEY/ALLINTRA `cb_delta_rdmult_enabled == delta_q_present_flag` (when `!disable_deltaq_for_intl_arfs`) so stills are unchanged. |
| `95cf99e94d`, `431ff8b214`, `5b0036d480` | TPL stage: lower/clip QP in VBR; `MV_COST_NONE` in TPL subpel | `tpl_model.rs` |
| `de45b6e71c`, `d50b000ac8`, `136038baa1` | TPL: previous-GOP ARF matching; store TPL recon of last ARF (`prev_gop_arf_tpl_recon`, new `alloc_y_plane_only`); fix TPL recon buffer | `tpl_model.rs` (has `prev_gop_arf`) |
| `9acf475fe0`, `883c1ab25b`, `4ff7ae6260` (untagged) | RC: MI-grid dims for TPL block stats; `worst_quality` decision (`pass2_strategy.c`); `bmp_factor` update in VBR | `ratectrl_*.rs`, `pass2_model.rs`; `bmp_factor` absent |
| `2f67d6b97d` | qindex used in filtered-frame (ARF TF) decision (`encode_strategy.c`) | `temporal_filter.rs` / `ref_gop.rs` |
| `28ea9e6b7a` | `mv_limits` for 64x64 blocks in temporal filter | `temporal_filter.rs` |
| `6f75fec9dc` | no unintended RD eval of the second MV (`motion_search_facade.c`) | `inter_me.rs`, `rdopt_mv.rs` |
| `da3a5a6209` | `masked_compound_type_rd()` also invoked for `COMP_WEDGE`; new sf `enable_comp_wedge_search_using_model_rd` (set at speed ≥1) | `compound_type.rs`; sf absent |
| `0f197b2ccf`, `452bb4c07a`, `508d804560`, `540738827f`, `323c765036` | speed-feature extensions to speed 0/1/2: one-sided-compound prune (this is the commit that also drops `setup_prune_ref_frame_mask`'s threshold from `selective_ref_frame >= 2` to `>= 1`, `encodeframe.c`; port has `selective_ref_frame` in `rdopt_gate.rs`), `prune_inter_modes_based_on_tpl` (now set in the base block, so speed 0), `skip_model_rd_uv` (speed ≥1; was a later block), `disable_extensive_joint_motion_search` (speed 0), `prune_h_or_v_4part_using_sms_info` (`boosted ? false : true`) | all four sf names **absent**. `prune_h_or_v_4part_using_sms_info` is `false` for boosted frames, i.e. KEY — no stills effect. |
| `15d987ce53` (untagged) | `warped_motion_update_num_proj_ref` helper; `direct_partition_merging` now recomputes `num_proj_ref` | `partition_pick.rs` |

### 2.1 `get_variance_stats` — the port deliberately reproduces a bug that v3.15 fixes (`0a70e28d36`, `af806c9c8c`, `7a210280be`)
This is the one place where the port's own comments say the opposite of what the new C does.
`crates/aom-encode/src/rdopt_var_rd.rs:36-52`: *"The scratch buffer's stride is `bw`, NOT `bw + 2`,
and that is deliberate … a 'corrected' `bw + 2` stride gives different `src_var` and `rec_var` …
the differential fails if the stride is widened."* That described v3.14.1 exactly. In v3.15 the
filter moved to `aom_dsp/variance.c::aom_calc_variance_stat_c` / `aom_highbd_calc_variance_stat_c`
(plus new AVX2 versions) and uses `pstride = bw + 2`: the row-aliasing is gone. Separately
`7a210280be` rounds both hbd variances by `2*(bd-8)`.

- Effect: `adjust_rdcost`/`adjust_cost` gates (`tune=IQ`/`SSIMULACRA2`) on **non-KF/GF/ARF** frames
  (`frame_is_kf_gf_arf` returns early). Stills KEY frames are unaffected; inter frames under the
  IQ tunes are.
- **The port must change from `bw` to `bw + 2` stride and add the hbd rounding, and the
  "deliberate" doc block must be rewritten**, not just the code. Do both in one commit.
- `get_variance_stats` is also the first place where `aom_dsp` gained new RTCD-dispatched
  functions whose AVX2 variant must equal `_c` — gate it against the *dispatched* oracle (PARITY.md
  KB-41 root #26 explains why `_c` is not enough).

### 2.2 `av1_interp_cubic_rate_dist` (`a7eed89f4f`, `b12f167ca2`)
`av1_model_rd_curvfit` now takes `double rate_dist_f[2]` and calls a new RTCD fn
`av1_interp_cubic_rate_dist` with an **SSE2** specialization (`x86/model_rd_sse2.c`). Feeds
`model_rd_with_curvfit` (`model_rd.h`). The C reference is two scalar `interp_cubic` calls; the
SSE2 path is the one a real encode runs on x86-64. Expected bit-identical (same op order, two
lanes, no FMA on x86; `-ffp-contract=off` pinned) but that is a **SOURCE** claim — add a
dispatched-vs-port differential (as `cnn_partition_nn_diff` does). Port: `interp_rd.rs`,
`curvfit_tables.rs`.

### 2.3 Restructured motion-mode bias (lc-dec only)
`rdopt.c` replaced `increase_motion_mode_rd` with `increase_motion_mode_rdstats` / `scale_rdstats` /
`get_global_mv_mode_bias` / `increase_motion_mode_rate` and moved `inter_mode_data_push` before the
scaling. It is gated by `bias_{warp,obmc,gm}_mode_rd_scale_pct`, which are only non-zero under
low-complexity-decode — so no effect unless lc-dec is in scope (§3).

## 3. Features the port does not have (decide scope; do not assume)

| Feature | Commits | Port |
|---|---|---|
| **Low-complexity decode mode** (VOD, speed 1-3, 608p-1080p, `sharpness != 3`) | `72030e86dd`, `ea30b63756`, `c8db5b2007`, `608ad413b1`, `673071a141`, `2b9ffa6a12`, `set_good_speed_features_lc_dec_*` | **absent entirely** (`enable_low_complexity_decode` not found). Video-only; PARITY.md already scopes video out. Includes new `weighted_chroma_distortion`, `bias_gm_mode_rd_scale_pct`, `dual_sgr_penalty_level` retune, `gm_erroradv_tr_level`. |
| **RTC screen-content IntraBC** | `4a527cabbb`, `6350573c69`, `bb172c89e7`, `7d579ce59f`, `591b71f06a`, `91f0010688`, `8ad2e72af7`, `2bd5813529`, `137bcff61e`, `10566b238b`, `d5c00b9159`, `7daca92834` | `rt_use_intrabc`, `rt_prune_intrabc_nonrd`, `rt_intrabc_miss_mode` **absent**. Realtime usage only (`av1_need_dv_costs`, `rd_pick_intrabc_mode_sb`, `av1_use_hash_me` now admit `rt_use_intrabc`). Out of scope unless REALTIME usage is. |
| **`force_max_q`** (`--force-max-q`, VBR) | `ee698f212e`, `360acb00d7` | The name appears only in a doc comment at `ratectrl_init.rs:238`; no field. VBR-only; trivial (`top_index = MAXQ`). |
| **`AV1E_SET_MODE_REF_DELTA_ENABLED`** | `3712e6af6f` | see §1.4 |
| API/robustness only (no output change) | bitwriter/`aom_start_encode` size arg (`1b5a433c0a`, `c213343c8d`), `eb53911fc8`, SVC param validation, `fopen` checks, `aom_img_flip`, `validate_img` matrix-coeffs source (`2ba2565bda`), aom_image padding fill (`a7ffc3343e`) | none — the port has its own API. Noted so nobody re-audits them. |

## 4. Oracle / harness work — the first blocker

Nothing in §1-§3 can be *measured* until the oracle builds at v3.15.1.

1. **Gitlink bump** to `07a36fde` (v3.15.1), update `.gitmodules` comment, `reference/BUILD_CONFIG.md`,
   `reference/build.sh` (`LIBAOM_TAG`), `crates/aom-sys-ref/build.rs::PINNED_SHA`, `CLAUDE.md` line 3
   ("≥ v3.14.1" stays true), `PARITY.md` line 123, and the "pinned … v3.14.1" strings across docs.
   The oracle build is cached by submodule SHA, so it will rebuild once.
2. **CMake**: `ENABLE_EXAMPLES` was *redefined* to examples only; `aomenc`/`aomdec` now come from the new
   `ENABLE_APPS` (default ON). `BUILD_CONFIG.md`'s flag list (`-DENABLE_EXAMPLES=1 -DENABLE_TOOLS=1`)
   should add `-DENABLE_APPS=1` explicitly rather than lean on the default. `_FORTIFY_SOURCE` is no
   longer force-disabled in Release (`ec cmake: don't disable _FORTIFY_SOURCE`) — a flag difference
   on the oracle TUs; harmless but record it in `BUILD_CONFIG.md`.
3. **Shims that `#include` a changed libaom `.c`** (these compile libaom's *static* functions
   and will break or silently change on any signature/body edit). Changed files in parentheses:
   - `rdopt_shim.c` → `rdopt.c` (430 lines: `get_variance_stats`→`aom_calc_variance_stat`, `get_sse`,
     `update_search_state(cpi,…)`, `store_winner_mode_stats(cpi,…)`, motion-mode bias)
   - `tpl_c_shim.c` → `tpl_model.c` (143) — it `#define`s `av1_subtract_block` to a shim symbol
     (`tpl_c_shim.c:115`); the new signature `(const MACROBLOCK*, …, plane, plane_bsize, blk_col, blk_row, tx_type, do_border_pad)`
     breaks that wrapper
   - `compound_type_shim.c` → `compound_type.c` (`compute_sse_plane` now takes `cpi`; `COMP_WEDGE` hook)
   - `nonrd_pick_shim.c` → `nonrd_pickmode.c` (235; RTC intrabc)
   - `pass2_shim.c` → `pass2_strategy.c` (136)
   - `fp_shim.c` → `firstpass.c` (24)
   - `reconinter_enc_shim.c` → `reconinter_enc.c` (28; `aom_upsampled_pred_c` now uses `comp_pred` as scratch)
   - `tf_static_shim.c` → `temporal_filter.c` (15)
   - unchanged upstream, so safe: `ratectrl_shim.c`, `vbp_static_shim.c`, `cnn_cscalar.c`.
4. **`interintra_shim.c`** — its comment ("`av1_init_wedge_masks()` guards opens with `memset(wedge_masks, 0, …)`
   … `masks[][]` entry reads back NULL") describes machinery that no longer exists. `wedge_masks` is now a
   `const uint32_t` table of **byte offsets** into a `static const` `wedge_mask_buf`, and
   `av1_init_wedge_masks()` is an empty function. Read masks through `av1_get_contiguous_soft_mask()`
   (now out-of-line in `reconinter.c`); `av1_wedge_params_lookup[].masks[s][i]` is an offset, not a pointer.
   Same for `compound_shim.c` / `inter_shim.c` callers (they only call the no-op init — fine).
5. Shims that call symbols whose **signatures** changed: `rd_shim.c` (`intra_model_rd`, `av1_pixel_diff_dist`,
   `av1_init_plane_quantizers` semantics), `me_shim.c` / `comp_pred_shim.c` / `rtcd_probe_shim.c`
   (`aom_upsampled_pred`), `txb_shim.c` (`av1_optimize_txb`, `get_txb_ctx_*` — now table-hoisted, same values),
   `wb_shim.c` (`encode_loopfilter`), `dec_shim.c` (`aom_start_encode` now takes a size).
6. Test-vector / data: `test/test-data.sha1` and `test_data_util.cmake` changed (svc-L2T1/L2T2 vectors
   updated, `293d25779b`) — encoder-side vectors; check whether `conformance/` consumes them.

## 5. Behavior-neutral changes (no port action; the differential is the proof)

- **Wedge/inter-intra masks precomputed** (`8695b3ebac`, `tools/gen_wedge_masks_data.py` →
  `wedge_masks_data.inc`). Same data, now `.rdata`. The port generates its own masks; add one gate that
  the port's table equals `wedge_mask_buf` byte-for-byte (cheap, and it converts "should be identical" to measured).
  `ii_weights1d`/`build_smooth_interintra_mask` are now `CONFIG_AV1_HIGHBITDEPTH`-only in the C (the lowbd path
  uses the precomputed buffer) — same numbers.
- **`av1_optimize_txb` speedups** (`f55fa5ee4e`, `f8f91ecd56`, `43f2b6a990`): `update_coeff_{general,simple,eob}`
  rewritten to track `dist - dist0` directly (`dqc*(dqc - 2*tqc) << 2*shift` when `qmatrix == NULL`) and to skip
  `get_dqv` when `abs_qc == 1`. Algebraically the same decision (the `dist0` term cancels in `rd` vs `rd_low`),
  and not tagged `STATS_CHANGED`, so libaom claims bit-exactness — but the `abs_qc == 1` branch of
  `update_coeff_simple` derives `rate_low` as `rate - base_cost[5]` instead of calling
  `get_two_coeff_cost_simple`. **Re-verify against `aom-dsp/src/txb/optimize.rs` + `trellis_cost.rs` with the
  existing trellis differential on the new oracle; do not assume.**
- `get_txb_ctx_*` table hoist (`9af67ac5c8`), `av1_inv_txfm2d.h` + `get_log_range*` helpers, `highbd_inv_txfm_*`
  shift fixes, `picklpf.c` comment-only 10/12-bit text, `debugmodes.c`/grain-table/`noise_model` fopen checks.
- **SIMD only** (oracle dispatch, not semantics): Highway AVX2/AVX512 `convolve_2d_sr`, `convolve_x/y`,
  `warp_affine`; AVX2 `dist_wtd_convolve_2d`, `variance` subpel bilinear, `variance_stats`, `pixel_proj_error`;
  Arm convolve/variance; `sum_squares_sve`. Two are *fixes of C-vs-SIMD mismatches* —
  `92094f8c5f` "Convolve 2d: fix mismatch in hwy" and `34c63b10ea` "ASAN overflow in AVX512 convolve_2d_sr" —
  worth reading when a differential against dispatched C disagrees after the bump. `CONFIG_HIGHWAY` is not
  in the oracle's config today (`BUILD_CONFIG.md`); the default did not change in `cmake/aom_config_defaults.cmake`.

## 6. Suggested order

1. **Bump the oracle + fix the 8 shims (§4).** Without it nothing else is measurable. Land the shim
   fixes before the gitlink so `main` is never red.
2. Re-run the proven byte-identity envelope (`just gate-encode`, `encoder_gate_*`, `content_family_census`)
   against v3.15.1. Anything that moves is, by the table above, one of §1.1/§1.2/§1.3 — attribute before fixing.
3. §2.1 `get_variance_stats` (smallest, fully specified, and the doc currently asserts the opposite).
4. §1.3 noise model (stills scope, three small hunks).
5. §1.2 `do_border_pad` — after measuring whether a 1-frame GOOD encode reaches it.
6. §1.1 — two edits in `scm_trial_determine` plus its doc blocks; re-baseline `zenaom_scm_trial`.
7. §2 inter/RC rows, in `PARITY.md` family order.

**Not verified here, and worth a line in the next session's first measurement:** (a) whether a single-frame
GOOD encode sets `do_border_pad`; (b) that `av1_interp_cubic_rate_dist_sse2` equals the scalar pair;
(c) the `update_coeff_simple` `abs_qc == 1` rate derivation; (d) which of the 24 `STATS_CHANGED` commits
actually move a default `aomenc` output on the existing census corpora.
