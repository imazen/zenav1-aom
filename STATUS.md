> **Read first:** `docs/CYCLE_LEDGER_2026-09-08_11.md` (what the last cycle did and left open) and `docs/ITERATION_PLAYBOOK.md` (how to iterate). This file is the per-landing narrative, newest first, ~360 KB — grep it for a KB number or a benchmark name rather than reading it top to bottom. Landings before 2026-09-08 are in `docs/archive/STATUS_2026-07-14_to_2026-09-04.md` (moved 2026-09-25; nothing rewritten).

## Production prep: publishable, linted, documented, decluttered (2026-09-25, `maint/prod-prep`)

Five landings on one branch, each behind the full landing gate in a clean worktree:

- **Publish metadata + policy** (`a647137`): readme/keywords/categories + docs.rs block
  on the four published crates; `cargo publish --dry-run -p zenav1-aom-dsp` reaches
  Uploading; tarballs 5.4 / 1.0 / 6.3 MB + the facade. `deny.toml` measured over the
  93-package graph (all permissive; our AGPL branch allowed per published crate only;
  `cc`/`cmake` banned from the published path); `cargo audit` clean. CHANGELOG
  rewritten for the cycle. `just deny` in the gate + a CI job.
- **Docs** (`f8a7bd8`): the 130 undocumented items on the encode/decode consumer
  surfaces documented; `#![warn(missing_docs)]` on both (default build);
  rustdoc warnings on the published crates 36/11/54 -> 0; `just doc-check`
  (`-D warnings`) in the gate + a CI job.
- **Clippy** (`18e0bb7`): 866 -> 0 with `-D warnings` over every target. Per-crate
  policy allows name the lints that fight a line-for-line C port (each with its
  reason); `--fix` for the machine-applicable rest; ~70 hand fixes; dead code deleted
  or `cfg(test)`. Both nextest passes 1548/1548 after the rewrites — byte identity
  held. `just clippy` in the gate + a CI job.
- **Declutter** (`0230285`): STATUS.md 408 -> 67 KB (older cycles archived verbatim),
  CONTRIBUTING.md, README developer section = the gate, `handoff/` and
  `coverage-audit/` under `docs/`, the integration review archived as a dated record,
  docs index refreshed.
- **dsp internals gate** (this landing): `zenav1-aom-dsp` gains the same default-off
  `__internals` feature as encode/decode. `intra::edge`, `restore::wiener` and the
  transform's 1-D primitives (`cospi`, `fdct`, `txfm1d_gen`, `special`,
  `inv_txfm1d_gen` + their root re-exports) are `pub(crate)` without it — measured:
  no consumer crate reaches any of them; the differentials do. Default surface 62 ->
  56 modules, 596 -> 535 free functions (649 at the start of the day). The
  inter/convolve kernel families stay public on purpose: `feat/experimental-video`
  is routing the decoder through them now. `tests/all` requires the feature and
  `tests/internals_feature_guard.rs` fails loudly without it (the KB-42 pattern).

Not done, stated: the remaining ~165 dsp free functions used only by the crate's own
tests sit inside modules that consumers also use, so they need per-item work or
in-crate test moves — deferred until after the first crates.io publish fixes the
contract. `rust-version` is not set (untested MSRV; CI runs stable 1.98).

## `experimental-video` step 4 close-out: the inter conformance scope measured + pinned (2026-09-25, branch `feat/experimental-video`)

Step 4's definition of done included the conformance corpus's INTER scope
under the feature. Fetched `python3 xtask/conformance.py --fetch --scope
inter` — the 5 inter vectors (`av1-1-b8-05-mv`, `-06-mfmv`,
`-22-svc-L{1,2}T{1,2}`) — and ran them feature-on. Per-vector tally,
`inter_conformance_vectors_pinned_outcomes` (`crates/aom-decode/tests/all/
conformance_inter.rs`):

| bucket | count | vectors |
|---|---|---|
| byte-exact vs golden MD5 | 0 | — |
| refused by name | 5 | all |
| diverged | 0 | — |

No divergence: every vector stops at a named envelope boundary, never
corrupts. The two reachable inter-tool boundaries (both now `docs/
COVERAGE_QUEUE.md` T3 rows):

- `05-mv`, `06-mfmv` → `"inter: non-identity global motion not supported in
  this decode envelope"` (`crates/aom-decode/src/lib.rs` ~:3660,
  `inter.gm_wmtype[ref] != 0`). A reference carrying a ROTZOOM/AFFINE global
  model needs the real global-MV base + `is_global_mv_block` gating.
- `22-svc-L{1,2}T{1,2}` → `"partial tile group (multiple tile groups per frame
  not supported)"` (`crates/aom-decode/src/frame.rs` ~:285). SVC frames split
  tiles across >1 tile-group OBU (`--num-tile-groups>1`); the per-tile-group
  decode loop is a later step.

**The gate.** `conformance_inter.rs` pins each fetched vector's outcome —
`Outcome::ByteExact` (decode + byte-compare vs `aom_codec_av1_dx` per frame AND
vs the shipped golden `.ivf.md5`) or `Outcome::Refused(name-substring)` — so
both directions are loud: a vector that un-refuses when a feature lands forces
the pin to flip to `ByteExact`, and a byte-exact vector that regresses fails.
A byte DIVERGENCE or a PANIC is never a pinned outcome — always a hard fail.
It **skips-by-name** when no inter vector is fetched (CI provisions only
`--scope intra`, so the gate is a local/dev-time pin until the inter scope is
provisioned) and feature-off (the inter envelope is an `experimental-video`
contract; feature-off these streams are not a supported surface).

**Warp, stated precisely** (correcting the loose "warp prediction refused"):
local WARPED_CAUSAL (`motion_mode = 2`) is implemented, not refused — single
ref, unscaled, `num_proj_ref >= 1 && allow_warped_motion` per
`motion_mode_ceiling`; compound is never WARPED_CAUSAL (matches C
`blockd.h:1487`). The refused warp case is **global** warped motion — the
non-identity global-motion model refusal above. Full mechanism in
`docs/HANDOFF-EXPERIMENTAL-VIDEO.md` "Which warp refuses".

## `experimental-video` step 4b: masked compound (wedge / diff-weighted) routed through the two-buffer mask blend (2026-09-25, branch `feat/experimental-video`)

Per `docs/HANDOFF-EXPERIMENTAL-VIDEO.md` step 4, second landing — the masked
arm (`comp_group_idx = 1`). Every kernel the C masked path needs already
existed and was byte-locked (`interintra::wedge_mask_signed` for the wedge
codebook; `compound::build_compound_diffwtd_mask_{,_d16,_highbd}` +
`lowbd/highbd_blend_a64_d16_mask` for the blend), so this step was routing, not
a new kernel — matching the step-7 stop rule, none was missing. With
`experimental-video` ON a frame coding masked-compound blocks decodes
**byte-exact** against the pinned C oracle; OFF keeps the named refusal
`compound` (the group-agnostic gate that fires before the masked read).

**The routing.** `read_compound_type_info`'s masked outputs
(`comp_type`/`wedge_index`/`wedge_sign`/`mask_type`) — previously parsed for
stream sync and discarded — now feed a `MaskedCompound` descriptor. The decoder
allocates one luma-resolution `seg_mask` scratch per compound block.
`build_masked_compound_inter_predictor` (the port of
`av1_make_masked_inter_predictor` + `build_masked_compound_no_round`,
`reconinter.c:629`/`:602`) convolves each ref into its OWN `d16` buffer via the
step-4a-extracted `convolve_one_compound_ref` (both `do_average = 0`), then:

- `COMPOUND_WEDGE` — fetches the luma-resolution wedge codebook mask
  (`wedge_mask_signed(luma_bsize, index, sign)`); chroma refetches the same
  codebook entry.
- `COMPOUND_DIFFWTD` — builds `seg_mask` from the two luma d16 intermediates
  **only on luma** (`!inter_pred_params->conv_params.plane`,
  `reconinter.c:655`); the chroma call reuses that same luma-resolution mask.

Both then blend `lowbd_blend_a64_d16_mask`/`highbd_blend_a64_d16_mask` with
`mask_stride = block_size_wide[mi->bsize]` (luma width) and the plane's
`subw`/`subh` — `d16_mask_at` subsamples the luma mask for chroma. The group-0
shared-`dst16` + `do_average` combine is unchanged.

**Byte-gates.** `masked_compound_decode_envelope` — the conformant
`masked-compound.obu` (wedge + diffwtd blocks over an occlusion boundary)
decodes all 4 shown frames byte-identically to `aom_codec_av1_dx` under the
feature, and refuses `compound` by name without it. The new
`inter_pred_diff::masked_compound_facade_matches_c` drives
`build_masked_compound_inter_predictor` vs a real-C masked-assembly shim
(`ref_masked_compound_inter_predictor` /
`ref_highbd_masked_compound_inter_predictor` — two `inter_predictor` CONV_BUF
gathers + `av1_get_compound_type_mask`/`av1_build_compound_diffwtd_mask_d16` +
the `_c` d16 blend) across wedge index/sign and both diffwtd `mask_type`s,
luma + 4:2:0 chroma, 7 wedge-capable bsizes, bd 8/10/12, 3 filter pairs and 4
MV pairs — 1,260 blocks, 2,520 plane calls, byte-exact. The `aom-decode`
`unsupported_refusals` masked pin flipped: ON `masked-compound` decodes (4
shown frames), OFF still refuses `compound`. KB-42-clean: both new gates are
integration targets, none `--lib`.

## `experimental-video` step 4a: compound group-0 (average / dist-weighted) routed through the two-ref predictor (2026-09-25, branch `feat/experimental-video`)

Per `docs/HANDOFF-EXPERIMENTAL-VIDEO.md` step 4 (compound prediction), first
landing — the dist-weighted convolve routing. With `experimental-video` ON a
frame coding compound-reference blocks decodes **byte-exact** against the
pinned C oracle; OFF keeps the named refusal `compound references unsupported`.
Group-1 masked compound (wedge / diff-weighted, `comp_group_idx = 1`) stays
refused by name — `inter: masked compound (wedge/diffwtd) not yet supported in
this decode envelope` — until step 4b lands the mask builders; its syntax is
still consumed for stream sync before the refusal.

**The plumbing.** `InterCdfs` gained the eight compound tables C's
`FRAME_CONTEXT` carries (`comp_ref_type`, `uni_comp_ref`, `comp_ref`,
`comp_bwdref`, `inter_compound_mode`, `compound_idx`, `comp_group_idx`,
`compound_type`) — defaults generated by `xtask/gen_default_cdfs.py` and
byte-locked against the compiled C in `default_cdfs_diff.rs` (9 tables).
`ref_frame_cdfs` now assembles compound slots 1..9 alongside the single-ref
10..15, each at its pred context off `collect_neighbors_ref_counts` +
`get_comp_reference_type_context`, and the decoder copies the adapted rows back
per block so compound adaptation persists. `reset_cdf_counters` resets all
eight (the previous gap would have drifted a `primary_ref_frame` inheritance).

**The MV scan.** `find_inter_mv_refs` takes the reference PAIR `rf: [i32; 2]`
and, for a compound block, runs `setup_ref_mv_list`'s compound arm —
`add_ref_mv_candidate`/`add_tpl_ref_mv` push both candidate orders, and
`process_compound_ref_mv_candidate` builds `StackEntry` pairs (`this_mv` for
ref0 in `stack`, `comp_mv` for ref1 in `comp_stack`) — byte-locked by
`dv_ref_diff::find_compound_mv_refs_matches_c` (1,800 cases: 21 legal ref
pairs × grids × precision flags vs the real `av1_find_mv_refs`).

**The decode.** `decode_block_inter` reads the compound ref pair, the compound
mode (`read_inter_compound_mode` on `mode_context_analyzer`), the DRL index,
then resolves both MVs per `assign_mv` — nearest/near off the pair stack,
new-MV deltas via `read_mv`, `GLOBAL_GLOBALMV` off the two `global_mv` slots —
then `read_compound_type_info` (group/index/type, consuming the wedge symbols
for sync), then the per-direction interp filters. Both refs bind + validate;
each gets its own UMV border clamp (luma and chroma) and `ScaleFactors`.
`build_compound_inter_predictor` (`aom-dsp/src/inter/mod.rs`) runs the two-ref
`convolve_2d_facade` combine: `compound_rounds` (`(3,7)` — COMPOUND_ROUND1_BITS,
not the single-ref round_1 — held even at bd12), a shared `dst16` accumulator,
`do_average` flipping on the second ref, dist-wtd offsets from
`dist_wtd_comp_weight_assign` fed by the bound refs' order hints
(`InterFrameCfg::ref_order_hints`). Scaled refs route through
`convolve_2d_scale`'s compound arm per ref. Both MVs + `compound_idx` /
`comp_group_idx` stamp into `DvNbr` (so the next block's compound contexts see
them) and `stamp_frame_mvs` writes both ref MVs. Every port fn is byte-locked
by `inter_pred_diff::compound_facade_matches_c` (504 cases: 3 bit depths × 7
block shapes × 6 MV pairs × 4 blends vs the real `inter_predictor`/
`highbd_inter_predictor` two-ref loop, scalar `_c` dispatch pinned to dodge the
AVX2 oracle's alignment fault).

**Coverage.** New gate `aom-bench/tests/all/compound_decode_envelope.rs`:
`compound-refs.obu` (4-frame 64×64 stream coding group-0 compound blocks)
decodes **byte-exact vs `aom_codec_av1_dx`** — 4/4 shown frames — under the
feature, and refuses `compound` by name without it. New fixture
`masked-compound.obu` (wedge-compound aomenc stream, provenance in
`unsupported_refusals.rs`) pins the group-1 named refusal under the feature —
the step-4a boundary step 4b flips. The `aom-decode`
`unsupported_refusals` pins now split compound: OFF both streams refuse
`compound`; ON `compound-refs` decodes while `masked-compound` refuses
`masked compound` by name. KB-42-clean: every differential above is an
integration target, none is `--lib`.


## `experimental-video` step 3: scaled references routed through `convolve_2d_scale` (2026-09-25, branch `feat/experimental-video`)

Per `docs/HANDOFF-EXPERIMENTAL-VIDEO.md` step 3: with `experimental-video` ON
an inter frame whose bound ref has different luma crop dims decodes
**byte-exact** against the pinned C oracle in both scale directions; OFF keeps
the named refusal `frame_size_override (frame != sequence max dims)`.

**The routing.** `aom-decode/src/frame.rs` lifts the `frame_size_override`
gate under the feature (keeping the C bounds checks — dims <= sequence max,
nonzero), seeds `frame_size_with_refs` from the bound refs' crop dims so
`found_ref` resolution runs like C's `setup_frame_size_with_refs`, computes
`InterFrameCfg::ref_sf` (one `ScaleFactors` per bound ref, C's
`av1_setup_scale_factors_for_frame`) and rejects the frame as malformed when
no bound ref is a valid size (`valid_ref_frame_size`). `build_tile_cfg`'s
`mi_rows` now uses the frame's actual height instead of the sequence maximum —
the old seq-max value over-allocated the MI grid for size-override frames.
`aom_dsp::inter::build_inter_predictor` takes the raw (coded) MV alongside the
UMV-clamped MV plus `&ScaleFactors`; scaled refs dispatch to
`scaled_inter_predictor`, which mirrors `dec_calc_subpel_params`'s scaled arm
(q10 sub-pel positions, per-sample `x_step_qn`/`y_step_qn`, `SCALE_EXTRA_OFF`
border, `AOM_LEFT_TOP_MARGIN_SCALED` clamps, always-on scaled border pad via
`build_mc_border`/`build_mc_border_highbd`) and then calls the already-locked
`convolve::scaled::{convolve_2d_scale, highbd_convolve_2d_scale}` kernels.
Warped prediction is disabled on scaled refs in both places C disables it:
`motion_mode_ceiling` (symbol read — WARPED_CAUSAL never offered) and the
`warp_luma`/`warp_chroma` apply-sites (`av1_allow_warp`'s `is_scaled` recheck),
so scaled blocks take the translational path. Compound refs still refuse by
name in both build states.

**Coverage.** New gate `aom-bench/tests/all/scaled_ref_decode_envelope.rs`:
three committed OBU fixtures (generated with `aomenc --resize-mode=1`,
provenance in the file doc), each C-decoded through
`ref_decode_av1_stream_frame_opt` against PINNED per-frame dims — the shim
panics on a dims mismatch, so a port geometry error cannot hide behind
`is_scaled`. Feature ON: `scaled-ref-down` (128² KEY + five 64² inters
referencing the 2×-larger ref), `scaled-ref-up` (64² KEY + four 128² inters
referencing the 2×-smaller ref) and `frame-size-override` (resized KEY) all
**byte-exact vs `aom_codec_av1_dx`** — 12/12 shown frames. Feature OFF: each
refuses `frame_size_override` by name. The aom-decode
`unsupported_refusals::scaled_reference_frame_refused_by_name` pin flips to
decode-success under the feature (1/6/5 shown frames), and
`inter_pred_diff.rs` gained scaled smoke coverage at bd8/10/12 in both
directions. Larger exploratory streams (512²↔256²) were byte-exact end-to-end
during development.


## `experimental-video` step 2: highbd sub-pel MC routed through the u16 convolve kernels (2026-09-25, branch `feat/experimental-video`)

Per `docs/HANDOFF-EXPERIMENTAL-VIDEO.md` step 2: with `experimental-video` ON a
bd10/bd12 inter frame codes nonzero MVs end-to-end and decodes **byte-exact**
against the pinned C oracle; OFF keeps the named refusal
`inter: sub/nonzero-pel MC above bd8 not yet supported` (the
`#[cfg(not(feature = "experimental-video"))]` guard is compiled out only on the
feature leg — `aom-decode/src/lib.rs`).

**The routing.** `aom_dsp::inter::build_inter_predictor` gains a `bd` parameter
(the port's `is_cur_buf_hbd(xd)`) and, above bd8, gathers the edge-replicated
reference through the new `build_mc_border_highbd` (u16 twin of
`build_mc_border` — the `is_highbd` instantiation of C's shared
`BUILD_MC_BORDER` body) into a u16 scratch, then dispatches through the new
`highbd_inter_predictor` facade to the already-locked
`convolve::highbd::highbd_convolve_{x,y,2d}_sr` kernels — never touching the u8
scratch. The rounding pair is the new `single_ref_rounds(bd)`, the
`get_conv_params_no_round(.., is_compound = 0)` arm `(3,11)` @ bd8/10, `(5,9)` @
bd12 — the same derivation `compound_convolve_diff.rs` and
`warp_highbd_diff.rs` already verify against C. All seven decoder call sites
(luma, sub8x8 chroma, per-plane chroma, and the four OBMC neighbour strips)
pass `cfg.bd`, so every translational MC path is depth-correct, and the two
local-warp sites route `WARPED_CAUSAL` blocks through the existing C-diffed
`highbd_warp_affine` at bd>8 — leaving the bd8 `warp_affine` on u16 data would
have been a silent wrong-pixel hole under the widened envelope. Compound,
scaled-reference, and non-identity global motion still refuse by name in both
build states.

**Coverage.** `aom-bench/tests/all/highbd_inter_decode_envelope.rs` part 3 now
asserts the per-state contract cell-by-cell: feature ON, every bd10/bd12
nonzero-MV cell — the integer-pel sweep (4 chroma shapes) plus new half-pel
sweeps forcing x-only, y-only and 2-D sub-pel phases — is **byte-exact vs
`aom_codec_av1_dx`** (30/30 measured); feature OFF each refuses with the named
`unsupported feature` string. A temporary probe confirmed the sweep drives all
three kernel dispatch arms at both depths (subpel phases 4/8/12, EIGHTTAP_SMOOTH
and subsampled-chroma blocks included). The committed
`unsupported_refusals::highbd_nonzero_mv_refused_by_name` pin flips to
decode-success under the feature via `aom_decode::EXPERIMENTAL_VIDEO`.

Integration targets: `aom-decode::all` (unsupported_refusals),
`aom-dsp::all` (inter_pred_diff signature), `aom-bench::all`
(highbd_inter_decode_envelope), `test-next-video` leg, api-doc snapshot
(`build_inter_predictor` + `bd`, three new fns).

Gates at landing: `just gate-landing` — test-next **1551/1551**,
test-next-scalar **1551/1551**, test-next-video (feature ON) **1551/1551**
(identical counts = byte-inert on the default envelope), plus upstream-check,
fmt-check, ci-yaml-check, census-gate, test-whereat, api-doc-check all green.

## `experimental-video` step 1: the default-off feature is scaffolded and conformant-but-unsupported tools refuse by name (2026-09-25, branch `feat/experimental-video`)

Per `docs/HANDOFF-EXPERIMENTAL-VIDEO.md`: `zenav1-aom-decode` gains
`experimental-video` (default OFF, forwarded by the `zenav1-aom` facade with an
implied `decode`). Nothing is routed yet — that is the point of the scaffold:
the flag compiles the same code either way and the default envelope is
byte-inert under it, which the third workspace leg now proves every push.

**The typed `unsupported` channel.** `TileKf`/`KfTileDecode` carry
`unsupported: Option<&'static str>` alongside `corrupt`, drained in `frame.rs`
as `DecodeError::UnsupportedFeature`. `mark_corrupt`/`mark_unsupported` are
first-reason-wins across both channels and `is_corrupt` tests both, so the
re-entrant guards unwind identically — only the surfaced category differs.
Every conformant-but-out-of-envelope inter guard moved off `mark_corrupt`:
the three named handoff refusals (compound references, highbd sub/nonzero-pel
MC, `frame_size_override` — the last already `UnsupportedFeature` at
`frame.rs`) plus segmentation / skip_mode / delta-q / tx_mode ONLY_4X4 /
non-identity global motion / BILINEAR interp filter / non-uniform var-tx.
Genuinely malformed input (out-of-range or inconsistent ref coding,
unavailable reference slots, invalid chroma/partition sizes) stays
`Malformed` — the compound guard was split so its folded range-check no
longer mislabels bad streams as unsupported or vice versa.

**Pins, not just code.** `tests/data/inter/` commits three sub-2 KB OBU
fixtures verified conformant by the pinned `aomdec` (a 4-frame compound
stream — blend content wins compound RD — a `--resize-mode`
frame-size-override stream, and a bd10 nonzero-MV stream; generation recipes
in the test header). `tests/all/unsupported_refusals.rs` asserts each is
refused BY NAME (`UnsupportedFeature` containing the family name), panics on
`Ok` with an instruction to flip the arm when its step lands, and runs in
both build states. `aom_decode::EXPERIMENTAL_VIDEO` is the pub const
harnesses branch on — a test crate's own `cfg!(feature)` cannot see the
workspace-unified flag, so the crate self-reports (recorded in the API
snapshot).

**Gate shape changed.** `just test-next-video` is the feature-on workspace
nextest recipe (`--features zenav1-aom-decode/experimental-video`, the
unifying invocation), now a `gate-landing` step, and a third linux
differential leg (`test-linux-experimental-video`) in CI.

Gates at landing: `just gate-landing` — test-next **1551/1551**,
test-next-scalar **1551/1551**, test-next-video (feature ON) **1551/1551** —
identical counts across legs = the byte-inert contract — plus fmt-check,
ci-yaml-check, census-gate, test-whereat, api-doc-check, upstream pristine
`03087864`.

## Merged to main; branches pruned; the tree is rustfmt-clean and gated (2026-09-25)

`perf/gate3-txfm-i16-batch` fast-forwarded `origin/main` `1434bc3` -> `75b5abc` (171
commits) after the full landing gate; the push started the first CI run this code has
had. Every merged branch is deleted locally and on origin (the five `integrate/*`,
`maint/*`, `perf/*` and `handoff/*` refs plus `fix/issue-8-highbd-inter-envelope`,
superseded by `74e45c5`); only `main` and the archival `preserve/*` remain. Then one
whole-tree `cargo fmt --all` commit (333 files; `.git-blame-ignore-revs` lists it),
with `just fmt` / `fmt-check`, `fmt-check` as the second step of `gate-landing`, and a
`rustfmt --check` CI job that fails in seconds ahead of the 20-minute legs.

**First CI verdict on the merged code, part 2 — the aarch64 legs (KB-70):** 10 red
differentials under default dispatch, 3 under the pin, every whole-frame byte gate
against ARM `aomenc` green on both. Read off the logs (no ARM box here): three tests
pinned x86-only kernel properties (AVX2 saturation, SSE4 side-effect writes, the i16 tap
bound) and are now x86-64-only; three drew into `_c`'s signed-overflow UB (`i32::MIN`,
`±2^27`) where aarch64 clang differs from the x86 shape the port mirrors — those lanes
are x86-64-only, KB-ARM-FLOAT root #3's resolution; the lowbd fuzz compared a pinned
scalar port against the NEON oracle — now pin-aware. The five encode-level `_c`-chain
tests (libaom's NEON quantize is not bit-exact with its own `_c`; the port mirrors NEON)
assert on the forced-scalar ARM leg only until NEON-chain oracle shims exist — OPEN,
queued T3. **Verified: run 36107492113 on `4e760d4` is green on every job** — both x86-64
differential legs, both aarch64 legs, the four portability targets, rustfmt and the
public-API check — the first fully green CI run since the branch began on 2026-08-05.

**First CI verdict on the merged code:** the `portability i686` leg failed at the
build step — `aom-dsp/src/sse_neon.rs` re-exported `imp::*` from a module that exists
only on x86-64 (real intrinsics) and aarch64 (NEON twins); on 32-bit x86 there was
nothing to re-export (E0432). Every consumer was already arch-gated, so the fix is
gating the module declaration the same way. Reproduced and verified locally with
`cargo check --target i686-unknown-linux-gnu` on the four published crates and the
decode-only stack (`rustup target add` is all it needs; no `cross`). The remaining legs
of that run were green or still running when this was written.

## KB-69: the branch's 4x peak-memory growth is CLOSED (222 -> 66.7 MB at 1024² s0 vs main's 54.5) and the estimate is speed-, chroma- and thread-aware (2026-09-24)

The bisected estimate-contract break (`e1a97fe`) is closed by re-fitting the model, not
by bounding the retention. heaptrack on the gate's own binary at 1024² s0 attributes
~160 MB of a 190 MB encoder peak to per-LEAF state kept for the whole frame: the
retained coefficient outcomes (`encode_intra.rs:322`, 72 MB), the y/u/v walk outputs
(`encode_sb.rs:1407/1543/1546`, 65 MB) and the partition tree (`partition_pick.rs:3524`,
25 MB). Per pixel: s0/s3 197..211 B, s6 156..179, s9 104..144; threads add 0 at peak.
`estimate()` = 1 MiB + padded x 32 + px x (240 | 180 by speed band + 40 x
samples-per-pixel) + 4 MiB per extra worker; grid widened to 22 cells (incl. t4/t8),
all bounded, worst slack 4.84x. **Then the memory itself, on the user's call that more-than-main is a bug:** the pack
reads only `tx_type`/`eob`/3 ctx bytes/`qcoeff` per txb, so `encode_sb::RetainedLeaf`
keeps exactly that — 12-byte headers + one eob-trimmed level slab per leaf, `i16` when
every level fits — and `pack_leaf` inflates one leaf at a time. Measured on the same
22 cells against `origin/main` run on the same box: 1024² s0 222 -> 66.7 MB (main
54.5), 1024² s3 218 -> 65.9, 256² s6 11.3 -> 3.08 (main 3.01), mono/bd12/tall/1x1 now
BELOW main; the +22 % residual at full-RD on noise is the retained levels themselves.
Ship cell A/B: bytes identical, ~1 % faster. Model re-fitted to the compact grid:
1 MiB + padded x 16 + px x (56 | 32 + 8 x spp) + 1 MiB/worker, slack 1.34x..3.21x.
Body + the full table: `docs/KNOWN_BUGS.md` KB-69.

Gates: `encode_limits_and_estimate` 4/4; aom-encode suite 790/790 with the compact
retention; `just gate-landing` at HEAD (see the commit); api-doc regenerated
(`resolved_threads` public; `RetainedLeaf`/`RetainedTxb` under `__internals`).

## Integration review of `perf/gate3-txfm-i16-batch` + the first real workspace gate on it (2026-09-24)

`docs/archive/INTEGRATION_REVIEW_2026-09-24.md` is the full record. The branch (160 commits, 0 behind main)
had never had `just gate-landing` run on it by any record, and CI never ran on it (the
workflow triggered only on main/PRs). The first run stopped in 20 s: three aarch64-only
scratch examples in `aom-dsp` (`89e90f6`) imported NEON-only shims and broke every
x86-64 `--workspace` build — invisible to `gate-encode`, which does not build examples.
Then, in order:

- **C instrumentation is versioned** (`docs/upstream-instrumentation/`, two patch sets +
  a gate inventory; `just upstream-instrument` / `upstream-pristine` / `upstream-check`,
  the last now the first step of `gate-landing`). The submodule had carried 965 lines of
  env-gated traces for two weeks behind `.gitmodules`' `ignore = dirty`, and
  `build.rs`'s SHA-only cache stamp let the instrumented `libaom.a` survive a revert —
  the stamp now digests the submodule's working-tree diff.
- **Traces are a runtime API**: `aom_dsp::trace` (`Trace`/`Focus` enums, `install` /
  `clear`, sink, `trace_on!`/`trace_focus!`/`trace_out!`). The published crates never
  read the environment on a trace path; `trace-env` (default OFF, on for the harness via
  `__internals`) keeps `AOM_*` working in-tree. 77 library `eprintln!` sites now go
  through the sink. Proven: `cargo tree --no-dev-dependencies` has no `trace-env`.
- **Duplicate test binary** `aom-bench/tests/highbd_inter_decode_envelope.rs` removed.
- **CI**: nextest on the three differential legs (the same suite measured 57 min ->
  341 s locally), and pushes to `perf/**`, `fix/**`, `feat/**`, `integrate/**`,
  `maint/**` now trigger it.
- **Scalar-pin leg**: five `aom-dsp` mirror differentials compared a scalar dispatch
  against the C AVX2/SSE2 kernel because they probed the token before the pin was
  applied — fixed by reading the pin first; no assertion weakened.
- **OPEN, bisected, not fixed: `e1a97fe` (retained leaf payloads replayed in pack)
  breaks the KB-50 estimate contract** — 256x256 s6 peaks at 11,255,676 B against an
  estimate of 6,373,376 B. Blocks the merge until the payloads are bounded or the model
  is made speed-aware.

Gate at the end: `upstream-check` pristine; `test-next` 1545/1547; `test-next-scalar`
1545/1547 (the 2 are the estimate pair); `census-gate` 4/4; `test-whereat` 4/4;
`api-doc-check` green. Public-API snapshots regenerated.

## KB-68 closed: `screen_512.yuv` byte-identical — IntraBC `predict_skip_txfm` runs DEFAULT_EVAL, not MODE_EVAL (2026-09-24)

The last open real-witness divergence closed in two stacked fixes:
- `2931675` — the phase-2 repack re-encoded IntraBC leaves into a
  src-initialized buffer; replayed intra leaves never write their
  reconstruction, so a leaf's DV source read src pixels instead of the
  coded recon (residual → 0 → false eob/skip at mi(16,86)). Phase 2 now
  seeds from phase-1's encoder recon — the state C's `write_modes`
  pass sees.
- `347a8c8` — `2931675`'s `predict_skip_txfm` plumbing indexed the
  MODE_EVAL column of `predict_skip_levels` (level 2 at allintra
  speed ≥ 3) into `rd_pick_intrabc_mode_sb` — but `pick_sb_modes`
  resets `set_mode_eval_params(DEFAULT_EVAL)` at rdopt.c:3666
  immediately before the intrabc call (rdopt.c:3688); rd.h:94 names
  DEFAULT_EVAL "e.g. intrabc". C's intrabc predict is the level-1
  mse-gate + per-subblock DCT check at every allintra speed. The
  level-2 SSE gate under-fired on a 4x4 leaf at mi(23,80), letting the
  coeff arm's block-level `skip_txfm_rd <= no_skip_txfm_rd` overwrite
  downgrade an all-zero txb — pricing the leaf at rate 22723 vs C's
  26418 and winning a SPLIT at mi(22,80) that C's budget rejects.
  `skip_txfm_level_default_eval` plumbs column 0.

Verified: `screen_512` 13,423 B = C byte-for-byte; `photo_1024`,
`photo_512`, `photo2_512` still identical; `screen_content_tools_byte_
match_real_aomenc` + `rdopt_skip_diff` + `self_contained_tools`
(7/7 incl. quality_knobs) green; 129/129 aom-encode lib tests.

**2026-09-24 corpus sweep** (`eprof_yuv` port-vs-C, cq27 s3,
tools-on, `ref_encode_av1_kf_screen_content` oracle): **164/164
byte-identical** across gb82-sc 10/10 screen captures (IntraBC-heavy),
CID22-512 validation 41/41, CLIC2025 training 32/32, kadid10k 81/81 at
512x384 — zero divergences, including the class KB-68's two fixes
touched.

## PGO harvest: source-level capture of the layout win (−6% ship, −2.2% plain) (2026-09-18)

Diverse-corpus PGO (10 cells) bound measured at **−7.4% wall**; harvested
into source: 41 `#[inline]`/`#[inline(always)]` hints on fns LLVM inlined
under PGO (txfm_rd_in_plane_intra, intra-predict chain, try_* dispatch,
OdEcEnc::normalize, ...) + `#[cold]` on scalar SIMD-fallback twins.
Plain-release build now **−2.2%**; new opt-in `[profile.ship]`
(lto="fat", codegen-units=1) reaches **−6.0%** — ~80% of the PGO bound
with no profile machinery. Byte-identical, deterministic 1w/4w,
466/466 dsp tests, aarch64 clean. Profile notes: root-workspace
profiles don't propagate to consumers — zenavif-side `lto="fat"` is
the documented consumer lever; `#[inline(always)]` errors on
`#[target_feature]` fns. Full record:
benchmarks/encoder_serial_residuals_2026-09-17.md batch 6-7.

## LR u8 kernel twins landed + measured: committed-plane swap evaluated NO (2026-09-18)

The kernel-twin program the prior entry named as the next lever is now
landed and measured, in two commits:

- `937f9ea` — read side: `PlaneCtx` stages `dgd_pad8`/`src8` once per
  plane at bd8 and the whole read-side kernel family is generic over a
  new `LrPixel` trait (`compute_stats`/`acc_stat_line`,
  `pixel_proj_error`, `calc_proj_params`/`get_proj_subspace`,
  `selfguided_restoration` + `integral_image` + `sgr_final_*`,
  `sse_none` u8×u8, `sse_dst` mixed-width). Per-type x86 loads ride as
  associated const fn-pointers to small `#[arcane]` widen helpers —
  LLVM devirtualizes, so the v3 bodies monomorphize to their previous
  code.
- `152da43` — apply side: `filter_unit`, `StripeBoundaries`,
  `StripeScratch`, `extend_frame`/`extend_lines`,
  `wiener_convolve_add_src_into`, `apply_selfguided_restoration` generic
  over `LrPixel`; boundary rows staged u8 via a `BndStore` narrow (~2
  rows/stripe from the u16 source planes — no `LrPlanePixels` API
  change); aarch64 got a third `NeonToken` const-pointer load shape
  (`vld1_u8`+`vmovl_u8`). At bd8 the u16 `dgd_pad` is not even
  allocated; `dst_pad` stays u16 (apply narrows i32→u16 either way).

**Measured** (eprof_yuv, real photo_1024, 1024² cq27 s3, interleaved
pairs): **~1% wall** total vs pre-change HEAD (1503–1512 ms vs
1510–1522 ms; ship cell unchanged at 1.295× — the LR bandwidth win is
under noise on mirror-tiled content); **−112M Ir at 512²** (21.000G vs
21.113G). Byte-identical at both sizes; all 466 aom-dsp tests pass
incl. `compute_stats_lowbd_matches_c` and the every-tier SIMD sweep;
aarch64 cross-check clean; api-doc regenerated.

**The committed-plane u8 storage swap is now measured and rejected, not
deferred.** The complete u8-twin program for LR yields ~1% wall — the
staging pattern captures the locality win without touching the walk.
The residual port-vs-C gap in restoration is ALGORITHMIC: port's
integral-image SGR stats (~349M Ir) vs C's boxsum sliding window
(~120M), `acc_stat_line` 88M vs C ~40M — per-kernel instruction-count
differences a storage swap cannot close (u8 instantiation was Ir-flat
to slightly-negative: `pixel_proj_error<u8>` +5% Ir from the extra
`cvtepu8` per load). The swap's remaining upside is the staging narrows
themselves — two passes per plane — against a ~44-site blast radius
across predictors/transforms/CFL/IntraBC/pack. Named residual, bounded:
the LR algorithm-structure gap (boxsum-vs-integral SGR, maddubs-shaped
wiener stats) is a kernel-port program, documented here; `dst_pad` u8
stores left as a tail item (~0.1-0.2% class).

**Side measurement worth keeping:** at 4 workers on real photo_1024
(eprof_yuv, 1x1/4w), port **1508 ms vs C 1890 ms = 0.80×** — the port
scales with workers where the C shim arm does not (1w: 1.27×). 1T
re-verification: new-vs-base **−0.8% at workers=1** — the twin gain is
single-core-real, not core-count compensation; output bytes identical
across 1T/4T.

## Post-walk u8 staging complete; ship cell re-measured 1.29× (2026-09-18)

`944ee88` finishes the post-walk half of the u8 split: at bd8 the
deblock APPLY now filters the lf pick's staged-u8 recon natively
(`loop_filter_frame_u8_opt` in place) and widens once into the u16
`deblocked` planes CDEF/LR read — the u16 clone + narrow + widen round
trip is gone (a fresh narrow covers the speed>=6 no-pick path).
Byte-identical 8928 B at 512²; `encoder_gate_lf_level_bit_exact_vs_real`
+ `encoder_gate_e2e_nonzero_lf_sweep` pass vs live C; Ir-neutral
(21.115G vs 21.113G at 512² — the win is one less full-plane u16
round-trip and a single staging owner, not instructions).

**Ship cell re-measured at HEAD** (`eprof_x86 {port,c} 1024 1024 27 3
1`, 6 interleaved pairs, byte-identical 40,237 B): port median
**2113.5 ms** vs C **1634.3 ms** = **1.29×** (was 1.384× at the
2026-09-15 record — the inv16x16/rect816 ymm + lf-u8-staging landings
moved it). Real-image cross-check (`eprof_yuv`, photo_1024.yuv real
frame, 1x1/1w): port 2413 ms vs C 1901 ms = **1.27×**; the port stream
is byte-identical to HEAD~1 (no regression), and its port-vs-C byte
difference on that cell is the pre-existing fleet-photo divergence
class, unchanged by this work.

**Evaluated and deferred, with the evidence:** the remaining pending
items — a `PlaneView`/typed-carrier refactor over `SbEncodeEnv`'s
planes and the committed-recon u16→u8 storage swap inside the tile
walk — are NOT currently justified. The measured residual at 512² is
kernel arithmetic, not storage: restoration's u16 SIMD kernels total
~1.0G Ir (acc_stat_line 177M, try_restoration_unit 156M, sgr calc_ab
145M, wiener 129M, pixel_proj_error 112M, calc_proj_params 83M,
integral_image 70M, sgr_final 123M) vs C's ~380M lowbd u8 twins — and
`PlaneCtx` stages padded u16 copies (`dgd_pad`/`dst_pad`) regardless of
the walk's plane type, so a storage swap would not touch it. Closing
it needs u8 twins of the LR statistics/projection kernel family (C's
`compute_stats`/`pixel_proj_error`/`sgr_*` lowbd set) reading staged u8
buffers — a kernel-port program, not a refactor. CDEF's `cdef_frame_u8`
is already documented 6.6% SLOWER than widening→u16-CDEF→narrowing
(narrow stores don't vectorize in the current abstraction), so CDEF
stays u16. The committed-storage swap additionally reaches predictors,
inverse transforms, CFL, IntraBC, pack and phase-2 repack — the
highest-risk surface in the encoder — for unmeasured ROI. What IS
landed: every post-walk stage that profits from u8 runs u8; every
conversion routes through `aom_dsp::lowbd`; offsets are one named seam;
band ownership is panic-free. aarch64 cross-compile clean;
`api-doc-check` green (no new pub items).

## u8 lowbd prep: lf-search staging lands −92.5M; band handout panic-free (2026-09-18)

The first slice of the u8 plane-storage split (the "u16-at-bd8 tax" —
the named clause-(4) residual) landed as the stacked-PR prep:

- `aom_dsp::lowbd::{narrow_u16_to_u8, narrow_u16_to_u8_into,
  widen_u8_to_u16}` — the canonical conversion helpers the split will
  use; `loop_filter_frame_opt`'s bd8 arm and `superres_downscale_plane`
  now route through them (`b8f9cd6`, `d174bc8`).
- `LfSearchFrame<P: LfTrialPixel>` (`b8f9cd6`): the trial loop is
  pixel-typed; `u16` keeps the old path, `u8` copies the staged plane
  straight into scratch, filters via `loop_filter_frame_u8_opt` (the
  identical walk the u16 arm reached through its own narrow/widen) and
  SSEs u8×u8 SIMD. `key_frame` stages the six planes u8 ONCE per pick
  at bd8 — the per-trial u16 copy + all-3-plane narrow + widen +
  scalar u16 SSE is gone. Measured: **21.113G vs 21.205G Ir = −92.5M**
  at 512² cq27 s3 2x2-tile/4-worker callgrind (≈−370M projected at the
  1024² ship cell). Byte-identical 8928 B; the live-C differentials
  `encoder_gate_lf_level_bit_exact_vs_real` +
  `encoder_gate_e2e_nonzero_lf_sweep` pass — picked levels are the same
  integers on either representation.
- Band-handout panic paths removed (`d174bc8`): the phase-1/phase-2
  tile-row cursors moved from `Vec<Option<&mut>> + take().expect(...)`
  to `Vec::into_iter` under the same mutex — iterator position IS the
  cursor, disjoint `&mut` ownership and deterministic merge unchanged,
  three `expect`s gone. The merge's `expect("every superblock belongs
  to exactly one tile")` became `KeyFrameError::InternalInvariant`
  (new `#[non_exhaustive]` variant — reports, never panics, on an
  unreachable path). Byte-identical under both 2x2/4w and 4x4/8w.
- `SbEncodeEnv::{y_off, uv_off}` (`9ebf4e8`): the `(mi*4)*stride +
  mi*4 - base_*` band convention — open-coded at ~21 sites across
  encode_sb/pack/partition_pick/nonrd_pickmode — is now one named
  seam, so the storage-type split's diff is about STORAGE, not
  re-derived offsets. `#[inline]`, byte-identical.

aarch64: `cargo check --target aarch64-unknown-linux-gnu` clean at
HEAD (the merge's NEON work + this refactor both compile).

## aarch64 encode reaches Gate 3: 1.47× C via verbatim NEON quantize + wiener + fwd-txfm twins (2026-09-17)

First aarch64-apple-darwin run of the encoder gates on this branch —
HEAD did not even compile (`loopfilter/simd.rs` `store!` referenced
`c` as a free variable under `magetypes` hygiene; aarch64-gated so CI
never saw it). Serial baseline on M4 Pro, `1024x1024 cq27 s3`:
**port 2226 ms vs C 1258 ms = 1.77×**; this landing brings it to
**1892 ms vs 1292 ms = 1.47×**, byte-identical (40237 B both arms).

**Mechanism — `sse_neon` shim (`aom_dsp::sse_neon`, new).** The fused
i16 txfm bodies were verbatim SSE2 transcriptions on
`archmage::intrinsics::x86_64`, which does not exist on aarch64. The
shim re-exports the real intrinsics on x86_64 and supplies the same
names over NEON on aarch64 (`#[target_feature(enable="neon")]` free
fns — legal under `forbid(unsafe_code)`; loads/stores go through
canonical byte-array copies that LLVM folds to single `ld1`/`st1`).
That un-gated the fused path per-tier unchanged and bought
2226→2000 ms. But shimmed SSE2 on NEON still loses ~3-5× to C's native
NEON (constants materialize as `ldp`+`sub`+`bfi` scalar chains vs C's
immediate lane ops), so the shim is the bootstrap, not the landing.

**Verbatim C-NEON twins.** Where C's NEON kernel is a *different
algorithm* than its AVX2 one (quantize fp/lp/b: `vqdmulhq` +
shift-compensation vs i32-lane; fwd txfm: `vmull_lane`/`vmlal_lane`/
`vqrshrn` butterflies vs the SSE2 `_mm_mulhrs` shape), the shared-body
trick cannot work — each got a `#[arcane]`-over-`NeonToken` twin
transcribed line-for-line from `upstream/`:

- `quant/simd.rs`: `quantize_fp_impl_neon`, `quantize_lp_impl_neon` —
  bit-identical to `av1_quantize_fp_neon`/`av1_quantize_lp_neon` on
  45k+60k adversarial cases (new oracles `ref_quantize_fp_neon`,
  `ref_quantize_lp_simd` pick the per-arch C tier).
- `quant/mod.rs`: `quantize_b_impl_neon` (incl. the 32x32/64x64
  variants) — bit-identical to `aom_quantize_b_neon` on 30k cases.
- `restore/wiener.rs`: verbatim `wiener_convolve_neon.c` twin —
  ~45 ms end-to-end.
- `transform/simd/fwd_neon.rs` (new): verbatim
  `av1_fwd_txfm2d_neon.c` — 4x4, 8x8, 4x8/8x4, 8x16/16x8, 16x16 fused
  drivers + the x4/x8 butterfly, transpose, flip, fdct/fadst/fidentity
  kernels. Per-shape accept bounds preserved (e.g. 8x8 declines at
  |in|>511, not 512 — the perm-diff gatespike caught that).
  Microbenches went from 3-6× C to parity (4x4: 61→7 ns vs C 11;
  8x8: 77→35 vs C 28; rects ≈1.0-1.2× C).

**Differential rule that made this safe:** on aarch64 the port's NEON
tier is now compared against the REAL C NEON kernel, not scalar C —
C's own tiers disagree with each other out of domain, so NEON-vs-NEON
is the only honest oracle on this arch (aom-sys-ref gained
`ref_quantize_fp_neon` / quantize-b NEON wrappers).

**Pre-existing aarch64 failures, unchanged by this diff** (HEAD never
compiled here, so these are first-run latent bugs, logged not fixed):
`block_error_matches_c_avx2_full_domain`,
`highbd_filter_intra_edge_at_byte_identical`,
`highbd_quantize_b_differential`,
`highbd_quantize_b_simd_bit_identical_to_c_at_every_tier` (garbage at
pinned `i32::MIN` lanes — C signed-overflow UB compiled differently
for aarch64 than the port mirrors), plus `encode_limits_and_estimate`
under-stating measured peak 6.4 MB vs 11.7 MB at 256x256 (allocator
accounting is arch-dependent; the estimate formula needs an aarch64
calibration pass).

**Gates run:** `cargo test --release -p zenav1-aom-dsp --test all`
(401 pass / the 4 above); `just gate-encode` (aom-encode +
aom-bench integration targets — all pass except the two named
pre-existing failures); x86_64 `cargo check` clean. eprof byte gates:
40237 B both arms, cq27 s3 1024².

**Remaining aarch64 levers, ranked by last sample:** fwd_txfm residual
(~150 ms — the 4x16/16x4/32x32+ shapes still take the generic pass
path; C NEON has fused kernels through 32x32), search_tx (~150 ms),
intra_pred (~90 ms), var_dist (~40 ms), `cnn.rs` conv_valid (x86-only,
C NEON exists at `cnn_avx2`→`cnn_neon.c`, ~1144 lines).

## Thread curve, content matrix, rayon backend (2026-09-16, `74fa859`)

Full same-session sweep vs the threaded aomenc, row-tile geometry,
1024x1024 cq27 s3:

| t | C | port | ratio | port eff | C tile-mt eff |
|---|---|---|---|---|---|
| 1 | 1822 | 2334 | 1.28 | — | — |
| 2 | 1020 | 1227 | 1.20 | 95% | 89% |
| 3 | 744 | 854 | 1.15 | 91% | 82% |
| 4 | 601 | 655 | 1.09 | 89% | 76% |
| 8 | 439 | 430 | 0.98 | 68% | 52% |
| 16 | 384 | 332 | 0.86 | 44% | 30% |

Port scaling efficiency exceeds C's tile-mt at every point; the serial
gap narrows monotonically and inverts at 8t. `KeyFrameConfig::threads=0`
is now AUTO (`available_parallelism`, clamped to tile rows — the frontier
is unchanged: a 1-row grid still serializes).

Content/size matrix (port/C, same-geometry): 1024² photo 1.09x@4t /
0.98x@8t; noise 0.87x/0.80x (0.93x serial — faster than C); gradient
0.51x/0.41x; **screen 1.70x/1.19x — IntraBC DV-search tax, the one class
outside the bar**; 1080p photo 1.12x/1.01x; 1080p screen 1.70x/1.18x;
4K photo 1.11x/0.99x; 512² photo 1.07x/0.88x.

`aom_dsp::par::{map_workers, join}` now fronts all six spawn sites;
opt-in `rayon` feature (`zenav1-aom-encode/rayon`) runs the same items
on the host's shared pool — zenavif already depends on rayon, so this
avoids oversubscribing a server; `threads` becomes a ceiling not a
spawn count. Byte-identical on both backends; wall identical
(433.5/331.3 rayon vs 435.9/331.7 std at 8t/auto).

Threads guidance from the measured knee: efficiency holds ~85-90%
through 4t, drops to ~68% at 8t/1MP. Throughput servers: threads=1..4
and parallelize across requests. Latency mode: threads≈tile rows (auto
clamps there anyway). Bigger frames keep efficiency higher at high t
(4K: 71% at 8t).

## Threaded vs threaded C: within 10% at 4t/8t on row-tile geometry (2026-09-16)

`e7cdd08` — the honest threaded-C baseline exists now: a SEPARATE
libaom v3.14.1 `aomenc` built from `upstream/` with multithread ON
(the in-repo oracle is `CONFIG_MULTITHREAD=0` + `g_threads=1` — it can
gate bytes, not threaded wall time). Ship cell 1024x1024 cq27
`--cpu-used 3`, allintra, cdef off, restoration on, sb64; C arm
`aomenc --allintra --good --tile-rows=N --threads=T`, port arm
`eprof_x86` with the same row-tile geometry — all numbers below are
one-session interleaved medians on a 7900X (earlier figures were
polluted by a concurrent build; treat these as the record):

| arm | serial | 4 threads | 8 threads |
|---|---|---|---|
| C row-mt (no tiles) | 1754 | 680 | 577 |
| C row tiles, like-for-like | — | 607.6 (16 rows) | 441.5 (8 rows) |
| C best tile config | — | 597 (4x4) | 413.1 (4x4) |
| **port** | ~2230 | **666.5 (16 rows)** | **435.9 (8 rows)** |

Ratios: **4t 1.097x** vs same-geometry C (1.12x vs C's 4x4 best);
**8t 0.988x vs same-geometry C — faster than C — and 1.055x vs C's
best config.** The user's <=1.10x bar is met at both thread counts on
the like-for-like contract, and at 8t against every C geometry.

What landed to get there (all byte-identical, `forbid(unsafe_code)`
intact): phase-1 + phase-2 bands now pull off a shared mutex cursor —
dynamic scheduling instead of static interleave (a worker that draws a
cheap row takes the next one); loop-restoration search threaded by
tile-row ranges with private `RscState`/cloned `PlaneCtx` and
commutative totals merge (`LrSearchInput::threads`);
`build_lf_mi_grid_mt` SB-row banding; `pick_filter_level_mt` — U/V on
workers plus the sequential luma chain's independent low/high trial
pairs on a second worker (4t sees Y(2)+U(1)+V(1)). Phase-1 scaling
2097 -> 342 ms at 8t = 6.13x; the residual ~10% inefficiency is
worker-boundary setup + memory, not scheduling. Heaptrack peak heap
112.9M at 8t vs ~74M serial — the +40M is the LR workers' `PlaneCtx`
clones (~5M/worker transient, bounded linear-in-workers, documented
here rather than refactored: splitting `PlaneCtx` into shared/mutable
halves is the follow-up if it ever matters).

What is NOT matched: C's square-tile (4x4) parallelism. Column tiles
interleave within every plane row, so `split_at_mut` row-banding can't
express them under `forbid(unsafe_code)` — closing that last ~5-10%
needs per-tile owned recon buffers + stitching or a different access
model. Documented, not pursued: at 8t we're already at parity, and at
4t the like-for-like bar is met.

`AOM_TIME_PHASES=1` now eprints per-phase spans (phase-1, lf_pick,
deblock, cdef, lr_search, pack2, assemble) — the instrumented
breakdown that drove the tail work.

## Phase-1 tile walk is threaded by row bands + bd8 lpf runs the u8 kernels (2026-09-16)

`6ada3da` — `KeyFrameConfig::threads` (default 1, serial path byte-untouched).
With `threads > 1` the phase-1 `pack_tile_stop` walk partitions tile ROWS
into contiguous bands; `split_at_mut` gives each `std::thread::scope`
worker disjoint `&mut` recon slices, so no `unsafe` anywhere. The
`SbEncodeEnv::base_y`/`base_uv` convention FLIPPED from additive to
subtracted band bases (`slice[absolute - base]`), including inside
`chroma_plane_offset` and the `var_part`/`allintra_vis` base args — the
serial path sits at base 0 and is unchanged. One out-of-band reader,
`extract_intra_cnn_window`'s crop-clamped 65x65 window, gets the full
frame via a new `src_y_frame` field; every other read is tile-bounded
(IntraBC clamps to `tile.mi_row_start`, above-neighbour reads are
guarded by `mi_row > tile_row_start`, no post-filters run inside pack).
Bugs found by the differential, not by review: reversed band→worker
assignment (`pop()` took from the end) — invisible on the
vertically-symmetric mirror-tiled photo, caught on asymmetric content;
and the last band dropping the plane's edge-extended tail, which partial
bottom superblocks legitimately read/write (HOG ±1-px gradients,
predictor writes). Gate:
`encoder_gate_multitile::encoder_threaded_row_bands_match_serial` —
flat/gradient-h/gradient-v/noise/asymmetric cells x {1x2, 1x4, 2x2} tile
grids x {2, 4} threads, whole-stream byte equality, plus the asymmetric
+ non-SB-aligned heights that caught both bugs. Measured 1024² cq27 s3:
**serial ~2231 ms → 809 ms at 4 threads (2.76x) / ~511 ms at 8 (4.4x,
8 tile rows)** — byte-identical to serial every rep; heaptrack peak heap
73.7M @1 thread vs 73.9M @8 (worker state bounded, planes shared).
This is the absolute-time lever the clause-(4) record priced at
~3.2x/~5.1x; it does not move the port:C ratio (both arms thread).

`41c9c06` — the bd8 loop filter now runs the u8 kernels C uses:
`loop_filter_frame_opt` narrows the u16 planes to a u8 workspace at
bd==8, runs `filter_block_plane_u8_opt` (the lowbd twin of the
`lpf_opt_level==1` batched walk, with new `vertical_n`/`horizontal_n`
u8 wrappers — C's `_dual`/`_quad` as a per-segment loop) and widens
back. Byte-identical at both witnesses (40,237 B; 11,188 B at 512²) vs
real aomenc; `lf_apply_diff` 4/4 incl. the `u_tx=false` opt-divergence
cell that REJECTED the first version routed through the level-0 u8 walk
(the batching is semantic on non-uniform-tx grids). Wall-flat at the
ship cell — the win is the ~2.5x kernel-Ir cut on the lpf class and
removing the kernel-shape divergence; the u16-at-bd8 tax now reduces to
restoration (needs u8 wiener/sgrproj ports) and `highbd_variance` (u8
kernels exist scalar-only; the SIMD tier plus a u8 source plane are the
structural remainder). Also fixes `encode_intra_plane_uv_diff`: a test
site still passing `chroma_plane_offset` a margin base under the old
additive convention.

Downstream: `zenavif` pin bumped `fbea6b4` -> `41c9c065` on branch
`bump/zenav1-aom-encode-41c9c065` — `#[non_exhaustive]` migration
(`allintra_speed0` + field assignment), and one seam bug the new
`validate_configuration` caught: the alpha aux item's mono encode
inherited the colour item's MC_IDENTITY, which upstream refuses on mono.
zenavif gates: `aom_encode_backend` 22/22, `aom_roundtrip_loss` 5/5,
`resolved_routing` 6/6 — all at the real git pin.

## Ship cell under the user's 1.40× bar — median ~1.36× (Gate 3, 2026-09-16)

1024² cq27 `--cpu-used 3` (the zenavif preset), 12 interleaved port/C
pairs, byte-identical 40,237 B every rep: **median ≈1.36×** (port
~2213 ms / C ~1625 ms) — ~2.5% margin vs ±1.5% run noise. Third batch:
(a) the encoder loop filter runs C's `lpf_opt_level==1` traversal —
`filter_block_plane_opt` dual/quad `nseg` batching wired via
`loop_filter_frame_opt` (decode stays level-0), kernel calls
879K→402K/512² rep, differential-verified against the real exported
`*_opt` symbols; (b) retained-coeff alloc halving — `TxbEncode`
qcoeff‖dqcoeff merged into one `SmallVec<[i32;64]>` (~99.5k→~57.7k
allocs/rep at 512²), `TxbsVec` pooled leaf vecs, per-SB `var_cache` on
`TileCtxState` (frame-scoped, per-worker under tile threading),
`StripeScratch` + `inter_rd` TLS scratch; (c) const-eval flattened
`FWD_CFG`/`INV_CFG` transform dispatch tables. Named residual gaps: the
u16-at-bd8 tax (C's `av1_lowbd_*` u8 kernels — `lpf_impl_v3` ~33M,
restoration and `highbd_variance` u16 kernels vs C's 16-lane u8; the u8
kernels exist on the decode path but the encoder's u16 planes never
route to them — widening + a u8 workspace is the remaining program) and
safe-Rust bounds-check/min-max overhead in the trellis
(`forbid(unsafe_code)`). Record:
`benchmarks/encoder_ship_cell_1_40_2026-09-15.md`.

## `pack_leaf` replays retained leaf payloads — the pack re-encode is gone (Gate 3, 2026-09-15)

`pack_leaf` re-ran `encode_b_intra_dry` on every committed leaf — 3,312
calls vs the pick's 1,656 at the 192² s9 cell, ~44% of it — where C's
`write_modes_b` reads the `cb_coef_buff` coefficients the OUTPUT_ENABLED
encode already produced. The port now mirrors that: `LeafWinner::replay:
Option<LeafEncodeOut>` retains the output-enabled walk's result
(`encode_sb_dry`'s leaf sites via `finish_leaf_out`, plus
`nonrd_leaf_pick_and_encode`; ordinary intra leaves only — inter/intrabc
keep the re-encode fallback since their early-return arms stamp a
different ctx pattern). The encode's persistent-context steps 4+6 moved
into shared `stamp_leaf_ctx`, which the replay arm re-runs on the pack
pass's `TileCtxState` — recomputed-identical `(txb_skip_ctx,
dc_sign_ctx)` pairs, `cul == txb_entropy_ctx` asserts live. The payload
is restored to the winner so the phase-2 `pack_tile_from_trees_lr`
re-walk replays too.

Same-protocol callgrind `192x192 cq27 s9 reps=5`: **229.85M → 166.93M
Ir (−27.4%)** — `encode_b_intra_dry` calls 4,968 → 1,524. Session
cumulative at the cell: **271.1M → 166.93M (−38.4%)**. Perf gate
(`encode_perf_vs_libaom`, 12/12 byte-identical): worst cell **3.42× →
2.12×**, s9 cells 3.42×/3.3× → 2.03×/1.96×. Byte-identical at both
witnesses; encode `all` 654/654. Record:
`benchmarks/encoder_s9_pack_replay_2026-09-15.md`.

## Mode split: `KeyFrameMode::{LibaomExact, Zenaom}` — the SCM trial lands as the first opt-in zenaom feature (KB-67, 2026-09-13)

`KeyFrameConfig` gained `mode` (default `LibaomExact`) and
`screen_likelihood: Option<f32>`. `Zenaom` opts into measured deviations;
its first citizen is the KB-66 machinery, re-landed: two
`FIXED_PARTITION`/`BLOCK_32X32` trial encodes at `q = max(q_orig, 244)`
through the `rd_use_partition_real` replay over a pre-stamped grid,
all-plane PSNR + `palette_pixel_num`, C's `0.9 dB` / `ratio>4` decision,
flip → `allow_screen_content_tools=1, allow_intrabc=0`. Everything the
trial could change flows through the same `FeatureFlags` C flips, so
detector-positive frames encode byte-identically in either mode.

The nomination gate is the measured piece, not boilerplate: C runs the
trial on EVERY detector-negative key frame, but in the still-image
envelope that is the common case. The port fires it only when the
detector's net score is merely positive (`count_palette*16 > count_photo`)
or the caller's `screen_likelihood` hint reaches `SCM_TRIAL_HINT_MIN`
(0.10 — the zenanalyze `patch_fraction` seam, filled caller-side in
zenavif; `aom-encode` stays publishable with no sibling dep). A
fractional-threshold gate was measured INSUFFICIENT — C's own trial wins
cells at 3.9% of threshold — and pure photographic noise scores deep
negative, so "any net palette evidence" is the cut. Measured cost
(`ztrial_cost`, release): firing cells 1.07x–1.19x, gate-off and photo
cells 1.00x. Decision agreement with C's own two-pass trial on every
sweep cell; on 256x256 cq32 s0 the flipped stream is byte-identical to
C's `LAST_PASS` output. Tests: `zenaom_scm_trial.rs` (flip + C-decoder
conformance, photo no-flip, exact-contract 9 cells); sweep kept as
`#[ignore]`d `probe_zenaom_trial_matrix`. Honest bound recorded in KB-67:
patches below the net-positive cut on large frames (e.g. 64x64 inside
512x256) do not nominate — the hint is the escape hatch.

## The SCM trial was ported, proven byte-exact — and is UNREACHABLE in the port's envelope, so it was reverted (KB-66, 2026-09-13)

`av1_determine_sc_tools_with_encoding` (the C3 "SCM trial" bullet, long the
next open class) was ported whole: a `FIXED_PARTITION` stamp driver
(`var_part::set_fixed_partitioning` + a `rd_use_partition_real` replay arm in
the tile walk), the two-pass trial at `q = max(q_orig, 244)`, all-plane PSNR
at stream depth, `palette_pixel_num`, and C's `0.9` / `ratio>4` decision —
and it was RIGHT: on a flipping cell (detector-negative mixed photo/UI,
256x128 cq45 s0) the port's trial-flipped stream was **byte-identical to the
C shim's `AOM_RC_LAST_PASS` output** (833 B), the only path that executes it.

Then the reachability check inverted the item. The call site sits inside
`encode_with_recode_loop` (encoder.c:3326), entered only when
`sf.hl_sf.recode_loop != DISALLOW_RECODE` — and
`oxcf.pass == AOM_RC_ONE_PASS && has_no_stats_stage` (no lookahead;
aomenc allintra forces `g_lag_in_frames=0` AND `passes=1`) forces DISALLOW
(speed_features.c:2785, encoder.h:4159). Verified live on the instrumented
oracle: `recode_loop=0 no_stats=1 pass=0 lap=0` — **the one-pass shim never
calls the trial**. The port's envelope is exactly that (no passes/lag knobs
exist), so running the trial made the port flip `allow_screen_content_tools`
where reachable-C stays off — 12/12 probe cells REGRESSED, all restored
byte-exact by the revert. The mechanism is recorded in KB-66's entry if
a lag/two-pass surface ever ships.

Consequence for the inventory: the "SCM trial gap" is NOT open — it was never
a divergence in this envelope. The "14 tiny" fleet class loses its C3
attribution (its cells span cpu-used 4/6/8; GOOD mode disables
`extra_sc_testing` at speed >= 6 and allintra never reaches the call — the
s6/s8 cells cannot be the trial under ANY usage) and needs re-attribution
when the fleet planes are re-staged. The two committed adversarial probes
(`probe_sc_tools_trial_gap_*`, 105 cells) now pin the inverted claim: port
and one-pass C must NEVER disagree on `allow_screen_content_tools`.

## Coded-lossless IntraBC lands — conformance + 47/48 probe cells byte-exact (KB-64 / KB-65, 2026-09-13)

Two landings, one mechanism each:

**KB-64** — `intra_model_rd` walked its model-prediction tiles in the mu-64
CHUNK order of `txfm_rd_in_plane_intra` where C walks flat raster
(`for row; for col`, intra_mode_search_utils.h:637). The order is
observable — each tile predicts off in-plane predictions written by earlier
tiles — so any leaf > 64 px got a different model-RD on left/below-left-edge
modes, flipping the top-k prune set. Closed the `mono_cq63` SB128 pin (one
byte, angle-delta near-tie at a 128x128 leaf), promoted to a byte gate.

**KB-65** — the coded-lossless IntraBC arm, plus two roots it exposed:
`var_tx::choose_smallest_tx_size_inter` ports C's lossless dispatch
(`av1_pick_uniform_tx_size_type_yrd` -> `choose_smallest_tx_size` — flat
TX_4X4 + FWHT, `predict_skip_txfm` gated `!lossless`); the `!coded_lossless`
decline is gone from `key_frame.rs`. Then: (a) the write-side walks
(`ibc_encode_block_inter_y`, `pack_vartx_txb`) rooted their txb recursion at
`MAX_TXSIZE_RECT_LOOKUP[bsize]` where `get_vartx_max_txsize` (blockd.h:1452)
returns TX_4X4 at lossless — quadtree-DFS emission into a raster-reading
decoder = NON-CONFORMANT streams, rejected by the real C decoder with
`Invalid intrabc dv`; (b) `search_allow_intrabc` threaded the detector's
raw `sct.allow_intrabc` without `&= enable_intrabc` (encodeframe.c:2194) —
a +51/leaf phantom flag charge in knob-off screen encodes that biased
partition candidates by leaf count and flipped five lossless near-ties.

48-cell probe matrix (`examples/ibc_lossless_probe`: screen/detail x
{64,128,256,512x384} x cq{0,32} x s{0,3,6}, palette off): **47/48
byte-exact, every stream conformant on both real decoders**. Residual:
`Screen 512x384 cq0 s0` +3 B (0.01 %) — C commits DC+filter_intra3 at leaf
mi(50,67) where the port commits PAETH, a few-unit rate-model near-tie;
recon identical. `screen_content_tools_byte_match_real_aomenc` now covers
cq0 + a knob-off leg + a matrix-level IntraBC-engagement assert.

## The intra variance factor's 4x4 walk banded — `variance_4x4_units` (Gate 3, 2026-09-13)

The lever map's top unlanded row was the variance family — 80.9 % of all
`dist::variance` samples at the shipping preset, because
`intra_rd_variance_factor` walks every candidate block in 4x4 units and
called a 16-element scalar variance per unit, per candidate-mode eval.
New `aom_dsp::dist::variance_4x4_units` batches the raw (sum, sumsq) walk
for a whole band row: a `#[arcane]` v3 kernel folds FOUR units per ymm
(`madd` pair-sums accumulated over each unit's 4 rows, one `hadd_epi32`
per group), an xmm twin takes the 2-unit tail, the scalar walk the last
unit, and rows beyond 32 units — wider than any reachable bsize — keep
the per-unit calls. The per-bd fold (`norm_var_4x4`) is inlined verbatim
from `highbd_variance`: wrap at bd8, ROUND_POWER_OF_TWO + clamp at
bd10/12. Exact in i32 on the pixel domain (samples < 4096).

Verified bit-identical to the scalar twin at every token permutation
(`var4x4_units_simd_diff`: bd {8,10,12} x unit counts 1..32 covering
every group/tail residue x strided/random/all-max/flat data, non-vacuity
asserted) and byte-identical on the shipping cell (40,237 B). Band N=24
rotated, 1024x1024 cq27 s3: **-0.383 % paired median, 18/24 faster,
p=0.0227** against a same-binary null of -0.143 % (p=0.31) — a small but
real strictly-less-work landing.

## The size-axis finding-B residual closes — `winner_tx_type_map` paired chunk-ordered winners with raster positions (KB-63, 2026-09-13)

The last pinned size-axis cells — 480x480 `--enable-1to4-partitions=0`,
480x480 `--enable-ab-partitions=0 --enable-1to4-partitions=0`, 512x512
`--enable-ab-partitions=0` (mono cq63 s0 SB128, mirror-tiled conformance
content) — are now byte-identical to real aomenc. They had been documented
as bounded per-cell RD near-ties; that attribution was wrong about mechanism.
`txfm_rd_in_plane_intra` produces its `TxbWinners` in
`av1_foreach_transformed_block_in_plane`'s mu-64 chunk order (64x64 chunks,
raster inside each), but `winner_tx_type_map` stamped them in flat raster
order — identical on <=64x64 leaves, a PERMUTED committed `tx_type_map` on
any SB128 leaf wider/taller than 64 px whose winning tx types are
non-uniform. On the witness leaf (mi(16,96) bs14, the HORZ sub of a 128x128
node, TX_16X16) every per-txb eval matched C exactly — mode, rate, dist, even
the searched `tx_type=2` for the divergent `blk(4,8)` — while the committed
map held `0` and the output walk re-quantized it under DCT. That is why it
read as a near-tie: the search was right, the committed map was scrambled.

Fix: `winner_tx_type_map` mirrors the walk's chunk enumeration, plus a
`debug_assert_eq!` tying the winner count to the enumeration. Verified: all
three pinned cells byte-identical (834/829/936 B), decode-localizer partition
trees identical on all 243 nodes, `sb128_e2e` 5/5. The finding-B arm of
`size_axis_open_divergences_pinned` is now a byte gate; the 480/512 contexts
stay out of `ALL_SIZE_CONTEXTS` only on the 200 s budget ceiling. Still open
(different roots): the SCM trial gap and the cpu-8 photo rows.

## Every speed-0 near-tie band closes — the AB-reuse clone carried a stale `tx_type_map` (KB-62, 2026-09-13)

`MONO_S0_OPEN` (KB-27), `SPEED0_1080P_OPEN` + `HD_HBD_OPEN` (KB-38's residual),
the crop-axis mono pins, and `config_permutations`'s `scr_mono_b10/cq32/diag=0`
are all now byte-identical to real aomenc — one root, and none of them was
what its shape suggested.

At a `BLOCK_16X16` node the port picked `VERT_B` where C picked `VERT`. The
difference was in the AB candidate's first sub-block: `VERT_B` sub-0 reuses
the `VERT` sub-0's searched winner (`is_rect_ctx_is_ready`), and C's
`av1_update_state` does `xd->tx_type_map = ctx->tx_type_map` — an ALIAS
(encodeframe_utils.c:217), so the mid-stage `encode_superblock(DRY_RUN)`'s
eob-0 -> DCT_DCT resets write through it into `ctx->tx_type_map` itself.
The AB stage's `av1_copy_tree_context` therefore copies a POST-reset map.
The port cloned `w0` BEFORE the mid-stage `encode_b_intra_dry`, so its clone
carried the post-search map — one eob-0 txb still held its searched
`ADST_DCT` — and the reused leaf's re-encode read that stale type for its
forward transform (`get_tx_type_y`), producing different coefficients and a
different right column, which is the next sub-block's left edge. The
split-child reuse was already correct — it clones the committed
`SbTree::Leaf`, whose own commit `encode_sb_dry` walk had already reset the
map. Fix: take the clone after the dry-run.

The "monochrome cq24" and "speed-0 >=1080p" shapes were counting artifacts:
the stale entry only moves a stream when a reuse-eligible rect sub-0 has an
eob-0 txb that searched a non-DCT type AND the shifted recon flips a
near-tie — rare per block, so it needed ~2 MP of superblocks (or the one
lucky 64x64 cell) to land inside a pin.

Verified: `mono_speed0_size_qindex_localize` window fully clean (cq18..30 x
s0..7, kept as a guard); `partial_sb_speed_axis_chroma_formats_byte_match`
96/96, `MONO_S0_OPEN = &[]`; crop straddle 6/6 incl. controls;
`s4cov_hd_format_axis` 4/4 gates green with `SPEED0_1080P_OPEN`/`HD_HBD_OPEN`
= `&[]` and `speed0_1080p_qindex_arm_localize` reporting `divergent rows: []`;
`mono_vector_open_divergences_pinned` 6/6 exact, `CONTENT_DIVERGENT_CELLS =
&[]`; `bd12_dispatch_tier_agreement`'s >=1080p map all-zero (`1920x1080 cq24`
+59 -> 0 — the cell the standing goal names); `self_contained_key_frame`
10/10 (549 cells), e2e 32/32,
`encoder_gate_bd10_diff` 7/7, coding-tools 48/48, `speed_envelope` + all 43
`combinations_*` green. Still open (different roots): the SCM trial gap,
and the cpu-8 photo rows whose content is not in-repo. (Size-axis finding B
closed same-day under KB-63 — a `tx_type_map` ORDERING defect, not reuse
staleness. `NONRD_CQ63_OPEN` also reads clean — but that was KB-58's per-SB-
qindex plumbing, a stale pin re-measured today, not this fix.)

## The entire `HBD_OPEN` band closes — the intra-CNN prune window truncated u16 samples to u8 (KB-61, 2026-09-13)

Every high-bit-depth divergence pin in the tree — `self_contained_key_frame`'s
six `HBD_OPEN` cells (bd10/bd12, single- and multi-tile, 4:2:0/4:4:4/mono,
cq32/cq5/cq0), `s4cov_qm_axis`'s 49-row `HBD_OPEN` set, and
`config_permutations`'s `b10_64` probe (the real `av1-1-b10-00-quantizer-00`
conformance source) — is now byte-identical to real aomenc. One root:
`partition_pick::extract_intra_cnn_window` built the CNN partition-prune's
65x65 luma window with `env.src_y[..] as u8` — wrapping every >255 sample
mod 256. C dispatches on bit depth (`av1_cnn_predict_img_multi_out` vs
`_highbd`) and feeds BOTH the raw samples, normalizing inside layer 0 by
`1/((1<<bd)-1)`. First localized on `vgrad256 bd10 cq24 s4`: the port's
NONE-leaf cost at `mi(0,16)` matched C exactly (165746189) but the CNN prune
flipped `square_split_disabled` where C descended — the per-speed band shapes
(bd10 s4-5 vs bd12 s1-3 vs bd12-multi all of 1..6) were ONE threshold
phenomenon, not a family of HBD RD defects; a flipped flag changes the stream
only where the partition decision is close.

The fix keeps the window `u16` end to end and normalizes by
`((1<<bd)-1) as f32` — byte-inert at bd8 (identical floats to the old
`p/255.0`). The oracle got the matching split: `rd_shim.c`'s CNN shims take
`const uint16_t *` and dispatch bd8→real lowbd / bd>8→real
`av1_cnn_predict_img_multi_out_highbd` (scalar arm via the renamed
`cnn_cscalar.c` entry) — no transcribed CNN. Direct CNN buffer+decision
differentials now cover bd8/10/12 (6/6).

Verified: `self_contained_key_frame` **549/549** (was 447 — the HBD axes J/N
widened to the full speed band once the band closed; all six pins
self-promoted, `open = &[]`); `s4cov_qm_axis` 360/360 exact with
`HBD_OPEN = &[]`; `speed_envelope_stock_map_is_pinned` passes with
`b10_64 = &[]`; `encoder_gate_bd10_diff` 7/7; e2e byte-match 32/32; dump sweep
bd10+bd12 x s0..s9 = 20/20. **Not** this bug and still pinned: the speed-0
>=1080p band (`SPEED0_1080P_OPEN`, `HD_HBD_OPEN` — KB-38's residual; the CNN
prune does not run at s0) and `MONO_S0_OPEN` (bd8, s0).

## The entire tune bundle closes — `x->rdmult` is per-NODE under a perceptual tune, and C folds it at three different scopes (KB-59, 2026-09-13)

All 84 tune cells at s0/s3, all 8 fast-preset tune cells at s6/s8, and the
six chroma-delta-q tune ramps are now byte-identical vs real aomenc — the
largest pinned-open class in the tools gate (was 0/84). One mechanism:
`handle_tuning` arms `av1_set_mb_ssim_rdmult_scaling` (encoder.c:4301),
a per-SB grid of geometric-mean variance factors that `av1_set_ssim_rdmult`
(partition_search.c:596-657) folds into `x->rdmult` at EVERY recursion node
via `setup_block_rdmult` — so the trellis rdmult is position- and
size-dependent, and the port's per-SB-constant `env.rdmult` was wrong at
three different scopes at once:

- `pick_sb_modes` folds to the leaf and restores on exit — leaf RD and the
  leaf trellis run on the leaf fold.
- `rd_pick_rect_partition` (partition_search.c:3500) does NOT fold — its
  `best_remain` subtraction, `this_rdc` recompute and `sum_rdc` accumulate
  run on the PARENT node's rdmult.
- `rd_try_subblock` (:3133 — AB and HORZ_4/VERT_4) folds itself and runs
  everything at leaf fold, including refolding the by-value budget and the
  mid-stage dry-run encode.
- `rectangular_partition_search` and `av1_rd_use_partition` mid-stage
  dry-runs run on the PARENT fold — the committed recon a sibling leaf
  reads was trellis-quantized with the parent rdmult. (The decisive
  evidence: two C dry-run encodes of the same leaf, identical
  tcoeff/ctx/quantizer/eob, different qcoeff tails — `x->rdmult` differed,
  126597 vs 150061.)

The port now carries `SbEncodeEnv.ssim` (the per-SB
`av1_set_mb_ssim_rdmult_scaling` grid + `pre_rdmult`/`intra_modifier` in
`allintra_vis.rs`), folds via `node_env`/`node_rdmult`
(encode_sb.rs:532-561 — identity when `ssim` is `None`, so non-tune paths
are byte-inert), and splits scope the way C does: a new `rd_try_subblock`
primitive for AB/4-way, `rd_pick_rect_partition` kept on the parent fold
with `this_rdc.rdcost` recomputed there, and `encode_b_intra_dry` taking a
`refold_leaf_rdmult` flag. A failed leaf's rdcost must go through
`rd_cost_update`'s invalid guard — bare `rdcost()` wrapped `INT_MAX` rate
into a value that fed `evaluate_ab_partition_based_on_split` and regressed
a 4:2:2 s3 cell (caught by `self_contained_key_frame`, fixed before
landing). Quality knobs are now **75/75**; `self_contained_key_frame`
447/447 and `encoder_gate_e2e_byte_match` 32/32 — no non-tune regression.

## The whole `--deltaq-mode` axis closes — three per-SB-qindex plumbing bugs (KB-58, 2026-09-12)

All nine `deltaq` quality-knob pins are closed; modes 2/3/6 x cq{20,44} x
s{0,3,7,8,9} are byte-identical vs real aomenc. Three distinct bugs, one
theme — the pre-pass derived the per-SB qindex but downstream consumers read
stale/frame state:

1. `pack_tile_from_trees_lr` recomputed modes 2/3 with the VarianceBoost
   formula — the wire delta diverged from the search's quantizer. One shared
   `DeltaQFrameCtx::sb_qindex` mode dispatch now serves the pre-pass, the
   search pack and the repack.
2. `setup_delta_q_nonrd` (encodeframe.c:246-285) deadzone-quantizes against
   `xd->current_base_qindex`, which stays at the frame base for the whole
   tile (deferred token emit — `av1_update_state` never advances it). The
   port advanced it. `DeltaQFrameCtx::nonrd` freezes the adjust base; the
   nonrd rdmult fold (`av1_get_cb_rdmult`, gated `!use_nonrd_pick_mode` at
   partition_search.c:621-624) is skipped too — nonrd RD evals run at frame
   `RDMULT` while the quantizer rows follow the SB qindex.
3. `x->qindex` is the SB's adjusted qindex and the SEARCH reads it
   (`num_win_thresh` at partition_search.c:4033 — the observed
   `HORZ_4`/`HORZ_B` flip; the qidx rect prune at partition_strategy.c:1742;
   tx early-skip thresholds at tx_search.c:189/225). `sb_pick_cfg.qindex`
   now carries `sb_current_qindex` per SB instead of the frame base.

Plus: `av1_choose_var_based_partitioning` rebuilds its thresholds per SB
from `base + delta_qindex` (var_based_part.c:1683-1690) —
`choose_var_based_partitioning_key` takes `sb_qindex` (closed the mode-6 s7
cells, unpinned but divergent). Quality knobs **69/75**; the open residual is
the six chroma-delta-q ramp cells and the 84-cell tune bundle.

## The `CDEF_ADAPTIVE` halve/zero-low cells close — the thresholds read the mapped qindex, not the cq dial (KB-57, 2026-09-12)

The four `cdef-adaptive {420,mono} cq{20,60}` quality-knob pins are closed.
`av1_cdef_search` gates its adaptive arms on `cpi->oxcf.rc_cfg.cq_level`
(pickcdef.c:850/:927) — which `set_encoder_config` fills with
`av1_quantizer_to_qindex(cq_level)` (av1_cx_iface.c:1256), a 0..=255 QINDEX.
The port compared the raw 0..=63 dial, so it took the off arm for every cq in
9..=32 (C searches there: `qindex <= 32` is cq <= 8) and halved strengths for
cq >= 56 (C's `qindex <= 220` is cq <= 55). cq 8 and cq 40 agreed by
coincidence, which is why exactly 20/60 were pinned. One-line fix:
`CdefAdaptive.cq_level` now receives `rc::quantizer_to_qindex(cfg.cq_level)`.
Verified with the new `dump_tools_cell` example (the tools test's `planes()`
generator through `ref_encode_av1_kf_cfg`): cq {8,20,40,55,56,60,63} — 55/56
straddle the halve boundary exactly — 420 and mono, all byte-identical.
Quality knobs now 60/75; the delta-q residual named here was closed by
KB-58 later the same day — the open residual is the chroma-delta-q ramps —
all payload divergences (first diff = frame OBU size), all pinned.

## The `--enable-cdef=1` speed >= 4 header divergence closes — `cdef_pick_method` was never set past LVL1 (KB-56, 2026-09-12)

`PIN_cdef_speed4` — every `--enable-cdef=1` cell at `--cpu-used` 4..9 diverged
`HeaderOnly` in `cdef_strengths` (e.g. 64x64 s4: port 12/12, C 16/34) — is
closed. `SpeedFeatures::set_allintra` only ever assigned
`CDEF_FAST_SEARCH_LVL1` (speed >= 1); C's `speed >= 4`/`:497` LVL3, `>= 6`/
`:558` LVL4 and `>= 7`/`:572` PICK_FROM_Q writes had been filed "CDEF off in
the allintra envelope" before `--enable-cdef=1` was wired in and never
revisited. The s>=7 arm needed `av1_pick_cdef_from_qp` ported — closed-form
quadratic polynomials over `ac_quant_QTX(base_qindex) >> (bd-8)`, no MSE
search (intra-only set only; `use_screen_content_model` needs an
`AOM_CONTENT_SCREEN` tune knob the port does not carry). The dispatch sits
inside `av1_cdef_search_adaptive` AFTER the adaptive `cq_level <= 32`
early-off, matching C's ordering (pickcdef.c:846-857 precedes :866).
Verified byte-identical vs real aomenc: 64x64 s{4..9}, plus {64,100x60,128,
192,256,384,512} across cq {0,20,32,44} — FAST-search and FROM_Q arms both.
The CDEF-on sweep axis now runs speeds 0..9 (+12 cells); `PIN_cdef_speed4`
is removed. 447/447 standalone cells byte-identical.

## The whole `--cpu-used >= 7` VBP divergence closes — the phase-2 repack folded the per-SB rdmult modifier (KB-55, 2026-09-12)

`PIN_256x256_speed7` — every `--cpu-used` >= 7 frame above ~3x3 superblocks —
is closed. The port's own search produced `eob=0` at the divergent leaf while
its final encode emitted `eob=2` on a bit-identical residual: every trellis
input matched except `rdmult` (68,796 vs 42,997 = 68796*80>>7). C's
`intra_sb_rdmult_modifier` is recomputed ONLY in `av1_rd_pick_partition`'s SB
root (partition_search.c:5715); on the VAR_BASED_PARTITION arm it stays at the
per-SB reset 128 (encodeframe.c:1303), so `setup_block_rdmult`'s fold is
identity at speed >= 7. Phase-1 `pack_tile` modelled that guard; phase-2
`pack_tile_from_trees_lr` — the final bitstream emit, which runs at EVERY
speed — folded unconditionally under `allintra`, shrinking the repack trellis's
rdmult on any SB whose variance tripped the modifier gate. One-predicate fix:
the repack fold now carries the same `!use_var_based_partition` guard.
Verified byte-identical vs real C: {256..1024}² x s{7,8,9} cq32, 512² s7
cq{20,44}, 100x60 s{7,8,9}, 4160x64 s{7,8,9}, real-photo 12/12 including all
s9 cells. Both self-promoting pins flipped and moved into `sweep_cells`
(axis O + axis I s7..s9); `encode_perf_vs_libaom`'s s9 RD-assertion exemption
is removed. The KB-44 "newly measured" 100x60 s9 cell was the same root.
435/435 standalone cells byte-identical.

## A size-gated speed-0 divergence class closes — C's `winner_mode_params` snapshot was modelled as a live write (KB-54, 2026-09-12)

The `>=1080p && qindex<=108` speed-0 arm diverged from real C on every large cell
tested (mirror-tile >=1536², 3840x2160, a real 4K photo) while <=1024² stayed
byte-exact — the `is_1080p_or_larger` boundary exactly. Root cause is a C
call-order quirk: `winner_mode_params` is memcpied from the speed-feature
levels at speed_features.c:2800 inside `framesize_independent`, and the
`tx_domain_dist_level = boosted ? 1 : 2` write at :2928 runs later in
`qindex_dependent` — **dead for the frame**, so C runs those cells on the
level-0 row (tx-domain distortion OFF) while the port indexed the post-bump
level-1 row. `SpeedFeatures` now carries `tx_domain_dist_{level,thres_level}_
copied`, snapshotted at the end of `set_allintra`; `tx_type_search_policy`
indexes the tables by the snapshot. Verified byte-identical at 1024²/1536²/
2048²/2560²/3072²/3840x2160 cq27 s0 and a real 4K photo (634,991 B both arms);
57/57 `encoder_gate_*`. En route the `dump_kf_stream`/`eprof_x86`/`eprof_yuv`
C arms were found feeding `ref_encode_av1_kf` (palette/IntraBC hard-OFF) the
port's tools-ON config — now `ref_encode_av1_kf_screen_content` with the port's
resolved knobs, and the palette path is bit-exact under the matched oracle.
Env-gated hunt tooling kept in-tree (`AOM_TX_DBG`, `AOM_PART_DBG`,
`AOM_SCT_DBG`, `AOM_HDR_TRACE`/`AOM_HDR_DUMP`); the matching C-side prints are
`docs/upstream-instrumentation/2026-09-12-kb55-58-traces.patch` with the submodule reverted
to pristine. The `PIN_256x256_speed7` nonrd arm closed the same day as KB-55.

## The publish window is still OPEN — none of the four names are taken, and the facade has no consumer (2026-09-10)

Measured, not assumed: `zenav1-aom`, `-dsp`, `-encode` and `-decode` all return
404 from the crates.io API while `zenavif` returns 200. **Nothing is published,
so the published set is still fully revisable — and it stops being revisable at
the first `cargo publish`.** The user's constraint ("crates cannot be deleted
from crates") therefore binds the FUTURE, not the present; the whole value of
deciding now is that the decision is still free.

**`zenav1-aom` — the facade — has no consumer.** zenavif is the only external
consumer and it depends on `zenav1-aom-decode` and `zenav1-aom-encode` DIRECTLY
by git rev (`Cargo.toml:123-126`); `zenav1_aom::` appears nowhere in its source.
That is precisely the case the new publish gate was written to catch, live in
the tree right now.

**It is deliberately NOT resolved, because the argument runs both ways and the
losing move is unrecoverable either direction:**
* don't publish it — a crate with no consumer is the mistake the pin exists to
  prevent, and NOT publishing is reversible where publishing is not;
* publish it — `zenav1-aom` is the PRIMARY name, and publishing the three
  suffixed crates while leaving the unsuffixed one free lets someone else take
  the brand, which is equally unrecoverable.

Left in `PUBLISHED` (the name-defence reading) because that is the status quo,
with the tradeoff recorded in the gate's own doc comment so a publisher meets it
at the point of decision. **Resolve before publishing, not after.**

The other four workspace crates (`aom-bench`, `aom-dsp-bench`, `aom-sys-ref`,
`aom-target`) are `publish = false` and stay that way — they are test/bench
infrastructure, and `aom-dsp-bench` carries an explicit "do not fold this back
into aom-dsp" rationale (aom-dsp dev-depends on the C oracle, so an in-crate
bench could not build without the libaom submodule + cmake).

## The public-API snapshots were enforced NOWHERE, so all four had drifted — and the publish scan found three crates that could not be published at all (2026-09-10)

`docs/public-api/` and `just api-doc-check` have existed since 2026-09-09. They
appeared in **neither `.github/workflows/*.yml` nor `just gate-landing`**. An
unenforced snapshot is a document, not a gate, and it behaved exactly like one:
the first run of the check as a gate failed on **all four crates**.

**The headline number is the `__internals` gating for `aom-encode`, landed in
the same change: `zenav1-aom-encode.txt` goes 4,305 lines -> 281.** All 77
`pub mod` lines become `pub mod key_frame;` (the one module zenavif calls) plus
an `impl_mods!` macro over the other 76, `pub` under a default-OFF `__internals`
feature and `pub(crate)` without it — the pattern `aom-decode` landed first,
including the non-gated `tests/internals_feature_guard.rs` that stops
`required-features` from silently skipping the whole integration suite (KB-42).

**The drift was not only that landing's.** `aom-dsp` had quietly accreted this
cycle's `predict_intra_high_in_place` family (KB-PERF-56) and
`aom_quantize_b_no_qmatrix`'s new `iscan` parameter (KB-PERF-57); `aom-decode`
had `ReconPlane` and the avx512 feature. None of it was reviewed as a surface
change, because nothing asked.

**The crates.io half found three real blockers, confirmed with cargo rather than
inferred from metadata.** `zenav1-aom`, `zenav1-aom-encode` and
`zenav1-aom-decode` each carried an intra-workspace **path dependency with no
version requirement**; `cargo publish --dry-run` rejects that outright (*"all
dependencies must have a version requirement specified when publishing"*).
`zenav1-aom-encode` also had no `repository`. Fixed in the manifests, and the
scan additionally pins each requirement to the member's actual version so it
cannot drift silently to the next publish.

**Why the gate is a metadata scan and not `cargo publish --dry-run`:** a first
publish is bottom-up, and cargo resolves a downstream crate's requirements
against the real index — so `--dry-run` on `zenav1-aom-encode` cannot succeed
until `zenav1-aom-dsp` is actually on crates.io. That failure is inherent, not a
defect, so it cannot gate. Everything cargo checks BEFORE it touches the index
can, and all three defects lived there.

**The published SET is pinned by name, because a crates.io name is permanent.**
The stakes are asymmetric: a broken manifest costs a retry; publishing a name
nobody meant to own, or a surface nobody reviewed, cannot be undone. So a crate
joining `PUBLISHED` takes an edit, not merely dropping `publish = false`.
Bite-proved five ways — four perturbations fail exactly one test each, while "a
`publish = false` crate becomes publishable" fails FOUR at once, which is the
right shape for the one mistake that is unrecoverable.

`api-doc-check` is now the fifth step of `gate-landing` and its own CI job. It
needs no C oracle, no conformance corpus and no codec build: **7.7 s**, which is
why it can sit in a per-landing gate. Gate green: 1508/1508 in both dispatch
modes, census 4/4, whereat 4/4, api-doc 6/6.

**Found while wiring the CI job, pre-existing:** three `run:` lines in `ci.yml`
ended a plain scalar with `:` (`-- whereat_entries::`), which is invalid YAML
that GitHub's lenient parser happens to accept — `yaml.safe_load` refuses the
file at `HEAD`. Quoted, so the workflow can now be validated locally before
being pushed.

## `aom_quantize_b_no_qmatrix` walks raster order, not scan order — −0.134 % pooled, and the 11.89x kernel is now vectorisable in principle (2026-09-10, KB-PERF-57)

The lever map's "biggest single un-taken kernel left" (+28.7 ms at 11.89x, the
only addressable row with no SIMD at all). Re-profiled at HEAD first: 0.86 % =
25.7 ms, matching the ranked 26.9; `objdump` confirms 217 instructions, zero
ymm, zero xmm, 14 `cmp`, 8 panic-call sites and two real `memset` calls.

**A throwaway counter decided the design.** C's pre-scan trims trailing
dead-zone coefficients, so scan order runs `non_zero_count` iterations where
raster order runs `n`. Measured on the shipping cell: 538,094 calls per 1 MP
encode, mean n = 60.8, mean non_zero_count = 32.7 — **raster order pays 1.86x
the iterations**, and has to buy that back.

It buys it back by deleting both `memset`s (every position is written
unconditionally; C's pre-scan and main-loop dead-zone tests are exact
complements, so the set of positions that WRITE is unchanged), every bounds
check, the `scan[i]` load, the gather/scatter, and the per-coefficient class
select — C's `ac = (scan[i] != 0)` is asking whether the RASTER position is the
DC, which in raster order is `i != 0`, so the DC peels off and five per-class
constants become loop scalars. The EOB comes from `iscan`, exactly as
`quant/simd.rs` already does for `quantize_fp` and as libaom's own
`aom_quantize_b_avx2` does (its `iscan` argument exists for this; the `_c` body
ignores it).

**Measured: −0.134 % pooled over two independent 30-round bands, 40/60,
p = 0.0135** (A −0.169 %, B −0.106 %; pooled null −0.013 %). Pooled because A
was marginal and B alone is not significant; the pooled median sits between the
two band medians, so no favourable band was selected. That is ~4.0 ms of 25.7 —
16 % of the kernel.

**Reported as a trade, not a removal:** it removes two memsets, all bounds
checks and the indirection but adds arithmetic on 0.46n extra coefficients. It
measured positive and significant, but only just. The reason to keep it is as
much structural — the kernel is now in the only shape a vector tier can take.
What blocks that tier is `t2 * quant_shift` reaching ~2^35 (the rest of the
chain is provably i32-safe); libaom clears it with 16-bit arithmetic, which this
project already rules structural for `quantize_fp`.

Bite proof aimed at the eob (KB-12): deriving it from the raster index instead
of `iscan` fails exactly one test, on the eob, while 37 other quantize/txb tests
stay green. `just gate-landing` 1507/1507 twice.

## The intra predictor stopped writing into a scratch the caller then copied back — −0.50 % at the shipping preset, −0.79 % at speed 0, byte-identical (2026-09-10, KB-PERF-56)

libaom's `av1_predict_intra_block_facade` hands the predictor `pd->dst`, so the
prediction lands in the reconstruction plane directly. The port's two-slice
`predict_intra_high` made every encoder call site fill a tight `txw * txh`
scratch and copy the block back row by row — and
`benchmarks/encoder_lever_map_s3_2026-09-10.md` had annotated `intra_model_rd_y`
and found that copy is its **hottest single instruction**, at the encoder's
highest trip count: per txb PER CANDIDATE MODE.

`predict_intra_high_in_place` is additive. Each of the three highbd builders now
splits into a `plan_*` half that performs EVERY read of the reference plane
(above/left/corner into owned local arrays, plus the directional degenerate
arm's two corner reads) and a `write_*` half that touches no reference pixel —
so reads complete before writes and aliasing the two slices cannot change a
value. The two-slice entry points are unchanged, which keeps the DECODER out of
the blast radius.

**Five call sites converted, two deliberately not.** `intra_model_rd_y`,
`intra_model_rd_uv`, both arms of `predict_uv_txb` and
`nonrd_pick_intra_mode` drop the scratch, the memset and the copy; their
consumers (subtract, the speed-9 SAD prune, the CfL DC cache) read the plane
strided instead. `txfm_rd_in_plane_intra` and `encode_intra_block_plane_*` were
NOT converted: their prediction feeds `dist_block_px_domain` per TX TYPE, which
needs a tight `w * h` buffer, so predicting in place there would relocate the
memcpy rather than remove it.

**Measured** (rotated arms, same-binary null, byte-length identity checked on
four cells first): **1024x1024 cq27 s3 −0.497 %, 28/30 rounds, p = 8.7e-07**
against a null of +0.080 %; **512x512 cq27 s0 −0.793 %, 24/24, p = 1.2e-07**
against +0.042 %. It pays MORE at speed 0 because speed 0 evaluates more
candidates per transform block — a different overhead class from the
per-transform-call setup, which pays LESS at faster presets.

New gate `predict_intra_in_place_diff` asserts the WHOLE PLANE is byte-identical
to predict-then-copy over every mode x angle-delta x filter-intra mode x 19 tx
sizes x 6 availability combinations x bd {8,10,12}, and that the block still
equals the real exported C predictor. `just gate-landing` 1507/1507 twice.

