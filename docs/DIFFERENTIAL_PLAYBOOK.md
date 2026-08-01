# Differential playbook — how work gets validated in this repo

**Why this file exists.** The techniques below were each derived the expensive
way, usually after something passed green while testing nothing. They are not
style preferences; every one of them has a specific failure it prevents, and the
failure is named. Read this before adding a gate, porting a kernel, or
diagnosing a divergence.

Companion docs: `PARITY.md` (what is proven), `CLAUDE.md` (Known Bugs +
coordination rules), `docs/LIBAOM_UPSTREAM_NOTES.md` (libaom's own quirks),
`HANDOFF-TOGGLES.md` (the sibling-C dump recipe).

**Citation audit, 2026-07-31.** Every in-repo reference here was independently
re-checked by a session that did not write the doc. Corrections are inline;
the substantive ones were a misattributed doc (§7's "31 decoder cells" lives in
`DECODER_CONFIG_COVERAGE`, not `CONFIG_PERMUTATION_DESIGN`), a symmetrised
benchmark figure (§6's "±16%" is a one-sided +16.5%), an undercount (§10's
"nine single-flag reverts" is ten flags), a stale path in the §10 recipe, and
three §11 cost anchors that have **no second record anywhere in the repo** and
are now marked as such.

---

## 1. A test that cannot fail is worse than no test

**The failure it prevents:** the transform SIMD carried a `neon` tier for months
that was entirely dead code on aarch64, while its differential passed green —
because `for_each_token_permutation` silently excluded `neon` (a
compile-time-guaranteed token archmage refuses to disable), so the test compared
the scalar path against itself and reported `simd_perms=0`. Note the helper
itself is `archmage::testing` (registry dep, not defined in-repo); the in-tree
record of the exclusion and its fix — archmage's `testable_dispatch` dev-feature
— is `crates/aom-dsp/src/dispatch/mod.rs:163` with the guarding assertion at
`:167-173`.

**Before shipping any gate, break the thing it guards and watch it fail.** Then
revert and confirm `git diff` is clean. Quote the failure message in the commit.

**The strong form — asymmetry.** When a gate exists because a hole was
suspected, the proof is not that your gate fails; it is that your gate fails
*while the existing suite stays green*. Examples from this repo:

- perturbing `delta_lf_from_base` → new decoder gate fails in 0.02 s, **all 25
  pre-existing decoder gates stay green**;
- mis-threading a knob combination → new permutation gate fails 21 of 27, while
  `toggles_rd_close` stays **25/25**;
- a SIMD-only splat perturbation → the i32 arm's test fails and the i16 arm's
  passes, and vice versa, proving each test reaches its own arm.

If the pre-existing suite *also* fails, you have not found a hole — you have
found a regression, which is a different (and more urgent) report.

**Watch for inert perturbations.** In `cdef_find_dir` the obvious change
(`- 128` → `- 127`) is provably a no-op, because `DIV_TABLE[n] = 840/n`
(`upstream/av1/common/cdef_block.c:67`, and the "output is then 840 times
larger, but we don't care for finding the max" comment at `:64-66`) normalises
a DC shift identically across every direction. It passed, which is
what forced building a proper accept-observer. If your perturbation passes, ask
whether it was reachable before concluding the gate is broken.

**When one landing fixes several things, revert each ONE ALONE — and compare
which cells fail.** This is what distinguishes "two roots" from "one root plus a
redundant change", and it has now paid twice in a row:

- KB-21 root #2: reverting the rate-composition defect alone failed the envelope
  on `q00_128 cpu-4` + KB-13's two 128x128 cpu4 cells; reverting the quantizer
  switch alone failed those four **plus** both noise gates' cq63 pairs.
- KB-21 root #3 (QM x speed>=4): reverting the QUANT_PARAM half alone failed the
  same 12 cells with the same byte counts as the pre-fix map; reverting the
  trellis half alone failed a **different, smaller set of 7**.

**Different cell sets = genuinely different roots.** Identical cell sets are a
warning that one of your two changes may be doing nothing, or that they are two
spellings of one fix — investigate before claiming both. Note also that a
per-root bite proof is strictly stronger than one bite proof for the whole
landing: the latter passes even if one of your changes is inert.

## 2. Non-vacuity must be asserted, not assumed

`assert!(report.permutations_run >= 2)` counts **permutations**, not **vector**
permutations — it is satisfied on a machine with no vector tier at all. Seven
differentials shipped with exactly that assertion and were technically vacuous.
(Live `permutations_run` assertions: `crates/aom-dsp/tests/cdef_find_dir_simd_diff.rs:309`,
`cdef_filter_simd_diff.rs:130`, `quantize_fp_simd_diff.rs:210`,
`cdef_lowbd_simd_diff.rs:235`/`:265`, `wiener_simd_diff.rs:130`.)

The correct form counts permutations in which a vector tier is live, and is
per-architecture (`NeonToken` on aarch64, `X64V3Token` elsewhere — testing only
`X64V3Token` counts every aarch64 permutation as scalar, because that token is a
stub off x86):

```rust
if if cfg!(target_arch = "aarch64") { NeonToken::summon().is_some() }
   else { X64V3Token::summon().is_some() } { simd_perms += 1; }
...
assert!(simd_perms >= 1, "...");
```

**Do NOT add a pre-flight `summon()` check outside the harness.** It looks like a
non-vacuity guard and is an ordering trap: under `AOM_FORCE_SCALAR=1` the pin
disables every runtime token *process-wide*, so on x86 `summon()` correctly
returns `None` until `for_each_token_permutation` resets that state. Tests that
fire the pin first — the documented order — fail on the linux scalar-pin CI leg
while passing on aarch64. Removed in `854b2ac` ("fix(test): drop the pre-flight
AVX2 summon check — it broke the linux scalar-pin CI leg"); there is a note at
each site.

## 3. Verify on both targets, from whichever box you have

`cargo check --target x86_64-apple-darwin` (or the reverse) catches an entire
class of error the host build cannot see. Real instances:

- a nested `incant!` whose default tier list looks for a `_v4` variant the
  family does not have — invisible on aarch64, a hard error on x86;
- the pre-flight `summon()` trap above, which is `#[cfg(target_arch = "x86_64")]`
  and does not even compile on ARM.

`cargo check` is not enough for C shims — it does **not** recompile shim C for a
cross target. To verify a shim change on the other target, generate that
target's libaom config and compile the TU against it, then check `nm` for the
symbols you expect and, crucially, the ones you expect to be *absent*.

## 4. When you cannot run the other target, predict it

The strongest available substitute for executing on a target you lack: derive a
falsifiable prediction on the box you have, then let CI adjudicate.

Worked example (KB-20 root #4). The x86 arm of an ISA-conditional kernel could
not be executed on an ARM box. Instead: instrument the `_c`/NEON arm to count,
per cell, how many `aom_hadamard_16x16` outputs leave `int16` — the quantity the
hypothesis says drives x86 divergence — and check it against the observed x86
failure set. Result: **7/7 recall on the divergent cells, 22/24 overall**, the
two misses false-negative by construction. CI then confirmed.

That is a mechanism understood before confirmation, not a constant tuned until
CI went green. Prefer it to iterating on CI blind, which costs ~30 min a cycle.

## 5. Self-promoting pins for divergences you cannot close yet

A known divergence is pinned with its **exact current state**, and the test fails
in *both* directions — if it regresses, and if it silently starts matching. The
second half is the point: when someone later fixes the root, the pin fires and
tells them to re-pin rather than letting the fix pass unnoticed.

Live examples: KB-17's screen-content set, KB-22's ≥2160p residual (both
CLAUDE.md), `SCREEN_ARRAY_OPEN_ROWS`
(`crates/aom-bench/tests/config_permutations.rs:1806`, re-pin message at
`:1886`), `size_axis_open_divergences_pinned` (`:2899`).

**Never** resolve a divergence by widening a tolerance, adding `#[ignore]`, or
gating on `target_arch` so nothing is asserted. If a contract genuinely differs
per target, write *two* tests with two explicitly-stated contracts — see
`crates/aom-encode/tests/hog_prune_diff.rs`, where the x86 test asserts
bit-equality against an AVX2 kernel (`hog_nn_predict_matches_avx2_and_dispatch`,
`:99-101`) and the non-x86 test asserts lattice membership + mask parity
(`MAX_MASK_FLIPS = 0`, `:216`) + a pinned count of differing lanes
(`MAX_ONE_QUANTUM_LANES = 56`, `:207`)
(`hog_nn_predict_agrees_with_dispatch_within_one_prec_quantum`, `:189-191`).
Rule of thumb: **if you cannot state what the test
proves in one sentence without the word "approximately", you have written the
banned version.**

## 6. Benchmarks: report the control band first

On this hardware a **same-binary re-run** moved one row by **+16.5%**
(`intra::v_16x16`, `benchmarks/dsp_neon_i16_2026-07-28.tsv:46`, discussed at
`benchmarks/dsp_neon_i16_2026-07-28.md:26`), and sub-30 µs cells swing ±10–29%.
A single row's delta therefore proves nothing. *(This said "±16%" until
2026-07-31; the measurement is one-sided +16.52% — the symmetric form was a
rounding, not a second observation.)*

- Always run untouched cells as a negative control and quote their spread
  *alongside* the result.
- Read whole size sweeps, not individual rows.
- `AOM_FORCE_SCALAR=1` is **not** a scalar baseline on aarch64 — `neon` is a
  compile-time-guaranteed token that cannot be disabled outside test builds, so
  a pinned/unpinned pair measures the same code twice (a 77-row pair came back
  inside ±3%: noise). Use a two-**build** pair instead.
- Take object-code evidence from `--release`: `test-fast`'s `overflow-checks`
  de-vectorizes `#[autoversion]` kernels (`sad_simd`: 0 NEON ops under
  test-fast, 37 under release —
  `benchmarks/simd_reach_neon_census_2026-07-28.tsv:4`,
  `docs/SIMD_REACH_AUDIT_2026-07-28.md:83`).
- Commit `benchmarks/<thing>_<YYYY-MM-DD>.{md,tsv,meta}` with git commit, host,
  command line, and `uptime` load.

## 7. Config-permutation gates: collapse, don't enumerate

Architecture in `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md`; the load-bearing
ideas:

- **Effective-config collapse.** Hash the *resolved* internal state
  (`SpeedFeatures` + `PackCfg` + header bits) and keep one representative per
  signature. Validate the engine by checking it re-derives the known-inert cases
  in `HANDOFF-TOGGLES.md` rather than hardcoding them. Raw cartesian 14,155,776
  → 777,600 effective configs, a 13.7× collapse
  (`docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md:64-66`).
- **Independence must be measured.** A four-corner (`{A0B0, A0B1, A1B0, A1B1}`)
  footprint experiment over all 210 axis pairs found **zero independent pairs**
  — on an intra encoder every knob feeds the same RD loop. Do not assume
  orthogonality; the answer here was that there is none. Method at
  `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md:164-175`, verdict at `:336`,
  and all 210 rows in `benchmarks/config_perm_independence_2026-07-30.tsv`
  (117 INTERACTING, 39/28/3 INERT-*, 22 SIGNALLING-ONLY, 1 ILLEGAL — no row
  carries an `INDEPENDENT` verdict).
- **Cells compose.** Byte-identity is all-or-nothing over the pipeline, so one
  cell with N features live covers every pairwise interaction among them. 31
  decoder cells cover ~120 crossings
  (`docs/DECODER_CONFIG_COVERAGE_2026-07-30.md:138` — *this figure is not in
  `CONFIG_PERMUTATION_DESIGN`; corrected 2026-07-31*).
- **Coverage counts must not mix derived with replayed axes.** The port never
  authors a sequence header, so seq-level axes are replayed from a bootstrap
  stream, and a gate asserting "the seq bit equals the knob" is an agreement
  check, not evidence the port can produce that configuration.

**The thesis, empirically:** bugs live in the gap between two individually-green
PARITY rows. KB-20 was a *panic* sitting between "cpu-used 8/9 byte-identical"
and "bd10/bd12 byte-identical", each true on its own grid, never crossed.

## 8. Derive coverage from artefacts, not from names

A test called `..._superres_...` that decodes a stream with `use_superres=0`
covers nothing. Build coverage matrices by **parsing the bitstreams** (or
asserting the derived config field) and counting what is actually exercised.

Doing this revealed that the AV1 intra conformance corpus — Gate 1's authority —
is a deep sweep of *one* sequence shape: **235/235 carry 4:2:0 subsampling
flags** (2 of those are monochrome), bd8 (169) or bd10 (66) only, with zero
superres, tiles>1, QM, segmentation, `reduced_tx_set`, `disable_cdf_update`,
4:2:2, 4:4:4 or 12-bit
(`benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`; every column
re-tallied 2026-07-31, all as claimed — the `mono=1` pair is the only nuance).

Corollary: assert liveness per axis. Two knobs (`--enable-cfl-intra=0`,
`--enable-tx64=0`) were covered hundreds of times on the primary context
**without ever being exercised**
(`docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md:308`).

## 9. Distrust in-tree comments claiming a feature is inert

`part_sf.early_term_after_none_split` was omitted from the port under the
comment *"(C) INERT on this path (byte no-op, verified) — NONE always yields a
valid rd on textured content"*. It fires at **6 nodes in a single 64×64 frame**
of real photographic content (`crates/aom-encode/src/speed_features.rs:171`;
repro cell `av1-1-b8-00-quantizer-00` cropped 64×64@(64,64) mono,
`--cpu-used=4`). A previous session verified inertness against synthetic
textured content and generalised. The offending comment is no longer in the
tree — it was at `crates/aom-encode/src/speed_features.rs:698` and was deleted
by the fix, so cite it as `83de077^:crates/aom-encode/src/speed_features.rs:698`.

When a comment says "verified inert", check *what it was verified against*. The
same applies to handoff notes describing what is missing: KB-20's assert named
one bd8-specific step and there were **three**, the two it omitted being the
substantive ones (CLAUDE.md:1722-1735;
`crates/aom-encode/src/nonrd_pickmode.rs:1222` marks the second).

## 10. Diagnose to the decision, not to the byte count

A byte delta is a symptom. Drive to the **first divergent block** and compare the
decisions on both sides. The exemplary close (KB-21 root #1) went further and
proved everything *around* the divergence agreed — the NONE arm to the unit, the
entire SPLIT accumulation child-for-child, the remaining budget entering the last
child — leaving exactly one BLOCK_8X8 leaf where C early-terminates and the port
falls through.

**Never infer the mechanism from the SIZE of the delta.** KB-22's ledger entry
reasoned that 150 bytes over ~1,156 superblocks — 0.035% of the payload — "argues
near-tie, not a missing tool", and (to its credit) flagged that as unestablished.
It was wrong. The localization put the first divergence at **node 1**: the first
32x32 of the first superblock, real aomenc choosing `PARTITION_VERT_B` where the
port chose `SPLIT`, with the whole frame differing in shape (33,371 vs 29,105
tree nodes). The cause was an entire unmodelled speed-feature pass
(`av1_set_speed_features_qindex_dependent`). A systematic search-configuration
difference can wear a tiny byte delta, because two different-but-comparable
search outcomes cost nearly the same number of bits. Small delta means "the two
encoders agree about most of the picture", NOT "the two encoders nearly agreed
about one decision".

Method: the sibling-C dump in `HANDOFF-TOGGLES.md:42-46` (ar-swap an
instrumented `libaom.a`, run the pinned cell, **revert everything**). Verify the
revert by byte-comparing the restored archive against a pristine backup. *Stale
path warning (2026-07-31): the recipe names `reference/libaom/build/libaom.a`,
but `build_dir` now derives from `upstream/build`; `reference/libaom` is only a
gitignored fallback (`reference/BUILD_CONFIG.md:2-3`). The mechanism is right,
the path is not. Repointing `build.rs` is also unnecessary when the instrumented
TU replaces its member in `upstream/build/libaom.a` directly — `ar r` +
`ranlib`, then `cargo clean -p zenav1-aom-sys-ref` to force the relink, since
the build script's SHA-stamped cache will not rebuild libaom over your swap.*

Rule the search space down by A/B rather than by intuition: KB-21's root-2 entry
recorded **ten** single-flag reverts across eleven settings that did *not*
reproduce C's numbers. That A/B was what proved the divergence was not "one
speed-4 sf field is set wrong" — and it was right: the two actual roots were a
rate-composition rule and a per-tx-type quantizer switch, neither of them an sf
field, so no flag revert could ever have reproduced C.

**Then dump the layer below the one you localized to.** KB-21 root #2 sat at
"same leaf, same winner mode, same tx_size, different rate and dist" for a day.
What closed it was dumping the per-txb (entry state, per-candidate, winner)
triple on both sides and aligning them by group index: 51 groups at the suspect
block, agreeing on 11 and splitting on the 12th, with the entry state
(`allowed_tx_mask`, `block_sse`, `block_mse_q8`, `qstep`, `rdmult`,
`ref_best_rd`) byte-equal — which localized the fault to the *ordering* the
candidate loop walked, not to any quantity it computed. Print the loop's inputs
alongside its outputs; an identical input set with a different output ORDER is a
much sharper signal than a rate delta.

**And when a fix moves the number the wrong way, suspect a half-applied
change.** Switching the SATD arm's quantizer kind to B *increased* the error
(dist 48017 → 152297 against C's 57169) because `QuantParams` bakes the
quantizer's table choice: the B algorithm ran on FP's tables. A "fix" that
overshoots in the direction you intended is usually correct-but-incomplete, not
wrong.

## 11. Environment facts that have cost time

- `conformance/data` is a **plain gitignored directory** as of `ae1c93d`
  ("fix(repo): untrack the conformance/data symlink — it handed every fresh
  worktree a fake baseline"; `.gitignore:7` is now slashless).
  Populate with `python3 xtask/conformance.py --fetch --scope intra` or set
  `AOM_CONFORMANCE_DIR`. (It was previously a *tracked symlink* to a path
  existing only on the original box, because `.gitignore` said
  `conformance/data/` **with a trailing slash** — which matches a directory but
  not a symlink. Every fresh worktree started with ~10 unrelated failures and
  three agents in one session mistook that for their own baseline.)
- `git worktree remove` refuses on worktrees containing submodules — use
  `--force`, after pushing.
- The repo is **not** rustfmt-clean (419 diffs in aom-dsp alone — **NOT
  ESTABLISHED**: checked 2026-07-31, this count has no second record in the
  repo; re-run `cargo fmt -p aom-dsp --check` before quoting it). Do **not** run
  `cargo fmt`; it buries the change.
- CI runs `--profile test-fast`, not debug (`.github/workflows/ci.yml:83`,
  `:121`, `:235`). The debug profile put the x86-64 leg at 5h47m and then 6h00m
  — GitHub's hard cap, reported as "cancelled" (recorded at
  `.github/workflows/ci.yml:23`). Coverage is identical (`debug-assertions` and
  `overflow-checks` stay on); it is ~17× faster.
- Cost anchors for budgeting: ~60 ms/cell for a 64² e2e byte-identity cell with
  the C oracle; per-cell cost falls steeply with speed (114 ms at speed 0 → 2 ms
  at speed 9) and rises steeply with frame size (~12.6 s/cell at 480p, ~250 s at
  2160²). Only the frame-size pair is corroborated
  (`docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md:738`, `:797`); the 60 ms and
  114 ms→2 ms figures are **NOT ESTABLISHED** — checked 2026-07-31, they appear
  nowhere else in the repo (no `benchmarks/*` row, no commit message). Use them
  as order-of-magnitude only, and re-time before budgeting on them.

## 12. A green unit differential does not clear the code that composes it

**The failure it prevents:** KB-21 root #2 was a rate-composition defect —
`prune_txk_type_intra` added the tx-type signalling cost to `eob == 0`
candidates, where C adds it only on the `eob > 0` path
(`upstream/av1/encoder/txb_rdopt.c:742-744` early-returns
`txb_skip_cost[ctx][1]`; the cost is reached only inside
`warehouse_efficients_txb_laplacian`, `:674`). The port sorted its candidate list
by signalling cost where C sorted by distortion, so `txk_map` came out in a
different ORDER and the two encoders evaluated different candidate sets.

**The DSP differential for that cost function stayed green the entire time** —
it deliberately scopes the tx-type cost out, so it could not see the defect. The
kernel was right; its caller composed it wrongly.

So: a passing unit differential licenses the kernel, and **nothing else**. When a
divergence survives a green unit suite, do not re-examine the kernel — examine
what the caller does with its result: the order it walks candidates, which
branch it adds a cost on, which parameter block it hands down. Ask specifically
*what the differential scopes out*, because that is where the surviving bug is.

Corollary, from the same close: **ruling out every flag does not rule out the
mechanism.** Ten single-flag reverts across eleven settings failed to reproduce
C's numbers, which correctly proved "no speed-4 sf field is set wrong" — and
that proof is compatible with two real defects, because neither root was a flag.
An exhaustive A/B over the wrong axis is still exhaustive, and still tells you
nothing about the right one.

## 13. Re-deriving a config from its base constructor silently drops every later pass

**The failure it prevents:** KB-26. The port's speed>=4 winner-mode two-pass
(`wm_parts`) built its tx policies from a **fresh** `SpeedFeatures::set_allintra`,
which reproduces only C's framesize-INdependent cascade. Anything resolved later —
`set_allintra_speed_feature_framesize_dependent`,
`av1_set_speed_features_qindex_dependent` — is silently back at its default in that
copy. So `prune_tx_type_using_stats` arrived as 0 on every frame >= 480p, and the
whole luma tx search ran with the prune off. C never has this problem: it reads one
frame-level `cpi->sf` in `get_tx_mask` (`tx_search.c:1887`), identical across all
three stages.

The signature is distinctive and worth recognising: **a setting that is provably
load-bearing at one speed and provably inert at the next, on the same frame.** That
is not a threshold moving — it is a code path that stopped seeing the value. In
KB-26 speeds 0-3 used the caller's resolved policy and never entered `wm_parts`,
which is exactly where the split fell.

**Rule:** carry the resolved frame-level value down; do not reconstruct it. The fix
was one function — `TxTypeSearchPolicy::carry_frame_level_tx_sf` — replacing two
hand-copied lines.

**Audit before adding any internal re-derivation.** Done for this tree on
2026-08-01: production encoder code has exactly two `set_allintra` call sites, and
the second (the palette derivation, `partition_pick.rs:1050`) is safe because every
upstream assignment to both fields it reads is inside a `*_framesize_independent`
function. Note *how* that was settled — by enumerating the assignments in upstream
source, not by reasoning about what the fields "probably" are. A field's name tells
you nothing about which pass sets it.

