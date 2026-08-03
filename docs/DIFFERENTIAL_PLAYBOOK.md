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

**A kernel with no differential is the same hole, and it hides best when the
kernel is nearly-invariant.** KB-12 sat pinned for sixteen days as "a genuine
leaf-mode near-tie" in the nonrd estimate arm. It was `hadamard_lp_8x8` dropping
the trailing transpose `aom_hadamard_lp_8x8_c` performs at
`upstream/aom_dsp/avg.c:232-236`, so the port's coefficients were the exact
TRANSPOSE of libaom's. Two things kept it invisible:

- **every consumer except one is order-invariant.** `aom_satd_lp` and
  `av1_block_error_lp` are sums over the whole array; `eob == 0` is a set
  property; `eob == 1` can only mean the DC, which is the transpose's fixed
  point. Rate, distortion and skippability were all CORRECT. Only `eob` moved —
  on 477 of 4,000 blocks — and only through `eob_cost += get_msb(eob + 1)`;
- **the unit tests that existed were transpose-blind.** `hadamard_lp_8x8_flat`
  fed a constant block, which puts all the energy at coefficient 0. It passed
  the whole time. A test whose input is symmetric under the transformation you
  got wrong cannot fail.

So when choosing a probe for a kernel, ask what symmetry the input has, and pick
one the suspected defect breaks. And when a kernel is hand-transcribed rather
than reused, lock it against the exported C symbol even if "the code matches the
C line-for-line" — that phrase is in KB-12's own ledger entry, written about
this function.

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
- **A fixed-order interleave confounds ARM with POSITION, and on this box the
  position is worth as much as the effect.** Measured 2026-08-03
  (`benchmarks/encoder_intra_smooth_paeth_2026-08-03.md` §4): two copies of ONE
  identical binary sitting at round positions 5 and 6 came out **0.34 pp apart**
  while the copies at positions 1 and 2 agreed to 0.11 pp; pooled over all arms
  the gradient across a round is **1.7 %**. The same drift is on record for
  `windows-11-arm` (`winperf_content_census_2026-08-03.md` §5), where it is
  corrected for by *pooling* the copies on each side after the fact.
  `scripts/eprof_ab.sh` now takes **`ROTATE=1`** (default off, adds a `position`
  column; `eprof_ab_stats.py` reads both TSV shapes), which rotates the arm
  order one step per round so every arm spends `N/k` rounds in each position and
  the drift cancels by construction instead of by a correction. **Use it for new
  bands**; a null arm that disagrees with its own twin is the symptom.
- **A same-binary null arm beats an argument about noise.** Run the winning
  build twice, under two labels, from two copies of the identical executable,
  interleaved with everything else. Its delta IS the box's noise floor, measured
  rather than asserted, and it is the only thing a real delta can honestly be
  read against on hardware you do not own (`scripts/winperf_ab.py`; nulls came
  in at 0.02–0.20 % on three boxes where raw spreads ran to 16 %).

## 6b. A performance number is scoped to the platform it was taken on

**The failure it prevents:** KB-PERF-2's headline. `−1.34 ms` for the
per-txb scratch-reuse lever was recorded as *the* value of the change. It was
the value on an Apple M4 Pro against Apple's allocator. Re-measured on Windows
(`benchmarks/winperf_windows_2026-08-02.md`) the same lever is worth **~5x more**
— **−2.38 % / −2.54 %** against Darwin's **−0.49 %** — and it goes from **21 %
of the landing to 86–99 %** of it, swapping rank with its sibling lever. The
allocator *call counts* are byte-identical on all three platforms, so nothing
about the code differs; only the **price per call** does.

- **Levers whose mechanism is a call into the platform** — allocator traffic,
  syscalls, TLS, atomics, unwinding — have a per-platform price and their
  one-box numbers do not generalise. Levers whose mechanism is arithmetic or
  memory bandwidth (here: the sibling `memset` lever, −2.02 % on Darwin against
  −0.50 % on a runner) travel much better. Name the mechanism, then decide
  whether one box is enough.
- **"Not measured on platform X" is not "measured and the same."** Before this
  study, `windows-11-arm` CI *built* the crates and ran nothing, so every
  Windows performance statement about this encoder was vacuous. A build job is
  not a measurement job.
- **Take the platform comparison on ONE harness, not by quoting two studies.**
  The comparison here is the same generated content, the same fixture, the same
  five arms, on all three boxes — never "Windows synthetic vs Darwin
  photograph". An integer-only source generator is what makes that possible;
  the identical censuses across three targets are the proof it worked.
- **And content can move the same lever as much as the platform can.** On one
  box in one afternoon, lever 3 measured −1.26 % / −0.49 % / +0.31 % on three
  contents. Bracket the content or say which one the number belongs to.
- **A synthetic harness source is only representative of the axis it was tuned
  on — check before reading a band from it.** `winperf`'s `detail` / `smooth`
  were tuned so their **allocator call count** brackets the study photograph's,
  which made them the right harness for KB-PERF-2 (allocation) and KB-PERF-3
  (the forward transform) — both touch every block regardless of coding mode.
  They are **not** representative of MODE DISTRIBUTION: measured 2026-08-03,
  directional intra prediction is **20.8 %** of predicted pixels on the
  photograph, **13.2 %** on `smooth` and **0.15 %** on `detail`, where `z1`
  fires six times in a whole 1 MP frame. KB-PERF-4's Windows band was therefore
  reading a structural zero, not a platform result. A null you cannot
  distinguish from "the code never runs" is not a measurement.
- **The census is committed; run it, do not reason about it.**
  `cargo run -p zenav1-aom-bench --features census --example content_census --
  winperf:<name> …` reports intra mode family x transform size, forward
  transform type x size, coded leaf size, **and (since 2026-08-03) every other
  coding-tool family the encoder can enter**: filter-intra, palette Y and UV,
  intraBC, the UV mode distribution (so CFL is a count), the per-plane split of
  the intra predictor, the CFL predictor itself, rect-vs-square leaves, small
  leaves and angle deltas. Sources may be a harness content, a raw `.yuv`, a
  raw `.yuv` bootstrapped through the SCREEN config (`scr:`, so the frame
  header can carry `allow_screen_content_tools`), or a conformance vector
  decoded back to pixels (`real:`, i.e. literally the differential corpus).
  `benchmarks/winperf_family_census_2026-08-03.md` is the committed corpus x
  family table; `benchmarks/winperf_content_census_2026-08-03.md` is the
  earlier mode-distribution one. Before quoting a band for a lever scoped to a
  family, look your family up; a `0.00` there means the code under test does
  not run and any null you measure is structural.
- **"Unreachable" has three different causes and they need different fixes.**
  The 2026-08-03 family census separated them, and only ONE of the three is a
  content problem:
  * **speed-gated** — filter-intra is `0.00` on every source at the harness
    cell, and no content can change that: `--cpu-used 6` sets
    `prune_filter_intra_level = 2`, i.e. `rd_pick_filter_intra_sby` is never
    called. On the SAME content it reads 10.46 % of leaves at cpu-used 5. A
    family in this class needs a different CELL, not a different image.
  * **knob-gated** — palette and intraBC are `--enable-palette` /
    `--enable-intrabc`, both default OFF, and additionally need
    `allow_screen_content_tools` in the frame header, which real `aomenc` sets
    from its own detection. A band for either is a statement about a
    non-default encoder and has to say so.
  * **content-gated** — CFL is the clean example: 0.00-0.29 % of
    chroma-reference leaves on the three synthetic winperf contents against
    4.59 % on the study photograph and 23.02 % on a CLIC image. Nothing is
    gating it; the synthetics simply do not have the luma-chroma correlation.
    This is the class a new content fixes.
- **Check whether something ALREADY reaches your family before generating
  content for it.** The differential corpus (`EncodeCell::real_content`, KB-13's
  cells, at cq32 / cpu-used 0) reaches filter-intra at **21-31 % of leaves**,
  directional prediction at 51-54 % of predicted pixels, rectangular leaves at
  74-82 % and 4-pt transforms at 67-75 %. The byte gates were never blind to
  those families; only the PERF harness was, and only because of its cell. A
  `real:`/`yuv:` census is minutes of work and can retire the question.
- **Fitting content is legitimate; fitting it to the OUTCOME is not.** The fix
  for the above was a third content (`winperf::Content::Photo`) fitted by
  sweeping 467 candidates against **the reference's mode distribution**, with
  the objective (L1 over the intra-class pixel-share vector) declared before the
  sweep ran and the lever's delta measured exactly once afterwards
  (`crates/aom-bench/examples/content_fit.rs`). Turning the knob until a lever's
  number looks good produces a harness that measures the knob — that is §14
  wearing a content costume. Two consequences worth carrying: an optimum sitting
  on a grid EDGE is not an optimum (two extra passes here), and when the
  landscape plateaus, take the argmin of the DECLARED objective rather than
  quietly re-ranking on a secondary one.
- **A content fitted on one axis will drift on the others — keep both, do not
  replace.** `photo` matches the photograph's mode mix (L1 5.7 pp against
  `detail`'s 47.4) but its allocator traffic fell to **73 %** of the reference
  where `detail`'s is **95 %**. Neither content dominates; the harness runs the
  one whose census reaches the lever's mechanism, which is now a
  `contents:` input on the workflow rather than a fixed pair.

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

**Make the coverage question a compile error.** KB-29's decode gate does not
carry a hand-written list of which knobs need guarding — a no-`..` destructure of
`ToggleKnobs` means adding a field breaks the build until someone classifies it,
and the gate then RECOMPUTES "unguarded" as "flipping this knob emits no
`aome_enc_control_id`". Measured answer: **7 of 31 knobs were unguarded**
(`enable_palette`, `enable_intrabc`, `disable_tx_stats_prune`, `delta_lf_mode`,
`qm`, `deltaq_mode2`, `deltaq_mode3`). The destructure paid for itself
immediately — it caught `deltaq_mode2`/`deltaq_mode3`, which the author's hand
list had missed. A list of what to cover goes stale the day someone adds a field;
a compile error cannot.

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

**The strongest form of the trap is a comment that cites the right C line and
still concludes wrongly.** KB-32: `var_part`'s module doc said
`force_large_partition_blocks_intra` *"is 0 on this path — it only rises at
allintra speed>=8/720p+ (speed_features.c:327)"* and dropped both of that
field's arms. The citation is correct; the conclusion was an artefact of the
envelope — no gate in the tree had ever run `--cpu-used >= 8` above 640 px, and
the port had shipped speeds 8 and 9 since KB-12. Both of GitHub issue #7's
reported size bands were those two arms. So when a comment justifies an omission
with a *gate* (a speed, a size, a bit depth), check whether the gate is
unreachable **in the format** or merely unreached **by the current cells** — and
say which. The same file's *second* claim, that the VBP tree "stamps squares
8x8..64x64 only", was false for the same reason and only surfaced once the first
fix made the HORZ/VERT arms win.

**And the fix for THAT wrote a third one, in the same commit.** KB-32 replaced
the "squares only" comment with a refusal whose own note read *"REACHABILITY,
MEASURED 2026-08-01: of 18 large cells probed at speeds 8 and 9 (768² through
5472x3648), NONE reach a non-square leaf. The only cell in the tree that does is
issue #6's 12000x9000 at cpu9."* It was measured, it named its grid, it was
honest about being a measurement — and it was wrong by **four orders of
magnitude in area**: KB-34's sweep found 627 reaching cells, the smallest a
**100x100 thumbnail**, and 768² — the first size inside the note's own stated
range — reaches at cq32 cpu9. Two sessions had already contradicted it (KB-28
at 0.9 MP, the encoder profile at 1024²) before anyone re-derived it.

Two lessons, both cheap:

- **"Measured on N cells" is not a reachability result; it is a statement about
  those N cells.** A reachability claim needs a *predicate* — something of the
  form "reachable exactly when X" that a reader can falsify on one new cell. The
  18-cell probe had no predicate, so nothing about it was checkable.
- **When you do find the predicate, try to break it before writing it down.** The
  first hypothesis here (`min(w,h) >= 720`, i.e. the very speed feature KB-32 was
  about) fitted the entire first sweep and is FALSE — 1272x716 reaches 22 leaves.
  What actually governs is whether the frame has a partial superblock, which the
  size-shaped hypothesis could never have expressed. The straddle test that would
  have refuted it costs two encodes.

## 10. Diagnose to the decision, not to the byte count

A byte delta is a symptom. Drive to the **first divergent block** and compare the
decisions on both sides. The exemplary close (KB-21 root #1) went further and
proved everything *around* the divergence agreed — the NONE arm to the unit, the
entire SPLIT accumulation child-for-child, the remaining budget entering the last
child — leaving exactly one BLOCK_8X8 leaf where C early-terminates and the port
falls through.

**A decoder's error message names the first check that FAILED, not the defect.**
KB-29's stream was rejected by aomdec and dav1d with `Invalid intrabc dv`. The
obvious reading — and the one the task brief was written around — is that the
encoder emitted a DV violating the spec's validity constraints, so: check the
wavefront lag, the sb_size-relative reference area, tile clamping, the
delta-left-of-SB rule. **All four measured clean.** The port's `is_dv_valid` is
differential-locked against C's and its inputs matched the decoder's exactly at
every diverging site.

What actually happened was a tile-payload DESYNC six roots deep (a missing chroma
coefficient write, a stale txfm-partition context, a palette leaking onto an
IntraBC winner, plus three decoder-side walk defects). Once the bitstream is
desynced, a later block reads garbage, and `use_intrabc` + a garbage DV diff is
simply the first thing libaom hard-errors on. The message pointed at the tripwire,
not the cause.

So treat a conformance rejection as "the stream went wrong at or before here",
never as "the named field is what is wrong". The method that found it: instrument
BOTH sides of the same stream with a per-block dump plus a running symbol count
and `od_ec_*_tell`/range, align the walks, and find the first block whose symbol
DELTA differs (a desync) or whose range differs at equal count (a CDF-index
divergence). That distinction tells you which of the two failure modes you have
before you know anything else.

**The same trap sprung a second time, on the DECODE side, and was settled by a
one-line revert rather than by more DV analysis.** GitHub #5 (KB-33) reported the
decoder rejecting 37 of 100 conformant SVT-AV1 streams with `intrabc DV failed
validity`, and reasoned — from the fact that `is_dv_valid` is
differential-locked — that the *reconstructed* DV must be at fault, i.e. that
`find_dv_ref_mvs` picks a different reference DV on SVT-reachable neighbour
configurations. Wrong again: the cause was KB-29's missing/misordered
coefficient reads, and the cheapest proof was not a `dv_ref` audit but
**reverting each candidate root ALONE and watching the exact reported message
come back with the DV code untouched**. When you inherit a hypothesis about
which *value* is wrong, first ask whether the reported check is downstream of a
desync — and prefer a revert-one-root experiment over any amount of reading, it
is minutes of work and it is decisive.

Corollary on where these bugs hide: **a differential is blind to inputs no
available encoder produces.** Every intrabc conformance vector is
libaom-encoded, and this port's own gates decode its own encoder's output, so
neither could reach the block/neighbour shapes that tripped KB-29's decoder
roots — a *third* encoder found them. When a decoder path is guarded only by
streams from encoders you also implement, that path is unguarded in exactly the
way §1 means. Fix it by pulling in a foreign encoder
(`crates/aom-bench/tests/svt_interop_decode_gate.rs` does this with real SVT-AV1
C encodes), and check what the foreign encoder *cannot* reach either: SVT emits
no IntraBC block above BLOCK_64X64, so KB-29's >64×64 multi-chunk residual is
still uncovered and is recorded as such rather than counted as breadth.

**Never infer the mechanism from the delta's SHAPE either.** KB-12's residual
was sign-random, under one byte per superblock, flat in area, confined to
leaves, with the partition trees agreeing to 45,780 nodes and every leaf field
equal except `y_mode` — and all four candidate modes inside the estimate arm's
own four-entry `intra_mode_list`. Three sessions read that as proof of a genuine
tie; the KB-32 gate even asserted the shape. It was a dropped transpose in a
kernel with no differential. Every one of those observations was true and none
of them was evidence about the mechanism: a defect that perturbs one small
additive term in an N-way comparison produces exactly that shape. The shape
tells you *where* to look (the estimate arm's rate composition), never *what*
is wrong.

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
- **`zenav1-aom-bench` has a default-ON `c-oracle` feature.** With it off the
  crate still compiles — `EncodeCell` + `port_encode*` + `aom_bench::winperf` —
  with NO C toolchain, no cmake, no nasm and no `upstream/` submodule, which is
  how the encoder runs on the Windows CI runners. Everything else in that crate
  requires the feature (all ~40 differential tests, the bench, `gate3_profile`,
  every C-comparing example, and `zensim`). `--no-default-features` is supported
  for `--lib` and the `winperf*` examples only; the tests are NOT gated, so
  build those with default features as CI does.
- **The encoder CAN be run on Windows now** — `.github/workflows/winperf.yml`,
  `workflow_dispatch`-only (so it costs nothing per push), both
  `windows-11-arm` and `windows-latest`, the two jobs in parallel. MEASURED:
  **8m35s** at `rounds=12` on one content, **11m35s** at `rounds=16` on two —
  four release builds of aom-dsp+aom-encode+aom-bench dominate, the timing
  bands are a few minutes. Before 2026-08-02 the only Windows job was
  `portability`, which builds and executes nothing.
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

## 14. A profile's ranked table and its stated attribution limits can contradict each other

**The failure it prevents:** KB-PERF-2. The encoder re-profile ranked
`alloc/libc` at **+24.76 ms, 20.9 % of the gap**, and the follow-up landed a
−40 % reduction in allocator calls (854,053 → 512,557) for **−1.34 ms**. The
ceiling was **18x optimistic**.

The reason was already written in the same document, in its own "Attribution
limits" section: `alloc/libc` is a **leaf class matched by symbol name**, and most
of its mass is `_platform_memset` (5.36 % of window) plus `_platform_memmove`
(2.68 %) against the allocator's own `xzm_free` (2.34 %). Removing malloc/free
pairs does not remove the bytes those buffers still get zeroed with. The caveat
and the ranked table were both correct; they were about different quantities, and
**the ranked table is the part people act on.**

**Rules that follow:**

- When a profile ranks a *class* rather than a function, read what the class
  actually matches before sizing a lever off it. "Allocation" that is mostly
  `memset` is a zeroing cost, and the fix for zeroing is not reuse.
- A lever's projected ceiling should name the **mechanism** it removes, not just
  the stage it targets — "removes N allocator calls" and "removes N bytes of
  zeroing" have different ceilings even when the profiler puts them in one row.
- Read a profile's caveats section as part of its ranking, not as boilerplate
  after it. If a caveat undercuts a row in the table, fix the table.

**The companion positive result:** the sibling lever (3a, tiering the forward
transform scratch) projected +5.30 ms and delivered −3.13 ms — right order,
because its mechanism was named exactly (a 4 KiB memset per forward transform at
every size, against tiered inverse passes in the same file). Same profile, same
day; the difference is whether the projection was tied to a mechanism or to a row.

**It happened again, one landing later, at 5x.** KB-PERF-3 took the same
profile's `dsp:transform` row (+19.63 ms) and delivered **−3.84 ms**. Two
measurements, taken rather than argued, say where the rest was: the row had been
sampled BEFORE the sibling memset lever took −3.13 ms out of the same two
functions, and a temporary counter in both pass entry points showed **only
51.6 % / 55.1 % of forward-transform calls are even eligible** for a 16-lane
batch (the rest are 8-dim). So the rule generalises past allocation: **when a
lever cannot reach a whole stage, COUNT what it reaches before quoting the stage
total.** A call census is cheap — one atomic per call, thrown away afterwards —
and it converts "why did we only get a fifth of it" from speculation into a
table.

**And measure the obvious extension rather than reasoning about it.** The same
landing's follow-up — run the ineligible 8-dim blocks as half-idle 16-lane
batches — had a clean a-priori argument (equal op counts, a much cheaper
`half_btf`) and measured **−0.006 % against a +0.08 % null**. It was built in
full, differentials and all, and reverted. An argument about instruction counts
is not a measurement of them, and on this hardware the load/store path around a
kernel routinely eats what the kernel saves.

**And measure the levers separately even when you ship them together.** Here 3a
alone was −3.128 ms with **zero** allocator-call change (its scratch is a stack
array and cannot appear in a census — proven by the two censuses being identical
to the digit), while lever 3 alone was −1.339 ms with −341,496 calls. Summed:
4.467 vs 4.641 measured. Had they been shipped as one change, the 18x-optimistic
projection would have been quietly absorbed by the honest one.

