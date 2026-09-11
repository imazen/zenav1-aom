# zenav1-aom — project handoff (2026-08-03)

Current, verified state of the port for a new developer and/or a new machine.
Everything below was checked against `origin/main` on 2026-08-03; where a claim
has a proof artifact, it is cited. Older handoff snapshots are superseded by
this file. Deep technical state lives in [`STATUS.md`](STATUS.md) (module log,
newest first), [`PARITY.md`](PARITY.md) (stills-parity ledger),
[`CLAUDE.md`](CLAUDE.md) (coordination rules + the Known Bugs ledger), and
[`PORTING.md`](PORTING.md) (Rust module ↔ upstream C map).

**Read these two before your first change** — they exist so the expensive
lessons are not re-derived:

- [`docs/DIFFERENTIAL_PLAYBOOK.md`](docs/DIFFERENTIAL_PLAYBOOK.md) — how work
  gets validated here. Each rule names the specific failure it prevents (a
  differential that passed for months while comparing the scalar path against
  itself; a non-vacuity assert satisfiable with zero vector permutations; a
  benchmark claim inside the noise band). If you are adding a gate, porting a
  kernel, or chasing a divergence, start here.
- [`docs/LIBAOM_UPSTREAM_NOTES.md`](docs/LIBAOM_UPSTREAM_NOTES.md) — libaom's
  own ISA divergences, undefined behaviour, and surprising-but-intended
  behaviour. Same source really does produce different results per target in
  several places, and libaom's own cross-tier tests do not catch them because
  they exercise the domain where the tiers agree. Check here before concluding
  the port is wrong.

## Fresh-box setup

```sh
git clone --recurse-submodules https://github.com/imazen/zenav1-aom.git
cd zenav1-aom
cargo test          # builds the C oracle once (needs cmake, nasm, a C compiler), then the differential suite
python3 xtask/conformance.py --fetch --scope intra   # decode-conformance vectors (gitignored)
```

- The `upstream/` submodule (the pinned libaom v3.14.1 C oracle, `03087864`)
  resolves from **github.com/imazen/libaom-mirror** — a pure daily-updated
  mirror of `aomedia.googlesource.com/aom`. The mirror syncs via this repo's
  `.github/workflows/mirror-libaom.yml` (05:23 UTC cron; pushes over a
  repo-scoped deploy key stored as the `LIBAOM_MIRROR_SSH_KEY` secret).
- `just test` / `just test-scalar` / `just test-fast` / `just bench-gate3` wrap
  the common flows. `AOM_FORCE_SCALAR=1` forces every SIMD kernel through its
  scalar twin — the full suite must pass in BOTH dispatch modes.
- **Box-local dependency (the one thing not in git):** the Gate-3 wall-bench
  vectors `conformance/data/mosaic-{2k,4k}-cq{20,40}.ivf` are gitignored.
  Regeneration: `benchmarks/mk_mosaic_y4m.rs` tiles 25 gb82 576×576 lossless
  photos into a y4m (sources on the 2026-07 dev box at `/root/mosaic-sources/`;
  the y4m was verified byte-identical to the pre-wipe original), then encode
  with the aomenc settings recorded in `benchmarks/gate3_*.meta`. Without them
  the bench's mosaic cells skip via `from_vector_opt` and the conformance cells
  still run. If you are migrating boxes: copy `/root/mosaic-sources/` (or the
  four `.ivf`s) across, and back them up off-box.

## Where the four gates stand

- **Gate 1 — decoder byte-identity: intra scope DONE.** Bit-identical to C
  across the CI-wired intra conformance corpus (byte-identity + golden MD5),
  incl. q62/q63, SB128 (230/235) and film grain (2/235). **Corrected 2026-07-31:
  this line previously also claimed superres and multi-tile; the corpus contains
  ZERO of either** — every vector's frame 0 was parsed
  (`benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`) and the corpus is
  233/235 4:2:0 + 2 monochrome (the subsampling flags read 4:2:0 on all 235,
  but 2 carry `mono=1`), bd8/bd10 only, with no superres, tiles>1, QM, segmentation,
  `reduced_tx_set`, `disable_cdf_update`, 4:2:2, 4:4:4 or 12-bit. Those axes are
  covered by the PORT-GENERATED gates (`real_bitstream`,
  `config_permutations_decode`), not by conformance — see
  `docs/DECODER_CONFIG_COVERAGE_2026-07-30.md`. Inter-frame decode is in progress
  through a single-reference feature ladder (concurrent track).
- **Gate 2 — encoder byte-identity: DONE for ALLINTRA across --cpu-used 0-9**
  on the synthetic grids, and on real conformance-decoded content at speed 0
  (KB-6, 30/30) plus **58/60** at speeds 1-4 (KB-13; was 45/60 — KB-21's three
  roots and KB-23 promoted 13 cells on 2026-07-30/08-01). **The two still open are
  both `cpu3 cq63`; speed 4 is 12/12 on every cell.** Non-default stills knobs
  (QM, CDEF search, LR search, SB128, multi-tile, film grain, lossless,
  10/12-bit, tune=IQ/SSIMULACRA2, deltaq modes, toggles) are byte-exact —
  see PARITY.md section A. Open cells are pinned by self-promoting gates
  (a fix flips the gate red → promote the cell; nothing can silently drift).
- **aarch64 (added 2026-07-25):** the transform vector path now runs on ARM —
  one `#[magetypes(v3, neon, -scalar)]` body per kernel, the whole
  architecture-dependent surface in `transform/simd/prims.rs`. 30 of 34
  transform bench cells improved 18–66% on an Apple M4 Pro
  (`benchmarks/dsp_neon_transform_2026-07-25.md`). Two things to know before
  working here: **as of 2026-07-30 KB-ARM-FLOAT is CLOSED and the full workspace
  is 875/875 on ARM** (this line previously said 755/770 with 15 pre-existing
  float failures — all four roots are fixed, and CI's `test-macos-aarch64` leg
  now runs `--workspace` in both dispatch modes); and `AOM_FORCE_SCALAR=1` is a
  no-op for the NEON tier outside test builds, so never read a pinned/unpinned
  bench pair as scalar-vs-SIMD on ARM.
  When you touch anything under `transform/simd`, check BOTH targets
  (`cargo check --target x86_64-apple-darwin` catches tier-list errors the host
  build cannot).
- **Gate 3 — performance.** DECODER: see below, ≤1.5× met at 4K only. ENCODER (new,
  2026-08-02/03): **10.66× → 3.15× vs libaom** across five byte-identical levers — see "The
  ENCODER performance programme" section for the levers, the three scoping dimensions, and the
  rotation caveat on three of the headlines.
- **Gate 3 (decoder) — the user-set ≤1.5× bar is met at the 4K headline
  cells** — 4K decode wall ≈1.22× C (cq20) / ≈1.19× (cq40) after the bd8 lowbd
  u8 pipeline, the i16 transform column+row narrowing, and the
  deblock/wiener/CDEF filter work. 2K and small-frame cells still exceed the
  bar (1.66–1.9× at 2K, up to ~2.4× on tiny/entropy-dominated cells). See
  `benchmarks/gate3_peak_wall_2026-07-25.md` for the committed run + caveats,
  the `bd8_*`/`gate3_*` series for per-lever Ir attribution, and
  `benchmarks/gate3_filters_2026-07-22.md` for the ranked remaining levers.
- **Gate 4 — coverage/integration:** `coverage-audit/COVERAGE.md` is the gap
  matrix; the zenavif integration contract is specced in CLAUDE.md ("Zen codec
  cross-cutting compliance") — DecodeError/limits/stop/alloc landed, probe +
  estimate still open there.

## What changed on 2026-07-31 / 08-01 (ten KB closures — read this before trusting older notes)

A large sweep closed KB-21 (3 roots), KB-22, KB-23, KB-24, KB-25, KB-26, KB-29
(6 roots), KB-31 (2 roots) and KB-32 (2 roots), plus GitHub issues #6 and #7.
Anything written before 2026-07-31 about the encoder's speed/size envelope is
likely stale. The headlines:

- **Every one of those bugs came from measuring an axis nobody had measured** —
  above 640 px, QM on at speed>=4, partial superblocks, >=720p, screen tools
  armed, frames that REQUIRE a tile split. Speed 4 looked closed on the morning
  of 07-31; it was closed only on what had been swept. Treat the "still
  unmeasured" list in each KB entry as a work queue, not as reassurance.
- **Three separate blind spots were doc claims of inertness that source did not
  support** (playbook §9): `early_term_after_none_split`, the `var_part` field
  "is 0 on this path", and KB-22's "unmodelled LR fields". Each cited a correct C
  line and drew a conclusion true only of the envelope that had been run.
- **The decoder gained two real fixes from encoder work**: superres x
  multi-tile-column (previously refused outright) and a tile-info re-derivation
  that mis-parsed any real stream with `log2_cols > min_log2_cols && min_log2 > 0`
  (KB-31 root #2 — no conformance vector reaches it).
- **A whole class of gate was missing and now exists**: `armed_tools_decode_gate`
  round-trips any encode with a non-default tool armed through the C decoder and
  dav1d. Byte-identity to a reference proves nothing for configs the reference
  never encodes; 7 of 31 `ToggleKnobs` were unguarded, and its coverage is a
  compile error rather than a list.
- **Cross-encoder position is now measured**, including libaom C:
  `benchmarks/xbench_2026-08-01.md`. At a >=1 MP/s budget the port has **no
  coding deficit vs libaom on photographs** (identical BD-rate, 9/10 images
  byte-identical at every quantizer) — its gap is that it runs 10.8x slower at
  matched cpu-used. On screen content there is a real gap, and it is a SPEED
  problem: IntraBC is correct now but costs 558x libaom's time for that tool.

## What changed on 2026-08-03 (two more of the same class, and the queue is now in one place)

Working the "still unmeasured / still refused" lists those ten closures left behind found two
more bugs of exactly the same shape, and neither needed new machinery:

- **KB-35** — the nonrd estimate arm's palette refusal tested `allow_screen_content_tools`
  alone, **one of four terms** of C's `try_palette`. `--cpu-used 8` therefore refused to encode
  a plain smooth gradient at every size >= 1024x1024 and every quantizer, on cells where the C
  oracle passes `--enable-palette=0` and libaom provably never enters the palette search. 22 of
  25 measured PANIC rows are now byte-identical; the 3 that remain are the genuine arm. An
  over-broad REFUSAL is exactly as wrong as an over-broad inertness claim — it just fails
  loudly.
- **KB-36** — `default_min_partition_size`'s `speed >= 6 && is_1080p_or_larger` arm was
  unmodelled, so every >= 1080p frame at `--cpu-used 6` searched 4x4 partitions C had stopped
  at 8x8 (-127 B at 1920x1080, +79 B at 2560x1440, with 1920x1072 byte-exact). The window is
  **one speed wide** — speed 7 sets the same field framesize-independently — so a speed sweep
  at any size under 1080 and a size sweep at any speed but 6 are both green with it missing.
  It is the SECOND arm found on that field; KB-19 modelled the first and its entry said so.
- **The consolidated queue now lives in `CLAUDE.md` under "Coverage queue"**, ranked by
  (reachability x blast radius), with a cost estimate per remaining axis. It was scattered
  across ~15 KB entries and gate-file doc comments, which is why nobody had worked it. Keep it
  current in the same commit as the landing.

## The ENCODER performance programme (2026-08-02/03) — 10.66x -> 3.15x, all byte-identical

Gate 2 finished with zero pinned cells, so the encoder's remaining gap was speed. It had
**never been profiled** (every profiling artefact in `benchmarks/` was decode-side). Five
levers landed, each byte-identical to libaom — Gate 2 kept zero pinned cells throughout:

| | vs libaom (Darwin, 1 MP photo, cq44, cpu-used 6) |
|---|---|
| start | **10.66x** |
| KB-PERF-1 cache the intra-mode CNN per 64x64 (C does; the port re-ran it at every node) | 3.36x |
| KB-PERF-2 per-txb scratch reuse + tiered forward-pass scratch | 3.25x |
| KB-PERF-3 i16-lane bd8 forward transform | 3.19x |
| KB-PERF-4 vector directional intra predictors | 3.187x |
| KB-PERF-5 u16-lane SMOOTH kernels | **3.150x** |

**Read `benchmarks/encoder_hotspot_reprofile_2026-08-02.md` before picking the next lever, and
read its two scoping banners first.** The single most reusable finding of this programme is that
a profiler's ranked share is scoped three ways and none of them are obvious:

- **platform** — allocation is worth ~5x more on Windows than Darwin (identical call counts on
  both, so the whole difference is cost per call), and the top-ranked lever is ARM-only and does
  nothing on x86-64;
- **content** — the same lever reads 20.8 % on the study photograph and 0.15 % on `detail`;
- **speed** — directional intra is 20.78 % of predicted pixels at cpu-used 6 and **56.61 % at
  cpu-used 5** on identical content.

So a share is a measurement of one cell, not a property of the encoder. Relatedly: **the port's
ratio is WORST where nothing is profiled** — 5.64x at cpu-used 9 and 7.76x at cpu-used 4 against
3.15x at the profiled cell.

Three projections overshot badly (18x, 5x, 13x) and one landed within 8 %. The difference is
recorded as playbook §14: a ceiling tied to a **named mechanism** holds; one tied to a
profiler's ranked *stage* does not, because the stage is a symbol-name leaf class that may be
mostly `memset`.

Supporting machinery, all reusable: `scripts/eprof_ab.sh` (+ `ROTATE=1`, and
`eprof_ab_position.py` for the position table), `.github/workflows/winperf.yml`
(`workflow_dispatch`, real Windows numbers on two targets), and `aom_dsp::census` behind a
default-off `census` feature with `just census-gate` pinning per-family floors AND ceilings.

**Re-verified 2026-08-03/04 — the answer, not the warning.** KB-PERF-2/3/4's Darwin headlines
were all taken with a fixed-order interleave, which confounds arm with position. All three were
re-taken under `ROTATE=1` from rebuilt, sha-verified binaries; record
**`benchmarks/encoder_rotate_reverify_2026-08-03.md`**:

| lever | published | rotated | verdict |
|---|---:|---:|---|
| KB-PERF-2 allocation | −3.05 % (n=12) | −2.99/−3.00 % (n=50) | **survives**, to 0.06 pp |
| KB-PERF-3 i16 fwd txfm | −2.56 % (n=24) | −1.89/−2.16 % (n=50) | **moves ~0.5 pp**, conclusion holds |
| KB-PERF-4 directional intra | −0.75 % (n=36) | **−0.64 %** (n=150, idle box) | **settled**, ~15 % smaller |

Quote those rotated numbers. Two things the re-verification established about the *harness*,
which bind any future band: (1) the 0.34 pp position confound and the "1.7 % gradient" were
**contended-box artifacts** — on an idle box, six copies of ONE binary over 150 rotated rounds
give a position gradient of **0.055 pp (means) / 0.172 pp (medians)** with no arm significantly
different from any other; (2) but in a band whose arms have *different* speeds, two copies of one
binary still disagreed by **0.270 pp at p < 0.0001, replicated in two independent bands** — so a
single arm's systematic floor is ~0.27 pp even rotated, well above its statistical MDE.
**`ROTATE` now defaults ON.**

## Live tracks (each has its own docs; concurrent sessions may be active)

- **Inter decode + encode** ("THE REST"): INTER-ROADMAP.md,
  INTER-ENCODE-ROADMAP.md, INTER-FEATURES-PLAN.md, INTER_DECODE_ENVELOPE.md,
  INTER-CHUNK{1,2}-HANDOFF.md. Encoder is at the single-ref translational
  P-frame skeleton stage (KB-16 in STATUS).
- **KB-15 intrabc** (screen content): CLAUDE.md KB-15 — six roots fixed, witness
  pinned at first-diff 1120. **Corrected 2026-08-03: this line called the residual
  "an mi(40,28) partition near-tie" and titled the track "coeff arm"; both are the
  pre-root-3 description.** Root 3 made the intrabc leaf RD equal C to the unit and
  mi(40,28) now picks VERT matching C; the coeff arm is wired end to end and its
  re-encoded txbs match C's coded tree. What is left is the block's PACK **symbol**
  coding (the DV diff / `use_intrabc` flag / `write_tx_size_vartx` txfm-partition
  split-flag context) at port 1886 B vs C 1891 B. Do not re-chase the DV search, the
  NN prunes, or the coeff re-encode.
- **Decoder robustness/fuzz**: fuzz harness landed, campaign notes in STATUS;
  keep the no-panic property as features land.
- **Encoder near-tie residuals — nearly all closed as of 2026-08-03.** KB-13 is at
  **58/60** (both remaining are `cpu3 cq63`; speed 4 is 12/12 on every cell).
  **KB-12 is CLOSED** — its "leaf-mode near-tie" was never a near-tie but a
  dropped transpose in `aom_hadamard_lp_8x8` (libaom's `_c` ends with one,
  *"Extra transpose to match SSE2 behavior"*, `avg.c:232-236`); Gate 2 went to
  **zero pinned cells** and `SPEED_OPEN_SINGLETONS`/`_COMBINATIONS` are both
  empty at every speed 0..9. **KB-10/KB-11's noise-cq63 pairs and the whole
  cpu-4/5 "fragile band" are CLOSED** (KB-21 root #2). What is left: 2 palette
  128² (KB-P29), 1 toggle cell, and the pins listed in the Coverage queue's T4
  tier. The sibling-C RD-dump method (KB-3/KB-7) is the standard close.

  **A lesson worth carrying from KB-12:** those five lowbd estimate kernels were
  the only hand-transcribed kernels in the tree with **no differential**, and the
  two unit tests that existed fed FLAT blocks — structurally transpose-blind,
  since flat input puts all energy at coefficient 0. Tests existed; they could
  not have caught it. Every consumer of those coefficients except the EOB is
  order-invariant, which is why a transposed kernel produced *correct* rate and
  distortion and wore a near-tie's clothes for weeks.
- **A red `interintra_diff` on aarch64 is now attributable.** It used to SIGSEGV
  intermittently and was recorded as a runner flake; it was a **real data race in
  the oracle** (`CONFIG_MULTITHREAD=0` selects the unsynchronised `aom_once`, and
  `av1_init_wedge_masks` opens by memsetting its own table, so every mask pointer
  reads NULL for the duration of the init). Fixed 2026-08-03 by funnelling all
  four `aom_once`-guarded initialisers through `ref_init`'s Rust `Once`; playbook
  §11 carries the full derivation. **Its regression gate is committed as an
  unproven probe, not a teeth-tested gate** — reverting the fix does not make it
  fail on the dev box, so establishing that it bites is an open item for CI's
  aarch64 runner, where the crash actually reproduces.

## Conventions that keep multi-agent work safe here

- jj (colocated) on `main` only; claim the repo with a `.workongoing` marker
  before ANY work; push via `jj bookmark set main -r @ && jj git push
  --bookmark main`; verify with `git merge-base --is-ancestor <sha> origin/main`.
  **Never** `jj git push --change` (orphans work off-branch).
- `conformance/data` is a plain gitignored directory. Populate it with
  `python3 xtask/conformance.py --fetch --scope intra`, or point
  `AOM_CONFORMANCE_DIR` at an existing copy — every consumer checks that env
  var first. (Until 2026-07-28 a *tracked symlink* lived at that path, aimed at
  `/root/aom-rs/conformance/data` — self-referential on the box it was made on,
  dangling everywhere else. `.gitignore` said `conformance/data/` with a
  trailing slash, which matches a directory but not a symlink, so it was
  committable. Every fresh worktree therefore started with ~10 conformance
  failures unrelated to whatever was being changed, and three agents in one
  session mistook that for their own baseline. The symlink is gone and the
  ignore pattern is slashless.)
- Differential-first: every kernel lands behind a diff test against the REAL
  exported C function (`crates/aom-sys-ref` shims); e2e byte-gates pin whole
  configurations; open divergences are pinned ASSERTED-PRESENT so fixes
  self-promote. Never weaken a gate.
- The C oracle build is deterministic single-threaded
  (`reference/BUILD_CONFIG.md`); `reference/libaom/` is a gitignored working
  copy — the tracked truth is the `upstream/` submodule.

## Recent history you might otherwise re-derive (2026-07-23 → 07-25)

- The "17 invalid AV1 streams" finding from
  `benchmarks/intra_tiebreak_deltas_2026-07-23.md` was **refuted**: a KB-13
  harness bug (floor(mi/16) SB walk + unpadded source dropped the partial edge
  SB). Fixed in `c08a4c1`; the encoder was correct; 196² cq63 promoted. The
  writeup carries a correction banner — trust the banner, not the old headline.
- Inherited-C hygiene: upstream LICENSE/PATENTS live in `upstream-notices/`;
  the only tracked `.c` files outside the submodule are our own oracle shims.
- bd8 peak-perf series landed: Phase A/B/C (u8 planes → u8 kernels → i16
  columns), then `9f49ebc3` (i16 rows) and `ea61406f` (CDEF find_dir), each
  validated 418/418 in both dispatch modes before push.
- All historical agent worktrees/bookmarks were audited landed-or-superseded
  and deleted on 2026-07-24/25; there is NO stranded WIP. If you find a
  `worktree-*` bookmark in the future it is new work, not archaeology.
