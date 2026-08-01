# zenav1-aom — project handoff (2026-07-25)

Current, verified state of the port for a new developer and/or a new machine.
Everything below was checked against `origin/main` on 2026-07-25; where a claim
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
  (KB-6, 30/30) plus **50/60** at speeds 1-4 (KB-13; was 45/60 — KB-21's two roots
  promoted 5 cells on 2026-07-30/31). Non-default stills knobs
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
- **Gate 3 — performance: the user-set ≤1.5× bar is met at the 4K headline
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

## Live tracks (each has its own docs; concurrent sessions may be active)

- **Inter decode + encode** ("THE REST"): INTER-ROADMAP.md,
  INTER-ENCODE-ROADMAP.md, INTER-FEATURES-PLAN.md, INTER_DECODE_ENVELOPE.md,
  INTER-CHUNK{1,2}-HANDOFF.md. Encoder is at the single-ref translational
  P-frame skeleton stage (KB-16 in STATUS).
- **KB-15 intrabc coeff arm** (screen content): CLAUDE.md KB-15 — six roots
  fixed, witness pinned at first-diff 1120 (an mi(40,28) partition near-tie).
- **Decoder robustness/fuzz**: fuzz harness landed, campaign notes in STATUS;
  keep the no-panic property as features land.
- **Encoder near-tie residuals** (shrunk sharply 2026-07-30/31): KB-13 is at
  50/60, KB-12's 4 speed-8 diag estimate-arm cells stand, 2 palette 128²
  (KB-P29) and 1 toggle cell (HANDOFF-TOGGLES.md holds its localization notes).
  **KB-10/KB-11's noise-cq63 speed-6/7 pairs are CLOSED** (by KB-21 root #2), as
  is the whole cpu-4/5 "fragile band" on bd8 — KB-21 is closed and the RD-search
  speeds 0..7 now have NO open singleton in the speed axis; the only two left are
  nonrd (speed 8). All remaining cells are pinned self-promoting; the sibling-C
  RD-dump method (KB-3/KB-7) is the standard close.

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
