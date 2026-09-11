# Cycle ledger — 2026-09-08 11:40Z to 2026-09-11 03:30Z

Reconstructed 2026-09-11 from the session transcript (21,728 events, 3,303 shell commands,
11 context compactions, 140 goal-hook check-ins), the 172 commits since 2026-09-08, and the
benchmark records. The session ended on **"Prompt is too long"** with one task in flight
(§6). Nothing in this file is new evidence; every number cites its record.

## 1. What the user asked for, in order

| when (UTC) | ask | outcome |
|---|---|---|
| 09-08 11:52 | Integrate issue #16's branches; force-push, merge, track | Done; #16 open as the tracking issue |
| 09-08 14:03 | **Standing goal set** (ship as zenavif's default still backend; contract that never lies; no panics/refusals; encode time within libaom; parity capped at measured/attributed/bounded/documented; breadth over depth) | Drives everything below; clauses (1)(2)(3)(5)(6) met or reduced, **(4) open at 1.94x vs 1.5x** |
| 09-08 21:36–22:34 | "how is feature parity?", "why cq63? alpha? all zenpixels formats least-lossy?", "don't stop until all remaining are implemented", "ramp all channels across formats", queries must be accurate | Colour config, alpha (Cs400 item), 4:4:4 kernel, cq-0 lossless, grayscale 10/12 landed in **zenavif** (`e73e3d7`..`a7c56be`); support query derived from the encode path |
| 09-08 22:2x | Global rule: push often; prove loss level with pixel round-trips + zensim-regress | Memory `push-often-and-prove-loss`; `aom_roundtrip_loss.rs` |
| 09-08 23:20 | Close #15? zenavif-serialize derivation? | #15 closed; "both, derivation first" — `seq_header.rs` parser in zenavif-serialize |
| 09-09 01:0x | Test real sizes (cache); worker on archmage/magetypes gaps | Size-controlled bands (192² vs 1024²); vocabulary corrected both ways (`MAGETYPES_VOCABULARY.md`) |
| 09-09 03:52–05:22 | Gather soundness; "why are gates so slow"; ≤12 test binaries; `__internals` visibility; max test opt flags; consolidate crates?; **keep avx512**; use a worktree, run in background; "proceed with all levers" | 320→7 test binaries, nextest (57→6 min), disk 8.2 GB→202 MB, `__internals` gating (decode then encode), avx512 kept |
| 09-10 ~04:00 | Count fn calls for duplicate work; port whatever C does with intrinsics via magetypes/archmage + `cargo asm`; mix raw intrinsics, `rite` tiers, `autoversion` | No redundant work found; hadamard/wiener/filter-intra intrinsics landings; `#[autoversion]` −0.88 % |
| 09-10 ~10:00–13:00 | Wrap up optimisation soon; check copies/borrow-checker workarounds; open a zenav1-svt issue with the lessons; mem 6.7x C?; alloc pass (heaptrack + reading); smallvec/scratch; merge alloc reductions under 1 %; where are the other 7 M?; how many does C allocate | Copy audit (−0.48 %); zenav1-svt **#25**; allocations 10.4 M→5.1 M vs C 1.06 M; smallvec beats tinyvec by 0.5 % of the encode |
| 09-10 ~19:2x | Update docs and memories so compaction is safe | Done at `9239828`..`2ad8974` |
| 09-10 22:4x–23:4x | CPU vs wall? threading? memory strategy? How fast could multithreading be vs C? **Low on tokens: ship blockers, bugs, API sprawl, crate count.** Public-API `.txt` autogen + crates.io scan; **crates cannot be deleted** | CPU==wall, single-threaded both; threading ceiling 3.2x@4; API snapshots enforced; encode surface 4,305→281 lines; publish set pinned by name |
| 09-11 00:xx–02:xx | (hook-driven) zenrav1e comparison; palette+IntraBC wiring; README tables; ravif/svt comparison | 10.2x vs incumbent; screen tools wired + header bug fixed; corrected sweep completed 02:54Z, **unrecorded** |

## 2. What landed (172 commits), grouped

**Contracts and refusals (09-08):** KB-46 nonrd delta-q, KB-47 tile grid refusal, KB-48/45
decoder cancellation windows, KB-49 encoder stop token, KB-50 limits+estimate+categories,
KB-51 bd8 sample-range panic (found by the new fuzz sweep on input 59), KB-52 `AllocMode`.
Clause (1) measured at the zenavif seam; clause (4) measured in-repo for the first time
(3.24x..4.03x).

**Loop-restoration SIMD (09-08):** wiener stats, `pixel_proj_error`, SGR box-sum, then the
four-pixel fold and the SGR A/B pass without a gather (KB-PERF-6/7/8). 2.733x→2.475x at 192².

**Test infrastructure (09-09 early):** 320→7 integration binaries, nextest gate, `__internals`
feature on aom-decode then aom-encode, avx512 restored, dispatch-permutation sweeps
serialised (a consolidation-induced flake), public-API snapshots enforced (09-10).

**Kernel series at speed 0, 1 MP (09-09):** fwd column 4-wide arm, Hadamard arrays,
filter-intra window, edge-filter window, txb-walk allocations, txfm cfg pointers,
`txb_init_levels` stores, magetypes 0.9.29 (KB-PERF-9..15). 2.521x→2.399x.

**Transform fusion (09-09 night):** fused 4x4 fwd/inv, then SIMD-preserving 8x8 fwd
(−2.02 %), 8x8 inv (−1.54 %), 16x16 fwd/inv, 4x8/8x4 fwd/inv, 8x16/16x8 fwd/inv, unpad
(KB-PERF-16..31). Effect tracked call share across a 5x span. Shipping preset 2.287x→2.104x.

**Shipping-preset profile and lane width (09-09/10):** first profile at `--cpu-used 3`
(ranking reorders — intra-pred/rd doubles, LR halves); lane-width audit (nine families
3–7.5x slower, cause is i32 vs i16 lanes); raw intrinsics compile under
`forbid(unsafe_code)` → hadamard AVX2, wiener madd, filter-intra taps (KB-PERF-20/21/22).

**Bounds checks, copies, allocations (09-10):** `highbd_subtract_block` −3.17 % (six lines,
found by `perf annotate`), row loop −0.23 %, `block_error` autoversion −0.88 %,
`sum_squares_2d_i16` −0.35 %, copy sweep −0.48 %, `quantize_fp` −0.18 %, SmallVec winners
−0.46 %, stack `offs`, TLS pool (under-1 % policy), intra scratch pools −2.27 M allocs,
variance 4x4 −0.83 %, z2 prefix −0.27 %, intra predict in place −0.50 %, `quantize_b`
raster −0.13 % (KB-PERF-32..57). **1.994x→1.943x.**

**Measurements that changed the plan:** ratio worsens monotonically with `--cpu-used`
(2.19x s0 → 5.29x s9 at 512²) but DIPS at the shipping s3 at 1 MP; the u16-plane root is
24 % of the gap not 42 %; the fusion sequence bypassed the i16 path (~1 % lever, built as a
half-step, null); named levers = 27 % of the bar; halving every class gap = 1.505x exactly;
threading = 3.2x absolute at 4 threads, ratio-neutral; CPU==wall; Windows transfers at
full strength (−7.1..−7.9 %); allocation levers 6.9x on Windows x86-64; **port 10.2x faster,
0.5 % smaller, +3.0 SSIMULACRA2 than zenavif's current default backend**.

**Wiring and gates (09-10 night):** palette + IntraBC wired into `encode_key_frame`
(they never were; the 427/427 oracle is palette-disabled), matched-oracle gate 54/54,
header-length stream-corruption bug fixed, README feature tables, cross-encoder harness
(`drv-ravif`, `encbench`, `xtool decode/score-rgb`, `enc_rd_compare.py`), `drv-aom` switched
to the shipping path.

## 3. Rejected, with the rule each taught

| attempt | measured | rule |
|---|---|---|
| scalar fused 8x8 transform | +7.07 % | a fused kernel must KEEP the SIMD passes; remove the driver, not the vectors |
| fwd row pass `row_n == 4` arm | null | the row pass pays a full 8x8 transpose on half-zero data |
| i16 inverse column pass (u16 twin) | −0.03 % | i16x16 and i32x8 are both 256 bits; lane count alone changes nothing when the pass is under 16 wide |
| i16 in the fused 16x16 column pass | +0.035 pp vs null | a lever at 0.12 % of the encode cannot be validated alone; build the whole family or not at all |
| exact `with_capacity` | +0.77 %, 0 allocs removed | a capacity hint is arithmetic; measure the allocation COUNT |
| TLS scratch pool | +0.23 % glibc / −0.60 % Win-ARM | a rejection is per-allocator; merged on the under-1 % compounding policy |
| `#[inline]` on `intra_avail` | +0.24 % | prologue/argument samples are attributed call cost, not an inlining win |
| hoist `quantize_b` invariants | +0.15 % | traded a load for a branch — reshaping, not removal |
| z1/z3 per-pixel bounds removal | +0.12 % null | removal is necessary, not sufficient; check the trip count |
| `up==1` two-tap run doubling | +0.34 % | the kernel was STORE-bound; the fix (one `copy_from_slice`) later paid −0.22 % |
| vectorised filter-intra edge | null | trip count 1–7 per call; setup dominates |
| hand-vectorised `pixel_proj_error` squaring | null | LLVM had already vectorised it — disassemble before costing |
| `#[autoversion]` on `variance_raw` | null | dispatch never amortises on 4x4 blocks; and the recorded REASON was wrong (its callers are the ALLINTRA variance factor, later −0.83 %) |
| recon in place (both intra walks) | −0.15 % vs −0.08 % null, kept | a frame-pointer caller share counts every copy in the caller, not the one you removed |
| hog division removal | not built | reshapes one division into up to eleven i64 multiplies |
| tinyvec | −0.08 % vs smallvec −0.46 % | enum discriminant branch on every access |

## 4. Perf trajectory (clause 4), all arms byte-identical

| when | cell | ratio | what moved it |
|---|---|---|---|
| 09-08 12:35 | 128²/192² cpu 0/6 | 3.24x–4.03x | first in-repo measurement |
| 09-08 23:05 | 192² s0 | 2.475x | LR SIMD tiers |
| 09-09 14:43 | 1024² s0 | 2.399x | KB-PERF-9..15 |
| 09-09 14:09 | **1024² s3** (shipping) | 2.258x | first measurement at the preset that ships |
| 09-10 00:17 | 1024² s3 | 2.104x | eight fused transforms |
| 09-10 10:38 | 1024² s3 | 1.994x | `subtract_block` −3.17 %, `block_error`, `sum_squares` |
| 09-10 13:00 | 1024² s3 | 1.9443x | variance 4x4 −0.83 % |
| 09-10 14:05 | 1024² s3 | **1.943x** | z2 prefix; KB-PERF-56/57 measured by paired band only |

Bar: ≤ 1.5x. Gap 1,463 ms; needs 716 ms; named levers carry ~409 ms. Route: breadth.

## 5. Tooling that now exists (and what was only in a scratchpad)

Committed this cycle: `eprof_x86` (profiling driver), `eprof_alloc` (allocation census on
the shipping cell), `dump_cell_yuv`, `screen_tools_gap`, `drv-ravif`, `encbench`,
`xtool decode`/`score-rgb`, `scripts/enc_rd_compare.py`, `scripts/eprof_ab*.py`,
`winperf.yml arms: prepost`, `just gate-landing`, `just api-doc(-check)`,
`docs/public-api/*`, `apidoc/` publishability scan.

**Only in `/tmp` until 2026-09-11:** `band.sh` + `stats.py`, the rotated three-arm band and
paired sign test that produced every KB-PERF-20..57 number. Now `scripts/perf_band.sh`,
`scripts/perf_band_stats.py`, `scripts/perf_arms.sh`, `just perf-arms/perf-band/perf-profile`.

## 6. State at session end (03:30Z), and what this pass did with it

* **CI red since 09-09 06:31Z.** `a88e739` (test consolidation) left `ci.yml` unparsable;
  GitHub ran **zero jobs** on ~85 commits until `52aba0e` quoted the offending lines. Since
  then only the `public-api` job fails: the committed `zenav1-aom-encode.internal.txt` was
  generated with the local nightly (2026-07-24) and CI's tracking nightly (1.100) renders
  two extra auto-trait lines (`!Freeze !Unpin`). Every other leg — both aarch64, both x86,
  i686, Windows ARM, macOS — is green from `52aba0e` on. **Fixed here:** nightly pinned to
  `nightly-2026-09-09` in CI and the justfile, snapshot regenerated, `ci-yaml-check` added to
  `gate-landing`, `just ci-status`.
* **Verified after the fix (run 34561262330 / 34561274930, 04:11Z): `public API surface + crates.io publishability` is GREEN.**
* **A second CI defect, seen on `a038d59`'s run (docs-only commit): both x86 differential legs failed on
  `cdef_find_dir_simd_diff::cdef_find_dir_simd_bit_identical_to_scalar_at_every_tier`** (395 passed / 1 failed
  in the consolidated `aom-dsp` binary). This is the KNOWN dispatch-permutation race the test consolidation
  introduced (`a88e739`): the token sweeps are process-global and CI runs `cargo test --workspace`, whose
  intra-binary thread pool lets a sweep overlap a test that merely calls a kernel. `e1652b4` serialised
  sweep-vs-sweep only. `gate-landing` is immune because nextest runs one process per test. **Open, and
  the first item of the plan: run the differential legs under nextest (or the `aom-dsp` `all` binary with
  `--test-threads=1`), which also makes CI ~3-10x faster.** Until then a red x86 leg naming that test is
  the flake, not a regression; re-run the job.
* **The corrected cross-encoder sweep completed** (02:48–02:54Z, ~6 min, not the "~50 min"
  estimated) and wrote `benchmarks/enc_rd_cross_2026-09-10.tsv` with `drv-aom` on the
  shipping path; the session died before charting or committing it. Recorded here in
  `benchmarks/enc_rd_cross_2026-09-10.md`. `encbench` (the speed claim) was never run.
* `benchmarks/xbench/Cargo.lock` (drv-aom's new dependency) was uncommitted; committed.
* zenavif: clean, last landing `a7c56be` (09-08). Its pin of `zenav1-aom-encode` predates
  `KeyFrameConfig::{enable_palette, enable_intrabc}`; the next bump must add both fields to
  the literal at `encoder_aom.rs:444`.
* The goal Stop hook was session-scoped and is gone. Re-arm deliberately if wanted; it drove
  140 check-ins and roughly half of this cycle's direction.

## 7. Decisions that are the user's

1. **Flip zenavif's default to `Zenav1Aom`?** Against the incumbent the port is 10.2x faster,
   0.5 % smaller and +3.0 SSIMULACRA2 at the same zenavif config (one content, one size);
   against the bar it is 1.94x libaom. Two one-line flips plus the pin bump.
2. **Publish the `zenav1-aom` facade?** Zero consumers; the name cannot be reclaimed either
   way. Window still open (all four names 404 on crates.io).
3. **Fund clause (4) further?** ~8 more transform-class-sized cycles for the 1.5x bar, or
   accept 1.94x for stills. Threading is a separate 3.2x absolute-time lever.
4. **Which thread count does "within libaom" mean?** The comparison excludes libaom's threads.
5. **`cnn_avx2.c` port** — a parity item (KB-41 root #27, the last carrier of 13 datagen
   cells), not a clause-(4) lever at s3.
