# Test suite / cycle time — where it actually goes (2026-07-19)

Question: can we speed up the test suites and cycles via binary consolidation or other means?

**Answer: not by binary consolidation, and not by changing the linker — both measured and
rejected. The bottleneck is that `cargo test` runs test *binaries* sequentially while the
cost is concentrated in a handful of e2e tests.**

## Scale

| | count |
|---|---|
| test binaries (autodiscovered, 1 per `tests/*.rs`) | **219** |
| ├ `zenav1-aom-dsp` | 94 |
| ├ `zenav1-aom-encode` | 95 |
| ├ `zenav1-aom-bench` | 17 (+1 bench) |
| ├ `zenav1-aom-decode` | 11 |
| └ `zenav1-aom-sys-ref` | 2 |

Per-binary: ~16 MiB, of which **7.2 MiB is `.text`** — the crate + libaom duplicated into
every one. `target/` totals 19.6 GB.

## REJECTED: binary consolidation (link time is not the bottleneck)

Full `cargo test --no-run -p zenav1-aom-dsp` (94 tests + lib), **fresh target dir**:

| linker | wall | CPU |
|---|---|---|
| default bfd | **17.65 s** | 68.85 s |
| rust-lld (bundled with the toolchain) | **19.27 s** | 67.94 s |

lld is **9% slower** in wall time; CPU is a wash (1%). The whole cold test build of the
94-test crate is ~18 s, so collapsing 219 link steps into ~5 would save seconds at best.
Consolidation is not worth the migration or the loss of `cargo test --test <name>`.

(Feasibility was checked anyway and it *is* safe — 0 tests mutate env vars, call
`process::exit`, hold global mutable state, or `set_current_dir`, and duplicate `fn` names
across files would not collide under `mod`. It's simply not worth doing.)

## THE ACTUAL BOTTLENECK: sequential binaries + concentrated cost

`cargo test` runs each test binary in turn, threading only *within* a binary.

**`zenav1-aom-decode`** — per-binary wall times, summing to 458.5 s (matching total wall,
which confirms serial execution):

```
364.39 s   real_bitstream (15 tests)   <-- 80% of the suite
 39.17 s   ·   37.41 s   ·   12.73 s   ·   3.27 s   ·   1.54 s
  ~0.00 s  x 8 more binaries
```

**`zenav1-aom-dsp`** — same shape: 57.4 s serial across 95 binaries, top 10 = **61%**,
most at ~0.00 s. Slowest: `dv_ref_diff` 10.23 s, `hbd_dist_diff` 6.32 s, `dist_diff` 4.65 s.

### Measured gain from a global work pool

`aom-dsp` is **342 tests across 95 binaries**. Run-only wall time, same box, back-to-back:

| | wall | |
|---|---|---|
| sequential (today's `cargo test`) | **57.02 s** | 1.0x |
| global pool, `xargs -P 16` (process-per-*binary*) | 12.57 s | 4.5x |
| **`cargo nextest run`** (process-per-*test*) | **12.07 s** | **4.7x** |
| floor (slowest single test) | 11.31 s | |

nextest spawns ~3.6x more processes than the per-binary pool, so per-test spawn overhead
was a real concern — measured, it is negligible, and nextest actually **beats** the
per-binary pool. It schedules better: the 11.31 s long pole starts immediately instead of
waiting for its binary's turn in the queue. It lands within 7% of the theoretical floor.

Measured with an encoder agent competing for CPU, so the real gain is at least this.

nextest also gives per-test timing for free (stock libtest will not, on stable), which
immediately identifies the long poles — in `aom-dsp`:
`dv_ref_diff::find_samples_matches_c` **11.31 s**, `hbd_dist_diff::hbd_sad_variance_byte_identical`
**6.63 s**. Those two are the entire floor.

Installed for this measurement: `cargo-nextest 0.9.140` (prebuilt, `get.nexte.st`).

### Caveat: the pool barely helps `aom-decode` — and the reason is NOT "one binary"

MEASURED under nextest: the decode suite is **364.0 s** vs `cargo test`'s 458.5 s — only
**1.26x**. The floor is not a binary, it is a single **test**:

```
SLOW [363.9 s]  real_bitstream::multi_tile_streams_decode_byte_identical_to_c
SLOW [285.8 s]  real_bitstream::sb128_streams_decode_byte_identical_to_c
SLOW [122.9 s]  real_bitstream::real_bitstreams_decode_byte_identical_to_c
```

Those three already run *concurrently* under nextest, so the suite total is just the
longest one. **Splitting the test BINARY would achieve nothing** — under a global pool the
test, not the binary, is the scheduling unit. (An earlier revision of this file recommended
sharding the binary. That was wrong and is corrected here.)

The real shape: each of those tests sweeps a **matrix inside one `#[test]`**, serially. For
`multi_tile_streams_...` that is `sizes[2] x combos[5]` = 10 cells, each a real `aomenc`
encode plus a dual decode. The lever is to split the matrix into per-cell `#[test]`s so the
pool can schedule the cells — same assertions, same coverage, finer granularity. That is
the only thing that moves the decode floor.

Generalisation: **nextest's speedup is bounded by your longest single test.** Where work
spreads across many tests (`aom-dsp`: 342 tests, longest 11.3 s) it is 4.7x. Where one test
serialises a whole matrix (`aom-decode`) it is 1.26x.

## Other findings

- **Profile matters more than anything above.** `target/debug/deps` is 7.2 GB at 18 MiB
  median because the default `test` profile inherits full debuginfo. `profile.test-fast`
  (opt-level 3, `debug = "line-tables-only"`) already exists and the e2e byte gates are
  10-20x faster under it — the win is making sure it is the path actually used.
- **The C oracle cache is per-worktree.** `aom-sys-ref` caches libaom keyed on the submodule
  SHA, but the stamp lives in `upstream/build/` — so every new worktree pays a full cmake
  libaom build. Relevant to the multi-agent workflow.
- **192 GB across 65 stale agent worktrees** in `.claude/worktrees/`, 60 with their own
  `target/`. Disk is 50% full so this is not yet causing pressure, but it is pure
  accumulation. Not cleaned up here — some may predate this session and one belonged to a
  live agent.

## Test-hygiene check (done while here)

Exactly **one** `#[ignore]` exists in the whole workspace: `aom-dsp/tests/sad_simd.rs:50`,
`avx2_sad_perf_ratio`. It is legitimate, not a relaxation — it prints a coarse wall-time
ratio and is documented "Not a CI gate (single untuned run, no fixed HW pinning)". The
correctness assertions for the same kernel (`simd == scalar`, `simd == ref_sad` against the
real exported C fn) live in the non-ignored test directly above it and run every time.
No silent runtime self-skips were found either.

## Recommendation

1. Adopt `cargo-nextest` for the global pool — **4.7x on `aom-dsp`**, within 7% of the
   floor. Bonus: free per-test timing and `--partition` for CI sharding. Note the gain is
   crate-dependent: **1.26x on `aom-decode`**, because one test is the entire floor there.
2. Split the matrix-sweeping tests into per-cell `#[test]`s — this is the only thing that
   moves the decode floor, and it is worth more than item 1 for that crate. Targets, in
   order: `real_bitstream::multi_tile_streams_...` (363.9 s, sweeps 2 sizes x 5 combos
   serially), `::sb128_streams_...` (285.8 s), `::real_bitstreams_...` (122.9 s). Same
   assertions, finer granularity — NOT a coverage reduction. Then
   `dv_ref_diff::find_samples_matches_c` (11.3 s), the `aom-dsp` floor.
3. Keep `--profile test-fast` as the default developer path.
4. Do NOT consolidate test binaries. Do NOT switch to lld. Both measured, both rejected.

---

# ADDENDUM 2026-09-09 — consolidation REVISITED and now DONE, on different grounds

**The 2026-07-19 rejection above is not overturned on its own terms, and this
addendum starts by agreeing with it.** Consolidation is not a build-time win of
consequence, and the linker is not the bottleneck. Both were re-measured today,
independently, before this record was re-read — and both reproduced:

| re-measured 2026-09-09 | 2026-07-19 said | today |
|---|---|---|
| cold build, aom-dsp test targets | 17.65 s (94 targets) | 23.83 s (111 targets) -> **5.48 s** (1 target) |
| linker | lld 9 % SLOWER than bfd | mold 2.7x faster on an isolated 9.8 MB link, but link is a minority of a 22.8 s whole-workspace test build -> **dropped, not worth a CI dependency** |
| the actual bottleneck | sequential binaries; nextest 4.7x | **confirmed: `cargo test` 57.4 min vs nextest 340.9 s = ~10x on the full workspace** |

So the ranking in the original record was right, and two hypotheses of mine
("link time dominates", "relink after touching a lib dominates") were wrong and
were corrected by measurement — the relink win is only 13.6 s -> 10.7 s, because
that time is recompiling the LIB at opt-level 3, not linking.

## What changed the decision: three grounds the original did not weigh

1. **Disk. 8.19 GB -> 202 MB (40x).** 446 test executables at a ~21.8 MB median,
   each statically linking the port + libaom.a, against 6 harnesses totalling
   202 MB. The original noted `target/` at 19.6 GB but did not treat it as a
   reason; it is now 22.68 GB and the single largest measured effect here.
2. **Its stated cost is answerable.** The original declined partly for "the loss
   of `cargo test --test <name>`". Measured today: `--test all -- <module>::`
   works, and libtest accepts MULTIPLE positional filters (verified against a
   real test binary: `--list a b` returns the union), so `just test-simd`'s ten
   `--test` flags collapse into one invocation with ten filters. No capability
   is lost; 20 call sites in `justfile`/`ci.yml` were rewritten.
3. **It unlocks two things that are not about speed at all**, and these are the
   actual mandate:
   * **the public-API leak.** 4,149 `pub` items exist across the workspace, and
     the measured REAL consumer surface is **two modules** — zenavif uses only
     `aom_encode::key_frame` and `aom_decode::frame`. The rest is public because
     320 separate test crates had no other way in. One test target per crate
     needs ONE declared door (`__internals`) instead of 4,149 `pub`s.
   * **per-package `opt-level`.** All 134k lines of test code compile at
     `opt-level = 3`, which only the code under test needs. Cargo cannot set a
     different opt-level for a package's lib vs its tests — but it can per
     PACKAGE, which becomes expressible once tests are consolidated.

## Result

320 integration binaries -> **6** (7 with the `census` target). Two files stay
their own target for stated reasons, not taste:

* `encode_limits_and_estimate` — it defines the counting `#[global_allocator]`
  (KB-50). Its counters are process-global, and KB-50 already records two
  measuring tests corrupting each other under intra-binary concurrency; merging
  141 other files' allocations into that process would corrupt the peak it gates.
* `content_family_census` — builds under `--no-default-features --features
  census`, which its siblings cannot compile under.

**Verified equal, not assumed:** the full workspace runs **1506 -> 1502** tests,
all passing. The 4-test delta is fully explained and is a duplicate REMOVAL:
`tests/common/mod.rs` carries a `#[test]`, which was compiled and executed once
per including binary (5 copies of `common::md5::md5_known_vectors` in
aom-decode); it now runs once. `1506 - 5 + 1 = 1502`, exactly.
