# KB-PERF-1 landed — caching the intra-mode CNN per 64x64: 10.66x -> 3.36x (2026-08-02)

**One cache. −357 ms off a 521 ms encode. Not one byte moved.**

[`encoder_hotspot_profile_2026-08-02.md`](encoder_hotspot_profile_2026-08-02.md)
found that `cnn_partition::cnn::cnn_predict` was **74.7 % self of the port's
entire encode** and **81.5 % of its 10.7x gap** to libaom, because the port ran
the cascade at every 64/32/16/8 partition node — **2558 times per 1 MP frame
against libaom's 256** — while libaom computes it once per `BLOCK_64X64` and
caches it (`av1/encoder/partition_strategy.c:160`, the
`bsize == BLOCK_64X64 && !part_info->cnn_output_valid` latch). This is that
lever, implemented and measured.

Provenance, commands, box and the honest gaps:
[`encoder_cnn_cache_2026-08-02.meta`](encoder_cnn_cache_2026-08-02.meta).
Data: `.control.tsv` (the interleaved 3-arm band), `.breadth.tsv` (size x
cpu-used x quantizer).

---

## Control band — read this before reading any delta

Three arms, **interleaved** (base, cache, C, base, cache, C, …) over 9
independent process invocations each, every invocation 2 warm-up + 7 timed
encodes with its own median taken. 1024x1024 photo, cq 44, cpu-used 6 — the
profile's own cell. Script: `scripts/eprof_control_ab.sh`, the A/B form of the
profile's `eprof_control.sh`.

| arm | median | min | max | spread | stdev | coded bytes |
|---|---:|---:|---:|---:|---:|---:|
| `port-base` (`5e29589`, no cache) | **520.93 ms** | 512.84 | 530.09 | 3.31 % | 0.94 % | 4472 |
| `port-cache` (this change) | **164.03 ms** | 159.19 | 167.23 | 4.90 % | 1.65 % | 4472 |
| `libaom-c` | **48.88 ms** | 47.53 | 48.97 | 2.95 % | 1.40 % | 4472 |

**All three arms emit the same 4472-byte stream, byte for byte.**

| ratio | median | the 9 PAIRED ratios |
|---|---:|---|
| base / libaom-c | 10.657x | 10.48 10.61 10.62 10.65 10.66 10.72 10.91 11.03 11.15 |
| **cache / libaom-c** | **3.356x** | 3.25 3.26 3.36 3.38 3.40 3.41 3.44 3.44 3.46 |
| base / cache (the speed-up) | 3.176x | 3.12 3.13 3.17 3.18 3.18 3.20 3.22 3.22 3.25 |

**10.66x -> 3.36x.** The *worst* paired cache/C ratio (3.46) is 3x below the
*best* paired base/C ratio (10.48) — this delta is nowhere near the noise floor,
which on the ratio itself is ~6 %. The base arm reproduces the profile's 10.72x
control on a different day, so the two bands are comparable.

> The box was NOT idle: the same `zenmetrics` fleet as the original profile,
> whole-box load average 30-32 of 12 cores. That is why the arms are
> interleaved and why the spread is quoted before the delta (playbook §6).

The profile *projected* 3.48x-3.71x from arithmetic on its measured self-costs.
Measured: **3.36x** — slightly better than the projection, because the cache
also removes 2302 of the 2558 window extractions, which the projection had
counted under a different lever.

---

## What changed, in one sentence

`decision::PartitionSearchInfo` is C's `x->part_search_info`
(`av1/encoder/block.h:391-398`): `cnn_output_valid` + `cnn_buffer[1636]` +
`log_q`. `rd_pick_partition_real` threads it, invalidates it at the same
`BLOCK_64X64` re-anchor that already resets `quad_tree_idx`
(`partition_search.c:3339-3343` — literally the same two lines of C), and
`intra_mode_cnn_partition` computes the cascade only on the
`bsize == BLOCK_64X64 && !valid` branch and returns `None` where C returns at
`:227`. The 65x65 window is produced by a closure, so it is extracted only on
the computing path.

KB-23's separate `cnn_root_whole_in_frame` predicate is **deleted**: with a real
latch it is emergent, exactly as in C. At a 64x64 node the per-block
`av1_is_whole_blk_in_frame` test IS the containing-64 test (blocks are aligned
to their own size), so a frame-edge 64x64 never computes, the latch stays 0, and
every smaller node inside it prunes nothing — which is precisely what KB-23
established C does. Asserted rather than argued: see the 9-of-16 row below and
the partial-SB axis in the gate list.

---

## The call count, measured two ways

`crates/aom-bench/examples/eprof_alloc.rs` (counting `GlobalAlloc` with
exact-size call-site counters — the profile's own method), 1024x1024 cq44
cpu-used 6:

| exact allocation size | call site | before | after | per 64x64 SB |
|---|---|---:|---:|---:|
| 20480 B | `cnn_predict` layer-0 output | 2558 | **256** | 1.00 |
| 16900 B | `cnn_predict` layer-0 input | 2558 | **256** | 1.00 |
| 5120 B | layer-1 output | 2558 | **256** | 1.00 |
| 1280 B | layer-2 output | 2558 | **256** | 1.00 |
| **4225 B** | **`extract_intra_cnn_window`'s `vec![0u8; 65*65]`** | 2558 | **256** | 1.00 |

**2558 -> 256 — exactly libaom's 256, exactly 1.00 per 64x64 superblock.** The
window extraction (profiled at 3.70 ms + 2558 allocations, and the one symbol
in the port's top ten with *no* libaom counterpart) fell with it, because the
closure is only invoked on the computing path.

From the other side, `crates/aom-bench/examples/eprof_cnn_verify.rs` counts the
cascade itself rather than its allocations:

```
cascade COMPUTES   256  (1.00 per 64x64)
cache READS        2302  (all re-verified bit-identical)
nodes total        2558  (9.99 per 64x64 — what the uncached port ran)
```

256 + 2302 = **2558**, reconstructing the profile's count exactly from an
independent counter.

---

## Proving the cached value IS the value — 2302 reads, checked

The correctness claim ("all ~10 runs per superblock convolve the identical
window") is not taken on argument. `decision::set_cnn_cache_verify(true)` makes
**every** cache read re-extract its window, re-run the full 5-layer cascade, and
assert bit-identity with what is cached — and the same for `log_q`.

Armed on the profile cell: **2302 of 2302 cache reads bit-identical**, and the
frame is byte-identical to libaom's (4458-byte OBU payload inside the 4472-byte
stream). Armed in CI at four smaller cells by
`crates/aom-bench/tests/cnn_cache_identity.rs`, which additionally pins the
latch *structurally* — cascades computed must equal the number of whole-in-frame
64x64 nodes:

| cell | computes | expected | reads | vs real aomenc |
|---|---:|---:|---:|---|
| 256x256 cq24 cpu2 (SB-exact) | 16 | 16 | 1220 | byte-identical |
| **196x196 cq24 cpu2 (partial SB)** | **9** | **9** | 652 | byte-identical |
| 256x256 cq24 cpu6 | 16 | 16 | 1220 | byte-identical |
| **256x256 cq24 cpu2 `--sb-size=128`** | 16 | 16 | 1220 | byte-identical |

The 196x196 row is KB-23 restated as an assertion: 16 superblocks, 7 of them
frame-edge, **9** cascades — the seven edge 64x64s compute nothing and therefore
prune nothing, produced by the latch rather than by a separate predicate.

The SB128 row is the row that makes the test *able to fail* (playbook §1) — see
the bite proof.

---

## Bite proof

**(a) Neuter the latch** — recompute at every node, still writing the cache
(`bsize_idx == 1 && !info.cnn_output_valid` -> `true`):

| | with the latch | latch neutered |
|---|---:|---:|
| CNN cascade runs / 1 MP frame | 256 | **2558** (9.99 per SB) |
| window extractions | 256 | **2558** |
| wall, 1024² cq44 cpu6 | 164.03 ms | **565.69 ms** |
| coded bytes | 4472 | 4472 |

The whole delta returns. (565.69 ms is slightly *above* the 520.93 ms true
baseline: the neutered build still copies the 6544-byte cache buffer at all 2558
nodes, which the baseline never did, and this is one invocation's median rather
than an interleaved 9-invocation one. Read it as "the delta returns", not as a
fourth arm.) The bytes staying at 4472 with 2558 recomputes is itself a third
independent statement that the cached value equals the recomputed one.

**(b) Delete `invalidate_cnn()` at the BLOCK_64X64 re-anchor:**

| test | with the reset | reset deleted |
|---|---|---|
| `s4cov_partial_sb_axis::sb128_partial_sb_speed_axis_byte_matches` | 48/48 byte-exact | **12/48** — 36 DIVERGE |
| `cnn_cache_identity` (SB128 row) | pass | **panics**: `intra-CNN cache read differs from a recomputation at bsize_idx 1 quad_tree_idx 0` |
| `s4cov_partial_sb_axis::partial_sb_speed_axis_chroma_formats_byte_match` | pass | pass |
| `s4cov_partial_sb_axis::partial_sb_high_bitdepth_byte_matches_where_interpretable` | pass | pass |
| `s4cov_partial_sb_axis::mono_speed0_size_qindex_localize` | pass | pass |

The 36 divergent cells are exactly **every size x cpu-used 1..6**; cpu-used 0
(where `intra_cnn_based_part_prune_level` is 0) and cpu-used 7 (VAR_BASED, never
enters the RD search) match. That is KB-24's cell set.

**Honest scoping of (b), per the KB-24 precedent.** The reset only bites under
`--sb-size=128`. Under SB64 a superblock *is* a 64x64, so `pack_tile`'s fresh
per-superblock `PartitionSearchInfo` (C's `encodeframe.c:692` reset) is already a
per-64x64 reset and the explicit call is inert — measured: with it deleted, the
1024²/256²/2048² SB64 encodes above stay byte-identical at 4472 / 888 / 11418,
and every SB64 test stays green. Both resets are kept because both are in C and
the pair is what makes the invariant true by construction; the honest claim is
that only one of them is load-bearing on any given superblock size.

Two perturbations, two different failure modes — throughput for (a), correctness
for (b) — and (b)'s failure set is disjoint from every other test in its own
binary. Different symptoms and different cell sets = genuinely different roles
(playbook §1).

---

## Breadth — both arms across size, cpu-used and quantizer

`scripts/eprof_breadth.sh`, arms back to back per cell. The "before" columns are
[`encoder_hotspot_profile_2026-08-02.md`](encoder_hotspot_profile_2026-08-02.md)'s
breadth table — same box, same script, same source files.

| cell | before ratio | **after ratio** | before ms | after ms | CNN/SB before | **after** | bytes agree |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 256x256 cq44 cpu6 | 13.88x | **3.32x** | 79.85 | 19.01 | 26.4 | **1.00** | yes (888) |
| 1024x1024 cq44 cpu6 | 10.88x | **3.33x** | 496.32 | 158.93 | 9.99 | **1.00** | yes (4472) |
| 2048x2048 cq44 cpu6 | 9.85x | **3.27x** | 1460.03 | 512.03 | 7.50 | **1.00** | yes (11418) |
| 1024², cpu-used 8 | 3.45x | 3.39x | 37.64 | 38.45 | 0 | 0 | yes (4836) |
| 1024², cpu-used 7 | 2.69x | 2.66x | 101.02 | 99.91 | 0 | 0 | yes (4978) |
| 1024², cpu-used 6 | 10.91x | **3.35x** | 497.67 | 159.32 | 9.99 | **1.00** | yes (4472) |
| 1024², cpu-used 5 | 5.68x | **4.16x** | 1191.11 | 905.08 | 7.89 | **1.00** | yes (4152) |
| 1024², cpu-used 4 | 9.39x | **7.95x** | 2719.01 | 2215.96 | 15.78 | **2.00** | yes (4222) |
| 1024², cpu-used 3 | 8.18x | **6.88x** | 3585.96 | 3037.27 | 15.77 | **2.00** | yes (4227) |
| 1024², cq 10 | 15.23x | **2.90x** | 1164.13 | 214.04 | 22.39 | **1.00** | yes (47758) |
| 1024², cq 26 | 10.51x | **3.06x** | 657.45 | 193.35 | 13.28 | **1.00** | yes (16941) |
| 1024², cq 58 | 10.92x | **3.34x** | 403.98 | 125.63 | 8.34 | **1.00** | yes (1773) |

Four things to read off it:

* **`port_bytes == c_bytes` at all 12 measured cells**, before and after. These
  are the same encode on both sides, at every cell.
* **The worst cells improved the most.** 256x256 was the worst *size* ratio in
  the original profile (13.88x at 26.4 cascades per superblock) and cq 10 the
  worst ratio measured anywhere (15.23x at 22.4 per SB); they are now 3.32x and
  2.90x. The redundancy was largest exactly where the search descends deepest,
  so removing it is worth most there.
* **cpu-used 7 and 8 are unchanged** (2.69x -> 2.66x, 3.45x -> 3.39x, both
  inside the control spread, CNN runs 0 both ways). They take
  `VAR_BASED_PARTITION` and never enter `rd_pick_partition_real`, so they are a
  **negative control**: the change is confined to the CNN path, and its absence
  there is measured rather than assumed.
* **cpu-used 3 and 4 read 2.00 per 64x64, not 1.00** — the SB walk runs *twice*
  at those speeds. The before-data shows the identical doubling (15.78 and 15.77
  against cpu-used 5's 7.89), so the per-walk count is 1.00 at every armed cell,
  and the two speeds improve less (9.39x -> 7.95x, 8.18x -> 6.88x) because the
  profile already measured the CNN as only 24 % / 18 % of the gap there. *Which*
  second walk it is was NOT traced here; loop-restoration is on at allintra
  speeds 0-4 and off from 5 (`speed_features.c:519-520`), which fits, but that is
  a hypothesis, not a measurement.

`cpu-used 9` still refuses this cell (`nonrd_pickmode.rs:1135`, the KB-32 handoff
panic the profile flagged). Pre-existing, separately tracked, untouched here.

### What it buys in the currency xbench measures

The 1 MP/s bar is a preset ladder, so the useful question is which `cpu-used` the
port can afford. Both columns are **measured** wall times at the cell:

| cpu-used | before | after |
|---|---|---|
| 6 | 497.67 ms = **2.11 MP/s** | 159.32 ms = **6.58 MP/s** |
| 5 | 1191.11 ms = **0.88 MP/s** *(fails the bar)* | 905.08 ms = **1.16 MP/s** *(clears it)* |
| 4 | 2719.01 ms = 0.39 MP/s | 2215.96 ms = 0.47 MP/s |
| 3 | 3585.96 ms = 0.29 MP/s | 3037.27 ms = 0.35 MP/s |

**The port's `>= 1 MP/s` qualifying mode moves from `cpu-used 6` to `cpu-used
5`** — one of the five preset steps that separate it from libaom's `cpu-used 1`,
whose total is the +11.64 pp `xbench_2026-08-01.md` identified as the port's
entire photographic BD-rate deficit. No coding decision changes and no output
byte changes. (How much BD-rate one step is worth was not measured — xbench
published BD-rate at the qualifying modes, not a per-preset ladder. Do not assume
11.64/5.)

---

## What is NOT measured here

* **One image, one content class** (photographic), 8-bit 4:2:0, single tile,
  single thread — the same envelope as the profile this follows up. Byte
  identity across the rest of the configuration space is carried by the test
  suite, not by this benchmark.
* **cpu-used 0, 1, 2 were not measured**; cpu-used 9 refuses this cell.
* **No re-profile.** This is a wall-clock A/B plus exact call counts; the
  symbol-level rollup was not re-run, so the composition of the *residual* 3.36x
  is not measured here. The profile's arithmetic says it should now be dominated
  by the forward/inverse transform (+23.8 ms), allocation (+24.1 ms) and intra
  prediction + mode RD (+15.1 ms) — its levers 3 and 4 — but that is a
  projection, not a measurement of this build.
* **No instruction-count profile** (no valgrind on Apple Silicon), so IPC and
  cache behaviour are folded into every wall number.
* **The box was not idle.**

---

## Gates

| run | result |
|---|---|
| `cargo nextest run --cargo-profile test-fast --workspace --run-ignored all` | **945/945** (see the note below) |
| the same, default tier | 919/919 |
| the same, `AOM_FORCE_SCALAR=1` | 919/919 |
| `cargo check --target x86_64-apple-darwin --workspace --all-targets` | clean |

Gate 2 keeps **zero pinned cells** across cpu-used 0..9.

**The first `--run-ignored all` pass reported 2 failures, and neither was this
change.** `kb31_mandatory_tiles::mandatory_tile_split_byte_identical_across_speeds`
(its OPEN set `[(4032,8),(4160,8)]` observed as `[]` — 20/20 byte-identical) and
`::issue6_reported_sizes_encode` (5472x3648 now byte-identical to real aomenc,
was +339 B). Both are self-promoting pins firing in the GOOD direction: cells
that used to diverge now match. Both are `#[ignore]`d nightly tests that had not
run since 2026-07-30 — 74 commits, including **KB-12 (`0953fa7`)**, the dropped
Hadamard transpose in the nonrd estimate arm, which both tests' own doc comments
already named as the owner of exactly those cells. Both are cpu-used 8/9, where
the intra-CNN prune runs **zero** times (measured, breadth table above), so the
cache cannot reach them — and both **reproduce on the pristine baseline**,
verified by re-running that file with the change stashed. Re-pinned to the
tighter expectation each test's own failure message asked for (empty OPEN set /
hard `assert_eq!`); nothing was relaxed. The three regenerated
`benchmarks/config_perm_*_2026-07-30.tsv` evidence sweeps in the same landing
have the same provenance.
