# KB-PERF-43 — the temporaries problem: **−78 % of temporary allocations**, −23 % of all allocations, **−0.46 %**

**2026-09-10.** Byte-identical; **−0.457 %** at 1024x1024 cq27 `--cpu-used 3`
(21 of 24 rounds, p=0.0003), against a null of −0.023 % (p=0.84).

**The time delta is the least interesting number here.** The allocation counts
are the result, and they are why this is expected to be worth substantially more
on the platforms this ships to:

| | before | after |
|---|---:|---:|
| allocations per 1 MP encode | 10,419,121 | **8,010,969** (−23 %) |
| **temporary allocations** | 1,931,251 | **417,969 (−78 %)** |

## The measurement that made the fix obvious

`benchmarks/encoder_alloc_pass_2026-09-10.md` located the single biggest source:
**`txfm_rd_in_plane_intra`, 823,028 temporary allocations of 1,359,984 in total
(60.5 %)** — the `winners: Vec<TxbWinner>` it RETURNS, one allocation per call.

The blocker was that the list **escapes four layers up**
(`uniform_txfm_yrd_intra` -> `choose_tx_size_type_from_rd_intra` ->
`pick_uniform_tx_size_type_yrd_intra` -> `rd_pick_intra_sby_mode_y`), and at each
level a candidate's list has to survive while losing candidates' are dropped. A
caller-owned scratch would need a two-buffer keep-best swap through all four —
a real refactor.

**An inline small vector is a drop-in for the return type instead**, and the
walk-length distribution says it is a perfect fit. Instrumented on the shipping
cell (counters added, run, reverted):

    NTXB total=1,359,984   [1]=50.6%  [2]=35.6%  [4]=13.8%

**1, 2 or 4 entries — nothing above 4, ever.** And the call count matches
heaptrack's allocation count for that function *exactly*, which is what proves
`winners` was 100 % of it. So `TinyVec<[TxbWinner; 4]>` — 4 x 16 bytes inline —
removes every one of those allocations with no heap fallback taken on this
workload.

## `smallvec` vs `tinyvec` — MEASURED, and the difference is large

The first implementation used `tinyvec`, on the reasoning that it is itself
`#![forbid(unsafe_code)]` and so matches this workspace's policy. That reasoning
was sound and **the measurement overrode it**. Four-arm rotated band, one
encode per arm per round, the two containers built from the same tree with
**identical allocation counts** (8,010,969 / 417,969 both):

| arm | vs base | rounds faster | p |
|---|---:|---:|---:|
| tinyvec | −0.076 % | 14/24 | 0.54 — **not significant** |
| **smallvec** | **−0.457 %** | **21/24** | **0.0003** |
| **smallvec vs tinyvec** | **−0.500 %** | 20/24 | **0.0015** |
| null | −0.023 % | 13/24 | 0.84 |

**Half a percent of the whole encode separates two containers holding the same
data with the same number of allocations.** The mechanism is structural:
`TinyVec` is an **enum** (`Inline(ArrayVec) | Heap(Vec)`), so every push, index
and iteration branches on the discriminant; `SmallVec` keeps a capacity field
and a union, and the inline path does not pay that branch. On a container
touched millions of times per encode, that branch is the whole difference.

**It also corrects this record's own first number.** The initial three-arm band
put tinyvec at −0.26 % (p=0.0014); the cleaner four-arm rotated band puts it at
−0.076 % and not significant. Quote −0.457 % for the landed (smallvec) form.

`smallvec` carries `unsafe` inside the dependency — not in this crate, and
`forbid(unsafe_code)` is per-crate, so nothing in the workspace's policy is
weakened. It also drops the `Default` bound `tinyvec` needed.

## Correctness

* Byte-identical on four cells: 40,237 B (1024x1024 s3), 39,694 (1024 s0),
  10,912 (512 s0), 11,961 (512 s6) — two sha256-**distinct** binaries.
* `-p zenav1-aom-encode` **772/772**.
* Nothing about the values changed — `TinyVec` is a `Vec`-shaped container and
  every push, index and iteration is the same sequence. The only semantic
  addition is `TxbWinner: Default`, required by `tinyvec` and unused at runtime.

## Platform: the glibc number understates this

**KB-PERF-2 measured this class of lever at 21 % of the win on Darwin and
86-99 % on Windows.** glibc is fast on a hot, repeatedly-reused size class — it
is close to a free-list pop — which is exactly the regime these 1.5 M removed
allocations were in. The same removal on Microsoft's or Apple's allocator should
be worth considerably more than −0.26 %.

`winperf.yml`'s `arms: prepost` mode can settle it on `windows-11-arm` and
`windows-latest`; that run is the honest place to quote this lever's value, not
this box.

## KB-PERF-44 — the next site, found by re-attributing with frame pointers

Re-running heaptrack on a `-C force-frame-pointers=yes` build (the rule this
record's predecessor added, after unresolved stacks cost three wrong fixes)
named the largest remaining site immediately:

    220,000 temporary allocations of 220,000 in total (100.00%)
      from aom_encode::partition_pick::perpixel_variance_y

One line:

    let offs = vec![128u16 << (bd - 8); w];

**A heap allocation of a CONSTANT-valued buffer**, at most 128 entries wide (the
widest `BLK_W`), 220,000 times per 1 MP encode. `highbd_variance` reads it with
stride 0, so only `offs[..w]` is ever touched. Replaced by a `[u16; 128]` stack
array with a `w`-element fill.

| | before | after |
|---|---:|---:|
| allocations | 8,010,969 | **7,790,969** (−220,000, exactly as predicted) |
| **temporary allocations** | 417,969 | **197,969 (−53 %)** |

**Cumulative across KB-PERF-43 + 44: temporary allocations 1,931,251 ->
197,969, a 90 % reduction.**

**Wall: −0.074 % / −0.065 %, NOT significant on glibc** (15/24 both, p=0.31;
null +0.034 %). **It is kept anyway, and the distinction from the REVERTED
capacity hint is the point:**

* the capacity hint **ADDED** arithmetic (two integer divisions per call) to
  save allocations that did not exist — it measured **+0.77 %** and was reverted;
* this **strictly REMOVES** work — no `malloc`, no `free`, and the fill is the
  same `w` stores `vec![v; w]` already performed. It cannot be slower in
  principle, and it measures on the right side of zero, just below what 24
  rounds resolve (220,000 malloc/free pairs at ~30 cycles is ~1.5 ms = 0.05 %,
  which matches the observed −0.07 %).

On the platforms this ships to that arithmetic is different: see the platform
section above.

## KB-PERF-45 — a thread-local scratch pool: 412,208 allocations removed, and it measured SLOWER

The next site after KB-PERF-44 was unambiguous: **4,822,422 of 7,790,969
allocations (62 %) are `RawVecInner::finish_grow`**, and every stack is
`xform_quant_into` <- `xform_quant_optimize_split_into` <-
`encode_intra_block_plane_{y,uv}` <- `encode_b_intra_dry` <- the partition
recursion. Cause: each plane walk built its own `XformQuantScratch::default()`,
so its buffers grew from empty on **every leaf**.

`encode_b_intra_dry` has **28 call sites** through a recursive walk, so a
threaded parameter is a large change; the two plane functions have five between
them and **neither has an early return**, so a `thread_local!` pool taken at
entry and put back at exit is total and needs no signature churn.

**It worked, and it was still slower.**

| | before | after |
|---|---:|---:|
| allocations | 7,790,969 | **7,378,761 (−412,208)** |
| wall | — | **+0.269 % / +0.196 %** (10/24 and 6/24; null −0.030 %) |

Reverted; band committed as `.tlspool.rejected.tsv`.

**Why:** `XQ_POOL.with(...)` is a TLS lookup plus a `RefCell` borrow check,
**twice per plane call** — about a million of each per encode. Dynamic TLS
access is a real function call on Linux. That cost exceeds the ~412 k
`malloc`/`free` pairs it removed, which on glibc are close to a free-list pop.

## The rule this and its three siblings establish

Four allocation-reduction attempts this session, and the outcome is predicted
entirely by **what replaces the allocation**, not by how many are removed:

| change | allocations removed | mechanism added | result |
|---|---:|---|---:|
| `SmallVec` for `winners` | 1.5 M | inline storage (none) | **−0.457 %** |
| stack array in `perpixel_variance_y` | 220 k | none | −0.074 % (kept) |
| exact `with_capacity` | **0** | two integer divisions/call | **+0.77 %** (reverted) |
| thread-local scratch pool | 412 k | TLS + `RefCell`, 2x/call | **+0.23 %** (reverted) |

**An allocation removal only pays if its replacement is cheaper than the
allocation was.** On glibc a hot repeatedly-reused size class is nearly free, so
the bar is low: inline storage and stack arrays clear it, TLS pools and extra
arithmetic do not. On Windows and macOS allocators the bar is much higher and
these verdicts could flip — which is exactly why the platform note above matters,
and why `winperf.yml` should re-run the two rejected variants before anyone
concludes they are dead.

## What is left

Temporary allocations are down to 417,969 and total to 8.0 M. The remaining mass
was not re-attributed after this landing — **re-run heaptrack with
`-C force-frame-pointers=yes` before picking the next one**, because the release
build's stacks come back unresolved and an unresolved stack invites a guess (the
alloc-pass record has the incident where that cost three wrong fixes).

## Not covered

One box, one content class, `--cpu-used 3`, x86-64. The chroma walk
(`intra_uv_rd`) and `rd_pick`'s `winners` fields were converted for type
consistency but their own call counts were not measured separately.
