# The alloc pass: 10.4 M allocations per 1 MP encode — where they are, and one hypothesis measured and REFUTED

**2026-09-10.** Prompted by the observation that the `memory` class runs at
**6.7x libaom** and might be the perf problem. Measured with **heaptrack** plus
a code reading, not inferred from the profile.

## The measurement

One 1024x1024 cq27 `--cpu-used 3` encode:

| | |
|---|---:|
| allocations | **10,419,121** |
| temporary allocations | 1,931,251 |
| leaked | 1 |
| calls through `RawVecInner::finish_grow` (aom_encode) | **5,870,590 — 56 %** |

**Over ten million allocations for a single still image, and more than half of
them are a `Vec` GROWING.** libaom's equivalent work is a handful of frame-level
buffers.

## CORRECTED — the attribution below was WRONG, and heaptrack with frame pointers says so

**The "two `Vec<i32>` per transform block" reading in the next section is wrong.**
It was inferred from `finish_grow`'s count and a code reading, without checking
that the named paths are live. Three separate fixes were written on it — exact
`with_capacity` at three sites, and a full scratch conversion of
`intra_mode_rd_eval` — and **all three left the allocation count identical to
the digit (10,419,121)**. That is not a coincidence: `intra_mode_rd_eval` turns
out to have **no callers on the encode path at all**, and the capacity sites
were single-element pushes.

**Re-run under heaptrack with `-C force-frame-pointers=yes`, which resolves the
stacks the release build could not**, the answer is one function:

    823,028 temporary allocations of 1,359,984 in total (60.52 %)
      from aom_encode::tx_search::txfm_rd_in_plane_intra

and its stack is `uniform_txfm_yrd_intra` <- `choose_tx_size_type_from_rd_intra`
<- `pick_uniform_tx_size_type_yrd_intra` <- `rd_pick_intra_sby_mode_y` <-
`rd_pick_intra_mode_sb` <- the partition recursion.

**The source is the `winners: Vec<TxbWinner>` the function RETURNS — one
allocation per call, ~823 k calls per 1 MP encode.** `TxbWinner` itself is three
scalar fields, so it is the `Vec` and not its contents. KB-PERF-13 gave it an
exact capacity and explicitly left it allocating *because it is returned*.

**The fix is not a scratch.** The value escapes four layers up, and at each
level a candidate's winners must SURVIVE while losing candidates' are discarded
— so it needs a **two-buffer keep-best swap** (current + best, swapped on
improvement), which is a real refactor of that call chain rather than a
`&mut` parameter.

**Two lessons, both paid for here:**

1. **A release build's heaptrack stacks can be unresolved, and an unresolved
   stack invites a guess.** Build with frame pointers before attributing
   allocations — the same flag that made KB-PERF-15's `memset` callers visible.
2. **Verify the path is live before fixing it.** "Allocation count unchanged to
   the digit" is the cheapest possible check and it caught three wrong fixes in
   a row.

## Where they come from — the SUPERSEDED code reading

`finish_grow` at 5.87 M is `Vec` growth, and the arithmetic points at one shape:
**two `Vec<i32>` per transform block**.

* `XformQuantScratch` holds `coeff` / `qcoeff` / `dqcoeff`, and the `_into`
  forms reuse it — that is KB-PERF-2's landing and it works.
* But **the OWNED forms `xform_quant_optimize` / `xform_quant` allocate fresh
  `Vec`s per call**, and they still have live callers on hot paths:
  `intra_rd.rs:181` (`intra_mode_rd_eval`), `encode_sb.rs:2063` and `:2071`,
  `lib.rs:858` / `:907`, `var_tx.rs:583` / `:603`.
* `TxbEncode` itself **owns** `qcoeff: Vec<i32>` and `dqcoeff: Vec<i32>`, one
  pair per txb, and those are retained output rather than churn — so they cannot
  simply move into a scratch.

`5,870,590 / 2 ≈ 2.9 M` transform blocks with a growing pair each, which is the
right order for this cell. **The fix is structural**: an arena — one flat
`Vec<i32>` per walk with `(offset, len)` in `TxbEncode` — not a capacity hint.

## The hypothesis that was measured and REFUTED

The obvious cheap fix was to give the growing `Vec`s an exact capacity, as
KB-PERF-13 did for the luma `winners`. Applied at three sites — the luma and
chroma `Vec<TxbEncode>` walks in `encode_intra.rs` and the chroma `winners` in
`intra_uv_rd.rs`, all named by heaptrack.

**Result: allocations UNCHANGED at 10,419,121 (identical to the digit), and the
encode measured +0.77 % SLOWER** (+0.859 % and +0.673 % against the two base
copies, **1 of 24 rounds faster**, p<0.0001; null +0.131 %, p=0.31).

Two reasons, both worth keeping:

1. **Most blocks are single-txb.** A 4x4 transform in a 4x4 block pushes ONE
   element, and `Vec::new()` + one push allocates exactly once — the same as
   `with_capacity(1)`. There was no growth chain to remove. `temporary`
   allocations fell by only 13,922 of 1.93 M.
2. **The capacity expression costs two integer divisions** (`div_ceil` twice)
   on a path called millions of times. Divisions are ~20-40 cycles; the
   allocations they were meant to save did not exist.

**Reverted.** Band committed as `.capacity.rejected.tsv`.

**The transferable rule: a capacity hint is arithmetic, and on a hot path it has
to earn its keep like any other change. Measure the allocation COUNT before and
after — if it does not move, the hint is pure cost.**

## Platform: this is worth more than the Linux number suggests

glibc is fast on a hot, repeatedly-reused size class, which is why the capacity
experiment could measure +0.77 % while removing real allocations. **KB-PERF-2
measured the same class of lever at 21 % of the win on Darwin and 86-99 % on
Windows** — so 823 k allocations per encode is likely to cost substantially more
on the platforms this ships to than it does on the box it was measured on.
`winperf.yml`'s `arms: prepost` mode can settle that on `windows-11-arm` and
`windows-latest` once the keep-best refactor exists.

## What this says about the `memory` class

It does **not** say the 6.7x class ratio is the perf problem, and the evidence
is that the class's own components have been attacked directly all cycle with
small returns, while the two biggest wins of the session (`highbd_subtract_block`
−3.17 %, `block_error` −0.88 %) were **arithmetic** loops found by disassembly,
not allocation.

The honest reading: **10.4 M allocations is genuinely bad and worth fixing, but
the fix is the arena refactor, and its size is unmeasured.** The `finish_grow`
mass is real; whether removing it is worth 0.5 % or 3 % cannot be known until
the arena exists, because glibc's allocator is fast on a hot, repeatedly-reused
size class and these are all the same few sizes.

## The remaining 62 %: sizing data for the arena / SmallVec decision

After KB-PERF-43/44, **4,822,422 of 7,378,761 allocations are still
`finish_grow`**, and they are `TxbEncode`'s two owned `Vec<i32>` (`qcoeff`,
`dqcoeff`) — one pair per transform block. **The TLS-pool platform flip
(`encoder_tlspool_windows_2026-09-10.md`) is what makes this worth funding**:
Linux systematically understates it, and Windows ARM reversed the sign on a
change removing eight times fewer allocations.

**The measurements a next session needs, so it does not re-derive them:**

Coefficient counts per transform (s3 census, forward):

| inline capacity | covers | bytes inline per field |
|---:|---:|---:|
| 16 | **40.5 %** | 64 |
| 64 | **84.4 %** | 256 |
| 256 | 99.3 % | 1024 |

Txb counts per walk (`NTXB` histogram, measured for KB-PERF-43): **1 txb
50.6 %, 2 txbs 35.6 %, 4 txbs 13.8 %, nothing above 4.**

**Why the choice is not obvious, and both effects must be measured, not
argued:**

* `TxbEncode` is ~80 bytes today. Inline-16 makes it ~208, inline-64 ~592 — and
  it lives in a `Vec<TxbEncode>` of up to 4, so the struct-size growth is paid
  on every move of that vector.
* The producer currently does `core::mem::take(&mut xq.qcoeff)`, a **free
  move**. A `SmallVec` must `from_slice`, a **copy**. Trivial at 16 elements
  (64 bytes), but for the 0.7 % of blocks at 1024 coefficients it becomes a 4 KB
  copy *plus* a heap spill where there used to be a pointer move.
* So inline-16 buys 40.5 % of the allocations and pays a copy on exactly those;
  inline-64 buys 84.4 % and pays a much larger struct everywhere.

**A true arena** — one flat `Vec<i32>` per walk with `(offset, len)` in
`TxbEncode` — avoids both the struct growth and the copy, at the cost of
threading the arena's lifetime alongside the outcome struct. That is the shape
KB-PERF-45's record already names, and this data says it is preferable to a
`SmallVec` field on both counts.

**Measure on `winperf.yml`, not only here.** This cycle's two kept allocation
changes were platform-robust because they added no mechanism; an arena adds
indexing, which is a mechanism, so it belongs in the class where the platforms
have now been shown to disagree.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, x86-64. Peak heap and
total bytes were not extracted; only counts. The arena refactor was not
attempted.
