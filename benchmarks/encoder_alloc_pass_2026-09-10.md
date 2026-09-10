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

## The libaom BASELINE, which nobody had measured

heaptrack on the **C arm** of the same 1 MP cell:

| | allocations | temporary |
|---|---:|---:|
| **libaom (C)** | **1,055,245** | **7,209** |
| port, at the start of this pass | 10,419,121 | 1,931,251 |
| port, after KB-PERF-43/44/48/49/50 | **5,106,723** | 336,064 |

**The port was 9.9x libaom on allocation count and 268x on temporaries; it is
now 4.8x and 47x.** libaom is not at zero either — a floor near 1 M is what
"as good as C" means here, not zero.

## KB-PERF-50 — where the remaining millions actually were

`intra_model_rd_y` sat on top of **3,527,376 of 6,922,557 allocations**, and
**nothing inside it allocates** — it only does `clear()` + `resize()` on a
scratch that is correctly threaded in. The cause was one line upstream
(`intra_rd.rs:1103`):

    let mut txs = crate::tx_search::IntraTxScratch::default();

**`rd_pick_intra_sby_mode_y` built a fresh scratch on every call**, and it is
called once per leaf. Its own comment says the scratch exists so "every mode x
tx size x txb x candidate tx type shares it" — and they do, *within* a call. It
was thrown away *between* calls, so every buffer started empty and regrew. The
chroma twin `rd_pick_intra_sbuv_mode` had the same line.

Both pooled (the KB-PERF-48 pattern; neither function has an early return, so
take-at-entry / restore-at-exit is total).

| | allocations |
|---|---:|
| before | 7,378,761 |
| + `SmallVec<[i32; 32]>` coefficients (KB-PERF-49) | 6,922,557 |
| + both scratch pools (KB-PERF-50) | **5,106,723** |
| **total** | **−2,272,038 (−31 %)** |

**Wall: −0.151 % / −0.068 %** against the two base copies (14/24 and 16/24, not
significant; null −0.050 %). Byte-identical on four cells.

**Merged under the standing policy — take allocation reductions that cost under
a percent, because they compound and locality only improves once the churn is
gone.** Note the pool *paid for* the coefficient change: KB-PERF-49 alone
measured **+0.46 %**, and together they are net negative.

**The gate earned its keep here.** The `SmallVec` field type broke nine
differential assertions that compare a txb's coefficients against the C oracle's
`Vec` — and `-p zenav1-aom-encode` passed 772/772 while the integration targets
did not compile. That is KB-42's rule exactly: a crate's unit tests are not the
gate. Fixed by comparing as slices, which preserves each assertion's meaning.

## The two palette sites — the same defect, on a path the photo cells cannot reach

`rd_pick_palette_intra_sby` and `rd_pick_palette_intra_sbuv` each built their own
`IntraTxScratch::default()` **per call**, one call per leaf that reaches the
palette search — the same one-line defect as the two intra mode searches, one
function down. Both are now pooled.

**The chroma pool is NOT taken at entry, and that is the whole subtlety.**
`rd_pick_palette_intra_sbuv` returns early when the colour count misses the
threshold, and the scratch used to be built *before* that return. Taking from the
pool there would hand it back **empty** on every early exit — i.e. it would
reproduce the defect being fixed, on the majority path. The take is placed after
the return instead; the luma twin has no early return at all, so it is total.

Luma and chroma get **separate** statics because the chroma search runs nested
inside `rd_pick_intra_sbuv_mode`, which already holds its own pool. One static
shared between them would have to be re-entrant.

### Reach, stated rather than assumed

Palette is **knob- AND header-gated** (`--enable-palette` plus
`allow_screen_content_tools`), so on the photographic cells every other number in
this record is measured on, these two functions are **never called** and the
change is inert by construction. It is measured where it does fire —
`winperf::SCREEN_GATE_CELL`, 512x384 cq44 s6, `Content::Screen`, palette on:

| | allocations | bytes | coded |
|---|---:|---:|---:|
| base | 251,496 | 154,039,321 | 6,131 B |
| pooled | **241,140** | 150,584,345 | 6,131 B |
| | **−10,356 (−4.1 %)** | −3.5 MB | **identical** |

Two sha256-distinct binaries from one tree. No timing band was run: this is a
4 % cut on a path the default still-image encode does not take, so a wall
measurement on it would be a statement about screenshots, and the byte gates
(`rd_close_palette`, `kb35_nonrd_palette_arm`, `kb37_nonrd_palette_search`, the
KB-41 census) are what carry the correctness claim.

### A documentation defect fixed with it

Inserting a `thread_local!` block immediately above a function puts it **between
that function's doc comment and the function**, which orphans the docs onto the
macro. `rustc` says so (`unused doc comment`) and three sites had it — the two
pools landed above plus KB-PERF-48's `encode_intra.rs`. At
`intra_uv_rd.rs` it was worse: the block landed between
`#[allow(clippy::too_many_arguments)]` and the function, so the **attribute** was
orphaned too and clippy would have started flagging the function. All three
blocks moved above their doc runs; the warning count is zero.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, x86-64. Peak heap and
total bytes were not extracted for the photo cells; only counts. The arena
refactor was not attempted. The palette sites carry no wall measurement.
