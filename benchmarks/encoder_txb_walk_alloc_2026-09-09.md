# KB-PERF-13 — three heap allocations per `txfm_rd_in_plane_intra` call, one of them growing element by element

**Landed 2026-09-09.** Byte-identical output; **−0.50 %** at 1024x1024 and
**−0.53 %** at 512x512, each against a same-binary null, each over a rotated
interleaved band. Ratio at the 1 MP profile cell **2.410x → 2.398x**.

## How it was found

The re-profile taken after KB-PERF-9..12 (four landings had moved the ranking,
so §14 required a fresh one) put the **memory class at 559.4 ms**, still the
12x-vs-C outlier the 1 MP class table named. Its composition is what pointed at
the fix:

| symbol | ms |
|---|---:|
| `__memset_avx512_unaligned_erms` | 155.3 |
| `__memmove_avx512_unaligned_erms` | 128.9 |
| `RawVecInner::finish_grow` (two instantiations) | 65.0 |
| `malloc` / `_int_malloc` / `cfree` / `_int_free_merge_chunk` | 124.1 |
| `do_reserve_and_handle` | 19.3 |
| **`RawVec<aom_encode::tx_search::TxbWinner>::grow_one`** | **9.1** |

`grow_one` names a type, and that is the whole lead: a `Vec<TxbWinner>` being
pushed one element at a time, reallocating and memcpying as it goes — which also
explains a share of `finish_grow` and of the `memmove` above it.

## The defect

`txfm_rd_in_plane_intra` (the per-block transform search walk) allocated **three
times per call**:

    let mut t_above: Vec<i8> = env.above_ctx[..max_blocks_wide].to_vec();
    let mut t_left:  Vec<i8> = env.left_ctx[..max_blocks_high].to_vec();
    ...
    let mut winners: Vec<TxbWinner> = Vec::new();   // then `push` per txb

Two of those are tiny copies of the neighbour entropy contexts —
`MI_SIZE_WIDE_B` / `MI_SIZE_HIGH_B` top out at **32** (BLOCK_128X128 in 4x4
units), so they are at most 32 bytes each and never needed a heap at all. The
third grows.

## The fix

* `t_above` / `t_left` become `[i8; 32]`, with a `debug_assert` on the bound.
  **Every use slices to `max_blocks_wide` / `max_blocks_high`** so the lengths
  handed to `get_txb_ctx` and to the per-txb `above:` / `left:` fields are
  exactly the ones the `Vec` form passed — a longer slice would be a silent
  behaviour change, not just a faster one.
* `winners` becomes `Vec::with_capacity(n_txbs)` — one allocation of the exact
  size.

**The count is knowable before the walk**, which is what makes the exact
capacity safe rather than a guess: the walk visits every grid point of the
`txw_unit` x `txh_unit` lattice inside the visible extent, and the mu-64
chunking does not change that set, because a chunk start is a multiple of 16 and
every `txw_unit`/`txh_unit` (1, 2, 4, 8, 16) divides 16. So

    n_txbs = blocks_wide_visible.div_ceil(txw_unit)
           * blocks_high_visible.div_ceil(txh_unit)

Deliberately **not** `max_blocks_*`-based: that would over-allocate to 32x32 =
1024 entries for a TX_4X4 walk on a large block, trading growth for a 24 KB
allocation per call, which is worse.

**Not moved into the scratch**, though `IntraTxScratch` is already threaded into
this function: `winners` is *returned* (`Option<(RdStats, Vec<TxbWinner>)>`), so
reusing a scratch buffer would mean changing the signature and every caller. One
right-sized allocation is the proportionate fix.

## Correctness

No arithmetic changed; only storage and capacity. Byte-identical output verified
directly on two cells before any timing: **39,694 B at 1024x1024 and 10,912 B at
512x512, both arms**. `-p zenav1-aom-encode --lib` 128/128 and the tx-search
integration differentials green.

## Measurement

Two sha256-distinct binaries from one tree, arms **rotated** each round
(playbook §6), a same-binary null (`baseB`) in every band, one encode per arm
per round.

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 10143.8 ms | 10092.3 ms | **−0.50 %** | **14/14** | 0.0001 | +0.13 %, 5/14, p=0.42 |
| 512x512 cq27 s0 | 3640.3 ms | 3621.8 ms | **−0.53 %** | **22/24** | <0.0001 | −0.10 %, 15/24, p=0.31 |

Raw spreads 1.1–1.6 %. Bands: `.band1024.tsv`, `.band512.tsv`.
**Ratio 2.410x → 2.398x** (C median 4208.8 ms, same band).

**The call-count census was NOT taken, and that is a gap worth naming.** For an
allocation fix the number of calls removed is stronger evidence than a timing
delta, and the instrument exists (`examples/eprof_alloc.rs`, which KB-PERF-1 and
KB-PERF-2 both used) — but it takes a `.yuv` path and this cell is the in-repo
mirror-tiled `av1-1-b8-01-size-196x196`, which the harness synthesises rather
than reading from a file. Wiring `eprof_alloc` to `EncodeCell` would make every
future allocation lever cheaper to prove and is the follow-up.

## Not covered

* **`__memset` 155.3 ms and `__memmove` 128.9 ms remain the bulk of the class**
  and are untouched here. KB-PERF-2's lesson applies directly: *"most of the mass
  of `alloc/libc` is `memset`/`memcpy`, not allocator bookkeeping — split them
  before crediting any lever with the stage total."* This landing addressed the
  bookkeeping half.
* Other `to_vec()` / `Vec::new()` sites on hot paths were not swept; this was one
  function, chosen because `grow_one` named its type.
* One box, one content class, one speed, `--cpu-used 0`, x86-64.

## Gate status

`just gate-landing` green in full — `test-next` 1503/1503,
`test-next-scalar` 1503/1503, `census-gate` 4/4, `test-whereat` 4/4, all exit 0.
