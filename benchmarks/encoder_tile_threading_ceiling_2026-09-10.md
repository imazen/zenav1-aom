# The multithreading ceiling, measured — and it does NOT change the ratio

Cell: 1024x1024 cq27 `--cpu-used 3` (the preset zenavif ships), bd8 4:2:0,
CDEF off / restoration on, x86-64 Linux, one box. Instrument:
`crates/aom-bench/examples/eprof_x86.rs` (7th arg = tile grid, `N` square or
`C,R`) and `perf record --call-graph fp` on a `force-frame-pointers` build.

## 1. The port spawns no threads, and NEITHER DOES THE ORACLE — so "compare with C" is not measurable here

Pinned three ways, verified rather than assumed: the oracle is built
`-DCONFIG_MULTITHREAD=0` (`build.rs:414`, `reference/BUILD_CONFIG.md`) AND sets
`cfg.g_threads = 1` at runtime (`dec_shim.c:498`), and `grep` for
`thread::spawn`/`rayon` across `aom-encode`/`aom-dsp` is empty.

**So every band in this repo is single-thread vs single-thread, and this repo
CANNOT measure a threaded libaom.** That is the first honest answer to "compare
with C on threading": the comparison does not exist yet, and building it means
a second oracle build with `CONFIG_MULTITHREAD=1`.

## 2. The parallel fraction is 92 %, and it is stable

`pack_tile_lr_stop` is the per-tile subtree. Inclusive share of the encode:

| tile grid | pack_tile subtree |
|---|---:|
| 1 tile | **91.44 %** |
| 4 tiles (1,1) | **92.10 %** |
| 16 tiles (2,2) | **92.22 %** |

The ~8 % serial remainder is DIFFUSE — no symbol above 0.4 % — and is
frame-level work: the loop-restoration search (frame-global by construction, so
it cannot be tiled), frame setup, border extension and header assembly. Its
decomposition was not pursued further.

## 3. Tiles are not free, in BYTES or in SERIAL TIME

Each tile resets the entropy contexts and re-pays per-tile setup, so tiling
costs on both axes even before a thread is spawned:

| grid (cols,rows log2) | tiles | bytes | Δ bytes | 1-thread ms | Δ ms |
|---|---:|---:|---:|---:|---:|
| 0,0 | 1 | 40,237 | — | 2984.1 | — |
| **0,1** | **2 rows** | 40,619 | **+0.95 %** | 2994.7 | +0.36 % |
| **0,2** | **4 rows** | 41,035 | **+1.98 %** | 3004.3 | +0.68 % |
| **0,3** | **8 rows** | 41,493 | **+3.12 %** | 3042.6 | +1.96 % |
| 1,1 | 4 square | 41,086 | +2.11 % | 3042.5 | +1.96 % |
| 2,2 | 16 square | 42,399 | +5.37 % | 3098.5 | +3.83 % |

**ROW-ONLY TILING IS CHEAPER THAN A SQUARE GRID AT THE SAME TILE COUNT, on both
axes** — 4 row tiles cost +1.98 % bytes / +0.68 % time against 4 square tiles'
+2.11 % / +1.96 %. That matters because row tiles are also the only shape
reachable safely (§5), so the cheap option and the implementable option are the
same one. That was not obvious in advance and is the reason to measure the two
spellings separately.

## 4. The ceiling: ~3.2x at 4 threads, ~5.1x at 8

Amdahl on the measured p = 0.921, applied to each grid's own single-threaded
time (so the tiling overhead is paid, not hidden):

| config | projected ms | vs 1-tile baseline | vs C (1548.93 ms) |
|---|---:|---:|---:|
| 1 tile, 1 thread (today) | 2984 | 1.00x | **1.93x** |
| 4 row tiles, 4 threads | **929** | 3.21x | **0.60x** |
| 8 row tiles, 8 threads | **591** | 5.05x | **0.38x** |
| 16 square, 16 threads | 420 | 7.11x | 0.27x |

**These are CEILINGS.** Amdahl assumes perfect load balance and tiles are not
equal work — a photographic frame's tiles differ in coded cost, so real speedup
is lower, and by an amount nothing here measures.

## 5. THE FINDING THAT DECIDES THE WORK: row tiles need no `unsafe`

The per-tile loop already exists (KB-31 wired the multi-tile frame driver and it
is byte-exact against real aomenc). The blocker is not the algorithm — it is
that `aom-dsp` and `aom-encode` are `#![forbid(unsafe_code)]` and the tiles
share one `&mut [u16]` recon plane, whose disjointness the borrow checker cannot
prove.

* **COLUMN tiles interleave within every row**, so their regions are not
  contiguous slices at all. Threading them needs per-tile staging buffers plus a
  copy-back, or a chunked row split — real work, and it reintroduces the copies
  this cycle has been deleting.
* **ROW tiles own contiguous row BANDS.** `split_at_mut` hands out disjoint
  `&mut` bands with zero `unsafe`, which is exactly the shape the recon plane
  has. Together with §3 (row tiles are also the cheapest) this is the path.

## 6. THE CONCLUSION, AND IT CUTS AGAINST SPENDING THE SESSION ON IT

**Threading changes ABSOLUTE time, not the RATIO.** libaom threads too (tile
threading plus row-mt), so a threaded-vs-threaded comparison at matched thread
count would leave clause (4)'s ratio near 1.94x — threading moves both arms.

So which bar is being measured decides whether this is the top lever or a
non-lever:

* **as a RATIO at matched thread count** — the reading every band in this repo
  uses — threading is **neutral**, and does nothing for clause (4);
* **as ABSOLUTE wall time for one still image** it is by far the largest lever
  available: ~3.2x at 4 threads against the **1.29x** that closing EVERY named
  kernel lever completely would give (+408.5 ms of a 1462.7 ms gap,
  `encoder_lever_map_s3_2026-09-10.md`).

`CLAUDE.md`'s numeric caveat says the 1.5x bar is the reading "until the user
says otherwise", and the bar is stated without a thread count. **That ambiguity
is now worth resolving, because the two readings rank the work differently** —
under the ratio reading the next move is more kernel breadth; under the absolute
reading it is row-tile threading, which is one `split_at_mut` away and buys
3.2x for +1.98 % bytes.
