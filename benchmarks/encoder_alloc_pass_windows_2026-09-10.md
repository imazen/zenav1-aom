# The allocation pass, sized on Windows — and the platform ratio measured on
# IDENTICAL work for the first time

**Run 34498748007** (`winperf.yml`, `arms: prepost`, `base_sha` =
`7062a06` — the commit before the first allocation landing — vs HEAD `cccb26a`,
24 rounds, contents `detail photo`, both Windows runners), plus a **matched
Linux band built and run locally from the same two commits with the same
harness**.

## Why this run exists

Every band in the allocation pass was taken on one Linux box against glibc, and
KB-PERF-2 is on record that this class of lever measured **21 % of the win on
Darwin and 86-99 % on Windows**. The user's instruction that opened the pass
said so directly: *"we run with slow allocators on mac and win"*. So the pass's
own Linux numbers were known to be a floor, not a value — and the two rejections
inside it (`with_capacity`, the first TLS pool) were rejected on that floor.

## The result

| runner | `detail` | `photo` |
|---|---:|---:|
| **linux x86-64 / glibc** | **−0.350 %** (20/24, p=0.0015, 1.9x floor) | **−0.712 %** (21/24, p=0.0003, 3.4x floor) |
| **windows-11-arm** | **−1.433 %** (24/24, p<0.0001, 6.6x floor) | **−1.188 %** (24/24, p<0.0001, 11.7x floor) |
| **windows-latest x86-64** | **−2.432 %** (20/24, p=0.0015, 5.4x floor) | **−2.249 %** (19/24, p=0.0066, 5.2x floor) |

All six bands are negative and significant. **The same source change is worth
6.9x more on Windows x86-64 than on Linux x86-64 at `detail`, and 3.2x more at
`photo`.**

## What makes this the sharp version of the platform claim

KB-PERF-2 inferred the effect across *different levers on different boxes*. This
run compares **one pair of commits, one harness, one content generator** — and
the allocator censuses settle that the two x86-64 platforms do **identical
work**:

| content | linux x86-64 | windows x86-64 | windows-11-arm |
|---|---|---|---|
| `detail` calls | 408,878 -> 300,515 | **408,878 -> 300,515** | 446,868 -> 317,774 |
| `detail` bytes | 261,032,010 -> 183,750,250 | **identical** | 294,796,362 -> 201,374,826 |
| `photo` calls | 317,426 -> 239,630 | **317,426 -> 239,630** | 342,571 -> 249,511 |

**Identical to the digit on the two x86-64 platforms** — same call count, same
bytes, same buckets. So nothing about the *amount* of work differs, and the
entire 6.9x is **cost per allocator call**: Microsoft's heap charges more for one
than glibc does. (The aarch64 runner differs because a different SIMD tier is
dispatched, which changes scratch shapes; that is expected and is why it is
reported separately rather than merged.)

`framebytes` is **8,734 / 5,301 on every arm of every runner** — the whole pass
is byte-identical on Windows too, not only on the box it was developed on.

## What this says about the two rejections

Both rejected variants in this pass were rejected on a Linux band:

* the exact `with_capacity` hint (**+0.77 %**) removed **zero** allocations and
  added two integer divisions per call. It has nothing to trade, so no allocator
  can rescue it and it stays rejected on every platform.
* the first TLS scratch pool (**+0.23 % Linux**) was **already overturned on
  `windows-11-arm` at −0.600 %** (run 34489034382) and merged under the
  under-a-percent policy. This run is the second, independent confirmation of
  that direction.

**The standing rule this pass was run under — take allocation reductions that
cost under a percent — is measured correct at a ratio, not argued.** On the
platform the encoder ships to, the reductions this pass rejected on a Linux
regression of 0.2 pp were worth roughly 6x that in the other direction.

## Not covered

* winperf's cell is **1024x1024 cq44 `--cpu-used 6`**, not the shipping
  `cq27 --cpu-used 3` cell clause (4) is quoted at. It reaches ~409k allocations
  where the shipping cell reaches ~10.4M, so this band sizes the *platform
  ratio*, not the pass's absolute value at the preset that ships.
* winperf builds `--no-default-features`, so there is no libaom arm and no
  Windows figure for the clause-(4) ratio itself.
* macOS is still unmeasured on this pass; KB-PERF-2's Darwin figure (21 %) is
  the only data point and it is from a different lever.
* The palette pooling (`cccb26a`) is inert on all four contents here — palette
  is knob- and header-gated and none of these sources reach it.
