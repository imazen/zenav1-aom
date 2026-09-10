# The cycle measured on WINDOWS: −7.1 % to −7.9 %, and it transfers at full strength

**2026-09-10.** GitHub Actions run
[34487059129](https://github.com/imazen/zenav1-aom/actions/runs/34487059129),
`winperf.yml` in `arms: prepost` mode, `base_sha = b1463c8` (KB-PERF-28) to
`HEAD = 69919a1`, **24 rounds x 2 contents x 2 runners**.

| runner | content | effect | rounds | noise floor | vs floor |
|---|---|---:|---:|---:|---:|
| `windows-11-arm` | photo | **−7.885 %** | **24/24** | 0.306 % | 25.8x |
| `windows-11-arm` | detail | **−7.939 %** | **24/24** | 0.231 % | 34.4x |
| `windows-latest` (x86-64) | photo | **−7.130 %** | **24/24** | 0.128 % | **55.8x** |
| `windows-latest` (x86-64) | detail | **−7.677 %** | **24/24** | 0.712 % | 10.8x |

**p < 0.0001 on all four**, and every one is many multiples of its own
same-binary noise floor. `windows-latest` has been unable to resolve sub-1 %
effects all project (KB-PERF-4's and KB-PERF-5's bands died there); at 55x the
floor it resolves this one without difficulty.

## What it settles

The twelve landings over that span compound to **−7.22 % on Linux/glibc**. The
four Windows measurements are **−7.13 % to −7.94 %**.

**The cycle's gains transfer to Windows at full strength — they are not a glibc
artifact.** That was a real open question: this session ran every band on one
Linux box with an unusually fast allocator, and KB-PERF-2 is on record measuring
a lever at 21 % of its win on Darwin and 86-99 % on Windows. For THIS cycle the
platforms agree to well under a percentage point.

**Why they agree, and it is consistent rather than lucky:** the mass of this
cycle is *arithmetic and bounds-check* work — `highbd_subtract_block` (−3.17 %),
`block_error` (−0.88 %), `sum_squares_2d_i16` (−0.35 %), eight fused transform
kernels — none of which touches the allocator. Only two landings were
allocation-shaped, and they are a small share of the total.

## What it does NOT settle

* **The two rejected allocation variants are still unmeasured on Windows.** The
  exact-`with_capacity` (+0.77 % on glibc) and the thread-local pool (+0.23 %)
  are precisely the shape KB-PERF-2 says the platforms disagree about, and
  neither is in this run — this run measures HEAD against a mid-cycle base, and
  both were reverted. Sizing them needs branches and a separate dispatch. The
  same goes for the arena.
* **This is a DELTA, not a ratio.** `winperf` builds
  `--no-default-features` (no C oracle, no cmake, no nasm), so there is no
  libaom arm on these runners and no Windows figure for clause (4) itself.
* **Different cell.** winperf's contents are its own synthetic `photo`/`detail`
  (~293 ms per encode) fitted to the study photograph's mode distribution and
  allocator traffic respectively — not the 1024x1024 mirror-tiled cell the
  Linux bands use (~3059 ms).
* **The base is mid-cycle** (`b1463c8`, KB-PERF-28), not the true cycle start,
  which is why the comparison above is against the 12 landings after it rather
  than all 17.

## Not covered

Two runners, two contents, one round count. macOS is unmeasured — there is no
Darwin runner in `winperf.yml`, and KB-PERF-2's Darwin figures came from a local
box that is not this one.
