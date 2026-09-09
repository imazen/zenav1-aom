# magetypes/archmage 0.9.28 -> 0.9.29 — the primitive three landings were blocked on

**Landed 2026-09-09.** A capability landing: byte-identical output, full gate
green, and — measured rather than assumed — **−0.48 % on its own** (17/20
rounds, p=0.0026) because 0.9.29's backends are faster for the kernels already
written against them.

## Why

Three separate landings this cycle stopped at a missing primitive, and each
recorded it:

* **KB-PERF-7** — *"the workspace pins magetypes 0.9.28, whose `i16x16` has no
  `madd_adjacent` (it lands in 0.9.29)"*, so the Wiener `compute_stats` fold
  could not take libaom's `_mm256_madd_epi16` shape.
* **KB-PERF-15** — magetypes 0.9.28 has **no i32->u8 narrowing primitive** at
  all, so `txb_init_levels` could not take libaom's
  `_mm256_packs_epi32` + `_mm256_packus_epi16` shape and stores a built 8-byte
  run instead.
* **KB-PERF-8** — no runtime-variable shifts, worked around with a
  `blend`-chain selecting between const-shift results.

And the largest *unaddressed* item in the loop-restoration class has the same
cause: `wiener_impl_v3` is **192.9 ms against `av1_wiener_convolve_add_src_avx2`
25.5 ms — 7.6x, the worst ratio in the class** — because libaom convolves in
i16 with `madd_epi16` (16 lanes, two taps per instruction) where the port uses
`i32x8` with a separate multiply and add per tap. That is ~4x the work per
pixel, and no amount of restructuring closes it without the primitive.

## What 0.9.29 adds that this tree wants

`madd_adjacent`, `pairwise_widen_add`; `narrow_saturating_i32_to_u16`,
`narrow_saturating_i16_to_u8` (+ the i8/u8/i16/u16 family);
`widen_low` / `widen_high` (+ typed variants); `shl_uniform`,
`shr_arithmetic_uniform`, `shr_logical_uniform` (runtime shifts);
`saturating_add` / `saturating_sub`; `abs_diff`, `sum_abs_diff`;
`reduce_add_u32`; `bitcast_i16x32` / `bitcast_u16x32`.

**Still absent, so do not plan on them:** shuffle, permute, unpack and gather.
KB-PERF-10's in-register 8x8 transpose and KB-PERF-8's table gather remain
inexpressible.

## Cost of the bump

The manifest already required `"0.9.27"` (a caret range that permits 0.9.29), so
**only `Cargo.lock` changed** — no manifest edit, and `archmage` /
`archmage-macros` move with it.

## Verification

* **Byte-identical at three cells** before anything else: 39,694 B at
  1024x1024, 10,912 B at 512x512, 1,177 B at 192x192.
* `just gate-landing` green in full — `test-next` **1503/1503**,
  `test-next-scalar` **1503/1503** (the leg that exercises the scalar tiers
  through the byte gates), `census-gate` 4/4, `test-whereat` 4/4, exit 0. That
  is the check this landing exists for: the bump touches every SIMD kernel in
  the tree at once.
* **Not a regression, and in fact an improvement** — a dependency bump can
  silently change backend codegen, so it was A/B'd like any other lever: rotated
  arms, same-binary null, 512x512 cq27 s0, 20 rounds.

  | arm | median | paired | rounds faster | p |
  |---|---:|---:|---:|---:|
  | base (0.9.28) | 3615.3 ms | — | — | — |
  | null (same binary) | 3616.2 ms | +0.09 % | 8/20 | 0.5034 |
  | **0.9.29** | **3597.1 ms** | **−0.48 %** | **17/20** | **0.0026** |

## What this unblocks (none of it done here)

1. `wiener_impl` in i16 with `madd_adjacent` — the 7.6x, ~167 ms.
2. `acc_stat_line`'s fold in i16 — KB-PERF-7's own named follow-up.
3. `txb_init_levels` with a real `narrow_saturating` pack — KB-PERF-15's.
4. KB-PERF-8's blend-chain shift select, replaceable by `shr_logical_uniform`.

Each needs its own bit-exactness argument and its own band; this landing only
makes them possible.
