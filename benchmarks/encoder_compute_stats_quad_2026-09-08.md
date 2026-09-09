# `compute_stats`: four-pixel fold of the `H` accumulator — 2026-09-08

KB-PERF-7. The step KB-PERF-6 pre-registered ("not wider lanes but register
blocking ... `H` touched once per tile instead of once per pixel"), sized by that
entry's own 2026-09-08 correction at **446 ms on a 1 MP frame**, the largest
loop-restoration symbol and the worst kernel ratio (8.2x) in the x86 profile.

## What changed

`aom_dsp::restore::pick::acc_stat_line_impl` folds FOUR adjacent source columns
before each `H` read-modify-write. New `gather_window_quad` spans `win + 3`
columns rather than `4 * win` (70 plane loads at win7 against 196). Per 8 `H`
elements: 4 x (load y, mul, load H, add, store H) -> (4 x load y, 4 x mul,
3 add, load H, add, store H) — 20 ops to 13, one quarter of the `H` traffic.

Bit-exactness is by construction but on a WEAKER argument than the tier it
replaces, and the source says so: the previous form reassociated nothing;
this one folds `h += p0; h += p1; h += p2; h += p3` into
`h += ((p0 + p1) + p2) + p3`. Integer addition is associative — over `i32` even
on overflow, wrapping add being a group operation — so accumulators end equal
and the products are untouched.

libaom folds PAIRS with `_mm256_madd_epi16`. That instruction is unreachable
here: the workspace pins **magetypes 0.9.28**, whose `i16x16` has no
`madd_adjacent` (added in 0.9.29), and `aom-dsp` is `#![forbid(unsafe_code)]`.
It does not matter — KB-PERF-6 measured that the 16-bit multiply is not the
limit and the per-element `H` RMW is.

## Method

Two binaries from one tree (`git stash` the hunk, build, restore, build).
Arms interleaved and ROTATED one position per round; a same-binary null arm
(`baseB`) in every band; every arm's output byte-length identical and equal to
libaom's. Driver `crates/aom-bench/examples/eprof_x86.rs`, 1 warm + N timed
encodes per invocation, `nice -n 19`.

## Result

| cell | base | quad | libaom-c | ratio | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 192x192 cq27 s0 | 453.60 ms | 446.04 ms | 178.50 ms | 2.5412x -> **2.4988x** | **-1.68 %** | 20/20 | 1.9e-6 | +0.21 % |
| 1024x1024 cq27 s0 | 10810.31 ms | 10600.42 ms | 4121.87 ms | 2.6227x -> **2.5717x** | **-2.00 %** | 8/8 | 0.0078 | -0.04 % |

Raw round-by-round: `.band192.tsv`, `.band1024.tsv` (arm, round, position,
per_ms, bytes). Per-invocation spread 0.5-1.7 %. The 192x192 base arm
reproduces the published 2.542x to four digits, so both deltas are read against
a live control rather than a remembered number.

## Attribution — by symbol, not from the wall

The KB-PERF-6 roll-up error was subtracting micro-measured symbol deltas from a
stage total; this is the check that avoids repeating it. `perf record -F 499 -g`
on one 1024x1024 encode per arm:

| symbol | base | quad |
|---|---:|---:|
| `restore::pick::acc_stat_line_impl_v3` | **4.01 %** | **1.90 %** |
| `restore::sgr::calculate_intermediate` | 2.41 % | 2.45 % |
| `restore::pick::pixel_proj_error_impl_v3` | 1.99 % | 1.86 % |
| `restore::wiener::wiener_impl_v3` | 1.73 % | 1.74 % |
| `restore::sgr::calculate_intermediate::{closure#0}` | 1.32 % | 1.48 % |

4.01 % of 10810 ms = 433 ms -> 1.90 % of 10600 ms = 201 ms, i.e. **-232 ms**
against a **-210 ms** wall delta (consistent within sampling). Nothing else in
the stage moves. **Kernel 2.2x faster; ratio to `compute_stats_win7_avx2` +
win5 + `_c` (54.4 ms) 8.2x -> ~3.7x; no longer the largest LR symbol.**

## The size regime, measured from the other direction

The lever is worth MORE at 1024x1024 (-2.00 %) than at 192x192 (-1.68 %). At
192x192 the bd8 u16 planes are ~108 KiB and L2-resident, which is exactly the
regime the clause-(4) row already flags as under-measuring a traffic lever —
and there the i16 inverse null was measured. Here the same regime shows the
opposite sign because the lever genuinely reduces traffic. Quote ratios with
their cell.

## What is now the largest LR symbol, and why it is a different kind of change

`calculate_intermediate` (2.45 %), whose hot loop is
`X_BY_XPLUS1[z.min(255)]` — a 256-entry table lookup. Verified first-hand:
**magetypes contains no `gather`**, in 0.9.28 or 0.9.29, so there is no vector
form of that lookup in this vocabulary. Closing it means replacing the table
with arithmetic (`256*z/(z+1)`, exhaustively checkable over 256 inputs) and
proving equality over the whole domain — not a lane-width change.

Also checked rather than assumed: `magetypes/src/simd/impls/x86_v4.rs`
implements only 512-bit backends (`I32x16Backend`, `I16x32Backend`, ...), so
adding `v4` to an existing `define(i32x8)` kernel cannot compile — AVX-512 is a
separate kernel body, not a tier addition.

## Not done

`compute_stats_highbd` (bd10/12, separate i64 loop). The 16-bit multiply half
of libaom's fold (needs magetypes >= 0.9.29 or an equivalent widening
multiply-add). Loop restoration as a stage is +1406 ms of the 1024x1024 gap
before this landing — this is one lever inside it.
