# A SIMD tier for the Wiener stats inner loop — 2026-09-08

The first encoder perf lever taken from the x86-64 profile
(`benchmarks/encoder_x86_profile_2026-09-08.md`), and the first encoder perf
work in this repo measured on Linux/x86-64 rather than aarch64-apple-darwin.

## Why this one

That profile's second-largest lever: **loop-restoration search is 26 % of the
speed-0 gap to libaom and had never been profiled**, because libaom disables
Wiener + SGR at `speed >= 5` (`speed_features.c:519-520`) and every prior
profile in this repo was taken at `--cpu-used 6`, where the stage is
structurally absent. Three of its four kernels carried no SIMD tier at all.

`compute_stats` was the largest of them and the largest scalar-only function in
the encoder: **32.2 ms of a 485 ms encode against libaom's
`compute_stats_win7_avx2` at 2.4 ms**.

**Before writing any SIMD, the profile's own lesson was applied** (KB-PERF-1:
the single biggest perf finding in this project was *redundancy*, not slowness).
Counted: 26 calls, 165,888 stat pixels, **160.25 M multiply-accumulates per
encode**, retiring at **4.98 G/s ≈ 1.06 per cycle**. No redundancy — the work is
real and purely scalar-bound. libaom does the same 160 M in ~2.4 ms ≈ 14 per
cycle, which is `_mm256_madd_epi16`'s rate.

## Bit-exact by construction, not by luck

The vectorization is across elements `l` of one `H` row (and across `k` for
`M`), **never across the pixel loop `j`** — so every accumulator receives the
same products in the same order as the scalar tier. Integer products, integer
adds, no saturation, no reassociation. The scalar tier is retained verbatim as
the reference and `tests/pick_diff.rs` compares BOTH tiers against the real
exported C.

Accumulator width is C's own: `|y|, |x| <= 255` at bd8, so a product is at most
65025 and a row of at most 256 pixels stays under 16.7 M — inside `i32`, which
is why C uses `int32_t` here too.

## Measured

Cell: `av1-1-b8-01-size-196x196` cropped 192x192, cq27, bd8 4:2:0,
`--cpu-used 0`, CDEF off / restoration on. **Both arms and both builds emit the
same 1177-byte stream.** Two binaries built from the same tree with only this
change, run interleaved with the arm order ROTATED each round so an arm is not
confounded with its position:

| | min | median | spread |
|---|---:|---:|---:|
| before | 483.01 ms | 483.30 | 0.86 % |
| after | **466.79 ms** | 468.85 | 1.02 % |
| libaom-c | 176.75 ms | 177.51 | 1.96 % |

**Paired median −3.13 %, faster in 8 of 8 rounds** (sign test p = 0.008).
Ratio to libaom **2.725x → 2.639x**; the per-round ratio ranges do not overlap
(before 2.681-2.745, after 2.617-2.647).

Stage cost: `compute_stats` **32.2 ms → 17.5 ms (1.84x)**.

## Two hypotheses tested and REFUTED — do not re-spend them

8-wide lanes did not buy 8x, and the two obvious explanations are both wrong:

* **"the scalar tails dominate."** At win7 the per-`k` tails are ~171 scalar
  madds per pixel against only 132 vector iterations, which looks decisive.
  Removing them entirely — padding the `H` row stride to a whole vector so a
  vector starting at any `l < win2` stays inside its row — was worth
  **0.9 ms of 18.4**.
* **"it is L1 bandwidth on the `H` accumulator."** The read-modify-write streams
  ~1.25 GB per encode, which is **71 GB/s** — about 16 % of this core's L1
  ceiling. Not the limit.

What remains is the shape of the loop: ~2.8 cycles per vector iteration for a
load + load + multiply + add + store, i.e. the per-element read-modify-write of
`H` itself.

## A variant that was built, measured and REJECTED

Starting each row's sweep at `k & !7` instead of `k` makes both slices a whole
number of 8-lane chunks, so `chunks_exact` drops the per-iteration bounds check
— which this crate can only remove structurally, being `#![forbid(unsafe_code)]`.
It is **correct** (the extra lanes at `l < k` land in row `k`'s lower triangle,
which is zeroed per source row and never read — the fold walks `l in k..win2`)
and it is **slower**: at win7 it costs **217 vector iterations per pixel against
175, +24 %**, which the removed bounds checks do not pay for. Measured
**471.6 ms against 468.7**. Reverted, with the arithmetic recorded at the site
so it is not re-tried.

## The next step is register blocking, not wider lanes

libaom's `acc_stat_win7_one_line_avx2` holds `H` tiles in registers ACROSS the
pixel loop and folds pairs of pixels with `_mm256_madd_epi16`, so `H` is touched
once per tile instead of once per pixel. That is a different loop nest, not a
tweak to this one. Vectorizing over `j` (pixels) rather than over `l` is the
same idea and stays bit-exact, because integer addition is associative.

## Honest limits

* **bd10/12 is still scalar.** `compute_stats_highbd` is a separate loop that
  accumulates directly into `i64` (C's own structure for the highbd variant) and
  shares none of this code. Untouched.
* One cell, one content, bd8 4:2:0, `--cpu-used 0`. The stage is structurally
  absent from speeds >= 5, so this lever cannot move the speed-6 number at all.
* `pixel_proj_error` (17.8 ms) and the SGR `calculate_intermediate` /
  `selfguided_restoration` pair (28.1 ms) remain scalar — together still larger
  than what this landing addressed.

## Reproduce

```
cargo build --release -p zenav1-aom-bench --example eprof_x86
./target/release/examples/eprof_x86 port 192 192 27 0 10
./target/release/examples/eprof_x86 c    192 192 27 0 10
```
