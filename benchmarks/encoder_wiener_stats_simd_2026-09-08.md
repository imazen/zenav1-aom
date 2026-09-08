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

---

# Second lever, same stage: `pixel_proj_error` — 2.641x -> 2.573x

The SGR projection error, the second of the three loop-restoration kernels the
x86-64 profile found with no SIMD tier: **17.8 ms against libaom's
`av1_lowbd_pixel_proj_error_avx2` at 2.0 ms**.

## Measured

Same cell and protocol (two binaries from one tree, arm order rotated):

| | min | median |
|---|---:|---:|
| before (= the Wiener-stats landing) | 468.19 ms | 469.28 |
| after | **454.97 ms** | 457.18 |
| libaom-c | 176.95 ms | — |

**Paired median −2.47 %, faster in 6 of 6 rounds.** Ratio **2.641x → 2.573x**.

**Session cumulative: 483.01 → 454.97 ms, −5.8 %, ratio 2.733x → 2.571x.**

## The trade that was NOT taken, and why

The fast form squares in `i32` lanes and reduces per chunk. That needs
`8 * e^2 < 2^31`, i.e. **`|e| < 16384`** — and working the arithmetic through
(`xq` reaches ~96 via `SGRPROJ_PRJ_MIN0/MAX0`, `flt - u` reaches ~2^16 at bd12,
so `v` reaches ~2^24 and `e` ~2^13..2^14) puts `|e|` **at** that bound rather
than comfortably inside it.

`restore/pick.rs` feeds RD decisions and therefore the encoder byte gates, so the
shipped form squares and accumulates in **the scalar tier's own order** —
`err += e as i64 * e as i64` over `j` ascending — and vectorizes only the
arithmetic that produces `e`. That is bit-exact **by construction**, not within a
margin, and needs no bound at all.

Anyone who wants those milliseconds should DERIVE the bound from
`SGRPROJ_PRJ_MIN0/MAX0` and the SGR output range, then gate it at runtime the way
`intra/dir_simd.rs` gates its tap bound — that file is the worked precedent for a
data-dependent gate with reach and bite pins on both sides.

## What is left on this stage

* SGR `calculate_intermediate` + `selfguided_restoration`, **28.1 ms, scalar** —
  now the largest scalar item in loop restoration.
* `compute_stats` at 17.5 ms wants the register-blocked rewrite (above), not more
  lanes.
* bd10/12 remains scalar throughout: `compute_stats_highbd` is a separate i64
  loop, and `pixel_proj_error`'s highbd arm shares this tier but was measured
  only through the bd8 cell.
