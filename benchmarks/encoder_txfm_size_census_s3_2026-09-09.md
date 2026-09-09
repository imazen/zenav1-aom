# The transform census at the SHIPPING preset — and the inverse mix, which had never been measured

**2026-09-09. No code changed.** `encoder_txfm_size_census_2026-09-09.md`
measured the forward transform size distribution at **`--cpu-used 0`** and closed
by naming two gaps in itself:

> *"this is the FORWARD side ... its size distribution is NOT measured here —
> `note_fwd_txfm` has no inverse twin. Adding one is the obvious next census,
> and it should be done before assuming the inverse mix matches."*

> *"this is a 196x196 cell. The distribution is a property of the SEARCH ...
> so it should carry to 1 MP — but that is an argument, not a measurement."*

KB-PERF-17 added the inverse twin (`note_inv_txfm`). This runs both sides at
speed 0 **and at `--cpu-used 3`, the preset zenavif actually ships**
(`encoder_shipping_preset_1mp_2026-09-09.md`). Data:
`encoder_txfm_size_census_s3_2026-09-09.tsv`.

Method: `content_census --speed {0,3} --cq 27
real:av1-1-b8-01-size-196x196:196x196`, built
`--no-default-features --features c-oracle,census`. **The speed-0 forward run
reproduces the committed record exactly** — 1,196,534 transforms, 4x4 50.70 % —
which is the check that the tool and the invocation are the same ones.

## The distribution, both sides, both speeds

| tx size | s0 fwd | s0 inv | **s3 fwd** | **s3 inv** |
|---|---:|---:|---:|---:|
| **4x4** | 50.70 % | 44.43 % | **40.51 %** | **40.26 %** |
| **8x8** | 22.30 | 25.66 | **25.74** | **26.96** |
| 8x4 | 6.72 | 6.35 | 9.13 | 8.00 |
| 4x8 | 5.64 | 5.87 | 7.60 | 7.16 |
| 16x16 | 4.80 | 5.85 | 5.62 | 5.04 |
| 16x8 | 2.96 | 3.58 | 5.14 | 5.54 |
| 8x16 | 2.06 | 2.60 | 3.88 | 3.98 |
| 16x4 | 2.30 | — | 1.02 | 1.05 |
| 4x16 | 1.71 | — | 0.43 | 0.61 |
| both dims >= 32 | 0.55 | ~0.9 | 0.64 | 0.96 |
| **total calls** | **1,196,534** | **866,354** | **303,588** | **188,040** |

## 1. The inverse mix DOES match the forward — measured, not assumed

At the shipping preset the two sides agree to within a point on every size
(4x4 40.51 vs 40.26; 8x8 25.74 vs 26.96). At speed 0 they differ more (50.70 vs
44.43) but in the same order. **The assumption the earlier record flagged as an
argument is now a measurement, and it held.** Anything that specialises by size
can be reasoned about once and applied to both directions.

## 2. The distribution SHIFTS with speed, against the small sizes

Going s0 -> s3:

* **4x4 falls 50.70 % -> 40.51 %** of forwards;
* 8x8 rises 22.30 -> 25.74; the rectangles rise together (8x4 6.72 -> 9.13,
  16x8 2.96 -> 5.14, 8x16 2.06 -> 3.88);
* **4x4 + 8x8 falls 73.0 % -> 66.3 %**;
* **`fwd_tx_4pt` (any 4-point dimension) falls 67.08 % -> 58.69 %**;
* **`fwd_tx_non_dct` falls 76.24 % -> 61.66 %**.

Total forward calls fall **3.94x** (1.20 M -> 304 k), which is the speed preset
doing its job.

**The consequence, and it is the point of running this: KB-PERF-16 and
KB-PERF-17's fused 4x4 paths were sized and measured at speed 0, where 4x4 is
half of every transform. At the preset that ships they cover 40.5 %, so they
reach about a fifth less than their landing records imply.** They are still the
single best-covered specialisation available — no other size is close — but
their reach should be quoted with a speed, exactly as the ratio now is.

The `fwd_tx_non_dct` move matters for the opposite reason: a DCT-only fast path
misses 76 % of calls at s0 and 62 % at s3. Still most, so the earlier record's
conclusion — **do not build a DCT-only specialisation** — survives at the
shipping preset, with a smaller margin.

## 3. What this says about the remaining transform levers

Transform is **+789 ms, 40.5 % of the shipping-preset gap, 3.78x** — the #1
item at s3 exactly as at s0. Sized against this census, from the s3 profile:

| candidate | reach at s3 | measured size | verdict |
|---|---|---:|---|
| driver round-trip (`av1_fwd_txfm2d_into` 58.4 + `av1_inv_txfm2d_add_into` 54.2 + the two gate fns 58.4 + `get_fwd_txfm_cfg` 9.8) | all calls | **~181 ms, no C counterpart** | the largest structural item left |
| fused 4x4 (LANDED) | 40.5 % | — | best-covered specialisation |
| fused 8x8 | 25.7 % | — | **REJECTED, +7.07 %** — at 8x8 both SIMD passes run full-width, so scalar fusion trades the SIMD away. Size-structural, so speed does not change it; do not retry |
| forward ROW pass `row_n == 4` arm (KB-PERF-9's named gap) | **8x4 9.13 % + 16x4 1.02 % = 10.2 %** (4x4 is fused, so it does not go through the driver) | **~21 ms** of scalar `av1_fdct8`/`fadst8`/`fdct16`/`fadst16` | ~1 % of the gap, and KB-PERF-9 warns its row-major loads are the gather case that measured **+9.8 %** on the other pass. Small and not obviously positive |

**Two things were ruled out by reading the profile rather than by assuming.**
`run_inv1d_v3` (107.8 ms) and `run_fwd1d_v3` (81.5 ms) look like per-batch
12-arm dispatchers, and they are — but the kernel bodies are INLINED into them
(no `av1_idct8_impl` symbol exists separately), so that 189 ms is the butterflies
themselves, not dispatch overhead. And the i16 applicability scans
(`fwd_row_i16_applies` is an O(col_n x row_n) max-abs pass) run only where
`row_n % 16 == 0 && col_n % 8 == 0`, i.e. **~10 % of calls**, so they are not
the 58.4 ms of self-cost in the two gate functions either.

## Honest scope

196x196, cq27, bd8 4:2:0, one content class, one box. The distribution is a
property of the SEARCH, so it should carry to 1 MP — that is still an argument
and not a measurement, because `content_census` takes a crop of the source
vector and cannot mirror-tile to 1 MP. What IS now measured is the axis that was
actually in doubt: **speed**, and it moves the distribution by ten points.
