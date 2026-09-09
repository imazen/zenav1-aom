# The transform programme, measured: 4x4 is HALF the forward transforms

**2026-09-09.** The ranking has called transform *"the largest and still a
programme: libaom's size-specialised whole-transform entry points against the
port's generic driver"* since the 1 MP re-profile — **+2263 ms, 38.3 % of the
gap**, and the only class large enough that clause (4) cannot be satisfied
without it. That framing is right and is not actionable: "size-specialised" does
not say WHICH size. This measures it.

Method: the committed `aom_dsp::census` already records `[tx_type][tx_size]` at
`av1_fwd_txfm2d_into` (`note_fwd_txfm`), so the distribution is free —
`content_census --speed 0 --cq 27 real:av1-1-b8-01-size-196x196:196x196`, the
differential corpus at the profile cell's speed and quantizer.

## The distribution

**1,196,534 forward transforms on a 196x196 frame** — 31 per pixel, which is
what a speed-0 tx search costs.

| tx size | count | share |
|---|---:|---:|
| **4x4** | 606,616 | **50.70 %** |
| 8x8 | 266,786 | 22.30 % |
| 8x4 | 80,442 | 6.72 % |
| 4x8 | 67,464 | 5.64 % |
| 16x16 | 57,456 | 4.80 % |
| 16x8 | 35,410 | 2.96 % |
| 16x4 | 27,580 | 2.30 % |
| 8x16 | 24,624 | 2.06 % |
| 4x16 | 20,508 | 1.71 % |
| everything with BOTH dims >= 32 | 6,556 | **0.55 %** |

**4x4 alone is half of every forward transform; 4x4 + 8x8 is 73 %.** Any
transform with both dimensions at least 32 is together under 1 %, so the large
sizes — the ones a "whole-transform" rewrite instinctively starts with, because
their kernels are the biggest — are worth almost nothing here.

Type mix: DCT_DCT 23.76 %, DCT_ADST 19.54 %, ADST_DCT 19.23 %, ADST_ADST
18.55 %, H_DCT 13.09 %, V_DCT 4.71 %, IDTX 1.12 %. The four DCT/ADST
combinations are **81.1 %**, so a DCT-only fast path would miss three quarters
of the calls (`fwd_tx_non_dct` reads **76.24 %**).

## What this says the lever IS — and it is not lane width

For a 4x4 the port's generic driver spends, per call: `get_fwd_txfm_cfg` (a
match plus table lookups), the scratch `clear` + `resize`, a column-pass
dispatch, a write into the intermediate `buf`, a row-pass dispatch reading it
back, the output write, and the `tx_size` post-process match — **all to produce
sixteen coefficients.** KB-PERF-14 already measured the config derivation alone
at `get_fwd_txfm_cfg` 46.7 + `get_inv_txfm_cfg` 34.5 = **81.2 ms with NO C
counterpart at all**, because libaom's entry points are size-specialised at
compile time and never derive a config. `av1_fwd_txfm2d_into`'s own self-cost is
a further 206 ms, and KB-PERF-13's annotate found that class of symbol to be
diffuse per-call setup rather than a hot loop.

So the 4x4 target is **per-call overhead**, and two consequences follow:

* **KB-PERF-3's `fadst4` rejection does not block it.** That audit rejected
  fadst4 for the *i16* path (it works in a pre-shift domain, `M*` is 1-11). A
  fused 4x4 that keeps the port's existing i32 arithmetic and removes the
  driver round-trip is unaffected, and it can serve all four DCT/ADST
  combinations — 81 % of types — rather than DCT-only.
* **The transform class does not need 19 specialised kernels.** Two sizes cover
  73 % of the calls. That is the difference between "a much larger programme"
  and a bounded one.

> **FOLLOW-UP TAKEN, same day — both gaps this section names are now measured**
> (`encoder_txfm_size_census_s3_2026-09-09.md`). The inverse twin exists
> (KB-PERF-17's `note_inv_txfm`) and **the inverse mix DOES match the forward**
> — at `--cpu-used 3` they agree to within a point on every size — so the
> assumption flagged below held. The SPEED axis, however, moves the
> distribution by ten points: at the preset zenavif ships (`--cpu-used 3`)
> **4x4 is 40.51 % of forwards, not 50.70 %**, 4x4+8x8 is 66.3 % not 73.0 %,
> and `fwd_tx_non_dct` is 61.66 % not 76.24 %. **The 4x4 conclusion below
> survives — no other size is close — but its reach should be quoted with a
> speed, and KB-PERF-16/17's fused paths cover about a fifth less at the
> shipping preset than at the speed 0 they were measured at.** The frame-size
> half is still an argument, not a measurement.

## Honest scope

This is the FORWARD side. The inverse is larger in the profile
(`run_inv1d_v3` 472 ms, `inv_col_pass_core_v3` 374, `inv_row_pass_core_v3` 229,
`av1_inv_txfm2d_add_into` 127 against the forward's 259/188/128/206), and its
size distribution is NOT measured here — `note_fwd_txfm` has no inverse twin.
Adding one is the obvious next census, and it should be done before assuming the
inverse mix matches.

Also: this is a 196x196 cell. The distribution is a property of the SEARCH (how
many candidate transforms each leaf tries), which is speed- and content-driven
rather than frame-size-driven, so it should carry to 1 MP — but that is an
argument, not a measurement, and the 1 MP census has not been run.

## What this does NOT change

The arithmetic that makes this the only path: the gap is 5906 ms and the bar
needs it at ~2111 ms, so **-3795 ms**. Quantize's whole +171 ms (2.9 %) cannot
reach it even fully closed, and it is blocked on a missing magetypes `mul_high`
besides. Transform at +2263 ms is the only class big enough to matter, and this
census says where inside it to start.

---

# CORRECTION 2026-09-09: 8x8 was BUILT, measured **+7.07 %**, and REVERTED

**The census above ranks by CALL COUNT, and that is not sufficient to pick a
fusion target.** This section is the measurement that proves it, written into
the census record itself so the ranking cannot be read naively again.

Following the census's own ordering, the next target after 4x4 (50.70 %
forward / 44.43 % inverse) was 8x8 (22.30 % / 25.66 %). Both fused kernels were
written and both are bit-exact — the full transform differential suite against
the **real exported C**, forward and inverse across bd 8/10/12 and the whole TX
grid, passes **23/23**. Then the band:

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 512x512 cq27 s0 | 3469.3 ms | 3717.1 ms | **+7.07 %** | **0/16** | <0.0001 | +0.03 %, 7/16, p=0.80 |

**Zero of sixteen rounds faster.** Reverted; band kept as
`encoder_fused_txfm8x8_2026-09-09.rejected.band512.tsv`.

## Why — and it is the thing the census cannot see

A fused kernel replaces the generic driver's passes with its own **scalar**
ones. That is a win only where the generic path was not already vectorised:

| | `try_fwd_col_pass` | `try_fwd_row_pass` | what fusion trades away |
|---|---|---|---|
| **4x4** | a HALF-FILLED 4-wide batch (KB-PERF-9's arm) | **declines** — still scalar at `row_n == 4` | almost nothing |
| **8x8** | full 8-wide SIMD (`col_n % 8 == 0`) | full 8-wide SIMD (`row_n % 8 == 0`) | **both vector passes** |

So at 4x4 the fusion removed per-call overhead from an essentially scalar path,
and at 8x8 it removed per-call overhead *and* the SIMD — and the SIMD is worth
far more than the overhead. **Same change, opposite sign, because the baseline
differs.** That is KB-PERF-5's lesson exactly (its half-batch was worth it where
KB-PERF-3's was not, for the same reason), and KB-PERF-9's, arriving a third
time from a new direction.

## What the census IS good for, restated

It correctly said the big kernels are worth nothing (both dims >= 32 is ~0.5 %)
and correctly identified 4x4 as the one size where half the calls live. What it
cannot say is **whether the generic path is already fast for that size**. The
usable rule is the conjunction:

> fuse where the call count is high **AND** the generic path's SIMD passes
> decline or run half-filled.

By that rule 4x4 was the only forward/inverse target available, and the
transform class's remaining headroom is NOT another fusion — it is making the
8-wide passes themselves cheaper, or reducing how many transforms the speed-0
search asks for (1,196,534 forward transforms on a 196x196 frame is 31 per
pixel).

**Do not re-attempt an 8x8 fusion against a SIMD-capable generic path.** A
future one would have to be vectorised itself to beat what is already there.
