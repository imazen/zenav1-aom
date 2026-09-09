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
