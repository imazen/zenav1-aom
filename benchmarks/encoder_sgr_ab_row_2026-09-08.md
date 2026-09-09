# SGR A/B pass: vectorized without a gather — 2026-09-08

KB-PERF-8. After KB-PERF-7 the largest loop-restoration symbol was
`sgr::calculate_intermediate` — `av1_selfguided_restoration_c`'s A/B pass — at
2.71 % = 288 ms on a 1 MP frame. KB-PERF-6 had deferred it as "a gather the
current vector vocabulary does not handle well, and not worth forcing for 8 ms".

## Both halves of that deferral were wrong

* The 8 ms omitted `calculate_intermediate::{closure#0}`, which KB-PERF-6's own
  correction blockquote already identifies as an inlining sink.
* The lookup is **not the cost**. `X_BY_XPLUS1[z.min(255)]` is ONE operation of
  about twenty in the loop body (two ROUND_POWER_OF_TWOs, three multiplies, a
  saturating subtract, another rounding shift, the `b` chain, two stores). It
  was never slow — it was blocking vectorization of the other nineteen.

So the lookup stays eight scalar loads through a stack round trip, and
everything around it goes eight-wide. No new primitive, no new proof obligation.

## The gather question, settled

* The instruction would be `_mm256_i32gather_epi32` (VPGATHERDD).
* **magetypes has no gather at all** — checked in the pinned 0.9.28 and in
  0.9.29 — and no shuffle/permute either, only `blend`, so a `pshufb`
  in-register LUT is also unavailable (the low 16 entries would have fit: all
  <= 240, so they are byte-sized).
* It **would** be soundly wrappable: `&[i32; 256]` plus an index masked to
  `N - 1` is in-bounds by construction, with the `unsafe` inside magetypes
  exactly as archmage already does for `safe_unaligned_simd`. Soundness was
  never the obstacle.
* It would very likely lose anyway: AVX2-only (aarch64 has nothing before SVE2,
  wasm128 nothing), and VPGATHERDD is ~12-20 cycles of throughput for 8 lanes
  against 8 L1 loads at ~0.5 each, on a 1 KB permanently-L1-resident table.

## The arithmetic form: proven, held in reserve

The domain is `z in 0..=255`, so **exhaustive enumeration is a complete proof**,
not sampling. Measured over all 256 inputs:

* `X_BY_XPLUS1[z] == round(256z/(z+1))` at **254 of 256** entries, with two
  deliberate endpoints — `z=0 -> 1` (libaom's own comment: "corresponding to a
  value of 1/256") and `z=255 -> 256`.
* An f32 form — `256.0/(z+1)`, `+0.5`, floor, `256 - r`, two endpoint blends —
  is **exact on all 256 inputs, 0 mismatches**.
* Margin: the nearest half-integer to any `256/(z+1)` is **0.002924** away,
  against an f32 absolute error of ~1.5e-5. ~200x headroom, not knife-edge.

This is the follow-up if the store-forward round trip ever dominates. It must
use true `Div`; **never `rcp_approx`/`recip`** — approximate reciprocal is
specified only to a relative-error bound and its exact bits differ between
vendors, which would make the encoder's output depend on whose CPU ran it.

## Bit-exactness: no range argument at all

Stronger than KB-PERF-7's. Each column `j` is independent, so nothing is
reassociated. The lane arithmetic is C's `uint32_t` arithmetic exactly: `+`,
`-`, `*` are the low 32 bits (signedness cannot change a low-32 result); every
right shift is `shr_logical`; and `saturating_sub` — the one signedness-
sensitive step — does not lean on the box-sum bound, because the compare is made
unsigned by flipping both operands' sign bits (`x ^ i32::MIN`). The single
signed compare kept is `z.min(255)`, unconditionally safe since `z` is a logical
shift right by 20 and lies in `0..=4095`.

Recorded asymmetry (clause 2(b)): the scalar tier spells `a * n` and `v + half`
as checked Rust ops, so a `debug-assertions` build would panic where these lanes
wrap. Unreachable — `a <= 25*4095^2 >> 8` ~ 1.64e6 so `a * n <= ~4.1e7`, three
orders under `u32::MAX` — and a release build wraps on both sides.

## Result

Two binaries from one tree, arms interleaved and ROTATED, same-binary null arm
in every band, byte-identical output on every arm.

| cell | base | vec | libaom-c | ratio | paired median | rounds | p | null |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 192x192 cq27 s0 | 441.19 ms | 436.13 ms | 176.20 ms | 2.5039x -> **2.4752x** | **-1.20 %** | 20/20 | 1.9e-6 | +0.10 % (p=0.50) |
| 1024x1024 cq27 s0 | 10615.10 ms | 10424.88 ms | 4118.45 ms | 2.5775x -> **2.5313x** | **-2.00 %** | 8/8 | 0.0078 | -0.24 % (p=0.29) |

Raw rounds: `.band192.tsv`, `.band1024.tsv`. The 1024x1024 base arm reads 10615
against KB-PERF-7's just-pushed 10600 (0.14 % apart), so this is measured
against a live control that reproduces the landed number.

## Attribution

| symbol (1024x1024) | base | vec |
|---|---:|---:|
| `sgr::calculate_intermediate` (A/B loop inlined in) | 2.71 % = **287.7 ms** | 0.05 % = 5.2 ms |
| `sgr::ab_row_impl_v3` | — | 0.63 % = **65.7 ms** |
| **A/B pass** | **287.7 ms** | **70.9 ms** (4.1x) |

-216.8 ms attributed against a -190.2 ms wall delta; the box-sum closure reads
+8.8 ms in the same single-run profile, within one-sample noise. **The 8-round
wall band is the ground truth — the symbol split attributes the work that MOVED
and is not a per-symbol measurement of everything that did not.** Reading it the
other way is the KB-PERF-6 roll-up error. The A/B pass leaves the
loop-restoration symbol list entirely.

## Gates

`sgr.rs` is on the DECODE restoration path, so this can move Gate 1:
`-p zenav1-aom-decode` 19/19 ok; `pick_diff` 6/6 (incl.
`selfguided_flt_producer_matches_c`), `pick_search` 3/3, `kernels_diff` 2/2 —
all in both dispatch modes.

## Session total for clause (4)

2.5412x -> **2.4752x** at 192x192 and 2.6227x -> **2.5313x** at 1024x1024 across
KB-PERF-7 + KB-PERF-8, all byte-identical.

## Still scalar here

`calculate_intermediate`'s box-sum closure (vectorized in KB-PERF-6, ~155 ms),
`selfguided_restoration` itself, `compute_stats_highbd`. Loop restoration was
+1406 ms of the 1024x1024 gap before these two landings.
