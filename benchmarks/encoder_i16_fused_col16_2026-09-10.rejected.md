# i16 lanes in the 16x16 fused COLUMN pass — BUILT, BYTE-IDENTICAL, MEASURED NULL-TO-SLOWER, REVERTED

The ranked #1 remaining clause-(4) lever, in its cheapest form. Cell:
1024x1024 cq27 `--cpu-used 3` (the shipping preset), x86-64 Linux.
Band: `encoder_i16_fused_col16_2026-09-10.rejected.tsv`.

## What was built

`fwd_16x16_fused`'s column pass holds 16 columns as TWO `i32x8` vectors and so
calls `run_fwd1d` **twice**. With `i16x16` it is ONE vector and ONE call. The
change gated on the generic driver's own bound (`fwd_col_i16_applies`, shift0=2
— the same `max|input| << 2 <= M*` test, so both paths admit exactly the same
blocks), ran `run_fwd1d_i16` once, then widened immediately with
`prims16::widen_lo`/`widen_hi` so the round-shift, lr flip, transpose and row
pass below were byte for byte the code the i32 path ran.

Contained: `#[magetypes(define(i32x8, i16x16), v3, -scalar)]`, one extra
`Option<Fwd1dI16>` parameter, and the existing i32 path kept as the decline.
**The magetypes/token plumbing compiled with three trivial path-qualification
errors and nothing else** — cross-module `incant!` into `lowbd16_fwd`'s i16
runner works, which is worth knowing for the next attempt.

## It was correct, and reached

* **Byte-identical on all four pinned cells** — 40,237 / 39,694 / 10,912 / 11,961.
* **28/28 transform differentials green**, including `txfm2d_differential_fuzz`
  against the real exported C.
* **BITE PROOF, asymmetric.** Perturbing ONLY the new branch (shift0 2 -> 1)
  moves the shipping cell to **41,045 bytes** — so the branch is genuinely
  entered and load-bearing, not dead code behind a green suite — and fails
  exactly **3** tests (`txfm2d_edge_cases`, `txfm2d_differential_fuzz`,
  `txfm2d_simd_equals_scalar_at_every_permutation`) while **25 stay green**.

## And it does not pay

30 rotated rounds, three arms (base, new, and a second copy of base as the
same-binary null), one invocation each per round:

| comparison | paired median | rounds faster | p |
|---|---:|---:|---:|
| **new vs base** | **+0.185 %** | 7/30 | 0.0052 |
| null (baseB vs base) | **+0.150 %** | 10/30 | 0.0987 |

**Lever minus null is ~+0.035 pp — inside the copy systematic this repo already
documents at ~0.27 pp.** Read it as NULL to slightly slower, not as a
significant regression. All 90 invocations emitted 40,237 bytes.

## Why, and what it corrects

The ceiling was always small and the arithmetic now says so out loud: 16x16 is
**5.62 % of forwards**, and this changed **one of its two passes**, so the
reachable share is ~1.4 % of 1-D transform work ~ **0.12 % of the encode** —
BELOW what a 30-round band on this box resolves (~0.2 %). It was never going to
be measurable alone.

Against that ~0.12 % ceiling the change also ADDS work, which is the session's
own predictor of null-or-worse:
1. `fwd_col_i16_applies` scans all 256 input samples per transform to take the
   bound — a cost the i32 path does not pay at all;
2. 32 extra `vpmovsxwd` per transform to widen 16 `i16x16` into `wlo`/`whi`,
   partially offsetting the one saved kernel call.

**So the honest correction to the ~1 % estimate: that figure is for the FULL
step across BOTH families and BOTH passes. The per-family, per-pass slice is
~0.12 % and is unmeasurable, and each slice pays the bound scan separately.**
A lever that only exists in aggregate cannot be landed or validated in
increments — which is the thing this experiment establishes and the reason to
record it rather than re-attempt it.

## What is NOT refuted

The FULL step — both passes in i16, staying narrow through a 16-lane i16
transpose (`_mm256_unpacklo_epi16`, the same raw-AVX2 technique
`fwd_16x16_fused`'s own `tr8` already uses for i32) — is untouched by this
result. It would halve BOTH passes on both families and pay the bound scan once
per transform rather than per pass. **But it must be built across all four
kernels (`{fwd,inv}_16x16_fused`, `{fwd,inv}_rect816_fused`) and banded as ONE
change**, because no single piece of it clears the noise floor.

Whoever takes it: the plumbing above is proven, the widen ordering is proven,
the bite proof recipe is here, and the band harness is
`scripts/eprof_ab.sh`-shaped but needs the `eprof_x86` CLI rather than
`drv-aom`'s.
