# KB-PERF-18 — the inverse transform intermediate goes on the stack for 86.7 % of calls

**Landed 2026-09-09.** Byte-identical; **−0.73 % at 512x512**, 24/24 rounds
faster, p < 0.0001, against a same-binary null of +0.05 %.

## This is the synthesis of the 8x8 failure, not another fusion

The 8x8 fusion measured **+7.07 %** and was reverted, and the reason was
specific: a fused kernel replaces the driver's passes with SCALAR ones, and at
8x8 the generic path's passes are already full 8-wide SIMD
(`encoder_txfm_size_census_2026-09-09.md`, the CORRECTION section). The lesson
that fell out is that the driver's per-call setup and the driver's passes are
separable — so **remove the setup and keep the SIMD**.

That is exactly this change. Every pass below it, SIMD included, runs unchanged;
only where the intermediate `buf` lives changes.

## The change

`av1_inv_txfm2d_add_into` reached for the caller-owned `InvTxfmScratch` on every
call — `clear()`, `resize(col_n * row_n, 0)`, then a slice through a heap
pointer. **86.7 % of inverse transforms need at most 64 coefficients** (4x4
44.43 % + 8x8 25.66 % + 8x4 6.35 % + 4x8 5.87 % + 16x4 2.33 % + 4x16 2.02 %),
so those now take a `[i32; 64]` stack array sliced to `n`.

The array is zeroed exactly as the `Vec` was, so the two are **semantically
identical rather than merely equivalent in practice** — this does not lean on
the (true, and stated in the function's own comment) fact that the row pass
overwrites every element before the column pass reads it. Anything above 64
coefficients keeps the heap scratch untouched.

## Measured

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 512x512 cq27 s0 | 3464.9 ms | 3440.7 ms | **−0.73 %** | **24/24** | <0.0001 | +0.05 %, 10/24, p=0.54 |

Output byte-identical (10,912 B). Transform differentials against the **real
exported C**, forward and inverse across bd 8/10/12 and the full TX grid:
**23/23**.

## The FORWARD twin was built, measured, and REJECTED

The same change on `av1_fwd_txfm2d_into` is the obvious symmetric move, and 89.4
% of forward transforms are within the same 64-coefficient bound. It does not
pay:

| variant | paired median | rounds faster | p |
|---|---:|---:|---:|
| inverse only | **−0.73 %** | **24/24** | <0.0001 |
| inverse + forward | −0.70 % | 20/24 | 0.0015 |

Statistically indistinguishable, and the combined band is the NOISIER of the
two. The reason is structural rather than mysterious: **the 4x4 half of the
forward's 89.4 % already returns through KB-PERF-16's fused path before ever
reaching the scratch**, so the forward change serves only the 38.7 % remainder —
against the inverse's full 86.7 %, since KB-PERF-17's fused inverse covers 4x4
but the 8x8/rect remainder is 42 % on that side.

Reverted rather than shipped, on the KB-PERF-3 (half-batch) and KB-PERF-5
(PAETH) precedent: **added surface with no measured benefit is a cost.** Band
kept as `.fwd_variant_rejected.tsv`.

## Not covered

* Sizes above 64 coefficients (16x16 and up, ~10 % of inverse calls) keep the
  heap scratch. A larger stack array would cover them but 32x32 is 4 KiB and
  64x64 is 16 KiB of stack per call, which is a different trade and was not
  measured.
* One box, one content class, `--cpu-used 0`, x86-64, 512x512. The 1 MP cell was
  not banded for this one — the effect is per-call and the 512 cell resolves it
  more sharply, but that is a choice worth stating rather than a measurement.
