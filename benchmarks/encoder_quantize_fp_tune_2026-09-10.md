# KB-PERF-36 — `quantize_fp`: a per-lane eob load and four loop-invariant splats — **−0.18 %**

**2026-09-10.** Byte-identical; **−0.18 %** at 1024x1024 cq27 `--cpu-used 3`,
pooled over **two independent 30-round rotated bands** (−0.185 % at 42/60,
p=0.0027, and −0.170 % at 45/60, p=0.0001, against the two base copies). The
pooled null is not significant (−0.059 %, 33/60, p=0.52).

**Why pooled:** the first band came back at −0.130 % / −0.124 % with only ONE of
the two comparisons clearing p<0.05 (p=0.36 and p=0.043) — consistent in sign
and size but marginal, so a second band was run rather than reporting a
borderline number. The check that matters: **the pooled median (−0.18 %) sits
between the two band medians (−0.13 %, −0.25 %)**, so no favourable band was
selected.

## Why this class was examined at all

`quantize` is the one class the 1 MP table describes as *"never examined, and the
one class that GREW"*. At the shipping preset it is **+124 ms, 7.2 % of the gap,
ratio 3.27** — and the like-for-like split says the two halves are very
different problems:

| | port | C | ratio |
|---|---:|---:|---:|
| `quantize_fp_impl_v3` | 78.1 ms | `av1_quantize_fp_avx2` + `_32x32` 25.1 ms | **3.1** |
| `aom_quantize_b_no_qmatrix` | 26.9 ms | `aom_quantize_b_avx2` + facade 3.6 ms | **7.5** |

## The two defects fixed, both loop-level, neither arithmetic

1. **The eob path built its vector per lane, every chunk.**

       let isc = i32x8::from_array(token,
           core::array::from_fn(|k| iscan[base + k] as i32 + 1));

   Eight bounds-checked `i16` loads per chunk against a runtime slice length.
   It now reads the eight entries as one fixed-size `[i16; 8]` — **one bounds
   check per chunk instead of eight**. This is the same defect KB-PERF-34 and
   KB-PERF-35 each measured at ~0.2 pp, in a third kernel.

2. **Four `splat`s were rebuilt every chunk though only chunk 0 differs.**
   `mk(dc, ac, first)` returns `splat(ac)` whenever `first` is false, so the
   threshold, rounding, quant and dequant vectors are loop-invariant from chunk
   1 on. They are now built once. On a 32x32 block that is 4 x 127 splats
   removed.

Neither changes a value or a lane — strictly less work for the same result,
which is why this cannot be a regression by construction and why the marginal
first band was worth resolving rather than discarding.

## What was NOT done, and why — the 7.5x half

`aom_quantize_b_no_qmatrix` has **no SIMD at all** (no `__arcane_` prefix in the
profile) and the worst ratio in the class. It was left alone deliberately:

its main loop walks **scan order** and writes `qcoeff[rc]` / `dqcoeff[rc]` at
scattered raster positions, so vectorizing it means processing in RAW order
instead. That is provably value-identical for the coefficients — the pre-scan's
dead-zone test is the same test the main loop applies, so positions it skips
would quantize to zero anyway, and they are already zero from the opening
`fill(0)` — **but it changes how `eob` is derived**, and `eob` is precisely the
quantity KB-12 documents as the single order-sensitive output of this family:
rate, distortion and skippability are all order-invariant sums, and a dropped
transpose there read as a genuine RD near-tie across four separate localization
passes.

So the rewrite is tractable and worth ~23 ms, but it needs its own landing with
an explicit eob argument and a bite proof aimed at the eob, not a rushed edit at
the end of a cycle.

## Correctness

* Byte-identical on four cells: 40,237 B (1024x1024 s3), 39,694 (1024 s0),
  10,912 (512 s0), 11,961 (512 s6) — two sha256-**distinct** binaries, checked
  before the band (the rule KB-PERF-34/35's stale-binary incident added).
* `-p zenav1-aom-dsp` 446/446 in both dispatch modes.

## Not covered

One box, one content class, `--cpu-used 3`, x86-64.
