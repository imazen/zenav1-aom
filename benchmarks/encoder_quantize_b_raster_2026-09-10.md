# `aom_quantize_b_no_qmatrix` walks RASTER order now, not scan order: −0.134 % pooled, and the 11.89x kernel is finally in a shape that can be vectorised

**2026-09-10.** Byte-identical; `just gate-landing` 1507/1507 twice. This is the
row `encoder_lever_map_s3_2026-09-10.md` named as *"the biggest single un-taken
kernel left"* — +28.7 ms at **11.89x**, the worst ratio of any addressable row
and the only one with **no SIMD at all**.

## §14 first: the ranked figure holds

Re-profiled at HEAD (1024x1024 cq27 s3, `perf record -F 499`):
`aom_dsp::quant::aom_quantize_b_no_qmatrix` is **0.86 % of the profile ≈ 25.7 ms**,
against the lever map's 26.9 ms. No re-scoping needed.

`perf annotate` would not resolve on this binary, so the shape came from
`objdump`: **217 instructions, ZERO ymm, ZERO xmm, 14 `cmp`, 8 panic-call sites
and 2 `call memset@GLIBC`.** A fully scalar, heavily bounds-checked loop with
two real memset calls — exactly what the 11.89x says.

## The measurement that decided the DESIGN, taken before any code was written

The plan on file was "process in RAW order", and the open question was trip
count: C's pre-scan trims trailing dead-zone coefficients, so scan order runs
`non_zero_count` iterations where raster order runs `n`. A throwaway counter
(added, run, reverted) on the shipping cell:

| | value |
|---|---:|
| calls per 1 MP encode | **538,094** |
| mean `n` (coefficients per call) | **60.8** |
| mean `non_zero_count` | **32.7** |
| **`nzc / n`** | **0.538** |
| ns per call (25.7 ms / 538,094) | 47.8 |

So raster order pays **1.86x the iterations**. It has to buy that back, and the
counter is what said it plausibly could: at 32.7 processed coefficients per call
the two `memset`s cover 60.8 x 4 bytes EACH — i.e. the fixed per-call zeroing is
comparable to the whole variable part.

## What raster order removes, and the three facts that make it exact

* **Both `memset`s.** Every position is now written unconditionally, with the
  same zero on the dead-zone side. C's pre-scan trims exactly the positions
  satisfying `|coeff| < zbin[ac]`, and C's main loop writes only where
  `|coeff| >= zbin[ac]` — **exact complements**, so a trimmed position would have
  written nothing anyway and `memset` had already left it zero. The set of
  positions that WRITE is unchanged; only the set that is *visited* grew.
* **Every bounds check, the `scan[i]` load, the gather and the scatter.** The
  three slices are iterated, not indexed by a data-dependent `rc`.
* **The per-coefficient class select.** C computes `ac = (scan[i] != 0)` — it is
  asking whether the RASTER position is the DC. In raster order that is `i != 0`,
  so the DC peels off and all five per-class constants (`zbin` gate, the
  `ROUND_POWER_OF_TWO`'d round, `quant`, `quant_shift`, the `iwt`-folded
  `dequant`) become scalars for the whole walk. C re-derives them per
  coefficient.
* **The EOB.** `eob = 1 + max(iscan[rc])` over written-nonzero positions equals
  C's scan-order maximum because `iscan` inverts `scan`. **This is the ONE
  order-sensitive output of the family** (KB-12 is the standing proof that
  losing it reads as an RD near-tie for four localization passes), so it carries
  its own bite proof.

This is not an invented design: `crates/aom-dsp/src/quant/simd.rs` already does
exactly this for `quantize_fp` (its module doc carries the same `iscan`
argument), and **libaom's own `aom_quantize_b_avx2` takes an `iscan` argument
that the `_c` body ignores — for precisely this reason.** The port now matches
that structure.

## Measured

Two sha256-distinct binaries from one tree, arms ROTATED, same-binary null.
Byte-length identity first on four cells (40,237 / 39,694 / 10,912 / 11,961 —
identical on both arms).

| band | paired | rounds | p | null |
|---|---:|---:|---:|---:|
| A, 1024x1024 cq27 s3, n=30 | −0.169 % | 21/30 | 0.0428 | −0.078 % (p=0.58) |
| B, same, independent, n=30 | −0.106 % | 19/30 | 0.2005 | +0.037 % (p=0.86) |
| **POOLED, n=60** | **−0.134 %** | **40/60** | **0.0135** | −0.013 % (p=0.90) |

**Pooled because band A came back marginal and band B alone is not significant**
— KB-PERF-36's protocol. The check that makes pooling meta-analysis rather than
selection: **the pooled median (−0.134 %) sits BETWEEN the two band medians**
(−0.169 %, −0.106 %), so no favourable band was chosen.

−0.134 % of 2986 ms is **~4.0 ms of the kernel's 25.7** — about 16 %.

## Read this honestly: it is a TRADE, and a small one

Most of this cycle's wins strictly remove work. **This one does not** — it
removes two memsets, all bounds checks, the scan indirection and the class
select, and it ADDS arithmetic on 0.46n extra coefficients. The session's own
rule says trades are where perf changes go wrong, and the measurement bears the
caution out: significant, clean null, and only a sixth of the kernel.

**The reason to keep it is structural as much as numeric.** The kernel is now
raster-order, sequential, unconditionally-storing, with loop-invariant
constants — the only shape a vector kernel can take, and the shape libaom's
`_avx2` already has. The scan-order form could not be vectorised at all.

## What blocks the vector tier, stated so the next session does not re-derive it

The i32 chain is safe up to the last step. With `wt = 32` and `u = clamped`
(`|u| <= 32767`):

* `(u*32*quant) >> 16` **is exactly** `(u*quant) >> 11`, and `u*quant <= 2^30`
  — i32-safe.
* `t2 = (u*quant >> 11) + u*32 <= ~2^20.6` — i32-safe.
* `t2 * quant_shift` reaches **~2^35** — **NOT i32-safe**, and this is the wall.

libaom's AVX2 kernel gets past it the same way `av1_quantize_fp_avx2` does: 16-bit
arithmetic. `CLAUDE.md` already rules that structural for `quantize_fp` (the i16
`dqcoeff` wrap is REACHABLE — `xtask/audit_quantize_dqcoeff_range.py` puts bd8 at
33,137 against `i16::MAX` 32,767, a margin of one unit), and the same argument
applies here. A vector tier therefore needs a DERIVED bound on
`t2 * quant_shift` plus a runtime gate — and note the differential deliberately
draws `quant_shift` INDEPENDENTLY of `dequant` (the KB-ARM-FLOAT root-#3
unrealizable-triple shape), so no bound may be assumed from how libaom builds
its tables.

## Gate and bite proof

`quantize_b_diff` now inverts its (random-permutation) scan to build `iscan`:
**240,000 cases** — 20,000 x {n = 16, 64, 256, 1024} x log_scale {0,1,2} —
against the real exported `ref_quantize_b`, checking qcoeff, dqcoeff AND eob.
Random permutations stress the inverse far harder than the real
`av1_scan_orders` rows do.

**Bite proof, aimed at the EOB as KB-12 requires, and maximally asymmetric:**
deriving the eob from the RASTER index instead of `iscan` fails **exactly one
test** — `quantize_b_diff::quantize_b_differential_fuzz`, and it fails on the
**eob** (`left: 16, right: 15`) with qcoeff and dqcoeff still matching — while
the other **37** quantize/txb tests stay green, including
`quantize_b_adaptive_diff`, `quantize_qm_diff` and `highbd_quant_diff`, whose
kernels were deliberately left in scan order.

`just gate-landing`: `test-next` 1507/1507, `test-next-scalar` 1507/1507,
`census-gate` 4/4, `test-whereat` 4/4.

## Not converted, deliberately

`aom_highbd_quantize_b_no_qmatrix`, `aom_quantize_b_qm` /
`aom_highbd_quantize_b_qm` and `aom_quantize_b_adaptive_helper` all carry the
same scan-order shape and are untouched — they are bd10/12 or QM paths that do
not appear in the shipping profile, and leaving them is what keeps the bite
proof above asymmetric.
