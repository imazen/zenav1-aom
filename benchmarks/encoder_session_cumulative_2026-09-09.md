# The cycle's three kernels, measured TOGETHER: −1.054 % at the shipping preset

**2026-09-09.** Three perf landings went in this cycle, each with its own band.
Summing three separately-taken deltas is exactly what `DIFFERENTIAL_PLAYBOOK.md`
§6 forbids, so this is the composed measurement.

## The band

`drv_base` = the cycle's starting HEAD (`dc08470`), `drv_cum` = HEAD with
KB-PERF-20 (filter-intra taps), KB-PERF-21 (8x8 Hadamard AVX2) and KB-PERF-22
(wiener madd) all in. Two sha256-distinct binaries, 33 rotated rounds,
1024x1024 cq27 `--cpu-used 3`, a same-binary null:

| arm | median | paired median | rounds faster | p |
|---|---:|---:|---:|---:|
| base (`dc08470`) | 3531.26 ms | — | — | — |
| baseB (same binary) | 3528.89 | −0.048 % | 19/33 | 0.4869 |
| **all three landings** | **3494.37** | **−1.054 %** | **33 of 33** | **<0.0001** |

**33 of 33 rounds**, against a null that is flat at −0.048 %.

## They compose at ~85 %, and that is the number to quote

| | |
|---|---:|
| KB-PERF-20 filter-intra | −0.372 % |
| KB-PERF-21 Hadamard | −0.470 % |
| KB-PERF-22 wiener | −0.400 % |
| **arithmetic sum** | **−1.242 %** |
| **measured together** | **−1.054 %** |

Sub-additive by ~15 %, which is what should happen: each landing shrinks the
denominator the next one is measured against. **The composed −1.054 % is the
honest figure; the sum is not.**

## The ratio, measured in the same window

The C arm was re-measured immediately after the band rather than carried from an
earlier one (the clause-(4) row already warns that the C reference drifts ~1 %
between bands, so a cross-window ratio is not resolvable to three digits):

| | port | libaom C | ratio |
|---|---:|---:|---:|
| base | 3530.56 ms | 1541.3 ms | **2.291x** |
| **after all three** | **3493.19** | 1541.3 | **2.266x** |

C measured twice, 1541.83 and 1540.82 ms — agreeing to 0.07 %.

Against the cycle's recorded headline of 2.258x (measured in an earlier window
where C read 1549.38), the same −1.054 % gives **~2.234x**. Both readings are
given because neither is more true than the other: **the paired delta is the
solid quantity, the ratio digits carry the C arm's drift.**

## Byte identity across the whole cycle

Every arm of every band — base, null, each landing, and the C oracle — emitted
**40,237 bytes** at this cell. The three landings are byte-identical to each
other, to the cycle's start, and to libaom.

## What it does not say

One cell, one content class, one box, bd8 4:2:0, cq27, `--cpu-used 3`, x86-64.
The three kernels were each measured at this preset because filter-intra is 0 %
of leaves at `--cpu-used 6` and would read nothing there; their behaviour at
other speeds is not measured. And −1.054 % of a 2.29x ratio is **~37 ms of a
~1950 ms gap** — real, gated and byte-identical, but the bar needs ~1175 ms.
