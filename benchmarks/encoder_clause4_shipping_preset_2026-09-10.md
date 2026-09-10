# Clause (4) at the shipping preset, re-measured at HEAD: **1.962x**, under 2x for the first time

**2026-09-10, no code changed.** `eprof_x86`, 1024x1024 cq27 **`--cpu-used 3`**
(the preset zenavif ships — `zenavif/src/encoder.rs:506` `speed: 4`,
`encoder_aom.rs:248` `speed_to_cpu_used(s) = s - 1`), 8 rotated rounds of
4 reps, arms alternating position each round.

| | median | spread | paired per-round ratio |
|---|---:|---:|---|
| libaom C | 1549.47 ms | 1.41 % | min 1.9437 |
| port | 3040.11 ms | 0.79 % | median **1.9623** |
| **ratio (medians)** | **1.9620x** | | max 1.9750 |

**Both arms emit 40,237 bytes on every round** — byte-identical, so this is an
RD-parity check as well as a timing, and the ratio is not bought with a
different rate-distortion point.

The paired per-round ratios span 1.944-1.975, i.e. the digit is stable; the
median-of-medians and the median-of-paired-ratios agree to 0.0003.

## Where clause (4) now stands

| | gap | the 1.5x bar needs |
|---|---:|---:|
| 2026-09-09 (cycle start, 2.258x) | 1949.4 ms | **1174.7 ms = 60 % of the gap** |
| **2026-09-10 (HEAD, 1.9620x)** | **1490.6 ms** | **715.9 ms = 48 % of the gap** |

So this cycle took the shipping-preset ratio **2.258x -> 1.962x** and the
requirement from *"more than half of every class gap"* to *"just under half"*.
The bar is still not met and nothing here claims otherwise — but the reachability
argument in `encoder_clause4_reachability_2026-09-10.md` (halving every class gap
lands exactly on 1.505x) is now a smaller ask than when it was written.

## What this supersedes

The clause-(4) row led with **2.266x**, taken before the last twelve landings.
Quote **1.962x at 1024x1024 `--cpu-used 3`** instead. The `--cpu-used 0` and
`--cpu-used 6` rows of the ladder were NOT re-taken here and are older; do not
mix them with this number.

## Not covered

One box, one content class (mirror-tiled `av1-1-b8-01-size-196x196`), one
quantizer, x86-64 Linux. No Windows ratio exists for clause (4) at all — winperf
builds `--no-default-features`, so it has no libaom arm.
