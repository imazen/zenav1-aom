# Clause (4) is measured at the port's BEST speed — the ratio gets WORSE as `--cpu-used` rises

**2026-09-09. No code changed.** Every clause-(4) band in this cycle, and the
1 MP profile the whole ranking is built on, is `--cpu-used 0`. Nothing had
measured the ratio ACROSS speeds at HEAD, and the assumption worth checking was
the optimistic one — that a still-image backend picking a mid speed would find a
better ratio than the speed-0 headline. **It finds a worse one.**

## Measured (512x512 cq27, x86-64, HEAD after KB-PERF-16..19)

| `--cpu-used` | reps | port | libaom C | **ratio** |
|---:|---:|---:|---:|---:|
| 0 | 2 | 3410.02 ms | 1557.75 ms | **2.189x** |
| 3 | 5 | 978.17 | 440.06 | **2.223x** |
| 6 | 25 | 124.61 | 46.99 | **2.652x** |
| 8 | 60 | 46.02 | 14.70 | **3.131x** |
| 9 | 120 | 21.42 | 4.05 | **5.289x** |

Reps are scaled so each row totals a few seconds; a first single-shot pass gave
the same monotone shape (2.170 / 2.236 / 2.451 / 2.879 / 4.355), so the trend is
not an artefact of one timing.

## What it means

**Speed 0 is the port's best case, not a worst case.** Every optimisation this
cycle was measured there, so the landed −2.3 % at 1 MP is the improvement in the
most favourable configuration the port has. At `--cpu-used 6` the port is
**2.65x**, and at 9 it is **5.29x**.

**Mechanism, and it is consistent with everything else measured today.** As the
speed preset rises libaom sheds search work fast; the port sheds the same search
work but keeps its fixed per-call and per-block overhead, so that overhead
becomes a larger FRACTION of a shrinking total. Every landing in this cycle
attacked exactly that overhead — the fused 4x4 transforms (config derivation,
scratch churn, buffer round trip), the inverse stack buffer, the quantize
`clear()` — which is why they paid at speed 0 and should pay MORE at speed 6+,
where the same absolute cost is a bigger share.

**None of them has been measured at speed 6+**, and that is the immediate gap.

## Two consequences for clause (4)

1. **The clause-(4) headline should carry its speed, the way it already carries
   its cell.** `~2.34x at 1024x1024` is true and is `--cpu-used 0`. A reader
   entitled to assume it describes a shipping still-image configuration would be
   wrong by up to 2.4x.
2. **Which speed zenavif selects now matters to the goal, not just to the
   benchmark.** Clause (1) reduces to clause (4); if the backend ships at a mid
   or fast preset, the bar is further away than the speed-0 number implies. That
   is a question for the integration, and it is worth answering before more
   speed-0 optimisation is done.

## What this does NOT say

* One cell (512x512 cq27), one box, one content class. The SIZE axis is
  separately known to matter and in the other direction — `encode_perf_vs_libaom_
  2026-09-08.md` records 3.24x-4.03x on 128x128/192x192 at cpu {0, 6}, i.e.
  small frames are worse, and this 512x512 speed-0 row is 2.19x. Ratio improves
  with frame size and worsens with speed; both are what a fixed-overhead
  weakness looks like.
* It is a ratio of two single measurements per row, not a rotated band with a
  null. It is strong enough to establish the monotone trend and the ~2.4x spread
  across the speed axis; it is not a landing-grade delta.
