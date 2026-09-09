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


---

# FOLLOW-UP, SAME DAY: the prediction above is REFUTED, and there are TWO overhead classes

The section above predicted that this cycle's landings — all of which attack
fixed per-call overhead — "should pay MORE at speed 6+, where the same absolute
cost is a bigger share", and flagged that none had been measured there. They
have now been, and **they pay 4.6x LESS.**

## Measured: the cycle's four landings, the SAME two binaries, at two speeds

`base` is `807f928` (immediately before KB-PERF-16); `new` is HEAD. Rotated
arms, same-binary null, output byte-identical at BOTH speeds (10,912 B at s0,
11,961 B at s6 — so the landings are byte-inert at speed 6 as well, which
nothing had checked).

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 512x512 cq27 **s0** | 3596.3 ms | 3405.9 ms | **−5.39 %** | **20/20** | <0.0001 | −0.11 % |
| 512x512 cq27 **s6** | 125.9 ms | 124.4 ms | **−1.18 %** | **24/24** | <0.0001 | +0.03 % |

Ratio at 512x512 s0: **2.309x → 2.187x**.

## Why the prediction was wrong, and what it means

**The overhead this cycle removed is per-TRANSFORM-CALL**, and a speed-6 encode
issues far fewer transform calls per pixel because the tx search is pruned hard.
So the absolute saving shrinks with the search, and shrinks FASTER than the
total encode time does — the opposite of the reasoning above, which treated
"fixed overhead" as one undifferentiated thing.

It is not: there are at least **two** overhead classes with opposite speed
behaviour.

| class | scales with | biggest at | this cycle |
|---|---|---|---|
| per-transform-call setup (config derivation, scratch churn, buffer round trip) | the SEARCH | **speed 0** | removed −5.39 % of it |
| whatever drives the ratio from 2.19x to 5.29x as the preset rises | frame / block count, not search depth | **speed 9** | **untouched, unidentified** |

The ratio still worsens monotonically with speed, so the second class is real
and is what a fast still-image configuration actually pays. **This cycle did not
touch it**, and the speed-0 profile the entire ranking is built on cannot see it
— at speed 0 the search-scaled work dominates and hides it.

## The actionable conclusion

**Profile at the speed that will ship, not at speed 0.** Every ranked table in
this repo (`encoder_x86_reprofile_1024_2026-09-09.after.md` and its
predecessors) is a speed-0 profile, so it ranks the first class and is blind to
the second. A speed-6 or speed-8 profile has never been taken at HEAD, and it is
the cheapest remaining step on clause (4) — it would name the second class
rather than leaving it as "whatever drives the ratio".

That also re-reads this cycle honestly: **−5.39 % at speed 0 is a real result on
the class it targeted**, and it is a smaller result (−1.18 %) on the
configuration that matters most for clause (1). Both numbers are true; only the
first was measured before today.
