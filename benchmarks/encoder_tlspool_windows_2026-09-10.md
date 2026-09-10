# The TLS scratch pool: **+0.23 % on Linux, −0.60 % on Windows ARM.** The rejection was platform-local

**2026-09-10.** GitHub Actions run
[34489034382](https://github.com/imazen/zenav1-aom/actions/runs/34489034382),
branch `perf/tls-pool-windows-probe` against `main` (`ae9960e`), `winperf.yml`
`arms: prepost`, 24 rounds x 2 contents x 2 runners.

**The change is byte-identical on every platform.** Only its cost differs.

| platform | effect | rounds faster | p | vs noise floor |
|---|---:|---:|---:|---:|
| **Linux / glibc** (the landing band) | **+0.234 %** | 6/24 | 0.023 | — |
| **`windows-11-arm`, photo** | **−0.600 %** | **24/24** | **<0.0001** | 1.60x |
| **`windows-11-arm`, detail** | **−0.365 %** | 23/24 | **<0.0001** | 0.97x |
| `windows-latest` x86-64, photo | −0.505 % | 16/24 | 0.15 | 0.85x |
| `windows-latest` x86-64, detail | −0.668 % | 16/24 | 0.15 | 0.40x |

**A ~0.8 percentage-point swing in the same code, and the sign reverses.**

## What it settles

KB-PERF-45 rejected this change on a Linux band: the pool removed **412,208
allocations per 1 MP encode** and still measured slower, because
`POOL.with(..)` is a TLS lookup plus a `RefCell` borrow check twice per plane
call, and on glibc the `malloc`/`free` pairs it removed are close to a
free-list pop.

**That reasoning was right about glibc and wrong as a verdict.** On Microsoft's
heap the removed allocations are worth more than the TLS access costs, and the
change becomes a real win — 24/24 and 23/24 rounds at p<0.0001 on
`windows-11-arm`.

This is the first direct confirmation in this repo of the platform effect
KB-PERF-2 inferred (21 % of the win on Darwin, 86-99 % on Windows) — **measured
on the same source change, both ways, rather than across different levers.**

`windows-latest` x86-64 agrees in SIGN on both contents but cannot resolve it:
its noise floor was 0.59-1.65 % this run against an effect near 0.5 %. That
runner has failed to resolve sub-1 % effects throughout the project; it resolved
the whole-cycle −7 % at 55x its floor, and cannot resolve this.

## The decision this hands over — and it is a decision, not a measurement

The change is **byte-identical everywhere**, so nothing about parity, RD or the
gates is at stake. What is at stake is a per-platform performance trade:

* **adopt unconditionally** — −0.6 % on `windows-11-arm`, +0.23 % on Linux;
* **`cfg`-gate it** — take the win where it is a win. Byte-inert either way,
  but it makes the port's *performance* platform-dependent, which this project
  has been careful to avoid making true of its *output* (KB-ARM-FLOAT root #1's
  reasoning);
* **leave it rejected** — simplest, and forfeits ~0.6 % on a shipping platform.

**It is NOT merged.** The branch `perf/tls-pool-windows-probe` carries the
change and its commit says it is not for merge. Whoever decides should also note
that the Linux regression is itself marginal (+0.234 %, one of the two base-copy
comparisons at p=0.54).

## The general lesson

**A performance rejection measured on one allocator is a rejection ON THAT
ALLOCATOR.** Of the four allocation experiments this cycle, the two that were
*kept* removed allocations with no added mechanism (SmallVec inline storage; a
stack array) and are platform-robust. The two that were *rejected* both added a
mechanism, and at least one of them is a win where the allocator is slower.

**Re-run allocation rejections on `winperf.yml` before treating them as closed.**
The remaining one — the exact-`with_capacity` variant (+0.77 % on Linux) —
removed **zero** allocations, so it has nothing to trade and does not need a
Windows run.

## Not covered

Two Windows runners, two synthetic contents (~293 ms per encode), 24 rounds.
macOS is unmeasured — `winperf.yml` has no Darwin runner. This is a DELTA, not a
ratio: winperf builds `--no-default-features`, so there is no libaom arm.
