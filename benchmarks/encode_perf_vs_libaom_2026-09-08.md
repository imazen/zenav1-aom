# Standalone encode time vs libaom — first in-repo measurement

**Measured 2026-09-08** on `main` at `b1baf31`, x86-64 Linux, 24-core (AMD), glibc,
`--profile test-fast`, box otherwise idle. Gate:
`crates/aom-bench/tests/encode_perf_vs_libaom.rs` (`just gate-encode-perf`).

## Why this file exists

The standing goal (CLAUDE.md, 2026-09-08) asks for "encode time within libaom" and for
"matching the RD of C". Neither had an in-repo measurement. The only figures were two
retained fleet witnesses quoted in GitHub #16 — **2.49x** (8402 photo, QP27, cpu0) and
**2.65x** (9334 photo, QP27, cpu0) — taken on a different machine with a different harness,
and explicitly flagged there as belonging to different hardware cohorts.

## RD is measured as BYTE IDENTITY, which is stronger than any RD metric

If the port emits the same bytes as `aomenc` for the same source and settings, it made every
partition, mode, transform and coefficient decision C made: the rate is identical and the
reconstruction is identical, so the rate-distortion point is the SAME point, not a nearby
one. No SSIM or BD-rate comparison can say more. Every timed row below is asserted
byte-identical, except the four excluded by an existing pin (see below), so the ratios
compare two encoders producing identical output rather than different work.

## Method

Arms are INTERLEAVED within each repetition (port, libaom, port-again) rather than run in
separate phases, so machine drift hits both equally; the reported figure is the per-cell
MINIMUM over 3 repetitions, the right reduction for wall time since contention can only add;
and the second port timing gives a same-arm NULL so the reader can see the floor each ratio
is resolved against. The test asserts the null is small beside the gap on every asserted
row, so an unresolvable cell fails rather than reporting noise.

Content is real photographic material cropped from `av1-1-b8-01-size-196x196`. Settings are
real aomenc's ALLINTRA defaults: CDEF off (`av1_cx_iface.c:3067`), restoration on.

## Result

| cell | pixels | port ms | libaom ms | ratio | null ms | bytes |
|---|---:|---:|---:|---:|---:|---:|
| photo_128x128_cq27_s0 | 16,384 | 295.75 | 90.36 | **3.27x** | 0.16 | 648 |
| photo_128x128_cq45_s0 | 16,384 | 222.78 | 57.51 | **3.87x** | 0.23 | 356 |
| photo_128x128_cq27_s6 | 16,384 | 10.69 | 3.30 | **3.24x** | 0.05 | 798 |
| photo_128x128_cq45_s6 | 16,384 | 9.81 | 2.79 | **3.51x** | 0.02 | 412 |
| photo_128x128_cq27_s9 | 16,384 | 1.81 | 0.50 | 3.63x | 0.01 | 1,015 |
| photo_128x128_cq45_s9 | 16,384 | 1.36 | 0.41 | 3.32x | 0.01 | 432 |
| photo_192x192_cq27_s0 | 36,864 | 644.82 | 176.50 | **3.65x** | 0.57 | 1,177 |
| photo_192x192_cq45_s0 | 36,864 | 442.00 | 109.58 | **4.03x** | 0.10 | 641 |
| photo_192x192_cq27_s6 | 36,864 | 20.80 | 6.16 | **3.38x** | 0.04 | 1,473 |
| photo_192x192_cq45_s6 | 36,864 | 18.68 | 5.11 | **3.66x** | 0.16 | 758 |
| photo_192x192_cq27_s9 | 36,864 | 3.72 | 0.78 | 4.78x | 0.01 | 1,894 |
| photo_192x192_cq45_s9 | 36,864 | 2.80 | 0.62 | 4.55x | 0.03 | 815 |

Bold rows are byte-identical to libaom and therefore RD-identical; the six `_s9` /
non-bold rows are covered by the note below.

**Worst ratio over byte-identical cells: 4.03x. Range 3.24x .. 4.03x.**

## Verdict against the goal

**Clause (4) — "encode time within libaom" — is NOT met, and the gap is larger than the
figures previously quoted.** Gate 3's standing bar is <= 1.5x C; this measures **3.24x ..
4.03x** on cells whose output is byte-identical, i.e. **~2.2x to ~2.7x over the bar**. The
fleet witnesses' 2.49x / 2.65x are the optimistic end of this range, not a typical value.

Nothing here is a regression: this is the first time the standalone entry has been timed
against libaom in-repo, so it establishes the baseline rather than moving one.

## The four excluded rows

`--cpu-used` >= 7 above roughly 3x3 superblocks is an ALREADY-PINNED divergence, not a
finding of this measurement. `self_contained_key_frame.rs`'s pin table records *"at speed 9,
192x192 is not [byte-exact] either"* for the unlocalized VAR_BASED_PARTITION / nonrd arm
(`PIN_256x256_speed7`). Those rows are still timed and reported — speed 9 is the fastest
preset and the most interesting throughput number — but their ratios compare different work,
so they are excluded from the RD assertion. The gate ALSO asserts they still diverge, so the
exclusion cannot go stale: if the pin closes they must move into the asserted set.

## What this does not measure

* Only 128x128 and 192x192, one content source, bd8 4:2:0, single tile, SB64. The fleet
  witnesses are 512x512; larger frames may have a different ratio (the encoder's per-frame
  fixed costs amortise differently).
* Wall time only — no allocation, cache or instruction-count attribution. The last committed
  encoder profile is `benchmarks/encoder_hotspot_reprofile_2026-08-02.md` and its ranked
  levers are arithmetic on a build that has since had several perf landings; per
  `DIFFERENTIAL_PLAYBOOK` §14 it must be RE-profiled before its ranking is trusted again.
* One machine. The ratio is a property of this CPU and this build as much as of the port.
