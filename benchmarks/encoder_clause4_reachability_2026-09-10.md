# Clause (4) IS reachable by kernel work — correcting my own conclusion

**2026-09-10.** Earlier in this session I concluded *"there is no remaining lever
above ~1 %, and 1.5x is not reachable by kernel work"* and recommended either
shipping at 2.09x or reconsidering the bar. **The second half of that is wrong**,
and the arithmetic that shows it is two lines.

## The correction

At the shipping preset the bar needs **−915 ms** of a **1814 ms** classified
gap — **50 % of every class gap**, not 100 % of any one:

| class | port | C | ratio | gap | ratio if its gap HALVES |
|---|---:|---:|---:|---:|---:|
| transform | 707 | 254 | 2.78 | 453 | **1.89x** |
| intra-pred/rd | 589 | 201 | 2.93 | 388 | **1.97x** |
| rd-driver | 481 | 220 | 2.19 | 261 | **1.59x** |
| txb/trellis | 701 | 512 | 1.37 | 189 | 1.18x |
| memory | 155 | 23 | 6.74 | 132 | 3.87x |
| quantize | 178 | 54 | 3.30 | 124 | 2.15x |
| loop-restoration | 149 | 36 | 4.14 | 113 | 2.57x |
| distortion | 190 | 82 | 2.32 | 108 | 1.66x |
| postfilter | 57 | 11 | 5.18 | 46 | 3.09x |

**Halve every class gap and the encode is 2334 ms = 1.505x.** The bar, exactly.

**And that is not a hypothetical rate.** The transform class went **4.26x -> 2.78x
this cycle** — its gap fell **45 %** — through the eight fused whole-transform
kernels (KB-PERF-23..31). One class, one cycle, essentially the required
reduction. The demand is to do that to the rest.

## Why my "no lever above 1 %" was true and still misleading

Both statements are correct and they do not conflict:

* no single remaining lever is worth more than ~1 %;
* the bar needs 50 % off nine class gaps.

What I got wrong was inferring the second is unreachable from the first. **The
transform class was not closed by a big lever either** — it was closed by eight
landings of 0.3–2.0 % each, exactly the size I was dismissing. A programme of
many sub-1 % landings is precisely what moved the one class that has moved.

## What is genuinely ruled out, so the programme does not chase it

**Redundant work — the KB-PERF-1 shape — is structurally excluded here**, and
that is worth stating because it is the natural first suspect (it was the largest
finding in this project's history: the intra CNN recomputed ~10x per superblock).

The port's output is **byte-identical to libaom** on this cell and on 427 gated
cells, and the speed features are ported. Byte identity means the same RD
decisions, which means the same candidates evaluated, which means the same
kernel invocation counts. The profile corroborates: the **drivers** run at
**1.29x** (`search_tx_type_intra_into` 143 ms vs `search_tx_type` 111) and
**1.37x** (txb/trellis) while the **kernels** run at 2.8x–6.7x. Redundant calls
would inflate the drivers too, and they do not.

**So the entire 1690 ms is per-operation cost.** That is good news for
tractability: it is a kernel-efficiency problem with a known end state, not a
structural one.

## The honest shape of the programme

Nine classes, ~50 % off each. The transform precedent says a class takes roughly
a cycle of focused work. Ranked by gap, with what is already known about each:

1. **transform +453** — the i16-in-fused-kernels lever
   (`encoder_clause4_narrowpath_2026-09-10.md`, 63 ms) plus the remaining
   unfused sizes. Partially mapped.
2. **intra-pred/rd +388** — the s3 profile's newly-promoted #2, barely touched
   (one landing, `generate_hog`). `highbd_filter_intra_edge` 4.1x, CfL 17 ms,
   and the CNN is PARITY-gated (KB-41 root #27).
3. **rd-driver +261** — never drilled into at s3. `txfm_rd_in_plane_intra` is
   124 ms against ~44 ms of C equivalents (2.8x) and is the #1 `memmove` caller.
4. **memory +132 at 6.74x** — allocator bookkeeping is closed; what is left is
   `memset`/`memmove` inside transform and txb scratch handling.
5. **quantize +124** — `aom_quantize_b_no_qmatrix` has no SIMD at all (7.5x,
   ~23 ms), scoped in `encoder_quantize_fp_tune_2026-09-10.md` with its `eob`
   hazard named.
6. loop-restoration +113, distortion +108, postfilter +46 — the last two never
   examined at any preset.

**txb/trellis is already at 1.37x and should be left alone** — `optimize_txb_core`
is faster than libaom's trellis, confirmed at both speeds.

## What this changes

The recommendation is no longer "ship at 2.09x or reconsider the bar". It is
**"the bar is reachable; it is roughly eight more cycles of the work this cycle
did to the transform class."** Whether that is worth funding is a business
decision, not a technical one — but it should be made against a reachable
target, which is not what I said an hour ago.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, bd8, x86-64. Class
assignment is by symbol name with a ~129 ms C long tail in `other`, so read the
class gaps as +-5 %. The "halve every class" figure is an existence result about
the arithmetic, not a claim that every class halves equally easily — memory at
6.74x and postfilter at 5.18x are small but very unequal in difficulty.
