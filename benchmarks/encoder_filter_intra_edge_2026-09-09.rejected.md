# REJECTED: vectorising the intra EDGE filter — the work moved and got faster, but only by ~3.7 ms

**2026-09-09. Built, gated, measured, REVERTED.** The named follow-up of
KB-PERF-20 and the second of the two levers
`encoder_simd_lane_width_audit_2026-09-09.md` produced: `highbd_filter_intra_edge`
**20.3 ms against `av1_filter_intra_edge_sse4_1`'s 5.6 = 3.63x, +14.7 ms**, and
the port's version scalar.

## What was built

An 8-sample-per-iteration `i32x8` body for the 5-tap sliding filter.

**The in-place hazard is only two samples deep**, which is what made it look
cheap: output `i` reads ORIGINAL `p[i-2 ..= i+2]` while writing `p[i]`, and the
loop writes left to right, so a batch covering outputs `i .. i+8` needs the
twelve originals `p[i-2 ..= i+9]` of which only two have been overwritten.
Carrying those two scalars and reading ten fresh reproduces the scalar window
exactly — no scratch copy, no zero-init, keeping KB-PERF-12's finding.

Bit-exact by construction: lane `k` evaluates the same ascending-`j`
`sum w[j]*taps[j]` and the same `(s + 8) >> 4` as the scalar loop.

## Correct and reached — both proven

* Encoder output **byte-identical** on four cells (40,237 / 39,694 / 10,912 /
  11,961 B).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**.
* **Bite proof on the riskiest line, the cross-batch carry** (`c0 = win[8]` ->
  `win[7]`): fails `edge_diff::highbd_filter_intra_edge_byte_identical`,
  `build_dir_diff::build_directional_matches_c`,
  `predict_intra_diff::predict_intra_matches_c` and `intra_lowbd_diff` — while
  **391 stay green**. So the multi-batch handover is genuinely exercised.

## The band — null

Measured against the just-landed KB-PERF-20 HEAD, rotated arms, same-binary null:

| arm | median | paired median | rounds faster | p |
|---|---:|---:|---:|---:|
| base (KB-PERF-20 HEAD) | 3518.47 ms | — | — | — |
| baseB (null) | 3521.53 | +0.120 % | 11/24 | 0.8388 |
| edge SIMD | 3521.11 | **+0.058 %** | 9/24 | 0.3075 |

The same-binary null is LARGER than the "effect".

## Not a harness miss — checked, and the work DID move

| symbol | before | after |
|---|---:|---:|
| `edge::highbd_filter_intra_edge` (scalar) | 20.3 ms | **5.3 ms** |
| `filter_simd::filter_intra_edge_impl_v3` | — | **11.3 ms** |
| total | 20.3 | **16.6** |

So the kernel is real and it IS faster — **by ~3.7 ms, 0.11 % of the encode**,
which is at or below this band's resolution (its own null is 0.12 %). It did not
reach C's 5.6 ms.

**Why: `sz` is small.** The intra edge is 9..65 samples, so a batch of 8 runs
one to seven times and the per-batch setup — twelve scalar reads into the
window, five overlapping vector loads — plus the scalar tail dominate. The
5.3 ms residual is the `sz < 9` declines, which the kernel refuses outright.

**The general lesson, and it is the third instance this session**: a kernel with
a real ISA counterpart in libaom is not automatically a lever. KB-PERF-20's
predictor paid (−0.372 %, p<0.0001) because its unit of work is a 4x2 block
repeated across a whole transform; this one's unit of work is a **single short
1-D edge**, so vector width has almost nothing to amortise over. **Check the
trip count, not just the ratio.**

To make it pay you would have to attack the setup rather than the arithmetic —
a 4-wide batch for short `sz`, or filtering all four edges of a block in one
call. Neither is obviously worth 0.1 %.

Band kept: `.rejected.band1024s3.tsv`.

## Recorded: KB-PERF-20 independently confirmed

The same profile shows the landed predictor at
`filter_intra_predict_high_impl_v3` **13.4 ms**, against **31.1 ms** scalar
before it — a 17.7 ms drop, so **4.26x -> 1.84x** against libaom's 7.3 ms. That
is an independent symbol-level confirmation of a landing whose band read
−0.372 % (~13 ms).

## Correction to KB-PERF-20's record

Its closing note said `cdef_find_dir_simd_diff` fails under unrelated bite-proof
perturbations and called that *"reproducible rather than a one-off"*. With a
third perturbation now run (this one), it did **NOT** fail. So it is **flaky /
order-dependent — two of three — not deterministic**, and the stronger wording
was wrong.
