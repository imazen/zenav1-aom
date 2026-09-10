# KB-PERF-31 — the KB-PERF-28 fix: **−0.34 %**, and it recovers most of the padding shortfall without closing it

**2026-09-09.** Byte-identical; **−0.34 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.217 %** at 22/30, p=0.0161, and **−0.462 %** at
25/30, p=0.0003), over a 30-round rotated band. The null is not significant and
points the other way (+0.07 %, 13/30, p=0.58).

This is the fix KB-PERF-28 named for itself and that KB-PERF-29 and KB-PERF-30
then justified from both directions.

## What it was, and what it is now

KB-PERF-28 was the one kernel of the fusion sequence to miss its call-share
prediction — **0.020 percentage points per share-point** against the other
kernels' 0.048–0.107 — and attributed the miss to padding glue: five
`from_array`/`to_array` sites against `inv_8x8_fused`'s two. Three of those five
are removable by branching on the shape, because **each shape has one dimension
that is a whole vector**:

| site | 4x8 (`col_n` 4, `row_n` 8) | 8x4 (`col_n` 8, `row_n` 4) |
|---|---|---|
| row-pass coefficient load | **now `from_slice`** | stays padded (4-long column) |
| `lr_flip` reversal | stays an array round trip | **now `revv`** |
| reconstruction load/store | 4-element fixed-size read (was a slice walk) | **now the whole-vector form the unpadded inverses use** |

The 4-wide reconstruction is a *tail*, not a slice walk: a `[u16; 4]` read via
`try_into` is one bounds check per row rather than one per lane, which is the
same shape the unpadded kernels have, just half as wide.

## The result, in the sequence's own units

| kernel | share | measured | per share-point |
|---|---:|---:|---:|
| 8x8 forward | 25.74 % | −2.017 % | 0.078 |
| 8x8 inverse | 26.96 % | −1.538 % | 0.057 |
| 4x8+8x4 forward | 16.73 % | −0.804 % | 0.048 |
| 16x16 forward | 5.62 % | −0.600 % | 0.107 |
| 16x16 inverse | 5.04 % | −0.480 % | 0.095 |
| 8x16+16x8 forward | 9.02 % | −0.534 % | 0.059 |
| 8x16+16x8 inverse | 9.52 % | −0.54 % | 0.057 |
| 4x8+8x4 inverse, **before** | 15.16 % | −0.310 % | *0.020* |
| **4x8+8x4 inverse, after** | **15.16 %** | **−0.65 %** | **0.043** |

**The glue was most of the shortfall but not all of it.** 0.020 → 0.043 against
an unpadded floor of 0.048 recovers roughly four fifths of the gap, and the
residual is consistent with the one thing branching cannot fix: **in each shape
one of the two passes runs eight lanes with four live.** That is inherent to a
4-dimension, and it is corroborated from the other side — KB-PERF-27, the
FORWARD twin, has the same half-idle passes and sits at 0.048, the lowest of the
unpadded band.

So the honest reading is that the padded pair has a real ceiling slightly under
the unpadded band, the glue explanation carried most of the distance to it, and
there is no further fix of this kind left to make here.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6) — two sha256-distinct
  binaries built from one tree.
* `-p zenav1-aom-dsp` **446/446 in both dispatch modes**, green on the first run.
* Nothing arithmetic changed: the same values are loaded, reversed and stored,
  by a different mechanism. The `lr_flip` branch is the one place that needed
  care — `revv` reverses all eight lanes, which is correct only when all eight
  are live, so it is taken at `col_n == 8` and nowhere else.

## The band

`.band1024s3.tsv`, 30 rotated rounds, one encode per arm per round.

| arm | median ms | spread | paired median | rounds faster | p |
|---|---:|---:|---:|---:|---:|
| base | 3261.4 | 2.4 % | — | — | — |
| baseB | 3265.1 | 3.1 % | +0.07 % | 13/30 | 0.5847 |
| new vs base | 3255.7 | 1.8 % | −0.217 % | 22/30 | 0.0161 |
| new vs baseB | | | −0.462 % | 25/30 | 0.0003 |

The two base copies disagree by 0.245 pp — the ~0.27 pp copy systematic this
harness has carried in four earlier bands and that rotation cannot remove — so
the headline is their mean, **−0.34 %**, not the better of the two.

## Remaining in this sequence

* **4x16 + 16x4** — 1.66 % of inverses and 3.11 % of forwards at s3. Padded on
  one dimension like this pair, so at ~0.043 per share-point it is worth
  ~−0.2 pp: the smallest kernel yet, and the first whose expected effect is
  close to what a 30-round band can resolve.
* Both dimensions >= 32 is ~0.6 % and is not worth a kernel.

## Not covered

One box, one content class, `--cpu-used 3` (the shipping preset), x86-64.
