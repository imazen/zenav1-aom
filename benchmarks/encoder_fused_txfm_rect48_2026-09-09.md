# KB-PERF-27 — the fused 4x8 / 8x4 forward transforms: **−0.804 %**

**2026-09-09.** Byte-identical; **−0.804 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.891 %** at 28/30 and **−0.716 %** at 27/30, both
p < 0.0001), over a 30-round rotated band.

**The largest of the four landings after the 8x8 pair**, which is what its share
predicted: 4x8 and 8x4 are together **16.7 % of forward transforms** at this
preset — three times 16x16's 5.62 %.

## One body, two shapes

Both fit inside eight vectors of eight lanes with some lanes or vectors unused,
so a single kernel serves both: `col_n` live lanes per row vector, `row_n` row
vectors, ONE 8x8 transpose (padding vectors are zero and the unused transpose
outputs are simply not read), then `col_n` row vectors with `row_n` live lanes.

`FWD_SHIFT` is `[2, -1, 0]` for both — the same recipe as 8x8.

## Two things differ from the square kernels

* **`rect_type == +-1`, so the row pass applies the `NEW_SQRT2` scaling** that
  4x4 / 8x8 / 16x16 all skip. That is the bite proof.
* **`lr_flip` is an ARRAY-ORDER reversal, not a lane reverse — and this is
  simpler than what the square kernels do.** The generic column pass writes lane
  `j`'s result to `buf[r][col_n-1-j]`: it permutes which column POSITION each
  source column occupies before the row pass reads them in order. After the
  transpose `t[c]` already holds source column `c`, so handing the row pass
  `t[col_n-1-p]` is the same permutation **and costs nothing**. KB-PERF-23/25
  reverse lanes instead, which is equivalent at those sizes; this form is
  cheaper and works at any width. It is not worth re-opening the landed square
  kernels to change it, but a future size should use this shape.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**; green on the first
  run, as with KB-PERF-25/26.
* The config comes from **`get_fwd_txfm_cfg` itself** rather than hand-derived
  table lookups, so it cannot drift from the generic path — a tightening over
  the 8x8 and 16x16 hooks, which index `TXFM_TYPE_LS` / `COS_BIT_*` by hand.
  (`txfm_type_col` is indexed by HEIGHT and `txfm_type_row` by WIDTH, which is
  exactly the sort of thing worth not restating.)
* **Bite proof**: dropping the `NEW_SQRT2` scaling fails
  `txfm2d_differential_fuzz`, `txfm2d_edge_cases` (both against the **real
  exported C**) and `txfm2d_simd_perm_diff` while **392 stay green**.

## The band

30 rotated rounds, two sha256-distinct binaries, TWO copies of the base:

| comparison | paired median | rounds faster | p |
|---|---:|---:|---:|
| `baseB` vs `base` (same binary) | −0.165 % | 24/30 | 0.0014 |
| fused 4x8/8x4 vs `base` | **−0.891 %** | 28/30 | <0.0001 |
| fused 4x8/8x4 vs `baseB` | **−0.716 %** | 27/30 | <0.0001 |

The null is again significant in the same direction (the ~0.27 pp copy
systematic), so the quoted figure is the **mean, −0.804 %**.

## The fusion sequence so far

| kernel | share of its direction | measured |
|---|---:|---:|
| 8x8 forward | 25.74 % | −2.017 % |
| 8x8 inverse | 26.96 % | −1.538 % |
| **4x8 + 8x4 forward** | **16.73 %** | **−0.804 %** |
| 16x16 forward | 5.62 % | −0.600 % |
| 16x16 inverse | 5.04 % | −0.480 % |

Effect tracks call share closely across a 5x span, which makes the remaining
sizes predictable rather than speculative.

## What is left

* **The 4x8 / 8x4 INVERSE** (15.16 % of inverses) is untouched and is now the
  largest single remaining item. `INV_SHIFT[TX_4X8] / [TX_8X4]` and the
  `NEW_INV_SQRT2` row scaling are its own recipe.
* 16x8 + 8x16 are 9.02 % of forwards, also rectangular; 4x16 + 16x4 are 1.45 %.
* Everything with both dimensions >= 32 is ~0.6 % and is not worth a kernel.
