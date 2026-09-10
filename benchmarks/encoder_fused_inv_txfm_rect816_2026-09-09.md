# KB-PERF-30 — the fused 8x16 / 16x8 INVERSE transforms: **−0.54 %**, and the padding diagnosis now holds from BOTH sides

**2026-09-09.** Byte-identical; **−0.54 %** at 1024x1024 cq27 `--cpu-used 3`
(25/30 rounds faster, p=0.0003) over a 30-round rotated band, against a null
that is **exactly flat** — +0.00 %, 15/30, p=1.0000.

**The null being dead flat is what makes this reading unusually clean.** The
previous four bands all carried a ~0.27 pp copy systematic that rotation cannot
remove, so their headline had to be quoted as the mean of two base copies. Here
the two base copies agree to 0.00 %, so **−0.54 % is the number, not an
average of two disagreeing ones.**

## It is the control for KB-PERF-28, from the inverse side, and it confirms it

KB-PERF-28 (the padded 4x8/8x4 inverse) was the one kernel of the sequence to
miss its call-share prediction, and attributed the miss to padding: at a
4-dimension a load covers four of eight lanes, so vectors are built per-lane
with `from_array` and the reconstruction stores per-lane — five such sites
against `inv_8x8_fused`'s two. KB-PERF-29 tested that on the FORWARD side and
landed back in the unpadded band. This is the same test on the inverse side,
where the glue is worse (the reconstruction must LOAD the destination, add,
clamp and store back, per lane).

| kernel | share of its domain | measured | per share-point |
|---|---:|---:|---:|
| 8x8 forward | 25.74 % | −2.017 % | 0.078 |
| 8x8 inverse | 26.96 % | −1.538 % | **0.057** |
| 4x8+8x4 forward | 16.73 % | −0.804 % | 0.048 |
| 16x16 forward | 5.62 % | −0.600 % | 0.107 |
| 16x16 inverse | 5.04 % | −0.480 % | 0.095 |
| 8x16+16x8 forward | 9.02 % | −0.534 % | 0.059 |
| **8x16+16x8 inverse** | **9.52 %** | **−0.54 %** | **0.057** |
| *4x8+8x4 inverse (padded)* | *15.16 %* | *−0.310 %* | *0.020* |

**0.057 is the inverse-8x8 rate to three decimals.** So the unpadded band is
0.048–0.107 across seven kernels and both directions, the padded inverse sits
at 0.020, and the explanation is padding rather than "rectangular" or
"inverse". **The KB-PERF-28 fix is worth doing rather than speculative, and it
is now the largest remaining item in this sequence.**

## The kernel

Written over GROUPS exactly like KB-PERF-29's forward — `CG = col_n / 8` column
groups and `RG = row_n / 8` row groups — so one body covers both shapes and has
the same structure as the landed 8x8 (1,1) and 16x16 (2,2) inverses. Both
dimensions are >= 8, so every coefficient load is `from_slice`. The transpose is
`CG * RG` 8x8 blocks; `tt[rg][cg * 8 + k]` is row `rg * 8 + k` over columns
`cg * 8 .. cg * 8 + 8`, which is exactly what the column pass consumes.

Three recipe facts, each READ from the table rather than carried over from a
neighbouring kernel:

* **`INV_SHIFT` is `[-1, -4]`** — 8x8's, not 4x8/8x4's `[0, -4]` (whose row
  shift vanishes) and not 16x16's `[-2, -4]`. The row pass ends in a rounding
  shift by 1.
* **`rect_type == +-1`**, so the row pass carries the `NEW_INV_SQRT2` scaling,
  and it lands **before** the clamp, mirroring the driver's own order.
* **`lr_flip` is the LANE reversal** the inverse side uses (output column `c`
  reads `buf` column `col_n - 1 - c`), which at `CG == 2` makes the two column
  groups **exchange as well as reverse** — the shape the fused 16x16 inverse
  already had, generalised over `CG`.

Config comes from `get_inv_txfm_cfg`, not hand-indexed tables.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6) — two sha256-distinct
  binaries built from one tree.
* `-p zenav1-aom-dsp` **446/446 in both dispatch modes**, green on the first
  run.
* **Bite proof, asymmetric**: dropping the `lr_flip` column-group exchange
  (`sg = cg`) fails `inv_txfm2d_differential_fuzz` and
  `inv_txfm2d_lowbd_differential_fuzz` — both against the **real exported C** —
  plus two more, while **442 stay green**. Note the perturbation is inert at
  `CG == 1`, so it is the 16x8 half that carries it; that is the point of
  choosing it over a blanket shift change.

## The band

`.band1024s3.tsv`, 30 rotated rounds, `base` / `new` / `baseB` cycling position
each round, one encode per arm per round.

| arm | median ms | spread | paired median | rounds faster | p |
|---|---:|---:|---:|---:|---:|
| base | 3273.0 | 1.9 % | — | — | — |
| baseB | 3273.8 | 1.2 % | +0.00 % | 15/30 | 1.0000 |
| new | 3251.8 | 2.2 % | **−0.54 %** | **25/30** | **0.0003** |

## Remaining, in order

* **The KB-PERF-28 fix** — branch on shape so the padded 4x8/8x4 inverse uses
  whole-vector `from_slice`/`store` with a tail. Now supported by two controls
  rather than one, and worth ~0.55 pp if it reaches the unpadded rate.
* **4x16 + 16x4** — 0.61 % + 1.05 % of inverses at s3, 1.45 % / 1.66 % of
  forwards. Padded on one dimension, so it should be done AFTER the KB-PERF-28
  fix, which is the same problem.
* Both dimensions >= 32 is ~0.6 % and is not worth a kernel.

## Not covered

One box, one content class, `--cpu-used 3` (the shipping preset), x86-64.
