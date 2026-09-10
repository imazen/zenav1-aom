# KB-PERF-29 — the fused 8x16 / 16x8 forward transforms: **−0.534 %**, and it CONFIRMS KB-PERF-28's diagnosis

**2026-09-09.** Byte-identical; **−0.534 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.510 %** at 27/30 and **−0.559 %** at 26/30, both
p <= 0.0001), over a 30-round rotated band with a **clean null** (+0.045 %,
13/30, p=0.58).

## The prediction it was built to test

KB-PERF-28 (4x8/8x4 inverse) was the first kernel of the sequence to miss its
call-share prediction — 0.020 percentage points per share-point against the
others' 0.048-0.107 — and attributed it to **padding**: at a 4-dimension a load
covers four of eight lanes, so vectors are built per-lane with `from_array`
instead of `from_slice`, and the reconstruction stores per-lane. Five such sites
against `inv_8x8_fused`'s two.

**8x16 / 16x8 is the control for that hypothesis**: rectangular 2:1 like
4x8/8x4, but with **both dimensions >= 8**, so no padding — every load is
`from_slice` and every store a whole-vector `store`.

| kernel | share | per share-point |
|---|---:|---:|
| 8x8 forward | 25.74 % | 0.078 |
| 8x8 inverse | 26.96 % | 0.057 |
| 4x8+8x4 forward | 16.73 % | 0.048 |
| 16x16 forward | 5.62 % | 0.107 |
| 16x16 inverse | 5.04 % | 0.095 |
| **4x8+8x4 INVERSE (padded)** | 15.16 % | **0.020** |
| **8x16+16x8 forward (unpadded)** | **9.02 %** | **0.059** |

**It lands squarely back in the unpadded band.** The diagnosis holds, which
makes the KB-PERF-28 fix worth doing rather than speculative.

## The kernel

Written over GROUPS rather than a fixed size — `CG = col_n / 8` column groups
and `RG = row_n / 8` row groups — so one body covers both shapes, and it is the
same structure as the landed 8x8 (1,1) and 16x16 (2,2) kernels. The transpose is
`CG * RG` 8x8 blocks, each landing on exactly the (column group, row group) pair
the row pass consumes.

`FWD_SHIFT` is `[2, -2, 0]` for both — 16x16's recipe — with `rect_type == +-1`,
so the row pass carries the `NEW_SQRT2` scaling. `lr_flip` uses KB-PERF-27's
array-order reversal.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**, green on the first run.
* Config from **`get_fwd_txfm_cfg`** rather than hand-indexed tables.
* **Bite proof**: writing a transpose block to the wrong slot within its column
  group fails `txfm2d_differential_fuzz` (against the **real exported C**) and
  `txfm2d_simd_perm_diff` while **393 stay green**.

## What is left

* **The 8x16 / 16x8 INVERSE** — 9.52 % of inverses, `INV_SHIFT = [-1, -4]`.
  Unpadded, so it should behave like this one.
* **The KB-PERF-28 fix** — branch on shape so the padded inverse uses
  whole-vector loads/stores with a tail. Its own band.
* 4x16 + 16x4: 1.45 % / 1.66 %.
* Both dimensions >= 32: ~0.6 % combined — not worth a kernel.
