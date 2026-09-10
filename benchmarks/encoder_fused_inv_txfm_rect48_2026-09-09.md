# KB-PERF-28 — the fused 4x8 / 8x4 INVERSE transforms: **−0.310 %**, and the first kernel to MISS its share prediction

**2026-09-09.** Byte-identical; **−0.310 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.254 %** at 22/30, p=0.0161, and **−0.366 %** at
24/30, p=0.0014), over a 30-round rotated band. The null is **not** significant
here (+0.165 %, 13/30, p=0.58).

**It is the first of the six fusion kernels to fall short of what its call share
predicted**, and that is the interesting part.

## The prediction, and the miss

Across the previous five, effect tracked call share almost linearly:

| kernel | share | measured | per share-point |
|---|---:|---:|---:|
| 8x8 forward | 25.74 % | −2.017 % | 0.078 |
| 8x8 inverse | 26.96 % | −1.538 % | 0.057 |
| 4x8+8x4 forward | 16.73 % | −0.804 % | 0.048 |
| 16x16 forward | 5.62 % | −0.600 % | 0.107 |
| 16x16 inverse | 5.04 % | −0.480 % | 0.095 |
| **4x8+8x4 inverse** | **15.16 %** | **−0.310 %** | **0.020** |

The inverse-8x8 rate (0.057) would have predicted **~−0.87 %**. It delivered a
third of that.

## The likely cause, and it is measurable rather than speculative

**The rect kernel carries 2.5x the scalar glue of the square ones**: five
`from_array` / `to_array` sites against `inv_8x8_fused`'s two. The padded
dimension is why — at `row_n == 4` a load covers four of eight lanes, so the
vector is built per-lane instead of with `from_slice`, and the reconstruction
stores per-lane instead of with a whole-vector `store`. Each of those is a stack
round trip that the square kernels do not pay.

**The forward twin (KB-PERF-27) has the same padding but hit its prediction**,
which is consistent: its store is `copy_from_slice` of a `to_array` — one round
trip — whereas the inverse's reconstruction must LOAD the destination, add,
clamp and store back, per lane.

**So this is a fixable shortfall, not a ceiling.** The follow-up is to branch on
the shape: at `col_n == 8` (the 8x4 case) the reconstruction can use whole-vector
`from_slice`/`store` with a 4-lane tail, and at `row_n == 8` (4x8) the row-pass
load can use `from_slice` directly. That was not done here because the effect is
already positive and significant and the change deserves its own band.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**, green on the first run.
* **`INV_SHIFT` is `[0, -4]` for both sizes, so there is NO row shift at all** —
  the generic's `round_shift_array(_, -shift[0])` early-returns at 0. Simpler
  than 8x8's, and read from the table rather than assumed.
* The rect scaling is `NEW_INV_SQRT2` and lands in the row pass **before** the
  clamp, mirroring the driver's order. **That is the bite proof**: swapping them
  fails `inv_txfm2d_differential_fuzz`, `inv_txfm2d_lowbd_differential_fuzz`
  (both against the **real exported C**) and `txfm2d_simd_perm_diff` while
  **392 stay green**.
* **`lr_flip` is a LANE reverse here, not KB-PERF-27's array-order reversal, and
  the asymmetry is real.** The forward's flip permutes which column POSITION a
  result occupies before the row pass reads positions in order; the inverse
  reads `buf[r][col_n-1-c]` for output column `c`, i.e. across LANES of one
  vector. At `col_n == 4` that is a reverse of the four LIVE lanes inside an
  eight-lane vector, which `revv` does not do — so the flipped path goes through
  `to_array`, paid only on FLIPADST types.

## The band

| comparison | paired median | rounds faster | p |
|---|---:|---:|---:|
| `baseB` vs `base` (same binary) | +0.165 % | 13/30 | 0.5847 |
| fused inverse 4x8/8x4 vs `base` | −0.254 % | 22/30 | 0.0161 |
| fused inverse 4x8/8x4 vs `baseB` | −0.366 % | 24/30 | 0.0014 |

## What is left in the sequence

* **16x8 + 8x16: 9.02 % of forwards, 9.52 % of inverses** — rectangular with a
  2:1 ratio like these, but both dimensions are >= 8, so **no padding and none of
  the scalar glue above**. On the pattern that should behave like the square
  kernels.
* 4x16 + 16x4: 1.45 % / 1.66 %.
* Both dimensions >= 32: ~0.6 % combined — not worth a kernel.
* And the fix named above for this kernel.
