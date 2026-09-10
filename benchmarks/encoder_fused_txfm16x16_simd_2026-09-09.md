# KB-PERF-25 — the fused 16x16 forward transform: **−0.600 %**

**2026-09-09.** Byte-identical; **−0.600 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.706 %** at 27/30 and **−0.493 %** at 26/30, both
p <= 0.0001), over a 30-round rotated band.

The third size in the KB-PERF-23/24 sequence, and **above the −0.44 % predicted**
from scaling the 8x8 result by call share.

## It was sized honestly beforehand, including its worse cost/benefit

`encoder_reprofile_after_8x8_2026-09-09.md` costed this before it was written and
said plainly that it is *"the first one in this sequence whose cost/benefit is
materially worse than its predecessor"* — 5.62 % of forwards against 8x8's
25.74 %, for roughly twice the code. That held: it delivered ~0.6 % against the
8x8 forward's 2.017 %, for a kernel about 1.7x the size.

The estimate of 3x the code was pessimistic. Factoring the 8x8 transpose into
one closure makes 16x16 **two vectors per row, four transpose calls, two kernel
calls per pass** — additive rather than multiplicative.

## The 16x16 shape

`i32x8` holds eight lanes, so sixteen columns is TWO vectors per row and the
16x16 transpose is **four 8x8 block transposes**: `w_lo[0..8]`, `w_lo[8..16]`,
`w_hi[0..8]`, `w_hi[8..16]` give columns 0-7 x rows 0-7, columns 0-7 x rows
8-15, columns 8-15 x rows 0-7 and columns 8-15 x rows 8-15 — **exactly the two
row-groups the row pass consumes**, so no off-diagonal swap is needed. The row
pass then runs once per 8-row group.

## What differs from 8x8, and both were read from the tables

* `FWD_SHIFT[TX_16X16] = [2, -2, 0]`, so `shift1` is a round-shift by **2**
  (8x8's is 1); the tail is again a no-op and `rect_type == 0`.
* **`COS_BIT_ROW[2][2]` is 12, not 13** — the row and column cos bits DIFFER at
  this size, unlike 8x8 where both are 13. Copying 8x8's single `cos_bit` would
  have been silently wrong on the row pass.
* **`lr_flip` is a half-swap AND a lane reverse**: source column `c` lands at
  `15 - c`, so the two halves exchange (`wlo <- rev(hi)`, `whi <- rev(lo)`).
  At 8x8 it was a plain lane reverse. **That is the bite proof.**

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes** — and green on the
  FIRST run, unlike KB-PERF-23 and -24, whose only defects were an
  index-computed transpose stage and a swapped `opt_range` tuple. Writing both
  out explicitly this time is why.
* **Bite proof**: reversing the halves without swapping them (`wlo <- rev(lo)`)
  fails `txfm2d_differential_fuzz` (against the **real exported C**) and
  `txfm2d_simd_perm_diff` while **393 stay green**.

## The band, and why it is quoted as a mean

30 rotated rounds, two sha256-distinct binaries from one tree, TWO copies of the
base:

| comparison | paired median | rounds faster | p |
|---|---:|---:|---:|
| `baseB` vs `base` (same binary) | **−0.238 %** | 24/30 | 0.0014 |
| fused 16x16 vs `base` | **−0.706 %** | 27/30 | <0.0001 |
| fused 16x16 vs `baseB` | **−0.493 %** | 26/30 | 0.0001 |

**The null is significant in the SAME direction** — the copy systematic
`encoder_rotate_reverify_2026-08-03.md` measures at ~0.27 pp — so the raw
−0.706 % would overstate. Quoting the mean of the two, **−0.600 %**, as
KB-PERF-22 did for the same reason.

## Not covered

* **The INVERSE 16x16 is untouched** (5.04 % of inverses). `INV_SHIFT[TX_16X16]
  = [-2, -4]`, so like the 8x8 pair it needs its own shifts, plus the two
  `clamp_buf` calls and the reconstruction tail.
* **8x4 and 4x8 are together 16.7 % of forwards — more than 16x16** — but are
  rectangular (`NEW_SQRT2` scaling) with one 4-dimension, so one pass runs a
  half-filled batch. The standalone `row_n == 4` arm measured NULL; inside a
  fused kernel the transpose is in registers either way, so that calculus
  changes, but it is unmeasured.
* x86-64 only; aarch64 and wasm keep the generic driver.
