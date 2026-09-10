# KB-PERF-26 — the fused 16x16 INVERSE transform: **−0.480 %**

**2026-09-09.** Byte-identical; **−0.480 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.586 %** at 28/30 and **−0.374 %** at 26/30, both
p <= 0.0001), over a 30-round rotated band. **Above the ~0.3 % predicted.**

The fourth and last size in the KB-PERF-23/24/25 sequence, and the point at
which it has become mechanical: **green on the first run**, like KB-PERF-25 and
unlike the first two.

## What it is

KB-PERF-24's inverse at twice the width, with KB-PERF-25's two-vectors-per-row
structure. `INV_SHIFT[TX_16X16] = [-2, -4]`, so the row shift is **2** (8x8's is
1) and the column shift is again 4.

The 8x8 inverse's structural advantage carries: the input is COLUMN-major
(`mod_input[c * row_n + r]`), so the row pass loads contiguously with lane = r
and needs no transpose. The four 8x8 block transposes sit between the passes,
and under `lr_flip` the two column halves exchange AND reverse — the same shape
as the fused 16x16 forward, and different from 8x8's plain lane reverse.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**.
* **Bite proof on the axis that is easy to confuse**: `ud_flip` REORDERS the
  sixteen output vectors (output row `r` takes `to[15-r]`) — it is not a lane
  operation, unlike `lr_flip`. Dropping that reorder fails FOUR tests —
  `inv_txfm2d_differential_fuzz`, `inv_txfm2d_lowbd_differential_fuzz`,
  `recon_lowbd_differential_fuzz` (all against the **real exported C**) and
  `txfm2d_simd_perm_diff` — while **391 stay green**.

## The band

30 rotated rounds, two sha256-distinct binaries, TWO copies of the base:

| comparison | paired median | rounds faster | p |
|---|---:|---:|---:|
| `baseB` vs `base` (same binary) | −0.226 % | 22/30 | 0.0161 |
| fused inverse 16x16 vs `base` | **−0.586 %** | 28/30 | <0.0001 |
| fused inverse 16x16 vs `baseB` | **−0.374 %** | 26/30 | 0.0001 |

The null is again significant in the same direction (the ~0.27 pp copy
systematic), so the figure quoted is the **mean of the two, −0.480 %**.

## The sequence, complete

| size | share of its direction | measured |
|---|---:|---:|
| 8x8 forward | 25.74 % | **−2.017 %** |
| 8x8 inverse | 26.96 % | **−1.538 %** |
| 16x16 forward | 5.62 % | **−0.600 %** |
| 16x16 inverse | 5.04 % | **−0.480 %** |

**The two 16x16 kernels beat their call-share prediction** (−0.44 % and ~0.3 %
respectively). Larger transforms cost more per call, so the per-call driver
saving is a smaller FRACTION of each — but there is also more `buf` traffic to
remove, and that appears to dominate.

## What is left, and it is the largest remaining share

**8x4 + 4x8 are together 16.7 % of forwards — three times 16x16's share** — and
are untouched. They are RECTANGULAR (`rect_type = +-1`, so the `NEW_SQRT2`
scaling applies) with one 4-dimension, so one of the two passes runs a
half-filled batch. The standalone `row_n == 4` arm measured NULL
(`encoder_fwd_row_4wide_2026-09-09.rejected.md`) because an 8x8 transpose ran at
full cost on half-zero data — **but inside a fused kernel the transpose is in
registers either way, so that calculus changes.** It is unmeasured, and it is
the next thing to try.

32x32 and above are together ~0.6 % of transforms and are not worth a kernel.
