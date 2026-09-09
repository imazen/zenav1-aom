# KB-PERF-17 — the fused 4x4 INVERSE, and the inverse census that had never been taken

**Landed 2026-09-09**, on top of KB-PERF-16's forward twin. Byte-identical;
**−0.79 % at 1024x1024** (14/14) and **−1.32 % at 512x512** (24/24).

## The census had to come first, and it did not exist

KB-PERF-16's record named this precisely: *"the INVERSE side is untouched and is
LARGER in the profile ... its size distribution is still unmeasured —
`note_fwd_txfm` has no inverse twin — and that census should come before
assuming the mixes match."* So the twin was added
(`census::note_inv_txfm`, hooked at `av1_inv_txfm2d_add_into`, printed by
`content_census`) before any kernel work.

**It was worth measuring rather than assuming.** Same cell, same speed:

| | forward | inverse |
|---|---:|---:|
| total transforms | 1,196,534 | 866,354 |
| **4x4** | **50.70 %** | **44.43 %** |
| 8x8 | 22.30 % | 25.66 % |
| 4x4 + 8x8 | 73.0 % | 70.1 % |

Close, and **not equal** — 6.3 pp apart on the size that matters. Near enough
that the forward's conclusion carries; far enough that "assume it matches" was
a claim, not a fact.

## The inverse recipe collapses DIFFERENTLY, and copying the forward's argument
## would have been wrong

This is the part worth keeping. At TX_4X4:

| | forward | inverse |
|---|---|---|
| shift table | `FWD_SHIFT[TX_4X4] = [2, 0, 0]` | `INV_SHIFT[TX_4X4] = [0, -4]` |
| which shift vanishes | both POST-pass shifts (0) | the ROW shift (0) |
| what survives | an exact `* 4` on the input | a real rounding shift by 4 on the COLUMN output |

So the two are near-mirror images: the forward's no-ops are the inverse's live
shift and vice versa. A fused inverse written by copying the forward's argument
would have dropped a rounding shift and been wrong in a way the byte gates would
have caught only after the fact.

Everything else is specialised rather than changed — same kernels via
`inv_txfm_func`, same `INV_COS_BIT`, both `clamp_buf` calls, the `opt_range(bd)`
stage ranges, the flips, and `highbd_clip_pixel_add`. `remap_input` already
borrowed rather than copied for every non-64-point size, so TX_4X4 never needed
it and the fused path indexes `input` directly. Each precondition is CHECKED and
the function declines to the generic driver rather than diverging.

## Measured

Two sha256-distinct binaries from one tree, rotated arms, same-binary null.
The base arm's 1 MP median is **9892.0 ms** against KB-PERF-16's post-landing
**9892.1 ms** — a 0.1 ms agreement that confirms the baseline is exactly the
landed HEAD rather than a re-derived number. Output byte-identical (10,912 B /
39,694 B).

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 9892.0 ms | 9820.8 ms | **−0.79 %** | **14/14** | 0.0001 | **−0.29 %**, 9/14, p=0.42 |
| 512x512 cq27 s0 | 3515.9 ms | 3468.6 ms | **−1.32 %** | **24/24** | <0.0001 | +0.16 %, 8/24, p=0.15 |

**The 1 MP null is worth flagging rather than burying: −0.29 %, a third of the
effect.** It is not significant (p=0.42) and the effect is 14/14 rounds at
2.7x its size, but this band is less clean than the 512x512 one (+0.16 % null,
24/24) and than KB-PERF-16's (−0.04 %). Read the 512 band as the sharper
measurement and the 1 MP one as confirming sign and order.

Ratio at 1 MP against the same C median (4170.3 ms): **2.372x → 2.355x**.

## Combined with KB-PERF-16

Both 4x4 fusions together, from the cycle's opening baseline:
**10051.6 → 9820.8 ms at 1 MP (−2.30 %)** and **3595.7 → 3468.6 ms at 512x512
(−3.53 %)**; ratio **2.410x → 2.355x**.

## Not covered

* **8x8 is next and is 25.66 % of the inverse** (22.30 % of the forward).
  `INV_SHIFT[TX_8X8]` is `[-1, -4]`, so BOTH shifts are live there — neither 4x4
  argument transfers, and the proof has one more moving part again.
* The bd8 **lowbd** inverse twin (`u8` destination) is untouched; it is the
  decoder's path, and this landing did not measure it.
* One box, one content class, `--cpu-used 0`, x86-64.
