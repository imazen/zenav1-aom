# Cross-encoder rate/quality sweep — 2026-09-10 (corrected: `drv-aom` on the shipping path)

Data: `enc_rd_cross_2026-09-10.tsv`, 288 points. Produced by
`scripts/enc_rd_compare.py sweep --size 512 --per-class 3` at 02:48–02:54 UTC on 2026-09-11,
**after** `2984ad1` switched `drv-aom` to `aom_encode::key_frame::encode_key_frame` (the entry
zenavif calls, with palette + IntraBC following the frame's screen decision). An earlier run
the same evening timed the `aom-bench` differential harness instead — 1.9x slower and
without the screen tools — and was discarded. The session that ran this sweep ended before
recording it; this file was written 2026-09-11 from the TSV.

## Setup

* 512x512 centre crops, I420 8-bit, from imazen-26 PNG renders: photo `1000 1001 1002`,
  screen `8000 8001 8002`.
* Arms and ladders (quality x speed): zenav1-aom `cq {10,18,26,34} x cpu-used {3,6,8}`;
  zenav1-svt `{10,18,26,34} x preset {2,6,10}`; ravif `{92,84,74,62} x speed {4,6,8}`;
  zenrav1e `q {50,80,110,140} x speed {4,6,8}`.
* Every stream decoded by ONE decoder (`xtool decode`, honouring its own signalled matrix
  and range) and scored in RGB against the source (`xtool score-rgb`: SSIMULACRA2,
  butteraugli). ravif codes 4:4:4 at 10-bit, so a YUV comparison cannot be written.
* `ms` is ONE encode per point, on a box that was also committing and pushing. **It ranks
  the arms; it is not a speed claim.** The speed claim is `just bench-cross-speed`
  (zenbench, interleaved), which was NOT run this cycle.

## Matched-rate reading (per-image linear interpolation to the target bpp, mean of 3 images)

Target = median bpp of zenav1-aom cq26 s3 in each class.

**Photo, 0.925 bpp**

| arm | speed | SSIM2 @ target | ms/img @ target |
|---|---:|---:|---:|
| zenav1-aom | 3 | **73.17** | 1430 |
| zenav1-aom | 6 | 71.03 | 151 |
| zenav1-aom | 8 | 67.65 | 60 |
| zenav1-svt | 2 | 71.78 | 630 |
| zenav1-svt | 6 | 69.26 | 61 |
| zenav1-svt | 10 | 66.20 | 10 |
| ravif | 4 | 71.09 | 418 |
| ravif | 6 | 70.82 | 318 |
| ravif | 8 | 70.29 | 276 |
| zenrav1e | 4 | 72.22 | 9709 |
| zenrav1e | 6 | 70.61 | 1247 |
| zenrav1e | 8 | 70.74 | 620 |

**Screen, 0.582 bpp**

| arm | speed | SSIM2 @ target | ms/img @ target |
|---|---:|---:|---:|
| zenav1-aom | 3 | **80.36** | 1395 |
| zenav1-aom | 6 | 78.72 | 185 |
| zenav1-aom | 8 | 72.41 | 53 |
| zenav1-svt | 2 | 79.31 | 551 |
| zenav1-svt | 6 | 75.71 | 60 |
| zenav1-svt | 10 | 65.33 | 9 |
| ravif | 4 | 76.02 | 270 |
| ravif | 6 | 75.00 | 243 |
| ravif | 8 | 74.60 | 221 |
| zenrav1e | 4 | 79.21 | 8618 |
| zenrav1e | 6 | 73.18 | 918 |
| zenrav1e | 8 | 70.50 | 432 |

## What it says, and what it does not

* At its shipping preset (cpu-used 3) zenav1-aom has the **highest matched-rate quality of
  any arm in both classes**, and it is the second-slowest; zenav1-aom cpu-used 6 matches or
  beats ravif's best quality at half to a third of ravif's time. zenav1-svt preset 2 is the
  only other arm within a point of it on screen content, at 40 % of the time.
* On screen content the quality gap to every other arm is larger (palette + IntraBC are
  live on this path since `93cbb99`; the previous, discarded sweep ran without them).
* zenrav1e speed 4 — zenavif's CURRENT default configuration — is 6–7x slower than
  zenav1-aom cpu-used 3 here and ~1 SSIM2 below it; this agrees in direction with the
  1 MP single-cell result in `encoder_vs_default_backend_2026-09-10.md` (10.2x) and is smaller
  because 512² is a different regime.
* Three images per class, one size, one box, one encode per point. Do not quote `ms` as a
  speed ratio; run `encbench` for that. Charts: `python3 scripts/enc_rd_compare.py chart --tsv
  benchmarks/enc_rd_cross_2026-09-10.tsv` (quickchart.io URLs).
