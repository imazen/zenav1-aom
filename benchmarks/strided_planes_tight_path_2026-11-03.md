# Tight-path cost of strided KeyFramePlanes (2026-11-03)

Snapshot, not a live claim. Host r7900x (Ryzen 9 7900X, Zen 4), **not measured on
Zen 5**. No `-C target-cpu=native`, release profile with `debug = 1`, one encode
thread. Base `10be4beb`, change `930042e6`. RSS not measured.

Harness: a throwaway binary calling `encode_key_frame` once per iteration on a synthetic
8-bit 4:2:0 frame, tight planes, `allintra_speed0` with `cpu_used = 9`, cq 32. One process
per side, interleaved, first call of each process dropped. Two repetitions.

| size | n per side | median ms base | median ms change | ratio | instructions base | instructions change | delta |
|---|---|---|---|---|---|---|---|
| 64x64 | 420 | 0.490 / 0.446 | 0.486 / 0.450 | 0.992 / 1.008 | 132,860,431 | 132,911,550 | +0.04% |
| 512x512 | 60 | 23.793 / 23.758 | 23.527 / 23.597 | 0.989 / 0.993 | 1,864,722,290 | 1,864,813,713 | +0.005% |
| 1024x1024 | 18 | 100.075 / 100.268 | 99.437 / 99.205 | 0.994 / 0.989 | 4,385,426,006 | 4,385,537,533 | +0.003% |

Instructions are over 21, 5 and 3 calls. The added work is a plane-length check and a
per-row range scan. Output bytes are not compared by this harness; the differential gates
and `tests/all/strided_planes.rs` cover them.
