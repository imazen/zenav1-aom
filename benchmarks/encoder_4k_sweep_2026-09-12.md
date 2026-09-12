# 3840x2160 (4K) encode sweep vs libaom — first 2160p timing, per-speed, with the 2026-09-11/12 improvements

**Measured 2026-09-12** on the perf branch (`perf/gate3-txfm-i16-batch`), x86-64 Linux,
box otherwise idle. Nothing prior had timed the port at 2160p: the clause-(4) headline is
1024x1024 and the per-speed table is 512x512 — this fills the large-frame cell the
fixed-overhead model predicts should be the port's *best* case.

## Method

- Content: `dump_cell_yuv 3840 2160` — the 196x196 photographic conformance cell
  mirror-tiled to 4K (the same recipe every HD gate and band uses). i420 bd8, cq27,
  ALLINTRA defaults (CDEF off, restoration on), single tile.
- Arms: `eprof_x86 <port|c> 3840 2160 27 <speed> <reps>` — the same binary drives the
  port's `encode_key_frame` (the path zenavif calls) and real libaom through the shim.
- Three columns per speed: **base** = `1434bc3` (branch point vs main, 2026-09-10 —
  pre-improvements), **new** = `772476a` (HEAD), **c** = libaom.
- reps: 2 timed (after 1 untimed warm) for s6-s9, 1 timed for s0-s5. Single-digit-round
  timings at 4K: read ratios as ±a few %, not band-grade. Bytes and first-diff are exact.
- Streams dumped by `dump_kf_stream` (new aom-bench example); decoded by `aomdec` and
  scored by `xtool score` (SSIMULACRA2 + butteraugli) against the source YUV.

## Times (ms) and ratios

| cpu-used | base port | **HEAD port** | libaom C | **HEAD/C** | base/C | base→HEAD |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 46,985 | 38,645 | 23,143 | **1.67x** | 2.03x | −17.8 % |
| 1 | 26,803 | 22,675 | 13,990 | **1.62x** | 1.92x | −15.4 % |
| 2 | 24,405 | 20,480 | 12,280 | **1.67x** | 1.99x | −16.1 % |
| 3 | 20,301 | 16,910 | 10,484 | **1.61x** | 1.94x | −16.7 % |
| 4 | 12,955 | 10,097 | 5,426 | **1.86x** | 2.39x | −22.1 % |
| 5 | 10,096 | 7,884 | 4,217 | **1.87x** | 2.39x | −21.9 % |
| 6 | 3,194 | 2,637 | 1,405 | **1.88x** | 2.27x | −17.5 % |
| 7 | 2,175 | 1,967 | 930 | **2.12x** | 2.34x | −9.6 % |
| 8 | 1,200 | 1,095 | 451 | **2.43x** | 2.66x | −8.7 % |
| 9 | 645 | 594 | 183 | **3.24x** | 3.52x | −7.8 % |

**Read:** the size trend the 512² table predicted holds — 4K is the port's best measured
regime. The shipping preset (s3) is **1.61x at 4K vs 1.94x at 1 MP**, and the quality
speeds s0-s3 sit at 1.6-1.7x — near the 1.5x bar, not met. The speed axis still worsens
monotonically; the improvements compress the mid speeds most (−22 % at s4/s5).

## Byte identity and stream size

| cpu-used | port bytes | C bytes | identical? |
|---:|---:|---:|---|
| 0 | 316,655 | 316,991 | **NO — port −336 B (−0.11 %), first diff byte 13** |
| 1-6 | matches | matches | **YES** (s1,s3 verified first_diff=-1; s2,s4,s5,s6 equal length, same class) |
| 7 | 372,521 | 355,137 | NO — port +4.9 %, first diff byte 13 |
| 8 | 379,634 | 363,971 | NO — port +4.3 % |
| 9 | 391,589 | 375,109 | NO — port +4.4 % |

**The s0-at-2160p divergence is NEW** — not in the pinned set, which covered
speed≥7 above ~3x3 superblocks at ≤1 MP. Byte 13 is inside the frame OBU header
region (same first-diff offset as s7-s9, but a different mechanism: s7-9 emit
*larger* streams, s0 *smaller*). Both arms decode cleanly to identical-dimension
frames. Candidate mechanism: a `is_4k_or_larger`-gated decision arm whose config
fix (KB-19) landed but whose search-side consequences diverge — localize by
decode-both before quoting.

## Quality at the divergent speeds (SSIMULACRA2 / butteraugli vs source)

| speed | arm | SSIM2 ↑ | BA max ↓ | BA_3N ↓ |
|---:|---|---:|---:|---:|
| 0 | port | 78.686 | 3.636 | 1.082 |
| 0 | C | 78.691 | 3.642 | 1.084 |
| 7 | port | 73.238 | 4.683 | **1.343** |
| 7 | C | 72.406 | **4.030** | 1.370 |
| 8 | port | **73.376** | **4.782** | **1.382** |
| 8 | C | 72.344 | 5.212 | 1.431 |
| 9 | port | **73.346** | **4.542** | **1.370** |
| 9 | C | 71.943 | 5.119 | 1.445 |

At s0 the streams are RD-equivalent for practical purposes (−0.005 SSIM2 at −0.11 %
bytes). At s7-s9 the port emits 4-5 % more bytes at *equal or better* SSIM2 — the
pinned nonrd/VAR_BASED_PARTITION divergence; quality is not the problem there.

## What this does not establish

- One content (mirror-tiled 196² — periodic, not a real 4K photo), one cq, one box,
  coarse rep counts. A real-4K-photo cell may shift both arms.
- No band/null protocol — these are single/double-rep readings; the 15-22 %
  improvements dwarf the ±0.3 % arm drift so the direction is safe, the decimals less so.
- The s0 byte divergence is unlocalized; treat it as a candidate new pin, not a
  confirmed mechanism.
