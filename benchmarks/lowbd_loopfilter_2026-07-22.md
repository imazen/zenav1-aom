# bd8 lowbd deblock (loop filter) — lowbd port measurement, 2026-07-22

The loop-filter (deblock) family's entry in the second, parallel bd8 "lowbd"
(u8 pixel) decode pipeline. A native u8 deblock kernel + whole-frame walk
(`loop_filter_frame_u8`) replacing the bd8-via-highbd path (`loop_filter_frame`
run at `bd = 8`) the port takes today. Nothing here is projected or estimated —
exact callgrind Ir, deterministic.

## Byte-identity — PROVEN (kernel AND whole-frame, SIMD AND scalar-forced)

* `crates/aom-dsp/tests/loopfilter_lowbd_diff.rs` — the u8 deblock kernel, over
  both edge directions x all 4 widths {4,6,8,14} x the mask/flat/flat2
  pixel-window + threshold ranges (>200k checks/entry), is byte-identical to
  BOTH the REAL C lowbd kernels (`aom_lpf_*_c`) AND the port's u16 highbd path at
  `bd = 8`. Checked for the SIMD-dispatched entry AND the never-dispatched
  pure-scalar reference.
* `crates/aom-dsp/tests/lf_apply_diff.rs` — the whole-frame walk
  `loop_filter_frame_u8` reproduces the u16 port result (== C) byte-for-byte over
  the full randomized frame space (all sizes, subsamplings, strides, params, tx
  sizes) the u16 path is already swept on, at bd8.
* `crates/aom-dsp/tests/lpf_simd_diff.rs` — the u8 SIMD entry equals the u8 scalar
  reference at EVERY archmage token tier (the `simd == scalar, no slip` Gate-3
  gate; guards the per-tier lowering, e.g. the NEON const-0-shift class of bug).

Green on BOTH the default SIMD path and `AOM_FORCE_SCALAR=1`. The highbd path is
byte-untouched (all additive), so the bd10/bd12 + full conformance corpus cannot
regress (re-verified: `zenav1-aom-decode` 240 vectors + golden MD5 unchanged).

## Callgrind Ir — a genuine WIN (−5.6% whole-frame, −8.8% kernel)

Microbench: `lowbd_loopfilter_profile <u8|u16> 120` (`crates/aom-bench/src/bin/`),
a fixed 256x256 4:2:0 all-intra frame (BLOCK_16X16 blocks, rotating luma tx in
{4x4,8x8,16x16} so the walk fires luma widths {4,8,14} + chroma {4,6}; near-flat
+ noisy bands so both the `filter4` and wide flat/flat2 arms run), a nonzero luma
+ chroma filter level, run through the u8 vs the u16 (bd=8) whole-frame entry on
identical pixels. callgrind, release, no `-C target-cpu=native`, AVX2 (`_v3`)
tier. 121 frames each (120 loop + 1 shared byte-identity cross-check).

| metric (inclusive Ir / frame)          | u8 (lowbd) | u16 (highbd bd8) |        delta |
|----------------------------------------|-----------:|-----------------:|-------------:|
| `loop_filter_frame(_u8)` (whole frame) |  3,435,048 |        3,637,208 | **−202,160 (−5.56%)** |
| deblock SIMD kernel (`lpf_impl(_u8)`)  |  2,212,792 |        2,426,304 |   −213,512 (−8.80%) |

Runtime sinks identical (`sink=128` both sides) — a second, live byte-identity
check; the internal iter-0 u8-vs-u16 assertion also gates every profiled run.

**Reading (why this is a real win, unlike the transform's neutral safe-step):**
the deblock is a shallow kernel — its per-lane pixel load/store and the bias/
threshold/clamp arithmetic are a LARGE fraction of the work (the transform's win
was hidden behind the dominant i32 butterfly, which is why its u8-storage step
was Ir-neutral). Two effects compound here:

1. **u8 plane access** — the tap gather and result scatter touch half the plane
   bytes (u8 vs u16), fewer instructions per edge.
2. **`bd` folded to a constant** — the u16 path is called with `bd = 8` as a
   RUNTIME field (`buf.bd`), so it still executes the `<< (bd-8)` / `>> (bd-8)`
   bias/threshold shifts every edge; the u8 kernel hardcodes `shift = 0`
   (`bias = 0x80`, `lim = 128`, thresholds unshifted), eliminating that
   arithmetic.

The loop filter's SIMD width is fixed at 4 (the 4 edge positions of one
`aom_lpf_*` call), independent of pixel width — so there is NO i32->i16 lane
doubling to chase here (unlike the transform); the i32x4 lane math is shared
verbatim. The −5.6% is the whole win of the dedicated lowbd path, and callgrind
`--cache-sim=no` does NOT count the additional memory-bandwidth benefit of the
narrower plane (real, uncounted here).

## Wiring status — BLOCKED on the recon-plane u8 conversion (not a loopfilter item)

The −5.6% is realized in the DECODE path only once the recon plane is `u8`. The
decode driver still stores every recon plane as `Vec<u16>` at all bit depths
(`aom-decode/src/lib.rs` `TileKf.recon{,_u,_v}`); the `ReconPlanes`
(`LowBd(Vec<u8>)`/`HighBd(Vec<u16>)`) carrier + the `FrameDecode`-boundary widen
described in `aom_dsp::lowbd` are NOT yet on main (the foundation landed the
per-family kernel entries + the dispatch contract, additive, decode untouched).
That conversion is cross-cutting (every family reads/writes the recon plane), so
it is NOT owned by the loopfilter family. Until it lands, `apply_deblock`
(`aom-decode/src/frame.rs`) keeps calling the u16 `loop_filter_frame` (correct,
no regression); wiring it is a one-line `if bit_depth == 8` flip to
`loop_filter_frame_u8` once the recon plane is `u8`. The kernel + walk + the
proof + this measurement are the family's deliverable, mirroring how the
transform foundation was measured (isolated `lowbd_txfm_profile`, decode
untouched).

## Provenance

Commit: see `git log` for the landing SHA. Box: dedicated aom-rs container.
Corpus/refs: libaom v3.14.1 submodule pin `03087864`. Method: exact callgrind Ir,
`--cache-sim=no --branch-sim=no`, load-independent; measured, not extrapolated.
