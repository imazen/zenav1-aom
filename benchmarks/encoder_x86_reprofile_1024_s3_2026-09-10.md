# Encoder x86-64 re-profile at 1024x1024 — **at the SHIPPING PRESET**, and the ranking reorders

**Cell:** `av1-1-b8-01-size-196x196` mirror-tiled to 1024x1024, cq27,
**`--cpu-used 3`** — the preset zenavif actually selects (`encoder.rs:506`
`speed: 4`, `speed_to_cpu_used(s) = clamp(s,1,10)-1`) — bd8 4:2:0, CDEF off,
loop-restoration on. **Box:** x86-64 Linux, AVX-512 capable.
**Method:** flat `perf record -F 499` on `examples/eprof_x86`, one recording per
arm, symbol shares converted to ms with each arm's own measured wall time.
Both arms emit the same 40,237-byte stream.

    port  3261.4 ms      C  1543.6 ms      ratio 2.104x      gap 1718 ms

**Why this record exists.** Every ranked table in this repo is `--cpu-used 0`.
The ratio the standing goal is judged against is measured at the SHIPPING
preset, and playbook §14's own lesson — a ranking does not survive a change of
regime — applies to speed exactly as it applied to platform (KB-PERF-2) and to
frame size (the 192x192 → 1 MP move). It does not survive here either.

## The class table (self cost, ms/encode; 1715 of the 1718 ms gap covered)

| class | port | C | ratio | gap | % of gap | was, at s0 |
|---|---:|---:|---:|---:|---:|---:|
| **transform** | 707 | 254 | 2.78 | **+452** | **26.4 %** | 38.3 % |
| **intra-pred/rd** | 589 | 201 | 2.93 | **+388** | **22.6 %** | **8.1 %** |
| rd-driver | 481 | 220 | 2.19 | +261 | 15.2 % | 16.1 % |
| txb/trellis | 701 | 512 | **1.37** | +188 | 11.0 % | 7.6 % |
| memory | 155 | 23 | 6.65 | +132 | 7.7 % | 7.6 % |
| quantize | 178 | 54 | 3.27 | +124 | 7.2 % | 5.3 % |
| loop-restoration | 149 | 36 | 4.20 | +114 | 6.6 % | 14.1 % |
| distortion | 190 | 82 | 2.33 | +109 | 6.3 % | 4.6 % |
| postfilter | 57 | 11 | 5.31 | +47 | 2.7 % | — |
| entropy/pack | 4 | 5 | 0.86 | −1 | −0.0 % | 0.9 % |

**Two rows moved by more than a factor of two, in opposite directions.**

* **intra-pred/rd goes 8.1 % → 22.6 % of the gap and is now second.** It is not
  that the port got slower — it is that the intra search is a larger fraction of
  a speed-3 encode than of a speed-0 one, while the transform and
  loop-restoration work that dominates at speed 0 shrinks.
* **loop-restoration goes 14.1 % → 6.6 %.** libaom disables Wiener and SGR at
  `speed >= 5` and prunes them hard below that, so the stage KB-PERF-6/7/8 spent
  three landings on is a much smaller share of the preset that ships.
* **transform is still first at 26.4 %, but its ratio is down to 2.78** from
  4.26 — that is the eight fusion kernels (KB-PERF-23 through KB-PERF-31)
  showing up in the class table rather than only in the wall clock.

**`optimize_txb_core` re-checked at this preset and the earlier verdict holds:**
414 ms against C's `av1_optimize_txb` + `update_coeff_general` at 426 ms. The
port's trellis is still faster, the class still runs near parity (1.37), and it
is still not a lever — now confirmed at both speeds rather than one.

## Inside the new #2, like for like

| | port | C | ratio |
|---|---:|---:|---:|
| `z2_high` | 76.3 | `av1_dr_prediction_z2_avx2` 38.9 | 2.0 |
| `z3_high` + **`z3_high_scalar`** | 13.0 + **27.7** | `av1_dr_prediction_z3_avx2` 9.3 | **4.4** |
| `generate_hog` | **36.2** | `prune_intra_mode_with_hog` 5.2 | **7.0** |
| `cnn_predict` | 34.9 | the `cnn_convolve_*` family ~2 | ~17 |
| `highbd_filter_intra_edge` | 27.7 | `av1_filter_intra_edge_sse4_1` 6.8 | 4.1 |
| `cfl_predict_block` | 17.0 | (below the 0.5 % cut) | — |
| edge assembly (`assemble_dir_edges` + `assemble_nd_edges`) | 43.3 | `build_directional_and_filter_intra_predictors` + `build_non_directional_intra_predictors` 50.9 | **0.85** |

Three things worth carrying forward:

1. **The edge assembly is FASTER than C's** (0.85x). Do not spend a landing
   there — the same shape as the `optimize_txb_core` finding, one class over.
2. **`z3_high_scalar` is 27.7 ms of genuinely scalar work.** KB-PERF-4
   vectorized z1/z2/z3 behind a runtime tap bound plus a shape gate, and its own
   reach pins record z3 admitting 16 of 19 shapes with `up == 0` on 85.1 % of
   calls. So a quarter of the z3 work is landing on the declining side at this
   preset. **Whether that is the tap bound or the shape gate is UNMEASURED** —
   the reach counters exist and would answer it in one run.
3. **`cnn_predict` is the biggest single ratio in the class**, and it is
   PARITY work, not perf work: KB-41 root #27 requires a bit-exact `cnn_avx2.c`
   port before the port's CNN can be made fast without changing decisions.

## What this says about where the remaining gap is

The gap is **1718 ms** and no class holds more than 26 % of it. Reaching the
1.5x bar needs the port at ~2315 ms, i.e. **−946 ms, 55 % of the gap** — which
is not one class, let alone one kernel. The honest reading is that clause (4)
now needs breadth across at least the top four classes, and that the levers
inside each are 20-40 ms apiece (0.6-1.2 % of the encode) rather than the
2 % kernels this cycle opened with.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, x86-64. The class
assignment is by symbol name and the C arm has a ~129 ms long tail this
classifier leaves in `other`, so read the class ratios as ±5 % rather than
exact.
