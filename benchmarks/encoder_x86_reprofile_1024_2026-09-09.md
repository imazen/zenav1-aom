# Encoder x86-64 re-profile at 1024x1024 — the ranked levers, corrected

**Cell:** `av1-1-b8-01-size-196x196` mirror-tiled to 1024x1024, cq27, `--cpu-used 0`,
bd8 4:2:0, CDEF off, loop-restoration on (the ALLINTRA defaults).
**Box:** x86-64 Linux, AVX-512 capable (`__memset_avx512_*` in both arms).
**Method:** `perf record -F 499 -g` on `examples/eprof_x86`, one recording per arm,
symbol shares converted to ms with each arm's own measured wall time.
Both arms emit the same 39,694-byte stream.

    port  10582.2 ms      C  4212.0 ms      ratio 2.513x      gap 6370 ms

**Why this record exists.** Playbook §14 (re-profile before trusting a ranked lever)
fired again, and this time it inverted the table rather than rescaling it. Two
landings this cycle (KB-PERF-7, KB-PERF-8) had moved the ranking, and the previous
ranked record is a **192x192** profile — an L2-resident cell, so its class shares do
not carry to a 1 MP frame.

## The finding: the port's LARGEST symbol is not a lever

`aom_dsp::txb::optimize::optimize_txb_core` is **11.44 %** of the port arm — 2.6x the
next symbol, and it had never been named as a lever. Like for like against C:

| | share | wall | ms |
|---|---:|---:|---:|
| port `optimize_txb_core` | 11.44 % | 10582 | **1211** |
| C `av1_optimize_txb` | 29.87 % | 4212 | **1258** |

**The port's coefficient trellis is 47 ms FASTER than libaom's**, and the whole
txb/trellis class runs at **ratio 1.32** — the best of any class. The 11.44 % headline
is entirely an artifact of the denominator: C spends a *larger* share of its own,
smaller encode there. Do not spend a landing on this symbol.

This is §14 with a new twist worth naming. Every previous §14 instance was a lever
costed too high (5x, 13x, 18x optimistic). This one is a lever that **does not exist
at all**, and nothing but a like-for-like comparison could have shown it: the symbol
is genuinely hot, genuinely the port's biggest, and genuinely at parity.

## The class table (self cost, ms/encode)

| class | port | C | ratio | gap | % of gap |
|---|---:|---:|---:|---:|---:|
| **transform** | 3188 | 749 | 4.26 | **+2440** | **38.3 %** |
| rd-driver | 1420 | 397 | 3.58 | +1023 | 16.1 % |
| loop-restoration | 1173 | 272 | 4.31 | +900 | 14.1 % |
| intra-pred | 970 | 457 | 2.13 | +514 | 8.1 % |
| memory (memset/memmove/malloc) | 528 | 43 | **12.41** | +486 | 7.6 % |
| txb/trellis | 2012 | 1529 | **1.32** | +483 | 7.6 % |
| quantize | 498 | 159 | 3.14 | +340 | 5.3 % |
| distortion | 464 | 169 | 2.74 | +295 | 4.6 % |
| entropy/pack | 70 | 10 | 6.91 | +60 | 0.9 % |
| other | 165 | 393 | 0.42 | -228 | -3.6 % |

Classification script and both flat symbol tables are in the session scratch; the
class regexes are per-arm (the two encoders share no symbol names).

## Transform, broken down — 38.3 % of the gap, and it is two different things

**Inverse** — port ~1275 ms (`run_inv1d_v3` 461, `inv_col_pass_core_v3` 408,
`inv_row_pass_core_v3` 250, driver 156) against C's `lowbd_inv_txfm2d_*` / `idct*` /
`iadst*` ~235 ms, i.e. **6.3x**. `run_inv1d`'s 461 ms **is** the kernels — they inline
into the dispatch switch — so there is no dispatch overhead to remove. This is the
i32x8-vs-i16x16 lane width plus libaom's size specialisation, and CLAUDE.md's clause-(4)
row already records that the lane-width half was BUILT and measured NULL at three frame
sizes, with a mechanism (the port's inverse kernels hold 17-bit transients in a
two-domain i32 representation, so narrowing storage does not narrow the arithmetic).
**That null stands; do not rebuild it.**

**Forward** — port ~1400 ms against C ~380 ms. Two items inside it are NOT the
specialisation programme:

* **~294 ms of SCALAR 1-D forward kernels** (`av1_fadst4` 65, `av1_fdct4` 54,
  `av1_fadst16` 47, `av1_fdct16` 46, `av1_fadst8` 42, `av1_fdct8` 41) — the port
  falling off its own SIMD path whenever a transform has a 4-length dimension.
  **This is what `KB-PERF-9` closes** (`encoder_fwd_col_4wide_2026-09-09.md`).
* **`get_fwd_txfm_cfg` 54 ms + `get_inv_txfm_cfg` 46 ms = 99 ms of pure config
  lookup with NO C counterpart** — libaom's whole-transform entry points are
  specialised per size at compile time, so it never pays this. ~1.6 % of the gap in
  two functions that are table lookups and a struct build. Not attempted here.

## Named, not attempted

* **memory at 12.41x, +486 ms.** KB-PERF-2 addressed encoder allocation churn on
  Darwin and warned explicitly that its rank does not survive a platform change
  (21 % of the win on Darwin, 86-99 % on Windows). **This is the first x86-64 Linux
  measurement of that class**, and it is 7.6 % of the gap. Attribution is only
  class-level: release builds omit frame pointers, so the inverted callgraph is too
  thin to name callers. Getting further needs a counting allocator
  (`examples/eprof_alloc.rs` is the existing instrument), not a profiler.
* **rd-driver +1023 ms (16.1 %)** — `search_tx_type_intra_into` 390,
  `txfm_rd_in_plane_intra` 317, `intra_model_rd_y` 305 against C's `search_tx_type`
  236 and `block_rd_txfm` 37. Unprofiled below the class level.
* **intra-pred +514 ms** — `z2_high` 196 (scalar; C's `av1_dr_prediction_z2_avx2` 90),
  `filter_intra_predict_high` 145 (scalar, **no vector path at all**; C's
  `av1_filter_intra_predictor_sse4_1` 24, a 6.1x ratio on a simple 7-tap filter),
  `highbd_filter_intra_edge` 87.

## Caveats

One box, one cell, one content class, one speed. The 192x192 record's own finding —
that the STAGE RANKING transfers across frame size but the RATIO does not — is why
this was re-taken at 1 MP; it has not been re-taken at another content class or speed,
and `--cpu-used 6` runs a structurally different pipeline (no loop restoration, no
full-RD partition search).
