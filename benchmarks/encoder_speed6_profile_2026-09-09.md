# The first speed-6 profile ever taken here — and the second overhead class has a name: the CNN

**2026-09-09.** Every ranked profile in this repo is `--cpu-used 0`. The
preceding record measured that the ratio worsens monotonically with speed
(2.19x at s0 to 5.29x at s9) and that this cycle's landings pay 4.6x less at s6
than at s0, and concluded there must be a **second overhead class** that
speed-0 profiling is structurally blind to. This is that profile, and the class
has a name.

## The cell

**1024x1024 cq27 `--cpu-used 6`**, x86-64, HEAD:

    port 461.83 ms    C 171.79 ms    ratio 2.688x    gap 290.04 ms

## The finding: `cnn_predict` is 7.68 % of the port and does not exist at speed 0

| | port | C |
|---|---:|---:|
| CNN | `cnn_predict` **7.68 %** = **35.47 ms** | `cnn_convolve_no_maxpool_padding_valid_5x5_avx2` 1.27 % + `av1_nn_predict_avx2` 0.72 % = **3.42 ms** |

**10.4x, +32.05 ms — 11.1 % of the whole speed-6 gap.** It is the
second-largest symbol in the port's speed-6 profile, behind only
`optimize_txb_core`.

**And it is 0 % of the speed-0 gap.** `intra_cnn_based_part_prune_level` is 0 at
speed 0 (`speed_features.c:387-388`, KB-23), so the intra-mode CNN never runs
there — which is exactly why every speed-0 profile in this repo, and therefore
every ranked lever table, is blind to it.

> **CORRECTION, same day — the clause-(4) half of this record does NOT hold at
> the preset zenavif ships.** `encoder_shipping_preset_1mp_2026-09-09.md`
> establishes that zenavif's default is `speed: 4` -> **`--cpu-used 3`**, and
> measures the CNN there at **40.9 ms vs C's 9.3 = +31.6 ms, which is 1.6 % of
> the 1949 ms speed-3 gap** — against the 11.1 % below. The CNN's ABSOLUTE cost
> is speed-invariant (~32 ms, computed once per 64x64 by KB-PERF-1's cache);
> only the DENOMINATOR moved. So the "one piece of work serves both" claim
> below should be read as: **it serves KB-41 root #27, and at the shipping
> preset it is not a ranked clause-(4) lever.** The ranked lever at s3 is
> transform, +789 ms, 40.5 % of the gap — the same item the speed-0 table names.

## The fix is already scoped, and it closes an open PARITY root too

The mechanism was measured by KB-PERF-1 and never acted on: the port's
`conv_valid` is a transcription of `av1_cnn_convolve_no_maxpool_padding_valid_c`
— **scalar** — while libaom dispatches the AVX2 variant. KB-PERF-1 measured
144.5 us per cascade run against libaom's dispatched 16.4 us (8.8x); this
profile measures 10.4x on the same comparison at a different cell and ISA,
which is consistent.

The C profile also settles the SCOPE, and it matches what KB-41 root #27
predicted from source: the only CNN convolve symbol libaom spends time in is
**`cnn_convolve_no_maxpool_padding_valid_5x5_avx2`**, i.e. the 5x5/skip-4 layer-0
specialisation. KB-41 root #27's own note — *"`cnn_avx2.c` is 532 lines but only
TWO specializations are reachable here — 5x5/skip-4 (layer 0) and 2x2/skip-2
(layers 1-3), everything else falls through to `_c`"* — is confirmed by
measurement rather than by reading.

**So one piece of work serves two open items at once:**

* **clause (4)** — 11.1 % of the speed-6 gap, invisible to every existing
  profile;
* **KB-41 root #27** — the last open carrier of the aom-rs datagen wave's 10
  photo + 3 screen cells, where the `_c`-vs-AVX2 CNN divergence in the 7th digit
  puts a branch logit on adjacent 1/512 quanta and flips `do_square_split`.

That root is currently blocked on exactly this port, and the parity half is the
HARDER one: perf only needs the AVX2 shape to be fast, parity needs it
bit-identical to the dispatched C.

## The rest of the speed-6 table, for whoever picks this up

Like-for-like, port ms against C ms at this cell:

| | port | C | note |
|---|---:|---:|---|
| `optimize_txb_core` / `av1_optimize_txb` | 37.1 | 34.8 | **1.07x — still near parity, as at speed 0. Not a lever.** |
| **CNN** | **35.5** | **3.4** | **10.4x — the finding** |
| `generate_hog` / `prune_intra_mode_with_hog` | 8.8 | 2.2 | ~4x |
| memory (`memmove` + `_int_malloc` + `memset`) | 21.2 | — | 4.6 % of the port |
| `intra_model_rd_y`, `txfm_rd_in_plane_intra` | 16.3, 11.8 | — | **inlining sinks — do not cost levers off these** (KB-PERF-10) |

## Limits

Single profiling run per arm (25 reps inside each), one cell, one box, one
content class. Shares are each arm's own wall. This is a ranking instrument, not
a landing-grade delta — the same status the speed-0 tables carry.
