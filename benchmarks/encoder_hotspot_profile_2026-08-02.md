# Encoder hotspot profile — where the 10.8x goes (2026-08-02)

> **Lever 1 has since LANDED** (same day, KB-PERF-1): the CNN is now cached per
> 64x64 as C does. Measured **10.66x -> 3.36x**, cascade runs 2558 -> 256, no
> byte moved — `encoder_cnn_cache_2026-08-02.md`. Everything below is the
> pre-fix state, and its ranked levers 2-5 are arithmetic on the pre-fix
> self-costs; re-profile before ranking them again.

**The gap is CONCENTRATED, not diffuse. One function is 81.5 % of it.**

`benchmarks/xbench_2026-08-01.md` established that the port has no coding
deficit against libaom on photographs — byte-identical at matched `cpu-used` on
9 of 10 images at all 14 quantizers — and that its entire photographic BD-rate
gap is the +11.64 pp cost of not being able to afford a slower preset, because
it runs **10.8x slower than libaom at matched preset**. Nobody had ever profiled
the encoder; every profiling artifact in this directory before today is
decode-side. This is that profile.

Provenance, commits, box, exact commands and the honest list of what is not
measured: [`encoder_hotspot_profile_2026-08-02.meta`](encoder_hotspot_profile_2026-08-02.meta).
Data: `.stages.tsv`, `.symbols.tsv`, `.control.tsv`, `.breadth.tsv`,
`.cnn_share.tsv`, `.alloc_callers.tsv`.

---

## Control band — read this before reading any delta

Nine INDEPENDENT process invocations per arm, **interleaved** (port, C, port, C,
…) so background drift lands on both, each invocation 2 warm-up + 7 timed
encodes with its own median taken. 1024x1024 photo, cq 44, cpu-used 6.

| arm | median | min | max | spread | stdev | distinct byte counts |
|---|---|---|---|---|---|---|
| `zenav1-aom` | **506.17 ms** | 500.37 | 514.96 | **2.88 %** | 1.11 % | 1 |
| `libaom-c` | **47.20 ms** | 46.42 | 48.81 | **5.06 %** | 2.08 % | 1 |

**Ratio 10.72x**, and the nine PAIRED ratios are 10.25, 10.26, 10.53, 10.65,
10.77, 10.79, 10.81, 10.90, 10.91 — so the spread on the ratio itself is 6.4 %.
The xbench study's 10.8x (492.76 ms vs 45.72 ms) **reproduces**, on a different
commit (`0953fa7` vs `ea3bed3`/`7b560c6`) and on a box that was NOT idle. Both
arms emit the same 4472-byte stream, byte for byte, at this cell.

> The box carried a concurrent `zenmetrics` fleet at ~312 % CPU of 1200 %
> throughout. That is why the control band is interleaved and why it is quoted
> first. A delta under ~6 % on the ratio is noise here.

---

## Method, and what this tool can and cannot tell you

`valgrind`/`callgrind` — the deterministic instruction-count method the
decode-side profiles in this directory used
([`gate3_decode_profile_2026-07-19.md`](gate3_decode_profile_2026-07-19.md)) —
is not available on Apple Silicon. This is a **wall-clock sampling** profile
(`/usr/bin/sample`, 1 ms, 60 s window, started 5 s in so the `.yuv` read, the C
bootstrap and the warm-up encode are outside it): ~45 800 samples on the port
arm, ~46 300 on the libaom-c arm.

Three consequences, stated because they change what you may conclude:

1. **Shares are shares of elapsed time, not of instructions.** IPC and cache
   behaviour are not separated from instruction count here, the way the
   decode-side gate3 series separates them.
2. **Inlined frames are not expanded.** Self cost is charged to the outermost
   non-inlined function — so `cnn_predict`'s self includes its inlined
   `conv_valid`, and `av1_quantize_fp_no_qmatrix_dispatch`'s self is the kernel
   body inlined into the dispatcher, not dispatch overhead. Where that mattered,
   the finding is cross-checked by measurements that do not use the profiler at
   all (an exact call count and a direct microbenchmark).
3. **Both sides are measured, then converted to absolute milliseconds.** A share
   is a share of a different denominator on each arm (the port's encode is 10x
   longer), so `stage X is 5 % here and 10 % there` means nothing on its own.
   Each side's self-share is multiplied by ITS OWN measured ms/encode. Symbols
   are classified to stages by name and the unmatched residual is printed:
   **0.23 %** on the port, **3.28 %** on libaom-c.

The debug-info build used for symbolication is not a different encoder: it
produced a **byte-identical** `.obu` and ran at 514.3 ms against the plain
release build's 506.2 ms — inside the control spread. The libaom-c arm's
sampled window is **99.85 % timed encode** (measured: 200 reps, `sum(NS)`
9.556 s vs 9.57 s wall), so no correction is applied for its untimed
`enc_init`/`destroy`.

---

## Where the 463 ms goes

Self cost, both arms, in absolute ms/encode. Denominators are the medians
measured DURING the profile runs (port 511.69 ms, libaom-c 48.171 ms → 10.62x).

| stage | port ms | % | C ms | % | gap ms | % of gap | port/C |
|---|---:|---:|---:|---:|---:|---:|---:|
| **cnn-partition-prune** | **382.39** | 74.73 | **4.49** | 9.33 | **+377.89** | **81.5 %** | **85.1x** |
| alloc / libc | 26.95 | 5.27 | 2.89 | 6.00 | +24.05 | 5.2 % | 9.3x |
| dsp:transform | 28.47 | 5.56 | 4.66 | 9.67 | +23.81 | 5.1 % | 6.1x |
| dsp:intra-pred | 12.16 | 2.38 | 0.81 | 1.67 | +11.36 | 2.5 % | 15.1x |
| entropy / rate-model | 7.52 | 1.47 | 2.03 | 4.22 | +5.49 | 1.2 % | 3.7x |
| tx-search | 7.48 | 1.46 | 2.13 | 4.43 | +5.35 | 1.2 % | 3.5x |
| dsp:dist (sad/var/satd) | 7.73 | 1.51 | 2.81 | 5.83 | +4.92 | 1.1 % | 2.8x |
| partition-search | 4.62 | 0.90 | 0.53 | 1.11 | +4.08 | 0.9 % | 8.7x |
| intra-mode-rd | 7.56 | 1.48 | 3.80 | 7.89 | +3.76 | 0.8 % | 2.0x |
| dsp:quant | 4.36 | 0.85 | 2.46 | 5.11 | +1.90 | 0.4 % | 1.8x |
| os / setjmp | 2.65 | 0.52 | 0.76 | 1.58 | +1.89 | 0.4 % | 3.5x |
| encode-driver | 1.88 | 0.37 | 0.00 | 0.00 | +1.88 | 0.4 % | — |
| **hog-intra-prune** | 13.38 | 2.62 | 12.03 | 24.98 | **+1.35** | 0.3 % | **1.1x** |
| other | 1.19 | 0.23 | 1.58 | 3.28 | −0.40 | −0.1 % | 0.8x |
| pack / entropy-write | 0.16 | 0.03 | 0.65 | 1.35 | **−0.49** | −0.1 % | **0.2x** |
| **trellis (optimize_txb)** | 3.06 | 0.60 | 3.97 | 8.25 | **−0.91** | −0.2 % | **0.8x** |
| postfilter-search | 0.06 | 0.01 | 0.44 | 0.91 | −0.38 | −0.1 % | 0.1x |
| screen-content-detect | 0.00 | 0.00 | 2.08 | 4.33 | −2.08 | −0.4 % | — |
| **TOTAL** | **511.69** | 100 | **48.17** | 100 | **+463.52** | 100 % | **10.62x** |

Two boundary caveats on that table, both from classifying by symbol name:

* libaom's `av1_predict_intra_block` lands in `intra-mode-rd` while the port's
  equivalent (`aom_dsp::intra::*`) lands in `dsp:intra-pred`, which inflates
  `dsp:intra-pred`'s 15.1x and deflates `intra-mode-rd`'s 2.0x. **Combined they
  are 19.72 ms port vs 4.61 ms C = 4.28x, +15.11 ms** — use that number.
* `screen-content-detect` is 2.08 ms of real libaom work the port never does,
  because the harness hands it a parsed sequence header (the port never authors
  one). That flatters the port by 0.45 % of its wall. It is a property of the
  measurement boundary, not of the encoder.

Three stages where **the port is already faster than libaom**, which bounds how
much of the residual is real work rather than a porting deficit: trellis
(`optimize_txb`) 0.77x, pack/entropy-write 0.24x, and the HOG intra-mode prune
is a near-tie at 1.11x — 13.38 ms vs 12.03 ms, on the single biggest symbol in
libaom's whole profile (`compute_gradient_info_sb`, 22.11 % of the C encode).

### The same thing at symbol level, with libaom's counterpart beside each

Top ten port self-costs, each paired by hand with the libaom symbol(s) doing the
same job. Full lists for both arms in `.symbols.tsv`.

| port symbol | self % | ms | libaom counterpart | self % | ms | port/C |
|---|---:|---:|---|---:|---:|---:|
| `cnn_partition::cnn::cnn_predict` | 74.66 | 382.01 | `av1_cnn_convolve_*_{neon,2x2_neon,c}` + `av1_nn_predict_neon` | 8.81 | 4.25 | **89.97x** |
| `hog::generate_hog` | 2.58 | 13.22 | `compute_gradient_info_sb` | 22.11 | 10.65 | 1.24x |
| `transform::simd::___arcane_run_fwd1d_neo` | 2.40 | 12.28 | `fdct8x32_{row,col}_neon`, `fdct8x16_*`, `fadst8x16_*` | 5.59 | 2.69 | 4.56x |
| `platform_memset` | 1.86 | 9.54 | `platform_memset` | 2.25 | 1.08 | 8.81x |
| `transform::simd::try_fwd_col_pass` | 1.07 | 5.49 | `lowbd_fwd_txfm2d_{32,16,8}x*_neon` | 1.56 | 0.75 | 7.33x |
| `platform_memmove` | 0.89 | 4.54 | `platform_memmove` | 1.67 | 0.80 | 5.66x |
| `xzm_free` (allocator) | 0.81 | 4.13 | `xzm_free` | 0.41 | 0.20 | 20.98x |
| `quant::simd::av1_quantize_fp_no_qmatrix_dispatch` | 0.81 | 4.15 | `av1_quantize_fp_32x32_neon` + `av1_quantize_fp_neon` | 4.72 | 2.27 | 1.82x |
| `partition_pick::extract_intra_cnn_window` | 0.72 | 3.70 | **none — C passes `src.buf - stride - 1` in place** | 0 | 0 | — |
| `txb::optimize::optimize_txb_core` | 0.57 | 2.94 | `av1_optimize_txb` | 7.15 | 3.44 | **0.85x** |

(The 89.97x on row 1 is the strict symbol-to-symbol pairing; the 85.1x in the
stage table additionally charges libaom's per-node branch-DNN work to the same
stage. Both are reported rather than one being picked.)

Read that column of libaom percentages: **nothing in libaom's profile is above
22 %, and its top ten spans seven different stages.** libaom's encode is diffuse;
the port's is one function plus a long thin tail. That is the whole answer to
"concentrated or diffuse" — the *gap* is concentrated because the port has one
pathology, not because the encoder has one bottleneck.

---

## Finding 1 — the port re-runs the intra-mode CNN 10x per superblock; libaom runs it once

`aom_encode::cnn_partition::cnn::cnn_predict` is **74.66 % self of the entire
port encode**, in one symbol. Its gap to libaom is +377.89 ms of the +463.52 ms
total. Everything else on the list put together is 85.6 ms.

The gap factors cleanly into two independent things, both measured directly.

### (a) Call count: 2558 vs 256 — a 9.99x redundant recomputation

libaom computes the CNN **once per 64x64 node and caches it**
(`av1/encoder/partition_strategy.c:160` `if (bsize == BLOCK_64X64 &&
!part_info->cnn_output_valid)`, invalidated per 64x64 at
`partition_search.c:3342`, and every smaller node bails at
`if (!part_info->cnn_output_valid) return;` on :227 and runs only the small
branch DNN).

The port's `partition_pick.rs:2805` calls
`cnn_partition::decision::predict_decision`, which at `decision.rs:232` calls
`cnn::cnn_predict` **unconditionally**, at every node with `bsize_idx != 0` — i.e. at 64x64, 32x32,
16x16 and 8x8. `extract_intra_cnn_window` (`partition_pick.rs:2493`) snaps its origin to the
containing 64x64 (`(mi_row / SB64_MIB) * SB64_MIB * 4`), so **every one of those calls
convolves the identical 65x65 window and produces the identical 1636-float
buffer.** The result is byte-correct; the work is thrown away.

Counted exactly, not inferred, by a counting `GlobalAlloc` with exact-size watch
counters (`crates/aom-bench/examples/eprof_alloc.rs` — each of the CNN cascade's
five distinct buffer sizes has exactly one call site, and all five move in
lockstep, which is what proves the attribution):

| | port | libaom | ratio |
|---|---:|---:|---:|
| CNN cascade runs per 1 MP frame | **2558** | **256** | **9.99x** |
| per 64x64 superblock | 9.99 | 1.00 | |

### (b) Per-call cost: the port is 8.8x libaom's dispatched kernel — and 1.15x its scalar one

`crates/aom-bench/examples/eprof_cnn_bench.rs`, same 65x65 window, outputs
compared before any timing is printed, median of 7 rounds x 2000 iterations
(per-round spread 5.5-6.0 %):

| arm | ns/call | vs libaom NEON |
|---|---:|---:|
| port — scalar Rust `conv_valid` | 144 548 | **8.80x** |
| libaom `av1_cnn_convolve_..._valid_c` — scalar C | 125 208 | 7.63x |
| libaom, dispatched (NEON) — what libaom actually runs | **16 420** | 1.00x |

So **the Rust transcription is within 15 % of the C it transcribes**; the
per-call gap is entirely that libaom ships a NEON kernel here and the port does
not. (The port is bit-identical to the `_c` oracle, as designed.)

### (c) The two factors multiply, and they reconstruct the profile

9.99x x 8.80x = **87.9x** predicted; the profile measures **85.1x** at stage
level and **89.97x** symbol-to-symbol. Independently:
2558 x 144.5 µs = **369.8 ms** predicted against the profile's **382.4 ms**
(3.3 % apart — the profiler additionally charges the call site's allocation and
zeroing), and 256 x 16.42 µs = **4.20 ms** against the profile's **4.49 ms**
(the C number also carries `av1_nn_predict_neon`, which runs per node rather
than per superblock). Two methods that share no machinery agree.

---

## Finding 2 — take the CNN out and the port is ~3x libaom, not ~10x

Arithmetic on the table: port 511.69 − 382.39 = **129.30 ms** of non-CNN work
against libaom's 48.17 − 4.49 = **43.68 ms** → **2.96x**.

That is not just arithmetic. At `cpu-used` 7 and 8 the port switches to
`VAR_BASED_PARTITION` and never enters `rd_pick_partition_real`, so the CNN
prune never runs — measured `cnn_calls = 0` at both. The measured wall ratios
there are **2.69x** and **3.45x**, bracketing the 2.96x predicted from the
profile. Two independent routes to the same number.

**So the residual, after the one hotspot, is genuinely diffuse — and modest.**
Of the 85.6 ms residual gap: allocation 24.05 ms (28 %), forward/inverse
transform 23.81 ms (28 %), intra prediction + mode RD 15.11 ms (18 %), then a
tail where nothing exceeds 5.5 ms.

---

## Finding 3 — the cheap structural suspects, checked before any kernel claim

The brief's checklist, each answered with a measurement rather than an
inspection.

**Redundant recomputation across the search: YES, and it is the whole story.**
Finding 1. This is the encode-side analogue of the `nsys` api_sum lesson: the
hot kernel is within 15 % of the C it transcribes, so "the Rust loop is slow" is
NOT the finding — the findings are that libaom vectorizes this kernel and that
the port calls it ten times too often. Either alone removes ~90 % of the stage
(caching only: 382.4 → 38.3 ms; SIMD only: 382.4 → 42.0 ms), and only both
together reach libaom's 4.5 ms. Caching is free and byte-safe; SIMD is neither
(see lever 2). Anyone who had opened this file and gone straight to hand-tuning
`conv_valid` would have spent the week on the second-best half of the problem.

**Per-block heap allocation: YES, large, and independent of the CNN.** One
1024x1024 encode makes **870 167 allocator calls** (244 083 `alloc` + 619 812
`alloc_zeroed` + 6 272 `realloc`) moving **559.7 MB**, at a peak live set of only
27.7 MB. That is **3 399 allocations per 64x64 superblock** and **829 856 per
megapixel**. The CNN accounts for 17 906 of those calls (2.1 %) and 123 MB
(22 %); the rest is per-transform-block. Caller attribution of the whole
allocator + `memset`/`memcpy` leaf class (5.79 % of the window):

| caller | % of the class |
|---|---:|
| `aom_dsp::transform::simd::try_fwd_col_pass` | 8.5 % |
| `aom_dsp::transform::simd::try_fwd_row_pass` | 8.4 % |
| `aom_encode::cnn_partition::cnn::cnn_predict` | 7.4 % |
| `aom_encode::tx_search::txfm_rd_in_plane_intra` | 7.3 % |
| `aom_encode::xform_quant` | 6.4 % |
| `aom_encode::tx_search::intra_model_rd_y` | 6.2 % |
| `aom_dsp::transform::txfm2d::av1_fwd_txfm2d` | 5.9 % |
| `aom_encode::encode_intra::encode_intra_block_plane_y` | 5.8 % |

libaom's own allocator class is a nearly identical **share** (5.86 %) but 2.89 ms
absolute against the port's 26.95 ms — **9.3x**. Its callers are mostly
in-kernel `memset` (`av1_quantize_fp_32x32_neon`, 21 % of its class) plus a
handful of real per-superblock allocations (`av1_alloc_pmc` / `av1_free_pmc`,
the pc_tree). This is the same shape the decode profile found in 2026-07 and
fixed with `ReconScratch` / `InvTxfmScratch`; **the encode side has no
equivalent.**

**Bounds checks in hot indexing: measured, and NOT a first-order term.** The
cleanest possible test is in the data already: `conv_valid`'s inner loop indexes
three slices with runtime-computed indices and is **1.15x** the same algorithm
written in scalar C. Whatever bounds checking survives there costs ≤15 % of that
loop. The same conclusion from a second angle: `dsp:quant` (very many very small
calls, where per-call overhead would show first) is 1.8x, and the HOG prune is
1.1x.

**`Vec::push` growth in inner loops: present, small, and localized.**
`alloc::raw_vec::finish_grow` is 0.23 % of the window inclusive (≈1.15 ms) —
real, but 1/21st of the allocation lever it sits inside. Attributed to its
callers, it is three `Vec`s that want `with_capacity`:
`aom_encode::encode_intra::TxbEncode` (16.5 % of the class),
`aom_encode::tx_search::TxbWinner` (13.6 %), and one anonymous
`Vec<i64>` grow-one under `aom_encode` (42.7 %). Same shape as the
`DecodedBlockKf::txbs` growth the decode profile logged as its lever 3.

**Copies C does in place: YES, and the clearest one is next to the hotspot.**
`extract_intra_cnn_window` materializes a fresh `vec![0u8; 65*65]` and copies
4225 clamped pixels into it, 2558 times a frame — **3.70 ms + 2558 allocations**.
libaom hands the CNN `x->plane[0].src.buf - stride - 1` and a stride, and copies
nothing (`partition_strategy.c:215`). Its counterpart cost is **zero**; there is
no such symbol in its profile. More broadly: `platform_memset` 9.54 ms port vs
1.08 ms C (8.8x), `platform_memmove` 4.54 ms vs 0.80 ms (5.7x) — both riding the
per-txb buffer churn above.

**Per-call SIMD dispatch overhead: NOT SEPARABLE with this tool, and bounded
small.** The archmage dispatch symbols carry their kernel bodies inlined, so
their self time is kernel work; splitting the token cost out needs an
instruction-count profile. The 1.1x on HOG and 1.8x on quant put a low ceiling
on it regardless.

---

## Finding 4 — the second lever is the same one the decoder has

`dsp:transform` is 6.1x (+23.81 ms) and intra prediction + mode RD is 4.28x
(+15.11 ms). Look at which symbols each side runs:

| | port | libaom |
|---|---|---|
| forward transform | `transform::simd::___arcane_run_fwd1d_neo`, `try_fwd_col_pass`, `try_fwd_row_pass` | `lowbd_fwd_txfm2d_{8,16,32}x*_neon`, `fdct8x32_{row,col}_neon`, `fadst8x16_*_neon` |
| intra prediction | `intra::build_non_directional_intra_high`, `predict_highbd`, `dir::z{1,2,3}_high` | `av1_dr_prediction_z2_neon`, `av1_filter_intra_edge_neon` |
| inverse transform | `___arcane_inv_col_pass_core_neo` | `av1_lowbd_inv_txfm2d_add_neon` |

Every libaom symbol there is a **lowbd** (8-bit-content, i16-lane) kernel; every
port symbol is the **highbd** (u16 buffer, wide lane) path. This is exactly
Finding 1 of `gate3_decode_profile_2026-07-19.md` — "the port runs the highbd
pipeline at every bit depth", there worth ~50 % of the decode gap — showing up
again on the encode side. **The bd8 lowbd lane path is a lever on both halves of
the codec.** The decode side has since landed part of it (`bd8_*` series); the
encode side runs the wide path.

---

## Breadth — is the profile cell representative?

Same two arms, back to back, across three axes. `cnn_calls` is the exact count
from the allocation census. `.breadth.tsv` / `.cnn_share.tsv`.

| cell | port ms | C ms | ratio | CNN calls | per SB | CNN share of gap |
|---|---:|---:|---:|---:|---:|---:|
| 256x256, cq44, cpu6 | 79.85 | 5.76 | **13.88x** | 422 | 26.4 | 82.0 % |
| 1024x1024, cq44, cpu6 | 496.32 | 45.62 | **10.88x** | 2558 | 9.99 | 81.1 % |
| 2048x2048, cq44, cpu6 | 1460.03 | 148.25 | **9.85x** | 7683 | 7.50 | 83.4 % |
| 1024², cpu-used 9 | *port refuses* | — | — | — | — | — |
| 1024², cpu-used 8 | 37.64 | 10.91 | **3.45x** | **0** | 0 | 0 % |
| 1024², cpu-used 7 | 101.02 | 37.56 | **2.69x** | **0** | 0 | 0 % |
| 1024², cpu-used 6 | 497.67 | 45.60 | 10.91x | 2558 | 9.99 | 80.9 % |
| 1024², cpu-used 5 | 1191.11 | 209.81 | 5.68x | 2019 | 7.89 | 29.3 % |
| 1024², cpu-used 4 | 2719.01 | 289.48 | 9.39x | 4040 | 15.78 | 23.9 % |
| 1024², cpu-used 3 | 3585.96 | 438.64 | 8.18x | 4036 | 15.77 | 18.4 % |
| 1024², cq 10 | 1164.13 | 76.44 | **15.23x** | 5731 | 22.39 | 75.8 % |
| 1024², cq 26 | 657.45 | 62.54 | 10.51x | 3400 | 13.28 | 81.9 % |
| 1024², cq 58 | 403.98 | 36.99 | 10.92x | 2134 | 8.34 | 82.9 % |

**Every cell here is byte-identical between port and libaom-c** (both `BYTES`
columns agree at all 12 measured cells), so these are genuinely the same encode
on both sides.

What it says:

* **The CNN is armed at every allintra `cpu-used >= 1`**
  (`partition_pick.rs:2759`: level 2 whenever `allintra && speed >= 1` on
  non-screen content), so this is not a quirk of one preset. Only `cpu-used 0`
  escapes it — and cpu-used 0 is outside the 1 MP/s budget for the port anyway.
* **The redundancy gets WORSE as the frame gets smaller** (26.4 CNN runs per SB
  at 256², 7.5 at 2048²) — the search descends relatively deeper — and worse as
  the quantizer drops (22.4 per SB at cq 10). The 15.2x at cq 10 is the worst
  ratio measured anywhere.
* **cpu-used 6, the port's qualifying mode, sits near the peak of the CNN's
  share of the gap** — which is exactly the mode the +15.9 % photographic
  BD-rate deficit is measured at.
* **At cpu-used 3-5 the CNN is only 18-29 % of the gap.** The deep-RD tier has a
  large non-CNN gap that this profile does NOT decompose (only cpu-used 6 was
  profiled). Do not carry the ranking below to cpu-used ≤ 5 without profiling
  there.

---

## Two incidental findings, recorded because they are load-bearing elsewhere

**1. `cpu-used 9` refuses this cell at `0953fa7`.** `drv-aom 1024 1024 44 9`
panics in `crates/aom-encode/src/nonrd_pickmode.rs:1135` with the KB-32 handoff
message ("nonrd estimate arm at non-square leaf bsize 4 …"). cpu-used 8 on the
same cell is fine. `xbench_2026-08-01.md` publishes a cpu-used 9 throughput
number for this exact cell (44.62 MP/s), measured at `ea3bed3`. Not diagnosed
here — flagged.

**2. libaom's own NEON CNN kernel is not bit-identical to libaom's own `_c`
kernel.** Measured on this window: **906 of 1636** output floats differ, max
|Δ| **5.28e-6** — the class of upstream ISA divergence catalogued in
`docs/LIBAOM_UPSTREAM_NOTES.md`. The port is bit-exact to the `_c` variant,
libaom ships the NEON one at runtime, and the frames still come out
byte-identical — i.e. the CNN's threshold comparisons happen to be robust to
~5e-6 on this corpus. That is an observation, not a proof, and it is the reason
lever 2 below carries a correctness gate that lever 1 does not.

---

## Ranked next levers — by MEASURED ceiling

Ceilings are arithmetic on the measured self-costs above, applied in order, and
labelled as such: **they are projections from measurement, not end-to-end
measurements of a change** (nothing was optimized in this session). Each is
quoted against the profiled 511.69 ms / 48.171 ms = 10.62x baseline.

| # | lever | measured ceiling | resulting ratio | risk |
|---|---|---:|---:|---|
| **1** | **Cache the CNN output per 64x64, as C does** | **−344 ms (67 % of wall)** | **10.62x → ~3.5x** | **none — byte-identical by construction** |
| 2 | NEON the CNN convolution (after 1) | −34 ms | ~3.5x → ~2.8x | **real**: not bit-neutral |
| 3 | bd8 lowbd lane path for fwd/inv transform + intra pred | −39 ms | ~2.8x → ~2.0x | large, structural |
| 4 | Per-transform-block scratch reuse (kill the 870 k allocs) | −22 ms | ~2.0x → ~1.5x | mechanical; decoder has the pattern |
| 5 | the tail (entropy/rate 5.5, tx-search 5.4, dist 4.9, partition 4.1, quant 1.9) | −22 ms total | ~1.5x → ~1.05x | diffuse; nothing above 1.2 % of the gap |

Read the bottom of that column with suspicion: closing *every* measured gap
lands at parity by definition, so "1.05x" is what the arithmetic says, not a
forecast. Levers 1-3 are the credible block; 4 and 5 are the accounting of
what would be left.

**Lever 1 in detail — this is the one to do.** Hoist the CNN cascade to the
64x64 node and memoize it exactly as `part_info->cnn_output_valid` does,
leaving `predict_decision`'s branch DNN per node. 2558 → 256 calls. The saving
is `382.39 x (1 − 256/2558) = 344.1 ms` from the profile, or
`(2558 − 256) x 144.5 µs = 332.8 ms` from the microbench — the two methods
bracket **3.48x–3.71x** as the post-fix ratio. Knock-on: 16 100 fewer allocator
calls and 111 MB less allocation+zeroing per frame (20 % of the port's total
allocation traffic), which is inside lever 4's ceiling, so do not count it
twice. **It cannot change a single output byte**: all 9.99 calls per superblock
already convolve the same window and return the same buffer, which is why the
five exact-size allocation counters move in perfect lockstep. A byte-identity
gate over the existing corpus is the whole verification.

**Lever 2's risk, explicitly.** The port's CNN is bit-exact to libaom's `_c`
variant, and libaom runs its NEON variant, and those two differ (Finding above).
Any SIMD rewrite of `conv_valid` must decide which it reproduces, and the
current byte-identity to libaom rests on threshold decisions being insensitive
to ~5e-6 — measured true on this corpus, not proven in general. Gate it on the
full byte-identity corpus before believing it, and expect it to be the kind of
change that needs an `AOM_FORCE_SCALAR` twin.

**Lever 3 is the decoder's #1 lever, again.** See Finding 4. It is the largest
remaining item after the CNN and it is shared work with the decode side.

Everything inside lever 5 is under 1.2 % of the gap each. **Do not start there.**

### What lever 1 alone would buy in the currency xbench measures

The 1 MP/s budget xbench qualifies on is a preset ladder, so the useful question
is which `cpu-used` the port could afford. Applying the measured per-call cost
(144.5 µs) to the measured per-cell CNN call count — both measured AT the cell in
question, so this is arithmetic on measurement, not extrapolation across an axis:

| cpu-used | measured now | with lever 1 |
|---|---|---|
| 6 | 498 ms = **2.11 MP/s** | 165 ms = **6.36 MP/s** |
| 5 | 1191 ms = **0.88 MP/s** *(fails the bar)* | 936 ms = **1.12 MP/s** *(clears it)* |
| 4 | 2719 ms = 0.39 MP/s | 2172 ms = 0.48 MP/s |

**One caching change would move the port's qualifying mode from `cpu-used 6` to
`cpu-used 5`** — one of the five preset steps that separate it from libaom's
`cpu-used 1`, whose total is worth the +11.64 pp that xbench identified as the
port's *entire* photographic BD-rate deficit. No coding decision changes, and
no output byte changes.

(How much of the +11.64 pp one step is worth was not measured — xbench published
BD-rate at the qualifying modes, not a per-preset ladder for the port. Do not
assume it is 11.64/5.)

---

## What is NOT measured here

* **One image, one content class.** Photographic, 1024x1024 (plus 256²/2048²
  in the breadth sweep). No screen content — and xbench showed the screen class
  has a *different* problem (a 113 pp capability gap, and an IntraBC search at
  558x libaom's time). None of this profile speaks to that.
* **8-bit 4:2:0 only.** No 10-bit, no 4:4:4, no monochrome.
* **cpu-used 0, 1, 2 were not measured at all**, and **cpu-used 3-5 were not
  profiled** — only their wall ratios and CNN call counts are known. The
  composition of the large non-CNN gap at the deep-RD tier is unknown.
* **No instruction-count profile.** No valgrind on Apple Silicon; IPC and cache
  effects are folded into every number above.
* **No multi-threaded measurement.** Neither arm is threaded.
* **No change was made and nothing was re-measured after a change.** Every
  ceiling in the ranked table is arithmetic on measured self-costs.
* **The box was not idle** (~312 % of 1200 % CPU held by another agent's fleet
  throughout). The control band was taken under that load and is quoted first.
