# Encoder hotspot RE-profile — where the 3.34x goes now (2026-08-02)

**The gap is DIFFUSE. Nothing is over 30 % of it, and the top three are
30.0 / 21.8 / 18.6 %.** That is a materially different programme from the one
[`encoder_hotspot_profile_2026-08-02.md`](encoder_hotspot_profile_2026-08-02.md)
published this morning, whose headline was "one function is 81.5 % of it".

That profile ranked its levers against a **520.93 ms** encode in which
`cnn_predict` was 74.7 % self. KB-PERF-1 (`6432e06`) cached the CNN per 64x64 as
C does; the encode is now **159.78 ms**. Every *share* in the old ranking is
therefore stale, and its levers 2-5 were explicitly arithmetic on the pre-fix
self-costs. This is the re-profile the old file asked for, run at the same cell,
with the same tooling, against the same libaom build.

Two other landings sit between the two profiles: `ab17489` (KB-34, the nonrd
non-square leaf walk — which also unblocked `cpu-used 9`, previously a refusal,
now measurable and the worst ratio in the breadth sweep) and `bc545df`.

Provenance, box, load, exact commands and the honest list of what is not
measured: [`encoder_hotspot_reprofile_2026-08-02.meta`](encoder_hotspot_reprofile_2026-08-02.meta).
Data: `.stages.tsv`, `.symbols.tsv`, `.control.tsv`, `.breadth.tsv`,
`.alloc_callers.tsv`, `.stability.tsv`.

**Nothing was optimized in this session.** Every ceiling below is arithmetic on
a measured self-cost, labelled as such.

> **Levers 3a and 3 have since LANDED, and lever 3's ceiling below was 18x
> optimistic** — `encoder_alloc_scratch_2026-08-02.md`. Measured 3.346x →
> 3.248x (−4.64 ms), allocator calls 854 053 → 512 557, zero bytes moved.
> 3a returned −3.13 ms against its +5.30 projection; lever 3 returned
> **−1.34 ms** against its +24.76. The reason is "Attribution limits" item 4
> below, which the ranked table at the bottom of this file contradicts: the
> `alloc/libc` stage is a **leaf class matched by symbol name**, and most of its
> mass is `_platform_memset` (5.36 % of the window) + `_platform_memmove`
> (2.68 %) rather than the allocator's own `xzm_free` (2.34 %). Scratch reuse
> removes malloc/free pairs; it does not remove the bytes those buffers still
> have to be zeroed with. **Split those symbols before crediting any lever with
> that stage total.**

---

## Control band — read this before reading any delta

Nine INDEPENDENT process invocations per arm, **interleaved** (port, C, port, C,
…) so background drift lands on both, each invocation 2 warm-up + 7 timed
encodes with its own median taken. 1024x1024 photo, cq 44, cpu-used 6.

| arm | median | min | max | spread | stdev | distinct byte counts |
|---|---|---|---|---|---|---|
| `zenav1-aom` | **159.78 ms** | 159.07 | 163.83 | **2.98 %** | 0.92 % | 1 |
| `libaom-c` | **47.80 ms** | 47.40 | 49.17 | **3.69 %** | 1.72 % | 1 |

**Ratio 3.343x**, and the nine PAIRED ratios are 3.25, 3.27, 3.29, 3.33, 3.35,
3.36, 3.37, 3.37, 3.37 — spread on the ratio itself **3.9 %**. The CNN-cache
study's 3.356x (`encoder_cnn_cache_2026-08-02.md`) **reproduces**, on a
different commit (`ab17489` vs that study's working tree at `5e29589`) and under
a different background load. Both arms emit the same 4472-byte stream, byte for
byte.

> The box carried a concurrent `zenmetrics` fleet throughout; whole-box load
> average 28-34 of 12 cores. That is why the control band is interleaved and why
> it is quoted first. A delta under ~4 % on the ratio is noise here.

**A second control, new here: the port arm was sampled TWICE**, independently,
60 s each. The two runs' stage shares agree within **5 % relative on every stage
above 2 % of the window** (worst: `alloc/libc` −9.9 %, 1.57 pp), and within
0.6 pp absolute on all but that one. `.stability.tsv` has all 19 rows. Shares
below ~1 % of the window swing 15-38 % relative and should not be read as
precise. The table below uses the second (clean) run; the first is the control.

---

## Method

Unchanged from the first profile, so the two are comparable:
`/usr/bin/sample`, 1 ms, 60 s window started 5 s in; 45 541 samples on the port
arm, 46 219 on libaom-c; `self(node) = count − Σ children` summed over every
call site; symbols classified to stages by name with the unmatched residual
printed (**0.88 %** port, **3.28 %** libaom-c); each side's self-share
multiplied by ITS OWN measured ms/encode, because a share is a share of a
different denominator on each arm.

The three consequences the first profile stated still hold and still bound what
may be concluded — shares are of elapsed time not instructions; **inlined frames
are not expanded**, so self cost is charged to the outermost non-inlined
function; libaom's `static` functions symbolicate to the nearest preceding
exported symbol because `libaom.a` carries no `-g`. See "Attribution limits"
below for which specific rows that damages.

The debug-info build used for symbolication produced a **byte-identical**
`.obu` and ran at 158.3-158.8 ms against the plain release build's 159.78 ms —
inside the control spread.

That the method reproduces is checkable independently: **libaom's own stage
table is the same today as it was this morning**, to a tenth of a point —
hog 24.92 % vs 24.98, dsp:transform 9.62 vs 9.67, cnn 9.36 vs 9.33, trellis
8.30 vs 8.25. Same binary, same cell, two sessions, two sampling runs.

---

## Where the 118 ms goes now

Self cost, both arms, in absolute ms/encode. Denominators are the medians
measured DURING the profile runs (port 166.02 ms, libaom-c 47.65 ms → 3.48x;
the clean interleaved control is 3.343x — `sample` suspends the target to walk
stacks, which inflates the port arm's during-profile median by ~4 %).

| stage | port ms | % | C ms | % | gap ms | % of gap | port/C |
|---|---:|---:|---:|---:|---:|---:|---:|
| **cnn-partition-prune** | **40.02** | 24.11 | 4.46 | 9.36 | **+35.56** | **30.0 %** | **8.97x** |
| **dsp:transform** | **30.40** | 18.31 | 4.59 | 9.62 | **+25.81** | **21.8 %** | **6.63x** |
| **alloc / libc** | **24.92** | 15.01 | 2.87 | 6.03 | **+22.05** | **18.6 %** | **8.68x** |
| dsp:intra-pred | 11.82 | 7.12 | 0.85 | 1.78 | +10.97 | 9.3 % | 13.95x |
| entropy / rate-model | 7.86 | 4.73 | 2.05 | 4.31 | +5.80 | 4.9 % | 3.83x |
| tx-search | 7.65 | 4.61 | 2.09 | 4.39 | +5.56 | 4.7 % | 3.66x |
| dsp:dist (sad/var/satd) | 7.78 | 4.69 | 2.71 | 5.68 | +5.07 | 4.3 % | 2.87x |
| intra-mode-rd | 7.26 | 4.37 | 3.69 | 7.74 | +3.57 | 3.0 % | 1.97x |
| dsp:quant | 4.47 | 2.69 | 2.50 | 5.24 | +1.98 | 1.7 % | 1.79x |
| os / setjmp *(see note)* | 2.71 | 1.63 | 0.81 | 1.69 | +1.90 | 1.6 % | 3.36x |
| encode-driver | 1.69 | 1.02 | 0.00 | 0.00 | +1.69 | 1.4 % | — |
| partition-search | 1.45 | 0.87 | 0.54 | 1.14 | +0.91 | 0.8 % | 2.67x |
| **hog-intra-prune** | 12.73 | 7.67 | 11.88 | 24.92 | **+0.85** | 0.7 % | **1.07x** |
| other | 1.46 | 0.88 | 1.56 | 3.28 | −0.10 | −0.1 % | 0.93x |
| postfilter-search | 0.11 | 0.07 | 0.41 | 0.86 | −0.30 | −0.3 % | 0.27x |
| **trellis (optimize_txb)** | 3.51 | 2.11 | 3.95 | 8.30 | **−0.45** | −0.4 % | **0.89x** |
| **pack / entropy-write** | 0.12 | 0.07 | 0.65 | 1.37 | **−0.53** | −0.4 % | **0.19x** |
| screen-content-detect | 0.00 | 0.00 | 2.01 | 4.21 | −2.01 | −1.7 % | — |
| **TOTAL** | **166.02** | 100 | **47.65** | 100 | **+118.37** | 100 % | **3.48x** |

Three corrections to that table, all from classifying by symbol name. Use the
corrected numbers, not the raw rows:

* **`os / setjmp` on the port arm is NOT setjmp — it is more allocator.** The
  stage rule matches `mach_`, and `mach_absolute_time` is 1.63 % of the port
  window. Attributed to its callers, **98.7 % of it is under `xzm_free`** —
  macOS's xzone allocator doing internal bookkeeping. On libaom the same stage
  IS genuine: 97.6 % of its `sigprocmask`/`sigaltstack` is under `setjmp`.
  **So the allocation class is really 24.92 + 2.71 = 27.63 ms port vs 2.87 ms C
  = 9.63x, +24.76 ms = 20.9 % of the gap** — and `os/setjmp` should be read as
  +0 for the port and −0.81 for C. The same correction applies retroactively to
  the first profile's table (its alloc gap was +24.05, really +26.71).
* **libaom's `av1_predict_intra_block` lands in `intra-mode-rd`** while the
  port's equivalent (`aom_dsp::intra::*`) lands in `dsp:intra-pred`, which
  inflates `dsp:intra-pred`'s 13.95x and deflates `intra-mode-rd`'s 1.97x.
  **Combined they are 19.08 ms port vs 4.53 ms C = 4.21x, +14.54 ms (12.3 % of
  the gap)** — use that number.
* **`screen-content-detect` is 2.01 ms of real libaom work the port never
  does**, because the harness hands it a parsed sequence header (the port never
  authors one). That flatters the port by 1.2 % of its wall. It is a property
  of the measurement boundary, not of the encoder.

Corrected, the gap ranks: **CNN 35.56 (30.0 %), transform 25.81 (21.8 %),
allocation 24.76 (20.9 %), intra pred+RD 14.54 (12.3 %)** — then a tail where
nothing exceeds 5.8 ms. Those four are 100.7 ms of 118.37 = **85 %**.

Three stages where **the port is still faster than libaom**, which bounds how
much of the residual is real work rather than a porting deficit: pack/entropy-
write 0.19x, trellis (`optimize_txb`) 0.89x, and the HOG intra-mode prune is a
near-tie at 1.07x — 12.73 ms vs 11.88 ms, on the single biggest symbol in
libaom's whole profile (`compute_gradient_info_sb`, 22.11 % of the C encode).

### Top symbols, both arms

| port symbol | self % | ms | | libaom symbol | self % | ms |
|---|---:|---:|---|---|---:|---:|
| `cnn_partition::cnn::cnn_predict` | 23.87 | 39.62 | | `compute_gradient_info_sb` | 22.11 | 10.54 |
| `transform::simd::___arcane_run_fwd1d_neo` | 7.83 | 13.00 | | `av1_cnn_convolve_no_maxpool_padding_valid_neon` | 7.12 | 3.39 |
| `hog::generate_hog` | 7.57 | 12.57 | | `av1_optimize_txb` | 7.10 | 3.38 |
| `platform_memset` | 5.36 | 8.90 | | `av1_cost_tokens_from_cdf` | 3.37 | 1.61 |
| `transform::simd::try_fwd_col_pass` | 3.54 | 5.88 | | `av1_quantize_fp_32x32_neon` | 2.99 | 1.43 |
| `platform_memmove` | 2.68 | 4.44 | | `prune_intra_mode_with_hog` | 2.81 | 1.34 |
| `quant::simd::av1_quantize_fp_no_qmatrix_dispatch` | 2.58 | 4.28 | | `search_tx_type` | 2.75 | 1.31 |
| `xzm_free` (allocator) | 2.34 | 3.88 | | `av1_set_screen_content_options` | 2.65 | 1.26 |
| `txb::optimize::optimize_txb_core` | 2.02 | 3.36 | | `av1_predict_intra_block` | 2.54 | 1.21 |
| `transform::simd::try_fwd_row_pass` | 1.83 | 3.03 | | `av1_rd_pick_intra_sby_mode` | 2.27 | 1.08 |

Both arms are now diffuse. The port's top symbol is 23.87 % and its top ten
spans six stages; libaom's top symbol is 22.11 % and its top ten spans seven.
**The port's profile has come to look like libaom's** — which is what "the gap
is no longer concentrated" means concretely.

---

## Finding 1 — how much of the old ranking survived: the CNN cache moved exactly one row

Old and new side by side. The *shares* all moved, because the denominator fell
by 345 ms. The **absolute gaps** are the honest comparison, and they are the
striking column: apart from the CNN itself, and one row that is the CNN's
call-site copy, **nothing moved by more than 0.5 ms**.

| stage | old gap ms | old % of gap | **new gap ms** | **new % of gap** | Δ gap ms |
|---|---:|---:|---:|---:|---:|
| cnn-partition-prune | +377.89 | 81.5 | **+35.56** | **30.0** | **−342.33** |
| dsp:transform | +23.81 | 5.1 | **+25.81** | **21.8** | +2.00 |
| alloc / libc | +24.05 | 5.2 | **+22.05** | **18.6** | −2.00 |
| dsp:intra-pred | +11.36 | 2.5 | +10.97 | 9.3 | −0.39 |
| entropy / rate-model | +5.49 | 1.2 | +5.80 | 4.9 | +0.31 |
| tx-search | +5.35 | 1.2 | +5.56 | 4.7 | +0.21 |
| dsp:dist (sad/var/satd) | +4.92 | 1.1 | +5.07 | 4.3 | +0.15 |
| **partition-search** | **+4.08** | 0.9 | **+0.91** | 0.8 | **−3.17** |
| intra-mode-rd | +3.76 | 0.8 | +3.57 | 3.0 | −0.19 |
| dsp:quant | +1.90 | 0.4 | +1.98 | 1.7 | +0.08 |
| os / setjmp | +1.89 | 0.4 | +1.90 | 1.6 | +0.01 |
| encode-driver | +1.88 | 0.4 | +1.69 | 1.4 | −0.19 |
| hog-intra-prune | +1.35 | 0.3 | +0.85 | 0.7 | −0.50 |
| trellis (optimize_txb) | −0.91 | −0.2 | −0.45 | −0.4 | +0.46 |
| pack / entropy-write | −0.49 | −0.1 | −0.53 | −0.4 | −0.04 |
| screen-content-detect | −2.08 | −0.4 | −2.01 | −1.7 | +0.07 |
| **TOTAL** | **+463.52** | 100 | **+118.37** | 100 | **−345.15** |

The two rows that moved are both the change under test, and they add up:
`−342.33` (CNN) `− 3.17` (partition-search) `= −345.50` against the measured
total move of **−345.15 ms**. That is a **0.35 ms** residual across sixteen
independently-sampled stages, and it is a strong statement that the cache
changed exactly what it was designed to change and nothing else.

The partition-search row is not a mystery either: `extract_intra_cnn_window`
lives in `aom_encode::partition_pick` and so classified there. It ran 2558 times
before and runs 256 times now (measured, below), and it was **3.70 ms** of that
stage in the old profile. 3.70 x (1 − 256/2558) = 3.33 ms predicted; 3.17 ms
measured.

**So: none of the old ranking's *ordering* survived below rank 1, and all of its
*absolute* numbers did.** Levers 2-5 of the old list were arithmetic on
self-costs that this profile confirms were, and remain, correct — they were
simply being quoted as fractions of a denominator that has since fallen by 3.2x.

---

## Finding 2 — the CNN is still rank 1, and now the whole of it is per-call cost

`cnn_predict` is **23.87 % self** of the port encode and **+35.56 ms** of the
+118.37 ms gap. But the *shape* of that gap has completely changed, and the two
factors the first profile separated can be re-checked independently.

**(a) Call count: 256 vs 256 — the redundancy is gone.** Re-counted exactly with
the same counting `GlobalAlloc` (`crates/aom-bench/examples/eprof_alloc.rs`):
all five of the CNN cascade's exact-size buffers report **n = 256, 1.00 per
64x64 superblock**, which is libaom's number. (Two of the seven watch sizes, 64
and 80 bytes, are shared with unrelated call sites and report 71 908 / 1 852;
the five load-bearing ones move in lockstep at 256.) The 4225-byte window copy
is 256 too.

**(b) Per-call cost: 8.57x, unchanged.** `eprof_cnn_bench`, same 65x65 window,
outputs compared before any timing is printed, median of 7 rounds x 2000
iterations:

| arm | ns/call | vs libaom NEON | first profile |
|---|---:|---:|---:|
| port — scalar Rust `conv_valid` | 146 171 | **8.57x** | 144 548 (8.80x) |
| libaom `av1_cnn_convolve_..._valid_c` — scalar C | 128 692 | 7.54x | 125 208 (7.63x) |
| libaom, dispatched (NEON) | **17 059** | 1.00x | 16 420 |

The Rust transcription is within **13.6 %** of the C it transcribes (was 15 %).
The per-call gap is entirely that libaom ships a NEON kernel here and the port
does not.

**(c) The two reconstruct the profile, again.** 256 x 146.17 µs = **37.42 ms**
predicted against the profile's **40.02 ms** (6.9 % apart — the profiler
additionally charges the call site's allocation and zeroing). On the C side
256 x 17.06 µs = **4.37 ms** against the profile's **4.46 ms** (2.1 % apart).
Two methods that share no machinery agree, at the new baseline as at the old.

---

## Finding 3 — the second and fourth levers are one structural thing, and it is the decoder's #1 lever

`dsp:transform` is **6.63x (+25.81 ms)** and intra prediction + mode RD is
**4.21x (+14.54 ms)**. Together **+40.35 ms = 34.1 % of the gap — larger than
the CNN.** Split finer, and cross-checked against both sources rather than
inferred from the profiler:

| | port ms | C ms | ratio | gap ms | % of gap |
|---|---:|---:|---:|---:|---:|
| forward transform | 23.44 | 3.80 | **6.16x** | +19.63 | 16.6 % |
| inverse transform | 6.96 | 0.81 | **8.55x** | +6.15 | 5.2 % |
| intra pred + mode RD | 19.08 | 4.53 | **4.21x** | +14.54 | 12.3 % |

**The forward transform runs at half libaom's lane width, and that is read from
the source, not from the profiler.** libaom's `fdct8x{4,8,16,32,64}_neon` /
`fadst8x{8,16}_neon` / `fidentity8x8_neon` all take `const int16x8_t *`
(`upstream/av1/encoder/arm/av1_fwd_txfm2d_neon.c:401-1476`) — 8 coefficients per
128-bit register. The port's `fwd_col_pass` / `fwd_row_pass`
(`crates/aom-dsp/src/transform/simd/mod.rs:574`, `:685`) are
`#[magetypes(define(i32x8), v3, neon, -scalar)]` and `widen16()` the i16 input
to i32 on load — 4 coefficients per register. **`lowbd16.rs` contains inverse
kernels only; there is no i16-lane forward path in the port at all.**

**The inverse transform is a hybrid, and the landed bd8 work reaches a fifth of
it.** `lowbd16::___arcane_run_inv1d_i16_neo` +
`lowbd16::___arcane_inv_row_pass_i16_core_neo` are **1.47 ms**; the wide i32
`___arcane_inv_col_pass_core_neo` + `inv1d_v3_gen::___arcane_av1_idct32_impl_neo`
+ `___arcane_run_inv1d_neo` + `___arcane_inv_row_pass_core_neo` are **5.49 ms**
— **79 % of the port's encode-side inverse transform still runs the wide path.**
`lowbd16.rs:69-76` names why: only the DCT family passed the i16 audit;
`iadst4/8/16` and `identity4/8/16/32` route to i32.

**Intra prediction is the same story in different kernels.** Every port symbol
is the highbd (u16) path — `intra::dir::z{1,2,3}_high` **3.49 ms** combined,
`intra::simd::smooth` 2.09, `intra::simd::paeth` 0.96,
`intra::edge::highbd_filter_intra_edge` 0.78,
`intra::build_non_directional_intra_high` 0.78, `intra::predict_highbd` 0.75 —
against libaom's u8 lowbd `av1_dr_prediction_z2_neon` **0.27 ms**,
`aom_smooth_predictor_32x32_neon` 0.13, `av1_filter_intra_edge_neon` 0.10,
`aom_paeth_predictor_32x32_neon` 0.10, `av1_dr_prediction_z1_neon` 0.09.

This is Finding 1 of
[`gate3_decode_profile_2026-07-19.md`](gate3_decode_profile_2026-07-19.md) —
"the port runs the highbd pipeline at every bit depth", there worth ~50 % of the
decode gap — showing up on the encode side at **34 % of the gap**. It was the
first profile's lever 3 at 8.4 % of a gap the CNN dominated. **It is now the
largest single programme in the encoder, and it is shared work with the decoder,
which has already built the audit machinery for the inverse half
(`xtask/audit_i16_safety.py`).**

---

## Finding 4 — the allocation lever grew a named, cheap sub-lever

**854 053 allocator calls** (241 781 `alloc` + 606 000 `alloc_zeroed` + 6 272
`realloc`) moving **448.8 MB**, at a peak live set of **27.7 MB**, for one
1024x1024 encode. That is **3 336 allocations per 64x64 superblock** and
**814 488 per megapixel**. The CNN cache took 16 114 calls and 110.9 MB out of
the first profile's 870 167 / 559.7 MB — which is the 16 100 / 111 MB that
profile projected, landing within 0.1 %.

Corrected for the `mach_absolute_time` misclassification above, the class is
**27.63 ms port vs 2.87 ms C — 9.63x, +24.76 ms, 20.9 % of the gap.**

Caller attribution of the whole allocator + `memset`/`memcpy` leaf class
(16.65 % of the port window; `.alloc_callers.tsv` has all rows):

| caller | % of the class | ms |
|---|---:|---:|
| `aom_dsp::transform::simd::try_fwd_col_pass` | 9.90 % | 2.74 |
| `aom_dsp::transform::simd::try_fwd_row_pass` | 9.28 % | 2.57 |
| `aom_dsp::transform::txfm2d::av1_fwd_txfm2d` | 7.62 % | 2.11 |
| `aom_encode::tx_search::intra_model_rd_y` | 6.79 % | 1.88 |
| `aom_encode::tx_search::txfm_rd_in_plane_intra` | 6.61 % | 1.83 |
| `aom_encode::tx_search::search_tx_type_intra` | 6.07 % | 1.68 |
| `aom_encode::xform_quant` | 5.87 % | 1.62 |
| `aom_encode::encode_intra::encode_intra_block_plane_y` | 5.54 % | 1.53 |

`cnn_predict` was 7.4 % of this class in the first profile and is **not in the
top eighteen** now.

**The top two callers are one four-line defect, and the repo already knows it.**
`fwd_col_pass` (`mod.rs:592`) and `fwd_row_pass` (`mod.rs:703`) each open with
`let mut tin = [i32x8::zero(t); 64]; let mut tout = [i32x8::zero(t); 64];` — a
flat **4 KiB memset per forward-transform call, at every transform size**,
including 4x4. The *inverse* passes in the same file are tiered {8, 16, 64}
(`mod.rs:430-443`, `:784-797`, `:994-1000`), and `lowbd16.rs:132` states why in
so many words: *"The vector scratch is TIERED by `row_n` for the same reason the
i32 pass tiers by `col_n`: a flat 64-entry array zero-init is a memset that
dominates the small transforms."* The forward passes never got the same
treatment. The profiler agrees with the source: the sample tree shows
`try_fwd_col_pass + 124 → _platform_memset` at `mod.rs:0`, i.e. the prologue.

Attributed cost of the two: **5.30 ms, 3.2 % of the port's wall, 4.5 % of the
gap** — for a change with the same shape as one already made three times in the
same file.

**Vec growth in inner loops: present, small, localized.**
`alloc::raw_vec::finish_grow` is 2.35 % of the class (**0.65 ms**), attributed
to `aom_encode` callers that want `with_capacity`. Same shape as the first
profile found, and still 1/38th of the allocation lever it sits inside.

**Bounds checks are still not a first-order term.** Same cleanest-possible test
as before, re-measured: `conv_valid`'s inner loop indexes three slices with
runtime indices and is **1.136x** the same algorithm in scalar C. `dsp:quant`
(very many very small calls, where per-call overhead shows first) is 1.79x, and
the HOG prune is 1.07x.

---

## Breadth — is the profile cell still representative?

Same two arms, back to back, across three axes. `cnn_calls` is the exact count
from the allocation census. `.breadth.tsv`. **Every cell is byte-identical
between port and libaom-c** (all 13 `port_bytes == c_bytes`).

| cell | port ms | C ms | ratio | *(pre-cache ratio)* | CNN/SB |
|---|---:|---:|---:|---:|---:|
| 256x256, cq44, cpu6 | 19.25 | 5.76 | **3.34x** | *13.88x* | 1.00 |
| 1024x1024, cq44, cpu6 | 159.36 | 47.29 | **3.37x** | *10.88x* | 1.00 |
| 2048x2048, cq44, cpu6 | 511.12 | 155.46 | **3.29x** | *9.85x* | 1.00 |
| 1024², cpu-used 9 | 22.86 | 4.05 | **5.64x** | *port refused* | 0 |
| 1024², cpu-used 8 | 38.24 | 11.33 | 3.38x | *3.45x* | 0 |
| 1024², cpu-used 7 | 100.80 | 38.30 | **2.63x** | *2.69x* | 0 |
| 1024², cpu-used 6 | 163.23 | 48.32 | 3.38x | *10.91x* | 1.00 |
| 1024², cpu-used 5 | 915.17 | 213.63 | 4.28x | *5.68x* | 1.00 |
| 1024², cpu-used 4 | 2180.69 | 281.01 | **7.76x** | *9.39x* | 2.00 |
| 1024², cpu-used 3 | 3056.46 | 441.60 | **6.92x** | *8.18x* | 2.00 |
| 1024², cq 10 | 214.76 | 74.22 | 2.89x | *15.23x* | 1.00 |
| 1024², cq 26 | 194.60 | 62.96 | 3.09x | *10.51x* | 1.00 |
| 1024², cq 58 | 130.13 | 37.54 | 3.47x | *10.92x* | 1.00 |

What it says:

* **The size and quantizer axes have flattened.** 3.29-3.47x across 256²→2048²
  and cq 26→58, where the pre-cache spread was 9.85-15.23x. cq 10 is now the
  *best* cell (2.89x) where it was the worst (15.23x).
* **`cpu-used 9` is newly reachable** (KB-34, `ab17489`) **and is now the worst
  ratio measured anywhere: 5.64x.** The port does 1 MP in 22.86 ms (43.7 MP/s,
  consistent with xbench's published 44.62 MP/s) while libaom does it in 4.05 ms
  (247 MP/s). The CNN never runs there — this is entirely the nonrd path, and
  **this profile does not decompose it** (only cpu-used 6 was profiled).
* **The deep-RD tier is still the big unprofiled gap:** cpu-used 4 at 7.76x and
  cpu-used 3 at 6.92x, with the CNN running twice per SB (the SB walk runs
  twice at those presets — the pre-cache data shows the identical doubling).
  **Do not carry the ranking below to cpu-used ≤ 5 or to cpu-used 9 without
  profiling there.**
* **cpu-used 7 at 2.63x is the negative control** and is unchanged from
  pre-cache (2.69x) — the CNN never ran there either way.

---

## Attribution limits — which rows this tooling cannot resolve

Stated per the brief, rather than guessed at:

1. **Per-call SIMD dispatch overhead is NOT separable.** `sample` does not
   expand inlines, so `___arcane_run_fwd1d_neo`'s 13.00 ms is kernel body, and
   `av1_quantize_fp_no_qmatrix_dispatch`'s 4.28 ms is the quantizer body inlined
   into the dispatcher — not token cost. Splitting it needs an instruction-count
   profile, which Apple Silicon has no valgrind for. The 1.07x on HOG and 1.79x
   on quant put a low ceiling on it regardless.
2. **`cnn_predict`'s 39.62 ms includes its inlined `conv_valid`** — which is
   why Finding 2 cross-checks it against a direct microbenchmark and an exact
   call count, neither of which uses the profiler.
3. **libaom's `static` functions symbolicate to the nearest preceding exported
   symbol** (`libaom.a` has no `-g`). Globals — nearly all of the top of its
   profile — are exact; `fdct8x32_col_neon` and friends are `AOM_FORCE_INLINE`
   statics whose attribution to the enclosing `lowbd_fwd_txfm2d_*_neon` is
   approximate. The *stage* total is unaffected (both land in `dsp:transform`).
4. **The `alloc/libc` and `os/setjmp` stages are leaf classes matched by name**,
   and the `mach_` rule mis-files the port's allocator bookkeeping — see the
   three corrections under the main table. Caller attribution
   (`.alloc_callers.tsv`) is where the actionable number lives.
5. **Shares below ~1 % of the window are not reliable to better than ±30 %**
   — the two independent port runs disagree by 15-38 % relative down there
   (`.stability.tsv`). Every ranked lever below is above 4 % of the window.
6. **IPC and cache behaviour are folded into every number.** No instruction
   count, so "the port executes more instructions" and "the port stalls more"
   are not separated anywhere in this document.

---

## Ranked next levers — by MEASURED share of the 118.37 ms gap

Ceilings are arithmetic on the measured self-costs above, applied in order:
**they are projections from measurement, not end-to-end measurements of a
change.** Each is quoted against the profiled 166.02 ms / 47.65 ms = 3.48x
baseline (control-band equivalent 3.343x).

| # | lever | measured gap | ceiling | resulting ratio | byte-identity risk |
|---|---|---:|---:|---:|---|
> **CROSS-PLATFORM SCOPING, added 2026-08-02 after the Windows study**
> (`benchmarks/winperf_windows_2026-08-02.md`). **This ranking is Darwin-only, and
> ordering by its shares is wrong for a cross-platform product.** Two corrections:
>
> 1. **Lever 1 is architecture-specific.** "NEON the CNN convolution" does nothing
>    on `windows-latest` x86-64 — where most Windows users are. It is the top
>    lever by Darwin share and NOT the top cross-platform lever. An AVX2 twin
>    would be separate work with its own byte-risk gate.
> 2. **Allocation is worth ~5x more on Windows and its rank inverts.** Lever 3
>    measured −0.49 % on Darwin vs **−2.38 %** (`windows-11-arm`) and **−2.54 %**
>    (`windows-latest`), i.e. 21 % of the combined landing on Darwin against
>    **86–99 %** on Windows. The allocator call COUNTS are identical on all three
>    platforms (835,638 → 488,750), so the platforms do identical work and the
>    entire difference is cost per call. The "16.4 % residual, not worth chasing"
>    verdict below is a Darwin verdict; on Windows that residual is likely the
>    largest remaining lever.
>
> The levers that are genuinely cross-platform are the **bd8/lowbd lane-width
> programme** (2, 4, 5 — i16 lanes help NEON and AVX2 alike; the re-profile puts
> the combined programme at **34.1 %**, larger than the CNN ever was) and any
> further **allocation** work. Rank by those for shipping decisions.

| **1** | **NEON the CNN convolution** | **+35.56 ms (30.0 %)** | −35.6 ms | 3.48x → **2.74x** | **REAL** — see below |
| **2** | **bd8 i16-lane FORWARD transform** | **+19.63 ms (16.6 %)** | −19.6 ms | → **2.33x** | large, structural; precedent exists |
| **3** | **Per-txb scratch reuse (kill the 854 k allocs)** | **+24.76 ms (20.9 %)** | −24.8 ms | → **1.81x** | mechanical; decoder has the pattern |
| 3a | *(sub-lever)* tier the two forward-pass scratch arrays | +5.30 ms (4.5 %) | −5.3 ms | −0.11x on its own | **low** — same change already made 3x in the file |
| **4** | **bd8 lowbd intra predictors** | **+14.54 ms (12.3 %)** | −14.5 ms | → **1.50x** | large, structural; same programme as 2 |
| 5 | widen the landed i16 inverse path (79 % still wide) | +6.15 ms (5.2 %) | −6.2 ms | → 1.37x | audit machinery exists (iadst/identity are the gap) |
| 6 | the tail — entropy/rate 5.80, tx-search 5.56, dist 5.07, quant 1.98, driver 1.69, partition 0.91, hog 0.85 | +21.86 ms total | −21.9 ms | → **0.91x** | diffuse; nothing above 4.9 % of the gap |

Read the bottom of that column with suspicion. Closing every *positive* gap does
not land at 1.00x but at **0.91x**, because the port already beats libaom on
four stages (trellis −0.45, pack −0.53, postfilter −0.30, `other` −0.10) and
because libaom spends 2.01 ms on screen-content detection the harness never asks
the port for. That is arithmetic, not a forecast, and it is the reason the
per-lever ceilings should be read as *shares of a gap* rather than as a
countdown to parity.

**Lever 1's risk, explicitly, because it is the top of the list.** The port's
CNN is bit-exact to libaom's `_c` variant; libaom ships its `_neon` variant at
runtime; and those two **differ** — re-measured today on this window, **906 of
1636 output floats differ, max |Δ| 5.278e-6**, identical to this morning's
figure. The frames still come out byte-identical, i.e. the CNN's threshold
comparisons are robust to ~5e-6 *on this corpus* — an observation, not a proof.
So the lever splits:

* **A bit-exact vectorization** (same float summation order, wider lanes) is
  byte-safe by construction but cannot necessarily reach 17 µs, because the
  summation order is what constrains it. Its ceiling is not measured.
* **Transcribing libaom's NEON kernel** reaches the measured 17 µs but adopts
  libaom's rounding, and would need the full byte-identity corpus as a gate plus
  an `AOM_FORCE_SCALAR` twin. `docs/LIBAOM_UPSTREAM_NOTES.md` catalogues this
  divergence class.

**Levers 2, 4 and 5 are one programme**, and it is the decoder's #1 lever:
+40.32 ms, **34.1 % of the gap**, versus the CNN's 30.0 %. If the question is
"which single body of work is worth most", the answer is the bd8 lowbd lane
path, not the CNN — and unlike the CNN it is bit-exactness-preserving in
principle (integer kernels with an audited i16 domain, which the repo already
has a tool for).

**Lever 3a is the cheapest thing on this page**: 4.5 % of the gap for the tiering
change the inverse passes in the same file already have.

### Realistic floor — a PROJECTION, not a measurement

The first profile predicted "remove the CNN and the port is ~3x libaom" and that
verified at 3.36x. The equivalent statement now:

* **Levers 1 + 3a + lever 2**, which is the credible near-term block:
  166.02 − 35.6 − 5.3 − 19.6 = **105.5 ms → 2.21x**. If lever 2 lands at only
  the half-credit its 2x-lane-width argument guarantees, rather than full parity
  with libaom's kernels, it is 166.02 − 35.6 − 5.3 − 9.8 = **115.3 ms → 2.42x**.
* **Levers 1-4 at their measured ceilings**: **1.50x**.
* **Everything measured, closed**: **0.91x** — see the caveat above.

The honest headline is the first bullet: **the port has three levers left worth
more than 15 % of the gap each and a fourth worth 12 %, and landing the three
credible near-term ones lands somewhere near 2.2-2.4x.** There is no second
`cnn_predict` in this profile. Anyone hoping for another single 3x is going to
be disappointed; the work from here is four separate programmes, each worth
0.2-0.7x on the ratio.

---

## What is NOT measured here

* **One image, one content class.** Photographic, 1024x1024 (plus 256²/2048² in
  breadth). No screen content — and xbench showed the screen class has a
  *different* problem (a 113 pp capability gap, an IntraBC search at 558x
  libaom's time). None of this profile speaks to that.
* **8-bit 4:2:0 only.** No 10-bit, no 4:4:4, no monochrome. Note the irony: the
  largest lever here is "run the 8-bit path at 8-bit", and the 10-bit behaviour
  of that lever is not measured at all.
* **cpu-used 0, 1, 2 were not measured**, and **cpu-used 3, 4, 5, 9 were not
  profiled** — only their wall ratios and CNN call counts are known. cpu-used 9
  (5.64x) and cpu-used 4 (7.76x) are both *worse* than the profiled cell and
  neither is decomposed.
* **No instruction-count profile.** No valgrind on Apple Silicon; IPC and cache
  effects are folded into every number.
* **No multi-threaded measurement.** Neither arm is threaded.
* **No change was made and nothing was re-measured after a change.** Every
  ceiling in the ranked table is arithmetic on measured self-costs.
* **The box was not idle** (a concurrent `zenmetrics` fleet, whole-box load
  28-34 of 12 cores). The control band was taken under that load, is quoted
  first, and the port arm was sampled twice as a stability control.
