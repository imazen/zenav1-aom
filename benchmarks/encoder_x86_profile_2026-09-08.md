# The FIRST x86-64 profile of this encoder — 2026-09-08

**Every encoder perf number in this repo before today was taken on
aarch64-apple-darwin**, with macOS `sample` (`scripts/eprof_sample.sh` is Darwin-only
by construction), on an M4 Pro, against Apple's allocator, dispatching NEON.
KB-PERF-1 through KB-PERF-5 and `benchmarks/encoder_hotspot_reprofile_2026-08-02.md`
are all that platform. But the ratio the standing goal is judged against —
`benchmarks/encode_perf_vs_libaom_2026-09-08.md`, 3.24x-4.03x — is an **x86-64**
number, and KB-PERF-2 already measured that a lever's RANK does not survive a
platform change (its allocation lever was 21 % of the win on Darwin and 86-99 %
on Windows). So the Darwin ranking must not be spent here.

Box: AMD Ryzen 9 7900X, Linux, glibc, AVX2/AVX-512 available.
Subject: `crates/aom-bench/examples/eprof_x86.rs` (new), `perf record`.
Cell: `av1-1-b8-01-size-196x196` cropped 192x192, cq27, bd8 4:2:0, CDEF off /
restoration on (real aomenc's ALLINTRA defaults). **Both arms emit identical
bytes** (1177 B at speed 0, 1473 B at speed 6), so this compares two encoders
doing the same work, not different work.

Reproduce:
```
cargo build --release -p zenav1-aom-bench --example eprof_x86
perf record -F 999 -g -- ./target/release/examples/eprof_x86 port 192 192 27 0 15
```

## Two headline findings

### 1. The port's LARGEST symbol is already at parity — do not chase it

`optimize_txb_core` is the port's top self-time symbol at 11.8 %, and the
coefficient trellis is **32.7 %** of libaom's. As a stage it is **1.29x, +20 ms**
— 6.6 % of the gap. A session that profiled only the port and optimized its top
symbol would spend itself on the one stage that is already fast. This is the
whole reason a like-for-like C profile is taken alongside.

### 2. Loop-restoration search is 26 % of the gap at speeds 0-4 and has NEVER been profiled

**Because every prior profile was taken at `--cpu-used 6`, where it is
structurally OFF.** libaom disables Wiener + SGR at `speed >= 5`
(`speed_features.c:519-520` -> `enable_restoration &= 0`), so the Darwin cell
(1024x1024 cq44 cpu-used 6) could not see this stage at all. Measured here: it
is **0.0 ms in both arms at speed 6** and **90.6 ms vs 10.2 ms at speed 0**.
Restoration is ON by default in allintra at speeds 0-4 — the quality end a
still-image caller selects.

## Speed 0 — port 485.0 ms vs libaom 180.5 ms = **2.69x**, gap +304.5 ms

| stage | port ms | C ms | ratio | gap ms | % of gap |
|---|---:|---:|---:|---:|---:|
| transform (fwd+inv) | 140.0 | 33.5 | 4.18 | **+106.5** | **35.0 %** |
| loop-restoration search | 90.6 | 10.2 | 8.85 | **+80.4** | **26.4 %** |
| encoder RD drivers | 59.1 | 15.8 | 3.73 | +43.2 | 14.2 % |
| coeff trellis (`optimize_txb`) | 90.0 | 69.7 | **1.29** | +20.2 | 6.6 % |
| intra predictors | 34.7 | 14.6 | 2.38 | +20.1 | 6.6 % |
| quantize | 22.4 | 7.3 | 3.07 | +15.1 | 5.0 % |
| memset/memcpy | 14.3 | 1.8 | 7.82 | +12.4 | 4.1 % |
| distortion/variance | 19.4 | 8.0 | 2.44 | +11.4 | 3.8 % |
| allocation | 7.3 | 0.6 | 11.52 | +6.6 | 2.2 % |
| entropy/pack | 3.1 | 0.5 | 6.05 | +2.6 | 0.8 % |
| intra-mode CNN | 0.0 | 0.1 | — | -0.1 | — |
| other (unclassified) | 2.3 | 17.1 | — | -14.8 | -4.9 % |

**The top two are 61 % of the gap.**

### Lever 2 in detail — the port has NO SIMD tier for three of four LR kernels

A `__arcane_*_v3` suffix is the repo's marker that a kernel has a dispatched
SIMD tier. Three of these four do not have one:

| work | port | C | ratio |
|---|---|---|---:|
| `compute_stats` | **32.2 ms, scalar** | `compute_stats_win7_avx2` + `_win5_avx2` 2.4 ms | **13.4x** |
| sgr `calculate_intermediate` + `selfguided_restoration` | **28.1 ms, scalar** | `av1_selfguided_restoration_avx2` 4.3 ms | 6.5x |
| `pixel_proj_error` | **17.8 ms, scalar** | `av1_lowbd_pixel_proj_error_avx2` 2.0 ms | 8.9x |
| wiener convolve | 6.7 ms (`__arcane_wiener_impl_v3`) | `av1_wiener_convolve_add_src_avx2` 0.9 ms | 7.4x |

The three scalar ones are 78.1 of the 90.6 ms. This is integer accumulation
work, so a bit-exact SIMD tier is available in principle — and bit-exactness is
mandatory, because these feed RD decisions and therefore the byte gates.

### Lever 1 in detail — the transform, and it is the INVERSE half

| | port ms | C ms |
|---|---:|---:|
| forward | `av1_fwd_txfm2d_into` 23.3, `run_fwd1d_v3` 6.6, `fwd_row_pass_core_v3` 5.2, `fdct4` 4.5, i16 pair 7.4, `fwd_col_pass_core_v3` 3.6 | `av1_lowbd_fwd_txfm2d_4x4_sse2` 2.0, `_8x8_avx2` 1.1, `fdct8x8_new_sse2` 1.1, `fadst4x4_new_sse2` 1.1, `fdct16x16_new_avx2` 1.0 |
| inverse | `run_inv1d_v3` 20.7, `inv_col_pass_core_v3` 17.2, `inv_row_pass_core_v3` 9.7, `inv_txfm2d_add_into` 5.7 | `lowbd_inv_txfm2d_add_no_identity_avx2` 1.5, `_8x8_` 1.2, `av1_idct8_sse2` 1.1, `_ssse3` 1.0 |

C runs **lowbd** (16-bit) kernels on both halves. The port's inverse is
**53.3 ms against C's ~4.8 ms** and runs the wide i32 path — which is exactly
what KB-PERF-3's own record already flagged and left open: *"Of the port's
encode-side inverse, 79 % still runs the wide i32 path (`lowbd16.rs:69-76` —
only the DCT family passed the i16 audit)."* That named residual is the biggest
single sub-lever on this platform.

## Speed 6 — port 18.51 ms vs libaom 6.12 ms = **3.02x**, gap +12.4 ms

| stage | port ms | C ms | ratio | gap ms | % of gap |
|---|---:|---:|---:|---:|---:|
| encoder RD drivers | 4.3 | 1.0 | 4.30 | +3.3 | 26.7 % |
| transform (fwd+inv) | 3.2 | 0.7 | 4.37 | +2.5 | 19.8 % |
| intra predictors | 2.7 | 0.7 | 4.04 | +2.1 | 16.7 % |
| coeff trellis | 3.0 | 1.7 | 1.78 | +1.3 | 10.6 % |
| intra-mode CNN | 1.5 | 0.2 | 6.80 | +1.3 | 10.5 % |
| allocation | 1.0 | 0.1 | 7.68 | +0.9 | 6.9 % |
| distortion/variance | 0.8 | 0.2 | 4.89 | +0.6 | 5.2 % |
| memset/memcpy | 0.7 | 0.1 | 6.25 | +0.6 | 4.7 % |
| quantize | 0.9 | 0.4 | 2.42 | +0.5 | 4.1 % |
| **loop-restoration search** | **0.0** | **0.0** | — | **-0.0** | **-0.1 %** |

Diffuse, like the Darwin re-profile — but **allocation is 6.9 % here against
Darwin's 20.9 %, and the CNN 10.5 % against 30 %.** Both differences are the
platform effect KB-PERF-2 predicted: glibc prices an allocation differently from
Apple's heap, and the CNN's NEON-vs-AVX2 gap differs.

## Accounting limits — read these before quoting a row

1. **Wall-clock sampling, not instruction counts.** Shares are shares of elapsed
   time in the sampled window.
2. **Self time, no call graph in the rollup.** A stage is a set of leaf symbols
   matched by name (`scratchpad/rollup.py`), so inlined work is attributed to
   whatever it inlined into. C's "other" still holds 17.1 ms of small symbols
   (`av1_optimize_b`, `av1_get_eob_pos_token`, `get_tx_type_cost`,
   `av1_inverse_transform_block`) that belong to the trellis and transform rows —
   so **the trellis is even closer to parity than 1.29x, and the transform gap is
   slightly larger than +106.5 ms.** Neither correction changes a ranking.
3. **Absolute ms = share x per-encode wall**, valid because the untimed warm-up
   is 1 of 16 encodes in both arms and setup is milliseconds.
4. **One cell, one content, bd8 4:2:0.** The `--cpu-used` axis is measured (0 and
   6) and it MOVES the ranking; frame size, bit depth and content are not.
5. 0.26 % (port) / 0.71 % (C) of samples were unresolved kernel frames, dropped.

## Ranked, for this platform

1. **Transform, inverse half** — +53 ms at speed 0, a named open residual of
   KB-PERF-3 (the i16 inverse audit covered only the DCT family).
2. **Loop-restoration search SIMD** — +78 ms at speeds 0-4 across three kernels
   that have no SIMD tier at all. Invisible to every prior profile.
3. **Transform, forward half** — +53 ms; KB-PERF-3 landed the i16 forward pass on
   Darwin and it is clearly not closing the gap here.
4. Encoder RD drivers +43 ms — diffuse, no single mechanism named yet.

Do NOT carry the Darwin ranking (CNN 30 %, allocation 20.9 %) onto this platform:
measured here, the CNN is 10.5 % of a speed-6 gap and 0 % of a speed-0 one, and
allocation is 2.2-6.9 %.
