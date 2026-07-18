# Gate-3 intra-predictor SIMD — callgrind Ir deltas (2026-07-18)

Whole-decode instruction-reference (Ir) counts measured with valgrind callgrind
on the headline 2K/4K photographic stills-decode cells, **port side**, before
and after the non-directional highbd intra-predictor SIMD (SMOOTH / SMOOTH_V /
SMOOTH_H / PAETH lane-batched into `i32x8` magetypes kernels, AVX2 / `X64V3`
tier; DC family / V / H rewritten to memset/memcpy slice ops). Ir is
deterministic, so these are load-independent.

The "before" is `79e7a6d` — it already carries the transform + deblock + cdef +
txb + intra-edge SIMD, so `predict_highbd` (the per-block non-directional intra
predictor) was the **top remaining scalar decode hotspot** on these
`aomenc --allintra` stills. Its true cost is its self-Ir **plus** the std-lib
lines callgrind attributes to it via inlining (the per-pixel `r*stride+c` index
math in `uint_macros`, the bounds `cmp`, the `range`/`take` iterators) — the
same "cluster" methodology as `benchmarks/decode_hotspots_2026-07-17.md`.

The reference edges (`assemble_nd_edges`, already `#[autoversion]`) and the
directional predictors (`z1/z2/z3_high`) are **byte-identical Ir** in both
binaries — they are NOT part of this landing — so the whole-decode delta is the
non-directional predictor kernels alone.

## Whole-decode Ir (`dec` cell; N port decodes + 1 constant C-oracle that cancels in the delta)

| cell | iters | before (`79e7a6d`) | after (intra SIMD) | Δ whole-decode |
|---|---|---|---|---|
| `dec_mosaic_2k_cq20` (qindex 80, coeff-heavy)   | 3 | 3,023,561,607 | **2,958,546,363** | **−2.15 %** |
| `dec_mosaic_2k_cq40` (qindex 160, deblock-heavy) | 2 | 1,595,867,536 | **1,543,024,372** | **−3.31 %** |
| `dec_mosaic_4k_cq20` (qindex 80, 4K)             | 2 | 11,779,319,418 | **11,579,739,360** | **−1.69 %** |

## Intra-prediction cluster (the moved work)

`predict_highbd` self+inlined **before** (all modes inside it) vs, **after**,
`predict_highbd` (now just DC/V/H fills + the DC sum + inlined) plus the two
dispatched SIMD kernels the compute predictors moved into
(`simd::__arcane_{smooth,paeth}_impl_v3`, each incl. its inlined std lines):

| cell | `predict_highbd` before | `predict_highbd` after | +SMOOTH/PAETH SIMD | intra total after | Δ intra |
|---|---|---|---|---|---|
| `dec_mosaic_2k_cq20` | 128,618,872 | 38,128,732 | 30,853,948 | **68,982,680** | **−46.4 %** |
| `dec_mosaic_2k_cq40` | 103,728,603 | 21,526,707 | 33,014,061 | **54,540,768** | **−47.4 %** |
| `dec_mosaic_4k_cq20` | 414,769,290 | 130,541,256 | 104,972,556 | **235,513,812** | **−43.2 %** |

The intra-cluster delta (−59.6 M / −49.2 M / −179.3 M) accounts for the whole-
decode delta (−65.0 M / −52.8 M / −199.6 M) to within the DC/V/H fill work that
moved to memset/memcpy + code-layout noise between two independently-compiled
binaries. cq40 (higher qindex → fewer coefficients, more of the frame predicted)
shows the larger intra share, as expected.

## SIMD confirmed live (callgrind_annotate, after, `dec_mosaic_2k_cq20`)

- `simd::__arcane_smooth_impl_v3` fires (7.2 M self + 11.0 M inlined index math),
  `simd::__arcane_paeth_impl_v3` fires (9.6 M) — the AVX2 tier is live in the
  real decode, not just the unit differential. (`smooth_v`/`smooth_h` kernels are
  compiled + differentially proven but this photographic cell codes no SMOOTH_V/H
  blocks, so they show 0 here — content-dependent.)
- `predict_highbd` self dropped 71,667,060 → 9,412,648; the per-pixel index-math
  clusters it used to carry (`uint_macros` 20.9 M, `cmp` 10.4 M, `range` 22.2 M,
  `take` 3.2 M) collapsed with the slice-op / vector rewrite.
- `assemble_nd_edges` / `z1/z2/z3_high` Ir UNCHANGED before/after (not touched).
- The decoder Gate-1 `conformance_corpus` (byte-identity + golden MD5) passed on
  the after binary WITH SIMD live AND under `AOM_FORCE_SCALAR` → the SIMD decode
  is byte-identical to the C oracle in the real decode, not just the unit diff.

## What is SIMD vs still scalar (honest fraction)

- **SIMD (this landing):** the four compute non-directional predictors —
  SMOOTH (4-term weighted blend), SMOOTH_V / SMOOTH_H (2-term), PAETH (base-
  distance select) — as `i32x8` magetypes kernels, vectorized over columns.
  DC / DC_TOP / DC_LEFT / DC_128 (whole-block fill) and H (per-row fill) → glibc
  `memset`; V (row copy) → `memcpy`. Together these are all 10 non-directional
  modes of `predict_highbd`.
- **Still scalar (documented follow-ups):**
  1. **`bw == 4` blocks** take the scalar tail in the compute kernels (the
     vector body runs `bw & !7` columns; 4-wide blocks have 0 full vectors). A
     2-rows-per-`i32x8` path (like the cdef w4 kernel) would vectorize them, but
     4-wide blocks carry little of the per-block pixel volume. Bit-exact today
     (the tail IS the scalar core).
  2. **Directional predictors** `z1/z2/z3_high` (`dir.rs`; `z2_high` ~0.9 % of
     decode alone) + **CfL** (`cfl_predict_block` ~0.6 %) are separate scalar
     kernels, a natural next intra landing.
  3. **AVX-512 (`X64V4`, 16-lane) + NEON/WASM** — the kernels are
     `#[magetypes(define(i32x8), v3, neon, wasm128)]`, so NEON/WASM tiers exist;
     the perf box is x86-AVX2, AVX-512 is a wider follow-up.
  4. **lowbd 8-bit lane path** — the decoder runs the highbd (u16, i32-lane)
     predictors at every bit depth (per the decode analysis). The i32 lanes are
     bit-exact for all bd; an i16/lowbd specialization for bd8 is a follow-up.

## Reproduce

```
# before binary: build gate3_profile at 79e7a6d (scalar predict_highbd)
# after  binary: build gate3_profile at 44b0b1c (intra SIMD)
valgrind --tool=callgrind --callgrind-out-file=<out> \
  ./target/profiling/gate3_profile dec port dec_mosaic_2k_cq20 3
callgrind_annotate --auto=no --inclusive=no <out>
```

Raw callgrind outputs: NOT committed (>800 KB each); regenerate as above. The
`mosaic-{2k,4k}-*.ivf` cells are gitignored (regenerable per
`benchmarks/decode_hotspots_2026-07-17.md`).
