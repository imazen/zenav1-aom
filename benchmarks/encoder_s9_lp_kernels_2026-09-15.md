# s9 low-precision RD-estimate kernels → dispatched SIMD — 2026-09-15

Profile cell: `port 192x192 cq27 s9 reps=5` (callgrind `Ir`). Same protocol as
`encoder_s9_alloc_class_2026-09-15.md` / `encoder_s9_lp_hadamard_2026-09-15.md`.

Shipping witness `port 1024x1024 cq27 s3`: **byte-identical 40,237 B**.
192x192 s9 witness: **byte-identical 1,894 B**. Full encode `all` suite
**654/654** incl. every e2e byte gate; all 9 `nonrd_block_yrd_lp_diff` tests
green under both runtime AVX2 dispatch and `AOM_FORCE_SCALAR`.

## What landed

The three per-coefficient scalar loops inside `block_yrd_lowbd` — the s9
nonrd estimate's remaining ~20M self Ir — moved into `aom-dsp` as
runtime-dispatched kernels mirroring the C tiers RTCD actually runs:

- **`av1_quantize_lp`** — `quant/mod.rs` carries the `_c` transcription;
  `quant/simd.rs::av1_quantize_lp_dispatch` adds a v3 body mirroring
  `av1_quantize_lp_avx2` instruction-for-instruction (`abs`/`adds`/`mulhi`/
  `sign`/`mullo` lanes, `cmpgt(abs_q,0)` nz test, eob folded from `iscan` as
  `(iscan - nz) & nz` under `max_epi16`, C's param permutes replicated as
  `[row0..7 | row4..7×2]` for chunk 0 and `[row4..7 ×4]` after). Non-x86
  tiers run the `_c` semantics in raster order (eob via the
  inverse-permutation identity).
- **`aom_satd_lp`** — `dist/hadamard.rs::satd_lp` (`_c` transcription) +
  `satd_lp_simd` v3 mirror of `aom_satd_lp_avx2` (`abs_epi16` +
  `madd(., 1)` + i32 accumulate + C's cascade fold).
- **`av1_block_error_lp`** — `dist/simd.rs::block_error_lp` (`_c`
  transcription: i32 `diff`, i32-wrapping square, i64 accumulate) +
  `block_error_lp_simd` v3 mirror of `av1_block_error_lp_avx2`, including its
  three named arms (n==16 `hadd` tree, n==32 two-madd fold, 64-wide loop) and
  their i32-wrap-then-zero-extend semantics.

`quantize_lp` gained the `iscan` parameter the C signature carries — the
scalar arm ignores it exactly as `_c` does; the SIMD tier consumes it for the
raster-order eob. Callers updated: `block_yrd_lowbd` passes the
lp-16x16/8x8-transpose scan+iscan pairs and `scan_orders[TX_4X4][DCT_DCT]`
scan+iscan; `nonrd_idtx` returns and passes the matching
`AV1_FAST_IDTX_ISCAN_*` tables. The new constant
`AV1_DEFAULT_ISCAN_LP_16X16_TRANSPOSE` is transcribed from
`nonrd_opt.h:294` (distinct from the fp table); the diff test asserts the
inverse-permutation property.

## Tier agreement — measured, not assumed

C's own tiers of these kernels genuinely differ in corners; the new
`lp_quantize_tiers_agree_over_the_reachable_range` /
`lp_satd_block_error_tiers_agree_over_the_reachable_range` tests measure the
boundaries against the real exported `_c`, `_sse2` and `_avx2` symbols:

- `quantize_lp`: `_sse2`'s eob tests `dqcoeff != 0` where `_c`/`_avx2`/`_neon`
  test the quantized magnitude. Divergence needs `tmp*dequant ≡ 0 mod 2^16`;
  an exhaustive hunt over all 256 qindices × 3 planes × both lanes found
  **max `tmp*d` = 32768** and **0 realizable wrap inputs** — the wrap-to-zero
  product 65536 is structurally unreachable. All four tiers agree on every
  reachable input.
- `satd_lp`: `abs_epi16` maps -32768 → -32768 where `_c` widens to +32768.
  That is the **only** divergence on the full i16 domain (dense-swept both
  directions), and -32768 is above the lp transforms' ~32654 output bound.
- `block_error_lp`: SIMD wraps the i16 sub and the i32 madd-accumulate;
  1,388 synthetic opposite-sign pairs confirmed divergent but unreachable —
  `quantize_lp` never emits them (dqcoeff carries coeff's sign on every
  reachable input). Measured reachable `|dqcoeff - coeff| ≤ 2308`, under the
  2,896 worst-case bound for the 256-lane i32 accumulation.

The dispatched port is also asserted directly against the
runtime-dispatched C SIMD (`ref_*_lp_simd`, AVX2 on this host) inside
`lp_quantize_satd_block_error_match_c`.

## Measured (192x192 cq27 s9 reps=5, same protocol)

| arm | total Ir | vs base |
|---|---:|---:|
| base (post `5406ec3`) | 246.36M | — |
| + lp kernels dispatched | **229.85M** | **-6.7 %** |

Cumulative session delta at this cell: **271.1M -> 229.85M Ir (-15.2 %)**.

Per-call check: `block_yrd_lowbd` self dropped **~20M -> 0.60M** — the
per-coeff loops are gone; the v3 kernels charge their own symbols
(`quantize_lp_v3` ~0.58M, `block_error_lp_v3` ~0.33M, `satd_lp_v3` ~0.13M,
`hadamard_lp_8x8_v3` 1.08M/6,942 ≈ 155 Ir/call — at C parity).

## Residual attribution (unchanged, named)

- `pack_leaf` re-runs `encode_b_intra_dry` per committed leaf (~44 % of the
  cell) — structural; replay of retained `TxbEncode` is the next lever.
- `optimize_txb_core` 3x C's call count (chroma trellis + the pack re-walk).
- Allocator dispatch ~1M Ir residual; memset/memset floor is
  `forbid(unsafe_code)` bounds.
