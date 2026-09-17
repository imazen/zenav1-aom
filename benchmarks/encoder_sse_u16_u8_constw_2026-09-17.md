# sse_u16_u8 const-W monomorphisation + a u8 `aom_sse` twin — 2026-09-17

Cell: the real `photo_512` witness (`/home/lilith/tmp/real/photo_512.yuv`,
512x512 i420, cq27 s3, `eprof_yuv` arm `port`, callgrind `Ir`).

## What landed (`69f2f85`)

`crates/aom-dsp/src/dist/simd_variance.rs`:

- `sse_u16_u8_impl_v3` routes the power-of-two tx widths (8/16/32/64) into
  a const-generic row loop (`sse_u16_u8_rows_v3<W>`): one upfront
  `(h-1)*stride + W` region check replaces the per-row slice checks, and
  `&[u16; W]`/`&[u8; W]` row views carry the len in the type so the chunk
  guard and every `try_into` fold at compile time — W=8 emits only the
  xmm arm, W%16==0 only ymm, fully unrolled. The generic loop stays for
  the other assert-permitted multiples of 8 (24/48/128...).
- `sse_u16_u8_scalar` is `#[inline(never)]`: with the const-W dispatch
  LLVM was inlining the scalar early-return into the arcane body and
  auto-vectorising the 4-wide loop into `vpmovzxwq`/`vpmuldq` i64-lane
  mulch — slower than the call it replaced (measured: the family went
  *up* ~92M before the attribute).
- `dist::sse` gained a dispatched u8x8 kernel — the `aom_sse` twin:
  magetypes generic + arcane v3 (`cvtepu8_epi16` / `sub_epi16` /
  `madd_epi16` / single accumulator / one hadd tree). `sse_scalar` keeps
  the transcription.

## Measured

| | before | after | C |
|---|---|---|---|
| sse_u16_u8 cluster (self Ir) | ~574M | ~266M | `aom_sse` ~102 Ir/call |
| v3 kernel Ir/call | ~622 | ~335 (397K v3 calls) | — |
| encode total | 13.950G | **13.894G** | 9.390G |
| ratio | 1.486x | **1.480x** | — |
| wall | ~399-417ms | ~400-402ms | ~295ms |

## The measured rejection this replaces

The prior plan staged the txb's `src` block to u8 once per transform
block (`walk.src_u8`) and SSE'd `aom_sse`-shape u8×u8 — C's own structure.
Built, measured, REVERTED (`git checkout` of the encoder side; the DSP
kernel stays as the dispatched API twin):

- The u8 kernel does **+1 uop/16px**, not less: the u16 kernel loads the
  source straight into i16 lanes (one ymm load, no widening) while the
  u8 kernel needs `vpmovzxbw` on BOTH operands. 698 vs 622 Ir/call.
- Staging cost ~30M (`downcast_pred_rows` src side) + ~116M caller-side
  Option/dispatch growth in `dist_block_px_domain_into`.
- Net: 13.950G -> 14.030G = **+80M regression**. The C-side advantage
  (a native u8 source plane, never widened) is structural and cannot be
  recovered by per-txb staging.

## Byte gate

photo_512: byte-identical to the post-lpf port stream AND to C (8,822 B).
screen_512: byte-identical to the post-lpf port stream (13,398 B); its
byte-13 C divergence is the documented IntraBC RD residual, unchanged.

## Tests

New `sse_simd_diff`: dispatched vs scalar at every token permutation, all
const-W arms + generic fallback + w==4 scalar route + the
non-multiple-of-8 guard, strided planes, max-diff/flat boundaries.
`dist_diff` already pins `sse` against the real `aom_sse_c` across all 22
block sizes.
