# lowbd loopfilter u8-lane SSE2 mirror — 2026-09-17

Cell: the real `photo_512` witness (`/home/lilith/tmp/real/photo_512.yuv`,
512x512 i420, cq27 s3, `eprof_yuv` arm `port|c`, callgrind `Ir`). C's matched
profile is `/tmp/cg_ii_photo_c.out` (9.390G Ir).

## What landed

`crates/aom-dsp/src/loopfilter/simd.rs` gained an `#[archmage::arcane]` v3
implementation of the lowbd loop filter — a verbatim mirror of
`upstream/aom_dsp/x86/loopfilter_sse2.c`'s eight single-segment kernels
(`aom_lpf_{horizontal,vertical}_{4,6,8,14}_sse2` + `lpf_common_sse2.h`
internals): packed u8 lanes, `adds/subs_epu8` saturating mask arithmetic,
the `filter4`/`filter6`/`filter8`/`filter14` chains, and the unpack-tree
transposes on the vertical axis (including width 14's `transpose_pq_14` and
the flat/flat2 branch). The magetypes `i32x4` kernel lost `v3` from its tier
list so the arcane fn shadows it (the wiener pattern); the span check falls
back to the scalar transcription when the kernel's wider load window is not
fully inside `buf`.

## Measured

| | before | after | C |
|---|---|---|---|
| lpf cluster (self Ir) | ~280M | ~165M | ~52M |
| `lpf_impl_u8_v3` Ir/call | ~390 | 180 (552,488 calls) | ~80 ops/4px |
| encode total | 14.029G | 13.950G | 9.390G |
| ratio | 1.495x | **1.486x** | — |
| wall (3 reps median) | ~415ms | ~399ms | ~295ms |

Wall ratio ~1.352x at 512sq. The residual lpf gap (~165M vs ~52M) is now
mostly per-call front-matter: C's production dispatch batches edges
(`_dual`/`_quad` kernels run 2-4 edge positions per call on wider
transposes), the port still calls the single-segment kernel per position —
a follow-up lever, not landed here.

## Byte gate

photo_512 stream byte-identical to the pre-change port output AND to the C
oracle (8,822 B). screen_512 keeps its documented IntraBC RD residual
(13,398 vs 13,423 B) and stays decoder-conformant + pixel-identical under
the real C decoder (`svt_interop_probe`: c_dec OK, port_dec OK, pixels EQ).

## The C-internal divergence this surfaced

C's lowbd `_sse2` and `_c` kernels are NOT bit-identical to each other: the
mask sums run in saturating `adds_epu8`, so when `blimit+limit >= 255` a
capped sum (255) can pass the criterion where `_c`'s unclamped sum (>255)
fails it. The v3 mirror follows `_sse2` — the kernel libaom actually
dispatches — which is the correct oracle per the evidence hierarchy. The
`_c`-oracled sweeps (`lpf_diff`, `loopfilter_lowbd_diff`, the lowbd tier
permutation in `lpf_simd_diff`) are bounded to `bl+li <= 254` with the
mechanism named in comments; the new `lowbd_lpf_sse2_diff` gates the port
against the real `aom_lpf_*_sse2` exports over the FULL domain including a
pinned `blimit=255` corner (>1000 cells asserted exercised). Production
thresholds never reach the corner (levels are capped well under 63), so the
bounding loses no reachable coverage.
