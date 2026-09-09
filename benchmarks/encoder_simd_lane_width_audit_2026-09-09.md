# The port's SIMD is 3-7.5x slower than libaom's in NINE operations — one cause: lane width

**2026-09-09. No code changed.** A systematic port-vs-C comparison at the
SHIPPING preset (`--cpu-used 3`, 1024x1024 cq27 — see
`encoder_shipping_preset_1mp_2026-09-09.md`), restricted to operations where
**both sides are already vectorised**. It replaces "transform is 40 % of the
gap" with a cause.

## The coverage question first, because the obvious hypothesis is wrong

* **libaom** spends **32.48 % of its wall (503.2 ms) in symbols carrying an
  explicit ISA suffix** (`_avx2` / `_sse2` / `_ssse3` / `_sse4_1`), 225 of them.
* **the port** spends **28.17 % of its wall (985.6 ms) in magetypes vector
  bodies** (`__arcane_*`), 37 of them.

So the port is **not** scalar-where-libaom-is-SIMD as a general matter. Its
vector bodies simply cost about twice what libaom's SIMD costs, and the
comparison below says why.

## The paired table

Same operation, port against C, at the shipping preset:

| operation | port | C | ratio | **gap** |
|---|---:|---:|---:|---:|
| directional intra z1/z2/z3 | 146.2 ms | 45.6 ms | 3.21x | **+100.7** |
| quantize_fp | 79.1 | 24.5 | 3.23x | +54.6 |
| **filter_intra** | 54.6 | 7.3 | **7.50x** | +47.3 |
| loopfilter | 57.4 | 12.5 | 4.57x | +44.8 |
| SGR self-guided | 50.7 | 14.6 | 3.48x | +36.2 |
| wiener restore | 40.9 | 6.2 | 6.61x | +34.7 |
| txb_init_levels | 44.4 | 13.6 | 3.26x | +30.8 |
| hadamard | 39.5 | 13.0 | 3.04x | +26.5 |
| pixel_proj_error | 23.4 | 7.9 | 2.97x | +15.5 |
| **nine operations** | | | | **+391 ms = 20 % of the whole gap** |
| trellis (`optimize_txb`) | 388.0 | 415.2 | **0.93x** | −27.2 |

Three more where the port has **no symbol at all** because the work is inlined
into an RD driver — `subtract block` (C 25.9 ms), `block_error` (8.8),
`convolve copy` (7.3). Those are a MEASUREMENT gap, not necessarily a
performance gap, and must not be counted either way.

## The cause: same register width, fewer elements

Read the port's `#[magetypes(define(...))]` against the lane type libaom's
kernel actually operates on:

| operation | port lane type | elements | libaom | elements | predicted | **measured** |
|---|---|---:|---|---:|---:|---:|
| **loopfilter** | `i32x4` (128-bit!) | 4 | `aom_lpf_*_sse2` on u8 | 16 | **4x** | **4.57x** |
| quantize_fp | `i32x8` | 8 | `av1_quantize_fp_avx2` on i16 | 16 | 2x | 3.23x |
| wiener | `i32x8` | 8 | `av1_wiener_convolve_add_src_avx2` on i16 | 16 | 2x | 6.61x |
| txb_init_levels | `i32x8` | 8 | `av1_txb_init_levels_avx2`, packs to u8 | 32 out | >=2x | 3.26x |
| SGR | `i32x8` | 8 | `av1_selfguided_restoration_avx2` on i32 | 8 | 1x | 3.48x |
| filter_intra | **scalar taps** | 1 | `av1_filter_intra_predictor_sse4_1` | 8-16 | >>1 | 7.50x |

**The loopfilter prediction lands almost exactly (4x predicted, 4.57x
measured), and it is the cleanest case: `define(i32x4)` is a 128-BIT type, so
its `v3` (AVX2) body emits no 256-bit instruction at all.** Verified in the
disassembly of the shipped binary:

    port lpf_impl_v3:              insns=2603  ymm=0     xmm=1056
    port quantize_fp_impl_v3:      insns=365   ymm=80    xmm=37   mem=110
    port txb_init_levels_impl_v3:  insns=185   ymm=14    xmm=12   mem=49

**And libaom's loopfilter is `_sse2` too** — 128-bit on both sides, with one
`_quad_avx2` at 0.03 %. So this is NOT "the port is missing AVX2 where C has
it". Both are SSE-class; libaom puts **16 u8 elements** in that register and the
port puts **4 i32**.

That is the same structural root `CLAUDE.md` already records — the encoder holds
planes as `u16` at every bit depth and runs the highbd kernels at bd8 — but
stated as a LANE COUNT rather than a bit depth, measured across nine independent
operations, and at the preset that ships. It also explains why the two landings
that DID narrow a lane type paid (KB-PERF-3's i16 forward transform, KB-PERF-5's
`u16x16` SMOOTH) while three lane-width-neutral attempts measured null
(KB-PERF-3's half-batch, the inverse i16 twin, and today's forward row arm
below).

## The two ranked levers this produces

1. **`filter_intra`, +47.3 ms, and the port's taps are SCALAR where libaom has
   SSE4.1.** KB-PERF-11 removed that kernel's oversized memset and said so
   explicitly: *"the taps themselves are still scalar — 8 outputs x 7 taps per
   4x2 block is the natural 8-lane dot product libaom's SSE4.1 vectorises — its
   own landing."* It is live at the shipping preset (54.6 ms measured here),
   unlike at `--cpu-used 6` where it is 0 % of leaves.
2. **`loopfilter`, +44.8 ms, `define(i32x4)` -> a narrower type.** Its operations
   are add / sub / abs / min / max / compare / blend, all of which magetypes has
   for narrow types — so unlike `wiener` (blocked on integer interleave) and
   `quantize_fp` (blocked on `mul_high`), it is not blocked on vocabulary.

`hadamard` at 3.04x is a third, and KB-PERF-10 already records why it is hard:
an in-register 8x8 i16 transpose needs a shuffle magetypes does not have.

## Honest limits

Shares are each arm's own wall from a single profiling run per side; the pairing
is by operation and is hand-made, so a symbol the regex missed lands in neither
column. The three inlined operations cannot be compared at all. One cell, one
content class, one box, bd8 4:2:0, `--cpu-used 3`, x86-64. The "predicted"
column is a lane-count ratio, not a model — it ignores per-call overhead, which
is why the measured numbers are uniformly WORSE than predicted rather than equal
to it.
