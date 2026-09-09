# Quantize class investigated — two levers refuted, the real one is BLOCKED on magetypes

**No code changed.** This is a diagnosis-only record, written because the
re-taken ranking (`encoder_x86_reprofile_1024_2026-09-09.after.md`) called
quantize *"the cheapest never-examined item"* and that is **wrong**. Correcting
it here so the next session does not spend a cycle finding out.

## Like-for-like split of the +384 ms

Port shares are of the port's own 10128 ms wall, C's of its own 4222 ms.

| | port | C | ratio | gap |
|---|---:|---:|---:|---:|
| the fp kernel (`quantize_fp_impl_v3` + dispatch) | 241 ms | `av1_quantize_fp_avx2` + 32x32 70 ms | **3.4x** | +171 |
| the driver (`xform_quant_into` + `_optimize_split_into`) | 242 ms | `av1_quant` + facade + `av1_xform_quant` 79 ms | **3.1x** | +163 |
| quantize_b | 53 ms | 4 ms | 13.9x | +49 |

The driver half is as large as the kernel half, which is not what the class
table suggests.

## REFUTED #1 — the `iscan` load is not a stack round-trip. Check codegen, not source shape.

`quantize_fp_impl` builds its EOB candidate with

    let isc = i32x8::from_array(token, core::array::from_fn(|k| iscan[base + k] as i32 + 1));

which is the same shape KB-PERF-15 and KB-PERF-10 both found to be a real stack
round-trip — per-lane slice indexing plus a scalar add, feeding a stack array.
So it looked like a third instance, and `widen_low_i16_to_i32` (new in 0.9.29,
contract *"Widen in natural lane order: result[i] = a[i + 0] as i32"*, AVX2
`_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a))` — no per-128-bit-lane permute
trap) looked like the fix.

**Disassembling the shipped kernel refutes it outright:**

    vpmovsxwd (widening loads)  5
    movswl    (scalar i16 loads) 0
    bounds-check panics          0
    stack spills (movw ..(%rsp)) 0

LLVM had already contracted the whole `from_fn` into widening loads. **The
lesson is the inverse of KB-PERF-15's**: that one's round-trip was real and
visible in the object code; this one's was an illusion of the source shape.
Disassemble before costing a lever off a `from_fn`/`to_array` pattern — it is
~30 s and it saved a full landing cycle here.

## REFUTED #2 — the driver has no hot loop; its cost is per-call overhead

`perf annotate` on `xform_quant_into` (158 samples): **no line above 3.84 %
local, and everything above 1 % is prologue, the `TX_W`/`TX_H` table lookups,
the size assert and the `log_scale`/scan jump tables.** `txb_entropy_context`
(0.59 %) and `get_txb_ctx` (0.60 %) are their own symbols, so they are *not*
inlined into it, and `xform_quant_optimize_split_into` already runs on
`XformQuantScratch` — the KB-PERF-13 allocation shape is **already fixed** here.

So the driver's 242 ms is a large **fixed cost per transform-block call**
(re-deriving `TX_W*TX_H`, `txb_wide*txb_high`, `tx_scale`, `scan`, `iscan`,
`resolve_qm` on every call, all pure functions of `(tx_size, tx_type)` that C
hoists into its caller). There is no loop to vectorize; closing it means
changing the call signature so the caller passes the derived block, which is a
refactor across every call site, not a lever.

## THE REAL LEVER, AND IT IS BLOCKED

libaom's `av1_quantize_fp_avx2` packs two `i32x8` into one **`i16x16`**
(`_mm256_packs_epi32`, `load_coefficients_avx2`) and quantizes **16
coefficients per vector**, using one instruction where the port uses several:

| step | libaom (i16, 16 lanes) | port (i32, 8 lanes) |
|---|---|---|
| round + clamp | `_mm256_adds_epi16` | `min(abs,cap) + rnd` then `clamp` — 3 ops |
| `(a*q) >> 16` | **`_mm256_mulhi_epi16`** | `mul` then `shr_arithmetic` — 2 ops |

2x the lanes and ~2x fewer ops ≈ the measured 3.4x. This is exactly the
KB-PERF-3/4/5 recipe (a bd8-gated i16 path with the i32 path as the runtime
fallback), and the bound is the one KB-PERF-3's audit already established —
coefficients are `bd + 8` bits, so they fit i16 at bd8, which is *why* libaom's
saturating pack is safe there.

**It cannot be written against magetypes 0.9.29.** Checked exhaustively, not
assumed — the complete set of multiply primitives in the crate is:

    fn mul        fn mul_add        fn mul_sub

`_mm256_mulhi_epi16` is **never referenced anywhere in magetypes**. Without a
high-multiply, `(a*q) >> 16` in i16 lanes has to widen to i32, multiply, shift
and narrow back, which spends the entire lane advantage. `saturating_add`,
`abs`, `widen_low/high_i16_to_i32` and `narrow_saturating_i32_to_i16` all DO
exist, so `mul_high` is the single missing piece.

## What this changes about the ranking

**Quantize is not a cheap unexamined lever — it is a blocked one**, in the same
class as loop-restoration's wiener (`docs/MAGETYPES_VOCABULARY.md`). Two of the
three largest remaining kernel gaps are now known to be gated on **missing
magetypes integer primitives**, not on port-side effort:

| gap | needs | present in 0.9.29 |
|---|---|---|
| loop-restoration wiener, +901 ms at 7.6x | integer `interleave` | no |
| quantize fp kernel, +171 ms at 3.4x | `mul_high` (i16) | no |

That is a **magetypes contribution**, not a version bump — the 0.9.29 bump
(`9f86281`) is already in and does not supply either. Until one lands, the
remaining port-side headroom in clause (4) is concentrated in the transform
programme (+2263 ms) and in per-call overhead of the kind the driver shows here.

## Method notes

* One box, x86-64, 1024x1024 cq27 `--cpu-used 0`, the profile behind
  `encoder_x86_reprofile_1024_2026-09-09.after.md`.
* `perf annotate` on a `::`-qualified Rust symbol is slow on this binary
  (~2 min) but works; rank its lines with `sort -rn` rather than reading in
  address order, or the prologue hides the shape.
