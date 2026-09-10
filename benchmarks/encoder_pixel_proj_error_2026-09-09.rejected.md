# REJECTED: hand-vectorising `pixel_proj_error`'s squaring — **LLVM had already done it**

**2026-09-09. Built, gated, measured, REVERTED.** The last bounded item on the
lane-width audit's list, and the one KB-PERF-6 explicitly scoped:
*"To claim those milliseconds, DERIVE the bound and gate it at runtime the way
`intra/dir_simd.rs` gates its tap bound."*

## The reasoning was right, and the target was already gone

`pixel_proj_error_impl` ends each 8-pixel chunk with

    for x in e.to_array() { err += x as i64 * x as i64; }

which reads as a 32-byte stack round-trip plus eight scalar `i64`
multiply-accumulates. Costed per 8 pixels: **~25 uops against ~6** for a
vector form — a ~54 % cut on the kernel's tail, worth ~0.29 % of the gap.

**KB-PERF-6's bound does not block it**, and that part of the analysis holds:
that record declined because reducing the squares in `i32` needs
`8 * e^2 < 2^31`, i.e. `|e| < 16384`, and `|e|` sits AT that bound. **But the
bound is on the SUM, not on the square.** `e^2` alone needs `e^2 < 2^31`, true
with ~8x margin even at that worst case (16384^2 = 2.68e8 against 2.1e9), and
accumulating into **`i64` lanes** removes the sum bound entirely. Integer
addition is associative, so the reassociation is exact and no gate is needed.

That was all correct. It was also beside the point.

## The measurement, and then the disassembly

48 rounds, rotated, same-binary null:

| arm | paired median | rounds faster | p |
|---|---:|---:|---:|
| baseB (same binary) | +0.164 % | 20/48 | 0.3123 |
| hand-vectorised | **−0.083 %** | 28/48 | **0.3123** |

Null and effect are the same size and neither is significant. Band spreads were
2.5–3.3 % against this box's usual 0.9–1.6 %, so it was a noisy window — but the
disassembly settles it without needing a quieter one:

| | instructions | `imul` | `vpmovsxdq` | `vpaddq` | `vpmulld` |
|---|---:|---:|---:|---:|---:|
| **base** (before the change) | 383 | 3 | **2** | **3** | **2** |
| hand-vectorised | 376 | 3 | 2 | 4 | 3 |

**The base build already had the i32->i64 widen and the i64 vector accumulate.**
LLVM auto-vectorised `for x in e.to_array() { err += x*x }` on its own — legally,
because an integer reduction is reassociable — and the hand-written version
moved 383 instructions to 376.

## The lesson, and it is the directive's own

*"if it doesn't move the needle it's a test harness issue"* — checked, and it is
not: the path is identical because **there was no scalar path to replace.**

**A scalar-looking source line is not scalar machine code.** The lane-width audit
ranks by comparing port symbols against C symbols, and KB-PERF-6's comment
("squares in the scalar tier's exact order") describes the SOURCE. Neither
establishes what the compiler emitted. The check costs one `objdump` and would
have saved this implementation outright.

**Add it to the costing step**: before hand-vectorising anything that looks
scalar, disassemble it. This session's three landings all replaced code the
disassembly confirmed was scalar — `filter_intra` had `ymm=0`, `hadamard_8x8`
had `ymm=0`, `wiener` had `vpmaddwd=0`. This one did not, and that difference
was visible before any code was written.

## Residual

`pixel_proj_error` remains **23.4 ms against `av1_lowbd_pixel_proj_error_avx2`'s
7.9 = 2.97x**. Since the squaring is already vectorised, the gap is in the `e`
computation — where the port is `i32x8` (8 lanes) and libaom's kernel is `i16`
(16 lanes) on `u8` source. That is the **u16-at-bd8 root**, not a missing
kernel, and it is not reachable from inside this function.

Band kept: `.rejected.band1024s3.tsv`.
