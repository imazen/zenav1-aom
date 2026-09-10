# KB-PERF-39 — `block_error` via `#[autoversion]`: **−0.88 %**

**2026-09-10.** Byte-identical; **−0.88 %** at 1024x1024 cq27 `--cpu-used 3`
(−0.887 % and −0.864 % against the two base copies, 25/28 rounds each,
p<0.0001), null +0.058 % (11/28, p=0.34).

The third landing found by **annotating a symbol** rather than reading a class
table, and the cheapest of the three: no new kernel, no intrinsics, no `unsafe`.

## How it was found

The class table had `distortion` at +108 ms / 6.3 % of the gap — an
unremarkable row, never examined at any preset. Annotating its symbol
`dist_block_tx_domain` (61 ms against libaom's `av1_block_error_avx2` at
8.4 ms — **7.3x**) showed a loop that was already bounds-check-free and simply
**scalar**:

    12.00  movl    0x4(%rdi,%rcx,4), %ebx
     9.82  imull   %ebx, %ebx
     7.26  subl    (%rdx,%rcx,4), %ebp
     7.15  cltq
     4.65  movslq  %r11d, %r11

A 32-bit multiply, a sign-extend and a 64-bit add, two-way unrolled. libaom does
the same arithmetic 8 lanes wide.

## The fix — the smallest one that could work

`#[autoversion]` (archmage) compiles a function once per SIMD tier with the
target features enabled and dispatches at runtime. That is enough on its own:
LLVM's auto-vectorizer lowers this loop to 8-lane `pmulld` with widening
accumulation once it is allowed to emit AVX2. **No raw intrinsics, no `unsafe`,
no hand-written kernel** — the body is the transcribed scalar loop, unchanged.

The crate already had this pattern (`sad_simd` in the same file), so this is
reusing a tool that was sitting there rather than adding one.

## Bit-exactness

The per-element arithmetic is untouched — `wrapping_sub` and `wrapping_mul` in
`i32` (the wrap is load-bearing and matches C's `int` multiply; KB-ARM-FLOAT
root #3 is the entry that established that domain), then sign-extended to `i64`.

Vectorizing **reassociates** the two sums, and that is exact here rather than
approximately so: `i64` addition is associative, and neither accumulator can
overflow — `|diff * diff| <= 2^31` over at most 4096 coefficients is under
`2^43`. `dqcoeff` is sliced to `coeff.len()` so a short `dqcoeff` panics exactly
where the indexed form did.

The verbatim scalar loop is kept as `block_error_scalar_ref`, so the
transcription and its wrapping semantics remain readable next to the dispatched
form.

## Correctness

* Byte-identical on four cells: 40,237 B (1024x1024 s3), 39,694 (1024 s0),
  10,912 (512 s0), 11,961 (512 s6) — two sha256-**distinct** binaries.
* `-p zenav1-aom-dsp` **446/446 in both dispatch modes**, including
  `block_error_diff` against the real exported C.

## The band

| arm | median ms | paired median | rounds faster | p |
|---|---:|---:|---:|---:|
| base | 3124.2 | — | — | — |
| baseB | 3121.4 | +0.058 % | 11/28 | 0.3449 |
| new vs base | 3094.0 | **−0.887 %** | 25/28 | **<0.0001** |
| new vs baseB | | **−0.864 %** | 25/28 | **<0.0001** |

## Not covered

`highbd_block_error` (the bd10/bd12 arm) is untouched — its products are `i64`,
which vectorizes far less well, and this cell is bd8. One box, one content
class, `--cpu-used 3`, x86-64.
