# KB-PERF-37 — `highbd_subtract_block` did three bounds checks per PIXEL: **−3.17 %**

**2026-09-10.** Byte-identical; **−3.17 %** at 1024x1024 cq27 `--cpu-used 3`
(−3.170 % and −3.188 % against the two base copies, **30/30 rounds faster on
both**, p<0.0001), null dead flat (−0.014 %, 17/30, p=0.58).

**Ten times the largest landing of this cycle, from six lines.** And it was
found only by disassembling a symbol, not by reading a profile's class table.

## How it was found — annotate the symbol, not the class

The class table said `rd-driver` was +261 ms at ratio 2.19 and had never been
drilled into at the shipping preset. Its top symbol, `txfm_rd_in_plane_intra`,
is **124 ms** against ~44 ms of C equivalents. `perf annotate` on that symbol,
sorted by sample count, put ~20 % of it in one basic block:

    3.16  leaq    (%r11,%rbx), %rdi
    3.02  movzwl  (%r13,%rbx,2), %edi
    2.88  cmpq    %rcx, %rdi
    2.60  subw    (%rsi,%rbx,2), %di
    2.25  leaq    (%r9,%rbx), %rdi
    1.96  movw    %di, (%rdx,%rbx,2)
    1.87  cmpq    %rax, %rdi
    1.62  cmpq    %r14, %rdi

That is `highbd_subtract_block` inlined: **one 16-bit subtract per iteration,
three `cmp` bounds checks around it, and no vectorization at all.** libaom's
counterpart is `aom_subtract_block_sse2` at 10.9 ms.

**The lesson is about instrument choice.** Nine landings this cycle came from
class tables and symbol shares, and they averaged 0.3 pp. This came from
`perf annotate` on a single symbol and is worth 3.17 pp. **A class table cannot
see a hot loop that is inlined into a large caller** — the caller's self-cost
absorbs it and looks like diffuse driver overhead.

## The defect and the fix

    for r in 0..rows {
        let (d, s, p) = (r * diff_stride, r * src_stride, r * pred_stride);
        for c in 0..cols {
            diff[d + c] = (src[s + c] as i32 - pred[p + c] as i32) as i16;
        }
    }

Three independent runtime slice lengths indexed by a computed base plus `c`, so
the compiler emits a bounds check per buffer per element and cannot prove the
accesses contiguous — which blocks vectorization of what is otherwise a plain
16-lane job.

Fixed by taking one row slice per buffer per row and zipping them, so there is
**one bounds check per row per buffer instead of one per element**, and the
inner loop is a contiguous `i16` subtract LLVM vectorizes.

**Bit-exactness is by construction:** `(a as i32 - b as i32) as i16` truncates
to 16 bits, which is exactly the 16-bit wrapping difference, so a vector
subtract computes the identical value. The slice bounds are the same ones the
indexed form required, so it panics on exactly the inputs the old form panicked
on.

## Correctness

* Byte-identical on four cells: 40,237 B (1024x1024 s3), 39,694 (1024 s0),
  10,912 (512 s0), 11,961 (512 s6) — two sha256-**distinct** binaries, checked
  before the band.
* `-p zenav1-aom-dsp` **446/446 in both dispatch modes**. The function is
  covered against the real exported C by the dist differentials, and it feeds
  every intra and inter residual in the encoder — a wrong value here could not
  produce byte-identical output on four cells.

## The band

| arm | median ms | spread | paired median | rounds faster | p |
|---|---:|---:|---:|---:|---:|
| base | 3229.6 | 1.0 % | — | — | — |
| baseB | 3228.0 | 0.9 % | −0.014 % | 17/30 | 0.5847 |
| new vs base | 3127.9 | 1.2 % | **−3.170 %** | **30/30** | **<0.0001** |
| new vs baseB | | | **−3.188 %** | **30/30** | **<0.0001** |

## Siblings, same shape, not yet measured

* **`subtract_block`** (the lowbd `u8` twin, same file) — identical triple
  indexing. Not in the encoder's profile because the encoder is u16-through at
  every bit depth, but it is on the DECODER path.
* **`sum_squares_2d_i16`** — one bounds check per element plus a serial `u64`
  accumulator, blocking vectorization of a trivially reducible sum.

Both are free to fix by the same method and should be taken next.

## Not covered

One box, one content class, `--cpu-used 3`, x86-64.
