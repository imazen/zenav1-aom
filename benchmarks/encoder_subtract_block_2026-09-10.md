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

## KB-PERF-38 — the follow-up annotate, and why one fix was not enough

Re-profiling after the above (port 3241 -> 3148 ms) moved
`highbd_subtract_block` OUT of its callers and into its own symbol at **54 ms —
still 5x libaom's 10.9 ms**. Annotating it again showed the remaining cost was
**the row loop, not the inner one**:

    6.74  movq    (%r9,%rbx,2), %xmm0      <- 4 lanes of real work
    4.46  cmpq    0x68(%rsp), %r11
    4.41  addq    %r14, %rdx
    3.70  cmpq    %rax, %rsi
    2.93  imulq   0x70(%rsp), %r14
    2.26  imulq   0x58(%rsp), %rbx

Two `imul`s and three bounds checks **per row**, around a `movq` — eight bytes,
**four** `i16` lanes. **TX_4X4 is 40 % of transforms**, so a typical row is four
columns and the per-row overhead is larger than the work it guards.

And the rewrite had **cost its own inlining**: before KB-PERF-37 the body was
inlined into its callers and LLVM specialised it on their constant `cols` and
strides; as a standalone symbol that specialisation was gone.

**Fixed** with running offsets (`off += stride`, the same values the
multiplication produced, so the slice bounds are unchanged) plus `#[inline]` to
restore the specialisation. **−0.23 %** (−0.228 % at 21/29, p=0.0241, and
−0.225 % at 26/29, p<0.0001; null +0.038 %, p=0.71), byte-identical, distinct
binaries.

**The transferable part: a fix that makes a hot inlined loop into a standalone
symbol can lose more to lost specialisation than it gains.** Re-annotate after
landing, not just re-band — the band said +3.17 % and looked finished.

## The same fix applied to a THIRD kernel, and it measured NULL

`txb_init_levels_impl_v3` (49 ms, ~2.6x libaom's `av1_txb_init_levels_avx2`) has
the identical per-column index arithmetic — `perf annotate` put **4.1 % of it in
a single `imulq`** and 6.6 % in the per-column pad store. Running offsets were
applied exactly as in KB-PERF-38.

**Measured null**: +0.163 % against one base copy (8/24, p=0.15) and +0.019 %
against the other (12/24, p=1.00), with the null arm itself at +0.105 %.
Reverted; band committed as `.txbinit.rejected.tsv`.

**Why, and it bounds the whole technique:** 4.1 % of a 49 ms symbol is **~2 ms**,
which is **0.06 % of the encode** — an order of magnitude below what a 24-round
band can resolve. The same edit was worth −0.23 % in `highbd_subtract_block`
because that kernel is called far more often and its rows are shorter.

**So "remove the per-row `imul`" is not a general win — it is a win where the
row count is high and the row is short.** Check the symbol's absolute ms and the
share the annotate attributes before assuming a repeat.

## KB-PERF-46 — the named sibling, taken: **−0.35 %**

`sum_squares_2d_i16` was listed below as a sibling of the same shape and left
unmeasured. Taking it: **−0.338 % and −0.360 %** against the two base copies
(23/24 and 21/24, p<=0.0003; null −0.080 %, p=0.15), byte-identical.

It is live inside `search_tx_type_intra_into` — annotating that symbol shows its
`movswl` / `imull` chain with **two `cmp` bounds checks** around it, exactly the
pattern KB-PERF-37 fixed one file over.

Two defects, both already-proven fixes:

* **one bounds check per element** (a runtime slice length indexed by
  `base + c`) — replaced by a row slice, one check per row;
* **a serial `u64` accumulator that blocked vectorization** — unlocked with
  `#[autoversion]`, the same attribute that gave KB-PERF-39 its −0.88 %.

Bit-exact: each `v * v` is computed in `i32` exactly as before (non-negative,
at most `2^30` for an `i16` input, so the `as u64` is exact), and vectorizing
reassociates the sum, which is exact because `u64` addition is associative and
the total cannot overflow (`2^30` over at most `128 * 128` elements is under
`2^44`).

**Both remaining siblings named in this record are now resolved**: this one
landed, and the lowbd `subtract_block` twin is decoder-path only (the encoder is
u16-through at every bit depth) so it does not appear in the encoder profile.

## Siblings, same shape, not yet measured

* **`subtract_block`** (the lowbd `u8` twin, same file) — identical triple
  indexing. Not in the encoder's profile because the encoder is u16-through at
  every bit depth, but it is on the DECODER path.
* **`sum_squares_2d_i16`** — one bounds check per element plus a serial `u64`
  accumulator, blocking vectorization of a trivially reducible sum.

Both are free to fix by the same method and should be taken next.

## Not covered

One box, one content class, `--cpu-used 3`, x86-64.
