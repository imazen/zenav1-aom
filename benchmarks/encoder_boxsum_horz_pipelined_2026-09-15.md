# boxsum_horz — software-pipelined vector loop, row copy eliminated (2026-09-15)

## Mechanism

`crates/aom-dsp/src/restore/sgr.rs` `boxsum_horz_impl` (the horizontal half of
`boxsum1`/`boxsum2`, feeding `calculate_intermediate` in the SGR pick path) ran
the sliding-window box filter over a per-call `scratch = vec![0i32; width]`
copy of each `dst` row. The copy existed because the scalar recipe's rolling
window reads positions the vector stores would otherwise clobber — each
`dst[j..j+8]` store overlaps the NEXT chunk's `s[j+6]/s[j+7]` (r5) / `s[j+7]`
(r3) reads.

The vector loop is now software-pipelined instead: all five (r5) / three (r3)
source vectors for chunk j+8 are loaded BEFORE chunk j's store lands, so every
lane still sums original row values. Two small extra pieces make the
no-scratch version exact:

- The left-edge writes (`dst[0]`, `dst[1]` for r5; `dst[0]` for r3) are
  computed from captured `s0..s3`/`s0..s1` and stored AFTER the vector loop —
  the first chunk loads `s[0..10)`/`s[0..9)`, which those stores would
  otherwise clobber.
- The scalar tail is the scalar tier's own rolling `a..e`/`a..c` window,
  seeded with a snapshot (`ta`/`tb`) of the last vector store's final two/one
  cells — the only clobbered cells the tail can read — plus direct `dst`
  reads at positions `>= j` that no store has reached yet.

Result: the per-call `vec![width]` (a memset + alloc + free) and the per-row
`copy_from_slice` (a full-width memcpy) are gone entirely. Bit-exactness is
unchanged — identical `i32` add semantics, identical panic domain (the narrow
`width` contracts match the scalar tier's own unconditional seed reads).

## Measurement

Callgrind, 196x196 cq27 `--cpu-used 3`, 5 reps, paired against the same-cell
profile of `fdc2ce3` (the base arm this builds on), identical binary layout
only apart from this change:

| | boxsum_horz_impl_v3 inclusive Ir | calls |
|---|---|---|
| base (`cg_head196_port.out`) | 60,342,852 | 2,688 |
| new (`cg_boxsum_port.out`) | 53,795,616 | 2,688 |

**−10.9 % kernel inclusive** (~1.31M Ir/rep on a ~1.81G Ir/rep cell → ~0.07 %
of encode). The residual children (the `memcpy`/`memset` edges the old version
paid per row and per call) are gone; the pipelined loop's `more` bookkeeping
costs slightly more self-Ir than the old straight-line chunk loop, so the net
is ~6.5M of memory traffic removed for ~0.7M of scalar bookkeeping added.

Below the ~0.25 % same-binary band resolution — landed on the
`txb_init_levels`/`acc_stat` precedent (strict instruction-count + structural
reduction, wall-unmeasurable at this size).

## Correctness

- `pick_diff` 6/6 (real-C `av1_compute_stats`/`_highbd` oracles exercise this
  exact kernel through `compute_stats` on both lowbd and highbd).
- `kernels_diff` `apply_selfguided_matches_c` green.
- `self_contained_key_frame` 10/10 including the 921 s real-aomenc
  byte-match sweep.
- Shipping cell byte-identical: 1024x1024 cq27 s3 → 40,237 B, ~2,370 ms.
