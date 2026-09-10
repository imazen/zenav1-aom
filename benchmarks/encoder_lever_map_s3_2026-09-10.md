# Every named lever left in clause (4), measured like-for-like at the shipping preset — and together they are only 27 % of what the bar needs

**2026-09-10, no code changed by this document.** Fresh `perf record -F 499` of
both arms at **1024x1024 cq27 `--cpu-used 3`**, symbol shares converted with each
arm's own measured wall time (port 3040.11 ms, C 1549.47 ms, ratio **1.9620x**,
gap **1490.6 ms**, both arms 40,237 bytes).

## The table

| item | port ms | C ms | ratio | gap | % of gap |
|---|---:|---:|---:|---:|---:|
| memset + memmove | 111.9 | 15.3 | 7.29x | **+96.5** | 6.5 % |
| variance family | 89.7 | 18.3 | 4.91x | **+71.4** | 4.8 % |
| *trellis — DO NOT TOUCH* | *512.9* | *455.7* | *1.13x* | *+57.2* | *3.8 %* |
| directional intra z2 | 84.2 | 36.9 | 2.28x | +47.3 | 3.2 % |
| loopfilter | 57.2 | 12.2 | 4.67x | +44.9 | 3.0 % |
| quantize_fp | 64.1 | 22.9 | 2.80x | +41.2 | 2.8 % |
| *intra-CNN — PARITY item, not clause (4)* | *42.0* | *4.3* | *9.67x* | *+37.6* | *2.5 %* |
| directional intra z1/z3 | 50.5 | 15.6 | 3.22x | +34.8 | 2.3 % |
| quantize_b — **no SIMD at all** | 31.3 | 2.6 | 11.89x | +28.7 | 1.9 % |
| `intra_avail` — no C counterpart | 22.5 | — | n/a | +22.5 | 1.5 % |
| hog | 29.2 | 8.1 | 3.62x | +21.1 | 1.4 % |

## The finding, and it is the point of the document

**Closing EVERY named lever completely — to exact parity with libaom, which no
landing in this project has ever achieved on any kernel — is +408.5 ms, i.e.
27.4 % of the gap. The 1.5x bar needs 715.9 ms (48 %).**

So the bar is **not** reachable by the currently-named levers, and a plan that
lists them and stops is short by ~307 ms. This does not contradict
`encoder_clause4_reachability_2026-09-10.md` — that record's route is *halving
every class gap* (745 ms), which is breadth across the whole profile, not depth
on named symbols. **What this table adds is the first quantification of how much
of the requirement the named symbols actually carry: about a quarter.**

## The asymmetric-pairing hazard bit again, and it was material

The first version of this table read C's symbols at `--percent-limit 0.10` and
got **variance 11.2 ms (8.04x)** and **loopfilter 4.6 ms (12.30x)**. Both were
wrong, because C size-specialises those families into ~20 kernels each
(`aom_variance4x4_sse2`, `aom_lpf_vertical_14_dual_sse2`, ...) and most fall
below the threshold, while the port has ONE generic symbol that clears it.
Re-read at `0.005 %`: **variance 18.3 ms (4.91x)** and **loopfilter 12.2 ms
(4.67x)** — ratios cut roughly in half.

`CLAUDE.md` already records this exact error twice (the filter-intra row, the
loopfilter row). It recurs because the natural way to read a profile is
"top N symbols", and top-N is precisely what discards a size-specialised family.
**Sum a FAMILY on both sides at a threshold low enough to include C's smallest
variant, or the ratio is inflated by construction.**

## Two negatives measured today, both on the largest row

The `memset + memmove` row is the biggest, and two attempts on it this session
both failed:

1. **The redundant reconstruct round trip** (`encoder_recon_in_place_2026-09-10.md`)
   — real, byte-identical, kept, and **null** (−0.147 % against a −0.077 %
   null). A frame-pointer *caller* attribution is not a *copy* attribution: that
   caller holds three copies with different trip counts and only one was
   removable.
2. **`#[inline]` on `intra_avail`** — **REJECTED, and it is significantly
   SLOWER: +0.238 %, 4 of 24 rounds faster, p = 1.0000** against a −0.128 %
   null. `perf annotate` put that function's samples squarely in a six-push
   prologue and a run of `movl 0xNN(%rsp)` reloading its 21 stack-passed
   arguments, which reads as textbook call overhead. **It is not: inlining a
   206-instruction body at ten call sites costs more in I-cache and register
   pressure than the call it removes.** Band committed as
   `.inline_rejected.tsv`.

   **The transferable rule: `perf annotate` showing cost in a prologue or in
   argument marshalling does NOT imply inlining wins.** Those samples are where
   a call's unavoidable cost is *attributed*, not evidence that the call is
   avoidable. This cycle's three annotate-driven wins (KB-PERF-37/38/46) all
   replaced *loop bodies*; both annotate-driven attempts at *call overhead*
   failed.

## A THIRD rejection on this map, and it completes the pattern

**Hoisting `aom_quantize_b_no_qmatrix`'s loop invariants: +0.149 %, 8 of 24
rounds faster, p = 1.0000** against a −0.045 % null. REVERTED; band committed as
`.quantize_b_hoist_rejected.tsv`.

Every per-`ac` table lookup in that loop, plus both derived constants
(`round_power_of_two(round[ac], log_scale)` and the `dequant_v` rounding), is
genuinely loop-invariant and was genuinely being recomputed per coefficient. The
hoist was exact and byte-identical on four cells. It is still slower, because
what it actually did was **replace a two-element indexed lookup with a branch
per iteration** — `if is_ac { ac_const } else { dc_const }` — and on this loop
the branch costs more than the load it removed.

**The pattern across this session's three rejections is now clear and is worth
more than any one of them:**

| change | what it did to the work | result |
|---|---|---|
| `#[inline]` on `intra_avail` | moved a call's cost, removed none | **+0.238 %** |
| `quantize_b` invariant hoist | traded a load for a branch | **+0.149 %** |
| the reconstruct round trip | removed 2 copies on a minority of txbs | −0.147 %, **null** |
| **the 4x4 variance direct read** | **removed a copy + a walk on EVERY 4x4 unit** | **−0.827 %** |

**Changes that REMOVE work on a hot inner loop pay. Changes that RESHAPE how the
same work is reached — inlining, hoisting, re-scoping a lookup — do not, and
have measured positive (slower) twice.** That is the same distinction
KB-PERF-44/45 drew for allocation (removal pays; adding a mechanism does not),
arriving independently from the arithmetic side.

## `optimize_txb_core` is not a lever — but the RECORDED REASON is wrong

Every prior entry says "do not touch the trellis: it is FASTER than libaom"
(1.13x as a class; `optimize_txb_core` 417 ms against C's 426). **That reasoning
optimises the wrong objective.** Clause (4) is absolute encode time, not a
per-kernel ratio, and `optimize_txb_core` is **the largest single symbol in the
port at 13.73 % = 417 ms**. A 5 % improvement there would be 21 ms — more than
most landings in this cycle. "We already beat C here" is not an argument that no
time can be recovered.

So it was checked properly, and the conclusion survives on **measured** grounds:

* **the cost is diffuse.** 1495 instructions; the hottest single instruction is
  **1.49 %** of the symbol, and the top twenty sum to under 20 %. There is no
  block to attack — the samples are spread over bounds checks against spilled
  slice lengths (`cmpq 0x718(%rsp)`, `cmpq 0x80(%rsp)`) and stack-array reads.
* **the oversized scratch, which looked like the obvious win, is not one.**
  `let mut levels = [0u8; TX_PAD_2D]` compiles to a **1312-byte `memset` at
  function entry** (`movl $0x520, %edx; xorl %esi, %esi; callq`), sized for 32x32
  whatever the actual transform — textbook KB-PERF-11 / KB-PERF-2-lever-3a shape,
  and KB-PERF-11 explicitly says to grep the tree for it. **But it carries only
  0.22 % of the symbol's samples = ~1 ms**, because `optimize_txb_core` runs per
  WINNING transform under `perform_block_coeff_opt`, not per candidate. The
  count, not the shape, decides.

**So: leave it alone — but for the reason that its cost is diffuse and its
obvious scratch is cold, not because its ratio to libaom is good.** A future
session comparing ratios would keep skipping the port's biggest symbol; one
comparing milliseconds would rightly look, and should find this note.

This is the KB-PERF-47 lesson again from the other side: there, a wrong reason
kept a real row closed; here, a wrong reason would have kept a *correct* closure
resting on an argument that does not support it.

## What the table says to do next, in order

1. **variance family, +71.4 ms at 4.91x** — the largest addressable row that has
   never had a landing. C size-specialises; the port has one generic body.
   KB-PERF-47 rejected `#[autoversion]` here, but that rejected ONE mechanism
   (per-call dispatch not amortised on small blocks), not the row.
2. **quantize_b, +28.7 ms at 11.89x** — the worst ratio of any addressable row
   and the only one with **no SIMD at all**. Fully specified already: the
   rewrite must process in RAW order, which is provably value-identical for the
   coefficients but changes how `eob` is derived, and `eob` is the one
   order-sensitive output in this family (KB-12). Needs a bite proof aimed at
   the eob.
3. **loopfilter, +44.9 ms at 4.67x** — `define(i32x4)` is a 128-bit type holding
   4 lanes where libaom's `_sse2` holds 16 `u8`. Its ops are all in magetypes
   for narrow types, unlike wiener and quantize_fp. But closing it properly also
   needs the frame driver to batch adjacent segments (`_dual`/`_quad`), so it is
   a programme, not a lever.
4. **directional intra z2, +47.3 ms** — the left half is a true gather
   (`base_y` is not affine in `c`), which is why KB-PERF-4 vectorised only the
   above half.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, x86-64. Flat
`perf record` shares, so an inlined callee is attributed to its caller — the
`trellis` and `edge assembly` rows in particular are sinks. The two italic rows
are excluded from the addressable subtotal: the trellis runs at 1.13x and is
faster than libaom's per its own three prior measurements, and the intra-CNN is
a PARITY item (KB-41 root #27) whose cost is speed-invariant.
