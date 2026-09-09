# KB-PERF-10 — the Hadamard/SATD kernels were bounds-check bound, not arithmetic bound

**Landed 2026-09-09.** Byte-identical output; **−0.94 %** at 1024x1024 and
**−0.74 %** at 512x512, each against a same-binary null, each over a rotated
interleaved band. Ratio at the 1 MP profile cell **2.488x → 2.464x**.

## How it was found, and the correction that came with it

The 1 MP re-profile (`encoder_x86_reprofile_1024_2026-09-09.md`) ranked
**rd-driver at +1023 ms / 16.1 % of the gap**, unprofiled below class level.
Drilling in produced two named counterparts with enormous ratios:

| port | ms | C | ms | ratio |
|---|---:|---|---:|---:|
| `intra_model_rd_y` | 342.8 | `intra_model_rd.constprop.0` | 23.4 | 14.6x |
| `txfm_rd_in_plane_intra` | 318.0 | `av1_txfm_rd_in_plane` + `block_rd_txfm` | 84.9 | 3.7x |

**Neither is a lever, and playbook §14 is why.** The port has **no `subtract`
and no `satd` symbol at all** in the profile, while C has
`aom_subtract_block_sse2` 25.5, `_avx2` 15.1, `av1_subtract_block` 7.1,
`av1_subtract_txb` 5.0, `aom_satd_avx2` 11.7 and `av1_quick_txfm` 5.0. Those
callees are **inlined into `intra_model_rd_y`** on the port side and dispatched
on C's, so their self-costs are not comparable quantities. The 14.6x is an
artifact of inlining, not a gap. Any lever costed off that row would have been
costed off C's *dispatch overhead*.

What survived the check is the one member of that neighbourhood that **is a
separate symbol on both sides**:

| | port | C |
|---|---:|---:|
| `hadamard_col8` | 126.0 | — (inlined in C's own kernels) |
| `hadamard_8x8` | 39.2 | `aom_hadamard_8x8_sse2` 13.4 |
| `hadamard_4x4` | 33.0 | `aom_hadamard_4x4_sse2` 7.1 |
| `hadamard_16x16` | 18.6 | `hadamard_16x16_avx2` 4.2 |
| `hadamard_32x32` | 4.1 | `aom_hadamard_32x32_avx2` 7.1 |
| **total** | **220.9** | **~31.8** |

**6.9x.** `crates/aom-dsp/src/dist/hadamard.rs` had no SIMD path of any kind —
the same shape as KB-PERF-6's loop-restoration finding.

## Why this is NOT a SIMD landing, and what blocks one

A 2-D Hadamard is two passes with a transpose between them, and only one pass
can be lane-parallel at a time. Working the index algebra: with lanes = source
column, pass 1's butterflies across the eight loaded rows are lane-parallel and
pass 2's are horizontal-within-vector; choosing the other assignment swaps
which pass is which. libaom's SSE2 resolves this with
`_mm_unpacklo_epi16`-family shuffles.

**magetypes has no shuffle, permute or unpack in the pinned 0.9.28 or in
0.9.29 — only `blend`** (verified first-hand for KB-PERF-8's gather question and
re-checked here). So an in-register 8x8 transpose is not expressible in this
vocabulary, and routing it through memory costs most of what the vectorized
butterflies would win. That is the honest reason this is not a SIMD landing.

## What the cost actually was

`hadamard_col8` carries 24 `wrapping_add`/`wrapping_sub` — and took **126 ms**,
more than the three 2-D drivers combined. It was declared

    fn hadamard_col8(src: &[i16], off: usize, stride: usize) -> [i16; 8]

and read `src[off + k * stride]` eight times. A **slice** parameter means a
bounds check per element against a runtime length: **128 bounds-checked strided
loads per 8x8 block** (the block runs it sixteen times), an `[i16; 8]` returned
by value into a `copy_from_slice`, and a runtime length that stops LLVM keeping
intermediates in registers or vectorizing the caller.

The fix is a signature, not an algorithm: `hadamard_col8(s: [i16; 8])` and
`hadamard_col4(s: [i16; 4])` take already-gathered values, the drivers read
rows into `[[i16; 8]; 8]` with one bounds check per row (eight contiguous
16-byte copies, which vectorize), and every subsequent index is into a
fixed-size array.

## The trailing transpose is FUSED, not dropped — and that distinction is KB-12

C writes `buffer[idx*8 + k] = A[idx][k]` where `A[idx] = col8(source column
idx)`. Its second pass reads `buffer[idx + k*8]`, which in that layout is
`A[k][idx]`, so `B[idx] = col8([A[k][idx] for k])` and it writes
`buffer2[idx*8 + k] = B[idx][k]`. The final *"Extra transpose to match SSE2
behavior"* (`aom_dsp/avg.c:232-236`) is `coeff[i*8 + j] = buffer2[j*8 + i]` —
which is exactly `B[j][i]`.

So emitting `B[j][i]` in place **is** the transpose. It is not an omission of
one, and the derivation is written out at the function so a later reader cannot
mistake it for a dropped step. **KB-12 is the standing reason to be explicit
here**: that transpose being silently absent moved the `eob` and nothing else —
rate, distortion and skippability are all order-invariant — and read as a
genuine RD near-tie across four separate localization passes before anyone
found it.

## Correctness

Same arithmetic, `wrapping` included; only the indexing changed. Gated by the
two differentials against the **real exported C**:
`hadamard_diff::hadamard_satd_byte_identical` and
`highbd_hadamard_diff::highbd_hadamard_satd_byte_identical`.

**BITE PROOF, asymmetric.** Perturbing the fused transpose in `hadamard_8x8`
alone (`b[j][i]` → `b[i][j]`) fails `hadamard_diff` while
`highbd_hadamard_diff` — a separate, untouched code path — stays green. So the
lowbd path is genuinely reached and the green is not vacuous.

## Measurement

Two sha256-distinct binaries from one tree, arms **rotated** each round
(playbook §6), a same-binary null (`baseB`) in every band, one encode per arm
per round. Output byte-identical at **39,694 B** on both arms.

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 10301.7 ms | 10204.1 ms | **−0.94 %** | **12/12** | 0.0005 | +0.08 %, 4/12, p=0.39 |
| 512x512 cq27 s0 | 3688.0 ms | 3660.1 ms | **−0.74 %** | **18/20** | 0.0004 | −0.08 %, 12/20, p=0.50 |

Raw spreads 0.4–1.9 %. Bands: `.band1024.tsv`, `.band512.tsv`.
**Ratio 2.488x → 2.464x** (C median 4140.9 ms over 5 invocations, same box).

**§14 again, at 2.4x.** The class row said ~190 ms addressable and the delta is
~97 ms at 1 MP. That is the right order and the reason is not mysterious: this
removed the *indexing overhead* around the butterflies, not the butterflies. The
remaining port-vs-C Hadamard gap is the arithmetic itself, and closing it needs
the shuffle vocabulary above.

## Not covered

* **The highbd Hadamard path is untouched** (`highbd_hadamard_8x8/16x16/32x32`,
  which work in `i32`) — it has the same slice-parameter shape and was left
  alone so the bite proof stays asymmetric. It is the obvious follow-up.
* `hadamard_16x16`/`32x32`'s cross-quadrant combine loops still index flat
  `[i32; 256]` / `[i32; 1024]` arrays; those are already bounds-check-free
  (fixed-size arrays) and were not the cost.
* One box, one content class, one speed, `--cpu-used 0`, x86-64.

## Gate status

`just gate-landing` green in full — `test-next` 1503/1503,
`test-next-scalar` 1503/1503, `census-gate` 4/4, `test-whereat` 4/4, all exit 0.
