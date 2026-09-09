# KB-PERF-21 — the 8x8 Hadamard was scalar; raw intrinsics unblocked it — LANDED, −0.470 % at the shipping preset

**2026-09-09.** Byte-identical; **−0.470 %** at 1024x1024 cq27 `--cpu-used 3`,
**24 of 24 rounds faster, p < 0.0001**, against a same-binary null of **+0.143 %**
(i.e. the null runs the OTHER way, so the effect is if anything understated).

## Why it was reachable at all — the block was a wrong inference, not a wrong fact

`encoder_simd_lane_width_audit_2026-09-09.md` ranked hadamard at
**39.5 ms against libaom's 14.7** and KB-PERF-10 had explained why it was stuck:
*"an in-register 8x8 i16 transpose needs a shuffle magetypes does not have."*
That is TRUE — verified again here against the pinned **0.9.29**: interleave and
transpose exist for `f32` only, for no integer type.

But `archmage::intrinsics::x86_64` re-exports `core::arch`, and a magetypes
`_v3` body carries `target_feature(avx2)`, so under target-feature-11 the unpack
family is callable **with `#![forbid(unsafe_code)]` still in force**
(`encoder_intrinsics_unblock_2026-09-09.md` is the probe). Loads and stores go
through `safe_unaligned_simd`'s reference-based forms, so nothing here is
`unsafe` either.

**This is the first landing to mix raw intrinsics into a magetypes body**, and
it is the pattern for the two remaining blocked levers (`wiener` +34.7 ms,
`quantize_fp` +54.6 ms).

Like for like at the shipping preset:

| | port | C | ratio | gap |
|---|---:|---:|---:|---:|
| `hadamard_8x8` | **31.1 ms** | `aom_hadamard_8x8_sse2` 7.9 | **3.94x** | **+23.2** |
| all sizes | 39.5 | 14.7 | 2.69x | +24.8 |

`hadamard_16x16` and `_32x32` call `hadamard_8x8`, so they inherit the fix.

## The kernel — libaom's own shape

The scalar core is two passes of `hadamard_col8` with the trailing transpose
FUSED into the output indexing (KB-PERF-10). Vectorised with lane = column:

* eight **contiguous** row loads (`_mm_loadu_si128` on `&[i16; 8]`);
* pass 1 is vertical — the butterfly network of `hadamard_col8` applied across
  eight `__m128i`, `_mm_add_epi16`/`_mm_sub_epi16`, same order and same output
  permutation;
* pass 2 is HORIZONTAL in that layout, which is exactly what the transpose is
  for: one three-stage in-register 8x8 i16 transpose, then the identical network
  again;
* widen with `_mm256_cvtepi16_epi32` and store eight `i32` per row.

Emitted code, from the shipped binary:

    24 vpaddw + 24 vpsubw   the two butterfly passes
    24 vpunpck{lo,hi}{wd,dq,qdq}   the 8x8 transpose
     8 vpmovsxwd            the i16 -> i32 widening

## Bit-exactness

`_mm_add_epi16` / `_mm_sub_epi16` wrap, which is precisely `i16::wrapping_add` /
`wrapping_sub` — the scalar core's own operations. The network is the same
network in the same order, including `hadamard_col8`'s output permutation
(`o[0], o[7], o[3], o[4], o[2], o[6], o[1], o[5]`). The transpose only MOVES
lanes. So there is no bound and no numeric argument to make.

**That is deliberately not left to inspection.** KB-12 is the standing proof
that a lost transpose in this exact kernel perturbs the **`eob` and nothing
else** — rate, distortion and skippability are all order-invariant — so it reads
as a genuine RD near-tie and survived four localization passes. The guard is the
differential against the real exported C.

## Correctness, and the bite proof is maximally asymmetric

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in BOTH dispatch modes** (default and
  `AOM_FORCE_SCALAR=1` — the pin routes to `hadamard_8x8_avx2_scalar`, which
  declines to the untouched scalar core).
* **Bite proof: swapping two lanes of the transpose's last stage fails EXACTLY
  ONE test — `hadamard_diff::hadamard_satd_byte_identical`, against the real
  exported C — while 394 stay green.** One perturbation, one failure, in the one
  differential that owns this kernel.

## The band

Two sha256-distinct binaries from one tree, arms **rotated**, same-binary null:

| arm | median | paired median | rounds faster | p |
|---|---:|---:|---:|---:|
| base (KB-PERF-20 HEAD) | 3517.25 ms | — | — | — |
| baseB (null) | 3520.60 | +0.143 % | 5/24 | 0.0066 |
| **hadamard AVX2** | **3500.53** | **−0.470 %** | **24/24** | **<0.0001** |

Raw spreads 0.6–1.5 %. Measured −0.470 % is ~16.5 ms against the +23.2 ms
like-for-like gap — **71 % of it**, the best conversion ratio of the three
kernels tried this session (filter-intra got ~55 %, the edge filter ~25 %).

**Note the null is +0.143 % at p=0.0066**, i.e. this band's two identical
binaries differ significantly. That is the position/copy systematic
`encoder_rotate_reverify_2026-08-03.md` records; it runs opposite to the
measured effect here, so it cannot manufacture it.

## Not covered

* **`hadamard_4x4` is untouched** (3.5 ms vs `aom_hadamard_4x4_sse2` 2.2) — its
  `hadamard_col4` widens to i32 and shifts before narrowing, a different proof.
* **aarch64 and wasm get nothing**: the body is `cfg(target_arch = "x86_64")`
  and everything else routes to the unchanged scalar core. A NEON twin would use
  `vzip`/`vtrn` and is its own landing.
* The `lp` (low-precision) Hadamard family used by the nonrd estimate arm is a
  separate set of kernels and is not touched here.
* One box, one content class, bd8 4:2:0, cq27, `--cpu-used 3`.
