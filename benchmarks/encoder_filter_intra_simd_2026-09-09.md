# KB-PERF-20 — filter-intra's TAPS were scalar where libaom has SSE4.1 — LANDED, −0.372 % at the shipping preset

**2026-09-09.** Byte-identical; **−0.372 %** at 1024x1024 cq27 `--cpu-used 3`,
**22/24 rounds faster, p < 0.0001**, against a same-binary null of −0.041 %.

## How it was chosen — the first lever picked BY the lane-width audit

`encoder_simd_lane_width_audit_2026-09-09.md` compared every operation where
port and libaom are BOTH vectorised and found filter-intra to be one of only two
where the port was still **scalar against a real C SIMD kernel**. KB-PERF-11 had
named it in as many words when it removed that function's 2178-byte scratch
memset: *"the taps themselves are still scalar — 8 outputs x 7 taps per 4x2
block is the natural 8-lane dot product libaom's SSE4.1 vectorises — its own
landing."*

**A correction to that audit, made here.** Its table read filter_intra at
+47.3 ms / 7.50x, and that row was **overstated by an asymmetric pairing**: the
port side summed `filter_intra_predict_high` AND `highbd_filter_intra_edge`
while the C side counted only `av1_filter_intra_predictor_sse4_1`. Like for
like at the shipping preset:

| | port | C | ratio | gap |
|---|---:|---:|---:|---:|
| predictor | `filter_intra_predict_high` 31.1 ms | `av1_filter_intra_predictor_sse4_1` 7.3 | **4.26x** | **+23.8** |
| edge filter | `highbd_filter_intra_edge` 20.3 | `av1_filter_intra_edge_sse4_1` 5.6 | 3.63x | +14.7 |

This landing is the **predictor** half. The edge filter is still scalar and is
the named follow-up.

## The kernel

One 4x2 sub-block produces **eight** outputs and the vector is **eight** i32
lanes, so lane `k` is output `k` — **no idle lanes**, which is why this is a
scalar-to-vector win rather than a lane-width one. Each output is a 7-tap dot
product over the SAME seven neighbours, so the neighbours splat and the taps
are per-lane constants:

    acc = sum_j  splat(p[j]) * TAPS_T[mode][j]        (j = 0..6)

**56 scalar multiplies and 56 adds become 7 vector multiply-adds.** `TAPS_T` is
a `const fn` transpose of the same `FILTER_INTRA_TAPS` the scalar path reads, so
the two cannot drift (`taps_transpose_matches_the_scalar_table` asserts it
element by element).

**Bit-exact by construction, with no bound and no runtime gate.** Every lane
runs the identical `i32` expression the scalar loop runs — same products, same
accumulation order (`j` ascending), same `(acc + 8) >> 4` arithmetic shift, same
`clamp(0, max_v)`. Nothing is reassociated, nothing is narrowed, so the kernel
is correct at every bit depth rather than admitted by an audit.

## The tier is held across the whole loop nest, verified in the disassembly

The `#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]` body contains the
ENTIRE `r`/`c` loop nest, so `_v3` carries `target_feature(avx2)` throughout and
dispatch happens **once per predict call**, not per 4x2 block. Confirmed by
`objdump` on the shipped binary:

| | instructions | ymm | `vpmulld` |
|---|---:|---:|---:|
| before (scalar) | 547 | **0** | — |
| after (`_v3`) | 355 | **69** | **7** — exactly the seven tap multiplies |

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` differentials **395/395 in BOTH dispatch modes** (default
  and `AOM_FORCE_SCALAR=1`), `--lib` 51/51.
* **Bite proof, asymmetric**: perturbing the first tap fails
  `filter_intra_diff::filter_intra_matches_c` (against the **real exported C**),
  `build_filter_intra_diff::build_filter_intra_matches_c` and
  `intra_lowbd_diff` — while **391 stay green**. So the kernel is genuinely
  reached and load-bearing.

## The band

Two sha256-distinct binaries from one tree, arms **rotated**, same-binary null:

| arm | median | paired median | rounds faster | p |
|---|---:|---:|---:|---:|
| base | 3531.70 ms | — | — | — |
| baseB (null) | 3527.34 | −0.041 % | 14/24 | 0.5413 |
| **filter-intra SIMD** | **3520.30** | **−0.372 %** | **22/24** | **<0.0001** |

Raw spreads 0.8–1.6 %. The measured −0.372 % is ~13 ms against the +23.8 ms
like-for-like gap, i.e. **roughly half of it** — the remainder is the
three-row bookkeeping, the neighbour gather and the stores, which this landing
does not touch.

## Not covered

* **`highbd_filter_intra_edge` is still scalar** (+14.7 ms, 3.63x vs
  `av1_filter_intra_edge_sse4_1`). It is a 5-tap sliding filter over a 1-D
  array — contiguous loads, no transpose — so it is the cleanest remaining
  member of the audit's list.
* The neighbour gather (`p[0..7]`) is still seven scalar loads per 4x2 block.
* One box, one content class, bd8 4:2:0, cq27, `--cpu-used 3`, x86-64.
  **Reachability caveat from KB-PERF-11 still applies and is why this was
  measured at s3**: filter-intra is 0 % of leaves at `--cpu-used 6` and would
  measure nothing there.

## Recorded, unexplained

Under BOTH bite-proof perturbations this session — this one, and the unrelated
forward-row-pass one in `crates/aom-dsp/src/transform/simd/mod.rs` —
`cdef_find_dir_simd_diff` ALSO failed, while passing 395/395 on the clean tree
before and after each. Two different perturbations in two different files, so it
is reproducible rather than a one-off, and it has no relation to either change.
Worth a look: a test that fails whenever an unrelated test in the same binary
fails is not measuring what it claims.
