# Gate-3 transform-SIMD — callgrind Ir deltas (2026-07-17)

Whole-encode / whole-decode instruction-reference (Ir) counts measured with
valgrind callgrind on the two canonical Gate-3 profile cells, **port side**,
before and after the transform-SIMD stack (lane-batched 1-D kernels + the four
2-D vector passes, AVX2 / `X64V3` tier). Ir is deterministic, so these are
load-independent (the box was busy; wall-clock is a separate quiet-box run).

The "before" is `origin/main` @ `fa1c55c` — it already carries the earlier
SIMD kernels #1–#5 (quantize_fp, cdef_filter, wiener_convolve, txb_init_levels,
highbd_variance64) but **not** the transform stack, so the delta below is the
transform SIMD alone. The all-scalar column is the perf-track origin
(`057bde2`, no SIMD dispatch) for cumulative context.

## Measured Ir

| cell (iters) | all-scalar `057bde2` | before (`fa1c55c`, #1–#5) | after (transform SIMD) | Δ this stack | Δ cumulative |
|---|---|---|---|---|---|
| `enc_s0_128_cq32` (6)  | 155,210,572,291 | 142,723,403,866 | **99,886,396,237** | **−30.0 %** | **−35.6 %** |
| `dec_352x288_q32` (30) |   3,656,695,999 |   2,552,717,598 |  **2,139,898,111** | **−16.2 %** | **−41.5 %** |

- Encode saves **42.84 B Ir** (−30.0 %) — the RD search runs both the forward
  transform (analysis) and the inverse transform (reconstruction) per candidate,
  so both the fwd col/row and inv col/row passes fire heavily.
- Decode saves **0.413 B Ir** (−16.2 %) — inverse transform only.

## SIMD confirmed live (callgrind_annotate, after)

The AVX2 tier fires — the vector passes and the 1-D lane dispatch appear in the
profile with their AVX2 intrinsics attributed:

- encode: `simd::__arcane_inv_col_pass` 1.22 %, `simd::run_inv1d` (avx2) 1.26 %,
  `simd::run_fwd1d` (avx2) 1.05 %, `__arcane_inv_row_pass` 0.45 %,
  `__arcane_fwd_col_pass` 0.43 %, `__arcane_fwd_row_pass` 0.34 %.
- decode: `simd::__arcane_inv_col_pass` 0.80 %, `simd::run_inv1d` (avx2) 0.90 %,
  `__arcane_inv_row_pass` 0.28 %.

## Remaining scalar transform work (next follow-ups, by profile)

Still scalar (the passes gate on `col_n % 8 == 0` / `row_n % 8 == 0`, so any
W=4 or H=4 block falls to the per-column scalar loop): the size-4 kernels
`av1_fdct4` (0.80 %), `av1_fadst4` (0.74 %), `av1_iadst4` (0.60 %),
`av1_idct4` (0.53 %) are the largest residual transform cost in encode — the
W=4/H=4 half-vector arms are the obvious next chunk. AVX-512 (`X64V4`, 16-lane,
native `vpsraq`) is a further follow-up; the current stack is AVX2 only and runs
its `X64V3` path on this AVX-512 host.
