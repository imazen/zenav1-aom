# Gate-3 decode — Lever 1: fixed-array bounds-check elimination (2026-07-22)

Task #37, DECODE-PERF. Lever 1 = the claudehints "fixed-array BCE" pattern:
`try_into::<&[T; N]>()` / exact-length reslice at a function/loop boundary so
LLVM proves every interior index in-bounds and drops the per-access bounds
branch. Bit-exact BY CONSTRUCTION — same values, no panic branches removed from
the arithmetic, only never-taken bounds checks.

## Finding: most hot-path bounds checks were already eliminated; two hot self-contained kernels still had them

Re-profiling first showed **no `panic_bounds_check` symbol anywhere** and the
aggregate bounds-check machinery (`core/src/slice/index.rs`) at **< 1 %** of both
q00 and q32 — the prior perf chunks (a00aa516's `update_cdf`/`decode_cdf_q15`
reslices, the force-inlined txb-ctx helpers, the SIMD `try_into` stores) had
already picked the low-hanging fruit. So Lever 1's *aggregate* headroom is small.

But that metric understates per-kernel wins: the residual checks are **inline**
(folded into the kernel's own lines, no separate panic symbol). Converting the
two hottest self-contained kernels that still carried them gave real, measured
reductions:

1. **`av1_highbd_iwht4x4_16_add`** (the lossless 4×4 Walsh–Hadamard inverse —
   dominant transform on q00): `input: &[i32]` → `let input: &[i32; 16] =
   input[..16].try_into().unwrap()` at the top. The 16 interior `input[i..i+12]`
   reads become check-free. (The `out` intermediate was already a fixed
   `[i32; 16]`.) The caller always passes a `[0i32; 16]` dqcoeff block, so the
   one boundary check never fails.
2. **`dequant_txb`**: `qcoeff`/`dqcoeff` resliced to `[..area]` once, so every
   `qcoeff[pos]`/`dqcoeff[pos]` with `pos < area` is provably in-bounds.

## Measured (callgrind Ir, `gate3_profile dec port <cell> 3`; baseline = post-Lever-2)

| cell | metric | baseline | Lever 1 | Δ |
|---|---|---:|---:|---:|
| dec_352x288_q00 (lossless) | program total | 506,476,248 | 504,329,250 | **-0.42%** |
| dec_352x288_q32 | program total | 238,445,542 | 237,647,788 | **-0.33%** |
| dec_352x288_q00 | `av1_highbd_iwht4x4_add` self-Ir | 9,979,732 | 9,019,772 | **-9.6%** |
| dec_352x288_q00 | `dequant_txb` self-Ir | 7,694,144 | 7,063,904 | -8.2% |
| dec_352x288_q32 | `dequant_txb` self-Ir | 2,964,804 | 2,219,008 | **-25.2%** |

The WHT win is q00-specific (only lossless content uses the 4×4 WHT); the
dequant reslice helps every cell (it is the per-coefficient dequant loop).

## NOT converted (deliberately)

- `filter_intra_predict_high` (1.53 % on q00): its `[[u16;33];33]` buffer safety
  rests on `bw,bh <= 32`, which is a `debug_assert` (gone in release) and the
  odd-step loop hides the true max from LLVM. Giving LLVM the bound cleanly would
  need `unsafe` (forbidden) or a hard `assert!` that adds a panic path to the
  hardened decoder (compliance §5 panic-freedom) — not worth 1.5 % of one cell.
- Coeff scan `sc[c]` reslice: `tcoeff[pos]` (pos = a scan-table value) can't be
  proven in-bounds without asserting the scan-value range, and the `sc[..eob]`
  reslice shifts a panic condition; the residual per-check cost there is a
  fraction of a percent. Left as-is to keep the panic surface unchanged.

## Bit-exactness proof

`inv_txfm2d_diff` (3) + `txfm2d_diff` (2) + `dequant_txb_diff` (1) + `txb_diff`
(1) + `read_txb_full_diff` (4) all green vs the REAL C kernels; full decode
conformance `conformance_single_frame_intra_byte_identical_to_c_and_golden`
(240 vectors, byte-identity + golden MD5) + `high_qindex_gt64_partition_byte_identical_to_c`
green. Corpus at 240.
