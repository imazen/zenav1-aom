# Apple M4 Pro (aarch64 macOS) — Hadamard NEON tier measurements

- Host: Apple M4 Pro, macOS 27.0 (BuildVersion 26A428), CLT SDKs up to MacOSX26.5
- Source: working copy after `main | investigate: measure Mac ARM AOM DSP tiers`
  (31b341d27bc5c37bb960824c831a3903a3d08bcd) + the NEON hadamard tier of this
  commit; oracle `upstream/` = libaom v3.15.1 `44d0a577`
- Bench harness: `crates/aom-dsp-bench/benches/dsp_kernels.rs`, group
  `hadamard`, added in this commit. `WORK_PX = 1<<16` elements per batch;
  `reps = WORK_PX / (n*n)` calls of `hadamard_*_into` / `hadamard_lp_*` per
  call batch. zenbench tuning: 12–200 rounds, 250 ms warmup, 10 s max time.
- Command:
  ```sh
  TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 4 -- \
    cargo bench -p zenav1-aom-dsp-bench --bench dsp_kernels -- \
    --group=hadamard --format=llm
  ```
- Full log: `~/tmp/zenav1-aom-mac-hadamard.log` (not committed — machine-local)
- run-heavy summary: `done rc=0 38s | peak-RSS 0.26GiB | min-avail 10615MiB |
  peak-load 16.87` (mem-cap=NONE on macOS; RSS guard 12884901888 B)
- Units: mean wall time per 64 Ki-element batch (µs), zenbench paired-arms
  `native_simd` (runtime dispatch) vs `forced_scalar` (`AOM_FORCE_SCALAR`
  semantics — the exact code aarch64 ran before this change; it is the
  pre-change baseline, not a different build).

## Results

| cell | native_simd (NEON) | forced_scalar | scalar slower | 95% CI | sig |
|---|---|---|---|---|---|
| fp_08x08 | 6.28 µs | 30.87 µs | +397.98 % | [+395.45, +400.47] | yes |
| fp_16x16 | 18.01 µs | 50.28 µs | +179.24 % | [+162.82, +196.65] | yes |
| fp_32x32 | 18.02 µs | 44.05 µs | +147.64 % | [+145.06, +150.34] | yes |
| lp_08x08 | 5.38 µs | 60.56 µs | +983.98 % | [+980.92, +987.19] | yes |
| lp_16x16 | 8.69 µs | 73.48 µs | +714.65 % | [+707.11, +722.80] | yes |

Speedups (scalar / native): fp_08x08 4.9×, fp_16x16 2.8×, fp_32x32 2.4×,
lp_08x08 11.3×, lp_16x16 8.5×.

## What the NEON tier covers

`hadamard_8x8_into` (`aom_hadamard_8x8_neon`), `hadamard_lp_8x8`
(`aom_hadamard_lp_8x8_neon`) and the `hadamard_lp_16x16` combine pass
(`aom_hadamard_lp_16x16_neon`'s second half) — verbatim transcriptions of
`upstream/aom_dsp/arm/hadamard_neon.c` plus `transpose_elems_inplace_s16_8x8`
from `transpose_neon.h`. `hadamard_16x16`/`_32x32`/`lp_8x8_dual` get NEON for
free through composition; their combine passes remain scalar (they are
element-wise i32 loops LLVM already vectorises — upstream's fused
combine+reorder was not needed to reach these numbers).

Departure from upstream noted in `crates/aom-dsp/src/dist/hadamard.rs`:
the lp-16 combine uses `vshrq_n_s16` over the wrapped `vaddq`/`vsubq`
(the port's wrap-then-shift on the full domain, same as the v3 tier's
`srai(add, 1)`) where upstream uses `vhaddq`/`vhsubq`; identical on the
reachable lp domain (|a ± b| ≤ i16::MAX holds for the 9-bit diff inputs).

## Exact-output gates

- `hadamard_diff::hadamard_satd_byte_identical` — **pass** (aarch64 NEON vs
  exported C, all permutations; non-vacuity asserted by the test).
- `highbd_hadamard_diff::highbd_hadamard_satd_byte_identical` — **pass**.
- `block_error_diff::{block_error_differential, highbd_block_error_differential}`
  and `block_error_qm_diff::*` — **pass**.

## Limitations

- `fp_16x16` native CV was 41 % (round noise on this run; CI still
  significant and 2.8× wide).
- 16×16/32×32 fp combine passes remain scalar — a visible further win if the
  fp combine is later NEON'd the way the lp combine was.
- Environment note: this host's `cc` resolves to Homebrew LLVM clang, which
  derives a nonexistent `MacOSX27.sdk` sysroot on macOS 27 and cannot build
  the sys-ref C oracle. Tests were run with `PATH=/usr/bin:$PATH` so
  `pick_c_compiler` finds Apple clang; `upstream/` was also checked out to
  the pinned `44d0a577` (was v3.14.1 locally).
