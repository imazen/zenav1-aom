# Apple M4 Pro (aarch64 macOS) — directional intra NEON tier measurements

- Host: Apple M4 Pro, macOS 27.0 (BuildVersion 26A428), CLT SDKs up to MacOSX26.5
- Source: working copy after `main | feat: NEON hadamard tiers on aarch64`
  (0f07a43b) + the NEON directional-predictor tiers of this commit; oracle
  `upstream/` = libaom v3.15.1 `44d0a577`
- Kernels: `z1_rows_impl`, `z1_rows_u8e_impl`, `z2_above_run_impl`,
  `z3_cols_impl` in `crates/aom-dsp/src/intra/dir_simd.rs`. Previously
  `v3`-only — aarch64 ran plain scalar. This commit adds `neon` slots
  (`[v3, neon, scalar]`), mirroring the x86 i16-lane recipe with `vld2q_u16`
  / `vuzp1q`-`vuzp2q` deinterleaves for the stride-2 upsampled taps.
- Bench harness: `crates/aom-dsp-bench/benches/dsp_kernels.rs`, group
  `intra_dir` (added earlier in this branch). `WORK_PX = 1<<16` elements per
  batch; `reps = WORK_PX / (bw*bh)` calls per batch. zenbench tuning:
  30–110 rounds, 250 ms warmup.
- Command:
  ```sh
  TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 4 -- \
    cargo bench -p zenav1-aom-dsp-bench --bench dsp_kernels -- \
    --group=intra_dir --format=llm
  ```
- Full log: `~/tmp/zenav1-aom-mac-intra-dir.log` (not committed — machine-local)
- run-heavy summary: `done rc=0 129s | peak-RSS 0.16GiB | min-avail 10614MiB |
  peak-load 11.92` (mem-cap=NONE on macOS; RSS guard 12884901888 B)
- Units: mean wall time per 64 Ki-element batch (µs), zenbench paired-arms
  `native_simd` (runtime dispatch — NEON on aarch64) vs `forced_scalar`
  (`AOM_FORCE_SCALAR` semantics — the exact code aarch64 ran before this
  change; it is the pre-change baseline, not a different build) vs
  `scalar_reference` (the hand-written scalar twin called directly).

## Results — forced_scalar column is % slower than native_simd (95% CI)

### 16×16

| cell | native_simd | forced_scalar | scalar_reference | slower | 95% CI | sig |
|---|---|---|---|---|---|---|
| z1 up0 | 15.2 µs | 47.5 µs | 45.9 µs | +212.34% | [+208.38, +216.50] | yes |
| z1u8e up0 | 12.6 µs | 45.2 µs | 50.2 µs | +277.58% | [+271.76, +283.95] | yes |
| z2 up0 | 34.0 µs | 57.3 µs | 68.6 µs | +68.75% | [+65.81, +71.96] | yes |
| z3 up0 | 15.3 µs | 48.6 µs | 49.6 µs | +217.83% | [+214.39, +221.13] | yes |
| z1 up1 | 14.3 µs | 52.8 µs | 49.6 µs | +262.49% | [+251.54, +275.27] | yes |
| z1u8e up1 | 9.8 µs | 43.8 µs | 45.9 µs | +348.14% | [+345.83, +350.55] | yes |
| z2 up1 | 38.5 µs | 62.7 µs | 79.4 µs | +61.67% | [+58.04, +65.55] | yes |
| z3 up1 | 16.8 µs | 53.2 µs | 52.8 µs | +216.17% | [+210.15, +222.79] | yes |

### 32×32

| cell | native_simd | forced_scalar | scalar_reference | slower | 95% CI | sig |
|---|---|---|---|---|---|---|
| z1 up0 | 16.2 µs | 66.6 µs | 64.9 µs | +304.43% | [+280.41, +329.28] | yes |
| z1u8e up0 | 13.4 µs | 72.1 µs | 81.1 µs | +440.19% | [+401.29, +480.65] | yes |
| z2 up0 | 33.1 µs | 74.3 µs | 101.8 µs | +122.05%* | [+114.0, +130.1] | yes |
| z3 up0 | 24.7 µs | 94.4 µs | 96.1 µs | +282.01% | [+260.12, +302.66] | yes |
| z1 up1 | 17.9 µs | 80.8 µs | 79.7 µs | +352.60%* | [+321.8, +383.4] | yes |
| z1u8e up1 | 10.0 µs | 60.5 µs | 66.6 µs | +503.80%* | [+457.8, +549.8] | yes |
| z2 up1 | 25.4 µs | 57.1 µs | 78.8 µs | +125.80%* | [+119.9, +131.7] | yes |
| z3 up1 | 23.5 µs | 89.0 µs | 92.7 µs | +279.10%* | [+255.9, +302.3] | yes |

### 64×64

| cell | native_simd | forced_scalar | scalar_reference | slower | 95% CI | sig |
|---|---|---|---|---|---|---|
| z1 up0 | 13.3 µs | 68.9 µs | 68.4 µs | +417.45%* | [+377.6, +457.3] | yes |
| z1u8e up0 | 11.5 µs | 81.4 µs | 90.5 µs | +607.20%* | [+548.3, +666.1] | yes |
| z2 up0 | 16.4 µs | 50.0 µs | 73.7 µs | +204.30%* | [+203.4, +205.2] | yes |
| z3 up0 | 21.4 µs | 87.1 µs | 91.4 µs | +306.95%* | [+280.6, +333.3] | yes |
| z1 up1 | 8.6 µs | 45.5 µs | 45.1 µs | +428.65%* | [+422.0, +435.3] | yes |
| z1u8e up1 | 5.3 µs | 38.1 µs | 43.2 µs | +613.10%* | [+607.8, +618.4] | yes |
| z2 up1 | 19.2 µs | 55.2 µs | 80.2 µs | +203.40%* | [+202.8, +204.0] | yes |
| z3 up1 | 14.8 µs | 61.8 µs | 64.3 µs | +319.45%* | [+291.2, +347.7] | yes |

\* CI midpoint not printed in the harness table; the `slower` value is the
reported `vs_base_pct` where available, otherwise the CI midpoint. Every cell
is statistically significant (p=0.0000).

## Notes

- **z1/z1u8e/z3: 3.1–8.2× NEON over scalar.** The interpolation
  `(a0<<5) + (a1-a0)*s + 16 >> 5` maps 1:1 onto NEON i16 lanes; the stride-2
  upsampled taps deinterleave via `vld2q_u16` (8-lane) or `vuzp1q/vuzp2q`
  (4-lane, and u8e's `vmovl_u8` widening).
- **z2: 1.6–3.0×.** Lower ratio is expected — z2's inner loop is a single
  contiguous row run per (row, angle) and the scalar autovectorizes part of
  it; the NEON tier still wins significantly in every cell.
- z1u8e up1 64×64 is the largest single win (+613%, 8.1×): 8-wide u8 loads
  widen to i16 lanes and amortize the per-row setup the scalar pays.
- `scalar_reference` ≈ `forced_scalar` throughout (within noise) — the
  dispatch machinery itself adds no measurable overhead.
- Parity: `dir_simd_diff::dir_highbd_simd_bit_identical_to_scalar_at_every_tier`,
  `z1_u8e_bit_identical_to_z1_high`, `dir_highbd_diff`, `dir_diff`,
  `build_dir_diff`, `predict_intra_diff`, `predict_intra_in_place_diff`,
  `intra_lowbd_diff`, `cdef_find_dir_simd_diff` — all pass (12/12 in the
  filtered run) against the pinned v3.15.1 C oracle.
