# `half_batch_pays(8)` — closing the interpolated rung (aarch64)

**Date:** 2026-07-31
**Box:** Apple M4 Pro (aarch64-apple-darwin), `--release`, no `target-cpu=native`
**Harness:** `crates/aom-dsp-bench/benches/dsp_kernels.rs` (zenbench 0.1.9,
interleaved round-robin), port-only — no C oracle in the loop
**Command:** `cargo build --release --benches -p zenav1-aom-dsp-bench` then
`target/release/deps/dsp_kernels-* --group=inv_txfm_u8` (~12 s per arm)

## What this closes

`aom_dsp::transform::simd::half_batch_pays` decides whether a transform whose
vectorized dimension is 4 should run one half-idle lane batch. On aarch64 the
threshold is `kernel_points >= 8`. Its doc block carried this caveat:

> `kernel_points == 8` is INTERPOLATED, not measured: the bench's `TX_CELLS` has
> no 4x8 cell. […] add a 4x8 cell and re-measure before relying on it.

The shipped constant therefore rested on an interpolation between the measured
4-point cell (half batch loses) and the 16-point cell (half batch wins big).
This run adds the missing cells and measures the rung directly.

## Method

`TX_CELLS` gained `(5, "04x08")` and `(6, "08x04")` — TX_4X8 and TX_8X4, the two
1:2-aspect shapes the grid was missing. `04x08` is the 4-wide (half-batch) column
at exactly `kernel_points = 8`.

A/B by flipping the threshold itself, rebuilding between arms:

- **ON**  — `kernel_points >= 8` (shipped): 4x8 takes the half batch.
- **OFF** — `kernel_points >= 16`: 4x8 takes the full batch.

Nothing else differs between the arms.

## Result — the interpolation held

| cell | ON (`>= 8`) | OFF (`>= 16`) | ON is |
|---|---|---|---|
| `inv_txfm_u8::04x08_dct`  | 216.1 ±2.7 µs (303 Mpx/s) | 273.4 ±3.4 µs (240 Mpx/s) | **21.0% faster** |
| `inv_txfm_u8::04x08_adst` | 248.0 ±2.6 µs (264 Mpx/s) | 317.6 ±2.9 µs (206 Mpx/s) | **21.9% faster** |
| `inv_txfm_u8::08x04_dct`  | 208.6 ±2.5 µs (314 Mpx/s) | 253.0 ±2.9 µs (259 Mpx/s) | **17.6% faster** |
| `inv_txfm_u8::08x04_adst` | 234.4 ±4.0 µs (280 Mpx/s) | 296.1 ±3.6 µs (221 Mpx/s) | **20.8% faster** |

Every delta is far outside the run-to-run band (zenbench's own ±ranges are
~1–2%; the repo's stated control band for this bench series is ±16% on the
noisiest cells, and these are 17–22% on cells whose own CI is ~1%).

**Verdict: `kernel_points >= 8` is correct and is now measured, not inferred.**
The mechanism's predicted monotonicity — a fixed per-batch cost amortized over
the kernel, so 4 loses / 8 pays / 16 pays more — is what the data shows.

`08x04` moving is expected, not a surprise: TX_8X4's other pass is an 8-point
kernel, so it sits on the same rung and flips with it.

## Scope

- aarch64 only. The x86-64 arm of `half_batch_pays` is unconditional `true` and
  was not re-measured here (its basis is `gate3_transform_simd_2026-07-17.md`).
- Port-only timings. This says nothing about parity with the C kernels; that is
  the differential suite's job, and both batch shapes must produce identical
  output either way (they do — the transform differentials pass in both dispatch
  modes with the cells added).
