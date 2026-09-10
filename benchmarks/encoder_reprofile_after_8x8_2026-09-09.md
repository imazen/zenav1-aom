# Re-profile after the two 8x8 fusions — the mechanism, confirmed by symbol

**2026-09-09. No code changed.** §14 says re-profile before ranking the next
lever; KB-PERF-23/24 moved the encode ~3.5 %, so the old table is stale.

## The two landings did exactly what "SIMD-preserving" claimed

Per-symbol, 1024x1024 cq27 `--cpu-used 3`, before (KB-PERF-22 HEAD) and after:

| symbol | before | after | delta |
|---|---:|---:|---:|
| `inv_col_pass_core_v3` | 93.8 ms | 59.4 | **−34.4** |
| `fwd_col_pass_core_v3` | 51.1 | 24.0 | **−27.1** |
| `fwd_row_pass_core_v3` | 45.8 | 21.3 | **−24.5** |
| `inv_row_pass_core_v3` | 54.6 | 32.8 | **−21.8** |
| **pass scaffolding total** | 245.3 | 137.5 | **−107.8** |
| `run_fwd1d_v3` (kernel arithmetic) | 81.5 | 82.1 | **+0.6** |
| `run_inv1d_v3` (kernel arithmetic) | 107.8 | 109.8 | **+2.0** |
| `av1_fwd_txfm2d_into` (driver) | 58.4 | 51.3 | −7.1 |
| `av1_inv_txfm2d_add_into` (driver) | 54.2 | 55.7 | +1.5 |

**The scaffolding fell by 107.8 ms and the kernels did not move.** That is the
claim "it keeps both vector passes and removes only the driver and the `buf`
round trip", measured at symbol level rather than argued — and it is the
distinction that made these two work where KB-PERF-16's SCALAR 8x8 fusion
measured **+7.07 %**: that one removed the kernels too.

The symbol deltas sum to ~−117 ms against the two bands' −2.017 % and −1.538 %
(~124 ms), consistent within single-run sampling.

## Where the transform class stands

**984.7 ms**, down from 1072.4. Against libaom's 283.4 the gap is ~701 ms (was
789) — still the largest single class, and still ~36 % of the shipping-preset
gap.

What is left inside it:

| | ms | note |
|---|---:|---|
| `run_inv1d_v3` + `run_fwd1d_v3` | 191.9 | **irreducible** — the 1-D butterflies themselves |
| remaining `*_pass_core_v3` | 137.5 | the scaffolding of the sizes NOT yet fused |
| drivers (`*_txfm2d_into`) | 107.0 | per-call setup for those sizes |
| the `try_*` gates | ~53 | |

## The next lever, sized — and it is 3x the work for 1/3 the gain

By census share at this preset, after 4x4 (40.51 % of forwards, fused) and 8x8
(25.74 %, fused), the ranking is **8x4 9.13 %, 4x8 7.60 %, 16x16 5.62 %,
16x8 5.14 %, 8x16 3.88 %**.

* **16x16** is the only remaining SQUARE size, so its recipe is the closest copy
  of KB-PERF-23/24 — but `i32x8` holds 8 lanes and 16x16 needs **two vectors per
  row**, i.e. 32 vectors and a 16x16 transpose (four 8x8 block transposes plus
  the off-diagonal swap), roughly **3x the code** of the 8x8 pair. Scaling the
  8x8 result by call share gives **~−0.44 % forward and ~−0.29 % inverse**.
* **8x4 and 4x8 together are 16.7 % of forwards** — more than 16x16 — but they
  are RECTANGULAR (`rect_type = +-1`, so the `NEW_SQRT2` scaling applies) and one
  dimension is 4, which puts a half-filled batch on one of the two passes. The
  `row_n == 4` experiment measured **null** (`encoder_fwd_row_4wide_...rejected`)
  because the 8x8 transpose ran at full cost on half-zero data — **inside a
  fused kernel that calculus changes**, since the transpose is in registers
  either way, but it has not been measured and should not be assumed.

**So the next landing is 16x16, and it is the first one in this sequence whose
cost/benefit is materially worse than its predecessor.** That is worth stating
plainly rather than discovering halfway through.

## Limits

Single profiling run per side; shares are the port's own wall. One cell, one
content class, one box, bd8 4:2:0, cq27, `--cpu-used 3`. `optimize_txb_core` is
now the largest port symbol at 401.5 ms — and the port still BEATS libaom there
(0.93x at s3), so it remains a non-lever for the fourth consecutive profile.
