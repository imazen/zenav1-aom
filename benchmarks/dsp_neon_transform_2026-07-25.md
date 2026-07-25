# Transform SIMD on aarch64 — the NEON tier's first measurement (2026-07-25)

The port's transform vector path had never run on ARM. Every kernel and pass
driver was `#[rite]`/`#[arcane]` over `X64V3Token`, and `#[arcane]` cfg's its
output to `target_arch = "x86_64"`, so on aarch64 the whole `transform/simd`
module was dead code and the 2-D drivers' SIMD arms were `#[cfg(x86_64)]` too:
aarch64 ran the scalar per-column/per-row driver loops. This is the before/after
pair around the landing that makes it one `#[magetypes(v3, neon, -scalar)]` body
per kernel with a live NEON tier.

## Setup

| | |
|---|---|
| box | Apple M4 Pro, 12 cores, macOS 26.5.2 |
| target | `aarch64-apple-darwin`, rustc 1.97.1 |
| harness | `zenav1-aom-dsp-bench` `dsp_kernels` (zenbench 0.1.9), port-only — no C oracle |
| before | `4b92e2b` (clean tree, sibling worktree), baseline `arm-pre-neon` |
| after | this landing, `cargo bench -- --baseline=arm-pre-neon` |
| batching | every cell runs `WORK_PX = 65536` pixels of back-to-back calls |

`AOM_FORCE_SCALAR=1` is **not** a scalar baseline on aarch64 (`neon` is a
compile-time-guaranteed token archmage refuses to disable), which is why this is
a two-build pair rather than a pinned/unpinned pair. See the `dsp_kernels`
module docs and `aom_dsp::dispatch`.

## Headline

**30 of 34 transform cells improved 18–66%.** The inverse transform — the decode
hot path — improved 58–66% at every size from 8x8 up, in both the bd8/u8 and the
10-bit entry points.

| cell | before → after | delta |
|---|---|---|
| `inv_txfm_hbd10::64x64` | 435,992 → 147,492 ns | **−66.2%** |
| `inv_txfm_u8::64x64_dct` | 432,046 → 151,965 ns | **−64.8%** |
| `inv_txfm_u8::16x16_adst` | 340,256 → 122,852 ns | **−63.9%** |
| `inv_txfm_hbd10::16x08` | 332,161 → 120,910 ns | **−63.6%** |
| `inv_txfm_u8::32x32_dct` | 335,451 → 128,528 ns | **−61.7%** |
| `inv_txfm_u8::08x08_dct` | 326,498 → 137,783 ns | **−57.8%** |
| `fwd_txfm::64x64_dct` | 261,361 → 122,971 ns | **−52.9%** |
| `fwd_txfm::16x16_dct` | 188,023 → 91,547 ns | **−51.3%** |
| `fwd_txfm::08x08_adst` | 228,698 → 170,222 ns | **−25.6%** |
| `inv_txfm_u8::04x16_dct` | 305,178 → 196,075 ns | **−35.8%** |

Full table: `dsp_neon_transform_2026-07-25.tsv`.

## The 4-wide cells, and the gate they produced

The first pass regressed four cells, all with a 4-wide vectorized dimension:
`inv_txfm_u8::04x04_adst` **+26.6%** and `fwd_txfm::04x16_dct` **+9.8%** were
flagged regressions, with `fwd_txfm::04x04_{dct,adst}` and `04x16_adst` +3–5%.

Mechanism: a transform whose vectorized dimension is 4 runs ONE lane batch with
half the lanes idle, and its strided side degrades from 8x8 transposes to
per-lane gather/scatter. Both costs are fixed per batch. On aarch64 they have
less to beat than on x86-64, because `neon` is a compile-time baseline there and
LLVM already auto-vectorizes the scalar driver loop.

Two arms were measured separately, and they do NOT behave the same:

**Half batches are repaid by a bigger kernel.** The fixed cost amortizes over
the 1-D kernel's point count, i.e. the *other* dimension:

| cell (col_n × row_n) | half batch ON | blunt OFF | final (`kernel_points >= 8`) |
|---|---|---|---|
| `inv_txfm_u8::04x04_adst` (4×4) | **+26.6%** | +4.3% | **+4.8%** |
| `inv_txfm_u8::04x16_dct` (4×16) | **−36.2%** | −15.6% | **−35.8%** |
| `inv_txfm_u8::04x16_adst` (4×16) | −29.3% | −7.8% | **−29.0%** |
| `inv_txfm_hbd10::04x16` (4×16) | −40.1% | −20.0% | **−39.8%** |

A flag that simply switched the 4-wide arms off fixed 4×4 but gave back ~20
points on every 4×16 cell. The shipped gate is therefore a predicate on the
kernel's point count — `half_batch_pays(kernel_points)` in
`transform/simd/mod.rs` — which keeps both. `kernel_points == 8` is
**interpolated, not measured**: `TX_CELLS` has no 4x8 cell. Add one before
relying on that boundary.

**Gathers are not repaid.** `fwd_row_pass` at `col_n < 8` loads by per-lane
gather, and `fwd_txfm::04x16_dct` measured +9.8% with `row_n = 16` — a big
kernel did not save it. That arm gets a flat decline on aarch64, and the number
lands at **−0.3%**. The inverse row pass is deliberately not gated on `col_n`:
its loads are contiguous and only its stores scatter, which is why the same
shape is the biggest 4-wide *win* in the sweep. Gathers cost; scatters don't.

x86-64 keeps every 4-wide arm — that is the shape `gate3_transform_simd_2026-07-17.md`
measured and kept, and nothing here re-measured it on an AVX2 box.

## Noise floor, from the negative control

The bench's other 43 cells (cdef, loopfilter, dist, quant, intra) run code this
landing does not touch, so they measure cross-build variance directly:

```
control cells: 43   min −16.2%   p50 +0.7%   p90 +4.6%   max +12.0%
```

Three were flagged REGRESSION (`intra::v_04x04` +12.0%, `dist::sad_64x64` +9.0%,
`cdef::filter_u8_08x08` +8.4%) on untouched code. So the residual 4-wide
transform deltas (+1.3% to +6.8%) sit **inside** the band that identical code
produces across two builds — and structurally so: with the gate live, every
remaining 4x4 transform cell declines the vector path in both builds. In
particular `fwd_txfm::04x04_*` never takes it in either build (`try_fwd_col_pass`
needs `col_n % 8 == 0`, `try_fwd_row_pass` needs `row_n % 8 == 0`; a 4x4
transform fails both), so its +4.5–6.8% cannot be attributable to this change.

Small-cell rows here are layout/alignment-sensitive. Do not read a single ±10%
row on a 4x4 cell as a result; read the direction of a whole size sweep.

## Parity

Byte-identity was proven before any of this was timed — the numbers are only
valid because nothing moved:

- in-module SIMD-vs-scalar differential: **25 permutations, 24 with a vector
  tier live** (it was vacuous on ARM before: the closure gated on `X64V3Token`,
  a stub off x86-64).
- `txfm2d_simd_perm_diff`: `simd_perms=24 scalar_perms=1`.
- the C-oracle differentials (`inv_txfm2d`, `txfm2d`, `inv_txfm1d`, `txfm1d`,
  `fdct`, `inv_txfm2d_lowbd`) green → C == scalar == every SIMD tier.
- full `aom-dsp` suite 352/352 in BOTH dispatch modes.

## What is still on the table

- **The bd8 i16-lane specialization is x86-64 only** (`lowbd16` +
  `inv1d_v3_i16_gen`): raw AVX2 pack/unpack/madd with no magetypes expression.
  aarch64 takes the i32 path — correct, one lane batch instead of two. On x86 it
  was worth −31.5% on the column pass (`gate3_transform_simd_2026-07-17.md`), so
  a NEON i16 twin is the largest single remaining transform lever here.
- **A 4x8 cell in `TX_CELLS`**, to measure rather than interpolate the
  `half_batch_pays` boundary.
- **magetypes gaps** that keep `prims.rs` hand-written per tier at 0.9.28:
  integer widening (i32 → i64; `generic::cross_width` is f32-only), `Mul` on
  `i64x4`, a runtime-count arithmetic shift, and any integer cross-lane
  permute/transpose (`block_ops_i32x8` has array/byte views only; `transpose_8x8`
  exists for f32x8/f64 only). Closing those would let `hb`, `rshiftv`,
  `mul_rshiftv`, `shl_clamp64v`, `transpose8` and the whole `V64` family become
  one generic body instead of two.
