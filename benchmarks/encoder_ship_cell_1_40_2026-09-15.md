# Ship cell under the 1.40× bar — 2026-09-15 (perf/gate3-txfm-i16-batch)

Cell: `eprof_x86 {port,c} 1024 1024 27 3 1` — the preset zenavif ships
(1024² mirror-tiled photo, cq27, `--cpu-used 3`, ALLINTRA, restoration on,
CDEF off). Interleaved port/C pairs, single-threaded, byte-verified
40,237 B on both arms every rep (`first_diff=-1`).

## Result

10 rotated pairs after the batch below:

| pair | port ms | C ms | ratio |
|---|---|---|---|
| 1 | 2227.6 | 1594.5 | 1.397 |
| 2 | 2205.5 | 1591.6 | 1.386 |
| 3 | 2209.4 | 1596.6 | 1.384 |
| 4 | 2199.2 | 1598.4 | 1.376 |
| 5 | 2238.0 | 1587.7 | 1.410 |
| 6 | 2203.8 | 1593.2 | 1.383 |
| 7 | 2199.7 | 1601.2 | 1.374 |
| 8 | 2205.0 | 1623.8 | 1.358 |
| 9 | 2240.9 | 1605.4 | 1.396 |
| 10 | 2233.9 | 1615.6 | 1.383 |

Median pair ratio **≈1.384×** (9/10 pairs < 1.40; port median 2207 ms /
C median 1597 ms = 1.382×). The user's ≤1.40× ask is met at this cell
with ~1% margin — inside the observed ±1.5% run noise, so treat 1.40 as
met-but-thin, not banked.

## What landed in this batch (all byte-identical at the ship cell)

Branch tip `5c9bf6e`. Per-landing Ir measured at `196² cq32 s0`
profiling-profile callgrind (debug-line attribution now available via
`--profile profiling`):

- `a48a1f2` HOG per-SB Sobel+bin gradient cache (C `pixel_gradient_info`)
  — `TileCtxState.hog_grad: RefCell<HogGradCache>`, lazy per-plane fill,
  4:2:2 geometry handled, synthetic-oversized blocks fall back. Largest
  single win of the batch: −470M Ir at 512² s0.
- `21209dd` z2_left band-major 8×8 tile transpose + Bresenham per-column
  geometry — z2 family ~496M → ~421M self at 512²; the column loop now
  gathers affine taps (contiguous/stride-2 loads) and stores whole rows.
- `2e0041e` filter_intra_edge interior chunking (356 Ir/call, was 419;
  the naive 16-granular madd had regressed to 595) + `sz<3` underflow
  fix + `txb_init_levels` comptime-length pad stores (killed a ~106 Ir
  memset idiom per call ×1.7M).
- `5f872b8` transform dispatch guards: 4×4 inv/fwd input-bound scans
  moved inside the fused magetypes bodies as vector max-reduce
  (`abs_epi32`/`abs_epi16` — i32::MIN/i16::MIN still decline, exactly the
  scalar `unsigned_abs` bound); `sr_*` `iter().all` scans → `[i8;12]`
  array equality across all try_* wrappers.
- `5c9bf6e` optimize_txb monomorphized on `tx_class` (the C
  `UPDATE_COEFF_EOB_CASE`/`UPDATE_COEFF_SIMPLE_CASE` macro structure):
  `optimize_txb_run<const TXC: u8>` folds every per-coefficient
  `match tx_class` in get_nz_mag/get_br_ctx/ctx tables. Plus
  `#[inline(always)] two_coeff_cost_simple` (was a real call, ~82M Ir/enc)
  and `get_dqv` reusing the loaded `ci`. −131M Ir at 196² s0.

## Residual gap map (196² s0 cq32, per encode, port − C)

`optimize_txb_core` family ~1.07G vs C ~0.93G — still the biggest named
kernel gap, now diffuse: `cmp.rs` min/max chains ~170M/enc inside it and
per-element `levels[]`/`qcoeff[]` bounds checks are the safe-Rust tax
(`forbid(unsafe_code)`, no hardware gather). Restoration family ~800M vs
~380M C and `highbd_variance64` ~113M vs ~16M are the **u16-at-bd8 tax**
— C runs lowbd u8 kernels (`av1_lowbd_*`) at bd8; the port's u8 intra
path exists (`assemble_dir_edges_u8`) but the encoder never routes to it.
That conversion is a structural program, not a lever; it remains the
named mechanism behind most of the remaining ~38%.

`getenv` in the C arm (~100M/enc at 196²) is aomenc's own overhead, not
work the port is missing.

## Verification

- `cargo test -p zenav1-aom-dsp txfm2d` — 12/12 incl. every-tier scalar
  and permutation differentials.
- `cargo test -p zenav1-aom-dsp txb` / `optimize` — 17/17, 2/2 incl.
  `optimize_txb_round_trip_identical` (QM and non-QM).
- HOG, z2, filter_intra_edge, nz_map, txb_init_levels differentials all
  green vs the real exported C functions.
- Byte witnesses: 1024² s3 cq27 40,237 B identical every rep.
