# Ship cell under the 1.40× bar — 2026-09-15 (perf/gate3-txfm-i16-batch)

Cell: `eprof_x86 {port,c} 1024 1024 27 3 1` — the preset zenavif ships
(1024² mirror-tiled photo, cq27, `--cpu-used 3`, ALLINTRA, restoration on,
CDEF off). Interleaved port/C pairs, single-threaded, byte-verified
40,237 B on both arms every rep (`first_diff=-1`).

## Result — third batch (through `71aab23`)

4 interleaved pairs after the lpf opt-walk landing + the alloc-pooling
pass: port {2224.5, 2227.3, 2227.3, 2216.6} ms / C {1633.0, 1631.3,
1638.7, 1633.0} ms — median ratio **≈1.36×** (port ~2225 / C ~1633).
Byte-identical 40,237 B every rep; `self_contained_key_frame` byte-match
+ tune bundles + 35 targeted differentials green.

The lpf opt walk is confirmed live by callgrind: `hbd_lpf` kernel calls
dropped **879K → 402K** per 512² rep (dual/quad batching mirroring C's
`lpf_opt_level == 1`), yet wall time held ~flat — the remaining lpf gap
is kernel *shape*, not traversal: C's bd8 path runs u8 `aom_lpf_*`
(~150M Ir total) where the port runs u16 kernels (~400M). That is the
u16-at-bd8 program, already mapped below.

The alloc pass cut Rust-side allocator calls roughly in half at 512²
(heavyweights post-pass: `txb_coeffs` SmallVec spills ~50K/rep — retained
per-txb coeff arrays, needs the flat-`coeffs`+range refactor; Vec
`finish_grow` chains ~37K/rep). Wall delta on glibc is sub-percent as
predicted; the same calls cost ~7–11× more on Windows/macOS heaps
(`encoder_alloc_pass_windows_2026-09-10.md`), which is where this class
pays. Heaptrack: pooled state is bounded — TLS scratch ~100 KB/thread
worst case, `var_cache` ≤ 32 KB per *tile ctx* (per-worker, not
per-thread, under tile threading).

## Result — second batch (through `9386d02`)

Refresh after the second batch (through `9386d02`), same cell — C's own
clock ran ~3% slower this round so ratios are the honest unit:

| pair | port ms | C ms | ratio |
|---|---|---|---|
| 1 | 2234.9 | 1635.1 | 1.367 |
| 2 | 2227.7 | 1643.3 | 1.356 |
| 3 | 2225.0 | 1640.4 | 1.356 |
| 4 | 2239.6 | 1645.3 | 1.361 |
| 5 | 2223.9 | 1632.8 | 1.362 |
| 6 | 2223.3 | 1650.8 | 1.347 |
| 7 | 2237.3 | 1638.7 | 1.365 |
| 8 | 2286.4 | 1640.0 | 1.394 |
| 9 | 2239.2 | 1636.6 | 1.368 |
| 10 | 2228.6 | 1647.5 | 1.353 |

Median pair ratio **≈1.36×** (9/10 pairs < 1.37; port median ~2233 ms /
C median ~1641 ms = 1.360×). The ≤1.40× ask now carries ~3% margin —
still within run noise, but two batches deeper than the 1.384 reading.

### Earlier batch (through `5c9bf6e`)

10 rotated pairs, median pair ratio **≈1.384×** (9/10 pairs < 1.40;
port median 2207 ms / C median 1597 ms = 1.382×), same cell.

## What landed in the second batch (all byte-identical at the ship cell)

Branch tip `9386d02`. Ir deltas measured at 512² cq27 s3 callgrind:

- `a939780` `txb_init_levels` `as_chunks` bodies — per-iter
  `checked_sub` bounds checks killed (~172→~131 Ir/call).
- `745a2cd` `filter_intra_edge` sliding-window kernel — C
  `av1_highbd_filter_intra_edge_sse4_1` mirror via slack-aware
  `highbd_filter_intra_edge_at` (370→97 Ir/call).
- `1e5490a` `block_error` instruction-level `av1_block_error_avx2`
  mirror (packs→madd→widen, bug-compatible full-domain incl.
  saturation) + dedicated `w==4`/`h==4` variance arm.
- `9be086a` `nz_map_contexts` windowed tile loads — one pre-slice per
  tile whose end equals the old largest access (same panic domain);
  372→90M Ir.
- `8bc0c70` `z3_cols` windowed edge taps + banded dst stores — same
  window trick; 387→205M Ir.
- `9386d02` **fwd 16×16 at 16 lanes** — `lowbd_fwd_txfm2d_16x16_avx2`
  shape composed from the existing `run_fwd1d_i16` kernels plus a
  verbatim `transpose_16bit_16x16_avx2` port, replacing the two 8-lane
  halves. Same `FWD16_I16_BOUND` gate so the accept/decline domain is
  unchanged; ~383M→~77M Ir inclusive. C's own AVX2 8×8 is `__m128i`
  internally, so 16×16 was the last real lane-width gap in fwd square
  transforms.

## What landed in the first batch (all byte-identical at the ship cell)

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

## Residual gap map (512² s3 cq27 callgrind, post-`9386d02`)

The remaining ~36% is now diffuse — no single kernel dominates. Named
mechanisms:

- **u16-at-bd8 tax** (the big structural one): `lpf_impl_v3` (~33M),
  `highbd_variance64` residual, and the restoration u16 kernels all run
  8-lane u16 where C runs 16-lane u8 `aom_lpf_*`/`aom_variance*`.
  Loop-filter traversal batching has now landed (post-`9386d02`:
  dual/quad `nseg` walk, 879K→402K kernel calls at 512²), so the
  residual is kernel shape — transpose-gather u8 kernels + a u8
  workspace through the post-filters, i.e. a program, not a lever.
- **`quantize_fp_impl_v3`** (~88M): already a tight C mirror at
  ~212 vs ~180 Ir/call — thin residual.
- **`build_directional_intra_high_in_place`** (~87M @ 44/call): mostly
  the two 160-element `above_data`/`left_data` fills, matching C's
  own behaviour.
- **Driver layers**: `av1_fwd_txfm2d_into` (67M @ 20/call),
  `av1_inv_txfm2d_add_into` (29M), `get_txb_ctx_general` (8M) — table
  lookups and dispatch, already specialised.
- **`optimize_txb_scratch`** (~44M): coefficient-context helpers already
  windowed; residual is `min3`/cmp chains inherent to safe Rust.
- **memcpy/memset class** (~215M): allocator/copy traffic through PLT —
  the ~4× allocator-call surplus noted in earlier records.

### Earlier residual map (196² s0 cq32, per encode, port − C)

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
