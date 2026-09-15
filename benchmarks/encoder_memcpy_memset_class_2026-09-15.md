# memcpy/memset call-class elimination — 2026-09-15

Profile cell: `port 196x196 cq27 s3 reps=5` (callgrind `Ir`, same protocol every arm).
Shipping witness: `port 1024x1024 cq27 s3` — **byte-identical 40,237 B at every
step**; the focused differentials and the 652-test encode `all` suite (incl. all
e2e byte gates) are green.

The fresh gap map put `__memcpy_avx_unaligned_erms` at **4.33M calls / 162M Ir**
and `__memset_avx2_unaligned_erms` at **188M Ir** per profile run, vs C's whole
`memcpy`/`memset` classes at 9.8M / 33.2M. This batch attacks the *call-count*
class: hundreds of thousands of small runtime-length copies that glibc charges
~90-175 Ir each to dispatch.

## What landed

**Literal-size copy arms.** `copy_edge` (`aom-dsp/src/intra/mod.rs`), `copy_ctx`
(`aom-encode/src/tx_search.rs`, `pub(crate)`), `copy_row`
(`aom-encode/src/intra_uv_rd.rs`): `match` on the reachable copy widths
(`{0,1,2,4,8,16,32,64}`) so each arm is a `copy_from_slice` with a *literal*
length — inline vector moves, no PLT call; the `_` arm keeps exact semantics for
any other `n` (including `0`, which still emitted a `memcpy` PLT call). Applied
to: `assemble_dir_edges{,_u8}`/`assemble_nd_edges{,_u8}` edge assembly, the
`V_PRED` row copy, lowbd filter-intra dst rows, filter-intra `row0`/`row1`
staging and dst stores (`intra/filter_simd.rs`), `smooth_*_impl` row stores
(full 16-lane chunks store the vector directly; tails go through `copy_edge`),
CfL DC-cache + row-replicate + ctx copies (`intra_uv_rd.rs`), luma `t_above`/
`t_left` ctx init, and `SavedCtx` save/restore (`partition_pick.rs`).

**Hadamard `_into` APIs** (`aom-dsp/src/dist/hadamard.rs`): `hadamard_8x8_into` +
`highbd_hadamard_8x8_into` write the second pass straight into a caller
`&mut [i32; 64]`; the 16x16/32x32 `_into` variants now take the `_into` path for
their quadrant sub-blocks (4 x 256 B return-value copies per call gone, plus the
sret forward through the `incant!` dispatch). Call sites rewired in
`tx_search.rs` (both SATD paths, via a grow-only `b64` scratch helper) and
`nonrd_pickmode.rs`.

**Fixed arrays where Vecs were payload.** `SavedCtx` was ten `Vec`s per save —
now `[[i8; 32]; 3]`/`[i8; 32]`/`[u8; 32]` (MI extents <= 32), copies only the
live prefix via `copy_ctx`. `leaf_pick_sb_modes`'s six `to_vec()` ctx snapshots
are `[i8; 32]` arrays passed as exact-len slices. `EncodeIntraPlaneOutcome`'s
`ta`/`tl` are `[i8; 32]` (the `__internals`-gated struct's consumers all read
through `get_txb_ctx`-bounded prefixes; tests now assert the semantic prefix).
`txb_coeffs()` builds `SmallVec<[i32; 32]>` via `from_buf_and_len` for n <= 32 —
no memcpy call, no heap — the common small-transform case in both encode walks.

**SGR scratch pool** (`aom-dsp/src/restore/sgr.rs`): `calculate_intermediate`'s
`a_buf`/`b_buf` (~86 KB per call, 538 calls/rep), `selfguided_restoration`'s
`dgd32`, and `apply_selfguided_restoration`'s `flt0`/`flt1` all move to a
`thread_local` `SgrTlsScratch` taken/put-back by `mem::take` (the XQ_POOL
shape). The pools resize WITHOUT re-zeroing: every read is provably preceded by
a write in the same call — the `ab_row` ring (rows 2..h+4, cols 2..w+4) sits
inside boxsum's written `0..h_ext x 0..w_ext`; `sgr_widen` fills all of `dgd32`;
`flt[i]`'s `rads[i] > 0` read guard is its write guard and `sgr_final*` writes
every cell. That is exactly C's uninitialized-`aom_malloc` contract, so dirty
reuse is bit-identical.

**`LvMapCoeffCost` fixed arrays** (`aom-dsp/src/txb/fill.rs`): the six `Vec<i32>`
tables are compile-time sizes (26/12/336/18/6/546 i32) — now arrays; ~6 calloc
dispatches per fill call gone (11,880 calls/profile). `fill_eob_cost_from_arena`'s
per-ctx `vec![0i32; nsy]` is a `[i32; 12]` + exact slice.

## Measured

| class | before batch | after | delta |
|---|---:|---:|---:|
| memcpy calls | 4,330,000 | 899,873 | **-79 %** |
| memcpy Ir | 161.9M | 103.8M | -36 % |
| memset Ir | 188.4M | 142.0M | **-24.6 %** |
| total Ir | ~9.04G | 8.944G | ~-1.1 % |

Same-protocol A/B for just the SGR-pool + `copy_ctx`-0 + restore + LvMap-array
step (cg_memcpy_batch -> cg196_final): memset 188.4M -> 142.0M, memcpy
105.5M -> 103.8M Ir / 1.00M -> 0.90M calls, total 8.9694G -> 8.9438G (-0.29 %).

## What was checked and deliberately NOT done

- `lf_search::try_filter_plane` shows 49.9M memcpy Ir / 300 calls — a
  full-plane `extend_from_slice` per LF-level trial. C pays the same copy (it
  saves/restores the plane per trial via `aom_yv12_copy_y_c` row-wise, ~10.5k
  Ir/trial); the port's figure is inflated by callgrind charging `rep movsb`
  ~1 Ir/byte. One contiguous copy already beats C's two row-wise copies on
  wall — skipped as an instrumentation artifact, not a lever.
- `key_frame` `Vec<u16>` plane clones (9.0M/54 calls) — semantic
  whole-plane snapshots C makes too; same rep-movs accounting note.
- `build_directional_intra_high_in_place` (26.1M/725k) zeroes two
  `[u16; 160]` edge arrays per call. Dropping the zero-init needs reads-subset-
  of-writes proof across every angle/size/availability combo, and safe Rust
  can't express uninit anyway (`#![forbid(unsafe_code)]`). Deferred.
- `cost_coeffs_txb` (10.7M) and `nz_map_contexts` v3 (4.8M) stack/table
  zeroing — the v3 tier's internal memset already covers the caller's init;
  eliminating it needs the same uninit proof. Deferred.
- `PlaneCtx::new` (22.3M/108 calls) — plane-sized `dgd_pad`/`dst_pad`/`flt*`
  buffers; poolable (Drop-returns-to-TLS or a scratch param) but the struct
  owns them and it is a wider refactor than this batch's call-site work.
  Named, deferred.
- `fill_lv_map_coeff_cost`'s `coeff_contexts`/`levels_buf` in
  `cost_coeffs_txb` — same defensive-zero class.
