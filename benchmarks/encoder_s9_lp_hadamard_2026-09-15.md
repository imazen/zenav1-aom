# s9 nonrd scratch pool + lp-Hadamard SSE2 mirrors — 2026-09-15

Profile cell: `port 192x192 cq27 s9 reps=5` (callgrind `Ir`). Same protocol as
`encoder_s9_alloc_class_2026-09-15.md` — the perf gate's worst ratio cell.

Shipping witness `port 1024x1024 cq27 s3`: **byte-identical 40,237 B**.
192x192 s9 witness: **byte-identical 1,894 B**. All 7 lp-hadamard differential
tests green vs the real exported C (`nonrd_block_yrd_lp_diff`: matches-C,
tiers-agree-over-the-reachable-range, transpose-is-load-bearing pins);
nonrd suite 40/40 and intra suite 52/52 green on the final tree.

## What landed

**`nonrd_pick_intra_mode` TLS scratch pool** (`nonrd_pickmode.rs`, commit
`a694919`): `visits: Vec<(usize, usize)>` and `diff: Vec<i16>` were two
per-call Vecs (~6.6k alloc/free pairs under the fn per profile). Both are pure
scratch — `visits` is rebuilt from `push`es every call; `diff` is fully
overwritten by `highbd_subtract_block` before `block_yrd_*` reads it, so
neither needs re-zeroing (C's own scratch is uninitialized `malloc`, the same
contract). Grow-only reuse, exact live slices, take-at-entry /
put-back-at-exit (the fn never nests and has no early returns, so the pool is
total).

**`aom_hadamard_lp_*` v3 SIMD bodies** (`aom-dsp/src/dist/hadamard.rs`, commit
`5406ec3`): the s9 nonrd estimate ran a scalar lp-Hadamard chain —
16x `hadamard_col8` + a scalar transpose per 8x8, ~863 Ir/call vs C's
`aom_hadamard_lp_8x8_sse2` at ~149. The new dispatched kernels mirror the C
SSE2 bodies: eight unaligned 128-bit row loads, the int16 butterfly, the
in-register transpose (the iter-0 fused transpose emits the same
EOB-affecting lane order the scalar recipe's trailing transpose produces), a
second butterfly, eight stores. `hadamard_lp_16x16`'s combine is
`_mm_srai_epi16` after wrapping add/sub — truncate-then-shift, the same
semantics the old `wrapping_add(..) >> 1` documented. Wrapping int16
arithmetic throughout; safe reference-based loads/stores under
`forbid(unsafe_code)`; scalar fallback kept for the non-v3 tier and scalar
forcing. The encode-side wrappers in `nonrd_pickmode.rs` now delegate to the
dispatched dsp kernels.

## Measured (same protocol every arm, 192x192 s9 reps=5)

| arm | total Ir | vs this batch's base |
|---|---:|---:|
| base (post `bf0fdaa`) | 257,235,565 | — |
| + nonrd scratch pool | ~257.24M | net-neutral (~13k fewer alloc ops; TLS bookkeeping offsets ~35 Ir/op) |
| + lp hadamard v3 | 246.36M | **-4.2 %** |

Cumulative session delta at this cell: **271.1M -> 246.36M Ir (-9.1 %)**.

Per-call check: `hadamard_lp_8x8_v3` self = 1.08M Ir for 8,040 calls
(~134 Ir/call) vs C's SSE2 ~149 — at parity. The old scalar chain
(`hadamard_lp_8x8` + `hadamard_col8` + internals) was ~10.3M self Ir.

Wall spot-check after the landing: port `256x256 cq27 s9` 6.50 ms/encode vs C
1.76 ms/encode; port `192x192 s9` ~104 ms/encode over 5 reps (~521 ms).

## Structural attribution: the pack re-walk is the s9 headline

The remaining s9 gap is structural, not kernels. `pack_leaf` calls
`encode_b_intra_dry` **again** for every committed leaf (3,312 calls vs
1,656 pick calls at this cell) to regenerate coefficients and reconstruction
that the pick already computed and retains in `LeafWinner`/`LeafEncodeOut.txbs`.
C's `write_modes_b` writes the retained coefficients; it does not re-encode.

- port `pack_leaf` inclusive: **~113M Ir — ~44 % of the s9 cell**
- C `write_modes_b`: ~11.7M inclusive at the same cell
- port `optimize_txb_core` runs 9,162x vs C `optimize_b` 3,054x (luma count
  matches C 1:1; the port also trellises the chroma planes — `encode_b_*` runs
  y once and uv twice per call — and pack re-runs the whole thing)

`pack.rs`'s module doc records why re-running is *correct* in this envelope
(same winning leaves/contexts; final-pass trellis needs an output-enabled
re-encode in some modes). Eliminating it means replaying retained `TxbEncode`
data in pack while preserving: output-enabled final trellis, exact CDF/context
progression, reconstruction, palette, IntraBC, chroma, and every byte gate.
That is the next lever and a scoped architectural change, not a kernel.

## Remaining kernel items at s9 (attributed, deferred)

- `block_yrd_lowbd` ~20M self Ir — inline scalar `quantize_lp` / `satd_lp` /
  `block_error_lp` per-coeff loops (C ~7x cheaper inclusive). dsp-side lp
  quantize/satd/block_error SIMD is the next contained kernel item.
- `optimize_txb_core` ~18M self — the trellis DP; call-count gap above.
- `encode_intra_block_plane_uv` ~13M self — per-plane walk.
- `write_txb_body` ~4.5 % — entropy symbol emission.
