# s9 allocator-class reductions — 2026-09-15

Profile cell: `port 192x192 cq27 s9 reps=5` (callgrind `Ir`). This is the perf
gate's worst ratio cell (port 3.42x C on wall before this batch): the
`nonrd_pickmode` fast path is dominated by per-call fixed overhead, not kernels,
so allocator/memcpy call counts are the lever here.

Shipping witness: `port 1024x1024 cq27 s3` — **byte-identical 40,237 B**;
192x192 s9 witness — **byte-identical 1,894 B**. 52/52 intra differential tests
green (52/52 post residual-pool; all green after each mechanism).

## What landed

**`encode_b_intra_dry` fixed ctx arrays** (`aom-encode/src/encode_sb.rs`): the
six per-call `Vec<i8>` ctx snapshots (luma above/left + chroma U/V above/left)
become `[i8; 32]` arrays filled by `copy_ctx`'s literal-size arms, passed to
`EncodeIntraYEnv`/`UvRdEnv` as exact-length slices. Allocator calls under the
fn: **69,552 -> 9,936** per profile.

**Exact `txbs` capacity** (`aom-encode/src/encode_intra.rs`): the walk appends
exactly `ceil(w/txw) * ceil(h/txh)` txbs; `Vec::with_capacity` reserves once
instead of paying the log-growth realloc chain. `RawVec::finish_grow` halved
(30,254 -> 15,350).

**TLS `residual` pool** (`aom-encode/src/encode_intra.rs`): `residual` was a
fresh `Vec` per call (~1 alloc + 1 free x ~30k calls/profile). Now taken from a
`thread_local` pool at entry, returned at exit — the `XQ_POOL` shape.
`highbd_subtract_block` writes every element before any read and the live
prefix is sliced to `txw*txh`, so dirty reuse needs no re-zero — C's own
scratch is uninitialized `malloc`, the same contract.

## Measured (same protocol every arm, 192x192 s9 reps=5)

| arm | total Ir | vs base |
|---|---:|---:|
| base (post `db29bfb`) | 271,113,xxx | — |
| + ctx arrays + txbs cap | 259,649,337 | -4.2 % |
| + residual pool | 257,235,565 | **-5.1 %** |

Allocator-call deltas at the final state:

| site | calls before | calls after |
|---|---:|---:|
| `encode_b_intra_dry` | 69,552 | 9,936 |
| `RawVec::finish_grow` | 30,254 | 15,350 |
| `encode_intra_block_plane_uv` allocs | 19,872 | 9,936 |
| `encode_intra_block_plane_y` allocs | 9,936 | 4,968 |

## Remaining allocator items at s9 (attributed, deferred)

- `LeafEncodeOut` drop_glue: 30,312 calls — the `txbs` Vec drops. The Vec is
  retained output (moved into `SbTree`/`LeafWinner`), not scratch; pooling it
  would need a lifetime-crossing arena. Semantic, skip.
- `txb_coeffs` 15,408 calls — `SmallVec<[i32; 32]>` spills to heap for
  n > 32 (16x16+ transforms). Retained per-txb payload, not churn.
- `nonrd_pick_intra_mode` 6,624 calls — unexamined; next-lever candidate.
- `SbTree` drop_glue 4,272 — per-SB tree teardown, structural.
