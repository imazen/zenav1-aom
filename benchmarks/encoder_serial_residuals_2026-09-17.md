# Serial residual batch #2 — 2026-09-17

Cell: the real `photo_512` witness (`/home/lilith/tmp/real/photo_512.yuv`,
512x512 i420, cq27 s3, `eprof_yuv` arm `port`, callgrind `Ir`, C arm
9.389G). Real-image only; every landing byte-gated.

## Landings

| commit | change | witness delta |
|---|---|---|
| `d78ec46` | `cost_coeffs_txb_scratch` — `levels_buf`/`coeff_contexts` zero-inits hoisted to caller scratch (write-before-read contract) | 13.894G → **13.856G** (−38M) |
| `8a096f1` | `sse_u16_u8` packed `w==4` arm — 4 strided rows per xmm pair (`loadu_si64`/`si32` + unpack + madd), scalar tail for `h%4` | 13.856G → **13.774G** (−82M) |
| `6c50baf` | intra dir-pred audit: z3 `bw>=8 && bh>=8` floor restored (narrow admissions measured+reverted), pow2 `n_act` shift, z1 4-lane `n_act%8` tail | 13.774G → **13.769G** (−5M) |
| `1e8abf6` | `assemble_{nd,dir}_edges{,_u8}` strided left-edge gather: dst pre-slice folds the write-side bounds check | 13.769G → **13.754G** (−15M) |
| `825d994` | lpf `_n` batching — one dispatch + batch span-check + shared setup per `nseg∈{2,4}` run | 13.754G → **13.755G** (kernel cluster −10M, net flat) |

Cumulative serial session: 13.950G → **13.755G** ≈ **1.465×** C (Ir).
Wall ≈ 395–400 ms vs C ~295–298 ms (~1.34×).

## Measured rejections this round

- **`iter().step_by` + zip for the strided edge gather** (`1e8abf6`
  first attempt): +25M cluster — StepBy/NonNull iterator machinery costs
  more than the bounds checks it removes. Reverted to the indexed loop
  with only the dst pre-slice.
- **`chunks_exact` gather** (same site, first attempt): silently dropped
  the last edge element — `floor(((n-1)*rs+1)/rs) = n-1` chunks. Caught
  by `predict_intra_in_place_diff` before any gate ran.
- **z1 4-wide row vectorization + z3 `bh>=4` admissions** (`6c50baf`
  cycle): per-row `n_act`/splat/slice setup (~55 Ir) exceeds the ~48 Ir
  of scalar multiply-adds it replaces on 4-wide work; +10M → reverted
  gates, kept the pow2-shift and the `n_act%8` tail arm.

## Threaded re-measure (real 1024² photo, cq27 s3, tiles=2 → 16 tiles)

| | port | C (serial shim) |
|---|---|---|
| threads=1 | 2111 ms | 1535 ms (1.375×) |
| threads=4 | 788 ms | — (**0.51×** vs C) |
| threads=8 | 788 ms | — (**0.51×** vs C) |

1t and 8t streams byte-identical (deterministic merge). The C shim has
no tile-threading arm; the comparison is port-threaded vs C-serial wall.

## Byte gates

photo_512: byte-identical to the pre-batch gate (8,822 B) at every step.
1024²: 1t↔8t identical (81,432 B).

## Residual map after this batch (Ir, matched callgrind)

Named, bounded, and no cheap lever found:

- **lpf merged-lane `_dual`/`_quad` internals**: C packs 2–4 segments
  into shared xmm lanes (`lpf_internal_*_dual_sse2`, `filter4_dual`,
  quad AVX2) — ~23M of C's ~68M lpf cost. Port equivalent is a ~800-line
  internals port for an estimated ~15–25M (0.1–0.2 % of total). Bounded
  and documented; not pursued at this ROI.
- **`optimize_txb` cluster** ~2.9G vs C 2.62G: the residual is diffuse —
  `search_tx_type_intra_into` self ~390M is per-candidate front-matter
  (mask checks, txk_map init, ctx setup), no single lever; the rest is
  the documented u16-at-bd8 structural tax + safe-Rust trellis checks.
- **transforms / quantize / txb helpers**: all C-mirrored; remaining
  gaps are per-call overhead grinds (~1.3–2×/call) inside
  byte-identical kernels.
- **memcpy/memset classes**: diffuse across call sites (no dominant
  site; already class-mined on 2026-09-15).
