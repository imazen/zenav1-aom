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
| `80a742d` | `optimize_txb_core`: `assert!(ci < n)` once per scan read + sibling arrays re-sliced to `n` — folds the 4–5 `ci`-indexed bounds checks per scan position | 13.755G → **13.729G** (−26M) |
| `8e8f54a` | `nz_map_ctx_offset(tx_size)` resolved once per txb and passed as `&[i8]` through `get_lower_levels_ctx` — the 24-arm match + slice construction + `nz_off[ci]` check had been rebuilt per coefficient | 13.729G → **13.392G** (−337M) |

Cumulative serial session: 13.950G → **13.392G** ≈ **1.426×** C (Ir).
Wall ≈ 390–395 ms vs C ~295–298 ms (~1.32×).

The `nz_map_ctx_offset` landing is a lesson in profile attribution:
the helper's *self* cost was only 78.7M, but it is `#[inline(always)]`
and its real cost — a 24-arm `match` per coefficient visit plus the
slice ptr+len rebuild plus the `nz_off[ci]` bounds check — was smeared
across `cmp.rs` (−227M) and `tables.rs` (−92M) attribution. Hoisting a
tiny `#[inline(always)]` match helper out of a per-element loop can be
worth several times its annotated self cost.

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
- **SmallVec for `XformQuantScratch.levels`/`cost_ctx`** (pooled
  scratch, `SmallVec<[u8;1536]>`/`[i8;1024]`): +153M vs the asserted
  no-SmallVec baseline — the tagged inline-vs-spilled check on every
  `resize`/deref in hot loops costs far more than the one-time
  allocation it saves. `Vec` grow-only stays. SmallVec remains correct
  where the container is short-lived and commonly small
  (`TxbCoeffPair`, `TxbWinners`); it is wrong for long-lived pooled
  scratch whose per-access cost dominates.

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
- **`optimize_txb` cluster** ~2.8G inclusive vs C's `av1_optimize_txb`
  4.19G — the port's trellis is now ~1.5× *faster* than C; the residual
  moved to the orchestration side. `search_tx_type_intra_into` self
  ~390M is per-candidate front-matter (mask checks, txk_map init, ctx
  setup), no single lever; the rest is the documented u16-at-bd8
  structural tax + safe-Rust per-scan-position asserts (~54M, the
  `ci < n` invariant guards — they buy back ~100M+ of folded checks).
- **transforms / quantize / txb helpers**: all C-mirrored; remaining
  gaps are per-call overhead grinds (~1.3–2×/call) inside
  byte-identical kernels.
- **memcpy/memset classes**: diffuse across call sites (no dominant
  site; already class-mined on 2026-09-15).

## Batch 3 tail (evening): bd8-u8 predictor probe + restoration folds

Commits: `5441b36` (acc_stat_line 16-bit madd pair-fold + highbd_sse/variance
SIMD swap in pick.rs), `10084a3` (calc_ab setup hoisted to per-rect),
`00201ae` (z1 u8e: staged u8 edge → cvtepu8/pshufb i16-lane kernel → u16 dst).

photo_512 cq27 s3 serial: 13,392,435,2xx → **13,324,204,447 Ir** (-68M this
phase; session start 13,950,629,3xx → -626M total). Byte-identical at every
step (8,822 B).

### The u8-predictor verdict, measured

- **z1 u8e −8.6M net.** The kernel is -13.3M (89.5M→76.2M); the per-call
  u16→u8 edge stage costs ~14M after dropping the defensive >255 scan to a
  `debug_assert` (the bd==8 gate makes u8 a codec invariant, same trust C's
  lowbd buffers carry).
- **z2 u8e REJECTED (+5M, reverted).** Its per-row suffix shrinks
  (`c_end` grows down the rows), so most rows land on the 8-px arm where
  `loadu_si64 + cvtepu8` (2 ops) costs more than one u16 `loadu` for the
  same 8 i16 lanes. **u8 widens the load window, not the arithmetic
  density** — it only pays on rows that fill the 16-px arm.
- **z3 u8e not attempted**: the 8-row band transpose fixes lanes at 8
  regardless of load width; u8 halves bytes loaded, not instructions.
- **wiener/convolve pair-fold analyzed, skipped**: each output lane
  consumes a different src position, so no load sharing — the madd fold
  trades widen+mul+add for load+unpack+madd at par (~1.6x best case via
  u8 staging ≈ -25M on a 94M kernel).

### Updated residual map deltas

- restoration flat self ≈ 740M vs C ≈ 300M: `calc_ab` 108M is i32
  box-sum bound (LUT gather needs `vpgatherdd`, no safe archmage form);
  `acc_stat_line` 120M still 3x C's win5 40M — the remaining gap is C's
  maddubs u8-pair density needing a stats-layout rewrite (~deferred);
  `integral_image` 3-pass vs running-sum is algorithmic.
- predict: port ~1.39G vs C ~976M; `plan_dir_intra_high`+`assemble_dir_edges`
  (196M) ≈ C's builder (165M) — edge pipeline is already competitive; the
  gap concentrates in z2 (368M vs 194M — left-half gather; C uses
  maskload+pshufb windows) and z1/z3 scalar arms on small blocks.
- fwd txfm: port ~1.42G vs C ~890M — 4x4 at par, 16x16 ahead, 8x8 xmm-vs-ymm
  (~285M vs 215M); generic-driver overhead ~140M self on unfused sizes.
- chroma (`intra_sbuv` +490M) and committed-recon u16 paths: unchanged
  verdict — the u8 recon-plane split (~300 sites) is the real fix and its
  ROI per measured u8e kernel results is now demonstrably marginal; stays
  a documented structural residual, not a blind spot.

## Batch 4 (same day, later): inverse-ymm landings + matched re-baseline

Witness: `/tmp/real/photo_512.yuv` 512x512, cq27, `--cpu-used 3`, tile 1x1,
1 worker — now a MATCHED pair (C arm re-profiled on the same cell).

- `7719f35` inv 16x16 ymm fused inverse (stacked-halves): 21.659G -> 21.322G,
  **-337M**, byte-identical.
- `f329857` rect816 lane-doubling (ymm on the 16-element axis, xmm on the
  other; ymm twins of idct8/iadst8/iidtx8): 21.322G -> 21.195G, **-127M**,
  byte-identical.
- `6884d15` fwd_8x8 rnd/cnt hoist + output preflight: -> 21.184G, **-11M**
  (real; a -104M apparent kernel drop was LLVM un-merging a closure shared
  with fwd_rect816 — renumbering noise, net measured on the total).

### Matched re-baseline (same witness, same settings)

Port **21.184G** vs C **15.256G** = **1.388x** (this cell; the retained
1024^2 ship-cell gate stands at 1.384x median). Matched flat deltas:

| item | port | C | note |
|---|---|---|---|
| optimize_txb | 5.75G | 5.45G | near parity (+5%) |
| fwd txfm cluster | ~1.45G | ~0.80G | 8x8 428M vs 272M; per-call 397 vs 208 — codegen, not lanes (both xmm) |
| inv txfm cluster | ~1.0G | ~0.55G | post-ymm: 16x16 w16 394M, rect816 xmm+ymm 351M, 8x8 214M |
| quantize_fp | 498M | 276M | per-call 245 vs 143 Ir; instruction-level mirror already — residual is safe-Rust codegen + dispatch |
| z3_cols | 196M | 77M | already band-transposed; per-8col scalar setup |
| z2 cluster | 328M | 237M | left gather already tile-transposed |
| txb_init_levels | 265M | 180M | already AVX2, shape arms; residual is pack fixups |
| nz_map_contexts | 138M | 38M | safe-Rust scatter + per-tile checks; bounded ~1.4x on scatter |
| getenv | ~0 | 350M | port caches env reads (OnceLock) — a port WIN |

The transform gap is no longer lane-width on the inverse side (ymm landed);
what remains is per-instruction codegen density inside already-SIMD kernels
— the same class as the measured `#[inline]`/hoist experiments that net
~0-11M each. The next real structural lever remains the bd8 u8 plane split
(documented in `encoder_plane_refactor_concurrency_2026-09-17.md`), whose
ROI is bounded by the measured u8e results: ~15% on 16px-dominant paths.
