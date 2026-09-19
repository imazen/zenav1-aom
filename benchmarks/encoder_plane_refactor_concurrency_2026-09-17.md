# bd8 u8-plane refactor — dependency + concurrency analysis

2026-09-17, branch `perf/gate3-txfm-i16-batch`, witness: `photo_512` cq27 s3
(`/tmp/real/photo_512.yuv`), port 21.322G Ir vs C 9.389G-equivalent baseline
(earlier witness settings); serial residual map from
`/tmp/cg_w16.out`.

This is the analysis behind the standing "u16-at-bd8 tax" residual: the
encoder stores every plane as `u16` at every bit depth while C runs its
lowbd (`u8`) kernels at bd8. This document maps every producer/consumer of
the source and reconstruction planes, identifies which consumers could run
u8-natively, prices the transition, and proves (or refutes) safety under the
tile-row parallel walk.

## 1. Where the u16 planes live today

`crates/aom-encode/src/key_frame.rs`:

- `src_y/src_u/src_v: Vec<u16>` — allocated at `stride * buf_h` where `buf_h`
  includes border-extension padding past `mi_rows*4` (key_frame.rs:2895).
  `extend_plane` fills the padding so edge reads never go out of range.
- `recon_y/u/v = src.clone()` — phase-1 reconstruction planes; the tile walk
  predicts into them and inverse-transforms in place
  (`encode_intra.rs` writes `recon[txb_off..]` directly — prediction and
  `av1_inverse_transform_add_into` both write the plane).
- `deblocked_y/u/v = recon.clone()` — post-walk; `loop_filter_frame_opt`
  runs in place. At bd8 this internally narrows to `Vec<u8>`, runs the u8
  loop-filter walk, widens back (loopfilter/frame.rs:966-999).
- `cur_y/u/v = deblocked.clone()` — only when restoration is on; `cdef_frame`
  applies the picked CDEF into it. Read by the LR search as the post-CDEF
  "current" frame.
- `recon2_y/u/v = src.clone()` — phase-2 repack planes (key_frame.rs:4164).
  Separate allocations — phase 2 re-encodes every SB's committed tree from
  scratch and must not alias phase-1 recon.

So at bd8 there are **up to five full-frame u16 copies** alive in the
post-walk stages (src, recon, deblocked, cur, recon2) plus the u8
narrow/widen round-trips inside `loop_filter_frame_opt` and `try_filter_plane`.

## 2. Consumer matrix (who reads/writes each plane, and the u8 feasibility)

| consumer | plane | access | u8-native at bd8? |
|---|---|---|---|
| intra predictors (`dr_predict_high`, DC/smooth/paeth/filter_intra) | recon edges + write dst | row/col edges of committed recon; write block | edges must come from recon; predictors are u16-lane SIMD (u8 twins exist for z1 via staged edge — `z1_rows_u8e`, −8.6M); a full u8 edge pipeline is C's own shape (assemble u8 edges → u8 filters → `maddubs` predictors) |
| forward transforms / tx search | src + pred | block reads | already staged: `pred_u8` per txb, `sse_u16_u8` for dist (measured better than u8×u8 — one load vs two widenings); `subtract_block_u16_u8` |
| `av1_inverse_transform_add_into` | recon | in-place add into plane | u8 dst variant exists (`InvDst::U8`) used in candidate scratch; committed path writes u16 plane |
| CFL `cfl_store_tx` | recon (luma) | sequential reads of just-reconstructed luma during the walk | storage-typed; reads whatever recon is |
| IntraBC `intrabc_predict_luma/chroma` + hash table | recon + src | block copy at DV offsets + hash over src | u16 today; u8 twin is a memcpy-type kernel — easy |
| loop filter search `try_filter_plane` | recon copy + src | per-trial clone→filter→SSE | **the big u8 win**: today pays u16 copy + u16→u8 narrow + u8 filter + u8→u16 widen + scalar u16 `sse_plane` per trial |
| deblock apply `loop_filter_frame_opt` | deblocked | in-place filter | already narrows internally to u8 at bd8 — the round-trip is pure overhead if the plane were u8 |
| CDEF search `av1_cdef_search_adaptive` | deblocked + src | 8x8 unit reads | kernels already u8-capable at bd8 (C's `cdef_filter_block` lowbd path) |
| CDEF apply `cdef_frame` | cur | in-place | same |
| restoration search `pick_filter_restoration` | src + deblocked + cur | RU-windowed reads, per-candidate stats | kernels work on i32 box-sums of u8-range pixels; u8 lane-density port is C's `maddubs` shape — the 740M-vs-300M residual |
| `LfSearchFrame` SSE (`sse_plane`) | recon copy vs src | whole plane, scalar | **scalar u16×u16** — no SIMD; u8×u8 `aom_dsp::dist::sse` exists |
| CNN-prune window | `src_y_frame` (full frame) | crop-clamped full-frame read | read-only; u8 staging of src is a one-time cost |
| phase-2 repack | recon2 + src | same as phase-1 walk | mirrors phase 1 |
| HOG / `ssim_rdmult_scaling_factors` | src | gradient windows | u16 i16-lane kernels; u8 would need widening anyway |

## 3. The concurrency model — what a refactor must preserve

Phase 1 tiles (key_frame.rs:3653-3793):

- Tile **rows** are the parallel unit. `n_workers = cfg.threads.min(n_tile_rows)`;
  `band_y`/`band_uv` map each tile row to a plane-element range
  `(tile_row_start*4*stride, tile_row_end*4*stride)` — SB-aligned so chroma
  bands partition exactly by `>> ss_y`.
- `split_row_bands` produces disjoint `&mut [u16]` bands of each recon plane
  via `split_at_mut` — **no unsafe, no aliasing**. Bands are handed out
  through a `Mutex<(cursor, Vec<Option<&mut>>)>` work-stealing cursor; each
  band is `take()`n exactly once.
- Inside a worker, `pack_tile_stop` walks all tile columns in the row
  serially; the recon band is `&mut wy[..]` — a tile-local sequential walk
  where intra prediction legitimately reads already-committed rows/cols
  **within the tile** (availability is clamped to `tile_row_start`/
  `tile_col_start`, and IntraBC DV limits clamp to tile bounds too).
- The ONLY full-frame read inside the walk is the CNN-prune window through
  `env.src_y_frame` (immutable `&src_y`, crop-clamped — safe to share).
- `src` slices handed to workers are `&src_y[band_y[tr].0..]` — immutable,
  deliberately extend *below* the band end (bottom SB padding reads stay
  correct); reads above the band start are impossible by construction.
- Worker results merge into frame-raster `frame_trees` after
  `map_workers` (std::thread::scope join) — all recon writes are complete
  and synchronized before `lf_pick`/`deblock`/`cdef`/`lr` run serially.
- Phase 2 repeats the same band structure on `recon2_*` — separate
  allocations, so no aliasing with phase-1 state.

**Invariants a u8 refactor must keep:**

1. Disjoint `&mut` band partition per tile row, no unsafe. Any u8 recon
   plane must be splittable the same way — `Vec<u8>` + `split_at_mut` is
   identical mechanics.
2. The intra walk's sequential dataflow: predictor reads above/left recon
   *within the same tile* — a u8 recon is fine, but a *dual* u16+u8 recon
   would fork the source of truth and race across bands (predictors would
   need to read the same representation they write).
3. Post-walk stages run after join — their representation is independent
   of the walk's; they may be u8-staged without touching concurrency.
4. Phase 1/phase 2 recon planes are separate allocations; do not share.

## 4. Design options priced

**A. Global `Vec<u8>` planes at bd8.** Rejected as the first step: ~300
touch sites; the highbd paths share every signature (`&[u16]` on
`SbEncodeEnv`, `pack_tile_stop`, `encode_intra`, `LfSearchFrame`,
`LrPlanePixels`, CDEF frame, `cfl_store_tx`, IntraBC). A generic
`Plane<T>` refactor would double the monomorphized encoder or force
trait-object dispatch in the inner loop. It is also *not clearly needed*:
the measured u8-vs-u16 kernel gains (z1 +76M→−13M kernel, net −8.6M after
staging; z2 measured *worse* on short rows) show lane density only pays on
16px-dominant paths — the tax is concentrated in per-candidate/per-trial
*conversion*, not the walk's committed pixels.

**B. Post-walk staging (recommended first step).** Convert
`LfSearchFrame`/`LfFrameBuf`/`CdefSearchFrame`/`LrPlanePixels` to u8 at
bd8 — stage `src8`/`recon8` ONCE per frame (one u16→u8 pass per plane),
then every `try_filter_plane` trial is `u8 copy + u8 filter + u8 SSE`
instead of `u16 copy + narrow + filter + widen + scalar SSE`. Restoration
search reads staged u8 planes; CDEF search/apply likewise. The deblock
apply's internal narrow/widen disappears. All serial post-join code — zero
concurrency surface. Estimated −150-300M on this witness class (the
per-trial conversions are ~1MB round-trips × tens of trials, plus
`sse_plane`'s scalar loop → SIMD `sse`). **Risk:** the u8 variants of the
loop-filter/CDEF/restoration kernels must be byte-proven — the lpf u8 walk
already is (`loopfilter_lowbd_diff`); the deblocked plane at bd8 is exactly
u8-range by codec invariant, so `.min(255)` staging is a no-op gate.

**C. u8 recon plane in the tile walk.** Predictors write u8, i8-add
inverse transform, CFL/IntraBC read u8 — eliminates the per-txb u16
storage and the `InvDst::U16` committed path. Half the memory traffic in
the walk, and `pred` needs no staging. **Blockers:** every recon-plane
consumer in `encode_intra`/`encode_sb`/`pack` is `&mut [u16]`-typed;
predictors' u8 SIMD coverage is partial (only z1 has a u8e kernel today);
the phase-2 repack duplicates the surface; chroma strides/padding and the
full-frame `src_y_frame` read must stay consistent. This is the "real" fix
for the +490M chroma/committed-recon tax but it is a multi-hundred-site
program.

**D. Dual representation (u16 walk planes + u8 post-filter planes).**
Equivalent to B if the u8 staging happens at the lf_pick boundary; the
u16 planes remain authoritative through the walk. B is D without the
dual-walk complexity — same thing, honest framing.

## 5. Concurrency hazards specifically checked

- **Cross-band aliasing:** a u8 refactor that keeps `split_at_mut` bands is
  trivially safe; one that introduces a shared `Vec<u8>` scratch per worker
  must keep it worker-local (scratch is already per-worker: `TxWalkScratch`,
  `OdEcEnc`, `KfFrameContext` are constructed inside the worker closure).
- **Neighbor reads across tile rows:** the walk reads recon above `r0` only
  for the top-edge of a tile *column* in the same row — never across a tile
  row (availability is tile-bounded). The post-filter stages read the whole
  plane but run after join.
- **CDEF/loop-filter cross-row reads:** both read ±2 rows of neighboring
  pixels — but they run serially post-join on full planes. If they are ever
  parallelized by row band, the band split must extend ±2 rows (halo) or use
  boundary-line staging like C's `save_boundary_lines`. Not today.
- **Restoration search parallelism:** `pick_filter_restoration` takes
  `threads: n_workers` — check it partitions by RU rows on immutable planes
  (it does: RUs read src/deblocked/cur immutably and write only per-RU
  params).
- **`sse_plane` scalar:** not a concurrency issue, just ~3-4 instr/px that
  u8 SIMD halves and vectorizes.

## 6. Recommendation

1. Land B first: stage `src8`/`recon8` (u8) once per frame at the post-walk
   boundary; add a u8 `LfSearchFrame`/`sse_plane` twin; run lf_pick, CDEF
   search, deblock-apply and LR search on u8 planes at bd8. Pure post-join,
   no concurrency surface, collapses the per-trial conversion tax.
2. Keep the u16 walk planes (A/C deferred) — the measured u8e predictor
   results show the walk's committed-pixel traffic is not where the residual
   lives; revisit only if B + kernel work leave >100M in the tx walk's u16
   pixel path.
3. Do NOT stage `src` to u8 inside tx search (measured: `sse_u16_u8` beats
   `sse_u8_u8` by an op per 16 px — 6 vs 7 ops; `subtract_block` is scalar).

## 7. Addendum — what landed and what the measurement said (2026-09-18)

Landed (all byte-identical, all pushed):

- `b8f9cd6` — `LfSearchFrame<P: LfTrialPixel>` + `stage_lowbd`/`as_lowbd`:
  lf pick stages the six planes u8 once and runs trials u8-native.
  **−92.5M Ir** at 512² (2x2/4w), levels bit-identical vs live C.
- `944ee88` — the deblock APPLY at bd8 filters the staged-u8 recon in
  place and widens once into `deblocked` (clone + narrow + widen round
  trip gone). Ir-neutral; structural win is the single staging owner.
- `d174bc8` — panic-free band handout (`into_iter` cursor) +
  `KeyFrameError::InternalInvariant`; `9ebf4e8` — `SbEncodeEnv::
  {y_off,uv_off}` centralizing the band-offset convention.

Measured-and-deferred, contra §6 step 1's breadth:

- **CDEF stays u16.** `cdef_frame_u8` exists and is byte-identical but
  measures ~6.6% heavier than narrow→u16-CDEF→widen (its working domain
  is u16 and the u8 stores don't vectorize in the current abstraction).
  Routing CDEF search/apply through u8 planes would be a pessimization.
- **LR stays u16 for now, and that is a kernel program, not a storage
  one.** `PlaneCtx` stages padded u16 copies (`dgd_pad`/`dst_pad`) and
  borrows `src` regardless of the walk's plane representation; the
  residual is the u16 SIMD kernel set (~1.0G Ir at 512²: acc_stat_line
  177M, calc_ab 145M, wiener 129M, pixel_proj_error 112M, sgr_final
  123M, …) vs C's ~380M lowbd twins. The fix is u8 twins of C's
  `compute_stats`/`pixel_proj_error`/`sgr_*` lowbd kernels reading
  staged u8 buffers — bounded per-kernel, byte-gated each.
- **Option C (committed-recon u8 in the walk) remains deferred** — the
  walk's residual is codegen density in already-SIMD kernels
  (optimize_txb at parity+, quantize_fp/z3/nz_map bounded), and the
  storage swap's unmeasured ROI does not cover its blast radius
  (predictors, transforms, CFL, IntraBC, pack, phase-2 repack).

Ship cell re-measured at HEAD: **1.29×** (median, 6 pairs, byte-exact
40,237 B); real photo_1024 cross-check 1.27×.

## 8. Addendum 2 — the u8 twins landed and Option C is now measured NO (2026-09-18)

The "kernel program, not a storage one" from addendum 1 is DONE and
measured:

- `937f9ea` — `LrPixel` generic read side: `dgd_pad8`/`src8` staged
  once per plane; `compute_stats`/`pixel_proj_error`/`calc_proj_params`/
  `get_proj_subspace`/`selfguided_restoration`/`integral_image`/
  `sgr_final_*`/`sse_*` consume u8 at bd8. x86 per-type loads as
  const-fn-pointer arcane helpers (devirtualized at monomorphization).
- `152da43` — apply side: `filter_unit`/`wiener`/`apply_selfguided`/
  `StripeBoundaries`/`StripeScratch` generic; boundary rows narrowed u8
  via `BndStore`; u16 `dgd_pad` unallocated at bd8; `dst_pad` u16.

Measured (eprof_yuv real photo_1024, 1024² cq27 s3, interleaved):
**~1% wall** total; **−112M Ir at 512²** (21.000G vs 21.113G). Every
kernel Ir-flat to +5% — the gain is bandwidth/cache, confirming the
twice-the-bytes hypothesis but bounding its value at ~1% for the
entire LR read+apply conversion.

**Option C verdict: measured, rejected.** With read-side staging in,
the committed-storage swap's remaining upside is the two staging
narrows per plane plus apply-side stores — versus a ~44-site blast
radius over predictors/transforms/CFL/IntraBC/pack/phase-2. The
residual LR gap is ALGORITHMIC (integral-image vs C boxsum ~349M vs
~120M; acc_stat 88M vs ~40M — instruction counts a storage swap does
not move). That is now the named residual: a boxsum/maddubs-shaped
kernel-port program, byte-gated per kernel.

Ship cell unchanged: **1.295×** (byte-exact 40,237 B). Side finding:
at 4 workers the port beats the C shim on real photo_1024, 0.80×.

**1T re-verified** (ruling out core-count compensation): at workers=1
the same interleaved A/B is new 2391.3 ms vs base ~2409.7 ms = **−0.8%**
— the gain is single-core-real, not a scheduling/bandwidth-contention
artifact. Port-vs-C at 1T is **1.27×** (C is worker-insensitive; the
0.80× at 4T is the port scaling). Output bytes identical across 1T/4T
and across arms — the staging is deterministic by construction.
