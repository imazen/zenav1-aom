# Decode hotspot profile + rav1d-safe structural gap — 2026-07-17

Why aom-rs decodes 2K/4K photographic KEY-frame stills ~2.2× slower than
rav1d-safe (`benchmarks/decode_4way_2026-07-17.csv`), where the time goes, and
which levers are real. **Analysis-first**; the landable easy wins are small and
enumerated at the end.

- Repo commit: `ed606df`. Host: `dev-32gb`, 16 cores. Tool: valgrind-3.26.0
  callgrind (Ir = instructions retired; **load-tolerant**, valid on a busy box).
  Build: `profiling` profile (release + debuginfo), **no** `-Ctarget-cpu`.
- Driver: `gate3_profile dec port <cell> N` (adds the `dec_mosaic_*` cells to
  `aom_bench::decode_cells`). Reproduce: `just profile dec port dec_mosaic_2k_cq20 3`.
- Raw outputs (not committed, >800 KB): `/tmp/cg_mosaic2k_cq{20,40}.out`.

## 0. The headline workload is NOT what the prior profile measured

`benchmarks/gate3_profile_ranking_2026-07-16.md` profiled `dec_352x288_q32`, a
conformance vector that **codes CDEF (bits=3) + loop-restoration (Wiener)** — so
CDEF (27%) + LR (13%) dominated that ranking. **The headline 2K/4K photographic
cells are `aomenc --allintra` encodes → CDEF off, LR off, QM off** (verified by
parsing each stream's frame header):

| cell | tiles | bd | ss | base_qindex | deblock lvl | CDEF | LR |
|---|---|---|---|---|---|---|---|
| mosaic-2k/4k-cq20 | **1×1** | 8 | 4:2:0 | 80 | **4** (on) | off (bits0/str0) | NONE |
| mosaic-2k/4k-cq40 | **1×1** | 8 | 4:2:0 | 160 | **19** (on) | off | NONE |
| intrabc (2K conf) | **1×1** | 8 | 4:2:0 | 116 | **0** (off) | off | NONE |

So on the headline cells the only live post-filter is **deblock**; CDEF/LR do not
run. The correct hotspot ranking is the one below, on `dec_mosaic_2k_cq20`
(coeff-heavy, qindex 80) and `dec_mosaic_2k_cq40` (deblock-heavy, qindex 160).

## 1. Hotspot ranking — `dec_mosaic_2k_cq20` (4.83 B Ir: 1 C-oracle + 4 port decodes)

Self-Ir aggregated per function **plus** the std-lib lines callgrind attributes
to it via inlining (`clamp`/`cmp.rs`, `int_macros`, `slice::index`, `range`) —
those are the function's real cost. C-oracle (`ref_decode`) ≈ 268 M (~5.5%),
excluded from the port clusters. Shares are of the 4.83 B total.

| # | cluster (port) | ~Ir | ~share | owner | SIMD in rav1d? | aom-rs today |
|---|---|---:|---:|---|---|---|
| 1 | **inverse transform** — `av1_inv_txfm2d_add` ~734 M + `idct32/16/8` + `iadst16/8` | **~1.82 B** | **~38 %** | **aom-transform (OTHER agent)** | yes | scalar |
| 2 | **deblock** — `filter4` 224 M, `lpf_14` 140 M, `lpf_8` 105 M, `lpf_6` 71 M, `lpf_4` 37 M, `filter8` 36 M, `set_lpf_parameters` 95 M, `get_filter_level` 26 M, `get_transform_size` 24 M | **~810 M** | **~17 %** | **mine (aom-loopfilter)** | yes (AVX2) | **scalar (no simd.rs)** |
| 3 | **entropy decode** — `read_symbol` 224 M, `decode_cdf_q15` 223 M, `decode_bool_q15` 43 M | ~490 M | ~10 % | mine (aom-entropy) | **NO (scalar in both)** | scalar — at parity |
| 4 | **coeff / txb read** — `read_txb_body` 103 M, `dequant_txb` 69 M, `get_lower_levels_ctx` 66 M, `get_nz_mag` 85 M | ~323 M | ~7 % | mine (aom-txb) | dequant scalar; levels/ctx yes | mostly scalar |
| 5 | **memset / alloc** — `__memset_avx2` 133 M, `__memcpy` 36 M, `calloc` 33 M, `free` 18 M | ~221 M | ~4.6 % | mine + transform | arena (0 per-block alloc) | per-block `vec!` |
| 6 | **intra prediction** — `predict_highbd` 115 M, `z2_high` 27 M, `cfl_predict_block` 18 M | ~160 M | ~3.3 % | mine (aom-intra) | yes (AVX2/512) | scalar |
| 7 | `decode_block` control-flow | ~89 M | ~1.8 % | mine (aom-decode) | n/a | scalar |

`dec_mosaic_2k_cq40` (qindex 160) shifts the mix as expected: transform stays #1,
**deblock rises** (`filter4` 3.2 %, `lpf_14` 2.9 %, `set_lpf_parameters` 2.3 %),
`predict_highbd` rises to 2.5 % (more prediction per coded byte), entropy/txb
fall (fewer coefficients). Deblock is **~17–20 %** on both.

**One-line reading:** transform (38 %) + deblock (17 %) + intra-pred (3.3 %) —
all SIMD in rav1d, all scalar in aom-rs — are **~58 %** of the decode. Entropy
(10 %) is scalar in *both* decoders, so aom-rs is at parity there. The gap is
almost entirely **un-vectorized pixel kernels**, not algorithm and not threading.

## 2. rav1d-safe structural gap (source-verified at `/root/work/rav1d-safe`)

### Threading — rav1d does NOT thread these stills

- **Frame threading is compiled OUT**: `let n_fc = 1;` (`lib.rs:124-128`), every
  path asserts `fc.len()==1` (`decode.rs:4929`). No frame pipelining, ever.
- **The managed API defaults to `threads: 1`** (`managed.rs:196`) — the benchmark
  ran rav1d-safe **single-threaded**. Worker threads spawn only if the caller
  raises `n_tc` (`lib.rs:216-224`); at `n_tc==1` decode runs fully inline
  (`decode.rs:4815` "no threading … the full process runs in-line").
- Even at `n_tc>1`, a **1-tile** frame creates **exactly one** serial
  `TileReconstruction` task (`thread_task.rs:456,468`) that walks its sb-rows one
  at a time (self-re-enqueue `t.sby+=1`, `thread_task.rs:1266-1288`). Entropy +
  coeff + intra-pred + inverse-transform + reconstruct are **serial on one
  thread**. Only the post-filters (`DeblockCols/Rows/Cdef/…LR`,
  `thread_task.rs:1352-1458`) run as separate sb-row tasks that *pipeline behind*
  reconstruction — a bounded overlap, not N-way scaling, and on our cells CDEF/LR
  are off so only deblock could overlap.

**→ rav1d earns its ~2.2× on these single-tile stills running effectively
single-threaded. Threading is not the source of the gap.**

### SIMD — the actual source of the gap (rav1d no-asm build)

Mechanism: `core::arch` intrinsics behind the archmage token system
(`is_x86_feature_detected!` → `summon_avx2/avx512`, `cpu.rs:261-282`), *not*
portable_simd; the `asm` feature (hand asm) is OFF.

| kernel (still/intra path) | rav1d no-asm | aom-rs |
|---|---|---|
| intra prediction (DC/PAETH/SMOOTH/Z1-3/FILTER) | **SIMD** AVX2(+512) `ipred.rs:105` | scalar |
| inverse transform + reconstruct-add (fused) | **SIMD** AVX2(+512) `itx.rs:451`, `recon.rs:1471` | scalar |
| CDEF | **SIMD** `cdef.rs:111` | SIMD (`aom-cdef/simd.rs`) — but off on these cells |
| deblock / loopfilter | **SIMD** AVX2 `loopfilter.rs:103` | **scalar** |
| loop restoration | **SIMD** `looprestoration.rs:181` | scalar — off on these cells |
| **dequant** (coef×q) | **scalar** `recon.rs:1121` | scalar — parity |
| **entropy (msac)** | **scalar** `msac.rs:515` | scalar — parity |

### Memory

- Picture buffers are **pooled/reused** across frames (`MemPool`, `mem.rs:25`;
  `picture.rs:154`). Reused pages skip re-zeroing.
- **Zero per-block heap alloc**: coefficients decode into a persistent per-task
  arena `t.cf` (`recon.rs:1389`); intra/palette/edge scratch is a persistent
  union reused every block (`internal.rs:1028`). aom-rs allocates fresh
  `vec![0i32; area]` + `vec![0u16; …]` per coding block (`aom-decode/src/lib.rs:2101-2102`)
  and per txb inside the transform (`aom-transform/src/inv_txfm2d.rs:128,195`).

## 3. TILING verdict (the specific question)

1. **Do the streams use multiple tiles?** No — **every headline cell is 1×1
   (single tile)** (parsed from `tile_info.cols*rows`; `aomenc --allintra` at
   these sizes defaults to one tile — a 4K frame fits one tile). The intrabc conf
   vector is also 1×1.
2. **Can aom-rs decode tile-parallel?** Structurally it *could* — tiles are
   independent (`decode_frame_tiles_kf`, `aom-decode/src/lib.rs:3123`, gives each
   tile its own `OdEcDec` + fresh `KfFrameContext`), but the loop is a plain
   serial `for tb in tiles`. Adding tile-parallelism is mechanically clean
   (independent byte ranges → disjoint recon regions, deterministic).
3. **Realistic parallelism for STILLS — honest verdict:**
   - **Tile-level: ZERO for these stills** (1 tile). It would only help
     multi-tile video / deliberately multi-tiled encodes.
   - **Within-frame post-filter row-threading** (dav1d's trick): bounded, and on
     these cells CDEF/LR are off so only the ~17 % deblock could overlap — a few
     percent of wall at best, for a task-queue's worth of complexity. **rav1d
     gets its full 2.2× WITHOUT doing this** (single-threaded).
   - **Entropy + coeff decode of one tile is inherently serial** (arithmetic
     decoder) — not row-splittable without re-deriving per-row bit offsets, which
     AV1 stills don't provide (no per-sb-row entry points within a tile).
   - **→ For single-tile stills, SIMD is the only lever that matters.** Threading
     is a non-lever here; pursue it (if ever) only for multi-tile or multi-frame
     workloads, as a separately-gated deterministic pass.

## 4. Existing SIMD infrastructure (already proven in-repo)

The archmage dispatch pattern is **already established and differential-gated**:
`incant!(…, [v3, neon, wasm128, scalar])` + `aom_dispatch::scalar_forced()` +
the `AOM_FORCE_SCALAR` process-wide pin + `just test` / `just test-scalar`. Live
today: `aom_txb::txb_init_levels` (magetypes i32x8, `aom-txb/src/simd.rs`, pinned
by `txb_init_levels_simd_diff.rs`) and CDEF (`aom-cdef/src/simd.rs`). So new SIMD
kernels are *templated*, not greenfield — but each is still a real kernel + a
per-kernel differential vs the scalar twin, run under both token modes. That is
why none of them is "trivial."

## 5. Ranked easy-wins / proposals

Bit-exactness is the hard constraint (this decoder is the conformance oracle).
Every SIMD kernel must be per-kernel differential-identical to its scalar twin
and pass the full decode suite + golden-MD5 gate under **both** `just test` and
`just test-scalar`.

### (A) SIMD on hot non-transform kernels — real payoff, NOT trivial

| target | file:line | ~payoff | effort | risk |
|---|---|---|---|---|
| **deblock** (new `aom-loopfilter/src/simd.rs`): `lpf_4/6/8/14` + `filter4/8` + vectorize the `filter_block_plane` mask/level derivation | `aom-loopfilter/src/highbd.rs:98,134,179-213`; `frame.rs:393` | **~15–17 %** | high (data-dependent flat/hev masks per edge; 4 filter widths; the `set_lpf_parameters` per-4px scalar derivation is itself ~2.4 % and should become vectorized whole-region masks like libaom/rav1d) | med — deblock math must stay bit-identical; differential per filter width |
| **intra prediction**: `predict_highbd` (DC/PAETH/SMOOTH are reduction+broadcast, very SIMD-friendly) + `z2_high` directional | `aom-intra/src/lib.rs:209`; `dir.rs:215` | ~3.3 % | med | med |
| **dequant_txb** (lane-pure; for these cells `iqmatrix==None` → dc/ac scalar, no matrix) + `get_nz_mag` | `aom-txb/src/read.rs:253`; `aom-txb/src/lib.rs` | ~2–3 % | med (follows the `txb_init_levels` template closest) | low-med |

Ordering rationale: deblock is by far the largest single MY-territory lever
(bigger than intra+txb+alloc combined) and its crate has **no** `simd.rs` yet.

### (B) alloc-reuse / bounds-check — small, oracle-risk-sensitive

| target | file:line | ~payoff | effort | risk |
|---|---|---|---|---|
| Hoist per-block scratch (`tcoeff`, `scratch`, `txbs`, `txbs_uv`, `leaves`, `nu_tcoeff/nu_scratch`) to reusable `TileKf` fields sized to max — matches rav1d's zero-per-block-alloc arena | `aom-decode/src/lib.rs:2101-2102,2135-2136,2247` | ~1–1.5 % (calloc 33 M + free 18 M; wall a bit more) | med | **med — touches the oracle's hottest fn (KB-1 64×64-chunk ordering lives here); must pass full golden-MD5 gate** |
| Per-txb `vec![0i32; …]` in the inverse transform (same arena idea) | `aom-transform/src/inv_txfm2d.rs:128,195` | part of the 4.6 % memset/alloc | med | **OTHER agent (aom-transform)** — flagged for them |

Not landed: the ~1 % payoff does not justify risking the oracle's byte-exactness
on `decode_block` in this pass. Proposed with exact sites for a follow-up that
runs the full byte-gate.

### (C) threading — DESIGN NOTE, do not build for stills

Verdict from §3: **not applicable to single-tile stills.** rav1d achieves 2.2×
single-threaded. If pursued later it must be a separately-gated, output-identical
pass (deterministic by construction): tile-parallel only helps multi-tile
streams; post-filter row-pipelining is a bounded (~few-%) win here because CDEF/LR
are off. Not worth the task-queue complexity while ~58 % of the frame is
un-vectorized scalar pixel work.

## 6. Landed vs proposed (this pass)

- **Landed:** the four `dec_mosaic_{2k,4k}_cq{20,40}` decode cells added to
  `aom_bench::decode_cells` (skipped gracefully when the gitignored `.ivf`s are
  absent, via `DecodeCell::from_vector_opt`) — so the **headline** stills-decode
  workload is now a first-class profiling target (the repo previously only
  profiled the CDEF/LR-coded conformance vectors, which mis-rank the real work).
  Zero decoder-path change → byte-exactness untouched.
- **Proposed (not landed):** every (A)/(B) perf item above. None met the
  "genuinely-easy + oracle-safe" bar: the (A) SIMD kernels are real
  differential-gated ports (hours each), and the (B) alloc-reuse is ~1 % on the
  oracle's most delicate function. Highest-value next step: a new
  `aom-loopfilter/src/simd.rs` for the deblock filters (~15–17 %), following the
  established `txb_init_levels` / CDEF dispatch+differential template.
