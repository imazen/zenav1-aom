# Changelog

## Workspace

### [Unreleased]

### Added

- **The high-bit-depth INTER decode envelope is now gated against the live C
  decoder, and GitHub #8 does not reproduce (KB-40).** #8 reported that
  `colors-animated-12bpc-keyframes-0-2-3.avif` frame 1 — an inter frame —
  decodes to different RGBA than rav1d-safe. New gate
  `aom-bench/tests/highbd_inter_decode_envelope.rs` diffs the port's
  `decode_frames` against **`aom_codec_av1_dx` in-process** rather than against
  the md5 goldens committed beside the animated fixtures: **8/8 tracks, 40/40
  shown frames byte-exact**, including that vector's 12-bit 4:2:2 color track
  (frame 1 included) and its 12-bit monochrome alpha track. The sweep arm
  extends `inter_harness_chunk0`'s bd8-4:2:0-only envelope map onto the axes it
  never covered — real `aomenc` `[KEY, P]` clips at bd {8, 10, 12} x {4:2:0,
  4:2:2, 4:4:4, mono} x cq {20, 60}: **24/24 cells, 48/48 frames byte-exact**,
  so 12-bit intra AND 12-bit zero-MV inter are both inside the byte-exact
  envelope. The third arm pins the honest boundary: nonzero-MV inter above bd8
  is **refused** (`sub/nonzero-pel MC above bd8 not yet supported`), 8/8 cells,
  **0 wrong-pixel cells**. Every number identical under `AOM_FORCE_SCALAR=1`.
  The fixture was re-extracted from the live zenavif vector first and compares
  byte-identical, so a stale fixture is ruled out. Record:
  `benchmarks/highbd_inter_decode_envelope_2026-08-06.{md,tsv,meta}`.
  Additive oracle helper `aom_sys_ref::ref_decode_av1_stream_frame_opt` (returns
  `None` on "fewer shown frames" instead of panicking) so C's shown-frame COUNT
  is derived independently of the port's.

- **Four coverage axes swept, three closed byte-exact, and one new unmodelled
  speed-feature arm found (KB-38).** All four were named-but-unmeasured entries
  in the coverage queue. (1) *partial-SB x high bit depth beyond bd10 4:2:0* —
  7 formats (bd10 {mono, 4:4:4, 4:2:2}, bd12 {4:2:0, mono, 4:4:4, 4:2:2}) x
  KB-23's four sizes x `--cpu-used` {0, 7}, **56/56 byte-exact**
  (`s4cov_partial_sb_axis::partial_sb_high_bitdepth_formats_byte_match`).
  (2) *the >=1080p band at every format KB-36 did not sweep* — 4:4:4 / 4:2:2 /
  monochrome / SB128 / cq5 / cq40 / cq63, each at both 1920x1072 and 1920x1080
  at `--cpu-used 6`, **14/14 byte-exact**. (3) *1440..2160 at speed >= 1* —
  1920x1920 + 2560x1600 x `--cpu-used` 1..9, **18/18 byte-exact**. (4) *crops
  straddling `is_4k_or_larger`* — 2154x2160 vs 2160x2160, **2/2 byte-exact in
  35 s** rather than the queue's estimated 25-30 min, because
  `is_4k_or_larger` is speed-unconditional and speed 5 observes it (the
  estimate had costed the arm at the speed it was *found* at). New gate file
  `aom-bench/tests/s4cov_hd_format_axis.rs`; record
  `benchmarks/s4cov_hd_format_2026-08-04.{tsv,meta}` +
  `benchmarks/s4cov_partial_sb_hbd_2026-08-04.{tsv,meta}`. (f99ac44, f085be4)

- **bd12 x the 480/720 crop straddle — the last cheap residual of KB-28's
  format axis.** 474x480 and 714x720 plus their SB-exact controls at their own
  mi-aligned extents, at `--cpu-used` {0, 7}, **8/8 byte-exact** in 53 s
  (`s4cov_crop_format_axis::crop_straddle_bd12_byte_matches_where_interpretable`).

- **KB-38 — `av1_set_speed_features_qindex_dependent`'s
  `is_1080p_or_larger && base_qindex <= 108` sub-block (speed_features.c:2926-2935)
  is now modelled.** It was omitted under a comment claiming the whole block was
  "all inter-only, and the port carries no field for them" — both halves false:
  the port carries all five fields and four are intra-live, including
  `skip_tx_search`, which `search_tx_type` reads directly (tx_search.c:2362).
  The window is three terms wide (`speed == 0` x `min(w,h) >= 1080` x
  `base_qindex <= 108`), so KB-36's >=1080p grid (`--cpu-used` 1..9) and
  KB-19/KB-22's speed-0 2160p cell (cq32 = qindex 128) each missed a different
  term. **The cells it moves are not closed** — porting it took bd8 1920x1080
  cq24 from -536 B to -726 B without closing it, and a twelfth row diverges
  outside its predicate — so the 12-cell map is recorded as a self-promoting
  pin (`speed0_1080p_band_map_is_pinned`), not as a fix. Unit lock:
  `speed_features::tests::qindex_dependent_speed0_1080p_q108_subarm`. (f085be4)
- **`--deltaq-mode` 2/3 (and the `--delta-lf-mode=1` that rides on them) now
  encode on MULTI-TILE frames — the last T1 refusal reachable by flags (the one
  remaining is blocked on producing an input no encoder emits).** The harness
  refused the combination outright (*"multi-tile x per-SB delta-q is unmodelled
  … — see KB-31"*), and the queue's recorded fix was backwards: C re-seeds
  `xd->current_base_qindex` from the frame `base_qindex` at the top of **every
  tile**, on all three sides (`encode_sb_row`, encodeframe.c:1232-1239;
  `write_modes`, bitstream.c:1745-1751; `decodeframe.c:2948`/`:3023`), so
  `pack_tile`'s per-tile restart was already right and the harness's own
  frame-raster replays — which DERIVE `delta_q_present` and the per-SB delta-lf
  rather than reading them off the bootstrap — were not. One root, replaced by a
  tile-ordered `replay_sb_qindex_tile_order`. Also corrects the queue's
  reachability note: `AV1E_SET_TILE_COLUMNS`/`_ROWS` reach the axis at ANY frame
  size, no 4033 px needed. New gate
  `aom-bench/tests/kb31_deltaq_multitile.rs`: 53 byte-identical cells (the
  size-forced 4096x64 split with both a single-tile and a deltaq-0 control; the
  {1x1, 2x1, 1x2, 2x2} grid matrix x {deltaq 0, 3, 2}; the same grids with
  `--delta-lf-mode=1`) plus 6 port-decoder round-trips and an `--ignored`
  9.55 MP area-forced ROW split. **Honest bound:** no byte-level bite was found
  for the reordering — the pre-fix replay still passes all 53 cells — so this is
  a C-fidelity alignment plus the removal of a refusal, not a demonstrated byte
  fix; the bite is at unit level
  (`replay_resets_the_running_base_at_every_tile`). Two residuals recorded:
  delta-q diverges at `--cpu-used >= 1` **single-tile too** on some content
  (`DELTAQ_SPEED_OPEN`, T4), and asserts at `--cpu-used >= 8` (T2). See KB-39.

- **Lossless (`--cq-level 0`) now encodes byte-identically at every
  `--cpu-used` 0..9 — the last T1 refusal on a default-reachable configuration.**
  Two independent roots. (1) The e2e harness (`aom-bench`) parsed the frame
  header once with `coded_lossless = false` and refused qindex 0 outright, so
  KB-5's 2026-07-16 parity rested entirely on drivers that hardcode
  `speed = 0`; it now mirrors the decoder's two-pass `coded_lossless` probe and
  models `is_loopfilter_used` (encoder.h:4419). (2) The nonrd estimate arm's two
  TX_4X4 `block_yrd` arms were `unimplemented!()` — they are the CODED-LOSSLESS
  arms, because `select_tx_mode` returns `ONLY_4X4` there (rdopt_utils.h:392),
  which `nonrd_leaf_tx_size` had modelled as a constant TX_64X64. Ported from
  `nonrd_opt.c:246-263`: `aom_fdct4x4_lp` + `av1_quantize_lp` (lowbd),
  `aom_fdct4x4` + `av1_quantize_fp` (hbd), both over the normal
  `av1_scan_orders[TX_4X4][DCT_DCT]` pair. **`aom_fdct4x4` turns out to be
  ISA-conditional** — both SIMD tiers are int16-only where `_c` uses
  `tran_high_t`, invisible at a 9-bit residual and real at bd10/bd12 (new
  `docs/LIBAOM_UPSTREAM_NOTES.md` **A6**) — so the hbd arm models the dispatched
  kernel; substituting `_c` diverges 4 of 8 hbd cells. New gates:
  `aom-bench/tests/kb5_lossless_speed_axis.rs` (52 lossless cells, 0 divergences,
  + 10 cq1 controls, with estimate-arm reach asserted) and `aom_fdct4x4{,_lp}` locks against the real
  exported C symbols in both `nonrd_block_yrd_{lp,hbd}_diff.rs`.

- **A committed content census, and a third `winperf` content fitted to what it
  measures
  ([`benchmarks/winperf_content_census_2026-08-03.md`](benchmarks/winperf_content_census_2026-08-03.md)).**
  `winperf`'s two synthetic sources were fitted on *allocator call count* and
  were then quoted for a lever inside the directional intra predictors, where
  `detail` reaches `z1` six times in a whole 1 MP frame — so the Windows band
  was reading a structural zero. `aom_dsp::census` (new, behind a **default-off**
  `census` feature, so timing builds carry no counter) plus
  `crates/aom-bench/examples/content_census.rs` report intra mode family x
  transform size, forward transform type x size, and coded leaf size for any
  content against a reference. `examples/content_fit.rs` used that to fit a new
  `winperf::Content::Photo` over 467 candidates on a pre-declared objective:
  intra-class L1 to the study photograph falls from **47.4 pp** (`detail`) to
  **5.7 pp**, and directional predicted pixels from **0.15 %** to **17.9 %**
  (reference 20.8 %). It does **not** replace `detail` — its allocator traffic
  is 73 % of the reference where `detail`'s is 95 % — so
  `.github/workflows/winperf.yml` gains a `contents` input instead of a fixed
  pair, and `scripts/winperf_prepost_stats.py` (new) reads a `prepost` band with
  the identical copies on each side pooled. With the new content the KB-PERF-4
  lever resolves on `windows-11-arm` at **−0.50 pp against a `detail` control**,
  matching Darwin's −0.49 pp on the same binaries; `windows-latest` x86-64 stays
  unresolvable and now has a measured floor (MDE 0.50-0.86 % at n=24) saying
  why. No coded byte moves; gates 968/968 green.
- **The encoder has been profiled for the first time
  ([`benchmarks/encoder_hotspot_profile_2026-08-02.md`](benchmarks/encoder_hotspot_profile_2026-08-02.md)).**
  Every prior profiling artifact in `benchmarks/` was decode-side. The 10.8x
  matched-preset gap to libaom that `xbench_2026-08-01.md` measured reproduces at
  **10.72x** (9 interleaved invocations per arm, 2.88 % / 5.06 % control spread)
  and is **concentrated, not diffuse**: `cnn_partition::cnn::cnn_predict` is
  74.7 % of the port's whole encode and **81.5 % of the entire gap**, because the
  port recomputes the intra-mode CNN at every 64/32/16/8 node (2558 runs/frame)
  where libaom computes it once per 64x64 and caches (256). Take it out and the
  port is ~3x libaom — confirmed at `cpu-used` 7/8 where the CNN never runs
  (2.69x / 3.45x measured). Ranked levers, per-stage port-vs-libaom absolute-ms
  alignment, an exact allocation census (870 167 allocator calls / 559.7 MB per
  1 MP encode) and a size/speed/quantizer breadth sweep are in the writeup;
  durable summary in `CLAUDE.md` **KB-PERF-1**. Nothing was optimized — this is
  measurement only. New reusable harness: `scripts/eprof_{control,sample,breadth}.sh`,
  `scripts/eprof_{rollup,align,callers}.py`, and
  `crates/aom-bench/examples/eprof_{alloc,cnn_bench}.rs`.

### Changed

- **All files inherited from upstream libaom now live in a subfolder or the
  submodule.** The upstream `LICENSE` (BSD-2-Clause) and `PATENTS` (AOM Patent
  License 1.0) moved from the repo root into
  [`upstream-notices/`](upstream-notices/) (byte-identical to the copies in the
  `upstream/` submodule; a `README` there records provenance). The full libaom C
  tree remains the pinned `upstream/` submodule; the gitignored working copy at
  `reference/libaom/` is untracked. No inherited C source is tracked outside a
  subfolder — the only tracked `.c` files are our own FFI oracle shims under
  `crates/aom-sys-ref/shim/`.

### Fixed

- **The differential oracle raced on libaom's lazy tables, which is what
  SIGSEGV'd `interintra_diff` on the aarch64 runner.** It was recorded as an
  unowned "runner flake"; it was a data race, and it predated every commit that
  was bisected for it. The oracle is built `CONFIG_MULTITHREAD=0`, which selects
  libaom's no-synchronisation `aom_once` (`upstream/aom_ports/aom_once.h:70-80`),
  and `av1_init_wedge_masks` publishes through a table it first `memset`s to
  zero — so a second libtest thread re-NULLs entries the first already published
  and the shim `memcpy`s from NULL. `aom_sys_ref::ref_init` now forces all seven
  of libaom's `aom_once`-guarded initialisers under one Rust `Once`, and the
  nine wrappers that run a real encoder (`av1_initialize_enc` calls six of the
  seven, unguarded) funnel through it. MEASURED on Apple Silicon: 16/400 (4.0%)
  SIGSEGV before, **0/1000 after**; the isolating gate
  `crates/aom-sys-ref/tests/wedge_init_race.rs` went 133/200 (66.5%) → 0/300.
  `shim_ii_wedge_mask` also returns `-1` on a NULL table so any future
  regression panics with a name instead of faulting — with the funnel removed
  but the guard in place, 121/200 and 17/300 failures were **all** named panics
  and none were signal 11. No coded byte moves. `docs/DIFFERENTIAL_PLAYBOOK.md`
  §11 + `docs/LIBAOM_UPSTREAM_NOTES.md` C7.
- **Encoder: `--cpu-used 9` refused to encode ordinary images — the nonrd
  estimate arm could not code a NON-SQUARE leaf (KB-34).** Since KB-32,
  `nonrd_pickmode::nonrd_leaf_tx_size` panicked with a named HANDOFF whenever
  `set_vt_partitioning` stamped a rect leaf, because `max_txsize_lookup` gives
  the square tx of the block's SHORT side and
  `av1_foreach_transformed_block_in_plane` then visits more than one txb. The
  refusal carried a measured claim that only a 12000x9000 frame reached it; a
  2,012-cell sweep found **609 reaching cells and the smallest at 100x100**. The
  predictor is not size or quality but whether the frame has a **partial
  superblock**: 68.9 % of partial-SB rows reach it against 1.7 % of SB-exact
  ones, so 1920x1080, 1280x720 and essentially any non-multiple-of-64 crop are in
  the reaching class. `av1_nonrd_pick_intra_mode` now runs C's real per-txb walk
  (per-txb predict into the recon plane before the next txb reads it, the
  TXB-sized `av1_block_yrd` clamp against the LEAF's `mb_to_*_edge`, and C's
  assign-don't-AND `skippable`), and the frame-edge single-strip rect
  constructor one line behind it got KB-25's poisoned-slot fix. Byte-identical on
  every newly-encodable cell measured, including the encoder profile's
  1024x1024 cq44 cpu-used 9 photograph and **issue #6's 12000x9000 108 MP frame
  (11,520,317 B, delta 0)**, which had been the refusal's own example.
  `kb28_crop_dims::vbp_band_crop_dims_byte_match` goes 28/30 → **30/30**. New
  gate `crates/aom-bench/tests/kb34_nonsquare_nonrd_leaf.rs`; sweep and
  per-root bite proof in `benchmarks/nonsquare_leaf_reach_2026-08-02.{tsv,meta}`
  and `..._bite_2026-08-02.tsv`.

- **Encoder: the nonrd (`--cpu-used` 8/9) estimate arm computed its Hadamard
  coefficients TRANSPOSED (KB-12).** `aom_hadamard_lp_8x8_c` ends with a
  transpose the SIMD tiers get for free (`aom_dsp/avg.c:232-236`, *"Extra
  transpose to match SSE2 behavior"*); the port wrote its intermediate buffer
  straight out, so `hadamard_lp_8x8` produced libaom's output transposed and
  `hadamard_lp_16x16` its per-64-quadrant transpose. Every order-invariant
  consumer — `aom_satd_lp`, `av1_block_error_lp`, the `eob == 0` skippable flag
  — was blind to it, so rate, distortion and skippability were correct and only
  the `eob` drifted, through `eob_cost += get_msb(eob + 1)`. That is why it read
  as a "leaf-mode near-tie" from 2026-07-17: it flipped occasional winners in
  `av1_nonrd_pick_intra_mode`'s four-mode loop and did nothing else. Closes the
  four pinned speed-8 `diag` cells (`encoder_gate_speed8_textured_allintra`
  60/64 → **64/64**), KB-32's entire surviving residual (the cpu8 size ladder
  512²–2048² and the 2176² cpu9 cell, all promoted from shape/bound pins to hard
  byte gates), KB-28's speed-8/9 rows (`vbp_band_crop_dims_byte_match` 18 open →
  0), and both `config_permutations` pin lists
  (`SPEED_OPEN_SINGLETONS` / `SPEED_OPEN_COMBINATIONS` are now **empty at every
  speed 0..9**). `_c` and `_neon` agree bit-for-bit over the reachable 9-bit
  residual domain, so nothing here is ISA-conditional (contrast
  `aom_hadamard_16x16`, KB-20 root #4). New gate:
  `crates/aom-encode/tests/nonrd_block_yrd_lp_diff.rs` — the lowbd twin of the
  hbd differential, locking all five estimate kernels and the `block_yrd_lowbd`
  walk against the exported C symbols. Its absence was the root's root.

- **Encoder: every frame with `min(w,h) >= 480` diverged from real aomenc at
  `--cpu-used >= 4` (KB-26).** The speed≥4 winner-mode two-pass in
  `partition_pick` derived its MODE_EVAL / WINNER_MODE_EVAL transform policies
  from a fresh `SpeedFeatures::set_allintra`, which is framesize-blind by design,
  so the framesize-derived `tx_type_search.prune_tx_type_using_stats`
  (`is_480p_or_larger`, `speed_features.c:261/299`) arrived as 0 for the whole
  luma tx search — while speeds 0..3, which use the caller's resolved policy
  directly, kept it. Fixed by `TxTypeSearchPolicy::carry_frame_level_tx_sf`,
  which carries the frame-level stage-independent tx-search inputs across the
  stage derivation. `s4cov_hd_speed_axis` goes 15/28 → 26/28 byte-exact (the two
  remaining are KB-28's speed-7 refusal) and the 256..640 `--cpu-used=4` size
  ladder goes 4/7 → 7/7. Record:
  `benchmarks/kb26_large_frame_speed4_2026-08-01.tsv`.

- **Test harness: the KB-13 real-content speed≥1 gate mis-reported partial-SB
  frames (196×196) as encoder divergences / "invalid streams".** The harness
  (`attempt_case_content_uv_sep`) walked `floor(mi/16)` superblocks over an
  unpadded `h+4`-row source, silently dropping the partial edge SB (196px = 50
  mi = 3.0625 SBs) and coding a short tile the real C decoder rejects. Given the
  KB-6 `run_case` partial-SB setup — `ceil(mi/16)` SBs over an SB-aligned,
  border-extended source (matching C's `aom_extend_frame_borders`) — the 196²
  cq63 cells byte-match real aomenc (4/12 promoted; the whole gate 41/60 → 45/60)
  and the rest are ordinary valid-stream near-ties. The port **encoder** was
  correct throughout (KB-6 speed-0 30/30); this was a harness-only bug.

- **Encoder intrabc (screen content): DV search + var-tx cost now match libaom to
  the unit at the KB-15 witness leaf mi(40,28)** — three independent roots, each
  localized by a byte-inert instrumented sibling-C dump (0cd64bf):
  1. the DV-search `error_per_bit` used the frame rdmult instead of the per-block
     `x->rdmult` (per-SB `intra_sb_rdmult_modifier` fold) — now
     `av1_set_error_per_bit(env.rdmult)`;
  2. the intrabc pixel search modelled NSTEP (12-point tangent stages) where
     libaom uses NSTEP_8PT (16 stages, 8-point, `tan=radius`) — the diamond is now
     parameterized by an `eight_pt` flag, intrabc passing NSTEP_8PT;
  3. the intrabc var-tx `txfm_partition_cost` was a frame constant instead of the
     per-SB (INTERNAL_COST_UPD_SB) value from the adapting `txfm_partition` CDF —
     `txfm_partition_costs` added to `RealCosts`/`SbEncodeEnv`.
  The port now finds C's exact `dv=(-816,-888)` and flips mi(40,28) to
  PARTITION_VERT matching C. Intrabc-only / per-SB-additive: intra envelope
  byte-inert (aom-encode+aom-bench 340/340). The witness stays PINNED (first-diff
  floor 1120) — the remaining byte-1120 divergence is a separate PACK-side residual.

### QUEUED BREAKING CHANGES

- **`zenav1-aom-decode`: `KfTileDecode.recon/recon_u/recon_v` are now
  `ReconPlane { LowBd(Vec<u8>), HighBd(Vec<u16>) }` instead of `Vec<u16>`**
  (bd8 frames store `u8` planes; `ReconPlane::to_u16()`/`px()` widen
  bit-exactly). `FrameDecode` and `RefFrame` stay `u16` — only consumers
  reading the pre-filter tile planes directly must migrate. (5336e65)

- **`zenav1-aom-decode` public entry points now return `Result<_, DecodeError>`
  instead of `Result<_, String>`.** `decode_frame_obus` / `decode_frames` (and
  the parse helpers) carry a structured, category-bearing `DecodeError` enum
  (implements `core::error::Error`; `pub use` of `DecodeError` + `LimitKind`).
  Consumers matching on the old `String` error must migrate to the enum. (c43440b)

### Added

- **bd8 decode: i16-lane inverse-transform ROW pass** — the five audited DCT
  kernels (idct4/8/16/32/64) run the row pass on `i16x16` lanes for every bd8
  transform with `row_n % 16 == 0`, byte-identical to the scalar port (same
  audited-domain design as the Phase-C columns; iadst/identity + short rows
  stay i32 by audit/design). Measured: narrowed row subset −45.4% Ir, whole
  4K decode −1.26% (cq20) / −1.71% (cq40); see
  `benchmarks/bd8_i16_rows_2026-07-23.md`. Validated 418/418 in default AND
  `AOM_FORCE_SCALAR` dispatch before landing. (9f49ebc)
- **CDEF `cdef_find_dir` per-row slice-add restructure** — the per-pixel
  8-direction scattered adds regrouped into per-row contiguous slice adds
  (byte-identical: wrapping adds commute; gated by the real-C
  `cdef_find_dir_matches_c` differential). Measured: 1570 → 442 Ir/call
  (−71.8%), q32 whole-decode 2.067× → 2.032×; see
  `benchmarks/gate3_filters_2026-07-22.md` fix 3. (ea61406)
- **Gate-3 decode wall baseline committed** — 4K stills decode ≈1.22× C (cq20)
  / ≈1.19× (cq40) after the full bd8 peak-perf series; the ≤1.5× acceptance
  bar is met at the 4K headline cells (2K/small cells 1.66–2.4×, levers
  ranked). Two agreeing runs + measurement caveats:
  `benchmarks/gate3_peak_wall_2026-07-25.{md,meta}`.
- **`zenav1-aom-dsp-bench`** — a seventh workspace member (`publish = false`):
  port-only DSP kernel benchmarks that time the port's own dispatch entry points
  with **no** libaom C-oracle dependency, so they run on every target including
  ones where the oracle cannot be built. (4b92e2b; changelog entry added
  2026-08-03 — the crate had landed with no record here, which is why
  `docs/ARCHITECTURE.md` still said "Six packages".)
- **`CONTEXT-HANDOFF.md` rewritten as the current project handoff** (fresh-box
  setup incl. the mirror-backed submodule and mosaic-vector regeneration, the
  four gates' verified state, live tracks, open pinned cells, jj/marker
  conventions). Removed the consumed `HANDOFF-SCREEN.md` / `HANDOFF-TXSIMD.md`
  (their content lives in CLAUDE.md KB-15/KB-P29 and the landed
  `transform/simd` + STATUS entries; `HANDOFF-TOGGLES.md` stays — it holds the
  live localization notes for the one open toggle cell).

- **bd8 decode Phase C: i16-lane inverse-transform column pass** — the u8
  column pass runs idct4/8/16/32/64 on `i16x16` lanes (16 columns per AVX2
  vector; two-domain design keeps the unclamped butterfly transients in exact
  i32 pairs so it is byte-identical to the scalar port, NOT the libaom lowbd
  saturate-early shape). iadst/identity columns stay i32 (audited not
  i16-safe: `xtask/audit_i16_safety.py`). Measured: DCT columns −57% Ir,
  whole column pass −31.5%, 4K decode −1.3%/−2.6% Ir; see
  `benchmarks/bd8_i16_transform_2026-07-22.md`. (1d29acaf)

- **`zenav1-aom-decode` production-hardening surface** (deliberate API additions
  for the untrusted-input / zenavif decode path):
  - `DecodeConfig` / `DecodeLimits` threaded through `decode_frame_obus_with` /
    `decode_frames_with` / `_prefilter_with` — bounded resource limits for
    untrusted bitstreams. (e25c556)
  - Cooperative cancellation via `enough::Stop`, polled per SB-row / tile /
    frame → `DecodeError::Cancelled`. (e6c7795)
  - Optional `whereat` feature (default OFF) adding `*_at` source-located error
    entries. (edaf579)
  - `AllocMode` fallible-alloc pre-flight (`try_reserve` probe → `AllocFailed`)
    + `max_memory_bytes` enforcement — a byte-preserving allocation ceiling
    against attacker-controlled dimensions. (70b50c6)
  - Malformed-input hardening: frame-dimension DoS ceiling (reject >2^28 px
    before recon alloc) + panic→`Err` conversions found by a structured-random
    fuzz sweep + a stable-toolchain fuzz regression harness. (1b65d61, 88b4de3,
    606813d, 5922c47, bbd7bc4)
  Decode output is byte-identical on valid input (the error type is a rename;
  limits / stop / whereat / alloc all default to unchanged behavior).

### Changed

- **Decoder bd8 lowbd Phase B: the u8 kernels are LIVE** — bd8 frames now
  decode through `predict_intra_u8`, `reconstruct_txb_u8_into`,
  `av1_iwht4x4_add_u8`, u8 intrabc/palette stores (43b7d60), and the salvaged
  `loop_filter_frame_u8` deblock walk (3ca1495, 1ae33ee). CDEF stays on the
  byte-identical widen/narrow delegation by measurement (direct-u8 is +6.61%
  Ir worse); LR/superres/inter-MC/CfL keep delegation (no u8 kernels).
  Output bit-identical at every bit depth (full decode suite, default +
  `AOM_FORCE_SCALAR=1`).

- **Decoder bd8 recon planes are stored as `u8` (`ReconPlane::LowBd`), Phase A
  of the lowbd pipeline** — every kernel still runs the unchanged highbd path
  via byte-identical widen/narrow delegation (no u8 kernel wired yet), so
  decoded output is bit-identical at every bit depth (full decode suite green
  in default + `AOM_FORCE_SCALAR=1`); bd10/12 keep `u16` planes untouched.
  Phase B swaps the delegation arms for the landed `*_u8` kernels. (5336e65)

- **Consolidated the 13 DSP/entropy kernel crates into one `zenav1-aom-dsp`**
  (transform, quant, txb, cdef, restore, intra, loopfilter, dist, inter,
  convolve, recon, dispatch, entropy) — each is now a module, e.g.
  `aom_dsp::transform`, `aom_dsp::entropy`. Shrinks the release surface from 12
  publishable sub-crates to one. Byte-exactness unchanged (pure namespacing —
  only module paths moved); the differential gates stay green. (GitHub #2;
  20324ad, cf0541e, a9a995e, be7586b, c63c3f9, c51fdce, e57c31e)
- **Renamed every crate to the `zenav1-aom-*` prefix** (`zenav1-aom-dsp`,
  `zenav1-aom-decode`, `zenav1-aom-encode`, `zenav1-aom-sys-ref`,
  `zenav1-aom-bench`). Short `[lib] name`s (`aom_dsp`, `aom_decode`, …) are
  retained so interior `use aom_dsp::…` does not churn; only package names, dep
  keys, and CI/justfile `-p` args changed. (GitHub #3 Phase 2; 52be170)
- Publish flags corrected: `zenav1-aom-sys-ref` is now `publish = false` (was
  wrongly publish=default); `zenav1-aom-decode` / `zenav1-aom-encode` are now
  publishable (the facade re-exports them). End state: 4 publishable
  (`zenav1-aom`, `-dsp`, `-decode`, `-encode`) + 2 dev-only (`-sys-ref`,
  `-bench`). (52be170)
- Relicensed to `AGPL-3.0-only OR LicenseRef-Imazen-Commercial` — the standard
  Imazen dual license (LICENSE-AGPL3 + LICENSE-COMMERCIAL added). The inherited
  upstream libaom LICENSE (BSD-2-Clause) and PATENTS (AOM Patent License 1.0)
  live in [`upstream-notices/`](upstream-notices/) (and the `upstream/`
  submodule); they continue to cover the upstream work this port derives from.
  We will release this port under MIT or the original upstream license if
  Imazen's 2026 AI + server costs are covered. (527852efc15a)
- CI: added the org-bar platform matrix — `windows-11-arm`, `macos-15-intel`,
  and `i686-unknown-linux-gnu` (via cross) — as pure-Rust portability jobs
  (invariant A: no C toolchain, no cmake/nasm), while the full C-oracle
  differential suite stays on the linux jobs. Also renamed the CI comment's
  stale `crates/aom-dispatch` ref to `aom_dsp::dispatch`. (GitHub #3 Phase 4;
  fb7e8da)

### Added

- **`zenav1-aom` facade crate** re-exporting `dsp` plus feature-gated `decode` /
  `encode` (both default). `default-features = false, features = ["decode"]`
  builds a decode-only stack (the encoder crate is never compiled) for
  size-sensitive / wasm consumers. (GitHub #2; 52be170)
- Rust-consumer docs for the 4-crate `zenav1-aom-*` structure (GitHub #3
  Phase 3): a rewritten Rust-facing README.md (crate map, install snippet,
  honest early-dev status, fresh-box `--recurse-submodules && cargo test` flow,
  `imazen/zenav1-aom` badges; 5bfa09a); `PORTING.md`, the C→Rust auditability
  map pairing each module with its `upstream/` libaom source + differential gate
  (9d8ddce); and minimal per-crate READMEs for the 4 published crates (e8ec2c1).
  (initial README + this changelog: 527852efc15a)
