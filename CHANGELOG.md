# Changelog

## Workspace

### [Unreleased]

### Added

- **Self-contained KEY-frame encode — no C bootstrap anywhere in the path.**
  `aom_encode::key_frame::encode_key_frame(planes, cfg)` returns a complete AV1
  temporal unit (temporal-delimiter OBU + a PORT-AUTHORED sequence-header OBU +
  `OBU_FRAME`). Until now every encoder path in the repo ran real libaom first
  and parsed its headers (`aom-bench`'s `port_encode(bootstrap: &[u8])`) or
  spliced C's sequence-header bytes verbatim (`avif_parity.rs`), and
  `write_sequence_header_obu` had zero call sites in any `crates/*/src`.
  Everything is derived: `seq_level_idx`/`tier` (`set_bitstream_level_tier`,
  new `aom_encode::seq_level`), profile / bit depth / subsampling / the
  reduced-still-picture framing, `base_qindex`, `allow_screen_content_tools`
  (the ported detector now DRIVES the encode instead of being asserted against
  C's header), the tile grid, the loop-filter levels, and the `tx_mode`
  SELECT→LARGEST flip via a new `key_frame::txb_split_count` over the port's own
  winner trees. **96/96 cells byte-identical to real aomenc** across cq 0..63
  (step 5, plus a 1..19 step-2 low-q arm), {mono, 4:2:0, 4:2:2, 4:4:4} × bd
  {8, 10, 12}, 16×16..512×512, 12 crop/partial-SB sizes including 1×1, and 5
  content classes; both the real C decoder and `aom-decode` decode the port's
  own stream to the pixels real aomenc's stream decodes to. All four
  (CDEF, loop-restoration) combinations are covered, **including both on —
  real aomenc's ALLINTRA default**; configurations outside the gated envelope
  (`--cpu-used != 0`, non-ALLINTRA usage, multi-tile) are REFUSED by name via
  `KeyFrameError`. Total: **96/96 cells byte-identical**.
  Gate: `aom-encode/tests/self_contained_key_frame.rs` (6 tests, including a
  measured proof that a stream without the sequence header is rejected by the
  real C decoder, and a mutation proof that both the byte gate and the pixel
  gate can go red).

- **`aom_encode::pack::pack_tile_from_trees_lr`** — the phase-2 pack for a frame
  with CDEF **and** loop restoration on, which is real aomenc's ALLINTRA
  default and which neither predecessor covered (`pack_tile_from_trees` carried
  only the CDEF strength literals, `pack_tile_lr` only the interleaved per-RU
  restoration params). Additive: `pack_tile_from_trees` now delegates to it
  with `lr: None` and is byte-unchanged, and the LR block is the same one
  `pack_tile_lr` writes, at the same point in the walk
  (`write_modes_sb`, bitstream.c:1625-1645). `encode_key_frame` follows C's
  `cdef_restoration_frame` order: deblock → `av1_cdef_search` →
  `av1_cdef_frame` (apply) → `av1_pick_filter_restoration` on the POST-CDEF
  reconstruction. Gated by the 27 post-filter cells of
  `self_contained_key_frame.rs` (9 per combination).

- **`aom_encode::speed_features::lr_search_sf_allintra`** — moved out of
  `aom-bench` so `key_frame::encode_key_frame` can derive the loop-restoration
  search speed features (`aom-encode` cannot depend on `aom-bench`).
  `aom_bench::lr_search_sf_allintra` delegates to it, so every existing caller
  and gate is unchanged.

### Fixed

- **Screen-content detector was handed the CROP instead of the 8-aligned
  `y_width`/`y_height`.** C's
  `estimate_screen_content_antialiasing_aware` reads
  `cpi->unfiltered_source->y_width` / `->y_height`, which for a
  `YV12_BUFFER_CONFIG` are `(dim + 7) & ~7`, not `y_crop_*`. Both readings feed
  the `area` denominator of the two frame decisions and the 16×16 block-loop
  bound, so on a non-8-aligned crop they differ by up to ~3% of area and flip
  borderline frames. Measured on the new bootstrap-free gate: 258×258, 260×260
  and 262×262 (textured 4:2:0 cq 32) coded `allow_screen_content_tools = 1`
  where real aomenc codes 0; with the aligned size all three agree and 258/262
  became byte-identical end to end. Fixed in both callers
  (`aom-encode/src/key_frame.rs`, `aom-bench/src/lib.rs`) and documented on the
  function; the full `aom-bench` suite is unchanged at 197 passed / 0 failed.

- **The C oracle for `aom_{,highbd_}comp_mask_upsampled_pred` reached a
  broken libaom AVX2 tier** — `linux differential (x86-64)` and
  `linux differential (forced-scalar pin)` had been red on
  `inter_pred_enc_diff::highbd_comp_mask_upsampled_pred_matches_c` for ~63
  consecutive CI runs (first red `aa27d8b8`, 2026-08-31, the commit that added
  the test), while both aarch64 legs were green.
  `aom_highbd_comp_mask_pred_avx2`'s `width == 8` path
  (`upstream/aom_dsp/x86/variance_avx2.c:459`) loads the ODD row's mask from
  `mask + 8` — a hardcoded width — while advancing by `mask_stride << 1`, so at
  `mask_stride != width` every odd row blends against the wrong mask bytes
  (exactly the reported rows-0/2/4/6-identical signature). Its SSE2 tier, its
  NEON tier and its own 8-bit twin all use `mask + mask_stride` and agree with
  `_c`; libaom never hits the arm because every production caller and its own
  test pass `mask_stride == width`. The PORT was correct — it transcribes
  `aom_highbd_comp_mask_pred_c` and the forced-scalar leg (which pins only the
  Rust dispatch) failed identically. Fixed by pinning the oracle:
  `shim/reconinter_enc_shim.c` now compiles libaom's own `reconinter_enc.c`
  with `aom_{,highbd_}upsampled_pred` and `aom_{,highbd_}comp_mask_pred`
  rebound to their `_c` tiers (the `shim/cnn_cscalar.c` technique), and
  `shim/comp_pred_shim.c` routes the two comp-mask upsampled entries to those
  copies instead of the archive's RTCD-dispatched ones.

### Changed

- **`zenav1-aom-target` is a workspace member again — the sibling-repo path
  dep is gone.** The Zq target-search crate landed 2026-08-29 with
  `zensim = { path = "../../../zensim/zensim" }`, a path into a SIBLING
  REPOSITORY. Cargo loads every member manifest for any workspace command, so
  as a member it made the whole workspace unresolvable on any checkout lacking
  that sibling — every CI runner — and it was excluded (with a bare
  `[workspace]` table of its own) as a stopgap. Fixed properly: `zensim` is now
  git-pinned to `main` rev `f7051113` (VERIFIED to expose both
  `custom-profiles` and `feature-regime-v2` and all four APIs the census uses;
  the crates.io index confirms **no published zensim through 0.2.7 has either
  feature**, so a registry dep cannot work), and it plus `png` are **optional
  and default-off** behind a new `census` feature that also gates the
  `zq_census` example. The library itself keeps its zero-dependency
  dependency-injection contract, so `cargo check`/`cargo test --workspace`
  compile it and its 4 unit tests without pulling the judge in at all. The
  `exclude` entry, the crate's own `[workspace]` table, and the two
  `crates/aom-target/{target,Cargo.lock}` `.gitignore` lines that existed only
  because it was a separate workspace are all removed. Resolution verified with
  the sibling `zensim` checkout invisible.

- **Third-party lockfile refreshed within the existing requirements**
  (`c8a8dfb`). `Cargo.lock` only — no manifest requirement moved. 16 packages
  advanced: `libc` 0.2.186 → 0.2.189, `thiserror`/`-impl` 2.0.18 → 2.0.20,
  `serde`/`serde_core`/`serde_derive` 1.0.228 → 1.0.229, `serde_json` 1.0.150 →
  1.0.151, `clap` 4.6.2 → 4.6.6 (plus `clap_builder`, `clap_derive` 4.6.1 →
  4.6.4), `bytemuck` 1.25.1 → 1.25.2, `imgref` 1.12.2 → 1.12.3, `either`
  1.16.0 → 1.18.0, `proc-macro2` 1.0.106 → 1.0.107, `quote` 1.0.46 → 1.0.47 and
  `syn` 2.0.118 → 2.0.119. Nothing entered or left the graph. Every zen-family
  package was held byte-identical (`archmage`/`magetypes` 0.9.27, `enough`
  0.4.4, `whereat` 0.1.5, `zensim`, `zenbench`, and the `zenavif-serialize` git
  rev pin). Afterwards `cargo update --dry-run` reports **0 packages behind**,
  so no manifest requirement is blocking anything and no bump is needed.
  Gated on aarch64-apple-darwin with CI's differential command
  (`cargo test --profile test-fast --workspace --no-fail-fast`): **978 passed,
  0 failed, 53 ignored across 281 test binaries**, including the 64
  byte-identical / identical-to-C assertions against the C libaom oracle.
  CI additionally runs the `AOM_FORCE_SCALAR` forced-scalar-pin leg, which was
  not reproduced locally.

### Fixed

- **CI green again after six red runs — three test-side roots, no library change**
  (`c80b40d1`, `cb76cda9`; KB-42). The published crates' encoder is untouched: all
  three roots were in the differential harnesses and the coverage census. (1) 19 of 19
  `PackCfg` construction sites passed the placeholder `search_tx_mode_is_select: false`
  added in `38a92657`, freezing the SEARCH's tx-size cost table at its frame-init
  snapshot and breaking 23 byte/RD gates; now derived from C's `select_tx_mode`.
  (2) `content_family_census`'s IntraBC row was re-pinned (both floors raised) for the
  reachability gain `735a0a6d` legitimately produced. (3) the nonrd-palette decode gate
  bootstrapped with `--tune-content=screen` without declaring it to the port. Run
  `33325340898` is green on all seven legs.

- **Two decoder panics that libaom treats as corrupt-frame REJECTIONS, plus the
  two `debug_assert`s that hid the same failure in release builds
  (`harden/decoder-panic-surface`).** `av1_ss_size_lookup` has no valid chroma
  plane size for some luma shapes; libaom's `decode_mbmi_block`
  (decodeframe.c:393-401) returns `AOM_CODEC_CORRUPT_FRAME` there. The port
  `assert_ne!`d instead, justified in a comment by "the roundtrip never
  produces them" — a warrant about our own ENCODER, which says nothing about
  what a crafted bitstream can reach, on a decoder that ships into zenavif's
  untrusted AVIF path where a panic is a denial of service. `decode_block` and
  the chroma txb loop now `mark_corrupt` and unwind (the deeper one also covers
  the sub-8x8 shapes C's `bsize >= BLOCK_8X8` gate exempts but which still
  index `MAX_TXSIZE_RECT_LOOKUP`: `BLOCK_4X8` at 4:2:2 IS chroma-reference at
  odd `mi_col`). Both `max_uv_txsize`s (aom-decode + the aom-dsp loop filter)
  had a `debug_assert_ne!` that vanished in release, leaving an anonymous
  "index out of bounds" — same bounds check, now a named panic with a
  `# Panics` section. Byte-inert on conformant streams: `decode_partition`
  already turns the only reachable (4:2:2) case away a frame up, and a new test
  pins why — of 88 (bsize, ss) combinations, the 16 `BLOCK_INVALID` entries are
  all in the ss=(1,0)/(0,1) columns. Record:
  `benchmarks/decoder_panic_surface_2026-08-06.{md,meta}`.

- **`corrupt frame header (bit-reader error / out-of-range syntax value)` was
  one message and one category for two different failures.** A short file and a
  corrupt file are not the same thing to a consumer, and `DecodeError` already
  distinguishes `Truncated` from `Malformed`. `ReadBitBuffer` gains
  `mark_syntax_error(field)` beside the overread flag; the three film-grain
  point-count rejections name their field; the frame-header site returns
  `Truncated(..)` for an overread and `Malformed("frame header syntax out of
  range: <field>")` otherwise. Measured live, not asserted: 180 of 182 prefixes
  of a decoding seed report `truncated`, and the committed film-grain POC
  reports `malformed` naming `num_y_points`.

### Changed

- **Panic messages that named no value, no bound and no contract, named**
  (`convolve` "bad filter type", lowbd + highbd `lpf_scalar` "bad width", the
  three bare `unreachable!()` in the intra predictors, two bare `assert!` on
  the Wiener restoration-unit width, six bare precondition asserts in
  `cdef_frame`). Message-only — each arm is reached solely to panic.

### Added

- **The whole reference/GOP + fixed-Q rate-control surface is ported — 71 of
  the 112 C functions in `encode_strategy.c` and `ratectrl.c`, with the other
  41 reasoned out per function and ZERO left unaccounted** (`72f6536`,
  `6752cd5`, `b0fba02`, `c075208`, `a6afcec`, `006df29`, `8d6de8c`). New:
  `aom_encode::{ref_gop, frame_source, ratectrl, ratectrl_rate, ratectrl_init,
  ratectrl_update, ratectrl_pick}` plus
  `crates/aom-sys-ref/shim/{refgop_shim.c, ratectrl_shim.c, rcarchive_shim.c,
  rc_state_params.h}` and seven differential harnesses (56 tests, green on
  aarch64 AND x86_64). Covers: which buffer slots a frame refreshes and which
  buffer each named reference points at (`av1_get_ref_frames`,
  `av1_get_refresh_frame_flags`, `av1_configure_buffer_updates` and their
  statics); the minq lookup tables and the boost-interpolated active-quality
  curves; the rate model's search layer (`av1_estimate_bits_at_q`,
  `av1_rc_regulate_q`, both qindex-by-rate searches); RC initialisation
  (`av1_primary_rc_init`, `av1_rc_init`, `av1_rc_update_framerate`); the
  per-frame state advance (`av1_rc_postencode_update`,
  `av1_rc_update_rate_correction_factors`, `update_buffer_level`); and the
  q-and-bounds dispatcher (`av1_rc_pick_q_and_bounds` and the four statics
  under it). Evidence is tier 1 wherever an exported symbol exists — a second
  shim TU, `rcarchive_shim.c`, deliberately does NOT include `ratectrl.c` so
  those names bind to `libaom.a` — and tier 1c through a verbatim compile of
  `ratectrl.c` for the ~20 file-statics, with the 1c-vs-1 gap MEASURED on the
  functions each file actually tests rather than on a proxy. Four functions
  are tier 4 (C is `static` with no exported caller short of
  `av1_encode_strategy`) and are labelled as such in their module docs.
  Three defects were found by the differentials while building it:
  `p_rc->arf_boost_factor` is `float_t` == **float**, not double, so
  `get_active_best_quality`'s boost multiply is single precision;
  `SEQ_LEVELS` is **28**, not 24 (`SEQ_LEVEL_2_0`..`SEQ_LEVEL_8_3`), which
  moved `av1_primary_rc_init`'s `avg_frame_qindex`; and
  `rc_pick_q_and_bounds` reads `gf_group->frame_type[gf_index]`, NOT
  `cm->current_frame.frame_type`, for `frame_type_qdelta`. A fourth finding is
  recorded rather than fixed: `rc_pick_q_and_bounds_no_stats`'s
  `delta_rate[FIXED_GF_INTERVAL]` is sixteen long with only EIGHT
  initialisers, so half of all leaf frames take a rate factor of 0.0.

- **`av1/encoder/rdopt.c`'s decision layer — the inter RD brain — is ported:
  69 of its 105 functions, in eight modules, all gated against libaom**
  (`9163133`, `fc27f88`, `1f71285`, `9e55561`, `e184722`, `2003403`,
  `4f86e6a`). New:
  `aom_encode::rdopt_{mv,skip,model,single_state,obmc,var_rd,gate,sse}` plus
  `crates/aom-sys-ref/shim/rdopt_shim.c` and eight differential harnesses
  (66 tests, green on aarch64 AND x86_64). Covers the ref-MV/DRL layer, the
  mode/reference skip mask and its master gate, the inter-mode RD model, the
  single-reference state table and compound-skip gate, the OBMC target
  (`calc_target_weighted_pred` + both visitors), the variance-based RD
  adjustment, and the NEWMV compound assembly. **None of it is wired into the
  encoder yet** — it is the decision layer the top-level driver will call.
  The 27 functions NOT ported are named individually with a reason in
  `docs/RDOPT_C_COVERAGE_2026-09-01.md`; do not read a re-run of
  `tools/c_surface_inventory.py` as coverage, because every module cites its C
  function by name in a doc comment and the tool matches names.

- **A new oracle technique for file-`static` C: "tier 1c"** (`9163133`).
  `nm -g upstream/build/libaom.a` reports TEN exported symbols for the whole of
  rdopt.c, so none of its decision helpers has an address a differential could
  take. `shim/rdopt_shim.c` therefore compiles libaom's OWN rdopt.c into the
  shim archive — its ten exports renamed out of the way, built with libaom's
  Release flags — and exposes flat wrappers around the statics. The bodies
  under test are libaom's source, not a transcription of it; same technique and
  justification as the pre-existing `shim/cnn_cscalar.c`. The one gap versus
  tier 1 (that it is a SECOND COMPILATION) is closed by measurement:
  `rdopt_mv_diff::rdopt_shim_tu_agrees_with_archive` drives the shim TU's
  `av1_block_error_c` and `av1_get_horver_correlation_full_c` against the
  ARCHIVE's exported symbols and asserts bit equality, so the tier claim fails
  loudly if it ever stops holding.

- **The stable decoder fuzz sweep was green for the wrong reason, and now
  measures its own reach.** Its mutation ops (bit flips, truncation,
  length-field corruption, splices, insert/delete) leave a mostly-valid
  arithmetic stream, so the symbol decoder stays near the states a real encoder
  produced. Two ops added: a HOSTILE TILE PAYLOAD (keep the headers, replace
  the tail with PRNG bytes, so `OdEcDec` walks arbitrary symbol sequences) and
  PAYLOAD EXTENSION (append PRNG bytes so the range decoder keeps reading past
  the real tile end). `probe()` now classifies every input as decoded /
  deep-err / shallow-err, prints the histogram, and FAILS below
  `MIN_DEEP_REACH_PPM` — a no-panic result over inputs the OBU parser rejected
  is not evidence about the decoder. Measured: 47.8 % deep reach, 10.0 % fully
  decoded (50 208 decoded / 188 550 deep / 261 242 shallow of 500 000), floor
  pinned 5x under that. **0 panics** over the runs recorded in the `.md`.

- **Four hostile-input contract tests** (`tests/fuzz_regression.rs`), all
  anti-vacuous: no committed POC may reach `DecodeError::Internal` (that
  variant means decoder bug, so an attacker-reachable one is a defect — 11 POCs
  checked); the 4:2:2 POC must be a typed `malformed` naming the chroma
  condition; the `BLOCK_INVALID` table shape the new guards' reasoning rests on;
  and `max_uv_txsize`'s named panic.

- **`aom-dsp/tests/inv_txfm_decodable_pairs.rs` — the inverse transform's
  `assert!(cfg.valid, ..)` is proven unreachable from a bitstream, exhaustively
  rather than by argument.** `TXFM_TYPE_LS` has holes (only DCT is defined at
  64 points) and the decoder feeds both `tx_type` and `tx_size` straight from
  the stream, so "can a crafted file reach that assert?" is a real question
  about the untrusted surface. The test enumerates every `(tx_size, is_inter,
  reduced)` state the decoder can be in, takes every `tx_type` that state's
  ext-tx set marks decodable, and requires a kernel: **314 decodable selections,
  all with kernels; 111 of the 304 `(tx_size, tx_type)` pairs have no kernel and
  none is selectable.** The 111 make the constraint bite, so this cannot pass
  vacuously — and a table edit that opens a hole now fails here instead of as a
  decoder panic on a crafted file.

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
