# CONFIG AXIS INVENTORY — 2026-07-30

**What this is.** An inventory of every configuration axis that can change what this
codebase computes, derived from **source**, not from `PARITY.md` / `HANDOFF-TOGGLES.md`.
It exists as the independent control on a specific failure mode: the encoder-side and
decoder-side permutation gates are both being built from the same two prose documents,
so an axis those documents omit is omitted by both gates *identically* and the gap is
invisible.

**Read at commit** `854b2ac` (`origin/main`). Sibling docs read for
overlap-avoidance only, at these blob shas: `PARITY.md` = `cdc9ed414e5ae1de89517551fc129449c839bbaf`,
`HANDOFF-TOGGLES.md` = `c225daec176573b3f4e862a7aa4febc52da4c355`. The two sibling agents'
own docs (`docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md`,
`docs/DECODER_CONFIG_COVERAGE_2026-07-30.md`) did not exist yet when this was written.

**This document changes no code.** It is a map. Nothing here is a fix.

---

## 0. HEADLINE — read this part even if you read nothing else

### 0.1 The single most important structural fact: the harness cannot express most combinations

The encoder is not driven through one config object. It is driven through **15 separate
C-shim encode entry points** (`grep -n 'pub fn ref_encode_av1_kf' crates/aom-sys-ref/src/lib.rs`),
each with a *fixed* parameter list, plus **three mutually-exclusive knob structs**:

| Surface | Where | Reaches |
|---|---|---|
| `ToggleKnobs` (29 fields) | `crates/aom-bench/src/lib.rs:305` | 24 ctrl ids via `c_encode_ctrls` (of the 25 in `PROBE_TABLE`) |
| `RefTuneKnobs` (11 fields) / `PortTune` (8 fields) | `crates/aom-sys-ref/src/lib.rs:13743` / `crates/aom-encode/tests/encoder_gate_tune_iq_e2e.rs:113` | the tune=IQ/SSIMULACRA2 bundle only |
| ad-hoc shim params | `crates/aom-sys-ref/src/lib.rs:8847, 8906, 9074, 9133, 9188, 9250, 11780, 11856, 11930, 12018, 12106, 12680, 12776, 12859, 13796` | superres / film-grain / lossless / QM / sb128 / tiles / screen-content / min-max-q / defaults |

The generic toggle path is bounded by `PROBE_TABLE` — declared as
`[(i32, i32); 25]` at `crates/aom-sys-ref/src/lib.rs:9031-9057`, mirroring the C probe at
`crates/aom-sys-ref/shim/dec_shim.c:799-824` (25 `case` arms, counted). Of those 25,
`AV1E_SET_DV_COST_UPD_FREQ` (index 24) is never emitted by `c_ctrls`, so **24 controls are
actually reachable** — against **155** encoder controls in `enum aome_enc_control_id`
(`upstream/aom/aomcx.h`, counted: 155 numbered entries) and **41** decoder controls in
`enum aom_dec_control_id` (`upstream/aom/aomdx.h`). `AV1E_SET_SUPERBLOCK_SIZE` is deliberately
outside `PROBE_TABLE` and reachable only through the dedicated sb128 shim
(`crates/aom-sys-ref/src/lib.rs:9022-9025`).

**Consequence:** cross-family permutations are structurally unreachable. There is no way
today to encode `--tune=iq --enable-rect-partitions=0`, or `--superres-denominator=12
--enable-cdef=1`, or `--enable-qm=1 --min-partition-size=16`, in *any* harness. (`--sb-size=128`
× `--tile-columns` IS reachable — `ref_encode_av1_kf_tiles` takes both,
`crates/aom-sys-ref/src/lib.rs:9250-9270` — which is exactly the point: composability is
whatever a shim signature happened to bundle, not a property of the design.) A permutation
gate built on `ToggleKnobs` alone will sweep one 29-dimensional box and leave every other
box untouched — and will look complete while doing so.

### 0.2 The aom-bench encode harness has five hard envelope asserts

`EncodeCell::port_encode_full` (`crates/aom-bench/src/lib.rs:958`) *panics* outside its
envelope. Anyone extending the aom-bench matrix must know these — they are the walls of
the box:

| Assert | Line | Excludes |
|---|---|---|
| `frame_obu_type == OBU_FRAME` | `crates/aom-bench/src/lib.rs:980` | split `OBU_FRAME_HEADER` + `OBU_TILE_GROUP` framing |
| `!p.prefix.show_existing_frame` | `crates/aom-bench/src/lib.rs:1060` | show-existing-frame |
| `p.prefix.frame_type == 0` | `crates/aom-bench/src/lib.rs:1061` | INTER (1), INTRA_ONLY (2), SWITCH (3) |
| `p.quant.base_qindex > 0` | `crates/aom-bench/src/lib.rs:1081` | **all lossless / cq0 cells** |
| `tiles_log2 == 0` | `crates/aom-bench/src/lib.rs:1086` | **all multi-tile cells** |

Lossless and multi-tile *are* gated — but by their own private harnesses
(`encoder_gate_lossless_cq0_e2e_kb5_repro` in
`crates/aom-encode/tests/encoder_gate_chroma_ss_e2e.rs`;
`encoder_gate_multitile_e2e` in `crates/aom-encode/tests/encoder_gate_multitile.rs`).
Neither is reachable from `ToggleKnobs`, so neither composes with any toggle.

### 0.3 Controls the port SILENTLY IGNORES

"Silently ignores" = the harness or the shim accepts the control and forwards it to real
libaom (so the C reference *does* change), while nothing on the port side consumes it —
or the port's own config field exists and is never read. These are correctness traps: the
encode differs from what was asked for, with no error.

| # | Control / field | Driven from | Port-side consumer | Evidence |
|---|---|---|---|---|
| **S1** | `AV1E_SET_COEFF_COST_UPD_FREQ` (126) | `crates/aom-bench/src/lib.rs:620-625` | **NONE** | `grep -rn --include='*.rs' '\bcoeff_cost_upd_freq\b' crates/` → 5 hits, all in `aom-bench/src/lib.rs` (decl `:401`, default `:467`, emission `:620-625`). Zero reads in `aom-encode`. |
| **S2** | `AV1E_SET_MODE_COST_UPD_FREQ` (127) | `crates/aom-bench/src/lib.rs:626-628` | **NONE** | same grep for `mode_cost_upd_freq` → 4 hits, all decl/default/emission. |
| **S3** | `AV1E_SET_ENABLE_CDEF` **value 3** (`CDEF_CONTROL_ADAPTIVE`, "enable adaptively based on frame qindex", `upstream/aom/aomcx.h:681`) | `crates/aom-sys-ref/shim/dec_shim.c:3406` forwards the raw i32; `RefTuneKnobs::enable_cdef` is `i32` (`crates/aom-sys-ref/src/lib.rs:13768`) | **NONE for 3** | `crates/aom-encode/src/pickcdef.rs:33-38`: "Every `apply_adaptive_cdef` arm (`cdef_control == CDEF_ADAPTIVE` only …)" is listed under *What is intentionally NOT ported*. Every other encode shim takes `enable_cdef` as a **`bool`** (`crates/aom-sys-ref/src/lib.rs:8859`), so 3 is only reachable through the tune shim. (Value 2 = "disable for non-reference frames", `aomcx.h:680` — **inert on a lone KEY frame**, which is always a reference frame; listed for completeness, not as a trap.) |
| **S4** | `AV1E_SET_AQ_MODE` (`aq_mode` shim param) + `two_pass` | `crates/aom-sys-ref/src/lib.rs:8863-8864` → `crates/aom-sys-ref/shim/dec_shim.c:402` | **NONE — the encoder has no segmentation at all** | `crates/aom-encode/src/pack.rs:256-257` hardcodes `segid_preskip: false, seg_enabled: false`; `grep -rn 'seg_enabled\|segid_preskip\|write_segment' crates/aom-encode/src crates/aom-encode/tests crates/aom-bench` returns **only those two lines**. The decoder models segmentation fully (`crates/aom-decode/src/lib.rs:1955`, `crates/aom-decode/src/frame.rs:2054, 2140`). |
| **S5** | `--usage` (`AOM_USAGE_GOOD_QUALITY`=0 / `REALTIME`=1 / `ALLINTRA`=2) | `crates/aom-bench/src/lib.rs:284`, forwarded to C at `:785, :810, :834, :857, :881` | **collapsed to one bit** | `crates/aom-bench/src/lib.rs:1087` — `let allintra = self.usage == 2;` is the *only* port-side read. 0 and 1 are indistinguishable to the port. |
| **S6** | `--cpu-used` under `--usage != allintra` | `crates/aom-bench/src/lib.rs:1356-1357` | **wrong cascade** | `SpeedFeatures::set_allintra(speed, …)` is called unconditionally at `crates/aom-bench/src/lib.rs:1357` regardless of `usage`; **there is no `set_good_speed_features_*` port** (`grep -rn 'set_good' crates/` finds only doc-comment references and `lr_search_sf_good` at `crates/aom-bench/src/lib.rs:2668`, which covers the `lpf_sf` restoration slice *only*). The `cfg.allintra &&` guards in `crates/aom-encode/src/partition_pick.rs` (lines 779, 823, 835, 972, 1045, 1075, 1084, 1266, 1283, 1297, 1306, 1638, 2586, 2626, 2659, 2781, 2884, 3311, 3397, 3502, 3623) then suppress the allintra deltas for `usage != 2` — so a GOOD-usage encode at `--cpu-used >= 1` receives **neither** C's GOOD deltas **nor** the allintra ones. |
| **S7** | `--enable-hdr-deltaq` | not driven by any harness; the C default is off | **unported** | `crates/aom-dsp/src/quant/build_quantizer.rs:333-336` — `av1_set_quantizer` is documented as "minus the HDR-deltaq arm (`enable_hdr_deltaq` — 10-bit BT.2020-only, out of the stills envelope)". Not currently a trap (nothing drives it), but it *becomes* one the moment a 10-bit BT.2020 cell is added. |
| **S8** | `AOM_FORCE_SCALAR` on **aarch64 in non-test builds** | env, `crates/aom-dsp/src/dispatch/mod.rs:87` | **silently a no-op for NEON** | `crates/aom-dsp/src/dispatch/mod.rs:60-71` — the pin only reaches NEON when `archmage/testable_dispatch` is on, i.e. test builds only; making it total in production "is NOT implemented". Benchmarks and shipped code run in the un-pinnable configuration. |
| **S9** | `AV1E_SET_ENABLE_ADAPTIVE_SHARPNESS` (172) at **speed ≥ 6** | `crates/aom-sys-ref/shim/dec_shim.c:3381-3382` | **half-modelled** | the qindex cap is implemented in `crates/aom-encode/src/lf_search.rs:405-414`, but `pick_filter_level_from_q` — the speed≥6 closed-form LF entry — takes `(base_qindex, bit_depth, allintra, sharpness_cfg)` and **not** the adaptive flag (`crates/aom-encode/src/lf_search.rs:490-495`; the comment at `:478` says the flag is "default-off and out of this envelope"). Whether libaom's adaptive sharpness also feeds the trellis rdmult path (`crates/aom-encode/src/tx_search.rs:855-856` → `:785` uses the raw CLI value) is **NOT ESTABLISHED**. |

S6 is the one with the largest blast radius. `--usage=good` is **aomenc's default**; `--allintra`
is the non-default flag. The port's byte-exactness claims are all in the non-default mode.
A pinned, self-promoting test for the speed-0 case already exists
(`good_usage_key_frame0_pinned_divergent`, `crates/aom-bench/tests/inter_e2e_search.rs:180-191`,
with a 12-line explanation at `:168-179`) — but it lives in an *inter* test file and appears
**nowhere in `PARITY.md` §A/§B/§C or `HANDOFF-TOGGLES.md`** (grepped both for `usage` / `GOOD` /
`set_good`: `PARITY.md:6, 140, 145-150` only, all inside the C2 loop-restoration narrative).
That is exactly the "invisible to both permutation agents" case this audit was created to catch.

### 0.35 The port never authors a sequence header — every seq-level axis is read-only

`write_sequence_header_obu` (`crates/aom-dsp/src/entropy/header.rs:1046`) has **zero call sites in
any `crates/*/src`** — only four test sites and the C reference. `SequenceHeaderObu`
(`:1012`) has **no `Default`**, so it cannot be constructed piecemeal. Every encoder path parses
the seq header out of a real aomenc bootstrap stream and emits only an `OBU_FRAME`.

So bit depth, monochrome, subsampling, profile, SB size and every seq-level `enable_*` tool bit
are **modelled on decode and replayed on encode, never derived**. Anything that reads "the seq
bit is asserted equal to the knob" (e.g. `--enable-filter-intra` /
`--enable-intra-edge-filter`, `crates/aom-bench/src/lib.rs:1093-1102`) is an agreement check
against libaom's bits, not evidence the port can produce them. Full detail: §3.6a.

### 0.4 Axes that exist only as compile-time or env switches

These never appear in any runtime config matrix, so no permutation gate will ever reach them.
Full detail in §4.

1. `zenav1-aom-dsp/avx512` **off** — `crates/aom-dsp/Cargo.toml:23-24`. Default-on; the off-state is
   built by no CI leg (`--no-default-features` appears only at `.github/workflows/ci.yml:243, 264`,
   both scoped to `-p zenav1-aom`, which cannot reach `aom-dsp`'s own default).
2. The `wasm128` SIMD tier — 31 declaration sites (`crates/aom-dsp/src/cdef/simd.rs`,
   `intra/simd.rs`, `loopfilter/simd.rs`, `quant/simd.rs`, `restore/wiener.rs`, `txb/{mod,simd}.rs`,
   `dist/{mod,simd_variance}.rs`). No wasm32 target is built or run anywhere.
3. The `#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]` scalar transform path —
   `crates/aom-dsp/src/transform/mod.rs:12`, `inv_txfm2d.rs:281, 322, 426, 465`,
   `txfm2d.rs:228, 264`. This is the i686/wasm32 path; i686 is *built* (`ci.yml:262`) but its only
   test step (`ci.yml:266`) runs `-p zenav1-aom`, a 28-line crate with no tests.
4. `archmage/testable_dispatch` **off** — the state every non-test and every bench build is in
   (`crates/aom-dsp/Cargo.toml:37`). See S8.
5. `FUZZ_SMOKE_SEED` (`crates/aom-decode/tests/fuzz_sweep.rs:243`) and `FUZZ_SMOKE_ITERS` (`:239`) —
   never overridden by CI, so the decoder's untrusted-input sweep explores one fixed trajectory
   forever.
6. `overflow-checks`/`debug-assertions` **off** — the `profiling` profile (root `Cargo.toml:21-23`)
   and any release/bench build. ~100+ `debug_assert!`s go dark and integer overflow wraps.
7. The shim C compiler — first of `clang`/`cc`/`gcc` found (`crates/aom-sys-ref/build.rs:185-186`).
   The *oracle's own arithmetic definition* depends on an unpinned host probe.

---

## 1. PORT CONFIG TYPES — the encoder

### 1.1 `EncodeCell` — `crates/aom-bench/src/lib.rs:277`

| Field | Line | Domain | Modelled | Notes |
|---|---|---|---|---|
| `w`, `h` | `:279-280` | any; partial-SB (non-multiple-of-64) supported | yes | partial-SB is the KB-6 sub-gap, closed |
| `mono` | `:281` | bool | yes | |
| `ss_x`, `ss_y` | `:282-283` | (0,0)=4:4:4, (1,0)=4:2:2, (1,1)=4:2:0 | yes | (0,1) 4:4:0 is not an AV1 format |
| `usage` | `:284` | 0 GOOD / 1 REALTIME / 2 ALLINTRA | **partially — see S5/S6** | port reads it once, at `:1087` |
| `cq_level` | `:285` | 0..63 | yes for ≥1; **0 rejected** at `:1081` | |
| `speed` | `:287` | 0..9 (`--cpu-used`) | yes for `usage==2` only | Gate 2 is 0..9 per `PARITY.md:47-59` |
| `bd` | `:288` | 8 / 10 / 12 | yes | |

### 1.2 `ToggleKnobs` — `crates/aom-bench/src/lib.rs:305` (29 fields)

`Default` at `:441-476` reproduces the stock aomenc envelope. `c_ctrls()` (`:508-630`) emits
only non-default knobs. Reference-count of port-side uses (`grep -rn --include='*.rs' '\b<field>\b' crates/`,
excluding the declaration block):

| Field | Line | Domain | C ctrl emitted | Port refs | Verdict |
|---|---|---|---|---|---|
| `enable_rect_partitions` | `:308` | bool | `:512-517` | 35 | MODELLED |
| `enable_ab_partitions` | `:310` | bool | `:518-523` | 37 | MODELLED |
| `enable_1to4_partitions` | `:312` | bool | `:524-529` | 34 | MODELLED |
| `min_partition_size_px` | `:314` | {4,8,16,32,64,128} | `:530-535` | 6 | MODELLED; tested values 4/8/16 only (`crates/aom-bench/tests/toggles_rd_close.rs:153, 181`) |
| `max_partition_size_px` | `:316` | {4,8,16,32,64,128} | `:536-541` | 8 | MODELLED; tested 32/64/128 (`toggles_rd_close.rs:166, 182`, `sb128_e2e.rs:86`) |
| `enable_intra_edge_filter` | `:321` | bool (SEQ header bit) | `:542-547` | 44 | MODELLED; seq bit asserted `:1098` |
| `enable_filter_intra` | `:325` | bool (SEQ header bit) | `:548-553` | 164 | MODELLED; seq bit asserted `:1093` |
| `enable_smooth_intra` | `:328` | bool | `:554-559` | 22 | MODELLED |
| `enable_paeth_intra` | `:330` | bool | `:560-562` | 22 | MODELLED |
| `enable_cfl_intra` | `:332` | bool | `:563-565` | 8 | MODELLED |
| `enable_directional_intra` | `:335` | bool | `:566-571` | 22 | MODELLED |
| `enable_diagonal_intra` | `:337` | bool | `:572-577` | 22 | MODELLED |
| `enable_angle_delta` | `:340` | bool | `:578-580` | 22 | MODELLED |
| `enable_tx64` | `:343` | bool | `:581-583` | 49 | MODELLED |
| `enable_rect_tx` | `:345` | bool | `:584-586` | 48 | MODELLED |
| `enable_flip_idtx` | `:348` | bool | `:587-589` | 52 | MODELLED |
| `use_intra_dct_only` | `:351` | bool | `:590-592` | 73 | MODELLED, **pinned-open** (`PARITY.md:98`) |
| `use_intra_default_tx_only` | `:355` | bool | `:593-598` | 10 | MODELLED |
| `reduced_tx_type_set` | `:360` | bool (FRAME header bit) | `:599-604` | 5 | MODELLED **via bootstrap** — the port reads the parsed header bit and merely asserts it equals the knob (`:1107-1111`) |
| `enable_tx_size_search` | `:366` | bool | `:605-610` | 13 | MODELLED; one-directional assert `:1118` |
| `cdf_update_mode` | `:372` | 0/1/2 | `:611-613` | 7 | MODELLED for 0 and 1; **mode 2 (selective) not swept** (`:369-371` says identical on a lone KEY frame) |
| `enable_palette` | `:377` | bool | not emitted (port-side) | 31 | MODELLED |
| `disable_trellis_quant` | `:385` | 0/1/2/3 | `:614-619` | 9 | MODELLED for 1/2/3; **0 (FULL) verified vacuous** (`HANDOFF-TOGGLES.md:29`) |
| `coeff_cost_upd_freq` | `:401` | 0/1/2/3 | `:620-625` | 5 (decl only) | **SILENTLY IGNORED — S1** |
| `mode_cost_upd_freq` | `:403` | 0/1/2/3 | `:626-628` | 4 (decl only) | **SILENTLY IGNORED — S2** |
| `deltaq_mode3` | `:410` | bool | not emitted | 7 | MODELLED |
| `deltaq_mode2` | `:416` | bool | not emitted | 9 | MODELLED |
| `disable_tx_stats_prune` | `:424` | bool | not emitted | 6 | anti-vacuity witness only |
| `delta_lf_mode` | `:430` | bool | not emitted | 5 | MODELLED |
| `enable_intrabc` | `:438` | bool | not emitted (C via `c_encode_screen`) | 18 | MODELLED |

**Axes that `ToggleKnobs` structurally cannot express** (each has its own separate harness, so
none of them composes with any of the 29 above): `--tune`, `--sharpness`, `--dist-metric`,
`--enable-chroma-deltaq`, `--enable-adaptive-sharpness`, `--deltaq-mode=6`, `--deltaq-strength`,
`--enable-qm`/`--qm-min`/`--qm-max`, `--enable-cdef`, `--enable-restoration`,
`--superres-mode`/`--superres-denominator`, `--film-grain-table`, `--sb-size=128`,
`--tile-columns`/`--tile-rows`, `--min-q`/`--max-q`, `--lossless`, `--aq-mode`, `--usage`.

### 1.3 `TuneKnobs` — `crates/aom-encode/src/lib.rs:115`

| Field | Line | Domain | Modelled | Tested |
|---|---|---|---|---|
| `use_qm_dist_metric` | `:123` | bool (`--dist-metric=qm-psnr`) | yes (`crates/aom-encode/src/tx_search.rs:999-1001, 1294, 1328, 1474, 1492`) | `crates/aom-encode/tests/encoder_gate_tune_iq_e2e.rs:522-524` |
| `iq_tuning` | `:127` | bool (`tune=IQ`/`SSIMULACRA2`) | yes (`tx_search.rs:785-788, 813-824, 1340`) | same |

**`TuneKnobs` is not reachable from the aom-bench harness at all.** `port_encode_full` hardcodes
`TuneMetric::Psnr` (`crates/aom-bench/src/lib.rs:1156`) and every other e2e harness does the same
(`grep -rn 'TuneMetric' crates/` → `TuneMetric::Psnr` at
`encoder_gate_chroma_ss_e2e.rs:503`, `kb5_lossless_localize.rs:375`, `kb4_bd10_rd_localize.rs:354`,
`decode_diff_noise_case.rs:393`, `encoder_gate_bd10_diff.rs:302`, `kb11_speed7_noise_localize.rs:372`,
`avif_parity.rs:329`, `kb4_txb_tie_probe.rs:46`, `decode_diff_multisb.rs:347`, `kb7_rd_localize.rs:383`,
`encoder_gate_e2e_byte_match.rs:555`, `decode_diff_ab_probe.rs:463`, `encoder_gate_multitile.rs:321`,
`kb6_real_rd_localize.rs:447`, `encoder_gate_superres_e2e.rs` / `encoder_gate_cdef_e2e.rs` via
`aom-bench`). The only non-PSNR path in the workspace is
`crates/aom-encode/tests/encoder_gate_tune_iq_e2e.rs` and the enumeration differential
`crates/aom-encode/tests/rd_mult_diff.rs:22` (`TUNINGS = [Psnr, Iq, Ssimulacra2]`, which exercises
`av1_compute_rd_mult` alone, not an encode).

### 1.4 `PortTune` / `RefTuneKnobs` — the second knob surface

`RefTuneKnobs` (`crates/aom-sys-ref/src/lib.rs:13743-13769`) is the only place these 11 controls are
reachable; `PortTune` (`crates/aom-encode/tests/encoder_gate_tune_iq_e2e.rs:113-138`) is its
port-side twin. Defaults are `-1` = "do not issue the ctrl" (`:13771-13787`).

| Field | Line | Domain | Port field | Tested |
|---|---|---|---|---|
| `tuning` (`AOME_SET_TUNING`) | `:13747` | `AOM_TUNE_IQ`=10 / `AOM_TUNE_SSIMULACRA2`=11 | `QuantTuning` (`crates/aom-dsp/src/quant/build_quantizer.rs:305-317`) | yes |
| `sharpness` (`AOME_SET_SHARPNESS`) | `:13749` | 0..=7 | `TxTypeSearchPolicy::sharpness` (`crates/aom-encode/src/tx_search.rs:856`), `SbEncodeEnv::sharpness` (`crates/aom-encode/src/encode_sb.rs:399`) | yes, `PARITY.md:77`; **but hardcoded 0 in aom-bench** (`crates/aom-bench/src/lib.rs:1403`, and `sf.tx_type_search_policy(false, 0)` at `:1456`) |
| `enable_adaptive_sharpness` | `:13751` | 0/1 | `lf_search::frame_lf_sharpness` | yes |
| `dist_metric` | `:13754` | PSNR / QM_PSNR | `TuneKnobs::use_qm_dist_metric` | yes |
| `enable_chroma_deltaq` | `:13756` | 0/1 | `av1_set_quantizer(enable_chroma_deltaq, ..)` (`build_quantizer.rs:361`) | yes |
| `deltaq_mode` | `:13758` | 0 / 6 (harness uses only these) | `DeltaQFrameCtx` (`crates/aom-encode/src/encode_sb.rs:443`) | yes for 0/6; modes 2/3 via `ToggleKnobs` instead; **mode 1 OBJECTIVE is TPL-gated and inert** (`PARITY.md:274`) |
| `deltaq_strength` | `:13760` | percent, default 100 | `DeltaQFrameCtx::deltaq_strength` (`encode_sb.rs:454`) | yes |
| `enable_qm` / `qm_min` / `qm_max` | `:13762-13766` | 0/1 + 0..15 | `PickFrameCfg::qm_levels` (`crates/aom-encode/src/partition_pick.rs:581`), `QmCtx` (`crates/aom-encode/src/lib.rs:137`) | yes |
| `enable_cdef` | `:13768` | **0/1/2/3** | 0/1 only | **value 3 = S3**; value 2 is inert on a lone KEY frame |

### 1.5 `PackCfg` — `crates/aom-encode/src/pack.rs:114`

| Field | Line | Domain | Notes |
|---|---|---|---|
| `enable_filter_intra` | `:119` | bool | seq-header bit; must equal `PickFrameCfg::enable_filter_intra` |
| `tx_mode_is_select` | `:122` | bool | `TX_MODE_SELECT` vs `TX_MODE_LARGEST`; `TX_MODE_ONLY_4X4` (0) never emitted |
| `signal_gate` | `:128` | bool | reduces to `base_qindex > 0` **because segmentation is off** |
| `allow_update_cdf` | `:131` | bool | |
| `base_qindex` | `:134` | 0..255 | |
| `delta_q_present` | `:138` | bool | |
| `delta_q_res` | `:141` | 1/2/4/8 | read only when `delta_q_present` |
| `allow_screen_content_tools` | `:149` | bool | |
| `allow_intrabc` | `:153` | bool | |

`kf_block_state` (`:254-299`) **hardcodes** the following, so they are axes the encoder cannot
express: `segid_preskip=false` (`:256`), `seg_enabled=false` (`:257`), `update_map=false` (`:258`),
`seg_pred=0` (`:259`), `seg_cdf_num=0` (`:260`), `last_active_segid=0` (`:261`),
`seg_skip_feature=[false;8]` (`:262`), `dlf_multi=false` (`:277`).
`delta_lf_multi=1` is also listed as an open follow-up at `PARITY.md:279`.

### 1.6 `SpeedFeatures` — `crates/aom-encode/src/speed_features.rs:119` (52 fields)

Single constructor: `set_allintra(speed, allow_screen_content_tools, use_hbd)` (`:456`). Groups
mirrored from C: `part_sf` (`:120-202`), `rt_sf` (`:204-214`), `intra_sf` (`:216-291`),
`tx_sf` (`:293-363`, `:410-417`), `winner_mode_sf` (`:365-409`), `rd_sf` (`:419-429`),
`lpf_sf` (`:431-443`).

**There is no `set_good_speed_features_framesize_independent` port and no
`set_rt_speed_features` port.** The only GOOD-mode derivation anywhere is
`lr_search_sf_good` (`crates/aom-bench/src/lib.rs:2668`), which covers the loop-restoration
`lpf_sf` slice only — its own doc at `:2666-2667` says "Only GOOD speed-0 cells are gated in this
harness; GOOD speed>=1 needs dedicated gate cells". See S6.

Also note `prune_tx_type_using_stats` (`:363`) is framesize-dependent and is set **outside** the
constructor, in `crates/aom-bench/src/lib.rs:1363-1374` — gated on `allintra && w.min(h) >= 480`.
Every gate frame below 480p leaves it 0.

### 1.7 `PickFrameCfg` — `crates/aom-encode/src/partition_pick.rs:506`

Steering fields (non-cost-table): `allintra` (`:527`), `speed` (`:528`), `qindex` (`:529`),
`enable_filter_intra` (`:530`), `enable_tx64` (`:531`), `enable_rect_tx` (`:532`),
`intra_pruning_with_hog` (`:535`), `enable_rect_partitions` (`:538`),
`less_rectangular_check_level` (`:541`), `max_partition_size` (`:544`), `min_partition_size` (`:549`),
`enable_1to4_partitions` (`:557`), `enable_ab_partitions` (`:567`),
`allow_screen_content_tools` (`:576`), `qm_levels` (`:581`), `palette_costs` (`:588`),
`intrabc` (`:595`), `inter` (`:603`), `intra_tools` (`:607`).

`IntraToolCfg` (`:679-690`): `enable_diagonal_intra`, `enable_directional_intra`,
`enable_smooth_intra`, `enable_paeth_intra`, `enable_angle_delta` — all bool, all default true
(`:692-702`).

`TxTypeSearchPolicy` (`crates/aom-encode/src/tx_search.rs:831`) carries 23 further steering fields
including `skip_trellis`, `sharpness`, `use_screen_content_tools`, `prune_tx_type_using_stats`,
`predict_dc_level`, `enable_flip_idtx`, `use_intra_dct_only`, `enable_tx_size_search`,
`use_qm_dist_metric`, `iq_tuning`.

`UvLoopPolicy` (`crates/aom-encode/src/intra_uv_rd.rs:1394`) carries 12 chroma-loop gates.

### 1.8 `SbEncodeEnv` — `crates/aom-encode/src/encode_sb.rs:365`

Config-bearing fields: `sb_size` (`:366`, 64 or 128), tile bounds (`:370-373`),
`monochrome` (`:374`), `ss_x`/`ss_y` (`:375-376`), `bd` (`:377`), `lossless` (`:378`),
`reduced_tx_set_used` (`:379`), `disable_edge_filter` (`:380`), `filter_type` (`:381`),
`rdmult` (`:398`), `sharpness` (`:399`), `enable_optimize_b` (`:400`, `TrellisOptType`),
`use_chroma_trellis_rd_mult` (`:402`), `qm_levels` (`:420`), `tune` (`:424`),
`deltaq` (`:430`), `ref_frame` (`:436`).

`DeltaQFrameCtx` (`:443`): `base_qindex`, `delta_q_res`, `deltaq_strength`, `perceptual_ai`,
`perceptual_wavelet`, `sb_mi`, `delta_lf_present`.

### 1.9 `QuantParams` — `crates/aom-encode/src/lib.rs:156`

Config-bearing: `qm`/`iqm`/`qm_ctx` (`:167-172`), `bd` (`:176`), `lossless` (`:183`),
**`adaptive`** (`:191`) = `oxcf.q_cfg.quant_b_adapt` (`--quant-b-adapt`).

`adaptive` is read **only** by `QuantKind::B`. On the default speed-0 allintra envelope the
trellis is on, so `AV1_XFORM_QUANT_FP` is selected and `quant_b_adapt` is inert — this is
correctly documented at `HANDOFF-TOGGLES.md:23`. It is live only combined with
`--disable-trellis-quant=1/2`, and that combination has **no kernel** (`aom_quantize_b_adaptive`
family unported). `QuantKind::Dc` (`crates/aom-encode/src/lib.rs:107`) is documented at `:99` as
"not modelled yet" for the `AV1_XFORM_QUANT_DC` dispatch.

### 1.10 Enums that steer decisions

| Enum | Line | Variants |
|---|---|---|
| `QuantKind` | `crates/aom-encode/src/lib.rs:101` | `Fp` / `B` / `Dc` |
| `TrellisOptType` | `crates/aom-encode/src/encode_intra.rs:95` | Full / No / FinalPass / NoEstimateYrd |
| `EncMode` | `crates/aom-encode/src/rd.rs:145` | `Good`=0 / `Realtime`=1 / `Allintra`=2 |
| `TuneMetric` | `crates/aom-encode/src/rd.rs:133` | Psnr / Iq / Ssimulacra2 |
| `FrameUpdateType` | `crates/aom-encode/src/rd.rs:112` | (rd-mult row selector) |
| `FrameType` | `crates/aom-encode/src/rd.rs:157` | branches only on `!= KEY` |
| `QuantTuning` | `crates/aom-dsp/src/quant/build_quantizer.rs:305` | Psnr / Iq / Ssimulacra2 |
| `TxMode` | `crates/aom-dsp/src/entropy/partition.rs:611` | ONLY_4X4 / LARGEST / SELECT |
| `SuperresAutoSearchType` | `crates/aom-encode/src/superres_select.rs:233` | (AUTO-mode search strategy) |
| `NoiseShape` / `NoiseStatus` | `crates/aom-encode/src/noise_model.rs:585, 603` | denoise estimation |
| `TxPruneType` | `crates/aom-encode/src/tx_search.rs:2211` | NN tx-depth prune outcome |
| `AllocMode` | `crates/aom-decode/src/config.rs:100` | Fallible (default) / Infallible |
| `LimitKind` | `crates/aom-decode/src/error.rs:23` | Pixels / Width / Height / MemoryBytes / … |

---

## 2. PORT CONFIG TYPES — the decoder

### 2.1 `DecodeConfig` — `crates/aom-decode/src/config.rs:119` (`#[non_exhaustive]`)

| Field | Line | Domain | Modelled | Tested |
|---|---|---|---|---|
| `limits: DecodeLimits` | `:121` | `max_pixels`, `max_width`, `max_height`, `max_memory_bytes` (all `Option`) + the hardcoded pixel ceiling | yes | `crates/aom-decode/src/config.rs:228, 245, 268, 286` (4 unit tests) |
| `stop: Option<&dyn Stop>` | `:125` | none / a cooperative token; polled at SB-row/tile/frame boundaries | yes (`:190-195`) | NOT ESTABLISHED — no test named in `config.rs`'s own `#[cfg(test)]` block covers cancellation; a full test-tree grep for a cancellation gate was not performed |
| `alloc: AllocMode` | `:129` | Fallible / Infallible | yes (`:164-185`) | `crates/aom-decode/src/config.rs:294` |

This is the **entire** public decoder configuration surface. Everything else the decoder branches
on comes from the bitstream (sequence header / frame header), not from the caller — see §3.

### 2.2 Decoder axes explicitly NOT modelled

- **Large-scale tile / `enable_large_scale_tile`** — `crates/aom-decode/src/frame.rs:24`:
  "Large-scale tile mode (`enable_large_scale_tile`) is not modelled". This corresponds to
  `AV1D_SET_TILE_MODE`, `AV1D_EXT_TILE_DEBUG`, `AV1D_SET_TILE_ROW`/`_COL`,
  `AV1D_SET_OUTPUT_ALL_LAYERS` in `upstream/aom/aomdx.h`.
- **Threading** — the port is single-threaded end to end: `grep -rn 'rayon\|std::thread\|spawn(\|num_threads\|row_mt' crates/*/src` returns **zero hits**. So `--threads`, `--row-mt`,
  `AV1D_SET_ROW_MT`, and the tile/frame-parallel decoder controls are non-axes for the port
  (bit-exactness is unaffected in libaom, but any future threading work has no test scaffolding).

---

## 3. BITSTREAM-DERIVED AXES (sequence + frame header)

These are the **decoder's** true configuration axes: the decoder has no knob API beyond
`DecodeConfig` (§2.1), so everything it branches on arrives as a syntax element. On the encoder
side the same fields are the *output* the port must reproduce — and, in the current harness, most
are **bootstrapped from the C reference's parsed header** rather than derived, which is why they
look "covered" without being independently exercised.

> Scope note: a deeper per-element decoder-coverage pass is Agent K's deliverable
> (`docs/DECODER_CONFIG_COVERAGE_2026-07-30.md`). What follows is the axis *list* plus the
> hardcoded-element findings, which are the part that bears on the "is the axis list complete"
> question. A per-element coverage classification of every field below (which are
> **PIXEL**-tested vs **HDR-only** round-trip-tested vs untested) was produced by a sub-audit
> and is summarised at §3.8; the two structural findings it produced (§3.6a, §3.6b) were
> independently re-verified here before being written down.

### 3.1 `SequenceHeaderParams` — `crates/aom-dsp/src/entropy/header.rs:784`

`num_bits_width` / `num_bits_height` (`:785-786`), `max_frame_width` / `max_frame_height`
(`:787-788`), `reduced_still_picture_hdr` (`:789`), `frame_id_numbers_present_flag` (`:790`),
`delta_frame_id_length` (`:791`), `frame_id_length` (`:792`), **`sb_size_128`** (`:793`),
**`enable_filter_intra`** (`:794`), **`enable_intra_edge_filter`** (`:795`),
`enable_interintra_compound` (`:796`), `enable_masked_compound` (`:797`),
`enable_warped_motion` (`:798`), `enable_dual_filter` (`:799`), `enable_order_hint` (`:800`),
`enable_dist_wtd_comp` (`:801`), `enable_ref_frame_mvs` (`:802`),
`force_screen_content_tools` (`:803`, **0/1/2=SELECT**), `force_integer_mv` (`:804`, **0/1/2**),
`order_hint_bits_minus_1` (`:805`), **`enable_superres`** (`:806`), **`enable_cdef`** (`:807`),
**`enable_restoration`** (`:808`).

Bold = drives a coding tool the port models. The seven `enable_*` inter/compound flags are
carried and serialized but inert on KEY frames.

### 3.2 `ColorConfigParams` — `crates/aom-dsp/src/entropy/header.rs:887`

`bit_depth` (`:888`, 8/10/12), `profile` (`:889`, 0/1/2), `monochrome` (`:890`),
`color_primaries` (`:891`), `transfer_characteristics` (`:892`), `matrix_coefficients` (`:893`),
`color_range` (`:894`), `subsampling_x` / `subsampling_y` (`:895-896`),
`chroma_sample_position` (`:897`), `separate_uv_delta_q` (`:898`).

**Coverage, precisely:** the CICP group (`color_primaries`, `transfer_characteristics`,
`matrix_coefficients`, `color_range`, `chroma_sample_position`) IS driven to randomized
non-default values — but only in a **self round-trip** harness
(`crates/aom-dsp/tests/rb_diff.rs:661-702, 1943-1946`; the module doc at `:1-4` states it is a
`WriteBitBuffer` → `ReadBitBuffer` inverse check, **not** a C differential for these fields). No
*encode* ever produces a non-default CICP value, and the encoder never derives them — they are
bootstrapped. `separate_uv_delta_q` is threaded from the bootstrap at
`crates/aom-bench/src/lib.rs:1033`.

### 3.3 Timing / decoder-model group

`TimingInfo` — `num_units_in_display_tick` (`:955`), `time_scale` (`:956`),
`equal_picture_interval` (`:957`), `num_ticks_per_picture` (`:958`).
`DecoderModelInfo` — `encoder_decoder_buffer_delay_length` (`:975`),
`num_units_in_decoding_tick` (`:976`), `buffer_removal_time_length` (`:977`),
`frame_presentation_time_length` (`:978`).
All bootstrapped through `FrameHeaderPrefix` (`crates/aom-bench/src/lib.rs:1000-1018`); never
driven to a non-default value by any encode.

### 3.4 `FrameHeaderPrefix` — `crates/aom-dsp/src/entropy/header.rs:1101`

38 fields (`:1102-1139`). The ones that steer decoding: `show_existing_frame` (`:1103`),
`frame_type` (`:1112`, KEY=0 / INTER=1 / INTRA_ONLY=2 / S=3), `show_frame` (`:1113`),
`showable_frame` (`:1114`), `error_resilient_mode` (`:1115`), `disable_cdf_update` (`:1116`),
`force_screen_content_tools` (`:1117`), `allow_screen_content_tools` (`:1118`),
`force_integer_mv` (`:1119`), `cur_frame_force_integer_mv` (`:1120`),
`superres_upscaled_width` / `_height` (`:1121-1122`), `current_frame_id` (`:1125`),
`order_hint` (`:1127`), `primary_ref_frame` (`:1129`), `buffer_removal_time_present` (`:1130`),
`operating_points_cnt_minus_1` (`:1131`), `operating_point_idc[32]` (`:1133`),
`temporal_layer_id` / `spatial_layer_id` (`:1134-1135`), `refresh_frame_flags` (`:1138`).

The encode harness pins `temporal_layer_id = 0` and `spatial_layer_id = 0`
(`crates/aom-bench/src/lib.rs:1019-1020`) and asserts `frame_type == 0` and
`!show_existing_frame` (`:1060-1061`).

`error_resilient_mode` is pinned `false` on the port's inter path
(`crates/aom-encode/src/inter_frame.rs:173`) and is never driven true by any encode. On the
decode side it is *parsed* and gates two things (`crates/aom-decode/src/frame.rs:538, 545, 722`).

### 3.5 Per-tool header groups

| Group | Struct | Line |
|---|---|---|
| Quantization + delta-q | `DeltaQParams` | `crates/aom-dsp/src/entropy/header.rs:516` |
| Film grain | `FilmGrainParams` | `crates/aom-dsp/src/entropy/header.rs:560` |
| Warped motion | `WarpedMotionParams` | `crates/aom-dsp/src/entropy/header.rs:686` (and `crates/aom-dsp/src/inter/warp.rs:55`) |
| Loop filter | `LfParams` | `crates/aom-dsp/src/loopfilter/frame.rs:222` |
| CDEF | `CdefFrameParams` | `crates/aom-dsp/src/cdef/frame.rs:184` |
| Loop restoration | `LrFrameConfig` | `crates/aom-dsp/src/entropy/lr.rs:361` |
| Superres | — | `crates/aom-decode/src/superres.rs` |
| Quant matrices | — | `crates/aom-decode/src/qm.rs`, `qm_tables.rs` |
| Segmentation | `Segmentation` | via `crates/aom-dsp/src/quant/quant_common.rs:196-202` (`SEG_LVL_MAX`=8, `SEG_LVL_ALT_Q`=0, `SEG_LVL_SKIP`=6) |

**Segmentation asymmetry (this is S4 seen from the bitstream side):** the decoder models it fully
— per-block segment ids, `SEG_LVL_ALT_Q` dequant shifts, `SEG_LVL_SKIP` forced skips,
`SEG_LVL_ALT_LF_*` deblock deltas (`crates/aom-decode/src/lib.rs:95-98, 615-616, 669-695, 1903-1955`;
`crates/aom-decode/src/frame.rs:66-68, 154-158, 340, 2054, 2140`). The **encoder cannot emit any of
it**: `crates/aom-encode/src/pack.rs:256-262`. A `grep -rn 'seg_enabled|segid_preskip|write_segment'`
over `crates/aom-encode/src`, `crates/aom-encode/tests` and `crates/aom-bench` returns **exactly
those two hardcoded `false` lines and nothing else**.

### 3.6 Syntax elements the port HARDCODES (cannot express)

| Element | Hardcoded to | Site |
|---|---|---|
| **the ENTIRE sequence header (all 39 elements of §3.1–§3.3)** | never authored | see §3.6a below — the single largest hardcoded surface in the port |
| `segmentation_enabled` + all 8 `SEG_LVL_*` features | off / 0 | `crates/aom-encode/src/pack.rs:256-262` |
| `delta_lf_multi` | `false` (`DEFAULT_DELTA_LF_MULTI`) | `crates/aom-encode/src/pack.rs:277`; open follow-up at `PARITY.md:279` |
| `num_tg` (tile groups) | `1` | `crates/aom-encode/src/obu_assemble.rs:2, 102, 107-115` |
| `tile_start_and_end_present_flag` | `0` | `crates/aom-encode/src/obu_assemble.rs:150-152` |
| `temporal_layer_id` / `spatial_layer_id` | `0` / `0` | `crates/aom-bench/src/lib.rs:1019-1020` |

### 3.6a THE PORT NEVER AUTHORS A SEQUENCE HEADER — **SUPERSEDED 2026-09-02**

> **Update 2026-09-02.** This section was true when written and is now true only of
> `aom-bench`'s `port_encode*`. `aom_encode::key_frame::derive_sequence_header` /
> `sequence_header_obu` (`crates/aom-encode/src/key_frame.rs`) call
> `write_sequence_header_obu` from `crates/aom-encode/src`, so the port DOES author a
> sequence header, and `encode_key_frame` emits a complete temporal unit (TD +
> sequence-header OBU + `OBU_FRAME`) with no C bytes in the path — **69/69 cells
> byte-identical to real aomenc**, gated by
> `crates/aom-encode/tests/self_contained_key_frame.rs`. The table row above
> ("the ENTIRE sequence header ... never authored") is likewise superseded for that
> entry point. Its envelope is ALLINTRA / `--cpu-used 0` / single tile / SB64 /
> CDEF off / LR off; everything else is refused by name (`KeyFrameError`). The text
> below is retained unedited as the 2026-07-30 record.


`write_sequence_header_obu` (`crates/aom-dsp/src/entropy/header.rs:1046`) has **zero call sites in
any `crates/*/src`**. `grep -rn --include='*.rs' 'write_sequence_header_obu' crates/ xtask/`
returns only: the definition, four **test** sites
(`crates/aom-encode/tests/seq_header_matches_real_encoder.rs:116`,
`crates/aom-encode/tests/frame_header_matches_real_encoder.rs:195`,
`crates/aom-dsp/tests/rb_diff.rs:2079`, `crates/aom-dsp/tests/header_diff.rs:1063`), the C
reference (`crates/aom-sys-ref/src/lib.rs:700, 4283-4303`), and doc comments.
`SequenceHeaderObu` (`crates/aom-dsp/src/entropy/header.rs:1012`) derives only `Clone, Debug` —
**no `Default`** — so it cannot even be partially constructed.

Every encoder path *parses* a sequence header out of a real aomenc bootstrap stream
(`crates/aom-bench/src/lib.rs:970, 2404, 2502`) and emits only an `OBU_FRAME`
(`crates/aom-encode/src/obu_assemble.rs:42`, the sole `OBU_FRAME` writer).

**Consequence:** bit depth, monochrome, subsampling, profile, SB size, and every seq-level
`enable_*` tool bit are **read-only axes** — modelled on decode, replayed on encode, never
derived. A standalone port encoder (one not handed a C reference stream) cannot produce a
sequence header at all. This is the deepest instance of the bootstrap dependency: several axes
that look "covered" in §3.1–§3.3 are covered only in the sense that the port can *parse* what
libaom wrote.

### 3.6b CORRECTION — `context_update_tile_id` / `tile_size_bytes_minus_1` are NOT hardcoded

An earlier revision of this document (commit `20596b4`) listed these as hardcoded syntax
elements, and a third as "the whole multi-tile frame header is not re-serialized". **All three
were wrong.** Recorded here rather than silently deleted.

The literals at `crates/aom-dsp/src/entropy/header.rs:426-427` are **placeholders**.
`write_frame_header_obu` records the position immediately before them via
`wb.mark_saved_position()`, and `assemble_multitile_frame_obu_payload_derived`
(`crates/aom-encode/src/obu_assemble.rs:215`) overwrites both with **derived** values at
`:254-255`:

```
wb.overwrite_literal(saved, largest_tile_id, ctx_bits);
wb.overwrite_literal(saved + ctx_bits as usize, tile_size_bytes - 1, 2);
```

mirroring C's `write_tile_obu_size`. Both are **anti-vacuity asserted** — a non-zero
`context_update_tile_id` and a `tile_size_bytes > 1` must each occur at least once
(`crates/aom-encode/tests/obu_assemble_multitile_diff.rs:341-351`).

What is true: there are **two** multi-tile assemblers, and only one derives.
`assemble_multitile_frame_obu_payload` (`crates/aom-encode/src/obu_assemble.rs:143`) takes the
header as **raw bootstrapped bytes** and is what `encoder_gate_multitile_e2e` uses
(`crates/aom-encode/tests/encoder_gate_multitile.rs:477`); the deriving variant is used by
`obu_assemble_multitile_diff.rs:202` and `encoder_gate_chroma_ss_e2e.rs:762`. So the
**multi-tile byte gate runs on the non-deriving path** while the derivation is proven separately
— worth knowing before treating `encoder_gate_multitile_e2e` as end-to-end proof of the tile
header.

Note the doc-path drift found while checking this: `crates/aom-encode/src/obu_assemble.rs:131-134`
points at `crates/aom-entropy/src/header.rs`, a crate that does not exist in this worktree
(`ls crates/` → `aom-bench aom-decode aom-dsp aom-dsp-bench aom-encode aom-sys-ref zenav1-aom`).
The writer is at `crates/aom-dsp/src/entropy/header.rs:423-429`. Cosmetic, but it will send a
reader to a dead path.

### 3.7 Decoder-side bitstream axes explicitly NOT modelled

`enable_large_scale_tile` / tile-list OBUs (`crates/aom-decode/src/frame.rs:24-26`), and
everything that follows from it (`AV1_SET_TILE_MODE`, `AV1D_SET_EXT_REF_PTR`,
`AV1D_EXT_TILE_DEBUG`, `AV1_SET_DECODE_TILE_ROW`/`_COL`, `AV1D_SET_OUTPUT_ALL_LAYERS`,
`AV1D_SET_OPERATING_POINT` — §5.6).

---

### 3.8 Coverage classification — the "HDR-only" trap

A sub-audit classified all ~39 sequence-header and ~55 frame-header elements by *what kind* of
test drives them to a non-default value. The distinction that matters:

- **PIXEL** — a real decode / encode with a non-default value (e.g. `sb_size_128`,
  `enable_superres`, `enable_cdef`, `enable_restoration`, `bit_depth`, `profile`, `monochrome`,
  `subsampling_*`, `allow_intrabc`, `allow_screen_content_tools`, `disable_cdf_update`,
  `using_qmatrix` + all 16 `qmatrix_level_*`, `delta_q_*`, `delta_lf_*`, `reduced_tx_set_used`,
  `coded_lossless`, `film_grain_params_present`, `show_existing_frame`, `frame_type` 0/1).
- **HDR-only** — randomized *only* in a header write→read→C-oracle round-trip
  (`crates/aom-dsp/tests/header_diff.rs`, `crates/aom-dsp/tests/rb_diff.rs`). The bits are
  proven correct; **nothing proves the decode or encode that consumes them is correct**, because
  no pixel path ever sees a non-default value.
- **ENC-BYTE** — proven only through an encoder byte gate.

The **HDR-only** set is large and is the part most likely to be mistaken for coverage:
`color_primaries`, `transfer_characteristics`, `color_range`, `chroma_sample_position`, all
timing / decoder-model / display-model / per-operating-point fields (`tier`, `op_*`,
`seq_level_idx` apart from a real-stream check), `frame_id_*` + `current_frame_id`,
`buffer_removal_time_*`, `temporal_layer_id` / `spatial_layer_id`, `error_resilient_mode`,
`cur_frame_force_integer_mv`, `refresh_frame_context_disabled`, `reference_mode_select`,
`skip_mode_allowed` / `skip_mode_flag`, `allow_warped_motion`, `separate_uv_delta_q`, and the
seq-level inter flags (`enable_interintra_compound`, `enable_masked_compound`,
`enable_dual_filter`, `enable_dist_wtd_comp`, `enable_ref_frame_mvs`, `enable_warped_motion`,
`enable_order_hint`, `force_screen_content_tools`, `force_integer_mv`,
`order_hint_bits_minus_1`).

Three items in that set are worth separating out because a pixel path *does* consume them:

| element | status |
|---|---|
| `interp_filter` non-zero on a real pixel decode | randomized only at `crates/aom-dsp/tests/rb_diff.rs:2601`; the encoder gate asserts 0 (`crates/aom-encode/tests/inter_pack_tile_diff.rs:376`). Pixel coverage **NOT ESTABLISHED** |
| `allow_ref_frame_mvs = true` on a pixel decode | implied by `crates/aom-decode/tests/animated_avif.rs:54-61` (the temporal-MV field is built only under it, `crates/aom-decode/src/frame.rs:1492`) but **no assertion pins it** — **NOT ESTABLISHED** |
| `FilmGrainParams::num_y_points == 0` (luma-off grain) | the arms at `crates/aom-decode/src/film_grain.rs:125, 185, 346` are unreachable: `crates/aom-encode/tests/film_grain_diff.rs:121` guarantees ≥1 and `:532` asserts `> 0` |

**Verification status of this subsection.** §3.6a, §3.6b and the `large_scale` finding were
independently re-verified against source before being written down (the greps are quoted in
those sections). The per-field PIXEL / HDR-only / ENC-BYTE classification above is **sourced
from the sub-audit and NOT independently re-verified element by element** — treat it as a lead
list to check, not as established fact. The sub-audit also self-flagged one single-file-scoped
claim: `FilmGrainParams::monochrome` / `subsampling_x` / `subsampling_y`
(`crates/aom-dsp/src/entropy/header.rs:568-571`) are unread *only within*
`crates/aom-decode/src/film_grain.rs` — `add_film_grain` takes independent `mono`/`ss_x`/`ss_y`
arguments (`:1084-1086`) — while the writer/parser does read them (`header.rs:615-620, 3168-3170`).
The correct statement is "the synthesis stage ignores the header copies and trusts separately
passed arguments, so a mismatch would be silent", **not** "unported".

## 4. COMPILE-TIME AND ENVIRONMENT AXES

Sources read: root `Cargo.toml`, all seven `crates/*/Cargo.toml`,
`crates/aom-decode/fuzz/Cargo.toml`, `tools/avif-extract/Cargo.toml`,
`.github/workflows/ci.yml` (266 lines), `justfile` (95 lines),
`crates/aom-sys-ref/build.rs`. There is **no `xtask/Cargo.toml`** — `xtask/` is Python
scripts, not a Rust crate. No `.cargo/config.toml` exists anywhere in the tree, and
`RUSTFLAGS` / `target-cpu` appear in no `*.yml`, `*.toml`, `justfile`, or `*.py` — runtime
dispatch is the only tier selector.

### 4.1 Cargo features

| Feature | file:line | Domain | Behaviour-changing | Tested-by |
|---|---|---|---|---|
| `zenav1-aom-dsp/avx512` (**default ON**) | `crates/aom-dsp/Cargo.toml:23-24` | on/off | **YES** — enables `X64V4Token` codegen. No `incant!`/`#[magetypes]` kernel lists a `v4`/`v4x` tier (tally: 51× `[v3, neon]`, 10× `[v3, neon, wasm128, scalar]`, 4× `[v3, neon, scalar]`), but the three `#[autoversion]` fns emit an AVX-512 variant per their own docs: `crates/aom-dsp/src/intra/mod.rs:448, 476, 821`; `crates/aom-dsp/src/dist/simd.rs:2-3, 18` | **OFF-state UNTESTED** — `--no-default-features` appears only at `ci.yml:243, 264`, both `-p zenav1-aom`, which cannot reach `aom-dsp`'s default. ON-state reach depends on runner CPU — **NOT ESTABLISHED** whether `ubuntu-latest` exposes AVX-512. No `for_each_token_permutation` test covers an `#[autoversion]` fn |
| `zenav1-aom-decode/whereat` (default OFF) | `crates/aom-decode/Cargo.toml:26, 30` | on/off | NO — gates `whereat::define_at_crate_info!()` (`crates/aom-decode/src/lib.rs:156`) and two `.map_err` wrappers (`crates/aom-decode/src/frame.rs:857, 867`) | **UNTESTED** — never enabled in `ci.yml` or `justfile`; its own test module `crates/aom-decode/src/frame.rs:2326` (`#[cfg(all(test, feature = "whereat"))]`) therefore never compiles |
| `zenav1-aom/default = ["decode","encode"]` | `crates/zenav1-aom/Cargo.toml:20` | — | NO | `ci.yml:83, 121, 241, 245, 262, 266` |
| `zenav1-aom/decode` | `crates/zenav1-aom/Cargo.toml:21`, use at `crates/zenav1-aom/src/lib.rs:24` | on/off | NO — re-export gate | `ci.yml:243, 264` |
| `zenav1-aom/encode` | `crates/zenav1-aom/Cargo.toml:22`, use at `crates/zenav1-aom/src/lib.rs:27` | on/off | NO — re-export gate | **encode-only combination UNTESTED** — `--features encode` has zero hits in `ci.yml` + `justfile` |
| `archmage/testable_dispatch` (dev-dep) | `crates/aom-dsp/Cargo.toml:37` | on (test) / off (everything else) | **YES — dispatch selection.** `crates/aom-dsp/src/dispatch/mod.rs:48-66`: with it, `NeonToken` becomes disableable and `AOM_FORCE_SCALAR` is total on aarch64; **without it `AOM_FORCE_SCALAR=1` is a no-op for NEON** and `for_each_token_permutation` reports `simd_perms=0` | ON: `crates/aom-dsp/src/dispatch/mod.rs:167-173`, `ci.yml:166-200`. **OFF-state (production + bench dispatch on ARM) UNTESTED** |
| `enough/std` | `tools/avif-extract/Cargo.toml:15` | on | NO — out-of-workspace tool (own `[workspace]` at `:11`) | **UNTESTED** — `tools/avif-extract` is in no CI step and no justfile recipe |
| fuzz crate | `crates/aom-decode/fuzz/Cargo.toml` (own `[workspace]` at `:19`) | no `[features]` | — | **UNTESTED in CI** — `fuzz` has zero hits in `.github/workflows/*.yml` |

**Feature combinations CI actually builds:** `ci.yml:83`/`:121` `cargo test --profile test-fast
--workspace` (all defaults, x86-64 Linux, twice — the second with `AOM_FORCE_SCALAR: "1"` at
`:95`); `ci.yml:200` `cargo test -p zenav1-aom-dsp` (defaults, aarch64 macOS, ×2 dispatch modes
at `:178`); `ci.yml:241, 262` `cargo build` of the four crates (defaults; windows-11-arm /
macos-15-intel / i686); `ci.yml:243, 264` `--no-default-features --features decode`;
`ci.yml:245, 266` `cargo test -p zenav1-aom` — **this step asserts nothing**:
`crates/zenav1-aom/src/lib.rs` is 28 lines with no `#[cfg(test)]`, no `tests/` dir, and one
fenced block that is `toml` (`:14-19`).

### 4.2 `#[cfg(...)]` gates beyond features

Tally of `#[cfg` attribute forms in `crates/`: 25× `#[cfg(test)]`, 7×
`#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]`, 6× `#[cfg(not(any(…)))]`,
4× `#[cfg(target_arch = "x86_64")]`, 5× `#[cfg(target_arch = "aarch64")]`.

| Gate | file:line | Domain | Behaviour-changing | Tested-by |
|---|---|---|---|---|
| SIMD transform module existence | `crates/aom-dsp/src/transform/mod.rs:12` | x86_64/aarch64 vs rest | **YES** — on any other arch the lane-batched transform module is absent | x86_64 `ci.yml:83, 121`; aarch64 `ci.yml:200`. **i686/wasm32 branch BUILT ONLY** (`ci.yml:262`); the i686 test step `ci.yml:266` targets a crate with no tests |
| Inverse-transform row/col SIMD vs forced-`false` | `crates/aom-dsp/src/transform/inv_txfm2d.rs:269`+`:281`, `:307`+`:322`, `:414`+`:426`, `:451`+`:465` | same | **YES** — selects SIMD pass vs the scalar loop | x86_64/aarch64: `inv_txfm2d_u8_simd_diff.rs`, `txfm2d_simd_perm_diff.rs`, `inv_txfm2d_diff.rs`, `inv_txfm2d_lowbd_diff.rs`. **`not(any(…))` arm UNTESTED** |
| Forward-transform col/row SIMD vs forced-`false` | `crates/aom-dsp/src/transform/txfm2d.rs:214`+`:228`, `:253`+`:264` | same | **YES** | as above; **`not(any(…))` arm UNTESTED** |
| Per-arch SIMD primitives | `crates/aom-dsp/src/transform/simd/prims.rs:103, 320, 535, 540`; `prims16.rs:78, 275, 471, 476` | x86_64 vs aarch64 | **YES** — different intrinsics | both legs, pinned scalar by the permutation differentials |
| Half-batch threshold (runtime `cfg!`) | `crates/aom-dsp/src/transform/simd/mod.rs:140` | aarch64 vs rest | **YES** (path selection). Docs at `:107-137` state `kernel_points == 8` is **INTERPOLATED, not measured** | aarch64 `ci.yml:200`, x86 `ci.yml:83`; bit-identity pinned by `txfm2d_simd_perm_diff.rs` |
| Forward-row-pass gather gate | `crates/aom-dsp/src/transform/simd/mod.rs:660` | aarch64 vs rest | **YES** | as above |
| `wasm128` SIMD tier | 31 sites — `cdef/simd.rs:91, 194, 234, 354, 517, 593, 633, 756, 937, 972`; `intra/simd.rs:97, 143, 220, 257, 315, 352, 412, 447`; `loopfilter/simd.rs:80, 101, 428, 448`; `quant/simd.rs:70, 99`; `restore/wiener.rs:145, 196`; `txb/mod.rs:154`; `txb/simd.rs:34`; `dist/mod.rs:190`; `dist/simd_variance.rs:38` | wasm32 | **YES — a whole distinct numeric kernel tier** | **UNTESTED** — `wasm` has zero hits in `.github/workflows/*.yml` and `justfile` |
| Apple C++-runtime link naming | `crates/aom-sys-ref/build.rs:124` | apple vs rest | NO — link-lib names only | apple `ci.yml:196`, linux `ci.yml:59, 112` |
| Cargo profile `debug-assertions`/`overflow-checks` | root `Cargo.toml:21-23` (`profiling`, both OFF), `:33-37` (`test-fast`, both ON) | debug / test-fast / release / profiling | **YES in effect** — overflow wraps instead of panicking; ~100+ `debug_assert!`s go dark (27 in `crates/aom-encode/src/encode_sb.rs`, 12 in `intra_uv_rd.rs`). No `#[cfg(debug_assertions)]` exists anywhere, so no *source* branches on it | `test-fast` (both ON): `ci.yml:83, 121, 200`. **`profiling`/release UNTESTED** — only reachable via `justfile:82` and `cargo bench`, neither in CI |

### 4.3 The `AOM_FORCE_SCALAR` dispatch pin

Implementation: `crates/aom-dsp/src/dispatch/mod.rs:84-93` (`scalar_forced()`), `:102-122`
(`disable_all_simd_tokens()`).

- **Read:** `std::env::var_os("AOM_FORCE_SCALAR")` at `:87`. **Domain: forced iff set,
  non-empty, and not the literal `"0"`.** `"false"` / `"no"` therefore *force scalar*.
- **One-shot:** cached in `static PIN: OnceLock<bool>` (`:85`). The **first** call applies the
  pin; it is order-sensitive — a dispatch that runs before any `scalar_forced()` call is not
  pinned. **No test asserts the call precedes dispatch.**
- **Mechanism:** `archmage::<T>::dangerously_disable_token_process_wide(true)` for 14 tokens —
  x86 `X64V1`, `X64V2`, `X64Crypto`, `X64V3`, `X64V3Crypto`, `X64V4`, `X64V4x`, `Avx512Fp16`
  (`:107-114`); arm `Neon`, `NeonAes`, `NeonSha3`, `NeonCrc`, `Arm64V2`, `Arm64V3` (`:116-121`).
  Every `Token::summon()` then returns `None` so each `incant!` falls to its `_scalar` arm.
- **Call contract:** every dispatch entry must call `scalar_forced()` before its first
  `incant!` (`:19-27`); e.g. `crates/aom-dsp/src/transform/simd/mod.rs:664`.
- **Scope limits, stated in-source:** `X64V1` refuses disablement, documented harmless
  (`:104-106`). On **aarch64** the pin only reaches NEON under `archmage/testable_dispatch` —
  test builds only; making it total in production "is NOT implemented" (`:60-71`). **This is S8.**
- **Tested by:** `ci.yml:90-121` (Linux `--workspace`, pin at `:95`); `ci.yml:166-200`
  (aarch64 matrix, `force_scalar` `"0"`/`"1"` at `:172-178`, **scoped `-p zenav1-aom-dsp` only**
  per the comment at `ci.yml:157-158` — `aom-decode`/`aom-encode` have **no ARM differential at
  all**). Unit: `dispatch/mod.rs:146, 194`. Locally `justfile:12, 26, 48`.

### 4.4 Environment variables

Grepped `env::var`, `std::env::var`, `env::vars`, `option_env!`, `env!`, `getenv` across
`crates/`, `tools/`, `xtask/`, and `crates/aom-sys-ref/build.rs` (the only `build.rs`).
`option_env!` has zero occurrences; `xtask/*.py` and `tools/` contain no `os.environ`/`getenv`.

| Var | file:line | Domain | Behaviour-changing | Tested-by |
|---|---|---|---|---|
| `AOM_FORCE_SCALAR` | `crates/aom-dsp/src/dispatch/mod.rs:87` | unset/`""`/`"0"` → off; anything else → on | **YES — process-wide SIMD tier** | `ci.yml:95, 178`; `justfile:13, 27, 49` |
| `AOM_CONFORMANCE_DIR` | `crates/aom-decode/tests/conformance_corpus.rs:310`; `inter_ratchet.rs:45`; `inter_walking_skeleton.rs:27`; `inter_real_frame.rs:63`; `crates/aom-encode/tests/encoder_gate_e2e_byte_match.rs:3378`; `encoder_gate_chroma_ss_e2e.rs:970`; `kb6_real_rd_localize.rs:66`; `crates/aom-bench/src/lib.rs:78` | any path; default `<manifest>/../../conformance/data` | NO to the codec; **YES to which vectors run** | **variable never set** — zero hits in `ci.yml` + `justfile`; the default path is provisioned by `ci.yml:75, 119` |
| `AOM_DBG_BLOCKS` | read `crates/aom-decode/src/lib.rs:1135` (`OnceLock`, `is_some()` — **any** value including `""`/`"0"` enables) | set/unset | NO — four `eprintln!` consumers only (`crates/aom-decode/src/frame.rs:1543`; `crates/aom-decode/src/lib.rs:3012, 3178, 4487`) | **UNTESTED** — zero hits in `ci.yml`, `justfile`, `xtask/`. Note it is **not** `"0"`-aware, unlike `AOM_FORCE_SCALAR` |
| `AOM_BENCH_SMOKE` | `crates/aom-bench/benches/gate3.rs:23`, used `:29, :104` | unset/`"0"` → real; else smoke | NO to output; **YES to measurement validity** — collapses to 2-3 rounds / 2 s budget (`:29-35`); header at `:11` and `justfile:73-74` say "NUMBERS ARE MEANINGLESS" | `justfile:75`. **No CI leg** — `bench` has zero hits in `ci.yml` |
| `KB7_OUT_DIR` | `crates/aom-encode/tests/kb7_rd_localize.rs:260, 563` | any writable dir; absent → skip write | NO — dumps `.av1` for a sibling-C harness | **UNTESTED** — zero hits in `ci.yml`, `justfile` |
| `FUZZ_CRASH_DIR` | `crates/aom-decode/tests/fuzz_sweep.rs:131` | any path; default hardcoded `/root/fuzz-corpus/aom-rs/stable-crashes` | NO — crash sink; the default is an absolute `/root/…` that will not exist on a CI runner or a macOS box | **UNTESTED** — zero hits |
| `FUZZ_SMOKE_ITERS` | `crates/aom-decode/tests/fuzz_sweep.rs:239` | parseable `u64`, default `60_000`; unparseable → silent fallback | **YES to coverage** | **never overridden** — zero hits |
| `FUZZ_SMOKE_SEED` | `crates/aom-decode/tests/fuzz_sweep.rs:243` | parseable `u64`, default `0x9E37_79B9_7F4A_7C15`, forced odd at `:245` | **YES — determines the entire mutation corpus explored** | **never overridden** — CI fuzzes one fixed trajectory every run |
| `NUM_JOBS` | `crates/aom-sys-ref/build.rs:282` | positive int; else `available_parallelism()`, else 4 | NO — cmake `-j` | implicit on `ci.yml:59, 112, 196` |
| `OUT_DIR` | `crates/aom-sys-ref/build.rs:347` | cargo-provided | NO | `ci.yml:59, 112, 196` |
| `CARGO_MANIFEST_DIR` (`env!`) | `crates/aom-sys-ref/build.rs:93`; `crates/aom-bench/src/lib.rs:81`; `crates/aom-decode/tests/{conformance_corpus.rs:313, fuzz_regression.rs:23, fuzz_sweep.rs:71, inter_ratchet.rs:48, inter_walking_skeleton.rs:30, inter_real_frame.rs:66, animated_avif.rs:23}`; `crates/aom-encode/tests/{encoder_gate_e2e_byte_match.rs:3381, encoder_gate_chroma_ss_e2e.rs:973, kb6_real_rd_localize.rs:69}` | cargo-provided | NO | every CI leg |
| `LIBAOM_SHA` (CI only, read by no Rust code) | `.github/workflows/ci.yml:17`, used `:56, 110, 194` | pinned `03087864…` | NO at runtime; **YES to cache validity** — it is only a cache *key*; the real pin is `.gitmodules` + `crates/aom-sys-ref/build.rs:23` | **nothing asserts it equals `PINNED_SHA`** — `LIBAOM_SHA` has zero hits in `crates/` |

### 4.5 Unpinned oracle-definition axes (same forgotten class, not env vars)

`crates/aom-sys-ref/build.rs` hardcodes the oracle's own definition:
`ORACLE_FP_CFLAGS = "-ffp-contract=off"` (`:44`, applied at `:260` to libaom and `:355` to every
shim); `-DCMAKE_BUILD_TYPE=Release -DCONFIG_MULTITHREAD=0 -DCONFIG_AV1_{DECODER,ENCODER}=1`
(`:262-267`); `PINNED_SHA` (`:23`); `extra_shim_cflags` giving `cnn_cscalar` `-O3 -DNDEBUG` while
every other shim gets `-O2` (`:85-89`).

**The shim C compiler is chosen by first-hit probe — `clang`, then `cc`, then `gcc`
(`crates/aom-sys-ref/build.rs:185-186`).** Which compiler builds the differential oracle's shims
is whatever is installed; nothing records or asserts it, and no test compares shim output across
compilers. `-ffp-contract=off` pins contraction but not the compiler.

---

## 5. LIBAOM CONTROL SURFACE CROSS-REFERENCE

Header: pinned submodule `upstream/aom/aomcx.h` / `aomdx.h` at libaom v3.14.1
(`03087864cf4b…`, `reference/BUILD_CONFIG.md:3`). Counted: **155** numbered entries in
`enum aome_enc_control_id`, **41** in `enum aom_dec_control_id`.
Port-side id constants: `crates/aom-sys-ref/src/lib.rs:8960-9058` (`cx_ctrl` + `PROBE_TABLE`,
cross-checked against the real enum by `shim_cx_ctrl_id_by_probe`,
`crates/aom-sys-ref/shim/dec_shim.c:799-823`).

Verdict legend: **MODELLED** · **DRIVEN-BUT-UNMODELLED** (harness emits it to C; nothing on the
port consumes it) · **PINNED** (the shim base config forces one constant; the port assumes it) ·
**NOT-DRIVEN** · **N/A** (inter / RC / SVC / realtime-only — listed anyway).

### 5.1 Encoder controls that are MODELLED

`AOME_SET_CPUUSED` 13 (`aomcx.h:223`) · `AOME_SET_SHARPNESS` 16 (`:250`) ·
`AOME_SET_TUNING` 24 (`:291`, IQ/SSIMULACRA2/PSNR arms only — VMAF/BUTTERAUGLI/SSIM not ported) ·
`AOME_SET_CQ_LEVEL` 25 (`:301`) · `AV1E_SET_LOSSLESS` 31 (`:366`) ·
`AV1E_SET_CDF_UPDATE_MODE` 44 (`:519`, modes 0/1) · `AV1E_SET_SUPERBLOCK_SIZE` 56 (`:664`) ·
`AV1E_SET_ENABLE_CDEF` 58 (`:684`, **values 0/1 only** — see S3) ·
`AV1E_SET_DISABLE_TRELLIS_QUANT` 62 (`:721`, all 4 values) ·
`AV1E_SET_ENABLE_QM` 63 / `QM_MIN` 64 / `QM_MAX` 65 (`:732, 745, 757`) ·
`AV1E_SET_ENABLE_RECT_PARTITIONS` 73 (`:824`) · `AB_PARTITIONS` 74 (`:832`) ·
`1TO4_PARTITIONS` 75 (`:840`) · `MIN_PARTITION_SIZE` 76 (`:851`) · `MAX_PARTITION_SIZE` 77 (`:862`) ·
`ENABLE_INTRA_EDGE_FILTER` 78 (`:870`) · `ENABLE_TX64` 80 (`:890`) · `ENABLE_FLIP_IDTX` 81 (`:914`) ·
`ENABLE_RECT_TX` 82 (`:926`) · **`ENABLE_CHROMA_DELTAQ` 87 (`:976`)** ·
`ENABLE_FILTER_INTRA` 98 (`:1073`) · `ENABLE_SMOOTH_INTRA` 99 (`:1084`) ·
`ENABLE_PAETH_INTRA` 100 (`:1092`) · `ENABLE_CFL_INTRA` 101 (`:1102`) ·
`ENABLE_SUPERRES` 102 (`:1110`, reached via `cfg.rc_superres_*` not this ctrl) ·
`ENABLE_PALETTE` 104 (`:1123`) · `ENABLE_INTRABC` 105 (`:1127`) · `ENABLE_ANGLE_DELTA` 106 (`:1131`) ·
`DELTAQ_MODE` 107 (`:1151`, modes 0/2/3/6) · `DELTALF_MODE` 108 (`:1159`) ·
`FILM_GRAIN_TEST_VECTOR` 112 (`:1193`) · `FILM_GRAIN_TABLE` 113 (`:1198`) ·
`DENOISE_NOISE_LEVEL` 114 / `DENOISE_BLOCK_SIZE` 115 (`:1201, 1204`, modelled but not ctrl-driven) ·
`CHROMA_SUBSAMPLING_X/Y` 116/117 (`:1207, 1210`) · `REDUCED_TX_TYPE_SET` 118 (`:1213`) ·
`INTRA_DCT_ONLY` 119 (`:1216`) · `INTRA_DEFAULT_TX_ONLY` 121 (`:1223`) ·
`QUANT_B_ADAPT` 122 (`:1226`, modelled but not ctrl-driven and inert on the default envelope) ·
`ENABLE_DIAGONAL_INTRA` 141 (`:1369`) · `ENABLE_DIRECTIONAL_INTRA` 145 (`:1398`) ·
`ENABLE_TX_SIZE_SEARCH` 146 (`:1408`) · `DELTAQ_STRENGTH` 148 (`:1420`) ·
**`ENABLE_RESTORATION` 59 (`:694`)**.

### 5.2 PINNED by the shim base config (the port assumes the pinned value)

| ctrl | `aomcx.h` | pinned to | pin site |
|---|---|---|---|
| `AV1E_SET_TILE_COLUMNS` 33 | `:393` | log2 = 0 **in the base shim** | `crates/aom-sys-ref/shim/dec_shim.c:393`; the aom-bench encode path asserts single-tile at `crates/aom-bench/src/lib.rs:1086`. **NOT globally pinned** — `ref_encode_av1_kf_tiles` (`crates/aom-sys-ref/src/lib.rs:9250`) drives it non-zero, used by `crates/aom-decode/tests/real_bitstream.rs:879-889, 951` and by the encoder gate `encoder_gate_multitile_e2e` (`crates/aom-encode/tests/encoder_gate_multitile.rs`) |
| `AV1E_SET_TILE_ROWS` 34 | `:411` | log2 = 0 **in the base shim** | `dec_shim.c:394`; same non-pinned caveat |
| `AV1E_SET_NUM_TG` 70 | `:803` | 1 (by construction) | `crates/aom-encode/src/obu_assemble.rs:2, 102, 107-115` |
| `AV1E_SET_DELTAQ_MODE` 107 | `:1151` | 0 in the base set | `dec_shim.c:395` (overridden per-family elsewhere) |
| `AV1E_SET_AQ_MODE` 40 | `:481` | 0 at every encode call site | `dec_shim.c:402`; see S4 |

### 5.3 DRIVEN-BUT-UNMODELLED (the silent-ignore set — see §0.3)

`AV1E_SET_COEFF_COST_UPD_FREQ` 126 (`aomcx.h:1254`) and `AV1E_SET_MODE_COST_UPD_FREQ` 127
(`:1264`) — **S1/S2**. `AV1E_SET_AQ_MODE` 40 (`:481`) — **S4** (encoder side; the decoder track
*does* drive it non-zero at `crates/aom-decode/tests/real_bitstream.rs:98-102, 182, 204, 224`,
which is correct because the decoder models segmentation).
`AV1E_SET_ENABLE_CDEF` 58 value **3** — **S3** (value 2, "disable for non-reference frames", is inert on a lone KEY frame).
`AV1E_SET_ENABLE_ADAPTIVE_SHARPNESS` 172 (`:1624`) — **partially** modelled: the qindex cap is
implemented in `crates/aom-encode/src/lf_search.rs:405-414`, but `pick_filter_level_from_q`
(the speed≥6 entry) takes `(base_qindex, bit_depth, allintra, sharpness_cfg)` and **not** the
adaptive flag (`crates/aom-encode/src/lf_search.rs:490-495`; the comment at `:478` says the flag
is "default-off and out of this envelope"), and the trellis consumer takes the raw CLI sharpness
(`crates/aom-encode/src/tx_search.rs:855-856` → `:785`). Whether libaom's adaptive sharpness also
feeds the trellis rdmult path — i.e. whether that second half is a real divergence — is
**NOT ESTABLISHED**.
`AV1E_SET_DV_COST_UPD_FREQ` 142 (`:1379`) — a declared constant
(`crates/aom-sys-ref/src/lib.rs:9015`), a `PROBE_TABLE` entry (`:9056`) and a shim probe case
(`dec_shim.c:824`), but never emitted and with no `ToggleKnobs` field; self-documented inert at
`crates/aom-sys-ref/src/lib.rs:9013-9014` and at `HANDOFF-TOGGLES.md:32`.

### 5.4 NOT-DRIVEN and NOT-MODELLED — the genuinely absent tools

Ranked by relevance to the allintra-KEY scope:

1. **`AV1E_SET_ENABLE_KEYFRAME_FILTERING` 36 (`aomcx.h:430`)** — the temporal filter on key
   frames. **Zero occurrences workspace-wide**: `grep -rn --include='*.rs' -iE
   'keyframe_filtering|temporal_filter' crates/` → 0. **This is a declared scope exclusion, not
   an oversight** — `PARITY.md:7` puts "temporal filtering" out of scope explicitly. Listed here
   because it is the largest unmodelled KEY-frame tool and the exclusion is easy to forget when
   the port is described as a "drop-in replacement".
2. **`AV1E_SET_SCREEN_CONTENT_DETECTION_MODE` 171 (`:1608`)** and **`AV1E_SET_TUNE_CONTENT` 43
   (`:510`)** — the *detector* is not ported. The port consumes `allow_screen_content_tools` from
   the bootstrapped frame header (`crates/aom-encode/src/pack.rs:142-149`) and takes
   `is_screen_content` as a caller-supplied bool (`crates/aom-encode/src/allintra_vis.rs:467-469`,
   `crates/aom-encode/src/encode_sb.rs:462-465`). Cannot silently diverge on the current harness
   (the bit is bootstrapped), but a standalone port encoder has no way to derive it. **The allintra
   default is mode 2 (AA-aware)**, `aomcx.h:1608-1613`. Already tracked at `PARITY.md:106, 212`.
3. **`AV1E_SET_QM_Y/U/V` 66/67/68 (`:769, 781, 793`)** — per-plane QM level overrides. The port
   models per-plane levels (`crates/aom-encode/src/lib.rs:141`) but derives them from
   `qm_min`/`qm_max` (`crates/aom-dsp/src/quant/build_quantizer.rs:411-427`); there is no override
   path. Verified not driven: `grep -rn --include='*.rs' -iE '\bqm_y\b|SET_QM_Y' crates/` → 0
   control hits (the `qm_v` hits in `xform_quant_optimize*_diff.rs` are quant-matrix *vector*
   locals, unrelated).
4. **`AV1E_SET_LOOPFILTER_CONTROL` 149 (`:1429`)** — only a doc mention of the default
   `LOOPFILTER_ALL` at `crates/aom-encode/src/lf_search.rs:39`.
5. **`AV1E_SET_AUTO_INTRA_TOOLS_OFF` 151 (`:1441`)** — 0 hits.
6. **CICP / color config**: `COLOR_PRIMARIES` 45, `TRANSFER_CHARACTERISTICS` 46,
   `MATRIX_COEFFICIENTS` 47, `CHROMA_SAMPLE_POSITION` 48, `COLOR_RANGE` 52, `RENDER_SIZE` 53
   (`:540, 565, 586, 593, 619, 626`) — serialized by the header writer
   (`crates/aom-dsp/src/entropy/header.rs:891-927`, `:278-313`) but never derived and never driven.
7. **Level signalling**: `TARGET_SEQ_LEVEL_IDX` 54 / `GET_SEQ_LEVEL_IDX` 55 (`:649, 656`) —
   `does_level_match` / `set_bitstream_level_tier` exist **only** in
   `crates/aom-encode/tests/seq_level_idx_diff.rs:86-116`; there is no `src` module.
8. `ROI_MAP` 8, `ACTIVEMAP` 9, `SCALEMODE` 11, `STATIC_THRESHOLD` 17, `NOISE_SENSITIVITY` 42,
   `SINGLE_TILE_DECODING` 109, `MTU` 71, `TIER_MASK` 129, `MIN_CR` 130, `VMAF_MODEL_PATH` 134,
   `ENABLE_DNL_DENOISING` 140, `PARTITION_INFO_PATH` 143, `EXTERNAL_PARTITION` 144,
   `SKIP_POSTPROC_FILTERING` 157, `ENABLE_SB_QP_SWEEP` 158, `ENABLE_RATE_GUIDE_DELTAQ` 160,
   `RATE_DISTRIBUTION_INFO` 161, `AUTO_TILES` 166, `ENABLE_LOW_COMPLEXITY_DECODE` 170,
   `VALIDATE_HBD_INPUT` — none driven, none modelled.
9. `ROW_MT` 32, `FP_MT` 153, `FRAME_PARALLEL_DECODING` 37 — bit-inert here (the oracle is built
   `CONFIG_MULTITHREAD=0`, `crates/aom-sys-ref/build.rs:262-267`, and the port is single-threaded).
10. All inter / RC / SVC / realtime controls (~60 ids) — out of the current scope by design.

Unused enum slots (no control): 10, 15, 18, 23, 30, 69, 72 (`aomcx.h:193, 233, 257, 284, 354, 795, 816`).

### 5.5 Two things that are NOT `aom_codec_control` ids at all

- **`--dist-metric`** (`AOM_DIST_METRIC_PSNR` / `_QM_PSNR`, `aomcx.h:1816, 1820`) has **no
  control id**. It is only reachable through `aom_codec_set_option(ctx, "dist-metric", …)`, which
  is exactly what the shim does (`crates/aom-sys-ref/shim/dec_shim.c:3384-3392`).
- **Superres** is set via `cfg.rc_superres_mode` / `cfg.rc_superres_denominator`
  (`crates/aom-sys-ref/shim/dec_shim.c:3061-3063`), **not** via `AV1E_SET_SUPERRES_MODE` /
  `AV1E_SET_SUPERRES_DENOMINATOR`. **Those identifiers do not exist in libaom v3.14.1**
  (`grep -rn 'AV1E_SET_SUPERRES_MODE' upstream/` → no matches) — the doc comment naming them at
  `crates/aom-sys-ref/src/lib.rs:12011-12012` is wrong. Doc bug, no behavioural impact.

### 5.6 Decoder controls (`aomdx.h`) vs `DecodeConfig`

`DecodeConfig` (`crates/aom-decode/src/config.rs:117-130`) is **not** a port of
`aom_dec_control_id`; it is an orthogonal resource-safety surface. Exactly **one** of the 41
decoder controls has a semantic counterpart: `AOMD_SET_FRAME_SIZE_LIMIT` 296 (`aomdx.h:477`)
≈ `DecodeLimits.max_width` / `max_height` (`crates/aom-decode/src/config.rs:28, 30`, enforced
`:60-91`). Everything else in `DecodeConfig` (`max_pixels`, `max_memory_bytes`, the cooperative
stop token, `AllocMode::Fallible`'s try-reserve) has **no libaom analog**. Nothing in
`DecodeConfig` is silently ignored — every field has an enforcement site (`config.rs:60-91`,
`:164-185`, `:190-195`).

The decoder controls whose absence is **behaviourally visible**:

| ctrl | `aomdx.h` | port behaviour |
|---|---|---|
| `AV1D_SET_SKIP_FILM_GRAIN` 282 | `:390` | grain is applied unconditionally whenever the bitstream asks — `crates/aom-decode/src/frame.rs:883`: `if header.film_grain_params_present && header.film_grain.apply_grain` |
| `AV1_SET_SKIP_LOOP_FILTER` 267 | `:278` | no such switch; `crates/aom-decode/src/frame.rs:1742` only mirrors C's internal gate |
| `AV1_SET_TILE_MODE` 272 | `:314` | large-scale tile explicitly out of scope, `crates/aom-decode/src/frame.rs:24-26` |
| `AV1D_SET_OPERATING_POINT` 279 | `:362` | seq fields are parsed (`crates/aom-decode/src/frame.rs:484-485`) but never selectable |
| `AV1D_SET_OUTPUT_ALL_LAYERS` 280 | `:374` | 0 hits |
| `AV1D_SET_ROW_MT` 277 | `:347` | 0 hits (single-threaded port) |
| `AV1D_SET_IS_ANNEXB` 278 | `:352` | 0 hits |
| `AV1_SET_DECODE_TILE_ROW/COL` 270/271 | `:308, 309` | no partial-tile decode |

### 5.7 CORRECTIONS — two sub-audit claims that were WRONG, verified here

Recorded so nobody re-derives them. Both were `grep`-scoping artifacts (searching only
`crates/aom-encode/src` for a feature that lives in `crates/aom-dsp`).

1. **`AV1E_SET_ENABLE_CHROMA_DELTAQ` (87) is MODELLED, not silently ignored.**
   `crates/aom-dsp/src/quant/build_quantizer.rs:362` (`enable_chroma_deltaq: bool` parameter of
   `av1_set_quantizer`) and `:377` (the consuming branch). It is e2e byte-gated by
   `encoder_gate_chroma_deltaq_e2e` (`crates/aom-encode/tests/encoder_gate_tune_iq_e2e.rs:943`),
   consistent with `PARITY.md:77`.
2. **`av1_pick_filter_restoration` IS ported.** `crates/aom-dsp/src/restore/pick.rs`
   (`pick_filter_restoration`, module doc at `:6`, the search at `:772`), driven from
   `crates/aom-bench/src/lib.rs:1862` via the import at `:60`, unit-gated by
   `crates/aom-dsp/tests/pick_search.rs:203, 240, 332, 342, 366, 385` and e2e-gated by
   `crates/aom-bench/tests/lr_restoration_gate.rs`. `AV1E_SET_ENABLE_RESTORATION` (59) is
   **MODELLED**, consistent with `PARITY.md:76`.

---

## 6. RANKED: UN-COVERED AXES BY RISK

Ranking rule: an axis that steers **pixel output or bitstream bytes** outranks one that
affects only speed, diagnostics, or measurement. "Un-covered" means *no test drives it to a
non-default value*, or the port accepts it and does nothing.

### Tier 1 — steers bitstream bytes, and nothing catches the divergence

| # | Axis | Why it is tier 1 | Evidence |
|---|---|---|---|
| 1 | **`--usage=good` (0) / `realtime` (1) at any `--cpu-used`** | `usage` is aomenc's *default*; `--allintra` is the flag. The port collapses 0 and 1 (`crates/aom-bench/src/lib.rs:1087`), applies the ALLINTRA speed cascade unconditionally (`:1357`), and has no `set_good_speed_features_*` port. A known-divergent pin exists at speed 0 (`crates/aom-bench/tests/inter_e2e_search.rs:180-191`); **speed ≥ 1 is untested in either direction** | S5/S6 |
| 2 | **`--coeff-cost-upd-freq` / `--mode-cost-upd-freq` non-default** | the C reference changes cost-table refresh cadence; the port keeps rebuilding per-SB. A cell with either knob set is an A/B mismatch that reports as a *port* bug | S1/S2, `crates/aom-bench/src/lib.rs:388` says so in-source |
| 3 | **`--aq-mode=1/2` + `--passes=2`** | enables **segmentation** in C; the port hardcodes `seg_enabled: false` (`crates/aom-encode/src/pack.rs:257`) with no encoder-side segmentation anywhere. The shim parameter exists and is unguarded (`crates/aom-sys-ref/src/lib.rs:8862-8864`) | S4 |
| 4 | **`--enable-cdef=3`** (adaptive-by-qindex, `upstream/aom/aomcx.h:681`) | the tune shim forwards the raw i32 (`crates/aom-sys-ref/shim/dec_shim.c:3406`); the adaptive arms are explicitly unported (`crates/aom-encode/src/pickcdef.rs:33-38`). Every other shim narrows the parameter to `bool`, so the trap is reachable only through the tune path | S3 |
| 5 | **cross-family knob composition (nearly all of it)** | e.g. `--tune=iq` × any C8/C9/C10 partition/tx/intra toggle; `--enable-qm` × `--min-partition-size`; `--superres-denominator` × `--enable-cdef`; lossless × anything. **Structurally unreachable** — 15 fixed-signature encode shims + 3 disjoint knob structs, and `PARITY.md:237` claims composition only *within* the tune bundle. What IS composable is whatever one shim signature happened to bundle (e.g. `--sb-size=128` × `--tile-columns` via `ref_encode_av1_kf_tiles`, `crates/aom-sys-ref/src/lib.rs:9250-9270`) | §0.1, §0.2 |
| 6 | **`--sharpness` ≠ 0 outside the tune gate** | reachable only via `RefTuneKnobs`; the aom-bench harness hardcodes `sharpness: 0` (`crates/aom-bench/src/lib.rs:1403`, and `sf.tx_type_search_policy(false, 0)` at `:1456`). So sharpness never combines with any of the 29 `ToggleKnobs` | §1.4 |
| 7 | **`--enable-adaptive-sharpness` at `--cpu-used >= 6`** | `pick_filter_level_from_q` does not take the flag (`crates/aom-encode/src/lf_search.rs:490-495`) | S9 |
| 8 | **`--qm-y` / `--qm-u` / `--qm-v` per-plane overrides** | modelled per-plane internally but derived only from qm_min/qm_max; no override path, never driven | §5.4 item 3 |
| 9 | **`--enable-hdr-deltaq`** | the `av1_set_quantizer` HDR arm is unported (`crates/aom-dsp/src/quant/build_quantizer.rs:333-336`). Latent until the first 10-bit BT.2020 cell is added | S7 |
| 10 | **CICP color config** (`--color-primaries` / `--transfer-characteristics` / `--matrix-coefficients` / `--chroma-sample-position` / `--color-range` / `--render-size`) | these are **sequence-header bytes**. Randomized non-default values ARE exercised — but only in a **self round-trip** (`crates/aom-dsp/tests/rb_diff.rs:661-702, 1943-1946`; module doc `:1-4` — write→read inverse, not a C differential for these fields). No *encode* ever produces a non-default value and the encoder never derives them | §3.2, §5.4 item 6 |
| 10b | **`large_scale` (`large_scale_tile`)** — the one frame-header field with **zero** non-default coverage anywhere | live write branch `crates/aom-dsp/src/entropy/header.rs:1565` and read branch `:3173`; every struct literal in the tree is `false` (`crates/aom-dsp/tests/rb_diff.rs:2423, 2734`; `crates/aom-dsp/tests/header_diff.rs:1751`). Declared out of scope on decode (`crates/aom-decode/src/frame.rs:24-26`) — but the **writer branch exists and is unreachable by any test** | §3.7 |
| 11 | **`--enable-cdef=1` at `--cpu-used` 1..6** (`CDEF_FAST_SEARCH_LVL1..5`) | the fast search levels are ported and table-unit-tested but **not e2e-gated** — only speed-0 `CDEF_FULL_SEARCH` is. Self-declared at `PARITY.md:113-118`, which also names the cheap fix ("CDEF-on cells at `--cpu-used=1..6`") | `PARITY.md:113-118` |
| 12 | **SB128 with CDEF on** | blocked on the pack's SB64 envelope; the search's >64-fb arms exist | `PARITY.md:117-118` |

### Tier 2 — steers numeric results, but only on a build configuration CI never produces

| # | Axis | Evidence |
|---|---|---|
| 13 | the `#[cfg(not(any(x86_64, aarch64)))]` scalar transform path — i686 and wasm32 | `crates/aom-dsp/src/transform/mod.rs:12`, `inv_txfm2d.rs:281, 322, 426, 465`, `txfm2d.rs:228, 264`. i686 is *built* (`ci.yml:262`) but its only test step (`:266`) runs a crate with no tests |
| 14 | the entire `wasm128` SIMD tier (31 sites) | §4.2; `wasm` has zero hits in `ci.yml` + `justfile` |
| 15 | `zenav1-aom-dsp/avx512 = off` | `crates/aom-dsp/Cargo.toml:23-24`; `--no-default-features` never reaches `aom-dsp` |
| 16 | `archmage/testable_dispatch = off` — the configuration production and benchmarks run in | `crates/aom-dsp/src/dispatch/mod.rs:60-71`; S8 |
| 17 | `aom-decode` / `aom-encode` differentials on **aarch64** | `ci.yml:200` is scoped `-p zenav1-aom-dsp` per the comment at `ci.yml:157-158`; the decoder and encoder composition layers have **no ARM differential at all** |
| 18 | `overflow-checks` / `debug-assertions` off (release, `profiling`, `cargo bench`) | root `Cargo.toml:21-23` vs `:33-37`; ~100+ `debug_assert!`s go dark |
| 19 | the shim C compiler (`clang` / `cc` / `gcc`, first found) | `crates/aom-sys-ref/build.rs:185-186` — the *oracle's own arithmetic definition* is host-dependent and unrecorded |

### Tier 3 — affects coverage or measurement, not output

| # | Axis | Evidence |
|---|---|---|
| 20 | `FUZZ_SMOKE_SEED` / `FUZZ_SMOKE_ITERS` never varied | `crates/aom-decode/tests/fuzz_sweep.rs:239, 243` — CI explores one fixed mutation trajectory forever |
| 21 | `LIBAOM_SHA` is a cache key that nothing checks against `PINNED_SHA` | `.github/workflows/ci.yml:17` vs `crates/aom-sys-ref/build.rs:23`; drift silently reuses a stale `upstream/build` |
| 22 | `AOM_CONFORMANCE_DIR` never set; `AOM_DBG_BLOCKS` not `"0"`-aware | §4.4 |
| 23 | `zenav1-aom` `encode`-without-`decode` never built; `whereat` never enabled (its own test module never compiles) | §4.1 |
| 24 | `tools/avif-extract` and `crates/aom-decode/fuzz` in no CI step | §4.1 |
| 25 | `min_partition_size_px` ∈ {32,64,128} and `max_partition_size_px` ∈ {4,8,16} never tested | tested values are min 4/8/16 (`crates/aom-bench/tests/toggles_rd_close.rs:153, 181`) and max 32/64/128 (`:166, 182`, `crates/aom-bench/tests/sb128_e2e.rs:86`) |
| 26 | `--cdf-update-mode=2` (selective) never swept | `crates/aom-bench/src/lib.rs:369-371` argues it is identical to mode 1 on a lone KEY frame — plausible, but unasserted |

### What a permutation gate built only from `PARITY.md` + `HANDOFF-TOGGLES.md` would miss

Items **1, 5, 6, 7, 8, 10** and all of tier 2 and tier 3. Cross-checked by grepping both
documents: `usage` / `GOOD` / `set_good` appear only at `PARITY.md:6, 140, 145-150` (inside the
C2 loop-restoration narrative, never as an axis); `combination` / `permutation` / `compose`
appear only at `PARITY.md:237`, and only about composition *within* the tune bundle; neither
document mentions cargo features, `#[cfg]` gates, `AOM_FORCE_SCALAR`, or any environment
variable. Items **2, 3, 4, 9** are partially present (`HANDOFF-TOGGLES.md:22` for the cost-upd
pair, `PARITY.md:281-287` for aq-mode/segmentation, `PARITY.md:113-118` for the CDEF control
values) — but as prose inside feature narratives, not as a checkable axis list.
