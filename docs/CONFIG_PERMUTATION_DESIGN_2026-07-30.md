# Config-permutation gate — design, evidence, and honest coverage arithmetic

**Landed 2026-07-30.** Code: `crates/aom-bench/src/config_perm.rs` (collapse
engine) + `crates/aom-bench/tests/config_permutations.rs` (the gate).
Evidence: `benchmarks/config_perm_independence_2026-07-30.tsv`.

## The gap this closes

`crates/aom-bench/tests/toggles_rd_close.rs` gates ~25 CLI-toggle knob families
**one at a time**, each on its own 3-cell grid, and every one of them is
byte-identical to real aomenc *alone* (PARITY.md §A). Nothing gated them
**together**. That matters here specifically, because this repo has already
found two defects that are only visible in a particular configuration — the C11
cdf-update pack bug and the `--disable-trellis-quant=2` FINAL_PASS
`dry_run_output_enabled` bug (HANDOFF-TOGGLES.md).

The gate covers the *combination* space with small images, in **26.3 s wall**,
by collapsing rather than exploding.

> **Extended 2026-07-30 with a CONTENT axis** — see the section at the end of
> this document. Short version: 12 content probes across 5 classes, a
> 468-cell content-sensitivity matrix, and full-strength covering arrays on the
> one class that moves. Exactly **one** of the 21 axes is content-sensitive,
> and its root cause is now code-cited. Default tier: 44 tests, 3,352
> byte-identity cells, 39.6 s.

## What a cell proves — DERIVED vs REPLAYED (read this before quoting a number)

**The port never authors a sequence header.** `write_sequence_header_obu`
(`crates/aom-dsp/src/entropy/header.rs:1046`) has no call site in any encoder
path — the eight `crates/*/src` references are its own definition, three doc
comments, and the C-oracle FFI shim; the only real callers are tests. Every
encode parses a sequence header out of a real aomenc bootstrap stream and emits
an `OBU_FRAME` payload alone (`aom_encode::obu_assemble`). Verified
independently for this gate on 2026-07-30; cross-checked against
`docs/CONFIG_AXIS_INVENTORY_2026-07-30.md`.

So the 21 axes are **not one homogeneous coverage count**
(`config_perm::AxisKind`, reported by the gate itself):

| kind | n | axes | what a cell proves |
|---|---|---|---|
| **DERIVED** | 16 | rect, ab, p14, minp, maxp, smth, paeth, cfl, dir, diag, adlt, tx64, rtx, flip, dtxo, trel | end-to-end: the port computes this configuration itself |
| **REPLAYED seq-header bit** | 2 | fint (`--enable-filter-intra`), edgf (`--enable-intra-edge-filter`) | the port behaves correctly *given* this bit; `port_encode_with` (aom-bench/src/lib.rs:1093-1102) asserts the bootstrap's bit equals the knob — an agreement check, not evidence the port can produce it |
| **REPLAYED frame-header bit** | 3 | rtxs, txss, cdf | same, for a frame-header bit the port parses from the bootstrap; the port *does* derive the downstream search/pack behaviour, just not the coded bit |

The **cell contexts** (bit depth, monochrome, chroma subsampling, frame size,
superblock size) are **replayed in full** — they all arrive from the bootstrap
sequence header. They are therefore not covering-array factors at all; they are
contexts the array is replayed under. No count in this document should be read
as "the port can produce these formats".

**Not reachable, noted rather than forced:** `large_scale`
(`aom_dsp::entropy::header`, live write at :1565 and read at :3173) has no
non-default coverage anywhere in the tree. There is no `ToggleKnobs` axis and no
control pair in `EncodeCell::c_encode_ctrls` that reaches it, so this gate
cannot. Reaching it needs a new knob plus a C control — encoder-harness work
outside this gate's ownership.

## Coverage arithmetic (computed by `config_permutation_coverage_arithmetic`, pinned)

| stage | count | note |
|---|---:|---|
| raw cartesian product of the 21 axes | **14,155,776** | 15 binary + 2 binary + 3 ternary + 1 quaternary |
| − C-forbidden combinations | **10,616,832** | −3,538,944 (25.0%), one exclusion, cited below |
| → distinct **effective** configs, 64×64 4:2:0 | **777,600** | 13.7× collapse |
| → distinct effective configs, 64×64 monochrome | **388,800** | 27.3× — the whole UV loop dies |
| → distinct effective configs, 32×32 4:2:0 | **622,080** | 17.1× — no 64px block ⇒ tx64 dies |
| − independence collapse | **0** | measured, see below: no pair qualifies |
| t-wise covering array | **17 / 63 / 187 rows** | t=2 / t=3 / t=4 |
| − effective collapse applied to the array | **0–1 rows** | see "where the collapse actually pays" |
| **cells actually run (default tier)** | **2,617** | 14 (context × quality) points × 187 rows |

Plus, outside the array: ~60 collapse-proof encodes, 26 per-axis liveness
probes, 17 `--use-intra-dct-only` cells, 6 monochrome-vector finding cells.

**Continuous axes.** There are none among the knobs. `cq` is continuous in
principle; it is sampled at **{5, 12, 20, 32, 40, 48, 55, 63}** — denser at the
aggressive end where the RD balance moves most, per the sweep-discipline rule.
It is a *context*, not an array factor, so the array is replayed whole at each
sampled quality rather than crossed with it.

### The one C-forbidden combination

`--enable-tx-size-search=0` + `--enable-tx64=0`:

> `assert(oxcf->txfm_cfg.enable_tx64 || tx_search_type != USE_LARGESTALL);`
> — libaom `av1/encoder/encodeframe.c:2461`

`--enable-tx-size-search=0` forces `tx_size_search_level = 3` = `USE_LARGESTALL`
(`speed_features.c:2726`), so the pair aborts a debug-built libaom and is
undefined in release. It is excluded from rows *and* from the tuple set (a
t-tuple only reachable through an illegal row is unreachable, period), and
`config_perm::illegal_reason` carries the citation. A grep of every
`assert(...)` in `av1_cx_iface.c`, `encoder.c`, `encodeframe.c`,
`speed_features.c`, `partition_search.c`, `tx_search.c` and
`intra_mode_search.c` mentioning `oxcf` / `*_cfg` found no other constraint
reachable from these axes; the remaining `oxcf` asserts are about global motion,
lossless, `min_cr` recode and superres, none of which this matrix touches.

## Mechanism 1 — effective-config collapse

`Effective::resolve(row, ctx)` maps a knob row to the encoder state it actually
produces. Every canonicalisation cites the libaom line that makes the knob dead:

| collapse | citation |
|---|---|
| `--enable-rect-partitions=0` ⇒ AB **and** 1to4 dead | `do_rectangular_split` gates `partition_rect_allowed` (`partition_search.c:3383`, `:3389`); AB needs it (`:5166`, `:5172`), and HORZ_4/VERT_4 need `partition_rect_allowed[HORZ]` (`:5181`, `:5187`) |
| `--max-partition-size=64` ≡ the 128 default at SB64 | `min(sf_default, dim_to_size(CLI), sb_size)`, `partition_strategy.h:214` |
| `--enable-directional-intra=0` ⇒ diagonal, angle-delta **and** the intra edge filter dead | mode skips at `intra_mode_search.c:1555-1559`; angle delta at `:1317`/`:1585`; the edge filter runs only for directional modes — `build_directional_and_filter_intra_predictors` returns early for filter-intra (`reconintra.c:1198`) and then `assert(is_dr_mode)` precedes `if (!disable_edge_filter)` (`:1204-1207`) |
| monochrome ⇒ the whole UV loop and the CFL knob dead | `num_planes == 1` |
| frame smaller than one SB, or `max-partition-size < 64` ⇒ `--enable-tx64` dead | a 64-point transform needs a 64px block, which needs a non-force-split BLOCK_64X64 root (`av1_blk_has_rows_and_cols`, `partition_search.c:3389`) |
| `--use-intra-dct-only` / `--use-intra-default-tx-only` subsume `--enable-flip-idtx` | the policy has already narrowed the search to one type before `get_tx_mask`'s `DCT_ADST_TX_MASK` arm applies |
| `--cdf-update-mode=2` ≡ `=1` | `encoder.c:4375-4395`: case 2 is `frame_is_intra_only ? 0 : 1` ⇒ 0 on a lone KEY frame; the only other reader, `should_force_mode_cost_update` (`rd.c:762`), is `rt_sf`-gated |
| `--disable-trellis-quant=0` (FULL) ≡ `=3` (NO_ESTIMATE_YRD, the default) | `init_rd_sf` (`speed_features.c:2479-2498`) + `is_trellis_used`; the two differ only in `estimate_yrd_for_sb`, which is inter-only |

**The documented "verified-INERT" cases in HANDOFF-TOGGLES.md are re-derived, not
hardcoded** — `config_perm::tests::engine_rediscovers_the_documented_inert_cases`
asserts the engine reaches the same verdict for `--disable-trellis-quant=0`,
`--cdf-update-mode=2` and `--max-partition-size=64` without any special case.
(`--dv-cost-upd-freq` and `--quant-b-adapt` are not `ToggleKnobs` axes, so they
are out of this matrix; `--disable-trellis-quant=0` is the one the engine can
speak to, and it does.)

### The collapse is falsifiable, in four places

1. `effective_collapse_is_real_{64cq32,32cq32,mono}` — for **every** equivalence
   the engine predicts, encode both rows with the port **and** with real aomenc
   and require all four payloads identical. 13–14 equivalences per context, each
   also checked on a non-default background so the equivalence is shown to
   survive composition. Two distinct failure modes, both informative: C payloads
   differing means the signature is **under-refined** (the array is
   under-covering); C agreeing while the port differs means the **port** steers
   on a knob C ignores.
2. `run_array`'s in-situ duplicate check — every row the collapse *drops* must
   really reproduce its representative's bytes on both sides.
3. `run_array`'s stock check — a row the engine resolves to the stock effective
   config must produce the stock C payload.
4. `every_axis_level_is_live_in_some_context`'s **over-collapse detector** — if
   the engine resolves a singleton row to the stock effective config (i.e.
   claims the level cannot change anything) then real aomenc must agree, on
   every context. This is the sound direction of the implication and it covers
   every axis, not just the listed equivalences.

Plus `redundant_levels_are_globally_redundant`, a full combinatorial walk
proving the three claimed level-equivalences hold at **every** background
(10,616,832 legal rows each), not only at the sampled points.

### Where the collapse actually pays — and where it does not

Honest result: **applying the effective collapse to the covering array removes
0–1 of the 187 rows.** A covering array is already a sparse, maximally-diverse
sample; two of its rows almost never resolve to the same state. The collapse's
leverage is entirely **on the space** (14.2 M → 777.6 k, and 388.8 k on
monochrome), i.e. on what the array has to *aim at*, not on the sample it
produces. Reporting it as a row-count reduction would overstate it.

The concrete, checkable payoff is the level-equivalence set — three levels
(`maxp=64`, `cdf=2`, `trel=0`) that are inert at every background, proven both
combinatorially and against the oracle, and therefore never need a cell of their
own in any context.

## Mechanism 2 — independence: measured, and the answer is zero

**Method** (`independence_evidence_sweep`, `--ignored`, offline, one run, baked
in): for each of the 210 axis pairs, encode the four corners {A0B0, A0B1, A1B0,
A1B1} with the **real C encoder** (the oracle, not the port — independence is a
property of libaom's configuration semantics), decode each with the port
decoder, and measure

* **stream change** — did the coded frame payload move at all? and
* **footprint** — the set of 4×4 blocks, **across all planes**, whose
  reconstruction changes when the axis flips.

A pair is INDEPENDENT only when **both axes are live** (each moves the stream
under both settings of the other) **and** `footprint(A | B=0) == footprint(A |
B=1)` and symmetrically — each axis lands on the same part of the decision state
whatever the other is doing.

**Result on the 64×64 cq32 reference context** (raw:
`benchmarks/config_perm_independence_2026-07-30.tsv`):

| verdict | pairs | meaning |
|---|---:|---|
| INTERACTING | 117 | both axes live, footprints differ ⇒ **must be crossed** |
| INERT-A / INERT-B / INERT-BOTH | 39 / 28 / 3 | one axis does not move the stream at all here ⇒ effective-collapse territory, not independence |
| SIGNALLING-ONLY-A / -B | 6 / 16 | the axis moves the payload but not the reconstruction (e.g. `--cdf-update-mode=0`) ⇒ the footprint measure cannot speak, so the pair stays crossed |
| ILLEGAL | 1 | the `txss=0 × tx64=0` corner |
| **INDEPENDENT** | **0** | — |

So `config_perm::INDEPENDENT_PAIRS` is **empty**, and every pair is crossed. That
is the honest outcome, not a shortfall: on an intra encoder every configuration
knob feeds the same RD loop, so changing one shifts which blocks exist and
therefore where a second knob can even apply. The first two rows of the TSV show
it cleanly — `rect × ab` and `rect × p14` come out INERT-B precisely because
turning rect off *kills* the other knob, which is the transitive death the
effective collapse already models, arriving from a second, independent direction.

**Stated blind spot** rather than hidden: the footprint is a *reconstruction*
measure, so it cannot see a change that alters only the coded symbols. That is
exactly why the stream-change flag gates the verdict — such an axis is
classified SIGNALLING-ONLY and never independent. The sweep also self-guards: it
asserts at least one pair comes out INTERACTING, so a broken footprint measure
fails loudly instead of reporting universal independence.

## The covering array

AETG-style randomised construction with a fixed seed, so the array is
byte-reproducible across runs and machines (the gate pins the row counts —
17 / 63 / 187 — so a silent shrink is a failure). Each candidate row is seeded
from a still-uncovered tuple, guaranteeing progress; the best of 64 candidates
is kept; a final pass drops rows covering nothing unique. Row 0 is always the
all-default row — the stock byte-exact envelope, i.e. the harness-faithfulness
control. `covering_array_is_complete_legal_and_deterministic` re-derives every
reachable tuple and asserts it is covered.

**Strength: t = 4.** t=2 (pairwise) is the usual defensible choice, but at 187
rows and ~17 ms–140 ms per cell the whole budget is 26 s, so there is no reason
to stop at pairwise: t=4 covers **every 4-way knob interaction** and both
historical defects in this repo were configuration-specific. t=5 was measured
and rejected: 521 rows, but **98 s just to build the array**, which is worse than
the encoding it would gate.

## Tiers

* **Default** (`cargo test -p zenav1-aom-bench --test config_permutations`) —
  32 tests, everything below.
* **Deep** (`-- --ignored`) — one test, `independence_evidence_sweep`. It is
  opt-in because it *writes* `benchmarks/`, not because it is slow (5.8 s).

Nothing was moved out of the default tier to hit the time budget.

| what | contexts | strength | cells |
|---|---|---|---|
| covering-array byte-identity gate | 64²cq32, 64²cq63, 32²cq32, 4:4:4, 4:2:2, bd10, monochrome, 128² (4 SBs) | t=4 | 8 × 187 = 1,496 |
| quality ladder on 64² | cq 5, 12, 20, 40, 48, 55 | t=4 | 6 × 187 = 1,122 |
| effective-collapse proof | 64²cq32, 32²cq32, monochrome | — | ~60 encodes |
| per-axis liveness | all 8, early-exit | — | 26 probes |
| `--use-intra-dct-only` (known-open) | 64²cq32 | t=2 | 17 |
| monochrome-vector finding | `av1-1-b10-24-monochrome`, cq 12/32/63 | — | 6 |
| arithmetic + exhaustive redundancy | — | — | no encoding |

**Every cell asserts byte-identity of the frame OBU payload against real
aomenc.** That is both the strongest available contract and cheaper than the
`rd_close` path — no decode and no zensim on the exact path. A cell that is
*not* byte-identical additionally has its port stream decoded by the port
decoder and its geometry checked, so no cell ever "encodes and checks nothing".

## Measured result

**2,617 array cells, 32 tests, 0 failures, 26.3 s wall / 241 s CPU** on a 12-core
M4 (macOS, `cargo test --profile test-fast -j 4`, default libtest thread count).
Anti-vacuity: **100 % of non-stock rows change real aomenc's own output** on
every one of the 14 (context × quality) points.

### Teeth (both demonstrated, both reverted; `git diff` clean afterwards)

1. **Combination-only mis-threading.** In `aom-bench/src/lib.rs`,
   `pol.enable_flip_idtx = knobs.enable_flip_idtx` was changed to
   `knobs.enable_flip_idtx || knobs.reduced_tx_type_set` — so the knob threads
   correctly on its own and only mis-threads when combined with
   `--reduced-tx-type-set=1`.
   * `toggles_rd_close` (the existing 25-test per-knob suite): **25 passed** — it
     cannot see this.
   * `config_permutations`: **21 of 27 tests FAILED**, e.g.
     > `32cq32: 2 of 93 covering-array cells are NOT byte-identical to real aomenc — a knob COMBINATION diverges where every knob is exact alone. Offenders: 32cq32_p140-minp8-maxp32-dir0-tx640-flip0-rtxs1-cdf0 (port 183B vs C 180B), 32cq32_ab0-p140-minp8-paeth0-cfl0-adlt0-fint0-flip0-rtxs1-txss0-cdf2-trel1 (port 174B vs C 175B)`
2. **Over-collapsing engine.** `Effective::resolve` was changed to fold
   `--disable-trellis-quant=1` (NO_TRELLIS) into the default — a collapse that
   is not real.
   > `the collapse engine resolves these axis levels to the STOCK effective config — i.e. it claims they cannot change anything — but real aomenc reacted to them. The Effective signature is OVER-COLLAPSING: every row it folds away on that basis is coverage silently lost: ["trel=1 on ctx=64cq63"]`

## Findings

### 1. `--use-intra-default-tx-only=1` and `--enable-diagonal-intra=0` diverge on `av1-1-b10-24-monochrome` (pinned open)

Both knobs are byte-identical to real aomenc on every gated context, and both
diverge on the corpus's one native monochrome vector. 64×64 crop at (64,64),
speed-0 ALLINTRA:

| knob | cq12 | cq20 | cq32 | cq48 | cq63 |
|---|---|---|---|---|---|
| `--use-intra-default-tx-only=1` | DIVERGE 623/623 B | DIVERGE 418/424 | DIVERGE 229/240 | DIVERGE 79/80 | DIVERGE 14/15 |
| `--enable-diagonal-intra=0` | exact | exact | DIVERGE 225/231 | exact | exact |

Isolated to the **content**, not the format: a full 27-knob singleton sweep is
clean on bd8 4:2:0, on bd8 monochrome derived from `av1-1-b8-01-size-64x64`, on
bd10 4:2:0, and on bd10 monochrome derived from `av1-1-b10-00-quantizer-00`. A
bd12 promotion of the same monochrome content reproduces the
`default-tx-only` divergence, so it is not a bd10-specific quantizer path. The
equal-size and one-byte deltas are the KB-10 / KB-12 "cheaper RD decision"
near-tie signature. The **stock encode of this content is byte-exact**, so the
proven envelope is unaffected.

`toggles_rd_close.rs`'s grid is bd8 4:2:0 only, which is why this was invisible
until the knobs were replayed over other contexts.
`mono_vector_open_divergences_pinned` pins the table exactly and is
self-promoting: a cell that starts matching fails the test.

This is a **finding, not a fix** — no encoder change was made. The monochrome
*context* of the covering array is therefore derived from clean content
(`av1-1-b10-00-quantizer-00` with the chroma planes dropped), so the monochrome
axis is still covered at full strength.

### 2. Two axes were vacuous on the primary context (fixed by adding a gate)

The independence sweep's per-axis liveness table exposed that, at 64×64 cq32:

* `--enable-cfl-intra=0` **never changes the C stream** — CFL never wins on that
  content — and only bites on the multi-superblock and bd10 contexts;
* `--enable-tx64=0` **never changes the C stream** at cq32 (no 64-point
  transform is chosen there) and only bites at cq63.

Without noticing this, both axes would have been "covered" hundreds of times
without ever being exercised. `every_axis_level_is_live_in_some_context` now
pins the per-axis claim — every axis level must, on its own, move real aomenc on
at least one default-tier context, and the witness context is reported:

| axis=level | witness | axis=level | witness |
|---|---|---|---|
| rect=0 | 64cq63 | edgf=0 | 64cq63 |
| ab=0 | 422cq32 | tx64=0 | 64cq63 |
| p14=0 | 64cq63 | rtx=0 | 64cq63 |
| minp=8 / 16 | 32cq32 | flip=0 | 422cq32 |
| maxp=64 | *inert everywhere (proven collapse)* | dtxo=1 | 64cq63 |
| maxp=32 | 64cq63 | rtxs=1 | 64cq63 |
| smth=0 | 32cq32 | txss=0 | 64cq63 |
| paeth=0 | 422cq32 | cdf=0 | 64cq63 |
| cfl=0 | 128cq32 | cdf=2 | *inert everywhere (proven collapse)* |
| dir=0 | 64cq63 | trel=1 | 64cq63 |
| diag=0 | 64cq63 | trel=2 | 32cq32 |
| adlt=0 | 64cq63 | trel=0 | *inert everywhere (proven collapse)* |
| fint=0 | 64cq63 | | |

### 3. Independence collapse buys nothing here

See mechanism 2. Zero of 210 pairs qualify. The reduction all comes from the
effective-config collapse plus the covering array. Recorded so a future session
does not re-derive it — but the sweep is committed and cheap (5.8 s), so it
should be re-run if the axis set changes.

### 4. All 2,617 combination cells are byte-identical

No combination defect was found. Given finding 1 (a knob that is exact on its
gated grid and divergent on other content), the more useful reading is that the
*combination* axis is clean while the *content and format* axis was not — which
is where the next effort belongs.

## Chunk 2 — what a follow-up should do

1. **Root-cause finding 1.** Localise the `--use-intra-default-tx-only=1`
   divergence on `av1-1-b10-24-monochrome` with the decode-both / sibling-C dump
   recipe (KB-2/KB-6/KB-7). It is the same shape as the pinned
   `--use-intra-dct-only` UV-loop mis-model but on a luma-only frame, so the
   chroma suspects are excluded from the start — likely a cheaper localisation.
   Promote the pinned table to a hard byte-identity assert when it closes.
2. **Widen the content axis, not the knob axis.** Finding 1 says the residual
   risk is content, not combinations. Replay the t=4 array over more conformance
   vectors (the `av1-1-b8-01-size-*` family covers partial-SB geometries at
   16..226 px) rather than raising t. Budget: at ~17 ms/cell for
   knob-narrowed rows, another ten contexts is ~30 s CPU.
3. **Reach the axes this matrix cannot.** `--sb-size=128` (encoder walk is
   SB64-only, HANDOFF-TOGGLES.md), `--coeff/mode-cost-upd-freq` (C ctrls emitted,
   port gate unwired), `--quant-b-adapt` (needs the `aom_quantize_b_adaptive`
   kernel family), and `large_scale`. Each needs a `ToggleKnobs` field plus port
   plumbing first; `CellCtx.sb_px` and `Effective::resolve` are already written
   to follow the superblock size, and the cost-update axes are exactly the ones
   whose collapse is superblock-count-dependent, so the 128² context is where
   they would first bite.
4. **Multi-tile.** `tiles_log2 == 0` is asserted by `port_encode_with`, so the
   whole matrix is single-tile. Note the audit's caveat: the multi-tile byte gate
   uses the NON-deriving assembler (`obu_assemble.rs:143`), with the deriving
   path proven separately by `obu_assemble_multitile_diff.rs:341-351` — so
   multi-tile is not yet end-to-end proof of header derivation, and a
   multi-tile permutation context would need that resolved first.
5. **Lossless.** `port_encode_with` asserts `base_qindex > 0`, so cq0 is out of
   this matrix; the lossless envelope has its own gate
   (`encoder_gate_lossless_cq0_e2e_kb5_repro`) and collapses several axes
   (`init_rd_sf` forces `NO_TRELLIS_OPT`), which `Effective::resolve` would need
   a lossless arm to model.

---

# The CONTENT axis (added 2026-07-30)

Chunk-2 item 2 above says *"widen the content axis, not the knob axis"*. This
section is that work: a content taxonomy derived from the encoder's own
branch points, a measurement of which of the 21 axes actually move with
content, and an expansion sized by that measurement rather than applied
uniformly.

## 1. The taxonomy, and why it is not a list of adjectives

"Textured vs flat" is only a coverage axis if it *steers a decision*. So the
taxonomy is derived from the content-dependent branch points that are live on
this harness's envelope (speed 0, ALLINTRA, KEY, single tile, `--enable-palette=0`
and `--enable-intrabc=0` — both forced off by `EncodeCell::c_encode_ctrls`).

Walking those branches gives exactly **one** content property that survives:

### `allow_screen_content_tools` — libaom's screen detector

`av1_set_screen_content_options` (`av1/encoder/encoder.c:2439`) →
`estimate_screen_content` (`:2042-2100`, the default detector: aomenc's
`screen_detection_mode` defaults to `AOM_SCREEN_DETECTION_STANDARD`,
`av1_cx_iface.c:405` — the anti-aliasing-aware variant is set only for
`AOM_TUNE_IQ` / `AOM_TUNE_SSIMULACRA2`, `:1969`). It is a **hard threshold on a
countable source statistic**:

```text
for each full 16x16 luma block:
    n_colors = |{ pix >> (bd - 8) }|         // av1_count_colors{,_highbd}
    if 1 < n_colors <= 4: ++counts_1         // kColorThresh = 4
allow_screen_content_tools = counts_1 * 256 * 10 > width * height
```

Transcribed as `config_perm::screen_stat`, which is the classifier the whole
section runs on. Two properties make it the right axis:

* **it is bit-depth independent by construction** —
  `av1_count_colors_highbd` (`intra_mode_search.c:352-357`) down-converts to
  the 8-bit domain before binning, explicitly "to provide consistency of
  behavior for palette search between lbd and hbd encodes". So the same clip at
  bd8 and bd10 classifies identically, which is what makes this a *content*
  axis and not a smuggled format axis;
* **it changes the meaning of one of our 21 knobs.** `get_tx_mask`
  (`tx_search.c:1806-1808`) resolves `--use-intra-default-tx-only=1` through
  `get_default_tx_type(PLANE_TYPE_Y, xd, tx_size, cpi->use_screen_content_tools)`,
  which returns `DCT_DCT` **when the screen flag is set** instead of the
  mode-derived tx type.

It also moves things that are *not* the mechanism here, listed so they are not
mistaken for it: the per-block palette flag written by `write_palette_mode_info`
and priced by `intra_mode_info_cost_y` (both gated on
`av1_allow_palette(allow_screen_content_tools, bsize)`, independent of
`--enable-palette`), and the speed-feature reads at `speed_features.c:375-381`
and `:2909` (whose fields are inter-only on this envelope).

### The classes actually probed

| class | branch-point rationale | probes |
|---|---|---|
| **SCREEN** | `estimate_screen_content` fires ⇒ `get_tx_mask`'s default-tx-type arm changes, per-block palette flag coded+priced | `scr_mono_b8`, `scr_mono_b10`, `scr_ibc_b8` |
| **NATURAL** | the detector's negative class — photographic, high colour count | `nat_64x64`, `nat_allintra`, `nat_cdfupd`, `nat_b10` |
| **NOISE** | high-frequency energy drives the tx-type near-ties the `flip` / `rtx` / `rtxs` / `dtxo` axes narrow | `grain_b8`, `grain_b10` (film-grain-synthesised decodes) |
| **DC-FLAT** | decoded at a crushing qindex ⇒ DC-dominated, large flat regions — the a-priori candidate for triggering the colour-count detector | `flat_b8_q63`, `flat_b10_q63` |
| **DETAIL** | decoded at qindex 0 ⇒ maximum AC energy, the opposite pole | `det_b8_q00` |

**Measured, and worth stating because it is counter-intuitive:** the DC-FLAT
class does **not** trigger the screen detector. `flat_b8_q63` and
`flat_b10_q63` score `0/16` blocks with ≤4 colours — quantisation ringing and
dither keep the per-block colour count well above the threshold. Flatness in
the perceptual sense is not the same property as libaom's colour-count
statistic, and only the latter steers anything.

Class populations are pinned by `content_taxonomy_is_measured_and_pinned`, and
the classifier is **anchored to the oracle in the sound direction**: if
`allow_screen_content_tools == 0` then `av1_allow_palette` is false for every
block and `--enable-palette=1` *cannot* change the C payload, so any content
this file calls NOT-screen must produce byte-identical C encodes with palette
on and off. It does; all three SCREEN probes move. A wrong classifier fails
that assert instead of silently invalidating the matrix.

## 2. Which axes are content-sensitive — measured

`run_content_matrix` replays **all 26 singleton axis levels** on each content
(9 non-screen at cq32; the 3 screen contents at cq12/32/63 — 468 cells), each
cell asserting byte-identity against real aomenc. The full 12 × 3 × 26 = 936
cell grid is in `benchmarks/config_perm_content_axis_2026-07-30.tsv`
(`content_axis_evidence_sweep`, `--ignored`).

| axis | content-sensitive? | evidence |
|---|---|---|
| `dtxo` (`--use-intra-default-tx-only=1`) | **YES** | diverges on **every** SCREEN content and **no** non-screen content: `scr_mono_b8` cq12/32/63, `scr_mono_b10` cq12/32/63, `scr_ibc_b8` cq12/32 (cq63 is inert on the C side, so there is nothing to diverge on) |
| `diag` (`--enable-diagonal-intra=0`) | knock-on only | one cell — `scr_mono_b10` cq32, the KB-17 near-tie; the bd8 twin of the same clip does **not** reproduce it, so it is a near-tie coincidence, not a class property |
| the other **19** axes (rect, ab, p14, minp, maxp, smth, paeth, cfl, dir, adlt, fint, edgf, tx64, rtx, flip, rtxs, txss, cdf, trel) | **NO** | 0 divergences across 12 contents × 26 levels, spanning bd8/bd10, 4:2:0/4:0:0, natural/noise/flat/detail/screen |

So the honest headline is close to the honest-stop clause, but not identical to
it: **content is a live axis, but it is a one-axis phenomenon with a single
mechanism**, and that mechanism is now root-caused (below) rather than
mysterious. KB-17 is not a lone outlier in the sense of "unexplained"; it is
the *visible tip* of one class-wide defect that reproduces on bd8 4:2:0
non-monochrome content — which the KB-17 write-up could not know, because the
only screen content it had was the two monochrome vectors.

### Root cause of the `dtxo` class divergence (found 2026-07-30, NOT fixed)

`crates/aom-encode/src/speed_features.rs:991`:

```rust
// Non-screen textured envelope; screen-content would thread the real
// cpi->use_screen_content_tools here.
use_screen_content_tools: false,
```

The port models `get_default_tx_type` faithfully
(`aom_encode::tx_search::get_default_tx_type_y`, which takes
`use_screen_content_tools` and returns `DCT_DCT` when it is set) — but the
caller that builds `TxTypeSearchPolicy` hardcodes the flag to `false`. On
screen-detected content C therefore searches `DCT_DCT` under
`--use-intra-default-tx-only=1` while the port searches the mode-derived tx
type. Every observed divergence in the matrix is that one line.

This is a **finding, not a fix** — no encoder change was made here. The pin is
self-promoting: `check_content_shard` fails if a cell starts matching (the flag
got threaded → re-pin and promote `dtxo` into the screen contexts' array) and
fails if a cell starts diverging.

**This supersedes the CONTENT-vs-FORMAT framing of KB-17 finding 1**: the
divergence is not "the corpus's one native monochrome vector", it is "any
content on which `estimate_screen_content` fires", and the corpus's three such
vectors all reproduce it (two monochrome, one 4:2:0). The isolation KB-17
performed was correct — the mono-ised *natural* content it tested as a control
does not trigger the detector, so it was byte-exact for the right reason.

## 3. Is 2,617 valid on the content axis?

**On its own terms, yes; as a statement about content, no.** The 2,617 array
cells are 14 (context × quality) points that all draw luma from **two
photographic clips** (`av1-1-b8-01-size-*` and the `*-00-quantizer-*` family) —
one content class out of five, and the class in which zero of the 21 axes
misbehave. Nothing in the old count was wrong; it simply could not speak about
content, and the one place the port *does* steer on content was invisible to
it (as KB-17 discovered by accident on a probe outside the array).

The expansion is sized by the measurement, not applied uniformly:

| addition | cells | why this size |
|---|---:|---|
| content-sensitivity matrix (26 singleton levels) | **468** | 9 non-screen × cq32 + 3 screen × cq{12,32,63}. The non-screen class is flat across quality in the 936-cell evidence sweep, so paying for its quality ladder in the default tier buys nothing |
| t=4 covering array on `scr_ibc_b8` (bd8 4:2:0 SCREEN) | **187** | the class that moves gets FULL strength, in the *same format* as the primary 64cq32 context, so anything it catches is attributable to content alone |
| t=3 covering array on `scr_mono_b8` (bd8 4:0:0 SCREEN) | **63** | the corpus's only bd8 monochrome content and the bd8 twin of KB-17's vector; t=3 because it is a second probe of a class already covered at t=4 |
| `dtxo=1` × t=2 array on SCREEN content | **17** | the combination companion to the standalone class divergence |
| **new total** | **735** | |

Classes that changed **no** axis's outcome (NATURAL, NOISE, DC-FLAT, DETAIL)
get the 26-level matrix and **no covering array of their own** — that is the
"expand where the measurement says to" rule applied honestly. A content class
on which every one of 26 axis levels behaves identically to the primary context
does not need 187 more rows to say so again.

Array cells go **2,617 → 2,867**; total byte-identity cells in the file go
**2,617 → 3,352** (plus 48 stock-encode byte-identity checks).

`dtxo` is pinned to its default level inside `run_content_array`. Forcing one
column of a covering array to a constant leaves every t-tuple among the other
columns covered, so the screen t=4 context still proves every 4-way interaction
among the remaining 20 axes; the pinned axis is covered separately and
completely by the matrix (standalone) and by the t=2 verdict set (in
combination). This is the treatment `--use-intra-dct-only` already gets.

**Result: all 735 new cells byte-identical to real aomenc except the pinned
`dtxo` set.** No knob COMBINATION diverges on any content — the screen t=4
array is clean at 100 % C-moved. The content risk is entirely the one
standalone axis.

## 4. Budget

| tier | wall |
|---|---|
| pre-existing gate (33 tests) | 26.3 s |
| + the content axis (11 tests, 735 cells) → 44 tests | **39.6 s (+13.3 s)** |
| the content tests alone, run as a filtered set | 15.6 s |
| deep tier (`--ignored`): `content_axis_evidence_sweep`, 936 cells | 106.5 s, opt-in |

(12-core M4, `cargo test --profile test-fast -j 4`, default libtest thread
count, machine shared with two other agents — so treat these as an upper
bound rather than a clean measurement.)

Nothing was thinned to hit the budget: the default tier runs every content
probe, every axis level on the class that moves, and a full-strength t=4 array
on it. What is in the deep tier is the *redundant* part — the non-screen
classes' quality ladder, which the evidence sweep shows is flat.

## 5. Teeth — the asymmetry, demonstrated

The point of new cells is that they catch something the old ones cannot. One
perturbation, applied to a **content-gated** path and then reverted
(`git diff` on the port crates clean afterwards, verified):

`crates/aom-encode/src/pack.rs:508`

```rust
-    kfs.allow_palette = allow_palette(cfg.allow_screen_content_tools, bsize);
+    kfs.allow_palette = allow_palette(false, bsize); // TEETH
```

This is invisible on every non-screen content **by construction**:
`allow_screen_content_tools` is already `false` there, so the two expressions
are the same value. It only bites where the detector fires.

Result of one full run with the perturbation in place — **34 passed, 10
failed**:

* **all 2,617 pre-existing covering-array cells stayed GREEN** — every
  `combinations_t4_*` shard (8 contexts), all three `combinations_quality_ladder_*`,
  `combinations_dct_only_verdict_set_pinned`, both collapse proofs, the
  per-axis liveness test, the arithmetic. 31 of the 32 pre-existing tests
  passed;
* the one pre-existing test that fired is `mono_vector_open_divergences_pinned`
  — the KB-17 probe, which is the *only* pre-existing cell in the file that
  touches screen content, and is not part of the 2,617;
* **9 of the 12 new content tests failed**, and the 3 that passed are exactly
  the three non-screen shards (`content_sensitivity_natural_s0/s1/s2`) — the
  classes the taxonomy says cannot see this path.

Two of the failure messages, quoted:

> `scr_ibc_b8cq32: 63 of 63 covering-array cells on SCREEN-class content are
> NOT byte-identical to real aomenc. `dtxo` is pinned to default here, so this
> is a knob combination that diverges on this CONTENT and nowhere else.
> Offenders: scr_ibc_b8cq32_stock (port 54B vs C 54B),
> scr_ibc_b8cq32_rect0-ab0-p140-maxp32-smth0-paeth0-tx640-rtx0-flip0-rtxs1-trel2
> (port 66B vs C 66B), …`

> `scr_ibc_b8/cq12: the STOCK encode of this content is NOT byte-identical to
> real aomenc — that is a plain envelope regression, not a knob-vs-content
> interaction`

After reverting: **44 passed, 0 failed, 2 ignored, 34.9 s.**

## 6. What a follow-up should do

1. **Thread `cpi->use_screen_content_tools`** into `TxTypeSearchPolicy`
   (`crates/aom-encode/src/speed_features.rs:991`) from the parsed frame
   header's `allow_screen_content_tools`, then re-pin
   `CONTENT_DIVERGENT_CELLS`, `SCREEN_DTXO_DIVERGENT_ROWS` and KB-17's table,
   and promote `dtxo` out of `pin_dtxo_default` into the screen arrays.
   Watch the `scr_mono_b10` cq32 `diag=0` cell separately — it is a near-tie
   that may or may not move with the same change.
2. **Palette and intrabc as real axes.** Both are forced off by
   `c_encode_ctrls`, so the largest consequence of screen detection is outside
   the matrix entirely. `EncodeCell::c_encode_screen` and
   `ToggleKnobs::enable_palette` already exist; wiring them as axes would need
   the port's palette RD search gated into `port_encode_with`.
3. **More screen content.** The corpus has exactly three screen-detected
   vectors and this section uses all three. A fourth class of screen content
   (synthetic text/UI) would have to be generated, not fetched.

---

# The SIZE axis (added 2026-07-30, same day)

The section above replays the t=4 array at **three** frame geometries — 64x64,
32x32, 128x128, all SB64 — and calls the result 2,617 cells. This section
answers, with data rather than intuition, whether three is the right number.

## Short answer

**2,617 is valid on the framesize-SPEED-FEATURE axis and invalid on the frame
GEOMETRY axis.** The two halves need separating, because the obvious reading of
`crates/aom-encode/src/speed_features.rs:695` —

> `prune_tx_type_using_stats = 2` (needs is_480p_or_larger — false on the
> {64,128}^2 grid)

— is that a speed feature is pinned to 0 in all 2,617 cells and could therefore
be wrong without any cell noticing. That reading is **incorrect at this array's
speed**: `prune_tx_type_using_stats` needs speed >= 2 (`speed_features.c:261`)
or speed >= 4 (`:299`), and every cell in the matrix is **speed 0**
(`Ctx::cell` -> `EncodeCell::real_content(.., 0)`). Real aomenc also computes 0
there, so nothing is unexercised that a cell could witness. The same is true for
every other framesize-dependent field: see the table below — each is either
speed-gated above 0, or dead on an intra frame, with the citation.

The **one** exception is `use_square_partition_only_threshold`, and the size
axis turns out to be a *geometry* axis at speed 0, not a speed-feature axis:
frame-edge partial superblocks and the superblock size itself are where the
encoder's behaviour actually moves.

## The size -> effective-config table

Model: `config_perm::size_derived(ctx, speed) -> SizeDerived` — the size analogue
of `Effective`. It resolves a `CellCtx` to the encoder state its geometry
determines, so two sizes with equal `SizeDerived` cannot differ *because of
size*. `config_perm::size_class_partition` collapses a candidate size list into
classes; `size_class_inventory_is_pinned` pins the result.

### Framesize-dependent derivations, both sides cited

`set_allintra_speed_feature_framesize_dependent`, libaom
`av1/encoder/speed_features.c:166-340`. LIVE = observable on the all-intra KEY
path this harness encodes.

| derivation | threshold | libaom | port | live at speed 0? |
|---|---|---|---|---|
| `use_square_partition_only_threshold` | `min(w,h) >= 480` -> BLOCK_128X128 else BLOCK_64X64 (and 720p tiers at speed>=1) | `speed_features.c:175-183`, `:211-217`, `:238-242`, `:315` | `aom-encode/src/partition_pick.rs:2446-2470`, applied `:2586-2594` | **YES — but only at SB128.** Its sole intra consumer is the rect-kill `if (bsize > threshold)` (`partition_search.c:5700`; port `partition_pick.rs:2593`), which needs a block strictly larger than the threshold. At SB64 the largest block IS BLOCK_64X64, so `bsize > BLOCK_64X64` is unsatisfiable and both sides of the 480p branch behave identically. (Its other reader, `partition_search.c:4265`, is inside a `!frame_is_intra_only` block.) |
| `default_min_partition_size = BLOCK_8X8` | `min(w,h) >= 2160` | `speed_features.c:187-189` | **UNMODELLED** — `aom-encode/src/speed_features.rs:471` pins BLOCK_4X4 below speed 6 and `:891` sets BLOCK_8X8 unconditionally at speed>=6 | **YES, and it is a port gap.** Read by `set_max_min_partition_size` (`partition_strategy.h:225`) on every frame type. See "Out of budget" below. |
| `prune_tx_type_using_stats` | `>= 480p` AND speed>=2 (=1) / speed>=4 (=2) | `speed_features.c:261`, `:299` | `aom-bench/src/lib.rs:1364` (the sf setter itself is framesize-blind by design) | **no** — speed-gated. Already gated at speed 2 by `tx_stats_prune_e2e.rs` on a 512x512 cell. |
| `prune_tx_size_level` | `< 480p` AND `use_highbitdepth` | `speed_features.c:184`, `:263-265`, `:289` | not modelled | **no** — read only by `select_tx_block` (`tx_search.c:2631`), the INTER var-tx recursion under `av1_pick_recursive_tx_size_type_yrd`. |
| `auto_max_partition_based_on_simple_motion` | 480p / 720p tiers | `speed_features.c:176-180`, `:305-309` | not modelled | **no** — `use_auto_max_partition` is `!frame_is_intra_only && ... && sb_size == BLOCK_128X128` (`partition_strategy.h:193`). |
| `ml_partition_search_breakout_thresh[]` + `_model_index` | `< 720p` / `>= 720p` | `speed_features.c:192-201`, `:219-236` | not modelled | **no** — `av1_ml_predict_breakout` is called under `!frame_is_intra_only` (`partition_search.c:4260`). |
| `ml_early_term_after_part_split_level` | `< 720p` | `speed_features.c:200`, `:207`, `:269` | not modelled | **no** — `av1_ml_early_term_after_split` under `!frame_is_intra_only` (`partition_search.c:4322`). |
| `mv_sf.use_downsampled_sad` | `>= 720p` | `speed_features.c:203-206` | not modelled | **no** — motion search (`mcomp.c:131`). |
| `partition_search_breakout_{dist,rate}_thr` | 720p tiers, speed>=2 | `speed_features.c:244-251`, `:273-286`, `:293-297` | not modelled | **no** — same `!frame_is_intra_only` block. |
| `part_sf.max_intra_bsize` | `< 720p`, speed>=3 | `speed_features.c:283` | not modelled | **no** — speed-gated, and no framesize DISTINCTION below 720p. |

Cross-checked against C in both directions: the port is **not** missing a
threshold among the live fields (`use_square_partition_only_threshold_allintra`
reproduces all four speed tiers and both framesize tiers), and it **is** missing
one among fields it does not model at all — `default_min_partition_size` at 4K.

### Geometry-dependent derivations (no sf involved)

| derivation | condition | libaom | port |
|---|---|---|---|
| forced partitions at the frame edge | `!av1_blk_has_rows_and_cols` | `partition_search.c:3389` | KB-6 chunk series |
| edge partition costs gathered from the FRAME-INIT cdf, not the adapting tile state | edge block | `set_partition_cost_for_edge_blk`, `partition_search.c:3415` | KB-6 CHUNK 3 (`4b8b1f1`) |
| beyond-visible entropy-context tail zeroed | edge txb | `av1_set_entropy_contexts`, `blockd.c:29` | KB-6 `4567e58` |
| distortion clipped to the visible area | edge block | `max_block_units` | KB-6 CHUNK 1-2 |
| no SB-sized coding block (root force-split) | frame smaller than one SB | `partition_search.c:3389` | modelled by `CellCtx::has_full_sb_block` |
| >64 coding blocks: L/U/V 64x64-chunk coefficient interleave | SB128 with a coded >64 leaf | `encodetxb.c:431-472` | KB-1 encoder cross-check |

All four edge paths are **live at speed 0** and are the historically densest bug
region in this repo (KB-6). None of the eight pre-existing contexts has a
partial superblock except 32x32, which is *smaller* than one superblock and
therefore never reaches the multi-SB edge interactions at all.

### The resulting size classes at speed 0 (pinned, printed by the gate)

| class | sb | full SB block | multi (col,row) | partial (x,y) | rect-kill reachable | representative | status |
|---:|---|---|---|---|---|---|---|
| 1 | 64 | yes | (0,0) | (0,0) | no | 64x64 | covered before |
| 2 | 64 | no | (0,0) | (1,1) | no | 32x32 | covered before |
| 3 | 64 | yes | (1,1) | (0,0) | no | 128x128 — **and 512x512** | covered before |
| 4 | 64 | yes | (1,1) | (0,1) | no | 128x96 | **ADDED** |
| 5 | 64 | yes | (1,1) | (1,1) | no | 68x68, 96x96 (196x196 collapses in) | **ADDED** |
| 6 | 64 | yes | (1,1) | (1,1) | no, `default_min_partition_size=BLOCK_8X8` | 2160x2160 | **out of budget** |
| 7 | 128 | no | (0,0) | (1,1) | no | 64x64 SB128 | **ADDED** |
| 8 | 128 | yes | (1,1) | (0,0) | no (>= 480p) | 512x512 SB128 | **pinned open** (finding B) |
| 9 | 128 | yes | (1,1) | (1,1) | no (>= 480p) | 576x576 SB128 | **ADDED** |
| 10 | 128 | yes | (0,0) | (0,0) | **yes** | 128x128 SB128 | **ADDED** |
| 11 | 128 | yes | (1,1) | (1,1) | **yes** | 192x192 SB128 | **ADDED** |

**Eleven classes; the pre-existing array covered three.** The collapse is doing
real work in both directions and both directions are asserted:

* **it collapses** — 512x512 SB64 lands in the same class as 128x128 SB64, so a
  12.6 s/cell 480p SB64 context would buy exactly nothing. That is the honest
  half of "is 2,617 valid": on SB64, going bigger is not a new configuration.
* **it splits** — the same pair does NOT collapse at SB128, because there the
  >= 480p threshold decides whether the rect-kill fires.

`SizeDerived` deliberately does **not** carry the raw
`use_square_partition_only_threshold` value (exposed separately as
`config_perm::sq_only_threshold_allintra`): at speed 0 on intra it is
unobservable except through `rect_kill_reachable`, and carrying it would split
512x512 SB64 away from 128x128 SB64 for a difference no cell could witness.
Likewise the superblock GRID is carried as one-vs-many booleans rather than an
exact count — a third superblock column adds no structure a second one did not.

## What was added, and what each cell buys

| gate | context | class | strength | cells | ms/cell | what it buys |
|---|---|---|---|---:|---:|---|
| `size_t4_part68_s{0,1,2}` | 68x68 SB64 | 5 | **t=4** | 187 | 199 | the KB-6 frame-edge class at FULL knob strength — four edge code paths, each interacting with essentially every knob (which blocks land on the edge is a function of the partition and transform knobs) |
| `size_t2_part96` | 96x96 SB64 | 5 | t=2 | 17 | 439 | the second overhang magnitude (32 px vs 4 px) — which transform footprints the tail-zero clips |
| `size_t2_part128x96` | 128x96 SB64 | 4 | t=2 | 17 | 500 | partial in ONE dimension: above-context and left-context clipping are separate paths |
| `size_t2_sb128_64` | 64x64 SB128 | 7 | t=2 | 17-6 | 170 | frame smaller than an SB128 superblock (root force-split, no 128 block) |
| `size_t2_sb128_128_s{0,1}` | 128x128 SB128 | 10 | t=2 | 17-6 | 1028 | breadth on the class where the rect-kill first becomes reachable |
| `size_ix_sb128_128_s{0,1}` | 128x128 SB128 | 10 | full rect-kill cross | 24-8 | 1028 | the rect-kill interaction set at FULL cross, where it is affordable |
| `size_ix_sb128_192_s{0,1}` | 192x192 SB128 | 11 | rect-kill cross | 16 | 1987 | rect-kill AND frame-edge live at once |
| `size_ix_sb128_576_s{0,1}` | 576x576 SB128 | 9 | rect-kill cross | 16 | ~4000 | **the >= 480p class** — the one framesize speed feature live at speed 0 |
| `size_class_inventory_is_pinned` | — | — | — | 0 | — | the arithmetic above, pinned |
| `size_axis_teeth_are_real` | — | — | — | 0 | — | the rect-kill stays reachable in the added contexts and dead in the old ones |
| `size_axis_open_divergences_pinned` | — | — | — | 5 | — | the two findings, self-promoting |

**Cells: 2,617 -> 2,910** (+293 array/interaction cells, +5 pinned-finding
cells). The `-6` / `-8` entries are the `--max-partition-size=32` rows skipped
on SB128 by finding A.

### Why the expensive contexts get a reduced array, argued from the interaction set

At speed 0 below 2160p the only live size-derived state is
`use_square_partition_only_threshold`, whose sole intra consumer acts on
`partition_rect_allowed` at a block larger than the threshold. An axis can
interact with that only by changing whether rectangular partitions exist
(`--enable-rect-partitions`), whether the over-threshold block is reached at all
(`--max-partition-size`, which force-splits the root below the SB size), or
which rect-derived types are offered (`--enable-ab-partitions` and
`--enable-1to4-partitions`, both gated on `partition_rect_allowed` at
`partition_search.c:5166/5172/5181/5187`). That is
`config_perm::RECT_KILL_INTERACTION_SET`, and every other axis composes with the
kill only *through* those four — a composition already covered at full t=4
strength on the cheap contexts. The same set is run at **full cross** on the
1.0 s/cell SB128 context, so the reduction applied at 2-4 s/cell is
demonstrated sufficient at the same mechanism rather than assumed.

`--min-partition-size` is excluded: it raises the partition floor, never the
root, so it cannot change whether an over-threshold block exists.

### Out of budget, recorded rather than faked

**`default_min_partition_size = BLOCK_8X8` at `is_4k_or_larger`
(`speed_features.c:187-189`) is a genuine unmodelled port arm**
(`config_perm::PORT_GAP_DEFAULT_MIN_PARTITION_SIZE`;
`aom-encode/src/speed_features.rs:471`). It is class 6 and it is not gated: a
480x480 speed-0 cell already costs 12.6 s, a 2160x2160 one is ~20x that per
cell, so even a single cell would blow the whole suite's budget several times
over. Gating it needs an `--ignored` deep tier and a decision about CI wall
time; it is NOT closed by this landing. The speed>=6 1080p arm (`:311-313`) is
subsumed by the port's unconditional speed-6 assignment, so only the 4K arm is a
real divergence, and only below speed 6.

The other structural limit is **speed**: this array is speed-0 everywhere, so
the four framesize x speed interactions (`prune_tx_type_using_stats` at 480p,
`use_square_partition_only_threshold`'s 720p tiers, `max_intra_bsize`,
`prune_tx_size_level`) are outside it by construction.
`size_class_inventory_is_pinned` pins `size_derived(.., {0,2,4})` so that raising
the array's speed cannot silently leave them unexercised.

## Findings

### A. `--sb-size=128` x `--max-partition-size=32` trips a port assertion C contradicts

C restores the partition context CONDITIONALLY at the end of the SPLIT stage:

> `if (bsize <= x->sb_enc.max_partition_size || bsize == cm->seq_params->sb_size)`
> `  av1_restore_context(x, x_ctx, mi_row, mi_col, bsize, av1_num_planes(cm));`
> — `partition_search.c:4646`

The port restores UNCONDITIONALLY and encodes C's condition as a `debug_assert!`
instead (`aom-encode/src/partition_pick.rs:3055-3057`, commented "always true
here"). It is not always true: it fails when a block size sits strictly between
the max-partition cap and the superblock size. At SB64 that window is empty for
every legal cap, which is why no pre-existing context saw it; at SB128 with a
32 px cap `bsize == BLOCK_64X64` satisfies neither clause. In a debug-assertions
build the port panics; without them it performs a restore C skips — a silent
state divergence. Pinned and self-promoting in
`size_axis_open_divergences_pinned`; the affected rows are skipped by
`SizeCtx::skip_reason` with that citation. **No encoder change was made** (this
agent does not own `crates/aom-encode/src`).

### B. Three open near-ties on >= 480p SB128 monochrome cq63 — surfaced by size, NOT attributable to a size class

See the doc comment on `size_axis_open_divergences_pinned` for the full table.
The important part is the methodology: "480x480 diverges, 448x448 and 512x512 do
not" reads like a real class property until **576x576 — the same size class as
480x480 — comes out exact on the identical knob rows**, and 640x640 does the
same for 512x512. A property that holds for one member of an equivalence class
and fails for another is not a property of the class. These are per-cell RD
near-ties (KB-10/KB-12 signature), surfaced because the size axis encoded this
content at sizes the harness had never reached. They are pinned, not gated, and
the gated >= 480p context uses a clean class-mate so no size gate rests on a
near-tie.

### C. The "SB64 only" scope note was stale

`cell_ctx` carried "`--sb-size=128` encode is unstarted; HANDOFF-TOGGLES.md".
`crates/aom-bench/tests/sb128_e2e.rs` proves SB128 encode byte-exact vs real
aomenc, including a coded 128-level leaf. SB128 is where the >= 480p threshold
stops being inert, so that stale note was hiding the single most consequential
size class.

## Teeth

The size-gated derivation under test is `use_square_partition_only_threshold`'s
`>= 480p` arm (`speed_features.c:175-183`; port `partition_pick.rs:2450`).
Perturbation: replace `partition_pick.rs:2451` with `let mut t: usize = 12;`
(the `>= 480p` arm dropped).

**The added `>= 480p` SB128 cell FAILS:**

> `TEETH 480m_cq63_sb128_stock  exact=false port 874B real 854B`

**Every control stays GREEN** — the two knob rows that make the kill moot, the
sub-480p SB128 contexts, and both SB64 sizes:

> `TEETH 480m_cq63_sb128_rect0 exact=true port 864B real 864B`
> `TEETH 480m_cq63_sb128_maxp64 exact=true port 874B real 874B`
> `TEETH 128_cq32_sb128_stock exact=true port 1968B real 1968B`
> `TEETH 128_cq63_sb128_stock exact=true port 70B real 70B`
> `TEETH 64_cq32_sb64_stock exact=true port 415B real 415B`
> `TEETH 128_cq32_sb64_stock exact=true port 1947B real 1947B`

**And the whole pre-existing 2,617-cell gate is untouched by the same
perturbation:**

> `test result: ok. 32 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 32.30s`

2,617 cells that cannot see a broken `>= 480p` threshold, against an added cell
that fails on it, is the asymmetry — the size gap was real. The perturbation was
reverted; `git diff crates/aom-encode/` is clean.

## Chunk 2 — what a follow-up should do

1. **Close finding A.** Make `restore_context` conditional exactly as
   `partition_search.c:4646` is, delete the `debug_assert!`, remove
   `SizeCtx::skip_reason`'s entry and let the SB128 contexts run
   `--max-partition-size=32` at full strength. Needs `crates/aom-encode/src`
   ownership.
2. **Root-cause finding B** with the decode-both / sibling-C recipe. The 576/640
   class-mate controls narrow it to an RD near-tie at specific content
   statistics, which is the KB-10/KB-12 localisation shape.
3. **Gate class 6 (4K) in an `--ignored` deep tier.** One 2160x2160 stock cell
   plus the `--min-partition-size` interaction set would close the only
   unmodelled framesize arm; budget it as a nightly, not a default-tier gate.
4. **Cross the size axis with speed.** Everything here is speed 0. The
   framesize x speed interactions are pinned as arithmetic in
   `size_class_inventory_is_pinned` but not encoded. The cheapest real cell is
   `tx_stats_prune_e2e.rs`'s 512x512 cpu-2 shape.
5. **Give the >= 480p contexts more breadth.** They currently run only the
   rect-kill interaction set. If the budget grows, t=2 over all 21 axes at
   576x576 is ~17 x 4 s = 68 s CPU.
