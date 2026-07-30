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
