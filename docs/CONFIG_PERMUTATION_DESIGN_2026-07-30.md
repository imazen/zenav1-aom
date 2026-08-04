# Config-permutation gate — design, evidence, and honest coverage arithmetic

> **DATED RECORD, 2026-07-30 — the DESIGN is live, the OPEN-CELL findings are not.**
> (Header added 2026-08-03; the body is deliberately not rewritten — it is the evidence for
> the landing, and the reasoning behind the collapse engine is why the gate is shaped as it
> is.) Two things have since changed:
>
> - **§"Root cause of the `dtxo` class divergence (found 2026-07-30, NOT fixed)" IS NOW
>   FIXED** — that is KB-17 (`use_screen_content_tools` hardcoded `false`), closed the same
>   day. `mono_vector_open_divergences_pinned` has been inverted from an open-divergence pin
>   into the **KB-17 regression gate**: all `dtxo` cells are byte-identical and asserted so.
>   `dtxo` is no longer content-sensitive.
> - **The gate has ZERO pinned cells since 2026-08-02.** Both `SPEED_OPEN_SINGLETONS` and
>   `SPEED_OPEN_COMBINATIONS` in `crates/aom-bench/tests/config_permutations.rs` are empty:
>   every single-axis perturbation and every covering-array combination is byte-identical to
>   real `aomenc` at every speed 0..9. The specific "2 of 93 covering-array cells" and the
>   speed-4/speed-8 combination rows quoted below closed via KB-21 (three roots) and KB-12
>   (`aom_hadamard_lp_8x8`'s dropped trailing transpose).
>
> The gate file's own doc comments are the live status. This document is the design record.

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

---

# The SPEED axis (added 2026-07-30, same day)

The three sections above run at **one speed**. `Ctx::cell`, `Content::cell` and
`SizeCtx::cell` all call `EncodeCell::real_content(.., 0)`, so every one of the
2,910 cells is `--cpu-used=0`. PARITY.md §A separately gates `--cpu-used 0..9`,
but each speed on its **own** grid with **stock** knobs. The knob axes and the
speed axis had never been crossed. This section is that crossing.

## Short answer

**2,910 is valid at speed 0 and says nothing above it, and the gap is not
uniform — it is concentrated in two places nobody would have guessed.**

* Replaying the array across the speed range found **8 knob COMBINATIONS that
  diverge from real aomenc at a speed and nowhere else** — 3 at `--cpu-used=4`,
  5 at `--cpu-used=8`. Each row's individual levels are byte-exact alone at that
  speed, and the whole row is byte-exact at speed 0.
* The **fragile band is speeds 4-5, not the nonrd speeds**. On three of five
  contents the **stock** encode — every knob at its default — diverges at
  `--cpu-used` 4 and 5 while matching at 0-3 and 6-9. Speeds 7-9 (`VAR_BASED_
  PARTITION`, nonrd PICKMODE) are the *cleanest* levels above 0.
* **bd10 is open at almost every speed.** `av1-1-b10-00-quantizer-00` is
  byte-exact at speed 0 (a gated context of the main array) and at speed 7, and
  diverges at 1, 2, 3, 4, 5, 6 — and at 8/9 it does not diverge, it **panics**:
  `nonrd_pick_intra_mode` (`aom-encode/src/nonrd_pickmode.rs:601`) asserts
  `env.bd == 8`, "HANDOFF: hbd estimate arm (av1_quantize_fp + fp scans) not
  ported". The KB-12 nonrd path is bd8-only.
* Two axes are genuinely **speed-sensitive in their meaning**, not just their
  strength: `txss` (`--enable-tx-size-search`) stops being a configuration at
  all from speed 8, and with it the matrix's single C-forbidden pair lapses.
* And the coverage a cell can buy **decays monotonically with speed**: at
  `--cpu-used=9` only **4 of 24** reachable axis levels still change real
  aomenc's own output, so an unreduced t=4 array there would be ~85% vacuous.

Plus one harness defect found and fixed en route, without which none of speeds
6-9 was reachable at all (below).

## Which axes move with speed — measured

`run_speed_matrix` replays **every singleton axis level at every speed** on the
primary context (bd8 4:2:0 photographic 64x64 cq32, `av1-1-b8-01-size-64x64` —
the same cell as the `64cq32` context of the main array, so anything that moves
is attributable to speed alone). 264 cells, each asserting byte-identity.

Unlike the content axis — where the answer was "1 of 21 axes, one mechanism" —
speed does not sort the axes into sensitive and insensitive. It changes **how
much of the configuration reaches the encoder at all**:

| `--cpu-used` | axis levels that still move real aomenc | of reachable | divergent |
|---:|---:|---:|---:|
| 0 | 20 | 26 | 0 |
| 1 | 20 | 26 | 0 |
| 2 | 20 | 26 | 0 |
| 3 | 20 | 26 | 0 |
| 4 | 17 | 26 | **3** (`dir0`, `rtx0`, `flip0`) |
| 5 | 18 | 26 | **1** (`minp16`) |
| 6 | 14 | 26 | 0 |
| 7 | 13 | 25 | 0 |
| 8 | 12 | 24 | **2** (`rtxs1`, `trel2`) |
| 9 | **4** | 24 | 0 |

(`SPEED_LIVE_LEVELS` pins the exact sets, not just the counts, so an axis that
silently goes inert at a speed fails the test rather than quietly turning that
speed's array into vacuous cells. The reachable count drops at speed 7/8 because
`txss=0` becomes unreachable — see below.)

The four survivors at speed 9 are `fint0`, `rtxs1`, `cdf0`, `trel1`: two
header bits, the CDF-update switch, and trellis. Every partition and intra-mode
knob is dead — the nonrd pickmode does not consult them.

**The strengths in `SPEED_CONTEXTS` are set from this table, not uniformly**:
t=4 at speed 2, t=3 at 4 / 6 / 7 / 8, t=2 at 0 / 1 / 3 / 5 / 9.

### The speed-gated derivations, enumerated

`speed_class_inventory_is_pinned` computes and pins the ALLINTRA speed-feature
class partition — which `--cpu-used` steps move the resolved `SpeedFeatures` at
all. Measured: **eight classes over ten speeds**, `{0} {1} {2} {3} {4} {5} {6}
{7,8,9}` (identical at bd10 and under screen-content detection, also asserted).
Per-step field deltas: 18 fields at 0→1, 5 at 1→2, 6 at 2→3, 12 at 3→4, **1** at
4→5 (`multi_winner_mode_type` only), 21 at 5→6, 3 at 6→7, **0** at 7→8 and 8→9.

**That partition is NOT a valid collapse, and the refutation is the useful
part.** The encoder also branches on the raw `PickFrameCfg::speed`, at
thresholds no `SpeedFeatures` field represents —
`pack.rs:1474` (`speed >= 7`, `VAR_BASED_PARTITION`), `pack.rs:1791` /
`partition_pick.rs:4569` (`speed >= 8`, `av1_nonrd_use_partition`),
`pack.rs:1685`/`:2117` (`speed >= 9`, `cost_upd_off`),
`partition_pick.rs:4772` (`hybrid_intra_pickmode`), and
`partition_pick.rs:4854-4856` (three `speed >= 9` intra prunes). So the test
asserts the refutation **against the oracle** rather than by inspection: real
aomenc's own payload must differ at `--cpu-used` 7 vs 8 vs 9 on the same cell.
It does (516 / 515 / 497 B). A collapse the oracle contradicts is not a
collapse, so every speed keeps its own context.

The four derivations gated on speed **and** framesize together are enumerated
with citations in `cp::SPEED_X_FRAMESIZE_DERIVATIONS`; only one is live on this
harness's all-intra KEY path, and it is the next section.

## SPEED x SIZE — the crossing the size axis could not reach

The SIZE section's chunk-2 item 4 reads *"Cross the size axis with speed.
Everything here is speed 0."* That is closed here, on the exact cell it names.

`prune_tx_type_using_stats` needs **both** `is_480p_or_larger` **and**
`speed >= 2` (level 1) / `speed >= 4` (level 2) — libaom
`speed_features.c:261`, `:299`. It is therefore 0 on all 2,910 speed-0 cells
(in the port *and* in real aomenc, which is why the size section correctly
reported nothing unexercised there) and 0 on every sub-480p cell at any speed.
**512x512 monochrome at `--cpu-used=2` is the smallest cell where it is 1**, and
`speed_size_txstats_*` runs a t=2 array there: 17 rows, all byte-identical to
real aomenc.

**The field is proved non-zero, not assumed.** `ToggleKnobs::disable_tx_stats_
prune` forces the *port's* `sf.prune_tx_type_using_stats` to 0 while the C side
(driven by `--cpu-used` alone) keeps pruning, so "port-without != port-with" is
a direct measurement that the field is non-zero and load-bearing on this cell.
Both directions are gated, which is what makes the witness meaningful:

* `stock` and `trel1` **must** witness — trellis does not touch the tx-TYPE
  candidate set, so the prune still has an IDTX/FLIPADST winner to remove;
* `flip0` (`--enable-flip-idtx=0`) **must not** — the knob masks the
  FLIPADST/IDTX family out of the ext-tx set (`get_tx_mask`'s `DCT_ADST_TX_MASK`
  arm) *before* the stats prune runs, so forcing the prune off cannot change a
  byte. A witness there would mean the harness knob perturbs something other
  than the prune and every positive witness would be suspect.

Measured over 16 singleton rows: **11 witness**. The five that do not are
`rect0` and `cdf0` (the prune fires but does not flip the winner) plus the three
that structurally disarm it — `flip0`, `dtxo1` (one tx type) and `dir0` (no
directional mode left to carry a non-DCT default).

**Level 2 (speed >= 4) is NOT gated**, and the reason is recorded rather than
papered over: the same 512x512 cell's *stock* encode already diverges at
`--cpu-used=4` (125,629 vs 125,630 B) — the pre-existing cpu-4 near-tie
`tx_stats_prune_e2e.rs` documents. Gating level 2 would gate a divergence.

## Findings

### A. The harness could not encode ANY speed >= 6 cell — root-caused and FIXED

Before this section, `EncodeCell::port_encode_with` diverged from real aomenc at
`--cpu-used >= 6` on **every** content tried (photographic, monochrome, bd10 and
the synthetic diag/vgrad content the `aom-encode` speed-6..9 gates use), with
stock knobs. Since those `aom-encode` gates pass on that same content at those
same speeds, the defect was in the bench harness's config threading, not the
encoder.

Root cause, one line: `aom-bench/src/lib.rs:1712` ran the loop-filter **search**
at every speed. C's `lpf_sf.lpf_pick` is `LPF_PICK_FROM_FULL_IMAGE` (DUAL) at
allintra speed 0-3, `..._NON_DUAL` at 4/5 (`speed_features.c:496`), and the
closed-form `LPF_PICK_FROM_Q` at **speed >= 6** (`:559`) — no search at all, the
level is a fit on the AC quantizer. The `aom-encode` e2e gate has carried that
arm since the speed-6 landing (`pick_filter_level_from_q`, oracle-validated by
`speed6_prep_lf_from_q_matches_real_aomenc`); the bench harness never got it.

The evidence that it was header-only: at cpu-6 on 128x128 diag the port's
payload was 1,297 B against C's 1,297 B with the **first difference at payload
byte 2** and the tile payload already byte-identical — the deblock-level field,
and (on the 64x64 cell) the two extra bytes C writes because a non-zero level
gates the loop-filter delta block.

Fixed in the same landing; speeds 6-9 go from "diverges on everything" to
byte-exact on every default-tier context.

### B. Eight knob COMBINATIONS diverge at a speed and nowhere else (pinned open)

`SPEED_OPEN_COMBINATIONS`: 3 rows of 63 at `--cpu-used=4`, 5 of 63 at
`--cpu-used=8`. Speeds 0, 1, 2, 3, 5, 6, 7 and 9 are clean at their gated
strength — **including the full 187-row t=4 array at speed 2**. The signature is
uniform and is the KB-10/KB-12 "cheaper RD decision" near-tie: port payloads 0-4
bytes short of C's.

The two speeds are exactly where the search **structure** changes rather than
its thresholds — speed 4 is the winner-mode / multi-winner tier
(`multi_winner_mode_type=2`, `prune_chroma_modes_using_luma_winner`,
`fast_intra_tx_type_search=2`) and speed 8 is nonrd PICKMODE. Every speed-8 row
carries `dir0` or `dtxo1` or both, i.e. a narrowed luma tx-type/mode set feeding
the nonrd intra pickmode.

> **Update 2026-08-02 — that lead was right and both lists are now EMPTY.** The
> cpu-4 rows closed 2026-07-31 (KB-21 root #2); the cpu-8 rows and the two
> remaining `SPEED_OPEN_SINGLETONS` (`rtxs1`, `trel2`) closed 2026-08-02 with
> KB-12's root: `hadamard_lp_8x8` dropped the trailing transpose at
> `aom_dsp/avg.c:232-236`, so the nonrd estimate arm's `eob` — its only
> order-sensitive output — drifted, and narrowing the mode/tx-type set changes
> how often that drift is decisive. The "cheaper RD decision near-tie" signature
> named here is what a small unmodelled RATE TERM looks like, not evidence of a
> tie. Emptying the singleton list also broadens the speed-8 covering array
> (`remap_open_levels` no longer folds those two levels back to default); the
> broadened array is 63/63 exact.

That is a lead, not a root cause; **no encoder change
was made** (this section does not own `crates/aom-encode/src`).

Six singleton axis levels diverge at a speed too (`SPEED_OPEN_SINGLETONS`:
`dir0`/`rtx0`/`flip0` at 4, `minp16` at 5, `rtxs1`/`trel2` at 8) and are pinned
the same way. Both pins are self-promoting in both directions, and the pinned
levels are remapped to their defaults inside that speed's array —
level-granular, so pinning `trel=2` open at speed 8 does not cost the array its
`trel=1` coverage.

### C. `--enable-tx-size-search=0` stops being a configuration at speed >= 8

```c
if (!oxcf->txfm_cfg.enable_tx_size_search && sf->rt_sf.use_nonrd_pick_mode == 0)
  sf->winner_mode_sf.tx_size_search_level = 3;
```
— libaom `av1/encoder/speed_features.c:2726-2729`

and `set_allintra_speed_features_framesize_independent` sets
`rt_sf.use_nonrd_pick_mode = 1` at `speed >= 8` (`:579`). From speed 8 the CLI
knob never reaches `tx_size_search_level`, `select_tx_mode` does not return
`TX_MODE_LARGEST`, and the harness's

```rust
assert!(knobs.enable_tx_size_search || !p.tx_mode_select)   // lib.rs:1119-1123
```

**panics on a stream real aomenc happily produced**. Measured: at `--cpu-used=8`
the header codes `TX_MODE_SELECT` with the knob off; at `--cpu-used=9` on the
same cell it happens to code LARGEST (C's post-hoc `txb_split_count == 0`
demotion), so the panic is *data*-dependent from speed 8 up, not a clean
threshold.

Two consequences, both modelled: `cp::axis_level_dead_at_speed` removes the level
from the matrix and the arrays at speed >= 8 (nothing is lost — it is inert
there), and `cp::illegal_reason_at_speed` records that the matrix's **single
C-forbidden pair lapses**: `txss=0 x tx64=0` exists because
`assert(enable_tx64 || tx_search_type != USE_LARGESTALL)`
(`encodeframe.c:2461`) trips when the CLI forces `USE_LARGESTALL`, and at speed
>= 8 it no longer does. The matrix does not exploit that, but the model must not
claim an exclusion libaom does not have. Pinned by
`speed_txss_nonrd_lapse_is_pinned`.

### D. The speed ENVELOPE, mapped — the fragile band is 4-5, and bd10 is open

The gated speed contexts all ride one content, chosen because its stock encode
is byte-exact at every speed. That is right for isolating the knob x speed
interaction and would be dishonest as the only statement, so
`speed_envelope_stock_map_is_pinned` maps four more contents and pins the result
(the full grid is the committed TSV):

| content | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `sz64` (the gated one) | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| `q00_64` | ok | ok | ok | ok | **X** | **X** | ok | ok | ok | ok |
| `q00_mono64` | ok | ok | ok | ok | **X** | **X** | ok | ok | ok | ok |
| `q00_128` | ok | ok | ok | ok | **X** | **X** | ok | ok | ok | ok |
| `b10_64` | ok | **X** | **X** | **X** | **X** | **X** | **X** | ok | **panic** | **panic** |

`X` = the STOCK encode (every knob default) is not byte-identical to real
aomenc. The bd10 speed-8/9 entries are not near-ties at all: they are the
unported hbd nonrd arm (`nonrd_pickmode.rs:601`).

This is why no bd10 speed context is gated — it would gate a divergence — and
why the per-speed arrays ride the one content whose envelope is intact at all
ten speeds.

## What was added, and what each cell buys

| gate | speed | strength | cells | ms/cell | what it buys |
|---|---|---|---:|---:|---|
| `speed_sensitivity_s{0,1,2}` | 0-9 | 26 singletons | 264 | 2-114 | the sensitivity table above; every axis level x every speed |
| `combinations_t4_speed2_s{0,1,2}` | 2 | **t=4** | 187 | 48 | every 4-way interaction at the first ML/tx-stats tier |
| `combinations_t3_speed{4,6,7,8}` | 4/6/7/8 | t=3 | 4 x 63 | 3-30 | the four structure changes: winner-mode, last-RD, VAR_BASED, nonrd |
| `combinations_t2_speed{0,1,3,5,9}` | 0/1/3/5/9 | t=2 | 5 x 17 | 2-114 | pairwise where the sf delta is small; speed 0 is the runner-agrees control |
| `speed_size_txstats_s{0,1,2}` | 2 | t=2 @ 512x512 | 17 | 780 | **the SPEED x SIZE closure** + the prune liveness witness |
| `speed_envelope_stock_map_is_pinned` | 0-9 | stock | 50 | — | finding D |
| `speed_txss_nonrd_lapse_is_pinned` | 6-9 | 1 knob | 4 | — | finding C |
| `speed_axis_teeth_are_real` | 0-9 | stock | 10 | — | the envelope control + the no-duplicate-context invariants |
| `speed_class_inventory_is_pinned` | — | — | 3 | — | the arithmetic + the oracle refutation of the collapse |
| `speed_axis_budget_is_accounted` | — | — | 0 | — | the budget, pinned |

**Cells: 2,910 -> 3,647** (+737). Nothing was thinned to hit the budget: every
speed runs the full singleton matrix and an array, and what is in the
`--ignored` deep tier is the *redundant* part — the four secondary contents'
per-speed matrices, whose value is the envelope map that the default tier
already pins in summary form.

## Budget

| tier | wall |
|---|---|
| pre-existing gate (2,910 cells, 44 tests) | 47.7 s |
| + the speed axis (14 tests, 737 cells) -> 86 tests | **58.9-65.1 s** |
| decoder gate, unchanged | 3.3 s |
| deep tier (`--ignored`): `speed_axis_evidence_sweep`, 1,342 cells | 106 s, opt-in |

(12-core M4, `cargo test --profile test-fast -j 4`, machine shared with two
concurrent agents — upper bounds, and the 58.9/65.1 spread is that sharing.)
Total default tier ~62-68 s against the 120 s ceiling.

## Teeth — the asymmetry, twice

Two perturbations of **speed-gated** paths, each applied, run, and reverted
(`git diff` on `crates/*/src` clean afterwards, verified).

**1. The framesize x speed derivation.** `aom-bench/src/lib.rs`, the
`prune_tx_type_using_stats` gate, `speed >= 2` -> `speed >= 3`, so the port stops
pruning at cpu-2 while real aomenc still does.

Result: **84 passed, 2 failed** — and both failures are new speed x size cells:

> `txstats512s2: 1 of 6 SPEED x SIZE cells are NOT byte-identical to real aomenc
> — a knob combination diverges where the framesize-gated speed feature
> prune_tx_type_using_stats is LIVE. Offenders: txstats512s2_stock
> (port 126058B vs C 126057B)`

> `txstats512s2 \`trel1\`: the witness row itself is not byte-identical to real
> aomenc`

Every one of the 2,910 pre-existing cells stayed GREEN — all eight
`combinations_t4_*` contexts, all three quality-ladder shards, all eleven `size_*`
gates, all twelve content gates, both collapse proofs, the per-axis liveness
test and the arithmetic.

**2. The speed >= 6 loop-filter derivation** (finding A, reverted to its
pre-fix state): drop the `LPF_PICK_FROM_Q` arm.

Result: **78 passed, 8 failed** — again all eight are new speed-section tests
(`speed_sensitivity_s1`/`s2`, `combinations_t3_speed{6,7,8}`,
`combinations_t2_speed9`, `speed_axis_teeth_are_real`,
`speed_envelope_stock_map_is_pinned`):

> `s6: the primary speed context's stock encode is not byte-identical to real
> aomenc`

> `the SPEED ENVELOPE moved. ... ["sz64 cpu-used=6: stock encode is now
> \`diverge\` (pinned \`ok\`)", "sz64 cpu-used=7: ...", ... "b10_64 cpu-used=7:
> stock encode is now \`diverge\` (pinned \`ok\`)"]`

and the whole 2,910-cell speed-0 gate untouched again.

2,910 cells that cannot see either a broken framesize x speed threshold or a
broken speed >= 6 loop-filter derivation, against added cells that fail on both,
is the asymmetry. Both perturbations were reverted.

## Chunk 2 — what a follow-up should do

1. **Root-cause the speed-4/5 stock divergences** (finding D). Three of five
   contexts diverge with *every knob at its default* at cpu-4 and cpu-5, and the
   sf delta from 4 to 5 is a single field (`multi_winner_mode_type` 2 -> 1), so
   the winner-mode machinery is the whole suspect list. Decode-both / sibling-C
   dump per KB-2/KB-6/KB-7. Closing it promotes three contexts into the gated
   set and probably closes several of the eight combination rows with it.
2. **Port the hbd nonrd arm** (`nonrd_pickmode.rs:601`, `av1_quantize_fp` + fp
   scans). Until it lands, bd10 at speed >= 8 is not a divergence but an
   unimplemented path, and no bd10 x speed>=8 cell can exist.
3. **Widen speed x size.** Only `prune_tx_type_using_stats` level 1 is gated.
   Level 2 (speed >= 4) is blocked behind the cpu-4 near-tie in item 1; the
   720p tiers of `use_square_partition_only_threshold` need an SB128 >= 720p
   context at speed >= 1, which the size section's cost table puts at ~4 s/cell —
   an `--ignored` deep-tier candidate, not a default cell.
4. **Cross speed with content.** The screen-content class (`dtxo`, KB-17) is
   measured at speed 0 only, and `get_tx_mask`'s screen arm interacts with the
   tx-type prunes that speed turns on. `scr_ibc_b8` x cpu-{2,6} is ~17 cells.
5. **Fix the harness's TX_MODE_LARGEST assertion** (finding C) so `txss=0`
   becomes reachable at speed >= 8 instead of pinned dead — it needs the
   assertion made conditional on `speed < 8`, which is `aom-bench/src` work plus
   a re-pin of `axis_level_dead_at_speed`.
