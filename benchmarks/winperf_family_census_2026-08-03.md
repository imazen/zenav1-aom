# Whole tool families were invisible to every benchmark we have (2026-08-03)

[`winperf_content_census_2026-08-03.md`](winperf_content_census_2026-08-03.md)
§4 ended with a list of what the harness still could not see:

> **filter-intra, palette, intraBC, CFL/chroma and 8x8 leaves are at or near
> zero on every source — INCLUDING the study photograph.**

A list is a document. This lands the same statement as an **instrument, a
corpus measurement, a content and a gate**, and in the course of doing so the
list turns out to have been conflating three very different problems.

**The headline:** of the five families named, exactly **one** was a content
problem. Filter-intra is a **speed** zero (`--cpu-used 6` never calls the
search; the same content reads 10.46 % of leaves at cpu-used 5). Palette and
intraBC are **knob** zeros (`--enable-palette` / `--enable-intrabc`, both
default off, plus a header bit real `aomenc` sets from its own detection). CFL
and the chroma path were **never censused at all** — the counters were on the
luma predictor. And the differential corpus, which nobody had censused, reaches
filter-intra at **21-31 % of leaves**: the byte gates were never blind to it.

Provenance, boxes, exact commands, and what is *not* measured:
[`winperf_family_census_2026-08-03.meta`](winperf_family_census_2026-08-03.meta).
Data: `.photo.tsv`, `.screen.tsv`, `.diffcorpus.tsv`, `.speed.tsv`, `.fit.tsv`.

---

## 1. The instrument

`aom_dsp::census` gained three things (all still behind the default-OFF `census`
feature; a default build constructs none of them because `census::enabled()` is
a `const fn` over `cfg!`):

* **`census::Leaf`** replaces the two-scalar coded-leaf hook. The bitstream
  writer already knows a leaf's filter-intra flag and mode, its Y and UV palette
  sizes, its `use_intrabc`, its UV mode (so `UV_CFL_PRED` is a **count**, not an
  inference), its signalled `tx_size`, both angle deltas, `skip_txfm`,
  `is_inter` and whether it is a chroma reference. Counting them where they are
  *written* keeps the existing "decision, not search visit" discipline.
* **`census::note_plane_intra_pred`** tags each `predict_intra_high` call with
  the plane its ENCODER call site knows it to be. `predict_intra_high` is a
  published `aom_dsp` entry point and gains no argument; the split is an
  additive annotation at each of the eight encoder call sites. **The census
  asserts `plane_total() == intra_total_calls()`** — the DSP-side hook cannot be
  missed, the annotations can, so the two totals agreeing is what proves none
  was (playbook §2).
* **`census::note_cfl_predict`** counts the CFL predictor. CFL does not route
  through `predict_intra_high`, so no intra-prediction count could ever have
  included it.

`Counts::since` now destructures with **no `..`** (playbook §8): adding a
counter breaks the build until its author says how it subtracts, and
`since_subtracts_every_field` is the runtime half of the same guard.

The tool grew sources and options to match: `scr:<path>:<w>x<h>` (bootstrap
through `c_encode_screen`, so the frame header can carry
`allow_screen_content_tools`), `real:<vector>[:<w>x<h>+x+y]` (a conformance
vector decoded back to pixels — literally the differential corpus),
`--speed N`, `--cq N` and `--knobs palette,intrabc`. It also reports
`allow_screen_content_tools` per source, because without that bit a
`palette: 0.00 %` row cannot distinguish "no palette in this content" from "the
tool was never legal here".

---

## 2. The corpus x family census

### 2a. Photographic sources, DEFAULT knobs, cpu-used 6 (`.photo.tsv`)

Percent of the named denominator. `photograph` is the dev box's 1 MP study
image; `clic` and `cid22` are native centre crops from `codec-corpus`.

| family | denom | photograph | `photo` | `detail` | `smooth` | photocrop | clic | cid22 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| filter-intra | leaves | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| palette Y | leaves | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| intraBC | leaves | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **CFL** | chroma-ref leaves | **4.59** | 0.07 | **0.00** | 0.29 | 8.02 | **23.02** | 1.77 |
| CFL predictor | predicted px | 8.80 | 8.15 | 8.59 | 10.53 | 10.23 | 11.57 | 5.74 |
| **chroma prediction** | pred calls | **32.63** | 36.67 | 31.07 | 47.15 | 32.26 | 28.63 | 28.40 |
| directional | predicted px | 20.78 | 17.92 | **0.15** | 13.19 | 21.44 | 29.76 | 41.62 |
| nonzero angle delta | leaves | 11.91 | 11.00 | **0.00** | 2.54 | 11.41 | 15.79 | 19.49 |
| rect leaves | leaves | 16.00 | 6.42 | **0.00** | **0.00** | 6.50 | 42.20 | 35.71 |
| leaves ≤ 8 px | leaves | 16.50 | 6.42 | **0.00** | **0.00** | 6.78 | 83.62 | 72.23 |
| 4-pt fwd tx | fwd tx | 16.65 | 7.11 | 0.48 | **0.00** | 8.68 | 57.41 | 47.36 |
| non-DCT fwd tx | fwd tx | 35.37 | 33.48 | 44.50 | 8.27 | 36.75 | 39.24 | 31.38 |
| coded bytes | | 4 458 | 5 301 | 8 734 | 2 302 | 6 153 | 43 179 | 2 857 |

Three readings:

* **CFL is content-gated, and the fix is cheap.** It is reachable — 4.59 % of
  chroma-reference leaves on the photograph, **23.02 % on a CLIC image** — and
  near-zero only on the synthetics (0.00-0.29 %). Nothing is turning it off;
  the value-noise contents simply have no luma-chroma correlation for
  chroma-from-luma to exploit. A CFL lever should be read on a real photograph
  or on `clic`, not on any winperf content, and that costs a `yuv:` argument
  rather than a new generator.
* **The chroma path is a third of the intra predictor** (28-47 % of calls) and
  was **entirely uncounted** before this landing. Any "intra predictor" number
  taken from the old census was a luma+chroma total nobody had split.
* **`photo` is light on the small-block families**: 6.42 % rect leaves against
  the photograph's 16.00 %, 7.11 % 4-pt transforms against 16.65 %. It is not
  blind (as `detail` and `smooth` are, at 0.00), but a lever priced on 4x4
  transforms is measured ~2.3x light on it.

### 2b. Screen sources, SCREEN bootstrap + both screen knobs (`.screen.tsv`)

The ten `codec-corpus/gb82-sc` screenshots at 512x384 native centre crops.
**These numbers describe a NON-DEFAULT encoder**: `--enable-palette` and
`--enable-intrabc` are both default-off, and both additionally require
`allow_screen_content_tools`, which real `aomenc` signals from its own
detection (the `scdet` row — nothing here forces it).

| family | codec_wiki | gui | terminal | imessage | gmessages | windows | imac_g3 | windows95 | graph |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scdet` | 1 | 1 | 1 | 1 | **0** | 1 | 1 | 1 | 1 |
| palette Y (% leaves) | 22.73 | 6.26 | 16.10 | 16.15 | 0.00 | 21.68 | 12.80 | **32.95** | 9.36 |
| palette UV | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 | 0.52 | 0.00 |
| intraBC | 0.25 | 5.12 | 47.08 | 12.89 | 0.00 | 38.55 | **58.82** | 28.43 | 5.86 |
| CFL (% chroma-ref) | 1.79 | 1.22 | 6.75 | 0.00 | 0.00 | 6.73 | 2.96 | 0.93 | 5.96 |
| leaves ≤ 8 px | 41.92 | 71.79 | 86.28 | 63.11 | 0.00 | 88.75 | 86.14 | 90.78 | 73.46 |
| 4-pt fwd tx | 49.01 | 62.03 | 56.24 | 50.22 | 0.00 | 58.97 | 58.97 | 59.35 | 63.94 |
| filter-intra | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| leaves | 396 | 879 | 1 385 | 675 | 192 | 1 564 | 1 695 | 1 551 | 972 |
| census wall | 3.4 s | 11.2 s | 19.1 s | 4.4 s | 0.0 s | 23.2 s | 18.1 s | 33.7 s | 12.5 s |

* **Palette and intraBC are richly reachable** on real screen content once both
  gates are open — up to 32.95 % and 58.82 % of leaves. Neither is reachable at
  all on any photographic source, at any knob setting, because detection does
  not fire there.
* **`gmessages` is the control that makes the row non-vacuous**: aomenc's own
  detection declined it (`scdet 0`, and it codes 47 bytes — the crop is
  near-blank), and with the bit absent both tools read exactly 0.00 despite
  both knobs being on. That is the header gate doing its job, measured rather
  than assumed.
* **`gb82-sc/imac_dark` has no row.** Its intraBC displacement search does not
  terminate in a workable time — over 20 minutes at 1024x1024 and over 10
  minutes at 512x384, on the same binary where `codec_wiki` takes 47 s / 2.8 s.
  A ~200-400x content-dependent spread in the intraBC search is a real property
  of the port and is recorded here as an open observation, not worked around.
  It is also why the coverage gate runs its screen row at 512x384 rather
  than at 1 MP.
* One 1 MP screen row exists for comparison: `codec_wiki` at 1024x1024 censuses
  at 2 060 leaves, **262 palette (12.7 %)**, **134 intraBC (6.5 %)**, 273 CFL
  leaves, 4 294 coded bytes, 47.2 s. Note intraBC's share **grows with frame
  size** (0.25 % at 512x384 → 6.5 % at 1 MP): there is more causal region to
  copy from. A per-frame-size statement about intraBC is not a per-content one.

### 2c. The DIFFERENTIAL corpus — already reaching what the perf harness cannot (`.diffcorpus.tsv`)

`EncodeCell::real_content`: conformance vectors decoded back to pixels, i.e.
literally the cells `aom_bench::encode_cells()` builds and KB-13's byte-parity
map is measured on. Censused at **that harness's own cell** — cq32,
**cpu-used 0**, default knobs.

| family | denom | 64x64 | 128x128 crop | 196x196 |
|---|---|---:|---:|---:|
| **filter-intra** | leaves | **30.43** | **31.48** | **21.13** |
| filter-intra | coded px | 62.50 | 27.25 | 11.28 |
| directional | predicted px | 53.88 | 53.43 | 51.26 |
| nonzero angle delta | leaves | 21.74 | 24.07 | 17.53 |
| CFL | chroma-ref leaves | 0.00 | 8.76 | 5.17 |
| rect leaves | leaves | 73.91 | 80.56 | 81.96 |
| leaves ≤ 8 px | leaves | 91.30 | 94.91 | 82.47 |
| 4-pt fwd tx | fwd tx | 71.53 | 74.65 | 66.68 |
| non-DCT fwd tx | fwd tx | 76.82 | 77.87 | 76.12 |
| palette / intraBC | leaves | 0.00 | 0.00 | 0.00 |

**This is the cheapest finding in the study.** Four of the five families the
old §4 called blind spots are exercised heavily by the byte gates, at
14/136/82 filter-intra leaves per cell. The perf harness's blindness was never
a property of the encoder or of the corpus; it was a property of **one cell**
(1024x1024 / cq44 / cpu-used 6) and of three synthetic sources.

---

## 3. Filter-intra is a SPEED zero, and it is worth being precise about (`.speed.tsv`)

Same content (`winperf:photo`), same quantizer, four speeds:

| `--cpu-used` | leaves | filter-intra % (abs) | directional px % | rect leaves % | leaves ≤ 8 px % | 4-pt tx % | CFL % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **6** | 1 464 | **0.00 (0)** | 17.92 | 6.42 | 6.42 | 7.11 | 0.07 |
| 5 | 545 | 10.46 (57) | 56.61 | 57.80 | 0.00 | 15.03 | 0.00 |
| 4 | 1 024 | 8.40 (86) | 56.22 | 56.64 | 0.00 | 14.47 | 0.00 |
| 2 | 1 012 | 8.30 (84) | 68.88 | 27.87→65.81 | 0.00 | 8.74 | 0.00 |

The mechanism is exact and named in the port's own source: at `speed >= 6`,
`sf.prune_filter_intra_level = 2`
(`crates/aom-encode/src/speed_features.rs`, libaom `speed_features.c:529`),
which makes `rd_pick_filter_intra_sby` return without searching
(`intra_mode_search.c:244`). **No source can reach filter-intra at the harness
cell.** A filter-intra lever needs a different `--cpu-used`, not a different
image, and this is now a `--speed N` argument to the census tool.

Two more things fall out of the same table and are worth carrying:

* **The study cell's directional share is a speed-6 number.** The photograph's
  much-quoted 20.78 % becomes 56.61 % on the SAME content at cpu-used 5 —
  `intra_pruning_with_hog = 4` at speed 6 is doing most of that. Quoting
  "directional prediction is ~20 % of predicted pixels" without the speed
  attached is a 3x error.
* **Small leaves invert.** `photo` has 6.42 % leaves ≤ 8 px at speed 6 and
  **0.00 % at every slower speed** — speed 6 caps `default_max_partition_size`
  at `BLOCK_32X32` and splits differently. "Slower speed ⇒ smaller blocks" is
  not true here.

---

## 4. The new content: `winperf::Content::Screen`

Palette and intraBC are the one pair with **no reachable content anywhere in
the harness** — every winperf source is photographic-class and every one of
them is declined by aomenc's screen detection. So this is the family that
actually needed a generator.

### 4a. Why the other three provably cannot do this

The same shape of argument as `Photo`'s isotropy one — a property of the
generator, not of its parameters:

* **Palette** codes a block as an index map over at most 8 colours, so it can
  only win where a block contains at most a few distinct values. A 32x32 block
  of value noise carries hundreds; the colour cache and index map then cost more
  than a transform at every quantizer. **Measured**, on the source pixels, over
  16x16 blocks: `screen` never exceeds **8 distinct luma values** in any block;
  the photographic contents exceed **64** somewhere. Pinned by
  `screen_source_is_few_coloured_and_repetitive_and_the_others_are_not`, which
  measures the comparators in the same test so it cannot pass vacuously.
* **Intra block copy** codes a block as a displacement into the already-
  reconstructed part of the same frame, so it needs an exact earlier copy.
  Independent value noise repeats an 8x8 block with probability ~0. **Measured**:
  ≥ 20 % of `screen`'s 8x8 blocks are exact duplicates of an earlier one;
  < 1 % of any photographic content's are.

The generator: a grid of UI panels, each drawing a background level, a
foreground level and a chroma pair from a frame-wide ladder by hashing its grid
position, and each either flat or filled with a `glyph_px` grid of glyphs from
an `n_glyphs` alphabet. Antialiasing (one step at ink boundaries) is a realism
knob — libaom's screen detection is explicitly ANTIALIASING_AWARE — and it
preserves exact repetition, because the same glyph with the same colour pair
antialiases the same way. Chroma is **flat per panel** rather than the noise
plane the other three share; that is what makes the UV palette reachable at
all. Integer-only throughout, so the source bytes are identical on every target
and the cross-platform argument the whole harness rests on still holds.

### 4b. The fit, and the objective declared before it ran

**`L1_screen`**: the L1 distance, in percentage points, between the candidate's
and the target's **coded-leaf screen-tool class share** vector. The classes
partition coded leaves — each leaf falls in exactly one, tested in a fixed
order — `[intrabc, palette_y, filter_intra, directional_y, other]`. A leaf
distribution rather than the photo fit's predicted-pixel one, because palette
and intraBC leaves issue no intra prediction at all and a pixel-share objective
would be blind to precisely the two tools this content exists to reach.
`palette_uv`, `cfl`, `leaves` and coded bytes are reported alongside and are
**not** part of the objective.

**The target is a rule, not an image.** The first reference tried was a single
screenshot, and censusing `gb82-sc/imac_dark` at 512x512 returned **69.2 %
intraBC / 4.7 % directional** — a profile dominated by large flat black regions
and not representative of anything. Rather than pick a nicer image after seeing
that (which is §14 wearing a reference costume), the declared target is the
**per-class median over the screen corpus**, taken over the eight sources whose
`scdet` fired:

    --target 20.66,16.12,0.00,8.55,47.07
             ibc   pal    fi   dir   other

That rule was fixed, and the tool changed to take a vector rather than a path,
**before any candidate row was read**.

**The sweep**: 56 candidates over three passes, at 512x384 (the size the target
was measured at), each candidate a full SCREEN-bootstrapped encode with both
knobs on. Full table: `.fit.tsv`.

| pass | what it swept | best `L1_screen` |
|---|---|---:|
| 1 | colour alphabet x glyph alphabet x glyph size, everything else fixed | 22.17 pp |
| 2 | the `image_q8` axis pass 1 did not have, x `n_levels` x `text_q8` | 20.70 pp |
| 3 | walking the `n_levels` grid EDGE out | **16.77 pp** |

**Pass 1 could not reach the target from any point, and that is why pass 2
exists.** Every one of its 24 candidates put 6-11 % of leaves in the `other`
class against the target's 47.07 %, and 1-2 % directional against 8.55 %: a
frame that is wall-to-wall flat UI and glyphs has no ordinary intra blocks in
it. Real screenshots carry photos, icons and gradients, so `image_q8` (a share
of panels filled with the oriented photographic field) was added as the missing
degree of freedom. **An optimum outside the grid is the same objection as an
optimum ON a grid edge** — the fix is to widen the generator, not to pick a
friendlier target.

That axis has a clean interior optimum where it matters. At pass 2's leading
shape, `L1_screen` runs **36.01 → 20.70 → 22.27 → 24.56 → 46.05 → 83.23** as
`image_q8` goes 0 / 64 / 102 / 128 / 154 / 179. The last two rows are also
where **aomenc's own screen detection stops firing** (`scdet 0` at 179): there
is a hard ceiling on how photographic this content can be and still be screen
content by the encoder's test, and the sweep found it rather than assumed it.

**Pass 3 exists because pass 1's winner sat at the LARGEST `n_levels` on its
grid.** Walking that axis out: `L1_screen` runs **78.54 → 41.64 → 22.17 →
16.77 → 29.68 → 20.09** for `n_levels` 6 / 10 / 16 / 24 / 32 / 48. 24 is an
interior optimum; the 32/48 pair is non-monotone, so the far end of that axis
is noisier than the near end and the record says so rather than smoothing it.

**Shipped** (`winperf::SCREEN`, the argmin of the declared objective and
nothing else): `n_levels 24`, `glyph_px 16`, `n_glyphs 8`, `panel_px 128`,
`text_q8 154`, `ink_q8 96`, `aa 1`, `n_chroma 4`, `image_q8 0`.

| class (share of coded leaves) | target (corpus median) | shipped `screen` |
|---|---:|---:|
| intraBC | 20.66 | 24.04 |
| palette Y | 16.12 | 21.61 |
| filter-intra | 0.00 | 0.00 |
| directional Y | 8.55 | 3.96 |
| other | 47.07 | 50.38 |
| **L1** | 0.00 | **16.77** |

Three honest caveats:

* **The winner does not use `image_q8`.** It is 0 at the argmin — a large
  colour alphabet turns out to produce the `other` mass more cheaply than
  photographic panels do. The axis is kept because pass 1 proved the generator
  needs it to be *able* to reach that mass, and because a future refit against
  a different corpus may want it; but this content ships without it, and saying
  "we added an axis and the fit declined it" is more useful than quietly
  dropping it.
* **`text_q8` was not resolved at this frame size.** At 512x384 with
  `panel_px 128` the frame has only 4x3 = 12 panels, and no panel's layout hash
  falls between the two swept thresholds, so 102 and 154 produce **identical
  output to the digit** (visible as duplicate row pairs in `.fit.tsv`). The
  shipped value is the grid default, not an optimum, and is reported as
  unfitted.
* **`directional_y` is the class the fit gets most wrong** (3.96 against 8.55).
  A screen generator whose non-UI content is one photographic field does not
  reproduce the mix of antialiased artwork real screenshots carry. That is the
  next axis if this content is ever refit.

### Cross-checking the shipped content

At the gate cell (512x384, both knobs, screen bootstrap): **palette 21.61 %,
intraBC 24.04 %, leaves ≤ 8 px 75.19 %** of coded leaves — all three inside the
corpus's own range (`.screen.tsv`: palette 6.26-32.95, intraBC 0.25-58.82,
small leaves 41.92-90.78).

> **RE-MEASURED 2026-08-30 (KB-42), same cell: palette 22.75 %, intraBC
> 33.63 %, leaves ≤ 8 px 80.84 %.** The two large moves are the ceiling half of
> the gate doing its job: `735a0a6d` (KB-41 roots #3-#6) ported libaom's
> speed-dependent IntraBC search — `intrabc_search_level`, the hash-8x8
> block-count cap, the 64-candidate hash prune, the DIAMOND / CLAMPED_DIAMOND
> site configs and the ≥ 720p skip-row SAD — verified byte-identical against
> the C oracle on 30/30 datagen cells. The port therefore *finds* IntraBC
> matches libaom finds and it previously missed, and because IntraBC winners
> are overwhelmingly small blocks, `leaves_le_8px` rose with it. All three
> stay inside the corpus range above. `content_family_census.rs` is re-pinned
> to intraBC `[25.0, 42.0)` and leaves ≤ 8 px `[73.0, 88.0)` — both **floors
> rise** — keeping each row's original relative shape; `palette_y` keeps
> `[16.0, 28.0)`. This gate ran only on the two `portability` CI legs, so the
> re-pin was six commits late; see CLAUDE.md KB-42. **UV palette and CFL are 0.00 on it**, which the
corpus also mostly is (UV palette 0.00-0.52 %); neither is reachable through
this content and that is on the unreachable list in §6.

**A screen-tool share is a statement about frame SIZE as well as content**, and
the gate found that the hard way: at **256x256** this same content reaches
palette and intraBC **exactly zero times**, which is why the gate cell is
512x384. The corpus shows the same shape — `codec_wiki`'s intraBC share goes
0.25 % at 512x384 to 6.5 % at 1 MP. Both tools need enough already-
reconstructed frame to be worth coding against.

---

## 5. The gate

`crates/aom-bench/tests/content_family_census.rs`, run by `just census-gate`
and by CI's `portability` job:

    cargo test --release -p zenav1-aom-bench --no-default-features \
        --features census --test content_family_census

No libaom, no conformance corpus, **~21 s**. Four tests:

1. **`every_pinned_family_is_still_reached`** — each pinned family's share on
   each content, failing **in both directions** (playbook §5): below the floor
   means the family stopped being exercised and any band read against it is a
   structural zero; at or above the ceiling means it became more reachable and
   the record needs re-pinning. It also re-asserts
   `plane_total() == intra_total_calls()` per content, so a future encoder call
   site added without a plane tag fails here.
2. **`the_known_zeros_are_still_zero_and_still_for_the_stated_reason`** — the
   census must be able to tell an unreached family from a rare one, so the
   known zeros are pinned too, each with its stated cause: filter-intra at
   cpu-used 6 (speed), palette/intraBC with the knobs off (knob). If either
   starts firing, the coverage record is out of date in the good direction and
   should say so.
3. **`only_the_screen_bootstrap_signals_screen_content_tools`** — the screen
   fixture carries `allow_screen_content_tools` and the other three do not.
   Asserting the negative half is what stops this passing vacuously against a
   harness that forced the bit everywhere.
4. **`screen_source_is_few_coloured_and_repetitive_and_the_others_are_not`** —
   the two structural properties of §4a, measured on the SOURCE PIXELS, so they
   still say something if a future encoder change stops picking the tools.

The screen row runs at **512x384** (`winperf::SCREEN_GATE_CELL`, its own
committed 142-byte bootstrap) rather than 1 MP, because §2b's intraBC cost
finding makes a 1 MP screen census a multi-minute job and a gate nobody runs is
not a gate. Perf bands still use `winperf::CELL`. The whole gate is **~21 s**,
of which the screen row's palette + intraBC searches are ~20.

**Fixture size.** `Content::Screen`'s 1 MP bootstrap is a 23 201-byte aomenc
stream — 47 KB as committed hex, well past what belongs in git (an earlier
parameter point was 41 035 bytes / 83 KB). The harness
consumes only the sequence header and the frame OBU's uncompressed header, so
`winperf_bootstrap_gen` now trims the frame OBU payload to 128 bytes and
rewrites its leb128 size, giving a **142-byte** fixture. The trim is proven
safe rather than argued: the generator asserts the port emits **byte-identical
output** from the trimmed and untrimmed bootstraps before writing the file. The
three older fixtures are deliberately left untrimmed — every recorded winperf
band was taken against the bytes in the tree.

---

## 6. What remains unreachable, and what it would need

| family | status | what it would take |
|---|---|---|
| **filter-intra** | unreachable at the harness cell, richly reachable elsewhere | a `--cpu-used ≤ 5` band (10.46 % of leaves on `photo` at 5), or read it off the differential corpus, which already runs it at 21-31 % |
| **palette Y** | reachable, **knob-gated + header-gated + size-dependent** | `--enable-palette`, a `scr:` bootstrap whose detection fired, and screen content. `winperf:screen` supplies all three at ≥ 512x384; at 256x256 it reaches ZERO |
| **palette UV** | **effectively unreachable** | 0.00 % on `winperf:screen` and 0.00-0.52 % of leaves on the whole gb82-sc corpus. Nothing here makes it common; it would need content whose CHROMA is few-coloured where its luma is not, which no source in this corpus is |
| **intraBC** | reachable, **knob-gated + header-gated + size-dependent** | as above; and note its share grows with frame area (0.25 % → 6.5 % from 512x384 to 1 MP on the same source) |
| **CFL** | **content-gated only** | a real photograph (4.59 %) or `clic` (23.02 %); the synthetics reach 0.00-0.29 % and no knob is involved |
| **8x8 leaves specifically** | still ~0 on every winperf content at the study cell | the photograph itself only reaches 0.50 %; the differential corpus reaches 82-95 % of leaves ≤ 8 px. Not a gap worth generating content for — census `real:` or `clic` |
| **inter blocks** | 0 by construction | every cell here is a KEY frame. `leaf_inter` exists and is wired; an inter census needs an inter harness |
| **4:2:2 / 4:4:4 / 10-bit / monochrome** | not censused | the census is format-agnostic; the harness cells are all 8-bit 4:2:0 |
| **`gb82-sc/imac_dark`** | no completed row | the port's intraBC search on it does not terminate in a workable time; that is a port observation, not a census one |
