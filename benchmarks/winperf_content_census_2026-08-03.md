# The winperf harness could not see the code it was asked to measure (2026-08-03)

**A synthetic harness source is only representative of the axis it was fitted
on.** `winperf`'s two contents were fitted on **allocator call count**. They
were then quoted for a lever inside the **directional intra predictors**, where
`detail` reaches `z1` **six times in a whole 1 MP frame**. The Windows band was
reading a structural zero.

This lands three things:

1. a **committed, re-runnable census** (`aom_dsp::census`, feature-gated,
   `crates/aom-bench/examples/content_census.rs`) — so "is this content
   representative?" is a table rather than an argument;
2. a **third content**, `winperf::Content::Photo`, fitted to the study
   photograph's **intra mode distribution** by sweeping a parameter grid
   (`examples/content_fit.rs`), not by eye;
3. the **measurement**, on both Windows runners, of whether that makes
   KB-PERF-4's lever resolvable — see §5.

Provenance, commands, grid: [`winperf_content_census_2026-08-03.meta`](winperf_content_census_2026-08-03.meta).
Data: `.census.txt` (raw census, four sources) and `.fit.tsv` (467 candidates).

---

## 1. The census — winperf's sources against the study photograph

One `port_encode` of the study cell (1024x1024 / cq44 / cpu-used 6 / ALLINTRA /
8-bit 4:2:0 / 1 thread), warm-up subtracted. Counts are exact.

### Intra prediction, share of PREDICTED PIXELS

| class | study photograph | `detail` | `smooth` | **`photo`** (new) |
|---|---:|---:|---:|---:|
| non-directional | 76.05 % | 99.77 % | 80.17 % | **76.10 %** |
| `z1` (0–90°) | 5.94 % | **0.01 %** | 3.11 % | 4.46 % |
| `z2` (90–180°) | 9.79 % | **0.06 %** | 5.07 % | 8.45 % |
| `z3` (180–270°) | 5.05 % | **0.08 %** | 5.00 % | 5.00 % |
| `V` (exactly 90°) | 1.49 % | 0.01 % | 2.05 % | 3.00 % |
| `H` (exactly 180°) | 1.68 % | 0.08 % | 4.60 % | 2.99 % |
| **directional (`z1`+`z2`+`z3`)** | **20.78 %** | **0.15 %** | 13.19 % | **17.92 %** |
| directional, share of CALLS | 32.79 % | 0.28 % | 10.29 % | 23.30 % |
| **L1 distance to the photograph** | 0.00 pp | **47.43 pp** | 15.18 pp | **5.72 pp** |

In absolute terms `detail` predicts about **29 000 directional pixels** in the
whole frame against the photograph's **3 979 000**. `photo` predicts about
**3 005 000** — a **104x** increase in the work a directional-predictor lever
can reach, from 0.7 % of the reference's to 76 % of it.

### Coded leaf block size — the partition-depth distribution

Taken at the bitstream writer, so it is the decision, not a search visit.

| `bsize` | photograph | `detail` | `smooth` | `photo` |
|---|---:|---:|---:|---:|
| 8x8 | 0.50 % | — | — | — |
| 8x16 | 7.32 % | — | — | 3.69 % |
| 16x8 | 8.68 % | — | — | 2.73 % |
| 16x16 | 29.34 % | 18.79 % | — | 32.58 % |
| 32x32 | 54.16 % | 81.21 % | **100 %** | 61.00 % |
| leaves | 1 612 | 1 192 | 1 024 | 1 464 |
| L1 to photograph | 0.00 pp | 54.10 pp | 91.69 pp | 20.16 pp |

**`smooth` never splits below 32x32 anywhere in the frame** — 1 024 leaves is
exactly four per superblock. Any lever whose reach depends on small blocks is
invisible on it, and `detail` reaches no rectangular leaf at all.

### Forward transform mix

| tx size | photograph | `detail` | `smooth` | `photo` |
|---|---:|---:|---:|---:|
| 4x4 | 2.17 % | **0 %** | **0 %** | 1.48 % |
| 8x8 | 30.32 % | 30.94 % | **0 %** | 18.77 % |
| 16x16 | 31.98 % | 60.33 % | 66.86 % | 53.63 % |
| 32x32 | 8.03 % | 7.49 % | 33.14 % | 14.38 % |
| rect (4x8…16x8) | 27.50 % | 1.24 % | **0 %** | 11.74 % |
| total | 89 904 | 100 723 | 25 877 | 59 031 |
| L1 to photograph (type x size) | 0.00 pp | 63.86 pp | 132.46 pp | 56.29 pp |

`detail` issues **no 4x4 forward transform at all**, and `smooth` issues only
16x16 and 32x32. A transform lever scoped to the small sizes is unmeasurable on
either.

### Allocator calls — the axis the old contents WERE fitted on

Post-lever arm, exact (`examples/winperf_alloc.rs`):

| content | calls | vs photograph (512 557) |
|---|---:|---:|
| study photograph | 512 557 | — |
| `detail` | 488 750 | **95 %** |
| `photo` | 374 603 | 73 % |
| `smooth` | 222 464 | 43 % |

**This is why `photo` does not replace `detail`.** Fitting the mode
distribution moved the allocator count *away* from the reference. The two
contents are fitted on two different axes and the harness needs both;
which one a study should quote depends on the lever's mechanism, and that is
now a table lookup instead of a guess.

---

## 2. What the new content is, and why the old ones could not do this

`detail` and `smooth` are fractional-Brownian value noise. That is **isotropic
at every scale**: its expected structure is the same in every direction, so
within any block the eight directional gradients are near-equal (measured:
`detail`'s median max/min ratio over 32x32 blocks is **1.17**), the directional
predictors never win the RD decision, and no amount of re-tuning the amplitude
ladder changes that. It is a property of the generator, not of its parameters.

`Content::Photo` adds a **streak field**: 1-D value noise evaluated on the
projection of `(x, y)` onto a direction, which is exactly constant along the
perpendicular — the structure a directional predictor exists to extrapolate. A
low-frequency field selects that direction from eight spanning a half-turn and
blends the two nearest, so orientation **rotates smoothly across the frame** and
`z1`, `z2`, `z3`, `V` and `H` all get regions where they are the right answer.
The streak field is mixed with the isotropic ladder; the mixture weight is the
knob that moves the directional share.

Still integer-only, so it stays bit-identical on every target — the property the
whole cross-platform comparison rests on.

Measured local anisotropy (median max/min of the eight directional gradients
over 32x32 blocks): `detail` **1.169**, `photo` **1.761**. Pinned by
`winperf::tests::photo_is_locally_oriented_and_detail_is_not`, which measures
`detail` in the same test as the comparator so it cannot pass vacuously.

---

## 3. The fit — how the parameters were chosen

**The objective was fixed before the sweep ran**: L1 distance, in percentage
points, between the candidate's and the reference's intra-class
predicted-pixel-share vector. Not any lever's delta. Fitting content until a
lever's number looks good is how a harness stops measuring anything
(`docs/DIFFERENTIAL_PLAYBOOK.md` §14); fitting it to the reference distribution
and *then* measuring once is not.

467 candidates over four passes (`.fit.tsv`; `examples/content_fit.rs` is the
tool, and the grid is in its source):

| pass | what it swept | best L1 |
|---|---|---:|
| 1 | the mixture axis whole, everything else coarse | 9.14 pp |
| 2 | streak periods, fine amplitude, contrast | 7.49 pp |
| 3 | refinement where pass 2's best sat on a grid EDGE | 5.84 pp |
| 4 | walking the remaining edge (`streak_p`) out | **5.72 pp** |

The mixture axis has a clean interior optimum — at `orient 128 / streak_p
(64,16) / contrast 256` the L1 runs 45.72 → 44.00 → 35.41 → 18.48 → **9.14** →
20.80 → 36.39 as the streak weight goes 2/10 … 8/10 — so the fit is not a
saturated edge. Passes 3 and 4 exist because pass 2's winner sat at the smallest
`contrast` and smallest fine amplitude on their axes, and an edge optimum is not
an optimum.

Passes 3 and 4 plateau: the top three rows are **5.72 / 5.75 / 5.84 pp**, which
is inside the grid's own resolution. The winner is the **argmin of the declared
objective** and nothing else — the 5.75 row happens to have better *secondary*
distances (leaf-size 18.47 vs 20.16, transform 52.03 vs 56.29, coded bytes 4 701
vs 5 301, against the reference's 4 458) and was not chosen, because those were
declared as reported-but-not-optimised.

Shipped (`winperf::PHOTO`): `orient_period 128`, `streak_p (96, 24)`,
`streak_a (40, 4)`, `iso_a [26,22,18,15,12,10]`, `mix (12, 8)`, `contrast 208`.

---

## 4. What the harness can now see, and what it still cannot

**Can see** (was blind, now within ~6 pp of the reference distribution):

* directional intra prediction — `z1` / `z2` / `z3` at 4.5 / 8.5 / 5.0 % of
  predicted pixels against the reference's 5.9 / 9.8 / 5.1;
* rectangular partitions and 4x4 / 4x8 / 8x4 transforms, none of which
  `detail` or `smooth` produce in any quantity;
* the same allocator and forward-transform axes as before, on `detail`, which
  is unchanged and still the right content for those.

**Still cannot see** — and this list is the point of committing a census:

* **`V` and `H` are over-represented** (3.0 % each against 1.5 / 1.7). The
  orientation field spends time near the axis-aligned directions.
* **directional CALLS are under-represented** (23.3 % against 32.8 %) even
  though the pixel share matches — `photo`'s directional blocks are larger than
  the photograph's. A lever priced per CALL rather than per pixel is still
  measured ~30 % light.
* **8x8 leaves** (0.50 % of the reference's) essentially do not occur, and
  16x8/8x16 occur at a third of the reference's rate.
* **filter-intra is zero on all four sources**, including the photograph. Any
  filter-intra lever has no content here at all.
* **palette / intraBC / screen content: nothing.** All four sources are
  photographic-class. A screen-content lever needs a fourth content and this
  study does not provide one.
* **CFL and the chroma path** are not censused at all — the counters are on the
  luma predictor, the forward transform and the leaf writer.
* **one cell.** 1024x1024 / cq44 / cpu-used 6. At another speed the partition
  and mode mix moves; nothing here says by how much.
* the census is taken on **Darwin only**. It is quoted for the runners on the
  strength of the byte gate: identical source bytes in, identical coded bytes
  out of a deterministic single-threaded search, therefore identical decisions.
  The allocator census *is* measured on all three targets and is byte-for-byte
  equal, which is the same argument with the evidence attached.

---

## 5. Does the harness now resolve KB-PERF-4's lever?

**On `windows-11-arm`, yes. On `windows-latest` x86-64, no — and that is now a
quantified statement about the runner rather than a shrug.**

The positive control is KB-PERF-4 itself (the i16-lane directional-predictor
path, `71c924a` against base `0279544`): **−0.75 %** on the study photograph on
Darwin, and unresolvable on both runners when read off `detail`
(`encoder_intra_dir_i16_2026-08-03.md` §7).

Run **30798500036**, `arms: prepost`, `rounds: 24`, `contents: photo detail`,
both runners — so **`photo` and `detail` are measured in the same job, on the
same VM, minutes apart**. `detail` is the control: the census says the lever
cannot reach it, so whatever `detail` reports is the harness, not the lever.
Darwin was run the same day on the same two binaries
(`.darwin_photo.tsv` / `.darwin_detail.tsv`, 24 rounds, 4 arms, load ~1.4-2.5).

### The bands

Pooled per-round paired median of the post-side copies against the pre-side
copies (`scripts/winperf_prepost_stats.py` — `prepost` mode gives three
identical pre-side arms and two identical post-side arms, and reading them
pooled is what makes the numbers below stable):

| | `photo` | `detail` (control) | **difference** | sd/round (`photo`) | **MDE at 95 %, n=24** |
|---|---:|---:|---:|---:|---:|
| Darwin M4 Pro | **−0.356 %** | +0.131 % | **−0.487 pp** | 0.189 | 0.076 % |
| `windows-11-arm` | **−0.332 %** | +0.167 % | **−0.499 pp** | 0.450 | 0.180 % |
| `windows-latest` x86-64 | +0.153 % | −0.077 % | +0.230 pp | 1.250 | 0.500 % |

Sign test on the per-round ratios, `windows-11-arm`: `photo` **19/24 rounds
post-side faster, p = 0.0066**; `detail` 6/24, i.e. significantly the *other*
way. The two contents disagree in sign, with significance, on one VM in one job.
That disagreement is the whole point of the content change.

**`windows-11-arm`'s −0.499 pp and Darwin's −0.487 pp agree to 0.012 pp.** This
lever's mechanism is integer lane arithmetic, not a call into the platform, so
§6b predicts it travels — and now that the harness reaches it, it measurably
does. Contrast KB-PERF-2's allocator lever, which is 5x larger on Windows.

### Read the difference, not the raw band, and here is why

The raw post-vs-pre figures are contaminated by a **within-round position
drift**: on `windows-11-arm`, among *identical* binaries, each successive arm in
a round is faster than the last (`pre` → `l3a` +0.185 %, `l3a` → `l3` +0.277 %).
`pre` is always first and pays for it. The pre-side arms track each other
**across the two contents to ≤ 0.09 pp**, which is what licenses differencing:
the drift is a property of the round, not of the content, so `photo − detail`
removes it and leaves the lever.

This is why the naive `--vs pre` reading of this run (`post` vs `pre` −0.80 % on
`photo`) overstates it, and why the within-side null it has to clear (0.376 %)
looks so large. Both numbers are mostly position.

### The control band, and the noise floor per target

Two nulls are worth distinguishing, because they answer different questions:

| | adjacent-arm null (identical binaries, one position apart) | round-to-round sd of the paired ratio | **smallest effect resolvable at 95 %, n=24** |
|---|---:|---:|---:|
| Darwin M4 Pro | +0.11 % / −0.07 % | 0.17-0.19 | **0.07 %** |
| `windows-11-arm` | +0.11 % / +0.08 % | 0.17-0.45 | **0.07-0.18 %** |
| `windows-latest` x86-64 | −0.03 % / −0.09 % | **1.25-2.14** | **0.50-0.86 %** |

`windows-latest`'s per-invocation null is as good as anyone's; its **round-to-
round** dispersion is 7-12x worse (raw arm spreads 5-11 % against
`windows-11-arm`'s 0.7-1.9 %). It is a shared VM with neighbours, and no amount
of interleaving fixes a 4 % round. **At n=24 it cannot see anything under about
0.5 %, so it could not have resolved this lever at any content**, and saying so
is a fact about the runner rather than about the change.

Concretely: `windows-11-arm` at n=24 resolves ~0.2 %; to resolve 0.2 % on
`windows-latest` would need roughly `(1.25/0.45)² ≈ 8x` the rounds.

### A finding the old harness could not have produced

`detail`'s control band is **not zero**: +0.167 % on `windows-11-arm` and
+0.131 % on Darwin — the same sign and nearly the same size on two different
CPUs, on content where the census says the lever's kernel essentially never
runs. KB-PERF-4's landing noted that its runtime gate makes non-qualifying
blocks decline and that "the cost of that decline is folded into the reported
win rather than isolated". This is that cost, isolated: **about +0.15 % on
content with no directional prediction**, which the lever repays several times
over on content that has it. It took two contents whose mode censuses differ to
see it at all.

### Cross-platform determinism of the new content

`photo`'s allocator census is byte-for-byte identical on `darwin-arm64`,
`aarch64-pc-windows-msvc` and `x86_64-pc-windows-msvc` — **374 603 calls,
252 359 139 bytes** on all three, `peak_live` differing by 2 bytes out of
17.6 MB — and every arm on every target emits **5 301** coded bytes. The
integer-only generator holds, and the same argument that licenses quoting the
Darwin mode census for the runners (identical bytes in, identical bytes out of a
deterministic search) now has this evidence behind it.

---

## 6. Cost

Run 30798500036, `rounds: 24`, two contents: **15m01s** (`windows-11-arm`) /
**16m13s** (`windows-latest`). Build and setup is ~3.4 min of that; the timing
phase is linear in `rounds x sum(per-content cost)`, at roughly 12.9 s per round
for `photo`, 16.1 s for `detail`, 10.4 s for `smooth` (5 arms, 9 encodes each).

So, at the workflow's default `rounds: 12`:

| `contents` | timing | total job |
|---|---:|---:|
| `detail smooth` (the old fixed pair) | 5.3 min | ~8.7 min |
| **`detail smooth photo` (the new default)** | 7.9 min | **~11.3 min** |
| `photo` alone — a mode-scoped lever | 2.6 min | **~6.0 min** |

The default costs **+2.6 min (+30 %)**. A study that dispatches only the content
its census says reaches its lever costs **less than the old default did**, which
is the shape this should have had all along.
