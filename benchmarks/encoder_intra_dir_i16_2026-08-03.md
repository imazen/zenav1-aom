# An i16-lane path for the DIRECTIONAL intra predictors — 3.2115x → 3.1872x, and the ranked row was 13x optimistic before a line was written (2026-08-03)

Lever 4 of [`encoder_hotspot_reprofile_2026-08-02.md`](encoder_hotspot_reprofile_2026-08-02.md)
— **"bd8 lowbd intra predictors", +14.54 ms / 12.3 % of the encoder gap**.

**This session re-measured the row FIRST**, per that document's own
CROSS-PLATFORM SCOPING banner and
[`DIFFERENTIAL_PLAYBOOK.md §14`](../docs/DIFFERENTIAL_PLAYBOOK.md), because the
two preceding levers were 18x and 5x optimistic for the same structural reason.
The re-measurement changed the scope of the work **before** any code was
written, which is the point of the exercise; the delivered number is small and
is reported as such.

Provenance, box, exact commands, what is NOT measured:
[`.meta`](encoder_intra_dir_i16_2026-08-03.meta). Data: `.control.tsv` (the
session-opening 2-arm band), `.ab.tsv` (the 36-round headline band), and
`.ab_split.tsv` (the 24-round 7-arm band that splits the two halves),
`.stage.tsv` (the like-for-like intra symbol table, both arms),
`.census.txt` (the intra call census).

---

## 1. What the row is worth NOW — the answer is "it depends what you call the row"

Re-profiled at the post-i16-fwd baseline (`0279544`), same cell, same tooling,
same libaom build, two independent port sampling runs plus one C run.
Denominators are the medians taken DURING the profile runs (port 150.539 ms,
libaom-c 45.602 ms → 3.30x; the clean interleaved control is 3.2383x — `sample`
suspends the target and inflates the port arm ~4 %).

**Total gap: 118.37 ms → 104.94 ms** since the re-profile (the CNN cache, the
allocation levers and the i16 forward transform have landed in between).

| reading | port ms | C ms | ratio | gap ms | % of the 104.94 ms gap |
|---|---:|---:|---:|---:|---:|
| the re-profile's ranked row (`dsp:intra-pred` + `intra-mode-rd`, combined as it instructs) | 19.75 | 4.36 | 4.53x | **+15.39** | **14.7 %** |
| — of which the encoder's intra **RD drivers** (`rd_pick_intra_sby_mode`, `intra_uv_rd`, `encode_intra`, `intra_rd`) | 8.12 | 1.82 | 4.46x | +6.30 | 6.0 % |
| **the intra PREDICTOR class, like for like** | **9.91** | **2.33** | **4.25x** | **+7.58** | **7.2 %** |
| chroma-from-luma (`cfl`), split out because the two arms file it differently | 1.72 | 0.21 | 8.34x | +1.52 | 1.4 % |

The three sub-rows reconcile to the row: **7.58 + 6.30 + 1.52 = 15.40 against
the combined 15.39** — nothing is double-counted or dropped.

**So the stale 12.3 % has not shrunk — as a stage it has GROWN to 14.7 %,
because the denominator fell 118.37 → 104.94 ms. What has changed is that the
row is now known to be only HALF (49 %) the thing the lever is named after.**
`intra-mode-rd` on the port side matches
`aom_encode::{intra_rd,rd_pick,intra_uv_rd,encode_intra}` — mode loops, RD
bookkeeping, transform-block walks — and porting predictors does not move them.
The honest reach of "bd8 lowbd intra predictors" is **+7.58 ms, 7.2 %**, and
that is the number a future ranking should carry.

### 1b. Inside the predictor class, where it actually goes

`.stage.tsv` has every row. Grouped:

| | port ms | C ms | ratio | gap ms |
|---|---:|---:|---:|---:|
| **directional `z1`/`z2`/`z3`** | **3.50** | **0.37** | **9.6x** | **+3.13** |
| `smooth` + `paeth` (already i32x8 SIMD in the port) | 2.97 | 0.34 | 8.7x | +2.63 |
| edge filter / upsample | 0.86 | 0.13 | 6.6x | +0.74 |
| DC / V / H fills (`predict_highbd` self) | 0.80 | 0.08 | ~10x | +0.71 |
| edge assembly + mode routing (`build_*`, `assemble_*`, `dr_predict_high`) | 1.79 | 1.42 | 1.26x | **+0.37** |

Two things worth naming:

* **The port's edge assembly and mode routing are near parity with C** (1.26x).
  The gap is in the KERNELS, not in the plumbing around them, which is a useful
  negative result: there is no structural per-block overhead to remove here.
* **The directional predictors are the worst ratio in the class and the only
  intra kernels with no vector path at either bit depth.** `dir.rs`'s
  `z1_high` / `z2_high` / `z3_high` — and their lowbd `z1`/`z2`/`z3` twins —
  were pure scalar, while libaom dispatches `av1_dr_prediction_z{1,2,3}_neon`
  (`aom_dsp/arm/intrapred_neon.c:1290-1482`) and `_avx2` / `_sse4_1`. **So for
  this sub-lever the brief's framing — "the port runs the highbd path where
  libaom runs lowbd_*_neon" — is not quite right: there is no highbd-vs-lowbd
  choice to make, because neither port path is vectorized at all.** For
  `smooth`/`paeth` the framing IS right (the port has an `i32x8` kernel where
  libaom has a `u8`/`u16` one), and those stay for a follow-up.

> **Attribution caveat, stated rather than glossed:** libaom's per-`z` split is
> approximate. `dr_prediction_z1_{4,8,16,32,64}xH_neon` and friends are
> `AOM_FORCE_INLINE` **statics**, and `libaom.a` carries no `-g`, so they
> symbolicate to the nearest preceding exported symbol (re-profile "Attribution
> limits" item 3). `av1_dr_prediction_z3_neon` reporting 0.004 ms against 968 k
> predicted pixels is that effect, not a real number. The **class total**
> (0.37 ms) is sound because every one of those statics lands inside it.

---

## 2. The call census — what a vector kernel can actually reach

One encode of the profile cell, exact counts from a temporary counter in
`predict_intra_high` and in the three `dir` kernels (both removed before
commit; the patch is preserved, see the `.meta`). Full output in
`.census.txt`.

**85 423 predictor calls / 19 151 632 predicted pixels per frame** — 333.7 calls
and 74 811 predicted pixels per 64×64 superblock, i.e. the encoder predicts
**18.3x the frame's pixels** in its mode search. **100 % of calls are bd8**
(85 423 / 85 423), so unlike the forward-transform lever there is no bit-depth
eligibility loss at all. Filter-intra never fires at this cell.

| class | calls | pixels | % of predicted px |
|---|---:|---:|---:|
| non-directional (DC / SMOOTH\* / PAETH) | 53 606 | 14 565 184 | 76.1 % |
| **z2** | 12 846 | 1 874 176 | 9.8 % |
| **z1** | 8 520 | 1 136 944 | 5.9 % |
| **z3** | 6 640 | 968 064 | 5.1 % |
| V (angle 90) / H (angle 180) | 3 811 | 607 264 | 3.2 % |

And inside the directional kernels, the branch census that determines what a
CONTIGUOUS-run vector kernel can take:

| | pixels | share of that kernel |
|---|---:|---:|
| z1, `upsample == 0` | 993 952 | **87.4 %** |
| z1, `upsample == 1` (stride-2 gather) | 142 960 | 12.6 % |
| z2, above-path (contiguous, constant `shift`) | 933 196 | **49.8 %** |
| z2, left-path (true gather, `base_y` not affine in `c`) | 940 980 | 50.2 % |
| z3, `upsample == 0` | 823 616 | **85.1 %** |
| z3, `upsample == 1` | 144 448 | 14.9 % |

By transform width, `w >= 16` is 69.8 % of directional pixels, `w == 8` is
29.4 %, `w == 4` is 1.3 %. **Pixel-weighted, 68.4 % of directional pixels are
addressable**, so the addressable port cost is `0.684 × 3.50 = 2.39 ms` —
before any question of how much faster the kernel is.

> **The 8-wide blocks ARE worth covering here, and that is NOT a contradiction
> of the forward-transform lever's refuted half-batch.** There, running an
> 8-dim block as a half-idle `i16x16` competed against an already-vectorized
> full `i32x8` batch and measured −0.006 %. Here the competitor is a **scalar
> loop**, so a half-filled 16-lane vector is still eight lanes at once. Same
> shape, opposite verdict, because the baseline is different.

---

## 3. Control band — read this before reading any delta

36 rounds, one invocation of each of **six** arms per round, interleaved
(`scripts/eprof_ab.sh`), each invocation 2 warm-up + 7 timed encodes with its own
median. 1024×1024 photo, cq 44, cpu-used 6. Box load 2.0-2.7 of 12 cores.

| arm | median | min | max | spread | bytes |
|---|---:|---:|---:|---:|---:|
| `base` (`0279544`) | 149.188 ms | 145.989 | 150.833 | 3.32 % | 4472 |
| **`baseB` (null — 2nd copy of `base`)** | **149.166 ms** | 146.543 | 150.415 | 2.64 % | 4472 |
| **`all` (as it ships)** | **148.062 ms** | 145.181 | 149.191 | 2.76 % | 4472 |
| **`allB` (null — 2nd copy of `all`)** | **147.972 ms** | 145.842 | 149.743 | 2.67 % | 4472 |
| `allC` (rejected store variant, §5) | 148.292 ms | 146.244 | 149.846 | 2.46 % | 4472 |
| `libaom-c` | 46.455 ms | 45.734 | 47.019 | 2.81 % | 4472 |

**The noise floor is measured on BOTH sides and is essentially zero here:**
`baseB` vs `base` is **−0.01 %** (paired-median **+0.03 %**) and `allB` vs
`all` is **−0.06 %** (paired −0.01 pp). All six arms emit the same 4472-byte
`.obu`, and the five port arms plus the C oracle produce the same file **by
sha**.

---

## 4. The result

| | vs `base` | paired-median | ratio vs libaom-c |
|---|---:|---:|---:|
| null (`baseB`) | −0.022 ms (−0.01 %) | +0.03 % | 3.2110x |
| **`all`** | **−1.126 ms (−0.75 %)** | **−0.75 %** | **3.1872x** |
| null (`allB`) | −1.216 ms (−0.82 %) | −0.76 % | 3.1853x |
| `allC` (rejected) | −0.896 ms (−0.60 %) | −0.59 % | 3.1922x |

**Ratio 3.2115x → 3.1872x, −0.024 on the ratio.** The paired per-round ratios
overlap at the extremes (base 3.183-3.249, all 3.151-3.222) — this is a 0.75 %
effect on a box whose raw spread is 2.6-3.3 %, and it is resolvable only
because the arms are interleaved and the nulls are measured.

> **RE-VERIFIED UNDER ROTATION 2026-08-03 — the SIGN survives, the MAGNITUDE
> −0.75 % is NOT RE-VERIFIED
> ([`encoder_rotate_reverify_2026-08-03.md`](encoder_rotate_reverify_2026-08-03.md)
> §4).** This band was taken with a FIXED arm order, which confounds arm with
> position (playbook §6), and `ROTATE=1` did not exist yet — the drift that
> motivated it is worth **45 % of this lever's whole effect**. Re-taken from the
> same two commits (`0279544` → `71c924a`, rebuilt, sha-verified) at 5 arms ×
> 50 and then × **150 rotated rounds**. The n=150 band: paired-median
> **−0.648 % / −0.623 %** for the two post-side copies (agreeing to 0.025 pp)
> against a null of **+0.095 %**, with **115/150 and 108/150 rounds faster,
> p < 0.0001** while the null sits at 71/150, p = 0.57. Ratio 3.1785x →
> 3.1592x/3.1588x (−0.020) against the −0.024 here. **BUT both rotated bands
> ran against a heavily loaded box (raw spreads to 212 %) and neither met the
> pre-registered resolution gate — MDE95 1.155 and 1.475 pp against a required
> 0.375.** So the effect is real and reproducible in sign across six independent
> post-vs-base comparisons, and the best available rotated reading is
> **−0.62 to −0.65 %**, but that is a non-resolving band and must not be quoted
> as a correction to the −0.75 % below. **One 150-round rotated band on an idle
> box (~20 minutes) would settle it.**

### The two halves, measured separately (§14's closing rule)

An earlier 24-round **7-arm** band (`.ab_split.tsv`) ran `z1z3` (z2's gate
forced false) and `z2` (z1's and z3's gates forced false) as separate arms:

| arm | vs `base` | paired-median |
|---|---:|---:|
| null (`baseB`) | −0.17 % | +0.02 % |
| `z1z3` alone | −0.20 % | −0.37 % |
| `z2` alone | −0.45 % | −0.33 % |
| `all` | −0.58 % | **−0.81 %** |
| null (`allB`) | −0.65 % | −0.54 % |

That band is **an independent replication of the headline** (−0.58 %/−0.81 %
against the 36-round band's −0.75 %/−0.75 %), taken earlier, on a different arm
set, in a different time window. Its per-half numbers are **consistent with
two roughly equal halves that sum to the composed effect**, but they are
2-3x that band's own −0.17 % null and should not be quoted as separate
resolved values. The 36-round band was run precisely because the first band's
nulls were a quarter of the effect — the same correction
[`encoder_i16_fwd_2026-08-02.md`](encoder_i16_fwd_2026-08-02.md) had to make.

### Why −1.13 ms and not −14.5 ms

Three named steps, each a measurement rather than an argument:

1. **The ranked row is 14.7 % but the predictor class is 7.2 %** — 41 % of the
   row is the encoder's own intra RD drivers and 10 % is CFL (§1).
2. **The directional sub-lever is 3.0 % of the gap** (+3.13 ms), because the
   non-directional predictors, the edge conditioning and the DC/V/H fills are
   the other 4.45 ms of the class and are untouched here (§1b).
3. **68.4 % of directional pixels are addressable** (§2), so the addressable
   port cost is 2.39 ms; **delivered 1.13 ms is 47 % of that.** The kernel
   removes the two loads, the two multiplies and the shift per pixel; it does
   not remove the per-run gate scan, the per-output store, or z3's strided
   scatter.

Read against the ranked row that is **13x optimistic**; read against the
sub-lever's own addressable cost it is **2.1x**. Both numbers are in this file
because the first is the one a ranking table would have produced and the second
is the one a named mechanism produces.

---

## 5. The audit — one tight bound, and it is the whole thing

The vector form is libaom's re-association (`intrapred_neon.c:1307-1308`):

```text
a0 * (32 - shift) + a1 * shift  ==  (a0 << 5) + (a1 - a0) * shift
```

an identity over the integers, so the two agree exactly provided no i16 lane
wraps. With `shift ∈ [0, 31]` (it is `((x << up) & 0x3F) >> 1`) and every tap
`0 <= v <= M`:

| intermediate | bound |
|---|---|
| `a0 << 5` | `<= 32M` |
| `a1 - a0` | `|.| <= M` |
| `(a1 - a0) * shift` | `|.| <= 31M` |
| the sum (`== a0*(32-s) + a1*s`, a convex combination scaled by 32) | `∈ [0, 32M]` |
| `+ 16` | `<= 32M + 16` |

All inside i16 **iff `32M + 16 <= 32767`, i.e. `M <= 1023`**. That is
`dir_simd::I16_TAP_MAX`, it is **tight** (at `M = 1024`, `a0 << 5` is exactly
`-32768`), and it is taken at **runtime** over the `O(bw + bh)` edge span each
predictor reads — against `O(bw × bh)` of predictor work, so it is a per-block
scan and never a per-pixel one.

In bit-depth terms the bound admits **bd8 and bd10** and declines **bd12**. It
is stated on the DATA, not on `bd`, so the path is sound for any caller of the
public predictors. **No bound was widened**, and nothing about the bd12 path
changed.

### The rejected micro-variant, measured not argued

`allC` replaces the kernel's per-lane store loop (`res.store(&mut [i16;16])`
then 16 `as u16` writes) with `res.bitcast_u16x16().to_array()` +
`copy_from_slice` — one memcpy instead of a lane loop, and a-priori the better
shape. It measured **−0.60 % against `all`'s −0.75 %**, i.e. **0.15 pp WORSE**
against a null of 0.01-0.06 pp. Reverted. The lane loop stays, and the reason
it wins is not established here.

---

## 6. Gates

* **`dir_simd_diff`** — the three dispatching entries vs the never-dispatched
  `z1_high_scalar` / `z2_high_scalar` / `z3_high_scalar` cores, at **every
  archmage token permutation (25 run)**, over every AV1 transform shape, every
  signalled angle through the real `dr_intra_derivative` table, bd 8/10/12
  sample ranges, tight AND padded stride.
  * `upsample` is **DERIVED** exactly as `build_directional_intra_high` derives
    it (`edge::use_upsample`), not swept freely — sweeping it freely walks the
    160-entry edge buffers off their ends **in the scalar kernel too**, which is
    how that was caught. The grid is therefore reachable triples only.
  * Probes are **asymmetric on purpose** (playbook §1 / KB-12): the change is a
    re-association whose new term is `(a1 - a0) * shift`, and a FLAT edge makes
    `a1 - a0 == 0` — a constant-edge probe is invariant under exactly the thing
    being tested. The six probe shapes are dense-random, ramp, max sawtooth,
    reverse ramp, step, and flat (kept only as a control).
* **`dir_simd::tests::two_tap_matches_scalar_at_every_tier`** — the kernel
  against its scalar core over run lengths 1..64, every `shift` in 0..32, five
  start offsets, eight edge shapes, with a `vector_cells > 1000` non-vacuity
  assertion.
* **`dir::reach`, pinned counts, both directions** — over the worst-case bd8
  edge (every sample 255) the gate admits **16 of 19 shapes for z1 and for z3**
  (the three `bw == 4` shapes and the three `bh == 4` shapes decline on the
  run-length floor) and **19 of 19 for z2**; `up == 1` never vectorizes. And
  `the_gate_declines_above_the_tap_bound`: 1023 is admitted, **1024 is not**,
  and a bd12-range edge declines at every shape.
* **`dir_simd::tests::the_tap_bound_is_load_bearing`** (playbook §2) — the
  other side of the gate: at exactly `I16_TAP_MAX` every shift agrees, and one
  sample over it the vector path genuinely **diverges**. A gate nothing can
  violate is decorative.
  * **This test failed the scalar-pinned leg on its first run**, and the failure
    was real: under `AOM_FORCE_SCALAR=1` `two_tap_run` routes to
    `two_tap_run_scalar`, so it cannot diverge from itself. The divergence half
    is now conditioned on `dispatch::scalar_forced()`; the gate's own
    rejection (1024 rejected, 1023 accepted) is asserted **unconditionally**,
    because that half is arithmetic on the span and has no tier. Nothing was
    relaxed — a precondition the test had left implicit was made explicit,
    which is the mirror image of playbook §1's "a test that cannot fail":
    a test that cannot pass in one dispatch mode is equally a defect in the
    test.
* **Bite proofs, with the asymmetry**:
  * dropping the `* shift` from the difference term fails the kernel
    differential alone — *"n=8 shift=0 start=0 rep=0, left: [661, 164, 195,
    176, 894, 596, 789, 557]"* — while the three `reach`/`bound` pins stay
    green (they do not compare against the scalar core);
  * swapping two rows in z3's scatter fails **`dir_simd_diff`**,
    **`dir_highbd_diff` (vs the real C symbol)** and **`intra_lowbd_diff`**,
    while **`intra_simd_diff`, `highbd_diff` and `build_nd_diff` stay green** —
    those three cover the non-directional family and must not see a directional
    defect.
* **Gate 2 keeps zero pinned cells**: `config_permutations` 87/87 (3 ignored),
  every `--cpu-used` 0..9 cell byte-exact against real aomenc.
* Full workspace with `--run-ignored all` in **both dispatch modes** (SIMD live
  and `AOM_FORCE_SCALAR=1`) plus
  `cargo check --target x86_64-apple-darwin --workspace --all-targets` — see
  the `.meta` for the counts.

---

## 7. x86-64 and Windows — MEASURED, NOT RESOLVED, and the harness is why

The lever is ONE `#[magetypes(define(i16x16, u16x16), v3, neon, -scalar)]` body
precisely so the AVX2 tier is the same source as the NEON one — that is what
makes it a **cross-platform** lever rather than the re-profile's ARM-only rank 1.
So the claim was taken rather than argued, on both Windows runners, in one
dispatch of `.github/workflows/winperf.yml` with **`arms: prepost`** (`base_sha`
`0279544`, 16 rounds, `l3a`/`l3` become extra COPIES of `pre`, so each band
carries **three** nulls — two pre-side and one post-side). Run
[30792984795](https://github.com/imazen/zenav1-aom/actions/runs/30792984795);
bands committed as `.win_<runner>_<content>.tsv`.

**Read the nulls first, and then stop.**

All figures are **paired medians** (per-round ratios, then the median), which is
the right statistic on an interleaved band:

| | `windows-11-arm` `detail` | `windows-11-arm` `smooth` | `windows-latest` `detail` | `windows-latest` `smooth` |
|---|---:|---:|---:|---:|
| null — `l3a` vs `pre` (a copy of `pre`) | +0.14 % | −0.21 % | +0.22 % | +0.65 % |
| null — `l3` vs `pre` (a copy of `pre`) | +0.22 % | −0.00 % | +0.38 % | −0.07 % |
| null — `postB` vs `post` (a copy of `post`) | +0.14 % | +0.11 % | −0.24 % | −0.19 % |
| **`post` vs `pre` (this landing)** | **+0.35 %** | **+0.05 %** | **+0.80 %** | **−0.68 %** |
| raw band spread | 0.9-1.8 % | 0.8-1.3 % | **3.2-22.8 %** | **4.9-31.4 %** |

**Verdict: NOT RESOLVED on either Windows runner.** Every `post − pre` figure
sits inside the span of that band's own same-binary nulls, and three of the four
have the wrong sign for a speed-up. This is not "the lever does not work on
x86" — it is "the measurement cannot see a 0.75 %-class effect here".
`windows-latest` is unusable on its face (raw spreads to 31.4 %, nulls to
±0.65 %). This is the opposite outcome to
[`encoder_i16_fwd_2026-08-02.md`](encoder_i16_fwd_2026-08-02.md), whose −2.8 to
−7.4 % cleared the same runners' nulls comfortably — and the difference is
mostly that this effect is **3-10x smaller**.

### But there is a second, bigger reason, and it is a property of the harness

`winperf` cannot ship the 1.5 MB study photograph, so it generates `detail` and
`smooth` with integer-only arithmetic (which is what makes its allocator census
bit-identical across three targets). **Those two sources barely use directional
intra prediction at all.** The same census, re-run on winperf's own content
(`.census_winperf.txt`):

| source | directional predicted px | of all predicted px | **share** |
|---|---:|---:|---:|
| the study photograph | 3 979 184 | 19 151 632 | **20.8 %** |
| winperf `smooth` | 1 905 152 | 14 447 616 | 13.2 % |
| winperf **`detail`** | **29 312** | 19 206 144 | **0.15 %** |

`detail` — the content the ARM runner's headline band uses — predicts **99.8 %
of its pixels with non-directional modes** (DC 49 %, SMOOTH 41 %, PAETH 10 %).
z1 fires **six times in the whole frame.** A lever that only touches directional
prediction has essentially **no work to do** on it; the expected effect there is
on the order of −0.005 %, and no amount of rounds resolves that.

**So the honest statement is: `windows-11-arm` / `detail` is a vacuous cell for
THIS lever, and `windows-11-arm` / `smooth` (13.2 % directional, nulls of
0.00-0.21 %) is the only Windows cell that could plausibly have seen it — it
measured **+0.05 %**, i.e. nothing.** The cross-platform claim rests on the mechanism
(integer lane arithmetic, no platform call — playbook §6b) and on the AVX2 tier
compiling and passing its differential, NOT on a Windows timing.

> **Durable lesson for `winperf` itself, and it generalises past this lever:**
> its two synthetic sources were tuned to bracket the photograph's *allocator
> call count* (`winperf.rs:63-71` says so), which made them the right harness for
> KB-PERF-2 (allocation) and KB-PERF-3 (the forward transform) — both of which
> touch every block regardless of mode. They are **not** representative of MODE
> DISTRIBUTION, and any future lever scoped to a mode family (directional intra,
> palette, filter-intra, CFL) must run this census on `detail`/`smooth` BEFORE
> reading a winperf band, or it will read a structural zero as a platform result.

`cargo check --target x86_64-apple-darwin --workspace --all-targets` passes, but
that only ever proved the AVX2 tier compiles.

## 8. What is NOT measured here

* **One cell, one image, one content class.** cpu-used 3/4/5/9 are not measured;
  the re-profile records 9 (5.64x) and 4 (7.76x) as worse cells and neither is
  decomposed. The census above is likewise one cell — at a preset that picks
  different block sizes and angles, the 68.4 % addressable share moves and the
  lever's value with it.
* **8-bit 4:2:0 only.** The gate is a runtime bound, so bd12 blocks simply
  decline; the cost of that decline (one `O(bw+bh)` scan that then takes the
  scalar path) is **folded into the reported win, not isolated** — no arm was
  built with the scan present and the vector path disabled.
* **No instruction count** (no valgrind on Apple Silicon), so "no vector kernel
  before, a 16-lane one now" and "0.75 % less time" are two different
  measurements and only the second is what the ratio moved on.
* **Single-threaded, one frame.**
* **x86-64 is COMPILED and differentially tested but not TIMED anywhere useful.**
  §7: the Windows bands are not resolvable, and the ARM runner's headline
  content has 0.15 % directional pixels. No Linux measurement at all.
* **The remaining 4.45 ms of the predictor class is untouched** — `smooth` +
  `paeth` (+2.63 ms, already `i32x8` where libaom is `u8`/`u16`), the edge
  filter (+0.74), the DC/V/H fills (+0.71). Those are the genuine "half the lane
  width" targets and are the obvious follow-up; nothing here measures what they
  would return.
* **z2's left half (50.2 % of its pixels), z1/z3's `upsample == 1` runs
  (12.6 % / 14.9 %), and every `w == 4` block (1.3 %) stay scalar** — reaching
  them needs a gather and, for z3's remaining cost, an i16 lane transpose that
  magetypes does not expose (`transform::simd::prims16` audits that surface).
* **The lowbd `u8` twins `dir::z1`/`z2`/`z3` are untouched.** The decoder's bd8
  path (`predict_intra_u8`) therefore gets nothing from this change; it is an
  encoder lever, and the encoder holds its planes as `u16` at every bit depth.
