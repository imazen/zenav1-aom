# `u16` lanes for SMOOTH — and PAETH built, measured null in four bands, reverted (2026-08-03)

The named follow-up of KB-PERF-4
([`encoder_intra_dir_i16_2026-08-03.md`](encoder_intra_dir_i16_2026-08-03.md)
§1b / §8): **"smooth + paeth, +2.63 ms, already `i32x8` where libaom is
`u8`/`u16`"**.

That landing found the brief's framing was *wrong* for the directional
predictors — `z1`/`z2`/`z3` had no vector path at any bit depth, so there was no
lane-width choice to make. **Here the framing is right**: both kernels ran a
32-bit lane per 8-bit sample where libaom runs `vmull_u8` / `vhaddq_u16` (and
`_avx2` / `_sse4_1` twins of the same shape). Half of it paid.

**What ships:** a `u16`-lane SMOOTH / SMOOTH_V / SMOOTH_H kernel behind a
runtime bound. **−0.38 % on the encoder, 3.1613x → 3.150x vs libaom-c** — 60
rounds, arm order rotated so position cannot confound, **51/60 and 52/60 rounds
faster on two independent copies of the shipped binary (p < 0.0001 each, paired
means agreeing to 0.005 pp)** against a same-binary null of −0.009 % at 30/60
(p = 1.00).

**What does not:** the `i16`-lane PAETH kernel. It was written, audited, gated,
differentially proved, and then measured as a **dead null in four independent
bands (~165 interleaved rounds, two store shapes, one of them with the arm order
rotated so position could not confound it)**. §4 is that result and the
mechanism behind it.

Provenance, box, exact commands, what is NOT measured:
[`.meta`](encoder_intra_smooth_paeth_2026-08-03.meta). Data: `.control.tsv`,
`.stage.tsv`, `.census.txt`, `.audit.txt`, `.ab.tsv`, `.ab_paethstore.tsv`,
`.ab_fixedorder.tsv`, `.ab_rotated.tsv`, `.ab_shipped.tsv`,
`.win_<runner>_<content>.tsv`.

---

## 1. What the row is worth NOW

Re-profiled at the post-KB-PERF-4 baseline (`1f366a9`), same cell, same tooling,
same libaom build, two independent port sampling runs plus one C run.
Denominators are the medians taken DURING the profile runs (port 155.101 ms,
libaom-c 47.202 ms); `sample` suspends the target and inflates the port arm, so
ratios are quoted from the clean interleaved bands in §3, never from here.

The like-for-like intra predictor class, same regexes as KB-PERF-4 (`.stage.tsv`
has every symbol):

| | port ms | C ms | ratio | gap ms |
|---|---:|---:|---:|---:|
| **`smooth` + `paeth`** | **3.178** | **0.326** | **9.7x** | **+2.852** |
| — of which **SMOOTH** | 2.176 | 0.189 | 11.5x | **+1.986** |
| — of which **PAETH** | 1.003 | 0.137 | 7.3x | **+0.865** |
| directional `z1`/`z2`/`z3` (after KB-PERF-4) | 2.773 | 0.437 | 6.3x | +2.336 |
| edge filter / upsample | 0.955 | 0.146 | 6.5x | +0.809 |
| DC / V / H fills | 0.846 | 0.107 | 7.9x | +0.738 |
| edge assembly + mode routing | 1.726 | 1.445 | **1.19x** | +0.281 |
| CFL | 1.688 | 0.286 | 5.9x | +1.403 |
| **the class** | **11.165** | **2.748** | 4.06x | **+8.418** |

**The projection held — the first one in this sequence that did.** KB-PERF-4
sized this sub-lever at **+2.63 ms** from its own like-for-like table;
re-measured one landing later it is **+2.852 ms**, 8 % larger. KB-PERF-2 was 18x
optimistic, KB-PERF-3 5x, KB-PERF-4 13x against its ranked row. The difference is
exactly the one [`DIFFERENTIAL_PLAYBOOK.md` §14](../docs/DIFFERENTIAL_PLAYBOOK.md)
names: those three projections came from a profiler's ranked **stage**, this one
came from a **named mechanism** measured like for like — two symbols on one side,
ten on the other, nothing else in the row.

**The split inside it turns out to be the whole story.** SMOOTH is 1.40 % of the
encode and PAETH 0.65 %. §4 shows that a 0.65 % row is at the floor of what this
box resolves, and that PAETH's kernel is store-bound anyway — but the split was
in the table before either band was run, which is why both halves were built as
separate arms from the start.

The directional row moved 3.50 → 2.77 ms, which is KB-PERF-4's own landing
showing up in a fresh profile. Everything else is within noise of its previous
reading.

---

## 2. The census — what a column kernel reaches, by MODE and by WIDTH

**Not throwaway instrumentation this time.** `aom_dsp::census` gained
`nd_mode_tx_calls` / `nd_mode_tx_px` (the non-directional mode axis split by
transform shape) and `content_census.rs` gained the `nd_mode_x_tx` /
`nd_mode_x_width` tables, because a column-vectorized predictor is priced by
**block width** — how much of a lane vector it can fill — and the committed
census could not report that. Both are in the tree; `.census.txt` is the raw
output for four sources.

At the profile cell, of **19 151 632 predicted pixels**:

| mode | calls | pixels | share of predicted px |
|---|---:|---:|---:|
| DC | 31 302 | 6 217 504 | 32.5 % |
| **SMOOTH** | **13 619** | **5 258 656** | **27.5 %** |
| **PAETH** | **8 685** | **3 089 024** | **16.1 %** |
| H (exactly 180) | 1 930 | 321 472 | 1.7 % |
| V (exactly 90) | 1 881 | 285 792 | 1.5 % |
| **SMOOTH_V** | **0** | **0** | **0 %** |
| **SMOOTH_H** | **0** | **0** | **0 %** |

**SMOOTH + PAETH is 43.6 % of every pixel the encoder predicts** — more than
twice the directional family's 20.8 %. **SMOOTH_V and SMOOTH_H are never chosen
at all**, on the study photograph or on any of the three winperf contents; they
ship because they are the same twenty lines and the same audited bound as
SMOOTH, and this file makes no performance claim about them.

### Eligible fraction

Two axes, both counted rather than assumed:

* **Bit depth: 100 % eligible.** The cell is 8-bit, so every sample is `<= 255`
  and the gate (`M* = 255`) admits every block. Contrast KB-PERF-3, where only
  51.6 % of forward-transform calls were eligible, and KB-PERF-4's 68.4 %
  pixel-weighted directional share.
* **Block width**, share of that mode's own predicted pixels:

| | `bw = 4` | `bw = 8` | `bw = 16` | `bw = 32` |
|---|---:|---:|---:|---:|
| SMOOTH | 0.08 % | 7.09 % | 19.78 % | 73.04 % |
| PAETH | 0.17 % | 7.40 % | 23.37 % | 69.05 % |

**`bw >= 16` — where a 16-lane vector is completely full — is 92.8 % of SMOOTH's
and 92.4 % of PAETH's pixels.** The remaining 7.2 % run at 8 live lanes and
0.2 % at 4, which is no worse than the `i32x8` path they replace. So the
addressable fraction is **~100 %** of the row, on both halves.

That is the honest ceiling, and it is also why §4's null is a real finding rather
than an eligibility artefact: nothing was standing between PAETH's kernel and its
work.

### And the census says winperf can see this lever — unlike the last one

Run on all four sources in one invocation (`.census.txt`). SMOOTH + PAETH as a
share of predicted pixels:

| source | SMOOTH + PAETH | (KB-PERF-4's directional share, for contrast) |
|---|---:|---:|
| the study photograph | **43.6 %** | 20.8 % |
| winperf `detail` | **50.6 %** | **0.15 %** |
| winperf `photo` | **39.1 %** | 17.9 % |
| winperf `smooth` | **38.3 %** | 13.2 % |

`detail` — the content on which KB-PERF-4's Windows band was structurally
vacuous — carries **more** of this lever's mode family than the reference
photograph does. This is the census being run *before* the band rather than as a
post-mortem, and this time it says go ahead.

---

## 3. The result

### Control band first (playbook §6)

The session-opening 2-arm control (`.control.tsv`, 9 interleaved invocations in a
quiet window): port **150.07 ms**, libaom-c **47.60 ms**, per-arm spread 1.6 %.

**BOX CAVEAT, stated before any number:** the dev box was shared with two other
agents for most of this session, and their load is bursty — individual bands
below run from 1.6 % raw spread (quiet) to 80 % (a neighbour's test suite
landing mid-round). Every figure is a **paired per-round** statistic read
against **same-binary null arms measured in the same band**, and the sign test
is quoted because the raw distribution has a heavy tail that makes a parametric
MDE meaningless on the contended bands.

### The shipped kernel — 60 rounds, rotated, on the literal shipped binary (`.ab_shipped.tsv`)

Five arms x 60 rounds, `ROTATE=1` so each arm spends exactly 12 rounds in each
of the 5 positions. `all` / `allB` are two copies of the binary that this
source rebuilds to, **verified by sha256**. Raw spreads 2.7-3.5 %.

| arm | median | min | max | spread | bytes |
|---|---:|---:|---:|---:|---:|
| `base` (all three gates forced false) | 152.128 ms | 151.256 | 156.610 | 3.54 % | 4472 |
| **`baseB` (null — 2nd copy of `base`)** | **152.167 ms** | 151.255 | 156.261 | 3.31 % | 4472 |
| **`all` (as it ships)** | **151.688 ms** | 150.962 | 155.007 | 2.68 % | 4472 |
| **`allB` (null — 2nd copy of `all`)** | **151.586 ms** | 150.648 | 155.748 | 3.39 % | 4472 |
| `libaom-c` | 48.180 ms | 47.916 | 49.368 | 3.03 % | 4472 |

| arm | paired median | paired mean | sd | MDE at 95 %, n=60 | rounds faster | p (sign) |
|---|---:|---:|---:|---:|---:|---:|
| **null (`baseB`)** | **−0.009 %** | −0.060 % | 0.536 | 0.136 % | **30/60** | **1.00** |
| **`all`** | −0.200 % | **−0.373 %** | 0.574 | 0.145 % | **51/60** | **<0.0001** |
| **`allB`** (same binary as `all`) | −0.448 % | **−0.378 %** | 0.627 | 0.159 % | **52/60** | **<0.0001** |

**Read the two post-side copies together: their paired MEANS agree to 0.005 pp
(−0.373 % and −0.378 %) while their paired medians sit 0.25 pp apart, which is
what a 0.38 % effect against a 0.55 sd looks like at n=60.** The headline is
**−0.38 %**. The pre-side null is a textbook zero — 30 of 60 rounds, p = 1.00 —
and the position gradient that ruined the earlier fixed-order bands is gone:
pooled over all arms, positions 1-5 run 152.640 / 152.518 / 152.594 / 152.485 /
152.566 ms, a 0.1 % spread against band 4's 1.7 %.

**Ratio against libaom-c, paired per-round median: `base` 3.1613x →
3.1528x (`all`) / 3.1474x (`allB`)**, i.e. **3.1613x → 3.150x**, −0.011 on the
ratio, −0.49 ms on the arm medians.

All arms emit the same 4472-byte `.obu`, **by sha256**, including the C oracle.

### The three earlier bands, which say the same thing

Measured as the `smoothonly` arm (SMOOTH-family gates live, PAETH's forced
false, so PAETH's kernel is dead-code-eliminated) — functionally the shipped
kernel, differing only in panic-location line numbers:

| band | n | arm order | raw spread | paired median | rounds faster | p (sign) | null (`baseB`) |
|---|---:|---|---:|---:|---:|---:|---:|
| 3 (`.ab_fixedorder.tsv`, quiet) | 44 | fixed | 1.6-2.2 % | −0.362 % | 36/44 | <0.0001 | +0.108 % (15/44) |
| 1 (`.ab.tsv`, contended) | 44 | fixed | 10.6-13.6 % | −0.279 % | 32/44 | 0.0037 | +0.070 % (19/44) |
| 4 (`.ab_rotated.tsv`, contended) | 49 | rotated | 65-80 % | −0.306 % | 31/49 | 0.085 | +0.151 % (23/49) |

Four bands, ~200 interleaved rounds, two arm orders, contended and quiet:
**−0.28 / −0.31 / −0.36 / −0.38 %.** Nothing here is a single reading.

### Why −0.38 % and not −1.0 %

The addressable cost is ~100 % of SMOOTH's +1.99 ms row (§2), i.e. 1.40 % of the
encode; delivered is 0.38 %, so **the kernel returned about 27 % of its own
row**. What this file can say about the rest is bounded rather than decomposed:
it is not bit-depth eligibility (100 %, counted), not block width (92.8 % full
vectors, counted), and not the store shape (§4, measured). What is left is that
halving the *arithmetic* does not halve the *kernel*: the loads, the per-row
splats, and the 32-byte stores into a `u16` plane — twice libaom's `u8` bytes for
the same pixels — are all unchanged, and on this hardware the load/store path
around a kernel routinely eats what the kernel saves. That is the same closing
note KB-PERF-3's refuted half-batch ends on, and §4 is the same observation with
a control attached.

---

## 4. PAETH: built, audited, gated, measured NULL four times, reverted

This is the part of the named lever that did not survive contact with a band, and
it is reported at length because a refuted hypothesis is a result.

The kernel was complete: `i16x16` lanes, the same column-outer hoisting, an
exhaustively-derived bound (`M* = 16383`, admitting bd8 **and bd10 and bd12** —
wider reach than the SMOOTH half), a full differential at every token tier over
four sample ranges, a reach pin and a bite pin. It is preserved at
`~/tmp/smooth/simd16.withpaeth.rs` and the bound is still derived by
`xtask/audit_nd16_lanes.py`.

**Every band it appeared in:**

| band | n | arm order | PAETH alone, paired median | rounds faster | p (sign) |
|---|---:|---|---:|---:|---:|
| 1 (`.ab.tsv`) | 44 | fixed | +0.083 % | 20/44 | 0.65 |
| 2 (`.ab_paethstore.tsv`) | 36 | fixed | −0.013 % | 18/36 | **1.00** |
| 3 (`.ab_fixedorder.tsv`, quiet) | 44 | fixed | +0.091 % | 18/44 | 0.29 |
| 4 (`.ab_rotated.tsv`) | 49 | **rotated** | +0.037 % | 21/49 | 0.39 |

And the decisive comparison — **composed (SMOOTH+PAETH) against SMOOTH alone, in
the same rounds**:

| band | arm order | composed − smoothonly | rounds faster | p (sign) |
|---|---|---:|---:|---:|
| 1 | fixed | −0.302 % | 31/44 | 0.0096 |
| 3 | fixed | −0.004 % | 22/44 | **1.00** |
| **4** | **rotated** | **−0.007 %** | **25/49** | **1.00** |

Band 1's apparent −0.30 % is a **position artefact**, not PAETH: see the
position finding below. In the position-balanced band the composed binary and the
SMOOTH-only binary are indistinguishable to three decimal places.

**The mechanism, which makes the null a finding rather than a shrug.** PAETH's
inner loop is `add, abs, 2x compare, 2x blend` — six vector ops per chunk, then
one store per chunk. On NEON and AVX2 those six cost the same per *instruction*
regardless of lane width, so doubling the lanes should have halved them. It did
not move the wall clock, which says the kernel is **store-bound rather than
arithmetic-bound**. That has a control: band 2 replaced the per-lane store loop
with `bitcast_u16x16` + `copy_from_slice` (one 32-byte memcpy — a-priori the
better shape, and the same micro-variant KB-PERF-4 rejected for its own kernel),
and measured **+0.143 %, 13/36, p = 0.13 against `paethA`** — no measurable
difference either way. Two different stores, same null: the bottleneck is the
2 bytes per pixel going into a `u16` plane, which no lane-width change touches.

SMOOTH survives the same argument because it does **four multiplies, four adds
and a halving add** per chunk against the same store — its arithmetic-to-store
ratio is roughly three times PAETH's, and it is the half that measured.

Reverted per `DIFFERENTIAL_PLAYBOOK.md` §14: *"An argument about instruction
counts is not a measurement of them."* Shipping it would have added ~120 lines
and a runtime gate to the hot path for a benefit of 0.00 ± 0.05 %.

### A harness finding that came out of this, and it generalises

Band 3 is what caught it. Its two copies of one identical binary, at positions 5
and 6 of a fixed-order round, came out **0.34 pp apart** (−0.353 % and −0.010 %)
while the two copies at positions 1 and 2 agreed to 0.11 pp. **In a fixed-order
interleave, an arm's position inside the round is worth as much as the effect
being measured.** The same drift is on record for `windows-11-arm`
(`winperf_content_census_2026-08-03.md` §5), where it is handled by *pooling* the
copies on each side after the fact.

`scripts/eprof_ab.sh` now takes **`ROTATE=1`**, which rotates the arm order by
one each round so that over `N` rounds every arm spends `N/k` of them in each of
the `k` positions (the TSV gained a `position` column; `eprof_ab_stats.py` parses
both formats, so previously recorded bands still read). Band 4 is the first band
taken that way — occupancy is exactly 7/7/7/7/7/7/7 per arm — and the drift is
then directly measurable as a property of the round rather than of any arm:
pooled over all arms, **position 1 runs 162.45 ms and position 6 runs
159.67 ms, a 1.7 % gradient**.

Default off, so every band recorded before today stays reproducible
command-for-command.

---

## 5. The audit — one bound, tight, by exhaustive enumeration

`xtask/audit_nd16_lanes.py` (committed; output in `.audit.txt`). Unlike
`audit_i16_fwd.py`, which propagates an exact linear form through a 64-point
butterfly network, every intermediate here is a function of at most four small
scalars — so the bound is established by **enumerating the whole product space**,
not by an inequality. Nothing in it is an estimate.

| kernel | lane | binding intermediate at `M*` | `M*` | admits | witness at `M* + 1` |
|---|---|---|---:|---|---|
| SMOOTH | `u16` | `((A+B)>>1) + 128 = 65408` | **255** | bd8 | `(256-w)*b = 65536` |
| SMOOTH_V / SMOOTH_H | `u16` | `w*a + (256-w)*b + 128 = 65408` | **255** | bd8 | `(256-w)*b = 65536` |
| *PAETH (derived, not shipped — §4)* | `i16` | `base - top_left = -32766` | *16383* | *bd8/10/12* | *`32768`* |

The bound is **tight** — the `M* + 1` column is a witness that the audit left no
headroom, which is the other half of "is this bound sound?" and the half usually
skipped. **No bound was widened**, and the SMOOTH family genuinely declines bd10
and bd12: at those depths `super::simd`'s existing `i32x8` kernel runs exactly as
before.

The gate is stated on the **DATA, not on `bd`** (pinned by
`reach::the_bound_is_on_the_data_not_the_bit_depth`): a legal bd12 block whose
samples all happen to be `<= 255` takes the narrow path, and a bd8 caller that
somehow presented a larger sample is declined. That makes the path sound for any
caller of the public predictors, not only for the encoder.

### SMOOTH needs a halving add, and it must be the truncating one

SMOOTH's full numerator `p = wh*above + (256-wh)*below + ww*left +
(256-ww)*right` reaches `2*256*M`, outside `u16` for every `M >= 128`. So the two
halves stay separate and combine exactly as libaom's `vhaddq_u16` +
`vrshrn_n_u16` pair does:

```text
A = wh*above + (256-wh)*below            B = ww*left + (256-ww)*right
out = (((A + B) >> 1) + 128) >> 8   ==   (A + B + 256) >> 9
```

Both sides are functions of `A + B` alone, so the audit's sweep over **every
reachable sum, 0..130560**, is a complete verification rather than a sample. It
is specifically the **truncating** halving add that makes it exact: the audit
also runs the rounding form and reports it **first wrong at `A + B = 255`** — a
demonstration rather than a remark, and the perturbation §7's first bite proof
uses.

magetypes has no halving add, so the kernel writes
`floor((A+B)/2) == (A & B) + ((A ^ B) >> 1)`, which cannot overflow because its
value *is* the in-range result.

---

## 6. The kernel

`crates/aom-dsp/src/intra/simd16.rs`. Three bodies, each ONE
`#[magetypes(define(u16x16), v3, neon, wasm128, -scalar)]` function, so the AVX2,
NEON and WASM tiers come from the same source — §6b's condition for a claim that
travels. `super::simd`'s dispatch entries consult the gate and fall through to
the unchanged `i32x8` body when it declines.

**`wasm128` is in the tier list on purpose and it is not decoration.** The
`i32x8` kernels it shadows carry a `wasm128` tier; leaving it off would have
silently de-vectorized WASM for every bd8 block — a performance regression
invisible to every gate in the repo. On aarch64 the extra tier is inert, and
provably so: the arm binaries rebuilt with it were **identical by sha256** to the
ones built without.

Two structural choices beyond the lane width, both of them libaom's:

* **Column chunk OUTER, row inner.** The above samples, the width weights and
  the entire `(256 - ww) * right` product are row-invariant. libaom hoists the
  two loads across rows (`intrapred_neon.c:2515-2585` keeps `top_v[]` and
  `weights_x_v[]` in registers); making the chunk the outer loop hoists the
  product as well.
* **Every block width is handled**, including `bw == 4`: the column-varying
  inputs are staged into a `[u16; 64]` array once per block, so a 16-lane load is
  always in bounds and a narrow block just leaves lanes idle. No width floor to
  tune, no second code path to test.

---

## 7. Gates

* **`simd16::tests::every_kernel_matches_the_scalar_core`** — the three dispatch
  entries against `simd.rs`'s scalar cores (the same cores the C differential
  reference uses), over **all 19 AV1 transform shapes**, eight edge shapes, tight
  AND padded stride, at every archmage token tier, with a `cells > 250`
  non-vacuity assertion.
  * Probes are **asymmetric on purpose** (playbook §1 / KB-12): a FLAT edge makes
    `above[c] == below` and `left[r] == right`, under which each half-term is
    independent of its weight — invariant under exactly the arithmetic being
    tested. Flat is kept only as a control; the other seven shapes carry the
    test.
* **`simd16::reach`, pinned counts, both directions** — over the worst-case edge
  at each bit depth the gates admit **19 of 19 shapes at bd8 and 0 of 19 at bd10
  and bd12**, plus `the_bound_is_on_the_data_not_the_bit_depth`. A bound that is
  sound but never fires is as useless as no path at all, and only a counted pin
  says which shipped.
* **`simd16::tests::the_lane_bounds_are_load_bearing`** (playbook §2) — the other
  side: the gates accept at `M*` and reject `M* + 1` (asserted
  **unconditionally**, since that half is arithmetic on the span and has no
  tier), and one sample over the bound makes all three kernels genuinely
  **diverge** from the scalar core. The divergence half is conditioned on
  `dispatch::scalar_forced()` — under `AOM_FORCE_SCALAR=1` the entry *is* the
  scalar core and cannot diverge from itself, the defect
  `dir_simd::the_tap_bound_is_load_bearing` hit on its first scalar-pinned run.
  The over-bound sample goes in the *span*, never in the `below`/`right` corner
  scalars: those are folded into a per-row product **before** the lanes, so an
  out-of-range corner trips the kernel's own host-side overflow check in a debug
  build instead of demonstrating a lane wrap. Both are correct rejections; only
  one is what the test is about.

### Bite proofs — and they are what proves the integration differential REACHES this code

The §1 requirement is not "a differential exists" but "the differential can reach
the new code". Two perturbations, each applied alone, every test binary run
(`--no-fail-fast`). B is on the PAETH kernel and was taken while it was still in
the tree; it is kept here because it is the evidence that the bd10/12 test flips
with the *bound* rather than with the kernel.

| | **A** — SMOOTH's truncating halving add → the rounding one | **B** — PAETH's tie order `simd_le` → `simd_lt` |
|---|---|---|
| `simd16::every_kernel_matches_the_scalar_core` | **FAIL** | green |
| `simd16::paeth_matches_...` (then in tree) | green | **FAIL** |
| `build_nd_diff` (vs the C symbol) | **FAIL** | **FAIL** |
| `intra_simd_diff` (bd 8/10/12, dispatch vs scalar) | **FAIL** | **FAIL** |
| `intra_lowbd_diff` | **FAIL** | **FAIL** |
| `predict_intra_diff` | **FAIL** | **FAIL** |
| `highbd_diff` (**bd10/12 only**) | **green** | **FAIL** |
| `intra_diff` (the `u8` `predict` path) | green | green |
| `dir_simd_diff` / `dir_highbd_diff` | green | green |

The asymmetry is the result, not decoration. `highbd_diff` sweeps bd 10 and 12
only — exactly where the SMOOTH gate declines — so it **must** stay green under A
and **must** fail under B (PAETH's bound admitted those depths), and it does. The
`u8` and directional suites stay green through both, because neither family was
touched.

* **Full workspace, `--run-ignored all`, in BOTH dispatch modes** (SIMD live and
  `AOM_FORCE_SCALAR=1`) plus `cargo check --target x86_64-apple-darwin --workspace
  --all-targets` and `cargo check --target wasm32-unknown-unknown`. Counts in the
  `.meta`.
* **Gate 2 keeps zero pinned cells**: `config_permutations`, every `--cpu-used`
  0..9 cell byte-exact against real aomenc. The three
  `benchmarks/config_perm_*_2026-07-30.tsv` evidence sweeps regenerate identical
  apart from the commit stamp and the `ms` timing column (diffed column-wise: 0
  non-timing differing data rows in all three), so they are left as committed.

---

## 8. x86-64 and Windows

**RESOLVED on `windows-11-arm`, on BOTH contents, at 1.7x and 2.7x that band's
own noise floor — and the effect ORDERS with the census share, which is the
census making a prediction and the band confirming it.** Not resolvable on
`windows-latest` x86-64, which is a statement about the runner and is quantified
as such.

`winperf.yml` `arms: prepost` (`base_sha` = `5884f49`, the commit immediately
before this landing, so `post − pre` is exactly this lever and nothing else;
`l3a` and `l3` become extra COPIES of `pre`, so each band carries **three**
nulls). Run [30819647374](https://github.com/imazen/zenav1-aom/actions/runs/30819647374),
24 rounds, two runners x two contents, read with
`scripts/winperf_prepost_stats.py` (which pools the copies on each side —
the position correction §4 discusses). Bands committed as
`.win_<runner>_<content>.tsv`.

**Contents chosen from the census (§2), not from habit.** SMOOTH+PAETH is
**50.6 %** of predicted pixels on `detail` and **39.1 %** on `photo`, so
`detail` — the content that was a structural zero for KB-PERF-4 — is the
RICHEST content for this lever, and `photo` is the leaner control.

| | `windows-11-arm` `detail` | `windows-11-arm` `photo` | `windows-latest` `detail` | `windows-latest` `photo` |
|---|---:|---:|---:|---:|
| SMOOTH+PAETH share of predicted px | **50.6 %** | 39.1 % | 50.6 % | 39.1 % |
| worst null of identical copies (pre side) | +0.224 % | +0.302 % | +0.293 % | +0.282 % |
| worst null of identical copies (post side) | +0.361 % | +0.168 % | +0.415 % | +0.425 % |
| **`post` vs `pre` (this landing)** | **−0.961 %** | **−0.512 %** | −0.058 % | **+0.362 %** |
| effect / noise floor | **2.66x** | **1.70x** | 0.14x | 0.85x |
| sign test, rounds post-side faster | **22/24, p < 0.0001** | **22/24, p < 0.0001** | 14/24, p = 0.54 | 4/24, p = 0.0015 |

**On `windows-11-arm` the richer content gives the bigger effect** — −0.961 % on
`detail` (50.6 % of predicted pixels in the lever's mode family) against
−0.512 % on `photo` (39.1 %), measured on the same VM in the same job minutes
apart. That is the census's own prediction, taken before the run, coming back
confirmed. Darwin's −0.38 % on the study photograph (43.6 %) sits between them,
on a different CPU, so only the ORDERING within the ARM runner is a controlled
comparison; the cross-platform magnitudes are not.

**The allocator census is identical to the digit on every arm on both runners**
(488 750 calls / 296 669 580 bytes on `detail`, 374 603 / 252 359 139 on
`photo`), with `peak_live` differing by **one or two bytes out of 17.4 MB** —
proof that this moves arithmetic and not allocation, and that the arms are the
arms they claim to be. Every arm on every target codes the same frame
(8 734 bytes on `detail`, 5 301 on `photo`).

### `windows-latest` x86-64: not resolvable, and one figure that needs naming

`detail` — the content with MORE of the lever's mode family — comes back a flat
**−0.058 % at 14/24 rounds (p = 0.54)**, i.e. nothing, at 0.14x its own floor.
On the same runner in the same job, `photo` comes back **+0.362 %** with
**4/24 rounds faster (p = 0.0015)** — the wrong sign, significant by the sign
test, but **at 0.85x that band's own noise floor** and comfortably under the
0.50-0.86 % MDE this runner was measured at
([`winperf_content_census_2026-08-03.md`](winperf_content_census_2026-08-03.md)
§5).

**Reported rather than buried, and read as follows.** If the AVX2 tier were
genuinely slower, the effect would have to be *larger* on `detail`, which has
30 % more of the mode family — and `detail` is a flat null. Two contents on one
VM disagreeing in sign, with the richer one showing nothing, is the signature of
the runner rather than of the kernel. But this is an argument, not a
measurement: **the honest statement is that `windows-latest` cannot resolve a
0.4 %-class effect at n=24, that its `photo` band leans the wrong way inside its
own floor, and that a re-run at higher `rounds` on that runner is the open
item.** The AVX2 tier is otherwise compiled (`cargo check --target
x86_64-apple-darwin`) and differentially tested (the `v3` tier is in the token
permutations `simd16`'s differential runs).

This is the first lever in the sequence whose mode family the winperf harness
could see on BOTH of its established contents — KB-PERF-4's could not be seen on
`detail` at all — and §2 shows that was checked before the run rather than
discovered afterwards.

---

## 9. What is NOT measured here

* **SMOOTH_V and SMOOTH_H ship with a correctness gate and NO timing evidence.**
  The census says the encoder picks them **zero** times at this cell, on the study
  photograph and on all three winperf contents. They are in the landing because
  they are the same twenty lines and the same audited bound as SMOOTH; this file
  claims nothing about them.
* **One cell, one image, one preset.** cpu-used 0/3/4/9 are not measured; at a
  preset that picks different block sizes the width distribution moves and the
  lever's value with it.
* **8-bit 4:2:0 only.** The gate declines bd10/bd12 by construction, so the narrow
  path is unmeasured there because it never runs.
* **The cost of the gate's DECLINE is folded into the reported win, not
  isolated** — no arm was built with the scan present and the vector path
  disabled. KB-PERF-4 measured that cost for its own gate at about **+0.15 %** on
  content its kernel never reaches; this gate declines almost nothing at this
  cell, so the same measurement is not available here.
* **Wall clock only.** No instruction count (no valgrind on Apple Silicon), so
  "half the vector ops" and "0.38 % less time" are two different measurements and
  only the second is what the ratio moved on.
* **Single-threaded, one frame.**
* **The box was shared** with two other agents for most of the session; §3 says
  what that did to the spreads, and it is why three bands are reported instead of
  one.
* **The lowbd `u8` predictors (`intra::predict`) are untouched**, so the decoder's
  bd8 path gets nothing from this change — it is an encoder lever, and the encoder
  holds its planes as `u16` at every bit depth.
* **Linux is unmeasured.**
* **`windows-latest` x86-64 is measured and NOT resolved**, and its `photo` band
  leans +0.36 % — inside its own floor, but the wrong sign. §8.
* **The rest of the predictor class is untouched**: PAETH (+0.87 ms, §4), the edge
  filter (+0.81), the DC/V/H fills (+0.74), CFL (+1.40). The DC/V/H fills are
  already memset/memcpy slice ops, so that row is a *dispatch and call* cost
  rather than a kernel one and needs a different lever entirely.
