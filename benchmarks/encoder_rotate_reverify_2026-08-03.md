# Three Darwin headlines, re-taken with the arm order ROTATED (2026-08-03)

`ROTATE=1` landed in `scripts/eprof_ab.sh` **after** KB-PERF-2, KB-PERF-3 and
KB-PERF-4 were measured, so all three headlines came out of a **fixed-order**
interleave — and a fixed-order interleave confounds ARM with POSITION
([`DIFFERENTIAL_PLAYBOOK.md` §6](../docs/DIFFERENTIAL_PLAYBOOK.md)): two copies
of ONE identical binary at round positions 5 and 6 once came out **0.34 pp
apart**, which is **45 % of KB-PERF-4's whole published effect**. This file
re-takes all three under rotation and says, for each, whether the number
survived.

**No encoder source was touched.** Every arm is a `--release` build of an
already-landed commit; `git diff main -- crates/` over this session is empty.
Provenance, binary sha256s, exact commands, what is NOT measured:
[`.meta`](encoder_rotate_reverify_2026-08-03.meta).

| lever | published (fixed order) | rotated here | verdict |
|---|---:|---:|---|
| **KB-PERF-2** allocation | −3.05 % paired-median, n=12 | **−2.99 % / −3.00 %**, n=50 | **SURVIVES** — to 0.06 pp |
| **KB-PERF-3** i16 fwd transform | −2.56 % paired-median, n=24 | **−1.89 % / −2.16 %**, n=50 | **MOVES ~0.5 pp, conclusion holds** |
| **KB-PERF-4** directional intra | −0.75 % paired-median, n=36 | **−0.65 % / −0.62 %**, n=150 | **SIGN survives; MAGNITUDE NOT re-verified** |

**And the control does not reproduce the 0.34 pp that prompted this.** Six
copies of one binary, 36 rounds each way, agree to **0.22 pp rotated** and
**0.17 pp fixed** — both inside their own MDE, so on this box in these windows
the position penalty is not resolvable at n=36 either way. What rotation
unambiguously buys is that the position effect becomes **measurable** (§1)
instead of sitting invisibly inside each arm's number.

> **BOX CAVEAT, before any number.** The box was shared with two other agents
> for the whole session and their load is bursty: `uptime` ran 2.9–13.0 and one
> neighbour's `config_permutations` test at ~960 % CPU inflated encodes from
> ~150 ms to ~250–470 ms *inside* three of the bands. Raw per-arm spreads run
> from 3.9 % (the quiet control) to 212 % (the KB-PERF-4 n=150 band). Every
> figure below is a **paired per-round** statistic read against **same-binary
> null arms measured in the same band**, and the sign test is quoted throughout
> because that tail makes a parametric MDE optimistic-looking and fragile at the
> same time. Load is stamped before and after every band in the committed
> `<band>.env` files.

---

## 0. The reading rule, pre-registered

Written to `~/tmp/reverify2/PREREGISTERED.md` while the first lever band was at
round 17 of 50 and **before any lever result was read**, so it cannot have been
fitted to an outcome (§14). Verbatim:

> **When a rotated band is allowed to be called "resolving"** — both must hold:
> (1) the paired-**mean** MDE at 95 % (`1.96·sd/√n`) is **≤ half the published
> Darwin headline** — KB-PERF-2 ≤ 1.455 pp, KB-PERF-3 ≤ 1.245 pp, KB-PERF-4
> ≤ 0.375 pp; (2) **both** same-binary null arms in that band have |paired mean|
> **smaller than the measured post-vs-pre effect**.
>
> **If it does not hold** — extend ROUNDS on the same arms, same binaries, same
> cell, until it does, or report the lever **unresolved on this box in this
> window**. Nothing else may change: not the content, not the cell, not the arm
> set, not the statistic. Extra rounds are a *new, longer band*, not rounds
> appended to a short one that already read badly.
>
> **The headline statistic** is the paired **mean**, with the paired median and
> the sign test beside it... The originals published paired **medians**, so both
> are reported side by side.

It fired once, on KB-PERF-4, and §4 is what it produced — including the part
where the escalation **still did not clear the gate**, which is reported as a
failure to re-verify rather than quietly re-scored on a friendlier statistic.

---

## 1. The control band, first (§6) — SIX copies of ONE binary

The 0.34 pp observation came from the two copies that happened to sit at
positions 5 and 6 of one band. This measures the thing directly: **six files
with one sha256**, so *every* arm-to-arm difference is position plus noise and
nothing else. Run twice back to back on the same six copies, rotated then fixed.

| | rounds | window | load | per-arm medians | worst pairwise paired-median | occupancy |
|---|---:|---|---|---|---:|---|
| **rotated** | 36 | 20:30–20:35 | 9.8 → 2.9 | 151.190–151.411 ms | **0.215 pp** | exactly 6/6/6/6/6/6 |
| **fixed** | 36 | 20:35–20:41 | 2.9 → 4.6 | 151.295–151.499 ms | **0.169 pp** | one position per arm |

Every pairwise paired-median against `n1`, both bands:

| | n2 | n3 | n4 | n5 | n6 |
|---|---:|---:|---:|---:|---:|
| rotated | −0.150 | +0.065 | −0.125 | +0.016 | −0.021 |
| fixed | +0.001 | −0.046 | −0.160 | +0.007 | +0.009 |

**Neither band reproduces a 0.34 pp position penalty, and neither band could
have**: their MDE95 runs 0.19–0.47 pp, i.e. the same size as the effect they are
being asked to resolve. The honest statement is *not* "the confound is gone" —
it is **"at n=36 on this box today the confound is not resolvable in either
direction, and it is not currently large."**

**What rotation does buy, unambiguously.** Under `ROTATE=1` the position effect
stops hiding inside the arm estimates and becomes a directly readable row:

```
POOLED         +0.035   -0.334   +0.027   +0.028   +0.143   +0.102
  pooled position gradient (max-min) = 0.477 pp
```

Under `ROTATE=0` the same tool prints, correctly, that the quantity **cannot be
estimated at all** — position and arm are perfectly aliased, there is no
residual to look at, and the effect is entirely inside each arm's number. That
asymmetry, not a smaller null, is the argument for rotation.

**Today's position gradient, from all four rotated bands** (each arm normalised
to its own mean, so arm effects cancel):

| band | rounds × arms | pooled gradient |
|---|---|---:|
| control (identical binaries) | 36 × 6 | **0.477 pp** |
| KB-PERF-3 | 50 × 5 | **0.353 pp** |
| KB-PERF-2 | 50 × 5 | **1.279 pp** |
| KB-PERF-4 | 50 × 5 | 1.276 pp |
| KB-PERF-4 | 150 × 5 | **1.309 pp** |

**0.35–1.31 pp**, tracking load, against the 1.7 % recorded on a heavily
contended band and 0.1 % on a quiet one in
[`encoder_intra_smooth_paeth_2026-08-03.md`](encoder_intra_smooth_paeth_2026-08-03.md)
§4. So the concern that motivated this re-verification was **well founded** —
the gradient is the same order as KB-PERF-4's entire effect — even though the
specific 0.34 pp figure sits inside today's noise.

---

## 2. KB-PERF-2 (allocation levers 3a + 3) — SURVIVES

Base `578653f` → post `99a10ab`. Published: 5 arms × **12** fixed-order rounds,
no null arm, `base` 159.594 → `final` 154.953 ms, **−2.91 % wall / −3.05 %
paired-median**, ratio 3.3457x → 3.2484x.

Re-taken at 5 arms × **50** rotated rounds with nulls on both sides:

| arm | median | spread | paired-median vs `base` | paired-mean | faster | p (sign) |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 165.127 ms | 13.95 % | — | — | — | — |
| **`baseB` (null)** | 164.841 ms | 41.02 % | **−0.050** | +1.033 | 27/50 | **0.67** |
| **`final`** | 160.205 ms | 31.34 % | **−2.986** | −2.486 | 46/50 | **<0.0001** |
| **`finalB` (null of `final`)** | 160.248 ms | 18.01 % | **−3.004** | −2.526 | 45/50 | **<0.0001** |
| `libaom-c` | 49.121 ms | 22.35 % | — | — | — | — |

**Published −3.05 % against rotated −2.986 % / −3.004 % — agreement to 0.06 pp,
and the two post-side copies agree with each other to 0.018 pp.** MDE95 0.941 ≤
the pre-registered 1.455; the null is 0.05 pp on the median. **Resolving, and
the number is unchanged.**

Ratio against `libaom-c`, paired per round: **3.3537x → 3.2517x / 3.2522x**
(−0.102 on the ratio), against the published 3.3457x → 3.2484x (−0.097).

Its fixed-order twin, same binaries, same n, 14 minutes later: null −0.113,
`final` **−3.115** (48/50), `finalB` **−2.987** (46/50) — i.e. fixed and
rotated land on top of each other for this lever. Nothing to correct.

---

## 3. KB-PERF-3 (the i16 forward transform) — SURVIVES, magnitude MOVES

Base `590e525` → post `7976c0f`. Published: 7 arms × **24** fixed-order rounds,
nulls both sides (−0.06 % / +0.16 %), `base` 154.474 → `both` 150.630 ms,
**−2.49 % wall / −2.56 % paired-median**, ratio 3.2737x → 3.1922x.

Re-taken at 5 arms × **50** rotated rounds:

| arm | median | spread | paired-median vs `base` | paired-mean | faster | p (sign) |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 156.474 ms | 18.17 % | — | — | — | — |
| **`baseB` (null)** | 156.805 ms | 19.29 % | **−0.086** | −0.139 | 29/50 | **0.32** |
| **`both`** | 153.820 ms | 15.28 % | **−1.893** | −2.014 | 47/50 | **<0.0001** |
| **`bothB` (null of `both`)** | 153.372 ms | 9.49 % | **−2.163** | −2.454 | 48/50 | **<0.0001** |
| `libaom-c` | 48.680 ms | 10.49 % | — | — | — | — |

MDE95 0.403 ≤ the pre-registered 1.245, null 0.09 pp: **resolving**. The effect
is 20–25x its own null and 47–48 of 50 rounds are faster, so **the lever is not
in doubt**. What moved is the size: **published −2.56 %, rotated −1.89 % /
−2.16 % (mean of the two copies −2.03 %)**, i.e. about **0.5 pp smaller**.

**How much of that 0.5 pp is the protocol is NOT established here, and the file
says so rather than crediting it.** Three reasons for restraint:

* the two post-side copies of the *same binary* differ by **0.27 pp** on the
  median inside this very band — a majority of the shift, from copy noise alone;
* the fixed-order twin taken 11 minutes later reads **−2.541 / −2.344**, i.e.
  0.4–0.5 pp *larger* than rotated in the same session, same binaries — which is
  the direction the confound predicts (`base` sits at position 1 and the post
  arms at 3–4, and later positions run faster), but a single pair of bands in
  two different load windows cannot separate protocol from window;
* the ratio statistic, which is paired against the C oracle inside each round
  and is the most drift-robust number here, moves much less: **3.2232x →
  3.1604x / 3.1517x** (−0.063/−0.072) against the published −0.082.

**Correction applied**: the record and KB-PERF-3 now carry both readings with
their protocols named, rather than the fixed-order number alone.

---

## 4. KB-PERF-4 (directional intra) — the rule fired, and the escalation did not clear it either

Base `0279544` → post `71c924a`. Published: 6 arms × **36** fixed-order rounds,
nulls both sides (−0.01 % / −0.01 pp), `base` 149.188 → `all` 148.062 ms,
**−0.75 % wall / −0.75 % paired-median**, ratio 3.2115x → 3.1872x. This is the
lever the 0.34 pp position observation was worth **45 %** of.

**Band 1, 50 rotated rounds — FAILED the pre-registered gate.** A neighbour's
test suite landed inside it (raw spreads 64–72 %, encodes to 262 ms):
`all` −0.543 median / 32 of 50 faster / p = 0.065, null `baseB` +0.286 — the
null is over half the effect, and **MDE95 1.155 pp against the required
0.375**. Per §0 that is an escalation in ROUNDS and nothing else.

**Band 2, 150 rotated rounds, same arms, same binaries, same cell.** The box was
*worse*, not better (load 10.8 → 11.6, spreads 106–212 %, one encode at 470 ms):

| arm | median | spread | paired-median vs `base` | paired-mean | faster | p (sign) |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 156.342 ms | 106.38 % | — | — | — | — |
| **`baseB` (null)** | 156.564 ms | 112.71 % | **+0.095** | +0.533 | 71/150 | **0.57** |
| **`all`** | 155.305 ms | 211.60 % | **−0.648** | +0.087 | **115/150** | **<0.0001** |
| **`allB` (null of `all`)** | 155.301 ms | 164.36 % | **−0.623** | +0.000 | **108/150** | **<0.0001** |
| `libaom-c` | 49.221 ms | 91.74 % | — | — | — | — |

**MDE95 1.475 pp — the gate is still not met, and at sd = 9.2 it would take
~2 300 rounds (≈4.5 h) to meet it while the box stays this loaded.** So, stated
plainly:

* **the MAGNITUDE of −0.75 % is NOT re-verified.** It is not refuted either; it
  is unmeasured to the precision that was pre-registered for it. Anyone quoting
  −0.75 % should know it rests on the original fixed-order band and on the two
  paired medians here, not on a band that cleared its own resolution test.
* **the SIGN and the EXISTENCE of the effect DO survive**, on the one statistic
  that is valid under this tail: **115/150 and 108/150 rounds faster,
  p < 0.0001 on both post copies, against a null at 71/150, p = 0.57.** The two
  post copies agree to **0.025 pp** on the median (−0.648 / −0.623) while the
  null sits at +0.095 — three numbers that are hard to produce by accident.
* **six independent post-vs-base comparisons across three bands are all
  negative** — rotated n=50 (−0.543 / −0.496), fixed n=50 (−0.720 / −0.631),
  rotated n=150 (−0.648 / −0.623) — with the four from the two larger-signal
  bands at p ≤ 0.001.

**Best available reading: −0.62 % to −0.65 % rotated against −0.75 % published,
so the lever is real and slightly smaller than recorded** — but that comparison
is between a resolving band and a non-resolving one and must not be quoted as a
correction. The ratio statistic agrees on the direction and the size of the
shift: **3.1785x → 3.1592x / 3.1588x** (−0.020) against the published −0.024.

Its fixed-order twin at n=50 read −0.720 / −0.631 (37/50 twice, p = 0.0009) —
larger than rotated, same direction as KB-PERF-3's fixed-vs-rotated gap, same
caveat about load windows.

**What would settle it**: one 150-round rotated band on a genuinely idle box.
That is ~20 minutes of wall time and is the single cheapest open item here.

---

## 5. Should `ROTATE` default ON? — YES, and it is now

Changed in this commit, with an occupancy guard. The reasoning, and the one
argument against that turns out not to hold:

* **"Rotation costs a reproducible ordering" is false as stated.** The rotation
  is `ARMS[(j + i - 1) % K]` — **deterministic**, not shuffled. The same command
  with the same `N` and the same arm list produces the same order sequence every
  time. There is no seed, no randomness, and nothing to record beyond what the
  `position` column already records. The reproducibility cost of flipping the
  default is therefore only that a *pre-2026-08-03* recorded command line, re-run
  today, now rotates — which the `.env`/`position` column makes visible rather
  than silent, and which `ROTATE=0` restores exactly.
* **The cost is zero.** Same binaries, same count of invocations, same wall time.
* **The failure mode it prevents is real and the same order as the effects being
  measured** — 0.35–1.31 pp today (§1), 1.7 % on a contended band on record.
* **Default-off means every band is confounded by default**, and the confound is
  invisible in the output: a fixed-order band cannot even *estimate* the
  quantity, so nobody reviewing one can tell whether it mattered. Default-on
  makes the position table appear in every band, which is how a reader notices a
  bad window.
* **The guard**: rotation only balances if `N % k == 0`; otherwise some arm
  spends an extra round in a favourable position and the confound comes back
  *partially and silently*. `eprof_ab.sh` now prints a loud `WARNING` naming the
  imbalance and the two nearest multiples of `k`. It warns rather than refuses,
  because an unbalanced rotated band is still strictly better than a fixed one
  and the occupancy column lets the reader check.

`ROTATE=0` remains available and is what to pass when reproducing a
pre-2026-08-03 band command-for-command.

---

## 6. What this file does NOT establish

* **Nothing about Windows.** The Windows re-measurements of these levers
  (`winperf_windows_2026-08-02.md`, `winperf_content_census_2026-08-03.md`) use
  their own harness and their own pooled-copy position correction; they are not
  re-run here and nothing above changes them.
* **Nothing about the per-half arms.** `l3a`/`l3`, `col`/`row` and `z1z3`/`z2`
  were not re-taken — each needs its own band and its own nulls, and none was
  ever the published headline.
* **No clean-box measurement.** Every band here ran against two other agents'
  builds and test suites. The one band that was quiet (the 36-round control) is
  the one that shows what this box looks like when it behaves: 3.9–8.0 % raw
  spread against the 212 % seen at 21:16.
* **No pooling across bands.** §0 forbids appending rounds to a band that
  already read badly, so the two KB-PERF-4 rotated bands are reported side by
  side and never averaged.
