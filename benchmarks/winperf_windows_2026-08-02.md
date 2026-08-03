# The allocation lever is worth ~5x more on Windows than on Darwin (2026-08-02)

**`-1.34 ms` was never the value of the change. It was the value of the change
on one allocator.**

[`encoder_alloc_scratch_2026-08-02.md`](encoder_alloc_scratch_2026-08-02.md)
landed two levers together and measured them on an Apple M4 Pro: the memset
lever (3a) at −3.13 ms and the **allocator** lever (3, −41 % of allocator calls)
at −1.34 ms — a fifth of the combined win, and the smaller of the two. Measured
on Windows, at the same cell, on the same harness, that ordering **inverts**:

| lever 3 (per-txb scratch reuse), `detail` content | paired-median | vs Darwin |
|---|---:|---:|
| Darwin, Apple M4 Pro, n=24 | **−0.49 %** | — |
| `windows-11-arm` runner, n=16 | **−2.38 %** | **4.9x** |
| `windows-latest` (x86-64) runner, n=16 | **−2.54 %** | **5.2x** |

and its share of the combined landing goes from **21 % on Darwin to 86–99 % on
Windows**. On Darwin the allocation programme is the junior lever; on Windows it
is essentially the whole thing.

**Nobody had ever run this encoder on Windows.** `ci.yml`'s `windows-11-arm` job
is `portability` — it *builds* the published crates and runs no tests and no
benches. Every Windows statement about this encoder was therefore unmeasured
rather than unreported, including this one. It is now a
`workflow_dispatch`-able job (`.github/workflows/winperf.yml`) so it stays
measurable.

Provenance, boxes, exact commands, and what is *not* measured:
[`winperf_windows_2026-08-02.meta`](winperf_windows_2026-08-02.meta). Data: six
`.tsv` bands + `.census.tsv` + `.darwin_photo_control.tsv` +
`.replication_run1.tsv`.

---

## Control bands — read these before reading any delta

Five arms interleaved, one invocation of each per round, each invocation 2
warm-up + 7 timed encodes contributing its own median. **`postB` is a second
copy of the identical `post` binary**: its delta is the runner's noise floor,
and it is what every other number below has to clear.

### `detail` content, 1024x1024 / cq44 / cpu-used 6 / ALLINTRA / 1 thread

| arm | Darwin M4 Pro (n=24) | `windows-11-arm` (n=16) | `windows-latest` x86-64 (n=16) |
|---|---:|---:|---:|
| `pre` (578653f) | 175.125 ms (4.14 %) | 405.533 ms (0.60 %) | 427.920 ms (3.43 %) |
| `l3a` (3a only) | 171.119 ms (3.62 %) | 403.592 ms (1.77 %) | 423.640 ms (2.96 %) |
| `l3` (lever 3 only) | 173.865 ms (3.44 %) | 396.582 ms (1.91 %) | 416.021 ms (1.84 %) |
| `post` (both) | 169.077 ms (3.38 %) | 395.151 ms (2.44 %) | 415.877 ms (1.92 %) |
| `postB` (null) | 169.275 ms (3.29 %) | 394.575 ms (1.77 %) | 416.258 ms (2.16 %) |
| **null, paired** | **+0.08 %** | **−0.20 %** | **−0.05 %** |

### `smooth` content, same cell

| arm | Darwin M4 Pro (n=24) | `windows-11-arm` (n=16) | `windows-latest` x86-64 (n=16) |
|---|---:|---:|---:|
| `pre` | 110.522 ms (1.39 %) | 253.007 ms (0.89 %) | 245.504 ms (12.83 %) |
| `post` | 110.187 ms (1.32 %) | 252.850 ms (1.42 %) | 245.141 ms (9.60 %) |
| **null, paired** | **+0.02 %** | **−0.06 %** | **−0.18 %** |

The hosted runners are **2.2–2.4x slower** than the dev box at this cell, and
`windows-latest` is much the noisiest of the three (up to 16 % spread on
`smooth`). The paired-median column is the load-robust statistic and is what the
deltas below quote; the raw medians are in the table above so the reader can see
what it is robust *to*.

---

## The deltas

Paired-median, against the null in the same column.

### `detail`

| | Darwin M4 Pro | `windows-11-arm` | `windows-latest` x86-64 |
|---|---:|---:|---:|
| null (`postB` vs `post`) | +0.08 % | −0.20 % | −0.05 % |
| **3a alone** (memset lever) | **−2.02 %** (−4.007 ms) | −0.50 % (−1.941 ms) | −0.77 % (−4.280 ms) |
| **3 alone** (allocator lever) | −0.49 % (−1.260 ms) | **−2.38 %** (−8.951 ms) | **−2.54 %** (−11.900 ms) |
| both | −3.31 % (−6.048 ms) | −2.67 % (−10.382 ms) | −2.85 % (−12.043 ms) |
| lever 3's share of "both" | **21 %** | **86 %** | **99 %** |

### `smooth`

| | Darwin M4 Pro | `windows-11-arm` | `windows-latest` x86-64 |
|---|---:|---:|---:|
| null | +0.02 % | −0.06 % | −0.18 % |
| 3a alone | −0.12 % | −0.15 % | −0.15 % |
| 3 alone | +0.31 % | −0.30 % | +0.48 % |
| both | −0.23 % | −0.16 % | −0.25 % |

**On `smooth`, nothing is resolvable on any platform.** Every arm sits within
about 2x of its own null, on all three boxes, in both directions. That is the
honest reading and it is reported as such: `smooth` removes only 69 240 of
291 704 calls (−23.7 %) where `detail` removes 346 888 of 835 638 (−41.5 %), so
there is roughly a fifth as much lever there to find.

### Replication

An earlier dispatch (run `30778499397`, n=12, `detail` only, a different pair of
runner VMs and a differently-linked pair of binaries) put lever 3 at
**−2.70 %** on `windows-11-arm` and **−4.79 %** on `windows-latest`
(`.replication_run1.tsv`). Same sign, same order, same conclusion; the x86-64
magnitude was inflated by that run's 7–20 % spreads, which is why the tighter
n=16 run is quoted above and the noisier one is quoted here.

---

## The allocator census is IDENTICAL on all three platforms

Exact counts, not timings — one `port_encode`, counting `GlobalAlloc`:

| content | arm | calls | bytes | per SB |
|---|---|---:|---:|---:|
| `detail` | `pre` / `l3a` | 835 638 | 564 518 284 | 3 264.2 |
| `detail` | `l3` / `post` | 488 750 | 296 669 580 | 1 909.2 |
| `smooth` | `pre` / `l3a` | 291 704 | 332 018 112 | 1 139.5 |
| `smooth` | `l3` / `post` | 222 464 | 234 443 712 | 869.0 |

**Every one of those numbers is byte-for-byte the same on `darwin-arm64`,
`aarch64-pc-windows-msvc` and `x86_64-pc-windows-msvc`** (`.census.tsv`). Only
`peak_live` differs, by 467 bytes out of 17.4 MB — std/allocator bookkeeping, not
encoder behaviour. So the platforms are doing **identical work** and the entire
difference measured above is **cost per allocator call**, not more calls.

That is also the answer to "does the call count differ from Darwin's 854 053 /
512 557?" — it does not differ *by platform*. It differs by **content**: this
harness's `detail` returns 835 638 → 488 750 (−41.5 %) against the study
photograph's 854 053 → 512 557 (−40.0 %), which is what `detail` was tuned to
match.

The census also proves the arms are the arms they claim to be, on every box:
`pre` and `l3a` agree **to the digit** (3a's scratch is a stack array and cannot
appear in a census) and `l3` and `post` agree to the digit. All four arms emit
8 734 / 2 302 coded bytes, asserted fail-loud in CI.

---

## Why the two levers rank differently on the two platforms

They remove different things, and the two platforms price those things
differently:

* **3a removes `memset` bytes** — a 4 KiB zero-fill per forward transform, i.e.
  memory-bandwidth work. On `windows-latest` it saves **−4.280 ms against
  Darwin's −4.007 ms — essentially the same absolute work**, and reads as a
  smaller *share* (−0.77 % vs −2.02 %) only because the denominator is 2.4x
  longer. `windows-11-arm` is the odd one out: **−1.941 ms**, less than half the
  absolute saving, which the denominator does not explain and this study does
  not either. Recorded as an open observation, not glossed.
* **3 removes 346 888 malloc/free pairs** — and Microsoft's heap charges
  materially more per call than Apple's. On Darwin that buys −0.49 %; on Windows
  −2.38 % / −2.54 %.

This is KB-PERF-2's own lesson arriving from the other side. That landing found
that a lever aimed at allocator *calls* could not collect the `memset` half of
the `alloc/libc` class. The same split explains this: **the two halves of that
class have different per-platform prices, so a lever aimed at one of them has a
different value per platform.** A single-box measurement of an allocator lever
does not generalise, and this one was off by 5x.

---

## Does lever 3's rank change on Windows? Yes.

`encoder_alloc_scratch_2026-08-02.md` closes by ranking the *residual*
allocation work at 16.4 % of the port/C gap and noting the delivered lever was
worth −1.34 ms. On Darwin that is a fair summary and the remaining allocation
work is correctly ranked below the CNN convolution, the forward transform and
the intra predictors.

On Windows it is not. Lever 3 alone returned **86–99 % of the combined
landing**, at 2.4–2.5 % of a *whole encode* — and the residual it left behind
(`finish_grow`, the retained `qcoeff`/`dqcoeff`, the surviving zeroing) is
priced on the same slower heap. **Any future ranking of allocation work should
carry a platform, and if it is going to carry only one, Windows is the one that
makes the lever look worth doing.**

What does NOT change: the levers are still worth landing, still byte-identical,
and 3a is still the better lever on the dev box. Nothing here argues for
reverting or re-ordering anything already shipped. It argues that the *number*
attached to lever 3 was scoped to a box nobody ships on.

---

## Content — SUPERSEDED IN PART, 2026-08-03

Everything below this heading is what was measured on 2026-08-02 and stands.
What has changed since is the **scope** the two contents can be quoted for.

`detail` and `smooth` were fitted on **allocator call count**, and this study is
an allocation study, so they are the right harness for it. They are not the
right harness for anything else, and on 2026-08-03 a directional-intra lever was
read against them and came back "not resolved" because the code under test
essentially never ran: directional predicted-pixel share is **20.8 %** on the
photograph, **13.2 %** on `smooth` and **0.15 %** on `detail`.

The harness now carries a **census** (`aom_dsp::census` +
`examples/content_census.rs`) and a **third content**, `photo`, fitted to the
photograph's intra mode distribution instead — plus a `contents:` input on the
workflow so a study picks the one its lever reaches. Full record, including the
list of mode families the harness *still* cannot see:
[`winperf_content_census_2026-08-03.md`](winperf_content_census_2026-08-03.md).

**That "still cannot see" list was itself revised the same day**
([`winperf_family_census_2026-08-03.md`](winperf_family_census_2026-08-03.md)),
and the revision matters for how any future band off this harness is read. The
census now names every coding-tool family the encoder can enter — filter-intra,
palette Y/UV, intraBC, CFL and the per-plane chroma path, leaf size to 4x4,
rect-vs-square leaves, tx type x size — and the five families the list called
blind spots turn out to have **three different causes**:

* **filter-intra is a SPEED zero, not a content zero.** At this study's
  `--cpu-used 6` the port sets `prune_filter_intra_level = 2`, so
  `rd_pick_filter_intra_sby` is never called and no source can reach it. The
  same content reads **10.46 %** of leaves at cpu-used 5.
* **palette and intraBC are KNOB zeros** (`--enable-palette` /
  `--enable-intrabc`, both default off) plus a header-bit gate real `aomenc`
  sets from its own screen detection. A fourth content, `winperf:screen`,
  reaches both once all three gates are open.
* **CFL is genuinely content-gated** and was never censused at all: 4.59 % of
  chroma-reference leaves on the photograph and 23.02 % on a CLIC image against
  **0.00-0.29 %** on all three contents here. A CFL lever read off `detail`,
  `smooth` or `photo` would be reading a structural zero.

Also relevant to every number on this page: the **chroma path is 28-47 % of
intra-predictor calls** on these contents and was uncounted before that
landing, so "intra prediction" totals quoted from the earlier census are
luma+chroma sums nobody had split. `just census-gate` now pins the whole
picture.

Two facts from that census that bear directly on the numbers above:

* the two contents' allocator call counts are unchanged and the tables above are
  unaffected — `photo` is an addition, not a replacement;
* `photo`'s own allocator traffic is **374 603 calls, 73 %** of the
  photograph's, against `detail`'s **95 %**. So `photo` is a *worse* harness for
  an allocation lever than `detail` is, exactly as `detail` is a worse harness
  for a mode-scoped one. The harness needs both, and which to quote is now a
  table lookup.

## Content, and the honest bit about it

The harness cannot ship the dev box's 1.5 MB photograph, so it generates its
source with integer-only arithmetic (bit-identical on every target — the census
above is the proof). `detail` was **tuned against the photograph** until it
provoked the same amount of encoder work: 95 % of its allocator calls, 102 % of
its wall time. It does not match its coded size (8 734 vs 4 472 bytes), which is
fine — size is not what drives per-txb allocation churn.

But content matters more than expected, so two contents ship and are never
averaged. Measured on the dev box, same day, same box, `.darwin_photo_control.tsv`:

| lever 3 on Darwin | paired-median |
|---|---:|
| study photograph, 2026-08-02 landing (load 30–41) | −1.03 % |
| study photograph, re-measured today (load ~2) | −1.26 % |
| `detail` synthetic | −0.49 % |
| `smooth` synthetic | +0.31 % |

The photograph re-measures at **−4.51 ms / −2.81 % for the combined landing
against the record's −4.641 ms / −3.05 %** on a box at load ~2 instead of load
30–41, so **the landing reproduces and background load is not what varies.**
Content is: lever 3 ranges from −1.26 % (photo) through −0.49 % (`detail`) to
+0.31 % (`smooth`) on one box in one afternoon. All of those are small, and all
of them are dwarfed by the platform effect.

An interim 12-round `detail` band put lever 3 at **+2.49 %** before the box was
quiet and before the binaries settled; the 24-round quiet-box band says
−0.49 %. That excursion is recorded here rather than dropped, because it is the
size of the thing a single 12-round band on this hardware can invent, and
because §6 says a single row proves nothing.

---

## What is NOT measured here

* **Only Windows and Darwin.** Linux (the third tier the crates ship to, and the
  one with yet another allocator) is not measured. `ubuntu-latest` would be a
  one-line matrix addition to the same workflow.
* **One cell.** 1024x1024 / cq44 / cpu-used 6. No size, quality or speed sweep,
  on any platform.
* **Anything that is not an allocation lever.** The two contents here are fitted
  on allocator call count and on nothing else; a lever scoped to a coding-mode
  family is unmeasurable on them (see the 2026-08-03 note above). This was not
  known when the study was written and is the one respect in which its harness
  claims were too broad.
* **Two synthetic contents plus one photograph, and the photograph only on
  Darwin.** Nothing here says what lever 3 is worth on a *photograph* on
  Windows. The platform ratio is measured on `detail`, where Darwin's own value
  (−0.49 %) is less than half the photograph's (−1.26 %) — so if the Windows
  boxes track Darwin's content ordering, the ratio on a photograph would be
  *smaller* than 4.9x/5.2x, not larger. That is an inference from one box's
  content ordering and is not claimed as a measurement; the only defensible
  statement is the one made above, which is a `detail`-content statement.
* **Hosted CI VMs, shared and unspecified.** The runner CPU model is not
  recorded (`wmic` is gone from these images and `systeminfo` does not name the
  part). Absolute milliseconds on those boxes are worth nothing on their own;
  only within-run, interleaved, null-referenced comparisons are used.
* **No instruction counts.** Wall clock and exact allocator-call counts are two
  independent measurements here as they were in the landing; neither derives the
  other.
* **Single-threaded, one frame, no allocator contention.** Windows' heap is
  known to be worst under multi-threaded contention, and this encode has none —
  so the multi-threaded gap is presumably larger than 5x and is entirely
  unmeasured.
* **`--no-default-features`** (no C oracle) on every arm, so there is no
  byte-exactness differential in this job. The 8 734 / 2 302 assertion says the
  arms agree with each other, not that they agree with libaom; the ~40
  differential tests that do that are unchanged and unaffected.
