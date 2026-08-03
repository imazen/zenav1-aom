# An i16-lane FORWARD transform path — 3.274x → 3.192x, and the ceiling was 5x optimistic (2026-08-02)

Lever 2 of [`encoder_hotspot_reprofile_2026-08-02.md`](encoder_hotspot_reprofile_2026-08-02.md)
(**+19.63 ms, 16.6 %** of the 118.37 ms encoder gap), and the half of the bd8
lane-width programme that is genuinely **cross-platform**: i16 lanes help NEON
and AVX2 alike, unlike that ranking's ARM-only rank-1 CNN lever — see its
CROSS-PLATFORM SCOPING banner and
[`winperf_windows_2026-08-02.md`](winperf_windows_2026-08-02.md).

**Delivered −3.84 ms against a projected −19.6 ms.** Same lesson class as
KB-PERF-2's 18x-optimistic allocation ceiling, and for the same structural
reason (playbook §14): the ranked row was a *stage total*, and the mechanism
this change removes is only part of it. The mechanism is named and measured
below rather than inferred.

Provenance, box, exact commands and what is *not* measured:
[`encoder_i16_fwd_2026-08-02.meta`](encoder_i16_fwd_2026-08-02.meta). Data:
`.ab.tsv` (the 24-round band), `.halfbatch.tsv` (the rejected extension), and
the audit output [`i16_fwd_audit_2026-08-02.txt`](i16_fwd_audit_2026-08-02.txt).

---

## Control band — read this before reading any delta

24 rounds, one invocation of each of **seven** arms per round, interleaved
(`scripts/eprof_ab.sh`), each invocation 2 warm-up + 7 timed encodes with its
own median. 1024x1024 photo, cq 44, cpu-used 6. Box load average 1.4-2.1 of 12
cores.

| arm | median | min | max | spread | bytes |
|---|---:|---:|---:|---:|---:|
| `base` (590e525) | 154.474 ms | 152.930 | 155.287 | 1.54 % | 4472 |
| **`baseB` (null — a 2nd copy of `base`)** | **154.405 ms** | 153.063 | 155.591 | 1.65 % | 4472 |
| `col` (i16 column pass only) | 152.773 ms | 151.541 | 153.967 | 1.60 % | 4472 |
| `row` (i16 row pass only) | 152.808 ms | 151.505 | 153.446 | 1.28 % | 4472 |
| `both` (as it ships) | 150.630 ms | 149.374 | 151.423 | 1.37 % | 4472 |
| **`bothB` (null — a 2nd copy of `both`)** | **150.857 ms** | 149.596 | 151.430 | 1.23 % | 4472 |
| `libaom-c` | 47.187 ms | 46.597 | 47.616 | 2.19 % | 4472 |

**The noise floor is measured, not asserted, and on BOTH sides**: `baseB` vs
`base` is **−0.04 %** (paired-median −0.06 %) and `bothB` vs `both` is
**+0.15 %** (paired +0.16 %). Every delta below clears that by 6-16x.

Every arm emits the same 4472-byte stream, and the five port arms and the C
oracle produce the same `.obu` **by sha**, not merely the same length.

> A first 12-round band put the post-side null at +0.70 %, which is a quarter
> of the effect. It is recorded here rather than dropped: 12 rounds on this box
> was not enough to resolve a 2.5 % effect cleanly, and the fix was more rounds
> plus a null on the pre side as well. Playbook §6's "a single row proves
> nothing" applies to a single *band* too.

---

## The result

| | vs `base` | paired-median | ratio vs libaom-c |
|---|---:|---:|---:|
| null (`baseB`) | −0.069 ms (−0.04 %) | −0.06 % | 3.2722x |
| **`col` alone** | **−1.700 ms (−1.10 %)** | **−0.99 %** | 3.2376x |
| **`row` alone** | **−1.665 ms (−1.08 %)** | **−1.16 %** | 3.2384x |
| **`both`** | **−3.843 ms (−2.49 %)** | **−2.56 %** | **3.1922x** |
| null (`bothB`) | −3.616 ms (−2.34 %) | −2.40 % | 3.1970x |

**Ratio 3.2737x → 3.1922x**, i.e. **−0.082 on the ratio**, and the paired
per-round ratios do not overlap: `base` spans 3.244-3.305 and `both` spans
3.172-3.221 across all 24 rounds.

**The two passes are measured separately even though they ship together**
(playbook §14's closing rule). They are close to equal — −1.70 ms and
−1.67 ms — and they compose to **more** than their sum: 1.700 + 1.665 = 3.365
against 3.843 measured together, a **+0.48 ms** super-additivity that is 3x the
noise floor. Not explained here. The plausible mechanism is that the column
pass's i16 output is what the row pass then re-reads, so the two together touch
less of the i32 `buf` per transform than either alone; that is a hypothesis, and
no instruction-count or cache measurement was taken to test it.

---

## Why −3.8 ms and not −19.6 ms: the mechanism, measured

The re-profile's `dsp:transform` row is a **stage total** and the i16 path
removes only part of it. Two measurements say which part:

**(a) The projected row was already partly spent.** The +19.63 ms forward-
transform figure was taken *before* lever 3a, which tiered the two forward
passes' scratch arrays and returned **−3.13 ms**
([`encoder_alloc_scratch_2026-08-02.md`](encoder_alloc_scratch_2026-08-02.md))
— all of it out of exactly these two functions, which were the top TWO
allocator/`memset` callers in the profile. So the forward transform's cost at
*this* change's baseline is nearer 16.5 ms than 19.6 ms before anything else is
argued.

**(b) Half the forward-transform calls are structurally ineligible.** Counted
exactly with a temporary counter in both pass entry points, one encode of the
profile cell (1024x1024 / cq 44 / cpu-used 6):

| vectorized dim | column: i16 | column: i32 | row: i16 | row: i32 |
|---|---:|---:|---:|---:|
| 8 | 0 | **39 816** | 0 | **33 671** |
| 16 | 35 169 | 0 | 34 049 | 0 |
| 32 | 7 218 | 0 | 7 218 | 0 |

**42 387 of 82 203 column-pass calls (51.6 %) and 41 267 of 74 938 row-pass
calls (55.1 %) take the i16 path.** The remainder are 8-wide / 8-tall, which a
16-lane batch cannot fill. (4-wide blocks appear in neither column: the i32
pass declines them before the counter, and they run scalar in both arms.)

And within the eligible calls the i16 path halves the *lane* cost and replaces
the i64 round trip in `half_btf` — it does not remove the loads, the stores, the
8x8 transposes or the driver plumbing, all of which are unchanged.

**So the honest form of the ceiling is: the lever's mechanism is "half the
vector width, and a cheaper `half_btf`, on the half of the calls whose
vectorized dimension is >= 16" — not "the dsp:transform row".** Named that way
it is a ~2.5 % lever, which is what it measured.

### The half-batch extension — built, measured NULL, not shipped

The obvious way to recover the other half is to run 8-wide / 8-tall blocks as a
HALF batch (8 live lanes, 8 idle). The a-priori argument is not silly: an
`i16x16` and an `i32x8` are both 256 bits, so a half-idle i16 batch runs the
*same* number of vector ops as the i32 pass's full batch, while `fbtf16` (a
widening multiply-accumulate that stays in i32) is much cheaper than
`prims::hb` (which widens to i64 and back). It was implemented in full — both
passes, differentials extended to `col_n`/`row_n` = 8, all green, reach 81 →
129 column cells and 73 → 113 row cells — and then measured:

| arm (24 rounds, same harness) | median | vs `base` | ratio |
|---|---:|---:|---:|
| `base` | 154.113 ms | — | 3.2875x |
| `both` (16-lane batches only) | 150.221 ms | −2.53 % | 3.2045x |
| `half` (+ half batches) | 150.212 ms | −2.53 % | 3.2043x |
| `halfB` (null) | 150.332 ms | −2.45 % | 3.2069x |

That band is also an **independent replication of the headline**: its own
`base` → `both` pair, taken later, on a different arm set, in a different time
window, is **−2.53 %** against the 24-round band's **−2.49 %**. Two bands, one
number — which is the check playbook §6 asks for and which the first 12-round
band failed.

**`half` vs `both` is −0.009 ms (−0.006 %)** against a same-binary null of
**+0.08 %**. That is a null, not a small win, and it is a null on a change that
moves 48 % of the column calls and 45 % of the row calls onto a different code
path. **Reverted**: it is added surface, added risk on well-tested cells, and
zero measured benefit. The reasoning above about op counts is therefore
*wrong somewhere* — most likely the partial group's array-build load and
single-half store give back what the cheaper butterfly wins — and it is
recorded as a refuted hypothesis rather than a caveat. The implementation is
preserved at `~/tmp/i16fwd/lowbd16_fwd_halfbatch.rs` for anyone who wants to
re-test it on x86 or at another cell.

---

## The audit — 11 of 12 kernels, and one honest rejection

`xtask/audit_i16_fwd.py` (`just audit-i16-fwd`) is the forward twin of
`audit_i16_safety.py`, and it asks a different question, because the two
directions are built differently:

* the INVERSE kernels `clamp_value(_, stage_range[i])` at every stage, so that
  audit is a **domain** question — is every value either an i16 clamp output or
  a bounded transient;
* the FORWARD kernels carry **no clamp at all**, so nothing bounds a value
  except the input, and the audit is a **bound** question: how large may
  `|input|` be before some intermediate leaves i16.

It propagates each value's EXACT linear form — `sum c_i * input_i + e` with
`|e| <= E`, coefficients as exact `Fraction`s (every denominator is a power of
two) — and reports `M*`, the largest `M` with `M * sum|c_i| + E <= i16::MAX` at
every value of the kernel. That single condition proves both things the vector
code needs: no `wrapping_*` in the scalar kernel actually wraps at `M*` (so i16
lane arithmetic computes the same integer), and every `half_btf` product is
`<= 2^28` with a pair sum `<= 2^29`, exact in the i32 accumulator a widening
multiply gives.

**The bound is tight, and that matters.** The sign vertex
`input_i = M * sign(c_i)` attains `M * sum|c_i|`, so `M*` cannot be improved by
a better argument. The loose triangle-inequality bound — treat each butterfly
operand as independently maximal — is 1.5-2x larger and **rejects fdct32's
column pass at bd8**, which is provably safe. Getting this right is the
difference between 11 kernels and 9.

| kernel | M* (min over cos_bit 10..13) | peak `sum\|c\|` | peak E | verdict |
|---|---:|---:|---:|---|
| fdct4 | **11 583** | 2.8286 | 0.50 | i16 |
| fdct8 | **5 791** | 5.6572 | 1.19 | i16 |
| fdct16 | **2 895** | 11.3145 | 2.83 | i16 |
| fdct32 | **1 447** | 22.6289 | 6.72 | i16 |
| fdct64 | **723** | 45.2578 | 15.89 | i16 |
| fadst8 | **6 419** | 5.1015 | 2.83 | i16 |
| fadst16 | **3 214** | 10.1907 | 6.72 | i16 |
| fidentity4 | **23 167** | 1.4143 | 0.50 | i16 |
| fidentity8 | **16 383** | 2.0000 | 0.00 | i16 |
| fidentity16 | **11 583** | 2.8286 | 0.50 | i16 |
| fidentity32 | **8 191** | 4.0000 | 0.00 | i16 |
| **fadst4** | **1-11** | **21 901** | 0.50 | **REJECTED — i32 path** |

`sum|c|` is exactly `N * sqrt(2) / 2` for the whole fdct family (an AV1 `fdctN`
is `sqrt(2)` x the orthonormal DCT-II, and the DC row is what binds), so
`M*(fdctN) = floor(46340 / N)`. That closed form is a useful check on the
propagator: it was not put in, it came out.

**fadst4 was rejected, and no bound was widened to admit it.** It works in a
PRE-SHIFT domain — its stage-1 values are `sinpi[j] * x` held UNSHIFTED, with
the `>> cos_bit` only at its four terminals — so `sum|c|` peaks at 21 901 and
`M*` is 1 to 11 depending on `cos_bit`. There is no useful input range at which
it narrows. It stays on the i32 path, which is what
`lowbd16_fwd::fwd_kernel_i16` records with a comment naming the reason.

### bd8 reach, from the audit and from the code

Pushing `|residual| <= 255` through the driver — `col_in = 255 << shift[0]`,
`row_in = round_shift(column_output_bound, -shift[1])` — the audit reports
**169 of 193 `(tx_size, tx_type)` cells reachable in the column pass and 166 of
193 in the row pass**; the 24 the column pass cannot reach are exactly the
fadst4 cells.

The shipped **shape** gate (`col_n % 16 == 0` / `row_n % 16 == 0`) is narrower
than the bound gate, and `reach::the_gate_fires_across_the_bd8_grid` pins what
survives both, measured on a worst-case `|residual| = 255` block run through
the real scalar column loop:

* **column: 81 of 81 shape-eligible cells fire.**
* **row: 70 of 73 fire.** The 3 declines are `TX_64X64`, `TX_32X64` and
  `TX_64X32` at `DCT_DCT` — exactly the cells the audit predicts, where the
  column output overshoots fdct64's `M* = 723` / fdct32's `1447`.

That test is a pin, not a report: a change that quietly narrows the gate fails
it rather than silently costing the lever.

---

## Gates

* **957/957 workspace tests pass** with `--run-ignored all` (the nightly tier
  included), SIMD dispatch live, 0 failed / 0 skipped, 732 s. **Gate 2 keeps
  zero pinned cells** — every `config_permutations` cell across `--cpu-used`
  0..9 is byte-exact against real aomenc.
* The three `benchmarks/config_perm_*_2026-07-30.tsv` evidence sweeps
  regenerated **identical** to the committed files apart from the commit stamp
  and the `ms` column (diffed with the timing column stripped: empty).
* Scalar-pinned (`AOM_FORCE_SCALAR=1`) full run, same tier.
* Per-kernel and per-pass differentials vs the scalar kernels over the **full
  admitted domain** (`[-M*, M*]`), at every `cos_bit` in 10..13, at every token
  permutation on both architectures — 24 of 25 permutations vector. Probes are
  dense-random or asymmetric boundary patterns and the pass differentials seed
  the two buffers with **distinct** sentinels, per KB-12: a flat block puts all
  the energy in the DC and cannot see a dropped permutation.
* `gate_bite::the_bound_is_load_bearing` pins the OTHER side of the gate
  (playbook §2): every kernel must genuinely **diverge** outside its `M*`, else
  the gate is decorative. It does, and it records the slack — observed first
  divergence at **1.06x-2.97x M\***, because `M*` is sound but not attained
  (the `E` term is a worst case no single input realises).
* `txfm2d_simd_perm_diff` gained a **bd8 residual arm**. Its existing forward
  arm is full-range i16, which is over `M*` for every kernel, so without the new
  arm the integration test could not reach this code at all — playbook §1, a
  test that cannot fail.
* **Bite proof, with the asymmetry**: dropping fdct4's output permutation (the
  KB-12 defect class) fails both new differentials — *"fdct4: rand rep0
  cos_bit=10 lane=0 row=1 ... left: 9209, right: 1453"* — while `lowbd16`'s two
  inverse i16 differentials and the i32 `simd::tests` differential stay green.
* The generated `fwd1d_v3_i16_gen.rs` reproduces byte-identically from
  `just gen-txfm1d`, and the scalar regeneration from the same extracted C
  reproduces the committed `txfm1d_gen.rs` apart from two `use` paths.

---

## x86-64 and Windows: MEASURED, and the lever is worth MORE there

The whole reason this lever was picked over the re-profile's rank 1 is that it
is cross-platform, so "it helps on AVX2 too" is the load-bearing claim. It was
taken rather than argued, on both Windows runners, in one dispatch of
`.github/workflows/winperf.yml` (which gained a generic **`arms: prepost`** mode
for this: `pre` = a whole-src checkout of `base_sha`, `post` = the dispatched
ref, plus a same-binary null on each side). 16 interleaved rounds per arm, two
contents, never averaged. Run
[30788058276](https://github.com/imazen/zenav1-aom/actions/runs/30788058276);
bands committed as `.win_<runner>_<content>.tsv`.

**In `prepost` mode the `l3a` and `l3` arms are additional COPIES of `pre`**, so
each band carries THREE nulls — two on the pre side and one (`postB`) on the
post side. Read them first:

| `detail`, 1024x1024 / cq 44 / cpu-used 6 | `windows-11-arm` | `windows-latest` x86-64 | Darwin M4 Pro |
|---|---:|---:|---:|
| null — `l3a` vs `pre` (copy of pre) | +0.20 % | +0.55 % | — |
| null — `l3` vs `pre` (copy of pre) | +0.08 % | −0.17 % | — |
| null — `postB` vs `post` (copy of post) | +0.34 % | −0.80 % | +0.15 % |
| **`post` vs `pre` (this landing)** | **−7.43 %** (paired −7.47 %) | **−2.75 %** (paired −3.15 %) | −2.49 % |
| raw medians (pre → post) | 394.36 → 365.04 ms | 415.19 → 403.78 ms | 154.47 → 150.63 ms |
| band spread | 0.5-1.2 % | **5.9-8.0 %** | 1.2-1.7 % |

| `smooth` | `windows-11-arm` | `windows-latest` x86-64 |
|---|---:|---:|
| nulls | +0.14 % / −0.15 % | −0.39 % / −0.39 % |
| **`post` vs `pre`** | **−7.22 %** (paired −7.29 %) | **−4.32 %** (paired −4.19 %) |
| raw medians | 253.34 → 235.05 ms | 246.06 → 235.44 ms |

**Both Windows targets win, and both win by more than Darwin.** The claim is no
longer an argument. Three qualifications, stated rather than glossed:

* **`windows-latest` is noisy** — 5.9-8.0 % raw spread on `detail` and up to
  24 % on `smooth`, against 0.5-1.4 % on `windows-11-arm`. Its −2.75 % clears a
  ±0.8 % null but not by the margin the ARM runner's −7.4 % clears its ±0.3 %.
  The honest x86-64 statement is **"−2.8 % to −4.3 % depending on content, on a
  runner whose noise floor is ~0.8 %"** — a real win, not a precise one.
* **`windows-11-arm`'s −7.4 % is 3x Darwin's −2.5 %, on the same architecture**,
  which no mechanism in this change explains. Both are aarch64 running the same
  NEON kernels. The likely confounder is the harness content: winperf uses the
  integer-generated `detail`/`smooth` sources (its runners cannot ship the study
  photograph), and KB-PERF-2 already measured content moving a lever's value by
  more than 2x on ONE box. Cross-platform *ordering* here is sound; the
  cross-platform *ratio* is not, because platform and content are confounded.
* **This is not an allocation lever, and the census proves it.** Every arm's
  allocator census is identical to the digit on both runners apart from
  `peak_live`, which differs by **1 byte** out of 17.4 MB. That is the
  KB-PERF-2 comparison exactly inverted: that lever moved 346 888 calls and no
  arithmetic; this one moves no calls and only arithmetic. It is also why this
  one travels — playbook §6b: levers whose mechanism is a call into the platform
  have a per-platform price, levers whose mechanism is arithmetic do not.

All four arms encode 8748 / 2316 bytes on both runners, asserted fail-loud in
the job.

`cargo check --target x86_64-apple-darwin --workspace --all-targets` also
passes, but that only ever proved the AVX2 tier compiles; the numbers above are
what makes the claim.

---

## What is NOT measured here

* **One cell, one image, one content class.** cpu-used 3/4/5/9 are not measured
  — the re-profile records 9 (5.64x) and 4 (7.76x) as worse cells, neither
  decomposed. The call-routing census above is likewise one cell: at a preset
  that picks different transform sizes, the 51.6 % / 55.1 % eligibility would
  move, and the lever's value with it.
* **8-bit 4:2:0 only.** The gate is a runtime bound, so bd10/bd12 blocks simply
  decline; the cost of that decline (one max-abs scan per block that then takes
  the i32 path) is not measured.
* **No instruction count** (no valgrind on Apple Silicon), so "half the lane
  width" and "less time" are two different measurements and only the second one
  is what the ratio moved on. The +0.48 ms super-additivity of the two passes is
  unexplained for the same reason.
* **Single-threaded, one frame.**
* **The Windows runs use synthetic content, the Darwin runs a photograph**, so
  the platform comparison in the x86-64 section is confounded with content and
  only its ORDERING should be read. `winperf` cannot ship the 1.5 MB study
  photograph; that is a property of the harness, not of the encoder.
* **The gate's own cost is not isolated.** Each pass pre-scans its input for
  `max|lane|` before deciding, and no arm was built with the scan present and
  the i16 path disabled, so the scan's cost is folded into the reported win
  rather than reported separately.
