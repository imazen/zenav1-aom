# Allocation levers 3a + 3 — landed, and worth a third of what the profile projected (2026-08-02)

**3.346x → 3.248x. 854 053 allocator calls → 512 557 (−40.0 %), 448.8 MB → 267.5 MB,
3 336 → 2 002 per superblock. Not one byte moved, in either dispatch mode,
including the `--run-ignored all` tier.**

And the headline finding is the shortfall, not the win.
[`encoder_hotspot_reprofile_2026-08-02.md`](encoder_hotspot_reprofile_2026-08-02.md)
ranked these two levers at **+5.30 ms** (3a) and **+24.76 ms** (3) of the
118.37 ms gap — 4.5 % and 20.9 %. Landed, they returned **−3.13 ms** and
**−1.34 ms**. The 3a projection was about right; the lever-3 projection was
**18x optimistic**, and the reason is legible in the profile itself: that
lever's "allocation" class is a **leaf class matched by name**, and most of its
mass is `memset`/`memcpy`, not allocator bookkeeping. Removing 341 496
malloc/free pairs does not remove the bytes those buffers still have to be
zeroed with.

Provenance, box, load, exact commands, and the honest list of what is not
measured: [`encoder_alloc_scratch_2026-08-02.meta`](encoder_alloc_scratch_2026-08-02.meta).
Data: `.ab.tsv` (control band), `.alloc.tsv` (census), `.callers.tsv`
(post-change caller attribution).

---

## Control band — read this before reading any delta

Twelve INTERLEAVED rounds, one invocation of every arm per round, each
invocation 2 warm-up + 7 timed encodes with its own median. 1024x1024 photo,
cq 44, cpu-used 6. Whole-box load average 30-41 of 12 cores throughout (the
same concurrent `zenmetrics` fleet both encoder profiles ran under), which is
why the arms are interleaved rather than run back to back.

| arm | median | min | max | spread | vs libaom-c | bytes |
|---|---:|---:|---:|---:|---:|---:|
| `base` (578653f) | 159.594 ms | 159.118 | 164.698 | 3.51 % | **3.3457x** | 4472 |
| `l3a` (3a only) | 156.465 ms | 155.551 | 160.487 | 3.17 % | 3.2801x | 4472 |
| `l3` (lever 3 only) | 158.254 ms | 157.586 | 163.002 | 3.44 % | 3.3176x | 4472 |
| **`final` (both)** | **154.953 ms** | 154.647 | 159.347 | **3.04 %** | **3.2484x** | 4472 |
| `libaom-c` | 47.701 ms | 47.321 | 49.577 | 4.77 % | — | 4472 |

Paired per-round ratios, `final` vs `libaom-c`: 3.126 3.130 3.160 3.176 3.216
3.248 3.248 3.265 3.268 3.271 3.274 3.284 — **spread 5.05 %**, the tightest of
the four arms. The base arm measures **3.3457x** here against the re-profile's
published **3.343x**, on a different day and a different background load: the
baseline reproduces, so the deltas below are read against a live control, not
against a quoted number.

**Total: −4.641 ms, −2.91 % (paired-median −3.05 %), 3.3457x → 3.2484x.**

`scripts/eprof_ab.sh` + `scripts/eprof_ab_stats.py` are new here.
`eprof_control.sh` interleaves exactly two fixed arms; a perf landing needs to
compare four port builds, and comparing separately-taken medians on this box is
what `docs/DIFFERENTIAL_PLAYBOOK.md` §6 exists to forbid (a same-binary re-run
has moved a row +16.5 % here).

---

## Per-lever bite proof — two levers, and they bite differently

Reverting each one alone, measured in the same interleaved run:

| | wall delta vs base | paired-median | allocator calls | bytes moved |
|---|---:|---:|---:|---:|
| **3a alone** (fwd-pass scratch tiering) | **−3.128 ms** (−1.96 %) | −2.13 % | **0** (854 053, unchanged) | 0 |
| **3 alone** (per-txb scratch reuse) | **−1.339 ms** (−0.84 %) | −1.03 % | **−341 496** (→ 512 557) | −181.4 MB |
| both | −4.641 ms (−2.91 %) | −3.05 % | −341 496 | −181.4 MB |

They move by different amounts on **both** axes, which is the §1 test for "two
roots" rather than "one fix spelled twice":

* **3a moves time and zero calls.** Its scratch is a stack array
  (`[i32x8; N]`), so it cannot appear in an allocator census at all — and the
  census proves it rather than asserting it: the `l3` census and the `final`
  census are **identical to the digit** (512 557 calls / 267 465 686 bytes), so
  adding 3a on top of lever 3 changes nothing an allocator counter can see.
  What it removes is a `memset`.
* **3 moves calls hugely and time barely.** −40.0 % of the allocator traffic for
  −0.84 % of wall.
* They are **additive**: 3.128 + 1.339 = 4.467 against 4.641 measured, 3.9 %
  apart — inside the control spread, so no interaction is claimed either way.

---

## Why lever 3 under-delivered — the class was misread, and the profile said so

The re-profile's allocation figure comes from a stage rule that matches leaf
symbols by name:

```
("alloc/libc", r"(malloc|free|realloc|calloc|bzero|memset|memcpy|memmove|xzm_|_platform_|...)")
```

`_platform_memset` alone was **5.36 %** of the port window (8.90 ms) and
`_platform_memmove` **2.68 %** (4.44 ms) against the allocator's own `xzm_free`
at **2.34 %**. So the 27.63 ms class was always *mostly bytes being written*,
and only a minority of it was the malloc/free bookkeeping that scratch reuse
removes. The re-profile itself flags this in "Attribution limits" item 4 — "the
`alloc/libc` and `os/setjmp` stages are leaf classes matched by name … caller
attribution is where the actionable number lives" — but its **ranked-lever
table still credits the lever with the whole class**, and that is the number
that was 18x off.

Measured after the change (same tooling, 60 s sample, 45 543 samples,
during-profile median 164.562 ms, `.callers.tsv`):

| | before | after |
|---|---:|---:|
| allocator+memset+memcpy leaf class, % of window | 16.65 % | **12.41 %** |
| the same class in ms | 27.63 ms | **20.42 ms** |
| ratio vs libaom-c's 2.87 ms | 9.63x | **7.11x** |
| its share of the port/C gap | 20.9 % | **16.4 %** |

The class self-cost fell **7.21 ms** while wall fell **4.64 ms**. Those are two
different measurements — a 60 s sample under load against a 12-round
interleaved control band, with denominators 1 % apart — and the honest
statement is the wall number. The sampled figure is quoted because it says
*where* the time went, not *how much*.

**Caller attribution, before → after** (% of the class):

| caller | before | after |
|---|---:|---:|
| `transform::simd::try_fwd_col_pass` | 9.90 % | **gone** (below the 22nd row) |
| `transform::simd::try_fwd_row_pass` | 9.28 % | **gone** |
| `transform::txfm2d::av1_fwd_txfm2d` | 7.62 % | 2.85 % |
| `tx_search::intra_model_rd_y` | 6.79 % | 3.22 % |
| `tx_search::txfm_rd_in_plane_intra` | 6.61 % | 5.64 % |
| `tx_search::search_tx_type_intra` | 6.07 % | 4.72 % |
| `xform_quant` | 5.87 % | 5.98 % |
| `encode_intra::encode_intra_block_plane_y` | 5.54 % | 7.59 % |
| `encode_intra::encode_intra_block_plane_uv` | 4.58 % | 7.38 % |
| `intra_uv_rd::txfm_rd_in_plane_uv_p` | 4.44 % | 3.10 % |
| `intra_uv_rd::predict_uv_txb` | 3.42 % | 4.12 % |
| `intra_uv_rd::intra_model_rd_uv` | 3.22 % | 1.96 % |
| `alloc::raw_vec::finish_grow` | 2.35 % | **8.85 % (now rank 1)** |

The two 3a callers vanishing is the caller-level bite proof for 3a. The three
rows that *grew* as a share are the ones whose remaining allocations are
**retained output, not churn**: `encode_intra_block_plane_{y,uv}` move their
`qcoeff`/`dqcoeff` into a per-txb `TxbEncode` the pack stage reads later, and
`xform_quant_into` still zero-fills those two buffers on every call. Scratch
reuse cannot touch either — the first is live data, the second is a `memset`
the byte-identical form keeps (below).

---

## What was built

**Lever 3a — tier the forward-pass scratch** (`aom-dsp/src/transform/simd/mod.rs`).
`fwd_col_pass`/`fwd_row_pass` opened with a flat `[i32x8::zero(t); 64]` x2 —
4 KiB of `memset` per forward transform at every size, including 4x4. Split each
into a tiering wrapper + a `#[magetypes]` `*_core` over caller-sized scratch,
tiered `{8, 16, 64}` on `row_n` (col pass) / `col_n` (row pass) — **the same
three-tier shape the inverse passes in the same file already had at `:430-443`,
`:784-797`, `:994-1000`**, for the reason `lowbd16.rs:132` states in so many
words. No new scheme was invented.

**Lever 3 — per-txb scratch reuse**, the decoder's `ReconScratch` /
`InvTxfmScratch` pattern applied encode-side:

* `aom_dsp::transform::txfm2d::FwdTxfmScratch` + `av1_fwd_txfm2d_into` — the
  forward twin of `InvTxfmScratch`, same `clear()` + `resize(n, 0)` refill.
* `aom_encode::XformQuantScratch` + `xform_quant_into` /
  `xform_quant_optimize_split_into` — `coeff`/`qcoeff`/`dqcoeff` + the forward
  buffer, one set instead of three `Vec`s per candidate tx type.
* `aom_encode::tx_search::{TxWalkScratch, TxSearchScratch, IntraTxScratch}` +
  `search_tx_type_intra_into` / `dist_block_px_domain_into` — the per-txb
  prediction, residual, `recon_intra` reconstruction, the winner's coefficients
  (C keeps those by buffer swap; the port now does one memcpy per improvement
  instead of two allocations per candidate), and the pixel-domain distortion's
  reconstruction buffer.
* **Owned one level up from where it is used**: `rd_pick_intra_sby_mode_y` owns
  one for its whole luma mode loop, `rd_pick_intra_sbuv_mode` one for the whole
  chroma mode loop, `PaletteRdState` one per palette search. Every mode x tx
  size x transform block x candidate tx type underneath shares it. An earlier
  revision owned it inside `txfm_rd_in_plane_intra` and returned only **−6.4 %**
  of the calls (854 053 → 799 552) — most walks are a single transform block, so
  a per-walk scratch has nothing to reuse. That intermediate is why the
  ownership sits where it does.
* `encode_intra_block_plane_{y,uv}` and the walks hoist their per-txb
  `pred`/`residual`/`tight` out of the loop.

Every public entry point kept its signature by delegating to the new
`*_into` form over a fresh scratch (`xform_quant`, `xform_quant_optimize_split`,
`search_tx_type_intra`, `av1_fwd_txfm2d`, `dist_block_px_domain`). The
intra-search chain that threads the scratch — `txfm_rd_in_plane_intra`,
`uniform_txfm_yrd_intra`, `choose_tx_size_type_from_rd_intra`,
`pick_uniform_tx_size_type_yrd_intra`, `intra_model_rd_y`, and the UV/CfL twins
— took a new trailing parameter; five test call sites pass
`&mut …Scratch::default()`, which is behaviourally what they had before.

### The `memset` was deliberately kept — measured, not assumed

Every reused buffer is refilled with `clear()` + `resize(n, 0)` (or
`clear()` + `extend_from_slice`), which writes **exactly** what the `vec![0; n]`
it replaces wrote. Reuse is therefore byte-identical *by construction*: no
argument about which elements the producer overwrites is load-bearing anywhere.

A variant that skipped the re-zero was built and measured, because all three
`XformQuantScratch` buffers are provably overwritten in full (`av1_fwd_txfm2d`
writes every `coeff[..full]`, including the coded-lossless `av1_fwht4x4` arm;
all twelve quantizer entry points in `aom_dsp::quant` open with
`qcoeff[..n].fill(0); dqcoeff[..n].fill(0)`, and the SIMD `quantize_fp` kernel
stores all `n/8` chunks of both). **It landed inside the control band** — 16
interleaved rounds, −3.34 % vs the safe form's −2.40 % against the same
baseline, each inside the other's spread — so it bought nothing measurable and
was reverted. Per the brief's own rule: a safe reuse pattern that costs some of
the win is the one that ships, and this one did not even cost any.

---

## Byte-identity

**Zero bytes moved.** `cargo nextest run --workspace --run-ignored all`:

| dispatch mode | result |
|---|---|
| SIMD live | **950 run, 950 passed, 0 skipped** (734.999 s) |
| `AOM_FORCE_SCALAR=1` | **950 run, 950 passed, 0 skipped** (893.951 s) |

Gate 2's `--cpu-used 0..9` grid still has **zero pinned cells**. The
`--run-ignored all` tier is included deliberately (the KB-PERF-1 session found
two stale pins that way) and is what the 950 includes — `0 skipped` is the
non-vacuity statement.

A fourth, unplanned gate: the `config_permutations` sweep regenerates
`benchmarks/config_perm_{speed,content,independence}_axis_2026-07-30.tsv` on
every run. Across **616 rows** — 5 contents x 10 `--cpu-used` levels x every
singleton knob axis, port vs real `aomenc` — the regenerated files differ from
the committed ones in the **timing column only**; every `exact`, `port_len` and
`c_len` field is unchanged (verified by diffing all-but-the-last column: 0
differing lines in all three files). Those TSVs are therefore left as committed
rather than re-pinned with numbers taken on a load-31 box.

`cargo check --target x86_64-apple-darwin --workspace --all-targets`: clean.

The 1024x1024/cq44/cpu-used 6 `.obu` is byte-identical at every intermediate
step, and the debug-info build used for the sample emits the same 4472 bytes as
the release build.

---

## Where this leaves the gap

The port/C wall gap at this cell is now **154.953 − 47.701 = 107.25 ms**
(was 111.89 ms). Rebasing the re-profile's ranking on that:

| lever | measured gap | share of the new gap |
|---|---:|---:|
| NEON the CNN convolution | +35.56 ms | 33.2 % |
| bd8 i16-lane FORWARD transform | +19.63 ms | 18.3 % |
| bd8 lowbd intra predictors | +14.54 ms | 13.6 % |
| **allocation (this landing's residual)** | **+17.55 ms** | **16.4 %** |
| widen the landed i16 inverse path | +6.15 ms | 5.7 % |

Allocation stays a large share **because the part that is left is the part
scratch reuse cannot reach**: `finish_grow` (Vec growth in `aom_encode`, now
rank 1 of the class at 8.85 %), the `qcoeff`/`dqcoeff` that
`encode_intra_block_plane_{y,uv}` legitimately retain per transform block, and
the `memset` bytes themselves. Attacking it further means changing what the
encoder *stores*, not where it stores it — a different kind of change from this
one, and one with real byte-identity risk.

**The transferable lesson: a leaf class matched by symbol name is not a lever.**
The next ranking that quotes `alloc/libc` should split the allocator symbols
from `_platform_memset`/`_platform_memmove` before crediting anything with the
total, because a fix aimed at one of those two halves cannot collect the other.

## What is NOT measured here

* **One cell, one image, one content class.** cpu-used 3/4/5/9 were not
  measured; the re-profile records 9 (5.64x) and 4 (7.76x) as worse cells that
  nobody has decomposed.
* **Wall clock only.** No instruction count (no valgrind on Apple Silicon), so
  "341 496 fewer allocator calls" and "4.64 ms less wall" are two independent
  measurements and neither derives the other.
* **The box was not idle** (load 30-41 of 12 cores). The control band is
  interleaved, is quoted first, and the baseline arm reproduces the
  re-profile's published ratio to 0.1 %.
* **No multi-threaded measurement.** Neither arm is threaded.
* Peak live memory is **unchanged** (27 705 399 B on every arm) — this was
  churn, as the re-profile said, and nothing here reduces footprint.
