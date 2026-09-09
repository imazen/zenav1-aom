# KB-PERF-9 — the forward transform COLUMN pass had no 4-wide arm, so every 4-wide forward transform ran the driver's scalar per-column loop

**Measured 2026-09-09.** Byte-identical output; **−1.63 %** at 1024x1024 and
**−1.76 %** at 512x512, each against a same-binary null, each over a rotated
interleaved band.

**Gates: `just gate-landing` GREEN in full** — `test-next` 1503/1503,
`test-next-scalar` 1503/1503, `census-gate` 4/4, `test-whereat` 4/4, all exit 0.

## The defect

`try_fwd_col_pass` opened with a flat

    if col_n % 8 != 0 { return false; }

so any transform whose WIDTH is 4 declined the vector column pass and fell back to
`fwd_txfm2d_core`'s scalar per-column loop. The **inverse** column pass in the same
file has had the arm since the 2026-07-17 AVX2 landing:

    if col_n % 8 != 0 && !(col_n == 4 && half_batch_pays(row_n)) { return false; }

and `half_batch_pays`'s own doc says *"x86-64 always says yes: the 4-wide arms are the
shape the 2026-07-17 AVX2 landing measured and kept"*. The forward twin simply never
got it. Measured cost in the 1024x1024 re-profile
(`encoder_x86_reprofile_1024_2026-09-09.md`): **~294 ms/encode of scalar 1-D forward
kernels** — `av1_fadst4` 65, `av1_fdct4` 54, `av1_fadst16` 47, `av1_fdct16` 46,
`av1_fadst8` 42, `av1_fdct8` 41 — inside a transform class that is 38.3 % of the gap.

## Why a half-filled batch pays here when KB-PERF-3 measured one NULL

This is the KB-PERF-5 lesson exactly — *"same shape, opposite verdict, because the
baseline differs"*. KB-PERF-3 built a half-idle `i16x16` batch for 8-dim blocks and
measured it null; there the competition was a **full `i32x8` vector pass**, and an
`i16x16` and an `i32x8` are both 256 bits, so nothing changed. Here the competition is
the driver's **scalar loop**, so 4 live lanes is still ~4x: the vector batch does one
1-D transform for 4 columns where the scalar loop does four sequential ones.

Both ends also stay contiguous at 4-wide — an i16 run of the source row in, a 4-entry
run of `buf` out — so this is the inverse COLUMN pass's shape, not the row pass's. That
distinction is already load-bearing in this file: `try_fwd_row_pass` carries a comment
recording that `col_n < 8` degrades its LOADS to a gather and measured **+9.8 %** on
`fwd_txfm::04x16_dct`, *"gathers cost; scatters don't"*.

## Correctness

The idle lanes are zero and stay inert end to end: the 1-D kernels are linear,
`shl_clamp64v` and `rshiftv` map 0 to 0, and only `active` lanes are stored. So a half
batch computes the same values for its live columns as a full one — by construction,
not by a numeric bound.

The `lr_flip` store index generalises without a new case. Lane `j` holds source column
`cg + j`, which lands at `col_n - 1 - (cg + j)` — a descending run based at
`col_n - cg - active`, i.e. the lanes reversed. That is the expression the 8-lane arm
already used, with `active` in place of `8`.

**Gates run:** `txfm2d_simd_perm_diff::txfm2d_simd_equals_scalar_at_every_permutation`
(the SIMD-vs-scalar-reference differential at every `(tx_type, tx_size)`) and
`txfm2d_diff::txfm2d_differential_fuzz` (vs the real exported C), plus the whole
23-test transform suite — all green.

**BITE PROOF, and it is asymmetric.** Swapping two lanes in the new 4-wide non-flip
store — the only perturbation reachable *solely* through the new arm — fails
`txfm2d_differential_fuzz` and `txfm2d_simd_equals_scalar_at_every_permutation` while
**all 10 inverse tests stay green**. So the arm is genuinely reached (the green is not
vacuous) and genuinely gated.

## Measurement

Two binaries built from one tree (sha256-distinct), arms **rotated** each round so no
arm is confounded with its slot (playbook §6), a same-binary null (`baseB`) in every
band, one encode per arm per round.

| cell | base | new | paired median | rounds faster | sign-test p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 10541.2 ms | 10362.6 ms | **−1.63 %** | **10/10** | 0.0020 | −0.15 %, 8/10, p=0.11 |
| 512x512 cq27 s0 | 3778.6 ms | 3712.2 ms | **−1.76 %** | **20/20** | <0.0001 | −0.19 %, 11/20, p=0.82 |

Raw spreads 0.5–1.8 %. Bands: `.band1024.tsv`, `.band512.tsv`.

**Ratio at the 1024x1024 profile cell: 2.521x → 2.478x** (C median 4181.9 ms over 5
invocations on the same box). Output byte-identical at **39,694 B** in both arms.

Removing ~172 ms of the ~294 ms of scalar forward kernels is the right order: the
vector path is not free, and it reaches only the COLUMN pass.

## What this does NOT cover, precisely

The column kernel spans `row_n` points and the row kernel spans `col_n`, so the arm
reaches exactly the transforms whose WIDTH is 4:

* **fixed** — TX_4X4, TX_4X8, TX_4X16 column passes;
* **still scalar** — the forward ROW pass whenever `row_n == 4` (TX_4X4, TX_8X4,
  TX_16X4), because `try_fwd_row_pass` still gates on `row_n % 8 != 0`. That arm is
  genuinely harder and must not be assumed symmetric: its loads from row-major `buf`
  are strided, so a 4-row batch is the gather case this file has already measured
  losing.

**aarch64 is unmeasured.** `half_batch_pays` returns `kernel_points >= 8` there, so a
4-wide column pass with `row_n == 4` (TX_4X4) declines on ARM while TX_4X8 / TX_4X16
take the arm. That is the same policy the inverse column pass already applies; no ARM
band was run.

One box, one content class, one speed, `--cpu-used 0`.

## Gate status

Run to completion on the landing tree:

* `just test-next` — **1503 run, 1503 passed, 58 skipped** (313.6 s), exit 0.
* `just test-next-scalar` — **1503 run, 1503 passed, 58 skipped** (336.4 s), exit 0.
  Under `AOM_FORCE_SCALAR=1` this arm declines via `fwd_col_pass_scalar`, so the
  scalar leg exercises the unchanged scalar loop; it is inert there by
  construction and now measured so rather than assumed.
* `just census-gate` — 4/4, exit 0.
* `just test-whereat` — 4/4, exit 0.

The first attempt at this gate died partway through the scalar leg when the
box hit a **disk quota** (`EDQUOT`), which presents as every shell command
returning exit 1 with empty output — the harness cannot write its command
script. Freeing space restored it; `target/` rebuilds to 911 MB.
