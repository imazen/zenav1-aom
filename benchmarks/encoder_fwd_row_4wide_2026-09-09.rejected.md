# REJECTED: the forward ROW pass `row_n == 4` arm — measured NULL, and it is NOT a harness issue

**2026-09-09. Built, gated, measured, REVERTED.** The symmetric twin of
KB-PERF-9's landed forward COLUMN arm, and the gap that record named in itself:
*"the forward ROW pass is still scalar whenever `row_n == 4` (TX_4X4, TX_8X4,
TX_16X4)"*.

## What was built

A `row_n == 4` half-filled 8-lane batch in `try_fwd_row_pass` /
`fwd_row_pass_core`, mirroring `try_inv_row_pass` (which already had one) and
KB-PERF-9's column arm: `active = (row_n - rg).min(8)`, idle lanes zero, partial
store of `active` lanes. Bit-exact by the same argument — `run_fwd1d` is linear
and per-lane, `rshiftv`/`mul_rshiftv` map 0 to 0, only live lanes are stored.

**KB-PERF-9's gather caveat does not apply**: it is a statement about `col_n < 8`
(and is `cfg!(aarch64)`-gated, so already off on x86). At `row_n == 4` the
reachable shapes are TX_8X4 and TX_16X4 — TX_4X4 is served by
`fwd_txfm2d_4x4_fused` and never arrives — and both have `col_n >= 8`, so the
loads stay the contiguous 8x8-transpose-tile form.

Reach at the shipping preset: **8x4 9.13 % + 16x4 1.02 % = 10.2 % of forward
transforms** (`encoder_txfm_size_census_s3_2026-09-09.md`).

## Correctness and reach, both verified

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` differentials **395/395**.
* **Bite proof, asymmetric**: perturbing the new partial store fails
  `txfm2d_differential_fuzz` and `txfm2d_edge_cases` (both against the real
  exported C) and `txfm2d_simd_perm_diff`, while **391 stay green**; restored,
  395/395. So the arm is genuinely reached and load-bearing.

## The bands — null at both cells

Two sha256-distinct binaries from one tree, arms **rotated**, same-binary null:

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 512x512 cq27 s0 | 3406.07 ms | 3407.77 | **+0.085 %** | 10/24 | 0.5413 | +0.004 %, 12/24, p=1.00 |
| 1024x1024 cq27 **s3** | 3526.22 | 3519.57 | **−0.065 %** | 14/24 | 0.5413 | **−0.155 %**, 16/24, p=0.15 |

At the shipping preset the same-binary null is LARGER in magnitude than the
"effect". Both cells are indistinguishable from zero.

## It is NOT a harness issue — the work moved, and it moved the wrong way

The standing suspicion for a null on a kernel libaom vectorises is that the
harness never reached it. Checked directly by profiling both binaries at the
same cell and diffing symbol shares:

| symbol | base | new |
|---|---:|---:|
| `av1_fdct8` (scalar) | 0.20 % | **0.00 %** |
| `av1_fadst8` (scalar) | 0.15 % | **0.00 %** |
| `av1_fdct16` (scalar) | 0.14 % | **0.00 %** |
| `av1_fadst16` (scalar) | 0.10 % | **0.00 %** |
| `__arcane_fwd_row_pass_core_v3` | 1.31 % | **2.14 %** |

**The scalar row kernels are eliminated exactly — 0.59 % to 0.00 % — and the
vector row pass grows 0.83 %.** So the path changed, completely, and the trade
is a net **+0.24 %**: the half-filled vector pass costs MORE than the scalar
loop it replaced.

## Mechanism — same shape as KB-PERF-9, opposite sign, because the fixed cost differs

The forward COLUMN pass loads strided `i16` straight into lanes: no transpose,
so a half batch costs half the loads and half the kernel, against a scalar loop.
It paid.

The forward ROW pass loads through an **8x8 register transpose** per 8 columns.
At `row_n == 4` that transpose runs at FULL cost on half-zero data, and the
kernel runs 8 lanes for 4 useful rows — so the fixed overhead per useful element
DOUBLES, and there is only ~4 rows of work to amortise it over. KB-PERF-5's rule
again: *same change, opposite sign, because the baseline differs* — here it is
the fixed per-batch cost that differs, not the baseline.

**Do not re-attempt this without removing the transpose** (a 4x8 transpose, or a
row-major kernel), which is a different piece of work.

Bands kept: `.rejected.band512.tsv`, `.rejected.band1024s3.tsv`.

## Recorded while here

Under the perturbed (bite-proof) build, `cdef_find_dir_simd_diff` also failed —
a test with no relation to the forward transform. It passed on the clean tree
before and after (395/395 both times). Observed once, not reproduced, not
explained; noted so a future session that sees it does not treat it as new.
