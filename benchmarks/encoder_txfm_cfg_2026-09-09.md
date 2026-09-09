# KB-PERF-14 — the transform config built two function pointers per call that only the scalar fallback reads

**Landed 2026-09-09.** Byte-identical output; **−0.30 %** at 1024x1024 and
**−0.24 %** at 512x512, each against a same-binary null, each over a rotated
interleaved band. Ratio at the 1 MP profile cell **2.402x → 2.395x**.

**The interesting result is a rejected first attempt** — the same change with the
lookup one scope deeper measured **+0.20 %** (5/24 rounds faster, p=0.0066), a
**0.44 pp swing from where a `match` sits**. Both bands are committed.

## The defect

`get_fwd_txfm_cfg` and `get_inv_txfm_cfg` cost **46.7 + 34.5 = 81.2 ms** on a
1 MP speed-0 encode and have **no C counterpart at all**: libaom's whole-2-D
entry points are size-specialised at compile time, so it never derives a config
per transform. That makes this pure port overhead rather than a kernel gap.

Two things in it were avoidable:

* **The `Cfg` struct stored `func_col` / `func_row`**, two resolved `Txfm1d`
  pointers from a 12-arm match. They are read **only on the scalar fallback
  path** (`if !cols_done` / `if !rows_done` in the forward core, and the two
  scalar row/column loops in each inverse core) — which the SIMD passes usually
  skip — yet both matches ran, and both pointers were stored, on **every**
  transform.
* **`log2_idx` was a five-arm `match`** over `{4,8,16,32,64}` compiling to a
  comparison chain, called twice per config, i.e. twice per transform. The
  dimensions are powers of two, so `trailing_zeros() - 2` is the same value in
  one instruction.

Deriving on demand is exact rather than approximately so: `valid` is asserted at
all three entry points (`txfm2d.rs:336`, `inv_txfm2d.rs:246` and `:396`) before
any core runs, so the old `if valid { .. } else { av1_fdct4 }` fallback arm was
already unreachable from the cores.

## The rejected attempt, and why it is worth recording

The obvious edit puts `txfm_func(cfg.txfm_type_col)` at the call site. But in
both the forward and the inverse cores **the call site is inside the per-column
or per-row loop**, so that moves the match from once-per-transform to
once-per-column — strictly more work than storing the pointer.

Measured, 512x512 cq27 s0, rotated band, 24 rounds:

| variant | paired median | rounds faster | p |
|---|---:|---:|---:|
| lookup at the call site (inside the loop) | **+0.20 %** | 5/24 | 0.0066 |
| lookup hoisted to the top of each scalar pass | **−0.24 %** | 21/24 | 0.0003 |

Same removal of two stores and two matches per transform; the only difference is
which scope the surviving lookup sits in. **A 0.44 pp swing, and the wrong
version was significant in the wrong direction** — it would have been landed as
a regression by anyone reading the diff rather than the band. The rejected band
is committed as `.band512_in_loop_rejected.tsv`.

## Correctness

No arithmetic changed. Byte-identical output verified directly on two cells
before any timing: **39,694 B at 1024x1024 and 10,912 B at 512x512**. The 23-test
transform differential suite (`txfm2d_diff`, `txfm2d_simd_perm_diff`,
`inv_txfm2d_diff`, `inv_txfm2d_lowbd_diff`, `inv_txfm2d_u8_simd_diff`,
`txfm1d_diff`, `inv_txfm1d_diff`, all against the real exported C) is green, as
is `-p zenav1-aom-dsp --lib` 50/50.

## Measurement

Two sha256-distinct binaries from one tree, arms **rotated** each round
(playbook §6), a same-binary null (`baseB`) in every band.

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 10109.0 ms | 10079.3 ms | **−0.30 %** | 14/16 | 0.0042 | +0.01 %, 7/16, p=0.80 |
| 512x512 cq27 s0 | 3620.8 ms | 3610.9 ms | **−0.24 %** | 21/24 | 0.0003 | +0.11 %, 9/24, p=0.31 |

Raw spreads 0.6–2.0 %. **Ratio 2.402x → 2.395x** (C median 4207.9 ms, same
band; and see KB-PERF-12's caveat — the C arm drifts ~1 % between bands, so the
paired deltas are the solid numbers, not the ratio digits).

## Not covered

* **The config is still derived per transform.** Removing that entirely means
  either a `const` table of 16x19 `Cfg` (~14 KB, and worth measuring rather than
  assuming — it trades arithmetic for cache) or hoisting the config out to the
  caller, which knows `(tx_type, tx_size)` is fixed for a whole walk. Either is a
  larger change than this one and neither is obviously a win.
* One box, one content class, one speed, `--cpu-used 0`, x86-64.

## Gate status

`just gate-landing` green in full — `test-next` 1503/1503,
`test-next-scalar` 1503/1503, `census-gate` 4/4, `test-whereat` 4/4, all exit 0.
