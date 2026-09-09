# KB-PERF-16 — the first size-specialised transform entry point: fused 4x4 forward

**Landed 2026-09-09.** Byte-identical output; **−1.55 % at 1024x1024** and
**−2.15 % at 512x512**, each 14/14 and 24/24 rounds faster against a same-binary
null. **The largest single lever of this cycle**, and the first landing inside
the transform class — the one holding 38.3 % of the clause-(4) gap.

## It was chosen by measurement, not by size

The ranking had called transform *"a programme: libaom's size-specialised
whole-transform entry points against the port's generic driver"* — right, and
not actionable, because "size-specialised" does not say WHICH size. The census
(`encoder_txfm_size_census_2026-09-09.md`) answered it from the already-committed
`note_fwd_txfm` hook:

| tx size | share of forward transforms |
|---|---:|
| **4x4** | **50.70 %** |
| 8x8 | 22.30 % |
| everything with both dims >= 32 | **0.55 %** |

**A whole-transform rewrite instinctively starts with the big kernels, and they
are worth under 1 % here.** 4x4 alone is half of every forward transform.

## The defect, and why it is not lane width

For a 4x4 the generic driver spends, to produce SIXTEEN coefficients: a
`get_fwd_txfm_cfg` derivation, a heap scratch `clear` + `resize`, a column-pass
dispatch, a write into the intermediate `buf`, a row-pass dispatch reading it
back, and the `tx_size` post-process match. KB-PERF-14 had already measured the
config derivation alone at **81.2 ms with no C counterpart at all**, because
libaom's entry points are specialised at compile time and never derive one.

So the lever is **per-call overhead**, and that has a consequence worth stating:
**KB-PERF-3's `fadst4` rejection does not apply.** That audit rejected fadst4
for the *i16* path (it works in a pre-shift domain; `M*` is 1-11). This keeps
the port's existing i32 arithmetic exactly, so it serves every 4x4 tx_type —
the four DCT/ADST combinations are 81 % of types, and non-DCT is 76 %, so a
DCT-only fast path would have missed three quarters of the calls.

## Bit-exactness: the recipe COLLAPSES at 4x4, and each collapse is checked

The arithmetic is the generic driver's, specialised rather than changed — same
1-D kernels via `txfm_func`, same `cos_bit`s read from the same tables. What
makes the fused form small is that at TX_4X4 the shift recipe degenerates:

* `FWD_SHIFT[TX_4X4]` is `[2, 0, 0]`, so `round_shift_array(_, -2)` is an exact
  `* 4` — a 16-bit source reaches 131072, three orders under the clamp that
  call would otherwise apply — and the two remaining shifts are `0`, which
  `round_shift_array` early-returns on;
* `get_rect_tx_log_ratio(4, 4)` is `0`, so the `NEW_SQRT2` row scaling does not
  run.

**Every one of those is a precondition, so every one is CHECKED at runtime and
the function DECLINES rather than diverging** if a table ever moves — the gate
shape KB-PERF-3 and KB-PERF-9 both use. A decline falls through to the generic
driver, which is always correct.

## Measured

Two sha256-distinct binaries from one tree, arms **rotated** each round, a
same-binary null (`baseB`) in every band. Output byte-identical on both arms
before any timing: **10,912 B at 512x512** and **39,694 B at 1024x1024** — the
same values KB-PERF-13/14/15 record.

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 10051.6 ms | 9892.1 ms | **−1.55 %** | **14/14** | 0.0001 | −0.04 %, 8/14, p=0.79 |
| 512x512 cq27 s0 | 3595.7 ms | 3516.7 ms | **−2.15 %** | **24/24** | <0.0001 | +0.11 %, 8/24, p=0.15 |

**Ratio 2.410x → 2.372x** against this band's own C median (4170.3 ms, n=5).
Per KB-PERF-12's caveat the C arm drifts ~1 % between bands, so quote the
PAIRED deltas when precision matters; the ratio digits carry ~±0.6 %.

**The lever is bigger at the SMALLER cell** (−2.15 % vs −1.55 %), which is the
opposite of KB-PERF-7's traffic lever and is what a per-call-overhead lever
should do: a 512x512 frame runs proportionally more small transforms per pixel
of real work.

## Not covered — and 8x8 is a different proof, not a copy

* **8x8 is the obvious next target at 22.3 %**, but its recipe does NOT
  collapse: `FWD_SHIFT[TX_8X8]` is `[2, -1, 0]`, so `shift[1]` is a real
  rounding shift between the passes rather than a no-op. The fusion is still
  available; the bit-exactness argument has one more moving part and must be
  made rather than copied.
* **The INVERSE side is untouched and is LARGER in the profile**
  (`run_inv1d_v3` 472 ms, `inv_col_pass_core_v3` 374, `inv_row_pass_core_v3`
  229, `av1_inv_txfm2d_add_into` 127 against the forward's 259/188/128/206).
  Its size distribution is still **unmeasured** — `note_fwd_txfm` has no inverse
  twin — and that census should come before assuming the mixes match.
* One box, one content class, `--cpu-used 0`, x86-64.

## Gate status

`just gate-landing` green in full, and the transform differentials against the
**real exported C** (`txfm2d_diff`, `txfm2d_simd_perm_diff`, `txfm1d_diff` and
the inverse suite) are what license the arithmetic: 12/12.
