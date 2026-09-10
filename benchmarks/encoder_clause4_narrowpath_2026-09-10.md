# Clause (4): the fusion sequence BYPASSED the i16 path, and that — not a plane refactor — is the ranked lever

**2026-09-10.** A measurement taken *instead of* starting the multi-session u16
plane refactor, because the figure that refactor was sized by (`~42 % of the
gap`) comes from a **192x192 `--cpu-used 0`** entry, and this session corrected
two other stale regime-carrying figures already (the s0 class ranking, and
KB-PERF-4's reach shares).

**It changes the recommendation.** The largest single lever for clause (4) is
not a refactor. It is a regression-in-effect that the fusion sequence introduced
and that nothing could have caught.

## The finding

`benchmarks/encoder_x86_reprofile_1024_s3_2026-09-10.*`, at the shipping preset:

| | ms |
|---|---:|
| port 1-D transform kernels on the **wide i32** path (`run_fwd1d_v3` + `run_inv1d_v3`) | **285.9** |
| port kernels on the **narrow i16** path (`lowbd16*`) | 28.5 |
| libaom's narrow 1-D equivalents (`fdct8x8_new_sse2`, `fadst8x8_new_sse2`, `av1_idct8_sse2`, `av1_iadst8_sse2`) | 45.0 |

**Only 9.1 % of the port's 1-D transform work runs on the i16 path, and the wide
path is 6.4x libaom's narrow one.**

**Why.** KB-PERF-3 landed i16 forward kernels behind a runtime bound and
measured **−2.0 to −2.5 %**. They are reached from the GENERIC driver's passes —
`try_fwd_col_pass` / `try_fwd_row_pass` / `try_inv_row_pass`. Every fused
whole-transform kernel this cycle added (KB-PERF-23, 24, 25, 26, 27, 28, 29, 30)
calls `run_fwd1d` / `run_inv1d` — the **i32x8** kernels — directly, and the
driver hooks route around the generic passes entirely. By the s3 census the
fused hooks now cover **~97 % of transforms by call share**, so KB-PERF-3's
landing is bypassed for almost everything.

**Nothing could have caught this.** Every fusion landing was byte-identical, so
no gate moved; and every one was net *negative* on the wall clock, because the
fusion win exceeded the i16 loss it was silently taking. Only a like-for-like
symbol comparison against libaom's narrow kernels shows it. **A landing can be a
real win and still give back a larger win that already existed** — check whether
a new fast path bypasses an older one before crediting its band.

## Why libaom's speed IS the narrow path — measured, not asserted

Classifying every C symbol by whether it exists only because bd8 has a narrow
(u8/i16) variant: **268.3 ms of libaom's 1550.7 ms encode — 17.3 % — is on
narrow-path kernels**, led by `fdct8x8_new_sse2` (14.9), `fadst8x8_new_sse2`
(13.8), `lowbd_inv_txfm2d_add_no_identity_ssse3` (13.0),
`lowbd_fwd_txfm2d_16x8_avx2` (11.3), `aom_subtract_block_sse2` (10.9),
`av1_lowbd_fwd_txfm2d_8x8_avx2` (9.0). So the narrow path is how libaom gets its
transform speed, and the port giving it up is a real cost rather than a
theoretical one.

## What this does to the clause-(4) plan

**Ranked lever: restore i16 inside the fused kernels.** Not a refactor — a bound
check plus a kernel selection inside code written this cycle. The fused
structure (column pass, one in-register transpose, row pass) is unchanged; what
changes is that the intermediates are held as `i16x16` where the runtime bound
admits, which is 16 lanes instead of 8 for the same work plus the cheaper
`fbtf16` `half_btf`.

### CORRECTED, same day, before anyone spent a session on it

**The first version of this section said "halving the 285.9 ms would be −136 ms
≈ 4.2 % of the encode". That was WRONG by ~2x, and this repo's own measurement
refutes it.**

KB-PERF-3 built the half-idle variant — running 8-dimension blocks as 16-lane
i16 batches — and **measured it NULL** (−0.009 % against a +0.08 % null): an
`i16x16` and an `i32x8` are both 256 bits, so a pass whose vectorized dimension
is under 16 gets **nothing**. Halving assumed every pass fills. It does not:

* **82.98 % of forward transforms have BOTH dimensions < 16** (s3 census), so
  neither of their passes can fill a 16-lane batch;
* of the rest, most fill only ONE of the two passes (8x16 fills its row pass,
  not its column pass).

Weighting each pass by its real vector work — `ceil(dim / lanes)` groups of an
`n log n` kernel, summed over the census — the recoverable share is **22.2 % of
1-D transform work**, i.e.

| | |
|---|---:|
| ceiling | **63 ms = 1.96 % of the encode** |
| with this session's measured 1.7x–4.8x optimism | **0.41 % – 1.15 %** |

So the lever is a **~1 % item, not a 4 % one.** It is still the largest single
thing on the board — the last seven landings averaged 0.3 pp — but it does not
change the shape of the clause-(4) problem, and the paragraph below should be
read with that correction applied.

**Two things already settled that this must not re-litigate:**

* the **inverse** i16 lane-width change was built and **measured null** as a
  standalone (`−0.03 %`, 5/8 rounds) — because an `i16x16` and an `i32x8` are
  both 256 bits, so lane count changes and work does not. That null was measured
  on the GENERIC two-pass driver, where the intermediates round-trip through
  `buf`. Inside a FUSED kernel the intermediates stay in registers, so the
  comparison is not the same one — but it is close enough that **the inverse
  half must be measured on its own band before being assumed**;
* the **forward** i16 change was NOT null (−2.0 to −2.5 %), and is the half with
  evidence behind it. Start there.

**The u16 plane refactor is DEMOTED, not dismissed.** Its lane-width half is
either landed (forward) or measured null (inverse), and its
size-specialisation half is ~97 % done by the fusion sequence. What remains
uniquely to it is u8 loads/stores — real, but it is the residue after this
lever, not the headline, and sizing it needs its own measurement rather than the
192x192 figure.

## Where the 63 ms sits, so the next session does not re-derive it

Recoverable 1-D work by transform, as a share of ALL 1-D transform work
(weighted `ceil(dim / lanes)` groups x `n log n`, over the s3 census):

| transform | share of 1-D work | ms |
|---|---:|---:|
| **fwd 16x16** | **7.82 %** | **22.4** |
| **inv 16x16** | **4.34 %** | **12.4** |
| fwd 16x8 | 1.34 % | 3.8 |
| fwd 8x16 | 1.01 % | 2.9 |
| fwd 32x32 | 0.97 % | 2.8 |
| inv 32x32 | 0.92 % | 2.6 |
| inv 16x8 | 0.89 % | 2.6 |
| fwd 32x16 / inv 32x16 / fwd 16x32 | 2.32 % | 6.6 |
| everything else | 2.57 % | 7.3 |
| **total** | **22.18 %** | **63.4** |

**Two kernels carry 55 % of it** — `fwd_16x16_fused` and `inv_16x16_fused`, both
written this cycle, and both the case where BOTH passes fill a 16-lane batch
(every other shape fills at most one). So the first increment is those two and
nothing else.

**And it is small.** `fwd 16x16` at 22.4 ms is **0.69 % of the encode as a
ceiling**, i.e. **0.14 %–0.41 % delivered** at this session's measured optimism
— for a kernel rewrite that needs a 16x16 i16 in-register transpose (raw AVX2:
magetypes has no integer transpose), a fresh bound audit, and new differentials.
That is a poor trade in isolation and it is stated here so nobody starts it
expecting otherwise; it is worth doing as part of an i16 fused programme
covering the whole table above, not as a one-off.

## Honest position on the clause

At **2.090x** the bar needs **−915 ms, 54 % of the 1690 ms gap**. This lever at
its 1–2 % realistic range is 30–65 ms of that. **It does not close clause (4)
either** — but it is the difference between a programme with one 4 %-class item
in it and a programme of 141 landings at 0.2 %, and it should be taken before
any decision to ship at 2.09x or to open the refactor.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, bd8, x86-64. The
narrow-path classification is by symbol suffix (`lowbd`, `_sse2`, `_ssse3`,
excluding `highbd`), so read the 17.3 % as +-2 rather than exact.
