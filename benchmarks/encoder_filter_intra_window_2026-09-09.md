# KB-PERF-11 — filter-intra zeroed a 2178-byte scratch per call to fill as few as 25 cells

**Landed 2026-09-09.** Byte-identical output; **−0.66 %** at 1024x1024 and
**−0.65 %** at 512x512, each against a same-binary null, each over a rotated
interleaved band. Ratio at the 1 MP profile cell **2.457x → 2.440x**.

## The defect

`aom_dsp::intra::filter_intra_predict_high` opened with

    let mut buf = [[0u16; 33]; 33];

— **2178 bytes of zero-init on every call**, sized for the largest transform
(32x32) regardless of the actual one. A 4x4 filter-intra prediction fills a 5x5
corner of it: **25 useful cells, 1089 zeroed**. The arithmetic in that call is
two 4x2 blocks of 8 outputs x 7 taps = 112 multiply-accumulates, so the memset
is the dominant term at small transform sizes, and small is the common case.

This is the shape KB-PERF-2's lever 3a already found and fixed once, in a
different file: *"a flat `[i32x8; 64]` zero-init compiles to a 2 KiB memset per
array, which dominates the small transforms"*. The forward/inverse transform
passes were tiered `{8,16,64}` for exactly this reason
(`transform/simd/mod.rs`, and `lowbd16.rs:132` states it); filter-intra was
never given the same treatment.

Measured cost before the change, 1024x1024 cq27 speed 0:

| port | ms | C | ms | ratio |
|---|---:|---|---:|---:|
| `filter_intra_predict_high` | 145.6 | `av1_filter_intra_predictor_sse4_1` | 27.6 | **5.3x** |

## The fix is a sliding window, not a tier

Tiering by transform size would need ~14 const-generic monomorphisations (the
rectangular sizes give independent `bw`/`bh`). It is unnecessary: **C's
recursion only ever reads rows `r-1`, `r` and `r+1`**, so three rows suffice and
the values are identical. 198 bytes instead of 2178, no size dispatch, and one
code path for every transform size.

Row mapping, so a later reader can check it against the C without re-deriving:
`prev` is C's `buffer[r-1]`, `row0` is `buffer[r]`, `row1` is `buffer[r+1]`;
column 0 of each is the left edge and row 0 is the above edge with the corner at
column 0, exactly as `av1_filter_intra_predictor_c` initialises them. C's
`buffer[r][0] = left[r-1]` and `buffer[r+1][0] = left[r]` are re-established at
the top of each iteration; the loop steps by 2 with `r <= bh - 1`, so both
indices are inside `left[..bh]`.

C copies `buffer[r+1][1..bw+1]` into `dst` row `r` in a final pass over the
whole scratch. Emitting both rows at the end of each iteration is the same
assignment one iteration earlier, which is what lets the scratch shrink.

**Every cell C reads is either an initialised edge or a previously written
cell** — the zero-init was never load-bearing, only the sizing was. That is why
this is exact rather than approximately-exact.

## Correctness

Same arithmetic, same taps, same rounding and clamp; only the storage changed.
Gated against the **real exported C** by
`filter_intra_diff::filter_intra_matches_c` and
`build_filter_intra_diff::build_filter_intra_matches_c`, plus
`predict_intra_diff::predict_intra_matches_c`,
`highbd_diff::highbd_intra_byte_identical` and
`intra_diff::intra_predictors_byte_identical`.

**BITE PROOF, asymmetric.** Swapping the `k < 4` / `k >= 4` row split — the one
place the three-row window could disagree with C's flat buffer — fails **3**
tests, while 11 unrelated intra tests in the same binary (warp, inter-intra,
directional) stay green. So the window is genuinely reached and the green is not
vacuous.

## Measurement

Two sha256-distinct binaries from one tree, arms **rotated** each round
(playbook §6), a same-binary null (`baseB`) in every band, one encode per arm
per round. Output byte-identical at **39,694 B** on both arms.

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 10229.3 ms | 10157.8 ms | **−0.66 %** | **12/12** | 0.0005 | +0.05 %, 5/12, p=0.77 |
| 512x512 cq27 s0 | 3663.4 ms | 3639.0 ms | **−0.65 %** | **24/24** | <0.0001 | −0.02 %, 13/24, p=0.84 |

Raw spreads 0.6–1.6 %. Bands: `.band1024.tsv`, `.band512.tsv`.
**Ratio 2.457x → 2.440x** (C median 4162.8 ms over 5 invocations, same box).

**§14, and this one is close for once**: the row said ~118 ms addressable and the
delta is ~71 ms at 1 MP. The residual is the 5.3x kernel gap that remains — this
removed the memset, not the taps.

## Not covered

* **The taps themselves are still scalar.** 8 outputs x 7 taps per 4x2 block is
  a natural 8-lane dot product and is what libaom's SSE4.1 kernel vectorises;
  that is a separate landing and needs its own bound work.
* **`highbd_filter_intra_edge` is untouched** — 85.7 ms against
  `av1_filter_intra_edge_sse4_1` 16.7 (5.1x), the next item in the same class,
  left alone so this bite proof stays asymmetric.
* The lowbd `filter_intra_predict` twin (if reached on a u8 path) was not
  measured; this cell is bd8-through-u16, which is the encoder's own
  representation.
* One box, one content class, one speed, `--cpu-used 0`, x86-64.
* **Reachability caveat worth carrying**: filter-intra is **0.00 %** of leaves at
  `--cpu-used 6` (`prune_filter_intra_level = 2` means `rd_pick_filter_intra_sby`
  is never called) and 10.46 % at `--cpu-used 5`
  (`benchmarks/winperf_family_census_2026-08-03.md`). This lever is worth
  nothing at speed 6 and is measured here at speed 0, where the differential
  corpus reaches filter-intra on 21–31 % of leaves.

## Gate status

`just gate-landing` green in full — `test-next` 1503/1503,
`test-next-scalar` 1503/1503, `census-gate` 4/4, `test-whereat` 4/4, all exit 0.
