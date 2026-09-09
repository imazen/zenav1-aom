# KB-PERF-12 — the intra edge filter copied the whole edge to read a 5-wide window

**Landed 2026-09-09.** Byte-identical output; **−0.32 %** at 1024x1024 and
**−0.29 %** at 512x512, each against a same-binary null, each over a rotated
interleaved band. KB-PERF-11's finding one function over.

## The defect

`aom_dsp::intra::edge::highbd_filter_intra_edge` opened with

    let mut edge = [0i32; 129];
    edge[..sz].copy_from(p[..sz]);

— **516 bytes of zero-init plus an `sz`-element copy on every call**, so that a
**5-tap** filter could read original samples while writing `p` in place. Measured
before the change, 1024x1024 cq27 speed 0: **85.7 ms** against
`av1_filter_intra_edge_sse4_1` at **16.7 ms — 5.1x**.

C needs the scratch because it is C (`av1_highbd_filter_intra_edge_c` has no
cheap way to carry state), but only a 5-wide window of originals is ever live,
and it can be carried forward. Neither the zero-init nor the copy is necessary.

## The window, and why it is exact

**Invariant:** on entry to iteration `i`, `w[j] == original p[clamp(i-2+j)]`.

* It holds at `i = 1` by construction — nothing has been written yet, so the
  five clamped reads are all original.
* The shift re-establishes it: old `w[j+1]` is
  `original p[clamp((i-1)-2+j+1)]`, which is the same index as the new `w[j]`.
* The incoming `w[4]` reads `p[clamp(i+2)]`, whose index is `>= i` and therefore
  **not yet written** — including at `i == sz-1`, where the clamp makes it `i`
  itself and the read precedes the write in the same iteration.

The clamp is what makes a naive "`k == i-1` → history slot 1, `k == i-2` →
history slot 2" mapping wrong (at `i = 1` both `j = 0` and `j = 1` collapse onto
index 0), which is why the window is indexed positionally and shifted rather
than addressed by offset.

## Correctness

Same taps, same rounding, same clamp; only the storage changed. Gated against
the **real exported C** by `edge_diff::highbd_filter_intra_edge_byte_identical`,
with `predict_intra_diff::predict_intra_matches_c` and
`intra_lowbd_diff::predict_intra_lowbd_matches_c_and_highbd` covering it through
the full predictor.

**BITE PROOF, asymmetric.** Dropping the `w[4]` refresh — which staleness is the
one way a carried window can diverge from C's flat copy — fails exactly **3**
tests while **42** stay green, including
`edge_diff::filter_intra_edge_byte_identical`, the **lowbd twin that was
deliberately left untouched**. That is the asymmetry: the same file, the same
shape of function, unmodified and still passing.

## Measurement

Two sha256-distinct binaries from one tree, arms **rotated** each round
(playbook §6), a same-binary null (`baseB`) in every band, one encode per arm
per round. Output byte-identical at **39,694 B** on both arms.

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 s0 | 10179.7 ms | 10154.3 ms | **−0.32 %** | 13/16 | 0.0213 | +0.03 %, 7/16, p=0.80 |
| 512x512 cq27 s0 | 3646.8 ms | 3635.3 ms | **−0.29 %** | 18/24 | 0.0227 | −0.05 %, 13/24, p=0.84 |

Raw spreads 0.7–1.4 %. Bands: `.band1024.tsv`, `.band512.tsv`.
The 1024 band was run at **16 rounds rather than 12** because the 512 band put
the effect near the floor — KB-PERF-6's corollary, size the band to *this*
effect rather than to the last one.

**This is the smallest of the four levers this session** (−0.32 % against
−0.94 %, −0.66 %, −1.63 %) and it is reported as such: p ≈ 0.02 rather than
p < 0.001, significant and replicated but not comfortable.

## A methodological caveat on the ratio chain — read this before quoting digits

The `2.478x → 2.464x → 2.440x → 2.424x` chain in the KB entries is **four
separate ratios, each measured against a C median taken in its own band**, and
the C arm drifts: **4181.9 / 4140.9 / 4162.8 / 4188.5 ms across this session, a
1.15 % spread**. So each ratio carries roughly **±0.6 %** from the reference
alone, and cross-session ratio *differences* smaller than ~0.015 are not
resolvable.

**The paired port-vs-port deltas are the solid numbers**; the ratios are
convenient headline figures with a real error bar. In this band the ratio is
**2.430x → 2.424x**, which is lower than KB-PERF-11's closing 2.440x purely
because that band's C reference was 0.6 % faster — not because anything
regressed.

## Not covered

* **The lowbd `filter_intra_edge` twin has the identical shape** and is
  untouched, deliberately, so the bite proof stays asymmetric. It did not appear
  in this profile (the encoder is u16-through at every bit depth), so it is a
  decoder-side item.
* `highbd_upsample_intra_edge`'s `[0i32; 19]` is 76 bytes and was left alone —
  measured negligible, and shrinking it would not be worth the proof.
* One box, one content class, one speed, `--cpu-used 0`, x86-64.

## Gate status

`just gate-landing` green in full — `test-next` 1503/1503,
`test-next-scalar` 1503/1503, `census-gate` 4/4, `test-whereat` 4/4, all exit 0.
