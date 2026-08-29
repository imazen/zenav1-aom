# zenav1-aom Zq target loop — REGISTERED 2026-08-29 BEFORE ANY CENSUS RUN

USER DIRECTIVE (2026-08-29, verbatim anchors): "we need aom backend to
support zq as well as svt and rav1e, even if we use a dep injection
interface for now. it should have better integration than ssim2
targeting." This OVERRIDES the earlier premature ruling (GOAL criterion 4
listed the aom target loop as ruled premature because no whole-frame
pure-Rust encode entry exists yet).

## Design (frozen)

- **Crate `crates/aom-target`** — the svtav1-target ruling pattern taken
  one step further: the loop is judge-agnostic AND **encoder-agnostic**
  (full dependency injection). `search_target_qindex(target, opts, trial)`
  brackets AV1 **qindex [0,255]** with `trial: FnMut(u8) -> Result<f64>`
  composing {injected encode} + {injected decode} + {injected judge}. The
  crate holds ZERO codec, ffi, or metric dependencies — when the in-repo
  pure-Rust whole-frame encoder matures it swaps in with no loop change.
- **Census harness `examples/zq_census.rs`** injects: encode = `aomenc`
  CLI (libaom 3.13.1, still AVIF-class keyframe: `--end-usage=q
  --cq-level=<q/4 mapping? no — qindex via --min-q/--max-q pin>`, y4m
  C444 full-range in, IVF out); decode = `aomdec` CLI (y4m out, same
  matrix pair both directions — an in-harness lossless-roundtrip GATE
  fails loud on any matrix drift); judge = zensim **Profile C**
  (folded-944 + `score_features_with_profile`, the frozen north-anchor
  bytes — the same judge family as the avif census, so numbers are
  comparable). "Better integration than ssim2 targeting" = seeded-capable
  + inner-bracketed + decoded-pixel Profile-C judging + census
  discipline, vs an outer ssim2 re-encode bisection.
- **Census (phase A):** corpus9 (family 9-ref set) × t{70,80,88} ×
  k{2,3}, tolerance 0 (census mode), emit-best; blind midpoint seed (the
  content-blind control). Seeds (S1 anchors / head) are phase B,
  fitted from phase A cells, family bars per the svt precedent (S1
  passed at k2 ≥25% improvement).
- Endgame: cells + this md committed; the GOAL criterion-4 aom line
  moves from "ruled premature" to measured; default wiring of anything
  is USER-GATED.

## PHASE-A CENSUS RESULT (2026-08-29, same session) — the aom Zq baseline exists

Smoke + matrix roundtrip gate green (near-lossless roundtrip max-diff ≤ 12
enforced per image). One aomenc CLI note: `--min-q == --max-q` is refused
("differ by at least 8"), so the operating point pins via
`--end-usage=q --cq-level=<q>` alone — monotone, which is all the search
needs (recorded; the pure-Rust encoder later exposes true qindex pinning).

| arm | k | median \|err\| | ±2 hits | photo | nonphoto |
|---|---|---|---|---|---|
| blind midpoint | 2 | 3.497 | 9/27 | 3.61 | 3.27 |
| blind midpoint | 3 | **1.476** | **19/27** | 1.71 | 0.44 |

Judge = zensim Profile C (frozen north-anchor bake, folded-944 — the avif
census's judge family; numbers comparable). Cells:
`benchmarks/aomzq_census_k{2,3}.tsv`. Family reading: the narrow CLI cq
domain (0-63, midpoint 31) makes the blind seed far less catastrophic
than svt's qp staircase was (17.6 k2); the fitted-seed phase B registers
next with the family bar (≥25% k2 improvement, hits not regressed).
