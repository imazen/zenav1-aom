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
- **Census harness `crates/aom-target/examples/zq_census.rs`** injects:
  encode = `aomenc` CLI (libaom 3.13.1, still AVIF-class keyframe:
  `--end-usage=q --cq-level=<q>` — the drafted `--min-q`/`--max-q` pin is
  NOT what shipped, see the operating-point note under the phase-A result;
  y4m C444 full-range in, IVF out); decode = `aomdec` CLI (y4m out, same
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

## Building and running the harness (updated 2026-08-31)

`crates/aom-target` is a normal member of the root workspace. The library
carries no dependencies at all, so `cargo check --workspace` / `cargo test
--workspace` build it (and its 4 unit tests) with nothing extra pulled in.

The census harness is behind the crate's non-default `census` feature, which
is what gates the zensim judge (git-pinned to zensim `main`
`f70511133d3056099de2ddc73a064ce417f4f593` for `custom-profiles` +
`feature-regime-v2`; no published zensim exposes either) and `png`:

```
cargo run --release -p zenav1-aom-target --features census --example zq_census -- \
    <corpus.tsv> <targets,csv> <max_encodes> <out.tsv>
```

It additionally needs `aomenc` / `aomdec` on `PATH` and a zensim bake at
`$ZQ_BAKE` (default:
`/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin`);
`$ZQ_TMP` (default `/home/lilith/tmp/aomzq`) holds the per-trial y4m/IVF scratch.

HISTORY: as landed on 2026-08-29 the crate was EXCLUDED from the workspace
with a `[workspace]` table of its own, because its zensim dev-dependency was
a `path` into the sibling `zensim` checkout — cargo loads every member
manifest for any workspace command, so as a member it made the whole
workspace unresolvable on any checkout without that sibling, i.e. every CI
runner. The git pin plus the default-off `census` feature removes both
halves of that problem, and the exclusion is gone.
