# HANDOFF — `experimental-video`: wire the implemented-but-unrouted inter kernels (2026-09-25)

**Owner: Devin. Reviewer: the user. Written by the integration-review session that measured
the gap; nothing below is guessed — every line has a file reference.**

## Why this exists

A public-API census of `aom-dsp` (`docs/archive/INTEGRATION_REVIEW_2026-09-24.md`, "Follow-up 2026-09-25")
found 15 functions that nothing in the workspace calls. They are not dead code: they are
**implemented, C-differential-verified inter-prediction kernels the decoder never routes
to**, and the decoder's own refusal comments say so:

| family | kernels (all in `crates/aom-dsp/src/`) | C differential | why nothing reaches it |
|---|---|---|---|
| highbd sub-pel MC | `convolve/highbd.rs`: `highbd_convolve_{2d,x,y}_sr` | `crates/aom-dsp/tests/all/convolve_diff.rs` | `aom-decode/src/lib.rs:3795` refuses any nonzero MV above bd8: *"widen when highbd subpel convolve lands"* — it has landed |
| compound (dist-weighted) MC | `convolve/compound.rs`: `dist_wtd_convolve_{2d,2d_copy,x,y}` + `highbd_*` twins | `compound_convolve_diff.rs`, `compound_diff.rs` | `aom-decode/src/lib.rs:3323` marks any compound-reference block corrupt |
| scaled-reference MC | `convolve/scaled.rs`: `convolve_2d_scale`, `highbd_convolve_2d_scale`; `inter/scale.rs`: `scale_mv` | `convolve_scale_diff.rs` | the decoder assumes unscaled references; `valid_ref_frame_size` is the only `inter::scale` item it touches |

`convolve::kernel_row` is a helper for the callers of the first two families and is
uncalled for the same reason. Two more uncalled functions are **reference twins kept for
differentials, not features**: `restore::wiener::wiener_convolve_add_src_scalar` (a second
scalar Wiener body next to the dispatcher's own scalar tier) and
`intra::edge::highbd_filter_intra_edge` (the plain C shape; the wired one is `_at`). Leave
those alone or fold them — they are not this task.

## The product decision (from the user, 2026-09-25)

> We want inter behind an `experimental-video` compile flag, but if we have basic support
> working that's not a bad thing.

Read that as:

1. **What decodes byte-exactly TODAY stays on by default.** zenavif's animated-AVIF
   census needs the current inter envelope: single-reference, lowbd sub-pel, and
   zero-MV above bd8 (`crates/aom-bench/tests/all/highbd_inter_decode_envelope.rs`:
   8/8 tracks, 40/40 shown frames byte-exact against the in-process C decoder, incl. a
   12-bit 4:2:2 inter frame). Do not move that behind the flag.
2. **Everything this handoff widens goes behind `experimental-video`**, a cargo feature
   on `zenav1-aom-decode` (and `zenav1-aom-encode` for the KB-16 inter-encode rung, if you
   touch it), forwarded by the `zenav1-aom` facade. **Default OFF.** With the feature off,
   the three refusals above must still fire by name — the same `DecodeError::UnsupportedFeature`
   category (`aom-decode/src/error.rs:66`), never a panic, never `mark_corrupt` on a
   conformant stream.
3. **The flag must be byte-inert on the default envelope.** Every existing gate passes with
   the feature on AND off. That is the contract, and it is what `gate-landing` will check
   (add the feature-on run to it; see gates).

## Order of work — each is its own landing, byte-gated before the next starts

1. **Scaffold.** Add `experimental-video` to `crates/aom-decode/Cargo.toml` (and the facade
   forward in `crates/zenav1-aom/Cargo.toml`). Under `cfg(not(feature = "experimental-video"))`
   the three refusals stay exactly as they are. Regenerate the public-API snapshots
   (`just api-doc`); the feature's items will show in `zenav1-aom-decode.features.txt`.
   `just gate-landing` green. Commit.
2. **highbd sub-pel MC** — the smallest and most valuable. Site: `aom-decode/src/lib.rs:3786-3800`
   and the MC routing in `crates/aom-dsp/src/inter/mod.rs` (`build_inter_predictor`, which
   already takes a `&[u16]` reference plane but runs the sub-pel chain through a u8 scratch —
   that is the thing to widen). The kernels take `&[u16]`/`bd` directly. Gate: extend
   `highbd_inter_decode_envelope.rs`'s "nonzero-MV bd10/bd12" cells from **REFUSED (8/8)** to
   **byte-exact against `ref_decode_av1_stream_frame_opt`**, under the feature; with the
   feature off they must still refuse by name (pin both). This retires the residual KB-40
   documents.
3. **Scaled references** — `convolve_2d_scale` + `scale_mv`; find where the decoder derives
   the reference geometry (`valid_ref_frame_size` call site) and route scaled refs instead
   of assuming identity. Gate: a C-oracle differential over a synthetic scaled-ref stream
   (the conformance corpus has scaled-reference vectors under the "inter" scope of
   `xtask/conformance.py`; use `--scope` to find them).
4. **Compound prediction** — the largest: `aom-decode/src/lib.rs:3323` plus the mask /
   weight plumbing (`aom_dsp::inter::{blend_a, get_obmc_mask, interintra}` exist; wedge and
   diff-weighted masks may not). Land the dist-weighted convolve routing first, then masks.
   Gate: C-oracle differential per compound mode on synthetic streams; conformance
   "inter" vectors that use compound.

Stop after any step and hand back if a step needs a NEW kernel rather than routing to an
existing one — the point of this handoff is that the kernels exist; discovering one is
missing is a finding to write down, not a reason to write it blind.

## What "cleanly" means in this repository

- **Bit-exact or refused by name.** A path that cannot match the C decoder byte-for-byte on
  its gate does not ship behind the flag either; it stays refused with the reason in the
  message. Read `CLAUDE.md` "Gates" and the KB-42 rule: a gate is an integration target in
  `tests/`, and `-p <crate> --lib` is not one.
- **`#![forbid(unsafe_code)]` stays.** Routing, not intrinsics.
- **No new `pub` without need.** The decoder's implementation modules are behind
  `__internals`; keep new items `pub(crate)` unless a test in `tests/` needs them, and then
  prefer the `__internals` gate over widening the default surface.
- **Trace, don't `eprintln!`.** Diagnostics go through `aom_dsp::trace` (`trace_on!`,
  `trace_out!`); the env names are in `docs/upstream-instrumentation/README.md`. If you
  instrument the C oracle, `just upstream-instrument` / `upstream-pristine` — the landing
  gate refuses a dirty oracle.
- **Every landing: `just gate-landing`** (with the feature off — the default — AND
  `cargo nextest run --cargo-profile test-fast --workspace --features zenav1-aom-decode/experimental-video`
  for the on state; add that second run as a recipe, `test-next-video`, and to the CI
  matrix as a third linux differential leg so the flag is exercised on every push). The
  commit message names the integration targets it ran — not unit-test counts.
- **Record as you go:** a KB entry per real bug (`docs/KNOWN_BUGS.md` body + `CLAUDE.md`
  index line — keep `CLAUDE.md` under 50 KB), the STATUS.md landing entry, and strike the
  KB-40 residual and the T3 "inter" rows in `docs/COVERAGE_QUEUE.md` as they close.
- **Do not push to `main` directly.** Branch `feat/experimental-video`, push it (CI runs on
  `feat/**`), and stop when CI is green on all legs including aarch64; the user merges.

## Facts you will want, so you do not re-derive them

- Decoder inter entry points: `decode_block_inter` (`aom-decode/src/lib.rs:3059`), the ten
  `build_inter_predictor` call sites (`:2781`, `:2852`, `:2943`, `:3012`, `:3854`, `:3954`,
  `:4050`, …), OBMC blends at `:2722` / `:2893`.
- Existing inter gates: `crates/aom-bench/tests/all/{highbd_inter_decode_envelope,inter_harness_chunk0,inter_e2e_search,inter_header_derive_diff,inter_pack_tile_diff,inter_rc_qindex_diff}.rs`,
  `crates/aom-decode/tests/all/{inter_ratchet,inter_real_frame,inter_walking_skeleton}.rs`.
- The C oracle is the pinned `upstream/` submodule (libaom v3.14.1 `03087864`), built by
  `aom-sys-ref/build.rs`; shims live in `crates/aom-sys-ref/src/lib.rs` (`ref_*`). Add a shim
  per kernel you route to if one is missing; the differential is the only evidence that
  counts.
- The conformance corpus's intra scope is the CI gate today; inter vectors exist but are not
  gated (`CLAUDE.md` Gate 1 caveat). Gating them under the feature is part of step 4's
  definition of done, not step 1's.
- KB-16 is the encoder's inter rung (zero-MV P frame byte-exact, single SB). It is out of
  scope unless a decoder change forces a shared type through it.
