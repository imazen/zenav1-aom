# Contributing

This is a bit-exact port of libaom v3.14.1. The rules below are what keeps it one; the
reasoning and the measurements behind each rule live in [`CLAUDE.md`](CLAUDE.md) and
[`docs/ITERATION_PLAYBOOK.md`](docs/ITERATION_PLAYBOOK.md).

## The contract

- **Bit-exact or refused by name.** A code path ships when its output is byte-identical
  to the pinned C oracle on a gate in `tests/`, or when the configuration is refused with
  a named error before any allocation. Never a panic, never a `Malformed` on a
  conformant stream, never silently different bytes.
- **`#![forbid(unsafe_code)]`** in every published crate. Performance comes from
  kernel shape and runtime SIMD dispatch (`archmage`), not from `unsafe`.
- **No C toolchain on the published path.** The libaom oracle (`upstream/`, built by
  `zenav1-aom-sys-ref`) is a dev-dependency of the harnesses only; `deny.toml` bans
  `cc`/`cmake` from the dependency graph to keep it that way.
- **The oracle stays pristine.** Instrumented C lives as versioned patches in
  `docs/upstream-instrumentation/`; `just upstream-instrument` / `upstream-pristine`.

## Landing a change

1. Branch from `main` (`perf/**`, `fix/**`, `feat/**`, `integrate/**`, `maint/**`
   all trigger CI on push).
2. Run **`just gate-landing`** before every push. It is the same set of checks CI
   runs: oracle pristine, `rustfmt --check`, `clippy -D warnings` (harness feature
   set), `cargo deny`, rustdoc with warnings as errors on the four published crates,
   workflow YAML, nextest in both dispatch modes, the census, `whereat`, and the
   public-API snapshots. `cargo test -p <crate> --lib` is **not** a gate: every byte
   gate is an integration target.
3. Commit with **explicit paths** (`git add <files>`, never `-A`/`-u`/`.`), and name
   the integration targets and counts you ran in the message. `.git-blame-ignore-revs`
   lists the whole-tree format commit.
4. If a `pub` item changed, `just api-doc` and commit the regenerated snapshots.
5. Record as you go: a `KB-*` body in `docs/KNOWN_BUGS.md` plus its one-line index in
   `CLAUDE.md` for every real bug (keep `CLAUDE.md` under 50 KB); a `STATUS.md`
   entry per landing (newest first; older cycles are archived under
   `docs/archive/`); perf records as `benchmarks/*.md` with a pointer from
   `docs/CLAUSE_STATUS_LOG.md`; named-but-unmeasured axes in
   `docs/COVERAGE_QUEUE.md`.
6. Push the branch, let CI go green on every leg (the aarch64 legs are the ones a
   local x86-64 box cannot run), then fast-forward `main`.

## Lint and doc policy

- `clippy -D warnings` over every target. The per-crate `#![allow(...)]` blocks
  ("Clippy policy" at the top of each `lib.rs` and `tests/all/main.rs`) name the lints
  that fight a line-for-line C port — argument counts and index loops that mirror the
  reference, verbatim C constants, NaN-sensitive float comparisons — each with its
  reason. Add to those lists only with a reason of the same kind.
- `rustdoc -D warnings` on the published crates. `zenav1-aom-encode` and
  `zenav1-aom-decode` warn on missing docs for their default (consumer) surface.
  Links to implementation items behind `__internals` are allowed by policy.
- Diagnostics go through `aom_dsp::trace` (`trace_on!`, `trace_out!`), never a raw
  `eprintln!` or an environment read in a published crate.

## Public API

The consumer surface is `zenav1-aom-encode::key_frame` and `zenav1-aom-decode::frame`
plus their config/error types; everything else is `pub(crate)` unless the default-off
`__internals` feature is on (the harnesses enable it). New public items need a doc
comment, a snapshot regeneration, and a reason to be public. Every `pub` type on the
`key_frame` surface is `#[non_exhaustive]`.
