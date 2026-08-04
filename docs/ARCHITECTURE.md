# zenav1-aom — architecture & gate enforcement

Target reference: **libaom v3.14.1** (pinned). Everything is defined as bit-for-bit
equivalence against a from-source build of that exact tag.

> Rewritten 2026-07-19 to match the tree. The previous version described an original
> design that was never built (a 10-crate split with `aom-predict` / `aom-rc` /
> `aom-rdo`, a `harness/` directory, `perf/baseline.json`, criterion benches). None of
> those exist. What follows is verified against the source.

## Guiding principle

We never claim a module "done" on inspection. A module is done when a **differential
harness** feeds identical input to the C reference and the Rust port and asserts
byte-identical output, over (a) a fixed corpus and (b) randomized fuzzing. The harness
is the product; the port is downstream of it.

**Evidence hierarchy** — not all oracles are equal, and the difference has cost real
debugging time:

1. **Real exported C function** — best. The port is compared against libaom's own symbol.
2. **Facade over a real C function** — acceptable. A thin shim that calls the real thing.
3. **Verbatim transcription** — weakest. A hand-copied C algorithm can carry a *shared*
   bug that the differential structurally cannot catch. This has happened here (see the
   KB-5 `is_cfl_allowed` entry, where port and reference shared a transcribed gate).

## Crate decomposition

**Seven packages** (corrected 2026-08-03: this said "Six" and omitted `zenav1-aom-dsp-bench`).
The 2026-07 consolidation collapsed the 13 DSP/entropy kernel crates into one — fine-grained
crates aided parallel porting, but a public release wants a small surface. Four are
publishable; three are `publish = false`.

| crate | mirrors (libaom) | bit-exact oracle |
|-------|------------------|------------------|
| `zenav1-aom` | facade: re-exports under `decode` / `encode` features | — |
| `zenav1-aom-dsp` | `aom_dsp/` + `av1/common/` kernels (see module table) | per-fn C export |
| `zenav1-aom-decode` | `av1/decoder/*` | full-frame decode |
| `zenav1-aom-encode` | `av1/encoder/*` | full bitstream |
| `zenav1-aom-sys-ref` | FFI to the pinned C libaom (oracle only, **dev-dep**) | — |
| `zenav1-aom-bench` | Gate-3 harness + every whole-frame encoder gate (bench-only) | — |
| `zenav1-aom-dsp-bench` | port-only DSP kernel benchmarks, no C oracle (bench-only) | — |

`zenav1-aom-dsp` is the shared kernel crate, organised by libaom module. **LOC re-measured
2026-08-03** (`find crates/aom-dsp/src/<module> -name '*.rs' | xargs wc -l`); the previous
figures were the 2026-07-19 snapshot and several had nearly doubled — `transform` read 8,155
against 15,140, `intra` 2,672 against 4,631, `cdef` 1,556 against 2,561. Treat any number in
this table as decaying from its measurement date, not as a property of the tree.

| module | LOC | module | LOC |
|---|---|---|---|
| `entropy` | 16,565 | `cdef` | 2,561 |
| `transform` | 15,140 | `loopfilter` | 2,116 |
| `quant` | 10,659 | `inter` | 1,769 |
| `txb` | 4,787 | `census.rs` | 1,060 |
| `intra` | 4,631 | `dist` | 847 |
| `restore` | 3,450 | `dispatch` | 206 |
| | | `convolve` | 160 |
| | | `recon` | 134 |
| | | `lowbd.rs` | 112 |

### Two invariants worth protecting

**A consumer build compiles zero C.** `zenav1-aom-sys-ref` is a *dev-dependency* of every
crate and is the sole `build.rs` in the workspace, so nothing downstream invokes cmake or
touches the oracle. Only `cargo test` builds C.

**The facade's features gate cleanly.** `decode` and `encode` are optional dependencies;
`--no-default-features --features decode` excludes the encoder entirely. Note the payoff
is bounded: `aom-dsp` is not optional, so decode-only still compiles all of it. Measured
at `benchmarks/build_time_decompose_2026-07-19.md` — 9.5% wall-clock saving, 30% CPU.

### SIMD

Dispatch goes through **archmage 0.9.27** + **magetypes** (capability tokens, `#[arcane]` /
`#[rite]`), with an `avx512` cargo feature. Scalar paths are the bit-exact reference;
vector paths must produce identical output, matching libaom's own C-vs-SIMD contract.
There is no per-crate `simd/` submodule — that was the old design.

## The C oracle

`upstream/` is a git submodule pinned to the reference commit. `crates/aom-sys-ref/build.rs`
checks the toolchain, auto-initialises the submodule, cmake-builds libaom in the
deterministic single-thread config, caches the result stamped by submodule SHA, then
compiles and links the shim translation units. No manual setup step.

The exact oracle build config lives in `reference/BUILD_CONFIG.md`.

## The four gates, made mechanical

1. **Decoder correctness** — `crates/aom-decode/tests/` (16 files as of 2026-08-03 — this
   read "11 files as of 2026-07-19"; `conformance_corpus.rs` and `real_bitstream.rs` are the
   broad ones, and `config_permutations_decode.rs` covers the crossings the conformance corpus
   does not contain). The corpus is the
   official AV1 decode-conformance set; `xtask/conformance.py` parses libaom's own
   `upstream/test/test-data.sha1` manifest, fetches the vectors, and categorises them by
   bit-depth and feature. Each vector ships a companion `.md5` holding one MD5 per decoded
   frame — that per-frame list is the golden answer libaom's own tests assert against, and
   it is ours. CI scope: `xtask/conformance.py --fetch --scope intra`.

2. **Encoder correctness** — `crates/aom-encode/tests/` (99 files as of 2026-08-03) plus the
   whole-frame gates in `crates/aom-bench/tests/` (39 — this read 17). The contract is
   byte-identity of the emitted bitstream vs real `aomenc` across `--cpu-used 0..9`. Gates are
   named `encoder_gate_*` / `kb*` / `s4cov_*` and assert full byte-identity, not a tolerance.
   As of 2026-08-02 the configuration-permutation grid (`config_permutations.rs`, 10 speeds x
   26 axis levels) has **zero pinned cells**.

   Several gates are **self-promoting**: they pin a *known* divergence by asserting it is
   still present, so the test fails the moment the port becomes byte-exact — at which point
   you promote the cell into the byte-exact list. This keeps a known gap honest instead of
   letting it rot as a silent skip.

3. **Performance** — `crates/aom-bench`, using **zenbench** paired/interleaved benchmarks
   against the real C oracle in-process via `aom-sys-ref`, plus a callgrind profile driver.
   **The acceptance bar is ≤ 1.5× C** (the 2026-07-20 user directive); ≤ 1.20× is retained as
   the stretch target. (Corrected 2026-08-03: this line said "Target is ≤ 1.20× C", which
   contradicted CLAUDE.md's Gate 3 and CHANGELOG.md.) Decode meets the bar at the 4K headline
   cells only; the encoder was first profiled on 2026-08-02 and is at 3.15× on the study cell.
   Results are committed under `benchmarks/` with a companion `.meta` recording commit, host,
   and the exact command. The encoder A/B harness is `scripts/eprof_ab.sh`, and `ROTATE=1`
   (rotating the arm order so position cannot confound) is now its default — see
   `docs/DIFFERENTIAL_PLAYBOOK.md` §6.

4. **Coverage** — `xtask/coverage.py` auto-derives the feature checklist from libaom's live
   CLI (`aomenc --help` / `aomdec --help`) and the control-enum surface, then cross-references
   `coverage/feature_map.json`. A feature is green **only** if it maps to a passing test id;
   the tool does not invent green. Standing audits live in `coverage-audit/`.

5. **zenavif integration** — `crates/aom-encode/tests/avif_parity.rs` muxes the port's
   byte-exact AV1 payload into an AVIF still via `zenavif-serialize`, then closes the loop
   twice: the container round-trip must return the coded bytes verbatim, and the extracted
   stream must decode (through the port's own decoder) to pixels matching a decode of real
   aomenc's stream.

## Divergence policy

Bit-identity is the default and the enforced contract.

Known divergences are tracked as numbered **KB entries in `CLAUDE.md`** ("Known Bugs"), each
with `file:line` references, the root cause once found, and the gate that closes it. There
is no `docs/DIVERGENCES.md`. Module-level progress is tracked in `STATUS.md`.

A divergence is closed only by a landed fix verified on `origin/main` — never by relaxing a
test. Adding `#[ignore]`, loosening a threshold, changing a golden, or letting a test skip
itself at runtime all count as relaxing it.

Classes where C libaom is not self-consistent (multithread tie-breaks, `--enable-thread` vs
single, some `float` reduction orders) are pinned to the **single-thread, C-scalar,
deterministic** build config of the reference so the target is well-defined. We match *that*.
