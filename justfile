# aom-rs task runner. `just --list` for a summary.

# Full differential + gate suite (tokens enabled — SIMD dispatch live).
test:
    cargo test --workspace --no-fail-fast

# SCALAR-PIN mode: AOM_FORCE_SCALAR disables every archmage SIMD token
# process-wide (crates/aom-dispatch), so every kernel dispatch falls through
# to the scalar port. The full suite passing here proves SIMD work left the
# scalar path untouched. Run BOTH `just test` AND `just test-scalar` before
# landing any kernel change.
test-scalar:
    AOM_FORCE_SCALAR=1 cargo test --workspace --no-fail-fast

# FAST full-suite run — identical coverage to `just test`, built at opt-level 3
# (the `test-fast` profile) so the e2e byte gates (real full encodes/decodes,
# minutes each unoptimized) run far quicker. debug-assertions + overflow-checks
# stay on, integer results are identical (byte gates stay byte-exact). The
# first run pays a one-time optimized compile; reruns are fast. Use this for
# routine "did I break anything" checks; `just test` remains the debug default.
test-fast:
    cargo test --profile test-fast --workspace --no-fail-fast

# FAST scalar-pin run (AOM_FORCE_SCALAR, opt-level 3). Pair with `test-fast`
# for the both-dispatch-modes parity gate at a fraction of the debug wall time.
test-fast-scalar:
    AOM_FORCE_SCALAR=1 cargo test --profile test-fast --workspace --no-fail-fast

# FASTEST full-suite run — same coverage as `just test-fast`, but scheduled by
# cargo-nextest. Stock `cargo test` drains test BINARIES sequentially, threading
# only within each one; nextest puts every test across every binary into ONE
# global work pool. That matters here because the cost is wildly concentrated:
# aom-decode sums to 458s with 364s (80%) in a single binary, and aom-dsp is 57s
# across 95 binaries with most at ~0.00s.
#
# MEASURED 2026-07-19 (aom-dsp, 342 tests / 95 binaries, run-only wall):
#   cargo test sequential  57.02s | nextest  12.07s  = 4.7x, within 7% of the
#   floor (the slowest single test, dv_ref_diff::find_samples_matches_c 11.31s).
# A pool cannot beat its slowest single test — see benchmarks/test_cycle_time_
# 2026-07-19.md, which measured binary consolidation and lld and rejected both.
# Its BUILD-TIME finding still stands and was reproduced 2026-09-09 (lld/mold are
# a wash; the relink win is 13.6s -> 10.7s because that time is the LIB compile,
# not links). Consolidation was done anyway on three grounds that record did not
# weigh — disk 8.19 GB -> 202 MB, and it is what makes the `__internals` API gate
# and a per-package test opt-level expressible. See its 2026-09-09 addendum.
#
# Needs cargo-nextest (prebuilt: curl -LsSf https://get.nexte.st/latest/linux
# | tar zxf - -C ~/.cargo/bin). CI still uses plain `cargo test`.
test-next:
    cargo nextest run --cargo-profile test-fast --workspace --no-fail-fast

# Same, scalar-pinned (pairs with `test-next` for the both-dispatch-modes gate).
test-next-scalar:
    AOM_FORCE_SCALAR=1 cargo nextest run --cargo-profile test-fast --workspace --no-fail-fast

# Where is the suite time actually going? Prints the 25 slowest tests. nextest
# reports per-test timing, which stock libtest will not on stable — this is how
# you find the long poles worth splitting.
test-slowest:
    cargo nextest run --cargo-profile test-fast --workspace --no-fail-fast --final-status-level slow 2>&1 | grep -E '(PASS|SLOW|FAIL)' | sort -t'[' -k2 -rn | head -25

# QUICK SIMD-parity subset (opt-level 3) — the Gate-3 kernel crates' per-kernel
# SIMD==scalar differentials + the transform 2-D permutation-equality gate,
# WITHOUT the minutes-long encoder e2e gates. For tight iteration on SIMD /
# transform work; run `just test-fast` + `just test-fast-scalar` before landing.
# Measured 2026-07-17: ~45s cold (optimized build-dominated), test-RUN a few
# seconds — the transform per-kernel differential is 1.5s here vs ~10s in debug.
test-simd:
    cargo test --profile test-fast -p zenav1-aom-dsp --test all --no-fail-fast -- txfm2d_simd_perm_diff:: quantize_fp_simd_diff:: cdef_filter_simd_diff:: sad_simd:: hbd_variance_simd_diff:: txb_init_levels_simd_diff:: intra_simd_diff:: lpf_simd_diff:: wiener_simd_diff:: convolve_diff::

# CONTENT-FAMILY COVERAGE GATE. Censuses the four committed `winperf` contents
# through the port and asserts every coding-tool family (directional intra,
# CFL/chroma, rect + small leaves, 4-pt transforms, palette, intraBC) is still
# reached above a pinned share — so a content, encoder or speed-feature change
# that silently stops exercising a family fails loudly instead of turning a
# future band into a structural zero (benchmarks/winperf_family_census_2026-08-03.md).
#
# Needs `--features census` (default-off: a census build is not a timing build)
# and does NOT need the C oracle, so it runs on any box in ~6 s.
census-gate:
    cargo test --release -p zenav1-aom-bench --no-default-features --features census --test content_family_census

# THE ENCODER LANDING GATE (added 2026-08-30, KB-42). `-p <crate> --lib` runs NONE
# of the byte-identity gates — they all live in `tests/` integration targets, and
# the coverage census lives behind a non-default feature. KB-42 is what that costs:
# four landings gated on `--lib` + named diff tests, 23 byte/RD gates and the census
# broken across six red CI runs before anyone looked. Run this before pushing any
# encoder change; it is the subset of CI that an encoder edit can break.
gate-encode:
    cargo test --profile test-fast -p zenav1-aom-encode --no-fail-fast
    cargo test --profile test-fast -p zenav1-aom-bench --no-fail-fast
    just census-gate

# THE LANDING GATE. Same coverage as the old `gate-encode` + `test-fast` +
# `test-fast-scalar` trio, MEASURED, in a fraction of the wall time.
#
# Two measurements from 2026-09-09 justify the shape, and both are the kind of
# claim KB-42 says not to make on a name-level argument:
#
#  1. `gate-encode` is a strict SUBSET of the workspace run. Measured on the
#     2026-09-08 gate log: 189 distinct binaries vs 319, and the only one not in
#     the superset is `content_family_census` (a non-default `census` build). The
#     old trio therefore ran 188 of 189 binaries TWICE -- 779 s = 13 min per run.
#     Feature resolution was checked empirically, not assumed: after a
#     `--workspace` build, `-p zenav1-aom-encode` and `-p zenav1-aom-bench`
#     recompile NOTHING and `--workspace` does not rebuild after them either, so
#     the two selections share artifacts and cannot differ in features.
#  2. `cargo test` drains binaries serially -- measured load 3.94 on 24 cores,
#     ~84 % idle. nextest: whole workspace 57.4 min -> 340.9 s (~10x).
#
# `gate-encode` is KEPT as the fast pre-check while iterating on encoder work.
# It is just no longer worth running alongside the workspace gate.
gate-landing:
    just test-next
    just test-next-scalar
    just census-gate
    just gate-whereat

# The `whereat` feature gates 4 sites in aom-decode AND its own test module, so
# a default build never COMPILES them. That is a verification hole, and it bit:
# consolidating tests/ moved `whereat_entries.rs` one directory deeper, breaking
# its `include_bytes!("data/...")`, and a 1502-test green run did not notice
# because the target was never built. Any feature that gates a test target must
# be built by the landing gate, or the gate is lying about its coverage.
gate-whereat:
    cargo test --profile test-fast -p zenav1-aom-decode --features whereat --test all -- whereat_entries::

# The census TOOL. `just census-corpus` prints the family table for the four
# harness contents; add `yuv:<path>:<w>x<h>`, `scr:<path>:<w>x<h>` (screen
# bootstrap) or `real:<vector>` sources, `--speed N`, `--cq N` and
# `--knobs palette,intrabc` by editing the line or calling cargo directly.
census-corpus:
    cargo run --release -p zenav1-aom-bench --features census --example content_census -- winperf:photo winperf:detail winperf:smooth winperf:screen

# Gate-3 paired benchmark, port vs C oracle (zenbench interleaved rounds).
# QUIET BOX ONLY — the resource gate flags noisy rounds; a loaded box makes
# the numbers worthless. Results: commit to benchmarks/ per CLAUDE.md.
bench-gate3:
    cargo bench -p zenav1-aom-bench --bench gate3

# Harness smoke: proves the bench runs end-to-end (byte-verify + tiny rounds,
# resource gate off). NUMBERS ARE MEANINGLESS — never quote them.
bench-smoke:
    AOM_BENCH_SMOKE=1 cargo bench -p zenav1-aom-bench --bench gate3

# Callgrind instruction-count profile of one Gate-3 cell (load-tolerant).
# kind=enc|dec side=port|c cell=<label> iters=N; see gate3_profile --help
# for cell labels. Output: /tmp/cg_<cell>_<side>.out (annotate with
# `callgrind_annotate --threshold=95 <file>`).
profile kind side cell iters:
    cargo build --profile profiling -p zenav1-aom-bench --bin gate3_profile
    valgrind --tool=callgrind --callgrind-out-file=/tmp/cg_{{cell}}_{{side}}.out \
        ./target/profiling/gate3_profile {{kind}} {{side}} {{cell}} {{iters}}
    callgrind_annotate --threshold=95 /tmp/cg_{{cell}}_{{side}}.out | head -60

# Regenerate the transform 1-D kernels (scalar + AVX2 lane twins) from the
# extracted C. Scalar output must be byte-identical to the committed files
# (verified: `diff` after regenerating). The lane files are the SIMD twins.
gen-txfm1d:
    python3 xtask/transpile_txfm1d.py --inv reference/extracted/idct4.c reference/extracted/idct8.c reference/extracted/idct16.c reference/extracted/idct32.c reference/extracted/idct64.c reference/extracted/iadst8.c reference/extracted/iadst16.c > crates/aom-dsp/src/transform/inv_txfm1d_gen.rs
    python3 xtask/transpile_txfm1d.py reference/extracted/fdct8.c reference/extracted/fdct16.c reference/extracted/fdct32.c reference/extracted/fdct64.c reference/extracted/fadst8.c reference/extracted/fadst16.c > crates/aom-dsp/src/transform/txfm1d_gen.rs
    python3 xtask/transpile_txfm1d.py --inv --lanes reference/extracted/idct4.c reference/extracted/idct8.c reference/extracted/idct16.c reference/extracted/idct32.c reference/extracted/idct64.c reference/extracted/iadst8.c reference/extracted/iadst16.c > crates/aom-dsp/src/transform/simd/inv1d_v3_gen.rs
    python3 xtask/transpile_txfm1d.py --lanes reference/extracted/fdct8.c reference/extracted/fdct16.c reference/extracted/fdct32.c reference/extracted/fdct64.c reference/extracted/fadst8.c reference/extracted/fadst16.c > crates/aom-dsp/src/transform/simd/txfm1d_v3_gen.rs
    python3 xtask/transpile_txfm1d.py --inv --lanes16 reference/extracted/idct4.c reference/extracted/idct8.c reference/extracted/idct16.c reference/extracted/idct32.c reference/extracted/idct64.c > crates/aom-dsp/src/transform/simd/inv1d_v3_i16_gen.rs
    python3 xtask/transpile_txfm1d.py --lanes16f reference/extracted/fdct8.c reference/extracted/fdct16.c reference/extracted/fdct32.c reference/extracted/fdct64.c reference/extracted/fadst8.c reference/extracted/fadst16.c > crates/aom-dsp/src/transform/simd/fwd1d_v3_i16_gen.rs

# The forward i16 lane-narrowing audit: prints M* per (kernel, cos_bit) and the
# bd8 reach per (tx_size, tx_type). The M* table in
# `transform/simd/lowbd16_fwd.rs` is the MINIMUM over cos_bit of what this
# prints; re-run it if any kernel or the cospi table changes.
audit-i16-fwd:
    python3 xtask/audit_i16_fwd.py --cells

# ARMED-TOOL DECODE GATE (KB-29). Byte-identity gates prove conformance only
# where a reference stream exists and is asserted equal; every `ToggleKnobs` arm
# aomenc cannot be driven into is a configuration the port can PRODUCE and that
# nothing ever decodes. This runs each such arm through the REAL C decoder, the
# port decoder, and (when `dav1d` is on PATH) dav1d, asserting all three agree.
# The dav1d leg is wired HERE, not inside the test — the skip decision belongs
# to the caller. Included in `just test` / `just test-fast` as an ordinary
# aom-bench test; this recipe exists to add the dav1d leg.
gate-armed-decode:
    AOM_DAV1D_BIN="$(command -v dav1d || true)" cargo test --profile test-fast -p zenav1-aom-bench --test all -- --nocapture armed_tools_decode_gate::

# ENCODE TIME vs libaom, on the standalone entry, with byte-identity asserted
# per cell (byte identity IS the RD claim — same bytes means the same
# rate-distortion point, not a nearby one). `#[ignore]`d in the ordinary suite
# because it is a timing measurement and `CLAUDE.md` requires those isolated.
# Record: benchmarks/encode_perf_vs_libaom_2026-09-08.md.
gate-encode-perf:
    cargo test --profile test-fast -p zenav1-aom-bench --test all -- --ignored --test-threads 1 --nocapture encode_perf_vs_libaom::

# CANCELLATION-LATENCY GATE (the three timing arms of
# `crates/aom-bench/tests/cancel_latency.rs`). They are `#[ignore]`d in the
# ordinary suite ON PURPOSE: they measure WALL-CLOCK latency, which a parallel
# test pool cannot measure. Their own `timing_serial()` mutex serialises the
# three arms against each other but is process-local, so `cargo nextest`
# (one process per test) defeats it and ~20 unrelated test processes compete —
# measured 33.1 ms against a 22.8 ms bound in a whole-workspace nextest run and
# 17.0 ms for the same code run alone.
#
# `--test-threads 1` is the methodology the committed record
# (`benchmarks/decode_cancel_latency_2026-08-06.*`) was taken with. Also wired
# as its own CI step, so the coverage is not lost by the `#[ignore]`.
gate-cancel-latency:
    cargo test --profile test-fast -p zenav1-aom-bench --test all -- --ignored --test-threads 1 --nocapture cancel_latency::

# LOCATED-ERROR ENTRIES (`whereat` feature, default-off). `decode_frame_obus_at`
# and `decode_frames_at` are `#[cfg(feature = "whereat")]`, so no workspace test
# run compiles them — the feature could be broken indefinitely and nothing would
# say so. This is the only invocation that builds it; also wired as a CI step on
# the pure-Rust portability job.
test-whereat:
    cargo test -p zenav1-aom-decode --features whereat --test all -- whereat_entries::

# CROSS-ENCODER INTRABC DECODE GATE (GitHub #5). The armed gate above decodes
# streams from THIS port's encoder, and the conformance corpus is entirely
# libaom-encoded — so neither covers "a conformant IntraBC stream from a
# different encoder", which reaches neighbour configurations libaom's own
# encoder never emits. This decodes four committed REAL SVT-AV1 v4.2.0 C
# encodes through the real C decoder, the port decoder, and (when `dav1d` is on
# PATH) dav1d, asserting all agree pixel-for-pixel. Same caller-owns-the-skip
# rule: the dav1d leg is wired here, not inside the test. Regenerate/extend the
# corpus with scripts/svt_interop/.
gate-svt-interop:
    AOM_DAV1D_BIN="$(command -v dav1d || true)" cargo test --profile test-fast -p zenav1-aom-bench --test all -- --nocapture svt_interop_decode_gate::

# ZENSIM-QUALITY (Zq) TARGET CENSUS — the phase-A harness for the
# dependency-injected target loop (`zenav1-aom-target`). The crate's LIBRARY
# has zero dependencies (encoder, decoder and judge all live behind the
# caller's `trial(qindex)` closure), so the judge is bolted on here: the
# `census` feature pulls the git-pinned zensim + png that only this example
# needs, which is why a default `cargo test --workspace` never builds either.
#
# Needs `aomenc` / `aomdec` on PATH (the injected encoder/decoder) and a
# zensim bake at $ZQ_BAKE. Args: <corpus.tsv: path\tname\tclass> <targets,csv>
# <max_encodes k> <out.tsv>. Cells + methodology:
# benchmarks/zensim_zq_target_wave_2026-08-29.md
zq-census corpus targets k out:
    cargo run --release -p zenav1-aom-target --features census --example zq_census -- {{corpus}} {{targets}} {{k}} {{out}}

# Interleaved native/scalar DSP pairs; no C oracle required.
arm-dsp-tiers-macos group="":
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 /usr/bin/time -l cargo bench --locked -p zenav1-aom-dsp-bench --bench dsp_kernels -- --group={{group}} --format=llm > "$HOME/tmp/aom-arm-dsp-tiers.log" 2>&1

# Regenerate the public-API surface snapshots (docs/public-api/)
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml
