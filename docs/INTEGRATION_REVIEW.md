# Integration review — `perf/gate3-txfm-i16-batch` -> `origin/main`

Reviewed 2026-09-24 at `973f503` (working tree clean; local is 4 commits ahead of
`origin/perf/gate3-txfm-i16-batch`). Every claim below was measured in the repo at that
SHA, not inferred from commit messages. Nothing was merged, rebased, pushed or committed.

## (a) Verdict

**Fast-forwardable and content-ready, but NOT yet gate-verified. Do not merge until the
three items in (b) are done.** The branch is 160 commits ahead / 0 behind `origin/main`
(`git rev-list --left-right --count origin/main...HEAD` = `0 160`), the tree is clean, the
upstream gitlink is unchanged (`03087864` on both sides), `Cargo.toml`/justfile drift is all
intentional and documented, and the one class of change that could hide a relaxed gate
(61 removed `assert` lines) was verified line by line: every removal is a refactor to an
equivalent accessor, a pin promoted to a byte-identity gate, or an envelope refusal
replaced by a ported arm (details in (c)-7). No `#[ignore]` was added except the
documented `probe_zenaom_trial_matrix` probe.

What blocks is process, not code: the mandatory landing gate has not been run on this
branch by any record, CI has run zero times on it, and the enforced public-API snapshot is
stale at HEAD, which means `just gate-landing` fails today.

## (b) Blocking issues

1. **`just api-doc-check` FAILS at HEAD** (`apidoc` test `public_api_surface_docs_are_current`:
   "committed public-API snapshot for zenav1-aom-encode is stale"). The drift is from
   `347a8c8` (KB-68 fix): `predict_skip_txfm` gained a `u32` parameter, two
   `skip_txfm_level` fields and two `skip_txfm_level_*_eval` fns were added, and the
   `zenav1-aom-encode.txt` header line changed (internal count 4927 -> 4931). Last regen
   was `9ebf4e8` (2026-09-18). Fix: `just api-doc` and commit
   `docs/public-api/zenav1-aom-encode.{txt,internal.txt}` (73-line diff, captured and
   reverted during this review). This is the anti-KB-42 gate catching exactly what it was
   built for; it is also the proof that `gate-landing` was not run on the last commits.
2. **No STATUS entry newer than 2026-09-10 claims `gate-landing`, `test-next`, nextest or a
   `--workspace` run** (`STATUS.md` lines 1-868 mention only `gate-encode` once and
   `self_contained_key_frame`/`self_contained_tools`). The two unpushed fix commits
   `2931675`/`347a8c8` name no integration targets at all in their bodies (CLAUDE.md:
   "A landing's own gate list in the commit message must name the integration targets it
   ran"); their STATUS entry names four targets plus "129/129 aom-encode lib tests", and
   CLAUDE.md is explicit that `--lib` is not a gate. Overall 85/160 commit bodies mention
   any gate at all. Run the full `just gate-landing` once at HEAD (see (d)).
3. **CI has never run on this branch.** `.github/workflows/ci.yml` triggers only on
   `push: [main]`, `pull_request: [main]` and `workflow_dispatch`; `just ci-status` shows
   the newest run is `1434bc3` (2026-09-11), the merge-base era, and `gh run list --branch
   perf/gate3-txfm-i16-batch` is empty. All 160 commits are CI-unverified, including the
   x86-64 and aarch64 legs that KB-43 shows can diverge from a local box. Open a PR (or
   `workflow_dispatch` the pushed tip) and get a green run before the fast-forward.

## (c) Decluttering / unification punch-list

### 1. `upstream/` submodule working tree violates the documented convention (HIGH)

The convention is documented in three places and the tree breaks all three:

- `docs/KNOWN_BUGS.md:754-760` (KB-55 body): "The matching C-side prints are preserved as
  `docs/upstream-divergence-debug-2026-09-12.patch` — the `upstream/` submodule itself
  was reverted to pristine so the oracle stays the pinned tree."
- `STATUS.md:828-831`: same statement.
- `docs/HANDOFF-TOGGLES.md:113-116` + `docs/ITERATION_PLAYBOOK.md:160`: C instrumentation
  goes in a **sibling** ar-swapped `libaom.a`, never the shared oracle in place, and is
  reverted before landing.

Measured state: `git -C upstream status` shows **17 modified files, +965/-9 lines**
(`git -C upstream diff --stat`), plus untracked `build/` (the oracle build output that
`crates/aom-sys-ref/build.rs:193` writes; harmless, hidden by `ignore = dirty` in
`.gitmodules`). The superproject cannot see this: `.gitmodules` sets `ignore = dirty`, so
`git status` at the root reports clean while the oracle every differential test links
against is instrumented. The gitlink SHA is unchanged, so nothing is committed wrong; the
risk is the oracle binary silently carrying `fprintf` side effects and a 589-line patch doc
that no longer describes what is on disk.

The instrumentation is gated on 17 env names. Cross-referenced against the port:

| C env gate (`getenv`) | upstream files | port twin committed? | served KB |
|---|---|---|---|
| `AOM_TX_DBG` (14 sites) | reconintra.c, encodemb.c, intra_mode_search.c/.h, palette.c, partition_search.c, tx_search.c | yes (`tx_search.rs:3377`, cached) | KB-55/59/60 (closed) |
| `AOM_PART_DBG` (10) | partition_search.c | yes (`partition_pick.rs:2970`, cached) | KB-55/59 (closed) |
| `AOM_UV_DBG` (8) | reconintra.c, intra_mode_search.c, tx_search.c | yes (`intra_uv_rd.rs:167`, cached) | KB-60 (closed) |
| `IBC_TRACE` (5) | rdopt.c, tx_search.c | yes (`intrabc_search.rs:2466,2648`, cached) | KB-68 (closed 2026-09-24) |
| `AOM_SCT_TRIAL_DBG` (3) | encoder.c, encoder_utils.c | yes (`key_frame.rs:2517`) | KB-66/67 (resolved) |
| `AOM_CDEF_DBG` (3) | bitstream.c, pickcdef.c | yes (`pickcdef.rs:1171`, `header.rs:157`) | KB-56/57 (closed) |
| `AOM_SYM_TRACE` (2) | entdec.c | yes (`entropy/dec.rs:45`) | KB-65/68 |
| `AOM_PAL_TRACE` (2) | decodemv.c | yes (`entropy/partition.rs:5170`) | KB-65 |
| `AOM_DQ_DBG` (2) | encodeframe.c | yes (`key_frame.rs:3275`) | KB-58 (closed) |
| `AOM_SSM_DBG` (1) | partition_search.c | yes (`encode_sb.rs:637`) | KB-59 (closed) |
| `AOM_SCT_DBG` (1) | encoder.c | yes (`key_frame.rs:2972,3928`) | KB-55/67 |
| `AOM_PART_TRACE` (1) | decodeframe.c | yes (`aom-decode/src/lib.rs:6435`) | KB-65 |
| `AOM_CTXB_TRACE` (1) | decodetxb.c | yes (`pack.rs:3022`) | KB-65 |
| `AOM_LEAF_TRACE` (3) | decodeframe.c | **no port twin** | — |
| `AOM_CMODE_TRACE` (2) | decodemv.c | **no port twin** | — |
| `AOM_CDV_TRACE` (2) | decodemv.c | **no port twin** | — |
| `AOM_CDP_TRACE` (1) | decodeframe.c | **no port twin** | — |

Every KB these served is closed, so the whole C-side set is dead for current work. The four
unpaired decoder-side traces (`AOM_LEAF/CMODE/CDV/CDP_TRACE`) are one-sided clutter with
nothing on the port side to pair against.

Action: regenerate the patch doc from the live tree (`git -C upstream diff >
docs/upstream-divergence-debug-2026-09-24.patch`, or overwrite the 09-12 file and update
the two references at `STATUS.md:830` and `docs/KNOWN_BUGS.md:759`), then
`git -C upstream checkout -- .` and `cargo clean -p zenav1-aom-sys-ref` so the oracle is
relinked pristine before `gate-landing` runs. Do the gate run AFTER the revert, so the
byte-identity numbers are against the pinned oracle. Consider dropping `ignore = dirty`
from `.gitmodules` (or adding a `just`/CI check that `git -C upstream diff --quiet`) so this
cannot recur invisibly; CLAUDE.md's rule "an unenforced check is a document" applies.

### 2. `docs/upstream-divergence-debug-2026-09-12.patch` (MEDIUM) — keep, but it is stale

589 lines, 4 env gates (`AOM_TX_DBG` x8, `AOM_PART_DBG` x3, `AOM_CDEF_DBG` x3,
`AOM_DQ_DBG` x2) over 10 hunks; the live tree has 965 lines and 17 gates. It does not
reverse-apply against the live tree (`git -C upstream apply --check -R` fails on
encodemb.c:756, tx_search.c:2367, intra_mode_search.c:1635), i.e. the working tree has
moved past it. Verdict: **keep the artifact class** (it is the designated home for C-side
prints and is referenced from STATUS and KB-55), but **replace its content** with the full
current diff per item 1, and name it by the date it was captured. It belongs in `docs/`
rather than `benchmarks/` because it is method, not measurement.

### 3. Scratch-looking files that are intentional (LOW — no action)

- `scripts/callgrind_counts.py`: referenced from `docs/ITERATION_PLAYBOOK.md:81-82` as
  part of the callgrind loop; all 24 tracked `scripts/*` are tool inventory. Keep.
- `benchmarks/*.md` + `.band1024s3.tsv` (28 files added on the branch): the repo
  convention (CLAUDE.md clause (4), `docs/CLAUSE_STATUS_LOG.md`) is that every landing's
  band record is a committed `benchmarks/` file. Keep. Four records are **orphans**,
  referenced from no `.md`/`.rs`/justfile outside `benchmarks/`:
  `encoder_4k_sweep_2026-09-12.md`, `encoder_lowbd_lpf_u8_sse2_mirror_2026-09-17.md`,
  `encoder_sse_u16_u8_constw_2026-09-17.md`, `encoder_txb_init_levels_chunks_2026-09-15.md`.
  Add one-line pointers in `docs/CLAUSE_STATUS_LOG.md` row (4) so they are discoverable
  (the log is the declared complete record of that row).
- `benchmarks/arm_audit_2026-09-06/*.log` (10 tracked `.log` files) predate the branch
  (`a7b1ab1`, on main); not this branch's clutter, but the only `.log` files in the tree.
- `[profile.ship]` in `Cargo.toml:24` (fat LTO, cgu=1) is documented in place and wired to
  no recipe or CI job; fine as an opt-in, but note it is measured (-6 %) and not what any
  gate builds.

### 4. Committed-vs-uncommitted trace pairs (MEDIUM)

Port side: 26 `AOM_*`/`IBC_TRACE` env reads were added on the branch, all committed, in
`crates/aom-encode/src/{key_frame,tx_search,partition_pick,pack,intrabc_search,pickcdef,
intra_uv_rd,encode_sb}.rs`, `crates/aom-dsp/src/entropy/{header,dec,enc,partition}.rs`
and `crates/aom-decode/src/lib.rs`. C side: none committed (item 1). So every pair is
half-committed by design; the patch doc is the mechanism that makes the C half durable and
it is stale. Resolving items 1-2 resolves this.

Two port-side nits while here:

- `crates/aom-decode/src/lib.rs:6435` (`AOM_PART_TRACE`, added `fca9b4e`) and `:4779`
  (`AOM_DV_TRACE`) call `env::var_os` uncached **per partition symbol / per DV** in the
  decoder walk. The encoder's equivalents were `OnceLock`-cached in `7099a04` after
  getenv measured 1.4 % flat; the decoder has the same shape and Gate 3 covers it too.
  Same for `crates/aom-dsp/src/entropy/partition.rs:5532` (`AOM_DQ_DEC`, per SB when
  delta-q is on). Cache them or delete them now that KB-65/68 are closed.
- `crates/aom-dsp/src/entropy/header.rs:157,1490` and `crates/aom-encode/src/pickcdef.rs:1171`
  are per-frame and harmless.

### 5. Manifests, justfile, public-API snapshots (LOW except the snapshot)

- Root `Cargo.toml`: only `[profile.ship]` added. `Cargo.lock`: `rayon` and `archmage`
  (dev-dep, `testable_dispatch`) — both explained in the crate manifests.
- `crates/aom-dsp/Cargo.toml`, `aom-encode`, `aom-bench`: the opt-in `rayon` feature chain
  (documented, byte-inert, default off) and `archmage` dev-dep. No `[patch]` sections
  anywhere.
- `justfile`: byte-identical to `origin/main` on this branch. No experiment recipes.
- `docs/public-api/`: stale for `zenav1-aom-encode` (blocking item 1). Since the last
  zenavif pin bump (`05879e7`), the **`aom-dsp` supported surface gained ~20 pub items**
  (`sse_u16_u8*`, `subtract_block_u16_u8`, `z1_high_u8e`, `kmeans::*`, `loopfilter::*_n`,
  `lowbd::*`, `par::*`). CLAUDE.md clause (1): "Re-run the `[patch]` compose whenever a
  `pub` signature in `aom-dsp`/`aom-encode` changes." Add that to the pre-merge list.
- `.gitmodules`, `.github/`, `CHANGELOG.md`: unchanged on the branch. CHANGELOG's
  `[Unreleased]` still ends at KB-48; 160 commits including KB-53..68 and tile threading
  have no entry. Not a gate, but it is the user-facing record if these crates publish.

### 6. `.devin/` committed inside a functional commit (LOW)

`.devin/hooks.v1.json` + `.devin/hooks/exec-timeout.py` (79 lines) landed in `56192f9`
("encode: per-SB stop polling, shipped Deadline token, fuzz bounded in time"), an
unrelated code commit. `.gitignore` ignores `.claude/` but not `.devin/`. Decide whether
agent tooling config is repo content; if yes, it deserves its own commit and a line in
`docs/ITERATION_PLAYBOOK.md`; if no, `git rm` it and add `.devin/` to `.gitignore`.
156/160 commits carry a `Generated with [Devin]` trailer, which the repo's memory rule on
co-author trailers may or may not intend to cover.

### 7. Duplicate test target (LOW, mechanical)

`crates/aom-bench/tests/highbd_inter_decode_envelope.rs` is **byte-identical** (blob
`4ab178ca`) to `crates/aom-bench/tests/all/highbd_inter_decode_envelope.rs`, which
`tests/all/main.rs:29` already includes. `aom-bench` does not set `autotests = false`, so
the top-level copy compiles and runs as a second, un-consolidated binary on every
`cargo test -p zenav1-aom-bench` — undoing the `a88e739` "320 binaries into 7"
consolidation for this one file. `origin/main` has only the `all/` copy. Delete the
top-level file. (The other top-level files — `content_family_census.rs`, the two
`internals_feature_guard.rs`, `encode_limits_and_estimate.rs` — are intentional
non-gated guards or `required-features` targets.)

### 8. Removed asserts — verified, none weakened (record for the merge note)

| file | commit | verdict |
|---|---|---|
| `aom-bench/tests/all/toggles_rd_close.rs` | `9dae19f` (KB-60) | stronger: 3 pinned-divergence asserts -> `run_grid_and_gate(.., true)` :342 requiring `bit_identical` on all 3 cells |
| `aom-encode/tests/all/self_contained_key_frame.rs` | `93afcea`/`7bae9f8`/`ff9df25` | stronger: `pins = vec![]` :1645, trap kept :1774-1780, all 8 ex-pins now in byte-identity sweep axes |
| `rd_pick_intra_sb_diff.rs`, `encode_sb_diff.rs`, `encode_intra_plane{,_uv}_diff.rs` | `b7820fa`, `aced766` | equivalent: `&r.qcoeff[..]` -> `r.qcoeff()` accessors (exact halves, `encode_intra.rs:308-315`); ta/tl now prefix-compared over C's length because the port array became fixed `[i8; 32]` — tail unchecked, recon equality unchanged |
| `cnn_partition_decision_diff.rs`, `cnn_partition_cnn_diff.rs` | `06d2879` | stronger: 1e-2 tolerance ceiling -> bit-exact vs the dispatched engine, + bd10/bd12 |
| `src/speed_features.rs` | `d217055` (KB-54) | stronger: disjunctive `assert!` -> exact `assert_eq!` :2652-2656 |
| `src/pickcdef.rs` | `7bae9f8` (KB-56) | envelope refusal replaced by the ported `CDEF_PICK_FROM_Q` arm :1024-1033 + range assert :1036 |

Optional strictness: assert the ta/tl tail equals its init value in the two plane-diff
tests; note that the two CNN diffs now trust `v3_tier_active()` to match real dispatch.

## (d) Gates to run before merge

In this order, all at HEAD, after item (c)-1's upstream revert + `cargo clean -p
zenav1-aom-sys-ref` so the oracle is the pinned tree:

1. `just api-doc` -> commit the regenerated `docs/public-api/zenav1-aom-encode.*` (else
   step 2 fails at its last stage).
2. `just gate-landing` (~11 min): `ci-yaml-check` + `test-next` + `test-next-scalar` +
   `census-gate` + `test-whereat` + `api-doc-check`. This is the mandatory list and no
   record shows it ran on any of the 160 commits.
3. KB-41 plane-dir census (`crates/aom-bench/tests/all/kb41_screen_detected_defaults.rs`
   with `ZENAV1_PLANES_DIR`) — CLAUDE.md requires it for IntraBC/palette work and the last
   four commits are IntraBC fixes.
4. `just gate-encode-perf` / one `just perf-band` on the ship cell against the
   `origin/main` arm — clause (4) is "met-but-thin" (1.384x, ~1 % margin) and the last
   two fixes touch `intrabc_search.rs`/`partition_pick.rs`; confirm the photo cell is
   still byte-identical AND inside 1.40x.
5. zenavif `[patch]` compose re-verify (`aom_encode_backend`, `aom_roundtrip_loss`,
   `resolved_routing`) because `aom-dsp`'s pub surface changed since the pin bump.
6. Push the branch, open the PR, get CI green on both arch legs (item (b)-3).

Should be RE-run rather than trusted: everything in 2, because the only recent evidence
is `gate-encode`, `self_contained_*` and `--lib` counts; and the 549/549 + 164/164 figures
in the branch description, which are correct as of their STATUS entries but were taken
against an instrumented oracle build.

## (e) `origin/fix/issue-8-highbd-inter-envelope` disposition

**Superseded — delete after merge.** Its single commit `9eafdcb` (2026-08-05, "gate the
highbd inter envelope on the LIVE C decoder — KB-40") is present on this branch as
`74e45c5` with the same subject and a byte-identical test file (blob `4ab178ca` on both).
All 7 files it touches exist at HEAD; the `ref_decode_av1_stream_frame_opt` shim it added
to `aom-sys-ref` is present; KB-40 is closed in `docs/KNOWN_BUGS.md:5872`. `git cherry`
reports it as not-applied only because the CLAUDE.md/CHANGELOG hunks differ. Note
`74e45c5` is NOT on `origin/main` today — it rides in with this branch, which is another
reason the branch must land before the remote branch is pruned. The one thing it left
behind is the duplicate top-level test file in (c)-7.

## (f) Stale-branch cleanup

Merged into `origin/main` already — safe to delete now:

- local: `fix/issue-17-frame-state-alloc`, `integrate/decode-cancel-latency`,
  `integrate/dedup-and-table-tests`, `integrate/encoder-policy-handoff`,
  `perf/tls-pool-windows-probe`
- remote: `origin/handoff/2026-09-08-encoder-policy`, `origin/integrate/decode-cancel-latency`,
  `origin/integrate/dedup-and-table-tests`, `origin/integrate/encoder-policy-handoff`,
  `origin/maint/dedup-and-table-tests`, `origin/perf/decode-cancel-latency`,
  `origin/perf/tls-pool-windows-probe`

Delete only after this branch is on `origin/main`:

- `origin/fix/issue-8-highbd-inter-envelope` (item (e))
- `perf/gate3-txfm-i16-batch` + `origin/perf/gate3-txfm-i16-batch` themselves

`origin/preserve/*` untouched per instruction.

---

## Follow-up actions taken 2026-09-24 (same session, on the branch, NOT pushed)

The user asked for four things after reading the review: version the C
instrumentation, deduplicate the test binaries, fix the env-var reads, and look at
`.devin/`, build/test time and CI. Each is a separate local commit with explicit
staging; nothing was pushed.

### 1. C instrumentation is now versioned, and the oracle cannot silently link it

- `docs/upstream-instrumentation/` holds both trace generations as patches against the
  pinned commit: `2026-09-12-kb55-58-traces.patch` (moved from `docs/`, 589 lines,
  4 gates) and `2026-09-24-kb59-68-traces.patch` (the live tree, 965 lines, 17 gates).
  They are not a superset chain (149 lines of the older set were dropped later), so
  they stay separate; both verified to apply cleanly to the pristine tree. `README.md`
  there is the gate inventory: every C `getenv` paired with its port site and KB.
- `just upstream-instrument [PATCH]` / `just upstream-pristine` / `just upstream-check`.
  `upstream-check` is now the FIRST step of `just gate-landing`, so a dirty oracle fails
  the landing gate instead of hiding behind `.gitmodules`' `ignore = dirty`.
- **Found while reverting:** `aom-sys-ref/build.rs` keyed the libaom build cache on the
  submodule SHA + FP flags only, so `git -C upstream checkout -- .` left a `libaom.a`
  compiled from the instrumented sources in `upstream/build/` and every "pristine" link
  reused it. The stamp now also carries a digest of `git -C upstream diff HEAD`
  (`worktree_digest`, `build.rs`); the first build after the change rebuilt
  incrementally in 24 s and the stamp reads `... clean`.
- The submodule working tree is pristine now (`upstream-check: pristine at 03087864cf`).

### 2. Env-gated traces: one helper, one read per process, `NAME=0` is off

- `crates/aom-dsp/src/envflag.rs` + `aom_dsp::env_flag!("AOM_X")` /
  `aom_dsp::env_target!("AOM_X")` (the `<r>,<c>` node selectors). Both expand to a
  call-site `OnceLock`. Semantics unified with `AOM_FORCE_SCALAR`: empty or `0` = off.
- 44 edits across 14 files replaced every hand-rolled `static OnceLock<bool>` and every
  uncached `env::var_os(..).is_some()` — including the decoder's per-partition
  (`aom-decode/src/lib.rs` `[dp]`) and per-DV (`[dv]`) reads and the per-SB `[dq-dec]`
  read, which were the getenv-on-a-hot-path shape `7099a04` had already fixed on the
  encoder side. `AOM_HDR_DUMP` (a path, not a flag) and the dispatch pin are unchanged.
  Two unit tests cover the parsing. Public-API snapshots regenerated (`aom-dsp` +3
  items: the module and two macros; `aom-encode` header line).

### 3. Duplicate test binary removed

`crates/aom-bench/tests/highbd_inter_decode_envelope.rs` deleted; the byte-identical
`tests/all/` copy is the one `main.rs:29` includes.

### 4. `.devin/` — evaluated: keep, harmless, but it was smuggled

`hooks.v1.json` registers one PreToolUse hook on Devin's `exec` tool;
`hooks/exec-timeout.py` (86 lines) wraps each command in `timeout --kill-after=10s`
(900 s foreground, 3600 s for `timeout: 0` background), passing through already-wrapped
commands, lone shell-state builtins and tty sessions. It is a real fuse against the
hung-process class the playbook's session-hygiene section warns about, it never runs
for humans or CI, and it has no effect on repo content. Verdict: **keep it tracked**
(it is repo-level agent policy, like `CLAUDE.md`), but note it landed inside
`56192f9`, an unrelated encoder commit, with no mention anywhere — that is the
process defect, not the file. The `eval` builtin is passed through unwrapped, so the
fuse is advisory, not a sandbox; fine for its purpose.

Devin's commit hygiene on this branch, measured: 156/160 commits carry its trailer;
85/160 name a gate or a pass count in the body; the last four name only `--lib` counts
and `self_contained_*` targets, never `gate-landing`. Two consequences showed up in
this session: the stale API snapshot, and item 5 below.

### 5. The workspace did not build on x86-64 — caught by the first real `gate-landing`

`just gate-landing` at HEAD failed in 20 s: `crates/aom-dsp/examples/qrepro.rs`
imported `aom_sys_ref::ref_quantize_fp_neon`, which is `#[cfg(target_arch =
"aarch64")]`. Two siblings from the same commit (`89e90f6`) had the same defect:
`qb_neon_repro.rs` (`ref_quantize_b_neon`) and `fwd_neon_bench.rs` (nine
`av1_fwd_txfm2d_*_neon` extern symbols that only an ARM libaom exports — a LINK
failure, masked because cargo stopped at the compile error first). `cargo test
--workspace` and nextest build examples; `cargo test -p zenav1-aom-encode` does not,
which is exactly why `gate-encode` was green for a week while the workspace gate would
have been red on every x86-64 box and both linux CI legs. All three are now arch-gated
(`mod bench` under `cfg(target_arch = "aarch64")`, a stub `main` elsewhere) and build
and run on x86-64.

### 6. CI and build/test time

- **Triggers:** `ci.yml` now runs on pushes to `perf/**`, `fix/**`, `feat/**`,
  `integrate/**`, `maint/**` as well as `main`/PRs. The no-cancel-in-progress policy is
  kept as the file's own comment argues for it. `scripts/ci_yaml_check.py` passes.
- **nextest on the three differential legs** (x86-64 default, x86-64 scalar-pinned,
  aarch64): `cargo test --profile test-fast --workspace` measured 21 m / 36 m on the
  hosted runners and drains ~300 test binaries one at a time; the same suite under
  nextest measured 57.4 min -> 340.9 s locally (CLAUDE.md, 2026-09-09) and is what
  `just gate-landing` already runs. `taiki-e/install-action@nextest` is added after the
  toolchain step on those legs only; the census, whereat, cancel-latency and
  portability jobs are unchanged.
- **Local:** the oracle relink after `upstream-pristine` is 24 s incremental (cmake
  reuses its build dir; only the stamp forces the check). `cargo clean -p
  zenav1-aom-sys-ref` in the recipes drops 2.7 GiB of dependants' artifacts, so the
  first gate after an instrument/pristine flip pays a full workspace relink; that is
  the cost of a correct oracle and is not worth optimizing around.
- **Not done, noted:** the repo is not rustfmt-clean (877 diffs in `aom-dsp` alone), so
  `cargo fmt -p` is unusable as a per-edit tool — it reformats whole crates. A one-time
  `cargo fmt --all` commit plus a `fmt --check` CI step would fix that; it is a
  1,000-file diff and a decision, so it was not made here.

### 7. Runtime trace API (the user's "#1")

`crates/aom-dsp/src/trace.rs` replaces the env-flag helper from item 2 (which never
landed as a commit): a `#[non_exhaustive] enum Trace` (21 kinds, one per print family,
each carrying its documented `env_name()`), `enum Focus` (the three `<r>,<c>` node
selectors), `TraceConfig::new().with(..).with_focus(..).with_hdr_dump(..).with_sink(..)`,
process-wide `install` / `clear`, and three site macros `trace_on!` / `trace_focus!` /
`trace_out!`. State is an `AtomicU32` mask + three `AtomicU64` focus slots (one relaxed
load per site, the same cost as the `OnceLock` deref it replaces) and an `RwLock`ed sink
touched only when a line is emitted. All 77 library `eprintln!` sites now go through
`trace_out!`, so a host can capture or silence them; the default sink is stderr, so
output is byte-identical to before.

The environment is a front door, not the library's behaviour: `TraceConfig::from_env()`
/ `install_from_env()` map the `AOM_*` names onto the API for hosts that want them, and
a new `trace-env` cargo feature (default OFF; `__internals` implies it in `aom-encode`
and `aom-decode`, `aom-bench` enables it directly) seeds that lazily on first query so
every in-tree harness, script and KB recipe keeps working unchanged. Proven both ways:
`cargo tree -e features --no-dev-dependencies -p zenav1-aom-encode` (the published,
zenavif-shaped build) contains no `trace-env`; the test build unifies it on through
`aom-decode/__internals`; an encode under `AOM_SCT_DBG=1` prints `[sct-port]` through
the feature and is silent with `AOM_SCT_DBG=0`. Four unit tests cover install/clear,
the env mapping (`0`/empty = off, node parsing), bit/name uniqueness and the focus
packing. `AOM_FORCE_SCALAR` stays a set-once pin outside this module, as discussed.
Public-API snapshots regenerated (`aom-dsp` +1 module, +4 types, +3 macros).

### 8. Estimate-contract regression, bisected — RESOLVED the same day as KB-69: retention compacted (222 -> 66.7 MB at 1024² s0, main 54.5) AND the model re-fitted (see `docs/KNOWN_BUGS.md` KB-69 for the per-cell table against main)

The two `encode_limits_and_estimate` failures from the first full gate run reproduce at
bare `973f503` with this session's edits stashed, so they are the branch's. `git bisect
run` over `origin/main..HEAD` on `the_estimate_is_an_upper_bound_and_stays_honest`
names **`e1a97fe` "encode: replay retained leaf payloads in pack instead of
re-encoding" (2026-09-15)** as the first bad commit. Measured at 256x256 `--cpu-used
6`: peak **11,255,676 B** against an estimate of **6,373,376 B** (the model is 1 MiB +
64 B per padded sample; the cell now costs ~133 B/sample), and the tall/wide per-pixel
ratio the second test pins fell to 1.95x (needs > 2x). This is a KB-50 "support
contract that never lies" break: a caller sizing a budget from `estimate()` will OOM.
The options are (a) bound the retained payloads (free them per SB row or cap the
retention), or (b) make the estimate speed-aware — raising the constant alone fails the
`ESTIMATE_MAX_SLACK` honesty ceiling at 1x1. Either is a decision about a perf lever
the user landed, so it is reported, not changed. It blocks the merge until resolved.

### 9. The scalar-pin leg was red on five `aom-dsp` differentials — fixed

`just test-next-scalar` (the `AOM_FORCE_SCALAR=1` half of the gate, and CI's
"forced-scalar pin" job) failed five tests that all pass unpinned, all from the
2026-09-14..17 "mirror the real C SIMD kernel" landings, all reproducing at bare
`973f503`: `block_error_matches_c_avx2_full_domain`,
`highbd_filter_intra_edge_at_byte_identical`, `lowbd_lpf_v3_matches_real_sse2_kernels`,
`quantize_fp_v3_bit_identical_to_real_avx2_full_domain` and
`quantize_fp_simd_bit_identical_to_scalar_at_every_tier`. One mechanism, two shapes:

- **Four tests probed `X64V3Token::summon()` before anything had applied the pin.**
  `scalar_forced()` is what disables the tokens, and it runs on the first dispatch, so
  the probe saw a live token, the dispatch then ran the scalar body, and the test
  compared scalar output against the AVX2/SSE2 C kernel it mirrors — which genuinely
  differs (`bl=255` saturation, `packs_epi32`, the SSE4 side-effect writes). Fix: read
  the pin first (`aom_dsp::dispatch::scalar_forced()`), and under it skip with an
  explicit message where the only oracle is a vector kernel (the scalar contract is
  asserted by the sibling `_diff` against the real `_c`), or drop only the SIMD-specific
  side-effect asserts (`edge_diff`, where the edge-region byte identity still holds).
- **The permutation sweep had the pin applied mid-permutation.** Same first-call
  mechanism inside `for_each_token_permutation`: token state flipped under
  `[all enabled]` and a scalar dispatch was compared against C-avx2. Fix: read the pin
  before enumerating, which is the order `cdef_find_dir_simd_diff` already documents.
  Token state stays the routing truth (the dispatchers `incant!` by token, the harness
  re-enables per permutation), so no assertion was weakened: `simd_perms >= 1` is
  unconditional and the sweep passes in both modes.

Verified per test in both modes (5/5 pinned, 5/5 unpinned) and by a full
`test-next-scalar` pass (below). This is the third defect the workspace gate found that
`gate-encode` cannot see, after the examples (item 5) and the API snapshot.

### Gate result (2026-09-24, pristine oracle, after every change above)

| step | result |
|---|---|
| `upstream-check` | pristine at `03087864cf` |
| `ci-yaml-check` | 3/3 workflows ok |
| `test-next` (default dispatch) | **1545 / 1547**, 60 skipped, 301 s — the 2 failures are item 8 |
| `test-next-scalar` (`AOM_FORCE_SCALAR=1`) | **1545 / 1547**, 60 skipped, 350 s — same 2 |
| `census-gate` | 4/4 |
| `test-whereat` | 4/4 |
| `api-doc-check` | 5/5 + 1/1 (snapshots current, crates publishable) |

Before this session the same gate stopped in 20 s at a compile error (item 5) and, once
that was fixed, showed 7 scalar-leg failures (items 8 + 9). What remains red is exactly
the bisected estimate regression, which is the user's call.

### Commits landed on the branch (local only, not pushed)

Listed in `git log`; each names its gate. Nothing was pushed, rebased or merged.
