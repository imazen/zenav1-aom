# Iteration playbook — the fast loops (2026-09-11)

This is the operational half of `CLAUDE.md`: how to land a perf lever, port a C feature,
close a divergence, compare against other encoders, and know CI actually ran — each as a
short loop with one command per step. The **rules** here were all paid for in the
2026-09-08..11 cycle (`CYCLE_LEDGER_2026-09-08_11.md`); the evidence for each is cited.

The shipping cell, used by every perf number unless stated: **1024x1024, cq27,
`--cpu-used 3`** (zenavif's default `speed: 4` maps to cpu-used 3), 8-bit 4:2:0, the
in-repo `av1-1-b8-01-size-196x196` photograph mirror-tiled. Both arms emit **40,237 B**;
a different byte count is an RD change, not a perf change.

---

## 0. Before any session

```
just ci-status          # did the last pushes RUN jobs? which leg is red?
git log --oneline -8    # the ledger is newest-first; STATUS.md has the narrative
```

CI ran **zero jobs for 41 hours** (2026-09-09 06:31Z to 09-10 23:25Z) because `ci.yml` was
unparsable, and every local gate was green throughout. A red run with no jobs and a red run
with a failing leg look identical in the commit list. `just ci-yaml-check` is now the first
step of `gate-landing`; `just ci-status` prints the job count per run.

## 1. Land a perf lever (clause 4)

Budget: ~25 min build+band+gate per lever on this box. Every landing this cycle followed
exactly this loop; the ones that skipped a step (§1.2, §1.5) are the ones in the rejected list.

1. **Pick from the lever map, not from intuition.**
   `benchmarks/encoder_lever_map_s3_2026-09-10.md` ranks every kernel family like-for-like
   against libaom at the shipping preset. Rules that decide whether a row is real:
   * **Like-for-like against C's symbol**, summed as a FAMILY at a low `--percent-limit`
     (C size-specialises into ~20 kernels; a top-N read inflates the ratio ~2x — it did for
     variance and loopfilter).
   * **A closure or inlining sink is not a lever** (`calculate_intermediate::{closure#0}`,
     `intra_model_rd_y`, `txfm_rd_in_plane_intra` all absorb their callees' cost).
   * **Ratio is not the objective; milliseconds are.** `optimize_txb_core` is 1.13x C and the
     port's largest symbol — checked and found diffuse (1495 insns, hottest 1.5 %), so leave
     it; but the reason to skip it is "diffuse", never "good ratio".
   * **Check the REGIME**: reach and cost figures carry their speed, frame size and platform.
     A speed-0 or 192x192 or Darwin figure is not a shipping-preset figure.
2. **Annotate before costing.** `just perf-profile port` (frame pointers on), then
   `perf annotate <symbol>` or, when annotate hangs on this 30 MB binary,
   `objdump -d --no-show-raw-insn target/release/examples/eprof_x86 | awk '/<symbol>:/,/^$/' | grep -cE 'ymm|cmp|call'`.
   What you are looking for, in order of payoff this cycle:
   * **bounds checks per ELEMENT in a 2-D loop** (`buf[base + c]` with three runtime slice
     lengths) — three `cmp` around one arithmetic op. Fix: one row slice per buffer per row,
     zipped, running offsets. Paid −3.17 % (`highbd_subtract_block`), −0.35 %, −0.27 %,
     −0.83 %.
   * **a scalar loop that is already bounds-check-free** — try `#[autoversion]` first
     (−0.88 % on `block_error`, one attribute). It only pays when per-call work amortises a
     dispatch: it measured NULL on `variance_raw` (tiny blocks).
   * **a copy into a scratch that a caller copies back** (predict-then-publish, per candidate
     mode) — −0.50 %. Only where the consumer is strided; where it needs a tight buffer the
     copy just moves.
   * **a per-lane `to_array()` store around correct SIMD** — `txb_init_levels`,
     `two_tap_run`, `quantize_fp` each ~0.2 pp.
   * **a hot allocation replaced by something CHEAPER than the allocation** — `SmallVec`
     inline storage (−0.46 %), a stack array. NOT a `with_capacity` hint (+0.77 %, removed zero
     allocations) and NOT a TLS pool on glibc (+0.23 %; −0.60 % on Windows ARM, merged on the
     under-a-percent policy because the drops compound).
   * **lane width**: the port is i32/u16 where libaom is i16/u8 in nine kernel families. Raw
     AVX2 intrinsics compile inside a `#[magetypes]`/`#[rite(v3)]` body under
     `forbid(unsafe_code)` (hadamard −0.47 %, wiener −0.40 %). Check `Cargo.lock`'s
     magetypes version and `docs/MAGETYPES_VOCABULARY.md` before declaring an op missing.
   **What does NOT pay, measured twice each: reshaping work** — inlining a call (+0.24 %),
   hoisting a lookup into a branch (+0.15 %), re-scoping a lookup into a loop (+0.20 %),
   doubling arithmetic to avoid a shuffle in a store-bound kernel (+0.34 %). "Removes work
   on every iteration of a hot loop" is necessary; it is not sufficient (z1/z3 bounds removal
   measured +0.12 % null).
3. **Cost it honestly.** Predictions this cycle ran 1.7x–4.8x optimistic against the kernel's
   own self cost and up to 13x against a class row. A lever under ~0.2 % of the encode is
   below this box's band resolution and **cannot be validated alone** — build the whole
   family or don't start (the i16 fused column pass: correct, reached, and null at 0.12 %).
4. **Build two arms and band them.**
   ```
   just perf-arms <BASE_SHA>      # base in a worktree, new = working tree; sha256 must differ
   just perf-band 24              # rotated base/new/baseB; paired median + sign test
   ```
   A lever counts only if it clears the same-binary null AND p < 0.05 with ≥ ~20/24 rounds.
   Two copies of one binary differ by ~0.25 pp systematically — quote the mean of two base
   copies when it matters, never the flattering one. Sub-0.5 % landings are invisible in the
   port/C RATIO (the C arm drifts ±0.3 %): quote the paired band, re-take the ratio only for
   the headline.
5. **Gate, record, push.** `just gate-landing` (both dispatch modes; ~11 min). Byte-length
   check on four cells is inside `perf-arms`; the differentials against the real exported C
   are what make "bit-exact" an assertion. Write `benchmarks/encoder_<lever>_<date>.md` with
   the band TSV, the mechanism, the bite proof (which tests fail when the change is
   perturbed — asymmetric: the differential fails, unrelated suites stay green), and what
   was deliberately NOT done. Append one paragraph to `docs/CLAUSE_STATUS_LOG.md` row (4).
   **Re-annotate after landing**: `subtract_block`'s fix lost its own inlining and a second
   annotate found another −0.23 %.
6. **Check whether the new fast path bypasses an older one.** Eight fused transform kernels
   each measured net-negative and byte-identical while routing around the i16 path an
   earlier landing had added. Nothing in the gates can see this; only a like-for-like symbol
   comparison did.

Windows: `gh workflow run winperf.yml -f base_sha=<sha> -f rounds=24` (`arms: prepost`)
sizes a lever on Windows ARM/x86-64 with no C oracle. Allocation levers are worth ~6.9x more
there than on glibc; arithmetic levers transfer within a point.

## 2. Port and wire a C encoder feature (clause 6)

1. **Find the C entry point and its dispatch**: `grep -n <fn> upstream/av1/encoder/*.c`,
   and `rtcd_defs.pl` for whether it is RTCD-dispatched. If it is, the port targets the
   variant libaom actually RUNS on x86-64 (`_avx2`/`_sse4_1`), not `_c` — three roots of
   KB-41 were `_c` transcriptions of dispatched kernels. `docs/LIBAOM_UPSTREAM_NOTES.md`
   lists the kernels whose tiers disagree with each other.
2. **Transcribe with provenance** (`file.c:line` on every function), following `PORTING.md`.
   Re-derive nothing from `SpeedFeatures::set_allintra` inside the encoder — it is
   framesize-blind, and every framesize/qindex-resolved field is lost (KB-26).
3. **Differential against the REAL exported C symbol first** (`aom-sys-ref` shim; add an
   entry to `shim/*.c` if the symbol is static). Sweep the inputs C can reach, assert the
   tiers agree, and write the **bite proof**: perturb the port, exactly the new
   differential fails. A flat/symmetric probe is transpose-blind (KB-12) — use asymmetric
   inputs.
4. **Wire it into `encode_key_frame`** (the only entry zenavif calls), not only into the
   `aom-bench` harness. Six ported features are still harness-only (README table); palette
   and IntraBC sat ported-and-gated for months while the shipping path ran neither.
5. **Byte gate with a MATCHED oracle.** Check what the oracle shim hardcodes
   (`shim/dec_shim.c`: `shim_encode_av1_kf` forces `--enable-palette=0 --enable-intrabc=0`).
   A green 427/427 against a palette-disabled oracle says nothing about palette. Assert
   non-vacuity per cell (the oracle's tools-ON stream must differ from its tools-OFF stream).
6. **Contract surfaces**: `configuration_support.rs` / `refusal_census.rs` must still agree
   with the encoder; a new `KeyFrameConfig` field is a **two-repo change** (zenavif builds it
   as a struct literal at `encoder_aom.rs:444`) — land knowing the pin bump will add it.
7. **`just gate-landing`**, then `just api-doc` if any `pub` item changed (snapshots are
   enforced), then the `[patch]` compose against zenavif when a `pub` signature changed.

## 3. Close a byte divergence (clause 5)

Never infer the mechanism from the delta's size or shape (a dropped transpose read as an RD
near-tie for four localisation passes). Signature that IS diagnostic: a **constant** byte
delta over a byte-identical tile payload is a header-length bug, not RD.

1. Decode both streams, first divergent block: the `decode_diff_*` / `kb*_localize` tests
   print it.
2. Dump C's per-block decisions with the instrumented sibling libaom
   (`docs/HANDOFF-TOGGLES.md`, the ar-swap; for decoder desyncs `/root/aom-inspect`).
3. Compare RD to the unit at that node; the first field that differs names the root.
4. Pin self-promoting (`assert_ne!` that fires when the cell closes), fix, bite-prove.
5. Under the ship cap, a divergence that is measured, attributed, bounded and written down
   may ship — record it in `docs/COVERAGE_QUEUE.md` T4 and stop.

## 4. Compare against other encoders

```
cargo run --release -p zenav1-aom-bench --example dump_cell_yuv -- 1024 1024 ~/tmp/photo_1024.yuv
just bench-cross-speed ~/tmp/photo_1024.yuv 1024 1024     # zenbench, interleaved: the SPEED claim
just bench-cross-rd                                        # 4 arms x quality x speed ladders, ~6 min at 512², charts
```

`drv-aom` times **`encode_key_frame`** (the shipping path) by default since `2984ad1`; the
old harness path (`XBENCH_AOM_BENCH_HARNESS=1`) is 1.9x slower and was what every earlier
xbench column measured. Score in RGB through one decoder (`xtool decode` + `score-rgb`),
because ravif codes 4:4:4/10-bit. The sweep's `ms` column is one encode per point on a
non-idle box — take speed from `encbench`, quality/rate from the sweep.

## 5. Gates, and how long they take

| command | what | wall |
|---|---|---|
| `just gate-landing` | ci-yaml + nextest both dispatch modes + census + whereat + api-doc | ~11 min |
| `just gate-encode` | aom-encode + aom-bench integration targets + census (fast pre-check) | ~6 min |
| `just api-doc-check` | public-API snapshots current (pinned `nightly-2026-09-09`) + crates.io scan | 7 s |
| `just census-gate` | tool-family reach census on the four harness contents | 6 s |
| `just gate-encode-perf` | the 3.24x..4.03x in-repo encode-time record | minutes |
| `just gate-cancel-latency` | decoder cancellation bars (own gate, needs an idle box) | ~1 min |

`-p <crate> --lib` is NOT a gate (KB-42). Compare `sha256sum` of arms before trusting a band;
`cargo build | grep '^error'` matches nothing because cargo colours its output.

## 6. Session hygiene (why the last one died)

* `CLAUDE.md` is loaded every turn. At 744 KB it forced a compaction every ~3 hours and
  ended the session with "Prompt is too long". Keep it under ~50 KB; bodies go in `docs/`.
* Background waiters: never `until ! pgrep -f "cargo test"` (matches itself); use a
  `Monitor`/file-existence wait. Run gates under `--profile test-fast`.
* Push each landing (`git merge-base --is-ancestor HEAD origin/main`) and then look at
  `just ci-status` once. Docs and memories updated in the same commit as the code.
