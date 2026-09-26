This tracks the AOM work transferred to another machine, plus its interfaces with zenavif, SVT and the measurement fleet. **The handoff is WIP, not a claim that the full encoder-policy goal or local gates are complete. CI remains deferred until local gates pass.**

## Start here / exact source

All three repositories have `handoff/2026-09-08-encoder-policy` branches:

- [AOM work + evidence](https://github.com/imazen/zenav1-aom/tree/handoff/2026-09-08-encoder-policy): shared `KeyFrameConfig::validate_configuration()`, called by the actual encoder, and `configuration_support` integration tests. The original branch started from `496d2496`; at the user’s request it has now been cleanly rebased onto fetched `origin/main` at `a7b1ab13`, preserving the newer ARM work. PRs #13 and #12 remain separate, not integrated by this rebase.
- [zenavif work](https://github.com/imazen/zenavif/tree/handoff/2026-09-08-encoder-policy): `src/backend_router.rs` queries three backends and intersects their answers with wrapper validation. **It does not implement automatic execution dispatch or measured optimality.** The older commit title overstates this. Dependency pins still need to be advanced to the actual support-query implementations; a local sibling build is insufficient proof that a fresh clone works.
- [SVT work and full goal](https://github.com/imazen/zenav1-svt/blob/handoff/2026-09-08-encoder-policy/ENCODER-POLICY-GOAL.md): signed native -1 and named mainline/hybrid references, opt-in intra-edge/restoration extensions, published SIMD dependencies, parity fixes. SVT ownership remains separate; coordinate interfaces instead of taking over that entire workstream.
- [Handoff evidence directory](https://github.com/imazen/zenav1-aom/tree/handoff/2026-09-08-encoder-policy/handoff/2026-09-08): timestamped copies of comparison reports and TSVs, source/artifact receipts. These are historical evidence, not live canonical measurement code. This issue/README supersedes older progress statements in copied reports.

Read this repo's `CLAUDE.md` working rules, `STATUS.md`, `PARITY.md`, `reference/BUILD_CONFIG.md`, and the named integration targets. Some historical queue entries are stale: reconcile with newer tests and landed fixes before redoing work. The only open GitHub issue found before creating this tracker was **#15**. Its original statement that standalone encoding is entirely absent is obsolete; its SCM trial residual remains useful.

## Open PRs: integrate existing work before duplicating it

The full open-PR audit found **#13 and #12**, both still open and already pushed:

- [#13](https://github.com/imazen/zenav1-aom/pull/13), `perf/decode-cancel-latency` at `91d979435954e8b678db22e98a4fc6286042472b`, targets main. It replaces auto-closed #11. It makes post-filter/film-grain work cancellable; reported worst 4096² cancellation latency drops 115.4 ms to 2.74 ms, with existing non-cancellable entry points preserved. Remaining evidence gaps: enabled loop restoration, active superres (only stage-boundary polling), high depth, other chroma formats, tiles and inter frames. Encoder cancellation is absent and is a useful separate API/runtime task.
- [#12](https://github.com/imazen/zenav1-aom/pull/12), `maint/dedup-and-table-tests` at `dacf1642dceca5098c0204864d4382d6d4b84c87`, targets #13's branch. It deduplicates inverse-QM storage, pins 56 geometry copies across 12 files, fixes a latent wrong `SUB_TX_SIZE_MAP` tail in `pack_tile_roundtrip`, exercises `whereat`, and removes verified dead symbols. Remaining items include migrating copies onto the canonical derivation and assessing decode-only build inclusion of unwired `restore::pick`. Do not delete intentionally retained symbols or split C-corresponding functions gratuitously.

- [ ] Review current diffs and rerun required local gates for #13 against current main, then #12 on the reconciled stack; integrate in that dependency order and update #12's base when appropriate. Preserve both branches. The PR descriptions contain historical measurements; this handoff has not independently rerun them.
- [ ] Reconcile the support-query handoff with this stack and latest ARM fixes. Pushing the handoff does **not** integrate these PRs. Test active-tool cancellation and error cleanup, including film grain and real non-NONE restoration, and record the encoder stop-token gap explicitly.

## Priorities and acceptance criteria

### P0 — establish a reproducible, honest baseline

- [ ] Integrate the handoff support-query patch with latest main, update zenavif pins, and test a fresh checkout. Query and production validation must share all configuration predicates, including dimensions/overflow, tiles, depth, format and controls; successful validation is not a promise that arbitrary source buffers are valid or allocations succeed.
- [ ] Run the actual mandatory integration gates: `just gate-encode`, `just test-fast` (or `just test-next`), and `just test-fast-scalar`. Provision pinned C oracle/conformance data as instructed by the repo. Use `~/work/claudehints/scripts/run-heavy`, serialize heavy jobs, and keep timing jobs isolated. Unit tests alone are not the gate.
- [ ] Run screen-content census for screen/IntraBC/palette changes (`kb41_screen_detected_defaults` with the 14 plane dirs), plus standalone AVIF parse/read-back and zenavif consumer gates. Record source/build/CPU, test names, counts, failures and pinned divergences. No new full AOM gate run is claimed by this handoff.
- [ ] Regenerate the support/parity inventory from executable public paths and current pins. Distinguish implemented, wired, byte-exact, independently decoded, unsupported and unmeasured. Do not weaken assertions, remove features, or relabel a divergence as a harmless near-tie without root-cause evidence.

### P1 — finish useful still parity and screen-content behavior

- [ ] Port `av1_determine_sc_tools_with_encoding` completely through the standalone entry point. The antialiasing-aware screenshot detector is already ported; the second encoding-based SCM decision is not. Follow `PARITY.md` C3: two fixed-32x32 trial encodes, qindex-dependent setup with `max(q_orig,244)` except lossless, packed projected size, stream-depth PSNR, palette-pixel/IntraBC statistics, exact decision thresholds, and restoration of trial state. Handle all production call sites and preserve C parity. Do not turn a missing trial into an unsupported blanket screen-content claim.
- [ ] Reproduce current tiny-image SCM witnesses: historical survivors include origins 8468 (59x128), 5052 (78x128), 8020 (115x128). The older 8468 plane directory is documented incomplete; regenerate from the canonical PNG instead of silently skipping. Reconcile historical counts (35, then 14) with actual current source before claiming closure of #15.
- [ ] Investigate **all five libaom/Rust AOM screenshot divergences** in the September 8 standalone baseline: canonical origin 8100, Lanczos3 512x320, SDR8 BT.709 limited 420, CPU-used 0, QP 5/12/20/32/48. First compare effective controls/SCM/headers and oracle build identity, then the first coding-order tree/symbol/reconstruction divergence. Successful decode or equal file size is not byte parity. Do not assume these are the SCM-trial bug without proof.
- [ ] Close fast-CDEF header-only divergence at speeds 4..9, fast VAR_BASED_PARTITION/nonrd payload divergences above roughly 3x3 superblocks and their mandatory-multitile witness; these are explicitly pinned by `self_contained_key_frame.rs`.
- [ ] Close current standalone high-depth pins: `HBD_OPEN` (bd10/12, speeds 1..6, including cq0), multi-tile/depth combinations, and the nine documented standalone pinned cases. See `self_contained_key_frame.rs` and its measured attribution. At cq0, retain exact source-sample reconstruction independently of C-bitstream matching.
- [ ] Audit remaining current toggle/format pins, prioritizing reachable defaults: KB-38 1080p low-q band, bd12 scalar/default dispatch disagreement, delta-q modes 2/3 at speeds >=8, C9 intra-DCT-only UV early-out, and the documented crop/tile/depth gaps. Reproduce before fixing: e.g. older CLAUDE text calls CNN root #27 open while newer PARITY evidence records its resolution. Do not redo resolved roots.
- [ ] Expand real-image and synthetic activation coverage across all CPU-used 0..9, cq extremes/near-lossless, odd/partial dimensions, SB64/128, tiles, mono/420/422/444, 8/10/12-bit, palette/IntraBC, transforms, CDEF/restoration, and supported tuning modes. Demonstrate each ported technique is exercised, including independent decoder/reconstruction checks.

### P2 — make AOM a usable alternative backend in zenavif

| Requirement | C SVT reference | zenav1-svt | Standalone zenav1-aom | Current zenavif AOM seam / remaining work |
|---|---|---|---|---|
| 8/10-bit 420 stills | Supported | Wired; four known native10 witnesses remain | Supported; named parity pins remain | Wired; broaden public consumer gates |
| 422/444 and 12-bit | Outside pinned C SVT 4.2 validator envelope | Route elsewhere when unsupported | Accepts mono/420/422/444 at 8/10/12 | RGB 420 at all three depths; 422/444 need pixel/API/mux wiring |
| Monochrome | Verify reference-specific semantics | Public wrapper available at 8/10 | Native mono at 8/10/12 | Gray8 wired; high-depth mono needs wiring and tests |
| Coded lossless | Audit exact reference settings | Existing wrapper path; prove requested sample domain | cq0 implemented: documented 248/248 sample reconstructions | Preserve sample losslessness; RGB-to-YUV loss is a separate contract |
| Alpha / premultiplication | Container/auxiliary-plane responsibility | Existing AVIF wrapper support | Raw codec is not an AVIF alpha contract | Implement auxiliary mono item + precision/range/premultiplication preservation; currently refused |
| Full range / RGB identity / HDR | Audit codec vs container separately | Preserve useful HDR-fork work; broader gates remain | Current standalone sequence signaling requires extension/audit | Full range, RGB identity, HDR metadata/precision must be wired end-to-end or refused |
| Screenshot handling | Has screen tools/detection | Existing auto/forced controls and IntraBC | AA-aware detection exists; encoding-based trial missing | Wire actual supported tools; verify source-conditioned decisions |
| Backend suitability / automatic dispatch | Not a C feature | Support validator exists, calibration incomplete | Shared validator WIP; measured suitability absent | Query-only WIP; actual selection and encode execution remain |

- [ ] Each backend owns side-effect-free support and measured suitability reports for the request's speed/quality balance. Include calibration revision, tested envelope, uncertainty and resolved settings. Unknown estimates remain unknown.
- [ ] zenavif intersects codec support with wrapper support, rejects hard requirements before encoding, then actually executes the chosen plan. Preserve format, precision, color/range, alpha, metadata and requested controls. Explicit backend selection retains its contract. Do not silently fall back after an encoder failure and hide a bug.
- [ ] Coordinate shared checked continuous effort, policy and versioned resolved-plan/replay/fingerprint interfaces with SVT. Strict `SvtParity` stays on the named C-equivalent SVT path; it must not silently route to AOM. Fractional effort must resolve to real bounded work, not interpolation of preset numbers. Keep ordinary controls simple; diagnostic enhancement sets remain available.
- [ ] Evaluate zenrav1e/zenravif for useful formats and quality regions too. Native backend support is not identical to the existing wrapper's capabilities. Preserve useful SVT extensions and route broad out-of-scope features to a backend that actually satisfies them.

### P3 — performance, RD, matched-time routing and representative corpus

The static `zenmetrics/benchmarks/av1-compare` binary already has C SVT, Rust SVT, libaom, standalone Rust AOM and rav1e arms. Do not time the old C-bootstrap/replay harness as a Rust standalone encoder.

Concrete measured anchors (AV1 payload bytes; SSIMULACRA2; median API encode time):

| Source / setting | libaom | zenav1-aom |
|---|---|---|
| 8100 screenshot, QP20, cpu0, local baseline | 15,765 B / 0.770 bpp / 85.179 / 3,811 ms | 17,170 B / 0.838 bpp / 84.267 / 3,132 ms |
| 8402 photo, QP27, cpu0, retained fleet result | 8,945 B / 0.5922 bpp / 74.2238 / 1094.933 ms | same bytes/quality / 2725.375 ms |
| 9334 photo, QP27, cpu0, retained fleet result | 9,757 B / 0.4484 bpp / 72.7595 / 1454.911 ms | same bytes/quality / 3857.454 ms |

The two fleet origins have different hardware cohorts; do not pool their timings. Equal QP is not matched quality or matched time. Decoder Gate-3 historical ratios are not standalone encoder performance results.

- [ ] Profile standalone AOM's approximately 2.49x/2.65x libaom photo time witnesses and wider representative sources. Preserve exact bytes while fixing missing SIMD/search/packing optimizations. Audit dependency migration to published archmage/magetypes 0.9.29 where applicable; SVT/zenavif/comparator migration does not prove AOM's own dependency policy is updated.
- [ ] Finish the full canonical **imazen-26 training scout before selecting a minimal representative set or generalizing optimization wins**. Pin `187fbf338ce08e8e6654db7f04ddae58d5263da2`: 1084 origins, 1082 available SDR PNGs (DNG 1444/1458 unavailable), 1078 unique inputs after four exact duplicate pairs. Keep all origin/split assignments and hashes. No held-out fitting.
- [ ] Current planned grid is max edge512 SDR8/420, QP5/16/27/38/49/60, SVT -1/0/4/8/13 plus isolated -1 intra-edge/restoration arms, libaom/AOM 0/4/8, three interleaved rounds: 84,084 cells / 252,252 planned encodes. It is NOT complete, and excludes C-SVT/rav1e population arms and HDR/alpha/high-depth formats. Expand sizes, formats, content families and relevant slow/fast presets after durable fleet execution works.
- [ ] Record bytes/bpp, independent decoded perceptual quality, timing distributions, conversion ceilings, settings, build/CPU/ISA/thread cohort and output hashes. Compare matched-time fronts and matched-quality rate/time, not preset numbers. Separate conversion/mux time from codec time and retain outputs for rescoring.
- [ ] Define RD/RD-speed zones and prespecify coverage/regret tolerances; select the smallest representative training subset with an optimality bound or explicit heuristic gap, then validate on the full held-out split. Tiny two-image ablations are activation/performance witnesses, not policy evidence.
- [ ] Identify libaom techniques useful immediately beyond SVT native -1, with source/control-flow port maps and one-technique ablations. SVT intra-edge and restoration-unit search are already opt-in; their small mixed ablations do not justify a default bundle. AOM owner supplies evidence/candidates; SVT owner implements its extensions.

## Fleet recovery and transferable evidence

**Population scout is stopped.** Only two population artifacts completed: 468 timed rows /156 cells /84 independent SVT reconstruction checks, plus a separate 234-row smoke. Serial worker mode buffered ledger rows for the entire population pass; a 7200-second pass timeout could lose those rows despite uploaded blobs. Zero population ledger objects were persisted. Do not report this as a completed broad sweep.

- [ ] Inspect dev's unpushed fleet work before inventing a replacement. `dev.dev.lan` failed DNS; configured SSH alias `dev` reached the host but authentication was denied. No remote checkout was inspected, so unpushed remote work remains unknown. Nomad/object-store access works; credentials stay in the canonical resolver, never issue text.
- [ ] Read canonical zenmetrics `docs/RUNNING_JOBS.md`, `docs/PLAN_SWEEPS.md`, `scripts/jobsys/README.md`, fleet orchestration status and worker code. Recover the two verified blobs into the canonical ledger through supported recovery, without fabricated outcomes or needless re-encodes. Fix/use durable bounded checkpointing, smoke-test inside the deployed image, then resume. Do not build a parallel scheduler.

Available objects:

- Manifest: `s3://zentrain/jobs/av1-training-scout-v2-20260908/manifest.json`
- Retained population: `s3://zentrain/jobs/av1-training-scout-v2-20260908/blobs/2c6eee64c352bc2e7e4095403dd57de97590ce868db1154d69dcc070e69277fb` and `s3://zentrain/jobs/av1-training-scout-v2-20260908/blobs/a75d4c011db40b86bded96d30139d5f4508f4f26f70fa41f6dd91df4169ce7d9`
- Smoke: `s3://zentrain/jobs/av1-training-scout-smoke-v2-20260908/blobs/dc50ccec7dd91df51ed7f67c4797033b3152b23e314e18191e307fc4c2694a70`
- **Transferable original two-origin baseline**, including PNGs, OBUs, rows, provenance and local source patches: `s3://zentrain/handoffs/av1-2026-09-08/baseline-c6d4fa9ca281e512c85391b5a796a90283c472b65c477b4ff5f25d3f74a151ed.tar.gz`. Download SHA-256 verified; see `baseline-artifact-receipt.json`.
- **Transferable selected zenmetrics source overlay**, including comparator, canonical fleet tooling and reports: `s3://zentrain/handoffs/av1-2026-09-08/zenmetrics-source-e77470bcd661d68b0d01dc7a621483135ccef7f1bc58ed68d20ff0d0357ea4e4.tar.gz`. Download SHA-256 verified. Overlay on zenmetrics base `3ab5f7910dfb0c6e0993daf93065c20e4325ee74`; receipt lists every file and source revision. This is not a full repo or a newly deployed worker. zenmetrics itself was outside the requested three-repo push.
- Deployed comparator SHA: `82ab0044e7243fe742edeb6b13bbf08d65e7b7f618df38d6e444b4c9c73256f1`. Later local `prepare-input` binary SHA `ae201a3706ceef8b243298c273762ffe4259cf4d625487448212897e9b9fe26d` was **not deployed**. Do not reuse cached job identities after source/build/protocol changes.

`imazen26_training_scout_preparation.json` and `IMAZEN26_SCOUT_STATUS.md` contain archive/image pins and verification details. The selected overlay's `prepare-input` command reproduces exact shared SDR conversion with a SHA/build receipt; do not substitute another image library's color conversion and call it a codec regression.

## SVT coordination / landing gates

SVT's two 8-bit bugs (partial-edge chroma reads and skipped partial-quadrant PD0 evaluation) fixed **53/53 historical mismatches**, with 168/168 exact replay pairs, 2627/2627 workspace tests, 139/139 spotchecks and 1100/1100 matrix cells. These gates precede the latest native10 source/debug edits. Expanded boundary matrix is **16/20**, with native10 failures p1q10, p4q10, p4q30, p5q10. The current source-precision edit does not fix the p1q10 witness. Do not claim the pushed tip passes all gates or merge it incidentally while updating AOM pins.

- [ ] Land focused, fully validated changes; rerun affected mandatory local integration/consumer gates on the reconciled source, then permit CI and merge ready work. Keep experimental branches/evidence available.
- [ ] Update this issue, #15, support tables and cross-repo pins with exact revisions and remaining limits. Completion requires working execution routing and representative evidence, not only APIs, documentation or query success.

### Push audit / one deferred historical workflow

The three current handoff branches and all other preserved local histories were pushed and remote hashes checked. The historical SVT branch `preserve/2026-09-08-86adc9495b17` alone was rejected by GitHub because the OAuth login lacks `workflow` scope. Its exact original workflow and commit patch were archived (download/hash verified) at `s3://zentrain/handoffs/av1-2026-09-08/svt-deferred-workflow-734e8d228e2ca3467e51f0cef034d16e905aba4bae12cdebf99af91b5487843f.tar.gz`. This preserves the work for transfer without activating CI; pushing that branch as-is still requires an appropriately scoped login. Main branches were not moved by the handoff.

Rebase verification: `cargo test --profile test-fast -p zenav1-aom-encode --test configuration_support` passed **2/2** through `run-heavy` after rebasing onto `a7b1ab13`. This is a targeted query integration check, not the full encoder/workspace/scalar landing gate.
