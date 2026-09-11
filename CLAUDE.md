# aom-rs — project instructions & durable bug log

Pure-Rust, **bit-exact** reimplementation of libaom ≥ v3.14.1 as a drop-in replacement.
Validated behind differential harnesses against the REAL exported C functions (priority of
evidence: real exported C fn > synthetic-facade-over-real-fn > verbatim transcription —
transcribed oracles can carry shared bugs).

**Module-progress source of truth:** `STATUS.md` (updated per landing by the track agents).
**This file** holds the standing goal, the gates, the coordination rules and an INDEX of the durable
**Known Bugs** log. Bodies live in `docs/KNOWN_BUGS.md`; the clause-(4) perf record in
`docs/CLAUSE_STATUS_LOG.md`; the fast iteration loops and tool index in `docs/ITERATION_PLAYBOOK.md`.
**Keep this file under ~50 KB** — it is loaded into every turn of every session (at 744 KB it cost a
compaction every ~3 hours and killed a session with "Prompt is too long" on 2026-09-11).

## THE STANDING GOAL (set by the user 2026-09-08) — read this before picking work

> Make `zenav1-aom` a backend zenavif can select **by default** for still images, with a
> support contract that never lies, no panics or refusals on inputs a caller can produce,
> and encode time within libaom — **capping parity work at "measured, attributed, bounded
> and documented" rather than "closed"**. Do work to ensure we match the RD of C, doing
> bitwise parity if needed to get RD close enough, but **prioritizing a sensible
> conversion, wiring, and testing of all of the aom encoder from C.**

What it means in practice (the original three-consequence text, with every measurement that
closed an item, is preserved as an appendix of `docs/CLAUSE_STATUS_LOG.md`):

1. **Breadth beats depth.** Porting, wiring and testing another piece of the C encoder outranks
   closing the last byte of an existing pin. Byte parity is the MEANS to RD-matching; a
   divergence that is measured, attributed to a named mechanism, bounded in reach and written
   down may SHIP.
2. **Refusals are not divergences and must close.** A configuration a caller can reach must
   encode. As of 2026-09-10 **no refusal is open on the public API** (`refusal_census.rs` pins the
   documented refusals in both directions). The SCM trial encode is unported but measured to be a
   divergence, not a refusal; the harness `aom-bench` may still refuse cells it cannot model.
3. **The six zen contracts are landed on the encoder** (KB-49..52). Only clause (4), encode time,
   is open.

**Non-goals until after ship** (say so out loud rather than drifting into them):
inter-frame parity, closing `HBD_OPEN`, the full imazen-26 corpus sweep, beating SVT.

**Numeric caveat, stated because the number was elided in the directive:** "within libaom"
carries no multiple. Gate 3's standing bar is <= 1.5x C, so that is the reading until the
user says otherwise; the two retained fleet photo witnesses are at 2.49x / 2.65x today.

### Clause status, MEASURED (do not re-derive these; each names its record)

| clause | state |
|---|---|
| (1) a backend zenavif can select **by default** | Seam **works**, re-verified live at HEAD 2026-09-10 against this repo (not the pinned rev) via a temporary `[patch]` compose: `aom_encode_backend` 22/22, `aom_roundtrip_loss` 5/5, `resolved_routing` 6/6. "By default" needs two one-line zenavif flips (`default` feature set; `#[default]` on `Av1Backend`) and is a **decision**, not a measurement: against the BAR the port is 1.94x libaom (row 4); against the INCUMBENT zenavif default (`Zenravif`, a rav1e fork, speed 4 = our `--cpu-used 3`) it is **10.2x faster, 0.5 % fewer bytes, +3.0 SSIMULACRA2**, and threading rav1e buys 3 % (`benchmarks/encoder_vs_default_backend_2026-09-10.md`). zenavif builds `KeyFrameConfig` as a struct LITERAL (`encoder_aom.rs:444`), so every added field is a two-repo change — `enable_palette`/`enable_intrabc` (2026-09-10) must be added at the next pin bump. Re-run the `[patch]` compose whenever a `pub` signature in `aom-dsp`/`aom-encode` changes. Full history: `docs/CLAUSE_STATUS_LOG.md`. |
| (2) a support contract that never lies | `configuration_support.rs` + `refusal_census.rs` — the support query, the knob ranges and the documented refusals are asserted against the encoder's own behaviour. **RE-VERIFIED LIVE 2026-09-10 at HEAD: 14/14** across both files plus `encode_cancel`, including `the_documented_refusals_are_exactly_these` (pinned in BOTH directions), `every_format_and_depth_encodes_at_an_awkward_size`, `every_enumerated_knob_encodes_across_its_whole_range` (90 knob cells, 92 s) and `screen_shaped_tiny_cells_encode_rather_than_refuse` (60/60). |
| (3) no panics or refusals on reachable inputs | 90k fuzz inputs, 0 panics (`encode_fuzz_sweep.rs`); KB-51 was found this way. **RE-HUNTED AT HEAD 2026-09-10 — the committed default is only 600 inputs and the 90k figure predated ~60 commits: 45,000 inputs across seeds {1, 7, 101}, 13,315 of them reaching a REAL encode, 0 panics** (per seed ~4,440 encoded / ~5,770 unsupported / ~2,840 plane-size / ~1,950 sample-range, ~40 s each). All four non-vacuity classes fire, so the sweep is not testing `validate_configuration` and calling it panic-freedom. |
| (4) encode time within libaom | **NOT MET — 1.943x at 1024x1024 cq27 `--cpu-used 3`, the preset zenavif ships** (port 3011.65 ms / C 1548.93 ms, 8 rotated rounds, both arms 40,237 B; CPU == wall on both sides, both pinned single-threaded, so the comparison EXCLUDES libaom's threading). The 2026-09-08..10 cycle took it 3.24x -> 1.94x over ~45 byte-identical landings (transform fusion, lane-width kernels, bounds-check removal, allocation cuts) and ~15 measured rejections. Two things decide what to build next, both measured: closing EVERY named lever completely is only **27 % of the 716 ms the 1.5x bar needs** (`benchmarks/encoder_lever_map_s3_2026-09-10.md`), so the route is BREADTH — halving every class gap, roughly eight more transform-class-sized cycles; and tile threading is a **~3.2x at 4 threads / ~5.1x at 8** ABSOLUTE-time lever that does NOT move the ratio (`benchmarks/encoder_tile_threading_ceiling_2026-09-10.md`). The port makes ~4x libaom's allocator calls in ~0.75x its footprint, and allocation levers are worth ~6.9x more on Windows than on glibc. **Rules for choosing a lever, with their evidence, are in `docs/ITERATION_PLAYBOOK.md`; the COMPLETE measured record of this row (~120 KB: every landing, null, rejection and self-correction) is `docs/CLAUSE_STATUS_LOG.md` row (4). Read it before re-deriving anything, and append there.** |
| (5) match the RD of C | byte identity is the strongest available evidence and holds on 427/427 standalone cells; the pinned divergences are the measured/attributed/bounded residual the directive permits to ship |
| (6) sensible conversion + wiring + testing of all of the C encoder | **2026-09-11: every harness-only feature is now REACHABLE from `encode_key_frame`** — `quality` (tune IQ/SSIMULACRA2 bundle via `apply_tune`, QM, sharpness, chroma delta-q, delta-q modes 2/3/6, delta-lf, adaptive CDEF), `tools` (the 24 C8–C11 toggles), `film_grain`, `superres_denom` — each byte-gated against a MATCHED oracle (`ref_encode_av1_kf_cfg`) with a decode round-trip on every cell (`self_contained_tools.rs`, 236 cells, all conformant). Byte parity at landing: superres **18/18**, film grain **10/10**, coding tools **46/48**, quality knobs alone **48/71**, the tune bundle **0/84** — every open cell is a PAYLOAD divergence (first differing byte is the frame OBU size), pinned self-promotingly, to localize by decode-both (KB-53). `cdf_update_mode = 0` is REFUSED by name: its stream is rejected by the real decoder (KB-53). Palette + IntraBC wired 2026-09-10 (54/54, matched oracle). The SCM trial encode is unported and measured to be a divergence, not a refusal. Full history: `docs/CLAUSE_STATUS_LOG.md`. |

**The one-line reading:** everything except encode time is either met or reduced to encode
time. Rank perf work first until 3.24x moves.

## Public API surface — the `__internals` gate (2026-09-09)

**MEASURED: 4,149 `pub` items across the workspace, and the real external
consumer surface is TWO modules.** zenavif — the only external consumer, and it
depends on `zenav1-aom-decode` / `zenav1-aom-encode` DIRECTLY, not on the
`zenav1-aom` facade (which has no consumer at all) — uses exactly
`aom_decode::frame` and `aom_encode::key_frame`, plus the config/error types.
Everything else was `pub` because 320 separate integration-test crates had no
other way in.

**A correction worth keeping, because the raw count misleads:** `aom-dsp`'s 1,022
items are NOT the leak. `aom-encode` and `aom-decode` are separate crates that
legitimately consume them, so they must stay `pub` for in-workspace use. The
externally-visible surface is `aom-encode` (1,832) and `aom-decode` (106).

**The pattern, landed for `aom-decode` first:**
* the crate's real API stays `pub` (`frame` + the `config`/`error` re-exports);
* implementation modules are `pub` only under a default-OFF `__internals`
  feature, `pub(crate)` otherwise — verified: without it they are genuinely
  unreachable (`E0603 module is private`);
* in-workspace crates that reach inside declare
  `features = ["__internals"]`, so a `--workspace` build unifies it on and the
  harness runs (measured: 76 aom-decode tests visible under `--workspace`);
* the consolidated `tests/all` target carries `required-features` AND an
  explicit `path` — **cargo resolves `[[test]] name = "all"` to `tests/all.rs`,
  not `tests/all/main.rs`, so without the path the section silently does nothing
  and the harness builds anyway.** That cost a debug cycle here.

**The anti-KB-42 guard, which is what makes `required-features` safe.** A
`required-features` target is SILENTLY SKIPPED when the feature is off, so a
plain `cargo test -p zenav1-aom-decode` would report green having built no
integration tests — precisely the failure KB-42 documents. So
`tests/internals_feature_guard.rs` is deliberately NOT gated: it always compiles,
always runs, and fails with the exact command to re-run. Verified in both
directions (fails without the feature, passes with it).

**`aom-encode` IS NOW DONE TOO — 2026-09-10, and the number is the headline:
`docs/public-api/zenav1-aom-encode.txt` goes from 4,305 lines to 281.** All 77
`pub mod` lines are replaced by `pub mod key_frame;` (the one module zenavif
calls) plus an `impl_mods!` macro over the remaining 76, expanding to `pub` under
`__internals` and `pub(crate)` without it. `aom-bench` and `aom-decode` declare
the feature; `tests/all` carries `required-features` AND the explicit `path`
(the cargo trap above); `tests/internals_feature_guard.rs` is the non-gated
anti-KB-42 guard. The blast radius predicted above was real but mechanical —
every consumer is in-workspace, exactly as the note said.

**AND THE SNAPSHOTS ARE NOW ENFORCED, WHICH THEY WERE NOT — this is the part
worth keeping.** `docs/public-api/` and `just api-doc-check` existed from
2026-09-09 and were wired into **NEITHER CI NOR `gate-landing`**. An unenforced
snapshot is a document, not a gate, and it behaved like one: when the
enforcement landed **all four crates were stale**, including `aom-dsp`, which had
silently accreted this cycle's `predict_intra_high_in_place` family and
`aom_quantize_b_no_qmatrix`'s new `iscan` parameter. `api-doc-check` is now the
fifth step of `gate-landing` and its own CI job (`public-api`); it needs no C
oracle, no conformance corpus and no codec build, and **measures 7.7 s**, which
is why it can sit in a per-landing gate at all.

**The second half of that gate is crates.io publishability, and it found three
crates that could not have been published** (`apidoc/tests/crates_io_publishable.rs`,
5 tests): `zenav1-aom`, `zenav1-aom-encode` and `zenav1-aom-decode` carried
intra-workspace **path dependencies with no version requirement**, which cargo
rejects outright (*"all dependencies must have a version requirement specified
when publishing"* — confirmed with `cargo publish --dry-run`, not inferred), and
`zenav1-aom-encode` had no `repository` field. Fixed in the manifests.

**Why a metadata scan and NOT `cargo publish --dry-run` as the gate:** a first
publish is bottom-up, and cargo resolves a downstream crate's version
requirements against the real index — so `--dry-run` on `zenav1-aom-encode`
cannot succeed until `zenav1-aom-dsp` is actually on crates.io (*"no matching
package named `zenav1-aom-dsp` found"*). That failure is inherent, not a defect,
so it cannot be a gate. Everything cargo checks BEFORE it touches the index can
be, and that is where all three defects lived.

**THE SET OF PUBLISHED CRATES IS PINNED BY NAME, and the reason is the user's:
a crate cannot be deleted from crates.io.** The stakes are asymmetric — a broken
manifest costs a retry, whereas publishing a name nobody meant to own, or a
surface nobody reviewed, is unrecoverable. So `PUBLISHED` is an explicit list
and a crate joining it takes an edit, not merely dropping `publish = false`.
Bite-proved five ways, and the asymmetry is informative: four perturbations fail
exactly one test each, while "a `publish = false` crate becomes publishable"
fails FOUR at once — the loudest signal for the one mistake that cannot be
undone.

**Found while wiring the CI job, pre-existing and latent:** three `run:` lines in
`ci.yml` ended a plain scalar with `:` (e.g. `-- whereat_entries::`), which is
invalid YAML that GitHub's lenient parser happens to accept — `yaml.safe_load`
refuses the file at `HEAD`. Quoted, so the workflow can be validated locally
before it is pushed.

## Gates (definition of done)

- **Gate 1 — Decoder:** bit-identical to C across the AV1 conformance corpus (intra scope
  wired in CI: `xtask/conformance.py --fetch --scope intra`; gate = byte-identity + golden MD5).
  **Scope caveat, MEASURED 2026-07-30** (`benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`,
  every vector's frame 0 parsed): the intra corpus is a deep sweep of ONE sequence shape —
  **233/235 are 4:2:0 and the other 2 are monochrome** (the subsampling flags read
  4:2:0 on all 235, but 2 carry `mono=1` and so have no chroma planes at all),
  bd8 (169) or bd10 (66), 230/235 SB128, and ZERO carry superres,
  tiles>1, QM, segmentation, `reduced_tx_set`, `disable_cdf_update`, `delta_lf_present`,
  4:2:2, 4:4:4 or 12-bit. Those axes are covered by the port-generated gates instead
  (`real_bitstream`, `config_permutations_decode`), NOT by conformance. Do not read "the
  conformance corpus passes" as breadth across the format.
- **Gate 2 — Encoder:** bitstream bit-identical for every `--cpu-used 0..9`.
  **Scope caveat — SUPERSEDED IN PART 2026-09-02.** The 2026-07-30 caveat read: *"the port
  never AUTHORS a sequence header — `write_sequence_header_obu` has zero call sites in any
  `crates/*/src`; every encoder path parses a seq header out of a real aomenc bootstrap
  stream and emits only an `OBU_FRAME`, so bit depth, monochrome, subsampling, profile, SB
  size and every seq-level `enable_*` bit are REPLAYED, not derived."* That is now true only
  of `aom-bench`'s `port_encode*`. `aom_encode::key_frame::encode_key_frame`
  (`crates/aom-encode/src/key_frame.rs`) AUTHORS the sequence header and the frame header
  from config, and emits a complete temporal unit (TD + seq + `OBU_FRAME`) with no C bytes in
  the path — **427/427 cells byte-identical to real aomenc** (69/69 when this caveat was
  written; 186/186 on 2026-09-02; 241/241 after the SB128 + explicit-tile landings on
  2026-09-03; +186 coded-lossless cells landed 2026-09-04, KB-44), both decoders agreeing on the
  pixels (`aom-encode/tests/self_contained_key_frame.rs`). Its envelope is ALLINTRA,
  `--cpu-used` 0..=9, SB64 **and SB128**, single tile plus mandatory and explicitly-requested
  multi-tile, and all four (CDEF, loop-restoration) combinations (the caveat above was written
  when it was speed 0 / single tile / SB64 / post-filters off); everything outside that is
  REFUSED by name (`KeyFrameError`), and `port_encode*`'s replay caveat still applies to every
  gate that goes through the bench. So: "the seq bit equals the knob" gates through the bench
  remain agreement checks; the standalone gate is the evidence that the port can produce a
  configuration.
- **Gate 3 — Performance:** user-set acceptance bar ≤ 1.5× C (2026-07-20 directive).
  **ENCODER, MEASURED IN-REPO FOR THE FIRST TIME 2026-09-08 — the bar is NOT met.** The
  standalone `encode_key_frame` runs at **3.24x .. 4.03x libaom** on cells whose output is
  BYTE-IDENTICAL to it (real photographic content, ALLINTRA defaults, `--cpu-used` {0, 6},
  128x128 and 192x192, cq {27, 45}), i.e. ~2.2x to ~2.7x over the bar. Record + method:
  `benchmarks/encode_perf_vs_libaom_2026-09-08.md`; gate `just gate-encode-perf`. The
  2.49x / 2.65x fleet witnesses quoted in GitHub #16 are the OPTIMISTIC end of that range,
  not a typical value, and were taken on other hardware with another harness. Byte identity
  is also the RD evidence: same bytes means the same rate-distortion POINT, so "match the RD
  of C" is satisfied exactly where the byte gates hold and nowhere else.
  **Met at the 4K headline cells** (≈1.22× cq20 / ≈1.19× cq40 wall after the bd8 lowbd +
  i16-rows + CDEF find_dir landings); 2K and small-frame cells still exceed it
  (1.66–1.9× at 2K, up to ~2.4× on tiny/entropy-dominated cells) — see
  `benchmarks/gate3_peak_wall_2026-07-25.md` for the committed run, caveats, and the
  ranked remaining levers. The original ≤ 1.20× figure is retained as the stretch target.
- **Gate 4 — Coverage checklist** (+ a zenavif integration gate).

### MANDATORY per-landing gate list (added 2026-08-30 after KB-42)

**`-p <crate> --lib` IS NOT A GATE.** Every byte-identity gate in this repo lives in a
`tests/` INTEGRATION target, and the census gate lives behind a non-default feature —
so a landing that reports "311/311 aom-encode unit tests" has run none of them. That is
literally how KB-42 happened: four consecutive landings gated on `--lib` plus named diff
tests, and 23 byte/RD gates plus the coverage census stayed broken across six red CI runs.

Before pushing any encoder change, run:

```
just gate-landing       # = test-next + test-next-scalar + census-gate
                        #   + test-whereat + api-doc-check
```

**This REPLACED the old `gate-encode` + `test-fast` + `test-fast-scalar` trio on
2026-09-09, on measurement, and the coverage is identical** — the reduction is
exactly the kind of claim KB-42 says not to make from a name-level argument, so
both halves were measured:

* **`gate-encode` is a strict SUBSET of the workspace run.** On the 2026-09-08
  gate log it ran **189 distinct binaries against `test-fast`'s 319**, and the
  ONLY one not in the superset is `content_family_census` (a non-default
  `census` feature build, kept as its own step). The trio therefore ran 188 of
  189 binaries **twice — 779 s = 13 min of every gate run.**
* **Feature resolution was checked EMPIRICALLY, not argued.** `-p` and
  `--workspace` could in principle resolve a shared dependency's features
  differently (resolver v2 unions features across selected packages). Measured:
  after a `--workspace` build, `-p zenav1-aom-encode` and `-p zenav1-aom-bench`
  recompile **nothing**, and `--workspace` does not rebuild after them either —
  no oscillation, so the two selections share artifacts and cannot differ.
* **`cargo test` drains test BINARIES serially** (threading only within one),
  measured at load **3.94 on 24 cores, ~84 % idle**. nextest puts every test in
  one global pool: the whole workspace goes **57.4 min -> 340.9 s (~10x)**.

`just gate-encode` is KEPT as the fast pre-check while iterating on encoder work
— it is simply not worth running alongside the workspace gate. `just test-fast` /
`just test-fast-scalar` also remain, for a box with no `cargo-nextest`.

For screen-content / IntraBC / palette work also run the KB-41 plane-dir census.

The KB-41 plane-dir census is
`crates/aom-bench/tests/all/kb41_screen_detected_defaults.rs` with
`ZENAV1_PLANES_DIR` over the 14 plane dirs — 104/104 byte-identical.

A landing's own gate list in the commit message must name the integration targets it
ran, not just the unit-test count.

Primary configuration: ALLINTRA (usage=2), speed-0 KEY frame. **Single-frame (KEY-frame)
work must reach byte-exactness across BOTH tracks before inter-frame ("the rest") starts.**

## Decoder desyncs: use the instrumented libaom decoder FIRST (do not bisect by hand)

An arithmetic-decoder desync almost never shows up where it is caused. The cheapest way to
localize one is to compare against the REAL libaom decoder's own instrumentation, which dumps
both C's per-block mode info and C's exact per-symbol sequence. **A build already exists at
`/root/aom-inspect/examples/inspect`** (`CONFIG_INSPECTION=1 CONFIG_ACCOUNTING=1`; rebuild the
same way if it is ever lost):

```
/root/aom-inspect/examples/inspect --limit=<N> -bs -ts -m -r -mm <vector>.ivf   # per-MI block grid
/root/aom-inspect/examples/inspect --limit=<N> -a          <vector>.ivf         # per-symbol accounting
```

The block dump is per-MI (replicated over each block's footprint), so collapse it to block
top-lefts to get C's block list; diff that against the port's walk to find the FIRST structural
divergence. Then read the accounting around that block: it gives every `aom_read_symbol` tagged
by its reading function, so **"the port read the same symbol VALUES but off a different CDF row"**
is directly visible instead of being inferred. That failure mode (a probability-only drift that
desyncs several reads later, with everything looking correct locally) is the dominant one on this
codebase — it is the KB-6 signature on the encoder side and was BOTH inter-decode roots found on
2026-07-19.

Two gotchas, both verified:
- **The JSON emitter drops the first symbol at every new block position** (`put_accounting`,
  examples/inspect.c: on a context change it emits `[x,y]` *instead of* that symbol's triple). So
  `read_skip_txfm` — the first read of most blocks — never appears. Do not conclude a symbol is
  unread; infer it from what follows (e.g. no coeff/var-tx symbols at all ⇒ that block is skip).
- Entries are `[id, bits, samples]` against `symbolsMap`, **not** `[id, value, bits]`; `samples`
  is the number of `aom_read_symbol` calls aggregated at that (tag, position). Reconciling those
  counts against the port's expected read sequence is itself a strong check — e.g. inter-intra:
  40 allowed blocks ⇒ 40 flag reads, +2 wedge flags +1 wedge index = the 43 C records.

## Coverage queue

The ranked list of named-but-unmeasured axes (T1 refusals, T2 default-reachable, T3 harness-blocked, T4 pinned divergences) lives in **`docs/COVERAGE_QUEUE.md`**. When you close an axis, strike it there and in its KB; when a landing names a new one, add it there in the same commit.

## Known Bugs — index (bodies in `docs/KNOWN_BUGS.md`)

Record real bugs here immediately with file:line refs (survives context loss). Do NOT close
an entry by relaxing/excluding a test — only by a landed fix verified on `origin/main`. **Write the entry in `docs/KNOWN_BUGS.md` and add its one-line index entry here.**

- **KB-53** — Encoder: the RD/tune/tool knobs (tune IQ/SSIM2, QM, sharpness, chroma/delta-q, adaptive CDEF, C8–C11 toggles, film grain, superres) were HARNESS-ONLY — WIRED into `encode_key_frame` 2026-09-11 with a matched oracle; superres 18/18 + grain 10/10 byte-exact, tune bundle 0/84 pinned open (payload), `cdf_update_mode=0` refused (corrupt stream)
- **KB-52** — Encoder: `AllocMode` closes the last of the six zen contracts — and the gate caught its own instrument, 202…
- **KB-51** — Encoder: a bd8 encode handed 16-bit samples PANICKED with an arithmetic overflow — FIXED ✅ 2026-09-08, foun…
- **KB-50** — Encoder: resource limits + a side-effect-free peak-memory estimate — 2026-09-08 (contracts 2 of 6)
- **KB-49** — Encoder: there was no cancellation at all — `EncodeConfig` + a per-superblock-row stop token, 2026-09-08
- **KB-48** — Decoder: the remaining un-pollable windows, and why the cancellation bar had to become machine-relative — 2…
- **KB-47** — Encoder: `encode_key_frame` REFUSED a tile grid real aomenc accepts — the `rows*cols == 2^log2` invariant i…
- **KB-46** — Encoder: `--deltaq-mode` 2/3 at `--cpu-used` >= 8 PANICKED — FIXED ✅ 2026-09-08, and the queue's diagnosis…
- **KB-45** — Decoder: the per-mi DV grid was 12 ms of un-pollable frame setup at 4096² — FIXED ✅ 2026-09-08 (GitHub #17)
- **KB-44** — Encoder: `--cq-level 0` (coded-lossless) tripped `tx_size_to_depth`'s `depth <= MAX_TX_DEPTH` assert — FIXE…
- **KB-43** — CI red on the x86-64 legs since 2026-08-31: THREE roots, all fixed 2026-09-02 (root #1 VERIFIED green on ru…
- **KB-42** — CI red since `735a0a6d`: THREE independent roots, all localized and FIXED 2026-08-30 — **CLOSED, run `33325…
- **KB-1** — Decoder: recon divergence at base_qindex ≥ 249 (quantizer-62/-63) — REAL CORRUPTION, CI-quarantined
- **KB-2** — Encoder: `diag+vbars16 256x256 cq62` strong cell — FIXED ✅ (per-block intra edge filter type)
- **KB-3** — Encoder: `vgrad 256x256 cq32` cpu-used=1 cell — FIXED (missing speed-1 `use_square_partition_only_threshold…
- **KB-4** — Encoder: bd10/bd12 coded-eob divergence (was "RD-decision divergence at high bit depth") — FIXED ✅ (BOTH ro…
- **KB-5** — Encoder: lossless (cq0 / qindex 0) KEY encode — FIXED ✅ (mono + 4:2:0 both byte-exact, hard-asserted; #32 c…
- **KB-6** — Encoder: REAL-content RD divergence at bd8 4:2:0 (PRIMARY config) — FIXED ✅ (all roots landed; real-content…
- **KB-7** — Encoder: `--cpu-used=3/4` cq12/cq32 4:2:0 partition flips — FIXED ✅ (TWO speed-feature-port roots; speed-3…
- **KB-8** — Encoder: `--cpu-used=4` speed-4 deltas — PORTED ✅ (64/64 after the KB-7 roots; luma was byte-exact at 59/64)
- **KB-9** — Encoder: `--cpu-used=5` speed-5 deltas — PORTED ✅ (64/64 byte-identical, 0 residuals)
- **KB-10** — Encoder: `--cpu-used=6` speed-6 deltas — PORTED ✅ (64/64 canon; the noise-extension cq63 near-tie CLOSED 20…
- **KB-11** — Encoder: `--cpu-used=7` speed-7 VAR_BASED_PARTITION — PORTED ✅ (64/64 canon; the KB-10-twin noise near-tie…
- **KB-P29** — Encoder: palette 128² AB/4-way partition near-tie (2 cells) — PINNED (genuine; palette machinery C-faithful)
- **KB-12** — Encoder: `--cpu-used=8/9` nonrd PICKMODE — PORTED ✅, and the estimate-arm residual is CLOSED ✅ 2026-08-02 (…
- **KB-13** — Encoder: REAL-content byte-parity at speed >= 1 — ROOT FOUND (**58/60 byte-exact**; the 2 still open are BO…
- **KB-14** — Decoder: superres single-SB-column coded frame decoded flat — FIXED ✅ (header coded-lossless false-positive…
- **KB-15** — Encoder: IntraBC (screen content) — SEARCH + skip-arm + COEFF ARM all LANDED; the witness is PINNED on a PA…
- **KB-ARM-FLOAT** — aarch64: 15 float C-differentials in aom-encode fail — CLOSED ✅ 2026-07-30 (all four roots; `--workspace` g…
- **KB-20** — Encoder: bd10/bd12 x `--cpu-used>=8` PANICKED (unported hbd nonrd estimate arm) — FIXED ✅ 2026-07-30 (roots…
- **KB-21** — Encoder: cpu-4/5 is the fragile band — CLOSED ✅ (bd8). ROOT #1 FIXED 2026-07-30 (`early_term_after_none_spl…
- **KB-18** — Encoder: SB128 x `--max-partition-size=32` performs a `restore_context` that C skips — FIXED ✅ 2026-07-30
- **KB-19** — Encoder: `default_min_partition_size`'s >=2160p arm was UNMODELLED — FIXED ✅ 2026-07-30
- **KB-22** — Encoder: `av1_set_speed_features_qindex_dependent`'s speed-0 >=720p arm was UNMODELLED — FIXED ✅ 2026-07-31…
- **KB-23** — Encoder: the intra-CNN partition prune fired inside FRAME-EDGE superblocks (C's `cnn_output_valid` latch wa…
- **KB-24** — Encoder: the intra-CNN `quad_tree_idx` was anchored at the SUPERBLOCK, not at the 64×64 — `--sb-size=128` P…
- **KB-25** — Encoder: the speed-7 VAR_BASED walk PANICKED on a frame-edge single-strip rect — FIXED ✅ 2026-08-01
- **KB-26** — Encoder: LARGE FRAMES (`min(w,h) >= 480`) diverge at `--cpu-used >= 4` — FIXED ✅ 2026-08-01 (framesize-deri…
- **KB-27** — Encoder: MONOCHROME at `--cq-level 24` (`base_qindex` 96), speed 0 — a single-point near-tie — OPEN, pinned
- **KB-28** — Encoder: the framesize predicates read the mi-aligned extent, not `cm->width`/`cm->height` — an EXACTLY 128…
- **KB-29** — Encoder: the IntraBC-armed encode produced a NON-CONFORMANT bitstream (`Invalid intrabc dv`) — FIXED ✅ 2026…
- **KB-30** — Encoder: `cid22_6292444` at `--cpu-used=6` diverges at EVERY quantizer (1 of 10 real photographs) — CLOSED…
- **KB-31** — Encoder: every frame big enough to REQUIRE more than one tile PANICKED (`single-tile envelope only`) — FIXE…
- **KB-32** — Encoder: `--cpu-used` 8 (every size >= 512²) and `--cpu-used` 9 (>= ~1 MP) diverged on real content — BOTH…
- **KB-17** — Encoder: `use_screen_content_tools` was hardcoded `false`, so `--use-intra-default-tx-only=1` diverged on A…
- **KB-16** — INTER-ENCODE rung 1 ✅ (the port's OWN search codes the zero-MV P byte-exact, single-SB) + two pinned follow…
- **KB-33** — Decoder: conformant IntraBC streams from a NON-libaom encoder were rejected — FIXED ✅ (by KB-29 roots 4+5),…
- **KB-34** — Encoder: the fastest preset REFUSED ordinary images — the nonrd estimate arm could not code a NON-SQUARE le…
- **KB-35** — Encoder: the nonrd estimate arm's palette refusal fired on the FRAME FLAG, one of four terms of C's `try_pa…
- **KB-36** — Encoder: `default_min_partition_size`'s >=1080p arm was UNMODELLED — every >=1080p frame at `--cpu-used 6`…
- **KB-37** — Encoder: `av1_search_palette_mode_luma` is PORTED for the nonrd estimate arm — `--cpu-used 8/9` screen cont…
- **KB-38** — Encoder: `av1_set_speed_features_qindex_dependent`'s `is_1080p_or_larger && base_qindex <= 108` sub-block w…
- **KB-39** — Encoder: multi-tile x `--deltaq-mode` 2/3 was REFUSED with a loud assert — FIXED ✅ 2026-08-04 (the refusal'…
- **KB-41** — Encoder: the datagen arm's refused real-content cells are ALL screen-detected frames; harness mismatch (too…
- **KB-40** — Decoder: reported 12bpc inter divergence (GitHub #8) — NOT REPRODUCIBLE against the C oracle; the highbd in…
- **KB-PERF-6** — Encoder: loop-restoration search had NO SIMD anywhere — three tiers LANDED ✅ 2026-09-08 (2.733x -> 2.557x,…
- **KB-PERF-15** — Encoder: `txb_init_levels` computed eight lanes in parallel and stored them ONE BYTE AT A TIME — LANDED ✅ 2…
- **KB-PERF-14** — Encoder: the transform config built two function pointers per call that only the scalar fallback reads — LA…
- **KB-PERF-13** — Encoder: three heap allocations per `txfm_rd_in_plane_intra` call, one growing element by element — LANDED…
- **KB-PERF-12** — Encoder: the intra edge filter copied the whole edge to read a 5-wide window — LANDED ✅ 2026-09-09 (byte-id…
- **KB-PERF-11** — Encoder: filter-intra zeroed a 2178-byte scratch per call to fill as few as 25 cells — LANDED ✅ 2026-09-09…
- **KB-PERF-10** — Encoder: the Hadamard/SATD kernels were BOUNDS-CHECK bound, not arithmetic bound — LANDED ✅ 2026-09-09 (2.4…
- **KB-PERF-9** — Encoder: the forward transform COLUMN pass had no 4-wide arm, so every 4-wide forward transform ran the dri…
- **KB-PERF-8** — Encoder+Decoder: the SGR A/B pass was scalar because ONE of its ~20 operations is a table lookup — LANDED ✅…
- **KB-PERF-7** — Encoder: `compute_stats` touched the `H` accumulator once per PIXEL — the four-pixel fold KB-PERF-6 pre-reg…
- **KB-PERF-1** — Encoder: the intra-mode CNN is recomputed ~10x per superblock (C computes it ONCE and caches) — FIXED ✅ 202…
- **KB-PERF-2** — Encoder: per-txb allocation churn + the un-tiered forward-pass scratch — LANDED ✅ 2026-08-02, and the proje…
- **KB-PERF-3** — Encoder: the forward transform ran at HALF libaom's lane width — LANDED ✅ 2026-08-02, and the projection wa…
- **KB-PERF-4** — Encoder: the DIRECTIONAL intra predictors had no vector path at any bit depth — LANDED ✅ 2026-08-03, and th…
- **KB-PERF-5** — Encoder: SMOOTH ran at HALF libaom's lane width — LANDED ✅ 2026-08-03; PAETH's half built, measured NULL fo…

## Encoder single-frame primary envelope

Primary config = ALLINTRA (usage=2), speed-0 KEY frame, matching libaom's allintra DEFAULTS: **CDEF off, loop-restoration ON (speeds 0-4, off at >= 5), QM off, screen-detection ANTIALIASING_AWARE**. Palette + IntraBC follow the frame's screen decision. The verified derivation of each default, the historical envelope of the `encoder_gate_e2e_*` gates, and the confirmed non-divergences (do not re-chase) are in **`docs/ENCODER_PRIMARY_ENVELOPE.md`**.

## Iterating quickly — the loops and where they live

See **`docs/ITERATION_PLAYBOOK.md`** for the step-by-step loops. The one-command forms:

```
just gate-landing              # THE landing gate (nextest x2 dispatch modes + census + whereat + api-doc + ci-yaml), ~11 min
just gate-encode               # fast pre-check while iterating on encoder code
just perf-arms <BASE_SHA>      # build eprof_x86 for BASE_SHA and HEAD into ~/tmp/arms, sha256-checked
just perf-band [N]             # rotated interleaved band base/new/baseB on the shipping cell, paired stats
just perf-profile <port|c>     # perf record/report of one arm on the shipping cell
just ci-status                 # did CI actually RUN jobs on HEAD, and which failed
just bench-cross-rd / bench-cross-speed   # vs ravif, zenav1-svt, zenrav1e
```

## Coordination (parallel tracks)

- Max clean parallelism = **2** (one decoder agent + one encoder agent); cargo's shared
  target-dir lock serializes builds, which keeps the box safe.
- Strict crate ownership; commit with **explicit per-file staging** (`git add <paths>`, never
  `-A`/`-u`/`.`); shared `STATUS.md` via `git add -p`. Push `git push origin HEAD:main`; verify
  `git merge-base --is-ancestor HEAD origin/main`.
- Coordinator independently verifies every landing (on origin, boundary-clean, no `#[ignore]`
  / weakened asserts, gate is a real byte-identity assertion, CI green). Never trust a claim.

## Zen codec cross-cutting compliance

The six zen contracts (limits, estimate, located/categorised errors, panic-freedom + fallible alloc, stop token, fuzz target) are **landed on both the decoder and the encoder** (encoder: KB-49..52, 2026-09-08). The original spec, its priority order and the fuzz-campaign record are in **`docs/ZEN_COMPLIANCE_SPEC.md`**. Remaining asymmetry: `whereat`-located errors have no encoder twin.

