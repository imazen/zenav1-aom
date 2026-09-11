# zenav1-aom [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenav1-aom/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenav1-aom/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenav1-aom?style=flat-square)](https://crates.io/crates/zenav1-aom) [![lib.rs](https://img.shields.io/crates/v/zenav1-aom?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenav1-aom) [![docs.rs](https://img.shields.io/docsrs/zenav1-aom?style=flat-square)](https://docs.rs/zenav1-aom) [![license](https://img.shields.io/crates/l/zenav1-aom?style=flat-square)](#license)

Pure-Rust, bit-exact reimplementation of [libaom](https://aomedia.googlesource.com/aom)
(the Alliance for Open Media AV1 reference codec), built module-by-module behind
differential harnesses. Every ported kernel is validated against a pinned C libaom
**v3.14.1** oracle (`03087864`), and landed decode/encode paths are held to
byte-exact bitstream gates — the port is measured against the real exported C
functions, not a transcription of them.

`#![forbid(unsafe_code)]` · runtime SIMD dispatch via [archmage](https://github.com/imazen/archmage) · `libaom v3.14.1`

## Crates

Four published crates. The facade is the one to depend on; the others are its
building blocks, publishable on their own for consumers who want a narrower slice.

| Crate | What it is |
|---|---|
| **`zenav1-aom`** | Thin facade. Re-exports the DSP kernels plus the feature-gated decoder and encoder as one dependency. Start here. |
| **`zenav1-aom-dsp`** | The consolidated DSP + entropy kernels: transform, quant, txb, cdef, restore, intra, loopfilter, dist, inter, convolve, recon, dispatch, and the MSAC range coder — each a module. |
| **`zenav1-aom-decode`** | The AV1 decoder: partition walk, per-leaf mode-info/coeff decode, intra predict, inverse transform, and the post-filter (deblock/CDEF/restoration) frame walk. |
| **`zenav1-aom-encode`** | The AV1 encoder: RD partition/mode/tx search, forward transform + quantize + entropy coding, and bitstream pack. |

Four more crates are dev-only (`publish = false`) and never ship: `zenav1-aom-sys-ref`
(the C-libaom FFI oracle the differential harnesses diff against), `zenav1-aom-bench`
(the Gate-3 port-vs-C performance harness plus the whole-frame stills-parity gates),
`zenav1-aom-dsp-bench` (port-only DSP kernel benchmarks, no C oracle), and
`zenav1-aom-target` (the dependency-injected zensim-quality target search — a
zero-dependency bisection over qindex whose encode/decode/judge cycle is entirely the
caller's closure; its `zq_census` harness is behind the non-default `census` feature,
which is what keeps the zensim judge out of a default workspace build).

### Install

```toml
[dependencies]
# decoder + encoder (default)
zenav1-aom = "0.0.1"
```

The encoder is feature-gated, so a size-sensitive or wasm consumer can build a
decode-only stack — the encoder crate is then never compiled:

```toml
[dependencies]
zenav1-aom = { version = "0.0.1", default-features = false, features = ["decode"] }
```

Depending on any published crate pulls in **no C toolchain and no `build.rs`** —
the C libaom oracle is a dev-dependency of the harnesses only, never a normal
dependency of the shipping crates.

## Feature support at a glance

Three tables: what the **shipping encoder API** does, what the **decoder** does,
and where **video** stands. "Bit-exact" always means *byte-identical to the
pinned C libaom v3.14.1 oracle on a named gate*, never "looks right".

Read the first column of the stills-encoder table carefully — it is the one
distinction this project has repeatedly had to re-learn. A feature can be
**ported and byte-gated** while not being reachable from
`aom_encode::key_frame::encode_key_frame`, the self-contained entry point
zenavif calls. Those rows are marked *harness-only*: the port has them, proven
against C, but the shipping API exposes no knob for them yet.

### Stills — encoder (ALLINTRA, KEY frame)

| Feature | In `encode_key_frame` | Bit-exact vs libaom | Gate |
|---|:---:|:---:|---|
| ALLINTRA KEY, `--cpu-used` 0–9 | ✅ default | ✅ | `self_contained_key_frame` (427/427) |
| 8 / 10 / 12-bit | ✅ | ✅ | same |
| 4:2:0 / 4:2:2 / 4:4:4 / monochrome | ✅ | ✅ | same |
| Coded-lossless (`cq 0`) | ✅ | ✅ | same (189 cq-0 cells) + 248/248 exact reconstruction |
| Superblock 64 **and** 128 | ✅ | ✅ | same |
| Single tile + multi-tile (mandatory and requested) | ✅ | ✅ | same |
| Loop-restoration search (Wiener / SGR) | ✅ default on | ✅ | `lr_restoration_gate` (8/8) |
| CDEF search | ✅ opt-in | ✅ speeds 0–3 · ⚠️ 4–9 | `encoder_gate_cdef_*` (14/14); speeds 4–9 diverge in the header's `cdef_strengths` only, pinned |
| **Palette** (screen content) | ✅ default on | ✅ | `screen_content_tools_byte_match_real_aomenc` (54/54, matched oracle) |
| **IntraBC** (screen content) | ✅ default on | ✅ | same · ⚠️ declined at coded-lossless (documented divergence, pixels unaffected) |
| Quantization matrices (`--enable-qm`) | ❌ harness-only | ✅ | `qm_encode_witness` (40 cells) |
| `tune=IQ` / `tune=SSIMULACRA2` bundle | ❌ harness-only | ✅ | `encoder_gate_tune_iq_e2e` (54/54) |
| Superres, fixed denominator | ❌ harness-only | ✅ | `encoder_gate_superres_*` (13/13 bd8 + 16/16 hbd) |
| `--deltaq-mode` 2 / 3 / 6, `--delta-lf-mode` | ❌ harness-only | ✅ | `deltaq_mode2_e2e`, `deltaq_mode3_e2e`, `delta_lf_mode_e2e` |
| Film-grain table inject | ❌ harness-only | ✅ | `film_grain_gate` |
| Partition / intra-tool / tx-control disable knobs (C8–C11) | ❌ harness-only | ✅ | `toggles_rd_close::toggles_c8..c11` |
| Screen-tools **trial encode** (`av1_determine_sc_tools_with_encoding`) | ❌ | ❌ unported | scoped in [`PARITY.md`](PARITY.md) C3 |
| Inter / video encode | ❌ | — | see the video table |

**One thing the 427/427 does not cover, stated because it took a year to
notice:** that gate's oracle is `shim_encode_av1_kf`, which hardcodes
`--enable-palette=0 --enable-intrabc=0`. It is therefore a parity claim against
a *palette-disabled* libaom, and was blind to both screen-content tools by
construction. They have their own gate with a matched oracle. The general
lesson — **check what the oracle was configured with before reading a parity
count as coverage** — is written up in
[`benchmarks/encoder_screen_tools_2026-09-10.md`](benchmarks/encoder_screen_tools_2026-09-10.md).

### Stills — decoder

| Feature | Supported | Bit-exact vs libaom | Gate |
|---|:---:|:---:|---|
| AV1 intra conformance corpus | ✅ | ✅ | `conformance_corpus`, 235 vectors, byte-identity + golden MD5 |
| 8 / 10-bit (corpus) · 12-bit | ✅ | ✅ | corpus · port-generated (`config_permutations_decode`) |
| 4:2:0 (corpus) · 4:2:2 / 4:4:4 / mono | ✅ | ✅ | corpus · port-generated |
| Superblock 128 · multi-tile · superres | ✅ | ✅ | `real_bitstream` family, `superres_diff` |
| Superres × multi-tile columns | ✅ | ✅ | `superres_tiles_diff`, 44 real streams |
| Quantization matrices · segmentation · lossless | ✅ | ✅ | `real_bitstream` family |
| Palette · IntraBC · `disable_cdf_update` | ✅ | ✅ | `real_bitstream`, `svt_interop_decode_gate` (streams from a *third* encoder) |
| Film-grain synthesis | ✅ | ✅ | `film_grain_diff` |
| Resource limits · cancellation · fallible alloc · fuzz | ✅ | n/a | `cancel_latency` (own gate), `fuzz_sweep`, 45k inputs / 0 panics |

**Scope caveat, measured:** the conformance corpus is a deep sweep of *one*
sequence shape — 233/235 are 4:2:0, all 8- or 10-bit, and **zero** carry
superres, multi-tile, QM, segmentation, `reduced_tx_set`, 4:2:2, 4:4:4 or
12-bit. Those axes are held by the port-generated gates above, not by
conformance. Do not read "the conformance corpus passes" as breadth across the
format.

### Video (inter frames)

This is an **ALLINTRA (still-picture) port**. Inter is a live track, not
finished work, and nothing here is a shipping claim.

| Feature | Encode | Decode |
|---|:---:|:---:|
| Zero-MV P frame, single superblock | ✅ byte-exact (`inter_e2e_search`) | ✅ |
| `[KEY, P]` across bd 8/10/12 × 4:2:0/4:2:2/4:4:4/mono | ❌ | ✅ 24/24 cells, 48/48 frames (`highbd_inter_decode_envelope`) |
| Animated multi-frame tracks | ❌ | ✅ 8/8 tracks, 40/40 shown frames |
| Nonzero-MV motion compensation | ⚠️ skeleton | ✅ bd8 · ❌ **refused by name** above bd8 |
| Switchable interpolation-filter rate model | ✅ ported | ✅ |
| Compound / OBMC / warped motion / global motion | ❌ | partial |
| GOP structure, rate control beyond fixed-Q, TPL, temporal filtering | ❌ | n/a |

Inter encode's two open items are pinned self-promoting gates:
`av1_simple_motion_search_term_none` (an unported linear early-termination
model) and GOOD-usage (`usage=0`) KEY-frame byte-exactness — every landed
byte gate in this repo is ALLINTRA.

## Status: early development

This is a work in progress, not yet a drop-in libaom replacement. **Scope: this is
an ALLINTRA (still-picture) port.** Inter-frame decode and encode are live tracks,
not finished work. What holds today, measured against the C oracle (state verified
2026-08-03):

- **Decoder — intra is conformance-clean.** Bit-identical to C across the AV1
  intra conformance scope (the CI-wired `xtask/conformance.py --scope intra`, 235
  vectors: byte-identity + golden per-plane MD5), including the aggressive q62/q63
  quantizer range, 128×128 superblocks (230 of 235 vectors) and film-grain
  synthesis (2 of 235). **Read that scope narrowly.** Every vector's frame 0 was
  parsed (`benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`) and the corpus
  is a deep sweep of *one* sequence shape: 233/235 4:2:0 plus 2 monochrome, 8-bit
  (169) or 10-bit (66), and **zero** vectors carrying superres, multi-tile,
  quantization matrices, segmentation, `reduced_tx_set`, `disable_cdf_update`,
  `delta_lf_present`, 4:2:2, 4:4:4 or 12-bit. (Until 2026-08-03 this bullet also
  listed superres and multi-tile as part of the conformance scope; the corpus
  contains neither.) Those axes are held by port-generated gates instead —
  `superres_diff`, `real_bitstream` (SB128 + multi-tile), `film_grain_diff`,
  `disable_cdf_update_diff` and `config_permutations_decode`. The 8-bit path runs
  a dedicated `u8`-plane pipeline (byte-identical to the reference, verified in
  both SIMD and `AOM_FORCE_SCALAR` dispatch) for speed. Inter-frame decode is
  progressing byte-exact through a single-reference feature ladder and several
  real frames.
- **Encoder — ALLINTRA byte-matches aomenc at every speed.** The all-intra
  (usage=2), KEY-frame path byte-matches real `aomenc` across `--cpu-used 0..9` on
  the synthetic grids, with **zero pinned cells** on the 10-speed × 26-axis
  configuration-permutation grid (both open-cell lists in
  `crates/aom-bench/tests/config_permutations.rs` have been empty since
  2026-08-02). On real conformance-decoded content it is 30/30 at speed 0 and
  **60/60 at speeds 1–4** — including partial-superblock (non-64-aligned) frames
  (the last two `--cpu-used 3` cq63 cells closed 2026-08-30 with KB-41's
  search-context CDF shadows).
  Non-default stills knobs are byte-exact too: QM, CDEF search, loop-restoration
  search, SB128, multi-tile, film grain, lossless (at every speed), 10/12-bit,
  `tune=IQ`/SSIMULACRA2, the deltaq modes and the toggle set ([`PARITY.md`](PARITY.md)
  section A). Still open, each pinned by a self-promoting gate: the two remaining
  real-content cells (both `--cpu-used 3` at cq63; speed 4 is 12/12), two palette
  128² near-ties, IntraBC's inter var-tx **coefficient** arm — the IntraBC-armed
  bitstream itself became conformant on 2026-08-01 — and the measured-but-
  unlocalized divergences tiered in CLAUDE.md's *Coverage queue*. That queue also
  lists the configuration axes nobody has measured yet, which is where every bug
  closed between 2026-07-30 and 08-03 came from. Inter-frame encode is an early
  skeleton.
- **Performance — the bar is ≤1.5× C, and it is not met.** *Decoder:*
  met at the 4K headline cells (≈1.22× at cq20, ≈1.19× at cq40); 2K and small
  frames still exceed it — 1.66–1.9× at 2K, up to ~2.4× on tiny,
  entropy-dominated cells ([`benchmarks/gate3_peak_wall_2026-07-25.md`](benchmarks/gate3_peak_wall_2026-07-25.md)).
  *Encoder:* **1.94× libaom at 1024×1024, cq27, `--cpu-used 3`** — the preset the
  zenavif integration actually ships — down from 10.66× when the encoder was first
  profiled on 2026-08-02, across ~50 byte-identical levers with Gate 2 holding zero
  pinned cells throughout. **Quote that ratio with its cell AND its speed**: the
  ratio is not flat across the speed axis, and a figure taken at `--cpu-used 6` on
  a different box is not comparable. It is a **CPU-time ratio as well as a wall
  one** (both arms measure 99 % CPU) and both arms are **single-threaded by
  construction**, so the comparison excludes libaom's threading, which is a real
  capability this port does not have.

  The remaining gap is **diffuse**: no class is over 26 % of it, and closing every
  named lever completely would recover ~27 % of what the bar needs
  ([`benchmarks/encoder_lever_map_s3_2026-09-10.md`](benchmarks/encoder_lever_map_s3_2026-09-10.md)).
  The route to 1.5× is halving every class gap — breadth, not a list of symbols.

  **Against the backend it would actually replace**, rather than against the bar:
  measured at the same zenavif config on photographic content, this port is
  **~10× faster than zenavif's current default AV1 backend at +3.01 SSIMULACRA2
  and 0.5 % fewer bytes**, and threading the incumbent recovers ~3 % of that at a
  6 % rate cost
  ([`benchmarks/encoder_vs_default_backend_2026-09-10.md`](benchmarks/encoder_vs_default_backend_2026-09-10.md)).
  Both readings are true and they answer different questions.

Every open item is held by a gate that *asserts the divergence is still present*,
so the moment a fix makes a pinned cell byte-match, its gate fails and the cell is
promoted — the suite can't silently drift, and "done" always means measured on the
real C oracle, never asserted by hand.

[`docs/archive/CONTEXT-HANDOFF.md`](docs/archive/CONTEXT-HANDOFF.md) is the current-state entry point;
[`STATUS.md`](STATUS.md) tracks what has landed module-by-module; [`PARITY.md`](PARITY.md)
is the stills-parity ledger; [`PORTING.md`](PORTING.md) maps each Rust module to the
`upstream/` libaom file(s) it ports and to the differential test that gates it;
[`CLAUDE.md`](CLAUDE.md) holds the Known Bugs ledger and the ranked coverage queue;
[`docs/DIFFERENTIAL_PLAYBOOK.md`](docs/DIFFERENTIAL_PLAYBOOK.md) is how work gets
validated here.

## Building and testing (fresh box)

The C libaom oracle lives in-repo as a pinned git submodule at `upstream/`, and the
test build drives it through cargo — there is no manual oracle-build step:

```sh
git clone --recurse-submodules https://github.com/imazen/zenav1-aom.git
cd zenav1-aom
cargo test          # builds the libaom oracle once, then runs the differential suite
```

The first `cargo test` (or `cargo build -p zenav1-aom-sys-ref`) auto-initializes the
`upstream/` submodule if it is empty and builds libaom once via cmake, in the
deterministic single-thread oracle config ([`reference/BUILD_CONFIG.md`](reference/BUILD_CONFIG.md)),
cached forever after on the submodule SHA. It needs **cmake, nasm, and a C compiler**
on `PATH` — if any is missing the build fails loud with the one-line install
(`sudo apt-get install cmake nasm build-essential`), never a cryptic linker error.

[`just`](justfile) wraps the common flows: `just test` (full differential suite),
`just test-scalar` (the `AOM_FORCE_SCALAR` pin that forces every SIMD kernel through
its scalar twin), `just test-fast` (same coverage, optimized), and `just bench-gate3`
(the Gate-3 port-vs-C paired benchmark).

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

Upstream C code from [libaom](https://aomedia.googlesource.com/aom) is
BSD-2-Clause with the Alliance for Open Media Patent License 1.0 — see
[`upstream-notices/LICENSE`](upstream-notices/LICENSE) and
[`upstream-notices/PATENTS`](upstream-notices/PATENTS) (the inherited upstream
files, also carried in the `upstream/` submodule); those terms continue to cover
the upstream work this port derives from. libaom is battle-tested, carefully
engineered code — this port stands entirely on that foundation.

### Path to MIT

If someone covers Imazen's 2026 AI + server costs, we'll release this port
under MIT — or under the original upstream license (BSD-2-Clause + AOM
Patent License 1.0). Contact support@imazen.io.
