# Cross-encoder IntraBC interop: 1,992 real SVT-AV1 C streams — re-measured

**2026-08-02.** GitHub issue **#5**: "Decoder rejects conformant SVT-AV1 intrabc
streams". Filed 2026-07-23 after **37 of 100** real SVT-AV1 v4.2.0 C
screen-content encodes were rejected by `aom-decode` with

```
corrupt frame: intrabc DV failed validity (non-conformant stream)
```

while real libaom 3.14.1 decoded every one of them clean.

## Headline

**Re-measured on `main` (`5925983`): 0 of 1,992 rejected, 0 pixel differences.**
Every stream is accepted by the REAL `aom_codec_av1_dx` **and** by the port, and
the two decoders produce **byte-identical planes** on all 1,992.

The rate moved 37 % → 0 % because of **KB-29** (2026-08-01), which closed six
IntraBC roots, three of them decoder-side. The 37/100 figure predates all of
them. Both of the two decoder roots the fixtures reach were confirmed causal by
reverting each alone (teeth, below).

| | issue #5, 2026-07-23 | this run, 2026-08-02 |
|---|---|---|
| streams | 100 | **1,992** (2,042 decodes incl. 50 repeated configs) |
| rejected by the port | **37** | **0** |
| rejected by real libaom | 0 | **0** |
| pixel differences vs the C decoder | not measured | **0** |

Raw: [`svt_interop_2026-08-02.tsv`](svt_interop_2026-08-02.tsv) (per
corpus × preset × qp).

## The verdict issue #5 asked for: desync, not dv_ref derivation

The issue's stated hypothesis was that the decoder-side **`dv_ref` derivation**
(`find_dv_ref_mvs` / the INTRA_FRAME ref-mv stack) picks a different reference DV
than libaom on SVT-reachable neighbour configurations, so `dv = coded_diff +
dv_ref` lands outside the valid region. **That hypothesis is refuted.**

1. **Source review.** `find_dv_ref_mvs`, `find_ref_dv`, `is_mv_valid`,
   `is_dv_valid` and the `assign_dv` composition in
   `crates/aom-dsp/src/entropy/dv_ref.rs` were read line-for-line against
   `upstream/av1/common/mvref_common.h:267-338` (`av1_find_ref_dv`,
   `av1_is_dv_valid`) and `upstream/av1/decoder/decodemv.c:677-731`
   (`assign_dv`, `read_intrabc_info`). No divergence: the tile-bound checks, the
   sub-8x8 chroma special case, `total_sb64_per_row`, the `INTRABC_DELAY_SB64`
   ordering gate and the wavefront gradient all transcribe exactly, including
   C's truncate-toward-zero division.
2. **Direct experiment.** Reverting **KB-29 decoder root 4** alone — re-gating
   the 64x64 chunk walk (which contains the chroma read) on `do_uniform` —
   reproduces the issue's message verbatim on a real SVT stream, with the DV
   code untouched. Same for **root 5** (leaf-vs-raster var-tx walk selection).

Both roots are **missing/misordered coefficient reads**, i.e. a **tile-payload
desync**. Once the bitstream desyncs, a later block reads a garbage
`use_intrabc` plus a garbage DV diff, and DV validity is simply the first check
that hard-errors. This is exactly `docs/DIFFERENTIAL_PLAYBOOK.md` §10 — *a
decoder's error message names the first check that FAILED, not the defect* —
and the same trap KB-29 itself fell into.

**Defect class: port-only, decoder-side.** Nothing in libaom is wrong here; the
port's walk over an already-correct C algorithm was wrong. Not a port-fidelity
defect in the DV math, which is faithful.

## Teeth

The gate (`crates/aom-bench/tests/svt_interop_decode_gate.rs`) fails when either
covered KB-29 decoder root is reverted alone, and passes on `main`:

| reverted root | result |
|---|---|
| none (`main`) | **pass**, 4/4 fixtures |
| root 4 — chunk walk re-gated on `do_uniform` | **FAIL** |
| root 5 — walk selected on "leaf SIZES differ" instead of "the quadtree was READ" | **FAIL** |
| root 6 — leaf-arm CfL luma store disabled | pass (not covered — see below) |

Quoted failure, root 4 reverted (root 5's is identical):

```
svt420_codecwiki_512_ibc64x64.obu: the REAL C decoder ACCEPTED this SVT-AV1
stream but the PORT decoder REJECTED it: malformed bitstream: corrupt frame:
intrabc DV failed validity (non-conformant stream).
```

That is the issue-#5 symptom, character for character, produced from a real
SVT-AV1 encode by a one-line decoder revert.

**Root 6 is NOT covered** by these fixtures and is stated rather than hidden:
its condition is `!chroma_ref || uv_mode == UV_CFL_PRED`, which none of the four
fixtures' IntraBC blocks satisfy. They do reach the leaf arm — an instrumented
run counts **70 IntraBC blocks, 40 of them `do_uniform == false`** — so the arm
root 5 guards is genuinely exercised; only root 6's CfL sub-condition is not.

## What was measured

Encoder: the REAL C **SVT-AV1 v4.2.0-62-gdfbfe849c** (`~/work/zen/zenav1-svt-c`,
built out-of-tree, never modified), driven at the same still/AVIF CQP config as
the xbench `svt-c` arm — `rate_control_mode 0`, `aq_mode 0`, `avif 1`,
`level_of_parallelism 1`, 8-bit 4:2:0 — plus `screen_content_mode = 1`, which is
what arms IntraBC and palette. Driver: `scripts/svt_interop/svt_scc_encode.c`.

Sources: all 10 `codec-corpus/gb82-sc` screen-content images, tiled by
`xtool prep at:WxH+X+Y`.

| axis | coverage |
|---|---|
| sources | 10 gb82-sc images |
| crops | 50 square tiles (512², 448²) + 28 geometry cells (2560x1664, 1920x1080, 1280x720, 2048x256, 256x1536, 832x64, 128x480, non-multiple-of-SB extents …) |
| presets | 0, 1, 2, 3, 6, 9 |
| quantizers | 15, 20, 30, 35, 45, 48, 58 |
| tiles | 1x1 (1,936 streams) + **2x2 / 4x1 / 4x2 (56 streams)** |
| bit depth / format | 8-bit 4:2:0 |

Non-vacuity, measured with the REAL libaom `inspect` example
(`CONFIG_INSPECTION=1`, `-ibc -bs`) built from the pinned `upstream/` submodule:
**1,328 of the 2,042 decodes carry IntraBC blocks, 3,100,958 IntraBC mi units in
total.** IntraBC block shapes observed span BLOCK_4X4 / 4X8 / 8X4 / 4X16 / 16X4
(the 4-px-side class KB-29 root 1 was about) through BLOCK_64X64.

## Two facts about SVT that bound this result

- **SVT emits IntraBC only at presets ≤ 3.** Presets 6 and 9 produce **zero**
  IntraBC blocks at every cell measured, so those 500-odd streams are vacuous
  for this bug and are reported separately rather than counted as coverage.
- **SVT never emits an IntraBC block larger than BLOCK_64X64 here**, and no
  128-px block sizes appear at all. So the one remaining documented decoder-side
  IntraBC gap — KB-29 residual (a), where a **>64x64** multi-chunk non-uniform
  IntraBC block reads all luma leaves before any chroma while C interleaves
  L,U,V per 64x64 chunk — **is not reachable from this encoder** and stays open,
  untested by this corpus. It needs an encoder that emits IntraBC on
  BLOCK_128X128 / 128X64 / 64X128, i.e. an SB128 screen-content encode.

## Discrepancy with the issue's named repro, stated

The issue names its repro as "a 512² crop of `gb82-sc/windows95.png`".
**`windows95.png` is 640x480** in the corpus today, so no 512² crop of it
exists. The nearest reachable equivalents (480² centre crop, and 448² tiles at
five offsets, all at preset 3 / qp 48 / `scm 1`) were encoded and all decode
clean on both decoders. The original artefacts cited in the issue
(`/root/aom-rs-oracle-reject-repro-2026-07-23/`) are a Linux path that does not
exist on this machine, so the exact original bytes were not re-decoded — this is
a regeneration, not a replay. That is the one honest gap in the re-measurement.

## Reproducing

```bash
scripts/svt_interop/build.sh                      # SVT-AV1 C lib + the driver
TILESET=square PRESETS="0 1 2 3" QPS="15 30 45 58" scripts/svt_interop/gen_corpus.sh
cargo build --profile test-fast -p zenav1-aom-bench --example svt_interop_probe
./target/test-fast/examples/svt_interop_probe ~/tmp/i5p/manifest.tsv
```

`TILESET=geom` sweeps frame geometry instead of crops; `SVT_TILE_COLS` /
`SVT_TILE_ROWS` (log2) drive the multi-tile arm. The per-stream TSVs are not
committed (≈180 KB); they are deterministic from the committed scripts, driver
and probe, and the committed summary carries every aggregate they support.

Standing gate: `just gate-svt-interop`.
