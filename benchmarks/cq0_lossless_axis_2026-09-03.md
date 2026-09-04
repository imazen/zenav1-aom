# `--cq-level 0` (coded-lossless) parity axis — measured 2026-09-03

Provenance: base commit `c3e1b4a` + the zenavif#45 fix in this landing.
Host `mac` (aarch64-apple-darwin), rustc 1.98.0, `--profile test-fast`
(inherits `release`, keeps `debug-assertions = true`).
Oracle: `aom_sys_ref::ref_encode_av1_kf` / `ref_decode_av1_kf` — the REAL
exported libaom v3.14.1 encoder and decoder, same source planes on both sides.
Port side: `aom_encode::key_frame::encode_key_frame` (bootstrap-free; no C bytes
in the path).

All cells: ALLINTRA (`usage = 2`), CDEF off, loop-restoration off, single tile
unless the width mandates a split, chroma planes flat mid-grey (the
`self_contained_key_frame.rs` convention), luma from that file's five content
generators.

## Why this axis was measured

zenavif#45: `Av1Backend::Zenav1Aom` at zenavif quality 100 maps to
`--cq-level 0` and hit `debug_assert!(depth <= MAX_VARTX_DEPTH)` in
`aom_dsp::entropy::partition::tx_size_to_depth`. Before this landing the whole
gate had exactly ONE cq-0 cell (`A_cq0_64x64_420_bd8_tex`, `--cpu-used` 0).

## 1. What a real `--release` build does (correction to the issue)

The issue reports "a plain `assert!` … so it fires in release builds too". It is
a `debug_assert!`. Built at `--profile release` (debug-assertions OFF) the same
nine repro configurations do **not** panic and do **not** hang — they complete
and emit a stream:

| cell (64x64 4:2:0 cq0 `--cpu-used` 6) | release output |
|---|---|
| bd8 flat / gradient / texture | 24 / 426 / 1417 B |
| bd10 flat / gradient / texture | 25 / 1212 / 2904 B |
| bd12 flat / gradient / texture | 26 / 1905 / 4120 B |

Byte-for-byte the same lengths the fixed build produces, because the assert was
the only thing the walk did wrong here: `tx_size_to_depth(TX_4X4, BLOCK_64X64)`
returns 4, and `count_leaf` only compares that against 0.

A hang IS structurally possible in `tx_size_to_depth` — `sub_tx_size_map[TX_4X4]
== TX_4X4`, so an *off-chain* `tx_size` spins forever with `NDEBUG`, in C as
much as in the port — but coded-lossless is not that case: `TX_4X4` is on every
chain. Hardened anyway (see the landing's `tx_size_to_depth` doc comment).

## 2. Grid A — 720 cells, cq 0, vs 720 at cq 32

Axes: {mono, 4:2:0, 4:2:2, 4:4:4} x bd {8, 10, 12} x {flat, gradient, texture,
noise, checker} x `--cpu-used` {0, 3, 6, 9} x {64x64, 128x128, 100x60}.

| quantizer | byte-identical to real aomenc |
|---|---|
| cq 0 | **564 / 720** |
| cq 32 | **596 / 720** |

Divergent-set shape, IDENTICAL at both quantizers:

| slice | cq 0 divergent | cq 32 divergent |
|---|---|---|
| bd8 | **0 / 240** | 8 / 240 (see note below) |
| bd10 | 84 / 240 | 60 / 240 |
| bd12 | 72 / 240 | 56 / 240 |
| `--cpu-used` 0 | **0** | 0 |
| `--cpu-used` 3 | 72 | 56 |
| `--cpu-used` 6 | 84 | 40 |
| `--cpu-used` 9 | **0** | 28 |
| mono / 420 / 422 / 444 | 39 / 39 / 39 / 39 | — |

The four chroma formats diverge in equal counts with flat chroma, i.e. the
divergence is LUMA-borne.

**Note, a NEWLY MEASURED axis, not part of this landing.** The eight bd8 cq-32
divergences are all `--cpu-used` 9 at **100x60** — {tex, noise} x all four
formats — and 64x64 and 128x128 are byte-exact at the same speed and quantizer.
100x60 is not in `sweep_cells`' speed arm (which is 64x64 and 128x128), so this
is a cq-32 hole, unrelated to cq 0, in the same `--cpu-used` >= 7 nonrd family
`PIN_256x256_speed7` already records. Registered here so it is not lost; NOT
fixed or pinned by this landing.

## 3. Grid B — the full speed axis, 64x64 4:2:0

`--cpu-used` 0..9 x bd {8, 10, 12} x 5 contents, at cq 0 and cq 32 (300 cells):
**242 / 300** byte-identical. Divergent set:

| | `--cpu-used` 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| cq 0, bd8 | . | . | . | . | . | . | . | . | . | . |
| cq 0, bd10 | . | X | X | X | X | X | X | . | . | . |
| cq 0, bd12 | . | X | X | X | X | X | X | . | . | . |
| cq 32, bd10 | . | X | X | X | X | X | X | . | . | . |
| cq 32, bd12 | . | X | X | X | X | X | X | . | . | . |

That is `HBD_OPEN` verbatim (CLAUDE.md T4: *"bd10 AND bd12, `--cpu-used` 1..6,
LUMA-borne, reaches 4:4:4 + mono, qindex-dependent speed reach"*). It is
**not** a coded-lossless finding: bd8 is clean at cq 0 at every speed, and every
depth is clean at cq 0 at speeds 0, 7, 8 and 9. A lossless/WHT root would be
neither bit-depth- nor speed-conditional. The *reach within* the band does move
with the quantizer (60 cq-0 cells diverge where their cq-32 twin does not, and
28 the other way) — which is the "qindex-dependent speed reach" already
recorded for `HBD_OPEN`.

## 4. Grid C — size ladder at cq 0

{1x1, 4x4, 8x8, 16x16, 32x32, 48x48, 64x64, 96x96, 100x60, 128x128, 130x70,
192x192, 256x256, 258x258} x bd {8, 10, 12} x 5 contents x `--cpu-used` {0, 9},
4:2:0: **420 / 420 byte-identical**, and **0 / 420 lossless violations** (the
real C decoder returns the encoder's own input on every plane).

## 5. What landed in the gate

`crates/aom-encode/tests/self_contained_key_frame.rs`:

* `self_contained_key_frame_byte_matches_real_aomenc`: **186 -> 372 cells**, all
  byte-identical. The new J arm is cq 0 over 4 formats x bd {8, 10, 12} x 5
  contents x `--cpu-used` {0, 9} (+ bd8 at {3, 6}), plus a 13-point size ladder
  1x1..258x258.
* `coded_lossless_reconstructs_the_source_exactly`: **248 cells**, real-C-decode
  AND port-decode of the port's own cq-0 stream both return the source exactly.
  Covers the `HBD_OPEN` band, which the byte gate cannot.
* `open_divergences_are_pinned`: two new self-promoting pins, `PIN_cq0_bd10_grad`
  (`--cpu-used` 6) and `PIN_cq0_bd12_tex` (`--cpu-used` 3), both measured
  `TilePayloadOnly` — every derived frame-header field equals C's.
* Whole file: 26.2 s -> 30.1 s.

## 6. Mutation proof (the gate is not vacuous)

| mutation | result |
|---|---|
| `count_leaf` reverted to `tx_size_to_depth(..) != 0` (the pre-fix code) | 4/7 tests FAIL; first red cell `LL_mono_bd8_flat_64x64_cq0_s0`, `assertion failed: depth <= MAX_VARTX_DEPTH` |
| `coded_lossless = base_qindex == 0` -> `false` | byte gate **185/372** — exactly the 187 cq-0 cells go red and nothing else; the lossless and decode gates fail too |
| both reverted | sha256 of both sources restored byte-identically; 372/372 and 248/248 green again |
