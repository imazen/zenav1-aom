# Issue #8 — the 12-bit inter divergence is not in this decoder (2026-08-06)

**8/8 animated tracks, 40/40 shown frames byte-identical to the real C libaom
decoder — including the exact frame issue #8 names. Plus 24/24 `[KEY, P]` cells
across bd 8/10/12 × {4:2:0, 4:2:2, 4:4:4, mono} × cq {20, 60}, both frames each.
Nothing in this repo reproduces the reported divergence.**

Provenance, commands, box, dispatch modes and the honest gaps:
[`highbd_inter_decode_envelope_2026-08-06.meta`](highbd_inter_decode_envelope_2026-08-06.meta).
Data: [`highbd_inter_decode_envelope_2026-08-06.tsv`](highbd_inter_decode_envelope_2026-08-06.tsv).
Gate: `crates/aom-bench/tests/highbd_inter_decode_envelope.rs`.

---

## What was reported

[imazen/zenav1-aom#8](https://github.com/imazen/zenav1-aom/issues/8): decoding
`colors-animated-12bpc-keyframes-0-2-3.avif` through zenavif's aom backend
produces different RGBA than through rav1d-safe, first at **frame 1** — an
inter frame (the file's keyframes are 0, 2, 3). Over the first 32768 output
bytes: 328 differ, every differing index is `idx % 4 == 0`, and the byte deltas
are quantized: `{1: 60, 16: 76, 32: 132, 224: 60}`. Reproduced under
`AOM_FORCE_SCALAR=1`, so no SIMD kernel is implicated.

## What was measured here

Three arms, all against `aom_codec_av1_dx` (libaom v3.14.1, the pinned
submodule oracle) in-process — **not** against the md5 goldens already committed
next to the fixtures, and not against rav1d-safe.

| arm | cells | result |
|---|---:|---|
| animated tracks vs C, per shown frame | 8 tracks / 40 frames | **40/40 byte-exact** |
| `[KEY, P]` bd × chroma × cq, per frame | 24 cells / 48 frames | **48/48 byte-exact** |
| nonzero-MV highbd P | 12 cells | 4/4 bd8 byte-exact, 8/8 bd10+12 **refused**, **0 wrong-pixel** |

Identical under default dispatch and `AOM_FORCE_SCALAR=1`.

**The envelope, stated as the issue asks it:**

- *Is it 12-bit-only, or does 10-bit inter diverge too?* Neither diverges.
  10-bit zero-MV inter: 8/8 cells byte-exact. 12-bit zero-MV inter: 8/8 cells
  byte-exact, plus the vector's own two real 12-bit inter frames.
- *Is it inter-only, or do 12-bit INTRA frames diverge?* 12-bit intra is
  byte-exact: 8/8 KEY cells in the sweep, and frames 0, 2, 3 of the vector
  (all keyframes) plus all five frames of its 12-bit monochrome alpha track.
- *Which decode stage first diverges?* **None.** The comparison is at the final
  cropped, post-filter reconstruction — downstream of prediction, inverse
  transform, reconstruction and the whole deblock/CDEF/LR chain. It is
  sample-identical, so no stage bisect is possible: there is nothing to bisect.
- *How many cells diverged out of how many run?* **0 of 100** (40 animated
  frames + 48 sweep frames + 12 nonzero-MV cells).

## The bitstream is the same bitstream

The committed fixture was re-extracted from the live vector in the zenavif
checkout with `tools/avif-extract` and compares byte-identical on both tracks.
A stale fixture was the first thing that had to be ruled out, and it is ruled
out.

Coded properties, measured with `examples/inspect_headers` (not taken from the
2026-07-23 census): profile 2, 64×64, **bd 12, 4:2:2** (`ss_x=1, ss_y=0`),
SB128; frames `KEY, INTER(primary_ref=6), KEY, KEY, INTER` — the issue's
"keyframes are 0, 2, 3", confirmed. CICP as coded: `mc=2` (UNSPECIFIED), `cp=2`,
`tc=2`, `full_range=false`.

## Where that leaves the reported delta

This half is **inference from reading zenavif at `svtav1-rs-backend`
(read-only)**, not measurement — zenavif was not built or run for this record.
It is written down because it changes which side to look at next.

1. The vector carries an alpha track and codes 12 bits, so zenavif's output
   buffer is `Rgba<u16>` — **8 bytes per pixel**, not 4. Under that layout
   `idx % 4 == 0` selects byte offsets 0 and 4 of each pixel: the **low bytes of
   R and B**. The issue's "confined to the R channel" reading assumed RGBA8.
   The issue's own sample window corroborates the 8-byte layout: the frame is
   64×64 (measured above) and 64 × 64 × 8 = **32768** — the "first 32768 bytes"
   it sampled is the entire frame, exactly once.
2. `scale_pixels_to_u16` maps 12-bit codes to full `u16` (×65535/4095 ≈ 16.004),
   so adjacent 12-bit codes land ~16 apart. Byte deltas of 16 and 32 are then
   **±1 and ±2 codes at 12 bits**; 224 is consistent with a wrapping −32; the
   1s are the scale's own rounding. So the divergence is ~1-2 LSB at 12 bits on
   ~4% of the frame, in the two channels driven by chroma (R by V, B by U) with
   G — which depends on both, with smaller coefficients — rounding to no change.
3. Both backends call the **same** conversion kernel with the same arguments:
   `yuv_convert::yuv16_to_rgbx_strip::<Rgba<u16>>(sampling, y, u, v, …, range,
   matrix, bit_depth, out)` — the aom path via `aom_planar_to_buffer`, the
   rav1d path via `convert_16bit_planar_inhouse`. The only inputs that differ
   between the two are the **plane values, strides and dims**.
4. Frame 0 of the same track — 12-bit, 4:2:2, same conversion path — **agrees**
   between the two backends (the zenavif test reached frame 1 before asserting).
   A conversion-kernel rounding difference would have to be value-conditional to
   survive that.

Points 3 and 4 together say the RGBA delta almost certainly comes from
**different decoded planes**, and this record shows this port's planes for those
exact frames equal libaom's exactly. The remaining suspect is rav1d-safe's
12-bit inter reconstruction. That is a hypothesis, not a result: rav1d-safe was
not run here.

**The one-command check for whoever owns that side:** rav1d-safe's decoded
planes for this track, hashed in the golden layout (Y then U then V, cropped
dims, little-endian 16-bit), must equal
`crates/aom-decode/tests/data/animated/colors-animated-12bpc-keyframes-0-2-3.color.md5`
line 2 (`9f2291bf3f75dd440bc1d64ae26e0ac8`, frame 1). This run re-proved that
file equal to the live C decoder, so it is now an oracle-backed reference and
not just a stored golden. If rav1d-safe's hash differs there, the defect is on
that side and the RGBA comparison never needed to be involved.

## What this does not cover

- **Nonzero-MV inter above 8 bits is refused, not verified.** The port
  fail-loud-guards it (`crates/aom-decode/src/lib.rs`, "sub/nonzero-pel MC above
  bd8 not yet supported") because the sub-pel filter chain still runs on a u8
  scratch. All 8 bd10/bd12 nonzero-MV cells hit that guard, which is why the
  arm's value is "0 wrong-pixel cells", not "byte-exact". Widening it needs the
  highbd convolve kernels.
- The 12-bit and 4:2:2 KEY coverage above is **port-generated**, because the AV1
  intra conformance corpus contains no 12-bit and no 4:2:2 vector at all
  (README). It is oracle-checked, but it is not conformance-corpus breadth.
- Only cpu-used 0 was swept on the `[KEY, P]` grid (the 8-bit `cpu 0..6`
  envelope map lives in `inter_harness_chunk0`). A bd × speed grid is unmeasured.
