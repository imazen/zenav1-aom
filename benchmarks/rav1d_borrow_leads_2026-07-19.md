# Decode perf: concrete borrow leads from rav1d-safe — 2026-07-19

Read-only reconnaissance by the coordinator (rav1d-safe at `/root/work/rav1d-safe/src`).
Two high-confidence borrows, ranked by risk/reward. Both target the Gate-3 decode gap
(currently 1.45x wall / re-profile in `gate3_decode_profile_2026-07-19.md`: transform 4.90x,
intra-pred 3.19x, loopfilter 2.12x by ratio; entropy dominates coeff-heavy 2.8x content).

Note rav1d-safe is `#![forbid(unsafe_code)]` too, so borrows are directly idiom-compatible.
rav1d's `msac` and aom's `od_ec` are two implementations of the SAME normative AV1 symbol
decoder (dav1d/aom both decode the identical bitstream) — so implementation-level speedups
that preserve the decoded symbol stream are bit-exact by construction.

## LEAD 1 (do first — surgical, low risk, matches the "entropy" hypothesis)

### Widen the entropy window 32→64 bits + bulk refill

**Ours** (`crates/aom-dsp/src/entropy/dec.rs`):
- `OD_EC_WINDOW_SIZE = 32`, `dif: u32`, `rng: u16` (dec.rs:6, :21-22).
- `refill()` (dec.rs:53-70) is a **byte-at-a-time** loop:
  ```
  while s >= 0 && bptr < end { dif ^= (self.buf[bptr] as u32) << s; cnt += 8; s -= 8; bptr += 1; }
  ```
  → up to 3 iterations per refill, one bounds-checked byte load each.

**rav1d-safe** (`src/msac.rs`, `ctx_refill` ~:375): `EC_WIN_SIZE = 64` (`dif: u64`), and the
common path is a **single 8-byte big-endian load** (`u64::from_be_bytes(buf[..8]) ` then
invert/mask/shift), with the per-byte loop kept only as the <8-byte tail. One masked 64-bit
load replaces up to 6-8 byte iterations, and a 64-bit window refills ~half as often.

**Why bit-exact:** the window is just how many not-yet-consumed bits are buffered; the AV1
od_ec decode result (`read_symbol`/`read_bool` in `cdf.rs`) is identical for any window width
≥ the spec minimum. aom's own `od_ec.h` uses a `size_t`-width `OD_EC_WINDOW` — 64-bit on
64-bit builds — so this literally matches the C reference on the target we benchmark.

**Plan:** change `dif` to `u64`, `OD_EC_WINDOW_SIZE` to 64, port the bulk-load refill from
rav1d (keeping the byte tail for the last <8 bytes and the end-of-buffer `LOTS_OF_BITS`
handling), and adjust `tell_offs`/`cnt` init constants (dec.rs:41-44) for the new width.
`normalize()` (dec.rs:74-78) and `read_symbol` need the widened `dif` threaded through.

**Verify:** Gate-1 conformance byte-identical (this touches EVERY decoded symbol — a single
wrong bit fails the whole corpus, which is exactly the safety net you want), then bench the
coeff-heavy cell (`q00`) where entropy dominates.

## LEAD 2 (bigger, structural — the ~50% lever the profile named)

### bd8 lowbd pipeline: i16 coefficients + u8 pixels for 8-bit content

**Ours:** the inverse transform works in `i32` throughout
(`crates/aom-dsp/src/transform/inv_txfm2d.rs`: `type Txfm1d = fn(&[i32], &mut [i32], ...)`,
`highbd_clip_pixel_add(dest: u16, ...)`) and stores pixels as `u16` at every bit depth. The
re-profile attributes ~50% of the whole gap (+836M of +1676M Ir at 4K) to running this
highbd u16/i32 pipeline for 8-bit content where C uses lowbd kernels.

**rav1d-safe** (`src/itx.rs`): generic over a `BitDepth` trait (`src/include/common/bitdepth.rs`)
where `BD::Pixel = u8` and `BD::Coef = i16` for 8-bit (`BPC8`), `u16`/`i32` for HBD.
`inv_txfm_add<BD: BitDepth>` (itx.rs:68) and `inv_txfm_add_rust<W,H,TYPE,BD>` (itx.rs:168)
carry the bd-generic types through the whole butterfly.

**Caution — this is where bit-exactness gets subtle.** AV1's inverse transform has normative
intermediate clamping (`av1_gen_inv_stage_range` / `clamp_value`, which the port already does
in `transform/fdct.rs`). A narrower intermediate must still produce the spec's exact clamped
result. Do NOT assume rav1d's i16-coef path is bit-identical to aom's — dav1d and aom agree on
the FINAL pixels but may differ in internal representation. Prove byte-identity on the full
transform differential (`txfm2d_diff` / `inv_txfm2d_diff`) before trusting it, per the evidence
hierarchy. The safe framing: keep i32 intermediates (spec-normative) but store the DST pixel
buffer as u8 for 8-bit and skip the u16 widening on the add-back — that alone removes a lot of
the highbd overhead without touching the butterfly precision. Measure each sub-step.

Secondary (from the profile): every 4-tall/4-wide inverse transform falls to scalar
`av1_idct4`/`av1_iadst4` because the landed SIMD gates on `row_n % 8 == 0` (~274k+138k scalar
calls per 4K decode). A 4-row SIMD path or a differently-gated dispatch is a smaller,
independent win.

## Explicitly REFUTED (do not chase — from the re-profile)

"Arena allocation" — the port's alloc share (8.6%) is proportionally LOWER than C's (12.9%).
It cannot close the gate.
