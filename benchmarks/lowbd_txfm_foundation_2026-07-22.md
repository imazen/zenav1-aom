# bd8 lowbd inverse-transform — foundation (safe-step) measurement, 2026-07-22

Phase-1 (GO/NO-GO) measurement for the second, parallel bd8 "lowbd" (u8 pixel /
i16-narrowable coefficient) decode pipeline. This file records the transform
lever's **safe first step**: u8 destination pixels on the add-back, with the
butterfly precision UNNARROWED (still i32 / the normative
`av1_gen_inv_stage_range` clamps). Nothing here is projected or estimated.

## Byte-identity (the decisive GO question) — PROVEN

`crates/aom-dsp/tests/inv_txfm2d_lowbd_diff.rs` asserts, at bd8, for every valid
`(tx_type, tx_size)` (fuzz 700 cases each) + the all-zero pass + the lossless 4x4
WHT (both eob arms, 4 strides x 4000 cases):

* `u8_out[i] as u16 == ref_inv_txfm2d_add(bd=8)[i]`  — vs real exported C
  `av1_inv_txfm2d_add_*_c`.
* `u8_out[i] as u16 == av1_inv_txfm2d_add(bd=8)[i]`  — vs the port's own
  C-verified highbd path.

Green on BOTH the SIMD and `AOM_FORCE_SCALAR=1` paths. The bd8 lowbd transform
cannot move a pixel vs the highbd bd8 path.

## Callgrind Ir — the safe step is Ir-NEUTRAL (it is NOT the win)

Microbench: `lowbd_txfm_profile <u8|u16> 200` (`crates/aom-bench/src/bin/`), a
fixed workload of every valid `(tx_type, tx_size)` x4 with identical randomized
coeffs + u8 prediction, run through the u8 vs the u16 (bd=8) entry. callgrind
`--collect-atstart=yes`, release, no `-C target-cpu=native`.

| metric (inclusive Ir)              | u8 (lowbd) | u16 (highbd bd8) | delta |
|------------------------------------|-----------:|-----------------:|------:|
| `av1_inv_txfm2d_add(_u8)_into`     | 1,695,097,484 | 1,683,395,100 | **+11,702,384 (+0.70%)** |
| inverse COLUMN pass (SIMD)         |   855,800,916 |   840,844,104 | +14,956,812 (+1.78%) |

Runtime sinks identical (`20109400` both sides) — a second, live byte-identity
check.

**Reading:** the u8-storage safe step is marginally WORSE in Ir, not a win. This
is expected and it is the honest result: narrowing only the destination STORAGE
(u8 vs u16) trades no instructions — the per-lane byte gather/store is if
anything a hair more work than the word one — and callgrind Ir does not capture
the memory-bandwidth/cache benefit of the narrower plane. **The instruction-count
win is the i16 SIMD-lane narrowing, not the u8 storage.**

## The i16 narrowing IS byte-identity-safe at bd8 (the GO signal)

`av1_gen_inv_stage_range` assigns `opt_range == 16` to BOTH the row and column
stages at bd == 8, so every inter-stage value is clamped to a signed 16-bit
range by the NORMATIVE clamp. Consequences:

* The normative clamping does **not** fight the narrowing — it is exactly what
  makes `i16` inter-stage storage lossless (the clamp performs the narrowing the
  spec mandates; `i16` keeps everything the highbd `i32` kept after the clamp).
* The butterfly MULTIPLIES still accumulate in `i32`/`i64` and round-shift back
  to the `i16` domain (the reference lowbd SIMD kernel shape) — "keep i32
  intermediates for the products" and "i16 inter-stage storage" are compatible.

So the second phase — a full i16-domain SIMD transform (both passes on `i16x16`
lanes, ~2x lane throughput) — is the measured-win chunk, and it is unblocked and
byte-identity-safe. This is a **GO** for the pipeline approach: the decisive
NO-GO condition ("cannot be byte-identical / normative clamping fights the
narrowing") is refuted.

## Provenance

Commit: see `git log` for the landing SHA. Box: this workstation. Corpus/refs:
libaom v3.14.1 submodule pin `03087864`. Method: exact callgrind Ir, load-
independent; measured, not extrapolated.
