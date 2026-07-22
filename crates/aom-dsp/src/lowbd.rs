//! Lowbd (bd8, u8 pixel) decode pipeline — the dispatch contract for the
//! second, parallel 8-bit reconstruction path.
//!
//! # Why a second pipeline
//!
//! The existing "highbd" decode pipeline stores every reconstruction plane as
//! `Vec<u16>` and runs `i32` transform intermediates. That is correct for all
//! bit depths but wastes ~2x the pixel-plane memory bandwidth and half the SIMD
//! lane throughput at bit depth 8, where the reference libaom uses a dedicated
//! `u8`/`i16` "lowbd" path. This module is the contract for a parallel bd8 path
//! (`u8` recon planes, `i16`-narrowable coefficients) that lives ALONGSIDE the
//! highbd path — bd8 frames route to lowbd, bd10/bd12 stay highbd. Neither
//! path is deleted; the highbd path keeps handling 10/12-bit unchanged.
//!
//! # BitDepth split (the routing key)
//!
//! The routing key is `bit_depth == 8`. There is deliberately NO trait-object
//! or generic `BitDepth` abstraction threaded through the decoder: this
//! codebase uses concrete kernel functions everywhere, and a `dyn`/generic
//! layer would either defeat the SIMD `#[target_feature]` inlining (dyn) or
//! monomorphize the entire 5000-line tile driver twice (generic). Instead each
//! kernel FAMILY exposes a `*_u8` lowbd entry point next to its highbd one, and
//! the caller picks the entry by `bit_depth`:
//!
//! ```ignore
//! if bit_depth == 8 {
//!     aom_dsp::recon::reconstruct_txb_u8_into(dst_u8, .., scratch);   // lowbd
//! } else {
//!     aom_dsp::recon::reconstruct_txb_into(dst_u16, .., bd, scratch); // highbd
//! }
//! ```
//!
//! The recon-plane type is the mirror of this split: a bd8 tile holds `Vec<u8>`
//! planes, a bd10/12 tile holds `Vec<u16>`. A `ReconPlanes` enum
//! (`LowBd(Vec<u8>)` / `HighBd(Vec<u16>)`) is the intended carrier; the crop to
//! `FrameDecode` widens `u8 -> u16` at the boundary (bit-exact — `u8 as u16` on
//! a bd8 sample) so the public `FrameDecode` surface is unchanged.
//!
//! # Per-family dispatch pattern (what a fan-out agent adds)
//!
//! Each kernel family is ported to lowbd INDEPENDENTLY. The pattern, so two
//! agents on two families never touch the same lines:
//!
//! 1. In the family's module, add a `*_u8` entry with `u8` pixel slices in
//!    place of `u16`, `bd` fixed at 8 (drop the `bd` parameter). Keep the
//!    highbd entry byte-for-byte untouched — the lowbd entry is a NEW function,
//!    so the highbd conformance path cannot regress.
//! 2. If the family has a SIMD pass, the i32-domain inner math is pixel-type
//!    independent; only the destination load/store width changes. Reuse the
//!    shared i32 passes where they exist (e.g. the inverse-transform ROW pass
//!    is shared verbatim — only the COLUMN pass, which touches pixels, is
//!    duplicated for `u8`). See [`crate::transform::simd::try_inv_col_pass_u8`].
//! 3. Add a differential in `crates/aom-dsp/tests/<family>_lowbd_diff.rs` that
//!    asserts, at bd8, `u8_out[i] as u16 == c_ref[i]` AND
//!    `u8_out[i] as u16 == highbd_port[i]` over the family's full input space,
//!    with an `AOM_FORCE_SCALAR=1` variant. Model it on
//!    `inv_txfm2d_lowbd_diff.rs`.
//! 4. Route the family's call site in `aom-decode` behind `if bit_depth == 8`.
//!
//! Until a family is ported, a bd8 tile may DELEGATE that family to the highbd
//! kernel by widening its plane region `u8 -> u16`, running the highbd kernel,
//! and narrowing back — byte-identical (a bd8 sample round-trips exactly), at a
//! conversion cost that vanishes the moment the family is ported. The order to
//! port families is "hottest first" (the inverse transform is the largest single
//! consumer and is done); intra prediction / CfL / dequant are next.
//!
//! # Byte-identity: why lowbd cannot move a pixel at bd8
//!
//! The inverse-transform per-stage clamping is normative
//! (`av1_gen_inv_stage_range` / `clamp_value`). At bd == 8 that function assigns
//! `opt_range == 16` to BOTH the row and column stages, so every inter-stage
//! value is clamped to a signed 16-bit range and the final reconstruction is
//! `clamp(dest + residual, 0, (1<<8)-1 == 255)`. Therefore:
//!
//! * Storing the clamped pixel as `u8` instead of `u16` cannot change it — both
//!   clamp to 255. (The SAFE first step: only the destination width changes;
//!   the transform's `i32` intermediates are untouched. Proven exhaustively by
//!   `inv_txfm2d_lowbd_diff`.)
//! * A LATER narrowing of the inter-stage coefficient STORAGE to `i16` is also
//!   byte-identity-safe, because the normative `opt_range == 16` clamp already
//!   bounds every value that would be stored to a signed 16-bit range — the
//!   clamp does the narrowing the codec spec mandates, so `i16` storage loses no
//!   information the highbd `i32` storage kept. (The butterfly MULTIPLIES still
//!   accumulate in `i32`/`i64` and round-shift back to the `i16` domain, exactly
//!   as the reference lowbd SIMD kernels do.) This is the measured-win lever and
//!   is UNBLOCKED by the scaffold; it is intentionally NOT taken in the safe
//!   first step.
//!
//! This is the GO signal for the whole approach: bd8 lowbd is byte-identity-
//! achievable, and the `i16` narrowing that produces the SIMD-lane win does not
//! fight the normative clamping.
