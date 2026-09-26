//! Consolidated integration-test harness for `aom-decode`.
//!
//! Every `tests/*.rs` here is a MODULE of one binary rather than a binary of
//! its own. Cargo makes one test target per `tests/*.rs`, and each target
//! statically links the whole port plus libaom.a (~22 MB a piece, measured), so
//! N files cost N near-identical link steps. Folding them into one target
//! replaces that with one link, and lets `cargo test`'s intra-binary thread
//! pool run ALL of these concurrently -- it runs separate binaries one after
//! another, which is why the suite left this 24-core box ~84 % idle.
//!
//! To run one module's tests: `cargo test --test all -- <module>::`.
//!
//! Files kept OUT of here, each for a stated reason, are listed in the crate's
//! tests/ directory alongside this one.

// Clippy policy for the differential harnesses: constants are copied verbatim from
// libaom (byte groupings and hex casing included), branches that document distinct
// cases are kept even when their bodies coincide, and const-gated asserts pin build
// state on purpose. Kernel-shape lints follow the crate policy.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::mixed_case_hex_literals,
    clippy::if_same_then_else,
    clippy::assertions_on_constants,
    clippy::field_reassign_with_default,
    clippy::chunks_exact_to_as_chunks,
    clippy::manual_clamp,
    clippy::manual_checked_ops,
    clippy::manual_range_contains
)]
mod animated_avif;
mod chroma_facades_cdiff;
mod common;
mod config_permutations_decode;
mod conformance_corpus;
mod disable_cdf_update_diff;
mod film_grain_diff;
mod fuzz_regression;
mod fuzz_sweep;
mod inter_ratchet;
mod inter_real_frame;
mod inter_walking_skeleton;
mod real_bitstream;
mod superres_diff;
mod superres_tiles_diff;
mod tile_roundtrip;
mod unsupported_refusals;
mod warp_census;
mod whereat_entries;
