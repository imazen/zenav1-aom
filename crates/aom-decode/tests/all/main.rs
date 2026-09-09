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

mod common;
mod animated_avif;
mod chroma_facades_cdiff;
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
mod warp_census;
mod whereat_entries;
