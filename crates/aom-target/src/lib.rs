//! Dependency-injected zensim-quality (Zq) target search for the aom
//! backend — the loop the GOAL's criterion 4 requires, buildable TODAY.
//!
//! **The dependency-injection contract (user directive 2026-08-29):** this
//! LIBRARY holds ZERO codec, FFI, or metric dependencies — with the
//! default feature set the crate builds against nothing at all. The whole
//! encode→decode→judge cycle is the caller's `trial(qindex)` closure; the
//! census harness injects `aomenc`/`aomdec` (libaom CLI) + the zensim
//! Profile-C judge, and the in-repo pure-Rust whole-frame encoder swaps
//! in later with no loop change. Registration + census:
//! `benchmarks/zensim_zq_target_wave_2026-08-29.md`.
//!
//! The judge and the PNG reader the census example needs are optional
//! dependencies behind the non-default `census` feature, so nothing but
//! that example ever pulls them in — `cargo test --workspace` compiles
//! this crate and its unit tests with an empty dependency graph. Run the
//! harness with `just zq-census`.
#![forbid(unsafe_code)]

mod search;
pub use search::{search_target_qindex, TargetOptions, TargetSearchResult};
