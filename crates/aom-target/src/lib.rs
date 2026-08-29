//! Dependency-injected zensim-quality (Zq) target search for the aom
//! backend — the loop the GOAL's criterion 4 requires, buildable TODAY.
//!
//! **The dependency-injection contract (user directive 2026-08-29):** this
//! crate holds ZERO codec, FFI, or metric dependencies. The whole
//! encode→decode→judge cycle is the caller's `trial(qindex)` closure; the
//! census harness injects `aomenc`/`aomdec` (libaom CLI) + the zensim
//! Profile-C judge, and the in-repo pure-Rust whole-frame encoder swaps
//! in later with no loop change. Registration + census:
//! `benchmarks/zensim_zq_target_wave_2026-08-29.md`.
#![forbid(unsafe_code)]

mod search;
pub use search::{search_target_qindex, TargetOptions, TargetSearchResult};
