//! aom-dsp — consolidated DSP + entropy kernels for the pure-Rust, bit-exact
//! libaom v3.14.1 port.
//!
//! Each former per-kernel crate is a module here: [`transform`], [`quant`],
//! [`txb`], [`cdef`], [`restore`], [`intra`], [`loopfilter`], [`dist`],
//! [`inter`], [`convolve`], [`recon`], [`lowbd`], [`dispatch`], the entropy
//! coder and the syntax layers on it ([`entropy`]), the default-off content
//! census ([`census`]), and the shared `BLOCK_SIZE`/`TX_SIZE` geometry
//! ([`blocksize`]). Consolidating them into one publishable crate keeps the
//! release surface small (a single `cargo publish` / version bump) while
//! preserving the exact kernel byte-for-byte — the module paths are the only
//! thing that changed (`aom_transform::X` → `aom_dsp::transform::X`).
//!
//! The consolidation is DONE: every former sub-crate is physically absorbed
//! into `src/<family>/` and there are no `pub use aom_X as X` shims left.
#![forbid(unsafe_code)]

pub mod blocksize;
pub mod cdef;
pub mod census;
pub mod convolve;
pub mod dispatch;
pub mod dist;
pub mod entropy;
pub mod inter;
pub mod intra;
pub mod loopfilter;
pub mod lowbd;
pub mod quant;
pub mod recon;
pub mod restore;
pub mod transform;
pub mod txb;
