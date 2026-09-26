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
// Clippy policy for a bit-exact C port (see docs/DIFFERENTIAL_PLAYBOOK.md): kernels
// mirror libaom's signatures and loop shapes line for line so a differential can be
// read against the C side, and some C float idioms are NaN-sensitive.
#![allow(
    clippy::too_many_arguments,   // C kernel signatures mirrored 1:1
    clippy::needless_range_loop,  // index loops mirror the C reference line for line
    clippy::manual_memcpy,        // explicit copy loops mirror the C reference
    clippy::type_complexity,      // harness-facing tuples
    clippy::neg_cmp_op_on_partial_ord, // `!(a < b)` is not `a >= b` under NaN; the CNN mirrors C
    clippy::field_reassign_with_default, // C init-then-assign mirrors
    clippy::chunks_exact_to_as_chunks,   // `as_chunks` is not on the pinned toolchain
    clippy::collapsible_if,       // nested ifs mirror the C control flow, with C line refs between them
    clippy::manual_clamp,         // C's chained min/max: `clamp` panics on inverted bounds and differs under NaN
    clippy::manual_range_contains, // C comparison chains kept verbatim
    clippy::excessive_precision,  // NN weight tables copied verbatim from C
    clippy::while_let_loop,       // explicit `loop { let Some(..) = .. else { break } }` keeps the scheduler's exits readable
    clippy::large_enum_variant    // `SbTree` variants are the partition shapes; boxing the leaf is measured slower
)]
// The docs deliberately link implementation items that live behind the default-off
// `__internals` feature (or are `pub(crate)`): the links resolve for a harness build
// and read as plain code for a consumer. `just doc-check` runs with `-D warnings`.
#![allow(rustdoc::private_intra_doc_links)]
#![forbid(unsafe_code)]

pub mod blocksize;
pub mod cdef;
pub mod census;
pub mod cnn;
pub mod convolve;
pub mod crc32c;
pub mod dispatch;
pub mod dist;
pub mod entropy;
pub mod inter;
pub mod intra;
pub mod kmeans;
pub mod loopfilter;
pub mod lowbd;
pub mod par;
pub mod quant;
pub mod recon;
pub mod restore;
// The SSE2/AVX2 intrinsic vocabulary the fused transform kernels are written
// in: the real `core::arch` module on x86-64, NEON twins on aarch64. Every
// consumer is gated the same way, so on any other target (i686, wasm32, ...)
// the module must not exist at all — its `pub(crate) use imp::*` has nothing
// to re-export there. MEASURED 2026-09-25: CI's `portability i686` leg failed
// with E0432 on that line the first time it ran on this code.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod sse_neon;
pub mod trace;
pub mod transform;
pub mod txb;
