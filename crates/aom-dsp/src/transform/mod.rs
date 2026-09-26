//! aom-transform — bit-exact AV1 transform kernels (port of libaom v3.14.1).
//!
//! Every public function is validated byte-for-byte against the C reference by
//! a differential harness in `tests/`. Scalar-first; SIMD specializations must
//! match this scalar output exactly (the same contract libaom holds internally).

internal_mods!(cospi, fdct, inv_txfm1d_gen);
pub mod inv_txfm2d;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod simd;
internal_mods!(special, txfm1d_gen);
pub mod txfm2d;

// The 1-D primitives: internal building blocks of the 2-D kernels, public only
// for the per-stage differentials (`__internals`).
#[cfg(feature = "__internals")]
pub use fdct::{av1_fdct4, clamp_value, half_btf};
#[cfg(not(feature = "__internals"))]
pub(crate) use fdct::{av1_fdct4, clamp_value, half_btf};
#[cfg(feature = "__internals")]
pub use inv_txfm1d_gen::{
    av1_iadst16, av1_iadst8, av1_idct16, av1_idct32, av1_idct4, av1_idct64, av1_idct8,
};
#[cfg(not(feature = "__internals"))]
pub(crate) use inv_txfm1d_gen::{
    av1_iadst16, av1_iadst8, av1_idct16, av1_idct32, av1_idct4, av1_idct64, av1_idct8,
};
#[cfg(feature = "__internals")]
pub use special::{
    av1_fadst4, av1_fidentity16, av1_fidentity32, av1_fidentity4, av1_fidentity8, av1_iadst4,
    av1_iidentity16, av1_iidentity32, av1_iidentity4, av1_iidentity8,
};
#[cfg(not(feature = "__internals"))]
pub(crate) use special::{
    av1_fadst4, av1_fidentity16, av1_fidentity32, av1_fidentity4, av1_fidentity8, av1_iadst4,
    av1_iidentity16, av1_iidentity32, av1_iidentity4, av1_iidentity8,
};
#[cfg(feature = "__internals")]
pub use txfm1d_gen::{av1_fadst16, av1_fadst8, av1_fdct16, av1_fdct32, av1_fdct64, av1_fdct8};
#[cfg(not(feature = "__internals"))]
pub(crate) use txfm1d_gen::{
    av1_fadst16, av1_fadst8, av1_fdct16, av1_fdct32, av1_fdct64, av1_fdct8,
};
