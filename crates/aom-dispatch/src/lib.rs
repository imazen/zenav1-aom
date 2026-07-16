//! SIMD dispatch policy for aom-rs — the **scalar pin**.
//!
//! Gate-3 SIMD work must never perturb byte-exactness. Two mechanisms defend
//! that, and this crate is the first:
//!
//! 1. **The scalar pin (this crate).** `AOM_FORCE_SCALAR=1` disables every
//!    runtime-dispatchable archmage SIMD token process-wide, so every
//!    `incant!` dispatch in every aom-rs kernel crate falls through to its
//!    `_scalar` variant — the transcribed C port untouched by SIMD. Running
//!    the FULL byte-exactness suites under the pin proves the scalar path is
//!    intact:
//!    ```text
//!    AOM_FORCE_SCALAR=1 cargo test --workspace --no-fail-fast
//!    ```
//! 2. **Per-kernel differentials** (in each kernel crate): SIMD variants are
//!    bit-identical to the scalar port at every dispatch tier, via
//!    `archmage::testing::for_each_token_permutation`.
//!
//! Every dispatch entry point (the ONE `#[arcane]`/`incant!` boundary per hot
//! loop) must call [`scalar_forced`] before its first dispatch:
//!
//! ```ignore
//! pub fn kernel(args...) {
//!     let _ = aom_dispatch::scalar_forced(); // one-time env pin (~1ns after init)
//!     incant!(kernel_impl(args...))
//! }
//! ```
//!
//! The call is a cached `OnceLock` read after the first invocation. The pin
//! works through archmage's own token-availability atomics, so the REAL
//! dispatch path (including its scalar fallback plumbing) is what runs under
//! the pin — not a parallel code path.
//!
//! Scope: the pin covers every token that is runtime-detected in this build
//! (x86-64: v2/crypto/v3/v3crypto/v4/v4x/fp16; aarch64: neon family/arm-v2/
//! arm-v3). Tokens whose features are compile-time guaranteed cannot be
//! disabled (e.g. wasm32 built WITH `+simd128`, or any build using
//! `-Ctarget-cpu`); aom-rs never compiles kernels with `-Ctarget-cpu`
//! (runtime dispatch is what users get), so on native CI targets the pin is
//! total.

#![forbid(unsafe_code)]

use std::sync::OnceLock;

/// True when the `AOM_FORCE_SCALAR` environment variable pins this process
/// to scalar dispatch (set, non-empty, and not `"0"`).
///
/// The FIRST call applies the pin: every runtime-dispatchable archmage SIMD
/// token is disabled process-wide, so `Token::summon()` returns `None` at
/// every subsequent dispatch site and `incant!` falls through to `_scalar`.
/// Call this at every dispatch entry point BEFORE `incant!` (see the crate
/// docs for the pattern); after initialization it is a single atomic load.
pub fn scalar_forced() -> bool {
    static PIN: OnceLock<bool> = OnceLock::new();
    *PIN.get_or_init(|| {
        let forced = std::env::var_os("AOM_FORCE_SCALAR").is_some_and(|v| !v.is_empty() && v != "0");
        if forced {
            disable_all_simd_tokens();
        }
        forced
    })
}

/// Disable every runtime-dispatchable archmage token, process-wide.
///
/// Tokens are zero-sized stubs on foreign architectures and
/// compile-time-guaranteed tokens refuse disablement — both return `Err`,
/// which is fine: a stub's `summon()` already returns `None`, and aom-rs
/// builds never bake target features in (`-Ctarget-cpu` is banned), so on
/// native targets every SIMD token here is disableable.
fn disable_all_simd_tokens() {
    use archmage as a;
    // x86-64 hierarchy (V1 is compile-time on the x86-64 baseline — sse2 —
    // and aom-rs emits no _v1/_v2-only kernels; disabling it is refused,
    // which is fine because everything above it IS disabled).
    let _ = a::X64V1Token::dangerously_disable_token_process_wide(true);
    let _ = a::X64V2Token::dangerously_disable_token_process_wide(true);
    let _ = a::X64CryptoToken::dangerously_disable_token_process_wide(true);
    let _ = a::X64V3Token::dangerously_disable_token_process_wide(true);
    let _ = a::X64V3CryptoToken::dangerously_disable_token_process_wide(true);
    let _ = a::X64V4Token::dangerously_disable_token_process_wide(true);
    let _ = a::X64V4xToken::dangerously_disable_token_process_wide(true);
    let _ = a::Avx512Fp16Token::dangerously_disable_token_process_wide(true);
    // aarch64 hierarchy.
    let _ = a::NeonToken::dangerously_disable_token_process_wide(true);
    let _ = a::NeonAesToken::dangerously_disable_token_process_wide(true);
    let _ = a::NeonSha3Token::dangerously_disable_token_process_wide(true);
    let _ = a::NeonCrcToken::dangerously_disable_token_process_wide(true);
    let _ = a::Arm64V2Token::dangerously_disable_token_process_wide(true);
    let _ = a::Arm64V3Token::dangerously_disable_token_process_wide(true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    /// The pin's token-disable sweep actually flips `summon()` to `None` for
    /// the tiers our kernels dispatch on (exercised via the disable fn
    /// directly — the env var route is process-global state we don't toggle
    /// inside a shared test process).
    #[test]
    fn disable_sweep_kills_summon_and_reenables() {
        let _lock = archmage::testing::lock_token_testing();
        // Only meaningful where a SIMD tier exists at runtime (x86-64/aarch64 CI).
        let had_v3 = archmage::X64V3Token::summon().is_some();
        let had_neon = archmage::NeonToken::summon().is_some();
        disable_all_simd_tokens();
        assert!(
            archmage::X64V3Token::summon().is_none(),
            "v3 must be pinned off"
        );
        assert!(
            archmage::X64V4Token::summon().is_none(),
            "v4 must be pinned off"
        );
        assert!(
            archmage::NeonToken::summon().is_none(),
            "neon must be pinned off"
        );
        // Re-enable (undo — other tests in this process must see real tiers).
        use archmage as a;
        let _ = a::X64V2Token::dangerously_disable_token_process_wide(false);
        let _ = a::X64CryptoToken::dangerously_disable_token_process_wide(false);
        let _ = a::X64V3Token::dangerously_disable_token_process_wide(false);
        let _ = a::X64V3CryptoToken::dangerously_disable_token_process_wide(false);
        let _ = a::X64V4Token::dangerously_disable_token_process_wide(false);
        let _ = a::X64V4xToken::dangerously_disable_token_process_wide(false);
        let _ = a::Avx512Fp16Token::dangerously_disable_token_process_wide(false);
        let _ = a::NeonToken::dangerously_disable_token_process_wide(false);
        let _ = a::NeonAesToken::dangerously_disable_token_process_wide(false);
        let _ = a::NeonSha3Token::dangerously_disable_token_process_wide(false);
        let _ = a::NeonCrcToken::dangerously_disable_token_process_wide(false);
        let _ = a::Arm64V2Token::dangerously_disable_token_process_wide(false);
        let _ = a::Arm64V3Token::dangerously_disable_token_process_wide(false);
        assert_eq!(archmage::X64V3Token::summon().is_some(), had_v3);
        assert_eq!(archmage::NeonToken::summon().is_some(), had_neon);
    }

    #[test]
    fn env_pin_absent_means_not_forced() {
        // The bench/test processes that DON'T set the env must never pin.
        // (Processes running the pinned suite set AOM_FORCE_SCALAR before
        // spawn; this test asserts the default-off behavior in this process.)
        if std::env::var_os("AOM_FORCE_SCALAR").is_none() {
            assert!(!scalar_forced());
            assert!(
                archmage::ScalarToken::summon().is_some(),
                "scalar is always available"
            );
        }
    }
}
