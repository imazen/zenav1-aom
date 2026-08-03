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
//! ```text
//! pub fn kernel(args...) {
//!     let _ = crate::dispatch::scalar_forced(); // one-time env pin (~1ns after init)
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
//! (x86-64: v2/crypto/v3/v3crypto/v4/v4x/fp16; aarch64: arm-v2/arm-v3 and the
//! optional neon extensions — aes/sha3/crc). Tokens whose features are
//! compile-time guaranteed cannot be disabled (archmage returns `Err`), and
//! aom-rs never compiles kernels with `-Ctarget-cpu` (runtime dispatch is what
//! users get), so on **x86-64** the pin is total: every tier above the `sse2`
//! baseline is runtime-detected.
//!
//! # aarch64: the pin needs `testable_dispatch`, and only works in TEST builds
//!
//! MEASURED 2026-07-25 (Apple M4 Pro, `aarch64-apple-darwin`) — plain `neon` is
//! a **compile-time-guaranteed baseline feature** of every aarch64 target, and
//! archmage refuses to disable compile-time-guaranteed tokens. So whether the
//! pin reaches NEON depends on a cargo feature:
//!
//! * **With `archmage/testable_dispatch`** (enabled in this crate's
//!   `[dev-dependencies]`) archmage forces RUNTIME detection even where
//!   compile-time detection would succeed, so `NeonToken` becomes disableable
//!   and the pin IS total. This is the configuration the byte-exactness suites
//!   run in, so mechanism 1 does cover aarch64 — but only because that
//!   dev-feature is present. It was NOT present until 2026-07-25; before that,
//!   `AOM_FORCE_SCALAR=1` on ARM silently re-ran the NEON path, and
//!   `for_each_token_permutation` silently EXCLUDED neon from its permutation
//!   set (the transform differential reported `simd_perms=0`, i.e. it compared
//!   the scalar path against itself; with the feature it reports 25
//!   permutations). Keep the feature.
//! * **Without it — i.e. every non-test build, including benchmarks** —
//!   `dangerously_disable_token_process_wide(true)` returns `Err`,
//!   `NeonToken::summon()` keeps returning `Some`, and `AOM_FORCE_SCALAR=1` is a
//!   **no-op for the NEON tier**. Do not use a pinned-vs-unpinned pair as a
//!   scalar/SIMD benchmark on ARM: a full 77-row pair on this box came back
//!   inside ±3% (noise), because both halves ran the same code. See the module
//!   docs of `zenav1-aom-dsp-bench`'s `dsp_kernels` bench.
//!
//! Making the pin total in production builds too would mean branching on
//! [`scalar_forced`] at each dispatch entry and calling the `_scalar` variant
//! directly instead of relying on token availability. That is a deliberate
//! design change, not a bug fix, and is NOT implemented.


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

    /// The pin's token-disable sweep flips `summon()` to `None` for every tier
    /// it CAN disable, and the aarch64 `neon` baseline is pinned as the
    /// documented exception it actually is (exercised via the disable fn
    /// directly — the env var route is process-global state we don't toggle
    /// inside a shared test process).
    ///
    /// This test previously asserted `NeonToken::summon().is_none()`
    /// unconditionally. That passes vacuously on x86-64 (where `NeonToken` is a
    /// foreign-architecture stub whose `summon()` is always `None`) and FAILS on
    /// a real aarch64 host — measured 2026-07-25 on an Apple M4 Pro, the first
    /// time this suite was run natively on ARM. `neon` is a compile-time
    /// guaranteed baseline feature of every aarch64 target, and archmage
    /// deliberately refuses to disable compile-time-guaranteed tokens, so the
    /// old assertion encoded a belief that is false on the architecture it was
    /// nominally about. See this module's docs for what that costs (the scalar
    /// pin is an x86-64-only byte-exactness mechanism).
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
        // On aarch64 this only holds because the dev-dependency enables
        // archmage's `testable_dispatch`, which forces RUNTIME detection for
        // tokens whose features are compile-time guaranteed and thereby makes
        // baseline `neon` disableable. Without that feature the sweep silently
        // fails to pin neon (verified), so this assertion is also the guard that
        // the feature stays enabled — see this crate's Cargo.toml and the
        // "production builds" caveat in this module's docs.
        assert!(
            archmage::NeonToken::summon().is_none(),
            "neon must be pinned off — on aarch64 this requires archmage's \
             `testable_dispatch` dev-feature (it forces runtime detection for \
             the compile-time-guaranteed neon baseline); if this fails, that \
             feature was dropped from dev-dependencies"
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
