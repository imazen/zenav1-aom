//! The one test in this crate that is NOT behind `required-features`.
//!
//! `tests/all` carries every integration test and requires `__internals`, so a
//! plain `cargo test -p zenav1-aom-decode` would build NONE of them and report
//! a cheerful green. That is exactly the KB-42 failure mode -- four landings
//! gated on a selection that ran no byte gates, 23 broken gates across six red
//! CI runs -- and a `required-features` target reintroduces it by construction
//! unless something unconditional notices.
//!
//! This is that something. It always compiles and always runs, so the suite can
//! never be silently empty: either `__internals` is on and the real harness ran,
//! or this fails and says what to pass.
#[test]
fn the_real_suite_requires_the_internals_feature() {
    assert!(
        cfg!(feature = "__internals"),
        "the `tests/all` harness was NOT built: it requires the `__internals` \
         feature. Run `cargo test -p zenav1-aom-decode --features __internals` \
         (or `just gate-landing`, which does). Without it this crate's \
         integration tests are silently skipped."
    );
}
