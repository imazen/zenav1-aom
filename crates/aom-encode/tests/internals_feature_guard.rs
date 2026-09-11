//! The anti-KB-42 guard for this crate's `required-features` harness.
//!
//! `[[test]] name = "all"` carries `required-features = ["__internals"]`, and a
//! `required-features` target is **silently skipped** when the feature is off —
//! so `cargo test -p zenav1-aom-encode` alone would report green having built
//! none of the integration byte gates. That is precisely the failure KB-42
//! documents (four consecutive landings gated on `--lib` while 23 byte/RD gates
//! stayed broken across six red CI runs).
//!
//! This file is deliberately NOT gated: it always compiles, always runs, and
//! fails with the exact command to re-run. Verified in both directions.
#[test]
fn the_integration_harness_requires_the_internals_feature() {
    if cfg!(feature = "__internals") {
        return;
    }
    panic!(
        "the `all` integration harness was SKIPPED: it carries \
         required-features = [\"__internals\"] and that feature is off, so every \
         byte-identity gate in this crate just reported green without building. \
         Re-run one of:\n\
         \x20 just gate-landing\n\
         \x20 cargo nextest run --cargo-profile test-fast --workspace\n\
         \x20 cargo test -p zenav1-aom-encode --features __internals"
    );
}
