//! Serialises the tests that permute the PROCESS-GLOBAL dispatch state.
//!
//! `archmage::testing::for_each_token_permutation` enables and disables
//! dispatch tokens **process-wide**. Thirteen modules of this binary call it,
//! and `cargo test` runs a binary's tests on an intra-binary thread pool — so
//! two sweeps overlap and each sees the other's token state. The symptom is a
//! kernel taking the SIMD path while the harness believes the tier is scalar,
//! i.e. `cdef_find_dir_simd_diff`'s *"the scalar tier must decline so the entry
//! runs the scalar port"*.
//!
//! **This is a regression from the test consolidation** (`a88e739`, "consolidate
//! 320 test binaries into 7"): before it every `tests/*.rs` was its own binary
//! and therefore its own PROCESS, so permutation tests could not see each other.
//! That commit's doc comment names the mechanism — *"lets `cargo test`'s
//! intra-binary thread pool run ALL of these concurrently"* — without noticing
//! that global dispatch state is not thread-safe to permute.
//!
//! # Scope — MEASURED 2026-09-09, and this guard does NOT close all of it
//!
//! | runner | `cargo test --test all cdef_find_dir` |
//! |---|---|
//! | `cargo test`, multi-threaded | **flaky — 3 failures in 15 runs** |
//! | `cargo test -- --test-threads=1` | **0 in 10** |
//! | `cargo nextest` (process per test) | **0 in 10** |
//!
//! So the cause is intra-binary CONCURRENCY, confirmed from both directions.
//!
//! **This mutex removes the sweep-versus-sweep race only.** A residual remains
//! and is stated rather than papered over: a sweep also races any concurrent
//! test that merely *reads* dispatch state by calling a kernel, and those tests
//! cannot take this lock without every one of the crate's hundreds of tests
//! taking it. Closing that would need an `RwLock` across the whole binary,
//! which is disproportionate to a hazard both supported runners already avoid.
//!
//! **`just gate-landing` is unaffected, and that is measured, not assumed**: it
//! runs `cargo nextest`, which gives every test its own process, so the race
//! cannot occur there and this mutex is a no-op — 0 failures in 10 runs above,
//! and the full gate has read 1506/1506 on every run today. What is left is the
//! local multi-threaded `cargo test` path, where a spurious red costs exactly
//! the time it cost here: an unrelated change was implicated for several
//! minutes. **If you hit it, use `--test-threads=1` or `cargo nextest`.**
//!
//! Same class as KB-50's counting-allocator mutex and KB-48's `timing_serial()`:
//! process-global state plus intra-binary concurrency.

use std::sync::{Mutex, OnceLock};

/// Take this before `for_each_token_permutation` and hold it for the whole
/// sweep. Poisoning is deliberately ignored — a panicking permutation test has
/// already failed, and its neighbours should report their own result rather
/// than a cascade of poison errors.
pub fn dispatch_serial() -> std::sync::MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
