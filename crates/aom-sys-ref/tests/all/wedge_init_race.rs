//! Regression gate for the `av1_init_wedge_masks` init race.
//!
//! **The failure it prevents:** `crates/aom-dsp/tests/interintra_diff.rs`
//! SIGSEGV'd intermittently on the `macos-latest` (aarch64) CI runner, signal 11,
//! before any test printed. The oracle is built `-DCONFIG_MULTITHREAD=0`
//! (`crates/aom-sys-ref/build.rs:263`, deliberate — it is the determinism
//! definition), which selects the **no-synchronisation** `aom_once`
//! (`upstream/aom_ports/aom_once.h:70-80`: `static volatile int done; if (!done)
//! { func(); done = 1; }`). `av1_init_wedge_masks`
//! (`upstream/av1/common/reconinter.c:600`) is guarded by exactly that, and its
//! body opens with `memset(wedge_masks, 0, sizeof(wedge_masks))`
//! (`:497`) — so for the whole duration of the init every
//! `av1_wedge_params_lookup[bsize].masks[s][w]` pointer is NULL.
//! `av1_get_contiguous_soft_mask` (`upstream/av1/common/reconinter.h:456-460`)
//! returns that pointer straight to the shim's `memcpy`. libtest runs a binary's
//! tests on concurrent threads, `interintra_diff` has two that reach the wedge
//! table, and a second thread entering the unsynchronised `aom_once` re-NULLs
//! entries the first thread has already published → `memcpy` from NULL → SIGSEGV.
//!
//! The fix forces every `aom_once`-guarded libaom init from `ref_init`'s Rust
//! `Once` (single-threaded funnel), so `done` is already 1 by the time any test
//! thread reaches C. This test drives the exact shape that crashed: K threads
//! released together, each pulling a wedge mask as its first C call.
//!
//! Pre-fix this binary dies with SIGSEGV (or the `NULL wedge mask` panic the
//! shim now raises instead of faulting); post-fix it passes.

use std::sync::{Arc, Barrier};

/// wedge-eligible (bsize, bw, bh), same set as `interintra_diff`.
const WEDGE_BSIZES: [(usize, usize, usize); 9] = [
    (3, 8, 8),
    (4, 8, 16),
    (5, 16, 8),
    (6, 16, 16),
    (7, 16, 32),
    (8, 32, 16),
    (9, 32, 32),
    (18, 8, 32),
    (19, 32, 8),
];

#[test]
fn concurrent_first_wedge_fetch_is_safe() {
    // More threads than the runner has cores is the point: it maximises the
    // skew between two threads inside the unsynchronised `aom_once`.
    let threads = 8usize;
    let barrier = Arc::new(Barrier::new(threads));
    let mut handles = Vec::with_capacity(threads);
    for t in 0..threads {
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            // First C call this thread makes, exactly as in interintra_diff.
            let mut n = 0u32;
            for &(bsize, bw, bh) in &WEDGE_BSIZES {
                for index in 0..16usize {
                    let m = aom_sys_ref::ref_ii_wedge_mask(bsize, index, bw, bh)
                        .unwrap_or_else(|| panic!("thread {t}: no wedge mask for bsize={bsize}"));
                    assert_eq!(m.len(), bw * bh);
                    // A raced fetch reads a half-published table: the mask is
                    // all-zero where the memset landed after the copy. A real
                    // wedge mask is never uniformly zero (it is a 0..64 ramp).
                    assert!(
                        m.iter().any(|&v| v != 0),
                        "thread {t}: all-zero wedge mask bsize={bsize} index={index} — \
                         av1_init_wedge_masks raced (oracle is CONFIG_MULTITHREAD=0, so \
                         aom_once does not synchronise)"
                    );
                    n += 1;
                }
            }
            assert_eq!(n, 9 * 16);
        }));
    }
    for h in handles {
        h.join().expect("wedge-fetch thread panicked");
    }
    eprintln!("wedge_init_race: {threads} concurrent first-fetch threads, no NULL/zero mask");
}
