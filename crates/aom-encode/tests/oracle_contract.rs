//! Contract tests for the C oracle itself, not for any ported kernel.
//!
//! Two whole classes of differential defect are invisible on this aarch64 host
//! and only surface on x86. This file pins what CAN be checked here and states
//! plainly what cannot, so the next session does not read a green run as
//! evidence the contract holds on every ISA.
//!
//! # 1. RTCD dispatch
//! libaom resolves most kernels through `RTCD_EXTERN` function pointers that are
//! NULL until `aom_dsp_rtcd()` / `av1_rtcd()` run. A shim that reaches one
//! without `aom_sys_ref::ref_init()` segfaults — but only where the name really
//! is a pointer. On this build `config/aom_dsp_rtcd.h` `#define`s many of them
//! straight to their NEON implementations, so **no pointer exists and nothing
//! can be null**. `nm -g` on the archive shows the same split: `aom_sad32x32`
//! and `av1_warp_affine` are `C` (common) symbols, while `aom_upsampled_pred`,
//! `aom_comp_mask_pred`, `aom_convolve_copy` and the whole `aom_obmc_*` family
//! are absent entirely.
//!
//! Consequence: **a missing `ref_init()` in a wrapper that reaches one of the
//! `#define`d names cannot be detected by any test on this host.** The
//! mitigation is structural rather than observational — every `ref_*` wrapper
//! added for the inter-encode port calls `ref_init()` unconditionally, whether
//! or not its call tree is believed to dispatch.
//!
//! # 2. Buffer alignment
//! libaom's own callers hand these kernels `DECLARE_ALIGNED(16, ...)` locals or
//! `aom_memalign`'d frame buffers; a Rust `Vec` is 1-byte aligned. The x86 SIMD
//! kernels may use aligned loads/stores, the NEON ones do not — so a shim that
//! passes a `Vec` pointer straight through faults on x86 and passes here. The
//! inter-encode shims bounce every such buffer through 64-byte-aligned scratch.
//! That is also unobservable here; this file records it rather than testing it.

use aom_sys_ref::{RTCD_PROBE_NAMES, ref_init, ref_rtcd_probe};

#[test]
fn ref_init_leaves_no_rtcd_pointer_null() {
    ref_init();
    let mut pointer_names = Vec::new();
    let mut define_names = Vec::new();
    for (i, name) in RTCD_PROBE_NAMES.iter().enumerate() {
        let (is_ptr, non_null) = ref_rtcd_probe(i);
        if is_ptr {
            assert!(
                non_null,
                "{name} is an RTCD function pointer in this build and is still NULL \
                 after ref_init() — any shim reaching it will segfault"
            );
            pointer_names.push(*name);
        } else {
            define_names.push(*name);
        }
    }
    // Not an assertion about correctness — a printed record of how much of the
    // check this ISA can actually perform.
    println!(
        "RTCD probe: {} of {} names are real pointers here ({:?}); {} are #defined \
         to a direct implementation and cannot be observed ({:?})",
        pointer_names.len(),
        RTCD_PROBE_NAMES.len(),
        pointer_names,
        define_names.len(),
        define_names
    );
    assert!(
        !pointer_names.is_empty(),
        "no probed name is a function pointer in this build, so this test proves \
         nothing at all — the probe list needs a name that stays dispatched here"
    );
}

#[test]
fn rtcd_pointers_are_null_before_init_is_not_asserted() {
    // Deliberately NOT a test that the pointers are null before ref_init: the
    // test binary shares one process, any earlier test may have initialised
    // them, and `ref_init` is a `Once`. Asserting a pre-init state here would
    // be order-dependent and would pass or fail for reasons unrelated to the
    // contract. What matters — and what the sibling test above checks — is that
    // after `ref_init()` nothing reachable is null.
    ref_init();
    for i in 0..RTCD_PROBE_NAMES.len() {
        let (is_ptr, non_null) = ref_rtcd_probe(i);
        if is_ptr {
            assert!(non_null);
        }
    }
}
