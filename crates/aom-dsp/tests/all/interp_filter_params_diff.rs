//! Differential harness for the block-size-dependent interpolation-filter
//! selection — `av1_get_interp_filter_params_with_block_size`
//! (`av1/common/filter.h:249`) and the tables it selects — against
//! `aom_dsp::convolve::{filter_table_id, kernel_row}`.
//!
//! # Evidence tier
//!
//! **Tier 1c.** The C function is a `static inline` in a header, so it has no
//! address; `crates/aom-sys-ref/shim/interp_params_shim.c` compiles it and
//! reports what it returned. Both halves cross the boundary — WHICH table and
//! the table's CONTENTS — so a port with the right selector and one mistyped
//! coefficient fails as loudly as one with the wrong selector.
//!
//! # Why this exists
//!
//! On an axis of four pixels or fewer the encoder switches to
//! `av1_interp_4tap`: a different set of coefficients at the SAME eight taps
//! (`filter.h:240-246` declares them `SUBPEL_TAPS`), on which MULTITAP_SHARP
//! collapses onto the REGULAR entry. Every chroma plane of an 8x8 luma block
//! takes that path, so an inter predictor build that ignored it would be wrong
//! on most chroma blocks in the stream — while looking right on every luma
//! one.

use aom_dsp::convolve::{filter_table_id, kernel_row};
use aom_sys_ref as cref;

/// `SWITCHABLE_FILTERS` (3) plus BILINEAR — the `InterpFilter` values this
/// selection is reachable with. `MULTITAP_SHARP2` (4) is excluded by C's own
/// `interp_filter != MULTITAP_SHARP2` guard and is reachable only from
/// temporal filtering, whose blocks are >= 16.
const FILTERS: [i32; 4] = [0, 1, 2, 3];

#[test]
fn interp_filter_dims_match_the_shim() {
    let (shifts, taps) = cref::ref_interp_filter_dims();
    assert_eq!(shifts, 16, "SUBPEL_SHIFTS");
    assert_eq!(taps, 8, "SUBPEL_TAPS");
}

#[test]
fn filter_table_selection_matches_c() {
    // Every block extent the encoder can filter along one axis, plus the two
    // sides of the `<= 4` switch and the degenerate small values C's `w <= 4`
    // test admits.
    let sizes: Vec<i32> = (1..=8).chain([16, 32, 64, 128]).collect();
    let mut narrow = 0usize;
    let mut wide = 0usize;
    let mut sharp_collapses = 0usize;

    for &f in &FILTERS {
        for &w in &sizes {
            let (taps, want) = cref::ref_interp_filter_table(f, w);
            assert_eq!(taps, 8, "av1_interp_4tap declares SUBPEL_TAPS, not 4");
            let id = filter_table_id(f as usize, w as usize);
            for (subpel, want_row) in want.iter().enumerate() {
                assert_eq!(
                    kernel_row(id, subpel),
                    want_row,
                    "filter={f}, block_size={w}, subpel={subpel}"
                );
            }
            if w <= 4 {
                narrow += 1;
                // C's comment at filter.h:239 — "For w<=4, MULTITAP_SHARP is
                // the same as EIGHTTAP_REGULAR". Check it rather than trust it.
                if f == 2 {
                    let (_, reg) = cref::ref_interp_filter_table(0, w);
                    assert_eq!(want, reg, "SHARP must collapse onto REGULAR at w<=4");
                    sharp_collapses += 1;
                }
            } else {
                wide += 1;
            }
        }
    }
    assert!(narrow > 0 && wide > 0, "both sides of the w<=4 switch must run");
    assert!(sharp_collapses > 0, "the SHARP collapse was never checked");
}

/// The narrow tables must actually DIFFER from the wide ones, or the selector
/// is being compared against itself.
#[test]
fn narrow_and_wide_tables_differ() {
    for &f in &FILTERS {
        let (_, narrow) = cref::ref_interp_filter_table(f, 4);
        let (_, wide) = cref::ref_interp_filter_table(f, 8);
        if f == 3 {
            // BILINEAR is the same table at every size — asserted, not assumed.
            assert_eq!(narrow, wide, "BILINEAR must not switch on block size");
        } else {
            assert_ne!(narrow, wide, "filter {f}: the w<=4 table must differ");
        }
    }
}
