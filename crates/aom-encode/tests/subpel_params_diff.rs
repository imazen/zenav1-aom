//! Differential harness for the encoder-side subpel-parameter derivation —
//! `enc_calc_subpel_params` (`av1/encoder/reconinter_enc.c:32`) and
//! `init_subpel_params` (`av1/common/reconinter.h:131`) — against the port in
//! `aom_encode::inter_pred_enc`.
//!
//! # Evidence tier
//!
//! **Tier 1c.** Both C functions are `static inline`; neither has an address
//! a differential could take. `crates/aom-sys-ref/shim/reconinter_enc_shim.c`
//! compiles libaom's own reconinter_enc.c (which `#include`s the header the
//! second one lives in) verbatim and calls them.
//!
//! The scale factors are NOT constructed on the C side by hand: the shim runs
//! `av1_setup_scale_factors_for_frame`, the real exported function, and
//! reports what it produced, so [`scale_factors_agree`] can check that the
//! port's own constructor built the same thing before any subpel test relies
//! on it. If those ever diverged, every other test here would be comparing
//! two different reference frames.
//!
//! # Why the source pointer crosses as an OFFSET
//!
//! C returns `pre_buf->buf0 + (pos_y >> SCALE_SUBPEL_BITS) * stride +
//! (pos_x >> SCALE_SUBPEL_BITS)`. That offset is **signed**: `pos_x`/`pos_y`
//! are clamped to `inter_pred_params->{left,top}`, which are negative (the
//! reference plane's border margin), so a block predicting up and to the left
//! of the plane origin legitimately lands before `buf0`. The shim passes
//! `buf0 == NULL` and returns the resulting pointer as an integer, which makes
//! that offset a value to compare rather than a wild address to dereference.

mod common;
use common::Rng;

use aom_dsp::inter::scale::ScaleFactors;
use aom_encode::inter_pred_enc::{InterBlockParams, enc_calc_subpel_params, init_subpel_params};
use aom_sys_ref as cref;

/// Frame-size pairs `valid_ref_frame_size` accepts (a reference may be at most
/// 2x larger and 16x smaller per axis), plus the unscaled identity.
fn size_pairs() -> Vec<((i32, i32), (i32, i32))> {
    let this = [(64, 64), (176, 144), (352, 288), (1280, 720)];
    let mut v = Vec::new();
    for &(w, h) in &this {
        v.push(((w, h), (w, h))); // 1:1 — the unscaled fast path
        v.push(((2 * w, 2 * h), (w, h)));
        v.push(((w / 2, h / 2), (w, h)));
        v.push(((w * 3 / 2, h), (w, h))); // scaled on one axis only
        v.push(((w, h * 3 / 2), (w, h)));
    }
    v
}

#[test]
fn scale_factors_agree() {
    let mut scaled = 0usize;
    for (r, t) in size_pairs() {
        let port = ScaleFactors::for_frame(r.0, r.1, t.0, t.1);
        let want = cref::ref_rie_scale_factors(r, t);
        assert_eq!(
            [
                port.x_scale_fp,
                port.y_scale_fp,
                port.x_step_q4,
                port.y_step_q4
            ],
            want,
            "av1_setup_scale_factors_for_frame(ref={r:?}, this={t:?})"
        );
        if port.is_scaled() {
            scaled += 1;
        }
    }
    assert!(scaled > 0, "no scaled reference in the sweep");
}

#[test]
fn enc_calc_subpel_params_matches_c() {
    let mut rng = Rng(0x5EED_0E01);
    let mut clamped_lo = 0usize;
    let mut clamped_hi = 0usize;
    let mut scaled_cells = 0usize;
    let mut negative_offsets = 0usize;

    for (r, t) in size_pairs() {
        let sf = ScaleFactors::for_frame(r.0, r.1, t.0, t.1);
        for (ssx, ssy) in [(0u32, 0u32), (1, 1), (1, 0), (0, 1)] {
            let (pre_w, pre_h) = (r.0 >> ssx, r.1 >> ssy);
            let pre_stride = pre_w + 32;
            for _ in 0..300 {
                // A block position anywhere in the plane, INCLUDING the
                // extremes: the derivation's whole job at the edges is the
                // clamp against the border margin.
                let pix_row = rng.range(0, (pre_h).max(1));
                let pix_col = rng.range(0, (pre_w).max(1));
                // MVs at the coded range (`MV_UPP`/`MV_LOW` are ±(1<<14)) and
                // small ones, so both the clamped and unclamped arms fire.
                let big = rng.next() % 3 == 0;
                let span = if big { 1 << 14 } else { 64 };
                let mv = (rng.range(-span, span) as i16, rng.range(-span, span) as i16);
                let params = InterBlockParams::new(pix_row, pix_col, ssx, ssy);
                let (sp, off) = enc_calc_subpel_params(mv, &params, &sf, pre_w, pre_h, pre_stride);
                let want = cref::ref_rie_enc_calc_subpel_params(
                    mv,
                    pix_row,
                    pix_col,
                    ssx as i32,
                    ssy as i32,
                    r,
                    t,
                    (pre_w, pre_h),
                    pre_stride,
                );
                assert_eq!(
                    [
                        sp.xs,
                        sp.ys,
                        sp.subpel_x,
                        sp.subpel_y,
                        sp.pos_x,
                        sp.pos_y,
                        off
                    ],
                    want,
                    "enc_calc_subpel_params(mv={mv:?}, pix=({pix_row},{pix_col}), \
                     ss=({ssx},{ssy}), ref={r:?}, this={t:?})"
                );

                if sp.pos_x == params.left || sp.pos_y == params.top {
                    clamped_lo += 1;
                }
                let bottom = (pre_h + 4) << 10;
                let right = (pre_w + 4) << 10;
                if sp.pos_x == right || sp.pos_y == bottom {
                    clamped_hi += 1;
                }
                if sf.is_scaled() {
                    scaled_cells += 1;
                }
                if off < 0 {
                    negative_offsets += 1;
                }
            }
        }
    }
    // Every arm the derivation has must have been reached, or the agreement
    // says less than it looks: the two clamps, the scaled path, and the
    // negative (into-the-border) source offset.
    assert!(clamped_lo > 0, "the top/left clamp never fired");
    assert!(clamped_hi > 0, "the bottom/right clamp never fired");
    assert!(scaled_cells > 0, "the scaled path was never taken");
    assert!(
        negative_offsets > 0,
        "no cell produced a negative source offset"
    );
}

/// `init_subpel_params` alone, driven at the exact clamp boundaries rather
/// than at random, so an off-by-one in either limit is visible.
#[test]
fn init_subpel_params_boundaries_match_c() {
    let (r, t) = ((352, 288), (352, 288));
    let sf = ScaleFactors::for_frame(r.0, r.1, t.0, t.1);
    let mut checked = 0usize;
    for (ssx, ssy) in [(0u32, 0u32), (1, 1)] {
        let (pre_w, pre_h) = (r.0 >> ssx, r.1 >> ssy);
        let params = InterBlockParams::new(0, 0, ssx, ssy);
        // `top`/`left` are in SCALE_SUBPEL units; an MV is in 1/8-pel luma,
        // scaled by `1 << (1 - ss)` into 1/16-pel plane units and then by
        // `1 << SCALE_EXTRA_BITS`. Walk the MV across the value that lands
        // exactly on the limit, and across the far edge as well.
        let per_mv = (1 << (1 - ssy)) * (1 << 6); // one MV step, in pos units
        let at_limit = params.top / per_mv;
        for d in -2..=2i32 {
            for &mv_row in &[at_limit + d, -(at_limit + d)] {
                if !(-(1 << 14)..(1 << 14)).contains(&mv_row) {
                    continue;
                }
                let mv = (mv_row as i16, 0i16);
                let got = init_subpel_params(mv, &params, &sf, pre_w, pre_h);
                let want = cref::ref_rie_enc_calc_subpel_params(
                    mv,
                    0,
                    0,
                    ssx as i32,
                    ssy as i32,
                    r,
                    t,
                    (pre_w, pre_h),
                    pre_w + 32,
                );
                assert_eq!(
                    [
                        got.xs,
                        got.ys,
                        got.subpel_x,
                        got.subpel_y,
                        got.pos_x,
                        got.pos_y
                    ],
                    want[..6],
                    "init_subpel_params at the clamp boundary (mv_row={mv_row}, ss=({ssx},{ssy}))"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 10,
        "the boundary walk covered only {checked} cells"
    );
}
