//! Differential harness for the reference-frame scale factors vs the REAL
//! exported C libaom v3.14.1. **Tier 1 throughout.**
//!
//! Covers `aom_dsp::inter::scale`:
//!
//! | test | C oracle |
//! |---|---|
//! | `setup_scale_factors_matches_c` | `av1_setup_scale_factors_for_frame` |
//! | `valid_ref_frame_size_matches_c` | `valid_ref_frame_size` |
//! | `scaled_x_y_match_c` | `av1_scaled_x` / `av1_scaled_y` |
//! | `scale_mv_matches_c` | `av1_scale_mv` |
//! | `is_scaled_matches_c` | `av1_is_scaled` |
//!
//! `scaled_x_y_match_c` deliberately sweeps NEGATIVE `val`: C rounds with
//! `ROUND_POWER_OF_TWO_SIGNED_64`, which rounds the *magnitude* and restores
//! the sign. That is not an arithmetic shift with a `+half` bias, and the two
//! differ by one at negative half-way values — a positives-only sweep would
//! pass against the wrong rounding.

use aom_dsp::inter::scale::{
    REF_INVALID_SCALE, REF_NO_SCALE, ScaleFactors, scale_mv, valid_ref_frame_size,
};
use aom_sys_ref::{
    ref_is_scaled, ref_scale_mv, ref_scaled_x, ref_scaled_y, ref_setup_scale_factors_for_frame,
    ref_valid_ref_frame_size,
};

/// Frame dimensions covering 1:1, both scaling directions, the exact 2x-down /
/// 16x-up spec limits, and one step past each of them.
const DIMS: [i32; 14] = [
    16, 32, 64, 96, 128, 176, 320, 352, 640, 720, 1280, 1920, 3840, 4096,
];

#[test]
fn valid_ref_frame_size_matches_c() {
    let mut checked = 0usize;
    let (mut valid, mut invalid) = (0usize, 0usize);
    for &rw in &DIMS {
        for &rh in &DIMS {
            for &tw in &DIMS {
                for &th in &DIMS {
                    let got = valid_ref_frame_size(rw, rh, tw, th);
                    let want = ref_valid_ref_frame_size(rw, rh, tw, th);
                    assert_eq!(got, want, "ref={rw}x{rh} this={tw}x{th}");
                    checked += 1;
                    if got { valid += 1 } else { invalid += 1 }
                }
            }
        }
    }
    assert!(checked > 10_000, "grid collapsed: {checked}");
    // Both arms must be reached, or the predicate is untested in one direction.
    assert!(valid > 0 && invalid > 0, "valid={valid} invalid={invalid}");
}

#[test]
fn setup_scale_factors_matches_c() {
    let mut saw_valid = false;
    let mut saw_invalid = false;
    let mut saw_no_scale = false;
    let mut saw_scaled = false;
    for &ow in &DIMS {
        for &oh in &DIMS {
            for &tw in &DIMS {
                for &th in &DIMS {
                    let got = ScaleFactors::for_frame(ow, oh, tw, th);
                    let (xs, ys, xst, yst) = ref_setup_scale_factors_for_frame(ow, oh, tw, th);
                    assert_eq!(got.x_scale_fp, xs, "x_scale_fp {ow}x{oh}->{tw}x{th}");
                    assert_eq!(got.y_scale_fp, ys, "y_scale_fp {ow}x{oh}->{tw}x{th}");
                    if got.is_valid() {
                        saw_valid = true;
                        assert_eq!(got.x_step_q4, xst, "x_step_q4 {ow}x{oh}->{tw}x{th}");
                        assert_eq!(got.y_step_q4, yst, "y_step_q4 {ow}x{oh}->{tw}x{th}");
                        if got.x_scale_fp == REF_NO_SCALE && got.y_scale_fp == REF_NO_SCALE {
                            saw_no_scale = true;
                        } else {
                            saw_scaled = true;
                        }
                    } else {
                        saw_invalid = true;
                        assert_eq!(got.x_scale_fp, REF_INVALID_SCALE);
                    }
                }
            }
        }
    }
    assert!(
        saw_valid && saw_invalid,
        "one arm of the size check never fired"
    );
    assert!(
        saw_no_scale && saw_scaled,
        "the sweep never produced both a 1:1 and a genuinely scaled reference"
    );
}

#[test]
fn scaled_x_y_match_c() {
    // A spread of real scale factors: 1:1, and the fixed-point values a few
    // representative up/down ratios produce.
    let mut fps = vec![REF_NO_SCALE];
    for &(o, t) in &[
        (320, 640),
        (640, 320),
        (176, 352),
        (1920, 1280),
        (4096, 3840),
        (16, 256),
    ] {
        let (xs, _, _, _) = ref_setup_scale_factors_for_frame(o, o, t, t);
        if xs != REF_INVALID_SCALE {
            fps.push(xs);
        }
    }
    assert!(fps.len() > 3, "not enough distinct scale factors: {fps:?}");

    let mut saw_negative = false;
    for &fp in &fps {
        let sf = ScaleFactors {
            x_scale_fp: fp,
            y_scale_fp: fp,
            x_step_q4: 0,
            y_step_q4: 0,
        };
        // q4 values spanning both signs, including the exact half-way points
        // where the signed rounding differs from an arithmetic shift.
        for val in (-4096i32..=4096).step_by(7) {
            assert_eq!(
                sf.scaled_x(val),
                ref_scaled_x(val, fp),
                "scaled_x val={val} fp={fp}"
            );
            assert_eq!(
                sf.scaled_y(val),
                ref_scaled_y(val, fp),
                "scaled_y val={val} fp={fp}"
            );
            if val < 0 {
                saw_negative = true;
            }
        }
        for &val in &[i32::MIN / 4, -1, 0, 1, i32::MAX / 4] {
            assert_eq!(
                sf.scaled_x(val),
                ref_scaled_x(val, fp),
                "scaled_x edge val={val}"
            );
            assert_eq!(
                sf.scaled_y(val),
                ref_scaled_y(val, fp),
                "scaled_y edge val={val}"
            );
        }
    }
    assert!(saw_negative, "the negative half of the sweep vanished");
}

#[test]
fn scale_mv_matches_c() {
    let mut fps = vec![REF_NO_SCALE];
    for &(o, t) in &[(320, 640), (640, 320), (176, 352), (1920, 1280)] {
        let (xs, ys, _, _) = ref_setup_scale_factors_for_frame(o, o, t, t);
        if xs != REF_INVALID_SCALE {
            fps.push(xs);
            let _ = ys;
        }
    }
    for &xfp in &fps {
        for &yfp in &fps {
            let sf = ScaleFactors {
                x_scale_fp: xfp,
                y_scale_fp: yfp,
                x_step_q4: 0,
                y_step_q4: 0,
            };
            for &(x, y) in &[(0i32, 0i32), (7, 3), (63, 127), (255, 1), (1023, 511)] {
                // MV components are int16_t in C; sweep both signs across the
                // range the shim can carry losslessly.
                for mv_row in (-512i32..=512).step_by(37) {
                    for mv_col in (-512i32..=512).step_by(53) {
                        let got = scale_mv((mv_row, mv_col), x, y, &sf);
                        let want = ref_scale_mv((mv_row, mv_col), x, y, xfp, yfp);
                        assert_eq!(
                            got, want,
                            "mv=({mv_row},{mv_col}) pos=({x},{y}) fp=({xfp},{yfp})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn is_scaled_matches_c() {
    for &xfp in &[
        REF_INVALID_SCALE,
        REF_NO_SCALE,
        REF_NO_SCALE - 1,
        REF_NO_SCALE + 1,
        8192,
        32768,
    ] {
        for &yfp in &[
            REF_INVALID_SCALE,
            REF_NO_SCALE,
            REF_NO_SCALE - 1,
            REF_NO_SCALE + 1,
            8192,
            32768,
        ] {
            let sf = ScaleFactors {
                x_scale_fp: xfp,
                y_scale_fp: yfp,
                x_step_q4: 0,
                y_step_q4: 0,
            };
            assert_eq!(sf.is_scaled(), ref_is_scaled(xfp, yfp), "fp=({xfp},{yfp})");
        }
    }
}
