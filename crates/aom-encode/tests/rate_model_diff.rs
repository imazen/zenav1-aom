//! Differential harness for the rate-control rate model vs the REAL exported C
//! libaom v3.14.1. **Tier 1.**
//!
//! | test | C oracle |
//! |---|---|
//! | `convert_qindex_to_q_matches_c` | `av1_convert_qindex_to_q` |
//! | `find_qindex_matches_c` | `av1_find_qindex` |
//! | `compute_qdelta_matches_c` | `av1_compute_qdelta` |
//! | `rc_bits_per_mb_matches_c` | `av1_rc_bits_per_mb` (non-CBR arm) |
//! | `rc_get_default_min_gf_interval_matches_c` | `av1_rc_get_default_min_gf_interval` |
//!
//! The C oracle for `av1_rc_bits_per_mb` is driven with
//! `oxcf.rc_cfg.mode = AOM_Q`, which is the arm the port implements. Leaving
//! the mode zeroed would select `AOM_VBR`; that happens to take the same branch
//! today, but pinning an arm you did not choose is exactly how a port ends up
//! matching the wrong code.

use aom_encode::rate_model::{
    bpmb_enumerator, compute_qdelta, convert_qindex_to_q, find_qindex, rc_bits_per_mb,
    rc_get_default_min_gf_interval,
};
use aom_sys_ref::{
    ref_compute_qdelta, ref_convert_qindex_to_q, ref_find_qindex, ref_rc_bits_per_mb,
    ref_rc_get_default_min_gf_interval,
};

#[test]
fn convert_qindex_to_q_matches_c() {
    for &bd in &[8u8, 10, 12] {
        for qindex in 0..256i32 {
            let got = convert_qindex_to_q(qindex, bd);
            let want = ref_convert_qindex_to_q(qindex, bd);
            assert_eq!(got.to_bits(), want.to_bits(), "bd={bd} qindex={qindex}");
        }
    }
}

#[test]
fn convert_qindex_to_q_bit_depths_differ() {
    // The three depths divide the same AC quantizer by 4 / 16 / 64. If they
    // ever agreed, the bd sweeps here and in `find_qindex` would be one test
    // repeated and a bd-8-only formula would pass.
    for qindex in [0i32, 64, 128, 255] {
        let a = convert_qindex_to_q(qindex, 8);
        let b = convert_qindex_to_q(qindex, 10);
        let c = convert_qindex_to_q(qindex, 12);
        assert!(a != b || b != c, "qindex={qindex}: a={a} b={b} c={c}");
    }
}

#[test]
fn find_qindex_matches_c() {
    // The largest Q the table can produce is a few hundred at bd 8 and grows
    // with the bit depth, so the desired-Q sweep has to run well past it for
    // the "no index reaches it, clamp to worst" outcome to occur at all.
    let mut saw_clamped = false;
    let mut saw_interior = false;
    let mut saw_best = false;
    for &bd in &[8u8, 10, 12] {
        for &(best, worst) in &[(0i32, 255i32), (0, 63), (100, 200), (255, 255), (7, 8)] {
            for step in 0..300 {
                let desired = -10.0 + f64::from(step) * 12.5;
                let got = find_qindex(desired, bd, best, worst);
                let want = ref_find_qindex(desired, bd, best, worst);
                assert_eq!(
                    got, want,
                    "bd={bd} range=({best},{worst}) desired={desired}"
                );
                if best != worst {
                    if got == worst {
                        saw_clamped = true;
                    } else if got == best {
                        saw_best = true;
                    } else {
                        saw_interior = true;
                    }
                }
            }
        }
    }
    // All three outcomes must occur somewhere, or the binary search is only
    // being asked questions with one kind of answer.
    assert!(saw_best, "the search never returned the best-quality end");
    assert!(saw_interior, "the search never returned an interior qindex");
    assert!(
        saw_clamped,
        "the search never clamped to the worst-quality end"
    );
}

#[test]
fn compute_qdelta_matches_c() {
    for &bd in &[8u8, 10, 12] {
        for &(best, worst) in &[(0i32, 255i32), (20, 200)] {
            for si in 0..25 {
                for ti in 0..25 {
                    let qstart = 0.5 + f64::from(si) * 4.3;
                    let qtarget = 0.5 + f64::from(ti) * 4.3;
                    let got = compute_qdelta(qstart, qtarget, bd, best, worst);
                    let want = ref_compute_qdelta(qstart, qtarget, bd, best, worst);
                    assert_eq!(
                        got, want,
                        "bd={bd} range=({best},{worst}) qstart={qstart} qtarget={qtarget}"
                    );
                }
            }
        }
    }
}

#[test]
fn rc_bits_per_mb_matches_c() {
    // MIN_BPB_FACTOR / MAX_BPB_FACTOR bracket the correction factor C asserts
    // on; the sweep stays inside them.
    for &bd in &[8u8, 10, 12] {
        for &is_key in &[true, false] {
            for &is_screen in &[true, false] {
                for &cf in &[0.005f64, 0.25, 1.0, 4.0, 16.0] {
                    // qindex 0 gives an AC quantizer of 4 at bd 8, so q is
                    // never 0 and the divide is always defined.
                    for qindex in (0..256i32).step_by(3) {
                        let got = rc_bits_per_mb(is_key, is_screen, qindex, cf, bd);
                        let want = ref_rc_bits_per_mb(is_key, is_screen, qindex, cf, bd);
                        assert_eq!(
                            got, want,
                            "bd={bd} key={is_key} screen={is_screen} cf={cf} q={qindex}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bpmb_enumerator_has_four_distinct_values() {
    // The enumerator is the only thing frame type and screen-content
    // classification change in the non-CBR arm. If any two of the four
    // combinations collapsed, `rc_bits_per_mb_matches_c`'s outer loops would be
    // duplicates.
    let vals = [
        bpmb_enumerator(true, true),
        bpmb_enumerator(false, true),
        bpmb_enumerator(true, false),
        bpmb_enumerator(false, false),
    ];
    for a in 0..vals.len() {
        for b in (a + 1)..vals.len() {
            assert_ne!(vals[a], vals[b], "enumerators {a} and {b} collapsed");
        }
    }
}

#[test]
fn rc_get_default_min_gf_interval_matches_c() {
    let mut saw_scaled = false;
    let mut saw_default = false;
    // Resolutions either side of the "4K at 20 fps" pixel-rate threshold.
    for &(w, h) in &[
        (176i32, 144i32),
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
        (7680, 4320),
    ] {
        for &fps in &[1.0f64, 15.0, 23.976, 24.0, 30.0, 47.952, 60.0, 120.0, 240.0] {
            let got = rc_get_default_min_gf_interval(w, h, fps);
            let want = ref_rc_get_default_min_gf_interval(w, h, fps);
            assert_eq!(got, want, "{w}x{h} @ {fps}");
            if f64::from(w) * f64::from(h) * fps > 3840.0 * 2160.0 * 20.0 {
                saw_scaled = true;
            } else {
                saw_default = true;
            }
        }
    }
    assert!(
        saw_scaled && saw_default,
        "only one side of the pixel-rate threshold was reached"
    );
}
