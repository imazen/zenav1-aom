//! Differential harness for the INTER arm of `av1/encoder/var_based_part.c`
//! and the `aom_dsp/avg.c` kernels it stands on.
//!
//! | test | C oracle | tier |
//! |---|---|---|
//! | `avg_kernels_match_c` | `aom_avg_4x4_c`, `aom_avg_8x8_c`, `aom_avg_8x8_quad_c` + both highbd | 1 |
//! | `minmax_kernels_match_c` | `aom_minmax_8x8_c`, `aom_highbd_minmax_8x8_c` | 1 |
//! | `force_skip_low_temp_var_matches_c` | `av1_get_force_skip_low_temp_var` (:901) | 1 |
//! | `force_skip_low_temp_var_small_sb_matches_c` | `av1_get_force_skip_low_temp_var_small_sb` (:852) | 1 |
//! | `avg_8x8_quad_agrees_with_per_block_avg` | the two averaging paths of `fill_variance_8x8avg_lowbd` | 1 |
//!
//! | `vbp_static_shim_tu_matches_archive` | the TU's copies vs libaom.a's | 1 vs 1c |
//! | `fill_variance_8x8avg_matches_c` | `fill_variance_8x8avg` (:330) + both arms | 1c |
//! | `compute_minmax_8x8_matches_c` | `compute_minmax_8x8` (:349), both depths | 1c |
//! | `all_blks_inside_matches_c` | `all_blks_inside` (:255) | 1c |
//! | `scale_part_thresh_content_matches_c` | `scale_part_thresh_content` (:425) | 1c |
//! | `mv_distance_matches_c` | `mv_distance` (:1259) | 1c |
//! | `zeromv_skip_gate_matches_c` | `is_set_force_zeromv_skip_based_on_src_sad` (:1549) | 1c |
//!
//! The six file-statics have no exported address, so they are reached by
//! `shim/vbp_static_shim.c`, which compiles var_based_part.c verbatim -- NOT
//! by re-deriving the expected value here, which would only compare the port
//! against a second transcription of the same logic.
//! `vbp_static_shim_tu_matches_archive` is what makes that tier mean
//! something. `avg_8x8_quad_agrees_with_per_block_avg` is separate: it exists
//! because `fill_variance_8x8avg_lowbd` picks one of two averaging kernels on
//! `all_blks_inside`, and the port would be wrong-but-passing if the two ever
//! disagreed.
//!
//! # What bounds the generators
//! * `variance_low` is `PartitionSearchInfo::variance_low`, a 105-byte array
//!   of 0/1 flags. The sweep fills it with a DISTINCT value per slot instead,
//!   so a wrong index is visible; a 0/1 fill would let most index errors pass.
//! * `mi_row` / `mi_col` are mi units inside a superblock, so 0..32 for SB128
//!   and 0..16 for SB64, both stepped by the block's own mi size.

use aom_dsp::dist::avg::{
    avg_4x4, avg_8x8, avg_8x8_quad, highbd_avg_4x4, highbd_avg_8x8, highbd_minmax_8x8, minmax_8x8,
};
use aom_encode::var_part::{
    SourceSad, all_blks_inside, compute_minmax_8x8, fill_variance_8x8avg,
    get_force_skip_low_temp_var, get_force_skip_low_temp_var_small_sb,
    is_set_force_zeromv_skip_based_on_src_sad, mv_distance, scale_part_thresh_content,
};
use aom_encode::var_part::{compute_minmax_8x8_highbd, fill_variance_8x8avg_highbd};
use aom_sys_ref::{
    ref_vbp_avg_4x4, ref_vbp_avg_8x8, ref_vbp_avg_8x8_quad, ref_vbp_force_skip_low_temp_var,
    ref_vbp_force_skip_low_temp_var_small_sb, ref_vbp_highbd_avg_4x4, ref_vbp_highbd_avg_8x8,
    ref_vbp_highbd_minmax_8x8, ref_vbp_minmax_8x8, ref_vbps_all_blks_inside,
    ref_vbps_compute_minmax_8x8_highbd, ref_vbps_compute_minmax_8x8_lowbd,
    ref_vbps_fill_variance_8x8avg_highbd, ref_vbps_fill_variance_8x8avg_lowbd,
    ref_vbps_is_set_force_zeromv_skip, ref_vbps_mv_distance, ref_vbps_scale_part_thresh_content,
    ref_vbps_tu_force_skip_low_temp_var, ref_vbps_tu_force_skip_low_temp_var_small_sb,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// `BLOCK_SIZE` values this file names, in C's enum order (`enums.h:100`).
const BLOCK_16X16: usize = 6;
const BLOCK_16X32: usize = 7;
const BLOCK_32X16: usize = 8;
const BLOCK_32X32: usize = 9;
const BLOCK_32X64: usize = 10;
const BLOCK_64X32: usize = 11;
const BLOCK_64X64: usize = 12;
const BLOCK_64X128: usize = 13;
const BLOCK_128X64: usize = 14;
const BLOCK_128X128: usize = 15;

// ---------------------------------------------------------------------------
// aom_dsp/avg.c.
// ---------------------------------------------------------------------------

#[test]
fn avg_kernels_match_c() {
    let mut rng = Rng::new(0xA7_0000);
    for &stride in &[8usize, 16, 33, 64] {
        for trial in 0..40 {
            let rows = 40;
            let plane8: Vec<u8> = (0..rows * stride)
                .map(|i| match trial % 3 {
                    // A flat block pins the rounding term: (sum + 32) >> 6 with
                    // sum = 64 * v must come back exactly v, and a truncating
                    // port fails only on values that are not already rounded.
                    0 => 200,
                    1 => (i % 7) as u8 * 36,
                    _ => rng.below(256) as u8,
                })
                .collect();
            assert_eq!(
                avg_4x4(&plane8, stride),
                ref_vbp_avg_4x4(&plane8, stride),
                "avg_4x4 stride {stride} trial {trial}"
            );
            assert_eq!(
                avg_8x8(&plane8, stride),
                ref_vbp_avg_8x8(&plane8, stride),
                "avg_8x8 stride {stride} trial {trial}"
            );
            if stride >= 16 {
                for (x16, y16) in [(0usize, 0usize), (0, 16), (stride - 16, 0)] {
                    let got = avg_8x8_quad(&plane8, stride, x16, y16);
                    let want = ref_vbp_avg_8x8_quad(&plane8, stride, x16, y16);
                    for k in 0..4 {
                        assert_eq!(
                            got[k] as i32, want[k],
                            "avg_8x8_quad[{k}] stride {stride} at ({x16},{y16})"
                        );
                    }
                }
            }

            for &bd in &[10u32, 12] {
                let max = (1u32 << bd) - 1;
                let plane16: Vec<u16> = (0..rows * stride)
                    .map(|_| rng.below(max + 1) as u16)
                    .collect();
                assert_eq!(
                    highbd_avg_4x4(&plane16, stride),
                    ref_vbp_highbd_avg_4x4(&plane16, stride),
                    "highbd_avg_4x4 bd{bd} stride {stride}"
                );
                assert_eq!(
                    highbd_avg_8x8(&plane16, stride),
                    ref_vbp_highbd_avg_8x8(&plane16, stride),
                    "highbd_avg_8x8 bd{bd} stride {stride}"
                );
            }
        }
    }
}

#[test]
fn minmax_kernels_match_c() {
    let mut rng = Rng::new(0x11_44_77);
    let (mut saw_zero_spread, mut saw_max) = (false, false);
    for &stride in &[8usize, 24, 48] {
        for trial in 0..60 {
            let n = 16 * stride;
            let (s8, d8): (Vec<u8>, Vec<u8>) = match trial % 4 {
                // Identical buffers: every diff is 0, so min == max == 0 and
                // the seeded `min = 255` must be overwritten.
                0 => ((0..n).map(|_| rng.below(256) as u8).collect(), vec![0; n]),
                1 => {
                    let a: Vec<u8> = (0..n).map(|_| rng.below(256) as u8).collect();
                    let b = a.clone();
                    (a, b)
                }
                // The extremes: 0 vs 255 gives the largest possible diff.
                2 => (vec![255; n], vec![0; n]),
                _ => (
                    (0..n).map(|_| rng.below(256) as u8).collect(),
                    (0..n).map(|_| rng.below(256) as u8).collect(),
                ),
            };
            let got = minmax_8x8(&s8, stride, &d8, stride);
            let want = ref_vbp_minmax_8x8(&s8, stride, &d8, stride);
            assert_eq!(got, want, "minmax_8x8 stride {stride} trial {trial}");
            if got.0 == got.1 {
                saw_zero_spread = true;
            }
            if got.1 == 255 {
                saw_max = true;
            }

            for &bd in &[10u32, 12] {
                let max = (1u32 << bd) - 1;
                let s16: Vec<u16> = (0..n).map(|_| rng.below(max + 1) as u16).collect();
                let d16: Vec<u16> = if trial % 4 == 1 {
                    s16.clone()
                } else {
                    (0..n).map(|_| rng.below(max + 1) as u16).collect()
                };
                assert_eq!(
                    highbd_minmax_8x8(&s16, stride, &d16, stride),
                    ref_vbp_highbd_minmax_8x8(&s16, stride, &d16, stride),
                    "highbd_minmax_8x8 bd{bd} stride {stride} trial {trial}"
                );
            }
        }
    }
    assert!(
        saw_zero_spread,
        "the identical-buffer (min == max == 0) case never fired"
    );
    assert!(saw_max, "the full-range (max == 255) case never fired");

    // The highbd `min` SEED (65535, not the lowbd 255) is only observable when
    // EVERY diff in the 8x8 window exceeds 255 -- a uniform random draw over
    // [0, 1023] essentially never does, so the seed survives that sweep
    // untested. These windows are built so the smallest diff is 300.
    for &bd in &[10u32, 12] {
        let stride = 8usize;
        let base = 1u16 << (bd - 1);
        let s16: Vec<u16> = (0..64u16).map(|i| base + (i % 5)).collect();
        let d16: Vec<u16> = s16.iter().map(|&v| v - 300).collect();
        let got = highbd_minmax_8x8(&s16, stride, &d16, stride);
        let want = ref_vbp_highbd_minmax_8x8(&s16, stride, &d16, stride);
        assert_eq!(got, want, "highbd min seed, bd{bd}");
        assert!(
            got.0 > 255,
            "the construction failed: min diff {} is not above the lowbd seed",
            got.0
        );
    }
}

#[test]
fn avg_8x8_quad_agrees_with_per_block_avg() {
    // `fill_variance_8x8avg_lowbd` takes the QUAD kernel when every sub-block
    // is inside and the per-block kernel otherwise. The port branches the same
    // way, which is only safe while the two agree -- so that is measured here
    // against the C oracle rather than assumed.
    let mut rng = Rng::new(0x51D3_0091);
    let stride = 48usize;
    let plane: Vec<u8> = (0..stride * 48).map(|_| rng.below(256) as u8).collect();
    for y16 in (0..32).step_by(16) {
        for x16 in (0..32).step_by(16) {
            let quad = ref_vbp_avg_8x8_quad(&plane, stride, x16, y16);
            for k in 0..4 {
                let x8 = x16 + ((k & 1) << 3);
                let y8 = y16 + ((k >> 1) << 3);
                let per = ref_vbp_avg_8x8(&plane[y8 * stride + x8..], stride);
                assert_eq!(
                    quad[k], per as i32,
                    "quad[{k}] vs per-block at ({x16},{y16})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// var_based_part.c's two exported force-skip lookups.
// ---------------------------------------------------------------------------

/// `PartitionSearchInfo::variance_low` is 105 bytes. Filled with a distinct
/// value per slot so a wrong INDEX is visible; the real array holds 0/1 flags,
/// under which most index errors would silently pass.
fn distinct_variance_low() -> Vec<u8> {
    (0..105u16).map(|i| (i + 1) as u8).collect()
}

#[test]
fn force_skip_low_temp_var_small_sb_matches_c() {
    let variance_low = distinct_variance_low();
    let mut nonzero = 0usize;
    for &bsize in &[
        BLOCK_16X16,
        BLOCK_16X32,
        BLOCK_32X16,
        BLOCK_32X32,
        BLOCK_32X64,
        BLOCK_64X32,
        BLOCK_64X64,
        // Two sizes C's switch does NOT name, which must return 0.
        BLOCK_128X128,
        3,
    ] {
        for mi_row in 0..32i32 {
            for mi_col in 0..32i32 {
                let want = ref_vbp_force_skip_low_temp_var_small_sb(
                    &variance_low,
                    mi_row,
                    mi_col,
                    bsize as i32,
                );
                let got =
                    get_force_skip_low_temp_var_small_sb(&variance_low, mi_row, mi_col, bsize);
                assert_eq!(got, want, "bsize {bsize} at ({mi_row},{mi_col})");
                if want != 0 {
                    nonzero += 1;
                }
            }
        }
    }
    assert!(
        nonzero > 200,
        "only {nonzero} non-zero lookups - the sweep is near-vacuous"
    );
}

#[test]
fn force_skip_low_temp_var_matches_c() {
    let variance_low = distinct_variance_low();
    let mut nonzero = 0usize;
    // C asserts `(mi_col & 0x1F) == 0` for BLOCK_128X64 and
    // `(mi_row & 0x1F) == 0` for BLOCK_64X128, so those two are only swept at
    // the positions the encoder can produce. Every other size is swept fully.
    for mi_row in 0..32i32 {
        for mi_col in 0..32i32 {
            for &bsize in &[
                BLOCK_16X16,
                BLOCK_16X32,
                BLOCK_32X16,
                BLOCK_32X32,
                BLOCK_32X64,
                BLOCK_64X32,
                BLOCK_64X64,
                BLOCK_128X128,
                // Not in C's switch -> 0.
                3,
            ] {
                let want =
                    ref_vbp_force_skip_low_temp_var(&variance_low, mi_row, mi_col, bsize as i32);
                let got = get_force_skip_low_temp_var(&variance_low, mi_row, mi_col, bsize);
                assert_eq!(got, want, "bsize {bsize} at ({mi_row},{mi_col})");
                if want != 0 {
                    nonzero += 1;
                }
            }
            if mi_col & 0x1F == 0 {
                let want = ref_vbp_force_skip_low_temp_var(
                    &variance_low,
                    mi_row,
                    mi_col,
                    BLOCK_128X64 as i32,
                );
                assert_eq!(
                    get_force_skip_low_temp_var(&variance_low, mi_row, mi_col, BLOCK_128X64),
                    want
                );
            }
            if mi_row & 0x1F == 0 {
                let want = ref_vbp_force_skip_low_temp_var(
                    &variance_low,
                    mi_row,
                    mi_col,
                    BLOCK_64X128 as i32,
                );
                assert_eq!(
                    get_force_skip_low_temp_var(&variance_low, mi_row, mi_col, BLOCK_64X128),
                    want
                );
            }
        }
    }
    assert!(
        nonzero > 200,
        "only {nonzero} non-zero lookups - the sweep is near-vacuous"
    );
}

// ---------------------------------------------------------------------------
// The file-statics, assembled from the tier-1 kernels above.
// ---------------------------------------------------------------------------

#[test]
fn vbp_static_shim_tu_matches_archive() {
    // Without this, "the tier-1c TU behaves like libaom.a" is an assumption
    // and every 1c result below is measuring an unverified binary.
    let variance_low = distinct_variance_low();
    for mi_row in 0..32i32 {
        for mi_col in 0..32i32 {
            for &bsize in &[BLOCK_16X16, BLOCK_32X32, BLOCK_64X64, BLOCK_128X128] {
                assert_eq!(
                    ref_vbps_tu_force_skip_low_temp_var(
                        &variance_low,
                        mi_row,
                        mi_col,
                        bsize as i32
                    ),
                    ref_vbp_force_skip_low_temp_var(&variance_low, mi_row, mi_col, bsize as i32),
                    "TU vs archive, low_temp_var bsize {bsize} at ({mi_row},{mi_col})"
                );
                assert_eq!(
                    ref_vbps_tu_force_skip_low_temp_var_small_sb(
                        &variance_low,
                        mi_row,
                        mi_col,
                        bsize as i32
                    ),
                    ref_vbp_force_skip_low_temp_var_small_sb(
                        &variance_low,
                        mi_row,
                        mi_col,
                        bsize as i32
                    ),
                    "TU vs archive, small_sb bsize {bsize} at ({mi_row},{mi_col})"
                );
            }
        }
    }
}

#[test]
fn all_blks_inside_matches_c() {
    for pw in 0..40usize {
        for ph in 0..40usize {
            for y16 in (0..48).step_by(8) {
                for x16 in (0..48).step_by(8) {
                    assert_eq!(
                        all_blks_inside(x16, y16, pw, ph),
                        ref_vbps_all_blks_inside(x16, y16, pw, ph),
                        "({x16},{y16}) in {pw}x{ph}"
                    );
                }
            }
        }
    }
}

#[test]
fn fill_variance_8x8avg_matches_c() {
    let mut rng = Rng::new(0xF111_8A76);
    let stride = 64usize;
    let src8: Vec<u8> = (0..stride * 64).map(|_| rng.below(256) as u8).collect();
    let dst8: Vec<u8> = (0..stride * 64).map(|_| rng.below(256) as u8).collect();
    let (mut inside, mut clipped) = (0usize, 0usize);

    for &(pw, ph) in &[(64usize, 64usize), (40, 64), (64, 40), (24, 20), (8, 8)] {
        for y16 in (0..48).step_by(16) {
            for x16 in (0..48).step_by(16) {
                let got = fill_variance_8x8avg(&src8, stride, &dst8, stride, x16, y16, pw, ph);
                let want = ref_vbps_fill_variance_8x8avg_lowbd(
                    &src8, stride, &dst8, stride, x16, y16, pw, ph,
                );
                assert_eq!(got, want, "lowbd at ({x16},{y16}) in {pw}x{ph}");
                if all_blks_inside(x16, y16, pw, ph) {
                    inside += 1;
                } else {
                    clipped += 1;
                }
            }
        }
    }
    // Both averaging paths of fill_variance_8x8avg_lowbd must have run.
    assert!(
        inside > 0 && clipped > 0,
        "one path never fired: {inside}/{clipped}"
    );

    // The highbd arm, which has no quad fast path at all.
    for &bd in &[10u32, 12] {
        let max = (1u32 << bd) - 1;
        let src16: Vec<u16> = (0..stride * 64)
            .map(|_| rng.below(max + 1) as u16)
            .collect();
        let dst16: Vec<u16> = (0..stride * 64)
            .map(|_| rng.below(max + 1) as u16)
            .collect();
        for &(pw, ph) in &[(64usize, 64usize), (40, 64), (24, 20)] {
            for y16 in (0..48).step_by(16) {
                for x16 in (0..48).step_by(16) {
                    let got = fill_variance_8x8avg_highbd(
                        &src16, stride, &dst16, stride, x16, y16, pw, ph,
                    );
                    let want = ref_vbps_fill_variance_8x8avg_highbd(
                        &src16, stride, &dst16, stride, x16, y16, pw, ph,
                    );
                    assert_eq!(got, want, "highbd bd{bd} at ({x16},{y16}) in {pw}x{ph}");
                }
            }
        }
    }
}

#[test]
fn compute_minmax_8x8_matches_c() {
    let mut rng = Rng::new(0x3C_1234);
    let stride = 64usize;
    let src8: Vec<u8> = (0..stride * 64).map(|_| rng.below(256) as u8).collect();
    let dst8: Vec<u8> = (0..stride * 64).map(|_| rng.below(256) as u8).collect();
    let mut saw_negative = false;

    for &(pw, ph) in &[(64usize, 64usize), (40, 64), (24, 20), (1, 1)] {
        for y16 in (0..48).step_by(16) {
            for x16 in (0..48).step_by(16) {
                let got = compute_minmax_8x8(&src8, stride, &dst8, stride, x16, y16, pw, ph);
                let want = ref_vbps_compute_minmax_8x8_lowbd(
                    &src8, stride, &dst8, stride, x16, y16, pw, ph,
                );
                assert_eq!(got, want, "lowbd at ({x16},{y16}) in {pw}x{ph}");
                if want < 0 {
                    saw_negative = true;
                }
            }
        }
    }
    // C never resets minmax_max = 0 / minmax_min = 255, so a 16x16 with no
    // in-frame sub-block returns -255. If this stops firing the sweep has lost
    // its fully-clipped cells.
    assert!(
        saw_negative,
        "the no-in-frame-sub-block case (returning -255) never fired"
    );

    for &bd in &[10u32, 12] {
        let max = (1u32 << bd) - 1;
        let src16: Vec<u16> = (0..stride * 64)
            .map(|_| rng.below(max + 1) as u16)
            .collect();
        let dst16: Vec<u16> = (0..stride * 64)
            .map(|_| rng.below(max + 1) as u16)
            .collect();
        for &(pw, ph) in &[(64usize, 64usize), (24, 20)] {
            for y16 in (0..48).step_by(16) {
                for x16 in (0..48).step_by(16) {
                    assert_eq!(
                        compute_minmax_8x8_highbd(&src16, stride, &dst16, stride, x16, y16, pw, ph),
                        ref_vbps_compute_minmax_8x8_highbd(
                            &src16, stride, &dst16, stride, x16, y16, pw, ph
                        ),
                        "highbd bd{bd} at ({x16},{y16}) in {pw}x{ph}"
                    );
                }
            }
        }
    }
}

#[test]
fn scale_part_thresh_content_matches_c() {
    for &base in &[0i64, 1, 7, 1000, 1 << 40] {
        for speed in 0..12i32 {
            for &non_ref in &[false, true] {
                for &is_static in &[false, true] {
                    assert_eq!(
                        scale_part_thresh_content(base, speed, non_ref, is_static),
                        ref_vbps_scale_part_thresh_content(base, speed, non_ref, is_static),
                        "base {base} speed {speed} non_ref {non_ref} static {is_static}"
                    );
                }
            }
        }
    }
}

#[test]
fn mv_distance_matches_c() {
    let mut rng = Rng::new(0xD157_0000);
    // The extremes are what force the i32 widening: C's int16_t operands
    // promote to int before the subtraction, so a same-width port overflows.
    let mut cases = vec![
        ((0i16, 0i16), (0i16, 0i16)),
        ((5, -7), (5, -7)),
        ((i16::MIN, i16::MAX), (i16::MAX, i16::MIN)),
        ((i16::MAX, i16::MIN), (i16::MIN, i16::MAX)),
    ];
    for _ in 0..400 {
        let mut d = || rng.below(u32::from(u16::MAX) + 1) as u16 as i16;
        cases.push(((d(), d()), (d(), d())));
    }
    for &(a, b) in &cases {
        assert_eq!(
            mv_distance(a, b),
            ref_vbps_mv_distance(a, b),
            "{a:?} vs {b:?}"
        );
    }
    assert_eq!(
        ref_vbps_mv_distance((i16::MIN, i16::MAX), (i16::MAX, i16::MIN)),
        65535 + 65535,
        "the widened result is what C produces"
    );
}

#[test]
fn zeromv_skip_gate_matches_c() {
    use SourceSad::{High, Low, Med, VeryLow, Zero};
    for level in -2..=6i32 {
        for (raw, sad) in [(0i32, Zero), (1, VeryLow), (2, Low), (3, Med), (4, High)] {
            assert_eq!(
                is_set_force_zeromv_skip_based_on_src_sad(level, sad),
                ref_vbps_is_set_force_zeromv_skip(level, raw),
                "level {level} sad {sad:?}"
            );
        }
    }
}
