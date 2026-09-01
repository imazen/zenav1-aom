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
    /// A value in `[lo, hi]`.
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32) as i32
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

// ---------------------------------------------------------------------------
// The low-temporal-variance flag setters, and the round trip through the two
// force-skip lookups they feed.
// ---------------------------------------------------------------------------

use aom_encode::var_part::{
    MiGrid, VARIANCE_LOW_LEN, VarianceTree, set_low_temp_var_flag, set_low_temp_var_flag_64x64,
    set_low_temp_var_flag_128x128,
};
use aom_sys_ref::{ref_vbps_set_low_temp_var_flag, ref_vbps_variance_low_len};

/// The 105 tree values in the boundary's layout (see `shim/vbp_static_shim.c`).
fn tree_to_flat(vt: &VarianceTree) -> [i32; 105] {
    let mut f = [0i32; 105];
    f[0..5].copy_from_slice(&vt.l0);
    for a in 0..4 {
        f[5 + a * 5..5 + a * 5 + 5].copy_from_slice(&vt.l1[a]);
    }
    f[25..41].copy_from_slice(&vt.l2);
    f[41..105].copy_from_slice(&vt.l3);
    f
}

/// Block sizes the RT partitioner can stamp into the mi grid, plus sizes the
/// setters do NOT name (which must leave the flags alone).
const GRID_BSIZES: [i32; 9] = [
    BLOCK_16X16 as i32,
    BLOCK_16X32 as i32,
    BLOCK_32X16 as i32,
    BLOCK_32X32 as i32,
    BLOCK_32X64 as i32,
    BLOCK_64X32 as i32,
    BLOCK_64X64 as i32,
    BLOCK_128X128 as i32,
    3, // BLOCK_8X8 -- named by neither setter
];

#[test]
fn variance_low_length_matches_c() {
    assert_eq!(VARIANCE_LOW_LEN, ref_vbps_variance_low_len());
}

#[test]
fn set_low_temp_var_flag_matches_c() {
    let mut rng = Rng::new(0x107E_4A11);
    let mi_stride = 96usize;
    let grid_rows = 96usize;
    let grid_len = mi_stride * grid_rows;
    let mut set_any = 0usize;
    let mut set_none = 0usize;

    for &is_small_sb in &[true, false] {
        for _ in 0..400 {
            // The tree values straddle the shifted thresholds: `>> 8` makes the
            // 16x16 threshold tiny, so a wide uniform draw would never fire it.
            let mut vt = VarianceTree::default();
            let mut draw = |rng: &mut Rng| match rng.below(4) {
                0 => rng.below(4) as i32,
                1 => rng.below(64) as i32,
                2 => rng.below(1 << 12) as i32,
                _ => rng.below(1 << 20) as i32,
            };
            for v in vt.l0.iter_mut() {
                *v = draw(&mut rng);
            }
            for row in vt.l1.iter_mut() {
                for v in row.iter_mut() {
                    *v = draw(&mut rng);
                }
            }
            for v in vt.l2.iter_mut() {
                *v = draw(&mut rng);
            }
            for v in vt.l3.iter_mut() {
                *v = draw(&mut rng);
            }

            // Thresholds in the shape set_vbp_thresholds produces: a base and
            // its divisions. `>> 8` on thresholds[2]/[3] is why the base has
            // to be large for the 16x16 arm to ever fire.
            // All five must be DISTINCT: the setters read [0], [1], [2] and
            // [3] at different levels, and a table with repeats hides a
            // wrong-index port (thresholds[2] vs [3] in the SB128 16x16 arm was
            // exactly that -- it survived until these stopped coinciding).
            let base = 1i64 << (10 + rng.below(12));
            let thresholds = [base, base * 3, base * 7, base * 13, base * 21];

            // A grid whose cells are mostly present but sometimes NULL, since
            // both setters check for that before dereferencing.
            let mi_bsize_i32: Vec<i32> = (0..grid_len)
                .map(|_| {
                    if rng.below(8) == 0 {
                        -1
                    } else {
                        GRID_BSIZES[rng.below(GRID_BSIZES.len() as u32) as usize]
                    }
                })
                .collect();
            let mi_bsize: Vec<Option<usize>> = mi_bsize_i32
                .iter()
                .map(|&b| if b < 0 { None } else { Some(b as usize) })
                .collect();

            // mi_rows / mi_cols sometimes cut through the superblock, which is
            // what makes the bounds `continue`s reachable.
            let mi_row = (rng.below(4) * 8) as usize;
            let mi_col = (rng.below(4) * 8) as usize;
            let mi_rows = mi_row + 1 + rng.below(40) as usize;
            let mi_cols = mi_col + 1 + rng.below(40) as usize;

            let grid = MiGrid {
                bsize: &mi_bsize,
                mi_stride,
                mi_rows,
                mi_cols,
            };

            for &cur_bsize in &GRID_BSIZES {
                // Both the LAST_FRAME arm and a reference that must leave the
                // flags untouched.
                for &ref_frame in &[1i32, 4] {
                    let mut want = vec![0u8; VARIANCE_LOW_LEN];
                    ref_vbps_set_low_temp_var_flag(
                        is_small_sb,
                        ref_frame,
                        cur_bsize,
                        &tree_to_flat(&vt),
                        &mi_bsize_i32,
                        mi_stride,
                        mi_rows,
                        mi_cols,
                        mi_row,
                        mi_col,
                        &thresholds,
                        &mut want,
                    );

                    // C's caller zeroes variance_low per superblock
                    // (var_based_part.c:1702), so the port starts from zeros too.
                    let mut got = [0u8; VARIANCE_LOW_LEN];
                    set_low_temp_var_flag(
                        &grid,
                        &mut got,
                        cur_bsize as usize,
                        &vt,
                        &thresholds,
                        ref_frame as usize,
                        mi_col,
                        mi_row,
                        is_small_sb,
                    );
                    assert_eq!(
                        &got[..],
                        &want[..],
                        "small_sb {is_small_sb} ref {ref_frame} bsize {cur_bsize} at ({mi_row},{mi_col}) mi {mi_rows}x{mi_cols}"
                    );
                    if got.iter().any(|&f| f != 0) {
                        set_any += 1;
                    } else {
                        set_none += 1;
                    }
                }
            }
        }
    }
    assert!(
        set_any > 500 && set_none > 500,
        "one arm never fired: {set_any} set / {set_none} clear"
    );
}

#[test]
fn set_low_temp_var_flag_dispatches_on_sb_size() {
    // The dispatcher's only job is to pick between the two setters and to gate
    // on LAST_FRAME; this checks it does that rather than duplicating either.
    let mi_bsize = vec![Some(BLOCK_32X32); 96 * 96];
    let grid = MiGrid {
        bsize: &mi_bsize,
        mi_stride: 96,
        mi_rows: 96,
        mi_cols: 96,
    };
    let mut vt = VarianceTree::default();
    // Every node below every threshold, so both setters would set something.
    vt.l0 = [0; 5];
    vt.l1 = [[0; 5]; 4];
    let thresholds = [1i64 << 20; 5];

    for &is_small_sb in &[true, false] {
        let mut direct = [0u8; VARIANCE_LOW_LEN];
        if is_small_sb {
            set_low_temp_var_flag_64x64(&grid, &mut direct, BLOCK_64X64, &vt, &thresholds, 0, 0);
        } else {
            set_low_temp_var_flag_128x128(
                &grid,
                &mut direct,
                BLOCK_128X128,
                &vt,
                &thresholds,
                0,
                0,
            );
        }
        let mut via = [0u8; VARIANCE_LOW_LEN];
        set_low_temp_var_flag(
            &grid,
            &mut via,
            if is_small_sb {
                BLOCK_64X64
            } else {
                BLOCK_128X128
            },
            &vt,
            &thresholds,
            1,
            0,
            0,
            is_small_sb,
        );
        assert_eq!(
            direct, via,
            "the dispatcher must be a pure pick, small_sb {is_small_sb}"
        );
        assert!(via.iter().any(|&f| f != 0), "the construction set nothing");

        // A non-LAST reference leaves it untouched.
        let mut none = [0u8; VARIANCE_LOW_LEN];
        set_low_temp_var_flag(
            &grid,
            &mut none,
            if is_small_sb {
                BLOCK_64X64
            } else {
                BLOCK_128X128
            },
            &vt,
            &thresholds,
            4,
            0,
            0,
            is_small_sb,
        );
        assert_eq!(
            none, [0u8; VARIANCE_LOW_LEN],
            "GOLDEN_FRAME must set nothing"
        );
    }
}

#[test]
fn low_temp_var_flags_round_trip_through_the_force_skip_lookups() {
    // The setters and the lookups are two halves of one mechanism: nothing
    // else writes variance_low and nothing else reads it. This drives the pair
    // the way the encoder does -- set the flags for a superblock, then ask for
    // each block inside it -- so a slot-numbering disagreement between the two
    // halves shows up as a lookup that can never be 1.
    let mi_bsize = vec![Some(BLOCK_32X32); 96 * 96];
    let grid = MiGrid {
        bsize: &mi_bsize,
        mi_stride: 96,
        mi_rows: 96,
        mi_cols: 96,
    };
    let vt = VarianceTree::default(); // every variance 0, i.e. below anything
    let thresholds = [1i64 << 24; 5];

    let mut flags = [0u8; VARIANCE_LOW_LEN];
    set_low_temp_var_flag_64x64(&grid, &mut flags, BLOCK_32X32, &vt, &thresholds, 0, 0);
    // A 64x64 superblock of 32x32 leaves sets slots 5..9.
    assert_eq!(
        &flags[5..9],
        &[1, 1, 1, 1],
        "the 32x32 leaf slots were not set"
    );
    let mut reachable = 0usize;
    for mi_row in (0..16i32).step_by(8) {
        for mi_col in (0..16i32).step_by(8) {
            if get_force_skip_low_temp_var_small_sb(&flags, mi_row, mi_col, BLOCK_32X32) != 0 {
                reachable += 1;
            }
        }
    }
    assert_eq!(
        reachable, 4,
        "the SB64 setter and lookup disagree about the 32x32 slot numbering"
    );
}

// ---------------------------------------------------------------------------
// The two remaining INTER decisions. Both C functions also do buffer plumbing
// the port replaces rather than translates; the shim runs it and reports only
// the decision outputs, which is what these tests compare.
// ---------------------------------------------------------------------------

use aom_encode::var_part::{
    calc_chroma_thresh_for_zeromv_skip, set_force_zeromv_skip_for_sb, set_ref_frame_for_partition,
};
use aom_sys_ref::{ref_vbps_set_force_zeromv_skip_for_sb, ref_vbps_set_ref_frame_for_partition};

#[test]
fn set_force_zeromv_skip_for_sb_matches_c() {
    let mut rng = Rng::new(0x2E40_4001);
    let (mut exits, mut twos, mut zeros) = (0usize, 0usize, 0usize);
    // BLOCK_64X64 is 16 mi wide, BLOCK_128X128 is 32.
    for &(sb_size, sb_mi) in &[(BLOCK_64X64 as i32, 16i32), (BLOCK_128X128 as i32, 32)] {
        for _ in 0..3000 {
            let level = rng.below(5) as i32;
            let sad_raw = rng.below(5) as i32;
            let sad = [
                SourceSad::Zero,
                SourceSad::VeryLow,
                SourceSad::Low,
                SourceSad::Med,
                SourceSad::High,
            ][sad_raw as usize];
            let increase = rng.below(2) == 0;
            let early_exit = rng.below(4) as i32;
            // A real zeromv_skip_thresh_exit_part value, and SADs that
            // straddle it and its three-quarter chroma sibling.
            let thresh = 1u32 << (6 + rng.below(12));
            let draw_sad = |rng: &mut Rng| match rng.below(4) {
                0 => 0u32,
                1 => thresh / 2,
                2 => thresh,
                _ => thresh + rng.below(thresh.max(1)),
            };
            let y_sad = draw_sad(&mut rng);
            let uv_sad = [draw_sad(&mut rng), draw_sad(&mut rng)];
            let (mi_row, mi_col) = (
                (rng.below(3) * sb_mi as u32) as i32,
                (rng.below(3) * sb_mi as u32) as i32,
            );
            // Tile ends that sometimes cut the superblock off.
            let tile_end = (
                mi_row + sb_mi - rng.below(2) as i32 * (sb_mi / 2),
                mi_col + sb_mi - rng.below(2) as i32 * (sb_mi / 2),
            );
            let mi_dims = (128i32, 128i32, 128i32);
            let bsize = sb_size;

            let want = ref_vbps_set_force_zeromv_skip_for_sb(
                level, sad_raw, increase, early_exit, sb_size, bsize, thresh, mi_row, mi_col,
                tile_end, y_sad, uv_sad, mi_dims,
            );
            let got = set_force_zeromv_skip_for_sb(
                level, sad, increase, early_exit, sb_mi, sb_mi, thresh, mi_row, mi_col, tile_end.0,
                tile_end.1, y_sad, uv_sad,
            );
            assert_eq!(
                (got.exit_partitioning, got.force_zeromv_skip_for_sb),
                want,
                "level {level} sad {sad_raw} inc {increase} exit {early_exit} thresh {thresh} \
                 y {y_sad} uv {uv_sad:?} at ({mi_row},{mi_col}) tile {tile_end:?}"
            );
            match got.force_zeromv_skip_for_sb {
                1 => exits += 1,
                2 => twos += 1,
                _ => zeros += 1,
            }
        }
    }
    // All three outcomes are separate code paths.
    assert!(exits > 100, "the exit path fired only {exits} times");
    assert!(
        twos > 50,
        "the force_zeromv_skip_for_sb = 2 arm fired only {twos} times"
    );
    assert!(zeros > 100, "the no-op path fired only {zeros} times");
}

#[test]
fn chroma_zeromv_threshold_is_three_quarters() {
    // A one-line macro, pinned because the `>> 2` is easy to mistake for a
    // `/ 3` and the result feeds two comparisons.
    for t in [0u32, 1, 3, 4, 100, 1 << 20] {
        assert_eq!(calc_chroma_thresh_for_zeromv_skip(t), (3 * t) >> 2, "t {t}");
    }
}

#[test]
fn set_ref_frame_for_partition_matches_c() {
    let mut rng = Rng::new(0x4EF4_4002);
    let mut picked = [0usize; 3];
    for _ in 0..8000 {
        let spatial_layer_id = rng.below(3) as i32;
        let has_lower = rng.below(2) == 0;
        // The three SADs are drawn from one band so the 0.9 / 1.0 margin
        // actually decides; independent wide draws make the comparison
        // trivial almost every time.
        let base = 1 + rng.below(1 << 20);
        let draw = |rng: &mut Rng| -> u32 {
            let span = base / 4 + 1;
            base.saturating_sub(span / 2) + rng.below(span)
        };
        let y_sad = draw(&mut rng);
        let y_sad_g = draw(&mut rng);
        let y_sad_alt = draw(&mut rng);
        let prune_cfg = rng.below(4) as i32;

        let want = ref_vbps_set_ref_frame_for_partition(
            spatial_layer_id,
            has_lower,
            y_sad,
            y_sad_g,
            y_sad_alt,
            prune_cfg,
        );
        let got = set_ref_frame_for_partition(
            spatial_layer_id,
            has_lower,
            y_sad,
            y_sad_g,
            y_sad_alt,
            prune_cfg,
        );
        assert_eq!(
            got.ref_frame_partition, want.0,
            "ref_frame, sads {y_sad}/{y_sad_g}/{y_sad_alt}"
        );
        assert_eq!(
            got.y_sad, want.1,
            "y_sad, sads {y_sad}/{y_sad_g}/{y_sad_alt}"
        );
        assert_eq!(
            got.nonrd_prune_ref_frame_search, want.2,
            "prune, sads {y_sad}/{y_sad_g}/{y_sad_alt}"
        );
        // The shim seeds sb_me_partition at -1; C's LAST arm leaves it alone.
        assert_eq!(
            got.sb_me_partition.unwrap_or(-1),
            want.3,
            "sb_me, sads {y_sad}/{y_sad_g}/{y_sad_alt}"
        );
        match got.ref_frame_partition {
            4 => picked[0] += 1,
            7 => picked[1] += 1,
            _ => picked[2] += 1,
        }
    }
    assert!(
        picked[0] > 100 && picked[1] > 100 && picked[2] > 100,
        "one reference was never picked: {picked:?}"
    );
}

// ---------------------------------------------------------------------------
// evaluate_neighbour_mvs — the last in-scope INTER function of
// var_based_part.c.
// ---------------------------------------------------------------------------

use aom_encode::var_part::{
    FullMvLimits, INTRA_MODE_END, NeighbourMv, evaluate_neighbour_mvs, set_subpel_mv_search_range,
};
use aom_sys_ref::{RefNeighbourMvCtx, ref_vbps_evaluate_neighbour_mvs, ref_vbps_intra_mode_end};

#[test]
fn intra_mode_end_matches_c() {
    assert_eq!(INTRA_MODE_END, ref_vbps_intra_mode_end());
}

#[test]
fn evaluate_neighbour_mvs_matches_c() {
    let mut rng = Rng::new(0x4E16_5001);
    // A margin large enough for any MV the limits below can produce, so
    // get_buf_from_fullmv always lands inside the reference plane.
    const MARGIN: i32 = 40;
    let (mut took_above, mut took_left, mut took_neither) = (0usize, 0usize, 0usize);
    let mut early_returns = 0usize;

    for &(is_small_sb, block_dim) in &[(true, 64i32), (false, 128i32)] {
        let bd = block_dim as usize;
        let src_stride = bd + 32;
        let ref_h = bd + 2 * MARGIN as usize;
        let ref_stride = bd + 2 * MARGIN as usize + 32;
        for _ in 0..220 {
            // The reference is the source shifted by a small offset, so a
            // neighbour MV near that offset genuinely wins -- an independently
            // drawn reference makes every SAD equally bad and the acceptance
            // test never fires.
            let base: Vec<u8> = (0..ref_stride * ref_h)
                .map(|_| rng.below(256) as u8)
                .collect();
            let shift_r = rng.range(-8, 8);
            let shift_c = rng.range(-8, 8);
            let mut src = vec![0u8; src_stride * bd];
            for y in 0..bd {
                for x in 0..bd {
                    let ry = (y as i32 + MARGIN + shift_r) as usize;
                    let rx = (x as i32 + MARGIN + shift_c) as usize;
                    src[y * src_stride + x] = base[ry * ref_stride + rx];
                }
            }

            // 1/8-pel MVs whose full-pel forms land in +/-16. A third of the
            // draws land EXACTLY on the reference's own shift, because that is
            // the only way a neighbour candidate's SAD is small enough to beat
            // the block's -- a uniform draw over a 33x33 full-pel grid hits it
            // about a tenth of a percent of the time and the acceptance arm
            // never fires.
            let draw_mv = |rng: &mut Rng| -> (i16, i16) {
                if rng.below(3) == 0 {
                    ((shift_r * 8) as i16, (shift_c * 8) as i16)
                } else {
                    (rng.range(-128, 128) as i16, rng.range(-128, 128) as i16)
                }
            };
            // Neighbours that are absent, intra, inter-from-GOLDEN and
            // inter-from-LAST -- only the last is used.
            let draw_nb = |rng: &mut Rng| -> (bool, i32, i32, (i16, i16)) {
                match rng.below(4) {
                    0 => (false, 0, 0, (0, 0)),
                    1 => (true, 0, 1, draw_mv(rng)), // DC_PRED: intra
                    2 => (true, 16, 4, draw_mv(rng)), // NEWMV from GOLDEN
                    _ => (true, 13 + rng.below(4) as i32, 1, draw_mv(rng)),
                }
            };
            let above = draw_nb(&mut rng);
            let left = draw_nb(&mut rng);
            // Half the draws give both neighbours the SAME MV, which is what
            // makes C's "left also differs from above" guard observable.
            let left = if rng.below(2) == 0 && above.0 {
                (left.0, left.1, left.2, above.3)
            } else {
                left
            };

            let ctx = RefNeighbourMvCtx {
                source_sad_nonrd: rng.below(5) as i32,
                est_motion: rng.below(5) as i32,
                is_small_sb,
                best_mv: draw_mv(&mut rng),
                above,
                left,
                mv_limits: (-24, 24, -24, 24),
                // A plausible 64x64 / 128x128 SAD rather than a uniform draw
                // over a huge range: the acceptance test is
                // `cand < (multi * y_sad) >> 3`, so y_sad has to be in the
                // same magnitude band as the SADs the closure returns.
                y_sad: 1 + rng.below(1 << 13) * (block_dim as u32 / 16),
            };

            let (want_sad, want_mv) = ref_vbps_evaluate_neighbour_mvs(
                &ctx,
                block_dim,
                MARGIN,
                &src,
                src_stride as i32,
                &base,
                ref_stride as i32,
            );

            // The port's SAD closure: the reference plane's origin is at
            // (MARGIN, MARGIN), and the candidate MV indexes from there.
            let mut sad_at = |mv: (i32, i32)| -> u32 {
                let off = ((MARGIN + mv.0) as usize) * ref_stride + (MARGIN + mv.1) as usize;
                aom_dsp::dist::sad(&src, src_stride, &base[off..], ref_stride, bd, bd)
            };
            let limits = FullMvLimits {
                row_min: ctx.mv_limits.0,
                row_max: ctx.mv_limits.1,
                col_min: ctx.mv_limits.2,
                col_max: ctx.mv_limits.3,
            };
            let to_nb = |n: (bool, i32, i32, (i16, i16))| -> Option<NeighbourMv> {
                n.0.then(|| NeighbourMv {
                    mode: n.1,
                    ref_frame0: n.2,
                    mv: (i32::from(n.3.0), i32::from(n.3.1)),
                })
            };
            let mut y_sad = ctx.y_sad;
            let got = evaluate_neighbour_mvs(
                [
                    SourceSad::Zero,
                    SourceSad::VeryLow,
                    SourceSad::Low,
                    SourceSad::Med,
                    SourceSad::High,
                ][ctx.source_sad_nonrd as usize],
                ctx.est_motion,
                &limits,
                (i32::from(ctx.best_mv.0), i32::from(ctx.best_mv.1)),
                to_nb(above),
                to_nb(left),
                &mut y_sad,
                &mut sad_at,
            );

            assert_eq!(got.y_sad, want_sad, "y_sad, ctx {ctx:?}");
            assert_eq!(
                (got.mv.0 as i16, got.mv.1 as i16),
                want_mv,
                "mv, ctx {ctx:?}"
            );

            if ctx.est_motion > 2 && ctx.source_sad_nonrd > 3 {
                early_returns += 1;
            } else if got.y_sad != ctx.y_sad {
                if (got.mv.0 as i16, got.mv.1 as i16) == {
                    let l = set_subpel_mv_search_range(&limits, (0, 0));
                    let m = aom_encode::var_part::clamp_mv(
                        (i32::from(above.3.0), i32::from(above.3.1)),
                        &l,
                    );
                    let f = aom_encode::var_part::get_fullmv_from_mv(m);
                    let b = aom_encode::var_part::get_mv_from_fullmv(f);
                    let c = aom_encode::var_part::clamp_mv(b, &l);
                    (c.0 as i16, c.1 as i16)
                } {
                    took_above += 1;
                } else {
                    took_left += 1;
                }
            } else {
                took_neither += 1;
            }
        }
    }
    assert!(
        early_returns > 5,
        "the est_motion early return fired only {early_returns} times"
    );
    assert!(
        took_neither > 20,
        "the block's own MV was never kept ({took_neither})"
    );
    // Both acceptance arms are separate code paths and are mutually exclusive
    // through C's `above_y_sad < left_y_sad` / `left_y_sad < above_y_sad`.
    assert!(
        took_above > 10 && took_left > 10,
        "an acceptance arm never fired ({took_above} above / {took_left} left)"
    );
}

#[test]
fn evaluate_neighbour_mvs_acceptance_boundaries_match_c() {
    // Three things the random sweep above cannot separate, because each needs
    // a candidate SAD to land in a narrow window relative to `y_sad`:
    //
    //  (a) `multi` is 7 rather than 8 when est_motion > 2 and the source SAD
    //      is above kLowSad. The two differ only for a candidate in
    //      `[(7 * y_sad) >> 3, (8 * y_sad) >> 3)`.
    //  (b) that gate is `> kLowSad`, not `> kMedSad`.
    //  (c) the two acceptance arms are mutually exclusive through
    //      `above_y_sad < left_y_sad` / `left_y_sad < above_y_sad`; without
    //      those guards a candidate that loses to the other one would still
    //      overwrite.
    //
    // The construction measures the two candidates' SADs first and then picks
    // `y_sad` to place them, rather than sampling and hoping.
    const MARGIN: i32 = 40;
    let mut rng = Rng::new(0xACC7_5002);
    let (block_dim, is_small_sb) = (64i32, true);
    let bd = block_dim as usize;
    let src_stride = bd + 32;
    let ref_h = bd + 2 * MARGIN as usize;
    let ref_stride = bd + 2 * MARGIN as usize + 32;

    let base: Vec<u8> = (0..ref_stride * ref_h)
        .map(|_| rng.below(256) as u8)
        .collect();
    let (shift_r, shift_c) = (3i32, -2i32);
    let mut src = vec![0u8; src_stride * bd];
    for y in 0..bd {
        for x in 0..bd {
            let ry = (y as i32 + MARGIN + shift_r) as usize;
            let rx = (x as i32 + MARGIN + shift_c) as usize;
            src[y * src_stride + x] = base[ry * ref_stride + rx];
        }
    }
    let sad_of = |mv: (i32, i32)| -> u32 {
        let off = ((MARGIN + mv.0) as usize) * ref_stride + (MARGIN + mv.1) as usize;
        aom_dsp::dist::sad(&src, src_stride, &base[off..], ref_stride, bd, bd)
    };

    // One candidate sits on the shift (a near-zero SAD) and the other is a
    // pixel off. BOTH orderings are driven, because the two acceptance arms
    // are guarded by `above_y_sad < left_y_sad` / the reverse and a single
    // ordering leaves one guard untested.
    let on_shift = (shift_r, shift_c);
    let off_shift = (shift_r + 1, shift_c);
    assert!(
        sad_of(on_shift) < sad_of(off_shift),
        "the construction failed: the on-shift SAD is not the smaller one"
    );

    let mut checked = 0usize;
    // Candidate PAIRS whose SADs are CLOSE. That is what the multiplier's
    // exact value and the two mutual-exclusion guards need: whenever the two
    // SADs are far apart the loser can never win under any multiplier, so the
    // window `[(6 * y) >> 3, (8 * y) >> 3)` has no observable width and every
    // variant of the guards agrees. Two one-pixel offsets in different
    // directions land within a few percent of each other.
    let near_a = (shift_r + 1, shift_c);
    let near_b = (shift_r, shift_c + 1);
    for &(above_full, left_full) in &[
        (on_shift, off_shift),
        (off_shift, on_shift),
        (near_a, near_b),
        (near_b, near_a),
    ] {
        let sad_above = sad_of(above_full);
        let sad_left = sad_of(left_full);
        // A 1/8-pel MV that is NOT a multiple of 8, so get_fullmv_from_mv's
        // sign-aware `+/- 4` rounding is observable -- every MV built from a
        // full-pel shift alone hides it.
        let above = (
            true,
            16i32,
            1i32,
            ((above_full.0 * 8 + 3) as i16, (above_full.1 * 8 - 3) as i16),
        );
        let left = (
            true,
            16i32,
            1i32,
            ((left_full.0 * 8 - 3) as i16, (left_full.1 * 8 + 3) as i16),
        );
        // A best MV far from both, so neither candidate is skipped for equality.
        let best_mv = (16i16 * 8, 16i16 * 8);

        for &(est_motion, sad_raw) in &[
            (0i32, 4i32), // multi == 8
            (3, 2),       // est_motion > 2, source_sad == kLowSad -> still 8
            (3, 3),       // est_motion > 2, source_sad == kMedSad -> multi == 7
        ] {
            // Place each candidate against the 6/8, 7/8 and 8/8 thresholds, so
            // the multiplier's exact value is separated and not just its
            // presence.
            for &cand in &[sad_above, sad_left] {
                for num in [6u64, 7, 8] {
                    for delta in -2i64..=2 {
                        let y_sad = (((u64::from(cand) << 3) / num) as i64 + delta).max(1) as u32;
                        let ctx = RefNeighbourMvCtx {
                            source_sad_nonrd: sad_raw,
                            est_motion,
                            is_small_sb,
                            best_mv,
                            above,
                            left,
                            mv_limits: (-24, 24, -24, 24),
                            y_sad,
                        };
                        let (want_sad, want_mv) = ref_vbps_evaluate_neighbour_mvs(
                            &ctx,
                            block_dim,
                            MARGIN,
                            &src,
                            src_stride as i32,
                            &base,
                            ref_stride as i32,
                        );
                        let limits = FullMvLimits {
                            row_min: -24,
                            row_max: 24,
                            col_min: -24,
                            col_max: 24,
                        };
                        let mut sad_at = |mv: (i32, i32)| sad_of(mv);
                        let mut ys = y_sad;
                        let got = evaluate_neighbour_mvs(
                            [
                                SourceSad::Zero,
                                SourceSad::VeryLow,
                                SourceSad::Low,
                                SourceSad::Med,
                                SourceSad::High,
                            ][sad_raw as usize],
                            est_motion,
                            &limits,
                            (i32::from(best_mv.0), i32::from(best_mv.1)),
                            Some(NeighbourMv {
                                mode: above.1,
                                ref_frame0: above.2,
                                mv: (i32::from(above.3.0), i32::from(above.3.1)),
                            }),
                            Some(NeighbourMv {
                                mode: left.1,
                                ref_frame0: left.2,
                                mv: (i32::from(left.3.0), i32::from(left.3.1)),
                            }),
                            &mut ys,
                            &mut sad_at,
                        );
                        assert_eq!(
                            got.y_sad, want_sad,
                            "y_sad at est {est_motion} sad {sad_raw} num {num} d {delta} y_sad {y_sad}"
                        );
                        assert_eq!(
                            (got.mv.0 as i16, got.mv.1 as i16),
                            want_mv,
                            "mv at est {est_motion} sad {sad_raw} num {num} d {delta} y_sad {y_sad}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checked, 4 * 3 * 2 * 3 * 5, "the boundary grid shrank");
}
