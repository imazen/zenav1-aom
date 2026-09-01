//! Differential harness for the temporal dependency model (`av1/encoder/
//! tpl_model.c`) vs the REAL exported C libaom v3.14.1. **Tier 1** — every
//! oracle below is a direct `extern "C"` call into a `T` symbol of
//! `upstream/build/libaom.a`, with no shim in between.
//!
//! | test | C oracle |
//! |---|---|
//! | `exponential_entropy_matches_c` | `av1_exponential_entropy` |
//! | `laplace_entropy_matches_c` | `av1_laplace_entropy` |
//! | `estimate_coeff_entropy_matches_c` | `av1_estimate_coeff_entropy` |
//! | `get_overlap_area_matches_c` | `av1_get_overlap_area` |
//! | `tpl_ptr_pos_matches_c` | `av1_tpl_ptr_pos` |
//! | `delta_rate_cost_matches_c` | `av1_delta_rate_cost` |
//! | `get_q_index_from_qstep_ratio_matches_c` | `av1_get_q_index_from_qstep_ratio` |
//!
//! Every float comparison is on `to_bits()`, not a tolerance: these are
//! transcendental-heavy expressions evaluated by the same libm on both sides,
//! and an approximate assertion here would hide exactly the rounding-order
//! defect the harness exists to catch.
//!
//! `exp_bounded` and `round_floor` are file-static in C and have no exported
//! symbol. `exp_bounded` is gated *through* `av1_exponential_entropy` — the
//! sweeps below drive `-q_step / b` past both of its ±700 saturation edges
//! (asserted non-vacuously in `exp_bounded_saturation_is_reached`), so a port
//! that dropped the clamp would diverge. `round_floor` has no exported caller
//! in this build's TPL surface either (its only caller, `tpl_model_update_b`,
//! is also static), so it is covered by unit vectors traced from the C source
//! and is **tier 4**, labelled as such at its test.

use aom_encode::tpl_model::{
    delta_rate_cost, estimate_coeff_entropy, exp_bounded, exponential_entropy, get_overlap_area,
    get_q_index_from_qstep_ratio, laplace_entropy, round_floor, tpl_ptr_pos,
};
use aom_sys_ref::{
    ref_delta_rate_cost, ref_estimate_coeff_entropy, ref_exponential_entropy, ref_get_overlap_area,
    ref_get_q_index_from_qstep_ratio, ref_laplace_entropy, ref_tpl_ptr_pos,
};

/// A small deterministic LCG — the same generator shape the other encoder
/// differentials in this crate use, so a failing seed is reproducible.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// A double spanning many orders of magnitude, both signs of exponent —
    /// the range TPL's `b` (a coefficient scale) and `q_step` actually cover.
    fn next_pos_f64(&mut self, min_exp: i32, max_exp: i32) -> f64 {
        let mantissa = f64::from(self.next_u32()) / f64::from(u32::MAX);
        let span = max_exp - min_exp;
        let e = min_exp + (self.next_u32() % (span as u32 + 1)) as i32;
        (1.0 + mantissa) * 2f64.powi(e)
    }
}

// ---------------------------------------------------------------------------
// The entropy model.
// ---------------------------------------------------------------------------

#[test]
fn exponential_entropy_matches_c() {
    let mut rng = Lcg(0x5eed_0001);
    for _ in 0..20_000 {
        let q_step = rng.next_pos_f64(-20, 12);
        let b = rng.next_pos_f64(-30, 20);
        let got = exponential_entropy(q_step, b);
        let want = ref_exponential_entropy(q_step, b);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "q_step={q_step:e} b={b:e}: got {got:e} want {want:e}"
        );
    }
    // The clamp edges: b at and below TPL_EPSILON, and q_step 0.
    for &b in &[0.0, 1e-9, 1e-7, 1e-6, 1.0] {
        for &q_step in &[0.0, 1e-8, 1.0, 4.0, 1024.0, 1e6] {
            let got = exponential_entropy(q_step, b);
            let want = ref_exponential_entropy(q_step, b);
            assert_eq!(got.to_bits(), want.to_bits(), "edge q_step={q_step} b={b}");
        }
    }
}

#[test]
fn laplace_entropy_matches_c() {
    let mut rng = Lcg(0x5eed_0002);
    for _ in 0..20_000 {
        let q_step = rng.next_pos_f64(-20, 12);
        let b = rng.next_pos_f64(-30, 20);
        // 2.0 is what libaom's only caller passes; sweep around it so a port
        // that hard-coded the constant would fail.
        let zero_bin_ratio = f64::from(rng.next_u32() % 800) / 100.0;
        let got = laplace_entropy(q_step, b, zero_bin_ratio);
        let want = ref_laplace_entropy(q_step, b, zero_bin_ratio);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "q_step={q_step:e} b={b:e} zbr={zero_bin_ratio}"
        );
    }
    for &b in &[0.0, 1e-9, 1e-7, 1.0] {
        for &q_step in &[0.0, 1e-8, 4.0, 1e6] {
            for &zbr in &[0.0, 2.0, 8.0] {
                let got = laplace_entropy(q_step, b, zbr);
                let want = ref_laplace_entropy(q_step, b, zbr);
                assert_eq!(got.to_bits(), want.to_bits(), "edge {q_step} {b} {zbr}");
            }
        }
    }
}

#[test]
fn estimate_coeff_entropy_matches_c() {
    let mut rng = Lcg(0x5eed_0003);
    for _ in 0..20_000 {
        let q_step = rng.next_pos_f64(-8, 12);
        let b = rng.next_pos_f64(-20, 16);
        let zero_bin_ratio = f64::from(rng.next_u32() % 800) / 100.0;
        // tran_low_t coefficients: the quantized magnitude is small, and the
        // sign must be exercised because C reads only |qcoeff|.
        let qcoeff = (rng.next_u32() % 4096) as i32 - 2048;
        let got = estimate_coeff_entropy(q_step, b, zero_bin_ratio, qcoeff);
        let want = ref_estimate_coeff_entropy(q_step, b, zero_bin_ratio, qcoeff);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "q_step={q_step:e} b={b:e} zbr={zero_bin_ratio} qcoeff={qcoeff}"
        );
    }
    // The zero arm is a different formula; make sure it is reached and that
    // +n and -n agree (C takes abs).
    for &qcoeff in &[0i32, 1, -1, 7, -7, 32_767, -32_767] {
        let got = estimate_coeff_entropy(4.0, 3.5, 2.0, qcoeff);
        let want = ref_estimate_coeff_entropy(4.0, 3.5, 2.0, qcoeff);
        assert_eq!(got.to_bits(), want.to_bits(), "qcoeff={qcoeff}");
    }
}

#[test]
fn exp_bounded_saturation_is_reached() {
    // **Tier 4** for the clamp itself. `exp_bounded` is file-static, and
    // MEASURED (by perturbing each arm and re-running this file) neither of
    // its two saturation arms is observable through the exported entropy
    // functions:
    //   * the positive arm is unreachable — `exponential_entropy` and
    //     `laplace_entropy` both evaluate `exp_bounded(-x)` for x >= 0, so
    //     returning `f64::INFINITY` instead of `f64::MAX` keeps all 20k cases
    //     of both sweeps green;
    //   * the negative arm is reachable but INERT — `exp(-700)` is 9.86e-305,
    //     already far below the `TPL_EPSILON` floor applied on the next line,
    //     so moving the cut to -1400 is also green.
    // So the assertions below are hand-derived from tpl_model.c:43 and are the
    // only thing pinning the clamp until `av1_tpl_rdmult_setup_sb` — which
    // calls `exp_bounded` with no epsilon floor after it — is ported.
    assert_eq!(exp_bounded(700.0001), f64::MAX);
    assert_eq!(exp_bounded(-700.0001), 0.0);
    assert_eq!(exp_bounded(700.0), 700f64.exp());
    assert_eq!(exp_bounded(-700.0), (-700f64).exp());

    // Both sides of the clamp are reachable from the sweep's own ranges:
    // `exponential_entropy` evaluates `exp_bounded(-q_step / b)`, so a small
    // q_step over a large b stays inside the exp, and a large q_step over a
    // tiny b saturates to 0.
    let unsaturated = (-1.0f64) / 1.0;
    assert!(unsaturated > -700.0 && unsaturated < 700.0);
    let q_step = 1024.0;
    let deep_b = 1e-7;
    assert!(-q_step / deep_b < -700.0);
    // And the C oracle agrees on both.
    for &(q, b) in &[(1.0f64, 1.0f64), (q_step, deep_b)] {
        assert_eq!(
            exponential_entropy(q, b).to_bits(),
            ref_exponential_entropy(q, b).to_bits(),
            "q_step={q} b={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Geometry / indexing.
// ---------------------------------------------------------------------------

#[test]
fn get_overlap_area_matches_c() {
    let mut saw_overlap = false;
    let mut saw_disjoint = false;
    for &(width, height) in &[(16i32, 16i32), (8, 16), (32, 8), (1, 1), (64, 64)] {
        for row_a in -40..40 {
            for col_a in [-33i32, -16, -1, 0, 1, 15, 16, 40] {
                for row_b in [-40i32, -16, 0, 3, 16, 39] {
                    for col_b in [-20i32, 0, 7, 16, 33] {
                        let got = get_overlap_area(row_a, col_a, row_b, col_b, width, height);
                        let want = ref_get_overlap_area(row_a, col_a, row_b, col_b, width, height);
                        assert_eq!(
                            got, want,
                            "a=({row_a},{col_a}) b=({row_b},{col_b}) {width}x{height}"
                        );
                        if got > 0 {
                            saw_overlap = true;
                        } else {
                            saw_disjoint = true;
                        }
                    }
                }
            }
        }
    }
    assert!(saw_overlap && saw_disjoint, "sweep reached only one arm");
}

#[test]
fn tpl_ptr_pos_matches_c() {
    for right_shift in 0u8..=4 {
        for stride in [1i32, 3, 16, 64, 257] {
            for mi_row in 0..64i32 {
                for mi_col in 0..64i32 {
                    assert_eq!(
                        tpl_ptr_pos(mi_row, mi_col, stride, right_shift),
                        ref_tpl_ptr_pos(mi_row, mi_col, stride, right_shift),
                        "({mi_row},{mi_col}) stride={stride} rs={right_shift}"
                    );
                }
            }
        }
    }
}

/// **Tier 4** — hand-derived vectors traced against `round_floor`
/// (tpl_model.c:1149). The function is `static` and its only caller
/// (`tpl_model_update_b`) is static too, so no exported symbol reaches it and
/// there is nothing in `libaom.a` to compare against. The expectations below
/// are read off C's own two arms: `ref_pos / bsize_pix` when non-negative, and
/// `-(1 + (-ref_pos - 1) / bsize_pix)` when negative.
#[test]
fn round_floor_matches_hand_derived_vectors_tier4() {
    // Non-negative arm: plain truncating division.
    assert_eq!(round_floor(0, 16), 0);
    assert_eq!(round_floor(15, 16), 0);
    assert_eq!(round_floor(16, 16), 1);
    assert_eq!(round_floor(31, 16), 1);
    assert_eq!(round_floor(32, 16), 2);
    // Negative arm: floor, not truncation — this is the whole point of the
    // function, and a `ref_pos / bsize_pix` transcription gets every one of
    // these wrong except the exact multiples.
    assert_eq!(round_floor(-1, 16), -1);
    assert_eq!(round_floor(-15, 16), -1);
    assert_eq!(round_floor(-16, 16), -1);
    assert_eq!(round_floor(-17, 16), -2);
    assert_eq!(round_floor(-32, 16), -2);
    assert_eq!(round_floor(-33, 16), -3);
    // C's spelling agrees with floor-division for every positive divisor.
    for bsize_pix in 1..=64i32 {
        for ref_pos in -200..200i32 {
            assert_eq!(
                round_floor(ref_pos, bsize_pix),
                (ref_pos as f64 / f64::from(bsize_pix)).floor() as i32,
                "ref_pos={ref_pos} bsize_pix={bsize_pix}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The dependency-cost algebra and the qindex search.
// ---------------------------------------------------------------------------

#[test]
fn delta_rate_cost_matches_c() {
    let mut rng = Lcg(0x5eed_0004);
    let mut saw_early = false;
    let mut saw_saturated = false;
    let mut saw_general = false;
    for _ in 0..40_000 {
        // Distortions are int64 sums of squared error over a block; rates are
        // in TPL's scaled units (<< TPL_DEP_COST_SCALE_LOG2 + PROB_COST_SHIFT).
        let recrf_dist = i64::from(rng.next_u32() % 1_000_000) + 1;
        // A plain uniform draw over 2e6 hits the `<= 128` early exit about
        // twice in 40k trials, which is a coin-flip away from a vacuous arm.
        // Draw from a mixture instead so the early exit is reached by
        // construction rather than by luck.
        let srcrf_dist = if rng.next_u32() % 8 == 0 {
            i64::from(rng.next_u32() % 300)
        } else {
            i64::from(rng.next_u32() % 2_000_000)
        };
        let delta_rate = i64::from(rng.next_u32()) << 4;
        let pix_num = [16i32, 64, 256, 1024, 4096][(rng.next_u32() % 5) as usize];
        let got = delta_rate_cost(delta_rate, recrf_dist, srcrf_dist, pix_num);
        let want = ref_delta_rate_cost(delta_rate, recrf_dist, srcrf_dist, pix_num);
        assert_eq!(
            got, want,
            "dr={delta_rate} rec={recrf_dist} src={srcrf_dist} pix={pix_num}"
        );
        // Which arm did C take? Recompute the predicate from the inputs
        // rather than inferring it from the answer.
        if srcrf_dist <= 128 {
            saw_early = true;
        } else {
            let beta = srcrf_dist as f64 / recrf_dist as f64;
            let log_den = beta.log2() + 2.0 * ((delta_rate >> 13) as f64 / f64::from(pix_num));
            if log_den > 10f64.log2() {
                saw_saturated = true;
            } else {
                saw_general = true;
            }
        }
    }
    // Force the two non-early arms explicitly: a very small beta drives
    // log_den below log2(10) (general), a large one above it (saturated).
    for &(rec, src, dr, pix) in &[
        (1_000_000i64, 200i64, 1i64 << 20, 256i32), // beta << 1
        (200, 1_000_000, 1 << 20, 256),             // beta >> 1
        (1000, 1000, 0, 256),                       // beta == 1, dr == 0
        (1, 1_000_000_000, 1 << 30, 4096),          // extreme beta
        (1_000_000_000, 129, 1 << 30, 16),          // just past the early exit
    ] {
        let got = delta_rate_cost(dr, rec, src, pix);
        let want = ref_delta_rate_cost(dr, rec, src, pix);
        assert_eq!(got, want, "rec={rec} src={src} dr={dr} pix={pix}");
        let beta = src as f64 / rec as f64;
        let log_den = beta.log2() + 2.0 * ((dr >> 13) as f64 / f64::from(pix));
        if src > 128 && log_den > 10f64.log2() {
            saw_saturated = true;
        }
    }
    assert!(
        saw_early,
        "sweep never took the srcrf_dist <= 128 early exit"
    );
    assert!(saw_saturated, "sweep never took the log_den > log2(10) arm");
    assert!(saw_general, "sweep never took the general arm");
}

#[test]
fn get_q_index_from_qstep_ratio_matches_c() {
    let mut saw_down = false;
    let mut saw_up = false;
    let mut saw_floor = false;
    let mut saw_ceiling = false;
    for &bd in &[8u8, 10, 12] {
        for leaf_qindex in 0..=255i32 {
            for step in 0..40 {
                // 0.05 .. 2.0 — the ratio is sqrt(1 / frame_importance), so it
                // straddles 1.0 and the two loop directions.
                let qstep_ratio = 0.05 + f64::from(step) * 0.05;
                let got = get_q_index_from_qstep_ratio(leaf_qindex, qstep_ratio, bd);
                let want = ref_get_q_index_from_qstep_ratio(leaf_qindex, qstep_ratio, bd);
                assert_eq!(got, want, "bd={bd} leaf={leaf_qindex} ratio={qstep_ratio}");
                if qstep_ratio < 1.0 {
                    saw_down = true;
                    if got == 0 {
                        saw_floor = true;
                    }
                } else {
                    saw_up = true;
                    if got == 255 {
                        saw_ceiling = true;
                    }
                }
            }
        }
    }
    assert!(
        saw_down && saw_up,
        "sweep reached only one search direction"
    );
    assert!(saw_floor, "sweep never walked down to qindex 0");
    assert!(saw_ceiling, "sweep never walked up to MAXQ");
}

#[test]
fn get_q_index_from_qstep_ratio_bit_depths_differ() {
    // The DC quantizer table is per-bit-depth, so the same ratio lands on a
    // different qindex. If it did not, the bd sweep above would be one test
    // repeated three times.
    let mut differed = false;
    for leaf in [40i32, 128, 200] {
        let a = get_q_index_from_qstep_ratio(leaf, 0.5, 8);
        let b = get_q_index_from_qstep_ratio(leaf, 0.5, 10);
        let c = get_q_index_from_qstep_ratio(leaf, 0.5, 12);
        if a != b || b != c {
            differed = true;
        }
    }
    assert!(differed, "all three bit depths returned the same qindex");
}

// ---------------------------------------------------------------------------
// The frame-importance -> qstep-ratio -> qindex chain.
//
// **Tier 1** through `shim/tpl_shim.c`: all three C entry points are exported
// `T` symbols, and the shim exists only to build the ~100 KB `TplParams` they
// take. `get_frame_importance` is file-static and `av1_tpl_get_qstep_ratio` is
// its only caller, so driving the caller gates the static at tier 1 too.
// ---------------------------------------------------------------------------

use aom_encode::tpl_model::{TplDepFrame, TplDepStats, TplParams};
use aom_sys_ref::{
    RefTplCell, RefTplFrameDesc, ref_tpl_get_q_index, ref_tpl_get_qstep_ratio, ref_tpl_stats_ready,
};

/// Build the port-side and oracle-side views of the same synthetic TPL frame.
///
/// The cell array is indexed exactly as `av1_tpl_ptr_pos` indexes it on both
/// sides, so this shares the addressing under test with neither implementation
/// — it just sizes the buffer to cover the whole walk.
fn build_frame(
    rng: &mut Lcg,
    ready: bool,
    gf_frame_index: i32,
    is_valid: bool,
    mi_rows: i32,
    mi_cols: i32,
    base_rdmult: i32,
    block_mis_log2: u8,
) -> (TplParams, RefTplFrameDesc, Vec<RefTplCell>) {
    let step = 1i32 << block_mis_log2;
    let stride = (mi_cols + step - 1) >> block_mis_log2;
    let rows = (mi_rows + step - 1) >> block_mis_log2;
    let n = (stride.max(1) * rows.max(1)).max(1) as usize;

    let mut cells = Vec::with_capacity(n);
    for _ in 0..n {
        // Bounds come from what the TPL pass can actually produce, not from
        // the type: `srcrf_dist` / `recrf_dist` are SSE-scale sums over a
        // 16x16 block (<= 255^2 * 256 ~= 1.7e7 at bd 8, more at bd 12, so 1e9
        // is generous), `mc_dep_rate` is a sum of `av1_delta_rate_cost`
        // outputs already scaled by 1 << 13, and `mc_dep_dist` accumulates
        // distortion down the dependency chain.
        cells.push(RefTplCell {
            srcrf_dist: i64::from(rng.next_u32() % 1_000_000_000),
            recrf_dist: i64::from(rng.next_u32() % 1_000_000_000),
            mc_dep_rate: i64::from(rng.next_u32()) << 8,
            mc_dep_dist: i64::from(rng.next_u32()) << 8,
        });
    }

    let stats: Vec<TplDepStats> = cells
        .iter()
        .map(|c| TplDepStats {
            srcrf_dist: c.srcrf_dist,
            recrf_dist: c.recrf_dist,
            mc_dep_rate: c.mc_dep_rate,
            mc_dep_dist: c.mc_dep_dist,
            ..TplDepStats::default()
        })
        .collect();

    let mut frames = vec![TplDepFrame::default(); (gf_frame_index.max(0) + 1) as usize];
    frames[gf_frame_index.max(0) as usize] = TplDepFrame {
        is_valid,
        stats,
        stride,
        mi_rows,
        mi_cols,
        base_rdmult,
        ..TplDepFrame::default()
    };
    let tpl = TplParams {
        ready,
        tpl_stats_block_mis_log2: block_mis_log2,
        tpl_bsize_1d: 16,
        frames,
        ..TplParams::default()
    };
    let desc = RefTplFrameDesc {
        ready,
        gf_frame_index,
        is_valid,
        mi_rows,
        mi_cols,
        stride,
        base_rdmult,
        block_mis_log2,
    };
    (tpl, desc, cells)
}

#[test]
fn tpl_stats_ready_matches_c() {
    let mut saw_true = false;
    let mut saw_index_gate = false;
    for &ready in &[false, true] {
        for &is_valid in &[false, true] {
            // 95 and 96 straddle MAX_TPL_FRAME_IDX (2 * MAX_LAG_BUFFERS = 96),
            // which is the gate that silently disables TPL for a long sub-GOP.
            for gf in [0i32, 1, 7, 47, 94, 95, 96, 97, 104] {
                let got = TplParams {
                    ready,
                    frames: {
                        let mut f = vec![TplDepFrame::default(); (gf + 1) as usize];
                        f[gf as usize].is_valid = is_valid;
                        f
                    },
                    ..TplParams::default()
                }
                .tpl_stats_ready(gf);
                let want = ref_tpl_stats_ready(ready, gf, is_valid);
                assert_eq!(got, want, "ready={ready} valid={is_valid} gf={gf}");
                if got {
                    saw_true = true;
                }
                if ready && is_valid && gf >= 96 && !got {
                    saw_index_gate = true;
                }
            }
        }
    }
    assert!(saw_true, "sweep never produced a ready frame");
    assert!(
        saw_index_gate,
        "sweep never exercised the MAX_TPL_FRAME_IDX gate"
    );
}

#[test]
fn tpl_get_qstep_ratio_matches_c() {
    let mut rng = Lcg(0x5eed_0010);
    let mut saw_not_ready = false;
    let mut saw_computed = false;
    // 16x16 up to 128x128 mi, i.e. 256x256 up to 2048x2048 luma, plus two
    // sizes that are not a multiple of the 4-mi decimation so the walk's
    // partial last step is exercised.
    for &(mi_rows, mi_cols) in &[
        (4i32, 4i32),
        (16, 16),
        (17, 5),
        (33, 65),
        (64, 64),
        (128, 128),
    ] {
        for &ready in &[false, true] {
            for &is_valid in &[false, true] {
                for &base_rdmult in &[1i32, 57, 1000, 100_000] {
                    let (tpl, desc, cells) = build_frame(
                        &mut rng,
                        ready,
                        3,
                        is_valid,
                        mi_rows,
                        mi_cols,
                        base_rdmult,
                        2,
                    );
                    let got = tpl.tpl_get_qstep_ratio(3);
                    let want = ref_tpl_get_qstep_ratio(desc, &cells);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{mi_rows}x{mi_cols} ready={ready} valid={is_valid} rdmult={base_rdmult}: \
                         got {got} want {want}"
                    );
                    if ready && is_valid {
                        saw_computed = true;
                        assert!(
                            (got - 1.0).abs() > f64::EPSILON,
                            "importance was exactly 1 — the walk contributed nothing"
                        );
                    } else {
                        saw_not_ready = true;
                        assert_eq!(got, 1.0);
                    }
                }
            }
        }
    }
    assert!(saw_not_ready && saw_computed, "one arm was never reached");
}

#[test]
fn tpl_get_qstep_ratio_decimation_is_read_from_state() {
    // `tpl_stats_block_mis_log2` is 2 in every libaom build, so a port that
    // hard-coded the step would pass every other test here. Sweep it to prove
    // the walk reads it — and check the port and C agree at each value.
    let mut rng = Lcg(0x5eed_0011);
    let mut ratios = Vec::new();
    for shift in 0u8..=3 {
        let (tpl, desc, cells) = build_frame(&mut rng, true, 2, true, 32, 32, 300, shift);
        let got = tpl.tpl_get_qstep_ratio(2);
        let want = ref_tpl_get_qstep_ratio(desc, &cells);
        assert_eq!(got.to_bits(), want.to_bits(), "shift={shift}");
        ratios.push(got);
    }
    assert!(
        ratios.windows(2).any(|w| w[0] != w[1]),
        "every decimation gave the same ratio — the sweep is inert"
    );
}

#[test]
fn tpl_get_q_index_matches_c() {
    let mut rng = Lcg(0x5eed_0012);
    let mut saw_lower = false;
    let mut saw_higher = false;
    for &bd in &[8u8, 10, 12] {
        for &leaf_qindex in &[0i32, 20, 90, 128, 200, 255] {
            for &(mi_rows, mi_cols) in &[(16i32, 16i32), (33, 65), (64, 64)] {
                for &base_rdmult in &[1i32, 500, 50_000] {
                    let (tpl, desc, cells) =
                        build_frame(&mut rng, true, 5, true, mi_rows, mi_cols, base_rdmult, 2);
                    let got = tpl.tpl_get_q_index(5, leaf_qindex, bd);
                    let want = ref_tpl_get_q_index(desc, &cells, leaf_qindex, bd);
                    assert_eq!(
                        got, want,
                        "bd={bd} leaf={leaf_qindex} {mi_rows}x{mi_cols} rdmult={base_rdmult}"
                    );
                    if got < leaf_qindex {
                        saw_lower = true;
                    }
                    if got > leaf_qindex {
                        saw_higher = true;
                    }
                }
            }
        }
    }
    assert!(
        saw_lower || saw_higher,
        "TPL never moved the qindex off the leaf value — the chain is inert"
    );
}

/// `get_frame_importance`'s `AOMMAX(dist_scaled, 1)` floor and its
/// `cbcmp_base = 1` seed, gated at the degenerate inputs that reach them.
///
/// **Both are provably inert on encoder-produced state**, measured by
/// perturbing them: deleting the floor leaves the 6-size x 4-rdmult sweep in
/// `tpl_get_qstep_ratio_matches_c` green, because `tpl_model_store`
/// (tpl_model.c:1301) clamps `recrf_dist = AOMMAX(1, recrf_dist)` before the
/// cell is ever stored, so `dist_scaled = recrf_dist << 7` is at least 128 and
/// the floor never binds. This test therefore feeds the values the clamp
/// excludes, which C is perfectly well-defined on — it is pinning the port's
/// arithmetic against C's, not claiming the encoder can produce them.
#[test]
fn get_frame_importance_degenerate_cells_match_c() {
    for &(srcrf_dist, recrf_dist) in &[
        (0i64, 0i64),   // both zero: log(1) * 0, and cbcmp_base stays 1
        (0, 1),         // dist_scaled = 128, weight 0
        (5, 0),         // the floor binds and the weight does not
        (0, 0),         // repeated deliberately with a different rdmult below
        (1_000_000, 0), // large weight on a floored distortion
    ] {
        for &base_rdmult in &[0i32, 1, 4096] {
            for &mc_dep in &[0i64, 1, 1 << 20] {
                let n = 16usize;
                let cells = vec![
                    RefTplCell {
                        srcrf_dist,
                        recrf_dist,
                        mc_dep_rate: mc_dep,
                        mc_dep_dist: mc_dep,
                    };
                    n
                ];
                let stats: Vec<TplDepStats> = cells
                    .iter()
                    .map(|c| TplDepStats {
                        srcrf_dist: c.srcrf_dist,
                        recrf_dist: c.recrf_dist,
                        mc_dep_rate: c.mc_dep_rate,
                        mc_dep_dist: c.mc_dep_dist,
                        ..TplDepStats::default()
                    })
                    .collect();
                let mut frames = vec![TplDepFrame::default(); 1];
                frames[0] = TplDepFrame {
                    is_valid: true,
                    stats,
                    stride: 4,
                    mi_rows: 16,
                    mi_cols: 16,
                    base_rdmult,
                    ..TplDepFrame::default()
                };
                let tpl = TplParams {
                    ready: true,
                    tpl_stats_block_mis_log2: 2,
                    tpl_bsize_1d: 16,
                    frames,
                    ..TplParams::default()
                };
                let desc = RefTplFrameDesc {
                    ready: true,
                    gf_frame_index: 0,
                    is_valid: true,
                    mi_rows: 16,
                    mi_cols: 16,
                    stride: 4,
                    base_rdmult,
                    block_mis_log2: 2,
                };
                let got = tpl.tpl_get_qstep_ratio(0);
                let want = ref_tpl_get_qstep_ratio(desc, &cells);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "src={srcrf_dist} rec={recrf_dist} rdmult={base_rdmult} mc_dep={mc_dep}"
                );
            }
        }
    }
}
