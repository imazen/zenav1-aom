//! Differential: the HOG intra-mode prune vs the REAL C pieces compiled from
//! the header (hog_shim.c includes intra_mode_search_utils.h — its own static
//! weights/nnconfig and the real lowbd/highbd_generate_hog bodies):
//! - the NN kernel `hog_nn_predict` vs libaom's, under TWO explicitly
//!   different contracts, because the thing being asserted is genuinely
//!   different per target (see the two `hog_nn_predict_*` tests below);
//! - `generate_hog` vs the real Sobel-histogram statics across depths,
//!   content classes and frame-edge-clipped dims;
//! - `prune_intra_mode_with_hog_y` end-to-end mask equality, thresholds
//!   including the speed-0 `-1.2f`.
//!
//! ## Why the NN kernel has two target-conditional contracts
//!
//! The port's [`hog_nn_predict`] replicates the **x86 AVX2** kernel's lane
//! math by design (`hog.rs` module docs), because that is the accumulation
//! order the shipping reference platform dispatches to. `av1_nn_predict_avx2`
//! is not merely unselected on ARM — `ml_avx2.c` is x86-intrinsic source that
//! is **never compiled** into an ARM libaom, so no shim, oracle build option
//! or RTCD manipulation can produce AVX2-order scores on an ARM box. Making
//! the port choose its lane order by `cfg!(target_arch)` is rejected for the
//! same reason `-ffp-contract=off` is pinned in the oracle (KB-ARM-FLOAT root
//! #1): the port's own output must not be host-dependent.
//!
//! So the bit-exactness assertion is stated as what it is — an **x86-64**
//! contract — and the non-x86 target gets its own, separately-named test
//! asserting the properties that ARE defined there (the 1/512
//! `av1_nn_output_prec_reduce` quantum, one-quantum agreement, and prune-mask
//! parity at every production threshold). Neither test is a relaxation of the
//! other: they assert different, individually-complete things, and both fail
//! loudly if the kernel drifts.

use aom_encode::hog::{
    HOG_BINS, generate_hog, hog_nn_predict, prune_intra_mode_with_hog_uv,
    prune_intra_mode_with_hog_y,
};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    fn f01(&mut self) -> f32 {
        (self.next() % (1 << 20)) as f32 / (1u64 << 20) as f32
    }
}

/// The NN input corpus, shared verbatim by both target-conditional contracts
/// below so the x86 and non-x86 bars are stated over the SAME 20,000
/// histograms in the same order (the RNG is advanced identically): 4 regimes
/// — histogram-shaped (normalized non-negative, summing ~1), one-hot-ish
/// (mass in a few bins, the typical directional HOG), raw signed floats, and
/// all-zero. The kernel math must match on any input, not just realizable
/// histograms.
fn hog_case_hist(case: usize, rng: &mut Rng) -> [f32; HOG_BINS] {
    let mut hist = [0f32; HOG_BINS];
    match case % 4 {
        0 => {
            let mut total = 0f32;
            for h in hist.iter_mut() {
                *h = rng.f01();
                total += *h;
            }
            for h in hist.iter_mut() {
                *h /= total;
            }
        }
        1 => {
            for _ in 0..3 {
                hist[(rng.next() % 32) as usize] = rng.f01();
            }
        }
        2 => {
            for h in hist.iter_mut() {
                *h = (rng.f01() - 0.5) * 8.0;
            }
        }
        _ => {} // all-zero
    }
    hist
}

/// **x86-64 contract, in one sentence:** on the reference platform the port's
/// `hog_nn_predict` is BIT-IDENTICAL (`f32` bit pattern, both `reduce_prec`
/// settings) to `av1_nn_predict_avx2`, and libaom's RTCD dispatch on this
/// machine resolves to that same AVX2 variant — so the port reproduces
/// exactly the kernel the shipping encoder runs.
///
/// This assertion is x86-only because `av1_nn_predict_avx2` exists only in an
/// x86 libaom (`ml_avx2.c` is x86-intrinsic source); see the module docs and
/// `hog_nn_predict_agrees_with_dispatch_within_one_prec_quantum` for the
/// contract that holds elsewhere.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn hog_nn_predict_matches_avx2_and_dispatch() {
    c::ref_init();
    let mut rng = Rng(0x09a1_a55e_11ea_a7ed);
    let mut nonpos_scores = 0usize;
    let mut pos_scores = 0usize;
    for case in 0..20_000 {
        let hist = hog_case_hist(case, &mut rng);
        for reduce in [false, true] {
            let got = hog_nn_predict(&hist, reduce);
            let want = c::ref_hog_nn_predict(&hist, reduce);
            let disp = c::ref_hog_nn_predict_dispatched(&hist, reduce);
            for i in 0..8 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "avx2 score[{i}] {} vs {} case={case} reduce={reduce}",
                    got[i],
                    want[i],
                );
                assert_eq!(
                    want[i].to_bits(),
                    disp[i].to_bits(),
                    "RTCD dispatch is not the AVX2 variant on this machine \
                     (score[{i}] {} vs {}, case={case})",
                    want[i],
                    disp[i],
                );
                if got[i] <= 0.0 {
                    nonpos_scores += 1;
                } else {
                    pos_scores += 1;
                }
            }
        }
    }
    assert!(
        nonpos_scores > 10_000,
        "non-positive scores: {nonpos_scores}"
    );
    assert!(pos_scores > 10_000, "positive scores: {pos_scores}");
}

/// `av1_nn_output_prec_reduce` (upstream/av1/encoder/ml.c:19-25) transcribed:
/// `prec_bits = 9`, `prec = 1 << prec_bits`, `inv_prec = (float)(1.0 / prec)`,
/// `output[i] = ((int)(output[i] * prec + 0.5)) * inv_prec`. The `* prec`
/// happens in `float`; `+ 0.5` is a `double` literal so the add is in
/// `double`; `(int)` truncates toward zero; `inv_prec` is an exact power of
/// two. Every reduced score is therefore an EXACT integer multiple of
/// `1/512` — that integer is the "quantum index" used below, and the
/// one-quantum bound is read off this function, not chosen as a tolerance.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
mod prec {
    /// `1 << prec_bits`, ml.c:20-21.
    pub const PREC: f32 = 512.0;
    /// `(float)(1.0 / prec)`, ml.c:22 — exactly representable.
    pub const INV_PREC: f32 = 1.0 / 512.0;

    pub fn reduce(v: f32) -> f32 {
        ((f64::from(v * PREC) + 0.5) as i32) as f32 * INV_PREC
    }

    /// The exact integer `n` with `v == n * (1/512)`. Panics (via the caller's
    /// assert) if `v` is not on the lattice, which would mean the value never
    /// went through `av1_nn_output_prec_reduce`.
    pub fn quantum_index(v: f32) -> Option<i64> {
        let scaled = f64::from(v) * f64::from(PREC);
        if scaled == scaled.trunc() {
            Some(scaled as i64)
        } else {
            None
        }
    }
}

/// **Non-x86 contract, in one sentence:** where libaom has no AVX2 kernel to
/// be bit-identical to, the port's `hog_nn_predict` and the RTCD-dispatched
/// libaom kernel both land exactly on the `1/512` lattice that
/// `av1_nn_output_prec_reduce` defines, never differ by more than ONE lattice
/// step, and produce the SAME `score <= th` prune mask at every threshold the
/// production encoder uses — with the number of lanes that differ at all
/// pinned so it cannot grow.
///
/// Each clause is a hard equality or an exact integer bound; none of them is
/// a widened tolerance, and none of them is weaker than what the x86 test
/// asserts *about a different quantity*. What is deliberately NOT asserted
/// here is raw `f32` bit-equality against `av1_nn_predict_avx2`, because that
/// function does not exist on this target — asserting it would be asserting
/// against nothing.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[test]
fn hog_nn_predict_agrees_with_dispatch_within_one_prec_quantum() {
    c::ref_init();

    /// Every threshold `prune_intra_mode_with_hog` is invoked with in
    /// production, de-duplicated: luma intra `{-1.2, -1.2, -0.6, 0.4}`
    /// (intra_mode_search.c:1505), luma inter `{-1.2, 0.0, 0.0, 1.2}`
    /// (:1321), chroma inter/intra (:961-964, the same two rows). `0.0` is
    /// itself ON the 1/512 lattice, so it is the threshold most exposed to a
    /// one-quantum difference — it is deliberately included.
    const PRODUCTION_THRESHOLDS: [f32; 5] = [-1.2, -0.6, 0.0, 0.4, 1.2];

    /// Pinned characterization of the ARM/AVX2 accumulation-order difference
    /// at `reduce_prec = true` (the only setting the encoder uses), over the
    /// 20,000-case corpus x 8 lanes = 160,000 lanes. Measured on
    /// aarch64-apple-darwin; asserted as an upper bound so the divergence
    /// cannot GROW silently. See CLAUDE.md KB-ARM-FLOAT.
    const MAX_ONE_QUANTUM_LANES: usize = 56;
    /// Pinned count of prune-mask disagreements across all thresholds. A
    /// one-quantum score difference flips the mask only when the threshold
    /// falls strictly between the two lattice points, so this is far rarer
    /// than the lane count above — MEASURED at 0 over this corpus, i.e. the
    /// port and the dispatched kernel steer the encode identically here. If a
    /// libaom or toolchain change ever makes a straddle real, raising this
    /// must be a deliberate, measured decision recorded in CLAUDE.md
    /// KB-ARM-FLOAT — not a quick edit to make CI green.
    const MAX_MASK_FLIPS: usize = 0;

    let mut rng = Rng(0x09a1_a55e_11ea_a7ed);
    let mut lanes = 0usize;
    let mut one_quantum_lanes = 0usize;
    let mut worst_gap: i64 = 0;
    let mut over_one_quantum: Vec<String> = Vec::new();
    let mut mask_flips: Vec<String> = Vec::new();
    let mut mask_flip_total = 0usize;
    // Anti-vacuity: per threshold, how often each polarity of `<= th` occurs.
    let mut below = [0usize; PRODUCTION_THRESHOLDS.len()];
    let mut above = [0usize; PRODUCTION_THRESHOLDS.len()];

    for case in 0..20_000 {
        let hist = hog_case_hist(case, &mut rng);

        // The raw (unreduced) kernel outputs, and the reduced ones. Asserting
        // reduce(raw) == reduced on BOTH sides proves each implementation
        // really applies ml.c:19-25 — i.e. the 1/512 lattice claim below is
        // established, not assumed, and the `reduce_prec = false` regime is
        // covered rather than dropped.
        let got_raw = hog_nn_predict(&hist, false);
        let got = hog_nn_predict(&hist, true);
        let want_raw = c::ref_hog_nn_predict(&hist, false);
        let want = c::ref_hog_nn_predict(&hist, true);
        let disp = c::ref_hog_nn_predict_dispatched(&hist, true);

        for i in 0..8 {
            // The widest SIMD variant libaom HAS on this target and what RTCD
            // dispatches to must be the same function — the surviving half of
            // the x86 test's dispatch check, and a real statement here (NEON
            // is bound at compile time via `#define av1_nn_predict
            // av1_nn_predict_neon`, so a change would be visible).
            assert_eq!(
                want[i].to_bits(),
                disp[i].to_bits(),
                "RTCD dispatch is not the widest SIMD variant on this machine \
                 (score[{i}] {} vs {}, case={case})",
                want[i],
                disp[i],
            );

            assert_eq!(
                got[i].to_bits(),
                prec::reduce(got_raw[i]).to_bits(),
                "port's reduce_prec output is not av1_nn_output_prec_reduce \
                 of its own raw output (score[{i}], case={case})",
            );
            assert_eq!(
                want[i].to_bits(),
                prec::reduce(want_raw[i]).to_bits(),
                "oracle's reduce_prec output is not av1_nn_output_prec_reduce \
                 of its own raw output (score[{i}], case={case})",
            );

            let qg = prec::quantum_index(got[i]).unwrap_or_else(|| {
                panic!(
                    "port score[{i}] {} is off the 1/512 lattice (case={case})",
                    got[i]
                )
            });
            let qw = prec::quantum_index(want[i]).unwrap_or_else(|| {
                panic!(
                    "oracle score[{i}] {} is off the 1/512 lattice (case={case})",
                    want[i]
                )
            });

            let gap = qg - qw;
            worst_gap = worst_gap.max(gap.abs());
            if gap != 0 {
                one_quantum_lanes += 1;
            }
            if gap.abs() > 1 && over_one_quantum.len() < 8 {
                over_one_quantum.push(format!(
                    "case={case} score[{i}] port={} ({qg}/512) oracle={} ({qw}/512) gap={gap} quanta",
                    got[i], want[i]
                ));
            }

            for (t, &th) in PRODUCTION_THRESHOLDS.iter().enumerate() {
                let mg = got[i] <= th;
                let mw = want[i] <= th;
                if mg {
                    below[t] += 1;
                } else {
                    above[t] += 1;
                }
                if mg != mw {
                    mask_flip_total += 1;
                    if mask_flips.len() < 8 {
                        mask_flips.push(format!(
                            "th={th} case={case} score[{i}] port={} ({qg}/512, prune={mg}) \
                             oracle={} ({qw}/512, prune={mw}) gap={gap} quanta",
                            got[i], want[i]
                        ));
                    }
                }
            }
            lanes += 1;
        }
    }

    eprintln!(
        "hog NN non-x86 contract: lanes={lanes} one_quantum_lanes={one_quantum_lanes} \
         worst_gap={worst_gap} quanta mask_flips={mask_flip_total}"
    );

    // (1) PRUNE-MASK PARITY — the property that actually steers the encode.
    //
    // `MAX_MASK_FLIPS` is currently 0, so clippy sees `<= 0` on a `usize` and
    // suggests `==`. Keep the `<=`: the bound is documented as raisable by a
    // deliberate, measured decision (see its doc comment and CLAUDE.md
    // KB-ARM-FLOAT), and `==` would turn a raised bound into an assertion that
    // the divergence must be EXACTLY that large.
    #[allow(clippy::absurd_extreme_comparisons)]
    let mask_parity_holds = mask_flip_total <= MAX_MASK_FLIPS;
    assert!(
        mask_parity_holds,
        "prune-mask parity broke: {mask_flip_total} flips (pinned max {MAX_MASK_FLIPS}) \
         across {lanes} lanes x {} production thresholds. Samples:\n  {}",
        PRODUCTION_THRESHOLDS.len(),
        mask_flips.join("\n  "),
    );

    // (2) ONE-QUANTUM AGREEMENT — no score may differ by more than a single
    // av1_nn_output_prec_reduce step.
    assert!(
        over_one_quantum.is_empty(),
        "scores differ by more than one 1/512 quantum (worst {worst_gap}):\n  {}",
        over_one_quantum.join("\n  "),
    );

    // (3) The difference is CHARACTERIZED and BOUNDED, not tolerated: pin how
    // many lanes differ at all so a regression that widens the gap fails even
    // though every individual gap is still one quantum.
    assert!(
        one_quantum_lanes <= MAX_ONE_QUANTUM_LANES,
        "one-quantum lane divergence grew: {one_quantum_lanes} of {lanes} \
         (pinned max {MAX_ONE_QUANTUM_LANES})"
    );
    assert!(
        one_quantum_lanes > 0,
        "expected the non-x86 kernel to differ from the port's AVX2 order on \
         at least one lane; got 0 of {lanes} — if libaom's dispatched kernel \
         now reproduces AVX2 order exactly, this target belongs under the \
         bit-exact x86 contract instead of this one"
    );

    // (4) Anti-vacuity: mask parity means nothing unless both polarities of
    // every threshold are actually exercised.
    for (t, &th) in PRODUCTION_THRESHOLDS.iter().enumerate() {
        assert!(
            below[t] > 1_000 && above[t] > 1_000,
            "threshold {th} not exercised on both sides: below={} above={}",
            below[t],
            above[t]
        );
    }
}

/// Fill a rows x cols window with one content class.
#[allow(clippy::too_many_arguments)]
fn fill_content(
    rng: &mut Rng,
    plane: &mut [u16],
    off: usize,
    stride: usize,
    cols: usize,
    rows: usize,
    class: usize,
    bd: u8,
) {
    let maxv = (1i64 << bd) - 1;
    let base = (rng.next() % (1 << bd)) as i64;
    for r in 0..rows {
        for cx in 0..cols {
            let v: i64 = match class {
                0 => base,                                              // flat (all-zero hist)
                1 => (rng.next() % (1 << bd)) as i64,                   // noise
                2 => base + 3 * cx as i64,                              // vertical edges (dy=0)
                3 => base + 3 * r as i64,                               // horizontal edges (dx=0)
                4 => base + 2 * (cx as i64 + r as i64),                 // diagonal
                _ => base + ((cx / 4 + r / 4) % 2) as i64 * (maxv / 2), // checker
            };
            plane[off + r * stride + cx] = v.clamp(0, maxv) as u16;
        }
    }
}

#[test]
fn generate_hog_matches_c() {
    c::ref_init();
    let mut rng = Rng(0x50be_1097_ad1e_0714);
    const STRIDE: usize = 160;
    let mut nonzero_hists = 0usize;
    for case in 0..900 {
        let bd: u8 = [8, 10, 12][case % 3];
        let class = case % 6;
        // rows/cols: full block dims and frame-edge-clipped (non-multiple)
        // values, incl. degenerate 2/3 (interior walk empty -> all-zero hist).
        let dims = [2usize, 3, 4, 6, 8, 12, 16, 30, 32, 64];
        let rows = dims[(rng.next() as usize) % dims.len()];
        let cols = dims[(rng.next() as usize) % dims.len()];
        let off = 8 * STRIDE + 8;
        let mut plane = vec![0u16; STRIDE * 96];
        for v in plane.iter_mut() {
            *v = (rng.next() % (1 << bd)) as u16;
        }
        fill_content(&mut rng, &mut plane, off, STRIDE, cols, rows, class, bd);

        let got = generate_hog(&plane, off, STRIDE, rows, cols);
        let want = c::ref_generate_hog(&plane, off, STRIDE, rows, cols, bd);
        for b in 0..HOG_BINS {
            assert_eq!(
                got[b].to_bits(),
                want[b].to_bits(),
                "hist[{b}] {} vs {} case={case} bd={bd} class={class} {rows}x{cols}",
                got[b],
                want[b],
            );
        }
        if got.iter().any(|&v| v != 0.0) {
            nonzero_hists += 1;
        }
    }
    assert!(nonzero_hists > 400, "nonzero histograms: {nonzero_hists}");
}

#[test]
fn prune_intra_mode_with_hog_matches_c() {
    c::ref_init();
    let mut rng = Rng(0xd09f_00d5_2026_0714);
    const STRIDE: usize = 160;
    let mut some_pruned = 0usize;
    let mut none_pruned = 0usize;
    let mut all_pruned = 0usize;
    let mut clipped_cases = 0usize;
    for case in 0..400 {
        let bd: u8 = [8, 10, 12][case % 3];
        let bsize = [0usize, 3, 4, 6, 9, 12][case % 6];
        let class = (rng.next() as usize) % 6;
        // Speed-0 threshold -1.2 plus sweeps around the score range so the
        // <= boundary and both mask polarities get exercised.
        let th = match case % 4 {
            0 => -1.2f32,
            1 => -6.0,
            2 => 6.0,
            _ => (rng.range(-40, 41) as f32) / 10.0,
        };
        const BLK_W: [usize; 22] = [
            4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
        ];
        const BLK_H: [usize; 22] = [
            4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
        ];
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        // Frame-edge overhang on some cases (1/8-pel negative edges).
        let (right_edge, bottom_edge) = if case % 5 == 4 && bw >= 8 {
            clipped_cases += 1;
            (-(8 * (bw as i32 / 2)), -(8 * (bh as i32 / 4).max(1)))
        } else {
            (1 << 12, 1 << 12)
        };
        let off = 8 * STRIDE + 8;
        let mut plane = vec![0u16; STRIDE * 96];
        for v in plane.iter_mut() {
            *v = (rng.next() % (1 << bd)) as u16;
        }
        fill_content(&mut rng, &mut plane, off, STRIDE, bw, bh, class, bd);

        let mut got = [false; 13];
        prune_intra_mode_with_hog_y(
            &plane,
            off,
            STRIDE,
            bsize,
            right_edge,
            bottom_edge,
            th,
            &mut got,
        );
        let want = c::ref_prune_intra_mode_with_hog_y(
            &plane,
            off,
            STRIDE,
            bsize,
            right_edge,
            bottom_edge,
            bd,
            th,
        );
        assert_eq!(
            got, want,
            "mask case={case} bsize={bsize} bd={bd} class={class} th={th}"
        );
        let n = got.iter().filter(|&&b| b).count();
        if n == 0 {
            none_pruned += 1;
        } else if n == 8 {
            all_pruned += 1;
        } else {
            some_pruned += 1;
        }
    }
    assert!(some_pruned > 60, "partial prunes: {some_pruned}");
    assert!(none_pruned > 20, "no-prune cases: {none_pruned}");
    assert!(all_pruned > 20, "all-pruned cases: {all_pruned}");
    assert!(clipped_cases > 30, "edge-clipped cases: {clipped_cases}");
}

/// CHROMA HOG prune (`prune_intra_mode_with_hog_uv`, is_chroma=1) vs an
/// independently-grouped reference built from the SAME REAL C pieces
/// (`ref_generate_hog` + `ref_hog_nn_predict`) plus C's `collect_hog_data`
/// chroma path: rows/cols = (edge-clipped LUMA dims) `>> ss`, then every bin
/// scaled by `(1 + ss_x) * (1 + ss_y)`. This is the exact math C runs at
/// `intra_mode_search.c:959-972` for `chroma_intra_pruning_with_hog` (the
/// speed-3 delta). The reference groups the `>> ss` with explicit parens so an
/// operator-precedence slip (the shift binding to only the else-branch) in the
/// port's function would flip interior-block dims and FAIL here.
#[test]
fn prune_intra_mode_with_hog_uv_matches_c() {
    let mut rng = Rng(0x51ce_d00d_1234_9f01);
    const STRIDE: usize = 160;
    const BLK_W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const BLK_H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    let mut some_pruned = 0usize;
    let mut none_pruned = 0usize;
    let mut clipped_cases = 0usize;
    // 4:2:0 (1,1), 4:2:2 (1,0) and 4:4:4 (0,0) exercise every `(ss_x, ss_y)`
    // scale/shift combination.
    for case in 0..900usize {
        let bd = [8u8, 10, 12][case % 3];
        // Only bsizes that are chroma-searched with angle-delta in practice are
        // >= 8x8, but the function is size-agnostic — sweep the rect + square set.
        let bsize = [3usize, 4, 5, 6, 7, 8, 9, 12][case % 8];
        let (ss_x, ss_y) = [(1usize, 1usize), (1, 0), (0, 0)][case % 3];
        let class = (rng.next() as usize) % 6;
        let th = match case % 4 {
            0 => -1.2f32, // the intra-frame level-2 threshold (speed 3)
            1 => -0.6,    // the intra-frame level-3 threshold
            2 => 6.0,
            _ => (rng.range(-40, 41) as f32) / 10.0,
        };
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let (right_edge, bottom_edge) = if case % 5 == 4 && bw >= 16 {
            clipped_cases += 1;
            (-(8 * (bw as i32 / 2)), -(8 * (bh as i32 / 4).max(1)))
        } else {
            (1 << 12, 1 << 12)
        };
        let off = 8 * STRIDE + 8;
        let mut plane = vec![0u16; STRIDE * 160];
        for v in plane.iter_mut() {
            *v = (rng.next() % (1 << bd)) as u16;
        }
        fill_content(&mut rng, &mut plane, off, STRIDE, bw, bh, class, bd);

        // Reference: C's collect_hog_data chroma path, explicitly parenthesized.
        let clip_w = if right_edge >= 0 {
            bw as i32
        } else {
            (right_edge >> 3) + bw as i32
        };
        let clip_h = if bottom_edge >= 0 {
            bh as i32
        } else {
            (bottom_edge >> 3) + bh as i32
        };
        let cols_ref = (clip_w >> ss_x) as usize;
        let rows_ref = (clip_h >> ss_y) as usize;
        let mut hist = c::ref_generate_hog(&plane, off, STRIDE, rows_ref, cols_ref, bd);
        let scale = ((1 + ss_x) * (1 + ss_y)) as f32;
        for b in hist.iter_mut() {
            *b *= scale;
        }
        let scores = c::ref_hog_nn_predict(&hist, true);
        let mut want = [false; 13];
        for mode in 1..=8usize {
            if scores[mode - 1] <= th {
                want[mode] = true;
            }
        }

        let mut got = [false; 13];
        prune_intra_mode_with_hog_uv(
            &plane,
            off,
            STRIDE,
            bsize,
            ss_x,
            ss_y,
            right_edge,
            bottom_edge,
            th,
            &mut got,
        );
        assert_eq!(
            got, want,
            "chroma mask case={case} bsize={bsize} bd={bd} ss=({ss_x},{ss_y}) class={class} th={th}"
        );
        let n = got.iter().filter(|&&b| b).count();
        if n == 0 {
            none_pruned += 1;
        } else {
            some_pruned += 1;
        }
    }
    assert!(some_pruned > 40, "chroma partial/all prunes: {some_pruned}");
    assert!(none_pruned > 20, "chroma no-prune cases: {none_pruned}");
    assert!(
        clipped_cases > 20,
        "chroma edge-clipped cases: {clipped_cases}"
    );
}
