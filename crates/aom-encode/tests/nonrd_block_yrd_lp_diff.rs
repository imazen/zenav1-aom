//! KB-12 — differential harness for the **lowbd** arm of `av1_block_yrd`
//! (`av1/encoder/nonrd_opt.c:126`, the `use_hbd == 0` branch +
//! `update_yrd_loop_vars`, `:43`). The twin of `nonrd_block_yrd_hbd_diff.rs`,
//! and the gate whose ABSENCE was KB-12's root.
//!
//! ## Why it exists
//! The lowbd estimate arm's five kernels — `aom_hadamard_lp_8x8`,
//! `aom_hadamard_lp_16x16`, `av1_quantize_lp`, `aom_satd_lp`,
//! `av1_block_error_lp` — were hand-transcribed into
//! `crates/aom-encode/src/nonrd_pickmode.rs` and, unlike every other kernel in
//! this tree, were never locked against the exported C symbol. KB-12 recorded
//! "the whole traced estimate chain matches libaom to the line", which was a
//! READING, not a measurement: `hadamard_lp_8x8` omitted the trailing
//! transpose C performs at `aom_dsp/avg.c:232-236` (*"Extra transpose to match
//! SSE2 behavior"*), so the port's coefficients were the exact TRANSPOSE of
//! libaom's, and `aom_hadamard_lp_16x16`'s were the per-64-quadrant transpose.
//!
//! That defect is nearly invisible by construction, which is why it survived
//! four localization passes: **every consumer of the coefficients except the
//! EOB is order-invariant.** `aom_satd_lp` and `av1_block_error_lp` are sums
//! over the whole array; `eob == 0` (the `skippable` flag) and `eob == 1`
//! (which can only mean the DC, a transpose fixed point) are invariant too.
//! The single quantity that moves is `eob` itself, through
//! `eob_cost += get_msb(eob + 1)` and thence `rate += eob_cost << 9` — a rate
//! perturbation small enough that it only ever flipped near-ties in
//! `av1_nonrd_pick_intra_mode`'s four-mode loop.
//!
//! ## What this gates
//! * each kernel against the REAL exported C symbol (`aom_hadamard_lp_*_c`,
//!   `av1_quantize_lp_c`, `aom_satd_lp_c`, `av1_block_error_lp_c`) — an oracle
//!   of the same class the rest of the tree uses, not a second transcription;
//! * that the specialised SIMD tier agrees with `_c` over the range this call
//!   site can actually reach, so — unlike `aom_hadamard_16x16`
//!   (LIBAOM_UPSTREAM_NOTES A1 / KB-20 root #4) — there is nothing
//!   ISA-conditional to model here;
//! * the COMPOSITION: the sub-block walk, the `max_blocks_*` edge clamps, the
//!   scan choice per clamped tx size, `update_yrd_loop_vars`' accumulation and
//!   the final `rate <<= 2 + AV1_PROB_COST_SHIFT; rate += eob_cost << SHIFT`
//!   (playbook §12 — a green kernel differential licenses the kernel and
//!   nothing else);
//! * TEETH: that the transpose is load-bearing, and *which* output it moves.

use aom_encode::nonrd_pickmode::{
    DEFAULT_SCAN_8X8_TRANSPOSE, DEFAULT_SCAN_LP_16X16_TRANSPOSE, block_error_lp, block_yrd_lowbd,
    hadamard_lp_8x8, hadamard_lp_16x16, quantize_lp, satd_lp,
};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// A bd8 intra residual: `src - pred` with both in `[0, 255]`, i.e. the
    /// 9-bit `[-255, 255]` domain libaom's own range comments assume.
    fn residual(&mut self) -> i16 {
        ((self.next() % 511) as i32 - 255) as i16
    }
}

/// A CORRELATED residual — the shape a real intra prediction leaves behind.
/// White noise is the wrong grid for this kernel: it spreads energy evenly over
/// the 64 coefficients so the EOB lands at the very end of the scan on nearly
/// every block, which is exactly the regime where a reordering of the
/// coefficients cannot change the EOB. (Same lesson as
/// `nonrd_block_yrd_hbd_diff`'s correlated grid, for the opposite reason.)
fn correlated_residual(rng: &mut Rng, w: usize, h: usize, amp: i32) -> Vec<i16> {
    let (gx, gy) = ((rng.next() % 9) as i32 - 4, (rng.next() % 9) as i32 - 4);
    let dc = (rng.next() % 41) as i32 - 20;
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            let n = ((rng.next() % (2 * amp as u64 + 1)) as i32) - amp;
            (dc + gx * x / 4 + gy * y / 4 + n).clamp(-255, 255) as i16
        })
        .collect()
}

/// `get_msb` (aom_dsp/bitops.h).
fn get_msb(n: u32) -> i32 {
    31 - n.leading_zeros() as i32
}

/// The scan `av1_block_yrd`'s LOWBD arm selects for a clamped tx size
/// (nonrd_opt.c:266-295). `av1_quantize_lp_c` ignores its iscan
/// (`(void)iscan`, av1_quantize.c:219) but the C signature demands a pointer.
fn lp_scan(tx_size: usize) -> &'static [i16] {
    match tx_size {
        1 => &DEFAULT_SCAN_8X8_TRANSPOSE,
        2 => &DEFAULT_SCAN_LP_16X16_TRANSPOSE,
        _ => unreachable!("clamped to TX_8X8 / TX_16X16 on the square-leaf grid"),
    }
}

/// The real quantizer rows at bd8 and a qindex — `av1_build_quantizer` +
/// `set_q_index`, the pair the encoder installs into `MACROBLOCK_PLANE`.
fn q_rows(qindex: usize) -> ([i16; 8], [i16; 8], [i16; 8]) {
    let mut quants = aom_dsp::quant::Quants::zeroed();
    let mut deq = aom_dsp::quant::Dequants::zeroed();
    aom_dsp::quant::av1_build_quantizer(8, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
    let rows = aom_dsp::quant::set_q_index(&quants, &deq, qindex, 0);
    (*rows.round_fp, *rows.quant_fp, *rows.dequant)
}

// ---------------------------------------------------------------------------
// Kernel locks
// ---------------------------------------------------------------------------

/// `aom_hadamard_lp_8x8` and `aom_hadamard_lp_16x16`, port vs the exported
/// `_c`, over both a white-noise and a correlated 9-bit residual grid.
#[test]
fn lp_hadamard_matches_c() {
    c::ref_init();
    let mut rng = Rng(0x_4b12_0000_0001);
    let mut checked = 0usize;
    for &n in &[8usize, 16] {
        for iter in 0..800 {
            let src: Vec<i16> = if iter % 2 == 0 {
                (0..n * n).map(|_| rng.residual()).collect()
            } else {
                correlated_residual(&mut rng, n, n, 24)
            };
            let want = c::ref_hadamard_lp(n, &src, n);
            let mut got = vec![0i16; n * n];
            match n {
                8 => hadamard_lp_8x8(&src, n, &mut got),
                _ => hadamard_lp_16x16(&src, n, &mut got),
            }
            assert_eq!(
                got, want,
                "aom_hadamard_lp_{n}x{n} iter {iter}: the port's coefficients \
                 differ from the exported C kernel's"
            );
            checked += 1;
        }
    }
    assert!(checked >= 1_600, "only {checked} blocks compared");
}

/// **`aom_hadamard_lp_*` is NOT ISA-conditional over its reachable domain** —
/// the specialised tier and `_c` agree on every 9-bit residual.
///
/// This is the check LIBAOM_UPSTREAM_NOTES A1 exists for, run on the OTHER
/// kernel: `aom_hadamard_16x16`'s 4-way combine is int32 in `_c`/NEON and
/// int16-with-wrapping on x86, which matters because the hbd estimate pushes
/// it to +-65534. The lp kernels combine in int16 in `_c` too, and the lowbd
/// arm's input is bounded at 9 bits by construction (`src - pred`, both u8), so
/// the 8x8 stage peaks at 255 * 64 = 16320 and the 16x16 combine's
/// `a0 + a1` peaks at 32640 — inside int16 on every tier. The assertion below
/// measures that on the tier this build actually has, and the magnitude bound
/// is asserted rather than assumed.
#[test]
fn lp_hadamard_tiers_agree_over_the_reachable_range() {
    c::ref_init();
    let mut rng = Rng(0x_4b12_0000_0002);
    let mut peak = 0i32;
    for &n in &[8usize, 16] {
        for iter in 0..800 {
            // Sign-correlated content is what pushes the 4-way combine highest
            // (the four 8x8 quadrants must agree in sign to add rather than
            // cancel), so include the worst case: a full-amplitude checkerboard.
            let src: Vec<i16> = if iter == 0 {
                (0..n * n)
                    .map(|i| if (i / n + i % n) % 2 == 0 { 255 } else { -255 })
                    .collect()
            } else if iter % 2 == 0 {
                (0..n * n).map(|_| rng.residual()).collect()
            } else {
                correlated_residual(&mut rng, n, n, 200)
            };
            let want = c::ref_hadamard_lp(n, &src, n);
            let simd = c::ref_hadamard_lp_simd(n, &src, n);
            assert_eq!(
                want, simd,
                "aom_hadamard_lp_{n}x{n} iter {iter}: the SIMD tier and _c \
                 disagree over the bd8 residual domain — this kernel would then \
                 need the cfg-dispatched treatment KB-20 root #4 gave \
                 aom_hadamard_16x16"
            );
            peak = peak.max(want.iter().map(|v| i32::from(*v).abs()).max().unwrap());
        }
    }
    assert!(
        peak >= 16_000,
        "the grid never drove |coeff| above {peak} — it does not reach the \
         range where the tiers could differ, so this test proves nothing \
         (playbook §2)"
    );
    // Non-vacuity, per target rather than per host: aarch64 and x86-64 both have
    // a specialised tier this harness can call unconditionally (NEON; SSE2,
    // which is x86-64 baseline), so on those two the comparison above is real.
    // Anywhere else it is `_c` against itself and this must SAY so rather than
    // report coverage it does not have.
    assert_eq!(
        c::REF_HADAMARD_LP_SIMD_IS_DISTINCT,
        cfg!(any(target_arch = "aarch64", target_arch = "x86_64")),
        "the aom_hadamard_lp tier availability model disagrees with the target: \
         on aarch64 and x86-64 ref_hadamard_lp_simd must be a genuinely \
         different kernel from _c, and elsewhere it must be _c itself"
    );
}

/// `av1_quantize_lp`, `aom_satd_lp` and `av1_block_error_lp` against their
/// exported C symbols, on real quantizer rows across the qindex range.
#[test]
fn lp_quantize_satd_block_error_match_c() {
    c::ref_init();
    let mut rng = Rng(0x_4b12_0000_0003);
    let mut nonzero_eobs = 0usize;
    for &qindex in &[20usize, 60, 128, 200, 255] {
        let (round_fp, quant_fp, dequant) = q_rows(qindex);
        for &tx in &[1usize, 2] {
            let n = 1usize << (2 * (tx + 2));
            let scan = lp_scan(tx);
            for iter in 0..200 {
                let src = correlated_residual(&mut rng, 1 << (tx + 2), 1 << (tx + 2), 60);
                let coeff = c::ref_hadamard_lp(1 << (tx + 2), &src, 1 << (tx + 2));

                let (wq, wdq, weob) =
                    c::ref_quantize_lp(&coeff, &round_fp, &quant_fp, &dequant, scan, scan);
                let (mut gq, mut gdq) = (vec![0i16; n], vec![0i16; n]);
                let geob = quantize_lp(
                    &coeff, n, &round_fp, &quant_fp, &mut gq, &mut gdq, &dequant, scan,
                );
                assert_eq!(weob, geob, "q{qindex} tx{tx} iter {iter}: eob");
                assert_eq!(wq, gq, "q{qindex} tx{tx} iter {iter}: qcoeff");
                assert_eq!(wdq, gdq, "q{qindex} tx{tx} iter {iter}: dqcoeff");
                if weob > 0 {
                    nonzero_eobs += 1;
                }

                assert_eq!(
                    c::ref_satd_lp(&gq),
                    satd_lp(&gq, n),
                    "q{qindex} tx{tx} iter {iter}: aom_satd_lp"
                );
                assert_eq!(
                    c::ref_block_error_lp(&coeff, &gdq),
                    block_error_lp(&coeff, &gdq, n),
                    "q{qindex} tx{tx} iter {iter}: av1_block_error_lp"
                );
            }
        }
    }
    assert!(
        nonzero_eobs >= 100,
        "only {nonzero_eobs} blocks quantized to a nonzero eob — the grid is \
         too coarse to exercise the eob path"
    );
}

// ---------------------------------------------------------------------------
// The composition
// ---------------------------------------------------------------------------

/// `av1_block_yrd`'s lowbd arm rebuilt from the REAL exported C kernels.
/// Returns `(rate, dist, skippable)` plus the eob histogram, so the caller can
/// prove the walk reached the coded (eob > 1) regime.
#[allow(clippy::too_many_arguments)]
fn c_block_yrd_lowbd(
    diff: &[i16],
    bw4: usize,
    max_blocks_wide: usize,
    max_blocks_high: usize,
    tx_size: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
) -> ((i32, i64, bool), usize) {
    let diff_stride = 4 * bw4;
    let block_step = 1usize << tx_size;
    let n = 1usize << (2 * (tx_size + 2));
    let scan = lp_scan(tx_size);

    let (mut rate, mut dist, mut eob_cost) = (0i32, 0i64, 0i32);
    let mut skippable = true;
    let mut coded = 0usize;
    let mut r = 0usize;
    while r < max_blocks_high {
        let mut cc = 0usize;
        while cc < max_blocks_wide {
            let src = &diff[(r * diff_stride + cc) * 4..];
            let coeff = c::ref_hadamard_lp(1 << (tx_size + 2), src, diff_stride);
            let (qcoeff, dqcoeff, eob) =
                c::ref_quantize_lp(&coeff, round_fp, quant_fp, dequant, scan, scan);
            let ncoeffs = eob as usize;
            skippable &= ncoeffs == 0;
            eob_cost += get_msb(ncoeffs as u32 + 1);
            if ncoeffs == 1 {
                rate += i32::from(qcoeff[0]).abs();
            } else if ncoeffs > 1 {
                rate += c::ref_satd_lp(&qcoeff);
                coded += 1;
            }
            dist += c::ref_block_error_lp(&coeff, &dqcoeff) >> 2;
            let _ = n;
            cc += block_step;
        }
        r += block_step;
    }
    // AV1_PROB_COST_SHIFT = 9.
    let rate = (rate << (2 + 9)) + (eob_cost << 9);
    ((rate, dist, skippable), coded)
}

/// The square leaf shapes the KEY VBP tree stamps, as
/// `(num_4x4_w, num_4x4_h, clamped tx_size)`: BLOCK_8X8 -> TX_8X8;
/// 16X16 / 32X32 / 64X64 -> TX_16X16 after `AOMMIN(mi->tx_size, TX_16X16)`.
const SHAPES: &[(usize, usize, usize)] = &[(2, 2, 1), (4, 4, 2), (8, 8, 2), (16, 16, 2)];

#[test]
fn block_yrd_lowbd_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0x_4b12_0000_0004);
    let (mut checked, mut skippable_seen, mut coded_seen, mut clamped_seen) = (0, 0, 0, 0);
    for &qindex in &[20usize, 60, 128, 200, 255] {
        let (round_fp, quant_fp, dequant) = q_rows(qindex);
        for &(bw4, bh4, tx) in SHAPES {
            for iter in 0..120 {
                let (bw, bh) = (bw4 * 4, bh4 * 4);
                let diff = if iter % 3 == 0 {
                    (0..bw * bh).map(|_| rng.residual()).collect()
                } else {
                    correlated_residual(&mut rng, bw, bh, 20 + (iter as i32 % 5) * 30)
                };
                // Exercise the frame-edge clamps too: every third iteration
                // truncates the walk the way `mb_to_right_edge < 0` does.
                let step = 1usize << tx;
                let (mbw, mbh) = if iter % 3 == 2 && bw4 > step {
                    clamped_seen += 1;
                    (bw4 - step, bh4)
                } else {
                    (bw4, bh4)
                };
                let (want, coded) = c_block_yrd_lowbd(
                    &diff, bw4, mbw, mbh, tx, &round_fp, &quant_fp, &dequant,
                );
                let got = block_yrd_lowbd(
                    &diff, bw4, bh4, mbw, mbh, tx, &round_fp, &quant_fp, &dequant,
                );
                assert_eq!(
                    got, want,
                    "block_yrd_lowbd q{qindex} {bw}x{bh} tx{tx} iter {iter} \
                     (walk {mbw}x{mbh}): (rate, dist, skippable)"
                );
                if want.2 {
                    skippable_seen += 1;
                }
                coded_seen += coded;
                checked += 1;
            }
        }
    }
    // Non-vacuity (playbook §2): the grid must reach BOTH the all-zero
    // (skippable) regime and the multi-coefficient regime, and must actually
    // exercise the edge clamp — otherwise it tests one branch of three.
    assert!(checked >= 2_000, "only {checked} walks compared");
    assert!(
        skippable_seen >= 10 && coded_seen >= 200 && clamped_seen >= 100,
        "coverage: {skippable_seen} skippable walks, {coded_seen} coded txbs, \
         {clamped_seen} edge-clamped walks"
    );
}

// ---------------------------------------------------------------------------
// Teeth
// ---------------------------------------------------------------------------

/// **The transpose is load-bearing, and this names exactly what it moves.**
///
/// KB-12's defect was `hadamard_lp_8x8` writing `buffer2` straight out instead
/// of `coeff[i * 8 + j] = buffer2[j * 8 + i]`. Re-create the pre-fix kernel
/// here and assert three things, because the combination is the whole reason
/// the bug hid for four localization passes:
///
/// 1. the pre-fix coefficients ARE the transpose of C's (so this is a
///    reproduction of the defect, not an unrelated perturbation);
/// 2. `aom_satd_lp`, `av1_block_error_lp` and `eob == 0` are IDENTICAL under
///    it — the order-invariant consumers cannot see it;
/// 3. `eob` — and therefore `rate`, through `eob_cost << 9` — DOES move.
#[test]
fn lp_hadamard_transpose_is_load_bearing_and_only_moves_the_eob() {
    c::ref_init();
    /// `hadamard_lp_8x8` exactly as it stood before the KB-12 fix.
    fn pre_fix_lp_8x8(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
        let mut want = vec![0i16; 64];
        hadamard_lp_8x8(src_diff, src_stride, &mut want);
        // The fixed kernel is C's; undo the transpose to recover the old one.
        for i in 0..8 {
            for j in 0..8 {
                coeff[i * 8 + j] = want[j * 8 + i];
            }
        }
    }

    let mut rng = Rng(0x_4b12_0000_0005);
    let (round_fp, quant_fp, dequant) = q_rows(128);
    let scan = lp_scan(1);
    let (mut eob_moved, mut trials) = (0usize, 0usize);
    for iter in 0..4_000 {
        let src = correlated_residual(&mut rng, 8, 8, 20 + (iter as i32 % 7) * 25);
        let want = c::ref_hadamard_lp(8, &src, 8);
        let mut old = vec![0i16; 64];
        pre_fix_lp_8x8(&src, 8, &mut old);
        // (1) the reproduction is faithful.
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(
                    old[i * 8 + j],
                    want[j * 8 + i],
                    "the pre-fix reproduction is not C's transpose"
                );
            }
        }
        let (nq, ndq, neob) =
            c::ref_quantize_lp(&want, &round_fp, &quant_fp, &dequant, scan, scan);
        let (oq, odq, oeob) = c::ref_quantize_lp(&old, &round_fp, &quant_fp, &dequant, scan, scan);
        // (2) every order-invariant consumer is blind to it.
        assert_eq!(
            c::ref_satd_lp(&nq),
            c::ref_satd_lp(&oq),
            "iter {iter}: aom_satd_lp is order-invariant and must not move"
        );
        assert_eq!(
            c::ref_block_error_lp(&want, &ndq),
            c::ref_block_error_lp(&old, &odq),
            "iter {iter}: av1_block_error_lp is order-invariant and must not move"
        );
        assert_eq!(
            neob == 0,
            oeob == 0,
            "iter {iter}: skippable (eob == 0) is order-invariant and must not move"
        );
        // (3) the eob is the one thing that does.
        if neob != oeob {
            eob_moved += 1;
        }
        trials += 1;
    }
    assert!(
        eob_moved >= 100,
        "the pre-fix kernel changed the eob on only {eob_moved} of {trials} \
         blocks — if that is 0 the transpose is inert and the KB-12 fix is not \
         what closed the near-tie class"
    );
    eprintln!("pre-fix transpose moved the eob on {eob_moved}/{trials} blocks");
}
