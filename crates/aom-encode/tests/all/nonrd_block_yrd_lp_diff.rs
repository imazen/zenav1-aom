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
    fdct4x4_lp, hadamard_lp_8x8, hadamard_lp_16x16, quantize_lp, satd_lp,
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
        // TX_4X4 (the coded-lossless arm) takes the NORMAL scan
        // `av1_scan_orders[TX_4X4][DCT_DCT]` — nonrd_opt.c:185 + :248-250.
        0 => aom_dsp::txb::scan(0, 0),
        1 => &DEFAULT_SCAN_8X8_TRANSPOSE,
        2 => &DEFAULT_SCAN_LP_16X16_TRANSPOSE,
        _ => unreachable!("clamped to TX_4X4 / TX_8X8 / TX_16X16"),
    }
}

/// The lowbd forward transform `av1_block_yrd` runs for a clamped tx size —
/// `aom_hadamard_lp_{8x8,16x16}` at TX_8X8/TX_16X16, and `aom_fdct4x4_lp` at
/// TX_4X4 (nonrd_opt.c:258-263). REAL exported C symbols throughout.
fn c_lp_forward(tx_size: usize, src: &[i16], stride: usize) -> Vec<i16> {
    match tx_size {
        0 => c::ref_fdct4x4_lp(src, stride),
        n => c::ref_hadamard_lp(1 << (n + 2), src, stride),
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

/// **`aom_fdct4x4_lp`, port vs the exported `_c`** — the TX_4X4 forward
/// transform `av1_block_yrd`'s `default:` arm runs, reached only when
/// `cm->features.coded_lossless` forces `select_tx_mode` to `ONLY_4X4`
/// (rdopt_utils.h:392). Until 2026-08-03 that arm was an `unimplemented!()` and
/// this kernel had never been executed by anything.
///
/// The port's transcription carried a `// HANDOFF: verify the final rounding
/// loop of aom_fdct4x4_lp_c`. It is verified — fwd_txfm.c:143-146 — and this
/// test is what makes that a measurement rather than a re-reading: the loop is
/// `(v + 1) >> 2`, so dropping it scales every coefficient by 4.
#[test]
fn fdct4x4_lp_matches_c() {
    c::ref_init();
    let mut rng = Rng(0x_4b12_0000_0006);
    let (mut checked, mut peak) = (0usize, 0i32);
    for iter in 0..2_000 {
        // Include the extreme corners: the +-255 checkerboard maximises the
        // second pass (where the C range comment's 16-bit bound is tightest),
        // and an all-zero block exercises the `if (i == 0 && in_high[0])`
        // nonzero-bias branch from the other side.
        let src: Vec<i16> = match iter {
            0 => (0..16).map(|i| if (i / 4 + i % 4) % 2 == 0 { 255 } else { -255 }).collect(),
            1 => vec![0i16; 16],
            2 => vec![255i16; 16],
            3 => vec![-255i16; 16],
            n if n % 2 == 0 => (0..16).map(|_| rng.residual()).collect(),
            _ => correlated_residual(&mut rng, 4, 4, 40),
        };
        let want = c::ref_fdct4x4_lp(&src, 4);
        let mut got = [0i16; 16];
        fdct4x4_lp(&src, &mut got, 4);
        assert_eq!(
            got.to_vec(),
            want,
            "aom_fdct4x4_lp iter {iter}: the port's coefficients differ from the \
             exported C kernel's (src {src:?})"
        );
        peak = peak.max(want.iter().map(|v| i32::from(*v).abs()).max().unwrap());
        checked += 1;
    }
    assert!(checked >= 2_000, "only {checked} blocks compared");
    // The bd8 bound this kernel's ISA-independence argument rests on: the
    // second pass tops out at 32654, inside int16. Assert the grid got near it
    // rather than assuming the argument was exercised (playbook §2).
    assert!(
        (8_000..=32_767).contains(&peak),
        "peak |coeff| was {peak}: the grid either never loaded the transform or \
         left the int16 range the lowbd arm is supposed to be bounded to"
    );
}

/// **`aom_fdct4x4_lp` is NOT ISA-conditional over its reachable domain.** The
/// NEON/SSE2 tiers hold every intermediate in int16 where `_c` uses int32
/// `in_high[]`/`step[]`, so they agree with `_c` only inside int16 — which the
/// 9-bit lowbd residual guarantees (see `fdct4x4_lp`'s doc comment for the
/// arithmetic). This measures that on the tier this build actually has, and is
/// the reason the lowbd TX_4X4 arm calls the `_c` transcription directly while
/// its hbd twin needs `fdct4x4_dispatched`.
#[test]
fn fdct4x4_lp_tiers_agree_over_the_reachable_range() {
    c::ref_init();
    let mut rng = Rng(0x_4b12_0000_0007);
    let mut peak = 0i32;
    for iter in 0..2_000 {
        let src: Vec<i16> = match iter {
            0 => (0..16).map(|i| if (i / 4 + i % 4) % 2 == 0 { 255 } else { -255 }).collect(),
            1 => vec![255i16; 16],
            2 => vec![-255i16; 16],
            n if n % 2 == 0 => (0..16).map(|_| rng.residual()).collect(),
            _ => correlated_residual(&mut rng, 4, 4, 200),
        };
        let want = c::ref_fdct4x4_lp(&src, 4);
        let simd = c::ref_fdct4x4_lp_simd(&src, 4);
        assert_eq!(
            want, simd,
            "aom_fdct4x4_lp iter {iter}: the SIMD tier and _c disagree over the \
             bd8 residual domain — the lowbd TX_4X4 arm would then need the \
             cfg-dispatched treatment its hbd twin has"
        );
        peak = peak.max(want.iter().map(|v| i32::from(*v).abs()).max().unwrap());
    }
    assert!(peak >= 8_000, "the grid never loaded the transform (peak {peak})");
    assert_eq!(
        c::REF_FDCT4X4_SIMD_IS_DISTINCT,
        cfg!(any(target_arch = "aarch64", target_arch = "x86_64")),
        "the aom_fdct4x4 tier availability model disagrees with the target"
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
        // tx 0 is the coded-lossless TX_4X4 arm (nonrd_opt.c:246-263).
        for &tx in &[0usize, 1, 2] {
            let n = 1usize << (2 * (tx + 2));
            let scan = lp_scan(tx);
            for iter in 0..200 {
                let src = correlated_residual(&mut rng, 1 << (tx + 2), 1 << (tx + 2), 60);
                let coeff = c_lp_forward(tx, &src, 1 << (tx + 2));

                let iscan = lp_iscan(tx);
                let (wq, wdq, weob) =
                    c::ref_quantize_lp(&coeff, &round_fp, &quant_fp, &dequant, scan, &iscan);
                let (mut gq, mut gdq) = (vec![0i16; n], vec![0i16; n]);
                let geob = quantize_lp(
                    &coeff, n, &round_fp, &quant_fp, &mut gq, &mut gdq, &dequant, scan, &iscan,
                );
                // The DISPATCHED port vs the runtime-dispatched C SIMD —
                // the tier the encoder actually runs, not just `_c`.
                let (sq, sdq, seob) =
                    c::ref_quantize_lp_simd(&coeff, &round_fp, &quant_fp, &dequant, scan, &iscan);
                assert_eq!(seob, geob, "q{qindex} tx{tx} iter {iter}: eob vs C simd");
                assert_eq!(sq, gq, "q{qindex} tx{tx} iter {iter}: qcoeff vs C simd");
                assert_eq!(sdq, gdq, "q{qindex} tx{tx} iter {iter}: dqcoeff vs C simd");
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
                    c::ref_satd_lp_simd(&gq),
                    satd_lp(&gq, n),
                    "q{qindex} tx{tx} iter {iter}: aom_satd_lp vs C simd"
                );
                assert_eq!(
                    c::ref_block_error_lp(&coeff, &gdq),
                    block_error_lp(&coeff, &gdq, n),
                    "q{qindex} tx{tx} iter {iter}: av1_block_error_lp"
                );
                assert_eq!(
                    c::ref_block_error_lp_simd(&coeff, &gdq),
                    block_error_lp(&coeff, &gdq, n),
                    "q{qindex} tx{tx} iter {iter}: av1_block_error_lp vs C simd"
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
            let coeff = c_lp_forward(tx_size, src, diff_stride);
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
/// The square leaf shapes the KEY VBP tree stamps, as
/// `(num_4x4_w, num_4x4_h, clamped tx_size)`. The `tx 0` rows are the
/// CODED-LOSSLESS ones: `select_tx_mode` returns `ONLY_4X4` there
/// (rdopt_utils.h:392), so `mi->tx_size` is TX_4X4 at every bsize and a
/// 64x64 leaf walks 16x16 = 256 txbs.
const SHAPES: &[(usize, usize, usize)] = &[
    (2, 2, 1),
    (4, 4, 2),
    (8, 8, 2),
    (16, 16, 2),
    (1, 1, 0),
    (2, 2, 0),
    (4, 4, 0),
    (16, 16, 0),
];

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
// Tier agreement — the check the SIMD landing needs
// ---------------------------------------------------------------------------

/// The iscan `av1_block_yrd` hands `av1_quantize_lp` for a clamped tx size —
/// `av1_default_iscan_8x8_transpose` /
/// `av1_default_iscan_lp_16x16_transpose` / `scan_order->iscan`
/// (nonrd_opt.c:185, :222-246). Computed as the inverse of [`lp_scan`]'s
/// permutation rather than transcribed: `iscan[scan[i]] = i` is the
/// definition, so the inverse cannot carry a transcription slip.
fn lp_iscan(tx_size: usize) -> Vec<i16> {
    let scan = lp_scan(tx_size);
    let mut iscan = vec![0i16; scan.len()];
    for (i, &rc) in scan.iter().enumerate() {
        iscan[rc as usize] = i as i16;
    }
    iscan
}

/// **`av1_quantize_lp`'s tiers genuinely differ — this test finds WHERE and
/// proves the port cannot reach it.**
///
/// `_c` walks `scan` order computing in `int` and tests `tmp != 0` (the
/// quantized magnitude). The specialised tiers walk raster order reading
/// `iscan`, and differ from `_c` in two places:
///
/// 1. **abs(-32768) wraps.** Every SIMD abs is lane-width
///    (`_mm{,256}_abs_epi16`, NEON `vabsq`, SSE2's `(c^sign)-sign` in i16)
///    so `-32768` stays `-32768`; `_c`'s int abs gives 32768. The qcoeff,
///    dqcoeff AND eob all move.
/// 2. **The eob test's subject.** `_sse2` tests `dqcoeff != 0` —
///    `qcoeff*dequant` is an i16 product that can WRAP: `q*d == 65536`
///    reads as zero to `_sse2` while `_c`/`_avx2`/`_neon` (the latter two
///    test the magnitude `abs_qcoeff > 0`, identical to `_c`'s `tmp`) count
///    it nonzero. So `_sse2` alone can emit a smaller eob.
///
/// Whether either is observable is a domain question, and this test measures
/// it rather than arguing it: (a) an exhaustive arithmetic hunt over EVERY
/// real quantizer row (256 qindex x dc/ac lanes) for the `(tmp, dequant)`
/// pairs that make `q*d` wrap, each candidate verified against the real C
/// kernels at the raster position where the eob divergence would be largest;
/// (b) a dense sweep of the int16 boundary including `-32768`; (c) the
/// transform-realistic grid. Every divergence found is then checked against
/// the transform's output bound, which `lp_hadamard_tiers_agree_*` and
/// `fdct4x4_lp_tiers_agree_*` measure at <= ~32654 — if the divergent set
/// needs a larger |coeff|, the tiers agree over the whole reachable domain.
#[test]
fn lp_quantize_tiers_agree_over_the_reachable_range() {
    c::ref_init();
    let mut quants = aom_dsp::quant::Quants::zeroed();
    let mut deq = aom_dsp::quant::Dequants::zeroed();
    aom_dsp::quant::av1_build_quantizer(8, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);

    // ---- (a) the wrap hunt -------------------------------------------
    // For each row, `tmp` ranges 1..=tmp_max where
    // `tmp = (|c| + round) * quant_fp >> 16`. `dqcoeff = tmp*dequant` wraps
    // (i16) iff `tmp*dequant >= 32768` — that is when _sse2's `dqcoeff != 0`
    // test could ever disagree with `_c`'s `tmp != 0` (needs == 0 mod 2^16,
    // i.e. >= 65536) — and when a NEGATIVE dqcoeff could oppose a positive
    // coeff inside `block_error_lp`'s wrapping i16 diff. For each
    // wrap-producing tmp, invert the quantize to the |coeff| range that
    // produces it and verify against the kernel's own arithmetic.
    //
    // Measured result this test encodes: over EVERY real quantizer row the
    // largest tmp*dequant is exactly 32768 (qindex 0 dc: tmp 8192 x d 4),
    // and producing it needs |coeff| >= 32766 — above the lp transforms'
    // ~32654 output bound. So `tmp*d` stays inside [0, 32767] for every
    // reachable input: dqcoeff never wraps, carries coeff's sign, and is
    // nonzero iff tmp is. _sse2's eob test, _c's, and _avx2/_neon's
    // magnitude test therefore ALL agree on the reachable domain — which is
    // why the port's scalar arm (the _c model) is also correct against the
    // runtime-dispatched kernel, and why the v3 port may use either shape.
    let (mut wrap_inputs, mut eob_wrap_inputs, mut min_wrap_coeff) = (0usize, 0usize, i32::MAX);
    let mut max_prod_seen = 0i64;
    for qindex in 0..256usize {
        for plane in 0..3usize {
            let (round_fp, quant_fp, dequant) = {
                let rows = aom_dsp::quant::set_q_index(&quants, &deq, qindex, plane);
                (*rows.round_fp, *rows.quant_fp, *rows.dequant)
            };
            for lane in 0..2usize {
                let (r, fp, d) = (
                    i32::from(round_fp[lane]),
                    i32::from(quant_fp[lane]),
                    i32::from(dequant[lane]),
                );
                if fp <= 0 || d <= 0 {
                    continue;
                }
                let tmp_max = (32767 + r) * fp >> 16;
                for t in 1..=tmp_max {
                    let prod = t as i64 * d as i64;
                    max_prod_seen = max_prod_seen.max(prod);
                    if prod < 32768 {
                        continue;
                    }
                    // |coeff| values producing this tmp:
                    // t <= (a+r)*fp/65536 < t+1  =>  a in [ceil(t*2^16/fp)-r, ...).
                    let lo = (t * 65536 + fp - 1) / fp - r;
                    let hi = ((t + 1) * 65536 + fp - 1) / fp - 1 - r;
                    for a in [lo, lo + 1, hi - 1, hi] {
                        // Verify this |c| really produces tmp == t under the
                        // kernel's own arithmetic.
                        let tmp = ((a + r).clamp(-32768, 32767) * fp) >> 16;
                        if tmp != t || a > i16::MAX as i32 {
                            continue;
                        }
                        min_wrap_coeff = min_wrap_coeff.min(a);
                        wrap_inputs += 1;
                        if prod % 65536 == 0 {
                            eob_wrap_inputs += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "wrap hunt: {wrap_inputs} realizable wrap inputs, \
         min |coeff| needed {min_wrap_coeff}, max tmp*d {max_prod_seen}, \
         wrap-to-zero candidates {eob_wrap_inputs}"
    );
    // The structural bound, asserted: no realizable wrap input ever exists
    // inside int16, and even the unrealizable hunt never reaches 65536.
    assert_eq!(
        eob_wrap_inputs, 0,
        "a (tmp, dequant) row reached tmp*d == 0 mod 2^16 — _sse2's eob test \
         CAN diverge from _c's there, and the port's scalar arm is no longer \
         a model of the runtime kernel"
    );
    assert!(
        min_wrap_coeff > 32654,
        "a dqcoeff wrap needs only |coeff| = {min_wrap_coeff} — inside the lp \
         transforms' ~32654 bound, so the i16 tier semantics are reachable"
    );

    // ---- (b) the int16 boundary + realistic grid vs the real kernels ----
    let mut rng = Rng(0x_4b12_0000_0009);
    let mut divergences: Vec<String> = Vec::new();
    for &qindex in &[0usize, 60, 128, 200, 255] {
        let (round_fp, quant_fp, dequant) = q_rows(qindex);
        for &tx in &[0usize, 1, 2] {
            let n = 1usize << (2 * (tx + 2));
            let scan = lp_scan(tx);
            let iscan = lp_iscan(tx);
            for &v in &[-32768i16, -32767, -31000, 31000, 32640, 32654, 32766, 32767] {
                for &pos in &[scan[0] as usize, scan[n / 2] as usize, scan[n - 1] as usize] {
                    let mut coeff = vec![0i16; n];
                    coeff[pos] = v;
                    let cw =
                        c::ref_quantize_lp(&coeff, &round_fp, &quant_fp, &dequant, scan, &iscan);
                    let sw = c::ref_quantize_lp_simd(
                        &coeff, &round_fp, &quant_fp, &dequant, scan, &iscan,
                    );
                    // -32768 is the one KNOWN divergence (the wrapping abs);
                    // it is above the transform bound and must be the only one.
                    if cw != sw && v != -32768 {
                        divergences.push(format!(
                            "q{qindex} tx{tx} pos {pos} coeff {v}: _c eob {} \
                             vs simd eob {}",
                            cw.2, sw.2
                        ));
                    }
                    #[cfg(target_arch = "x86_64")]
                    {
                        let ew = c::ref_quantize_lp_sse2(
                            &coeff, &round_fp, &quant_fp, &dequant, scan, &iscan,
                        );
                        if cw != ew && v != -32768 {
                            divergences.push(format!(
                                "q{qindex} tx{tx} pos {pos} coeff {v}: _c eob {} \
                                 vs _sse2 eob {}",
                                cw.2, ew.2
                            ));
                        }
                    }
                }
            }
            for iter in 0..400 {
                let side = 1usize << (tx + 2);
                let src = correlated_residual(&mut rng, side, side, 60);
                let coeff = c_lp_forward(tx, &src, side);
                let cw = c::ref_quantize_lp(&coeff, &round_fp, &quant_fp, &dequant, scan, &iscan);
                let sw =
                    c::ref_quantize_lp_simd(&coeff, &round_fp, &quant_fp, &dequant, scan, &iscan);
                if cw != sw {
                    divergences.push(format!(
                        "q{qindex} tx{tx} iter {iter}: transform-reachable coeff diverged"
                    ));
                }
            }
        }
    }
    assert!(
        divergences.is_empty(),
        "lp quantize tiers diverge on {} reachable inputs:\n{}",
        divergences.len(),
        divergences[..divergences.len().min(12)].join("\n")
    );
}

/// **`aom_satd_lp` / `av1_block_error_lp` tier agreement.** Both SIMD tiers
/// wrap where `_c` widens: `satd_lp`'s `_mm_abs_epi16` gives -32768 for input
/// -32768 (scalar: +32768), and `block_error_lp`'s `_mm_sub_epi16(dq, c)`
/// wraps when |dq - c| >= 32768 (scalar computes in i64). The second needs a
/// |coeff|+|dqcoeff| > 32767 opposite-sign pair; with |coeff| <= 32640 the
/// only reachable wraps need |dqcoeff| > 127 of the opposite sign, which the
/// quantize wrap hunt bounds. This measures the boundary directly.
#[test]
fn lp_satd_block_error_tiers_agree_over_the_reachable_range() {
    c::ref_init();
    let mut rng = Rng(0x_4b12_0000_000a);
    let mut quants = aom_dsp::quant::Quants::zeroed();
    let mut deq = aom_dsp::quant::Dequants::zeroed();
    aom_dsp::quant::av1_build_quantizer(8, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);

    // satd: dense at the boundary, random inside. `-32768` is the KNOWN
    // divergence — every SIMD abs wraps it while `_c` widens — and it is the
    // ONLY one (abs is the sole wrap point, and |v| <= 32767 can't wrap it).
    for &n in &[16usize, 64, 256] {
        {
            let coeff = vec![-32768i16; n];
            assert_ne!(
                c::ref_satd_lp(&coeff),
                c::ref_satd_lp_simd(&coeff),
                "aom_satd_lp at all--32768 is the known wrap; if the tiers now \
                 AGREE there, this test's reachable-domain argument needs re-derivation"
            );
        }
        for v in -32767i32..=-31000 {
            let coeff = vec![v as i16; n];
            assert_eq!(
                c::ref_satd_lp(&coeff),
                c::ref_satd_lp_simd(&coeff),
                "aom_satd_lp n={n} all={v}: tiers disagree"
            );
        }
        for v in 31000i32..=32767 {
            let coeff = vec![v as i16; n];
            assert_eq!(
                c::ref_satd_lp(&coeff),
                c::ref_satd_lp_simd(&coeff),
                "aom_satd_lp n={n} all={v}: tiers disagree"
            );
        }
        for _ in 0..400 {
            // Full i16 range EXCEPT -32768 — that lane is the measured-known
            // wrap and is probed explicitly above.
            let coeff: Vec<i16> = (0..n)
                .map(|_| (((rng.next() % 65535) as i32) - 32767) as i16)
                .collect();
            assert_eq!(
                c::ref_satd_lp(&coeff),
                c::ref_satd_lp_simd(&coeff),
                "aom_satd_lp n={n} random: tiers disagree"
            );
        }
    }

    // ---- block_error_lp ----------------------------------------------
    // TWO wrap classes exist in the SIMD tiers, both measured below:
    //  (i)   `_mm_sub_epi16(dq, c)` wraps iff |dq - c| >= 32768 — needs
    //        OPPOSITE-sign lanes (or the -32768 edge). Unreachable: the
    //        quantize hunt above shows dqcoeff always carries coeff's sign.
    //  (ii)  the `_mm_madd_epi16` pair sums and the i32 accumulation wrap
    //        mod 2^32 — `_c` sums in i64. At n = 256 this needs
    //        rms |dq - c| >= ~2896. Reachable? `dq = tmp*d` with
    //        `tmp = (|c| + round)*fp >> 16`, `fp = floor(2^16/d)`, so
    //        |dq - |c|| <= ~d/2 + |c|*d/2^16 ~= 1200 + 1200 at the largest
    //        dequant — far under. Measure the real bound below.
    //
    // First: demonstrate class (i) exists (informational — every pair in it
    // is unreachable by construction).
    let mut div_pairs = 0usize;
    for c in (30000i32..=32767)
        .step_by(16)
        .chain((-32768i32..=-30000).step_by(16))
    {
        for dq in [
            -32768i32, -32760, -32000, -31000, 31000, 32000, 32760, 32767,
        ] {
            if (dq - c) as i16 as i32 == dq - c {
                continue;
            }
            let coeff = vec![c as i16; 16];
            let dqc = vec![dq as i16; 16];
            if c::ref_block_error_lp(&coeff, &dqc) != c::ref_block_error_lp_simd(&coeff, &dqc) {
                div_pairs += 1;
            }
        }
    }
    eprintln!(
        "block_error: {div_pairs} opposite-sign wrap pairs confirmed \
               divergent (unreachable: quantize_lp never emits them)"
    );

    // Second: measure the real reachable |dq - c| bound using the scalar
    // quantize formula (proven == `av1_quantize_lp_c` by
    // `lp_quantize_satd_block_error_match_c` over the whole reachable grid).
    let mut max_dqc_err = 0i64;
    for qindex in 0..256usize {
        for plane in 0..3usize {
            let (round_fp, quant_fp, dequant) = {
                let rows = aom_dsp::quant::set_q_index(&quants, &deq, qindex, plane);
                (*rows.round_fp, *rows.quant_fp, *rows.dequant)
            };
            for lane in 0..2usize {
                let (r, fp, d) = (
                    i32::from(round_fp[lane]),
                    i32::from(quant_fp[lane]),
                    i32::from(dequant[lane]),
                );
                for a in (0i32..=32767).step_by(4) {
                    let tmp = ((a + r).clamp(-32768, 32767) * fp) >> 16;
                    let dq = (tmp as i64 * d as i64) as i16 as i64; // i16 assign
                    max_dqc_err = max_dqc_err.max((dq - a as i64).abs());
                }
            }
        }
    }
    eprintln!("block_error: max reachable |dqcoeff - coeff| = {max_dqc_err}");
    // The bound must clear BOTH wrap classes with margin: the i16 sub needs
    // < 32768; the i32 accumulation at n = 256 (the lp path's largest txb)
    // needs 256 * err^2 < 2^31.
    assert!(
        max_dqc_err < 2896,
        "reachable |dqcoeff - coeff| = {max_dqc_err} — the SIMD i32 \
         accumulation can wrap mod 2^32 at n = 256, so the scalar arm is no \
         longer a safe model of the runtime kernel"
    );

    // Third: real (coeff, dqcoeff) pairs through both tiers — boundary coeffs
    // quantized by the REAL _c kernel, not synthetic pairs.
    let mut divergences = 0usize;
    for &qindex in &[0usize, 60, 128, 200, 255] {
        let (round_fp, quant_fp, dequant) = q_rows(qindex);
        for &tx in &[0usize, 1, 2] {
            let n = 1usize << (2 * (tx + 2));
            let scan = lp_scan(tx);
            let iscan = lp_iscan(tx);
            // Boundary single-coeff blocks.
            for &v in &[-32767i16, -31000, -1000, 1000, 31000, 32640, 32767] {
                for &pos in &[scan[0] as usize, scan[n - 1] as usize] {
                    let mut coeff = vec![0i16; n];
                    coeff[pos] = v;
                    let (_q, dq, _e) =
                        c::ref_quantize_lp(&coeff, &round_fp, &quant_fp, &dequant, scan, &iscan);
                    if c::ref_block_error_lp(&coeff, &dq) != c::ref_block_error_lp_simd(&coeff, &dq)
                    {
                        divergences += 1;
                    }
                }
            }
            // Transform-reachable blocks.
            for _ in 0..300 {
                let side = 1usize << (tx + 2);
                let src = correlated_residual(&mut rng, side, side, 60);
                let coeff = c_lp_forward(tx, &src, side);
                let (_q, dq, _e) =
                    c::ref_quantize_lp(&coeff, &round_fp, &quant_fp, &dequant, scan, &iscan);
                if c::ref_block_error_lp(&coeff, &dq) != c::ref_block_error_lp_simd(&coeff, &dq) {
                    divergences += 1;
                }
                assert_eq!(
                    c::ref_satd_lp(&coeff),
                    c::ref_satd_lp_simd(&coeff),
                    "q{qindex} tx{tx}: satd tiers disagree on a \
                     transform-reachable block"
                );
            }
        }
    }
    assert_eq!(
        divergences, 0,
        "block_error_lp tiers diverge on {divergences} real-quantizer \
         (coeff, dqcoeff) pairs"
    );
}

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
