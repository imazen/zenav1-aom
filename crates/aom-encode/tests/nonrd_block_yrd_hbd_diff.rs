//! KB-20 — differential harness for the **hbd** arm of `av1_block_yrd`
//! (`av1/encoder/nonrd_opt.c:126`, the `use_hbd` branch + `update_yrd_loop_vars_hbd`).
//!
//! ## What this gates
//! The port's `block_yrd_hbd` composes four kernels that are each already
//! differentially locked on their own (`hadamard_diff`, `quantize_fp_diff`,
//! `hbd_dist_diff`). What this harness gates is the COMPOSITION, which is where
//! a port of this shape actually goes wrong:
//!
//! * the scan/iscan PAIRING — `default_scan_8x8_transpose` +
//!   `av1_default_iscan_8x8_transpose` for TX_8X8 and
//!   `default_scan_fp_16x16_transpose` + `av1_default_iscan_fp_16x16_transpose`
//!   for TX_16X16. The 16x16 **fp** pair is NOT the **lp** pair the lowbd arm
//!   uses (the fp Hadamard carries `aom_hadamard_16x16_c`'s extra
//!   AVX2-matching column shift), and libaom's own comments say each pair "has
//!   to be used together". The C oracle derives its EOB from the forward SCAN
//!   while the port's quantizer derives its from the ISCAN, so a mispaired
//!   table shows up here immediately;
//! * the sub-block walk (`block_step` / `step` / the `max_blocks_*` edge
//!   clamps) and which slice length each kernel sees (`step << 4`);
//! * `update_yrd_loop_vars_hbd`'s `eob_cost` / `rate` / `dist` accumulation and
//!   the `2*(bd-8)` shift inside `av1_highbd_block_error`;
//! * the final `rate <<= 2 + AV1_PROB_COST_SHIFT; rate += eob_cost << SHIFT`.
//!
//! The reference is built ONLY out of real exported C functions
//! (`aom_hadamard_{8x8,16x16}_c`, `av1_quantize_fp_c`, `aom_satd_c`,
//! `av1_highbd_block_error_c`), so it is an oracle of the same class the rest
//! of the tree uses — not a second transcription of the port.
//!
//! ## What it deliberately does NOT gate, and why
//! `av1_quantize_fp_c` is **not** the function real aomenc runs: every SIMD
//! tier is a 16-bit kernel that narrows `tran_low_t` on load and computes
//! `dqcoeff` in 16 bits, and the tiers stop agreeing with `_c` (and with each
//! other — NEON truncates, x86 saturates) once a coefficient or a
//! `qcoeff * dequant` leaves the `int16` range. `aom_hadamard_16x16` reaches
//! +-65534, so the hbd nonrd estimate leaves that range routinely at bd10/bd12.
//! See `nonrd_pickmode::quantize_fp_dispatched` for the measurement.
//!
//! Therefore the walk differential below runs on **8-bit-magnitude residuals**,
//! where every tier and `_c` provably agree — and it ASSERTS that agreement
//! precondition rather than assuming it. The out-of-range regime's only honest
//! oracle is real aomenc itself: that is
//! `config_permutations::speed_nonrd_hbd_byte_identity` (bd10/bd12 x cq x
//! cpu-used 8/9, 24 cells, hard byte-identity).

use aom_encode::nonrd_pickmode::{
    AV1_DEFAULT_ISCAN_8X8_TRANSPOSE, AV1_DEFAULT_ISCAN_FP_16X16_TRANSPOSE,
    DEFAULT_SCAN_8X8_TRANSPOSE, DEFAULT_SCAN_FP_16X16_TRANSPOSE, DEFAULT_SCAN_LP_16X16_TRANSPOSE,
    block_yrd_hbd, quantize_fp_dispatched,
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
    /// Residual of `bits` magnitude: `[-(2^bits - 1), 2^bits - 1]`.
    fn diff(&mut self, bits: u32) -> i16 {
        let range = (1i32 << bits) - 1;
        ((self.next() % (2 * range as u64 + 1)) as i32 - range) as i16
    }
}

/// `get_msb` (aom_dsp/bitops.h).
fn get_msb(n: u32) -> i32 {
    31 - n.leading_zeros() as i32
}

/// The scan/iscan pair `av1_block_yrd`'s hbd arm selects for a clamped tx size.
fn hbd_scans(tx_size: usize) -> (&'static [i16], &'static [i16]) {
    match tx_size {
        1 => (
            &DEFAULT_SCAN_8X8_TRANSPOSE,
            &AV1_DEFAULT_ISCAN_8X8_TRANSPOSE,
        ),
        2 => (
            &DEFAULT_SCAN_FP_16X16_TRANSPOSE,
            &AV1_DEFAULT_ISCAN_FP_16X16_TRANSPOSE,
        ),
        _ => unreachable!("clamped to TX_8X8 / TX_16X16"),
    }
}

/// The real quantizer rows at a bit depth and qindex — `av1_build_quantizer` +
/// `set_q_index`, the pair the encoder installs into `MACROBLOCK_PLANE`.
fn q_rows(bd: u8, qindex: usize) -> ([i16; 8], [i16; 8], [i16; 8]) {
    let mut quants = aom_dsp::quant::Quants::zeroed();
    let mut deq = aom_dsp::quant::Dequants::zeroed();
    aom_dsp::quant::av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
    let rows = aom_dsp::quant::set_q_index(&quants, &deq, qindex, 0);
    (*rows.round_fp, *rows.quant_fp, *rows.dequant)
}

/// `av1_block_yrd`'s hbd arm, rebuilt from the REAL exported C kernels.
///
/// Also returns the largest `|coeff|` and `|dqcoeff|` it saw, so the caller can
/// assert the inputs stayed inside the range where `av1_quantize_fp_c` and the
/// SIMD tiers agree.
#[allow(clippy::too_many_arguments)]
fn c_block_yrd_hbd(
    diff: &[i16],
    bw4: usize,
    max_blocks_wide: usize,
    max_blocks_high: usize,
    tx_size: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
    bd: u8,
) -> ((i32, i64, bool), i32, i32) {
    let diff_stride = 4 * bw4;
    let block_step = 1usize << tx_size;
    let q2 = [quant_fp[0], quant_fp[1]];
    let r2 = [round_fp[0], round_fp[1]];
    let d2 = [dequant[0], dequant[1]];
    let (scan, _iscan) = hbd_scans(tx_size);

    let (mut rate, mut dist, mut eob_cost) = (0i32, 0i64, 0i32);
    let mut skippable = true;
    let (mut max_coeff, mut max_dq) = (0i32, 0i32);
    let mut r = 0usize;
    while r < max_blocks_high {
        let mut cc = 0usize;
        while cc < max_blocks_wide {
            let src = &diff[(r * diff_stride + cc) * 4..];
            let coeff = c::ref_hadamard(1 << (tx_size + 2), src, diff_stride);
            let (qcoeff, dqcoeff, eob) = c::ref_quantize_fp(0, &coeff, &r2, &q2, &d2, scan);
            max_coeff = max_coeff.max(coeff.iter().map(|v| v.abs()).max().unwrap());
            max_dq = max_dq.max(dqcoeff.iter().map(|v| v.abs()).max().unwrap());
            let ncoeffs = eob as usize;
            skippable &= ncoeffs == 0;
            eob_cost += get_msb(ncoeffs as u32 + 1);
            if ncoeffs == 1 {
                rate += qcoeff[0].abs();
            } else if ncoeffs > 1 {
                rate += c::ref_satd(&qcoeff);
            }
            dist += c::ref_highbd_block_error(&coeff, &dqcoeff, bd).0 >> 2;
            cc += block_step;
        }
        r += block_step;
    }
    // AV1_PROB_COST_SHIFT = 9.
    let rate = (rate << (2 + 9)) + (eob_cost << 9);
    ((rate, dist, skippable), max_coeff, max_dq)
}

/// The square leaf shapes the KEY VBP tree can stamp, as
/// `(num_4x4_w, num_4x4_h, clamped tx_size)`: BLOCK_8X8 -> TX_8X8;
/// 16X16 / 32X32 / 64X64 -> TX_16X16 after `AOMMIN(mi->tx_size, TX_16X16)`.
const SHAPES: &[(usize, usize, usize)] = &[(2, 2, 1), (4, 4, 2), (8, 8, 2), (16, 16, 2)];

#[test]
fn block_yrd_hbd_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0x_4b20_0bad_c0de_0001);
    let mut checked = 0usize;
    let mut skippable_seen = 0usize;
    let mut coded_seen = 0usize;
    for &bd in &[10u8, 12] {
        for &(bw4, bh4, tx) in SHAPES {
            for iter in 0..600 {
                // 8-bit residual magnitude: `aom_hadamard_16x16` then peaks at
                // 2 * 64 * 255 = 32640, inside int16, so the `_c` oracle IS the
                // dispatched kernel here (asserted below).
                let diff: Vec<i16> = (0..bw4 * bh4 * 16).map(|_| rng.diff(8)).collect();
                let qindex = (rng.next() % 256) as usize;
                let (round_fp, quant_fp, dequant) = q_rows(bd, qindex);
                // Frame-edge clamps: full extent most of the time, a clipped
                // right/bottom edge otherwise (`max_blocks_*` is what the C
                // `mb_to_*_edge >> 5` arithmetic produces).
                let (mbw, mbh) = if iter % 4 == 3 && bw4 > 2 {
                    (bw4 - (1 << tx), bh4 - (1 << tx))
                } else {
                    (bw4, bh4)
                };
                let got = block_yrd_hbd(
                    &diff, bw4, bh4, mbw, mbh, tx, &round_fp, &quant_fp, &dequant, bd,
                );
                let (want, max_coeff, max_dq) =
                    c_block_yrd_hbd(&diff, bw4, mbw, mbh, tx, &round_fp, &quant_fp, &dequant, bd);
                assert!(
                    max_coeff <= i16::MAX as i32 && max_dq <= i16::MAX as i32,
                    "the agreement precondition broke (|coeff| {max_coeff}, |dqcoeff| \
                     {max_dq}): outside int16 `av1_quantize_fp_c` is NOT the kernel real \
                     aomenc runs, so it is not a valid oracle — shrink the residual range"
                );
                assert_eq!(
                    got, want,
                    "block_yrd_hbd mismatch: bd={bd} shape={bw4}x{bh4} tx={tx} \
                     clamps=({mbw},{mbh}) qindex={qindex} iter={iter}"
                );
                if got.2 {
                    skippable_seen += 1;
                } else {
                    coded_seen += 1;
                }
                checked += 1;
            }
        }
    }
    assert!(checked >= 4_800, "the hbd block_yrd grid shrank: {checked}");
    // Anti-vacuity: both the all-zero-eob and the coded regimes must occur, or
    // the walk's rate/eob_cost accumulation was never exercised.
    assert!(
        skippable_seen > 100 && coded_seen > 100,
        "degenerate grid: {skippable_seen} skippable / {coded_seen} coded cells"
    );
}

/// `quantize_fp_dispatched` must reduce EXACTLY to `av1_quantize_fp_c` while
/// every coefficient and every `qcoeff * dequant` fits in `int16` — that is the
/// regime where libaom's own C and SIMD tiers are contractually identical, and
/// it is what makes the model safe to use at the one call site that needs it.
#[test]
fn quantize_fp_dispatched_reduces_to_c_inside_int16() {
    c::ref_init();
    let mut rng = Rng(0x_4b20_0bad_c0de_0002);
    let mut agreed = 0usize;
    for &bd in &[10u8, 12] {
        for &n in &[64usize, 256] {
            let (scan, iscan) = hbd_scans(if n == 64 { 1 } else { 2 });
            for _ in 0..3_000 {
                let qindex = (rng.next() % 256) as usize;
                let (round_fp, quant_fp, dequant) = q_rows(bd, qindex);
                let (r2, q2, d2) = (
                    [round_fp[0], round_fp[1]],
                    [quant_fp[0], quant_fp[1]],
                    [dequant[0], dequant[1]],
                );
                // int16-range coefficients only.
                let coeff: Vec<i32> = (0..n).map(|_| i32::from(rng.diff(14))).collect();
                let (qw, dqw, eobw) = c::ref_quantize_fp(0, &coeff, &r2, &q2, &d2, scan);
                if dqw.iter().any(|v| v.abs() > i16::MAX as i32) {
                    continue; // outside the agreement regime — see the module docs
                }
                let (mut qg, mut dqg) = (vec![0i32; n], vec![0i32; n]);
                let eobg =
                    quantize_fp_dispatched(&coeff, &r2, &q2, &d2, scan, iscan, &mut qg, &mut dqg);
                assert_eq!(
                    (eobg, &qg, &dqg),
                    (eobw, &qw, &dqw),
                    "bd={bd} n={n} qindex={qindex}"
                );
                agreed += 1;
            }
        }
    }
    assert!(
        agreed > 5_000,
        "too few in-range samples exercised: {agreed}"
    );
}

/// TEETH for the model's existence: outside `int16` the dispatched kernel and
/// `av1_quantize_fp_c` genuinely disagree, so calling `_c` at this site would be
/// a silent bitstream defect rather than a stylistic choice.
///
/// (Arch-independent as a statement — every SIMD tier narrows — even though
/// *which* answer differs between the NEON and x86 tiers.)
#[test]
fn quantize_fp_dispatched_differs_from_c_outside_int16() {
    c::ref_init();
    let mut rng = Rng(0x_4b20_0bad_c0de_0003);
    let (scan, iscan) = hbd_scans(2);
    let n = 256usize;
    // qindex 255 at bd12 — the KB-20 cell's own regime.
    let (round_fp, quant_fp, dequant) = q_rows(12, 255);
    let (r2, q2, d2) = (
        [round_fp[0], round_fp[1]],
        [quant_fp[0], quant_fp[1]],
        [dequant[0], dequant[1]],
    );
    let mut differed = 0usize;
    let trials = 2_000usize;
    for _ in 0..trials {
        // `aom_hadamard_16x16`'s real output range: up to +-65534.
        let coeff: Vec<i32> = (0..n)
            .map(|_| (rng.next() % 131_069) as i32 - 65_534)
            .collect();
        let (qw, dqw, eobw) = c::ref_quantize_fp(0, &coeff, &r2, &q2, &d2, scan);
        let (mut qg, mut dqg) = (vec![0i32; n], vec![0i32; n]);
        let eobg = quantize_fp_dispatched(&coeff, &r2, &q2, &d2, scan, iscan, &mut qg, &mut dqg);
        if (eobg, &qg, &dqg) != (eobw, &qw, &dqw) {
            differed += 1;
        }
    }
    assert!(
        differed * 2 > trials,
        "the dispatched model agreed with av1_quantize_fp_c on {} of {trials} \
         out-of-int16 blocks — if that is genuinely true on this ISA, KB-20's \
         root cause needs re-measuring",
        trials - differed
    );
}

/// The scan tables the hbd arm introduces are transcriptions of
/// `av1/encoder/nonrd_opt.h`; this pins the two properties a typo breaks, and
/// the property libaom's own comments demand (each scan/iscan pair "has to be
/// used together").
#[test]
fn hbd_transpose_scans_are_inverse_permutations() {
    let pairs: [(&[i16], &[i16], &str); 2] = [
        (
            &DEFAULT_SCAN_8X8_TRANSPOSE,
            &AV1_DEFAULT_ISCAN_8X8_TRANSPOSE,
            "8x8",
        ),
        (
            &DEFAULT_SCAN_FP_16X16_TRANSPOSE,
            &AV1_DEFAULT_ISCAN_FP_16X16_TRANSPOSE,
            "fp_16x16",
        ),
    ];
    for (scan, iscan, tag) in pairs {
        let n = scan.len();
        assert_eq!(iscan.len(), n, "{tag}: scan/iscan length");
        let mut seen = vec![false; n];
        for &s in scan {
            let s = s as usize;
            assert!(s < n && !seen[s], "{tag}: scan is not a permutation");
            seen[s] = true;
        }
        for (i, &s) in scan.iter().enumerate() {
            assert_eq!(
                iscan[s as usize] as usize, i,
                "{tag}: iscan is not the inverse of scan at scan index {i}"
            );
        }
    }
    // The lp and fp 16x16 scans are DIFFERENT tables — pairing the hbd arm with
    // the lp scan (the obvious copy-paste error) must not silently be a no-op.
    assert_ne!(
        DEFAULT_SCAN_FP_16X16_TRANSPOSE, DEFAULT_SCAN_LP_16X16_TRANSPOSE,
        "the fp and lp 16x16 transposed scans must differ (aom_hadamard_16x16_c \
         carries an extra column shift that aom_hadamard_lp_16x16_c does not)"
    );
}
