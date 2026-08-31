//! Small exported encoder helpers that have no home in a larger module — a
//! coefficient-dropout pass, an intra RD penalty, and the two flat-block
//! predicates the intrabc/hash search uses.
//!
//! Each is a whole exported C function that the inter-encode surface inventory
//! listed as unported, and each is individually tier-1 gated. They are grouped
//! by size rather than by subsystem; when one of their subsystems lands
//! properly, its entry should move there.
//!
//! | Rust | C |
//! |---|---|
//! | [`dropout_qcoeff_num`] | `av1_dropout_qcoeff_num` (encoder/encodemb.c:168) |
//! | [`get_intra_cost_penalty`] | `av1_get_intra_cost_penalty` (encoder/rd.c) |
//! | [`hash_is_horizontal_perfect`] | `av1_hash_is_horizontal_perfect` (encoder/hash_motion.c) |
//! | [`hash_is_vertical_perfect`] | `av1_hash_is_vertical_perfect` (hash_motion.c) |
//!
//! # Differential coverage
//! `tests/enc_misc_diff.rs`, tier 1 against the real exported C.

/// `DROPOUT_COEFF_MAX` (encodemb.c:109) — the largest coefficient magnitude the
/// dropout pass is allowed to zero.
const DROPOUT_COEFF_MAX: i32 = 2;
/// `DROPOUT_CONTINUITY_MAX` (encodemb.c:113) — how many consecutive non-zeros
/// force the run to be kept.
const DROPOUT_CONTINUITY_MAX: i32 = 2;

/// `av1_dropout_qcoeff_num` (encoder/encodemb.c:168): zero out short runs of
/// small coefficients that sit between long runs of zeros, then shrink the eob.
///
/// Returns the new `eob`. `qcoeff` and `dqcoeff` are edited in place; `scan` is
/// the block's scan order and `max_eob` its coefficient count.
///
/// The state machine has three interacting counters and is easy to get subtly
/// wrong. Points the differential pins:
///
/// * a coefficient with `|q| > DROPOUT_COEFF_MAX` resets **all** the counters
///   AND the pending index, and advances `eob` — a large coefficient is not
///   merely "not dropped", it re-anchors the run;
/// * a non-zero seen before `dropout_num_before` zeros have accumulated resets
///   `count_zeros_before` to 0 rather than leaving it;
/// * the trailing-zero credit `max_eob - eob` is added only on the LAST
///   iteration and only when a candidate run is pending (`idx != -1`);
/// * `eob` is only advanced by the keep paths, so a block whose whole tail is
///   dropped ends with a smaller eob than any index it visited.
///
/// # Contract: `dropout_num_after >= 1`
/// C has no guard here, and at `dropout_num_after == 0` its
/// `count_zeros_after >= dropout_num_after` test fires on the first iteration
/// while the pending index is still `-1`, so its zeroing loop runs
/// `for (j = -1; j <= i; ++j) qcoeff[scan[j]]` — an out-of-bounds read and
/// write. (Measured: the oracle returns a fully-zeroed block there.) For
/// `dropout_num_after >= 1` the OOB cannot occur, because `count_zeros_after`
/// only ever increments on paths guarded by `idx != -1`.
///
/// `av1_dropout_qcoeff`, the only caller, computes both counts as
/// `multiplier * CLIP(max(tx_w, tx_h), 16, 32)` with `multiplier` in `[2, 8]`,
/// so it never passes less than 32. This port asserts the contract rather than
/// reproducing undefined behaviour or silently diverging from it.
#[allow(clippy::too_many_arguments)]
pub fn dropout_qcoeff_num(
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    eob_in: usize,
    max_eob: usize,
    scan: &[i16],
    dropout_num_before: i32,
    dropout_num_after: i32,
) -> usize {
    assert!(
        dropout_num_after >= 1,
        "av1_dropout_qcoeff_num is undefined at dropout_num_after == 0 (C indexes \
         scan[-1] there); its only caller never passes less than 32"
    );
    // Early return, exactly as C's.
    if eob_in == 0
        || eob_in as i32 <= dropout_num_before
        || (max_eob as i32) <= dropout_num_before + dropout_num_after
    {
        return eob_in;
    }

    let mut count_zeros_before = 0i32;
    let mut count_zeros_after = 0i32;
    let mut count_nonzeros = 0i32;
    let mut idx: i32 = -1;
    let mut eob = 0usize;

    for i in 0..eob_in {
        let scan_idx = scan[i] as usize;
        if qcoeff[scan_idx].abs() > DROPOUT_COEFF_MAX {
            count_zeros_before = 0;
            count_zeros_after = 0;
            idx = -1;
            eob = i + 1;
        } else if qcoeff[scan_idx] == 0 {
            if idx == -1 {
                count_zeros_before += 1;
            } else {
                count_zeros_after += 1;
            }
        } else if count_zeros_before >= dropout_num_before {
            if idx == -1 {
                idx = i as i32;
            }
            count_nonzeros += 1;
        } else {
            count_zeros_before = 0;
            eob = i + 1;
        }

        if count_nonzeros > DROPOUT_CONTINUITY_MAX {
            count_zeros_before = 0;
            count_zeros_after = 0;
            count_nonzeros = 0;
            idx = -1;
            eob = i + 1;
        }

        if idx != -1 && i == eob_in - 1 {
            count_zeros_after += (max_eob - eob_in) as i32;
        }

        if count_zeros_after >= dropout_num_after {
            for j in idx as usize..=i {
                qcoeff[scan[j] as usize] = 0;
                dqcoeff[scan[j] as usize] = 0;
            }
            count_zeros_before += i as i32 - idx + 1;
            count_zeros_after = 0;
            count_nonzeros = 0;
        } else if i == eob_in - 1 {
            eob = i + 1;
        }
    }

    eob
}

/// `av1_get_intra_cost_penalty` (encoder/rd.c): the RD penalty added to an
/// intra mode inside an inter frame, from the DC quantizer.
///
/// The three bit depths are three different formulas, not one scaled — bd 8 is
/// `20 * q`, bd 10 is `5 * q`, and bd 12 is `round(5 * q, 2)`.
#[must_use]
pub fn get_intra_cost_penalty(qindex: i32, qdelta: i32, bit_depth: u8) -> i32 {
    let q = i32::from(aom_dsp::quant::av1_dc_quant_qtx(qindex, qdelta, bit_depth));
    match bit_depth {
        8 => 20 * q,
        10 => 5 * q,
        12 => (5 * q + 2) >> 2,
        _ => panic!("av1_get_intra_cost_penalty: bit_depth must be 8, 10 or 12, got {bit_depth}"),
    }
}

/// `av1_hash_is_horizontal_perfect` (encoder/hash_motion.c): every row of the
/// `block_size` square is a constant.
///
/// Note C compares each sample against `p[0]` — the row's FIRST sample, freshly
/// re-based per row — so this is "each row is flat", not "the whole block is
/// flat".
pub fn hash_is_horizontal_perfect(
    plane: &[u16],
    stride: usize,
    block_size: usize,
    x_start: usize,
    y_start: usize,
) -> bool {
    let base = y_start * stride + x_start;
    for i in 0..block_size {
        let row = base + i * stride;
        for j in 1..block_size {
            if plane[row + j] != plane[row] {
                return false;
            }
        }
    }
    true
}

/// `av1_hash_is_vertical_perfect` (hash_motion.c): every column is a constant.
///
/// C's loop here does NOT re-base per row — it indexes `p[j * stride + i]`
/// against `p[i]` off the single block origin, so the outer variable `i` is the
/// COLUMN. Transcribing it with the same loop shape as the horizontal twin
/// gives the wrong function.
pub fn hash_is_vertical_perfect(
    plane: &[u16],
    stride: usize,
    block_size: usize,
    x_start: usize,
    y_start: usize,
) -> bool {
    let base = y_start * stride + x_start;
    for i in 0..block_size {
        for j in 1..block_size {
            if plane[base + j * stride + i] != plane[base + i] {
                return false;
            }
        }
    }
    true
}
