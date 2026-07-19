//! Quantization-matrix (QM) selection for the dequantizer — the decode-side of
//! `av1/common/quant_common.c`'s `av1_get_iqmatrix` / `av1_qm_init`.
//!
//! When a frame signals `using_qmatrix`, each 2-D-transform coefficient is
//! dequantized with a per-position inverse-QM weight instead of the flat step:
//! `dqcoeff = (qcoeff * dequant * iqmatrix[pos] + 16) >> 5` (folded into
//! [`aom_dsp::txb::dequant_txb`] via `get_dqv`). This module supplies the `iqmatrix`
//! slice for a given (qm level, plane, tx size, tx type).
//!
//! The inverse-QM bases `iwt_matrix_ref[NUM_QM_LEVELS-1][2][QM_TOTAL_SIZE]` are
//! ported verbatim from libaom (generated into [`qm_tables`]); `av1_qm_init`
//! packs each tx size's matrix contiguously in the 3344-entry array in
//! `TX_SIZES_ALL` order, reusing the `av1_get_adjusted_tx_size` matrix for the
//! 64-point sizes. The per-tx offsets [`QM_OFFSET`] reproduce that packing.

use crate::qm_tables::IWT_MATRIX_REF;

/// `NUM_QM_LEVELS` (`quant_common.h`): 16 QM sets. Level `NUM_QM_LEVELS - 1`
/// (15) is the flat/no-op matrix and is signalled as "no matrix" (`None`).
pub(crate) const NUM_QM_LEVELS: usize = 16;

/// `IDTX` (`enums.h`): the identity 2-D transform. `is_2d_transform(tx_type)`
/// is `tx_type < IDTX`; 1-D and identity transforms take the flat matrix.
const IDTX: usize = 9;

/// Byte offset into an `iwt_matrix_ref[q][c]` row (`QM_TOTAL_SIZE` = 3344) of
/// each `TX_SIZE`'s matrix, indexed by the RAW tx size. Reproduces
/// `av1_qm_init`'s packing: iterate `TX_SIZES_ALL` in enum order, appending
/// `tx_size_2d[t]` bytes only for the canonical sizes (`t ==
/// av1_get_adjusted_tx_size(t)`); the 64-point sizes reuse their adjusted
/// (32-capped) matrix's offset. Verified to accumulate to exactly 3344.
#[rustfmt::skip]
const QM_OFFSET: [usize; 19] = [
    0, 16, 80, 336, 336, 1360, 1392, 1424, 1552, 1680, 2192, 336, 336, 2704,
    2768, 2832, 3088, 1680, 2192,
];

/// `av1_get_iqmatrix` (`quant_common.c`): the inverse quantization matrix for a
/// coefficient block, or `None` for the flat (no-weighting) case.
///
/// Returns `None` — flat dequant, matching libaom's `giqmatrix[NUM_QM_LEVELS-1]`
/// NULL rows — when the level is the flat top level (QM off, or a lossless
/// segment) or the transform is 1-D / identity (`!is_2d_transform`). Otherwise
/// the `iwt_matrix_ref` slice for `(qm_level, plane>=1 ? chroma : luma,
/// adjusted tx size)`, laid out in raster order over the adjusted transform
/// block — exactly what [`aom_dsp::txb::dequant_txb`] indexes by coefficient
/// position.
// `pub` (via the doc-hidden `pub mod qm`) so the encoder's forward-QM quantizer
// can select the inverse-QM weights from the same `iwt_matrix_ref` bases,
// avoiding a duplicate table. See the module declaration in `lib.rs`.
pub fn iqmatrix(
    qm_level: usize,
    plane: usize,
    tx_size: usize,
    tx_type: usize,
) -> Option<&'static [u8]> {
    // Flat matrix (i.e. no weighting) for the top level and for 1-D / Identity
    // transforms (av1_get_iqmatrix: giqmatrix[NUM_QM_LEVELS-1][0][.] == NULL).
    if qm_level >= NUM_QM_LEVELS - 1 || tx_type >= IDTX {
        return None;
    }
    // Plane group: luma (0) vs chroma (both U and V share the c>=1 bases).
    let c = usize::from(plane >= 1);
    let off = QM_OFFSET[tx_size];
    // Length == tx_size_2d[av1_get_adjusted_tx_size(tx_size)] — the coded
    // region the inverse transform reads (txb_wide/txb_high are adjusted).
    let len = aom_dsp::txb::txb_wide(tx_size) * aom_dsp::txb::txb_high(tx_size);
    Some(&IWT_MATRIX_REF[qm_level][c][off..off + len])
}

#[cfg(test)]
mod tests {
    use super::*;

    // TX_SIZE indices used below (enums.h order).
    const TX_4X4: usize = 0;
    const TX_8X8_ANY: usize = 1;
    const TX_16X16: usize = 2;
    const TX_32X32: usize = 3;
    const TX_64X64: usize = 4;
    // TX_TYPE indices (enums.h order).
    const DCT_DCT: usize = 0;
    const V_DCT: usize = 11; // a 1-D transform (>= IDTX)

    /// The ported table anchors to the exact C `iwt_matrix_ref` bytes: level 0
    /// luma 4x4 begins `32, 43, 73, 97, ...` and level 0 chroma 4x4 differs
    /// (verbatim from quant_common.c). Guards against an extraction shift.
    #[test]
    fn table_anchors_to_c_source() {
        let luma0 = iqmatrix(0, 0, TX_4X4, DCT_DCT).unwrap();
        assert_eq!(&luma0[..4], &[32, 43, 73, 97], "level 0 luma 4x4 head");
        // DC weight (position 0) is the flat 1<<AOM_QM_BITS == 32 at every level.
        for lvl in 0..15 {
            assert_eq!(
                iqmatrix(lvl, 0, TX_16X16, DCT_DCT).unwrap()[0],
                32,
                "DC weight must be flat (32) at level {lvl}"
            );
        }
        // The flat top level (15) selects no matrix, as does level 14+? No —
        // only 15 is flat; 14 is a real (near-flat) matrix.
        assert!(iqmatrix(14, 0, TX_16X16, DCT_DCT).is_some());
    }

    /// `av1_get_iqmatrix` flat cases: the top level and 1-D / identity
    /// transforms take no matrix (`None`).
    #[test]
    fn flat_cases_select_none() {
        // Level NUM_QM_LEVELS-1 (15) is the flat no-op.
        assert!(iqmatrix(15, 0, TX_16X16, DCT_DCT).is_none());
        assert!(iqmatrix(15, 1, TX_8X8_ANY, DCT_DCT).is_none());
        // 1-D / identity transforms are always flat, even at a steep level.
        assert!(iqmatrix(0, 0, TX_16X16, V_DCT).is_none());
        assert!(iqmatrix(0, 0, TX_16X16, IDTX).is_none());
        // 2-D transform at a non-flat level: a real matrix.
        assert!(iqmatrix(0, 0, TX_16X16, DCT_DCT).is_some());
    }

    /// Slice length equals the coded-region area, and the 64-point sizes reuse
    /// their adjusted (32-capped) matrix — i.e. `av1_qm_init`'s aliasing.
    #[test]
    fn lengths_and_64pt_aliasing() {
        assert_eq!(iqmatrix(3, 0, TX_4X4, DCT_DCT).unwrap().len(), 16);
        assert_eq!(iqmatrix(3, 0, TX_16X16, DCT_DCT).unwrap().len(), 256);
        assert_eq!(iqmatrix(3, 0, TX_32X32, DCT_DCT).unwrap().len(), 1024);
        // TX_64X64 aliases TX_32X32 (same slice — offset 336, length 1024).
        let a = iqmatrix(3, 0, TX_32X32, DCT_DCT).unwrap();
        let b = iqmatrix(3, 0, TX_64X64, DCT_DCT).unwrap();
        assert_eq!(a, b, "TX_64X64 must reuse the TX_32X32 matrix");
    }

    /// Luma and chroma use distinct bases (`c >= 1`), so their matrices differ.
    #[test]
    fn luma_chroma_bases_differ() {
        let luma = iqmatrix(0, 0, TX_16X16, DCT_DCT).unwrap();
        let chroma_u = iqmatrix(0, 1, TX_16X16, DCT_DCT).unwrap();
        let chroma_v = iqmatrix(0, 2, TX_16X16, DCT_DCT).unwrap();
        assert_eq!(chroma_u, chroma_v, "U and V share the c>=1 base");
        assert_ne!(luma, chroma_u, "luma and chroma bases must differ");
    }

    /// The effectful proof underpinning the byte-identity gate: for every
    /// non-flat level, dequantizing a unit AC coefficient WITH the matrix
    /// differs from the flat dequant — so a decoder that ignored QM would
    /// produce different pixels than one that applied it. Uses the exact kernel
    /// (`aom_dsp::txb::dequant_txb`) and matrix selection the decode path uses.
    #[test]
    fn qm_dequant_differs_from_flat() {
        const AREA: usize = 1024; // TX_32X32
        let dequant = [100i16, 100];
        for lvl in 0..15usize {
            let iqm = iqmatrix(lvl, 0, TX_32X32, DCT_DCT).unwrap();
            let mut differed = false;
            for pos in 1..AREA {
                let mut qc = vec![0i32; AREA];
                qc[pos] = 3;
                let mut dq_flat = vec![0i32; AREA];
                let mut dq_qm = vec![0i32; AREA];
                aom_dsp::txb::dequant_txb(&qc, &mut dq_flat, TX_32X32, dequant, None, 8);
                aom_dsp::txb::dequant_txb(&qc, &mut dq_qm, TX_32X32, dequant, Some(iqm), 8);
                if dq_flat[pos] != dq_qm[pos] {
                    differed = true;
                    break;
                }
            }
            assert!(
                differed,
                "QM level {lvl} produced flat dequant everywhere (secretly flat)"
            );
        }
    }
}
