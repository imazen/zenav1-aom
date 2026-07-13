//! Encoder composition layer: `av1_xform_quant` (libaom `av1/encoder/encodemb.c`).
//!
//! The per-block encoder workhorse = `av1_xform` (forward 2-D transform) +
//! `av1_quant` (quantize + entropy-context write). Every sub-step is already
//! bit-exact against C in its own crate:
//! - forward transform: [`aom_transform::txfm2d::av1_fwd_txfm2d`]
//! - quantizers: [`aom_quant`] (fp/b, flat/QM — lowbd 8-bit here)
//! - neighbour context: [`aom_txb::txb_entropy_context`]
//!
//! This crate wires them in the exact order/params libaom uses, so a residual
//! block maps to byte-identical (qcoeff, dqcoeff, eob, txb_entropy_ctx). The
//! forward transform is bd=8 (lowbd); highbd composition follows once the highbd
//! forward transform lands.
#![forbid(unsafe_code)]

use aom_quant::{
    aom_quantize_b_no_qmatrix, aom_quantize_b_qm, av1_quantize_fp_no_qmatrix, av1_quantize_fp_qm,
};
use aom_transform::txfm2d::av1_fwd_txfm2d;
use aom_txb::{scan, txb_entropy_context, txb_high, txb_wide};

/// Full (un-adjusted) transform width per `TX_SIZE` — the residual/coeff buffer
/// dimensions the forward transform reads/writes before 64-point repacking.
const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
/// `tx_size_2d[tx_size]` — full pel count, drives `av1_get_tx_scale` (log_scale).
const TX_SIZE_2D: [i32; 19] =
    [16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024];

/// `av1_get_tx_scale(tx_size)` = `(pels > 256) + (pels > 1024)`.
#[inline]
pub fn tx_scale(tx_size: usize) -> i32 {
    let pels = TX_SIZE_2D[tx_size];
    (pels > 256) as i32 + (pels > 1024) as i32
}

/// Which quantizer `av1_quant` dispatches to (`quant_func_list` row). The DC-only
/// fast path (`AV1_XFORM_QUANT_DC`) is not modelled yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    /// `AV1_XFORM_QUANT_FP` — the fast (round-only) quantizer.
    Fp,
    /// `AV1_XFORM_QUANT_B` — the dead-zone (`zbin` + `quant`/`quant_shift`) quantizer.
    B,
}

/// The `[dc, ac]` quantizer parameter tables (the `*_QTX` fields libaom's
/// `MACROBLOCK_PLANE` feeds the quantizer). `zbin`/`quant_shift` are only read by
/// [`QuantKind::B`]; `round` is the fp round for FP and the b round for B.
#[derive(Clone, Copy, Debug)]
pub struct QuantParams<'a> {
    pub zbin: &'a [i16; 2],
    pub round: &'a [i16; 2],
    pub quant: &'a [i16; 2],
    pub quant_shift: &'a [i16; 2],
    pub dequant: &'a [i16; 2],
    /// Per-position quant matrix (`qm`) and its inverse (`iqm`), both indexed by
    /// raster position (length = block area). `None` = flat (no quant matrix).
    pub qm: Option<&'a [u8]>,
    pub iqm: Option<&'a [u8]>,
}

/// Output of [`xform_quant`]: the quantized block plus the propagated context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XformQuantResult {
    pub coeff: Vec<i32>,
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    pub eob: u16,
    pub txb_entropy_ctx: u8,
}

/// `av1_xform_quant` for a single (bd=8) transform block: forward-transform the
/// `residual` (full `TX_W*TX_H`, row-major, stride `TX_W`), quantize with `kind`,
/// and (unless `use_optimize_b`, which defers to the trellis) write the neighbour
/// entropy context. Returns byte-identical output to libaom's `av1_xform` +
/// `av1_quant`.
pub fn xform_quant(
    residual: &[i16],
    tx_size: usize,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    use_optimize_b: bool,
) -> XformQuantResult {
    let full = TX_W[tx_size] * TX_H[tx_size];
    assert_eq!(residual.len(), full, "residual must be full TX_W*TX_H");
    let n_coeffs = txb_wide(tx_size) * txb_high(tx_size); // == av1_get_max_eob
    let log_scale = tx_scale(tx_size);
    let sc = scan(tx_size, tx_type);

    // av1_xform: forward 2-D transform into a full-size buffer (64-point sizes
    // repack their valid area into the first n_coeffs entries in-place).
    let mut coeff = vec![0i32; full];
    av1_fwd_txfm2d(residual, &mut coeff, TX_W[tx_size], tx_type, tx_size);

    // av1_quant: quantize the valid coefficient block.
    let mut qcoeff = vec![0i32; n_coeffs];
    let mut dqcoeff = vec![0i32; n_coeffs];
    let src = &coeff[..n_coeffs];
    let eob = match (kind, qp.qm, qp.iqm) {
        (QuantKind::Fp, Some(qm), Some(iqm)) => av1_quantize_fp_qm(
            qp.round, qp.quant, qp.dequant, log_scale, qm, iqm, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::Fp, _, _) => av1_quantize_fp_no_qmatrix(
            qp.quant, qp.dequant, qp.round, log_scale, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, Some(qm), Some(iqm)) => aom_quantize_b_qm(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, qm, iqm, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, _, _) => aom_quantize_b_no_qmatrix(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
    };

    // av1_quant tail: entropy ctx is deferred to optimize_b when it will run.
    let txb_entropy_ctx = if use_optimize_b {
        0
    } else {
        txb_entropy_context(&qcoeff, tx_size, tx_type, eob as usize)
    };

    XformQuantResult { coeff, qcoeff, dqcoeff, eob, txb_entropy_ctx }
}
