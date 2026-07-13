//! Uncompressed frame-header components (libaom `av1/encoder/bitstream.c`),
//! written through [`WriteBitBuffer`]. Byte-identical to C libaom. The functions
//! here are `static inline` in libaom, so their oracles are the same control flow
//! driven through the real `aom_wb` primitives (validated by `wb_diff`), plus
//! independent spec-layout anchors in the tests.

use crate::wb::WriteBitBuffer;

/// `write_delta_q`: a present-flag + 7-bit inverse-signed value (0 => just the flag).
fn write_delta_q(wb: &mut WriteBitBuffer, delta_q: i32) {
    if delta_q != 0 {
        wb.write_bit(1);
        wb.write_inv_signed_literal(delta_q, 6);
    } else {
        wb.write_bit(0);
    }
}

/// The `CommonQuantParams` fields the frame-header quantization block reads.
#[derive(Clone, Copy, Debug)]
pub struct QuantParamsHeader {
    pub base_qindex: i32,
    pub y_dc_delta_q: i32,
    pub u_dc_delta_q: i32,
    pub u_ac_delta_q: i32,
    pub v_dc_delta_q: i32,
    pub v_ac_delta_q: i32,
    pub using_qmatrix: bool,
    pub qmatrix_level_y: i32,
    pub qmatrix_level_u: i32,
    pub qmatrix_level_v: i32,
}

/// `encode_quantization`: the frame-header quantization params — base qindex
/// (`QINDEX_BITS`=8), the y/u/v dc/ac delta-qs (u/v only for `num_planes > 1`,
/// with the `diff_uv_delta` and `separate_uv_delta_q` gating), and the quant
/// matrix flag + levels (`QM_LEVEL_BITS`=4).
pub fn encode_quantization(
    wb: &mut WriteBitBuffer,
    qp: &QuantParamsHeader,
    num_planes: usize,
    separate_uv_delta_q: bool,
) {
    wb.write_literal(qp.base_qindex, 8);
    write_delta_q(wb, qp.y_dc_delta_q);
    if num_planes > 1 {
        let diff_uv_delta =
            qp.u_dc_delta_q != qp.v_dc_delta_q || qp.u_ac_delta_q != qp.v_ac_delta_q;
        if separate_uv_delta_q {
            wb.write_bit(diff_uv_delta as u32);
        }
        write_delta_q(wb, qp.u_dc_delta_q);
        write_delta_q(wb, qp.u_ac_delta_q);
        if diff_uv_delta {
            write_delta_q(wb, qp.v_dc_delta_q);
            write_delta_q(wb, qp.v_ac_delta_q);
        }
    }
    wb.write_bit(qp.using_qmatrix as u32);
    if qp.using_qmatrix {
        wb.write_literal(qp.qmatrix_level_y, 4);
        wb.write_literal(qp.qmatrix_level_u, 4);
        if separate_uv_delta_q {
            wb.write_literal(qp.qmatrix_level_v, 4);
        }
    }
}
