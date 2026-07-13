//! Differential harness for the frame-header quantization params
//! (encode_quantization) vs C libaom's control flow (driven through the real
//! aom_wb primitives), plus an independent spec-layout anchor.

use aom_entropy::header::{encode_quantization, QuantParamsHeader};
use aom_entropy::wb::WriteBitBuffer;
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
    fn dq(&mut self) -> i32 {
        // delta-q is a 7-bit inverse-signed field: [-63, 63], often 0.
        if self.next().is_multiple_of(3) { 0 } else { (self.next() % 127) as i32 - 63 }
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

#[test]
fn encode_quantization_matches_c() {
    let mut rng = Rng(0x9a17_c0de_a11a_0009);
    for _ in 0..200_000 {
        let qp = QuantParamsHeader {
            base_qindex: rng.range(0, 256),
            y_dc_delta_q: rng.dq(),
            u_dc_delta_q: rng.dq(),
            u_ac_delta_q: rng.dq(),
            v_dc_delta_q: rng.dq(),
            v_ac_delta_q: rng.dq(),
            using_qmatrix: rng.next().is_multiple_of(2),
            qmatrix_level_y: rng.range(0, 16),
            qmatrix_level_u: rng.range(0, 16),
            qmatrix_level_v: rng.range(0, 16),
        };
        let num_planes = if rng.next().is_multiple_of(4) { 1 } else { 3 };
        let separate_uv = rng.next().is_multiple_of(2);

        let mut wb = WriteBitBuffer::new();
        encode_quantization(&mut wb, &qp, num_planes, separate_uv);
        let got = wb.bytes().to_vec();
        let want = c::ref_encode_quantization(
            qp.base_qindex, qp.y_dc_delta_q, qp.u_dc_delta_q, qp.u_ac_delta_q, qp.v_dc_delta_q,
            qp.v_ac_delta_q, qp.using_qmatrix, qp.qmatrix_level_y, qp.qmatrix_level_u,
            qp.qmatrix_level_v, num_planes, separate_uv,
        );
        assert_eq!(got, want, "encode_quantization {qp:?} np={num_planes} sep={separate_uv}");
    }
}

#[test]
fn encode_quantization_spec_anchor() {
    // Monochrome (num_planes=1), all deltas 0, no qm: base_qindex byte + two 0
    // bits (y_dc absent-flag, using_qmatrix) => [base, 0x00].
    let qp = QuantParamsHeader {
        base_qindex: 0x5a,
        y_dc_delta_q: 0,
        u_dc_delta_q: 0,
        u_ac_delta_q: 0,
        v_dc_delta_q: 0,
        v_ac_delta_q: 0,
        using_qmatrix: false,
        qmatrix_level_y: 0,
        qmatrix_level_u: 0,
        qmatrix_level_v: 0,
    };
    let mut wb = WriteBitBuffer::new();
    encode_quantization(&mut wb, &qp, 1, false);
    assert_eq!(wb.bytes(), &[0x5a, 0x00]);
}
