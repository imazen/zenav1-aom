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

#[test]
fn encode_loopfilter_matches_c() {
    use aom_entropy::header::{encode_loopfilter, LoopfilterHeader};
    let mut rng = Rng(0x10f1_c0de_a11a_0009);
    for _ in 0..200_000 {
        let deltas8 = |rng: &mut Rng| -> [i8; 8] {
            let mut a = [0i8; 8];
            for x in &mut a {
                *x = (rng.next() % 127) as i8 - 63;
            }
            a
        };
        let deltas2 = |rng: &mut Rng| -> [i8; 2] {
            [(rng.next() % 127) as i8 - 63, (rng.next() % 127) as i8 - 63]
        };
        // Sometimes make last == current so "changed"/"meaningful" go both ways.
        let ref_deltas = deltas8(&mut rng);
        let last_ref = if rng.next().is_multiple_of(3) { ref_deltas } else { deltas8(&mut rng) };
        let mode_deltas = deltas2(&mut rng);
        let last_mode = if rng.next().is_multiple_of(3) { mode_deltas } else { deltas2(&mut rng) };
        let lf = LoopfilterHeader {
            allow_intrabc: rng.next().is_multiple_of(7),
            filter_level: [rng.range(0, 64), rng.range(0, 64)],
            filter_level_u: rng.range(0, 64),
            filter_level_v: rng.range(0, 64),
            sharpness_level: rng.range(0, 8),
            mode_ref_delta_enabled: rng.next().is_multiple_of(2),
            mode_ref_delta_update: rng.next().is_multiple_of(2),
            ref_deltas,
            mode_deltas,
            last_ref_deltas: last_ref,
            last_mode_deltas: last_mode,
        };
        let num_planes = if rng.next().is_multiple_of(4) { 1 } else { 3 };
        let mut wb = WriteBitBuffer::new();
        encode_loopfilter(&mut wb, &lf, num_planes);
        let got = wb.bytes().to_vec();
        let want = c::ref_encode_loopfilter(
            lf.allow_intrabc, lf.filter_level, lf.filter_level_u, lf.filter_level_v,
            lf.sharpness_level, lf.mode_ref_delta_enabled, lf.mode_ref_delta_update, &lf.ref_deltas,
            &lf.mode_deltas, &lf.last_ref_deltas, &lf.last_mode_deltas, num_planes,
        );
        assert_eq!(got, want, "encode_loopfilter {lf:?} np={num_planes}");
    }
}

#[test]
fn encode_cdef_matches_c() {
    use aom_entropy::header::{encode_cdef, CdefHeader};
    let mut rng = Rng(0xcde1_c0de_a11a_0009);
    for _ in 0..200_000 {
        let cdef_bits = rng.range(0, 4); // nb_cdef_strengths = 1<<cdef_bits (1..8)
        let nb = 1usize << cdef_bits;
        let mut y = [0i32; 8];
        let mut uv = [0i32; 8];
        for k in 0..8 {
            y[k] = rng.range(0, 64);
            uv[k] = rng.range(0, 64);
        }
        let cdef = CdefHeader {
            enable_cdef: rng.next().is_multiple_of(5),
            allow_intrabc: rng.next().is_multiple_of(7),
            cdef_damping: rng.range(3, 7), // damping-3 fits 2 bits => damping 3..6
            cdef_bits,
            nb_cdef_strengths: nb,
            cdef_strengths: y,
            cdef_uv_strengths: uv,
        };
        let num_planes = if rng.next().is_multiple_of(4) { 1 } else { 3 };
        let mut wb = WriteBitBuffer::new();
        encode_cdef(&mut wb, &cdef, num_planes);
        let got = wb.bytes().to_vec();
        let want = c::ref_encode_cdef(cdef.enable_cdef, cdef.allow_intrabc, cdef.cdef_damping, cdef.cdef_bits, nb, &y, &uv, num_planes);
        assert_eq!(got, want, "encode_cdef {cdef:?} np={num_planes}");
    }
}
