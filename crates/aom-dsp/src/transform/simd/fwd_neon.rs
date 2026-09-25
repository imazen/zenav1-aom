//! Verbatim NEON transcriptions of `av1/encoder/arm/av1_fwd_txfm2d_neon.c` —
//! the hot small forward-transform shapes. C's NEON fused kernels are a
//! different structure from the SSE2/AVX2 bodies the sibling `mod.rs`
//! transcriptions mirror: `int16x4`/`int16x8` lane kernels using
//! `vmull_lane`/`vmlal_lane`/`vmlsl_lane` widening multiplies against the
//! q13 cospi/sinpi tables, `vrshrn`/`vqrshrn` narrowing, and
//! `vtrn`/`vzip`-based transposes. These bodies are what C actually dispatches
//! on aarch64, so they are both the faster path and the more faithful oracle.
//!
//! The kernels and drivers below follow the C source function-for-function;
//! each notes its upstream counterpart.

use archmage::intrinsics::aarch64::*;

/// `av1_cospi_arr_q13_data` (av1/common/av1_txfm.c) — i16 cospi constants in
/// the packed (+,−) pair layout the NEON kernels index by `4*j`.
#[rustfmt::skip]
pub(crate) static COSPI_Q13: [[i16; 128]; 4] = [
    [
        5792,  5792,  -5792, -5792, 7568,  3136,  -7568, -3136, 8032,  1600,
        -8032, -1600, 6808,  4552,  -6808, -4552, 8152,  800,   -8152, -800,
        7840,  2376,  -7840, -2376, 7224,  3864,  -7224, -3864, 6336,  5200,
        -6336, -5200, 8184,  400,   -8184, -400,  8104,  1200,  -8104, -1200,
        7944,  1992,  -7944, -1992, 7712,  2760,  -7712, -2760, 7408,  3504,
        -7408, -3504, 7024,  4208,  -7024, -4208, 6576,  4880,  -6576, -4880,
        6072,  5504,  -6072, -5504, 8192,  200,   -8192, -200,  8168,  600,
        -8168, -600,  8128,  1000,  -8128, -1000, 8072,  1400,  -8072, -1400,
        7992,  1792,  -7992, -1792, 7896,  2184,  -7896, -2184, 7776,  2568,
        -7776, -2568, 7640,  2952,  -7640, -2952, 7488,  3320,  -7488, -3320,
        7320,  3680,  -7320, -3680, 7128,  4040,  -7128, -4040, 6920,  4384,
        -6920, -4384, 6696,  4720,  -6696, -4720, 6456,  5040,  -6456, -5040,
        6200,  5352,  -6200, -5352, 5936,  5648,  -5936, -5648,
    ],
    [
        5792,  5792,  -5792, -5792, 7568,  3136,  -7568, -3136, 8036,  1600,
        -8036, -1600, 6812,  4552,  -6812, -4552, 8152,  804,   -8152, -804,
        7840,  2380,  -7840, -2380, 7224,  3860,  -7224, -3860, 6332,  5196,
        -6332, -5196, 8184,  400,   -8184, -400,  8104,  1204,  -8104, -1204,
        7948,  1992,  -7948, -1992, 7712,  2760,  -7712, -2760, 7404,  3504,
        -7404, -3504, 7028,  4212,  -7028, -4212, 6580,  4880,  -6580, -4880,
        6068,  5500,  -6068, -5500, 8188,  200,   -8188, -200,  8168,  604,
        -8168, -604,  8132,  1004,  -8132, -1004, 8072,  1400,  -8072, -1400,
        7992,  1796,  -7992, -1796, 7896,  2184,  -7896, -2184, 7780,  2568,
        -7780, -2568, 7644,  2948,  -7644, -2948, 7488,  3320,  -7488, -3320,
        7316,  3684,  -7316, -3684, 7128,  4036,  -7128, -4036, 6920,  4384,
        -6920, -4384, 6696,  4716,  -6696, -4716, 6460,  5040,  -6460, -5040,
        6204,  5352,  -6204, -5352, 5932,  5648,  -5932, -5648,
    ],
    [
        5792,  5792,  -5792, -5792, 7568,  3134,  -7568, -3134, 8034,  1598,
        -8034, -1598, 6812,  4552,  -6812, -4552, 8152,  802,   -8152, -802,
        7840,  2378,  -7840, -2378, 7224,  3862,  -7224, -3862, 6332,  5196,
        -6332, -5196, 8182,  402,   -8182, -402, 8104,  1202,  -8104, -1202,
        7946,  1990,  -7946, -1990, 7714,  2760,  -7714, -2760, 7406,  3502,
        -7406, -3502, 7026,  4212,  -7026, -4212, 6580,  4880,  -6580, -4880,
        6070,  5502,  -6070, -5502, 8190,  202,   -8190, -202,  8170,  602,
        -8170, -602,  8130,  1002,  -8130, -1002, 8072,  1400,  -8072, -1400,
        7992,  1794,  -7992, -1794, 7896,  2184,  -7896, -2184, 7778,  2570,
        -7778, -2570, 7644,  2948,  -7644, -2948, 7490,  3320,  -7490, -3320,
        7318,  3684,  -7318, -3684, 7128,  4038,  -7128, -4038, 6922,  4382,
        -6922, -4382, 6698,  4718,  -6698, -4718, 6458,  5040,  -6458, -5040,
        6204,  5350,  -6204, -5350, 5934,  5648,  -5934, -5648,
    ],
    [
        5793,  5793,  -5793, -5793, 7568,  3135,  -7568, -3135, 8035,  1598,
        -8035, -1598, 6811,  4551,  -6811, -4551, 8153,  803,   -8153, -803,
        7839,  2378,  -7839, -2378, 7225,  3862,  -7225, -3862, 6333,  5197,
        -6333, -5197, 8182,  402,   -8182, -402, 8103,  1202,  -8103, -1202,
        7946,  1990,  -7946, -1990, 7713,  2760,  -7713, -2760, 7405,  3503,
        -7405, -3503, 7027,  4212,  -7027, -4212, 6580,  4880,  -6580, -4880,
        6070,  5501,  -6070, -5501, 8190,  201,   -8190, -201,  8170,  603,
        -8170, -603,  8130,  1003,  -8130, -1003, 8071,  1401,  -8071, -1401,
        7993,  1795,  -7993, -1795, 7895,  2185,  -7895, -2185, 7779,  2570,
        -7779, -2570, 7643,  2948,  -7643, -2948, 7489,  3320,  -7489, -3320,
        7317,  3683,  -7317, -3683, 7128,  4038,  -7128, -4038, 6921,  4383,
        -6921, -4383, 6698,  4717,  -6698, -4717, 6458,  5040,  -6458, -5040,
        6203,  5351,  -6203, -5351, 5933,  5649,  -5933, -5649,
    ],
];

/// `av1_sinpi_arr_q13_data` (av1/common/av1_txfm.c).
#[rustfmt::skip]
pub(crate) static SINPI_Q13: [[i16; 4]; 4] = [
    [2640, 4968, 6688, 7608],
    [2640, 4964, 6688, 7604],
    [2642, 4964, 6688, 7606],
    [2642, 4964, 6689, 7606],
];

/// `TXFM_COS_BIT_MAX` (txfm_common.h) — the lowbd kernels' fixed rounding
/// shift; `cos_bit` selects the table row only.
const TXFM_COS_BIT_MAX: i32 = 13;
/// `NewSqrt2` / `NewSqrt2Bits` (av1/common/av1_txfm.h).
const NEW_SQRT2: i16 = 5793;
const NEW_SQRT2_BITS: i32 = 12;

#[inline]
fn cospi_row(cos_bit: i32) -> &'static [i16; 128] {
    &COSPI_Q13[(cos_bit - 10) as usize]
}
#[inline]
#[target_feature(enable = "neon")]
fn sinpi_row(cos_bit: i32) -> int16x4_t {
    vld1_s16(&SINPI_Q13[(cos_bit - 10) as usize])
}

// ---------------------------------------------------------------------------
// `fdct4x4_neon` / `fadst4x4_neon` / `fidentity4x4_neon` — the 4-point 1-D
// kernels (av1_fwd_txfm2d_neon.c:320, :454, :1456).
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
fn fdct4x4_neon(i: &[int16x4_t; 4], o: &mut [int16x4_t; 4], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let c16 = vld1_s16(<&[i16; 4]>::try_from(&cospi[4..8]).unwrap());
    let in12a = vadd_s16(i[1], i[2]);
    let in12s = vsub_s16(i[1], i[2]);
    let in03a = vadd_s16(i[0], i[3]);
    let in03s = vsub_s16(i[0], i[3]);
    let u0ad1 = vmull_n_s16(in12a, cospi[0]);
    let u0ad2 = vmull_n_s16(in03a, cospi[0]);
    let u0 = vaddq_s32(u0ad1, u0ad2);
    let u1 = vsubq_s32(u0ad2, u0ad1);
    let u2 = vmlal_lane_s16::<0>(vmull_lane_s16::<1>(in12s, c16), in03s, c16);
    let u3 = vmlsl_lane_s16::<0>(vmull_lane_s16::<1>(in03s, c16), in12s, c16);
    o[0] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u0);
    o[1] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u2);
    o[2] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u1);
    o[3] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u3);
}

#[target_feature(enable = "neon")]
fn fadst4x4_neon(i: &[int16x4_t; 4], o: &mut [int16x4_t; 4], cos_bit: i32) {
    let sinpi = sinpi_row(cos_bit);
    let u01 = vqadd_s16(i[0], i[1]);
    let v5 = vmull_lane_s16::<2>(i[2], sinpi);
    let v0 = vmlal_lane_s16::<0>(vmull_lane_s16::<1>(i[1], sinpi), i[0], sinpi);
    let v1 = vmlal_lane_s16::<3>(v5, i[3], sinpi);
    let v2 = vmull_lane_s16::<2>(u01, sinpi);
    let v3 = vmlsl_lane_s16::<0>(vmull_lane_s16::<3>(i[0], sinpi), i[1], sinpi);
    let v4 = vmlsl_lane_s16::<1>(v5, i[3], sinpi);
    let u0 = vaddq_s32(v0, v1);
    let u1 = vmlsl_lane_s16::<2>(v2, i[3], sinpi);
    let u2 = vsubq_s32(v3, v4);
    let u3 = vmlaq_n_s32(vsubq_s32(u2, u0), v5, 3);
    o[0] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u0);
    o[1] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u1);
    o[2] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u2);
    o[3] = vrshrn_n_s32::<TXFM_COS_BIT_MAX>(u3);
}

#[target_feature(enable = "neon")]
fn fidentity4x4_neon(i: &[int16x4_t; 4], o: &mut [int16x4_t; 4]) {
    for k in 0..4 {
        o[k] = vqrshrn_n_s32::<NEW_SQRT2_BITS>(vmull_n_s16(i[k], NEW_SQRT2));
    }
}

/// `transpose_elems_inplace_s16_4x4` (aom_dsp/arm/transpose_neon.h) —
/// `vtrn_s16` then `vtrn_s32`.
#[target_feature(enable = "neon")]
fn transpose4x4(i: &[int16x4_t; 4]) -> [int16x4_t; 4] {
    let b0 = vtrn_s16(i[0], i[1]);
    let b1 = vtrn_s16(i[2], i[3]);
    let c0 = vtrn_s32(vreinterpret_s32_s16(b0.0), vreinterpret_s32_s16(b1.0));
    let c1 = vtrn_s32(vreinterpret_s32_s16(b0.1), vreinterpret_s32_s16(b1.1));
    [
        vreinterpret_s16_s32(c0.0),
        vreinterpret_s16_s32(c1.0),
        vreinterpret_s16_s32(c0.1),
        vreinterpret_s16_s32(c1.1),
    ]
}

/// The fused whole-block 4x4 forward — `lowbd_fwd_txfm2d_4x4_neon`
/// (av1_fwd_txfm2d_neon.c:1905), minus the tx_type switch (the port's
/// dispatcher resolves the two 1-D kernels ahead of time). Keeps the fused
/// kernels' decline contract: `|input| <= 512` (the i16-domain bound) and
/// in-range slices, else `false` -> the caller's generic fallback.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_4x4_fused(
    _t: archmage::NeonToken,
    kc: super::Fwd4,
    kr: super::Fwd4,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    // TRANSFORM_COL: load 4 rows (ud-flipped), `shift_left_2`, and the
    // fused |input| <= 512 bound — `vabs_s16(i16::MIN)` reads 32768 > 512
    // as u16, the same bound the SSE2 body computes lane-wise.
    let mut mx = vdup_n_u16(0);
    let mut b = [vdup_n_s16(0); 4];
    for r in 0..4 {
        let s = if ud_flip { 3 - r } else { r };
        let Some(row) = input.get(s * stride..s * stride + 4) else {
            return false;
        };
        let x = vld1_s16(<&[i16; 4]>::try_from(row).unwrap());
        mx = vmax_u16(mx, vreinterpret_u16_s16(vabs_s16(x)));
        b[r] = vshl_n_s16::<2>(x);
    }
    if vmaxv_u16(mx) > 512 {
        return false;
    }

    let mut col = b;
    match kc {
        super::Fwd4::Dct => fdct4x4_neon(&b, &mut col, cos_bit_col),
        super::Fwd4::Adst => fadst4x4_neon(&b, &mut col, cos_bit_col),
        super::Fwd4::Idtx => fidentity4x4_neon(&b, &mut col),
    }

    // transpose_arrays_s16_4x4, then lr_flip's flip_buf_4.
    let mut w = transpose4x4(&col);
    if lr_flip {
        w.reverse();
    }

    let mut out4 = w;
    match kr {
        super::Fwd4::Dct => fdct4x4_neon(&w, &mut out4, cos_bit_row),
        super::Fwd4::Adst => fadst4x4_neon(&w, &mut out4, cos_bit_row),
        super::Fwd4::Idtx => fidentity4x4_neon(&w, &mut out4),
    }

    // TRANSFORM_ROW's store_buffer_s16_x4: sign-extend, column-major
    // `output[c*4 + r]` (the port's i32 coefficient layout — same bytes C
    // writes at stride 4).
    for (c, v) in out4.iter().enumerate() {
        let ext = vmovl_s16(*v);
        let Some(d) = output.get_mut(c * 4..c * 4 + 4) else {
            return false;
        };
        vst1q_s32(<&mut [i32; 4]>::try_from(d).unwrap(), ext);
    }
    true
}

// ---------------------------------------------------------------------------
// x8 butterflies (av1_fwd_txfm2d_neon.c:103-179) — the lane-multiply core of
// every 8-wide kernel. `butterfly_s16_s32_x8_neon` widens both halves,
// `vrshrn` narrows (wrapping — the x4 twin uses `vqrshrn`, keep verbatim).
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "neon")]
fn bfly_s16_s32_x8<const L0: i32, const L1: i32, const L2: i32, const L3: i32>(
    w: int16x4_t,
    in0: int16x8_t,
    in1: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let i0l = vget_low_s16(in0);
    let i0h = vget_high_s16(in0);
    let i1l = vget_low_s16(in1);
    let i1h = vget_high_s16(in1);
    let u0 = vmlal_lane_s16::<L1>(vmull_lane_s16::<L0>(i0l, w), i1l, w);
    let u1 = vmlal_lane_s16::<L1>(vmull_lane_s16::<L0>(i0h, w), i1h, w);
    let v0 = vmlal_lane_s16::<L3>(vmull_lane_s16::<L2>(i0l, w), i1l, w);
    let v1 = vmlal_lane_s16::<L3>(vmull_lane_s16::<L2>(i0h, w), i1h, w);
    (
        vcombine_s16(vrshrn_n_s32::<13>(u0), vrshrn_n_s32::<13>(u1)),
        vcombine_s16(vrshrn_n_s32::<13>(v0), vrshrn_n_s32::<13>(v1)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn bfly_x8_0112(w: int16x4_t, i0: int16x8_t, i1: int16x8_t) -> (int16x8_t, int16x8_t) {
    bfly_s16_s32_x8::<0, 1, 1, 2>(w, i0, i1)
}
#[inline]
#[target_feature(enable = "neon")]
fn bfly_x8_0332(w: int16x4_t, i0: int16x8_t, i1: int16x8_t) -> (int16x8_t, int16x8_t) {
    bfly_s16_s32_x8::<0, 3, 3, 2>(w, i0, i1)
}
#[inline]
#[target_feature(enable = "neon")]
fn bfly_x8_1003(w: int16x4_t, i0: int16x8_t, i1: int16x8_t) -> (int16x8_t, int16x8_t) {
    bfly_s16_s32_x8::<1, 0, 0, 3>(w, i0, i1)
}

/// `butterfly_dct_pre_s16_x8` (:499) — first half qadd pairs, second half qsub.
#[inline]
#[target_feature(enable = "neon")]
fn dct_pre_x8<const N: usize>(i: &[int16x8_t], o: &mut [int16x8_t]) {
    for k in 0..N / 2 {
        o[k] = vqaddq_s16(i[k], i[N - k - 1]);
    }
    for k in 0..N / 2 {
        o[N / 2 + k] = vqsubq_s16(i[N / 2 - k - 1], i[N / 2 + k]);
    }
}

/// `butterfly_dct_post_s16_x8` (:551).
#[inline]
#[target_feature(enable = "neon")]
fn dct_post_x8<const N: usize>(i0: &[int16x8_t], i1: &[int16x8_t], o: &mut [int16x8_t]) {
    for k in 0..N / 4 {
        o[k] = vqaddq_s16(i0[k], i1[N / 2 - k - 1]);
    }
    for k in 0..N / 4 {
        o[N / 4 + k] = vqsubq_s16(i0[N / 4 - k - 1], i1[N / 4 + k]);
    }
    for k in 0..N / 4 {
        o[N / 2 + k] = vqsubq_s16(i0[N - k - 1], i1[N / 2 + k]);
    }
    for k in 0..N / 4 {
        o[3 * N / 4 + k] = vqaddq_s16(i0[3 * N / 4 + k], i1[3 * N / 4 - k - 1]);
    }
}

/// `fdct8x8_neon` (:646).
#[target_feature(enable = "neon")]
fn fdct8x8_neon(i: &[int16x8_t; 8], o: &mut [int16x8_t; 8], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let cospi32_16 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[0..8]).unwrap());
    let cospi8_24 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[8..16]).unwrap());
    let cospi32 = vget_low_s16(cospi32_16);
    let cospi16 = vget_high_s16(cospi32_16);
    let cospi8 = vget_low_s16(cospi8_24);
    let cospi24 = vget_high_s16(cospi8_24);

    let mut x1 = *i;
    dct_pre_x8::<8>(i, &mut x1);

    let mut x2 = x1;
    dct_pre_x8::<4>(&x1[..4], &mut x2[..4]);
    let (x2_6, x2_5) = bfly_x8_0112(cospi32, x1[6], x1[5]);
    x2[6] = x2_6;
    x2[5] = x2_5;

    let mut x3 = x2;
    let (o0, o4) = bfly_x8_0112(cospi32, x2[0], x2[1]);
    o[0] = o0;
    o[4] = o4;
    let (o2, o6) = bfly_x8_0112(cospi16, x2[3], x2[2]);
    o[2] = o2;
    o[6] = o6;
    dct_post_x8::<4>(&x1[4..], &x2[4..], &mut x3[4..]);

    let (o1, o7) = bfly_x8_0112(cospi8, x3[7], x3[4]);
    o[1] = o1;
    o[7] = o7;
    let (o5, o3) = bfly_x8_1003(cospi24, x3[6], x3[5]);
    o[5] = o5;
    o[3] = o3;
}

/// `fadst8x8_neon` (:1171).
#[target_feature(enable = "neon")]
fn fadst8x8_neon(i: &[int16x8_t; 8], o: &mut [int16x8_t; 8], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let cospi32_16 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[0..8]).unwrap());
    let cospi4_12 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[16..24]).unwrap());
    let cospi20_28 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[24..32]).unwrap());
    let cospi32 = vget_low_s16(cospi32_16);
    let cospi16 = vget_high_s16(cospi32_16);
    let cospi4 = vget_low_s16(cospi4_12);
    let cospi12 = vget_high_s16(cospi4_12);
    let cospi20 = vget_low_s16(cospi20_28);
    let cospi28 = vget_high_s16(cospi20_28);

    let mut x2 = [vdupq_n_s16(0); 8];
    let (x2_2, x2_3) = bfly_x8_0332(cospi32, i[4], i[3]);
    x2[2] = x2_2;
    x2[3] = x2_3;
    let (x2_7, x2_6) = bfly_x8_0112(cospi32, i[2], i[5]);
    x2[7] = x2_7;
    x2[6] = x2_6;

    let mut x3 = [vdupq_n_s16(0); 8];
    x3[0] = vqaddq_s16(i[0], x2[2]);
    x3[1] = vqsubq_s16(x2[3], i[7]);
    x3[2] = vqsubq_s16(i[0], x2[2]);
    x3[3] = vqaddq_s16(i[7], x2[3]);
    x3[4] = vqsubq_s16(x2[6], i[1]);
    x3[5] = vqaddq_s16(i[6], x2[7]);
    x3[6] = vqaddq_s16(i[1], x2[6]);
    x3[7] = vqsubq_s16(i[6], x2[7]);

    let (x3_4, x3_5) = bfly_x8_0112(cospi16, x3[4], x3[5]);
    x3[4] = x3_4;
    x3[5] = x3_5;
    let (x3_6, x3_7) = bfly_x8_0112(cospi16, x3[7], x3[6]);
    x3[6] = x3_6;
    x3[7] = x3_7;

    let mut x5 = [vdupq_n_s16(0); 8];
    x5[0] = vqaddq_s16(x3[0], x3[4]);
    x5[1] = vqaddq_s16(x3[1], x3[5]);
    x5[2] = vqaddq_s16(x3[2], x3[6]);
    x5[3] = vqsubq_s16(x3[7], x3[3]);
    x5[4] = vqsubq_s16(x3[0], x3[4]);
    x5[5] = vqsubq_s16(x3[1], x3[5]);
    x5[6] = vqsubq_s16(x3[2], x3[6]);
    x5[7] = vqaddq_s16(x3[3], x3[7]);

    let (o7, o0) = bfly_x8_0112(cospi4, x5[0], x5[1]);
    o[7] = o7;
    o[0] = o0;
    let (o5, o2) = bfly_x8_0112(cospi20, x5[2], x5[3]);
    o[5] = o5;
    o[2] = o2;
    let (o3, o4) = bfly_x8_1003(cospi28, x5[4], x5[5]);
    o[3] = o3;
    o[4] = o4;
    let (o6, o1) = bfly_x8_0112(cospi12, x5[6], x5[7]);
    o[6] = o6;
    o[1] = o1;
}

/// `fidentity8x8_neon` (:1476) — `shift_left_1` (adds, not a shift).
#[target_feature(enable = "neon")]
fn fidentity8x8_neon(i: &[int16x8_t; 8], o: &mut [int16x8_t; 8]) {
    for k in 0..8 {
        o[k] = vaddq_s16(i[k], i[k]);
    }
}

/// `aom_vtrnq_s64_to_s16` + `transpose_arrays_s16_8x8`
/// (transpose_neon.h:914, :1003).
#[inline]
#[target_feature(enable = "neon")]
fn trn_s64_to_s16(a0: int32x4_t, a1: int32x4_t) -> (int16x8_t, int16x8_t) {
    (
        vcombine_s16(
            vreinterpret_s16_s32(vget_low_s32(a0)),
            vreinterpret_s16_s32(vget_low_s32(a1)),
        ),
        vcombine_s16(
            vreinterpret_s16_s32(vget_high_s32(a0)),
            vreinterpret_s16_s32(vget_high_s32(a1)),
        ),
    )
}

#[target_feature(enable = "neon")]
fn transpose8x8(a: &[int16x8_t; 8]) -> [int16x8_t; 8] {
    let b0 = vtrnq_s16(a[0], a[1]);
    let b1 = vtrnq_s16(a[2], a[3]);
    let b2 = vtrnq_s16(a[4], a[5]);
    let b3 = vtrnq_s16(a[6], a[7]);
    let c0 = vtrnq_s32(vreinterpretq_s32_s16(b0.0), vreinterpretq_s32_s16(b1.0));
    let c1 = vtrnq_s32(vreinterpretq_s32_s16(b0.1), vreinterpretq_s32_s16(b1.1));
    let c2 = vtrnq_s32(vreinterpretq_s32_s16(b2.0), vreinterpretq_s32_s16(b3.0));
    let c3 = vtrnq_s32(vreinterpretq_s32_s16(b2.1), vreinterpretq_s32_s16(b3.1));
    let d0 = trn_s64_to_s16(c0.0, c2.0);
    let d1 = trn_s64_to_s16(c1.0, c3.0);
    let d2 = trn_s64_to_s16(c0.1, c2.1);
    let d3 = trn_s64_to_s16(c1.1, c3.1);
    [d0.0, d1.0, d2.0, d3.0, d0.1, d1.1, d2.1, d3.1]
}

/// `lowbd_fwd_txfm2d_8x8_neon` (:2077). Structure per tx_type is uniform:
/// col kernel -> `shift_right_1_round` (`vrhaddq` vs zero) -> transpose ->
/// (lr flip) -> row kernel with `store_buffer_s16_x8` (stride 8, column-major).
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_8x8_fused(
    _t: archmage::NeonToken,
    kc: super::Fwd8,
    kr: super::Fwd8,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let mut mx = vdupq_n_u16(0);
    let mut b = [vdupq_n_s16(0); 8];
    for r in 0..8 {
        let s = if ud_flip { 7 - r } else { r };
        let Some(row) = input.get(s * stride..s * stride + 8) else {
            return false;
        };
        let x = vld1q_s16(<&[i16; 8]>::try_from(row).unwrap());
        mx = vmaxq_u16(mx, vreinterpretq_u16_s16(vabsq_s16(x)));
        b[r] = vshlq_n_s16::<2>(x);
    }
    // The fused 8x8's bound is 511, not the 4x4's 512 — pass-1 outputs reach
    // 11,563 and `round_shift_16bit`'s adds(_, 1) must not saturate.
    if vmaxvq_u16(mx) > 511 {
        return false;
    }

    let mut col = b;
    match kc {
        super::Fwd8::Dct => fdct8x8_neon(&b, &mut col, cos_bit_col),
        super::Fwd8::Adst => fadst8x8_neon(&b, &mut col, cos_bit_col),
        super::Fwd8::Idtx => fidentity8x8_neon(&b, &mut col),
    }
    // shift_right_1_round_s16_x8
    let z = vdupq_n_s16(0);
    for v in col.iter_mut() {
        *v = vrhaddq_s16(*v, z);
    }

    let mut w = transpose8x8(&col);
    if lr_flip {
        w.reverse();
    }

    let mut out8 = w;
    match kr {
        super::Fwd8::Dct => fdct8x8_neon(&w, &mut out8, cos_bit_row),
        super::Fwd8::Adst => fadst8x8_neon(&w, &mut out8, cos_bit_row),
        super::Fwd8::Idtx => fidentity8x8_neon(&w, &mut out8),
    }

    for (c, v) in out8.iter().enumerate() {
        let Some(d) = output.get_mut(c * 8..c * 8 + 8) else {
            return false;
        };
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut d[..4]).unwrap(),
            vmovl_s16(vget_low_s16(*v)),
        );
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut d[4..]).unwrap(),
            vmovl_s16(vget_high_s16(*v)),
        );
    }
    true
}

// ---------------------------------------------------------------------------
// x4 butterflies (:103-136) — same lane-multiply core at 4 lanes, but the
// narrow is `vqrshrn` (saturating) where the x8 twin uses `vrshrn`. Verbatim.
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "neon")]
fn bfly_s16_s32_x4<const L0: i32, const L1: i32, const L2: i32, const L3: i32>(
    w: int16x4_t,
    in0: int16x4_t,
    in1: int16x4_t,
) -> (int16x4_t, int16x4_t) {
    let u = vmlal_lane_s16::<L1>(vmull_lane_s16::<L0>(in0, w), in1, w);
    let v = vmlal_lane_s16::<L3>(vmull_lane_s16::<L2>(in0, w), in1, w);
    (vqrshrn_n_s32::<13>(u), vqrshrn_n_s32::<13>(v))
}

#[inline]
#[target_feature(enable = "neon")]
fn bfly_x4_0112(w: int16x4_t, i0: int16x4_t, i1: int16x4_t) -> (int16x4_t, int16x4_t) {
    bfly_s16_s32_x4::<0, 1, 1, 2>(w, i0, i1)
}
#[inline]
#[target_feature(enable = "neon")]
fn bfly_x4_0332(w: int16x4_t, i0: int16x4_t, i1: int16x4_t) -> (int16x4_t, int16x4_t) {
    bfly_s16_s32_x4::<0, 3, 3, 2>(w, i0, i1)
}
#[inline]
#[target_feature(enable = "neon")]
fn bfly_x4_1003(w: int16x4_t, i0: int16x4_t, i1: int16x4_t) -> (int16x4_t, int16x4_t) {
    bfly_s16_s32_x4::<1, 0, 0, 3>(w, i0, i1)
}

/// `butterfly_dct_pre_s16_x4` (:488).
#[inline]
#[target_feature(enable = "neon")]
fn dct_pre_x4<const N: usize>(i: &[int16x4_t], o: &mut [int16x4_t]) {
    for k in 0..N / 2 {
        o[k] = vqadd_s16(i[k], i[N - k - 1]);
    }
    for k in 0..N / 2 {
        o[N / 2 + k] = vqsub_s16(i[N / 2 - k - 1], i[N / 2 + k]);
    }
}

/// `butterfly_dct_post_s16_x4` (:532).
#[inline]
#[target_feature(enable = "neon")]
fn dct_post_x4<const N: usize>(i0: &[int16x4_t], i1: &[int16x4_t], o: &mut [int16x4_t]) {
    for k in 0..N / 4 {
        o[k] = vqadd_s16(i0[k], i1[N / 2 - k - 1]);
    }
    for k in 0..N / 4 {
        o[N / 4 + k] = vqsub_s16(i0[N / 4 - k - 1], i1[N / 4 + k]);
    }
    for k in 0..N / 4 {
        o[N / 2 + k] = vqsub_s16(i0[N - k - 1], i1[N / 2 + k]);
    }
    for k in 0..N / 4 {
        o[3 * N / 4 + k] = vqadd_s16(i0[3 * N / 4 + k], i1[3 * N / 4 - k - 1]);
    }
}

/// `fdct4x8_neon` (:614) — the 8-point DCT on int16x4 lanes.
#[target_feature(enable = "neon")]
fn fdct4x8_neon(i: &[int16x4_t; 8], o: &mut [int16x4_t; 8], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let cospi32_16 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[0..8]).unwrap());
    let cospi8_24 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[8..16]).unwrap());
    let cospi32 = vget_low_s16(cospi32_16);
    let cospi16 = vget_high_s16(cospi32_16);
    let cospi8 = vget_low_s16(cospi8_24);
    let cospi24 = vget_high_s16(cospi8_24);

    let mut x1 = *i;
    dct_pre_x4::<8>(i, &mut x1);
    let mut x2 = x1;
    dct_pre_x4::<4>(&x1[..4], &mut x2[..4]);
    let (x2_6, x2_5) = bfly_x4_0112(cospi32, x1[6], x1[5]);
    x2[6] = x2_6;
    x2[5] = x2_5;

    let mut x3 = x2;
    let (o0, o4) = bfly_x4_0112(cospi32, x2[0], x2[1]);
    o[0] = o0;
    o[4] = o4;
    let (o2, o6) = bfly_x4_0112(cospi16, x2[3], x2[2]);
    o[2] = o2;
    o[6] = o6;
    dct_post_x4::<4>(&x1[4..], &x2[4..], &mut x3[4..]);

    let (o1, o7) = bfly_x4_0112(cospi8, x3[7], x3[4]);
    o[1] = o1;
    o[7] = o7;
    let (o5, o3) = bfly_x4_1003(cospi24, x3[6], x3[5]);
    o[5] = o5;
    o[3] = o3;
}

/// `fadst4x8_neon` (:347) — the 8-point ADST on int16x4 lanes.
#[target_feature(enable = "neon")]
fn fadst4x8_neon(i: &[int16x4_t; 8], o: &mut [int16x4_t; 8], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let cospi32_16 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[0..8]).unwrap());
    let cospi4_12 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[16..24]).unwrap());
    let cospi20_28 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[24..32]).unwrap());
    let cospi32 = vget_low_s16(cospi32_16);
    let cospi16 = vget_high_s16(cospi32_16);
    let cospi4 = vget_low_s16(cospi4_12);
    let cospi12 = vget_high_s16(cospi4_12);
    let cospi20 = vget_low_s16(cospi20_28);
    let cospi28 = vget_high_s16(cospi20_28);

    let mut x2 = [vdup_n_s16(0); 8];
    let (x2_2, x2_3) = bfly_x4_0332(cospi32, i[4], i[3]);
    x2[2] = x2_2;
    x2[3] = x2_3;
    let (x2_7, x2_6) = bfly_x4_0112(cospi32, i[2], i[5]);
    x2[7] = x2_7;
    x2[6] = x2_6;

    let mut x3 = [vdup_n_s16(0); 8];
    x3[0] = vqadd_s16(i[0], x2[2]);
    x3[1] = vqsub_s16(x2[3], i[7]);
    x3[2] = vqsub_s16(i[0], x2[2]);
    x3[3] = vqadd_s16(i[7], x2[3]);
    x3[4] = vqsub_s16(x2[6], i[1]);
    x3[5] = vqadd_s16(i[6], x2[7]);
    x3[6] = vqadd_s16(i[1], x2[6]);
    x3[7] = vqsub_s16(i[6], x2[7]);

    let mut x4 = [vdup_n_s16(0); 8];
    let (x4_4, x4_5) = bfly_x4_0112(cospi16, x3[4], x3[5]);
    x4[4] = x4_4;
    x4[5] = x4_5;
    let (x4_6, x4_7) = bfly_x4_0112(cospi16, x3[7], x3[6]);
    x4[6] = x4_6;
    x4[7] = x4_7;

    let mut x5 = [vdup_n_s16(0); 8];
    x5[0] = vqadd_s16(x3[0], x4[4]);
    x5[1] = vqadd_s16(x3[1], x4[5]);
    x5[2] = vqadd_s16(x3[2], x4[6]);
    x5[3] = vqsub_s16(x4[7], x3[3]);
    x5[4] = vqsub_s16(x3[0], x4[4]);
    x5[5] = vqsub_s16(x3[1], x4[5]);
    x5[6] = vqsub_s16(x3[2], x4[6]);
    x5[7] = vqadd_s16(x3[3], x4[7]);

    let (o7, o0) = bfly_x4_0112(cospi4, x5[0], x5[1]);
    o[7] = o7;
    o[0] = o0;
    let (o5, o2) = bfly_x4_0112(cospi20, x5[2], x5[3]);
    o[5] = o5;
    o[2] = o2;
    let (o3, o4) = bfly_x4_1003(cospi28, x5[4], x5[5]);
    o[3] = o3;
    o[4] = o4;
    let (o6, o1) = bfly_x4_0112(cospi12, x5[6], x5[7]);
    o[6] = o6;
    o[1] = o1;
}

/// `fdct8x4_neon` (:589) — the 4-point DCT on int16x8 lanes.
#[target_feature(enable = "neon")]
fn fdct8x4_neon(i: &[int16x8_t; 4], o: &mut [int16x8_t; 4], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let cospi32_16 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[0..8]).unwrap());
    let cospi32 = vget_low_s16(cospi32_16);
    let cospi16 = vget_high_s16(cospi32_16);
    let mut x1 = [vdupq_n_s16(0); 4];
    dct_pre_x8::<4>(i, &mut x1);
    let (x2_0, x2_1) = bfly_x8_0112(cospi32, x1[0], x1[1]);
    let (x2_2, x2_3) = bfly_x8_0112(cospi16, x1[3], x1[2]);
    o[0] = x2_0;
    o[1] = x2_2;
    o[2] = x2_1;
    o[3] = x2_3;
}

/// `fadst8x4_neon` (:401) — the 4-point ADST on int16x8 lanes, lo/hi split.
#[target_feature(enable = "neon")]
fn fadst8x4_neon(i: &[int16x8_t; 4], o: &mut [int16x8_t; 4], cos_bit: i32) {
    let sinpi = sinpi_row(cos_bit);
    let u01 = vqaddq_s16(i[0], i[1]);
    let l = |v: int16x8_t| vget_low_s16(v);
    let h = |v: int16x8_t| vget_high_s16(v);

    let mut ul = [vmlal_lane_s16::<0>(vmull_lane_s16::<1>(l(i[1]), sinpi), l(i[0]), sinpi); 4];
    let mut uh = [vmlal_lane_s16::<0>(vmull_lane_s16::<1>(h(i[1]), sinpi), h(i[0]), sinpi); 4];
    ul[0] = vmlal_lane_s16::<3>(ul[0], l(i[3]), sinpi);
    uh[0] = vmlal_lane_s16::<3>(uh[0], h(i[3]), sinpi);
    ul[0] = vmlal_lane_s16::<2>(ul[0], l(i[2]), sinpi);
    uh[0] = vmlal_lane_s16::<2>(uh[0], h(i[2]), sinpi);

    ul[1] = vmull_lane_s16::<2>(l(u01), sinpi);
    uh[1] = vmull_lane_s16::<2>(h(u01), sinpi);

    ul[2] = vmull_lane_s16::<3>(l(i[0]), sinpi);
    uh[2] = vmull_lane_s16::<3>(h(i[0]), sinpi);
    ul[2] = vmlsl_lane_s16::<0>(ul[2], l(i[1]), sinpi);
    uh[2] = vmlsl_lane_s16::<0>(uh[2], h(i[1]), sinpi);
    ul[2] = vmlal_lane_s16::<1>(ul[2], l(i[3]), sinpi);
    uh[2] = vmlal_lane_s16::<1>(uh[2], h(i[3]), sinpi);
    ul[2] = vmlsl_lane_s16::<2>(ul[2], l(i[2]), sinpi);
    uh[2] = vmlsl_lane_s16::<2>(uh[2], h(i[2]), sinpi);

    ul[1] = vmlsl_lane_s16::<2>(ul[1], l(i[3]), sinpi);
    uh[1] = vmlsl_lane_s16::<2>(uh[1], h(i[3]), sinpi);

    ul[3] = vsubq_s32(ul[2], ul[0]);
    uh[3] = vsubq_s32(uh[2], uh[0]);
    let sinpix3 = vmul_n_s16(sinpi, 3);
    ul[3] = vmlal_lane_s16::<2>(ul[3], l(i[2]), sinpix3);
    uh[3] = vmlal_lane_s16::<2>(uh[3], h(i[2]), sinpix3);

    for k in 0..4 {
        o[k] = vcombine_s16(vrshrn_n_s32::<13>(ul[k]), vrshrn_n_s32::<13>(uh[k]));
    }
}

/// `fidentity4x8_neon` (:1470) — `shift_left_1`.
#[target_feature(enable = "neon")]
fn fidentity4x8_neon(i: &[int16x4_t; 8], o: &mut [int16x4_t; 8]) {
    for k in 0..8 {
        o[k] = vadd_s16(i[k], i[k]);
    }
}

/// `fidentity8x4_neon` (:1463) — `round_shift_sqrt2` per lane.
#[target_feature(enable = "neon")]
fn fidentity8x4_neon(i: &[int16x8_t; 4], o: &mut [int16x8_t; 4]) {
    for k in 0..4 {
        o[k] = vcombine_s16(
            vqrshrn_n_s32::<12>(vmull_n_s16(vget_low_s16(i[k]), NEW_SQRT2)),
            vqrshrn_n_s32::<12>(vmull_n_s16(vget_high_s16(i[k]), NEW_SQRT2)),
        );
    }
}

/// `transpose_arrays_s16_4x8` (transpose_neon.h:1386, aarch64 path) —
/// [int16x4; 8] -> [int16x8; 4].
#[target_feature(enable = "neon")]
fn transpose4x8(i: &[int16x4_t; 8]) -> [int16x8_t; 4] {
    let z = vdup_n_s16(0);
    let a0 = vzip1q_s16(vcombine_s16(i[0], z), vcombine_s16(i[1], z));
    let a1 = vzip1q_s16(vcombine_s16(i[2], z), vcombine_s16(i[3], z));
    let a2 = vzip1q_s16(vcombine_s16(i[4], z), vcombine_s16(i[5], z));
    let a3 = vzip1q_s16(vcombine_s16(i[6], z), vcombine_s16(i[7], z));
    let b02 = vzipq_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1));
    let b13 = vzipq_s32(vreinterpretq_s32_s16(a2), vreinterpretq_s32_s16(a3));
    [
        vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s32(b02.0),
            vreinterpretq_s64_s32(b13.0),
        )),
        vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s32(b02.0),
            vreinterpretq_s64_s32(b13.0),
        )),
        vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s32(b02.1),
            vreinterpretq_s64_s32(b13.1),
        )),
        vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s32(b02.1),
            vreinterpretq_s64_s32(b13.1),
        )),
    ]
}

/// `transpose_arrays_s16_8x4` (transpose_neon.h:1435) —
/// [int16x8; 4] -> [int16x4; 8].
#[target_feature(enable = "neon")]
fn transpose8x4(i: &[int16x8_t; 4]) -> [int16x4_t; 8] {
    let b0 = vtrnq_s16(i[0], i[1]);
    let b1 = vtrnq_s16(i[2], i[3]);
    let c0 = vtrnq_u32(vreinterpretq_u32_s16(b0.0), vreinterpretq_u32_s16(b1.0));
    let c1 = vtrnq_u32(vreinterpretq_u32_s16(b0.1), vreinterpretq_u32_s16(b1.1));
    [
        vget_low_s16(vreinterpretq_s16_u32(c0.0)),
        vget_low_s16(vreinterpretq_s16_u32(c1.0)),
        vget_low_s16(vreinterpretq_s16_u32(c0.1)),
        vget_low_s16(vreinterpretq_s16_u32(c1.1)),
        vget_high_s16(vreinterpretq_s16_u32(c0.0)),
        vget_high_s16(vreinterpretq_s16_u32(c1.0)),
        vget_high_s16(vreinterpretq_s16_u32(c0.1)),
        vget_high_s16(vreinterpretq_s16_u32(c1.1)),
    ]
}

/// `store_rect_buffer_s16_x8`'s lane op — `round_shift_sqrt2_s16_s32`:
/// sign-extend to i32 after the ×sqrt2 round. The rect output rows get the
/// sqrt2 area scaling C applies in `TRANSFORM_ROW_RECT`.
#[inline]
#[target_feature(enable = "neon")]
fn rect_store_lane(v: int16x4_t) -> int32x4_t {
    vrshrq_n_s32::<12>(vmull_n_s16(v, NEW_SQRT2))
}

/// `lowbd_fwd_txfm2d_4x8_neon` (:2001) — 8-pt column kernel on x4 lanes,
/// `vrhadd` halving, `transpose_arrays_s16_4x8`, 4-pt row kernel on x8 lanes
/// with the rect (sqrt2) store. Bound: |input| <= 1023 (the fused rect48
/// gate).
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_4x8_fused(
    _t: archmage::NeonToken,
    kc: super::FwdR,
    kr: super::FwdR,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let mut mx = vdup_n_u16(0);
    let mut b = [vdup_n_s16(0); 8];
    for r in 0..8 {
        let s = if ud_flip { 7 - r } else { r };
        let Some(row) = input.get(s * stride..s * stride + 4) else {
            return false;
        };
        let x = vld1_s16(<&[i16; 4]>::try_from(row).unwrap());
        mx = vmax_u16(mx, vreinterpret_u16_s16(vabs_s16(x)));
        b[r] = vshl_n_s16::<2>(x);
    }
    if vmaxv_u16(mx) > 1023 {
        return false;
    }

    let mut col = b;
    match kc {
        super::FwdR::Dct => fdct4x8_neon(&b, &mut col, cos_bit_col),
        super::FwdR::Adst => fadst4x8_neon(&b, &mut col, cos_bit_col),
        super::FwdR::Idtx => fidentity4x8_neon(&b, &mut col),
    }
    let z = vdup_n_s16(0);
    for v in col.iter_mut() {
        *v = vrhadd_s16(*v, z);
    }

    // transpose_arrays_s16_4x8 writes 4 vectors; lr_flip flips only those.
    let mut w = transpose4x8(&col);
    if lr_flip {
        w.reverse();
    }

    let mut out4 = w;
    match kr {
        super::FwdR::Dct => fdct8x4_neon(&w, &mut out4, cos_bit_row),
        super::FwdR::Adst => fadst8x4_neon(&w, &mut out4, cos_bit_row),
        super::FwdR::Idtx => fidentity8x4_neon(&w, &mut out4),
    }

    for (c, v) in out4.iter().enumerate() {
        let Some(d) = output.get_mut(c * 8..c * 8 + 8) else {
            return false;
        };
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut d[..4]).unwrap(),
            rect_store_lane(vget_low_s16(*v)),
        );
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut d[4..]).unwrap(),
            rect_store_lane(vget_high_s16(*v)),
        );
    }
    true
}

/// `lowbd_fwd_txfm2d_8x4_neon` (:2053) — 4-pt column kernel on x8 lanes,
/// `vrhaddq` halving, `transpose_arrays_s16_8x4`, 8-pt row kernel on x4 lanes
/// with the rect store.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_8x4_fused(
    _t: archmage::NeonToken,
    kc: super::FwdR,
    kr: super::FwdR,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let mut mx = vdupq_n_u16(0);
    let mut b = [vdupq_n_s16(0); 4];
    for r in 0..4 {
        let s = if ud_flip { 3 - r } else { r };
        let Some(row) = input.get(s * stride..s * stride + 8) else {
            return false;
        };
        let x = vld1q_s16(<&[i16; 8]>::try_from(row).unwrap());
        mx = vmaxq_u16(mx, vreinterpretq_u16_s16(vabsq_s16(x)));
        b[r] = vshlq_n_s16::<2>(x);
    }
    if vmaxvq_u16(mx) > 1023 {
        return false;
    }

    let mut col = b;
    match kc {
        super::FwdR::Dct => fdct8x4_neon(&b, &mut col, cos_bit_col),
        super::FwdR::Adst => fadst8x4_neon(&b, &mut col, cos_bit_col),
        super::FwdR::Idtx => fidentity8x4_neon(&b, &mut col),
    }
    let z = vdupq_n_s16(0);
    for v in col.iter_mut() {
        *v = vrhaddq_s16(*v, z);
    }

    let mut w = transpose8x4(&col);
    if lr_flip {
        w.reverse();
    }

    let mut out8 = w;
    match kr {
        super::FwdR::Dct => fdct4x8_neon(&w, &mut out8, cos_bit_row),
        super::FwdR::Adst => fadst4x8_neon(&w, &mut out8, cos_bit_row),
        super::FwdR::Idtx => fidentity4x8_neon(&w, &mut out8),
    }

    for (c, v) in out8.iter().enumerate() {
        let Some(d) = output.get_mut(c * 4..c * 4 + 4) else {
            return false;
        };
        vst1q_s32(<&mut [i32; 4]>::try_from(d).unwrap(), rect_store_lane(*v));
    }
    true
}

#[inline]
#[target_feature(enable = "neon")]
fn bfly_x8_1223(w: int16x4_t, i0: int16x8_t, i1: int16x8_t) -> (int16x8_t, int16x8_t) {
    bfly_s16_s32_x8::<1, 2, 2, 3>(w, i0, i1)
}

/// `fdct8x16_neon` (:740) — the 16-point DCT on int16x8 lanes.
#[target_feature(enable = "neon")]
fn fdct8x16_neon(i: &[int16x8_t; 16], o: &mut [int16x8_t; 16], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let cospi32_16 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[0..8]).unwrap());
    let cospi8_24 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[8..16]).unwrap());
    let cospi4_12 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[16..24]).unwrap());
    let cospi20_28 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[24..32]).unwrap());
    let cospi32 = vget_low_s16(cospi32_16);
    let cospi16 = vget_high_s16(cospi32_16);
    let cospi8 = vget_low_s16(cospi8_24);
    let cospi24 = vget_high_s16(cospi8_24);
    let cospi4 = vget_low_s16(cospi4_12);
    let cospi12 = vget_high_s16(cospi4_12);
    let cospi20 = vget_low_s16(cospi20_28);
    let cospi28 = vget_high_s16(cospi20_28);

    let mut x1 = *i;
    dct_pre_x8::<16>(i, &mut x1);

    let mut x2 = x1;
    dct_pre_x8::<8>(&x1[..8], &mut x2[..8]);
    let (a, b) = bfly_x8_0112(cospi32, x1[13], x1[10]);
    x2[13] = a;
    x2[10] = b;
    let (a, b) = bfly_x8_0112(cospi32, x1[12], x1[11]);
    x2[12] = a;
    x2[11] = b;

    let mut x3 = x2;
    dct_pre_x8::<4>(&x2[..4], &mut x3[..4]);
    let (a, b) = bfly_x8_0112(cospi32, x2[6], x2[5]);
    x3[6] = a;
    x3[5] = b;
    dct_post_x8::<8>(&x1[8..], &x2[8..], &mut x3[8..]);

    let mut x4 = x3;
    let (a, b) = bfly_x8_0112(cospi32, x3[0], x3[1]);
    o[0] = a;
    o[8] = b;
    let (a, b) = bfly_x8_0112(cospi16, x3[3], x3[2]);
    o[4] = a;
    o[12] = b;
    dct_post_x8::<4>(&x2[4..8], &x3[4..8], &mut x4[4..8]);
    let (a, b) = bfly_x8_0112(cospi16, x3[14], x3[9]);
    x4[14] = a;
    x4[9] = b;
    let (a, b) = bfly_x8_1223(cospi16, x3[13], x3[10]);
    x4[13] = a;
    x4[10] = b;

    let mut x5 = x4;
    let (a, b) = bfly_x8_0112(cospi8, x4[7], x4[4]);
    o[2] = a;
    o[14] = b;
    let (a, b) = bfly_x8_1003(cospi24, x4[6], x4[5]);
    o[10] = a;
    o[6] = b;
    dct_post_x8::<4>(&x3[8..12], &x4[8..12], &mut x5[8..12]);
    dct_post_x8::<4>(&x3[12..], &x4[12..], &mut x5[12..]);

    let (a, b) = bfly_x8_0112(cospi4, x5[15], x5[8]);
    o[1] = a;
    o[15] = b;
    let (a, b) = bfly_x8_1003(cospi28, x5[14], x5[9]);
    o[9] = a;
    o[7] = b;
    let (a, b) = bfly_x8_0112(cospi20, x5[13], x5[10]);
    o[5] = a;
    o[11] = b;
    let (a, b) = bfly_x8_1003(cospi12, x5[12], x5[11]);
    o[13] = a;
    o[3] = b;
}

/// `fadst8x16_neon` (:1340) — the 16-point ADST on int16x8 lanes.
#[target_feature(enable = "neon")]
fn fadst8x16_neon(i: &[int16x8_t; 16], o: &mut [int16x8_t; 16], cos_bit: i32) {
    let cospi = cospi_row(cos_bit);
    let cospi32_16 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[0..8]).unwrap());
    let cospi8_24 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[8..16]).unwrap());
    let cospi2_6 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[32..40]).unwrap());
    let cospi10_14 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[40..48]).unwrap());
    let cospi18_22 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[48..56]).unwrap());
    let cospi26_30 = vld1q_s16(<&[i16; 8]>::try_from(&cospi[56..64]).unwrap());
    let cospi32 = vget_low_s16(cospi32_16);
    let cospi16 = vget_high_s16(cospi32_16);
    let cospi8 = vget_low_s16(cospi8_24);
    let cospi24 = vget_high_s16(cospi8_24);
    let cospi2 = vget_low_s16(cospi2_6);
    let cospi6 = vget_high_s16(cospi2_6);
    let cospi10 = vget_low_s16(cospi10_14);
    let cospi14 = vget_high_s16(cospi10_14);
    let cospi18 = vget_low_s16(cospi18_22);
    let cospi22 = vget_high_s16(cospi18_22);
    let cospi26 = vget_low_s16(cospi26_30);
    let cospi30 = vget_high_s16(cospi26_30);

    let mut x2 = [vdupq_n_s16(0); 8];
    let (a, b) = bfly_x8_0332(cospi32, i[8], i[7]);
    x2[0] = a;
    x2[1] = b;
    let (a, b) = bfly_x8_0112(cospi32, i[4], i[11]);
    x2[3] = a;
    x2[2] = b;
    let (a, b) = bfly_x8_0112(cospi32, i[6], i[9]);
    x2[5] = a;
    x2[4] = b;
    let (a, b) = bfly_x8_0332(cospi32, i[10], i[5]);
    x2[6] = a;
    x2[7] = b;

    let mut x3 = [vdupq_n_s16(0); 16];
    x3[0] = vqaddq_s16(i[0], x2[0]);
    x3[1] = vqsubq_s16(x2[1], i[15]);
    x3[2] = vqsubq_s16(i[0], x2[0]);
    x3[3] = vqaddq_s16(i[15], x2[1]);
    x3[4] = vqsubq_s16(x2[2], i[3]);
    x3[5] = vqaddq_s16(i[12], x2[3]);
    x3[6] = vqaddq_s16(i[3], x2[2]);
    x3[7] = vqsubq_s16(i[12], x2[3]);
    x3[8] = vqsubq_s16(x2[4], i[1]);
    x3[9] = vqaddq_s16(i[14], x2[5]);
    x3[10] = vqaddq_s16(i[1], x2[4]);
    x3[11] = vqsubq_s16(i[14], x2[5]);
    x3[12] = vqaddq_s16(i[2], x2[6]);
    x3[13] = vqsubq_s16(x2[7], i[13]);
    x3[14] = vqsubq_s16(i[2], x2[6]);
    x3[15] = vqaddq_s16(i[13], x2[7]);

    let (a, b) = bfly_x8_0112(cospi16, x3[4], x3[5]);
    x3[4] = a;
    x3[5] = b;
    let (a, b) = bfly_x8_0112(cospi16, x3[7], x3[6]);
    x3[6] = a;
    x3[7] = b;
    let (a, b) = bfly_x8_0112(cospi16, x3[12], x3[13]);
    x3[12] = a;
    x3[13] = b;
    let (a, b) = bfly_x8_0332(cospi16, x3[14], x3[15]);
    x3[15] = a;
    x3[14] = b;

    let mut x5 = [vdupq_n_s16(0); 16];
    x5[0] = vqaddq_s16(x3[0], x3[4]);
    x5[1] = vqaddq_s16(x3[1], x3[5]);
    x5[2] = vqaddq_s16(x3[2], x3[6]);
    x5[3] = vqsubq_s16(x3[7], x3[3]);
    x5[4] = vqsubq_s16(x3[0], x3[4]);
    x5[5] = vqsubq_s16(x3[1], x3[5]);
    x5[6] = vqsubq_s16(x3[2], x3[6]);
    x5[7] = vqaddq_s16(x3[3], x3[7]);
    x5[8] = vqaddq_s16(x3[8], x3[12]);
    x5[9] = vqaddq_s16(x3[9], x3[13]);
    x5[10] = vqsubq_s16(x3[14], x3[10]);
    x5[11] = vqaddq_s16(x3[11], x3[15]);
    x5[12] = vqsubq_s16(x3[8], x3[12]);
    x5[13] = vqsubq_s16(x3[9], x3[13]);
    x5[14] = vqaddq_s16(x3[10], x3[14]);
    x5[15] = vqsubq_s16(x3[11], x3[15]);

    let (a, b) = bfly_x8_0112(cospi8, x5[8], x5[9]);
    x5[8] = a;
    x5[9] = b;
    let (a, b) = bfly_x8_1003(cospi24, x5[10], x5[11]);
    x5[10] = a;
    x5[11] = b;
    let (a, b) = bfly_x8_1003(cospi8, x5[13], x5[12]);
    x5[13] = a;
    x5[12] = b;
    let (a, b) = bfly_x8_1003(cospi24, x5[15], x5[14]);
    x5[14] = a;
    x5[15] = b;

    let mut x7 = [vdupq_n_s16(0); 16];
    x7[0] = vqaddq_s16(x5[0], x5[8]);
    x7[1] = vqaddq_s16(x5[1], x5[9]);
    x7[2] = vqaddq_s16(x5[2], x5[10]);
    x7[3] = vqaddq_s16(x5[3], x5[11]);
    x7[4] = vqaddq_s16(x5[4], x5[12]);
    x7[5] = vqaddq_s16(x5[5], x5[13]);
    x7[6] = vqaddq_s16(x5[6], x5[14]);
    x7[7] = vqsubq_s16(x5[15], x5[7]);
    x7[8] = vqsubq_s16(x5[0], x5[8]);
    x7[9] = vqsubq_s16(x5[1], x5[9]);
    x7[10] = vqsubq_s16(x5[2], x5[10]);
    x7[11] = vqsubq_s16(x5[3], x5[11]);
    x7[12] = vqsubq_s16(x5[4], x5[12]);
    x7[13] = vqsubq_s16(x5[5], x5[13]);
    x7[14] = vqsubq_s16(x5[6], x5[14]);
    x7[15] = vqaddq_s16(x5[7], x5[15]);

    let (a, b) = bfly_x8_0112(cospi2, x7[0], x7[1]);
    o[15] = a;
    o[0] = b;
    let (a, b) = bfly_x8_0112(cospi10, x7[2], x7[3]);
    o[13] = a;
    o[2] = b;
    let (a, b) = bfly_x8_0112(cospi18, x7[4], x7[5]);
    o[11] = a;
    o[4] = b;
    let (a, b) = bfly_x8_0112(cospi26, x7[6], x7[7]);
    o[9] = a;
    o[6] = b;
    let (a, b) = bfly_x8_1003(cospi30, x7[8], x7[9]);
    o[7] = a;
    o[8] = b;
    let (a, b) = bfly_x8_1003(cospi22, x7[10], x7[11]);
    o[5] = a;
    o[10] = b;
    let (a, b) = bfly_x8_1003(cospi14, x7[12], x7[13]);
    o[3] = a;
    o[12] = b;
    let (a, b) = bfly_x8_0112(cospi6, x7[14], x7[15]);
    o[14] = a;
    o[1] = b;
}

/// `fidentity8x16_neon` (:1486) — `round_shift_2sqrt2` per lane.
#[target_feature(enable = "neon")]
fn fidentity8x16_neon(i: &[int16x8_t; 16], o: &mut [int16x8_t; 16]) {
    const S2: i16 = 2 * 5793;
    for k in 0..16 {
        o[k] = vcombine_s16(
            vqrshrn_n_s32::<12>(vmull_n_s16(vget_low_s16(i[k]), S2)),
            vqrshrn_n_s32::<12>(vmull_n_s16(vget_high_s16(i[k]), S2)),
        );
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn run_16_x8(k: super::FwdRB, i: &[int16x8_t; 16], cos_bit: i32) -> [int16x8_t; 16] {
    let mut o = *i;
    match k {
        super::FwdRB::Dct => fdct8x16_neon(i, &mut o, cos_bit),
        super::FwdRB::Adst => fadst8x16_neon(i, &mut o, cos_bit),
        super::FwdRB::Idtx => fidentity8x16_neon(i, &mut o),
    }
    o
}

#[inline]
#[target_feature(enable = "neon")]
fn run_8_x8(k: super::FwdRB, i: &[int16x8_t; 8], cos_bit: i32) -> [int16x8_t; 8] {
    let mut o = *i;
    match k {
        super::FwdRB::Dct => fdct8x8_neon(i, &mut o, cos_bit),
        super::FwdRB::Adst => fadst8x8_neon(i, &mut o, cos_bit),
        super::FwdRB::Idtx => fidentity8x8_neon(i, &mut o),
    }
    o
}

/// `shift_right_2_round_s16_x8` — `vrshrq_n_s16(v, 2)`.
#[inline]
#[target_feature(enable = "neon")]
fn shr2_x8(v: &mut [int16x8_t]) {
    for x in v.iter_mut() {
        *x = vrshrq_n_s16::<2>(*x);
    }
}

/// `lowbd_fwd_txfm2d_8x16_neon` (:2182) — one 16-pt column pass on x8 lanes,
/// `vrshrq 2`, two `transpose_arrays_s16_8x8`, then two 8-pt row groups with
/// the rect (sqrt2) store at stride 16. Bound `FWD816_I16_BOUND[kc][kr]`.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_8x16_fused(
    _t: archmage::NeonToken,
    kc: super::FwdRB,
    kr: super::FwdRB,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let bound = super::FWD816_I16_BOUND[super::fwdrb_idx(kc)][super::fwdrb_idx(kr)] as u16;
    let mut mx = vdupq_n_u16(0);
    let mut b = [vdupq_n_s16(0); 16];
    for r in 0..16 {
        let s = if ud_flip { 15 - r } else { r };
        let Some(row) = input.get(s * stride..s * stride + 8) else {
            return false;
        };
        let x = vld1q_s16(<&[i16; 8]>::try_from(row).unwrap());
        mx = vmaxq_u16(mx, vreinterpretq_u16_s16(vabsq_s16(x)));
        b[r] = vshlq_n_s16::<2>(x);
    }
    if vmaxvq_u16(mx) > bound {
        return false;
    }

    let mut col = run_16_x8(kc, &b, cos_bit_col);
    shr2_x8(&mut col);

    // Two 8x8 transposes -> buf1[16] (8 columns of 16 rows).
    let mut w = [vdupq_n_s16(0); 16];
    w[..8].copy_from_slice(&transpose8x8(&col[..8].try_into().unwrap()));
    w[8..].copy_from_slice(&transpose8x8(&col[8..].try_into().unwrap()));

    // Two 8-pt row groups; each writes its half-block of rows at stride 16.
    for i in 0..2 {
        let mut g: [int16x8_t; 8] = w[i * 8..i * 8 + 8].try_into().unwrap();
        if lr_flip {
            g.reverse();
        }
        let out = run_8_x8(kr, &g, cos_bit_row);
        for (j, v) in out.iter().enumerate() {
            let Some(d) = output.get_mut(j * 16 + 8 * i..j * 16 + 8 * i + 8) else {
                return false;
            };
            vst1q_s32(
                <&mut [i32; 4]>::try_from(&mut d[..4]).unwrap(),
                rect_store_lane(vget_low_s16(*v)),
            );
            vst1q_s32(
                <&mut [i32; 4]>::try_from(&mut d[4..]).unwrap(),
                rect_store_lane(vget_high_s16(*v)),
            );
        }
    }
    true
}

/// `lowbd_fwd_txfm2d_16x8_neon` (:2270) — two 8-pt column halves on x8 lanes
/// (each covering 8 of the 16 columns), `vrshrq 2`, per-half
/// `transpose_arrays_s16_8x8`, one 16-pt row pass with the rect store at
/// stride 8. Bound `FWD168_I16_BOUND[kc][kr]`.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_16x8_fused(
    _t: archmage::NeonToken,
    kc: super::FwdRB,
    kr: super::FwdRB,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let bound = super::FWD168_I16_BOUND[super::fwdrb_idx(kc)][super::fwdrb_idx(kr)] as u16;
    let mut mx = vdupq_n_u16(0);
    let mut w = [vdupq_n_s16(0); 16];
    for i in 0..2 {
        // Column half i: 8 rows of int16x8 at row-offset 8*i (columns 8i..8i+8).
        let mut b = [vdupq_n_s16(0); 8];
        for r in 0..8 {
            let s = if ud_flip { 7 - r } else { r };
            let Some(row) = input.get(s * stride + 8 * i..s * stride + 8 * i + 8) else {
                return false;
            };
            let x = vld1q_s16(<&[i16; 8]>::try_from(row).unwrap());
            mx = vmaxq_u16(mx, vreinterpretq_u16_s16(vabsq_s16(x)));
            b[r] = vshlq_n_s16::<2>(x);
        }
        let mut col = run_8_x8(kc, &b, cos_bit_col);
        shr2_x8(&mut col);
        w[i * 8..i * 8 + 8].copy_from_slice(&transpose8x8(&col));
    }
    if vmaxvq_u16(mx) > bound {
        return false;
    }

    if lr_flip {
        w.reverse();
    }
    let out = run_16_x8(kr, &w, cos_bit_row);
    for (j, v) in out.iter().enumerate() {
        let Some(d) = output.get_mut(j * 8..j * 8 + 8) else {
            return false;
        };
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut d[..4]).unwrap(),
            rect_store_lane(vget_low_s16(*v)),
        );
        vst1q_s32(
            <&mut [i32; 4]>::try_from(&mut d[4..]).unwrap(),
            rect_store_lane(vget_high_s16(*v)),
        );
    }
    true
}

/// The 16x16 family's `Fwd16` tags map 1:1 onto `FwdRB`'s kind tags.
#[inline]
fn fwd16_as_rb(k: super::Fwd16) -> super::FwdRB {
    match k {
        super::Fwd16::Dct => super::FwdRB::Dct,
        super::Fwd16::Adst => super::FwdRB::Adst,
        super::Fwd16::Idtx => super::FwdRB::Idtx,
    }
}

/// `lowbd_fwd_txfm2d_16x16_neon` (:2294) — two 16-pt column halves on x8
/// lanes, `vrshrq 2`, four `transpose_arrays_s16_8x8` into a 16x16 staging
/// buffer, then two 16-pt row halves with the plain (non-rect) store at
/// stride 16. Bound `FWD16_I16_BOUND[kc][kr]`.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fwd_16x16_fused(
    _t: archmage::NeonToken,
    kc: super::Fwd16,
    kr: super::Fwd16,
    input: &[i16],
    output: &mut [i32],
    stride: usize,
    cos_bit_col: i32,
    cos_bit_row: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    let bound = super::FWD16_I16_BOUND[super::fwd16_idx(kc)][super::fwd16_idx(kr)] as u16;
    let mut mx = vdupq_n_u16(0);
    let mut w = [vdupq_n_s16(0); 32];
    for i in 0..2 {
        let mut b = [vdupq_n_s16(0); 16];
        for r in 0..16 {
            let s = if ud_flip { 15 - r } else { r };
            let Some(row) = input.get(s * stride + 8 * i..s * stride + 8 * i + 8) else {
                return false;
            };
            let x = vld1q_s16(<&[i16; 8]>::try_from(row).unwrap());
            mx = vmaxq_u16(mx, vreinterpretq_u16_s16(vabsq_s16(x)));
            b[r] = vshlq_n_s16::<2>(x);
        }
        let mut col = run_16_x8(fwd16_as_rb(kc), &b, cos_bit_col);
        shr2_x8(&mut col);
        w[8 * i..8 * i + 8].copy_from_slice(&transpose8x8(&col[..8].try_into().unwrap()));
        w[16 + 8 * i..16 + 8 * i + 8].copy_from_slice(&transpose8x8(&col[8..].try_into().unwrap()));
    }
    if vmaxvq_u16(mx) > bound {
        return false;
    }

    // w[c*16 + r'] — hmm: buf1[j*16 + 8*i] holds column (8i+j)'s 16 rows.
    for i in 0..2 {
        let mut g: [int16x8_t; 16] = w[i * 16..i * 16 + 16].try_into().unwrap();
        if lr_flip {
            g.reverse();
        }
        let out = run_16_x8(fwd16_as_rb(kr), &g, cos_bit_row);
        for (j, v) in out.iter().enumerate() {
            let Some(d) = output.get_mut(j * 16 + 8 * i..j * 16 + 8 * i + 8) else {
                return false;
            };
            vst1q_s32(
                <&mut [i32; 4]>::try_from(&mut d[..4]).unwrap(),
                vmovl_s16(vget_low_s16(*v)),
            );
            vst1q_s32(
                <&mut [i32; 4]>::try_from(&mut d[4..]).unwrap(),
                vmovl_s16(vget_high_s16(*v)),
            );
        }
    }
    true
}
