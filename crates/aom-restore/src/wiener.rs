//! The Wiener restoration convolution — `av1_wiener_convolve_add_src_c` /
//! `av1_highbd_wiener_convolve_add_src_c` (av1/common/convolve.c) on u16
//! planes.
//!
//! Both C variants run the same two-pass separable filter: a horizontal pass
//! into a u16 intermediate at extra precision (offset so values stay
//! non-negative), then a vertical pass removing the offset and clipping to
//! the pixel range. The filters are FIXED 8-tap kernels (the `x_step_q4 = 16`
//! / `get_filter_base` subpel machinery degenerates to "apply the one kernel
//! at integer positions" — the wiener taps occupy slots 0..6, slot 7 is 0).
//! `get_conv_params_wiener` picks the rounding split: `round_0 = 3`,
//! `round_1 = 11`, shifted by 2 at 12-bit so the intermediate fits 16 bits.
//!
//! The lowbd variant computes `h + 7` intermediate rows and the highbd one
//! `h + 8`; the vertical pass reads exactly `h + 7` (output row `h-1` reads
//! intermediate rows `h-1 .. h+6`), so the extra highbd row is dead work —
//! this port computes `h + 7` for both (verified byte-identical to both C
//! variants in tests/wiener_diff.rs).

/// `FILTER_BITS` (av1/common/filter.h).
const FILTER_BITS: i32 = 7;
/// `SUBPEL_TAPS` (the fixed kernel length).
const SUBPEL_TAPS: usize = 8;
/// `MAX_SB_SIZE` — the C intermediate row stride.
const MAX_SB_SIZE: usize = 128;

/// `get_conv_params_wiener` (av1/common/convolve.h): `(round_0, round_1)`.
pub fn conv_params_wiener(bd: i32) -> (i32, i32) {
    let mut round_0 = 3; // WIENER_ROUND0_BITS
    let mut round_1 = 2 * FILTER_BITS - round_0;
    let intbufrange = bd + FILTER_BITS - round_0 + 2;
    if intbufrange > 16 {
        round_0 += intbufrange - 16;
        round_1 -= intbufrange - 16;
    }
    (round_0, round_1)
}

/// `ROUND_POWER_OF_TWO` on a signed value (C's arithmetic shift).
#[inline]
fn round_power_of_two(v: i32, n: i32) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

/// `av1_[highbd_]wiener_convolve_add_src_c`: filter a `w x h` block whose
/// top-left source sample is `src[src_off]` into `dst[dst_off]`. The source
/// is read at `[-3, +4]` rows/cols around each output position (slot-7 taps
/// are zero but the sample is still loaded, exactly like C) — the caller
/// provides a buffer with sufficient margins. `w <= 128`.
#[allow(clippy::too_many_arguments)]
pub fn wiener_convolve_add_src(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    hfilter: &[i16; 8],
    vfilter: &[i16; 8],
    w: usize,
    h: usize,
    bd: i32,
) {
    assert!(w <= MAX_SB_SIZE);
    let (round_0, round_1) = conv_params_wiener(bd);
    let intermediate_height = h + SUBPEL_TAPS - 1;
    let mut temp = vec![0u16; intermediate_height * MAX_SB_SIZE];

    // convolve_add_src_horiz_hip: src starts SUBPEL_TAPS/2 - 1 = 3 rows above
    // and 3 columns left of the output origin.
    let clamp_limit = 1i32 << (bd + 1 + FILTER_BITS - round_0); // WIENER_CLAMP_LIMIT
    let horiz_base = src_off as isize - 3 * src_stride as isize - 3;
    for y in 0..intermediate_height {
        for x in 0..w {
            let s = (horiz_base + (y * src_stride + x) as isize) as usize;
            let src_x = &src[s..s + SUBPEL_TAPS];
            let rounding = ((src_x[3] as i32) << FILTER_BITS) + (1 << (bd + FILTER_BITS - 1));
            let mut sum = rounding;
            for k in 0..SUBPEL_TAPS {
                sum += src_x[k] as i32 * hfilter[k] as i32;
            }
            temp[y * MAX_SB_SIZE + x] =
                round_power_of_two(sum, round_0).clamp(0, clamp_limit - 1) as u16;
        }
    }

    // convolve_add_src_vert_hip: reads intermediate rows y .. y+7 for output
    // row y; the centre-tap offset is removed and the result clipped to bd.
    let pixel_max = (1i32 << bd) - 1;
    for x in 0..w {
        for y in 0..h {
            let base = y * MAX_SB_SIZE + x;
            let rounding =
                ((temp[base + 3 * MAX_SB_SIZE] as i32) << FILTER_BITS) - (1 << (bd + round_1 - 1));
            let mut sum = rounding;
            for k in 0..SUBPEL_TAPS {
                sum += temp[base + k * MAX_SB_SIZE] as i32 * vfilter[k] as i32;
            }
            dst[dst_off + y * dst_stride + x] =
                round_power_of_two(sum, round_1).clamp(0, pixel_max) as u16;
        }
    }
}
