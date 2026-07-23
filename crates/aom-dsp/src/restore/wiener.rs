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

use archmage::prelude::*;

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

/// Reusable intermediate-row scratch for [`wiener_convolve_add_src_into`],
/// killing the per-call `vec![0u16; (h + 7) * 128]` allocation (measured
/// 12.8 % of the kernel's Ir on `dec_352x288_q32`). Reuse is byte-identical:
/// both the SIMD and scalar passes write every `temp` cell the vertical pass
/// reads (rows `0..h+7`, cols `0..w`) before reading it — same argument as
/// the 2026-07-19 `ReconScratch`/`InvTxfmScratch` landing.
#[derive(Default)]
pub struct WienerScratch {
    temp: Vec<u16>,
}

impl WienerScratch {
    pub fn new() -> Self {
        Self::default()
    }
    fn ensure(&mut self, n: usize) -> &mut [u16] {
        if self.temp.len() < n {
            self.temp.resize(n, 0);
        }
        &mut self.temp[..n]
    }
}

/// `av1_[highbd_]wiener_convolve_add_src_c`: filter a `w x h` block whose
/// top-left source sample is `src[src_off]` into `dst[dst_off]`. The source
/// is read at `[-3, +4]` rows/cols around each output position (slot-7 taps
/// are zero but the sample is still loaded, exactly like C) — the caller
/// provides a buffer with sufficient margins. `w <= 128`.
///
/// SIMD-dispatched (Gate 3): width >= 8 takes the magetypes i32x8 kernel —
/// bit-identical to [`wiener_convolve_add_src_scalar`] by construction (no
/// reformulation at all: both passes run the scalar port's exact i32
/// expressions lane-wise; width tails re-run the LAST vector overlapped back
/// to `w-8`, recomputing identical pure per-column values) and by the
/// differentials (`kernels_diff.rs` drives THIS entry against the REAL C
/// kernels incl. odd widths; `wiener_simd_diff.rs` pins SIMD == scalar at
/// every token permutation). Width < 8 and the `AOM_FORCE_SCALAR` pin run
/// the scalar twin.
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
    let mut scratch = WienerScratch::new();
    wiener_convolve_add_src_into(
        src,
        src_off,
        src_stride,
        dst,
        dst_off,
        dst_stride,
        hfilter,
        vfilter,
        w,
        h,
        bd,
        &mut scratch,
    )
}

/// [`wiener_convolve_add_src`] with a caller-owned [`WienerScratch`] (the
/// frame walk's hot entry — one scratch per plane instead of one heap
/// allocation per 64-wide chunk).
#[allow(clippy::too_many_arguments)]
pub fn wiener_convolve_add_src_into(
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
    scratch: &mut WienerScratch,
) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    let temp = scratch.ensure((h + SUBPEL_TAPS - 1) * MAX_SB_SIZE);
    if w < 8 {
        return wiener_scalar_into(
            src, src_off, src_stride, dst, dst_off, dst_stride, hfilter, vfilter, w, h, bd, temp,
        );
    }
    archmage::incant!(
        wiener_impl(
            src, src_off, src_stride, dst, dst_off, dst_stride, hfilter, vfilter, w, h, bd, temp
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed port, verbatim.
#[allow(clippy::too_many_arguments)]
fn wiener_impl_scalar(
    _t: archmage::ScalarToken,
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
    temp: &mut [u16],
) {
    wiener_scalar_into(
        src, src_off, src_stride, dst, dst_off, dst_stride, hfilter, vfilter, w, h, bd, temp,
    )
}

/// `ROUND_POWER_OF_TWO` on 8 lanes with a runtime shift in 1..=15 (the
/// wiener rounds are per-call bd-derived constants; the match arm is
/// perfectly predicted).
macro_rules! shr_round_by {
    ($v:expr, $n:expr, $half:expr) => {
        match $n {
            1 => ($v + $half).shr_arithmetic_const::<1>(),
            2 => ($v + $half).shr_arithmetic_const::<2>(),
            3 => ($v + $half).shr_arithmetic_const::<3>(),
            4 => ($v + $half).shr_arithmetic_const::<4>(),
            5 => ($v + $half).shr_arithmetic_const::<5>(),
            6 => ($v + $half).shr_arithmetic_const::<6>(),
            7 => ($v + $half).shr_arithmetic_const::<7>(),
            8 => ($v + $half).shr_arithmetic_const::<8>(),
            9 => ($v + $half).shr_arithmetic_const::<9>(),
            10 => ($v + $half).shr_arithmetic_const::<10>(),
            11 => ($v + $half).shr_arithmetic_const::<11>(),
            12 => ($v + $half).shr_arithmetic_const::<12>(),
            13 => ($v + $half).shr_arithmetic_const::<13>(),
            14 => ($v + $half).shr_arithmetic_const::<14>(),
            _ => ($v + $half).shr_arithmetic_const::<15>(),
        }
    };
}

#[archmage::magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn wiener_impl(
    token: Token,
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
    temp: &mut [u16],
) {
    assert!(w >= 8 && w <= MAX_SB_SIZE);
    let (round_0, round_1) = conv_params_wiener(bd);
    let intermediate_height = h + SUBPEL_TAPS - 1;
    debug_assert_eq!(temp.len(), intermediate_height * MAX_SB_SIZE);

    // One-bounds-check `[u16; 8]` fixed-array load + `as i32` widen (LLVM:
    // vpmovzxwd) instead of 8 checked scalar loads via `from_fn`; the
    // lane VALUES are identical, so the arithmetic is untouched.
    let widen = |s: &[u16]| -> i32x8 {
        let a: [u16; 8] = s[..8].try_into().unwrap();
        i32x8::from_array(
            token,
            [
                a[0] as i32,
                a[1] as i32,
                a[2] as i32,
                a[3] as i32,
                a[4] as i32,
                a[5] as i32,
                a[6] as i32,
                a[7] as i32,
            ],
        )
    };
    // Fixed-array narrow store (single 16-byte copy) — same `v as u16` lane
    // narrowing as the previous per-element loop, one bounds check.
    macro_rules! store8 {
        ($dst:expr, $d0:expr, $v:expr) => {{
            let a = ($v).to_array();
            let n: [u16; 8] = [
                a[0] as u16,
                a[1] as u16,
                a[2] as u16,
                a[3] as u16,
                a[4] as u16,
                a[5] as u16,
                a[6] as u16,
                a[7] as u16,
            ];
            $dst[$d0..$d0 + 8].copy_from_slice(&n);
        }};
    }

    // ---- horizontal pass (lanes = 8 adjacent output columns) ----
    let clamp_limit = 1i32 << (bd + 1 + FILTER_BITS - round_0);
    let zero = i32x8::zero(token);
    let lim_v = i32x8::splat(token, clamp_limit - 1);
    let h_half = i32x8::splat(token, 1 << (round_0 - 1));
    let hbias = i32x8::splat(token, 1 << (bd + FILTER_BITS - 1));
    let htap: [i32x8; 8] = core::array::from_fn(|k| i32x8::splat(token, hfilter[k] as i32));
    let horiz_base = src_off as isize - 3 * src_stride as isize - 3;
    for y in 0..intermediate_height {
        let row = (horiz_base + (y * src_stride) as isize) as usize;
        let mut xs = 0usize;
        loop {
            let x0 = xs.min(w - 8); // overlap-back tail (recomputes identical values)
            let s0 = row + x0;
            // rounding = (src[+3] << FILTER_BITS) + (1 << (bd + FILTER_BITS - 1))
            let mut sum = widen(&src[s0 + 3..s0 + 11]).shl_const::<7>() + hbias;
            for k in 0..SUBPEL_TAPS {
                sum = sum + widen(&src[s0 + k..s0 + k + 8]) * htap[k];
            }
            let r = shr_round_by!(sum, round_0, h_half).clamp(zero, lim_v);
            store8!(temp, y * MAX_SB_SIZE + x0, r);
            if x0 + 8 >= w {
                break;
            }
            xs += 8;
        }
    }

    // ---- vertical pass (lanes = 8 adjacent output columns; iteration order
    // differs from the scalar port's x-outer loop, but every (x, y) output is
    // a pure function of `temp`, so the bytes are identical) ----
    let pixel_max = i32x8::splat(token, (1i32 << bd) - 1);
    let v_half = i32x8::splat(token, 1 << (round_1 - 1));
    let vbias = i32x8::splat(token, 1 << (bd + round_1 - 1));
    let vtap: [i32x8; 8] = core::array::from_fn(|k| i32x8::splat(token, vfilter[k] as i32));
    for y in 0..h {
        let mut xs = 0usize;
        loop {
            let x0 = xs.min(w - 8);
            let base = y * MAX_SB_SIZE + x0;
            let c0 = base + 3 * MAX_SB_SIZE;
            let mut sum = widen(&temp[c0..c0 + 8]).shl_const::<7>() - vbias;
            for k in 0..SUBPEL_TAPS {
                let o = base + k * MAX_SB_SIZE;
                sum = sum + widen(&temp[o..o + 8]) * vtap[k];
            }
            let r = shr_round_by!(sum, round_1, v_half).clamp(zero, pixel_max);
            store8!(dst, dst_off + y * dst_stride + x0, r);
            if x0 + 8 >= w {
                break;
            }
            xs += 8;
        }
    }
}

/// The scalar transcription (the reference twin — never SIMD-routed).
#[allow(clippy::too_many_arguments)]
pub fn wiener_convolve_add_src_scalar(
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
    let mut temp = vec![0u16; (h + SUBPEL_TAPS - 1) * MAX_SB_SIZE];
    wiener_scalar_into(
        src, src_off, src_stride, dst, dst_off, dst_stride, hfilter, vfilter, w, h, bd, &mut temp,
    )
}

/// The scalar body on a caller-provided intermediate buffer (identical
/// arithmetic; `temp` is fully written before it is read, so a reused
/// buffer is byte-identical to a fresh zeroed one).
#[allow(clippy::too_many_arguments)]
fn wiener_scalar_into(
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
    temp: &mut [u16],
) {
    assert!(w <= MAX_SB_SIZE);
    let (round_0, round_1) = conv_params_wiener(bd);
    let intermediate_height = h + SUBPEL_TAPS - 1;
    debug_assert_eq!(temp.len(), intermediate_height * MAX_SB_SIZE);

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
