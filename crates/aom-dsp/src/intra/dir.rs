//! Directional intra predictors z1/z2/z3, bit-exact port of libaom v3.14.1
//! `av1/common/reconintra.c` (`av1_dr_prediction_z{1,2,3}_c`). Plus the
//! `dr_intra_derivative` angle table and `av1_get_dx/dy`.
//!
//! z2 reads `above`/`left` at negative offsets, so callers must provide edges
//! via [`EdgeRef`] with front padding.

/// `dr_intra_derivative[90]` from `reconintra.h`.
#[rustfmt::skip]
pub static DR_INTRA_DERIVATIVE: [i16; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0,
    178, 0, 0, 151, 0, 0, 132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80,
    0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0, 45, 0, 0, 0, 40, 0, 0, 35,
    0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0, 0,
    7, 0, 0, 3, 0, 0,
];

pub fn get_dx(angle: i32) -> i32 {
    if angle > 0 && angle < 90 {
        DR_INTRA_DERIVATIVE[angle as usize] as i32
    } else if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32
    } else {
        1
    }
}

pub fn get_dy(angle: i32) -> i32 {
    if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32
    } else if angle > 180 && angle < 270 {
        DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32
    } else {
        1
    }
}

/// An edge (above/left) with `pad` valid samples before index 0.
pub struct EdgeRef<'a> {
    data: &'a [u8],
    pad: usize,
}
impl<'a> EdgeRef<'a> {
    pub fn new(data: &'a [u8], pad: usize) -> Self {
        EdgeRef { data, pad }
    }
    #[inline]
    fn at(&self, i: i32) -> i32 {
        self.data[(self.pad as i32 + i) as usize] as i32
    }
}

#[inline]
fn rpo2_5(v: i32) -> u8 {
    ((v + 16) >> 5) as u8
}

#[inline]
fn rpo2_5_16(v: i32) -> u16 {
    ((v + 16) >> 5) as u16
}

/// An edge (above/left) of 10/12-bit samples with `pad` valid samples before
/// index 0 (the highbd analogue of [`EdgeRef`]).
pub struct EdgeRef16<'a> {
    data: &'a [u16],
    pad: usize,
}
impl<'a> EdgeRef16<'a> {
    pub fn new(data: &'a [u16], pad: usize) -> Self {
        EdgeRef16 { data, pad }
    }
    #[inline]
    fn at(&self, i: i32) -> i32 {
        self.data[(self.pad as i32 + i) as usize] as i32
    }
    /// Absolute index of edge sample `i` in the backing array.
    #[inline]
    fn idx(&self, i: i32) -> usize {
        (self.pad as i32 + i) as usize
    }
    #[inline]
    fn data(&self) -> &[u16] {
        self.data
    }
}

/// `av1_dr_prediction_z1_c` (dy == 1, dx > 0).
pub fn z1(dst: &mut [u8], stride: usize, bw: usize, bh: usize, above: &EdgeRef, up: i32, dx: i32) {
    let max_base_x = (((bw + bh) as i32) - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut x = dx;
    for r in 0..bh {
        let base = x >> frac_bits;
        let shift = ((x << up) & 0x3F) >> 1;
        if base >= max_base_x {
            let fillv = above.at(max_base_x) as u8;
            for rr in r..bh {
                for c in 0..bw {
                    dst[rr * stride + c] = fillv;
                }
            }
            return;
        }
        let mut base = base;
        for c in 0..bw {
            dst[r * stride + c] = if base < max_base_x {
                rpo2_5(above.at(base) * (32 - shift) + above.at(base + 1) * shift)
            } else {
                above.at(max_base_x) as u8
            };
            base += base_inc;
        }
        x += dx;
    }
}

/// `av1_dr_prediction_z2_c` (dx > 0, dy > 0).
#[allow(clippy::too_many_arguments)]
pub fn z2(
    dst: &mut [u8],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef,
    left: &EdgeRef,
    up_above: i32,
    up_left: i32,
    dx: i32,
    dy: i32,
) {
    let min_base_x = -(1 << up_above);
    let frac_bits_x = 6 - up_above;
    let frac_bits_y = 6 - up_left;
    for r in 0..bh {
        for c in 0..bw {
            let y = (r + 1) as i32;
            let x = ((c as i32) << 6) - y * dx;
            let base_x = x >> frac_bits_x;
            let val = if base_x >= min_base_x {
                let shift = ((x * (1 << up_above)) & 0x3F) >> 1;
                rpo2_5(above.at(base_x) * (32 - shift) + above.at(base_x + 1) * shift)
            } else {
                let x2 = (c + 1) as i32;
                let y2 = ((r as i32) << 6) - x2 * dy;
                let base_y = y2 >> frac_bits_y;
                let shift = ((y2 * (1 << up_left)) & 0x3F) >> 1;
                rpo2_5(left.at(base_y) * (32 - shift) + left.at(base_y + 1) * shift)
            };
            dst[r * stride + c] = val;
        }
    }
}

/// `av1_dr_prediction_z3_c` (dx == 1, dy > 0).
pub fn z3(dst: &mut [u8], stride: usize, bw: usize, bh: usize, left: &EdgeRef, up: i32, dy: i32) {
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut y = dy;
    for c in 0..bw {
        let mut base = y >> frac_bits;
        let shift = ((y << up) & 0x3F) >> 1;
        for r in 0..bh {
            if base < max_base_y {
                dst[r * stride + c] =
                    rpo2_5(left.at(base) * (32 - shift) + left.at(base + 1) * shift);
                base += base_inc;
            } else {
                let fillv = left.at(max_base_y) as u8;
                for rr in r..bh {
                    dst[rr * stride + c] = fillv;
                }
                break;
            }
        }
        y += dy;
    }
}

// ===========================================================================
// Highbd dispatching entries
// ===========================================================================
//
// Each of these checks ONE runtime bound over the edge span it will read
// ([`crate::intra::dir_simd::span_fits_i16`], `O(bw + bh)`) and then runs the
// i16-lane vector kernel over the CONTIGUOUS tap runs, falling back to the
// scalar core otherwise. The `*_scalar` bodies below are the never-dispatched
// references the differentials compare against (`tests/dir_simd_diff.rs`) and
// are exactly the C transcriptions they always were.

use crate::intra::dir_simd::{MIN_VEC_RUN, span_fits_i16, two_tap_run};

/// The z1 vector-path predicate, named so the driver and the reach test cannot
/// drift apart (`dir_simd::reach`). Contiguous taps only, a run long enough to
/// pay for a 16-lane round trip, and every sample the run reads inside the i16
/// bound.
pub(crate) fn z1_vec_applies(above: &EdgeRef16, bw: usize, bh: usize, up: i32) -> bool {
    let max_base_x = (((bw + bh) as i32) - 1) << up;
    up == 0
        && bw >= MIN_VEC_RUN
        && span_fits_i16(above.data(), above.idx(0), above.idx(max_base_x))
}

/// The z2 ABOVE-half vector-path predicate (the left half is a gather and stays
/// scalar at every input).
pub(crate) fn z2_vec_applies(above: &EdgeRef16, bw: usize, up_above: i32) -> bool {
    let min_base_x = -(1 << up_above);
    let hi = above.idx(((bw as i32 - 1) << up_above) + 1);
    up_above == 0 && span_fits_i16(above.data(), above.idx(min_base_x), hi)
}

/// The z3 vector-path predicate.
pub(crate) fn z3_vec_applies(left: &EdgeRef16, bw: usize, bh: usize, up: i32) -> bool {
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    up == 0 && bh >= MIN_VEC_RUN && span_fits_i16(left.data(), left.idx(0), left.idx(max_base_y))
}

/// `av1_highbd_dr_prediction_z1_c` (dy == 1, dx > 0). Highbd analogue of [`z1`];
/// `bd` is unused (the two-tap interpolation of in-range samples stays in range).
pub fn z1_high(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef16,
    up: i32,
    dx: i32,
) {
    let max_base_x = (((bw + bh) as i32) - 1) << up;
    // The vector kernel handles contiguous runs only; `up == 1` is a stride-2
    // gather. The span read is `above[0 ..= max_base_x]` (the `+1` tap of the
    // last interpolated output is `<= max_base_x`).
    if !z1_vec_applies(above, bw, bh, up) {
        z1_high_scalar(dst, stride, bw, bh, above, up, dx);
        return;
    }
    let mut x = dx;
    for r in 0..bh {
        let base = x >> 6;
        let shift = (x & 0x3F) >> 1;
        let row = r * stride;
        if base >= max_base_x {
            let fillv = above.at(max_base_x) as u16;
            for rr in r..bh {
                dst[rr * stride..rr * stride + bw].fill(fillv);
            }
            return;
        }
        // Outputs `c` with `base + c < max_base_x` interpolate; the rest fill.
        let n_act = bw.min((max_base_x - base) as usize);
        two_tap_run(
            &mut dst[row..row + n_act],
            above.data(),
            above.idx(base),
            shift,
            n_act,
        );
        if n_act < bw {
            let fillv = above.at(max_base_x) as u16;
            dst[row + n_act..row + bw].fill(fillv);
        }
        x += dx;
    }
}

/// `av1_highbd_dr_prediction_z1_c` — the never-dispatched scalar core.
pub fn z1_high_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef16,
    up: i32,
    dx: i32,
) {
    let max_base_x = (((bw + bh) as i32) - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut x = dx;
    for r in 0..bh {
        let base = x >> frac_bits;
        let shift = ((x << up) & 0x3F) >> 1;
        if base >= max_base_x {
            let fillv = above.at(max_base_x) as u16;
            for rr in r..bh {
                for c in 0..bw {
                    dst[rr * stride + c] = fillv;
                }
            }
            return;
        }
        let mut base = base;
        for c in 0..bw {
            dst[r * stride + c] = if base < max_base_x {
                rpo2_5_16(above.at(base) * (32 - shift) + above.at(base + 1) * shift)
            } else {
                above.at(max_base_x) as u16
            };
            base += base_inc;
        }
        x += dx;
    }
}

/// `av1_highbd_dr_prediction_z2_c` (dx > 0, dy > 0). Highbd analogue of [`z2`].
///
/// Only the ABOVE half is vectorized. Within a row `x(c) = (c << 6) - y*dx` is
/// strictly increasing in `c`, so `base_x(c) >= min_base_x` is a **suffix**
/// condition — the row splits into a left-gather prefix and an above prefix-free
/// suffix, with no per-pixel branch. On that suffix `base_x` advances by exactly
/// `1 << up_above` per column (because `64` is a multiple of `2^(6-up_above)`)
/// and `shift` is CONSTANT (adding `(c-c0) * 2^(6+up_above)` cannot change
/// `(x << up_above) & 0x3F`), so the suffix is exactly one contiguous two-tap
/// run when `up_above == 0`. The left half's `base_y` is not affine in `c`, so
/// it stays scalar.
#[allow(clippy::too_many_arguments)]
pub fn z2_high(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef16,
    left: &EdgeRef16,
    up_above: i32,
    up_left: i32,
    dx: i32,
    dy: i32,
) {
    let min_base_x = -(1 << up_above);
    let frac_bits_x = 6 - up_above;
    let frac_bits_y = 6 - up_left;
    // Span the above suffix can read: `[min_base_x, ((bw-1) << up_above) + 1]`.
    if !z2_vec_applies(above, bw, up_above) {
        z2_high_scalar(dst, stride, bw, bh, above, left, up_above, up_left, dx, dy);
        return;
    }
    for r in 0..bh {
        let y = (r + 1) as i32;
        let row = r * stride;
        let mut c = 0usize;
        // Left-gather prefix, verbatim from the scalar core.
        while c < bw {
            let x = ((c as i32) << 6) - y * dx;
            if (x >> frac_bits_x) >= min_base_x {
                break;
            }
            let x2 = (c + 1) as i32;
            let y2 = ((r as i32) << 6) - x2 * dy;
            let base_y = y2 >> frac_bits_y;
            let shift = ((y2 * (1 << up_left)) & 0x3F) >> 1;
            dst[row + c] = rpo2_5_16(left.at(base_y) * (32 - shift) + left.at(base_y + 1) * shift);
            c += 1;
        }
        if c < bw {
            let x = ((c as i32) << 6) - y * dx;
            let base_x = x >> frac_bits_x;
            let shift = (x & 0x3F) >> 1;
            let n = bw - c;
            if n >= MIN_VEC_RUN {
                two_tap_run(
                    &mut dst[row + c..row + bw],
                    above.data(),
                    above.idx(base_x),
                    shift,
                    n,
                );
            } else {
                crate::intra::dir_simd::two_tap_run_scalar(
                    &mut dst[row + c..row + bw],
                    above.data(),
                    above.idx(base_x),
                    shift,
                    n,
                );
            }
        }
    }
}

/// `av1_highbd_dr_prediction_z2_c` — the never-dispatched scalar core.
#[allow(clippy::too_many_arguments)]
pub fn z2_high_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef16,
    left: &EdgeRef16,
    up_above: i32,
    up_left: i32,
    dx: i32,
    dy: i32,
) {
    let min_base_x = -(1 << up_above);
    let frac_bits_x = 6 - up_above;
    let frac_bits_y = 6 - up_left;
    for r in 0..bh {
        for c in 0..bw {
            let y = (r + 1) as i32;
            let x = ((c as i32) << 6) - y * dx;
            let base_x = x >> frac_bits_x;
            let val = if base_x >= min_base_x {
                let shift = ((x * (1 << up_above)) & 0x3F) >> 1;
                rpo2_5_16(above.at(base_x) * (32 - shift) + above.at(base_x + 1) * shift)
            } else {
                let x2 = (c + 1) as i32;
                let y2 = ((r as i32) << 6) - x2 * dy;
                let base_y = y2 >> frac_bits_y;
                let shift = ((y2 * (1 << up_left)) & 0x3F) >> 1;
                rpo2_5_16(left.at(base_y) * (32 - shift) + left.at(base_y + 1) * shift)
            };
            dst[r * stride + c] = val;
        }
    }
}

/// `av1_highbd_dr_prediction_z3_c` (dx == 1, dy > 0). Highbd analogue of [`z3`].
///
/// z3 walks a COLUMN per outer iteration, so the interpolated run is contiguous
/// in the edge but strided in `dst`. The kernel writes the column into a stack
/// buffer and the driver scatters it; libaom instead builds a transposed block
/// and transposes it, which needs an i16 lane shuffle magetypes does not expose
/// (`transform::simd::prims16` audits that surface).
pub fn z3_high(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &EdgeRef16,
    up: i32,
    dy: i32,
) {
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    if !z3_vec_applies(left, bw, bh, up) {
        z3_high_scalar(dst, stride, bw, bh, left, up, dy);
        return;
    }
    let mut col = [0u16; 64];
    let mut y = dy;
    for c in 0..bw {
        let base = y >> 6;
        let shift = (y & 0x3F) >> 1;
        let n_act = if base >= max_base_y {
            0
        } else {
            bh.min((max_base_y - base) as usize)
        };
        if n_act > 0 {
            two_tap_run(
                &mut col[..n_act],
                left.data(),
                left.idx(base),
                shift,
                n_act,
            );
        }
        for (r, &v) in col[..n_act].iter().enumerate() {
            dst[r * stride + c] = v;
        }
        if n_act < bh {
            let fillv = left.at(max_base_y) as u16;
            for r in n_act..bh {
                dst[r * stride + c] = fillv;
            }
        }
        y += dy;
    }
}

/// `av1_highbd_dr_prediction_z3_c` — the never-dispatched scalar core.
pub fn z3_high_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &EdgeRef16,
    up: i32,
    dy: i32,
) {
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut y = dy;
    for c in 0..bw {
        let mut base = y >> frac_bits;
        let shift = ((y << up) & 0x3F) >> 1;
        for r in 0..bh {
            if base < max_base_y {
                dst[r * stride + c] =
                    rpo2_5_16(left.at(base) * (32 - shift) + left.at(base + 1) * shift);
                base += base_inc;
            } else {
                let fillv = left.at(max_base_y) as u16;
                for rr in r..bh {
                    dst[rr * stride + c] = fillv;
                }
                break;
            }
        }
        y += dy;
    }
}

#[cfg(test)]
mod reach {
    //! REACH: how much of the encoder's real domain the runtime tap bound
    //! admits, and the OTHER side — that it genuinely declines outside it.
    //! Bit-identity would also pass on a gate that admitted nothing
    //! (`transform::simd::lowbd16_fwd::reach` is the precedent), so these
    //! counts are PINNED: a change that quietly narrows the gate fails here
    //! rather than silently costing the lever.

    use super::*;

    const PAD: usize = 16;
    const BUF: usize = 160;
    const TX_DIMS: [(usize, usize); 19] = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (4, 8),
        (8, 4),
        (8, 16),
        (16, 8),
        (16, 32),
        (32, 16),
        (32, 64),
        (64, 32),
        (4, 16),
        (16, 4),
        (8, 32),
        (32, 8),
        (16, 64),
        (64, 16),
    ];

    #[test]
    fn the_gate_fires_across_the_bd8_grid() {
        // Worst-case bd8 edge: every sample at 255, which is the largest value
        // the encoder's 8-bit reconstruction plane can hold.
        let buf = vec![255u16; BUF];
        let e = EdgeRef16::new(&buf, PAD);
        let (mut z1n, mut z2n, mut z3n) = (0usize, 0usize, 0usize);
        let (mut z1d, mut z3d) = (0usize, 0usize);
        for &(bw, bh) in &TX_DIMS {
            for &up in &[0i32, 1] {
                if z1_vec_applies(&e, bw, bh, up) {
                    z1n += 1;
                } else if up == 0 {
                    z1d += 1;
                }
                if z2_vec_applies(&e, bw, up) {
                    z2n += 1;
                }
                if z3_vec_applies(&e, bw, bh, up) {
                    z3n += 1;
                } else if up == 0 {
                    z3d += 1;
                }
            }
        }
        // 19 shapes x {up=0, up=1}. `up == 1` never vectorizes (stride-2
        // gather), so the ceiling is 19 each.
        // z1/z3 additionally need the vectorized dimension >= MIN_VEC_RUN: the
        // THREE shapes with bw == 4 ((4,4), (4,8), (4,16)) decline for z1, and
        // the three with bh == 4 ((4,4), (8,4), (16,4)) for z3.
        assert_eq!((z1n, z1d), (16, 3), "z1 admitted/declined at bd8");
        assert_eq!(z2n, 19, "z2 admitted at bd8 (no width floor: the run length \
                             varies per row, so the length test is per-run)");
        assert_eq!((z3n, z3d), (16, 3), "z3 admitted/declined at bd8");
    }

    #[test]
    fn the_gate_declines_above_the_tap_bound() {
        // bd10 max is exactly the bound and is admitted; one over it is not.
        let mut buf = vec![1023u16; BUF];
        let e = EdgeRef16::new(&buf, PAD);
        assert!(z1_vec_applies(&e, 16, 16, 0));
        assert!(z2_vec_applies(&e, 16, 0));
        assert!(z3_vec_applies(&e, 16, 16, 0));
        buf[PAD + 5] = 1024;
        let e = EdgeRef16::new(&buf, PAD);
        assert!(!z1_vec_applies(&e, 16, 16, 0), "1024 must decline");
        assert!(!z2_vec_applies(&e, 16, 0), "1024 must decline");
        assert!(!z3_vec_applies(&e, 16, 16, 0), "1024 must decline");
        // A bd12-range edge declines everywhere.
        let buf = vec![4095u16; BUF];
        let e = EdgeRef16::new(&buf, PAD);
        for &(bw, bh) in &TX_DIMS {
            assert!(!z1_vec_applies(&e, bw, bh, 0));
            assert!(!z2_vec_applies(&e, bw, 0));
            assert!(!z3_vec_applies(&e, bw, bh, 0));
        }
    }
}
