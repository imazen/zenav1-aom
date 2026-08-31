//! Reference-frame scale factors — port of libaom v3.14.1 `av1/common/scale.c`
//! and the inline helpers in `av1/common/scale.h`.
//!
//! AV1 lets a reference frame have different dimensions from the frame being
//! coded (within a 2x-down / 16x-up window). Everything downstream — the
//! scaled convolve, the sub-pel parameter derivation, and the encoder's
//! `av1_is_scaled` branches all over the motion search — keys off the four
//! fixed-point numbers this module produces.
//!
//! | Rust | C |
//! |---|---|
//! | [`ScaleFactors::for_frame`] | `av1_setup_scale_factors_for_frame` (scale.c:44) |
//! | [`ScaleFactors::scaled_x`] / [`ScaleFactors::scaled_y`] | `av1_scaled_x` / `av1_scaled_y` (scale.h:36/45) |
//! | [`scale_mv`] | `av1_scale_mv` (scale.c:33) |
//! | [`ScaleFactors::is_valid`] / [`ScaleFactors::is_scaled`] | `av1_is_valid_scale` / `av1_is_scaled` (scale.h:65/70) |
//! | [`valid_ref_frame_size`] | `valid_ref_frame_size` (scale.h:77) |
//!
//! # Differential coverage
//! `tests/scale_diff.rs`, tier 1 against the real exported C.

/// `REF_SCALE_SHIFT` (scale.h:24).
pub const REF_SCALE_SHIFT: i32 = 14;
/// `REF_NO_SCALE` (scale.h:25) — the fixed-point value meaning 1:1.
pub const REF_NO_SCALE: i32 = 1 << REF_SCALE_SHIFT;
/// `REF_INVALID_SCALE` (scale.h:26) — the sentinel
/// [`ScaleFactors::for_frame`] writes when the size ratio is out of spec.
pub const REF_INVALID_SCALE: i32 = -1;
/// `SUBPEL_BITS` (aom_dsp/aom_filter.h:23).
const SUBPEL_BITS: i32 = 4;
/// `SCALE_SUBPEL_BITS` (aom_filter.h:28).
const SCALE_SUBPEL_BITS: i32 = 10;
/// `SCALE_EXTRA_BITS` (aom_filter.h:31).
const SCALE_EXTRA_BITS: i32 = SCALE_SUBPEL_BITS - SUBPEL_BITS;

/// `ROUND_POWER_OF_TWO_SIGNED_64` (aom_ports/mem.h:57): rounds the MAGNITUDE
/// and restores the sign, which is **not** the same as an arithmetic shift with
/// a `+half` bias — at a negative half-way value the two differ by one.
#[inline]
fn round_pow2_signed_64(value: i64, n: i32) -> i64 {
    if value < 0 {
        -(((-value) + (1i64 << (n - 1))) >> n)
    } else {
        (value + (1i64 << (n - 1))) >> n
    }
}

/// `ROUND_POWER_OF_TWO(value, n)` for i32.
#[inline]
fn round_pow2(value: i32, n: i32) -> i32 {
    (value + (1 << (n - 1))) >> n
}

/// `struct scale_factors` (scale.h:28).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleFactors {
    /// `x_scale_fp` — horizontal fixed-point scale factor.
    pub x_scale_fp: i32,
    /// `y_scale_fp`
    pub y_scale_fp: i32,
    /// `x_step_q4`
    pub x_step_q4: i32,
    /// `y_step_q4`
    pub y_step_q4: i32,
}

/// `valid_ref_frame_size` (scale.h:77) — AV1 spec §6.8.6: a reference may be at
/// most 2x larger and at most 16x smaller than the current frame, per axis.
#[inline]
pub fn valid_ref_frame_size(ref_w: i32, ref_h: i32, this_w: i32, this_h: i32) -> bool {
    2 * this_w >= ref_w && 2 * this_h >= ref_h && this_w <= 16 * ref_w && this_h <= 16 * ref_h
}

/// `get_fixed_point_scale_factor` (scale.c:19).
#[inline]
fn fixed_point_scale_factor(other_size: i32, this_size: i32) -> i32 {
    ((other_size << REF_SCALE_SHIFT) + this_size / 2) / this_size
}

/// `fixed_point_scale_to_coarse_point_scale` (scale.c:28).
#[inline]
fn to_coarse_point_scale(scale_fp: i32) -> i32 {
    round_pow2(scale_fp, REF_SCALE_SHIFT - SCALE_SUBPEL_BITS)
}

impl ScaleFactors {
    /// `av1_setup_scale_factors_for_frame` (scale.c:44).
    ///
    /// On an out-of-spec size ratio C writes `REF_INVALID_SCALE` into BOTH
    /// scale fields and **leaves `x_step_q4` / `y_step_q4` untouched** — it
    /// returns early before setting them. This port zeroes them instead of
    /// inventing a value, and the invalid case is only ever consumed through
    /// [`ScaleFactors::is_valid`], which reads the scale fields alone.
    #[must_use]
    pub fn for_frame(other_w: i32, other_h: i32, this_w: i32, this_h: i32) -> Self {
        if !valid_ref_frame_size(other_w, other_h, this_w, this_h) {
            return ScaleFactors {
                x_scale_fp: REF_INVALID_SCALE,
                y_scale_fp: REF_INVALID_SCALE,
                x_step_q4: 0,
                y_step_q4: 0,
            };
        }
        let x_scale_fp = fixed_point_scale_factor(other_w, this_w);
        let y_scale_fp = fixed_point_scale_factor(other_h, this_h);
        ScaleFactors {
            x_scale_fp,
            y_scale_fp,
            x_step_q4: to_coarse_point_scale(x_scale_fp),
            y_step_q4: to_coarse_point_scale(y_scale_fp),
        }
    }

    /// `av1_is_valid_scale` (scale.h:65).
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.x_scale_fp != REF_INVALID_SCALE && self.y_scale_fp != REF_INVALID_SCALE
    }

    /// `av1_is_scaled` (scale.h:70): valid AND not 1:1 on at least one axis.
    #[inline]
    #[must_use]
    pub fn is_scaled(&self) -> bool {
        self.is_valid() && (self.x_scale_fp != REF_NO_SCALE || self.y_scale_fp != REF_NO_SCALE)
    }

    /// `av1_scaled_x` (scale.h:36). `val` is q4.
    #[inline]
    #[must_use]
    pub fn scaled_x(&self, val: i32) -> i32 {
        let off = (self.x_scale_fp - (1 << REF_SCALE_SHIFT)) * (1 << (SUBPEL_BITS - 1));
        let tval = i64::from(val) * i64::from(self.x_scale_fp) + i64::from(off);
        round_pow2_signed_64(tval, REF_SCALE_SHIFT - SCALE_EXTRA_BITS) as i32
    }

    /// `av1_scaled_y` (scale.h:45). `val` is q4.
    #[inline]
    #[must_use]
    pub fn scaled_y(&self, val: i32) -> i32 {
        let off = (self.y_scale_fp - (1 << REF_SCALE_SHIFT)) * (1 << (SUBPEL_BITS - 1));
        let tval = i64::from(val) * i64::from(self.y_scale_fp) + i64::from(off);
        round_pow2_signed_64(tval, REF_SCALE_SHIFT - SCALE_EXTRA_BITS) as i32
    }
}

/// `av1_scale_mv` (scale.c:33): map a q4 MV at integer block position `(x, y)`
/// through the scale factors, as a delta from the scaled block origin.
///
/// Returns `(row, col)` — C's `MV32` is `{ row, col }`, i.e. **y first**, and
/// the `y`/`x` arguments are the other way round from the return order.
#[must_use]
pub fn scale_mv(mv: (i32, i32), x: i32, y: i32, sf: &ScaleFactors) -> (i32, i32) {
    let (mv_row, mv_col) = mv;
    let x_off_q4 = sf.scaled_x(x << SUBPEL_BITS);
    let y_off_q4 = sf.scaled_y(y << SUBPEL_BITS);
    (
        sf.scaled_y((y << SUBPEL_BITS) + mv_row) - y_off_q4,
        sf.scaled_x((x << SUBPEL_BITS) + mv_col) - x_off_q4,
    )
}
