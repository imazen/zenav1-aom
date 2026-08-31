//! Compound (two-reference) inter-prediction DSP — port of libaom v3.14.1's
//! wedge / difference-weighted / distance-weighted compound kernels.
//!
//! Every inter mode that blends **two** predictors goes through this module:
//!
//! ```text
//! COMPOUND_WEDGE     -> get_compound_type_mask -> contiguous_soft_mask (signed)
//! COMPOUND_DIFFWTD   -> build_compound_diffwtd_mask{,_d16,_highbd}
//! COMPOUND_DISTWTD   -> dist_wtd_comp_weight_assign  (offsets, not a mask)
//! COMPOUND_AVERAGE   -> the plain 1/2:1/2 case dist_wtd_comp_weight_assign returns
//! ```
//!
//! plus the three wedge RD-search primitives the encoder scores candidate
//! wedges with (`av1/encoder/wedge_utils.c`), which have no decoder counterpart
//! and are pure integer reductions over a residual pair.
//!
//! Ported C (all `av1/common/reconinter.c` unless noted):
//!
//! | Rust | C |
//! |---|---|
//! | [`wedge_sse_from_residuals`] | `av1_wedge_sse_from_residuals_c` (encoder/wedge_utils.c:52) |
//! | [`wedge_sign_from_residuals`] | `av1_wedge_sign_from_residuals_c` (wedge_utils.c:101) |
//! | [`wedge_compute_delta_squares`] | `av1_wedge_compute_delta_squares_c` (wedge_utils.c:123) |
//! | [`build_compound_diffwtd_mask`] | `av1_build_compound_diffwtd_mask_c` (:351) |
//! | [`build_compound_diffwtd_mask_d16`] | `av1_build_compound_diffwtd_mask_d16_c` (:319) |
//! | [`build_compound_diffwtd_mask_highbd`] | `av1_build_compound_diffwtd_mask_highbd_c` (:431) |
//! | [`get_compound_type_mask`] | `av1_get_compound_type_mask` (:290) |
//! | [`dist_wtd_comp_weight_assign`] | `av1_dist_wtd_comp_weight_assign` (:669) |
//!
//! The signed wedge mask fetch itself (`av1_get_contiguous_soft_mask`) lives in
//! the sibling [`super::interintra`] module, where the wedge codebook and the
//! master oblique masks already were — see
//! [`super::interintra::wedge_mask_signed`].
//!
//! # Differential coverage
//! `tests/compound_diff.rs` locks every function here against the **real
//! exported C** through `aom-sys-ref` (`shim_wedge_*`,
//! `shim_build_compound_diffwtd_mask*`, `shim_dist_wtd_comp_weight_assign`).
//! Nothing here is gated by a transcription.

use super::interintra::wedge_mask_signed;

/// `AOM_BLEND_A64_MAX_ALPHA` (`aom_dsp/blend.h`).
const AOM_BLEND_A64_MAX_ALPHA: i32 = 64;
/// `DIFF_FACTOR` = `1 << DIFF_FACTOR_LOG2` (`aom_dsp/blend.h:42`).
const DIFF_FACTOR: i32 = 16;
/// `WEDGE_WEIGHT_BITS` (`av1/common/enums.h`); `MAX_MASK_VALUE = 1 << it`.
const WEDGE_WEIGHT_BITS: u32 = 6;
const MAX_MASK_VALUE: i32 = 1 << WEDGE_WEIGHT_BITS;
/// `FILTER_BITS` (`aom_dsp/aom_filter.h`).
const FILTER_BITS: i32 = 7;
/// `MAX_FRAME_DISTANCE` = `(1 << FRAME_OFFSET_BITS) - 1`, FRAME_OFFSET_BITS = 5
/// (`av1/common/enums.h:67-68`).
const MAX_FRAME_DISTANCE: i32 = 31;

#[inline]
fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// `ROUND_POWER_OF_TWO(value, n)` for a u64 accumulator.
#[inline]
fn round_pow2_u64(v: u64, n: u32) -> u64 {
    (v + (1u64 << (n - 1))) >> n
}

/// `ROUND_POWER_OF_TWO(value, n)` for i32, with `n == 0` well defined
/// (C's macro is `((value) + (((1 << (n)) >> 1))) >> (n)`, which for n == 0 is
/// `(value + 0) >> 0`).
#[inline]
fn round_pow2_i32(v: i32, n: i32) -> i32 {
    if n <= 0 { v } else { (v + (1 << (n - 1))) >> n }
}

// ===================================================================
// wedge_utils.c — the encoder's wedge RD-search primitives
// ===================================================================

/// `av1_wedge_sse_from_residuals_c` (`av1/encoder/wedge_utils.c:52`).
///
/// SSE of the compound predictor formed by blending `p0`/`p1` with `m`, scaled
/// up by `MAX_MASK_VALUE**2` and rounded down by `2 * WEDGE_WEIGHT_BITS`:
///
/// ```text
/// sum(clamp16(64*r1[i] + m[i]*d[i])^2)  >>  12   (rounded)
/// ```
///
/// where `r1 = source - p1` and `d = p1 - p0`.
///
/// The clamp to signed 16 bits is **load-bearing, not a guard**: C does it
/// explicitly so a SIMD implementation that saturates in 16-bit lanes stays
/// bit-identical. Dropping it would change the result on large residuals.
pub fn wedge_sse_from_residuals(r1: &[i16], d: &[i16], m: &[u8], n: usize) -> u64 {
    assert!(r1.len() >= n && d.len() >= n && m.len() >= n);
    let mut csse: u64 = 0;
    for i in 0..n {
        let t = MAX_MASK_VALUE * i32::from(r1[i]) + i32::from(m[i]) * i32::from(d[i]);
        let t = clamp_i32(t, i32::from(i16::MIN), i32::from(i16::MAX));
        // |t| <= 32767 so t*t <= 2^30 — the C `int` product cannot overflow.
        csse += (t as i64 * t as i64) as u64;
    }
    round_pow2_u64(csse, 2 * WEDGE_WEIGHT_BITS)
}

/// `av1_wedge_sign_from_residuals_c` (`wedge_utils.c:101`): true when the
/// **negated** mask has the lower SSE, i.e. `sum(ds[i] * m[i]) > limit`.
///
/// C returns `int8_t` 0/1; the Rust type is `bool` for the same information.
/// C's loop is a `do { } while (--N)`, so `n == 0` is undefined there — this
/// port asserts instead of reading out of bounds.
pub fn wedge_sign_from_residuals(ds: &[i16], m: &[u8], n: usize, limit: i64) -> bool {
    assert!(
        n > 0,
        "av1_wedge_sign_from_residuals_c is a do-while: N >= 1"
    );
    assert!(ds.len() >= n && m.len() >= n);
    let mut acc: i64 = 0;
    for i in 0..n {
        acc += i64::from(i32::from(ds[i]) * i32::from(m[i]));
    }
    acc > limit
}

/// `av1_wedge_compute_delta_squares_c` (`wedge_utils.c:123`):
/// `d[i] = clamp16(a[i]^2 - b[i]^2)`.
pub fn wedge_compute_delta_squares(d: &mut [i16], a: &[i16], b: &[i16], n: usize) {
    assert!(d.len() >= n && a.len() >= n && b.len() >= n);
    for i in 0..n {
        let ai = i32::from(a[i]);
        let bi = i32::from(b[i]);
        d[i] = clamp_i32(ai * ai - bi * bi, i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}

// ===================================================================
// reconinter.c — the difference-weighted (DIFFWTD) compound masks
// ===================================================================

/// `DIFFWTD_MASK_TYPE` (`av1/common/enums.h`). `Diffwtd38Inv` is the same mask
/// complemented against `AOM_BLEND_A64_MAX_ALPHA`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DiffwtdMaskType {
    /// `DIFFWTD_38`
    Diffwtd38 = 0,
    /// `DIFFWTD_38_INV`
    Diffwtd38Inv = 1,
}

impl DiffwtdMaskType {
    #[inline]
    fn which_inverse(self) -> bool {
        self == DiffwtdMaskType::Diffwtd38Inv
    }
}

/// The `mask_base` both DIFFWTD types pass (`av1_build_compound_diffwtd_mask_c`
/// hard-codes 38 for each arm).
const DIFFWTD_MASK_BASE: i32 = 38;

/// `av1_build_compound_diffwtd_mask_c` (`reconinter.c:351`) via the static
/// `diffwtd_mask` (:336) — the 8-bit-pixel-domain mask.
///
/// `mask` is written contiguously at stride `w`.
#[allow(clippy::too_many_arguments)]
pub fn build_compound_diffwtd_mask(
    mask: &mut [u8],
    mask_type: DiffwtdMaskType,
    src0: &[u8],
    src0_stride: usize,
    src1: &[u8],
    src1_stride: usize,
    h: usize,
    w: usize,
) {
    let inv = mask_type.which_inverse();
    for i in 0..h {
        for j in 0..w {
            let diff =
                (i32::from(src0[i * src0_stride + j]) - i32::from(src1[i * src1_stride + j])).abs();
            let m = clamp_i32(
                DIFFWTD_MASK_BASE + diff / DIFF_FACTOR,
                0,
                AOM_BLEND_A64_MAX_ALPHA,
            );
            mask[i * w + j] = if inv {
                (AOM_BLEND_A64_MAX_ALPHA - m) as u8
            } else {
                m as u8
            };
        }
    }
}

/// `av1_build_compound_diffwtd_mask_d16_c` (`reconinter.c:319`) via the static
/// `diffwtd_mask_d16` (:301) — the **convolve-buffer domain** mask, built from
/// the two 16-bit `CONV_BUF_TYPE` intermediates before the final round.
///
/// The extra shift `round = 2*FILTER_BITS - round_0 - round_1 + (bd - 8)`
/// brings the intermediate difference back to the pixel domain.
#[allow(clippy::too_many_arguments)]
pub fn build_compound_diffwtd_mask_d16(
    mask: &mut [u8],
    mask_type: DiffwtdMaskType,
    src0: &[u16],
    src0_stride: usize,
    src1: &[u16],
    src1_stride: usize,
    h: usize,
    w: usize,
    round_0: i32,
    round_1: i32,
    bd: i32,
) {
    let inv = mask_type.which_inverse();
    let round = 2 * FILTER_BITS - round_0 - round_1 + (bd - 8);
    for i in 0..h {
        for j in 0..w {
            let diff =
                (i32::from(src0[i * src0_stride + j]) - i32::from(src1[i * src1_stride + j])).abs();
            let diff = round_pow2_i32(diff, round);
            let m = clamp_i32(
                DIFFWTD_MASK_BASE + diff / DIFF_FACTOR,
                0,
                AOM_BLEND_A64_MAX_ALPHA,
            );
            mask[i * w + j] = if inv {
                (AOM_BLEND_A64_MAX_ALPHA - m) as u8
            } else {
                m as u8
            };
        }
    }
}

/// `av1_build_compound_diffwtd_mask_highbd_c` (`reconinter.c:431`) via the
/// static `diffwtd_mask_highbd` (:368) — the high-bit-depth pixel-domain mask.
///
/// The C helper is written as four unrolled arms (bd==8 vs bd>8, crossed with
/// `which_inverse`), and the bd>8 arm shifts the absolute difference down by
/// `bd - 8` **before** dividing by `DIFF_FACTOR`, not after. This port keeps
/// that order; `(x >> s) / 16` and `x / (16 << s)` agree for non-negative `x`,
/// but the C order is what the SIMD versions reproduce.
///
/// C reaches the clamp through `negative_to_zero(mask_base + diff)` followed by
/// `AOMMIN(m, 64)` rather than `clamp(.., 0, 64)`. With `mask_base == 38` and
/// `diff >= 0` the lower bound can never bind, so the two are the same function
/// here — this port writes the C form so the equivalence stays visible rather
/// than assumed.
#[allow(clippy::too_many_arguments)]
pub fn build_compound_diffwtd_mask_highbd(
    mask: &mut [u8],
    mask_type: DiffwtdMaskType,
    src0: &[u16],
    src0_stride: usize,
    src1: &[u16],
    src1_stride: usize,
    h: usize,
    w: usize,
    bd: u32,
) {
    assert!(bd >= 8);
    let inv = mask_type.which_inverse();
    let bd_shift = bd - 8;
    for i in 0..h {
        for j in 0..w {
            let abs_diff =
                (i32::from(src0[i * src0_stride + j]) - i32::from(src1[i * src1_stride + j])).abs();
            let diff = if bd == 8 {
                abs_diff / DIFF_FACTOR
            } else {
                (abs_diff >> bd_shift) / DIFF_FACTOR
            };
            // negative_to_zero(mask_base + diff) then AOMMIN(.., 64). Kept as
            // C's two-step form rather than `.clamp()` so the reader can see
            // that the lower bound is C's and check that it never binds.
            #[allow(clippy::manual_clamp)]
            let m = (DIFFWTD_MASK_BASE + diff)
                .max(0)
                .min(AOM_BLEND_A64_MAX_ALPHA);
            mask[i * w + j] = if inv {
                (AOM_BLEND_A64_MAX_ALPHA - m) as u8
            } else {
                m as u8
            };
        }
    }
}

// ===================================================================
// reconinter.c — compound type mask selection
// ===================================================================

/// `COMPOUND_TYPE` (`av1/common/enums.h`), restricted to the two masked types
/// `av1_get_compound_type_mask` distinguishes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompoundType {
    /// `COMPOUND_WEDGE` — the mask comes from the wedge codebook.
    Wedge {
        /// `comp_data->wedge_index`
        index: usize,
        /// `comp_data->wedge_sign`
        sign: usize,
    },
    /// `COMPOUND_DIFFWTD` (and every other type) — the mask is the caller's
    /// `seg_mask`, built by [`build_compound_diffwtd_mask_d16`].
    SegMask,
}

/// `av1_get_compound_type_mask` (`reconinter.c:290`): the blend mask for a
/// masked compound block.
///
/// C returns a pointer — into the baked wedge table for `COMPOUND_WEDGE`, or
/// into `comp_data->seg_mask` otherwise. Rust returns `Some(mask)` for the
/// wedge case (a fresh `bw*bh` buffer at stride `bw`) and `None` for the
/// seg-mask case, where the caller already owns the buffer.
///
/// Returns `None` for the wedge case too when `bsize` has no wedge codebook,
/// which C reaches only through a failed assert.
pub fn get_compound_type_mask(comp_type: CompoundType, bsize: usize) -> Option<Vec<u8>> {
    match comp_type {
        CompoundType::Wedge { index, sign } => wedge_mask_signed(bsize, index, sign),
        CompoundType::SegMask => None,
    }
}

// ===================================================================
// reconinter.c — distance-weighted compound offsets
// ===================================================================

/// `quant_dist_weight[4][2]` (`av1/common/common_data.h:417`).
const QUANT_DIST_WEIGHT: [[i32; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, MAX_FRAME_DISTANCE]];
/// `quant_dist_lookup_table[4][2]` (`common_data.h:421`).
const QUANT_DIST_LOOKUP_TABLE: [[i32; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];

/// What [`dist_wtd_comp_weight_assign`] produces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DistWtdWeights {
    /// `*fwd_offset`
    pub fwd_offset: i32,
    /// `*bck_offset`
    pub bck_offset: i32,
    /// `*use_dist_wtd_comp_avg`
    pub use_dist_wtd_comp_avg: bool,
}

/// `av1_dist_wtd_comp_weight_assign` (`reconinter.c:669`): the two blend
/// offsets for a distance-weighted compound block.
///
/// The C signature reads its inputs out of `AV1_COMMON` + `MB_MODE_INFO`; this
/// port takes the four scalars that walk actually depends on, so the boolean
/// index below cannot be fed the wrong buffer by accident:
///
/// * `bck_order_hint` is the order hint of `mbmi->ref_frame[0]`'s buffer,
/// * `fwd_order_hint` is the order hint of `mbmi->ref_frame[1]`'s buffer.
///
/// (That pairing is easy to invert: C names the *first* ref `bck_buf`.)
/// A ref whose buffer is absent contributes order hint 0, as in C.
///
/// `d0` measures forward-to-current and `d1` current-to-backward; `order` is
/// the **boolean** `d0 <= d1`, used as the low index of both tables.
#[allow(clippy::too_many_arguments)]
pub fn dist_wtd_comp_weight_assign(
    enable_order_hint: bool,
    order_hint_bits_minus_1: i32,
    cur_order_hint: i32,
    fwd_order_hint: i32,
    bck_order_hint: i32,
    compound_idx: bool,
    is_compound: bool,
) -> DistWtdWeights {
    if !is_compound || compound_idx {
        return DistWtdWeights {
            fwd_offset: 8,
            bck_offset: 8,
            use_dist_wtd_comp_avg: false,
        };
    }

    let rel = |a: i32, b: i32| {
        crate::entropy::partition::get_relative_dist(
            enable_order_hint,
            order_hint_bits_minus_1,
            a,
            b,
        )
    };
    let d0 = clamp_i32(
        rel(fwd_order_hint, cur_order_hint).abs(),
        0,
        MAX_FRAME_DISTANCE,
    );
    let d1 = clamp_i32(
        rel(cur_order_hint, bck_order_hint).abs(),
        0,
        MAX_FRAME_DISTANCE,
    );

    let order = usize::from(d0 <= d1);

    if d0 == 0 || d1 == 0 {
        return DistWtdWeights {
            fwd_offset: QUANT_DIST_LOOKUP_TABLE[3][order],
            bck_offset: QUANT_DIST_LOOKUP_TABLE[3][1 - order],
            use_dist_wtd_comp_avg: true,
        };
    }

    let mut i = 0usize;
    while i < 3 {
        let c0 = QUANT_DIST_WEIGHT[i][order];
        let c1 = QUANT_DIST_WEIGHT[i][1 - order];
        let d0_c0 = d0 * c0;
        let d1_c1 = d1 * c1;
        if (d0 > d1 && d0_c0 < d1_c1) || (d0 <= d1 && d0_c0 > d1_c1) {
            break;
        }
        i += 1;
    }

    DistWtdWeights {
        fwd_offset: QUANT_DIST_LOOKUP_TABLE[i][order],
        bck_offset: QUANT_DIST_LOOKUP_TABLE[i][1 - order],
        use_dist_wtd_comp_avg: true,
    }
}

// ===================================================================
// aom_dsp/blend_a64_mask.c — the D16 (convolve-buffer domain) mask blend
// ===================================================================

/// `AOM_BLEND_A64_ROUND_BITS` (`aom_dsp/blend.h:23`).
const AOM_BLEND_A64_ROUND_BITS: i32 = 6;
/// `AOM_BLEND_AVG(v0, v1)` (`blend.h:40`).
#[inline]
fn blend_avg(v0: i32, v1: i32) -> i32 {
    round_pow2_i32(v0 + v1, 1)
}

/// Read the blend mask for output pixel `(i, j)` under the `(subw, subh)`
/// sub-sampling C's four unrolled branches encode.
///
/// `subw`/`subh` are **mask** sub-sampling, i.e. how many mask samples cover one
/// output pixel in each direction — so `subw == 1` averages a horizontal PAIR.
/// The 2x2 case averages four with a single `round(sum, 2)`, which is not the
/// same as two nested `AOM_BLEND_AVG` roundings.
#[inline]
fn d16_mask_at(mask: &[u8], mask_stride: usize, i: usize, j: usize, subw: bool, subh: bool) -> i32 {
    match (subw, subh) {
        (false, false) => i32::from(mask[i * mask_stride + j]),
        (true, true) => round_pow2_i32(
            i32::from(mask[(2 * i) * mask_stride + 2 * j])
                + i32::from(mask[(2 * i + 1) * mask_stride + 2 * j])
                + i32::from(mask[(2 * i) * mask_stride + 2 * j + 1])
                + i32::from(mask[(2 * i + 1) * mask_stride + 2 * j + 1]),
            2,
        ),
        (true, false) => blend_avg(
            i32::from(mask[i * mask_stride + 2 * j]),
            i32::from(mask[i * mask_stride + 2 * j + 1]),
        ),
        (false, true) => blend_avg(
            i32::from(mask[(2 * i) * mask_stride + j]),
            i32::from(mask[(2 * i + 1) * mask_stride + j]),
        ),
    }
}

/// `aom_lowbd_blend_a64_d16_mask_c` (`aom_dsp/blend_a64_mask.c:36`): blend the
/// two 16-bit convolve intermediates with the A64 mask and bring the result
/// back to the pixel domain.
///
/// This is the final step of `av1_make_masked_inter_predictor` — the blend that
/// turns two `dist_wtd_convolve` outputs into a masked compound predictor.
/// Unlike [`super::interintra::blend_a64_mask`], which blends pixels, this one
/// blends the pre-rounding intermediates and therefore has to subtract the
/// convolve's `round_offset` before the final rounding.
///
/// `mask` weights `src0`; `64 - mask` weights `src1`.
#[allow(clippy::too_many_arguments)]
pub fn lowbd_blend_a64_d16_mask(
    dst: &mut [u8],
    dst_stride: usize,
    src0: &[u16],
    src0_stride: usize,
    src1: &[u16],
    src1_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    w: usize,
    h: usize,
    subw: bool,
    subh: bool,
    round_0: i32,
    round_1: i32,
) {
    const BD: i32 = 8;
    let offset_bits = BD + 2 * FILTER_BITS - round_0;
    let round_offset = (1i32 << (offset_bits - round_1)) + (1i32 << (offset_bits - round_1 - 1));
    let round_bits = 2 * FILTER_BITS - round_0 - round_1;

    for i in 0..h {
        for j in 0..w {
            let m = d16_mask_at(mask, mask_stride, i, j, subw, subh);
            let s0 = i32::from(src0[i * src0_stride + j]);
            let s1 = i32::from(src1[i * src1_stride + j]);
            let res = (m * s0 + (AOM_BLEND_A64_MAX_ALPHA - m) * s1) >> AOM_BLEND_A64_ROUND_BITS;
            let res = res - round_offset;
            dst[i * dst_stride + j] = round_pow2_i32(res, round_bits).clamp(0, 255) as u8;
        }
    }
}

/// `aom_highbd_blend_a64_d16_mask_c` (`blend_a64_mask.c:124`).
///
/// Same arithmetic as the lowbd twin with a `bd`-dependent saturation. C spells
/// the clamp as `negative_to_zero(..)` then `AOMMIN(v, saturation_value)`
/// rather than `clip_pixel_highbd`; the two agree, and the port writes the C
/// form so the equivalence stays visible.
#[allow(clippy::too_many_arguments)]
pub fn highbd_blend_a64_d16_mask(
    dst: &mut [u16],
    dst_stride: usize,
    src0: &[u16],
    src0_stride: usize,
    src1: &[u16],
    src1_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    w: usize,
    h: usize,
    subw: bool,
    subh: bool,
    round_0: i32,
    round_1: i32,
    bd: u32,
) {
    let offset_bits = bd as i32 + 2 * FILTER_BITS - round_0;
    let round_offset = (1i32 << (offset_bits - round_1)) + (1i32 << (offset_bits - round_1 - 1));
    let round_bits = 2 * FILTER_BITS - round_0 - round_1;
    // C's `switch (bd)` defaults to 255 for anything that is not 10 or 12,
    // which is not the same as `(1 << bd) - 1` for an out-of-spec bd.
    let saturation_value: i32 = match bd {
        10 => 1023,
        12 => 4095,
        _ => 255,
    };

    for i in 0..h {
        for j in 0..w {
            let m = d16_mask_at(mask, mask_stride, i, j, subw, subh);
            let s0 = i32::from(src0[i * src0_stride + j]);
            let s1 = i32::from(src1[i * src1_stride + j]);
            let res = (m * s0 + (AOM_BLEND_A64_MAX_ALPHA - m) * s1) >> AOM_BLEND_A64_ROUND_BITS;
            let res = res - round_offset;
            let v = round_pow2_i32(res, round_bits).max(0);
            dst[i * dst_stride + j] = v.min(saturation_value) as u16;
        }
    }
}
