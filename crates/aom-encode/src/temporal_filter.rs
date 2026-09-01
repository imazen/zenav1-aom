//! Temporal (alt-ref) filtering kernels — port of `av1/encoder/temporal_filter.c`.
//!
//! Reached when `--lag-in-frames >= ALT_MIN_LAG` and alt-ref is enabled: the
//! encoder motion-compensates `num_frames` neighbours onto the frame being
//! filtered and non-local-means blends them, producing the ARF source. Nothing
//! here touches the bitstream directly; it changes the *pixels* the ARF is
//! encoded from, so it has to be byte-exact for an ARF encode to be.
//!
//! | Rust | C |
//! |---|---|
//! | [`estimate_noise_from_single_plane`] | `av1_estimate_noise_from_single_plane_c` (:1426) |
//! | [`highbd_estimate_noise_from_single_plane`] | `av1_highbd_estimate_noise_from_single_plane_c` (:1465) |
//! | [`estimate_noise_level`] | `av1_estimate_noise_level` (:1505) |
//! | [`apply_temporal_filter`] | `av1_apply_temporal_filter_c` (:795) and `av1_highbd_apply_temporal_filter_c` (:964), which forwards to it |
//! | [`compute_square_diff`] | `compute_square_diff` (:698, static) |
//! | [`compute_luma_sq_error_sum`] | `compute_luma_sq_error_sum` (:742, static) |
//! | [`approx_exp`] / [`iroundpf`] | `approx_exp` / `iroundpf` (`aom_dsp/mathutils.h`) |
//!
//! # Sentinel → `Option`
//! C returns `-1.0` from both noise estimators to mean "too few smooth pixels
//! to trust the estimate" and every caller tests `< 0`. The port returns
//! [`Option<f64>`] instead; [`estimate_noise_level`] maps `None` back to
//! `-1.0` because the value it fills is consumed as a plain `double` by
//! `av1_apply_temporal_filter` (which feeds it to `log(2 * n + 5)` — at `-1.0`
//! that is `log(3)`, a real code path, not an error).
//!
//! # Integer widths that are part of the contract
//! * `square_diff` is `uint32_t` in C and `u32` here: a 12-bit difference
//!   squares to at most `4095^2 = 16_769_025`, so the window sum over 25 taps
//!   plus the luma sum still needs the `u64` accumulator C uses.
//! * `accum` (`uint32_t`) and `count` (`uint16_t`) are accumulated across every
//!   filtered frame by the caller, so both are added with `wrapping_add` —
//!   unsigned overflow is defined in C and this reproduces it rather than
//!   panicking in a debug build. The encoder cannot reach either bound
//!   (`weight <= TF_WEIGHT_SCALE = 1000`, at most `MAX_LAG_BUFFERS = 48`
//!   frames: `count <= 48_000 < u16::MAX`).
//!
//! # Floating point
//! `pow(x, 2)` is written `x * x`: for IEEE-754 doubles the correctly-rounded
//! `pow(x, 2.0)` *is* `round(x * x)`, which is what the multiply computes. The
//! `exp` and `ln` calls go to the same libm C's do. `approx_exp` is a bit-level
//! reinterpretation and is transcribed exactly, including its truncating
//! `float -> int32_t` cast.
//!
//! # Differential coverage
//! `tests/temporal_filter_diff.rs` — tier 1 against the real exported
//! `av1_apply_temporal_filter_c`, `av1_highbd_apply_temporal_filter_c`,
//! `av1_estimate_noise_from_single_plane_c`,
//! `av1_highbd_estimate_noise_from_single_plane_c` and `av1_estimate_noise_level`.

/// `SQRT_PI_BY_2` (`temporal_filter.h:42`) — `sqrt(pi / 2)`, as C spells it.
pub const SQRT_PI_BY_2: f64 = 1.253_314_137_32;
/// `NOISE_ESTIMATION_EDGE_THRESHOLD` (`temporal_filter.h:87`).
pub const NOISE_ESTIMATION_EDGE_THRESHOLD: i32 = 50;
/// `TF_WINDOW_LENGTH` (`temporal_filter.h:36`) — the non-local-means window.
pub const TF_WINDOW_LENGTH: i32 = 5;
/// `NUM_16X16` (`temporal_filter.h:39`) — 16x16 blocks in one 64x64 TF block.
pub const NUM_16X16: usize = 16;
/// `TF_WEIGHT_SCALE` (`temporal_filter.h:49`).
pub const TF_WEIGHT_SCALE: i32 = 1000;
/// `TF_WINDOW_BLOCK_BALANCE_WEIGHT` (`temporal_filter.h:54`).
pub const TF_WINDOW_BLOCK_BALANCE_WEIGHT: i32 = 5;
/// `TF_Q_DECAY_THRESHOLD` (`temporal_filter.h:60`).
pub const TF_Q_DECAY_THRESHOLD: i32 = 20;
/// `TF_SEARCH_ERROR_NORM_WEIGHT` (`temporal_filter.h:65`).
pub const TF_SEARCH_ERROR_NORM_WEIGHT: i32 = 20;
/// `TF_STRENGTH_THRESHOLD` (`temporal_filter.h:70`).
pub const TF_STRENGTH_THRESHOLD: i32 = 4;
/// `TF_SEARCH_DISTANCE_THRESHOLD` (`temporal_filter.h:80`).
pub const TF_SEARCH_DISTANCE_THRESHOLD: f64 = 0.1;
/// `TF_QINDEX_CUTOFF` (`temporal_filter.h:85`).
pub const TF_QINDEX_CUTOFF: i32 = 128;

/// `ROUND_POWER_OF_TWO(value, n)` (`aom_ports/mem.h`) on a non-negative `i32`.
#[inline]
fn round_power_of_two(value: i32, n: u32) -> i32 {
    (value + ((1 << n) >> 1)) >> n
}

/// `iroundpf` (`aom_dsp/mathutils.h:90`) — round a NON-NEGATIVE `float` to the
/// nearest `int` by truncating `x + 0.5f`. C asserts `x >= 0.0`; so does this.
#[inline]
#[must_use]
pub fn iroundpf(x: f32) -> i32 {
    debug_assert!(x >= 0.0, "iroundpf is only defined for x >= 0 (got {x})");
    (x + 0.5f32) as i32
}

/// `approx_exp` (`aom_dsp/mathutils.h:129`) — `exp(y)` by writing the IEEE-754
/// exponent field directly.
///
/// Transcribed bit-for-bit, including the truncating `(int32_t)(y * A)` cast.
/// Rust's `as i32` saturates where C's cast is undefined; the caller's range is
/// `y in [-7, 0]`, four orders of magnitude inside `i32`, so the two agree.
#[inline]
#[must_use]
pub fn approx_exp(y: f32) -> f32 {
    /// `(1 << 23) / ln(2)`.
    const A: f32 = (1u32 << 23) as f32 / core::f32::consts::LN_2;
    /// IEEE-754 exponent bias.
    const B: i32 = 127;
    /// Magic number controlling the approximation's accuracy.
    const C: i32 = 60801;
    f32::from_bits((((y * A) as i32) + ((B << 23) - C)) as u32)
}

/// A pixel of either bit depth, as the temporal filter reads one.
///
/// C carries both through a single `uint8_t *` and re-reads it as `uint16_t *`
/// under `is_high_bitdepth`; the port makes the depth a type parameter so the
/// branch disappears without changing any arithmetic.
pub trait TfPixel: Copy {
    /// `is_frame_high_bitdepth(frame)` (temporal_filter.c:520) as a type-level
    /// fact: it decides only how many bits the noise estimator's
    /// `ROUND_POWER_OF_TWO` normalises away.
    const HIGH_BITDEPTH: bool;
    /// The value C's `ref_value` / `tgt_value` / `pred_value` locals hold.
    fn value(self) -> u32;
}

impl TfPixel for u8 {
    const HIGH_BITDEPTH: bool = false;
    #[inline]
    fn value(self) -> u32 {
        u32::from(self)
    }
}

impl TfPixel for u16 {
    const HIGH_BITDEPTH: bool = true;
    #[inline]
    fn value(self) -> u32 {
        u32::from(self)
    }
}

/// `compute_square_diff` (temporal_filter.c:698).
///
/// Writes `height * width` squared differences, contiguously, from two strided
/// windows. C passes byte offsets into both buffers; the port slices instead,
/// so `ref` and `tgt` start at what C calls `ref + ref_offset` / `tgt + tgt_offset`.
pub fn compute_square_diff<P: TfPixel>(
    reference: &[P],
    ref_stride: usize,
    target: &[P],
    tgt_stride: usize,
    height: usize,
    width: usize,
    square_diff: &mut [u32],
) {
    for i in 0..height {
        let ref_row = &reference[i * ref_stride..][..width];
        let tgt_row = &target[i * tgt_stride..][..width];
        let out_row = &mut square_diff[i * width..][..width];
        for ((out, &r), &t) in out_row.iter_mut().zip(ref_row).zip(tgt_row) {
            let diff = r.value().abs_diff(t.value());
            *out = diff * diff;
        }
    }
}

/// `compute_luma_sq_error_sum` (temporal_filter.c:742).
///
/// Folds the luma plane's squared differences down onto the chroma grid:
/// each chroma sample accumulates the `1 << (ss_x_shift + ss_y_shift)` luma
/// cells it covers. C *adds into* `luma_sse_sum`, and so does this — the
/// caller zeroes it once and both chroma planes reuse the result.
pub fn compute_luma_sq_error_sum(
    square_diff: &[u32],
    luma_sse_sum: &mut [u32],
    block_height: usize,
    block_width: usize,
    ss_x_shift: u32,
    ss_y_shift: u32,
) {
    // Width of the Y-plane grid `square_diff` is laid out on.
    let ww = block_width << ss_x_shift;
    for i in 0..block_height {
        for j in 0..block_width {
            let mut sum = 0u32;
            for ii in 0..(1usize << ss_y_shift) {
                for jj in 0..(1usize << ss_x_shift) {
                    let yy = (i << ss_y_shift) + ii;
                    let xx = (j << ss_x_shift) + jj;
                    sum = sum.wrapping_add(square_diff[yy * ww + xx]);
                }
            }
            let cell = &mut luma_sse_sum[i * block_width + j];
            *cell = cell.wrapping_add(sum);
        }
    }
}

/// Shared body of `av1_estimate_noise_from_single_plane_c` (:1426) and its
/// highbd twin (:1465). The only difference between the two is the shift
/// applied by `ROUND_POWER_OF_TWO`, which is `bit_depth - 8`.
fn estimate_noise_impl<P: TfPixel>(
    src: &[P],
    height: usize,
    width: usize,
    stride: usize,
    shift: u32,
    edge_thresh: i32,
) -> Option<f64> {
    if height < 3 || width < 3 {
        // C's loops are `1..height-1` / `1..width-1`, so a plane thinner than
        // 3 in either axis visits nothing and falls into the `count < 16` arm.
        return None;
    }
    let mut accum: i64 = 0;
    let mut count: i64 = 0;

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            // C's `mat[3][3]`, read as i32 exactly as C's `int mat[3][3]` is.
            let center = i * stride + j;
            let mut mat = [[0i32; 3]; 3];
            for (ii, row) in mat.iter_mut().enumerate() {
                for (jj, cell) in row.iter_mut().enumerate() {
                    let idx = center + (ii * stride + jj) - (stride + 1);
                    *cell = src[idx].value() as i32;
                }
            }
            // Sobel gradients.
            let gx =
                (mat[0][0] - mat[0][2]) + (mat[2][0] - mat[2][2]) + 2 * (mat[1][0] - mat[1][2]);
            let gy =
                (mat[0][0] - mat[2][0]) + (mat[0][2] - mat[2][2]) + 2 * (mat[0][1] - mat[2][1]);
            let ga = round_power_of_two(gx.abs() + gy.abs(), shift);
            if ga < edge_thresh {
                // Only count smooth pixels: accumulate the Laplacian.
                let v = 4 * mat[1][1] - 2 * (mat[0][1] + mat[2][1] + mat[1][0] + mat[1][2])
                    + (mat[0][0] + mat[0][2] + mat[2][0] + mat[2][2]);
                accum += i64::from(round_power_of_two(v.abs(), shift));
                count += 1;
            }
        }
    }

    // C returns -1.0 (unreliable estimation) when there are too few smooth pixels.
    if count < 16 {
        None
    } else {
        Some(accum as f64 / (6 * count) as f64 * SQRT_PI_BY_2)
    }
}

/// `av1_estimate_noise_from_single_plane_c` (temporal_filter.c:1426).
///
/// `None` is C's `-1.0`: fewer than 16 smooth pixels, estimate not usable.
#[must_use]
pub fn estimate_noise_from_single_plane(
    src: &[u8],
    height: usize,
    width: usize,
    stride: usize,
    edge_thresh: i32,
) -> Option<f64> {
    estimate_noise_impl(src, height, width, stride, 0, edge_thresh)
}

/// `av1_highbd_estimate_noise_from_single_plane_c` (temporal_filter.c:1465).
///
/// Identical to the lowbd arm except that both `ROUND_POWER_OF_TWO`s shift by
/// `bit_depth - 8`, normalising the 10/12-bit gradients back onto the 8-bit
/// `edge_thresh` scale.
#[must_use]
pub fn highbd_estimate_noise_from_single_plane(
    src: &[u16],
    height: usize,
    width: usize,
    stride: usize,
    bit_depth: u32,
    edge_thresh: i32,
) -> Option<f64> {
    estimate_noise_impl(src, height, width, stride, bit_depth - 8, edge_thresh)
}

/// One plane of the frame the noise estimator / temporal filter reads.
#[derive(Clone, Copy, Debug)]
pub struct TfPlane<'a, P> {
    /// Pixels, starting at the plane origin.
    pub data: &'a [P],
    /// `frame->strides[plane == AOM_PLANE_Y ? 0 : 1]`.
    pub stride: usize,
    /// `frame->crop_widths[plane != AOM_PLANE_Y]`.
    pub crop_width: usize,
    /// `frame->crop_heights[plane != AOM_PLANE_Y]`.
    pub crop_height: usize,
}

/// `av1_estimate_noise_level` (temporal_filter.c:1505) for `planes` in order.
///
/// C fills `noise_level[plane_from..=plane_to]`; the port returns one entry per
/// supplied plane and leaves the placement to the caller. `None` is C's `-1.0`.
#[must_use]
pub fn estimate_noise_level<P: TfPixel>(
    planes: &[TfPlane<'_, P>],
    bit_depth: u32,
    edge_thresh: i32,
) -> Vec<Option<f64>> {
    let shift = if P::HIGH_BITDEPTH { bit_depth - 8 } else { 0 };
    planes
        .iter()
        .map(|p| {
            estimate_noise_impl(
                p.data,
                p.crop_height,
                p.crop_width,
                p.stride,
                shift,
                edge_thresh,
            )
        })
        .collect()
}

/// Everything `av1_apply_temporal_filter_c` reads out of `cpi` / `mbd` /
/// the block geometry, gathered so the kernel's own signature stays readable.
#[derive(Clone, Debug)]
pub struct TfFilterParams {
    /// `block_size_wide[block_size]` — 64 for `TF_BLOCK_SIZE`.
    pub block_width: usize,
    /// `block_size_high[block_size]`.
    pub block_height: usize,
    /// `mb_row` / `mb_col`, in units of the block size.
    pub mb_row: usize,
    /// See [`Self::mb_row`].
    pub mb_col: usize,
    /// `mbd->plane[p].subsampling_x` for each of the `num_planes` planes.
    pub subsampling_x: [u32; 3],
    /// `mbd->plane[p].subsampling_y`.
    pub subsampling_y: [u32; 3],
    /// `mbd->bd`. Drives only the `sum_square_diff >>= (bd - 8) * 2` rescale.
    pub bd: u32,
    /// `q_factor` — `q`, not `qindex`.
    pub q_factor: i32,
    /// `filter_strength`, in `[0, 6]`.
    pub filter_strength: i32,
    /// `tf_wgt_calc_lvl`: 0 selects libm `exp`, non-zero selects [`approx_exp`].
    pub wgt_calc_lvl: i32,
}

/// `av1_apply_temporal_filter_c` (temporal_filter.c:795), and therefore also
/// `av1_highbd_apply_temporal_filter_c` (:964), which only forwards to it.
///
/// `frame_planes[p]` is `frame_to_filter`'s plane `p` sliced from its ORIGIN
/// (C indexes it by `frame_offset`, which this derives the same way);
/// `pred` is the concatenation of the per-plane predictors C lays out
/// back to back, and `accum` / `count` share that layout. All three are
/// `mb_pels`-shaped per plane, so their combined length is
/// `sum over planes of (block_height >> ss_y) * (block_width >> ss_x)`.
///
/// `accum` and `count` are ACCUMULATED into, not overwritten: the caller runs
/// this once per reference frame.
///
/// # Panics
/// If `noise_levels`, `subblock_mvs` or `subblock_mses` is shorter than the
/// plane count / [`NUM_16X16`], or a plane slice is too short for the block
/// the parameters select.
pub fn apply_temporal_filter<P: TfPixel>(
    frame_planes: &[TfPlane<'_, P>],
    params: &TfFilterParams,
    noise_levels: &[f64],
    subblock_mvs: &[(i16, i16); NUM_16X16],
    subblock_mses: &[i32; NUM_16X16],
    pred: &[P],
    accum: &mut [u32],
    count: &mut [u16],
) {
    let num_planes = frame_planes.len();
    assert!(num_planes <= 3, "AV1 has at most 3 planes");
    assert!(noise_levels.len() >= num_planes);

    let mb_height = params.block_height;
    let mb_width = params.block_width;
    let mb_pels = mb_height * mb_width;

    // Frame information (C reads the LUMA crop size for both).
    let frame_height = frame_planes[0].crop_height;
    let frame_width = frame_planes[0].crop_width;
    let min_frame_size = frame_height.min(frame_width) as f64;

    // Variables to simplify combined error calculation.
    let inv_factor =
        1.0 / f64::from((TF_WINDOW_BLOCK_BALANCE_WEIGHT + 1) * TF_SEARCH_ERROR_NORM_WEIGHT);
    let weight_factor = f64::from(TF_WINDOW_BLOCK_BALANCE_WEIGHT) * inv_factor;

    // Adjust filtering based on q: larger q -> stronger filtering -> larger weight.
    let mut q_decay = {
        let x = f64::from(params.q_factor) / f64::from(TF_Q_DECAY_THRESHOLD);
        x * x
    };
    q_decay = q_decay.clamp(1e-5, 1.0);
    if params.q_factor >= TF_QINDEX_CUTOFF {
        // Max q_factor is 255, so the upper bound of q_decay is 8 — no clip needed.
        let x = f64::from(params.q_factor) / 64.0;
        q_decay = 0.5 * (x * x);
    }
    // Smaller strength -> smaller filtering weight.
    let s_decay = {
        let x = f64::from(params.filter_strength) / f64::from(TF_STRENGTH_THRESHOLD);
        (x * x).clamp(1e-5, 1.0)
    };

    // Larger noise -> larger filtering weight.
    let mut decay_factor = [0.0f64; 3];
    for (d, &noise) in decay_factor.iter_mut().zip(noise_levels).take(num_planes) {
        let n_decay = 0.5 + (2.0 * noise + 5.0).ln();
        *d = 1.0 / (n_decay * q_decay * s_decay);
    }

    // Each 16x16 block's d_factor: larger MV -> smaller filtering weight.
    let mut d_factor = [0.0f64; NUM_16X16];
    for (d, &(row, col)) in d_factor.iter_mut().zip(subblock_mvs) {
        let (r, c) = (f64::from(row), f64::from(col));
        let distance = (r * r + c * c).sqrt();
        let distance_threshold = (min_frame_size * TF_SEARCH_DISTANCE_THRESHOLD).max(1.0);
        *d = (distance / distance_threshold).max(1.0);
    }

    // Pixel-wise squared differences and the accumulated luma squared error.
    // Both are `mb_pels`-sized regardless of subsampling, exactly as C's
    // `aom_memalign(.., mb_pels * sizeof(uint32_t))` pair is.
    let mut square_diff = vec![0u32; mb_pels];
    let mut luma_sse_sum = vec![0u32; mb_pels];

    // Window size for pixel-wise filtering.
    const _: () = assert!(TF_WINDOW_LENGTH % 2 == 1);
    let half_window = TF_WINDOW_LENGTH >> 1;

    let mut plane_offset = 0usize;
    for plane in 0..num_planes {
        let subsampling_y = params.subsampling_y[plane];
        let subsampling_x = params.subsampling_x[plane];
        let h = mb_height >> subsampling_y; // Plane height.
        let w = mb_width >> subsampling_x; // Plane width.
        let frame = &frame_planes[plane];
        let frame_stride = frame.stride;
        let frame_offset = params.mb_row * h * frame_stride + params.mb_col * w;
        let ss_y_shift = subsampling_y - params.subsampling_y[0];
        let ss_x_shift = subsampling_x - params.subsampling_x[0];
        let num_ref_pixels = TF_WINDOW_LENGTH * TF_WINDOW_LENGTH
            + if plane > 0 {
                1 << (ss_x_shift + ss_y_shift)
            } else {
                0
            };
        let inv_num_ref_pixels = 1.0 / f64::from(num_ref_pixels);

        // Filter U and V using the Y plane's error, because motion search only
        // ran on Y. The luma sse sum is computed once and reused by both.
        if plane == 1 {
            compute_luma_sq_error_sum(
                &square_diff,
                &mut luma_sse_sum,
                h,
                w,
                ss_x_shift,
                ss_y_shift,
            );
        }
        compute_square_diff(
            &frame.data[frame_offset..],
            frame_stride,
            &pred[plane_offset..],
            w,
            h,
            w,
            &mut square_diff,
        );

        // Perform filtering.
        for i in 0..h {
            for j in 0..w {
                // Non-local mean approach: sum the window's squared differences.
                let mut sum_square_diff: u64 = 0;
                for wi in -half_window..=half_window {
                    for wj in -half_window..=half_window {
                        let y = (i as i32 + wi).clamp(0, h as i32 - 1) as usize;
                        let x = (j as i32 + wj).clamp(0, w as i32 - 1) as usize;
                        sum_square_diff += u64::from(square_diff[y * w + x]);
                    }
                }
                sum_square_diff += u64::from(luma_sse_sum[i * w + j]);

                // Scale down the difference for high bit depth input.
                if params.bd > 8 {
                    sum_square_diff >>= (params.bd - 8) * 2;
                }

                // Combine window error and block error, and normalize it.
                let window_error = sum_square_diff as f64 * inv_num_ref_pixels;

                // 16x16 block index within the 64x64 TF block.
                let y32 = i / (h / 2);
                let x32 = j / (w / 2);
                let y16 = (i % (h / 2)) / (h / 4);
                let x16 = (j % (w / 2)) / (w / 4);
                let subblock_idx = (y32 * 2 + x32) * 4 + (y16 * 2 + x16);
                let block_error = f64::from(subblock_mses[subblock_idx]);
                let combined_error = weight_factor * window_error + block_error * inv_factor;

                // Compute filter weight.
                let scaled_error =
                    (combined_error * d_factor[subblock_idx] * decay_factor[plane]).min(7.0);
                let weight = if params.wgt_calc_lvl == 0 {
                    ((-scaled_error).exp() * f64::from(TF_WEIGHT_SCALE)) as i32
                } else {
                    let fweight = approx_exp(-scaled_error as f32) * TF_WEIGHT_SCALE as f32;
                    iroundpf(fweight)
                };

                let idx = plane_offset + i * w + j; // Index with plane shift.
                let pred_value = pred[idx].value() as i32;
                accum[idx] = accum[idx].wrapping_add((weight * pred_value) as u32);
                count[idx] = count[idx].wrapping_add(weight as u16);
            }
        }
        plane_offset += h * w;
    }
}
