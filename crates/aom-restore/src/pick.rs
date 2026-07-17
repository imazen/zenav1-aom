//! Loop-restoration ENCODER search — the numeric core of
//! `av1/encoder/pickrst.c` (v3.14.1): Wiener autocorrelation stats, the
//! iterative separable-symmetric Wiener solve, the filter-score gate and the
//! integer tap finalization; the SGR projection least-squares + error and
//! the `ep` search live here too. The per-unit RD walk and the frame-level
//! decision (`av1_pick_filter_restoration`) build on these in this module.
//!
//! All pixel buffers are `u16` planes (the port-wide convention); the lowbd
//! (bd 8) arithmetic matches C's `uint8_t` paths exactly because every value
//! fits in the u8 range and the accumulator widths below are C's.

use crate::sgr::SGR_PARAMS;
use aom_entropy::lr::{WIENER_HALFWIN, WIENER_WIN};

/// `WIENER_WIN2` / `WIENER_HALFWIN1` (restoration.h).
pub const WIENER_WIN2: usize = WIENER_WIN * WIENER_WIN;
const WIENER_HALFWIN1: usize = WIENER_HALFWIN + 1;
/// `WIENER_WIN_REDUCED` (restoration.h): the 5-tap luma window under
/// `lpf_sf.reduce_wiener_window_size`.
pub const WIENER_WIN_REDUCED: usize = WIENER_WIN - 2;
/// `WIENER_STATS_DOWNSAMPLE_FACTOR` (restoration.h).
pub const WIENER_STATS_DOWNSAMPLE_FACTOR: i32 = 4;
/// `WIENER_FILT_STEP` = `1 << WIENER_FILT_PREC_BITS` (restoration.h).
pub const WIENER_FILT_STEP: i64 = 1 << 7;
/// `WIENER_FILT_BITS` = `(4 + 5 + 6) * 2` (restoration.h).
const WIENER_FILT_BITS: i64 = 30;
/// `WIENER_TAP_SCALE_FACTOR` (pickrst.c): working precision of the solve.
const WIENER_TAP_SCALE_FACTOR: i64 = 1 << 16;
/// `NUM_WIENER_ITERS` (pickrst.c).
const NUM_WIENER_ITERS: i32 = 5;

/// `find_average` (pickrst.h): the u8-truncating mean of the lowbd window.
/// `u16` values in u8 range; identical arithmetic.
fn find_average(dgd: &[u16], h_start: i32, h_end: i32, v_start: i32, v_end: i32, stride: i32) -> u16 {
    let mut sum: u64 = 0;
    for i in v_start..v_end {
        for j in h_start..h_end {
            sum += dgd[(i * stride + j) as usize] as u64;
        }
    }
    (sum / (((v_end - v_start) * (h_end - h_start)) as u64)) as u16
}

/// `acc_stat_one_line` (pickrst.c): one source row's contribution to the
/// int32 row accumulators (`count` = the dgd row this line is centred on).
#[allow(clippy::too_many_arguments)]
fn acc_stat_one_line(
    dgd: &[u16],
    src_row: &[u16],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    count: i32,
) {
    let mut y = [0i16; WIENER_WIN2];
    for j in h_start..h_end {
        let x = src_row[j as usize] as i16 - avg as i16;
        let mut idx = 0usize;
        for k in -wiener_halfwin..=wiener_halfwin {
            for l in -wiener_halfwin..=wiener_halfwin {
                y[idx] = dgd[((count + l) * dgd_stride + (j + k)) as usize] as i16 - avg as i16;
                idx += 1;
            }
        }
        debug_assert_eq!(idx, wiener_win2);
        for k in 0..wiener_win2 {
            m_row[k] += y[k] as i32 * x as i32;
            for l in k..wiener_win2 {
                // H is symmetric; fill the upper triangle here (copied down
                // outside the pixel loops).
                h_row[k * wiener_win2 + l] += y[k] as i32 * y[l] as i32;
            }
        }
    }
}

/// `av1_compute_stats_c` (pickrst.c): the lowbd Wiener autocorrelation
/// vector `M[win2]` and matrix `H[win2 * win2]` of the window around each
/// source pixel in the `[h_start, h_end) x [v_start, v_end)` rect, about the
/// dgd mean, optionally with 4x vertical downsampling
/// (`lpf_sf.use_downsampled_wiener_stats`).
#[allow(clippy::too_many_arguments)]
pub fn compute_stats(
    wiener_win: usize,
    dgd: &[u16],
    src: &[u16],
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    dgd_stride: i32,
    src_stride: i32,
    m: &mut [i64],
    h: &mut [i64],
    use_downsampled_wiener_stats: bool,
) {
    let wiener_win2 = wiener_win * wiener_win;
    let wiener_halfwin = (wiener_win >> 1) as i32;
    let avg = find_average(dgd, h_start, h_end, v_start, v_end, dgd_stride);
    let mut m_row = [0i32; WIENER_WIN2];
    let mut h_row = [0i32; WIENER_WIN2 * WIENER_WIN2];
    let mut downsample_factor = if use_downsampled_wiener_stats {
        WIENER_STATS_DOWNSAMPLE_FACTOR
    } else {
        1
    };

    m[..wiener_win2].fill(0);
    h[..wiener_win2 * wiener_win2].fill(0);

    let mut i = v_start;
    while i < v_end {
        if use_downsampled_wiener_stats && (v_end - i < WIENER_STATS_DOWNSAMPLE_FACTOR) {
            downsample_factor = v_end - i;
        }
        m_row[..wiener_win2].fill(0);
        h_row[..wiener_win2 * wiener_win2].fill(0);
        acc_stat_one_line(
            dgd,
            &src[(i * src_stride) as usize..],
            dgd_stride,
            h_start,
            h_end,
            avg,
            wiener_halfwin,
            wiener_win2,
            &mut m_row,
            &mut h_row,
            i,
        );
        for k in 0..wiener_win2 {
            // Scale by the downsampling factor (1 when not downsampling).
            m[k] += m_row[k] as i64 * downsample_factor as i64;
            for l in k..wiener_win2 {
                h[k * wiener_win2 + l] +=
                    h_row[k * wiener_win2 + l] as i64 * downsample_factor as i64;
            }
        }
        i += downsample_factor;
    }

    for k in 0..wiener_win2 {
        for l in k + 1..wiener_win2 {
            h[l * wiener_win2 + k] = h[k * wiener_win2 + l];
        }
    }
}

/// `find_average_highbd` (pickrst.h).
fn find_average_highbd(
    dgd: &[u16],
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    stride: i32,
) -> u16 {
    // Identical to the lowbd form on u16 planes.
    find_average(dgd, h_start, h_end, v_start, v_end, stride)
}

/// `av1_compute_stats_highbd_c` (pickrst.c): i64 accumulation with the
/// `bit_depth_divider` normalization (1 / 4 / 16 for bd 8 / 10 / 12).
#[allow(clippy::too_many_arguments)]
pub fn compute_stats_highbd(
    wiener_win: usize,
    dgd: &[u16],
    src: &[u16],
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    dgd_stride: i32,
    src_stride: i32,
    m: &mut [i64],
    h: &mut [i64],
    bit_depth: i32,
) {
    let wiener_win2 = wiener_win * wiener_win;
    let wiener_halfwin = (wiener_win >> 1) as i32;
    let avg = find_average_highbd(dgd, h_start, h_end, v_start, v_end, dgd_stride);
    let bit_depth_divider: i64 = match bit_depth {
        12 => 16,
        10 => 4,
        _ => 1,
    };

    m[..wiener_win2].fill(0);
    h[..wiener_win2 * wiener_win2].fill(0);
    let mut y = [0i32; WIENER_WIN2];
    for i in v_start..v_end {
        for j in h_start..h_end {
            let x = src[(i * src_stride + j) as usize] as i32 - avg as i32;
            let mut idx = 0usize;
            for k in -wiener_halfwin..=wiener_halfwin {
                for l in -wiener_halfwin..=wiener_halfwin {
                    y[idx] =
                        dgd[((i + l) * dgd_stride + (j + k)) as usize] as i32 - avg as i32;
                    idx += 1;
                }
            }
            debug_assert_eq!(idx, wiener_win2);
            for k in 0..wiener_win2 {
                m[k] += y[k] as i64 * x as i64;
                for l in k..wiener_win2 {
                    h[k * wiener_win2 + l] += y[k] as i64 * y[l] as i64;
                }
            }
        }
    }
    for k in 0..wiener_win2 {
        m[k] /= bit_depth_divider;
        h[k * wiener_win2 + k] /= bit_depth_divider;
        for l in k + 1..wiener_win2 {
            h[k * wiener_win2 + l] /= bit_depth_divider;
            h[l * wiener_win2 + k] = h[k * wiener_win2 + l];
        }
    }
}

/// `wrap_index` (pickrst.c).
#[inline]
fn wrap_index(i: usize, wiener_win: usize) -> usize {
    let wiener_halfwin1 = (wiener_win >> 1) + 1;
    if i >= wiener_halfwin1 {
        wiener_win - 1 - i
    } else {
        i
    }
}

/// `split_wiener_filter_coefficients` (pickrst.c): `w = w1 * SCALE + w2`.
fn split_wiener_filter_coefficients(wiener_win: usize, w: &[i32], w1: &mut [i32], w2: &mut [i32]) {
    for i in 0..wiener_win {
        w1[i] = w[i] / WIENER_TAP_SCALE_FACTOR as i32;
        w2[i] = w[i] - w1[i] * WIENER_TAP_SCALE_FACTOR as i32;
        debug_assert_eq!(w[i] as i64, w1[i] as i64 * WIENER_TAP_SCALE_FACTOR + w2[i] as i64);
    }
}

/// `multiply_and_scale` (pickrst.c): `x * w / SCALE` where
/// `w = w1 * SCALE + w2`, without overflowing the direct product.
#[inline]
fn multiply_and_scale(x: i64, w1: i32, w2: i32) -> i64 {
    x * w1 as i64 + x * w2 as i64 / WIENER_TAP_SCALE_FACTOR
}

/// `linsolve_wiener` (pickrst.c): Gaussian elimination with partial pivoting
/// and the b/278065963 overflow-reworked scaling; taps out in
/// `WIENER_TAP_SCALE_FACTOR` fixed point. Returns false when singular.
fn linsolve_wiener(n: usize, a: &mut [i64], stride: usize, b: &mut [i64], x: &mut [i64]) -> bool {
    for k in 0..n.saturating_sub(1) {
        // Partial pivoting: bring the row with the largest pivot to the top.
        for i in (k + 1..n).rev() {
            if a[(i - 1) * stride + k].abs() < a[i * stride + k].abs() {
                for j in 0..n {
                    a.swap(i * stride + j, (i - 1) * stride + j);
                }
                b.swap(i, i - 1);
            }
        }

        let mut max_abs_akj: i64 = 0;
        for j in 0..n {
            let abs_akj = a[k * stride + j].abs();
            if abs_akj > max_abs_akj {
                max_abs_akj = abs_akj;
            }
        }
        let scale_threshold: i64 = 1 << 22;
        let scaler_a: i64 = if max_abs_akj < scale_threshold { 1 } else { 1 << 6 };
        let scaler_c: i64 = if max_abs_akj < scale_threshold { 1 } else { 1 << 7 };
        let scaler = scaler_c * scaler_a;

        // Forward elimination (row-echelon form).
        for i in k..n - 1 {
            if a[k * stride + k] == 0 {
                return false;
            }
            let c = a[(i + 1) * stride + k] / scaler_c;
            let cd = a[k * stride + k];
            for j in 0..n {
                a[(i + 1) * stride + j] -= a[k * stride + j] / scaler_a * c / cd * scaler;
            }
            b[i + 1] -= c * b[k] / cd * scaler_c;
        }
    }
    // Back-substitution.
    for i in (0..n).rev() {
        if a[i * stride + i] == 0 {
            return false;
        }
        let mut c: i64 = 0;
        for j in i + 1..n {
            c += a[i * stride + j] * x[j] / WIENER_TAP_SCALE_FACTOR;
        }
        x[i] = WIENER_TAP_SCALE_FACTOR * (b[i] - c) / a[i * stride + i];
    }
    true
}

/// `update_a_sep_sym` / `update_b_sep_sym` (pickrst.c): fix one direction's
/// taps, re-solve the other. `dir == 0` updates `a` (vertical) from fixed
/// `b`; `dir == 1` updates `b` from fixed `a`. `m`/`h` are the win2 /
/// win2*win2 stats.
fn update_sep_sym(dir: usize, wiener_win: usize, m: &[i64], h: &[i64], a: &mut [i32], b: &mut [i32]) {
    let wiener_win2 = wiener_win * wiener_win;
    let wiener_halfwin1 = (wiener_win >> 1) + 1;
    let mut s = [0i64; WIENER_WIN];
    let mut aa = [0i64; WIENER_HALFWIN1];
    let mut bb = [0i64; WIENER_HALFWIN1 * WIENER_HALFWIN1];
    let mut f1 = [0i32; WIENER_WIN];
    let mut f2 = [0i32; WIENER_WIN];

    // Mc[i] = M + i*win (row i); Hc[i*win + j] = H + i*win*win2 + j*win.
    let mc = |i: usize, j: usize| m[i * wiener_win + j];
    let hc = |i: usize, j: usize, k: usize| h[i * wiener_win * wiener_win2 + j * wiener_win + k];

    if dir == 0 {
        // update_a_sep_sym: A[jj] += Mc[i][j] * b[i] / SCALE
        for i in 0..wiener_win {
            for j in 0..wiener_win {
                let jj = wrap_index(j, wiener_win);
                aa[jj] += mc(i, j) * b[i] as i64 / WIENER_TAP_SCALE_FACTOR;
            }
        }
        split_wiener_filter_coefficients(wiener_win, b, &mut f1, &mut f2);
        for i in 0..wiener_win {
            for j in 0..wiener_win {
                for k in 0..wiener_win {
                    let kk = wrap_index(k, wiener_win);
                    for l in 0..wiener_win {
                        let ll = wrap_index(l, wiener_win);
                        // Hc[j * win + i][k * win2 + l] * b[i] / SCALE, then
                        // * b[j] / SCALE via multiply_and_scale.
                        let x = hc(j, i, k * wiener_win2 + l) * b[i] as i64
                            / WIENER_TAP_SCALE_FACTOR;
                        bb[ll * wiener_halfwin1 + kk] += multiply_and_scale(x, f1[j], f2[j]);
                    }
                }
            }
        }
    } else {
        // update_b_sep_sym: A[ii] += Mc[i][j] * a[j] / SCALE
        for i in 0..wiener_win {
            let ii = wrap_index(i, wiener_win);
            for j in 0..wiener_win {
                aa[ii] += mc(i, j) * a[j] as i64 / WIENER_TAP_SCALE_FACTOR;
            }
        }
        split_wiener_filter_coefficients(wiener_win, a, &mut f1, &mut f2);
        for i in 0..wiener_win {
            let ii = wrap_index(i, wiener_win);
            for j in 0..wiener_win {
                let jj = wrap_index(j, wiener_win);
                for k in 0..wiener_win {
                    for l in 0..wiener_win {
                        let x = hc(i, j, k * wiener_win2 + l) * a[k] as i64
                            / WIENER_TAP_SCALE_FACTOR;
                        bb[jj * wiener_halfwin1 + ii] += multiply_and_scale(x, f1[l], f2[l]);
                    }
                }
            }
        }
    }

    // Normalization enforcement in the system of equations itself.
    for i in 0..wiener_halfwin1 - 1 {
        aa[i] -= aa[wiener_halfwin1 - 1] * 2 + bb[i * wiener_halfwin1 + wiener_halfwin1 - 1]
            - 2 * bb[(wiener_halfwin1 - 1) * wiener_halfwin1 + (wiener_halfwin1 - 1)];
    }
    for i in 0..wiener_halfwin1 - 1 {
        for j in 0..wiener_halfwin1 - 1 {
            bb[i * wiener_halfwin1 + j] -= 2
                * (bb[i * wiener_halfwin1 + (wiener_halfwin1 - 1)]
                    + bb[(wiener_halfwin1 - 1) * wiener_halfwin1 + j]
                    - 2 * bb[(wiener_halfwin1 - 1) * wiener_halfwin1 + (wiener_halfwin1 - 1)]);
        }
    }
    if linsolve_wiener(wiener_halfwin1 - 1, &mut bb, wiener_halfwin1, &mut aa, &mut s) {
        s[wiener_halfwin1 - 1] = WIENER_TAP_SCALE_FACTOR;
        for i in wiener_halfwin1..wiener_win {
            s[i] = s[wiener_win - 1 - i];
            s[wiener_halfwin1 - 1] -= 2 * s[i];
        }
        let out = if dir == 0 { a } else { b };
        for i in 0..wiener_win {
            out[i] = s[i].clamp(-(1 << (WIENER_FILT_BITS - 1)), (1 << (WIENER_FILT_BITS - 1)) - 1)
                as i32;
        }
    }
}

/// `wiener_decompose_sep_sym` (pickrst.c): 4 alternating solve iterations
/// from the identity-ish init filter; outputs the two directions' taps in
/// `WIENER_TAP_SCALE_FACTOR` fixed point.
pub fn wiener_decompose_sep_sym(
    wiener_win: usize,
    m: &[i64],
    h: &[i64],
    a: &mut [i32; WIENER_WIN],
    b: &mut [i32; WIENER_WIN],
) {
    // init_filt = WIENER_FILT_TAP{0,1,2,3}_MIDV mirror.
    const INIT_FILT: [i32; WIENER_WIN] = [3, -7, 15, 106, 15, -7, 3];
    let plane_off = (WIENER_WIN - wiener_win) >> 1;
    for i in 0..wiener_win {
        let v = (WIENER_TAP_SCALE_FACTOR / WIENER_FILT_STEP) as i32 * INIT_FILT[i + plane_off];
        a[i] = v;
        b[i] = v;
    }
    let mut iter = 1;
    while iter < NUM_WIENER_ITERS {
        update_sep_sym(0, wiener_win, m, h, a, b);
        update_sep_sym(1, wiener_win, m, h, a, b);
        iter += 1;
    }
}

/// `compute_score` (pickrst.c): `x'Hx - 2x'M` of the finalized integer
/// filter minus the identity filter's score; positive means the learned
/// filter is WORSE than identity (revert to RESTORE_NONE).
pub fn compute_score(
    wiener_win: usize,
    m: &[i64],
    h: &[i64],
    vfilt: &[i16; 8],
    hfilt: &[i16; 8],
) -> i64 {
    let mut ab = [0i32; WIENER_WIN * WIENER_WIN];
    let mut a = [0i16; WIENER_WIN];
    let mut b = [0i16; WIENER_WIN];
    let plane_off = (WIENER_WIN - wiener_win) >> 1;
    let wiener_win2 = wiener_win * wiener_win;

    a[WIENER_HALFWIN] = WIENER_FILT_STEP as i16;
    b[WIENER_HALFWIN] = WIENER_FILT_STEP as i16;
    for i in 0..WIENER_HALFWIN {
        a[i] = vfilt[i];
        a[WIENER_WIN - i - 1] = vfilt[i];
        b[i] = hfilt[i];
        b[WIENER_WIN - i - 1] = hfilt[i];
        a[WIENER_HALFWIN] -= 2 * a[i];
        b[WIENER_HALFWIN] -= 2 * b[i];
    }
    for k in 0..wiener_win {
        for l in 0..wiener_win {
            ab[k * wiener_win + l] = a[l + plane_off] as i32 * b[k + plane_off] as i32;
        }
    }
    let mut p: i64 = 0;
    let mut q: i64 = 0;
    for k in 0..wiener_win2 {
        p += ab[k] as i64 * m[k] / WIENER_FILT_STEP / WIENER_FILT_STEP;
        for l in 0..wiener_win2 {
            q += ab[k] as i64 * h[k * wiener_win2 + l] * ab[l] as i64
                / WIENER_FILT_STEP
                / WIENER_FILT_STEP
                / WIENER_FILT_STEP
                / WIENER_FILT_STEP;
        }
    }
    let score = q - 2 * p;

    let ip = m[wiener_win2 >> 1];
    let iq = h[(wiener_win2 >> 1) * wiener_win2 + (wiener_win2 >> 1)];
    let iscore = iq - 2 * ip;

    score - iscore
}

/// `finalize_sym_filter` (pickrst.c): fixed-point taps to the coded integer
/// taps with rounding, per-tap clips, symmetric mirror and the implicit
/// centre; the 5-tap window shifts its taps into slots 1/2.
pub fn finalize_sym_filter(wiener_win: usize, f: &[i32; WIENER_WIN], fi: &mut [i16; 8]) {
    const TAP_MINV: [i16; 3] = [-5, -23, -17];
    const TAP_MAXV: [i16; 3] = [10, 8, 46];
    let wiener_halfwin = wiener_win >> 1;
    *fi = [0; 8];
    for i in 0..wiener_halfwin {
        let dividend = f[i] as i64 * WIENER_FILT_STEP;
        let divisor = WIENER_TAP_SCALE_FACTOR;
        fi[i] = if dividend < 0 {
            ((dividend - divisor / 2) / divisor) as i16
        } else {
            ((dividend + divisor / 2) / divisor) as i16
        };
    }
    if wiener_win == WIENER_WIN {
        fi[0] = fi[0].clamp(TAP_MINV[0], TAP_MAXV[0]);
        fi[1] = fi[1].clamp(TAP_MINV[1], TAP_MAXV[1]);
        fi[2] = fi[2].clamp(TAP_MINV[2], TAP_MAXV[2]);
    } else {
        fi[2] = fi[1].clamp(TAP_MINV[2], TAP_MAXV[2]);
        fi[1] = fi[0].clamp(TAP_MINV[1], TAP_MAXV[1]);
        fi[0] = 0;
    }
    // Satisfy filter constraints.
    fi[WIENER_WIN - 1] = fi[0];
    fi[WIENER_WIN - 2] = fi[1];
    fi[WIENER_WIN - 3] = fi[2];
    // The central element has an implicit +WIENER_FILT_STEP.
    fi[3] = -2 * (fi[0] + fi[1] + fi[2]);
}

// ---------------------------------------------------------------------------
// SGR search numeric core.
// ---------------------------------------------------------------------------

/// `SGRPROJ_RST_BITS` / `SGRPROJ_PRJ_BITS` (restoration.h).
const SGRPROJ_RST_BITS: i32 = 4;
const SGRPROJ_PRJ_BITS: i32 = 7;

/// `av1_lowbd_pixel_proj_error_c` + `av1_highbd_pixel_proj_error_c`
/// (pickrst.c) on u16 planes: the SSE of the xq-projected SGR restoration
/// against the source. The lowbd and highbd forms round differently
/// (ROUND_POWER_OF_TWO vs add-half-then-shift with `+d - s` recomposition) —
/// both are ported exactly.
#[allow(clippy::too_many_arguments)]
pub fn pixel_proj_error(
    src: &[u16],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[u16],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    xq: [i32; 2],
    ep: usize,
    highbd: bool,
) -> i64 {
    let (rads, _) = SGR_PARAMS[ep];
    let r0 = rads[0] > 0;
    let r1 = rads[1] > 0;
    let mut err: i64 = 0;
    if !highbd {
        for i in 0..height {
            for j in 0..width {
                let d = dat[dat_off + i * dat_stride + j] as i32;
                let s = src[src_off + i * src_stride + j] as i32;
                let u = d << SGRPROJ_RST_BITS;
                let mut v = u << SGRPROJ_PRJ_BITS;
                if r0 {
                    v += xq[0] * (flt0[i * flt0_stride + j] - u);
                }
                if r1 {
                    v += xq[1] * (flt1[i * flt1_stride + j] - u);
                }
                let e = if r0 || r1 {
                    // ROUND_POWER_OF_TWO(v, 11) - src
                    ((v + (1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)))
                        >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS))
                        - s
                } else {
                    d - s
                };
                err += e as i64 * e as i64;
            }
        }
    } else {
        let half: i32 = 1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1);
        for i in 0..height {
            for j in 0..width {
                let d = dat[dat_off + i * dat_stride + j] as i32;
                let s = src[src_off + i * src_stride + j] as i32;
                if r0 || r1 {
                    let u = d << SGRPROJ_RST_BITS;
                    let mut v = half;
                    if r0 {
                        v += xq[0] * (flt0[i * flt0_stride + j] - u);
                    }
                    if r1 {
                        v += xq[1] * (flt1[i * flt1_stride + j] - u);
                    }
                    let e = (v >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS)) + d - s;
                    err += e as i64 * e as i64;
                } else {
                    let e = d - s;
                    err += e as i64 * e as i64;
                }
            }
        }
    }
    err
}

/// `av1_calc_proj_params_c` + `_high_bd_c` (pickrst.c): the least-squares
/// normal-equation accumulators `H` (2x2) and `C` (2), divided by the pixel
/// count. Identical arithmetic for lowbd/highbd on u16 planes (the C pair
/// differs only in pointer types).
#[allow(clippy::too_many_arguments)]
pub fn calc_proj_params(
    src: &[u16],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[u16],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    ep: usize,
) -> ([[i64; 2]; 2], [i64; 2]) {
    let (rads, _) = SGR_PARAMS[ep];
    let size = (width * height) as i64;
    let mut hh = [[0i64; 2]; 2];
    let mut cc = [0i64; 2];
    let (r0, r1) = (rads[0] > 0, rads[1] > 0);
    for i in 0..height {
        for j in 0..width {
            let u = (dat[dat_off + i * dat_stride + j] as i32) << SGRPROJ_RST_BITS;
            let s = ((src[src_off + i * src_stride + j] as i32) << SGRPROJ_RST_BITS) - u;
            if r0 && r1 {
                let f1 = flt0[i * flt0_stride + j] - u;
                let f2 = flt1[i * flt1_stride + j] - u;
                hh[0][0] += f1 as i64 * f1 as i64;
                hh[1][1] += f2 as i64 * f2 as i64;
                hh[0][1] += f1 as i64 * f2 as i64;
                cc[0] += f1 as i64 * s as i64;
                cc[1] += f2 as i64 * s as i64;
            } else if r0 {
                let f1 = flt0[i * flt0_stride + j] - u;
                hh[0][0] += f1 as i64 * f1 as i64;
                cc[0] += f1 as i64 * s as i64;
            } else if r1 {
                let f2 = flt1[i * flt1_stride + j] - u;
                hh[1][1] += f2 as i64 * f2 as i64;
                cc[1] += f2 as i64 * s as i64;
            }
        }
    }
    if r0 && r1 {
        hh[0][0] /= size;
        hh[0][1] /= size;
        hh[1][1] /= size;
        hh[1][0] = hh[0][1];
        cc[0] /= size;
        cc[1] /= size;
    } else if r0 {
        hh[0][0] /= size;
        cc[0] /= size;
    } else if r1 {
        hh[1][1] /= size;
        cc[1] /= size;
    }
    (hh, cc)
}

/// `signed_rounded_divide` (pickrst.c).
#[inline]
fn signed_rounded_divide(dividend: i64, divisor: i64) -> i64 {
    if dividend < 0 {
        (dividend - divisor / 2) / divisor
    } else {
        (dividend + divisor / 2) / divisor
    }
}

/// `get_proj_subspace` (pickrst.c): solve the 2x2 (or scalar) normal
/// equations for the projection weights `xq`, with the C overflow guards.
#[allow(clippy::too_many_arguments)]
pub fn get_proj_subspace(
    src: &[u16],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[u16],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    ep: usize,
) -> [i32; 2] {
    let (rads, _) = SGR_PARAMS[ep];
    let mut xq = [0i32; 2];
    let (hh, cc) = calc_proj_params(
        src, src_off, width, height, src_stride, dat, dat_off, dat_stride, flt0, flt0_stride,
        flt1, flt1_stride, ep,
    );
    let h = [hh[0][0], hh[0][1], hh[1][0], hh[1][1]];
    let c = cc;
    if rads[0] == 0 {
        let det = h[3];
        if det == 0 {
            return xq;
        }
        xq[0] = 0;
        xq[1] = signed_rounded_divide(c[1] * (1 << SGRPROJ_PRJ_BITS), det) as i32;
    } else if rads[1] == 0 {
        let det = h[0];
        if det == 0 {
            return xq;
        }
        xq[0] = signed_rounded_divide(c[0] * (1 << SGRPROJ_PRJ_BITS), det) as i32;
        xq[1] = 0;
    } else {
        let det = h[0] * h[3] - h[1] * h[2];
        if det == 0 {
            return xq;
        }
        let shift: i64 = 1 << SGRPROJ_PRJ_BITS;
        let div1 = h[3] * c[0] - h[1] * c[1];
        xq[0] = if (div1 > 0 && i64::MAX / shift < div1) || (div1 < 0 && i64::MIN / shift > div1) {
            signed_rounded_divide(div1, det / shift) as i32
        } else {
            signed_rounded_divide(div1 * shift, det) as i32
        };
        let div2 = h[0] * c[1] - h[2] * c[0];
        xq[1] = if (div2 > 0 && i64::MAX / shift < div2) || (div2 < 0 && i64::MIN / shift > div2) {
            signed_rounded_divide(div2, det / shift) as i32
        } else {
            signed_rounded_divide(div2 * shift, det) as i32
        };
    }
    xq
}

/// `encode_xq` (pickrst.c): projection weights to the coded `xqd` domain
/// with the per-radius clamps.
pub fn encode_xq(xq: [i32; 2], ep: usize) -> [i32; 2] {
    use aom_entropy::lr::{SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1, SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1};
    let (rads, _) = SGR_PARAMS[ep];
    let mut xqd = [0i32; 2];
    if rads[0] == 0 {
        xqd[0] = 0;
        xqd[1] = ((1 << SGRPROJ_PRJ_BITS) - xq[1]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
    } else if rads[1] == 0 {
        xqd[0] = xq[0].clamp(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
        xqd[1] = ((1 << SGRPROJ_PRJ_BITS) - xqd[0]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
    } else {
        xqd[0] = xq[0].clamp(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
        xqd[1] = ((1 << SGRPROJ_PRJ_BITS) - xqd[0] - xq[1]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
    }
    xqd
}
