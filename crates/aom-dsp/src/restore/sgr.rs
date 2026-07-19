//! Self-guided restoration — `av1_selfguided_restoration_c` /
//! `av1_apply_selfguided_restoration_c` and their internals (boxsums, the
//! A/B intermediate, the r=2 "fast" pass and the r=1 full pass, `av1_decode_xq`)
//! from av1/common/restoration.c, on u16 planes.

use crate::entropy::lr::SGRPROJ_PRJ_BITS;

/// `SGRPROJ_*` kernel constants (restoration.h).
const SGRPROJ_SGR_BITS: i32 = 8;
const SGRPROJ_SGR: i32 = 1 << SGRPROJ_SGR_BITS;
const SGRPROJ_RST_BITS: i32 = 4;
const SGRPROJ_MTABLE_BITS: u32 = 20;
const SGRPROJ_RECIP_BITS: u32 = 12;
const SGRPROJ_BORDER_VERT: usize = 3;
const SGRPROJ_BORDER_HORZ: usize = 3;

/// `av1_sgr_params` (restoration.c): `(r[2], s[2])` per `ep`. Radius 0
/// disables the pass (s = -1 unused).
pub const SGR_PARAMS: [([i32; 2], [i32; 2]); 16] = [
    ([2, 1], [140, 3236]),
    ([2, 1], [112, 2158]),
    ([2, 1], [93, 1618]),
    ([2, 1], [80, 1438]),
    ([2, 1], [70, 1295]),
    ([2, 1], [58, 1177]),
    ([2, 1], [47, 1079]),
    ([2, 1], [37, 996]),
    ([2, 1], [30, 925]),
    ([2, 1], [25, 863]),
    ([0, 1], [-1, 2589]),
    ([0, 1], [-1, 1618]),
    ([0, 1], [-1, 1177]),
    ([0, 1], [-1, 925]),
    ([2, 0], [56, -1]),
    ([2, 0], [22, -1]),
];

/// `av1_x_by_xplus1[256]` (restoration.c) — 256 * x/(x+1), with 0 -> 1.
const X_BY_XPLUS1: [i32; 256] = [
    1, 128, 171, 192, 205, 213, 219, 224, 228, 230, 233, 235, 236, 238, 239, 240, 241, 242, 243,
    243, 244, 244, 245, 245, 246, 246, 247, 247, 247, 247, 248, 248, 248, 248, 249, 249, 249, 249,
    249, 250, 250, 250, 250, 250, 250, 250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252,
    252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253,
    253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
    253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 256,
];

/// `av1_one_by_x[25]` (restoration.c) — round(2^12 / (n+1)).
const ONE_BY_X: [u32; 25] = [
    4096, 2048, 1365, 1024, 819, 683, 585, 512, 455, 410, 372, 341, 315, 293, 273, 256, 241, 228,
    216, 205, 195, 186, 178, 171, 164,
];

/// Signed `ROUND_POWER_OF_TWO` (arithmetic shift, like the C macro on int).
#[inline]
fn rpot_i32(v: i32, n: u32) -> i32 {
    (v + ((1i32 << n) >> 1)) >> n
}

/// Unsigned `ROUND_POWER_OF_TWO` (the C macro instantiated at uint32_t).
#[inline]
fn rpot_u32(v: u32, n: u32) -> u32 {
    (v + ((1u32 << n) >> 1)) >> n
}

/// `boxsum1` — windowed 3x3 sums (or sums of squares) over `src` (dims
/// `width x height` at `src_stride`, offset `src_off`) into `dst`.
#[allow(clippy::too_many_arguments)]
fn boxsum1(
    src: &[i32],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    sqr: bool,
    dst: &mut [i32],
    dst_stride: usize,
) {
    let sq = |v: i32| if sqr { v * v } else { v };
    // Vertical sum over 3-pixel regions, from src into dst.
    for j in 0..width {
        let mut a = sq(src[src_off + j]);
        let mut b = sq(src[src_off + src_stride + j]);
        let mut c = sq(src[src_off + 2 * src_stride + j]);
        dst[j] = a + b;
        let mut i = 1;
        while i < height - 2 {
            dst[i * dst_stride + j] = a + b + c;
            a = b;
            b = c;
            c = sq(src[src_off + (i + 2) * src_stride + j]);
            i += 1;
        }
        dst[i * dst_stride + j] = a + b + c;
        dst[(i + 1) * dst_stride + j] = b + c;
    }
    // Horizontal sum over 3-pixel regions of dst.
    for i in 0..height {
        let row = i * dst_stride;
        let mut a = dst[row];
        let mut b = dst[row + 1];
        let mut c = dst[row + 2];
        dst[row] = a + b;
        let mut j = 1;
        while j < width - 2 {
            dst[row + j] = a + b + c;
            a = b;
            b = c;
            c = dst[row + j + 2];
            j += 1;
        }
        dst[row + j] = a + b + c;
        dst[row + j + 1] = b + c;
    }
}

/// `boxsum2` — windowed 5x5 sums (or sums of squares).
#[allow(clippy::too_many_arguments)]
fn boxsum2(
    src: &[i32],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    sqr: bool,
    dst: &mut [i32],
    dst_stride: usize,
) {
    let sq = |v: i32| if sqr { v * v } else { v };
    for j in 0..width {
        let mut a = sq(src[src_off + j]);
        let mut b = sq(src[src_off + src_stride + j]);
        let mut c = sq(src[src_off + 2 * src_stride + j]);
        let mut d = sq(src[src_off + 3 * src_stride + j]);
        let mut e = sq(src[src_off + 4 * src_stride + j]);
        dst[j] = a + b + c;
        dst[dst_stride + j] = a + b + c + d;
        let mut i = 2;
        while i < height - 3 {
            dst[i * dst_stride + j] = a + b + c + d + e;
            a = b;
            b = c;
            c = d;
            d = e;
            e = sq(src[src_off + (i + 3) * src_stride + j]);
            i += 1;
        }
        dst[i * dst_stride + j] = a + b + c + d + e;
        dst[(i + 1) * dst_stride + j] = b + c + d + e;
        dst[(i + 2) * dst_stride + j] = c + d + e;
    }
    for i in 0..height {
        let row = i * dst_stride;
        let mut a = dst[row];
        let mut b = dst[row + 1];
        let mut c = dst[row + 2];
        let mut d = dst[row + 3];
        let mut e = dst[row + 4];
        dst[row] = a + b + c;
        dst[row + 1] = a + b + c + d;
        let mut j = 2;
        while j < width - 3 {
            dst[row + j] = a + b + c + d + e;
            a = b;
            b = c;
            c = d;
            d = e;
            e = dst[row + j + 3];
            j += 1;
        }
        dst[row + j] = a + b + c + d + e;
        dst[row + j + 1] = b + c + d + e;
        dst[row + j + 2] = c + d + e;
    }
}

/// `calculate_intermediate_result`: boxsums over the extended block, then the
/// blended A (edge-strength) / B (offset) arrays including a 1-pixel ring,
/// at rows stepped by 2 for the fast (r=2) pass. Returns `(a_buf, b_buf,
/// buf_stride, origin_offset)`.
#[allow(clippy::too_many_arguments)]
fn calculate_intermediate(
    dgd: &[i32],
    dgd_origin: usize,
    width: usize,
    height: usize,
    dgd_stride: usize,
    bit_depth: i32,
    ep: usize,
    radius_idx: usize,
    pass: usize,
) -> (Vec<i32>, Vec<i32>, usize, usize) {
    let (rads, ss) = SGR_PARAMS[ep];
    let r = rads[radius_idx];
    let width_ext = width + 2 * SGRPROJ_BORDER_HORZ;
    let height_ext = height + 2 * SGRPROJ_BORDER_VERT;
    // "Adjusting the stride of A and B here appears to avoid bad cache
    // effects" — must match the C exactly (it changes nothing numerically,
    // but keep the layout for clarity).
    let buf_stride = ((width_ext + 3) & !3) + 16;
    let step = if pass == 0 { 1 } else { 2 };
    let mut a_buf = vec![0i32; buf_stride * (height_ext + 1)];
    let mut b_buf = vec![0i32; buf_stride * (height_ext + 1)];

    let ext_off = dgd_origin - dgd_stride * SGRPROJ_BORDER_VERT - SGRPROJ_BORDER_HORZ;
    let bx = |s: &mut [i32], sqr: bool| {
        if r == 1 {
            boxsum1(
                dgd, ext_off, width_ext, height_ext, dgd_stride, sqr, s, buf_stride,
            );
        } else {
            boxsum2(
                dgd, ext_off, width_ext, height_ext, dgd_stride, sqr, s, buf_stride,
            );
        }
    };
    bx(&mut b_buf, false);
    bx(&mut a_buf, true);

    let org = SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    // A[] / B[] with a 1-pixel ring: i in -1 ..= height, j in -1 ..= width.
    let n = ((2 * r + 1) * (2 * r + 1)) as u32;
    let s = ss[radius_idx] as u32;
    let mut i: i32 = -1;
    while i < height as i32 + 1 {
        for j in -1..=(width as i32) {
            let k = (org as i32 + i * buf_stride as i32 + j) as usize;
            let a = rpot_u32(a_buf[k] as u32, 2 * (bit_depth - 8) as u32);
            let b = rpot_u32(b_buf[k] as u32, (bit_depth - 8) as u32);
            // C: `p = (a * n < b * b) ? 0 : a * n - b * b` (the highbd
            // rounding artefact saturation).
            let p = (a * n).saturating_sub(b * b);
            // p * s < 2^32 for the valid s table (see the C bound comments);
            // wrapping matches C uint32 semantics exactly regardless.
            let z = rpot_u32(p.wrapping_mul(s), SGRPROJ_MTABLE_BITS);
            let a_out = X_BY_XPLUS1[z.min(255) as usize];
            a_buf[k] = a_out;
            b_buf[k] = rpot_u32(
                ((SGRPROJ_SGR - a_out) as u32)
                    .wrapping_mul(b_buf[k] as u32)
                    .wrapping_mul(ONE_BY_X[(n - 1) as usize]),
                SGRPROJ_RECIP_BITS,
            ) as i32;
        }
        i += step;
    }
    (a_buf, b_buf, buf_stride, org)
}

/// `selfguided_restoration_fast_internal` (the r=2 pass, A/B at odd rows).
#[allow(clippy::too_many_arguments)]
fn selfguided_fast(
    dgd: &[i32],
    dgd_origin: usize,
    width: usize,
    height: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    bit_depth: i32,
    ep: usize,
) {
    let (a, b, bs, org) = calculate_intermediate(
        dgd, dgd_origin, width, height, dgd_stride, bit_depth, ep, 0, 1,
    );
    for i in 0..height {
        let k_row = org + i * bs;
        let l_row = dgd_origin + i * dgd_stride;
        let m_row = i * dst_stride;
        if i & 1 == 0 {
            // even row: blend the rows above/below
            let nb = 5;
            for j in 0..width {
                let k = k_row + j;
                let va = (a[k - bs] + a[k + bs]) * 6
                    + (a[k - 1 - bs] + a[k - 1 + bs] + a[k + 1 - bs] + a[k + 1 + bs]) * 5;
                let vb = (b[k - bs] + b[k + bs]) * 6
                    + (b[k - 1 - bs] + b[k - 1 + bs] + b[k + 1 - bs] + b[k + 1 + bs]) * 5;
                let v = va * dgd[l_row + j] + vb;
                dst[m_row + j] = rpot_i32(v, (SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS) as u32);
            }
        } else {
            // odd row: this row's A/B directly
            let nb = 4;
            for j in 0..width {
                let k = k_row + j;
                let va = a[k] * 6 + (a[k - 1] + a[k + 1]) * 5;
                let vb = b[k] * 6 + (b[k - 1] + b[k + 1]) * 5;
                let v = va * dgd[l_row + j] + vb;
                dst[m_row + j] = rpot_i32(v, (SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS) as u32);
            }
        }
    }
}

/// `selfguided_restoration_internal` (the r=1 pass, every row).
#[allow(clippy::too_many_arguments)]
fn selfguided_full(
    dgd: &[i32],
    dgd_origin: usize,
    width: usize,
    height: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    bit_depth: i32,
    ep: usize,
) {
    let (a, b, bs, org) = calculate_intermediate(
        dgd, dgd_origin, width, height, dgd_stride, bit_depth, ep, 1, 0,
    );
    let nb = 5;
    for i in 0..height {
        for j in 0..width {
            let k = org + i * bs + j;
            let va = (a[k] + a[k - 1] + a[k + 1] + a[k - bs] + a[k + bs]) * 4
                + (a[k - 1 - bs] + a[k - 1 + bs] + a[k + 1 - bs] + a[k + 1 + bs]) * 3;
            let vb = (b[k] + b[k - 1] + b[k + 1] + b[k - bs] + b[k + bs]) * 4
                + (b[k - 1 - bs] + b[k - 1 + bs] + b[k + 1 - bs] + b[k + 1 + bs]) * 3;
            let v = va * dgd[dgd_origin + i * dgd_stride + j] + vb;
            dst[i * dst_stride + j] =
                rpot_i32(v, (SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS) as u32);
        }
    }
}

/// `av1_selfguided_restoration_c`: stage the `[-3, +3)`-extended source into
/// an i32 buffer, then run the enabled passes into `flt0` (r\[0\]=2 fast) and
/// `flt1` (r\[1\]=1 full), both at `flt_stride = width`.
#[allow(clippy::too_many_arguments)]
pub fn selfguided_restoration(
    dgd: &[u16],
    dgd_off: usize,
    dgd_stride: usize,
    width: usize,
    height: usize,
    flt0: &mut [i32],
    flt1: &mut [i32],
    flt_stride: usize,
    ep: usize,
    bit_depth: i32,
) {
    let dgd32_stride = width + 2 * SGRPROJ_BORDER_HORZ;
    let mut dgd32 = vec![0i32; dgd32_stride * (height + 2 * SGRPROJ_BORDER_VERT)];
    for i in 0..height + 2 * SGRPROJ_BORDER_VERT {
        for j in 0..dgd32_stride {
            // (i - 3, j - 3) relative to the block origin, via signed math.
            let src_idx = (dgd_off as isize
                + (i as isize - SGRPROJ_BORDER_VERT as isize) * dgd_stride as isize
                + (j as isize - SGRPROJ_BORDER_HORZ as isize)) as usize;
            dgd32[i * dgd32_stride + j] = dgd[src_idx] as i32;
        }
    }
    let origin = SGRPROJ_BORDER_VERT * dgd32_stride + SGRPROJ_BORDER_HORZ;
    let (rads, _) = SGR_PARAMS[ep];
    debug_assert!(!(rads[0] == 0 && rads[1] == 0));
    if rads[0] > 0 {
        selfguided_fast(
            &dgd32,
            origin,
            width,
            height,
            dgd32_stride,
            flt0,
            flt_stride,
            bit_depth,
            ep,
        );
    }
    if rads[1] > 0 {
        selfguided_full(
            &dgd32,
            origin,
            width,
            height,
            dgd32_stride,
            flt1,
            flt_stride,
            bit_depth,
            ep,
        );
    }
}

/// `av1_decode_xq` (restoration.c): the projection weights from the coded
/// `xqd` per the parameter set's radii.
pub fn decode_xq(xqd: &[i32; 2], ep: usize) -> [i32; 2] {
    let (rads, _) = SGR_PARAMS[ep];
    if rads[0] == 0 {
        [0, (1 << SGRPROJ_PRJ_BITS) - xqd[1]]
    } else if rads[1] == 0 {
        [xqd[0], 0]
    } else {
        [xqd[0], (1 << SGRPROJ_PRJ_BITS) - xqd[0] - xqd[1]]
    }
}

/// `av1_apply_selfguided_restoration_c`: the guided passes then the final
/// projection blend `w = u + (xq0*(flt0-u) + xq1*(flt1-u)) >> 11`, clipped to
/// the pixel range. Reads `dat[dat_off..]` at `[-3, +3)` margins; writes the
/// `w x h` block at `dst[dst_off..]`.
#[allow(clippy::too_many_arguments)]
pub fn apply_selfguided_restoration(
    dat: &[u16],
    dat_off: usize,
    stride: usize,
    width: usize,
    height: usize,
    ep: usize,
    xqd: &[i32; 2],
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    bit_depth: i32,
) {
    let mut flt0 = vec![0i32; width * height];
    let mut flt1 = vec![0i32; width * height];
    selfguided_restoration(
        dat, dat_off, stride, width, height, &mut flt0, &mut flt1, width, ep, bit_depth,
    );
    let (rads, _) = SGR_PARAMS[ep];
    let xq = decode_xq(xqd, ep);
    let pixel_max = (1i32 << bit_depth) - 1;
    for i in 0..height {
        for j in 0..width {
            let k = i * width + j;
            let pre_u = dat[dat_off + i * stride + j] as i32;
            let u = pre_u << SGRPROJ_RST_BITS;
            let mut v = u << SGRPROJ_PRJ_BITS;
            if rads[0] > 0 {
                v += xq[0] * (flt0[k] - u);
            }
            if rads[1] > 0 {
                v += xq[1] * (flt1[k] - u);
            }
            // C narrows through int16_t before the clip.
            let w = rpot_i32(v, (SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS) as u32) as i16;
            dst[dst_off + i * dst_stride + j] = (w as i32).clamp(0, pixel_max) as u16;
        }
    }
}
