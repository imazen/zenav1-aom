//! HOG-based intra directional-mode pruning (libaom
//! `av1/encoder/intra_mode_search_utils.h`): `generate_hog` (Sobel gradient
//! histogram, f32) -> `av1_nn_predict` on `av1_intra_hog_model_nnconfig`
//! (a 0-hidden-layer 32->8 linear model) -> the `<= threshold` skip mask over
//! the 8 directional modes. ACTIVE at speed-0 all-intra
//! (`intra_pruning_with_hog = 1`, threshold `-1.2f`,
//! intra_mode_search.c:1501-1510) — this fills the
//! `directional_mode_skip_mask` gating input of the mode loop.
//!
//! FP discipline:
//! - The histogram/normalize path is plain f32 in the C's exact accumulation
//!   order (no FMA in the reference build).
//! - The NN is RTCD-dispatched in production; on the AVX2-capable x86-64
//!   reference environment it resolves to `av1_nn_predict_avx2`, whose f32
//!   ACCUMULATION ORDER (8-lane mul + hadd/permute reduction trees) differs
//!   from the C and SSE3 variants at ULP level. [`hog_nn_predict`] therefore
//!   replicates the AVX2 kernel's exact lane math, scalar-emulated
//!   ([`hadd256`] etc. mirror `_mm256_hadd_ps` / `_mm256_permute2f128_ps`
//!   semantics element-for-element). The differential proves both that the
//!   port matches `av1_nn_predict_avx2` bit-for-bit and that the RTCD
//!   dispatch resolves to it on the running machine.
//! - Gradient caching (`generate_hog_using_gradient_cache`, the path the
//!   speed-0 encoder takes via `produce_gradients_for_sb`) is numerically
//!   IDENTICAL to the direct walk: the cached per-pixel Sobel values are
//!   computed from the same pixels (interior positions only read rows/cols
//!   inside the block), `>> 1` == `/ 2` for the non-negative sums, and the
//!   accumulation order matches — so this port implements the direct form.
//!
//! The u16 pixel type serves both depths: `lowbd_generate_hog` (u8 planes)
//! and `highbd_generate_hog` (u16) perform identical integer Sobel math on
//! the same values, so one implementation is bit-exact against either kernel.

/// `BINS` (intra_mode_search_utils.h).
pub const HOG_BINS: usize = 32;
/// `DIRECTIONAL_MODES` (enums.h).
pub const DIRECTIONAL_MODES: usize = 8;

/// `av1_intra_hog_model_bias` (intra_mode_search_utils.h) — literals kept
/// verbatim from the C header.
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
pub const INTRA_HOG_MODEL_BIAS: [f32; 8] = [
    0.450578, 0.695518, -0.717944, -0.639894, -0.602019, -0.453454, 0.055857, -0.465480,
];

/// `get_hist_bin_idx` thresholds (intra_mode_search_utils.h; the `INT32_MAX`
/// terminator makes the linear scan total).
#[rustfmt::skip]
const BIN_THRESHOLDS: [i32; HOG_BINS] = [
    -1334015, -441798, -261605, -183158, -138560, -109331, -88359, -72303,
    -59392,   -48579,  -39272,  -30982,  -23445,  -16400,  -9715,  -3194,
    3227,     9748,    16433,   23478,   31015,   39305,   48611,  59425,
    72336,    88392,   109364,  138593,  183191,  261638,  441831, i32::MAX,
];

/// `get_hist_bin_idx(dx, dy)`: fixed-point `dy/dx` ratio bisected into the 32
/// orientation bins (`FIX_PREC_BITS = 16`; C truncating signed division).
/// `dx != 0` (the caller splits the vertical case across bins 0 and 31).
pub fn get_hist_bin_idx(dx: i32, dy: i32) -> usize {
    let ratio = (dy * (1 << 16)) / dx;
    let (lo, hi) = if ratio <= BIN_THRESHOLDS[7] {
        (0usize, 7usize)
    } else if ratio <= BIN_THRESHOLDS[15] {
        (8, 15)
    } else if ratio <= BIN_THRESHOLDS[23] {
        (16, 23)
    } else {
        (24, 31)
    };
    (lo..=hi)
        .find(|&idx| ratio <= BIN_THRESHOLDS[idx])
        .expect("no valid histogram bin")
}

/// `PixelLevelGradientInfo` (block.h): one interior pixel's cached Sobel
/// gradient for `generate_hog_using_gradient_cache`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HogGradPx {
    /// `abs(dx) + abs(dy)` — max 2*4*4095 < u16::MAX at every depth.
    abs_dx_abs_dy_sum: u16,
    /// `get_hist_bin_idx(dx, dy)`, or -1 when `dx == 0`.
    hist_bin_idx: i8,
    is_dx_zero: bool,
}

/// `x->pixel_gradient_info + plane * MAX_SB_SQUARE` + `is_sb_gradient_cached`
/// (block.h): the per-superblock Sobel/bin table `produce_gradients_for_sb`
/// fills once (`compute_gradient_info_sb`) and every `collect_hog_data` in the
/// SB then reads. The port fills it LAZILY on the first hog call inside an SB
/// — same pure function of the same source pixels, so the histograms are
/// identical; only the recomputation count changes (C pays it once per SB,
/// the direct form paid it once per block).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HogGradSlot {
    /// The SB origin this table covers, in mi — `None` until first fill.
    key: Option<(i32, i32)>,
    /// Plane-unit SB extent: row stride `sb_w` x `sb_h` rows (4:2:2 is NOT
    /// square — `block_size_wide/high >> ss_x/ss_y`).
    sb_w: usize,
    sb_h: usize,
    px: Vec<HogGradPx>,
}

impl HogGradSlot {
    /// Fill `self` for the SB at (`sb_mi_row`, `sb_mi_col`) unless already
    /// cached. `sb_src_off` is the plane offset of that SB's origin pixel;
    /// the interior walk reads the `sb_w`×`sb_h` region rooted there
    /// (always inside the edge-extended source plane).
    fn ensure(
        &mut self,
        src: &[u16],
        sb_src_off: usize,
        src_stride: usize,
        sb_w: usize,
        sb_h: usize,
        sb_mi_row: i32,
        sb_mi_col: i32,
    ) {
        if self.key == Some((sb_mi_row, sb_mi_col)) && self.sb_w == sb_w && self.sb_h == sb_h {
            return;
        }
        self.px.clear();
        self.px.resize(sb_w * sb_h, HogGradPx::default());
        for r in 1..sb_h.saturating_sub(1) {
            let row = sb_src_off + r * src_stride;
            for c in 1..sb_w.saturating_sub(1) {
                let p = row + c;
                let dx = (i32::from(src[p - src_stride + 1])
                    + 2 * i32::from(src[p + 1])
                    + i32::from(src[p + src_stride + 1]))
                    - (i32::from(src[p - src_stride - 1])
                        + 2 * i32::from(src[p - 1])
                        + i32::from(src[p + src_stride - 1]));
                let dy = (i32::from(src[p + src_stride - 1])
                    + 2 * i32::from(src[p + src_stride])
                    + i32::from(src[p + src_stride + 1]))
                    - (i32::from(src[p - src_stride - 1])
                        + 2 * i32::from(src[p - src_stride])
                        + i32::from(src[p - src_stride + 1]));
                self.px[r * sb_w + c] = HogGradPx {
                    is_dx_zero: dx == 0,
                    abs_dx_abs_dy_sum: (dx.abs() + dy.abs()) as u16,
                    hist_bin_idx: if dx != 0 {
                        get_hist_bin_idx(dx, dy) as i8
                    } else {
                        -1
                    },
                };
            }
        }
        self.key = Some((sb_mi_row, sb_mi_col));
        self.sb_w = sb_w;
        self.sb_h = sb_h;
    }

    /// `generate_hog_using_gradient_cache` (intra_mode_search_utils.h:365):
    /// histogram walk over the cached per-pixel gradients. `block_off` is the
    /// block's offset INSIDE the table (`block_offset_in_grad_cache`); the
    /// walk order and f32 accumulation match the direct form exactly
    /// (`>> 1` == `/ 2` on the non-negative sums). `None` when the block's
    /// cache rect does not fit the table — in production blocks never exceed
    /// their SB (bsize <= sb_size), but synthetic callers may place a
    /// larger-than-SB block; the caller then takes the direct walk.
    fn hog(&self, block_off: usize, rows: usize, cols: usize) -> Option<[f32; HOG_BINS]> {
        if self.sb_w == 0 {
            return None;
        }
        let (row0, col0) = (block_off / self.sb_w, block_off % self.sb_w);
        if col0 + cols > self.sb_w || row0 + rows > self.sb_h {
            return None;
        }
        let mut hist = [0f32; HOG_BINS];
        let mut total = 0.1f32;
        if rows >= 3 && cols >= 3 {
            for r in 1..rows - 1 {
                let row = block_off + r * self.sb_w;
                for c in 1..cols - 1 {
                    let g = self.px[row + c];
                    let temp = i32::from(g.abs_dx_abs_dy_sum);
                    if temp == 0 {
                        continue;
                    }
                    total += temp as f32;
                    if g.is_dx_zero {
                        hist[0] += (temp >> 1) as f32;
                        hist[HOG_BINS - 1] += (temp >> 1) as f32;
                    } else {
                        hist[g.hist_bin_idx as usize] += temp as f32;
                    }
                }
            }
        }
        for h in hist.iter_mut() {
            *h /= total;
        }
        Some(hist)
    }
}

/// `x->pixel_gradient_info` pair (PLANE_TYPE_Y slot 0, PLANE_TYPE_UV slot 1).
/// Owned per tile-walk — a fresh `TileCtxState` per tile makes stale
/// cross-frame reuse impossible.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HogGradCache {
    pub y: HogGradSlot,
    pub uv: HogGradSlot,
}

/// `collect_hog_data`'s cached arm for the luma hog: ensure the Y table for
/// the block's SB, then walk it. `mi_row`/`mi_col` locate the block inside the
/// `sb_mi`-per-side superblock rooted at (`mi_row & !(sb_mi-1)`, …).
#[allow(clippy::too_many_arguments)]
pub fn generate_hog_y_cached(
    cache: &mut HogGradCache,
    src: &[u16],
    src_base: usize,
    src_stride: usize,
    mi_row: i32,
    mi_col: i32,
    sb_mi: i32,
    rows: usize,
    cols: usize,
) -> [f32; HOG_BINS] {
    let sb_mi_row = mi_row & !(sb_mi - 1);
    let sb_mi_col = mi_col & !(sb_mi - 1);
    let sb_px = (sb_mi as usize) * 4;
    let sb_off = src_base + (sb_mi_row as usize * 4) * src_stride + sb_mi_col as usize * 4;
    cache
        .y
        .ensure(src, sb_off, src_stride, sb_px, sb_px, sb_mi_row, sb_mi_col);
    let block_off = ((mi_row - sb_mi_row) as usize * 4) * sb_px + (mi_col - sb_mi_col) as usize * 4;
    cache.y.hog(block_off, rows, cols).unwrap_or_else(|| {
        generate_hog(
            src,
            src_base + (mi_row as usize * 4) * src_stride + mi_col as usize * 4,
            src_stride,
            rows,
            cols,
        )
    })
}

/// The chroma arm — cache indexes are in CHROMA plane units
/// (`sb_width = block_size_wide[sb_size] >> ss_x`), so the SB origin and block
/// offset shift by `ss` (intra_mode_search_utils.h:371-382).
#[allow(clippy::too_many_arguments)]
pub fn generate_hog_uv_cached(
    cache: &mut HogGradCache,
    src: &[u16],
    src_base: usize,
    src_stride: usize,
    mi_row: i32,
    mi_col: i32,
    sb_mi: i32,
    ss_x: usize,
    ss_y: usize,
    rows: usize,
    cols: usize,
) -> [f32; HOG_BINS] {
    let sb_mi_row = mi_row & !(sb_mi - 1);
    let sb_mi_col = mi_col & !(sb_mi - 1);
    let sb_w = ((sb_mi as usize) * 4) >> ss_x;
    let sb_h = ((sb_mi as usize) * 4) >> ss_y;
    let sb_off = src_base
        + ((sb_mi_row as usize * 4) >> ss_y) * src_stride
        + ((sb_mi_col as usize * 4) >> ss_x);
    cache
        .uv
        .ensure(src, sb_off, src_stride, sb_w, sb_h, sb_mi_row, sb_mi_col);
    let block_off = (((mi_row - sb_mi_row) as usize * 4) >> ss_y) * sb_w
        + (((mi_col - sb_mi_col) as usize * 4) >> ss_x);
    cache.uv.hog(block_off, rows, cols).unwrap_or_else(|| {
        generate_hog(
            src,
            src_base
                + ((mi_row as usize * 4) >> ss_y) * src_stride
                + ((mi_col as usize * 4) >> ss_x),
            src_stride,
            rows,
            cols,
        )
    })
}

/// `lowbd_generate_hog` / `highbd_generate_hog`: Sobel-gradient orientation
/// histogram over the interior pixels (`r`/`c` in `1..dim-1`) of the
/// edge-clipped `rows x cols` block at `src[src_off..]`, normalized by the
/// total gradient magnitude (+0.1f seed). f32 accumulation in the C's exact
/// walk order.
pub fn generate_hog(
    src: &[u16],
    src_off: usize,
    stride: usize,
    rows: usize,
    cols: usize,
) -> [f32; HOG_BINS] {
    let mut hist = [0f32; HOG_BINS];
    let mut total = 0.1f32;
    // KB-PERF-32: three row slices walked as overlapping 3-wide windows.
    //
    // The arithmetic, the walk order and the f32 accumulation order are exactly
    // what the per-pixel `src[src_off + r * stride + c]` closure this replaces
    // produced — window `i` of a `cols`-long row is column `c = i + 1`, and
    // `windows(3)` yields `cols - 2` of them, which is `1..cols - 1`. What
    // changes is that the compiler sees ONE bounds check per row instead of
    // EIGHT per pixel, and each tap is a `[u16; 3]`-shaped read whose indices
    // are statically in range.
    //
    // The `rows >= 3 && cols >= 3` guard is the same set the old loop bounds
    // selected (`1..rows - 1` and `1..cols - 1` are both empty below 3), except
    // that `cols == 0` returned a normalized all-zero histogram here where the
    // old `cols - 1` would have underflowed. No caller reaches that.
    if rows >= 3 && cols >= 3 {
        for r in 1..rows - 1 {
            let (ra, rb, rd) =
                (src_off + (r - 1) * stride, src_off + r * stride, src_off + (r + 1) * stride);
            let above = &src[ra..ra + cols];
            let cur = &src[rb..rb + cols];
            let below = &src[rd..rd + cols];
            for ((wa, wc), wb) in above.windows(3).zip(cur.windows(3)).zip(below.windows(3)) {
                let (a0, a1, a2) = (wa[0] as i32, wa[1] as i32, wa[2] as i32);
                let (c0, c2) = (wc[0] as i32, wc[2] as i32);
                let (b0, b1, b2) = (wb[0] as i32, wb[1] as i32, wb[2] as i32);
                // Sobel: dx from the right/left columns, dy from below/above rows.
                let dx = (a2 + 2 * c2 + b2) - (a0 + 2 * c0 + b0);
                let dy = (b0 + 2 * b1 + b2) - (a0 + 2 * a1 + a2);
                if dx == 0 && dy == 0 {
                    continue;
                }
                let temp = dx.abs() + dy.abs();
                if temp == 0 {
                    continue;
                }
                total += temp as f32;
                if dx == 0 {
                    hist[0] += (temp / 2) as f32;
                    hist[HOG_BINS - 1] += (temp / 2) as f32;
                } else {
                    hist[get_hist_bin_idx(dx, dy)] += temp as f32;
                }
            }
        }
    }
    // normalize_hog.
    for h in hist.iter_mut() {
        *h /= total;
    }
    hist
}

// ---------------------------------------------------------------------------
// av1_nn_predict_avx2 (av1/encoder/x86/ml_avx2.c) — scalar emulation of the
// exact 8-lane f32 math for the multiple-of-8 layer shape the HOG model uses.
// ---------------------------------------------------------------------------

type M256 = [f32; 8];

#[inline]
fn mul256(a: M256, b: M256) -> M256 {
    core::array::from_fn(|i| a[i] * b[i])
}

#[inline]
fn add256(a: M256, b: M256) -> M256 {
    core::array::from_fn(|i| a[i] + b[i])
}

/// `_mm256_hadd_ps(a, b)`: per 128-bit lane `[a0+a1, a2+a3, b0+b1, b2+b3]`.
#[inline]
fn hadd256(a: M256, b: M256) -> M256 {
    [
        a[0] + a[1],
        a[2] + a[3],
        b[0] + b[1],
        b[2] + b[3],
        a[4] + a[5],
        a[6] + a[7],
        b[4] + b[5],
        b[6] + b[7],
    ]
}

/// `_mm256_permute2f128_ps(a, b, 0x20)` = [lo(a), lo(b)];
/// `0x31` = [hi(a), hi(b)].
#[inline]
fn permute2f128(a: M256, b: M256, hi: bool) -> M256 {
    if hi {
        [a[4], a[5], a[6], a[7], b[4], b[5], b[6], b[7]]
    } else {
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }
}

#[inline]
fn load8(s: &[f32], off: usize) -> M256 {
    core::array::from_fn(|i| s[off + i])
}

/// `nn_propagate_8to8` (ml_avx2.c): one layer with `num_inputs_to_process`
/// (multiple of 8) inputs and `num_outputs` (multiple of 8) outputs —
/// per 8-output group, 4x two-row `mul`+`hadd` trees per 8-input chunk,
/// `hadd`/`permute2f128` reduced, accumulated 8-lane, bias added last.
#[allow(clippy::too_many_arguments)] // mirrors the C signature
fn nn_propagate_8to8(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    num_inputs_to_process: usize,
    tot_num_inputs: usize,
    num_outputs: usize,
    output_nodes: &mut [f32],
    is_clip_required: bool,
) {
    let mut out = 0usize;
    while out < num_outputs {
        let bias_reg = load8(bias, out);
        let mut in_result = [0f32; 8];
        let mut inp = 0usize;
        while inp < num_inputs_to_process {
            let inputs256 = load8(inputs, inp);
            let weight_idx = inp + out * tot_num_inputs;
            let mut hadd = [[0f32; 8]; 4];
            for (i, h) in hadd.iter_mut().enumerate() {
                let index = weight_idx + 2 * i * tot_num_inputs;
                let weight0 = load8(weights, index);
                let weight1 = load8(weights, index + tot_num_inputs);
                *h = hadd256(mul256(inputs256, weight0), mul256(inputs256, weight1));
            }
            let hh0 = hadd256(hadd[0], hadd[1]);
            let hh1 = hadd256(hadd[2], hadd[3]);
            let ht_0 = permute2f128(hh0, hh1, false);
            let ht_1 = permute2f128(hh0, hh1, true);
            in_result = add256(in_result, add256(ht_0, ht_1));
            inp += 8;
        }
        in_result = add256(in_result, bias_reg);
        if is_clip_required {
            for v in in_result.iter_mut() {
                *v = v.max(0.0);
            }
        }
        output_nodes[out..out + 8].copy_from_slice(&in_result);
        out += 8;
    }
}

/// `av1_nn_output_prec_reduce` (ml.c): quantize each output to 9 fractional
/// bits — `((int)(v * 512 + 0.5)) * (float)(1.0/512)`; the `+ 0.5` promotes
/// to double, the `(int)` truncates toward zero.
pub fn nn_output_prec_reduce(output: &mut [f32]) {
    const PREC: f32 = 512.0;
    const INV_PREC: f32 = (1.0 / 512.0) as f32;
    for v in output.iter_mut() {
        *v = ((f64::from(*v * PREC) + 0.5) as i32) as f32 * INV_PREC;
    }
}

/// `av1_nn_predict` on `av1_intra_hog_model_nnconfig` as the AVX2 kernel
/// computes it (0 hidden layers; 32 inputs / 8 outputs, both multiples of 8
/// => the `nn_propagate_8to8` path with `is_clip_required = false` on the
/// output layer), then the `reduce_prec` quantization.
pub fn hog_nn_predict(hist: &[f32; HOG_BINS], reduce_prec: bool) -> [f32; DIRECTIONAL_MODES] {
    let mut scores = [0f32; DIRECTIONAL_MODES];
    nn_propagate_8to8(
        hist,
        &intra_hog_model_weights::INTRA_HOG_MODEL_WEIGHTS,
        &INTRA_HOG_MODEL_BIAS,
        HOG_BINS,
        HOG_BINS,
        DIRECTIONAL_MODES,
        &mut scores,
        false,
    );
    if reduce_prec {
        nn_output_prec_reduce(&mut scores);
    }
    scores
}

/// `prune_intra_mode_with_hog` (luma: `is_chroma = 0`, plane Y, ss 0/0):
/// frame-edge clip the block dims (`mb_to_*_edge` 1/8-pel MACROBLOCKD
/// fields), HOG histogram, the (1+ss_x)*(1+ss_y) luma scale (an exact *1
/// no-op, kept for structure), NN scores, then `score <= th` SETS the mask
/// entry for V_PRED..D67_PRED (modes 1..=8; entries are never cleared — the
/// C caller zero-initializes). Speed-0 threshold: `-1.2f`
/// (`thresh[intra_pruning_with_hog - 1]`, intra_mode_search.c:1505).
#[allow(clippy::too_many_arguments)]
pub fn prune_intra_mode_with_hog_y(
    src: &[u16],
    src_base: usize,
    src_stride: usize,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    sb_mi: i32,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    th: f32,
    directional_mode_skip_mask: &mut [bool; 13],
    cache: &mut HogGradCache,
) {
    const BLK_W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const BLK_H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    let bh = BLK_H[bsize] as i32;
    let bw = BLK_W[bsize] as i32;
    let rows = if mb_to_bottom_edge >= 0 {
        bh
    } else {
        (mb_to_bottom_edge >> 3) + bh
    } as usize;
    let cols = if mb_to_right_edge >= 0 {
        bw
    } else {
        (mb_to_right_edge >> 3) + bw
    } as usize;

    let mut hog = generate_hog_y_cached(
        cache, src, src_base, src_stride, mi_row, mi_col, sb_mi, rows, cols,
    );
    // collect_hog_data: hog[b] *= (1 + ss_x) * (1 + ss_y) — luma ss 0/0.
    for b in hog.iter_mut() {
        *b *= 1.0;
    }

    let scores = hog_nn_predict(&hog, true);
    for mode in 1..=8usize {
        // UV_V_PRED..UV_D67_PRED == V_PRED..D67_PRED for luma.
        if scores[mode - 1] <= th {
            directional_mode_skip_mask[mode] = true;
        }
    }
}

/// `prune_intra_mode_with_hog` (chroma: `is_chroma = 1`, plane U): the chroma
/// analogue of [`prune_intra_mode_with_hog_y`]. `collect_hog_data`
/// (intra_mode_search_utils.h:406-435) computes the HOG on the **U-plane**
/// pixels using the block's rows/cols derived from the LUMA `bsize` clipped to
/// the frame edge (`mb_to_*_edge`, 1/8 luma-pel) then right-shifted by the
/// chroma subsampling, and finally scales every bin by `(1 + ss_x) * (1 + ss_y)`
/// (so luma and chroma HOG land on the same scale). The NN scores then set the
/// `UV_V_PRED..UV_D67_PRED` (modes 1..=8) skip-mask entries whose score `<= th`.
/// For an intra frame at chroma-prune level 2 the C threshold is
/// `thresh[1][1] = -1.2` (intra_mode_search.c:961-970).
#[allow(clippy::too_many_arguments)]
pub fn prune_intra_mode_with_hog_uv(
    src_u: &[u16],
    src_base: usize,
    src_stride: usize,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
    mi_row: i32,
    mi_col: i32,
    sb_mi: i32,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    th: f32,
    directional_mode_skip_mask: &mut [bool; 13],
    cache: &mut HogGradCache,
) {
    const BLK_W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const BLK_H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    // `collect_hog_data`: bh/bw are LUMA dims; the frame-edge clip uses the LUMA
    // `mb_to_*_edge` (1/8 luma-pel); the `>> ss` converts to chroma dims.
    let bh = BLK_H[bsize] as i32;
    let bw = BLK_W[bsize] as i32;
    let rows = (if mb_to_bottom_edge >= 0 {
        bh
    } else {
        (mb_to_bottom_edge >> 3) + bh
    } >> ss_y) as usize;
    let cols = (if mb_to_right_edge >= 0 {
        bw
    } else {
        (mb_to_right_edge >> 3) + bw
    } >> ss_x) as usize;

    let mut hog = generate_hog_uv_cached(
        cache, src_u, src_base, src_stride, mi_row, mi_col, sb_mi, ss_x, ss_y, rows, cols,
    );
    // collect_hog_data: hog[b] *= (1 + ss_x) * (1 + ss_y).
    let scale = ((1 + ss_x) * (1 + ss_y)) as f32;
    for b in hog.iter_mut() {
        *b *= scale;
    }

    let scores = hog_nn_predict(&hog, true);
    for mode in 1..=8usize {
        if scores[mode - 1] <= th {
            directional_mode_skip_mask[mode] = true;
        }
    }
}

mod intra_hog_model_weights;
pub use intra_hog_model_weights::INTRA_HOG_MODEL_WEIGHTS;
