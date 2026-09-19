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
//!
//! # WIRED INTO THE ENCODER — and it is ON by default at speeds 0-4
//!
//! **Corrected 2026-09-08. The previous note here said the opposite, and said
//! it in the direction that gets someone hurt:** *"Nothing in `aom-encode` calls
//! this module … a change to this module cannot move any encoder byte gate
//! (nothing on the encode path reads it)."* That was true when written
//! (2026-08-06) and is false now. `aom_encode::key_frame::encode_key_frame` —
//! the standalone entry point, the one zenavif's backend calls and the one
//! 427/427 byte-identity is measured on — calls
//! [`pick_filter_restoration`] at `key_frame.rs:2084`, on its own post-CDEF
//! reconstruction.
//!
//! So: **a change to this module CAN move an encoder byte gate.** Anything
//! edited here must keep `aom-encode`'s `self_contained_key_frame.rs` and the
//! `lr_*` gates byte-identical, not just `pick_diff` / `pick_search` green.
//!
//! Reach: loop restoration is ON by default in ALLINTRA at `--cpu-used` 0-4
//! and OFF from 5 (`speed_features.c:519-520` disables Wiener + SGR, so
//! `enable_restoration &= 0`). That speed split is why this module is invisible
//! in every profile taken at cpu-used 6 — see
//! `benchmarks/encoder_x86_profile_2026-09-08.md`, which measures the search at
//! **26 % of the encode-time gap to libaom at speed 0 and 0 % at speed 6**.
//!
//! Still true from the old note: these 2k lines compile into decode-only builds
//! too, since the facade always pulls `aom-dsp` — a `restore-search` cargo
//! feature would fix that and is the obvious follow-up.

use crate::restore::sgr::SGR_PARAMS;
use crate::entropy::lr::{WIENER_HALFWIN, WIENER_WIN};

/// `WIENER_WIN2` / `WIENER_HALFWIN1` (restoration.h).
pub const WIENER_WIN2: usize = WIENER_WIN * WIENER_WIN;
/// `WIENER_WIN2` rounded up to a whole `i32x8`.
///
/// The per-row `H` accumulator inside [`compute_stats`] is strided by
/// `round_up(win2, 8)` rather than by `win2`, and the window vector `y` is
/// zero-padded to the same length. That is PURELY a layout choice and changes
/// no value: the padding lanes hold `y[l] = 0`, so the products written past
/// column `win2` are zero added to zero, and nothing ever reads them (the fold
/// back into the `i64` `H` walks `l < win2`).
///
/// It exists because without it the vector loop needs a scalar tail per `k`:
/// at `win7` those tails are ~171 scalar multiply-accumulates per pixel against
/// only 132 vector iterations. Measured on the profile cell, the padding is
/// worth **18.4 ms -> 17.5 ms** — real, and much smaller than that op count
/// predicts, because the tails were not the limit (see `acc_stat_line_impl`).
const WIENER_H_STRIDE: usize = WIENER_WIN2.div_ceil(8) * 8;
/// Length of the internal padded `H` row accumulator.
const WIENER_H_ROW_LEN: usize = WIENER_WIN2 * WIENER_H_STRIDE;
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

/// Pixel carrier for the lowbd restoration-statistics kernels — `u8` at
/// bd8, `u16` for the highbd-shaped callers. The kernels' arithmetic is
/// identical on either carrier (bd8 pixel values fit `i16`/`u8`), so the
/// load-site conversions below are the only pixel-width surface; halving
/// them halves the `dgd`/`src` read bandwidth the window gathers pay.
pub trait LrPixel: Copy {
    const ZERO: Self;
    fn to_i16(self) -> i16;
    fn to_i32(self) -> i32;
    fn to_u16(self) -> u16;
    fn to_u64(self) -> u64;
    /// Widen 16 pixels to the `i16x16` lanes both carriers produce (u8
    /// through `cvtepu8`, u16 as the direct load).
    #[cfg(target_arch = "x86_64")]
    const LD16: fn(
        archmage::X64V3Token,
        &[Self; 16],
    ) -> archmage::intrinsics::x86_64::__m256i;
    /// Widen 8 pixels to the `i32x8` lanes both carriers produce.
    #[cfg(target_arch = "x86_64")]
    const LD8: fn(
        archmage::X64V3Token,
        &[Self; 8],
    ) -> archmage::intrinsics::x86_64::__m256i;
    /// Widen 8 pixels to the `s16x8` lanes both carriers produce.
    #[cfg(target_arch = "aarch64")]
    const LD8_S16: fn(
        archmage::NeonToken,
        &[Self; 8],
    ) -> archmage::intrinsics::aarch64::int16x8_t;
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn lr_ld16_u8(_t: archmage::X64V3Token, c: &[u8; 16]) -> archmage::intrinsics::x86_64::__m256i {
    use archmage::intrinsics::x86_64::*;
    _mm256_cvtepu8_epi16(_mm_loadu_si128(c))
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn lr_ld8_u8(_t: archmage::X64V3Token, c: &[u8; 8]) -> archmage::intrinsics::x86_64::__m256i {
    use archmage::intrinsics::x86_64::*;
    _mm256_cvtepu16_epi32(_mm_cvtepu8_epi16(_mm_loadu_si64(c)))
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn lr_ld16_u16(_t: archmage::X64V3Token, c: &[u16; 16]) -> archmage::intrinsics::x86_64::__m256i {
    use archmage::intrinsics::x86_64::*;
    _mm256_loadu_si256(c)
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn lr_ld8_u16(_t: archmage::X64V3Token, c: &[u16; 8]) -> archmage::intrinsics::x86_64::__m256i {
    use archmage::intrinsics::x86_64::*;
    _mm256_cvtepu16_epi32(_mm_loadu_si128(c))
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn lr_ld8_s16_u8(
    _t: archmage::NeonToken,
    c: &[u8; 8],
) -> archmage::intrinsics::aarch64::int16x8_t {
    use archmage::intrinsics::aarch64::*;
    vreinterpretq_s16_u16(vmovl_u8(vld1_u8(c)))
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn lr_ld8_s16_u16(
    _t: archmage::NeonToken,
    c: &[u16; 8],
) -> archmage::intrinsics::aarch64::int16x8_t {
    use archmage::intrinsics::aarch64::*;
    vreinterpretq_s16_u16(vld1q_u16(c))
}

impl LrPixel for u8 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[cfg(target_arch = "x86_64")]
    const LD16: fn(
        archmage::X64V3Token,
        &[u8; 16],
    ) -> archmage::intrinsics::x86_64::__m256i = lr_ld16_u8;
    #[cfg(target_arch = "x86_64")]
    const LD8: fn(
        archmage::X64V3Token,
        &[u8; 8],
    ) -> archmage::intrinsics::x86_64::__m256i = lr_ld8_u8;
    #[cfg(target_arch = "aarch64")]
    const LD8_S16: fn(
        archmage::NeonToken,
        &[u8; 8],
    ) -> archmage::intrinsics::aarch64::int16x8_t = lr_ld8_s16_u8;
}

impl LrPixel for u16 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[cfg(target_arch = "x86_64")]
    const LD16: fn(
        archmage::X64V3Token,
        &[u16; 16],
    ) -> archmage::intrinsics::x86_64::__m256i = lr_ld16_u16;
    #[cfg(target_arch = "x86_64")]
    const LD8: fn(
        archmage::X64V3Token,
        &[u16; 8],
    ) -> archmage::intrinsics::x86_64::__m256i = lr_ld8_u16;
    #[cfg(target_arch = "aarch64")]
    const LD8_S16: fn(
        archmage::NeonToken,
        &[u16; 8],
    ) -> archmage::intrinsics::aarch64::int16x8_t = lr_ld8_s16_u16;
}

/// `find_average` (pickrst.h): the u8-truncating mean of the lowbd window.
/// `u16` values in u8 range; identical arithmetic.
fn find_average<P: LrPixel>(
    dgd: &[P],
    dgd_origin: usize,
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    stride: i32,
) -> u16 {
    let mut sum: u64 = 0;
    for i in v_start..v_end {
        for j in h_start..h_end {
            sum += dgd[dgd_origin + (i * stride + j) as usize].to_u64();
        }
    }
    (sum / (((v_end - v_start) * (h_end - h_start)) as u64)) as u16
}

/// `acc_stat_one_line` (pickrst.c): one source row's contribution to the
/// int32 row accumulators (`count` = the dgd row this line is centred on).
///
/// # Why this has a SIMD tier
///
/// This is the inner loop of the Wiener stats, and it was the largest
/// scalar-only function in the encoder. Measured on x86-64
/// (`benchmarks/encoder_x86_profile_2026-09-08.md`): `compute_stats` is
/// **32.2 ms of a 485 ms speed-0 encode against libaom's 2.4 ms** — a 13.4x
/// gap, and 160.25 M multiply-accumulates per encode retiring at **1.06 per
/// cycle**, i.e. exactly scalar-bound. libaom has `compute_stats_win7_avx2`;
/// the port had nothing.
///
/// # Why the SIMD tier is BIT-EXACT by construction, not merely by luck
///
/// Every accumulator element receives *the same products in the same order* as
/// the scalar tier — the vectorization is across elements `l` of one `k` row of
/// `H` (and across `k` for `M`), never across the pixel loop `j`, so no
/// reassociation happens at all. Integer products, integer adds, no saturation.
/// The scalar tier below is retained verbatim as the reference, and
/// `tests/pick_diff.rs` compares BOTH tiers against the real exported C.
///
/// The accumulator width is C's own: `|y|, |x| <= 255` at bd8, so a product is
/// at most 65025 and a row of at most 256 pixels accumulates below 16.7 M —
/// well inside `i32`, which is why C uses `int32_t` here too.
#[allow(clippy::too_many_arguments)]
fn acc_stat_one_line<P: LrPixel>(
    dgd: &[P],
    dgd_origin: usize,
    src_row: &[P],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    hstride: usize,
    count: i32,
) {
    archmage::incant!(
        acc_stat_line_impl(
            dgd,
            dgd_origin,
            src_row,
            dgd_stride,
            h_start,
            h_end,
            avg,
            wiener_halfwin,
            wiener_win2,
            m_row,
            h_row,
            hstride,
            count
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Gather the `wiener_win2` window values around source column `j`, about the
/// dgd mean, in C's index order (`idx = (k + halfwin) * win + (l + halfwin)`,
/// i.e. column-major over the window).
///
/// Shared verbatim by both tiers so the two cannot drift in the one place a
/// drift would be invisible to a lane-level review.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gather_window<P: LrPixel>(
    dgd: &[P],
    dgd_origin: usize,
    dgd_stride: i32,
    avg: u16,
    wiener_halfwin: i32,
    j: i32,
    count: i32,
    y: &mut [i32; WIENER_H_STRIDE],
) -> usize {
    let mut idx = 0usize;
    for k in -wiener_halfwin..=wiener_halfwin {
        for l in -wiener_halfwin..=wiener_halfwin {
            // Window reads may go up to ±3 outside the rect — negative
            // plane coords land in the extended border BEFORE the
            // origin (C pointer semantics).
            let off = dgd_origin as isize + ((count + l) * dgd_stride + (j + k)) as isize;
            y[idx] = i32::from(dgd[off as usize].to_i16() - avg as i16);
            idx += 1;
        }
    }
    idx
}

/// Gather the windows of FOUR adjacent source columns `j .. j + 4` at once,
/// one array per column.
///
/// Two things come out of this shape, and only the second one is the point:
///
/// * the four windows overlap in all but three columns, so the quad spans
///   `win + 3` columns rather than `4 * win` — 70 plane loads at win7 against
///   196;
/// * it lets [`acc_stat_line_impl`] fold all four pixels' contributions to one
///   `H` element before touching `H`, which is what the per-element
///   read-modify-write measurement below asks for.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
// x86-64's v3 tier inlines its own const-generic copy of this gather.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
fn gather_window_quad<P: LrPixel>(
    dgd: &[P],
    dgd_origin: usize,
    dgd_stride: i32,
    avg: u16,
    wiener_halfwin: i32,
    j: i32,
    count: i32,
    y: &mut [[i32; WIENER_H_STRIDE]; 4],
) -> usize {
    let win = (2 * wiener_halfwin + 1) as usize;
    // One entry per (column of the strip, row of the window). Column `c` is
    // plane column `j - halfwin + c`; pixel `j + p` uses columns `p..p + win`.
    let mut col = [[0i32; WIENER_WIN]; WIENER_WIN + 3];
    for (c, cw) in col.iter_mut().enumerate().take(win + 3) {
        let x = j - wiener_halfwin + c as i32;
        for (l, v) in cw.iter_mut().enumerate().take(win) {
            // Same ±halfwin border reach as `gather_window`.
            let off = dgd_origin as isize
                + ((count - wiener_halfwin + l as i32) * dgd_stride + x) as isize;
            *v = i32::from(dgd[off as usize].to_i16() - avg as i16);
        }
    }
    let mut idx = 0usize;
    for k in 0..win {
        for l in 0..win {
            y[0][idx] = col[k][l];
            y[1][idx] = col[k + 1][l];
            y[2][idx] = col[k + 2][l];
            y[3][idx] = col[k + 3][l];
            idx += 1;
        }
    }
    idx
}

/// Scalar tier = the transcribed port, verbatim.
#[allow(clippy::too_many_arguments)]
fn acc_stat_line_impl_scalar<P: LrPixel>(
    _t: archmage::ScalarToken,
    dgd: &[P],
    dgd_origin: usize,
    src_row: &[P],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    hstride: usize,
    count: i32,
) {
    acc_stat_line_recipe(
        dgd,
        dgd_origin,
        src_row,
        dgd_stride,
        h_start,
        h_end,
        avg,
        wiener_halfwin,
        wiener_win2,
        m_row,
        h_row,
        hstride,
        count,
    );
}

/// The scalar tier's body, token-free so the x86 v3 tier can delegate on
/// inputs its window-size specialization does not cover (identical behaviour,
/// identical panics).
#[allow(clippy::too_many_arguments)]
fn acc_stat_line_recipe<P: LrPixel>(
    dgd: &[P],
    dgd_origin: usize,
    src_row: &[P],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    hstride: usize,
    count: i32,
) {
    let mut y = [0i32; WIENER_H_STRIDE];
    for j in h_start..h_end {
        let x = i32::from(src_row[j as usize].to_i16() - avg as i16);
        let idx = gather_window(dgd, dgd_origin, dgd_stride, avg, wiener_halfwin, j, count, &mut y);
        debug_assert_eq!(idx, wiener_win2);
        for k in 0..wiener_win2 {
            m_row[k] += y[k] * x;
            for l in k..wiener_win2 {
                // H is symmetric; fill the upper triangle here (copied down
                // outside the pixel loops).
                h_row[k * hstride + l] += y[k] * y[l];
            }
        }
    }
}

/// Vector tier: lanes are elements of one `H` row (and of `M`), so each
/// accumulator still sees the scalar tier's exact product sequence.
///
/// # What this achieves, and what still limits it — measured, not projected
///
/// On the profile cell (192x192 cq27 speed 0, 165,888 stat pixels,
/// 160.25 M multiply-accumulates per encode):
///
/// | build | `compute_stats` | madds/s | madds/cycle |
/// |---|---:|---:|---:|
/// | scalar (before) | 32.2 ms | 4.98 G | 1.06 |
/// | this tier | **17.5 ms** | 9.16 G | 1.95 |
/// | libaom `compute_stats_win7_avx2` | 2.4 ms | 66 G | ~14 |
///
/// **1.84x, and 8-wide lanes did NOT buy 8x.** Two hypotheses were tested and
/// both are refuted, so do not re-spend them:
///
/// * *"the scalar tails dominate"* — they were ~171 madds per pixel against 132
///   vector iterations, which looks decisive. Removing them entirely (the
///   `WIENER_H_STRIDE` padding) was worth 0.9 ms of 18.4.
/// * *"it is L1 bandwidth on the `H` accumulator"* — the read-modify-write
///   streams ~1.25 GB per encode, which is **71 GB/s**, about 16 % of this
///   core's L1 ceiling. Not the limit.
///
/// What was left is the shape of the loop: ~2.8 cycles per vector iteration for
/// a load + load + multiply + add + store + loop, i.e. the per-element
/// read-modify-write of `H` itself. That diagnosis predicted the next step —
/// **not wider lanes but folding several pixels before touching `H`** — and
/// that step is what this function now does.
///
/// # The four-pixel fold (2026-09-08), and why it is the SECOND thing tried
///
/// libaom's `acc_stat_win7_one_line_avx2` folds pairs of pixels with
/// `_mm256_madd_epi16`, so `H` is touched once per PAIR. **That exact
/// instruction is not available here**: the workspace pins magetypes 0.9.28,
/// whose `i16x16` has no `madd_adjacent` (it lands in 0.9.29), and this crate
/// is `#![forbid(unsafe_code)]` so the intrinsic cannot be reached by hand.
/// Do not re-derive that — check the lockfile before assuming an op exists.
///
/// It does not matter, because the 16-bit multiply was never the limit. The
/// measurement above says the limit is the `H` read-modify-write, and four
/// pixels folded in `i32` cut those by 4x while leaving the products alone:
/// per 8 `H` elements the loop went from 4 x (load y, mul, load H, add, store H)
/// to (4 x load y, 4 x mul, 3 add, load H, add, store H) — 20 ops to 13, and
/// one quarter of the `H` traffic.
///
/// MEASURED (two binaries from one tree, arms interleaved and ROTATED, a
/// same-binary null arm in every band; both arms byte-identical output):
///
/// | cell | before | after | vs libaom | paired | rounds | p | null |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | 192x192 cq27 s0 | 453.60 ms | 446.04 ms | 2.5412x -> 2.4988x | -1.68 % | 20/20 | 2e-6 | +0.21 % |
/// | 1024x1024 cq27 s0 | 10810.31 ms | 10600.42 ms | 2.6227x -> 2.5717x | -2.00 % | 8/8 | 0.008 | -0.04 % |
///
/// ATTRIBUTED, not inferred from the wall (the KB-PERF-6 roll-up error was
/// exactly that mistake): on the 1024x1024 cell this symbol is **4.01 % ->
/// 1.90 %** of the profile, i.e. **433 ms -> 201 ms**, against a wall delta of
/// -210 ms. Every other `restore::` symbol is unmoved (`calculate_intermediate`
/// 2.41 -> 2.45 %, `pixel_proj_error` 1.99 -> 1.86 %, `wiener` 1.73 -> 1.74 %).
/// So the kernel is **2.2x faster** and its ratio to
/// `compute_stats_win7_avx2` + win5 + `_c` (54.4 ms) goes **8.2x -> ~3.7x**.
/// It is no longer the largest loop-restoration symbol — `calculate_intermediate`
/// is, and its own hot loop is the 256-entry `X_BY_XPLUS1` gather that this
/// vector vocabulary has no instruction for (verified: magetypes 0.9.28 and
/// 0.9.29 both contain no `gather` at all).
///
/// **The lever is BIGGER at real image size**, which is why both cells are
/// quoted: at 192x192 the bd8 u16 planes are ~108 KiB and sit in L2, so a
/// traffic-reducing lever is under-measured there. Quote the ratio with its
/// cell.
///
/// Still not done, and named rather than implied: libaom folds pixels in
/// **16-bit** lanes on top of the fold, which is a further 2x on the multiply
/// side and needs either a magetypes bump to >= 0.9.29 (for `madd_adjacent`) or
/// an equivalent widening-multiply-add. `compute_stats_highbd` is untouched.
///
/// 2026-09-15: the x86-64 v3 tier moved out of this body into
/// [`acc_stat_line_v3_x86`], the same four-pixel fold monomorphized on the
/// window size (`WIN` is only ever 5 or 7) so every loop bound and accumulator
/// index is a compile-time constant. MEASURED on the 196x196 cq27 speed-0
/// profile cell, identical call counts both sides: **92,556 -> 53,247 Ir per
/// call (-42 %)**; the kernel's gap to libaom's `compute_stats_win5/7_avx2`
/// per restoration unit goes **4.5x -> 2.57x**. This body remains the neon and
/// wasm128 tier.
#[archmage::magetypes(define(i32x8), neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn acc_stat_line_impl<P: LrPixel>(
    token: Token,
    dgd: &[P],
    dgd_origin: usize,
    src_row: &[P],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    hstride: usize,
    count: i32,
) {
    // One bounds check + one 32-byte move per load/store, the shape
    // `restore::wiener` already uses; the lane VALUES are the scalar tier's.
    macro_rules! ld {
        ($s:expr, $o:expr) => {
            i32x8::from_slice(token, &$s[$o..$o + 8])
        };
    }
    macro_rules! st {
        ($s:expr, $o:expr, $v:expr) => {{
            let d: &mut [i32; 8] = (&mut $s[$o..$o + 8]).try_into().unwrap();
            ($v).store(d);
        }};
    }

    debug_assert!(hstride >= wiener_win2 && hstride % 8 == 0);
    // Each `y` is zero-padded to a whole vector and NEVER written past
    // `win2`, so the padding lanes contribute `0 * anything` — see
    // `WIENER_H_STRIDE`.
    let mut y = [[0i32; WIENER_H_STRIDE]; 4];

    // FOUR source columns at a time, folded before `H` is touched.
    //
    // Bit-exact by construction, and for a WEAKER reason than the per-element
    // form it replaces: that one reassociated nothing at all, this one
    // reassociates the four pixels' contributions to a given accumulator
    // element (`h += p0; h += p1; h += p2; h += p3` becomes
    // `h += ((p0 + p1) + p2) + p3`). Integer addition is associative — over
    // `i32` it is associative even on overflow, since wrapping addition is a
    // group operation — so every accumulator ends at the same value. The
    // products themselves are unchanged, and the width note above still bounds
    // them well inside `i32`.
    let mut j = h_start;
    while j + 3 < h_end {
        let idx = gather_window_quad(
            dgd,
            dgd_origin,
            dgd_stride,
            avg,
            wiener_halfwin,
            j,
            count,
            &mut y,
        );
        debug_assert_eq!(idx, wiener_win2);
        let xv = [0usize, 1, 2, 3].map(|p| {
            i32x8::splat(token, i32::from(src_row[j as usize + p].to_i16() - avg as i16))
        });

        // M: lanes are k. Runs off the end into the padding, which stays zero.
        let mut k = 0usize;
        while k < wiener_win2 {
            let acc = ld!(m_row, k)
                + ld!(y[0], k) * xv[0]
                + ld!(y[1], k) * xv[1]
                + ld!(y[2], k) * xv[2]
                + ld!(y[3], k) * xv[3];
            st!(m_row, k, acc);
            k += 8;
        }

        // H upper triangle: lanes are l, one broadcast `y[p][k]` per pixel per
        // row, and ONE read-modify-write of `H` per four pixels. That last
        // part is the lever — see the function's doc comment. No scalar tail:
        // the padded stride guarantees a whole vector starting at any
        // `l < win2` stays inside row `k`.
        //
        // MEASURED AND REJECTED (still true of this loop nest): starting the
        // sweep at `k & !7` so both slices are whole 8-lane chunks and
        // `chunks_exact` drops the per-iteration bounds check (this crate is
        // `#![forbid(unsafe_code)]`, so a check can only be removed
        // structurally). It is CORRECT — the extra lanes at `l < k` land in
        // row `k`'s lower triangle, which is zeroed per source row and never
        // read — and it is SLOWER: at win7 it costs 217 vector iterations per
        // pixel against 175, +24 %. Do not re-try it without that arithmetic.
        for k in 0..wiener_win2 {
            let k0 = i32x8::splat(token, y[0][k]);
            let k1 = i32x8::splat(token, y[1][k]);
            let k2 = i32x8::splat(token, y[2][k]);
            let k3 = i32x8::splat(token, y[3][k]);
            let base = k * hstride;
            let mut l = k;
            while l < wiener_win2 {
                let acc = ld!(h_row, base + l)
                    + ld!(y[0], l) * k0
                    + ld!(y[1], l) * k1
                    + ld!(y[2], l) * k2
                    + ld!(y[3], l) * k3;
                st!(h_row, base + l, acc);
                l += 8;
            }
        }
        j += 4;
    }

    // Column tail: up to three pixels, one at a time, the original shape.
    while j < h_end {
        let x = i32::from(src_row[j as usize].to_i16() - avg as i16);
        let idx = gather_window(
            dgd,
            dgd_origin,
            dgd_stride,
            avg,
            wiener_halfwin,
            j,
            count,
            &mut y[0],
        );
        debug_assert_eq!(idx, wiener_win2);

        let xv = i32x8::splat(token, x);
        let mut k = 0usize;
        while k < wiener_win2 {
            let acc = ld!(m_row, k) + ld!(y[0], k) * xv;
            st!(m_row, k, acc);
            k += 8;
        }
        for k in 0..wiener_win2 {
            let yk = i32x8::splat(token, y[0][k]);
            let base = k * hstride;
            let mut l = k;
            while l < wiener_win2 {
                let acc = ld!(h_row, base + l) + ld!(y[0], l) * yk;
                st!(h_row, base + l, acc);
                l += 8;
            }
        }
        j += 1;
    }
}

/// AVX2 tier (x86-64): the same four-pixel fold, but monomorphized on the
/// window size. `wiener_halfwin` is only ever 2 (win5) or 3 (win7), so the
/// generic tier's runtime `win2`/`hstride` bounds kept every accumulator
/// access bounds-checked; with `WIN` a literal, every index is statically
/// provable and the checks elide. Anything outside {5, 7} — including any
/// window a future caller could construct — delegates to the scalar recipe,
/// whose observable behaviour (including panics) is unchanged.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn acc_stat_line_impl_v3<P: LrPixel>(
    t: archmage::X64V3Token,
    dgd: &[P],
    dgd_origin: usize,
    src_row: &[P],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    hstride: usize,
    count: i32,
) {
    match wiener_halfwin {
        2 => acc_stat_line_v3_x86::<5, P>(
            t, dgd, dgd_origin, src_row, dgd_stride, h_start, h_end, avg, wiener_halfwin,
            wiener_win2, m_row, h_row, hstride, count,
        ),
        3 => acc_stat_line_v3_x86::<7, P>(
            t, dgd, dgd_origin, src_row, dgd_stride, h_start, h_end, avg, wiener_halfwin,
            wiener_win2, m_row, h_row, hstride, count,
        ),
        _ => acc_stat_line_recipe(
            dgd,
            dgd_origin,
            src_row,
            dgd_stride,
            h_start,
            h_end,
            avg,
            wiener_halfwin,
            wiener_win2,
            m_row,
            h_row,
            hstride,
            count,
        ),
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn acc_stat_line_v3_x86<const WIN: usize, P: LrPixel>(
    _t: archmage::X64V3Token,
    dgd: &[P],
    dgd_origin: usize,
    src_row: &[P],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    hstride: usize,
    count: i32,
) {
    use archmage::intrinsics::x86_64::*;

    // Loads/stores through a fixed-size referent: once the element range is
    // statically provable against the array length, no bounds check survives.
    // Macros rather than nested fns so the expansion stays inside this fn's
    // `#[target_feature]` context.
    macro_rules! ld8 {
        ($a:expr, $i:expr) => {{
            let lane: &[i32; 8] = $a[$i..$i + 8].try_into().unwrap();
            _mm256_loadu_si256(lane)
        }};
    }
    macro_rules! st8 {
        ($a:expr, $i:expr, $v:expr) => {{
            let v = $v;
            let lane: &mut [i32; 8] = (&mut $a[$i..$i + 8]).try_into().unwrap();
            _mm256_storeu_si256(lane, v)
        }};
    }

    let win2 = WIN * WIN;
    let hstride_v = win2.div_ceil(8) * 8;
    let halfwin = WIN as i32 / 2;
    // Preflight: any accumulator or layout that does not match this
    // specialization's static shape takes the scalar recipe, which panics on
    // genuinely-short buffers identically to before.
    if wiener_halfwin != halfwin
        || wiener_win2 != win2
        || hstride != hstride_v
        || m_row.len() < WIENER_H_STRIDE
        || h_row.len() < WIENER_H_ROW_LEN
    {
        return acc_stat_line_recipe(
            dgd,
            dgd_origin,
            src_row,
            dgd_stride,
            h_start,
            h_end,
            avg,
            wiener_halfwin,
            wiener_win2,
            m_row,
            h_row,
            hstride,
            count,
        );
    }
    let m: &mut [i32; WIENER_H_STRIDE] =
        (&mut m_row[..WIENER_H_STRIDE]).try_into().unwrap();
    let h: &mut [i32; WIENER_H_ROW_LEN] =
        (&mut h_row[..WIENER_H_ROW_LEN]).try_into().unwrap();

    // `y01`/`y23` hold the four windows as PACKED i16 pairs in i32 lanes:
    // `y01[k] = y0[k] | y1[k]<<16`. A `vpmaddwd` against a broadcast pair then
    // produces `y0[a]*y0[b] + y1[a]*y1[b]` per lane — C's 16-bit-lane fold —
    // and the second pair folds into the same read-modify-write. Values stay
    // in [-255,255] (dgd - avg at bd8), products pair-sum to <= 130050 in i32,
    // and the accumulation order over `h`/`m` is the same wrapping-i32 group
    // as the four-term form it replaces, so the result is identical.
    // Padding pairs are (0,0) — `0 * anything` — see `WIENER_H_STRIDE`.
    let mut y01 = [0i32; WIENER_H_STRIDE];
    let mut y23 = [0i32; WIENER_H_STRIDE];
    let mut y_tail = [0i32; WIENER_H_STRIDE];
    let mut j = h_start;
    while j + 3 < h_end {
        // `gather_window_quad` with static bounds: one checked row slice per
        // window row instead of a per-element index, and every `col`/`y`
        // index is a literal-bounded value against a fixed-size array.
        let mut col = [[0i32; WIENER_WIN]; WIENER_WIN + 3];
        for l in 0..WIN {
            // Signed intermediate, exactly like `gather_window`: window rows
            // may address into the extended border before the origin, and the
            // negative partial sums must not wrap before `dgd_origin` is added.
            let lo = (dgd_origin as isize
                + (((count - halfwin + l as i32) * dgd_stride + (j - halfwin)) as isize))
                as usize;
            let wrow = &dgd[lo..lo + WIN + 3];
            for c in 0..WIN + 3 {
                col[c][l] = i32::from(wrow[c].to_i16() - avg as i16);
            }
        }
        let mut idx = 0usize;
        for k in 0..WIN {
            for l in 0..WIN {
                y01[idx] = (col[k][l] as u16 as i32) | (col[k + 1][l] << 16);
                y23[idx] = (col[k + 2][l] as u16 as i32) | (col[k + 3][l] << 16);
                idx += 1;
            }
        }
        let s4: &[P; 4] = src_row[j as usize..j as usize + 4].try_into().unwrap();
        let x01 = _mm256_set1_epi32(
            (i32::from(s4[0].to_i16() - avg as i16) as u16 as i32)
                | (i32::from(s4[1].to_i16() - avg as i16) << 16),
        );
        let x23 = _mm256_set1_epi32(
            (i32::from(s4[2].to_i16() - avg as i16) as u16 as i32)
                | (i32::from(s4[3].to_i16() - avg as i16) << 16),
        );

        // M: lanes are k, one madd per pixel pair. Runs off the end into the
        // padding, which stays zero.
        let mut k = 0usize;
        while k < win2 {
            let acc = _mm256_add_epi32(
                ld8!(m, k),
                _mm256_add_epi32(
                    _mm256_madd_epi16(ld8!(&y01, k), x01),
                    _mm256_madd_epi16(ld8!(&y23, k), x23),
                ),
            );
            st8!(m, k, acc);
            k += 8;
        }

        // H upper triangle: one read-modify-write of `H` per four pixels, two
        // madds per eight `l` lanes.
        for k in 0..win2 {
            let k01 = _mm256_set1_epi32(y01[k]);
            let k23 = _mm256_set1_epi32(y23[k]);
            let base = k * hstride_v;
            let mut l = k;
            while l < win2 {
                let acc = _mm256_add_epi32(
                    ld8!(h, base + l),
                    _mm256_add_epi32(
                        _mm256_madd_epi16(ld8!(&y01, l), k01),
                        _mm256_madd_epi16(ld8!(&y23, l), k23),
                    ),
                );
                st8!(h, base + l, acc);
                l += 8;
            }
        }
        j += 4;
    }

    // Column tail: up to three pixels, one at a time, the original shape.
    while j < h_end {
        let x = i32::from(src_row[j as usize].to_i16() - avg as i16);
        let idx = gather_window(
            dgd,
            dgd_origin,
            dgd_stride,
            avg,
            halfwin,
            j,
            count,
            &mut y_tail,
        );
        debug_assert_eq!(idx, win2);

        let xv = _mm256_set1_epi32(x);
        let mut k = 0usize;
        while k < win2 {
            st8!(
                m,
                k,
                _mm256_add_epi32(ld8!(m, k), _mm256_mullo_epi32(ld8!(&y_tail, k), xv))
            );
            k += 8;
        }
        for k in 0..win2 {
            let yk = _mm256_set1_epi32(y_tail[k]);
            let base = k * hstride_v;
            let mut l = k;
            while l < win2 {
                st8!(
                    h,
                    base + l,
                    _mm256_add_epi32(
                        ld8!(h, base + l),
                        _mm256_mullo_epi32(ld8!(&y_tail, l), yk),
                    )
                );
                l += 8;
            }
        }
        j += 1;
    }
}

/// `av1_compute_stats_c` (pickrst.c): the lowbd Wiener autocorrelation
/// vector `M[win2]` and matrix `H[win2 * win2]` of the window around each
/// source pixel in the `[h_start, h_end) x [v_start, v_end)` rect, about the
/// dgd mean, optionally with 4x vertical downsampling
/// (`lpf_sf.use_downsampled_wiener_stats`).
#[allow(clippy::too_many_arguments)]
pub fn compute_stats<P: LrPixel>(
    wiener_win: usize,
    dgd: &[P],
    dgd_origin: usize,
    src: &[P],
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
    let avg = find_average(dgd, dgd_origin, h_start, h_end, v_start, v_end, dgd_stride);
    let mut m_row = [0i32; WIENER_H_STRIDE];
    let mut h_row = [0i32; WIENER_H_ROW_LEN];
    let hstride = wiener_win2.div_ceil(8) * 8;
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
        m_row[..hstride].fill(0);
        h_row[..wiener_win2 * hstride].fill(0);
        acc_stat_one_line(
            dgd,
            dgd_origin,
            &src[(i * src_stride) as usize..],
            dgd_stride,
            h_start,
            h_end,
            avg,
            wiener_halfwin,
            wiener_win2,
            &mut m_row,
            &mut h_row,
            hstride,
            i,
        );
        for k in 0..wiener_win2 {
            // Scale by the downsampling factor (1 when not downsampling).
            m[k] += m_row[k] as i64 * downsample_factor as i64;
            for l in k..wiener_win2 {
                h[k * wiener_win2 + l] +=
                    h_row[k * hstride + l] as i64 * downsample_factor as i64;
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
    dgd_origin: usize,
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    stride: i32,
) -> u16 {
    // Identical to the lowbd form on u16 planes.
    find_average(dgd, dgd_origin, h_start, h_end, v_start, v_end, stride)
}

/// `av1_compute_stats_highbd_c` (pickrst.c): i64 accumulation with the
/// `bit_depth_divider` normalization (1 / 4 / 16 for bd 8 / 10 / 12).
#[allow(clippy::too_many_arguments)]
pub fn compute_stats_highbd(
    wiener_win: usize,
    dgd: &[u16],
    dgd_origin: usize,
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
    let avg = find_average_highbd(dgd, dgd_origin, h_start, h_end, v_start, v_end, dgd_stride);
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
                    let off = dgd_origin as isize + ((i + l) * dgd_stride + (j + k)) as isize;
                    y[idx] = dgd[off as usize] as i32 - avg as i32;
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
/// Scalar tier = the transcribed port, verbatim.
fn pixel_proj_error_scalar<P: LrPixel>(
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
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
                let d = dat[dat_off + i * dat_stride + j].to_i32();
                let s = src[src_off + i * src_stride + j].to_i32();
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
                let d = dat[dat_off + i * dat_stride + j].to_i32();
                let s = src[src_off + i * src_stride + j].to_i32();
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

/// SIMD-dispatched (Gate 3). `width >= 8` takes the magetypes `i32x8` kernel;
/// narrower rows and the tail keep the scalar tier.
///
/// # Why this has a SIMD tier
///
/// `benchmarks/encoder_x86_profile_2026-09-08.md`: the loop-restoration SEARCH
/// is **26 % of the speed-0 encode-time gap to libaom** and had never been
/// profiled (libaom disables Wiener + SGR at `speed >= 5`, and every earlier
/// profile in this repo was taken at `--cpu-used 6`, where the stage is
/// structurally absent). This kernel was **17.8 ms of a 471 ms encode against
/// libaom's `av1_lowbd_pixel_proj_error_avx2` at 2.0 ms**, with no SIMD tier.
///
/// # Bit-exactness, and why no magnitude bound is needed
///
/// The vector tier computes `e` in `i32` lanes — the same width, the same
/// operations and the same wrapping as the scalar tier — and then **squares and
/// accumulates in the scalar tier's own order**, `err += e as i64 * e as i64`
/// over `j` ascending. So the i64 accumulator sees an identical sequence of
/// identical products: this is bit-exact by construction, not within a bound.
///
/// That choice is deliberate. Squaring in `i32` lanes and reducing per chunk
/// would be faster, but it needs `8 * e^2 < 2^31`, i.e. `|e| < 16384`, and the
/// arithmetic puts `|e|` at roughly `2^14` at bd12 (`xq` reaches ~96 and
/// `flt - u` reaches ~2^16, so `v` reaches ~2^24 and `e` ~2^13..2^14) — at the
/// edge of the bound rather than comfortably inside it. `restore/pick.rs` feeds
/// RD decisions and therefore the byte gates, so an unproven bound is not worth
/// the milliseconds. If someone wants them, DERIVE the bound from
/// `SGRPROJ_PRJ_MIN0/MAX0` and the SGR output range first, and gate it at
/// runtime the way `intra/dir_simd.rs` gates its tap bound.
#[allow(clippy::too_many_arguments)]
pub fn pixel_proj_error<P: LrPixel>(
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
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
    if width < 8 {
        return pixel_proj_error_scalar(
            src, src_off, width, height, src_stride, dat, dat_off, dat_stride, flt0, flt0_stride,
            flt1, flt1_stride, xq, ep, highbd,
        );
    }
    let _ = crate::dispatch::scalar_forced();
    archmage::incant!(
        pixel_proj_error_impl(
            src, src_off, width, height, src_stride, dat, dat_off, dat_stride, flt0, flt0_stride,
            flt1, flt1_stride, xq, ep, highbd
        ),
        [v3, neon, wasm128, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
fn pixel_proj_error_impl_scalar<P: LrPixel>(
    _t: archmage::ScalarToken,
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
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
    pixel_proj_error_scalar(
        src, src_off, width, height, src_stride, dat, dat_off, dat_stride, flt0, flt0_stride, flt1,
        flt1_stride, xq, ep, highbd,
    )
}

/// x86-64/AVX2 body for [`pixel_proj_error`] — a transcription of C's
/// `av1_lowbd_pixel_proj_error_avx2` + `av1_highbd_pixel_proj_error_avx2`
/// (pickrst_avx2.c:1548,2133). Raw intrinsics because the wins are exactly the
/// pieces the magetypes vector API cannot express: i16 lanes (16 px/iter), the
/// `madd_epi16` pair tricks (`[xq0,xq1]` against interleaved `f1,f2`
/// differences computes `xq0*f1 + xq1*f2` in ONE instruction; the single-filter
/// cases fold `u = d<<4` into the coefficient as `-xq<<4` and never form `u`),
/// and a per-ROW i32 accumulator widened to i64 once per row.
///
/// The port's `dat`/`src` are `u16` planes even for lowbd, so where C does
/// `cvtepu8_epi16(loadu_128)` we load 16 `u16` lanes directly — the lane
/// values are identical (lowbd pixel data is <= 255 either way).
///
/// # Bit-exactness
///
/// Every vector op is the same instruction C issues, on the same values, so
/// the result is C's AVX2 result — including C's own wraparound semantics
/// (`packs_epi32` saturation on `flt`, `sub_epi16`/`add_epi16` mod-2^16 on the
/// `vr + d - s` assembly, i32 `sum32` accumulation within a row). On every
/// input the encoder actually produces those match the scalar tier: `|flt| <
/// 2^15` (C's own assert, pickrst.c:244-245) keeps the i16 packing exact, and
/// `|e| < 2^15` (`u < 2^16` at bd12, `|xq| <= 96`) keeps `e` and the `madd`
/// squares exact. Where C's AVX2 and its scalar diverge on inputs outside
/// those bounds, libaom's own SIMD consistency testing guarantees they don't
/// occur — and this kernel is bit-identical to whichever answer C gives.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn pixel_proj_error_impl_v3<P: LrPixel>(
    _t: archmage::X64V3Token,
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
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
    use archmage::intrinsics::x86_64::*;
    const SHIFT: i32 = SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
    let (rads, _) = SGR_PARAMS[ep];
    let (r0, r1) = (rads[0] > 0, rads[1] > 0);
    let rounding = _mm256_set1_epi32(1 << (SHIFT - 1));
    let mut sum64 = _mm256_setzero_si256();
    let mut err: i64 = 0;

    // Per-row `as_chunks` views + indexed loops: one bounds check per row,
    // none per 16-px step — every view's len is `width / 16` (the same
    // expression), so LLVM folds the `d16[k]`/`s16[k]`/... checks against the
    // loop bound (this crate is `#![forbid(unsafe)]`, so a check can only be
    // removed structurally). The `width % 16` remainder is the scalar `tail!`
    // below.
    macro_rules! rows16 {
        ($dr:expr, $sr:expr, $f0r:expr, $f1r:expr) => {
            (
                dat[$dr..$dr + width].as_chunks::<16>().0,
                src[$sr..$sr + width].as_chunks::<16>().0,
                flt0[$f0r..$f0r + width].as_chunks::<16>().0,
                flt1[$f1r..$f1r + width].as_chunks::<16>().0,
            )
        };
    }
    macro_rules! rows16_1f {
        ($fs:expr, $dr:expr, $sr:expr, $fr:expr) => {
            (
                dat[$dr..$dr + width].as_chunks::<16>().0,
                src[$sr..$sr + width].as_chunks::<16>().0,
                $fs[$fr..$fr + width].as_chunks::<16>().0,
            )
        };
    }
    // Split a 16-wide i32 chunk into its two 8-lane halves — constant ranges on
    // a known-length array, so the checks fold away.
    macro_rules! half8 {
        ($c:expr, lo) => {{
            let w: &[i32; 8] = $c[..8].try_into().unwrap();
            _mm256_loadu_si256(w)
        }};
        ($c:expr, hi) => {{
            let w: &[i32; 8] = $c[8..].try_into().unwrap();
            _mm256_loadu_si256(w)
        }};
    }
    // Truncating i32x8,i32x8 -> i16x16 in `vpackssdw` lane order
    // ([a0..3, b0..3 | a4..7, b4..7]): vpshufb gathers each dword's low word
    // per 128-lane, vpunpcklqdq pairs them — 3 insns. `_mm256_packs_epi32` in
    // core_arch is spelled `imax(imin())`+shuffle, and LLVM keeps the (dead)
    // clamps next to the `vpackssdw` it selects — 5 insns. Bit-identical to
    // `packs` wherever the saturating pack never fires, which is this kernel's
    // whole documented domain: `|flt| < 2^15` (C's own assert) and
    // `|vr| < 2^14` (`|xq| <= 96`, `|f - u| < 2^17` -> `|v| < 2^25`).
    let tpack_mask = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    macro_rules! tpack {
        ($a:expr, $b:expr) => {{
            let at = _mm256_shuffle_epi8($a, tpack_mask);
            let bt = _mm256_shuffle_epi8($b, tpack_mask);
            _mm256_unpacklo_epi64(at, bt)
        }};
    }
    // The scalar tail (`for k = j; k < width; k++` in C) — identical body to
    // the scalar tier's per-pixel expression.
    macro_rules! tail {
        ($i:expr, $j:expr) => {{
            let i: usize = $i;
            let dr = dat_off + i * dat_stride;
            let sr = src_off + i * src_stride;
            let f0r = i * flt0_stride;
            let f1r = i * flt1_stride;
            for j in $j..width {
                let d = dat[dr + j].to_i32();
                let s = src[sr + j].to_i32();
                let e = if r0 || r1 {
                    let u = d << SGRPROJ_RST_BITS;
                    let mut v = if highbd {
                        1 << (SHIFT - 1)
                    } else {
                        u << SGRPROJ_PRJ_BITS
                    };
                    if r0 {
                        v += xq[0] * (flt0[f0r + j] - u);
                    }
                    if r1 {
                        v += xq[1] * (flt1[f1r + j] - u);
                    }
                    if highbd {
                        (v >> SHIFT) + d - s
                    } else {
                        ((v + (1 << (SHIFT - 1))) >> SHIFT) - s
                    }
                } else {
                    d - s
                };
                err += e as i64 * e as i64;
            }
        }};
    }
    // Per-row widen of the i32-lane accumulator, C's own choice per bd:
    // sign-extend for lowbd, zero-extend for highbd.
    macro_rules! widen_row {
        ($sum32:expr, signed) => {
            sum64 = _mm256_add_epi64(
                sum64,
                _mm256_add_epi64(
                    _mm256_cvtepi32_epi64(_mm256_castsi256_si128($sum32)),
                    _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>($sum32)),
                ),
            )
        };
        ($sum32:expr, unsigned) => {
            sum64 = _mm256_add_epi64(
                sum64,
                _mm256_add_epi64(
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128($sum32)),
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>($sum32)),
                ),
            )
        };
    }

    if r0 && r1 {
        if highbd {
            let xq0 = _mm256_set1_epi32(xq[0]);
            let xq1 = _mm256_set1_epi32(xq[1]);
            for i in 0..height {
                let dr = dat_off + i * dat_stride;
                let sr = src_off + i * src_stride;
                let f0r = i * flt0_stride;
                let f1r = i * flt1_stride;
                let mut sum32 = _mm256_setzero_si256();
                let (d16, s16, f016, f116) = rows16!(dr, sr, f0r, f1r);
                for k in 0..d16.len() {
                    let (dc, sc, f0c, f1c) = (&d16[k], &s16[k], &f016[k], &f116[k]);
                    let s0 = P::LD16(_t, sc);
                    let d0 = P::LD16(_t, dc);
                    let u0 = _mm256_slli_epi16::<{ SGRPROJ_RST_BITS }>(d0);
                    let u0l = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(u0));
                    let u0h =
                        _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(u0));
                    let f0l = _mm256_sub_epi32(half8!(f0c, lo), u0l);
                    let f0h = _mm256_sub_epi32(half8!(f0c, hi), u0h);
                    let f1l = _mm256_sub_epi32(half8!(f1c, lo), u0l);
                    let f1h = _mm256_sub_epi32(half8!(f1c, hi), u0h);
                    let vl = _mm256_add_epi32(
                        _mm256_mullo_epi32(f0l, xq0),
                        _mm256_mullo_epi32(f1l, xq1),
                    );
                    let vh = _mm256_add_epi32(
                        _mm256_mullo_epi32(f0h, xq0),
                        _mm256_mullo_epi32(f1h, xq1),
                    );
                    let vrl = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(vl, rounding));
                    let vrh = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(vh, rounding));
                    // `packs` order (no post-permute — that merges into the
                    // clamp+shuffle expansion); d0/s0 permuted to match.
                    let d0p = _mm256_permute4x64_epi64::<0xd8>(d0);
                    let s0p = _mm256_permute4x64_epi64::<0xd8>(s0);
                    let vr = tpack!(vrl, vrh);
                    let e0 =
                        _mm256_sub_epi16(_mm256_add_epi16(vr, d0p), s0p);
                    sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(e0, e0));
                }
                tail!(i, width & !15);
                widen_row!(sum32, unsigned);
            }
        } else {
            // pair_set_epi16(xq[0], xq[1])
            let xq_coeff = _mm256_set1_epi32((xq[0] & 0xffff) | (xq[1] << 16));
            for i in 0..height {
                let dr = dat_off + i * dat_stride;
                let sr = src_off + i * src_stride;
                let f0r = i * flt0_stride;
                let f1r = i * flt1_stride;
                let mut sum32 = _mm256_setzero_si256();
                let (d16, s16, f016, f116) = rows16!(dr, sr, f0r, f1r);
                for k in 0..d16.len() {
                    let (dc, sc, f0c, f1c) = (&d16[k], &s16[k], &f016[k], &f116[k]);
                    // Permute d0/s0 into `packs` order ([px0..3, px8..11,
                    // px4..7, px12..15]) ONCE instead of permuting each flt
                    // pack into pixel order: `packs(f0lo,f0hi)` feeds a
                    // non-shuffle consumer (`sub_epi16`), so LLVM keeps the
                    // real `vpackssdw` — with `permute4x64(packs(..))` the two
                    // shuffles merge into a ~9-insn clamp+shuffle expansion
                    // per pack. The unpack pair restores i32-lane order (v0 =
                    // px 0..7, v1 = px 8..15), so `packs(vr0,vr1)` lands back
                    // in packs order — exactly what `d0p`/`s0p` hold. The
                    // final madd sum is lane-order-invariant anyway.
                    let d0 = P::LD16(_t, dc);
                    let s0 = P::LD16(_t, sc);
                    let d0p = _mm256_permute4x64_epi64::<0xd8>(d0);
                    let s0p = _mm256_permute4x64_epi64::<0xd8>(s0);
                    let u0p = _mm256_slli_epi16::<{ SGRPROJ_RST_BITS }>(d0p);
                    let f0sub = _mm256_sub_epi16(
                        tpack!(half8!(f0c, lo), half8!(f0c, hi)),
                        u0p,
                    );
                    let f1sub = _mm256_sub_epi16(
                        tpack!(half8!(f1c, lo), half8!(f1c, hi)),
                        u0p,
                    );
                    let v0 =
                        _mm256_madd_epi16(xq_coeff, _mm256_unpacklo_epi16(f0sub, f1sub));
                    let v1 =
                        _mm256_madd_epi16(xq_coeff, _mm256_unpackhi_epi16(f0sub, f1sub));
                    let vr0 = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(v0, rounding));
                    let vr1 = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(v1, rounding));
                    let e0 = _mm256_sub_epi16(
                        _mm256_add_epi16(tpack!(vr0, vr1), d0p),
                        s0p,
                    );
                    sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(e0, e0));
                }
                tail!(i, width & !15);
                widen_row!(sum32, signed);
            }
        }
    } else if r0 || r1 {
        let xq_on = if r0 { xq[0] } else { xq[1] };
        let (flt, flt_stride) = if r0 { (flt0, flt0_stride) } else { (flt1, flt1_stride) };
        if highbd {
            let xq_active = _mm256_set1_epi32(xq_on);
            let xq_inactive = _mm256_set1_epi32(-xq_on * (1 << SGRPROJ_RST_BITS));
            for i in 0..height {
                let dr = dat_off + i * dat_stride;
                let sr = src_off + i * src_stride;
                let fr = i * flt_stride;
                let mut sum32 = _mm256_setzero_si256();
                let (d16, s16, f16) = rows16_1f!(flt, dr, sr, fr);
                for k in 0..d16.len() {
                    let (dc, sc, fc) = (&d16[k], &s16[k], &f16[k]);
                    let s0 = P::LD16(_t, sc);
                    let d0 = P::LD16(_t, dc);
                    let d0l = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(d0));
                    let d0h =
                        _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(d0));
                    let vl = _mm256_add_epi32(
                        _mm256_mullo_epi32(half8!(fc, lo), xq_active),
                        _mm256_mullo_epi32(d0l, xq_inactive),
                    );
                    let vh = _mm256_add_epi32(
                        _mm256_mullo_epi32(half8!(fc, hi), xq_active),
                        _mm256_mullo_epi32(d0h, xq_inactive),
                    );
                    let vrl = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(vl, rounding));
                    let vrh = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(vh, rounding));
                    // `packs` order (no post-permute — that merges into the
                    // clamp+shuffle expansion); d0/s0 permuted to match.
                    let d0p = _mm256_permute4x64_epi64::<0xd8>(d0);
                    let s0p = _mm256_permute4x64_epi64::<0xd8>(s0);
                    let vr = tpack!(vrl, vrh);
                    let e0 =
                        _mm256_sub_epi16(_mm256_add_epi16(vr, d0p), s0p);
                    sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(e0, e0));
                }
                tail!(i, width & !15);
                widen_row!(sum32, unsigned);
            }
        } else {
            // pair_set_epi16(xq_active, -xq_active * (1 << SGRPROJ_RST_BITS)):
            // madd over interleaved (flt, d) pairs gives xq*(flt - u) without
            // forming u.
            let xi = (-xq_on * (1 << SGRPROJ_RST_BITS)) & 0xffff;
            let xq_coeff = _mm256_set1_epi32((xq_on & 0xffff) | ((xi as i32) << 16));
            for i in 0..height {
                let dr = dat_off + i * dat_stride;
                let sr = src_off + i * src_stride;
                let fr = i * flt_stride;
                let mut sum32 = _mm256_setzero_si256();
                let (d16, s16, f16) = rows16_1f!(flt, dr, sr, fr);
                for k in 0..d16.len() {
                    let (dc, sc, fc) = (&d16[k], &s16[k], &f16[k]);
                    // Same trick as the two-filter arm: keep the flt pack in
                    // `packs` order so it feeds `unpack` directly (no
                    // packs+permq merge into the ~9-insn clamp+shuffle
                    // expansion); permute d0/s0 into packs order instead.
                    // unpacklo/hi restore i32-lane pixel order, and the final
                    // `packs(vr0,vr1)` lands back in packs order to match
                    // d0p/s0p — the madd accumulation is order-invariant.
                    let d0 = P::LD16(_t, dc);
                    let s0 = P::LD16(_t, sc);
                    let d0p = _mm256_permute4x64_epi64::<0xd8>(d0);
                    let s0p = _mm256_permute4x64_epi64::<0xd8>(s0);
                    let flt_16b =
                        tpack!(half8!(fc, lo), half8!(fc, hi));
                    let v0 =
                        _mm256_madd_epi16(xq_coeff, _mm256_unpacklo_epi16(flt_16b, d0p));
                    let v1 =
                        _mm256_madd_epi16(xq_coeff, _mm256_unpackhi_epi16(flt_16b, d0p));
                    let vr0 = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(v0, rounding));
                    let vr1 = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(v1, rounding));
                    let e0 = _mm256_sub_epi16(
                        _mm256_add_epi16(tpack!(vr0, vr1), d0p),
                        s0p,
                    );
                    sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(e0, e0));
                }
                tail!(i, width & !15);
                widen_row!(sum32, signed);
            }
        }
    } else {
        // Neither filter: e = d - s, 16 px/iter (C's lowbd shape; the values
        // fit i16 at every bd, so one body serves both — C splits 32px/iter
        // for highbd, same arithmetic).
        for i in 0..height {
            let dr = dat_off + i * dat_stride;
            let sr = src_off + i * src_stride;
            let mut sum32 = _mm256_setzero_si256();
            let (d16, s16) = (
                dat[dr..dr + width].as_chunks::<16>().0,
                src[sr..sr + width].as_chunks::<16>().0,
            );
            for k in 0..d16.len() {
                let d0 = P::LD16(_t, &d16[k]);
                let s0 = P::LD16(_t, &s16[k]);
                let diff = _mm256_sub_epi16(d0, s0);
                sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(diff, diff));
            }
            tail!(i, width & !15);
            if highbd {
                widen_row!(sum32, unsigned);
            } else {
                widen_row!(sum32, signed);
            }
        }
    }
    let s128 = _mm_add_epi64(_mm256_castsi256_si128(sum64), _mm256_extracti128_si256::<1>(sum64));
    let s64 = _mm_add_epi64(s128, _mm_unpackhi_epi64(s128, s128));
    err + _mm_cvtsi128_si64(s64)
}

/// Vector tier for the non-AVX2 backends: lanes are adjacent `j`. See the
/// dispatcher's doc for why the squares stay scalar. The `v3` arm is the
/// hand-written [`pixel_proj_error_impl_v3`] above — `incant!` resolves the
/// tier to that name, which is why `v3` is absent from this list.
#[archmage::magetypes(define(i32x8), neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn pixel_proj_error_impl<P: LrPixel>(
    token: Token,
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
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
    let vw = width & !7;

    let widen = |s: &[P]| -> i32x8 {
        let a: [P; 8] = s[..8].try_into().unwrap();
        i32x8::from_array(
            token,
            [
                a[0].to_i32(), a[1].to_i32(), a[2].to_i32(), a[3].to_i32(),
                a[4].to_i32(), a[5].to_i32(), a[6].to_i32(), a[7].to_i32(),
            ],
        )
    };
    let ld32 = |s: &[i32]| -> i32x8 { i32x8::from_slice(token, &s[..8]) };

    let half = i32x8::splat(token, 1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1));
    let xq0 = i32x8::splat(token, xq[0]);
    let xq1 = i32x8::splat(token, xq[1]);

    for i in 0..height {
        let dr = dat_off + i * dat_stride;
        let sr = src_off + i * src_stride;
        let f0r = i * flt0_stride;
        let f1r = i * flt1_stride;
        let mut j = 0usize;
        while j < vw {
            let d = widen(&dat[dr + j..dr + j + 8]);
            let sv = widen(&src[sr + j..sr + j + 8]);
            let e = if r0 || r1 {
                let u = d.shl_const::<{ SGRPROJ_RST_BITS as i32 }>();
                // lowbd starts from `u << PRJ_BITS`; highbd starts from the
                // rounding constant and adds `d` after the shift. Both are the
                // scalar tier's own expressions, lane for lane.
                let mut v = if highbd { half } else { u.shl_const::<{ SGRPROJ_PRJ_BITS as i32 }>() };
                if r0 {
                    v = v + xq0 * (ld32(&flt0[f0r + j..]) - u);
                }
                if r1 {
                    v = v + xq1 * (ld32(&flt1[f1r + j..]) - u);
                }
                if highbd {
                    v.shr_arithmetic_const::<{ (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as i32 }>()
                        + d
                        - sv
                } else {
                    (v + half)
                        .shr_arithmetic_const::<{ (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as i32 }>()
                        - sv
                }
            } else {
                d - sv
            };
            // Squares in the scalar tier's exact order — see the dispatcher doc.
            for x in e.to_array() {
                err += x as i64 * x as i64;
            }
            j += 8;
        }
        while j < width {
            let d = dat[dr + j].to_i32();
            let s = src[sr + j].to_i32();
            let e = if r0 || r1 {
                let u = d << SGRPROJ_RST_BITS;
                let mut v = if highbd {
                    1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)
                } else {
                    u << SGRPROJ_PRJ_BITS
                };
                if r0 {
                    v += xq[0] * (flt0[f0r + j] - u);
                }
                if r1 {
                    v += xq[1] * (flt1[f1r + j] - u);
                }
                if highbd {
                    (v >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS)) + d - s
                } else {
                    ((v + (1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)))
                        >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS))
                        - s
                }
            } else {
                d - s
            };
            err += e as i64 * e as i64;
            j += 1;
        }
    }
    err
}

/// `av1_calc_proj_params_c` + `_high_bd_c` (pickrst.c): the least-squares
/// normal-equation accumulators `H` (2x2) and `C` (2), divided by the pixel
/// count. Identical arithmetic for lowbd/highbd on u16 planes (the C pair
/// differs only in pointer types).
#[allow(clippy::too_many_arguments)]
pub fn calc_proj_params<P: LrPixel>(
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    ep: usize,
) -> ([[i64; 2]; 2], [i64; 2]) {
    let _ = crate::dispatch::scalar_forced();
    if width < 8 {
        return calc_proj_params_impl_scalar(
            archmage::ScalarToken,
            src, src_off, width, height, src_stride, dat, dat_off, dat_stride, flt0, flt0_stride,
            flt1, flt1_stride, ep,
        );
    }
    archmage::incant!(
        calc_proj_params_impl(
            src, src_off, width, height, src_stride, dat, dat_off, dat_stride, flt0, flt0_stride,
            flt1, flt1_stride, ep
        ),
        [v3, scalar]
    )
}

/// x86-64/AVX2 body for [`calc_proj_params`] — raw intrinsics because the
/// magetypes vector API has no 64-bit lanes at all, and the products here are
/// genuinely 64-bit: `f = flt - u` reaches ~2^17, so `f1*f1` does NOT fit i32
/// (unlike `pixel_proj_error`'s `e`, whose bound makes `vpmulld` exact). C's
/// `av1_calc_proj_params_avx2` (pickrst_avx2.c) uses `_mm256_mul_epi32` —
/// signed 32x32 -> 64 full products on the even lanes, plus the same on the
/// `srli_epi64(_,32)`-shifted odd lanes — and that is what this does.
///
/// # Bit-exactness
///
/// `mul_epi32` sign-extends each lane's low i32 and produces the full i64
/// product — no input-range reasoning at all, exact for any i32 `f`/`s`. The
/// even/odd split plus the i64-lane accumulation reorder the scalar tier's
/// `hh += f1*f1` adds; integer addition is associative, so each accumulator is
/// bit-identical. `hh[0][1]`, `hh[1][0]` and the `/size` division are kept in
/// the scalar epilogue, matching C's unpack-and-store order.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn calc_proj_params_impl_v3<P: LrPixel>(
    _t: archmage::X64V3Token,
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    ep: usize,
) -> ([[i64; 2]; 2], [i64; 2]) {
    use archmage::intrinsics::x86_64::*;
    let (rads, _) = SGR_PARAMS[ep];
    let size = (width * height) as i64;
    let (r0, r1) = (rads[0] > 0, rads[1] > 0);
    let vw = width & !7;
    let mut h00 = _mm256_setzero_si256();
    let mut h01 = _mm256_setzero_si256();
    let mut h11 = _mm256_setzero_si256();
    let mut c0 = _mm256_setzero_si256();
    let mut c1 = _mm256_setzero_si256();
    let mut hh = [[0i64; 2]; 2];
    let mut cc = [0i64; 2];
    for i in 0..height {
        let dr = dat_off + i * dat_stride;
        let sr = src_off + i * src_stride;
        let f0r = i * flt0_stride;
        let f1r = i * flt1_stride;
        let mut j = 0usize;
        while j < vw {
            // Safe load wrappers take `&[T; N]` — the `j..j+8` windows are
            // inside the caller's row slices, so the length checks cannot fail.
            let dw: &[P; 8] = dat[dr + j..dr + j + 8].try_into().unwrap();
            let sw: &[P; 8] = src[sr + j..sr + j + 8].try_into().unwrap();
            let d = _mm256_slli_epi32::<{ SGRPROJ_RST_BITS }>(P::LD8(_t, dw));
            let s = _mm256_sub_epi32(
                _mm256_slli_epi32::<{ SGRPROJ_RST_BITS }>(P::LD8(_t, sw)),
                d,
            );
            // Signed low-i32 x low-i32 -> i64 per 64-bit lane, even lanes plus
            // the odd lanes shifted down — the scalar `f1 as i64 * f2 as i64`,
            // exactly.
            macro_rules! prod {
                ($a:expr, $b:expr) => {
                    _mm256_add_epi64(
                        _mm256_mul_epi32($a, $b),
                        _mm256_mul_epi32(
                            _mm256_srli_epi64::<32>($a),
                            _mm256_srli_epi64::<32>($b),
                        ),
                    )
                };
            }
            if r0 {
                let f: &[i32; 8] = flt0[f0r + j..f0r + j + 8].try_into().unwrap();
                let f1 = _mm256_sub_epi32(_mm256_loadu_si256(f), d);
                h00 = _mm256_add_epi64(h00, prod!(f1, f1));
                c0 = _mm256_add_epi64(c0, prod!(f1, s));
                if r1 {
                    let f: &[i32; 8] = flt1[f1r + j..f1r + j + 8].try_into().unwrap();
                    let f2 = _mm256_sub_epi32(_mm256_loadu_si256(f), d);
                    h11 = _mm256_add_epi64(h11, prod!(f2, f2));
                    h01 = _mm256_add_epi64(h01, prod!(f1, f2));
                    c1 = _mm256_add_epi64(c1, prod!(f2, s));
                }
            } else if r1 {
                let f: &[i32; 8] = flt1[f1r + j..f1r + j + 8].try_into().unwrap();
                let f2 = _mm256_sub_epi32(_mm256_loadu_si256(f), d);
                h11 = _mm256_add_epi64(h11, prod!(f2, f2));
                c1 = _mm256_add_epi64(c1, prod!(f2, s));
            }
            j += 8;
        }
        while j < width {
            let u = dat[dr + j].to_i32() << SGRPROJ_RST_BITS;
            let sv = (src[sr + j].to_i32() << SGRPROJ_RST_BITS) - u;
            if r0 && r1 {
                let f1 = flt0[f0r + j] - u;
                let f2 = flt1[f1r + j] - u;
                hh[0][0] += f1 as i64 * f1 as i64;
                hh[1][1] += f2 as i64 * f2 as i64;
                hh[0][1] += f1 as i64 * f2 as i64;
                cc[0] += f1 as i64 * sv as i64;
                cc[1] += f2 as i64 * sv as i64;
            } else if r0 {
                let f1 = flt0[f0r + j] - u;
                hh[0][0] += f1 as i64 * f1 as i64;
                cc[0] += f1 as i64 * sv as i64;
            } else if r1 {
                let f2 = flt1[f1r + j] - u;
                hh[1][1] += f2 as i64 * f2 as i64;
                cc[1] += f2 as i64 * sv as i64;
            }
            j += 1;
        }
    }
    let fold = |v: core::arch::x86_64::__m256i| -> i64 {
        let s128 = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s64 = _mm_add_epi64(s128, _mm_unpackhi_epi64(s128, s128));
        _mm_cvtsi128_si64(s64)
    };
    hh[0][0] += fold(h00);
    hh[0][1] += fold(h01);
    hh[1][1] += fold(h11);
    cc[0] += fold(c0);
    cc[1] += fold(c1);
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

/// Scalar tier (and non-x86 fallback) for [`calc_proj_params`] — the verbatim
/// C transcription.
#[allow(clippy::too_many_arguments)]
fn calc_proj_params_impl_scalar<P: LrPixel>(
    _t: archmage::ScalarToken,
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
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
            let u = dat[dat_off + i * dat_stride + j].to_i32() << SGRPROJ_RST_BITS;
            let s = (src[src_off + i * src_stride + j].to_i32() << SGRPROJ_RST_BITS) - u;
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
pub fn get_proj_subspace<P: LrPixel>(
    src: &[P],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[P],
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
    use crate::entropy::lr::{SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1, SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1};
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

// ---------------------------------------------------------------------------
// The per-unit RD search + frame-level decision
// (`restoration_search` / `av1_pick_filter_restoration`, pickrst.c).
// ---------------------------------------------------------------------------

use crate::restore::frame::{
    at, extend_frame, filter_unit, save_boundary_lines, StripeBoundaries, MARGIN_H, MARGIN_V,
};
use crate::restore::sgr::{decode_xq, selfguided_restoration};
use crate::entropy::lr::{
    count_sgrproj_bits, count_wiener_bits, lr_corners_in_sb, LrFrameConfig,
    LrUnitInfo, SgrprojInfoLr, WienerInfoLr, RESTORATION_PROC_UNIT_SIZE, RESTORATION_UNITSIZE_MAX,
    RESTORATION_UNIT_OFFSET, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
    SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1, SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1, WIENER_WIN_CHROMA,
};

/// `RESTORE_TYPES` / `RESTORE_SWITCHABLE_TYPES` (enums.h).
const RESTORE_TYPES: usize = 4;
const RESTORE_SWITCHABLE_TYPES: usize = 3;
/// `AV1_PROB_COST_SHIFT` (av1/encoder/cost.h).
const AV1_PROB_COST_SHIFT: i64 = 9;
/// `NUM_WIENER_ITERS` neighbours: search penalties (pickrst.c).
const DUAL_SGR_PENALTY_MULT: f64 = 0.01;
const WIENER_SGR_PENALTY_MULT: f64 = 0.005;
/// `RESTORATION_UNITPELS_MAX` (restoration.h): flt scratch sizing.
const RESTORATION_UNITPELS_MAX: usize =
    (RESTORATION_UNITSIZE_MAX as usize * 3 / 2 + 2 * 3 + 16)
        * (RESTORATION_UNITSIZE_MAX as usize * 3 / 2 + 2 * 3 + 8);

/// `sgproj_ep_grp1_seed` / `sgproj_ep_grp2_3` (pickrst.c): the pruned-ep
/// search ladder.
const SGRPROJ_EP_GRP1_START_IDX: i32 = 0;
const SGRPROJ_EP_GRP1_END_IDX: i32 = 9;
const SGRPROJ_EP_GRP1_SEED: [i32; 4] = [0, 3, 6, 9];
const SGRPROJ_EP_GRP2_3: [[i32; 14]; 2] = [
    [10, 10, 11, 11, 12, 12, 13, 13, 13, 13, -1, -1, -1, -1],
    [14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15],
];

/// `RDCOST_DBL_WITH_NATIVE_BD_DIST` (av1/encoder/rd.h).
#[inline]
fn rdcost_dbl_with_native_bd_dist(rdmult: i64, rate: i64, dist: i64, bd: i32) -> f64 {
    (rate as f64 * rdmult as f64) / ((1i64 << AV1_PROB_COST_SHIFT) as f64)
        + ((dist >> (2 * (bd - 8))) as f64) * ((1 << 7) as f64)
}

/// The `lpf_sf` slice `av1_pick_filter_restoration` consumes
/// (speed_features.h `LOOP_FILTER_SPEED_FEATURES`), plus the two frame
/// inputs the pruning heuristics need.
#[derive(Clone, Copy, Debug)]
pub struct LrSearchSf {
    pub disable_wiener_filter: bool,
    pub disable_sgr_filter: bool,
    pub disable_loop_restoration_luma: bool,
    pub disable_loop_restoration_chroma: bool,
    pub disable_wiener_coeff_refine_search: bool,
    /// 0 off; 1/2 = the `scale[]` ladder of the src-var prune.
    pub prune_wiener_based_on_src_var: i32,
    /// 0 off; 1 = rdcost-ratio gate; 2 = best-rtype gate.
    pub prune_sgr_based_on_wiener: i32,
    /// 0 full 16-ep; 1 = seeds+neighbours+groups; >=2 = seeds only.
    pub enable_sgr_ep_pruning: i32,
    pub reduce_wiener_window_size: bool,
    pub use_downsampled_wiener_stats: bool,
    pub dual_sgr_penalty_level: i32,
    pub switchable_lr_with_bias_level: i32,
    /// Luma-scale unit-size search bounds (`min/max_lr_unit_size`).
    pub min_lr_unit_size: i32,
    pub max_lr_unit_size: i32,
}

impl Default for LrSearchSf {
    /// Speed-0 defaults (speed_features.c framesize-independent tail +
    /// qindex-dependent size-search init).
    fn default() -> Self {
        LrSearchSf {
            disable_wiener_filter: false,
            disable_sgr_filter: false,
            disable_loop_restoration_luma: false,
            disable_loop_restoration_chroma: false,
            disable_wiener_coeff_refine_search: false,
            prune_wiener_based_on_src_var: 0,
            prune_sgr_based_on_wiener: 0,
            enable_sgr_ep_pruning: 0,
            reduce_wiener_window_size: false,
            use_downsampled_wiener_stats: false,
            dual_sgr_penalty_level: 0,
            switchable_lr_with_bias_level: 0,
            min_lr_unit_size: RESTORATION_PROC_UNIT_SIZE,
            max_lr_unit_size: RESTORATION_UNITSIZE_MAX,
        }
    }
}

/// One plane's pixels for the search: the ORIGINAL source, the deblocked
/// (pre-CDEF) recon and the current (post-CDEF) recon — `deblocked` and
/// `cur` may be the same content when CDEF did not run, matching the C
/// encoder's two `save_boundary_lines` passes.
pub struct LrPlanePixels<'a> {
    pub src: &'a [u16],
    pub deblocked: &'a [u16],
    pub cur: &'a [u16],
    pub stride: usize,
}

/// Frame-level inputs of `av1_pick_filter_restoration`.
pub struct LrSearchInput<'a> {
    pub planes: Vec<LrPlanePixels<'a>>,
    /// Luma crop dims (the RU grid domain; superres not in this envelope).
    pub crop_width: i32,
    pub crop_height: i32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bit_depth: i32,
    /// `seq_params->use_highbitdepth` routing: false = the lowbd (u8) C
    /// arithmetic (bd 8), true = the highbd paths.
    pub highbd: bool,
    /// `cpi->rd.RDMULT`.
    pub rdmult: i64,
    /// `av1_dc_quant_QTX(base_qindex, 0, bit_depth)` — only read when
    /// `prune_wiener_based_on_src_var > 0`.
    pub dc_quant_qtx: i32,
    /// Superblock geometry: `mib_size_log2` (4=sb64, 5=sb128) and the mi
    /// grid extent.
    pub mib_size_log2: i32,
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// Tile bounds in superblock units (`tiles.row_start_sb` pairs), raster
    /// iterated rows-outer. Single tile: `[(0, sb_rows)]` / `[(0, sb_cols)]`.
    pub tile_sb_rows: Vec<(i32, i32)>,
    pub tile_sb_cols: Vec<(i32, i32)>,
    /// `av1_fill_lr_rates` outputs (cost_tokens_from_cdf of the frame-init
    /// wiener/sgrproj/switchable restore CDFs).
    pub wiener_restore_cost: [i32; 2],
    pub sgrproj_restore_cost: [i32; 2],
    pub switchable_restore_cost: [i32; 3],
    /// Worker count for the per-tile search walk. `0`/`1` = the serial walk.
    /// The unit results are identical either way: `RscState`'s delta-coding
    /// references reset at every tile start (`rsc_on_tile`), so disjoint tile
    /// rows are independent and the per-type `total_bits`/`total_sse` sums
    /// merge commutatively.
    pub threads: usize,
    pub sf: LrSearchSf,
}

/// `av1_pick_filter_restoration`'s decision.
#[derive(Clone, Debug, Default)]
pub struct LrSearchOutcome {
    /// The chosen luma restoration unit size (all planes share it, `s = 0`).
    pub unit_size: i32,
    pub frame_restoration_type: [u8; 3],
    /// Per-plane unit params in unit-grid raster order for the chosen size
    /// (empty when that plane is `RESTORE_NONE`).
    pub units: [Vec<LrUnitInfo>; 3],
}

/// `RestUnitSearchInfo` (pickrst.h) — C zero-initializes (memset), so the
/// wiener/sgrproj members here are ZEROS, not the syntax defaults.
#[derive(Clone, Copy)]
struct RestUnitSearchInfo {
    best_rtype: [u8; 3],
    wiener: WienerInfoLr,
    sgrproj: SgrprojInfoLr,
}

impl Default for RestUnitSearchInfo {
    fn default() -> Self {
        RestUnitSearchInfo {
            best_rtype: [RESTORE_NONE; 3],
            wiener: WienerInfoLr {
                vfilter: [0; 8],
                hfilter: [0; 8],
            },
            sgrproj: SgrprojInfoLr { ep: 0, xqd: [0, 0] },
        }
    }
}

/// One plane's staged buffers: the extended dgd (recon) in the padded
/// frame-walk layout, the trial dst, the stripe boundaries, and the source.
#[derive(Clone)]
struct PlaneCtx<'a> {
    plane: usize,
    pw: i32,
    ph: i32,
    sx: usize,
    sy: usize,
    w_stride: usize,
    dgd_pad: Vec<u16>,
    dst_pad: Vec<u16>,
    bnd: StripeBoundaries<u16>,
    src: &'a [u16],
    src_stride: usize,
    // bd8 only (`!input.highbd`): the SAME buffers on a u8 carrier — the
    // lowbd kernels' arithmetic is identical either way, the loads are
    // half the width.
    dgd_pad8: Vec<u8>,
    src8: Vec<u8>,
    bnd8: StripeBoundaries<u8>,
    stripe_scratch8: crate::restore::frame::StripeScratch<u8>,
    flt0: Vec<i32>,
    flt1: Vec<i32>,
    wiener_scratch: crate::restore::wiener::WienerScratch,
    stripe_scratch: crate::restore::frame::StripeScratch<u16>,
}

impl<'a> PlaneCtx<'a> {
    /// Stage one plane: pad + `av1_extend_frame` the current recon, build
    /// the boundary context exactly as the encoder's two
    /// `av1_loop_restoration_save_boundary_lines` passes do.
    fn new(input: &LrSearchInput<'a>, plane: usize) -> PlaneCtx<'a> {
        let p = &input.planes[plane];
        let (sx, sy) = if plane > 0 {
            (input.ss_x, input.ss_y)
        } else {
            (0, 0)
        };
        let pw = (input.crop_width + (1 << sx) - 1) >> sx;
        let ph = (input.crop_height + (1 << sy) - 1) >> sy;
        let (pwu, phu) = (pw as usize, ph as usize);

        // Boundary buffers (av1_alloc_restoration_buffers geometry — stripes
        // counted on the LUMA extent).
        let mi_h = ((input.crop_height + 7) & !7) as usize;
        let ext_h = RESTORATION_UNIT_OFFSET as usize + mi_h;
        let num_stripes = ext_h.div_ceil(64);
        let b_stride = (pwu + 2 * 4 + 31) & !31;
        let mut bnd: StripeBoundaries<u16> = StripeBoundaries {
            above: Vec::new(),
            below: Vec::new(),
            stride: b_stride,
        };
        let mut bnd8: StripeBoundaries<u8> = StripeBoundaries {
            above: Vec::new(),
            below: Vec::new(),
            stride: b_stride,
        };
        // Encoder ordering (cdef_restoration_frame): pass 0 (internal stripe
        // context) on the DEBLOCKED frame BEFORE CDEF; pass 1 (frame edges)
        // on the CURRENT frame after CDEF.
        if input.highbd {
            bnd.above = vec![0; num_stripes * 2 * b_stride];
            bnd.below = vec![0; num_stripes * 2 * b_stride];
            save_boundary_lines(&mut bnd, p.deblocked, p.stride, pwu, phu, sy, false);
            save_boundary_lines(&mut bnd, p.cur, p.stride, pwu, phu, sy, true);
        } else {
            bnd8.above = vec![0; num_stripes * 2 * b_stride];
            bnd8.below = vec![0; num_stripes * 2 * b_stride];
            save_boundary_lines(&mut bnd8, p.deblocked, p.stride, pwu, phu, sy, false);
            save_boundary_lines(&mut bnd8, p.cur, p.stride, pwu, phu, sy, true);
        }

        // Padded dgd + trial dst (frame-walk layout). At bd8 the u16 carrier
        // is skipped entirely — the kernels all read the u8 twin.
        let w_stride = pwu + 2 * MARGIN_H;
        let (mut dgd_pad, mut dgd_pad8) = if input.highbd {
            (
                vec![0u16; w_stride * (phu + 2 * MARGIN_V)],
                Vec::new(),
            )
        } else {
            (
                Vec::new(),
                vec![0u8; w_stride * (phu + 2 * MARGIN_V)],
            )
        };
        if input.highbd {
            for r in 0..phu {
                dgd_pad[at(w_stride, r as isize, 0)..at(w_stride, r as isize, pw as isize)]
                    .copy_from_slice(&p.cur[r * p.stride..][..pwu]);
            }
            extend_frame(&mut dgd_pad, pwu, phu, w_stride);
        } else {
            for r in 0..phu {
                crate::lowbd::narrow_u16_to_u8_into(
                    &p.cur[r * p.stride..][..pwu],
                    &mut dgd_pad8[at(w_stride, r as isize, 0)..at(w_stride, r as isize, pw as isize)],
                );
            }
            extend_frame(&mut dgd_pad8, pwu, phu, w_stride);
        }
        let dst_pad = vec![0u16; w_stride * (phu + 2 * MARGIN_V)];

        let src8 = if input.highbd {
            Vec::new()
        } else {
            crate::lowbd::narrow_u16_to_u8(p.src)
        };

        PlaneCtx {
            plane,
            pw,
            ph,
            sx,
            sy,
            w_stride,
            dgd_pad,
            dst_pad,
            bnd,
            src: p.src,
            src_stride: p.stride,
            dgd_pad8,
            src8,
            bnd8,
            flt0: vec![0i32; RESTORATION_UNITPELS_MAX],
            flt1: vec![0i32; RESTORATION_UNITPELS_MAX],
            wiener_scratch: crate::restore::wiener::WienerScratch::new(),
            stripe_scratch: crate::restore::frame::StripeScratch::default(),
            stripe_scratch8: crate::restore::frame::StripeScratch::default(),
        }
    }

    /// Padded-buffer element offset of plane coord `(row, col)`.
    #[inline]
    fn pad_off(&self, row: i32, col: i32) -> usize {
        at(self.w_stride, row as isize, col as isize)
    }

    /// `sse_restoration_unit` (pickrst.c): SSE of source vs the trial dst
    /// over the unit rect.
    fn sse_dst(&self, limits: (i32, i32, i32, i32)) -> i64 {
        let (v0, v1, h0, h1) = limits;
        let (w, h) = ((h1 - h0) as usize, (v1 - v0) as usize);
        if !self.src8.is_empty() {
            return crate::dist::sse_u16_u8(
                &self.dst_pad[self.pad_off(v0, h0)..],
                self.w_stride,
                &self.src8[v0 as usize * self.src_stride + h0 as usize..],
                self.src_stride,
                w,
                h,
            );
        }
        crate::dist::highbd_sse(
            &self.src[v0 as usize * self.src_stride + h0 as usize..],
            self.src_stride,
            &self.dst_pad[self.pad_off(v0, h0)..],
            self.w_stride,
            w,
            h,
        )
    }

    /// SSE of source vs the CURRENT recon (RESTORE_NONE) over the rect.
    fn sse_none(&self, limits: (i32, i32, i32, i32)) -> i64 {
        let (v0, v1, h0, h1) = limits;
        let (w, h) = ((h1 - h0) as usize, (v1 - v0) as usize);
        if !self.src8.is_empty() {
            return crate::dist::sse(
                &self.dgd_pad8[self.pad_off(v0, h0)..],
                self.w_stride,
                &self.src8[v0 as usize * self.src_stride + h0 as usize..],
                self.src_stride,
                w,
                h,
            );
        }
        crate::dist::highbd_sse(
            &self.src[v0 as usize * self.src_stride + h0 as usize..],
            self.src_stride,
            &self.dgd_pad[self.pad_off(v0, h0)..],
            self.w_stride,
            w,
            h,
        )
    }

    /// `var_restoration_unit` (`aom_var_2d_u8/u16` / (w*h)): source variance
    /// over the rect. `highbd_variance` against a stride-0 zero reference at
    /// bd=8 returns the unnormalised `ss - s*s/n` (`src_var`'s numerator) —
    /// the bd>8 normalisation shifts would scale the raw sums, so bd stays 8
    /// regardless of stream depth.
    fn src_var(&self, limits: (i32, i32, i32, i32)) -> u64 {
        const ZEROS: [u16; 256] = [0; 256];
        let (v0, v1, h0, h1) = limits;
        let (w, h) = ((h1 - h0) as usize, (v1 - v0) as usize);
        let (var, _sse) = crate::dist::highbd_variance(
            &self.src[v0 as usize * self.src_stride + h0 as usize..],
            self.src_stride,
            &ZEROS[..w.min(256)],
            0,
            w,
            h,
            8,
        );
        u64::from(var) / (w * h) as u64
    }

    /// `try_restoration_unit` (pickrst.c): run the REAL per-unit filter
    /// (stripe boundaries, optimized_lr = 0 like the encoder) into the trial
    /// dst, return the unit SSE vs source.
    fn try_restoration_unit(
        &mut self,
        limits: (i32, i32, i32, i32),
        rui: &LrUnitInfo,
        bit_depth: i32,
    ) -> i64 {
        if !self.dgd_pad8.is_empty() {
            filter_unit(
                &mut self.dgd_pad8,
                &mut self.dst_pad,
                self.w_stride,
                rui,
                &self.bnd8,
                self.ph as usize,
                self.sx,
                self.sy,
                bit_depth,
                limits,
                false,
                &mut self.wiener_scratch,
                &mut self.stripe_scratch8,
            );
            return self.sse_dst(limits);
        }
        filter_unit(
            &mut self.dgd_pad,
            &mut self.dst_pad,
            self.w_stride,
            rui,
            &self.bnd,
            self.ph as usize,
            self.sx,
            self.sy,
            bit_depth,
            limits,
            false,
            &mut self.wiener_scratch,
            &mut self.stripe_scratch,
        );
        self.sse_dst(limits)
    }
}

/// `RestSearchCtxt`'s per-plane mutable search state.
struct RscState {
    sse: [i64; RESTORE_SWITCHABLE_TYPES],
    total_sse: [i64; RESTORE_TYPES],
    total_bits: [i64; RESTORE_TYPES],
    ref_wiener: WienerInfoLr,
    ref_sgrproj: SgrprojInfoLr,
    switchable_ref_wiener: WienerInfoLr,
    switchable_ref_sgrproj: SgrprojInfoLr,
    skip_sgr_eval: bool,
}

impl RscState {
    fn new() -> Self {
        RscState {
            sse: [0; RESTORE_SWITCHABLE_TYPES],
            total_sse: [0; RESTORE_TYPES],
            total_bits: [0; RESTORE_TYPES],
            ref_wiener: WienerInfoLr::default(),
            ref_sgrproj: SgrprojInfoLr::default(),
            switchable_ref_wiener: WienerInfoLr::default(),
            switchable_ref_sgrproj: SgrprojInfoLr::default(),
            skip_sgr_eval: false,
        }
    }

    /// `rsc_on_tile`.
    fn on_tile(&mut self) {
        self.ref_wiener = WienerInfoLr::default();
        self.ref_sgrproj = SgrprojInfoLr::default();
        self.switchable_ref_wiener = WienerInfoLr::default();
        self.switchable_ref_sgrproj = SgrprojInfoLr::default();
    }

    /// `reset_rsc`.
    fn reset(&mut self) {
        self.total_sse = [0; RESTORE_TYPES];
        self.total_bits = [0; RESTORE_TYPES];
    }
}

/// `search_norestore` (pickrst.c).
fn search_norestore(ctx: &PlaneCtx<'_>, limits: (i32, i32, i32, i32), rsc: &mut RscState) {
    rsc.sse[RESTORE_NONE as usize] = ctx.sse_none(limits);
    rsc.total_sse[RESTORE_NONE as usize] += rsc.sse[RESTORE_NONE as usize];
}

/// `finer_search_wiener` (pickrst.c): the ±{4,2,1} symmetric tap refinement
/// driven by real filter applications.
fn finer_search_wiener(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    limits: (i32, i32, i32, i32),
    rui: &mut LrUnitInfo,
    wiener_win: usize,
) -> i64 {
    let plane_off = (WIENER_WIN - wiener_win) >> 1;
    let mut err = ctx.try_restoration_unit(limits, rui, input.bit_depth);
    if input.sf.disable_wiener_coeff_refine_search {
        return err;
    }
    let tap_min = [-5i16, -23, -17];
    let tap_max = [10i16, 8, 46];
    const START_STEP: i16 = 4;

    // dir 0 = hfilter first (like C), then vfilter, at each step size.
    let mut s = START_STEP;
    while s >= 1 {
        for dir in 0..2 {
            for p in plane_off..WIENER_HALFWIN {
                let mut skip = false;
                loop {
                    let f = if dir == 0 {
                        &mut rui.wiener.hfilter
                    } else {
                        &mut rui.wiener.vfilter
                    };
                    if f[p] - s >= tap_min[p] {
                        f[p] -= s;
                        f[WIENER_WIN - p - 1] -= s;
                        f[WIENER_HALFWIN] += 2 * s;
                        let err2 = ctx.try_restoration_unit(limits, rui, input.bit_depth);
                        if err2 > err {
                            let f = if dir == 0 {
                                &mut rui.wiener.hfilter
                            } else {
                                &mut rui.wiener.vfilter
                            };
                            f[p] += s;
                            f[WIENER_WIN - p - 1] += s;
                            f[WIENER_HALFWIN] -= 2 * s;
                        } else {
                            err = err2;
                            skip = true;
                            // At the highest step size continue moving in the
                            // same direction.
                            if s == START_STEP {
                                continue;
                            }
                        }
                    }
                    break;
                }
                if skip {
                    break;
                }
                loop {
                    let f = if dir == 0 {
                        &mut rui.wiener.hfilter
                    } else {
                        &mut rui.wiener.vfilter
                    };
                    if f[p] + s <= tap_max[p] {
                        f[p] += s;
                        f[WIENER_WIN - p - 1] += s;
                        f[WIENER_HALFWIN] -= 2 * s;
                        let err2 = ctx.try_restoration_unit(limits, rui, input.bit_depth);
                        if err2 > err {
                            let f = if dir == 0 {
                                &mut rui.wiener.hfilter
                            } else {
                                &mut rui.wiener.vfilter
                            };
                            f[p] -= s;
                            f[WIENER_WIN - p - 1] -= s;
                            f[WIENER_HALFWIN] += 2 * s;
                        } else {
                            err = err2;
                            if s == START_STEP {
                                continue;
                            }
                        }
                    }
                    break;
                }
            }
        }
        s >>= 1;
    }
    err
}

/// `search_wiener` (pickrst.c).
#[allow(clippy::too_many_arguments)]
fn search_wiener(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    limits: (i32, i32, i32, i32),
    rsc: &mut RscState,
    rusi: &mut RestUnitSearchInfo,
) {
    let bits_none = input.wiener_restore_cost[0] as i64;

    // Skip Wiener search for low variance contents.
    if input.sf.prune_wiener_based_on_src_var > 0 {
        let scale = [0u64, 1, 2];
        let qs = (input.dc_quant_qtx >> 3) as u64;
        let thresh = (qs * qs * scale[input.sf.prune_wiener_based_on_src_var as usize]) >> 4;
        let src_var = ctx.src_var(limits);
        let prune_wiener = (src_var < thresh) || (rsc.sse[RESTORE_NONE as usize] == 0);
        if prune_wiener {
            rsc.total_bits[RESTORE_WIENER as usize] += bits_none;
            rsc.total_sse[RESTORE_WIENER as usize] += rsc.sse[RESTORE_NONE as usize];
            rusi.best_rtype[RESTORE_WIENER as usize - 1] = RESTORE_NONE;
            rsc.sse[RESTORE_WIENER as usize] = i64::MAX;
            if input.sf.prune_sgr_based_on_wiener == 2 {
                rsc.skip_sgr_eval = true;
            }
            return;
        }
    }

    let wiener_win = if ctx.plane == 0 {
        WIENER_WIN
    } else {
        WIENER_WIN_CHROMA
    };
    let reduced_wiener_win = if input.sf.reduce_wiener_window_size {
        if ctx.plane == 0 {
            WIENER_WIN_REDUCED
        } else {
            WIENER_WIN_CHROMA
        }
    } else {
        wiener_win
    };

    let mut m = [0i64; WIENER_WIN2];
    let mut h = [0i64; WIENER_WIN2 * WIENER_WIN2];
    let (v0, v1, h0, h1) = limits;
    let dgd_origin = ctx.pad_off(0, 0);
    if input.highbd {
        compute_stats_highbd(
            reduced_wiener_win,
            &ctx.dgd_pad,
            dgd_origin,
            ctx.src,
            h0,
            h1,
            v0,
            v1,
            ctx.w_stride as i32,
            ctx.src_stride as i32,
            &mut m,
            &mut h,
            input.bit_depth,
        );
    } else {
        compute_stats(
            reduced_wiener_win,
            &ctx.dgd_pad8,
            dgd_origin,
            &ctx.src8,
            h0,
            h1,
            v0,
            v1,
            ctx.w_stride as i32,
            ctx.src_stride as i32,
            &mut m,
            &mut h,
            input.sf.use_downsampled_wiener_stats,
        );
    }

    let mut vfilter = [0i32; WIENER_WIN];
    let mut hfilter = [0i32; WIENER_WIN];
    wiener_decompose_sep_sym(reduced_wiener_win, &m, &h, &mut vfilter, &mut hfilter);

    let mut rui = LrUnitInfo {
        restoration_type: RESTORE_WIENER,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: SgrprojInfoLr { ep: 0, xqd: [0, 0] },
    };
    finalize_sym_filter(reduced_wiener_win, &vfilter, &mut rui.wiener.vfilter);
    finalize_sym_filter(reduced_wiener_win, &hfilter, &mut rui.wiener.hfilter);

    // Filter-score gate: revert to identity (NONE) when the learned filter
    // does not reduce x'Hx - 2x'M.
    if compute_score(
        reduced_wiener_win,
        &m,
        &h,
        &rui.wiener.vfilter,
        &rui.wiener.hfilter,
    ) > 0
    {
        rsc.total_bits[RESTORE_WIENER as usize] += bits_none;
        rsc.total_sse[RESTORE_WIENER as usize] += rsc.sse[RESTORE_NONE as usize];
        rusi.best_rtype[RESTORE_WIENER as usize - 1] = RESTORE_NONE;
        rsc.sse[RESTORE_WIENER as usize] = i64::MAX;
        if input.sf.prune_sgr_based_on_wiener == 2 {
            rsc.skip_sgr_eval = true;
        }
        return;
    }

    rsc.sse[RESTORE_WIENER as usize] =
        finer_search_wiener(ctx, input, limits, &mut rui, reduced_wiener_win);
    rusi.wiener = rui.wiener;

    let bits_wiener = input.wiener_restore_cost[1] as i64
        + ((count_wiener_bits(wiener_win, &rusi.wiener, &rsc.ref_wiener) as i64)
            << AV1_PROB_COST_SHIFT);

    let cost_none = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_none >> 4,
        rsc.sse[RESTORE_NONE as usize],
        input.bit_depth,
    );
    let cost_wiener = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_wiener >> 4,
        rsc.sse[RESTORE_WIENER as usize],
        input.bit_depth,
    );

    let rtype = if cost_wiener < cost_none {
        RESTORE_WIENER
    } else {
        RESTORE_NONE
    };
    rusi.best_rtype[RESTORE_WIENER as usize - 1] = rtype;

    if input.sf.prune_sgr_based_on_wiener == 1 {
        rsc.skip_sgr_eval = cost_wiener > (1.01 * cost_none);
    } else if input.sf.prune_sgr_based_on_wiener == 2 {
        rsc.skip_sgr_eval = rusi.best_rtype[RESTORE_WIENER as usize - 1] == RESTORE_NONE;
    }

    rsc.total_sse[RESTORE_WIENER as usize] += rsc.sse[rtype as usize];
    rsc.total_bits[RESTORE_WIENER as usize] += if cost_wiener < cost_none {
        bits_wiener
    } else {
        bits_none
    };
    if cost_wiener < cost_none {
        rsc.ref_wiener = rusi.wiener;
    }
}

/// `apply_sgr` (pickrst.c): the SGR passes over the unit in procunit tiles,
/// producing flt0/flt1 at `flt_stride`.
#[allow(clippy::too_many_arguments)]
fn apply_sgr_unit(
    ctx: &mut PlaneCtx<'_>,
    ep: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    pu_width: usize,
    pu_height: usize,
    flt_stride: usize,
    bit_depth: i32,
) {
    let mut i = 0usize;
    while i < height {
        let h = pu_height.min(height - i);
        let mut j = 0usize;
        while j < width {
            let w = pu_width.min(width - j);
            let flt_off = i * flt_stride + j;
            let (f0, f1) = (&mut ctx.flt0[flt_off..], &mut ctx.flt1[flt_off..]);
            if !ctx.dgd_pad8.is_empty() {
                selfguided_restoration(
                    &ctx.dgd_pad8,
                    dgd_off + i * ctx.w_stride + j,
                    ctx.w_stride,
                    w,
                    h,
                    f0,
                    f1,
                    flt_stride,
                    ep,
                    bit_depth,
                );
            } else {
                selfguided_restoration(
                    &ctx.dgd_pad,
                    dgd_off + i * ctx.w_stride + j,
                    ctx.w_stride,
                    w,
                    h,
                    f0,
                    f1,
                    flt_stride,
                    ep,
                    bit_depth,
                );
            }
            j += pu_width;
        }
        i += pu_height;
    }
}

/// `get_pixel_proj_error` (pickrst.c): xqd -> xq, then the exact projected
/// SSE.
#[allow(clippy::too_many_arguments)]
fn get_pixel_proj_error_xqd(
    ctx: &PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    flt_stride: usize,
    xqd: [i32; 2],
    ep: usize,
) -> i64 {
    let xq = decode_xq(&xqd, ep);
    if !ctx.src8.is_empty() {
        return pixel_proj_error(
            &ctx.src8,
            src_off,
            width,
            height,
            ctx.src_stride,
            &ctx.dgd_pad8,
            dgd_off,
            ctx.w_stride,
            &ctx.flt0,
            flt_stride,
            &ctx.flt1,
            flt_stride,
            xq,
            ep,
            input.highbd,
        );
    }
    pixel_proj_error(
        ctx.src,
        src_off,
        width,
        height,
        ctx.src_stride,
        &ctx.dgd_pad,
        dgd_off,
        ctx.w_stride,
        &ctx.flt0,
        flt_stride,
        &ctx.flt1,
        flt_stride,
        xq,
        ep,
        input.highbd,
    )
}

/// `finer_search_pixel_proj_error` (pickrst.c): the ±{2,1} xqd refinement.
#[allow(clippy::too_many_arguments)]
fn finer_search_pixel_proj_error(
    ctx: &PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    flt_stride: usize,
    start_step: i32,
    xqd: &mut [i32; 2],
    ep: usize,
) -> i64 {
    let mut err =
        get_pixel_proj_error_xqd(ctx, input, src_off, dgd_off, width, height, flt_stride, *xqd, ep);
    let (rads, _) = SGR_PARAMS[ep];
    let tap_min = [SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1];
    let tap_max = [SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1];
    let mut s = start_step;
    while s >= 1 {
        for p in 0..2 {
            if (rads[0] == 0 && p == 0) || (rads[1] == 0 && p == 1) {
                continue;
            }
            let mut skip = false;
            loop {
                if xqd[p] - s >= tap_min[p] {
                    xqd[p] -= s;
                    let err2 = get_pixel_proj_error_xqd(
                        ctx, input, src_off, dgd_off, width, height, flt_stride, *xqd, ep,
                    );
                    if err2 > err {
                        xqd[p] += s;
                    } else {
                        err = err2;
                        skip = true;
                        if s == start_step {
                            continue;
                        }
                    }
                }
                break;
            }
            if skip {
                break;
            }
            loop {
                if xqd[p] + s <= tap_max[p] {
                    xqd[p] += s;
                    let err2 = get_pixel_proj_error_xqd(
                        ctx, input, src_off, dgd_off, width, height, flt_stride, *xqd, ep,
                    );
                    if err2 > err {
                        xqd[p] -= s;
                    } else {
                        err = err2;
                        if s == start_step {
                            continue;
                        }
                    }
                }
                break;
            }
        }
        s >>= 1;
    }
    err
}

/// `compute_sgrproj_err` (pickrst.c) for one `ep`.
#[allow(clippy::too_many_arguments)]
fn compute_sgrproj_err(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    pu_width: usize,
    pu_height: usize,
    ep: usize,
    flt_stride: usize,
) -> ([i32; 2], i64) {
    apply_sgr_unit(
        ctx,
        ep,
        dgd_off,
        width,
        height,
        pu_width,
        pu_height,
        flt_stride,
        input.bit_depth,
    );
    let exq = if !ctx.src8.is_empty() {
        get_proj_subspace(
            &ctx.src8,
            src_off,
            width,
            height,
            ctx.src_stride,
            &ctx.dgd_pad8,
            dgd_off,
            ctx.w_stride,
            &ctx.flt0,
            flt_stride,
            &ctx.flt1,
            flt_stride,
            ep,
        )
    } else {
        get_proj_subspace(
            ctx.src,
            src_off,
            width,
            height,
            ctx.src_stride,
            &ctx.dgd_pad,
            dgd_off,
            ctx.w_stride,
            &ctx.flt0,
            flt_stride,
            &ctx.flt1,
            flt_stride,
            ep,
        )
    };
    let mut exqd = encode_xq(exq, ep);
    let err = finer_search_pixel_proj_error(
        ctx, input, src_off, dgd_off, width, height, flt_stride, 2, &mut exqd, ep,
    );
    (exqd, err)
}

/// `search_selfguided_restoration` (pickrst.c): the ep ladder.
#[allow(clippy::too_many_arguments)]
fn search_selfguided_restoration(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    pu_width: usize,
    pu_height: usize,
) -> SgrprojInfoLr {
    let flt_stride = ((width + 7) & !7) + 8;
    let mut bestep = 0i32;
    let mut besterr: i64 = -1;
    let mut bestxqd = [0i32; 2];
    let consider = |ctx: &mut PlaneCtx<'_>,
                        ep: i32,
                        bestep: &mut i32,
                        besterr: &mut i64,
                        bestxqd: &mut [i32; 2]| {
        let (exqd, err) = compute_sgrproj_err(
            ctx, input, src_off, dgd_off, width, height, pu_width, pu_height, ep as usize,
            flt_stride,
        );
        if *besterr == -1 || err < *besterr {
            *bestep = ep;
            *besterr = err;
            *bestxqd = exqd;
        }
    };
    if input.sf.enable_sgr_ep_pruning == 0 {
        for ep in 0..16 {
            consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
        }
    } else {
        // Evaluate the four group-1 seeds.
        for &ep in &SGRPROJ_EP_GRP1_SEED {
            consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
        }
        if input.sf.enable_sgr_ep_pruning < 2 {
            // Left/right of the winner within group 1.
            let bestep_ref = bestep;
            let mut ep = bestep_ref - 1;
            while ep < bestep_ref + 2 {
                if ep >= SGRPROJ_EP_GRP1_START_IDX && ep <= SGRPROJ_EP_GRP1_END_IDX {
                    consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
                }
                ep += 2;
            }
            // The two group-2/3 rows indexed by the current winner.
            for idx in 0..2 {
                let ep = SGRPROJ_EP_GRP2_3[idx][bestep as usize];
                consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
            }
        }
    }
    SgrprojInfoLr {
        ep: bestep,
        xqd: bestxqd,
    }
}

/// `search_sgrproj` (pickrst.c).
fn search_sgrproj(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    limits: (i32, i32, i32, i32),
    rsc: &mut RscState,
    rusi: &mut RestUnitSearchInfo,
) {
    let bits_none = input.sgrproj_restore_cost[0] as i64;
    if rsc.skip_sgr_eval {
        rsc.total_bits[RESTORE_SGRPROJ as usize] += bits_none;
        rsc.total_sse[RESTORE_SGRPROJ as usize] += rsc.sse[RESTORE_NONE as usize];
        rusi.best_rtype[RESTORE_SGRPROJ as usize - 1] = RESTORE_NONE;
        rsc.sse[RESTORE_SGRPROJ as usize] = i64::MAX;
        return;
    }

    let (v0, v1, h0, h1) = limits;
    let dgd_off = ctx.pad_off(v0, h0);
    let src_off = v0 as usize * ctx.src_stride + h0 as usize;
    let procunit_width = (RESTORATION_PROC_UNIT_SIZE >> ctx.sx) as usize;
    let procunit_height = (RESTORATION_PROC_UNIT_SIZE >> ctx.sy) as usize;

    rusi.sgrproj = search_selfguided_restoration(
        ctx,
        input,
        src_off,
        dgd_off,
        (h1 - h0) as usize,
        (v1 - v0) as usize,
        procunit_width,
        procunit_height,
    );

    let rui = LrUnitInfo {
        restoration_type: RESTORE_SGRPROJ,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: rusi.sgrproj,
    };
    rsc.sse[RESTORE_SGRPROJ as usize] = ctx.try_restoration_unit(limits, &rui, input.bit_depth);

    let bits_sgr = input.sgrproj_restore_cost[1] as i64
        + ((count_sgrproj_bits(&rusi.sgrproj, &rsc.ref_sgrproj) as i64) << AV1_PROB_COST_SHIFT);
    let cost_none = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_none >> 4,
        rsc.sse[RESTORE_NONE as usize],
        input.bit_depth,
    );
    let mut cost_sgr = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_sgr >> 4,
        rsc.sse[RESTORE_SGRPROJ as usize],
        input.bit_depth,
    );
    if rusi.sgrproj.ep < 10 {
        cost_sgr *= 1.0 + DUAL_SGR_PENALTY_MULT * input.sf.dual_sgr_penalty_level as f64;
    }

    let rtype = if cost_sgr < cost_none {
        RESTORE_SGRPROJ
    } else {
        RESTORE_NONE
    };
    rusi.best_rtype[RESTORE_SGRPROJ as usize - 1] = rtype;

    rsc.total_sse[RESTORE_SGRPROJ as usize] += rsc.sse[if rtype == RESTORE_SGRPROJ {
        RESTORE_SGRPROJ as usize
    } else {
        RESTORE_NONE as usize
    }];
    rsc.total_bits[RESTORE_SGRPROJ as usize] += if cost_sgr < cost_none {
        bits_sgr
    } else {
        bits_none
    };
    if cost_sgr < cost_none {
        rsc.ref_sgrproj = rusi.sgrproj;
    }
}

/// `search_switchable` (pickrst.c).
fn search_switchable(
    ctx: &PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    rsc: &mut RscState,
    rusi: &mut RestUnitSearchInfo,
) {
    let wiener_win = if ctx.plane == 0 {
        WIENER_WIN
    } else {
        WIENER_WIN_CHROMA
    };

    let mut best_cost = 0.0f64;
    let mut best_bits: i64 = 0;
    let mut best_rtype = RESTORE_NONE;

    for r in 0..RESTORE_SWITCHABLE_TYPES as u8 {
        // Prune on SSE, not on the previous search's pick (see pickrst.c).
        if r > RESTORE_NONE && rsc.sse[r as usize] > rsc.sse[RESTORE_NONE as usize] {
            continue;
        }

        let sse = rsc.sse[r as usize];
        let coeff_pcost: i64 = match r {
            RESTORE_NONE => 0,
            RESTORE_WIENER => {
                count_wiener_bits(wiener_win, &rusi.wiener, &rsc.switchable_ref_wiener) as i64
            }
            _ => count_sgrproj_bits(&rusi.sgrproj, &rsc.switchable_ref_sgrproj) as i64,
        };
        let coeff_bits = coeff_pcost << AV1_PROB_COST_SHIFT;
        let bits = input.switchable_restore_cost[r as usize] as i64 + coeff_bits;
        let mut cost =
            rdcost_dbl_with_native_bd_dist(input.rdmult, bits >> 4, sse, input.bit_depth);
        if r == RESTORE_SGRPROJ && rusi.sgrproj.ep < 10 {
            cost *= 1.0 + DUAL_SGR_PENALTY_MULT * input.sf.dual_sgr_penalty_level as f64;
        }
        if r == RESTORE_WIENER || r == RESTORE_SGRPROJ {
            cost *= 1.0 + WIENER_SGR_PENALTY_MULT * input.sf.switchable_lr_with_bias_level as f64;
        }
        if r == 0 || cost < best_cost {
            best_cost = cost;
            best_bits = bits;
            best_rtype = r;
        }
    }

    rusi.best_rtype[RESTORE_SWITCHABLE as usize - 1] = best_rtype;

    rsc.total_sse[RESTORE_SWITCHABLE as usize] += rsc.sse[best_rtype as usize];
    rsc.total_bits[RESTORE_SWITCHABLE as usize] += best_bits;
    if best_rtype == RESTORE_WIENER {
        rsc.switchable_ref_wiener = rusi.wiener;
    }
    if best_rtype == RESTORE_SGRPROJ {
        rsc.switchable_ref_sgrproj = rusi.sgrproj;
    }
}

/// `av1_derive_flags_for_lr_processing` (pickrst.c).
fn derive_flags_for_lr_processing(sf: &LrSearchSf) -> [bool; RESTORE_TYPES] {
    let w = sf.disable_wiener_filter;
    let s = sf.disable_sgr_filter;
    [w && s, w, s, w || s]
}

/// `restoration_search` (pickrst.c): one plane at one unit size — the
/// SB-coding-order unit walk running each enabled search fn per unit.
#[allow(clippy::too_many_arguments)]
fn restoration_search(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    lr_geom: &LrFrameConfig,
    rsc: &mut RscState,
    rusi: &mut [RestUnitSearchInfo],
    disable_lr_filter: &[bool; RESTORE_TYPES],
) {
    rsc.reset();
    restoration_search_rows(
        ctx,
        input,
        lr_geom,
        rsc,
        rusi,
        None,
        &input.tile_sb_rows,
        disable_lr_filter,
    );
}

/// The tile-row body of `restoration_search` over `tile_rows` (a slice of
/// `input.tile_sb_rows`). `mask`, when present, is marked for every `rusi`
/// slot written — the threaded caller gives each worker a private `rusi` +
/// `mask` pair and merges the marked slots, since an RU row can straddle a
/// tile-row boundary and `split_at_mut` bands would overlap.
#[allow(clippy::too_many_arguments)]
fn restoration_search_rows(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    lr_geom: &LrFrameConfig,
    rsc: &mut RscState,
    rusi: &mut [RestUnitSearchInfo],
    mut mask: Option<&mut [bool]>,
    tile_rows: &[(i32, i32)],
    disable_lr_filter: &[bool; RESTORE_TYPES],
) {
    let plane = ctx.plane;
    let ru_size = lr_geom.unit_size[plane];
    let ext_size = ru_size * 3 / 2;
    let (horz_units, vert_units) = lr_geom.plane_units(plane, input.ss_x, input.ss_y);
    let plane_num_units = (horz_units * vert_units) as usize;
    let num_rtypes = if plane_num_units > 1 {
        RESTORE_TYPES
    } else {
        RESTORE_SWITCHABLE_TYPES
    };
    let mib_size = 1i32 << input.mib_size_log2;

    for &(sb_row_start, sb_row_end) in tile_rows {
        for &(sb_col_start, sb_col_end) in &input.tile_sb_cols {
            // Reset reference parameters for delta-coding at tile start.
            rsc.on_tile();

            for sb_row in sb_row_start..sb_row_end {
                let mi_row = sb_row << input.mib_size_log2;
                for sb_col in sb_col_start..sb_col_end {
                    let mi_col = sb_col << input.mib_size_log2;
                    let Some((rcol0, rcol1, rrow0, rrow1)) = lr_corners_in_sb(
                        lr_geom, plane, input.ss_x, input.ss_y, mi_row, mi_col, mib_size, mib_size,
                    ) else {
                        continue;
                    };

                    for rrow in rrow0..rrow1 {
                        let y0 = rrow * ru_size;
                        let remaining_h = ctx.ph - y0;
                        let h = if remaining_h < ext_size {
                            remaining_h
                        } else {
                            ru_size
                        };
                        let mut v_start = y0;
                        let mut v_end = y0 + h;
                        debug_assert!(v_end <= ctx.ph);
                        // Offset upwards to align with the processing stripe.
                        let voffset = RESTORATION_UNIT_OFFSET >> ctx.sy;
                        v_start = (v_start - voffset).max(0);
                        if v_end < ctx.ph {
                            v_end -= voffset;
                        }

                        for rcol in rcol0..rcol1 {
                            let x0 = rcol * ru_size;
                            let remaining_w = ctx.pw - x0;
                            let w = if remaining_w < ext_size {
                                remaining_w
                            } else {
                                ru_size
                            };
                            let limits = (v_start, v_end, x0, x0 + w);
                            let unit_idx = (rrow * horz_units + rcol) as usize;

                            rsc.skip_sgr_eval = false;
                            if let Some(m) = mask.as_deref_mut() {
                                m[unit_idx] = true;
                            }
                            for r in 0..num_rtypes {
                                if disable_lr_filter[r] {
                                    continue;
                                }
                                match r {
                                    0 => search_norestore(ctx, limits, rsc),
                                    1 => search_wiener(ctx, input, limits, rsc, &mut rusi[unit_idx]),
                                    2 => search_sgrproj(ctx, input, limits, rsc, &mut rusi[unit_idx]),
                                    _ => search_switchable(ctx, input, rsc, &mut rusi[unit_idx]),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `copy_unit_info` (pickrst.c).
fn copy_unit_info(frame_rtype: u8, rusi: &RestUnitSearchInfo) -> LrUnitInfo {
    debug_assert!(frame_rtype > 0);
    let rtype = rusi.best_rtype[frame_rtype as usize - 1];
    let mut u = LrUnitInfo {
        restoration_type: rtype,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: SgrprojInfoLr { ep: 0, xqd: [0, 0] },
    };
    if rtype == RESTORE_WIENER {
        u.wiener = rusi.wiener;
    } else {
        u.sgrproj = rusi.sgrproj;
    }
    u
}

/// `av1_pick_filter_restoration` (pickrst.c): the frame-level search over
/// unit sizes and planes. Returns the chosen unit size, per-plane frame
/// restoration types and per-unit parameters.
pub fn pick_filter_restoration(input: &LrSearchInput<'_>) -> LrSearchOutcome {
    let num_planes = input.planes.len();
    let sb_wide = 1i32 << (input.mib_size_log2 + 2); // block_size_wide[sb_size]

    // The minimum allowed unit size at a syntax level is 1 superblock.
    let min_lr_unit_size = input.sf.min_lr_unit_size.max(sb_wide);
    let max_lr_unit_size = input.sf.max_lr_unit_size.max(min_lr_unit_size);

    let mut outcome = LrSearchOutcome {
        unit_size: max_lr_unit_size,
        frame_restoration_type: [RESTORE_NONE; 3],
        units: [Vec::new(), Vec::new(), Vec::new()],
    };

    // Decide which planes to search.
    let plane_start = if input.sf.disable_loop_restoration_luma {
        1usize
    } else {
        0
    };
    let plane_end = if num_planes == 1 || input.sf.disable_loop_restoration_chroma {
        0usize
    } else {
        2
    };
    if plane_start > plane_end {
        return outcome;
    }

    let disable_lr_filter = derive_flags_for_lr_processing(&input.sf);
    // Wiener+SGR both disabled: nothing to search (the C search loop would
    // skip every fn and pick NONE everywhere).
    if disable_lr_filter[RESTORE_NONE as usize] {
        return outcome;
    }

    // Stage the searched planes (av1_extend_frame + boundary saves happen
    // once, before the size loop).
    let mut ctxs: Vec<PlaneCtx<'_>> = (plane_start..=plane_end)
        .map(|p| PlaneCtx::new(input, p))
        .collect();

    let mut best_cost = f64::MAX;
    let mut best_luma_unit_size = max_lr_unit_size;
    let mut rsc = RscState::new();

    let mut luma_unit_size = max_lr_unit_size;
    while luma_unit_size >= min_lr_unit_size {
        let lr_geom = LrFrameConfig {
            frame_restoration_type: [RESTORE_WIENER; 3], // corners fn ignores this
            unit_size: [luma_unit_size; 3],
            crop_width: input.crop_width,
            crop_height: input.crop_height,
            superres_denom: 0,
        };

        let mut bits_this_size: i64 = 0;
        let mut sse_this_size: i64 = 0;
        let mut best_rtype: [u8; 3] = [RESTORE_NONE; 3];
        let mut rusi_this_size: Vec<Vec<RestUnitSearchInfo>> = Vec::new();

        for (ci, plane) in (plane_start..=plane_end).enumerate() {
            let ctx = &mut ctxs[ci];
            let ctx_template = &*ctx;
            let (hu, vu) = lr_geom.plane_units(plane, input.ss_x, input.ss_y);
            let plane_num_units = (hu * vu) as usize;
            let mut rusi = vec![RestUnitSearchInfo::default(); plane_num_units];

            let n_tile_rows = input.tile_sb_rows.len();
            if input.threads > 1 && n_tile_rows > 1 {
                // Tile rows are independent: `rsc_on_tile` resets the
                // delta-coding references at every tile start, so disjoint
                // rows only share the commutative total_bits/total_sse sums.
                // Each worker stages its own PlaneCtx (identical inputs →
                // identical staging) and returns a masked private `rusi`,
                // which the merge copies back — the unit results are
                // byte-identical to the serial walk.
                let workers = input.threads.min(n_tile_rows);
                let (rows_per, rem) = (n_tile_rows / workers, n_tile_rows % workers);
                let tile_sb_rows = input.tile_sb_rows.as_slice();
                rsc.reset();
                // `par::map_workers` = std::thread::scope by default, or the
                // host's rayon pool under the `rayon` feature — same tasks.
                let outs = crate::par::map_workers(workers, |w| {
                    // Worker w's contiguous chunk: rows_per + (w < rem).
                    let start = w * rows_per + w.min(rem);
                    let take = rows_per + usize::from(w < rem);
                    let rows: Vec<usize> = (start..start + take).collect();
                    // Clone the already-staged ctx: ~3 MB memcpy
                    // instead of a full pad+extend+boundary restage.
                    let mut ctx_w = ctx_template.clone();
                    let mut rsc_w = RscState::new();
                    rsc_w.reset();
                    let mut rusi_w =
                        vec![RestUnitSearchInfo::default(); plane_num_units];
                    let mut mask_w = vec![false; plane_num_units];
                    let tile_rows: Vec<(i32, i32)> =
                        rows.iter().map(|&i| tile_sb_rows[i]).collect();
                    restoration_search_rows(
                        &mut ctx_w,
                        input,
                        &lr_geom,
                        &mut rsc_w,
                        &mut rusi_w,
                        Some(&mut mask_w),
                        &tile_rows,
                        &disable_lr_filter,
                    );
                    (rsc_w, rusi_w, mask_w)
                });
                for (rsc_w, rusi_w, mask_w) in outs {
                    for r in 0..RESTORE_TYPES {
                        rsc.total_sse[r] += rsc_w.total_sse[r];
                        rsc.total_bits[r] += rsc_w.total_bits[r];
                    }
                    for (i, &m) in mask_w.iter().enumerate() {
                        if m {
                            rusi[i] = rusi_w[i];
                        }
                    }
                }
            } else {
                restoration_search(ctx, input, &lr_geom, &mut rsc, &mut rusi, &disable_lr_filter);
            }

            let num_rtypes = if plane_num_units > 1 {
                RESTORE_TYPES
            } else {
                RESTORE_SWITCHABLE_TYPES
            };
            let mut best_cost_this_plane = f64::MAX;
            for r in 0..num_rtypes {
                if disable_lr_filter[r] {
                    continue;
                }
                // switchable_lr_with_bias_level restricts to SWITCHABLE.
                if input.sf.switchable_lr_with_bias_level > 0
                    && (r == RESTORE_WIENER as usize || r == RESTORE_SGRPROJ as usize)
                {
                    continue;
                }
                let cost_this_plane = rdcost_dbl_with_native_bd_dist(
                    input.rdmult,
                    rsc.total_bits[r] >> 4,
                    rsc.total_sse[r],
                    input.bit_depth,
                );
                if cost_this_plane < best_cost_this_plane {
                    best_cost_this_plane = cost_this_plane;
                    best_rtype[plane] = r as u8;
                }
            }

            bits_this_size += rsc.total_bits[best_rtype[plane] as usize];
            sse_this_size += rsc.total_sse[best_rtype[plane] as usize];
            rusi_this_size.push(rusi);
        }

        let cost_this_size = rdcost_dbl_with_native_bd_dist(
            input.rdmult,
            bits_this_size >> 4,
            sse_this_size,
            input.bit_depth,
        );

        if cost_this_size < best_cost {
            best_cost = cost_this_size;
            best_luma_unit_size = luma_unit_size;
            // Copy parameters out before the next size overwrites them.
            let mut all_none = true;
            for (ci, plane) in (plane_start..=plane_end).enumerate() {
                outcome.frame_restoration_type[plane] = best_rtype[plane];
                outcome.units[plane].clear();
                if best_rtype[plane] != RESTORE_NONE {
                    all_none = false;
                    for u in &rusi_this_size[ci] {
                        outcome.units[plane].push(copy_unit_info(best_rtype[plane], u));
                    }
                }
            }
            // Heuristic: all NONE at this size -> smaller sizes won't help.
            if all_none {
                break;
            }
        } else {
            // Heuristic: worse than the previous (larger) size -> stop.
            break;
        }

        luma_unit_size >>= 1;
    }

    outcome.unit_size = best_luma_unit_size;
    outcome
}
