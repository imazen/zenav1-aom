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

// The generic i32x8 body is only emitted for the neon/wasm128 tiers — on
// x86-64 the v3 mirror and the scalar port cover dispatch, so the prelude is
// unused there.
#[cfg(not(target_arch = "x86_64"))]
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
#[derive(Default, Clone)]
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
pub fn wiener_convolve_add_src<P: crate::restore::pick::LrPixel>(
    src: &[P],
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
pub fn wiener_convolve_add_src_into<P: crate::restore::pick::LrPixel>(
    src: &[P],
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
fn wiener_impl_scalar<P: crate::restore::pick::LrPixel>(
    _t: archmage::ScalarToken,
    src: &[P],
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
/// perfectly predicted). Used only by the neon/wasm128 body.
#[cfg(not(target_arch = "x86_64"))]
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

/// The v3 tier — an instruction-level mirror of the REAL
/// `av1_highbd_wiener_convolve_add_src_avx2` (the kernel actually dispatched
/// for our u16 planes at every bit depth).
///
/// Two structural differences from the generic i32x8 body it replaces:
///
/// * **Horizontal: shifted-window loads, no unpack.** For each 16-output tile
///   C loads `src[x0 + k .. x0 + k + 16)` for `k = 0..8` — eight overlapping
///   windows — and `madd_epi16`s them directly against `[f(2k) f(2k+1)]`
///   pairs, so the even/odd lane split IS the even/odd output-column split.
///   The `(in[c+3] << 7)` centre-tap term and the `1 << (bd + FILTER_BITS - 1)`
///   offset are folded into the coefficients (`tap[3] += 1 << FILTER_BITS`)
///   and the rounding constant — two fewer adds per tile.
/// * **Vertical: row-pair unpacks.** Eight 16-column row loads get
///   `unpacklo/hi_epi16`'d into (row_k[c], row_k+1[c]) pairs for the madds.
///   The horizontal pass stores `temp` in `packs_epi32` lane order
///   ([e0..e3 o0..o3 | e4..e7 o4..o7] over columns); the vertical's own
///   `unpacklo/hi_epi32` + final `packs_epi32` undo that permutation, so the
///   dst store lands in natural order with no fixup permute anywhere.
///
/// Width tails use the same overlap-back trick as the i32x8 body (`x0 =
/// min(xs, w-16)`): outputs are pure functions of the input window, so a
/// recomputed column stores an identical value — and, unlike C's
/// unconditional `j += 16` (which for `w % 16 == 8` writes 8 columns past
/// `w` into the next unit's dst region), the port never stores outside the
/// `w x h` block, which `kernels_diff.rs` asserts on the whole buffer.
///
/// `madd_epi16` reads its inputs as SIGNED i16 — for u16 samples >= 32768
/// (outside every valid bit depth) this kernel computes what C-avx2 computes,
/// which differs from the unsigned-widening i32x8/scalar bodies. On the
/// reachable domain (values <= (1<<bd)-1 <= 4095) all three are identical,
/// and the `packs_epi32` saturation before the epi16 clamp is a no-op because
/// the clamp ceiling is <= i16::MAX by construction (`conv_params_wiener`
/// raises `round_0` precisely so `bd + FILTER_BITS - round_0 + 2 <= 16`).
///
/// `w < 16` (which the real C kernel never sees — it asserts `w % 8 == 0` and
/// its tile step is 16) and any out-of-window slice fall back to the scalar
/// port: identical observable behaviour, and the same index-panic contract
/// for truly out-of-bounds callers.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn wiener_impl_v3<P: crate::restore::pick::LrPixel>(
    _t: archmage::X64V3Token,
    src: &[P],
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
    use archmage::intrinsics::x86_64::*;
    assert!(
        w >= 8 && w <= MAX_SB_SIZE,
        "wiener: restoration-unit width {w} outside 8..={MAX_SB_SIZE} — the SIMD path \
         loads 8 lanes at a time and `temp` is strided by MAX_SB_SIZE"
    );
    let (round_0, round_1) = conv_params_wiener(bd);
    let ih = h + SUBPEL_TAPS - 1;
    let hb = src_off as isize - 3 * src_stride as isize - 3;
    // Contract preflight: every horizontal load touches
    // `hb + y*src_stride + x0 + [0, 23)` for x0 <= w-16, and every vertical
    // store touches `dst_off + y*dst_stride + [x0, x0+16)`. A window outside
    // the slices routes to the scalar port — same values on [0, w) x [0, h),
    // same panic-on-real-OOB as the generic body.
    if h == 0
        || w < 16
        || hb < 0
        || (hb as usize) + (ih - 1) * src_stride + w + 7 > src.len()
        || temp.len() < ih * MAX_SB_SIZE
        || dst_off + (h - 1) * dst_stride + w > dst.len()
    {
        return wiener_scalar_into(
            src, src_off, src_stride, dst, dst_off, dst_stride, hfilter, vfilter, w, h, bd, temp,
        );
    }
    let hb = hb as usize;

    // Coefficient registers: C builds [f(2k) f(2k+1)] x8 per tap pair with the
    // "add_src" offset folded in — `_mm_add_epi16(coeffs, 1<<FILTER_BITS at
    // lane 3)` (wrapping i16, matching C exactly).
    let offset = _mm_insert_epi16::<3>(_mm_setzero_si128(), 1 << FILTER_BITS);
    let build = |f: &[i16; 8]| -> [__m256i; 4] {
        let cx = _mm_add_epi16(_mm_loadu_si128(f), offset);
        let c0123 = _mm_unpacklo_epi32(cx, cx);
        let c4567 = _mm_unpackhi_epi32(cx, cx);
        [
            _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(c0123, c0123)),
            _mm256_broadcastsi128_si256(_mm_unpackhi_epi64(c0123, c0123)),
            _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(c4567, c4567)),
            _mm256_broadcastsi128_si256(_mm_unpackhi_epi64(c4567, c4567)),
        ]
    };
    let ch = build(hfilter);
    let cv = build(vfilter);
    let zero = _mm256_setzero_si256();

    // ---- horizontal pass: convolve_lowbd_x's highbd twin — 8 shifted-window
    // loads + 8 madds per 16 outputs, stores to `temp` in packs order ----
    let clamp_limit = 1i32 << (bd + 1 + FILTER_BITS - round_0);
    let hi_h = _mm256_set1_epi16((clamp_limit - 1) as i16);
    let rc_h = _mm256_set1_epi32((1 << (round_0 - 1)) + (1 << (bd + FILTER_BITS - 1)));
    let sh_h = _mm_cvtsi32_si128(round_0);
    for y in 0..ih {
        let row = hb + y * src_stride;
        // One checked view per row; every tile window inside provably fits
        // (`x0 <= w-16` => `x0+23 <= w+7`), so the inner loop carries no
        // bounds checks at all.
        let rowview: &[P] = &src[row..row + w + 7];
        let trowview: &mut [u16] = &mut temp[y * MAX_SB_SIZE..y * MAX_SB_SIZE + w];
        let mut xs = 0usize;
        loop {
            let x0 = xs.min(w - 16);
            // One bounds-checked 23-sample window; the k-offset subslices of a
            // fixed-length array carry statically-known lengths, so the eight
            // loads below compile to bare vmovdqu.
            let win: &[P; 23] = rowview[x0..x0 + 23].try_into().unwrap();
            let ld = |k: usize| -> __m256i {
                let a: &[P; 16] = win[k..k + 16].try_into().unwrap();
                P::LD16(_t, a)
            };
            // res_even: outputs x0+2i; res_odd: outputs x0+2i+1.
            let e = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(ld(0), ch[0]),
                    _mm256_madd_epi16(ld(4), ch[2]),
                ),
                _mm256_add_epi32(
                    _mm256_madd_epi16(ld(2), ch[1]),
                    _mm256_madd_epi16(ld(6), ch[3]),
                ),
            );
            let o = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(ld(1), ch[0]),
                    _mm256_madd_epi16(ld(5), ch[2]),
                ),
                _mm256_add_epi32(
                    _mm256_madd_epi16(ld(3), ch[1]),
                    _mm256_madd_epi16(ld(7), ch[3]),
                ),
            );
            let e = _mm256_sra_epi32(_mm256_add_epi32(e, rc_h), sh_h);
            let o = _mm256_sra_epi32(_mm256_add_epi32(o, rc_h), sh_h);
            let r = _mm256_min_epi16(_mm256_max_epi16(_mm256_packs_epi32(e, o), zero), hi_h);
            let out: &mut [u16; 16] = (&mut trowview[x0..x0 + 16]).try_into().unwrap();
            _mm256_storeu_si256(out, r);
            if x0 + 16 >= w {
                break;
            }
            xs += 16;
        }
    }

    // ---- vertical pass: 8 row loads + row-pair unpacks per 16 outputs; the
    // packs-order temp layout is undone by the final packs_epi32 ----
    let rc_v = _mm256_set1_epi32((1 << (round_1 - 1)) - (1 << (bd + round_1 - 1)));
    let sh_v = _mm_cvtsi32_si128(round_1);
    let hi_v = _mm256_set1_epi16(((1 << bd) - 1) as i16);
    for y in 0..h {
        // Same hoist as the horizontal pass: one checked view per row over
        // the eight temp rows and the dst row, so every per-tile window
        // inside is statically provable (`x0 <= w-16`).
        let tview: &[u16] =
            &temp[y * MAX_SB_SIZE..y * MAX_SB_SIZE + 7 * MAX_SB_SIZE + w];
        let dview: &mut [u16] = &mut dst[dst_off + y * dst_stride..dst_off + y * dst_stride + w];
        let mut xs = 0usize;
        loop {
            let x0 = xs.min(w - 16);
            // One checked view per tile whose length is statically known, so
            // the eight literal-offset row loads below carry no bounds checks.
            let tstrip: &[u16; 7 * MAX_SB_SIZE + 16] =
                tview[x0..x0 + 7 * MAX_SB_SIZE + 16].try_into().unwrap();
            let d = |k: usize| -> __m256i {
                let a: &[u16; 16] =
                    tstrip[k * MAX_SB_SIZE..k * MAX_SB_SIZE + 16].try_into().unwrap();
                _mm256_loadu_si256(a)
            };
            let (d0, d1) = (d(0), d(1));
            let (d2, d3) = (d(2), d(3));
            let (d4, d5) = (d(4), d(5));
            let (d6, d7) = (d(6), d(7));
            let e = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(d0, d1), cv[0]),
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(d2, d3), cv[1]),
                ),
                _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(d4, d5), cv[2]),
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(d6, d7), cv[3]),
                ),
            );
            let o = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(d0, d1), cv[0]),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(d2, d3), cv[1]),
                ),
                _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(d4, d5), cv[2]),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(d6, d7), cv[3]),
                ),
            );
            let lo = _mm256_sra_epi32(
                _mm256_add_epi32(_mm256_unpacklo_epi32(e, o), rc_v),
                sh_v,
            );
            let hi = _mm256_sra_epi32(
                _mm256_add_epi32(_mm256_unpackhi_epi32(e, o), rc_v),
                sh_v,
            );
            let r = _mm256_min_epi16(
                _mm256_max_epi16(_mm256_packs_epi32(lo, hi), zero),
                hi_v,
            );
            let out: &mut [u16; 16] = (&mut dview[x0..x0 + 16]).try_into().unwrap();
            _mm256_storeu_si256(out, r);
            if x0 + 16 >= w {
                break;
            }
            xs += 16;
        }
    }
}

#[archmage::magetypes(define(i32x8), wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn wiener_impl<P: crate::restore::pick::LrPixel>(
    token: Token,
    src: &[P],
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
    assert!(
        w >= 8 && w <= MAX_SB_SIZE,
        "wiener: restoration-unit width {w} outside 8..={MAX_SB_SIZE} — the SIMD path \
         loads 8 lanes at a time and `temp` is strided by MAX_SB_SIZE"
    );
    let (round_0, round_1) = conv_params_wiener(bd);
    let intermediate_height = h + SUBPEL_TAPS - 1;
    debug_assert_eq!(temp.len(), intermediate_height * MAX_SB_SIZE);

    // One-bounds-check `[u16; 8]` fixed-array load + `as i32` widen (LLVM:
    // vpmovzxwd) instead of 8 checked scalar loads via `from_fn`; the
    // lane VALUES are identical, so the arithmetic is untouched.
    let widen = |s: &[P]| -> i32x8 {
        let a: [P; 8] = s[..8].try_into().unwrap();
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
fn wiener_scalar_into<P: crate::restore::pick::LrPixel>(
    src: &[P],
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
    assert!(
        w <= MAX_SB_SIZE,
        "wiener: restoration-unit width {w} exceeds MAX_SB_SIZE ({MAX_SB_SIZE}) — \
         `temp` is strided by MAX_SB_SIZE, so a wider unit would alias rows"
    );
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
            let rounding = (src_x[3].to_i32() << FILTER_BITS) + (1 << (bd + FILTER_BITS - 1));
            let mut sum = rounding;
            for k in 0..SUBPEL_TAPS {
                sum += src_x[k].to_i32() * hfilter[k] as i32;
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

/// The aarch64 NEON tier — a verbatim transcription of
/// `av1_highbd_wiener_convolve_add_src_neon`
/// (`av1/common/arm/highbd_wiener_convolve_neon.c`). C's NEON kernel is a
/// different structure from both the scalar port and the AVX2 mirror: it
/// exploits the wiener filter's symmetry, loading only the first four taps
/// (`vld1_s16`, +128 on tap 3 for the `<< FILTER_BITS` centre term), pairing
/// mirrored source rows/columns before a `vmlal_lane`/`vmlaq_lane`
/// widening-multiply accumulate, and narrowing with `vqrshrun` saturating
/// shifts. `i16` loads of the `u16` data are safe exactly as in C — source
/// samples are <= (1<<bd)-1 and the intermediate clamp keeps `temp` at
/// <= i16::MAX by construction (`conv_params_wiener`).
///
/// Two deviations from C's loop structure, both semantics-preserving:
/// `w % 8 != 0` is handled by overlapping the LAST 8-wide column block back
/// to `w - 8` (outputs are pure functions of the window — identical bytes,
/// no stores outside `w`), and `w < 8` keeps the dispatch-time scalar route.
/// Asymmetric filters (unreachable: wiener filters are symmetric by
/// construction, `filter[7] == 0` always) produce what C NEON produces,
/// not what the scalar port does — matching the oracle's priority.
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn wiener_impl_neon<P: crate::restore::pick::LrPixel>(
    t: archmage::NeonToken,
    src: &[P],
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
    let _ = t;
    use archmage::intrinsics::aarch64::*;
    assert!(
        w >= 8 && w <= MAX_SB_SIZE,
        "wiener: restoration-unit width {w} outside 8..={MAX_SB_SIZE} — the SIMD path \
         loads 8 lanes at a time and `temp` is strided by MAX_SB_SIZE"
    );
    let (round_0, round_1) = conv_params_wiener(bd);
    // C's shifts are compiled constants: the `highbd` variants use
    // WIENER_ROUND0_BITS / 2*FILTER_BITS-WIENER_ROUND0_BITS and `highbd_12`
    // shifts them by 2 — the same values conv_params_wiener returns for
    // bd<=10 / bd==12 respectively. For bd=11 (round_0=4) C NEON still uses
    // the base shifts — mirror that, constants and all.
    let extra: i32 = if bd == 12 { 2 } else { 0 };
    let h_shift = 3 + extra; // WIENER_ROUND0_BITS(+2)
    let v_shift = 11 - extra; // 2*FILTER_BITS - WIENER_ROUND0_BITS(-2)

    let x_taps = if hfilter[0] == 0 && hfilter[6] == 0 { 5usize } else { 7 };
    let y_taps = if vfilter[0] == 0 && vfilter[6] == 0 { 5usize } else { 7 };
    // First four taps + (1<<FILTER_BITS) folded into tap 3 — C's
    // `vld1_s16` + `vcreate_s16(128 << 48)`.
    let xf = vld1_s16(<&[i16; 4]>::try_from(&hfilter[..4]).unwrap());
    let xf = vadd_s16(xf, vcreate_s16(128u64 << 48));
    let yf = vld1_s16(<&[i16; 4]>::try_from(&vfilter[..4]).unwrap());
    let yf = vadd_s16(yf, vcreate_s16(128u64 << 48));
    // The vertical pass widens the filter to i32 lanes once.
    let yf32 = vmovl_s16(yf);
    let yf_lo = vget_low_s32(yf32);
    let yf_hi = vget_high_s32(yf32);

    let im_stride = MAX_SB_SIZE;
    let im_h = h + y_taps - 1;
    let horiz_off = x_taps / 2;
    let vert_off = (y_taps / 2) * src_stride;

    let clamp_limit = 1i32 << (bd + 1 + FILTER_BITS - round_0); // WIENER_CLAMP_LIMIT
    let im_max = vdupq_n_u16((clamp_limit - 1) as u16);
    let h_round = vdupq_n_s32(1 << (bd + FILTER_BITS - 1));
    let res_max = vdupq_n_u16(((1i32 << bd) - 1) as u16);
    let v_round = vdupq_n_s32(-(1 << (bd + round_1 - 1)));

    // ---- horizontal pass: u16 window rows -> clamped u16 im rows ----
    // im row 0 = src row `-vert_off`, im col 0 = src col `-horiz_off`.
    let ld = |s: &[P]| P::LD8_S16(t, <&[P; 8]>::try_from(&s[..8]).unwrap());
    let h_base = src_off as isize - vert_off as isize - horiz_off as isize;
    for y in 0..im_h {
        let row = (h_base + (y * src_stride) as isize) as usize;
        let mut xs = 0usize;
        loop {
            let x0 = xs.min(w - 8);
            let s = row + x0;
            let d = if x_taps == 5 {
                let (s0, s4) = (ld(&src[s..]), ld(&src[s + 4..]));
                let (s1, s3) = (ld(&src[s + 1..]), ld(&src[s + 3..]));
                let s2 = ld(&src[s + 2..]);
                let s04 = vaddq_s16(s0, s4);
                let s13 = vaddq_s16(s1, s3);
                let lo = vmlal_lane_s16::<1>(h_round, vget_low_s16(s04), xf);
                let lo = vmlal_lane_s16::<2>(lo, vget_low_s16(s13), xf);
                let lo = vmlal_lane_s16::<3>(lo, vget_low_s16(s2), xf);
                let hi = vmlal_lane_s16::<1>(h_round, vget_high_s16(s04), xf);
                let hi = vmlal_lane_s16::<2>(hi, vget_high_s16(s13), xf);
                let hi = vmlal_lane_s16::<3>(hi, vget_high_s16(s2), xf);
                let res = match h_shift {
                    3 => vcombine_u16(vqrshrun_n_s32::<3>(lo), vqrshrun_n_s32::<3>(hi)),
                    _ => vcombine_u16(vqrshrun_n_s32::<5>(lo), vqrshrun_n_s32::<5>(hi)),
                };
                vminq_u16(res, im_max)
            } else {
                let (s0, s6) = (ld(&src[s..]), ld(&src[s + 6..]));
                let (s1, s5) = (ld(&src[s + 1..]), ld(&src[s + 5..]));
                let (s2, s4) = (ld(&src[s + 2..]), ld(&src[s + 4..]));
                let s3 = ld(&src[s + 3..]);
                let s06 = vaddq_s16(s0, s6);
                let s15 = vaddq_s16(s1, s5);
                let s24 = vaddq_s16(s2, s4);
                let lo = vmlal_lane_s16::<0>(h_round, vget_low_s16(s06), xf);
                let lo = vmlal_lane_s16::<1>(lo, vget_low_s16(s15), xf);
                let lo = vmlal_lane_s16::<2>(lo, vget_low_s16(s24), xf);
                let lo = vmlal_lane_s16::<3>(lo, vget_low_s16(s3), xf);
                let hi = vmlal_lane_s16::<0>(h_round, vget_high_s16(s06), xf);
                let hi = vmlal_lane_s16::<1>(hi, vget_high_s16(s15), xf);
                let hi = vmlal_lane_s16::<2>(hi, vget_high_s16(s24), xf);
                let hi = vmlal_lane_s16::<3>(hi, vget_high_s16(s3), xf);
                let res = match h_shift {
                    3 => vcombine_u16(vqrshrun_n_s32::<3>(lo), vqrshrun_n_s32::<3>(hi)),
                    _ => vcombine_u16(vqrshrun_n_s32::<5>(lo), vqrshrun_n_s32::<5>(hi)),
                };
                vminq_u16(res, im_max)
            };
            vst1q_u16(
                <&mut [u16; 8]>::try_from(&mut temp[y * im_stride + x0..y * im_stride + x0 + 8])
                    .unwrap(),
                d,
            );
            if x0 + 8 >= w {
                break;
            }
            xs += 8;
        }
    }

    // ---- vertical pass: 5/7 im rows -> clamped u16 output rows ----
    let vrow = |s: &[u16], y: usize, x0: usize| {
        vreinterpretq_s16_u16(vld1q_u16(
            <&[u16; 8]>::try_from(&s[y * im_stride + x0..y * im_stride + x0 + 8]).unwrap(),
        ))
    };
    // One 8-lane vertical output from rows s0.. — shared by both tap counts.
    macro_rules! vtap5 {
        ($s0:expr, $s1:expr, $s2:expr, $s3:expr, $s4:expr) => {{
            let s04_lo = vaddl_s16(vget_low_s16($s0), vget_low_s16($s4));
            let s13_lo = vaddl_s16(vget_low_s16($s1), vget_low_s16($s3));
            let lo = vmlaq_lane_s32::<1>(v_round, s04_lo, yf_lo);
            let lo = vmlaq_lane_s32::<0>(lo, s13_lo, yf_hi);
            let lo = vmlaq_lane_s32::<1>(lo, vmovl_s16(vget_low_s16($s2)), yf_hi);
            let s04_hi = vaddl_s16(vget_high_s16($s0), vget_high_s16($s4));
            let s13_hi = vaddl_s16(vget_high_s16($s1), vget_high_s16($s3));
            let hi = vmlaq_lane_s32::<1>(v_round, s04_hi, yf_lo);
            let hi = vmlaq_lane_s32::<0>(hi, s13_hi, yf_hi);
            let hi = vmlaq_lane_s32::<1>(hi, vmovl_s16(vget_high_s16($s2)), yf_hi);
            let res = match v_shift {
                9 => vcombine_u16(vqrshrun_n_s32::<9>(lo), vqrshrun_n_s32::<9>(hi)),
                _ => vcombine_u16(vqrshrun_n_s32::<11>(lo), vqrshrun_n_s32::<11>(hi)),
            };
            vminq_u16(res, res_max)
        }};
    }
    macro_rules! vtap7 {
        ($s0:expr, $s1:expr, $s2:expr, $s3:expr, $s4:expr, $s5:expr, $s6:expr) => {{
            let s06_lo = vaddl_s16(vget_low_s16($s0), vget_low_s16($s6));
            let s15_lo = vaddl_s16(vget_low_s16($s1), vget_low_s16($s5));
            let s24_lo = vaddl_s16(vget_low_s16($s2), vget_low_s16($s4));
            let lo = vmlaq_lane_s32::<0>(v_round, s06_lo, yf_lo);
            let lo = vmlaq_lane_s32::<1>(lo, s15_lo, yf_lo);
            let lo = vmlaq_lane_s32::<0>(lo, s24_lo, yf_hi);
            let lo = vmlaq_lane_s32::<1>(lo, vmovl_s16(vget_low_s16($s3)), yf_hi);
            let s06_hi = vaddl_s16(vget_high_s16($s0), vget_high_s16($s6));
            let s15_hi = vaddl_s16(vget_high_s16($s1), vget_high_s16($s5));
            let s24_hi = vaddl_s16(vget_high_s16($s2), vget_high_s16($s4));
            let hi = vmlaq_lane_s32::<0>(v_round, s06_hi, yf_lo);
            let hi = vmlaq_lane_s32::<1>(hi, s15_hi, yf_lo);
            let hi = vmlaq_lane_s32::<0>(hi, s24_hi, yf_hi);
            let hi = vmlaq_lane_s32::<1>(hi, vmovl_s16(vget_high_s16($s3)), yf_hi);
            let res = match v_shift {
                9 => vcombine_u16(vqrshrun_n_s32::<9>(lo), vqrshrun_n_s32::<9>(hi)),
                _ => vcombine_u16(vqrshrun_n_s32::<11>(lo), vqrshrun_n_s32::<11>(hi)),
            };
            vminq_u16(res, res_max)
        }};
    }
    let mut xs = 0usize;
    loop {
        let x0 = xs.min(w - 8);
        let mut y = 0usize;
        while y + 4 <= h {
            let r = |k: usize| vrow(temp, y + k, x0);
            let (d0, d1, d2, d3) = if y_taps == 5 {
                (
                    vtap5!(r(0), r(1), r(2), r(3), r(4)),
                    vtap5!(r(1), r(2), r(3), r(4), r(5)),
                    vtap5!(r(2), r(3), r(4), r(5), r(6)),
                    vtap5!(r(3), r(4), r(5), r(6), r(7)),
                )
            } else {
                (
                    vtap7!(r(0), r(1), r(2), r(3), r(4), r(5), r(6)),
                    vtap7!(r(1), r(2), r(3), r(4), r(5), r(6), r(7)),
                    vtap7!(r(2), r(3), r(4), r(5), r(6), r(7), r(8)),
                    vtap7!(r(3), r(4), r(5), r(6), r(7), r(8), r(9)),
                )
            };
            for (dy, dv) in [d0, d1, d2, d3].iter().enumerate() {
                vst1q_u16(
                    <&mut [u16; 8]>::try_from(
                        &mut dst[dst_off + (y + dy) * dst_stride + x0
                            ..dst_off + (y + dy) * dst_stride + x0 + 8],
                    )
                    .unwrap(),
                    *dv,
                );
            }
            y += 4;
        }
        while y < h {
            let r = |k: usize| vrow(temp, y + k, x0);
            let d = if y_taps == 5 {
                vtap5!(r(0), r(1), r(2), r(3), r(4))
            } else {
                vtap7!(r(0), r(1), r(2), r(3), r(4), r(5), r(6))
            };
            vst1q_u16(
                <&mut [u16; 8]>::try_from(
                    &mut dst[dst_off + y * dst_stride + x0..dst_off + y * dst_stride + x0 + 8],
                )
                .unwrap(),
                d,
            );
            y += 1;
        }
        if x0 + 8 >= w {
            break;
        }
        xs += 8;
    }
}
