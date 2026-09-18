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


/// Build the two SGR integral images over the `[-3, +3)`-extended source —
/// `integral_images_highbd` (selfguided_sse4.c): `ii_sum[y][x]` is the sum of
/// `src[0..y][0..x]`, `ii_sq[y][x]` the sum of squares. Row 0 and column 0
/// are zero, so the planes have `(height + 1) x (width + 1)` live cells.
///
/// # Bit-exactness under i32 wraparound
///
/// A cell of `ii_sq` can exceed `i32::MAX` at high bit depth (a 70x70 unit at
/// bd12 holds up to ~8.2e10), so every accumulation is `wrapping_*`. The
/// four-corner `boxsum_ii` lookup takes DIFFERENCES of these cells; wraparound
/// cancels mod 2^32, and every true `(2r+1)^2` box sum is below 2^31, so the
/// wrapped arithmetic yields exactly the sum the old rolling-window boxsum
/// computed — just evaluated in a different (associativity-irrelevant) order.
///
/// # Why this replaced the rolling boxsum
///
/// The C scalar reference runs a vertical then a horizontal sliding window
/// per boxsum array (two passes for A and two for B, over an i32 staging
/// plane the C never materialises). C's SSE4/AVX2 kernels instead build the
/// two integral images in ONE pass over the raw pixel plane and get every
/// box sum from four corner loads. This ports that structure.
#[allow(clippy::too_many_arguments)]
fn integral_image_impl_scalar(
    _t: archmage::ScalarToken,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    width: usize,
    height: usize,
    ii_sq: &mut [i32],
    ii_sum: &mut [i32],
    ii_stride: usize,
) {
    ii_sq[..width + 1].fill(0);
    ii_sum[..width + 1].fill(0);
    for i in 0..height {
        let (above, cur) = (i * ii_stride, (i + 1) * ii_stride);
        let s0 = src_off + i * src_stride;
        ii_sq[cur] = 0;
        ii_sum[cur] = 0;
        let mut rs = 0i32;
        let mut rq = 0i32;
        for x in 0..width {
            let v = src[s0 + x] as i32;
            rs = rs.wrapping_add(v);
            rq = rq.wrapping_add(v.wrapping_mul(v));
            ii_sum[cur + 1 + x] = ii_sum[above + 1 + x].wrapping_add(rs);
            ii_sq[cur + 1 + x] = ii_sq[above + 1 + x].wrapping_add(rq);
        }
    }
}

/// x86-64/AVX2 body for [`integral_image`] — a transcription of
/// `integral_images_highbd` (selfguided_sse4.c): four i32 lanes per step, the
/// horizontal prefix computed by `scan_32` (byte-shift + add) plus the
/// `ldiff` carry. `_mm_madd_epi16(x, x)` squares each lane — the source
/// values are pixels (`<= (1 << bit_depth) - 1 <= 4095`), so the u16 lanes
/// are i16-positive and the madd result is the exact square.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn integral_image_impl_v3(
    _t: archmage::X64V3Token,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    width: usize,
    height: usize,
    ii_sq: &mut [i32],
    ii_sum: &mut [i32],
    ii_stride: usize,
) {
    use archmage::intrinsics::x86_64::*;
    ii_sq[..width + 1].fill(0);
    ii_sum[..width + 1].fill(0);
    for i in 0..height {
        let (above, cur) = (i * ii_stride, (i + 1) * ii_stride);
        let s0 = src_off + i * src_stride;
        ii_sq[cur] = 0;
        ii_sum[cur] = 0;
        // Tight row spans (`width` source cells, `width + 1` ii cells — the
        // stride is larger), so `span[j + c]` under `j + 4 <= width` is
        // provably in bounds and no per-load check survives into the loop.
        let srow = &src[s0..s0 + width];
        let (sm_a, sm_c) = ii_sum.split_at_mut(cur);
        let abv1 = &sm_a[above..above + width + 1];
        let dst1 = &mut sm_c[..width + 1];
        let (sq_a, sq_c) = ii_sq.split_at_mut(cur);
        let abv2 = &sq_a[above..above + width + 1];
        let dst2 = &mut sq_c[..width + 1];
        let mut ldiff1 = _mm_setzero_si128();
        let mut ldiff2 = _mm_setzero_si128();
        // `chunks_exact(4)` zips make every lane access provably in-range —
        // the four ii-row slices' `try_into`s compiled to per-iteration
        // `cmp`/`ja` bounds branches even after the tight re-slice, because
        // LLVM cannot relate a `split_at_mut` subslice's length back to the
        // `j + 4 <= width` guard. Zipped equal-length chunk iterators carry
        // the proof structurally instead.
        let mut j = 0usize;
        for ((((av1, av2), px), d1), d2) in abv1[1..]
            .chunks_exact(4)
            .zip(abv2[1..].chunks_exact(4))
            .zip(srow.chunks_exact(4))
            .zip(dst1[1..].chunks_exact_mut(4))
            .zip(dst2[1..].chunks_exact_mut(4))
        {
            let above1: &[i32; 4] = av1.try_into().unwrap();
            let above1 = _mm_loadu_si128(above1);
            let above2: &[i32; 4] = av2.try_into().unwrap();
            let above2 = _mm_loadu_si128(above2);

            let x1 = _mm_set_epi32(
                px[3] as i32,
                px[2] as i32,
                px[1] as i32,
                px[0] as i32,
            );
            let x2 = _mm_madd_epi16(x1, x1);

            let x01 = _mm_add_epi32(x1, _mm_slli_si128::<4>(x1));
            let sc1 = _mm_add_epi32(x01, _mm_slli_si128::<8>(x01));
            let y01 = _mm_add_epi32(x2, _mm_slli_si128::<4>(x2));
            let sc2 = _mm_add_epi32(y01, _mm_slli_si128::<8>(y01));

            let row1 = _mm_add_epi32(_mm_add_epi32(sc1, above1), ldiff1);
            let row2 = _mm_add_epi32(_mm_add_epi32(sc2, above2), ldiff2);

            let d1: &mut [i32; 4] = d1.try_into().unwrap();
            _mm_storeu_si128(d1, row1);
            let d2: &mut [i32; 4] = d2.try_into().unwrap();
            _mm_storeu_si128(d2, row2);

            ldiff1 = _mm_shuffle_epi32::<0xff>(_mm_sub_epi32(row1, above1));
            ldiff2 = _mm_shuffle_epi32::<0xff>(_mm_sub_epi32(row2, above2));
            j += 4;
        }
        // Column tail: continue the running prefix. `cur - above` at column
        // `j` recovers the horizontal prefix through source column `j - 1`
        // (for `j == 0` both cells are zero, so the seed is 0 either way).
        let mut rs = dst1[j].wrapping_sub(abv1[j]);
        let mut rq = dst2[j].wrapping_sub(abv2[j]);
        for x in j..width {
            let v = srow[x] as i32;
            rs = rs.wrapping_add(v);
            rq = rq.wrapping_add(v.wrapping_mul(v));
            dst1[1 + x] = abv1[1 + x].wrapping_add(rs);
            dst2[1 + x] = abv2[1 + x].wrapping_add(rq);
        }
    }
}

/// Dispatch the integral-image build.
#[allow(clippy::too_many_arguments)]
fn integral_image(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    width: usize,
    height: usize,
    ii_sq: &mut [i32],
    ii_sum: &mut [i32],
    ii_stride: usize,
) {
    archmage::incant!(
        integral_image_impl(
            src, src_off, src_stride, width, height, ii_sq, ii_sum, ii_stride
        ),
        [v3, scalar]
    );
}

/// One `(A, B)` cell of the SGR intermediate — the loop body of
/// `av1_selfguided_restoration_c`'s A/B pass.
///
/// Shared VERBATIM by the scalar tier and by the vector tier's column tail, so
/// the two cannot drift in the one place a drift would be invisible to a
/// lane-level review (the `gather_window` precedent in `restore::pick`).
#[inline(always)]
fn ab_one(
    a_raw: i32,
    b_raw: i32,
    n: u32,
    s: u32,
    one_by_x: u32,
    shift_a: u32,
    shift_b: u32,
) -> (i32, i32) {
    let a = rpot_u32(a_raw as u32, shift_a);
    let b = rpot_u32(b_raw as u32, shift_b);
    // C: `p = (a * n < b * b) ? 0 : a * n - b * b` (the highbd
    // rounding artefact saturation).
    let p = (a * n).saturating_sub(b * b);
    // p * s < 2^32 for the valid s table (see the C bound comments);
    // wrapping matches C uint32 semantics exactly regardless.
    let z = rpot_u32(p.wrapping_mul(s), SGRPROJ_MTABLE_BITS);
    let a_out = X_BY_XPLUS1[z.min(255) as usize];
    let b_out = rpot_u32(
        ((SGRPROJ_SGR - a_out) as u32)
            .wrapping_mul(b_raw as u32)
            .wrapping_mul(one_by_x),
        SGRPROJ_RECIP_BITS,
    ) as i32;
    (a_out, b_out)
}

/// Raw box sums out of the integral images — `boxsum_from_ii`
/// (selfguided_sse4.c). For a `(2r+1)`-sided window centred at
/// extended-source pixel `(y, x)` the sum is
/// `ii[y+r+1][x+r+1] - ii[y-r][x+r+1] - ii[y+r+1][x-r] + ii[y-r][x-r]`.
/// Shared VERBATIM by the scalar tier and by the vector tier's column tail.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn boxsum_ii_px(ii: &[i32], yt: usize, yb: usize, x: usize, r: usize) -> i32 {
    (ii[yb + x + r + 1].wrapping_sub(ii[yt + x + r + 1]))
        .wrapping_sub(ii[yb + x - r].wrapping_sub(ii[yt + x - r]))
}

/// Scalar tier of the fused pass — the transcribed boxsum feeding `ab_one`.
#[allow(clippy::too_many_arguments)]
fn calc_ab_impl_scalar(
    _t: archmage::ScalarToken,
    ii_sq: &[i32],
    ii_sum: &[i32],
    ii_stride: usize,
    y0: usize,
    ystep: usize,
    rows: usize,
    x0: usize,
    count: usize,
    r: usize,
    n: u32,
    s: u32,
    one_by_x: u32,
    shift_a: u32,
    shift_b: u32,
    a_buf: &mut [i32],
    b_buf: &mut [i32],
    ab_off: usize,
    abstep: usize,
) {
    for t in 0..rows {
        let y = y0 + t * ystep;
        let yt = (y - r) * ii_stride;
        let yb = (y + r + 1) * ii_stride;
        let ab = ab_off + t * abstep;
        let a_row = &mut a_buf[ab..ab + count];
        let b_row = &mut b_buf[ab..ab + count];
        for u in 0..count {
            let x = x0 + u;
            let (a_out, b_out) = ab_one(
                boxsum_ii_px(ii_sq, yt, yb, x, r),
                boxsum_ii_px(ii_sum, yt, yb, x, r),
                n,
                s,
                one_by_x,
                shift_a,
                shift_b,
            );
            a_row[u] = a_out;
            b_row[u] = b_out;
        }
    }
}

// The two shift amounts below are hardcoded as const generics; these pin the
// named constants they stand for, so a future change to either breaks the
// build instead of silently changing the rounding.
const _: () = assert!(SGRPROJ_MTABLE_BITS == 20);
const _: () = assert!(SGRPROJ_RECIP_BITS == 12);

/// Vector tier: lanes are columns `j` of one A/B row.
///
/// # Why this is bit-exact for EVERY bit pattern, with no range argument
///
/// Each `j` is independent — no lane reads another lane's output — so nothing
/// is reassociated at all (contrast `restore::pick`'s four-pixel fold, which
/// does reassociate and argues from the associativity of wrapping `i32` adds).
/// The lane arithmetic is C's `uint32_t` arithmetic exactly:
///
/// * `+`, `-` and `*` are the low 32 bits, which is what `wrapping_add` /
///   `wrapping_sub` / `wrapping_mul` on `u32` compute — signedness cannot
///   change a low-32 result;
/// * every right shift is `shr_logical`, i.e. the `u32` shift, never the
///   arithmetic one;
/// * `saturating_sub` is the one place signedness WOULD matter, so it does not
///   rely on a bound: the compare is made unsigned by flipping both operands'
///   sign bits (`x ^ i32::MIN`), which turns signed `>=` into `>=` on the u32
///   bit patterns for all inputs;
/// * `z.min(255)` is the one signed compare kept, and it is unconditionally
///   safe: `z` is a logical shift right by 20 of a 32-bit value, so it is in
///   `0 ..= 4095` and can never be negative.
///
/// So the A/B buffers hold the same bits the scalar tier writes, whatever the
/// box sums contain.
///
/// One asymmetry, stated because a tier disagreement is a differential hole
/// even when it is unreachable: the scalar tier spells `a * n` and `v + half`
/// as CHECKED Rust operations (it is the verbatim transcription, and C writes
/// them on `uint32_t`), while these lanes wrap. A `debug-assertions` build
/// would therefore panic in the scalar tier exactly where this one wraps. That
/// input does not exist: `a` is a box sum of squares shifted right by
/// `2 * (bit_depth - 8)`, so it is at most `25 * 4095^2 >> 8` ~ 1.64e6 and
/// `a * n` at most ~4.1e7, three orders under `u32::MAX`; `p * s` is already
/// spelled `wrapping_mul` on both sides. In a release build the scalar tier
/// wraps too, so the tiers agree unconditionally there.
///
/// # The table lookup, and why it did NOT need a gather
///
/// `X_BY_XPLUS1[z.min(255)]` is the one operation with no vector form here:
/// **magetypes has no gather at all** (checked in the pinned 0.9.28 AND in
/// 0.9.29) and no shuffle/permute either, only `blend` — so neither a
/// `vpgatherdd`-style lookup nor a `pshufb` in-register LUT is expressible.
///
/// It does not need one. The lookup is ONE operation of roughly twenty in this
/// loop body; it was never slow, it was *blocking vectorization of the other
/// nineteen*. So it stays eight scalar loads through a stack round trip and
/// everything around it goes eight-wide. Two things this deliberately avoids:
///
/// * a real gather would be **AVX2-only** (aarch64 has none before SVE2,
///   wasm128 none), and `VPGATHERDD` is ~12-20 cycles of throughput for 8 lanes
///   against 8 L1 loads at ~0.5 each — for a 1 KB permanently-L1-resident table
///   it would most likely LOSE to the scalar loads it replaced;
/// * an arithmetic reformulation IS available and IS provable — the table is
///   `round(256z/(z+1))` at 254 of its 256 entries, with two deliberate
///   endpoints (`z=0 -> 1`, libaom's "value of 1/256"; `z=255 -> 256`), and an
///   f32 form (`256.0/(z+1)`, `+0.5`, floor, `256 - r`, two endpoint blends) is
///   exact on **all 256 inputs**, the nearest half-integer being 0.002924 away
///   against an f32 error of ~1.5e-5. It is the follow-up if the store-forward
///   round trip ever dominates. It must use true `Div`, never `rcp_approx`:
///   approximate reciprocal is only specified to a relative-error bound and its
///   exact bits differ between vendors, which would make the encoder's output
///   depend on whose CPU ran it.
///
/// `& 255` on the stored index is a no-op after the clamp and makes the index
/// provably in range, so no bounds check survives into the loop.
///
/// # Measured
///
/// Two binaries from one tree, arms interleaved and ROTATED, a same-binary null
/// arm in every band, byte-identical output on every arm:
///
/// | cell | before | after | vs libaom | paired | rounds | p | null |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | 192x192 cq27 s0 | 441.19 ms | 436.13 ms | 2.5039x -> 2.4752x | -1.20 % | 20/20 | 1.9e-6 | +0.10 % (p=0.50) |
/// | 1024x1024 cq27 s0 | 10615.10 ms | 10424.88 ms | 2.5775x -> 2.5313x | -2.00 % | 8/8 | 0.008 | -0.24 % (p=0.29) |
///
/// Attributed by symbol on the 1024x1024 cell: the A/B pass was inlined into
/// `calculate_intermediate` at **2.71 % = 287.7 ms**; it is now
/// `ab_row_impl_v3` at 0.63 % = 65.7 ms plus a 0.05 % = 5.2 ms residual, i.e.
/// **287.7 -> 70.9 ms, 4.1x**, and it leaves the loop-restoration symbol list
/// altogether. That is -216.8 ms attributed against a -190.2 ms wall delta;
/// the box-sum closure reads +8.8 ms in the same single-run profile, which is
/// within one-sample noise. **The wall band (8 rounds, 8/8) is the ground
/// truth — the symbol split attributes the work that MOVED, and is not a
/// per-symbol measurement of everything that did not** (reading it that way is
/// the KB-PERF-6 roll-up error).
///
/// The lookup itself was never the cost, which is the point: it is one
/// operation of about twenty here, and eight scalar loads through a stack round
/// trip cost less than leaving the other nineteen scalar.
/// Vector tier of the fused pass — `calc_ab` (selfguided_sse4.c): the
/// four-corner `boxsum_from_ii` feeds the A/B transform directly, so the raw
/// box sums never touch memory. Lanes are columns `j` of one A/B row.
///
/// The eight corner spans are re-sliced to exactly `count` elements up front,
/// so every `span[j..j + 8]` under the `j + 8 <= count` loop guard is
/// provably in bounds — no per-load check survives into the loop (the crate
/// is `#![forbid(unsafe_code)]`, so elimination has to be structural).
#[archmage::magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn calc_ab_impl(
    token: Token,
    ii_sq: &[i32],
    ii_sum: &[i32],
    ii_stride: usize,
    y0: usize,
    ystep: usize,
    rows: usize,
    x0: usize,
    count: usize,
    r: usize,
    n: u32,
    s: u32,
    one_by_x: u32,
    shift_a: u32,
    shift_b: u32,
    a_buf: &mut [i32],
    b_buf: &mut [i32],
    ab_off: usize,
    abstep: usize,
) {
    // The two `ROUND_POWER_OF_TWO` shifts are frame-level constants but not
    // COMPILE-time ones, and this vocabulary has only const-generic shifts
    // (no `shr_logical_uniform` in 0.9.28). `bit_depth` is 8, 10 or 12, so
    // `shift_a` is 0/4/8 and `shift_b` is 0/2/4: each is a select over three
    // const shifts, with the masks hoisted out of the loop.
    let all = i32x8::splat(token, -1);
    let none = i32x8::zero(token);
    let m_a4 = if shift_a >= 4 { all } else { none };
    let m_a8 = if shift_a >= 8 { all } else { none };
    let m_b2 = if shift_b >= 2 { all } else { none };
    let m_b4 = if shift_b >= 4 { all } else { none };

    let nv = i32x8::splat(token, n as i32);
    let sv = i32x8::splat(token, s as i32);
    let obx = i32x8::splat(token, one_by_x as i32);
    let sgr = i32x8::splat(token, SGRPROJ_SGR);
    let half_a = i32x8::splat(token, ((1u32 << shift_a) >> 1) as i32);
    let half_b = i32x8::splat(token, ((1u32 << shift_b) >> 1) as i32);
    let half_z = i32x8::splat(token, ((1u32 << SGRPROJ_MTABLE_BITS) >> 1) as i32);
    let half_r = i32x8::splat(token, ((1u32 << SGRPROJ_RECIP_BITS) >> 1) as i32);
    let c255 = i32x8::splat(token, 255);
    // Flipping the sign bit of both operands turns signed `>=` into `>=` on the
    // u32 bit patterns — the unsigned compare `saturating_sub` needs.
    let bias = i32x8::splat(token, i32::MIN);

    for t in 0..rows {
        let y = y0 + t * ystep;
        let yt = (y - r) * ii_stride;
        let yb = (y + r + 1) * ii_stride;
        let xl = x0 - r;
        let xr = x0 + r + 1;
        let sq_tl = &ii_sq[yt + xl..yt + xl + count];
        let sq_tr = &ii_sq[yt + xr..yt + xr + count];
        let sq_bl = &ii_sq[yb + xl..yb + xl + count];
        let sq_br = &ii_sq[yb + xr..yb + xr + count];
        let sm_tl = &ii_sum[yt + xl..yt + xl + count];
        let sm_tr = &ii_sum[yt + xr..yt + xr + count];
        let sm_bl = &ii_sum[yb + xl..yb + xl + count];
        let sm_br = &ii_sum[yb + xr..yb + xr + count];
        let ab = ab_off + t * abstep;
        let a_row = &mut a_buf[ab..ab + count];
        let b_row = &mut b_buf[ab..ab + count];

        let mut o = 0usize;
        while o + 8 <= count {
            let a_raw = (i32x8::from_slice(token, &sq_br[o..o + 8])
                - i32x8::from_slice(token, &sq_tr[o..o + 8]))
                - (i32x8::from_slice(token, &sq_bl[o..o + 8])
                    - i32x8::from_slice(token, &sq_tl[o..o + 8]));
            let b_raw = (i32x8::from_slice(token, &sm_br[o..o + 8])
                - i32x8::from_slice(token, &sm_tr[o..o + 8]))
                - (i32x8::from_slice(token, &sm_bl[o..o + 8])
                    - i32x8::from_slice(token, &sm_tl[o..o + 8]));

            // a = ROUND_POWER_OF_TWO(a_raw, shift_a), b likewise, in u32.
            let xa = a_raw + half_a;
            let a = i32x8::blend(
                m_a8,
                xa.shr_logical::<8>(),
                i32x8::blend(m_a4, xa.shr_logical::<4>(), xa),
            );
            let xb = b_raw + half_b;
            let b = i32x8::blend(
                m_b4,
                xb.shr_logical::<4>(),
                i32x8::blend(m_b2, xb.shr_logical::<2>(), xb),
            );

            // p = (a * n).saturating_sub(b * b), on the u32 bit patterns.
            let an = a * nv;
            let bb = b * b;
            let ge = (an ^ bias).simd_ge(bb ^ bias);
            let p = i32x8::blend(ge, an - bb, none);

            // z = ROUND_POWER_OF_TWO(p.wrapping_mul(s), SGRPROJ_MTABLE_BITS).
            let zc = (((p * sv) + half_z).shr_logical::<20>()).min(c255);

            let mut zs = [0i32; 8];
            zc.store(&mut zs);
            let mut lut = [0i32; 8];
            for t in 0..8 {
                lut[t] = X_BY_XPLUS1[(zs[t] as usize) & 255];
            }
            let a_out = i32x8::from_slice(token, &lut[..]);

            let b_out = (((sgr - a_out) * b_raw * obx) + half_r).shr_logical::<12>();

            {
                let d: &mut [i32; 8] = (&mut a_row[o..o + 8]).try_into().unwrap();
                a_out.store(d);
            }
            {
                let d: &mut [i32; 8] = (&mut b_row[o..o + 8]).try_into().unwrap();
                b_out.store(d);
            }
            o += 8;
        }

        // Column tail: the shared scalar body, so the tiers cannot drift.
        for t in o..count {
            let x = x0 + t;
            let (a_out, b_out) = ab_one(
                boxsum_ii_px(ii_sq, yt, yb, x, r),
                boxsum_ii_px(ii_sum, yt, yb, x, r),
                n,
                s,
                one_by_x,
                shift_a,
                shift_b,
            );
            a_row[t] = a_out;
            b_row[t] = b_out;
        }
    }
}

/// `calc_ab` over a run of rows of the ring — fused boxsum + A/B transform.
/// Row `t` of the run reads ii row `y0 + t * ystep` and writes
/// `a_buf[ab_off + t * abstep .. + count]` (likewise `b_buf`) — one dispatch
/// and one constant setup per run rather than per row.
#[allow(clippy::too_many_arguments)]
fn calc_ab(
    ii_sq: &[i32],
    ii_sum: &[i32],
    ii_stride: usize,
    y0: usize,
    ystep: usize,
    rows: usize,
    x0: usize,
    count: usize,
    r: usize,
    n: u32,
    s: u32,
    one_by_x: u32,
    shift_a: u32,
    shift_b: u32,
    a_buf: &mut [i32],
    b_buf: &mut [i32],
    ab_off: usize,
    abstep: usize,
) {
    archmage::incant!(
        calc_ab_impl(
            ii_sq, ii_sum, ii_stride, y0, ystep, rows, x0, count, r, n, s, one_by_x,
            shift_a, shift_b, a_buf, b_buf, ab_off, abstep
        ),
        [v3, neon, wasm128, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
fn calculate_intermediate(
    ii_sq: &[i32],
    ii_sum: &[i32],
    ii_stride: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
    ep: usize,
    radius_idx: usize,
    pass: usize,
    a_buf: &mut Vec<i32>,
    b_buf: &mut Vec<i32>,
) -> (usize, usize) {
    let (rads, ss) = SGR_PARAMS[ep];
    let r = rads[radius_idx] as usize;
    let width_ext = width + 2 * SGRPROJ_BORDER_HORZ;
    let height_ext = height + 2 * SGRPROJ_BORDER_VERT;
    // "Adjusting the stride of A and B here appears to avoid bad cache
    // effects" — must match the C exactly (it changes nothing numerically,
    // but keep the layout for clarity). The integral images share it.
    let buf_stride = ii_stride;
    debug_assert_eq!(buf_stride, ((width_ext + 3) & !3) + 16);
    let step = if pass == 0 { 1 } else { 2 };
    // `resize` without clear: every cell `ab_row` reads was written by
    // `boxsum_ii_row` just above it, so stale pool contents are unreachable —
    // the same guarantee C gets from `aom_malloc`.
    a_buf.resize(buf_stride * (height_ext + 1), 0);
    b_buf.resize(buf_stride * (height_ext + 1), 0);

    let org = SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    // A[] / B[] with a 1-pixel ring: i in -1 ..= height, j in -1 ..= width.
    let n = ((2 * r + 1) * (2 * r + 1)) as u32;
    let s = ss[radius_idx] as u32;
    let one_by_x = ONE_BY_X[(n - 1) as usize];
    let shift_a = 2 * (bit_depth - 8) as u32;
    let shift_b = (bit_depth - 8) as u32;
    // `k` is contiguous in `j`, so each row of the ring is one contiguous run
    // of `width + 2` cells in each buffer (j = -1 ..= width). Only the rows
    // the A/B pass touches get box sums — the integral image makes skipped
    // rows free, where the old vert/horz boxsum had to cover the plane.
    let count = width + 2;
    // Ring rows are i = -1, -1 + step, ... while i < height + 1 — one
    // `calc_ab` run for all of them so the dispatch and constant setup
    // happen once per pass rather than once per row.
    let rows = (height + 2).div_ceil(step);
    calc_ab(
        ii_sq,
        ii_sum,
        ii_stride,
        SGRPROJ_BORDER_VERT - 1,
        step,
        rows,
        2,
        count,
        r,
        n,
        s,
        one_by_x,
        shift_a,
        shift_b,
        a_buf,
        b_buf,
        org - buf_stride - 1,
        step * buf_stride,
    );
    (buf_stride, org)
}

/// Per-thread SGR work buffers — C's `rst->tmpbuf`/`buf` equivalent. Every
/// buffer's read set is provably overwritten earlier in the same call (the
/// `ab_row` ring reads only cells `boxsum_ii_row` wrote; `integral_image`
/// writes all of `ii`; `flt`'s `rads[i] > 0` read guard is also its write
/// guard), so pooled buffers are resized WITHOUT re-zeroing — C's
/// uninitialized-`malloc` semantics, kept exact. This was the
/// `alloc_zeroed`/`calloc` memset class's top site (~56M Ir per 196x196 s3
/// rep).
#[derive(Default)]
struct SgrTlsScratch {
    ii: [Vec<i32>; 2],
    ab: [Vec<i32>; 2],
    flt: [Vec<i32>; 2],
}

thread_local! {
    static SGR_TLS: core::cell::RefCell<SgrTlsScratch> = const {
        core::cell::RefCell::new(SgrTlsScratch {
            ii: [Vec::new(), Vec::new()],
            ab: [Vec::new(), Vec::new()],
            flt: [Vec::new(), Vec::new()],
        })
    };
}

/// `selfguided_restoration_fast_internal` (the r=2 pass, A/B at odd rows).
#[allow(clippy::too_many_arguments)]
fn selfguided_fast(
    dgd: &[u16],
    dgd_off: usize,
    dgd_stride: usize,
    ii_sq: &[i32],
    ii_sum: &[i32],
    ii_stride: usize,
    width: usize,
    height: usize,
    dst: &mut [i32],
    dst_stride: usize,
    bit_depth: i32,
    ep: usize,
    ab: &mut [Vec<i32>; 2],
) {
    let [a, b] = ab;
    let (bs, org) = calculate_intermediate(
        ii_sq, ii_sum, ii_stride, width, height, bit_depth, ep, 0, 1, a, b,
    );
    sgr_final_fast(
        a, b, org, bs, dgd, dgd_off, dgd_stride, dst, dst_stride, width, height,
    );
}

/// `selfguided_restoration_internal` (the r=1 pass, every row).
#[allow(clippy::too_many_arguments)]
fn selfguided_full(
    dgd: &[u16],
    dgd_off: usize,
    dgd_stride: usize,
    ii_sq: &[i32],
    ii_sum: &[i32],
    ii_stride: usize,
    width: usize,
    height: usize,
    dst: &mut [i32],
    dst_stride: usize,
    bit_depth: i32,
    ep: usize,
    ab: &mut [Vec<i32>; 2],
) {
    let [a, b] = ab;
    let (bs, org) = calculate_intermediate(
        ii_sq, ii_sum, ii_stride, width, height, bit_depth, ep, 1, 0, a, b,
    );
    sgr_final_full(
        a, b, org, bs, dgd, dgd_off, dgd_stride, dst, dst_stride, width, height,
    );
}

/// `cross_sum` / `cross_sum_fast_*` (selfguided_sse4.c): the weighted
/// neighbourhood sum the final filter applies to the A/B intermediates.
/// `kind` selects the tap pattern: 0 = the full 3x3 ring (9 taps, weights
/// 4/3), 1 = the fast pass's even row (6 taps across the rows above/below,
/// weights 6/5), 2 = the fast pass's odd row (3 taps, weights 6/5).
/// `k` is the centre index; `bs` is the A/B buffer stride. Scalar per-pixel
/// body shared by the scalar tiers and the vector tails so they cannot drift.
#[inline]
fn sgr_final_px(buf: &[i32], k: usize, bs: usize, kind: u32) -> i32 {
    match kind {
        0 => {
            (buf[k] + buf[k - 1] + buf[k + 1] + buf[k - bs] + buf[k + bs]) * 4
                + (buf[k - 1 - bs] + buf[k - 1 + bs] + buf[k + 1 - bs] + buf[k + 1 + bs]) * 3
        }
        1 => {
            (buf[k - bs] + buf[k + bs]) * 6
                + (buf[k - 1 - bs] + buf[k - 1 + bs] + buf[k + 1 - bs] + buf[k + 1 + bs]) * 5
        }
        _ => buf[k] * 6 + (buf[k - 1] + buf[k + 1]) * 5,
    }
}

/// `final_filter` (selfguided_sse4.c): the r=1 apply pass. Per output pixel
/// `dst = rpot(va * dgd + vb, SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS)` where
/// `va`/`vb` are the 9-tap cross sums of A/B. Vectorized 8-wide over `j`;
/// every lane performs exactly the scalar tier's adds/mul in wrap-i32, so the
/// tiers are bit-identical (the products cannot be distinguished from the
/// scalar's wrapping i32 arithmetic either way — `mullo` is the low 32 bits,
/// the same thing the scalar `*` produces).
#[archmage::magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn sgr_final_full_impl(
    token: Token,
    a: &[i32],
    b: &[i32],
    org: usize,
    bs: usize,
    dgd: &[u16],
    dgd_origin: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    const SH: u32 = (SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS) as u32;
    let rnd = i32x8::splat(token, (1i32 << SH) >> 1);
    let c4 = i32x8::splat(token, 4);
    for i in 0..height {
        let k0 = org + i * bs;
        let l0 = dgd_origin + i * dgd_stride;
        let m0 = i * dst_stride;
        // Tight row spans: every tap of the 3x3 ring is an `x[j + c..j + c + 8]`
        // access under `j + 8 <= width` against a `width + 2`-cell slice, so no
        // per-load bounds check survives into the loop (`#![forbid(unsafe_code)]`
        // means elimination has to be structural). `x_c[x + 1]` is row `i`
        // centred on `j`; `x_u`/`x_d` the rows above/below.
        let a_c = &a[k0 - 1..k0 + width + 1];
        let a_u = &a[k0 - bs - 1..k0 - bs + width + 1];
        let a_d = &a[k0 + bs - 1..k0 + bs + width + 1];
        let b_c = &b[k0 - 1..k0 + width + 1];
        let b_u = &b[k0 - bs - 1..k0 - bs + width + 1];
        let b_d = &b[k0 + bs - 1..k0 + bs + width + 1];
        let drow = &dgd[l0..l0 + width];
        let dout = &mut dst[m0..m0 + width];
        let mut j = 0;
        while j + 8 <= width {
            // cross_sum = 4*(fours + threes) - threes (the C factorization —
            // identical wrap-i32 result to 4*fours + 3*threes).
            let fa = i32x8::from_slice(token, &a_c[j..j + 8])
                + i32x8::from_slice(token, &a_c[j + 1..j + 9])
                + i32x8::from_slice(token, &a_c[j + 2..j + 10])
                + i32x8::from_slice(token, &a_u[j + 1..j + 9])
                + i32x8::from_slice(token, &a_d[j + 1..j + 9]);
            let ta = i32x8::from_slice(token, &a_u[j..j + 8])
                + i32x8::from_slice(token, &a_u[j + 2..j + 10])
                + i32x8::from_slice(token, &a_d[j..j + 8])
                + i32x8::from_slice(token, &a_d[j + 2..j + 10]);
            let va = (fa + ta) * c4 - ta;
            let fb = i32x8::from_slice(token, &b_c[j..j + 8])
                + i32x8::from_slice(token, &b_c[j + 1..j + 9])
                + i32x8::from_slice(token, &b_c[j + 2..j + 10])
                + i32x8::from_slice(token, &b_u[j + 1..j + 9])
                + i32x8::from_slice(token, &b_d[j + 1..j + 9]);
            let tb = i32x8::from_slice(token, &b_u[j..j + 8])
                + i32x8::from_slice(token, &b_u[j + 2..j + 10])
                + i32x8::from_slice(token, &b_d[j..j + 8])
                + i32x8::from_slice(token, &b_d[j + 2..j + 10]);
            let vb = (fb + tb) * c4 - tb;
            let src = i32x8::from_array(
                token,
                core::array::from_fn(|t| drow[j + t] as i32),
            );
            let w = (va * src + vb + rnd).shr_arithmetic_const::<9>();
            w.store((&mut dout[j..j + 8]).try_into().unwrap());
            j += 8;
        }
        while j < width {
            let k = k0 + j;
            let v = sgr_final_px(a, k, bs, 0) * drow[j] as i32
                + sgr_final_px(b, k, bs, 0);
            dout[j] = rpot_i32(v, SH);
            j += 1;
        }
    }
}

/// Scalar tier — the transcribed port loop, verbatim.
#[allow(clippy::too_many_arguments)]
fn sgr_final_full_impl_scalar(
    _t: archmage::ScalarToken,
    a: &[i32],
    b: &[i32],
    org: usize,
    bs: usize,
    dgd: &[u16],
    dgd_origin: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    const SH: u32 = (SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS) as u32;
    for i in 0..height {
        for j in 0..width {
            let k = org + i * bs + j;
            let v = sgr_final_px(a, k, bs, 0)
                * dgd[dgd_origin + i * dgd_stride + j] as i32
                + sgr_final_px(b, k, bs, 0);
            dst[i * dst_stride + j] = rpot_i32(v, SH);
        }
    }
}

/// `final_filter_fast` (selfguided_sse4.c): the r=2 apply pass — even rows use
/// the 6-tap cross sum over the rows above/below (`nb = 5`), odd rows the
/// 3-tap horizontal sum (`nb = 4`).
#[archmage::magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn sgr_final_fast_impl(
    token: Token,
    a: &[i32],
    b: &[i32],
    org: usize,
    bs: usize,
    dgd: &[u16],
    dgd_origin: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    const SH_EVEN: u32 = (SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS) as u32;
    const SH_ODD: u32 = (SGRPROJ_SGR_BITS + 4 - SGRPROJ_RST_BITS) as u32;
    let rnd_even = i32x8::splat(token, (1i32 << SH_EVEN) >> 1);
    let rnd_odd = i32x8::splat(token, (1i32 << SH_ODD) >> 1);
    let c5 = i32x8::splat(token, 5);
    for i in 0..height {
        let k0 = org + i * bs;
        let l0 = dgd_origin + i * dgd_stride;
        let m0 = i * dst_stride;
        // Tight row spans, same argument as `sgr_final_full_impl`: `x_c[x + 1]`
        // is row `i` centred on `j`, `x_u`/`x_d` the rows above/below, each
        // slice `width + 2` cells so `x[j + c..j + c + 8]` under
        // `j + 8 <= width` is provably in bounds.
        let a_c = &a[k0 - 1..k0 + width + 1];
        let a_u = &a[k0 - bs - 1..k0 - bs + width + 1];
        let a_d = &a[k0 + bs - 1..k0 + bs + width + 1];
        let b_c = &b[k0 - 1..k0 + width + 1];
        let b_u = &b[k0 - bs - 1..k0 - bs + width + 1];
        let b_d = &b[k0 + bs - 1..k0 + bs + width + 1];
        let drow = &dgd[l0..l0 + width];
        let dout = &mut dst[m0..m0 + width];
        let mut j = 0;
        if i & 1 == 0 {
            // even row: sixes = x_t + x_b, fives = the four corners;
            // cross = 6*sixes + 5*fives = 5*(fives + sixes) + sixes.
            while j + 8 <= width {
                let sa = i32x8::from_slice(token, &a_u[j + 1..j + 9])
                    + i32x8::from_slice(token, &a_d[j + 1..j + 9]);
                let fa = i32x8::from_slice(token, &a_u[j..j + 8])
                    + i32x8::from_slice(token, &a_d[j..j + 8])
                    + i32x8::from_slice(token, &a_u[j + 2..j + 10])
                    + i32x8::from_slice(token, &a_d[j + 2..j + 10]);
                let va = (fa + sa) * c5 + sa;
                let sb = i32x8::from_slice(token, &b_u[j + 1..j + 9])
                    + i32x8::from_slice(token, &b_d[j + 1..j + 9]);
                let fb = i32x8::from_slice(token, &b_u[j..j + 8])
                    + i32x8::from_slice(token, &b_d[j..j + 8])
                    + i32x8::from_slice(token, &b_u[j + 2..j + 10])
                    + i32x8::from_slice(token, &b_d[j + 2..j + 10]);
                let vb = (fb + sb) * c5 + sb;
                let src = i32x8::from_array(
                    token,
                    core::array::from_fn(|t| drow[j + t] as i32),
                );
                let w = (va * src + vb + rnd_even).shr_arithmetic_const::<9>();
                w.store((&mut dout[j..j + 8]).try_into().unwrap());
                j += 8;
            }
            while j < width {
                let k = k0 + j;
                let v = sgr_final_px(a, k, bs, 1) * drow[j] as i32
                    + sgr_final_px(b, k, bs, 1);
                dout[j] = rpot_i32(v, SH_EVEN);
                j += 1;
            }
        } else {
            // odd row: sixes = x, fives = x_l + x_r.
            while j + 8 <= width {
                let sa = i32x8::from_slice(token, &a_c[j + 1..j + 9]);
                let fa = i32x8::from_slice(token, &a_c[j..j + 8])
                    + i32x8::from_slice(token, &a_c[j + 2..j + 10]);
                let va = (fa + sa) * c5 + sa;
                let sb = i32x8::from_slice(token, &b_c[j + 1..j + 9]);
                let fb = i32x8::from_slice(token, &b_c[j..j + 8])
                    + i32x8::from_slice(token, &b_c[j + 2..j + 10]);
                let vb = (fb + sb) * c5 + sb;
                let src = i32x8::from_array(
                    token,
                    core::array::from_fn(|t| drow[j + t] as i32),
                );
                let w = (va * src + vb + rnd_odd).shr_arithmetic_const::<8>();
                w.store((&mut dout[j..j + 8]).try_into().unwrap());
                j += 8;
            }
            while j < width {
                let k = k0 + j;
                let v = sgr_final_px(a, k, bs, 2) * drow[j] as i32
                    + sgr_final_px(b, k, bs, 2);
                dout[j] = rpot_i32(v, SH_ODD);
                j += 1;
            }
        }
    }
}

/// Scalar tier — the transcribed port loops, verbatim.
#[allow(clippy::too_many_arguments)]
fn sgr_final_fast_impl_scalar(
    _t: archmage::ScalarToken,
    a: &[i32],
    b: &[i32],
    org: usize,
    bs: usize,
    dgd: &[u16],
    dgd_origin: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    const SH_EVEN: u32 = (SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS) as u32;
    const SH_ODD: u32 = (SGRPROJ_SGR_BITS + 4 - SGRPROJ_RST_BITS) as u32;
    for i in 0..height {
        let k_row = org + i * bs;
        let l_row = dgd_origin + i * dgd_stride;
        let m_row = i * dst_stride;
        if i & 1 == 0 {
            for j in 0..width {
                let k = k_row + j;
                let v = sgr_final_px(a, k, bs, 1) * dgd[l_row + j] as i32
                    + sgr_final_px(b, k, bs, 1);
                dst[m_row + j] = rpot_i32(v, SH_EVEN);
            }
        } else {
            for j in 0..width {
                let k = k_row + j;
                let v = sgr_final_px(a, k, bs, 2) * dgd[l_row + j] as i32
                    + sgr_final_px(b, k, bs, 2);
                dst[m_row + j] = rpot_i32(v, SH_ODD);
            }
        }
    }
}

/// Dispatch the r=1 apply pass.
#[allow(clippy::too_many_arguments)]
fn sgr_final_full(
    a: &[i32],
    b: &[i32],
    org: usize,
    bs: usize,
    dgd: &[u16],
    dgd_origin: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    archmage::incant!(
        sgr_final_full_impl(
            a, b, org, bs, dgd, dgd_origin, dgd_stride, dst, dst_stride, width, height
        ),
        [v3, neon, wasm128, scalar]
    );
}

/// Dispatch the r=2 apply pass.
#[allow(clippy::too_many_arguments)]
fn sgr_final_fast(
    a: &[i32],
    b: &[i32],
    org: usize,
    bs: usize,
    dgd: &[u16],
    dgd_origin: usize,
    dgd_stride: usize,
    dst: &mut [i32],
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    archmage::incant!(
        sgr_final_fast_impl(
            a, b, org, bs, dgd, dgd_origin, dgd_stride, dst, dst_stride, width, height
        ),
        [v3, neon, wasm128, scalar]
    );
}

/// `av1_selfguided_restoration`: build the two integral images over the
/// `[-3, +3)`-extended source (`integral_images_highbd` — C never widens to a
/// staging plane, it reads the pixel plane directly), then run the enabled
/// passes into `flt0` (r\[0\]=2 fast) and `flt1` (r\[1\]=1 full), both at
/// `flt_stride = width`. `ii[0]` holds the square sums (C's `C`), `ii[1]`
/// the plain sums (C's `D`).
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
    let width_ext = width + 2 * SGRPROJ_BORDER_HORZ;
    let height_ext = height + 2 * SGRPROJ_BORDER_VERT;
    // Same "avoid bad cache effects" stride the A/B layout uses; the integral
    // images share it (needs `>= width_ext + 1` columns, always true).
    let ii_stride = ((width_ext + 3) & !3) + 16;
    // `mem::take` out of the TLS pool (no borrow held across the calls below —
    // `apply_selfguided_restoration` nests through this) and put back at the
    // end. See [`SgrTlsScratch`] for why dirty reuse is exact.
    let mut s = SGR_TLS.with(|c| core::mem::take(&mut *c.borrow_mut()));
    let [ii_sq, ii_sum] = &mut s.ii;
    ii_sq.resize(ii_stride * (height_ext + 1), 0);
    ii_sum.resize(ii_stride * (height_ext + 1), 0);
    let ext_off = dgd_off as isize
        - SGRPROJ_BORDER_VERT as isize * dgd_stride as isize
        - SGRPROJ_BORDER_HORZ as isize;
    integral_image(
        dgd,
        ext_off as usize,
        dgd_stride,
        width_ext,
        height_ext,
        ii_sq,
        ii_sum,
        ii_stride,
    );
    let (rads, _) = SGR_PARAMS[ep];
    debug_assert!(!(rads[0] == 0 && rads[1] == 0));
    if rads[0] > 0 {
        selfguided_fast(
            dgd,
            dgd_off,
            dgd_stride,
            &s.ii[0],
            &s.ii[1],
            ii_stride,
            width,
            height,
            flt0,
            flt_stride,
            bit_depth,
            ep,
            &mut s.ab,
        );
    }
    if rads[1] > 0 {
        selfguided_full(
            dgd,
            dgd_off,
            dgd_stride,
            &s.ii[0],
            &s.ii[1],
            ii_stride,
            width,
            height,
            flt1,
            flt_stride,
            bit_depth,
            ep,
            &mut s.ab,
        );
    }
    SGR_TLS.with(|c| *c.borrow_mut() = s);
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
    // Pool `flt0`/`flt1` field-wise (the nested `selfguided_restoration`
    // `mem::take`s the whole scratch, so the flt fields are already empty for
    // it) and put them back after the blend. See [`SgrTlsScratch`].
    let (mut flt0, mut flt1) = SGR_TLS.with(|c| {
        let mut s = c.borrow_mut();
        (
            core::mem::take(&mut s.flt[0]),
            core::mem::take(&mut s.flt[1]),
        )
    });
    flt0.resize(width * height, 0);
    flt1.resize(width * height, 0);
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
    SGR_TLS.with(|c| {
        let mut s = c.borrow_mut();
        s.flt = [flt0, flt1];
    });
}
