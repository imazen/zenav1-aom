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


/// The VERTICAL half of `boxsum1`/`boxsum2`, vectorized over `j`.
///
/// # Why this exists
///
/// `benchmarks/encoder_x86_profile_2026-09-08.md`: the loop-restoration search
/// is 26 % of the speed-0 encode-time gap to libaom and had no SIMD anywhere.
/// After the `compute_stats` and `pixel_proj_error` tiers landed, the box-sum
/// was the largest scalar item left in the stage (13.3 ms of a 455 ms encode).
///
/// # Bit-exact, and it also fixes an access pattern
///
/// Every output is an INDEPENDENT sum of `2r + 1` source rows at one column, so
/// vectorizing across `j` reorders nothing: lane `j` performs exactly the scalar
/// tier's adds in exactly its order. What changes besides the width is the walk
/// — the scalar form is column-OUTER and strides by `src_stride` on every step,
/// so it re-reads each row `width` times with no locality; this form carries a
/// strip of 8 columns down the rows at once. The tail (`width % 8`) stays
/// scalar and shares the same code as before via `boxsum_vert_scalar_cols`.
#[archmage::magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn boxsum_vert_impl(
    token: Token,
    src: &[i32],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    sqr: bool,
    dst: &mut [i32],
    dst_stride: usize,
    r5: bool,
) {
    let ld = |s: &[i32], o: usize| -> i32x8 {
        let v = i32x8::from_slice(token, &s[o..o + 8]);
        if sqr { v * v } else { v }
    };
    let st = |d: &mut [i32], o: usize, v: i32x8| {
        let t: &mut [i32; 8] = (&mut d[o..o + 8]).try_into().unwrap();
        v.store(t);
    };

    let mut j = 0usize;
    while j + 8 <= width {
        if r5 {
            let mut a = ld(src, src_off + j);
            let mut b = ld(src, src_off + src_stride + j);
            let mut c = ld(src, src_off + 2 * src_stride + j);
            let mut d = ld(src, src_off + 3 * src_stride + j);
            let mut e = ld(src, src_off + 4 * src_stride + j);
            st(dst, j, a + b + c);
            st(dst, dst_stride + j, a + b + c + d);
            let mut i = 2;
            while i < height - 3 {
                st(dst, i * dst_stride + j, a + b + c + d + e);
                a = b;
                b = c;
                c = d;
                d = e;
                e = ld(src, src_off + (i + 3) * src_stride + j);
                i += 1;
            }
            st(dst, i * dst_stride + j, a + b + c + d + e);
            st(dst, (i + 1) * dst_stride + j, b + c + d + e);
            st(dst, (i + 2) * dst_stride + j, c + d + e);
        } else {
            let mut a = ld(src, src_off + j);
            let mut b = ld(src, src_off + src_stride + j);
            let mut c = ld(src, src_off + 2 * src_stride + j);
            st(dst, j, a + b);
            let mut i = 1;
            while i < height - 2 {
                st(dst, i * dst_stride + j, a + b + c);
                a = b;
                b = c;
                c = ld(src, src_off + (i + 2) * src_stride + j);
                i += 1;
            }
            st(dst, i * dst_stride + j, a + b + c);
            st(dst, (i + 1) * dst_stride + j, b + c);
        }
        j += 8;
    }
    boxsum_vert_scalar_cols(
        src, src_off, j, width, height, src_stride, sqr, dst, dst_stride, r5,
    );
}

#[allow(clippy::too_many_arguments)]
fn boxsum_vert_impl_scalar(
    _t: archmage::ScalarToken,
    src: &[i32],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    sqr: bool,
    dst: &mut [i32],
    dst_stride: usize,
    r5: bool,
) {
    boxsum_vert_scalar_cols(
        src, src_off, 0, width, height, src_stride, sqr, dst, dst_stride, r5,
    );
}

/// The scalar vertical pass over columns `j0..width` — the transcribed port,
/// verbatim, and the reference the vector tier is compared against.
#[allow(clippy::too_many_arguments)]
fn boxsum_vert_scalar_cols(
    src: &[i32],
    src_off: usize,
    j0: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    sqr: bool,
    dst: &mut [i32],
    dst_stride: usize,
    r5: bool,
) {
    let sq = |v: i32| if sqr { v * v } else { v };
    for j in j0..width {
        if r5 {
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
        } else {
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
    }
}

/// Dispatch the vertical pass.
#[allow(clippy::too_many_arguments)]
fn boxsum_vert(
    src: &[i32],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    sqr: bool,
    dst: &mut [i32],
    dst_stride: usize,
    r5: bool,
) {
    archmage::incant!(
        boxsum_vert_impl(
            src, src_off, width, height, src_stride, sqr, dst, dst_stride, r5
        ),
        [v3, neon, wasm128, scalar]
    )
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
    // Vertical sum over 3-pixel regions, from src into dst.
    boxsum_vert(
        src, src_off, width, height, src_stride, sqr, dst, dst_stride, false,
    );
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
    boxsum_vert(
        src, src_off, width, height, src_stride, sqr, dst, dst_stride, true,
    );
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

/// Scalar tier = the transcribed port, verbatim.
#[allow(clippy::too_many_arguments)]
fn ab_row_impl_scalar(
    _t: archmage::ScalarToken,
    a_row: &mut [i32],
    b_row: &mut [i32],
    n: u32,
    s: u32,
    one_by_x: u32,
    shift_a: u32,
    shift_b: u32,
) {
    for t in 0..a_row.len() {
        let (a_out, b_out) = ab_one(a_row[t], b_row[t], n, s, one_by_x, shift_a, shift_b);
        a_row[t] = a_out;
        b_row[t] = b_out;
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
#[archmage::magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn ab_row_impl(
    token: Token,
    a_row: &mut [i32],
    b_row: &mut [i32],
    n: u32,
    s: u32,
    one_by_x: u32,
    shift_a: u32,
    shift_b: u32,
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

    let len = a_row.len();
    let mut o = 0usize;
    while o + 8 <= len {
        let a_raw = i32x8::from_slice(token, &a_row[o..o + 8]);
        let b_raw = i32x8::from_slice(token, &b_row[o..o + 8]);

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
    for t in o..len {
        let (a_out, b_out) = ab_one(a_row[t], b_row[t], n, s, one_by_x, shift_a, shift_b);
        a_row[t] = a_out;
        b_row[t] = b_out;
    }
}

/// `av1_selfguided_restoration_c`'s A/B pass over one row of the ring.
#[allow(clippy::too_many_arguments)]
fn ab_row(
    a_row: &mut [i32],
    b_row: &mut [i32],
    n: u32,
    s: u32,
    one_by_x: u32,
    shift_a: u32,
    shift_b: u32,
) {
    archmage::incant!(
        ab_row_impl(a_row, b_row, n, s, one_by_x, shift_a, shift_b),
        [v3, neon, wasm128, scalar]
    )
}

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
    let one_by_x = ONE_BY_X[(n - 1) as usize];
    let shift_a = 2 * (bit_depth - 8) as u32;
    let shift_b = (bit_depth - 8) as u32;
    // `k` is contiguous in `j`, so each row of the ring is one contiguous run
    // of `width + 2` cells in each buffer (j = -1 ..= width).
    let count = width + 2;
    let mut i: i32 = -1;
    while i < height as i32 + 1 {
        let k0 = (org as i32 + i * buf_stride as i32 - 1) as usize;
        ab_row(
            &mut a_buf[k0..k0 + count],
            &mut b_buf[k0..k0 + count],
            n,
            s,
            one_by_x,
            shift_a,
            shift_b,
        );
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
