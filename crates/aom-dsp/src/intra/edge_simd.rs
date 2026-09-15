//! **i32-lane** vector kernels for the two edge-prep loops in
//! [`super::edge`] — `highbd_filter_intra_edge` (5-tap FIR over an edge) and
//! `highbd_upsample_intra_edge` (4-tap doubling filter) — which were pure
//! scalar while libaom dispatches `av1_highbd_filter_intra_edge_sse4_1` /
//! `av1_highbd_upsample_intra_edge_sse4_1`.
//!
//! # `highbd_filter_intra_edge`
//!
//! The scalar port avoids C's whole-edge scratch copy with a 5-wide rolling
//! window, which is sound only because every read index `i + 2` is strictly
//! ahead of the write index `i`. A vector chunk writes `p[i..i+8)` and the
//! NEXT chunk reads from `p[i+6]`, so in-place vectorisation would read
//! filtered values where the filter wants originals. The vector path therefore
//! restores **C's own structure**: copy the edge into a stack scratch
//! (`av1_highbd_filter_intra_edge_c`'s `edge[129]`) and read originals from
//! it. `sz` is bounded by `n_top_px + 1 + txhpx <= 129` in production; the
//! scratch carries `sz + 16` of headroom so the 16-lane widening loads stay
//! in bounds, and anything larger falls back to the scalar walk
//! (unreachable, keeps the function total).
//!
//! The filter is a plain 5-tap FIR once reads are decoupled from writes:
//!
//! ```text
//! out[i] = (sum_j orig[clamp(i-2+j)] * taps[j] + 8) >> 4,   i in 1..sz
//! ```
//!
//! Interior outputs (`i in [2, sz-3]`, where `clamp` is the identity) come
//! from five unaligned `i16x16` loads `orig[i-2+j .. i-2+j+16)` narrowed by
//! `widen_low` to the `i32x8` lanes output `k` needs (`orig[k-2+j]`). `taps`
//! sum to 16 and samples are `<= 4095`, so the i32 sum `<= 16 * 4095 + 8`
//! and the result `<= 4095` — no clip, and the narrowing `as u16` is exact.
//! Clamped boundary outputs (`i = 1`, `i = sz-2`, `i = sz-1`) stay scalar.
//!
//! # `highbd_upsample_intra_edge`
//!
//! Doubling filter: `buf[off-2] = inp[0]`, then per `i < sz` the pair
//! `buf[off+2i-1] = clip((-inp[i] + 9 inp[i+1] + 9 inp[i+2] - inp[i+3] + 8) >> 4)`
//! and `buf[off+2i] = inp[i+2]`, where `inp` is the edge copied into scratch
//! with duplicated ends. The scalar already reads from `inp`, so writes cannot
//! race reads and no extra copy is needed. The filtered lane is a 4-tap FIR
//! over contiguous `inp` — `i32x8::from_slice` is already a widening-free
//! load. The interleaved pair store writes `2 * chunk` `u16`s at once via a
//! staged `[u16; 16]`/`[u16; 8]` array. `sz <= 16` in production
//! (`use_upsample` gates on `bw + bh <= 16`), matching `inp`'s 19-entry bound.

use archmage::prelude::*;

/// Stack scratch bound for the filter's originals copy — production `sz` tops
/// out at 129 (`64 + 1 + 64`); the extra 16 cover the over-read of the
/// `i16x16` widening loads (only the low 8 lanes are used).
const FILTER_SCRATCH: usize = 208;

/// Scalar per-output recipe — the differential reference AND the
/// boundary/remainder path. `orig` holds the ORIGINAL samples.
#[inline]
fn filter_edge_out(orig: &[i16], i: usize, sz: usize, taps: &[i32; 5]) -> u16 {
    let mut s = 0i32;
    for j in 0..5 {
        let k = (i as i32 + j as i32 - 2).clamp(0, sz as i32 - 1) as usize;
        s += i32::from(orig[k]) * taps[j];
    }
    ((s + 8) >> 4) as u16
}

/// `av1_highbd_filter_intra_edge_c` — in-place 5-tap filter, strength-gated.
/// `taps` is `KERNEL[strength - 1]`; `strength == 0` never reaches here (the
/// caller returns early, exactly like the scalar).
pub(crate) fn filter_intra_edge_run(p: &mut [u16], sz: usize, taps: [i32; 5]) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    #[cfg(target_arch = "x86_64")]
    {
        incant!(
            filter_intra_edge_impl_x86(p, sz, taps),
            [v3, scalar]
        );
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    incant!(filter_intra_edge_impl(p, sz, taps), [neon, wasm128, scalar])
}

/// x86 v3 body — `madd_epi16` pair-taps instead of the generic i32-lane
/// widening: i32 lane `m` of an `orig` load already holds the adjacent pair
/// `(orig[b+2m], orig[b+2m+1])` the (t_j, t_j+1) tap pair needs, so even
/// outputs come from loads at `i-2`/`i`/`i+2` and odd outputs at `i-1`/`i+1`/
/// `i+3`. Six loads + six madds per 16 outputs replace five widen+mul chains
/// per 8. The interleave-back is free: `(even & 0xFFFF) | (odd << 16)` IS the
/// u16 output pair layout. Exact: `orig <= 4095` and `|taps| <= 8`, so every
/// madd pair-sum `<= 2 * 8 * 4095` and the i32 total `<= 16 * 4095 + 8` —
/// same `(s + 8) >> 4 as u16` value the scalar writes (no clip; the `& 0xFFFF`
/// and `<< 16` keep exactly the low 16 bits the `as u16` cast keeps).
///
/// `sz` is `2k+1`-shaped in practice (`n_px = edge + 1 + ext`), so the
/// interior tail after the stride-16 walk is finished with ONE overlapping
/// backward chunk at `sz - 18` (idempotent: every interior output is a pure
/// function of `orig`, so rewriting an already-computed lane stores the same
/// value) rather than a long scalar tail.
#[cfg(target_arch = "x86_64")]
#[magetypes(define(i16x16), v3, -scalar)]
fn filter_intra_edge_impl_x86(token: Token, p: &mut [u16], sz: usize, taps: [i32; 5]) {
    use archmage::intrinsics::x86_64::*;
    let _ = token;
    if sz > 192 || sz < 2 {
        super::edge::filter_intra_edge_scalar_inplace(p, sz, taps);
        return;
    }
    let mut orig = [0i16; FILTER_SCRATCH];
    // Vector snapshot of the originals: same bits, u16->i16 needs no convert.
    let mut k = 0usize;
    while k + 16 <= sz {
        let s: &[u16; 16] = p[k..k + 16].try_into().unwrap();
        let d: &mut [i16; 16] = (&mut orig[k..k + 16]).try_into().unwrap();
        _mm256_storeu_si256(d, _mm256_loadu_si256(s));
        k += 16;
    }
    while k < sz {
        orig[k] = p[k] as i16;
        k += 1;
    }

    let pair = |a: i32, b: i32| -> __m256i {
        _mm256_set1_epi32(((a as u16 as u32) | ((b as u16 as u32) << 16)) as i32)
    };
    let t01 = pair(taps[0], taps[1]);
    let t23 = pair(taps[2], taps[3]);
    let t45 = pair(taps[4], 0);
    let eight = _mm256_set1_epi32(8);
    let lo16 = _mm256_set1_epi32(0xFFFF);

    // One 16-output chunk at position i; outputs i..i+15 must all be interior
    // (i >= 2, i + 15 <= sz - 3). Reads orig[i - 2 ..= i + 18] < FILTER_SCRATCH.
    let do16 = |p: &mut [u16], orig: &[i16; FILTER_SCRATCH], i: usize| {
        let ld = |b: usize| -> __m256i {
            let s: &[i16; 16] = orig[b..b + 16].try_into().unwrap();
            _mm256_loadu_si256(s)
        };
        let even = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(ld(i - 2), t01),
                _mm256_madd_epi16(ld(i), t23),
            ),
            _mm256_madd_epi16(ld(i + 2), t45),
        );
        let odd = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(ld(i - 1), t01),
                _mm256_madd_epi16(ld(i + 1), t23),
            ),
            _mm256_madd_epi16(ld(i + 3), t45),
        );
        let re = _mm256_srai_epi32::<4>(_mm256_add_epi32(even, eight));
        let ro = _mm256_slli_epi32::<16>(_mm256_srai_epi32::<4>(_mm256_add_epi32(odd, eight)));
        let out: &mut [u16; 16] = (&mut p[i..i + 16]).try_into().unwrap();
        _mm256_storeu_si256(out, _mm256_or_si256(_mm256_and_si256(re, lo16), ro));
    };
    // Same shape on xmm for sz < 20; reads orig[i - 2 ..= i + 10].
    let do8 = |p: &mut [u16], orig: &[i16; FILTER_SCRATCH], i: usize| {
        let pair128 = |a: i32, b: i32| -> __m128i {
            _mm_set1_epi32(((a as u16 as u32) | ((b as u16 as u32) << 16)) as i32)
        };
        let ld = |b: usize| -> __m128i {
            let s: &[i16; 8] = orig[b..b + 8].try_into().unwrap();
            _mm_loadu_si128(s)
        };
        let even = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(ld(i - 2), pair128(taps[0], taps[1])),
                _mm_madd_epi16(ld(i), pair128(taps[2], taps[3])),
            ),
            _mm_madd_epi16(ld(i + 2), pair128(taps[4], 0)),
        );
        let odd = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(ld(i - 1), pair128(taps[0], taps[1])),
                _mm_madd_epi16(ld(i + 1), pair128(taps[2], taps[3])),
            ),
            _mm_madd_epi16(ld(i + 3), pair128(taps[4], 0)),
        );
        let re = _mm_srai_epi32::<4>(_mm_add_epi32(even, _mm_set1_epi32(8)));
        let ro = _mm_slli_epi32::<16>(_mm_srai_epi32::<4>(_mm_add_epi32(
            odd,
            _mm_set1_epi32(8),
        )));
        let out: &mut [u16; 8] = (&mut p[i..i + 8]).try_into().unwrap();
        _mm_storeu_si128(out, _mm_or_si128(_mm_and_si128(re, _mm_set1_epi32(0xFFFF)), ro));
    };

    p[1] = filter_edge_out(&orig, 1, sz, &taps);
    let mut i = 2usize;
    while i + 16 <= sz - 2 {
        do16(p, &orig, i);
        i += 16;
    }
    if i <= sz - 3 {
        if sz >= 20 {
            // Backward-overlap finisher: covers (sz - 18)..=(sz - 3), all
            // interior; lanes it rewrites store identical values.
            do16(p, &orig, sz - 18);
            i = sz - 2;
        } else {
            while i + 8 <= sz - 2 {
                do8(p, &orig, i);
                i += 8;
            }
            if i <= sz - 3 && sz >= 12 {
                do8(p, &orig, sz - 10);
                i = sz - 2;
            }
        }
    }
    for k in i..sz {
        p[k] = filter_edge_out(&orig, k, sz, &taps);
    }
}

/// Scalar tier for the x86 dispatch — same in-place walk as the generic tier.
#[cfg(target_arch = "x86_64")]
fn filter_intra_edge_impl_x86_scalar(
    t: archmage::ScalarToken,
    p: &mut [u16],
    sz: usize,
    taps: [i32; 5],
) {
    filter_intra_edge_impl_scalar(t, p, sz, taps);
}

fn filter_intra_edge_impl_scalar(
    _t: archmage::ScalarToken,
    p: &mut [u16],
    sz: usize,
    taps: [i32; 5],
) {
    let mut orig = [0i16; FILTER_SCRATCH];
    if sz > 192 || sz < 2 {
        // Out-of-envelope sizes take the original in-place rolling window.
        super::edge::filter_intra_edge_scalar_inplace(p, sz, taps);
        return;
    }
    for k in 0..sz {
        orig[k] = p[k] as i16;
    }
    for i in 1..sz {
        p[i] = filter_edge_out(&orig, i, sz, &taps);
    }
}

#[magetypes(define(i16x16, i16x8, i32x8, i32x4), neon, wasm128, -scalar)]
fn filter_intra_edge_impl(token: Token, p: &mut [u16], sz: usize, taps: [i32; 5]) {
    if sz > 192 || sz < 2 {
        super::edge::filter_intra_edge_scalar_inplace(p, sz, taps);
        return;
    }
    let mut orig = [0i16; FILTER_SCRATCH];
    for k in 0..sz {
        orig[k] = p[k] as i16;
    }

    let t0 = i32x8::splat(token, taps[0]);
    let t1 = i32x8::splat(token, taps[1]);
    let t2 = i32x8::splat(token, taps[2]);
    let t3 = i32x8::splat(token, taps[3]);
    let t4 = i32x8::splat(token, taps[4]);
    let eight = i32x8::splat(token, 8);
    // 8 lanes of originals starting at `b`: `widen_low` keeps lanes 0..8 —
    // exactly the `orig[b + k]` each output lane k needs. The 16-lane load
    // reads up to `b + 15 <= i + 9 + 15 <= sz + 6 < FILTER_SCRATCH`.
    let w8 = |b: usize| -> i32x8 {
        i16x16::from_slice(token, &orig[b..b + 16]).widen_low()
    };

    p[1] = filter_edge_out(&orig, 1, sz, &taps);
    // Interior outputs i in [2, sz-3]: an 8-chunk starting at i covers
    // i..i+7, all interior iff i + 7 <= sz - 3.
    let mut i = 2usize;
    while i + 8 <= sz - 2 {
        let s = t0 * w8(i - 2)
            + t1 * w8(i - 1)
            + t2 * w8(i)
            + t3 * w8(i + 1)
            + t4 * w8(i + 2);
        let out = (s + eight).shr_arithmetic_const::<4>().to_array();
        p[i..i + 8].copy_from_slice(&out.map(|x| x as u16));
        i += 8;
    }
    // One 4-chunk: outputs i..i+3 interior iff i + 3 <= sz - 3; the 8-lane
    // loads stay < i + 9 <= sz + 3 < FILTER_SCRATCH.
    if i + 4 <= sz - 2 {
        let q0 = i32x4::splat(token, taps[0]);
        let q1 = i32x4::splat(token, taps[1]);
        let q2 = i32x4::splat(token, taps[2]);
        let q3 = i32x4::splat(token, taps[3]);
        let q4 = i32x4::splat(token, taps[4]);
        let w4 = |b: usize| -> i32x4 {
            i16x8::from_slice(token, &orig[b..b + 8]).widen_low()
        };
        let s = q0 * w4(i - 2)
            + q1 * w4(i - 1)
            + q2 * w4(i)
            + q3 * w4(i + 1)
            + q4 * w4(i + 2);
        let out = (s + i32x4::splat(token, 8)).shr_arithmetic_const::<4>().to_array();
        p[i..i + 4].copy_from_slice(&out.map(|x| x as u16));
        i += 4;
    }
    for k in i..sz {
        p[k] = filter_edge_out(&orig, k, sz, &taps);
    }
}

/// Scalar per-output recipe for the upsample filter — the differential
/// reference AND the remainder path. `inp` is the padded scratch.
#[inline]
fn upsample_out(inp: &[i32; 19], i: usize, max_v: i32) -> u16 {
    ((-inp[i] + 9 * inp[i + 1] + 9 * inp[i + 2] - inp[i + 3] + 8) >> 4).clamp(0, max_v) as u16
}

/// `av1_highbd_upsample_intra_edge_c` — doubling filter writing interleaved
/// (filtered, original) pairs.
pub(crate) fn upsample_intra_edge_run(buf: &mut [u16], off: usize, sz: usize, max_v: i32) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        upsample_intra_edge_impl(buf, off, sz, max_v),
        [v3, neon, wasm128, scalar]
    )
}

fn upsample_intra_edge_impl_scalar(
    _t: archmage::ScalarToken,
    buf: &mut [u16],
    off: usize,
    sz: usize,
    max_v: i32,
) {
    upsample_intra_edge_scalar_body(buf, off, sz, max_v);
}

/// The scalar body — shared by the scalar tier and the vector remainder.
/// `inp` construction mirrors `highbd_upsample_intra_edge` exactly.
fn upsample_intra_edge_scalar_body(buf: &mut [u16], off: usize, sz: usize, max_v: i32) {
    let mut inp = [0i32; 19];
    inp[0] = i32::from(buf[off - 1]);
    inp[1] = i32::from(buf[off - 1]);
    for i in 0..sz {
        inp[i + 2] = i32::from(buf[off + i]);
    }
    inp[sz + 2] = i32::from(buf[off + sz - 1]);
    buf[off - 2] = inp[0] as u16;
    for i in 0..sz {
        buf[off + 2 * i - 1] = upsample_out(&inp, i, max_v);
        buf[off + 2 * i] = inp[i + 2] as u16;
    }
}

#[magetypes(define(i32x8, i32x4), v3, neon, wasm128, -scalar)]
fn upsample_intra_edge_impl(token: Token, buf: &mut [u16], off: usize, sz: usize, max_v: i32) {
    if sz + 3 > 19 || sz == 0 {
        // `inp` holds sz + 3 entries; larger edges stay scalar (unreachable in
        // production — `use_upsample` gates `bw + bh <= 16`).
        upsample_intra_edge_scalar_body(buf, off, sz, max_v);
        return;
    }
    let mut inp = [0i32; 19];
    inp[0] = i32::from(buf[off - 1]);
    inp[1] = i32::from(buf[off - 1]);
    for i in 0..sz {
        inp[i + 2] = i32::from(buf[off + i]);
    }
    inp[sz + 2] = i32::from(buf[off + sz - 1]);
    buf[off - 2] = inp[0] as u16;

    let nine = i32x8::splat(token, 9);
    let eight = i32x8::splat(token, 8);
    let zero8 = i32x8::zero(token);
    let max8 = i32x8::splat(token, max_v);

    // Chunk of 8 filtered outputs i..i+7 reads inp[i .. i+11) — in bounds for
    // i + 10 <= sz + 2 (the padded scratch has sz + 3 live entries).
    let mut i = 0usize;
    while i + 8 <= sz {
        let f = (-i32x8::from_slice(token, &inp[i..i + 8])
            + nine * i32x8::from_slice(token, &inp[i + 1..i + 9])
            + nine * i32x8::from_slice(token, &inp[i + 2..i + 10])
            - i32x8::from_slice(token, &inp[i + 3..i + 11])
            + eight)
            .shr_arithmetic_const::<4>()
            .clamp(zero8, max8)
            .to_array();
        let mut pair = [0u16; 16];
        for k in 0..8 {
            pair[2 * k] = f[k] as u16;
            pair[2 * k + 1] = inp[i + k + 2] as u16;
        }
        buf[off + 2 * i - 1..off + 2 * i + 15].copy_from_slice(&pair);
        i += 8;
    }
    if i + 4 <= sz {
        let n4 = i32x4::splat(token, 9);
        let e4 = i32x4::splat(token, 8);
        let z4 = i32x4::zero(token);
        let m4 = i32x4::splat(token, max_v);
        let f = (-i32x4::from_slice(token, &inp[i..i + 4])
            + n4 * i32x4::from_slice(token, &inp[i + 1..i + 5])
            + n4 * i32x4::from_slice(token, &inp[i + 2..i + 6])
            - i32x4::from_slice(token, &inp[i + 3..i + 7])
            + e4)
            .shr_arithmetic_const::<4>()
            .clamp(z4, m4)
            .to_array();
        let mut pair = [0u16; 8];
        for k in 0..4 {
            pair[2 * k] = f[k] as u16;
            pair[2 * k + 1] = inp[i + k + 2] as u16;
        }
        buf[off + 2 * i - 1..off + 2 * i + 7].copy_from_slice(&pair);
        i += 4;
    }
    for k in i..sz {
        buf[off + 2 * k - 1] = upsample_out(&inp, k, max_v);
        buf[off + 2 * k] = inp[k + 2] as u16;
    }
}
