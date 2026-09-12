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
    incant!(filter_intra_edge_impl(p, sz, taps), [v3, neon, wasm128, scalar])
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

#[magetypes(define(i16x16, i16x8, i32x8, i32x4), v3, neon, wasm128, -scalar)]
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
