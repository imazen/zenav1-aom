//! SIMD column kernel for `txb_init_levels` — bit-identical to the
//! scalar port on the FULL i32 domain, at every dispatch tier.
//!
//! Same aom-rs SIMD pattern as `crate::quant::simd` / `crate::cdef::simd`: the
//! magetypes kernel handles heights 8/16/32 (whole 8-lane column chunks);
//! the `_scalar` incant variant and the height-4 route call the transcribed
//! port verbatim ([`crate::txb::txb_init_levels_scalar`]).
//!
//! # Bit-exactness (full domain)
//!
//! Per coefficient the scalar port computes `unsigned_abs().min(127) as u8`.
//! Lanes compute `a = (x ^ (x>>31)) - (x>>31)` (wrapping — `i32::MIN` stays
//! `i32::MIN`), then `blend(a < 0, 127, min(a, 127))`: the only negative `a`
//! is the `i32::MIN` lane, whose `unsigned_abs() = 2^31` also clamps to 127
//! in the scalar port. Everything else is `0 <= a <= i32::MAX`, where the
//! signed lane `min` equals the scalar's unsigned min. The final `as u8`
//! narrowing writes values already in `0..=127`.

use archmage::prelude::*;

use crate::txb::{TxClass, TX_PAD_BOTTOM, TX_PAD_END, TX_PAD_HOR};

/// Scalar tier = the transcribed port, verbatim.
pub(crate) fn txb_init_levels_impl_scalar(
    _t: archmage::ScalarToken,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    crate::txb::txb_init_levels_scalar(coeff, width, height, levels)
}

#[magetypes(define(i32x8), neon, wasm128, -scalar)]
pub(crate) fn txb_init_levels_impl(
    token: Token,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    let stride = height + TX_PAD_HOR;
    let tail = stride * width;
    levels[tail..tail + crate::txb::TX_PAD_BOTTOM * stride + crate::txb::TX_PAD_END].fill(0);

    let zero = i32x8::zero(token);
    let cap = i32x8::splat(token, i8::MAX as i32);
    // |x|.min(127) with the i32::MIN lane mapping to 127 exactly like the
    // scalar port's unsigned_abs().min(127) — see module docs.
    let abs127 = |x: i32x8| {
        let m = x.shr_arithmetic_const::<31>();
        let a = (x ^ m) - m; // |x| (wrapping; i32::MIN stays negative)
        i32x8::blend(a.simd_lt(zero), cap, a.min(cap))
    };

    if height == 4 {
        // One 8-lane vector = TWO 4-coeff columns; each column's 4 levels +
        // 4 pad zeros are 8 output bytes, so a pair writes bytes 0..4 and
        // 8..12 of a 16-byte window. Widths are powers of two >= 4.
        debug_assert!(width % 2 == 0);
        debug_assert_eq!(stride, 8, "height 4 + TX_PAD_HOR 4");
        for p in 0..width / 2 {
            let a = abs127(i32x8::from_slice(token, &coeff[p * 8..p * 8 + 8])).to_array();
            let out = &mut levels[p * 2 * stride..p * 2 * stride + 2 * stride];
            // One 8-byte store per column instead of four byte stores plus a
            // 4-byte fill: `stride == 8` here, so each column's four levels and
            // four pad zeros are exactly one aligned run. Values are already in
            // `0..=127` (module docs), so `as u8` is exact.
            out[..8].copy_from_slice(&[a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8, 0, 0, 0, 0]);
            out[stride..stride + 8]
                .copy_from_slice(&[a[4] as u8, a[5] as u8, a[6] as u8, a[7] as u8, 0, 0, 0, 0]);
        }
        return;
    }

    assert!(height % 8 == 0);
    for i in 0..width {
        let col = &coeff[i * height..(i + 1) * height];
        let out = &mut levels[i * stride..i * stride + stride];
        for c in 0..height / 8 {
            let a = abs127(i32x8::from_slice(token, &col[c * 8..c * 8 + 8])).to_array();
            // ONE 8-byte store, not eight byte stores. magetypes 0.9.28 has no
            // i32->u8 narrowing primitive (no pack/narrow/shuffle — only
            // `blend`), so libaom's `_mm256_packs_epi32` + `_mm256_packus_epi16`
            // shape is not expressible here; building the run and storing it
            // once is what is available, and it is where this kernel's time was
            // going — it was the top `__memset`/store caller in the frame-pointer
            // profile despite already being SIMD.
            let bytes = [
                a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8, a[4] as u8, a[5] as u8, a[6] as u8,
                a[7] as u8,
            ];
            out[c * 8..c * 8 + 8].copy_from_slice(&bytes);
        }
        out[height..height + TX_PAD_HOR].fill(0);
    }
}

// ---- av1_txb_init_levels_avx2 (encodetxb_avx2.c:24) -----------------------
//
// Hand-ported v3 tier: C's AVX2 works in i16/i8 lanes — 2 loads + packs_epi32
// + abs_epi16 + packs_epi16 + a lane-unscramble per 32 coefficients — where
// the magetypes i32x8 kernel spends ~8 ops per 8. Bit-identical to the scalar
// port on the FULL i32 domain: the one semantic difference from C-AVX2 itself
// is `min_epu16(abs, 127)` — C lets `packs_epi16` saturate, which maps the
// `packs_epi32(i32::MIN) == -32768` lane to -128 (0x80), while the port's
// contract is `unsigned_abs().min(127) == 127`. Everything reachable
// (|coeff| <= 32767) is identical to both. Non-{4,8,16,32} heights or narrow
// widths fall to the scalar transcription (the generic kernel asserted
// `height % 8 == 0`; the txb-adjusted dims are always in the set, so this is
// defence, not a reachability change).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub(crate) fn txb_init_levels_impl_v3(
    _t: archmage::X64V3Token,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    use archmage::intrinsics::x86_64::*;

    if width == 0
        || (height == 4 && !width.is_multiple_of(4))
        || (height == 8 && !width.is_multiple_of(4))
        || (height == 16 && !width.is_multiple_of(2))
        || !matches!(height, 4 | 8 | 16 | 32)
    {
        crate::txb::txb_init_levels_scalar(coeff, width, height, levels);
        return;
    }

    let stride = height + TX_PAD_HOR;
    let tail = stride * width;
    // One preflight covers every access below: short buffers take the scalar
    // path (which panics on them identically — same contract, no reachable
    // behaviour change), and the fast path is then check-free by shape.
    if levels.len() < tail + TX_PAD_BOTTOM * stride + TX_PAD_END || coeff.len() < width * height {
        crate::txb::txb_init_levels_scalar(coeff, width, height, levels);
        return;
    }

    // The tail pad is 48..160 bytes (4*stride+16, stride in {8,12,20,36}) —
    // always a multiple of 16. One aggregate copy_from_slice per arm: a
    // comptime-length copy lowers to inline vector stores, while a byte-wise
    // loop or `.fill(0)` on a dynamic slice gets idiom-recognised back into a
    // memset call (~106 Ir/call at 1.7M calls — measured).
    let pad_len = TX_PAD_BOTTOM * stride + TX_PAD_END;
    match pad_len {
        48 => levels[tail..tail + 48].copy_from_slice(&[0u8; 48]),
        64 => levels[tail..tail + 64].copy_from_slice(&[0u8; 64]),
        96 => levels[tail..tail + 96].copy_from_slice(&[0u8; 96]),
        _ => levels[tail..tail + 160].copy_from_slice(&[0u8; 160]),
    }

    let zero = _mm256_setzero_si256();
    let cap127 = _mm256_set1_epi16(127);
    let load = |cf: &[i32; 8]| -> __m256i { _mm256_loadu_si256(cf) };
    // abs_epi16 then min_epu16(127): packs_epi32 has already saturated the
    // i32 input to i16, and the unsigned min also maps the -32768 lane
    // (i32::MIN) to 127 — the scalar port's exact result on every input.
    let abs127_16 = |v: __m256i| _mm256_min_epu16(_mm256_abs_epi16(v), cap127);

    // Body loops walk fixed-size array chunks — after `as_chunks`, `c` is a
    // `&[i32; N]` and `o` a `&mut [u8; N]`, so every literal sub-range inside
    // is comptime-in-range and lowers to a bare pointer; the earlier flat
    // `coeff[q*32..]` form still paid ~23 Ir/call of checked_sub bounds
    // machinery at 1.7M calls (measured, uint_macros.rs in the profile).
    match height {
        8 => {
            // 32 coeffs = 4 columns of 8; res holds the four 8-byte column
            // bodies in order, and each column's pad is a 4B zero store.
            let (c32, _) = coeff[..width * 8].as_chunks::<32>();
            let (o48, _) = levels[..tail].as_chunks_mut::<48>();
            for (q, o) in o48.iter_mut().enumerate() {
                let c = &c32[q];
                let ab = _mm256_packs_epi32(
                    load(c[..8].try_into().unwrap()),
                    load(c[8..16].try_into().unwrap()),
                );
                let cd = _mm256_packs_epi32(
                    load(c[16..24].try_into().unwrap()),
                    load(c[24..].try_into().unwrap()),
                );
                let v = _mm256_packs_epi16(abs127_16(ab), abs127_16(cd));
                let r = _mm256_shuffle_epi32(_mm256_permute4x64_epi64(v, 0xd8), 0xd8);
                let r0 = _mm256_castsi256_si128(r);
                let r1 = _mm256_extracti128_si256(r, 1);
                // Column stride 12: 8 data bytes + 4 pad zeros. One 8B store
                // + one 4B store covers each column outright — no separate
                // pad fill.
                let b0: &mut [u8; 8] = (&mut o[..8]).try_into().unwrap();
                _mm_storeu_si64(b0, r0);
                o[8..12].copy_from_slice(&[0u8; 4]);
                let b1: &mut [u8; 8] = (&mut o[12..20]).try_into().unwrap();
                _mm_storeu_si64(b1, _mm_srli_si128(r0, 8));
                o[20..24].copy_from_slice(&[0u8; 4]);
                let b2: &mut [u8; 8] = (&mut o[24..32]).try_into().unwrap();
                _mm_storeu_si64(b2, r1);
                o[32..36].copy_from_slice(&[0u8; 4]);
                let b3: &mut [u8; 8] = (&mut o[36..44]).try_into().unwrap();
                _mm_storeu_si64(b3, _mm_srli_si128(r1, 8));
                o[44..48].copy_from_slice(&[0u8; 4]);
            }
        }
        4 => {
            // 16 coeffs = 4 columns; the pad zeros are interleaved by
            // packs_epi16 against a zero vector, then both shuffles restore
            // column order — one 32B store covers 4 columns (stride 8).
            let (c16, _) = coeff[..width * 4].as_chunks::<16>();
            let (o32, _) = levels[..tail].as_chunks_mut::<32>();
            for (q, o) in o32.iter_mut().enumerate() {
                let c = &c16[q];
                let p = _mm256_packs_epi32(
                    load(c[..8].try_into().unwrap()),
                    load(c[8..].try_into().unwrap()),
                );
                let v = _mm256_packs_epi16(abs127_16(p), zero);
                let r = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(v, 0xd8), 0xd8);
                _mm256_storeu_si256(o, r);
            }
        }

        16 => {
            // 32 coeffs = 2 columns of 16; res = [col | col].
            let (c32, _) = coeff[..width * 16].as_chunks::<32>();
            let (o40, _) = levels[..tail].as_chunks_mut::<40>();
            for (q, o) in o40.iter_mut().enumerate() {
                let c = &c32[q];
                let ab = _mm256_packs_epi32(
                    load(c[..8].try_into().unwrap()),
                    load(c[8..16].try_into().unwrap()),
                );
                let cd = _mm256_packs_epi32(
                    load(c[16..24].try_into().unwrap()),
                    load(c[24..].try_into().unwrap()),
                );
                let v = _mm256_packs_epi16(abs127_16(ab), abs127_16(cd));
                let r = _mm256_shuffle_epi32(_mm256_permute4x64_epi64(v, 0xd8), 0xd8);
                let b0: &mut [u8; 16] = (&mut o[..16]).try_into().unwrap();
                _mm_storeu_si128(b0, _mm256_castsi256_si128(r));
                o[16..20].copy_from_slice(&[0u8; 4]);
                let b1: &mut [u8; 16] = (&mut o[20..36]).try_into().unwrap();
                _mm_storeu_si128(b1, _mm256_extracti128_si256(r, 1));
                o[36..40].copy_from_slice(&[0u8; 4]);
            }
        }
        _ => {
            // height == 32 (txb dims never reach 64 — `adjusted_tx_size`
            // caps them): 32 coeffs = ONE column; res is the whole body.
            let (c32, _) = coeff[..width * 32].as_chunks::<32>();
            let (o36, _) = levels[..tail].as_chunks_mut::<36>();
            for (q, o) in o36.iter_mut().enumerate() {
                let c = &c32[q];
                let ab = _mm256_packs_epi32(
                    load(c[..8].try_into().unwrap()),
                    load(c[8..16].try_into().unwrap()),
                );
                let cd = _mm256_packs_epi32(
                    load(c[16..24].try_into().unwrap()),
                    load(c[24..].try_into().unwrap()),
                );
                let v = _mm256_packs_epi16(abs127_16(ab), abs127_16(cd));
                let r = _mm256_shuffle_epi32(_mm256_permute4x64_epi64(v, 0xd8), 0xd8);
                let b: &mut [u8; 32] = (&mut o[..32]).try_into().unwrap();
                _mm256_storeu_si256(b, r);
                o[32..36].copy_from_slice(&[0u8; 4]);
            }
        }
    }
}

// ================= av1_get_nz_map_contexts_sse2 (encodetxb_sse2.c) =================
//
// C's SSE2 computes the base context of EVERY raster position — the padded
// levels buffer is walked in raster order in 4x4 / 8x2 / 16x1 tiles of 16
// outputs each, eob-independent — then patches scan[eob-1] with the eob bucket.
// The port's contract (and scalar C's) writes only `scan[..eob]` and touches
// nothing else, so the raster pass computes into a stack scratch and the
// eob positions are scattered out of it.
//
// Bit-exactness: the kernel is `min(l,3)` summed over the five neighbours,
// `avg(count, 0) == (count+1)>>1`, `min(count,4)` — identical to the scalar
// `get_nz_mag` + `(stats+1)>>1).min(4)` — and the `pos_to_offset` tables are
// transcribed verbatim. Neighbour offsets per class (flat, padded stride s):
//   2D:    {0,1}=+1  {1,0}=+s  {0,2}=+2  {1,1}=+s+1  {2,0}=+2s
//   HORIZ: {0,1}=+1  {1,0}=+s  {0,2..4}=+2s,+3s,+4s
//   VERT:  {0,1}=+1  {1,0}=+s  {2..4,0}=+2,+3,+4
// Every load/store goes through bounds-checked slice indexing — the padded
// buffer (TX_PAD_2D) contains every tile read for every txb geometry.

/// Scalar tiers = the transcribed scan-order walk, verbatim.
pub(crate) fn nz_map_contexts_impl_scalar(
    _t: archmage::ScalarToken,
    levels: &[u8],
    scan: &[i16],
    eob: usize,
    tx_size: usize,
    tx_class: TxClass,
    coeff_contexts: &mut [i8],
) {
    crate::txb::nz_map_contexts_scalar(levels, scan, eob, tx_size, tx_class, coeff_contexts)
}

/// Non-x86 tiers run the same scalar walk — the raster-tile recipe is an
/// SSE2 shape; a Neon/wasm port can widen this later without touching the
/// dispatch site.
#[magetypes(neon, wasm128, -scalar)]
pub(crate) fn nz_map_contexts_impl(
    _t: Token,
    levels: &[u8],
    scan: &[i16],
    eob: usize,
    tx_size: usize,
    tx_class: TxClass,
    coeff_contexts: &mut [i8],
) {
    crate::txb::nz_map_contexts_scalar(levels, scan, eob, tx_size, tx_class, coeff_contexts)
}

thread_local! {
    /// Raster-tile scratch for the v3 nz-map body. The tile walk writes every
    /// position `0..width*height` before the scan scatter reads `ctx[p]` —
    /// the scatter's `p` is always inside the written raster area — so the
    /// buffer needs no per-call init; pooling it removes a 1 KB memset per
    /// call (~107M Ir at the 512x512 s3 profile, measured).
    static NZ_CTX: core::cell::RefCell<[i8; 32 * 32]> =
        const { core::cell::RefCell::new([0; 32 * 32]) };
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub(crate) fn nz_map_contexts_impl_v3(
    _t: archmage::X64V3Token,
    levels: &[u8],
    scan: &[i16],
    eob: usize,
    tx_size: usize,
    tx_class: TxClass,
    coeff_contexts: &mut [i8],
) {
    // C: `if (!last_idx) { coeff_contexts[0] = 0; return; }`. eob==0 is
    // unreachable in C (scan[-1] would be UB); the port's contract is a
    // no-op, kept verbatim.
    if eob <= 1 {
        if eob == 1 {
            coeff_contexts[0] = 0;
        }
        return;
    }
    NZ_CTX.with_borrow_mut(|ctx| {
        nz_map_ctx_body(ctx, levels, scan, eob, tx_size, tx_class, coeff_contexts)
    });
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn nz_map_ctx_body(
    ctx: &mut [i8; 32 * 32],
    levels: &[u8],
    scan: &[i16],
    eob: usize,
    tx_size: usize,
    tx_class: TxClass,
    coeff_contexts: &mut [i8],
) {
    use crate::txb::{txb_high, txb_wide};
    use archmage::intrinsics::x86_64::*;

    let width = txb_wide(tx_size); // padded-buffer column count (C `width`)
    let height = txb_high(tx_size); // padded column length (C `height`)
    let stride = height + TX_PAD_HOR;
    let area = width * height;

    let c3 = _mm_set1_epi8(3);
    let c4 = _mm_set1_epi8(4);
    // get_coeff_contexts_kernel_sse2.
    let kernel = |l: [__m128i; 5]| -> __m128i {
        let mut count = _mm_min_epu8(l[0], c3);
        count = _mm_add_epi8(count, _mm_min_epu8(l[1], c3));
        count = _mm_add_epi8(count, _mm_min_epu8(l[2], c3));
        count = _mm_add_epi8(count, _mm_min_epu8(l[3], c3));
        count = _mm_add_epi8(count, _mm_min_epu8(l[4], c3));
        _mm_min_epu8(_mm_avg_epu8(count, _mm_setzero_si128()), c4)
    };

    // The three tile shapes of load_levels_*x5_sse2. `base` is the tile's
    // padded-buffer offset; each lane-vector gathers one neighbour plane.
    // One range check per tile load buys out the per-row checks: on a
    // `k_max*stride + N` window each `w[k*stride .. k*stride+N]` bound folds
    // to `k <= k_max`. The window end is the old code's largest access, so
    // the panic domain is unchanged.
    let t4 = |base: usize, off: usize| -> __m128i {
        let w = &levels[base + off..base + off + 3 * stride + 4];
        let r0: &[u8; 4] = w[..4].try_into().unwrap();
        let r1: &[u8; 4] = w[stride..stride + 4].try_into().unwrap();
        let r2: &[u8; 4] = w[2 * stride..2 * stride + 4].try_into().unwrap();
        let r3: &[u8; 4] = w[3 * stride..3 * stride + 4].try_into().unwrap();
        _mm_unpacklo_epi64(
            _mm_unpacklo_epi32(_mm_loadu_si32(r0), _mm_loadu_si32(r1)),
            _mm_unpacklo_epi32(_mm_loadu_si32(r2), _mm_loadu_si32(r3)),
        )
    };
    let t8 = |base: usize, off: usize| -> __m128i {
        let w = &levels[base + off..base + off + stride + 8];
        let r0: &[u8; 8] = w[..8].try_into().unwrap();
        let r1: &[u8; 8] = w[stride..stride + 8].try_into().unwrap();
        _mm_unpacklo_epi64(_mm_loadu_si64(r0), _mm_loadu_si64(r1))
    };
    let t16 = |base: usize, off: usize| -> __m128i {
        let r: &[u8; 16] = levels[base + off..base + off + 16].try_into().unwrap();
        _mm_loadu_si128(r)
    };

    macro_rules! store16 {
        ($cc:expr, $v:expr) => {{
            let d: &mut [i8; 16] = (&mut ctx[$cc..$cc + 16]).try_into().unwrap();
            _mm_storeu_si128(d, $v);
        }};
    }

    match tx_class {
        TxClass::TwoD => {
            let offs = [2usize, stride + 1, 2 * stride];
            if height == 4 {
                // get_4_nz_map_contexts_2d.
                let mut pto = if width == 4 {
                    _mm_setr_epi8(0, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21)
                } else {
                    _mm_setr_epi8(0, 16, 16, 16, 16, 16, 16, 16, 6, 6, 21, 21, 6, 21, 21, 21)
                };
                let big = _mm_set1_epi8(21);
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                while col != 0 {
                    let l = [
                        t4(lv, 1),
                        t4(lv, stride),
                        t4(lv, offs[0]),
                        t4(lv, offs[1]),
                        t4(lv, offs[2]),
                    ];
                    store16!(cc, _mm_add_epi8(kernel(l), pto));
                    pto = big;
                    lv += 4 * stride;
                    cc += 16;
                    col -= 4;
                }
                ctx[0] = 0;
            } else if height == 8 {
                // get_8_coeff_contexts_2d: three rotating tables by width class.
                let (t0, t1) = if width == 8 {
                    (
                        _mm_setr_epi8(0, 1, 6, 6, 21, 21, 21, 21, 1, 6, 6, 21, 21, 21, 21, 21),
                        _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21),
                    )
                } else if width < 8 {
                    (
                        _mm_setr_epi8(0, 11, 6, 6, 21, 21, 21, 21, 11, 11, 6, 21, 21, 21, 21, 21),
                        _mm_setr_epi8(
                            11, 11, 21, 21, 21, 21, 21, 21, 11, 11, 21, 21, 21, 21, 21, 21,
                        ),
                    )
                } else {
                    (
                        _mm_setr_epi8(
                            0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                        ),
                        _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21),
                    )
                };
                let mut pto = [t0, t1, _mm_set1_epi8(21)];
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                while col != 0 {
                    let l = [
                        t8(lv, 1),
                        t8(lv, stride),
                        t8(lv, offs[0]),
                        t8(lv, offs[1]),
                        t8(lv, offs[2]),
                    ];
                    store16!(cc, _mm_add_epi8(kernel(l), pto[0]));
                    pto[0] = pto[1];
                    pto[1] = pto[2];
                    lv += 2 * stride;
                    cc += 16;
                    col -= 2;
                }
                ctx[0] = 0;
            } else {
                // get_16n_coeff_contexts_2d: 16x1 tiles, five rotating tables
                // keyed on the REAL (untransposed, unadjusted) tx dims.
                let real_width = super::TX_SIZE_WIDE[tx_size];
                let real_height = super::TX_SIZE_HIGH[tx_size];
                let t21 = _mm_set1_epi8(21);
                let mut pto = [t21; 5];
                let mut large = [t21; 3];
                if real_width == real_height {
                    pto[0] =
                        _mm_setr_epi8(0, 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                    pto[1] =
                        _mm_setr_epi8(1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                    pto[2] =
                        _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                    pto[3] = _mm_setr_epi8(
                        6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                    );
                } else if real_width < real_height {
                    pto[0] =
                        _mm_setr_epi8(0, 11, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                    pto[1] = _mm_setr_epi8(
                        11, 11, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                    );
                    pto[2] = _mm_setr_epi8(
                        11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                    );
                    pto[3] = pto[2];
                    pto[4] = pto[2];
                } else {
                    let t16 = _mm_set1_epi8(16);
                    pto[0] = t16;
                    pto[1] = t16;
                    pto[2] =
                        _mm_setr_epi8(6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                    pto[3] = _mm_setr_epi8(
                        6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                    );
                    pto[4] = t21;
                    large = [t16, t16, t21];
                }
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                loop {
                    let mut h = height;
                    while h != 0 {
                        let l = [
                            t16(lv, 1),
                            t16(lv, stride),
                            t16(lv, offs[0]),
                            t16(lv, offs[1]),
                            t16(lv, offs[2]),
                        ];
                        store16!(cc, _mm_add_epi8(kernel(l), pto[0]));
                        pto[0] = large[0];
                        lv += 16;
                        cc += 16;
                        h -= 16;
                    }
                    pto[0] = pto[1];
                    pto[1] = pto[2];
                    pto[2] = pto[3];
                    pto[3] = pto[4];
                    large[0] = large[1];
                    large[1] = large[2];
                    lv += TX_PAD_HOR;
                    col -= 1;
                    if col == 0 {
                        break;
                    }
                }
                ctx[0] = 0;
            }
        }
        TxClass::Horiz => {
            let offs = [2 * stride, 3 * stride, 4 * stride];
            let (s0, s5, s10) = (26i8, 31i8, 36i8); // SIG_COEF_CONTEXTS_2D + {0,5,10}
            if height == 4 {
                // get_4_nz_map_contexts_hor.
                let mut pto = _mm_setr_epi8(
                    s0, s0, s0, s0, s5, s5, s5, s5, s10, s10, s10, s10, s10, s10, s10, s10,
                );
                let big = _mm_set1_epi8(s10);
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                while col != 0 {
                    let l = [
                        t4(lv, 1),
                        t4(lv, stride),
                        t4(lv, offs[0]),
                        t4(lv, offs[1]),
                        t4(lv, offs[2]),
                    ];
                    store16!(cc, _mm_add_epi8(kernel(l), pto));
                    pto = big;
                    lv += 4 * stride;
                    cc += 16;
                    col -= 4;
                }
            } else if height == 8 {
                // get_8_coeff_contexts_hor.
                let mut pto = _mm_setr_epi8(
                    s0, s0, s0, s0, s0, s0, s0, s0, s5, s5, s5, s5, s5, s5, s5, s5,
                );
                let big = _mm_set1_epi8(s10);
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                while col != 0 {
                    let l = [
                        t8(lv, 1),
                        t8(lv, stride),
                        t8(lv, offs[0]),
                        t8(lv, offs[1]),
                        t8(lv, offs[2]),
                    ];
                    store16!(cc, _mm_add_epi8(kernel(l), pto));
                    pto = big;
                    lv += 2 * stride;
                    cc += 16;
                    col -= 2;
                }
            } else {
                // get_16n_coeff_contexts_hor: the whole column shares one
                // offset (26 -> 31 -> 36 by column), no per-tile rotation.
                let mut pto = [_mm_set1_epi8(s0), _mm_set1_epi8(s5), _mm_set1_epi8(s10)];
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                loop {
                    let mut h = height;
                    while h != 0 {
                        let l = [
                            t16(lv, 1),
                            t16(lv, stride),
                            t16(lv, offs[0]),
                            t16(lv, offs[1]),
                            t16(lv, offs[2]),
                        ];
                        store16!(cc, _mm_add_epi8(kernel(l), pto[0]));
                        lv += 16;
                        cc += 16;
                        h -= 16;
                    }
                    pto[0] = pto[1];
                    pto[1] = pto[2];
                    lv += TX_PAD_HOR;
                    col -= 1;
                    if col == 0 {
                        break;
                    }
                }
            }
        }
        TxClass::Vert => {
            let offs = [2usize, 3, 4];
            let (s0, s5, s10) = (26i8, 31i8, 36i8);
            if height == 4 {
                // get_4_nz_map_contexts_ver: same table every tile.
                let pto = _mm_setr_epi8(
                    s0, s5, s10, s10, s0, s5, s10, s10, s0, s5, s10, s10, s0, s5, s10, s10,
                );
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                while col != 0 {
                    let l = [
                        t4(lv, 1),
                        t4(lv, stride),
                        t4(lv, offs[0]),
                        t4(lv, offs[1]),
                        t4(lv, offs[2]),
                    ];
                    store16!(cc, _mm_add_epi8(kernel(l), pto));
                    lv += 4 * stride;
                    cc += 16;
                    col -= 4;
                }
            } else if height == 8 {
                // get_8_coeff_contexts_ver.
                let pto = _mm_setr_epi8(
                    s0, s5, s10, s10, s10, s10, s10, s10, s0, s5, s10, s10, s10, s10, s10, s10,
                );
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                while col != 0 {
                    let l = [
                        t8(lv, 1),
                        t8(lv, stride),
                        t8(lv, offs[0]),
                        t8(lv, offs[1]),
                        t8(lv, offs[2]),
                    ];
                    store16!(cc, _mm_add_epi8(kernel(l), pto));
                    lv += 2 * stride;
                    cc += 16;
                    col -= 2;
                }
            } else {
                // get_16n_coeff_contexts_ver: first 16-tile per column uses
                // [26,31,36..36], the rest all-36.
                let big = _mm_set1_epi8(s10);
                let (mut lv, mut cc, mut col) = (0usize, 0usize, width);
                loop {
                    let mut pto = _mm_setr_epi8(
                        s0, s5, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10,
                        s10,
                    );
                    let mut h = height;
                    while h != 0 {
                        let l = [
                            t16(lv, 1),
                            t16(lv, stride),
                            t16(lv, offs[0]),
                            t16(lv, offs[1]),
                            t16(lv, offs[2]),
                        ];
                        store16!(cc, _mm_add_epi8(kernel(l), pto));
                        pto = big;
                        lv += 16;
                        cc += 16;
                        h -= 16;
                    }
                    lv += TX_PAD_HOR;
                    col -= 1;
                    if col == 0 {
                        break;
                    }
                }
            }
        }
    }

    // Scatter scan[..eob] — positions past eob keep their prior contents
    // (scalar-C contract; the differential harness byte-compares the tail).
    // The `scan[..eob]` pre-slice is the old loop's own OOB domain
    // (first failing `scan[i]` panicked at the same `eob > scan.len()`).
    for &p16 in &scan[..eob] {
        let p = p16 as usize;
        coeff_contexts[p] = ctx[p];
    }
    // EOB bucket overrides the computed context (get_nz_map_ctx's is_eob arm;
    // C patches coeff_contexts[scan[last_idx]] the same way).
    let last = eob - 1;
    let pos = scan[last] as usize;
    coeff_contexts[pos] = if last <= area / 8 {
        1
    } else if last <= area / 4 {
        2
    } else {
        3
    };
}
