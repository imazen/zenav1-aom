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

use crate::txb::{TX_PAD_BOTTOM, TX_PAD_END, TX_PAD_HOR, TxClass};

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

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
pub(crate) fn txb_init_levels_impl(
    token: Token,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    let stride = height + TX_PAD_HOR;
    let tail = stride * width;
    levels[tail..tail + TX_PAD_BOTTOM * stride + TX_PAD_END].fill(0);

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
                a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8,
                a[4] as u8, a[5] as u8, a[6] as u8, a[7] as u8,
            ];
            out[c * 8..c * 8 + 8].copy_from_slice(&bytes);
        }
        out[height..height + TX_PAD_HOR].fill(0);
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
    use archmage::intrinsics::x86_64::*;
    use crate::txb::{txb_high, txb_wide};

    // C: `if (!last_idx) { coeff_contexts[0] = 0; return; }`. eob==0 is
    // unreachable in C (scan[-1] would be UB); the port's contract is a
    // no-op, kept verbatim.
    if eob <= 1 {
        if eob == 1 {
            coeff_contexts[0] = 0;
        }
        return;
    }

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
    let t4 = |base: usize, off: usize| -> __m128i {
        let r0: &[u8; 4] = levels[base + off..base + off + 4].try_into().unwrap();
        let r1: &[u8; 4] =
            levels[base + off + stride..base + off + stride + 4].try_into().unwrap();
        let r2: &[u8; 4] =
            levels[base + off + 2 * stride..base + off + 2 * stride + 4].try_into().unwrap();
        let r3: &[u8; 4] =
            levels[base + off + 3 * stride..base + off + 3 * stride + 4].try_into().unwrap();
        _mm_unpacklo_epi64(
            _mm_unpacklo_epi32(_mm_loadu_si32(r0), _mm_loadu_si32(r1)),
            _mm_unpacklo_epi32(_mm_loadu_si32(r2), _mm_loadu_si32(r3)),
        )
    };
    let t8 = |base: usize, off: usize| -> __m128i {
        let r0: &[u8; 8] = levels[base + off..base + off + 8].try_into().unwrap();
        let r1: &[u8; 8] =
            levels[base + off + stride..base + off + stride + 8].try_into().unwrap();
        _mm_unpacklo_epi64(_mm_loadu_si64(r0), _mm_loadu_si64(r1))
    };
    let t16 = |base: usize, off: usize| -> __m128i {
        let r: &[u8; 16] = levels[base + off..base + off + 16].try_into().unwrap();
        _mm_loadu_si128(r)
    };

    let mut ctx = [0i8; 32 * 32];
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
                        _mm_setr_epi8(11, 11, 21, 21, 21, 21, 21, 21, 11, 11, 21, 21, 21, 21, 21, 21),
                    )
                } else {
                    (
                        _mm_setr_epi8(0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16),
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
                    pto[3] =
                        _mm_setr_epi8(6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                } else if real_width < real_height {
                    pto[0] =
                        _mm_setr_epi8(0, 11, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                    pto[1] =
                        _mm_setr_epi8(11, 11, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21);
                    pto[2] = _mm_setr_epi8(
                        11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                    );
                    pto[3] = pto[2];
                    pto[4] = pto[2];
                } else {
                    let t16 = _mm_set1_epi8(16);
                    pto[0] = t16;
                    pto[1] = t16;
                    pto[2] = _mm_setr_epi8(
                        6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                    );
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
                let mut pto =
                    _mm_setr_epi8(s0, s0, s0, s0, s5, s5, s5, s5, s10, s10, s10, s10, s10, s10, s10, s10);
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
                let mut pto =
                    _mm_setr_epi8(s0, s0, s0, s0, s0, s0, s0, s0, s5, s5, s5, s5, s5, s5, s5, s5);
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
                let pto =
                    _mm_setr_epi8(s0, s5, s10, s10, s0, s5, s10, s10, s0, s5, s10, s10, s0, s5, s10, s10);
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
                let pto =
                    _mm_setr_epi8(s0, s5, s10, s10, s10, s10, s10, s10, s0, s5, s10, s10, s10, s10, s10, s10);
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
                        s0, s5, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10, s10,
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
    for i in 0..eob {
        let p = scan[i] as usize;
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
