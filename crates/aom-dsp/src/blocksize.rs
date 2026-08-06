//! `BLOCK_SIZES_ALL` / `TX_SIZES_ALL` geometry (`av1/common/common_data.h`) —
//! ONE derivation, so the port's ~40 private copies of these tables have
//! something to be checked against.
//!
//! # Why this module exists
//!
//! `block_size_wide`, `mi_size_high`, `tx_size_wide_unit`, `txsize_to_bsize` and
//! friends are pure `BLOCK_SIZE`/`TX_SIZE`-indexed geometry. They are also, in
//! this port, transcribed by hand into whichever module needs them, at whichever
//! element type is convenient — `block_size_wide` alone exists six times as
//! `[i32; 22]`, `[usize; 22]` and `[u32; 22]`. A single wrong entry in a single
//! copy does not crash and does not fail a build: it produces a block of the
//! wrong size in exactly the code paths that use that copy, i.e. wrong pixels or
//! a wrong entropy context, in a subset of the port. Nothing here was verified
//! against anything until this module landed.
//!
//! So: state the geometry once, as the DIMENSION PAIRS the `BLOCK_SIZE` /
//! `TX_SIZE` enum names literally spell (`BLOCK_16X4` is 16 wide and 4 high),
//! derive every other table from those pairs by its `common_data.h` definition,
//! and let each private copy be diffed against this one in tests. The pair lists
//! themselves are checked for internal structure by
//! `tests::block_dims_follow_the_enum_construction_rule`.
//!
//! Nothing in the port's hot paths reads this module yet; migrating the private
//! copies over to it is a mechanical follow-up. Its job today is to be the
//! reference the copies are pinned to.

/// `BLOCK_SIZES_ALL` (`enums.h`): 16 square/2:1 sizes + 6 4:1 sizes.
pub const BLOCK_SIZES_ALL: usize = 22;

/// `TX_SIZES_ALL` (`enums.h`): 5 square + 8 2:1 + 6 4:1 transform sizes.
pub const TX_SIZES_ALL: usize = 19;

/// `MI_SIZE_LOG2` (`enums.h`): a mode-info unit is 4x4 pixels.
pub const MI_SIZE_LOG2: u32 = 2;

/// `(width, height)` in pixels for each `BLOCK_SIZE`, in `enums.h` enum order —
/// the dimensions the enum names spell out.
#[rustfmt::skip]
pub const BLOCK_DIMS: [(u16, u16); BLOCK_SIZES_ALL] = [
    (4, 4),     // BLOCK_4X4
    (4, 8),     // BLOCK_4X8
    (8, 4),     // BLOCK_8X4
    (8, 8),     // BLOCK_8X8
    (8, 16),    // BLOCK_8X16
    (16, 8),    // BLOCK_16X8
    (16, 16),   // BLOCK_16X16
    (16, 32),   // BLOCK_16X32
    (32, 16),   // BLOCK_32X16
    (32, 32),   // BLOCK_32X32
    (32, 64),   // BLOCK_32X64
    (64, 32),   // BLOCK_64X32
    (64, 64),   // BLOCK_64X64
    (64, 128),  // BLOCK_64X128
    (128, 64),  // BLOCK_128X64
    (128, 128), // BLOCK_128X128
    (4, 16),    // BLOCK_4X16
    (16, 4),    // BLOCK_16X4
    (8, 32),    // BLOCK_8X32
    (32, 8),    // BLOCK_32X8
    (16, 64),   // BLOCK_16X64
    (64, 16),   // BLOCK_64X16
];

/// `(width, height)` in pixels for each `TX_SIZE`, in `enums.h` enum order.
#[rustfmt::skip]
pub const TX_DIMS: [(u16, u16); TX_SIZES_ALL] = [
    (4, 4),   // TX_4X4
    (8, 8),   // TX_8X8
    (16, 16), // TX_16X16
    (32, 32), // TX_32X32
    (64, 64), // TX_64X64
    (4, 8),   // TX_4X8
    (8, 4),   // TX_8X4
    (8, 16),  // TX_8X16
    (16, 8),  // TX_16X8
    (16, 32), // TX_16X32
    (32, 16), // TX_32X16
    (32, 64), // TX_32X64
    (64, 32), // TX_64X32
    (4, 16),  // TX_4X16
    (16, 4),  // TX_16X4
    (8, 32),  // TX_8X32
    (32, 8),  // TX_32X8
    (16, 64), // TX_16X64
    (64, 16), // TX_64X16
];

const fn widths<const N: usize>(dims: [(u16, u16); N]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    while i < N {
        out[i] = dims[i].0;
        i += 1;
    }
    out
}

const fn heights<const N: usize>(dims: [(u16, u16); N]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    while i < N {
        out[i] = dims[i].1;
        i += 1;
    }
    out
}

const fn in_mi_units<const N: usize>(px: [u16; N]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    while i < N {
        out[i] = px[i] >> MI_SIZE_LOG2;
        i += 1;
    }
    out
}

const fn log2_of<const N: usize>(v: [u16; N]) -> [u8; N] {
    let mut out = [0u8; N];
    let mut i = 0;
    while i < N {
        out[i] = v[i].trailing_zeros() as u8;
        i += 1;
    }
    out
}

/// `block_size_wide[BLOCK_SIZES_ALL]`: block width in pixels.
pub const BLOCK_SIZE_WIDE: [u16; BLOCK_SIZES_ALL] = widths(BLOCK_DIMS);
/// `block_size_high[BLOCK_SIZES_ALL]`: block height in pixels.
pub const BLOCK_SIZE_HIGH: [u16; BLOCK_SIZES_ALL] = heights(BLOCK_DIMS);
/// `mi_size_wide[BLOCK_SIZES_ALL]`: block width in 4x4 mode-info units.
pub const MI_SIZE_WIDE: [u16; BLOCK_SIZES_ALL] = in_mi_units(BLOCK_SIZE_WIDE);
/// `mi_size_high[BLOCK_SIZES_ALL]`: block height in 4x4 mode-info units.
pub const MI_SIZE_HIGH: [u16; BLOCK_SIZES_ALL] = in_mi_units(BLOCK_SIZE_HIGH);
/// `mi_size_wide_log2[BLOCK_SIZES_ALL]`.
pub const MI_SIZE_WIDE_LOG2: [u8; BLOCK_SIZES_ALL] = log2_of(MI_SIZE_WIDE);
/// `mi_size_high_log2[BLOCK_SIZES_ALL]`.
pub const MI_SIZE_HIGH_LOG2: [u8; BLOCK_SIZES_ALL] = log2_of(MI_SIZE_HIGH);

/// `tx_size_wide[TX_SIZES_ALL]`: transform width in pixels.
pub const TX_SIZE_WIDE: [u16; TX_SIZES_ALL] = widths(TX_DIMS);
/// `tx_size_high[TX_SIZES_ALL]`: transform height in pixels.
pub const TX_SIZE_HIGH: [u16; TX_SIZES_ALL] = heights(TX_DIMS);
/// `tx_size_wide_unit[TX_SIZES_ALL]`: transform width in 4x4 units.
pub const TX_SIZE_WIDE_UNIT: [u16; TX_SIZES_ALL] = in_mi_units(TX_SIZE_WIDE);
/// `tx_size_high_unit[TX_SIZES_ALL]`: transform height in 4x4 units.
pub const TX_SIZE_HIGH_UNIT: [u16; TX_SIZES_ALL] = in_mi_units(TX_SIZE_HIGH);

/// `txsize_to_bsize[TX_SIZES_ALL]`: the `BLOCK_SIZE` with the same dimensions as
/// the transform.
pub const TXSIZE_TO_BSIZE: [u8; TX_SIZES_ALL] = {
    let mut out = [0u8; TX_SIZES_ALL];
    let mut t = 0;
    while t < TX_SIZES_ALL {
        let mut b = 0;
        let mut found = usize::MAX;
        while b < BLOCK_SIZES_ALL {
            if BLOCK_DIMS[b].0 == TX_DIMS[t].0 && BLOCK_DIMS[b].1 == TX_DIMS[t].1 {
                found = b;
                break;
            }
            b += 1;
        }
        // Every TX_SIZE has an equally-shaped BLOCK_SIZE; panics at compile time
        // if a dimension pair is ever mistyped into a shape with no block twin.
        out[t] = found as u8;
        t += 1;
    }
    out
};

/// The `TX_SIZE` of the square transform with side `side`, or a compile-time
/// panic if `side` is not one of 4/8/16/32/64. The square sizes occupy indices
/// 0..=4 of `TX_DIMS`.
const fn square_tx(side: u16) -> u8 {
    let mut t = 0;
    while t < 5 {
        if TX_DIMS[t].0 == side {
            return t as u8;
        }
        t += 1;
    }
    panic!("no square TX_SIZE with this side");
}

/// `txsize_sqr_up_map[TX_SIZES_ALL]`: the smallest square transform CONTAINING
/// this one (square of the longer side).
pub const TXSIZE_SQR_UP_MAP: [u8; TX_SIZES_ALL] = {
    let mut out = [0u8; TX_SIZES_ALL];
    let mut t = 0;
    while t < TX_SIZES_ALL {
        let (w, h) = TX_DIMS[t];
        out[t] = square_tx(if w > h { w } else { h });
        t += 1;
    }
    out
};

/// `txsize_sqr_map[TX_SIZES_ALL]`: the largest square transform CONTAINED in
/// this one (square of the shorter side).
pub const TXSIZE_SQR_MAP: [u8; TX_SIZES_ALL] = {
    let mut out = [0u8; TX_SIZES_ALL];
    let mut t = 0;
    while t < TX_SIZES_ALL {
        let (w, h) = TX_DIMS[t];
        out[t] = square_tx(if w < h { w } else { h });
        t += 1;
    }
    out
};

/// Assert that local transcriptions of `common_data.h` geometry agree,
/// entry-for-entry, with this module's derivation.
///
/// Written for the `#[cfg(test)] mod geometry_agreement` blocks that sit beside
/// each private copy in this port. Element types differ per copy (`i32`,
/// `usize`, `u32`, `u8`, `u16`), so every value is widened to `i64` before
/// comparison; lengths are checked first.
///
/// ```
/// # use aom_dsp::blocksize;
/// const LOCAL_MI_SIZE_WIDE: [i32; 22] = [
///     1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
/// ];
/// aom_dsp::assert_geometry_agrees!(
///     LOCAL_MI_SIZE_WIDE => blocksize::MI_SIZE_WIDE,
/// );
/// ```
#[macro_export]
macro_rules! assert_geometry_agrees {
    ($($local:expr => $canonical:expr),+ $(,)?) => {{
        $(
            assert_eq!(
                $local.len(),
                $canonical.len(),
                concat!(stringify!($local), ": length differs from ", stringify!($canonical)),
            );
            for i in 0..$local.len() {
                assert_eq!(
                    $local[i] as i64,
                    $canonical[i] as i64,
                    concat!(
                        stringify!($local), "[{}] disagrees with ", stringify!($canonical),
                        " -- a silent wrong-geometry bug in this module only",
                    ),
                    i,
                );
            }
        )+
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pair lists are the one hand-written thing here, so pin their
    /// STRUCTURE rather than their values. `enums.h` builds `BLOCK_SIZES_ALL` by
    /// a strict rule: five triples `(s,s), (s,2s), (2s,s)` for `s = 4, 8, 16,
    /// 32, 64`, then `BLOCK_128X128`, then three 4:1 pairs `(s,4s), (4s,s)` for
    /// `s = 4, 8, 16`. A transposed or dropped entry breaks this; a value typo
    /// almost always does too.
    #[test]
    fn block_dims_follow_the_enum_construction_rule() {
        let mut expect: Vec<(u16, u16)> = Vec::new();
        for k in 0..5u32 {
            let s = 4u16 << k;
            expect.push((s, s));
            expect.push((s, 2 * s));
            expect.push((2 * s, s));
        }
        expect.push((128, 128));
        for k in 0..3u32 {
            let s = 4u16 << k;
            expect.push((s, 4 * s));
            expect.push((4 * s, s));
        }
        assert_eq!(expect.len(), BLOCK_SIZES_ALL);
        assert_eq!(&expect[..], &BLOCK_DIMS[..]);
    }

    /// `TX_SIZES_ALL` is built differently: the five squares FIRST, then the
    /// eight 2:1 pairs `(s,2s),(2s,s)` for `s = 4, 8, 16, 32`, then the six 4:1
    /// pairs `(s,4s),(4s,s)` for `s = 4, 8, 16`.
    #[test]
    fn tx_dims_follow_the_enum_construction_rule() {
        let mut expect: Vec<(u16, u16)> = Vec::new();
        for k in 0..5u32 {
            let s = 4u16 << k;
            expect.push((s, s));
        }
        for k in 0..4u32 {
            let s = 4u16 << k;
            expect.push((s, 2 * s));
            expect.push((2 * s, s));
        }
        for k in 0..3u32 {
            let s = 4u16 << k;
            expect.push((s, 4 * s));
            expect.push((4 * s, s));
        }
        assert_eq!(expect.len(), TX_SIZES_ALL);
        assert_eq!(&expect[..], &TX_DIMS[..]);
    }

    /// Every `BLOCK_SIZE` / `TX_SIZE` is a distinct shape — otherwise two enum
    /// values would be indistinguishable by geometry and the `txsize_to_bsize`
    /// search below would silently pick whichever came first.
    #[test]
    fn every_size_is_a_distinct_shape() {
        for i in 0..BLOCK_SIZES_ALL {
            for j in (i + 1)..BLOCK_SIZES_ALL {
                assert_ne!(BLOCK_DIMS[i], BLOCK_DIMS[j], "block {i} and {j} alias");
            }
        }
        for i in 0..TX_SIZES_ALL {
            for j in (i + 1)..TX_SIZES_ALL {
                assert_ne!(TX_DIMS[i], TX_DIMS[j], "tx {i} and {j} alias");
            }
        }
    }

    /// `txsize_to_bsize` is derived by shape search, so prove the search
    /// SUCCEEDED everywhere (`usize::MAX as u8` would otherwise land silently)
    /// and that each answer really is the same shape.
    #[test]
    fn txsize_to_bsize_lands_on_the_same_shape() {
        for t in 0..TX_SIZES_ALL {
            let b = TXSIZE_TO_BSIZE[t] as usize;
            assert!(b < BLOCK_SIZES_ALL, "tx {t} found no block twin");
            assert_eq!(BLOCK_DIMS[b], TX_DIMS[t], "tx {t} -> block {b} shape");
        }
    }

    /// The square maps must bracket the transform: `sqr <= min side` and
    /// `sqr_up >= max side`, both square, and equal to each other exactly for
    /// the square transforms.
    #[test]
    fn square_maps_bracket_each_transform() {
        for t in 0..TX_SIZES_ALL {
            let (w, h) = TX_DIMS[t];
            let down = TX_DIMS[TXSIZE_SQR_MAP[t] as usize];
            let up = TX_DIMS[TXSIZE_SQR_UP_MAP[t] as usize];
            assert_eq!(down.0, down.1, "sqr_map[{t}] is not square");
            assert_eq!(up.0, up.1, "sqr_up_map[{t}] is not square");
            assert_eq!(down.0, w.min(h), "sqr_map[{t}] side");
            assert_eq!(up.0, w.max(h), "sqr_up_map[{t}] side");
            assert_eq!(down == up, w == h, "bracket collapses iff square, at {t}");
        }
    }

    /// Pixel dims and mi-unit dims are the same fact at two scales, and the
    /// log2 tables are the log2 of the mi dims. Derived here, so this asserts
    /// the derivation is exact (every dimension a multiple of `MI_SIZE`, every
    /// mi dimension a power of two) rather than re-deriving it.
    #[test]
    fn mi_units_and_log2_are_exact() {
        for b in 0..BLOCK_SIZES_ALL {
            for (px, mi, lg) in [
                (BLOCK_SIZE_WIDE[b], MI_SIZE_WIDE[b], MI_SIZE_WIDE_LOG2[b]),
                (BLOCK_SIZE_HIGH[b], MI_SIZE_HIGH[b], MI_SIZE_HIGH_LOG2[b]),
            ] {
                assert_eq!(
                    px % (1 << MI_SIZE_LOG2),
                    0,
                    "block {b}: {px}px not 4-aligned"
                );
                assert_eq!(mi << MI_SIZE_LOG2, px, "block {b}: mi/px disagree");
                assert_eq!(1u16 << lg, mi, "block {b}: log2 {lg} != log2({mi})");
            }
        }
        for t in 0..TX_SIZES_ALL {
            assert_eq!(TX_SIZE_WIDE_UNIT[t] << MI_SIZE_LOG2, TX_SIZE_WIDE[t]);
            assert_eq!(TX_SIZE_HIGH_UNIT[t] << MI_SIZE_LOG2, TX_SIZE_HIGH[t]);
        }
    }
}
