//! `census` — count what the encoder actually *does* to a source image, so a
//! harness content can be checked against a reference instead of assumed.
//!
//! # Why this exists
//!
//! `benchmarks/winperf_windows_2026-08-02.md` built two synthetic sources
//! (`aom_bench::winperf::Content::{Detail, Smooth}`) and tuned them until their
//! **allocator call count** bracketed the dev box's study photograph. That made
//! them the right harness for a lever aimed at allocation (KB-PERF-2) and for
//! one aimed at the forward transform (KB-PERF-3) — both of which touch every
//! block regardless of how it is coded.
//!
//! It did not make them representative of anything else. When KB-PERF-4 landed
//! a lever inside the *directional intra predictors* and read a winperf band
//! for it, the band was measuring content in which **`z1` fires six times in a
//! whole 1 MP frame** (`benchmarks/encoder_intra_dir_i16_2026-08-03.md` §7).
//! The Windows result was reported honestly as "not resolved", but the reason
//! was structural: the code under test was never executed. A null you cannot
//! distinguish from "this never runs" is not a measurement
//! (`docs/DIFFERENTIAL_PLAYBOOK.md` §6b).
//!
//! That census was throwaway instrumentation, applied by hand from a patch kept
//! outside the tree. This module is the same census made **committed and
//! re-runnable**, which is what turns "is this content representative?" from an
//! argument into a table.
//!
//! # Cost when it is off: zero
//!
//! Every `note_*` entry point below has an empty body unless the `census`
//! cargo feature is on, so the timing binaries the `winperf` workflow builds
//! contain no counter at all. Turning the feature ON changes performance and
//! the resulting binary must never be used for a timing band — census runs and
//! timing runs are separate builds, on purpose.
//!
//! # Scope
//!
//! Four axes, chosen because they are the four a content choice can silently
//! get wrong:
//!
//! * **intra prediction mode family** — the axis that broke. Classified the way
//!   a predictor lever cares about: filter-intra / non-directional / `z1` /
//!   `z2` / `z3` / exactly-`V` / exactly-`H`, split by `tx_size`, counted in
//!   both calls and predicted pixels (the two differ by 4x between the
//!   extremes, and a per-pixel kernel is priced by the second).
//! * **forward transform type x size** — what a transform lever reaches.
//! * **coded leaf block size** — the partition-depth distribution, taken at the
//!   bitstream writer, so it is the *decision*, not a search visit.
//! * allocator calls are counted separately and already were, by
//!   `crates/aom-bench/examples/winperf_alloc.rs`; nothing here duplicates it.
//!
//! Counters are **thread-local**, matching
//! `aom_encode::nonrd_pickmode::multi_txb_leaf_counts`: `cargo test` runs one
//! binary's tests concurrently, so a process-global counter turns
//! `reset / encode / read` into a race. The census encodes are single-threaded,
//! so per-thread is exactly the right granularity.

/// Intra classes, in the order [`Counts::intra_calls`] indexes them.
pub const INTRA_CLASS: [&str; 7] =
    ["filter", "non-dir", "z1", "z2", "z3", "V(90)", "H(180)"];

/// `TX_SIZES_ALL`.
pub const N_TX_SIZE: usize = 19;
/// `TX_TYPES`.
pub const N_TX_TYPE: usize = 16;
/// `INTRA_MODES` + the two directional aliases this census folds back in.
pub const N_MODE: usize = 13;
/// `BLOCK_SIZES_ALL`.
pub const N_BSIZE: usize = 22;

/// `TX_SIZES_ALL` names, index-aligned with the port's `tx_size`.
pub const TX_SIZE_NAME: [&str; N_TX_SIZE] = [
    "4x4", "8x8", "16x16", "32x32", "64x64", "4x8", "8x4", "8x16", "16x8", "16x32", "32x16",
    "32x64", "64x32", "4x16", "16x4", "8x32", "32x8", "16x64", "64x16",
];

/// Width of each `TX_SIZES_ALL` entry.
pub const TX_SIZE_W: [usize; N_TX_SIZE] =
    [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
/// Height of each `TX_SIZES_ALL` entry.
pub const TX_SIZE_H: [usize; N_TX_SIZE] =
    [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

/// `TX_TYPES` names, index-aligned with the port's `tx_type`.
pub const TX_TYPE_NAME: [&str; N_TX_TYPE] = [
    "DCT_DCT", "ADST_DCT", "DCT_ADST", "ADST_ADST", "FLIPADST_DCT", "DCT_FLIPADST",
    "FLIPADST_FLIPADST", "ADST_FLIPADST", "FLIPADST_ADST", "IDTX", "V_DCT", "H_DCT", "V_ADST",
    "H_ADST", "V_FLIPADST", "H_FLIPADST",
];

/// `PREDICTION_MODES` for the intra set, index-aligned with the port's `mode`.
pub const MODE_NAME: [&str; N_MODE] = [
    "DC", "V", "H", "D45", "D135", "D113", "D157", "D203", "D67", "SMOOTH", "SMOOTH_V",
    "SMOOTH_H", "PAETH",
];

/// `BLOCK_SIZES_ALL` names, index-aligned with the port's `bsize`.
pub const BSIZE_NAME: [&str; N_BSIZE] = [
    "4x4", "4x8", "8x4", "8x8", "8x16", "16x8", "16x16", "16x32", "32x16", "32x32", "32x64",
    "64x32", "64x64", "64x128", "128x64", "128x128", "4x16", "16x4", "8x32", "32x8", "16x64",
    "64x16",
];

/// `MODE_TO_ANGLE` for the directional set (0 where the mode is not
/// directional); the same table `intra::predict_intra_high` classifies with.
const MODE_TO_ANGLE: [i32; N_MODE] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];

/// One census. All fields are counts since the last [`reset`] on this thread.
///
/// Boxed by every caller ([`snapshot`] returns it that way): it is ~5 KiB and
/// nothing here is on a path where that matters, but returning it by value
/// through a `#[inline]` no-op shim when the feature is off would be silly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    /// `[class][tx_size]` calls into `predict_intra_high`.
    pub intra_calls: [[u64; N_TX_SIZE]; 7],
    /// `[class][tx_size]` predicted pixels (`w*h` per call).
    pub intra_px: [[u64; N_TX_SIZE]; 7],
    /// Non-directional (plus exactly-`V` / exactly-`H`) calls by AV1 mode.
    pub nd_mode_calls: [u64; N_MODE],
    /// Non-directional (plus exactly-`V` / exactly-`H`) pixels by AV1 mode.
    pub nd_mode_px: [u64; N_MODE],
    /// `[tx_type][tx_size]` forward 2-D transforms.
    pub fwd_tx: [[u64; N_TX_SIZE]; N_TX_TYPE],
    /// Coded leaves by `bsize`, counted at the bitstream writer.
    pub leaf_bsize: [u64; N_BSIZE],
    /// Coded leaves by winning luma intra mode.
    pub leaf_mode: [u64; N_MODE],
}

impl Counts {
    /// All-zero.
    pub fn zero() -> Box<Self> {
        Box::new(Self {
            intra_calls: [[0; N_TX_SIZE]; 7],
            intra_px: [[0; N_TX_SIZE]; 7],
            nd_mode_calls: [0; N_MODE],
            nd_mode_px: [0; N_MODE],
            fwd_tx: [[0; N_TX_SIZE]; N_TX_TYPE],
            leaf_bsize: [0; N_BSIZE],
            leaf_mode: [0; N_MODE],
        })
    }

    /// `self - base`, field by field. Lets a caller discard a warm-up encode's
    /// contribution without a [`reset`] in between.
    pub fn since(&self, base: &Counts) -> Box<Counts> {
        let mut out = Counts::zero();
        for c in 0..7 {
            for t in 0..N_TX_SIZE {
                out.intra_calls[c][t] = self.intra_calls[c][t] - base.intra_calls[c][t];
                out.intra_px[c][t] = self.intra_px[c][t] - base.intra_px[c][t];
            }
        }
        for m in 0..N_MODE {
            out.nd_mode_calls[m] = self.nd_mode_calls[m] - base.nd_mode_calls[m];
            out.nd_mode_px[m] = self.nd_mode_px[m] - base.nd_mode_px[m];
            out.leaf_mode[m] = self.leaf_mode[m] - base.leaf_mode[m];
        }
        for ty in 0..N_TX_TYPE {
            for t in 0..N_TX_SIZE {
                out.fwd_tx[ty][t] = self.fwd_tx[ty][t] - base.fwd_tx[ty][t];
            }
        }
        for b in 0..N_BSIZE {
            out.leaf_bsize[b] = self.leaf_bsize[b] - base.leaf_bsize[b];
        }
        out
    }

    /// Total intra-prediction calls.
    pub fn intra_total_calls(&self) -> u64 {
        self.intra_calls.iter().flatten().sum()
    }

    /// Total predicted pixels.
    pub fn intra_total_px(&self) -> u64 {
        self.intra_px.iter().flatten().sum()
    }

    /// Predicted pixels in classes `z1`/`z2`/`z3` — the share a directional
    /// predictor lever can reach, and the number `detail` gets wrong.
    pub fn directional_px(&self) -> u64 {
        (2..=4).map(|c| self.intra_px[c].iter().sum::<u64>()).sum()
    }

    /// Calls in classes `z1`/`z2`/`z3`.
    pub fn directional_calls(&self) -> u64 {
        (2..=4).map(|c| self.intra_calls[c].iter().sum::<u64>()).sum()
    }

    /// Every counter is zero — i.e. either nothing ran or the crate was built
    /// without the `census` feature. Callers should fail loud on this rather
    /// than print a table of zeros (playbook §2).
    pub fn is_empty(&self) -> bool {
        self.intra_total_calls() == 0
            && self.fwd_tx.iter().flatten().sum::<u64>() == 0
            && self.leaf_bsize.iter().sum::<u64>() == 0
    }
}

/// Whether this build actually counts. `false` means every `note_*` below is a
/// no-op and [`snapshot`] returns zeros.
pub const fn enabled() -> bool {
    cfg!(feature = "census")
}

#[cfg(feature = "census")]
thread_local! {
    static COUNTS: core::cell::RefCell<Box<Counts>> =
        core::cell::RefCell::new(Counts::zero());
}

/// Zero this thread's counters.
pub fn reset() {
    #[cfg(feature = "census")]
    COUNTS.with(|c| *c.borrow_mut() = Counts::zero());
}

/// This thread's counters. All zero when the feature is off.
pub fn snapshot() -> Box<Counts> {
    #[cfg(feature = "census")]
    {
        COUNTS.with(|c| c.borrow().clone())
    }
    #[cfg(not(feature = "census"))]
    {
        Counts::zero()
    }
}

/// One `predict_intra_high` call. `mode` / `angle_delta` / `use_filter_intra` /
/// `tx_size` are that function's own arguments, and the classification here is
/// the same branch it takes.
#[inline(always)]
pub fn note_intra_pred(
    mode: usize,
    angle_delta: i32,
    use_filter_intra: bool,
    tx_size: usize,
) {
    #[cfg(feature = "census")]
    {
        let _ = note_intra_pred_impl(mode, angle_delta, use_filter_intra, tx_size);
    }
    #[cfg(not(feature = "census"))]
    {
        let _ = (mode, angle_delta, use_filter_intra, tx_size);
    }
}

#[cfg(feature = "census")]
fn note_intra_pred_impl(mode: usize, angle_delta: i32, use_filter_intra: bool, tx_size: usize) {
    if tx_size >= N_TX_SIZE || mode >= N_MODE {
        return;
    }
    let is_dr = (1..=8).contains(&mode);
    // `class`, and for the non-directional ones the AV1 mode to also charge.
    // Exactly-90 / exactly-180 get their own class (a directional MODE that a
    // directional-kernel lever does NOT reach: `predict_intra_high` routes them
    // to the V/H fills) AND a row in the mode table, which is why they are
    // counted twice on purpose.
    let (class, nd) = if use_filter_intra {
        (0usize, None)
    } else if !is_dr {
        (1, Some(mode))
    } else {
        let a = MODE_TO_ANGLE[mode] + angle_delta;
        if a > 0 && a < 90 {
            (2, None)
        } else if a > 90 && a < 180 {
            (3, None)
        } else if a > 180 && a < 270 {
            (4, None)
        } else if a == 90 {
            (5, Some(1))
        } else {
            (6, Some(2))
        }
    };
    let px = (TX_SIZE_W[tx_size] * TX_SIZE_H[tx_size]) as u64;
    COUNTS.with(|c| {
        let mut c = c.borrow_mut();
        c.intra_calls[class][tx_size] += 1;
        c.intra_px[class][tx_size] += px;
        if let Some(m) = nd {
            c.nd_mode_calls[m] += 1;
            c.nd_mode_px[m] += px;
        }
    });
}

/// One forward 2-D transform.
#[inline(always)]
pub fn note_fwd_txfm(tx_type: usize, tx_size: usize) {
    #[cfg(feature = "census")]
    {
        if tx_type < N_TX_TYPE && tx_size < N_TX_SIZE {
            COUNTS.with(|c| c.borrow_mut().fwd_tx[tx_type][tx_size] += 1);
        }
    }
    #[cfg(not(feature = "census"))]
    {
        let _ = (tx_type, tx_size);
    }
}

/// One coded leaf, at the bitstream writer: its `bsize` and its winning luma
/// intra mode. This is the partition DECISION, not a search visit.
#[inline(always)]
pub fn note_coded_leaf(bsize: usize, y_mode: usize) {
    #[cfg(feature = "census")]
    {
        COUNTS.with(|c| {
            let mut c = c.borrow_mut();
            if bsize < N_BSIZE {
                c.leaf_bsize[bsize] += 1;
            }
            if y_mode < N_MODE {
                c.leaf_mode[y_mode] += 1;
            }
        });
    }
    #[cfg(not(feature = "census"))]
    {
        let _ = (bsize, y_mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tables are index-aligned with the port's own enums, and the widths
    /// and heights are the ones the pixel counts are computed from. A typo here
    /// would silently mis-weight every `pct_px` column in a census.
    ///
    /// Non-vacuity (playbook §2): the assertions below fail against a table of
    /// the wrong length, a duplicated name, or a w/h pair that disagrees with
    /// the name it is filed under — which is exactly how such a typo appears.
    #[test]
    fn the_tables_agree_with_their_own_names() {
        for t in 0..N_TX_SIZE {
            let want = format!("{}x{}", TX_SIZE_W[t], TX_SIZE_H[t]);
            assert_eq!(TX_SIZE_NAME[t], want, "tx_size {t}");
        }
        for names in [&TX_SIZE_NAME[..], &TX_TYPE_NAME[..], &MODE_NAME[..], &BSIZE_NAME[..]] {
            let mut seen = names.to_vec();
            seen.sort_unstable();
            let n = seen.len();
            seen.dedup();
            assert_eq!(seen.len(), n, "duplicate name in a census table");
        }
        // V and H are the two modes the class table charges twice; the angle
        // table has to place them at exactly 90 and 180 for that to be right.
        assert_eq!(MODE_TO_ANGLE[1], 90);
        assert_eq!(MODE_TO_ANGLE[2], 180);
        assert!(MODE_TO_ANGLE.iter().filter(|&&a| a == 0).count() == 5, "DC/SMOOTH*/PAETH");
    }

    /// `since` really subtracts, and `is_empty` really detects a build with the
    /// feature off. Both are what the census example fails loud on.
    #[test]
    fn since_and_is_empty_are_load_bearing() {
        let base = Counts::zero();
        assert!(base.is_empty());
        let mut later = Counts::zero();
        later.intra_calls[2][1] = 7;
        later.intra_px[2][1] = 7 * 64;
        later.fwd_tx[0][1] = 3;
        later.leaf_bsize[3] = 2;
        assert!(!later.is_empty());
        let d = later.since(&base);
        assert_eq!(d.directional_calls(), 7);
        assert_eq!(d.directional_px(), 7 * 64);
        let same = later.since(&later);
        assert!(same.is_empty(), "x.since(x) must be zero");
    }

    /// With the feature off the hooks must be no-ops and `enabled()` must say
    /// so, because that is what stops a census-built binary from being used for
    /// a timing band by accident.
    #[test]
    fn the_hooks_match_what_enabled_reports() {
        reset();
        note_intra_pred(3, 0, false, 1);
        note_fwd_txfm(0, 1);
        note_coded_leaf(3, 0);
        let s = snapshot();
        assert_eq!(!s.is_empty(), enabled(), "hooks disagree with enabled()");
        if enabled() {
            // z1: D45 (angle 45) at TX_8X8.
            assert_eq!(s.intra_calls[2][1], 1);
            assert_eq!(s.intra_px[2][1], 64);
            assert_eq!(s.fwd_tx[0][1], 1);
            assert_eq!(s.leaf_bsize[3], 1);
            assert_eq!(s.leaf_mode[0], 1);
        }
        reset();
        assert!(snapshot().is_empty(), "reset did not clear");
    }
}
