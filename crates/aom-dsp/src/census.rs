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
//! Axes, chosen because they are the ones a content choice can silently get
//! wrong:
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
//! # 2026-08-03: the tool families the first cut could not name
//!
//! `benchmarks/winperf_content_census_2026-08-03.md` §4 listed what the census
//! itself was blind to, and the list was whole tool families:
//! **filter-intra, palette (luma and UV), intraBC, CFL and the chroma path**.
//! A lever inside any of them would have read as noise on every source in the
//! harness, and a *regression* in any of them would have been invisible. The
//! same document's `detail` row — "issues no 4x4 forward transform and reaches
//! no rectangular leaf" — is the same failure one axis over.
//!
//! Three additions close that:
//!
//! * [`Leaf`] replaces the two-scalar coded-leaf hook. The bitstream writer
//!   already knows a leaf's filter-intra flag, its Y and UV palette sizes, its
//!   `use_intrabc`, its UV mode (so `UV_CFL_PRED` is a count, not an
//!   inference), its signalled `tx_size` and both angle deltas. Counting them
//!   where they are *written* keeps the existing "decision, not search visit"
//!   discipline.
//! * [`note_plane_intra_pred`] tags each `predict_intra_high` call with the
//!   plane its ENCODER call site knows it to be. `predict_intra_high` is a
//!   published `aom_dsp` entry point and gains no argument; the plane split is
//!   an encoder-side annotation next to each call. [`Counts::plane_total`] and
//!   [`Counts::intra_total_calls`] must agree, and the census tool asserts they
//!   do — that is what catches a call site the annotation missed.
//! * [`note_cfl_predict`] counts the CFL predictor itself, which is the one
//!   chroma kernel no `predict_intra_high` count can ever include.
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
/// `UV_INTRA_MODES_CFL_ALLOWED` — the 13 intra modes plus `UV_CFL_PRED` (13).
pub const N_UV_MODE: usize = 14;
/// `FILTER_INTRA_MODES`.
pub const N_FILTER_INTRA_MODE: usize = 5;
/// `PALETTE_MAX_SIZE + 1`, so a palette size indexes the array directly and
/// index 0 means "no palette" (never incremented).
pub const N_PALETTE_SIZE: usize = 9;
/// `2 * MAX_ANGLE_DELTA + 1` (`MAX_ANGLE_DELTA == 3`), indexed
/// `angle_delta + 3`.
pub const N_ANGLE_DELTA: usize = 7;
/// Planes: Y, U, V.
pub const N_PLANE: usize = 3;

/// `UV_CFL_PRED` (enums.h), the value [`Leaf::uv_mode`] carries for a
/// chroma-from-luma leaf.
pub const UV_CFL_PRED: usize = 13;

/// [`Counts::leaf_uv_mode`] names.
pub const UV_MODE_NAME: [&str; N_UV_MODE] = [
    "UV_DC", "UV_V", "UV_H", "UV_D45", "UV_D135", "UV_D113", "UV_D157", "UV_D203", "UV_D67",
    "UV_SMOOTH", "UV_SMOOTH_V", "UV_SMOOTH_H", "UV_PAETH", "UV_CFL",
];

/// `FILTER_INTRA_MODE` names (`filter_intra_mode_kind`, enums.h).
pub const FILTER_INTRA_MODE_NAME: [&str; N_FILTER_INTRA_MODE] =
    ["FI_DC", "FI_V", "FI_H", "FI_D157", "FI_PAETH"];

/// Plane names for [`Counts::plane_calls`].
pub const PLANE_NAME: [&str; N_PLANE] = ["Y", "U", "V"];

/// `true` for the eight non-square entries of `BLOCK_SIZES_ALL` plus the six
/// 4:1 "extended" ones — i.e. every `bsize` whose width and height differ.
/// Derived from the names so it cannot drift from [`BSIZE_NAME`]
/// (`the_tables_agree_with_their_own_names` re-derives it).
pub const BSIZE_IS_RECT: [bool; N_BSIZE] = [
    false, true, true, false, true, true, false, true, true, false, true, true, false, true,
    true, false, true, true, true, true, true, true,
];

/// Width of each `BLOCK_SIZES_ALL` entry, in pixels.
pub const BSIZE_W: [usize; N_BSIZE] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
/// Height of each `BLOCK_SIZES_ALL` entry, in pixels.
pub const BSIZE_H: [usize; N_BSIZE] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

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

    // ---- plane split of the intra predictor (encoder-side annotation) ----
    /// `predict_intra_high` calls, by the plane the ENCODER call site knows it
    /// is predicting. Sums to [`Counts::intra_total_calls`] on an encode; a
    /// mismatch means a call site is not annotated.
    pub plane_calls: [u64; N_PLANE],
    /// Predicted pixels, same split.
    pub plane_px: [u64; N_PLANE],

    // ---- the chroma kernel no intra-predictor count can include ----
    /// `cfl_predict_block` calls, by chroma `tx_size`.
    pub cfl_calls: [u64; N_TX_SIZE],
    /// CFL-predicted chroma pixels.
    pub cfl_px: u64,

    // ---- coded-leaf decisions, all at the bitstream writer ----
    /// Luma pixels (`bsize` w*h) per coded leaf `bsize`.
    pub leaf_px: [u64; N_BSIZE],
    /// Coded leaves by winning UV mode, index 13 = `UV_CFL_PRED`. Counted only
    /// on chroma-reference leaves (the others code no UV mode at all).
    pub leaf_uv_mode: [u64; N_UV_MODE],
    /// Chroma-reference leaves — the denominator [`Counts::leaf_uv_mode`] is a
    /// share of.
    pub leaf_chroma_ref: u64,
    /// Coded leaves by signalled uniform luma `tx_size`.
    pub leaf_tx_size: [u64; N_TX_SIZE],
    /// Filter-intra leaves by `filter_intra_mode`.
    pub leaf_filter_intra: [u64; N_FILTER_INTRA_MODE],
    /// Luma pixels in filter-intra leaves.
    pub leaf_filter_intra_px: u64,
    /// Luma-palette leaves by `palette_size[0]` (2..=8; index 0/1 unused).
    pub leaf_palette_y: [u64; N_PALETTE_SIZE],
    /// Luma pixels in luma-palette leaves.
    pub leaf_palette_y_px: u64,
    /// UV-palette leaves by `palette_size[1]`.
    pub leaf_palette_uv: [u64; N_PALETTE_SIZE],
    /// Luma-plane pixels of the leaves carrying a UV palette (the chroma pixel
    /// count is this scaled by the subsampling, which the census does not
    /// carry — the share is what matters).
    pub leaf_palette_uv_px: u64,
    /// Intra-block-copy leaves.
    pub leaf_intrabc: u64,
    /// Luma pixels in intraBC leaves.
    pub leaf_intrabc_px: u64,
    /// Luma angle delta of directional-mode leaves, indexed `delta + 3`.
    pub leaf_angle_delta_y: [u64; N_ANGLE_DELTA],
    /// UV angle delta of directional-UV-mode leaves, indexed `delta + 3`.
    pub leaf_angle_delta_uv: [u64; N_ANGLE_DELTA],
    /// Leaves whose `skip_txfm` is set (no coded residual).
    pub leaf_skip_txfm: u64,
    /// Leaves coded as INTER (0 on every KEY-frame census).
    pub leaf_inter: u64,
}

/// `a - b`, element by element. Underflows are a bug (a census only ever grows
/// between two snapshots on the same thread) and panic in a debug build.
fn sub<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    core::array::from_fn(|i| a[i] - b[i])
}

/// [`sub`] for the two-dimensional counters.
fn sub2<const N: usize, const M: usize>(a: &[[u64; N]; M], b: &[[u64; N]; M]) -> [[u64; N]; M] {
    core::array::from_fn(|i| sub(&a[i], &b[i]))
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
            plane_calls: [0; N_PLANE],
            plane_px: [0; N_PLANE],
            cfl_calls: [0; N_TX_SIZE],
            cfl_px: 0,
            leaf_px: [0; N_BSIZE],
            leaf_uv_mode: [0; N_UV_MODE],
            leaf_chroma_ref: 0,
            leaf_tx_size: [0; N_TX_SIZE],
            leaf_filter_intra: [0; N_FILTER_INTRA_MODE],
            leaf_filter_intra_px: 0,
            leaf_palette_y: [0; N_PALETTE_SIZE],
            leaf_palette_y_px: 0,
            leaf_palette_uv: [0; N_PALETTE_SIZE],
            leaf_palette_uv_px: 0,
            leaf_intrabc: 0,
            leaf_intrabc_px: 0,
            leaf_angle_delta_y: [0; N_ANGLE_DELTA],
            leaf_angle_delta_uv: [0; N_ANGLE_DELTA],
            leaf_skip_txfm: 0,
            leaf_inter: 0,
        })
    }

    /// `self - base`, field by field. Lets a caller discard a warm-up encode's
    /// contribution without a [`reset`] in between.
    ///
    /// The destructure below has **no `..`** on purpose (playbook §8): adding a
    /// counter to [`Counts`] breaks this build until its author says how it
    /// subtracts. A field silently missing from here reads as a permanent zero
    /// in every census, which is precisely the class of blindness this module
    /// exists to end. `since_subtracts_every_field` is the runtime half of the
    /// same guard.
    pub fn since(&self, base: &Counts) -> Box<Counts> {
        let Counts {
            intra_calls,
            intra_px,
            nd_mode_calls,
            nd_mode_px,
            fwd_tx,
            leaf_bsize,
            leaf_mode,
            plane_calls,
            plane_px,
            cfl_calls,
            cfl_px,
            leaf_px,
            leaf_uv_mode,
            leaf_chroma_ref,
            leaf_tx_size,
            leaf_filter_intra,
            leaf_filter_intra_px,
            leaf_palette_y,
            leaf_palette_y_px,
            leaf_palette_uv,
            leaf_palette_uv_px,
            leaf_intrabc,
            leaf_intrabc_px,
            leaf_angle_delta_y,
            leaf_angle_delta_uv,
            leaf_skip_txfm,
            leaf_inter,
        } = self;
        Box::new(Counts {
            intra_calls: sub2(intra_calls, &base.intra_calls),
            intra_px: sub2(intra_px, &base.intra_px),
            nd_mode_calls: sub(nd_mode_calls, &base.nd_mode_calls),
            nd_mode_px: sub(nd_mode_px, &base.nd_mode_px),
            fwd_tx: sub2(fwd_tx, &base.fwd_tx),
            leaf_bsize: sub(leaf_bsize, &base.leaf_bsize),
            leaf_mode: sub(leaf_mode, &base.leaf_mode),
            plane_calls: sub(plane_calls, &base.plane_calls),
            plane_px: sub(plane_px, &base.plane_px),
            cfl_calls: sub(cfl_calls, &base.cfl_calls),
            cfl_px: cfl_px - base.cfl_px,
            leaf_px: sub(leaf_px, &base.leaf_px),
            leaf_uv_mode: sub(leaf_uv_mode, &base.leaf_uv_mode),
            leaf_chroma_ref: leaf_chroma_ref - base.leaf_chroma_ref,
            leaf_tx_size: sub(leaf_tx_size, &base.leaf_tx_size),
            leaf_filter_intra: sub(leaf_filter_intra, &base.leaf_filter_intra),
            leaf_filter_intra_px: leaf_filter_intra_px - base.leaf_filter_intra_px,
            leaf_palette_y: sub(leaf_palette_y, &base.leaf_palette_y),
            leaf_palette_y_px: leaf_palette_y_px - base.leaf_palette_y_px,
            leaf_palette_uv: sub(leaf_palette_uv, &base.leaf_palette_uv),
            leaf_palette_uv_px: leaf_palette_uv_px - base.leaf_palette_uv_px,
            leaf_intrabc: leaf_intrabc - base.leaf_intrabc,
            leaf_intrabc_px: leaf_intrabc_px - base.leaf_intrabc_px,
            leaf_angle_delta_y: sub(leaf_angle_delta_y, &base.leaf_angle_delta_y),
            leaf_angle_delta_uv: sub(leaf_angle_delta_uv, &base.leaf_angle_delta_uv),
            leaf_skip_txfm: leaf_skip_txfm - base.leaf_skip_txfm,
            leaf_inter: leaf_inter - base.leaf_inter,
        })
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

    /// Total coded leaves.
    pub fn leaves(&self) -> u64 {
        self.leaf_bsize.iter().sum()
    }

    /// Total luma pixels in coded leaves — the frame's luma pixel count when
    /// the frame divides evenly, and the denominator every per-pixel family
    /// share below is taken against.
    pub fn leaf_total_px(&self) -> u64 {
        self.leaf_px.iter().sum()
    }

    /// Coded leaves whose width and height differ.
    pub fn rect_leaves(&self) -> u64 {
        (0..N_BSIZE).filter(|&b| BSIZE_IS_RECT[b]).map(|b| self.leaf_bsize[b]).sum()
    }

    /// Coded leaves at or below 8x8 in EITHER dimension — the "small leaf"
    /// class `smooth` (which never splits below 32x32) reaches zero of.
    pub fn small_leaves(&self) -> u64 {
        (0..N_BSIZE)
            .filter(|&b| BSIZE_W[b] <= 8 || BSIZE_H[b] <= 8)
            .map(|b| self.leaf_bsize[b])
            .sum()
    }

    /// Filter-intra leaves, all modes.
    pub fn filter_intra_leaves(&self) -> u64 {
        self.leaf_filter_intra.iter().sum()
    }

    /// Luma-palette leaves, all palette sizes.
    pub fn palette_y_leaves(&self) -> u64 {
        self.leaf_palette_y.iter().sum()
    }

    /// UV-palette leaves, all palette sizes.
    pub fn palette_uv_leaves(&self) -> u64 {
        self.leaf_palette_uv.iter().sum()
    }

    /// `UV_CFL_PRED` leaves.
    pub fn cfl_leaves(&self) -> u64 {
        self.leaf_uv_mode[UV_CFL_PRED]
    }

    /// Total annotated per-plane `predict_intra_high` calls. Equals
    /// [`Counts::intra_total_calls`] iff every encoder call site is annotated —
    /// asserted by the census tool rather than assumed.
    pub fn plane_total(&self) -> u64 {
        self.plane_calls.iter().sum()
    }

    /// Directional-mode leaves that signalled a NONZERO angle delta — the
    /// `--enable-angle-delta` reach.
    pub fn nonzero_angle_delta_leaves(&self) -> u64 {
        self.leaf_angle_delta_y.iter().sum::<u64>() - self.leaf_angle_delta_y[3]
    }

    /// Forward transforms at 4x4 (either dimension 4) — the class `detail`
    /// issues none of.
    pub fn small_fwd_tx(&self) -> u64 {
        (0..N_TX_SIZE)
            .filter(|&t| TX_SIZE_W[t] <= 4 || TX_SIZE_H[t] <= 4)
            .map(|t| (0..N_TX_TYPE).map(|ty| self.fwd_tx[ty][t]).sum::<u64>())
            .sum()
    }

    /// Forward transforms whose width and height differ.
    pub fn rect_fwd_tx(&self) -> u64 {
        (0..N_TX_SIZE)
            .filter(|&t| TX_SIZE_W[t] != TX_SIZE_H[t])
            .map(|t| (0..N_TX_TYPE).map(|ty| self.fwd_tx[ty][t]).sum::<u64>())
            .sum()
    }

    /// Forward transforms that are NOT `DCT_DCT` — what a tx-type lever reaches.
    pub fn non_dct_fwd_tx(&self) -> u64 {
        (1..N_TX_TYPE).map(|ty| self.fwd_tx[ty].iter().sum::<u64>()).sum()
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

/// One `predict_intra_high` call, tagged with the plane its ENCODER call site
/// knows it is predicting (0 = Y, 1 = U, 2 = V).
///
/// `predict_intra_high` itself has no `plane` argument and does not gain one —
/// it is a published `aom_dsp` entry point and the decoder calls it too. So
/// this is a separate, additive annotation placed next to each encoder call,
/// and [`Counts::plane_total`] agreeing with [`Counts::intra_total_calls`] is
/// what proves none was missed.
#[inline(always)]
pub fn note_plane_intra_pred(plane: usize, tx_size: usize) {
    #[cfg(feature = "census")]
    {
        if plane < N_PLANE && tx_size < N_TX_SIZE {
            let px = (TX_SIZE_W[tx_size] * TX_SIZE_H[tx_size]) as u64;
            COUNTS.with(|c| {
                let mut c = c.borrow_mut();
                c.plane_calls[plane] += 1;
                c.plane_px[plane] += px;
            });
        }
    }
    #[cfg(not(feature = "census"))]
    {
        let _ = (plane, tx_size);
    }
}

/// One `cfl_predict_block` — the chroma-from-luma predictor, which no
/// `predict_intra_high` count can ever include because CFL does not route
/// through it.
#[inline(always)]
pub fn note_cfl_predict(tx_size: usize) {
    #[cfg(feature = "census")]
    {
        if tx_size < N_TX_SIZE {
            let px = (TX_SIZE_W[tx_size] * TX_SIZE_H[tx_size]) as u64;
            COUNTS.with(|c| {
                let mut c = c.borrow_mut();
                c.cfl_calls[tx_size] += 1;
                c.cfl_px += px;
            });
        }
    }
    #[cfg(not(feature = "census"))]
    {
        let _ = tx_size;
    }
}

/// Everything the bitstream writer knows about one coded leaf, which is every
/// tool family a content choice can silently fail to reach.
///
/// Filled at the writer, so each field is the **decision** the frame carries,
/// not a search visit — `rd_pick_partition_real` visits many shapes per leaf
/// and a search-visit count would say nothing about what the content made the
/// encoder emit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Leaf {
    /// `mbmi->bsize`.
    pub bsize: usize,
    /// `mbmi->mode` (the luma intra mode; `DC_PRED` on a palette or intraBC
    /// leaf, as AV1 codes it).
    pub y_mode: usize,
    /// `mbmi->angle_delta[PLANE_TYPE_Y]`, -3..=3.
    pub angle_delta_y: i32,
    /// `mbmi->uv_mode`, 13 = [`UV_CFL_PRED`]. Read only when `chroma_ref`.
    pub uv_mode: usize,
    /// `mbmi->angle_delta[PLANE_TYPE_UV]`, -3..=3.
    pub angle_delta_uv: i32,
    /// `mbmi->tx_size` — the signalled uniform luma transform size.
    pub tx_size: usize,
    /// `mbmi->filter_intra_mode_info.use_filter_intra`.
    pub use_filter_intra: bool,
    /// `mbmi->filter_intra_mode_info.filter_intra_mode`, 0..=4.
    pub filter_intra_mode: usize,
    /// `mbmi->palette_mode_info.palette_size[0]` (0 = none, else 2..=8).
    pub palette_y_size: usize,
    /// `mbmi->palette_mode_info.palette_size[1]`.
    pub palette_uv_size: usize,
    /// `mbmi->use_intrabc`.
    pub use_intrabc: bool,
    /// `mbmi->skip_txfm`.
    pub skip_txfm: bool,
    /// `is_inter_block(mbmi)`.
    pub is_inter: bool,
    /// `is_chroma_reference(..)` — false leaves code no UV mode at all, so
    /// counting them in the UV distribution would understate every share.
    pub chroma_ref: bool,
}

/// One coded leaf. See [`Leaf`].
#[inline(always)]
pub fn note_coded_leaf(leaf: &Leaf) {
    #[cfg(feature = "census")]
    {
        note_coded_leaf_impl(leaf);
    }
    #[cfg(not(feature = "census"))]
    {
        let _ = leaf;
    }
}

#[cfg(feature = "census")]
fn note_coded_leaf_impl(leaf: &Leaf) {
    let &Leaf {
        bsize,
        y_mode,
        angle_delta_y,
        uv_mode,
        angle_delta_uv,
        tx_size,
        use_filter_intra,
        filter_intra_mode,
        palette_y_size,
        palette_uv_size,
        use_intrabc,
        skip_txfm,
        is_inter,
        chroma_ref,
    } = leaf;
    let px = if bsize < N_BSIZE { (BSIZE_W[bsize] * BSIZE_H[bsize]) as u64 } else { 0 };
    // `angle_delta` is only signalled for a directional mode; folding a
    // non-directional leaf's (always 0) delta into the histogram would drown
    // the axis in DC/SMOOTH/PAETH blocks and make the "nonzero delta" share
    // meaningless.
    let dir_y = (1..=8).contains(&y_mode) && !use_filter_intra && palette_y_size == 0
        && !use_intrabc;
    let dir_uv = (1..=8).contains(&uv_mode) && palette_uv_size == 0;
    COUNTS.with(|c| {
        let mut c = c.borrow_mut();
        if bsize < N_BSIZE {
            c.leaf_bsize[bsize] += 1;
            c.leaf_px[bsize] += px;
        }
        if y_mode < N_MODE {
            c.leaf_mode[y_mode] += 1;
        }
        if tx_size < N_TX_SIZE {
            c.leaf_tx_size[tx_size] += 1;
        }
        if use_filter_intra && filter_intra_mode < N_FILTER_INTRA_MODE {
            c.leaf_filter_intra[filter_intra_mode] += 1;
            c.leaf_filter_intra_px += px;
        }
        if palette_y_size > 0 && palette_y_size < N_PALETTE_SIZE {
            c.leaf_palette_y[palette_y_size] += 1;
            c.leaf_palette_y_px += px;
        }
        if palette_uv_size > 0 && palette_uv_size < N_PALETTE_SIZE {
            c.leaf_palette_uv[palette_uv_size] += 1;
            c.leaf_palette_uv_px += px;
        }
        if use_intrabc {
            c.leaf_intrabc += 1;
            c.leaf_intrabc_px += px;
        }
        if skip_txfm {
            c.leaf_skip_txfm += 1;
        }
        if is_inter {
            c.leaf_inter += 1;
        }
        if chroma_ref {
            c.leaf_chroma_ref += 1;
            if uv_mode < N_UV_MODE {
                c.leaf_uv_mode[uv_mode] += 1;
            }
            if dir_uv && (-3..=3).contains(&angle_delta_uv) {
                c.leaf_angle_delta_uv[(angle_delta_uv + 3) as usize] += 1;
            }
        }
        if dir_y && (-3..=3).contains(&angle_delta_y) {
            c.leaf_angle_delta_y[(angle_delta_y + 3) as usize] += 1;
        }
    });
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
        // The block-size tables carry the same obligation as the tx ones: a
        // typo mis-weights every per-pixel leaf share, and `BSIZE_IS_RECT` is
        // re-derived here rather than trusted.
        for b in 0..N_BSIZE {
            let want = format!("{}x{}", BSIZE_W[b], BSIZE_H[b]);
            assert_eq!(BSIZE_NAME[b], want, "bsize {b}");
            assert_eq!(BSIZE_IS_RECT[b], BSIZE_W[b] != BSIZE_H[b], "BSIZE_IS_RECT[{b}]");
        }
        assert_eq!(UV_MODE_NAME.len(), N_UV_MODE);
        assert_eq!(UV_MODE_NAME[UV_CFL_PRED], "UV_CFL");
        // The 13 UV modes before CFL are the 13 luma modes in the same order —
        // that is what lets a reader compare `leaf_mode` against `leaf_uv_mode`.
        for m in 0..N_MODE {
            assert!(UV_MODE_NAME[m].starts_with("UV_"), "uv mode {m}");
        }
    }

    /// [`Counts::since`] must subtract EVERY field. The no-`..` destructure in
    /// `since` is the compile-time half of this guard; this is the runtime
    /// half, and it is written so it cannot pass vacuously: `fill` sets every
    /// field to a distinct nonzero value through its own no-`..` destructure,
    /// so a field added to `Counts` and forgotten in `since` shows up as
    /// `since(zero) != self`.
    #[test]
    fn since_subtracts_every_field() {
        let mut c = Counts::zero();
        // Distinct nonzero values everywhere. `n` increments so no two fields
        // share a value and a mis-wired `since` cannot cancel out.
        let mut n = 1u64;
        let mut next = move || {
            n += 1;
            n
        };
        for i in 0..7 {
            for t in 0..N_TX_SIZE {
                c.intra_calls[i][t] = next();
                c.intra_px[i][t] = next();
            }
        }
        for m in 0..N_MODE {
            c.nd_mode_calls[m] = next();
            c.nd_mode_px[m] = next();
            c.leaf_mode[m] = next();
        }
        for ty in 0..N_TX_TYPE {
            for t in 0..N_TX_SIZE {
                c.fwd_tx[ty][t] = next();
            }
        }
        for b in 0..N_BSIZE {
            c.leaf_bsize[b] = next();
            c.leaf_px[b] = next();
        }
        for p in 0..N_PLANE {
            c.plane_calls[p] = next();
            c.plane_px[p] = next();
        }
        for t in 0..N_TX_SIZE {
            c.cfl_calls[t] = next();
            c.leaf_tx_size[t] = next();
        }
        for m in 0..N_UV_MODE {
            c.leaf_uv_mode[m] = next();
        }
        for m in 0..N_FILTER_INTRA_MODE {
            c.leaf_filter_intra[m] = next();
        }
        for s in 0..N_PALETTE_SIZE {
            c.leaf_palette_y[s] = next();
            c.leaf_palette_uv[s] = next();
        }
        for d in 0..N_ANGLE_DELTA {
            c.leaf_angle_delta_y[d] = next();
            c.leaf_angle_delta_uv[d] = next();
        }
        c.cfl_px = next();
        c.leaf_chroma_ref = next();
        c.leaf_filter_intra_px = next();
        c.leaf_palette_y_px = next();
        c.leaf_palette_uv_px = next();
        c.leaf_intrabc = next();
        c.leaf_intrabc_px = next();
        c.leaf_skip_txfm = next();
        c.leaf_inter = next();

        // Every field is nonzero, so a field `since` forgot would come back 0.
        let zero = Counts::zero();
        assert_eq!(&*c.since(&zero), &*c, "since(zero) must be the identity");
        assert_eq!(&*c.since(&c), &*zero, "since(self) must be all-zero");
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
        note_plane_intra_pred(0, 1);
        note_fwd_txfm(0, 1);
        note_cfl_predict(1);
        note_coded_leaf(&Leaf {
            bsize: 3,
            y_mode: 0,
            tx_size: 1,
            uv_mode: UV_CFL_PRED,
            chroma_ref: true,
            ..Default::default()
        });
        let s = snapshot();
        assert_eq!(!s.is_empty(), enabled(), "hooks disagree with enabled()");
        if enabled() {
            // z1: D45 (angle 45) at TX_8X8.
            assert_eq!(s.intra_calls[2][1], 1);
            assert_eq!(s.intra_px[2][1], 64);
            assert_eq!(s.plane_calls[0], 1);
            assert_eq!(s.plane_total(), s.intra_total_calls());
            assert_eq!(s.fwd_tx[0][1], 1);
            assert_eq!(s.cfl_calls[1], 1);
            assert_eq!(s.cfl_px, 64);
            assert_eq!(s.leaf_bsize[3], 1);
            assert_eq!(s.leaf_px[3], 64);
            assert_eq!(s.leaf_mode[0], 1);
            assert_eq!(s.leaf_tx_size[1], 1);
            assert_eq!(s.cfl_leaves(), 1);
            assert_eq!(s.leaf_chroma_ref, 1);
        }
        reset();
        assert!(snapshot().is_empty(), "reset did not clear");
    }

    /// Each family hook fires on the leaf that carries it and on no other, and
    /// the derived roll-ups (`rect_leaves`, `small_leaves`, the angle-delta
    /// histogram's directional gate) mean what they say. Written as one leaf
    /// per family so a mis-indexed counter cannot hide behind a sum.
    #[test]
    fn each_family_counter_fires_only_on_its_own_family() {
        if !enabled() {
            // With the feature off every hook is a no-op by construction and
            // `the_hooks_match_what_enabled_reports` is the assertion that
            // says so. Asserting family shares here would be vacuous.
            return;
        }
        reset();
        // A palette leaf: BLOCK_16X8 (rect, small in one dimension), palette
        // size 4, DC_PRED as AV1 codes it.
        note_coded_leaf(&Leaf {
            bsize: 5,
            y_mode: 0,
            tx_size: 8,
            palette_y_size: 4,
            chroma_ref: true,
            uv_mode: 0,
            ..Default::default()
        });
        // A filter-intra leaf, mode FI_PAETH, at BLOCK_8X8.
        note_coded_leaf(&Leaf {
            bsize: 3,
            y_mode: 0,
            tx_size: 1,
            use_filter_intra: true,
            filter_intra_mode: 4,
            chroma_ref: true,
            uv_mode: 12,
            ..Default::default()
        });
        // An intraBC leaf at BLOCK_32X32, skip_txfm set.
        note_coded_leaf(&Leaf {
            bsize: 9,
            y_mode: 0,
            tx_size: 3,
            use_intrabc: true,
            skip_txfm: true,
            chroma_ref: true,
            uv_mode: 0,
            ..Default::default()
        });
        // A directional leaf with a nonzero angle delta, and a UV palette.
        note_coded_leaf(&Leaf {
            bsize: 6,
            y_mode: 3,
            angle_delta_y: -2,
            tx_size: 2,
            chroma_ref: true,
            uv_mode: 0,
            palette_uv_size: 2,
            ..Default::default()
        });
        // A non-chroma-reference leaf: it codes no UV mode, so it must not
        // land in the UV distribution at all.
        note_coded_leaf(&Leaf {
            bsize: 0,
            y_mode: 9,
            tx_size: 0,
            chroma_ref: false,
            uv_mode: UV_CFL_PRED,
            ..Default::default()
        });
        let s = snapshot();

        assert_eq!(s.leaves(), 5);
        assert_eq!(s.palette_y_leaves(), 1, "palette-Y fired on exactly one leaf");
        assert_eq!(s.leaf_palette_y[4], 1, "and at the right palette size");
        assert_eq!(s.leaf_palette_y_px, 16 * 8);
        assert_eq!(s.palette_uv_leaves(), 1);
        assert_eq!(s.leaf_palette_uv[2], 1);
        assert_eq!(s.filter_intra_leaves(), 1);
        assert_eq!(s.leaf_filter_intra[4], 1);
        assert_eq!(s.leaf_filter_intra_px, 64);
        assert_eq!(s.leaf_intrabc, 1);
        assert_eq!(s.leaf_intrabc_px, 32 * 32);
        assert_eq!(s.leaf_skip_txfm, 1);
        assert_eq!(s.leaf_inter, 0);
        // 16x8 and 4x4 and 8x8 and 32x32 and 16x16 -> one rect, two small.
        assert_eq!(s.rect_leaves(), 1);
        assert_eq!(s.small_leaves(), 3, "16x8, 8x8, 4x4");
        // The angle-delta histogram counts only the directional leaf, and the
        // intraBC / palette / filter-intra leaves (all `y_mode == DC_PRED` or
        // gated out) contribute nothing.
        assert_eq!(s.leaf_angle_delta_y.iter().sum::<u64>(), 1);
        assert_eq!(s.leaf_angle_delta_y[1], 1, "delta -2 -> index 1");
        assert_eq!(s.nonzero_angle_delta_leaves(), 1);
        // Four chroma references, and the fifth leaf's UV_CFL is NOT counted.
        assert_eq!(s.leaf_chroma_ref, 4);
        assert_eq!(s.cfl_leaves(), 0, "a non-chroma-ref leaf codes no UV mode");
        assert_eq!(s.leaf_uv_mode.iter().sum::<u64>(), 4);
        reset();
    }
}
