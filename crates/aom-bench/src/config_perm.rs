//! CONFIG-PERMUTATION COLLAPSE ENGINE — the combinatorial half of the
//! encoder-config gate (`tests/config_permutations.rs`).
//!
//! ## Why this exists
//!
//! `tests/toggles_rd_close.rs` gates ~25 CLI-toggle knob families **one at a
//! time**, each on its own 3-cell grid. Every one of them is byte-identical to
//! real aomenc *alone*. What no gate covered before this module is
//! **combinations**: a knob that is correct in isolation can still break when
//! composed (this repo already found two such defects — the C11 cdf-update
//! pack bug and the `--disable-trellis-quant=2` FINAL_PASS
//! `dry_run_output_enabled` bug, both only visible in a specific
//! configuration).
//!
//! The raw cross of the knob space is 14,155,776 points ([`raw_space_size`])
//! — unreachable. This module makes it tractable by **collapsing** instead of
//! exploding, along the two axes the design brief names:
//!
//! 1. **Effective-config collapse** ([`Effective::resolve`]): many distinct knob
//!    sets resolve to the *same* internal encoder state, in a way that depends
//!    on the cell context (frame size, superblock size, monochrome, …). One
//!    representative per [`Effective`] signature is enough — and the claim is
//!    *falsifiable*, not asserted: `config_permutations.rs` re-runs the dropped
//!    duplicates and requires them byte-identical to their representative on
//!    BOTH sides (port and the C oracle).
//! 2. **Independence collapse** ([`INDEPENDENT_PAIRS`]): axis pairs whose effects
//!    are confined to disjoint parts of the decision state do not need to be
//!    crossed. This is *measured*, once, offline, by the 2x2 footprint
//!    experiment in `config_permutations.rs::independence_evidence_sweep`
//!    (`--ignored`), and the measured result is baked in here with its
//!    evidence. See `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md`.
//!
//! What survives both collapses is covered by a **t-wise covering array**
//! ([`covering_array`]) over the interacting axes, with C-forbidden
//! combinations excluded by [`illegal_reason`] (each exclusion cites the libaom
//! source line that forbids it).
//!
//! ## Scope
//!
//! The proven ALLINTRA / speed-0 / KEY-frame / single-tile envelope
//! (`EncodeCell::port_encode_with`): `base_qindex > 0` (lossless is a separate
//! gate), `tiles_log2 == 0`, SB64. Every axis here is an
//! [`crate::ToggleKnobs`] field, i.e. a knob BOTH sides can be driven with —
//! the C side via `aom_codec_control`, the port side through
//! `port_encode_with`. Cell-context axes (bit depth, chroma format,
//! monochrome, frame size, qindex) are not knobs; they are the *contexts*
//! ([`CellCtx`]) the array is replayed under, and they participate in the
//! collapse because [`Effective::resolve`] is context-dependent.

use std::collections::{BTreeMap, HashSet};

use crate::ToggleKnobs;

// ---------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------

/// One configuration axis: a [`ToggleKnobs`] field the covering array varies.
///
/// Level 0 of every axis is the aomenc DEFAULT (verified in
/// `av1_cx_iface.c::default_extra_cfg`; HANDOFF-TOGGLES.md item 5), so the
/// all-zero row reproduces the stock byte-exact envelope exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Axis {
    /// `--enable-rect-partitions` {1, 0}.
    Rect,
    /// `--enable-ab-partitions` {1, 0}.
    Ab,
    /// `--enable-1to4-partitions` {1, 0}.
    P1to4,
    /// `--min-partition-size` px {4, 8, 16}.
    MinPart,
    /// `--max-partition-size` px {128, 64, 32}.
    MaxPart,
    /// `--enable-smooth-intra` {1, 0}.
    Smooth,
    /// `--enable-paeth-intra` {1, 0}.
    Paeth,
    /// `--enable-cfl-intra` {1, 0}.
    Cfl,
    /// `--enable-directional-intra` {1, 0}.
    Directional,
    /// `--enable-diagonal-intra` {1, 0}.
    Diagonal,
    /// `--enable-angle-delta` {1, 0}.
    AngleDelta,
    /// `--enable-filter-intra` {1, 0} (SEQUENCE-header bit).
    FilterIntra,
    /// `--enable-intra-edge-filter` {1, 0} (SEQUENCE-header bit).
    EdgeFilter,
    /// `--enable-tx64` {1, 0}.
    Tx64,
    /// `--enable-rect-tx` {1, 0}.
    RectTx,
    /// `--enable-flip-idtx` {1, 0}.
    FlipIdtx,
    /// `--use-intra-default-tx-only` {0, 1}.
    DefaultTxOnly,
    /// `--reduced-tx-type-set` {0, 1} (FRAME-header bit).
    ReducedTxSet,
    /// `--enable-tx-size-search` {1, 0}.
    TxSizeSearch,
    /// `--cdf-update-mode` {1, 0, 2}.
    CdfUpdate,
    /// `--disable-trellis-quant` {3, 1, 2, 0}.
    Trellis,
}

/// Every axis the covering array varies, in a fixed order (row index order).
pub const ALL_AXES: [Axis; 21] = [
    Axis::Rect,
    Axis::Ab,
    Axis::P1to4,
    Axis::MinPart,
    Axis::MaxPart,
    Axis::Smooth,
    Axis::Paeth,
    Axis::Cfl,
    Axis::Directional,
    Axis::Diagonal,
    Axis::AngleDelta,
    Axis::FilterIntra,
    Axis::EdgeFilter,
    Axis::Tx64,
    Axis::RectTx,
    Axis::FlipIdtx,
    Axis::DefaultTxOnly,
    Axis::ReducedTxSet,
    Axis::TxSizeSearch,
    Axis::CdfUpdate,
    Axis::Trellis,
];

/// How the port ACQUIRES an axis's configuration — the distinction between
/// "the port computes this" and "the port is told this".
///
/// **The port never authors a sequence header.** `write_sequence_header_obu`
/// (aom-dsp/src/entropy/header.rs:1046) has zero call sites in any encoder
/// path — the eight `crates/*/src` references are its own definition, doc
/// comments, and the C-oracle FFI shim; the only real callers are tests. Every
/// encode parses a sequence header out of a real aomenc bootstrap stream and
/// emits an `OBU_FRAME` payload alone (`aom_encode::obu_assemble`). Verified
/// independently 2026-07-30; cross-checked against
/// `docs/CONFIG_AXIS_INVENTORY_2026-07-30.md`.
///
/// So a covering-array cell on a bootstrap-carried axis proves *"the port
/// behaves correctly GIVEN this header bit"* — a real and useful claim, but a
/// strictly weaker one than *"the port produces this configuration"*. The two
/// must never be blurred into one coverage count, which is why every axis
/// carries this tag and the gate reports the split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AxisKind {
    /// Nothing about this knob is signalled or read back: it steers only the
    /// port's own search and pack decisions, which the port derives itself.
    /// A cell here IS end-to-end evidence about the encoder's configuration
    /// handling.
    Derived,
    /// The knob also corresponds to a SEQUENCE-header bit. The port cannot
    /// author a sequence header at all; `EncodeCell::port_encode_with` asserts
    /// the bootstrap's bit equals the knob (aom-bench/src/lib.rs:1093-1102).
    /// That assert is an AGREEMENT CHECK against libaom's bits, not evidence
    /// the port can produce them.
    BootstrapSeq,
    /// The knob also corresponds to a FRAME-header bit the port parses out of
    /// the bootstrap frame header and asserts equal to the knob. The port DOES
    /// derive the downstream search/pack behaviour, but not the coded bit.
    BootstrapFrame,
}

/// Number of axes ([`ALL_AXES`] length) — the row width.
pub const N_AXES: usize = ALL_AXES.len();

/// One point of the configuration space: one level index per [`ALL_AXES`] entry.
pub type Row = [u8; N_AXES];

/// The all-default row (every axis at level 0 == the aomenc default).
pub const DEFAULT_ROW: Row = [0u8; N_AXES];

impl Axis {
    /// The concrete knob values this axis takes, level 0 first (= the default).
    pub fn values(self) -> &'static [i32] {
        match self {
            Axis::Rect
            | Axis::Ab
            | Axis::P1to4
            | Axis::Smooth
            | Axis::Paeth
            | Axis::Cfl
            | Axis::Directional
            | Axis::Diagonal
            | Axis::AngleDelta
            | Axis::FilterIntra
            | Axis::EdgeFilter
            | Axis::Tx64
            | Axis::RectTx
            | Axis::FlipIdtx
            | Axis::TxSizeSearch => &[1, 0],
            Axis::DefaultTxOnly | Axis::ReducedTxSet => &[0, 1],
            Axis::MinPart => &[4, 8, 16],
            Axis::MaxPart => &[128, 64, 32],
            Axis::CdfUpdate => &[1, 0, 2],
            Axis::Trellis => &[3, 1, 2, 0],
        }
    }

    /// Level count.
    pub fn n_levels(self) -> usize {
        self.values().len()
    }

    /// Short stable tag used in cell labels and the report tables.
    pub fn tag(self) -> &'static str {
        match self {
            Axis::Rect => "rect",
            Axis::Ab => "ab",
            Axis::P1to4 => "p14",
            Axis::MinPart => "minp",
            Axis::MaxPart => "maxp",
            Axis::Smooth => "smth",
            Axis::Paeth => "paeth",
            Axis::Cfl => "cfl",
            Axis::Directional => "dir",
            Axis::Diagonal => "diag",
            Axis::AngleDelta => "adlt",
            Axis::FilterIntra => "fint",
            Axis::EdgeFilter => "edgf",
            Axis::Tx64 => "tx64",
            Axis::RectTx => "rtx",
            Axis::FlipIdtx => "flip",
            Axis::DefaultTxOnly => "dtxo",
            Axis::ReducedTxSet => "rtxs",
            Axis::TxSizeSearch => "txss",
            Axis::CdfUpdate => "cdf",
            Axis::Trellis => "trel",
        }
    }

    /// How the port acquires this axis — see [`AxisKind`].
    pub fn kind(self) -> AxisKind {
        match self {
            // Pure encoder search gates: never signalled, fully port-derived.
            Axis::Rect
            | Axis::Ab
            | Axis::P1to4
            | Axis::MinPart
            | Axis::MaxPart
            | Axis::Smooth
            | Axis::Paeth
            | Axis::Cfl
            | Axis::Directional
            | Axis::Diagonal
            | Axis::AngleDelta
            | Axis::Tx64
            | Axis::RectTx
            | Axis::FlipIdtx
            | Axis::DefaultTxOnly
            | Axis::Trellis => AxisKind::Derived,
            // Sequence-header bits (encoder.c:646-647, bitstream.c:2669-2670).
            Axis::FilterIntra | Axis::EdgeFilter => AxisKind::BootstrapSeq,
            // Frame-header bits: `reduced_tx_set_used` (encodeframe.c:2712),
            // `tx_mode` SELECT/LARGEST, `disable_cdf_update` (encoder.c:4375).
            Axis::ReducedTxSet | Axis::TxSizeSearch | Axis::CdfUpdate => {
                AxisKind::BootstrapFrame
            }
        }
    }

    /// Write this axis's level into a knob set.
    pub fn apply(self, level: u8, k: &mut ToggleKnobs) {
        let v = self.values()[level as usize];
        let b = v != 0;
        match self {
            Axis::Rect => k.enable_rect_partitions = b,
            Axis::Ab => k.enable_ab_partitions = b,
            Axis::P1to4 => k.enable_1to4_partitions = b,
            Axis::MinPart => k.min_partition_size_px = v as usize,
            Axis::MaxPart => k.max_partition_size_px = v as usize,
            Axis::Smooth => k.enable_smooth_intra = b,
            Axis::Paeth => k.enable_paeth_intra = b,
            Axis::Cfl => k.enable_cfl_intra = b,
            Axis::Directional => k.enable_directional_intra = b,
            Axis::Diagonal => k.enable_diagonal_intra = b,
            Axis::AngleDelta => k.enable_angle_delta = b,
            Axis::FilterIntra => k.enable_filter_intra = b,
            Axis::EdgeFilter => k.enable_intra_edge_filter = b,
            Axis::Tx64 => k.enable_tx64 = b,
            Axis::RectTx => k.enable_rect_tx = b,
            Axis::FlipIdtx => k.enable_flip_idtx = b,
            Axis::DefaultTxOnly => k.use_intra_default_tx_only = b,
            Axis::ReducedTxSet => k.reduced_tx_type_set = b,
            Axis::TxSizeSearch => k.enable_tx_size_search = b,
            Axis::CdfUpdate => k.cdf_update_mode = v as u32,
            Axis::Trellis => k.disable_trellis_quant = v as u32,
        }
    }

    /// This axis's level in `k` (inverse of [`Axis::apply`]).
    pub fn level_of(self, k: &ToggleKnobs) -> u8 {
        let cur: i32 = match self {
            Axis::Rect => k.enable_rect_partitions as i32,
            Axis::Ab => k.enable_ab_partitions as i32,
            Axis::P1to4 => k.enable_1to4_partitions as i32,
            Axis::MinPart => k.min_partition_size_px as i32,
            Axis::MaxPart => k.max_partition_size_px as i32,
            Axis::Smooth => k.enable_smooth_intra as i32,
            Axis::Paeth => k.enable_paeth_intra as i32,
            Axis::Cfl => k.enable_cfl_intra as i32,
            Axis::Directional => k.enable_directional_intra as i32,
            Axis::Diagonal => k.enable_diagonal_intra as i32,
            Axis::AngleDelta => k.enable_angle_delta as i32,
            Axis::FilterIntra => k.enable_filter_intra as i32,
            Axis::EdgeFilter => k.enable_intra_edge_filter as i32,
            Axis::Tx64 => k.enable_tx64 as i32,
            Axis::RectTx => k.enable_rect_tx as i32,
            Axis::FlipIdtx => k.enable_flip_idtx as i32,
            Axis::DefaultTxOnly => k.use_intra_default_tx_only as i32,
            Axis::ReducedTxSet => k.reduced_tx_type_set as i32,
            Axis::TxSizeSearch => k.enable_tx_size_search as i32,
            Axis::CdfUpdate => k.cdf_update_mode as i32,
            Axis::Trellis => k.disable_trellis_quant as i32,
        };
        self.values()
            .iter()
            .position(|&v| v == cur)
            .expect("knob value is one of the axis levels") as u8
    }
}

/// The knob set for one covering-array row.
pub fn knobs_of(row: &Row) -> ToggleKnobs {
    let mut k = ToggleKnobs::default();
    for (i, ax) in ALL_AXES.iter().enumerate() {
        ax.apply(row[i], &mut k);
    }
    k
}

/// Human-readable row label: the tags of every NON-default level.
/// The all-default row is labelled `stock`.
pub fn row_label(row: &Row) -> String {
    let mut parts = Vec::new();
    for (i, ax) in ALL_AXES.iter().enumerate() {
        if row[i] != 0 {
            parts.push(format!("{}{}", ax.tag(), ax.values()[row[i] as usize]));
        }
    }
    if parts.is_empty() {
        "stock".to_string()
    } else {
        parts.join("-")
    }
}

/// Size of the raw cartesian product of [`ALL_AXES`] (before ANY collapse).
pub fn raw_space_size() -> u64 {
    ALL_AXES.iter().map(|a| a.n_levels() as u64).product()
}

// ---------------------------------------------------------------------------
// C-forbidden combinations
// ---------------------------------------------------------------------------

/// Why this row must never be handed to the C encoder, or `None` if it is legal.
///
/// Each exclusion cites the libaom source line that forbids the combination.
/// An excluded row is dropped from the covering array *and* from the tuple set
/// (a t-tuple only reachable through an illegal row is unreachable, period).
pub fn illegal_reason(row: &Row) -> Option<&'static str> {
    let k = knobs_of(row);
    // libaom av1/encoder/encodeframe.c:2461
    //   assert(oxcf->txfm_cfg.enable_tx64 || tx_search_type != USE_LARGESTALL);
    // `--enable-tx-size-search=0` forces tx_size_search_level = 3 = USE_LARGESTALL
    // (speed_features.c:2726), so combining it with `--enable-tx64=0` aborts a
    // debug-built libaom and is undefined in release. Also recorded in
    // HANDOFF-TOGGLES.md ("Gotchas").
    if !k.enable_tx_size_search && !k.enable_tx64 {
        return Some(
            "--enable-tx-size-search=0 + --enable-tx64=0: libaom \
             encodeframe.c:2461 assert(enable_tx64 || tx_search_type != USE_LARGESTALL)",
        );
    }
    None
}

// ---------------------------------------------------------------------------
// Cell context
// ---------------------------------------------------------------------------

/// The parts of an [`crate::EncodeCell`] + bootstrap that make a knob live or
/// dead. [`Effective::resolve`] is a function of (knobs, context) — the same
/// knob set can collapse differently in different contexts, which is exactly
/// the "effective/computed unique code paths" the collapse is built on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellCtx {
    /// Frame width in pixels.
    pub w: usize,
    /// Frame height in pixels.
    pub h: usize,
    /// Monochrome (no chroma planes ⇒ the whole UV mode loop is dead).
    pub mono: bool,
    /// Superblock size in pixels (64 on this envelope; 128 under `--sb-size=128`).
    pub sb_px: usize,
}

impl CellCtx {
    /// `dim_to_size` (partition_strategy.h:201): pixel dimension → square
    /// `BLOCK_SIZE` ordinal.
    fn dim_to_bsize(px: usize) -> u8 {
        match px {
            4 => 0,
            8 => 3,
            16 => 6,
            32 => 9,
            64 => 12,
            128 => 15,
            _ => panic!("{px}px is not a square BLOCK dimension"),
        }
    }

    /// `sf.default_max_partition_size` at speed-0 ALLINTRA is BLOCK_128X128;
    /// `set_max_min_partition_size` (partition_strategy.h:214) then takes
    /// `min(sf_default, dim_to_size(CLI px), sb_size)`, so the live cap at SB64
    /// is BLOCK_64X64. Matching `ToggleKnobs::max_partition_bsize`, which the
    /// harness feeds to `PickFrameCfg::max_partition_size`.
    fn sb_bsize(self) -> u8 {
        Self::dim_to_bsize(self.sb_px)
    }

    /// Can a full `sb_px`-square block exist in this frame? `partition_none` at
    /// the superblock root needs `av1_blk_has_rows_and_cols`
    /// (partition_search.c:3389); a frame smaller than one SB in either
    /// dimension forces the root to split, so no SB-sized leaf is ever coded.
    fn has_full_sb_block(self) -> bool {
        self.w >= self.sb_px && self.h >= self.sb_px
    }

    /// `is_480p_or_larger` (speed_features.c:169): `AOMMIN(cm->width,
    /// cm->height) >= 480`. Port mirror: `use_square_partition_only_threshold_
    /// allintra` (`aom-encode/src/partition_pick.rs:2447`) and the harness's
    /// `prune_tx_type_using_stats` wiring (`aom-bench/src/lib.rs:1364`).
    pub fn is_480p_or_larger(self) -> bool {
        self.w.min(self.h) >= 480
    }

    /// `is_720p_or_larger` (speed_features.c:170).
    pub fn is_720p_or_larger(self) -> bool {
        self.w.min(self.h) >= 720
    }

    /// `is_4k_or_larger` (speed_features.c:172) — `>= 2160`.
    pub fn is_4k_or_larger(self) -> bool {
        self.w.min(self.h) >= 2160
    }

    /// Superblock grid extent: `(cols, rows)` of `sb_px`-square superblocks
    /// the frame walk visits (CEIL — `port_encode_full`'s `n_sb_x`/`n_sb_y`,
    /// `aom-bench/src/lib.rs:1172-1173`).
    pub fn sb_grid(self) -> (usize, usize) {
        (
            self.w.div_ceil(self.sb_px).max(1),
            self.h.div_ceil(self.sb_px).max(1),
        )
    }

    /// Does the frame have a PARTIAL superblock — a superblock the frame edge
    /// cuts through? Partial SBs are a distinct code path, not a smaller one:
    /// `av1_blk_has_rows_and_cols` forces partitions (partition_search.c:3389),
    /// `set_partition_cost_for_edge_blk` gathers its partition costs from the
    /// FRAME-INIT cdf rather than the adapting tile state (:3415), the
    /// persistent entropy stamp zeroes the beyond-visible tail
    /// (`av1_set_entropy_contexts`, blockd.c:29) and the distortion is clipped
    /// to the visible area. All four are KB-6 roots.
    pub fn has_partial_sb(self) -> bool {
        self.w % self.sb_px != 0 || self.h % self.sb_px != 0
    }
}

// ---------------------------------------------------------------------------
// Size-derived encoder state (the SIZE axis of the matrix)
// ---------------------------------------------------------------------------

/// The encoder state that a [`CellCtx`]'s **frame geometry** determines, at one
/// speed — i.e. everything `set_allintra_speed_feature_framesize_dependent`
/// (speed_features.c:166-340) plus the frame-edge geometry contributes.
///
/// This is the size analogue of [`Effective`]: two frame sizes that resolve to
/// the same `SizeDerived` cannot make the encoder behave differently *because
/// of their size*, so replaying the covering array at both is redundant. The
/// count of DISTINCT values over a candidate size list is exactly how many size
/// contexts the array needs — see `size_class_partition`.
///
/// Only fields that are LIVE on the all-intra KEY path are carried. The
/// framesize-dependent fields that C sets but this path never reads are listed
/// in the module docs rather than modelled, each with the citation that kills
/// them (they would otherwise inflate the class count with distinctions no
/// cell could ever witness).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SizeDerived {
    /// Superblock size in pixels. The partition-tree ROOT size is itself
    /// size-derived state: it selects the partition-symbol CDF group at the
    /// root, whether a >64 coding block can exist at all, and (for >64 blocks)
    /// the `av1_write_intra_coeffs_mb` 64x64-chunk L/U/V coefficient interleave
    /// (KB-1's encoder cross-check).
    pub sb_px: usize,
    /// Is the square-only rect-kill (`bsize > threshold` ⇒
    /// `partition_rect_allowed[HORZ] &= !has_rows`, partition_search.c:5700;
    /// port `partition_pick.rs:2593`) REACHABLE in this context? It needs a
    /// coding block strictly larger than the threshold, and the largest block
    /// the walk can offer is the superblock — so at SB64 with a `BLOCK_64X64`
    /// threshold it is structurally dead, which is why speed-0 SB64 never
    /// needed it (KB-3).
    pub rect_kill_reachable: bool,
    /// `part_sf.default_min_partition_size` (`BLOCK_SIZE` ordinal) —
    /// `BLOCK_8X8` at `is_4k_or_larger` (speed_features.c:187-189) and at
    /// speed>=6 `is_1080p_or_larger` (:311-313), else `BLOCK_4X4`
    /// (`init_part_sf`, :2285). Read by `set_max_min_partition_size`
    /// (partition_strategy.h:225) on EVERY frame type, so it is live on
    /// all-intra KEY. **The port does not model either arm** — see
    /// `PORT_GAP_DEFAULT_MIN_PARTITION_SIZE`.
    pub default_min_partition_size: u8,
    /// `tx_sf.tx_type_search.prune_tx_type_using_stats` — 1 at speed>=2 and 2
    /// at speed>=4, both only when `is_480p_or_larger` (speed_features.c:261,
    /// :299). Port: `aom-bench/src/lib.rs:1364`.
    pub prune_tx_type_using_stats: u8,
    /// More than one superblock COLUMN. One-vs-many is the structural
    /// distinction (a left-neighbour SB exists, the tile CDF has adapted before
    /// the second SB, the partition context carries across); a third column
    /// adds no new structure, so the grid extent is deliberately NOT carried as
    /// an exact count — that would over-refine every size into its own class.
    pub multi_sb_cols: bool,
    /// More than one superblock ROW (an above-neighbour SB exists; the
    /// above-context array is reset at a row start).
    pub multi_sb_rows: bool,
    /// The frame edge cuts superblocks HORIZONTALLY (`w % sb_px != 0`) ⇒ the
    /// KB-6 edge paths are live on the right column: forced partitions
    /// (`av1_blk_has_rows_and_cols`, partition_search.c:3389), the frame-init
    /// partition-cost gather (`set_partition_cost_for_edge_blk`, :3415), the
    /// beyond-visible entropy tail-zero (`av1_set_entropy_contexts`,
    /// blockd.c:29) and the visible-distortion clip.
    ///
    /// The overhang MAGNITUDE (4 px vs 32 px) is a continuous sub-axis inside
    /// the class — it selects which transform footprints the tail-zero clips —
    /// and is sampled, not classified.
    pub partial_sb_x: bool,
    /// The frame edge cuts superblocks VERTICALLY (`h % sb_px != 0`). Separate
    /// from `partial_sb_x` because the above-context and left-context clipping
    /// are separate code paths.
    pub partial_sb_y: bool,
    /// A full superblock-sized coding block can exist (so a 64-point transform
    /// is reachable, and the rect-kill has something to bite).
    pub full_sb_block: bool,
}

/// `part_sf.use_square_partition_only_threshold` (`BLOCK_SIZE` ordinal) for the
/// ALLINTRA path — speed_features.c:176/181 (base), :211-217 (speed 1),
/// :238-242 (speed 2), :315 (speed 6). Port mirror:
/// `use_square_partition_only_threshold_allintra`,
/// `aom-encode/src/partition_pick.rs:2446`.
///
/// Deliberately NOT a [`SizeDerived`] field: at speed 0 on an intra frame its
/// only consumer is the rect-kill (partition_search.c:5700), and its other
/// reader — the ML breakout gate at :4265 — is inside a `!frame_is_intra_only`
/// block. So two sizes with different thresholds but the same
/// [`SizeDerived::rect_kill_reachable`] are INDISTINGUISHABLE to this encoder
/// path, and carrying the raw value would over-refine the class count (it would
/// split 480x480 SB64 away from 128x128 SB64 for a difference no cell could
/// witness).
pub fn sq_only_threshold_allintra(ctx: &CellCtx, speed: i32) -> u8 {
    let min_dim = ctx.w.min(ctx.h);
    let (is_480, is_720) = (min_dim >= 480, min_dim >= 720);
    let mut thr: u8 = if is_480 { 15 } else { 12 };
    if speed >= 1 {
        thr = if is_720 {
            15
        } else if is_480 {
            12
        } else {
            9
        };
    }
    if speed >= 2 {
        thr = if is_720 { 12 } else { 9 };
    }
    if speed >= 6 {
        thr = 6;
    }
    thr
}

/// PORT GAP, recorded here so the size class table stays honest: the port's
/// [`crate::speed_features`] mirror pins `default_min_partition_size` to
/// `BLOCK_4X4` at every speed below 6 (`aom-encode/src/speed_features.rs:471`)
/// and to `BLOCK_8X8` unconditionally at speed>=6 (`:891`), modelling only the
/// framesize-INDEPENDENT setter (speed_features.c:570). Two framesize-DEPENDENT
/// arms are unmodelled:
///
/// * speed-0.. `is_4k_or_larger` ⇒ `BLOCK_8X8` (speed_features.c:187-189);
/// * speed>=6 `is_1080p_or_larger` ⇒ `BLOCK_8X8` (:311-313) — subsumed by the
///   port's unconditional speed-6 assignment, so only the 4K arm is a real
///   divergence, and only below speed 6.
///
/// `min(w,h) >= 2160` is out of this harness's budget by three orders of
/// magnitude (a 480x480 speed-0 cell already costs ~12.6 s), so the gap is
/// DOCUMENTED, not gated. Reaching it needs a tiered deep gate, not a default
/// cell.
pub const PORT_GAP_DEFAULT_MIN_PARTITION_SIZE: &str =
    "default_min_partition_size = BLOCK_8X8 at is_4k_or_larger \
     (speed_features.c:187-189) is unmodelled by aom-encode/src/speed_features.rs:471";

/// Resolve the size-derived encoder state for one context at one speed.
///
/// The all-intra KEY path reads only the fields carried by [`SizeDerived`].
/// The framesize-dependent fields deliberately NOT modelled, with the line that
/// makes each inert here:
///
/// | field | set at | why inert on all-intra KEY |
/// |---|---|---|
/// | `auto_max_partition_based_on_simple_motion` | :176-180, :305-309 | `use_auto_max_partition` is `!frame_is_intra_only && ... && sb_size == BLOCK_128X128` (partition_strategy.h:193) |
/// | `ml_partition_search_breakout_thresh[]`, `ml_partition_search_breakout_model_index` | :192-201, :219-236 | `av1_ml_predict_breakout` runs under `!frame_is_intra_only` (partition_search.c:4260) |
/// | `ml_early_term_after_part_split_level` | :200, :207, :269 | `av1_ml_early_term_after_split` runs under `!frame_is_intra_only` (partition_search.c:4322) |
/// | `mv_sf.use_downsampled_sad` | :203-206 | motion search only (mcomp.c:131) |
/// | `tx_sf.prune_tx_size_level` | :184, :263-265, :289 | read only by `select_tx_block` (tx_search.c:2631), the INTER var-tx recursion reached from `av1_pick_recursive_tx_size_type_yrd` |
/// | `partition_search_breakout_{dist,rate}_thr` | :244-251, :273-286, :293-297 | the breakout block is `!frame_is_intra_only` (partition_search.c:4260) |
/// | `part_sf.max_intra_bsize` | :283 | speed>=3 only, and sub-720p only — no framesize DISTINCTION below 720p |
/// | `rt_sf.*` | :323-336 | speed>=8 real-time path |
pub fn size_derived(ctx: &CellCtx, speed: i32) -> SizeDerived {
    let min_dim = ctx.w.min(ctx.h);
    let is_480 = min_dim >= 480;
    let thr = sq_only_threshold_allintra(ctx, speed);
    let sb_b = CellCtx::dim_to_bsize(ctx.sb_px);
    // The rect-kill needs a block strictly larger than the threshold; the
    // largest block the partition walk can offer is the superblock, and it can
    // only offer it when a full SB fits in the frame.
    let rect_kill_reachable = sb_b > thr && ctx.has_full_sb_block();
    let default_min_partition_size = if ctx.is_4k_or_larger() || (speed >= 6 && min_dim >= 1080) {
        3 // BLOCK_8X8
    } else {
        0 // BLOCK_4X4
    };
    let prune_tx_type_using_stats = if is_480 {
        if speed >= 4 {
            2
        } else if speed >= 2 {
            1
        } else {
            0
        }
    } else {
        0
    };
    let (sb_cols, sb_rows) = ctx.sb_grid();
    SizeDerived {
        sb_px: ctx.sb_px,
        rect_kill_reachable,
        default_min_partition_size,
        prune_tx_type_using_stats,
        multi_sb_cols: sb_cols > 1,
        multi_sb_rows: sb_rows > 1,
        partial_sb_x: ctx.w % ctx.sb_px != 0,
        partial_sb_y: ctx.h % ctx.sb_px != 0,
        full_sb_block: ctx.has_full_sb_block(),
    }
}

/// Collapse a candidate size list into its distinct size classes at one speed.
///
/// Returns, for each distinct [`SizeDerived`], the contexts that produce it,
/// cheapest (smallest pixel count) first. THIS is the answer to "how many size
/// contexts does the array need": one per returned class, not one per size.
pub fn size_class_partition(sizes: &[CellCtx], speed: i32) -> Vec<(SizeDerived, Vec<CellCtx>)> {
    let mut by_class: BTreeMap<SizeDerived, Vec<CellCtx>> = BTreeMap::new();
    for &c in sizes {
        by_class.entry(size_derived(&c, speed)).or_default().push(c);
    }
    for v in by_class.values_mut() {
        v.sort_by_key(|c| (c.w * c.h, c.w, c.h, c.sb_px));
    }
    by_class.into_iter().collect()
}

/// The axes whose interaction with the size-derived state is not structurally
/// zero — i.e. the ones a reduced-strength array at an expensive size context
/// must still cross.
///
/// The ONLY size-derived state that is live at speed 0 below 2160p is
/// [`SizeDerived::sq_only_threshold`] via [`SizeDerived::rect_kill_reachable`]
/// (everything else in the table above is either inert on intra or
/// speed-gated). The kill acts on `partition_rect_allowed[HORZ|VERT]` at a
/// block strictly larger than the threshold, so an axis can interact with it
/// only by changing (a) whether rectangular partitions exist at all, (b)
/// whether the over-threshold block is reached, or (c) which rect-derived
/// partition types are offered:
///
/// * [`Axis::Rect`] — `--enable-rect-partitions=0` clears
///   `partition_rect_allowed` outright (partition_search.c:3383), so the kill
///   is a no-op;
/// * [`Axis::MaxPart`] — `--max-partition-size` below the superblock forces the
///   root to SPLIT (`av1_set_square_split_only`), so the over-threshold block
///   is never evaluated;
/// * [`Axis::Ab`] / [`Axis::P1to4`] — AB and HORZ_4/VERT_4 are gated on
///   `partition_rect_allowed` (:5166, :5172, :5181, :5187), so the kill
///   propagates into their availability too.
///
/// [`Axis::MinPart`] is excluded: it raises the *floor*, never the root.
pub const RECT_KILL_INTERACTION_SET: &[Axis] = &[Axis::Rect, Axis::MaxPart, Axis::Ab, Axis::P1to4];

// ---------------------------------------------------------------------------
// Effective configuration (mechanism 1)
// ---------------------------------------------------------------------------

// Luma PREDICTION_MODE bit positions (av1/common/enums.h order).
const M_DC: u16 = 1 << 0;
const M_V: u16 = 1 << 1;
const M_H: u16 = 1 << 2;
const M_D45: u16 = 1 << 3;
const M_D135: u16 = 1 << 4;
const M_D113: u16 = 1 << 5;
const M_D157: u16 = 1 << 6;
const M_D203: u16 = 1 << 7;
const M_D67: u16 = 1 << 8;
const M_SMOOTH: u16 = 1 << 9;
const M_SMOOTH_V: u16 = 1 << 10;
const M_SMOOTH_H: u16 = 1 << 11;
const M_PAETH: u16 = 1 << 12;
/// UV_CFL_PRED — a chroma-only extra mode, not a luma one.
const M_CFL: u16 = 1 << 13;

/// `av1_is_diagonal_mode` (reconintra.h:55): `D45_PRED..=D67_PRED`.
const DIAGONAL_MASK: u16 = M_D45 | M_D135 | M_D113 | M_D157 | M_D203 | M_D67;
/// `av1_is_directional_mode`: `V_PRED..=D67_PRED`.
const DIRECTIONAL_MASK: u16 = M_V | M_H | DIAGONAL_MASK;
const SMOOTH_MASK: u16 = M_SMOOTH | M_SMOOTH_V | M_SMOOTH_H;
const ALL_LUMA_MODES: u16 = M_DC | DIRECTIONAL_MASK | SMOOTH_MASK | M_PAETH;

/// Which tx types the luma intra search may consider, after the three CLI
/// overrides collapse into one policy (`get_tx_mask`, tx_search.c; the
/// MODE_EVAL `use_default_intra_tx_type` OR-arm, rdopt_utils.h:579).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TxTypePolicy {
    /// The full ext-tx set for the block (shaped by flip/idtx + reduced set).
    Full,
    /// `--use-intra-default-tx-only=1`: only the mode's default type.
    DefaultOnly,
    /// `--use-intra-dct-only=1`: only DCT_DCT.
    DctOnly,
}

/// The RESOLVED encoder state a knob row produces in a given [`CellCtx`] —
/// the collapse key. Two rows with the same [`Effective`] must produce
/// byte-identical output; `config_permutations.rs::effective_collapse_is_real`
/// proves that against BOTH the port and the C oracle instead of assuming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Effective {
    // --- partition search ---
    /// HORZ/VERT arms live.
    pub part_rect: bool,
    /// HORZ_A/B + VERT_A/B live.
    pub part_ab: bool,
    /// HORZ_4/VERT_4 live.
    pub part_4way: bool,
    /// `min_partition_size` as a BLOCK_SIZE ordinal.
    pub min_part_bsize: u8,
    /// `max_partition_size` as a BLOCK_SIZE ordinal.
    pub max_part_bsize: u8,
    // --- intra prediction ---
    /// Luma modes the search may pick.
    pub luma_modes: u16,
    /// Chroma modes the search may pick (0 when monochrome).
    pub uv_modes: u16,
    /// Nonzero angle deltas reachable.
    pub angle_delta: bool,
    /// Filter-intra candidates + the per-block flag in the syntax.
    pub filter_intra: bool,
    /// Directional-prediction edge filter/upsample stage active.
    pub edge_filter: bool,
    // --- transform ---
    /// Tx-size RD search (vs USE_LARGESTALL).
    pub tx_size_search: bool,
    /// 64-point transform sizes reachable.
    pub tx64: bool,
    /// Rectangular transform sizes reachable.
    pub rect_tx: bool,
    /// Luma intra tx-TYPE policy.
    pub txtype_policy: TxTypePolicy,
    /// FLIPADST/IDTX family present in the ext-tx sets.
    pub flip_idtx: bool,
    /// `reduced_tx_set_used` FRAME-header bit (search set AND signalling).
    pub reduced_tx_set: bool,
    // --- entropy / quantization ---
    /// Symbol adaptation during the pack (`disable_cdf_update` inverted).
    pub allow_update_cdf: bool,
    /// `is_trellis_used(opt, DRY_RUN_NORMAL)` — trellis inside the RD search.
    pub search_trellis: bool,
    /// `is_trellis_used(opt, OUTPUT_ENABLED)` — trellis in the final pack.
    pub pack_trellis: bool,
}

impl Effective {
    /// Resolve a knob row to the encoder state it actually produces in `ctx`.
    ///
    /// Every canonicalisation below cites the libaom line that makes the knob
    /// dead. Nothing here is a hardcoded "known inert" list: the documented
    /// inert cases (HANDOFF-TOGGLES.md) fall out of the resolution, which is
    /// how the engine validates itself.
    pub fn resolve(row: &Row, ctx: &CellCtx) -> Effective {
        let k = knobs_of(row);

        // --- partitions ------------------------------------------------------
        // `do_rectangular_split` (partition_search.c:3383) gates
        // `partition_rect_allowed[HORZ|VERT]` (:3389). AB needs
        // `partition_rect_allowed` (:5166/:5172) and HORZ_4/VERT_4 need
        // `partition_rect_allowed[HORZ]` (:5181/:5187) — so with rect off, the
        // AB and 1to4 knobs are BOTH structurally dead.
        let part_rect = k.enable_rect_partitions;
        let part_ab = part_rect && k.enable_ab_partitions;
        let part_4way = part_rect && k.enable_1to4_partitions;
        // set_max_min_partition_size (partition_strategy.h:214). At SB64 the
        // 128px and 64px CLI levels BOTH clamp to BLOCK_64X64 — a real,
        // computed collapse of two distinct CLI values.
        let sb_b = ctx.sb_bsize();
        let max_part_bsize = CellCtx::dim_to_bsize(k.max_partition_size_px)
            .min(sb_b)
            .min(CellCtx::dim_to_bsize(128));
        let min_part_bsize = CellCtx::dim_to_bsize(k.min_partition_size_px).min(sb_b);

        // --- intra mode sets -------------------------------------------------
        // Luma: intra_mode_search.c:1555-1578. Chroma: :922-936 (+ CFL at :949).
        let mut luma_modes = ALL_LUMA_MODES;
        if !k.enable_directional_intra {
            luma_modes &= !DIRECTIONAL_MASK;
        }
        if !k.enable_diagonal_intra {
            luma_modes &= !DIAGONAL_MASK;
        }
        if !k.enable_smooth_intra {
            luma_modes &= !SMOOTH_MASK;
        }
        if !k.enable_paeth_intra {
            luma_modes &= !M_PAETH;
        }
        let uv_modes = if ctx.mono {
            // No chroma planes ⇒ the entire UV loop (and with it the CFL knob)
            // is unreachable.
            0
        } else {
            let mut m = luma_modes;
            if k.enable_cfl_intra {
                m |= M_CFL;
            }
            m
        };
        let any_directional = luma_modes & DIRECTIONAL_MASK != 0;
        // `enable_angle_delta` only gates NONZERO deltas on directional modes
        // (intra_mode_search.c:1317, :1585) — dead when no directional mode
        // survives.
        let angle_delta = k.enable_angle_delta && any_directional;
        // The intra edge filter/upsample runs ONLY for directional modes:
        // `build_directional_and_filter_intra_predictors` returns early for
        // filter-intra (reconintra.c:1198) and then asserts `is_dr_mode`
        // before the `if (!disable_edge_filter)` block (:1204-1207). With no
        // directional mode live the seq bit cannot change a single pixel of
        // the FRAME payload (its own seq-header bit is outside the compared
        // unit — `port_encode_with` returns the frame OBU payload and
        // `splice_frame_obu` keeps C's sequence header).
        let edge_filter = if any_directional {
            k.enable_intra_edge_filter
        } else {
            true
        };

        // --- transform -------------------------------------------------------
        let tx_size_search = k.enable_tx_size_search;
        // A 64-point transform needs a block with a 64px dimension, which needs
        // a BLOCK_64X64 root that is not force-split: both a max-partition cap
        // below 64 and a frame smaller than one SB kill it.
        let can_reach_64 =
            max_part_bsize >= CellCtx::dim_to_bsize(64) && ctx.has_full_sb_block();
        let tx64 = if can_reach_64 { k.enable_tx64 } else { true };
        let txtype_policy = if k.use_intra_dct_only {
            TxTypePolicy::DctOnly
        } else if k.use_intra_default_tx_only {
            TxTypePolicy::DefaultOnly
        } else {
            TxTypePolicy::Full
        };
        // `--enable-flip-idtx` masks FLIPADST/IDTX out of the ext-tx SET
        // (`get_tx_mask`'s DCT_ADST_TX_MASK arm) — irrelevant once the policy
        // has already narrowed the search to one type.
        let flip_idtx = if txtype_policy == TxTypePolicy::Full {
            k.enable_flip_idtx
        } else {
            true
        };

        // --- entropy / quantization ------------------------------------------
        // encoder.c:4375-4395: mode 0 ⇒ disable_cdf_update = 1; mode 1 ⇒ 0
        // (rt-only sub-arms are off in ALLINTRA); mode 2 ⇒
        // `frame_is_intra_only ? 0 : 1` ⇒ 0 on a lone KEY frame. So 1 and 2
        // resolve identically here. (`should_force_mode_cost_update`, rd.c:762,
        // is the only other reader and is gated on `rt_sf`.)
        let allow_update_cdf = k.cdf_update_mode != 0;
        // init_rd_sf (speed_features.c:2479-2498) + `is_trellis_used`:
        //   0 FULL              → search yes, pack yes
        //   1 NO                → search no,  pack no
        //   2 FINAL_PASS        → search no,  pack yes
        //   3 NO_ESTIMATE_YRD   → search yes, pack yes   (the default)
        // 0 and 3 differ ONLY in `estimate_yrd_for_sb`, which is inter-only —
        // so on this KEY envelope they collapse. HANDOFF-TOGGLES.md records
        // exactly this as verified-INERT; the engine re-derives it.
        let (search_trellis, pack_trellis) = match k.disable_trellis_quant {
            0 | 3 => (true, true),
            1 => (false, false),
            2 => (false, true),
            v => panic!("--disable-trellis-quant {v} out of range"),
        };

        Effective {
            part_rect,
            part_ab,
            part_4way,
            min_part_bsize,
            max_part_bsize,
            luma_modes,
            uv_modes,
            angle_delta,
            filter_intra: k.enable_filter_intra,
            edge_filter,
            tx_size_search,
            tx64,
            rect_tx: k.enable_rect_tx,
            txtype_policy,
            flip_idtx,
            reduced_tx_set: k.reduced_tx_type_set,
            allow_update_cdf,
            search_trellis,
            pack_trellis,
        }
    }
}

// ---------------------------------------------------------------------------
// Independence (mechanism 2)
// ---------------------------------------------------------------------------

/// Axis pairs MEASURED to be independent on this envelope, i.e. the spatial
/// footprint of axis A's effect on the reconstruction is the SAME set of 4x4
/// blocks under both settings of axis B (and vice versa).
///
/// Populated from `config_permutations.rs::independence_evidence_sweep`
/// (`--ignored`); the evidence table lives in
/// `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md` and the raw run in
/// `benchmarks/config_perm_independence_2026-07-30.tsv`. A pair listed here is
/// dropped from the covering array's tuple set — every OTHER pair is crossed.
///
/// The measured result on the reference context (see the design doc) is that
/// non-trivial independence is RARE: every axis pair where both axes are live
/// interacts through the shared RD/partition loop. The pairs that do qualify
/// are the ones where one axis is structurally dead under the other (which the
/// [`Effective`] collapse already removes), so this list is deliberately
/// conservative — an empty or near-empty list is the honest outcome, not a
/// failure of the method.
pub const INDEPENDENT_PAIRS: &[(Axis, Axis)] = &[];

/// Is this axis pair excused from being crossed by the covering array?
pub fn pair_is_independent(a: Axis, b: Axis) -> bool {
    INDEPENDENT_PAIRS
        .iter()
        .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
}

// ---------------------------------------------------------------------------
// t-wise covering array
// ---------------------------------------------------------------------------

/// One t-tuple: `t` (axis index, level) pairs, axis indices ascending.
type Tuple = Vec<(usize, u8)>;

/// Deterministic xorshift64* — the array must be byte-reproducible across runs
/// and machines (the gate pins its size).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Every t-tuple that a LEGAL row can realise, minus tuples whose two axes are
/// a measured-independent pair (t == 2 only; for t > 2 a tuple is dropped when
/// EVERY pair inside it is independent, which cannot happen with an empty
/// [`INDEPENDENT_PAIRS`]).
fn required_tuples(t: usize) -> Vec<Tuple> {
    assert!((2..=4).contains(&t), "strength must be 2..=4");
    let mut out = Vec::new();
    let mut idx = vec![0usize; t];
    // Iterate ascending axis-index combinations.
    loop {
        let ok = idx.windows(2).all(|w| w[0] < w[1]) && idx[t - 1] < N_AXES;
        if ok {
            // Drop a pair the measurement excused.
            let excused = t == 2 && pair_is_independent(ALL_AXES[idx[0]], ALL_AXES[idx[1]]);
            if !excused {
                let mut lv = vec![0u8; t];
                'levels: loop {
                    if tuple_is_reachable(&idx, &lv) {
                        out.push(idx.iter().copied().zip(lv.iter().copied()).collect());
                    }
                    // increment level odometer
                    let mut i = t;
                    loop {
                        if i == 0 {
                            break 'levels;
                        }
                        i -= 1;
                        lv[i] += 1;
                        if (lv[i] as usize) < ALL_AXES[idx[i]].n_levels() {
                            break;
                        }
                        lv[i] = 0;
                    }
                }
            }
        }
        // increment the axis-index odometer
        let mut i = t;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            idx[i] += 1;
            if idx[i] < N_AXES {
                // reset the tail ascending
                for j in i + 1..t {
                    idx[j] = idx[j - 1] + 1;
                }
                if idx[t - 1] < N_AXES {
                    break;
                }
            }
        }
    }
}

/// Can SOME legal row realise this (axes, levels) tuple?
fn tuple_is_reachable(idx: &[usize], lv: &[u8]) -> bool {
    let mut row = DEFAULT_ROW;
    for (&a, &l) in idx.iter().zip(lv.iter()) {
        row[a] = l;
    }
    if illegal_reason(&row).is_none() {
        return true;
    }
    // The only exclusion couples TxSizeSearch and Tx64; try flipping whichever
    // of the two is NOT pinned by the tuple.
    for probe in [Axis::TxSizeSearch, Axis::Tx64] {
        let ai = ALL_AXES.iter().position(|&a| a == probe).unwrap();
        if idx.contains(&ai) {
            continue;
        }
        let mut r2 = row;
        for l in 0..probe.n_levels() as u8 {
            r2[ai] = l;
            if illegal_reason(&r2).is_none() {
                return true;
            }
        }
    }
    false
}

/// Index of the two axes coupled by the single C-forbidden combination.
fn coupled_axis_indices() -> (usize, usize) {
    (
        ALL_AXES
            .iter()
            .position(|&a| a == Axis::TxSizeSearch)
            .unwrap(),
        ALL_AXES.iter().position(|&a| a == Axis::Tx64).unwrap(),
    )
}

/// A t-wise covering array over [`ALL_AXES`], containing only rows that are
/// legal for the C encoder.
///
/// AETG-style randomised construction with a FIXED seed ⇒ byte-reproducible
/// across runs and machines (the gate pins the row count, so a silent shrink
/// is a test failure). Each candidate row is seeded from a still-uncovered
/// tuple — guaranteeing forward progress — and the remaining axes are filled
/// randomly; the best of `CANDIDATES` candidates is kept. A final sweep drops
/// rows that ended up covering nothing unique.
///
/// Row 0 is always [`DEFAULT_ROW`]: the stock byte-exact envelope, i.e. the
/// harness-faithfulness control.
pub fn covering_array(t: usize) -> Vec<Row> {
    const CANDIDATES: usize = 64;
    let mut uncovered: HashSet<Tuple> = required_tuples(t).into_iter().collect();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut rows: Vec<Row> = Vec::new();
    let (txss_i, tx64_i) = coupled_axis_indices();

    let cover = |row: &Row, set: &mut HashSet<Tuple>| {
        set.retain(|tp| !tp.iter().all(|&(a, l)| row[a] == l));
    };
    let gain_of = |row: &Row, set: &HashSet<Tuple>| {
        set.iter()
            .filter(|tp| tp.iter().all(|&(a, l)| row[a] == l))
            .count()
    };

    rows.push(DEFAULT_ROW);
    cover(&DEFAULT_ROW, &mut uncovered);

    while !uncovered.is_empty() {
        // Deterministic seed pool: the uncovered set is a HashSet, so sort a
        // snapshot before indexing into it.
        let mut pool: Vec<&Tuple> = uncovered.iter().collect();
        pool.sort();
        let mut best: Option<(usize, Row)> = None;
        for _ in 0..CANDIDATES {
            let seed = pool[rng.below(pool.len())];
            let mut row = DEFAULT_ROW;
            let mut pinned = [false; N_AXES];
            for &(a, l) in seed.iter() {
                row[a] = l;
                pinned[a] = true;
            }
            for a in 0..N_AXES {
                if pinned[a] {
                    continue;
                }
                let mut l = rng.below(ALL_AXES[a].n_levels()) as u8;
                // The single constraint: `--enable-tx-size-search=0` may not
                // meet `--enable-tx64=0` (see `illegal_reason`). Steer the axis
                // being filled — never a PINNED one, which would silently drop
                // the seed tuple. tx64 (index 13) is filled before
                // tx_size_search (index 18), so both directions are covered.
                if a == txss_i && l == 1 && row[tx64_i] == 1 {
                    l = 0;
                }
                if a == tx64_i && l == 1 && pinned[txss_i] && row[txss_i] == 1 {
                    l = 0;
                }
                row[a] = l;
            }
            assert!(
                illegal_reason(&row).is_none(),
                "candidate construction produced an illegal row"
            );
            let gain = gain_of(&row, &uncovered);
            if best.as_ref().is_none_or(|(g, _)| gain > *g) {
                best = Some((gain, row));
            }
        }
        let (gain, row) = best.expect("at least one candidate");
        assert!(
            gain > 0,
            "covering-array construction stalled with {} tuples uncovered — \
             the constraint model and the tuple-reachability filter disagree",
            uncovered.len()
        );
        cover(&row, &mut uncovered);
        rows.push(row);
    }

    // Drop rows every one of whose tuples is also covered by another row (the
    // AETG random fill leaves a few). Row 0 (the stock control) is never
    // dropped. One pass over the rows, newest first, maintaining a per-tuple
    // cover count — O(rows x tuples).
    let all = required_tuples(t);
    let mut count: BTreeMap<&Tuple, usize> = BTreeMap::new();
    for tp in &all {
        let n = rows.iter().filter(|r| tp.iter().all(|&(a, l)| r[a] == l)).count();
        count.insert(tp, n);
    }
    let mut drop = vec![false; rows.len()];
    for i in (1..rows.len()).rev() {
        let covered: Vec<&Tuple> = all
            .iter()
            .filter(|tp| tp.iter().all(|&(a, l)| rows[i][a] == l))
            .collect();
        if covered.iter().all(|tp| count[*tp] >= 2) {
            drop[i] = true;
            for tp in covered {
                *count.get_mut(tp).unwrap() -= 1;
            }
        }
    }
    rows.iter()
        .enumerate()
        .filter(|(i, _)| !drop[*i])
        .map(|(_, r)| *r)
        .collect()
}

// ---------------------------------------------------------------------------
// Collapse bookkeeping
// ---------------------------------------------------------------------------

/// The result of applying the effective-config collapse to a covering array in
/// one [`CellCtx`].
#[derive(Clone, Debug)]
pub struct Collapsed {
    /// One representative row per distinct [`Effective`], in first-seen order.
    pub representatives: Vec<Row>,
    /// `(duplicate, representative)` pairs the collapse dropped — the
    /// falsifiable claim `config_permutations.rs` re-checks by encoding.
    pub duplicates: Vec<(Row, Row)>,
}

/// Group the rows of `array` by their [`Effective`] signature in `ctx`.
pub fn collapse(array: &[Row], ctx: &CellCtx) -> Collapsed {
    let mut seen: BTreeMap<Effective, Row> = BTreeMap::new();
    let mut representatives = Vec::new();
    let mut duplicates = Vec::new();
    for row in array {
        let eff = Effective::resolve(row, ctx);
        match seen.get(&eff) {
            Some(rep) => duplicates.push((*row, *rep)),
            None => {
                seen.insert(eff, *row);
                representatives.push(*row);
            }
        }
    }
    Collapsed {
        representatives,
        duplicates,
    }
}

// ---------------------------------------------------------------------------
// CONTENT axis — the classifier the content taxonomy is built on
// ---------------------------------------------------------------------------

/// libaom's own screen-content statistic for one luma plane, transcribed from
/// `estimate_screen_content` (av1/encoder/encoder.c:2042-2100).
///
/// This is the ONE content property the speed-0 ALLINTRA encoder branches on
/// that is computed from the source pixels rather than from the configuration,
/// and it is a *hard threshold on a countable statistic*, not an adjective —
/// which is why the content taxonomy in
/// `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md` is built on it:
///
/// ```text
/// for each full 16x16 luma block:
///     n_colors = |{ pix >> (bd - 8) }|          // av1_count_colors{,_highbd}
///     if 1 < n_colors <= 4: ++counts_1          // kColorThresh = 4
/// allow_screen_content_tools = counts_1 * 256 * 10 > width * height
/// ```
///
/// (`av1_count_colors_highbd`, intra_mode_search.c:338-370, down-converts to
/// the 8-bit domain before binning — "provides consistency of behavior for
/// palette search between lbd and hbd encodes" — so the statistic is bit-depth
/// independent by construction, and a bd10 vs bd8 pair of the SAME content
/// classifies identically. That is the fact that makes this a CONTENT axis and
/// not a format axis.)
///
/// What the flag then changes on this harness's envelope (palette and intrabc
/// are both forced off by `c_encode_ctrls`, so those are NOT the mechanism):
///
/// * `get_tx_mask` (tx_search.c:1806-1808) resolves
///   `--use-intra-default-tx-only=1` through
///   `get_default_tx_type(PLANE_TYPE_Y, xd, tx_size, cpi->use_screen_content_tools)`,
///   which returns `DCT_DCT` **when the flag is set** instead of the
///   mode-derived tx type. So this axis's meaning literally depends on the
///   content class.
/// * `write_palette_mode_info` / `intra_mode_info_cost_y` gate the per-block
///   palette flag on `av1_allow_palette(allow_screen_content_tools, bsize)`,
///   so every intra block's coded symbols AND rate move.
/// * `set_allintra_speed_features_framesize_independent`
///   (speed_features.c:375-381) and the qp-dependent speed-0 arm (:2909) read
///   it, though the fields they set are inter-only on this envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenStat {
    /// Full 16x16 luma blocks examined (`r + 16 <= h`, `c + 16 <= w`).
    pub blocks: usize,
    /// Blocks with `1 < n_colors <= 4` (libaom's `counts_1`).
    pub counts_1: usize,
    /// `counts_1 * 256 * 10 > w * h` — libaom's verdict.
    pub allow_screen_content_tools: bool,
}

/// Compute [`ScreenStat`] over a tightly-packed `w x h` luma plane.
///
/// `y` holds one sample per entry at every bit depth (the harness's
/// `EncodeCell::y` layout); `bd` is 8, 10 or 12.
pub fn screen_stat(y: &[u16], w: usize, h: usize, bd: u8) -> ScreenStat {
    assert_eq!(y.len(), w * h, "screen_stat: plane is not w*h");
    let shift = bd.saturating_sub(8) as u32;
    let mut blocks = 0usize;
    let mut counts_1 = 0usize;
    let mut r = 0usize;
    while r + 16 <= h {
        let mut c = 0usize;
        while c + 16 <= w {
            let mut bins = [false; 256];
            let mut n_colors = 0usize;
            for br in 0..16 {
                for bc in 0..16 {
                    let v = (y[(r + br) * w + c + bc] >> shift) as usize;
                    // `if (this_val >= max_bin_val) continue;` (the hbd arm).
                    if v >= 256 {
                        continue;
                    }
                    if !bins[v] {
                        bins[v] = true;
                        n_colors += 1;
                    }
                }
            }
            blocks += 1;
            if n_colors > 1 && n_colors <= 4 {
                counts_1 += 1;
            }
            c += 16;
        }
        r += 16;
    }
    ScreenStat {
        blocks,
        counts_1,
        allow_screen_content_tools: counts_1 as u64 * 256 * 10 > (w as u64) * (h as u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx64() -> CellCtx {
        CellCtx {
            w: 64,
            h: 64,
            mono: false,
            sb_px: 64,
        }
    }

    /// Level 0 of every axis must be the aomenc default, so `DEFAULT_ROW`
    /// emits no control pairs at all.
    #[test]
    fn default_row_is_the_stock_config() {
        assert_eq!(knobs_of(&DEFAULT_ROW), ToggleKnobs::default());
        assert!(knobs_of(&DEFAULT_ROW).c_ctrls().is_empty());
    }

    /// Round-trip: apply → read back.
    #[test]
    fn axis_levels_round_trip() {
        for (i, ax) in ALL_AXES.iter().enumerate() {
            for l in 0..ax.n_levels() as u8 {
                let mut row = DEFAULT_ROW;
                row[i] = l;
                assert_eq!(ax.level_of(&knobs_of(&row)), l, "{}", ax.tag());
            }
        }
    }

    /// The documented verified-INERT cases (HANDOFF-TOGGLES.md) must fall OUT
    /// of the resolution — none of them is hardcoded as an exception.
    #[test]
    fn engine_rediscovers_the_documented_inert_cases() {
        let ctx = ctx64();
        let base = Effective::resolve(&DEFAULT_ROW, &ctx);
        let with = |ax: Axis, level: u8| {
            let mut row = DEFAULT_ROW;
            row[ALL_AXES.iter().position(|&a| a == ax).unwrap()] = level;
            Effective::resolve(&row, &ctx)
        };
        // --disable-trellis-quant=0 (FULL) vs the default 3 (NO_ESTIMATE_YRD):
        // differs only in the inter-only estimate_yrd_for_sb.
        assert_eq!(with(Axis::Trellis, 3), base, "trellis 0 must collapse to 3");
        // --cdf-update-mode=2 is identical to 1 on a lone KEY frame.
        assert_eq!(with(Axis::CdfUpdate, 2), base, "cdf mode 2 must collapse to 1");
        // --max-partition-size=64 == the 128 default at SB64.
        assert_eq!(with(Axis::MaxPart, 1), base, "maxpart 64 must collapse at SB64");
        // These three must NOT collapse (sanity: the engine is not degenerate).
        assert_ne!(with(Axis::Trellis, 1), base);
        assert_ne!(with(Axis::CdfUpdate, 1), base);
        assert_ne!(with(Axis::MaxPart, 2), base);
    }

    /// Transitive deaths: rect off kills AB and 1to4; directional off kills
    /// diagonal, angle-delta and the intra edge filter.
    #[test]
    fn engine_collapses_transitively_dead_knobs() {
        let ctx = ctx64();
        let ix = |ax: Axis| ALL_AXES.iter().position(|&a| a == ax).unwrap();
        let mut rect_off = DEFAULT_ROW;
        rect_off[ix(Axis::Rect)] = 1;
        let mut rect_off_ab = rect_off;
        rect_off_ab[ix(Axis::Ab)] = 1;
        let mut rect_off_p14 = rect_off;
        rect_off_p14[ix(Axis::P1to4)] = 1;
        assert_eq!(
            Effective::resolve(&rect_off, &ctx),
            Effective::resolve(&rect_off_ab, &ctx)
        );
        assert_eq!(
            Effective::resolve(&rect_off, &ctx),
            Effective::resolve(&rect_off_p14, &ctx)
        );

        let mut dir_off = DEFAULT_ROW;
        dir_off[ix(Axis::Directional)] = 1;
        for dead in [Axis::Diagonal, Axis::AngleDelta, Axis::EdgeFilter] {
            let mut r = dir_off;
            r[ix(dead)] = 1;
            assert_eq!(
                Effective::resolve(&dir_off, &ctx),
                Effective::resolve(&r, &ctx),
                "{} must be dead with directional intra off",
                dead.tag()
            );
        }
    }

    /// Monochrome kills the whole chroma loop, so the CFL knob collapses.
    #[test]
    fn mono_collapses_the_chroma_knob() {
        let mono = CellCtx {
            mono: true,
            ..ctx64()
        };
        let ix = ALL_AXES.iter().position(|&a| a == Axis::Cfl).unwrap();
        let mut cfl_off = DEFAULT_ROW;
        cfl_off[ix] = 1;
        assert_eq!(
            Effective::resolve(&DEFAULT_ROW, &mono),
            Effective::resolve(&cfl_off, &mono)
        );
        // ... but NOT in 4:2:0.
        assert_ne!(
            Effective::resolve(&DEFAULT_ROW, &ctx64()),
            Effective::resolve(&cfl_off, &ctx64())
        );
    }

    /// A frame smaller than one superblock cannot code a 64x64 block, so tx64
    /// is inert there but live at 64x64.
    #[test]
    fn small_frame_collapses_tx64() {
        let ix = ALL_AXES.iter().position(|&a| a == Axis::Tx64).unwrap();
        let mut tx64_off = DEFAULT_ROW;
        tx64_off[ix] = 1;
        let small = CellCtx {
            w: 32,
            h: 32,
            ..ctx64()
        };
        assert_eq!(
            Effective::resolve(&DEFAULT_ROW, &small),
            Effective::resolve(&tx64_off, &small)
        );
        assert_ne!(
            Effective::resolve(&DEFAULT_ROW, &ctx64()),
            Effective::resolve(&tx64_off, &ctx64())
        );
    }

    /// The one C-forbidden corner is excluded, and only that one.
    #[test]
    fn illegal_pair_is_excluded() {
        let ix = |ax: Axis| ALL_AXES.iter().position(|&a| a == ax).unwrap();
        let mut bad = DEFAULT_ROW;
        bad[ix(Axis::TxSizeSearch)] = 1;
        bad[ix(Axis::Tx64)] = 1;
        assert!(illegal_reason(&bad).is_some());
        let mut ok = bad;
        ok[ix(Axis::Tx64)] = 0;
        assert!(illegal_reason(&ok).is_none());
        assert!(illegal_reason(&DEFAULT_ROW).is_none());
    }

    /// The covering array must actually cover every reachable t-tuple, contain
    /// only legal rows, and be deterministic.
    #[test]
    fn covering_array_is_complete_legal_and_deterministic() {
        for t in [2usize, 3] {
            let rows = covering_array(t);
            assert_eq!(rows[0], DEFAULT_ROW);
            for r in &rows {
                assert!(illegal_reason(r).is_none(), "{}", row_label(r));
            }
            for tp in required_tuples(t) {
                assert!(
                    rows.iter().any(|r| tp.iter().all(|&(a, l)| r[a] == l)),
                    "t={t}: tuple {tp:?} uncovered"
                );
            }
            assert_eq!(rows, covering_array(t), "t={t}: array is not deterministic");
        }
    }
}

// ---------------------------------------------------------------------------
// SPEED-derived encoder state (the SPEED axis of the matrix)
// ---------------------------------------------------------------------------

/// The `--cpu-used` levels the ALLINTRA encoder accepts.
pub const ALL_SPEEDS: [i32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Partition `speeds` into classes with an IDENTICAL resolved
/// [`aom_encode::speed_features::SpeedFeatures`] on the ALLINTRA path.
///
/// **This is a candidate collapse, and it is REFUTED — see
/// [`SPEED_SF_EQUALITY_IS_NOT_A_COLLAPSE`].** It is computed and pinned anyway
/// because the *shape* of the answer is the useful part: it says exactly which
/// speed steps move the speed-feature struct and which do not, so a speed step
/// that stops moving it (or starts) fails the pin instead of silently changing
/// what the matrix covers.
///
/// Returns one entry per class, each `(representative speed, member speeds)`,
/// in ascending representative order.
pub fn speed_sf_classes(speeds: &[i32], screen: bool, hbd: bool) -> Vec<(i32, Vec<i32>)> {
    let mut out: Vec<(aom_encode::speed_features::SpeedFeatures, i32, Vec<i32>)> = Vec::new();
    for &sp in speeds {
        let sf = aom_encode::speed_features::SpeedFeatures::set_allintra(sp, screen, hbd);
        match out.iter_mut().find(|(k, _, _)| *k == sf) {
            Some((_, _, v)) => v.push(sp),
            None => out.push((sf, sp, vec![sp])),
        }
    }
    out.into_iter().map(|(_, r, v)| (r, v)).collect()
}

/// Why [`speed_sf_classes`] must NOT be used to drop a speed context, even
/// though it reports `{7, 9}` as one class on the ALLINTRA path. (It reported
/// `{7, 8, 9}` until KB-32 modelled the `var_part_split_threshold_shift`
/// steps at speeds 8 and 9 — speed 8 now stands alone.)
///
/// The resolved `SpeedFeatures` struct is not the whole speed-derived state:
/// the encoder ALSO branches on the raw `PickFrameCfg::speed`, at thresholds
/// that are not represented in any `SpeedFeatures` field. The ones that
/// separate 7 / 8 / 9 (all in `crates/aom-encode/src`, each mirroring a libaom
/// `speed >= N` gate):
///
/// * `pack.rs:1474` — `use_var_based_partition = allintra && speed >= 7`
///   (`VAR_BASED_PARTITION`, speed_features.c:571);
/// * `pack.rs:1791` / `partition_pick.rs:4569` — `speed >= 8` runs
///   `av1_nonrd_use_partition` (partition_search.c:2960);
/// * `pack.rs:1685`/`:2117` — `cost_upd_off = allintra && speed >= 9`;
/// * `partition_pick.rs:4772` — `hybrid_intra_pickmode` 2 at speed 8, 0 at
///   speed >= 9;
/// * `partition_pick.rs:4854-4856` — `prune_h_pred_using_best_mode_so_far` /
///   `enable_intra_mode_pruning_using_neighbors` /
///   `prune_intra_mode_using_best_sad_so_far`, all `speed >= 9`.
///
/// And it is refuted EMPIRICALLY as well as by inspection: real aomenc's own
/// frame payload differs at `--cpu-used` 7 vs 8 vs 9 on the same cell, which
/// `speed_class_inventory_is_pinned` asserts against the oracle. A collapse the
/// oracle contradicts is not a collapse.
pub const SPEED_SF_EQUALITY_IS_NOT_A_COLLAPSE: &str =
    "SpeedFeatures equality collapses ALLINTRA {7,9}, but the encoder also \
     branches on the raw cfg.speed (pack.rs:1474/1685/1791, \
     partition_pick.rs:4569/4772/4854-4856) and real aomenc's payload differs \
     at cpu-used 7/8/9 — every speed keeps its own context";

/// Is this axis level DEAD (or unreachable for the harness) at `speed`, and
/// why? The speed analogue of [`illegal_reason`], and the reason the covering
/// array pins the axis to its default rather than skipping the whole row.
///
/// One entry, and it is a real libaom-semantics change rather than a harness
/// convenience:
///
/// **`--enable-tx-size-search=0` is a NO-OP at ALLINTRA speed >= 8.** The CLI
/// override is conditional:
///
/// > `if (!oxcf->txfm_cfg.enable_tx_size_search && sf->rt_sf.use_nonrd_pick_mode == 0)`
/// > `  sf->winner_mode_sf.tx_size_search_level = 3;`
/// > — libaom `av1/encoder/speed_features.c:2726-2729`
///
/// and `set_allintra_speed_features_framesize_independent` sets
/// `rt_sf.use_nonrd_pick_mode = 1` at `speed >= 8` (`:579`). So from speed 8 the
/// knob never reaches `tx_size_search_level`, the frame does NOT code
/// `TX_MODE_LARGEST`, and `EncodeCell::port_encode_with`'s
/// "knob OFF must never yield a SELECT header" assertion
/// (`aom-bench/src/lib.rs:1119-1123`) is no longer a valid claim — it fires.
///
/// TWO consequences, both modelled here:
/// 1. the level is covered nowhere at speed >= 8 (it cannot be), so the array
///    pins it to its default there and the fact is pinned as a finding;
/// 2. [`illegal_reason`]'s single C-forbidden pair LAPSES at speed >= 8 — see
///    [`illegal_reason_at_speed`].
pub fn axis_level_dead_at_speed(ax: Axis, level: u8, speed: i32) -> Option<&'static str> {
    if ax == Axis::TxSizeSearch && level == 1 && speed >= 8 {
        return Some(
            "--enable-tx-size-search=0 is inert at ALLINTRA speed>=8: the \
             override at speed_features.c:2726-2729 is gated on \
             use_nonrd_pick_mode==0, which :579 sets to 1 from speed 8",
        );
    }
    None
}

/// [`illegal_reason`], evaluated at a speed.
///
/// The exclusion `--enable-tx-size-search=0 + --enable-tx64=0` exists because
/// `assert(enable_tx64 || tx_search_type != USE_LARGESTALL)`
/// (encodeframe.c:2461) trips when the CLI forces `USE_LARGESTALL`. At ALLINTRA
/// speed >= 8 the CLI no longer forces it (see [`axis_level_dead_at_speed`]), so
/// `tx_search_type != USE_LARGESTALL` holds and the assert cannot fire — the
/// pair becomes LEGAL. The matrix does not exploit that (it pins the `txss`
/// axis to its default from speed 8 anyway, because the level is inert there),
/// but the model must not claim an exclusion libaom does not have.
pub fn illegal_reason_at_speed(row: &Row, speed: i32) -> Option<&'static str> {
    if speed >= 8 {
        return None;
    }
    illegal_reason(row)
}

/// The speed-gated ALLINTRA speed-feature derivations, as
/// `(speed threshold, framesize condition, field, libaom line)`.
///
/// This is the SPEED half of the table [`size_derived`] documents for framesize
/// — specifically the four derivations that are gated on speed AND framesize
/// together, i.e. the ones neither the speed-0 matrix nor a framesize-blind
/// speed sweep can reach alone. Every other entry of
/// `set_allintra_speed_feature_framesize_dependent` is either inert on an intra
/// frame or carries no framesize distinction (see [`size_derived`]'s doc table
/// for those, with citations).
///
/// `framesize` is the condition that must ALSO hold; `"-"` means none.
pub const SPEED_X_FRAMESIZE_DERIVATIONS: &[(&str, &str, &str, &str)] = &[
    (
        ">=2 (level 1) / >=4 (level 2)",
        "is_480p_or_larger",
        "tx_type_search.prune_tx_type_using_stats",
        "speed_features.c:261, :299",
    ),
    (
        ">=1",
        "is_480p_or_larger / is_720p_or_larger tiers",
        "part_sf.use_square_partition_only_threshold",
        "speed_features.c:211-217, :238-242, :315",
    ),
    (
        ">=3",
        "< 720p (no distinction below)",
        "part_sf.max_intra_bsize",
        "speed_features.c:283",
    ),
    (
        ">=2 / >=3",
        "< 480p AND use_highbitdepth",
        "tx_sf.prune_tx_size_level (INTER var-tx only)",
        "speed_features.c:263-265, :289",
    ),
];
