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
}

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
