//! **Self-contained KEY-frame encode** — the outer shell that turns this port's
//! already byte-exact search / transform / entropy / pack layers into a complete
//! AV1 elementary stream, with **no C bootstrap anywhere in the path**.
//!
//! # What was missing before this module
//!
//! Every encoder path in this repo used to start by running real libaom and
//! parsing its output (`aom-bench`'s `port_encode(bootstrap: &[u8])`,
//! `aom-encode/tests/avif_parity.rs`): the sequence-header OBU was **copied
//! verbatim** from the C stream and the frame header's values were **read back**
//! out of C's `OBU_FRAME` with `read_uncompressed_header`. `write_sequence_
//! header_obu` (`aom_dsp::entropy::header`) was byte-exact but had **zero call
//! sites** in any `crates/*/src` — the writers existed, nothing fed them
//! (`docs/CONFIG_AXIS_INVENTORY_2026-07-30.md:477`, `coverage-audit/COVERAGE.md`
//! rows 1.3/1.4).
//!
//! [`encode_key_frame`] closes that: it derives the sequence header, the frame
//! header and the temporal-unit framing from a [`KeyFrameConfig`] plus the source
//! planes, and returns `TD OBU ++ sequence-header OBU ++ OBU_FRAME`.
//!
//! # What is derived here (nothing is replayed)
//!
//! | field | derivation |
//! |---|---|
//! | `seq_level_idx[op]` / `tier[op]` | [`crate::seq_level`] (`set_bitstream_level_tier`, `encoder.c:464`) |
//! | profile, bit depth, subsampling, monochrome | [`KeyFrameConfig`] (the `av1_cx_iface` profile rules, `dec_shim.c` mirror) |
//! | `num_bits_width/height`, `max_frame_*` | `init_seq_coding_tools` (`encoder.c:587`) |
//! | `still_picture` / `reduced_still_picture_hdr` | `init_seq_coding_tools`: a 1-frame encode with no `--full-still-picture-hdr` |
//! | `base_qindex` | [`crate::rc::base_qindex_from_cq`] (gated by `qindex_from_cq_diff`) |
//! | `allow_screen_content_tools` / `allow_intrabc` | [`crate::screen_detect`] (`av1_set_screen_content_options`, `encoder.c:2439`) |
//! | tile grid + per-tile mi bounds | `av1_get_tile_limits` + `av1_calculate_tile_cols/rows` + `av1_tile_set_row`/`_col`'s `AOMMIN` clamp (`tile_common.c`) |
//! | loop-filter levels | [`crate::lf_search::pick_filter_level`] on THIS port's own recon |
//! | `tx_mode` (SELECT vs LARGEST) | the `txb_split_count == 0` flip (`encodeframe.c:2797`), counted off this port's own winner trees |
//! | CDEF damping / bits / per-unit strengths | [`crate::pickcdef::av1_cdef_search`] on the port's own deblocked recon |
//! | loop-restoration frame types / unit size / per-RU params | `aom_dsp::restore::pick::pick_filter_restoration` on the port's own post-CDEF recon |
//! | tile payload | [`crate::pack::pack_tile`] + [`crate::pack::pack_tile_from_trees_lr`] |
//!
//! # Envelope (asserted, not assumed)
//!
//! [`encode_key_frame`] returns [`KeyFrameError`] rather than silently
//! mis-encoding whenever the config leaves the envelope this module is gated on:
//!
//! * ALL-INTRA usage, one KEY frame, `--end-usage=q`, `--cpu-used` 0..=9
//!   (values outside that range are refused);
//! * CDEF and loop restoration in ALL FOUR combinations. NOTE: real aomenc's
//!   ALLINTRA default is CDEF **off** with loop restoration **on** —
//!   `av1_cx_iface.c:3067` sets `enable_cdef = 0` for `AOM_USAGE_ALL_INTRA`
//!   ("CDEF has been found to blur images"). That default is byte-gated at
//!   every speed 0..=9; `--enable-cdef=1` is byte-gated at speeds 0..3 and
//!   pinned-divergent at 4..9 (the FAST search levels, see below);
//! * superblock **64 or 128** ([`KeyFrameConfig::sb_size_128`] — 2026-09-03:
//!   the search/pack layers were already bsize-generic
//!   (`SbEncodeEnv::sb_size`, `rd_pick_partition_real`'s `bsize` param,
//!   `BLOCK_128X128` already used throughout `aom_dsp::entropy::partition`);
//!   the shell only had three hardcoded SB64 constants and the sequence
//!   header's `sb_size_128` bit. Byte-gated: multiple superblocks (up to
//!   3x3), a non-multiple-of-128 size (a partial edge superblock), CDEF+LR
//!   both on, bd10, speed 0 through 9, cq 0 and 63, and composed with an
//!   explicit multi-tile request), no superres, no QM, no film grain, no
//!   segmentation, no delta-q, palette + IntraBC search off (matching
//!   `aom-sys-ref`'s `shim_encode_av1_kf`, whose `--enable-palette=0
//!   --enable-intrabc=0` is what the byte gate compares against);
//! * **multi-tile** in the form `av1_get_tile_limits` MANDATES it (frames wider
//!   than `MAX_TILE_WIDTH` = 4096 px, or larger than `MAX_TILE_AREA`): each
//!   tile is packed independently with a fresh frame context — C's
//!   `write_modes` per-tile reset — and assembled through
//!   [`crate::obu_assemble::assemble_multitile_frame_obu_payload_derived`].
//!   Byte-gated at speeds 0..6. **Explicit `--tile-columns` / `--tile-rows`**
//!   ([`KeyFrameConfig::tile_columns_log2`] / [`KeyFrameConfig::tile_rows_log2`],
//!   2026-09-03) are now exposed too — `derive_tile_info` already implemented
//!   C's `set_tile_info` clamp (`AOMMAX(tile_columns_cfg, min_log2_cols)`)
//!   correctly, it just always received `0, 0`. Byte-gated: forcing more
//!   tiles than the uniform-spacing default, requesting above a mandatory
//!   frame's minimum, and requesting BELOW a mandatory minimum (must clamp
//!   up — verified byte-identical to the same frame's un-requested cell).
//!
//! # Not yet wired (named, with the specific entry points)
//!
//! * **`av1_determine_sc_tools_with_encoding`** (`encoder_utils.c:1214`) — C's
//!   two-pass trial encode that can turn screen-content tools ON after the
//!   detector said off. Unported. It returns early when the detector already
//!   said on, so it only ever matters on detector-negative content; the byte
//!   gate holds this accountable per cell, and (2026-09-03) two adversarial
//!   differential probes designed specifically to find a counterexample —
//!   including one that brackets the base detector's own threshold crossover
//!   from both sides — found none in 105 cells (`self_contained_key_frame.rs`'s
//!   `probe_sc_tools_trial_gap_*` tests). Still unported; the port cost is
//!   independently scoped at PARITY.md C3 ("(M)", "NOT a one-sitting port" —
//!   a fixed-32x32-partition trial-encode driver + PSNR-based decisioning
//!   this shell does not have).
//! * **`--cpu-used` >= 7 above roughly 3x3 superblocks** — measured bracket:
//!   at speed 7, 128x128 / 160x160 / 192x192 / 128x192 / 192x128 are
//!   byte-exact and 256x256 / 320x320 are not; at speed 9, 192x192 is not
//!   either. One unlocalized VAR_BASED_PARTITION / nonrd arm, pinned rather
//!   than refused (the streams are valid and decode).
//! * **`--enable-cdef=1` at `--cpu-used` >= 4** — `sf.cdef_pick_method` leaves
//!   `CDEF_FULL_SEARCH` for the FAST levels there, which PARITY.md C1 records
//!   as ported + table-unit-tested but never e2e-gated. MEASURED 2026-09-02:
//!   divergent from real aomenc on every cell tried at speeds 4..9, in the
//!   header's `cdef_strengths` set only (the per-unit indices in the tile
//!   payload are byte-identical). Not refused — the stream is valid and
//!   decodes — but pinned in the gate.
//!
//! # Post-filter composition (CDEF + loop restoration together)
//!
//! Neither pack entry point covered a frame with CDEF AND restoration on:
//! `pack_tile_from_trees` carried only the CDEF strength literals and
//! `pack_tile_lr` only the interleaved per-RU restoration params. That
//! combination is reachable with `--enable-cdef=1 --enable-restoration=1`
//! (it is NOT the ALLINTRA default — see the envelope note above), so this
//! landing added
//! [`crate::pack::pack_tile_from_trees_lr`] (both, additive —
//! `pack_tile_from_trees` now delegates to it with `lr: None` and is
//! byte-unchanged) and follows C's `cdef_restoration_frame` order exactly:
//! deblock → `av1_cdef_search` → `av1_cdef_frame` (apply) →
//! `av1_pick_filter_restoration` on the POST-CDEF reconstruction. The LR
//! search sees both frames — `deblocked` and `cur` — because C saves boundary
//! lines from each (`av1_loop_restoration_save_boundary_lines` calls 0 and 1).

use aom_dsp::cdef::frame::{CdefFrameParams, cdef_frame};
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::header::{
    CdefHeader, ColorConfigParams, DecoderModelInfo, DeltaQParams, FrameHeaderObu,
    FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader, QuantParamsHeader, RestorationHeader,
    SequenceHeaderObu, SequenceHeaderParams, TileInfoHeader, TimingInfoHeader,
    write_sequence_header_obu,
};
use aom_dsp::entropy::leb128::uleb_encode;
use aom_dsp::entropy::lr::{LrFrameConfig, RESTORE_NONE as LR_RESTORE_NONE};
use aom_dsp::entropy::obu::write_obu_header;
use aom_dsp::entropy::partition::{KfFrameContext, tx_size_to_depth};
use aom_dsp::entropy::wb::WriteBitBuffer;
use aom_dsp::loopfilter::frame::{LfFrameBuf, LfMiGrid, LfParams, loop_filter_frame};
use aom_dsp::quant::av1_dc_quant_qtx;
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_dsp::restore::pick::{LrPlanePixels, LrSearchInput, pick_filter_restoration};
use aom_dsp::txb::cost_tokens_from_cdf;

use crate::encode_intra::TrellisOptType;
use crate::encode_sb::{LeafWinner, SbEncodeEnv, SbTree};
use crate::intra_uv_rd::UvLoopPolicy;
use crate::lf_search::{
    LfSearchFrame, LoopFilterLevels, build_lf_mi_grid, pick_filter_level, pick_filter_level_from_q,
};
use crate::obu_assemble::{
    OBU_FRAME, assemble_multitile_frame_obu_payload_derived, assemble_obu_frame_single_tile,
};
use crate::pack::{CdefPackState, LrPackParams, PackCfg, pack_tile, pack_tile_from_trees_lr};
use crate::partition_pick::PickFrameCfg;
use crate::pickcdef::{CdefSearchFrame, av1_cdef_search};
use crate::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use crate::real_costs::derive_real_costs;
use crate::screen_detect::ScreenContentDecision;
use crate::speed_features::SpeedFeatures;
use crate::tx_search::MI_SIZE_WIDE_B;

/// `OBU_TEMPORAL_DELIMITER` (`av1/common/enums.h` `OBU_TYPE`).
pub const OBU_TEMPORAL_DELIMITER: u32 = 2;
/// `OBU_SEQUENCE_HEADER`.
pub const OBU_SEQUENCE_HEADER: u32 = 1;

/// `BLOCK_64X64` in the port's block-size enum.
const SB_BLOCK_64: usize = 12;
/// 64px / 4 — the mi units in one SB64 side.
const SB_MI_64: i32 = 16;
/// `BLOCK_128X128` in the port's block-size enum (`aom_dsp::entropy::partition`).
const SB_BLOCK_128: usize = 15;
/// 128px / 4 — the mi units in one SB128 side.
const SB_MI_128: i32 = 32;

/// `av1_set_default_ref_deltas` for a KEY frame (`loopfilter.c`) — the deltas a
/// frame with no primary ref starts from, and therefore the `last_*` the header
/// diffs against.
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
/// `av1_set_default_mode_deltas`.
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

/// `PRIMARY_REF_NONE`.
const PRIMARY_REF_NONE: i32 = 7;

/// Everything [`encode_key_frame`] needs that is not the pixels.
///
/// The field set is deliberately the CLI-equivalent one
/// `aom-sys-ref`'s `shim_encode_av1_kf` drives, so a byte diff against real
/// aomenc is a like-for-like comparison.
#[derive(Clone, Copy, Debug)]
pub struct KeyFrameConfig {
    /// `cfg.g_w` — the true crop width in luma pixels.
    pub width: usize,
    /// `cfg.g_h`.
    pub height: usize,
    /// `cfg.g_bit_depth` / `g_input_bit_depth`: 8, 10 or 12.
    pub bit_depth: u8,
    /// `cfg.monochrome`.
    pub monochrome: bool,
    /// Chroma subsampling. Monochrome carries `(1, 1)` (the `AOM_IMG_FMT_I420`
    /// the shim allocates for a mono image), matching every harness in this
    /// repo.
    pub ss_x: usize,
    /// See [`Self::ss_x`].
    pub ss_y: usize,
    /// `AOME_SET_CQ_LEVEL` — the 0..=63 quantizer level, mapped to
    /// `base_qindex` by [`crate::rc::base_qindex_from_cq`].
    pub cq_level: i32,
    /// `AOME_SET_CPUUSED` — 0..=9. Byte-parity with real aomenc holds across
    /// the whole range with CDEF off (the ALLINTRA default); see the module
    /// docs for the two pinned regions (`--enable-cdef=1` at >= 4, and >= 7
    /// above roughly 3x3 superblocks).
    pub cpu_used: i32,
    /// `aom_codec_enc_config_default`'s usage: 2 = `AOM_USAGE_ALL_INTRA`,
    /// 0 = `AOM_USAGE_GOOD_QUALITY`.
    pub usage: u32,
    /// `AV1E_SET_ENABLE_CDEF`. Real aomenc's ALLINTRA default is **0**
    /// (`av1_cx_iface.c:3067`).
    pub enable_cdef: bool,
    /// `AV1E_SET_ENABLE_RESTORATION`. Real aomenc's ALLINTRA default is **1**.
    /// Note the SEQUENCE-header bit is cleared at speed >= 5 regardless
    /// (`speed_features.c:2753`), which [`derive_sequence_header`] models.
    pub enable_restoration: bool,
    /// `AV1E_SET_TILE_COLUMNS` — the requested tile-column count is
    /// `2^tile_columns_log2` (`--tile-columns=N` on the aomenc CLI IS the
    /// log2 value already). `0` (the [`Self::allintra_speed0`] default)
    /// matches every existing gate: uniform spacing with no explicit
    /// request, so the MANDATORY minimum from `av1_get_tile_limits` /
    /// `set_tile_info` is what actually governs (see [`derive_tile_info`]).
    /// A request below that minimum is clamped up, exactly as C clamps it —
    /// this field cannot force FEWER tiles than a large frame requires.
    pub tile_columns_log2: i32,
    /// `AV1E_SET_TILE_ROWS`. See [`Self::tile_columns_log2`].
    pub tile_rows_log2: i32,
    /// `AV1E_SET_SUPERBLOCK_SIZE` (`AOM_SUPERBLOCK_SIZE_128X128` when true,
    /// the `_64X64` default otherwise). `false` (the
    /// [`Self::allintra_speed0`] default) matches every other gate in this
    /// file.
    pub sb_size_128: bool,
}

impl KeyFrameConfig {
    /// The `shim_encode_av1_kf` envelope with CDEF and loop restoration OFF:
    /// ALL-INTRA, `--cpu-used 0`. Set [`Self::enable_cdef`] /
    /// [`Self::enable_restoration`] for the other three post-filter
    /// combinations (all four are gated). Real aomenc's ALLINTRA default is
    /// CDEF **off** with restoration **on** (`av1_cx_iface.c:3067`).
    /// [`Self::cpu_used`] accepts 0..=9.
    pub fn allintra_speed0(
        width: usize,
        height: usize,
        bit_depth: u8,
        monochrome: bool,
        ss_x: usize,
        ss_y: usize,
        cq_level: i32,
    ) -> Self {
        Self {
            width,
            height,
            bit_depth,
            monochrome,
            ss_x,
            ss_y,
            cq_level,
            cpu_used: 0,
            usage: 2,
            enable_cdef: false,
            enable_restoration: false,
            tile_columns_log2: 0,
            tile_rows_log2: 0,
            sb_size_128: false,
        }
    }

    /// `cfg.g_profile` exactly as `encode_av1_kf_impl` (`dec_shim.c:508-518`)
    /// derives it from bit depth + subsampling, which is itself the
    /// `av1_cx_iface` rule: 4:4:4 at 8/10-bit is PROFILE_1, 12-bit and 4:2:2
    /// are PROFILE_2, everything else PROFILE_0.
    pub fn profile(&self) -> i32 {
        let is_444 = self.ss_x == 0 && self.ss_y == 0;
        let mut profile = if self.bit_depth == 12 {
            2
        } else if is_444 {
            1
        } else {
            0
        };
        if !self.monochrome && self.ss_x == 1 && self.ss_y == 0 {
            profile = 2; // 4:2:2
        }
        profile
    }

    /// `mono ? 1 : 3`.
    pub fn num_planes(&self) -> usize {
        if self.monochrome { 1 } else { 3 }
    }

    /// The chroma plane dimensions (`(0, 0)` when monochrome).
    pub fn chroma_dims(&self) -> (usize, usize) {
        if self.monochrome {
            (0, 0)
        } else {
            (
                (self.width + self.ss_x) >> self.ss_x,
                (self.height + self.ss_y) >> self.ss_y,
            )
        }
    }
}

/// Source planes for [`encode_key_frame`]: tightly packed `u16` samples in the
/// `bit_depth`-bit range (8-bit sources carry 8-bit values), `y.len() == w*h`
/// and, when not monochrome, `u.len() == v.len() == cw*ch`. Same convention as
/// `aom_sys_ref::ref_encode_av1_kf`, so a caller can hand the identical buffers
/// to both sides of a differential.
#[derive(Clone, Copy, Debug)]
pub struct KeyFramePlanes<'a> {
    /// Luma.
    pub y: &'a [u16],
    /// Cb (empty when monochrome).
    pub u: &'a [u16],
    /// Cr (empty when monochrome).
    pub v: &'a [u16],
}

/// Why [`encode_key_frame`] refused. Every variant is a configuration this
/// module has no gate for — never a silent fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyFrameError {
    /// A plane length did not match the config's dimensions.
    PlaneSize {
        /// 0 = Y, 1 = U, 2 = V.
        plane: usize,
        /// What the config implies.
        expected: usize,
        /// What was handed in.
        got: usize,
    },
    /// A config field is outside this shell's gated envelope. Carries the
    /// field name and the reason.
    Unsupported(&'static str),
}

impl core::fmt::Display for KeyFrameError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KeyFrameError::PlaneSize {
                plane,
                expected,
                got,
            } => write!(f, "plane {plane}: expected {expected} samples, got {got}"),
            KeyFrameError::Unsupported(what) => write!(f, "outside the gated envelope: {what}"),
        }
    }
}

/// `get_msb(n) + 1` for `n > 0`, else 0 (`common.h` `get_unsigned_bits`, as
/// `init_seq_coding_tools` uses it for `num_bits_width/height`).
fn num_bits_for_dim(dim: i32) -> u32 {
    if dim > 1 {
        (32 - (dim as u32 - 1).leading_zeros()).max(1)
    } else {
        1
    }
}

/// `CEIL_POWER_OF_TWO(value, n)`.
fn ceil_power_of_two(value: i32, n: u32) -> i32 {
    (value + (1 << n) - 1) >> n
}

/// `tile_log2(blk_size, target)` (`tile_common.c:32`).
fn tile_log2(blk_size: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// `mi_params->mi_cols` / `mi_rows` for a frame dimension in pixels
/// (`av1_set_mb_mi`: the 8px-aligned size in 4px mi units).
fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// `av1_get_tile_limits` + `av1_set_tile_info` + `av1_calculate_tile_cols` /
/// `av1_calculate_tile_rows` (`tile_common.c`) for the UNIFORM-spacing grid
/// libaom's encoder always emits at the default `--tile-columns/--tile-rows`.
///
/// `tile_cols_log2_cfg` / `tile_rows_log2_cfg` are the `AV1E_SET_TILE_COLUMNS` /
/// `_ROWS` values (0 by default); `av1_set_tile_info` takes
/// `AOMMAX(cfg, min_log2_*)`, so a frame whose limits force a split still gets
/// the right grid.
pub fn derive_tile_info(
    mi_cols: i32,
    mi_rows: i32,
    mib_size_log2: u32,
    tile_cols_log2_cfg: i32,
    tile_rows_log2_cfg: i32,
) -> TileInfoHeader {
    const MAX_TILE_WIDTH: i32 = 4096;
    const MAX_TILE_AREA: i32 = 4096 * 2304;
    const MAX_TILE_COLS: i32 = 64;
    const MAX_TILE_ROWS: i32 = 64;
    let sb_cols = ceil_power_of_two(mi_cols, mib_size_log2);
    let sb_rows = ceil_power_of_two(mi_rows, mib_size_log2);
    let sb_size_log2 = mib_size_log2 as i32 + 2;
    let max_width_sb = MAX_TILE_WIDTH >> sb_size_log2;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size_log2);
    let min_log2_cols = tile_log2(max_width_sb, sb_cols);
    let max_log2_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = tile_log2(max_tile_area_sb, sb_cols * sb_rows).max(min_log2_cols);

    let mut t = TileInfoHeader {
        mi_cols,
        mi_rows,
        mib_size_log2,
        uniform_spacing: true,
        min_log2_cols,
        max_log2_cols,
        min_log2_rows: 0,
        max_log2_rows,
        max_width_sb,
        max_height_sb: (max_tile_area_sb / max_width_sb.max(1)).max(1),
        ..Default::default()
    };
    // `set_tile_info` (`av1/encoder/encoder.c:382-392`), which is STRICTER than
    // `av1_get_tile_limits`' own `min_log2_cols`:
    //
    //   log2_cols = AOMMAX(tile_columns, tiles->min_log2_cols);
    //   int min_log2_cols = 0;
    //   for (; (max_width_sb << min_log2_cols) <= sb_cols; ++min_log2_cols) {}
    //   log2_cols = AOMMAX(log2_cols, min_log2_cols);
    //   log2_cols = AOMMIN(log2_cols, max_log2_cols);
    //
    // Note the `<=`, where `tile_log2` uses `<`. They differ by exactly one
    // when `sb_cols` is an exact multiple-by-power-of-two of `max_width_sb` —
    // i.e. at a frame EXACTLY `MAX_TILE_WIDTH` wide (4096 px at SB64: sb_cols
    // == 64 == max_width_sb, `tile_log2` says 0 tiles-log2 and this loop says
    // 1). Without it a 4096-wide frame codes ONE tile where real aomenc codes
    // two.
    t.log2_cols = tile_cols_log2_cfg.max(min_log2_cols);
    let mut strict_min_log2_cols = 0i32;
    while (max_width_sb << strict_min_log2_cols) <= sb_cols {
        strict_min_log2_cols += 1;
    }
    t.log2_cols = t.log2_cols.max(strict_min_log2_cols).min(max_log2_cols);
    // av1_calculate_tile_cols.
    let size_sb_c = ceil_power_of_two(sb_cols, t.log2_cols as u32);
    let mut start_sb = 0;
    let mut i = 0usize;
    while start_sb < sb_cols {
        t.col_start_sb[i] = start_sb;
        start_sb += size_sb_c;
        i += 1;
    }
    t.cols = i;
    t.col_start_sb[i] = sb_cols;
    t.min_log2_rows = (min_log2_tiles - t.log2_cols).max(0);
    t.max_height_sb = sb_rows >> t.min_log2_rows;
    // av1_set_tile_info + av1_calculate_tile_rows.
    t.log2_rows = tile_rows_log2_cfg.max(t.min_log2_rows).min(max_log2_rows);
    let size_sb_r = ceil_power_of_two(sb_rows, t.log2_rows as u32);
    let mut start_sb = 0;
    let mut j = 0usize;
    while start_sb < sb_rows {
        t.row_start_sb[j] = start_sb;
        start_sb += size_sb_r;
        j += 1;
    }
    t.rows = j;
    t.row_start_sb[j] = sb_rows;
    t
}

/// Author the sequence header from the config alone — `init_seq_coding_tools`
/// (`encoder.c:587`) + `av1_change_config`'s color-config fill, for the
/// one-frame still-picture envelope this shell encodes.
///
/// Everything the reduced still-picture header does NOT code (frame-id lengths,
/// the inter tool-enable bits, order-hint config) is set to the value C's init
/// leaves in `seq_params` anyway, so the struct is honest even where the bits
/// are absent.
pub fn derive_sequence_header(cfg: &KeyFrameConfig) -> SequenceHeaderObu {
    let w = cfg.width as i32;
    let h = cfg.height as i32;
    // `still_picture = !force_video_mode && limit == 1`; `reduced_still_picture_hdr
    // = still_picture && !full_still_picture_hdr` (encoder.c:594-597).
    let reduced_still_picture_hdr = true;
    let level = crate::seq_level::seq_header_seq_level_idx(
        w,
        h,
        crate::seq_level::STILL_PICTURE_FPS,
        crate::seq_level::SEQ_LEVEL_MAX,
    );
    SequenceHeaderObu {
        profile: cfg.profile(),
        still_picture: true,
        reduced_still_picture_hdr,
        // `timing_info_present` is `--timing-info != unspecified` (default off);
        // with it off the decoder-model / display-model blocks are unreachable.
        timing_info_present: false,
        timing_info: TimingInfoHeader {
            num_units_in_display_tick: 0,
            time_scale: 0,
            equal_picture_interval: false,
            num_ticks_per_picture: 1,
        },
        decoder_model_info_present_flag: false,
        // `av1_cx_iface`'s zero-init: the three `-1` length fields are written
        // as `len - 1`, so 1 is the value a fresh decoder-model block carries.
        decoder_model_info: DecoderModelInfo {
            encoder_decoder_buffer_delay_length: 1,
            num_units_in_decoding_tick: 0,
            buffer_removal_time_length: 1,
            frame_presentation_time_length: 1,
        },
        display_model_info_present_flag: false,
        operating_points_cnt_minus_1: 0,
        operating_point_idc: [0; 32],
        // set_bitstream_level_tier writes the SAME level into every operating
        // point (encoder.c:538-546); only op 0 exists here.
        seq_level_idx: [level; 32],
        tier: [0; 32],
        op_decoder_model_param_present: [false; 32],
        op_display_model_param_present: [false; 32],
        op_decoder_buffer_delay: [0; 32],
        op_encoder_buffer_delay: [0; 32],
        op_low_delay_mode_flag: [false; 32],
        op_initial_display_delay: [0; 32],
        seq_header: SequenceHeaderParams {
            num_bits_width: num_bits_for_dim(w),
            num_bits_height: num_bits_for_dim(h),
            max_frame_width: w,
            max_frame_height: h,
            reduced_still_picture_hdr,
            // `!reduced_still_picture_hdr && !large_scale_tile &&
            // error_resilient_mode` (encoder.c:601) -> false here.
            frame_id_numbers_present_flag: false,
            // FRAME_ID_LENGTH / DELTA_FRAME_ID_LENGTH: set unconditionally by
            // init_seq_coding_tools (:625-626) but never CODED in a reduced
            // still-picture header.
            delta_frame_id_length: 14,
            frame_id_length: 15,
            sb_size_128: cfg.sb_size_128,
            // `--enable-filter-intra` / `--enable-intra-edge-filter` default on
            // (`av1_cx_iface` extra_cfg defaults).
            enable_filter_intra: true,
            enable_intra_edge_filter: true,
            // The inter tool bits are not coded in a reduced still-picture
            // header; C's own init leaves them at the ALLINTRA-forced 0
            // (`av1_cx_iface.c` turns the inter tools off for usage 2).
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_warped_motion: false,
            enable_dual_filter: false,
            // reduced_still_picture_hdr forces order hint off (encoder.c:606-610).
            enable_order_hint: false,
            enable_dist_wtd_comp: false,
            enable_ref_frame_mvs: false,
            // Both forced to SELECT by init_seq_coding_tools (:599-600, :607-609).
            force_screen_content_tools: 2,
            force_integer_mv: 2,
            order_hint_bits_minus_1: -1,
            // `--enable-superres` defaults off for a still encode.
            enable_superres: false,
            enable_cdef: cfg.enable_cdef,
            // `av1_set_speed_features_framesize_independent`'s epilogue
            // (speed_features.c:2746-2758, `if (!seq_params_locked)`):
            //   seq->enable_restoration &= (!disable_wiener_filter ||
            //                               !disable_sgr_filter)
            // and the ALLINTRA cascade sets BOTH disables at speed >= 5
            // (`set_allintra_speed_features_framesize_independent`), so the
            // SEQUENCE-HEADER bit is cleared there regardless of
            // `--enable-restoration`. (The `num_workers > 1` twin at :2740 is
            // dead for this oracle: `CONFIG_MULTITHREAD=0` and `g_threads = 1`.)
            // MEASURED 2026-09-02: without this the standalone encode diverged
            // from real aomenc on EVERY `--enable-restoration=1` cell at speed
            // 5..9 and byte-matched at 0..4.
            //
            // The same epilogue also clears `enable_dual_filter` and
            // `enable_interintra_compound`; neither is coded in a reduced
            // still-picture header, so both are byte-inert here.
            enable_restoration: cfg.enable_restoration && cfg.cpu_used < 5,
        },
        color_config: ColorConfigParams {
            bit_depth: i32::from(cfg.bit_depth),
            profile: cfg.profile(),
            monochrome: cfg.monochrome,
            // AOM_CICP_*_UNSPECIFIED (2/2/2) -> `write_color_config` codes "no
            // colour description", which is what an unconfigured encode emits.
            color_primaries: 2,
            transfer_characteristics: 2,
            matrix_coefficients: 2,
            // AOM_CR_STUDIO_RANGE.
            color_range: false,
            subsampling_x: cfg.ss_x as i32,
            subsampling_y: cfg.ss_y as i32,
            // AOM_CSP_UNKNOWN.
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
        },
        film_grain_params_present: false,
    }
}

/// Author the frame header from the config + the sequence header, with the
/// post-search fields (`loopfilter.filter_level*`, `tx_mode_select`) left at
/// their pre-search values for [`encode_key_frame`] to fill from its own
/// reconstruction and winner trees.
///
/// `allow_screen_content_tools` is the caller's already-derived
/// [`ScreenContentDecision`] (it needs the source pixels, which this function
/// does not take).
pub fn derive_frame_header(
    cfg: &KeyFrameConfig,
    seq: &SequenceHeaderObu,
    sct: &ScreenContentDecision,
    tile_info: TileInfoHeader,
) -> FrameHeaderObu {
    let s = &seq.seq_header;
    let cc = &seq.color_config;
    let base_qindex = crate::rc::base_qindex_from_cq(cfg.cq_level);
    // `frame_is_coded_lossless` for a segmentation-off KEY frame with no
    // superres: base_qindex 0 and all five plane deltas 0.
    let coded_lossless = base_qindex == 0;
    FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            show_existing_frame: false,
            existing_fb_idx_to_show: 0,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time: 0,
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length
                as u32,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            frame_id_length: s.frame_id_length as u32,
            display_frame_id: 0,
            frame_type: 0, // KEY_FRAME
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: false,
            // `frame_parallel_decoding_mode` (default 0).
            disable_cdf_update: false,
            force_screen_content_tools: s.force_screen_content_tools,
            allow_screen_content_tools: sct.allow_screen_content_tools,
            force_integer_mv: s.force_integer_mv,
            // `cur_frame_force_integer_mv` is intra-only-forced 0 for a KEY
            // frame (`av1_setup_frame_features`).
            cur_frame_force_integer_mv: false,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            max_frame_width: s.max_frame_width,
            max_frame_height: s.max_frame_height,
            current_frame_id: 0,
            enable_order_hint: s.enable_order_hint,
            order_hint: 0,
            order_hint_bits_minus_1: s.order_hint_bits_minus_1,
            primary_ref_frame: PRIMARY_REF_NONE,
            buffer_removal_time_present: false,
            operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
            op_decoder_model_param_present: seq.op_decoder_model_param_present,
            operating_point_idc: seq.operating_point_idc,
            temporal_layer_id: 0,
            spatial_layer_id: 0,
            buffer_removal_times: [0; 32],
            buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length as u32,
            // A shown KEY frame refreshes every slot; not CODED for
            // `frame_type == KEY && show_frame` (write_frame_header_prefix).
            refresh_frame_flags: 0xff,
            ref_frame_map_order_hint: [0; 8],
        },
        allow_screen_content_tools: sct.allow_screen_content_tools,
        superres_scaled: false,
        // `allow_intrabc` is the SEARCH-time decision, flipped to 0 after the
        // frame when no block used IntraBC (encodeframe.c:2442). This shell
        // runs no IntraBC search, so the coded bit is 0; the search-time value
        // still rides in `sct.allow_intrabc` for the caller.
        allow_intrabc: false,
        frame_size: FrameSizeHeader {
            frame_size_override: false,
            num_bits_width: s.num_bits_width,
            num_bits_height: s.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            // SCALE_NUMERATOR — "no superres scaling".
            scale_denominator: 8,
            // `render_and_frame_size_different`: the render size equals the
            // frame size, so nothing is coded.
            scaling_active: false,
            render_width: s.max_frame_width,
            render_height: s.max_frame_height,
        },
        tile_info,
        context_update_tile_id: 0,
        tile_size_bytes: 1,
        quant: QuantParamsHeader {
            base_qindex,
            y_dc_delta_q: 0,
            u_dc_delta_q: 0,
            u_ac_delta_q: 0,
            v_dc_delta_q: 0,
            v_ac_delta_q: 0,
            using_qmatrix: false,
            qmatrix_level_y: 0,
            qmatrix_level_u: 0,
            qmatrix_level_v: 0,
        },
        num_planes: cfg.num_planes(),
        separate_uv_delta_q: cc.separate_uv_delta_q,
        segmentation: Default::default(),
        delta_q: DeltaQParams {
            base_qindex,
            delta_q_present: false,
            delta_q_res: 1,
            allow_intrabc: false,
            delta_lf_present: false,
            delta_lf_res: 1,
            delta_lf_multi: false,
        },
        all_lossless: coded_lossless,
        coded_lossless,
        loopfilter: LoopfilterHeader {
            allow_intrabc: false,
            // Filled by `encode_key_frame` from `pick_filter_level`.
            filter_level: [0, 0],
            filter_level_u: 0,
            filter_level_v: 0,
            sharpness_level: 0,
            // `av1_loop_filter_frame_init`: a frame with no primary ref starts
            // from the defaults with mode/ref deltas ENABLED and unchanged, so
            // `mode_ref_delta_update` is false and no delta bits are coded.
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: false,
            ref_deltas: KF_REF_DELTAS,
            mode_deltas: KF_MODE_DELTAS,
            last_ref_deltas: KF_REF_DELTAS,
            last_mode_deltas: KF_MODE_DELTAS,
        },
        cdef: CdefHeader {
            enable_cdef: s.enable_cdef,
            allow_intrabc: false,
            cdef_damping: 3,
            cdef_bits: 0,
            nb_cdef_strengths: 1,
            cdef_strengths: [0; 8],
            cdef_uv_strengths: [0; 8],
        },
        restoration: RestorationHeader {
            enable_restoration: s.enable_restoration,
            allow_intrabc: false,
            frame_restoration_type: [0; 3],
            sb_size_128: s.sb_size_128,
            restoration_unit_size: [256, 256, 256],
            subsampling_x: cc.subsampling_x,
            subsampling_y: cc.subsampling_y,
        },
        // Filled by `encode_key_frame` after the txb_split_count flip.
        tx_mode_select: !coded_lossless,
        reference_mode_select: false,
        skip_mode_allowed: false,
        derive_skip_mode_allowed: false,
        skip_mode_flag: false,
        might_allow_warped_motion: false,
        allow_warped_motion: false,
        // `--reduced-tx-type-set` default off.
        reduced_tx_set_used: false,
        global_motion: Default::default(),
        ref_global_motion: Default::default(),
        film_grain_params_present: false,
        film_grain: Default::default(),
        large_scale: false,
        // `refresh_frame_context == REFRESH_FRAME_CONTEXT_DISABLED` for the
        // reduced still-picture header (nothing coded).
        refresh_frame_context_disabled: true,
        inter_ref: Default::default(),
        frame_size_with_refs: Default::default(),
        cur_frame_force_integer_mv: false,
        allow_high_precision_mv: false,
        interp_filter: 0,
        switchable_motion_mode: false,
        might_allow_ref_frame_mvs: false,
        allow_ref_frame_mvs: false,
    }
}

/// `txb_split_count` (`partition_search.c:517` / `:555`) for an all-intra
/// frame: the number of CODED leaves whose winning uniform `tx_size` is not the
/// block's `max_txsize_rect_lookup[bsize]`. Both C increment sites reduce to
/// that test on the intra path (the `is_inter` arms are unreachable here), so
/// it is computable straight off the picked trees without a second pack.
///
/// Used for `av1_encode_frame`'s post-frame `tx_mode` flip
/// (`encodeframe.c:2797`): `TX_MODE_SELECT && txb_split_count == 0` →
/// `TX_MODE_LARGEST`.
///
/// **CODED is load-bearing.** The walk carries the same frame-bound guards
/// `stamp_lf_tree` (`crate::lf_search`) and the pack walk use: a `Split` child
/// or rect sub-block whose origin is off-frame is never coded by C's
/// `encode_sb` (`partition_search.c:1583`) and therefore never contributes.
/// Counting one is not merely a miscount — those tree slots hold placeholder
/// winners whose `tx_size` is not on `bsize`'s `sub_tx_size_map` chain at all,
/// which trips `tx_size_to_depth`'s own `depth <= MAX_TX_DEPTH` assert. (Found
/// exactly that way on the first partial-superblock cell, `128x96 4:4:4`.)
pub fn txb_split_count(
    trees: &[SbTree],
    mi_rows: i32,
    mi_cols: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
) -> u32 {
    let mut n = 0u32;
    for (idx, tree) in trees.iter().enumerate() {
        let r = idx as i32 / n_sb_cols;
        let c = idx as i32 % n_sb_cols;
        count_tree(
            tree,
            r * sb_mi,
            c * sb_mi,
            sb_size,
            mi_rows,
            mi_cols,
            &mut n,
        );
    }
    n
}

/// One coded leaf's contribution: `mbmi->tx_size != max_txsize_rect_lookup
/// [bsize]`, expressed as `tx_size_to_depth(..) != 0` (the depth is exactly the
/// number of `sub_tx_size_map` steps from that max, so zero ⟺ equal).
fn count_leaf(n: &mut u32, w: &LeafWinner) {
    if tx_size_to_depth(w.tx_size, w.bsize) != 0 {
        *n += 1;
    }
}

/// Frame-bound-guarded partition walk, structurally mirroring
/// `crate::lf_search`'s `stamp_lf_tree` (same guards, same sub-block origins).
fn count_tree(
    tree: &SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
    n: &mut u32,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
    let quarter = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
    match tree {
        SbTree::Absent => {}
        SbTree::Leaf(w) => count_leaf(n, w),
        SbTree::Split(kids) => {
            let sub = crate::partition::split_subsize(bsize);
            for (i, child) in kids.iter().enumerate() {
                count_tree(
                    child,
                    mi_row + ((i as i32) >> 1) * hbs,
                    mi_col + ((i as i32) & 1) * hbs,
                    sub,
                    mi_rows,
                    mi_cols,
                    n,
                );
            }
        }
        SbTree::Horz(subs) => {
            count_leaf(n, &subs[0]);
            if mi_row + hbs < mi_rows {
                count_leaf(n, &subs[1]);
            }
        }
        SbTree::Vert(subs) => {
            count_leaf(n, &subs[0]);
            if mi_col + hbs < mi_cols {
                count_leaf(n, &subs[1]);
            }
        }
        SbTree::Horz4(subs) => {
            for (i, w) in subs.iter().enumerate() {
                if i > 0 && mi_row + (i as i32) * quarter >= mi_rows {
                    break;
                }
                count_leaf(
                    n,
                    w.as_ref().expect("in-frame 4-way strip carries a winner"),
                );
            }
        }
        SbTree::Vert4(subs) => {
            for (i, w) in subs.iter().enumerate() {
                if i > 0 && mi_col + (i as i32) * quarter >= mi_cols {
                    break;
                }
                count_leaf(
                    n,
                    w.as_ref().expect("in-frame 4-way strip carries a winner"),
                );
            }
        }
        // AB partitions are interior-only (`SbTree::HorzA` docs): all three
        // sub-blocks are always coded, no frame-bound gating.
        SbTree::HorzA(subs) | SbTree::HorzB(subs) | SbTree::VertA(subs) | SbTree::VertB(subs) => {
            subs.iter().for_each(|w| count_leaf(n, w))
        }
    }
}

/// Wrap a payload in an OBU header + leb128 size.
fn wrap_obu(obu_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = write_obu_header(obu_type, false, obu_type != OBU_SEQUENCE_HEADER, 0);
    let size = uleb_encode(payload.len() as u64, 8).expect("OBU payload size fits a leb128 varint");
    out.extend_from_slice(&size);
    out.extend_from_slice(payload);
    out
}

/// The `OBU_TEMPORAL_DELIMITER` every temporal unit starts with — a zero-length
/// payload, so header byte + a single `0x00` leb128 size.
pub fn temporal_delimiter_obu() -> Vec<u8> {
    wrap_obu(OBU_TEMPORAL_DELIMITER, &[])
}

/// The sequence-header OBU (header byte + leb128 size + the written payload).
pub fn sequence_header_obu(seq: &SequenceHeaderObu) -> Vec<u8> {
    let mut wb = WriteBitBuffer::new();
    write_sequence_header_obu(&mut wb, seq);
    wrap_obu(OBU_SEQUENCE_HEADER, wb.bytes())
}

/// Encode ONE self-contained AV1 KEY frame: `TD OBU ++ sequence-header OBU ++
/// OBU_FRAME`, decodable by any conformant AV1 decoder, with **no C bootstrap
/// in the path**. See the module docs for what is derived and what is refused.
pub fn encode_key_frame(
    planes: KeyFramePlanes<'_>,
    cfg: &KeyFrameConfig,
) -> Result<Vec<u8>, KeyFrameError> {
    // ---- envelope + input validation -------------------------------------
    if cfg.usage != 2 {
        return Err(KeyFrameError::Unsupported(
            "usage: only AOM_USAGE_ALL_INTRA (2) is gated",
        ));
    }
    if !(0..=9).contains(&cfg.cpu_used) {
        return Err(KeyFrameError::Unsupported("cpu_used: must be 0..=9"));
    }
    if !matches!(cfg.bit_depth, 8 | 10 | 12) {
        return Err(KeyFrameError::Unsupported("bit_depth: must be 8, 10 or 12"));
    }
    if cfg.width == 0 || cfg.height == 0 {
        return Err(KeyFrameError::Unsupported("width/height: must be non-zero"));
    }
    if !(0..=63).contains(&cfg.cq_level) {
        return Err(KeyFrameError::Unsupported("cq_level: must be 0..=63"));
    }
    if cfg.monochrome && (cfg.ss_x, cfg.ss_y) != (1, 1) {
        return Err(KeyFrameError::Unsupported(
            "monochrome: ss must be (1, 1) (the AOM_IMG_FMT_I420 a mono image allocates)",
        ));
    }
    // AV1 has three chroma formats; (0, 1) is not one of them
    // (`aom_img_fmt_t`: I420 = (1,1), I422 = (1,0), I444 = (0,0)).
    if !matches!((cfg.ss_x, cfg.ss_y), (1, 1) | (1, 0) | (0, 0)) {
        return Err(KeyFrameError::Unsupported(
            "ss_x/ss_y: must be (1,1) 4:2:0, (1,0) 4:2:2 or (0,0) 4:4:4",
        ));
    }
    let (w, h) = (cfg.width, cfg.height);
    let (cw, ch) = cfg.chroma_dims();
    if planes.y.len() != w * h {
        return Err(KeyFrameError::PlaneSize {
            plane: 0,
            expected: w * h,
            got: planes.y.len(),
        });
    }
    if !cfg.monochrome {
        if planes.u.len() != cw * ch {
            return Err(KeyFrameError::PlaneSize {
                plane: 1,
                expected: cw * ch,
                got: planes.u.len(),
            });
        }
        if planes.v.len() != cw * ch {
            return Err(KeyFrameError::PlaneSize {
                plane: 2,
                expected: cw * ch,
                got: planes.v.len(),
            });
        }
    }

    // ---- headers ---------------------------------------------------------
    let seq = derive_sequence_header(cfg);
    let mib_size_log2 = if cfg.sb_size_128 { 5u32 } else { 4u32 }; // SB128 / SB64
    let mi_cols = mi_dim(w as i32);
    let mi_rows = mi_dim(h as i32);
    let tile_info = derive_tile_info(
        mi_cols,
        mi_rows,
        mib_size_log2,
        cfg.tile_columns_log2,
        cfg.tile_rows_log2,
    );
    let tiles_log2 = tile_info.log2_cols + tile_info.log2_rows;
    let n_tile_rows = tile_info.rows;
    let n_tile_cols = tile_info.cols;
    if n_tile_rows * n_tile_cols != 1usize << tiles_log2 {
        // `av1_calculate_tile_cols`/`_rows` derive `cols`/`rows` as loop counts;
        // for a uniform-spacing grid they always equal `1 << log2`. A stream
        // where they do not is one this shell has never seen.
        return Err(KeyFrameError::Unsupported(
            "tile grid: uniform spacing must give rows*cols == 2^(log2_cols+log2_rows)",
        ));
    }

    // ---- source planes: SB-aligned, border-extended (the harness recipe) --
    let bd = cfg.bit_depth;
    let sb_mi = if cfg.sb_size_128 { SB_MI_128 } else { SB_MI_64 };
    let sb_px = (sb_mi * 4) as usize;
    let n_sb_x = ((mi_cols + sb_mi - 1) / sb_mi).max(1);
    let n_sb_y = ((mi_rows + sb_mi - 1) / sb_mi).max(1);
    let sb_px_w = n_sb_x as usize * sb_px;
    let sb_px_h = n_sb_y as usize * sb_px;
    let stride = 320.max(sb_px_w + 4);
    let buf_h = (sb_px_h + 4).max(h + 4);
    let extend_plane = |dst: &mut [u16], pw: usize, ph: usize| {
        for r in 0..ph {
            let edge = dst[r * stride + pw - 1];
            for col in pw..stride {
                dst[r * stride + col] = edge;
            }
        }
        for r in ph..buf_h {
            dst.copy_within((ph - 1) * stride..ph * stride, r * stride);
        }
    };
    let mut src_y = vec![0u16; stride * buf_h];
    for r in 0..h {
        src_y[r * stride..r * stride + w].copy_from_slice(&planes.y[r * w..r * w + w]);
    }
    extend_plane(&mut src_y, w, h);
    let mut src_u = vec![0u16; stride * buf_h];
    let mut src_v = vec![0u16; stride * buf_h];
    if !cfg.monochrome {
        for r in 0..ch {
            src_u[r * stride..r * stride + cw].copy_from_slice(&planes.u[r * cw..r * cw + cw]);
            src_v[r * stride..r * stride + cw].copy_from_slice(&planes.v[r * cw..r * cw + cw]);
        }
        extend_plane(&mut src_u, cw, ch);
        extend_plane(&mut src_v, cw, ch);
    }

    // ---- screen-content decision (av1_set_screen_content_options) ---------
    // `force_screen_content_tools == SELECT` and no `--tune-content=screen`, so
    // the decision is the detector's — EXCEPT under
    // `rt_sf.use_nonrd_pick_mode && !rt_sf.hybrid_intra_pickmode` (allintra
    // speed 9), where C skips both estimators and sets the flags to 0
    // (encoder.c:2466-2470, KB-41 root #13). The detector needs
    // `screen_detection_mode2_fast_detection` (allintra speed >= 3), so the
    // speed features are built first — with `allow_screen_content_tools = false`
    // for this probe, since none of the three fields read here depends on it.
    //
    // `width`/`height` are C's `unfiltered_source->y_width`/`y_height` — the
    // 8-ALIGNED dimensions, not the crop (see the function's docs; passing the
    // crop mis-decides borderline frames).
    let speed = cfg.cpu_used;
    let sf_probe = SpeedFeatures::set_allintra(speed, false, bd > 8);
    let sct = if sf_probe.use_nonrd_pick_mode && sf_probe.hybrid_intra_pickmode == 0 {
        ScreenContentDecision::detection_disabled()
    } else {
        crate::screen_detect::estimate_screen_content_antialiasing_aware(
            &src_y,
            0,
            stride,
            (w + 7) & !7,
            (h + 7) & !7,
            bd,
            sf_probe.screen_detection_mode2_fast_detection,
        )
    };

    let mut p = derive_frame_header(cfg, &seq, &sct, tile_info);
    let qindex = p.quant.base_qindex;
    let coded_lossless = p.coded_lossless;

    // ---- quantizer + cost tables -----------------------------------------
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(
        bd,
        p.quant.y_dc_delta_q,
        p.quant.u_dc_delta_q,
        p.quant.u_ac_delta_q,
        p.quant.v_dc_delta_q,
        p.quant.v_ac_delta_q,
        &mut quants,
        &mut deq,
        0,
    );
    let rows_y = set_q_index(&quants, &deq, qindex as usize, 0);
    let rows_u = set_q_index(&quants, &deq, qindex as usize, 1);
    let rows_v = set_q_index(&quants, &deq, qindex as usize, 2);

    let enable_filter_intra = seq.seq_header.enable_filter_intra;
    let real = derive_real_costs(
        &KfFrameContext::default_for_qindex(qindex),
        enable_filter_intra,
        None,
    );
    let rdmult = av1_compute_rd_mult_based_on_qindex(
        bd,
        FrameUpdateType::Kf,
        qindex,
        TuneMetric::Psnr,
        EncMode::Allintra,
    );

    // ---- speed features ---------------------------------------------------
    let mut sf = SpeedFeatures::set_allintra(speed, sct.allow_screen_content_tools, bd > 8);
    // The modelled arms of `set_allintra_speed_feature_framesize_dependent`
    // (speed_features.c:166) and the ALLINTRA-reachable arms of
    // `av1_set_speed_features_qindex_dependent` (:2872) — C's second and third
    // passes, both framesize-blind in the `set_allintra` setter itself.
    sf.apply_allintra_framesize_dependent(w, h, speed);
    sf.apply_allintra_qindex_dependent(w, h, qindex, speed);
    // `prune_tx_type_using_stats`: ALLINTRA sets 1 at speed >= 2 and 2 at
    // speed >= 4, but ONLY `is_480p_or_larger` (speed_features.c:261/299).
    sf.prune_tx_type_using_stats = if w.min(h) >= 480 {
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

    let sb_block = if cfg.sb_size_128 {
        SB_BLOCK_128
    } else {
        SB_BLOCK_64
    };
    let mut env = SbEncodeEnv {
        ref_frame: None,
        sb_size: sb_block,
        mi_rows,
        mi_cols,
        // `cm->width` / `cm->height` — the TRUE crop (KB-28).
        frame_width: w as i32,
        frame_height: h as i32,
        // Placeholders — the real per-tile bounds are stamped in before every
        // `pack_tile` / `pack_tile_from_trees_lr` call below. They MATTER:
        // C's `av1_tile_set_row` / `_col` clamp the ends with
        // `AOMMIN(.., mi_rows/mi_cols)`, and a past-the-end sentinel instead of
        // that clamp changes the search's frame-edge decisions.
        // MEASURED 2026-09-02 by reverting to `1 << 16`: 131x131, 132x64,
        // 132x128, 132x132, 196x64, 196x196, 260x260 and 261x261 (textured
        // 4:2:0 cq32 speed 0) ALL diverge from real aomenc with the sentinel
        // and are ALL byte-identical with the clamp. Four of them had been
        // pinned in the gate as "RD near-ties"; they were this.
        tile_row_start: 0,
        tile_col_start: 0,
        tile_row_end: mi_rows,
        tile_col_end: mi_cols,
        monochrome: cfg.monochrome,
        ss_x: cfg.ss_x,
        ss_y: cfg.ss_y,
        bd,
        lossless: coded_lossless,
        reduced_tx_set_used: p.reduced_tx_set_used,
        disable_edge_filter: !seq.seq_header.enable_intra_edge_filter,
        filter_type: 0,
        stride,
        src_y: &src_y,
        src_u: &src_u,
        src_v: &src_v,
        base_y: 0,
        base_uv: 0,
        rows_y: &rows_y,
        rows_u: &rows_u,
        rows_v: &rows_v,
        rdmult,
        sharpness: 0,
        enable_optimize_b: if coded_lossless {
            TrellisOptType::NoTrellisOpt
        } else {
            TrellisOptType::FullTrellisOpt
        },
        use_chroma_trellis_rd_mult: true,
        coeff_costs_y: &real.coeff_costs_y,
        coeff_costs_uv: &real.coeff_costs_uv,
        txfm_partition_costs: [[0i32; 2]; 21],
        tx_type_costs: &real.tx_type_costs_y,
        qm_levels: None,
        tune: Default::default(),
        deltaq: None,
    };
    let pol = sf.tx_type_search_policy(false, 0); // (skip_trellis, sharpness)
    let pick_cfg = PickFrameCfg {
        // KB-32: carry the RESOLVED frame-level variance-partition values down
        // rather than letting the walk re-derive them from mi-ALIGNED dims.
        // Inert below speed 7 (the VBP path) and below speed 9 (`is_4k_or_larger`).
        fs_sf: crate::partition_pick::FrameSizeSf {
            vbp: crate::var_part::VbpSf {
                force_large_partition_blocks_intra: sf.force_large_partition_blocks_intra != 0,
                var_part_split_threshold_shift: sf.var_part_split_threshold_shift,
                allintra: true,
            },
            // `is_4k_or_larger` = `AOMMIN(cm->width, cm->height) >= 2160`.
            is_4k_or_larger: w.min(h) >= 2160,
        },
        inter: None,
        intrabc: None,
        search_allow_intrabc: false,
        intra_tools: Default::default(),
        mode_costs: &real.mode_costs,
        tx_size_costs: &real.tx_size_costs,
        skip_costs: &real.skip_costs,
        tx_type_costs_y: &real.tx_type_costs_y,
        pol: &pol,
        uv_lp: &UvLoopPolicy::speed0_allintra(),
        intra_uv_mode_cost: &real.mode_costs.intra_uv_mode_cost,
        cfl_costs: &real.cfl_costs,
        partition_costs: &real.partition_costs,
        partition_cdfs: &real.partition_cdf,
        allintra: true,
        speed,
        qindex,
        enable_filter_intra,
        enable_tx64: true,
        enable_rect_tx: true,
        intra_pruning_with_hog: sf.intra_pruning_with_hog != 0,
        enable_rect_partitions: true,
        // `av1_set_speed_features_qindex_dependent` runs AFTER the allintra
        // setters and overrides at speed 3 ONLY (speed_features.c:3032-3034);
        // its speed <= 2 and speed >= 4 arms equal the allintra values.
        less_rectangular_check_level: if speed == 3 {
            if qindex >= 170 { 1 } else { 2 }
        } else {
            sf.less_rectangular_check_level
        },
        // `set_max_min_partition_size` (partition_strategy.h:214/224) with the
        // default `--min-partition-size 4` / `--max-partition-size 128`.
        max_partition_size: sf.default_max_partition_size.min(sb_block),
        min_partition_size: sf.default_min_partition_size.min(sb_block),
        enable_1to4_partitions: true,
        enable_ab_partitions: true,
        allow_screen_content_tools: sct.allow_screen_content_tools,
        qm_levels: None,
        // `--enable-palette=0` (the shim's config): no palette search.
        palette_costs: None,
    };

    // ---- phase 1: search + encode (bits discarded) ------------------------
    // C's `encode_frame_internal`: adapt a tile context, produce the
    // reconstruction, count `txb_split_count`. The bits go to a throwaway
    // coder; only the trees, the recon and the split count are kept.
    // KB-41 root #12: `update_stats`' tx-size gate is the SEARCH-time
    // DEFAULT_EVAL tx mode (rdopt_utils.h:494), not the final header one. The
    // nonrd speeds still search SELECT.
    let search_tx_mode_is_select = !coded_lossless;
    let phase1_pack_cfg = PackCfg {
        enable_filter_intra,
        tx_mode_is_select: search_tx_mode_is_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
        delta_q_present: false,
        delta_q_res: 0,
        allow_screen_content_tools: sct.allow_screen_content_tools,
        allow_intrabc: false,
        search_allow_intrabc: false,
        search_tx_mode_is_select,
    };
    // Tile geometry in raster (tile-row-major) order:
    // `(mi_row_start, mi_col_start, mi_row_end, mi_col_end, n_sb_rows, n_sb_cols)`.
    // The mi ENDS are clamped to the frame exactly like C's `av1_tile_set_row` /
    // `_col` (`tile->mi_row_end = AOMMIN(.., mi_rows)`, tile_common.c). A single
    // tile is one entry covering the frame.
    let tile_grid: Vec<(i32, i32, i32, i32, i32, i32)> = (0..n_tile_rows)
        .flat_map(|trow| {
            let ti = &p.tile_info;
            (0..n_tile_cols).map(move |tcol| {
                let r0 = ti.row_start_sb[trow] << mib_size_log2;
                let r1 = (ti.row_start_sb[trow + 1] << mib_size_log2).min(mi_rows);
                let c0 = ti.col_start_sb[tcol] << mib_size_log2;
                let c1 = (ti.col_start_sb[tcol + 1] << mib_size_log2).min(mi_cols);
                (
                    r0,
                    c0,
                    r1,
                    c1,
                    ti.row_start_sb[trow + 1] - ti.row_start_sb[trow],
                    ti.col_start_sb[tcol + 1] - ti.col_start_sb[tcol],
                )
            })
        })
        .collect();
    debug_assert_eq!(
        tile_grid
            .iter()
            .map(|t| (t.4 * t.5) as usize)
            .sum::<usize>(),
        (n_sb_x * n_sb_y) as usize,
        "the tile grid must partition every superblock exactly once"
    );

    let mut recon_y = src_y.clone();
    let mut recon_u = src_u.clone();
    let mut recon_v = src_v.clone();
    // Frame-raster tree slots, filled tile by tile. C's `write_modes` resets the
    // tile context per tile (`av1_reset_loop_restoration` + a fresh copy of
    // `cm->fc`), which is why each tile gets its own `KfFrameContext`.
    let mut frame_trees: Vec<Option<SbTree>> = (0..(n_sb_x * n_sb_y)).map(|_| None).collect();
    for &(r0, c0, r1, c1, n_tr, n_tc) in &tile_grid {
        env.tile_row_start = r0;
        env.tile_col_start = c0;
        env.tile_row_end = r1;
        env.tile_col_end = c1;
        let mut kf_tile = KfFrameContext::default_for_qindex(qindex);
        let mut scratch = OdEcEnc::new();
        let t = pack_tile(
            &mut scratch,
            &env,
            &pick_cfg,
            &phase1_pack_cfg,
            &mut kf_tile,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            r0,
            c0,
            n_tr,
            n_tc,
            sb_mi,
            sb_block,
        );
        let _ = scratch.done();
        let (sb_r0, sb_c0) = (r0 / sb_mi, c0 / sb_mi);
        for (i, tree) in t.into_iter().enumerate() {
            let sb_r = sb_r0 + i as i32 / n_tc;
            let sb_c = sb_c0 + i as i32 % n_tc;
            frame_trees[(sb_r * n_sb_x + sb_c) as usize] = Some(tree);
        }
    }
    let trees: Vec<SbTree> = frame_trees
        .into_iter()
        .map(|t| t.expect("every superblock belongs to exactly one tile"))
        .collect();

    // `av1_encode_frame` (encodeframe.c:2796-2799): the frame codes
    // TX_MODE_LARGEST when the search ran at TX_MODE_SELECT and NO block split
    // its transform. Counted off this port's own winner trees.
    let splits = txb_split_count(&trees, mi_rows, mi_cols, n_sb_x, sb_mi, sb_block);
    p.tx_mode_select = search_tx_mode_is_select && splits > 0;

    // ---- loop-filter level: derived from THIS port's reconstruction -------
    let mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb_x, sb_mi, sb_block);
    let lf_frame = LfSearchFrame {
        recon_y: &recon_y,
        recon_u: &recon_u,
        recon_v: &recon_v,
        src_y: &src_y,
        src_u: &src_u,
        src_v: &src_v,
        stride,
        crop_width: w as u32,
        crop_height: h as u32,
        ss_x: cfg.ss_x,
        ss_y: cfg.ss_y,
        bd: i32::from(bd),
        monochrome: cfg.monochrome,
        mi: &mi_grid,
        mi_rows,
        mi_cols,
        delta_lf_present: false,
    };
    // `lpf_sf.lpf_pick`: LPF_PICK_FROM_FULL_IMAGE (DUAL) at allintra speed
    // 0..=3, ..._NON_DUAL at 4/5 (speed_features.c:496), and the CLOSED-FORM
    // LPF_PICK_FROM_Q at speed >= 6 (:559) — no search at all, the level is a
    // fit on the AC quantizer. `loopfilter_frame` (encoder.c:2875-2886) runs
    // `av1_pick_filter_level` only when `is_loopfilter_used(cm)` =
    // `!coded_lossless && !large_scale`, so a coded-lossless frame keeps
    // `cm->lf`'s zeroed levels (byte-inert — the header writer skips the whole
    // loop-filter block — but running a search C never runs would be a lie
    // about what this models). The third argument is SHARPNESS
    // (`--sharpness`, 0 here), not the speed.
    let derived_lf = if coded_lossless {
        LoopFilterLevels {
            filter_level: [0, 0],
            filter_level_u: 0,
            filter_level_v: 0,
            sharpness: 0,
        }
    } else if speed >= 6 {
        pick_filter_level_from_q(qindex, bd, true, 0)
    } else {
        pick_filter_level(&lf_frame, true, 0, speed >= 4)
    };
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;

    // ---- post-filter stages: deblock -> CDEF -> loop restoration ----------
    // C's order (`encoder.c` `loopfilter_frame` -> `cdef_restoration_frame`):
    // apply the picked deblock levels, then `av1_cdef_search` + `av1_cdef_frame`
    // on the deblocked recon, then `av1_pick_filter_restoration` on the
    // POST-CDEF recon. `allow_intrabc` disables the whole block
    // (`encoder.c:3780` wraps all of `loopfilter_frame`; KB-41 root #14 —
    // without that gate the LR repack writes units the header never announces
    // and both decoders reject the stream).
    let postfilter = !p.allow_intrabc && (cfg.enable_cdef || seq.seq_header.enable_restoration);
    // The deblocked reconstruction, kept separate from the phase-1 recon so the
    // LR search can see BOTH `deblocked` and the post-CDEF `cur` (C saves
    // boundary lines from each: `av1_loop_restoration_save_boundary_lines`
    // calls 0 and 1).
    let mut deblocked_y = Vec::new();
    let mut deblocked_u = Vec::new();
    let mut deblocked_v = Vec::new();
    if postfilter {
        deblocked_y = recon_y.clone();
        deblocked_u = recon_u.clone();
        deblocked_v = recon_v.clone();
        // `loop_filter_frame` no-ops per plane on a zero level, exactly like
        // C's apply site (`encoder.c:2887`).
        if derived_lf.filter_level[0] != 0 || derived_lf.filter_level[1] != 0 {
            let params = LfParams {
                filter_level: derived_lf.filter_level,
                filter_level_u: derived_lf.filter_level_u,
                filter_level_v: derived_lf.filter_level_v,
                sharpness: derived_lf.sharpness,
                mode_ref_delta_enabled: true,
                ref_deltas: KF_REF_DELTAS,
                mode_deltas: KF_MODE_DELTAS,
                delta_lf_present: false,
                delta_lf_multi: false,
                lossless: [false; 8],
                seg: Default::default(),
            };
            let grid = LfMiGrid {
                mi: &mi_grid,
                stride: mi_cols as usize,
                mi_rows,
                mi_cols,
            };
            let mut buf = LfFrameBuf {
                y: &mut deblocked_y,
                y_stride: stride,
                u: &mut deblocked_u,
                v: &mut deblocked_v,
                uv_stride: stride,
                crop_width: w as u32,
                crop_height: h as u32,
                ss_x: cfg.ss_x,
                ss_y: cfg.ss_y,
                bd: i32::from(bd),
            };
            loop_filter_frame(&mut buf, &grid, &params, 0, cfg.num_planes());
        }
    }

    // ---- CDEF: search on the deblocked recon, then APPLY it ---------------
    let mut cur_y = Vec::new();
    let mut cur_u = Vec::new();
    let mut cur_v = Vec::new();
    let cdef_pack = if postfilter && cfg.enable_cdef {
        let cdef_res = av1_cdef_search(
            &CdefSearchFrame {
                recon_y: &deblocked_y,
                recon_u: &deblocked_u,
                recon_v: &deblocked_v,
                src_y: &src_y,
                src_u: &src_u,
                src_v: &src_v,
                stride,
                mi: &mi_grid,
                mi_rows,
                mi_cols,
                ss_x: cfg.ss_x,
                ss_y: cfg.ss_y,
                monochrome: cfg.monochrome,
                bd,
                base_qindex: qindex,
                rdmult,
            },
            sf.cdef_pick_method,
        );
        p.cdef.cdef_damping = cdef_res.cdef_damping;
        p.cdef.cdef_bits = cdef_res.cdef_bits;
        p.cdef.nb_cdef_strengths = cdef_res.nb_cdef_strengths;
        p.cdef.cdef_strengths = cdef_res.cdef_strengths;
        p.cdef.cdef_uv_strengths = cdef_res.cdef_uv_strengths;
        // `av1_cdef_frame` — only needed as the LR search's input; when
        // restoration is off nothing reads the filtered pixels (phase 2
        // re-encodes from the source), so the apply is skipped.
        if seq.seq_header.enable_restoration {
            cur_y = deblocked_y.clone();
            cur_u = deblocked_u.clone();
            cur_v = deblocked_v.clone();
            let skip: Vec<bool> = mi_grid.iter().map(|m| m.skip_txfm).collect();
            cdef_frame(
                &mut cur_y,
                stride,
                &mut cur_u,
                &mut cur_v,
                stride,
                &CdefFrameParams {
                    mi_rows,
                    mi_cols,
                    num_planes: cfg.num_planes(),
                    ss_x: cfg.ss_x,
                    ss_y: cfg.ss_y,
                    bit_depth: i32::from(bd),
                    damping: cdef_res.cdef_damping,
                    cdef_strengths: cdef_res.cdef_strengths,
                    cdef_uv_strengths: cdef_res.cdef_uv_strengths,
                    skip_txfm: &skip,
                    unit_strength: &cdef_res.unit_strength,
                },
            );
        }
        Some(CdefPackState {
            cdef_bits: cdef_res.cdef_bits as u32,
            unit_strength: cdef_res.unit_strength.clone(),
            nhfb: cdef_res.nhfb,
        })
    } else {
        None
    };

    // ---- loop restoration: `av1_pick_filter_restoration` ------------------
    // `is_restoration_used` (`encoder.h:4431`) = `enable_restoration &&
    // !all_lossless && !large_scale`, plus the `allow_intrabc` gate folded into
    // `postfilter` above.
    // `is_restoration_used` reads the SEQUENCE bit, which speed >= 5 clears
    // (see `derive_sequence_header`) — not the raw `--enable-restoration` knob.
    let lr_stage = postfilter && seq.seq_header.enable_restoration && !coded_lossless;
    let lr_outcome = if lr_stage {
        // With CDEF off the post-CDEF frame IS the deblocked frame.
        let (lr_cur_y, lr_cur_u, lr_cur_v) = if cfg.enable_cdef {
            (&cur_y, &cur_u, &cur_v)
        } else {
            (&deblocked_y, &deblocked_u, &deblocked_v)
        };
        // Costs come from the FRAME-INIT LR CDFs (nothing adapts them before
        // the search in C); rdmult is the frame rdmult.
        let fc0 = KfFrameContext::default_for_qindex(qindex);
        let mut wiener_cost = [0i32; 2];
        let mut sgrproj_cost = [0i32; 2];
        let mut switchable_cost = [0i32; 3];
        cost_tokens_from_cdf(&mut wiener_cost, &fc0.wiener_restore, None);
        cost_tokens_from_cdf(&mut sgrproj_cost, &fc0.sgrproj_restore, None);
        cost_tokens_from_cdf(&mut switchable_cost, &fc0.switchable_restore, None);
        let planes = if cfg.monochrome {
            vec![LrPlanePixels {
                src: &src_y,
                deblocked: &deblocked_y,
                cur: lr_cur_y,
                stride,
            }]
        } else {
            vec![
                LrPlanePixels {
                    src: &src_y,
                    deblocked: &deblocked_y,
                    cur: lr_cur_y,
                    stride,
                },
                LrPlanePixels {
                    src: &src_u,
                    deblocked: &deblocked_u,
                    cur: lr_cur_u,
                    stride,
                },
                LrPlanePixels {
                    src: &src_v,
                    deblocked: &deblocked_v,
                    cur: lr_cur_v,
                    stride,
                },
            ]
        };
        let outcome = pick_filter_restoration(&LrSearchInput {
            planes,
            crop_width: w as i32,
            crop_height: h as i32,
            ss_x: cfg.ss_x,
            ss_y: cfg.ss_y,
            bit_depth: i32::from(bd),
            highbd: bd > 8,
            rdmult: i64::from(rdmult),
            dc_quant_qtx: i32::from(av1_dc_quant_qtx(qindex, 0, bd)),
            mib_size_log2: mib_size_log2 as i32,
            mi_rows,
            mi_cols,
            // `av1_pick_filter_restoration` walks tiles outer / SBs inner and
            // resets the per-RU delta-coding references at every tile start.
            // Single tile (asserted above) => one span per axis.
            // `av1_pick_filter_restoration` walks tiles outer / SBs inner and
            // resets the per-RU delta-coding references at every tile start, so
            // the spans must be the REAL ones.
            tile_sb_rows: (0..n_tile_rows)
                .map(|t| (p.tile_info.row_start_sb[t], p.tile_info.row_start_sb[t + 1]))
                .collect(),
            tile_sb_cols: (0..n_tile_cols)
                .map(|t| (p.tile_info.col_start_sb[t], p.tile_info.col_start_sb[t + 1]))
                .collect(),
            wiener_restore_cost: wiener_cost,
            sgrproj_restore_cost: sgrproj_cost,
            switchable_restore_cost: switchable_cost,
            sf: crate::speed_features::lr_search_sf_allintra(
                speed,
                qindex,
                w,
                h,
                sct.allow_screen_content_tools,
            ),
        });
        p.restoration.frame_restoration_type = outcome.frame_restoration_type;
        p.restoration.restoration_unit_size = [outcome.unit_size; 3];
        Some(outcome)
    } else {
        None
    };

    // ---- phase 2: the real pack over the already-picked trees -------------
    // `av1_pack_bitstream` seeds a SECOND fresh tile context from `cm->fc` and
    // re-writes every symbol, now with the FINAL `tx_mode`. `search_tx_mode_is
    // _select` keeps the tx-size CDF adapting exactly as the search's did even
    // when the flip removed the coded symbol (KB-42).
    let pack_cfg = PackCfg {
        tx_mode_is_select: p.tx_mode_select,
        ..phase1_pack_cfg
    };
    let mut recon2_y = src_y.clone();
    let mut recon2_u = src_u.clone();
    let mut recon2_v = src_v.clone();
    let lr_restores = lr_outcome.as_ref().is_some_and(|o| {
        o.frame_restoration_type
            .iter()
            .any(|&t| t != LR_RESTORE_NONE)
    });
    // An all-NONE restoration outcome codes no LR symbols at all, so it takes
    // the same path as restoration-off.
    let lr_pack = lr_restores.then(|| {
        let outcome = lr_outcome.as_ref().expect("lr_restores implies an outcome");
        LrPackParams {
            cfg: LrFrameConfig {
                frame_restoration_type: outcome.frame_restoration_type,
                unit_size: [outcome.unit_size; 3],
                crop_width: w as i32,
                crop_height: h as i32,
                superres_denom: 0,
            },
            units: [&outcome.units[0], &outcome.units[1], &outcome.units[2]],
            num_planes: cfg.num_planes(),
        }
    });
    let mut tile_payloads: Vec<Vec<u8>> = Vec::with_capacity(tile_grid.len());
    for &(r0, c0, r1, c1, n_tr, n_tc) in &tile_grid {
        env.tile_row_start = r0;
        env.tile_col_start = c0;
        env.tile_row_end = r1;
        env.tile_col_end = c1;
        // This tile's slice of the frame-raster trees, in the tile-local raster
        // order `pack_tile_from_trees_lr` indexes by.
        let (sb_r0, sb_c0) = (r0 / sb_mi, c0 / sb_mi);
        let mut tile_trees: Vec<SbTree> = (0..n_tr)
            .flat_map(|r| (0..n_tc).map(move |c| (r, c)))
            .map(|(r, c)| trees[((sb_r0 + r) * n_sb_x + sb_c0 + c) as usize].clone())
            .collect();
        let mut kf_tile = KfFrameContext::default_for_qindex(qindex);
        let mut enc = OdEcEnc::new();
        pack_tile_from_trees_lr(
            &mut enc,
            &env,
            &pick_cfg,
            &pack_cfg,
            &mut kf_tile,
            &mut recon2_y,
            &mut recon2_u,
            &mut recon2_v,
            &mut tile_trees,
            r0,
            c0,
            n_tr,
            n_tc,
            sb_mi,
            sb_block,
            cdef_pack.clone(),
            lr_pack.as_ref(),
        );
        tile_payloads.push(enc.done().to_vec());
    }

    // ---- temporal unit ----------------------------------------------------
    let frame_obu = if tile_payloads.len() == 1 {
        assemble_obu_frame_single_tile(&p, tiles_log2, &tile_payloads[0], false, 0)
    } else {
        // Multi-tile, `num_tg == 1`: one `OBU_FRAME` carrying all tiles in a
        // single tile group, with `context_update_tile_id` and
        // `tile_size_bytes_minus_1` overwritten from the real tile sizes
        // (`write_tile_obu_size`, bitstream.c:4053/4068) — the derived form, no
        // header bytes spliced from anywhere.
        wrap_obu(
            OBU_FRAME,
            &assemble_multitile_frame_obu_payload_derived(&p, &tile_payloads),
        )
    };
    let td = temporal_delimiter_obu();
    let seq_obu = sequence_header_obu(&seq);
    let mut out = Vec::with_capacity(td.len() + seq_obu.len() + frame_obu.len());
    out.extend_from_slice(&td);
    out.extend_from_slice(&seq_obu);
    out.extend_from_slice(&frame_obu);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_bits_matches_init_seq_coding_tools() {
        // seq->num_bits_width = (max > 1) ? get_msb(max - 1) + 1 : 1.
        assert_eq!(num_bits_for_dim(1), 1);
        assert_eq!(num_bits_for_dim(2), 1);
        assert_eq!(num_bits_for_dim(64), 6);
        assert_eq!(num_bits_for_dim(65), 7);
        assert_eq!(num_bits_for_dim(128), 7);
        assert_eq!(num_bits_for_dim(96), 7);
        assert_eq!(num_bits_for_dim(1920), 11);
    }

    #[test]
    fn profile_rules() {
        let mk = |bd, mono, ss_x, ss_y| {
            KeyFrameConfig::allintra_speed0(64, 64, bd, mono, ss_x, ss_y, 32).profile()
        };
        assert_eq!(mk(8, true, 1, 1), 0);
        assert_eq!(mk(8, false, 1, 1), 0);
        assert_eq!(mk(8, false, 0, 0), 1);
        assert_eq!(mk(10, false, 0, 0), 1);
        assert_eq!(mk(8, false, 1, 0), 2); // 4:2:2
        assert_eq!(mk(12, false, 1, 1), 2);
    }

    #[test]
    fn single_tile_grid_for_small_frames() {
        let t = derive_tile_info(mi_dim(64), mi_dim(64), 4, 0, 0);
        assert_eq!((t.log2_cols, t.log2_rows), (0, 0));
        assert_eq!((t.cols, t.rows), (1, 1));
        assert_eq!(t.col_start_sb[1], 1);
        assert_eq!(t.row_start_sb[1], 1);
    }

    #[test]
    fn temporal_delimiter_is_the_two_canonical_bytes() {
        // obu_type 2 << 3 | has_size_field << 1 == 0x12, then a 0-length leb128.
        assert_eq!(temporal_delimiter_obu(), vec![0x12, 0x00]);
    }
}
