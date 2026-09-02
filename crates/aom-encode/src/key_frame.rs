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
//! | tile grid | `av1_get_tile_limits` + `av1_calculate_tile_cols/rows` (`tile_common.c`) |
//! | loop-filter levels | [`crate::lf_search::pick_filter_level`] on THIS port's own recon |
//! | `tx_mode` (SELECT vs LARGEST) | the `txb_split_count == 0` flip (`encodeframe.c:2797`), counted off this port's own winner trees |
//! | tile payload | [`crate::pack::pack_tile`] + [`crate::pack::pack_tile_from_trees`] |
//!
//! # Envelope (asserted, not assumed)
//!
//! [`encode_key_frame`] returns [`KeyFrameError`] rather than silently
//! mis-encoding whenever the config leaves the envelope this module is gated on:
//!
//! * ALL-INTRA usage, `--cpu-used 0`, a single KEY frame, `--end-usage=q`;
//! * CDEF off and loop-restoration off (the `encoder_gate_e2e_*` bootstrap
//!   boundary; the CDEF search itself IS ported — `crate::pickcdef` — and is a
//!   named follow-on, see "Not yet wired" below);
//! * superblock 64, a single tile, no superres, no QM, no film grain, no
//!   segmentation, no delta-q, palette + IntraBC search off (matching
//!   `aom-sys-ref`'s `shim_encode_av1_kf`, whose `--enable-palette=0
//!   --enable-intrabc=0` is what the byte gate compares against).
//!
//! # Not yet wired (named, with the specific entry points)
//!
//! * **CDEF on** — [`crate::pickcdef::av1_cdef_search`] + the
//!   `pack_tile_from_trees(.., cdef: Some(..))` arm are ported and byte-gated
//!   in `aom-bench/tests/encoder_gate_cdef_e2e.rs`; wiring them here is a
//!   `derive_frame_header` + phase-2 change, not new algorithm.
//! * **Loop restoration on** — `aom_dsp::restore::pick` + `pack_tile_lr`'s
//!   `LrPackParams`, gated by `aom-bench/tests/lr_restoration_gate.rs`.
//! * **Multi-tile** — [`crate::obu_assemble::assemble_multitile_frame_obu_payload_derived`]
//!   exists and is gated (`obu_assemble_multitile_diff.rs`); this shell refuses
//!   `tiles_log2 > 0` rather than emit an untested composition.
//! * **`av1_determine_sc_tools_with_encoding`** (`encoder_utils.c:1214`) — C's
//!   two-pass trial encode that can turn screen-content tools ON after the
//!   detector said off. Unported. It returns early when the detector already
//!   said on, so it only ever matters on detector-negative content; the byte
//!   gate holds this accountable per cell.
//! * **Speeds > 0, bit depths beyond what the gate sweeps, SB128** — the
//!   pipeline pieces exist; the shell simply has no gate for them yet.

use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::header::{
    CdefHeader, ColorConfigParams, DecoderModelInfo, DeltaQParams, FrameHeaderObu,
    FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader, QuantParamsHeader, RestorationHeader,
    SequenceHeaderObu, SequenceHeaderParams, TileInfoHeader, TimingInfoHeader,
    write_sequence_header_obu,
};
use aom_dsp::entropy::leb128::uleb_encode;
use aom_dsp::entropy::obu::write_obu_header;
use aom_dsp::entropy::partition::{KfFrameContext, tx_size_to_depth};
use aom_dsp::entropy::wb::WriteBitBuffer;
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};

use crate::encode_intra::TrellisOptType;
use crate::encode_sb::{LeafWinner, SbEncodeEnv, SbTree};
use crate::intra_uv_rd::UvLoopPolicy;
use crate::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use crate::obu_assemble::assemble_obu_frame_single_tile;
use crate::pack::{PackCfg, pack_tile, pack_tile_from_trees};
use crate::partition_pick::PickFrameCfg;
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
    /// `AOME_SET_CPUUSED`.
    pub cpu_used: i32,
    /// `aom_codec_enc_config_default`'s usage: 2 = `AOM_USAGE_ALL_INTRA`,
    /// 0 = `AOM_USAGE_GOOD_QUALITY`.
    pub usage: u32,
    /// `AV1E_SET_ENABLE_CDEF`.
    pub enable_cdef: bool,
    /// `AV1E_SET_ENABLE_RESTORATION`.
    pub enable_restoration: bool,
}

impl KeyFrameConfig {
    /// The `shim_encode_av1_kf` default envelope: ALL-INTRA, `--cpu-used 0`,
    /// CDEF and loop-restoration off.
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
    /// The derived tile grid needs more than one tile
    /// (`av1_get_tile_limits`' mandatory split above 4096px wide / ~9.4 MP);
    /// the multi-tile assembler exists but this shell has no gate for it.
    MultiTileRequired {
        /// `log2_cols + log2_rows` the tile limits force.
        tiles_log2: i32,
    },
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
            KeyFrameError::MultiTileRequired { tiles_log2 } => write!(
                f,
                "frame requires {} tiles (tiles_log2={tiles_log2}); the multi-tile \
                 assembler exists (obu_assemble::assemble_multitile_frame_obu_payload_derived) \
                 but this shell has no gate for it",
                1 << tiles_log2
            ),
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
    // av1_set_tile_info: log2_cols = clamp(cfg, min, max).
    t.log2_cols = tile_cols_log2_cfg.max(min_log2_cols).min(max_log2_cols);
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
            sb_size_128: false,
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
            enable_restoration: cfg.enable_restoration,
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
    if cfg.cpu_used != 0 {
        return Err(KeyFrameError::Unsupported(
            "cpu_used: only speed 0 is gated",
        ));
    }
    if cfg.enable_cdef {
        return Err(KeyFrameError::Unsupported(
            "enable_cdef: the CDEF search is ported (pickcdef) but not wired into this shell",
        ));
    }
    if cfg.enable_restoration {
        return Err(KeyFrameError::Unsupported(
            "enable_restoration: the LR search is ported but not wired into this shell",
        ));
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
    let mib_size_log2 = 4u32; // SB64
    let mi_cols = mi_dim(w as i32);
    let mi_rows = mi_dim(h as i32);
    let tile_info = derive_tile_info(mi_cols, mi_rows, mib_size_log2, 0, 0);
    let tiles_log2 = tile_info.log2_cols + tile_info.log2_rows;
    if tiles_log2 != 0 {
        return Err(KeyFrameError::MultiTileRequired { tiles_log2 });
    }

    // ---- source planes: SB-aligned, border-extended (the harness recipe) --
    let bd = cfg.bit_depth;
    let sb_mi = SB_MI_64;
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
    // force_screen_content_tools == SELECT and no --tune-content=screen, so the
    // decision is the anti-aliasing-aware detector's. speed 0 => the
    // `use_nonrd_pick_mode && !hybrid_intra_pickmode` skip arm is not taken and
    // `screen_detection_mode2_fast_detection` is false.
    let sct = crate::screen_detect::estimate_screen_content_antialiasing_aware(
        &src_y, 0, stride, w, h, bd, false,
    );

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
    let mut sf = SpeedFeatures::set_allintra(0, sct.allow_screen_content_tools, bd > 8);
    sf.apply_allintra_framesize_dependent(w, h, 0);
    sf.apply_allintra_qindex_dependent(w, h, qindex, 0);
    // `prune_tx_type_using_stats` is ALLINTRA speed >= 2 only, and only
    // `is_480p_or_larger` — 0 at speed 0 either way.
    sf.prune_tx_type_using_stats = 0;

    let sb_block = SB_BLOCK_64;
    let env = SbEncodeEnv {
        ref_frame: None,
        sb_size: sb_block,
        mi_rows,
        mi_cols,
        // `cm->width` / `cm->height` — the TRUE crop (KB-28).
        frame_width: w as i32,
        frame_height: h as i32,
        tile_row_start: 0,
        tile_col_start: 0,
        tile_row_end: 1 << 16,
        tile_col_end: 1 << 16,
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
    let pol = sf.tx_type_search_policy(false, 0);
    let pick_cfg = PickFrameCfg {
        fs_sf: Default::default(),
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
        speed: 0,
        qindex,
        enable_filter_intra,
        enable_tx64: true,
        enable_rect_tx: true,
        intra_pruning_with_hog: sf.intra_pruning_with_hog != 0,
        enable_rect_partitions: true,
        less_rectangular_check_level: sf.less_rectangular_check_level,
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
    let mut kf1 = KfFrameContext::default_for_qindex(qindex);
    let mut recon_y = src_y.clone();
    let mut recon_u = src_u.clone();
    let mut recon_v = src_v.clone();
    let mut scratch = OdEcEnc::new();
    let mut trees = pack_tile(
        &mut scratch,
        &env,
        &pick_cfg,
        &phase1_pack_cfg,
        &mut kf1,
        &mut recon_y,
        &mut recon_u,
        &mut recon_v,
        0,
        0,
        n_sb_y,
        n_sb_x,
        sb_mi,
        sb_block,
    );
    let _ = scratch.done();

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
    let derived_lf = pick_filter_level(&lf_frame, true, 0, false);
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;

    // ---- phase 2: the real pack over the already-picked trees -------------
    // `av1_pack_bitstream` seeds a SECOND fresh tile context from `cm->fc` and
    // re-writes every symbol, now with the FINAL `tx_mode`. `search_tx_mode_is
    // _select` keeps the tx-size CDF adapting exactly as the search's did even
    // when the flip removed the coded symbol (KB-42).
    let pack_cfg = PackCfg {
        tx_mode_is_select: p.tx_mode_select,
        ..phase1_pack_cfg
    };
    let mut kf2 = KfFrameContext::default_for_qindex(qindex);
    let mut recon2_y = src_y.clone();
    let mut recon2_u = src_u.clone();
    let mut recon2_v = src_v.clone();
    let mut enc = OdEcEnc::new();
    pack_tile_from_trees(
        &mut enc,
        &env,
        &pick_cfg,
        &pack_cfg,
        &mut kf2,
        &mut recon2_y,
        &mut recon2_u,
        &mut recon2_v,
        &mut trees,
        0,
        0,
        n_sb_y,
        n_sb_x,
        sb_mi,
        sb_block,
        None,
    );
    let tile_bytes = enc.done().to_vec();

    // ---- temporal unit ----------------------------------------------------
    let frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &tile_bytes, false, 0);
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
