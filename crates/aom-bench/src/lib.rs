//! Gate-3 performance harness: pair the pure-Rust port against the REAL
//! libaom C oracle **in-process** (via `aom-sys-ref`), on real conformance
//! content, with the port's output **byte-verified** against the C output
//! before any timing is trusted.
//!
//! What is measured on each side (honest accounting):
//!
//! * **Decode** — symmetric: both sides take the same temporal-unit bytes and
//!   produce full decoded planes. C = `aom_codec_av1_dx` init + decode +
//!   plane copy-out + destroy (the still-image usage pattern); port =
//!   [`aom_decode::frame::decode_frame_obus`] (parse + tile decode + all
//!   post-filters + plane crop-out).
//! * **Encode** — C = `aom_codec_av1_cx` init + full KEY encode + destroy
//!   (`shim_encode_av1_kf`, the aomenc path). Port = everything the port does
//!   to produce the identical frame OBU payload from the same source pixels:
//!   header-field bootstrap parse (microseconds), quantizer + cost-table
//!   derivation, source strided-copy + border extension, the full SB
//!   search+pack walk (`pack_tile`), loop-filter level search, and OBU
//!   assembly. CAVEAT (documented, small): the port does not yet self-derive
//!   a handful of frame-header FIELDS (qindex mapping, tile limits, …) — it
//!   parses them from a reference stream encoded ONCE in untimed setup. The
//!   parse it performs per iteration IS timed; the reference encode that
//!   produced those bytes is not part of the port's work. The port's timed
//!   region produces the byte-identical bitstream payload end-to-end.
//!
//! Every cell is validated by [`EncodeCell::assert_byte_exact`] /
//! [`DecodeCell::assert_byte_exact`] before benchmarking: a cell where the
//! port and C do not produce identical bytes would be a meaningless timing
//! comparison (and a correctness regression).

#![forbid(unsafe_code)]

pub mod config_perm;
// Pure-Rust (no C oracle) encode harness for the Windows allocation A/B.
pub mod winperf;
// Both of these are C-differential harnesses end to end (inter_localize
// consumes `aom_sys_ref::RefDecodedFrame`; rd_close scores port-vs-C
// reconstructions with zensim), so neither exists without the oracle.
#[cfg(feature = "c-oracle")]
pub mod inter_localize;
#[cfg(feature = "c-oracle")]
pub mod rd_close;

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::{SbEncodeEnv, SbTree};
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{
    LfSearchFrame, build_lf_mi_grid, pick_filter_level, pick_filter_level_from_q,
};
use aom_encode::obu_assemble::{
    assemble_frame_obu_payload_single_tile, assemble_multitile_frame_obu_payload_derived,
};
use aom_encode::pack::{LrPackParams, pack_tile, pack_tile_lr};
use aom_encode::partition_pick::{IntrabcFrameCfg, PickFrameCfg};
use aom_encode::rd::{
    EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex, av1_set_sad_per_bit,
};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::speed_features::SpeedFeatures;
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::header::{
    CdefHeader, FilmGrainParams, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader,
    LoopfilterHeader, RestorationHeader, TileInfoHeader, read_sequence_header_obu,
    read_uncompressed_header,
};
use aom_dsp::entropy::lr::{LrFrameConfig, RESTORE_NONE as LR_RESTORE_NONE};
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::partition::KfFrameContext;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_dsp::loopfilter::frame::{LfFrameBuf, LfMiGrid, LfParams, loop_filter_frame};
use aom_dsp::quant::{
    Dequants, Quants, aom_get_qmlevel_allintra, av1_build_quantizer, av1_dc_quant_qtx, set_q_index,
};
use aom_dsp::restore::pick::{LrPlanePixels, LrSearchInput, LrSearchSf, pick_filter_restoration};
#[cfg(feature = "c-oracle")]
use aom_sys_ref as c;
use aom_dsp::txb::cost_tokens_from_cdf;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

// ---------------------------------------------------------------------------
// Corpus / container helpers (mirrors the e2e gates' verbatim helpers)
// ---------------------------------------------------------------------------

/// Conformance corpus directory (`AOM_CONFORMANCE_DIR` override, else
/// `<workspace>/conformance/data`).
pub fn corpus_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("AOM_CONFORMANCE_DIR") {
        return std::path::PathBuf::from(d);
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("data")
}

/// `allow_screen_content_tools`, parsed out of a stream's own sequence +
/// uncompressed frame headers.
///
/// The port's palette and intraBC searches are gated on this bit exactly as C's
/// are, so a harness that does not read it cannot tell "this content has no
/// palette in it" from "the tool was never legal here". Real `aomenc` sets it
/// from its own screen-content detection; nothing in this repo forces it, and
/// forcing it would make every source look palette-capable.
///
/// Panics on a stream without both OBUs — every caller passes an encoder
/// output, so that is a bug rather than a runtime condition.
pub fn stream_allows_screen_content_tools(stream: &[u8]) -> bool {
    stream_frame_header(stream).allow_screen_content_tools
}

/// `base_qindex` from a reference stream's uncompressed frame header — the
/// artefact-side witness that a `--cq-level 0` cell really did come back
/// CODED-LOSSLESS (`kb5_lossless_speed_axis`'s reach assertion; playbook §2).
///
/// A single-pass parse is exact for this field: `read_uncompressed_header` reads
/// quantization_params BEFORE any of the reads it gates on `cfg.coded_lossless`,
/// which is the same argument `port_encode_full`'s two-pass probe rests on.
pub fn stream_base_qindex(stream: &[u8]) -> i32 {
    stream_frame_header(stream).quant.base_qindex
}

/// The uncompressed frame header of a reference stream's `OBU_FRAME`, parsed
/// against the config its own sequence header implies.
pub fn stream_frame_header(stream: &[u8]) -> FrameHeaderObu {
    let mut pos = 0usize;
    let (mut seqp, mut framep): (Option<&[u8]>, Option<&[u8]>) = (None, None);
    while pos < stream.len() {
        let hdr = aom_dsp::entropy::obu::read_obu_header(&stream[pos..]).expect("obu header");
        let after = pos + hdr.header_len;
        let (size, nb) = aom_dsp::entropy::leb128::uleb_decode(&stream[after..]).expect("leb128");
        let (start, end) = (after + nb, after + nb + size as usize);
        match hdr.obu_type {
            t if t == OBU_SEQUENCE_HEADER => seqp = Some(&stream[start..end]),
            t if t == OBU_FRAME => framep = Some(&stream[start..end]),
            _ => {}
        }
        pos = end;
    }
    let seq = read_sequence_header_obu(&mut ReadBitBuffer::new(seqp.expect("seq OBU")));
    let s = &seq.seq_header;
    let cc = &seq.color_config;
    let cfg = FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length
                as u32,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            frame_id_length: s.frame_id_length as u32,
            force_screen_content_tools: s.force_screen_content_tools,
            force_integer_mv: s.force_integer_mv,
            max_frame_width: s.max_frame_width,
            max_frame_height: s.max_frame_height,
            enable_order_hint: s.enable_order_hint,
            order_hint_bits_minus_1: s.order_hint_bits_minus_1,
            operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
            operating_point_idc: seq.operating_point_idc,
            op_decoder_model_param_present: seq.op_decoder_model_param_present,
            buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length as u32,
            temporal_layer_id: 0,
            spatial_layer_id: 0,
            ..Default::default()
        },
        frame_size: FrameSizeHeader {
            num_bits_width: s.num_bits_width,
            num_bits_height: s.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            ..Default::default()
        },
        num_planes: if cc.monochrome { 1 } else { 3 },
        separate_uv_delta_q: cc.separate_uv_delta_q,
        cdef: CdefHeader { enable_cdef: s.enable_cdef, ..Default::default() },
        restoration: RestorationHeader {
            enable_restoration: s.enable_restoration,
            sb_size_128: s.sb_size_128,
            subsampling_x: cc.subsampling_x,
            subsampling_y: cc.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    read_uncompressed_header(&mut ReadBitBuffer::new(framep.expect("frame OBU")), &cfg)
}

/// IVF header frame dimensions.
pub fn ivf_hdr_dims(data: &[u8]) -> (usize, usize) {
    (
        u16::from_le_bytes([data[12], data[13]]) as usize,
        u16::from_le_bytes([data[14], data[15]]) as usize,
    )
}

/// Split an IVF container into per-frame temporal-unit payloads (raw OBU bytes).
pub fn ivf_temporal_units(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 12; // 4-byte size + 8-byte timestamp
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        assert!(hdr.obu_has_size_field, "shim always sets has_size_field");
        let (size, size_bytes) =
            aom_dsp::entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}

fn tile_log2(blk_size: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Replay C's per-superblock delta-q derivation across the whole frame in the
/// order — and with the state resets — the real encoder uses, returning the
/// per-SB adjusted qindex in **frame SB raster** plus `deltaq_used`.
///
/// **The running base restarts at every TILE, not once per frame.** Both sides
/// of libaom do it, unconditionally:
///
/// * search — `encode_sb_row`, *"Reset delta for quantizer and loof [sic]
///   filters at the beginning of every tile"*, `av1/encoder/encodeframe.c:1232-1239`:
///   `if (mi_row == tile_info->mi_row_start || row_mt_enabled) { if
///   (delta_q_present_flag) xd->current_base_qindex = base_qindex; ... }`. The
///   `|| row_mt_enabled` arm — which would reset at every SB ROW instead — is
///   DEAD against this harness's oracle: `mt_info->row_mt_enabled` is only set
///   when `oxcf->row_mt && num_workers > 1` (`encodeframe.c:2410-2415`) and the
///   reference shim pins `cfg.g_threads = 1` (`aom-sys-ref/shim/dec_shim.c:498`),
///   so `num_workers == 1`. It is ported as the tile reset alone; a threaded
///   oracle would need the row arm too;
/// * pack — `write_modes`, `av1/encoder/bitstream.c:1745-1751`, the same
///   assignment at the top of every tile's payload (plus
///   `av1_reset_loop_filter_delta`);
/// * decode — the mirror reset, `av1/decoder/decodeframe.c:2948` (serial tile
///   loop) and `:3023` (`tile_worker_hook_init`).
///
/// The base then advances one SB at a time, in tile raster, by
/// `xd->current_base_qindex = mbmi->current_qindex` (`partition_search.c:1476`
/// on the search side, `bitstream.c:979` on the write side) — which is what
/// makes the SB's *own* qindex order-dependent, via
/// `av1_adjust_q_from_delta_q_res(res, prev, curr)`'s deadzone rounding
/// against `prev` (`av1/encoder/rd.c:494-505`). [`aom_encode::pack::pack_tile`]
/// already models both resets (its `search_base_qindex` init and its fresh
/// per-call `KfBlockState`); this replay exists so the harness can DERIVE
/// `delta_q_present` (`td->deltaq_used |= (x->delta_qindex != 0)`,
/// `encodeframe.c:375`, OR-reduced over tiles at `:1593`, and folded into the
/// header at `bitstream.c:4286-4289`) and the per-SB delta-lf without reading
/// either off the bootstrap — so it has to walk the same order the pack does.
///
/// `sb_qindex(mi_row, mi_col, running_base) -> adjusted qindex` is the mode's
/// own per-SB derivation (`setup_delta_q`, `encodeframe.c:297`).
fn replay_sb_qindex_tile_order(
    tile_grid: &[(i32, i32, i32, i32, i32, i32)],
    n_sb_x: i32,
    sb_mi: i32,
    base_qindex: i32,
    mut sb_qindex: impl FnMut(i32, i32, i32) -> i32,
) -> (Vec<i32>, bool) {
    let n_sb = tile_grid.iter().map(|t| (t.4 * t.5) as usize).sum::<usize>();
    let mut per_sb = vec![base_qindex; n_sb];
    let mut used = false;
    for &(mi_row_start, mi_col_start, _, _, n_sb_rows, n_sb_cols) in tile_grid {
        // encodeframe.c:1235 / bitstream.c:1746 — per-TILE reset.
        let mut running = base_qindex;
        let (sb_row0, sb_col0) = (mi_row_start / sb_mi, mi_col_start / sb_mi);
        for r in 0..n_sb_rows {
            for c in 0..n_sb_cols {
                let adj = sb_qindex(mi_row_start + r * sb_mi, mi_col_start + c * sb_mi, running);
                used |= adj != base_qindex;
                running = adj;
                per_sb[((sb_row0 + r) * n_sb_x + sb_col0 + c) as usize] = adj;
            }
        }
    }
    (per_sb, used)
}

fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    const MAX_TILE_WIDTH: i32 = 4096;
    const MAX_TILE_AREA: i32 = 4096 * 2304;
    const MAX_TILE_COLS: i32 = 64;
    const MAX_TILE_ROWS: i32 = 64;
    let sb_cols = (mi_cols + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_rows = (mi_rows + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_size_log2 = mib_size_log2 as i32 + 2;
    let max_width_sb = MAX_TILE_WIDTH >> sb_size_log2;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size_log2);
    let min_log2_cols = tile_log2(max_width_sb, sb_cols);
    let max_log2_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = tile_log2(max_tile_area_sb, sb_cols * sb_rows).max(min_log2_cols);
    TileInfoHeader {
        mi_cols,
        mi_rows,
        mib_size_log2,
        min_log2_cols,
        max_log2_cols,
        min_log2_rows: (min_log2_tiles - min_log2_cols).max(0),
        max_log2_rows,
        max_width_sb,
        max_height_sb: (max_tile_area_sb / max_width_sb.max(1)).max(1),
        ..Default::default()
    }
}

fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

// ---------------------------------------------------------------------------
// Decode cells
// ---------------------------------------------------------------------------

/// One decode benchmark cell: a single KEY-frame temporal unit from a real
/// conformance vector.
pub struct DecodeCell {
    pub label: String,
    pub tu: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

impl DecodeCell {
    /// Load the FIRST temporal unit (KEY frame) of a conformance vector.
    pub fn from_vector(label: &str, vector: &str) -> Self {
        Self::from_vector_opt(label, vector).unwrap_or_else(|| {
            let path = corpus_dir().join(format!("{vector}.ivf"));
            panic!(
                "{vector}: conformance vector missing at {path:?}; fetch via \
                 `python3 xtask/conformance.py --fetch --scope intra`"
            )
        })
    }

    /// Like [`from_vector`](Self::from_vector) but returns `None` if the `.ivf`
    /// is absent, so optional/regenerable cells (the gitignored `mosaic-*`
    /// photographic vectors) can be skipped gracefully instead of panicking a
    /// bench/profiler run in an environment that only fetched the conformance
    /// corpus.
    pub fn from_vector_opt(label: &str, vector: &str) -> Option<Self> {
        let path = corpus_dir().join(format!("{vector}.ivf"));
        let ivf = std::fs::read(&path).ok()?;
        let (w, h) = ivf_hdr_dims(&ivf);
        let tus = ivf_temporal_units(&ivf);
        Some(DecodeCell {
            label: label.to_string(),
            tu: tus[0].clone(),
            w,
            h,
        })
    }

    /// C-oracle decode (init + decode + plane copy + destroy).
    #[cfg(feature = "c-oracle")]
    pub fn c_decode(&self) -> c::RefDecodedFrame {
        c::ref_decode_av1_kf(&self.tu, self.w, self.h)
    }

    /// Port decode (parse + tile decode + post-filters + crop-out).
    pub fn port_decode(&self) -> aom_decode::frame::FrameDecode {
        aom_decode::frame::decode_frame_obus(&self.tu)
            .unwrap_or_else(|e| panic!("{}: port rejected the KEY frame: {e}", self.label))
    }

    /// Setup-time validation: the port's planes are byte-identical to C's.
    #[cfg(feature = "c-oracle")]
    pub fn assert_byte_exact(&self) {
        c::ref_init();
        let cref = self.c_decode();
        let rust = self.port_decode();
        assert_eq!(rust.y, cref.y, "{}: luma differs from C oracle", self.label);
        assert_eq!(rust.u, cref.u, "{}: U differs from C oracle", self.label);
        assert_eq!(rust.v, cref.v, "{}: V differs from C oracle", self.label);
    }
}

/// The standard Gate-3 decode cell set: 3 sizes (64², 196² partial-SB,
/// 352×288) and 3 quantizer levels at the largest size.
///
/// The `dec_mosaic_*` cells are the HEADLINE stills-decode workload
/// (`benchmarks/decode_4way_2026-07-17.csv`): real photographic 2K/4K KEY
/// frames encoded `aomenc --allintra` (⇒ CDEF off, LR off, QM off), where
/// aom-rs is ~2.2× rav1d-safe. They are the correct profiling target for the
/// non-post-filter decode hotspots (entropy/coeff/intra-pred/recon), unlike
/// the small conformance vectors which code CDEF+LR. Regenerable via
/// `mk_mosaic_y4m` + `aomenc` (see the CSV's Content provenance); the `.ivf`s
/// live gitignored under `conformance/data/`.
pub fn decode_cells() -> Vec<DecodeCell> {
    let mut cells = vec![
        DecodeCell::from_vector("dec_64x64", "av1-1-b8-01-size-64x64"),
        DecodeCell::from_vector("dec_196x196", "av1-1-b8-01-size-196x196"),
        DecodeCell::from_vector("dec_352x288_q00", "av1-1-b8-00-quantizer-00"),
        DecodeCell::from_vector("dec_352x288_q32", "av1-1-b8-00-quantizer-32"),
        DecodeCell::from_vector("dec_352x288_q63", "av1-1-b8-00-quantizer-63"),
    ];
    // Headline stills-decode cells — present only when regenerated (gitignored);
    // skipped gracefully when the environment fetched just the conformance corpus.
    for (label, vector) in [
        ("dec_mosaic_2k_cq20", "mosaic-2k-cq20"),
        ("dec_mosaic_2k_cq40", "mosaic-2k-cq40"),
        ("dec_mosaic_4k_cq20", "mosaic-4k-cq20"),
        ("dec_mosaic_4k_cq40", "mosaic-4k-cq40"),
    ] {
        cells.extend(DecodeCell::from_vector_opt(label, vector));
    }
    cells
}

// ---------------------------------------------------------------------------
// Encode cells
// ---------------------------------------------------------------------------

/// One encode benchmark cell: source planes + config. `y/u/v` are tight
/// (stride == width) u16 planes as both encode paths consume them.
#[derive(Clone)]
pub struct EncodeCell {
    pub label: String,
    pub w: usize,
    pub h: usize,
    pub mono: bool,
    pub ss_x: usize,
    pub ss_y: usize,
    pub usage: u32,
    pub cq_level: i32,
    /// `--cpu-used` for the C side AND the port's `SpeedFeatures` level.
    pub speed: i32,
    pub bd: u8,
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
}

/// CLI-toggle knob set for [`EncodeCell::port_encode_with`] — the C8-C11
/// toggle-sweep families (PARITY.md). `Default` reproduces the stock
/// envelope (every knob at its aomenc default), under which
/// `port_encode_with` == `port_encode` byte-for-byte on the proven gates.
///
/// Each knob mirrors one `aome_enc_control_id` control
/// ([`c::cx_ctrl`]); [`ToggleKnobs::c_ctrls`] emits the non-default ones
/// for [`EncodeCell::c_encode_ctrls`], and `port_encode_with` threads the
/// same values into the port's search config, so one struct drives both
/// sides of an RD-closeness cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToggleKnobs {
    /// `--enable-rect-partitions` (default 1): HORZ/VERT (and, downstream,
    /// AB) partition arms in the search.
    pub enable_rect_partitions: bool,
    /// `--enable-ab-partitions` (default 1): HORZ_A/HORZ_B/VERT_A/VERT_B.
    pub enable_ab_partitions: bool,
    /// `--enable-1to4-partitions` (default 1): HORZ_4/VERT_4.
    pub enable_1to4_partitions: bool,
    /// `--min-partition-size` in PIXELS {4,8,16,32,64,128} (default 4).
    pub min_partition_size_px: usize,
    /// `--max-partition-size` in PIXELS {4,8,16,32,64,128} (default 128).
    pub max_partition_size_px: usize,
    /// `--enable-intra-edge-filter` (default 1) — a SEQUENCE-header bit
    /// (encoder.c:646); the port side drives `SbEncodeEnv::
    /// disable_edge_filter` from this knob and ASSERTS the C stream's seq
    /// header agrees (no bootstrap flow).
    pub enable_intra_edge_filter: bool,
    /// `--enable-filter-intra` (default 1) — a SEQUENCE-header bit
    /// (encoder.c:647); drives `PickFrameCfg::enable_filter_intra` + the
    /// cost derivation + `PackCfg::enable_filter_intra`, seq bit asserted.
    pub enable_filter_intra: bool,
    /// `--enable-smooth-intra` (default 1): SMOOTH/SMOOTH_V/SMOOTH_H in
    /// BOTH the luma and chroma mode loops.
    pub enable_smooth_intra: bool,
    /// `--enable-paeth-intra` (default 1): PAETH, both loops.
    pub enable_paeth_intra: bool,
    /// `--enable-cfl-intra` (default 1): UV_CFL_PRED in the chroma loop.
    pub enable_cfl_intra: bool,
    /// `--enable-directional-intra` (default 1): every directional mode
    /// (V/H/D45..D203 + deltas), both loops.
    pub enable_directional_intra: bool,
    /// `--enable-diagonal-intra` (default 1): D45..D203, both loops.
    pub enable_diagonal_intra: bool,
    /// `--enable-angle-delta` (default 1): nonzero deltas on directional
    /// modes, both loops.
    pub enable_angle_delta: bool,
    /// `--enable-tx64` (default 1): 64-pt transform sizes in the tx-size
    /// search (off caps the largest tx at 32).
    pub enable_tx64: bool,
    /// `--enable-rect-tx` (default 1): rectangular tx sizes.
    pub enable_rect_tx: bool,
    /// `--enable-flip-idtx` (default 1): the FLIPADST/IDTX tx-type family
    /// in the ext-tx sets (`get_tx_mask`'s DCT_ADST_TX_MASK arm).
    pub enable_flip_idtx: bool,
    /// `--use-intra-dct-only` (default 0): force DCT_DCT for every luma
    /// intra txb.
    pub use_intra_dct_only: bool,
    /// `--use-intra-default-tx-only` (default 0): each luma intra txb
    /// searches only its mode's default tx type (MODE_EVAL arm,
    /// rdopt_utils.h:579).
    pub use_intra_default_tx_only: bool,
    /// `--reduced-tx-type-set` (default 0) — a FRAME-header bit
    /// (`reduced_tx_set_used`, encodeframe.c:2712): both the search's
    /// ext-tx sets and the coded tx-type signalling shrink. The port side
    /// asserts the bootstrapped frame header bit equals this knob.
    pub reduced_tx_type_set: bool,
    /// `--enable-tx-size-search` (default 1). OFF forces every eval stage
    /// to USE_LARGESTALL (`winner_mode_sf.tx_size_search_level = 3`,
    /// speed_features.c:2726) and the frame codes `tx_mode = TX_MODE_LARGEST`
    /// (asserted against the bootstrap header). C FORBIDS combining this
    /// with `--enable-tx64=0` (encodeframe.c:2461 assert).
    pub enable_tx_size_search: bool,
    /// `--cdf-update-mode` (default 1 = update on every frame; 0 = no CDF
    /// update for any frame → the KEY header codes `disable_cdf_update=1`,
    /// asserted against the bootstrap, and the pack skips symbol
    /// adaptation via `PackCfg::allow_update_cdf`). Mode 2 (selective) is
    /// identical to 1 on a lone KEY frame — not swept.
    pub cdf_update_mode: u32,
    /// `--enable-palette=1` (NOT a C8-C11 toggle — the port's palette RD
    /// search enable, carried here so `port_encode_with` can thread it into
    /// `PickFrameCfg::palette_costs`; the search still requires the frame's
    /// `allow_screen_content_tools`, exactly as C). Default OFF.
    pub enable_palette: bool,
    /// `--disable-trellis-quant` (default 3): 0 = FULL_TRELLIS_OPT,
    /// 1 = NO_TRELLIS_OPT, 2 = FINAL_PASS_TRELLIS_OPT (trellis only in the
    /// OUTPUT_ENABLED pack pass), 3 = NO_ESTIMATE_YRD_TRELLIS_OPT
    /// (default; ≈ FULL on the intra envelope — estimate_yrd_for_sb is
    /// inter-only). Mapping: init_rd_sf (speed_features.c:2479-2498);
    /// search-side `skip_trellis = !is_trellis_used(opt, DRY_RUN_NORMAL)`,
    /// pack-side `enable_optimize_b` (is_trellis_used(opt, OUTPUT_ENABLED)).
    pub disable_trellis_quant: u32,
    /// `--coeff-cost-upd-freq` / `--mode-cost-upd-freq` (default 0 =
    /// COST_UPD_SB; 1 SBROW / 2 TILE / 3 OFF).
    /// HANDOFF: C ctrls emitted below; the PORT-side gate is NOT wired yet —
    /// pack.rs's per-SB `derive_real_costs(kf, .., None)` rebuild (the sb_real
    /// block) must split per table set and gate: SB = rebuild every SB
    /// (current behavior); SBROW = rebuild only at `c == 0` in pack_tile's
    /// SB loop (skip_cost_update's mi_col-at-tile-start arm); TILE/OFF =
    /// never rebuild (single-tile ⇒ identical outcomes; frame-init tables
    /// throughout). coeff gates `sb_env.{coeff_costs_*, tx_type_costs}`,
    /// mode gates `sb_pick_cfg.{mode_costs, tx_size_costs, skip_costs, ...}`
    /// — derive_real_costs returns both halves; USE the rebuilt half only
    /// when its knob says so. ALSO: C skips ALL cost updates when
    /// disable_cdf_update (av1_set_cost_upd_freq's early return,
    /// encodeframe_utils.c:1629) — the port is equivalent by construction
    /// (kf never adapts ⇒ rebuild == frame-init), keep it that way.
    pub coeff_cost_upd_freq: u32,
    /// See `coeff_cost_upd_freq`.
    pub mode_cost_upd_freq: u32,
    /// `--deltaq-mode=3` (`DELTA_Q_PERCEPTUAL_AI`, family C5): when set, the
    /// port builds the wiener-variance map ([`aom_encode::allintra_vis::
    /// av1_set_mb_wiener_variance`]) and threads the per-SB perceptual-AI
    /// qindex into the pack. The C side must be driven with
    /// `AV1E_SET_DELTAQ_MODE = 3` (the reference bootstrap stream). PORT-side
    /// only — not emitted by [`ToggleKnobs::c_ctrls`].
    pub deltaq_mode3: bool,
    /// `--deltaq-mode=2` (`DELTA_Q_PERCEPTUAL`, wavelet AC energy): when set,
    /// the port derives the per-SB qindex from the SB source wavelet energy
    /// ([`aom_encode::allintra_vis::setup_delta_q_perceptual`]) and threads it
    /// into the pack. The C side must be driven with `AV1E_SET_DELTAQ_MODE = 2`.
    /// PORT-side only — not emitted by [`ToggleKnobs::c_ctrls`].
    pub deltaq_mode2: bool,
    /// Anti-vacuity witness ONLY (PORT-side): force
    /// `sf.prune_tx_type_using_stats = 0` even on a >=480p speed>=2 frame where
    /// the framesize derivation would enable it. `--quant-b-adapt`-style speed
    /// features have no CLI control, so the C side is driven by `--cpu-used`
    /// alone; this flag lets a witness prove the ported stats-prune is
    /// LOAD-BEARING (port-without-prune must diverge from real aomenc, port-with
    /// must match). Default false — not emitted by [`ToggleKnobs::c_ctrls`].
    pub disable_tx_stats_prune: bool,
    /// `--delta-lf-mode=1` (`AV1E_SET_DELTALF_MODE`): derive per-SB
    /// `delta_lf_from_base` from each SB's `delta_qindex` and code it alongside
    /// the delta-qindex. Rides on a firing delta-q mode (combine with
    /// `deltaq_mode2`/`deltaq_mode3`). The C side must be driven with
    /// `AV1E_SET_DELTALF_MODE = 1` (+ the delta-q ctrl).
    pub delta_lf_mode: bool,
    /// `--enable-intrabc=1` (screen content). Like `enable_palette`, this is
    /// the PORT's intrabc RD-search enable — the C side is driven via
    /// [`EncodeCell::c_encode_screen`]. When the frame header codes
    /// `allow_intrabc`, the pack ALWAYS writes the per-block `use_intrabc`
    /// flags (the decoder reads them unconditionally); this knob only gates
    /// whether the port RUNS the DV search (so `false` gives the all-intra
    /// witness against `true`). Default OFF.
    pub enable_intrabc: bool,
    /// `--tune-content=screen` (`AV1E_SET_TUNE_CONTENT = AOM_CONTENT_SCREEN`):
    /// `av1_set_screen_content_options`' second arm (encoder.c:2449-2455) —
    /// screen tools + the search-time `allow_intrabc` ON without running the
    /// detector. The port cannot read this from the stream (the header only
    /// carries the outcome), so the cell declares it. KB-41 root #13.
    pub tune_content_screen: bool,
    /// `--enable-qm=1 --qm-min=<min> --qm-max=<max>` (quantization matrices).
    /// `Some((qm_min, qm_max))` makes the port derive the frame
    /// `qmatrix_level_{y,u,v}` per `av1_set_quantizer`'s allintra arm
    /// (`aom_get_qmlevel_allintra`) and thread them into the SB search + pack;
    /// the derived levels are CROSS-CHECKED against the levels the bootstrap
    /// stream's frame header actually signalled, so a wiring error fails loudly
    /// before any byte comparison. PORT-side only — not emitted by
    /// [`ToggleKnobs::c_ctrls`] (no `cx_ctrl` id is wired for QM); drive the C
    /// side with [`EncodeCell::c_encode_qm`], which is the same base config as
    /// [`EncodeCell::c_encode`] plus the three QM controls. Default `None`.
    pub qm: Option<(i32, i32)>,
}

impl Default for ToggleKnobs {
    fn default() -> Self {
        ToggleKnobs {
            enable_rect_partitions: true,
            enable_ab_partitions: true,
            enable_1to4_partitions: true,
            min_partition_size_px: 4,
            max_partition_size_px: 128,
            enable_intra_edge_filter: true,
            enable_filter_intra: true,
            enable_smooth_intra: true,
            enable_paeth_intra: true,
            enable_cfl_intra: true,
            enable_directional_intra: true,
            enable_diagonal_intra: true,
            enable_angle_delta: true,
            enable_tx64: true,
            enable_rect_tx: true,
            enable_flip_idtx: true,
            use_intra_dct_only: false,
            use_intra_default_tx_only: false,
            reduced_tx_type_set: false,
            enable_tx_size_search: true,
            cdf_update_mode: 1,
            enable_palette: false,
            disable_trellis_quant: 3,
            coeff_cost_upd_freq: 0,
            mode_cost_upd_freq: 0,
            deltaq_mode3: false,
            deltaq_mode2: false,
            disable_tx_stats_prune: false,
            delta_lf_mode: false,
            enable_intrabc: false,
            tune_content_screen: false,
            qm: None,
        }
    }
}

/// `dim_to_size` (partition_strategy.h:201): pixel dimension -> square
/// BLOCK_SIZE enum value.
fn dim_to_bsize(px: usize) -> usize {
    match px {
        4 => 0,    // BLOCK_4X4
        8 => 3,    // BLOCK_8X8
        16 => 6,   // BLOCK_16X16
        32 => 9,   // BLOCK_32X32
        64 => 12,  // BLOCK_64X64
        128 => 15, // BLOCK_128X128
        _ => panic!("partition size {px}px is not a square BLOCK dimension"),
    }
}

/// `init_rd_sf` (speed_features.c:2479-2498), non-lossless arm:
/// `--disable-trellis-quant` value → `TRELLIS_OPT_TYPE`.
fn trellis_opt_of_knob(v: u32) -> TrellisOptType {
    match v {
        0 => TrellisOptType::FullTrellisOpt,
        1 => TrellisOptType::NoTrellisOpt,
        2 => TrellisOptType::FinalPassTrellisOpt,
        3 => TrellisOptType::NoEstimateYrdTrellisOpt,
        _ => panic!("--disable-trellis-quant {v} out of range 0..=3"),
    }
}

impl ToggleKnobs {
    /// The `(ctrl_id, value)` pairs for the C side — only knobs that differ
    /// from the aomenc default are emitted (a default-knobs cell reproduces
    /// `EncodeCell::c_encode` exactly).
    #[cfg(feature = "c-oracle")]
    pub fn c_ctrls(&self) -> Vec<(i32, i32)> {
        use c::cx_ctrl::*;
        let d = ToggleKnobs::default();
        let mut out = Vec::new();
        if self.enable_rect_partitions != d.enable_rect_partitions {
            out.push((
                AV1E_SET_ENABLE_RECT_PARTITIONS,
                self.enable_rect_partitions as i32,
            ));
        }
        if self.enable_ab_partitions != d.enable_ab_partitions {
            out.push((
                AV1E_SET_ENABLE_AB_PARTITIONS,
                self.enable_ab_partitions as i32,
            ));
        }
        if self.enable_1to4_partitions != d.enable_1to4_partitions {
            out.push((
                AV1E_SET_ENABLE_1TO4_PARTITIONS,
                self.enable_1to4_partitions as i32,
            ));
        }
        if self.min_partition_size_px != d.min_partition_size_px {
            out.push((
                AV1E_SET_MIN_PARTITION_SIZE,
                self.min_partition_size_px as i32,
            ));
        }
        if self.max_partition_size_px != d.max_partition_size_px {
            out.push((
                AV1E_SET_MAX_PARTITION_SIZE,
                self.max_partition_size_px as i32,
            ));
        }
        if self.enable_intra_edge_filter != d.enable_intra_edge_filter {
            out.push((
                AV1E_SET_ENABLE_INTRA_EDGE_FILTER,
                self.enable_intra_edge_filter as i32,
            ));
        }
        if self.enable_filter_intra != d.enable_filter_intra {
            out.push((
                AV1E_SET_ENABLE_FILTER_INTRA,
                self.enable_filter_intra as i32,
            ));
        }
        if self.enable_smooth_intra != d.enable_smooth_intra {
            out.push((
                AV1E_SET_ENABLE_SMOOTH_INTRA,
                self.enable_smooth_intra as i32,
            ));
        }
        if self.enable_paeth_intra != d.enable_paeth_intra {
            out.push((AV1E_SET_ENABLE_PAETH_INTRA, self.enable_paeth_intra as i32));
        }
        if self.enable_cfl_intra != d.enable_cfl_intra {
            out.push((AV1E_SET_ENABLE_CFL_INTRA, self.enable_cfl_intra as i32));
        }
        if self.enable_directional_intra != d.enable_directional_intra {
            out.push((
                AV1E_SET_ENABLE_DIRECTIONAL_INTRA,
                self.enable_directional_intra as i32,
            ));
        }
        if self.enable_diagonal_intra != d.enable_diagonal_intra {
            out.push((
                AV1E_SET_ENABLE_DIAGONAL_INTRA,
                self.enable_diagonal_intra as i32,
            ));
        }
        if self.enable_angle_delta != d.enable_angle_delta {
            out.push((AV1E_SET_ENABLE_ANGLE_DELTA, self.enable_angle_delta as i32));
        }
        if self.enable_tx64 != d.enable_tx64 {
            out.push((AV1E_SET_ENABLE_TX64, self.enable_tx64 as i32));
        }
        if self.enable_rect_tx != d.enable_rect_tx {
            out.push((AV1E_SET_ENABLE_RECT_TX, self.enable_rect_tx as i32));
        }
        if self.enable_flip_idtx != d.enable_flip_idtx {
            out.push((AV1E_SET_ENABLE_FLIP_IDTX, self.enable_flip_idtx as i32));
        }
        if self.use_intra_dct_only != d.use_intra_dct_only {
            out.push((AV1E_SET_INTRA_DCT_ONLY, self.use_intra_dct_only as i32));
        }
        if self.use_intra_default_tx_only != d.use_intra_default_tx_only {
            out.push((
                AV1E_SET_INTRA_DEFAULT_TX_ONLY,
                self.use_intra_default_tx_only as i32,
            ));
        }
        if self.reduced_tx_type_set != d.reduced_tx_type_set {
            out.push((
                AV1E_SET_REDUCED_TX_TYPE_SET,
                self.reduced_tx_type_set as i32,
            ));
        }
        if self.enable_tx_size_search != d.enable_tx_size_search {
            out.push((
                AV1E_SET_ENABLE_TX_SIZE_SEARCH,
                self.enable_tx_size_search as i32,
            ));
        }
        if self.cdf_update_mode != d.cdf_update_mode {
            out.push((AV1E_SET_CDF_UPDATE_MODE, self.cdf_update_mode as i32));
        }
        if self.disable_trellis_quant != d.disable_trellis_quant {
            out.push((
                AV1E_SET_DISABLE_TRELLIS_QUANT,
                self.disable_trellis_quant as i32,
            ));
        }
        if self.coeff_cost_upd_freq != d.coeff_cost_upd_freq {
            out.push((
                AV1E_SET_COEFF_COST_UPD_FREQ,
                self.coeff_cost_upd_freq as i32,
            ));
        }
        if self.mode_cost_upd_freq != d.mode_cost_upd_freq {
            out.push((AV1E_SET_MODE_COST_UPD_FREQ, self.mode_cost_upd_freq as i32));
        }
        out
    }

    /// `x->sb_enc.max_partition_size` (set_max_min_partition_size,
    /// partition_strategy.h:214): `min(sf.default_max_partition_size,
    /// dim_to_size(oxcf px), sb_size)`. The auto-max ML arm is
    /// inter-only (`use_auto_max_partition` requires `!frame_is_intra_only`).
    fn max_partition_bsize(&self, sf_default_max: usize, sb_bsize: usize) -> usize {
        sf_default_max
            .min(dim_to_bsize(self.max_partition_size_px))
            .min(sb_bsize)
    }

    /// `x->sb_enc.min_partition_size` (set_max_min_partition_size,
    /// partition_strategy.h:224-230): `min(max(sf.default_min_partition_size,
    /// dim_to_size(oxcf px)), sb_size)`.
    ///
    /// `sf_default_min` is `BLOCK_4X4` (0) on the whole gated envelope —
    /// speed 0..6 sub-2160p — so the `max` is an identity there. It becomes
    /// load-bearing at speed >= 7 (:570) and, per KB-19, at `min(w, h) >=
    /// 2160` at any speed (speed_features.c:187-189).
    fn min_partition_bsize(&self, sf_default_min: usize, sb_bsize: usize) -> usize {
        sf_default_min
            .max(dim_to_bsize(self.min_partition_size_px))
            .min(sb_bsize)
    }
}

impl EncodeCell {
    /// Real-content cell: decode the first KEY frame of a conformance vector
    /// via the C oracle and (optionally) crop an SB-aligned window —
    /// exactly the KB-6 real-image gate's recipe, so byte-exactness of the
    /// port on these cells is already a landed CI gate at speed 0.
    #[cfg(feature = "c-oracle")]
    pub fn real_content(
        label: &str,
        vector: &str,
        crop: Option<(usize, usize, usize, usize)>, // (w, h, off_x, off_y)
        cq_level: i32,
        speed: i32,
    ) -> Self {
        c::ref_init();
        let path = corpus_dir().join(format!("{vector}.ivf"));
        let ivf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{vector}: conformance vector missing at {path:?} ({e}); fetch via \
                 `python3 xtask/conformance.py --fetch --scope intra`"
            )
        });
        let (fw, fh) = ivf_hdr_dims(&ivf);
        let tus = ivf_temporal_units(&ivf);
        let frame = c::ref_decode_av1_kf(&tus[0], fw, fh);
        let bd = frame.info[0] as u8;
        let mono = frame.info[1] != 0;
        let ss_x = frame.info[2] as usize;
        let ss_y = frame.info[3] as usize;
        let fcw = (fw + ss_x) >> ss_x;
        let (w, h, off_x, off_y) = match crop {
            None => (fw, fh, 0, 0),
            Some((cw, ch, ox, oy)) => (cw, ch, ox, oy),
        };
        assert!(
            off_x + w <= fw && off_y + h <= fh,
            "{label}: crop exceeds frame"
        );
        assert!(
            off_x % 2 == 0 && off_y % 2 == 0,
            "{label}: crop offset must be even"
        );
        let (cox, coy) = (off_x >> ss_x, off_y >> ss_y);
        let (cw, ch) = if mono {
            (0, 0)
        } else {
            ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
        };
        let mut y = vec![0u16; w * h];
        for r in 0..h {
            for col in 0..w {
                y[r * w + col] = frame.y[(r + off_y) * fw + (col + off_x)];
            }
        }
        let mut u = vec![0u16; cw * ch];
        let mut v = vec![0u16; cw * ch];
        if !mono {
            for r in 0..ch {
                for col in 0..cw {
                    u[r * cw + col] = frame.u[(r + coy) * fcw + (col + cox)];
                    v[r * cw + col] = frame.v[(r + coy) * fcw + (col + cox)];
                }
            }
        }
        EncodeCell {
            label: label.to_string(),
            w,
            h,
            mono,
            ss_x,
            ss_y,
            usage: 2, // ALLINTRA — the primary configuration
            cq_level,
            speed,
            bd,
            y,
            u,
            v,
        }
    }

    /// Synthetic diag-gradient 4:2:0 cell from the byte-exact speed-gate grid
    /// (`encoder_gate_speed4_textured_allintra`): luma `32 + (r+c)*190/(w+h)`,
    /// chroma `60 + (r*7 + c*3) % 80`. Used for the speed-4 point because the
    /// speed-4 byte gates are proven on this exact content.
    pub fn synthetic_diag(label: &str, w: usize, h: usize, cq_level: i32, speed: i32) -> Self {
        let mut y = vec![0u16; w * h];
        for r in 0..h {
            for col in 0..w {
                y[r * w + col] = (32 + (r + col) * 190 / (w + h)) as u16;
            }
        }
        let (cw, ch) = ((w + 1) >> 1, (h + 1) >> 1);
        let mut u = vec![0u16; cw * ch];
        let mut v = vec![0u16; cw * ch];
        for r in 0..ch {
            for col in 0..cw {
                let val = (60 + (r * 7 + col * 3) % 80) as u16;
                u[r * cw + col] = val;
                v[r * cw + col] = val;
            }
        }
        EncodeCell {
            label: label.to_string(),
            w,
            h,
            mono: false,
            ss_x: 1,
            ss_y: 1,
            usage: 2,
            cq_level,
            speed,
            bd: 8,
            y,
            u,
            v,
        }
    }

    /// The C oracle's full KEY encode (the aomenc path: codec init + encode +
    /// destroy), producing the reference bitstream. Also the untimed setup
    /// step that produces the header-bootstrap bytes for [`Self::port_encode`].
    #[cfg(feature = "c-oracle")]
    pub fn c_encode(&self) -> Vec<u8> {
        c::ref_encode_av1_kf(
            &self.y,
            &self.u,
            &self.v,
            self.w,
            self.h,
            i32::from(self.bd),
            self.mono,
            self.ss_x as i32,
            self.ss_y as i32,
            self.cq_level,
            self.speed,
            false,
            false,
            self.usage,
            0,
            false,
        )
    }

    /// The C oracle's KEY encode with explicit screen-content tool knobs
    /// (`--enable-palette` / `--enable-intrabc`, the
    /// `shim_encode_av1_kf_screen_content` path — otherwise identical to
    /// [`Self::c_encode`]).
    #[cfg(feature = "c-oracle")]
    pub fn c_encode_screen(&self, enable_palette: bool, enable_intrabc: bool) -> Vec<u8> {
        c::ref_encode_av1_kf_screen_content(
            &self.y,
            &self.u,
            &self.v,
            self.w,
            self.h,
            i32::from(self.bd),
            self.mono,
            self.ss_x as i32,
            self.ss_y as i32,
            self.cq_level,
            self.speed,
            false,
            false,
            self.usage,
            0,
            false,
            enable_palette,
            enable_intrabc,
        )
    }

    /// The C oracle's KEY encode with quantization matrices on
    /// (`--enable-qm=1 --qm-min --qm-max`, the `shim_encode_av1_kf_qm` path) —
    /// otherwise identical to [`Self::c_encode`], INCLUDING `--cpu-used =
    /// self.speed`. The port counterpart is
    /// `port_encode_with(.., ToggleKnobs { qm: Some((qm_min, qm_max)), .. })`.
    #[cfg(feature = "c-oracle")]
    pub fn c_encode_qm(&self, qm_min: i32, qm_max: i32) -> Vec<u8> {
        c::ref_encode_av1_kf_qm(
            &self.y,
            &self.u,
            &self.v,
            self.w,
            self.h,
            i32::from(self.bd),
            self.mono,
            self.ss_x as i32,
            self.ss_y as i32,
            self.cq_level,
            self.speed,
            false,
            false,
            self.usage,
            0,
            false,
            qm_min,
            qm_max,
        )
    }

    /// [`Self::c_encode`] plus extra `(ctrl_id, value)` control pairs
    /// ([`c::cx_ctrl`]) — the toggle-sweep C side. `&[]` reproduces
    /// `c_encode` exactly (same base config, no extra controls).
    #[cfg(feature = "c-oracle")]
    pub fn c_encode_ctrls(&self, ctrls: &[(i32, i32)]) -> Vec<u8> {
        c::ref_encode_av1_kf_ctrls(
            &self.y,
            &self.u,
            &self.v,
            self.w,
            self.h,
            i32::from(self.bd),
            self.mono,
            self.ss_x as i32,
            self.ss_y as i32,
            self.cq_level,
            self.speed,
            self.usage,
            ctrls,
        )
    }

    /// [`Self::c_encode`] with `--enable-restoration=1`
    /// (`AV1E_SET_ENABLE_RESTORATION`) — the reference stream for the
    /// loop-restoration-search parity gate.
    #[cfg(feature = "c-oracle")]
    pub fn c_encode_lr(&self) -> Vec<u8> {
        c::ref_encode_av1_kf(
            &self.y,
            &self.u,
            &self.v,
            self.w,
            self.h,
            i32::from(self.bd),
            self.mono,
            self.ss_x as i32,
            self.ss_y as i32,
            self.cq_level,
            self.speed,
            false,
            true, // enable_restoration
            self.usage,
            0,
            false,
        )
    }

    /// A plain `aomenc --allintra` encode with NO coding-tool flags — every
    /// tool sits at its ALLINTRA default (cdef OFF, **loop-restoration ON**, qm
    /// OFF). This is the true-default stream a drop-in replacement must match;
    /// the reference for the default-parity gate. (Contrast [`Self::c_encode`],
    /// which forces `--enable-restoration=0` — a NON-default config.)
    #[cfg(feature = "c-oracle")]
    pub fn c_encode_defaults(&self) -> Vec<u8> {
        c::ref_encode_av1_kf_defaults(
            &self.y,
            &self.u,
            &self.v,
            self.w,
            self.h,
            i32::from(self.bd),
            self.mono,
            self.ss_x as i32,
            self.ss_y as i32,
            self.cq_level,
            self.speed,
            self.usage,
        )
    }

    /// Extract the frame OBU payload from a reference stream (the byte-match
    /// target for [`Self::port_encode`]).
    pub fn frame_obu_payload(stream: &[u8]) -> Vec<u8> {
        walk_obus(stream)
            .iter()
            .find(|(t, _)| *t == OBU_FRAME)
            .map(|(_, p)| p.to_vec())
            .expect("no frame OBU in reference stream")
    }

    /// The port's full encode: bootstrap header-field parse (timed,
    /// microseconds) + quantizer/cost derivation + strided copy + border
    /// extension + the full SB search+pack walk + LF-level search + OBU
    /// assembly. Returns the assembled frame OBU payload — byte-identical to
    /// the reference stream's (asserted in [`Self::assert_byte_exact`]).
    ///
    /// This replicates the landed byte-exact e2e gates verbatim
    /// (`encoder_gate_chroma_ss_e2e.rs::run_case` partial-SB handling +
    /// `encoder_gate_e2e_byte_match.rs::attempt_case_content_uv` speed
    /// threading); cells at cq >= 1 only (the lossless two-pass probe is out
    /// of scope here).
    ///
    /// The DEFAULT path is restoration-aware: when the bootstrap's sequence
    /// header has `enable_restoration = 1` (the allintra default — see
    /// [`Self::c_encode_defaults`]), this additionally runs the byte-exact
    /// loop-restoration search + emits the restoration syntax, matching C's
    /// `is_restoration_used`. A restoration-off bootstrap (e.g. [`Self::c_encode`],
    /// `--enable-restoration=0`) skips it — unchanged from the historical
    /// behaviour.
    pub fn port_encode(&self, bootstrap: &[u8]) -> Vec<u8> {
        self.port_encode_with(bootstrap, &ToggleKnobs::default())
    }

    /// [`Self::port_encode`] with a port-DERIVED film-grain params block injected
    /// into the frame header (the C7 `--film-grain-table` parity path). `grain`
    /// is the params the port looked up from the grain table
    /// ([`aom_encode::grain_table`]) — NOT read from `bootstrap`; the harness
    /// overwrites the frame header's grain block with `grain` (context fields
    /// `monochrome`/`subsampling_*`/`is_inter_frame` set from this cell) and
    /// forces `film_grain_params_present`, so a byte-match vs a real
    /// `--film-grain-table` encode proves the port's own table→params→header
    /// chain (rule 4: no bootstrap leak — the bootstrap's grain bits are never
    /// read on this path). The coded tile bytes are the ordinary port encode
    /// (grain is decode-side synthesis, so the `-table` path leaves them
    /// unchanged). Requires `bootstrap` to be a film-grain stream (asserted).
    pub fn port_encode_film_grain(&self, bootstrap: &[u8], grain: &FilmGrainParams) -> Vec<u8> {
        self.port_encode_full(bootstrap, &ToggleKnobs::default(), false, Some(grain))
    }

    /// [`Self::port_encode`] with explicit CLI-toggle knobs threaded into
    /// the port's search config ([`ToggleKnobs`]; the toggle-sweep port
    /// side). `ToggleKnobs::default()` == `port_encode`. `knobs.enable_palette`
    /// additionally turns the palette RD search on (the port analogue of the C
    /// side's `--enable-palette=1`; the search still requires the frame's
    /// `allow_screen_content_tools`, exactly as C).
    pub fn port_encode_with(&self, bootstrap: &[u8], knobs: &ToggleKnobs) -> Vec<u8> {
        self.port_encode_full(bootstrap, knobs, false, None)
    }

    /// [`Self::port_encode`] plus the loop-restoration ENCODER stage
    /// (`--enable-restoration=1` parity): after the pack + LF-level
    /// derivation, APPLY the derived deblock to the reconstruction, run the
    /// ported `av1_pick_filter_restoration` search on (source, deblocked
    /// recon), and — when any plane restores — REPACK the tile with the
    /// per-RU parameters interleaved at each superblock root
    /// (`loop_restoration_write_sb_coeffs`) and write the derived
    /// frame-restoration header fields. The bootstrap must be an
    /// `enable_restoration=1` stream ([`Self::c_encode_lr`]); the
    /// restoration DECISIONS are never copied from it.
    pub fn port_encode_lr(&self, bootstrap: &[u8]) -> Vec<u8> {
        self.port_encode_full(bootstrap, &ToggleKnobs::default(), true, None)
    }

    fn port_encode_full(
        &self,
        bootstrap: &[u8],
        knobs: &ToggleKnobs,
        lr_stage: bool,
        film_grain: Option<&FilmGrainParams>,
    ) -> Vec<u8> {
        let (w, h, mono, ss_x, ss_y, bd) =
            (self.w, self.h, self.mono, self.ss_x, self.ss_y, self.bd);
        let obus = walk_obus(bootstrap);
        let seq_payload = obus
            .iter()
            .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
            .map(|(_, p)| *p)
            .expect("no sequence-header OBU");
        let mut seq_rb = ReadBitBuffer::new(seq_payload);
        let seq = read_sequence_header_obu(&mut seq_rb);
        let (frame_obu_type, frame_payload) = obus
            .iter()
            .find(|(t, _)| *t == OBU_FRAME || *t == 3)
            .map(|(t, p)| (*t, *p))
            .expect("no frame OBU");
        assert_eq!(frame_obu_type, OBU_FRAME, "expected combined OBU_FRAME");

        let s = &seq.seq_header;
        let cc = &seq.color_config;
        let num_planes = if cc.monochrome { 1 } else { 3 };
        let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
        let mi_cols = mi_dim(s.max_frame_width);
        let mi_rows = mi_dim(s.max_frame_height);
        // SB128: the seq header's `use_128x128_superblock` bit (parsed above)
        // selects the live superblock geometry. sb64 keeps the module SB/SB_MI
        // constants (BLOCK_64X64 / 16 mi); sb128 -> BLOCK_128X128 / 32 mi /
        // 128 px. Threaded into the SB grid, partition SB caps, pack walk, LF
        // grid and (out-of-default) delta-q setup below.
        let sb_block = if s.sb_size_128 { 15usize } else { SB };
        let sb_mi = if s.sb_size_128 { 2 * SB_MI } else { SB_MI };
        let sb_px = (sb_mi * 4) as usize;

        let cfg = FrameHeaderObu {
            prefix: FrameHeaderPrefix {
                reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
                decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
                equal_picture_interval: seq.timing_info.equal_picture_interval,
                frame_presentation_time_length: seq
                    .decoder_model_info
                    .frame_presentation_time_length
                    as u32,
                frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
                frame_id_length: s.frame_id_length as u32,
                force_screen_content_tools: s.force_screen_content_tools,
                force_integer_mv: s.force_integer_mv,
                max_frame_width: s.max_frame_width,
                max_frame_height: s.max_frame_height,
                enable_order_hint: s.enable_order_hint,
                order_hint_bits_minus_1: s.order_hint_bits_minus_1,
                operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
                operating_point_idc: seq.operating_point_idc,
                op_decoder_model_param_present: seq.op_decoder_model_param_present,
                buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length
                    as u32,
                temporal_layer_id: 0,
                spatial_layer_id: 0,
                ..Default::default()
            },
            frame_size: FrameSizeHeader {
                num_bits_width: s.num_bits_width,
                num_bits_height: s.num_bits_height,
                superres_upscaled_width: s.max_frame_width,
                superres_upscaled_height: s.max_frame_height,
                enable_superres: s.enable_superres,
                ..Default::default()
            },
            tile_info: tile_limits(mi_cols, mi_rows, mib_size_log2),
            num_planes,
            separate_uv_delta_q: cc.separate_uv_delta_q,
            loopfilter: LoopfilterHeader {
                last_ref_deltas: KF_REF_DELTAS,
                last_mode_deltas: KF_MODE_DELTAS,
                ..Default::default()
            },
            cdef: CdefHeader {
                enable_cdef: s.enable_cdef,
                ..Default::default()
            },
            restoration: RestorationHeader {
                enable_restoration: s.enable_restoration,
                sb_size_128: s.sb_size_128,
                subsampling_x: cc.subsampling_x,
                subsampling_y: cc.subsampling_y,
                ..Default::default()
            },
            // On the film-grain injection path the port derives the grain block
            // itself (below) — tell the header reader to SKIP the bootstrap's
            // grain bits (rule 4: no bootstrap leak; also avoids depending on the
            // reader's grain seq-context, which this cfg doesn't populate).
            film_grain_params_present: seq.film_grain_params_present && film_grain.is_none(),
            ..Default::default()
        };

        // Two-pass header parse for coded-lossless, mirroring the decoder
        // (aom-decode/src/frame.rs:466-490) and `encoder_gate_chroma_ss_e2e::
        // run_case`. `read_uncompressed_header` gates its loop-filter / CDEF /
        // restoration / tx-mode tail reads on `cfg.coded_lossless` /
        // `cfg.all_lossless` — a writer-mirror INPUT — but quant and
        // segmentation precede every gated read, so a probe pass with
        // `coded_lossless = false` yields exact quant regardless of the (then
        // mis-read) tail. Recompute `coded_lossless` from the probe's quant and,
        // when the stream IS coded-lossless (`--cq-level 0` / `--lossless=1`),
        // re-parse with the correct gating.
        //
        // This harness encodes segmentation-off KEY frames, so
        // `frame_coded_lossless` reduces to `base_qindex == 0` with all five
        // plane q-deltas 0; no superres here, so `all_lossless ==
        // coded_lossless`. Until 2026-08-03 the probe was missing and the whole
        // e2e lossless path was closed by an `assert!(base_qindex > 0)` — at
        // EVERY speed, which is what made the coverage queue's T1 entry a
        // harness bug rather than an encoder one.
        let mut probe_rb = ReadBitBuffer::new(frame_payload);
        let probe = read_uncompressed_header(&mut probe_rb, &cfg);
        let coded_lossless = !probe.prefix.show_existing_frame
            && probe.prefix.frame_type == 0
            && probe.prefix.show_frame
            && probe.quant.base_qindex == 0
            && probe.quant.y_dc_delta_q == 0
            && probe.quant.u_dc_delta_q == 0
            && probe.quant.u_ac_delta_q == 0
            && probe.quant.v_dc_delta_q == 0
            && probe.quant.v_ac_delta_q == 0;
        let mut p = if coded_lossless {
            let mut cfg2 = cfg.clone();
            cfg2.coded_lossless = true;
            cfg2.all_lossless = true;
            let mut rb2 = ReadBitBuffer::new(frame_payload);
            let mut p2 = read_uncompressed_header(&mut rb2, &cfg2);
            p2.coded_lossless = true;
            p2.all_lossless = true;
            p2
        } else {
            probe
        };
        assert!(!p.prefix.show_existing_frame);
        assert_eq!(p.prefix.frame_type, 0, "frame_type must be KEY");
        if let Some(grain) = film_grain {
            // The bootstrap MUST be a film-grain stream (its seq header carries
            // the present bit the C encoder set) — no seq bootstrap leak, we just
            // confirm the config we are matching.
            assert!(
                seq.film_grain_params_present,
                "{}: port_encode_film_grain needs a film-grain bootstrap",
                self.label
            );
            let mut g = grain.clone();
            // Context fields are NOT in the grain table — set from THIS cell's
            // config, exactly as C derives them from the seq/frame header.
            g.monochrome = mono;
            g.subsampling_x = ss_x as i32;
            g.subsampling_y = ss_y as i32;
            g.is_inter_frame = false; // KEY frame
            p.film_grain = g;
            p.film_grain_params_present = true;
        }
        // MULTI-TILE (KB-31). `av1_get_tile_limits` (tile_common.c) makes more
        // than one tile MANDATORY once either
        //   * `sb_cols > MAX_TILE_WIDTH >> sb_size_log2` (a frame wider than
        //     4096 px at SB64), or
        //   * `sb_cols * sb_rows > MAX_TILE_AREA >> 2*sb_size_log2` (2304 SB64s
        //     ~ 9.44 MP),
        // so `min_log2_tiles > 0` and libaom's own uniform-spacing default
        // (`tile_columns == 0`) still resolves `log2_cols + log2_rows >= 1`.
        // This harness asserted `tiles_log2 == 0` and therefore PANICKED on every
        // frame at or above that threshold, at every speed (issue #6). The
        // per-tile walk below is the composition of two independently
        // byte-proven pieces: the per-tile `pack_tile` + tile-bound isolation
        // (`aom-encode/tests/encoder_gate_multitile.rs`) and the derived
        // multi-tile header + tile-group assembler
        // (`aom-encode/tests/obu_assemble_multitile_diff.rs`).
        let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
        let n_tile_cols = p.tile_info.cols;
        let n_tile_rows = p.tile_info.rows;
        assert_eq!(
            n_tile_cols * n_tile_rows,
            1usize << tiles_log2,
            "{}: uniform-spacing tile grid must be 2^log2_cols x 2^log2_rows",
            self.label
        );
        let col_start_sb = p.tile_info.col_start_sb;
        let row_start_sb = p.tile_info.row_start_sb;
        let allintra = self.usage == 2;

        // Seq-level toggles (`--enable-filter-intra` / `--enable-intra-edge-
        // filter` are SEQUENCE-header bits, encoder.c:646-647): the port side
        // is driven by the KNOBS below (no bootstrap flow); the C stream's
        // seq header must agree or the two sides encode different configs.
        assert_eq!(
            s.enable_filter_intra, knobs.enable_filter_intra,
            "{}: bootstrap seq header enable_filter_intra != knob",
            self.label
        );
        assert_eq!(
            s.enable_intra_edge_filter, knobs.enable_intra_edge_filter,
            "{}: bootstrap seq header enable_intra_edge_filter != knob",
            self.label
        );
        // `--reduced-tx-type-set` is a FRAME-header bit (encodeframe.c:2712)
        // the port parses from the bootstrap; the knob must agree (the
        // search + pack read the parsed bit — config, not a per-block
        // decision, so no bootstrap leak).
        assert_eq!(
            p.reduced_tx_set_used, knobs.reduced_tx_type_set,
            "{}: bootstrap frame header reduced_tx_set_used != knob",
            self.label
        );
        // `--enable-tx-size-search=0` → the frame codes TX_MODE_LARGEST
        // (select_tx_mode via tx_size_search_level 3): knob OFF must never
        // yield a SELECT header. The converse does NOT hold — with the knob
        // ON, C post-hoc demotes SELECT to LARGEST when the coded frame had
        // ZERO tx splits (av1_encode_frame's txb_split_count == 0 arm, the
        // KB-10 cq63 shape) — so a LARGEST header is legal either way.
        assert!(
            knobs.enable_tx_size_search || !p.tx_mode_select,
            "{}: --enable-tx-size-search=0 but the bootstrap header codes              TX_MODE_SELECT",
            self.label
        );
        // `--cdf-update-mode=0` → the KEY header codes disable_cdf_update=1
        // (av1/encoder/encoder.c cdf-update-mode case 0).
        assert_eq!(
            p.prefix.disable_cdf_update,
            knobs.cdf_update_mode == 0,
            "{}: bootstrap frame header disable_cdf_update != knob",
            self.label
        );

        let qindex = p.quant.base_qindex;
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

        // Frame QM levels (`av1_set_quantizer`, av1_quantize.c: the `is_allintra`
        // arm selects `aom_get_qmlevel_allintra` for BOTH planes). Derived from
        // the knob's requested (qm_min, qm_max) and CROSS-CHECKED against the
        // levels the reference encoder signalled in the bootstrap frame header —
        // a wiring witness that fails before any byte comparison can. Mirrors
        // `encoder_gate_chroma_ss_e2e::run_case_ext`.
        let qm_levels = if let Some((qm_min, qm_max)) = knobs.qm {
            assert!(
                p.quant.using_qmatrix,
                "{}: knobs.qm asked for QM but the reference stream did not signal \
                 using_qmatrix (drive the C side with EncodeCell::c_encode_qm)",
                self.label
            );
            let ly = aom_get_qmlevel_allintra(qindex, qm_min, qm_max);
            let lu = aom_get_qmlevel_allintra(qindex + p.quant.u_ac_delta_q, qm_min, qm_max);
            let lv = if cc.separate_uv_delta_q {
                aom_get_qmlevel_allintra(qindex + p.quant.v_ac_delta_q, qm_min, qm_max)
            } else {
                lu
            };
            assert_eq!(
                [ly, lu, lv],
                [
                    p.quant.qmatrix_level_y,
                    p.quant.qmatrix_level_u,
                    p.quant.qmatrix_level_v
                ],
                "{}: derived qmatrix_level_{{y,u,v}} must match the reference header's \
                 signalled levels",
                self.label
            );
            Some([ly as usize, lu as usize, lv as usize])
        } else {
            assert!(
                !p.quant.using_qmatrix,
                "{}: reference stream signals using_qmatrix but knobs.qm is None",
                self.label
            );
            None
        };

        let kf_write = KfFrameContext::default_for_qindex(qindex);
        let real = derive_real_costs(&kf_write, knobs.enable_filter_intra, None);
        let rdmult = av1_compute_rd_mult_based_on_qindex(
            bd,
            FrameUpdateType::Kf,
            qindex,
            TuneMetric::Psnr,
            if allintra {
                EncMode::Allintra
            } else {
                EncMode::Good
            },
        );

        // Partial-SB support: CEIL the SB walk and replicate-extend the
        // source into the SB-aligned overhang (the chroma_ss_e2e recipe).
        let (cw, ch) = if mono {
            (0, 0)
        } else {
            ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
        };
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
        let mut src_y_strided = vec![0u16; stride * buf_h];
        for r in 0..h {
            src_y_strided[r * stride..r * stride + w].copy_from_slice(&self.y[r * w..r * w + w]);
        }
        extend_plane(&mut src_y_strided, w, h);
        let mut src_u_strided = vec![0u16; stride * buf_h];
        let mut src_v_strided = vec![0u16; stride * buf_h];
        if !mono {
            for r in 0..ch {
                src_u_strided[r * stride..r * stride + cw]
                    .copy_from_slice(&self.u[r * cw..r * cw + cw]);
                src_v_strided[r * stride..r * stride + cw]
                    .copy_from_slice(&self.v[r * cw..r * cw + cw]);
            }
            extend_plane(&mut src_u_strided, cw, ch);
            extend_plane(&mut src_v_strided, cw, ch);
        }

        // Tile geometry, raster (tile-row-major) order: each entry is
        // `(mi_row_start, mi_col_start, mi_row_end, mi_col_end, n_sb_rows,
        // n_sb_cols)`. The mi ENDS are clamped to the frame exactly like C's
        // `av1_tile_set_row`/`_col` (`tile->mi_row_end = AOMMIN(..., mi_rows)`,
        // av1/common/tile_common.c). Single tile => one entry covering the frame,
        // which is what every pre-existing gate encodes.
        //
        // Derived HERE, ahead of the delta-q replays below, because those replays
        // have to walk in tile order (see `replay_sb_qindex_tile_order`).
        let tile_grid: Vec<(i32, i32, i32, i32, i32, i32)> = (0..n_tile_rows)
            .flat_map(|trow| {
                (0..n_tile_cols).map(move |tcol| {
                    let r0 = row_start_sb[trow] << mib_size_log2;
                    let r1 = (row_start_sb[trow + 1] << mib_size_log2).min(mi_rows);
                    let c0 = col_start_sb[tcol] << mib_size_log2;
                    let c1 = (col_start_sb[tcol + 1] << mib_size_log2).min(mi_cols);
                    (
                        r0,
                        c0,
                        r1,
                        c1,
                        row_start_sb[trow + 1] - row_start_sb[trow],
                        col_start_sb[tcol + 1] - col_start_sb[tcol],
                    )
                })
            })
            .collect();
        assert_eq!(
            tile_grid.iter().map(|t| (t.4 * t.5) as usize).sum::<usize>(),
            (n_sb_x * n_sb_y) as usize,
            "{}: the tile grid must partition every superblock exactly once",
            self.label
        );

        // --deltaq-mode=3 (DELTA_Q_PERCEPTUAL_AI, family C5): build the wiener
        // map + derive the per-SB qindex and the delta_q header fields entirely
        // port-side (the map is never copied from the bootstrap; the header
        // fields are cross-checked below). bd8, dims a multiple of 8px.
        let weber_map = knobs.deltaq_mode3.then(|| {
            aom_encode::allintra_vis::av1_set_mb_wiener_variance(
                &src_y_strided,
                0,
                stride,
                mi_rows,
                mi_cols,
                qindex,
                bd,
                &quants,
                &deq,
                sb_block,
                sb_mi,
                !knobs.enable_intra_edge_filter,
            )
        });
        let (dq3_sb_qindex, dq3_present, dq3_res) = if let Some(map) = &weber_map {
            // delta_q_present = (any SB produced a nonzero delta) && qindex > 0
            // (bitstream.c:4287 resets it when deltaq_used == 0). Replays the
            // per-SB derivation to compute `deltaq_used` (delta_qindex != 0).
            let res = aom_encode::allintra_vis::DELTA_Q_RES_PERCEPTUAL;
            let (per_sb, used) = replay_sb_qindex_tile_order(
                &tile_grid,
                n_sb_x,
                sb_mi,
                qindex,
                |mi_row, mi_col, running| {
                    aom_encode::allintra_vis::setup_delta_q_perceptual_ai(
                        map, qindex, bd, res, sb_mi, mi_row, mi_col, running,
                    )
                },
            );
            (per_sb, used && qindex > 0, res)
        } else {
            (Vec::new(), false, 0)
        };
        if knobs.deltaq_mode3 {
            // Cross-check the port-derived header fields against the real
            // --deltaq-mode=3 bootstrap (leak-free: the port DERIVES; the
            // assert only confirms agreement), then write the derived values.
            assert_eq!(
                p.delta_q.delta_q_present, dq3_present,
                "{}: derived delta_q_present must match the real --deltaq-mode=3 header",
                self.label
            );
            if dq3_present {
                assert_eq!(
                    p.delta_q.delta_q_res, dq3_res,
                    "{}: derived delta_q_res must match the real header",
                    self.label
                );
                p.delta_q.delta_q_res = dq3_res;
            }
            p.delta_q.delta_q_present = dq3_present;
        }

        // --deltaq-mode=2 (DELTA_Q_PERCEPTUAL, wavelet AC energy): replay the
        // per-SB wavelet-energy qindex to derive delta_q_present + deltaq_used,
        // cross-check the real --deltaq-mode=2 header (leak-free), then write.
        // `is_screen_content_type` (the rate-model enumerator) tracks
        // `allow_screen_content_tools` on this (photographic, non-screen)
        // envelope; a mismatch would perturb the qindex and fail the byte gate.
        let dq2_screen = p.allow_screen_content_tools;
        let (dq2_sb_qindex, dq2_present, dq2_res) = if knobs.deltaq_mode2 {
            let res = aom_encode::allintra_vis::DELTA_Q_RES_PERCEPTUAL;
            // SB pixel extent (64 or 128 for sb128); num_pels_log2 = log2(sb_px²).
            let num_pels_log2 = (sb_px * sb_px).trailing_zeros();
            let (per_sb, used) = replay_sb_qindex_tile_order(
                &tile_grid,
                n_sb_x,
                sb_mi,
                qindex,
                |mi_row, mi_col, running| {
                    let sb_off = mi_row as usize * 4 * stride + mi_col as usize * 4;
                    aom_encode::allintra_vis::setup_delta_q_perceptual(
                        &src_y_strided,
                        sb_off,
                        stride,
                        bd,
                        qindex,
                        dq2_screen,
                        sb_px,
                        sb_px,
                        num_pels_log2,
                        res,
                        running,
                    )
                },
            );
            (per_sb, used && qindex > 0, res)
        } else {
            (Vec::new(), false, 0)
        };
        if knobs.deltaq_mode2 {
            assert_eq!(
                p.delta_q.delta_q_present, dq2_present,
                "{}: derived delta_q_present must match the real --deltaq-mode=2 header",
                self.label
            );
            if dq2_present {
                assert_eq!(
                    p.delta_q.delta_q_res, dq2_res,
                    "{}: derived delta_q_res must match the real header",
                    self.label
                );
                p.delta_q.delta_q_res = dq2_res;
            }
            p.delta_q.delta_q_present = dq2_present;
        }

        // --delta-lf-mode=1: `enable_deltalf_mode = (deltaq_mode != NO_DELTA_Q)
        // && deltalf_mode` (cx_iface.c:1326) -> `delta_lf_present_flag`
        // (encodeframe.c:2321). Rides on a firing delta-q mode; the per-SB
        // delta_lf is derived from delta_qindex in pack_leaf. DEFAULT_DELTA_LF_RES
        // = 2, DEFAULT_DELTA_LF_MULTI = 0. Cross-check the real header, then write.
        let dlf_present = knobs.delta_lf_mode && (dq3_present || dq2_present);
        if knobs.delta_lf_mode {
            assert_eq!(
                p.delta_q.delta_lf_present, dlf_present,
                "{}: derived delta_lf_present must match the real --delta-lf-mode=1 header",
                self.label
            );
            if dlf_present {
                assert_eq!(
                    p.delta_q.delta_lf_res, 2,
                    "{}: real delta_lf_res must be DEFAULT_DELTA_LF_RES (2)",
                    self.label
                );
                assert!(
                    !p.delta_q.delta_lf_multi,
                    "{}: real delta_lf_multi must be DEFAULT_DELTA_LF_MULTI (single)",
                    self.label
                );
                p.delta_q.delta_lf_res = 2;
                p.delta_q.delta_lf_multi = false;
            }
            p.delta_q.delta_lf_present = dlf_present;
        }

        let speed = self.speed;
        let mut sf = SpeedFeatures::set_allintra(speed, p.allow_screen_content_tools, false);
        // The modelled arms of set_allintra_speed_feature_framesize_dependent
        // (speed_features.c:166) — currently the `is_4k_or_larger`
        // default_min_partition_size arm (KB-19). Framesize-blind by design in
        // `set_allintra`; applied here from the frame's real dimensions.
        if allintra {
            sf.apply_allintra_framesize_dependent(w, h, speed);
            // The ALLINTRA-reachable arms of av1_set_speed_features_qindex_
            // dependent (speed_features.c:2872) — C's third pass, run after
            // BOTH set_allintra cascades (encoder.c:3114). At speed 0 its
            // `is_720p_or_larger && base_qindex <= 128` arm raises
            // perform_coeff_opt and the rectangular intra tx-size init depth
            // (KB-22); it is a no-op below 720p, which is every other cell in
            // this harness.
            sf.apply_allintra_qindex_dependent(w, h, qindex, speed);
        }
        // Framesize-dependent tx-type stats prune (set_allintra_speed_feature_
        // framesize_dependent, speed_features.c:261/299): ALLINTRA sets
        // prune_tx_type_using_stats = 1 at speed>=2 / 2 at speed>=4, but ONLY
        // when is_480p_or_larger (min(w,h) >= 480). 0 on every sub-480p gate ->
        // byte-inert there (the SpeedFeatures setter itself is framesize-blind).
        sf.prune_tx_type_using_stats =
            if allintra && w.min(h) >= 480 && !knobs.disable_tx_stats_prune {
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
        let mut env = SbEncodeEnv {
            ref_frame: None,
            sb_size: sb_block,
            mi_rows,
            mi_cols,
            // `cm->width`/`cm->height` — the TRUE crop (KB-28).
            frame_width: w as i32,
            frame_height: h as i32,
            tile_row_start: tile_grid[0].0,
            tile_col_start: tile_grid[0].1,
            tile_row_end: tile_grid[0].2,
            tile_col_end: tile_grid[0].3,
            monochrome: mono,
            ss_x,
            ss_y,
            bd,
            lossless: p.coded_lossless,
            reduced_tx_set_used: p.reduced_tx_set_used,
            // Knob-driven (seq bit asserted equal above — no bootstrap flow).
            disable_edge_filter: !knobs.enable_intra_edge_filter,
            filter_type: 0,
            stride,
            src_y: &src_y_strided,
            src_u: &src_u_strided,
            src_v: &src_v_strided,
            base_y: 0,
            base_uv: 0,
            rows_y: &rows_y,
            rows_u: &rows_u,
            rows_v: &rows_v,
            rdmult,
            sharpness: 0,
            // init_rd_sf: lossless forces NO_TRELLIS_OPT for every knob
            // value; else the knob maps per trellis_opt_of_knob. The stock
            // default (3, NO_ESTIMATE_YRD) is pack-equivalent to the prior
            // hardcoded FullTrellisOpt (is_trellis_used(OUTPUT_ENABLED) is
            // true for both; estimate_yrd_for_sb is inter-only).
            enable_optimize_b: if p.coded_lossless {
                TrellisOptType::NoTrellisOpt
            } else {
                trellis_opt_of_knob(knobs.disable_trellis_quant)
            },
            // QM-off by default (the allintra default); `knobs.qm` drives the
            // QM-on path (levels derived + cross-checked above).
            qm_levels,
            tune: Default::default(),
            deltaq: if dq3_present {
                Some(aom_encode::encode_sb::DeltaQFrameCtx {
                    quants: &quants,
                    deq: &deq,
                    base_qindex: qindex,
                    delta_q_res: dq3_res,
                    deltaq_strength: 0,
                    perceptual_ai: weber_map.as_ref(),
                    perceptual_wavelet: None,
                    sb_mi,
                    delta_lf_present: dlf_present,
                })
            } else if dq2_present {
                Some(aom_encode::encode_sb::DeltaQFrameCtx {
                    quants: &quants,
                    deq: &deq,
                    base_qindex: qindex,
                    delta_q_res: dq2_res,
                    deltaq_strength: 0,
                    perceptual_ai: None,
                    perceptual_wavelet: Some(dq2_screen),
                    sb_mi,
                    delta_lf_present: dlf_present,
                })
            } else {
                None
            },
            use_chroma_trellis_rd_mult: allintra,
            coeff_costs_y: &real.coeff_costs_y,
            coeff_costs_uv: &real.coeff_costs_uv,
            txfm_partition_costs: [[0i32; 2]; 21],
            tx_type_costs: &real.tx_type_costs_y,
        };
        // CLI tx-type toggles override the sf-derived policy (C reads oxcf
        // directly in get_tx_mask, stage-independent; the MODE_EVAL
        // default-tx-only OR happens in partition_pick's stage derivation
        // from this policy).
        let pol = {
            let mut pol = sf.tx_type_search_policy(false, 0);
            pol.enable_flip_idtx = knobs.enable_flip_idtx;
            pol.use_intra_dct_only = knobs.use_intra_dct_only;
            pol.use_default_intra_tx_type |= knobs.use_intra_default_tx_only;
            pol.enable_tx_size_search = knobs.enable_tx_size_search;
            // `--disable-trellis-quant` (init_rd_sf): the search runs
            // trellis iff is_trellis_used(opt, DRY_RUN_NORMAL) — FULL(0)/
            // NO_ESTIMATE_YRD(3) yes, NO(1)/FINAL_PASS(2) no.
            pol.skip_trellis = !aom_encode::encode_intra::is_trellis_used(
                trellis_opt_of_knob(knobs.disable_trellis_quant),
                false,
            );
            pol
        };
        // Chroma-loop tool toggles ride on the UvLoopPolicy (the sf-driven
        // fields keep their speed-0 values; the speed>=3 chroma rebuild in
        // partition_pick spreads `..cfg.uv_lp.clone()`, so these survive).
        let uv_lp = UvLoopPolicy {
            enable_diagonal_intra: knobs.enable_diagonal_intra,
            enable_directional_intra: knobs.enable_directional_intra,
            enable_smooth_intra: knobs.enable_smooth_intra,
            enable_paeth_intra: knobs.enable_paeth_intra,
            enable_cfl_intra: knobs.enable_cfl_intra,
            enable_angle_delta: knobs.enable_angle_delta,
            ..UvLoopPolicy::speed0_allintra()
        };
        // Intrabc (screen content): the frame header codes `allow_intrabc`
        // (from the `--enable-intrabc=1` C encode); the port runs the DV search
        // when the knob is also on. `av1_init_search_range(max(w,h))` =
        // mv_step_param; the hash table is built ONCE from the SOURCE luma.
        let init_search_range = |size: i32| -> usize {
            let size = size.max(16);
            let mut sr = 0usize;
            while (size << sr) < 1023 {
                sr += 1;
            }
            sr.min(9)
        };
        // Witness note: when the header codes allow_intrabc but the knob is off
        // (PORT-off anti-vacuous case), the search doesn't run yet the pack still
        // writes use_intrabc=0 flags (via `PackCfg::allow_intrabc`).
        // KB-41 root #7: C decides `allow_intrabc` BEFORE the search from the
        // anti-aliasing-aware screen census (encoder.c:2416) and only FLIPS the
        // header bit to 0 after the frame when no block used IntraBC
        // (encodeframe.c:2442) — the search still paid `intrabc_cost[0]` on every
        // intra candidate and ran the DV search. The oracle header carries that
        // FINAL bit, so the search-time decision is re-derived by the ported
        // detector; its screen-tools half must agree with the header (a
        // differential gate on the detector port), and a header that codes
        // allow_intrabc=1 implies the search-time decision was 1.
        // av1_set_screen_content_options (encoder.c:2440-2480): the shim's
        // screen encodes set only the palette/intrabc knobs (no
        // `--tune-content=screen`, seq force = SELECT), so the decision is the
        // detector's — except under `use_nonrd_pick_mode && !hybrid_intra_pickmode`
        // (allintra speed 9), where detection is skipped and both flags are 0
        // (KB-41 root #13; kb35's speed-9 control).
        let sct = if s.force_screen_content_tools != 2 {
            // seq-forced arm (:2443-2447): the forced bit is the decision.
            aom_encode::screen_detect::ScreenContentDecision::forced(
                s.force_screen_content_tools != 0,
            )
        } else if knobs.tune_content_screen {
            // `--tune-content=screen` arm (:2449-2455).
            aom_encode::screen_detect::ScreenContentDecision::tuned_screen()
        } else if sf.use_nonrd_pick_mode && sf.hybrid_intra_pickmode == 0 {
            aom_encode::screen_detect::ScreenContentDecision::detection_disabled()
        } else {
            aom_encode::screen_detect::estimate_screen_content_antialiasing_aware(
                &src_y_strided,
                0,
                stride,
                w,
                h,
                bd as u8,
                sf.screen_detection_mode2_fast_detection,
            )
        };
        assert_eq!(
            sct.allow_screen_content_tools, p.allow_screen_content_tools,
            "{w}x{h}: the ported screen-content decision disagrees with the oracle header's \
             allow_screen_content_tools (palette={} intrabc={} photo={} fast={}). If the \
             detector said 0 and the header says 1, the remaining C arm is \
             av1_determine_sc_tools_with_encoding's two-pass trial encode (encoder.c:3312, \
             live on allintra below speed 8) — NOT ported; or the cell was encoded with \
             --tune-content=screen without declaring `ToggleKnobs::tune_content_screen`",
            sct.count_palette, sct.count_intrabc, sct.count_photo,
            sf.screen_detection_mode2_fast_detection
        );
        assert!(
            !p.allow_intrabc || sct.allow_intrabc,
            "{w}x{h}: the oracle header codes allow_intrabc=1 but the ported detector's \
             search-time decision is 0 (palette={} intrabc={} photo={})",
            sct.count_palette, sct.count_intrabc, sct.count_photo
        );
        let search_allow_intrabc = sct.allow_intrabc;
        // rd_pick_intrabc_mode_sb's frame-wide gates (rdopt.c:3432-3434):
        // `!av1_allow_intrabc || !enable_intrabc || !mv_sf.use_intrabc ||
        // rt_sf.use_nonrd_pick_mode` -> no DV search for any block. The search
        // flag stays `search_allow_intrabc` (the search-ctx CDF still adapts,
        // root #8); only the search itself is off. KB-41 root #10.
        let run_intrabc_search = search_allow_intrabc
            && knobs.enable_intrabc
            && sf.mv_sf.use_intrabc
            && !sf.use_nonrd_pick_mode;
        let ibc_hash = run_intrabc_search.then(|| {
            aom_encode::intrabc_search::build_intrabc_hash_table(
                &src_y_strided,
                0,
                stride,
                w,
                h,
                bd > 8,
                64,
            )
        });
        let ibc_dv_costs = ibc_hash.as_ref().map(|_| {
            aom_encode::intrabc_search::fill_dv_costs(
                &kf_write.ndvc_joints,
                &kf_write.ndvc_comp0,
                &kf_write.ndvc_comp1,
            )
        });
        let ibc_txfm_costs = aom_encode::intrabc_search::fill_txfm_partition_costs(
            &aom_encode::intrabc_search::DEFAULT_TXFM_PARTITION_CDF,
        );
        let ibc_frame = match (ibc_hash.as_ref(), ibc_dv_costs.as_ref()) {
            (Some(hash), Some(dv_costs)) => Some(IntrabcFrameCfg {
                hash,
                dv_costs,
                txfm_partition_costs: ibc_txfm_costs,
                error_per_bit: (rdmult >> 6).max(1),
                sad_per_bit: av1_set_sad_per_bit(qindex, bd),
                mv_step_param: init_search_range(w.max(h) as i32),
                mv_sf: sf.mv_sf,
            }),
            _ => None,
        };

        let pick_cfg = PickFrameCfg {
            // KB-32 root #1. The KEY variance partitioner's two
            // `force_large_partition_blocks_intra` arms (var_based_part.c:
            // 539-544 and :552-554) were dropped. Carry the RESOLVED
            // frame-level values down — the walk must not re-derive them
            // from the frame dimensions it has (playbook §13); `pack_tile`
            // only ever sees mi-ALIGNED dimensions, so a re-derivation there
            // would be wrong on any crop within 3 px of the 720 boundary.
            fs_sf: aom_encode::partition_pick::FrameSizeSf {
                vbp: aom_encode::var_part::VbpSf {
                    force_large_partition_blocks_intra: sf.force_large_partition_blocks_intra != 0,
                    var_part_split_threshold_shift: sf.var_part_split_threshold_shift,
                    allintra,
                },
                // `is_4k_or_larger` (speed_features.c:172) — the predicate
                // the speed-9 cost-upd arm keys on (:648-651). KB-32 root #2.
                is_4k_or_larger: w.min(h) >= 2160,
            },
            inter: None,
            intra_tools: aom_encode::partition_pick::IntraToolCfg {
                enable_diagonal_intra: knobs.enable_diagonal_intra,
                enable_directional_intra: knobs.enable_directional_intra,
                enable_smooth_intra: knobs.enable_smooth_intra,
                enable_paeth_intra: knobs.enable_paeth_intra,
                enable_angle_delta: knobs.enable_angle_delta,
            },
            mode_costs: &real.mode_costs,
            tx_size_costs: &real.tx_size_costs,
            skip_costs: &real.skip_costs,
            tx_type_costs_y: &real.tx_type_costs_y,
            pol: &pol,
            uv_lp: &uv_lp,
            intra_uv_mode_cost: &real.mode_costs.intra_uv_mode_cost,
            cfl_costs: &real.cfl_costs,
            partition_costs: &real.partition_costs,
            partition_cdfs: &real.partition_cdf,
            allintra,
            speed,
            qindex,
            enable_filter_intra: knobs.enable_filter_intra,
            enable_tx64: knobs.enable_tx64,
            enable_rect_tx: knobs.enable_rect_tx,
            intra_pruning_with_hog: if allintra {
                sf.intra_pruning_with_hog != 0
            } else {
                true
            },
            enable_rect_partitions: knobs.enable_rect_partitions,
            less_rectangular_check_level: if allintra {
                // av1_set_speed_features_qindex_dependent runs AFTER the
                // allintra setters and overrides at speed 3 ONLY:
                // `(base_qindex >= 170) ? 1 : 2` (speed_features.c:3032-3034).
                // Its speed<=2 (:3029, ->1) and speed>=4 (:3048, ->2) arms
                // equal the allintra values, so only the speed-3 arm is live.
                if self.speed == 3 {
                    if qindex >= 170 { 1 } else { 2 }
                } else {
                    sf.less_rectangular_check_level
                }
            } else {
                i32::from(allintra)
            },
            // C's set_max_min_partition_size (partition_strategy.h:214):
            // min(sf default, CLI dim, sb). `sb_block` is the live SB size
            // (BLOCK_64X64 or, at --sb-size=128, BLOCK_128X128).
            max_partition_size: knobs.max_partition_bsize(sf.default_max_partition_size, sb_block),
            min_partition_size: knobs.min_partition_bsize(sf.default_min_partition_size, sb_block),
            enable_1to4_partitions: knobs.enable_1to4_partitions,
            enable_ab_partitions: knobs.enable_ab_partitions,
            allow_screen_content_tools: p.allow_screen_content_tools,
            qm_levels,
            palette_costs: knobs.enable_palette.then_some(&real.palette_costs),
            intrabc: ibc_frame,
        };
        let pack_cfg = aom_encode::pack::PackCfg {
            enable_filter_intra: knobs.enable_filter_intra,
            tx_mode_is_select: p.tx_mode_select,
            signal_gate: qindex > 0,
            allow_update_cdf: !p.prefix.disable_cdf_update,
            base_qindex: qindex,
            delta_q_present: dq3_present || dq2_present,
            delta_q_res: if dq3_present {
                dq3_res
            } else if dq2_present {
                dq2_res
            } else {
                0
            },
            allow_screen_content_tools: p.allow_screen_content_tools,
            allow_intrabc: p.allow_intrabc,
            search_allow_intrabc,
            // KB-41 root #12: update_stats' tx-size gate is the SEARCH-time
            // DEFAULT_EVAL tx mode (rdopt_utils.h:494), not the final header.
            search_tx_mode_is_select: !coded_lossless
                && (knobs.enable_tx_size_search || sf.use_nonrd_pick_mode),
        };

        let mut recon_y = src_y_strided.clone();
        let mut recon_u = src_u_strided.clone();
        let mut recon_v = src_v_strided.clone();
        // Per-tile pack in raster order. AV1 tiles are entropy-independent, so
        // every tile gets a FRESH `KfFrameContext` + `OdEcEnc`, exactly like C's
        // `av1_init_tile_data` (`cpi->tile_data[i].tctx = *cm->fc`). The
        // reconstruction buffers are FRAME-level and shared: each tile writes
        // only its own region, and the tile mi bounds in `env` are what stop
        // intra prediction / tx-size context / the RD search from reading across
        // a tile edge.
        let mut tile_payloads: Vec<Vec<u8>> = Vec::with_capacity(tile_grid.len());
        // Frame-raster SB trees, reassembled from the per-tile walks so the
        // loop-filter grid below (which indexes trees by FRAME SB raster) is
        // tile-layout-independent.
        let mut frame_trees: Vec<Option<SbTree>> = vec![None; (n_sb_x * n_sb_y) as usize];
        let mut port_intrabc_used = false;
        for (t, &(mi_row_start, mi_col_start, mi_row_end, mi_col_end, n_sb_rows, n_sb_cols)) in
            tile_grid.iter().enumerate()
        {
            env.tile_row_start = mi_row_start;
            env.tile_col_start = mi_col_start;
            env.tile_row_end = mi_row_end;
            env.tile_col_end = mi_col_end;
            let mut kf_tile = KfFrameContext::default_for_qindex(qindex);
            let mut enc = OdEcEnc::new();
            let trees = pack_tile(
                &mut enc,
                &env,
                &pick_cfg,
                &pack_cfg,
                &mut kf_tile,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                mi_row_start,
                mi_col_start,
                n_sb_rows,
                n_sb_cols,
                sb_mi,
                sb_block,
            );
            port_intrabc_used |= trees.iter().any(SbTree::any_intrabc);
            assert_eq!(
                trees.len(),
                (n_sb_rows * n_sb_cols) as usize,
                "{}: pack_tile must walk every SB of tile {t}",
                self.label
            );
            let (sb_row0, sb_col0) = (mi_row_start / sb_mi, mi_col_start / sb_mi);
            for (i, tree) in trees.into_iter().enumerate() {
                let sb_r = sb_row0 + i as i32 / n_sb_cols;
                let sb_c = sb_col0 + i as i32 % n_sb_cols;
                frame_trees[(sb_r * n_sb_x + sb_c) as usize] = Some(tree);
            }
            tile_payloads.push(enc.done().to_vec());
        }
        // KB-41 root #7 (the other direction): C writes allow_intrabc=1 only if a
        // block used IntraBC. If the port's search chose IntraBC where the oracle's
        // did not (or vice versa) the streams cannot match — fail loud with the
        // actionable half instead of a bare byte mismatch. Only meaningful when the
        // port actually searched IntraBC (the PORT-off witness leaves it off).
        if knobs.enable_intrabc && search_allow_intrabc {
            assert_eq!(
                port_intrabc_used, p.allow_intrabc,
                "{w}x{h}: IntraBC usage differs — port used it: {port_intrabc_used}, \
                 oracle header allow_intrabc: {}",
                p.allow_intrabc
            );
        }
        let trees: Vec<SbTree> = frame_trees
            .into_iter()
            .map(|t| t.expect("every superblock belongs to exactly one tile"))
            .collect();

        // Port-derived loop-filter level. allintra `lpf_pick` is DUAL for
        // speed 0..=3 and NON_DUAL for speed >= 4 (speed_features.c:496).
        let mut mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb_x, sb_mi, sb_block);
        // --delta-lf-mode=1: the LF trial deblock (and thus the derived
        // filter_level) reads per-SB delta_lf via get_filter_level. The per-SB
        // qindex the pack used was already replayed above, in TILE order with the
        // per-tile base reset (`replay_sb_qindex_tile_order`), and indexed by
        // FRAME SB raster — which is exactly what `stamp_lf_delta_lf` indexes by.
        // Deriving delta_lf from that shared vector (rather than a second,
        // independently-ordered walk) is what keeps the LF grid == the coded
        // tile's delta_lf once there is more than one tile.
        if dlf_present {
            let sb_qindex = if dq3_present {
                &dq3_sb_qindex
            } else {
                &dq2_sb_qindex
            };
            assert_eq!(
                sb_qindex.len(),
                (n_sb_x * n_sb_y) as usize,
                "{}: the per-SB delta-q replay must cover every superblock",
                self.label
            );
            let dlf_per_sb: Vec<i32> = sb_qindex
                .iter()
                .map(|&adj| {
                    // delta_lf_from_base = ((delta_qindex/4 + res/2) & ~(res-1)),
                    // res = DEFAULT_DELTA_LF_RES = 2, clamped (encodeframe.c:380).
                    let delta_qindex = adj - qindex;
                    ((delta_qindex / 4 + 1) & !1).clamp(-63, 63)
                })
                .collect();
            aom_encode::lf_search::stamp_lf_delta_lf(
                &mut mi_grid,
                &dlf_per_sb,
                mi_rows,
                mi_cols,
                n_sb_x,
                sb_mi,
            );
        }
        let lf_frame = LfSearchFrame {
            recon_y: &recon_y,
            recon_u: &recon_u,
            recon_v: &recon_v,
            src_y: &src_y_strided,
            src_u: &src_u_strided,
            src_v: &src_v_strided,
            stride,
            crop_width: w as u32,
            crop_height: h as u32,
            ss_x,
            ss_y,
            bd: i32::from(bd),
            monochrome: mono,
            mi: &mi_grid,
            mi_rows,
            mi_cols,
            delta_lf_present: dlf_present,
        };
        // `lpf_sf.lpf_pick`: LPF_PICK_FROM_FULL_IMAGE (DUAL) at allintra speed
        // 0..=3, ..._NON_DUAL at 4/5 (speed_features.c:496), and the CLOSED-FORM
        // LPF_PICK_FROM_Q at speed >= 6 (:559) — no search at all, the level is
        // a fit on the AC quantizer. Without the >= 6 arm this harness ran the
        // search at every speed, so EVERY speed >= 6 cell diverged in the frame
        // header's deblock levels while its tile payload was already
        // byte-identical (measured 2026-07-30: diag128 cq32 cpu-6, port 1297 B
        // vs C 1297 B, first diff at payload byte 2). The aom-encode e2e gate
        // has carried this arm since the speed-6 landing
        // (`encoder_gate_e2e_byte_match.rs`, `pick_filter_level_from_q`,
        // oracle-validated by `speed6_prep_lf_from_q_matches_real_aomenc`).
        //
        // `loopfilter_frame` (encoder.c:2875-2886) calls `av1_pick_filter_level`
        // only when `is_loopfilter_used(cm)` — `!coded_lossless &&
        // !tiles.large_scale` (encoder.h:4419-4421) — so at coded-lossless the
        // search never runs and `cm->lf` keeps its zeroed levels. The frame
        // header writer skips the whole loop-filter block there anyway
        // (`if (!all_lossless) { if (!coded_lossless) encode_loopfilter(..) }`,
        // aom-dsp header.rs:1536-1538), so this is byte-inert — but running a
        // full-image deblock search whose result C never computes is both a
        // waste and a lie about what the port models.
        let derived_lf = if p.coded_lossless {
            // `cm->lf` is zero-initialised and never touched at
            // coded-lossless (encoder.c:2884).
            aom_encode::lf_search::LoopFilterLevels {
                filter_level: [0, 0],
                filter_level_u: 0,
                filter_level_v: 0,
                sharpness: 0,
            }
        } else if allintra && speed >= 6 {
            pick_filter_level_from_q(qindex, bd, allintra, 0)
        } else {
            pick_filter_level(&lf_frame, allintra, 0, allintra && speed >= 4)
        };
        // C gates av1_pick_filter_level on `!coded_lossless && !allow_intrabc`
        // (picklpf.c); a screen-content intrabc frame forces the deblock levels
        // to 0 (the decoder does the same; LR is off ⇒ the LR stage below is
        // dead for intrabc). Otherwise use the derived levels.
        if p.allow_intrabc {
            p.loopfilter.filter_level = [0, 0];
            p.loopfilter.filter_level_u = 0;
            p.loopfilter.filter_level_v = 0;
        } else {
            p.loopfilter.filter_level = derived_lf.filter_level;
            p.loopfilter.filter_level_u = derived_lf.filter_level_u;
            p.loopfilter.filter_level_v = derived_lf.filter_level_v;
        }

        // ---- loop-restoration ENCODER stage (`--enable-restoration` parity).
        // C pipeline (encoder.c `loopfilter_frame` -> `cdef_restoration_frame`):
        // apply the picked deblock levels -> [CDEF off in this envelope] ->
        // `av1_pick_filter_restoration` on (source, deblocked recon) -> pack
        // the tile with the per-RU params interleaved at each SB root. The
        // restoration DECISIONS (frame types, unit size, per-RU params) are
        // derived by the port's own search — never copied from the bootstrap.
        // The port's DEFAULT allintra path runs the (byte-exact) LR search
        // whenever the frame's restoration is on — C's `is_restoration_used`
        // (encoder.h:4431: `enable_restoration && !all_lossless && !large_scale`).
        // Allintra's default is `enable_restoration = 1` (av1_cx_iface.c:286, NOT
        // cleared by the :3065 allintra override, kept non-realtime at :1273); at
        // speed >= 5 C clears the seq bit (speed_features.c:2754), so the parsed
        // bootstrap seq header's `enable_restoration` already encodes the speed
        // gate — reading it here IS the faithful `is_restoration_used`. The
        // explicit `lr_stage` (a `port_encode_lr` request, the C2 knob gate)
        // forces it on; the default path (`port_encode`) DERIVES it, so a plain
        // default (restoration-on) bootstrap runs the search + emits the
        // restoration syntax, matching a plain `aomenc --allintra`. Restoration-
        // off bootstraps (every `--enable-restoration=0` gate) derive `false` —
        // unchanged.
        let lr_stage = lr_stage || (s.enable_restoration && !p.coded_lossless);
        if lr_stage {
            assert!(
                s.enable_restoration,
                "the LR stage needs an enable_restoration=1 stream"
            );
            assert!(
                !p.coded_lossless,
                "is_restoration_used excludes all-lossless"
            );

            // (1) The deblocked reconstruction: the derived levels applied to
            //     a copy (`loop_filter_frame` gates itself on the Y levels,
            //     exactly like the C apply site).
            let mut db_y = recon_y.clone();
            let mut db_u = recon_u.clone();
            let mut db_v = recon_v.clone();
            {
                let lf_apply = LfParams {
                    filter_level: derived_lf.filter_level,
                    filter_level_u: derived_lf.filter_level_u,
                    filter_level_v: derived_lf.filter_level_v,
                    sharpness: 0,
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
                    y: &mut db_y,
                    y_stride: stride,
                    u: &mut db_u,
                    v: &mut db_v,
                    uv_stride: stride,
                    crop_width: w as u32,
                    crop_height: h as u32,
                    ss_x,
                    ss_y,
                    bd: i32::from(bd),
                };
                loop_filter_frame(&mut buf, &grid, &lf_apply, 0, num_planes as usize);
            }

            // (2) `av1_pick_filter_restoration`: costs = av1_fill_lr_rates
            //     over the FRAME-INIT LR CDFs (nothing adapts them before the
            //     search in C); rdmult = the frame RDMULT.
            let fc0 = KfFrameContext::default_for_qindex(qindex);
            let mut wiener_cost = [0i32; 2];
            let mut sgrproj_cost = [0i32; 2];
            let mut switchable_cost = [0i32; 3];
            cost_tokens_from_cdf(&mut wiener_cost, &fc0.wiener_restore, None);
            cost_tokens_from_cdf(&mut sgrproj_cost, &fc0.sgrproj_restore, None);
            cost_tokens_from_cdf(&mut switchable_cost, &fc0.switchable_restore, None);
            let lr_input = LrSearchInput {
                planes: if mono {
                    vec![LrPlanePixels {
                        src: &src_y_strided,
                        deblocked: &db_y,
                        cur: &db_y,
                        stride,
                    }]
                } else {
                    vec![
                        LrPlanePixels {
                            src: &src_y_strided,
                            deblocked: &db_y,
                            cur: &db_y,
                            stride,
                        },
                        LrPlanePixels {
                            src: &src_u_strided,
                            deblocked: &db_u,
                            cur: &db_u,
                            stride,
                        },
                        LrPlanePixels {
                            src: &src_v_strided,
                            deblocked: &db_v,
                            cur: &db_v,
                            stride,
                        },
                    ]
                },
                crop_width: w as i32,
                crop_height: h as i32,
                ss_x,
                ss_y,
                bit_depth: i32::from(bd),
                highbd: bd > 8,
                rdmult: i64::from(rdmult),
                dc_quant_qtx: i32::from(av1_dc_quant_qtx(qindex, 0, bd)),
                mib_size_log2: mib_size_log2 as i32,
                mi_rows,
                mi_cols,
                // `av1_pick_filter_restoration` walks tiles outer / SBs inner and
                // resets the per-RU delta-coding references at every tile start,
                // so the tile SB spans must be the REAL ones. Single tile =>
                // `[(0, n_sb_y)]` / `[(0, n_sb_x)]`, unchanged.
                tile_sb_rows: (0..n_tile_rows)
                    .map(|t| (row_start_sb[t], row_start_sb[t + 1]))
                    .collect(),
                tile_sb_cols: (0..n_tile_cols)
                    .map(|t| (col_start_sb[t], col_start_sb[t + 1]))
                    .collect(),
                wiener_restore_cost: wiener_cost,
                sgrproj_restore_cost: sgrproj_cost,
                switchable_restore_cost: switchable_cost,
                sf: if allintra {
                    lr_search_sf_allintra(speed, qindex, w, h, p.allow_screen_content_tools)
                } else {
                    lr_search_sf_good(speed, qindex, w, h, p.allow_screen_content_tools)
                },
            };
            let outcome = pick_filter_restoration(&lr_input);

            // (3) The derived frame-restoration header fields.
            p.restoration.frame_restoration_type = outcome.frame_restoration_type;
            p.restoration.restoration_unit_size = [outcome.unit_size; 3];

            // (4) Repack with the interleaved RU params when any plane
            //     restores (an all-NONE frame codes no LR symbols — the
            //     pass-1 tile bytes are already exactly right).
            if outcome
                .frame_restoration_type
                .iter()
                .any(|&t| t != LR_RESTORE_NONE)
            {
                let lr_pack = LrPackParams {
                    cfg: LrFrameConfig {
                        frame_restoration_type: outcome.frame_restoration_type,
                        unit_size: [outcome.unit_size; 3],
                        crop_width: w as i32,
                        crop_height: h as i32,
                        superres_denom: 0,
                    },
                    units: [&outcome.units[0], &outcome.units[1], &outcome.units[2]],
                    num_planes: num_planes as usize,
                };
                let mut ry2 = src_y_strided.clone();
                let mut ru2 = src_u_strided.clone();
                let mut rv2 = src_v_strided.clone();
                // Same per-tile shape as pass 1 — `pack_tile_lr` resets its
                // `LrRefState` (C's `av1_reset_loop_restoration`, called from
                // `write_modes` per tile) on every call, so the per-tile loop is
                // what makes the LR delta-coding references tile-local.
                for (t, &(mi_row_start, mi_col_start, mi_row_end, mi_col_end, n_sb_rows, n_sb_cols)) in
                    tile_grid.iter().enumerate()
                {
                    env.tile_row_start = mi_row_start;
                    env.tile_col_start = mi_col_start;
                    env.tile_row_end = mi_row_end;
                    env.tile_col_end = mi_col_end;
                    let mut kf2 = KfFrameContext::default_for_qindex(qindex);
                    let mut enc2 = OdEcEnc::new();
                    let trees2 = pack_tile_lr(
                        &mut enc2,
                        &env,
                        &pick_cfg,
                        &pack_cfg,
                        &mut kf2,
                        &mut ry2,
                        &mut ru2,
                        &mut rv2,
                        mi_row_start,
                        mi_col_start,
                        n_sb_rows,
                        n_sb_cols,
                        sb_mi,
                        sb_block,
                        Some(&lr_pack),
                        None,
                    );
                    assert_eq!(
                        trees2.len(),
                        (n_sb_rows * n_sb_cols) as usize,
                        "{}: LR repack must walk every SB of tile {t}",
                        self.label
                    );
                    tile_payloads[t] = enc2.done().to_vec();
                }
            }
        }

        if tiles_log2 == 0 {
            assemble_frame_obu_payload_single_tile(&p, tiles_log2, &tile_payloads[0])
        } else {
            // Multi-tile: the header is re-serialized here too (nothing is
            // spliced from the bootstrap), with `context_update_tile_id` =
            // `largest_tile_id` and `tile_size_bytes` derived from the packed
            // tile lengths the way `write_tile_obu_size` does, then each
            // non-last tile length-prefixed.
            assemble_multitile_frame_obu_payload_derived(&p, &tile_payloads)
        }
    }

    /// Setup-time validation: the port's assembled frame OBU payload is
    /// byte-identical to the C reference stream's. Returns the reference
    /// stream for reuse as the bench-loop bootstrap.
    #[cfg(feature = "c-oracle")]
    pub fn assert_byte_exact(&self) -> Vec<u8> {
        c::ref_init();
        let bootstrap = self.c_encode();
        assert!(!bootstrap.is_empty(), "{}: C encode failed", self.label);
        let ours = self.port_encode(&bootstrap);
        let real = Self::frame_obu_payload(&bootstrap);
        assert_eq!(
            ours, real,
            "{}: port frame OBU payload differs from real aomenc — timing a \
             divergent encode would be meaningless",
            self.label
        );
        bootstrap
    }
}

/// One frame's tight, row-major u16 planes (empty `u`/`v` when monochrome).
#[derive(Clone, Debug)]
pub struct FramePlanes {
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
}

/// INTER-ENCODE Chunk 0: a multi-frame encode cell — the 2-frame `[KEY, P]`
/// source + config that the inter-encode skeleton (chunk 2) plugs the port's
/// inter encoder into, and that produces the `aomenc` reference stream today.
///
/// `frames[0]` is the KEY source, `frames[1]` the P source; the "simplest inter
/// config" ([`Self::c_encode_inter`]) codes frame 1 as a single-reference
/// translational P against frame 0. Same geometry across frames.
#[derive(Clone, Debug)]
pub struct MultiFrameEncodeCell {
    pub label: String,
    pub w: usize,
    pub h: usize,
    pub mono: bool,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: u8,
    pub cq_level: i32,
    /// `--cpu-used` for the C side AND the port's `SpeedFeatures` level.
    pub speed: i32,
    /// Exactly 2 frames for chunk 0: `[KEY source, P source]`.
    pub frames: Vec<FramePlanes>,
}

impl MultiFrameEncodeCell {
    /// Build a 2-frame `[KEY, P]` cell whose P source is `base` (frame 0)
    /// translated by `(dx, dy)` LUMA pixels (chroma shifts by the subsampled
    /// amount), edge-clamped — a clean single-reference translational P that
    /// `aomenc` codes as NEWMV / NEAR* / GLOBALMV motion-compensated against
    /// frame 0. `dx == dy == 0` gives the degenerate zero-MV (near-skip) P.
    /// Reuses [`EncodeCell`]'s content for frame 0.
    pub fn translational(base: &EncodeCell, dx: i32, dy: i32) -> Self {
        let (w, h, mono, ss_x, ss_y) = (base.w, base.h, base.mono, base.ss_x, base.ss_y);
        let translate = |src: &[u16], pw: usize, ph: usize, sx: i32, sy: i32| -> Vec<u16> {
            if pw == 0 || ph == 0 {
                return Vec::new();
            }
            let mut out = vec![0u16; pw * ph];
            for r in 0..ph {
                let sr = (r as i32 - sy).clamp(0, ph as i32 - 1) as usize;
                for c in 0..pw {
                    let sc = (c as i32 - sx).clamp(0, pw as i32 - 1) as usize;
                    out[r * pw + c] = src[sr * pw + sc];
                }
            }
            out
        };
        let (cw, ch) = if mono {
            (0, 0)
        } else {
            ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
        };
        let f0 = FramePlanes {
            y: base.y.clone(),
            u: base.u.clone(),
            v: base.v.clone(),
        };
        let f1 = FramePlanes {
            y: translate(&base.y, w, h, dx, dy),
            u: translate(&base.u, cw, ch, dx >> ss_x, dy >> ss_y),
            v: translate(&base.v, cw, ch, dx >> ss_x, dy >> ss_y),
        };
        MultiFrameEncodeCell {
            label: format!("{}+p(dx={dx},dy={dy})", base.label),
            w,
            h,
            mono,
            ss_x,
            ss_y,
            bd: base.bd,
            cq_level: base.cq_level,
            speed: base.speed,
            frames: vec![f0, f1],
        }
    }

    /// Encode the 2-frame `[KEY, P]` clip with real `aomenc` at the "simplest
    /// inter config" (INTER-ENCODE-ROADMAP.md §3): `--end-usage=q
    /// --lag-in-frames=0 --cpu-used=<speed> --limit=2` with obmc / warp /
    /// global-motion / interintra / masked / diff-wtd / dual-filter /
    /// ref-frame-mvs all disabled. `enable_cdef` / `enable_restoration` select
    /// the faithful GOOD-quality defaults (`true`/`true`) or a smaller decoder
    /// envelope (`false`/`false`). Returns the concatenated 2-frame stream.
    #[cfg(feature = "c-oracle")]
    pub fn c_encode_inter(&self, enable_cdef: bool, enable_restoration: bool) -> Vec<u8> {
        assert_eq!(
            self.frames.len(),
            2,
            "chunk-0 cell carries exactly 2 frames"
        );
        let f0 = &self.frames[0];
        let f1 = &self.frames[1];
        c::ref_encode_av1_inter_2frame(
            (&f0.y, &f0.u, &f0.v),
            (&f1.y, &f1.u, &f1.v),
            self.w,
            self.h,
            i32::from(self.bd),
            self.mono,
            self.ss_x as i32,
            self.ss_y as i32,
            self.cq_level,
            self.speed,
            enable_cdef,
            enable_restoration,
        )
    }

    /// INTER-ENCODE chunk 2f/2g — encode frame 1 (the §3 low-delay zero-MV P)
    /// with the port's OWN search + pack, returning the frame-1 OBU payload
    /// (derived header + tile), the exact shape [`EncodeCell::port_encode`]
    /// returns for a KEY frame.
    ///
    /// The port CHOOSES the inter blocks: `pack_tile` runs the real partition
    /// search with `PickFrameCfg::inter` live, every leaf competes the inter
    /// SKIP arm ([`aom_encode::inter_rd::rd_pick_inter_mode_sb`]) against the
    /// intra winner, and the winning tree is packed through `pack_leaf`'s
    /// inter branch. Nothing block-level is copied from the reference stream.
    ///
    /// ## HONEST BOOTSTRAP (same contract as the KEY `port_encode`)
    ///
    /// - The sequence template + the three RECON-DEPENDENT frame-header fields
    ///   (loop-filter levels/deltas, CDEF, the frame `interp_filter` — a
    ///   per-frame RD decision) come from `bootstrap` (the real `aomenc`
    ///   2-frame stream), exactly as `LowDelayPHeaderParams` documents.
    /// - The REFERENCE frame is frame 0's decoded (filtered) recon, obtained
    ///   by decoding `bootstrap`'s frame 0 with the port's own byte-exact
    ///   decoder. Callers should separately assert the port's KEY encode of
    ///   frame 0 byte-matches (`frame0_cell().port_encode(..)`), which makes
    ///   the two frame-0 payloads — and hence this reference — identical.
    /// - `base_qindex` is DERIVED (`rc::base_qindex_lowdelay_p_from_cq`) and
    ///   cross-checked against the coded header.
    pub fn port_encode_inter_p(&self, bootstrap: &[u8]) -> Vec<u8> {
        use aom_encode::inter_frame::{
            LowDelayPHeaderParams, RefFrame, TWO_FRAME_P_REF_MAP_IDX, TWO_FRAME_P_REFRESH_FLAGS,
            derive_lowdelay_p_frame_header,
        };
        assert_eq!(self.frames.len(), 2, "2-frame [KEY, P] cell");
        assert_eq!(self.speed, 0, "the inter-encode skeleton is speed-0 scoped");
        let r = parse_inter_2frame_reference(bootstrap);
        let real = &r.real_f1;
        let (w, h, mono, ss_x, ss_y, bd) =
            (self.w, self.h, self.mono, self.ss_x, self.ss_y, self.bd);

        // The port's own P qindex — derived, then cross-checked.
        let qindex = aom_encode::rc::base_qindex_lowdelay_p_from_cq(self.cq_level);
        assert_eq!(
            qindex, real.quant.base_qindex,
            "{}: derived low-delay P qindex must match the coded header",
            self.label
        );
        assert!(
            !real.tx_mode_select,
            "{}: the §3 P codes TX_MODE_LARGEST",
            self.label
        );
        assert!(!real.cdef.enable_cdef || real.cdef.cdef_bits == 0);

        // --- the reference: frame 0's decoded, filtered recon ---
        let decoded =
            aom_decode::frame::decode_frames(bootstrap).expect("bootstrap stream must decode");
        assert!(decoded.len() >= 2, "2-frame stream");
        let f0 = &decoded[0];
        let ref_frame = RefFrame::new(
            f0.y.clone(),
            f0.u.clone(),
            f0.v.clone(),
            f0.width,
            f0.width_uv,
            f0.width,
            f0.height,
            f0.width_uv,
            f0.height_uv,
            0,
        );

        // --- frame-1 source planes, strided + edge-extended (the KEY recipe) ---
        let mi_cols = mi_dim(w as i32);
        let mi_rows = mi_dim(h as i32);
        // The superblock size the STREAM declares. The §3 GOOD-mode shim codes
        // SB128 (libaom's speed-0 GOOD default), so a 64x128 frame is ONE
        // column-cropped 128x128 superblock — its root codes a gathered 2-way
        // partition symbol before the two visible 64x64 children. Driving the
        // walk at SB64 dropped that symbol (the original "two-superblock tile"
        // divergence); a 64x64 frame is degenerate (both walks code identical
        // symbols), which is why the single-SB gate matched either way.
        let (sb_block, sb_mi, sb_px) = if r.seq_cfg.tile_info.mib_size_log2 == 5 {
            (15usize, 32i32, 128usize) // BLOCK_128X128
        } else {
            (12usize, 16i32, 64usize) // BLOCK_64X64
        };
        let (cw, ch) = if mono {
            (0, 0)
        } else {
            ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
        };
        let n_sb_x = ((mi_cols + sb_mi - 1) / sb_mi).max(1);
        let n_sb_y = ((mi_rows + sb_mi - 1) / sb_mi).max(1);
        let sb_px_w = n_sb_x as usize * sb_px;
        let sb_px_h = n_sb_y as usize * sb_px;
        let stride = 320.max(sb_px_w + 4);
        let buf_h = (sb_px_h + 4).max(h + 4);
        let f1 = &self.frames[1];
        let extend_plane = |dst: &mut [u16], pw: usize, ph: usize| {
            for row in 0..ph {
                let edge = dst[row * stride + pw - 1];
                for col in pw..stride {
                    dst[row * stride + col] = edge;
                }
            }
            for row in ph..buf_h {
                dst.copy_within((ph - 1) * stride..ph * stride, row * stride);
            }
        };
        let mut src_y_strided = vec![0u16; stride * buf_h];
        for row in 0..h {
            src_y_strided[row * stride..row * stride + w]
                .copy_from_slice(&f1.y[row * w..row * w + w]);
        }
        extend_plane(&mut src_y_strided, w, h);
        let mut src_u_strided = vec![0u16; stride * buf_h];
        let mut src_v_strided = vec![0u16; stride * buf_h];
        if !mono {
            for row in 0..ch {
                src_u_strided[row * stride..row * stride + cw]
                    .copy_from_slice(&f1.u[row * cw..row * cw + cw]);
                src_v_strided[row * stride..row * stride + cw]
                    .copy_from_slice(&f1.v[row * cw..row * cw + cw]);
            }
            extend_plane(&mut src_u_strided, cw, ch);
            extend_plane(&mut src_v_strided, cw, ch);
        }

        // --- quantizer / costs / rdmult (P = LF_UPDATE, GOOD mode) ---
        let mut quants = Quants::zeroed();
        let mut deq = Dequants::zeroed();
        av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
        let rows_y = set_q_index(&quants, &deq, qindex as usize, 0);
        let rows_u = set_q_index(&quants, &deq, qindex as usize, 1);
        let rows_v = set_q_index(&quants, &deq, qindex as usize, 2);
        let mut kf_write = KfFrameContext::default_for_qindex(qindex);
        let enable_filter_intra = r.enable_filter_intra;
        let frame_real = derive_real_costs(&kf_write, enable_filter_intra, None);
        let rdmult = av1_compute_rd_mult_based_on_qindex(
            bd,
            FrameUpdateType::Lf,
            qindex,
            TuneMetric::Psnr,
            EncMode::Good,
        );

        // Inter CDFs start at the spec defaults (primary_ref_frame = NONE) and
        // ADAPT through the pack; the frame-init cost tables derive from them
        // (pack_tile refreshes both per SB, INTERNAL_COST_UPD_SB).
        let mut inter_cdfs = aom_encode::inter_costs::InterFrameCdfs::defaults();
        let frame_inter_costs = aom_encode::inter_costs::derive_inter_mode_costs(&inter_cdfs);
        // The switchable interp-filter cost table (crate interp_rd): the §3
        // frame writes no filter symbols so the CDF never adapts — the
        // default-derived table is the per-SB refresh fixpoint.
        let interp_costs = aom_encode::interp_rd::SwitchableInterpCosts::from_default_cdfs();

        let sf = SpeedFeatures::set_allintra(0, false, bd > 8);
        let pol = sf.tx_type_search_policy(false, 0);
        let uv_lp = UvLoopPolicy::speed0_allintra();
        let env = SbEncodeEnv {
            ref_frame: Some(&ref_frame),
            sb_size: sb_block,
            mi_rows,
            mi_cols,
            // `cm->width`/`cm->height` — the TRUE crop (KB-28).
            frame_width: w as i32,
            frame_height: h as i32,
            tile_row_start: 0,
            tile_col_start: 0,
            tile_row_end: 1 << 16,
            tile_col_end: 1 << 16,
            monochrome: mono,
            ss_x,
            ss_y,
            bd,
            lossless: false,
            reduced_tx_set_used: real.reduced_tx_set_used,
            disable_edge_filter: !r.enable_intra_edge_filter,
            filter_type: 0,
            stride,
            src_y: &src_y_strided,
            src_u: &src_u_strided,
            src_v: &src_v_strided,
            base_y: 0,
            base_uv: 0,
            rows_y: &rows_y,
            rows_u: &rows_u,
            rows_v: &rows_v,
            rdmult,
            sharpness: 0,
            enable_optimize_b: trellis_opt_of_knob(3),
            qm_levels: None,
            tune: Default::default(),
            deltaq: None,
            use_chroma_trellis_rd_mult: false,
            coeff_costs_y: &frame_real.coeff_costs_y,
            coeff_costs_uv: &frame_real.coeff_costs_uv,
            txfm_partition_costs: [[0i32; 2]; 21],
            tx_type_costs: &frame_real.tx_type_costs_y,
        };
        let pick_cfg = PickFrameCfg {
            fs_sf: Default::default(),
            inter: Some(aom_encode::partition_pick::InterSearchCfg {
                costs: &frame_inter_costs,
                interp_costs: &interp_costs,
                // `use_more_sharp_interp = boosted ? 0 : 1` (GOOD base,
                // speed_features.c:1139): the low-delay P (LF_UPDATE) is
                // never boosted.
                use_more_sharp_interp: true,
                // dequant_QTX[1] — identical across planes (zero delta-q).
                dequant_ac: i32::from(rows_y.dequant[1]),
                allow_high_precision_mv: real.allow_high_precision_mv,
                is_integer_mv: real.cur_frame_force_integer_mv,
                sign_bias: [0i8; 8],
                allow_ref_frame_mvs: real.allow_ref_frame_mvs,
                global_mv: (0, 0),
                gm_wmtype: 0,
            }),
            intra_tools: Default::default(),
            mode_costs: &frame_real.mode_costs,
            tx_size_costs: &frame_real.tx_size_costs,
            skip_costs: &frame_real.skip_costs,
            tx_type_costs_y: &frame_real.tx_type_costs_y,
            pol: &pol,
            uv_lp: &uv_lp,
            intra_uv_mode_cost: &frame_real.mode_costs.intra_uv_mode_cost,
            cfl_costs: &frame_real.cfl_costs,
            partition_costs: &frame_real.partition_costs,
            partition_cdfs: &frame_real.partition_cdf,
            allintra: false,
            speed: 0,
            qindex,
            enable_filter_intra,
            enable_tx64: true,
            enable_rect_tx: true,
            intra_pruning_with_hog: true,
            enable_rect_partitions: true,
            less_rectangular_check_level: 0,
            max_partition_size: sb_block,
            min_partition_size: 0,
            enable_1to4_partitions: true,
            enable_ab_partitions: true,
            allow_screen_content_tools: false,
            qm_levels: None,
            palette_costs: None,
            intrabc: None,
        };
        let pack_cfg = aom_encode::pack::PackCfg {
            enable_filter_intra,
            tx_mode_is_select: false, // TX_MODE_LARGEST (asserted above)
            signal_gate: qindex > 0,
            allow_update_cdf: !real.prefix.disable_cdf_update,
            base_qindex: qindex,
            delta_q_present: false,
            delta_q_res: 0,
            allow_screen_content_tools: false,
            allow_intrabc: false,
            search_allow_intrabc: false,
            search_tx_mode_is_select: false,
        };

        let mut recon_y = src_y_strided.clone();
        let mut recon_u = src_u_strided.clone();
        let mut recon_v = src_v_strided.clone();
        let mut enc = OdEcEnc::new();
        let trees = pack_tile_lr(
            &mut enc,
            &env,
            &pick_cfg,
            &pack_cfg,
            &mut kf_write,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            0,
            0,
            n_sb_y,
            n_sb_x,
            sb_mi,
            sb_block,
            None,
            Some(&mut inter_cdfs),
        );
        assert_eq!(
            trees.len(),
            (n_sb_x * n_sb_y) as usize,
            "{}: P pack_tile must walk every SB",
            self.label
        );
        let tile_bytes = enc.done().to_vec();

        // --- the derived P header (recon-dependent tail bootstrapped) ---
        let p = LowDelayPHeaderParams {
            base_qindex: qindex,
            order_hint: 1,
            refresh_frame_flags: TWO_FRAME_P_REFRESH_FLAGS,
            ref_map_idx: TWO_FRAME_P_REF_MAP_IDX,
            disable_cdf_update: real.prefix.disable_cdf_update,
            reduced_tx_set_used: real.reduced_tx_set_used,
            interp_filter: real.interp_filter,
            loopfilter: real.loopfilter.clone(),
            cdef: real.cdef.clone(),
        };
        let derived = derive_lowdelay_p_frame_header(&r.seq_cfg, &p);
        assemble_frame_obu_payload_single_tile(&derived, 0, &tile_bytes)
    }

    /// Frame 0 (the KEY source) as a single-frame [`EncodeCell`] (usage = GOOD,
    /// the inter context) — for a KEY-only cross-check or, in chunk 2, the
    /// port's KEY encode of frame 0. Reuses the whole `EncodeCell` machinery.
    pub fn frame0_cell(&self) -> EncodeCell {
        let f0 = &self.frames[0];
        EncodeCell {
            label: format!("{}#f0", self.label),
            w: self.w,
            h: self.h,
            mono: self.mono,
            ss_x: self.ss_x,
            ss_y: self.ss_y,
            usage: 0, // GOOD_QUALITY — the inter (non-all-intra) context
            cq_level: self.cq_level,
            speed: self.speed,
            bd: self.bd,
            y: f0.y.clone(),
            u: f0.u.clone(),
            v: f0.v.clone(),
        }
    }
}

/// The parsed reference facts of a 2-frame `[KEY, P]` `aomenc` stream
/// ([`MultiFrameEncodeCell::c_encode_inter`]'s output): frame 1's OBU payload
/// + exact header bit length, the parsed frame-1 header, and the
/// sequence-derived [`FrameHeaderObu`] template the derive/parse paths share.
pub struct Inter2FrameRef {
    /// Frame 1's whole OBU payload (header + tile bytes).
    pub f1_payload: Vec<u8>,
    /// Frame 0's whole OBU payload.
    pub f0_payload: Vec<u8>,
    /// Bit length of frame 1's uncompressed header (the header/tile split).
    pub header_bits: usize,
    /// The parsed frame-1 header.
    pub real_f1: FrameHeaderObu,
    /// The sequence-derived header template (`read_uncompressed_header`'s cfg).
    pub seq_cfg: FrameHeaderObu,
    /// Sequence-header tool bits the port search threads.
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
}

/// Parse a real 2-frame `[KEY, P]` stream into [`Inter2FrameRef`] — the shared
/// front half of the inter-encode gates (the same construction
/// `inter_pack_tile_diff.rs` proved byte-faithful).
pub fn parse_inter_2frame_reference(stream: &[u8]) -> Inter2FrameRef {
    let obus = walk_obus(stream);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .expect("sequence header OBU");
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let num_planes = if c.monochrome { 1 } else { 3 };
    let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
    let mi_cols = mi_dim(s.max_frame_width);
    let mi_rows = mi_dim(s.max_frame_height);

    let mut cfg = FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length
                as u32,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            frame_id_length: s.frame_id_length as u32,
            force_screen_content_tools: s.force_screen_content_tools,
            force_integer_mv: s.force_integer_mv,
            max_frame_width: s.max_frame_width,
            max_frame_height: s.max_frame_height,
            enable_order_hint: s.enable_order_hint,
            order_hint_bits_minus_1: s.order_hint_bits_minus_1,
            operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
            operating_point_idc: seq.operating_point_idc,
            op_decoder_model_param_present: seq.op_decoder_model_param_present,
            buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length as u32,
            ..Default::default()
        },
        frame_size: FrameSizeHeader {
            num_bits_width: s.num_bits_width,
            num_bits_height: s.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            ..Default::default()
        },
        tile_info: tile_limits(mi_cols, mi_rows, mib_size_log2),
        num_planes,
        separate_uv_delta_q: c.separate_uv_delta_q,
        loopfilter: LoopfilterHeader {
            last_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            last_mode_deltas: [0, 0],
            ..Default::default()
        },
        cdef: CdefHeader {
            enable_cdef: s.enable_cdef,
            ..Default::default()
        },
        restoration: RestorationHeader {
            enable_restoration: s.enable_restoration,
            sb_size_128: s.sb_size_128,
            subsampling_x: c.subsampling_x,
            subsampling_y: c.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    cfg.might_allow_ref_frame_mvs = s.enable_ref_frame_mvs && s.enable_order_hint;
    cfg.might_allow_warped_motion = s.enable_warped_motion;

    let frames: Vec<&(u32, &[u8])> = obus.iter().filter(|(t, _)| *t == OBU_FRAME).collect();
    assert_eq!(frames.len(), 2, "expected [KEY, P] frame OBUs");
    let f0_payload = frames[0].1.to_vec();
    let f1_payload = frames[1].1.to_vec();
    let mut rb = ReadBitBuffer::new(&f1_payload);
    let real_f1 = read_uncompressed_header(&mut rb, &cfg);
    assert_eq!(real_f1.prefix.frame_type, 1, "frame 1 must be INTER");
    let header_bits = rb.bit_position();
    Inter2FrameRef {
        f1_payload,
        f0_payload,
        header_bits,
        real_f1,
        seq_cfg: cfg,
        enable_filter_intra: s.enable_filter_intra,
        enable_intra_edge_filter: s.enable_intra_edge_filter,
    }
}

/// The standard Gate-3 encode cell set (bd8 4:2:0 ALLINTRA KEY):
/// speed-0 on real content at 3 sizes x 3 cq levels (all cells are landed
/// KB-6 byte-match gates), plus one speed-4 point on the byte-exact
/// synthetic-diag grid cell (speed features change the profile shape).
/// Parse an encoded stream's frame-header LOOP-RESTORATION fields — the C
/// encoder's `av1_pick_filter_restoration` DECISION as coded by
/// `encode_restoration_mode`: per-plane `frame_restoration_type` + the coded
/// per-plane unit sizes. The decision-level differential witness for the
/// ported search (bitstream facts, not C-internals).
pub fn parse_restoration_decision(stream: &[u8]) -> ([u8; 3], [i32; 3]) {
    let obus = walk_obus(stream);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .expect("no sequence-header OBU");
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);
    let frame_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME)
        .map(|(_, p)| *p)
        .expect("no frame OBU");
    let s = &seq.seq_header;
    let cc = &seq.color_config;
    let num_planes = if cc.monochrome { 1 } else { 3 };
    let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
    let mi_cols = mi_dim(s.max_frame_width);
    let mi_rows = mi_dim(s.max_frame_height);
    let cfg = FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length
                as u32,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            frame_id_length: s.frame_id_length as u32,
            force_screen_content_tools: s.force_screen_content_tools,
            force_integer_mv: s.force_integer_mv,
            max_frame_width: s.max_frame_width,
            max_frame_height: s.max_frame_height,
            enable_order_hint: s.enable_order_hint,
            order_hint_bits_minus_1: s.order_hint_bits_minus_1,
            operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
            operating_point_idc: seq.operating_point_idc,
            op_decoder_model_param_present: seq.op_decoder_model_param_present,
            buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length as u32,
            temporal_layer_id: 0,
            spatial_layer_id: 0,
            ..Default::default()
        },
        frame_size: FrameSizeHeader {
            num_bits_width: s.num_bits_width,
            num_bits_height: s.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            ..Default::default()
        },
        tile_info: tile_limits(mi_cols, mi_rows, mib_size_log2),
        num_planes,
        separate_uv_delta_q: cc.separate_uv_delta_q,
        loopfilter: LoopfilterHeader {
            last_ref_deltas: KF_REF_DELTAS,
            last_mode_deltas: KF_MODE_DELTAS,
            ..Default::default()
        },
        cdef: CdefHeader {
            enable_cdef: s.enable_cdef,
            ..Default::default()
        },
        restoration: RestorationHeader {
            enable_restoration: s.enable_restoration,
            sb_size_128: s.sb_size_128,
            subsampling_x: cc.subsampling_x,
            subsampling_y: cc.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    let mut rb = ReadBitBuffer::new(frame_payload);
    let p = read_uncompressed_header(&mut rb, &cfg);
    (
        p.restoration.frame_restoration_type,
        p.restoration.restoration_unit_size,
    )
}

/// The `lpf_sf` loop-restoration slice for the ALLINTRA path:
/// `set_allintra_speed_features_framesize_independent` (speed_features.c:
/// dual_sgr/ep-pruning at speed>=1; wiener-src-var + sgr-from-wiener prunes
/// at speed>=2; reduced window / prune upgrades at speed>=3; full disable at
/// speed>=5 — moot here because the REAL encoder also clears the seq
/// `enable_restoration` bit at those speeds) + the qindex-dependent
/// unit-size-search bounds (`av1_set_speed_features_qindex_dependent`:
/// full 64..256 descent at speed 0; the single-size rule for allintra
/// speed>=1: 128 when qindex <= 96 on sub-1440p frames, else 256).
pub fn lr_search_sf_allintra(
    speed: i32,
    qindex: i32,
    w: usize,
    h: usize,
    allow_screen_content_tools: bool,
) -> LrSearchSf {
    let mut sf = LrSearchSf::default();
    if speed >= 1 {
        sf.dual_sgr_penalty_level = 1;
        sf.enable_sgr_ep_pruning = 1;
    }
    if speed >= 2 {
        sf.prune_wiener_based_on_src_var = 1;
        sf.prune_sgr_based_on_wiener = 1;
    }
    if speed >= 3 {
        sf.prune_sgr_based_on_wiener = if allow_screen_content_tools { 1 } else { 2 };
        sf.disable_loop_restoration_chroma = false;
        sf.reduce_wiener_window_size = true;
        sf.prune_wiener_based_on_src_var = 2;
    }
    if speed >= 5 {
        sf.disable_wiener_filter = true;
        sf.disable_sgr_filter = true;
    }
    // Unit-size search bounds (qindex-dependent setter, all modes).
    sf.min_lr_unit_size = 64; // RESTORATION_PROC_UNIT_SIZE
    sf.max_lr_unit_size = 256; // RESTORATION_UNITSIZE_MAX
    let is_1440p_or_larger = w.min(h) >= 1440;
    let is_720p_or_larger = w.min(h) >= 720;
    if speed >= 1 {
        if is_1440p_or_larger {
            sf.min_lr_unit_size = 256;
        } else if is_720p_or_larger {
            sf.min_lr_unit_size = 128;
        }
    }
    // `speed >= 3 || (mode == ALLINTRA && speed >= 1)` — this helper IS the
    // allintra arm.
    if speed >= 1 {
        if qindex <= 96 && !is_1440p_or_larger {
            sf.min_lr_unit_size = 128;
            sf.max_lr_unit_size = 128;
        } else {
            sf.min_lr_unit_size = 256;
            sf.max_lr_unit_size = 256;
        }
    }
    sf
}

/// The `lpf_sf` loop-restoration slice for the GOOD path
/// (`set_good_speed_features_framesize_independent`, :1091, + the
/// qindex-dependent unit-size bounds). VERIFIED line-by-line vs
/// speed_features.c (v3.14.1); bracket line numbers confirmed against the
/// `if (speed >= N)` guards at :1166/:1227/:1283/:1361/:1420:
/// //   :1164       reduce_wiener_window_size = 1 — UNCONDITIONAL (in the
/// //               "speed 0 for all" prologue, before if(speed>=1)@:1166),
/// //               UNLIKE allintra's speed>=3 gate (:467). GOOD therefore
/// //               searches the reduced 5-tap luma Wiener window at EVERY
/// //               speed — it is NOT default-equal at speed 0 (the prior
/// //               "GOOD speed-0 == defaults" note was wrong on this).
/// //   :1220-1221  dual_sgr_penalty_level=1, enable_sgr_ep_pruning=1 (speed>=1)
/// //   :1272-1274  prune_wiener_based_on_src_var=1, prune_sgr_based_on_wiener=1,
/// //               disable_loop_restoration_chroma = boosted ? 0 : 1 (speed>=2)
/// //   :1352-1358  prune_sgr_based_on_wiener = screen?1:2,
/// //               prune_wiener_based_on_src_var=2,
/// //               use_downsampled_wiener_stats=1 (speed>=3 — inside
/// //               if(speed>=3)@:1283, before if(speed>=4)@:1361; the
/// //               predecessor's `speed>=4` was an off-by-one, corrected)
/// //   :1452-1453  enable_sgr_ep_pruning=2,
/// //               disable_wiener_coeff_refine_search=true (speed>=5)
/// // Not on this path (verified): :648-649 (switchable_lr_with_bias_level,
/// // dual_sgr_penalty_level = boosted?1:3) live in
/// // `set_good_speed_features_lc_dec_framesize_dependent` (:619) — the
/// // large-scale/lc-dec arm a normal single-frame GOOD encode does not
/// // take. For a single KEY frame `boosted` (frame_is_boosted) is TRUE.
/// // Only GOOD speed-0 cells are gated in this harness; GOOD speed>=1 needs
/// // dedicated gate cells to exercise the >=1 arms (a follow-up).
pub fn lr_search_sf_good(
    speed: i32,
    qindex: i32,
    w: usize,
    h: usize,
    allow_screen_content_tools: bool,
) -> LrSearchSf {
    let mut sf = LrSearchSf::default();
    // :1164 — set UNCONDITIONALLY in the GOOD setter (the "speed 0 for all"
    // prologue, before if(speed>=1)@:1166); GOOD uses the reduced 5-tap
    // Wiener window at every speed, unlike allintra (speed>=3, :467).
    sf.reduce_wiener_window_size = true;
    // :1220-1221 (if speed>=1).
    if speed >= 1 {
        sf.dual_sgr_penalty_level = 1;
        sf.enable_sgr_ep_pruning = 1;
    }
    // :1272-1274 (if speed>=2). `boosted` is TRUE for a single KEY frame, so
    // disable_loop_restoration_chroma = boosted ? 0 : 1 = 0 (false).
    if speed >= 2 {
        sf.prune_wiener_based_on_src_var = 1;
        sf.prune_sgr_based_on_wiener = 1;
        sf.disable_loop_restoration_chroma = false;
    }
    // :1352-1358 (if speed>=3 — inside if(speed>=3)@:1283, before
    // if(speed>=4)@:1361; the predecessor's `speed>=4` was an off-by-one).
    if speed >= 3 {
        sf.prune_sgr_based_on_wiener = if allow_screen_content_tools { 1 } else { 2 };
        sf.prune_wiener_based_on_src_var = 2;
        sf.use_downsampled_wiener_stats = true;
    }
    // :1452-1453 (if speed>=5).
    if speed >= 5 {
        sf.enable_sgr_ep_pruning = 2;
        sf.disable_wiener_coeff_refine_search = true;
    }
    // Unit-size search bounds (qindex-dependent setter, all modes).
    sf.min_lr_unit_size = 64;
    sf.max_lr_unit_size = 256;
    let is_1440p_or_larger = w.min(h) >= 1440;
    let is_720p_or_larger = w.min(h) >= 720;
    if speed >= 1 {
        if is_1440p_or_larger {
            sf.min_lr_unit_size = 256;
        } else if is_720p_or_larger {
            sf.min_lr_unit_size = 128;
        }
    }
    // GOOD arm of `speed >= 3 || (ALLINTRA && speed >= 1)`.
    if speed >= 3 {
        if qindex <= 96 && !is_1440p_or_larger {
            sf.min_lr_unit_size = 128;
            sf.max_lr_unit_size = 128;
        } else {
            sf.min_lr_unit_size = 256;
            sf.max_lr_unit_size = 256;
        }
    }
    sf
}

#[cfg(feature = "c-oracle")]
pub fn encode_cells() -> Vec<EncodeCell> {
    let mut cells = Vec::new();
    for &(size_label, vector, crop) in &[
        ("64", "av1-1-b8-01-size-64x64", None),
        (
            "128",
            "av1-1-b8-00-quantizer-00",
            Some((128usize, 128usize, 64usize, 64usize)),
        ),
        ("196", "av1-1-b8-01-size-196x196", None),
    ] {
        for &cq in &[12i32, 32, 63] {
            cells.push(EncodeCell::real_content(
                &format!("enc_s0_{size_label}_cq{cq}"),
                vector,
                crop,
                cq,
                0,
            ));
        }
    }
    cells.push(EncodeCell::synthetic_diag(
        "enc_s4_128_cq32",
        128,
        128,
        32,
        4,
    ));
    cells
}

#[cfg(test)]
mod tests {
    use super::replay_sb_qindex_tile_order;

    /// The per-TILE reset of the delta-q running base, pinned at the level the
    /// fix lives at. The probe `|_, _, running| running + 4` makes the running
    /// base directly observable: under C's per-tile reset
    /// (`encodeframe.c:1232-1239` search side, `bitstream.c:1745-1751` pack
    /// side) every tile's chain restarts at `base_qindex`, so a 4x2-SB frame
    /// split into two 2x2-SB tile COLUMNS produces `base+4, base+8` twice —
    /// once per tile — laid back out in FRAME SB raster.
    ///
    /// **Bite proof, MEASURED:** replacing the per-tile `running` with one
    /// frame-level base (the pre-fix walk) fails exactly this test —
    /// `left: [104, 108, 120, 124, 112, 116, 128, 132]` — while the other 12
    /// `-p zenav1-aom-bench --lib` tests stay green.
    #[test]
    fn replay_resets_the_running_base_at_every_tile() {
        const SB_MI: i32 = 16; // 64px superblock
        const BASE: i32 = 100;
        // 4 SB columns x 2 SB rows, split into two 2x2 tile COLUMNS.
        // (mi_row_start, mi_col_start, mi_row_end, mi_col_end, n_sb_rows, n_sb_cols)
        let two_cols = [
            (0, 0, 2 * SB_MI, 2 * SB_MI, 2, 2),
            (0, 2 * SB_MI, 2 * SB_MI, 4 * SB_MI, 2, 2),
        ];
        let (per_sb, used) =
            replay_sb_qindex_tile_order(&two_cols, 4, SB_MI, BASE, |_, _, running| running + 4);
        // Frame SB raster: row 0 = [tile0 c0, tile0 c1, tile1 c0, tile1 c1].
        assert_eq!(
            per_sb,
            vec![104, 108, 104, 108, 112, 116, 112, 116],
            "each tile must restart the running base at base_qindex"
        );
        assert!(used, "every SB moved off the base here");

        // The same frame split into two tile ROWS instead: the reset lands at
        // (mi_row = 2 SBs, mi_col = 0), a position a column split never produces.
        let two_rows = [
            (0, 0, SB_MI, 4 * SB_MI, 1, 4),
            (SB_MI, 0, 2 * SB_MI, 4 * SB_MI, 1, 4),
        ];
        let (per_sb, _) =
            replay_sb_qindex_tile_order(&two_rows, 4, SB_MI, BASE, |_, _, running| running + 4);
        assert_eq!(per_sb, vec![104, 108, 112, 116, 104, 108, 112, 116]);

        // Single tile: the reset is an identity, so the whole frame is one chain
        // — the pre-existing single-tile behaviour, unchanged.
        let one = [(0, 0, 2 * SB_MI, 4 * SB_MI, 2, 4)];
        let (per_sb, _) =
            replay_sb_qindex_tile_order(&one, 4, SB_MI, BASE, |_, _, running| running + 4);
        assert_eq!(per_sb, vec![104, 108, 112, 116, 120, 124, 128, 132]);

        // `deltaq_used` is the OR over tiles of "this SB left the frame base"
        // (`td->deltaq_used |= (x->delta_qindex != 0)`, encodeframe.c:375,
        // OR-reduced at :1593): a probe that never moves off the base reports
        // false, which is what clears `delta_q_present` (bitstream.c:4286-4289).
        let (per_sb, used) =
            replay_sb_qindex_tile_order(&two_cols, 4, SB_MI, BASE, |_, _, _| BASE);
        assert_eq!(per_sb, vec![BASE; 8]);
        assert!(!used);
    }

    /// The per-SB callback must receive FRAME-absolute mi coordinates (the delta-q
    /// modes index a frame-level wiener map / the frame source by them), while the
    /// result lands at the frame SB raster slot — the two indexings a tile-local
    /// walk is easiest to get wrong in opposite directions.
    #[test]
    fn replay_passes_frame_absolute_mi_and_indexes_frame_raster() {
        const SB_MI: i32 = 16;
        let two_cols = [
            (0, 0, 2 * SB_MI, 2 * SB_MI, 2, 2),
            (0, 2 * SB_MI, 2 * SB_MI, 4 * SB_MI, 2, 2),
        ];
        let mut seen: Vec<(i32, i32)> = Vec::new();
        let (per_sb, _) = replay_sb_qindex_tile_order(&two_cols, 4, SB_MI, 100, |r, c, _| {
            seen.push((r, c));
            r * 1000 + c
        });
        assert_eq!(
            seen,
            vec![
                (0, 0),
                (0, 16),
                (16, 0),
                (16, 16), // tile 0, in tile raster
                (0, 32),
                (0, 48),
                (16, 32),
                (16, 48), // tile 1
            ]
        );
        assert_eq!(
            per_sb,
            vec![0, 16, 32, 48, 16000, 16016, 16032, 16048],
            "results must be laid out in FRAME SB raster, not tile order"
        );
    }
}
