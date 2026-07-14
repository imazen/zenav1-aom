//! `decode_frame_obus` — decode a real AV1 KEY-frame bitstream (aomenc /
//! `aom_codec_av1_cx` output) to pixels: OBU walk (temporal delimiter /
//! sequence header / frame header / tile group, incl. the combined OBU_FRAME),
//! header parsing through the validated aom-entropy readers, default
//! FRAME_CONTEXT init by `base_qindex`, and the KEY-frame tile decode driver.
//!
//! ENVELOPE — the feature set [`decode_tile_kf`] models. Anything outside it
//! is a hard [`Err`], never a mis-decode:
//! - KEY frame, shown, not show-existing; `error_resilient` accepted.
//! - 64x64 superblocks only (no `use_128x128_superblock`).
//! - single tile (1x1), single tile group.
//! - screen-content tools OFF (`allow_screen_content_tools` would put
//!   palette/intrabc flags in the block layer).
//! - CDEF / loop restoration / film grain disabled at the sequence level;
//!   superres not scaled; no frame-size override (frame == sequence max dims).
//! - loop-filter levels all zero (the deblocker is not applied by this
//!   driver; the C decoder skips it when the frame luma levels are zero).
//! - no segmentation, no quantization matrices, not (coded-)lossless.
//! - `disable_cdf_update` off (the driver always adapts).
//! - delta-q / delta-lf ARE in the envelope (per-block dequant recompute).
//!
//! The gold test (tests/real_bitstream.rs) compares the output planes
//! byte-identically against the REAL C decoder (`aom_codec_av1_dx`) on
//! bitstreams produced by the REAL encoder at `--cpu-used=0 --end-usage=q`.

use crate::{decode_tile_kf, KfTileConfig};
use aom_entropy::dec::OdEcDec;
use aom_entropy::header::{
    read_sequence_header_obu, read_tile_group_header, read_uncompressed_header, CdefHeader,
    FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader, RestorationHeader,
    SequenceHeaderObu, TileInfoHeader,
};
use aom_entropy::leb128::uleb_decode;
use aom_entropy::obu::read_obu_header;
use aom_entropy::partition::{KfFrameContext, TxMode};
use aom_entropy::rb::ReadBitBuffer;

/// aom-txb's `CDF_ARENA_LEN` (the coefficient region length
/// `KfFrameContext::default_for_qindex` fills).
const _: () = assert!(aom_txb::CDF_ARENA_LEN == 4045);

/// A decoded KEY frame: cropped planes + the header facts harnesses assert on.
#[derive(Clone, Debug)]
pub struct FrameDecode {
    /// Cropped luma, tight `width`-strided rows, u16 at every bit depth.
    pub y: Vec<u16>,
    /// Cropped chroma (empty when monochrome), tight `width_uv`-strided.
    pub u: Vec<u16>,
    pub v: Vec<u16>,
    pub width: usize,
    pub height: usize,
    pub width_uv: usize,
    pub height_uv: usize,
    pub bit_depth: i32,
    pub monochrome: bool,
    pub subsampling_x: usize,
    pub subsampling_y: usize,
    /// Frame quantizer facts (for harness assertions).
    pub base_qindex: i32,
    /// `[y0, y1, u, v]` loop-filter levels — all zero inside the envelope.
    pub filter_level: [i32; 4],
    /// `features.tx_mode` was TX_MODE_SELECT (vs LARGEST).
    pub tx_mode_select: bool,
    pub reduced_tx_set: bool,
    pub delta_q_present: bool,
}

/// `av1_get_tile_limits` (av1/common/tile_common.c) for the single-tile
/// envelope: the min/max tile log2 bounds the tile-info reader consumes.
/// `min_log2_rows` is `max(min_log2_tiles - log2_cols, 0)` in C with the CODED
/// `log2_cols`; passing `log2_cols = min_log2_cols` is exact whenever the
/// stream codes the minimum (a 1x1 tiling always does — the caller hard-errors
/// on any other tiling immediately after the parse).
fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    const MAX_TILE_WIDTH: i32 = 4096;
    const MAX_TILE_AREA: i32 = 4096 * 2304;
    const MAX_TILE_COLS: i32 = 64;
    const MAX_TILE_ROWS: i32 = 64;
    fn tile_log2(blk_size: i32, target: i32) -> i32 {
        let mut k = 0;
        while (blk_size << (2 * k)) < target {
            k += 1;
        }
        k
    }
    let sb_cols = (mi_cols + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_rows = (mi_rows + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_size_log2 = mib_size_log2 as i32 + 2; // MI_SIZE_LOG2
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

/// `set_mb_mi` (av1/common/alloccommon.c): frame mi dims, 8-pixel aligned.
fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// KF `av1_setup_past_independence` loop-filter delta defaults
/// (`av1_set_default_ref_deltas` / `_mode_deltas`) — the "previous" deltas the
/// loop-filter reader diffs against on a keyframe.
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

struct ParsedFrame {
    header: FrameHeaderObu,
    /// Byte offset of the tile data within the SAME payload (OBU_FRAME), or
    /// `None` when the header stood alone (tile group arrives as its own OBU).
    tile_data_off: Option<usize>,
}

/// Parse a frame header (standalone FRAME_HEADER OBU or the head of an
/// OBU_FRAME payload) and run every envelope gate. The gates run in stream
/// order so an early out-of-envelope field can never be masked by a later
/// mis-parse (`allow_screen_content_tools` precedes the frame size in the
/// bitstream, so it is checked first).
fn parse_frame_header(
    seq: &SequenceHeaderObu,
    payload: &[u8],
    is_obu_frame: bool,
) -> Result<ParsedFrame, String> {
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let num_planes = if c.monochrome { 1 } else { 3 };
    let mib_size_log2 = 4u32; // 64x64 superblocks (128 already rejected)
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
            num_bits_width: seq.seq_header.num_bits_width,
            num_bits_height: seq.seq_header.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            ..Default::default()
        },
        tile_info: tile_limits(mi_cols, mi_rows, mib_size_log2),
        num_planes,
        separate_uv_delta_q: c.separate_uv_delta_q,
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
            subsampling_x: c.subsampling_x,
            subsampling_y: c.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        // coded_lossless / all_lossless are inputs to this writer-mirror
        // reader but are stream facts on the decode side: parse as
        // non-lossless and hard-error afterwards if the quant params imply
        // lossless (the parse of a lossless stream is invalid).
        ..Default::default()
    };

    let mut rb = ReadBitBuffer::new(payload);
    let p = read_uncompressed_header(&mut rb, &cfg);

    // --- envelope gates, in bitstream order ---
    if p.prefix.show_existing_frame {
        return Err("show_existing_frame".into());
    }
    if p.prefix.frame_type != 0 {
        return Err(format!("frame_type {} (KEY only)", p.prefix.frame_type));
    }
    if !p.prefix.show_frame {
        return Err("unshown frame".into());
    }
    if p.prefix.disable_cdf_update {
        return Err("disable_cdf_update (driver always adapts)".into());
    }
    if p.prefix.allow_screen_content_tools {
        return Err("allow_screen_content_tools (palette/intrabc signaling)".into());
    }
    if p.frame_size.superres_upscaled_width != s.max_frame_width
        || p.frame_size.superres_upscaled_height != s.max_frame_height
    {
        return Err("frame_size_override (frame != sequence max dims)".into());
    }
    if p.frame_size.scale_denominator != 8 {
        return Err("superres scaled".into());
    }
    if p.allow_intrabc {
        return Err("intrabc".into());
    }
    if p.tile_info.cols != 1 || p.tile_info.rows != 1 {
        return Err(format!(
            "{}x{} tiles (single tile only)",
            p.tile_info.cols, p.tile_info.rows
        ));
    }
    if p.quant.using_qmatrix {
        return Err("quantization matrices".into());
    }
    let q = &p.quant;
    if q.base_qindex == 0
        && q.y_dc_delta_q == 0
        && q.u_dc_delta_q == 0
        && q.u_ac_delta_q == 0
        && q.v_dc_delta_q == 0
        && q.v_ac_delta_q == 0
    {
        // Also invalidates this parse (cfg.coded_lossless was false).
        return Err("(coded-)lossless stream".into());
    }
    if p.segmentation.enabled {
        return Err("segmentation".into());
    }
    let lf = &p.loopfilter;
    if lf.filter_level != [0, 0] || lf.filter_level_u != 0 || lf.filter_level_v != 0 {
        return Err(format!(
            "loop-filter levels [{},{},{},{}] != 0 (deblocking not applied)",
            lf.filter_level[0], lf.filter_level[1], lf.filter_level_u, lf.filter_level_v
        ));
    }

    let tile_data_off = if is_obu_frame {
        rb.byte_align();
        // Single tile group, single tile: the tile-group header codes nothing
        // (tiles_log2 == 0), then byte-aligns — the tile data starts here.
        let (ts, te, _) = read_tile_group_header(&mut rb, 0);
        debug_assert_eq!((ts, te), (0, 0));
        rb.byte_align();
        Some(rb.bytes_read())
    } else {
        None
    };
    Ok(ParsedFrame {
        header: p,
        tile_data_off,
    })
}

/// Decode a full AV1 KEY-frame bitstream (a temporal unit as emitted by
/// aomenc / `aom_codec_av1_cx`: temporal delimiter + sequence header + frame)
/// to cropped planes. Hard-errors on anything outside the documented envelope.
pub fn decode_frame_obus(data: &[u8]) -> Result<FrameDecode, String> {
    let mut pos = 0usize;
    let mut seq: Option<SequenceHeaderObu> = None;
    let mut pending_header: Option<FrameHeaderObu> = None;
    let mut decoded: Option<FrameDecode> = None;

    while pos < data.len() {
        let h = read_obu_header(&data[pos..]).ok_or("bad OBU header")?;
        if !h.obu_has_size_field {
            return Err("OBU without size field".into());
        }
        let (size, size_len) =
            uleb_decode(&data[pos + h.header_len..]).ok_or("bad OBU size leb128")?;
        let body = pos + h.header_len + size_len;
        let end = body + size as usize;
        if end > data.len() {
            return Err("OBU size past end of data".into());
        }
        let payload = &data[body..end];

        match h.obu_type {
            2 => {} // OBU_TEMPORAL_DELIMITER (empty)
            1 => {
                // OBU_SEQUENCE_HEADER
                let mut rb = ReadBitBuffer::new(payload);
                let sh = read_sequence_header_obu(&mut rb);
                let s = &sh.seq_header;
                if s.sb_size_128 {
                    return Err("128x128 superblocks".into());
                }
                if s.enable_cdef {
                    return Err("CDEF enabled (sequence) — not applied by this driver".into());
                }
                if s.enable_restoration {
                    return Err("loop restoration enabled (sequence)".into());
                }
                if sh.film_grain_params_present {
                    return Err("film grain enabled (sequence)".into());
                }
                if s.force_screen_content_tools == 1 {
                    return Err("screen content tools forced on (sequence)".into());
                }
                let c = &sh.color_config;
                if !c.monochrome {
                    let ss = (c.subsampling_x, c.subsampling_y);
                    if !matches!(ss, (0, 0) | (1, 0) | (1, 1)) {
                        return Err(format!("unsupported subsampling {ss:?}"));
                    }
                }
                seq = Some(sh);
            }
            3 => {
                // OBU_FRAME_HEADER
                let sh = seq.as_ref().ok_or("frame header before sequence header")?;
                let pf = parse_frame_header(sh, payload, false)?;
                pending_header = Some(pf.header);
            }
            4 | 6 => {
                // OBU_TILE_GROUP | OBU_FRAME
                if decoded.is_some() {
                    return Err("second frame in stream (single KEY frame only)".into());
                }
                let sh = seq.as_ref().ok_or("frame before sequence header")?;
                let (header, tile_data) = if h.obu_type == 6 {
                    let pf = parse_frame_header(sh, payload, true)?;
                    let off = pf.tile_data_off.unwrap();
                    (pf.header, &payload[off..])
                } else {
                    let header = pending_header
                        .take()
                        .ok_or("tile group without frame header")?;
                    // tiles_log2 == 0 (single tile): the tile-group header
                    // codes nothing; data starts byte-aligned at offset 0.
                    (header, payload)
                };
                decoded = Some(decode_tile_payload(sh, &header, tile_data)?);
            }
            5 | 15 => {} // OBU_METADATA | OBU_PADDING — content-neutral
            t => return Err(format!("unsupported OBU type {t}")),
        }
        pos = end;
    }

    decoded.ok_or_else(|| "no frame in stream".into())
}

/// Run the tile decoder over the (single) tile payload and crop the planes.
fn decode_tile_payload(
    seq: &SequenceHeaderObu,
    p: &FrameHeaderObu,
    tile_data: &[u8],
) -> Result<FrameDecode, String> {
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let width = s.max_frame_width as usize;
    let height = s.max_frame_height as usize;
    let (ss_x, ss_y) = if c.monochrome {
        (1usize, 1usize)
    } else {
        (c.subsampling_x as usize, c.subsampling_y as usize)
    };

    let cfg = KfTileConfig {
        mi_rows: mi_dim(s.max_frame_height),
        mi_cols: mi_dim(s.max_frame_width),
        bd: c.bit_depth,
        monochrome: c.monochrome,
        subsampling_x: ss_x,
        subsampling_y: ss_y,
        cdef_bits: p.cdef.cdef_bits as u32,
        disable_edge_filter: !s.enable_intra_edge_filter,
        enable_filter_intra: s.enable_filter_intra,
        tx_mode: if p.tx_mode_select {
            TxMode::Select
        } else {
            TxMode::Largest
        },
        reduced_tx_set: p.reduced_tx_set_used,
        base_qindex: p.quant.base_qindex,
        y_dc_delta_q: p.quant.y_dc_delta_q,
        u_dc_delta_q: p.quant.u_dc_delta_q,
        u_ac_delta_q: p.quant.u_ac_delta_q,
        v_dc_delta_q: p.quant.v_dc_delta_q,
        v_ac_delta_q: p.quant.v_ac_delta_q,
        delta_q_present: p.delta_q.delta_q_present,
        delta_q_res: p.delta_q.delta_q_res,
        delta_lf_present: p.delta_q.delta_lf_present,
        delta_lf_multi: p.delta_q.delta_lf_multi,
        delta_lf_res: p.delta_q.delta_lf_res,
    };
    let mut cdfs = KfFrameContext::default_for_qindex(cfg.base_qindex);
    let mut dec = OdEcDec::new(tile_data);
    let t = decode_tile_kf(&mut dec, &cfg, &mut cdfs, 0);

    // Crop the mi-aligned recon to the frame dims.
    let mut y = vec![0u16; width * height];
    for r in 0..height {
        y[r * width..(r + 1) * width].copy_from_slice(&t.recon[r * t.stride..r * t.stride + width]);
    }
    let (width_uv, height_uv) = if c.monochrome {
        (0, 0)
    } else {
        ((width + ss_x) >> ss_x, (height + ss_y) >> ss_y)
    };
    let mut u = vec![0u16; width_uv * height_uv];
    let mut v = vec![0u16; width_uv * height_uv];
    for r in 0..height_uv {
        u[r * width_uv..(r + 1) * width_uv]
            .copy_from_slice(&t.recon_u[r * t.stride_uv..r * t.stride_uv + width_uv]);
        v[r * width_uv..(r + 1) * width_uv]
            .copy_from_slice(&t.recon_v[r * t.stride_uv..r * t.stride_uv + width_uv]);
    }

    Ok(FrameDecode {
        y,
        u,
        v,
        width,
        height,
        width_uv,
        height_uv,
        bit_depth: c.bit_depth,
        monochrome: c.monochrome,
        subsampling_x: ss_x,
        subsampling_y: ss_y,
        base_qindex: p.quant.base_qindex,
        filter_level: [
            p.loopfilter.filter_level[0],
            p.loopfilter.filter_level[1],
            p.loopfilter.filter_level_u,
            p.loopfilter.filter_level_v,
        ],
        tx_mode_select: p.tx_mode_select,
        reduced_tx_set: p.reduced_tx_set_used,
        delta_q_present: p.delta_q.delta_q_present,
    })
}
