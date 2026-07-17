//! KB-4 (#31) CHROMA decode-both localizer (SCRATCH DIAGNOSTIC — not committed).
//!
//! The mono localizer (`kb4_bd10_rd_localize.rs`) only exercises the luma RD
//! path. KB-4's PRIMARY symptom is bd10 4:4:4/4:2:2 chroma. This file reproduces
//! the non-420 chroma cells, decodes BOTH the real aomenc stream and the port's
//! own re-wrapped stream with the (bit-exact) port decoder, and pins the FIRST
//! divergent leaf — reporting full CHROMA fields (uv_mode, angle_delta_uv, cfl,
//! txbs_uv) so we can classify: chroma edge-filter MODE flip (keystone class) vs
//! coefficient/trellis near-tie (modes match, only eob/coeffs differ).

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::obu_assemble::assemble_obu_frame_single_tile;
use aom_encode::pack::pack_tile;
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::speed_features::SpeedFeatures;
use aom_entropy::enc::OdEcEnc;
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::partition::{KfFrameContext, get_partition_subsize};
use aom_entropy::rb::ReadBitBuffer;
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
const SB: usize = 12;
const SB_MI: i32 = 16;
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const PARTITION_NAMES: [&str; 10] = [
    "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
];
const YMODE: [&str; 13] = [
    "DC", "V", "H", "D45", "D135", "D113", "D157", "D203", "D67", "SMOOTH", "SMOOTH_V", "SMOOTH_H",
    "PAETH",
];
fn uvmode_name(m: i32) -> String {
    // UV modes 0..12 mirror Y; 13 = UV_CFL_PRED.
    if m == 13 {
        "CFL".to_string()
    } else if (0..13).contains(&m) {
        YMODE[m as usize].to_string()
    } else {
        format!("uv?{m}")
    }
}
fn is_directional_uv(m: i32) -> bool {
    (1..=8).contains(&m) // V,H,D45,D135,D113,D157,D203,D67
}

fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}
fn raw_obu_span(bytes: &[u8], want_type: u32) -> &[u8] {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_end = after_header + size_bytes + size as usize;
        if hdr.obu_type == want_type {
            return &bytes[pos..payload_end];
        }
        pos = payload_end;
    }
    panic!("no OBU of type {want_type} found");
}
fn tile_log2(blk_size: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
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
#[allow(clippy::too_many_arguments)]
fn replay_tree(
    tree: &[i8],
    cursor: &mut usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
    out: &mut Vec<(i32, i32, usize, i8)>,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    let p = tree[*cursor];
    out.push((mi_row, mi_col, bsize, p));
    *cursor += 1;
    if p as usize == 3 {
        let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
        let subsize = get_partition_subsize(bsize, p as i32) as usize;
        replay_tree(tree, cursor, mi_row, mi_col, subsize, mi_rows, mi_cols, out);
        replay_tree(tree, cursor, mi_row, mi_col + hbs, subsize, mi_rows, mi_cols, out);
        replay_tree(tree, cursor, mi_row + hbs, mi_col, subsize, mi_rows, mi_cols, out);
        replay_tree(tree, cursor, mi_row + hbs, mi_col + hbs, subsize, mi_rows, mi_cols, out);
    }
}

/// Encode one chroma case at `bd`/`ss` with `tex` content, run the port pack,
/// decode BOTH streams, and pin the first divergent leaf (with chroma fields).
#[allow(clippy::too_many_arguments)]
fn localize_chroma(
    w: usize,
    h: usize,
    ss_x: usize,
    ss_y: usize,
    bd: u8,
    cq_level: i32,
    luma: impl Fn(usize, usize) -> u16,
    chroma: impl Fn(usize, usize) -> u16,
) -> bool {
    c::ref_init();
    let (mono, usage) = (false, 2u32);
    let maxv = (1u16 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = luma(r, col).min(maxv);
        }
    }
    let (cw, ch) = ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            let val = chroma(r, col).min(maxv);
            u[r * cw + col] = val;
            v[r * cw + col] = val;
        }
    }

    let bytes = c::ref_encode_av1_kf(
        &y, &u, &v, w, h, i32::from(bd), mono, ss_x as i32, ss_y as i32, cq_level, 0, false, false,
        usage, 0, false,
    );
    assert!(!bytes.is_empty());

    let obus = walk_obus(&bytes);
    let seq_payload = obus.iter().find(|(t, _)| *t == OBU_SEQUENCE_HEADER).map(|(_, p)| *p).unwrap();
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);
    let (_ft, frame_payload) = obus.iter().find(|(t, _)| *t == OBU_FRAME).map(|(t, p)| (*t, *p)).unwrap();

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
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length as u32,
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

    let mut rb = ReadBitBuffer::new(frame_payload);
    let p = read_uncompressed_header(&mut rb, &cfg);
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "single tile");

    let qindex = p.quant.base_qindex;
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(
        bd, p.quant.y_dc_delta_q, p.quant.u_dc_delta_q, p.quant.u_ac_delta_q,
        p.quant.v_dc_delta_q, p.quant.v_ac_delta_q, &mut quants, &mut deq, 0,
    );
    let rows_y = set_q_index(&quants, &deq, qindex as usize, 0);
    let rows_u = set_q_index(&quants, &deq, qindex as usize, 1);
    let rows_v = set_q_index(&quants, &deq, qindex as usize, 2);

    let mut kf_write = KfFrameContext::default_for_qindex(qindex);
    let real = derive_real_costs(&kf_write, s.enable_filter_intra);
    let allintra = usage == 2;
    let sf = SpeedFeatures::set_allintra(0, p.allow_screen_content_tools, false);
    let rdmult = av1_compute_rd_mult_based_on_qindex(
        bd, FrameUpdateType::Kf, qindex, TuneMetric::Psnr,
        if allintra { EncMode::Allintra } else { EncMode::Good },
    );

    let stride = 320.max(w + 4);
    let mut src_y_strided = vec![0u16; stride * (h + 4)];
    for r in 0..h {
        src_y_strided[r * stride..r * stride + w].copy_from_slice(&y[r * w..r * w + w]);
    }
    let mut src_u_strided = vec![0u16; stride * (h + 4)];
    let mut src_v_strided = vec![0u16; stride * (h + 4)];
    for r in 0..ch {
        src_u_strided[r * stride..r * stride + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
        src_v_strided[r * stride..r * stride + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
    }

    let env = SbEncodeEnv {
        sb_size: SB,
        mi_rows,
        mi_cols,
        tile_row_start: 0,
        tile_col_start: 0,
        tile_row_end: 1 << 16,
        tile_col_end: 1 << 16,
        monochrome: mono,
        ss_x,
        ss_y,
        bd,
        lossless: p.coded_lossless,
        reduced_tx_set_used: p.reduced_tx_set_used,
        disable_edge_filter: !s.enable_intra_edge_filter,
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
        enable_optimize_b: if p.coded_lossless {
            TrellisOptType::NoTrellisOpt
        } else {
            TrellisOptType::FullTrellisOpt
        },
        use_chroma_trellis_rd_mult: allintra,
        coeff_costs_y: &real.coeff_costs_y,
        coeff_costs_uv: &real.coeff_costs_uv,
        tx_type_costs: &real.tx_type_costs_y,
    };
    let pick_cfg = PickFrameCfg {
        mode_costs: &real.mode_costs,
        tx_size_costs: &real.tx_size_costs,
        skip_costs: &real.skip_costs,
        tx_type_costs_y: &real.tx_type_costs_y,
        pol: &sf.tx_type_search_policy(false, 0),
        uv_lp: &UvLoopPolicy::speed0_allintra(),
        intra_uv_mode_cost: &real.mode_costs.intra_uv_mode_cost,
        cfl_costs: &real.cfl_costs,
        partition_costs: &real.partition_costs,
        allintra,
        speed: 0,
        qindex,
        enable_filter_intra: s.enable_filter_intra,
        enable_tx64: true,
        enable_rect_tx: true,
        intra_pruning_with_hog: if allintra { sf.intra_pruning_with_hog != 0 } else { true },
        enable_rect_partitions: true,
        less_rectangular_check_level: if allintra {
            sf.less_rectangular_check_level
        } else {
            i32::from(allintra)
        },
        max_partition_size: 15,
        min_partition_size: 0,
        enable_1to4_partitions: true,
        enable_ab_partitions: true,
        allow_screen_content_tools: p.allow_screen_content_tools,
    };
    let pack_cfg = aom_encode::pack::PackCfg {
        enable_filter_intra: s.enable_filter_intra,
        tx_mode_is_select: p.tx_mode_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
        allow_screen_content_tools: p.allow_screen_content_tools,
    };

    let mut recon_y = src_y_strided.clone();
    let mut recon_u = src_u_strided.clone();
    let mut recon_v = src_v_strided.clone();
    let mut enc = OdEcEnc::new();
    let n_sb = (mi_cols / SB_MI).max(1);
    let _trees = pack_tile(
        &mut enc, &env, &pick_cfg, &pack_cfg, &mut kf_write, &mut recon_y, &mut recon_u,
        &mut recon_v, 0, 0, n_sb, n_sb, SB_MI, SB,
    );
    let our_tile_bytes = enc.done().to_vec();

    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER);
    let our_frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &our_tile_bytes, false, 0);
    let mut our_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    our_stream.extend_from_slice(seq_hdr_raw);
    our_stream.extend_from_slice(&our_frame_obu);

    let (t_real, _c1, _h1) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("decode REAL failed: {e}"));
    let (t_ours, _c2, _h2) = aom_decode::frame::decode_frame_obus_prefilter(&our_stream)
        .unwrap_or_else(|e| panic!("decode OURS failed: {e}"));

    // Partition tree divergence.
    let mut real_seq = Vec::new();
    let mut ours_seq = Vec::new();
    replay_tree(&t_real.tree, &mut 0, 0, 0, SB, mi_rows, mi_cols, &mut real_seq);
    replay_tree(&t_ours.tree, &mut 0, 0, 0, SB, mi_rows, mi_cols, &mut ours_seq);
    let mut partition_div: Option<(i32, i32, usize, i8, i8)> = None;
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        if (r.0, r.1, r.2) != (o.0, o.1, o.2) {
            break;
        }
        if r.3 != o.3 {
            partition_div = Some((r.0, r.1, r.2, r.3, o.3));
            break;
        }
    }

    // First divergent leaf, decode-block order, with classification.
    let mut leaf_div: Option<String> = None;
    for rbk in &t_real.blocks {
        if let Some(ob) = t_ours
            .blocks
            .iter()
            .find(|b| b.mi_row == rbk.mi_row && b.mi_col == rbk.mi_col)
        {
            let y_flip = ob.bsize != rbk.bsize
                || ob.partition != rbk.partition
                || ob.info.y_mode != rbk.info.y_mode
                || ob.info.angle_delta_y != rbk.info.angle_delta_y
                || ob.info.use_filter_intra != rbk.info.use_filter_intra
                || ob.tx_size != rbk.tx_size;
            let uv_flip = ob.info.uv_mode != rbk.info.uv_mode
                || ob.info.angle_delta_uv != rbk.info.angle_delta_uv
                || ob.info.cfl_alpha_idx != rbk.info.cfl_alpha_idx
                || ob.info.cfl_joint_sign != rbk.info.cfl_joint_sign;
            let y_txb_diff = ob.txbs != rbk.txbs;
            let uv_txb_diff = ob.txbs_uv != rbk.txbs_uv;
            if y_flip || uv_flip || y_txb_diff || uv_txb_diff {
                let class = if uv_flip {
                    "CHROMA-MODE-FLIP (uv_mode/adl_uv/cfl) <== keystone class if uv directional"
                } else if y_flip {
                    "LUMA-MODE-FLIP (partition/y_mode/adl_y/fi/tx)"
                } else if uv_txb_diff && !y_txb_diff {
                    "CHROMA-COEFF-NEARTIE (modes match; only txbs_uv differ)"
                } else if y_txb_diff && !uv_txb_diff {
                    "LUMA-COEFF-NEARTIE (modes match; only txbs differ)"
                } else {
                    "BOTH-COEFF-NEARTIE (modes match; luma+chroma txbs differ)"
                };
                leaf_div = Some(format!(
                    "(mi_row={}, mi_col={}) CLASS={class}\n     real bsize={} part={} y={}({}) adly={} uv={}({}) adluv={} cfl_a={} cfl_s={} tx={} \n          txbs_y={:?}\n          txbs_uv={:?}\n     ours bsize={} part={} y={}({}) adly={} uv={}({}) adluv={} cfl_a={} cfl_s={} tx={} \n          txbs_y={:?}\n          txbs_uv={:?}",
                    rbk.mi_row, rbk.mi_col,
                    rbk.bsize, rbk.partition, rbk.info.y_mode, YMODE.get(rbk.info.y_mode as usize).unwrap_or(&"?"),
                    rbk.info.angle_delta_y, rbk.info.uv_mode, uvmode_name(rbk.info.uv_mode),
                    rbk.info.angle_delta_uv, rbk.info.cfl_alpha_idx, rbk.info.cfl_joint_sign, rbk.tx_size,
                    rbk.txbs, rbk.txbs_uv,
                    ob.bsize, ob.partition, ob.info.y_mode, YMODE.get(ob.info.y_mode as usize).unwrap_or(&"?"),
                    ob.info.angle_delta_y, ob.info.uv_mode, uvmode_name(ob.info.uv_mode),
                    ob.info.angle_delta_uv, ob.info.cfl_alpha_idx, ob.info.cfl_joint_sign, ob.tx_size,
                    ob.txbs, ob.txbs_uv,
                ));
                // Extra: is the divergent block's uv_mode directional?
                if uv_flip {
                    let _ = is_directional_uv(rbk.info.uv_mode);
                }
                break;
            }
        }
    }

    // First divergent recon pixel — luma AND chroma.
    let mut recon_div_y: Option<(usize, usize, u16, u16)> = None;
    'ry: for row in 0..t_real.height.min(t_ours.height) {
        for col in 0..t_real.width.min(t_ours.width) {
            let rv = t_real.recon[row * t_real.stride + col];
            let ov = t_ours.recon[row * t_ours.stride + col];
            if rv != ov {
                recon_div_y = Some((row, col, rv, ov));
                break 'ry;
            }
        }
    }
    let mut recon_div_uv: Option<(usize, usize, u16, u16)> = None;
    if !t_real.recon_u.is_empty() && !t_ours.recon_u.is_empty() {
        'ru: for row in 0..t_real.height_uv.min(t_ours.height_uv) {
            for col in 0..t_real.width_uv.min(t_ours.width_uv) {
                let rv = t_real.recon_u[row * t_real.stride_uv + col];
                let ov = t_ours.recon_u[row * t_ours.stride_uv + col];
                if rv != ov {
                    recon_div_uv = Some((row, col, rv, ov));
                    break 'ru;
                }
            }
        }
    }

    let name = match (ss_x, ss_y) {
        (0, 0) => "444",
        (1, 0) => "422",
        (1, 1) => "420",
        _ => "chroma",
    };
    let decisions_match =
        partition_div.is_none() && leaf_div.is_none() && recon_div_y.is_none() && recon_div_uv.is_none();
    eprintln!(
        "\n=== bd{bd} {w}x{h} {name} cq{cq_level} (qindex={qindex}) === {}",
        if decisions_match { "MATCH".to_string() } else { "DIVERGE".to_string() }
    );
    if decisions_match {
        return true;
    }

    // Decoder check: port full-decode of C's stream vs C decode (luma+chroma) to
    // prove the port PREDICTION/recon is bit-exact -> pure ENCODER RD decision.
    let port_full = aom_decode::frame::decode_frame_obus(&bytes)
        .unwrap_or_else(|e| panic!("port full-decode of C stream failed: {e}"));
    let c_full = c::ref_decode_av1_kf(&bytes, w, h);
    let ccw = (w + ss_x) >> ss_x;
    let cch = (h + ss_y) >> ss_y;
    let mut dec_div: Option<String> = None;
    'd: for row in 0..h {
        for col in 0..w {
            let pv = port_full.y[row * port_full.width + col];
            let cv = c_full.y[row * w + col];
            if pv != cv {
                dec_div = Some(format!("LUMA ({row},{col}) port={pv} C={cv}"));
                break 'd;
            }
        }
    }
    if dec_div.is_none() && !port_full.u.is_empty() {
        'du: for row in 0..cch {
            for col in 0..ccw {
                let pv = port_full.u[row * port_full.width_uv + col];
                let cv = c_full.u[row * ccw + col];
                if pv != cv {
                    dec_div = Some(format!("CHROMA-U ({row},{col}) port={pv} C={cv}"));
                    break 'du;
                }
            }
        }
    }
    match &dec_div {
        Some(m) => eprintln!("  [DECODER CHECK] port-decode(C-stream) != C-decode at {m}  <== PORT PREDICTION diverges from C"),
        None => eprintln!("  [DECODER CHECK] port-decode(C-stream) == C-decode: port prediction/recon bit-exact -> divergence is a pure ENCODER RD decision"),
    }

    if let Some((mi_row, mi_col, bsize, pr, po)) = partition_div {
        eprintln!(
            "  >>> FIRST PARTITION DIVERGENCE at (mi_row={mi_row}, mi_col={mi_col}, bsize={bsize}): real=PARTITION_{} ({pr}) ours=PARTITION_{} ({po})",
            PARTITION_NAMES[pr as usize], PARTITION_NAMES[po as usize]
        );
    }
    if let Some(d) = &leaf_div {
        eprintln!("  >>> FIRST LEAF MISMATCH at {d}");
    }
    if let Some((row, col, rv, ov)) = recon_div_y {
        eprintln!("  >>> FIRST RECON-Y divergence at (row={row},col={col}): real={rv} ours={ov}");
    }
    if let Some((row, col, rv, ov)) = recon_div_uv {
        eprintln!("  >>> FIRST RECON-U divergence at (row={row},col={col}): real={rv} ours={ov}");
    }
    false
}

fn tex_luma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        let base = ((r * 37 + cc * 23) as u32) & mask;
        let hf = if (r ^ cc) & 1 == 1 { mask / 12 } else { 0 };
        (base ^ hf) as u16
    }
}
fn tex_chroma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        let base = ((r * 19 + cc * 29) as u32) & mask;
        let hf = if (r + cc) % 3 == 0 { mask / 20 } else { 0 };
        (base ^ hf) as u16
    }
}

#[test]
fn kb4_chroma_localize_bd10_non420() {
    // The 4 documented KB-4 non-420 cells. 3 now match; 444 128x128 diverges.
    let mut _any = false;
    for &(ss_x, ss_y) in &[(0usize, 0usize), (1usize, 0usize)] {
        for &sz in &[64usize, 128] {
            let m = localize_chroma(sz, sz, ss_x, ss_y, 10, 32, tex_luma(0x3ff), tex_chroma(0x3ff));
            _any |= !m;
        }
    }
    // Also probe bd12 444 to see if the same leaf class appears deeper.
    let _ = localize_chroma(128, 128, 0, 0, 12, 32, tex_luma(0xfff), tex_chroma(0xfff));
    eprintln!("\n=== KB-4 chroma localize complete ===");
}
