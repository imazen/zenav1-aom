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

pub mod rd_close;

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
use aom_encode::pack::{LrPackParams, pack_tile, pack_tile_lr};
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
use aom_entropy::partition::KfFrameContext;
use aom_entropy::rb::ReadBitBuffer;
use aom_entropy::lr::{LrFrameConfig, RESTORE_NONE as LR_RESTORE_NONE};
use aom_loopfilter::frame::{LfFrameBuf, LfMiGrid, LfParams, loop_filter_frame};
use aom_quant::{Dequants, Quants, av1_build_quantizer, av1_dc_quant_qtx, set_q_index};
use aom_restore::pick::{LrPlanePixels, LrSearchInput, LrSearchSf, pick_filter_restoration};
use aom_sys_ref as c;
use aom_txb::cost_tokens_from_cdf;

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
            aom_entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
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
        let path = corpus_dir().join(format!("{vector}.ivf"));
        let ivf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{vector}: conformance vector missing at {path:?} ({e}); fetch via \
                 `python3 xtask/conformance.py --fetch --scope intra`"
            )
        });
        let (w, h) = ivf_hdr_dims(&ivf);
        let tus = ivf_temporal_units(&ivf);
        DecodeCell {
            label: label.to_string(),
            tu: tus[0].clone(),
            w,
            h,
        }
    }

    /// C-oracle decode (init + decode + plane copy + destroy).
    pub fn c_decode(&self) -> c::RefDecodedFrame {
        c::ref_decode_av1_kf(&self.tu, self.w, self.h)
    }

    /// Port decode (parse + tile decode + post-filters + crop-out).
    pub fn port_decode(&self) -> aom_decode::frame::FrameDecode {
        aom_decode::frame::decode_frame_obus(&self.tu)
            .unwrap_or_else(|e| panic!("{}: port rejected the KEY frame: {e}", self.label))
    }

    /// Setup-time validation: the port's planes are byte-identical to C's.
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
pub fn decode_cells() -> Vec<DecodeCell> {
    vec![
        DecodeCell::from_vector("dec_64x64", "av1-1-b8-01-size-64x64"),
        DecodeCell::from_vector("dec_196x196", "av1-1-b8-01-size-196x196"),
        DecodeCell::from_vector("dec_352x288_q00", "av1-1-b8-00-quantizer-00"),
        DecodeCell::from_vector("dec_352x288_q32", "av1-1-b8-00-quantizer-32"),
        DecodeCell::from_vector("dec_352x288_q63", "av1-1-b8-00-quantizer-63"),
    ]
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

impl EncodeCell {
    /// Real-content cell: decode the first KEY frame of a conformance vector
    /// via the C oracle and (optionally) crop an SB-aligned window —
    /// exactly the KB-6 real-image gate's recipe, so byte-exactness of the
    /// port on these cells is already a landed CI gate at speed 0.
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
        assert!(off_x + w <= fw && off_y + h <= fh, "{label}: crop exceeds frame");
        assert!(off_x % 2 == 0 && off_y % 2 == 0, "{label}: crop offset must be even");
        let (cox, coy) = (off_x >> ss_x, off_y >> ss_y);
        let (cw, ch) = if mono { (0, 0) } else { ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y) };
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

    /// [`Self::c_encode`] with `--enable-restoration=1`
    /// (`AV1E_SET_ENABLE_RESTORATION`) — the reference stream for the
    /// loop-restoration-search parity gate.
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
    pub fn port_encode(&self, bootstrap: &[u8]) -> Vec<u8> {
        self.port_encode_impl(bootstrap, false)
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
        self.port_encode_impl(bootstrap, true)
    }

    fn port_encode_impl(&self, bootstrap: &[u8], lr_stage: bool) -> Vec<u8> {
        let (w, h, mono, ss_x, ss_y, bd) = (self.w, self.h, self.mono, self.ss_x, self.ss_y, self.bd);
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

        let cfg = FrameHeaderObu {
            prefix: FrameHeaderPrefix {
                reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
                decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
                equal_picture_interval: seq.timing_info.equal_picture_interval,
                frame_presentation_time_length: seq
                    .decoder_model_info
                    .frame_presentation_time_length as u32,
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
            film_grain_params_present: seq.film_grain_params_present,
            ..Default::default()
        };

        let mut rb = ReadBitBuffer::new(frame_payload);
        let mut p = read_uncompressed_header(&mut rb, &cfg);
        assert!(!p.prefix.show_existing_frame);
        assert_eq!(p.prefix.frame_type, 0, "frame_type must be KEY");
        assert!(
            p.quant.base_qindex > 0,
            "lossless cells are out of this harness's scope"
        );
        let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
        assert_eq!(tiles_log2, 0, "single-tile envelope only");
        let allintra = self.usage == 2;

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

        let mut kf_write = KfFrameContext::default_for_qindex(qindex);
        let real = derive_real_costs(&kf_write, s.enable_filter_intra);
        let rdmult = av1_compute_rd_mult_based_on_qindex(
            bd,
            FrameUpdateType::Kf,
            qindex,
            TuneMetric::Psnr,
            if allintra { EncMode::Allintra } else { EncMode::Good },
        );

        // Partial-SB support: CEIL the SB walk and replicate-extend the
        // source into the SB-aligned overhang (the chroma_ss_e2e recipe).
        let (cw, ch) = if mono { (0, 0) } else { ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y) };
        let n_sb_x = ((mi_cols + SB_MI - 1) / SB_MI).max(1);
        let n_sb_y = ((mi_rows + SB_MI - 1) / SB_MI).max(1);
        let sb_px_w = n_sb_x as usize * 64;
        let sb_px_h = n_sb_y as usize * 64;
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

        let speed = self.speed;
        let sf = SpeedFeatures::set_allintra(speed, p.allow_screen_content_tools, false);
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
            // Stock encode is QM-off (the allintra default; QM cells live in
            // the qm_encode_witness gate, not this harness).
            qm_levels: None,
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
            partition_cdfs: &real.partition_cdf,
            allintra,
            speed,
            qindex,
            enable_filter_intra: s.enable_filter_intra,
            enable_tx64: true,
            enable_rect_tx: true,
            intra_pruning_with_hog: if allintra {
                sf.intra_pruning_with_hog != 0
            } else {
                true
            },
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
            qm_levels: None,
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
        let trees = pack_tile(
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
            SB_MI,
            SB,
        );
        assert_eq!(
            trees.len(),
            (n_sb_x * n_sb_y) as usize,
            "{}: pack_tile must walk every SB",
            self.label
        );
        let our_tile_bytes = enc.done().to_vec();

        // Port-derived loop-filter level. allintra `lpf_pick` is DUAL for
        // speed 0..=3 and NON_DUAL for speed >= 4 (speed_features.c:496).
        let mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb_x, SB_MI, SB);
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
        };
        let derived_lf = pick_filter_level(&lf_frame, allintra, 0, allintra && speed >= 4);
        p.loopfilter.filter_level = derived_lf.filter_level;
        p.loopfilter.filter_level_u = derived_lf.filter_level_u;
        p.loopfilter.filter_level_v = derived_lf.filter_level_v;

        // ---- loop-restoration ENCODER stage (`--enable-restoration` parity).
        // C pipeline (encoder.c `loopfilter_frame` -> `cdef_restoration_frame`):
        // apply the picked deblock levels -> [CDEF off in this envelope] ->
        // `av1_pick_filter_restoration` on (source, deblocked recon) -> pack
        // the tile with the per-RU params interleaved at each SB root. The
        // restoration DECISIONS (frame types, unit size, per-RU params) are
        // derived by the port's own search — never copied from the bootstrap.
        let mut our_tile_bytes = our_tile_bytes;
        if lr_stage {
            assert!(
                s.enable_restoration,
                "port_encode_lr needs an enable_restoration=1 bootstrap stream"
            );
            assert!(!p.coded_lossless, "is_restoration_used excludes all-lossless");

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
                tile_sb_rows: vec![(0, n_sb_y)],
                tile_sb_cols: vec![(0, n_sb_x)],
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
                let mut kf2 = KfFrameContext::default_for_qindex(qindex);
                let mut ry2 = src_y_strided.clone();
                let mut ru2 = src_u_strided.clone();
                let mut rv2 = src_v_strided.clone();
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
                    0,
                    0,
                    n_sb_y,
                    n_sb_x,
                    SB_MI,
                    SB,
                    Some(&lr_pack),
                );
                assert_eq!(
                    trees2.len(),
                    (n_sb_x * n_sb_y) as usize,
                    "{}: LR repack must walk every SB",
                    self.label
                );
                our_tile_bytes = enc2.done().to_vec();
            }
        }

        assemble_frame_obu_payload_single_tile(&p, tiles_log2, &our_tile_bytes)
    }

    /// Setup-time validation: the port's assembled frame OBU payload is
    /// byte-identical to the C reference stream's. Returns the reference
    /// stream for reuse as the bench-loop bootstrap.
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
/// (`set_good_speed_features_framesize_independent` + the qindex-dependent
/// unit-size bounds). SPEED 0 IS VERIFIED-EQUAL TO DEFAULTS (the good arm
/// sets no LR fields at speed 0 and the single-size qindex rule requires
/// `speed >= 3` for GOOD) — GOOD speed-0 cells are safe. Speed >= 1 is a
/// TRANSCRIPTION FROM PARTIAL NOTES, not yet verified line-by-line:
/// // HANDOFF: verify each speed bracket against speed_features.c before
/// // gating GOOD speed>=1 cells. Known line anchors (v3.14.1):
/// //   :1220-1221  dual_sgr_penalty_level=1, enable_sgr_ep_pruning=1 (speed>=1)
/// //   :1272-1274  prune_wiener_based_on_src_var=1, prune_sgr_based_on_wiener=1,
/// //               disable_loop_restoration_chroma = boosted ? 0 : 1 (speed>=2)
/// //   :1352-1358  prune_sgr_based_on_wiener = screen?1:2,
/// //               prune_wiener_based_on_src_var=2,
/// //               use_downsampled_wiener_stats=1 (speed>=4? CHECK bracket)
/// //   :1452-1453  enable_sgr_ep_pruning=2,
/// //               disable_wiener_coeff_refine_search=true (speed>=5? CHECK)
/// //   :648-649    dual_sgr_penalty_level = boosted?1:3,
/// //               switchable_lr_with_bias_level=1 (which fn/bracket? CHECK —
/// //               it appeared in the earlier grep inside a speed>=? block of
/// //               the good framesize-DEPENDENT or rt arm)
/// // For a single KEY frame `boosted` (frame_is_boosted) is TRUE.
pub fn lr_search_sf_good(
    speed: i32,
    qindex: i32,
    w: usize,
    h: usize,
    allow_screen_content_tools: bool,
) -> LrSearchSf {
    let mut sf = LrSearchSf::default();
    if speed == 0 {
        // Verified: identical to defaults + the full 64..256 descent below.
    } else {
        // HANDOFF: unverified brackets — do NOT gate GOOD speed>=1 cells until
        // each line is re-read from speed_features.c (anchors above).
        let boosted = true; // single KEY frame
        if speed >= 1 {
            sf.dual_sgr_penalty_level = 1;
            sf.enable_sgr_ep_pruning = 1;
        }
        if speed >= 2 {
            sf.prune_wiener_based_on_src_var = 1;
            sf.prune_sgr_based_on_wiener = 1;
            sf.disable_loop_restoration_chroma = !boosted;
        }
        if speed >= 4 {
            // HANDOFF: bracket guess for :1352-1358 — verify.
            sf.prune_sgr_based_on_wiener = if allow_screen_content_tools { 1 } else { 2 };
            sf.prune_wiener_based_on_src_var = 2;
            sf.use_downsampled_wiener_stats = true;
        }
        if speed >= 5 {
            // HANDOFF: bracket guess for :1452-1453 — verify.
            sf.enable_sgr_ep_pruning = 2;
            sf.disable_wiener_coeff_refine_search = true;
        }
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
    cells.push(EncodeCell::synthetic_diag("enc_s4_128_cq32", 128, 128, 32, 4));
    cells
}
