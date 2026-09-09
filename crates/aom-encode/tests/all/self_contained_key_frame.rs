//! **The bootstrap-free encoder gate.** Proves that
//! [`aom_encode::key_frame::encode_key_frame`] emits a complete, decodable AV1
//! KEY-frame temporal unit with **no C bytes anywhere in the path** — the port
//! authors its own `OBU_TEMPORAL_DELIMITER`, its own sequence-header OBU and
//! its own frame header, then packs its own tile.
//!
//! # Why this file exists
//!
//! Every other encoder harness in this repo runs real libaom FIRST and builds
//! the port's stream on top of the result: `aom-bench`'s
//! `port_encode(bootstrap: &[u8])` parses C's sequence + frame headers, and
//! `aom-encode/tests/avif_parity.rs` splices C's sequence-header OBU bytes
//! verbatim in front of the port's frame OBU (see that file's own header). Those
//! gates prove the tile payload; they cannot prove the port can produce a
//! stream. [`no_seq_header_stream_is_rejected_by_the_c_decoder`] MEASURES that
//! gap instead of asserting it from source: the same port frame OBU, with the
//! sequence header removed, is refused by the real C decoder.
//!
//! # What each test asserts
//!
//! | test | claim |
//! |---|---|
//! | [`no_seq_header_stream_is_rejected_by_the_c_decoder`] | the gap is real AND the decode gate can go red: a frame OBU with no sequence header does not decode |
//! | [`self_contained_key_frame_byte_matches_real_aomenc`] | the port's WHOLE temporal unit is byte-identical to `shim_encode_av1_kf`'s (TD + seq + frame(s)), i.e. every derived header field equals C's |
//! | [`self_contained_key_frame_decodes_to_the_same_pixels`] | the real C decoder AND the port decoder both decode the port's own stream, to the same pixels C's own stream decodes to |
//! | [`mutated_sequence_header_is_caught`] | mutation proof: perturbing ONE derived header field (the coded `base_qindex`) makes the pixel gate fail and the byte gate fail — neither is vacuous |
//! | [`open_divergences_are_pinned`] | the cells that are NOT byte-identical, each with a MEASURED attribution asserted (which half of the frame OBU diverges), self-promoting so a fix goes red |
//! | [`refuses_configurations_it_has_no_gate_for`] | the shell returns [`KeyFrameError`] instead of silently mis-encoding outside its envelope |
//! | [`coded_lossless_reconstructs_the_source_exactly`] | at `--cq-level 0` the frame is `coded_lossless`, so BOTH decoders must return the encoder's own input on every plane — the property C cannot arbitrate inside the `HBD_OPEN` band |
//!
//! # Envelope
//!
//! ALL-INTRA, `--cpu-used` 0..=9, SB64, palette + IntraBC off — exactly
//! `aom_sys_ref::ref_encode_av1_kf`'s configuration, so the byte comparison is
//! like-for-like. All four (CDEF, loop-restoration) combinations are swept, as
//! are mandatory multi-tile frames. Real aomenc's ALLINTRA default is CDEF
//! **off** with restoration **on** (`av1_cx_iface.c:3067` sets
//! `enable_cdef = 0` for `AOM_USAGE_ALL_INTRA`) and is byte-gated at every
//! speed. The `key_frame` module documents the axes that are not wired yet.

use aom_dsp::entropy::header::{
    FrameHeaderObu, read_sequence_header_obu, read_uncompressed_header,
};
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_encode::key_frame::{
    KeyFrameConfig, KeyFrameError, KeyFramePlanes, derive_frame_header, derive_tile_info,
    encode_key_frame,
};
use aom_encode::screen_detect::ScreenContentDecision;
use aom_sys_ref as c;

const OBU_TEMPORAL_DELIMITER: u32 = 2;
const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;

/// Split an AV1 byte stream into `(obu_type, whole-OBU byte span)`.
fn walk_obus(bytes: &[u8]) -> Vec<(u32, std::ops::Range<usize>)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) = aom_dsp::entropy::leb128::uleb_decode(&bytes[after_header..])
            .expect("valid leb128 size");
        let end = after_header + size_bytes + size as usize;
        out.push((hdr.obu_type, pos..end));
        pos = end;
    }
    out
}

/// Split a stream's `OBU_FRAME` into (parsed frame header, uncompressed-header
/// bit length, tile-payload bytes). The reader needs a `cfg` whose fields gate
/// its conditional reads; that cfg is built with the PORT's own
/// `derive_frame_header` off the stream's own sequence header, so this works on
/// both the port's stream and real aomenc's.
fn split_frame_obu(stream: &[u8]) -> (FrameHeaderObu, usize, Vec<u8>) {
    let obus = walk_obus(stream);
    let seq_payload = {
        let span = obus
            .iter()
            .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
            .map(|(_, s)| s.clone())
            .expect("stream has a sequence header");
        // Strip the OBU header + leb128 size to reach the payload.
        let hdr = read_obu_header(&stream[span.start..]).expect("obu header");
        let after = span.start + hdr.header_len;
        let (sz, szb) = aom_dsp::entropy::leb128::uleb_decode(&stream[after..]).expect("leb128");
        stream[after + szb..after + szb + sz as usize].to_vec()
    };
    let seq = read_sequence_header_obu(&mut ReadBitBuffer::new(&seq_payload));
    let frame_payload = {
        let span = obus
            .iter()
            .find(|(t, _)| *t == OBU_FRAME)
            .map(|(_, s)| s.clone())
            .expect("stream has a frame OBU");
        let hdr = read_obu_header(&stream[span.start..]).expect("obu header");
        let after = span.start + hdr.header_len;
        let (sz, szb) = aom_dsp::entropy::leb128::uleb_decode(&stream[after..]).expect("leb128");
        stream[after + szb..after + szb + sz as usize].to_vec()
    };
    let kf_cfg = KeyFrameConfig::allintra_speed0(
        seq.seq_header.max_frame_width as usize,
        seq.seq_header.max_frame_height as usize,
        seq.color_config.bit_depth as u8,
        seq.color_config.monochrome,
        seq.color_config.subsampling_x as usize,
        seq.color_config.subsampling_y as usize,
        32,
    );
    let mi_dim = |px: i32| ((px + 7) & !7) >> 2;
    let tile_info = derive_tile_info(
        mi_dim(seq.seq_header.max_frame_width),
        mi_dim(seq.seq_header.max_frame_height),
        4,
        0,
        0,
    );
    let mut reader_cfg = derive_frame_header(
        &kf_cfg,
        &seq,
        &ScreenContentDecision::detection_disabled(),
        tile_info,
    );
    // `read_uncompressed_header` takes `coded_lossless` / `all_lossless` from
    // the CALLER (`header.rs:3138-3159`) rather than recomputing them from the
    // qindex it just read, and those two flags gate the loop-filter, CDEF,
    // loop-restoration and `tx_mode` reads. The cfg above is built at a
    // hardcoded cq 32, so a **coded-lossless (cq 0)** stream would be read
    // against `coded_lossless = false` and EVERY field after
    // `quantization_params` -- the loop-filter levels included -- would be
    // garbage, silently mis-attributing any cq-0 divergence. `base_qindex` is
    // read before all of them, so one probe pass is enough to learn the
    // stream's own quantizer and re-read with the flags it implies. (Found
    // 2026-09-03 by the first cq-0 pin, which classified as
    // `TilePayloadAndHeader` off LF levels [0, 27] on a frame whose loop-filter
    // syntax is not even present.)
    let probe = read_uncompressed_header(&mut ReadBitBuffer::new(&frame_payload), &reader_cfg);
    let q = &probe.quant;
    let lossless = q.base_qindex == 0
        && q.y_dc_delta_q == 0
        && q.u_dc_delta_q == 0
        && q.u_ac_delta_q == 0
        && q.v_dc_delta_q == 0
        && q.v_ac_delta_q == 0;
    reader_cfg.coded_lossless = lossless;
    // No superres in this envelope, so AllLossless == CodedLossless.
    reader_cfg.all_lossless = lossless;
    let mut rb = ReadBitBuffer::new(&frame_payload);
    let p = read_uncompressed_header(&mut rb, &reader_cfg);
    let bits = rb.bit_position();
    let tile_start = bits.div_ceil(8);
    (p, bits, frame_payload[tile_start..].to_vec())
}

/// The content classes the sweep covers. Deterministic and cheap so the gate
/// is reproducible on any host; no corpus dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Content {
    /// Flat mid-grey — the trivially-lossless control.
    Flat,
    /// Diagonal luminance ramp — smooth, DC/near-DC dominated.
    Gradient,
    /// Diagonal gradient + vertical bars (period 16) + a fine ripple. Same
    /// shape as `encoder_gate_e2e_byte_match.rs`'s strong-LF generator, so the
    /// loop-filter levels the port derives are genuinely non-trivial.
    Texture,
    /// Deterministic hash noise — worst case for the transform/trellis.
    Noise,
    /// 8x8 black/white checkerboard — a screen-content-ish extreme that
    /// exercises the palette/IntraBC detector's counters.
    Checker,
}

fn content_sample(kind: Content, r: usize, col: usize) -> i32 {
    match kind {
        Content::Flat => 128,
        Content::Gradient => (32 + (r + col) * 150 / 256) as i32,
        Content::Texture => {
            let grad = 32 + (r + col) * 150 / 256;
            let bar = if (col / 16) % 2 == 0 { 0 } else { 45 };
            let ripple = if (r + col) % 2 == 0 { 14 } else { -14 };
            grad as i32 + bar + ripple
        }
        Content::Noise => {
            let mut x = (r as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add((col as u32).wrapping_mul(40503));
            x ^= x >> 13;
            x = x.wrapping_mul(1_274_126_177);
            x ^= x >> 16;
            (x & 0xff) as i32
        }
        Content::Checker => {
            if ((r / 8) + (col / 8)) % 2 == 0 {
                16
            } else {
                235
            }
        }
    }
}

/// One gate cell.
#[derive(Clone, Debug)]
struct Cell {
    label: String,
    w: usize,
    h: usize,
    bd: u8,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    cq: i32,
    content: Content,
    /// `AV1E_SET_ENABLE_CDEF`.
    cdef: bool,
    /// `AV1E_SET_ENABLE_RESTORATION`.
    lr: bool,
    /// `AOME_SET_CPUUSED`.
    speed: i32,
    /// `AV1E_SET_TILE_COLUMNS`/`_ROWS` (the log2 value). `(0, 0)` (the
    /// default) means "no explicit request" and keeps `c_stream` on the
    /// plain `ref_encode_av1_kf` path, matching every cell above; a nonzero
    /// value switches `c_stream` to `ref_encode_av1_kf_tiles`.
    tile_cols_log2: i32,
    tile_rows_log2: i32,
    /// `AV1E_SET_SUPERBLOCK_SIZE` (128 when true). Switches `c_stream` to
    /// `ref_encode_av1_kf_sb128` (or `ref_encode_av1_kf_tiles` with its own
    /// `sb_size_128` param, if a tile request is ALSO set).
    sb128: bool,
}

impl Cell {
    fn new(
        label: String,
        w: usize,
        h: usize,
        bd: u8,
        mono: bool,
        ss_x: usize,
        ss_y: usize,
        cq: i32,
        content: Content,
    ) -> Self {
        Cell {
            label,
            w,
            h,
            bd,
            mono,
            ss_x,
            ss_y,
            cq,
            content,
            cdef: false,
            lr: false,
            speed: 0,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            sb128: false,
        }
    }

    /// The same cell at a different `--cpu-used`.
    fn at_speed(mut self, speed: i32) -> Self {
        self.label = format!("{}_s{speed}", self.label);
        self.speed = speed;
        self
    }

    /// The same cell with the two post-filter knobs set.
    fn with_postfilter(mut self, cdef: bool, lr: bool) -> Self {
        self.label = format!("{}_cdef{}_lr{}", self.label, u8::from(cdef), u8::from(lr));
        self.cdef = cdef;
        self.lr = lr;
        self
    }

    /// The same cell with an EXPLICIT `--tile-columns`/`--tile-rows` request
    /// (log2 values). Switches `c_stream` to `ref_encode_av1_kf_tiles`.
    fn with_tile_log2(mut self, cols_log2: i32, rows_log2: i32) -> Self {
        self.label = format!("{}_tc{cols_log2}tr{rows_log2}", self.label);
        self.tile_cols_log2 = cols_log2;
        self.tile_rows_log2 = rows_log2;
        self
    }

    /// The same cell with `--sb-size=128`.
    fn with_sb128(mut self) -> Self {
        self.label = format!("{}_sb128", self.label);
        self.sb128 = true;
        self
    }
}

/// The byte-exact sweep. Axes, and why each is here:
///
/// * **A — quantizer, cq 0..63 step 5 plus 63.** CLAUDE.md's sweep rule: the
///   low-q half carries the same density as the high-q half, because that is
///   where the structural problems hide. cq 0 is the `coded_lossless` arm
///   (ONLY_4X4 + WHT), cq 63 the degenerate high-q one.
/// * **B — chroma format x bit depth**, the full 4x3 grid (mono / 4:2:0 /
///   4:2:2 / 4:4:4 x bd 8/10/12). Profile 0, 1 and 2 are all reached.
/// * **C — size ladder from tiny to large**, 16x16 up to 512x512. The tiny end
///   is where the fixed header cost dominates (a 16x16 frame is 39 bytes, of
///   which ~12 are framing) and the large end is where the per-pixel work does.
/// * **D — crops and partial superblocks**, including 1x1/4x4/8x8 and sizes
///   whose mi grid is not a whole number of SB64s. 258x258 and 262x262 are
///   REGRESSION LOCKS on the 2026-09-02 screen-detector fix (they coded
///   `allow_screen_content_tools = 1` against C's 0 while the detector was
///   handed the crop instead of the 8-aligned `y_width`/`y_height`).
/// * **E — content classes** at two sizes.
/// * **F — low-q density**, cq 1..19 step 2, where every byte matters.
/// * **N — the CODED-LOSSLESS arm (`--cq-level 0`) in depth**, added
///   2026-09-03 with the zenavif#45 fix. `base_qindex == 0` with no deltas
///   makes the frame `coded_lossless`, and `select_tx_mode`
///   (`rdopt_utils.h:391-393`) then forces `ONLY_4X4` — a structurally
///   different encode from every other cell in this file (WHT instead of the
///   DCT family, no coded tx-size symbol, loop filter forced off). Axis A had
///   exactly ONE cq-0 cell (64x64 4:2:0 bd8 Texture at speed 0), and the arm
///   that broke was reachable at every other coordinate: J sweeps cq 0 over
///   {mono, 4:2:0, 4:2:2, 4:4:4} x bd {8, 10, 12} x all five content classes x
///   `--cpu-used` {0, 9}, plus bd8 at the intermediate speeds {3, 6} and a
///   13-point size ladder from 1x1 to 258x258. bd10/bd12 at `--cpu-used` 1..6
///   is the PRE-EXISTING `HBD_OPEN` band (CLAUDE.md T4) and is pinned in
///   [`open_divergences_are_pinned`], not swept — measured 2026-09-03 to be the
///   same band at cq 32, so it is not a lossless finding.
fn sweep_cells() -> Vec<Cell> {
    use Content::*;
    let mut v = Vec::new();
    for cq in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 63] {
        v.push(Cell::new(
            format!("A_cq{cq}_64x64_420_bd8_tex"),
            64,
            64,
            8,
            false,
            1,
            1,
            cq,
            Texture,
        ));
    }
    for (nm, mono, sx, sy) in [
        ("mono", true, 1usize, 1usize),
        ("420", false, 1, 1),
        ("422", false, 1, 0),
        ("444", false, 0, 0),
    ] {
        for bd in [8u8, 10, 12] {
            v.push(Cell::new(
                format!("B_{nm}_bd{bd}_64x64_cq32_tex"),
                64,
                64,
                bd,
                mono,
                sx,
                sy,
                32,
                Texture,
            ));
        }
    }
    for s in [16usize, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512] {
        v.push(Cell::new(
            format!("C_{s}x{s}_420_bd8_cq32_tex"),
            s,
            s,
            8,
            false,
            1,
            1,
            32,
            Texture,
        ));
    }
    for (w, h) in [
        (1usize, 1usize),
        (4, 4),
        (8, 8),
        (66, 34),
        (100, 60),
        (130, 70),
        (200, 200),
        (250, 130),
        (258, 258),
        (262, 262),
        (263, 263),
        (264, 264),
        // These eight were PINNED as "RD near-ties" until 2026-09-02, when the
        // shell started stamping C's clamped tile bounds
        // (`AOMMIN(tile_end, mi_rows/mi_cols)`) instead of a past-the-end
        // sentinel. Reverting the clamp makes all eight diverge again, so they
        // are also the regression lock on that fix.
        (131, 131),
        (132, 132),
        (132, 64),
        (132, 128),
        (196, 196),
        (196, 64),
        (260, 260),
        (261, 261),
    ] {
        v.push(Cell::new(
            format!("D_{w}x{h}_420_bd8_cq32_tex"),
            w,
            h,
            8,
            false,
            1,
            1,
            32,
            Texture,
        ));
    }
    for (nm, k) in [
        ("flat", Flat),
        ("grad", Gradient),
        ("tex", Texture),
        ("noise", Noise),
        ("check", Checker),
    ] {
        for s in [64usize, 128] {
            v.push(Cell::new(
                format!("E_{nm}_{s}x{s}_420_bd8_cq32"),
                s,
                s,
                8,
                false,
                1,
                1,
                32,
                k,
            ));
        }
    }
    for cq in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19] {
        v.push(Cell::new(
            format!("F_cq{cq}_128x128_420_bd8_tex"),
            128,
            128,
            8,
            false,
            1,
            1,
            cq,
            Texture,
        ));
    }
    // G -- the POST-FILTER axis: all four (CDEF, loop restoration) combinations,
    // including BOTH ON, which no pack entry point covered before
    // `pack::pack_tile_from_trees_lr`. (Real aomenc's ALLINTRA default is CDEF
    // OFF with restoration ON -- `av1_cx_iface.c:3067`; that default is swept
    // across every speed in axis H.) Swept
    // across the quantizer range, mono / 4:2:0 / 4:4:4, and three sizes,
    // because the CDEF strength search and the LR unit decision are both
    // content- and qindex-driven (a single cq exercises one branch of each).
    for (cdef, lr) in [(true, false), (false, true), (true, true)] {
        for (w, h, mono, sx, sy, cq) in [
            (64usize, 64usize, false, 1usize, 1usize, 32i32),
            (64, 64, true, 1, 1, 32),
            (64, 64, false, 1, 1, 5),
            (64, 64, false, 1, 1, 63),
            (128, 128, false, 1, 1, 12),
            (128, 128, false, 1, 1, 48),
            (128, 128, false, 0, 0, 32),
            (196, 196, false, 1, 1, 20),
            (256, 256, false, 1, 1, 32),
        ] {
            let nm = if mono {
                "mono"
            } else if sx == 0 {
                "444"
            } else {
                "420"
            };
            v.push(
                Cell::new(
                    format!("G_{w}x{h}_{nm}_bd8_cq{cq}_tex"),
                    w,
                    h,
                    8,
                    mono,
                    sx,
                    sy,
                    cq,
                    Texture,
                )
                .with_postfilter(cdef, lr),
            );
        }
    }
    // H -- the SPEED axis (`--cpu-used`). What is byte-exact, measured
    // 2026-09-02 and bounded by the pins below:
    //   * CDEF OFF (the ALLINTRA default -- `av1_cx_iface.c:3067` sets
    //     `enable_cdef = 0` for usage 2): speeds 0..9, with restoration off or
    //     on. Note the SEQUENCE bit `enable_restoration` is cleared by C at
    //     allintra speed >= 5 (`speed_features.c:2753` &&
    //     `disable_wiener_filter`/`disable_sgr_filter`), which
    //     `derive_sequence_header` models -- so the `lr = true` cells at speeds
    //     5..9 code restoration OFF, exactly like real aomenc.
    //   * CDEF ON: speeds 0..3 only. Speed >= 4 switches
    //     `sf.cdef_pick_method` to the FAST levels, which PARITY.md C1 records
    //     as ported + table-unit-tested but never e2e-gated; measured here,
    //     they diverge on every cell tried (pinned below).
    for speed in 0..=9 {
        for (w, h, cq) in [(64usize, 64usize, 32i32), (128, 128, 12), (128, 128, 32)] {
            for lr in [false, true] {
                let c = Cell::new(
                    format!("H_{w}x{h}_420_bd8_cq{cq}_tex_lr{}", u8::from(lr)),
                    w,
                    h,
                    8,
                    false,
                    1,
                    1,
                    cq,
                    Texture,
                );
                let c = if lr {
                    c.with_postfilter(false, true)
                } else {
                    c
                };
                v.push(c.at_speed(speed));
            }
        }
    }
    for speed in 0..=3 {
        for (w, h, cq) in [(64usize, 64usize, 32i32), (128, 128, 32)] {
            v.push(
                Cell::new(
                    format!("H_{w}x{h}_420_bd8_cq{cq}_tex"),
                    w,
                    h,
                    8,
                    false,
                    1,
                    1,
                    cq,
                    Texture,
                )
                .with_postfilter(true, true)
                .at_speed(speed),
            );
        }
    }
    // I -- MULTI-TILE, in the form `av1_get_tile_limits` MANDATES it: a frame
    // wider than MAX_TILE_WIDTH (4096 px) forces `min_log2_cols > 0`, so
    // libaom's own uniform-spacing default (`--tile-columns=0`) still resolves
    // to 2 tiles at 4160 px and 3 at 8320. Each tile is packed independently
    // with a fresh frame context, exactly as C's `write_modes` does, and
    // assembled through `assemble_multitile_frame_obu_payload_derived`.
    // Byte-exact at speeds 0..6; speeds 7..9 hit the same unlocalized
    // VAR_BASED_PARTITION / nonrd arm the 256x256 pin records (they are LARGE
    // frames, not a tile problem -- 4160x64 is byte-exact at 0..6).
    for (w, h, cq, speed) in [
        (4160usize, 64usize, 32i32, 0i32),
        (4160, 64, 32, 3),
        (4160, 64, 32, 6),
        (4160, 64, 12, 0),
        (4160, 64, 55, 0),
        (4160, 192, 32, 6),
        (4224, 128, 40, 0),
        (8320, 64, 32, 0),
        (8320, 64, 32, 6),
        // The EXACT-boundary cells. `set_tile_info` (encoder.c:386-390)
        // recomputes the column minimum with `(max_width_sb << k) <= sb_cols`,
        // one stricter than `av1_get_tile_limits`' own `tile_log2` (`<`). They
        // differ by exactly one when `sb_cols` is an exact
        // multiple-by-power-of-two of `max_width_sb`: at SB64 that is a frame
        // whose mi width rounds to `sb_cols == 64` (4033..4096 px, two tiles)
        // or `== 128` (8192 px, four tiles). Bite-proved 2026-09-02 by
        // weakening the loop to `<`: 4033x64, 4096x64 and 8192x64 diverge,
        // 4032x64 / 4097x64 / 4160x64 / 2048x64 do not.
        (4032, 64, 32, 0),
        (4033, 64, 32, 0),
        (4096, 64, 32, 0),
        (4097, 64, 32, 0),
        (8192, 64, 32, 0),
    ] {
        v.push(
            Cell::new(
                format!("I_{w}x{h}_420_bd8_cq{cq}_tex_multitile"),
                w,
                h,
                8,
                false,
                1,
                1,
                cq,
                Texture,
            )
            .at_speed(speed),
        );
    }
    // J -- HIGH BIT DEPTH x SPEED x TILE COUNT. Axis B (bit depth) above is
    // only exercised at speed 0 / single-tile 64x64; axis I (multi-tile) is
    // only exercised at bd8. Neither crosses bd10/12 with a non-zero speed,
    // which is EXACTLY the reach of the pre-existing `HBD_OPEN` / `b10_64`
    // pin recorded elsewhere in this repo (`CLAUDE.md` coverage queue,
    // `s4cov_qm_axis.rs` / `config_permutations.rs`): "bd10 AND bd12,
    // `--cpu-used` 1..6, LUMA-borne, reaches 4:4:4 + mono". This shell is a
    // SEPARATE code path from those harnesses (no bootstrap, self-derived
    // headers), so whether it inherits the same divergence is a measured
    // question, not an assumption -- and per the PREREQ-AOM-STANDALONE issue
    // (#15) admission note, the T4 HBD_OPEN pin is exactly why the standalone
    // path needs its own gate rather than inheriting the harness's C compare.
    // J1: bd x speed x tile-count at 4:2:0, isolating the speed reach.
    //
    // MEASURED 2026-09-03 (the full 2..6 x {single,multi} grid was run before
    // this list was pruned to the passing subset -- see `open_divergences_
    // are_pinned` for the failing half, pinned there with the same data):
    //   * bd10 single-tile 128x128: byte-exact at s1,s2,s3,s6; diverges s4,s5.
    //   * bd10 multi-tile 4160x64:  byte-exact at s6 ONLY; diverges s1..s5.
    //   * bd12 single-tile 128x128: byte-exact at s4,s5,s6; diverges s1,s2,s3.
    //   * bd12 multi-tile 4160x64:  diverges at EVERY speed 1..6.
    // So the divergence is not "bd10/12 x speed 1..6" as a block (the
    // pre-existing HBD_OPEN pin's own description) -- tile count measurably
    // widens or narrows the reach, and bd10 vs bd12 move in OPPOSITE
    // directions on the single-tile axis (bd10 fails in the middle of the
    // range, bd12 fails at the start of it).
    for (bd, speed) in [(10u8, 1i32), (10, 2), (10, 3), (10, 6)] {
        v.push(
            Cell::new(
                format!("J_bd{bd}_128x128_420_cq32_tex"),
                128,
                128,
                bd,
                false,
                1,
                1,
                32,
                Texture,
            )
            .at_speed(speed),
        );
    }
    v.push(
        Cell::new(
            "J_bd10_4160x64_420_cq32_tex".into(),
            4160,
            64,
            10,
            false,
            1,
            1,
            32,
            Texture,
        )
        .at_speed(6),
    );
    for speed in [4i32, 5, 6] {
        v.push(
            Cell::new(
                "J_bd12_128x128_420_cq32_tex".to_string(),
                128,
                128,
                12,
                false,
                1,
                1,
                32,
                Texture,
            )
            .at_speed(speed),
        );
    }
    // bd12 multi-tile 4160x64 has NO passing speed in 1..6 -- entirely pin
    // material, see `open_divergences_are_pinned`.
    //
    // J2: bd x chroma format at a speed EACH bd is measured byte-exact at
    // (single-tile) -- "LUMA-borne, reaches 4:4:4 + mono" from the same pin
    // description, at a speed that is not itself part of the divergence.
    for (nm, mono, sx, sy) in [
        ("mono", true, 1usize, 1usize),
        ("422", false, 1, 0),
        ("444", false, 0, 0),
    ] {
        v.push(
            Cell::new(
                format!("J2_bd10_{nm}_128x128_cq32_tex"),
                128,
                128,
                10,
                mono,
                sx,
                sy,
                32,
                Texture,
            )
            .at_speed(6),
        );
        v.push(
            Cell::new(
                format!("J2_bd12_{nm}_128x128_cq32_tex"),
                128,
                128,
                12,
                mono,
                sx,
                sy,
                32,
                Texture,
            )
            .at_speed(6),
        );
    }
    // K -- TINY / ODD / NARROW-TALL GEOMETRIES matching the historical
    // "14 tiny" poison class from the pre-`encode_key_frame` `port_encode`
    // differential fleet run (`avifaom-enc-20260830`, PARITY.md C3,
    // zenmetrics `benchmarks/avifaom_round3_2026-08-30_open.tsv`): a single
    // ODD-WIDTH, narrow, tall source (59x128) panicked the OLD bootstrap-
    // driven port at nearly every quantizer tried, at speeds 4 and 6. The
    // exact source pixels are not preserved (only larger poison cells' planes
    // were staged), so this reproduces the GEOMETRY class -- odd width, one
    // SB64 column or less, several SB64 rows -- with this file's own content
    // generators, on THIS shell's code path (which never existed at the time
    // of that run), across the two poisoned speeds plus a speed-0 control.
    for (w, h) in [(59usize, 128usize), (78, 128), (115, 128)] {
        for speed in [0i32, 4, 6] {
            for (nm, k) in [("tex", Texture), ("check", Checker)] {
                v.push(
                    Cell::new(
                        format!("K_{w}x{h}_420_bd8_cq32_{nm}"),
                        w,
                        h,
                        8,
                        false,
                        1,
                        1,
                        32,
                        k,
                    )
                    .at_speed(speed),
                );
            }
        }
    }
    // K2: the worst-hit historical dimension (59x128) across the quantizer
    // ladder at its poisoned speed -- the original class hit nearly every cq
    // tried, so this checks it is not a narrow single-cq coincidence.
    for cq in [0, 10, 20, 30, 40, 50, 63] {
        v.push(
            Cell::new(
                format!("K2_cq{cq}_59x128_420_bd8_tex"),
                59,
                128,
                8,
                false,
                1,
                1,
                cq,
                Texture,
            )
            .at_speed(4),
        );
    }
    // L -- EXPLICIT `--tile-columns`/`--tile-rows` (`derive_tile_info`'s
    // `tile_cols_log2_cfg`/`tile_rows_log2_cfg` params, previously always
    // called with `0, 0`; the C-side reference (`shim_encode_av1_kf_tiles`,
    // `ref_encode_av1_kf_tiles`) already existed, unwired, from an earlier
    // decoder-track multi-tile landing). Three shapes:
    //   * FORCE more tiles than the uniform-spacing default on a frame that
    //     would otherwise be single-tile (256x256 -> 2 columns / 2 rows).
    //   * REQUEST columns on a frame that is ALREADY mandatory multi-tile
    //     (4160x64 mandates >= 2 columns at SB64) -- above the minimum, to
    //     prove the explicit request composes with the mandatory floor
    //     rather than being overridden by it.
    //   * REQUEST fewer columns (log2=1, "2 columns") than the mandatory
    //     minimum on a large frame (8320x64, whose UNREQUESTED mandatory
    //     grid is already >= 3 columns per axis I's own comment) -- must
    //     CLAMP UP to the true minimum, exactly as C's `set_tile_info` does
    //     (`log2_cols = AOMMAX(tile_columns, min_log2_cols)`), not silently
    //     under-tile. Using a NONZERO-but-insufficient request (rather than
    //     0) exercises `ref_encode_av1_kf_tiles` on both sides, so the test
    //     is checking the clamp path itself, not merely re-running the
    //     already-covered "no explicit request" cell through a different
    //     C shim.
    for (label, w, h, cq, speed, cols_log2, rows_log2) in [
        (
            "L_force2cols_256x256",
            256usize,
            256usize,
            32i32,
            0i32,
            1i32,
            0i32,
        ),
        ("L_force2rows_256x256", 256, 256, 32, 0, 0, 1),
        ("L_force4cols4rows_256x256", 256, 256, 32, 3, 2, 2),
        ("L_above_min_4160x64", 4160, 64, 32, 0, 2, 0),
        ("L_below_min_clamped_8320x64", 8320, 64, 32, 0, 1, 0),
    ] {
        v.push(
            Cell::new(label.to_string(), w, h, 8, false, 1, 1, cq, Texture)
                .at_speed(speed)
                .with_tile_log2(cols_log2, rows_log2),
        );
    }
    // M -- SB128 (`--sb-size=128`). The single-cell probe that used to stand
    // here (128x128, one whole SB128, speed 0) was BYTE-EXACT on the first
    // try, confirming the underlying search/pack machinery genuinely was
    // already bsize-generic (`SbEncodeEnv::sb_size`, `rd_pick_partition_
    // real`'s `bsize` param, `BLOCK_128X128` already used throughout
    // `aom_dsp::entropy::partition`) and only the shell's three hardcoded
    // constants (`mib_size_log2`, `sb_mi`, `sb_block`) plus the sequence-
    // header bit were missing. This is the follow-up matrix: multiple
    // superblocks (256x256 = 2x2, 384x384 = 3x3), a size that is NOT a
    // multiple of 128 (200x150, forcing a partial superblock at the
    // frame edge), speeds spanning the range, CDEF+LR both on (their unit
    // sizing is independently derived from mi geometry, not sb_block, but
    // untested at sb128 until now), bd10, and SB128 composed with an
    // explicit multi-tile request on a frame large enough to mandate tiles
    // regardless of superblock size.
    for (label, w, h, bd, cq, speed, cdef, lr) in [
        (
            "M_128x128",
            128usize,
            128usize,
            8u8,
            32i32,
            0i32,
            false,
            false,
        ),
        ("M_256x256", 256, 256, 8, 32, 0, false, false),
        ("M_256x256", 256, 256, 8, 32, 6, false, false),
        ("M_384x384", 384, 384, 8, 32, 3, false, false),
        ("M_200x150_partial", 200, 150, 8, 32, 0, false, false),
        ("M_256x256", 256, 256, 8, 32, 0, true, true),
        ("M_128x128_bd10", 128, 128, 10, 32, 0, false, false),
        ("M_128x128", 128, 128, 8, 32, 9, false, false),
        ("M_128x128_cq0", 128, 128, 8, 0, 0, false, false),
        ("M_128x128_cq63", 128, 128, 8, 63, 0, false, false),
    ] {
        v.push(
            Cell::new(label.to_string(), w, h, bd, false, 1, 1, cq, Texture)
                .at_speed(speed)
                .with_postfilter(cdef, lr)
                .with_sb128(),
        );
    }
    // SB128 composed with an EXPLICIT multi-tile request (4224x128 mandates
    // >= 2 tile columns regardless of superblock size at SB64; check the
    // mandatory-tile derivation is consistent when sb_block is 128 too, and
    // that a request ABOVE the sb128 mandatory minimum still clamps/derives
    // correctly through `ref_encode_av1_kf_tiles`'s own sb_size_128 param).
    v.push(
        Cell::new(
            "M_4224x128_sb128_tiles".to_string(),
            4224,
            128,
            8,
            false,
            1,
            1,
            32,
            Texture,
        )
        .at_speed(0)
        .with_tile_log2(2, 0)
        .with_sb128(),
    );
    // N -- the CODED-LOSSLESS arm, `--cq-level 0`. `base_qindex == 0` with no
    // segment/delta-q makes the frame `coded_lossless` (`is_coded_lossless`,
    // encodeframe.c:2275), and `select_tx_mode` (rdopt_utils.h:391-393) then
    // returns ONLY_4X4 rather than TX_MODE_SELECT -- so no block codes a
    // tx-size symbol, every transform is the 4x4 WHT, and
    // `pick_filter_level` is skipped. Axis A carried a single cq-0 cell; the
    // arm that broke (zenavif#45) needed a lossless block at BLOCK_32X32 or
    // larger, which only appears once the partition search stops splitting --
    // i.e. at a speed and on content axis A never reached.
    //
    // MEASURED 2026-09-03 (`benchmarks/cq0_lossless_axis_2026-09-03.md`): every
    // cell below is byte-identical to real aomenc. bd10/bd12 at `--cpu-used`
    // 1..6 is NOT, and is the pre-existing `HBD_OPEN` band (bd10 AND bd12,
    // speeds 1..6, luma-borne) -- the SAME band at cq 32, so it is not a
    // lossless finding; it is pinned in `open_divergences_are_pinned`.
    for (nm, mono, sx, sy) in [
        ("mono", true, 1usize, 1usize),
        ("420", false, 1, 1),
        ("422", false, 1, 0),
        ("444", false, 0, 0),
    ] {
        for bd in [8u8, 10, 12] {
            for (cnm, content) in [
                ("flat", Flat),
                ("grad", Gradient),
                ("tex", Texture),
                ("noise", Noise),
                ("chk", Checker),
            ] {
                // Speeds 0 and 9 bracket the whole `--cpu-used` range at every
                // depth; the intermediate speeds are bd8-only because 1..6 at
                // bd10/bd12 is HBD_OPEN.
                let speeds: &[i32] = if bd == 8 { &[0, 3, 6, 9] } else { &[0, 9] };
                for &speed in speeds {
                    v.push(
                        Cell::new(
                            format!("N_{nm}_bd{bd}_{cnm}_64x64_cq0"),
                            64,
                            64,
                            bd,
                            mono,
                            sx,
                            sy,
                            0,
                            content,
                        )
                        .at_speed(speed),
                    );
                }
            }
        }
    }
    // J size ladder at cq 0: the fixed header cost dominates a 1x1 lossless
    // frame and the partition walk dominates 258x258 (a partial superblock),
    // and both are structurally different from the 64x64 cells above.
    for (w, h) in [
        (1usize, 1usize),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (48, 48),
        (96, 96),
        (100, 60),
        (128, 128),
        (130, 70),
        (192, 192),
        (256, 256),
        (258, 258),
    ] {
        for (cnm, content) in [("grad", Gradient), ("tex", Texture)] {
            v.push(
                Cell::new(
                    format!("N_{w}x{h}_420_bd8_{cnm}_cq0"),
                    w,
                    h,
                    8,
                    false,
                    1,
                    1,
                    0,
                    content,
                )
                .at_speed(0),
            );
        }
    }
    v
}

/// A smaller spread for the (slower) two-decoder pixel gate: every axis is
/// represented, so a framing or header regression cannot hide.
fn decode_cells() -> Vec<Cell> {
    let keep = [
        "A_cq0_64x64_420_bd8_tex",
        "A_cq63_64x64_420_bd8_tex",
        "B_mono_bd8_64x64_cq32_tex",
        "B_444_bd12_64x64_cq32_tex",
        "B_422_bd10_64x64_cq32_tex",
        "C_16x16_420_bd8_cq32_tex",
        "C_512x512_420_bd8_cq32_tex",
        "D_1x1_420_bd8_cq32_tex",
        "D_100x60_420_bd8_cq32_tex",
        "D_258x258_420_bd8_cq32_tex",
        "E_flat_64x64_420_bd8_cq32",
        "E_noise_128x128_420_bd8_cq32",
        "F_cq1_128x128_420_bd8_tex",
        "G_64x64_420_bd8_cq32_tex_cdef1_lr0",
        "G_64x64_420_bd8_cq32_tex_cdef0_lr1",
        "G_64x64_420_bd8_cq32_tex_cdef1_lr1",
        "G_256x256_420_bd8_cq32_tex_cdef1_lr1",
        "H_64x64_420_bd8_cq32_tex_lr0_s9",
        "H_128x128_420_bd8_cq12_tex_lr1_cdef0_lr1_s6",
        "H_64x64_420_bd8_cq32_tex_cdef1_lr1_s3",
        "I_4160x64_420_bd8_cq32_tex_multitile_s0",
        "L_force4cols4rows_256x256_s3_tc2tr2",
        "M_256x256_s0_cdef1_lr1_sb128",
        "M_200x150_partial_s0_cdef0_lr0_sb128",
        "M_4224x128_sb128_tiles_s0_tc2tr0_sb128",
        // The coded-lossless arm, one cell per structural axis it changes
        // (ONLY_4X4 + WHT, no tx-size symbol, loop filter off).
        "N_420_bd8_flat_64x64_cq0_s0",
        "N_420_bd8_tex_64x64_cq0_s6",
        "N_444_bd12_noise_64x64_cq0_s0",
        "N_mono_bd10_chk_64x64_cq0_s9",
        "N_258x258_420_bd8_tex_cq0_s0",
    ];
    let picked: Vec<Cell> = sweep_cells()
        .into_iter()
        .filter(|c| keep.contains(&c.label.as_str()))
        .collect();
    // A label that matches NOTHING silently drops a decode cell, and nothing
    // else in this file would notice. Found exactly that way on 2026-09-04:
    // `H_128x128_420_bd8_cq12_tex_lr1_s6` predated `with_postfilter`'s own
    // `_cdef{}_lr{}` label suffix, so the `lr = true` speed cell had been out
    // of the two-decoder gate since that suffix landed (24 cells ran where 25
    // were listed). The list and the sweep must agree by construction.
    let unmatched: Vec<&str> = keep
        .iter()
        .copied()
        .filter(|k| !picked.iter().any(|c| c.label == *k))
        .collect();
    assert!(
        unmatched.is_empty(),
        "decode_cells() lists {} label(s) that no sweep cell has: {unmatched:?}. \
         A stale label is a SILENT coverage loss -- fix the label (or the cell), \
         do not delete the entry",
        unmatched.len()
    );
    picked
}

/// Source planes for a cell: luma from the content generator scaled into the
/// cell's bit depth, chroma flat mid-grey (the convention
/// `encoder_gate_e2e_byte_match.rs` uses -- only the luma decision space is
/// stressed).
fn cell_planes(cell: &Cell) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (w, h) = (cell.w, cell.h);
    let maxv = (1u32 << cell.bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let v8 = content_sample(cell.content, r, col).clamp(0, 255) as u32;
            y[r * w + col] = ((v8 * maxv) / 255) as u16;
        }
    }
    let (cw, ch) = if cell.mono {
        (0, 0)
    } else {
        ((w + cell.ss_x) >> cell.ss_x, (h + cell.ss_y) >> cell.ss_y)
    };
    let mid = (maxv / 2 + 1) as u16;
    (y, vec![mid; cw * ch], vec![mid; cw * ch])
}

fn cell_cfg(cell: &Cell) -> KeyFrameConfig {
    let mut cfg = KeyFrameConfig::allintra_speed0(
        cell.w, cell.h, cell.bd, cell.mono, cell.ss_x, cell.ss_y, cell.cq,
    );
    cfg.enable_cdef = cell.cdef;
    cfg.enable_restoration = cell.lr;
    cfg.cpu_used = cell.speed;
    cfg.tile_columns_log2 = cell.tile_cols_log2;
    cfg.tile_rows_log2 = cell.tile_rows_log2;
    cfg.sb_size_128 = cell.sb128;
    cfg
}

/// Run the port's bootstrap-free encoder for a cell.
fn port_stream(cell: &Cell, y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
    encode_key_frame(KeyFramePlanes { y, u, v }, &cell_cfg(cell))
        .unwrap_or_else(|e| panic!("{}: encode_key_frame refused: {e}", cell.label))
}

/// Real aomenc's stream for the same cell + config. A cell with an explicit
/// tile-columns/rows request (`Cell::with_tile_log2`) goes through
/// `ref_encode_av1_kf_tiles` (which also carries `sb_size_128`, so it
/// composes with `Cell::with_sb128`); SB128-only goes through
/// `ref_encode_av1_kf_sb128`; every other cell keeps the plain
/// `ref_encode_av1_kf` path unchanged.
fn c_stream(cell: &Cell, y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
    let bytes = if cell.tile_cols_log2 != 0 || cell.tile_rows_log2 != 0 {
        c::ref_encode_av1_kf_tiles(
            y,
            u,
            v,
            cell.w,
            cell.h,
            cell.bd as i32,
            cell.mono,
            cell.ss_x as i32,
            cell.ss_y as i32,
            cell.cq,
            cell.speed,
            cell.cdef,
            cell.lr,
            2,
            0,
            false,
            cell.sb128,
            cell.tile_cols_log2,
            cell.tile_rows_log2,
        )
    } else if cell.sb128 {
        c::ref_encode_av1_kf_sb128(
            y,
            u,
            v,
            cell.w,
            cell.h,
            cell.bd as i32,
            cell.mono,
            cell.ss_x as i32,
            cell.ss_y as i32,
            cell.cq,
            cell.speed,
            cell.cdef,
            cell.lr,
            2,
            0,
            false,
            true,
        )
    } else {
        c::ref_encode_av1_kf(
            y,
            u,
            v,
            cell.w,
            cell.h,
            cell.bd as i32,
            cell.mono,
            cell.ss_x as i32,
            cell.ss_y as i32,
            cell.cq,
            cell.speed,
            cell.cdef,
            cell.lr,
            2,
            0,
            false,
        )
    };
    assert!(!bytes.is_empty(), "{}: C encode failed", cell.label);
    bytes
}

/// **The gap, measured.** Take the port's OWN self-contained stream and delete
/// the sequence-header OBU — exactly the stream shape every bootstrapped
/// harness in this repo would produce without C's header. The real C decoder
/// must REFUSE it. This is simultaneously (a) evidence that authoring the
/// sequence header was a real missing capability rather than a cosmetic one and
/// (b) the negative control proving the decode gate below is not vacuous.
#[test]
fn no_seq_header_stream_is_rejected_by_the_c_decoder() {
    c::ref_init();
    let cell = &Cell::new(
        "flat_mono_64x64_cq32".to_string(),
        64,
        64,
        8,
        true,
        1,
        1,
        32,
        Content::Flat,
    );
    let (y, u, v) = cell_planes(cell);
    let full = port_stream(cell, &y, &u, &v);

    let obus = walk_obus(&full);
    let types: Vec<u32> = obus.iter().map(|(t, _)| *t).collect();
    assert_eq!(
        types,
        vec![OBU_TEMPORAL_DELIMITER, OBU_SEQUENCE_HEADER, OBU_FRAME],
        "the port's temporal unit must be TD + sequence header + frame OBU"
    );

    // Positive control: the full stream DOES decode.
    let ok = c::ref_decode_av1_kf(&full, cell.w, cell.h);
    assert_eq!(ok.y.len(), cell.w * cell.h);

    // Negative: same bytes minus the sequence header.
    let mut stripped = Vec::new();
    for (t, span) in &obus {
        if *t != OBU_SEQUENCE_HEADER {
            stripped.extend_from_slice(&full[span.clone()]);
        }
    }
    assert!(stripped.len() < full.len(), "the strip must remove bytes");
    let refused = std::panic::catch_unwind(|| {
        c::ref_decode_av1_kf(&stripped, cell.w, cell.h);
    })
    .is_err();
    assert!(
        refused,
        "a frame OBU with NO sequence header decoded successfully — either the C \
         decoder shim is not reporting errors (the decode gate would be vacuous) or \
         the sequence header is not load-bearing, and both are defects"
    );
    eprintln!(
        "no_seq_header_stream_is_rejected_by_the_c_decoder: {} bytes full / {} bytes stripped; \
         stripped stream refused by the real C decoder",
        full.len(),
        stripped.len()
    );
}

/// The strongest available form of the gate: the port's ENTIRE temporal unit —
/// temporal delimiter, sequence-header OBU and frame OBU — is byte-identical to
/// what real aomenc produces for the same source and config. Every header field
/// the shell derives (profile, level, `num_bits_*`, the reduced-still-picture
/// framing, `base_qindex`, `allow_screen_content_tools`, the tile grid, the
/// loop-filter levels, the `tx_mode` flip) is therefore equal to C's, not
/// merely "decodable".
#[test]
fn self_contained_key_frame_byte_matches_real_aomenc() {
    c::ref_init();
    let cells = sweep_cells();
    let mut exact = 0usize;
    let mut failures = Vec::new();
    for cell in &cells {
        let (y, u, v) = cell_planes(cell);
        let ours = port_stream(cell, &y, &u, &v);
        let theirs = c_stream(cell, &y, &u, &v);
        if ours == theirs {
            exact += 1;
            eprintln!("{}: BYTE-EXACT ({} bytes)", cell.label, ours.len());
        } else {
            let first_diff = ours
                .iter()
                .zip(&theirs)
                .position(|(a, b)| a != b)
                .unwrap_or(ours.len().min(theirs.len()));
            let ours_obus: Vec<(u32, usize)> = walk_obus(&ours)
                .iter()
                .map(|(t, s)| (*t, s.len()))
                .collect();
            let theirs_obus: Vec<(u32, usize)> = walk_obus(&theirs)
                .iter()
                .map(|(t, s)| (*t, s.len()))
                .collect();
            failures.push(format!(
                "{}: port {} bytes vs C {} bytes, first difference at byte {first_diff}; \
                 port OBUs (type, len) {ours_obus:?} vs C {theirs_obus:?}",
                cell.label,
                ours.len(),
                theirs.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} cells byte-exact; failures:\n  {}",
        exact,
        cells.len(),
        failures.join("\n  ")
    );
    eprintln!(
        "self_contained_key_frame_byte_matches_real_aomenc: {exact}/{} cells byte-exact",
        cells.len()
    );
}

/// Decode the port's own bootstrap-free stream with BOTH decoders — real
/// libaom (`aom_codec_av1_dx`) and this repo's own AV1 decoder — and require
/// both to reproduce the pixels real aomenc's own stream decodes to.
#[test]
fn self_contained_key_frame_decodes_to_the_same_pixels() {
    c::ref_init();
    let cells = decode_cells();
    assert!(
        !cells.is_empty(),
        "decode_cells() must not silently select nothing"
    );
    for cell in &cells {
        let (y, u, v) = cell_planes(cell);
        let ours = port_stream(cell, &y, &u, &v);
        let theirs = c_stream(cell, &y, &u, &v);

        let c_ours = c::ref_decode_av1_kf(&ours, cell.w, cell.h);
        let c_theirs = c::ref_decode_av1_kf(&theirs, cell.w, cell.h);
        assert_eq!(
            c_ours.info, c_theirs.info,
            "{}: the C decoder reports different stream info for the port's stream",
            cell.label
        );
        assert_eq!(
            (&c_ours.y, &c_ours.u, &c_ours.v),
            (&c_theirs.y, &c_theirs.u, &c_theirs.v),
            "{}: real-C-decode(port stream) != real-C-decode(aomenc stream)",
            cell.label
        );

        let p_ours = aom_decode::frame::decode_frame_obus(&ours).unwrap_or_else(|e| {
            panic!(
                "{}: port decode of the port's own stream failed: {e}",
                cell.label
            )
        });
        assert_eq!(p_ours.width, cell.w, "{}: decoded width", cell.label);
        assert_eq!(p_ours.height, cell.h, "{}: decoded height", cell.label);
        assert_eq!(
            (&p_ours.y, &p_ours.u, &p_ours.v),
            (&c_ours.y, &c_ours.u, &c_ours.v),
            "{}: port-decode(port stream) != real-C-decode(port stream)",
            cell.label
        );

        // The flat cells are trivially reproducible: the decode must return the
        // source exactly.
        if cell.content == Content::Flat {
            assert_eq!(
                c_ours.y, y,
                "{}: flat decode.y must equal the source",
                cell.label
            );
        }
        eprintln!(
            "{}: decoded {}x{} by both decoders, pixel-equal to real aomenc's decode",
            cell.label, cell.w, cell.h
        );
    }
    eprintln!(
        "self_contained_key_frame_decodes_to_the_same_pixels: {} cells, both decoders",
        cells.len()
    );
}

/// **Mutation proof.** Perturb ONE field the shell derives — the coded
/// `base_qindex` in the frame header — by re-running the encoder at a different
/// `--cq-level` and splicing that frame OBU behind the unmutated sequence
/// header. Both gates above must go red on it:
///
/// * the byte gate: the mutated unit differs from C's;
/// * the pixel gate: the real C decoder returns DIFFERENT pixels.
///
/// Without this, a decode-only gate could pass on any stream that merely
/// parses, and a byte gate could pass on a comparison that never runs.
#[test]
fn mutated_sequence_header_is_caught() {
    c::ref_init();
    // A non-trivial reconstruction (textured 4:2:0 at cq 32).
    let cell = &Cell::new(
        "texture_420_64x64_cq32".to_string(),
        64,
        64,
        8,
        false,
        1,
        1,
        32,
        Content::Texture,
    );
    let (y, u, v) = cell_planes(cell);
    let truth = port_stream(cell, &y, &u, &v);
    let c_truth = c_stream(cell, &y, &u, &v);
    assert_eq!(
        truth, c_truth,
        "{}: precondition — the unmutated cell is byte-exact",
        cell.label
    );

    // The mutation: encode the SAME pixels at a different quantizer and keep
    // only its frame OBU, behind the original (unmutated) TD + sequence header.
    let mut mutated_cell_cfg = cell_cfg(cell);
    mutated_cell_cfg.cq_level = cell.cq + 20;
    let other = encode_key_frame(
        KeyFramePlanes {
            y: &y,
            u: &u,
            v: &v,
        },
        &mutated_cell_cfg,
    )
    .expect("the mutated config is inside the envelope");

    let truth_obus = walk_obus(&truth);
    let other_obus = walk_obus(&other);
    let mut mutated = Vec::new();
    for (t, span) in &truth_obus {
        if *t != OBU_FRAME {
            mutated.extend_from_slice(&truth[span.clone()]);
        }
    }
    let other_frame = other_obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME)
        .map(|(_, s)| s.clone())
        .expect("the mutated encode has a frame OBU");
    mutated.extend_from_slice(&other[other_frame]);

    // (a) the byte gate goes red.
    assert_ne!(
        mutated, c_truth,
        "{}: the mutated temporal unit still byte-equals C's — the byte gate cannot fail",
        cell.label
    );

    // (b) the pixel gate goes red: the mutated stream still DECODES (so this is
    //     a real quantizer change, not a parse failure) but to different pixels.
    let dec_mut = c::ref_decode_av1_kf(&mutated, cell.w, cell.h);
    let dec_truth = c::ref_decode_av1_kf(&truth, cell.w, cell.h);
    assert_ne!(
        dec_mut.y,
        dec_truth.y,
        "{}: the real C decoder returned IDENTICAL luma for a stream encoded at \
         cq {} vs cq {} — the pixel gate cannot fail",
        cell.label,
        cell.cq,
        cell.cq + 20
    );
    eprintln!(
        "mutated_sequence_header_is_caught: byte gate and pixel gate both go red on a \
         one-field mutation (cq {} -> {})",
        cell.cq,
        cell.cq + 20
    );
}

/// The shell REFUSES what it has no gate for, with a named reason, instead of
/// silently producing a stream outside the proven envelope.
#[test]
fn refuses_configurations_it_has_no_gate_for() {
    let base = KeyFrameConfig::allintra_speed0(64, 64, 8, true, 1, 1, 32);
    let y = vec![128u16; 64 * 64];
    let planes = KeyFramePlanes {
        y: &y,
        u: &[],
        v: &[],
    };

    // All four post-filter combinations are SUPPORTED (axis G of the sweep) --
    // asserted here too so this test can never silently become the reason a
    // regression that re-refuses them goes unnoticed.
    for (cdef, lr) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut cfg = base;
        cfg.enable_cdef = cdef;
        cfg.enable_restoration = lr;
        assert!(
            encode_key_frame(planes, &cfg).is_ok(),
            "cdef={cdef} lr={lr} must encode: all four post-filter combinations are gated"
        );
    }

    // `--cpu-used` 0..=9 are all SUPPORTED (axis H of the sweep); only values
    // outside the CLI range are refused.
    for speed in 0..=9 {
        let mut cfg = base;
        cfg.cpu_used = speed;
        assert!(
            encode_key_frame(planes, &cfg).is_ok(),
            "cpu_used={speed} must encode: speeds 0..=9 are gated"
        );
    }
    let mut bad_speed = base;
    bad_speed.cpu_used = 10;
    assert!(matches!(
        encode_key_frame(planes, &bad_speed),
        Err(KeyFrameError::Unsupported(_))
    ));

    let mut good = base;
    good.usage = 0;
    assert!(matches!(
        encode_key_frame(planes, &good),
        Err(KeyFrameError::Unsupported(_))
    ));

    // Plane-size validation, not a panic.
    let short = KeyFramePlanes {
        y: &y[..64],
        u: &[],
        v: &[],
    };
    assert!(matches!(
        encode_key_frame(short, &base),
        Err(KeyFrameError::PlaneSize { plane: 0, .. })
    ));

    // (0, 1) is not an AV1 chroma format.
    let mut bad_ss = base;
    bad_ss.monochrome = false;
    bad_ss.ss_x = 0;
    bad_ss.ss_y = 1;
    let cy = vec![128u16; 64 * 64];
    let cuv = vec![128u16; 64 * 32];
    assert!(matches!(
        encode_key_frame(
            KeyFramePlanes {
                y: &cy,
                u: &cuv,
                v: &cuv
            },
            &bad_ss
        ),
        Err(KeyFrameError::Unsupported(_))
    ));

    // Odd frame sizes at the nonrd / variance-partition speeds must return a
    // Result or a stream -- never panic. KB-34 was exactly such a refusal
    // (`nonrd_pickmode.rs` on a HORZ/VERT leaf), reachable from a 100x100
    // thumbnail, and a library entry point that unwinds instead of returning
    // is a defect regardless of who closed the root.
    for (w, h) in [(100usize, 100usize), (66, 34), (130, 70), (196, 196)] {
        for speed in [7, 8, 9] {
            let cfg = KeyFrameConfig::allintra_speed0(w, h, 8, false, 1, 1, 32);
            let cfg = KeyFrameConfig {
                cpu_used: speed,
                ..cfg
            };
            let y: Vec<u16> = (0..w * h)
                .map(|i| ((i * 7 + (i / w) * 13) % 256) as u16)
                .collect();
            let uv = vec![128u16; w.div_ceil(2) * h.div_ceil(2)];
            let r = std::panic::catch_unwind(|| {
                encode_key_frame(
                    KeyFramePlanes {
                        y: &y,
                        u: &uv,
                        v: &uv,
                    },
                    &cfg,
                )
            });
            assert!(
                r.is_ok(),
                "{w}x{h} at --cpu-used {speed} PANICKED; encode_key_frame must return a \
                 Result, never unwind"
            );
            assert!(
                r.unwrap().is_ok(),
                "{w}x{h} at --cpu-used {speed} was refused; it is inside the envelope"
            );
        }
    }

    // A frame wide enough that `av1_get_tile_limits` MANDATES a tile split
    // ENCODES (axis I of the sweep) rather than being refused.
    let wide = KeyFrameConfig::allintra_speed0(4160, 64, 8, true, 1, 1, 32);
    let wide_y = vec![128u16; 4160 * 64];
    assert!(
        encode_key_frame(
            KeyFramePlanes {
                y: &wide_y,
                u: &[],
                v: &[]
            },
            &wide
        )
        .is_ok(),
        "a mandatory-tile-split frame must encode"
    );
}

/// **Open divergences, PINNED.** These cells are NOT byte-identical to real
/// aomenc, and this test asserts the divergence is STILL PRESENT so a fix flips
/// it red and forces promotion into [`sweep_cells`] (this repo's self-promoting
/// pin convention — see the PARITY.md Tier-3 rows).
///
/// **None is a framing or header-derivation defect in the shell**, and the
/// attribution is MEASURED and ASSERTED per cell (which half of the frame OBU
/// diverges — see `Where` below), not left in prose:
///
/// | cell | measured attribution |
/// |---|---|
/// | `--enable-cdef=1` at `--cpu-used` >= 4 (64x64 4:2:0 cq32) | `HeaderOnly`. `sf.cdef_pick_method` leaves `CDEF_FULL_SEARCH` for the FAST levels at speed >= 4; PARITY.md C1 records those as ported + table-unit-tested but NEVER e2e-gated. Divergent on every cell tried at speeds 4..9 (5 sizes x 6 speeds), and ONLY in the header's `cdef_strengths` set — the per-unit strength indices in the tile payload are byte-identical. Speeds 0..3 are byte-exact and ARE in the sweep. |
/// | `--cpu-used` >= 7 above roughly 3x3 superblocks (256x256 cq32 s7) | `TilePayloadOnly`. One unlocalized VAR_BASED_PARTITION / nonrd arm. Bracket at speed 7: 128x128, 160x160, 192x192, 128x192 and 192x128 are byte-exact, 256x256 and 320x320 are not; at speed 9, 192x192 is not either. |
/// | the same arm through a MANDATORY two-tile frame (4160x64 cq32 s9) | `TilePayloadOnly`. Pinned separately so a tile-assembly regression cannot hide inside the large-frame one: 4160x64 is byte-exact at speeds 0..6 in the sweep, which is what proves the tile assembly is not the problem. |
/// | cq 0 at bd10 `--cpu-used` 6, and at bd12 `--cpu-used` 3 (64x64 4:2:0) | `TilePayloadOnly`. The pre-existing `HBD_OPEN` band (CLAUDE.md T4), observed on the coded-lossless arm (axis N). MEASURED 2026-09-03 over 720 cells x 2 quantizers: the divergent set is exactly bd {10, 12} x `--cpu-used` 1..6 at BOTH cq 0 and cq 32; bd8 is byte-exact at cq 0 across all four formats, five contents and speeds 0/3/6/9, and every depth is byte-exact at speeds 0, 7, 8, 9. So it is not a lossless finding. |
///
/// # What used to be here
///
/// 132x132, 196x196, 260x260 and 261x261 were pinned as "tile-payload RD
/// near-ties". **They were a bug in this shell, not the port's RD**: the pack
/// env carried a past-the-end sentinel for `tile_row_end` / `tile_col_end`
/// instead of C's `AOMMIN(.., mi_rows/mi_cols)` clamp
/// (`av1_tile_set_row` / `_col`). Bite-proved by reverting the clamp — those
/// four plus 131x131, 132x64, 132x128 and 196x64 all diverge with the sentinel
/// and are all byte-identical with it. All eight are now sweep cells and the
/// regression lock on that fix.
///
/// The lesson worth keeping: the *measurement* ("the divergence is in the tile
/// payload, every header field agrees") was right both times; the *inference*
/// ("therefore an RD near-tie") was wrong. `Where` records the measurement;
/// the prose next to it is the part that can be wrong.
///
/// The neighbours bracket the remaining pins: 130x70, 200x200, 250x130,
/// 258x258, 262x262, 263x263, 264x264, 256x256 and 320x320 are byte-exact in
/// [`sweep_cells`], the whole post-filter axis is 27/27, and multi-tile is
/// byte-exact at speeds 0..6.
#[test]
fn open_divergences_are_pinned() {
    c::ref_init();
    /// Which HALF of the frame OBU a pin's divergence lives in. Asserted per
    /// cell, so a pin cannot quietly change character (an RD tie growing a
    /// header defect, say) without going red.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Where {
        /// Every derived frame-header field agrees with C's; only the
        /// entropy-coded tile payload differs.
        TilePayloadOnly,
        /// The tile payload is byte-identical to C's; only a frame-header
        /// field differs.
        #[allow(dead_code)]
        HeaderOnly,
        /// The tile payload differs AND a derived header field differs. For a
        /// KEY frame that is the signature of ONE upstream root, not two: the
        /// loop-filter levels are derived from the port's own reconstruction
        /// (`pick_filter_level` on the phase-1 recon), so an RD divergence that
        /// changes the recon can move them too.
        TilePayloadAndHeader,
    }
    let pins: Vec<(Cell, Where, &str)> = vec![
        (
            Cell::new(
                "PIN_cdef_speed4".into(),
                64,
                64,
                8,
                false,
                1,
                1,
                32,
                Content::Texture,
            )
            .with_postfilter(true, false)
            .at_speed(4),
            Where::HeaderOnly,
            "CDEF FAST search levels: at allintra speed >= 4 `sf.cdef_pick_method` leaves \
             CDEF_FULL_SEARCH, and PARITY.md C1 records the FAST levels as ported + \
             table-unit-tested but NEVER e2e-gated. Measured 2026-09-02: divergent on \
             every `--enable-cdef=1` cell tried at speeds 4..9 (5/5 sizes x 6 speeds), and \
             the divergence is ONLY in the header's `cdef_strengths` set -- the per-unit \
             strength indices in the tile payload are byte-identical. Speeds 0..3 are \
             byte-exact and ARE in the sweep",
        ),
        (
            Cell::new(
                "PIN_4160x64_multitile_speed9".into(),
                4160,
                64,
                8,
                false,
                1,
                1,
                32,
                Content::Texture,
            )
            .at_speed(9),
            Where::TilePayloadOnly,
            "the same speed >= 7 arm as the 256x256 pin, reached through a MANDATORY \
             two-tile frame: 4160x64 is byte-exact at speeds 0..6 (in the sweep), so this \
             is the large-frame nonrd arm and not a tile-assembly defect",
        ),
        (
            Cell::new(
                "PIN_256x256_speed7".into(),
                256,
                256,
                8,
                false,
                1,
                1,
                32,
                Content::Texture,
            )
            .at_speed(7),
            Where::TilePayloadOnly,
            "one unlocalized VAR_BASED_PARTITION / nonrd arm above roughly 3x3 \
             superblocks. MEASURED bracket at speed 7: 128x128, 160x160, 192x192, \
             128x192 and 192x128 are BYTE-EXACT, 256x256 and 320x320 are not; at speed \
             9, 192x192 is not either. So it is size- AND speed-conditional, and the \
             nonrd path itself is not unported -- 64x64 and 128x128 are byte-exact at \
             7, 8 and 9, and multi-tile 4160x64 is byte-exact at 0..6",
        ),
        // The four HBD x speed x tile-count pins below are the failing half of
        // axis J (`sweep_cells()`), MEASURED 2026-09-03 -- see that axis's own
        // comment for the full bracket. Same PORT-SIDE code path as everything
        // above (no bootstrap, self-derived headers); this is a SEPARATE
        // divergence class from the pre-existing `HBD_OPEN` / `b10_64` pin in
        // the bootstrap-driven harnesses (`s4cov_qm_axis.rs` /
        // `config_permutations.rs`), reached through this shell instead, and
        // measurably NOT the same flat "bd10/12 x speed 1..6" shape that pin
        // describes -- tile count moves the reach in a bd-dependent direction.
        (
            Cell::new(
                "PIN_bd10_128x128_speed4".into(),
                128,
                128,
                10,
                false,
                1,
                1,
                32,
                Content::Texture,
            )
            .at_speed(4),
            Where::TilePayloadAndHeader,
            "bd10, single-tile, MID-band speed failure: byte-exact at s1,s2,s3,s6, \
             diverges at s4,s5. Same shape as `HBD_OPEN`/`b10_64` (bd10/12 x speed 1..6, \
             LUMA-borne -- J2's mono/422/444 cells at bd10 s4 diverge identically to \
             4:2:0), reached through the standalone shell instead of a bootstrap-driven \
             harness -- registered as its own pin rather than folded into that one \
             because the two have never been proven to share a root cause, and this \
             shell's speed BAND (s4,s5 only, not the full 1..6) is a new, narrower datum. \
             MEASURED: a derived header field (the loop-filter level) diverges here TOO, \
             unlike both multi-tile HBD pins below where every header field agrees and \
             only the tile payload differs -- single- vs multi-tile is a real split in \
             this family, not just in WHICH speeds fail but in WHERE the divergence \
             lands. `pick_filter_level` runs on the port's own reconstruction, so an \
             RD/coefficient difference at single-tile can cascade into the loop-filter \
             level the same way the 261x261 pin's does; the multi-tile pins' identical \
             LF suggests the per-tile fresh-context reset masks that cascade there",
        ),
        (
            Cell::new(
                "PIN_bd10_4160x64_multitile_speed1".into(),
                4160,
                64,
                10,
                false,
                1,
                1,
                32,
                Content::Texture,
            )
            .at_speed(1),
            Where::TilePayloadOnly,
            "bd10, MANDATORY multi-tile (2 tiles), LOW-band speed failure: byte-exact at \
             s6 ONLY, diverges s1..s5 -- the OPPOSITE band from the single-tile 128x128 \
             pin above (which fails s4,s5 and passes s1,s2,s3,s6) at the SAME bit depth. \
             Tile count is therefore a real axis in this divergence, not a confound: \
             going from one tile to two both widens the failing band (1..5 vs 4..5) and \
             flips which end of it is safe",
        ),
        (
            Cell::new(
                "PIN_bd12_128x128_speed1".into(),
                128,
                128,
                12,
                false,
                1,
                1,
                32,
                Content::Texture,
            )
            .at_speed(1),
            Where::TilePayloadAndHeader,
            "bd12, single-tile, LOW-band speed failure: byte-exact at s4,s5,s6, diverges \
             s1,s2,s3 -- the MIRROR of the bd10 single-tile pin's band (which fails \
             s4,s5 and passes s1,s2,s3,s6). Same speed_features/bit-depth interaction \
             family as `HBD_OPEN`, opposite band per bit depth. Also TilePayloadAndHeader \
             like the bd10 single-tile pin (not TilePayloadOnly like both multi-tile \
             pins) -- the single-vs-multi-tile split in WHERE the divergence lands holds \
             at both bit depths",
        ),
        (
            Cell::new(
                "PIN_bd12_4160x64_multitile_speed1".into(),
                4160,
                64,
                12,
                false,
                1,
                1,
                32,
                Content::Texture,
            )
            .at_speed(1),
            Where::TilePayloadOnly,
            "bd12, MANDATORY multi-tile (2 tiles): diverges at EVERY speed 1..6 tried -- \
             no passing speed at all, unlike every other HBD pin here. The most severe \
             cell in this family; representative of the full band rather than a \
             boundary, so there is no adjacent passing speed to bracket it against. Like \
             the bd10 multi-tile pin, every derived header field (including the \
             loop-filter levels) agrees with C's -- ONLY the tile payload differs, which \
             narrows this family further: at multi-tile, HBD does not perturb header \
             derivation at all, only the coefficient/RD arm",
        ),
        (
            Cell::new(
                "PIN_cq0_bd10_grad".into(),
                64,
                64,
                10,
                false,
                1,
                1,
                0,
                Content::Gradient,
            )
            .at_speed(6),
            Where::TilePayloadOnly,
            "the pre-existing HBD_OPEN band (CLAUDE.md T4: bd10 AND bd12, `--cpu-used` \
             1..6, LUMA-borne, reaches 4:4:4 + mono, qindex-dependent speed reach), \
             observed at cq 0. This is the EXACT coordinate zenavif#45 reported the \
             `tx_size_to_depth` assert at; the assert is fixed (`count_leaf` now writes \
             C's own inequality) and what remains is HBD_OPEN, not a lossless defect. \
             MEASURED 2026-09-03 over 720 cells x 2 quantizers: at cq 0 AND at cq 32 the \
             divergent set is exactly bd {10, 12} x `--cpu-used` 1..6 -- bd8 is 240/240 \
             byte-exact at cq 0 including speeds 3 and 6, and speeds 0, 7, 8 and 9 are \
             byte-exact at every depth (all in the N arm of `sweep_cells`). A lossless \
             root would not be bit-depth- or speed-conditional",
        ),
        (
            Cell::new(
                "PIN_cq0_bd12_tex".into(),
                64,
                64,
                12,
                false,
                1,
                1,
                0,
                Content::Texture,
            )
            .at_speed(3),
            Where::TilePayloadOnly,
            "the same HBD_OPEN band one depth and three speeds over, pinned separately \
             so bd12 cannot close silently behind bd10. Its cq-32 twin diverges too",
        ),
    ];

    for (cell, expect_where, why) in &pins {
        let (expect_where, why) = (*expect_where, *why);
        let (w, h) = (cell.w, cell.h);
        let (y, u, v) = cell_planes(cell);
        let ours = port_stream(cell, &y, &u, &v);
        let theirs = c_stream(cell, &y, &u, &v);
        assert_ne!(
            ours, theirs,
            "{}x{} is now BYTE-EXACT ({why}). That is good news: delete it from \
             open_divergences_are_pinned and add it to sweep_cells(), and say in the \
             commit message what closed it.",
            w, h
        );
        // The divergence must NOT be in the sequence header: the shell's own
        // derivation is proven correct even on the pinned cells.
        let ours_seq = walk_obus(&ours)
            .into_iter()
            .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
            .map(|(_, s)| ours[s].to_vec())
            .expect("port stream has a sequence header");
        let theirs_seq = walk_obus(&theirs)
            .into_iter()
            .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
            .map(|(_, s)| theirs[s].to_vec())
            .expect("C stream has a sequence header");
        assert_eq!(
            ours_seq, theirs_seq,
            "{w}x{h}: the PORT-AUTHORED sequence-header OBU diverges from C's. That is a \
             shell defect, not the pinned {why} — fix it rather than widening this pin."
        );
        // And both streams must still decode to a real frame: a pinned RD tie is
        // a different bitstream, never an invalid one.
        let dec = c::ref_decode_av1_kf(&ours, w, h);
        assert_eq!(
            dec.y.len(),
            w * h,
            "{w}x{h}: the port's pinned stream must decode"
        );
        // WHICH HALF diverges -- the attribution, asserted rather than left in
        // prose. The loop-filter levels are a frame-HEADER field that no tile
        // symbol reads, so a `pick_filter_level` divergence leaves the tile
        // payload byte-identical; an RD near-tie does the opposite.
        let (ours_hdr, _, ours_tile) = split_frame_obu(&ours);
        let (theirs_hdr, _, theirs_tile) = split_frame_obu(&theirs);
        // Every DERIVED frame-header field, so "HeaderOnly" cannot hide a
        // divergence in a field nobody thought to compare. (Reached exactly
        // that way: the CDEF FAST-level pin diverges ONLY in
        // `cdef.cdef_strengths`, with the per-unit indices in the tile payload
        // identical, and an earlier tuple without the CDEF block classified it
        // as "neither half differs".)
        let hdr_key = |p: &FrameHeaderObu| {
            (
                (
                    p.loopfilter.filter_level,
                    p.loopfilter.filter_level_u,
                    p.loopfilter.filter_level_v,
                    p.quant.base_qindex,
                    p.allow_screen_content_tools,
                    p.tx_mode_select,
                ),
                (
                    p.cdef.cdef_damping,
                    p.cdef.cdef_bits,
                    p.cdef.nb_cdef_strengths,
                    p.cdef.cdef_strengths,
                    p.cdef.cdef_uv_strengths,
                ),
                (
                    p.restoration.frame_restoration_type,
                    p.restoration.restoration_unit_size,
                ),
            )
        };
        let header_same = hdr_key(&ours_hdr) == hdr_key(&theirs_hdr);
        let got_where = match (ours_tile == theirs_tile, header_same) {
            (true, false) => Where::HeaderOnly,
            (false, true) => Where::TilePayloadOnly,
            (false, false) => Where::TilePayloadAndHeader,
            (true, true) => panic!(
                "{}: the streams differ but both the tile payload and every checked header \
                 field agree — the divergence is somewhere this test does not look (OBU \
                 framing? a header field not in `hdr_key`?). Investigate before touching \
                 this list.",
                cell.label
            ),
        };
        assert_eq!(
            got_where,
            expect_where,
            "{w}x{h}: the pin changed character. Expected {expect_where:?} ({why}), measured \
             {got_where:?}: port LF {:?}/{}/{} vs C {:?}/{}/{}, port ascs={} C ascs={}, \
             port tx_mode_select={} C={}, port tile {} bytes vs C {} bytes. Re-attribute it \
             before touching this list.",
            ours_hdr.loopfilter.filter_level,
            ours_hdr.loopfilter.filter_level_u,
            ours_hdr.loopfilter.filter_level_v,
            theirs_hdr.loopfilter.filter_level,
            theirs_hdr.loopfilter.filter_level_u,
            theirs_hdr.loopfilter.filter_level_v,
            ours_hdr.allow_screen_content_tools,
            theirs_hdr.allow_screen_content_tools,
            ours_hdr.tx_mode_select,
            theirs_hdr.tx_mode_select,
            ours_tile.len(),
            theirs_tile.len(),
        );
        if expect_where == Where::TilePayloadOnly {
            // Spell the "every derived header field agrees" half out on its own.
            assert_eq!(
                (
                    ours_hdr.loopfilter.filter_level,
                    ours_hdr.quant.base_qindex,
                    ours_hdr.allow_screen_content_tools,
                    ours_hdr.tx_mode_select,
                ),
                (
                    theirs_hdr.loopfilter.filter_level,
                    theirs_hdr.quant.base_qindex,
                    theirs_hdr.allow_screen_content_tools,
                    theirs_hdr.tx_mode_select,
                ),
                "{w}x{h}: a derived header field diverges too -- this pin is no longer a pure \
                 tile-payload RD tie"
            );
        }
        eprintln!(
            "PIN {w}x{h}: still divergent, {got_where:?} ({why}); seq header byte-exact; decodes"
        );
    }
}

/// **Probe for the `av1_determine_sc_tools_with_encoding` gap** (issue #15's
/// SCM-promotion bullet; `encoder_utils.c:1214`, PARITY.md C3). The shell's
/// screen-content decision is the base detector's ONLY -- C additionally runs
/// a two-pass trial encode (forced q >= 244, fixed 32x32 partition) whenever
/// the base decision is OFF and speed features allow it, and can flip the
/// decision ON if screen-content tools would buy >0.9dB PSNR
/// (`STRICT_PSNR_DIFF_THRESH`). That trial is unported here.
///
/// This is NOT a byte-exactness gate -- it targets the ONE bit the trial can
/// change (`allow_screen_content_tools`) with content chosen to be
/// detector-negative (few high-contrast 16x16-block transitions, so the
/// block-counting heuristic in `screen_detect.rs` stays under its own
/// threshold) but genuinely palette-friendly (few EXACT colours, so the
/// trial's PSNR-at-heavy-quantization comparison has a real gap to find):
/// sparse single-pixel dots, thin single-pixel-wide lines, and a posterized
/// (4-level) gradient. Run across every speed the trial's own C guard does
/// not statically exclude (`cpi->sf.rt_sf.use_nonrd_pick_mode` -- NOT the
/// same gate as the base detector's own `detection_disabled` cutoff, so this
/// covers a speed or two the detector-disabled cutoff does not).
///
/// This is deliberately NOT a hard-fail assertion: a real divergence here is
/// exactly the kind of finding the DoD (#15) asks be turned into a NAMED
/// refusal or a registered pin, not discovered as a red gate on an unrelated
/// commit. Divergences are printed and returned so the caller (a human, or a
/// future landing) can act on them; a lack of divergence is itself the
/// evidence this file's own doc comment claims ("the byte gate holds this
/// accountable per cell") without ever having tried adversarial content.
#[test]
fn probe_sc_tools_trial_gap_on_detector_negative_content() {
    c::ref_init();

    #[derive(Clone, Copy)]
    enum Probe {
        /// One in every 64 pixels flips colour; everything else is flat.
        SparseDots,
        /// Every 32nd COLUMN is a different colour; everything else flat.
        ThinLines,
        /// A diagonal ramp quantized to 4 exact levels (posterized).
        Posterized4,
    }
    fn sample(p: Probe, r: usize, col: usize) -> i32 {
        match p {
            Probe::SparseDots => {
                if (r * 131 + col * 197) % 64 == 0 {
                    40
                } else {
                    200
                }
            }
            Probe::ThinLines => {
                if col % 32 == 0 {
                    60
                } else {
                    180
                }
            }
            Probe::Posterized4 => {
                let level = ((r + col) / 24) % 4;
                32 + (level as i32) * 64
            }
        }
    }

    let w = 128usize;
    let h = 128usize;
    let mut findings = Vec::new();
    let mut cells_tried = 0usize;
    for (nm, p) in [
        ("sparse_dots", Probe::SparseDots),
        ("thin_lines", Probe::ThinLines),
        ("posterized4", Probe::Posterized4),
    ] {
        let mut y = vec![0u16; w * h];
        for r in 0..h {
            for col in 0..w {
                y[r * w + col] = sample(p, r, col).clamp(0, 255) as u16;
            }
        }
        let cw = (w + 1) >> 1;
        let ch = (h + 1) >> 1;
        let u = vec![128u16; cw * ch];
        let v = vec![128u16; cw * ch];
        for speed in 0..=8 {
            cells_tried += 1;
            let cell = Cell::new(format!("SCM_{nm}"), w, h, 8, false, 1, 1, 32, Content::Flat)
                .at_speed(speed);
            let ours = port_stream(&cell, &y, &u, &v);
            let theirs = c_stream(&cell, &y, &u, &v);
            let (ours_hdr, _, _) = split_frame_obu(&ours);
            let (theirs_hdr, _, _) = split_frame_obu(&theirs);
            if ours_hdr.allow_screen_content_tools != theirs_hdr.allow_screen_content_tools {
                findings.push(format!(
                    "{nm} s{speed}: port allow_screen_content_tools={} vs C={} -- the trial \
                     flipped it (or the port's shortcut disagrees for some other reason)",
                    ours_hdr.allow_screen_content_tools, theirs_hdr.allow_screen_content_tools
                ));
            }
            eprintln!(
                "{nm} s{speed}: port ascs={} C ascs={} ({} bytes / {} bytes)",
                ours_hdr.allow_screen_content_tools,
                theirs_hdr.allow_screen_content_tools,
                ours.len(),
                theirs.len()
            );
        }
    }
    eprintln!(
        "probe_sc_tools_trial_gap_on_detector_negative_content: {cells_tried} cells, {} \
         allow_screen_content_tools divergences",
        findings.len()
    );
    assert!(
        findings.is_empty(),
        "the SCM trial gap is REAL on this content -- {} of {cells_tried} cells disagree with \
         C on allow_screen_content_tools:\n  {}\nThis is the un-ported `av1_determine_sc_tools_\
         with_encoding` arm (encoder_utils.c:1214, PARITY.md C3) actually mattering within \
         encode_key_frame's envelope. Per issue #15's DoD, turn this into either a targeted \
         KeyFrameError refusal for the discovered condition, or a registered pin with the \
         measured attribution -- do not silently leave it failing here.",
        findings.len(),
        findings.join("\n  ")
    );
}

/// **A second, size-driven attempt at the same probe.** The first probe's
/// content all crossed the base detector's OWN threshold (every cell came
/// back `ascs=true` on both sides -- the trial never runs when the base
/// decision is already on, per the module doc, so that content never reached
/// the un-ported arm either). Reading `screen_detect.rs`'s formula --
/// `(palette - photo/16) * 256 * 10 > area` -- explains why: the multiplier
/// (2560) is large relative to a typical frame's block count, so crossing the
/// threshold takes only a HANDFUL of net "simple" blocks out of the total,
/// and the historical "14 tiny" poison cells (PARITY.md C3, the geometry axis
/// K/K2 above reproduces the SIZES but not this) are all SMALL frames -- where
/// `area` itself is small, so the crossover needs even FEWER blocks. This
/// probe targets that directly: a small, mostly-noisy frame with ONE
/// checker-textured patch (few EXACT colours WITH internal structure -- a
/// flat patch was tried first and NEVER crossed the detector at any size up
/// to the whole 64x64 frame, consistent with `screen_detect.rs`'s block
/// classification wanting variance alongside a low colour count, not a
/// perfectly flat DC region: ordinary photo content is full of flat DC
/// blocks and must not read as screen-like) whose size is swept from small
/// to large, searching for the crossover from BOTH sides -- does the base
/// detector's own threshold and C's actual (trial-augmented) decision move
/// together, or does C's trial flip the bit at a SMALLER patch than the base
/// detector alone would?
#[test]
fn probe_sc_tools_trial_gap_flat_patch_on_small_noisy_frame() {
    c::ref_init();
    let w = 64usize;
    let h = 64usize;
    let mut findings = Vec::new();
    let mut cells_tried = 0usize;
    // Patch sizes from a single 8x8 corner up to half the frame, so the
    // sweep brackets the detector's own crossover from both sides.
    for patch in [0usize, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64] {
        let mut y = vec![0u16; w * h];
        for r in 0..h {
            for col in 0..w {
                let v = if r < patch && col < patch {
                    content_sample(Content::Checker, r, col) // few colours, WITH structure
                } else {
                    content_sample(Content::Noise, r, col)
                };
                y[r * w + col] = v.clamp(0, 255) as u16;
            }
        }
        let cw = (w + 1) >> 1;
        let ch = (h + 1) >> 1;
        let u = vec![128u16; cw * ch];
        let v = vec![128u16; cw * ch];
        for speed in [0i32, 3, 6] {
            cells_tried += 1;
            let cell = Cell::new(
                format!("SCM_patch{patch}"),
                w,
                h,
                8,
                false,
                1,
                1,
                32,
                Content::Flat,
            )
            .at_speed(speed);
            let ours = port_stream(&cell, &y, &u, &v);
            let theirs = c_stream(&cell, &y, &u, &v);
            let (ours_hdr, _, _) = split_frame_obu(&ours);
            let (theirs_hdr, _, _) = split_frame_obu(&theirs);
            if ours_hdr.allow_screen_content_tools != theirs_hdr.allow_screen_content_tools {
                findings.push(format!(
                    "patch{patch} s{speed}: port ascs={} vs C ascs={}",
                    ours_hdr.allow_screen_content_tools, theirs_hdr.allow_screen_content_tools
                ));
            }
            eprintln!(
                "patch{patch} s{speed}: port ascs={} C ascs={} ({} / {} bytes)",
                ours_hdr.allow_screen_content_tools,
                theirs_hdr.allow_screen_content_tools,
                ours.len(),
                theirs.len()
            );
        }
    }
    eprintln!(
        "probe_sc_tools_trial_gap_flat_patch_on_small_noisy_frame: {cells_tried} cells, {} \
         allow_screen_content_tools divergences",
        findings.len()
    );
    assert!(
        findings.is_empty(),
        "the SCM trial gap is REAL on a size-swept flat-patch-on-noise probe -- {} of \
         {cells_tried} cells disagree with C on allow_screen_content_tools:\n  {}\nSame \
         un-ported `av1_determine_sc_tools_with_encoding` arm as the sibling probe above; \
         act on it the same way (targeted refusal or a registered, measured pin) rather than \
         leaving this failing.",
        findings.len(),
        findings.join("\n  ")
    );
}

/// **The coded-lossless arm actually reconstructs the source, bit for bit.**
///
/// Byte-parity against real aomenc is the primary gate, but it cannot speak for
/// the cells inside the pre-existing `HBD_OPEN` band, where the port's
/// bitstream legitimately differs from C's. `--cq-level 0` has a property that
/// is independent of C: `base_qindex == 0` with no deltas makes the frame
/// `coded_lossless` (`is_coded_lossless`, `encodeframe.c:2275`), so a
/// conforming decode of a conforming stream must return the ENCODER'S INPUT
/// exactly, on every plane. This test asserts that with the real libaom
/// decoder and with this repo's own decoder, over the whole grid the byte gate
/// can only half-cover — HBD_OPEN cells included.
///
/// It is also the direct regression lock on zenavif#45: `encode_key_frame` at
/// cq 0 must return a `Result` at every coordinate, never unwind. The reported
/// panic was `tx_size_to_depth`'s `depth <= MAX_TX_DEPTH` `debug_assert`
/// reached through `txb_split_count` -> `count_leaf`, and it fired only where
/// a lossless leaf is `BLOCK_32X32` or bigger (depth 3) — i.e. on content and
/// speeds the single cq-0 cell in axis A never produced. Note the profile:
/// `[profile.test-fast]` inherits `release` but keeps `debug-assertions = true`
/// (workspace `Cargo.toml`), which is why the assert is live here.
#[test]
fn coded_lossless_reconstructs_the_source_exactly() {
    use Content::*;
    c::ref_init();
    let mut cells: Vec<Cell> = Vec::new();
    for (nm, mono, sx, sy) in [
        ("mono", true, 1usize, 1usize),
        ("420", false, 1, 1),
        ("422", false, 1, 0),
        ("444", false, 0, 0),
    ] {
        for bd in [8u8, 10, 12] {
            for (cnm, content) in [
                ("flat", Flat),
                ("grad", Gradient),
                ("tex", Texture),
                ("noise", Noise),
                ("chk", Checker),
            ] {
                for speed in [0i32, 3, 6, 9] {
                    cells.push(
                        Cell::new(
                            format!("LL_{nm}_bd{bd}_{cnm}_64x64_cq0"),
                            64,
                            64,
                            bd,
                            mono,
                            sx,
                            sy,
                            0,
                            content,
                        )
                        .at_speed(speed),
                    );
                }
            }
        }
    }
    // Partial superblocks and the tiny end, where the block walk differs.
    for (w, h) in [(1usize, 1usize), (100, 60), (130, 70), (258, 258)] {
        for speed in [0i32, 6] {
            cells.push(
                Cell::new(
                    format!("LL_{w}x{h}_420_bd8_tex_cq0"),
                    w,
                    h,
                    8,
                    false,
                    1,
                    1,
                    0,
                    Texture,
                )
                .at_speed(speed),
            );
        }
    }
    assert!(
        cells.len() >= 240,
        "the lossless grid shrank to {} cells — it must stay dense enough to \
         reach a BLOCK_32X32-or-larger lossless leaf at every depth",
        cells.len()
    );

    let mut checked = 0usize;
    for cell in &cells {
        let (y, u, v) = cell_planes(cell);
        // zenavif#45's own shape: a panic out of a library entry point is a
        // defect regardless of the bytes it would have produced.
        let encoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_key_frame(
                KeyFramePlanes {
                    y: &y,
                    u: &u,
                    v: &v,
                },
                &cell_cfg(cell),
            )
        }))
        .unwrap_or_else(|_| {
            panic!(
                "{}: encode_key_frame PANICKED at --cq-level 0. It must return a \
                 Result, never unwind (zenavif#45)",
                cell.label
            )
        })
        .unwrap_or_else(|e| panic!("{}: encode_key_frame refused: {e}", cell.label));

        let dec = c::ref_decode_av1_kf(&encoded, cell.w, cell.h);
        assert_eq!(
            (&dec.y, &dec.u, &dec.v),
            (&y, &u, &v),
            "{}: --cq-level 0 is coded-lossless, so real-C-decode(port stream) must \
             return the encoder's own input on every plane",
            cell.label
        );
        let p_dec = aom_decode::frame::decode_frame_obus(&encoded)
            .unwrap_or_else(|e| panic!("{}: port decode failed: {e}", cell.label));
        assert_eq!(
            (&p_dec.y, &p_dec.u, &p_dec.v),
            (&y, &u, &v),
            "{}: port-decode(port stream) is not lossless",
            cell.label
        );
        checked += 1;
    }
    eprintln!(
        "coded_lossless_reconstructs_the_source_exactly: {checked}/{} cells decode to the \
         source exactly, on both decoders",
        cells.len()
    );
}
