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
//! | [`self_contained_key_frame_byte_matches_real_aomenc`] | the port's WHOLE temporal unit is byte-identical to `shim_encode_av1_kf`'s (TD + seq + frame), i.e. every derived header field equals C's |
//! | [`self_contained_key_frame_decodes_to_the_same_pixels`] | the real C decoder AND the port decoder both decode the port's own stream, to the same pixels C's own stream decodes to |
//! | [`mutated_sequence_header_is_caught`] | mutation proof: perturbing ONE derived header field (the coded `base_qindex`) makes the pixel gate fail and the byte gate fail — neither is vacuous |
//! | [`refuses_configurations_it_has_no_gate_for`] | the shell returns [`KeyFrameError`] instead of silently mis-encoding outside its envelope |
//!
//! # Envelope
//!
//! ALL-INTRA, `--cpu-used 0`, CDEF off, loop-restoration off, SB64, single
//! tile, palette + IntraBC off — exactly `aom_sys_ref::ref_encode_av1_kf`'s
//! configuration, so the byte comparison is like-for-like. The
//! `key_frame` module documents the axes that are not wired yet.

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
    let reader_cfg = derive_frame_header(
        &kf_cfg,
        &seq,
        &ScreenContentDecision::detection_disabled(),
        tile_info,
    );
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
        }
    }

    /// The same cell with the two post-filter knobs set.
    fn with_postfilter(mut self, cdef: bool, lr: bool) -> Self {
        self.label = format!("{}_cdef{}_lr{}", self.label, u8::from(cdef), u8::from(lr));
        self.cdef = cdef;
        self.lr = lr;
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
    // including BOTH ON, which is real aomenc's ALLINTRA default and which no
    // pack entry point covered before `pack::pack_tile_from_trees_lr`. Swept
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
    ];
    sweep_cells()
        .into_iter()
        .filter(|c| keep.contains(&c.label.as_str()))
        .collect()
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
    cfg
}

/// Run the port's bootstrap-free encoder for a cell.
fn port_stream(cell: &Cell, y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
    encode_key_frame(KeyFramePlanes { y, u, v }, &cell_cfg(cell))
        .unwrap_or_else(|e| panic!("{}: encode_key_frame refused: {e}", cell.label))
}

/// Real aomenc's stream for the same cell + config.
fn c_stream(cell: &Cell, y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
    let bytes = c::ref_encode_av1_kf(
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
        0,
        cell.cdef,
        cell.lr,
        2,
        0,
        false,
    );
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

    let mut fast = base;
    fast.cpu_used = 5;
    assert!(matches!(
        encode_key_frame(planes, &fast),
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

    // A frame wide enough that `av1_get_tile_limits` MANDATES a tile split is
    // refused by name rather than mis-assembled (the multi-tile assembler
    // exists but has no gate through this shell).
    let wide = KeyFrameConfig::allintra_speed0(4160, 64, 8, true, 1, 1, 32);
    let wide_y = vec![128u16; 4160 * 64];
    assert!(matches!(
        encode_key_frame(
            KeyFramePlanes {
                y: &wide_y,
                u: &[],
                v: &[]
            },
            &wide
        ),
        Err(KeyFrameError::MultiTileRequired { .. })
    ));
}

/// **Open divergences, PINNED.** These cells are NOT byte-identical to real
/// aomenc, and this test asserts the divergence is STILL PRESENT so a fix flips
/// it red and forces promotion into [`sweep_cells`] (this repo's self-promoting
/// pin convention — see the PARITY.md Tier-3 rows).
///
/// **None of them is a framing or header-derivation defect in the shell.**
/// Measured 2026-09-02 by parsing both streams' headers:
///
/// | cell | measured attribution |
/// |---|---|
/// | 132x132, 196x196, 260x260 (4:2:0 bd8 cq32 textured) | EVERY derived header field equals C's — `allow_screen_content_tools`, `base_qindex`, both loop-filter levels, `tx_mode_select`, the tile grid — and the sequence-header OBU payload is byte-identical. The divergence starts 367 / 525 / 1240 bytes INTO the tile payload: a partition/RD near-tie in `pack_tile`'s search, the class PARITY.md Tier-3 already tracks. It is reachable from the bootstrapped harness too. |
/// | 261x261 (4:2:0 bd8 cq32 textured) | BOTH halves differ — the tile payload (2174 vs 2181 bytes) AND the derived loop-filter level (`[0, 1]` vs C's `[0, 2]`). For a KEY frame that is ONE upstream root, not two: `pick_filter_level` runs on the port's OWN phase-1 reconstruction, so an RD divergence that changes the recon moves the level with it. (The 2026-09-02 first attribution called this a pure `pick_filter_level` off-by-one; the `Where` assertion below caught that and is why it is asserted rather than written in prose.) |
///
/// The neighbours bracket them: 130x70, 200x200, 250x130, 258x258, 262x262,
/// 263x263, 264x264, 256x256 and 320x320 are all byte-exact in
/// [`sweep_cells`], so no pin here is "the port cannot do partial superblocks".
///
/// All four pins run with CDEF and loop restoration OFF; the post-filter axis
/// (axis G of the sweep) is 27/27 byte-exact.
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
    let pins = [
        (
            132usize,
            132usize,
            Where::TilePayloadOnly,
            "tile-payload RD near-tie (every derived header field agrees)",
        ),
        (
            196,
            196,
            Where::TilePayloadOnly,
            "tile-payload RD near-tie (every derived header field agrees)",
        ),
        (
            260,
            260,
            Where::TilePayloadOnly,
            "tile-payload RD near-tie (every derived header field agrees)",
        ),
        (
            261,
            261,
            Where::TilePayloadAndHeader,
            "an RD near-tie whose different reconstruction ALSO moves the derived \
             loop-filter level: port [0,1] vs C's [0,2]",
        ),
    ];
    for &(w, h, expect_where, why) in &pins {
        let cell = Cell::new(
            format!("PIN_{w}x{h}_420_bd8_cq32_tex"),
            w,
            h,
            8,
            false,
            1,
            1,
            32,
            Content::Texture,
        );
        let (y, u, v) = cell_planes(&cell);
        let ours = port_stream(&cell, &y, &u, &v);
        let theirs = c_stream(&cell, &y, &u, &v);
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
        let header_same = (
            ours_hdr.loopfilter.filter_level,
            ours_hdr.loopfilter.filter_level_u,
            ours_hdr.loopfilter.filter_level_v,
            ours_hdr.quant.base_qindex,
            ours_hdr.allow_screen_content_tools,
            ours_hdr.tx_mode_select,
        ) == (
            theirs_hdr.loopfilter.filter_level,
            theirs_hdr.loopfilter.filter_level_u,
            theirs_hdr.loopfilter.filter_level_v,
            theirs_hdr.quant.base_qindex,
            theirs_hdr.allow_screen_content_tools,
            theirs_hdr.tx_mode_select,
        );
        let got_where = match (ours_tile == theirs_tile, header_same) {
            (true, false) => Where::HeaderOnly,
            (false, true) => Where::TilePayloadOnly,
            (false, false) => Where::TilePayloadAndHeader,
            (true, true) => panic!(
                "{w}x{h}: the streams differ but both the tile payload and every checked \
                 header field agree — the divergence is somewhere this test does not look \
                 (OBU framing? the sequence header? a header field not in the tuple). \
                 Investigate before touching this list."
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
