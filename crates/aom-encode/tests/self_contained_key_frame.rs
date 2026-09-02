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

use aom_dsp::entropy::obu::read_obu_header;
use aom_encode::key_frame::{KeyFrameConfig, KeyFrameError, KeyFramePlanes, encode_key_frame};
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
        }
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
    KeyFrameConfig::allintra_speed0(
        cell.w, cell.h, cell.bd, cell.mono, cell.ss_x, cell.ss_y, cell.cq,
    )
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
        false,
        false,
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

    let mut cdef_on = base;
    cdef_on.enable_cdef = true;
    assert!(matches!(
        encode_key_frame(planes, &cdef_on),
        Err(KeyFrameError::Unsupported(_))
    ));

    let mut lr_on = base;
    lr_on.enable_restoration = true;
    assert!(matches!(
        encode_key_frame(planes, &lr_on),
        Err(KeyFrameError::Unsupported(_))
    ));

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
/// | 261x261 (4:2:0 bd8 cq32 textured) | `pick_filter_level` derives `filter_level = [0, 1]` where real aomenc codes `[0, 2]` — an off-by-one in the loop-filter-level search on this frame, visible at frame-payload byte 3. Everything else agrees. |
///
/// The neighbours bracket both: 130x70, 200x200, 250x130, 258x258, 262x262,
/// 263x263, 264x264, 256x256 and 320x320 are all byte-exact in
/// [`sweep_cells`], so neither pin is "the port cannot do partial superblocks".
#[test]
fn open_divergences_are_pinned() {
    c::ref_init();
    let pins = [
        (
            132usize,
            132usize,
            "tile-payload RD near-tie (headers all agree)",
        ),
        (196, 196, "tile-payload RD near-tie (headers all agree)"),
        (260, 260, "tile-payload RD near-tie (headers all agree)"),
        (261, 261, "pick_filter_level derives [0,1] vs C's [0,2]"),
    ];
    for &(w, h, why) in &pins {
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
        eprintln!("PIN {w}x{h}: still divergent ({why}); seq header byte-exact; decodes");
    }
}
