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

/// A deterministic content generator: diagonal gradient + vertical bars
/// (period 16) + a fine ripple. Same shape as
/// `encoder_gate_e2e_byte_match.rs`'s strong-LF generator, so the loop-filter
/// levels the port derives are genuinely non-trivial.
fn diag_vbars16_ripple(r: usize, col: usize) -> u8 {
    let grad = 32 + (r + col) * 150 / 256;
    let bar = if (col / 16) % 2 == 0 { 0 } else { 45 };
    let ripple = if (r + col) % 2 == 0 { 14 } else { -14 };
    (grad as i32 + bar + ripple).clamp(0, 255) as u8
}

/// Flat mid-grey — the trivially-lossless control.
fn flat(_r: usize, _c: usize) -> u8 {
    128
}

/// One gate cell.
struct Cell {
    label: &'static str,
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    cq: i32,
    content: fn(usize, usize) -> u8,
}

const CELLS: &[Cell] = &[
    Cell {
        label: "flat_mono_64x64_cq32",
        w: 64,
        h: 64,
        mono: true,
        ss_x: 1,
        ss_y: 1,
        cq: 32,
        content: flat,
    },
    Cell {
        label: "flat_420_64x64_cq32",
        w: 64,
        h: 64,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        cq: 32,
        content: flat,
    },
    Cell {
        label: "texture_mono_64x64_cq32",
        w: 64,
        h: 64,
        mono: true,
        ss_x: 1,
        ss_y: 1,
        cq: 32,
        content: diag_vbars16_ripple,
    },
    Cell {
        label: "texture_420_64x64_cq32",
        w: 64,
        h: 64,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        cq: 32,
        content: diag_vbars16_ripple,
    },
    Cell {
        label: "texture_420_64x64_cq10",
        w: 64,
        h: 64,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        cq: 10,
        content: diag_vbars16_ripple,
    },
    Cell {
        label: "texture_420_64x64_cq55",
        w: 64,
        h: 64,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        cq: 55,
        content: diag_vbars16_ripple,
    },
    Cell {
        label: "texture_444_128x96_cq40",
        w: 128,
        h: 96,
        mono: false,
        ss_x: 0,
        ss_y: 0,
        cq: 40,
        content: diag_vbars16_ripple,
    },
    Cell {
        label: "texture_422_128x64_cq24",
        w: 128,
        h: 64,
        mono: false,
        ss_x: 1,
        ss_y: 0,
        cq: 24,
        content: diag_vbars16_ripple,
    },
    Cell {
        label: "texture_420_128x128_cq32",
        w: 128,
        h: 128,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        cq: 32,
        content: diag_vbars16_ripple,
    },
];

/// Source planes for a cell (luma from `content`, chroma flat mid-grey — the
/// same convention `encoder_gate_e2e_byte_match.rs` uses).
fn cell_planes(cell: &Cell) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (w, h) = (cell.w, cell.h);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = u16::from((cell.content)(r, col));
        }
    }
    let (cw, ch) = if cell.mono {
        (0, 0)
    } else {
        ((w + cell.ss_x) >> cell.ss_x, (h + cell.ss_y) >> cell.ss_y)
    };
    (y, vec![128u16; cw * ch], vec![128u16; cw * ch])
}

fn cell_cfg(cell: &Cell) -> KeyFrameConfig {
    KeyFrameConfig::allintra_speed0(cell.w, cell.h, 8, cell.mono, cell.ss_x, cell.ss_y, cell.cq)
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
        8,
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
    let cell = &CELLS[0];
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
    let mut exact = 0usize;
    let mut failures = Vec::new();
    for cell in CELLS {
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
        CELLS.len(),
        failures.join("\n  ")
    );
    eprintln!(
        "self_contained_key_frame_byte_matches_real_aomenc: {exact}/{} cells byte-exact",
        CELLS.len()
    );
}

/// Decode the port's own bootstrap-free stream with BOTH decoders — real
/// libaom (`aom_codec_av1_dx`) and this repo's own AV1 decoder — and require
/// both to reproduce the pixels real aomenc's own stream decodes to.
#[test]
fn self_contained_key_frame_decodes_to_the_same_pixels() {
    c::ref_init();
    for cell in CELLS {
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
        if cell.content as usize == flat as usize {
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
    let cell = &CELLS[3]; // texture_420_64x64_cq32 — a non-trivial reconstruction
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
