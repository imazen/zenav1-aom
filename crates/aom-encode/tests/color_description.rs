//! The sequence header's CICP colour description and pixel range are
//! CONFIGURATION, and this is the gate that says so.
//!
//! # Why this file exists
//!
//! `derive_sequence_header` used to hardcode `(2, 2, 2)` "unspecified" plus
//! `color_range = 0` (`AOM_CR_STUDIO_RANGE`) with no way for a caller to say
//! otherwise. That made two things UNREACHABLE rather than unsupported, and
//! both matter to a still-image backend:
//!
//! * **full-range stills** — zenavif's aom backend refuses
//!   `EncodePixelRange::Full` by name, citing this hardcode, so a full-range
//!   source is converted for limited range and loses ~13 % of the code range
//!   at 8 bits;
//! * **AVIF alpha** — an alpha plane is an auxiliary *monochrome, full-range*
//!   item, so the range pin blocked wiring the Cs400 item at all.
//!
//! What this gate asserts, in order of what would actually break:
//!
//! 1. the DEFAULT is byte-identical to the old hardcode (so the 427-cell
//!    byte-identity gate cannot move);
//! 2. a non-default description REACHES THE BITSTREAM and survives a
//!    round-trip through the port's own `read_color_config`;
//! 3. the streams still DECODE — a header field that decoders reject is worse
//!    than one that is wrong;
//! 4. the non-conformant combinations AV1 forbids are REFUSED by name rather
//!    than silently mis-signalled.

use aom_dsp::entropy::header::read_color_config;
use aom_dsp::entropy::leb128::uleb_decode;
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_encode::key_frame::{
    ColorDescription, KeyFrameConfig, KeyFramePlanes, encode_key_frame,
};

/// `OBU_SEQUENCE_HEADER`.
const OBU_SEQ: u32 = 1;
/// `OBU_TEMPORAL_DELIMITER`.
const OBU_TD: u32 = 2;

fn planes(cfg: &KeyFrameConfig) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << cfg.bit_depth) - 1;
    let y: Vec<u16> = (0..cfg.width * cfg.height)
        .map(|i| {
            let (r, c) = (i / cfg.width, i % cfg.width);
            (((r + c) * 37 % 200) as u32 * max / 255) as u16
        })
        .collect();
    let (cw, ch) = ((cfg.width + cfg.ss_x) >> cfg.ss_x, (cfg.height + cfg.ss_y) >> cfg.ss_y);
    let n = if cfg.monochrome { 0 } else { cw * ch };
    let u: Vec<u16> = (0..n).map(|i| ((i * 11 % 160) as u32 * max / 255) as u16).collect();
    let v: Vec<u16> = (0..n).map(|i| ((i * 17 % 140) as u32 * max / 255) as u16).collect();
    (y, u, v)
}

fn encode(cfg: &KeyFrameConfig) -> Vec<u8> {
    let (y, u, v) = planes(cfg);
    encode_key_frame(KeyFramePlanes { y: &y, u: &u, v: &v }, cfg)
        .expect("the cell must encode")
}

/// Walk the temporal unit to the sequence-header OBU and parse its
/// `color_config` back with the port's own reader — the inverse of the writer
/// under test, so a wrong field cannot round-trip by accident.
fn seq_color(stream: &[u8]) -> aom_dsp::entropy::header::ColorConfigParams {
    let mut off = 0usize;
    loop {
        let hdr = read_obu_header(&stream[off..]).expect("valid OBU header");
        let (size, size_len) =
            uleb_decode(&stream[off + hdr.header_len..]).expect("valid leb128 size");
        let size = size as usize;
        let payload_at = off + hdr.header_len + size_len;
        if hdr.obu_type == OBU_SEQ {
            let payload = &stream[payload_at..payload_at + size];
            let mut rb = ReadBitBuffer::new(payload);
            // seq_profile f(3), still_picture f(1), reduced_still_picture_header f(1)
            let profile = rb.read_literal(3) as i32;
            let _still = rb.read_bit();
            let reduced = rb.read_bit();
            assert_eq!(reduced, 1, "encode_key_frame emits a reduced still-picture header");
            // seq_level_idx[0] f(5), then frame width/height bits + max dims,
            // then the feature flags -- all fixed-width in the reduced header.
            let _level = rb.read_literal(5);
            let wbits = rb.read_literal(4) as u32 + 1;
            let hbits = rb.read_literal(4) as u32 + 1;
            let _w = rb.read_literal(wbits);
            let _h = rb.read_literal(hbits);
            // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter,
            // enable_superres, enable_cdef, enable_restoration
            for _ in 0..6 {
                let _ = rb.read_bit();
            }
            return read_color_config(&mut rb, profile);
        }
        assert_eq!(hdr.obu_type, OBU_TD, "only a TD may precede the sequence header");
        off = payload_at + size;
        assert!(off < stream.len(), "no sequence header in the stream");
    }
}

fn cfg_420(cq: i32) -> KeyFrameConfig {
    KeyFrameConfig::allintra_speed0(64, 64, 8, false, 1, 1, cq)
}

/// (1) The default must reproduce the historical hardcode BYTE for BYTE — this
/// is what keeps the 427-cell byte-identity gate from moving.
#[test]
fn the_default_description_is_the_historical_hardcode() {
    let d = ColorDescription::default();
    assert_eq!(
        (d.color_primaries, d.transfer_characteristics, d.matrix_coefficients, d.full_range),
        (2, 2, 2, false),
        "the default must stay AOM_CICP_*_UNSPECIFIED + AOM_CR_STUDIO_RANGE"
    );
    let c = seq_color(&encode(&cfg_420(32)));
    assert_eq!((c.color_primaries, c.transfer_characteristics, c.matrix_coefficients), (2, 2, 2));
    assert!(!c.color_range, "the default codes studio range");
}

/// (2) + (3) A non-default description reaches the bitstream, round-trips, and
/// the stream still decodes.
#[test]
fn a_full_range_description_reaches_the_bitstream_and_round_trips() {
    // Full range with an explicit BT.709 / sRGB-transfer / BT.601-matrix
    // description -- a description a still-image caller actually produces.
    let mut cfg = cfg_420(32);
    cfg.color = ColorDescription {
        color_primaries: 1,           // CP_BT_709
        transfer_characteristics: 13, // TC_SRGB
        matrix_coefficients: 6,       // MC_BT_601 (NOT identity: 4:2:0 is legal here)
        full_range: true,
    };
    let stream = encode(&cfg);
    let c = seq_color(&stream);
    assert_eq!(
        (c.color_primaries, c.transfer_characteristics, c.matrix_coefficients),
        (1, 13, 6),
        "the CICP triple must survive the writer/reader round trip"
    );
    assert!(c.color_range, "full_range must reach the coded color_range bit");

    // A field the decoder rejects is worse than one that is merely wrong.
    let dec = aom_decode::frame::decode_frame_obus(&stream)
        .expect("the port decoder must accept a full-range stream");
    assert_eq!(dec.width, cfg.width);
    assert_eq!(dec.height, cfg.height);

    // ... and the range bit is the ONLY difference from the default encode:
    // same samples, same everything else.
    let mut studio = cfg;
    studio.color.full_range = false;
    let s2 = encode(&studio);
    assert_ne!(stream, s2, "the range bit must be observable in the bytes");
    assert_eq!(stream.len(), s2.len(), "it is one bit inside a fixed-width field");
}

/// Monochrome carries a range bit too -- this is the path AVIF alpha needs.
#[test]
fn monochrome_full_range_is_codable_and_that_is_what_alpha_needs() {
    let mut cfg = KeyFrameConfig::allintra_speed0(64, 64, 8, true, 1, 1, 32);
    cfg.color.full_range = true;
    let stream = encode(&cfg);
    let c = seq_color(&stream);
    assert!(c.monochrome, "the cell is monochrome");
    assert!(c.color_range, "an alpha plane is a FULL-RANGE monochrome item");
    aom_decode::frame::decode_frame_obus(&stream).expect("must decode");
}

/// (4) The combinations AV1 forbids are refused BY NAME, not mis-signalled.
#[test]
fn non_conformant_descriptions_are_refused_by_name() {
    // MC_IDENTITY with subsampling: AV1 5.5.2 makes 4:4:4 a conformance
    // requirement, and libaom asserts it in `write_color_config`.
    let mut id420 = cfg_420(32);
    id420.color.matrix_coefficients = 0;
    let e = id420.validate_configuration().expect_err("MC_IDENTITY at 4:2:0 must be refused");
    assert!(
        format!("{e}").contains("MC_IDENTITY"),
        "the refusal must name the field, got: {e}"
    );

    // ... and is ACCEPTED at 4:4:4, so the refusal is about the pairing and
    // not a blanket ban (a refusal that always fires gates nothing).
    let mut id444 = KeyFrameConfig::allintra_speed0(64, 64, 8, false, 0, 0, 32);
    id444.color.matrix_coefficients = 0;
    id444.validate_configuration().expect("MC_IDENTITY at 4:4:4 is conformant");

    // The sRGB triple codes NO range bit -- the spec fixes it full -- so a
    // studio-range request there would be silently mis-signalled.
    let mut srgb_studio = KeyFrameConfig::allintra_speed0(64, 64, 8, false, 0, 0, 32);
    srgb_studio.color = ColorDescription {
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
        full_range: false,
    };
    let e = srgb_studio
        .validate_configuration()
        .expect_err("sRGB + studio range must be refused, not silently coded as full");
    assert!(format!("{e}").contains("sRGB"), "the refusal must name it, got: {e}");

    // CICP fields are f(8).
    let mut big = cfg_420(32);
    big.color.color_primaries = 256;
    big.validate_configuration().expect_err("a CICP code point past 255 must be refused");

    // Every refusal above is `unsupported`, not a transient failure.
    assert_eq!(
        id420.validate_configuration().unwrap_err().category(),
        "unsupported"
    );
}
