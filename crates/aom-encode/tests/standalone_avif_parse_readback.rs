//! **`zenavif-parse` read-back of the standalone stream** — the second half of
//! issue #15's definition-of-done that `self_contained_key_frame.rs` does not
//! cover: "validated by its own decode plus a `zenavif-parse` read-back."
//! `self_contained_key_frame.rs` proves the OBU stream itself is
//! decode-verifiable (both the real C decoder and this repo's own
//! `aom-decode`); this file proves the stream survives being muxed into a
//! real AVIF container and read back through an INDEPENDENTLY-maintained
//! parser (`zenavif-parse`, a fork of Mozilla's MP4 parser, structurally
//! unrelated to anything in this repo) rather than only this repo's own
//! `aom-decode` or `avif_parity.rs`'s hand-rolled `extract_mdat_payload` box
//! walker, which shares a lineage with the muxer it checks.
//!
//! Deliberately separate from `avif_parity.rs`: that file mixes real
//! aomenc's sequence header with the port's frame OBU (`prod.av1_stream =
//! seq_hdr_raw ++ our_frame_obu`, `seq_hdr_raw` extracted from the C
//! bootstrap) — it is Gate 4 of the C-parity family, unaffected by this
//! landing. This file feeds `Aviffy` the port's OWN self-contained
//! `encode_key_frame` output (its sequence header included), with no C
//! bytes anywhere in the chain from planes to the read-back assertions.
//!
//! Two independent checks, both against `AvifParser`:
//! 1. **Byte round-trip**: `parser.primary_data()` (the demuxed `mdat`
//!    payload, from a genuinely different ISOBMFF box walker than this
//!    repo's own) equals the exact `seq_header ++ frame_obu` bytes handed to
//!    the muxer.
//! 2. **Metadata cross-check**: `parser.primary_metadata()` re-parses the
//!    EXTRACTED bytes with `zenavif-parse`'s own independent AV1 OBU parser
//!    (`AV1Metadata::parse_av1_bitstream`, structurally unrelated to this
//!    repo's `aom_dsp`/`aom_decode`) and must agree with the
//!    [`KeyFrameConfig`] the port was asked to encode: width, height, bit
//!    depth, monochrome, and chroma subsampling — all SEQUENCE-header
//!    fields. `base_q_idx` / `lossless` (FRAME-header fields, read by a
//!    separate code path in `zenavif-parse`) are measured and printed but
//!    deliberately NOT asserted: see the comment at their call site for a
//!    genuine upstream bit-alignment bug this run uncovered, which could not
//!    be filed (`imazen/zenavif-parse` is archived).
//!
//! Both extracted payloads are also decoded via this repo's own
//! `aom_decode::frame::decode_frame_obus`, closing the loop container ->
//! independent-parser -> this port's decoder -> pixels.

use aom_dsp::entropy::leb128::uleb_decode;
use aom_dsp::entropy::obu::read_obu_header;
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_encode::rc::base_qindex_from_cq;
use zenavif_parse::{AvifParser, ChromaSubsampling as ParseChroma};
use zenavif_serialize::{Aviffy, ChromaSubsampling as MuxChroma};

const OBU_TEMPORAL_DELIMITER: u32 = 2;

/// Strip the leading `OBU_TEMPORAL_DELIMITER` from the port's self-contained
/// stream, returning `seq_header ++ frame_obu` — the shape `Aviffy::try_to_vec`
/// expects (matching `avif_parity.rs`'s `av1_stream`, which never carries a
/// TD either: AVIF's `mdat` holds the coded OBUs, not a full elementary
/// stream, and the TD is redundant once container framing exists).
fn strip_temporal_delimiter(stream: &[u8]) -> Vec<u8> {
    let hdr = read_obu_header(stream).expect("stream starts with a valid OBU header");
    assert_eq!(
        hdr.obu_type, OBU_TEMPORAL_DELIMITER,
        "encode_key_frame's stream must start with the temporal delimiter"
    );
    let (size, size_len) =
        uleb_decode(&stream[hdr.header_len..]).expect("TD OBU carries a valid leb128 size");
    assert_eq!(size, 0, "the temporal delimiter OBU has no payload");
    stream[hdr.header_len + size_len..].to_vec()
}

/// One cell: config + a small deterministic content generator (a diagonal
/// ramp with a period-16 bar pattern, matching `self_contained_key_frame.rs`'s
/// Texture generator in shape -- non-trivial enough that the frame header's
/// derived fields (loop-filter level, tx_mode) are not degenerate).
fn content(r: usize, col: usize) -> i32 {
    let grad = 32 + (r + col) * 150 / 256;
    let bar = if (col / 16) % 2 == 0 { 0 } else { 45 };
    grad as i32 + bar
}

fn planes(
    w: usize,
    h: usize,
    bd: u8,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let maxv = (1u32 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let v8 = content(r, col).clamp(0, 255) as u32;
            y[r * w + col] = ((v8 * maxv) / 255) as u16;
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
    };
    let mid = (maxv / 2 + 1) as u16;
    (y, vec![mid; cw * ch], vec![mid; cw * ch])
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    label: &str,
    w: usize,
    h: usize,
    bd: u8,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    cq: i32,
    sb128: bool,
) {
    let mut cfg = KeyFrameConfig::allintra_speed0(w, h, bd, mono, ss_x, ss_y, cq);
    cfg.sb_size_128 = sb128;
    let (y, u, v) = planes(w, h, bd, mono, ss_x, ss_y);
    let stream = encode_key_frame(
        KeyFramePlanes {
            y: &y,
            u: &u,
            v: &v,
        },
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{label}: encode_key_frame refused: {e}"));
    let payload = strip_temporal_delimiter(&stream);

    // ---- mux into an AVIF still --------------------------------------
    let mux_chroma = if mono {
        MuxChroma::NONE
    } else if ss_x == 1 && ss_y == 1 {
        MuxChroma::YUV420
    } else if ss_x == 1 && ss_y == 0 {
        MuxChroma::YUV422
    } else {
        MuxChroma::NONE // 4:4:4 -- see the module doc: no-subsampling and
        // monochrome share this variant, differentiated by set_monochrome
    };
    let avif = Aviffy::new()
        .set_monochrome(mono)
        .set_chroma_subsampling(mux_chroma)
        .try_to_vec(&payload, None, w as u32, h as u32, bd)
        .unwrap_or_else(|e| panic!("{label}: muxing a valid AVIF failed: {e:?}"));

    // ---- read back through an INDEPENDENT parser ----------------------
    let parser = AvifParser::from_bytes(&avif)
        .unwrap_or_else(|e| panic!("{label}: zenavif-parse could not parse the muxed AVIF: {e}"));
    let extracted = parser
        .primary_data()
        .unwrap_or_else(|e| panic!("{label}: zenavif-parse primary_data() failed: {e}"));
    assert_eq!(
        extracted.as_ref(),
        payload.as_slice(),
        "{label}: zenavif-parse's demuxed primary payload must byte-equal the \
         seq_header++frame_obu handed to the muxer -- the container round-trip \
         is lossless on this repo's own self-contained stream"
    );

    let meta = parser.primary_metadata().unwrap_or_else(|e| {
        panic!(
            "{label}: zenavif-parse primary_metadata() (its own \
             independent AV1 OBU re-parse) failed: {e}"
        )
    });
    assert_eq!(meta.max_frame_width.get() as usize, w, "{label}: width");
    assert_eq!(meta.max_frame_height.get() as usize, h, "{label}: height");
    assert_eq!(meta.bit_depth, bd, "{label}: bit depth");
    assert_eq!(meta.monochrome, mono, "{label}: monochrome");
    let expect_chroma = if mono {
        // A monochrome sequence header carries no chroma_subsampling
        // semantics; zenavif-parse's own parse reads the raw bits (both
        // `false` in a mono seq header here, i.e. NONE), so this axis is not
        // asserted for mono.
        None
    } else if ss_x == 1 && ss_y == 1 {
        Some(ParseChroma::YUV420)
    } else if ss_x == 1 && ss_y == 0 {
        Some(ParseChroma::YUV422)
    } else {
        // 4:4:4 -- `ChromaSubsampling::NONE` ("no chroma subsampling"),
        // structurally identical to the mono representation, differentiated
        // by `meta.monochrome` above.
        Some(ParseChroma::NONE)
    };
    if let Some(want) = expect_chroma {
        assert_eq!(
            meta.chroma_subsampling, want,
            "{label}: zenavif-parse's independently re-derived chroma subsampling"
        );
    }
    assert_eq!(
        meta.still_picture, true,
        "{label}: a single KEY frame with no fwd-kf must read as a still picture"
    );
    // NOT asserted, deliberately: `meta.base_q_idx` (and `meta.lossless`)
    // come from zenavif-parse's OWN convenience frame-header re-parse
    // (`parse_frame_header_quantization`), which has a genuine bit-alignment
    // bug for `reduced_still_picture_header` streams -- exactly what
    // `encode_key_frame` always emits (`reduced_still_picture_hdr = true`
    // unconditionally, `key_frame.rs:489`). Spec 5.9.2 reads
    // `disable_cdf_update` and (when `seq_force_screen_content_tools ==
    // SELECT`) `allow_screen_content_tools` UNCONDITIONALLY, right after the
    // `reduced_still_picture_header` branch -- confirmed against THIS repo's
    // own writer (`aom_dsp::entropy::header::write_frame_header_prefix`,
    // proven bit-for-bit identical to real aomenc across hundreds of gate
    // cells: both bits are written OUTSIDE the `if !reduced_still_picture_hdr`
    // block). zenavif-parse's `else` branch nests both reads inside the
    // non-reduced-still-picture arm, so they are skipped for every still
    // AVIF, misaligning every subsequent field including `base_q_idx`.
    // MEASURED here (not asserted, since the bug is upstream and this file's
    // job is to validate the PORT + the container round-trip, not
    // zenavif-parse's own frame-header convenience parser): the same cq=32
    // config reads back base_q_idx 48/64/0/72 depending on frame size
    // (expected 128 in every case -- stable per size, drifting with it,
    // consistent with the tile_info() bit-length, which depends on frame
    // size, compounding a fixed few-bit misalignment differently per size).
    // zenavif-parse's repo is archived (imazen/zenavif-parse, confirmed via
    // `gh issue create` refusing with "Repository was archived so is
    // read-only" 2026-09-03), so this could not be filed upstream; left here
    // instead as the durable record. Container-level extraction
    // (`primary_data()`, asserted above) is UNAFFECTED -- this bug is scoped
    // to the bonus frame-header fields only.
    eprintln!(
        "{label}: (informational, not asserted -- see the comment above) \
         zenavif-parse base_q_idx={:?} lossless={:?}, this port's own qindex={}",
        meta.base_q_idx,
        meta.lossless,
        base_qindex_from_cq(cq)
    );

    // ---- decode the independently-extracted payload with THIS port's decoder
    let dec = aom_decode::frame::decode_frame_obus(&extracted).unwrap_or_else(|e| {
        panic!(
            "{label}: this port's decoder could not decode \
             zenavif-parse's extracted payload: {e}"
        )
    });
    assert_eq!(dec.width, w, "{label}: decoded width");
    assert_eq!(dec.height, h, "{label}: decoded height");

    eprintln!(
        "{label}: {w}x{h} bd{bd} -> {} byte AV1 payload -> {} byte AVIF -> zenavif-parse \
         round-trip byte-exact, metadata agrees, decodes",
        payload.len(),
        avif.len()
    );
}

#[test]
fn standalone_stream_survives_avif_mux_and_zenavif_parse_readback() {
    // Format x bit depth (the B axis from self_contained_key_frame.rs, here
    // through the container instead of the raw byte gate).
    for (nm, mono, sx, sy) in [
        ("mono", true, 1usize, 1usize),
        ("420", false, 1, 1),
        ("422", false, 1, 0),
        ("444", false, 0, 0),
    ] {
        for bd in [8u8, 10, 12] {
            run_cell(
                &format!("fmt_{nm}_bd{bd}"),
                64,
                64,
                bd,
                mono,
                sx,
                sy,
                32,
                false,
            );
        }
    }
    // Sizes, incl. a partial-superblock crop.
    for (w, h) in [(16usize, 16usize), (258, 258), (512, 512)] {
        run_cell(&format!("size_{w}x{h}"), w, h, 8, false, 1, 1, 32, false);
    }
    // Quantizer extremes.
    run_cell("cq0_lossless_arm", 64, 64, 8, false, 1, 1, 0, false);
    run_cell("cq63", 64, 64, 8, false, 1, 1, 63, false);
    // The two capabilities landed alongside this file (2026-09-03): SB128,
    // and (implicitly, since it is the same shell) a non-SB64 tile grid --
    // this is the FIRST container-level check either has ever had.
    run_cell("sb128_256x256", 256, 256, 8, false, 1, 1, 32, true);
    run_cell("sb128_partial_200x150", 200, 150, 8, false, 1, 1, 32, true);
}
