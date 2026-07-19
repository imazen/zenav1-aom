//! CHUNK-2/3 GATE — the inter ratchet, now covering a partial-edge frame.
//!
//! # 16x16 (chunk 2) — multi-block, residual-carrying
//! Frame 1 of `av1-1-b8-01-size-16x16` is a real multi-block inter frame: the
//! 16x16 superblock is `PARTITION_HORZ_4` into four `BLOCK_16X4` strips — block 0
//! `NEWMV`, blocks 1-3 `NEARESTMV` (each reading its MV from the spatial ref-mv
//! scan of the block above), single `LAST` reference, EIGHTTAP (non-switchable),
//! `SIMPLE_TRANSLATION`, and — unlike the 64x64 skeleton — **every block carries
//! residual** (`skip = 0`). This exercises: the inter CDF `update_cdf` threading
//! across blocks, the spatial NEARESTMV scan, the 4-tap interp (16x4 luma /
//! sub-8x8 8x2 chroma strips), and the non-skip luma + chroma residual add.
//!
//! # 64x66 (chunk 3) — partial-edge single-ref, 128-SB
//! Frame 1 of `av1-1-b8-01-size-64x66` is the simplest PARTIAL-edge inter frame
//! (STEP-0 census `/tmp/inspect_frame`): a **single** `BLOCK_64X128` clipped to
//! the 64x66 frame — a `use_128x128_superblock` frame whose 128-SB roots a
//! `split_or_vert` forced partition at the right edge (`has_cols == false`),
//! yielding one 64x128 block at mi(0,0), `NEWMV` mv=(-1,-7), single `LAST`,
//! `SIMPLE_TRANSLATION`, `skip = 1` (pure MC, no residual). The partial-edge
//! wrinkle is entirely in motion compensation: the block's nominal 128-tall
//! predictor overhangs the 66px-tall frame, so its bottom interp taps must
//! edge-replicate at the reference's VISIBLE (crop) boundary (64x66 / UV 32x33),
//! NOT the SB/mi-aligned recon extent (64x72 / UV 32x36). The chunk-3 fix stores
//! the reference's crop dims in `RefFrame` (C's `av1_setup_pre_planes` loads
//! `crop_widths/crop_heights` into `pre_buf->width/height`). The `clamp_mv_to_
//! umv_border` frame-edge MV clamp already existed and does NOT fire here (the
//! MV is far inside the border), so 64x66 pins the crop-dim border path.
//!
//! (The other partial-edge `01-size-*` vectors — 16x18/16x34/16x66 — pull in
//! OBMC / WARPED_CAUSAL / switchable-interp-with-neighbours per the census, so
//! they are Part B chunk-4 feature targets, not gated here. See INTER-FEATURES-
//! PLAN.md.)
//!
//! Each gate decodes both frames through [`aom_decode::frame::decode_frames`] and
//! asserts BOTH reproduce the shipped golden per-frame MD5
//! (`md5_helper.h::Add(aom_image_t*)` exact layout) — a true byte-identity gate.

mod common;

use aom_decode::frame::{FrameDecode, decode_frames};
use common::md5::Md5;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    if let Ok(d) = std::env::var("AOM_CONFORMANCE_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("data")
}

fn ivf_temporal_units(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(data.len() >= 32 && &data[0..4] == b"DKIF", "not an IVF file");
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 12;
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

/// `md5_helper.h::Add(aom_image_t*)`: hash each cropped plane row-by-row.
fn image_md5(fd: &FrameDecode) -> String {
    let mut m = Md5::new();
    let hi = fd.bit_depth > 8;
    let push = |m: &mut Md5, plane: &[u16], pw: usize, ph: usize| {
        assert_eq!(plane.len(), pw * ph, "plane size mismatch");
        for r in 0..ph {
            let mut row = Vec::with_capacity(pw * if hi { 2 } else { 1 });
            for &s in &plane[r * pw..r * pw + pw] {
                if hi {
                    row.extend_from_slice(&s.to_le_bytes());
                } else {
                    row.push(s as u8);
                }
            }
            m.update(&row);
        }
    };
    push(&mut m, &fd.y, fd.width, fd.height);
    if fd.monochrome {
        let (cw, ch) = ((fd.width + 1) >> 1, (fd.height + 1) >> 1);
        let neutral = vec![1u16 << (fd.bit_depth - 1); cw * ch];
        push(&mut m, &neutral, cw, ch);
        push(&mut m, &neutral, cw, ch);
    } else {
        push(&mut m, &fd.u, fd.width_uv, fd.height_uv);
        push(&mut m, &fd.v, fd.width_uv, fd.height_uv);
    }
    m.finish()
}

/// Decode the 2-frame `vector` (KEY + INTER) and assert both frames reproduce
/// their shipped golden per-frame MD5s (a true byte-identity gate).
fn ratchet_two_frame(vector: &str, golden_f0: &str, golden_f1: &str) {
    let dir = corpus_dir();
    let ivf_path = dir.join(format!("{vector}.ivf"));
    let ivf = match std::fs::read(&ivf_path) {
        Ok(b) => b,
        Err(e) => panic!(
            "conformance vector {} not found ({e}). Fetch with \
             `python3 xtask/conformance.py --fetch --scope intra` or set AOM_CONFORMANCE_DIR.",
            ivf_path.display()
        ),
    };

    let tus = ivf_temporal_units(&ivf);
    assert_eq!(tus.len(), 2, "target vector has exactly 2 frames (KEY + INTER)");

    let mut stream = tus[0].clone();
    stream.extend_from_slice(&tus[1]);

    let frames = decode_frames(&stream).expect("multi-frame decode of the 2-frame stream");
    assert_eq!(frames.len(), 2, "two shown frames decoded");

    let md5_f0 = image_md5(&frames[0]);
    let md5_f1 = image_md5(&frames[1]);

    assert_eq!(md5_f0, golden_f0, "{vector}: frame 0 (KEY) does not match golden");
    assert_eq!(
        md5_f1, golden_f1,
        "{vector}: frame 1 (inter) does not match golden MD5"
    );
    eprintln!("inter ratchet {vector}: frame 0 {md5_f0} + frame 1 {md5_f1} byte-identical to golden");
}

#[test]
fn inter_ratchet_16x16_frame1_byte_identical() {
    ratchet_two_frame(
        "av1-1-b8-01-size-16x16",
        "6353b245c305a5f4f2845ee7ad2b128b",
        "f4b0078dfbc8b581fa959d4512b9940a",
    );
}

/// CHUNK-3 GATE: partial-edge single-ref inter (128-SB, `BLOCK_64X128` clipped to
/// 64x66). Pins the reference crop-dim border path — frame 1's bottom interp taps
/// edge-replicate at the visible 64x66 / UV 32x33 boundary, not the mi-aligned
/// 64x72 / UV 32x36 recon extent.
#[test]
fn inter_ratchet_64x66_partial_edge_frame1_byte_identical() {
    ratchet_two_frame(
        "av1-1-b8-01-size-64x66",
        "3cdad59695184adee0254b28bf2eb412",
        "86f20606b0408bd3ba6771a6a37df429",
    );
}
