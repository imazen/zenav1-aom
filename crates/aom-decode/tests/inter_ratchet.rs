//! CHUNK-2 GATE — the 16x16 inter ratchet.
//!
//! Frame 1 of `av1-1-b8-01-size-16x16` is a real multi-block inter frame: the
//! 16x16 superblock is `PARTITION_HORZ_4` into four `BLOCK_16X4` strips — block 0
//! `NEWMV`, blocks 1-3 `NEARESTMV` (each reading its MV from the spatial ref-mv
//! scan of the block above), single `LAST` reference, EIGHTTAP (non-switchable),
//! `SIMPLE_TRANSLATION`, and — unlike the 64x64 skeleton — **every block carries
//! residual** (`skip = 0`). This exercises: the inter CDF `update_cdf` threading
//! across blocks, the spatial NEARESTMV scan, the 4-tap interp (16x4 luma /
//! sub-8x8 8x2 chroma strips), and the non-skip luma + chroma residual add.
//!
//! The gate decodes both frames through [`aom_decode::frame::decode_frames`] and
//! asserts BOTH reproduce the shipped golden per-frame MD5
//! (`md5_helper.h::Add(aom_image_t*)` exact layout) — a true byte-identity gate.

mod common;

use aom_decode::frame::{FrameDecode, decode_frames};
use common::md5::Md5;
use std::path::PathBuf;

const VECTOR: &str = "av1-1-b8-01-size-16x16";
const GOLDEN_FRAME0: &str = "6353b245c305a5f4f2845ee7ad2b128b";
const GOLDEN_FRAME1: &str = "f4b0078dfbc8b581fa959d4512b9940a";

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

#[test]
fn inter_ratchet_16x16_frame1_byte_identical() {
    let dir = corpus_dir();
    let ivf_path = dir.join(format!("{VECTOR}.ivf"));
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

    assert_eq!(md5_f0, GOLDEN_FRAME0, "frame 0 (KEY) does not match golden");
    assert_eq!(
        md5_f1, GOLDEN_FRAME1,
        "frame 1 (16x16 inter ratchet) does not match golden MD5"
    );
    eprintln!("inter ratchet 16x16: frame 0 {md5_f0} + frame 1 {md5_f1} byte-identical to golden");
}
