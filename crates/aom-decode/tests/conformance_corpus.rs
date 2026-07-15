//! GATE-1 CONFORMANCE: the official AV1 decode-conformance vectors.
//!
//! Everything else in the decoder track validates against streams *we* asked
//! the C encoder to make (`real_bitstream.rs` and friends). Those prove the
//! port matches C on the encoder's own choices — but the port and that encoder
//! can share blind spots. This gate closes that: it decodes the standardized
//! AV1 conformance vectors (libaom v3.14.1's own `test-data.sha1` set, hosted
//! at `storage.googleapis.com/aom-test-data`) — bitstreams produced
//! INDEPENDENTLY of our tooling — and asserts three things per frame:
//!
//!   1. the port's decoded planes are byte-identical to the REAL C decoder
//!      (`aom_codec_av1_dx`, via [`aom_sys_ref::ref_decode_av1_kf`]) on the
//!      same bytes — the ground-truth gate;
//!   2. the C decoder's own planes reproduce the shipped golden per-frame MD5
//!      (`conformance/data/<name>.ivf.md5`), computed in libaom's exact
//!      `md5_helper.h::Add(aom_image_t*)` layout — this validates our MD5
//!      layout AND that the corpus bytes are intact;
//!   3. the port's planes reproduce that same golden MD5 — closing the loop
//!      independently of (1).
//!
//! # Scope — single-frame / all-intra / single-shown-KEY (caller-visible)
//!
//! The current decoder is KEY-frame, intra-only (no inter / motion comp). The
//! in-scope frame set is declared EXPLICITLY in [`CORPUS`] below (a
//! caller-visible manifest, not a silent in-test early-return): all-intra
//! vectors decode every frame; the `00-quantizer` vectors are `KEY,INTER` so
//! only frame 0 (the KEY frame) is in scope; the second (INTER) frame is
//! excluded by the manifest. Frame types were verified from the bitstream
//! headers (all listed in-scope frames are KEY, `reduced_still_picture=0`,
//! `show_existing_frame=0`, `frame_size_override=0`).
//!
//! Vectors exercised here (as fetched): the 39-frame 8-bit all-intra vector
//! (SB128 + CDEF + LR), six 10-bit `00-quantizer` vectors (SB128 + CDEF + LR +
//! high bit depth, frame 0 only), and the 8-bit intra-only intrabc "extreme
//! DV" vector (SB64 + intrabc, 1920x1080). These co-exercise SB128, 10-bit,
//! CDEF, loop restoration, and intrabc on streams our encoder never produced.
//!
//! # Corpus presence
//!
//! The `.ivf` bytes are gitignored (repo policy: no large binaries in git). The
//! corpus lives in `conformance/data/` and is fetched with
//! `python3 xtask/conformance.py --fetch --scope intra` (override the dir with
//! `AOM_CONFORMANCE_DIR`). If NOTHING is present the test FAILS LOUD (never a
//! silent pass) with the fetch command — per the no-graceful-skip rule.

use aom_decode::frame::{FrameDecode, decode_frame_obus};
use aom_sys_ref as c;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// RFC 1321 MD5 (safe Rust; self-tested below against known vectors). Used only
// to reproduce libaom's shipped golden per-frame hashes.
// ---------------------------------------------------------------------------
mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    /// Streaming MD5 so we never materialize a whole-frame byte buffer.
    pub struct Md5 {
        a: [u32; 4],
        buf: [u8; 64],
        buf_len: usize,
        total: u64,
    }
    impl Md5 {
        pub fn new() -> Self {
            Md5 {
                a: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
                buf: [0; 64],
                buf_len: 0,
                total: 0,
            }
        }
        pub fn update(&mut self, mut data: &[u8]) {
            self.total = self.total.wrapping_add(data.len() as u64);
            if self.buf_len > 0 {
                let need = 64 - self.buf_len;
                let take = need.min(data.len());
                self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
                self.buf_len += take;
                data = &data[take..];
                if self.buf_len == 64 {
                    let block = self.buf;
                    self.process(&block);
                    self.buf_len = 0;
                }
            }
            while data.len() >= 64 {
                let mut block = [0u8; 64];
                block.copy_from_slice(&data[..64]);
                self.process(&block);
                data = &data[64..];
            }
            if !data.is_empty() {
                self.buf[..data.len()].copy_from_slice(data);
                self.buf_len = data.len();
            }
        }
        pub fn finish(mut self) -> String {
            let bitlen = self.total.wrapping_mul(8);
            let mut pad = vec![0x80u8];
            while (self.total.wrapping_add(pad.len() as u64)) % 64 != 56 {
                pad.push(0);
            }
            pad.extend_from_slice(&bitlen.to_le_bytes());
            self.update(&pad);
            debug_assert_eq!(self.buf_len, 0);
            let mut out = String::with_capacity(32);
            for word in self.a {
                for byte in word.to_le_bytes() {
                    out.push_str(&format!("{byte:02x}"));
                }
            }
            out
        }
        fn process(&mut self, chunk: &[u8; 64]) {
            let mut m = [0u32; 16];
            for i in 0..16 {
                m[i] = u32::from_le_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            let [mut a, mut b, mut c, mut d] = self.a;
            for i in 0..64 {
                let (f, g) = match i {
                    0..=15 => ((b & c) | (!b & d), i),
                    16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let f = f
                    .wrapping_add(a)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g]);
                a = d;
                d = c;
                c = b;
                b = b.wrapping_add(f.rotate_left(S[i]));
            }
            self.a[0] = self.a[0].wrapping_add(a);
            self.a[1] = self.a[1].wrapping_add(b);
            self.a[2] = self.a[2].wrapping_add(c);
            self.a[3] = self.a[3].wrapping_add(d);
        }
    }

    pub fn hex(data: &[u8]) -> String {
        let mut m = Md5::new();
        m.update(data);
        m.finish()
    }
}

/// libaom `md5_helper.h::Add(aom_image_t*)`: hash each plane's cropped rows,
/// 2 bytes/sample little-endian at high bit depth. Chroma dims round UP
/// (`(d + shift) >> shift`). Planes are tightly packed (stride == width) here,
/// which is byte-identical to libaom hashing `w` bytes per strided row.
fn image_md5(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    ss_x: usize,
    ss_y: usize,
    monochrome: bool,
) -> String {
    let mut m = md5::Md5::new();
    let hi = bd > 8;
    let push_plane = |m: &mut md5::Md5, plane: &[u16], pw: usize, ph: usize| {
        assert_eq!(plane.len(), pw * ph, "plane size mismatch");
        let mut row = Vec::with_capacity(pw * if hi { 2 } else { 1 });
        for r in 0..ph {
            row.clear();
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
    push_plane(&mut m, y, w, h);
    if !monochrome {
        let cw = (w + ss_x) >> ss_x;
        let ch = (h + ss_y) >> ss_y;
        push_plane(&mut m, u, cw, ch);
        push_plane(&mut m, v, cw, ch);
    }
    m.finish()
}

/// Split an IVF container into per-frame temporal-unit payloads (raw OBU bytes).
fn ivf_temporal_units(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(data.len() >= 32 && &data[0..4] == b"DKIF", "not an IVF file");
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            as usize;
        off += 12; // 4-byte size + 8-byte timestamp
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

/// Parse `<name>.ivf.md5`: one `"<md5hex>  <name>-WxH-NNNN.i420"` line per
/// decoded frame, in frame order.
fn parse_golden(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .next()
                .expect("golden md5 line empty")
                .to_ascii_lowercase()
        })
        .collect()
}

/// Which frames of a vector are in scope (caller-visible; never a runtime skip).
#[derive(Clone, Copy)]
enum Scope {
    /// Every frame is KEY/intra — decode them all.
    AllIntra,
    /// First `n` frames are KEY; the rest need inter tooling — excluded here.
    FirstKey(usize),
}

struct Vector {
    name: &'static str,
    w: usize,
    h: usize,
    bd: i32,
    ss_x: usize,
    ss_y: usize,
    scope: Scope,
}

/// The Gate-1 single-frame / all-intra / single-shown-KEY corpus. Scope per
/// vector is fixed here from the verified bitstream frame types.
const CORPUS: &[Vector] = &[
    // 39-frame 8-bit all-intra (SB128 + CDEF + LR), every frame KEY.
    Vector { name: "av1-1-b8-02-allintra", w: 352, h: 288, bd: 8, ss_x: 1, ss_y: 1, scope: Scope::AllIntra },
    // 8-bit intra-only intrabc "extreme DV" (SB64), both frames KEY.
    Vector { name: "av1-1-b8-16-intra_only-intrabc-extreme-dv", w: 1920, h: 1080, bd: 8, ss_x: 1, ss_y: 1, scope: Scope::AllIntra },
    // 10-bit 00-quantizer (SB128 + CDEF + LR): KEY,INTER -> frame 0 only.
    Vector { name: "av1-1-b10-00-quantizer-00", w: 640, h: 360, bd: 10, ss_x: 1, ss_y: 1, scope: Scope::FirstKey(1) },
    Vector { name: "av1-1-b10-00-quantizer-01", w: 640, h: 360, bd: 10, ss_x: 1, ss_y: 1, scope: Scope::FirstKey(1) },
    Vector { name: "av1-1-b10-00-quantizer-02", w: 640, h: 360, bd: 10, ss_x: 1, ss_y: 1, scope: Scope::FirstKey(1) },
    Vector { name: "av1-1-b10-00-quantizer-03", w: 640, h: 360, bd: 10, ss_x: 1, ss_y: 1, scope: Scope::FirstKey(1) },
    Vector { name: "av1-1-b10-00-quantizer-04", w: 640, h: 360, bd: 10, ss_x: 1, ss_y: 1, scope: Scope::FirstKey(1) },
    Vector { name: "av1-1-b10-00-quantizer-05", w: 640, h: 360, bd: 10, ss_x: 1, ss_y: 1, scope: Scope::FirstKey(1) },
];

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

#[test]
fn md5_self_test() {
    // Anchor the MD5 implementation against RFC 1321 test vectors before it is
    // trusted to reproduce the shipped goldens.
    assert_eq!(md5::hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5::hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(
        md5::hex(b"The quick brown fox jumps over the lazy dog"),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
    // Cross a 64-byte block boundary (56..64 padding edge) and a multi-block msg.
    assert_eq!(
        md5::hex(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
        "57edf4a22be3c955ac49da2e2107b67a"
    );
}

#[test]
fn conformance_single_frame_intra_byte_identical_to_c_and_golden() {
    let dir = corpus_dir();

    // Presence is caller-visible: FAIL LOUD if nothing is fetched (never a
    // silent pass). Which subset is present is a corpus-management fact; the
    // decoder SCOPE (KEY-only) is fixed in CORPUS above.
    let present: Vec<&Vector> = CORPUS
        .iter()
        .filter(|v| dir.join(format!("{}.ivf", v.name)).exists())
        .collect();
    assert!(
        !present.is_empty(),
        "no conformance vectors found in {}\n\
         fetch them first:  python3 xtask/conformance.py --fetch --scope intra\n\
         (or point AOM_CONFORMANCE_DIR at a populated corpus dir)",
        dir.display()
    );

    let mut report = String::new();
    let mut failures: Vec<String> = Vec::new();
    let mut frames_checked = 0usize;
    let mut vectors_checked = 0usize;
    // Coverage witnesses (anti-vacuous: the corpus must genuinely exercise
    // high bit depth, SB128, CDEF, loop restoration, and intrabc).
    let (mut saw_bd10, mut saw_cdef, mut saw_lr, mut saw_intrabc) = (false, false, false, false);

    for v in &present {
        vectors_checked += 1;
        let ivf = std::fs::read(dir.join(format!("{}.ivf", v.name))).unwrap();
        let golden_txt =
            std::fs::read_to_string(dir.join(format!("{}.ivf.md5", v.name))).unwrap();
        let golden = parse_golden(&golden_txt);
        let tus = ivf_temporal_units(&ivf);
        assert_eq!(
            tus.len(),
            golden.len(),
            "{}: {} temporal units but {} golden md5 lines",
            v.name,
            tus.len(),
            golden.len()
        );

        let in_scope: Vec<usize> = match v.scope {
            Scope::AllIntra => (0..tus.len()).collect(),
            Scope::FirstKey(n) => (0..n.min(tus.len())).collect(),
        };

        for &i in &in_scope {
            let tu = &tus[i];
            let where_ = format!("{} frame {}", v.name, i);

            // (2) C reference decode + golden layout/corpus validation.
            let cref = c::ref_decode_av1_kf(tu, v.w, v.h);
            assert_eq!(cref.info[0], v.bd, "{where_}: C bit depth");
            let md5_c = image_md5(
                &cref.y, &cref.u, &cref.v, v.w, v.h, v.bd, v.ss_x, v.ss_y, cref.info[1] != 0,
            );
            if md5_c != golden[i] {
                failures.push(format!(
                    "{where_}: C-decoder MD5 {md5_c} != golden {} (corpus/layout mismatch)",
                    golden[i]
                ));
                continue;
            }

            // (1) Port decode -> byte-identical to C.
            let rust: FrameDecode = match decode_frame_obus(tu) {
                Ok(fd) => fd,
                Err(e) => {
                    failures.push(format!("{where_}: port rejected in-scope frame: {e}"));
                    continue;
                }
            };
            let mut frame_ok = true;
            if rust.y != cref.y {
                let n = rust.y.iter().zip(&cref.y).take_while(|(a, b)| a == b).count();
                let (x, yy) = (n % v.w, n / v.w);
                failures.push(format!(
                    "{where_}: LUMA differs at pixel {n} (x={x}, y={yy}) port={} c={}",
                    rust.y.get(n).copied().unwrap_or(0),
                    cref.y.get(n).copied().unwrap_or(0)
                ));
                frame_ok = false;
            }
            if !rust.monochrome && rust.u != cref.u {
                let n = rust.u.iter().zip(&cref.u).take_while(|(a, b)| a == b).count();
                failures.push(format!("{where_}: U differs at chroma sample {n}"));
                frame_ok = false;
            }
            if !rust.monochrome && rust.v != cref.v {
                let n = rust.v.iter().zip(&cref.v).take_while(|(a, b)| a == b).count();
                failures.push(format!("{where_}: V differs at chroma sample {n}"));
                frame_ok = false;
            }
            if !frame_ok {
                continue;
            }

            // (3) Port planes reproduce the golden MD5 independently.
            let md5_r = image_md5(
                &rust.y, &rust.u, &rust.v, v.w, v.h, v.bd, v.ss_x, v.ss_y, rust.monochrome,
            );
            if md5_r != golden[i] {
                failures.push(format!("{where_}: port MD5 {md5_r} != golden {}", golden[i]));
                continue;
            }

            // Coverage witnesses from the port's own parse.
            saw_bd10 |= rust.bit_depth >= 10;
            saw_cdef |= rust.cdef_bits != 0
                || rust.cdef_strengths[0] != 0
                || rust.cdef_uv_strengths[0] != 0;
            saw_lr |= rust.lr_frame_restoration_type.iter().any(|&t| t != 0);
            // The intrabc vector is 8-bit SB64 1920x1080; witness it by name +
            // successful decode (intrabc syntax isn't surfaced on FrameDecode).
            saw_intrabc |= v.name.contains("intrabc");

            frames_checked += 1;
            report.push_str(&format!("  OK  {where_}  ({}x{} bd{})\n", v.w, v.h, v.bd));
        }
    }

    eprintln!(
        "conformance: {} vectors present, {} in-scope frames byte-identical (port==C==golden)\n{}",
        vectors_checked, frames_checked, report
    );

    assert!(
        failures.is_empty(),
        "conformance gate: {} frame(s) FAILED:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // Anti-vacuous floors: prove the run actually exercised the tools the
    // corpus carries. These fire only when the relevant vectors are present.
    assert!(frames_checked > 0, "no in-scope frames were checked");
    if present.iter().any(|v| v.bd >= 10) {
        assert!(saw_bd10, "10-bit vectors present but no bd>=10 frame verified");
    }
    if present.iter().any(|v| v.name.contains("intrabc")) {
        assert!(saw_intrabc, "intrabc vector present but not verified");
    }
    // allintra + b10-quantizer both carry CDEF and LR in the sequence header;
    // if either family is present, at least one frame must have applied them.
    if present
        .iter()
        .any(|v| v.name.contains("allintra") || v.name.contains("quantizer"))
    {
        assert!(saw_cdef, "CDEF-carrying vectors present but no CDEF frame verified");
        assert!(saw_lr, "LR-carrying vectors present but no LR frame verified");
    }
}
