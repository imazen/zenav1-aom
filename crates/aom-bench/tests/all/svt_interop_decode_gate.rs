//! **Cross-encoder IntraBC decode gate: real SVT-AV1 C streams** (GitHub #5).
//!
//! # Why this file exists
//!
//! Every IntraBC decode gate in this repo feeds the decoder a stream that came
//! from **libaom** — the conformance corpus vectors are libaom-encoded, and
//! `armed_tools_decode_gate.rs` decodes this port's own encoder output. That
//! leaves one whole class uncovered: **a conformant IntraBC stream from a
//! different encoder**, which reaches per-block neighbour configurations
//! libaom's encoder simply never emits.
//!
//! GitHub #5 is exactly that hole. On 2026-07-23 the decoder rejected **37 of
//! 100** real SVT-AV1 v4.2.0 screen-content encodes with `corrupt frame:
//! intrabc DV failed validity (non-conformant stream)` while real libaom
//! decoded every one of them clean. KB-29 (2026-08-01) then fixed six IntraBC
//! roots, **three of them decoder-side** — the `do_uniform`-gated 64x64 chunk
//! walk (root 4), the leaf-vs-raster var-tx walk selection (root 5), and the
//! missing CfL luma store on the leaf arm (root 6). Those three are what makes
//! this gate pass today.
//!
//! Re-measured 2026-08-02 on `main` across **1,530 real SVT-AV1 v4.2.0
//! screen-content streams** (10 gb82-sc sources x crops/geometries x presets
//! 0..9 x quantizers x single- and multi-tile): **0 rejected, 0 pixel
//! differences** — see `benchmarks/svt_interop_2026-08-02.md`. This gate pins
//! four of those streams so the coverage cannot silently lapse again.
//!
//! # What this gate asserts, per fixture
//!
//! 1. the fixture bytes are the ones the census below was measured on
//!    (length + FNV-1a-64, so the "this stream contains N IntraBC blocks"
//!    claim is bound to exact bytes rather than assumed);
//! 2. the REAL `aom_codec_av1_dx` accepts it — the authority;
//! 3. the port's decoder accepts it;
//! 4. **both produce identical pixels** on every plane;
//! 5. the decoded frame geometry and tile grid are what the fixture table
//!    claims — in particular that one fixture really is multi-tile, because
//!    `av1_is_dv_valid` reads `tile->mi_col_start/end` directly and a corpus
//!    that is 1x1 everywhere has not exercised those bounds at all.
//!
//! Optionally (6) `dav1d` accepts it, when `AOM_DAV1D_BIN` names the binary —
//! caller-controlled via the justfile, never a skip decided inside the test.
//!
//! # Non-vacuity
//!
//! `ibc_mi` / `ibc_shapes` in the table are **measured**, by the REAL libaom
//! `inspect` example (`CONFIG_INSPECTION=1`, `-ibc -bs`) built from the pinned
//! `upstream/` submodule, and are bound to the fixture bytes by the digest
//! assertion. They are recorded rather than asserted at runtime because the
//! port decoder exposes no per-block IntraBC census and adding one would be a
//! public-API change.
//!
//! The gate's teeth were verified the way KB-29's were — by reverting a
//! decoder root alone and confirming these fixtures fail. See
//! `benchmarks/svt_interop_2026-08-02.md` for the quoted failure.
//!
//! Provenance of the fixtures: `scripts/svt_interop/` (an out-of-tree build of
//! the `zenav1-svt-c` sibling at `v4.2.0-62-gdfbfe849c`, driven at the same
//! still/AVIF CQP config as the `xbench` `svt-c` arm plus
//! `screen_content_mode = 1`). Nothing in that sibling repo is modified.

use std::path::PathBuf;

/// `(file, w, h, tile_cols, tile_rows, len, fnv1a64, ibc_mi, ibc_shapes)`.
///
/// `ibc_mi` is the number of 4x4 mi units whose block carries `use_intrabc`,
/// as counted by the real libaom `inspect`; `ibc_shapes` is that population's
/// block-size histogram (largest contributors first). Between them the four
/// fixtures cover: the largest IntraBC block size (BLOCK_64X64, i.e. the 64x64
/// chunk boundary root 4 lives on), non-square shapes (16X8 / 32X16 / 16X32),
/// 4-px-side shapes (BLOCK_4X4 / 8X4 — the class KB-29 root 1 was about), and
/// a 4x2 tile grid.
const FIXTURES: &[Fixture] = &[
    Fixture {
        file: "svt420_codecwiki_512_ibc64x64.obu",
        w: 512,
        h: 512,
        tiles: (1, 1),
        len: 194,
        fnv: 0x1ed1_3c4e_7407_2b7c,
        ibc_mi: 256,
        ibc_shapes: "BLOCK_64X64:256",
    },
    Fixture {
        file: "svt420_graph_448_ibc_rect.obu",
        w: 448,
        h: 448,
        tiles: (1, 1),
        len: 2308,
        fnv: 0xad80_5e36_443c_494c,
        ibc_mi: 727,
        ibc_shapes: "BLOCK_16X16:304,BLOCK_32X32:256,BLOCK_16X8:56,\
                     BLOCK_32X16:32,BLOCK_16X32:32,BLOCK_8X8:28",
    },
    Fixture {
        file: "svt420_imessage_512_ibc_mixed.obu",
        w: 512,
        h: 512,
        tiles: (1, 1),
        len: 1527,
        fnv: 0xe220_39f3_78dd_5d03,
        ibc_mi: 773,
        ibc_shapes: "BLOCK_64X64:512,BLOCK_32X32:128,BLOCK_16X16:96,\
                     BLOCK_8X8:28,BLOCK_8X16:8,BLOCK_4X4:1",
    },
    Fixture {
        file: "svt420_codecwiki_1920x1080_4x2tiles.obu",
        w: 1920,
        h: 1080,
        tiles: (4, 2),
        len: 4722,
        fnv: 0x1004_06e2_276b_6d20,
        ibc_mi: 107,
        ibc_shapes: "BLOCK_32X32:64,BLOCK_16X16:32,BLOCK_8X8:8,\
                     BLOCK_8X4:2,BLOCK_4X4:1",
    },
];

struct Fixture {
    file: &'static str,
    w: usize,
    h: usize,
    tiles: (usize, usize),
    len: usize,
    fnv: u64,
    ibc_mi: usize,
    ibc_shapes: &'static str,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/svt_interop")
}

/// FNV-1a 64. Binds the committed census to exact bytes without pulling in a
/// hashing dependency for four small files.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The gate proper.
#[test]
fn real_svt_av1_intrabc_streams_round_trip_against_the_c_decoder() {
    let dav1d = std::env::var("AOM_DAV1D_BIN").ok();
    eprintln!(
        "=== SVT-AV1 IntraBC interop gate (GitHub #5) (dav1d leg: {}) ===",
        dav1d.as_deref().unwrap_or("OFF — set AOM_DAV1D_BIN to enable")
    );
    let dir = fixture_dir();
    let mut ran = 0usize;
    let mut total_ibc = 0usize;
    let mut multi_tile = 0usize;

    for f in FIXTURES {
        let path = dir.join(f.file);
        let data = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{}: cannot read the committed SVT fixture ({e}). These are real \
                 SVT-AV1 v4.2.0 C encodes checked into the repo — regenerate with \
                 scripts/svt_interop/ if they are genuinely missing, but do NOT \
                 make this test skip: an absent fixture is a broken gate, not a \
                 reason to pass.",
                path.display()
            )
        });

        // (1) the bytes the census was measured on.
        assert_eq!(
            data.len(),
            f.len,
            "{}: fixture length changed — the committed IntraBC census \
             ({} mi units: {}) was measured on different bytes and no longer \
             applies",
            f.file,
            f.ibc_mi,
            f.ibc_shapes
        );
        assert_eq!(
            fnv1a64(&data),
            f.fnv,
            "{}: fixture content changed — the committed IntraBC census \
             ({} mi units: {}) was measured on different bytes and no longer \
             applies",
            f.file,
            f.ibc_mi,
            f.ibc_shapes
        );

        // (2) THE AUTHORITY: the real `aom_codec_av1_dx`. `ref_decode_av1_kf`
        // asserts on a non-zero shim rc, so a rejection surfaces as a panic.
        let c_dec = std::panic::catch_unwind(|| aom_sys_ref::ref_decode_av1_kf(&data, f.w, f.h));
        let c_dec = match c_dec {
            Ok(d) => d,
            Err(_) => panic!(
                "{}: the REAL C decoder REJECTED a real SVT-AV1 stream. That \
                 makes this fixture a bad stream, not an interop case — it must \
                 not be used as a conformance reference.",
                f.file
            ),
        };

        // (3) the port's decoder must accept it too. This is the assertion
        // GitHub #5 was filed about: 37 of 100 such streams failed it on
        // 2026-07-23 with "intrabc DV failed validity".
        let p_dec = aom_decode::frame::decode_frame_obus(&data).unwrap_or_else(|e| {
            panic!(
                "{}: the REAL C decoder ACCEPTED this SVT-AV1 stream but the PORT \
                 decoder REJECTED it: {e}. The C decoder is the authority, so this \
                 is a decoder-side interop defect — the GitHub #5 / KB-29 class. \
                 The stream carries {} IntraBC mi units ({}). NOTE (playbook §10, \
                 KB-29): a DV-validity message names the first check that failed, \
                 NOT the defect — check for a tile-payload desync upstream of the \
                 failing block before touching `is_dv_valid` or its inputs.",
                f.file, f.ibc_mi, f.ibc_shapes
            )
        });

        // (4) identical pixels on every plane.
        assert_eq!(
            (p_dec.width, p_dec.height),
            (f.w, f.h),
            "{}: port decode geometry",
            f.file
        );
        let count_diff = |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(x, y)| x != y).count();
        assert!(
            p_dec.y.len() == c_dec.y.len() && p_dec.y == c_dec.y,
            "{}: luma differs between the C decoder and the port decoder \
             ({} of {} samples)",
            f.file,
            count_diff(&p_dec.y, &c_dec.y),
            p_dec.y.len()
        );
        assert!(
            p_dec.u == c_dec.u && p_dec.v == c_dec.v,
            "{}: chroma differs between the C decoder and the port decoder — \
             U {} of {}, V {} of {}. (KB-29 root 6 — the missing CfL luma store \
             on the IntraBC leaf arm — presents exactly like this: luma \
             byte-identical, chroma off.)",
            f.file,
            count_diff(&p_dec.u, &c_dec.u),
            p_dec.u.len(),
            count_diff(&p_dec.v, &c_dec.v),
            p_dec.v.len()
        );

        // (5) the tile grid this fixture claims. `av1_is_dv_valid` reads the
        // tile bounds directly, so multi-tile coverage is a load-bearing
        // property of the corpus, not a detail.
        assert_eq!(
            (p_dec.tile_cols, p_dec.tile_rows),
            f.tiles,
            "{}: tile grid is not what the fixture table claims — the corpus's \
             multi-tile DV-bounds coverage has changed",
            f.file
        );

        // (6) optional dav1d leg — an INDEPENDENT implementation.
        if let Some(bin) = &dav1d {
            let out = std::process::Command::new(bin)
                .args(["-i".as_ref(), path.as_os_str(), "-o".as_ref(), "/dev/null".as_ref()])
                .output()
                .unwrap_or_else(|e| panic!("running {bin}: {e}"));
            assert!(
                out.status.success(),
                "{}: dav1d REJECTED a real SVT-AV1 stream that libaom accepts: {}",
                f.file,
                String::from_utf8_lossy(&out.stderr)
            );
        }

        eprintln!(
            "  {:<44} {:>5} B  {}x{}  tiles {}x{}  ibc_mi {}  OK",
            f.file, f.len, f.w, f.h, f.tiles.0, f.tiles.1, f.ibc_mi
        );
        ran += 1;
        total_ibc += f.ibc_mi;
        if f.tiles.0 > 1 || f.tiles.1 > 1 {
            multi_tile += 1;
        }
    }

    // Anti-vacuity on the SUITE, not just on each row: a table that lost its
    // IntraBC-bearing or multi-tile members would still pass every assertion
    // above while covering nothing this gate exists for.
    assert_eq!(ran, FIXTURES.len(), "not every fixture ran");
    assert!(
        ran >= 4 && total_ibc >= 1_000,
        "the fixture table no longer carries enough IntraBC content to exercise \
         the path this gate guards ({ran} fixtures, {total_ibc} IntraBC mi units)"
    );
    assert!(
        multi_tile >= 1,
        "no multi-tile fixture left — `av1_is_dv_valid` reads tile->mi_col_start/end \
         directly, so a 1x1-only corpus does not exercise its bounds"
    );
}
