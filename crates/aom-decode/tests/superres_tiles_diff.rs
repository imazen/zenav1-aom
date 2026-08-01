//! THE SUPERRES x MULTI-TILE-COLUMN GATE: decode real libaom-encoded KEY streams
//! that carry BOTH fixed-denominator superres AND more than one tile column, and
//! compare every plane BYTE-IDENTICALLY against the REAL C decoder.
//!
//! This combination was refused outright until 2026-07-31
//! (`DecodeError::UnsupportedFeature("multi-tile superres (out of envelope)")`)
//! — a legal AV1 bitstream the port would not decode. It is covered by NOTHING
//! else: the AV1 intra conformance corpus contains zero superres AND zero
//! `tiles > 1` vectors (`benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`),
//! and `superres_diff.rs` is single-tile throughout.
//!
//! What the upscale must do differently across a tile boundary
//! (`av1_upscale_normative_rows`, resize.c:1119): `upscale_normative_rect` pads
//! `UPSCALE_NORMATIVE_TAPS/2 + 1 = 5` columns with the tile's edge pixel ONLY at
//! the frame edges (`pad_left = (j == 0)`, `pad_right = (j == cols - 1)`), so an
//! interior boundary convolves over the neighbouring tile column's real
//! reconstructed pixels; the last tile column takes
//! `upscaled_x1 = upscaled_plane_width` directly rather than the rounded
//! quotient; and `x0_qn` carries between columns.
//!
//! ANTI-VACUITY (asserted per stream, hard — no graceful skip): the parsed
//! header must carry `SuperresDenom > 8` AND `tile_info.cols > 1`, the coded
//! width must be strictly below the upscaled width, at least one tile boundary
//! must fall strictly inside the plane, and the decoded output must be non-flat.
//! A stream that silently collapsed to one tile column FAILS the test rather
//! than passing it vacuously.
//!
//! CONFORMANCE BOUND (found while building these vectors): superres tightens
//! `av1_is_min_tile_width_satisfied` (tile_common.c:200) from "every inner tile
//! column >= 64 luma px" to ">= 128", measured on the CODED (downscaled) frame —
//! and libaom's own ENCODER ignores it, happily emitting `--tile-columns=2` at
//! denominator 16 over a 4-superblock-wide coded frame, which its own DECODER
//! then rejects as "Minimum tile width requirement not satisfied". So the legal
//! grid here is (denominator, frame width, tile-columns) triples whose inner
//! columns span >= 2 superblocks; `superres_multitile_below_min_tile_width_is_
//! rejected` pins the illegal side, asserting the port refuses exactly what C
//! refuses instead of decoding a stream the reference decoder will not.

use aom_decode::frame::{decode_frame_obus, decode_frame_obus_prefilter};
use aom_decode::superres::coded_frame_width;
use aom_sys_ref as c;

/// `AV1E_SET_TILE_COLUMNS` (aomcx.h:393) — the CODED value IS the log2 count.
const AV1E_SET_TILE_COLUMNS: i32 = 33;
/// `AV1E_SET_TILE_ROWS` (aomcx.h:400).
const AV1E_SET_TILE_ROWS: i32 = 34;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Photographic-ish content (smooth gradients + sinusoids + noise) — NOT
/// synthetic-few-colours, which could trip screen-content detection. High
/// frequency in X so the horizontal upscale has real AC to interpolate AND a
/// mis-sampled tile column shows up as a pixel difference.
fn gen_plane(w: usize, h: usize, bd: i32, seed: u64, chroma: bool) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 0.25 + 0.5 * (0.6 * fx + 0.4 * fy);
            let wave = 0.14 * ((fx * 41.0).sin() * (fy * 7.0).cos());
            let noise = ((rng.next() >> 40) as i64 % 33 - 16) as f64 / maxv as f64;
            let mut v = base + wave + noise * if chroma { 2.0 } else { 4.0 };
            v = v.clamp(0.0, 1.0);
            p[r * w + col] = (v * maxv as f64).round() as u16;
        }
    }
    p
}

/// Does the REAL C decoder refuse these bytes? `ref_decode_av1_kf` asserts on a
/// non-zero shim rc, so the rejection is observed by catching that panic (the
/// shim has no fallible variant and `aom-sys-ref` is another track's file). The
/// panic hook is silenced only for the duration of the call so the expected
/// failure does not print a misleading backtrace.
fn c_decoder_rejects(bytes: &[u8], w: usize, h: usize) -> bool {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = c::ref_decode_av1_kf(bytes, w, h);
    }));
    std::panic::set_hook(prev);
    outcome.is_err()
}

struct Cell {
    tile_cols: usize,
    /// Tile-column boundaries in DOWNSCALED luma pixels (`mi << MI_SIZE_LOG2`).
    tile_x: Vec<i32>,
    coded_w: i32,
    src_mi_width: i32,
    /// Narrowest INNER tile column in coded luma pixels (`min_inner_width`).
    min_inner_px: i32,
}

/// Encode with real libaom (fixed-denominator superres + `AV1E_SET_TILE_COLUMNS`),
/// decode with the port AND with the real C decoder, assert every plane is
/// byte-identical, and return the facts that prove the cell was not vacuous.
#[allow(clippy::too_many_arguments)]
fn run_cell(
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss: (i32, i32),
    cq: i32,
    denom: i32,
    tile_cols_log2: i32,
    tile_rows_log2: i32,
    cdef: bool,
    restoration: bool,
    usage: u32,
) -> Cell {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss.0 as usize) >> ss.0, (h + ss.1 as usize) >> ss.1)
    };
    let seed = ((w as u64) << 40)
        ^ ((h as u64) << 24)
        ^ ((bd as u64) << 12)
        ^ ((denom as u64) << 5)
        ^ ((tile_cols_log2 as u64) << 2)
        ^ cq as u64;
    let y = gen_plane(w, h, bd, seed ^ 0x1111, false);
    let u = gen_plane(cw, ch, bd, seed ^ 0x2222, true);
    let v = gen_plane(cw, ch, bd, seed ^ 0x3333, true);

    let bytes = c::ref_encode_av1_kf_superres_ctrls(
        &y,
        &u,
        &v,
        w,
        h,
        bd,
        mono,
        ss.0,
        ss.1,
        cq,
        /*cpu_used=*/ 0,
        cdef,
        restoration,
        usage,
        denom,
        /*two_pass=*/ false,
        &[
            (AV1E_SET_TILE_COLUMNS, tile_cols_log2),
            (AV1E_SET_TILE_ROWS, tile_rows_log2),
        ],
    );
    let label =
        format!("{w}x{h} bd{bd} mono={mono} ss={ss:?} D={denom} tcols_log2={tile_cols_log2}");
    eprintln!("    cell {label} cdef={cdef} lr={restoration} bytes={}", bytes.len());
    assert!(bytes.len() > 50, "{label}: suspiciously small stream");

    // Facts from OUR OWN parse of the real stream.
    let (_, ptcfg, header) = decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("{label}: prefilter rejected: {e}"));
    let denom_coded = header.frame_size.scale_denominator;
    let coded_w = coded_frame_width(header.frame_size.superres_upscaled_width, denom_coded);

    // ANTI-VACUITY 1 — the stream really is superres-scaled.
    assert_eq!(denom_coded, denom, "{label}: coded superres denom mismatch");
    assert!(denom_coded > 8, "{label}: stream is not superres-scaled");
    assert!(
        coded_w < w as i32,
        "{label}: no real downscale (coded {coded_w} !< upscaled {w})"
    );
    assert_eq!(header.frame_size.superres_upscaled_width, w as i32);

    // ANTI-VACUITY 2 — the stream really has more than one tile COLUMN. Without
    // this the cell would exercise the single-tile path and prove nothing about
    // the feature under test.
    let ti = &header.tile_info;
    assert!(
        ti.cols > 1,
        "{label}: stream collapsed to {} tile column(s) — VACUOUS for multi-tile \
         superres (the encoder clamped the request; widen the frame or lower the denom)",
        ti.cols
    );

    // ANTI-VACUITY 3 — the tile grid was derived on the DOWNSCALED mi grid (the
    // KB-14 regime: the header probe parses on the upscaled grid), and at least
    // one boundary falls strictly inside the plane so an interior (unpadded)
    // convolve genuinely happens.
    assert_eq!(
        ti.mi_cols, ptcfg.mi_cols,
        "{label}: tile grid is not on the downscaled mi grid"
    );
    let src_mi_width = ptcfg.mi_cols * 4;
    let tile_x: Vec<i32> = (0..=ti.cols)
        .map(|j| ((ti.col_start_sb[j] << ti.mib_size_log2).min(ti.mi_cols)) * 4)
        .collect();
    assert_eq!(tile_x[0], 0, "{label}: first boundary must be the origin");
    assert_eq!(
        tile_x[ti.cols], src_mi_width,
        "{label}: last boundary must be the mi-aligned plane width"
    );
    for j in 1..ti.cols {
        assert!(
            tile_x[j] > 0 && tile_x[j] < src_mi_width,
            "{label}: interior boundary {j} ({}) is degenerate",
            tile_x[j]
        );
        assert!(
            tile_x[j] > tile_x[j - 1],
            "{label}: tile boundaries are not increasing: {tile_x:?}"
        );
    }

    // ANTI-VACUITY 4 — the cell is on the CONFORMANT side of
    // `av1_is_min_tile_width_satisfied` (>= 128 coded luma px per inner column
    // under superres). A cell that violated it would be rejected by the C
    // decoder, so the byte-identity comparison below could never run; asserting
    // it here turns "the grid drifted illegal" into a precise message instead of
    // an opaque `shim_decode_av1_kf failed (2)`.
    let min_inner_px = (1..ti.cols)
        .map(|j| tile_x[j] - tile_x[j - 1])
        .min()
        .unwrap_or(i32::MAX);
    assert!(
        min_inner_px >= 128,
        "{label}: inner tile column {min_inner_px} px < 128 — libaom's encoder emitted a \
         NON-CONFORMANT stream (av1_is_min_tile_width_satisfied); pick a wider frame, a \
         smaller denominator, or fewer tile columns. Boundaries {tile_x:?}"
    );

    // THE GATE: full port decode vs the REAL C decoder at the upscaled dims.
    let port = decode_frame_obus(&bytes).unwrap_or_else(|e| {
        panic!("{label}: port decode rejected a legal multi-tile superres stream: {e}")
    });
    assert_eq!((port.width, port.height), (w, h), "{label}: upscaled dims");
    assert!(
        port.y.iter().any(|&px| px != port.y[0]),
        "{label}: upscaled luma is constant (nothing to interpolate — vacuous)"
    );
    let cref = c::ref_decode_av1_kf(&bytes, w, h);
    assert_eq!(cref.info[0], bd, "{label}: bit depth");
    assert_eq!(cref.info[1] != 0, mono, "{label}: monochrome flag");
    assert_eq!(port.y, cref.y, "{label}: LUMA mismatch vs the C decoder");
    if mono {
        assert!(port.u.is_empty() && port.v.is_empty(), "{label}: mono chroma");
    } else {
        assert_eq!(port.u, cref.u, "{label}: U mismatch vs the C decoder");
        assert_eq!(port.v, cref.v, "{label}: V mismatch vs the C decoder");
    }

    Cell {
        tile_cols: ti.cols,
        tile_x,
        coded_w,
        src_mi_width,
        min_inner_px,
    }
}

/// The conformant (width, denominator, `tile_columns_log2`) grid: every inner
/// tile column spans >= 2 superblocks in the CODED frame, so
/// `av1_is_min_tile_width_satisfied` holds under superres. Denominator 16 halves
/// the coded frame, so it needs either twice the width or half the columns.
const GRID: &[(usize, i32, i32)] = &[
    // (upscaled width, denom, tile_columns_log2)
    (512, 9, 2),  // coded 455 -> 8 SB -> 4 columns x 2 SB
    (512, 12, 2), // coded 341 -> 6 SB -> 3 columns x 2 SB
    (512, 16, 1), // coded 256 -> 4 SB -> 2 columns x 2 SB
    (768, 16, 2), // coded 384 -> 6 SB -> 3 columns x 2 SB
];

/// LUMA chunk: monochrome superres streams across the conformant grid, 8- and
/// 10-bit. Monochrome isolates the luma tile walk.
#[test]
fn superres_multitile_luma_byte_identical_to_c() {
    let mut n = 0u32;
    let mut max_cols = 0usize;
    let mut interior_boundaries = 0u32;
    for &h in &[96usize, 128] {
        for &bd in &[8i32, 10] {
            for &(w, denom, tcl) in GRID {
                let cell = run_cell(w, h, bd, true, (1, 1), 28, denom, tcl, 0, false, false, 0);
                max_cols = max_cols.max(cell.tile_cols);
                interior_boundaries += (cell.tile_cols - 1) as u32;
                assert!(cell.min_inner_px >= 128);
                n += 1;
            }
        }
    }
    assert_eq!(n as usize, 2 * 2 * GRID.len(), "multi-tile superres luma arm count");
    // A THREE-column stream has a tile with NEITHER pad_left NOR pad_right — the
    // fully-interior case, where both of the convolve's edges read a neighbour.
    assert!(
        max_cols >= 3,
        "no cell reached 3+ tile columns (max {max_cols}) — the fully-interior \
         tile column (neither pad_left nor pad_right) was never exercised"
    );
    eprintln!(
        "superres x multi-tile luma: {n} streams byte-identical, max {max_cols} tile columns, \
         {interior_boundaries} interior boundaries crossed"
    );
}

/// CHROMA chunk: 4:2:0 (where `ss_x` halves the tile-column boundaries) and
/// 4:4:4, 8/10-bit, with CDEF and loop restoration rotated in so the tile walk is
/// exercised composed with the filters that surround it.
#[test]
fn superres_multitile_chroma_byte_identical_to_c() {
    let mut n = 0u32;
    let mut subsampled_cells = 0u32;
    for &h in &[96usize, 128] {
        for &(bd, ss) in &[
            (8i32, (1i32, 1i32)), // 4:2:0 — subsampled boundaries
            (8, (0, 0)),          // 4:4:4
            (10, (1, 1)),
        ] {
            for &(w, denom, tcl) in GRID {
                let cdef = n % 2 == 0;
                let restoration = n % 3 == 0;
                let usage = if (n & 1) == 0 { 0u32 } else { 2 };
                let cell = run_cell(
                    w,
                    h,
                    bd,
                    false,
                    ss,
                    28,
                    denom,
                    tcl,
                    0,
                    cdef,
                    restoration,
                    usage,
                );
                // At 4:2:0 the chroma boundary list is the luma one >> 1; confirm
                // the luma boundaries really are even, so the shift is exact.
                if ss.0 == 1 {
                    for &x in &cell.tile_x {
                        assert_eq!(x % 2, 0, "4:2:0 luma boundary {x} is not shiftable");
                    }
                    subsampled_cells += 1;
                }
                assert!(cell.coded_w > 0 && cell.tile_cols > 1);
                n += 1;
            }
        }
    }
    assert_eq!(n as usize, 2 * 3 * GRID.len(), "multi-tile superres chroma arm count");
    assert!(
        subsampled_cells >= 8,
        "4:2:0 (subsampled tile boundaries) barely exercised ({subsampled_cells})"
    );
    eprintln!("superres x multi-tile chroma: {n} streams byte-identical");
}

/// CONFORMANCE gate for the constraint superres introduces: an inner tile column
/// narrower than `64 << superres_scaled` coded luma pixels is NOT a decodable
/// stream. libaom's encoder emits one for `--tile-columns=2` at denominator 16
/// over a 4-superblock-wide coded frame; its own decoder rejects it
/// ("Minimum tile width requirement not satisfied", decodeframe.c:5115). The
/// port must refuse it too — decoding a stream the reference decoder refuses is
/// a divergence, not a capability.
///
/// Structured as an ASYMMETRIC PAIR so the rejection is attributable. For each
/// frame, the SAME content/denominator is encoded twice, differing only in
/// `tile_columns_log2`:
///
/// * the CONTROL at `log2 = k` splits the coded frame into columns of exactly 2
///   superblocks (`min_inner_px == 128`, asserted) — C accepts it and the port
///   byte-matches C;
/// * the SUBJECT at `log2 = k + 1` halves those columns to 1 superblock
///   (64 px < 128) — C rejects it and the port must too.
///
/// So the only variable between accept and reject is the inner tile width, and
/// the control simultaneously proves the encode/decode chain is healthy for that
/// frame. Deleting the port's `av1_is_min_tile_width_satisfied` check flips the
/// subject half red while the control half stays green.
#[test]
fn superres_multitile_below_min_tile_width_is_rejected() {
    // (w, h, denom, control tile_columns_log2). The subject is control + 1.
    let cases: &[(usize, usize, i32, i32)] = &[
        (512, 96, 16, 1),  // coded 256 = 4 SB: 2 cols x 2 SB  ->  4 cols x 1 SB
        (512, 96, 12, 2),  // coded 341 = 6 SB: 3 cols x 2 SB  ->  6 cols x 1 SB
        (768, 128, 16, 2), // coded 384 = 6 SB: 3 cols x 2 SB  ->  6 cols x 1 SB
    ];
    let mut n = 0u32;
    for &(w, h, denom, ctl_log2) in cases {
        // CONTROL — conformant, byte-identical to C (run_cell asserts both, and
        // that every inner column is >= 128 px).
        let control = run_cell(w, h, 8, true, (1, 1), 28, denom, ctl_log2, 0, false, false, 0);
        assert_eq!(
            control.min_inner_px, 128,
            "{w}x{h} D={denom}: control inner width is {}px, not the 2-superblock \
             128px this pair depends on (boundaries {:?}) — the subject below may no \
             longer be the 1-superblock shape",
            control.min_inner_px, control.tile_x
        );

        // SUBJECT — same everything, one more tile-column halving: 64px inner
        // columns, below the superres floor of `64 << 1`.
        let y = gen_plane(w, h, 8, 0xC0FFEE ^ w as u64 ^ ((denom as u64) << 16), false);
        let bytes = c::ref_encode_av1_kf_superres_ctrls(
            &y,
            &[],
            &[],
            w,
            h,
            8,
            true,
            1,
            1,
            28,
            0,
            false,
            false,
            0,
            denom,
            false,
            &[
                (AV1E_SET_TILE_COLUMNS, ctl_log2 + 1),
                (AV1E_SET_TILE_ROWS, 0),
            ],
        );
        let label = format!("{w}x{h} D={denom} tcols_log2={}", ctl_log2 + 1);
        assert!(bytes.len() > 50, "{label}: suspiciously small stream");

        // HALF 1 — the REAL C decoder refuses it (aomdec reports "Minimum tile
        // width requirement not satisfied"). This is what makes the cell
        // non-conformant rather than merely unusual.
        assert!(
            c_decoder_rejects(&bytes, w, h),
            "{label}: the C decoder ACCEPTED it, while accepting the {}-column control \
             too — the conformance premise of this gate is wrong",
            control.tile_cols
        );

        // HALF 2 — so must the port, as MALFORMED (not "unsupported feature":
        // the tool is supported, the stream is not conformant).
        match decode_frame_obus(&bytes) {
            Ok(_) => panic!(
                "{label}: the port DECODED a non-conformant stream the C decoder rejects \
                 — av1_is_min_tile_width_satisfied is not being enforced"
            ),
            Err(e) => {
                assert_eq!(
                    e.category(),
                    "malformed",
                    "{label}: wrong rejection category for a conformance violation: {e}"
                );
                let msg = e.to_string();
                assert!(
                    msg.contains("minimum tile width"),
                    "{label}: rejected for the WRONG reason ({msg}) — this cell must be \
                     refused by the min-tile-width check, not by some earlier gate"
                );
                eprintln!(
                    "  min-tile-width pair {w}x{h} D={denom}: control {} cols @128px OK, \
                     subject log2={} rejected by C and by the port ({e})",
                    control.tile_cols,
                    ctl_log2 + 1
                );
            }
        }
        n += 1;
    }
    assert_eq!(n as usize, cases.len(), "all min-tile-width pairs ran");
}

/// TILE-ROWS chunk. Tile ROWS do not affect the horizontal upscale (superres is
/// horizontal only) but do change the tile decode order and the entropy contexts
/// feeding it, so a rows x cols grid composed with superres is the crossing worth
/// pinning. CDEF is on so the filter that immediately precedes the upscale is
/// composed with it.
#[test]
fn superres_multitile_grid_byte_identical_to_c() {
    // (w, h, bd, denom, tile_cols_log2, tile_rows_log2)
    let cases: &[(usize, usize, i32, i32, i32, i32)] = &[
        (512, 128, 8, 16, 1, 1),
        (512, 128, 8, 9, 2, 1),
        (512, 192, 8, 12, 2, 2),
        (768, 128, 10, 16, 2, 1),
    ];
    let mut n = 0u32;
    let mut cols_seen = Vec::new();
    for &(w, h, bd, denom, tcl, trl) in cases {
        let cell = run_cell(w, h, bd, true, (1, 1), 32, denom, tcl, trl, true, false, 2);
        cols_seen.push(cell.tile_cols);
        eprintln!(
            "  grid {w}x{h} bd{bd} D{denom} tcols={} boundaries={:?} coded_w={} mi_w={} \
             min_inner={}px",
            cell.tile_cols, cell.tile_x, cell.coded_w, cell.src_mi_width, cell.min_inner_px
        );
        n += 1;
    }
    assert_eq!(n as usize, cases.len(), "all grid cells ran");
    assert!(
        cols_seen.iter().all(|&c| c > 1),
        "a grid cell collapsed to one tile column: {cols_seen:?}"
    );
    eprintln!("superres x multi-tile grid: {n} streams byte-identical, cols {cols_seen:?}");
}
