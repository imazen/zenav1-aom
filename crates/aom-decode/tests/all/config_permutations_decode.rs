//! DECODER CONFIG-PERMUTATION GATE — the *combination* half of Gate 1.
//!
//! The decoder's configuration space is the set of BITSTREAM FEATURE
//! COMBINATIONS it must handle: sequence-header shape (SB 64/128, bit depth,
//! monochrome / 4:2:0 / 4:2:2 / 4:4:4, reduced-still-picture header) crossed
//! with frame-header shape (lossless, tx mode, reduced_tx_set,
//! disable_cdf_update, delta-q / delta-lf, screen-content tools, tile grid,
//! loop-filter levels, quantizer matrices, segmentation) crossed with the
//! post-reconstruction filter chain (deblock -> CDEF -> superres -> LR).
//!
//! The pre-existing decoder gates (`conformance_corpus.rs`, `real_bitstream.rs`
//! and friends) each own ONE of those axes and hold the rest at a default. That
//! leaves the *crossings* unproven. This file closes the highest-risk crossings
//! — the ones where two features steer the SAME code (per-SB delta-lf feeding
//! the deblock level derivation; per-tile context resets carrying the delta-q /
//! segment-id / CDF state; QM riding the per-segment dequant; the 4:2:2 chroma
//! grid under CDEF/LR at `mib_size = 32`).
//!
//! # What every cell asserts
//!
//! 1. The stream is produced by the REAL libaom v3.14.1 encoder
//!    (`aom_codec_av1_cx`, via the `aom_sys_ref` shims) — never a hand-built
//!    header, so no cell can be testing our own misunderstanding of the syntax.
//! 2. The REALIZED feature tuple is read back from the PORT'S OWN PARSE
//!    (`decode_frame_obus_prefilter` -> `KfTileConfig` + `FrameHeaderObu`) and
//!    checked against the cell's declared requirements. A cell named
//!    `..._deltalf_...` whose stream came out with `delta_lf_present = 0` FAILS
//!    — coverage is measured from bitstream CONTENT, never from a test name.
//! 3. The port's decode is BYTE-IDENTICAL to the REAL C decoder
//!    (`aom_codec_av1_dx`, `aom_sys_ref::ref_decode_av1_kf`) on every plane.
//!
//! # Why the cells COMPOSE instead of permuting
//!
//! The decoder is a pipeline: header parse -> tile split -> per-block symbol
//! decode -> reconstruct -> deblock -> CDEF -> superres -> loop restoration.
//! Byte-identity is an all-or-nothing property of the whole pipeline, so a
//! single cell that has N features LIVE AT ONCE exercises every pairwise
//! interaction among those N in one decode. Permuting the same N axes would
//! cost 2^N cells and prove nothing extra. Each cell below therefore stacks as
//! many axes as the encoder will co-emit, and the `Req` list pins which ones
//! are genuinely live so the composition cannot silently thin out.
//!
//! Axis pairs deliberately NOT crossed, with the reason, are tabulated in
//! `docs/DECODER_CONFIG_COVERAGE_2026-07-30.md` (§ "Collapse table"). The two
//! kinds are: (a) pairs whose port read-sites are provably disjoint, covered in
//! parallel instead; (b) pairs that are genuinely unreachable — either libaom
//! will not co-emit them (superres x lossless: `--lossless=1` makes libaom drop
//! superres, `features.all_lossless = coded_lossless && !av1_superres_scaled`,
//! `av1/encoder/encodeframe.c:276`) or the PORT rejects them by design
//! (superres x multi-tile-COLUMN, `aom-decode/src/frame.rs:752`) — recorded as
//! open holes, not as covered.
//!
//! # GROUP SR — the superres crossings (hole H14)
//!
//! Superres used to be uncrossable: `shim_encode_av1_kf_superres` took no
//! control list, so every superres stream in the suite was SB64 / single-tile /
//! no-QM / no-seg / 4:2:0 or 4:4:4 / no delta. The shim now takes the same
//! `extra_ctrl_ids/vals/n` passthrough `encode_kf_pass` has (plus a `two_pass`
//! flag, which is what makes `--aq-mode=1` genuinely segment a KEY frame), so
//! the SR cells below cross superres with SB128, tile ROWS, QM, segmentation,
//! 4:2:2, delta-q/delta-lf, `disable_cdf_update`, `reduced_tx_set`, mono and
//! 12-bit. That matters most for LOOP RESTORATION: `superres_scaled` is read
//! INSIDE the RU-grid derivation (`aom-dsp/src/entropy/lr.rs`,
//! `lr_corners_in_sb`) while the SB walk runs on the DOWNSCALED grid, so the
//! superres x SB128 pairing changes `mi_size_wide` from 16 to 32 on exactly
//! the arithmetic that superres rescales.
//!
//! # Encoder speed
//!
//! Cells pick `--cpu-used` per cell (0..6) purely as a means of reaching the
//! target feature tuple cheaply. The decoder does not care how a conformant
//! stream was produced — only the realized tuple matters, and the realized
//! tuple is asserted. Several tuples (nonzero loop-filter levels at moderate q,
//! `delta_q_present` from `--deltaq-mode=2`) are only reachable at specific
//! speeds on this content; the per-cell `cpu` value is the probed one.

use aom_decode::frame::{decode_frame_obus, decode_frame_obus_prefilter};
use aom_sys_ref as c;

// ---------------------------------------------------------------------------
// libaom v3.14.1 `aome_enc_control_id` values, cross-checked against the pinned
// header at `upstream/aom/aomcx.h` (the same source
// `aom_sys_ref::cx_ctrl` reads). The shim applies caller-supplied controls
// AFTER its own base set, in order, so these override the base values
// (`encode_kf_pass`, `crates/aom-sys-ref/shim/dec_shim.c`).
// ---------------------------------------------------------------------------
const AV1E_SET_LOSSLESS: i32 = 31; // aomcx.h:366
const AV1E_SET_TILE_COLUMNS: i32 = 33; // aomcx.h:393  (log2 count)
const AV1E_SET_TILE_ROWS: i32 = 34; // aomcx.h:411  (log2 count)
const AV1E_SET_AQ_MODE: i32 = 40; // aomcx.h:481 (1 = VARIANCE_AQ -> segments)
const AV1E_SET_CDF_UPDATE_MODE: i32 = 44; // aomcx.h (0 = never update)
const AV1E_SET_SUPERBLOCK_SIZE: i32 = 56; // aomcx.h:664 (1 = 128x128)
const AV1E_SET_ENABLE_CDEF: i32 = 58; // aomcx.h:684
const AV1E_SET_ENABLE_RESTORATION: i32 = 59; // aomcx.h:694
const AV1E_SET_ENABLE_QM: i32 = 63; // aomcx.h:732
const AV1E_SET_QM_MIN: i32 = 64; // aomcx.h:745
const AV1E_SET_QM_MAX: i32 = 65; // aomcx.h:757
const AV1E_SET_ENABLE_PALETTE: i32 = 104; // aomcx.h:1123
const AV1E_SET_DELTAQ_MODE: i32 = 107; // aomcx.h:1151
const AV1E_SET_DELTALF_MODE: i32 = 108; // aomcx.h:1159
const AV1E_SET_REDUCED_TX_TYPE_SET: i32 = 118; // aomcx.h:1213
const SB_128X128: i32 = 1; // aom_superblock_size_t

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

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

/// Photographic-ish: smooth gradients + sinusoids + noise. Deliberately NOT
/// few-colour/flat, which would trip the encoder's screen-content detection.
fn gen_photo(w: usize, h: usize, bd: i32, seed: u64, chroma: bool) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 0.25 + 0.5 * (0.6 * fx + 0.4 * fy);
            let wave = 0.12 * ((fx * 9.0).sin() * (fy * 7.0).cos());
            let noise = ((rng.next() >> 40) as i64 % 33 - 16) as f64 / maxv as f64;
            let v = (base + wave + noise * if chroma { 2.0 } else { 4.0 }).clamp(0.0, 1.0);
            p[r * w + col] = (v * maxv as f64).round() as u16;
        }
    }
    p
}

/// Screen-ish: eight flat colours on a 16px grid with hard edges — what the
/// encoder's screen-content detector wants before it turns palette on.
fn gen_screen(w: usize, h: usize, bd: i32) -> Vec<u16> {
    let maxv = (1u32 << bd) - 1;
    let pal = [
        0,
        maxv,
        maxv / 4,
        maxv / 2,
        (maxv * 3) / 4,
        maxv / 8,
        (maxv * 7) / 8,
        maxv / 3,
    ];
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let idx = ((col / 16) * 3 + (r / 16) * 5) % pal.len();
            let edge = usize::from((col % 16) < 3 || (r % 16) < 3);
            p[r * w + col] = pal[(idx + edge) % pal.len()] as u16;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Content {
    Photo,
    Screen,
}

/// Which real-libaom entry point makes the stream. All three route to the same
/// `encode_av1_kf_impl`; they differ in which knobs are reachable.
enum Enc {
    /// `ref_encode_av1_kf_ctrls` — one-pass, arbitrary control overrides.
    /// Segmentation is NOT reachable here: `--aq-mode` only produces
    /// `SEG_LVL_ALT_Q` segments through the two-pass recode loop.
    Ctrls(&'static [(i32, i32)]),
    /// `ref_encode_av1_kf_qm` — two-pass capable, so `aq` genuinely segments;
    /// carries cdef / restoration / qm level.
    Qm {
        aq: u32,
        qm_level: i32,
        cdef: bool,
        lr: bool,
    },
    /// `ref_encode_av1_kf_tiles` — two-pass capable AND tile/SB128 capable.
    Tiles {
        aq: u32,
        two_pass: bool,
        sb128: bool,
        tcl: i32,
        trl: i32,
        cdef: bool,
        lr: bool,
    },
    /// `ref_encode_av1_kf_superres_ctrls` — fixed-denominator superres
    /// (`rc_superres_mode = AOM_SUPERRES_FIXED`, `rc_superres_kf_denominator =
    /// denom`) with the SAME arbitrary-control passthrough `Ctrls` has, plus a
    /// `two_pass` flag (needed for `AV1E_SET_AQ_MODE` to genuinely segment —
    /// a one-pass encode takes libaom's `encode_without_recode`, which never
    /// runs `av1_vaq_frame_setup`). `w`/`h` are the FULL upscaled dims; the
    /// encoder codes at `(w*8 + denom/2)/denom` and the decoder upscales back.
    Superres {
        denom: i32,
        cdef: bool,
        lr: bool,
        two_pass: bool,
        ctrls: &'static [(i32, i32)],
    },
}

/// A realized-bitstream requirement. Every cell declares the features it claims
/// to cover; the gate reads them back from the port's own parse and FAILS the
/// cell if any is absent. This is what makes the coverage matrix evidence
/// rather than nomenclature.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Req {
    Sb128,
    Sb64,
    /// tile_cols >= a, tile_rows >= b (and cols*rows > 1).
    Tiles(usize, usize),
    DeltaQ,
    DeltaLf,
    ReducedTxSet,
    NoCdfUpdate,
    Qm,
    Seg,
    Lossless,
    ScreenTools,
    /// CDEF syntax present AND non-trivial (a strength or a per-SB literal).
    CdefLive,
    /// At least one plane's frame_restoration_type != RESTORE_NONE.
    LrLive,
    /// Luma deblock genuinely runs (filter_level[0] | [1] != 0).
    DeblockLuma,
    /// Chroma deblock genuinely runs (filter_level_u | _v != 0).
    DeblockChroma,
    TxSelect,
    /// `reduced_still_picture_hdr` in the sequence header.
    ReducedStillHdr,
    /// The frame is genuinely SUPERRES-SCALED (`SuperresDenom > 8`, i.e. the
    /// coded width is strictly below the upscaled width).
    Superres,
}

struct Cell {
    label: &'static str,
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss: (i32, i32),
    cq: i32,
    /// `--cpu-used`, chosen per cell to reach the target tuple cheaply.
    cpu: i32,
    /// `--usage`: 0 = GOOD, 2 = ALL_INTRA (the zenavif/avifenc still path).
    usage: u32,
    content: Content,
    enc: Enc,
    require: &'static [Req],
}

/// Everything the gate reads back out of the port's own parse.
struct Realized {
    sb128: bool,
    bd: i32,
    mono: bool,
    ss: (usize, usize),
    reduced_still_hdr: bool,
    superres: bool,
    sr_denom: i32,
    qindex: i32,
    coded_lossless: bool,
    tx_select: bool,
    reduced_tx_set: bool,
    disable_cdf_update: bool,
    delta_q: bool,
    delta_lf: bool,
    screen_tools: bool,
    intrabc: bool,
    tiles: (usize, usize),
    lf_luma: bool,
    lf_chroma: bool,
    cdef_live: bool,
    lr: [u8; 3],
    qm: bool,
    qm_level: usize,
    seg: bool,
    bytes: usize,
}

impl Realized {
    fn has(&self, r: Req) -> bool {
        match r {
            Req::Sb128 => self.sb128,
            Req::Sb64 => !self.sb128,
            Req::Tiles(a, b) => {
                self.tiles.0 >= a && self.tiles.1 >= b && self.tiles.0 * self.tiles.1 > 1
            }
            Req::DeltaQ => self.delta_q,
            Req::DeltaLf => self.delta_lf,
            Req::ReducedTxSet => self.reduced_tx_set,
            Req::NoCdfUpdate => self.disable_cdf_update,
            Req::Qm => self.qm && self.qm_level <= 14,
            Req::Seg => self.seg,
            Req::Lossless => self.coded_lossless,
            Req::ScreenTools => self.screen_tools,
            Req::CdefLive => self.cdef_live,
            Req::LrLive => self.lr.iter().any(|&t| t != 0),
            Req::DeblockLuma => self.lf_luma,
            Req::DeblockChroma => self.lf_chroma,
            Req::TxSelect => self.tx_select,
            Req::ReducedStillHdr => self.reduced_still_hdr,
            Req::Superres => self.superres && self.sr_denom > 8,
        }
    }

    fn summary(&self) -> String {
        format!(
            "sb128={} bd={} mono={} ss={:?} stillhdr={} srD={} q={} lossless={} txsel={} rtx={} \
             nocdf={} dq={} dlf={} screen={} ibc={} tiles={}x{} lf_y={} lf_uv={} cdef={} lr={:?} \
             qm={}(L{}) seg={} bytes={}",
            self.sb128 as u8,
            self.bd,
            self.mono as u8,
            self.ss,
            self.reduced_still_hdr as u8,
            self.sr_denom,
            self.qindex,
            self.coded_lossless as u8,
            self.tx_select as u8,
            self.reduced_tx_set as u8,
            self.disable_cdf_update as u8,
            self.delta_q as u8,
            self.delta_lf as u8,
            self.screen_tools as u8,
            self.intrabc as u8,
            self.tiles.0,
            self.tiles.1,
            self.lf_luma as u8,
            self.lf_chroma as u8,
            self.cdef_live as u8,
            self.lr,
            self.qm as u8,
            self.qm_level,
            self.seg as u8,
            self.bytes,
        )
    }
}

fn encode(cell: &Cell) -> Vec<u8> {
    let (cw, ch) = if cell.mono {
        (0, 0)
    } else {
        (
            (cell.w + cell.ss.0 as usize) >> cell.ss.0,
            (cell.h + cell.ss.1 as usize) >> cell.ss.1,
        )
    };
    let seed = 0x5EED ^ ((cell.w as u64) << 20) ^ ((cell.h as u64) << 8) ^ cell.bd as u64;
    let (y, u, v) = match cell.content {
        Content::Photo => (
            gen_photo(cell.w, cell.h, cell.bd, seed ^ 1, false),
            gen_photo(cw, ch, cell.bd, seed ^ 2, true),
            gen_photo(cw, ch, cell.bd, seed ^ 3, true),
        ),
        Content::Screen => (
            gen_screen(cell.w, cell.h, cell.bd),
            gen_screen(cw, ch, cell.bd),
            gen_screen(cw, ch, cell.bd),
        ),
    };
    let bytes = match cell.enc {
        Enc::Ctrls(ctrls) => c::ref_encode_av1_kf_ctrls(
            &y, &u, &v, cell.w, cell.h, cell.bd, cell.mono, cell.ss.0, cell.ss.1, cell.cq,
            cell.cpu, cell.usage, ctrls,
        ),
        Enc::Qm {
            aq,
            qm_level,
            cdef,
            lr,
        } => c::ref_encode_av1_kf_qm(
            &y, &u, &v, cell.w, cell.h, cell.bd, cell.mono, cell.ss.0, cell.ss.1, cell.cq,
            cell.cpu, cdef, lr, cell.usage, aq, /*two_pass=*/ true, qm_level, qm_level,
        ),
        Enc::Tiles {
            aq,
            two_pass,
            sb128,
            tcl,
            trl,
            cdef,
            lr,
        } => c::ref_encode_av1_kf_tiles(
            &y, &u, &v, cell.w, cell.h, cell.bd, cell.mono, cell.ss.0, cell.ss.1, cell.cq,
            cell.cpu, cdef, lr, cell.usage, aq, two_pass, sb128, tcl, trl,
        ),
        Enc::Superres {
            denom,
            cdef,
            lr,
            two_pass,
            ctrls,
        } => c::ref_encode_av1_kf_superres_ctrls(
            &y, &u, &v, cell.w, cell.h, cell.bd, cell.mono, cell.ss.0, cell.ss.1, cell.cq,
            cell.cpu, cdef, lr, cell.usage, denom, two_pass, ctrls,
        ),
    };
    assert!(
        bytes.len() > 40,
        "{}: suspiciously small stream ({} bytes)",
        cell.label,
        bytes.len()
    );
    bytes
}

/// Parse the realized tuple out of the PORT'S OWN header parse.
fn realize(cell: &Cell, bytes: &[u8]) -> Realized {
    let (_t, cfg, fh) = decode_frame_obus_prefilter(bytes)
        .unwrap_or_else(|e| panic!("{}: port rejected its own gate stream: {e}", cell.label));
    Realized {
        sb128: cfg.sb_size_128,
        bd: cfg.bd,
        mono: cfg.monochrome,
        ss: (cfg.subsampling_x, cfg.subsampling_y),
        reduced_still_hdr: fh.prefix.reduced_still_picture_hdr,
        superres: fh.superres_scaled,
        sr_denom: fh.frame_size.scale_denominator,
        qindex: cfg.base_qindex,
        coded_lossless: fh.coded_lossless,
        tx_select: fh.tx_mode_select,
        reduced_tx_set: fh.reduced_tx_set_used,
        disable_cdf_update: cfg.disable_cdf_update,
        delta_q: cfg.delta_q_present,
        delta_lf: cfg.delta_lf_present,
        screen_tools: cfg.allow_screen_content_tools,
        intrabc: cfg.allow_intrabc,
        tiles: (fh.tile_info.cols, fh.tile_info.rows),
        lf_luma: fh.loopfilter.filter_level[0] != 0 || fh.loopfilter.filter_level[1] != 0,
        lf_chroma: fh.loopfilter.filter_level_u != 0 || fh.loopfilter.filter_level_v != 0,
        cdef_live: fh.cdef.cdef_bits != 0
            || fh.cdef.cdef_strengths[0] != 0
            || fh.cdef.cdef_uv_strengths[0] != 0,
        lr: cfg.lr.frame_restoration_type,
        qm: cfg.using_qmatrix,
        qm_level: cfg.qm_y,
        seg: cfg.seg.enabled,
        bytes: bytes.len(),
    }
}

/// The gate proper: byte-identity against the REAL C decoder on every plane.
fn assert_byte_identical(cell: &Cell, bytes: &[u8], r: &Realized) {
    let rust = decode_frame_obus(bytes)
        .unwrap_or_else(|e| panic!("{} [{}]: port decode failed: {e}", cell.label, r.summary()));
    let cref = c::ref_decode_av1_kf(bytes, cell.w, cell.h);
    assert_eq!(cref.info[0], cell.bd, "{}: C bit depth", cell.label);
    assert_eq!(cref.info[1] != 0, cell.mono, "{}: C monochrome", cell.label);
    if rust.y != cref.y {
        let n = rust
            .y
            .iter()
            .zip(&cref.y)
            .take_while(|(a, b)| a == b)
            .count();
        panic!(
            "{} [{}]: LUMA differs from the C decoder at pixel {n} (x={}, y={}): port={} c={}",
            cell.label,
            r.summary(),
            n % cell.w,
            n / cell.w,
            rust.y.get(n).copied().unwrap_or(0),
            cref.y.get(n).copied().unwrap_or(0),
        );
    }
    if cell.mono {
        assert!(
            rust.u.is_empty() && rust.v.is_empty(),
            "{}: monochrome decode produced chroma planes",
            cell.label
        );
    } else {
        for (plane, (a, b)) in [("U", (&rust.u, &cref.u)), ("V", (&rust.v, &cref.v))] {
            if a != b {
                let n = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
                panic!(
                    "{} [{}]: {plane} differs from the C decoder at sample {n}: port={} c={}",
                    cell.label,
                    r.summary(),
                    a.get(n).copied().unwrap_or(0),
                    b.get(n).copied().unwrap_or(0),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The cell table.
//
// Each row composes as many axes as the real encoder will co-emit on this
// content. The `require` list is the anti-vacuity contract: it is checked
// against the port's parse of the produced bytes, so the cell cannot quietly
// degrade into a plain 4:2:0/SB64/no-tools stream.
// ---------------------------------------------------------------------------

const CELLS: &[Cell] = &[
    // ===================================================================
    // GROUP D — per-SB delta-q / delta-lf.
    //
    // `delta_lf_present` was NOT exercised end-to-end by ANY prior gate: it
    // is absent from all 235 conformance vectors, and `real_bitstream.rs`
    // pins `--deltaq-mode=0`. It is the only frame-header flag that changes
    // the per-superblock DEBLOCK LEVEL derivation, so it must be crossed with
    // genuinely-nonzero filter levels (D1/D2) or it proves nothing.
    // `delta_q_present` appears in exactly 2 corpus vectors (the film-grain
    // pair, SB64 / 4:2:0 / TX_MODE_LARGEST), never with tiles or SB128.
    // ===================================================================
    Cell {
        label: "D1_deltaq_deltalf_deblock_cdef_lr_420_bd10",
        w: 96,
        h: 80,
        bd: 10,
        mono: false,
        ss: (1, 1),
        cq: 52,
        cpu: 4,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_DELTAQ_MODE, 2),
            (AV1E_SET_DELTALF_MODE, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[
            Req::DeltaQ,
            Req::DeltaLf,
            Req::DeblockLuma,
            Req::DeblockChroma,
            Req::Sb64,
            Req::ReducedStillHdr,
        ],
    },
    Cell {
        // 4:2:2 + delta-lf + live chroma deblock: the 4:2:2 chroma level
        // derivation is the one place delta-lf and the subsampled chroma grid
        // meet. Width 130 -> chroma width 65 (ODD half-width edge).
        label: "D2_deltaq_deltalf_deblock_422_bd10",
        w: 130,
        h: 96,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 52,
        cpu: 4,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_DELTAQ_MODE, 3),
            (AV1E_SET_DELTALF_MODE, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[
            Req::DeltaQ,
            Req::DeltaLf,
            Req::DeblockLuma,
            Req::DeblockChroma,
        ],
    },
    Cell {
        // delta-q/lf x MULTI-TILE: `current_base_qindex` and the delta-lf
        // carry restart at each tile (`TileKf::start_tile`), so a per-SB delta
        // stream over >1 tile is a genuine two-feature interaction.
        label: "D3_deltaq_deltalf_tiles2x2_cdef_lr_420_bd8",
        w: 256,
        h: 192,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 4,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_DELTAQ_MODE, 3),
            (AV1E_SET_DELTALF_MODE, 1),
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[
            Req::DeltaQ,
            Req::DeltaLf,
            Req::Tiles(2, 2),
            Req::CdefLive,
            Req::LrLive,
        ],
    },
    Cell {
        // delta-q/lf x SB128 x 4:4:4: the delta carry with `mib_size = 32`.
        label: "D4_deltaq_deltalf_sb128_444_bd8",
        w: 192,
        h: 160,
        bd: 8,
        mono: false,
        ss: (0, 0),
        cq: 44,
        cpu: 5,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_DELTAQ_MODE, 2),
            (AV1E_SET_DELTALF_MODE, 1),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::DeltaQ, Req::DeltaLf, Req::Sb128],
    },
    Cell {
        // The densest delta cell: per-SB delta-q/lf carried across BOTH tile
        // axes at `mib_size = 32`, with CDEF and LR live on top.
        label: "D5_deltaq_deltalf_sb128_tiles2x2_cdef_lr_420_bd8",
        w: 384,
        h: 320,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 6,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_DELTAQ_MODE, 3),
            (AV1E_SET_DELTALF_MODE, 1),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::DeltaQ, Req::DeltaLf, Req::Sb128, Req::Tiles(2, 2)],
    },
    // ===================================================================
    // GROUP R — reduced_tx_set.
    //
    // `reduced_tx_set_used` is 0 in ALL 235 conformance vectors and is never
    // set by any prior real-bitstream gate: the flag switches the ext-tx SET
    // TYPE (and therefore the tx-type symbol alphabet and its CDF) for every
    // coded block. Prior coverage was symbol-level only (the synthetic mirror
    // encoder in `tile_roundtrip.rs`), never end-to-end against C.
    // ===================================================================
    Cell {
        label: "R1_reduced_tx_set_cdef_lr_420_bd8",
        w: 96,
        h: 80,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 2,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_REDUCED_TX_TYPE_SET, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::ReducedTxSet, Req::TxSelect, Req::Sb64],
    },
    Cell {
        label: "R2_reduced_tx_set_422_deblock_bd10",
        w: 130,
        h: 96,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_REDUCED_TX_TYPE_SET, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::ReducedTxSet, Req::DeblockLuma, Req::DeblockChroma],
    },
    Cell {
        label: "R3_reduced_tx_set_sb128_mono_cdef",
        w: 192,
        h: 160,
        bd: 8,
        mono: true,
        ss: (1, 1),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_REDUCED_TX_TYPE_SET, 1),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[
            Req::ReducedTxSet,
            Req::Sb128,
            Req::CdefLive,
            Req::DeblockLuma,
        ],
    },
    Cell {
        // reduced_tx_set x MULTI-TILE x 4:4:4: the tx-type CDF is one of the
        // per-tile-reset contexts, so the reduced alphabet has to be re-seeded
        // correctly at every tile start.
        label: "R4_reduced_tx_set_tiles2x2_444_bd8",
        w: 256,
        h: 192,
        bd: 8,
        mono: false,
        ss: (0, 0),
        cq: 36,
        cpu: 5,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_REDUCED_TX_TYPE_SET, 1),
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::ReducedTxSet, Req::Tiles(2, 2), Req::TxSelect],
    },
    // ===================================================================
    // GROUP Q — quantizer matrices, crossed.
    //
    // `qm_streams_decode_byte_identical_to_c` pins `cdef=false,
    // restoration=false, aq=0, single tile, SB64` — every QM crossing was
    // open, and QM is absent from the conformance corpus entirely (0/235).
    // QM enters the per-block dequant, which is exactly what segmentation's
    // per-segment qindex also moves, and what CDEF/LR then filter.
    // ===================================================================
    Cell {
        label: "Q1_qm5_cdef_lr_420_bd8",
        w: 96,
        h: 80,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 2,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_ENABLE_QM, 1),
            (AV1E_SET_QM_MIN, 5),
            (AV1E_SET_QM_MAX, 5),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Qm, Req::CdefLive, Req::LrLive],
    },
    Cell {
        label: "Q2_qm5_sb128_tiles2x2_cdef_420_bd8",
        w: 256,
        h: 192,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 5,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_ENABLE_QM, 1),
            (AV1E_SET_QM_MIN, 5),
            (AV1E_SET_QM_MAX, 5),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Qm, Req::Sb128, Req::Tiles(2, 2), Req::CdefLive],
    },
    Cell {
        // 4:2:2 QM: `qm_v` is only coded when `separate_uv_delta_q`, and the
        // U/V matrices index the 4:2:2 chroma tx shapes.
        label: "Q3_qm0_422_deblock_bd10",
        w: 130,
        h: 96,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_ENABLE_QM, 1),
            (AV1E_SET_QM_MIN, 0),
            (AV1E_SET_QM_MAX, 0),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Qm, Req::DeblockLuma, Req::DeblockChroma],
    },
    Cell {
        // QM x SEGMENTATION x CDEF x LR: two independent dequant modifiers
        // stacked (`av1_get_qindex(seg, ...)` then `av1_get_iqmatrix`).
        label: "Q4_qm5_seg_cdef_lr_420_bd8",
        w: 96,
        h: 80,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 2,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Qm {
            aq: 1,
            qm_level: 5,
            cdef: true,
            lr: true,
        },
        require: &[Req::Qm, Req::Seg, Req::CdefLive, Req::LrLive],
    },
    Cell {
        // QM x SEGMENTATION x 4:2:2 x LR x live deblock — the densest cell.
        label: "Q5_qm8_seg_422_lr_deblock_bd10",
        w: 130,
        h: 96,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 40,
        cpu: 2,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Qm {
            aq: 1,
            qm_level: 8,
            cdef: true,
            lr: true,
        },
        require: &[
            Req::Qm,
            Req::Seg,
            Req::LrLive,
            Req::DeblockLuma,
            Req::DeblockChroma,
        ],
    },
    Cell {
        // QM x monochrome x SB128: the mono path codes no `qm_u`/`qm_v`.
        label: "Q6_qm12_mono_sb128_cdef",
        w: 192,
        h: 160,
        bd: 8,
        mono: true,
        ss: (1, 1),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_ENABLE_QM, 1),
            (AV1E_SET_QM_MIN, 12),
            (AV1E_SET_QM_MAX, 12),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Qm, Req::Sb128, Req::CdefLive],
    },
    Cell {
        // QM x per-SB DELTA-Q: both modify the SAME per-block dequant —
        // `av1_get_qindex` picks the qindex (now carrying the SB delta) and
        // `av1_get_iqmatrix` then weights it. The one crossing where the two
        // dequant modifiers provably share code.
        label: "Q7_qm5_deltaq_deltalf_cdef_lr_420_bd8",
        w: 192,
        h: 160,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 4,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_ENABLE_QM, 1),
            (AV1E_SET_QM_MIN, 5),
            (AV1E_SET_QM_MAX, 5),
            (AV1E_SET_DELTAQ_MODE, 3),
            (AV1E_SET_DELTALF_MODE, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Qm, Req::DeltaQ, Req::DeltaLf],
    },
    // ===================================================================
    // GROUP T — multi-tile crossings.
    //
    // `multi_tile_streams_decode_byte_identical_to_c` sweeps 4:2:0 / 4:4:4 /
    // mono at bd 8+10 with CDEF/LR/segmentation, but never with
    // disable_cdf_update, lossless, 4:2:2, bd12, QM or screen tools. Tiles
    // are absent from the conformance corpus entirely (1x1 in all 235).
    // ===================================================================
    Cell {
        // disable_cdf_update x tiles: the reader's non-adapting mode must hold
        // across the per-tile `KfFrameContext` resets, and the header codes
        // `context_update_tile_id` + `tile_size_bytes` only when tiles > 1.
        label: "T1_nocdfupdate_tiles2x2_cdef_lr_420_bd8",
        w: 256,
        h: 192,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 40,
        cpu: 4,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_CDF_UPDATE_MODE, 0),
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[
            Req::NoCdfUpdate,
            Req::Tiles(2, 2),
            Req::CdefLive,
            Req::LrLive,
        ],
    },
    Cell {
        // lossless x SB128 x tiles: `all_lossless` drops the loop-filter /
        // CDEF / restoration header sections entirely, so the tile split has
        // to land on a header of a different SHAPE.
        label: "T2_lossless_sb128_tiles_420_bd8",
        w: 256,
        h: 192,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 0,
        cpu: 5,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_LOSSLESS, 1),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_TILE_COLUMNS, 1),
        ]),
        require: &[Req::Lossless, Req::Sb128, Req::Tiles(2, 1)],
    },
    Cell {
        label: "T3_422_tiles2x2_cdef_lr_bd8",
        w: 260,
        h: 192,
        bd: 8,
        mono: false,
        ss: (1, 0),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Tiles(2, 2), Req::CdefLive, Req::LrLive],
    },
    Cell {
        label: "T4_bd12_tiles2x2_cdef_deblock_420",
        w: 256,
        h: 192,
        bd: 12,
        mono: false,
        ss: (1, 1),
        cq: 24,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[
            Req::Tiles(2, 2),
            Req::CdefLive,
            Req::DeblockLuma,
            Req::DeblockChroma,
        ],
    },
    Cell {
        label: "T5_mono_tiles2x2_cdef_deblock_bd10",
        w: 256,
        h: 192,
        bd: 10,
        mono: true,
        ss: (1, 1),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_TILE_COLUMNS, 1),
            (AV1E_SET_TILE_ROWS, 1),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Tiles(2, 2), Req::CdefLive, Req::DeblockLuma],
    },
    Cell {
        // Screen-content tools (palette alphabet) x tiles x 4:4:4. The palette
        // colour cache is seeded from ABOVE/LEFT neighbours, which reset at
        // tile boundaries.
        label: "T6_screentools_tiles_444_bd8",
        w: 256,
        h: 128,
        bd: 8,
        mono: false,
        ss: (0, 0),
        cq: 20,
        cpu: 4,
        usage: 2,
        content: Content::Screen,
        enc: Enc::Ctrls(&[(AV1E_SET_ENABLE_PALETTE, 1), (AV1E_SET_TILE_COLUMNS, 1)]),
        require: &[Req::ScreenTools, Req::Tiles(2, 1)],
    },
    Cell {
        // SEGMENTATION x MULTI-TILE: the segment-id spatial predictor reads
        // above/left segment ids, which must restart at each tile edge.
        label: "T7_seg_tiles2x2_cdef_lr_420_bd8",
        w: 256,
        h: 192,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 3,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Tiles {
            aq: 1,
            two_pass: true,
            sb128: false,
            tcl: 1,
            trl: 1,
            cdef: true,
            lr: true,
        },
        require: &[Req::Seg, Req::Tiles(2, 2), Req::CdefLive],
    },
    Cell {
        // SEGMENTATION x SB128 x MULTI-TILE x 4:4:4.
        label: "T8_seg_sb128_tiles_444_bd8",
        w: 384,
        h: 256,
        bd: 8,
        mono: false,
        ss: (0, 0),
        cq: 36,
        cpu: 5,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Tiles {
            aq: 1,
            two_pass: true,
            sb128: true,
            tcl: 1,
            trl: 0,
            cdef: true,
            lr: true,
        },
        require: &[Req::Seg, Req::Sb128, Req::Tiles(2, 1)],
    },
    Cell {
        // disable_cdf_update x SB128 x 4:2:2 — the second `nocdf` witness, on
        // a different sequence shape from T1's SB64 4:2:0.
        label: "T9_nocdfupdate_sb128_422_cdef_bd10",
        w: 192,
        h: 160,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_CDF_UPDATE_MODE, 0),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::NoCdfUpdate, Req::Sb128, Req::CdefLive],
    },
    // ===================================================================
    // GROUP L — lossless crossings. The lossless gate sweeps 4:2:0 / 4:4:4 /
    // mono at SB64, single tile; the corpus adds SB128 4:2:0 (the two
    // `quantizer-00` vectors). 4:2:2 lossless was open.
    // ===================================================================
    Cell {
        label: "L1_lossless_422_bd10",
        w: 130,
        h: 96,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 0,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[(AV1E_SET_LOSSLESS, 1)]),
        require: &[Req::Lossless, Req::Sb64],
    },
    Cell {
        label: "L2_lossless_mono_tiles_bd8",
        w: 256,
        h: 128,
        bd: 8,
        mono: true,
        ss: (1, 1),
        cq: 0,
        cpu: 5,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[(AV1E_SET_LOSSLESS, 1), (AV1E_SET_TILE_COLUMNS, 1)]),
        require: &[Req::Lossless, Req::Tiles(2, 1)],
    },
    // ===================================================================
    // GROUP S — SB128 crossings not reached by `sb128_streams_...`
    // (which sweeps 4:2:0 / 4:4:4 / mono at bd 8+10 only).
    // ===================================================================
    Cell {
        label: "S1_422_sb128_cdef_lr_bd8",
        w: 192,
        h: 160,
        bd: 8,
        mono: false,
        ss: (1, 0),
        cq: 40,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Sb128, Req::CdefLive],
    },
    Cell {
        label: "S2_bd12_sb128_cdef_deblock_420",
        w: 192,
        h: 160,
        bd: 12,
        mono: false,
        ss: (1, 1),
        cq: 24,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            (AV1E_SET_ENABLE_CDEF, 1),
            (AV1E_SET_ENABLE_RESTORATION, 1),
        ]),
        require: &[Req::Sb128, Req::DeblockLuma, Req::DeblockChroma],
    },
    Cell {
        label: "S3_screentools_sb128_444_bd8",
        w: 256,
        h: 128,
        bd: 8,
        mono: false,
        ss: (0, 0),
        cq: 20,
        cpu: 4,
        usage: 2,
        content: Content::Screen,
        enc: Enc::Ctrls(&[
            (AV1E_SET_ENABLE_PALETTE, 1),
            (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
        ]),
        require: &[Req::ScreenTools, Req::Sb128],
    },
    Cell {
        // 12-bit 4:4:4 (sequence profile 2's other corner) x CDEF x LR —
        // `bd12_composition_decodes_byte_identical_to_c` covers bd12 at 4:2:0
        // and 4:4:4 SB64 single-tile; this adds the deblocked/CDEF 4:4:4 12-bit
        // shape alongside T4/S2's 4:2:0 12-bit tile/SB128 crossings.
        label: "S4_bd12_444_cdef_lr_deblock",
        w: 130,
        h: 96,
        bd: 12,
        mono: false,
        ss: (0, 0),
        cq: 24,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Ctrls(&[(AV1E_SET_ENABLE_CDEF, 1), (AV1E_SET_ENABLE_RESTORATION, 1)]),
        require: &[Req::DeblockLuma, Req::DeblockChroma],
    },
    // ===================================================================
    // GROUP SR — superres crossings (hole H14).
    //
    // Before the `shim_encode_av1_kf_superres` control passthrough, superres
    // could not be co-emitted with ANY other tool: it is an
    // `aom_codec_enc_cfg_t` field, not an `aome_enc_control_id`, and the shim
    // hardcoded SB64 / single tile / deltaq+aq off / no QM / one-pass. Every
    // superres stream in `superres_diff.rs` (90 of them) therefore sits at that
    // one point of the config space.
    //
    // Superres is a NORMATIVE post-CDEF stage: the frame is CODED at the
    // reduced width and upscaled horizontally before loop restoration. The
    // sharp edge is that `superres_scaled` is read inside the RU-grid
    // derivation (`lr_corners_in_sb`) — the RU grid is in UPSCALED units while
    // the SB/tile walk is in DOWNSCALED mi — so SB size (mi_size_wide 16 -> 32)
    // and the tile split genuinely interact with the superres rescale.
    // ===================================================================
    Cell {
        // THE core crossing: superres x SB128 x LR x CDEF. `lr_corners_in_sb`
        // runs at `mi_size_wide = 32` against a superres-rescaled RU column
        // mapping (`u = D * MI_SIZE * m / N`).
        //
        // The 640-px WIDTH is load-bearing, not decorative. libaom picks
        // `unit_size = 256` here, so the upscaled RU grid needs an upscaled
        // width > 512 to have THREE unit columns; only then does an SB128's
        // 32-mi span reach a unit corner that a 16-mi span would not
        // (at D=12 a unit is 2048/48 = 42.67 downscaled mi, so the SB at
        // mi_col 64 codes RU 2 with `mi_size_wide = 32` and NOTHING with 16).
        // At 384/512 px wide the grid is 2 columns and the SB size cannot
        // change the outcome — MEASURED: an SB64-assumption planted in the
        // superres arm of `lr_corners_in_sb` leaves 384x192 and 512x192
        // byte-identical and breaks 640x192 and 768x192. This is the teeth
        // cell — see § "Teeth" in
        // `docs/DECODER_CONFIG_COVERAGE_2026-07-30.md`.
        label: "SR1_superres_d12_sb128_lr_cdef_420_bd8",
        w: 640,
        h: 192,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 12,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[(AV1E_SET_SUPERBLOCK_SIZE, SB_128X128)],
        },
        require: &[Req::Superres, Req::Sb128, Req::LrLive, Req::CdefLive],
    },
    Cell {
        // superres x TILE ROWS x SB128 x LR x CDEF. Tile COLUMNS are out of the
        // port's envelope under superres (`frame.rs:752` — the per-tile-column
        // upscale walk of `av1_upscale_normative_rows` is unported), but the
        // upscale is horizontal-only, so a tile ROW split is in-envelope and is
        // a real crossing: the per-tile entropy/CDF reset and the tile-row
        // boundary both land inside a superres frame.
        label: "SR2_superres_d12_tilerows_sb128_lr_420_bd8",
        w: 384,
        h: 512,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 5,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 12,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[
                (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
                (AV1E_SET_TILE_COLUMNS, 0),
                (AV1E_SET_TILE_ROWS, 1),
            ],
        },
        require: &[
            Req::Superres,
            Req::Sb128,
            Req::Tiles(1, 2),
            Req::LrLive,
            Req::CdefLive,
        ],
    },
    Cell {
        // superres x per-SB DELTA-Q x DELTA-LF x live deblock x CDEF x LR. The
        // delta-lf carry moves the DEBLOCK level, which runs BEFORE the
        // superres upscale; the delta-q carry feeds the dequant of a frame
        // coded at the reduced width.
        label: "SR3_superres_d12_deltaq_deltalf_deblock_cdef_420_bd10",
        w: 256,
        h: 192,
        bd: 10,
        mono: false,
        ss: (1, 1),
        cq: 52,
        cpu: 4,
        usage: 0,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 12,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[(AV1E_SET_DELTAQ_MODE, 3), (AV1E_SET_DELTALF_MODE, 1)],
        },
        require: &[
            Req::Superres,
            Req::DeltaQ,
            Req::DeltaLf,
            Req::DeblockLuma,
            Req::DeblockChroma,
            Req::CdefLive,
        ],
    },
    Cell {
        // superres x QM x 4:2:2 x LR x live deblock at the STEEPEST denominator
        // (16 = exact 2:1). 4:2:2 chroma is upscaled at its subsampled width,
        // and the 4:2:2 chroma RU grid is the one `lr_corners_in_sb` derives
        // with `(sx, sy) = (1, 0)` on top of the superres rescale.
        label: "SR4_superres_d16_qm5_422_deblock_lr_bd10",
        w: 260,
        h: 160,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 24,
        cpu: 2,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 16,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[
                (AV1E_SET_ENABLE_QM, 1),
                (AV1E_SET_QM_MIN, 5),
                (AV1E_SET_QM_MAX, 5),
            ],
        },
        require: &[
            Req::Superres,
            Req::Qm,
            Req::DeblockLuma,
            Req::LrLive,
            Req::CdefLive,
        ],
    },
    Cell {
        // superres x SEGMENTATION x CDEF x LR. Needs the two-pass sequence: a
        // one-pass encode takes `encode_without_recode`, which never calls
        // `av1_vaq_frame_setup` (verified empirically — `--aq-mode=1` one-pass
        // comes out with `seg.enabled = 0`).
        label: "SR5_superres_d12_seg_twopass_cdef_lr_420_bd8",
        w: 256,
        h: 128,
        bd: 8,
        mono: false,
        ss: (1, 1),
        cq: 36,
        cpu: 3,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 12,
            cdef: true,
            lr: true,
            two_pass: true,
            ctrls: &[(AV1E_SET_AQ_MODE, 1)],
        },
        require: &[
            Req::Superres,
            Req::Seg,
            Req::CdefLive,
            Req::LrLive,
            Req::DeblockLuma,
        ],
    },
    Cell {
        // superres x disable_cdf_update x SB128 x 4:2:2 x LR, denom 16.
        label: "SR6_superres_d16_nocdfupdate_sb128_422_bd10",
        w: 260,
        h: 160,
        bd: 10,
        mono: false,
        ss: (1, 0),
        cq: 30,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 16,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[
                (AV1E_SET_CDF_UPDATE_MODE, 0),
                (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
            ],
        },
        require: &[
            Req::Superres,
            Req::NoCdfUpdate,
            Req::Sb128,
            Req::LrLive,
            Req::CdefLive,
        ],
    },
    Cell {
        // superres x QM x reduced_tx_set x SB128 x 4:4:4 — the reduced ext-tx
        // alphabet and the QM dequant, both on a superres-coded 4:4:4 frame.
        label: "SR7_superres_d12_qm5_reducedtx_sb128_444_bd8",
        w: 384,
        h: 192,
        bd: 8,
        mono: false,
        ss: (0, 0),
        cq: 30,
        cpu: 5,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 12,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[
                (AV1E_SET_SUPERBLOCK_SIZE, SB_128X128),
                (AV1E_SET_ENABLE_QM, 1),
                (AV1E_SET_QM_MIN, 5),
                (AV1E_SET_QM_MAX, 5),
                (AV1E_SET_REDUCED_TX_TYPE_SET, 1),
            ],
        },
        require: &[
            Req::Superres,
            Req::Qm,
            Req::ReducedTxSet,
            Req::Sb128,
            Req::CdefLive,
        ],
    },
    Cell {
        // superres x monochrome x SB128 x LR at denom 16 — the no-chroma
        // upscale path with the SB128 RU walk.
        label: "SR8_superres_d16_mono_sb128_lr_bd8",
        w: 384,
        h: 192,
        bd: 8,
        mono: true,
        ss: (1, 1),
        cq: 30,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 16,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[(AV1E_SET_SUPERBLOCK_SIZE, SB_128X128)],
        },
        require: &[Req::Superres, Req::Sb128, Req::LrLive],
    },
    Cell {
        // superres x 12-bit x SB128 x live deblock — `superres_diff.rs` covers
        // bd12 only at SB64.
        label: "SR9_superres_d12_bd12_sb128_deblock_420",
        w: 256,
        h: 160,
        bd: 12,
        mono: false,
        ss: (1, 1),
        cq: 24,
        cpu: 4,
        usage: 2,
        content: Content::Photo,
        enc: Enc::Superres {
            denom: 12,
            cdef: true,
            lr: true,
            two_pass: false,
            ctrls: &[(AV1E_SET_SUPERBLOCK_SIZE, SB_128X128)],
        },
        require: &[
            Req::Superres,
            Req::Sb128,
            Req::DeblockLuma,
            Req::DeblockChroma,
        ],
    },
];

#[test]
fn config_permutations_decode_byte_identical_to_c() {
    let t_start = std::time::Instant::now();
    let mut failures: Vec<String> = Vec::new();
    let mut report = String::new();
    // Axis witnesses over the whole run (each is also pinned per-cell by the
    // `require` lists; these are the run-level floors).
    let mut seen: std::collections::BTreeMap<&'static str, u32> = std::collections::BTreeMap::new();
    let mut bump = |k: &'static str| *seen.entry(k).or_insert(0) += 1;

    for cell in CELLS {
        let t0 = std::time::Instant::now();
        let bytes = encode(cell);
        let r = realize(cell, &bytes);

        // (a) sequence-level facts must match what we asked the encoder for —
        //     a mismatch means the cell is not testing what it claims.
        assert_eq!(r.bd, cell.bd, "{}: bit depth", cell.label);
        assert_eq!(r.mono, cell.mono, "{}: monochrome", cell.label);
        if !cell.mono {
            assert_eq!(
                r.ss,
                (cell.ss.0 as usize, cell.ss.1 as usize),
                "{}: subsampling",
                cell.label
            );
        }
        // (b) anti-vacuity: every declared feature must be LIVE in the bytes.
        let missing: Vec<String> = cell
            .require
            .iter()
            .filter(|&&q| !r.has(q))
            .map(|q| format!("{q:?}"))
            .collect();
        if !missing.is_empty() {
            failures.push(format!(
                "{}: VACUOUS — required feature(s) {} absent from the produced bitstream [{}]",
                cell.label,
                missing.join(", "),
                r.summary()
            ));
            continue;
        }
        // (c) the gate.
        assert_byte_identical(cell, &bytes, &r);

        // Run-level witnesses.
        if r.sb128 {
            bump("sb128");
        } else {
            bump("sb64");
        }
        if r.tiles.0 * r.tiles.1 > 1 {
            bump("multi_tile");
        }
        if r.tiles.0 > 1 && r.tiles.1 > 1 {
            bump("tiles_both_axes");
        }
        if r.delta_q {
            bump("delta_q");
        }
        if r.delta_lf {
            bump("delta_lf");
        }
        if r.delta_lf && (r.lf_luma || r.lf_chroma) {
            bump("delta_lf_with_live_deblock");
        }
        if r.reduced_tx_set {
            bump("reduced_tx_set");
        }
        if r.disable_cdf_update {
            bump("disable_cdf_update");
        }
        if r.qm {
            bump("qm");
        }
        if r.qm && r.seg {
            bump("qm_x_seg");
        }
        if r.seg {
            bump("seg");
        }
        if r.seg && r.tiles.0 * r.tiles.1 > 1 {
            bump("seg_x_tiles");
        }
        if r.coded_lossless {
            bump("lossless");
        }
        if r.screen_tools {
            bump("screen_tools");
        }
        if r.cdef_live {
            bump("cdef_live");
        }
        if r.lr.iter().any(|&t| t != 0) {
            bump("lr_live");
        }
        if r.lf_luma {
            bump("deblock_luma");
        }
        if r.lf_chroma {
            bump("deblock_chroma");
        }
        if r.mono {
            bump("monochrome");
        }
        match (r.mono, r.ss) {
            (false, (1, 1)) => bump("yuv420"),
            (false, (1, 0)) => bump("yuv422"),
            (false, (0, 0)) => bump("yuv444"),
            _ => {}
        }
        match r.bd {
            8 => bump("bd8"),
            10 => bump("bd10"),
            12 => bump("bd12"),
            _ => {}
        }
        if r.reduced_still_hdr {
            bump("reduced_still_picture_hdr");
        }
        // Superres crossings (GROUP SR). Each pairing gets its own witness so a
        // future edit cannot quietly collapse the group back onto the plain
        // SB64/single-tile/no-tool superres point `superres_diff.rs` already
        // owns.
        if r.superres {
            bump("superres");
            if r.sb128 {
                bump("superres_x_sb128");
            }
            if r.tiles.0 * r.tiles.1 > 1 {
                bump("superres_x_tiles");
            }
            if r.lr.iter().any(|&t| t != 0) {
                bump("superres_x_lr");
            }
            if r.sb128 && r.lr.iter().any(|&t| t != 0) {
                bump("superres_x_sb128_x_lr");
            }
            if r.qm {
                bump("superres_x_qm");
            }
            if r.seg {
                bump("superres_x_seg");
            }
            if r.delta_lf {
                bump("superres_x_deltalf");
            }
            if r.disable_cdf_update {
                bump("superres_x_nocdf");
            }
            if r.reduced_tx_set {
                bump("superres_x_reduced_tx_set");
            }
            if !r.mono && r.ss == (1, 0) {
                bump("superres_x_422");
            }
            if r.bd == 12 {
                bump("superres_x_bd12");
            }
            if r.mono {
                bump("superres_x_mono");
            }
            if r.sr_denom == 16 {
                bump("superres_d16");
            }
        }

        report.push_str(&format!(
            "  OK  {:<44} {:>6.0} ms  {}\n",
            cell.label,
            t0.elapsed().as_secs_f64() * 1e3,
            r.summary()
        ));
    }

    eprint!("{report}");
    eprintln!(
        "config permutations: {} cells, {:.1}s total\naxis witnesses: {:?}",
        CELLS.len(),
        t_start.elapsed().as_secs_f64(),
        seen
    );

    assert!(
        failures.is_empty(),
        "config-permutation gate: {} cell(s) FAILED:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // Run-level floors. These are what stop the table from silently thinning
    // out into redundant 4:2:0/SB64/no-tools cells if a future edit rewrites
    // the grid: every axis this file exists to cover must still be witnessed.
    //
    // The STRUCTURAL floors (sb128 / tiles / delta_q / delta_lf /
    // reduced_tx_set / disable_cdf_update / qm / seg / lossless / screen tools
    // / subsampling / bit depth / still-picture header) are forced by the
    // cells' encoder controls, so they equal the cell count that requests them
    // — a drop means a cell was removed or its control stopped landing. The
    // SEARCH-DEPENDENT floors (cdef_live / lr_live / deblock_*) depend on what
    // the encoder's rate-distortion search picks on this content, so they sit
    // a couple below the values MEASURED 2026-07-30 on libaom v3.14.1
    // (recorded in the run line above and in
    // `docs/DECODER_CONFIG_COVERAGE_2026-07-30.md`).
    for (k, min) in [
        ("sb64", 20u32),
        ("sb128", 11),
        ("multi_tile", 13),
        ("tiles_both_axes", 9),
        ("delta_q", 6),
        ("delta_lf", 6),
        ("delta_lf_with_live_deblock", 2),
        ("reduced_tx_set", 4),
        ("disable_cdf_update", 2),
        ("qm", 7),
        ("qm_x_seg", 2),
        ("seg", 4),
        ("seg_x_tiles", 2),
        ("lossless", 3),
        ("screen_tools", 2),
        ("cdef_live", 20),
        ("lr_live", 8),
        ("deblock_luma", 10),
        ("deblock_chroma", 7),
        ("monochrome", 4),
        ("yuv420", 13),
        ("yuv422", 8),
        ("yuv444", 6),
        ("bd8", 20),
        ("bd10", 8),
        ("bd12", 3),
        ("reduced_still_picture_hdr", 40),
        // GROUP SR — one witness per superres crossing this file exists to
        // close (hole H14). All are STRUCTURAL (forced by the cell's controls)
        // except `superres_x_lr` / `superres_x_sb128_x_lr`, which depend on the
        // encoder's restoration search and therefore sit below the value
        // MEASURED 2026-07-30 on libaom v3.14.1.
        ("superres", 9),
        ("superres_x_sb128", 6),
        ("superres_x_tiles", 1),
        ("superres_x_lr", 4),
        ("superres_x_sb128_x_lr", 2),
        ("superres_x_qm", 2),
        ("superres_x_seg", 1),
        ("superres_x_deltalf", 1),
        ("superres_x_nocdf", 1),
        ("superres_x_reduced_tx_set", 1),
        ("superres_x_422", 2),
        ("superres_x_bd12", 1),
        ("superres_x_mono", 1),
        ("superres_d16", 3),
    ] {
        let got = seen.get(k).copied().unwrap_or(0);
        assert!(
            got >= min,
            "axis '{k}' witnessed only {got} time(s), floor {min} — the config-permutation \
             grid has thinned out (witnesses: {seen:?})"
        );
    }
}
