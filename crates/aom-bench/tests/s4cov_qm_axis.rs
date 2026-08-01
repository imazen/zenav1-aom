//! **S4 coverage extension — the QM x `--cpu-used` axis beyond bd8 4:2:0 cq32.**
//!
//! KB-21 root #3 closed the quant-matrix drop inside `skip_trellis_opt_based_on_satd`
//! (`upstream/av1/encoder/tx_search.c:1981-2008` calling `av1_setup_quant`,
//! `encodemb.c:353-368`, whose last two statements NULL `qparam->qmatrix` /
//! `iqmatrix`). Its own closing note stated the residual plainly:
//!
//! > *"only 4:2:0 bd8 cq32 cells were run. bd10/bd12 QM at speed >= 4,
//! > monochrome, 4:4:4/4:2:2, and the qindex extremes (cq5 / cq63, which move
//! > the derived QM level) are unmeasured on this axis"*
//!
//! This file measures exactly that residual. Every cell here reaches a
//! `(bit depth, subsampling, qindex)` triple that `kb21_qm_speed4.rs` does not,
//! and each one is run in BOTH QM states at every RD speed 0..7:
//!
//! * **QM-ON is the subject** — the band where C quantizes (and, through
//!   `dist_block_tx_domain`, measures) with NO matrix while the port must drop
//!   the same one.
//! * **QM-OFF is the CONTROL** at the identical cell. Playbook §1: a divergence
//!   that is unchanged by turning the feature under test off is not that
//!   feature's — that is precisely how KB-23 was separated from KB-22's
//!   loop-restoration prediction. Without the control a QM-ON mismatch on a
//!   never-before-encoded cell shape could not be attributed to the QM axis at
//!   all.
//!
//! **Why the qindex extremes matter mechanically, not just as "more cells":**
//! `aom_get_qmlevel_allintra` maps `base_qindex` onto the `[qm_min, qm_max]`
//! range, so cq5 / cq32 / cq63 select DIFFERENT quant matrices — and at bd10 /
//! bd12 the same `--cq-level` maps to a different `base_qindex` again
//! (`av1_quantizer_to_qindex`, `av1_quantize.c:1033`, then the bit-depth
//! dequant tables). The derived levels are printed per cell and cross-checked
//! against the real header inside `port_encode_with`, so a wiring error fails
//! before any byte comparison.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test s4cov_qm_axis -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// Same QM range as the landed speed-0 gate (`encoder_gate_qm_on_e2e`) and
/// `kb21_qm_speed4`: `--qm-min=4 --qm-max=10`, which DERIVES a per-qindex level
/// rather than pinning one — the reason cq5/cq32/cq63 are three different
/// matrices and not three spellings of one.
const QM: (i32, i32) = (4, 10);

/// Every `--cpu-used`. 0..7 are the RD-search path (where KB-21 root #3's arm
/// lives); **8/9 are the nonrd PICKMODE path, which never enters
/// `search_tx_type` at all** — and are swept anyway, because "QM x nonrd" is
/// itself an axis nothing has measured and the cells cost milliseconds there.
const ALL_SPEEDS: [i32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

// ---------------------------------------------------------------------------
// Cell derivations.
//
// The intra conformance corpus is 4:2:0-only and bd8/bd10-only (235/235 carry
// 4:2:0 flags, 169 bd8 + 66 bd10 — `benchmarks/decoder_corpus_feature_tuples
// _2026-07-30.tsv`), so 4:4:4, 4:2:2 and bd12 cannot be READ from it. They are
// derived from real decoded photographic pixels instead of being synthesised
// from a formula, so the RD near-ties keep real-content statistics.
// ---------------------------------------------------------------------------

/// Drop the chroma planes (`mono = true`). The luma pixels are untouched, so a
/// mono cell isolates the luma quantizer path at the same content as its 4:2:0
/// sibling.
fn to_mono(base: &EncodeCell, label: &str) -> EncodeCell {
    EncodeCell {
        label: label.to_string(),
        mono: true,
        ss_x: 1,
        ss_y: 1,
        u: Vec::new(),
        v: Vec::new(),
        ..base.clone()
    }
}

/// Re-render a 4:2:0 cell at `(ss_x, ss_y)` by nearest-neighbour chroma
/// UPsampling. 4:4:4 (`0,0`) replicates each chroma sample 2x2; 4:2:2 (`1,0`)
/// replicates it vertically. Nearest-neighbour is deliberate: it neither
/// invents nor smooths chroma detail, so the chroma plane stays exactly as
/// informative as the source's, just carried at a different sampling grid.
fn to_ss(base: &EncodeCell, label: &str, ss_x: usize, ss_y: usize) -> EncodeCell {
    assert!(!base.mono, "{label}: source must carry chroma");
    assert_eq!((base.ss_x, base.ss_y), (1, 1), "{label}: source must be 4:2:0");
    let (bcw, _bch) = ((base.w + 1) >> 1, (base.h + 1) >> 1);
    let (cw, ch) = ((base.w + ss_x) >> ss_x, (base.h + ss_y) >> ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        // Map the target chroma row/col back onto the 4:2:0 source grid.
        let sr = (r << ss_y) >> 1;
        for col in 0..cw {
            let sc = (col << ss_x) >> 1;
            u[r * cw + col] = base.u[sr * bcw + sc];
            v[r * cw + col] = base.v[sr * bcw + sc];
        }
    }
    EncodeCell {
        label: label.to_string(),
        ss_x,
        ss_y,
        u,
        v,
        ..base.clone()
    }
}

/// Re-render a cell at a higher bit depth by BIT REPLICATION, not by a plain
/// left shift: `v << k | v >> (bits - k)`. A plain shift leaves the low `k`
/// bits zero, which is the "byte-identical samples encode at bd10/bd12"
/// regime KB-4 calls out as the EASY one — it never produces a coefficient
/// whose low bits matter. Replication fills the whole dynamic range, which is
/// the regime KB-4's mono full-range sweep had to reproduce before the bd10 /
/// bd12 divergence appeared at all.
fn to_bd(base: &EncodeCell, label: &str, bd: u8) -> EncodeCell {
    assert!(bd > base.bd, "{label}: to_bd only widens");
    let k = u32::from(bd - base.bd);
    let src_bits = u32::from(base.bd);
    let widen = |v: &u16| -> u16 { (v << k) | (v >> (src_bits - k)) };
    EncodeCell {
        label: label.to_string(),
        bd,
        y: base.y.iter().map(widen).collect(),
        u: base.u.iter().map(widen).collect(),
        v: base.v.iter().map(widen).collect(),
        ..base.clone()
    }
}

fn with_cq(base: &EncodeCell, label: &str, cq: i32) -> EncodeCell {
    EncodeCell {
        label: label.to_string(),
        cq_level: cq,
        ..base.clone()
    }
}

fn with_speed(base: &EncodeCell, speed: i32) -> EncodeCell {
    EncodeCell {
        speed,
        ..base.clone()
    }
}

/// Every cell on this axis, each carrying a `(bd, subsampling, cq)` triple that
/// `kb21_qm_speed4.rs` (bd8 / 4:2:0 / cq32 only) does not reach.
fn cells() -> Vec<EncodeCell> {
    c::ref_init();
    // Two real photographic sources: bd8 4:2:0 and bd10 4:2:0. 64x64 crops keep
    // every cell a single SB-exact superblock, so nothing here is confounded
    // with the frame-edge axis (that is `s4cov_partial_sb_axis.rs`).
    let b8 = EncodeCell::real_content("b8", "av1-1-b8-00-quantizer-00", Some((64, 64, 64, 64)), 32, 0);
    let b10 = EncodeCell::real_content("b10", "av1-1-b10-00-quantizer-00", Some((64, 64, 64, 64)), 32, 0);
    assert_eq!(b8.bd, 8, "bd8 source");
    assert_eq!(b10.bd, 10, "bd10 source");
    // The high-bit-depth MONO cells drop the chroma of the textured `b10`
    // crop rather than cropping `av1-1-b10-24-monochrome`. Measured
    // 2026-08-01: the (0,0) 64x64 corner of that vector is flat enough that
    // every speed codes an 18-22 byte frame — a byte-match there is a match on
    // an all-skip frame and proves nothing about the quantizer path. Dropping
    // `b10`'s chroma keeps genuine 10-bit luma texture (its 4:2:0 sibling
    // codes ~250 B at cq32) while isolating luma exactly the same way.
    let b10mono = to_mono(&b10, "b10mono");

    let mut out = vec![
        // --- qindex extremes at the ALREADY-COVERED (bd8, 4:2:0) shape. These
        // isolate "the derived QM LEVEL moved" from "the cell shape moved".
        with_cq(&b8, "bd8 420 cq5", 5),
        with_cq(&b8, "bd8 420 cq63", 63),
        // --- subsampling, at the covered bit depth.
        to_ss(&b8, "bd8 444 cq32", 0, 0),
        to_ss(&b8, "bd8 422 cq32", 1, 0),
        to_ss(&with_cq(&b8, "x", 5), "bd8 444 cq5", 0, 0),
        to_ss(&with_cq(&b8, "x", 63), "bd8 422 cq63", 1, 0),
        // --- monochrome.
        to_mono(&b8, "bd8 mono cq32"),
        to_mono(&with_cq(&b8, "x", 5), "bd8 mono cq5"),
        // --- bit depth: genuine 10-bit content.
        with_cq(&b10, "bd10 420 cq32", 32),
        with_cq(&b10, "bd10 420 cq5", 5),
        with_cq(&b10, "bd10 420 cq63", 63),
        to_ss(&b10, "bd10 444 cq32", 0, 0),
        with_cq(&b10mono, "bd10 mono cq32", 32),
        // --- bit depth: 12-bit, bit-replicated from the genuine 10-bit source.
        to_bd(&b10, "bd12 420 cq32", 12),
        to_bd(&with_cq(&b10, "x", 5), "bd12 420 cq5", 12),
        to_bd(&with_cq(&b10, "x", 63), "bd12 420 cq63", 12),
        to_bd(&b10mono, "bd12 mono cq32", 12),
    ];
    // A 444 bd12 cell — the corner furthest from anything previously measured.
    out.push(to_bd(&to_ss(&b10, "x", 0, 0), "bd12 444 cq32", 12));
    out
}

/// One `(cell, speed)` measurement: the QM-ON subject and its QM-OFF control.
struct Row {
    label: String,
    bd: u8,
    speed: i32,
    on_ok: bool,
    off_ok: bool,
    on_port: usize,
    on_real: usize,
    off_port: usize,
    off_real: usize,
    /// Anti-vacuity: `--enable-qm=1` must actually move the C reference stream.
    qm_changed_stream: bool,
}

fn measure(cell: &EncodeCell, speed: i32) -> Row {
    let cell = with_speed(cell, speed);
    // --- control: QM off, identical everything else.
    let c_off = cell.c_encode();
    assert!(!c_off.is_empty(), "{}: C encode failed", cell.label);
    let real_off = EncodeCell::frame_obu_payload(&c_off);
    let port_off = cell.port_encode(&c_off);

    // --- subject: QM on.
    let c_on = cell.c_encode_qm(QM.0, QM.1);
    assert!(!c_on.is_empty(), "{}: C QM encode failed", cell.label);
    let real_on = EncodeCell::frame_obu_payload(&c_on);
    let knobs = ToggleKnobs {
        qm: Some(QM),
        ..Default::default()
    };
    let port_on = cell.port_encode_with(&c_on, &knobs);

    Row {
        label: cell.label.clone(),
        bd: cell.bd,
        speed,
        on_ok: port_on == real_on,
        off_ok: port_off == real_off,
        on_port: port_on.len(),
        on_real: real_on.len(),
        off_port: port_off.len(),
        off_real: real_off.len(),
        qm_changed_stream: c_on != c_off,
    }
}

/// **The gate.** Three statements, in the order they fire — each one is what
/// the row *before* it makes interpretable.
///
/// 1. **Anti-vacuity.** `--enable-qm=1` must change the C reference stream on
///    every cell x speed. Without it a byte-match is compatible with the
///    harness silently never enabling QM.
///
/// 2. **bd8 is byte-exact across the whole extended axis, in BOTH QM states.**
///    `kb21_qm_speed4.rs` established bd8 4:2:0 cq32; this adds 4:4:4, 4:2:2,
///    monochrome and the cq5/cq63 qindex extremes (which move the derived QM
///    level) at every `--cpu-used` 0..9 — the nonrd speeds included.
///
/// 3. **QM never changes a verdict.** For EVERY row, QM-on and QM-off must
///    agree about whether the port byte-matches. This is the sharp form of the
///    QM statement and it is the only form the high-bit-depth cells can carry,
///    because their QM-OFF control diverges on its own (see the pin below) —
///    playbook §1: a divergence unchanged by turning the feature under test off
///    is not that feature's. Turning QM ON must not *add* one.
///
/// **MEASURED 2026-08-01** (`benchmarks/s4cov_axes_2026-08-01.tsv`): 180 rows,
/// **bd8 80/80 byte-exact in both QM states across every subsampling, both
/// qindex extremes and every `--cpu-used` 0..9**, and **verdict-invariance
/// 180/180 — QM never adds a divergence anywhere on the axis**. The QM-off
/// control divergences are confined to the `HBD_OPEN` set below, which is the
/// ALREADY-PINNED `b10_64` band of
/// `config_permutations.rs::speed_envelope_stock_map_is_pinned` (bd10 4:2:0
/// cq32, speeds 1..6) — reproduced here and shown to be **wider than that pin
/// records**: it is not 4:2:0-specific (4:4:4 diverges identically), not
/// bd10-specific (bd12 too), not chroma-borne (MONOCHROME diverges identically,
/// which puts the root on the LUMA path), and its speed reach is
/// qindex-dependent (cq5 reaches 1..6 like cq32; cq63 only reaches cpu6).
/// Speeds 0, 7, 8 and 9 are clean at every bit depth.
#[test]
#[ignore = "18 cells x 10 speeds x 2 QM states = 360 real-aomenc + 360 port encodes; run explicitly"]
fn qm_axis_bitdepth_subsampling_qindex_byte_matches() {
    let cells = cells();
    let mut rows: Vec<Row> = Vec::new();
    for cell in &cells {
        for &speed in &ALL_SPEEDS {
            rows.push(measure(cell, speed));
        }
    }

    for r in &rows {
        println!(
            "  {:<18} cpu{}  QM-on {:>5} B/{:>5} B {}  |  QM-off(control) {:>5} B/{:>5} B {}",
            r.label,
            r.speed,
            r.on_port,
            r.on_real,
            if r.on_ok { "MATCH   " } else { "MISMATCH" },
            r.off_port,
            r.off_real,
            if r.off_ok { "MATCH   " } else { "MISMATCH" },
        );
    }
    let n = rows.len();
    let bd8: Vec<&Row> = rows.iter().filter(|r| r.bd == 8).collect();
    println!(
        "  s4cov QM axis: {}/{n} QM-ON byte-exact, {}/{n} QM-OFF controls byte-exact | \
         bd8 subset {}/{} QM-ON, {}/{} QM-OFF | verdict-invariant rows {}/{n}",
        rows.iter().filter(|r| r.on_ok).count(),
        rows.iter().filter(|r| r.off_ok).count(),
        bd8.iter().filter(|r| r.on_ok).count(),
        bd8.len(),
        bd8.iter().filter(|r| r.off_ok).count(),
        bd8.len(),
        rows.iter().filter(|r| r.on_ok == r.off_ok).count(),
    );

    // 1. anti-vacuity.
    let inert: Vec<String> = rows
        .iter()
        .filter(|r| !r.qm_changed_stream)
        .map(|r| format!("{} cpu{}", r.label, r.speed))
        .collect();
    assert!(
        inert.is_empty(),
        "QM-on did not change the C reference stream, so these rows prove nothing about \
         the port's QM path: {inert:?}"
    );

    // 2. bd8, both QM states, every speed.
    let bd8_bad: Vec<String> = bd8
        .iter()
        .filter(|r| !r.on_ok || !r.off_ok)
        .map(|r| {
            format!(
                "{} cpu{} (QM-on {}, QM-off {})",
                r.label,
                r.speed,
                if r.on_ok { "ok" } else { "DIVERGE" },
                if r.off_ok { "ok" } else { "DIVERGE" }
            )
        })
        .collect();
    assert!(
        bd8_bad.is_empty(),
        "a bd8 cell diverged. The bd8 envelope is byte-exact at every `--cpu-used` 0..9 on \
         4:2:0 cq32 (`kb21_qm_speed4`, `speed_envelope_stock_map_is_pinned`); these rows \
         extend that to 4:4:4 / 4:2:2 / monochrome and to the cq5 / cq63 qindex extremes, \
         so a failure here is a genuinely new bd8 hole: {bd8_bad:?}"
    );

    // 3. QM never changes a verdict — the sharp QM statement, and the only one
    //    the high-bit-depth cells can carry.
    let flipped: Vec<String> = rows
        .iter()
        .filter(|r| r.on_ok != r.off_ok)
        .map(|r| {
            format!(
                "{} cpu{} (QM-on {}, QM-off {})",
                r.label,
                r.speed,
                if r.on_ok { "ok" } else { "DIVERGE" },
                if r.off_ok { "ok" } else { "DIVERGE" }
            )
        })
        .collect();
    assert!(
        flipped.is_empty(),
        "turning QM ON changed whether the port byte-matches. That is the KB-21 root #3 \
         shape (`av1_setup_quant`'s qmatrix NULLing inside `skip_trellis_opt_based_on_satd`, \
         tx_search.c:2001-2006) on a (bit depth, subsampling, qindex) triple it was never \
         measured at — or, in the other direction, a QM-only fix that closed a control \
         divergence and must be re-pinned: {flipped:?}"
    );

    // 4. The high-bit-depth control divergences, pinned in BOTH directions.
    //    This set is NOT this file's to close — it is the pre-existing `b10_64`
    //    band of `speed_envelope_stock_map_is_pinned` (bd10 4:2:0 cq32, speeds
    //    1..6). What is new here is its SHAPE: it is not 4:2:0-specific (bd10
    //    4:4:4 diverges identically), not bd10-specific (bd12 does too), and
    //    its speed reach depends on qindex (cq5 diverges at 1..6 like cq32,
    //    while cq63 only reaches it at cpu6).
    const HBD_OPEN: &[(&str, i32)] = &[
        ("bd10 420 cq32", 1), ("bd10 420 cq32", 2), ("bd10 420 cq32", 3),
        ("bd10 420 cq32", 4), ("bd10 420 cq32", 5), ("bd10 420 cq32", 6),
        ("bd10 420 cq5", 1), ("bd10 420 cq5", 2), ("bd10 420 cq5", 3),
        ("bd10 420 cq5", 4), ("bd10 420 cq5", 5), ("bd10 420 cq5", 6),
        ("bd10 420 cq63", 6),
        ("bd10 444 cq32", 1), ("bd10 444 cq32", 2), ("bd10 444 cq32", 3),
        ("bd10 444 cq32", 4), ("bd10 444 cq32", 5), ("bd10 444 cq32", 6),
        // MONOCHROME diverges identically -> the root is on the LUMA path.
        ("bd10 mono cq32", 1), ("bd10 mono cq32", 2), ("bd10 mono cq32", 3),
        ("bd10 mono cq32", 4), ("bd10 mono cq32", 5), ("bd10 mono cq32", 6),
        ("bd12 420 cq32", 1), ("bd12 420 cq32", 2), ("bd12 420 cq32", 3),
        ("bd12 420 cq32", 4), ("bd12 420 cq32", 5), ("bd12 420 cq32", 6),
        ("bd12 420 cq5", 1), ("bd12 420 cq5", 2), ("bd12 420 cq5", 3),
        ("bd12 420 cq5", 4), ("bd12 420 cq5", 5), ("bd12 420 cq5", 6),
        ("bd12 420 cq63", 6),
        ("bd12 mono cq32", 1), ("bd12 mono cq32", 2), ("bd12 mono cq32", 3),
        ("bd12 mono cq32", 4), ("bd12 mono cq32", 5), ("bd12 mono cq32", 6),
        ("bd12 444 cq32", 1), ("bd12 444 cq32", 2), ("bd12 444 cq32", 3),
        ("bd12 444 cq32", 4), ("bd12 444 cq32", 5), ("bd12 444 cq32", 6),
    ];
    let observed: Vec<(String, i32)> = rows
        .iter()
        .filter(|r| !r.off_ok)
        .map(|r| (r.label.clone(), r.speed))
        .collect();
    let pinned: Vec<(String, i32)> = HBD_OPEN.iter().map(|(l, s)| ((*l).to_string(), *s)).collect();
    assert_eq!(
        observed, pinned,
        "the high-bit-depth open set moved. A row that started MATCHING means the bd10/bd12 \
         speed-1..6 root closed — re-pin it here and in \
         `config_permutations.rs::speed_envelope_stock_map_is_pinned`'s `b10_64` row. A row \
         that started DIVERGING is a regression."
    );

    // 5. Non-vacuity of the cells themselves: a 20-byte all-skip frame
    //    byte-matches trivially. Every cell must code a frame with real content
    //    at its LOWEST-qindex speed-0 point.
    let trivial: Vec<String> = cells
        .iter()
        .filter(|c| c.cq_level <= 32)
        .filter_map(|c| {
            let r = rows
                .iter()
                .find(|r| r.label == c.label && r.speed == 0)
                .unwrap();
            (r.off_real < 100).then(|| format!("{} ({} B at cpu0)", c.label, r.off_real))
        })
        .collect();
    assert!(
        trivial.is_empty(),
        "a cq<=32 cell codes an almost-empty frame, so byte-matching it says nothing about \
         the quantizer path — pick textured content for it: {trivial:?}"
    );
}

/// **Reach assertion for the cell set** (playbook §8: derive coverage from
/// artefacts, not from names). Cheap — no encoding — so it runs by default and
/// keeps the expensive gate's claim honest even when that gate is not run.
///
/// Asserts that the derived cells genuinely cover the triples the KB-21 residual
/// named, i.e. that this file is not eighteen spellings of `bd8 4:2:0 cq32`.
#[test]
fn qm_axis_cells_reach_the_named_residual() {
    let cells = cells();
    let has = |f: &dyn Fn(&EncodeCell) -> bool| cells.iter().any(|c| f(c));
    assert!(has(&|c| c.bd == 10), "bd10 must be covered");
    assert!(has(&|c| c.bd == 12), "bd12 must be covered");
    assert!(has(&|c| c.mono), "monochrome must be covered");
    assert!(
        has(&|c| !c.mono && (c.ss_x, c.ss_y) == (0, 0)),
        "4:4:4 must be covered"
    );
    assert!(
        has(&|c| !c.mono && (c.ss_x, c.ss_y) == (1, 0)),
        "4:2:2 must be covered"
    );
    assert!(has(&|c| c.cq_level == 5), "the cq5 extreme must be covered");
    assert!(has(&|c| c.cq_level == 63), "the cq63 extreme must be covered");
    // Crossings, not just marginals: a bd12 non-420 cell and a bd12 qindex
    // extreme both exist, so the axis is not covered one factor at a time.
    assert!(
        has(&|c| c.bd == 12 && ((c.ss_x, c.ss_y) == (0, 0) || c.mono)),
        "a bd12 x (444 or mono) crossing must be covered"
    );
    assert!(
        has(&|c| c.bd == 12 && (c.cq_level == 5 || c.cq_level == 63)),
        "a bd12 x qindex-extreme crossing must be covered"
    );
    // And the content must actually USE the extra bit depth — a bd12 cell whose
    // samples all fit in 8 bits is the easy regime KB-4 warns about.
    let bd12 = cells.iter().find(|c| c.bd == 12 && !c.mono).unwrap();
    assert!(
        bd12.y.iter().any(|&s| s > 255) && bd12.y.iter().any(|&s| s & 0xF != 0),
        "the bd12 cells must carry FULL-dynamic-range content (samples above 255 and with \
         nonzero low bits) — a plain left shift would leave the low bits zero, which is the \
         representable-sample regime KB-4 records as the easy one"
    );
    assert!(
        cells.iter().filter(|c| !c.mono).all(|c| !c.u.is_empty()),
        "every non-mono cell must carry chroma planes"
    );
    assert!(
        cells.iter().filter(|c| c.mono).all(|c| c.u.is_empty()),
        "every mono cell must carry NO chroma planes"
    );
}
