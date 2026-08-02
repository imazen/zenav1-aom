//! CONFIG-PERMUTATION GATE — the *combination* half of the encoder-config
//! coverage that `toggles_rd_close.rs` leaves open.
//!
//! `toggles_rd_close.rs` gates ~25 knob families one at a time, each on its own
//! grid; every one is byte-identical to real aomenc **alone**. Nothing gated
//! them **together**. This file closes that gap with the collapse engine in
//! [`aom_bench::config_perm`]:
//!
//! * the raw knob cross (14,155,776 points) is reduced by the C-forbidden
//!   exclusion and by *effective-config* collapse (distinct resolved encoder
//!   states) — see [`config_permutation_coverage_arithmetic`], which prints the
//!   full arithmetic and pins it;
//! * mechanism 1 (effective-config collapse) is **proved, not asserted**:
//!   [`effective_collapse_is_real`] re-encodes the knob rows the engine claims
//!   are equivalent and requires them byte-identical on BOTH sides;
//! * mechanism 2 (independence collapse) is **measured** by
//!   [`independence_evidence_sweep`] (`--ignored`, writes the evidence TSV);
//!   the measured result is baked into `config_perm::INDEPENDENT_PAIRS`;
//! * what survives is covered by a t-wise covering array replayed over several
//!   cell contexts, each cell asserting **byte-identity against the real C
//!   encoder** — the strongest contract available, and cheaper than the
//!   rd_close path (no decode + no zensim on the exact path).
//!
//! ## Tiers
//!
//! * **Default** (`cargo test -p zenav1-aom-bench --test config_permutations`):
//!   t=4 (every 4-way knob interaction, 187 rows) on the five cheap contexts
//!   — bd8 4:2:0 at cq32 and cq63, sub-superblock 32x32, 4:4:4 and 4:2:2 —
//!   and t=3 (63 rows) on the three expensive ones — bd10, monochrome, and the
//!   four-superblock 128x128 cell — plus the collapse proof, the pinned
//!   monochrome-vector finding, and the arithmetic. Target: under 120 s wall.
//! * **Content** (section 5, added 2026-07-30): 12 content probes over 5
//!   taxonomy classes derived from `estimate_screen_content`
//!   (encoder.c:2042) — a 468-cell singleton-axis sensitivity matrix, a t=4
//!   array on bd8 4:2:0 screen content and a t=3 array on bd8 monochrome
//!   screen content, plus the pinned `dtxo` combination verdict. See
//!   `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md` §"The CONTENT axis".
//! * **Deep** (`-- --ignored`): [`independence_evidence_sweep`] and
//!   [`content_axis_evidence_sweep`], both of which are
//!   offline EVIDENCE GENERATION — it writes `benchmarks/`, so it is opt-in
//!   rather than slow. Everything else, including the quality ladder, the
//!   exhaustive redundant-level proof and the known-open
//!   `--use-intra-dct-only` arm, runs in the default tier.
//!
//! ## What a cell proves — DERIVED vs REPLAYED axes (read this before quoting
//! any coverage number)
//!
//! **The port never authors a sequence header.** `write_sequence_header_obu`
//! has no call site in any encoder path; every encode parses a sequence header
//! out of a real aomenc bootstrap stream and emits an `OBU_FRAME` payload alone.
//! (Verified independently 2026-07-30; see `AxisKind` and
//! `docs/CONFIG_AXIS_INVENTORY_2026-07-30.md`.)
//!
//! Therefore the axes split three ways ([`AxisKind`], reported by
//! [`config_permutation_coverage_arithmetic`]):
//!
//! * **16 DERIVED** — pure search gates. A cell here is end-to-end evidence
//!   that the port handles the configuration itself.
//! * **2 BOOTSTRAP-SEQ** (`--enable-filter-intra`, `--enable-intra-edge-filter`)
//!   and **3 BOOTSTRAP-FRAME** (`--reduced-tx-type-set`,
//!   `--enable-tx-size-search`, `--cdf-update-mode`) — the knob also names a
//!   header bit the port reads from the bootstrap and asserts equal to the
//!   knob. A cell here proves *"the port behaves correctly GIVEN this bit"*,
//!   which is real but weaker than *"the port produces this configuration"*.
//!
//! The CELL CONTEXTS (bit depth, monochrome, chroma subsampling, frame size,
//! superblock size) are **entirely replayed** — all of them arrive from the
//! bootstrap sequence header. They are therefore not covering-array factors at
//! all; they are contexts the array is replayed under, and no count in this
//! file should be read as "the port can produce these formats".
//!
//! Not reachable from this matrix, and noted rather than forced:
//! `large_scale` (`aom_dsp::entropy::header`, live write at :1565 and read at
//! :3173) has no non-default coverage anywhere in the tree. It is a
//! large-scale-tile mode with no `ToggleKnobs` axis and no control in
//! `EncodeCell::c_encode_ctrls`, so this gate cannot reach it. Reaching it
//! needs a new knob + a C control pair, which is encoder-harness work outside
//! this gate's ownership.
//!
//! Run with the per-cell tables: `... -- --nocapture`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use aom_bench::config_perm::{
    self as cp, ALL_AXES, Axis, AxisKind, CellCtx, DEFAULT_ROW, Effective, N_AXES, Row,
};
use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

/// One cell context the covering array is replayed under: real conformance
/// content + a crop + a quality point. The knob-independent axes (bit depth,
/// chroma format, monochrome, frame size, qindex) live HERE — they are not
/// covering-array factors, they are the contexts the array runs in, and they
/// participate in the collapse because [`Effective::resolve`] is
/// context-dependent.
struct Ctx {
    tag: &'static str,
    vector: &'static str,
    crop: Option<(usize, usize, usize, usize)>,
    cq: i32,
    /// Chroma format transform applied to the decoded conformance content.
    /// The luma is always untouched real content; only the chroma layout
    /// changes, which is exactly the axis being covered.
    ///
    /// (The corpus has exactly one native monochrome vector,
    /// `av1-1-b10-24-monochrome`, and it carries an open port divergence of its
    /// own — see [`mono_vector_open_divergences_pinned`] — so the monochrome
    /// CONTEXT is derived from clean content instead. There is no native 4:4:4
    /// or 4:2:2 vector in the intra scope at all.)
    format: Fmt,
}

/// Chroma-format transform for a [`Ctx`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fmt {
    /// Leave the decoded content as it is (4:2:0 for every corpus vector here).
    Native,
    /// Drop both chroma planes -> 4:0:0.
    Mono,
    /// Nearest-neighbour upsample chroma to full resolution -> 4:4:4.
    C444,
    /// Nearest-neighbour upsample chroma vertically only -> 4:2:2.
    C422,
}

/// 64x64 real content, mid quality — the densest bd8 4:2:0 cell (every knob
/// bites). ~137 ms/cell.
const C_64_CQ32: Ctx = Ctx {
    tag: "64cq32",
    vector: "av1-1-b8-01-size-64x64",
    crop: None,
    cq: 32,
    format: Fmt::Native,
};
/// 64x64, aggressive quality — large partitions / large transforms survive
/// where they are pruned at cq32. ~53 ms/cell.
const C_64_CQ63: Ctx = Ctx {
    tag: "64cq63",
    vector: "av1-1-b8-01-size-64x64",
    crop: None,
    cq: 63,
    format: Fmt::Native,
};
/// 32x32 — smaller than one superblock, so the root partition is force-split
/// and no 64px block (hence no 64-point transform) can exist. ~58 ms/cell.
const C_32_CQ32: Ctx = Ctx {
    tag: "32cq32",
    vector: "av1-1-b8-01-size-32x32",
    crop: None,
    cq: 32,
    format: Fmt::Native,
};
/// bd10 4:2:0 — the highbd quantizer/transform paths. ~140 ms/cell.
const C_B10_CQ32: Ctx = Ctx {
    tag: "b10cq32",
    vector: "av1-1-b10-00-quantizer-00",
    crop: Some((64, 64, 64, 64)),
    cq: 32,
    format: Fmt::Native,
};
/// Monochrome (4:0:0) bd10 — no chroma planes, so the whole UV mode loop (and
/// the CFL knob with it) is structurally dead. ~112 ms/cell.
const C_MONO_CQ32: Ctx = Ctx {
    tag: "monocq32",
    vector: "av1-1-b10-00-quantizer-00",
    crop: Some((64, 64, 64, 64)),
    cq: 32,
    format: Fmt::Mono,
};
/// 128x128 — FOUR superblocks, dense real content. The only context where
/// per-superblock behaviour (cost-update cadence, partition context carry,
/// CDF adaptation across superblocks) can differ from the 1-SB cells. ~800 ms/cell.
const C_128_CQ32: Ctx = Ctx {
    tag: "128cq32",
    vector: "av1-1-b8-00-quantizer-00",
    crop: Some((128, 128, 64, 64)),
    cq: 32,
    format: Fmt::Native,
};

/// 4:4:4 bd8 — full-resolution chroma, so every chroma-side knob operates on
/// luma-sized transform blocks (a different `get_plane_block_size` regime from
/// 4:2:0). ~125 ms/cell.
const C_444_CQ32: Ctx = Ctx {
    tag: "444cq32",
    vector: "av1-1-b8-01-size-64x64",
    crop: None,
    cq: 32,
    format: Fmt::C444,
};
/// 4:2:2 bd8 — the asymmetric subsampling case (ss_x=1, ss_y=0), where several
/// partition/transform shape lookups take their third branch. ~92 ms/cell.
const C_422_CQ32: Ctx = Ctx {
    tag: "422cq32",
    vector: "av1-1-b8-01-size-64x64",
    crop: None,
    cq: 32,
    format: Fmt::C422,
};

impl Ctx {
    fn cell(&self, label: &str) -> EncodeCell {
        let cell = EncodeCell::real_content(label, self.vector, self.crop, self.cq, 0);
        match self.format {
            Fmt::Native => cell,
            Fmt::Mono => {
                let mut c = cell;
                c.mono = true;
                c.u.clear();
                c.v.clear();
                c
            }
            Fmt::C444 => resample_chroma(&cell, 0, 0),
            Fmt::C422 => resample_chroma(&cell, 1, 0),
        }
    }
}

/// Nearest-neighbour chroma resample of a decoded cell to a different
/// subsampling. Luma is untouched; the chroma planes are replicated, so the
/// content stays real and the only thing that changes is the format the
/// encoder sees.
fn resample_chroma(cell: &EncodeCell, ss_x: usize, ss_y: usize) -> EncodeCell {
    assert!(!cell.mono, "resample_chroma needs chroma planes");
    let (w, h) = (cell.w, cell.h);
    let (scw, sch) = ((w + cell.ss_x) >> cell.ss_x, (h + cell.ss_y) >> cell.ss_y);
    let (dcw, dch) = ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y);
    let map = |src: &[u16]| {
        let mut out = vec![0u16; dcw * dch];
        for r in 0..dch {
            for col in 0..dcw {
                let sr = ((r << ss_y) >> cell.ss_y).min(sch - 1);
                let sc = ((col << ss_x) >> cell.ss_x).min(scw - 1);
                out[r * dcw + col] = src[sr * scw + sc];
            }
        }
        out
    };
    let mut o = cell.clone();
    o.u = map(&cell.u);
    o.v = map(&cell.v);
    o.ss_x = ss_x;
    o.ss_y = ss_y;
    o
}

/// Build the [`CellCtx`] the collapse engine resolves against, from the real
/// decoded cell (so monochrome / geometry come from the CONTENT, not a guess).
fn cell_ctx(cell: &EncodeCell) -> CellCtx {
    CellCtx {
        w: cell.w,
        h: cell.h,
        mono: cell.mono,
        // The proven envelope is SB64 (`--sb-size=128` encode is unstarted;
        // HANDOFF-TOGGLES.md). `port_encode_with` reads the real seq bit, so a
        // future sb128 bootstrap would need this to follow it.
        sb_px: 64,
    }
}

// ---------------------------------------------------------------------------
// One cell
// ---------------------------------------------------------------------------

/// The verdict for one (context, knob-row) cell.
struct Cell {
    label: String,
    /// The port's frame OBU payload equals real aomenc's, byte for byte.
    exact: bool,
    /// The C encoder's frame payload differs from its own stock (all-default)
    /// encode — i.e. this knob row genuinely reaches the C encoder.
    c_moved: bool,
    port_len: usize,
    c_len: usize,
}

/// Encode one cell on both sides and compare the FRAME OBU payloads.
///
/// The frame OBU payload is the unit the port produces and the byte gates
/// compare (`EncodeCell::frame_obu_payload`); the sequence header comes from
/// the C stream on both sides, so a sequence-only difference (e.g. the
/// `enable_intra_edge_filter` bit with no directional mode alive) is correctly
/// invisible here — that is exactly the equivalence `Effective::resolve`
/// models.
fn run_cell(cell: &EncodeCell, label: &str, knobs: &ToggleKnobs, c_stock: &[u8]) -> Cell {
    let ctrls = knobs.c_ctrls();
    let c_tu = cell.c_encode_ctrls(&ctrls);
    assert!(!c_tu.is_empty(), "{label}: C encode failed");
    let c_payload = EncodeCell::frame_obu_payload(&c_tu);
    let port = cell.port_encode_with(&c_tu, knobs);
    let exact = port == c_payload;
    if !exact {
        // A cell that is not byte-identical must at minimum produce a stream
        // the port decoder accepts, with the right geometry — never a silent
        // "encoded something, checked nothing".
        let tu = aom_bench::rd_close::splice_frame_obu(&c_tu, &port);
        let dec = aom_bench::rd_close::port_decode_tu(label, &tu);
        assert_eq!(
            (dec.width, dec.height, dec.monochrome),
            (cell.w, cell.h, cell.mono),
            "{label}: port stream decodes to the wrong geometry"
        );
    }
    Cell {
        label: label.to_string(),
        exact,
        c_moved: c_payload != c_stock,
        port_len: port.len(),
        c_len: c_payload.len(),
    }
}

/// The C encoder's stock (all-default-knob) frame payload for a cell — the
/// reference every row's `c_moved` witness is measured against. One encode per
/// context, not per row.
fn c_stock_payload(cell: &EncodeCell) -> Vec<u8> {
    EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&[]))
}

// ---------------------------------------------------------------------------
// The array runner
// ---------------------------------------------------------------------------

/// Run one shard of a t-wise covering array in one context.
///
/// Gate (all four must hold):
/// 1. **byte-identity** vs real aomenc on every cell whose row the ledger does
///    not list as open — the whole point;
/// 2. **anti-vacuity**: at least `min_moved_pct` of the non-stock rows must
///    change the C encoder's own output. An array of rows the C encoder ignores
///    would pass trivially;
/// 3. **collapse soundness**: any row the engine resolves to the SAME
///    [`Effective`] as the stock row must produce the stock C payload. A
///    violation means the signature is under-refined (it is missing state the
///    encoder actually steers on) — a bug in the engine, reported as such;
/// 4. **shard non-emptiness**: a mis-sharded run that tests nothing fails.
fn run_array(ctx: &Ctx, t: usize, shard: usize, n_shards: usize, min_moved_pct: f64) {
    c::ref_init();
    let cell = ctx.cell(&format!("cfgperm_{}", ctx.tag));
    let cctx = cell_ctx(&cell);
    let stock = c_stock_payload(&cell);
    let stock_eff = Effective::resolve(&DEFAULT_ROW, &cctx);

    let array = cp::covering_array(t);
    let collapsed = cp::collapse(&array, &cctx);
    let rows: Vec<Row> = collapsed
        .representatives
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % n_shards == shard)
        .map(|(_, r)| r)
        .collect();
    assert!(
        !rows.is_empty(),
        "{}: shard {shard}/{n_shards} is empty — the array shrank",
        ctx.tag
    );

    let mut cells = Vec::new();
    for row in &rows {
        let label = format!("{}_{}", ctx.tag, cp::row_label(row));
        let knobs = cp::knobs_of(row);
        let r = run_cell(&cell, &label, &knobs, &stock);
        if Effective::resolve(row, &cctx) == stock_eff {
            assert!(
                !r.c_moved,
                "{label}: the collapse engine resolves this row to the STOCK \
                 effective config, but the C encoder produced different bytes \
                 — the Effective signature is under-refined (it is missing \
                 state the encoder steers on)"
            );
        }
        cells.push(r);
    }

    // Mechanism-1 check IN SITU: every row the collapse DROPPED as a duplicate
    // must really produce the representative's bytes — on both sides. Without
    // this, an over-refined-away signature field would silently shrink the array.
    // (Only this shard's representatives are relevant.)
    let reps: BTreeSet<Row> = rows.iter().copied().collect();
    for (dup, rep) in &collapsed.duplicates {
        if !reps.contains(rep) {
            continue;
        }
        let dl = format!("{}_dup_{}", ctx.tag, cp::row_label(dup));
        let dup_c = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&cp::knobs_of(dup).c_ctrls()));
        let rep_c = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&cp::knobs_of(rep).c_ctrls()));
        assert_eq!(
            dup_c, rep_c,
            "{dl}: the collapse dropped this row as equivalent to {}, but real \
             aomenc produced different bytes — the Effective signature is \
             over-collapsing and the array is under-covering",
            cp::row_label(rep)
        );
        let dup_p = cell.port_encode_with(&cell.c_encode_ctrls(&cp::knobs_of(dup).c_ctrls()), &cp::knobs_of(dup));
        assert_eq!(
            dup_p, dup_c,
            "{dl}: collapsed-away row is not byte-identical to real aomenc"
        );
    }

    let non_stock: Vec<&Cell> = cells.iter().filter(|c| c.label != format!("{}_stock", ctx.tag)).collect();
    let moved = non_stock.iter().filter(|c| c.c_moved).count();
    let moved_pct = if non_stock.is_empty() {
        100.0
    } else {
        100.0 * moved as f64 / non_stock.len() as f64
    };

    println!("{}", render(&cells, ctx.tag, t, shard, n_shards, moved_pct));

    let open: Vec<&Cell> = cells.iter().filter(|c| !c.exact).collect();
    assert!(
        open.is_empty(),
        "{}: {} of {} covering-array cells are NOT byte-identical to real \
         aomenc — a knob COMBINATION diverges where every knob is exact alone. \
         Offenders: {}",
        ctx.tag,
        open.len(),
        cells.len(),
        open.iter()
            .map(|c| format!("{} (port {}B vs C {}B)", c.label, c.port_len, c.c_len))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(
        moved_pct >= min_moved_pct,
        "{}: only {moved_pct:.1}% of the {} non-stock rows changed the C \
         encoder's output (floor {min_moved_pct}%) — the array is drifting \
         toward vacuous cells; pick content/quality the knobs reach",
        ctx.tag,
        non_stock.len()
    );
}

fn render(
    cells: &[Cell],
    tag: &str,
    t: usize,
    shard: usize,
    n_shards: usize,
    moved_pct: f64,
) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n=== config-permutation gate: ctx={tag} t={t} shard={shard}/{n_shards} \
         cells={} exact={} C-moved={moved_pct:.1}%\n    axes: {} DERIVED (port \
         computes) + {} REPLAYED header bits (port agrees with the bootstrap, \
         cannot author them); the cell context itself is fully REPLAYED",
        cells.len(),
        cells.iter().filter(|c| c.exact).count(),
        ALL_AXES.iter().filter(|a| a.kind() == AxisKind::Derived).count(),
        ALL_AXES.iter().filter(|a| a.kind() != AxisKind::Derived).count()
    );
    for c in cells {
        let _ = writeln!(
            s,
            "  {:<7} {:<6} port={:>6}B c={:>6}B  {}",
            if c.exact { "EXACT" } else { "DIVERGE" },
            if c.c_moved { "moved" } else { "inert" },
            c.port_len,
            c.c_len,
            c.label
        );
    }
    s
}

// ---------------------------------------------------------------------------
// 1. Coverage arithmetic (no encoding)
// ---------------------------------------------------------------------------

/// The honest coverage arithmetic, printed and PINNED.
///
/// Every number here is computed, not quoted: the raw cartesian size, the
/// legal subset after the C-forbidden exclusion, the number of DISTINCT
/// resolved encoder states per context (the effective-config collapse), and
/// the covering-array sizes. The pins mean a silent shrink of the space or of
/// the array is a test failure.
#[test]
fn config_permutation_coverage_arithmetic() {
    let raw = cp::raw_space_size();
    assert_eq!(raw, 14_155_776, "the axis set changed — re-pin the arithmetic");

    // Exhaustive walk of the raw space: legality + distinct effective states.
    let ctxs = [
        ("64x64 4:2:0", CellCtx { w: 64, h: 64, mono: false, sb_px: 64 }),
        ("64x64 mono", CellCtx { w: 64, h: 64, mono: true, sb_px: 64 }),
        ("32x32 4:2:0", CellCtx { w: 32, h: 32, mono: false, sb_px: 64 }),
    ];
    let mut report = String::from("\n=== config-permutation coverage arithmetic ===\n");
    let _ = writeln!(report, "raw cartesian product of {N_AXES} axes : {raw}");

    // The DERIVED / REPLAYED split. A reader must never take the axis count as
    // a single homogeneous coverage number: five of the axes name header bits
    // the port cannot author, only agree with.
    let mut by_kind: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for ax in ALL_AXES {
        let k = match ax.kind() {
            AxisKind::Derived => "DERIVED (port computes it)",
            AxisKind::BootstrapSeq => "REPLAYED seq-header bit (port cannot author)",
            AxisKind::BootstrapFrame => "REPLAYED frame-header bit (port cannot author)",
        };
        by_kind.entry(k).or_default().push(ax.tag());
    }
    for (k, axes) in &by_kind {
        let _ = writeln!(report, "  {:>2} axes {k}: {}", axes.len(), axes.join(" "));
    }
    let _ = writeln!(
        report,
        "  cell CONTEXTS (bit depth, mono, subsampling, frame size, sb size) are \
         REPLAYED in full — they arrive from the bootstrap sequence header and are \
         not covering-array factors."
    );

    let mut legal_total = 0u64;
    let mut distinct: Vec<usize> = Vec::new();
    for (name, ctx) in &ctxs {
        let mut seen: BTreeSet<Effective> = BTreeSet::new();
        let mut legal = 0u64;
        walk_space(|row| {
            if cp::illegal_reason(row).is_none() {
                legal += 1;
                seen.insert(Effective::resolve(row, ctx));
            }
        });
        legal_total = legal;
        let _ = writeln!(
            report,
            "  ctx {name:<12}: legal {legal} -> {} distinct effective configs ({:.1}x collapse)",
            seen.len(),
            legal as f64 / seen.len() as f64
        );
        distinct.push(seen.len());
    }
    let _ = writeln!(
        report,
        "illegal-pair exclusion removed {} rows ({:.1}%): {}",
        raw - legal_total,
        100.0 * (raw - legal_total) as f64 / raw as f64,
        cp::illegal_reason(&{
            let mut r = DEFAULT_ROW;
            r[ALL_AXES.iter().position(|&a| a == Axis::TxSizeSearch).unwrap()] = 1;
            r[ALL_AXES.iter().position(|&a| a == Axis::Tx64).unwrap()] = 1;
            r
        })
        .unwrap()
    );

    for t in [2usize, 3, 4] {
        let rows = cp::covering_array(t);
        let mut per_ctx = Vec::new();
        for (name, ctx) in &ctxs {
            let col = cp::collapse(&rows, ctx);
            per_ctx.push(format!(
                "{name}: {} reps / {} dups",
                col.representatives.len(),
                col.duplicates.len()
            ));
        }
        let _ = writeln!(
            report,
            "t={t} covering array: {} rows  [{}]",
            rows.len(),
            per_ctx.join("; ")
        );
    }
    println!("{report}");

    assert_eq!(legal_total, 10_616_832, "legal-row count moved");
    assert_eq!(
        distinct,
        vec![777_600, 388_800, 622_080],
        "distinct effective-config counts moved — the resolution model changed; \
         re-derive and re-pin (and re-run the collapse proof)"
    );
    assert_eq!(
        ALL_AXES.iter().filter(|a| a.kind() == AxisKind::Derived).count(),
        16,
        "the DERIVED axis count moved — re-check AxisKind against the encoder \
         paths before re-pinning; a knob silently becoming bootstrap-carried \
         weakens every cell that covers it"
    );
    assert_eq!(
        ALL_AXES.iter().filter(|a| a.kind() == AxisKind::BootstrapSeq).count(),
        2
    );
    assert_eq!(
        ALL_AXES
            .iter()
            .filter(|a| a.kind() == AxisKind::BootstrapFrame)
            .count(),
        3
    );
    assert_eq!(cp::covering_array(2).len(), 17, "t=2 array size moved");
    assert_eq!(cp::covering_array(3).len(), 63, "t=3 array size moved");
    assert_eq!(cp::covering_array(4).len(), 187, "t=4 array size moved");
}

/// Visit every point of the raw cartesian space.
fn walk_space(mut f: impl FnMut(&Row)) {
    let mut row = DEFAULT_ROW;
    loop {
        f(&row);
        let mut i = N_AXES;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            row[i] += 1;
            if (row[i] as usize) < ALL_AXES[i].n_levels() {
                break;
            }
            row[i] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Mechanism 1 proof — effective-config collapse
// ---------------------------------------------------------------------------

fn ix(ax: Axis) -> usize {
    ALL_AXES.iter().position(|&a| a == ax).unwrap()
}

fn row_with(pairs: &[(Axis, u8)]) -> Row {
    let mut r = DEFAULT_ROW;
    for &(a, l) in pairs {
        r[ix(a)] = l;
    }
    r
}

/// Every equivalence the collapse engine predicts, as `(name, row_a, row_b)`
/// pairs that must produce IDENTICAL bytes on both sides in `ctx`.
fn predicted_equivalences(ctx: &CellCtx) -> Vec<(String, Row, Row)> {
    let mut out: Vec<(String, Row, Row)> = Vec::new();
    let mut push = |name: &str, a: Row, b: Row| {
        // Only claim it if the engine really does resolve them the same — this
        // list is a *reading* of the engine, never a parallel hardcoded truth.
        assert_eq!(
            Effective::resolve(&a, ctx),
            Effective::resolve(&b, ctx),
            "{name}: the engine does NOT predict this equivalence"
        );
        out.push((name.to_string(), a, b));
    };

    // Level equivalences that hold in EVERY context.
    push(
        "maxpart64_eq_128",
        DEFAULT_ROW,
        row_with(&[(Axis::MaxPart, 1)]),
    );
    push(
        "cdfupd2_eq_1",
        DEFAULT_ROW,
        row_with(&[(Axis::CdfUpdate, 2)]),
    );
    push(
        "trellis0_eq_3",
        DEFAULT_ROW,
        row_with(&[(Axis::Trellis, 3)]),
    );
    // ... and the same three on a NON-default background, so the equivalence is
    // shown to survive composition rather than only at the stock point.
    let bg = [
        (Axis::Smooth, 1u8),
        (Axis::RectTx, 1),
        (Axis::ReducedTxSet, 1),
        (Axis::MinPart, 2),
    ];
    let bg_row = row_with(&bg);
    for (name, ax, lv) in [
        ("maxpart64_eq_128_bg", Axis::MaxPart, 1u8),
        ("cdfupd2_eq_1_bg", Axis::CdfUpdate, 2),
        ("trellis0_eq_3_bg", Axis::Trellis, 3),
    ] {
        let mut b = bg_row;
        b[ix(ax)] = lv;
        push(name, bg_row, b);
    }

    // Transitive deaths: rect off kills the AB and 1to4 knobs.
    let rect_off = row_with(&[(Axis::Rect, 1)]);
    push(
        "rectoff_kills_ab",
        rect_off,
        row_with(&[(Axis::Rect, 1), (Axis::Ab, 1)]),
    );
    push(
        "rectoff_kills_1to4",
        rect_off,
        row_with(&[(Axis::Rect, 1), (Axis::P1to4, 1)]),
    );
    // Directional off kills diagonal, angle-delta and the intra edge filter.
    let dir_off = row_with(&[(Axis::Directional, 1)]);
    for (name, ax) in [
        ("diroff_kills_diagonal", Axis::Diagonal),
        ("diroff_kills_angledelta", Axis::AngleDelta),
        ("diroff_kills_edgefilter", Axis::EdgeFilter),
    ] {
        push(
            name,
            dir_off,
            row_with(&[(Axis::Directional, 1), (ax, 1)]),
        );
    }
    // dct-only / default-tx-only subsume the flip-idtx knob.
    push(
        "deftxonly_subsumes_flipidtx",
        row_with(&[(Axis::DefaultTxOnly, 1)]),
        row_with(&[(Axis::DefaultTxOnly, 1), (Axis::FlipIdtx, 1)]),
    );

    // Context-conditional deaths.
    if ctx.mono {
        push(
            "mono_kills_cfl",
            DEFAULT_ROW,
            row_with(&[(Axis::Cfl, 1)]),
        );
    }
    if ctx.w < ctx.sb_px || ctx.h < ctx.sb_px {
        push(
            "subsb_frame_kills_tx64",
            DEFAULT_ROW,
            row_with(&[(Axis::Tx64, 1)]),
        );
    }
    // A max-partition cap below 64 also removes every 64px block, so tx64 dies.
    push(
        "maxpart32_kills_tx64",
        row_with(&[(Axis::MaxPart, 2)]),
        row_with(&[(Axis::MaxPart, 2), (Axis::Tx64, 1)]),
    );
    out
}

/// Proof of mechanism 1: for every equivalence the engine predicts, encode BOTH
/// knob rows with the port AND with the real C encoder, and require all four
/// payloads identical.
///
/// This is what makes the collapse a claim rather than an assumption. Two ways
/// it can fail, both informative:
/// * the C payloads differ → the signature is **under-refined** (it declared
///   two genuinely different configurations equal, so the array is
///   under-covering);
/// * the C payloads agree but the PORT payloads differ → the port threads a
///   knob C ignores (or vice versa) — a real port defect.
fn run_collapse_proof(ctx: &Ctx) {
    c::ref_init();
    let cell = ctx.cell(&format!("collapse_{}", ctx.tag));
    let cctx = cell_ctx(&cell);
    let eqs = predicted_equivalences(&cctx);
    assert!(
        eqs.len() >= 12,
        "{}: only {} predicted equivalences — the proof set shrank",
        ctx.tag,
        eqs.len()
    );

    // Cache encodes: the same row shows up in several equivalences.
    let mut cache: BTreeMap<Row, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
    let enc = |row: &Row, cache: &mut BTreeMap<Row, (Vec<u8>, Vec<u8>)>| {
        if let Some(v) = cache.get(row) {
            return v.clone();
        }
        let knobs = cp::knobs_of(row);
        let c_tu = cell.c_encode_ctrls(&knobs.c_ctrls());
        let c_payload = EncodeCell::frame_obu_payload(&c_tu);
        let port = cell.port_encode_with(&c_tu, &knobs);
        cache.insert(*row, (c_payload.clone(), port.clone()));
        (c_payload, port)
    };

    // Encode everything and render the table FIRST, so a failure still shows
    // the whole picture instead of dying on the first row.
    let mut lines = format!(
        "\n=== effective-collapse proof: ctx={} ({} predicted equivalences)\n",
        ctx.tag,
        eqs.len()
    );
    let mut results = Vec::new();
    for (name, a, b) in &eqs {
        let (ca, pa) = enc(a, &mut cache);
        let (cb, pb) = enc(b, &mut cache);
        let _ = writeln!(
            lines,
            "  {:<28} {:<34} vs {:<34} c={} port={} base={}",
            name,
            cp::row_label(a),
            cp::row_label(b),
            if ca == cb { "same" } else { "DIFFER" },
            if pa == pb { "same" } else { "DIFFER" },
            if ca == pa { "exact" } else { "OPEN" }
        );
        results.push((name.clone(), *a, *b, ca, pa, cb, pb));
    }
    println!("{lines}");

    for (name, a, b, ca, pa, cb, pb) in &results {
        assert_eq!(
            ca, cb,
            "{}/{name}: the collapse engine calls these two knob rows \
             equivalent, but REAL AOMENC produced different bytes ({} vs {}) — \
             the Effective signature is under-refined and the covering array is \
             under-covering. Rows: {} vs {}",
            ctx.tag,
            ca.len(),
            cb.len(),
            cp::row_label(a),
            cp::row_label(b)
        );
        assert_eq!(
            pa, pb,
            "{}/{name}: real aomenc treats these two knob rows identically but \
             the PORT does not ({} vs {} bytes) — the port is steering on a \
             knob C ignores. Rows: {} vs {}",
            ctx.tag,
            pa.len(),
            pb.len(),
            cp::row_label(a),
            cp::row_label(b)
        );
        // The equivalence is only interesting if the port matches C at all —
        // otherwise "both sides moved together" could be two wrongs agreeing.
        assert_eq!(
            ca,
            pa,
            "{}/{name}: base row {} is not byte-identical to real aomenc, so \
             the equivalence it anchors proves nothing",
            ctx.tag,
            cp::row_label(a)
        );
    }
}

#[test]
fn effective_collapse_is_real_64cq32() {
    run_collapse_proof(&C_64_CQ32);
}

#[test]
fn effective_collapse_is_real_32cq32() {
    run_collapse_proof(&C_32_CQ32);
}

#[test]
fn effective_collapse_is_real_mono() {
    run_collapse_proof(&C_MONO_CQ32);
}

// ---------------------------------------------------------------------------
// 3. The covering-array gate (default tier)
// ---------------------------------------------------------------------------

/// Declare the sharded gate tests for one context at one strength.
macro_rules! gate {
    ($ctx:ident, t = $t:expr, shards = $n:expr, min_moved = $moved:expr,
     $($name:ident = $shard:expr),+ $(,)?) => {
        // A shard count that disagrees with the number of declared shard tests
        // would silently drop every n-th covering-array row. Caught at compile
        // time instead.
        const _: () = assert!(
            [$(stringify!($name)),+].len() == $n,
            "shards = N must equal the number of declared shard tests"
        );
        $(
            #[test]
            fn $name() {
                run_array(&$ctx, $t, $shard, $n, $moved);
            }
        )+
    };
}

// t=4 — EVERY 4-way knob interaction (187 rows) — on all eight contexts.
// Sharded so the libtest thread pool can spread them; each shard is a few
// seconds of CPU. `min_moved` is the anti-vacuity floor: the fraction of
// non-stock rows that must change the C encoder's own output. Measured 100%
// on every context (2026-07-30), so these floors have real headroom and still
// fail loudly if a context drifts toward configurations the encoder ignores.

gate!(C_64_CQ32, t = 4, shards = 3, min_moved = 90.0, combinations_t4_64cq32_s0 = 0, combinations_t4_64cq32_s1 = 1, combinations_t4_64cq32_s2 = 2);
gate!(C_64_CQ63, t = 4, shards = 2, min_moved = 70.0, combinations_t4_64cq63_s0 = 0, combinations_t4_64cq63_s1 = 1);
gate!(C_32_CQ32, t = 4, shards = 2, min_moved = 80.0, combinations_t4_32cq32_s0 = 0, combinations_t4_32cq32_s1 = 1);
gate!(C_444_CQ32, t = 4, shards = 3, min_moved = 90.0, combinations_t4_444cq32_s0 = 0, combinations_t4_444cq32_s1 = 1, combinations_t4_444cq32_s2 = 2);
gate!(C_422_CQ32, t = 4, shards = 2, min_moved = 90.0, combinations_t4_422cq32_s0 = 0, combinations_t4_422cq32_s1 = 1);
gate!(C_B10_CQ32, t = 4, shards = 3, min_moved = 90.0, combinations_t4_b10cq32_s0 = 0, combinations_t4_b10cq32_s1 = 1, combinations_t4_b10cq32_s2 = 2);
gate!(C_MONO_CQ32, t = 4, shards = 3, min_moved = 85.0, combinations_t4_mono_s0 = 0, combinations_t4_mono_s1 = 1, combinations_t4_mono_s2 = 2);
gate!(C_128_CQ32, t = 4, shards = 3, min_moved = 90.0, combinations_t4_128cq32_s0 = 0, combinations_t4_128cq32_s1 = 1, combinations_t4_128cq32_s2 = 2);

/// Every context used by the default tier, cheapest first.
const ALL_CONTEXTS: &[&Ctx] = &[
    &C_64_CQ63,
    &C_32_CQ32,
    &C_422_CQ32,
    &C_128_CQ32,
    &C_MONO_CQ32,
    &C_444_CQ32,
    &C_64_CQ32,
    &C_B10_CQ32,
];

/// PER-AXIS ANTI-VACUITY — the strong version.
///
/// `run_array`'s witness only shows that covering-array ROWS move the C
/// encoder, and a row flips many axes at once: an axis that never bites on any
/// context would hide inside rows that move for other reasons. This test pins
/// the per-AXIS claim: every axis level must, on its own, change real aomenc's
/// frame payload on at least one context in the default tier, and the witness
/// context is reported.
///
/// Measured 2026-07-30 (the raw per-axis liveness table is in the design doc):
/// `--enable-cfl-intra=0` is INERT on the 64x64 cq32 content (CFL never wins
/// there) and only bites on the multi-superblock and bd10 contexts;
/// `--enable-tx64=0` is INERT at cq32 (no 64-point transform is chosen) and
/// only bites at cq63. Without this test, both axes would have been covered
/// hundreds of times without ever being exercised.
///
/// Exempt: the three levels the collapse engine proves GLOBALLY inert (and
/// `effective_collapse_is_real` proves against real aomenc). Those must be
/// inert everywhere — asserted here in the opposite direction, so a level that
/// starts biting fails this test instead of silently invalidating the collapse.
#[test]
fn every_axis_level_is_live_in_some_context() {
    c::ref_init();
    const PROVEN_INERT: &[(Axis, u8)] = &[
        (Axis::MaxPart, 1),   // 64px == the 128px default at SB64
        (Axis::CdfUpdate, 2), // selective == always, on a lone KEY frame
        (Axis::Trellis, 3),   // FULL == NO_ESTIMATE_YRD on an intra frame
    ];
    // Lazily built per context: (cell, stock frame payload).
    let mut built: Vec<Option<(EncodeCell, Vec<u8>)>> = vec![None; ALL_CONTEXTS.len()];
    let mut dead: Vec<String> = Vec::new();
    let mut live_anyway: Vec<String> = Vec::new();
    let mut over_collapsed: Vec<String> = Vec::new();
    let mut report = String::from("\n=== per-axis liveness (default-tier contexts) ===\n");

    for (i, ax) in ALL_AXES.iter().enumerate() {
        for l in 1..ax.n_levels() as u8 {
            let mut row = DEFAULT_ROW;
            row[i] = l;
            if cp::illegal_reason(&row).is_some() {
                continue;
            }
            let knobs = cp::knobs_of(&row);
            let exempt = PROVEN_INERT.contains(&(*ax, l));
            let mut witness = None;
            for (ci, ctx) in ALL_CONTEXTS.iter().enumerate() {
                let entry = built[ci].get_or_insert_with(|| {
                    let cell = ctx.cell(&format!("live_{}", ctx.tag));
                    let stock = c_stock_payload(&cell);
                    (cell, stock)
                });
                let payload =
                    EncodeCell::frame_obu_payload(&entry.0.c_encode_ctrls(&knobs.c_ctrls()));
                let moved = payload != entry.1;
                // OVER-COLLAPSE DETECTOR. If the engine resolves this singleton
                // to the same Effective as the stock row, it is CLAIMING the
                // level cannot change anything here — so real aomenc must agree.
                // This is the sound direction of the implication (the converse
                // does not hold: two distinct states may coincidentally code the
                // same bytes), and it is what makes the collapse falsifiable for
                // every axis, not just the equivalences listed in
                // `predicted_equivalences`.
                let cctx = cell_ctx(&entry.0);
                if moved
                    && Effective::resolve(&row, &cctx) == Effective::resolve(&DEFAULT_ROW, &cctx)
                {
                    over_collapsed.push(format!(
                        "{}={} on ctx={}",
                        ax.tag(),
                        ax.values()[l as usize],
                        ctx.tag
                    ));
                }
                if moved {
                    witness = Some(ctx.tag);
                    // A proven-inert level must be inert on EVERY context, so it
                    // is never short-circuited; a live one can stop here.
                    if !exempt {
                        break;
                    }
                }
            }
            let name = format!("{}={}", ax.tag(), ax.values()[l as usize]);
            let _ = writeln!(
                report,
                "  {:<12} {}",
                name,
                match (witness, exempt) {
                    (Some(w), false) => format!("live, witness ctx={w}"),
                    (None, true) => "INERT everywhere (proven collapse — expected)".to_string(),
                    (Some(w), true) => format!("*** now LIVE on ctx={w} (was proven inert)"),
                    (None, false) => "*** DEAD on every context".to_string(),
                }
            );
            match (witness, exempt) {
                (None, false) => dead.push(name),
                (Some(w), true) => live_anyway.push(format!("{name} on {w}")),
                _ => {}
            }
        }
    }
    println!("{report}");
    assert!(
        dead.is_empty(),
        "these axis levels never change real aomenc's output on ANY default-tier \
         context, so every covering-array cell that sets them is vacuous for that \
         axis — add a context they reach, or document them as inert: {dead:?}"
    );
    assert!(
        over_collapsed.is_empty(),
        "the collapse engine resolves these axis levels to the STOCK effective \
         config — i.e. it claims they cannot change anything — but real aomenc \
         reacted to them. The Effective signature is OVER-COLLAPSING: every row \
         it folds away on that basis is coverage silently lost: {over_collapsed:?}"
    );
    assert!(
        live_anyway.is_empty(),
        "these levels are pinned as GLOBALLY INERT by the collapse engine but \
         real aomenc reacted to them — the effective-config collapse is unsound \
         and the covering array is under-covering: {live_anyway:?}"
    );
}

// ---------------------------------------------------------------------------
// 3b. FINDING (2026-07-30) — pinned open
// ---------------------------------------------------------------------------

/// FINDING, PARTIALLY CLOSED: two knobs that are byte-identical to real aomenc
/// on every gated context diverged on the corpus's native monochrome vector.
/// `--use-intra-default-tx-only=1` was KB-17 and is now FIXED; the single
/// `--enable-diagonal-intra=0` cell stays pinned open.
///
/// `av1-1-b10-24-monochrome`, 64x64 crop at (64,64), speed-0 ALLINTRA:
///
/// | knob | cq12 | cq20 | cq32 | cq48 | cq63 |
/// |---|---|---|---|---|---|
/// | `--use-intra-default-tx-only=1` (was) | DIVERGE 623/623 B | DIVERGE 418/424 | DIVERGE 229/240 | DIVERGE 79/80 | DIVERGE 14/15 |
/// | `--use-intra-default-tx-only=1` (KB-17 FIXED 2026-07-30) | exact 623 | — | exact 240 | — | exact 15 |
/// | `--enable-diagonal-intra=0`     | exact | exact | **DIVERGE 225/231** | exact | exact |
///
/// KB-17's root was `speed_features.rs`'s hardcoded `use_screen_content_tools:
/// false`: this vector is screen-detected, so C resolved the luma tx type
/// through `get_default_tx_type(..., cpi->use_screen_content_tools)` to
/// DCT_DCT (blockd.h:1183) while the port searched the mode-derived type.
/// Threading the parsed header's `allow_screen_content_tools` closed all three
/// `dtxo` cells here (measured 2026-07-30: 623/240/15 B, byte-identical).
///
/// **`diag=0` at cq32 did NOT move with that fix** — measured directly on the
/// same run, still 225 vs 231 B. That confirms the KB-17 entry's separate
/// classification: it is a KB-10 / KB-12 "cheaper RD decision" near-tie, not a
/// screen-content-tools consequence. Its bd8 twin does not reproduce it.
///
/// Isolated to the CONTENT, not to the format: the same knobs are byte-exact on
/// bd8 4:2:0, on bd8 monochrome derived from `av1-1-b8-01-size-64x64`, on bd10
/// 4:2:0, AND on bd10 monochrome derived from `av1-1-b10-00-quantizer-00` (a
/// full 27-knob singleton sweep over all six contexts reports 0 divergences).
///
/// The stock (all-default) encode of this content IS byte-exact, so the
/// envelope is unaffected.
///
/// This test pins the state exactly: it FAILS if a divergent cell starts
/// matching (fix landed → re-pin and consider promoting this content to a
/// context of the main array) and FAILS if a matching cell regresses — which
/// now includes every `dtxo` cell, i.e. it is the KB-17 regression gate for
/// this content.
#[test]
fn mono_vector_open_divergences_pinned() {
    c::ref_init();
    /// `(cq, knob tag, expected-exact)`.
    const EXPECTED: &[(i32, &str, bool)] = &[
        (12, "dtxo", true),
        (12, "diag", true),
        (32, "dtxo", true),
        (32, "diag", false),
        (63, "dtxo", true),
        (63, "diag", true),
    ];
    let mut measured = Vec::new();
    for &(cq, knob, _) in EXPECTED {
        let cell = EncodeCell::real_content(
            "monovec",
            "av1-1-b10-24-monochrome",
            Some((64, 64, 64, 64)),
            cq,
            0,
        );
        let knobs = match knob {
            "dtxo" => ToggleKnobs {
                use_intra_default_tx_only: true,
                ..Default::default()
            },
            "diag" => ToggleKnobs {
                enable_diagonal_intra: false,
                ..Default::default()
            },
            other => panic!("unknown knob {other}"),
        };
        let c_tu = cell.c_encode_ctrls(&knobs.c_ctrls());
        let c_payload = EncodeCell::frame_obu_payload(&c_tu);
        let port = cell.port_encode_with(&c_tu, &knobs);
        // The stock encode of this content must stay byte-exact — the finding
        // is about these two knobs, not about monochrome bd10 as such.
        let stock = c_stock_payload(&cell);
        let port_stock = cell.port_encode_with(&cell.c_encode_ctrls(&[]), &ToggleKnobs::default());
        assert_eq!(
            stock, port_stock,
            "monovec cq{cq}: the STOCK monochrome encode regressed — that is a \
             different (worse) bug than the pinned knob divergence"
        );
        println!(
            "  monovec cq{cq} {knob}: {} port={}B c={}B",
            if port == c_payload { "exact  " } else { "DIVERGE" },
            port.len(),
            c_payload.len()
        );
        measured.push((cq, knob, port == c_payload));
    }
    assert_eq!(
        measured,
        EXPECTED.to_vec(),
        "the av1-1-b10-24-monochrome divergence set MOVED (see this test's doc \
         comment for the pinned table). A cell flipping to exact means the \
         near-tie was fixed — re-pin. A cell flipping to divergent is a \
         regression."
    );
}

// ---------------------------------------------------------------------------
// 4. Quality ladder, exhaustive collapse proof, known-open knob, and the
//    offline independence-evidence generator (the only --ignored test)
// ---------------------------------------------------------------------------

/// MECHANISM 2 EVIDENCE — the 2x2 independence experiment, offline.
///
/// For each axis pair (A, B) this encodes the four corners {A0B0, A0B1, A1B0,
/// A1B1} with the REAL C encoder and measures two things per corner pair:
///
/// * **stream change** — did the coded frame payload move at all? An axis whose
///   payload never moves is INERT on this cell; that is effective-config
///   collapse territory, not independence, and it is reported separately so the
///   two can never be conflated.
/// * **footprint** — the set of 4x4 blocks (across ALL planes, each plane on its
///   own grid) whose reconstruction changes when the axis flips. Chroma is
///   included deliberately: a luma-only footprint would report every chroma-only
///   knob (`--enable-cfl-intra`) as having no effect.
///
/// A and B are called INDEPENDENT only when **both axes are live** (each moves
/// the stream under both settings of the other) **and** `footprint(A | B=0) ==
/// footprint(A | B=1)` and symmetrically for B — i.e. each axis lands on the
/// same part of the decision state whatever the other is doing.
///
/// Known blind spot, stated rather than hidden: the footprint is a
/// RECONSTRUCTION measure, so it cannot see a change that alters only the coded
/// symbols (`--cdf-update-mode=0` is the clean example — it moves the payload
/// while leaving the recon bit-identical). That is why the stream-change flag
/// gates the verdict: an axis that moves only the bitstream is classified
/// SIGNALLING-ONLY, never independent, and its pairs stay crossed.
///
/// The C encoder (not the port) is the subject: independence is a property of
/// libaom's configuration semantics, so using the oracle keeps the answer valid
/// even where the port might not thread a knob.
///
/// Writes `benchmarks/config_perm_independence_2026-07-30.tsv`; the summary is
/// transcribed into `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md` and the
/// verdict populates `config_perm::INDEPENDENT_PAIRS`.
#[test]
#[ignore = "offline evidence generation for INDEPENDENT_PAIRS; minutes, writes benchmarks/"]
fn independence_evidence_sweep() {
    c::ref_init();
    let ctx = &C_64_CQ32;
    let cell = ctx.cell("indep");
    // (frame payload, per-plane recon) per corner, cached across pairs.
    type Corner = (Vec<u8>, Vec<(Vec<u16>, usize)>);
    let mut cache: BTreeMap<Row, Corner> = BTreeMap::new();
    let encode = |row: Row, cache: &mut BTreeMap<Row, Corner>| -> Corner {
        if let Some(v) = cache.get(&row) {
            return v.clone();
        }
        let knobs = cp::knobs_of(&row);
        let tu = cell.c_encode_ctrls(&knobs.c_ctrls());
        let payload = EncodeCell::frame_obu_payload(&tu);
        let dec = aom_bench::rd_close::port_decode_tu("indep", &tu);
        let mut planes = vec![(dec.y.clone(), dec.width)];
        if !dec.monochrome {
            let cw = (dec.width + dec.subsampling_x) >> dec.subsampling_x;
            planes.push((dec.u.clone(), cw));
            planes.push((dec.v.clone(), cw));
        }
        let v = (payload, planes);
        cache.insert(row, v.clone());
        v
    };
    /// 4x4-block footprint over every plane (plane index folded into the key).
    fn footprint(a: &[(Vec<u16>, usize)], b: &[(Vec<u16>, usize)]) -> BTreeSet<(usize, usize)> {
        let mut s = BTreeSet::new();
        for (pi, ((pa, w), (pb, _))) in a.iter().zip(b.iter()).enumerate() {
            let bx = w.div_ceil(4);
            for (i, (x, y)) in pa.iter().zip(pb.iter()).enumerate() {
                if x != y {
                    s.insert((pi, (i / w / 4) * bx + (i % w) / 4));
                }
            }
        }
        s
    }

    let mut tsv = String::from(
        "axis_a\taxis_b\tverdict\ta_moves_stream_b0\ta_moves_stream_b1\t\
         b_moves_stream_a0\tb_moves_stream_a1\tfp_a_b0\tfp_a_b1\tfp_a_equal\t\
         fp_b_a0\tfp_b_a1\tfp_b_equal\n",
    );
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut independent: Vec<(Axis, Axis, usize, usize)> = Vec::new();
    for i in 0..N_AXES {
        for j in i + 1..N_AXES {
            let (ax, bx) = (ALL_AXES[i], ALL_AXES[j]);
            let corner = |la: u8, lb: u8| {
                let mut r = DEFAULT_ROW;
                r[i] = la;
                r[j] = lb;
                r
            };
            if cp::illegal_reason(&corner(1, 1)).is_some() {
                *counts.entry("ILLEGAL").or_default() += 1;
                let _ = writeln!(
                    tsv,
                    "{}\t{}\tILLEGAL\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-",
                    ax.tag(),
                    bx.tag()
                );
                continue;
            }
            let (s00, r00) = encode(corner(0, 0), &mut cache);
            let (s10, r10) = encode(corner(1, 0), &mut cache);
            let (s01, r01) = encode(corner(0, 1), &mut cache);
            let (s11, r11) = encode(corner(1, 1), &mut cache);
            let (a_b0, a_b1) = (s10 != s00, s11 != s01);
            let (b_a0, b_a1) = (s01 != s00, s11 != s10);
            let (fa0, fa1) = (footprint(&r00, &r10), footprint(&r01, &r11));
            let (fb0, fb1) = (footprint(&r00, &r01), footprint(&r10, &r11));
            let a_live = a_b0 && a_b1;
            let b_live = b_a0 && b_a1;
            let verdict = if !a_live && !b_live {
                "INERT-BOTH"
            } else if !a_live {
                "INERT-A"
            } else if !b_live {
                "INERT-B"
            } else if fa0.is_empty() && fa1.is_empty() {
                "SIGNALLING-ONLY-A"
            } else if fb0.is_empty() && fb1.is_empty() {
                "SIGNALLING-ONLY-B"
            } else if fa0 == fa1 && fb0 == fb1 {
                "INDEPENDENT"
            } else {
                "INTERACTING"
            };
            *counts.entry(verdict).or_default() += 1;
            if verdict == "INDEPENDENT" {
                independent.push((ax, bx, fa0.len(), fb0.len()));
            }
            let _ = writeln!(
                tsv,
                "{}\t{}\t{verdict}\t{a_b0}\t{a_b1}\t{b_a0}\t{b_a1}\t{}\t{}\t{}\t{}\t{}\t{}",
                ax.tag(),
                bx.tag(),
                fa0.len(),
                fa1.len(),
                fa0 == fa1,
                fb0.len(),
                fb1.len(),
                fb0 == fb1
            );
        }
    }
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/config_perm_independence_2026-07-30.tsv");
    std::fs::write(&out, format!("{}{tsv}", evidence_provenance()))
        .expect("write independence TSV");
    println!("\n=== independence sweep (ctx {} , C oracle) ===", ctx.tag);
    for (k, n) in &counts {
        println!("  {k:<18} {n}");
    }
    for (a, b, na, nb) in &independent {
        println!(
            "  INDEPENDENT {:<6} x {:<6} |fp A|={na} |fp B|={nb}",
            a.tag(),
            b.tag()
        );
    }
    println!("wrote {}", out.display());
    // Guard the experiment itself: a run where nothing is classified
    // INTERACTING would mean the footprint measure has stopped working.
    assert!(
        counts.get("INTERACTING").copied().unwrap_or(0) > 0,
        "no pair classified INTERACTING — the footprint measure is broken"
    );
}

/// EXHAUSTIVE proof that the level equivalences the collapse relies on hold at
/// EVERY background, not just the ones `effective_collapse_is_real` encodes.
///
/// Combinatorial (no encoding): a full walk of the space per claimed level.
/// The encode-level proof (`effective_collapse_is_real`) is the one that can
/// catch a wrong MODEL; this one catches a model that is right at the sampled
/// points and wrong elsewhere.
#[test]
fn redundant_levels_are_globally_redundant() {
    let ctx = CellCtx { w: 64, h: 64, mono: false, sb_px: 64 };
    // The three level equivalences the design claims are GLOBAL.
    let claims = [
        (Axis::MaxPart, 1u8, 0u8, "partition_strategy.h:214 min(sf,CLI,sb) at SB64"),
        (Axis::CdfUpdate, 2, 0, "encoder.c:4390 case 2 -> 0 on an intra-only frame"),
        (Axis::Trellis, 3, 0, "init_rd_sf: FULL vs NO_ESTIMATE_YRD differ only in inter-only estimate_yrd_for_sb"),
    ];
    for (ax, la, lb, why) in claims {
        let a = ix(ax);
        let mut checked = 0u64;
        walk_space(|row| {
            let mut ra = *row;
            let mut rb = *row;
            ra[a] = la;
            rb[a] = lb;
            if cp::illegal_reason(&ra).is_some() || cp::illegal_reason(&rb).is_some() {
                return;
            }
            checked += 1;
            assert_eq!(
                Effective::resolve(&ra, &ctx),
                Effective::resolve(&rb, &ctx),
                "{}: level {la} is NOT globally equivalent to level {lb} ({why}) \
                 at background {}",
                ax.tag(),
                cp::row_label(row)
            );
        });
        println!("{}: level {la} == level {lb} over {checked} backgrounds ({why})", ax.tag());
    }
}

/// THE QUALITY AXIS. The context set is built around cq32 (plus one cq63
/// context); this replays the t=3 array across the rest of the quality range on
/// the primary 64x64 content, where the RD balance — and therefore which knob a
/// combination actually reaches — changes most. 4 x 63 cells, a few seconds:
/// knob-narrowed configurations encode far faster than the stock search.
fn run_quality_ladder(qs: &[i32]) {
    for &cq in qs {
        let ctx = Ctx {
            tag: "ladder",
            vector: "av1-1-b8-01-size-64x64",
            crop: None,
            cq,
            format: Fmt::Native,
        };
        run_array(&ctx, 4, 0, 1, 70.0);
    }
}

#[test]
fn combinations_quality_ladder_high() {
    run_quality_ladder(&[5, 12]);
}
#[test]
fn combinations_quality_ladder_mid() {
    run_quality_ladder(&[20, 40]);
}
#[test]
fn combinations_quality_ladder_low() {
    run_quality_ladder(&[48, 55]);
}

/// `--use-intra-dct-only=1` is the one knob with a KNOWN open divergence
/// (`toggles_c9_intra_dct_only_pinned_open`: 64²cq32 out of band, a UV-loop
/// mis-model). It is therefore excluded from the byte-exact covering array.
///
/// This test pins its COMBINATION behaviour instead: a t=2 array over the other
/// axes, all with `--use-intra-dct-only=1`, recording which rows are exact.
/// It fails if a row that was exact stops being exact (a regression) OR if a
/// row that diverged starts matching (self-promoting: the UV-loop fix landed,
/// re-pin and move the knob into the main array).
#[test]
fn combinations_dct_only_verdict_set_pinned() {
    c::ref_init();
    let ctx = &C_64_CQ32;
    let cell = ctx.cell("dctonly");
    let stock = c_stock_payload(&cell);
    let mut diverged = BTreeSet::new();
    let mut n = 0usize;
    for row in cp::covering_array(2) {
        let mut knobs = cp::knobs_of(&row);
        knobs.use_intra_dct_only = true;
        let label = format!("dctonly_{}", cp::row_label(&row));
        let r = run_cell(&cell, &label, &knobs, &stock);
        n += 1;
        if !r.exact {
            diverged.insert(cp::row_label(&row));
        }
    }
    println!(
        "\n=== --use-intra-dct-only=1 x t=2 array: {}/{n} rows diverge\n{:#?}",
        diverged.len(),
        diverged
    );
    // Pinned at first measurement — see the design doc for the recorded set.
    let expected: BTreeSet<String> = DCT_ONLY_DIVERGENT_ROWS.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        diverged, expected,
        "the --use-intra-dct-only divergence set MOVED. Rows that started \
         matching mean the UV-loop mis-model (HANDOFF-TOGGLES.md) was fixed — \
         re-pin this set and promote the knob into the main covering array. \
         Rows that started diverging are a regression."
    );
}

/// The recorded divergence set for [`combinations_dct_only_verdict_set_pinned`]:
/// 11 of the 17 t=2 rows, measured 2026-07-30 on the 64x64 cq32 context.
/// `stock` (the knob alone, no other change) is in the set, which is the same
/// cell `toggles_c9_intra_dct_only_pinned_open` already pins — the combinations
/// inherit that one open divergence rather than adding new ones.
const DCT_ONLY_DIVERGENT_ROWS: &[&str] = &[
    "ab0-p140-maxp64-paeth0-cfl0-diag0-fint0-rtx0-txss0-cdf2-trel2",
    "minp16-maxp32-smth0-cfl0-diag0-rtx0-flip0-rtxs1-cdf2-trel0",
    "minp16-maxp64-cfl0-dir0-diag0-adlt0-edgf0-flip0-txss0-cdf0-trel1",
    "p140-minp8-smth0-paeth0-diag0-adlt0-fint0-flip0-dtxo1-txss0-cdf2",
    "rect0-ab0-diag0-fint0-edgf0-tx640-rtx0-flip0-dtxo1-rtxs1-trel1",
    "rect0-ab0-p140-minp16-paeth0-dir0-adlt0-fint0-edgf0-dtxo1-rtxs1-cdf0",
    "rect0-ab0-p140-minp8-paeth0-cfl0-dir0-diag0-rtx0-txss0-cdf2-trel1",
    "rect0-p140-minp16-maxp64-smth0-cfl0-diag0-adlt0-fint0-edgf0-rtx0-rtxs1-txss0-trel0",
    "rect0-p140-minp8-maxp32-adlt0-edgf0-rtx0-txss0-cdf0-trel2",
    "rect0-p140-minp8-maxp64-smth0-paeth0-adlt0-fint0-tx640-rtx0-dtxo1",
    "stock",
];

// ---------------------------------------------------------------------------
// 5. THE CONTENT AXIS (2026-07-30)
//
// See `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md` §"The content axis".
// Everything below is content generation + content contexts; the knob axes,
// the covering array and the collapse engine are untouched.
// ---------------------------------------------------------------------------

/// One CONTENT probe: real conformance content, a 64x64 crop, and the
/// taxonomy class it is *predicted* to fall in.
struct Content {
    tag: &'static str,
    vector: &'static str,
    crop: Option<(usize, usize, usize, usize)>,
    /// Predicted `allow_screen_content_tools` — the pinned half of the
    /// taxonomy. Computed by `cp::screen_stat` from the source pixels the
    /// encoder actually sees, i.e. AFTER the crop.
    screen: bool,
}

/// The content probe set. Every 64x64 crop, every one bd8/bd10 4:2:0 or 4:0:0
/// — the FORMAT is held as constant as the corpus allows so that anything that
/// moves is attributable to CONTENT.
const CONTENTS: &[Content] = &[
    // --- class NATURAL (screen detector does not fire) ---
    Content { tag: "nat_64x64", vector: "av1-1-b8-01-size-64x64", crop: None, screen: false },
    Content { tag: "nat_allintra", vector: "av1-1-b8-02-allintra", crop: Some((64, 64, 96, 96)), screen: false },
    Content { tag: "nat_cdfupd", vector: "av1-1-b8-04-cdfupdate", crop: Some((64, 64, 96, 96)), screen: false },
    Content { tag: "nat_b10", vector: "av1-1-b10-00-quantizer-00", crop: Some((64, 64, 64, 64)), screen: false },
    // --- class NOISE (film grain synthesised into the decoded source) ---
    Content { tag: "grain_b8", vector: "av1-1-b8-23-film_grain-50", crop: Some((64, 64, 96, 96)), screen: false },
    Content { tag: "grain_b10", vector: "av1-1-b10-23-film_grain-50", crop: Some((64, 64, 96, 96)), screen: false },
    // --- class DC-FLAT (decoded at a crushing qindex -> few colors per block) ---
    Content { tag: "flat_b8_q63", vector: "av1-1-b8-00-quantizer-63", crop: Some((64, 64, 96, 96)), screen: false },
    Content { tag: "flat_b10_q63", vector: "av1-1-b10-00-quantizer-63", crop: Some((64, 64, 64, 64)), screen: false },
    Content { tag: "det_b8_q00", vector: "av1-1-b8-00-quantizer-00", crop: Some((64, 64, 96, 96)), screen: false },
    // --- class SCREEN (the detector fires) ---
    Content { tag: "scr_mono_b8", vector: "av1-1-b8-24-monochrome", crop: Some((64, 64, 64, 64)), screen: true },
    Content { tag: "scr_mono_b10", vector: "av1-1-b10-24-monochrome", crop: Some((64, 64, 64, 64)), screen: true },
    Content { tag: "scr_ibc_b8", vector: "av1-1-b8-16-intra_only-intrabc-extreme-dv", crop: Some((64, 64, 64, 64)), screen: true },
];

impl Content {
    fn cell(&self, cq: i32) -> EncodeCell {
        EncodeCell::real_content(self.tag, self.vector, self.crop, cq, 0)
    }
}


/// The measured [`cp::screen_stat`] per content, PINNED:
/// `(tag, counts_1, blocks, allow_screen_content_tools)`. Measured 2026-07-30
/// on the exact 64x64 crops above.
const CONTENT_SCREEN_STATS: &[(&str, usize, usize, bool)] = &[
    ("nat_64x64", 0, 16, false),
    ("nat_allintra", 0, 16, false),
    ("nat_cdfupd", 0, 16, false),
    ("nat_b10", 0, 16, false),
    ("grain_b8", 0, 16, false),
    ("grain_b10", 0, 16, false),
    ("flat_b8_q63", 0, 16, false),
    ("flat_b10_q63", 0, 16, false),
    ("det_b8_q00", 0, 16, false),
    ("scr_mono_b8", 6, 16, true),
    ("scr_mono_b10", 4, 16, true),
    ("scr_ibc_b8", 8, 16, true),
];

/// THE CONTENT TAXONOMY, executable.
///
/// The taxonomy is built on the ONE content property the speed-0 ALLINTRA
/// encoder branches on that is derived from the source pixels:
/// `estimate_screen_content` (encoder.c:2042-2100) — see [`cp::screen_stat`]
/// for the transcription and the downstream branch citations.
///
/// Two claims are gated here:
///
/// 1. **the classification itself** — `counts_1 / blocks` and the resulting
///    verdict are pinned per content, so a content probe that silently drifts
///    out of its class (a changed crop, a changed vector) fails instead of
///    quietly weakening the matrix below;
/// 2. **the classifier is anchored to the real C encoder, in the SOUND
///    direction.** `allow_screen_content_tools == 0` makes `av1_allow_palette`
///    false for every block, so `--enable-palette=1` CANNOT change anything;
///    the two C encodes must be byte-identical. If a content this test calls
///    "not screen" reacts to `--enable-palette`, the classifier is wrong and
///    every conclusion drawn from it is void. (The converse — screen ⇒ the
///    payload must move — is empirical, not implied, so it is reported but
///    only asserted as a non-vacuity floor: at least one screen content must
///    react, else the anchor measures nothing.)
///
/// Cross-format control, also asserted: `scr_mono_b8` (bd8) and
/// `scr_mono_b10` (bd10) are the same source clip at two bit depths and must
/// land in the SAME class — `av1_count_colors_highbd` down-converts to the
/// 8-bit domain before binning (intra_mode_search.c:352-357), so the statistic
/// is bit-depth independent by construction. That is what makes this a
/// CONTENT axis rather than a format axis.
#[test]
fn content_taxonomy_is_measured_and_pinned() {
    c::ref_init();
    let mut measured = Vec::new();
    let mut report = String::from("\n=== content taxonomy (estimate_screen_content) ===\n");
    let mut screen_reacted = 0usize;
    for ct in CONTENTS {
        let cell = ct.cell(32);
        let st = cp::screen_stat(&cell.y, cell.w, cell.h, cell.bd);
        // Every content this file introduces carries a byte-identity assert.
        let c_tu = cell.c_encode_ctrls(&[]);
        assert_eq!(
            EncodeCell::frame_obu_payload(&c_tu),
            cell.port_encode_with(&c_tu, &ToggleKnobs::default()),
            "{}: the stock encode of this content probe is not byte-identical \
             to real aomenc",
            ct.tag
        );
        // Oracle anchor (sound direction): palette on vs off.
        let pal_off = EncodeCell::frame_obu_payload(&cell.c_encode_screen(false, false));
        let pal_on = EncodeCell::frame_obu_payload(&cell.c_encode_screen(true, false));
        let reacted = pal_off != pal_on;
        let _ = writeln!(
            report,
            "  {:<14} bd{} mono{} {:>2}/{:>2} blocks<=4colors -> screen={} | --enable-palette=1 {}",
            ct.tag,
            cell.bd,
            cell.mono as u8,
            st.counts_1,
            st.blocks,
            st.allow_screen_content_tools as u8,
            if reacted { "MOVED the C payload" } else { "inert" }
        );
        assert!(
            st.allow_screen_content_tools || !reacted,
            "{}: classified NOT-screen, but real aomenc reacted to \
             --enable-palette=1 — allow_screen_content_tools must therefore be \
             1 and the content classifier is WRONG. Every conclusion the \
             content matrix draws from this classification is void.",
            ct.tag
        );
        if st.allow_screen_content_tools && reacted {
            screen_reacted += 1;
        }
        assert_eq!(
            st.allow_screen_content_tools, ct.screen,
            "{}: measured screen verdict != the declared taxonomy class",
            ct.tag
        );
        measured.push((ct.tag, st.counts_1, st.blocks, st.allow_screen_content_tools));
    }
    println!("{report}");
    assert_eq!(
        measured,
        CONTENT_SCREEN_STATS.to_vec(),
        "the content taxonomy MOVED — a probe's crop/vector changed, or \
         cp::screen_stat changed. Re-derive the classes before re-pinning; the \
         content-sensitivity matrix is only meaningful against a fixed partition."
    );
    assert!(
        screen_reacted >= 1,
        "no screen-classified content reacted to --enable-palette=1, so the \
         oracle anchor is vacuous — the probe set no longer contains reachable \
         screen content"
    );
    let n_screen = CONTENTS.iter().filter(|c| c.screen).count();
    assert!(
        n_screen >= 3 && n_screen < CONTENTS.len(),
        "the taxonomy must partition the probe set into BOTH classes"
    );
}

// ---------------------------------------------------------------------------
// 5b. The content-sensitivity matrix — which axes move with content
// ---------------------------------------------------------------------------

/// One `(content, quality, axis=level)` cell whose port output is NOT
/// byte-identical to real aomenc, in the pinned wire format
/// `tag/cqN/axis=value`.
///
/// **Measured 2026-07-30** by [`run_content_matrix`] over 12 contents x 26
/// singleton axis levels (9 non-screen at cq32; the 3 screen contents at
/// cq12/32/63) = 468 cells. The original measurement found **exactly one of
/// the 21 axes content-sensitive**: `dtxo` (`--use-intra-default-tx-only`), on
/// exactly the screen-classified contents — 8 cells, root-caused as KB-17.
///
/// **KB-17 is FIXED (2026-07-30) and all 8 `dtxo` cells are gone from this
/// set.** Root cause was `crates/aom-encode/src/speed_features.rs`'s
/// hardcoded `use_screen_content_tools: false` in
/// `tx_type_search_policy_for_stage`: `get_tx_mask` (tx_search.c:1806-1808)
/// resolves `use_default_intra_tx_type` through
/// `get_default_tx_type(PLANE_TYPE_Y, xd, tx_size,
/// cpi->use_screen_content_tools)`, which returns `DCT_DCT` when the screen
/// flag is set instead of the mode-derived tx type. The port modelled the
/// function faithfully (`aom_encode::tx_search::get_default_tx_type_y`) but
/// pinned its 4th argument false, so on any screen-detected content it
/// searched the mode-derived tx type where C searches DCT_DCT. The fix
/// threads `SpeedFeatures::allow_screen_content_tools` (already an input to
/// `set_allintra`, sourced from the parsed frame header) into the policy.
/// With it, `dtxo` is no longer content-sensitive and is covered at full
/// strength by the screen covering arrays ([`run_content_array`]).
///
/// The ONE remaining entry, `scr_mono_b10/cq32/diag=0`, was measured on the
/// same run as NOT moving with the KB-17 fix — it is the separate
/// KB-10/KB-12 "cheaper RD decision" near-tie also pinned by
/// [`mono_vector_open_divergences_pinned`], whose bd8 twin does not reproduce
/// it.
///
/// This set is SELF-PROMOTING in both directions: a cell that starts matching
/// fails, and so does a cell that starts diverging.
const CONTENT_DIVERGENT_CELLS: &[&str] = &["scr_mono_b10/cq32/diag=0"];

/// Run the singleton-axis sweep for `contents` x `cqs` and return
/// `(divergent, inert)` cell keys plus the cell count.
///
/// Each cell is one axis level ALONE on one content at one quality, encoded on
/// both sides and compared byte-for-byte — the same contract as every
/// covering-array cell. The `inert` list is the anti-vacuity companion: an
/// axis level that never moves the C encoder on a content cannot be evidence
/// about that content either way, and the caller reports it.
fn run_content_matrix(contents: &[&Content], cqs: &[i32]) -> (BTreeSet<String>, usize, usize) {
    c::ref_init();
    let mut divergent = BTreeSet::new();
    let mut cells = 0usize;
    let mut inert = 0usize;
    let mut report = String::new();
    for ct in contents {
        for &cq in cqs {
            let cell = ct.cell(cq);
            let stock = c_stock_payload(&cell);
            let port_stock = cell.port_encode_with(&cell.c_encode_ctrls(&[]), &ToggleKnobs::default());
            assert_eq!(
                stock, port_stock,
                "{}/cq{cq}: the STOCK encode of this content is NOT byte-identical \
                 to real aomenc — that is a plain envelope regression, not a \
                 knob-vs-content interaction",
                ct.tag
            );
            let mut div_here: Vec<String> = Vec::new();
            for (i, ax) in ALL_AXES.iter().enumerate() {
                for l in 1..ax.n_levels() as u8 {
                    let mut row = DEFAULT_ROW;
                    row[i] = l;
                    if cp::illegal_reason(&row).is_some() {
                        continue;
                    }
                    let key = format!("{}/cq{cq}/{}={}", ct.tag, ax.tag(), ax.values()[l as usize]);
                    let r = run_cell(&cell, &key, &cp::knobs_of(&row), &stock);
                    cells += 1;
                    if !r.c_moved {
                        inert += 1;
                    }
                    if !r.exact {
                        divergent.insert(key.clone());
                        div_here.push(format!(
                            "{}={} (port {}B vs C {}B)",
                            ax.tag(),
                            ax.values()[l as usize],
                            r.port_len,
                            r.c_len
                        ));
                    }
                }
            }
            let _ = writeln!(
                report,
                "  {:<14} cq{:<2} bd{} mono{} screen={} : {} divergent [{}]",
                ct.tag,
                cq,
                cell.bd,
                cell.mono as u8,
                ct.screen as u8,
                div_here.len(),
                div_here.join(", ")
            );
        }
    }
    println!("\n=== content-sensitivity matrix ({cells} cells) ===\n{report}");
    (divergent, cells, inert)
}

/// Assert one shard's divergences are exactly the pinned subset for it.
fn check_content_shard(contents: &[&Content], cqs: &[i32]) {
    let (divergent, cells, inert) = run_content_matrix(contents, cqs);
    let tags: BTreeSet<&str> = contents.iter().map(|c| c.tag).collect();
    let expected: BTreeSet<String> = CONTENT_DIVERGENT_CELLS
        .iter()
        .filter(|k| {
            let tag = k.split('/').next().unwrap();
            let cq: i32 = k.split('/').nth(1).unwrap()[2..].parse().unwrap();
            tags.contains(tag) && cqs.contains(&cq)
        })
        .map(|s| s.to_string())
        .collect();
    assert!(cells > 0, "empty content shard");
    assert!(
        inert * 2 < cells,
        "over half this shard's cells ({inert}/{cells}) are INERT on the C \
         encoder — the content probes are drifting toward configurations the \
         encoder ignores, so the shard proves little about them"
    );
    assert_eq!(
        divergent, expected,
        "the CONTENT-sensitivity matrix MOVED. A cell that started diverging \
         is a regression — in particular any `dtxo=1` cell reappearing here \
         means the screen-content tx-type flag \
         (`SpeedFeatures::allow_screen_content_tools` -> \
         `TxTypeSearchPolicy::use_screen_content_tools`, KB-17) stopped being \
         threaded. A cell that started matching means an open near-tie closed \
         — re-pin."
    );
}

#[test]
fn content_sensitivity_natural_s0() {
    check_content_shard(&[&CONTENTS[0], &CONTENTS[1], &CONTENTS[2]], &[32]);
}
#[test]
fn content_sensitivity_natural_s1() {
    check_content_shard(&[&CONTENTS[3], &CONTENTS[4], &CONTENTS[5]], &[32]);
}
#[test]
fn content_sensitivity_natural_s2() {
    check_content_shard(&[&CONTENTS[6], &CONTENTS[7], &CONTENTS[8]], &[32]);
}
#[test]
fn content_sensitivity_screen_mono_b8() {
    check_content_shard(&[&CONTENTS[9]], &[12, 32, 63]);
}
#[test]
fn content_sensitivity_screen_mono_b10() {
    check_content_shard(&[&CONTENTS[10]], &[12, 32, 63]);
}
#[test]
fn content_sensitivity_screen_ibc_b8() {
    check_content_shard(&[&CONTENTS[11]], &[12, 32, 63]);
}

// ---------------------------------------------------------------------------
// 5c. The covering array, replayed on the SCREEN class
// ---------------------------------------------------------------------------

/// Covering-array rows on SCREEN-class content that are NOT byte-identical to
/// real aomenc, pinned open and self-promoting in both directions.
///
/// **One row, and it is PRE-EXISTING, not a consequence of the KB-17 fix.**
/// It only became visible when the KB-17 fix let `dtxo` out of
/// `pin_dtxo_default`, i.e. the pin had been hiding it. Direct A/B on
/// `combinations_t4_scr_ibc_s0` (2026-07-30, same binary, only
/// `TxTypeSearchPolicy::use_screen_content_tools` toggled):
///
/// | screen flag threaded | open cells / 63 | this row |
/// |---|---|---|
/// | no (pre-KB-17) | **23** | port 109 B vs C 79 B |
/// | yes (KB-17 fixed) | **1** | port 108 B vs C 79 B |
///
/// So the fix closed 22 of the 23, and this row was already divergent without
/// it. Its `dtxo=0` sibling is exact (the whole array was exact under the pin),
/// so it IS a `dtxo x <something>` interaction — the row also carries
/// `txss0` (`--enable-tx-size-search=0` -> TX_MODE_LARGEST), `maxp64`,
/// `minp16`, `smth0`, `diag0`, `flip0`, `cdf0`. The 29-byte gap is far outside
/// the KB-10/KB-12 near-tie signature, so this is a real second defect on the
/// screen tx-type path and is tracked as such rather than dismissed.
const SCREEN_ARRAY_OPEN_ROWS: &[&str] =
    &["scr_ibc_b8cq32_p140-minp16-maxp64-smth0-diag0-flip0-dtxo1-txss0-cdf0"];

/// Replay a covering array on one CONTENT probe, at FULL axis strength.
///
/// `dtxo` (`--use-intra-default-tx-only`) used to be pinned to its default
/// level here, because it was the one axis the content matrix measured as
/// content-sensitive (KB-17). **KB-17 is fixed and the pin is gone
/// (2026-07-30)** — the screen contexts now run every axis, including `dtxo`,
/// at full strength, so `dtxo x anything` on screen content is covered by the
/// same t-way guarantee as every other axis.
///
/// Same four-part gate as [`run_array`] (byte-identity, anti-vacuity,
/// collapse soundness in the stock direction, non-empty shard).
fn run_content_array(ct: &Content, cq: i32, t: usize, shard: usize, n_shards: usize, min_moved_pct: f64) {
    c::ref_init();
    let cell = ct.cell(cq);
    let cctx = cell_ctx(&cell);
    let stock = c_stock_payload(&cell);
    let stock_eff = Effective::resolve(&DEFAULT_ROW, &cctx);
    let tag = format!("{}cq{cq}", ct.tag);

    let mut seen: BTreeSet<Row> = BTreeSet::new();
    let pinned: Vec<Row> = cp::covering_array(t)
        .into_iter()
        .filter(|r| cp::illegal_reason(r).is_none() && seen.insert(*r))
        .collect();
    let collapsed = cp::collapse(&pinned, &cctx);
    let rows: Vec<Row> = collapsed
        .representatives
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % n_shards == shard)
        .map(|(_, r)| r)
        .collect();
    assert!(!rows.is_empty(), "{tag}: shard {shard}/{n_shards} is empty");

    let mut cells = Vec::new();
    for row in &rows {
        let label = format!("{tag}_{}", cp::row_label(row));
        let r = run_cell(&cell, &label, &cp::knobs_of(row), &stock);
        if Effective::resolve(row, &cctx) == stock_eff {
            assert!(
                !r.c_moved,
                "{label}: the collapse engine resolves this row to the STOCK \
                 effective config, but real aomenc produced different bytes on \
                 this CONTENT — the Effective signature is under-refined"
            );
        }
        cells.push(r);
    }
    let non_stock: Vec<&Cell> = cells.iter().filter(|c| !c.label.ends_with("_stock")).collect();
    let moved = non_stock.iter().filter(|c| c.c_moved).count();
    let moved_pct = if non_stock.is_empty() {
        100.0
    } else {
        100.0 * moved as f64 / non_stock.len() as f64
    };
    println!("{}", render(&cells, &tag, t, shard, n_shards, moved_pct));
    let open: BTreeSet<String> = cells
        .iter()
        .filter(|c| !c.exact)
        .map(|c| c.label.clone())
        .collect();
    // The pinned set is global; restrict it to the rows THIS shard actually
    // encoded, so the comparison is against what was measured here. (A pinned
    // row disappearing from the array altogether would show up as an empty or
    // shrunken shard, which the asserts around this one already cover.)
    let expected: BTreeSet<String> = SCREEN_ARRAY_OPEN_ROWS
        .iter()
        .map(|s| s.to_string())
        .filter(|l| cells.iter().any(|c| &c.label == l))
        .collect();
    assert_eq!(
        open, expected,
        "{tag}: the SCREEN-class covering-array divergence set MOVED ({} of {} \
         cells open). Every axis runs at full strength here (no pins since \
         KB-17 closed), so a NEW entry is a knob combination that diverges on \
         this CONTENT and nowhere else; a vanished entry means an open row \
         closed — re-pin SCREEN_ARRAY_OPEN_ROWS. Measured: {}",
        open.len(),
        cells.len(),
        cells
            .iter()
            .filter(|c| !c.exact)
            .map(|c| format!("{} (port {}B vs C {}B)", c.label, c.port_len, c.c_len))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(
        moved_pct >= min_moved_pct,
        "{tag}: only {moved_pct:.1}% of the {} non-stock rows changed the C \
         encoder's output (floor {min_moved_pct}%)",
        non_stock.len()
    );
}

// t=4 (every 4-way interaction among the 20 non-dtxo axes) on the bd8 4:2:0
// screen content — the format is the SAME as the primary 64cq32 context, so
// anything these cells catch is attributable to content alone.
#[test]
fn combinations_t4_scr_ibc_s0() {
    run_content_array(&CONTENTS[11], 32, 4, 0, 3, 80.0);
}
#[test]
fn combinations_t4_scr_ibc_s1() {
    run_content_array(&CONTENTS[11], 32, 4, 1, 3, 80.0);
}
#[test]
fn combinations_t4_scr_ibc_s2() {
    run_content_array(&CONTENTS[11], 32, 4, 2, 3, 80.0);
}
// t=3 on the bd8 native monochrome screen vector — the corpus's only bd8
// monochrome content, and the bd8 twin of KB-17's bd10 vector.
#[test]
fn combinations_t3_scr_mono_b8() {
    run_content_array(&CONTENTS[9], 32, 3, 0, 1, 70.0);
}

/// `--use-intra-default-tx-only=1` x a t=2 array, on SCREEN-class content:
/// the combination companion to the standalone divergence in
/// [`CONTENT_DIVERGENT_CELLS`].
///
/// **KB-17 closed this set to EMPTY (2026-07-30):** all 17 t=2 rows are now
/// byte-identical with `--use-intra-default-tx-only=1` forced on top, where 12
/// of 17 diverged before the screen-content tx-type flag was threaded. The
/// test is kept as the explicit, focused regression gate for that fix (the
/// axis is also covered inside [`run_content_array`] now that its pin is
/// gone); it is self-promoting in the regression direction — any row that
/// starts diverging fails.
#[test]
fn combinations_screen_dtxo_verdict_set_pinned() {
    c::ref_init();
    let ct = &CONTENTS[11];
    let cell = ct.cell(32);
    let stock = c_stock_payload(&cell);
    let mut diverged = BTreeSet::new();
    let mut n = 0usize;
    for row in cp::covering_array(2) {
        if cp::illegal_reason(&row).is_some() {
            continue;
        }
        let mut knobs = cp::knobs_of(&row);
        knobs.use_intra_default_tx_only = true;
        let label = format!("scr_dtxo_{}", cp::row_label(&row));
        let r = run_cell(&cell, &label, &knobs, &stock);
        n += 1;
        if !r.exact {
            diverged.insert(cp::row_label(&row));
        }
    }
    println!(
        "\n=== --use-intra-default-tx-only=1 x t=2 array on SCREEN content: \
         {}/{n} rows diverge\n{:#?}",
        diverged.len(),
        diverged
    );
    let expected: BTreeSet<String> = SCREEN_DTXO_DIVERGENT_ROWS.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        diverged, expected,
        "the screen-content --use-intra-default-tx-only divergence set MOVED. \
         It is EMPTY since KB-17 was fixed, so any row here is a REGRESSION of \
         the screen-content tx-type flag \
         (`SpeedFeatures::allow_screen_content_tools` -> \
         `TxTypeSearchPolicy::use_screen_content_tools` -> \
         `get_default_tx_type_y`, tx_search.c:1806-1808)."
    );
    assert!(n >= 17, "the t=2 array shrank to {n} legal rows — coverage lost");
}

/// Recorded divergence set for [`combinations_screen_dtxo_verdict_set_pinned`]:
/// **EMPTY since KB-17 was fixed (measured 2026-07-30, 0 of 17 rows diverge).**
///
/// It was **12 of the 17** t=2 rows on `scr_ibc_b8` (bd8 4:2:0 screen content)
/// at cq32 before the fix, with `stock` — the knob alone, no other change —
/// in the set: the combinations INHERITED the standalone content divergence
/// rather than adding new ones. The pre-fix set is recorded here so a
/// regression is recognisable by shape, not just by count:
///
/// ```text
/// ab0-p140-maxp64-paeth0-cfl0-diag0-fint0-rtx0-txss0-cdf2-trel2
/// minp16-maxp32-smth0-cfl0-diag0-rtx0-flip0-rtxs1-cdf2-trel0
/// p140-minp8-smth0-paeth0-diag0-adlt0-fint0-flip0-txss0-cdf2
/// rect0-ab0-diag0-fint0-edgf0-tx640-rtx0-flip0-rtxs1-trel1
/// rect0-ab0-minp16-cfl0-dir0-diag0-fint0-edgf0-trel2
/// rect0-ab0-p140-minp16-paeth0-dir0-adlt0-fint0-edgf0-rtxs1-cdf0
/// rect0-ab0-p140-minp8-paeth0-cfl0-dir0-diag0-rtx0-txss0-cdf2-trel1
/// rect0-p140-minp16-maxp64-smth0-cfl0-diag0-adlt0-fint0-edgf0-rtx0-rtxs1-txss0-trel0
/// rect0-p140-minp8-maxp32-adlt0-edgf0-rtx0-txss0-cdf0-trel2
/// rect0-p140-minp8-maxp64-smth0-paeth0-adlt0-fint0-tx640-rtx0
/// rect0-p140-paeth0-dir0-diag0-fint0-rtxs1-txss0-cdf0-trel0
/// stock
/// ```
const SCREEN_DTXO_DIVERGENT_ROWS: &[&str] = &[];

/// DEEP TIER (`--ignored`) — the full content-axis evidence grid, written to
/// `benchmarks/config_perm_content_axis_2026-07-30.tsv`.
///
/// The default tier runs a deliberately asymmetric subset (9 non-screen
/// contents at cq32; the 3 screen contents at cq12/32/63) because the
/// measurement below says the non-screen class is flat across quality. This
/// sweep is the evidence for that asymmetry: **all 12 contents x all 3
/// qualities x all 26 singleton axis levels = 936 cells**, plus the screen
/// statistic per content. It is opt-in because it writes `benchmarks/`, and
/// because ~120 s of encoding to re-derive a table that has not moved is not
/// worth the default-tier budget.
///
/// Re-run it whenever the axis set, the content probes, or the encoder's
/// content handling changes.
#[test]
#[ignore = "offline evidence generation for the content axis; ~2 min, writes benchmarks/"]
fn content_axis_evidence_sweep() {
    c::ref_init();
    let mut tsv = String::from(
        "# Content-axis sensitivity of the 21 config-permutation knob axes.\n\
         # Every data row is ONE singleton axis level on ONE content at ONE\n\
         # quality, encoded by the port and by real libaom v3.14.1 and compared\n\
         # byte for byte (frame OBU payload). speed 0, ALLINTRA, KEY, 1 tile.\n\
         # `screen` is cp::screen_stat (estimate_screen_content, encoder.c:2042).\n\
         # `c_moved` = the C encoder's own payload differs from its stock encode.\n\
         # The per-content constants are in the `#C` preamble rows, not repeated\n\
         # on every data row.\n\
         #C\tcontent\tvector\tbd\tmono\tw\th\tcounts_1\tblocks\tscreen\n",
    );
    for ct in CONTENTS {
        let cell = ct.cell(32);
        let st = cp::screen_stat(&cell.y, cell.w, cell.h, cell.bd);
        let _ = writeln!(
            tsv,
            "#C\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            ct.tag,
            ct.vector,
            cell.bd,
            cell.mono as u8,
            cell.w,
            cell.h,
            st.counts_1,
            st.blocks,
            st.allow_screen_content_tools as u8
        );
    }
    // `c_len` is written as `-` whenever the cell is byte-identical (the two
    // payloads are then the same bytes, so the column would be pure
    // duplication) — this keeps the committed evidence file small.
    tsv.push_str("content\tcq\taxis\tlevel\texact\tc_moved\tport_len\tc_len\n");
    for ct in CONTENTS {
        for cq in [12, 32, 63] {
            let cell = ct.cell(cq);
            let stock = c_stock_payload(&cell);
            for (i, ax) in ALL_AXES.iter().enumerate() {
                for l in 1..ax.n_levels() as u8 {
                    let mut row = DEFAULT_ROW;
                    row[i] = l;
                    if cp::illegal_reason(&row).is_some() {
                        continue;
                    }
                    let key = format!("{}/cq{cq}/{}={}", ct.tag, ax.tag(), ax.values()[l as usize]);
                    let r = run_cell(&cell, &key, &cp::knobs_of(&row), &stock);
                    let _ = writeln!(
                        tsv,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        ct.tag,
                        cq,
                        ax.tag(),
                        ax.values()[l as usize],
                        r.exact as u8,
                        r.c_moved as u8,
                        r.port_len,
                        if r.exact { "-".to_string() } else { r.c_len.to_string() }
                    );
                }
            }
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/config_perm_content_axis_2026-07-30.tsv");
    std::fs::write(&path, format!("{}{tsv}", evidence_provenance()))
        .expect("write content-axis evidence TSV");
    println!("wrote {}", path.display());
}

// ===========================================================================
// 8. THE SIZE AXIS — frame geometry as a first-class context (2026-07-30)
// ===========================================================================
//
// The eight contexts above are 64x64, 32x32 and 128x128, all SB64. That is
// THREE points on a size axis with more structure than three points, and the
// question this section answers with data is: how many distinct SIZE CLASSES
// exist, which ones does the pre-existing array cover, and what does each
// added cell buy?
//
// The model is `config_perm::size_derived` — the size analogue of `Effective`.
// It resolves a `CellCtx` to the encoder state its GEOMETRY determines, so two
// sizes with the same `SizeDerived` cannot behave differently *because of their
// size* and replaying the array at both is redundant. `size_class_partition`
// does the collapse; `size_class_inventory_is_pinned` pins the resulting table.
//
// HEADLINE RESULT (`size_axis_class_table`, printed by the test):
//
//   * At SPEED 0 — the speed the whole array runs at — every framesize-
//     dependent SPEED FEATURE below 2160p is either inert on the all-intra KEY
//     path or gated on speed >= 1.
//
//     SCOPE, CORRECTED 2026-07-31 (KB-22): this survey covered
//     `set_allintra_speed_feature_framesize_dependent` ONLY, and must not be
//     read as a statement about the port in general. C runs a THIRD
//     speed-feature pass — `av1_set_speed_features_qindex_dependent`
//     (speed_features.c:2873, called from encoder.c:3114) — whose `speed == 0`
//     block has an arm gated on `is_720p_or_larger && base_qindex <= 128`
//     (:2914) that moves `perform_coeff_opt` and
//     `intra_tx_size_search_init_depth_rect`. The port did not model that pass
//     at all until KB-22; it is live from 720p UP. Every cell in THIS array is
//     <= 640px, so the headline holds for the cells it was measured on — which
//     is exactly why the gap went unseen here and had to be found at 2160p.
//     Read this bullet as "sub-720p", not "below 2160p", for anything outside
//     this array.
//
//     The `prune_tx_type_using_stats` >= 480p
//     threshold the port comments at `aom-encode/src/speed_features.rs:695`
//     needs speed >= 2, so it is 0 in real aomenc too on every one of the 2,617
//     cells: that particular hole is NOT a divergence risk at this speed. The
//     size axis at speed 0 is a GEOMETRY axis, not a speed-feature axis.
//   * The ONE exception is `use_square_partition_only_threshold`
//     (speed_features.c:176/181; port `partition_pick.rs:2446`), which is
//     BLOCK_64X64 sub-480p and BLOCK_128X128 at >= 480p. Its only intra
//     consumer is the square-only rect-kill (`bsize > threshold`,
//     partition_search.c:5700; port `partition_pick.rs:2593`), which needs a
//     block STRICTLY LARGER than the threshold — impossible at SB64, where the
//     largest block is BLOCK_64X64. So it is structurally dead on all eight
//     pre-existing contexts and becomes live only under `--sb-size=128`.
//   * `--sb-size=128` IS reachable and byte-exact (`sb128_e2e.rs`), so
//     `cell_ctx`'s "SB64 only" comment was stale. At SB128 the >= 480p
//     threshold flips a real search-space decision, and no gate covered it.
//   * Frame-EDGE geometry (dimensions not a multiple of the superblock) is a
//     distinct code path with four KB-6 roots in it, and NONE of the eight
//     contexts has a partial superblock except 32x32, which is smaller than one
//     SB and therefore never reaches the multi-SB edge interactions.
//
// Out of budget, recorded rather than faked: `default_min_partition_size =
// BLOCK_8X8` at `is_4k_or_larger` (speed_features.c:187-189) is UNMODELLED by
// the port (`config_perm::PORT_GAP_DEFAULT_MIN_PARTITION_SIZE`). A 480x480
// speed-0 cell already costs ~2.9-12.6 s; 2160x2160 is ~20x that per cell. It
// is a documented port gap with a citation, not a gated one.

use aom_bench::config_perm::{size_class_partition, size_derived};

/// A SIZE context: geometry is the variable, so the content generator and the
/// superblock size are explicit rather than implied.
struct SizeCtx {
    tag: &'static str,
    w: usize,
    h: usize,
    /// `--sb-size=128` (the port reads `use_128x128_superblock` back out of the
    /// bootstrap seq header; `port_encode_full`, aom-bench/src/lib.rs:993).
    sb128: bool,
    cq: i32,
    mono: bool,
    /// Source: a crop of this conformance vector, or (when the frame is bigger
    /// than every vector in the intra scope) a mirror-tiling of it.
    vector: &'static str,
    /// Crop window `(w, h, ox, oy)` into the vector; `None` = mirror-tile the
    /// whole decoded frame up to `w x h`.
    crop: Option<(usize, usize, usize, usize)>,
    /// Measured cost of one cell (C encode + port encode) on the reference box,
    /// used to justify the strength chosen for this context.
    ms_per_cell: u32,
}

impl SizeCtx {
    fn ctx(&self) -> CellCtx {
        CellCtx {
            w: self.w,
            h: self.h,
            mono: self.mono,
            sb_px: if self.sb128 { 128 } else { 64 },
        }
    }

    fn cell(&self) -> EncodeCell {
        let base = EncodeCell::real_content(self.tag, self.vector, self.crop, self.cq, 0);
        let mut c = if base.w == self.w && base.h == self.h {
            base
        } else {
            mirror_tile(&base, self.tag, self.w, self.h, self.cq)
        };
        if self.mono {
            c.mono = true;
            c.u.clear();
            c.v.clear();
        }
        c
    }

    /// Rows this context must NOT run, each with the reason.
    ///
    /// **EMPTY since KB-18 was FIXED (2026-07-30).** The only entry this ever
    /// had was `--max-partition-size=32` under `--sb-size=128`, which the size
    /// axis found as an open port defect: C restores the partition context
    /// CONDITIONALLY,
    ///
    /// > `if (bsize <= x->sb_enc.max_partition_size || bsize == cm->seq_params->sb_size)`
    /// > `  av1_restore_context(...)` — partition_search.c:4646
    ///
    /// and the port restored UNCONDITIONALLY, asserting that condition instead.
    /// The predicate is FALSE whenever a block size sits strictly between the
    /// max-partition cap and the superblock size — impossible at SB64 (where
    /// `bsize == sb_size` covers the top and the cap covers the rest) and
    /// reachable at SB128 with a 32 px cap, where `bsize == BLOCK_64X64`
    /// satisfies neither clause. `partition_pick.rs` now takes the restore
    /// conditionally, so the SB128 contexts run `MaxPart` level 2 at full
    /// strength and this hook has no entries. The mechanism is kept (rather
    /// than deleted) because it is the honest place for the NEXT such finding.
    fn skip_reason(&self, row: &Row) -> Option<&'static str> {
        let _ = row;
        None
    }

    /// Every C encode in this context carries the superblock-size control, so
    /// the C reference and the port bootstrap agree on the SB geometry.
    fn ctrls(&self, knobs: &ToggleKnobs) -> Vec<(i32, i32)> {
        let mut v = knobs.c_ctrls();
        if self.sb128 {
            v.push((
                c::cx_ctrl::AV1E_SET_SUPERBLOCK_SIZE,
                c::cx_ctrl::AOM_SUPERBLOCK_SIZE_128X128,
            ));
        }
        v
    }
}

/// Mirror-tile a decoded cell up to `w x h`. Mirroring (rather than wrapping)
/// keeps the seam continuous, so the enlarged frame stays photographic content
/// with real local statistics instead of acquiring a synthetic edge grid every
/// tile period — the size axis must not smuggle in a content change.
fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, cq: i32) -> EncodeCell {
    let mir = |i: usize, n: usize| {
        let m = i % (2 * n);
        if m < n {
            m
        } else {
            2 * n - 1 - m
        }
    };
    let (bw, bh) = (base.w, base.h);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = base.y[mir(r, bh) * bw + mir(col, bw)];
        }
    }
    let (bcw, bch) = (
        (bw + base.ss_x) >> base.ss_x,
        (bh + base.ss_y) >> base.ss_y,
    );
    let (cw, ch) = ((w + base.ss_x) >> base.ss_x, (h + base.ss_y) >> base.ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = base.u[mir(r, bch) * bcw + mir(col, bcw)];
            v[r * cw + col] = base.v[mir(r, bch) * bcw + mir(col, bcw)];
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono: false,
        ss_x: base.ss_x,
        ss_y: base.ss_y,
        usage: 2,
        cq_level: cq,
        speed: 0,
        bd: base.bd,
        y,
        u,
        v,
    }
}

// --- the size contexts -----------------------------------------------------
// `av1-1-b8-00-quantizer-00` is 352x288 photographic bd8 4:2:0 — the same
// source the KB-6 real-image gate and the SB128 gate ride, so any divergence
// found here is attributable to GEOMETRY and not to unproven content.

/// CLASS 4 — multi-SB with a partial superblock in BOTH dimensions, 4 px of
/// overhang (68 = 64 + 4). The cheapest representative of the KB-6 frame-edge
/// class: `set_partition_cost_for_edge_blk`'s frame-init cdf gather
/// (partition_search.c:3415), `av1_set_entropy_contexts`' beyond-visible
/// tail-zero (blockd.c:29), the visible-distortion clips, and the forced
/// partitions from `av1_blk_has_rows_and_cols` (:3389) are ALL live, and all
/// four interact with essentially every knob (which blocks land on the edge is
/// a function of the partition and transform knobs). Full t=4 strength.
const S_PART68: SizeCtx = SizeCtx {
    tag: "part68",
    w: 68,
    h: 68,
    sb128: false,
    cq: 32,
    mono: false,
    vector: "av1-1-b8-00-quantizer-00",
    crop: Some((68, 68, 0, 0)),
    ms_per_cell: 199,
};

/// CLASS 4, second overhang magnitude — 32 px instead of 4 px. Overhang size is
/// a CONTINUOUS sub-axis inside the class (it selects which transform-block
/// footprints the tail-zero clips), not a class of its own, so it is sampled at
/// a second point with a t=2 array rather than replayed at full strength.
const S_PART96: SizeCtx = SizeCtx {
    tag: "part96",
    w: 96,
    h: 96,
    sb128: false,
    cq: 32,
    mono: false,
    vector: "av1-1-b8-00-quantizer-00",
    crop: Some((96, 96, 0, 0)),
    ms_per_cell: 439,
};

/// CLASS 5 — partial in ONE dimension only (128 is SB-aligned, 96 is not). The
/// above-context and left-context clipping are separate code paths, so
/// "partial in x" and "partial in y" are not the same state as "partial in
/// both"; this context is the asymmetric witness.
const S_PART128X96: SizeCtx = SizeCtx {
    tag: "part128x96",
    w: 128,
    h: 96,
    sb128: false,
    cq: 32,
    mono: false,
    vector: "av1-1-b8-00-quantizer-00",
    crop: Some((128, 96, 0, 0)),
    ms_per_cell: 500,
};

/// CLASS 9 — SB128 with a frame SMALLER than one superblock, so the 128 root is
/// force-split and no 128-sized block exists (`av1_blk_has_rows_and_cols`).
/// The SB128 analogue of the 32x32 context.
const S_SB128_64: SizeCtx = SizeCtx {
    tag: "sb128_64",
    w: 64,
    h: 64,
    sb128: true,
    cq: 32,
    mono: false,
    vector: "av1-1-b8-01-size-64x64",
    crop: None,
    ms_per_cell: 170,
};

/// CLASS 6 — SB128, one full 128 superblock, sub-480p. This is where
/// `use_square_partition_only_threshold` first becomes REACHABLE: BLOCK_128X128
/// > BLOCK_64X64, so the rect-kill fires and HORZ/VERT are removed at the 128
/// root for full superblocks. The pre-existing array never reaches it.
const S_SB128_128: SizeCtx = SizeCtx {
    tag: "sb128_128",
    w: 128,
    h: 128,
    sb128: true,
    cq: 32,
    mono: false,
    vector: "av1-1-b8-00-quantizer-00",
    crop: Some((128, 128, 64, 64)),
    ms_per_cell: 1028,
};

/// CLASS 7 — SB128 AND a partial superblock (192 = 128 + 64): the composition
/// of classes 4 and 6, where the edge paths and the rect-kill are live at once.
const S_SB128_192: SizeCtx = SizeCtx {
    tag: "sb128_192",
    w: 192,
    h: 192,
    sb128: true,
    cq: 32,
    mono: false,
    vector: "av1-1-b8-00-quantizer-00",
    crop: Some((192, 192, 0, 0)),
    ms_per_cell: 1987,
};

/// CLASS 8 — the >= 480p class: `min(w,h) >= 480` raises
/// `use_square_partition_only_threshold` to BLOCK_128X128, so the rect-kill
/// STOPS firing and HORZ/VERT return at the 128 root. This is THE
/// framesize-dependent speed feature that is live at speed 0, and the only one
/// below 2160p — and it is what `size_axis_teeth_are_real` perturbs.
///
/// Content/geometry/quality are chosen by measurement, not taste
/// (`benchmarks/config_perm_size_axis_2026-07-30.tsv`):
///
/// * cq63, because the 128-level partitions this context exists to exercise are
///   only chosen there — at cq40..55 real aomenc splits every 128 root and
///   `--max-partition-size=64` produces the IDENTICAL stream, i.e. the context
///   would be vacuous;
/// * monochrome, because the 4:2:0 cq63 cell carries a divergence that
///   reproduces identically at SB64 and is therefore not size-attributable
///   (`size_axis_open_divergences_pinned`);
/// * 576 rather than 480 or 512, because those two carry per-cell RD near-ties
///   (`size_axis_open_divergences_pinned` finding B) that class-mates at 576
///   and 640 do NOT reproduce — so they are content/near-tie divergences, not
///   properties of the size class, and gating on them would mis-attribute a
///   near-tie to the geometry.
const S_SB128_576: SizeCtx = SizeCtx {
    tag: "sb128_576m",
    w: 576,
    h: 576,
    sb128: true,
    cq: 63,
    mono: true,
    vector: "av1-1-b8-00-quantizer-00",
    crop: None,
    ms_per_cell: 4000,
};

/// >= 480p SB128, ALIGNED — NOT gated: `--enable-ab-partitions=0` carries the
/// open near-tie pinned by `size_axis_open_divergences_pinned` finding B.
const S_SB128_512_OPEN: SizeCtx = SizeCtx {
    tag: "sb128_512m_open",
    w: 512,
    h: 512,
    sb128: true,
    cq: 63,
    mono: true,
    vector: "av1-1-b8-00-quantizer-00",
    crop: None,
    ms_per_cell: 3200,
};

/// >= 480p SB128, frame-edge PARTIAL — NOT gated: `--enable-1to4-partitions=0`
/// carries the open near-tie pinned by `size_axis_open_divergences_pinned`.
const S_SB128_480_OPEN: SizeCtx = SizeCtx {
    tag: "sb128_480m_open",
    w: 480,
    h: 480,
    sb128: true,
    cq: 63,
    mono: true,
    vector: "av1-1-b8-00-quantizer-00",
    crop: None,
    ms_per_cell: 2850,
};

/// Every size context, cheapest first.
const ALL_SIZE_CONTEXTS: &[&SizeCtx] = &[
    &S_SB128_64,
    &S_PART68,
    &S_PART96,
    &S_PART128X96,
    &S_SB128_128,
    &S_SB128_192,
    &S_SB128_576,
];

/// One size cell: byte-identity against real aomenc, with the SB-size control
/// on both sides.
fn run_size_cell(sc: &SizeCtx, cell: &EncodeCell, label: &str, knobs: &ToggleKnobs, stock: &[u8]) -> Cell {
    let c_tu = cell.c_encode_ctrls(&sc.ctrls(knobs));
    assert!(!c_tu.is_empty(), "{label}: C encode failed");
    let c_payload = EncodeCell::frame_obu_payload(&c_tu);
    let port = cell.port_encode_with(&c_tu, knobs);
    let exact = port == c_payload;
    if !exact {
        let tu = aom_bench::rd_close::splice_frame_obu(&c_tu, &port);
        let dec = aom_bench::rd_close::port_decode_tu(label, &tu);
        assert_eq!(
            (dec.width, dec.height, dec.monochrome),
            (cell.w, cell.h, cell.mono),
            "{label}: port stream decodes to the wrong geometry"
        );
    }
    Cell {
        label: label.to_string(),
        exact,
        c_moved: c_payload != c_stock_slice(stock, &c_payload),
        port_len: port.len(),
        c_len: c_payload.len(),
    }
}

/// `c_moved` helper kept explicit so the comparison reads the same way as
/// `run_cell`'s.
fn c_stock_slice<'a>(stock: &'a [u8], _payload: &[u8]) -> &'a [u8] {
    stock
}

/// Replay a covering array of strength `t` at one size context.
fn run_size_array(sc: &SizeCtx, t: usize, shard: usize, n_shards: usize, min_moved_pct: f64) {
    c::ref_init();
    let cell = sc.cell();
    let cctx = sc.ctx();
    let stock = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&sc.ctrls(&ToggleKnobs::default())));
    let stock_eff = Effective::resolve(&DEFAULT_ROW, &cctx);

    let array = cp::covering_array(t);
    let collapsed = cp::collapse(&array, &cctx);
    let rows: Vec<Row> = collapsed
        .representatives
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % n_shards == shard)
        .map(|(_, r)| r)
        .collect();
    assert!(!rows.is_empty(), "{}: shard {shard}/{n_shards} is empty", sc.tag);

    let mut cells = Vec::new();
    let mut skipped = 0usize;
    for row in &rows {
        if sc.skip_reason(row).is_some() {
            skipped += 1;
            continue;
        }
        let label = format!("{}_{}", sc.tag, cp::row_label(row));
        let r = run_size_cell(sc, &cell, &label, &cp::knobs_of(row), &stock);
        if Effective::resolve(row, &cctx) == stock_eff {
            assert!(
                !r.c_moved,
                "{label}: the collapse engine resolves this row to the STOCK \
                 effective config, but the C encoder produced different bytes \
                 — the Effective signature is under-refined at this geometry"
            );
        }
        cells.push(r);
    }

    let non_stock: Vec<&Cell> = cells.iter().filter(|c| !c.label.ends_with("_stock")).collect();
    let moved = non_stock.iter().filter(|c| c.c_moved).count();
    let moved_pct = if non_stock.is_empty() {
        100.0
    } else {
        100.0 * moved as f64 / non_stock.len() as f64
    };
    println!("{}", render(&cells, sc.tag, t, shard, n_shards, moved_pct));
    if skipped > 0 {
        println!(
            "  {}: {skipped} row(s) skipped — {}",
            sc.tag,
            sc.skip_reason(&rows.iter().copied().find(|r| sc.skip_reason(r).is_some()).unwrap())
                .unwrap()
        );
    }

    let open: Vec<&Cell> = cells.iter().filter(|c| !c.exact).collect();
    assert!(
        open.is_empty(),
        "{}: {} of {} SIZE-context cells are NOT byte-identical to real aomenc \
         — a knob combination diverges at this GEOMETRY where it is exact on \
         the 64/32/128-square SB64 grid. Offenders: {}",
        sc.tag,
        open.len(),
        cells.len(),
        open.iter()
            .map(|c| format!("{} (port {}B vs C {}B)", c.label, c.port_len, c.c_len))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(
        moved_pct >= min_moved_pct,
        "{}: only {moved_pct:.1}% of the {} non-stock rows changed the C \
         encoder's output (floor {min_moved_pct}%)",
        sc.tag,
        non_stock.len()
    );
}

/// The rows an expensive size context is reduced to: the FULL cross of the
/// axes in `config_perm::RECT_KILL_INTERACTION_SET` (2 x 3 x 2 x 2 = 24 legal
/// rows), every other axis at its default.
///
/// The reduction is argued, not assumed. The only size-derived state that is
/// live at speed 0 below 2160p is `use_square_partition_only_threshold`, whose
/// sole intra consumer is the rect-kill on `partition_rect_allowed`. An axis can
/// interact with that only by changing whether rectangular partitions exist
/// (`Rect`), whether the over-threshold block is reached at all (`MaxPart`
/// force-splits the root below the SB size), or which rect-derived types are
/// offered (`Ab`, `P1to4`, both gated on `partition_rect_allowed` at
/// partition_search.c:5166/5172/5181/5187). Every other axis composes with the
/// kill only THROUGH those four, and its composition with them is already
/// covered at full t=4 strength on the cheap contexts.
fn interaction_rows(full_maxpart: bool) -> Vec<Row> {
    let idx = |a: Axis| ALL_AXES.iter().position(|x| *x == a).unwrap();
    let mut out = Vec::new();
    let maxp_levels: &[u8] = if full_maxpart { &[0, 1, 2] } else { &[0, 1] };
    for rect in 0..2u8 {
        for &maxp in maxp_levels {
            for ab in 0..2u8 {
                for p14 in 0..2u8 {
                    let mut row = DEFAULT_ROW;
                    row[idx(Axis::Rect)] = rect;
                    row[idx(Axis::MaxPart)] = maxp;
                    row[idx(Axis::Ab)] = ab;
                    row[idx(Axis::P1to4)] = p14;
                    if cp::illegal_reason(&row).is_none() {
                        out.push(row);
                    }
                }
            }
        }
    }
    out
}

/// Replay only the rect-kill interaction set at one size context.
fn run_interaction_set(sc: &SizeCtx, full_maxpart: bool, shard: usize, n_shards: usize) {
    c::ref_init();
    let cell = sc.cell();
    let stock = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&sc.ctrls(&ToggleKnobs::default())));
    let rows: Vec<Row> = interaction_rows(full_maxpart)
        .into_iter()
        .filter(|r| sc.skip_reason(r).is_none())
        .enumerate()
        .filter(|(i, _)| i % n_shards == shard)
        .map(|(_, r)| r)
        .collect();
    assert!(!rows.is_empty(), "{}: empty interaction shard", sc.tag);
    let mut cells = Vec::new();
    for row in &rows {
        let label = format!("{}_ix_{}", sc.tag, cp::row_label(row));
        cells.push(run_size_cell(sc, &cell, &label, &cp::knobs_of(row), &stock));
    }
    let moved = cells.iter().filter(|c| c.c_moved).count();
    println!(
        "  {} rect-kill interaction set: {} rows, {} moved C's output, shard {shard}/{n_shards}",
        sc.tag,
        cells.len(),
        moved
    );
    let open: Vec<&Cell> = cells.iter().filter(|c| !c.exact).collect();
    assert!(
        open.is_empty(),
        "{}: {} of {} rect-kill interaction cells are NOT byte-identical to \
         real aomenc: {}",
        sc.tag,
        open.len(),
        cells.len(),
        open.iter()
            .map(|c| format!("{} (port {}B vs C {}B)", c.label, c.port_len, c.c_len))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Anti-vacuity: at least one row must move the C encoder, else the whole
    // interaction set is measuring nothing at this geometry.
    assert!(
        moved > 0,
        "{}: NO row of the rect-kill interaction set changed real aomenc's \
         output — the reduced context is vacuous",
        sc.tag
    );
}

// --- the gates -------------------------------------------------------------

// CLASS 4 at FULL t=4 strength — the frame-edge class has four KB-6 roots in it
// and interacts with every knob, so it earns the same strength as the aligned
// contexts. 187 rows x 199 ms = 37 s CPU, sharded 3 ways.
#[test]
fn size_t4_part68_s0() { run_size_array(&S_PART68, 4, 0, 3, 85.0) }
#[test]
fn size_t4_part68_s1() { run_size_array(&S_PART68, 4, 1, 3, 85.0) }
#[test]
fn size_t4_part68_s2() { run_size_array(&S_PART68, 4, 2, 3, 85.0) }

// CLASS 4 second overhang magnitude + CLASS 5 (one-dimension partial) + CLASS 9
// (frame smaller than an SB128 superblock) at t=2. Pairwise is the defensible
// floor for a context whose purpose is to re-witness a class already covered at
// t=4 under a different geometry parameter.
#[test]
fn size_t2_part96() { run_size_array(&S_PART96, 2, 0, 1, 85.0) }
#[test]
fn size_t2_part128x96() { run_size_array(&S_PART128X96, 2, 0, 1, 85.0) }
#[test]
fn size_t2_sb128_64() { run_size_array(&S_SB128_64, 2, 0, 1, 80.0) }

// CLASS 6 — the cheap SB128 class where the rect-kill is live. t=2 for breadth
// PLUS the full 24-row rect-kill interaction cross, so the reduction applied to
// the expensive >= 480p context is demonstrated to be sufficient at the same
// mechanism where a full cross is affordable.
#[test]
fn size_t2_sb128_128_s0() { run_size_array(&S_SB128_128, 2, 0, 2, 85.0) }
#[test]
fn size_t2_sb128_128_s1() { run_size_array(&S_SB128_128, 2, 1, 2, 85.0) }
#[test]
fn size_ix_sb128_128_s0() { run_interaction_set(&S_SB128_128, true, 0, 2) }
#[test]
fn size_ix_sb128_128_s1() { run_interaction_set(&S_SB128_128, true, 1, 2) }

// CLASS 7 — SB128 + partial SB, interaction set only (1.99 s/cell).
#[test]
fn size_ix_sb128_192_s0() { run_interaction_set(&S_SB128_192, false, 0, 2) }
#[test]
fn size_ix_sb128_192_s1() { run_interaction_set(&S_SB128_192, false, 1, 2) }

// CLASS 8 — the >= 480p class. 2.85 s/cell, so the interaction set runs with
// MaxPart reduced to {128, 64}: level 2 (32 px) force-splits the 128 root
// strictly harder than level 1 (64 px) already does, so it cannot expose a
// rect-kill behaviour level 1 does not.
#[test]
fn size_ix_sb128_576_s0() { run_interaction_set(&S_SB128_576, false, 0, 2) }
#[test]
fn size_ix_sb128_576_s1() { run_interaction_set(&S_SB128_576, false, 1, 2) }

/// THE VALIDITY ANSWER, computed rather than asserted: collapse a candidate
/// size list into its distinct size classes and report which are covered by the
/// pre-existing eight contexts, which are covered by the size contexts added
/// here, and which remain out of budget.
///
/// This test does NO encoding — it is the arithmetic that justifies the cell
/// count, and it fails if a class silently loses its representative.
#[test]
fn size_class_inventory_is_pinned() {
    // The pre-existing contexts (all SB64), plus every size context added here,
    // plus probe geometries that are expected to COLLAPSE into an existing
    // class (so the collapse is exercised, not merely claimed).
    let candidates: Vec<(&str, CellCtx)> = vec![
        ("32x32 sb64 (existing)", CellCtx { w: 32, h: 32, mono: false, sb_px: 64 }),
        ("64x64 sb64 (existing)", CellCtx { w: 64, h: 64, mono: false, sb_px: 64 }),
        ("128x128 sb64 (existing)", CellCtx { w: 128, h: 128, mono: false, sb_px: 64 }),
        ("512x512 sb64 (probe)", CellCtx { w: 512, h: 512, mono: false, sb_px: 64 }),
        ("68x68 sb64 (added)", CellCtx { w: 68, h: 68, mono: false, sb_px: 64 }),
        ("96x96 sb64 (added)", CellCtx { w: 96, h: 96, mono: false, sb_px: 64 }),
        ("196x196 sb64 (probe)", CellCtx { w: 196, h: 196, mono: false, sb_px: 64 }),
        ("128x96 sb64 (added)", CellCtx { w: 128, h: 96, mono: false, sb_px: 64 }),
        ("64x64 sb128 (added)", CellCtx { w: 64, h: 64, mono: false, sb_px: 128 }),
        ("128x128 sb128 (added)", CellCtx { w: 128, h: 128, mono: false, sb_px: 128 }),
        ("192x192 sb128 (added)", CellCtx { w: 192, h: 192, mono: false, sb_px: 128 }),
        ("512x512 sb128 (PINNED OPEN)", CellCtx { w: 512, h: 512, mono: true, sb_px: 128 }),
        ("576x576 sb128 (added)", CellCtx { w: 576, h: 576, mono: true, sb_px: 128 }),
        ("480x480 sb128 (PINNED OPEN)", CellCtx { w: 480, h: 480, mono: true, sb_px: 128 }),
        ("2160x2160 sb64 (OUT OF BUDGET)", CellCtx { w: 2160, h: 2160, mono: false, sb_px: 64 }),
    ];
    let ctxs: Vec<CellCtx> = candidates.iter().map(|(_, c)| *c).collect();
    let classes = size_class_partition(&ctxs, 0);

    let mut report = String::from(
        "\n=== SIZE CLASSES at speed 0 (config_perm::size_derived) ===\n\
         (a class = one distinct size-derived encoder state; sizes inside a \
         class are redundant with each other)\n",
    );
    for (i, (sd, members)) in classes.iter().enumerate() {
        let names: Vec<&str> = candidates
            .iter()
            .filter(|(_, c)| members.contains(c))
            .map(|(n, _)| *n)
            .collect();
        let _ = writeln!(
            report,
            "  class {:>2}: sb{:<3} full_sb={} multi=({},{}) partial=({},{}) \
             rect_kill={} dmin_part={} tx_stats={}  <- {}",
            i + 1,
            sd.sb_px,
            sd.full_sb_block as u8,
            sd.multi_sb_cols as u8,
            sd.multi_sb_rows as u8,
            sd.partial_sb_x as u8,
            sd.partial_sb_y as u8,
            sd.rect_kill_reachable as u8,
            sd.default_min_partition_size,
            sd.prune_tx_type_using_stats,
            names.join(" | ")
        );
    }
    println!("{report}");

    // 1. The collapse is real, not decorative: 480x480 SB64 must land in the
    //    SAME class as 128x128 SB64. At speed 0 the only thing >= 480p changes
    //    is `use_square_partition_only_threshold`, and at SB64 that threshold
    //    has no block large enough to act on — so a 12.6 s/cell 480p SB64
    //    context would buy exactly nothing over the 1.0 s/cell 128x128 one.
    // (512 and 256 are both SB-aligned multi-SB frames, so >= 480p is the ONLY
    // property that differs between them — the clean discriminator.)
    let big64 = CellCtx { w: 512, h: 512, mono: false, sb_px: 64 };
    let mid64 = CellCtx { w: 256, h: 256, mono: false, sb_px: 64 };
    assert_eq!(
        size_derived(&big64, 0),
        size_derived(&mid64, 0),
        ">= 480p must collapse into the multi-SB class at SB64 and speed 0 \
         (the >= 480p threshold has no reachable consumer there)"
    );
    assert_ne!(
        cp::sq_only_threshold_allintra(&big64, 0),
        cp::sq_only_threshold_allintra(&mid64, 0),
        "the collapse above must be a claim about REACHABILITY, not about the \
         threshold being equal — the raw sf value does differ"
    );

    // 2. ... and it is NOT vacuous: the same size pair SPLITS at SB128, which
    //    is exactly the gap `size_ix_sb128_480_*` closes.
    let big128 = CellCtx { w: 512, h: 512, mono: false, sb_px: 128 };
    let mid128 = CellCtx { w: 256, h: 256, mono: false, sb_px: 128 };
    assert_ne!(
        size_derived(&big128, 0),
        size_derived(&mid128, 0),
        ">= 480p must SPLIT from sub-480p at SB128 — the rect-kill is \
         reachable there and the threshold decides whether it fires"
    );
    assert!(size_derived(&mid128, 0).rect_kill_reachable);
    assert!(!size_derived(&big128, 0).rect_kill_reachable);

    // 3. Every class in the candidate list has a representative that is either
    //    an existing context, an added context, or explicitly out of budget.
    for (sd, members) in &classes {
        let names: Vec<&str> = candidates
            .iter()
            .filter(|(_, c)| members.contains(c))
            .map(|(n, _)| *n)
            .collect();
        assert!(
            names.iter().any(|n| {
                n.contains("existing")
                    || n.contains("added")
                    || n.contains("OUT OF BUDGET")
                    || n.contains("PINNED OPEN")
            }),
            "size class {sd:?} has no gated representative — it is covered by \
             nothing: {names:?}"
        );
    }

    // 4. The count is pinned, so adding a size-dependent derivation to
    //    `size_derived` (or losing one) fails here instead of silently changing
    //    what "the array is size-valid" means.
    assert_eq!(
        classes.len(),
        11,
        "the size-class count moved; re-derive the coverage argument in \
         docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md before repinning"
    );

    // 5. Speed is the axis that turns the >= 480p threshold from a partition
    //    knob into a TRANSFORM knob. Pinned so that raising the array's speed
    //    (currently 0 everywhere) cannot silently leave `prune_tx_type_using_
    //    stats` unexercised.
    let s0 = size_derived(&big64, 0);
    let s2 = size_derived(&big64, 2);
    let s4 = size_derived(&big64, 4);
    assert_eq!(s0.prune_tx_type_using_stats, 0);
    assert_eq!(s2.prune_tx_type_using_stats, 1);
    assert_eq!(s4.prune_tx_type_using_stats, 2);
    assert_eq!(
        size_derived(&mid64, 4).prune_tx_type_using_stats,
        0,
        "sub-480p keeps the stats prune off at every speed"
    );
}

/// FINDINGS, pinned open — BOTH found by the size axis, both size-attributed,
/// neither swept into a gate. The test is self-promoting: when the port stops
/// diverging (or stops asserting), it FAILS and must be promoted.
///
/// ### Finding A — `--sb-size=128` x `--max-partition-size=32` trips a port
/// assertion that C's own code contradicts
///
/// C restores the partition context CONDITIONALLY at the end of the SPLIT stage
/// (partition_search.c:4643-4647):
///
/// > `// Restore the context for the following cases:`
/// > `// 1) Current block size not more than maximum partition size ...`
/// > `// 2) Current block size same as superblock size ...`
/// > `if (bsize <= x->sb_enc.max_partition_size || bsize == cm->seq_params->sb_size)`
/// > `  av1_restore_context(x, x_ctx, mi_row, mi_col, bsize, av1_num_planes(cm));`
///
/// The port restores UNCONDITIONALLY and encodes C's condition as a
/// `debug_assert!` instead (`aom-encode/src/partition_pick.rs:3055-3057`,
/// commented "always true here"). It is NOT always true: it fails exactly when a
/// block size sits strictly between the max-partition cap and the superblock
/// size. At SB64 that window is empty — the top block is `bsize == sb_size` and
/// everything below is `<= cap` for every legal cap — which is why the eight
/// pre-existing contexts never saw it. At SB128 with a 32 px cap,
/// `bsize == BLOCK_64X64` satisfies neither clause. In a debug-assertions build
/// the port panics; without them it performs a restore C SKIPS, which is a
/// silent state divergence.
///
/// ### Finding B — three open near-ties on >= 480p SB128 monochrome cq63,
/// SIZE-SURFACED but NOT size-class-attributed
///
/// Mirror-tiled `av1-1-b8-00-quantizer-00`, monochrome, cq63, speed-0
/// `--sb-size=128`. The middle column is the size class
/// (`config_perm::size_derived`), and it is what makes the attribution honest:
///
/// | frame | class | knob row | verdict |
/// |---|---|---|---|
/// | 480x480 | >= 480p, partial | `--enable-1to4-partitions=0` | **DIVERGE** port 849 B vs C 834 B |
/// | 480x480 | >= 480p, partial | `--enable-ab-partitions=0 --enable-1to4-partitions=0` | **DIVERGE** port 846 B vs C 829 B |
/// | 480x480 | >= 480p, partial | stock | exact |
/// | **576x576** | **>= 480p, partial (SAME CLASS)** | both of the above, and stock | **exact** |
/// | 512x512 | >= 480p, aligned | `--enable-ab-partitions=0` | **DIVERGE** port 929 B vs C 936 B |
/// | **640x640** | **>= 480p, aligned (SAME CLASS)** | `--enable-ab-partitions=0` | **exact** |
/// | 448x448 | sub-480p, partial | both p14 rows, and stock | exact |
/// | 256x256 / 384x384 | sub-480p, aligned | `ab0`, `p140`, stock | exact |
///
/// **The first attribution attempted here was WRONG and the class-mates
/// refuted it.** "480x480 diverges and 448/512 do not" reads like "the
/// intersection of `>= 480p` and a partial superblock is broken" — until
/// 576x576, which is in the SAME size class as 480x480, comes out exact on the
/// identical knob rows; and 640x640, the class-mate of 512x512, likewise. A
/// property that holds for one member of an equivalence class and fails for
/// another is not a property of the class. These are per-cell RD near-ties (the
/// KB-10/KB-12 "cheaper RD decision" signature: same order of magnitude, a
/// handful of bytes either way, appearing and disappearing with content
/// statistics), surfaced because the size axis encoded this content at sizes
/// the harness had never reached — the same shape as the pre-existing
/// `mono_vector_open_divergences_pinned` finding, which is a CONTENT finding.
///
/// They are therefore pinned, not gated, and the gated `>= 480p` contexts use
/// the clean class-mates (576x576) so that a size gate never rests on a
/// near-tie.
#[test]
fn size_axis_open_divergences_pinned() {
    c::ref_init();

    // --- Finding A: CLOSED 2026-07-30 (KB-18 fixed), promoted to a gate ----
    // The port used to `debug_assert!` C's restore CONDITION and then restore
    // unconditionally; `partition_pick.rs` now takes the restore only when
    // `bsize <= max_partition_size || bsize == sb_size` (partition_search.c:
    // 4645-4646). This arm is the direct byte gate for the geometry that
    // reaches the false branch: SB128 with a 32 px max-partition cap, where
    // BLOCK_64X64 satisfies neither clause. It is ALSO covered at full
    // strength by every SB128 size context now that `SizeCtx::skip_reason` is
    // empty; keeping the focused cell here makes the regression legible.
    let a_cell = S_SB128_128.cell();
    let a_knobs = ToggleKnobs {
        max_partition_size_px: 32,
        ..Default::default()
    };
    let a_ctrls = S_SB128_128.ctrls(&a_knobs);
    let c_tu = a_cell.c_encode_ctrls(&a_ctrls);
    assert!(!c_tu.is_empty(), "C must accept --sb-size=128 --max-partition-size=32");
    let a_real = EncodeCell::frame_obu_payload(&c_tu);
    let a_port = a_cell.port_encode_with(&c_tu, &a_knobs);
    println!(
        "  finding A (KB-18) sb128 x maxp32: port {} B vs C {} B ({})",
        a_port.len(),
        a_real.len(),
        if a_port == a_real { "MATCH" } else { "DIVERGE" }
    );
    assert_eq!(
        a_port, a_real,
        "KB-18 REGRESSED: --sb-size=128 x --max-partition-size=32 is no longer \
         byte-identical to real aomenc. The SPLIT-stage restore \
         (partition_pick.rs, partition_search.c:4645-4646) must stay \
         conditional on `bsize <= max_partition_size || bsize == sb_size`."
    );

    // --- Finding B ---------------------------------------------------------
    let mut open = Vec::new();
    let p14 = ToggleKnobs { enable_1to4_partitions: false, ..Default::default() };
    let ab0 = ToggleKnobs { enable_ab_partitions: false, ..Default::default() };
    let ab0p14 = ToggleKnobs {
        enable_ab_partitions: false,
        enable_1to4_partitions: false,
        ..Default::default()
    };
    let cells = [
        (&S_SB128_480_OPEN, "480m_p140", &p14),
        (&S_SB128_480_OPEN, "480m_ab0-p140", &ab0p14),
        (&S_SB128_512_OPEN, "512m_ab0", &ab0),
    ];
    for (sc, tag, knobs) in cells {
        let cell = sc.cell();
        let tu = cell.c_encode_ctrls(&sc.ctrls(knobs));
        let real = EncodeCell::frame_obu_payload(&tu);
        let port = cell.port_encode_with(&tu, knobs);
        println!(
            "  finding B {tag}: port {} B vs C {} B ({})",
            port.len(),
            real.len(),
            if port == real { "MATCH" } else { "DIVERGE" }
        );
        if port == real {
            open.push(tag);
        }
    }
    assert!(
        open.is_empty(),
        "FINDING B HAS CLOSED for {open:?}. These are pinned as OPEN near-ties; \
         if they now match, re-measure the whole table (including the 576/640 \
         class-mate controls) and either promote the cell into a gated size \
         context or delete its row here."
    );
}

/// TEETH — the proof that the size gap this section closes was REAL, in the
/// asymmetric form the decoder track's `delta_lf` demonstration used.
///
/// The size-gated derivation under test is the ONLY framesize-dependent speed
/// feature that is live at speed 0 below 2160p:
///
/// > `if (is_480p_or_larger) { sf->part_sf.use_square_partition_only_threshold = BLOCK_128X128; }`
/// > `else                   { sf->part_sf.use_square_partition_only_threshold = BLOCK_64X64;   }`
/// > — libaom `av1/encoder/speed_features.c:175-183`
///
/// ported at `aom-encode/src/partition_pick.rs:2450` and consumed by the
/// square-only rect-kill (`partition_search.c:5700`, port
/// `partition_pick.rs:2593`).
///
/// **Demonstrated 2026-07-30** by dropping the `>= 480p` arm — i.e. replacing
/// `partition_pick.rs:2451` with `let mut t: usize = 12;` — and re-running:
///
/// * the added `>= 480p` SB128 cell FAILS
///   > `TEETH 480m_cq63_sb128_stock  exact=false port 874B real 854B`
/// * every control stays GREEN — sub-480p SB128, both SB64 sizes, and the two
///   knob rows that make the kill moot:
///   > `TEETH 480m_cq63_sb128_rect0 exact=true`, `480m_cq63_sb128_maxp64 exact=true`,
///   > `128_cq32_sb128_stock exact=true`, `128_cq63_sb128_stock exact=true`,
///   > `64_cq32_sb64_stock exact=true`, `128_cq32_sb64_stock exact=true`
/// * and the whole pre-existing gate is untouched:
///   > `test result: ok. 32 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 32.30s`
///
/// 2,617 cells that cannot see a broken `>= 480p` threshold, against one added
/// cell that fails on it, is the asymmetry — the size gap was real.
///
/// What this test does at RUNTIME is the half that can run unperturbed: it
/// asserts the rect-kill is REACHABLE in the gated SB128 contexts and DEAD in
/// every pre-existing one, so a future refactor that quietly makes the added
/// contexts SB64 (or drops the SB128 control) turns the teeth vacuous and fails
/// here instead of passing silently.
#[test]
fn size_axis_teeth_are_real() {
    for ctx in [
        CellCtx { w: 64, h: 64, mono: false, sb_px: 64 },
        CellCtx { w: 32, h: 32, mono: false, sb_px: 64 },
        CellCtx { w: 128, h: 128, mono: false, sb_px: 64 },
        CellCtx { w: 68, h: 68, mono: false, sb_px: 64 },
    ] {
        assert!(
            !size_derived(&ctx, 0).rect_kill_reachable,
            "{ctx:?}: the square-only rect-kill must be structurally dead at \
             SB64 — if it is not, the pre-existing array's immunity to the \
             >= 480p threshold (the teeth control) no longer holds"
        );
    }
    let kill_on = S_SB128_128.ctx();
    let kill_off = S_SB128_576.ctx();
    assert!(
        size_derived(&kill_on, 0).rect_kill_reachable,
        "the sub-480p SB128 context must REACH the rect-kill"
    );
    assert!(
        !size_derived(&kill_off, 0).rect_kill_reachable,
        "the >= 480p SB128 context must NOT reach the rect-kill — that \
         difference IS the derivation the teeth perturb"
    );
    assert_ne!(
        cp::sq_only_threshold_allintra(&kill_on, 0),
        cp::sq_only_threshold_allintra(&kill_off, 0),
        "the two SB128 contexts must straddle the >= 480p threshold"
    );
    // And the contexts must actually be SB128 — an SB64 fallback would make
    // both sides of the teeth vacuous.
    assert!(S_SB128_128.sb128 && S_SB128_576.sb128 && S_SB128_192.sb128 && S_SB128_64.sb128);
}

/// BUDGET ACCOUNTING for the size axis — no encoding, just the arithmetic that
/// justifies the strength chosen per context, pinned so that adding an
/// expensive context (or raising a strength) fails here instead of quietly
/// costing the suite a minute.
///
/// `ms_per_cell` is measured, not estimated
/// (`benchmarks/config_perm_size_axis_2026-07-30.tsv`, `[cost]` section), on a
/// 12-core M4 shared with two concurrent agents — so the figures are upper
/// bounds and the CPU total below is conservative.
#[test]
fn size_axis_budget_is_accounted() {
    // (context, strength "t" or 0 for interaction-set-only, rows actually run)
    let plan: Vec<(&SizeCtx, &str, usize)> = vec![
        (&S_PART68, "t=4", cp::covering_array(4).len()),
        (&S_PART96, "t=2", cp::covering_array(2).len()),
        (&S_PART128X96, "t=2", cp::covering_array(2).len()),
        (&S_SB128_64, "t=2", cp::covering_array(2).len()),
        (&S_SB128_128, "t=2 + full rect-kill cross", cp::covering_array(2).len() + 24),
        (&S_SB128_192, "rect-kill cross", 16),
        (&S_SB128_576, "rect-kill cross", 16),
    ];
    assert_eq!(
        plan.len(),
        ALL_SIZE_CONTEXTS.len(),
        "every size context must appear in the budget plan"
    );
    let mut total_cells = 0usize;
    let mut total_ms = 0u64;
    let mut report = String::from("\n=== size-axis budget ===\n");
    for (sc, strength, rows) in &plan {
        assert!(
            ALL_SIZE_CONTEXTS.iter().any(|c| c.tag == sc.tag),
            "{} is budgeted but not in ALL_SIZE_CONTEXTS",
            sc.tag
        );
        // SB128 contexts skip the --max-partition-size=32 rows (finding A), so
        // the budget over-counts them rather than under-counting.
        total_cells += rows;
        total_ms += *rows as u64 * sc.ms_per_cell as u64;
        let _ = writeln!(
            report,
            "  {:<12} {:>4}x{:<4} sb{:<3} {:<26} {:>4} rows x {:>5} ms",
            sc.tag, sc.w, sc.h, if sc.sb128 { 128 } else { 64 }, strength, rows, sc.ms_per_cell
        );
    }
    let _ = writeln!(
        report,
        "  TOTAL {total_cells} cells, {:.1} s CPU (upper bound; libtest spreads \
         it across shards)",
        total_ms as f64 / 1000.0
    );
    println!("{report}");
    assert!(
        total_ms <= 200_000,
        "the size axis budget is {} s CPU — over the 200 s ceiling this section \
         was designed to; drop a strength or a context rather than letting the \
         suite drift",
        total_ms / 1000
    );
}


// ===========================================================================
// The SPEED axis (added 2026-07-30)
// ===========================================================================
//
// Everything above this line runs at ONE speed. `Ctx::cell`, `Content::cell`
// and `SizeCtx::cell` all call `EncodeCell::real_content(.., 0)`, so all 2,910
// cells are `--cpu-used=0`. PARITY.md §A separately gates `--cpu-used 0..9`,
// but each speed on its OWN grid with STOCK knobs — the knob axes and the speed
// axis had never been crossed.
//
// That is not a cosmetic hole. The encoder is STRUCTURALLY different across the
// range, so "replay the array at more speeds" is not a uniform expansion:
//
//   * speeds 0-6  RD partition search (`rd_pick_partition`);
//   * speed  7    `VAR_BASED_PARTITION` fixed tree + `av1_rd_use_partition`
//                 (`pack.rs:1474`, speed_features.c:571) — KB-11;
//   * speeds 8-9  nonrd PICKMODE, `av1_nonrd_pick_intra_mode`
//                 (`partition_pick.rs:4569`, partition_search.c:2960) — KB-12.
//
// A knob that steers the RD search can therefore be inert, or mean something
// different, at 7+. This section MEASURES which axes move with speed rather
// than assuming, exactly as the CONTENT section measured content-sensitivity,
// and sizes the expansion by that measurement.
//
// The speed-gated derivations that ALSO need a framesize are enumerated in
// `cp::SPEED_X_FRAMESIZE_DERIVATIONS` (four, each with its libaom line). Only
// one is live on this harness's all-intra KEY path at any speed —
// `prune_tx_type_using_stats` — and closing it is what `speed_size_txstats_*`
// does: it is the exact cell the SIZE section could not reach ("this array is
// speed-0 everywhere, so the four framesize x speed interactions are outside it
// by construction").

use aom_bench::config_perm::{ALL_SPEEDS, axis_level_dead_at_speed, speed_sf_classes};

/// Provenance stamp for the `--ignored` evidence sweeps.
///
/// These TSVs carry a DATE in their filename and in their own header, but both
/// are static strings in the writer — regenerating the file months later
/// reproduces the same claimed date over different content. And because the
/// sweeps are `--ignored`, a committed file can sit for weeks while the very
/// divergences it lists get fixed (exactly what happened between 2026-07-30 and
/// 2026-08-01: KB-21/23/26 closed rows these files still listed, and the drift
/// was only noticed because an unrelated agent happened to run the sweep).
///
/// Stamping the commit makes staleness self-evident: if this line does not match
/// `git rev-parse --short HEAD`, the rows below may already be closed. Same
/// principle as the `build_commit` rule for generated data.
fn evidence_provenance() -> String {
    let head = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "# generated at commit {head}. If that differs from `git rev-parse --short HEAD`,\n\
         # treat every divergence below as possibly-already-fixed and re-run the sweep.\n"
    )
}

// ---------------------------------------------------------------------------
// 7a. Contexts
// ---------------------------------------------------------------------------

/// The primary speed context's content: bd8 4:2:0 photographic, 64x64, one
/// superblock, SB64 — the SAME cell as [`C_64_CQ32`], so everything this section
/// reports is attributable to SPEED and nothing else. Its STOCK encode is
/// byte-identical to real aomenc at every one of the ten speeds
/// ([`speed_axis_teeth_are_real`] asserts that; it is the control the whole
/// section rests on).
const SPEED_VECTOR: &str = "av1-1-b8-01-size-64x64";
const SPEED_CQ: i32 = 32;

fn speed_cell(speed: i32) -> EncodeCell {
    EncodeCell::real_content("spd", SPEED_VECTOR, None, SPEED_CQ, speed)
}

/// One speed context: the covering-array strength this speed earns, and why.
///
/// Strength is set by the MEASURED per-speed axis liveness in
/// [`SPEED_LIVE_LEVELS`], not uniformly: at `--cpu-used=9` only 4 of the 26 axis
/// levels still change real aomenc's own output, so a 187-row t=4 array there
/// would be ~85% vacuous. Speeds where the encoder's structure changes (2 = the
/// first ML/tx-stats tier, 4 = winner-mode + chroma prune, 6 = the last
/// RD-search speed) earn more.
struct SpeedCtx {
    speed: i32,
    /// Covering-array strength; the singleton matrix runs at every speed.
    t: usize,
    /// Shard count for this speed's array test.
    shards: usize,
    /// MEASURED cost of one cell (C encode + port encode) on the reference box.
    ms_per_cell: u32,
    /// Anti-vacuity floor for this speed's array — necessarily lower at high
    /// speeds, where the encoder ignores more of the configuration. The floor
    /// is a floor, not a target: the measured value is printed by every run.
    min_moved_pct: f64,
}

/// Every speed, with the strength it earns. Ten contexts, no collapse — see
/// [`aom_bench::config_perm::SPEED_SF_EQUALITY_IS_NOT_A_COLLAPSE`] for why
/// `{7,8,9}` do NOT merge despite an identical resolved `SpeedFeatures`.
const SPEED_CONTEXTS: &[SpeedCtx] = &[
    SpeedCtx { speed: 0, t: 2, shards: 1, ms_per_cell: 114, min_moved_pct: 75.0 },
    SpeedCtx { speed: 1, t: 2, shards: 1, ms_per_cell: 56, min_moved_pct: 75.0 },
    SpeedCtx { speed: 2, t: 4, shards: 3, ms_per_cell: 48, min_moved_pct: 75.0 },
    SpeedCtx { speed: 3, t: 2, shards: 1, ms_per_cell: 42, min_moved_pct: 75.0 },
    SpeedCtx { speed: 4, t: 3, shards: 2, ms_per_cell: 30, min_moved_pct: 55.0 },
    SpeedCtx { speed: 5, t: 2, shards: 1, ms_per_cell: 27, min_moved_pct: 55.0 },
    SpeedCtx { speed: 6, t: 3, shards: 1, ms_per_cell: 16, min_moved_pct: 40.0 },
    SpeedCtx { speed: 7, t: 3, shards: 1, ms_per_cell: 3, min_moved_pct: 30.0 },
    SpeedCtx { speed: 8, t: 3, shards: 1, ms_per_cell: 3, min_moved_pct: 30.0 },
    SpeedCtx { speed: 9, t: 2, shards: 1, ms_per_cell: 2, min_moved_pct: 15.0 },
];

fn speed_ctx(speed: i32) -> &'static SpeedCtx {
    SPEED_CONTEXTS
        .iter()
        .find(|c| c.speed == speed)
        .expect("every speed 0..=9 has a context")
}

// ---------------------------------------------------------------------------
// 7b. The speed-sensitivity matrix — which axes move with speed
// ---------------------------------------------------------------------------

/// Every singleton axis level, as `(axis index, level)`, skipping the levels the
/// speed makes unreachable ([`axis_level_dead_at_speed`]) and the ones no legal
/// row can carry alone.
fn singleton_levels(speed: i32) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    for (i, ax) in ALL_AXES.iter().enumerate() {
        for l in 1..ax.n_levels() as u8 {
            let mut row = DEFAULT_ROW;
            row[i] = l;
            if cp::illegal_reason_at_speed(&row, speed).is_some() {
                continue;
            }
            if axis_level_dead_at_speed(*ax, l, speed).is_some() {
                continue;
            }
            out.push((i, l));
        }
    }
    out
}

/// OPEN, pinned: `(speed, singleton row label)` cells on the primary context
/// where the port is NOT byte-identical to real aomenc.
///
/// Measured 2026-07-30 over the full 10 speeds x 26 axis levels grid, re-pinned
/// three times as roots closed, and **EMPTY since 2026-08-02**: the port is
/// byte-identical to real aomenc on every single-axis perturbation at every
/// speed 0..9 on the primary context.
///
/// Self-promoting: a cell that starts matching fails here (re-pin, and unpin the
/// level from that speed's covering array in [`remap_open_levels`]); a cell that
/// starts diverging is a regression.
const SPEED_OPEN_SINGLETONS: &[(i32, &str)] = &[
    // `(4, "dir0")` and `(4, "rtx0")` closed 2026-07-30 with the KB-21
    // `early_term_after_none_split` root; `(4, "flip0")` and `(5, "minp16")`
    // closed 2026-07-31 with KB-21 root #2; `(8, "rtxs1")` and `(8, "trel2")`
    // closed 2026-08-02 with KB-12's `aom_hadamard_lp_8x8` transpose (both were
    // read as the "cheaper RD decision" near-tie signature, which is exactly
    // what a dropped transpose in the nonrd estimate arm looks like — see
    // `nonrd_block_yrd_lp_diff.rs`). Emptying this list also BROADENS the
    // speed-8 covering array, because `remap_open_levels` no longer folds
    // `rtxs1` / `trel2` back to their defaults there.
];

/// OPEN, pinned: `(speed, covering-array row label)` knob COMBINATIONS that are
/// not byte-identical to real aomenc at that speed, on the primary context.
///
/// These are the cells this section exists to find: every one of them is a row
/// whose individual axis levels are byte-exact alone at this speed (the
/// singleton matrix above gates that) and byte-exact in combination at speed 0
/// (the 2,910-cell array above gates that) — so they diverge only where the knob
/// axis and the speed axis MEET.
///
/// Measured 2026-07-30: 3 rows at `--cpu-used=4` (of 63) and 5 at `--cpu-used=8`
/// (of 63), re-measured the same day after the KB-21 root landed -> 2 at
/// `--cpu-used=4` on a BROADER array (see the cpu-4 note below) and the 5 at
/// `--cpu-used=8` unchanged, as expected: speed 8 is the nonrd PICKMODE path,
/// which never runs `rd_pick_partition`, so the KB-21 root cannot reach it;
/// speeds 0, 1, 2, 3, 5, 6, 7 and 9 are clean at their gated strength,
/// including the 187-row t=4 array at speed 2. The signature is uniform and is
/// the KB-10/KB-12 "cheaper RD decision" near-tie: port payloads 0-4 bytes short
/// of C's. Speed 4 is the winner-mode / multi-winner-mode tier
/// (`multi_winner_mode_type=2`, `prune_chroma_modes_using_luma_winner`,
/// `fast_intra_tx_type_search=2`) and speed 8 is nonrd PICKMODE (KB-12) — the two
/// places where the SEARCH STRUCTURE, not just its thresholds, changes.
///
/// Every speed-8 row carried `dir0` or `dtxo1` or both, i.e. a narrowed luma
/// tx-type/mode set feeding the nonrd intra pickmode. That was recorded as "a
/// lead, not a root cause", and it was a good lead: the root (2026-08-02) is
/// that `hadamard_lp_8x8` dropped the trailing transpose C performs at
/// `aom_dsp/avg.c:232-236`, so the nonrd estimate arm's `eob` — its one
/// order-sensitive output — drifted. Narrowing the mode/tx-type set changes how
/// often that drift is decisive, which is why these rows and no others.
///
/// Self-promoting: a row that starts matching fails here, and so does a row that
/// starts diverging.
const SPEED_OPEN_COMBINATIONS: &[(i32, &str)] = &[
    // cpu-4 went EMPTY 2026-07-31 with KB-21 root #2 (the `prune_txk_type`
    // eob==0 est-rd ordering + the SATD trellis-skip QUANT_B switch); cpu-8
    // went EMPTY 2026-08-02 with KB-12's `aom_hadamard_lp_8x8` transpose, on a
    // BROADER array than the one the five rows were measured on (emptying
    // SPEED_OPEN_SINGLETONS un-remaps `rtxs1` / `trel2` at speed 8). Every knob
    // COMBINATION in every speed's array is now byte-identical to real aomenc.
];

/// The axis levels that still change REAL AOMENC's own output at each speed, on
/// the primary 64x64 cq32 context — i.e. the coverage a cell at that speed can
/// actually buy. PINNED, because this table is the whole argument for the
/// per-speed strengths in [`SPEED_CONTEXTS`].
///
/// Measured 2026-07-30. The count decays monotonically with speed
/// (20/20/20/20/17/18/14/13/13/4 live out of 26/26/26/26/26/26/26/25/24/24
/// reachable levels): the faster the encoder, the more of the configuration it
/// ignores, so an unreduced array at speed 9 would spend ~85% of its cells on
/// rows real aomenc cannot react to.
const SPEED_LIVE_LEVELS: &[(i32, &[&str])] = &[
    (0, &["rect0", "ab0", "p140", "minp8", "minp16", "smth0", "paeth0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "rtx0", "flip0", "dtxo1", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (1, &["rect0", "ab0", "p140", "minp8", "minp16", "smth0", "paeth0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "rtx0", "flip0", "dtxo1", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (2, &["rect0", "ab0", "p140", "minp8", "minp16", "smth0", "paeth0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "rtx0", "flip0", "dtxo1", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (3, &["rect0", "ab0", "p140", "minp8", "minp16", "smth0", "paeth0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "rtx0", "flip0", "dtxo1", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (4, &["rect0", "p140", "minp8", "minp16", "smth0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "rtx0", "flip0", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (5, &["rect0", "minp8", "minp16", "smth0", "paeth0", "cfl0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "rtx0", "flip0", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (6, &["rect0", "minp16", "smth0", "dir0", "adlt0", "fint0", "edgf0", "rtx0", "flip0", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (7, &["smth0", "paeth0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "flip0", "rtxs1", "txss0", "cdf0", "trel1", "trel2"]),
    (8, &["smth0", "paeth0", "dir0", "diag0", "adlt0", "fint0", "edgf0", "flip0", "rtxs1", "cdf0", "trel1", "trel2"]),
    (9, &["fint0", "rtxs1", "cdf0", "trel1"]),
];

fn live_levels_at(speed: i32) -> BTreeSet<String> {
    SPEED_LIVE_LEVELS
        .iter()
        .find(|(s, _)| *s == speed)
        .map(|(_, v)| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

/// Replay every singleton axis level at one speed on the primary context.
///
/// Returns `(divergent labels, live labels, cells run)`. Every cell asserts
/// byte-identity via [`run_cell`]'s decode fallback contract; the verdicts are
/// compared against the pins by [`check_speed_shard`].
fn run_speed_matrix(speeds: &[i32]) -> (BTreeSet<String>, BTreeMap<i32, BTreeSet<String>>, usize) {
    c::ref_init();
    let mut divergent = BTreeSet::new();
    let mut live: BTreeMap<i32, BTreeSet<String>> = BTreeMap::new();
    let mut cells = 0usize;
    let mut report = String::new();
    for &speed in speeds {
        let cell = speed_cell(speed);
        let stock = c_stock_payload(&cell);
        // Harness-faithfulness control: the stock encode of this content must be
        // byte-exact at THIS speed, else nothing measured here is attributable
        // to a knob.
        let base = run_cell(&cell, &format!("s{speed}_stock"), &ToggleKnobs::default(), &stock);
        assert!(
            base.exact,
            "s{speed}: the STOCK encode of the primary speed context is NOT \
             byte-identical to real aomenc — that is a plain speed-envelope \
             regression, not a knob-vs-speed interaction"
        );
        cells += 1;
        let mut here = Vec::new();
        for (i, l) in singleton_levels(speed) {
            let mut row = DEFAULT_ROW;
            row[i] = l;
            let label = cp::row_label(&row);
            let r = run_cell(&cell, &format!("s{speed}_{label}"), &cp::knobs_of(&row), &stock);
            cells += 1;
            if !r.exact {
                divergent.insert(format!("s{speed}/{label}"));
                here.push(format!("{label}({}/{})", r.port_len, r.c_len));
            }
            if r.c_moved {
                live.entry(speed).or_default().insert(label);
            }
        }
        let _ = writeln!(
            report,
            "  cpu-used={speed:<2} {:>2} of {:>2} reachable levels move real aomenc; \
             {} divergent [{}]",
            live.get(&speed).map_or(0, |s| s.len()),
            singleton_levels(speed).len(),
            here.len(),
            here.join(", ")
        );
    }
    println!("\n=== speed-sensitivity matrix ({cells} cells) ===\n{report}");
    (divergent, live, cells)
}

/// Assert one shard of the speed matrix matches the pins exactly.
fn check_speed_shard(speeds: &[i32]) {
    let (divergent, live, cells) = run_speed_matrix(speeds);
    assert!(cells > 0, "empty speed shard");
    let expected: BTreeSet<String> = SPEED_OPEN_SINGLETONS
        .iter()
        .filter(|(s, _)| speeds.contains(s))
        .map(|(s, l)| format!("s{s}/{l}"))
        .collect();
    assert_eq!(
        divergent, expected,
        "the SPEED-sensitivity matrix MOVED. A cell that started matching means \
         a speed x knob near-tie closed — re-pin SPEED_OPEN_SINGLETONS and unpin \
         the level from that speed's array in remap_open_levels. A cell that \
         started diverging is a regression."
    );
    for &speed in speeds {
        assert_eq!(
            live.get(&speed).cloned().unwrap_or_default(),
            live_levels_at(speed),
            "s{speed}: the set of axis levels that move REAL AOMENC changed. \
             This table is the argument for this speed's covering-array strength \
             in SPEED_CONTEXTS — re-measure the strength before re-pinning."
        );
    }
}

#[test]
fn speed_sensitivity_s0() {
    check_speed_shard(&[0, 1, 2, 3]);
}
#[test]
fn speed_sensitivity_s1() {
    check_speed_shard(&[4, 5, 6]);
}
#[test]
fn speed_sensitivity_s2() {
    check_speed_shard(&[7, 8, 9]);
}

// ---------------------------------------------------------------------------
// 7c. The covering array, replayed per speed
// ---------------------------------------------------------------------------

/// Map any axis level this speed has pinned open ([`SPEED_OPEN_SINGLETONS`]) or
/// dead ([`axis_level_dead_at_speed`]) back to its DEFAULT level.
///
/// Level-granular rather than axis-granular on purpose: pinning `trel=2` open at
/// speed 8 must not cost the array its `trel=1` coverage. Forcing individual
/// cells of a covering array to a constant leaves every t-tuple among the
/// remaining (axis, level) pairs covered, and the pinned levels are covered
/// standalone by the matrix above — the same treatment `dtxo` gets on screen
/// content and `--use-intra-dct-only` gets globally.
fn remap_open_levels(row: &Row, speed: i32) -> Row {
    let mut r = *row;
    for (i, ax) in ALL_AXES.iter().enumerate() {
        if r[i] == 0 {
            continue;
        }
        let mut probe = DEFAULT_ROW;
        probe[i] = r[i];
        let label = cp::row_label(&probe);
        if SPEED_OPEN_SINGLETONS.iter().any(|(s, l)| *s == speed && *l == label)
            || axis_level_dead_at_speed(*ax, r[i], speed).is_some()
        {
            r[i] = 0;
        }
    }
    r
}

/// Replay a covering array at one speed on the primary context.
///
/// Same four-part gate as [`run_array`]: byte-identity on every cell,
/// anti-vacuity against real aomenc's own output, collapse soundness in the
/// stock direction, and a non-empty shard.
fn run_speed_array(speed: i32, shard: usize, n_shards: usize) {
    c::ref_init();
    let sc = speed_ctx(speed);
    let cell = speed_cell(speed);
    let cctx = cell_ctx(&cell);
    let stock = c_stock_payload(&cell);
    let stock_eff = Effective::resolve(&DEFAULT_ROW, &cctx);
    let tag = format!("s{speed}");

    let mut seen: BTreeSet<Row> = BTreeSet::new();
    let pinned: Vec<Row> = cp::covering_array(sc.t)
        .into_iter()
        .map(|r| remap_open_levels(&r, speed))
        .filter(|r| cp::illegal_reason_at_speed(r, speed).is_none() && seen.insert(*r))
        .collect();
    let collapsed = cp::collapse(&pinned, &cctx);
    let rows: Vec<Row> = collapsed
        .representatives
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % n_shards == shard)
        .map(|(_, r)| r)
        .collect();
    assert!(!rows.is_empty(), "{tag}: shard {shard}/{n_shards} is empty");

    let mut cells = Vec::new();
    for row in &rows {
        let label = format!("{tag}_{}", cp::row_label(row));
        let r = run_cell(&cell, &label, &cp::knobs_of(row), &stock);
        if Effective::resolve(row, &cctx) == stock_eff {
            assert!(
                !r.c_moved,
                "{label}: the collapse engine resolves this row to the STOCK \
                 effective config, but real aomenc produced different bytes at \
                 THIS SPEED — the Effective signature is under-refined for the \
                 speed-derived state"
            );
        }
        cells.push(r);
    }
    let non_stock: Vec<&Cell> = cells.iter().filter(|c| !c.label.ends_with("_stock")).collect();
    let moved = non_stock.iter().filter(|c| c.c_moved).count();
    let moved_pct = if non_stock.is_empty() {
        100.0
    } else {
        100.0 * moved as f64 / non_stock.len() as f64
    };
    println!("{}", render(&cells, &tag, sc.t, shard, n_shards, moved_pct));
    let open: BTreeSet<String> = cells
        .iter()
        .filter(|c| !c.exact)
        .map(|c| c.label[tag.len() + 1..].to_string())
        .collect();
    let here: BTreeSet<String> = rows.iter().map(|r| cp::row_label(r)).collect();
    let expected: BTreeSet<String> = SPEED_OPEN_COMBINATIONS
        .iter()
        .filter(|(sp, l)| *sp == speed && here.contains(*l))
        .map(|(_, l)| l.to_string())
        .collect();
    assert_eq!(
        open, expected,
        "{tag}: the set of covering-array cells that are NOT byte-identical to \
         real aomenc at --cpu-used={speed} MOVED. The levels this speed pins \
         open are remapped to default, so anything here is a knob COMBINATION \
         that diverges at this SPEED and nowhere else. A row that started \
         matching means a speed x combination near-tie closed — re-pin \
         SPEED_OPEN_COMBINATIONS. A row that started diverging is a regression. \
         ({} of {} cells open)",
        open.len(),
        cells.len()
    );
    assert!(
        moved_pct >= sc.min_moved_pct,
        "{tag}: only {moved_pct:.1}% of the {} non-stock rows changed real \
         aomenc's output (floor {}%) — at this speed the encoder ignores more \
         of the configuration than the strength assumes; lower t or re-measure \
         SPEED_LIVE_LEVELS",
        non_stock.len(),
        sc.min_moved_pct
    );
}

// t=4 at cpu-used 2 — every 4-way knob interaction at the first speed tier that
// turns on the ML/tx-stats machinery (`prune_tx_type_using_stats`,
// `ml_4_partition_search_level_index=2`, `disable_smooth_intra`,
// `prune_filter_intra_level`, `perform_coeff_opt=3`), and cheap enough
// (48 ms/cell) to afford full strength. 187 rows, sharded 3 ways.
#[test]
fn combinations_t4_speed2_s0() {
    run_speed_array(2, 0, 3)
}
#[test]
fn combinations_t4_speed2_s1() {
    run_speed_array(2, 1, 3)
}
#[test]
fn combinations_t4_speed2_s2() {
    run_speed_array(2, 2, 3)
}

// t=3 at the two other structure changes inside the RD-search range: speed 4
// (winner-mode multi-pass + `prune_chroma_modes_using_luma_winner` +
// `fast_intra_tx_type_search`) and speed 6 (the last RD speed —
// `default_max_partition_size=BLOCK_32X32`, `cfl_search_range=1`,
// MULTI_WINNER_MODE_OFF).
#[test]
fn combinations_t3_speed4_s0() {
    run_speed_array(4, 0, 2)
}
#[test]
fn combinations_t3_speed4_s1() {
    run_speed_array(4, 1, 2)
}
#[test]
fn combinations_t3_speed6() {
    run_speed_array(6, 0, 1)
}

// t=3 at the two NON-RD-search structures — speed 7 (VAR_BASED_PARTITION fixed
// tree, KB-11) and speed 8 (nonrd PICKMODE, KB-12). These are the speeds where
// a partition/mode knob's MEANING changes rather than its strength, and they
// cost ~3 ms/cell, so they earn t=3 for free.
#[test]
fn combinations_t3_speed7() {
    run_speed_array(7, 0, 1)
}
#[test]
fn combinations_t3_speed8() {
    run_speed_array(8, 0, 1)
}

// t=2 at the speeds whose sf delta is small (1, 3, 5) and at speed 9, where only
// 4 of 24 reachable levels still move real aomenc — pairwise is already generous
// there. Speed 0 gets a t=2 replay too: it is redundant with the 2,910-cell
// array above by construction, and that is exactly why it is worth 17 cells —
// it is the control proving this section's runner agrees with `run_array`.
#[test]
fn combinations_t2_speed0_control() {
    run_speed_array(0, 0, 1)
}
#[test]
fn combinations_t2_speed1() {
    run_speed_array(1, 0, 1)
}
#[test]
fn combinations_t2_speed3() {
    run_speed_array(3, 0, 1)
}
#[test]
fn combinations_t2_speed5() {
    run_speed_array(5, 0, 1)
}
#[test]
fn combinations_t2_speed9() {
    run_speed_array(9, 0, 1)
}

// ---------------------------------------------------------------------------
// 7d. SPEED x SIZE — the one framesize-gated speed feature that is live
// ---------------------------------------------------------------------------

/// A 512x512 monochrome pseudo-random-luma ALLINTRA cell — the content
/// `tx_stats_prune_e2e.rs` established for this speed feature.
///
/// The residual after intra prediction is high-frequency and uncorrelated, which
/// makes IDTX (identity transform, KF probability 2 < the threshold 10) genuinely
/// competitive; `prune_tx_type_using_stats` removes it, so the sf is
/// LOAD-BEARING on this content rather than a no-op. Monochrome keeps it
/// luma-only (the prune is luma-side).
fn speed_noise_cell(label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut y = vec![0u16; w * h];
    for p in y.iter_mut() {
        *p = (next() % 256) as u16;
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono: true,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u: Vec::new(),
        v: Vec::new(),
    }
}

/// THE SPEED x SIZE CLOSURE — `prune_tx_type_using_stats` crossed with the knob
/// axes, with the sf PROVEN non-zero rather than assumed.
///
/// This is the cell the SIZE section named as out of its reach and the reason
/// this task exists. The sf needs BOTH conditions:
///
/// > `if (is_480p_or_larger) { ... sf->tx_sf.tx_type_search.prune_tx_type_using_stats = 1; }`
/// > — inside `if (speed >= 2)`, libaom `av1/encoder/speed_features.c:261`
///
/// so it is 0 on every one of the 2,910 speed-0 cells (in the PORT *and* in real
/// aomenc — nothing was unexercised there, the SIZE section already established
/// that), and 0 on every sub-480p cell at any speed. 512x512 at `--cpu-used=2`
/// is the smallest cell where it is 1.
///
/// **Liveness is proved, not asserted.** `ToggleKnobs::disable_tx_stats_prune`
/// forces the port's `sf.prune_tx_type_using_stats` to 0 while leaving the C
/// side (driven by `--cpu-used` alone) pruning. So:
///
/// * port WITH the prune must byte-match real aomenc, and
/// * port WITHOUT it must DIVERGE from the port's own with-prune output —
///
/// which is only possible if the field is non-zero AND load-bearing on this
/// content. `tx_stats_prune_e2e.rs` proves that for the STOCK knob row; this
/// test additionally proves it composes — the witness is re-run on a non-stock
/// covering-array row, so "the sf is live" is established inside the
/// configuration space, not only at its origin.
fn run_speed_size_txstats(shard: usize, n_shards: usize) {
    c::ref_init();
    let speed = 2;
    let cell = speed_noise_cell("txstats_512_s2", 512, 512, 32, speed);
    let cctx = CellCtx { w: 512, h: 512, mono: true, sb_px: 64 };
    // The model must agree that this context is the one that turns the sf on.
    let sd = cp::size_derived(&cctx, speed);
    assert_eq!(
        sd.prune_tx_type_using_stats, 1,
        "the size model says prune_tx_type_using_stats is {} at 512x512 \
         cpu-used=2 — this whole context exists to exercise level 1",
        sd.prune_tx_type_using_stats
    );
    assert_eq!(
        cp::size_derived(&cctx, 0).prune_tx_type_using_stats,
        0,
        "the same geometry at speed 0 must leave the sf off — otherwise this \
         cell is not testing the SPEED x SIZE crossing"
    );

    let stock = c_stock_payload(&cell);
    let mut seen: BTreeSet<Row> = BTreeSet::new();
    let rows: Vec<Row> = cp::covering_array(2)
        .into_iter()
        .map(|r| remap_open_levels(&r, speed))
        .filter(|r| cp::illegal_reason_at_speed(r, speed).is_none() && seen.insert(*r))
        .enumerate()
        .filter(|(i, _)| i % n_shards == shard)
        .map(|(_, r)| r)
        .collect();
    assert!(!rows.is_empty(), "txstats: empty shard {shard}/{n_shards}");

    let mut cells = Vec::new();
    for row in &rows {
        let knobs = cp::knobs_of(row);
        let label = format!("txstats512s2_{}", cp::row_label(row));
        let c_tu = cell.c_encode_ctrls(&knobs.c_ctrls());
        assert!(!c_tu.is_empty(), "{label}: C encode failed");
        let c_payload = EncodeCell::frame_obu_payload(&c_tu);
        let port = cell.port_encode_with(&c_tu, &knobs);
        cells.push(Cell {
            label,
            exact: port == c_payload,
            c_moved: c_payload != stock,
            port_len: port.len(),
            c_len: c_payload.len(),
        });
    }
    let moved = cells.iter().filter(|c| c.c_moved).count();
    println!(
        "{}",
        render(&cells, "txstats512s2", 2, shard, n_shards, 100.0 * moved as f64 / cells.len() as f64)
    );
    let open: Vec<&Cell> = cells.iter().filter(|c| !c.exact).collect();
    assert!(
        open.is_empty(),
        "txstats512s2: {} of {} SPEED x SIZE cells are NOT byte-identical to \
         real aomenc — a knob combination diverges where the framesize-gated \
         speed feature prune_tx_type_using_stats is LIVE. Offenders: {}",
        open.len(),
        cells.len(),
        open.iter()
            .map(|c| format!("{} (port {}B vs C {}B)", c.label, c.port_len, c.c_len))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ---- the LIVENESS WITNESS, one curated row per shard ------------------
    //
    // `ToggleKnobs::disable_tx_stats_prune` forces the PORT's
    // `sf.prune_tx_type_using_stats` to 0 while the C side (driven by
    // `--cpu-used` alone) keeps pruning, so "port-without != port-with" is a
    // direct proof that the field is NON-ZERO and load-bearing on this cell —
    // not an inference from the speed/framesize model.
    //
    // Both directions are gated, which is what makes the witness meaningful
    // rather than a coin flip:
    //
    // * `stock` and `trel1` MUST witness — trellis has nothing to do with the
    //   tx-TYPE candidate set, so the prune still has an IDTX/FLIPADST winner to
    //   remove;
    // * `flip0` (`--enable-flip-idtx=0`) MUST NOT — the knob masks the
    //   FLIPADST/IDTX family out of the ext-tx set (`get_tx_mask`'s
    //   DCT_ADST_TX_MASK arm) BEFORE the stats prune runs, so the prune has
    //   nothing left to remove and forcing it off cannot change a byte. A
    //   witness there would mean the harness knob is perturbing something other
    //   than the stats prune, and every positive witness would be suspect.
    //
    // (Measured 2026-07-30 over 16 singleton rows: 11 witness, and the 5 that do
    // not are `rect0`/`cdf0` — where the prune fires but does not flip the
    // winner — plus the three that structurally disarm it: `flip0`, `dtxo1`
    // (`--use-intra-default-tx-only`, one tx type) and `dir0` (no directional
    // modes left to carry a non-DCT default).)
    const WITNESS_ROWS: &[(Axis, u8, bool)] = &[
        (Axis::CdfUpdate, 0, true),   // the stock row (level 0 = default)
        (Axis::Trellis, 1, true),     // --disable-trellis-quant=1
        (Axis::FlipIdtx, 1, false),   // --enable-flip-idtx=0: structurally blind
    ];
    let (ax, lv, must) = WITNESS_ROWS[shard % WITNESS_ROWS.len()];
    let mut wrow = DEFAULT_ROW;
    wrow[ix(ax)] = lv;
    let wknobs = cp::knobs_of(&wrow);
    let wlabel = cp::row_label(&wrow);
    let c_tu = cell.c_encode_ctrls(&wknobs.c_ctrls());
    let with = cell.port_encode_with(&c_tu, &wknobs);
    let without = cell.port_encode_with(
        &c_tu,
        &ToggleKnobs { disable_tx_stats_prune: true, ..wknobs },
    );
    let witnessed = without != with;
    println!(
        "  prune_tx_type_using_stats=1 LIVENESS on `{wlabel}`: forcing the sf \
         off {} the port's bytes ({} B -> {} B); expected {}",
        if witnessed { "CHANGES" } else { "leaves unchanged" },
        with.len(),
        without.len(),
        if must { "CHANGES" } else { "unchanged" }
    );
    assert_eq!(
        with,
        EncodeCell::frame_obu_payload(&c_tu),
        "txstats512s2 `{wlabel}`: the witness row itself is not byte-identical \
         to real aomenc"
    );
    assert_eq!(
        witnessed, must,
        "txstats512s2 `{wlabel}`: prune_tx_type_using_stats liveness flipped. \
         If a MUST-witness row stopped witnessing, the >=480p x speed>=2 cell no \
         longer exercises the sf and this whole context is vacuous. If the \
         structurally-blind row STARTED witnessing, `disable_tx_stats_prune` is \
         perturbing something other than the stats prune and every positive \
         witness here is suspect."
    );
}

#[test]
fn speed_size_txstats_s0() {
    run_speed_size_txstats(0, 3)
}
#[test]
fn speed_size_txstats_s1() {
    run_speed_size_txstats(1, 3)
}
#[test]
fn speed_size_txstats_s2() {
    run_speed_size_txstats(2, 3)
}

// ---------------------------------------------------------------------------
// 7e. The speed inventory — arithmetic, pinned
// ---------------------------------------------------------------------------

/// The speed-class arithmetic, computed and PINNED — plus the REFUTATION of the
/// one collapse the model could have offered.
///
/// Three claims:
///
/// 1. **the speed-feature class partition**: which `--cpu-used` steps move the
///    resolved ALLINTRA `SpeedFeatures` at all. Measured: **nine** classes over
///    ten speeds — `{0} {1} {2} {3} {4} {5} {6} {7,9} {8}`. The last class is
///    non-consecutive on purpose: speed 8 raises
///    `var_part_split_threshold_shift` to 8 (speed_features.c:581) and speed 9
///    puts it back to 7 (:601, *"intentionally lower than speed 8's"*), so 7
///    and 9 resolve to the SAME struct while 8 stands alone. It was
///    `{7,8,9}` until KB-32 — the shift steps were unmodelled, which is what
///    made `force_large_partition_blocks_intra`'s two arms invisible;
/// 2. **that partition is NOT a valid collapse.** The encoder also branches on
///    the raw `PickFrameCfg::speed` (six cited sites), so 7 / 8 / 9 are distinct
///    configurations with an identical `SpeedFeatures`. Asserted against the
///    ORACLE, not by inspection: real aomenc's own frame payload must differ at
///    `--cpu-used` 7 vs 8 vs 9 on the same cell. A collapse the oracle
///    contradicts is not a collapse, so every speed keeps its own context;
/// 3. **the speed x framesize table**: `cp::size_derived` evaluated across the
///    speed range, showing which framesize-gated speed features come alive and
///    where — the arithmetic the SIZE section pinned but could not encode.
#[test]
fn speed_class_inventory_is_pinned() {
    c::ref_init();
    let classes = speed_sf_classes(&ALL_SPEEDS, false, false);
    let shape: Vec<Vec<i32>> = classes.iter().map(|(_, v)| v.clone()).collect();
    println!("\n=== speed classes (ALLINTRA SpeedFeatures equality) ===\n  {shape:?}");
    assert_eq!(
        shape,
        vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7, 9],
            vec![8]
        ],
        "the ALLINTRA speed-feature class partition moved — a speed step \
         started (or stopped) changing the resolved SpeedFeatures. Re-read \
         set_allintra_speed_features_framesize_independent before re-pinning."
    );
    // bd10 and screen-content resolutions must partition the same way, else the
    // per-speed strengths chosen on the bd8 non-screen context do not carry.
    for (screen, hbd) in [(true, false), (false, true), (true, true)] {
        assert_eq!(
            speed_sf_classes(&ALL_SPEEDS, screen, hbd)
                .iter()
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>(),
            shape,
            "the speed-class partition differs at (screen={screen}, hbd={hbd})"
        );
    }

    // (2) The collapse is REFUTED by the oracle: C's own bytes differ at 7/8/9.
    let mut payloads = Vec::new();
    for speed in [7i32, 8, 9] {
        let cell = speed_cell(speed);
        payloads.push((speed, c_stock_payload(&cell)));
    }
    for i in 0..payloads.len() {
        for j in i + 1..payloads.len() {
            assert_ne!(
                payloads[i].1, payloads[j].1,
                "real aomenc produced IDENTICAL bytes at --cpu-used={} and \
                 --cpu-used={}. If that ever becomes true the SpeedFeatures \
                 collapse would be sound and these speeds could share a context \
                 — but it is currently false, which is why they do not.",
                payloads[i].0, payloads[j].0
            );
        }
    }
    println!(
        "  {} — oracle payloads at cpu-used 7/8/9: {} / {} / {} B (all distinct)",
        aom_bench::config_perm::SPEED_SF_EQUALITY_IS_NOT_A_COLLAPSE,
        payloads[0].1.len(),
        payloads[1].1.len(),
        payloads[2].1.len()
    );

    // (3) speed x framesize. Sub-480p vs >=480p at every speed.
    let small = CellCtx { w: 64, h: 64, mono: false, sb_px: 64 };
    let big = CellCtx { w: 512, h: 512, mono: true, sb_px: 64 };
    let mut table = String::from(
        "\n=== speed x framesize (cp::size_derived) ===\n  speed | prune_tx_type_using_stats \
         64x64 / 512x512 | sq_only_threshold 64x64 / 512x512\n",
    );
    for &sp in ALL_SPEEDS.iter() {
        let _ = writeln!(
            table,
            "  {sp:<5} | {} / {}                        | {} / {}",
            cp::size_derived(&small, sp).prune_tx_type_using_stats,
            cp::size_derived(&big, sp).prune_tx_type_using_stats,
            cp::sq_only_threshold_allintra(&small, sp),
            cp::sq_only_threshold_allintra(&big, sp),
        );
    }
    println!("{table}");
    // The four PINS that make the speed x size cell meaningful.
    assert_eq!(cp::size_derived(&big, 0).prune_tx_type_using_stats, 0);
    assert_eq!(cp::size_derived(&big, 2).prune_tx_type_using_stats, 1);
    assert_eq!(cp::size_derived(&big, 4).prune_tx_type_using_stats, 2);
    assert_eq!(
        cp::size_derived(&small, 9).prune_tx_type_using_stats,
        0,
        "sub-480p must never enable the stats prune at any speed"
    );
    // The four speed x framesize derivations are enumerated with citations.
    assert_eq!(cp::SPEED_X_FRAMESIZE_DERIVATIONS.len(), 4);
    for (thresh, size, field, cite) in cp::SPEED_X_FRAMESIZE_DERIVATIONS {
        assert!(
            cite.starts_with("speed_features.c:"),
            "{field}: every speed x framesize derivation must cite its libaom line"
        );
        println!("  speed{thresh:<32} x {size:<40} -> {field}  ({cite})");
    }
}

/// OPEN FINDING, pinned: `--enable-tx-size-search=0` stops being a
/// configuration at ALLINTRA speed >= 8, and the harness's header assertion
/// stops being a valid claim there.
///
/// > `if (!oxcf->txfm_cfg.enable_tx_size_search && sf->rt_sf.use_nonrd_pick_mode == 0)`
/// > `  sf->winner_mode_sf.tx_size_search_level = 3;`
/// > — libaom `av1/encoder/speed_features.c:2726-2729`
///
/// `set_allintra_speed_features_framesize_independent` sets
/// `rt_sf.use_nonrd_pick_mode = 1` at `speed >= 8` (`:579`), so from speed 8 the
/// CLI knob never reaches `tx_size_search_level`, `select_tx_mode` does not
/// return `TX_MODE_LARGEST`, and `EncodeCell::port_encode_with`'s
///
/// > `assert!(knobs.enable_tx_size_search || !p.tx_mode_select)`
/// > — `aom-bench/src/lib.rs:1119-1123`
///
/// PANICS on a stream real aomenc happily produced. Measured 2026-07-30: at
/// `--cpu-used=8` on the primary context the frame header codes TX_MODE_SELECT
/// with the knob off; at `--cpu-used=9` on the same cell it happens to code
/// LARGEST (C's post-hoc `txb_split_count == 0` demotion), so the panic is
/// DATA-dependent from speed 8 up, not a clean speed threshold.
///
/// Two consequences, both modelled rather than papered over:
/// [`axis_level_dead_at_speed`] removes the level from the matrix and the array
/// at speed >= 8 (it is inert there — nothing is lost), and
/// [`cp::illegal_reason_at_speed`] records that the single C-forbidden pair
/// (`txss=0 x tx64=0`, encodeframe.c:2461) LAPSES at speed >= 8 because the
/// assert's `tx_search_type != USE_LARGESTALL` disjunct now holds.
///
/// This is a FINDING, not a fix — the assertion lives in the shared harness.
/// Self-promoting: if the header ever codes LARGEST at speed 8 the pin fails.
#[test]
fn speed_txss_nonrd_lapse_is_pinned() {
    c::ref_init();
    let mut row = DEFAULT_ROW;
    row[ix(Axis::TxSizeSearch)] = 1;
    let knobs = cp::knobs_of(&row);
    // The model's claim.
    assert!(axis_level_dead_at_speed(Axis::TxSizeSearch, 1, 8).is_some());
    assert!(axis_level_dead_at_speed(Axis::TxSizeSearch, 1, 7).is_none());
    // The exclusion lapses with it.
    let mut both = row;
    both[ix(Axis::Tx64)] = 1;
    assert!(
        cp::illegal_reason(&both).is_some() && cp::illegal_reason_at_speed(&both, 7).is_some(),
        "txss=0 x tx64=0 must stay excluded below speed 8"
    );
    assert!(
        cp::illegal_reason_at_speed(&both, 8).is_none(),
        "txss=0 x tx64=0 is not forbidden at speed >= 8 — the CLI no longer \
         forces USE_LARGESTALL (speed_features.c:2726-2729)"
    );
    // The oracle's behaviour: C accepts the knob at speed 8 and still moves.
    let mut report = String::from("\n=== --enable-tx-size-search=0 across speed ===\n");
    for speed in [6i32, 7, 8, 9] {
        let cell = speed_cell(speed);
        let stock = c_stock_payload(&cell);
        let c_payload = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&knobs.c_ctrls()));
        let port = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cell.port_encode_with(&cell.c_encode_ctrls(&knobs.c_ctrls()), &knobs)
        }));
        let _ = writeln!(
            report,
            "  cpu-used={speed}: C {} B ({}), harness {}",
            c_payload.len(),
            if c_payload == stock { "inert" } else { "moved" },
            match &port {
                Ok(p) if *p == c_payload => "byte-identical".to_string(),
                Ok(p) => format!("DIVERGES ({} B)", p.len()),
                Err(_) => "PANICS on the TX_MODE_LARGEST assertion".to_string(),
            }
        );
        if speed == 8 {
            assert!(
                port.is_err(),
                "cpu-used=8 with --enable-tx-size-search=0 no longer trips the \
                 harness's TX_MODE_LARGEST assertion — the header now codes \
                 LARGEST, so re-pin this finding and let the level back into the \
                 speed-8 array (axis_level_dead_at_speed)"
            );
        }
    }
    println!("{report}");
}

/// TEETH INVARIANTS — the properties that make the added speed cells able to
/// catch something the 2,910 speed-0 cells cannot.
///
/// No encoding beyond the ten stock controls. Three claims:
///
/// 1. **the speed envelope is intact**: the primary context's STOCK encode is
///    byte-identical to real aomenc at every one of the ten speeds. This is the
///    control every other assertion in the section rests on, and it is what a
///    broken speed-gated derivation breaks first (the `LPF_PICK_FROM_Q` arm at
///    speed >= 6 is exactly this shape — see the harness note in
///    `aom-bench/src/lib.rs`);
/// 2. **the pre-existing contexts cannot see any of it**: every `Ctx` /
///    `SizeCtx` / `Content` cell is speed 0, where `SpeedFeatures::set_allintra`
///    equals its speed-0 resolution by definition and every `speed >= N` gate in
///    `aom-encode` is false;
/// 3. **the speeds this section adds do differ**: consecutive speeds must
///    resolve to different `SpeedFeatures` (up to the pinned `{7,9}` class,
///    which is non-consecutive), so no added context is a duplicate of its
///    neighbour.
#[test]
fn speed_axis_teeth_are_real() {
    c::ref_init();
    let mut report = String::from("\n=== speed-axis teeth ===\n");
    for &speed in ALL_SPEEDS.iter() {
        let cell = speed_cell(speed);
        let c_tu = cell.c_encode_ctrls(&[]);
        let c_payload = EncodeCell::frame_obu_payload(&c_tu);
        let port = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
        let _ = writeln!(
            report,
            "  TEETH s{speed}_stock exact={} port {}B real {}B",
            port == c_payload,
            port.len(),
            c_payload.len()
        );
        assert_eq!(
            port, c_payload,
            "s{speed}: the primary speed context's stock encode is not \
             byte-identical to real aomenc"
        );
    }
    println!("{report}");

    // (2) Everything above this section is speed 0.
    let zero = aom_encode::speed_features::SpeedFeatures::set_allintra(0, false, false);
    for &speed in ALL_SPEEDS.iter().skip(1) {
        let sf = aom_encode::speed_features::SpeedFeatures::set_allintra(speed, false, false);
        assert_ne!(
            sf, zero,
            "cpu-used={speed} resolves to the SPEED-0 feature set — a speed \
             context that is speed-0-equivalent buys nothing"
        );
    }
    // (3) EVERY consecutive pair now differs — 7->8 and 8->9 both move
    // `var_part_split_threshold_shift` (speed_features.c:581 / :601), which is
    // what KB-32 added. The one remaining equality is the NON-consecutive
    // {7, 9} class, asserted separately below so it cannot rot silently.
    for &speed in ALL_SPEEDS.iter().skip(1) {
        let a = aom_encode::speed_features::SpeedFeatures::set_allintra(speed - 1, false, false);
        let b = aom_encode::speed_features::SpeedFeatures::set_allintra(speed, false, false);
        assert_ne!(a, b, "cpu-used={} and {speed} resolve identically", speed - 1);
    }
    assert_eq!(
        aom_encode::speed_features::SpeedFeatures::set_allintra(7, false, false),
        aom_encode::speed_features::SpeedFeatures::set_allintra(9, false, false),
        "cpu-used 7 and 9 now resolve DIFFERENTLY — the pinned {{7,9}} class \
         split; re-pin speed_class_inventory_is_pinned. (They are equal only \
         because every OTHER speed-8/9 setting is modelled at its consumer \
         rather than in this struct — see SPEED_SF_EQUALITY_IS_NOT_A_COLLAPSE.)"
    );
}

/// BUDGET ACCOUNTING for the speed axis — no encoding, just the arithmetic that
/// justifies the strength chosen per speed, pinned so a strength bump fails here
/// instead of quietly costing the suite a minute.
///
/// `ms_per_cell` is MEASURED (`benchmarks/config_perm_speed_axis_2026-07-30.tsv`)
/// on a 12-core M4 shared with two concurrent agents, so the figures are upper
/// bounds and the CPU total below is conservative.
#[test]
fn speed_axis_budget_is_accounted() {
    let mut total_cells = 0usize;
    let mut total_ms = 0u64;
    let mut report = String::from("\n=== speed-axis budget ===\n");
    for sc in SPEED_CONTEXTS {
        let array_rows = cp::covering_array(sc.t).len();
        let matrix_rows = singleton_levels(sc.speed).len() + 1; // + the stock control
        let rows = array_rows + matrix_rows;
        total_cells += rows;
        total_ms += rows as u64 * sc.ms_per_cell as u64;
        let _ = writeln!(
            report,
            "  cpu-used={:<2} t={} {:>3} array + {:>2} matrix rows x {:>4} ms   \
             (live levels: {:>2})",
            sc.speed,
            sc.t,
            array_rows,
            matrix_rows,
            sc.ms_per_cell,
            live_levels_at(sc.speed).len()
        );
    }
    // The SPEED x SIZE closure: a t=2 array on a 512x512 cpu-2 cell, each row
    // encoded THREE times (C, port-with-sf, port-without-sf for the liveness
    // witness).
    let tx_rows = cp::covering_array(2).len();
    total_cells += tx_rows;
    total_ms += tx_rows as u64 * 780;
    let _ = writeln!(
        report,
        "  txstats512s2  t=2 {tx_rows:>3} rows x  780 ms  (>=480p x speed>=2, \
         3 encodes/row incl. the liveness witness)"
    );
    // KB-20: bd10/bd12 x cpu-used {8,9}. Nonrd cells are the cheapest in the
    // section (~17 ms measured incl. the C encode), which is why the crossing
    // the panic hid in can be gated at 24 cells for well under a second.
    let kb20_cells = 2 * 6 * 2;
    total_cells += kb20_cells;
    total_ms += kb20_cells as u64 * 17;
    let _ = writeln!(
        report,
        "  hbd_nonrd         {kb20_cells:>3} cells x   17 ms  (bd{{10,12}} x \
         cq{{5,12,20,32,48,63}} x cpu-used{{8,9}}, KB-20)"
    );
    let _ = writeln!(
        report,
        "  TOTAL {total_cells} cells, {:.1} s CPU (upper bound; libtest spreads \
         it across {} shards)",
        total_ms as f64 / 1000.0,
        SPEED_CONTEXTS.iter().map(|c| c.shards).sum::<usize>() + 3 + 3
    );
    println!("{report}");
    assert!(
        total_ms <= 120_000,
        "the speed axis budget is {} s CPU — over the 120 s ceiling this section \
         was designed to; drop a strength or a speed rather than letting the \
         suite drift",
        total_ms / 1000
    );
}

/// THE SPEED ENVELOPE, mapped and pinned: which (content, `--cpu-used`) pairs
/// the port encodes byte-identically to real aomenc with STOCK knobs.
///
/// The gated speed contexts above all ride ONE content, chosen because its stock
/// encode is byte-exact at every speed. That is the right choice for isolating
/// the knob x speed interaction — and it would be dishonest to leave as the only
/// statement, because it is not representative. This test maps the envelope on
/// four more contents and pins the result, so the section reports the shape of
/// the speed axis rather than the shape of its best cell.
///
/// **Measured 2026-07-30, and the shape is counter-intuitive.** The fragile band
/// is NOT the nonrd speeds (7-9), which are the cleanest above 0 — it is speeds
/// **4 and 5**, the winner-mode / multi-winner tiers, where three of the five
/// contexts diverge with every knob at its default. KB-21 root-caused the first
/// half of that band the same day — `early_term_after_none_split` (ALLINTRA
/// speed>=4, `speed_features.c:477`) was unported — which closed the cpu-5
/// column on `q00_64` and `q00_128`. **KB-21 root #2 (2026-07-31) closed the
/// rest of the bd8 band**: two coefficient-level defects in the speed>=4
/// tx-type search — `prune_txk_type`'s est-rd estimate added the tx-type cost
/// to `eob == 0` candidates that C costs as a bare txb-skip flag (so all-zero
/// candidates sorted by SIGNALLING cost and never reached `txk_map[0]`, where
/// C's `skip_tx_search` break ends the search on the first candidate), and the
/// SATD trellis-skip failed to switch that tx type's quantizer to
/// `AV1_XFORM_QUANT_B`. All three bd8 contexts are now byte-identical at every
/// speed 0..9, so the fragile band is now a bd10-only statement. And bd10 is open at most
/// speeds: `av1-1-b10-00-quantizer-00` is byte-exact at speed 0 (a gated
/// context of the main array), at speed 7, and — since the KB-20 landing
/// (2026-07-30) — at the nonrd speeds 8 and 9; it diverges at 1, 2, 3, 4, 5 and
/// 6. All of that carries the KB-10/KB-12 near-tie signature (0-6 byte deltas),
/// and none of it is reachable from a speed-0 matrix.
///
/// Speeds 8 and 9 read `panic` here until 2026-07-30: the KB-12 nonrd estimate
/// arm was bd8-only and asserted on `env.bd == 8`, so EVERY bd10/bd12 encode at
/// `--cpu-used >= 8` aborted. That is what [`speed_nonrd_hbd_byte_identity`]
/// now gates, cell by cell, at byte-identity.
///
/// Consequences, applied rather than just noted: no bd10 speed context is gated
/// (it would gate a divergence), and the per-speed arrays run on the one content
/// whose envelope is intact at all ten speeds. The full 1,342-cell grid behind
/// this table is `benchmarks/config_perm_speed_axis_2026-07-30.tsv`
/// ([`speed_axis_evidence_sweep`]).
///
/// Self-promoting in both directions.
#[test]
fn speed_envelope_stock_map_is_pinned() {
    c::ref_init();
    // (tag, vector, crop, mono, the speeds whose STOCK encode DIVERGES)
    // (tag, vector, crop, mono, [(speed, expected verdict)]) — every speed not
    // listed must be "ok".
    let probes: &[(&str, &str, Option<(usize, usize, usize, usize)>, bool, &[(i32, &str)])] = &[
        ("sz64", SPEED_VECTOR, None, false, &[]),
        // KB-21 (2026-07-30): the `early_term_after_none_split` root closed the
        // cpu-5 column on the two 4:2:0 contexts. KB-21 root #2 (2026-07-31)
        // closed the REST of the bd8 band — the `prune_txk_type` est-rd
        // ordering (tx-type cost added to an `eob == 0` laplacian estimate) and
        // the SATD trellis-skip's per-tx-type `AV1_XFORM_QUANT_B` switch — so
        // all three bd8 contexts are byte-identical at every speed 0..9.
        ("q00_64", "av1-1-b8-00-quantizer-00", Some((64, 64, 64, 64)), false, &[]),
        ("q00_mono64", "av1-1-b8-00-quantizer-00", Some((64, 64, 64, 64)), true, &[]),
        ("q00_128", "av1-1-b8-00-quantizer-00", Some((128, 128, 64, 64)), false, &[]),
        (
            "b10_64",
            "av1-1-b10-00-quantizer-00",
            Some((64, 64, 64, 64)),
            false,
            // 8 and 9 were "panic" until 2026-07-30: the KB-12 nonrd path was
            // bd8-only and `nonrd_pick_intra_mode` asserted `env.bd == 8`.
            // The hbd estimate arm is now ported (KB-20) and both are "ok" —
            // gated per-cell by [`speed_nonrd_hbd_byte_identity`].
            &[(1, "diverge"), (2, "diverge"), (3, "diverge"), (4, "diverge"),
              (5, "diverge"), (6, "diverge")],
        ),
    ];
    let mut report = String::from("\n=== speed envelope: stock byte-identity per (content, cpu-used) ===\n");
    let mut failures = Vec::new();
    // No probe is expected to panic any more (KB-20 closed the last one), but
    // the `catch_unwind` + silent hook stays: it is what lets a NEW unported
    // arm be reported as `panic` against its `ok` pin instead of aborting the
    // whole map.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for (tag, vector, crop, mono, open) in probes {
        let mut line = format!("  {tag:<11}");
        for &speed in ALL_SPEEDS.iter() {
            let mut cell = EncodeCell::real_content(tag, vector, *crop, SPEED_CQ, speed);
            if *mono {
                cell.mono = true;
                cell.u.clear();
                cell.v.clear();
            }
            let c_tu = cell.c_encode_ctrls(&[]);
            let c_payload = EncodeCell::frame_obu_payload(&c_tu);
            let got = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cell.port_encode_with(&c_tu, &ToggleKnobs::default())
            })) {
                Ok(p) if p == c_payload => "ok",
                Ok(_) => "diverge",
                Err(_) => "panic",
            };
            let want = open.iter().find(|(s, _)| *s == speed).map_or("ok", |(_, v)| *v);
            let _ = write!(line, " s{speed}={got}");
            if got != want {
                failures.push(format!(
                    "{tag} cpu-used={speed}: stock encode is now `{got}` (pinned `{want}`)"
                ));
            }
        }
        let _ = writeln!(report, "{line}");
    }
    std::panic::set_hook(hook);
    println!("{report}");
    assert!(
        failures.is_empty(),
        "the SPEED ENVELOPE moved. A (content, speed) that started matching \
         means a stock-level near-tie closed — re-pin, and consider promoting \
         that content to a gated speed context. One that started diverging is a \
         regression. {failures:?}"
    );
}

// ---------------------------------------------------------------------------
// 7e. BIT DEPTH x the nonrd speeds — KB-20
// ---------------------------------------------------------------------------

/// A high-bit-depth cell for the nonrd speeds, from the b10 conformance vector.
///
/// `bd = 10` is the vector's native depth. `bd = 12` promotes the same real
/// content into the 12-bit range (`pix << 2`) — the precedent
/// `deltaq_mode3_e2e.rs` established for the highbd FP-quantize arm: there is
/// no b12 conformance vector (the intra corpus is bd8/bd10 only, CLAUDE.md
/// Gate 1), and the depth enters this path only through the bd-indexed quant
/// tables, `av1_highbd_block_error`'s `2*(bd-8)` shift and
/// `aom_highbd_sadWxH_bits{10,12}`'s `bd-8` one — all of which shifted-but-real
/// spatial structure exercises faithfully. bd12 is also the depth that pushes
/// the Hadamard estimate furthest outside `int16`, which is where the
/// ISA-conditional quantizer lives (root 3 below).
fn hbd_speed_cell(bd: u8, cq: i32, speed: i32) -> EncodeCell {
    let mut c = EncodeCell::real_content(
        &format!("hbd{bd}_cq{cq}_s{speed}"),
        "av1-1-b10-00-quantizer-00",
        Some((64, 64, 64, 64)),
        cq,
        speed,
    );
    assert_eq!(c.bd, 10, "the b10 vector must decode as 10-bit");
    if bd == 12 {
        c.bd = 12;
        for p in [&mut c.y, &mut c.u, &mut c.v] {
            for v in p.iter_mut() {
                *v <<= 2;
            }
        }
    }
    c
}

/// KB-20 — **bd10/bd12 x `--cpu-used` 8 and 9, byte-identical to real aomenc.**
///
/// Speeds 8 and 9 are the nonrd PICKMODE path (KB-12), whose per-leaf estimator
/// `av1_block_yrd` (nonrd_opt.c:126) branches on `is_cur_buf_hbd(xd)`. Until
/// 2026-07-30 only the lowbd arm was ported and the port carried a hard
/// `assert!(env.bd == 8)` there, so **every** bd10/bd12 encode at `--cpu-used
/// >= 8` PANICKED — on a stream real aomenc produces without complaint. It sat
/// undiscovered because PARITY.md §A lists cpu-used 8/9 byte-identical AND
/// bd10/bd12 byte-identical, each established on its own grid, never crossed.
/// This test IS the crossing, and it is a byte-identity gate rather than a
/// panic pin: a path that returns wrong pixels without panicking would be
/// strictly worse than the assert.
///
/// THREE things were genuinely bd8-specific, two more than the old handoff
/// message named:
/// 1. `av1_block_yrd`'s hbd arm — `aom_hadamard_8x8/16x16` + `av1_quantize_fp`
///    over the `*_transpose` scan/iscan PAIRS + `aom_satd` +
///    `av1_highbd_block_error` (`nonrd_pickmode::block_yrd_hbd`). Every kernel
///    already existed in `aom-dsp`, so this really was the deltaq-mode-3 shape;
/// 2. the speed-9 SAD prune in `av1_estimate_block_intra` — `fn_ptr[bsize].sdf`
///    is `aom_highbd_sadWxH_bits{8,10,12}`, i.e. the raw SAD `>> (bd-8)`
///    (`MAKE_BFP_SAD_WRAPPER`, encoder_utils.h:158). Measured DECISION-INERT on
///    this grid (the prune is a ratio test) — but the wrong form `2*(bd-8)`
///    diverged here, which is how the right one was found;
/// 3. `av1_quantize_fp` is **ISA-conditional** once a coefficient leaves
///    `int16`, which `aom_hadamard_16x16` (+-65534) does routinely at bd10/12
///    on `_c`/NEON: NEON truncates on narrow, x86 saturates, `_c` does neither.
///    That is the root of the last 3 divergent cells on aarch64 and is modelled
///    by `nonrd_pickmode::quantize_fp_dispatched`;
/// 4. `aom_hadamard_16x16` is **ISA-conditional too, and it runs first** — its
///    4-way combine is int32 in `_c`/NEON but int16-WRAPPING in AVX2/SSE2, so
///    the x86 tiers change the coefficients before the quantizer, satd and
///    block-error ever see them (`nonrd_pickmode::hadamard_16x16_dispatched`).
///    Found by the FIRST x86 run of this gate (CI 30595796744): 6 of 24 cells
///    diverged there while both `quantize_fp_dispatched` unit teeth passed.
///
/// **This gate is therefore the one place that measures both models — it is
/// expected to be sensitive to the host ISA, because libaom's own encoder is.**
///
/// Nothing else on the arm was bd8-shaped: predict, subtract, the cost tables
/// and `rdmult` were all already bd-parameterised — the same shape the
/// deltaq-mode-3 landing found for `av1_set_mb_wiener_variance` (PARITY.md §A).
///
/// Speed 9 exercises `prune_intra_mode_using_best_sad_so_far` and the
/// all-estimate leaf dispatch; speed 8 runs the SAD prune off and the hybrid
/// full-RD arm on, so the pair covers both leaf dispatches. cq5..cq63 spans the
/// dense-coefficient and the everything-skips regimes of the estimator.
#[test]
fn speed_nonrd_hbd_byte_identity() {
    c::ref_init();
    let mut report = String::from(
        "\n=== KB-20: bd10/bd12 x the nonrd speeds (stock knobs) ===\n\
         bd\tcq\tspeed\tverdict\tport_B\tc_B\n",
    );
    let mut failures = Vec::new();
    let mut cells = 0usize;
    for bd in [10u8, 12] {
        for cq in [5, 12, 20, 32, 48, 63] {
            for speed in [8, 9] {
                let cell = hbd_speed_cell(bd, cq, speed);
                let stock = c_stock_payload(&cell);
                let label = format!("kb20_b{bd}_cq{cq}_s{speed}");
                let r = run_cell(&cell, &label, &ToggleKnobs::default(), &stock);
                cells += 1;
                let _ = writeln!(
                    report,
                    "{bd}\t{cq}\t{speed}\t{}\t{}\t{}",
                    if r.exact { "MATCH" } else { "MISMATCH" },
                    r.port_len,
                    r.c_len
                );
                if !r.exact {
                    failures.push(format!(
                        "bd{bd} cq{cq} cpu-used={speed}: port {} B vs real {} B",
                        r.port_len, r.c_len
                    ));
                }
            }
        }
    }
    println!("{report}");
    assert_eq!(
        cells, 24,
        "the KB-20 grid is bd{{10,12}} x cq{{5,12,20,32,48,63}} x s{{8,9}}"
    );
    assert!(
        failures.is_empty(),
        "KB-20 REGRESSED: the hbd nonrd estimate arm is no longer byte-identical \
         to real aomenc. Before 2026-07-30 these cells PANICKED (unported arm); \
         a divergence here means the ported arm computes the wrong estimate, \
         which is worse than the panic was.\n\
         FIRST THING TO CHECK IF THIS IS A NEW HOST/ISA rather than a code \
         change: TWO kernels on this arm are ISA-conditional, and they compose \
         in this order. (a) `nonrd_pickmode::hadamard_16x16_dispatched` — \
         `aom_hadamard_16x16`'s 4-way combine is int32 in `_c` and NEON but \
         int16-WRAPPING in AVX2 and SSE2, so on x86 every coefficient reaching \
         the quantizer is already int16-valued. (b) \
         `nonrd_pickmode::quantize_fp_dispatched` — outside int16 the \
         `av1_quantize_fp` tiers disagree with `av1_quantize_fp_c` and with each \
         other (NEON truncates on narrow, x86 saturates). Localise with the unit \
         gates in `aom-encode/tests/nonrd_block_yrd_hbd_diff.rs`: if the two \
         `quantize_fp_dispatched_*` teeth PASS and this gate fails, the \
         divergence is UPSTREAM of the quantizer — that is exactly how (a) was \
         found on the first x86 run. Offenders: {failures:?}"
    );
}


/// DEEP TIER (`--ignored`) — the full speed-axis evidence grid, written to
/// `benchmarks/config_perm_speed_axis_2026-07-30.tsv`.
///
/// The default tier runs the singleton matrix at all ten speeds but replays the
/// covering array on ONE content. This sweep additionally replays the singleton
/// matrix over five contents x ten speeds and records the per-cell timing that
/// [`SPEED_CONTEXTS`]'s `ms_per_cell` figures come from — including the contents
/// whose STOCK encode is NOT byte-exact at some speeds, which is why they are
/// not default-tier contexts.
#[test]
#[ignore]
fn speed_axis_evidence_sweep() {
    c::ref_init();
    // Schema note: the PRIMARY context (`sz64` — the one the default tier gates)
    // is emitted in full, every (speed, axis level) cell. The four secondary
    // contexts emit their stock row plus every DIVERGENT row, and their inert
    // exact rows are summarised in the `[summary]` block instead of listed —
    // that keeps the committed evidence under the repo's per-file size bar
    // without dropping a single finding (an exact-and-inert cell carries no
    // information the per-(content, speed) counts do not).
    let mut summary = String::from(
        "# [summary] one line per (content, speed): cells run, divergences, \
         panics, stock verdict\ncontent\tspeed\tcells\tdivergent\tpanic\tstock\n",
    );
    let mut tsv = String::from(
        "content\tspeed\trow\texact\tc_moved\tport_len\tc_len\tms\n",
    );
    struct Probe {
        tag: &'static str,
        vector: &'static str,
        crop: Option<(usize, usize, usize, usize)>,
        mono: bool,
    }
    let probes = [
        Probe { tag: "sz64", vector: SPEED_VECTOR, crop: None, mono: false },
        Probe { tag: "q00_64", vector: "av1-1-b8-00-quantizer-00", crop: Some((64, 64, 64, 64)), mono: false },
        Probe { tag: "q00_mono64", vector: "av1-1-b8-00-quantizer-00", crop: Some((64, 64, 64, 64)), mono: true },
        Probe { tag: "q00_128", vector: "av1-1-b8-00-quantizer-00", crop: Some((128, 128, 64, 64)), mono: false },
        Probe { tag: "b10_64", vector: "av1-1-b10-00-quantizer-00", crop: Some((64, 64, 64, 64)), mono: false },
    ];
    for p in &probes {
        for &speed in ALL_SPEEDS.iter() {
            let mut cell = EncodeCell::real_content(p.tag, p.vector, p.crop, SPEED_CQ, speed);
            if p.mono {
                cell.mono = true;
                cell.u.clear();
                cell.v.clear();
            }
            let stock = c_stock_payload(&cell);
            let stock_exact = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cell.port_encode_with(&cell.c_encode_ctrls(&[]), &ToggleKnobs::default()) == stock
            }))
            .unwrap_or(false);
            let full = p.tag == "sz64";
            let (mut cells, mut divergent, mut panics) = (0usize, 0usize, 0usize);
            let mut emit = |row: &Row, tsv: &mut String| {
                let t0 = std::time::Instant::now();
                let knobs = cp::knobs_of(row);
                let c_tu = cell.c_encode_ctrls(&knobs.c_ctrls());
                let c_payload = EncodeCell::frame_obu_payload(&c_tu);
                let port = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cell.port_encode_with(&c_tu, &knobs)
                }));
                let (exact, plen, panicked) = match &port {
                    Ok(v) => (*v == c_payload, v.len(), false),
                    Err(_) => (false, 0, true),
                };
                cells += 1;
                divergent += usize::from(!exact);
                panics += usize::from(panicked);
                if full || !exact || *row == DEFAULT_ROW {
                    let _ = writeln!(
                        tsv,
                        "{}\t{speed}\t{}\t{}\t{}\t{}\t{}\t{}",
                        p.tag,
                        cp::row_label(row),
                        exact,
                        c_payload != stock,
                        plen,
                        c_payload.len(),
                        t0.elapsed().as_millis()
                    );
                }
            };
            emit(&DEFAULT_ROW, &mut tsv);
            for (i, l) in singleton_levels(speed) {
                let mut row = DEFAULT_ROW;
                row[i] = l;
                emit(&row, &mut tsv);
            }
            let _ = writeln!(
                summary,
                "{}\t{speed}\t{cells}\t{divergent}\t{panics}\t{}",
                p.tag,
                if stock_exact { "exact" } else { "DIVERGENT" }
            );
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/config_perm_speed_axis_2026-07-30.tsv");
    let out = format!(
        "# config-permutation SPEED axis — evidence sweep, {}\n\
         # 5 contents x 10 --cpu-used levels x every singleton axis level, port \
         vs real aomenc\n{summary}\n# [cells] `sz64` in full; the other \
         contexts' stock + divergent rows\n{tsv}",
        "2026-07-30"
    );
    std::fs::write(&path, format!("{}{out}", evidence_provenance()))
        .expect("write the speed-axis evidence TSV");
    println!("wrote {}", path.display());
}
