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
//! * **Deep** (`-- --ignored`): only [`independence_evidence_sweep`], which is
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

/// FINDING, pinned open: two knobs that are byte-identical to real aomenc on
/// every gated context diverge on the corpus's native monochrome vector.
///
/// `av1-1-b10-24-monochrome`, 64x64 crop at (64,64), speed-0 ALLINTRA:
///
/// | knob | cq12 | cq20 | cq32 | cq48 | cq63 |
/// |---|---|---|---|---|---|
/// | `--use-intra-default-tx-only=1` | DIVERGE 623/623 B | DIVERGE 418/424 | DIVERGE 229/240 | DIVERGE 79/80 | DIVERGE 14/15 |
/// | `--enable-diagonal-intra=0`     | exact | exact | DIVERGE 225/231 | exact | exact |
///
/// Isolated to the CONTENT, not to the format: the same knobs are byte-exact on
/// bd8 4:2:0, on bd8 monochrome derived from `av1-1-b8-01-size-64x64`, on bd10
/// 4:2:0, AND on bd10 monochrome derived from `av1-1-b10-00-quantizer-00` (a
/// full 27-knob singleton sweep over all six contexts reports 0 divergences).
/// A bd12 promotion of the same monochrome content reproduces the
/// `default-tx-only` divergence, so it is not a bd10-specific quantizer path.
/// The equal-size / one-byte deltas are the KB-10 / KB-12 "cheaper RD decision"
/// near-tie signature.
///
/// The stock (all-default) encode of this content IS byte-exact, so the
/// envelope is unaffected; only these two search-narrowing knobs move.
/// `toggles_rd_close.rs`'s grid is bd8 4:2:0 only, which is why this was
/// invisible until the permutation gate replayed the knobs over other contexts.
///
/// This test pins the state exactly: it FAILS if a divergent cell starts
/// matching (fix landed → re-pin and consider promoting this content to a
/// context of the main array) and FAILS if a matching cell regresses.
#[test]
fn mono_vector_open_divergences_pinned() {
    c::ref_init();
    /// `(cq, knob tag, expected-exact)`.
    const EXPECTED: &[(i32, &str, bool)] = &[
        (12, "dtxo", false),
        (12, "diag", true),
        (32, "dtxo", false),
        (32, "diag", false),
        (63, "dtxo", false),
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
    std::fs::write(&out, &tsv).expect("write independence TSV");
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
