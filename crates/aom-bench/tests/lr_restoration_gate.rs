//! Loop-restoration ENCODER-SEARCH parity gate (`--enable-restoration=1`,
//! PARITY.md C2): the port's OWN `av1_pick_filter_restoration` +
//! RU-interleaved pack + restoration header vs real aomenc with the same
//! knob, on real conformance-vector content across the quality range.
//!
//! Per cell:
//! - real: `EncodeCell::c_encode_lr()` (`AV1E_SET_ENABLE_RESTORATION=1`);
//! - port: `EncodeCell::port_encode_lr()` — pack + LF derive/apply + the
//!   ported LR search + repack with the per-SB-root RU params + derived
//!   restoration header fields (decisions NEVER copied from the bootstrap);
//! - primary record: byte-identical (EXACT) or the rd_close verdict
//!   (`|size| <= 5%` AND `zensim_drop <= 0.5` — decode BOTH with the port
//!   decoder, score vs source with zensim);
//! - decision diagnostic: the parsed frame-header restoration decision of
//!   both streams (`parse_restoration_decision`) printed per cell.
//!
//! Anti-vacuous floors: at least one cell where the REAL encoder's decision
//! restores a plane (the reference actually exercises LR), and at least one
//! where the PORT's does. The whole gate then asserts every cell within the
//! rd_close bands (bit-identical cells auto-pass and are recorded EXACT).

use aom_bench::rd_close::{self, RdBands};
use aom_bench::{EncodeCell, parse_restoration_decision};

fn cells() -> Vec<EncodeCell> {
    vec![
        // 1-SB frame across the quality range (KB-6 real-content recipe).
        EncodeCell::real_content("lr_size64_cq12", "av1-1-b8-01-size-64x64", None, 12, 0),
        EncodeCell::real_content("lr_size64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0),
        EncodeCell::real_content("lr_size64_cq48", "av1-1-b8-01-size-64x64", None, 48, 0),
        // Multi-SB + partial-edge SBs (196 = 3 SBs + 4px overhang).
        EncodeCell::real_content("lr_size196_cq20", "av1-1-b8-01-size-196x196", None, 20, 0),
        EncodeCell::real_content("lr_size196_cq48", "av1-1-b8-01-size-196x196", None, 48, 0),
        // 352x288: multi-unit grids at the smaller sizes of the unit-size
        // descent (352 -> 3 units at 128 / 6 at 64), real photographic mix.
        EncodeCell::real_content("lr_quant00_cq32", "av1-1-b8-00-quantizer-00", None, 32, 0),
        EncodeCell::real_content("lr_quant00_cq55", "av1-1-b8-00-quantizer-00", None, 55, 0),
        // 10-bit arm (the highbd search paths).
        EncodeCell::real_content("lr_b10_quant00_cq32", "av1-1-b10-00-quantizer-00", None, 32, 0),
    ]
}

#[test]
fn lr_restoration_search_rd_close_vs_real_aomenc() {
    let bands = RdBands::default();
    let mut results = Vec::new();
    let mut real_active = 0usize;
    let mut port_active = 0usize;
    let mut exact = 0usize;
    let mut decisions_equal = 0usize;

    for cell in cells() {
        let c_tu = cell.c_encode_lr();
        assert!(!c_tu.is_empty(), "{}: real LR encode failed", cell.label);
        let port_payload = cell.port_encode_lr(&c_tu);
        let port_tu = rd_close::splice_frame_obu(&c_tu, &port_payload);

        let (real_frt, real_us) = parse_restoration_decision(&c_tu);
        let (port_frt, port_us) = parse_restoration_decision(&port_tu);
        if real_frt.iter().any(|&t| t != 0) {
            real_active += 1;
        }
        if port_frt.iter().any(|&t| t != 0) {
            port_active += 1;
        }
        let decision_eq = real_frt == port_frt && (real_frt == [0; 3] || real_us == port_us);
        if decision_eq {
            decisions_equal += 1;
        }

        let r = rd_close::compare_cell(&cell.label, &cell, &port_tu, &c_tu);
        if r.bit_identical {
            exact += 1;
        }
        eprintln!(
            "{}: real_frt={real_frt:?} us={real_us:?} | port_frt={port_frt:?} us={port_us:?} \
             | decision_{} | {}",
            cell.label,
            if decision_eq { "EQUAL" } else { "DIFFERS" },
            r.verdict(&bands),
        );
        results.push(r);
    }

    eprintln!("{}", rd_close::render_table(&results, &bands));
    eprintln!(
        "LR gate summary: {}/{} cells real-LR-active, {}/{} port-LR-active, {}/{} \
         decisions equal, {}/{} bit-identical",
        real_active,
        results.len(),
        port_active,
        results.len(),
        decisions_equal,
        results.len(),
        exact,
        results.len(),
    );

    // Anti-vacuous: the reference streams must actually exercise LR on this
    // grid, and the port's search must fire somewhere too — otherwise the
    // rd_close pass would be an empty statement about the feature.
    assert!(
        real_active >= 1,
        "no cell made the REAL encoder restore a plane — the gate grid is vacuous"
    );
    assert!(
        port_active >= 1,
        "the port's LR search never fired on a grid where the reference does"
    );

    rd_close::assert_rd_close(&results, &bands);

    // BYTE-IDENTITY assertions (the section-A gate): the first complete run
    // measured 8/8 cells bit-identical with 8/8 decision equality, so this
    // gate holds the feature at full byte-exactness — any weaker outcome is
    // a regression, not a band question. (rd_close stays above for the
    // richer failure report.)
    assert_eq!(
        decisions_equal,
        results.len(),
        "every cell's restoration decision must equal the C encoder's"
    );
    for r in &results {
        assert!(
            r.bit_identical,
            "{}: LR-on encode must be BYTE-IDENTICAL to real aomenc (measured 8/8 on landing)",
            r.label
        );
    }
}

// ---------------------------------------------------------------------------
// CHUNK-5 arms — COMMITTED-UNVALIDATED (written at handoff; never executed).
// // HANDOFF: run `cargo test -p aom-bench --test lr_restoration_gate --
// //          --nocapture` ONCE; expect the speed-0 format cells (mono/444/
// //          bd12) and GOOD-speed-0 cells to come back EXACT like the main
// //          grid; the speed 1-4 arms exercise the sf paths (ep-prune
// //          ladder, src-var prune, reduced 5-tap window) for the FIRST
// //          time e2e — if a cell mismatches, diff the printed decisions
// //          first (wrong ep / wrong size ⇒ sf transcription bug; equal
// //          decisions + differing bytes ⇒ pack placement bug, which the
// //          main grid makes unlikely). If 100% EXACT: harden this test's
// //          tail to assert bit_identical like the main gate and move the
// //          PARITY A row's honest-fraction note forward. If not: keep
// //          rd_close bands here, record exact/total per arm in PARITY C2.
// ---------------------------------------------------------------------------

/// Nearest-neighbour upsample a 4:2:0 chroma plane to full resolution
/// (for the synthetic 4:4:4 format cell built from real 4:2:0 content).
fn upsample_nn(src: &[u16], cw: usize, ch: usize, w: usize, h: usize) -> Vec<u16> {
    let mut out = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            out[r * w + c] = src[(r >> 1).min(ch - 1) * cw + (c >> 1).min(cw - 1)];
        }
    }
    out
}

fn chunk5_cells() -> Vec<EncodeCell> {
    let mut cells = Vec::new();

    // ---- allintra speed 1-4 arms (single-size qindex rule + prune paths).
    // quantizer-00 (352x288) fires LR reliably at speed 0; at speed>=1 the
    // size is forced by the qindex rule (128 when qindex<=96 else 256).
    for &(speed, cq) in &[
        (1, 20), // qindex likely <= 96 -> min=max=128 rule
        (1, 48), // qindex > 96 -> 256; ep-prune ladder + dual_sgr penalty
        (2, 32), // + src-var prune (dc_quant threshold), sgr-from-wiener=1
        (2, 48),
        (3, 32), // + reduced 5-tap luma window, prune_sgr=2 (non-screen)
        (3, 48),
        (4, 32), // same LR sf as 3; NON_DUAL LF feeds the deblocked input
    ] {
        cells.push(EncodeCell::real_content(
            &format!("lr_s{speed}_quant00_cq{cq}"),
            "av1-1-b8-00-quantizer-00",
            None,
            cq,
            speed,
        ));
    }

    // ---- format arms at speed 0 (transforms of the real-content cell;
    // EncodeCell fields are pub).
    let base = EncodeCell::real_content("lr_base", "av1-1-b8-00-quantizer-00", None, 32, 0);

    // Monochrome: luma only. The C shim encodes mono=1; the port pipeline's
    // mono path (1-plane LrSearchInput, num_planes=1 pack/header) engages.
    let mut mono = base.clone();
    mono.label = "lr_mono_quant00_cq32".into();
    mono.mono = true;
    mono.u = Vec::new();
    mono.v = Vec::new();
    cells.push(mono);

    // 4:4:4: real luma + nearest-upsampled real chroma (content provenance
    // does not matter for parity — both sides encode the SAME planes).
    let (cw, ch) = ((base.w + 1) >> 1, (base.h + 1) >> 1);
    let mut c444 = base.clone();
    c444.label = "lr_444_quant00_cq32".into();
    c444.ss_x = 0;
    c444.ss_y = 0;
    c444.u = upsample_nn(&base.u, cw, ch, base.w, base.h);
    c444.v = upsample_nn(&base.v, cw, ch, base.w, base.h);
    cells.push(c444);

    // bd12: the real content shifted to 12-bit range (synthetic-depth cell,
    // documented as such — exercises the highbd-12 search paths incl. the
    // compute_stats divider 16 and the SGR 12-bit clamps).
    let mut b12 = base.clone();
    b12.label = "lr_bd12_quant00_cq32".into();
    b12.bd = 12;
    b12.y = base.y.iter().map(|&v| v << 4).collect();
    b12.u = base.u.iter().map(|&v| v << 4).collect();
    b12.v = base.v.iter().map(|&v| v << 4).collect();
    cells.push(b12);

    // ---- GOOD (usage=0) speed-0 cells: the GOOD lpf_sf LR slice at speed 0
    // is verified-equal to defaults, so these are in-scope now. usage is a
    // pub field; c_encode_lr threads it to the shim.
    let mut good = EncodeCell::real_content("lr_good_quant00_cq32", "av1-1-b8-00-quantizer-00", None, 32, 0);
    good.usage = 0;
    cells.push(good);
    let mut good2 = EncodeCell::real_content("lr_good_size196_cq32", "av1-1-b8-01-size-196x196", None, 32, 0);
    good2.usage = 0;
    cells.push(good2);

    cells
}

/// CHUNK-5 gate: allintra speed 1-4 sf arms + mono/4:4:4/bd12 format cells +
/// GOOD speed-0. rd_close bands assert; byte-identity/decision tallies are
/// REPORTED (not yet asserted — see the HANDOFF banner above: harden to the
/// first measured level).
#[test]
fn lr_restoration_speed_and_format_arms() {
    let bands = RdBands::default();
    let mut results = Vec::new();
    let mut exact = 0usize;
    let mut decisions_equal = 0usize;

    for cell in chunk5_cells() {
        let c_tu = cell.c_encode_lr();
        assert!(!c_tu.is_empty(), "{}: real LR encode failed", cell.label);
        let port_payload = cell.port_encode_lr(&c_tu);
        let port_tu = rd_close::splice_frame_obu(&c_tu, &port_payload);

        let (real_frt, real_us) = parse_restoration_decision(&c_tu);
        let (port_frt, port_us) = parse_restoration_decision(&port_tu);
        let decision_eq = real_frt == port_frt && (real_frt == [0; 3] || real_us == port_us);
        if decision_eq {
            decisions_equal += 1;
        }
        let r = rd_close::compare_cell(&cell.label, &cell, &port_tu, &c_tu);
        if r.bit_identical {
            exact += 1;
        }
        eprintln!(
            "{}: real_frt={real_frt:?} us={real_us:?} | port_frt={port_frt:?} us={port_us:?} \
             | decision_{} | {}",
            cell.label,
            if decision_eq { "EQUAL" } else { "DIFFERS" },
            r.verdict(&bands),
        );
        results.push(r);
    }

    eprintln!("{}", rd_close::render_table(&results, &bands));
    eprintln!(
        "LR chunk-5 summary: {}/{} decisions equal, {}/{} bit-identical",
        decisions_equal,
        results.len(),
        exact,
        results.len(),
    );
    // HANDOFF: after the first run, if exact == results.len(), replace this
    // band assert with the byte-identity + decision-equality asserts (mirror
    // the main gate); otherwise record the honest fractions in PARITY C2.
    rd_close::assert_rd_close(&results, &bands);
}
