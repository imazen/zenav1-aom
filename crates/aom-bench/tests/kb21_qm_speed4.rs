//! KB-21 open item — the QM x `--cpu-used >= 4` axis.
//!
//! `av1_setup_quant` (`upstream/av1/encoder/encodemb.c:353-368`) ends by NULLing
//! `qparam->qmatrix` / `qparam->iqmatrix` (`:367-368`), and
//! `skip_trellis_opt_based_on_satd` (`upstream/av1/encoder/tx_search.c:1981-2008`)
//! calls it (`:2001-2006`) on every path that does NOT take its early return
//! (`:1988`, `skip_trellis || coeff_opt_satd_threshold == UINT_MAX`). Inside
//! `search_tx_type`'s per-tx-type loop that lands between the `av1_setup_qmatrix`
//! that INSTALLS the matrix (`:2204-2207`) and the `av1_quant` that would USE it
//! (`:2221`) — so with QM enabled at a speed whose SATD threshold is finite, C
//! quantizes every candidate of every txb with NO quant matrix, and the port
//! (which keeps one attached) diverges.
//!
//! The reachable band is exactly the finite-satd-threshold rows of
//! `coeff_opt_thresholds` (`speed_features.c:88-98`): `perform_coeff_opt` is 1/2/3
//! at ALLINTRA speeds 0/2/3 (`:383/:415/:433`, satd column UINT_MAX in every
//! MODE_EVAL slot), 5 at speeds 4-5 (`:493`) and 6 at speeds >= 6 (`:555`) — rows
//! whose DEFAULT_EVAL and MODE_EVAL satd thresholds are finite (97 / 16).
//! `WINNER_MODE_EVAL` is UINT_MAX in every row, so it is never affected.

use aom_bench::{EncodeCell, ToggleKnobs};

/// The QM range the landed speed-0 QM gate uses for its allintra arm
/// (`encoder_gate_qm_on_e2e`): `--qm-min=4 --qm-max=10`, which derives
/// per-qindex levels across 4..=10 rather than pinning one.
const QM: (i32, i32) = (4, 10);

/// Speeds whose ALLINTRA `coeff_opt_thresholds` row has a FINITE SATD threshold
/// in at least one eval stage — i.e. where `skip_trellis_opt_based_on_satd` runs
/// its body (and therefore re-runs `av1_setup_quant`) instead of early-returning.
/// Speed 8/9 are the nonrd PICKMODE path, which never enters `search_tx_type`.
const RD_SPEEDS: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

struct Cell {
    label: &'static str,
    vector: &'static str,
    crop: Option<(usize, usize, usize, usize)>,
    cq: i32,
}

const CELLS: [Cell; 3] = [
    Cell {
        label: "q00 420 64x64@64,64",
        vector: "av1-1-b8-00-quantizer-00",
        crop: Some((64, 64, 64, 64)),
        cq: 32,
    },
    Cell {
        label: "q00 420 128x128@64,64",
        vector: "av1-1-b8-00-quantizer-00",
        crop: Some((128, 128, 64, 64)),
        cq: 32,
    },
    Cell {
        label: "film50 420 64x64@96,64",
        vector: "av1-1-b8-23-film_grain-50",
        crop: Some((64, 64, 96, 64)),
        cq: 32,
    },
];

fn run(cell: &Cell, speed: i32) -> (bool, usize, usize) {
    let ec = EncodeCell::real_content(cell.label, cell.vector, cell.crop, cell.cq, speed);
    let c_stream = ec.c_encode_qm(QM.0, QM.1);
    let want = EncodeCell::frame_obu_payload(&c_stream);
    let knobs = ToggleKnobs {
        qm: Some(QM),
        ..Default::default()
    };
    let got = ec.port_encode_with(&c_stream, &knobs);
    (got == want, got.len(), want.len())
}

/// **Anti-vacuity witness.** QM must actually change the REFERENCE bitstream at
/// every speed in the band — otherwise a byte-match below would prove nothing
/// about the port's QM path. (`c_encode` is the identical config with QM off.)
#[test]
fn qm_changes_the_reference_stream_at_every_rd_speed() {
    for &speed in &RD_SPEEDS {
        for cell in &CELLS {
            let ec = EncodeCell::real_content(cell.label, cell.vector, cell.crop, cell.cq, speed);
            let off = ec.c_encode();
            let on = ec.c_encode_qm(QM.0, QM.1);
            assert!(!off.is_empty() && !on.is_empty());
            assert_ne!(
                off, on,
                "{} cpu-used={speed}: QM-on must change the C bitstream",
                cell.label
            );
        }
    }
}

/// **KB-21 open item, byte gate.** QM-on real-content encodes must byte-match
/// real aomenc at every RD speed 0..7 — including the speed >= 4 band where
/// `skip_trellis_opt_based_on_satd`'s `av1_setup_quant` call drops the quant
/// matrix out from under `av1_quant`.
///
/// Ignored by default: 24 real aomenc encodes plus 24 full port encodes, several
/// of them at speed 0 on a 128x128 frame. Run with `--ignored`.
#[test]
#[ignore = "24 real-aomenc + 24 port encodes; run explicitly"]
fn qm_speed_map_byte_matches() {
    let mut diverging: Vec<String> = Vec::new();
    for &speed in &RD_SPEEDS {
        for cell in &CELLS {
            let (ok, got, want) = run(cell, speed);
            println!(
                "{:<24} cpu-used={speed}  {}  port {got} B / real {want} B",
                cell.label,
                if ok { "MATCH   " } else { "MISMATCH" }
            );
            if !ok {
                diverging.push(format!("{} cpu-used={speed}", cell.label));
            }
        }
    }
    assert!(
        diverging.is_empty(),
        "QM-on encodes must byte-match real aomenc; diverging: {diverging:?}"
    );
}
