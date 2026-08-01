//! KB-21 open item — unit lock for the QM x speed>=4 arm.
//!
//! `av1_setup_quant` (`upstream/av1/encoder/encodemb.c:353`) ends by NULLing
//! `qparam->qmatrix` / `qparam->iqmatrix` (`:367-368`).
//! `skip_trellis_opt_based_on_satd` (`upstream/av1/encoder/tx_search.c:1981`)
//! calls it (`:2001-2006`) on every path that does not take its early return
//! (`:1988`), and that call sits INSIDE `search_tx_type`'s per-tx-type loop —
//! between the `av1_setup_qmatrix` that installs the matrix (`:2204-2207`) and
//! the `av1_quant` that would use it (`:2221`). So with quantization matrices
//! enabled at any speed whose SATD threshold is finite, the whole tx-type search
//! quantizes FLAT.
//!
//! This locks BOTH directions of that claim over speeds 0..9 and both QM states:
//! the arm must be UNREACHABLE (hence byte-inert, hence the pre-existing speed
//! 0..3 QM gates unaffected) below ALLINTRA speed 4, and must FIRE from speed 4
//! up (without which the fix would be dead code). The end-to-end byte evidence is
//! `aom-bench/tests/kb21_qm_speed4.rs::qm_speed_map_byte_matches`.

use aom_encode::speed_features::{DEFAULT_EVAL, MODE_EVAL, SpeedFeatures, WINNER_MODE_EVAL};
use aom_encode::tx_search::{qparam_qm_level_in_search, satd_trellis_skip_arm_runs};

/// A non-flat frame QM level (`NUM_QM_LEVELS - 1 == 15` is the flat one), of the
/// kind `aom_get_qmlevel_allintra` derives for a `--qm-min=4 --qm-max=10` encode.
const QM_LEVEL: usize = 8;

const STAGES: [(usize, &str); 3] = [
    (DEFAULT_EVAL, "DEFAULT_EVAL"),
    (MODE_EVAL, "MODE_EVAL"),
    (WINNER_MODE_EVAL, "WINNER_MODE_EVAL"),
];

/// The QUANT_PARAM-level quant matrix inside `search_tx_type` survives at
/// ALLINTRA speeds 0..3 and is dropped from speed 4 up — asserted in BOTH
/// directions, for BOTH QM states, at every eval stage.
#[test]
fn qparam_qm_is_dropped_exactly_from_allintra_speed_4() {
    for speed in 0..=9i32 {
        let sf = SpeedFeatures::set_allintra(speed, false, false);
        // Stages whose SATD threshold is finite, i.e. where the helper runs its
        // body and therefore re-runs `av1_setup_quant`.
        let mut firing: Vec<&str> = Vec::new();
        for (stage, name) in STAGES {
            let thresh = sf
                .tx_type_search_policy_for_stage(stage, false, 0)
                .coeff_opt_satd_threshold;

            // QM OFF: flat in, flat out, at every speed and stage. Without this
            // half the lock could not distinguish "the arm drops the matrix" from
            // "there was never a matrix".
            assert_eq!(
                qparam_qm_level_in_search(thresh, false, None),
                None,
                "speed {speed} {name}: QM-off must stay flat"
            );

            // The block-level trellis being off takes the early return whatever
            // the threshold is (`skip_trellis ||` is the first term at :1988), so
            // the matrix survives at EVERY speed on that path.
            assert_eq!(
                qparam_qm_level_in_search(thresh, true, Some(QM_LEVEL)),
                Some(QM_LEVEL),
                "speed {speed} {name}: with the block-level trellis already off, \
                 `skip_trellis_opt_based_on_satd` early-returns before it can \
                 re-run av1_setup_quant, so the installed matrix must survive"
            );

            let arm = satd_trellis_skip_arm_runs(thresh, false);
            assert_eq!(
                arm,
                thresh != u32::MAX,
                "speed {speed} {name}: the arm runs iff the threshold is finite"
            );
            assert_eq!(
                qparam_qm_level_in_search(thresh, false, Some(QM_LEVEL)),
                if arm { None } else { Some(QM_LEVEL) },
                "speed {speed} {name}: QM must be dropped iff the SATD arm runs"
            );
            if arm {
                firing.push(name);
            }
        }

        // WINNER_MODE_EVAL is UINT_MAX in every `coeff_opt_thresholds` row
        // (speed_features.c:88-98), so it can never drop the matrix.
        assert!(
            !firing.contains(&"WINNER_MODE_EVAL"),
            "speed {speed}: WINNER_MODE_EVAL has a UINT_MAX satd threshold in \
             every coeff_opt_thresholds row and must never drop the matrix"
        );

        if speed <= 3 {
            // INERTNESS: `perform_coeff_opt` is 1/2/3 here (speed_features.c
            // :383/:415/:433) and rows 1..3 are UINT_MAX in the satd column of
            // every stage — which is why the landed speed-0..3 QM byte gates are
            // untouched by this fix.
            assert!(
                firing.is_empty(),
                "speed {speed}: the SATD arm must be unreachable, so a QM encode \
                 keeps its matrix through the whole tx-type search; firing: \
                 {firing:?}"
            );
        } else {
            // REACHABILITY (non-vacuity): without this half the lock would pass
            // on a port that never enables the arm at all, and the QM drop would
            // be dead code.
            assert!(
                !firing.is_empty(),
                "speed {speed}: no eval stage has a finite coeff_opt_satd_threshold, \
                 so the QM drop modelled here could never fire \
                 (speed_features.c:88-98 coeff_opt_thresholds x perform_coeff_opt \
                 5 at :493 / 6 at :555)"
            );
        }
    }
}
