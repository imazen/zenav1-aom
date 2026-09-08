//! The side-effect-free configuration-support query, and the property that
//! makes it worth having: it must answer with the SAME predicates the encoder
//! enforces, so a caller that asks first and encodes second never gets a
//! surprise. A query that is merely a SUBSET of the encoder's checks would let
//! `validate_configuration() == Ok` be followed by `encode_key_frame() == Err`,
//! which is exactly the promise a backend router cannot afford to have broken.
//!
//! Scope, stated so it is not over-read: this reports CONFIGURATION support. It
//! is not a promise that a caller's source buffers are the right size (that is
//! `KeyFrameError::PlaneSize`, which needs the planes) nor that the allocation
//! will succeed.

use aom_encode::key_frame::{encode_key_frame, KeyFrameConfig, KeyFramePlanes, MAX_FRAME_DIM};

/// Every chroma format x bit depth x speed the standalone shell gates must be
/// accepted by the query — otherwise the query under-reports and a caller
/// routes a supported still somewhere else.
#[test]
fn query_accepts_native_still_formats() {
    for depth in [8, 10, 12] {
        for (mono, ss_x, ss_y) in [(true, 1, 1), (false, 1, 1), (false, 1, 0), (false, 0, 0)] {
            for speed in [0, 5, 9] {
                let mut config = KeyFrameConfig::allintra_speed0(65, 67, depth, mono, ss_x, ss_y, 0);
                config.cpu_used = speed;
                config.validate_configuration().expect("gated native format");
            }
        }
    }
}

/// SB128 and an explicitly-requested multi-tile grid are both inside the gated
/// envelope, so the tile derivation the query now runs must not refuse them.
///
/// HONEST NOTE: the tile REFUSAL itself has no test, because no configuration
/// is known to reach it — `av1_calculate_tile_cols`/`_rows` always return
/// `1 << log2` for uniform spacing, which is the only spacing this shell
/// emits. It is shared structurally instead: `derive_tiles` has exactly one
/// definition and both the query and `encode_key_frame` consume that one call,
/// so they cannot disagree even though neither can be observed disagreeing.
/// Without this the tile predicate could be satisfied vacuously by refusing
/// everything that reaches it.
#[test]
fn query_accepts_the_tile_and_superblock_envelope() {
    for sb128 in [false, true] {
        for (cols_log2, rows_log2) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let mut config = KeyFrameConfig::allintra_speed0(512, 512, 8, false, 1, 1, 32);
            config.sb_size_128 = sb128;
            config.tile_columns_log2 = cols_log2;
            config.tile_rows_log2 = rows_log2;
            let tiles = config
                .derive_tiles()
                .unwrap_or_else(|e| panic!("sb128={sb128} tiles={cols_log2}x{rows_log2}: {e}"));
            assert_eq!(
                tiles.rows * tiles.cols,
                1usize << (tiles.log2_cols + tiles.log2_rows),
                "the derivation must agree with itself"
            );
            config
                .validate_configuration()
                .unwrap_or_else(|e| panic!("sb128={sb128} tiles={cols_log2}x{rows_log2}: {e}"));
        }
    }
}

/// THE gate: query and encoder must return the IDENTICAL refusal, and the
/// encoder must reach it before it looks at a single source sample (the planes
/// here are empty, so anything that read them would fail differently).
#[test]
fn query_and_encoder_reject_invalid_config_before_reading_planes() {
    let base = KeyFrameConfig::allintra_speed0(65, 67, 8, false, 1, 1, 32);
    let mut cases = [base; 9];
    cases[0].usage = 0;
    cases[1].cpu_used = 10;
    cases[2].bit_depth = 9;
    cases[3].width = 0;
    cases[4].cq_level = 64;
    cases[5].ss_x = 0;
    cases[6].monochrome = true;
    cases[6].ss_y = 0;
    // A dimension one past what `frame_width_bits_minus_1` (`f(4)`) can code.
    // Before it was refused, `write_sequence_header` emitted
    // `write_literal(16, 4)` == 0 — signalling one width bit and then writing
    // seventeen, i.e. a silently corrupt sequence header rather than an error.
    cases[7].width = MAX_FRAME_DIM + 1;
    cases[8].height = MAX_FRAME_DIM + 1;
    for (i, config) in cases.into_iter().enumerate() {
        let query = config
            .validate_configuration()
            .expect_err(&format!("case {i} must be refused by the query"));
        let actual = encode_key_frame(
            KeyFramePlanes {
                y: &[],
                u: &[],
                v: &[],
            },
            &config,
        )
        .expect_err(&format!("case {i} must be refused by the encoder"));
        assert_eq!(query, actual, "case {i}: query and encoder must agree");
    }
}

/// The largest codeable dimension is ACCEPTED, so the ceiling above is a
/// boundary and not a blanket refusal of large frames. Query-only: encoding a
/// 65536-wide frame is not something to do in a unit test.
#[test]
fn the_dimension_ceiling_is_a_boundary_not_a_wall() {
    let mut config = KeyFrameConfig::allintra_speed0(MAX_FRAME_DIM, 64, 8, false, 1, 1, 32);
    config.validate_configuration().expect("65536 is codeable");
    config.width = 64;
    config.height = MAX_FRAME_DIM;
    config.validate_configuration().expect("65536 is codeable");
}
