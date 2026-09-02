//! `set_bitstream_level_tier` (`av1/encoder/encoder.c:464`) — the derivation of
//! the sequence header's coded `seq_level_idx[op]` / `tier[op]`.
//!
//! # Which C function actually writes the header level?
//!
//! The 2026-07-30 coverage audit named `av1_get_seq_level_idx`
//! (`av1/encoder/level.c:1366`) as "the ONE true missing header ALGORITHM"
//! (`coverage-audit/COVERAGE.md` row 1.4). Traced first-hand against
//! `reference/libaom`, **that is the wrong target for the written header**:
//!
//! * `av1_get_seq_level_idx` computes the *achieved* level from per-operating-
//!   point accumulated `AV1LevelInfo` statistics (`check_level_constraints` over
//!   `level_info[op]`), and it runs **only** when
//!   `level_params->keep_level_stats >> op & 1` — which is **0 by default**
//!   (`encoder.c:912`; set only under `--target-seq-level-idx` /
//!   `SEQ_LEVEL_KEEP_STATS`). It never feeds the written sequence header.
//! * The **written** `seq_level_idx[op]` (and `tier[op]`) is produced by
//!   **`set_bitstream_level_tier`** (`encoder.c:464`, called at init from
//!   `init_seq_coding_tools`, `encoder.c:649`), whose core is the pure function
//!   `does_level_match(width, height, fps, …)` (`encoder.c:451`). That is what
//!   this module ports.
//!
//! Differential gate: `aom-encode/tests/seq_level_idx_diff.rs` — the port's
//! value vs the `seq_level_idx[0]` real aomenc CODES in its (reduced
//! still-picture) sequence header, over a dimension sweep that walks the
//! 2.0/2.1/3.0/3.1/4.0 rungs and all three `does_level_match` clauses.
//!
//! [Moved 2026-09-02 from the body of `seq_level_idx_diff.rs`, where it was
//! parked because a concurrent `cargo fmt` WIP change was holding
//! `aom-encode/src/lib.rs`. The test now calls these functions instead of
//! carrying its own copy.]

// ---- AV1_LEVEL enum values (`av1/common/enums.h`) — the subset
//      `set_bitstream_level_tier` can produce, plus the sentinels. ----
/// `SEQ_LEVEL_2_0`.
pub const SEQ_LEVEL_2_0: i32 = 0;
/// `SEQ_LEVEL_2_1`.
pub const SEQ_LEVEL_2_1: i32 = 1;
/// `SEQ_LEVEL_3_0`.
pub const SEQ_LEVEL_3_0: i32 = 4;
/// `SEQ_LEVEL_3_1`.
pub const SEQ_LEVEL_3_1: i32 = 5;
/// `SEQ_LEVEL_4_0` — also the threshold at/above which the sequence header
/// codes a `tier` bit (`av1_write_sequence_header_obu`).
pub const SEQ_LEVEL_4_0: i32 = 8;
/// `SEQ_LEVEL_4_1`.
pub const SEQ_LEVEL_4_1: i32 = 9;
/// `SEQ_LEVEL_5_0`.
pub const SEQ_LEVEL_5_0: i32 = 12;
/// `SEQ_LEVEL_5_1`.
pub const SEQ_LEVEL_5_1: i32 = 13;
/// `SEQ_LEVEL_5_2`.
pub const SEQ_LEVEL_5_2: i32 = 14;
/// `SEQ_LEVEL_6_0`.
pub const SEQ_LEVEL_6_0: i32 = 16;
/// `SEQ_LEVEL_6_1`.
pub const SEQ_LEVEL_6_1: i32 = 17;
/// `SEQ_LEVEL_6_2`.
pub const SEQ_LEVEL_6_2: i32 = 18;
/// `SEQ_LEVELS` — the count of real levels; a target at or above this is "no
/// explicit target".
pub const SEQ_LEVELS: i32 = 28;
/// `SEQ_LEVEL_MAX` — the default `target_seq_level_idx[op]`.
pub const SEQ_LEVEL_MAX: i32 = 31;

/// The default init framerate for a single still image through
/// `aom_codec_enc_config_default(.., AOM_USAGE_ALL_INTRA)`: `g_timebase ==
/// {1, 30}` (`av1/av1_cx_iface.c:5265`) → `init_framerate = den/num = 30`
/// (`av1_cx_iface.c:1197`).
pub const STILL_PICTURE_FPS: f64 = 30.0;

/// `does_level_match` (`av1/encoder/encoder.c:451`). Pure integer/double
/// arithmetic on the frame dimensions + framerate against one level's caps.
pub fn does_level_match(
    width: i32,
    height: i32,
    fps: f64,
    lvl_width: i32,
    lvl_height: i32,
    lvl_fps: f64,
    lvl_dim_mult: i32,
) -> bool {
    let lvl_luma_pels = lvl_width as i64 * lvl_height as i64;
    let lvl_display_sample_rate = lvl_luma_pels as f64 * lvl_fps;
    let luma_pels = width as i64 * height as i64;
    let display_sample_rate = luma_pels as f64 * fps;
    luma_pels <= lvl_luma_pels
        && display_sample_rate <= lvl_display_sample_rate
        && width <= lvl_width * lvl_dim_mult
        && height <= lvl_height * lvl_dim_mult
}

/// The `set_bitstream_level_tier` level ladder (`encoder.c:472-509`): the lowest
/// level whose dimension/display-rate caps the frame fits under, else
/// `SEQ_LEVEL_MAX`.
///
/// The `CONFIG_CWG_C013` 7.x/8.x arm (`encoder.c:512-535`) is gated on
/// `target_seq_level_idx[0]` in `[SEQ_LEVEL_7_0, SEQ_LEVEL_8_3]` and is
/// therefore unreachable at the default `target == SEQ_LEVEL_MAX`; it is
/// deliberately omitted (documented, not silently dropped).
pub fn inferred_seq_level_from_dims(width: i32, height: i32, fps: f64) -> i32 {
    if does_level_match(width, height, fps, 512, 288, 30.0, 4) {
        SEQ_LEVEL_2_0
    } else if does_level_match(width, height, fps, 704, 396, 30.0, 4) {
        SEQ_LEVEL_2_1
    } else if does_level_match(width, height, fps, 1088, 612, 30.0, 4) {
        SEQ_LEVEL_3_0
    } else if does_level_match(width, height, fps, 1376, 774, 30.0, 4) {
        SEQ_LEVEL_3_1
    } else if does_level_match(width, height, fps, 2048, 1152, 30.0, 3) {
        SEQ_LEVEL_4_0
    } else if does_level_match(width, height, fps, 2048, 1152, 60.0, 3) {
        SEQ_LEVEL_4_1
    } else if does_level_match(width, height, fps, 4096, 2176, 30.0, 2) {
        SEQ_LEVEL_5_0
    } else if does_level_match(width, height, fps, 4096, 2176, 60.0, 2) {
        SEQ_LEVEL_5_1
    } else if does_level_match(width, height, fps, 4096, 2176, 120.0, 2) {
        SEQ_LEVEL_5_2
    } else if does_level_match(width, height, fps, 8192, 4352, 30.0, 2) {
        SEQ_LEVEL_6_0
    } else if does_level_match(width, height, fps, 8192, 4352, 60.0, 2) {
        SEQ_LEVEL_6_1
    } else if does_level_match(width, height, fps, 8192, 4352, 120.0, 2) {
        SEQ_LEVEL_6_2
    } else {
        SEQ_LEVEL_MAX
    }
}

/// `set_bitstream_level_tier`'s written `seq_level_idx[op]`
/// (`encoder.c:541-545`): a higher explicit `target_seq_level_idx[op]` overrides
/// the inferred level; at the default `target == SEQ_LEVEL_MAX`
/// (`>= SEQ_LEVELS`) the inferred level is used unchanged.
pub fn seq_header_seq_level_idx(
    width: i32,
    height: i32,
    fps: f64,
    target_seq_level_idx: i32,
) -> i32 {
    let level = inferred_seq_level_from_dims(width, height, fps);
    if target_seq_level_idx < SEQ_LEVELS && target_seq_level_idx > level {
        target_seq_level_idx
    } else {
        level
    }
}
