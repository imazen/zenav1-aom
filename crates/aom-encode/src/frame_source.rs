//! Frame-source decisions from `av1/encoder/encode_strategy.c` — the parts of
//! the lookahead walk that are *decisions* rather than ring-buffer plumbing.
//!
//! [`crate::ref_gop`] deliberately excludes this file's lookahead functions as
//! orchestration. That was too broad for three of them: they read only the
//! per-entry FLAGS and a handful of scalars, and what they return changes the
//! coded frame type or whether a frame is coded at all. They are ported here,
//! as pure functions over the flags rather than over a `struct lookahead_ctx`.
//!
//! | Rust | C (`av1/encoder/encode_strategy.c`) |
//! |---|---|
//! | [`forced_keyframe_pending`] | `is_forced_keyframe_pending` (:304) |
//! | [`allow_show_existing`] | `allow_show_existing` (:407, static) |
//! | [`adjust_frame_rate`] | `adjust_frame_rate` (:233, static) |
//! | [`new_framerate`] | `av1_new_framerate` (encoder.c:317), its clamp |
//!
//! What stays out, and why: `choose_frame_source` (:327) picks WHICH lookahead
//! entry to encode and whether to pop it, which is ring-buffer state the port
//! does not have; `denoise_and_encode` and `av1_encode_strategy` are pipeline
//! orchestration.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/frame_source_diff.rs`. **Tier 1** for
//! [`forced_keyframe_pending`] (exported; the shim builds a real
//! `struct lookahead_ctx` — the type is public in `lookahead.h`, so the ring
//! can be filled directly without allocating frame buffers) and for
//! [`new_framerate`]'s clamp. **Tier 4** for [`allow_show_existing`] and
//! [`adjust_frame_rate`], which are `static` with no exported symbol and whose
//! only caller is `av1_encode_strategy`; those two are hand-derived from the C
//! source with unit tests in this file only.

/// `AOM_EFLAG_FORCE_KF` (aom/aom_encoder.h:379).
pub const AOM_EFLAG_FORCE_KF: u32 = 1 << 0;
/// `AOM_EFLAG_ERROR_RESILIENT` (aom/aomcx.h:155).
pub const AOM_EFLAG_ERROR_RESILIENT: u32 = 1 << 28;
/// `AOM_EFLAG_SET_S_FRAME` (aom/aomcx.h:161).
pub const AOM_EFLAG_SET_S_FRAME: u32 = 1 << 29;

/// `is_forced_keyframe_pending` (encode_strategy.c:304): the index of the
/// first lookahead entry, at or before `up_to_index`, that carries a forced
/// key frame.
///
/// `flags` is the lookahead entries' `e->flags` in peek order — entry `i` is
/// `av1_lookahead_peek(lookahead, i, stage)`. A short slice models C reaching
/// the end of the buffer (`e == NULL`), which C reports as "no forced key
/// frame pending" rather than as an error.
///
/// Note the test C uses: `e->flags == AOM_EFLAG_FORCE_KF`, an EQUALITY, not a
/// bit test. An entry that also carries, say, `AOM_EFLAG_ERROR_RESILIENT` does
/// NOT match. Rewriting it as `flags & AOM_EFLAG_FORCE_KF` changes the answer.
#[must_use]
pub fn forced_keyframe_pending(flags: &[u32], up_to_index: usize) -> Option<usize> {
    flags
        .iter()
        .take(up_to_index + 1)
        .position(|&f| f == AOM_EFLAG_FORCE_KF)
}

/// `allow_show_existing` (encode_strategy.c:407): may this frame be coded as a
/// `show_existing_frame`?
///
/// `lookahead_src_flags` is `None` when the lookahead is empty
/// (`av1_lookahead_peek(.., 0, ..) == NULL`), which C answers `1` for —
/// an empty lookahead cannot contradict a show-existing.
///
/// A show-existing must not coincide with an error-resilient or S-frame,
/// except on a key frame, which depends on no previous frame.
#[must_use]
pub fn allow_show_existing(
    frame_number: u32,
    lookahead_src_flags: Option<u32>,
    cfg_error_resilient_mode: bool,
    cfg_enable_sframe: bool,
    frames_to_key: i32,
    frame_flags: u32,
) -> bool {
    if frame_number == 0 {
        return false;
    }
    let Some(src_flags) = lookahead_src_flags else {
        return true;
    };
    let is_error_resilient =
        cfg_error_resilient_mode || (src_flags & AOM_EFLAG_ERROR_RESILIENT) != 0;
    let is_s_frame = cfg_enable_sframe || (src_flags & AOM_EFLAG_SET_S_FRAME) != 0;
    let is_key_frame = frames_to_key == 0 || (frame_flags & crate::ref_gop::frame_flags::KEY) != 0;
    !(is_error_resilient || is_s_frame) || is_key_frame
}

/// `av1_new_framerate` (encoder.c:317), the clamp half: a frame rate below
/// `0.1` is replaced by `30`, not by the minimum.
///
/// The other half is the tail call to `av1_rc_update_framerate`, which is
/// [`crate::ratectrl_init::update_framerate`].
#[must_use]
pub fn new_framerate(framerate: f64) -> f64 {
    if framerate < 0.1 { 30.0 } else { framerate }
}

/// `TimeStamps` (encoder.h) as far as `adjust_frame_rate` reads and writes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimeStamps {
    /// `time_stamps->first_ts_start`.
    pub first_ts_start: i64,
    /// `time_stamps->prev_ts_start`.
    pub prev_ts_start: i64,
    /// `time_stamps->prev_ts_end`.
    pub prev_ts_end: i64,
}

/// What [`adjust_frame_rate`] produces: the new frame rate to feed
/// [`new_framerate`], and the timestamps to store back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameRateUpdate {
    /// `cpi->new_framerate`, or `None` when `this_duration == 0` and C leaves
    /// both `new_framerate` and `cpi->framerate` untouched.
    pub new_framerate: Option<f64>,
    /// The frame rate C actually passes to `av1_new_framerate` — the same as
    /// `new_framerate` except on the averaging path during a parallel encode,
    /// where C passes the OLD `cpi->framerate` instead.
    pub framerate_to_apply: Option<f64>,
    /// The timestamps to write back into [`TimeStamps`].
    pub time_stamps: TimeStamps,
}

/// `adjust_frame_rate` (encode_strategy.c:233), the non-RT, non-SVC path.
///
/// `ts_start` / `ts_end` are the frame's start and expected-next timestamps in
/// 10 MHz ticks, so `10_000_000.0 / duration` is a frame rate.
///
/// Three behaviours worth not "tidying":
/// * on the FIRST frame (`ts_start == first_ts_start`) the duration is
///   `ts_end - ts_start`; afterwards it is `ts_end - prev_ts_end`, which is a
///   different quantity — it spans any gap since the previous frame ENDED.
/// * the 10% step test is C integer division:
///   `(this - last) * 10 / last`, so a change of exactly 10% gives `1` and
///   anything under 10% gives `0`. It is also signed, so a duration that
///   SHRANK by more than 10% gives a negative non-zero step, which counts.
/// * the averaging path stores `new_framerate` but applies the OLD
///   `cpi->framerate` while a parallel encode is in flight. Both are returned
///   so a caller cannot silently conflate them.
///
/// The `is_one_pass_rt_params` / SVC early return at the top is not ported —
/// see the module note.
#[must_use]
pub fn adjust_frame_rate(
    time_stamps: TimeStamps,
    ts_start: i64,
    ts_end: i64,
    current_framerate: f64,
    frame_parallel: bool,
) -> FrameRateUpdate {
    let (this_duration, mut step) = if ts_start == time_stamps.first_ts_start {
        (ts_end - ts_start, 1i64)
    } else {
        let last_duration = time_stamps.prev_ts_end - time_stamps.prev_ts_start;
        let this_duration = ts_end - time_stamps.prev_ts_end;
        // A step update if the duration changed by 10%. C's `(int)` truncates
        // a 64-bit quotient; the operands here are durations in ticks, so the
        // quotient fits, but the truncation direction is toward zero.
        let step = if last_duration != 0 {
            (this_duration - last_duration) * 10 / last_duration
        } else {
            0
        };
        (this_duration, step)
    };
    if this_duration == 0 {
        step = 0;
    }

    let (new_framerate, framerate_to_apply) = if this_duration == 0 {
        (None, None)
    } else if step != 0 {
        let f = 10_000_000.0 / this_duration as f64;
        (Some(f), Some(f))
    } else {
        // Average this frame's rate into the last second's average frame rate.
        // Before a full second has elapsed, average over the interval seen.
        let interval = ((ts_end - time_stamps.first_ts_start) as f64).min(10_000_000.0);
        let mut avg_duration = 10_000_000.0 / current_framerate;
        avg_duration *= interval - avg_duration + this_duration as f64;
        avg_duration /= interval;
        let f = 10_000_000.0 / avg_duration;
        // For parallel frames cpi->framerate is updated later, in
        // av1_post_encode_updates, so the OLD value is applied now.
        let applied = if frame_parallel { current_framerate } else { f };
        (Some(f), Some(applied))
    };

    FrameRateUpdate {
        new_framerate,
        framerate_to_apply,
        time_stamps: TimeStamps {
            first_ts_start: time_stamps.first_ts_start,
            prev_ts_start: ts_start,
            prev_ts_end: ts_end,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_keyframe_test_is_equality_not_a_bit_test() {
        // C compares `e->flags == AOM_EFLAG_FORCE_KF`. An entry carrying the
        // bit PLUS another flag does not match.
        let flags = [
            0,
            AOM_EFLAG_FORCE_KF | AOM_EFLAG_ERROR_RESILIENT,
            AOM_EFLAG_FORCE_KF,
        ];
        assert_eq!(forced_keyframe_pending(&flags, 2), Some(2));
        assert_eq!(forced_keyframe_pending(&flags, 1), None);
    }

    #[test]
    fn a_short_lookahead_reports_no_forced_keyframe() {
        // C returns -1 when peek hits the end of the buffer, even though the
        // requested index was never examined.
        assert_eq!(forced_keyframe_pending(&[], 5), None);
        assert_eq!(forced_keyframe_pending(&[0, 0], 9), None);
    }

    #[test]
    fn allow_show_existing_frame_zero_is_never_allowed() {
        assert!(!allow_show_existing(0, Some(0), false, false, 5, 0));
    }

    #[test]
    fn allow_show_existing_empty_lookahead_is_allowed() {
        assert!(allow_show_existing(7, None, true, true, 5, 0));
    }

    #[test]
    fn allow_show_existing_key_frame_overrides_the_veto() {
        // Error-resilient vetoes it...
        assert!(!allow_show_existing(7, Some(0), true, false, 5, 0));
        // ...unless the frame is a key frame, by either of C's two tests.
        assert!(allow_show_existing(7, Some(0), true, false, 0, 0));
        assert!(allow_show_existing(
            7,
            Some(0),
            true,
            false,
            5,
            crate::ref_gop::frame_flags::KEY
        ));
        // The per-entry S-frame flag vetoes it too.
        assert!(!allow_show_existing(
            7,
            Some(AOM_EFLAG_SET_S_FRAME),
            false,
            false,
            5,
            0
        ));
    }

    #[test]
    fn new_framerate_replaces_a_tiny_rate_with_thirty() {
        assert_eq!(new_framerate(0.09).to_bits(), 30.0f64.to_bits());
        assert_eq!(new_framerate(0.0).to_bits(), 30.0f64.to_bits());
        assert_eq!(new_framerate(0.1).to_bits(), 0.1f64.to_bits());
        assert_eq!(new_framerate(60.0).to_bits(), 60.0f64.to_bits());
    }

    #[test]
    fn adjust_frame_rate_first_frame_takes_the_step_path() {
        let ts = TimeStamps {
            first_ts_start: 1000,
            prev_ts_start: 0,
            prev_ts_end: 0,
        };
        // ts_start == first_ts_start => step = 1 => the rate is the direct
        // reciprocal of this frame's duration, with no averaging.
        let out = adjust_frame_rate(ts, 1000, 1000 + 333_333, 30.0, false);
        assert_eq!(out.new_framerate, out.framerate_to_apply);
        let f = out
            .new_framerate
            .expect("a non-zero duration sets the rate");
        assert!((f - 30.000_030).abs() < 1e-3, "{f}");
        assert_eq!(out.time_stamps.prev_ts_start, 1000);
        assert_eq!(out.time_stamps.prev_ts_end, 1000 + 333_333);
    }

    #[test]
    fn adjust_frame_rate_zero_duration_changes_no_rate() {
        let ts = TimeStamps {
            first_ts_start: 0,
            prev_ts_start: 1000,
            prev_ts_end: 2000,
        };
        // ts_end == prev_ts_end => this_duration == 0 => C's `if
        // (this_duration)` is false and neither rate is written, but the
        // timestamps still are.
        let out = adjust_frame_rate(ts, 1500, 2000, 30.0, false);
        assert_eq!(out.new_framerate, None);
        assert_eq!(out.framerate_to_apply, None);
        assert_eq!(out.time_stamps.prev_ts_start, 1500);
        assert_eq!(out.time_stamps.prev_ts_end, 2000);
    }

    #[test]
    fn adjust_frame_rate_step_threshold_is_integer_division() {
        let ts = TimeStamps {
            first_ts_start: 0,
            prev_ts_start: 0,
            prev_ts_end: 1000,
        };
        // last_duration = 1000. A 9% growth gives (90 * 10) / 1000 = 0 -> the
        // averaging path; a 10% growth gives (100 * 10) / 1000 = 1 -> the step
        // path, where new_framerate is the direct reciprocal.
        let avg = adjust_frame_rate(ts, 500, 1000 + 1090, 30.0, false);
        let step = adjust_frame_rate(ts, 500, 1000 + 1100, 30.0, false);
        assert_ne!(
            avg.new_framerate.unwrap().to_bits(),
            (10_000_000.0f64 / 1090.0).to_bits(),
            "9% must NOT take the step path"
        );
        assert_eq!(
            step.new_framerate.unwrap().to_bits(),
            (10_000_000.0f64 / 1100.0).to_bits(),
            "10% must take the step path"
        );
    }

    #[test]
    fn adjust_frame_rate_parallel_applies_the_old_rate() {
        let ts = TimeStamps {
            first_ts_start: 0,
            prev_ts_start: 0,
            prev_ts_end: 1000,
        };
        // A sub-10% change takes the averaging path, where a parallel encode
        // applies cpi->framerate and stores new_framerate for later.
        let out = adjust_frame_rate(ts, 500, 1000 + 1050, 30.0, true);
        assert_eq!(out.framerate_to_apply.unwrap().to_bits(), 30.0f64.to_bits());
        assert_ne!(
            out.new_framerate.unwrap().to_bits(),
            30.0f64.to_bits(),
            "new_framerate must still be the averaged value"
        );
    }
}
