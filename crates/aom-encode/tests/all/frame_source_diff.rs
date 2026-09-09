//! Differential harness for the two EXPORTED frame-source decisions in
//! `av1/encoder/encode_strategy.c` / `encoder.c`, vs the REAL C libaom
//! v3.14.1. **Tier 1** — both drive the real exported symbol through
//! `crates/aom-sys-ref/shim/refgop_shim.c`.
//!
//! | test | C oracle |
//! |---|---|
//! | `forced_keyframe_pending_matches_c` | `is_forced_keyframe_pending` |
//! | `new_framerate_clamp_matches_c` | `av1_new_framerate` |
//!
//! The other two functions in `aom_encode::frame_source`
//! (`allow_show_existing`, `adjust_frame_rate`) are `static` in C with no
//! exported symbol and no exported caller short of `av1_encode_strategy`;
//! they are **tier 4** and are covered by unit tests in the module itself.
//! That split is stated in the module docs, not implied by this file's
//! silence about them.
//!
//! # Why `read_idx` is swept
//! `av1_lookahead_peek` maps peek index `i` to ring slot
//! `(read_idx + i) mod max_sz`. At `read_idx == 0` that is the identity, so a
//! port (or a shim) that ignored the wrap would pass. The sweep drives every
//! `read_idx` in `0..n`, which makes the mapping load-bearing.

use aom_encode::frame_source::{AOM_EFLAG_FORCE_KF, forced_keyframe_pending, new_framerate};
use aom_sys_ref::{ref_is_forced_keyframe_pending, ref_new_framerate};

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
    fn below(&mut self, n: u32) -> i32 {
        (self.next() % u64::from(n)) as i32
    }
}

/// Flag values that straddle the equality-vs-bit-test distinction: the exact
/// FORCE_KF value, FORCE_KF with another bit set (which must NOT match), other
/// flags alone, and zero.
const FLAG_CHOICES: [u32; 6] = [
    0,
    AOM_EFLAG_FORCE_KF,
    AOM_EFLAG_FORCE_KF | (1 << 28),
    AOM_EFLAG_FORCE_KF | (1 << 29),
    1 << 28,
    1 << 29,
];

#[test]
fn forced_keyframe_pending_matches_c() {
    let mut rng = Rng(0x0f0f_1e1e_2d2d_3c3c);
    let mut found = 0;
    let mut not_found = 0;
    let mut short_buffer = 0;
    for n in 1..=12usize {
        for read_idx in 0..n as i32 {
            for _ in 0..400 {
                let flags: Vec<u32> = (0..n)
                    .map(|_| FLAG_CHOICES[rng.below(6) as usize])
                    .collect();
                // up_to_index deliberately runs past the end of the buffer as
                // well as inside it: C answers "none pending" when peek
                // returns NULL, which is a different route to -1 than
                // "walked the whole range and matched nothing".
                let up_to_index = rng.below(n as u32 + 3);
                let want = ref_is_forced_keyframe_pending(
                    &flags.iter().map(|&f| f as i32).collect::<Vec<_>>(),
                    read_idx,
                    up_to_index,
                );
                let got =
                    forced_keyframe_pending(&flags, up_to_index as usize).map_or(-1, |i| i as i32);
                assert_eq!(
                    got, want,
                    "n={n} read_idx={read_idx} up_to={up_to_index} flags={flags:08x?}"
                );
                if want >= 0 {
                    found += 1;
                } else {
                    not_found += 1;
                }
                if up_to_index as usize >= n {
                    short_buffer += 1;
                }
            }
        }
    }
    // All three outcomes must be reached, or the sweep is only testing one.
    assert!(found > 1000, "a forced key frame was found {found} times");
    assert!(
        not_found > 1000,
        "the none-pending answer came back {not_found} times"
    );
    assert!(
        short_buffer > 1000,
        "the walk ran past the end of the buffer {short_buffer} times"
    );
}

#[test]
fn forced_keyframe_pending_rejects_the_bit_test_reading() {
    // The asymmetry that pins C's `==` against a `&`: an entry with FORCE_KF
    // plus another flag. If C used a bit test, index 0 would match.
    let flags = [AOM_EFLAG_FORCE_KF | (1 << 28), AOM_EFLAG_FORCE_KF];
    let want = ref_is_forced_keyframe_pending(&[flags[0] as i32, flags[1] as i32], 0, 1);
    assert_eq!(want, 1, "C must skip the entry that carries an extra flag");
    assert_eq!(
        forced_keyframe_pending(&flags, 1).map_or(-1, |i| i as i32),
        want
    );
}

#[test]
fn new_framerate_clamp_matches_c() {
    // Dense either side of the 0.1 threshold, plus the ordinary range.
    for milli in 0..400 {
        let f = f64::from(milli) / 1000.0;
        assert_eq!(
            new_framerate(f).to_bits(),
            ref_new_framerate(f).to_bits(),
            "framerate={f}"
        );
    }
    for quarter in 1..1200 {
        let f = f64::from(quarter) / 4.0;
        assert_eq!(
            new_framerate(f).to_bits(),
            ref_new_framerate(f).to_bits(),
            "framerate={f}"
        );
    }
    // The exact boundary in both directions.
    assert_eq!(
        new_framerate(0.1).to_bits(),
        ref_new_framerate(0.1).to_bits()
    );
    let just_under = 0.1f64 - f64::EPSILON;
    assert_eq!(
        new_framerate(just_under).to_bits(),
        ref_new_framerate(just_under).to_bits()
    );
}
