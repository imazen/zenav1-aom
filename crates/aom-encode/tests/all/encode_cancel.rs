//! The ENCODER's cooperative cancellation, which until now did not exist.
//!
//! The decoder has had a stop token since the zen hardening work; the encoder
//! had none, and that is not a theoretical gap — with screen-content tools on,
//! the IntraBC DV search runs ~80 s on a single 1080p screenshot at
//! `--cpu-used 6` against ~1 s for the oracle (CLAUDE.md KB-41's perf note),
//! and a `--cpu-used 4` cell has been observed not finishing in 40 minutes. A
//! caller had no way to say "stop".
//!
//! Three properties, and the third is the one that makes the first two mean
//! something:
//!
//! 1. a token that never fires produces the **byte-identical** stream
//!    `encode_key_frame` produces — the poll is inert;
//! 2. a token that fires is **observed**, and the encode returns
//!    `KeyFrameError::Cancelled` rather than a stream;
//! 3. it is observed **partway through**, at a bounded cadence — not merely at
//!    the end, which a `check()` in the return path would also satisfy while
//!    saving a caller nothing.

use aom_encode::key_frame::{
    EncodeConfig, KeyFrameConfig, KeyFrameError, KeyFramePlanes, encode_key_frame,
    encode_key_frame_with,
};
use enough::{Stop, StopReason};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Never stops, but counts how many times it was asked. The count IS the
/// cadence measurement: a poll that only happened once is not cancellation.
struct CountingStop(AtomicUsize);
impl Stop for CountingStop {
    fn check(&self) -> Result<(), StopReason> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Stops after `n` polls, so the cancellation lands at a known point in the
/// walk rather than at whichever instant a wall-clock racer happens to fire.
struct StopAfter(AtomicUsize);
impl Stop for StopAfter {
    fn check(&self) -> Result<(), StopReason> {
        if self.0.fetch_sub(1, Ordering::Relaxed) == 0 {
            return Err(StopReason::Cancelled);
        }
        Ok(())
    }
}

/// Textured, non-flat source: a flat frame codes to a handful of skip blocks
/// and would leave the search with almost nothing to interrupt.
fn planes(cfg: &KeyFrameConfig) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (w, h) = (cfg.width, cfg.height);
    let (cw, ch) = cfg.chroma_dims();
    let mk = |pw: usize, ph: usize, seed: u32| {
        let mut v = Vec::with_capacity(pw * ph);
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        for y in 0..ph {
            for x in 0..pw {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v.push((((x * 7 + y * 5) as u32 & 0xff) + ((s >> 26) & 0x3f)) as u16 & 0xff);
            }
        }
        v
    };
    (mk(w, h, 1), mk(cw, ch, 2), mk(cw, ch, 3))
}

fn cell() -> KeyFrameConfig {
    // 384x256 = 6x4 superblocks at SB64, so the search walks 4 superblock ROWS
    // and a per-row poll has something to be a cadence OVER.
    let mut c = KeyFrameConfig::allintra_speed0(384, 256, 8, false, 1, 1, 32);
    c.cpu_used = 6;
    c
}

/// (1) INERT: a never-firing token must not move a single byte, and (3) the
/// poll must happen more than once per encode.
#[test]
fn a_never_firing_token_is_byte_inert_and_polled_per_superblock_row() {
    let cfg = cell();
    let (y, u, v) = planes(&cfg);
    let p = || KeyFramePlanes {
        y: &y,
        u: &u,
        v: &v,
    };

    let baseline = encode_key_frame(p(), &cfg).expect("baseline encode");
    let counter = CountingStop(AtomicUsize::new(0));
    let with_token = encode_key_frame_with(p(), &cfg, &EncodeConfig::new().with_stop(&counter))
        .expect("a never-firing token cannot cancel");

    assert_eq!(
        with_token, baseline,
        "a stop token that never fires must produce the byte-identical stream"
    );

    // 6x4 superblocks: 4 search rows + 4 repack rows + 1 per-tile repack poll.
    // The exact number is an implementation detail; that it SCALES with the
    // frame rather than being 1 is the property.
    let polls = counter.0.load(Ordering::Relaxed);
    assert!(
        polls >= 4,
        "the token must be polled at least once per superblock row (4 rows \
         here), got {polls} — a single poll is not a cancellation cadence"
    );
    println!("384x256 cq32 s6: {polls} polls, {} B (inert)", baseline.len());
}

/// (2) OBSERVED, and (3) observed PARTWAY THROUGH. Cancelling at the very first
/// poll and at a later one must both return `Cancelled` — and the encode must
/// stop, which is what distinguishes a real poll site from a `check()` bolted
/// onto the return path.
#[test]
fn a_firing_token_cancels_the_encode_at_a_bounded_point() {
    let cfg = cell();
    let (y, u, v) = planes(&cfg);
    let p = || KeyFramePlanes {
        y: &y,
        u: &u,
        v: &v,
    };

    // How many polls a full encode takes, so "partway" is a measured fraction
    // rather than a guess.
    let counter = CountingStop(AtomicUsize::new(0));
    encode_key_frame_with(p(), &cfg, &EncodeConfig::new().with_stop(&counter))
        .expect("uncancelled");
    let total = counter.0.load(Ordering::Relaxed);
    assert!(total >= 4, "need a multi-poll encode to cancel partway");

    for budget in [0usize, 1, total / 2] {
        let token = StopAfter(AtomicUsize::new(budget));
        let r = encode_key_frame_with(p(), &cfg, &EncodeConfig::new().with_stop(&token));
        match r {
            Err(KeyFrameError::Cancelled(_)) => {}
            Err(e) => panic!("budget {budget}: wrong error {e}"),
            Ok(bytes) => panic!(
                "budget {budget} of {total} polls: the encode ran to completion \
                 ({} B) — the token was not observed",
                bytes.len()
            ),
        }
    }
    println!("384x256 cq32 s6: cancelled at poll budgets 0, 1 and {} of {total}", total / 2);
}

/// The refusal must be distinguishable from every other refusal — a router that
/// cannot tell "you cancelled" from "unsupported configuration" will retry the
/// wrong thing.
#[test]
fn cancellation_is_its_own_error_and_does_not_collide_with_unsupported() {
    let cfg = cell();
    let (y, u, v) = planes(&cfg);
    let token = StopAfter(AtomicUsize::new(0));
    let cancelled = encode_key_frame_with(
        KeyFramePlanes {
            y: &y,
            u: &u,
            v: &v,
        },
        &cfg,
        &EncodeConfig::new().with_stop(&token),
    )
    .expect_err("must cancel");

    let mut bad = cfg;
    bad.cq_level = 64;
    let unsupported = encode_key_frame(
        KeyFramePlanes {
            y: &[],
            u: &[],
            v: &[],
        },
        &bad,
    )
    .expect_err("must refuse");

    assert_ne!(cancelled, unsupported);
    assert!(matches!(cancelled, KeyFrameError::Cancelled(_)));
    assert!(matches!(unsupported, KeyFrameError::Unsupported(_)));
    // And a cancellation must SAY so: a router surfacing this to a user should
    // not have to match on the variant to get a readable reason.
    assert!(
        cancelled.to_string().contains("cancelled"),
        "Display must name the cancellation: {cancelled}"
    );
}
