//! DECODE CANCELLATION LATENCY — how long after `cancel()` does the decode
//! call actually return?
//!
//! `DecodeConfig::with_stop` documents "polled at SB-row / tile / frame
//! boundaries", and `config.rs::stop_token_check_stop_plumbing` proves the
//! plumbing exists. Neither says how *long* a caller waits. The acceptance bar
//! is **20 ms** (user directive, 2026-08-06), and a bar that is asserted rather
//! than measured is not a bar — the latency is floored by the spacing between
//! polls, and an SB row's wall time scales with frame width, so the answer is
//! necessarily a function of image size.
//!
//! # What is measured
//!
//! Three arms, all over the same four sizes (the repo's sweep discipline:
//! tiny / small / medium / large, so the fixed per-call overhead and the
//! per-pixel slope are separable):
//!
//! 1. [`cancel_latency_by_size`] — the HEADLINE. A worker thread runs
//!    `decode_frame_obus_with`; the main thread spins to a deadline placed at
//!    a fraction of the decode's own measured natural duration, stores the
//!    cancel flag, and timestamps. The worker timestamps the instant the
//!    decode call returns. `latency = t_return - t_cancel`. The fraction is
//!    swept across the whole decode so early / middle / late cancels are all
//!    sampled, and p50/p90/p99 are reported per size over the pooled samples.
//!
//! 2. [`poll_gap_map`] — the LOCALIZER, and it is deterministic (no threads,
//!    no scheduler). A `Stop` that timestamps every `check()` and never
//!    cancels. That yields the exact inter-poll gap distribution plus the
//!    **tail**: the interval from the final poll to the decode returning, i.e.
//!    the region in which a cancel cannot be observed at all. Arm 1's worst
//!    cell must be explainable by arm 2's worst gap; if it is not, the
//!    measurement is wrong, not the decoder.
//!
//! 3. [`film_grain_stage_cost`] — the BLIND SPOT. Arms 1 and 2 can only see
//!    stages the reference encoder actually put in the stream, and it never
//!    emits film grain, so the grain pass is invisible to them while still
//!    being a whole-frame pass on the decode's critical path. Timed directly
//!    (73.8 ms at 4096x4096 — over the bar on its own) and gated on its
//!    internal poll spacing.
//!
//! # WHICH PROPERTY IS GATED, AND WHY THE OTHER IS ONLY REPORTED
//!
//! Read this before "fixing" a gate back. End-to-end cancel latency decomposes
//! into three additive pieces:
//!
//! ```text
//!   t_return - t_cancel  =  (A) cross-thread visibility + descheduling
//!                        +  (B) wait for the decoder's next poll
//!                        +  (C) unwind: that poll returning -> the call returning
//! ```
//!
//! * **(B) is the decoder's own property** and is gated HARD, deterministically,
//!   by [`poll_gap_map`]: `max(worst inter-poll gap, tail) <= BAR_MS`, measured
//!   from inside the decode with no threads involved. This is the gate that the
//!   fix on this branch moved — the whole post-filter pipeline used to poll
//!   nothing (118.9 ms of a 192 ms 4096x4096 decode after the final poll). A
//!   stage that stops polling shows up here on any machine, at any speed.
//! * **(C) is also the decoder's property** — "once I have noticed, how fast do
//!   I get out" — and is gated HARD in [`cancel_latency_by_size`]. Both of its
//!   timestamps are taken ON THE WORKER THREAD (the token records the instant
//!   its `check()` returned `Err`), so no cross-thread wakeup is inside it. It
//!   catches a decoder that sees the cancel and then finishes the stage anyway.
//! * **(A) is NOT the decoder's property.** It is thread wakeup and OS
//!   scheduling, and on a shared CI runner it is unbounded: the harness' own
//!   `spin_until` burns a core while the worker decodes, so on a 2-vCPU runner
//!   the worker can simply lose its slot between the cancel and the return.
//!   Measured on GitHub's runners: p50 3.4 ms, p90 5.6 ms, **p99 = max
//!   23.4 ms** — a median that is fine and a tail 7x it, on a box whose natural
//!   decode is only 2x slower than the reference host. That shape is scheduler
//!   noise, not a decoder regression, and gating the *maximum* end-to-end
//!   sample on it made the build fail for it.
//!
//! So end-to-end is measured and reported at every size, and the user's 20 ms
//! bar is still asserted on it — but at `p90`, the statistic that stayed stable
//! (5.6 ms, 3.5x under the bar) on the runner where the max blew out — plus a
//! machine-scaled tripwire on the max (see [`MAX_TRIPWIRE_FRACTION`]) that a
//! stage which stopped polling cannot pass on any box, however slow. Nothing
//! about promptness has been conceded: (B) is gated at the full 20 ms bar on
//! its worst observed value, and (C) at the bar on p90 plus the same tripwire
//! on its worst — which IS the flat 20 ms on every cell under 160 ms.
//!
//! # Why the streams are what they are
//!
//! Real `aomenc` KEY-frame bitstreams over [`winperf::synth_i420`]
//! photographic-spectrum content, encoded **with CDEF and loop restoration
//! ON**. That is the conservative choice for this question: every post-filter
//! stage runs after the last tile-decode poll, so enabling them maximises the
//! un-pollable tail. Encoding is untimed setup and is disk-cached under
//! `~/tmp` (regenerable; see [`stream_for`]).
//!
//! # What this does NOT establish
//!
//! * **Loop restoration is asked for but not delivered.** `enable_restoration`
//!   is on for every stream above, and every one of them still codes
//!   `frame_restoration_type = RESTORE_NONE` (visible in the poll-gap map: the
//!   post-tile region has three gaps, not four). The LR stage is made pollable
//!   on the same reasoning as deblock and CDEF, but its cost at 4096x4096 is
//!   UNMEASURED — nothing here would catch it if it were the next 88 ms stage.
//! * Nothing about the ENCODER, which has no stop token at all.
//! * Only bd8 4:2:0 single-tile KEY frames are timed. The multi-tile and inter
//!   paths poll at the same sites, but their latency is not measured here.
//! * Nothing about a machine other than the one in the committed `.meta`.

use aom_decode::DecodeConfig;
use enough::{Stop, StopReason};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The user-set acceptance bar: a cancel must be honoured within this.
const BAR_MS: f64 = 20.0;

/// The machine-scaled tripwire on the WORST end-to-end sample, as a fraction of
/// that machine's own measured natural decode time. The bound is
/// `max(BAR_MS, MAX_TRIPWIRE_FRACTION * natural_ms)`, so it is the 20 ms bar on
/// any host fast enough for 20 ms to be the looser number, and scales up on a
/// slower one instead of failing it.
///
/// 0.125 is placed between the two measured quantities it has to separate:
/// * the regression it must catch — a pipeline stage that polls nothing —
///   contributes an un-pollable stretch that is a LARGE fraction of the decode.
///   The one this branch removed was 118.9 ms of 192.1 ms = **62 %**, and its
///   individual stages were CDEF 87.9 ms (46 %) and deblock 26.9 ms (14 %)
///   (`benchmarks/decode_cancel_latency_2026-08-06.meta`). All three are over
///   12.5 %, at any image size, on any machine — the ratio is scale-free.
/// * the noise it must tolerate — the worst descheduling tail observed on a
///   shared GitHub runner was 23.4 ms against a 381.9 ms natural decode =
///   6.1 %, i.e. this bound sits ~2x above it.
///
/// The tripwire is asserted only on cells where the scaled term wins (natural
/// decode > `BAR_MS / MAX_TRIPWIRE_FRACTION` = 160 ms). Where it degenerates to
/// the flat bar it would be gating pure scheduling — see the comment at its use
/// site, and [`poll_gap_map`], which gates those cells at the same 20 ms with a
/// deterministic instrument.
const MAX_TRIPWIRE_FRACTION: f64 = 0.125;

/// Traced repeats in [`poll_gap_map`]. The gate takes the MINIMUM over runs of
/// the worst un-pollable stretch, which is the right noise rejection for this
/// quantity: a deschedule between two polls can only ever ADD to a gap, so the
/// smallest observed worst-gap is the closest estimate of the decoder's own
/// spacing. A real regression (a stage that stopped polling) inflates every
/// run, so it survives the minimum; a one-off scheduler stall does not.
const TRACE_RUNS: usize = 3;

/// Serialises the three timing arms against each other. `cargo test` runs the
/// tests in one binary CONCURRENTLY by default, so without this a 4096x4096
/// film-grain pass (arm 3) and a traced 4096x4096 decode (arm 2) execute while
/// arm 1 is timing a thread wakeup — self-inflicted contention that on a 2-vCPU
/// runner is the same order as the quantity being measured. The committed
/// record was taken with `--test-threads 1`; this makes the default run match
/// the methodology it is compared against.
///
/// Poison is deliberately ignored: it only means an EARLIER arm's gate failed,
/// and the mutex guards nothing but exclusive access to the CPU. Propagating it
/// would replace the later arms' real verdicts with `PoisonError`, hiding
/// exactly the localisation this file exists to provide — verified: with
/// `cdef_frame_generic`'s per-fb-row poll removed, `expect()` here reported
/// `poison` for `poll_gap_map` instead of the 89.3 ms un-pollable stretch it
/// had actually measured.
fn timing_serial() -> std::sync::MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// `max(BAR_MS, MAX_TRIPWIRE_FRACTION * natural_ms)` — see
/// [`MAX_TRIPWIRE_FRACTION`].
fn tripwire_ms(natural_ms: f64) -> f64 {
    BAR_MS.max(MAX_TRIPWIRE_FRACTION * natural_ms)
}

/// `(label, width, height)`. The four-bucket size sweep — tiny catches the
/// fixed per-call cost, large catches the per-pixel slope that sets the poll
/// spacing.
const SIZES: &[(&str, usize, usize)] = &[
    ("tiny", 64, 64),
    ("small", 256, 256),
    ("medium", 1024, 1024),
    ("large", 4096, 4096),
];

/// `--cq-level` and `--cpu-used` for the reference encodes. 44/6 is the cell
/// the whole `winperf` record is written against, so the coded content is
/// comparable with the existing perf numbers.
const CQ: i32 = 44;
const SPEED: i32 = 6;

/// Cancel points as a fraction of the decode's own natural duration. Spread
/// deliberately unevenly: dense at both ends, because the first poll and the
/// post-last-poll tail are the two places a coarse gap can hide.
const FRACTIONS: &[f64] = &[
    0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995,
];

/// Repeats per (size, fraction) cell. 15 fractions x 7 = 105 samples per size,
/// enough for a p99 that is a measured order statistic rather than an
/// extrapolation.
const REPS: usize = 7;

// ---------------------------------------------------------------------------
// Stop-token implementations
// ---------------------------------------------------------------------------

/// A real cancel token: an atomic flag plus a poll counter, so a cell can
/// report how many polls the decode had actually reached when it was cancelled.
///
/// `observed` is the instant the FIRST `check()` refused, timestamped inside
/// the decode on the worker thread. It splits the end-to-end latency into the
/// part that waits for the decoder (`t_observed - t_cancel`, which is the poll
/// spacing plus whatever the scheduler added) and the part the decoder owns
/// outright (`t_return - t_observed`, the unwind) — see the module header.
#[derive(Default)]
struct CancelFlag {
    fired: AtomicBool,
    polls: AtomicU64,
    observed: OnceLock<Instant>,
}

impl Stop for CancelFlag {
    fn check(&self) -> Result<(), StopReason> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        if self.fired.load(Ordering::Acquire) {
            // Timestamp BEFORE returning, and only for the first refusal:
            // `set` on an already-set `OnceLock` is a no-op, so a decoder that
            // polls again on its way out cannot overwrite the instant it first
            // learned of the cancel.
            let _ = self.observed.set(Instant::now());
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// A `Stop` that never stops but timestamps every poll, relative to `t0`.
/// Single-threaded use; the `Mutex` exists only because `Stop: Send + Sync`
/// and `check(&self)` takes a shared reference.
struct PollTrace {
    t0: Instant,
    marks: Mutex<Vec<Duration>>,
}

impl Stop for PollTrace {
    fn check(&self) -> Result<(), StopReason> {
        // `unwrap` on a lock that is only ever taken here, on one thread, and
        // never held across a panic: poisoning is unreachable.
        self.marks.lock().unwrap().push(self.t0.elapsed());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

/// A real `aomenc --allintra` KEY-frame stream for `(w, h)` with CDEF **and**
/// loop restoration enabled, disk-cached under `~/tmp/zenav1-aom-cancel-cache`.
///
/// The cache is scratch: deleting it only costs the encode time. It exists
/// because a 4096x4096 single-threaded libaom encode is minutes of untimed
/// setup that would otherwise be paid on every run of this file.
fn stream_for(w: usize, h: usize) -> Vec<u8> {
    let dir = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join("tmp/zenav1-aom-cancel-cache");
    let path = dir.join(format!("photo_{w}x{h}_cq{CQ}_s{SPEED}_cdef1_lr1.obu"));
    if let Ok(bytes) = std::fs::read(&path) {
        if !bytes.is_empty() {
            return bytes;
        }
    }
    aom_sys_ref::ref_init();
    let cell = aom_bench::winperf::cell(w, h, CQ, SPEED, aom_bench::winperf::Content::Photo);
    let stream = aom_sys_ref::ref_encode_av1_kf(
        &cell.y,
        &cell.u,
        &cell.v,
        cell.w,
        cell.h,
        i32::from(cell.bd),
        cell.mono,
        cell.ss_x as i32,
        cell.ss_y as i32,
        cell.cq_level,
        cell.speed,
        true, // enable_cdef      — maximise the un-pollable post-filter tail
        true, // enable_restoration
        cell.usage,
        0,
        false,
    );
    assert!(!stream.is_empty(), "{w}x{h}: C encode produced no bytes");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(&path, &stream);
    stream
}

/// Median natural (uncancelled) decode duration, from `n` runs. The first run
/// is discarded as warm-up (first-touch page faults on the recon buffers).
fn natural_duration(stream: &[u8], n: usize) -> Duration {
    let mut v = Vec::with_capacity(n);
    for i in 0..=n {
        let t = Instant::now();
        let r = aom_decode::frame::decode_frame_obus(stream);
        let d = t.elapsed();
        r.unwrap_or_else(|e| panic!("uncancelled decode failed: {e}"));
        if i > 0 {
            v.push(d);
        }
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// How far out [`spin_until`] switches from `sleep` to a spin, and therefore
/// the scale of the deadline placement error under load: a `sleep` that
/// overshoots cannot be corrected, so a cancel can land up to about this late.
/// On a decode whose whole duration is not much more than this, EVERY cancel
/// point can legitimately land after the return — which is why the
/// "this cell produced samples" assertion is only made where the decode is
/// comfortably longer (see [`cancel_latency_by_size`]).
const SPIN_GUARD_MS: u64 = 2;

/// Spin (not `sleep`) to `deadline`. `thread::sleep` on Darwin overshoots by
/// milliseconds, which is the same order as the quantity being measured; a
/// coarse sleep up to [`SPIN_GUARD_MS`] out, then a spin, keeps the placement
/// error in the microseconds without burning a core for the whole wait.
fn spin_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let left = deadline - now;
        if left > Duration::from_millis(SPIN_GUARD_MS) {
            std::thread::sleep(left - Duration::from_millis(SPIN_GUARD_MS));
        } else {
            std::hint::spin_loop();
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics (measured order statistics — never an interpolation or a fit)
// ---------------------------------------------------------------------------

/// The `q`-quantile of `sorted` by nearest-rank, so every reported number is
/// an actual observed sample. Returns `f64::NAN` for an empty population
/// rather than inventing one.
fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

// ---------------------------------------------------------------------------
// Arm 1: real cancel latency
// ---------------------------------------------------------------------------

/// One cancel sample.
struct Sample {
    frac: f64,
    /// `t_return - t_cancel`, or `None` when the decode had already returned
    /// before the cancel could be issued (a raced cell — reported, not
    /// silently dropped, and never counted as a latency).
    latency: Option<Duration>,
    /// `t_observed - t_cancel`: flag set -> the decoder's next poll refuses.
    /// Poll spacing PLUS any descheduling of the worker. Reported only; the
    /// spacing half of it is what [`poll_gap_map`] gates deterministically.
    wait: Option<Duration>,
    /// `t_return - t_observed`: both instants taken on the worker thread, so
    /// this is the decoder's own unwind path with no cross-thread wakeup in
    /// it. GATED. `None` when the token never refused (raced / ran to
    /// completion).
    unwind: Option<Duration>,
    /// Did the decode return `DecodeError::Cancelled` (vs run to completion)?
    cancelled: bool,
    /// Polls the token saw over the whole call.
    polls: u64,
}

/// Run one cancel cell: decode on a worker, cancel at `at` after the decode
/// call starts.
fn cancel_once(stream: &[u8], at: Duration) -> Sample {
    let token = CancelFlag::default();
    let (tx, rx) = std::sync::mpsc::channel::<Instant>();
    let mut out = None;
    std::thread::scope(|s| {
        let worker = s.spawn(|| {
            let cfg = DecodeConfig::new().with_stop(&token);
            let t0 = Instant::now();
            tx.send(t0).expect("start-instant receiver alive");
            let r = aom_decode::frame::decode_frame_obus_with(stream, &cfg);
            let t_ret = Instant::now();
            let cancelled = matches!(r, Err(aom_decode::DecodeError::Cancelled(_)));
            if let Err(ref e) = r {
                assert!(
                    cancelled,
                    "decode failed for a reason other than cancellation: {e}"
                );
            }
            (t0, t_ret, cancelled)
        });
        let t0 = rx.recv().expect("worker sent its start instant");
        spin_until(t0 + at);
        token.fired.store(true, Ordering::Release);
        let t_cancel = Instant::now();
        let (_, t_ret, cancelled) = worker.join().expect("worker thread did not panic");
        let observed = token.observed.get().copied();
        out = Some(Sample {
            frac: 0.0,
            latency: t_ret.checked_duration_since(t_cancel),
            // `t_cancel` is read just AFTER the store, so a poll can legally
            // observe the flag a few ns before it; `checked_` yields None
            // there rather than a bogus number.
            wait: observed.and_then(|o| o.checked_duration_since(t_cancel)),
            unwind: observed.and_then(|o| t_ret.checked_duration_since(o)),
            cancelled,
            polls: token.polls.load(Ordering::Relaxed),
        });
    });
    out.expect("scope ran")
}

#[test]
fn cancel_latency_by_size() {
    let _serial = timing_serial();
    println!("\n=== decode cancellation latency: cancel() -> decode returns ===");
    println!(
        "bar = {BAR_MS:.0} ms; {REPS} reps x {} cancel points",
        FRACTIONS.len()
    );
    println!(
        "GATED: p90(end-to-end) <= bar; max(end-to-end) <= {:.1} % of natural (cells over \
         {:.0} ms only); unwind (worker-thread-only, decoder-owned) at bar + tripwire. \
         REPORTED: max(end-to-end) against the flat bar, and the wait split — \
         see the module header for why.",
        MAX_TRIPWIRE_FRACTION * 100.0,
        BAR_MS / MAX_TRIPWIRE_FRACTION,
    );
    let mut tsv = String::from(
        "size\tw\th\tstream_bytes\tnatural_ms\tfrac\trep\tcancel_at_ms\tlatency_ms\twait_ms\tunwind_ms\toutcome\tpolls\n",
    );
    let mut summary: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for &(label, w, h) in SIZES {
        let stream = stream_for(w, h);
        let natural = natural_duration(&stream, 5);
        let mut lat: Vec<f64> = Vec::new();
        let mut unw: Vec<f64> = Vec::new();
        let mut raced = 0usize;
        let mut completed = 0usize;
        let mut worst = (0.0f64, 0.0f64); // (latency_ms, frac)

        for &f in FRACTIONS {
            let at = natural.mul_f64(f);
            for rep in 0..REPS {
                let mut s = cancel_once(&stream, at);
                s.frac = f;
                let (lms, outcome) = match s.latency {
                    Some(d) => {
                        let v = ms(d);
                        lat.push(v);
                        if v > worst.0 {
                            worst = (v, f);
                        }
                        if !s.cancelled {
                            completed += 1;
                        }
                        (
                            format!("{v:.4}"),
                            if s.cancelled {
                                "cancelled"
                            } else {
                                "completed"
                            },
                        )
                    }
                    None => {
                        raced += 1;
                        ("na".to_string(), "raced-after-return")
                    }
                };
                if let Some(u) = s.unwind {
                    unw.push(ms(u));
                }
                let fmt = |d: Option<Duration>| match d {
                    Some(d) => format!("{:.4}", ms(d)),
                    None => "na".to_string(),
                };
                tsv.push_str(&format!(
                    "{label}\t{w}\t{h}\t{}\t{:.4}\t{f}\t{rep}\t{:.4}\t{lms}\t{}\t{}\t{outcome}\t{}\n",
                    stream.len(),
                    ms(natural),
                    ms(at),
                    fmt(s.wait),
                    fmt(s.unwind),
                    s.polls
                ));
            }
        }
        lat.sort_by(|a, b| a.partial_cmp(b).expect("no NaN latencies"));
        unw.sort_by(|a, b| a.partial_cmp(b).expect("no NaN unwinds"));
        let (p50, p90, p99, max) = (
            pct(&lat, 0.50),
            pct(&lat, 0.90),
            pct(&lat, 0.99),
            *lat.last().unwrap_or(&f64::NAN),
        );
        let (u90, umax) = (pct(&unw, 0.90), *unw.last().unwrap_or(&f64::NAN));
        let trip = tripwire_ms(ms(natural));
        let line = format!(
            "{label:<7} {w}x{h:<5} natural {:8.3} ms | n={:3} raced={raced} ran-to-completion={completed} \
             | p50 {p50:7.3} p90 {p90:7.3} p99 {p99:7.3} max {max:7.3} ms (worst at frac {:.3}) \
             | unwind n={:3} p90 {u90:7.4} max {umax:7.4} ms | max-tripwire {}{}",
            ms(natural),
            lat.len(),
            worst.1,
            unw.len(),
            if trip > BAR_MS {
                format!("{trip:.3} ms")
            } else {
                "n/a (decode too short for a scaled bound; poll_gap_map gates this cell)"
                    .to_string()
            },
            if max > BAR_MS {
                "  [max over the flat bar — reported, see p90 + tripwire]"
            } else {
                ""
            },
        );
        println!("{line}");
        summary.push(line);

        // Non-vacuity: a cell with no samples cannot fail any of the gates
        // below, so it must fail here instead — but only where a full race is
        // NOT a legitimate outcome. Under load a cancel can land up to
        // `SPIN_GUARD_MS` late (see [`spin_until`]), so on a decode that short
        // every cancel point can miss, and the reference host already records
        // 55 of 105 raced at 64x64. Above 5x the guard the deadline placement
        // error cannot explain a total race, so an empty cell there is a defect
        // in the harness or a decode that vanished, not scheduling.
        assert!(
            !lat.is_empty() || ms(natural) < 5.0 * SPIN_GUARD_MS as f64,
            "{label} {w}x{h}: every one of the {} cancel attempts raced the return on a \
             {:.3} ms decode — zero latency samples, so this cell gated nothing",
            FRACTIONS.len() * REPS,
            ms(natural),
        );
        if lat.is_empty() {
            println!(
                "          (all {} cancels raced the return on a {:.3} ms decode; \
                 poll_gap_map still gates this size)",
                FRACTIONS.len() * REPS,
                ms(natural),
            );
            continue;
        }

        // GATE 1 — the user's 20 ms bar on end-to-end latency, at the
        // percentile that survives a shared runner. The MAXIMUM is not gated
        // here on purpose: it is (A) + (B) + (C) and (A) is unbounded on a
        // 2-vCPU box (measured: p90 5.6 ms but max 23.4 ms on the same 77
        // samples). p90 is not a licence for 10 % of cancels to hang — a real
        // regression moves the whole distribution, because a stage that stops
        // polling swallows every cancel issued while it runs, which is a
        // double-digit percentage of the sweep's cancel points (the one this
        // branch fixed took p90 to 96.2 ms and p50 to 6.6 ms). What p90 drops
        // is exactly the handful of samples where the WORKER, not the decoder,
        // was descheduled.
        if p90 > BAR_MS {
            failures.push(format!(
                "{label} {w}x{h}: p90 end-to-end {p90:.3} ms > {BAR_MS:.0} ms bar \
                 (n={} p50 {p50:.3} p99 {p99:.3} max {max:.3}, natural {:.3} ms)",
                lat.len(),
                ms(natural),
            ));
        }
        // GATE 2 — the machine-scaled tripwire on the WORST sample. Catches a
        // stage that stopped polling even if it is narrow enough to leave p90
        // under the bar, and cannot be tripped by a slow box: the bound grows
        // with that box's own natural decode time.
        //
        // Applied only on cells where the bound IS the scaled one. On a cell
        // whose whole decode is shorter than `BAR_MS / MAX_TRIPWIRE_FRACTION`
        // (160 ms), the bound degenerates to the flat 20 ms bar — and there it
        // would be gating a quantity that is almost entirely (A): the decoder's
        // own contribution is sub-millisecond (measured: 0.31 ms max at
        // 1024x1024 against an 11.7 ms decode), so the only thing a flat 20 ms
        // max-gate can detect on those cells is a scheduler stall. Their
        // un-pollable stretch is gated at the SAME 20 ms bar by `poll_gap_map`,
        // deterministically and with no thread in the measurement — a strictly
        // better instrument for the same property. Detection is not lost here,
        // it is relocated to the arm that can do it without a false positive.
        if trip > BAR_MS && max > trip {
            failures.push(format!(
                "{label} {w}x{h}: max end-to-end {max:.3} ms > {trip:.3} ms \
                 (= max({BAR_MS:.0} ms bar, {:.1} % of this machine's {:.3} ms natural decode)). \
                 That is too large to be scheduler noise — look for a pipeline stage that \
                 stopped polling (n={} p50 {p50:.3} p90 {p90:.3} p99 {p99:.3})",
                MAX_TRIPWIRE_FRACTION * 100.0,
                ms(natural),
                lat.len(),
            ));
        }
        // GATE 3 — the unwind: from the poll that refused to the call
        // returning, BOTH timestamps taken on the worker thread. No wakeup and
        // no cross-thread visibility inside it, so unlike gate 1 this one is
        // gated at the bar on its worst observed value as well as at p90.
        if !unw.is_empty() {
            if u90 > BAR_MS || umax > trip {
                failures.push(format!(
                    "{label} {w}x{h}: the decoder took too long to RETURN after its own poll \
                     refused — unwind p90 {u90:.4} ms (bar {BAR_MS:.0}), max {umax:.4} ms \
                     (tripwire {trip:.3}). This is decoder-owned work, not scheduling: \
                     something finishes a stage after seeing the cancel"
                ));
            }
        } else {
            // Every cancel that landed before the return either ran to
            // completion or raced. Then gate 3 measured nothing, and the cell
            // is not entitled to pass silently.
            assert!(
                completed == lat.len(),
                "{label} {w}x{h}: {} cancels returned Cancelled but the token never recorded \
                 refusing — the harness lost the observation instant",
                lat.len() - completed
            );
        }
    }

    if let Ok(p) = std::env::var("AOM_CANCEL_TSV") {
        std::fs::write(&p, &tsv).unwrap_or_else(|e| panic!("write {p}: {e}"));
        println!("wrote {p}");
    }

    println!("\n--- summary ---");
    for l in &summary {
        println!("{l}");
    }
    assert!(
        failures.is_empty(),
        "cancellation latency gates failed:\n  {}",
        failures.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// Arm 2: where the polls actually are
// ---------------------------------------------------------------------------

/// One traced decode, reduced to the numbers the gate and the report need.
struct Trace {
    total: Duration,
    polls: usize,
    /// Inter-poll gaps in pipeline order, gap 0 being start -> first poll.
    gaps: Vec<f64>,
    /// The same, sorted, for the order statistics.
    sorted: Vec<f64>,
    first: Duration,
    tail: Duration,
    /// `max(worst gap, tail)` — the worst stretch in which a cancel would not
    /// be seen. THE gated quantity.
    worst: f64,
}

fn trace_once(stream: &[u8]) -> Trace {
    let tr = PollTrace {
        t0: Instant::now(),
        marks: Mutex::new(Vec::new()),
    };
    let cfg = DecodeConfig::new().with_stop(&tr);
    let t0 = Instant::now();
    let r = aom_decode::frame::decode_frame_obus_with(stream, &cfg);
    let total = t0.elapsed();
    r.unwrap_or_else(|e| panic!("traced decode failed: {e}"));
    let marks = tr.marks.into_inner().expect("trace lock");
    // Gaps between consecutive polls, plus the interval from the decode
    // call's start to the FIRST poll (header parse + allocation), which is
    // just as un-pollable as the tail.
    let mut gaps: Vec<f64> = Vec::with_capacity(marks.len());
    let mut prev = Duration::ZERO;
    for m in &marks {
        gaps.push(ms(m.saturating_sub(prev)));
        prev = *m;
    }
    let first = marks.first().copied().unwrap_or(total);
    let tail = total.saturating_sub(prev);
    let mut sorted = gaps.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN gaps"));
    let worst = sorted.last().copied().unwrap_or(f64::NAN).max(ms(tail));
    Trace {
        total,
        polls: marks.len(),
        gaps,
        sorted,
        first,
        tail,
        worst,
    }
}

/// THE HARD GATE on the property the decoder actually controls.
///
/// A cancel issued at the worst possible instant waits for the decoder's next
/// poll, so the decoder-side exposure is `max(worst inter-poll gap, tail)` —
/// the tail being the stretch after the LAST poll, in which a cancel is never
/// seen at all. Measured from inside the decode by a token that only
/// timestamps, with no worker thread and no cancel in flight, so unlike arm 1's
/// end-to-end number this one contains no thread wakeup and nothing the OS
/// scheduler can add on a shared runner — [`TRACE_RUNS`] repeats and the
/// minimum over them remove even the residual.
///
/// This is the assertion that the fix on this branch moved (118.9 ms of
/// un-pollable tail at 4096x4096 -> 0.0 ms), and the one that fails if any
/// pipeline stage stops polling again.
///
/// LIVENESS, verified 2026-08-07 (a gate that cannot fail is worse than no
/// gate): deleting the per-filter-block-row poll in `aom_dsp::cdef::frame`'s
/// `cdef_frame_generic` — one `s.check()?` — fails this assertion at
/// **89.275 ms**, per-run worst `[89.358 89.333 89.275]`, localised to trailing
/// gap `#164` with its neighbours still at 0.8 ms. The same break also fails
/// both of [`cancel_latency_by_size`]'s gates (p90 66.8 ms; max 91.6 ms against
/// a 24.9 ms tripwire). Restoring the poll returns all three to green.
#[test]
fn poll_gap_map() {
    let _serial = timing_serial();
    println!("\n=== poll spacing (deterministic; the token never fires) ===");
    println!(
        "GATED, at the {BAR_MS:.0} ms bar: min over {TRACE_RUNS} runs of \
         max(worst inter-poll gap, tail). This is the decoder-controlled half of \
         cancel latency."
    );
    let mut tsv = String::from(
        "size\tw\th\ttotal_ms\tpolls\tgap_p50_ms\tgap_p90_ms\tgap_p99_ms\tgap_max_ms\tfirst_poll_ms\ttail_ms\n",
    );
    for &(label, w, h) in SIZES {
        let stream = stream_for(w, h);
        // Warm the allocator / page cache so the traced runs measure decode
        // work rather than first-touch faults.
        aom_decode::frame::decode_frame_obus(&stream).expect("warm decode");
        let runs: Vec<Trace> = (0..TRACE_RUNS).map(|_| trace_once(&stream)).collect();
        // The minimum, not the median or the max: a deschedule between two
        // polls can only ADD to a gap, so the smallest observed worst-stretch
        // is the closest estimate of the decoder's own spacing. A stage that
        // stopped polling inflates EVERY run, so it survives the minimum.
        let best = runs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.worst.partial_cmp(&b.1.worst).expect("no NaN worst"))
            .expect("TRACE_RUNS > 0");
        let (best_i, t) = (best.0, best.1);
        let spread: Vec<String> = runs.iter().map(|r| format!("{:.3}", r.worst)).collect();
        println!(
            "{label:<7} {w}x{h:<5} total {:8.3} ms | polls {:5} | gap p50 {:7.4} p90 {:7.4} p99 {:7.4} max {:7.4} \
             | first-poll {:7.3} tail {:7.3} ms | worst per run [{}] -> gated on run {best_i} ({:.3})",
            ms(t.total),
            t.polls,
            pct(&t.sorted, 0.50),
            pct(&t.sorted, 0.90),
            pct(&t.sorted, 0.99),
            t.sorted.last().copied().unwrap_or(f64::NAN),
            ms(t.first),
            ms(t.tail),
            spread.join(" "),
            t.worst,
        );
        // Non-vacuity: a decode that never polled would make arm 1 meaningless.
        assert!(
            t.polls > 0,
            "{label}: the decode polled the stop token zero times"
        );
        // The TRAILING gaps, in order. Once `run_post_filters` /
        // `finish_and_grain` poll at their stage boundaries these ARE the
        // per-stage costs, in pipeline order (deblock, CDEF, [superres], LR,
        // crop, [film grain]) — which is how a future session attributes an
        // over-bar gap to a stage without re-instrumenting the decoder.
        let show = t.gaps.len().min(8);
        let trailing: Vec<String> = t.gaps[t.gaps.len() - show..]
            .iter()
            .enumerate()
            .map(|(i, g)| format!("#{}:{g:.3}", t.gaps.len() - show + i))
            .collect();
        println!(
            "          last {show} gaps (ms), pipeline order: {}",
            trailing.join("  ")
        );
        tsv.push_str(&format!(
            "{label}\t{w}\t{h}\t{:.4}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.4}\t{:.4}\n",
            ms(t.total),
            t.polls,
            pct(&t.sorted, 0.50),
            pct(&t.sorted, 0.90),
            pct(&t.sorted, 0.99),
            t.sorted.last().copied().unwrap_or(f64::NAN),
            ms(t.first),
            ms(t.tail),
        ));
        assert!(
            t.worst <= BAR_MS,
            "{label} {w}x{h}: worst un-pollable stretch {:.3} ms > {BAR_MS:.0} ms bar \
             (max inter-poll gap {:.3}, tail after last poll {:.3}, {} polls over {:.3} ms; \
             per-run worst over {TRACE_RUNS} runs [{}], so this is not scheduler noise). \
             Trailing gaps, pipeline order: {}",
            t.worst,
            t.sorted.last().copied().unwrap_or(f64::NAN),
            ms(t.tail),
            t.polls,
            ms(t.total),
            spread.join(" "),
            trailing.join("  "),
        );
    }
    if let Ok(p) = std::env::var("AOM_POLLGAP_TSV") {
        std::fs::write(&p, &tsv).unwrap_or_else(|e| panic!("write {p}: {e}"));
        println!("wrote {p}");
    }
}

// ---------------------------------------------------------------------------
// Arm 3: the two post-filter stages the sweep's streams never exercise
// ---------------------------------------------------------------------------

/// A fixed, spec-valid, chroma-bearing grain parameter set. Constants rather
/// than `rand_params_*` (as `film_grain_diff.rs` uses) because this arm is a
/// TIMING measurement: the cost must be reproducible run to run.
fn grain_params() -> aom_dsp::entropy::header::FilmGrainParams {
    let mut p = aom_dsp::entropy::header::FilmGrainParams {
        apply_grain: true,
        update_parameters: true,
        random_seed: 0x2f13,
        scaling_shift: 11,
        ar_coeff_lag: 3, // the widest AR template = the most expensive synthesis
        ar_coeff_shift: 8,
        grain_scale_shift: 0,
        overlap_flag: true, // overlap blending on: the expensive arm
        clip_to_restricted_range: false,
        num_y_points: 4,
        num_cb_points: 3,
        num_cr_points: 3,
        cb_mult: 128,
        cb_luma_mult: 192,
        cb_offset: 256,
        cr_mult: 128,
        cr_luma_mult: 192,
        cr_offset: 256,
        ..Default::default()
    };
    for (i, &(v, s)) in [(0, 40), (64, 60), (160, 70), (255, 90)].iter().enumerate() {
        p.scaling_points_y[i] = [v, s];
    }
    for (i, &(v, s)) in [(0, 30), (128, 50), (255, 62)].iter().enumerate() {
        p.scaling_points_cb[i] = [v, s];
        p.scaling_points_cr[i] = [v, s];
    }
    // 24 luma / 25 chroma AR coefficients at lag 3, small and mixed-sign.
    for (i, c) in p.ar_coeffs_y.iter_mut().enumerate() {
        *c = ((i as i32 % 7) - 3) * 2;
    }
    for (i, c) in p.ar_coeffs_cb.iter_mut().enumerate() {
        *c = ((i as i32 % 5) - 2) * 3;
    }
    for (i, c) in p.ar_coeffs_cr.iter_mut().enumerate() {
        *c = ((i as i32 % 5) - 2) * 3;
    }
    p
}

/// FILM GRAIN — the stage the sweep structurally cannot see, timed directly.
///
/// [`cancel_latency_by_size`] cannot answer this: the reference encoder puts no
/// grain in those streams, so `finish_and_grain`'s grain branch never runs and
/// its cost is absent from the poll-gap map. It is still a whole-frame pass on
/// the decode's critical path. **MEASURED: 73.8 ms at 4096x4096** — 3.7x the
/// bar on its own, which is why `add_film_grain_stop` exists.
///
/// Reports the stage cost per size (so a future session sees the shape) and
/// gates on the same property arm 2 gates: the worst un-pollable stretch
/// WITHIN the stage, not the stage's total cost. A 72 ms stage is fine as long
/// as no 20 ms window of it is blind to a cancel.
#[test]
fn film_grain_stage_cost() {
    let _serial = timing_serial();
    println!("\n=== film grain: whole-frame pass cost + its internal poll spacing ===");
    let p = grain_params();
    let mut over: Vec<String> = Vec::new();
    for &(label, w, h) in SIZES {
        let (cw, ch) = (w / 2, h / 2);
        // Content that actually exercises the scaling LUT across its range.
        let y: Vec<u16> = (0..w * h).map(|i| ((i * 37) % 256) as u16).collect();
        let u: Vec<u16> = (0..cw * ch).map(|i| ((i * 53) % 256) as u16).collect();
        let v: Vec<u16> = (0..cw * ch).map(|i| ((i * 91) % 256) as u16).collect();
        // Warm-up, then the median of 3 — same discipline as `natural_duration`.
        let mut runs = Vec::new();
        for i in 0..4 {
            let t = Instant::now();
            let (gy, _, _) =
                aom_decode::film_grain::add_film_grain(&p, 8, false, 1, 1, false, w, h, &y, &u, &v);
            let d = t.elapsed();
            // Non-vacuity: the pass must have changed pixels, else we timed a
            // no-op and the number means nothing.
            assert!(
                gy.len() == w * h && gy != y,
                "{label}: film grain left the luma plane unchanged"
            );
            if i > 0 {
                runs.push(d);
            }
        }
        runs.sort_unstable();
        let med = ms(runs[runs.len() / 2]);

        // Now the same pass through the stop-aware entry with a tracing token:
        // the gaps between its polls are the windows in which a cancel issued
        // during film grain would not be seen.
        let tr = PollTrace {
            t0: Instant::now(),
            marks: Mutex::new(Vec::new()),
        };
        let t0 = Instant::now();
        let r = aom_decode::film_grain::add_film_grain_stop(
            &p,
            8,
            false,
            1,
            1,
            false,
            w,
            h,
            &y,
            &u,
            &v,
            Some(&tr),
        );
        let total = t0.elapsed();
        let (gy, _, _) = r.expect("a never-firing token cannot cancel");
        assert_eq!(gy.len(), w * h, "{label}: stop-aware entry lost the plane");
        let marks = tr.marks.into_inner().expect("trace lock");
        assert!(
            !marks.is_empty(),
            "{label}: add_film_grain_stop polled zero times"
        );
        let mut prev = Duration::ZERO;
        let mut worst_gap = 0.0f64;
        for m in &marks {
            worst_gap = worst_gap.max(ms(m.saturating_sub(prev)));
            prev = *m;
        }
        let tail = ms(total.saturating_sub(prev));
        let worst = worst_gap.max(tail);
        let verdict = if worst > BAR_MS { "OVER BAR" } else { "under" };
        println!(
            "{label:<7} {w}x{h:<5} stage {med:8.3} ms | polls {:5} | worst un-pollable {worst:7.3} ms \
             (gap {worst_gap:.3}, tail {tail:.3})  [{verdict}]",
            marks.len(),
        );
        if worst > BAR_MS {
            over.push(format!(
                "{label} {w}x{h}: worst un-pollable stretch {worst:.3} ms > {BAR_MS:.0} ms \
                 (stage total {med:.3} ms, {} polls)",
                marks.len()
            ));
        }
    }
    assert!(
        over.is_empty(),
        "film grain has an un-interruptible window over the bar:\n  {}",
        over.join("\n  ")
    );
}
