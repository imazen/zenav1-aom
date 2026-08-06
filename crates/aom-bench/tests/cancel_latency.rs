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
//!    (72.4 ms at 4096x4096 — over the bar on its own) and gated on its
//!    internal poll spacing.
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
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The user-set acceptance bar: a cancel must be honoured within this.
const BAR_MS: f64 = 20.0;

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
#[derive(Default)]
struct CancelFlag {
    fired: AtomicBool,
    polls: AtomicU64,
}

impl Stop for CancelFlag {
    fn check(&self) -> Result<(), StopReason> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        if self.fired.load(Ordering::Acquire) {
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

/// Spin (not `sleep`) to `deadline`. `thread::sleep` on Darwin overshoots by
/// milliseconds, which is the same order as the quantity being measured; a
/// coarse sleep up to 2 ms out, then a spin, keeps the placement error in the
/// microseconds without burning a core for the whole wait.
fn spin_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let left = deadline - now;
        if left > Duration::from_millis(2) {
            std::thread::sleep(left - Duration::from_millis(2));
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
        out = Some(Sample {
            frac: 0.0,
            latency: t_ret.checked_duration_since(t_cancel),
            cancelled,
            polls: token.polls.load(Ordering::Relaxed),
        });
    });
    out.expect("scope ran")
}

#[test]
fn cancel_latency_by_size() {
    println!("\n=== decode cancellation latency: cancel() -> decode returns ===");
    println!(
        "bar = {BAR_MS:.0} ms; {REPS} reps x {} cancel points",
        FRACTIONS.len()
    );
    let mut tsv = String::from(
        "size\tw\th\tstream_bytes\tnatural_ms\tfrac\trep\tcancel_at_ms\tlatency_ms\toutcome\tpolls\n",
    );
    let mut summary: Vec<String> = Vec::new();
    let mut over_bar: Vec<String> = Vec::new();

    for &(label, w, h) in SIZES {
        let stream = stream_for(w, h);
        let natural = natural_duration(&stream, 5);
        let mut lat: Vec<f64> = Vec::new();
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
                tsv.push_str(&format!(
                    "{label}\t{w}\t{h}\t{}\t{:.4}\t{f}\t{rep}\t{:.4}\t{lms}\t{outcome}\t{}\n",
                    stream.len(),
                    ms(natural),
                    ms(at),
                    s.polls
                ));
            }
        }
        lat.sort_by(|a, b| a.partial_cmp(b).expect("no NaN latencies"));
        let (p50, p90, p99, max) = (
            pct(&lat, 0.50),
            pct(&lat, 0.90),
            pct(&lat, 0.99),
            *lat.last().unwrap_or(&f64::NAN),
        );
        let line = format!(
            "{label:<7} {w}x{h:<5} natural {:8.3} ms | n={:3} raced={raced} ran-to-completion={completed} \
             | p50 {p50:7.3} p90 {p90:7.3} p99 {p99:7.3} max {max:7.3} ms (worst at frac {:.3})",
            ms(natural),
            lat.len(),
            worst.1,
        );
        println!("{line}");
        summary.push(line);
        // The bar is on the WORST observed cancel, not on a percentile: a p99
        // formulation would license 1 % of cancels to hang. `p50/p90/p99` ride
        // along in the message so a failure separates "one descheduled sample"
        // (p99 fine, max out) from "systematically over" (p50 out).
        if max > BAR_MS {
            over_bar.push(format!(
                "{label} {w}x{h}: max {max:.3} ms > {BAR_MS:.0} ms bar \
                 (n={} p50 {p50:.3} p90 {p90:.3} p99 {p99:.3}, natural {:.3} ms)",
                lat.len(),
                ms(natural),
            ));
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
        over_bar.is_empty(),
        "cancellation latency exceeds the {BAR_MS:.0} ms bar:\n  {}",
        over_bar.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// Arm 2: where the polls actually are
// ---------------------------------------------------------------------------

#[test]
fn poll_gap_map() {
    println!("\n=== poll spacing (deterministic; the token never fires) ===");
    let mut tsv = String::from(
        "size\tw\th\ttotal_ms\tpolls\tgap_p50_ms\tgap_p90_ms\tgap_p99_ms\tgap_max_ms\tfirst_poll_ms\ttail_ms\n",
    );
    let mut rows: Vec<String> = Vec::new();
    for &(label, w, h) in SIZES {
        let stream = stream_for(w, h);
        // Warm the allocator / page cache so the traced run measures decode
        // work rather than first-touch faults.
        aom_decode::frame::decode_frame_obus(&stream).expect("warm decode");
        let tr = PollTrace {
            t0: Instant::now(),
            marks: Mutex::new(Vec::new()),
        };
        let cfg = DecodeConfig::new().with_stop(&tr);
        let t0 = Instant::now();
        let r = aom_decode::frame::decode_frame_obus_with(&stream, &cfg);
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
        let row = format!(
            "{label:<7} {w}x{h:<5} total {:8.3} ms | polls {:5} | gap p50 {:7.4} p90 {:7.4} p99 {:7.4} max {:7.4} \
             | first-poll {:7.3} tail {:7.3} ms",
            ms(total),
            marks.len(),
            pct(&sorted, 0.50),
            pct(&sorted, 0.90),
            pct(&sorted, 0.99),
            sorted.last().copied().unwrap_or(f64::NAN),
            ms(first),
            ms(tail),
        );
        println!("{row}");
        // The TRAILING gaps, in order. Once `run_post_filters` /
        // `finish_and_grain` poll at their stage boundaries these ARE the
        // per-stage costs, in pipeline order (deblock, CDEF, [superres], LR,
        // crop, [film grain]) — which is how a future session attributes an
        // over-bar gap to a stage without re-instrumenting the decoder.
        let show = gaps.len().min(8);
        let trailing: Vec<String> = gaps[gaps.len() - show..]
            .iter()
            .enumerate()
            .map(|(i, g)| format!("#{}:{g:.3}", gaps.len() - show + i))
            .collect();
        println!(
            "          last {show} gaps (ms), pipeline order: {}",
            trailing.join("  ")
        );
        rows.push(row);
        tsv.push_str(&format!(
            "{label}\t{w}\t{h}\t{:.4}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.4}\t{:.4}\n",
            ms(total),
            marks.len(),
            pct(&sorted, 0.50),
            pct(&sorted, 0.90),
            pct(&sorted, 0.99),
            sorted.last().copied().unwrap_or(f64::NAN),
            ms(first),
            ms(tail),
        ));
        // Non-vacuity: a decode that never polled would make arm 1 meaningless.
        assert!(
            !marks.is_empty(),
            "{label}: the decode polled the stop token zero times"
        );
        // The STRUCTURAL form of the bar, and the reason this arm exists:
        // a cancel issued at the worst possible instant waits for the next
        // poll, so the exposure is `max(worst inter-poll gap, tail)` — the
        // tail being the stretch after the LAST poll, in which a cancel is
        // never seen at all. Deterministic and thread-free, so unlike arm 1
        // it cannot be perturbed by the scheduler.
        let worst = sorted.last().copied().unwrap_or(f64::NAN).max(ms(tail));
        assert!(
            worst <= BAR_MS,
            "{label} {w}x{h}: worst un-pollable stretch {worst:.3} ms > {BAR_MS:.0} ms bar \
             (max inter-poll gap {:.3}, tail after last poll {:.3}, {} polls over {:.3} ms). \
             Trailing gaps, pipeline order: {}",
            sorted.last().copied().unwrap_or(f64::NAN),
            ms(tail),
            marks.len(),
            ms(total),
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
/// the decode's critical path. **MEASURED: 72.4 ms at 4096x4096** — 3.6x the
/// bar on its own, which is why `add_film_grain_stop` exists.
///
/// Reports the stage cost per size (so a future session sees the shape) and
/// gates on the same property arm 2 gates: the worst un-pollable stretch
/// WITHIN the stage, not the stage's total cost. A 72 ms stage is fine as long
/// as no 20 ms window of it is blind to a cancel.
#[test]
fn film_grain_stage_cost() {
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
