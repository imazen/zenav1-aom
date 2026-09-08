//! The encoder's resource CONTRACT: caller-supplied caps, and a side-effect-free
//! estimate a router can decide on before committing.
//!
//! An estimate nobody checks is a number, not a contract, so this file measures
//! the real peak with a counting `GlobalAlloc` and pins the bound in BOTH
//! directions:
//!
//! * it must never be UNDER the measured peak — an under-estimate is worse than
//!   none, because a caller sizing a budget from it will OOM;
//! * it must never exceed the measured peak by more than
//!   [`ESTIMATE_MAX_SLACK`] — without a ceiling, "upper bound" is satisfiable by
//!   returning `u64::MAX`, and the cap would be useless.
//!
//! The grid is chosen for the SHAPES that break a naive model, not for
//! coverage: both aspect extremes (8320x64 and 64x8320, whose padded geometry
//! differs from `w*h` by 3x), the smallest frame, all four chroma formats, all
//! three bit depths, both superblock sizes, and the speed extremes.

use aom_encode::key_frame::{
    AllocMode, EncodeConfig, EncodeLimits, ESTIMATE_MAX_SLACK, KeyFrameConfig, KeyFrameError,
    KeyFramePlanes, encode_key_frame_with,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Counts live and peak heap. Relaxed atomics; the encode is single-threaded,
/// so the numbers are exact and the cost is one RMW per allocator call.
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(l.size(), Relaxed) + l.size(), Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(l.size(), Relaxed) + l.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(new, Relaxed) + new, Relaxed);
            LIVE.fetch_sub(l.size(), Relaxed);
        }
        q
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

/// Serialises the tests in this file. The counters above are PROCESS-global, so
/// two tests measuring at once measure each other — `cargo test` runs the tests
/// in one binary concurrently by default, and the first version of this file
/// failed exactly that way (it passed when run alone). Under `cargo nextest`
/// each test is its own process and the guard is a no-op, which is also
/// correct: separate processes have separate counters.
fn measuring() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn planes(cfg: &KeyFrameConfig) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (w, h) = (cfg.width, cfg.height);
    let (cw, ch) = cfg.chroma_dims();
    let mk = |pw: usize, ph: usize, s0: u32| {
        let mut s = s0.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..pw * ph)
            .map(|i| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((((i % pw) * 7 + (i / pw) * 5) as u32 & 0xff) + ((s >> 26) & 0x3f)) as u16 & 0xff
            })
            .collect()
    };
    let y = mk(w, h, 1);
    if cfg.monochrome {
        (y, Vec::new(), Vec::new())
    } else {
        (y, mk(cw, ch, 2), mk(cw, ch, 3))
    }
}

/// Peak heap ABOVE the caller's source planes, which are already live when the
/// encode starts (resetting the counter to zero would underflow on their free).
///
/// **`AllocMode::Infallible`, deliberately.** The default `Fallible` mode
/// pre-flights by reserving exactly `estimate()` bytes, and that reservation is
/// heap the counter sees — so measuring under it makes `peak >= est` true BY
/// CONSTRUCTION and the upper-bound assertion tautological. It was: the first
/// run after the mode landed reported every slack as exactly 1.00x and the 1x1
/// peak as 2,441,145 against its real 610,974. Measuring the encoder means
/// switching the probe off.
fn measure_peak(cfg: &KeyFrameConfig) -> (u64, usize) {
    let (y, u, v) = planes(cfg);
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let bytes = encode_key_frame_with(
        KeyFramePlanes {
            y: &y,
            u: &u,
            v: &v,
        },
        cfg,
        &EncodeConfig::new().with_alloc(AllocMode::Infallible),
    )
    .expect("encode");
    let peak = PEAK.load(Relaxed).saturating_sub(base) as u64;
    (peak, bytes.len())
}

fn cell(w: usize, h: usize, bd: u8, mono: bool, sx: usize, sy: usize, sb128: bool, speed: i32) -> KeyFrameConfig {
    let mut c = KeyFrameConfig::allintra_speed0(w, h, bd, mono, sx, sy, 32);
    c.cpu_used = speed;
    c.sb_size_128 = sb128;
    c
}

/// THE gate: the estimate bounds the measured peak, and is not absurdly loose.
#[test]
fn the_estimate_is_an_upper_bound_and_stays_honest() {
    let _guard = measuring();
    let cells = [
        ("1x1", cell(1, 1, 8, false, 1, 1, false, 6)),
        ("64x64", cell(64, 64, 8, false, 1, 1, false, 6)),
        ("100x60", cell(100, 60, 8, false, 1, 1, false, 6)),
        ("256x256", cell(256, 256, 8, false, 1, 1, false, 6)),
        ("256 mono", cell(256, 256, 8, true, 1, 1, false, 6)),
        ("256 444", cell(256, 256, 8, false, 0, 0, false, 6)),
        ("256 422", cell(256, 256, 8, false, 1, 0, false, 6)),
        ("256 bd10", cell(256, 256, 10, false, 1, 1, false, 6)),
        ("256 bd12", cell(256, 256, 12, false, 1, 1, false, 6)),
        ("256 sb128", cell(256, 256, 8, false, 1, 1, true, 6)),
        ("256 s0", cell(256, 256, 8, false, 1, 1, false, 0)),
        ("256 s9", cell(256, 256, 8, false, 1, 1, false, 9)),
        ("1024 s0", cell(1024, 1024, 8, false, 1, 1, false, 0)),
        ("1024 444", cell(1024, 1024, 8, false, 0, 0, false, 6)),
        // The aspect extremes: `w * h` is identical, the padded geometry is
        // not, and a per-pixel model is wrong by 3x on the second.
        ("8320x64", cell(8320, 64, 8, false, 1, 1, false, 9)),
        ("64x8320", cell(64, 8320, 8, false, 1, 1, false, 9)),
    ];

    let mut worst_slack = 0.0f64;
    let mut worst_label = "";
    let mut report = String::new();
    for (label, cfg) in &cells {
        let (peak, _len) = measure_peak(cfg);
        let est = cfg.estimate().peak_memory_bytes;
        assert!(
            est >= peak,
            "{label}: the estimate UNDER-states the measured peak ({est} < {peak}). \
             An under-estimate is worse than none — a caller sizing a budget \
             from it will OOM."
        );
        let slack = est as f64 / peak.max(1) as f64;
        if slack > worst_slack {
            worst_slack = slack;
            worst_label = label;
        }
        report.push_str(&format!(
            "  {label:<10} peak {peak:>10} est {est:>10}  slack {slack:.2}x\n"
        ));
    }
    println!("estimate vs measured peak:\n{report}");
    assert!(
        worst_slack <= ESTIMATE_MAX_SLACK,
        "the estimate is too loose at {worst_label} ({worst_slack:.2}x > \
         {ESTIMATE_MAX_SLACK:.1}x). An 'upper bound' that can be satisfied by \
         returning a huge number is useless to a caller deciding on it:\n{report}"
    );
}

/// A per-PIXEL model is what a reader would reach for first, and it is wrong.
/// Pin the reason so nobody simplifies the estimate back into one.
#[test]
fn the_padded_geometry_is_why_a_per_pixel_model_fails() {
    let _guard = measuring();
    let wide = cell(8320, 64, 8, false, 1, 1, false, 9);
    let tall = cell(64, 8320, 8, false, 1, 1, false, 9);
    assert_eq!(
        wide.width * wide.height,
        tall.width * tall.height,
        "the two cells must have identical pixel counts, else this proves nothing"
    );
    let (wp, tp) = (
        measure_peak(&wide).0 as f64 / (wide.width * wide.height) as f64,
        measure_peak(&tall).0 as f64 / (tall.width * tall.height) as f64,
    );
    assert!(
        tp > wp * 2.0,
        "the tall cell must cost far more per PIXEL than the wide one \
         ({tp:.1} vs {wp:.1} B/px) — that gap is the 320-sample stride floor \
         and the superblock alignment, and it is why the estimate is keyed on \
         `padded_plane_geometry` rather than `width * height`"
    );
    // ...and the estimate must track the padded geometry, not the pixels.
    assert!(
        tall.estimate().peak_memory_bytes > wide.estimate().peak_memory_bytes,
        "the estimate must be LARGER for the tall cell despite equal pixels"
    );
    println!("per-pixel peak: wide {wp:.1} B/px, tall {tp:.1} B/px");
}

/// Limits refuse BEFORE allocating, by name, and each cap is independently
/// load-bearing.
#[test]
fn every_limit_refuses_by_name_and_before_allocating() {
    let _guard = measuring();
    let cfg = cell(256, 256, 8, false, 1, 1, false, 6);
    let cases: [(&str, EncodeLimits); 4] = [
        ("pixels", EncodeLimits { max_pixels: Some(1000), ..EncodeLimits::new() }),
        ("width", EncodeLimits { max_width: Some(64), ..EncodeLimits::new() }),
        ("height", EncodeLimits { max_height: Some(64), ..EncodeLimits::new() }),
        (
            "memory_bytes",
            EncodeLimits { max_memory_bytes: Some(1024), ..EncodeLimits::new() },
        ),
    ];
    for (what, limits) in cases {
        // Empty planes: a limit must be refused before a single source sample
        // is read, so the plane-size check must not fire first.
        let e = encode_key_frame_with(
            KeyFramePlanes { y: &[], u: &[], v: &[] },
            &cfg,
            &EncodeConfig::new().with_limits(limits),
        )
        .expect_err("must refuse");
        match e {
            KeyFrameError::LimitExceeded { what: w, .. } => assert_eq!(w, what),
            other => panic!("{what}: wrong error {other}"),
        }
        assert!(e.to_string().contains(what), "Display must name the cap: {e}");
        // The query must agree — that is what makes it usable for routing.
        assert!(cfg.check_limits(&limits).is_err());
    }

    // ...and generous caps must NOT refuse, else the checks are vacuous.
    let ok = EncodeLimits {
        max_pixels: Some(1 << 30),
        max_width: Some(65536),
        max_height: Some(65536),
        max_memory_bytes: Some(1 << 40),
    };
    cfg.check_limits(&ok).expect("generous caps must pass");
    let (y, u, v) = planes(&cfg);
    encode_key_frame_with(
        KeyFramePlanes { y: &y, u: &u, v: &v },
        &cfg,
        &EncodeConfig::new().with_limits(ok),
    )
    .expect("generous caps must encode");
}

/// The allocation mode, in two halves so that neither depends on a property
/// this port does not control.
///
/// The half that MATTERS to a router is unconditional: `AllocFailed` is the one
/// variant `is_transient()` calls true, because it is the only one the caller
/// need change nothing to get past. A router that treated it like `Unsupported`
/// would permanently blacklist a backend for a transient memory shortage.
///
/// The half that exercises the pre-flight END TO END depends on the kernel's
/// overcommit policy: a 275 GB reservation is refused under Linux's default
/// heuristic (`vm.overcommit_memory=0`) and would be GRANTED under
/// `=1`. Asserting it unconditionally would make this test a report on
/// `/proc/sys/vm/overcommit_memory`. So it asserts the outcome it gets and says
/// which one it saw, and can therefore neither flake nor pass vacuously.
#[test]
fn the_allocation_mode_is_honoured_and_alloc_failure_is_the_transient_case() {
    let _guard = measuring();

    // --- unconditional: the retry contract ------------------------------
    let alloc_failed = KeyFrameError::AllocFailed { bytes: 1 << 40 };
    assert_eq!(alloc_failed.category(), "alloc-failed");
    assert!(
        alloc_failed.is_transient(),
        "AllocFailed is the one variant a caller need change nothing to get past"
    );
    let mut bad = cell(64, 64, 8, false, 1, 1, false, 6);
    bad.cq_level = 64;
    for other in [
        bad.validate_configuration().unwrap_err(),
        KeyFrameError::PlaneSize { plane: 0, expected: 1, got: 2 },
        KeyFrameError::SampleRange { plane: 0, max: 255, got: 4096 },
        KeyFrameError::LimitExceeded { what: "pixels", actual: 2, max: 1 },
    ] {
        assert!(
            !other.is_transient(),
            "only AllocFailed may be transient, but {other} claims to be"
        );
        assert_ne!(other.category(), "alloc-failed");
    }

    // --- the mode must never change a coded byte ------------------------
    let ok = cell(64, 64, 8, false, 1, 1, false, 6);
    let (y, u, v) = planes(&ok);
    let a = encode_key_frame_with(
        KeyFramePlanes { y: &y, u: &u, v: &v },
        &ok,
        &EncodeConfig::new().with_alloc(AllocMode::Fallible),
    )
    .expect("fallible mode must still encode");
    let b = encode_key_frame_with(
        KeyFramePlanes { y: &y, u: &u, v: &v },
        &ok,
        &EncodeConfig::new().with_alloc(AllocMode::Infallible),
    )
    .expect("infallible mode must still encode");
    assert_eq!(a, b, "the allocation mode must not change a single coded byte");

    // --- end to end, where the platform lets it be observed -------------
    // 65536x65536 is a SUPPORTED configuration (the dimension ceiling is
    // 65536), so this is a frame that simply will not fit — the case the mode
    // exists for, not a disguised refusal.
    let huge = cell(65536, 65536, 8, false, 1, 1, false, 6);
    huge.validate_configuration()
        .expect("65536x65536 must be a SUPPORTED configuration, else this tests the wrong thing");
    let est = huge.estimate().peak_memory_bytes;
    assert!(
        est > (1u64 << 36),
        "the cell must estimate beyond any plausible machine ({est} bytes)"
    );
    match encode_key_frame_with(
        KeyFramePlanes { y: &[], u: &[], v: &[] },
        &huge,
        &EncodeConfig::new(),
    ) {
        Err(KeyFrameError::AllocFailed { bytes }) => {
            assert!(bytes > 0);
            println!("pre-flight refused {est} bytes ({bytes} reserved) — mode observed end to end");
        }
        // The allocator GRANTED a 275 GB reservation, which means this kernel
        // overcommits. The pre-flight then correctly does not fire, and the
        // call falls through to the plane-size check on the empty planes.
        Err(KeyFrameError::PlaneSize { .. }) => {
            println!(
                "this kernel granted a {est}-byte reservation (overcommit); the \
                 pre-flight cannot be observed here, and the retry contract above \
                 is asserted unconditionally"
            );
        }
        other => panic!("expected AllocFailed or PlaneSize, got {other:?}"),
    }
}
