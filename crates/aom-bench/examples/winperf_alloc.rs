//! `winperf_alloc` — the `eprof_alloc` allocation census, without the C oracle.
//!
//! `eprof_alloc` counts allocator calls for one `EncodeCell::port_encode` of
//! the study cell, but takes its bootstrap from a live `c_encode_defaults()`
//! and so cannot run where libaom is not built. This is the same counting
//! `GlobalAlloc`, over `aom_bench::winperf`'s generated source + committed
//! bootstrap, so the **call counts** — which are noise-free, unlike
//! milliseconds on a shared CI VM — can be taken on Windows and compared with
//! Darwin's directly.
//!
//! ```text
//! winperf_alloc
//! ```
//!
//! Counters are `Relaxed` atomics and the encode is single-threaded, so the
//! numbers are exact; the one atomic RMW per allocator call is why this is a
//! separate binary from `winperf` and why no timing is reported here.
//!
//! Output is `key<TAB>value` lines so a CI step can parse it without a JSON
//! dependency.

use aom_bench::winperf;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

const NBUCKET: usize = 32;

static N_ALLOC: AtomicUsize = AtomicUsize::new(0);
static N_ZEROED: AtomicUsize = AtomicUsize::new(0);
static N_REALLOC: AtomicUsize = AtomicUsize::new(0);
static N_DEALLOC: AtomicUsize = AtomicUsize::new(0);
static B_ALLOC: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static BUCKET: [AtomicUsize; NBUCKET] = [const { AtomicUsize::new(0) }; NBUCKET];

fn note(size: usize) {
    B_ALLOC.fetch_add(size, Relaxed);
    let b = (usize::BITS - size.leading_zeros()) as usize;
    BUCKET[b.min(NBUCKET - 1)].fetch_add(1, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

struct Counting;

// Every method forwards to `System` unchanged; the only added work is atomic
// bookkeeping, so the allocator contract is System's.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        N_ALLOC.fetch_add(1, Relaxed);
        note(l.size());
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        N_ZEROED.fetch_add(1, Relaxed);
        note(l.size());
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        N_REALLOC.fetch_add(1, Relaxed);
        LIVE.fetch_sub(l.size(), Relaxed);
        note(new);
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        N_DEALLOC.fetch_add(1, Relaxed);
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[derive(Clone, Copy)]
struct Snap {
    alloc: usize,
    zeroed: usize,
    realloc: usize,
    dealloc: usize,
    bytes: usize,
    peak: usize,
}

fn snap() -> Snap {
    Snap {
        alloc: N_ALLOC.load(Relaxed),
        zeroed: N_ZEROED.load(Relaxed),
        realloc: N_REALLOC.load(Relaxed),
        dealloc: N_DEALLOC.load(Relaxed),
        bytes: B_ALLOC.load(Relaxed),
        peak: PEAK.load(Relaxed),
    }
}

fn main() {
    let (w, h, q, s) = winperf::CELL;
    let cell = winperf::cell(w, h, q, s);
    let bootstrap = winperf::bootstrap();

    // Warm once so lazily-built statics/caches are not charged to the measured
    // encode — identical to `eprof_alloc`, so the counts are comparable.
    let _ = cell.port_encode(&bootstrap);
    let s0 = snap();
    let out = cell.port_encode(&bootstrap);
    let s1 = snap();

    let calls = (s1.alloc - s0.alloc) + (s1.zeroed - s0.zeroed) + (s1.realloc - s0.realloc);
    let sb = w.div_ceil(64) * h.div_ceil(64);
    println!("cell\t{w}x{h}_cq{q}_s{s}");
    println!("alloc\t{}", s1.alloc - s0.alloc);
    println!("alloc_zeroed\t{}", s1.zeroed - s0.zeroed);
    println!("realloc\t{}", s1.realloc - s0.realloc);
    println!("dealloc\t{}", s1.dealloc - s0.dealloc);
    println!("calls\t{calls}");
    println!("bytes\t{}", s1.bytes - s0.bytes);
    println!("peak_live\t{}", s1.peak.max(s0.peak));
    println!("per_superblock\t{:.1}", calls as f64 / sb as f64);
    println!("superblocks\t{sb}");
    println!("framebytes\t{}", out.len());
    for (i, c) in BUCKET.iter().enumerate() {
        let v = c.load(Relaxed);
        if v > 0 {
            println!("bucket_2p{i}\t{v}");
        }
    }
}
