//! `eprof_alloc` — allocation census for ONE encode of the xbench profile cell.
//!
//! Answers the "cheap structural suspect" question the encoder hotspot profile
//! asks first: is the port doing per-block / per-superblock heap traffic that
//! the C encoder does out of a pre-allocated workspace? A sampling profile can
//! only show the *time* spent in `malloc`; this counts the **calls**, which is
//! the number that tells you whether the shape is per-frame, per-SB or
//! per-block.
//!
//! It wraps `System` in a counting `GlobalAlloc` and reports, for the region
//! that `drv-aom` times (`EncodeCell::port_encode`) alone:
//!
//! * `alloc` / `alloc_zeroed` / `realloc` / `dealloc` call counts,
//! * total and peak live bytes,
//! * a histogram of allocation sizes by power-of-two bucket,
//!
//! with the untimed C bootstrap encode measured separately so it can be
//! subtracted rather than guessed at.
//!
//! Counters are `Relaxed` atomics; the encode is single-threaded, so the only
//! cost is one atomic RMW per allocator call and the numbers are exact.
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example eprof_alloc -- \
//!     1024 1024 44 6 ~/tmp/xb/src/photo_1024.yuv
//! ```

use aom_bench::EncodeCell;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

const NBUCKET: usize = 32;

/// Exact allocation sizes worth counting individually, because each one is
/// emitted by exactly ONE call site and so its count IS that site's call count.
/// (No hashing, no allocation inside the allocator — a linear scan of 8.)
///
///   4225  = `extract_intra_cnn_window`'s `vec![0u8; 65*65]`
///   16900 = `cnn_predict`'s layer-0 input `Vec<f32>` (65*65 f32)
///   20480 = `cnn_predict`'s layer-0 output (20 ch * 16 * 16 f32)
///    5120 = layer-1 output (20 * 8 * 8 f32)
///    1280 = layer-2 output (20 * 4 * 4 f32)
///      64 = layer-3 output (4 * 2 * 2 f32)
///      80 = layer-4 output (20 * 1 * 1 f32)
const WATCH: [usize; 7] = [4225, 16900, 20480, 5120, 1280, 64, 80];
static WATCH_N: [AtomicUsize; WATCH.len()] = [const { AtomicUsize::new(0) }; WATCH.len()];

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
    let mut w = 0;
    while w < WATCH.len() {
        if WATCH[w] == size {
            WATCH_N[w].fetch_add(1, Relaxed);
            break;
        }
        w += 1;
    }
    let b = (usize::BITS - size.leading_zeros()) as usize;
    BUCKET[b.min(NBUCKET - 1)].fetch_add(1, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

struct Counting;

// SAFETY-equivalent note: every method forwards to `System` unchanged; the only
// added work is atomic bookkeeping, so the allocator contract is System's.
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

fn report(tag: &str, a: Snap, b: Snap, px: usize, blocks: Option<usize>) {
    let calls = (b.alloc - a.alloc) + (b.zeroed - a.zeroed) + (b.realloc - a.realloc);
    println!("--- {tag} ---");
    println!("  alloc          {:>12}", b.alloc - a.alloc);
    println!("  alloc_zeroed   {:>12}", b.zeroed - a.zeroed);
    println!("  realloc        {:>12}", b.realloc - a.realloc);
    println!("  dealloc        {:>12}", b.dealloc - a.dealloc);
    println!("  TOTAL alloc-ish{:>12}", calls);
    println!("  bytes          {:>12}", b.bytes - a.bytes);
    println!("  peak live      {:>12}", b.peak.max(a.peak));
    println!("  per megapixel  {:>12.0}", calls as f64 / (px as f64 / 1e6));
    if let Some(n) = blocks {
        println!("  per superblock {:>12.1}", calls as f64 / n as f64);
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        eprintln!("usage: eprof_alloc <w> <h> <cq> <cpu-used> <in.yuv>");
        std::process::exit(2);
    }
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let q: i32 = a[3].parse().unwrap();
    let speed: i32 = a[4].parse().unwrap();
    let buf = std::fs::read(&a[5]).expect("read .yuv");
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(buf.len(), w * h + 2 * cw * ch, "I420 size mismatch");
    let up = |s: &[u8]| s.iter().map(|&b| u16::from(b)).collect::<Vec<u16>>();
    let cell = EncodeCell {
        label: "eprof".to_string(),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: q,
        speed,
        bd: 8,
        y: up(&buf[..w * h]),
        u: up(&buf[w * h..w * h + cw * ch]),
        v: up(&buf[w * h + cw * ch..]),
    };

    let watch0 = || WATCH_N.iter().map(|c| c.load(Relaxed)).collect::<Vec<_>>();
    let s0 = snap();
    let bootstrap = cell.c_encode_defaults();
    assert!(!bootstrap.is_empty());
    let s1 = snap();
    // Warm once so lazily-built statics/caches are not charged to the measured
    // encode (the timed region in drv-aom is likewise preceded by warmups).
    let _ = cell.port_encode(&bootstrap);
    let s2 = snap();
    let w2 = watch0();
    let out = cell.port_encode(&bootstrap);
    let s3 = snap();
    let w3 = watch0();

    let sb = ((w + 63) / 64) * ((h + 63) / 64);
    report("C bootstrap encode (UNTIMED in drv-aom)", s0, s1, w * h, None);
    report("port_encode, warm-up call", s1, s2, w * h, Some(sb));
    report("port_encode, MEASURED call", s2, s3, w * h, Some(sb));
    println!("--- size histogram, MEASURED call (log2 bucket: count) ---");
    println!("  (cumulative over the whole run; the two port calls dominate)");
    for (i, c) in BUCKET.iter().enumerate() {
        let v = c.load(Relaxed);
        if v > 0 {
            println!("  2^{:<2} .. {:<12} {:>12}", i, 1usize << i, v);
        }
    }
    println!("--- exact-size call-site counters, MEASURED call ---");
    println!("  (each of these sizes has exactly one call site; the count IS its call count)");
    for (i, sz) in WATCH.iter().enumerate() {
        let n = w3[i] - w2[i];
        println!("  size {sz:<7} n = {n:<10} per 64x64 SB = {:.2}", n as f64 / sb as f64);
    }
    println!("coded frame bytes = {}", out.len());
    println!("superblocks(64x64) = {sb}");
}
