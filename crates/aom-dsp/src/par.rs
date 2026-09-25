//! Cooperative-parallelism shim: the tiny spawn surface the encoder's
//! multi-worker spans need, over ONE of two executors.
//!
//! - default (no features): `std::thread::scope` — spawn `workers` OS
//!   threads, join them all. Zero dependencies.
//! - `rayon` feature: tasks go to the host's shared `rayon` pool instead
//!   of dedicated threads, so a caller that already runs a rayon system
//!   (zenavif does) doesn't get `encodes-in-flight × threads` runnable
//!   threads fighting its own pool for cores.
//!
//! Both backends give the same contract: `f(w)` for `w in 0..workers`
//! may run in any order on any thread, panics propagate to the caller,
//! and every result is collected — scheduling is free to vary, so work
//! must already be deterministic-under-reorder (the encoder's spans
//! are: bands are disjoint `&mut` and merges are keyed on tile index).
//! `workers` is a CONCURRENCY CEILING, not a dedicated-core promise —
//! under rayon the pool may run the tasks on fewer (or interleave them
//! with unrelated work), which is exactly the behaviour a shared server
//! wants.

#[cfg(feature = "rayon")]
pub fn map_workers<T, F>(workers: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Sync,
    T: Send,
{
    use rayon::prelude::*;
    (0..workers).into_par_iter().map(|w| f(w)).collect()
}

#[cfg(not(feature = "rayon"))]
pub fn map_workers<T, F>(workers: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Sync,
    T: Send,
{
    if workers <= 1 {
        return vec![f(0)];
    }
    let f = &f;
    std::thread::scope(|s| {
        (0..workers)
            .map(|w| s.spawn(move || f(w)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    })
}

/// Run two independent closures concurrently and return both results.
#[cfg(feature = "rayon")]
pub fn join<A, B, FA, FB>(a: FA, b: FB) -> (A, B)
where
    FA: FnOnce() -> A + Send,
    FB: FnOnce() -> B + Send,
    A: Send,
    B: Send,
{
    rayon::join(a, b)
}

#[cfg(not(feature = "rayon"))]
pub fn join<A, B, FA, FB>(a: FA, b: FB) -> (A, B)
where
    FA: FnOnce() -> A + Send,
    FB: FnOnce() -> B + Send,
    A: Send,
    B: Send,
{
    std::thread::scope(|s| {
        let hb = s.spawn(b);
        let ra = a();
        let rb = hb.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
        (ra, rb)
    })
}
