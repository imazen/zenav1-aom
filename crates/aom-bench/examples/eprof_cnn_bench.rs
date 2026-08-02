//! `eprof_cnn_bench` — three-way per-call cost of the intra-mode-CNN forward
//! pass, on ONE 65x65 window, for the encoder hotspot profile.
//!
//! The profile finds `aom_encode::cnn_partition::cnn::cnn_predict` to be the
//! single largest self cost in the port's encode. Two independent factors can
//! produce that: how OFTEN it runs (`eprof_alloc` counts that exactly) and how
//! much ONE run costs. This measures the second, against both C variants:
//!
//!   port      `cnn_partition::cnn::cnn_predict`     — scalar Rust
//!   c-scalar  `av1_cnn_convolve_..._valid_c`        — libaom's reference C
//!   c-simd    the runtime-dispatched libaom variant — NEON here
//!
//! All three are fed the SAME window and their outputs are compared for
//! bit-equality before any timing is reported, so a number can never come from
//! three functions that were not computing the same thing (playbook §1).
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example eprof_cnn_bench -- \
//!     1024 1024 <in.yuv> [iters]
//! ```

use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One timed batch: `iters` calls, returns ns/call.
fn bench<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_nanos() as f64 / iters as f64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: eprof_cnn_bench <w> <h> <in.yuv> [iters]");
        std::process::exit(2);
    }
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let iters: usize = a.get(4).map_or(2000, |s| s.parse().unwrap());
    let buf = std::fs::read(&a[3]).expect("read .yuv");
    assert!(buf.len() >= w * h);
    assert!(w >= 65 && h >= 65);

    // A real window out of the real source: the replicated-border 65x65 the
    // encoder extracts for superblock (1,1) — not synthetic content, because
    // the layer-0 RELU sparsity depends on it.
    let mut win = vec![0u8; 65 * 65];
    for i in 0..65 {
        let r = (64 + i).min(h - 1);
        for j in 0..65 {
            let c = (64 + j).min(w - 1);
            win[i * 65 + j] = buf[r * w + c];
        }
    }

    // --- non-vacuity + equality gate, BEFORE any timing is printed ----------
    let port = aom_encode::cnn_partition::cnn::cnn_predict(&win);
    let c_scalar = aom_sys_ref::ref_intra_cnn_run(&win, true);
    let c_simd = aom_sys_ref::ref_intra_cnn_run(&win, false);
    assert_eq!(port.len(), c_scalar.len(), "buffer length");
    assert_eq!(
        port.as_slice(),
        c_scalar.as_slice(),
        "port must be bit-identical to the C scalar oracle"
    );
    // libaom's own NEON variant is NOT bit-identical to its own `_c` variant
    // (different accumulation order / FMA contraction — the class of upstream
    // ISA divergence catalogued in docs/LIBAOM_UPSTREAM_NOTES.md). The port's
    // differential target is the `_c` oracle, which is what the assert above
    // pins; the dispatched variant is reported, not asserted.
    let ndiff = c_simd
        .iter()
        .zip(c_scalar.iter())
        .filter(|(a, b)| a != b)
        .count();
    let maxabs = c_simd
        .iter()
        .zip(c_scalar.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let nz = port.iter().filter(|v| **v != 0.0).count();
    assert!(nz > 100, "window produced a near-dead activation map (n={nz})");
    println!(
        "# port == c-scalar bit-exactly ({nz}/{} nonzero). \
         c-simd(neon) vs c-scalar: {ndiff}/{} elements differ, max |delta| = {maxabs:.3e}",
        port.len(),
        port.len()
    );

    // --- timing: interleaved rounds, per-round ns/call, median over rounds --
    const ROUNDS: usize = 7;
    let (mut p, mut cs, mut cv) = (vec![], vec![], vec![]);
    for _ in 0..ROUNDS {
        p.push(bench(iters, || {
            std::hint::black_box(aom_encode::cnn_partition::cnn::cnn_predict(std::hint::black_box(
                &win,
            )));
        }));
        cs.push(bench(iters, || {
            std::hint::black_box(aom_sys_ref::ref_intra_cnn_run(std::hint::black_box(&win), true));
        }));
        cv.push(bench(iters, || {
            std::hint::black_box(aom_sys_ref::ref_intra_cnn_run(std::hint::black_box(&win), false));
        }));
    }
    let (pm, csm, cvm) = (median(&mut p), median(&mut cs), median(&mut cv));
    println!("arm\tns_per_call\tvs_c_simd");
    println!("port(rust scalar)\t{pm:.0}\t{:.2}x", pm / cvm);
    println!("c-scalar(_c)\t{csm:.0}\t{:.2}x", csm / cvm);
    println!("c-simd(neon)\t{cvm:.0}\t1.00x");
    println!(
        "# iters/round={iters} rounds={ROUNDS}; per-round spread: port {:.1}% c-scalar {:.1}% c-simd {:.1}%",
        100.0 * (p[p.len() - 1] - p[0]) / pm,
        100.0 * (cs[cs.len() - 1] - cs[0]) / csm,
        100.0 * (cv[cv.len() - 1] - cv[0]) / cvm
    );
    println!(
        "# NOTE: the c-* arms include ref_intra_cnn_run's own `vec![0.0f32; 1636]` \
         result allocation ({} B), which the port arm does not have (it returns an array).",
        1636 * 4
    );
}
