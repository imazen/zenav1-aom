//! Microbench (aarch64 only): port forward transforms vs the REAL C
//! `av1_fwd_txfm2d_*_neon` symbols, which exist only in an aarch64 libaom.
//! Arch-gated so an x86-64 `--workspace` build (which compiles examples) does
//! not fail on a symbol the host oracle never exports (found 2026-09-24).
#[cfg(target_arch = "aarch64")]
mod bench {
    use aom_dsp::transform::txfm2d::av1_fwd_txfm2d;
    use std::time::Instant;

    unsafe extern "C" {
        fn av1_fwd_txfm2d_4x4_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_8x8_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_16x16_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_8x16_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_16x8_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_4x8_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_8x4_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_16x4_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
        fn av1_fwd_txfm2d_32x32_neon(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    }

    // TX_SIZE indices: 0=4x4 1=8x8 2=16x16 3=32x32 4=64x64 5=4x8 6=8x4 7=8x16 8=16x8
    pub fn main() {
        aom_sys_ref::ref_init();
        let shapes: &[(
            &str,
            usize,
            usize,
            usize,
            unsafe extern "C" fn(*const i16, *mut i32, i32, i32, i32),
        )] = &[
            ("4x4", 0, 4, 4, av1_fwd_txfm2d_4x4_neon),
            ("8x8", 1, 8, 8, av1_fwd_txfm2d_8x8_neon),
            ("16x16", 2, 16, 16, av1_fwd_txfm2d_16x16_neon),
            ("4x8", 5, 4, 8, av1_fwd_txfm2d_4x8_neon),
            ("8x4", 6, 8, 4, av1_fwd_txfm2d_8x4_neon),
            ("8x16", 7, 8, 16, av1_fwd_txfm2d_8x16_neon),
            ("16x8", 8, 16, 8, av1_fwd_txfm2d_16x8_neon),
            ("16x4", 13, 16, 4, av1_fwd_txfm2d_16x4_neon),
            ("32x32", 3, 32, 32, av1_fwd_txfm2d_32x32_neon),
        ];
        for &(name, idx, w, h, cf) in shapes {
            let n = w * h;
            let input: Vec<i16> = (0..n).map(|i| ((i * 37 + 13) % 512 - 256) as i16).collect();
            let mut out_p = vec![0i32; n.max(4096)];
            let mut out_c = vec![0i32; n.max(4096)];
            // DCT_DCT = tx_type 0
            for tx in [0usize, 1] {
                let reps = 200_000usize;
                let t = Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(av1_fwd_txfm2d(
                        black(input.as_slice()),
                        black(out_p.as_mut_slice()),
                        w,
                        tx,
                        idx,
                    ));
                }
                let port = t.elapsed();
                let t = Instant::now();
                for _ in 0..reps {
                    unsafe {
                        cf(
                            black(input.as_ptr()),
                            black(out_c.as_mut_ptr()),
                            w as i32,
                            tx as i32,
                            8,
                        )
                    };
                }
                let c = t.elapsed();
                let ok = out_p[..n] == out_c[..n];
                println!(
                    "{name} tx{tx}: port {:7.0}ns  C {:7.0}ns  ratio {:.2}  bytes_match={ok}",
                    port.as_nanos() as f64 / reps as f64,
                    c.as_nanos() as f64 / reps as f64,
                    port.as_nanos() as f64 / c.as_nanos() as f64
                );
            }
        }
    }
    fn black<T>(x: T) -> T {
        std::hint::black_box(x)
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    bench::main()
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("fwd_neon_bench: aarch64-only (needs the NEON C shim); nothing to run here");
}
