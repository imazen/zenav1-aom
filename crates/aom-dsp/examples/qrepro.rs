//! Microbench (aarch64 only): port NEON quantize transcription vs the REAL C
//! NEON kernel vs the port scalar body. `ref_quantize_fp_neon` exists only in an
//! aarch64 `aom-sys-ref`, so the whole bench is arch-gated — an ungated import
//! here broke every x86-64 `--workspace` build from `89e90f6` until 2026-09-24
//! (`cargo test --workspace` and nextest build examples; `-p aom-encode` does
//! not, which is why `gate-encode` never saw it).
#[cfg(target_arch = "aarch64")]
mod bench {
    use aom_dsp::quant::simd::av1_quantize_lp_dispatch;
    use aom_dsp::quant::{av1_quantize_fp_no_qmatrix, simd::av1_quantize_fp_no_qmatrix_dispatch};
    use aom_sys_ref::{ref_quantize_fp_neon, ref_quantize_lp_simd};
    use std::time::Instant;

    pub fn main() {
        let n = 256usize;
        let quant = [1500i16, 1600];
        let dequant = [40i16, 48];
        let round = [700i16, 750];
        let coeff: Vec<i32> = (0..n)
            .map(|i| ((i * 2654435761usize) % 30000) as i32 - 15000)
            .collect();
        let scan: Vec<i16> = (0..n as i16).collect();
        let mut iscan = vec![0i16; n];
        for (k, &p) in scan.iter().enumerate() {
            iscan[p as usize] = k as i16;
        }
        let mut q = vec![0i32; n];
        let mut d = vec![0i32; n];
        let iters = 20000;

        for ls in 0..3i32 {
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..iters {
                acc += av1_quantize_fp_no_qmatrix_dispatch(
                    &quant, &dequant, &round, ls, &scan, &iscan, &coeff, &mut q, &mut d,
                ) as u64;
            }
            let port = t.elapsed();
            let t = Instant::now();
            for _ in 0..iters {
                acc += ref_quantize_fp_neon(ls, &coeff, &round, &quant, &dequant, &scan, &iscan).2
                    as u64;
            }
            let cneon = t.elapsed();
            let t = Instant::now();
            for _ in 0..iters {
                acc += av1_quantize_fp_no_qmatrix(
                    &quant, &dequant, &round, ls, &scan, &coeff, &mut q, &mut d,
                ) as u64;
            }
            let pscalar = t.elapsed();
            eprintln!(
                "fp ls={ls}: port-neon {:?} | c-neon {:?} | port-scalar {:?} (acc {acc})",
                port, cneon, pscalar
            );
        }

        // lp
        let round8 = [700i16; 8];
        let quant8 = [1500i16; 8];
        let dequant8 = [40i16; 8];
        let coeff16: Vec<i16> = coeff.iter().map(|&c| c as i16).collect();
        let mut q16 = vec![0i16; n];
        let mut d16 = vec![0i16; n];
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            acc += av1_quantize_lp_dispatch(
                &round8, &quant8, &dequant8, &scan, &iscan, &coeff16, n, &mut q16, &mut d16,
            ) as u64;
        }
        let port = t.elapsed();
        let t = Instant::now();
        for _ in 0..iters {
            acc +=
                ref_quantize_lp_simd(&coeff16, &round8, &quant8, &dequant8, &scan, &iscan).2 as u64;
        }
        let cneon = t.elapsed();
        eprintln!(
            "lp:      port-neon {:?} | c-neon {:?} (acc {acc})",
            port, cneon
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    bench::main()
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("qrepro: aarch64-only microbench (needs the NEON C shim); nothing to run here");
}
