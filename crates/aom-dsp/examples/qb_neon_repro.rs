use aom_sys_ref as c;
use aom_dsp::quant::aom_quantize_b_no_qmatrix;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn coeff(&mut self) -> i32 { (self.next() % (1 << 19)) as i32 - (1 << 18) }
    fn pos_i16(&mut self, lo: i32, hi: i32) -> i16 { (lo + (self.next() % (hi - lo) as u64) as i32) as i16 }
}
fn perm(rng: &mut Rng, n: usize) -> Vec<i16> {
    let mut v: Vec<i16> = (0..n as i16).collect();
    for i in (1..n).rev() { let j = (rng.next() % (i as u64 + 1)) as usize; v.swap(i, j); }
    v
}
fn invert(scan: &[i16]) -> Vec<i16> {
    let mut iscan = vec![0i16; scan.len()];
    for (i, &rc) in scan.iter().enumerate() { iscan[rc as usize] = i as i16; }
    iscan
}

fn main() {
    let mut rng = Rng(0x_d1ce_f00d_a5a5_1234);
    let mut mism_neon = 0u32; let mut mism_scalar = 0u32; let mut total = 0u32;
    for log_scale in 0..=2i32 {
        for &n in &[8usize, 16, 64, 256, 1024] {
            for _ in 0..2000 {
                let coeff: Vec<i32> = (0..n).map(|_| rng.coeff()).collect();
                let zbin = [rng.pos_i16(1, 1000), rng.pos_i16(1, 1000)];
                let round = [rng.pos_i16(1, 2000), rng.pos_i16(1, 2000)];
                let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                let quant_shift = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
                let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];
                let scan = perm(&mut rng, n);
                let iscan = invert(&scan);
                let mut q_got = vec![0i32; n]; let mut dq_got = vec![0i32; n];
                let eob_got = aom_quantize_b_no_qmatrix(
                    &zbin, &round, &quant, &quant_shift, &dequant, log_scale,
                    &scan, &iscan, &coeff, &mut q_got, &mut dq_got);
                let (qn, dqn, en) = c::ref_quantize_b_neon(
                    log_scale, &coeff, &zbin, &round, &quant, &quant_shift, &dequant, &scan, &iscan);
                let (qs, dqs, es) = c::ref_quantize_b(
                    log_scale, &coeff, &zbin, &round, &quant, &quant_shift, &dequant, &scan);
                total += 1;
                if (q_got.clone(), dq_got.clone(), eob_got) != (qn.clone(), dqn.clone(), en) {
                    mism_neon += 1;
                    if mism_neon <= 3 {
                        let pos = q_got.iter().zip(&qn).position(|(a,b)| a!=b).unwrap_or(usize::MAX);
                        eprintln!("NEON MISM ls={log_scale} n={n} eob {eob_got} vs {en} qpos={pos}");
                        eprintln!("  coeff[..16]={:?}", &coeff[..16.min(n)]);
                        eprintln!("  zbin={zbin:?} round={round:?} quant={quant:?} qs={quant_shift:?} dq={dequant:?}");
                    }
                }
                if (qn, dqn, en) != (qs, dqs, es) { mism_scalar += 1; }
            }
        }
    }
    println!("total={total} port-vs-CNEON mism={mism_neon} CNEON-vs-Cscalar mism={mism_scalar}");
}
