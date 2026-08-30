//! Differential: the port's palette k-means kernels (`palette_search::{calc_indices,
//! k_means}` — transcribed from the `_c` templates) against the kernels real aomenc
//! DISPATCHES (`av1_calc_indices_dim{1,2}` sse2/avx2/neon via rtcd, and the
//! `av1_k_means_dim{1,2}` templates over them). Playbook §1/§12: before KB-41 no
//! differential locked these, so "the port's palette search picks differently"
//! could not be separated from "the kernels disagree on ties / accumulation".
//! Data shapes: 8-bit values (the bd8 palette path feeds raw samples), the hbd
//! `<< 4` range libaom's own SIMD test uses, and a tie-rich low-cardinality range
//! (2..6 distinct values) where nearest-centroid ties are dense.
use aom_encode::palette_search::{calc_indices, k_means};
use aom_sys_ref as c;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn run_case(rng: &mut Lcg, n: usize, k: usize, dim: usize, lo: i16, hi: i16, shift: u8) -> usize {
    let span = (hi - lo + 1) as u32;
    let data: Vec<i16> = (0..n * dim).map(|_| ((lo as i32 + (rng.next() % span) as i32) << shift) as i16).collect();
    let cents: Vec<i16> = (0..k * dim).map(|_| ((lo as i32 + (rng.next() % span) as i32) << shift) as i16).collect();
    let mut mism = 0;
    // calc_indices
    let mut ia = vec![0u8; n];
    let mut ib = vec![0u8; n];
    let da = calc_indices(&data, &cents, &mut ia, n, k, dim);
    let db = c::ref_calc_indices(&data, &cents, &mut ib, k, dim);
    if ia != ib || da != db {
        mism += 1;
        eprintln!("calc_indices MISMATCH n={n} k={k} dim={dim} range={lo}..{hi}<<{shift}: dist port={da} c={db}, first index diff at {:?}", ia.iter().zip(&ib).position(|(a, b)| a != b));
    }
    // k_means (both sides start from the same centroids)
    let mut ca = cents.clone();
    let mut cb = cents.clone();
    let mut ja = vec![0u8; n];
    let mut jb = vec![0u8; n];
    k_means(&data, &mut ca, &mut ja, n, k, dim, 50);
    c::ref_k_means(&data, &mut cb, &mut jb, k, dim, 50);
    if ca[..k * dim] != cb[..k * dim] || ja != jb {
        mism += 1;
        eprintln!("k_means MISMATCH n={n} k={k} dim={dim} range={lo}..{hi}<<{shift}: centroids port={:?} c={:?}", &ca[..k * dim], &cb[..k * dim]);
    }
    mism
}

#[test]
fn palette_kmeans_kernels_match_dispatched_oracle() {
    c::ref_init();
    let mut rng = Lcg(0x5eed_4b41);
    let mut mism = 0;
    let mut cases = 0;
    for &dim in &[1usize, 2] {
        for &(lo, hi, shift) in &[(0i16, 255i16, 0u8), (0, 255, 4), (100, 105, 0), (0, 1, 0), (0, 3, 4)] {
            for &n in &[16usize, 64, 256, 1024, 4096] {
                for k in 2..=8usize {
                    for _ in 0..3 {
                        mism += run_case(&mut rng, n, k, dim, lo, hi, shift);
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("palette k-means differential: {cases} cases, {mism} mismatching");
    assert_eq!(mism, 0, "port palette k-means kernels diverge from the dispatched oracle");
}
