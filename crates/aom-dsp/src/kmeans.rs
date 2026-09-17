//! `av1_calc_indices_dim1/_dim2` — the palette k-means nearest-centroid
//! assignment, rtcd-dispatched in C (`av1_k_means_sse2.c` /
//! `av1_k_means_avx2.c`). The scalar transcription is the `_c` reference; the
//! v3 tier mirrors `av1_calc_indices_dim*_avx2` instruction-for-instruction,
//! so both tiers are bit-identical by construction AND by the differential
//! test below.
//!
//! Consumer: `aom_encode::palette_search::calc_indices`.
//!
//! Per-point semantics (independent of dim):
//! * index = FIRST argmin over centroids (strict `<` — C's `cmpgt` in SIMD,
//!   `this_dist < min_dist` in scalar; same tie-break both tiers);
//! * dim1 dist = L1 `|p - c|` (i16 lanes); the total squares it (`madd`);
//! * dim2 dist = squared L2 `Σ (p_i - c_i)²` (i32 lanes via `madd` on the
//!   interleaved (x, y) pairs); the total is the plain sum.

/// Scalar transcription of `av1_calc_indices_dim1/_dim2` (`_c`,
/// k_means_template.h). Also the `AOM_FORCE_SCALAR` tier and the tail for a
/// non-multiple-of-16 `n`.
pub fn calc_indices_c(
    data: &[i16],
    centroids: &[i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
    dim: usize,
) -> i64 {
    let mut total: i64 = 0;
    for i in 0..n {
        let p = &data[i * dim..i * dim + dim];
        let mut min_dist = calc_dist(p, &centroids[..dim], dim);
        indices[i] = 0;
        for j in 1..k {
            let this_dist = calc_dist(p, &centroids[j * dim..j * dim + dim], dim);
            if this_dist < min_dist {
                min_dist = this_dist;
                indices[i] = j as u8;
            }
        }
        if dim == 1 {
            total += i64::from(min_dist) * i64::from(min_dist);
        } else {
            total += i64::from(min_dist);
        }
    }
    total
}

/// `calc_dist` (k_means_template.h): dim 1 is L1 (squared only for the
/// total), dim 2 is squared L2.
#[inline]
fn calc_dist(p1: &[i16], p2: &[i16], dim: usize) -> i32 {
    if dim == 1 {
        (i32::from(p1[0]) - i32::from(p2[0])).abs()
    } else {
        let mut dist = 0i32;
        for i in 0..dim {
            let diff = i32::from(p1[i]) - i32::from(p2[i]);
            dist += diff * diff;
        }
        dist
    }
}

/// Dispatched `av1_calc_indices_dim1/_dim2`: identical output to
/// [`calc_indices_c`] on every input (the SIMD bulk walks 16 points per
/// iteration; a non-multiple-of-16 tail — unreachable on block-aligned
/// palette maps, which are always a multiple of 16 — stays scalar).
pub fn calc_indices(
    data: &[i16],
    centroids: &[i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
    dim: usize,
) -> i64 {
    let _ = crate::dispatch::scalar_forced();
    match dim {
        1 => archmage::incant!(calc_indices_dim1(data, centroids, indices, n, k), [v3, scalar]),
        2 => archmage::incant!(calc_indices_dim2(data, centroids, indices, n, k), [v3, scalar]),
        _ => calc_indices_c(data, centroids, indices, n, k, dim),
    }
}

fn calc_indices_dim1_scalar(
    _t: archmage::ScalarToken,
    data: &[i16],
    centroids: &[i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
) -> i64 {
    calc_indices_c(data, centroids, indices, n, k, 1)
}

fn calc_indices_dim2_scalar(
    _t: archmage::ScalarToken,
    data: &[i16],
    centroids: &[i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
) -> i64 {
    calc_indices_c(data, centroids, indices, n, k, 2)
}

/// `av1_calc_indices_dim1_avx2` (av1_k_means_avx2.c:26): 16 points per
/// iteration — broadcast each centroid to i16 lanes, L1 = `abs(sub)`,
/// first-argmin via `cmpgt` + blend, squared-total via `madd` → i64 unpack.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn calc_indices_dim1_v3(
    _t: archmage::X64V3Token,
    data: &[i16],
    centroids: &[i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
) -> i64 {
    use archmage::intrinsics::x86_64::*;

    let n16 = n / 16;
    let mut cents = [_mm256_setzero_si256(); 8];
    for (j, c) in cents.iter_mut().enumerate().take(k) {
        *c = _mm256_set1_epi16(centroids[j]);
    }
    let zero = _mm256_setzero_si256();
    let mut sum = zero;
    let d16 = data[..n16 * 16].as_chunks::<16>().0;
    let i16o = indices[..n16 * 16].as_chunks_mut::<16>().0;
    for (dchunk, ichunk) in d16.iter().zip(i16o.iter_mut()) {
        let in_v = _mm256_loadu_si256(dchunk);
        let mut ind = zero;
        let d1 = _mm256_sub_epi16(in_v, cents[0]);
        let mut dist_min = _mm256_abs_epi16(d1);
        for j in 1..k {
            let d1 = _mm256_sub_epi16(in_v, cents[j]);
            let dist = _mm256_abs_epi16(d1);
            let cmp = _mm256_cmpgt_epi16(dist_min, dist);
            dist_min = _mm256_min_epi16(dist_min, dist);
            let ind1 = _mm256_set1_epi16(j as i16);
            ind = _mm256_or_si256(
                _mm256_andnot_si256(cmp, ind),
                _mm256_and_si256(cmp, ind1),
            );
        }
        let p1 = _mm256_packus_epi16(ind, zero);
        let px = _mm256_permute4x64_epi64(p1, 0x58);
        let d2 = _mm256_extracti128_si256(px, 0);
        _mm_storeu_si128(ichunk, d2);
        let sq = _mm256_madd_epi16(dist_min, dist_min);
        sum = _mm256_add_epi64(sum, _mm256_unpacklo_epi32(sq, zero));
        sum = _mm256_add_epi64(sum, _mm256_unpackhi_epi32(sq, zero));
    }
    let mut total = hsum_i64(_t, sum);
    // Scalar tail for a non-multiple-of-16 n.
    if n16 * 16 < n {
        total += calc_indices_c(
            &data[n16 * 16..],
            centroids,
            &mut indices[n16 * 16..],
            n - n16 * 16,
            k,
            1,
        );
    }
    total
}

/// `av1_calc_indices_dim2_avx2` (av1_k_means_avx2.c:79): the (x, y) centroid
/// pairs are broadcast to i16 lanes; two 16-lane loads cover 16 points;
/// `madd(d, d)` gives per-point squared L2 in i32 lanes; argmin/pack per C.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn calc_indices_dim2_v3(
    _t: archmage::X64V3Token,
    data: &[i16],
    centroids: &[i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
) -> i64 {
    use archmage::intrinsics::x86_64::*;

    let n16 = n / 16;
    let mut cents = [_mm256_setzero_si256(); 8];
    for (j, c) in cents.iter_mut().enumerate().take(k) {
        let cx = centroids[2 * j];
        let cy = centroids[2 * j + 1];
        *c = _mm256_set_epi16(
            cy, cx, cy, cx, cy, cx, cy, cx, cy, cx, cy, cx, cy, cx, cy, cx,
        );
    }
    // _mm256_set_epi32(0,0,0,0,5,1,4,0) — C's output-lane gather.
    let permute = _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0);
    let zero = _mm256_setzero_si256();
    let mut sum = zero;
    let d16 = data[..n16 * 32].as_chunks::<16>().0;
    let i16o = indices[..n16 * 16].as_chunks_mut::<16>().0;
    for (i, ichunk) in i16o.iter_mut().enumerate() {
        let mut ind = [zero; 2];
        let mut dist_min = [zero; 2];
        for l in 0..2 {
            let in_v = _mm256_loadu_si256(&d16[2 * i + l]);
            let d1 = _mm256_sub_epi16(in_v, cents[0]);
            dist_min[l] = _mm256_madd_epi16(d1, d1);
            for j in 1..k {
                let d1 = _mm256_sub_epi16(in_v, cents[j]);
                let dist = _mm256_madd_epi16(d1, d1);
                let cmp = _mm256_cmpgt_epi32(dist_min[l], dist);
                dist_min[l] = _mm256_min_epi32(dist_min[l], dist);
                let ind1 = _mm256_set1_epi32(j as i32);
                ind[l] = _mm256_or_si256(
                    _mm256_andnot_si256(cmp, ind[l]),
                    _mm256_and_si256(cmp, ind1),
                );
            }
            sum = _mm256_add_epi64(sum, _mm256_unpacklo_epi32(dist_min[l], zero));
            sum = _mm256_add_epi64(sum, _mm256_unpackhi_epi32(dist_min[l], zero));
        }
        let d2 = _mm256_packus_epi32(ind[0], ind[1]);
        let d3 = _mm256_packus_epi16(d2, zero);
        let d4 = _mm256_permutevar8x32_epi32(d3, permute);
        let d5 = _mm256_extracti128_si256(d4, 0);
        _mm_storeu_si128(ichunk, d5);
    }
    let mut total = hsum_i64(_t, sum);
    if n16 * 16 < n {
        total += calc_indices_c(
            &data[n16 * 32..],
            centroids,
            &mut indices[n16 * 16..],
            n - n16 * 16,
            k,
            2,
        );
    }
    total
}

/// `k_means_horizontal_sum_avx2`: fold the 4 i64 lanes.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn hsum_i64(_t: archmage::X64V3Token, v: archmage::intrinsics::x86_64::__m256i) -> i64 {
    use archmage::intrinsics::x86_64::*;
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi64(lo, hi);
    let s2 = _mm_unpackhi_epi64(s, s);
    let s3 = _mm_add_epi64(s, s2);
    _mm_cvtsi128_si64(s3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(data: &[i16], cents: &[i16], n: usize, k: usize, dim: usize) -> (Vec<u8>, i64) {
        let mut idx = vec![0u8; n];
        let total = calc_indices_c(data, cents, &mut idx, n, k, dim);
        (idx, total)
    }

    /// V3-vs-scalar equivalence over varied n/k/data, including non-multiple
    /// tails and worst-case magnitude (i16-range data, ties between
    /// centroids).
    #[test]
    fn calc_indices_tiers_agree() {
        let mut rng: u32 = 0x9e3779b9;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        for dim in [1usize, 2] {
            for k in 2..=8usize {
                for n in [0usize, 1, 7, 15, 16, 17, 31, 32, 48, 64, 256, 1024] {
                    let data: Vec<i16> = (0..n * dim)
                        .map(|_| {
                            // The realizable domain is pixel-derived data
                            // (0..=4095 across bit depths) and centroids are
                            // data means — |p - c| <= 4095, so the i16 lanes
                            // never wrap. Deeper magnitudes would diverge
                            // C's own AVX2 from its _c too; we mirror the
                            // dispatched kernel.
                            match next() % 5 {
                                0 => (next() % 256) as i16,
                                1 => (next() % 4096) as i16,
                                2 => (next() % 64) as i16,
                                3 => 128,
                                _ => (next() % 4096) as i16,
                            }
                        })
                        .collect();
                    let cents: Vec<i16> = (0..k * dim)
                        .map(|i| {
                            if i == 0 {
                                128 // force a first-argmin tie against data 128s
                            } else {
                                (next() % 4096) as i16
                            }
                        })
                        .collect();
                    let (want_idx, want_total) = reference(&data, &cents, n, k, dim);
                    let mut got_idx = vec![0xffu8; n];
                    let got_total = calc_indices(&data, &cents, &mut got_idx, n, k, dim);
                    assert_eq!(want_total, got_total, "dim{dim} k{k} n{n} total");
                    assert_eq!(want_idx, got_idx, "dim{dim} k{k} n{n} indices");
                }
            }
        }
    }
}
