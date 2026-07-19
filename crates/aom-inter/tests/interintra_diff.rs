//! Differential harness for the interintra blend + wedge codebook vs the REAL
//! exported C libaom v3.14.1:
//!
//!  1. `blend_a64_mask_matches_c` — the 2-D A64 masked blend (all four mask
//!     subsampling cases) vs `aom_blend_a64_mask_c` (`aom_sys_ref::ref_blend_a64_mask`).
//!  2. `wedge_mask_matches_c` — the compound wedge codebook (sign 0) vs the baked
//!     `av1_wedge_params_lookup[bsize].masks[0][index]` (`ref_ii_wedge_mask`).
//!  3. `combine_interintra_wedge_matches_c` — the whole wedge `combine_interintra`
//!     path (mask fetch + plane subsampling + blend) end-to-end vs C.
//!
//! These lock the arithmetic (both smooth + wedge share the blend) and the wedge
//! mask generation. The smooth mask itself is a direct transcription of C's
//! `ii_weights1d` (128 values, count-checked); `smooth_mask_structure` guards its
//! per-mode indexing, and the frame-MD5 decoder gate covers the smooth combine
//! end-to-end.

use aom_inter::interintra::{blend_a64_mask, build_smooth_interintra_mask, combine_interintra, wedge_mask};
use aom_sys_ref::{ref_blend_a64_mask, ref_ii_wedge_mask};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

// wedge-eligible (bsize, bw, bh) — BLOCK_8X8..BLOCK_32X32 + 8X32/32X8.
const WEDGE_BSIZES: [(usize, usize, usize); 9] = [
    (3, 8, 8),
    (4, 8, 16),
    (5, 16, 8),
    (6, 16, 16),
    (7, 16, 32),
    (8, 32, 16),
    (9, 32, 32),
    (18, 8, 32),
    (19, 32, 8),
];

#[test]
fn blend_a64_mask_matches_c() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let sizes = [4usize, 8, 16, 32, 64];
    let mut checked = 0u32;
    for &w in &sizes {
        for &h in &sizes {
            for &(subw, subh) in &[(false, false), (true, false), (false, true), (true, true)] {
                for _ in 0..40 {
                    // The mask is at the (possibly 2x) luma resolution.
                    let mw = if subw { w * 2 } else { w };
                    let mh = if subh { h * 2 } else { h };
                    let src0: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
                    let src1: Vec<u8> = (0..w * h).map(|_| rng.below(256) as u8).collect();
                    let mask: Vec<u8> = (0..mw * mh).map(|_| rng.below(65) as u8).collect();

                    let c = ref_blend_a64_mask(&src0, &src1, &mask, mw, w, h, subw, subh);

                    let s0: Vec<u16> = src0.iter().map(|&v| v as u16).collect();
                    let s1: Vec<u16> = src1.iter().map(|&v| v as u16).collect();
                    let mut dst = vec![0u16; w * h];
                    blend_a64_mask(&mut dst, w, &s0, w, &s1, w, &mask, mw, w, h, subw, subh);
                    let got: Vec<u8> = dst.iter().map(|&v| v as u8).collect();
                    assert_eq!(
                        got, c,
                        "blend_a64_mask mismatch w={w} h={h} subw={subw} subh={subh}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 4000, "expected a dense sweep, got {checked}");
    eprintln!("blend_a64_mask: {checked} configs byte-identical to aom_blend_a64_mask_c");
}

#[test]
fn wedge_mask_matches_c() {
    let mut n = 0u32;
    for &(bsize, bw, bh) in &WEDGE_BSIZES {
        for index in 0..16usize {
            let got = wedge_mask(bsize, index).expect("wedge bsize has a mask");
            let c = ref_ii_wedge_mask(bsize, index, bw, bh).expect("C wedge mask");
            assert_eq!(got.len(), bw * bh);
            assert_eq!(
                got, c,
                "wedge_mask mismatch bsize={bsize} index={index} (bw={bw} bh={bh})"
            );
            n += 1;
        }
    }
    assert_eq!(n, 9 * 16);
    eprintln!("wedge_mask: {n} (bsize,index) baked masks byte-identical to av1_wedge_params_lookup");
}

#[test]
fn combine_interintra_wedge_matches_c() {
    let mut rng = Rng(0xdead_beef_0bad_f00d);
    let mut n = 0u32;
    // Both 4:4:4 (plane == luma) and 4:2:0 (plane subsampled) — the subw/subh
    // averaging of the luma-resolution wedge mask is the interesting axis.
    for &(bsize, bw, bh) in &WEDGE_BSIZES {
        for &(ss_x, ss_y) in &[(0usize, 0usize), (1, 1)] {
            let pw = bw >> ss_x;
            let ph = bh >> ss_y;
            // get_plane_block_size for these bsizes at 4:2:0: reverse via dims.
            // combine_interintra takes plane_bsize; find it from (pw, ph).
            let plane_bsize = bsize_from_dims(pw, ph);
            for index in 0..16usize {
                let inter: Vec<u16> = (0..pw * ph).map(|_| rng.below(256) as u16).collect();
                let intra: Vec<u16> = (0..pw * ph).map(|_| rng.below(256) as u16).collect();
                let mut comp = vec![0u16; pw * ph];
                combine_interintra(
                    3, // II_SMOOTH (unused on the wedge path)
                    true, index, bsize, plane_bsize, &mut comp, pw, &inter, pw, &intra, pw,
                );
                // Reference: C blend of intra (src0) vs inter (src1) with the baked
                // wedge mask, subsampled by (subw, subh).
                let subw = (2 * (bw >> 2)) == pw; // 2*mi_size_wide == plane bw  (mi=4px)
                let subh = (2 * (bh >> 2)) == ph;
                let wmask = ref_ii_wedge_mask(bsize, index, bw, bh).unwrap();
                let intra_u8: Vec<u8> = intra.iter().map(|&v| v as u8).collect();
                let inter_u8: Vec<u8> = inter.iter().map(|&v| v as u8).collect();
                let c = ref_blend_a64_mask(&intra_u8, &inter_u8, &wmask, bw, pw, ph, subw, subh);
                let got: Vec<u8> = comp.iter().map(|&v| v as u8).collect();
                assert_eq!(
                    got, c,
                    "combine_interintra wedge mismatch bsize={bsize} ss=({ss_x},{ss_y}) index={index}"
                );
                n += 1;
            }
        }
    }
    eprintln!("combine_interintra (wedge): {n} configs byte-identical to C blend");
}

/// Reverse BLOCK_SIZE lookup by (w,h) in px, over the interintra-relevant sizes.
fn bsize_from_dims(w: usize, h: usize) -> usize {
    const W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    (0..22)
        .find(|&i| W[i] == w && H[i] == h)
        .unwrap_or_else(|| panic!("no BLOCK_SIZE for {w}x{h}"))
}

/// Structural guard for `build_smooth_interintra_mask`: II_DC is flat 32; II_V
/// varies down rows only; II_H across cols only; II_SMOOTH by min(row,col) — all
/// indexing the shared `ii_weights1d` at the plane's `ii_size_scales` stride.
#[test]
fn smooth_mask_structure() {
    // ii_size_scales for BLOCK_16X16 (bsize 6) == 8.
    let bsize = 6usize;
    let (bw, bh) = (16usize, 16usize);
    let dc = build_smooth_interintra_mask(0, bsize);
    assert!(dc.iter().all(|&m| m == 32), "II_DC flat 32");

    let v = build_smooth_interintra_mask(1, bsize);
    for i in 0..bh {
        // constant across a row
        for j in 1..bw {
            assert_eq!(v[i * bw + j], v[i * bw], "II_V constant across row {i}");
        }
    }
    // strictly non-increasing down rows (ii_weights1d falls off)
    for i in 1..bh {
        assert!(v[i * bw] <= v[(i - 1) * bw], "II_V falls off down rows");
    }

    let h = build_smooth_interintra_mask(2, bsize);
    for j in 0..bw {
        for i in 1..bh {
            assert_eq!(h[i * bw + j], h[j], "II_H constant down col {j}");
        }
    }

    let sm = build_smooth_interintra_mask(3, bsize);
    // II_SMOOTH[i][j] == II_V value at min(i,j)
    for i in 0..bh {
        for j in 0..bw {
            let k = i.min(j);
            assert_eq!(sm[i * bw + j], v[k * bw], "II_SMOOTH == weight[min(i,j)]");
        }
    }
    eprintln!("smooth_mask_structure: II_DC/V/H/SMOOTH indexing verified");
}
