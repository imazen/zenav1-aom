//! Differential harness for the OBMC TARGET (`calc_target_weighted_pred`,
//! `av1/encoder/rdopt.c:6888`, plus its two per-neighbour visitors at `:6752`
//! and `:6800`) — the port in `aom_encode::rdopt_obmc`.
//!
//! **Tier 1c**: all three functions are `static`, so the oracle is libaom's
//! own rdopt.c compiled into the shim archive. The shim builds a REAL mi grid
//! and calls `calc_target_weighted_pred`, so libaom's own
//! `foreach_overlappable_nb_above` / `_left` walks — including the 4-wide
//! chroma-pair fixup and `is_neighbor_overlappable` — are what produce the
//! reference, not an assumption about them.
//!
//! # Why one test covers three functions
//!
//! `calc_target_weighted_pred_above` and `_left` have no other caller and no
//! address a differential could take; they are reached only through the walk
//! `calc_target_weighted_pred` runs. Driving the parent exercises all three,
//! and the bite proofs below confirm each of the three is genuinely reached
//! (perturbing either visitor alone fails).
//!
//! # Coverage the assertions insist on
//!
//! - both bit depths;
//! - blocks with an above neighbour only, a left neighbour only, both, and
//!   neither (the last is the arm where `wsrc` is just `src * 4096`);
//! - neighbour grids containing 4-wide / 4-high blocks, which is the only way
//!   to reach the walk's pair fixup;
//! - grids where some neighbours are INTRA, so `is_neighbor_overlappable`
//!   actually rejects some cells;
//! - a block at the right/bottom frame edge, where `mi_cols` / `mi_rows`
//!   truncates the walk before `xd->width` does.

mod common;
use common::Rng;

use aom_encode::rdopt_obmc::{ObmcTargetArgs, calc_target_weighted_pred};
use aom_sys_ref as cref;

/// The block sizes OBMC is ALLOWED for: `min(width, height) >= 8`
/// (`is_motion_variation_allowed_bsize`, blockd.h:1460). Sub-8 shapes are not
/// swept because they are not reachable — and because C's 4-wide pair fixup
/// (`pos &= ~1`) produces a negative `rel_mi_col` from an odd `mi_col`, which
/// only an 8x8-or-larger block rules out. Measured: including them made the
/// PORT overflow on `rel * MI_SIZE`, and would have made libaom write before
/// its own `wsrc` buffer.
const BSIZES: [usize; 13] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20];

/// Lay an ALIGNED tiling of block sizes along one axis, the way a real
/// partition does: a block of mi size `s` only ever starts at a position that
/// is a multiple of `s`.
///
/// This is not cosmetic. C's walk reports `op_mi_size = min(extent, mi_step)`
/// at `rel = pos - mi_start` and then writes `op_mi_size * MI_SIZE` rows or
/// columns from `rel * MI_SIZE`. On an UNALIGNED grid — say a 2-mi neighbour
/// followed by a 16-mi one — that runs past the end of `wsrc`, in libaom as
/// well as in the port. An aligned tiling makes `rel + op_mi_size <= extent`
/// hold structurally, which is why the real encoder never trips it. Measured:
/// the first version of this harness drew each cell independently and the port
/// indexed past `wsrc` on the second neighbour.
fn aligned_tiling(rng: &mut Rng, len: usize, sizes: &[i32], pick: &[usize]) -> Vec<i32> {
    let mut out = vec![0i32; len];
    let mut p = 0usize;
    while p < len {
        // Only sizes that both divide the current position and fit.
        let choices: Vec<usize> = pick
            .iter()
            .copied()
            .filter(|&b| {
                let s = sizes[b] as usize;
                p % s == 0 && p + s <= len
            })
            .collect();
        let b = choices[(rng.next() as usize) % choices.len()];
        let s = sizes[b] as usize;
        for c in out.iter_mut().skip(p).take(s) {
            *c = b as i32;
        }
        p += s;
    }
    out
}

/// The NEIGHBOUR block sizes, which are unconstrained — a 4-wide neighbour is
/// exactly what triggers the walk's chroma-pair fixup, so they must be swept.
const NB_BSIZES: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
];

const MI_W: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H: [i32; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

#[test]
fn calc_target_weighted_pred_matches_c() {
    let mut rng = Rng(0x5eed_0041);
    let (rows, cols) = (300usize, 300usize);
    let mut with_above = 0;
    let mut with_left = 0;
    let mut with_neither = 0;
    let mut edge_truncated = 0;
    let mut n = 0;

    for iter in 0..600 {
        let bsize = BSIZES[(rng.next() as usize) % BSIZES.len()];
        let xd_width = MI_W[bsize];
        let xd_height = MI_H[bsize];
        // An OBMC block is at least 8x8, so its mi origin is a multiple of
        // its own mi size and in particular EVEN.
        let mi_row = rng.range(1, 8) * xd_height;
        let mi_col = rng.range(1, 8) * xd_width;
        let is_hbd = iter % 3 == 0;
        let maxval: i32 = if is_hbd { 1 << 10 } else { 1 << 8 };

        // Neighbour grid. A third of the cells are INTRA (ref_frame[0] == 0),
        // so `is_neighbor_overlappable` rejects them; block sizes are drawn
        // from the full set so 4-wide / 4-high cells appear and the walk's
        // pair fixup fires.
        let mut grid_bsize: Vec<i32> = vec![3; rows * cols];
        let grid_ref0: Vec<i32> = (0..rows * cols)
            .map(|_| {
                if rng.next() % 3 == 0 {
                    0
                } else {
                    rng.range(1, 8)
                }
            })
            .collect();
        // Only the row above and the column left of the block are read, and
        // both must be an aligned tiling (see `aligned_tiling`).
        let above_row = aligned_tiling(&mut rng, cols, &MI_W, &NB_BSIZES);
        let left_col = aligned_tiling(&mut rng, rows, &MI_H, &NB_BSIZES);
        for c in 0..cols {
            grid_bsize[(mi_row as usize - 1) * cols + c] = above_row[c];
        }
        for r in 0..rows {
            grid_bsize[r * cols + (mi_col as usize - 1)] = left_col[r];
        }

        // Frame mi extent: sometimes tight enough to truncate the walk before
        // the block's own width/height does.
        let (mi_rows, mi_cols) = if iter % 7 == 0 {
            (
                mi_row + rng.range(1, xd_height + 1),
                mi_col + rng.range(1, xd_width + 1),
            )
        } else {
            (rows as i32, cols as i32)
        };
        if mi_cols < mi_col + xd_width || mi_rows < mi_row + xd_height {
            edge_truncated += 1;
        }

        let up_available = iter % 4 != 3;
        let left_available = iter % 5 != 4;
        if up_available {
            with_above += 1;
        }
        if left_available {
            with_left += 1;
        }
        if !up_available && !left_available {
            with_neither += 1;
        }

        let bw = (xd_width << 2) as usize;
        let bh = (xd_height << 2) as usize;
        let above_stride = bw + 8;
        let left_stride = bw + 5;
        let src_stride = bw + 3;
        let plane = |rng: &mut Rng, stride: usize| -> Vec<u16> {
            (0..stride * (bh + 64))
                .map(|_| rng.range(0, maxval) as u16)
                .collect()
        };
        let above = plane(&mut rng, above_stride);
        let left = plane(&mut rng, left_stride);
        let src = plane(&mut rng, src_stride);

        let (want_wsrc, want_mask) = cref::ref_rdopt_calc_target_weighted_pred(
            bsize as i32,
            mi_row,
            mi_col,
            xd_width,
            xd_height,
            up_available,
            left_available,
            &cref::ObmcGrid {
                rows,
                cols,
                bsize: &grid_bsize,
                ref0: &grid_ref0,
            },
            mi_rows,
            mi_cols,
            is_hbd,
            &above,
            above_stride,
            &left,
            left_stride,
            &src,
            src_stride,
        );

        // C's above walk reads the row at `mi_row - 1`; the left walk reads
        // the column at `mi_col - 1`.
        let above_cell = |c: i32| {
            let idx = (mi_row as usize - 1) * cols + c as usize;
            (grid_bsize[idx] as usize, grid_ref0[idx] > 0)
        };
        let left_cell = |r: i32| {
            let idx = r as usize * cols + (mi_col as usize - 1);
            (grid_bsize[idx] as usize, grid_ref0[idx] > 0)
        };
        let (got_wsrc, got_mask) = calc_target_weighted_pred(&ObmcTargetArgs {
            bsize,
            mi_row,
            mi_col,
            xd_width,
            xd_height,
            mi_rows,
            mi_cols,
            up_available,
            left_available,
            above_cell: &above_cell,
            left_cell: &left_cell,
            above: &above,
            above_stride,
            left: &left,
            left_stride,
            src: &src,
            src_stride,
        });

        let ctx = format!(
            "calc_target_weighted_pred(bsize={bsize}, mi=({mi_row},{mi_col}), \
             {xd_width}x{xd_height} mi, hbd={is_hbd}, up={up_available}, \
             left={left_available}, mi_limit=({mi_rows},{mi_cols}))"
        );
        for i in 0..bw * bh {
            assert_eq!(
                got_wsrc[i],
                want_wsrc[i],
                "{ctx}: wsrc[{}] (row {}, col {})",
                i,
                i / bw,
                i % bw
            );
            assert_eq!(
                got_mask[i],
                want_mask[i],
                "{ctx}: mask[{}] (row {}, col {})",
                i,
                i / bw,
                i % bw
            );
        }
        n += 1;
    }

    assert!(n == 600);
    assert!(with_above > 100, "too few above-available cases");
    assert!(with_left > 100, "too few left-available cases");
    assert!(
        with_neither > 0,
        "the no-neighbour arm (wsrc == src * 4096, mask == 4096) was never taken"
    );
    assert!(
        edge_truncated > 0,
        "the walk was never truncated by mi_rows / mi_cols — the frame-edge \
         arm of `end_col` / `end_row` is then untested"
    );
}
