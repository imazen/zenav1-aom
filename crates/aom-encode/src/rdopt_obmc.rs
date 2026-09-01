//! The OBMC TARGET (`calc_target_weighted_pred`, `av1/encoder/rdopt.c:6888`
//! and its two per-neighbour visitors at `:6752` / `:6800`).
//!
//! OBMC blends a block's own prediction with its above and left neighbours'.
//! Rather than re-blending for every candidate MV, libaom precomputes a
//! WEIGHTED SOURCE and a MASK once per block, after which any predictor `P`'s
//! OBMC error is
//!
//! ```text
//! error(x, y) = wsrc(x, y) - mask(x, y) * P(x, y) / 64^2
//! ```
//!
//! which is what `aom_obmc_variance` / `aom_obmc_sub_pixel_variance` (already
//! ported and gated in `aom-dsp/tests/obmc_dist_diff.rs`) consume. This module
//! produces their two inputs, so it is the missing half of the OBMC RD path.
//!
//! Tier 1c (all three functions are `static`; the oracle is libaom's own
//! rdopt.c compiled into the shim archive). Gate:
//! `crates/aom-encode/tests/rdopt_obmc_diff.rs`.
//!
//! # Translation notes
//!
//! - **One neighbour walk, not two.** C's `foreach_overlappable_nb_above` and
//!   `_left` (`av1/common/obmc.h:25` / `:56`) are the same walk over
//!   transposed axes; [`overlappable_neighbours`] takes the direction. The
//!   4-wide fixup — a block one mi wide is half of a chroma pair, so the walk
//!   steps back to the pair start, reads the SECOND cell, and steps by two —
//!   therefore exists once instead of twice.
//! - **The visitors write disjoint regions, so they are plain loops over
//!   subslices** rather than the raw pointer walks C uses. `wsrc` and `mask`
//!   are `bw * bh` i32 buffers with row stride `bw` throughout.
//! - **Integer widths are C's.** `wsrc` and `mask` are `int32_t`; the
//!   intermediate `(tmp[col] << 6) * m1` is computed in `i32` and can be large
//!   (12-bit sample, `<< 6`, times 64 ⇒ about 2^24), so no widening is needed
//!   and none is done.

use aom_dsp::inter::get_obmc_mask;

/// `MI_SIZE` (`enums.h`).
pub const MI_SIZE: i32 = 4;
/// `MI_SIZE_LOG2`.
pub const MI_SIZE_LOG2: i32 = 2;
/// `AOM_BLEND_A64_MAX_ALPHA` (`aom_dsp/blend.h`).
pub const AOM_BLEND_A64_MAX_ALPHA: i32 = 64;
/// `AOM_BLEND_A64_ROUND_BITS`.
pub const AOM_BLEND_A64_ROUND_BITS: i32 = 6;

/// `block_size_wide[BLOCK_SIZES_ALL]` (`common_data.h`), in pixels.
pub const BLOCK_SIZE_WIDE: [i32; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
/// `block_size_high[BLOCK_SIZES_ALL]`, in pixels.
pub const BLOCK_SIZE_HIGH: [i32; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
/// `mi_size_wide[BLOCK_SIZES_ALL]`, in 4x4 units.
pub const MI_SIZE_WIDE: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
/// `mi_size_high[BLOCK_SIZES_ALL]`.
pub const MI_SIZE_HIGH: [i32; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
/// `mi_size_wide_log2[BLOCK_SIZES_ALL]`.
pub const MI_SIZE_WIDE_LOG2: [usize; 22] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4,
];
/// `mi_size_high_log2[BLOCK_SIZES_ALL]`.
pub const MI_SIZE_HIGH_LOG2: [usize; 22] = [
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2,
];
/// `max_neighbor_obmc[6]` (`blockd.h:1471`): how many overlappable neighbours
/// a block of a given mi-log2 size will blend with.
pub const MAX_NEIGHBOR_OBMC: [i32; 6] = [0, 1, 2, 3, 4, 4];
/// `BLOCK_64X64`.
pub const BLOCK_64X64: usize = 12;

/// The direction of an OBMC neighbour walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObmcDir {
    /// The mi row above the block.
    Above,
    /// The mi column left of the block.
    Left,
}

/// One overlappable neighbour the walk found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObmcNeighbour {
    /// C's `rel_mi_col` (above) or `rel_mi_row` (left) — the offset from the
    /// block's own origin, in mi units.
    pub rel: i32,
    /// C's `op_mi_size` — how many mi units of the block this neighbour
    /// overlaps.
    pub op_mi_size: i32,
}

/// `foreach_overlappable_nb_above` (`obmc.h:25`) / `_left` (`:56`), as an
/// enumeration instead of a callback.
///
/// `cell(pos)` returns `(bsize, is_inter)` for the mi at position `pos` along
/// the neighbouring row/column. `extent` is `xd->width` (above) or
/// `xd->height` (left); `mi_limit` is the frame's mi count along that axis.
///
/// Note the 4-wide fixup mutates C's loop variable, so a 1-mi-wide neighbour
/// at an odd position is reported at the EVEN position of its pair while the
/// cell that decides overlappability is the odd one. That asymmetry is
/// reproduced.
pub fn overlappable_neighbours(
    dir: ObmcDir,
    mi_start: i32,
    extent: i32,
    mi_limit: i32,
    nb_max: i32,
    available: bool,
    cell: impl Fn(i32) -> (usize, bool),
) -> Vec<ObmcNeighbour> {
    let mut out = Vec::new();
    if !available {
        return out;
    }
    let size_tbl = match dir {
        ObmcDir::Above => &MI_SIZE_WIDE,
        ObmcDir::Left => &MI_SIZE_HIGH,
    };
    let max_step = size_tbl[BLOCK_64X64];
    let end = (mi_start + extent).min(mi_limit);
    let mut pos = mi_start;
    while pos < end && (out.len() as i32) < nb_max {
        let (bsize, _) = cell(pos);
        let mut step = size_tbl[bsize].min(max_step);
        let mut read_at = pos;
        if step == 1 {
            // A 4-wide/4-high block is half of a pair whose SECOND cell owns
            // the chroma information; step back to the pair and read that one.
            pos &= !1;
            read_at = pos + 1;
            step = 2;
        }
        let (_, is_inter) = cell(read_at);
        if is_inter {
            out.push(ObmcNeighbour {
                rel: pos - mi_start,
                op_mi_size: extent.min(step),
            });
        }
        pos += step;
    }
    out
}

/// Geometry and neighbour availability for [`calc_target_weighted_pred`].
pub struct ObmcTargetArgs<'a> {
    /// `xd->mi[0]->bsize`.
    pub bsize: usize,
    /// `xd->mi_row` / `xd->mi_col`.
    pub mi_row: i32,
    /// See [`Self::mi_row`].
    pub mi_col: i32,
    /// `xd->width` / `xd->height`, in mi units.
    pub xd_width: i32,
    /// See [`Self::xd_width`].
    pub xd_height: i32,
    /// `cm->mi_params.mi_rows` / `mi_cols`.
    pub mi_rows: i32,
    /// See [`Self::mi_rows`].
    pub mi_cols: i32,
    /// `xd->up_available` / `xd->left_available`.
    pub up_available: bool,
    /// See [`Self::up_available`].
    pub left_available: bool,
    /// `(bsize, is_inter)` of the mi at column `c` of the row ABOVE the block.
    pub above_cell: &'a dyn Fn(i32) -> (usize, bool),
    /// `(bsize, is_inter)` of the mi at row `r` of the column LEFT of it.
    pub left_cell: &'a dyn Fn(i32) -> (usize, bool),
    /// The above neighbours' prediction of this block, `above_stride` per row.
    pub above: &'a [u16],
    /// See [`Self::above`].
    pub above_stride: usize,
    /// The left neighbours' prediction of this block.
    pub left: &'a [u16],
    /// See [`Self::left`].
    pub left_stride: usize,
    /// The block's own source pixels.
    pub src: &'a [u16],
    /// See [`Self::src`].
    pub src_stride: usize,
}

/// `calc_target_weighted_pred_above` (rdopt.c:6752): the above neighbour's
/// contribution.
///
/// Writes the top `overlap` rows only, and OVERWRITES rather than accumulates
/// — this always runs before the left pass.
fn apply_above(
    wsrc: &mut [i32],
    mask: &mut [i32],
    bw: usize,
    nb: ObmcNeighbour,
    overlap: usize,
    tmp: &[u16],
    tmp_stride: usize,
) {
    let mask1d = get_obmc_mask(overlap);
    let col0 = (nb.rel * MI_SIZE) as usize;
    let width = (nb.op_mi_size * MI_SIZE) as usize;
    for (row, &m) in mask1d.iter().enumerate().take(overlap) {
        let m0 = i32::from(m);
        let m1 = AOM_BLEND_A64_MAX_ALPHA - m0;
        let dst = row * bw + col0;
        let srcrow = row * tmp_stride + col0;
        for col in 0..width {
            wsrc[dst + col] = m1 * i32::from(tmp[srcrow + col]);
            mask[dst + col] = m0;
        }
    }
}

/// `calc_target_weighted_pred_left` (rdopt.c:6800): the left neighbour's
/// contribution.
///
/// Unlike the above pass this ACCUMULATES: it reads `wsrc`/`mask` back,
/// right-shifts by [`AOM_BLEND_A64_ROUND_BITS`] and re-blends. That asymmetry
/// is why the two passes cannot share a body, and why the `* 64` scaling
/// between them (in [`calc_target_weighted_pred`]) is load-bearing.
fn apply_left(
    wsrc: &mut [i32],
    mask: &mut [i32],
    bw: usize,
    nb: ObmcNeighbour,
    overlap: usize,
    tmp: &[u16],
    tmp_stride: usize,
) {
    let mask1d = get_obmc_mask(overlap);
    let row0 = (nb.rel * MI_SIZE) as usize;
    let height = (nb.op_mi_size * MI_SIZE) as usize;
    for row in 0..height {
        let dst = (row0 + row) * bw;
        let srcrow = (row0 + row) * tmp_stride;
        for col in 0..overlap {
            let m0 = i32::from(mask1d[col]);
            let m1 = AOM_BLEND_A64_MAX_ALPHA - m0;
            wsrc[dst + col] = (wsrc[dst + col] >> AOM_BLEND_A64_ROUND_BITS) * m0
                + (i32::from(tmp[srcrow + col]) << AOM_BLEND_A64_ROUND_BITS) * m1;
            mask[dst + col] = (mask[dst + col] >> AOM_BLEND_A64_ROUND_BITS) * m0;
        }
    }
}

/// `calc_target_weighted_pred` (rdopt.c:6888): build the OBMC weighted source
/// and mask for one block.
///
/// Returns `(wsrc, mask)`, each `bw * bh` with row stride `bw`, where
/// `bw = xd_width * 4` and `bh = xd_height * 4`.
///
/// The order is load-bearing: the above pass overwrites the top rows, then
/// EVERYTHING is scaled by 64, then the left pass re-blends its columns, then
/// the source is subtracted at `64 * 64`. Moving the scaling changes every
/// value in the overlap corner.
pub fn calc_target_weighted_pred(args: &ObmcTargetArgs<'_>) -> (Vec<i32>, Vec<i32>) {
    // OBMC is only allowed for blocks at least 8x8
    // (`is_motion_variation_allowed_bsize`, blockd.h:1460), which is what
    // makes `mi_row` / `mi_col` EVEN. That matters: the walk's 4-wide fixup
    // does `pos &= !1`, and from an odd start that lands BELOW `mi_start`,
    // giving a negative `rel` — at which point C's
    // `wsrc + rel_mi_col * MI_SIZE` writes before the buffer. Asserting the
    // precondition is how the port declines to reproduce that.
    debug_assert!(
        BLOCK_SIZE_WIDE[args.bsize].min(BLOCK_SIZE_HIGH[args.bsize]) >= 8,
        "OBMC is not allowed below 8x8 (is_motion_variation_allowed_bsize)"
    );
    debug_assert!(
        args.mi_row % 2 == 0 && args.mi_col % 2 == 0,
        "an OBMC block is at least 8x8, so its mi origin is even; an odd one \
         makes C's 4-wide pair fixup produce a negative rel_mi_col"
    );
    let bw = (args.xd_width << MI_SIZE_LOG2) as usize;
    let bh = (args.xd_height << MI_SIZE_LOG2) as usize;
    let mut wsrc = vec![0i32; bw * bh];
    let mut mask = vec![AOM_BLEND_A64_MAX_ALPHA; bw * bh];

    if args.up_available {
        let overlap = (BLOCK_SIZE_HIGH[args.bsize].min(BLOCK_SIZE_HIGH[BLOCK_64X64]) >> 1) as usize;
        let nb_max = MAX_NEIGHBOR_OBMC[MI_SIZE_WIDE_LOG2[args.bsize]];
        for nb in overlappable_neighbours(
            ObmcDir::Above,
            args.mi_col,
            args.xd_width,
            args.mi_cols,
            nb_max,
            true,
            args.above_cell,
        ) {
            apply_above(
                &mut wsrc,
                &mut mask,
                bw,
                nb,
                overlap,
                args.above,
                args.above_stride,
            );
        }
    }

    for v in wsrc.iter_mut() {
        *v *= AOM_BLEND_A64_MAX_ALPHA;
    }
    for v in mask.iter_mut() {
        *v *= AOM_BLEND_A64_MAX_ALPHA;
    }

    if args.left_available {
        let overlap = (BLOCK_SIZE_WIDE[args.bsize].min(BLOCK_SIZE_WIDE[BLOCK_64X64]) >> 1) as usize;
        let nb_max = MAX_NEIGHBOR_OBMC[MI_SIZE_HIGH_LOG2[args.bsize]];
        for nb in overlappable_neighbours(
            ObmcDir::Left,
            args.mi_row,
            args.xd_height,
            args.mi_rows,
            nb_max,
            true,
            args.left_cell,
        ) {
            apply_left(
                &mut wsrc,
                &mut mask,
                bw,
                nb,
                overlap,
                args.left,
                args.left_stride,
            );
        }
    }

    // `src_scale` is 64 * 64: wsrc is at that scale after the two passes.
    let src_scale = AOM_BLEND_A64_MAX_ALPHA * AOM_BLEND_A64_MAX_ALPHA;
    for row in 0..bh {
        let d = row * bw;
        let s = row * args.src_stride;
        for col in 0..bw {
            wsrc[d + col] = i32::from(args.src[s + col]) * src_scale - wsrc[d + col];
        }
    }
    (wsrc, mask)
}
