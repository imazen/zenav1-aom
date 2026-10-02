//! libaom v3.15 `do_border_pad`: the visible-extent clip and the residual fill
//! outside the real frame (`av1/encoder/encodemb.c`, `rdopt_utils.h`,
//! `encoder.h`).
//!
//! `do_border_pad` is set per frame in `av1_encode` (`encoder.c`) only for
//! `mode == GOOD && deltaq_mode == DELTA_Q_OBJECTIVE && enable_tpl_model &&
//! aq_mode == NO_AQ && !seg.enabled && !roi.enabled && !sb_qp_sweep &&
//! !use_ducky_encode && sharpness != 3`. **`encode_key_frame` refuses any usage
//! other than ALLINTRA (`key_frame.rs`, `usage: only AOM_USAGE_ALL_INTRA (2) is
//! gated`), so no encode this port can produce reaches the `true` arm** — this
//! module is the kernel layer for when GOOD-usage lands, gated now against the
//! real exported `av1_subtract_block` so it is not carried unmeasured.
//!
//! | Rust | C |
//! |---|---|
//! | [`set_pixels_to_frame_edge`] | `set_pixels_to_frame_edge` (`encoder.h:4290`) |
//! | [`get_visible_dimensions`] | `get_visible_dimensions` (`rdopt_utils.h:362`) |
//! | [`fill_residue_outside_frame`] | `fill_residue_outside_frame` (`encodemb.c:80`) |
//!
//! Gate: `tests/all/border_pad_diff.rs`.

/// `IDTX` (`enums.h`): the 2-D identity transform. Every type up to and including
/// it is a 2-D (non-1-D) type.
pub const IDTX: usize = 9;

/// `htx_tab[tx_type] == IDTX_1D` (`common_data.h:155`): the HORIZONTAL transform
/// is the identity — the 1-D `V_*` types.
const fn htx_is_identity(tx_type: usize) -> bool {
    matches!(tx_type, 9 | 10 | 12 | 14)
}

/// `set_pixels_to_frame_edge` (`encoder.h:4290`): the signed distances, in
/// pixels, from the bottom and right edges of the prediction block to the
/// frame's, as `(pix_to_bottom_edge, pix_to_right_edge)`. With
/// `do_border_pad == false` the frame is taken as its mi-aligned extent
/// (`mi_cols << 2`), which makes them `mb_to_*_edge / 8`.
#[allow(clippy::too_many_arguments)]
pub fn set_pixels_to_frame_edge(
    bw_mi: i32,
    bh_mi: i32,
    mi_col: i32,
    mi_row: i32,
    mi_cols: i32,
    mi_rows: i32,
    frame_width: i32,
    frame_height: i32,
    do_border_pad: bool,
) -> (i32, i32) {
    let boundary_w = if do_border_pad { frame_width } else { mi_cols << 2 };
    let boundary_h = if do_border_pad { frame_height } else { mi_rows << 2 };
    (
        boundary_h - ((mi_row + bh_mi) << 2),
        boundary_w - ((mi_col + bw_mi) << 2),
    )
}

/// `ROUND_POWER_OF_TWO(v, n)` for `v >= 0`.
const fn round_pow2(v: i32, n: u32) -> i32 {
    (v + ((1i32 << n) >> 1)) >> n
}

/// `get_visible_dimensions` (`rdopt_utils.h:362`): the transform block's extent
/// clipped to the frame, as `(visible_cols, visible_rows, clipped)`.
///
/// `blk_col` / `blk_row` are in 4-pel units; `cols` / `rows` are the transform
/// block's pixel dimensions; `plane_bw` / `plane_bh` the PLANE block's. With
/// `clip_dims == false` the block is returned whole, which is what every caller
/// that has not opted in sees.
#[allow(clippy::too_many_arguments)]
pub fn get_visible_dimensions(
    pix_to_bottom_edge: i32,
    pix_to_right_edge: i32,
    ss_x: u32,
    ss_y: u32,
    plane_bw: usize,
    plane_bh: usize,
    blk_col: usize,
    blk_row: usize,
    cols: usize,
    rows: usize,
    clip_dims: bool,
) -> (usize, usize, bool) {
    if (pix_to_bottom_edge >= 0 && pix_to_right_edge >= 0) || !clip_dims {
        return (cols, rows, false);
    }
    let valid_rows = if pix_to_bottom_edge >= 0 {
        rows as i32
    } else {
        let block_rows = -round_pow2(-pix_to_bottom_edge, ss_y) + plane_bh as i32;
        (block_rows - ((blk_row as i32) << 2)).clamp(0, rows as i32)
    };
    let valid_cols = if pix_to_right_edge >= 0 {
        cols as i32
    } else {
        let block_cols = -round_pow2(-pix_to_right_edge, ss_x) + plane_bw as i32;
        (block_cols - ((blk_col as i32) << 2)).clamp(0, cols as i32)
    };
    let (vc, vr) = (valid_cols as usize, valid_rows as usize);
    (vc, vr, vc < cols || vr < rows)
}

/// `DIVIDE_AND_ROUND_SIGNED(n, d)` (`mem.h:76`), `d > 0`; C's `/` truncates.
const fn div_round_signed(n: i32, d: i32) -> i32 {
    if n < 0 { (n - d / 2) / d } else { (n + d / 2) / d }
}

/// `avg_wxh_block_c`: the rounded mean of the `w x h` block, `0` when empty.
fn avg_wxh(diff: &[i16], stride: usize, w: usize, h: usize) -> i16 {
    let mut sum = 0i32;
    for r in 0..h {
        for c in 0..w {
            sum += i32::from(diff[r * stride + c]);
        }
    }
    if w * h > 0 { div_round_signed(sum, (w * h) as i32) as i16 } else { 0 }
}

/// `fill_residue_outside_frame` (`encodemb.c:80`): overwrite the part of a
/// transform block's residual that lies outside the real frame — zero for IDTX,
/// the in-frame mean for the other 2-D types, and per-row / per-column means or
/// zeros for the 1-D types — so the transform of the padded block does not spend
/// bits on pixels that are not coded. `tx_type` is a `TX_TYPE` index (0..=15).
pub fn fill_residue_outside_frame(
    diff: &mut [i16],
    stride: usize,
    tx_cols: usize,
    tx_rows: usize,
    visible_cols: usize,
    visible_rows: usize,
    tx_type: usize,
) {
    let complete_block_outside = visible_cols == 0 || visible_rows == 0;
    let right_pixels = tx_cols - visible_cols;
    if tx_type <= IDTX {
        let mut avg = 0i16;
        if tx_type != IDTX && !complete_block_outside {
            avg = avg_wxh(diff, stride, visible_cols, visible_rows);
        }
        for i in 0..tx_rows {
            let row = i * stride;
            diff[row + visible_cols..row + visible_cols + right_pixels].fill(avg);
        }
        for i in visible_rows..tx_rows {
            let row = i * stride;
            diff[row..row + visible_cols].fill(avg);
        }
    } else if htx_is_identity(tx_type) {
        if visible_rows < tx_rows {
            let mut out = [0i16; 64];
            if !complete_block_outside {
                for (col, o) in out.iter_mut().enumerate().take(visible_cols) {
                    let mut sum = 0i32;
                    for r in 0..visible_rows {
                        sum += i32::from(diff[r * stride + col]);
                    }
                    *o = if visible_rows > 0 {
                        div_round_signed(sum, visible_rows as i32) as i16
                    } else {
                        0
                    };
                }
            }
            for j in 0..visible_cols {
                for i in visible_rows..tx_rows {
                    diff[i * stride + j] = out[j];
                }
            }
        }
        if right_pixels > 0 {
            for i in 0..tx_rows {
                let row = i * stride;
                diff[row + visible_cols..row + visible_cols + right_pixels].fill(0);
            }
        }
    } else {
        // `vtx_tab[tx_type] == IDTX_1D`: the 1-D `H_*` types.
        if right_pixels > 0 {
            let mut out = [0i16; 64];
            if !complete_block_outside {
                for (row, o) in out.iter_mut().enumerate().take(visible_rows) {
                    let mut sum = 0i32;
                    for c in 0..visible_cols {
                        sum += i32::from(diff[row * stride + c]);
                    }
                    *o = if visible_cols > 0 {
                        div_round_signed(sum, visible_cols as i32) as i16
                    } else {
                        0
                    };
                }
            }
            for i in 0..visible_rows {
                let row = i * stride;
                diff[row + visible_cols..row + visible_cols + right_pixels].fill(out[i]);
            }
        }
        for i in visible_rows..tx_rows {
            let row = i * stride;
            diff[row..row + tx_cols].fill(0);
        }
    }
}
