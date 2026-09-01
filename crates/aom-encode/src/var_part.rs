//! `av1_choose_var_based_partitioning` (av1/encoder/var_based_part.c) — the
//! **KEY-frame (allintra) arm** of the variance-based partitioner that
//! `partition_search_type == VAR_BASED_PARTITION` (allintra speed >= 7)
//! switches the superblock encoder to (`encode_rd_sb`, encodeframe.c:876-895):
//! the partition tree is FIXED up front from source-variance thresholds (no
//! RD partition search), then `av1_rd_use_partition` walks the fixed tree
//! running the normal RD mode search per block
//! ([`crate::partition_pick::rd_use_partition_real`]).
//!
//! Scope (matching the port's Gate-2 envelope): **KEY frames only** —
//! `is_key_frame` is true, so the reference/motion machinery (setup_planes,
//! chroma_check color-sensitivity, y_sad, zeromv-skip, low-temp-var flags,
//! the 8x8-avg source-vs-ref leaf fill) is structurally unreachable and NOT
//! ported. What is ported, line-faithful:
//!
//! 1. **Thresholds** (`set_vbp_thresholds` KEY arm, var_based_part.c:654-673
//!    → `set_vbp_thresholds_key_frame` :535-560): `threshold_base = 120 *
//!    av1_ac_quant_QTX(qindex, 0, bit_depth)`; `[0]=[1]=base`; `<720p:
//!    [2]=base/3, [3]=base>>1` else `[2]=[3]=base>>2`; `[4]=base<<2`.
//!    `rt_sf.force_large_partition_blocks_intra` (which is what consumes
//!    `var_part_split_threshold_shift`) rises at allintra speed>=8 AND
//!    720p+ (speed_features.c:326-328) and in RT (:1647). **KB-32 root #1:
//!    this module previously asserted it "is 0 on this path" and dropped
//!    both of its arms** — true only of an envelope that stopped at 640 px,
//!    which is where every gate sat when that comment was written. It is now
//!    carried as [`VbpSf`] from the resolved frame-level speed features.
//!    The frame-level `av1_set_variance_partition_thresholds`
//!    copy + `threshold_minmax` are likewise dead here: the per-SB
//!    `set_vbp_thresholds` call fully overwrites all five local thresholds
//!    for key frames, and `threshold_minmax` feeds only the
//!    `compute_minmax_variance` arm which is hardcoded 0 (:1114).
//! 2. **The 4x4-downsampled variance tree fill**
//!    (`fill_variance_tree_leaves` KEY arm :1156-1167 → `fill_variance_4x4avg`
//!    :390-423): per 4x4 sub-block, `sum = aom_avg_4x4(src) - 128`,
//!    `sse = sum*sum` (dst is the implicit flat-128 "prediction");
//!    out-of-frame 4x4s (top-left at/past the crop) contribute zeros.
//!    `border_offset_4x4` stays 0 (temporal filtering never runs on key
//!    frames, :1135).
//! 3. **The force-split stage-2 walk** (:1788-1894, key arms): 16x16 nodes
//!    with `variance > thresholds[3]` force PARTITION_SPLIT up the tree
//!    (16→32→64→root); 32x32 nodes with `variance > thresholds[2]` force
//!    32→64→root. The 64x64/128x128 levels have no key-frame forcing rules;
//!    a 64x64 SB (`is_small_sb`) always forces the (structural) root split.
//! 4. **The partition assignment** (`set_vt_partitioning` :149-253 + the
//!    :1896-1942 descent): key frames take the split for `bsize >
//!    BLOCK_32X32` or `variance > (threshold << 4)`; NONE when the block
//!    fits and `variance < threshold`; otherwise the VERT/HORZ pair checks
//!    (both halves' variance under threshold + a valid chroma plane size);
//!    else descend, bottoming out at four BLOCK_8X8 leaves per 16x16. The
//!    result is written as `bsize` stamps at each leaf's top-left mi cell
//!    (`set_block_size` :136-147), the exact structure C's
//!    `get_partition` (av1_common_int.h:1775) reads back —
//!    [`get_partition_from_stamps`] here.
//!
//! Differential status: `av1_choose_var_based_partitioning` is not exported
//! from the reference build, so per the evidence hierarchy the tree logic is
//! validated transcription + the end-to-end byte gates (any partition
//! difference desyncs the bitstream immediately); the one arithmetic kernel,
//! [`avg_4x4`], is differentially locked against the REAL exported
//! `aom_avg_4x4_c` (`avg_4x4_diff.rs`).

use aom_dsp::entropy::partition::{get_partition_subsize, get_plane_block_size};
use aom_dsp::quant::av1_ac_quant_qtx;

use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B};

const BLOCK_8X8: usize = 3;
const BLOCK_16X16: usize = 6;
const BLOCK_16X32: usize = 7;
const BLOCK_32X16: usize = 8;
const BLOCK_32X32: usize = 9;
const BLOCK_32X64: usize = 10;
const BLOCK_64X32: usize = 11;
const BLOCK_64X64: usize = 12;
const BLOCK_64X128: usize = 13;
const BLOCK_128X64: usize = 14;
const BLOCK_128X128: usize = 15;
const BLOCK_INVALID: usize = 255;

/// `PART_EVAL_STATUS` (var_based_part.c:38-45).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PartEval {
    /// Evaluate all partition types.
    All,
    /// Force PARTITION_SPLIT.
    OnlySplit,
    /// Force PARTITION_NONE.
    OnlyNone,
}

/// `VPartVar` (encoder/block.h): running (sse, sum, log2_count) for one
/// variance-tree node + the derived `variance`.
#[derive(Clone, Copy, Default, Debug)]
struct VPartVar {
    sum_square_error: u32,
    sum_error: i32,
    log2_count: i32,
    variance: i32,
}

/// `VPVariance`: the none/horz/vert partition variances of one node.
#[derive(Clone, Copy, Default, Debug)]
struct VPVariance {
    none: VPartVar,
    horz: [VPartVar; 2],
    vert: [VPartVar; 2],
}

/// `VP8x8` (KEY usage): four 4x4 leaf records + the 8x8 sums. C's `VP4x4`
/// children carry a full `VPVariance` each, but on the key path only their
/// `.none` totals are ever written/read (`fill_variance_4x4avg` /
/// `tree_to_node(BLOCK_8X8)`), so the leaf is stored as a bare [`VPartVar`].
#[derive(Clone, Copy, Default, Debug)]
struct Vp8x8 {
    part_variances: VPVariance,
    split: [VPartVar; 4],
}

#[derive(Clone, Copy, Default, Debug)]
struct Vp16x16 {
    part_variances: VPVariance,
    split: [Vp8x8; 4],
}

#[derive(Clone, Copy, Default, Debug)]
struct Vp32x32 {
    part_variances: VPVariance,
    split: [Vp16x16; 4],
}

#[derive(Clone, Copy, Default, Debug)]
struct Vp64x64 {
    part_variances: VPVariance,
    split: [Vp32x32; 4],
}

/// `VP128x128` — the full per-SB tree (`vt->split` = `td->vt64x64`).
#[derive(Clone, Default, Debug)]
struct Vp128x128 {
    part_variances: VPVariance,
    split: [Vp64x64; 4],
}

/// `GET_BLK_IDX_X(idx, level)` (var_based_part.h:24).
#[inline]
fn blk_idx_x(idx: usize, level: usize) -> usize {
    (idx & 1) << level
}

/// `GET_BLK_IDX_Y(idx, level)` (var_based_part.h:25).
#[inline]
fn blk_idx_y(idx: usize, level: usize) -> usize {
    (idx >> 1) << level
}

/// `aom_avg_4x4_c` (aom_dsp/avg.c:32) / `aom_highbd_avg_4x4_c` (:74) —
/// identical arithmetic on this port's u16 pixel buffers (`(sum + 8) >> 4`;
/// a 4x4 of 12-bit samples sums to <= 65520, no overflow). Differentially
/// locked vs the REAL exported `aom_avg_4x4_c` in `avg_4x4_diff.rs`.
#[inline]
pub fn avg_4x4(src: &[u16], off: usize, stride: usize) -> i32 {
    let mut sum = 0u32;
    for r in 0..4 {
        let row = &src[off + r * stride..off + r * stride + 4];
        for &p in row {
            sum += u32::from(p);
        }
    }
    ((sum + 8) >> 4) as i32
}

/// `fill_variance` (var_based_part.c:103).
#[inline]
fn fill_variance(s2: u32, s: i32, c: i32, v: &mut VPartVar) {
    v.sum_square_error = s2;
    v.sum_error = s;
    v.log2_count = c;
}

/// `get_variance` (var_based_part.c:109) — the C expression verbatim,
/// including the u32 wrap of the `(sum*sum)>>log2` subtraction and the
/// int truncation of the `256 * (...) >> log2` scale.
#[inline]
fn get_variance(v: &mut VPartVar) {
    let sum_sq = (i64::from(v.sum_error) * i64::from(v.sum_error)) >> v.log2_count;
    let diff = v.sum_square_error.wrapping_sub(sum_sq as u32);
    // C: (int)(256 * diff >> log2_count) — `256 * (uint32)diff` is a u32
    // multiply in C (both operands int-promoted; diff is unsigned so the
    // arithmetic wraps mod 2^32), then >> log2_count, then (int) cast.
    v.variance = (256u32.wrapping_mul(diff) >> v.log2_count) as i32;
}

/// `sum_2_variances` (var_based_part.c:117).
#[inline]
fn sum_2_variances(a: &VPartVar, b: &VPartVar, r: &mut VPartVar) {
    debug_assert_eq!(a.log2_count, b.log2_count);
    fill_variance(
        a.sum_square_error + b.sum_square_error,
        a.sum_error + b.sum_error,
        a.log2_count + 1,
        r,
    );
}

/// `fill_variance_tree` (var_based_part.c:124) on one node: sums the four
/// children's `none` totals into the node's horz/vert/none partitions.
fn fill_variance_node(children: &[VPartVar; 4], pv: &mut VPVariance) {
    let mut horz0 = VPartVar::default();
    let mut horz1 = VPartVar::default();
    let mut vert0 = VPartVar::default();
    let mut vert1 = VPartVar::default();
    sum_2_variances(&children[0], &children[1], &mut horz0);
    sum_2_variances(&children[2], &children[3], &mut horz1);
    sum_2_variances(&children[0], &children[2], &mut vert0);
    sum_2_variances(&children[1], &children[3], &mut vert1);
    pv.horz = [horz0, horz1];
    pv.vert = [vert0, vert1];
    let (v0, v1) = (pv.vert[0], pv.vert[1]);
    sum_2_variances(&v0, &v1, &mut pv.none);
}

#[inline]
fn child_nones(pvs: [&VPVariance; 4]) -> [VPartVar; 4] {
    [pvs[0].none, pvs[1].none, pvs[2].none, pvs[3].none]
}

/// Frame/tile geometry + quantizer inputs for one SB's variance partitioning.
#[derive(Clone, Copy, Debug)]
pub struct VbpFrame {
    /// `cm->mi_params.mi_rows` / `mi_cols`.
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// `tile->mi_row_end` / `mi_col_end` (== mi_rows/mi_cols single-tile).
    pub tile_mi_row_end: i32,
    pub tile_mi_col_end: i32,
    /// `cm->width * cm->height` (the crop pixel count, for the resolution
    /// threshold arms).
    pub num_pixels: i64,
    /// `cm->seq_params->sb_size` (BLOCK_64X64 = 12 or BLOCK_128X128 = 15).
    pub sb_size: usize,
    /// The SB qindex (`base_qindex`; no delta-q/segments in this envelope —
    /// the :1683-1690 clamp chain resolves to `base_qindex`).
    pub qindex: i32,
    pub bit_depth: u8,
    /// Chroma subsampling as C's `xd->plane[AOM_PLANE_U]` carries it:
    /// monochrome frames get (1, 1) (`av1_setup_block_planes` sets planes
    /// >= num_planes to ss (1,1)).
    pub ss_x: usize,
    pub ss_y: usize,
    /// The resolved frame-level speed features `set_vbp_thresholds_key_frame`
    /// reads. See [`VbpSf`].
    pub sf: VbpSf,
}

/// The RESOLVED frame-level speed-feature inputs `set_vbp_thresholds_key_frame`
/// reads (var_based_part.c:535-560), carried down rather than re-derived
/// (playbook §13 / KB-26: anything reconstructed from a base constructor
/// silently drops every later pass).
///
/// `Default` is the pre-speed-8 state — `force_large_partition_blocks_intra`
/// off, which makes `var_part_split_threshold_shift` unread — so every caller
/// below speed 8 is byte-identical whether it fills this in or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VbpSf {
    /// `cpi->sf.rt_sf.force_large_partition_blocks_intra` (allintra
    /// `speed >= 8 && min(w, h) >= 720`, speed_features.c:326-328).
    pub force_large_partition_blocks_intra: bool,
    /// `cpi->sf.rt_sf.var_part_split_threshold_shift` (allintra: 7 at speed 7,
    /// **8 at speed 8**, 7 again at speed 9 — speed_features.c:574/581/601).
    /// Read only when `force_large_partition_blocks_intra` is set.
    pub var_part_split_threshold_shift: i32,
    /// `cpi->oxcf.mode == ALLINTRA` — selects both the `? 7 : 8` shift-steps
    /// base and the `? 1 : 0` `shift_val`.
    pub allintra: bool,
}

impl Default for VbpSf {
    fn default() -> Self {
        Self {
            force_large_partition_blocks_intra: false,
            // init_rt_sf:2589; unread while force_large is false.
            var_part_split_threshold_shift: 7,
            allintra: true,
        }
    }
}

/// `set_vbp_thresholds` (var_based_part.c:654) — KEY-frame arm only
/// (`set_vbp_thresholds_key_frame` :535). Returns `thresholds[5]`.
///
/// Both `force_large_partition_blocks_intra` arms are modelled (KB-32 root
/// #1): the `shift_steps` scaling of `threshold_base` (:539-544, LIVE at
/// speed 8 where the shift is 8 and therefore `shift_steps == 1`; a no-op at
/// speed 9 where the shift is back to 7) and the `shift_val` override in the
/// `num_pixels >= RESOLUTION_720P` arm (:552-554, LIVE at BOTH speeds).
pub fn set_vbp_thresholds_key(qindex: i32, bit_depth: u8, num_pixels: i64, sf: VbpSf) -> [i64; 5] {
    const RESOLUTION_720P: i64 = 1280 * 720;
    let ac_q = av1_ac_quant_qtx(qindex, 0, bit_depth);
    let mut threshold_base: i64 = 120i64 * i64::from(ac_q);
    if sf.force_large_partition_blocks_intra {
        // shift_steps = threshold_left_shift - (mode == ALLINTRA ? 7 : 8);
        // assert(shift_steps >= 0);  (var_based_part.c:540-543)
        let shift_steps = sf.var_part_split_threshold_shift - if sf.allintra { 7 } else { 8 };
        assert!(
            shift_steps >= 0,
            "var_part_split_threshold_shift {} is below the {} floor \
             set_vbp_thresholds_key_frame asserts (var_based_part.c:542)",
            sf.var_part_split_threshold_shift,
            if sf.allintra { 7 } else { 8 }
        );
        threshold_base <<= shift_steps;
    }
    let mut thresholds = [0i64; 5];
    thresholds[0] = threshold_base;
    thresholds[1] = threshold_base;
    if num_pixels < RESOLUTION_720P {
        thresholds[2] = threshold_base / 3;
        thresholds[3] = threshold_base >> 1;
    } else {
        let shift_val = if sf.force_large_partition_blocks_intra {
            i32::from(sf.allintra) // (mode == ALLINTRA ? 1 : 0)
        } else {
            2
        };
        thresholds[2] = threshold_base >> shift_val;
        thresholds[3] = threshold_base >> shift_val;
    }
    thresholds[4] = threshold_base << 2;
    thresholds
}

/// `set_block_size` (var_based_part.c:136): stamp `bsize` at the block's
/// top-left mi cell when it is inside the frame. `stamps` is the
/// frame-sized (mi_rows x mi_cols, row-major) bsize grid this SB's
/// assignment writes and [`get_partition_from_stamps`] reads.
#[inline]
fn set_block_size(stamps: &mut [u8], f: &VbpFrame, mi_row: i32, mi_col: i32, bsize: usize) {
    if f.mi_cols > mi_col && f.mi_rows > mi_row {
        stamps[(mi_row * f.mi_cols + mi_col) as usize] = bsize as u8;
    }
}

/// `get_plane_block_size` for the guard in `set_vt_partitioning` — the same
/// subsampled-lookup the partition search uses.
fn plane_block_size(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    get_plane_block_size(bsize, ss_x, ss_y)
}

/// One level's variance node view (`variance_node` / `tree_to_node`).
struct NodeView<'a> {
    part_variances: &'a mut VPVariance,
}

/// `set_vt_partitioning` (var_based_part.c:149) — KEY-frame arms live
/// (`frame_is_intra_only` true). Returns true when a partitioning was set
/// at this node (stop descending).
#[allow(clippy::too_many_arguments)]
fn set_vt_partitioning(
    stamps: &mut [u8],
    f: &VbpFrame,
    node: NodeView,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    threshold: i64,
    bsize_min: usize,
    force_split: PartEval,
) -> bool {
    let block_width = MI_SIZE_WIDE_B[bsize] as i32;
    let block_height = block_width; // square sizes only (C asserts this)
    let mut bs_width_check = block_width;
    let mut bs_height_check = block_height;
    let mut bs_width_vert_check = block_width >> 1;
    let mut bs_height_horiz_check = block_height >> 1;
    // "On the right and bottom boundary we only need to check if half the
    // bsize fits, because boundary is extended up to 64. So do this check
    // only for sb_size = 64X64." (:162-174)
    if f.sb_size == BLOCK_64X64 {
        if f.tile_mi_col_end == f.mi_cols {
            bs_width_check = (block_width >> 1) + 1;
            bs_width_vert_check = (block_width >> 2) + 1;
        }
        if f.tile_mi_row_end == f.mi_rows {
            bs_height_check = (block_height >> 1) + 1;
            bs_height_horiz_check = (block_height >> 2) + 1;
        }
    }

    if mi_col + bs_width_check <= f.tile_mi_col_end
        && mi_row + bs_height_check <= f.tile_mi_row_end
        && force_split == PartEval::OnlyNone
    {
        set_block_size(stamps, f, mi_row, mi_col, bsize);
        return true;
    }
    if force_split == PartEval::OnlySplit {
        return false;
    }

    if bsize == bsize_min {
        // (Structurally dead in this file's call graph — every call site
        // passes bsize > bsize_min — kept transcription-faithful.)
        get_variance(&mut node.part_variances.none);
        if mi_col + bs_width_check <= f.tile_mi_col_end
            && mi_row + bs_height_check <= f.tile_mi_row_end
            && i64::from(node.part_variances.none.variance) < threshold
        {
            set_block_size(stamps, f, mi_row, mi_col, bsize);
            return true;
        }
        false
    } else if bsize > bsize_min {
        // Variance already computed to set the force_split (key frames
        // recompute here, :202).
        get_variance(&mut node.part_variances.none);
        // For key frame: take split for bsize above 32X32 or very high
        // variance (:204-208).
        if bsize > BLOCK_32X32 || i64::from(node.part_variances.none.variance) > (threshold << 4) {
            return false;
        }
        // If variance is low, take the bsize (no split).
        if mi_col + bs_width_check <= f.tile_mi_col_end
            && mi_row + bs_height_check <= f.tile_mi_row_end
            && i64::from(node.part_variances.none.variance) < threshold
        {
            set_block_size(stamps, f, mi_row, mi_col, bsize);
            return true;
        }
        // Check vertical split (:217-232).
        if mi_row + bs_height_check <= f.tile_mi_row_end
            && mi_col + bs_width_vert_check <= f.tile_mi_col_end
        {
            let subsize = get_partition_subsize(bsize, 2) as usize;
            let pbs = plane_block_size(subsize, f.ss_x, f.ss_y);
            get_variance(&mut node.part_variances.vert[0]);
            get_variance(&mut node.part_variances.vert[1]);
            if i64::from(node.part_variances.vert[0].variance) < threshold
                && i64::from(node.part_variances.vert[1].variance) < threshold
                && pbs < BLOCK_INVALID
            {
                set_block_size(stamps, f, mi_row, mi_col, subsize);
                set_block_size(stamps, f, mi_row, mi_col + block_width / 2, subsize);
                return true;
            }
        }
        // Check horizontal split (:234-249).
        if mi_col + bs_width_check <= f.tile_mi_col_end
            && mi_row + bs_height_horiz_check <= f.tile_mi_row_end
        {
            let subsize = get_partition_subsize(bsize, 1) as usize;
            let pbs = plane_block_size(subsize, f.ss_x, f.ss_y);
            get_variance(&mut node.part_variances.horz[0]);
            get_variance(&mut node.part_variances.horz[1]);
            if i64::from(node.part_variances.horz[0].variance) < threshold
                && i64::from(node.part_variances.horz[1].variance) < threshold
                && pbs < BLOCK_INVALID
            {
                set_block_size(stamps, f, mi_row, mi_col, subsize);
                set_block_size(stamps, f, mi_row + block_height / 2, mi_col, subsize);
                return true;
            }
        }
        false
    } else {
        false
    }
}

/// `av1_choose_var_based_partitioning` (var_based_part.c:1601) — KEY arm.
/// Fixes the partition for the SB at `(mi_row, mi_col)` as `bsize` stamps at
/// leaf top-left mi cells in `stamps` (a `mi_rows * mi_cols` row-major grid).
///
/// `src_y`/`base_y`/`stride` — the frame source plane
/// ([`crate::encode_sb::SbEncodeEnv`] conventions); the SB's pixels start at
/// `base_y + (mi_row*4)*stride + mi_col*4` (C's `x->plane[0].src.buf` after
/// `av1_set_offsets`).
///
/// `vbp_prune_16x16_split_using_min_max_sub_blk_var` = the rt speed feature
/// (:1806-1809; allintra stays false through speed 8 — the speed-9 flip is
/// documented for KB-12).
#[allow(clippy::too_many_arguments)]
pub fn choose_var_based_partitioning_key(
    stamps: &mut [u8],
    f: &VbpFrame,
    src_y: &[u16],
    base_y: usize,
    stride: usize,
    mi_row: i32,
    mi_col: i32,
    vbp_prune_16x16_split_using_min_max_sub_blk_var: bool,
) {
    debug_assert!(f.sb_size == BLOCK_64X64 || f.sb_size == BLOCK_128X128);
    let is_small_sb = f.sb_size == BLOCK_64X64;
    let num_64x64_blocks = if is_small_sb { 1usize } else { 4 };

    let thresholds = set_vbp_thresholds_key(f.qindex, f.bit_depth, f.num_pixels, f.sf);

    // force_split[85]: 0 root, 1-4 the 64x64s, 5-20 the 32x32s, 21-84 the
    // 16x16s (:1610/:1699).
    let mut force_split = [PartEval::All; 85];

    let mut vt = Box::new(Vp128x128::default());

    // ---- fill_variance_tree_leaves (:1105), KEY arm: 4x4-downsampled
    //      leaf fill; pixels_wide/high from the SB's frame overhang
    //      (xd->mb_to_right_edge / mb_to_bottom_edge, :1125-1126). ----
    let sb_px = if is_small_sb { 64i32 } else { 128 };
    let pixels_wide = sb_px.min((f.mi_cols - mi_col) * 4);
    let pixels_high = sb_px.min((f.mi_rows - mi_row) * 4);
    let sb_off = base_y + (mi_row as usize * 4) * stride + mi_col as usize * 4;

    for blk64_idx in 0..num_64x64_blocks {
        let x64_idx = blk_idx_x(blk64_idx, 6);
        let y64_idx = blk_idx_y(blk64_idx, 6);
        let blk64_scale_idx = blk64_idx << 2;
        force_split[blk64_idx + 1] = PartEval::All;
        for lvl1_idx in 0..4usize {
            let x32_idx = x64_idx + blk_idx_x(lvl1_idx, 5);
            let y32_idx = y64_idx + blk_idx_y(lvl1_idx, 5);
            let lvl1_scale_idx = (blk64_scale_idx + lvl1_idx) << 2;
            force_split[5 + blk64_scale_idx + lvl1_idx] = PartEval::All;
            for lvl2_idx in 0..4usize {
                let x16_idx = x32_idx + blk_idx_x(lvl2_idx, 4);
                let y16_idx = y32_idx + blk_idx_y(lvl2_idx, 4);
                let split_index = 21 + lvl1_scale_idx + lvl2_idx;
                force_split[split_index] = PartEval::All;
                let vst = &mut vt.split[blk64_idx].split[lvl1_idx].split[lvl2_idx];
                // Go down to 4x4 down-sampling for variance (:1156-1167).
                for lvl3_idx in 0..4usize {
                    let x8_idx = x16_idx + blk_idx_x(lvl3_idx, 3);
                    let y8_idx = y16_idx + blk_idx_y(lvl3_idx, 3);
                    let vst2 = &mut vst.split[lvl3_idx];
                    // fill_variance_4x4avg (:390): border_offset_4x4 == 0 on
                    // key frames.
                    for idx in 0..4usize {
                        let x4_idx = x8_idx + blk_idx_x(idx, 2);
                        let y4_idx = y8_idx + blk_idx_y(idx, 2);
                        let mut sse = 0u32;
                        let mut sum = 0i32;
                        if (x4_idx as i32) < pixels_wide && (y4_idx as i32) < pixels_high {
                            let src_avg = avg_4x4(src_y, sb_off + y4_idx * stride + x4_idx, stride);
                            let dst_avg = 128;
                            sum = src_avg - dst_avg;
                            sse = (sum * sum) as u32;
                        }
                        fill_variance(sse, sum, 0, &mut vst2.split[idx]);
                    }
                }
            }
        }
    }

    // ---- the stage-2 force-split walk (:1788-1881), key arms ----
    for blk64_idx in 0..num_64x64_blocks {
        let blk64_scale_idx = blk64_idx << 2;
        for lvl1_idx in 0..4usize {
            let lvl1_scale_idx = (blk64_scale_idx + lvl1_idx) << 2;
            for lvl2_idx in 0..4usize {
                // (key frames only reach this body, :1796.)
                let vtemp = &mut vt.split[blk64_idx].split[lvl1_idx].split[lvl2_idx];
                for lvl3_idx in 0..4usize {
                    let sp = vtemp.split[lvl3_idx].split;
                    fill_variance_node(&sp, &mut vtemp.split[lvl3_idx].part_variances);
                }
                let nones = child_nones([
                    &vtemp.split[0].part_variances,
                    &vtemp.split[1].part_variances,
                    &vtemp.split[2].part_variances,
                    &vtemp.split[3].part_variances,
                ]);
                fill_variance_node(&nones, &mut vtemp.part_variances);
                // If variance of this 16x16 block is above the threshold,
                // force block to split (:1801-1813).
                get_variance(&mut vtemp.part_variances.none);
                if i64::from(vtemp.part_variances.none.variance) > thresholds[3] {
                    let split_index = 21 + lvl1_scale_idx + lvl2_idx;
                    force_split[split_index] = if vbp_prune_16x16_split_using_min_max_sub_blk_var {
                        // get_part_eval_based_on_sub_blk_var (:1530).
                        let mut max_8x8 = 0i32;
                        let mut min_8x8 = i32::MAX;
                        for sp in &mut vtemp.split {
                            get_variance(&mut sp.part_variances.none);
                            max_8x8 = max_8x8.max(sp.part_variances.none.variance);
                            min_8x8 = min_8x8.min(sp.part_variances.none.variance);
                        }
                        if i64::from(max_8x8 - min_8x8) > (thresholds[3] << 2) {
                            PartEval::OnlySplit
                        } else {
                            PartEval::OnlyNone
                        }
                    } else {
                        PartEval::OnlySplit
                    };
                    force_split[5 + blk64_scale_idx + lvl1_idx] = PartEval::OnlySplit;
                    force_split[blk64_idx + 1] = PartEval::OnlySplit;
                    force_split[0] = PartEval::OnlySplit;
                }
            }
            {
                let v32 = &mut vt.split[blk64_idx].split[lvl1_idx];
                let nones = child_nones([
                    &v32.split[0].part_variances,
                    &v32.split[1].part_variances,
                    &v32.split[2].part_variances,
                    &v32.split[3].part_variances,
                ]);
                fill_variance_node(&nones, &mut v32.part_variances);
                // 32x32 threshold check (:1825-1852; the !is_key_frame
                // second/third disjuncts are dead here).
                if force_split[5 + blk64_scale_idx + lvl1_idx] == PartEval::All {
                    get_variance(&mut v32.part_variances.none);
                    let var_32x32 = v32.part_variances.none.variance;
                    if i64::from(var_32x32) > thresholds[2] {
                        force_split[5 + blk64_scale_idx + lvl1_idx] = PartEval::OnlySplit;
                        force_split[blk64_idx + 1] = PartEval::OnlySplit;
                        force_split[0] = PartEval::OnlySplit;
                    }
                }
            }
        }
        if force_split[1 + blk64_idx] == PartEval::All {
            let v64 = &mut vt.split[blk64_idx];
            let nones = child_nones([
                &v64.split[0].part_variances,
                &v64.split[1].part_variances,
                &v64.split[2].part_variances,
                &v64.split[3].part_variances,
            ]);
            fill_variance_node(&nones, &mut v64.part_variances);
            get_variance(&mut v64.part_variances.none);
            // (the max/min 64x64 spread rule is !is_key_frame, :1873.)
        }
        if is_small_sb {
            force_split[0] = PartEval::OnlySplit;
        }
    }

    // Root 128x128 fill (:1883-1894): both root force rules are
    // !is_key_frame; the fill itself only runs when the root survived as
    // PART_EVAL_ALL (128-SB frames with no forced splits).
    if force_split[0] == PartEval::All {
        let nones = child_nones([
            &vt.split[0].part_variances,
            &vt.split[1].part_variances,
            &vt.split[2].part_variances,
            &vt.split[3].part_variances,
        ]);
        let mut pv = vt.part_variances;
        fill_variance_node(&nones, &mut pv);
        vt.part_variances = pv;
    }

    // ---- the partition assignment descent (:1896-1942) ----
    let root_set = mi_col + 32 <= f.tile_mi_col_end
        && mi_row + 32 <= f.tile_mi_row_end
        && set_vt_partitioning(
            stamps,
            f,
            NodeView {
                part_variances: &mut vt.part_variances,
            },
            BLOCK_128X128,
            mi_row,
            mi_col,
            thresholds[0],
            BLOCK_16X16,
            force_split[0],
        );
    if !root_set {
        for blk64_idx in 0..num_64x64_blocks {
            let x64_idx = blk_idx_x(blk64_idx, 4) as i32;
            let y64_idx = blk_idx_y(blk64_idx, 4) as i32;
            let blk64_scale_idx = blk64_idx << 2;
            if set_vt_partitioning(
                stamps,
                f,
                NodeView {
                    part_variances: &mut vt.split[blk64_idx].part_variances,
                },
                BLOCK_64X64,
                mi_row + y64_idx,
                mi_col + x64_idx,
                thresholds[1],
                BLOCK_16X16,
                force_split[1 + blk64_idx],
            ) {
                continue;
            }
            for lvl1_idx in 0..4usize {
                let x32_idx = blk_idx_x(lvl1_idx, 3) as i32;
                let y32_idx = blk_idx_y(lvl1_idx, 3) as i32;
                let lvl1_scale_idx = (blk64_scale_idx + lvl1_idx) << 2;
                if set_vt_partitioning(
                    stamps,
                    f,
                    NodeView {
                        part_variances: &mut vt.split[blk64_idx].split[lvl1_idx].part_variances,
                    },
                    BLOCK_32X32,
                    mi_row + y64_idx + y32_idx,
                    mi_col + x64_idx + x32_idx,
                    thresholds[2],
                    BLOCK_16X16,
                    force_split[5 + blk64_scale_idx + lvl1_idx],
                ) {
                    continue;
                }
                for lvl2_idx in 0..4usize {
                    let x16_idx = blk_idx_x(lvl2_idx, 2) as i32;
                    let y16_idx = blk_idx_y(lvl2_idx, 2) as i32;
                    let split_index = 21 + lvl1_scale_idx + lvl2_idx;
                    if set_vt_partitioning(
                        stamps,
                        f,
                        NodeView {
                            part_variances: &mut vt.split[blk64_idx].split[lvl1_idx].split
                                [lvl2_idx]
                                .part_variances,
                        },
                        BLOCK_16X16,
                        mi_row + y64_idx + y32_idx + y16_idx,
                        mi_col + x64_idx + x32_idx + x16_idx,
                        thresholds[3],
                        BLOCK_8X8,
                        force_split[split_index],
                    ) {
                        continue;
                    }
                    for lvl3_idx in 0..4usize {
                        let x8_idx = blk_idx_x(lvl3_idx, 1) as i32;
                        let y8_idx = blk_idx_y(lvl3_idx, 1) as i32;
                        set_block_size(
                            stamps,
                            f,
                            mi_row + y64_idx + y32_idx + y16_idx + y8_idx,
                            mi_col + x64_idx + x32_idx + x16_idx + x8_idx,
                            BLOCK_8X8,
                        );
                    }
                }
            }
        }
    }
}

/// `get_partition` (av1_common_int.h:1775) over the [`set_block_size`]
/// stamp grid: derive the partition type at `(mi_row, mi_col, bsize)` from
/// the stamped leaf bsizes. The variance tree only produces
/// NONE/HORZ/VERT/SPLIT shapes, but the derivation is transcribed in full
/// (the extended-partition disambiguation included) so it stays faithful at
/// frame edges.
pub fn get_partition_from_stamps(
    stamps: &[u8],
    mi_rows: i32,
    mi_cols: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> i32 {
    const PARTITION_NONE: i32 = 0;
    const PARTITION_HORZ: i32 = 1;
    const PARTITION_VERT: i32 = 2;
    const PARTITION_SPLIT: i32 = 3;
    const PARTITION_HORZ_A: i32 = 4;
    const PARTITION_HORZ_B: i32 = 5;
    const PARTITION_VERT_A: i32 = 6;
    const PARTITION_VERT_B: i32 = 7;
    const PARTITION_HORZ_4: i32 = 8;
    const PARTITION_VERT_4: i32 = 9;
    const PARTITION_INVALID: i32 = -1;

    if mi_row >= mi_rows || mi_col >= mi_cols {
        return PARTITION_INVALID;
    }
    let at = |r: i32, c: i32| -> usize { stamps[(r * mi_cols + c) as usize] as usize };
    let subsize = at(mi_row, mi_col);
    if subsize == bsize {
        return PARTITION_NONE;
    }
    let bhigh = MI_SIZE_WIDE_B[bsize] as i32; // square: high == wide
    let bwide = bhigh;
    let sshigh = MI_SIZE_HIGH_B[subsize] as i32;
    let sswide = MI_SIZE_WIDE_B[subsize] as i32;

    if bsize > BLOCK_8X8 && mi_row + bwide / 2 < mi_rows && mi_col + bhigh / 2 < mi_cols {
        // The block might be using an extended partition type.
        let mbmi_right = at(mi_row, mi_col + bwide / 2);
        let mbmi_below = at(mi_row + bhigh / 2, mi_col);
        if sswide == bwide {
            // PARTITION_HORZ_4, PARTITION_HORZ or PARTITION_HORZ_B.
            if sshigh * 4 == bhigh {
                return PARTITION_HORZ_4;
            }
            debug_assert_eq!(sshigh * 2, bhigh);
            if mbmi_below == subsize {
                return PARTITION_HORZ;
            }
            return PARTITION_HORZ_B;
        } else if sshigh == bhigh {
            // PARTITION_VERT_4, PARTITION_VERT or PARTITION_VERT_B.
            if sswide * 4 == bwide {
                return PARTITION_VERT_4;
            }
            debug_assert_eq!(sswide * 2, bwide);
            if mbmi_right == subsize {
                return PARTITION_VERT;
            }
            return PARTITION_VERT_B;
        } else {
            if sswide * 2 != bwide || sshigh * 2 != bhigh {
                return PARTITION_SPLIT;
            }
            if MI_SIZE_WIDE_B[mbmi_below] as i32 == bwide {
                return PARTITION_HORZ_A;
            }
            if MI_SIZE_HIGH_B[mbmi_right] as i32 == bhigh {
                return PARTITION_VERT_A;
            }
            return PARTITION_SPLIT;
        }
    }
    let vert_split = sswide < bwide;
    let horz_split = sshigh < bhigh;
    let split_idx = ((vert_split as usize) << 1) | horz_split as usize;
    debug_assert_ne!(split_idx, 0);
    [
        PARTITION_INVALID,
        PARTITION_HORZ,
        PARTITION_VERT,
        PARTITION_SPLIT,
    ][split_idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// KB-32 root #1 — `set_vbp_thresholds_key_frame`'s two
    /// `force_large_partition_blocks_intra` arms (var_based_part.c:539-544 and
    /// :552-554), locked in BOTH directions on both sides of
    /// `RESOLUTION_720P` (= 1280 * 720 pixels — an AREA threshold, not a
    /// dimension one).
    ///
    /// The speed-8 and speed-9 columns are the interesting pair: the arm is
    /// armed at both speeds on a >=720p frame, but `shift_steps` is 1 at speed
    /// 8 (shift 8) and 0 at speed 9 (shift 7), so speed 9 keeps the plain
    /// `threshold_base` and only its `shift_val` moves.
    #[test]
    fn kb32_force_large_intra_threshold_arms() {
        const SUB: i64 = 1280 * 720 - 1;
        const OVER: i64 = 1280 * 720;
        let off = VbpSf::default();
        let s8 = VbpSf {
            force_large_partition_blocks_intra: true,
            var_part_split_threshold_shift: 8,
            allintra: true,
        };
        let s9 = VbpSf {
            force_large_partition_blocks_intra: true,
            var_part_split_threshold_shift: 7,
            allintra: true,
        };
        // qindex 100 bd8: threshold_base = 120 * av1_ac_quant_QTX(100, 0, 8).
        let b = 120i64 * i64::from(av1_ac_quant_qtx(100, 0, 8));
        assert!(b > 0, "the fixture must have a non-degenerate base");

        // OFF — the pre-KB-32 behaviour, unchanged at every size.
        assert_eq!(
            set_vbp_thresholds_key(100, 8, SUB, off),
            [b, b, b / 3, b >> 1, b << 2]
        );
        assert_eq!(
            set_vbp_thresholds_key(100, 8, OVER, off),
            [b, b, b >> 2, b >> 2, b << 2]
        );

        // SPEED 8 — `threshold_base <<= (8 - 7)`, i.e. every threshold doubles;
        // above RESOLUTION_720P `shift_val` is also 1 instead of 2.
        let d = b << 1;
        assert_eq!(
            set_vbp_thresholds_key(100, 8, SUB, s8),
            [d, d, d / 3, d >> 1, d << 2],
            "speed 8 below RESOLUTION_720P: only the shift_steps arm applies"
        );
        assert_eq!(
            set_vbp_thresholds_key(100, 8, OVER, s8),
            [d, d, d >> 1, d >> 1, d << 2],
            "speed 8 at/above RESOLUTION_720P: BOTH arms apply"
        );

        // SPEED 9 — shift_steps is 0, so the base is untouched; only shift_val
        // moves, and only above the area threshold. Below it, speed 9 armed is
        // byte-identical to the arm being off, which is exactly why the cpu9
        // band has a SHARP area threshold while cpu8 does not.
        assert_eq!(
            set_vbp_thresholds_key(100, 8, SUB, s9),
            set_vbp_thresholds_key(100, 8, SUB, off),
            "speed 9 below RESOLUTION_720P must be indistinguishable from off"
        );
        assert_eq!(
            set_vbp_thresholds_key(100, 8, OVER, s9),
            [b, b, b >> 1, b >> 1, b << 2],
            "speed 9 at/above RESOLUTION_720P: shift_val 1, base untouched"
        );
        assert_ne!(
            set_vbp_thresholds_key(100, 8, OVER, s9),
            set_vbp_thresholds_key(100, 8, OVER, off),
            "the speed-9 arm must be observable above the area threshold, or \
             the cpu9 band would not exist"
        );

        // The area threshold is on `num_pixels`, exclusive at 1280*720 - 1 and
        // inclusive at 1280*720 — the boundary pair, both directions.
        assert_ne!(
            set_vbp_thresholds_key(100, 8, SUB, off)[2],
            set_vbp_thresholds_key(100, 8, OVER, off)[2]
        );
    }

    /// The `assert(shift_steps >= 0)` C carries (var_based_part.c:542) is a
    /// real bound, not decoration: a shift below the ALLINTRA floor of 7 with
    /// the arm armed is an unreachable state, and the port must say so loudly
    /// rather than shift by a negative amount.
    #[test]
    #[should_panic(expected = "below the 7 floor")]
    fn kb32_shift_steps_floor_is_asserted() {
        set_vbp_thresholds_key(
            100,
            8,
            64 * 64,
            VbpSf {
                force_large_partition_blocks_intra: true,
                var_part_split_threshold_shift: 5,
                allintra: true,
            },
        );
    }

    /// **KB-28 — `num_pixels` is `cm->width * cm->height`, the TRUE CROP area
    /// (var_based_part.c:667), and the mi-aligned area is a different number.**
    ///
    /// `av1_get_MBs` rounds the mi grid UP to 8 px (alloccommon.c:30-33), so
    /// feeding this function `mi_cols * 4 * mi_rows * 4` picks the wrong arm on
    /// every crop whose rounding crosses `RESOLUTION_720P`. Locked over all
    /// three speeds the VAR_BASED partitioner runs at (7, 8, 9) and both sides
    /// of the boundary, in both directions, on the exact crops
    /// `kb28_crop_dims`' byte gates encode — so a regression is caught here in
    /// microseconds instead of in a 30-cell encode sweep.
    #[test]
    fn kb28_num_pixels_is_the_crop_area_not_the_mi_area() {
        const fn mi_aligned(px: i64) -> i64 {
            (px + 7) & !7
        }
        // (crop_w, crop_h): every one has crop area < 1280*720 <= mi area.
        const STRADDLERS: &[(i64, i64)] = &[(1272, 724), (1274, 722), (954, 962)];
        // Speed 7 (arm off), speed 8 (shift 8), speed 9 (shift 7) — the three
        // `VbpSf` states reachable on the allintra KEY path.
        let sfs = [
            VbpSf::default(),
            VbpSf {
                force_large_partition_blocks_intra: true,
                var_part_split_threshold_shift: 8,
                allintra: true,
            },
            VbpSf {
                force_large_partition_blocks_intra: true,
                var_part_split_threshold_shift: 7,
                allintra: true,
            },
        ];
        for &(w, h) in STRADDLERS {
            let crop_px = w * h;
            let mi_px = mi_aligned(w) * mi_aligned(h);
            assert!(
                crop_px < 1280 * 720 && mi_px >= 1280 * 720,
                "{w}x{h} must straddle RESOLUTION_720P (crop {crop_px}, mi {mi_px})"
            );
            for (i, sf) in sfs.iter().enumerate() {
                let from_crop = set_vbp_thresholds_key(100, 8, crop_px, *sf);
                let from_mi = set_vbp_thresholds_key(100, 8, mi_px, *sf);
                // Speed 9 armed below the threshold is indistinguishable from
                // the arm being off, but the two READINGS still differ because
                // they land on opposite sides of it.
                assert_ne!(
                    from_crop, from_mi,
                    "{w}x{h} sf#{i}: the crop and mi-aligned areas must resolve \
                     DIFFERENT thresholds, or the byte gate for this shape is \
                     vacuous"
                );
                // And the crop reading must be the sub-720p arm.
                let b = from_crop[0];
                assert_eq!(
                    (from_crop[2], from_crop[3]),
                    (b / 3, b >> 1),
                    "{w}x{h} sf#{i}: the true crop is below RESOLUTION_720P, so \
                     thresholds[2]/[3] take the `base/3, base>>1` arm (:548-549)"
                );
            }
        }
        // Both controls: mi == crop, so the two readings must AGREE — that is
        // what makes those cells a valid negative control for this fix.
        for &(w, h) in &[(1280i64, 720i64), (1280, 712), (1280, 728), (1216, 768)] {
            assert_eq!(mi_aligned(w), w);
            assert_eq!(mi_aligned(h), h);
            for sf in &sfs {
                assert_eq!(
                    set_vbp_thresholds_key(100, 8, w * h, *sf),
                    set_vbp_thresholds_key(100, 8, mi_aligned(w) * mi_aligned(h), *sf)
                );
            }
        }
    }

    /// `get_variance` matches the plain (sse - sum^2/n) * 256 / n definition
    /// on in-range inputs (the C expression form is wrap-faithful; this
    /// pins the arithmetic on representative values).
    #[test]
    fn get_variance_matches_definition() {
        // 16 4x4 leaves of a 16x16: log2_count accumulates to 4 at the 16x16
        // none node (each leaf log2 0, three sum levels).
        let mut v = VPartVar {
            sum_square_error: 16 * 40 * 40,
            sum_error: 16 * 40,
            log2_count: 4,
            variance: 0,
        };
        get_variance(&mut v);
        // sse - sum^2 >> 4 = 25600 - (640*640)>>4 = 25600 - 25600 = 0.
        assert_eq!(v.variance, 0);

        let mut v = VPartVar {
            sum_square_error: 10_000,
            sum_error: 100,
            log2_count: 4,
            variance: 0,
        };
        get_variance(&mut v);
        // (10000 - 625) * 256 >> 4 = 9375 * 16 = 150000.
        assert_eq!(v.variance, 150_000);
    }

    /// Flat content ⇒ zero variance everywhere ⇒ the KEY tree bottoms out at
    /// 32x32 NONE stamps (64x64 must split per the `bsize > BLOCK_32X32`
    /// key rule; 32x32 var 0 < threshold).
    #[test]
    fn flat_64_sb_stamps_four_32x32() {
        let stride = 80usize;
        let src = vec![128u16; stride * 72];
        let f = VbpFrame {
            mi_rows: 16,
            mi_cols: 16,
            tile_mi_row_end: 16,
            tile_mi_col_end: 16,
            num_pixels: 64 * 64,
            sb_size: BLOCK_64X64,
            qindex: 100,
            bit_depth: 8,
            ss_x: 1,
            ss_y: 1,
            sf: Default::default(),
        };
        let mut stamps = vec![0u8; 16 * 16];
        choose_var_based_partitioning_key(&mut stamps, &f, &src, 0, stride, 0, 0, false);
        for (r, c) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
            assert_eq!(stamps[r * 16 + c], BLOCK_32X32 as u8, "at ({r},{c})");
        }
        assert_eq!(
            get_partition_from_stamps(&stamps, 16, 16, 0, 0, BLOCK_64X64),
            3, // PARTITION_SPLIT
        );
        for (r, c) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
            assert_eq!(
                get_partition_from_stamps(&stamps, 16, 16, r, c, BLOCK_32X32),
                0, // PARTITION_NONE
                "at ({r},{c})"
            );
        }
    }

    /// A hard vertical edge inside one 32x32 (flat 16x16 quadrants): the
    /// 32x32's own variance exceeds thresholds[2] so the stage-2 walk
    /// FORCE-SPLITS it (PART_EVAL_ONLY_SPLIT fires before the rect arms can
    /// run — on KEY frames the interior rect stamps are reachable only on
    /// exact `variance == threshold` ties); the flat 16x16s under it stay
    /// NONE. Pins the force-split propagation + the SPLIT derivation.
    #[test]
    fn interior_edge_32x32_force_splits_to_16s() {
        let stride = 80usize;
        let mut src = vec![64u16; stride * 72];
        // Top-left 32x32: columns 0..16 = 64, 16..32 = 192 (each 16x16
        // flat); rest of the SB flat 128.
        for r in 0..64 {
            for c in 0..64 {
                src[r * stride + c] = if r < 32 && c < 32 {
                    if c < 16 { 64 } else { 192 }
                } else {
                    128
                };
            }
        }
        let f = VbpFrame {
            mi_rows: 16,
            mi_cols: 16,
            tile_mi_row_end: 16,
            tile_mi_col_end: 16,
            num_pixels: 64 * 64,
            sb_size: BLOCK_64X64,
            qindex: 220,
            bit_depth: 8,
            ss_x: 1,
            ss_y: 1,
            sf: Default::default(),
        };
        let mut stamps = vec![0u8; 16 * 16];
        choose_var_based_partitioning_key(&mut stamps, &f, &src, 0, stride, 0, 0, false);
        assert_eq!(
            get_partition_from_stamps(&stamps, 16, 16, 0, 0, BLOCK_32X32),
            3, // PARTITION_SPLIT — forced by the 32x32 variance rule
        );
        for (r, c) in [(0, 0), (0, 4), (4, 0), (4, 4)] {
            assert_eq!(stamps[r * 16 + c], BLOCK_16X16 as u8, "at ({r},{c})");
            assert_eq!(
                get_partition_from_stamps(&stamps, 16, 16, r as i32, c as i32, BLOCK_16X16),
                0, // the flat 16x16s stay NONE
            );
        }
        // The flat quadrants stay NONE-32.
        assert_eq!(
            get_partition_from_stamps(&stamps, 16, 16, 8, 8, BLOCK_32X32),
            0
        );
    }

    /// Frame-edge rect stamps: on a 48x48 frame (mi_cols = 12) the (0,8)
    /// 32x32's right half is out of frame — the NONE fit check fails
    /// (`bs_width_check = (8>>1)+1 = 5`, 8+5 > 12) but the VERT half-width
    /// check passes (`(8>>2)+1 = 3`, 8+3 <= 12), so flat content stamps the
    /// visible 16x32 (the out-of-frame sibling stamp is skipped) and the
    /// stamp grid derives PARTITION_VERT via the base table (the
    /// ext-partition arm is bounds-gated off).
    #[test]
    fn edge_vert_single_strip_stamp() {
        let stride = 80usize;
        let src = vec![128u16; stride * 72];
        let f = VbpFrame {
            mi_rows: 12,
            mi_cols: 12,
            tile_mi_row_end: 12,
            tile_mi_col_end: 12,
            num_pixels: 48 * 48,
            sb_size: BLOCK_64X64,
            qindex: 100,
            bit_depth: 8,
            ss_x: 1,
            ss_y: 1,
            sf: Default::default(),
        };
        let mut stamps = vec![0u8; 12 * 12];
        choose_var_based_partitioning_key(&mut stamps, &f, &src, 0, stride, 0, 0, false);
        // Interior 32x32 at (0,0): NONE (flat).
        assert_eq!(
            get_partition_from_stamps(&stamps, 12, 12, 0, 0, BLOCK_32X32),
            0
        );
        // Right-edge 32x32 at (0,8): single-strip VERT.
        assert_eq!(stamps[8], 7, "16x32 stamp at (0,8)"); // BLOCK_16X32 = 7
        assert_eq!(
            get_partition_from_stamps(&stamps, 12, 12, 0, 8, BLOCK_32X32),
            2, // PARTITION_VERT
        );
        // Bottom-edge 32x32 at (8,0): single-strip HORZ (fit fails on rows,
        // horz half-height check passes) -- 32x16 = BLOCK_32X16 = 8.
        assert_eq!(stamps[8 * 12], 8, "32x16 stamp at (8,0)");
        assert_eq!(
            get_partition_from_stamps(&stamps, 12, 12, 8, 0, BLOCK_32X32),
            1, // PARTITION_HORZ
        );
    }
}

// ===========================================================================
// The INTER arm of var_based_part.c.
//
// Everything above is the KEY-frame path, where the "prediction" is a flat 128
// and no reference exists. On an inter frame the leaf fill is a SOURCE-vs-
// REFERENCE 8x8 average difference, the partition can be forced by the
// temporal-variance flags, and the whole superblock can be skipped outright
// when the source SAD says nothing moved. Those three are here.
//
// | Rust | C |
// |---|---|
// | [`all_blks_inside`] | `all_blks_inside` (:255) |
// | [`fill_variance_8x8avg`] | `fill_variance_8x8avg` (:330) + `_lowbd` (:290) / `_highbd` (:267) |
// | [`compute_minmax_8x8`] | `compute_minmax_8x8` (:349) |
// | [`scale_part_thresh_content`] | `scale_part_thresh_content` (:425) |
// | [`mv_distance`] | `mv_distance` (:1259) |
// | [`get_force_skip_low_temp_var`] | `av1_get_force_skip_low_temp_var` (:901) |
// | [`get_force_skip_low_temp_var_small_sb`] | `av1_get_force_skip_low_temp_var_small_sb` (:852) |
// | [`is_set_force_zeromv_skip_based_on_src_sad`] | `is_set_force_zeromv_skip_based_on_src_sad` (:1549) |
//
// Differential coverage: `tests/var_part_inter_diff.rs`.
// ===========================================================================

use aom_dsp::dist::avg::{avg_8x8, avg_8x8_quad, highbd_avg_8x8, highbd_minmax_8x8, minmax_8x8};

/// `SOURCE_SAD` (`av1/encoder/block.h:839`) — how much the source moved.
///
/// C compares these with `<=` and `==`, so the discriminants are load-bearing
/// and the ordering is the enum's own.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum SourceSad {
    /// `kZeroSad` — nothing moved at all.
    Zero = 0,
    /// `kVeryLowSad`.
    VeryLow = 1,
    /// `kLowSad`.
    Low = 2,
    /// `kMedSad`.
    Med = 3,
    /// `kHighSad`.
    High = 4,
}

/// `all_blks_inside` (var_based_part.c:255) — whether all four 8x8 sub-blocks
/// of the 16x16 at `(x16_idx, y16_idx)` start inside the visible frame.
///
/// C tests the sub-block's TOP-LEFT only, not its full extent, so a block that
/// starts inside and runs off the right edge counts as inside. That is
/// deliberate — the 8x8 average reads the frame's padded border there — and it
/// is reproduced rather than tightened.
#[must_use]
pub fn all_blks_inside(
    x16_idx: usize,
    y16_idx: usize,
    pixels_wide: usize,
    pixels_high: usize,
) -> bool {
    (0..4).all(|idx| {
        x16_idx + blk_idx_x(idx, 3) < pixels_wide && y16_idx + blk_idx_y(idx, 3) < pixels_high
    })
}

/// One 8x8 leaf record as `fill_variance` writes it: `(sum_square_error,
/// sum_error)`. `log2_count` is always 0 at this level, so it is not carried.
pub type Leaf8x8 = (u32, i32);

/// `fill_variance_8x8avg` (var_based_part.c:330) — the INTER leaf fill.
///
/// Each of the 16x16 block's four 8x8 sub-blocks contributes
/// `sum = avg8x8(src) - avg8x8(ref)` and `sse = sum * sum`. A sub-block whose
/// top-left is outside the visible frame contributes `(0, 0)`.
///
/// C returns through a `VP16x16 *`; the port returns the four records, which
/// is the whole observable — `fill_variance` writes nothing else.
///
/// # The two averaging paths are not interchangeable, and C knows it
/// When every sub-block is inside, C calls the QUAD kernel
/// (`aom_avg_8x8_quad`) on the whole 16x16; otherwise it calls
/// `aom_avg_8x8` per sub-block. Both compute the same four averages — the
/// quad kernel is a SIMD fast path — and the port takes the same branch so a
/// future divergence between them shows up here rather than being averaged
/// away.
///
/// The highbd arm (`fill_variance_8x8avg_highbd`, :267) has NO quad path at
/// all: libaom's own TODO notes it. That asymmetry is reproduced.
#[must_use]
pub fn fill_variance_8x8avg(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    x16_idx: usize,
    y16_idx: usize,
    pixels_wide: usize,
    pixels_high: usize,
) -> [Leaf8x8; 4] {
    let mut out = [(0u32, 0i32); 4];
    if all_blks_inside(x16_idx, y16_idx, pixels_wide, pixels_high) {
        let src_avg = avg_8x8_quad(src, src_stride, x16_idx, y16_idx);
        let dst_avg = avg_8x8_quad(dst, dst_stride, x16_idx, y16_idx);
        for (o, (&s, &d)) in out.iter_mut().zip(src_avg.iter().zip(&dst_avg)) {
            let sum = s as i32 - d as i32;
            *o = ((sum * sum) as u32, sum);
        }
    } else {
        for (idx, o) in out.iter_mut().enumerate() {
            let x8 = x16_idx + blk_idx_x(idx, 3);
            let y8 = y16_idx + blk_idx_y(idx, 3);
            if x8 < pixels_wide && y8 < pixels_high {
                let s = avg_8x8(&src[y8 * src_stride + x8..], src_stride) as i32;
                let d = avg_8x8(&dst[y8 * dst_stride + x8..], dst_stride) as i32;
                let sum = s - d;
                *o = ((sum * sum) as u32, sum);
            }
        }
    }
    out
}

/// `fill_variance_8x8avg_highbd` (var_based_part.c:267) — the 10/12-bit arm.
///
/// Per-sub-block only: there is no `aom_highbd_avg_8x8_quad`.
#[must_use]
pub fn fill_variance_8x8avg_highbd(
    src: &[u16],
    src_stride: usize,
    dst: &[u16],
    dst_stride: usize,
    x16_idx: usize,
    y16_idx: usize,
    pixels_wide: usize,
    pixels_high: usize,
) -> [Leaf8x8; 4] {
    let mut out = [(0u32, 0i32); 4];
    for (idx, o) in out.iter_mut().enumerate() {
        let x8 = x16_idx + blk_idx_x(idx, 3);
        let y8 = y16_idx + blk_idx_y(idx, 3);
        if x8 < pixels_wide && y8 < pixels_high {
            let s = highbd_avg_8x8(&src[y8 * src_stride + x8..], src_stride) as i32;
            let d = highbd_avg_8x8(&dst[y8 * dst_stride + x8..], dst_stride) as i32;
            let sum = s - d;
            *o = ((sum * sum) as u32, sum);
        }
    }
    out
}

/// `compute_minmax_8x8` (var_based_part.c:349) — the SPREAD of the four 8x8
/// source-vs-reference min/max ranges inside one 16x16 block.
///
/// C seeds `minmax_max = 0` and `minmax_min = 255` and never resets them, so a
/// 16x16 with NO in-frame sub-block returns `0 - 255 = -255`, not 0. That is
/// reproduced: the caller (`fill_variance_tree_leaves`) only reaches this
/// function for blocks it has already established are at least partly inside,
/// but the value is C's either way.
#[must_use]
pub fn compute_minmax_8x8(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    x16_idx: usize,
    y16_idx: usize,
    pixels_wide: usize,
    pixels_high: usize,
) -> i32 {
    let mut minmax_max = 0i32;
    let mut minmax_min = 255i32;
    for idx in 0..4 {
        let x8 = x16_idx + blk_idx_x(idx, 3);
        let y8 = y16_idx + blk_idx_y(idx, 3);
        if x8 < pixels_wide && y8 < pixels_high {
            let (min, max) = minmax_8x8(
                &src[y8 * src_stride + x8..],
                src_stride,
                &dst[y8 * dst_stride + x8..],
                dst_stride,
            );
            minmax_max = minmax_max.max(max - min);
            minmax_min = minmax_min.min(max - min);
        }
    }
    minmax_max - minmax_min
}

/// The 10/12-bit arm of [`compute_minmax_8x8`] — C selects it inside the same
/// function on `highbd_flag & YV12_FLAG_HIGHBITDEPTH`.
///
/// Note the seed asymmetry it inherits: `aom_highbd_minmax_8x8_c` seeds its
/// `min` at 65535 while `minmax_min` here still starts at 255, exactly as in
/// C. A 16x16 with no in-frame sub-block therefore returns `-255` at both
/// depths.
#[must_use]
pub fn compute_minmax_8x8_highbd(
    src: &[u16],
    src_stride: usize,
    dst: &[u16],
    dst_stride: usize,
    x16_idx: usize,
    y16_idx: usize,
    pixels_wide: usize,
    pixels_high: usize,
) -> i32 {
    let mut minmax_max = 0i32;
    let mut minmax_min = 255i32;
    for idx in 0..4 {
        let x8 = x16_idx + blk_idx_x(idx, 3);
        let y8 = y16_idx + blk_idx_y(idx, 3);
        if x8 < pixels_wide && y8 < pixels_high {
            let (min, max) = highbd_minmax_8x8(
                &src[y8 * src_stride + x8..],
                src_stride,
                &dst[y8 * dst_stride + x8..],
                dst_stride,
            );
            minmax_max = minmax_max.max(max - min);
            minmax_min = minmax_min.min(max - min);
        }
    }
    minmax_max - minmax_min
}

/// `scale_part_thresh_content` (var_based_part.c:425).
#[must_use]
pub fn scale_part_thresh_content(
    threshold_base: i64,
    speed: i32,
    non_reference_frame: bool,
    is_static: bool,
) -> i64 {
    let mut threshold = threshold_base;
    if non_reference_frame && !is_static {
        threshold = (3 * threshold) >> 1;
    }
    if speed >= 8 {
        return (5 * threshold) >> 2;
    }
    threshold
}

/// `mv_distance` (var_based_part.c:1259) — L1 distance between two full-pel MVs.
#[inline]
#[must_use]
pub fn mv_distance(mv0: (i16, i16), mv1: (i16, i16)) -> i32 {
    (i32::from(mv0.0) - i32::from(mv1.0)).abs() + (i32::from(mv0.1) - i32::from(mv1.1)).abs()
}

/// `pos_shift_16x16` (var_based_part.c:848) — where the 16x16 flag for the
/// `(i, j)` cell of a 64x64 superblock lives in `variance_low`.
const POS_SHIFT_16X16: [[usize; 4]; 4] = [
    [9, 10, 13, 14],
    [11, 12, 15, 16],
    [17, 18, 21, 22],
    [19, 20, 23, 24],
];

/// `av1_get_force_skip_low_temp_var_small_sb` (var_based_part.c:852) — the
/// SB64 lookup into `PartitionSearchInfo::variance_low`.
///
/// C names the two relative indices `mi_x = mi_row & 0xF` and
/// `mi_y = mi_col & 0xF` — note the SWAP: `mi_x` comes from the ROW. The
/// swapped names propagate into every branch below, so they are kept rather
/// than "corrected", and the two 32x32 sub-cases that look transposed
/// (`mi_y && !mi_x` -> slot 6, `!mi_y && mi_x` -> slot 7) are C's.
///
/// Returns 0 for any `bsize` C's `switch` does not name.
#[must_use]
pub fn get_force_skip_low_temp_var_small_sb(
    variance_low: &[u8],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> i32 {
    // Relative indices of the MB inside the superblock.
    let mi_x = mi_row & 0xF;
    let mi_y = mi_col & 0xF;
    // Relative indices of the 16x16 block inside the superblock.
    let i = (mi_x >> 2) as usize;
    let j = (mi_y >> 2) as usize;
    let at = |k: usize| i32::from(variance_low[k]);
    match bsize {
        BLOCK_64X64 => at(0),
        BLOCK_64X32 => match (mi_y == 0, mi_x == 0) {
            (true, true) => at(1),
            (true, false) => at(2),
            _ => 0,
        },
        BLOCK_32X64 => match (mi_y == 0, mi_x == 0) {
            (true, true) => at(3),
            (false, true) => at(4),
            _ => 0,
        },
        BLOCK_32X32 => match (mi_y == 0, mi_x == 0) {
            (true, true) => at(5),
            (false, true) => at(6),
            (true, false) => at(7),
            (false, false) => at(8),
        },
        BLOCK_32X16 | BLOCK_16X32 | BLOCK_16X16 => at(POS_SHIFT_16X16[i][j]),
        _ => 0,
    }
}

/// `av1_get_force_skip_low_temp_var` (var_based_part.c:901) — the SB128
/// lookup into the same array.
///
/// The three index derivations carry an upstream oddity that libaom's own
/// commented-out lines document: the intended `(y << 1) + x` was replaced by
/// `y + x` with the row masks narrowed to `0x17`, `0xB` and `0x5`. Those masks
/// are not powers of two minus one, so `idx64` / `idx32` / `idx16` are NOT the
/// raster indices the comments describe. Reproduced verbatim; "fixing" them
/// changes which flag every 32x32 and 16x16 block reads.
#[must_use]
pub fn get_force_skip_low_temp_var(
    variance_low: &[u8],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> i32 {
    let idx64 = (((mi_row & 0x17) >> 3) + ((mi_col & 0x1F) >> 4)) as usize;
    let idx32 = (((mi_row & 0xB) >> 2) + ((mi_col & 0xF) >> 3)) as usize;
    let idx16 = (((mi_row & 0x5) >> 1) + ((mi_col & 0x7) >> 2)) as usize;
    let at = |k: usize| i32::from(variance_low[k]);
    match bsize {
        BLOCK_128X128 => at(0),
        BLOCK_128X64 => {
            debug_assert_eq!(mi_col & 0x1F, 0);
            at(1 + usize::from((mi_row & 0x1F) != 0))
        }
        BLOCK_64X128 => {
            debug_assert_eq!(mi_row & 0x1F, 0);
            at(3 + usize::from((mi_col & 0x1F) != 0))
        }
        BLOCK_64X64 => at(5 + idx64),
        BLOCK_64X32 => {
            let x = (mi_col & 0x1F) >> 4;
            let y = (mi_row & 0x1F) >> 3;
            let idx64x32 = ((x << 1) + (y % 2) + ((y >> 1) << 2)) as usize;
            at(9 + idx64x32)
        }
        BLOCK_32X64 => {
            let x = (mi_col & 0x1F) >> 3;
            let y = (mi_row & 0x1F) >> 4;
            at(17 + ((y << 2) + x) as usize)
        }
        BLOCK_32X32 => at(25 + (idx64 << 2) + idx32),
        BLOCK_32X16 | BLOCK_16X32 | BLOCK_16X16 => at(41 + (idx64 << 4) + (idx32 << 2) + idx16),
        _ => 0,
    }
}

/// `is_set_force_zeromv_skip_based_on_src_sad` (var_based_part.c:1549) —
/// whether the source SAD is low enough that the speed feature's level lets
/// the whole superblock be coded as a zero-MV skip.
#[must_use]
pub fn is_set_force_zeromv_skip_based_on_src_sad(
    set_zeromv_skip_based_on_source_sad: i32,
    source_sad_nonrd: SourceSad,
) -> bool {
    match set_zeromv_skip_based_on_source_sad {
        0 => false,
        n if n >= 3 => source_sad_nonrd <= SourceSad::Low,
        2 => source_sad_nonrd <= SourceSad::VeryLow,
        1 => source_sad_nonrd == SourceSad::Zero,
        // C's cascade is `>= 3`, `>= 2`, `>= 1`, else false, so any NEGATIVE
        // level falls through to false along with 0.
        _ => false,
    }
}

// ===========================================================================
// The low-temporal-variance flag setters (var_based_part.c:691-846).
//
// These WRITE the `variance_low` array that
// [`get_force_skip_low_temp_var`] / [`get_force_skip_low_temp_var_small_sb`]
// read back, so the pair only means something once both halves exist. A
// superblock whose temporal variance is low enough gets its inter search
// short-circuited at the corresponding block size.
//
// | Rust | C |
// |---|---|
// | [`VarianceTree`] | the `VP128x128` node fields these read |
// | [`set_low_temp_var_flag_64x64`] | `set_low_temp_var_flag_64x64` (:691) |
// | [`set_low_temp_var_flag_128x128`] | `set_low_temp_var_flag_128x128` (:744) |
// | [`set_low_temp_var_flag`] | `set_low_temp_var_flag` (:829) |
// ===========================================================================

/// `PartitionSearchInfo::variance_low`'s length (`block.h`).
pub const VARIANCE_LOW_LEN: usize = 105;

/// The variance-tree node values the low-temp-var setters read, in the layout
/// the oracle boundary uses.
///
/// C walks a live `VP128x128`; only these 105 fields are ever read, so the
/// port carries exactly them:
/// * `l0` — the 128x128 node's `none, horz[0], horz[1], vert[0], vert[1]`;
/// * `l1[i]` — the four 64x64 nodes, same five fields each;
/// * `l2[i]` — the sixteen 32x32 nodes, `none` only;
/// * `l3[i]` — the sixty-four 16x16 nodes, `none` only.
///
/// On the SB64 path C passes `&vt->split[0]`, so only `l1[0]` and its
/// descendants (`l2[0..4]`, `l3[0..16]`) are read.
#[derive(Clone, Copy, Debug)]
pub struct VarianceTree {
    /// The 128x128 node: `none, horz[0], horz[1], vert[0], vert[1]`.
    pub l0: [i32; 5],
    /// The four 64x64 nodes, same five fields each.
    pub l1: [[i32; 5]; 4],
    /// The sixteen 32x32 nodes' `none` variance.
    pub l2: [i32; 16],
    /// The sixty-four 16x16 nodes' `none` variance.
    pub l3: [i32; 64],
}

impl Default for VarianceTree {
    fn default() -> Self {
        Self {
            l0: [0; 5],
            l1: [[0; 5]; 4],
            l2: [0; 16],
            l3: [0; 64],
        }
    }
}

/// The mi grid the setters consult, as `Option<BLOCK_SIZE>` per cell.
///
/// C reads `mi_params->mi_grid_base[idx]`, which is a POINTER and can be
/// NULL; both setters check for it before dereferencing. `None` is that NULL.
#[derive(Clone, Debug)]
pub struct MiGrid<'a> {
    /// One entry per mi cell, indexed `mi_stride * row + col`.
    pub bsize: &'a [Option<usize>],
    /// `mi_params->mi_stride`.
    pub mi_stride: usize,
    /// `mi_params->mi_rows`.
    pub mi_rows: usize,
    /// `mi_params->mi_cols`.
    pub mi_cols: usize,
}

impl MiGrid<'_> {
    /// `mi_grid_base[mi_stride * row + col]`, or `None` where C would read a
    /// NULL pointer or index outside the array the caller supplied.
    fn at(&self, row: usize, col: usize) -> Option<usize> {
        self.bsize
            .get(row * self.mi_stride + col)
            .copied()
            .flatten()
    }
}

/// `set_low_temp_var_flag_64x64` (var_based_part.c:691) — the SB64 arm.
///
/// The four `variance_low` regions it can write are disjoint by block size:
/// slot 0 for a whole 64x64, 1-2 for 64x32, 3-4 for 32x64, and for a split
/// superblock either 5-8 (a 32x32 leaf) or 9-24 (16x16 leaves inside a 32x32
/// that itself split). Nothing else is touched, so `variance_low` is in/out.
///
/// Three thresholds, three shifts, and they are NOT uniform: `>> 1` for the
/// whole block, `>> 2` for the halves, `(5 * t) >> 3` for a 32x32 leaf and
/// `>> 8` for a 16x16 one. Each is C's.
pub fn set_low_temp_var_flag_64x64(
    grid: &MiGrid<'_>,
    variance_low: &mut [u8; VARIANCE_LOW_LEN],
    cur_bsize: usize,
    vt: &VarianceTree,
    thresholds: &[i64; 5],
    mi_col: usize,
    mi_row: usize,
) {
    // C receives `&vt->split[0]`, so the "64x64 node" here is l1[0] and its
    // children are l2[0..4] / l3[0..16].
    let node = &vt.l1[0];
    if cur_bsize == BLOCK_64X64 {
        if i64::from(node[0]) < (thresholds[0] >> 1) {
            variance_low[0] = 1;
        }
    } else if cur_bsize == BLOCK_64X32 {
        for part_idx in 0..2 {
            if i64::from(node[1 + part_idx]) < (thresholds[0] >> 2) {
                variance_low[part_idx + 1] = 1;
            }
        }
    } else if cur_bsize == BLOCK_32X64 {
        for part_idx in 0..2 {
            if i64::from(node[3 + part_idx]) < (thresholds[0] >> 2) {
                variance_low[part_idx + 3] = 1;
            }
        }
    } else {
        const IDX: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
        for (lvl1_idx, &(dr, dc)) in IDX.iter().enumerate() {
            if grid.mi_cols <= mi_col + dc || grid.mi_rows <= mi_row + dr {
                continue;
            }
            let Some(this_bsize) = grid.at(mi_row + dr, mi_col + dc) else {
                continue;
            };
            if this_bsize == BLOCK_32X32 {
                let threshold_32x32 = (5 * thresholds[1]) >> 3;
                if i64::from(vt.l2[lvl1_idx]) < threshold_32x32 {
                    variance_low[lvl1_idx + 5] = 1;
                }
            } else if this_bsize == BLOCK_16X16
                || this_bsize == BLOCK_32X16
                || this_bsize == BLOCK_16X32
            {
                // For 32x16 and 16x32 the flag is set on each 16x16 inside.
                for lvl2_idx in 0..4 {
                    if i64::from(vt.l3[lvl1_idx * 4 + lvl2_idx]) < (thresholds[2] >> 8) {
                        variance_low[(lvl1_idx << 2) + lvl2_idx + 9] = 1;
                    }
                }
            }
        }
    }
}

/// `set_low_temp_var_flag_128x128` (var_based_part.c:744) — the SB128 arm.
///
/// Same shape one level deeper, and the two NULL/bounds checks are in the
/// OPPOSITE order to the SB64 arm's: C dereferences `mi_64` for its NULL test
/// BEFORE the mi_cols/mi_rows bounds test, and the reverse at the 32 level.
/// That ordering is reproduced — it decides which of the two `continue`s a
/// cell takes, which is unobservable here but is the sort of thing a later
/// reader would "tidy".
pub fn set_low_temp_var_flag_128x128(
    grid: &MiGrid<'_>,
    variance_low: &mut [u8; VARIANCE_LOW_LEN],
    cur_bsize: usize,
    vt: &VarianceTree,
    thresholds: &[i64; 5],
    mi_col: usize,
    mi_row: usize,
) {
    if cur_bsize == BLOCK_128X128 {
        if i64::from(vt.l0[0]) < (thresholds[0] >> 1) {
            variance_low[0] = 1;
        }
    } else if cur_bsize == BLOCK_128X64 {
        for part_idx in 0..2 {
            if i64::from(vt.l0[1 + part_idx]) < (thresholds[0] >> 2) {
                variance_low[part_idx + 1] = 1;
            }
        }
    } else if cur_bsize == BLOCK_64X128 {
        for part_idx in 0..2 {
            if i64::from(vt.l0[3 + part_idx]) < (thresholds[0] >> 2) {
                variance_low[part_idx + 3] = 1;
            }
        }
    } else {
        const IDX64: [(usize, usize); 4] = [(0, 0), (0, 16), (16, 0), (16, 16)];
        const IDX32: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
        for (lvl1_idx, &(dr64, dc64)) in IDX64.iter().enumerate() {
            let Some(bsize_64) = grid.at(mi_row + dr64, mi_col + dc64) else {
                continue;
            };
            if grid.mi_cols <= mi_col + dc64 || grid.mi_rows <= mi_row + dr64 {
                continue;
            }
            let threshold_64x64 = (5 * thresholds[1]) >> 3;
            let node = &vt.l1[lvl1_idx];
            if bsize_64 == BLOCK_64X64 {
                if i64::from(node[0]) < threshold_64x64 {
                    variance_low[5 + lvl1_idx] = 1;
                }
            } else if bsize_64 == BLOCK_64X32 {
                for part_idx in 0..2 {
                    if i64::from(node[1 + part_idx]) < (threshold_64x64 >> 1) {
                        variance_low[9 + (lvl1_idx << 1) + part_idx] = 1;
                    }
                }
            } else if bsize_64 == BLOCK_32X64 {
                for part_idx in 0..2 {
                    if i64::from(node[3 + part_idx]) < (threshold_64x64 >> 1) {
                        variance_low[17 + (lvl1_idx << 1) + part_idx] = 1;
                    }
                }
            } else {
                for (lvl2_idx, &(dr32, dc32)) in IDX32.iter().enumerate() {
                    let Some(bsize_32) = grid.at(mi_row + dr64 + dr32, mi_col + dc64 + dc32) else {
                        continue;
                    };
                    if grid.mi_cols <= mi_col + dc64 + dc32 || grid.mi_rows <= mi_row + dr64 + dr32
                    {
                        continue;
                    }
                    let threshold_32x32 = (5 * thresholds[2]) >> 3;
                    if bsize_32 == BLOCK_32X32 {
                        if i64::from(vt.l2[lvl1_idx * 4 + lvl2_idx]) < threshold_32x32 {
                            variance_low[25 + (lvl1_idx << 2) + lvl2_idx] = 1;
                        }
                    } else if bsize_32 == BLOCK_16X16
                        || bsize_32 == BLOCK_32X16
                        || bsize_32 == BLOCK_16X32
                    {
                        for lvl3_idx in 0..4 {
                            let v = vt.l3[(lvl1_idx * 16) + (lvl2_idx * 4) + lvl3_idx];
                            if i64::from(v) < (thresholds[3] >> 8) {
                                variance_low[41 + (lvl1_idx << 4) + (lvl2_idx << 2) + lvl3_idx] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `set_low_temp_var_flag` (var_based_part.c:829) — the dispatcher.
///
/// Only LAST_FRAME partitions get temporal-variance flags at all: the check is
/// against a reference the encoder has actually reconstructed at the same
/// position, and GOLDEN/ALTREF are too far away for the comparison to mean
/// anything. Every other reference leaves `variance_low` untouched.
#[allow(clippy::too_many_arguments)]
pub fn set_low_temp_var_flag(
    grid: &MiGrid<'_>,
    variance_low: &mut [u8; VARIANCE_LOW_LEN],
    cur_bsize: usize,
    vt: &VarianceTree,
    thresholds: &[i64; 5],
    ref_frame_partition: usize,
    mi_col: usize,
    mi_row: usize,
    is_small_sb: bool,
) {
    /// `LAST_FRAME`.
    const LAST_FRAME: usize = 1;
    if ref_frame_partition != LAST_FRAME {
        return;
    }
    if is_small_sb {
        set_low_temp_var_flag_64x64(
            grid,
            variance_low,
            cur_bsize,
            vt,
            thresholds,
            mi_col,
            mi_row,
        );
    } else {
        set_low_temp_var_flag_128x128(
            grid,
            variance_low,
            cur_bsize,
            vt,
            thresholds,
            mi_col,
            mi_row,
        );
    }
}

// ===========================================================================
// The two remaining INTER decisions of var_based_part.c.
//
// Both C functions mix a decision with buffer plumbing the port replaces
// rather than translates (`set_block_size` writing the mi grid, `aom_free(vt)`
// freeing the variance tree, `av1_setup_pre_planes` re-pointing the predictor
// planes). What is ported here is the DECISION; the plumbing has no Rust
// counterpart to compare against, and the gate says so.
//
// | Rust | C |
// |---|---|
// | [`ZeromvSkip`] / [`set_force_zeromv_skip_for_sb`] | `set_force_zeromv_skip_for_sb` (:1563) |
// | [`RefFrameForPartition`] / [`set_ref_frame_for_partition`] | `set_ref_frame_for_partition` (:1219) |
// ===========================================================================

/// `CALC_CHROMA_THRESH_FOR_ZEROMV_SKIP` (`var_based_part.h:35`).
#[inline]
#[must_use]
pub fn calc_chroma_thresh_for_zeromv_skip(thresh_exit_part: u32) -> u32 {
    (3 * thresh_exit_part) >> 2
}

/// What `set_force_zeromv_skip_for_sb` decides.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ZeromvSkip {
    /// C's return: the whole superblock is stamped at `bsize` and the
    /// partitioner exits.
    pub exit_partitioning: bool,
    /// `x->force_zeromv_skip_for_sb`, which C sets to 1 on the exit path and
    /// to 2 on a separate, WEAKER condition that does not exit.
    pub force_zeromv_skip_for_sb: i32,
}

/// `set_force_zeromv_skip_for_sb` (var_based_part.c:1563) — whether the whole
/// superblock can be coded as a zero-MV skip.
///
/// Three things have to hold together: the source SAD is low enough for the
/// speed feature's level ([`is_set_force_zeromv_skip_based_on_src_sad`]), the
/// superblock fits inside the tile, and all three plane SADs are under their
/// thresholds. The chroma threshold is three-quarters of the luma one, and is
/// divided by a further EIGHT when the source SAD is at least `VeryLow` and
/// `part_early_exit_zeromv` is exactly 1 — C's comment says that arm exists to
/// suppress a visual artefact the level-2 speed feature causes.
///
/// The `else if` arm is a genuinely different outcome, not a fallback: a
/// completely static superblock at `part_early_exit_zeromv >= 2` gets
/// `force_zeromv_skip_for_sb = 2` WITHOUT exiting the partitioner.
///
/// `mi_size_wide` / `mi_size_high` are of the SB size, not of `bsize`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn set_force_zeromv_skip_for_sb(
    set_zeromv_skip_based_on_source_sad: i32,
    source_sad_nonrd: SourceSad,
    increase_source_sad_thresh: bool,
    part_early_exit_zeromv: i32,
    sb_mi_width: i32,
    sb_mi_height: i32,
    thresh_exit_part_y_cfg: u32,
    mi_row: i32,
    mi_col: i32,
    tile_mi_row_end: i32,
    tile_mi_col_end: i32,
    y_sad: u32,
    uv_sad: [u32; 2],
) -> ZeromvSkip {
    if !is_set_force_zeromv_skip_based_on_src_sad(
        set_zeromv_skip_based_on_source_sad,
        source_sad_nonrd,
    ) {
        return ZeromvSkip {
            exit_partitioning: false,
            force_zeromv_skip_for_sb: 0,
        };
    }
    let shift = u32::from(increase_source_sad_thresh);
    let thresh_exit_part_y = thresh_exit_part_y_cfg << shift;
    let mut thresh_exit_part_uv = calc_chroma_thresh_for_zeromv_skip(thresh_exit_part_y) << shift;
    // Be more aggressive on chroma at source_sad >= VeryLow, to suppress the
    // artefact set_zeromv_skip_based_on_source_sad = 2 can cause. Only for
    // part_early_exit_zeromv == 1.
    if source_sad_nonrd >= SourceSad::VeryLow && part_early_exit_zeromv == 1 {
        thresh_exit_part_uv >>= 3;
    }
    if mi_col + sb_mi_width <= tile_mi_col_end
        && mi_row + sb_mi_height <= tile_mi_row_end
        && y_sad < thresh_exit_part_y
        && uv_sad[0] < thresh_exit_part_uv
        && uv_sad[1] < thresh_exit_part_uv
    {
        // C also stamps the block size into the mi grid and frees the variance
        // tree here; both are plumbing the port does by other means.
        return ZeromvSkip {
            exit_partitioning: true,
            force_zeromv_skip_for_sb: 1,
        };
    }
    if source_sad_nonrd == SourceSad::Zero && part_early_exit_zeromv >= 2 {
        return ZeromvSkip {
            exit_partitioning: false,
            force_zeromv_skip_for_sb: 2,
        };
    }
    ZeromvSkip {
        exit_partitioning: false,
        force_zeromv_skip_for_sb: 0,
    }
}

/// What `set_ref_frame_for_partition` decides.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RefFrameForPartition {
    /// The reference the partitioner will measure against.
    pub ref_frame_partition: i32,
    /// The SAD that goes with it.
    pub y_sad: u32,
    /// `x->nonrd_prune_ref_frame_search`.
    pub nonrd_prune_ref_frame_search: i32,
    /// `x->sb_me_partition`, which the two non-LAST arms clear and the LAST
    /// arm does NOT touch. `None` means "left as it was".
    pub sb_me_partition: Option<i32>,
}

/// `set_ref_frame_for_partition` (var_based_part.c:1219) — pick the reference
/// the variance partitioner measures against.
///
/// GOLDEN or ALTREF wins only by a MARGIN: its SAD has to beat `fac` times
/// LAST's, where `fac` is 0.9 except on an SVC enhancement layer that has a
/// lower-quality layer below it, where it is 1.0. Both tests also require
/// beating the other candidate outright, so the two cannot both fire.
///
/// On either non-LAST arm the reference-pruning speed feature is FORCED to 0
/// and `sb_me_partition` is cleared; on the LAST arm the speed feature's own
/// value is used and `sb_me_partition` is left alone.
///
/// C additionally calls `av1_setup_pre_planes` and writes `mi->ref_frame[0]` /
/// `mi->mv[0]` on the two non-LAST arms; that is predictor plumbing with no
/// Rust counterpart here.
#[must_use]
pub fn set_ref_frame_for_partition(
    spatial_layer_id: i32,
    has_lower_quality_layer: bool,
    y_sad: u32,
    y_sad_g: u32,
    y_sad_alt: u32,
    nonrd_prune_ref_frame_search_cfg: i32,
) -> RefFrameForPartition {
    /// `LAST_FRAME`.
    const LAST_FRAME: i32 = 1;
    /// `GOLDEN_FRAME`.
    const GOLDEN_FRAME: i32 = 4;
    /// `ALTREF_FRAME`.
    const ALTREF_FRAME: i32 = 7;

    let fac = if spatial_layer_id > 0 && has_lower_quality_layer {
        1.0
    } else {
        0.9
    };
    // C compares `unsigned int < double`, so the left side is converted to
    // double; the product is not truncated back to an integer first.
    let is_set_golden = f64::from(y_sad_g) < fac * f64::from(y_sad) && y_sad_g < y_sad_alt;
    let is_set_altref = f64::from(y_sad_alt) < fac * f64::from(y_sad) && y_sad_alt < y_sad_g;

    if is_set_golden {
        RefFrameForPartition {
            ref_frame_partition: GOLDEN_FRAME,
            y_sad: y_sad_g,
            nonrd_prune_ref_frame_search: 0,
            sb_me_partition: Some(0),
        }
    } else if is_set_altref {
        RefFrameForPartition {
            ref_frame_partition: ALTREF_FRAME,
            y_sad: y_sad_alt,
            nonrd_prune_ref_frame_search: 0,
            sb_me_partition: Some(0),
        }
    } else {
        RefFrameForPartition {
            ref_frame_partition: LAST_FRAME,
            y_sad,
            nonrd_prune_ref_frame_search: nonrd_prune_ref_frame_search_cfg,
            sb_me_partition: None,
        }
    }
}
