//! Inter-block decode for `TileKf` — the `decodemv.c` / `reconinter.c` side of
//! the per-block interleave: `decode_block_inter` (mode-info parse -> MV-ref
//! resolution -> predictor dispatch -> residual recon), the OBMC blend
//! helpers, the motion-mode ceiling, the CDF stamping (`stamp_interp`,
//! `stamp_frame_mvs`), and the MV-clamp / geometry helpers.
//!
//! `impl TileKf` is split across files: this module is a child of the crate
//! root, so it sees `TileKf`'s private fields and the root's private helpers.
//! `decode_block_inter` is `pub(crate)` because `decode_block` (the intra/inter
//! dispatcher) stays in the root; everything else here is module-private.

use super::*;

// PREDICTION_MODE inter values (enums.h): NEARESTMV=13 .. NEWMV=16 are the
// single-ref inter modes; SWITCHABLE is the frame interp_filter sentinel.
const NEARESTMV: i32 = 13;
const NEARMV: i32 = 14;
const GLOBALMV: i32 = 15;
const NEWMV: i32 = 16;
const SWITCHABLE: i32 = 4;

/// The inter mode-info fields `read_inter_mode_info` produces for the
/// predict + reconstruct stages — C's `MB_MODE_INFO` payload for an inter
/// block (decodemv.c `read_inter_block_mode_info` writes into `mbmi`; the
/// port passes the values through explicitly, and structures the mutually
/// exclusive pieces so states C merely never produces are inexpressible:
/// a single-ref block cannot carry a second ref/MV, compound weights, or an
/// inter-intra blend).
#[derive(Clone, Copy)]
struct InterModeInfo {
    /// `ref_frame[0]` — the primary (or only) reference, 1..=7.
    ref0: i32,
    /// PREDICTION_MODE after the inter-mode read (`mode` in decodemv.c).
    mode: i32,
    /// ref0's resolved `(mv_row, mv_col)`.
    mv_row: i32,
    mv_col: i32,
    /// `read_ref_frames`'s compound pair + `read_compound_type_info`'s blend
    /// — `Some` iff the block is compound (the `is_compound` flag it
    /// replaces is `compound.is_some()` at every use site).
    compound: Option<Compound>,
    /// `read_interintra_info` — `Some` iff the block blends its inter
    /// predictor with an intra one (single-ref blocks only, per
    /// `is_interintra_allowed`).
    interintra: Option<InterIntra>,
    /// `read_motion_mode` result (SIMPLE_TRANSLATION/OBMC_CAUSAL/WARPED_CAUSAL).
    motion_mode: i32,
    /// The derived local WARPED_CAUSAL model, `None` when invalid/absent.
    warp_luma: Option<aom_dsp::inter::warp::WarpedMotionParams>,
    /// `read_mb_interp_filter`/`set_default_interp_filters`: (y, x) filters.
    filter_y: usize,
    filter_x: usize,
    /// The block's luma tx size (var-tx leaf size under TX_MODE_SELECT).
    tx_size: usize,
}

/// The second half of a compound block's `MB_MODE_INFO`: `ref_frame[1]` and
/// its MV plus `read_compound_type_info`'s output. `comp_group_idx` /
/// `compound_idx` are the entropy context values — recon stamps them into the
/// neighbour grid for later blocks regardless of which blend `kind` selects.
#[derive(Clone, Copy)]
struct Compound {
    /// `ref_frame[1]`, 1..=7.
    ref1: i32,
    /// ref1's resolved `(mv_row, mv_col)`.
    mv1_row: i32,
    mv1_col: i32,
    /// `comp_group_idx`: 0 = average/dist-wtd family, 1 = masked family.
    comp_group_idx: i32,
    /// `compound_idx`: within group 0, 0 = distance-weighted, 1 = average.
    compound_idx: i32,
    /// The predictor family `comp_group_idx` selects.
    kind: CompoundKind,
}

/// `read_compound_type_info`'s two mutually exclusive blend families — C
/// branches on `comp_group_idx` at every build site; the port makes the two
/// payloads one enum so a masked block cannot also carry weights.
#[derive(Clone, Copy)]
enum CompoundKind {
    /// Group 0: `build_compound_inter_predictor` with the
    /// `av1_dist_wtd_comp_weight_assign` weights (or the plain 8/8 average).
    Weighted(aom_dsp::inter::compound::DistWtdWeights),
    /// Group 1: `build_masked_compound_inter_predictor` (wedge / diffwtd).
    Masked(aom_dsp::inter::MaskedCompound),
}

impl InterModeInfo {
    /// `ref_frame[1]` as C stores it in `mbmi` — the compound pair's second
    /// ref, `INTRA_FRAME` (0) when the block is inter-intra (`read_interintra_info`
    /// rewrites `ref_frame[1]` so the intra predictor reads as a ref), or
    /// `NONE_FRAME` (-1) for a plain single-ref block. Recon stamps this into
    /// the neighbour grid, where later blocks' ref contexts read it.
    fn ref1(&self) -> i32 {
        self.compound
            .map_or(if self.interintra.is_some() { 0 } else { -1 }, |c| c.ref1)
    }
}

/// `read_interintra_info`'s four outputs as one value — the wedge fields are
/// meaningful only together and only when the blend is active.
#[derive(Clone, Copy)]
struct InterIntra {
    /// INTERINTRA_MODE index (the blend-mask family).
    mode: i32,
    /// Wedge blend active (then `wedge_idx` selects the codebook mask).
    use_wedge: bool,
    /// `wedge_interintra` index.
    wedge_idx: i32,
}

impl<'c> TileKf<'c> {
    /// [`Self::stamp_dv`]'s interp-filter twin — stamp the block's resolved
    /// `(y_filter, x_filter)` over its frame-cropped mi footprint so later
    /// switchable inter blocks' `av1_get_pred_context_switchable_interp`
    /// neighbour reads see it (`cm->mi[..]->interp_filters`).
    fn stamp_interp(&mut self, mi_row: i32, mi_col: i32, bsize: usize, cell: (u8, u8)) {
        let x_mis = MI_SIZE_WIDE[bsize].min(self.cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(self.cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * self.cfg.mi_cols + mi_col) as usize;
            self.mi_interp[base..base + x_mis as usize].fill(cell);
        }
    }

    /// `av1_copy_frame_mvs` (mvref_common.c:41): store an inter block's motion
    /// into the per-8x8 frame MV grid (`cur_frame->mvs`) consumed by LATER
    /// frames' temporal motion-field projection (`av1_setup_motion_field`).
    /// Per 8x8 cell: `NONE` unless one of the block's refs is a non-future ref
    /// (`ref_frame_side[ref] == 0`) with both MV components within
    /// `REFMVS_LIMIT` — C iterates `idx = 0, 1`, the last qualifying ref wins.
    /// Intra blocks store `NONE`, which equals the grid's init value, so only
    /// the inter arm stamps. The stored MV is the CODED `mi->mv` (pre the
    /// MC-only UMV border clamp).
    #[allow(clippy::too_many_arguments)]
    fn stamp_frame_mvs(
        &mut self,
        inter: &InterFrameCfg,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        refs: [i32; 2],
        mvs: [(i32, i32); 2],
    ) {
        const REFMVS_LIMIT: i32 = (1 << 12) - 1;
        let stride = ((self.cfg.mi_cols + 1) >> 1) as usize;
        let x_mis = ((MI_SIZE_WIDE[bsize].min(self.cfg.mi_cols - mi_col) + 1) >> 1) as usize;
        let y_mis = ((MI_SIZE_HIGH[bsize].min(self.cfg.mi_rows - mi_row) + 1) >> 1) as usize;
        let mut cell = MvRefCell::default();
        for idx in 0..2 {
            let rf = refs[idx];
            if rf > 0 {
                if inter.ref_frame_side[rf as usize] != 0 {
                    continue;
                }
                let (mr, mc) = mvs[idx];
                if mr.abs() > REFMVS_LIMIT || mc.abs() > REFMVS_LIMIT {
                    continue;
                }
                cell = MvRefCell {
                    row: mr as i16,
                    col: mc as i16,
                    ref_frame: rf as i8,
                };
            }
        }
        let base_row = (mi_row >> 1) as usize;
        let base_col = (mi_col >> 1) as usize;
        for r in 0..y_mis {
            let b = (base_row + r) * stride + base_col;
            self.frame_mvs[b..b + x_mis].fill(cell);
        }
    }

    /// `foreach_overlappable_nb_above` (obmc.h): walk the mi row above the block
    /// across its width, stepping by each above-neighbour's mi width (a width-4
    /// block is pair-adjusted to its chroma-carrying second half), collecting
    /// inter (`is_neighbor_overlappable`) neighbours up to `nb_max`. Returns
    /// `(rel_mi_col, op_mi_size, neighbour)` per hit.
    fn overlappable_above(
        &self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        nb_max: i32,
    ) -> Vec<(i32, i32, DvNbr, (u8, u8))> {
        let cols = self.cfg.mi_cols;
        let mut out = Vec::new();
        if mi_row <= self.tile.mi_row_start {
            return out; // !up_available
        }
        // mi_size_wide[BLOCK_64X64] = 16.
        const MI64: i32 = 16;
        let width = MI_SIZE_WIDE[bsize];
        let end_col = (mi_col + width).min(cols);
        let mut amc = mi_col;
        while amc < end_col && (out.len() as i32) < nb_max {
            let d0 = DvNbr::from_packed(self.mi_dv[((mi_row - 1) * cols + amc) as usize]);
            let mut mi_step = MI_SIZE_WIDE[d0.bsize].min(MI64);
            // The neighbour mbmi (and its coded interp filter, for the OBMC strip
            // MC — C uses `above_mbmi->interp_filters`) come from the SAME grid cell.
            let (nb, nb_if) = if mi_step == 1 {
                amc &= !1;
                mi_step = 2;
                let idx = ((mi_row - 1) * cols + amc + 1) as usize;
                (DvNbr::from_packed(self.mi_dv[idx]), self.mi_interp[idx])
            } else {
                (d0, self.mi_interp[((mi_row - 1) * cols + amc) as usize])
            };
            if nb.use_intrabc || nb.ref_frame0 > 0 {
                out.push((amc - mi_col, width.min(mi_step), nb, nb_if));
            }
            amc += mi_step;
        }
        out
    }

    /// `foreach_overlappable_nb_left` (obmc.h): the left-column twin of
    /// [`Self::overlappable_above`]. Returns `(rel_mi_row, op_mi_size, neighbour)`.
    fn overlappable_left(
        &self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        nb_max: i32,
    ) -> Vec<(i32, i32, DvNbr, (u8, u8))> {
        let cols = self.cfg.mi_cols;
        let mut out = Vec::new();
        if mi_col <= self.tile.mi_col_start {
            return out; // !left_available
        }
        const MI64: i32 = 16;
        let height = MI_SIZE_HIGH[bsize];
        let end_row = (mi_row + height).min(self.cfg.mi_rows);
        let mut amr = mi_row;
        while amr < end_row && (out.len() as i32) < nb_max {
            let d0 = DvNbr::from_packed(self.mi_dv[(amr * cols + mi_col - 1) as usize]);
            let mut mi_step = MI_SIZE_HIGH[d0.bsize].min(MI64);
            let (nb, nb_if) = if mi_step == 1 {
                amr &= !1;
                mi_step = 2;
                let idx = ((amr + 1) * cols + mi_col - 1) as usize;
                (DvNbr::from_packed(self.mi_dv[idx]), self.mi_interp[idx])
            } else {
                (d0, self.mi_interp[(amr * cols + mi_col - 1) as usize])
            };
            if nb.use_intrabc || nb.ref_frame0 > 0 {
                out.push((amr - mi_row, height.min(mi_step), nb, nb_if));
            }
            amr += mi_step;
        }
        out
    }

    /// `av1_count_overlappable_neighbors` (reconinter.c:801): whether the block
    /// has any overlappable (inter) above/left neighbour — the OBMC/warp
    /// motion-mode gate. Only `!= 0` is consulted by `motion_mode_allowed`.
    fn has_overlappable_neighbors(&self, mi_row: i32, mi_col: i32, bsize: usize) -> bool {
        if !is_motion_variation_allowed_bsize(bsize) {
            return false;
        }
        if !self
            .overlappable_above(mi_row, mi_col, bsize, i32::MAX)
            .is_empty()
        {
            return true;
        }
        !self
            .overlappable_left(mi_row, mi_col, bsize, i32::MAX)
            .is_empty()
    }

    /// `av1_findSamples` (mvref_common.c:1118), count-only: the number of warp
    /// projection samples = above/left/top-left neighbours whose single reference
    /// matches `ref_frame` (`ref[0] == ref_frame && ref[1] == NONE`), capped at
    /// `LEAST_SQUARES_SAMPLES_MAX = 8`. Feeds the `num_proj_ref >= 1` arm of the
    /// motion-mode ceiling. The TOP-RIGHT scan (mvref_common.c:1223, gated by
    /// `has_top_right`) is not ported: it only ADDS samples, so the ceiling's
    /// `>= 1` test is unaffected whenever the above/left/top-left scan already
    /// yields a sample — true for every OBMC block in the chunk-4 target (whose
    /// OBMC block is at the frame's left+right edges, so its 4 samples all come
    /// from the above scan). The exact count + `record_samples` for warp land
    /// with chunk 5 (WARPED_CAUSAL).
    fn num_proj_ref(&self, mi_row: i32, mi_col: i32, bsize: usize, ref_frame: i32) -> i32 {
        const MAX: i32 = 8; // LEAST_SQUARES_SAMPLES_MAX
        let cols = self.cfg.mi_cols;
        let rows = self.cfg.mi_rows;
        let up = mi_row > self.tile.mi_row_start;
        let left = mi_col > self.tile.mi_col_start;
        let width = MI_SIZE_WIDE[bsize];
        let height = MI_SIZE_HIGH[bsize];
        let matches = |d: &DvNbr| d.ref_frame0 == ref_frame && d.ref_frame1 == -1;
        let mut np = 0i32;
        let mut do_tl = true;

        if up {
            let d0 = DvNbr::from_packed(self.mi_dv[((mi_row - 1) * cols + mi_col) as usize]);
            let sbw = MI_SIZE_WIDE[d0.bsize];
            if width <= sbw {
                let col_offset = (-mi_col) % sbw;
                if col_offset < 0 {
                    do_tl = false;
                }
                if matches(&d0) {
                    np += 1;
                    if np >= MAX {
                        return MAX;
                    }
                }
            } else {
                let end = width.min(cols - mi_col);
                let mut i = 0;
                while i < end {
                    let d =
                        DvNbr::from_packed(self.mi_dv[((mi_row - 1) * cols + mi_col + i) as usize]);
                    if matches(&d) {
                        np += 1;
                        if np >= MAX {
                            return MAX;
                        }
                    }
                    i += MI_SIZE_WIDE[d.bsize];
                }
            }
        }

        if left {
            let d0 = DvNbr::from_packed(self.mi_dv[(mi_row * cols + mi_col - 1) as usize]);
            let sbh = MI_SIZE_HIGH[d0.bsize];
            if height <= sbh {
                let row_offset = (-mi_row) % sbh;
                if row_offset < 0 {
                    do_tl = false;
                }
                if matches(&d0) {
                    np += 1;
                    if np >= MAX {
                        return MAX;
                    }
                }
            } else {
                let end = height.min(rows - mi_row);
                let mut i = 0;
                while i < end {
                    let d =
                        DvNbr::from_packed(self.mi_dv[((mi_row + i) * cols + mi_col - 1) as usize]);
                    if matches(&d) {
                        np += 1;
                        if np >= MAX {
                            return MAX;
                        }
                    }
                    i += MI_SIZE_HIGH[d.bsize];
                }
            }
        }

        if do_tl && left && up {
            let d = DvNbr::from_packed(self.mi_dv[((mi_row - 1) * cols + mi_col - 1) as usize]);
            if matches(&d) {
                np += 1;
                if np >= MAX {
                    return MAX;
                }
            }
        }
        np
    }

    /// `motion_mode_allowed` (blockd.h:1477): the motion-mode ceiling given the
    /// block's mode/refs and the frame's `allow_warped_motion`. Returns
    /// `0 = SIMPLE_TRANSLATION`, `1 = OBMC_CAUSAL`, `2 = WARPED_CAUSAL`. Global
    /// motion is identity here (`gm_type = IDENTITY`, not `> TRANSLATION`), so
    /// `is_global_mv_block` is unconditionally false and its early-out is skipped.
    fn motion_mode_ceiling(
        &self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        mode: i32,
        ref0: i32,
        ref1: i32,
        inter: &InterFrameCfg,
    ) -> i32 {
        // NEARESTMV=13 .. NEWMV=16 are the single-ref inter modes.
        const NEARESTMV: i32 = 13;
        if !self.has_overlappable_neighbors(mi_row, mi_col, bsize) {
            return 0;
        }
        let is_inter_mode = mode >= NEARESTMV;
        // ref1 != INTRA_FRAME (0) and !has_second_ref (ref1 <= INTRA_FRAME).
        if is_motion_variation_allowed_bsize(bsize) && is_inter_mode && ref1 < 0 {
            let npr = self.num_proj_ref(mi_row, mi_col, bsize, ref0);
            if npr >= 1
                && inter.allow_warped_motion
                && !inter.cur_frame_force_integer_mv
                && !inter.ref_sf[ref0 as usize - 1].is_scaled()
            {
                return 2; // WARPED_CAUSAL
            }
            return 1; // OBMC_CAUSAL
        }
        0
    }

    /// `dec_build_prediction_by_above_preds` + `av1_build_obmc_inter_prediction`
    /// (ABOVE half), luma: for each overlappable above-neighbour, MC-predict a
    /// narrow `op_mi_size*4`-wide × `overlap`-tall strip from the reference using
    /// the NEIGHBOUR's mv/ref/filter, then feather-blend it into the block's own
    /// predictor (already in `recon`) with the vertical OBMC mask. Chroma OBMC is
    /// skipped here (`av1_skip_u4x4_pred_in_obmc` returns 1 for a `<= 8x8`
    /// chroma-plane block in the ABOVE direction — true for this target's
    /// BLOCK_16X8 -> BLOCK_8X4 chroma); a chroma-carrying above-OBMC block is a
    /// later target and is asserted-guarded.
    fn obmc_above_blend(&mut self, mi_row: i32, mi_col: i32, bsize: usize, inter: &InterFrameCfg) {
        let (ss_x, ss_y) = (self.cfg.subsampling_x, self.cfg.subsampling_y);
        // Chroma above-OBMC fires unless `av1_skip_u4x4_pred_in_obmc(dir=0)` skips
        // it (a `<= 8x8` chroma plane block: BLOCK_8X8/8X16/16X8 at 4:2:0). The
        // chroma strips + blend are added per-plane after each neighbour's luma.
        let do_chroma = !self.cfg.monochrome && !skip_u4x4_pred_in_obmc(bsize, ss_x, ss_y, 0);
        let nb_max = MAX_NEIGHBOR_OBMC[mi_size_wide_log2(bsize)];
        let neighbours = self.overlappable_above(mi_row, mi_col, bsize, nb_max);
        if neighbours.is_empty() {
            return;
        }
        let bsize_high = BLOCK_SIZE_HIGH[bsize]; // px
        let overlap = bsize_high.min(64) >> 1; // luma overlap rows
        let width = MI_SIZE_WIDE[bsize];
        // dec_build_prediction_by_above_preds edge adjustments (decodeframe.c:736).
        let this_height = MI_SIZE_HIGH[bsize] * 4;
        let pred_height = (this_height / 2).min(32);
        let block_mb_to_right = (self.cfg.mi_cols - width - mi_col) * 32;
        let block_mb_to_bottom = (self.cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let nb_mb_to_bottom = block_mb_to_bottom + (this_height - pred_height) * 8;
        let nb_mb_to_top = -(mi_row * 32);
        for (rel_mi_col, op_mi_size, nb, nb_if) in neighbours {
            // The OBMC strip is MC'd from the NEIGHBOUR's reference frame
            // (dec_build_prediction_by_above_pred uses `above_mbmi->ref_frame`).
            let Some(last) = ((1..=7).contains(&nb.ref_frame0))
                .then(|| inter.refs[(nb.ref_frame0 - 1) as usize])
                .flatten()
            else {
                self.mark_corrupt(format!(
                    "inter OBMC: above neighbour references unavailable ref {}",
                    nb.ref_frame0
                ));
                return;
            };
            // The OBMC strip is MC'd with the NEIGHBOUR's coded interp filter
            // (av1_setup_build_prediction_by_above_pred keeps `above_mbmi->
            // interp_filters`). (y_filter, x_filter) = (nb_if.0, nb_if.1).
            let (nb_filter_y, nb_filter_x) = (nb_if.0 as usize, nb_if.1 as usize);
            let above_mi_col = mi_col + rel_mi_col;
            // Luma prediction dims (dec_build_prediction_by_above_pred, ss=0):
            // bw = op_mi_size*4; bh = clamp(bsize_high>>1, 4, 32).
            let bw = (op_mi_size * 4) as usize;
            let bh = (bsize_high >> 1).clamp(4, 32) as usize;
            // OBMC-neighbour mb edges (av1_setup_build_prediction_by_above_pred).
            let nb_mb_to_left = -(above_mi_col * 32);
            let nb_mb_to_right = block_mb_to_right + (width - rel_mi_col - op_mi_size) * 32;
            let (nmv_r, nmv_c) = clamp_mv_umv_border_px(
                nb.mv0_row,
                nb.mv0_col,
                bw as i32,
                bh as i32,
                nb_mb_to_left,
                nb_mb_to_right,
                nb_mb_to_top,
                nb_mb_to_bottom,
                0,
                0,
            );
            let mut scratch = vec![0u16; bw * bh];
            aom_dsp::inter::build_inter_predictor(
                &last.y,
                last.stride,
                last.width,
                last.height,
                &mut scratch,
                0,
                bw,
                (above_mi_col * 4) as usize, // ref-read pix col
                (mi_row * 4) as usize,       // ref-read pix row (current block row)
                bw,
                bh,
                nmv_r,
                nmv_c,
                nb.mv0_row,
                nb.mv0_col,
                0,
                0,
                nb_filter_x,
                nb_filter_y,
                self.cfg.bd as u32,
                &inter.ref_sf[nb.ref_frame0 as usize - 1],
            );
            // Blend the top `overlap` rows of the block's own predictor with the
            // neighbour strip (build_obmc_inter_pred_above -> aom_blend_a64_vmask).
            let plane_col = (rel_mi_col * 4) as usize;
            let dst_off = (mi_row * 4) as usize * self.stride + (mi_col * 4) as usize + plane_col;
            let mask = aom_dsp::inter::get_obmc_mask(overlap as usize);
            self.recon.with_wide_rect(
                dst_off,
                self.stride,
                bw,
                overlap as usize,
                &mut self.wide_rect,
                |dst, stride| {
                    aom_dsp::inter::blend_a64_vmask(
                        dst,
                        0,
                        stride,
                        &scratch,
                        0,
                        bw,
                        mask,
                        bw,
                        overlap as usize,
                    );
                },
            );
            // Chroma above-OBMC (build_obmc_inter_pred_above, U then V). Strip MC
            // dims (decodeframe.c:726): bw_c = (op_mi_size*4)>>ss_x, bh_c =
            // clamp(bsize_high>>(ss_y+1), 4, 64>>(ss_y+1)); blended over the top
            // `overlap>>ss_y` rows with the subsampled mask. Position is the same
            // chroma column as the strip's neighbour column.
            if do_chroma {
                let overlap_c = (overlap as usize) >> ss_y; // rows blended
                let bw_c = ((op_mi_size * 4) >> ss_x) as usize;
                let bh_c = ((bsize_high >> (ss_y + 1)).clamp(4, 64 >> (ss_y + 1))) as usize;
                let (cr, cc) = clamp_mv_umv_border_px(
                    nb.mv0_row,
                    nb.mv0_col,
                    bw_c as i32,
                    bh_c as i32,
                    nb_mb_to_left,
                    nb_mb_to_right,
                    nb_mb_to_top,
                    nb_mb_to_bottom,
                    ss_x,
                    ss_y,
                );
                let uv_col = ((above_mi_col * 4) >> ss_x) as usize;
                let uv_row = ((mi_row * 4) >> ss_y) as usize;
                let dst_off_c = uv_row * self.stride_uv + uv_col;
                let mask_c = aom_dsp::inter::get_obmc_mask(overlap_c);
                for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                    let mut scratch_c = vec![0u16; bw_c * bh_c];
                    aom_dsp::inter::build_inter_predictor(
                        src,
                        last.stride_uv,
                        last.width_uv,
                        last.height_uv,
                        &mut scratch_c,
                        0,
                        bw_c,
                        uv_col,
                        uv_row,
                        bw_c,
                        bh_c,
                        cr,
                        cc,
                        nb.mv0_row,
                        nb.mv0_col,
                        ss_x,
                        ss_y,
                        nb_filter_x,
                        nb_filter_y,
                        self.cfg.bd as u32,
                        &inter.ref_sf[nb.ref_frame0 as usize - 1],
                    );
                    dst.with_wide_rect(
                        dst_off_c,
                        self.stride_uv,
                        bw_c,
                        overlap_c,
                        &mut self.wide_rect,
                        |d, stride| {
                            aom_dsp::inter::blend_a64_vmask(
                                d, 0, stride, &scratch_c, 0, bw_c, mask_c, bw_c, overlap_c,
                            );
                        },
                    );
                }
            }
        }
    }

    /// `dec_build_prediction_by_left_preds` + `av1_build_obmc_inter_prediction`
    /// (LEFT half), luma: the left-column twin of [`Self::obmc_above_blend`],
    /// feather-blending with the horizontal OBMC mask. INERT for the chunk-4
    /// target (the OBMC block is at the frame's left edge -> no left neighbour),
    /// but faithful for a future left-OBMC target.
    fn obmc_left_blend(&mut self, mi_row: i32, mi_col: i32, bsize: usize, inter: &InterFrameCfg) {
        let (ss_x, ss_y) = (self.cfg.subsampling_x, self.cfg.subsampling_y);
        let nb_max = MAX_NEIGHBOR_OBMC[mi_size_high_log2(bsize)];
        let neighbours = self.overlappable_left(mi_row, mi_col, bsize, nb_max);
        if neighbours.is_empty() {
            return;
        }
        // Chroma LEFT-OBMC always fires for a non-mono block (av1_skip_u4x4_pred_in
        // _obmc skips only the ABOVE direction, dir==0). Added per-plane below.
        let do_chroma = !self.cfg.monochrome;
        let bsize_wide = BLOCK_SIZE_WIDE[bsize];
        let overlap = bsize_wide.min(64) >> 1; // luma overlap cols
        let height = MI_SIZE_HIGH[bsize];
        let this_width = MI_SIZE_WIDE[bsize] * 4;
        let pred_width = (this_width / 2).min(32);
        let block_mb_to_bottom = (self.cfg.mi_rows - height - mi_row) * 32;
        let block_mb_to_right = (self.cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let nb_mb_to_right = block_mb_to_right + (this_width - pred_width) * 8;
        let nb_mb_to_left = -(mi_col * 32);
        for (rel_mi_row, op_mi_size, nb, nb_if) in neighbours {
            // Strip MC reads the NEIGHBOUR's reference frame + coded filter.
            let Some(last) = ((1..=7).contains(&nb.ref_frame0))
                .then(|| inter.refs[(nb.ref_frame0 - 1) as usize])
                .flatten()
            else {
                self.mark_corrupt(format!(
                    "inter OBMC: left neighbour references unavailable ref {}",
                    nb.ref_frame0
                ));
                return;
            };
            let (nb_filter_y, nb_filter_x) = (nb_if.0 as usize, nb_if.1 as usize);
            let left_mi_row = mi_row + rel_mi_row;
            let bw = (bsize_wide >> 1).clamp(4, 32) as usize;
            let bh = (op_mi_size * 4) as usize;
            let nb_mb_to_top = -(left_mi_row * 32);
            let nb_mb_to_bottom = block_mb_to_bottom + (height - rel_mi_row - op_mi_size) * 32;
            let (nmv_r, nmv_c) = clamp_mv_umv_border_px(
                nb.mv0_row,
                nb.mv0_col,
                bw as i32,
                bh as i32,
                nb_mb_to_left,
                nb_mb_to_right,
                nb_mb_to_top,
                nb_mb_to_bottom,
                0,
                0,
            );
            let mut scratch = vec![0u16; bw * bh];
            aom_dsp::inter::build_inter_predictor(
                &last.y,
                last.stride,
                last.width,
                last.height,
                &mut scratch,
                0,
                bw,
                (mi_col * 4) as usize,
                (left_mi_row * 4) as usize,
                bw,
                bh,
                nmv_r,
                nmv_c,
                nb.mv0_row,
                nb.mv0_col,
                0,
                0,
                nb_filter_x,
                nb_filter_y,
                self.cfg.bd as u32,
                &inter.ref_sf[nb.ref_frame0 as usize - 1],
            );
            let plane_row = (rel_mi_row * 4) as usize;
            let dst_off = ((mi_row * 4) as usize + plane_row) * self.stride + (mi_col * 4) as usize;
            let mask = aom_dsp::inter::get_obmc_mask(overlap as usize);
            self.recon.with_wide_rect(
                dst_off,
                self.stride,
                overlap as usize,
                bh,
                &mut self.wide_rect,
                |dst, stride| {
                    aom_dsp::inter::blend_a64_hmask(
                        dst,
                        0,
                        stride,
                        &scratch,
                        0,
                        bw,
                        mask,
                        overlap as usize,
                        bh,
                    );
                },
            );
            // Chroma left-OBMC (build_obmc_inter_pred_left, U then V). Strip MC dims
            // (decodeframe.c:781): bw_c = clamp(bsize_wide>>(ss_x+1), 4, 64>>(ss_x+1)),
            // bh_c = (op_mi_size*4)>>ss_y; blended over the left `overlap>>ss_x` cols
            // (a SUBSET of the >=4-wide strip for small chroma) with the subsampled
            // mask. Never skipped (dir==1).
            if do_chroma {
                let overlap_c = (overlap as usize) >> ss_x; // cols blended
                let bw_c = ((bsize_wide >> (ss_x + 1)).clamp(4, 64 >> (ss_x + 1))) as usize;
                let bh_c = ((op_mi_size * 4) >> ss_y) as usize;
                let (cr, cc) = clamp_mv_umv_border_px(
                    nb.mv0_row,
                    nb.mv0_col,
                    bw_c as i32,
                    bh_c as i32,
                    nb_mb_to_left,
                    nb_mb_to_right,
                    nb_mb_to_top,
                    nb_mb_to_bottom,
                    ss_x,
                    ss_y,
                );
                let uv_col = ((mi_col * 4) >> ss_x) as usize;
                let uv_row = ((left_mi_row * 4) >> ss_y) as usize;
                let dst_off_c = uv_row * self.stride_uv + uv_col;
                let mask_c = aom_dsp::inter::get_obmc_mask(overlap_c);
                for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                    let mut scratch_c = vec![0u16; bw_c * bh_c];
                    aom_dsp::inter::build_inter_predictor(
                        src,
                        last.stride_uv,
                        last.width_uv,
                        last.height_uv,
                        &mut scratch_c,
                        0,
                        bw_c,
                        uv_col,
                        uv_row,
                        bw_c,
                        bh_c,
                        cr,
                        cc,
                        nb.mv0_row,
                        nb.mv0_col,
                        ss_x,
                        ss_y,
                        nb_filter_x,
                        nb_filter_y,
                        self.cfg.bd as u32,
                        &inter.ref_sf[nb.ref_frame0 as usize - 1],
                    );
                    dst.with_wide_rect(
                        dst_off_c,
                        self.stride_uv,
                        overlap_c,
                        bh_c,
                        &mut self.wide_rect,
                        |d, stride| {
                            aom_dsp::inter::blend_a64_hmask(
                                d, 0, stride, &scratch_c, 0, bw_c, mask_c, overlap_c, bh_c,
                            );
                        },
                    );
                }
            }
        }
    }

    /// `read_inter_frame_mode_info`'s else-arm (decodemv.c:1550): an
    /// INTRA-coded block inside an inter frame — the mode-info tail
    /// (shared with `read_mb_modes_kf_fc`, differing only in the Y-mode
    /// CDF) then `decode_intra_block_body` verbatim. Reads
    /// `ref_frame = [INTRA_FRAME, NONE_FRAME]` so every downstream gate
    /// takes its intra arm unchanged.
    fn decode_intra_in_inter_block(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mut icdfs: InterCdfs,
        bx: &BlockCtx,
    ) {
        use aom_dsp::entropy::partition as ep;
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            up_available,
            left_available,
            skip,
            cdef_strength,
            chroma_ref,
            above_mi,
            left_mi,
            ..
        } = *bx;
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        // Persist the inter CDFs adapted so far — `read_is_inter` mutated
        // `intra_inter[ii_ctx]`, and the intra path below never touches
        // `icdfs` again except for the Y-mode row (written back inline).
        let cfl_allowed =
            !cfg.monochrome && is_cfl_allowed(bsize, self.st.coded_lossless, ss_x, ss_y);
        let (above_palette, left_palette) = self.palette_neighbours(mi_row, mi_col);
        self.st.mi_row = mi_row;
        self.st.mi_col = mi_col;
        self.st.bsize = bsize;
        self.st.is_chroma_ref = chroma_ref;
        self.st.cfl_allowed = cfl_allowed;
        self.st.mb_to_top_edge = -(mi_row * 32);
        self.st.has_above = up_available;
        self.st.has_left = left_available;
        self.st.allow_palette = av1_allow_palette(cfg.allow_screen_content_tools, bsize);
        // The prefix `read_inter_frame_mode_info` already consumed
        // (segmentation / delta-q / skip-mode are asserted off above, so
        // segment_id is 0 and the delta-lf carry is the frame default).
        let mut info = MbModeInfoKf {
            segment_id: 0,
            skip,
            cdef_strength,
            current_qindex: cfg.base_qindex,
            delta_lf: [0; 4],
            delta_lf_from_base: 0,
            use_intrabc: 0,
            dv_row: 0,
            dv_col: 0,
            y_mode: 0,
            angle_delta_y: 0,
            uv_mode: 0,
            cfl_alpha_idx: 0,
            cfl_joint_sign: 0,
            angle_delta_uv: 0,
            palette_size: [0, 0],
            palette_colors: [0; 24],
            use_filter_intra: 0,
            filter_intra_mode: 0,
        };
        // `y_mode_cdf[size_group_lookup[bsize]]` (decodemv.c:1077). Copied
        // out and written back so the adaptation persists across blocks,
        // exactly as C adapts `ec_ctx->y_mode_cdf` in place.
        let grp = ep::y_mode_size_group(bsize);
        let mut y_cdf = icdfs.y_mode[grp];
        ep::read_intra_block_mode_info_fc(
            dec,
            cdfs,
            &mut self.st,
            &mut y_cdf,
            cfg.enable_filter_intra,
            above_mi.is_some(),
            left_mi.is_some(),
            above_palette,
            left_palette,
            &mut info,
        );
        icdfs.y_mode[grp] = y_cdf;
        self.inter_cdfs = icdfs;
        if dbg_blocks() {
            aom_dsp::trace_out!(
                "BLK mi({mi_row},{mi_col}) bs={bsize} intra skip={skip} y_mode={} fi={}",
                info.y_mode,
                info.use_filter_intra
            );
        }
        self.decode_intra_block_body(dec, cdfs, info, bx);
    }

    /// `read_inter_block_mode_info` (decodemv.c:1484+) through
    /// `parse_decode_block`'s tx-size + skip-entropy tail: read the ref
    /// pair, prediction mode, MV references and the block's MVs, the
    /// inter-intra / motion-mode / masked-compound / warp and
    /// interp-filter symbols, the var-tx quadtree (or LARGEST fallback),
    /// then `av1_reset_entropy_context` for a skipped block. Returns
    /// `Stage::Stop` on a named refusal / corrupt symbol — the caller exits
    /// the block without predicting or reconstructing.
    fn read_inter_mode_info(
        &mut self,
        dec: &mut OdEcDec,
        icdfs: &mut InterCdfs,
        inter: &InterFrameCfg,
        bx: &BlockCtx,
    ) -> Stage<InterModeInfo> {
        use aom_dsp::entropy::partition as ep;
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            partition,
            skip,
            up_available,
            left_available,
            above_dv,
            left_dv,
            above_if,
            left_if,
            ..
        } = *bx;
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let dv_inter = |d: DvNbr| d.use_intrabc || d.ref_frame0 > 0;
        // --- read_inter_block_mode_info (single reference) ---
        let rc = ep::collect_neighbors_ref_counts(
            up_available,
            above_dv.is_some_and(|d| d.use_intrabc),
            above_dv.map_or(0, |d| d.ref_frame0),
            above_dv.map_or(-1, |d| d.ref_frame1),
            left_available,
            left_dv.is_some_and(|d| d.use_intrabc),
            left_dv.map_or(0, |d| d.ref_frame0),
            left_dv.map_or(-1, |d| d.ref_frame1),
        );
        // `comp_ref_type` (slot 1) contexts off the edge neighbours'
        // compound-ref structure — `av1_get_comp_reference_type_context`
        // (pred_common.c:187), driven only by the comp-allowed case.
        let crt_ctx = ep::get_comp_reference_type_context(
            up_available,
            above_dv.map_or(0, |d| d.ref_frame0),
            above_dv.map_or(-1, |d| d.ref_frame1),
            above_dv.is_some_and(|d| d.use_intrabc),
            left_available,
            left_dv.map_or(0, |d| d.ref_frame0),
            left_dv.map_or(-1, |d| d.ref_frame1),
            left_dv.is_some_and(|d| d.use_intrabc),
        ) as usize;
        let (mut ref_cdfs, comp_ctxs) = icdfs.ref_frame_cdfs(&rc, crt_ctx);
        // `comp_inter` (read_ref_frames' first symbol): read for every
        // comp-allowed block (`is_comp_ref_allowed`: min dim >= 8) of a
        // REFERENCE_MODE_SELECT frame, on the neighbour-derived context
        // (av1_get_reference_mode_context). Slot 0 of the assembled array.
        let comp_allowed = BLOCK_SIZE_WIDE[bsize].min(BLOCK_SIZE_HIGH[bsize]) >= 8;
        let rm_ctx = ep::reference_mode_context(
            above_dv.map(|d| (d.ref_frame0, d.ref_frame1, dv_inter(d))),
            left_dv.map(|d| (d.ref_frame0, d.ref_frame1, dv_inter(d))),
        ) as usize;
        ref_cdfs[0] = icdfs.comp_inter[rm_ctx];
        let (is_compound, _crt, ref0, ref1) = ep::read_ref_frames(
            dec,
            &mut ref_cdfs,
            false,
            false,
            inter.reference_mode_select,
            comp_allowed,
        );
        icdfs.comp_inter[rm_ctx] = ref_cdfs[0];
        // `ref_frame_cdfs` assembled `ref_cdfs` from disjoint CDF rows (each
        // sub-tree at its own pred context); copy the adapted rows back so the
        // adaptation persists across blocks. Only the rows `read_ref_frames`
        // actually read changed; copying all is a no-op for the rest.
        icdfs.comp_ref_type[comp_ctxs.0] = ref_cdfs[1];
        icdfs.uni_comp_ref[comp_ctxs.1[0]][0] = ref_cdfs[2];
        icdfs.uni_comp_ref[comp_ctxs.1[1]][1] = ref_cdfs[3];
        icdfs.uni_comp_ref[comp_ctxs.1[2]][2] = ref_cdfs[4];
        icdfs.comp_ref[comp_ctxs.2[0]][0] = ref_cdfs[5];
        icdfs.comp_ref[comp_ctxs.2[1]][1] = ref_cdfs[6];
        icdfs.comp_ref[comp_ctxs.2[2]][2] = ref_cdfs[7];
        icdfs.comp_bwdref[comp_ctxs.3[0]][0] = ref_cdfs[8];
        icdfs.comp_bwdref[comp_ctxs.3[1]][1] = ref_cdfs[9];
        icdfs.single_ref[ep::single_ref_p1_context(&rc) as usize][0] = ref_cdfs[10];
        icdfs.single_ref[ep::pred_ctx_brfarf2_or_arf(&rc) as usize][1] = ref_cdfs[11];
        icdfs.single_ref[ep::pred_ctx_ll2_or_l3gld(&rc) as usize][2] = ref_cdfs[12];
        icdfs.single_ref[ep::pred_ctx_last_or_last2(&rc) as usize][3] = ref_cdfs[13];
        icdfs.single_ref[ep::pred_ctx_last3_or_gld(&rc) as usize][4] = ref_cdfs[14];
        icdfs.single_ref[ep::pred_ctx_brf_or_arf2(&rc) as usize][5] = ref_cdfs[15];
        // Keep the malformed and unsupported cases distinct: an out-of-range
        // or inconsistent ref coding is a bad stream (corrupt); a well-formed
        // compound pair is a valid AV1 tool (decoded below with the feature on,
        // refused by name with it off).
        if !(1..=7).contains(&ref0)
            || (!is_compound && ref1 != -1)
            || (is_compound && !(1..=7).contains(&ref1))
        {
            self.mark_corrupt(format!(
                "inter: inconsistent ref coding (ref0 {ref0}, ref1 {ref1}, compound {is_compound})"
            ));
            return Stage::Stop;
        }
        if is_compound && !crate::EXPERIMENTAL_VIDEO {
            self.mark_unsupported(
                "inter: only single-reference blocks are decoded in this envelope \
                 (compound references unsupported)",
            );
            return Stage::Stop;
        }
        // find_inter_mv_refs below hardcodes IDENTITY global motion (base MV (0,0),
        // gm_type 0). A frame whose reference carries non-identity global motion
        // (`global_motion[ref].wmtype > IDENTITY`) needs the real global-MV base +
        // is_global_mv_block gating — a later chunk. Guarded so such a frame pins
        // cleanly here rather than reading a wrong NEWMV base and desyncing (every
        // target through 16x18 is identity-GM; e.g. 16x66 uses global motion).
        // For a compound pair BOTH refs must be identity-GM.
        if inter.gm_wmtype[(ref0 - 1) as usize] != 0
            || (is_compound && inter.gm_wmtype[(ref1 - 1) as usize] != 0)
        {
            self.mark_unsupported(
                "inter: non-identity global motion not supported in this decode envelope",
            );
            return Stage::Stop;
        }

        // find_inter_mv_refs (identity GM, empty temporal field per the census).
        let dv_tile = DvTileBounds {
            mi_row_start: self.tile.mi_row_start,
            mi_row_end: self.tile.mi_row_end,
            mi_col_start: self.tile.mi_col_start,
            mi_col_end: self.tile.mi_col_end,
        };
        let mib_size = self.st.mib_size;
        let grid = MiDvGrid {
            mi_dv: &self.mi_dv,
            cols: cfg.mi_cols,
            rows: cfg.mi_rows,
            mi_row,
            mi_col,
        };
        let tpl_field = inter
            .tpl_cells
            .map(|cells| aom_dsp::entropy::dv_ref::TplField {
                cells,
                stride: inter.tpl_stride,
                cur_offset: inter.tpl_cur_offset,
            });
        let rf = if is_compound {
            [ref0, ref1]
        } else {
            [ref0, NONE_FRAME]
        };
        let imv = find_inter_mv_refs(
            rf,
            mi_row,
            mi_col,
            bsize,
            partition,
            up_available,
            left_available,
            dv_tile,
            cfg.mi_rows,
            cfg.mi_cols,
            mib_size,
            inter.allow_ref_frame_mvs,
            tpl_field.as_ref(),
            [(0, 0), (0, 0)],
            [0, 0],
            inter.ref_frame_sign_bias,
            inter.allow_high_precision_mv,
            inter.cur_frame_force_integer_mv,
            grid,
        );

        // read_inter_mode / read_inter_compound_mode (decodemv.c:1312-1316):
        // single-ref reads the NEWMV/GLOBALMV/NEARESTMV cascade; compound reads
        // the 8-symbol compound-mode row on `av1_mode_context_analyzer`'s ctx.
        let mode = if is_compound {
            let mctx = ep::mode_context_analyzer(imv.mode_context, true) as usize;
            ep::read_inter_compound_mode(dec, &mut icdfs.inter_compound_mode[mctx])
        } else {
            ep::read_inter_mode(
                dec,
                &mut icdfs.newmv,
                &mut icdfs.zeromv,
                &mut icdfs.refmv,
                imv.mode_context,
            )
        };
        // The coded mode's compoundness must agree with the ref coding
        // (decodemv.c:1323 — a desynced stream is corrupt, not unsupported).
        if is_compound != ep::is_inter_compound_mode(mode) {
            self.mark_corrupt(format!(
                "inter: mode {mode} inconsistent with ref coding (compound {is_compound})"
            ));
            return Stage::Stop;
        }
        // read_drl_idx: weights as u16 (values are well under 2^16, see dv_ref).
        // No-ops (returns 0, reads nothing) for non-NEW/non-NEAR modes — the
        // gate matches C's `mode == NEWMV || NEW_NEWMV || have_nearmv`.
        let weights_u16: [u16; 8] = std::array::from_fn(|i| imv.weight[i] as u16);
        let ref_mv_idx = ep::read_drl_idx(
            dec,
            &mut icdfs.drl,
            mode,
            imv.ref_mv_count as i32,
            &weights_u16,
        );

        // assign_mv: resolve the predictor(s) per mode, then read the coded MVs.
        // Compound resolves an MV per reference (decodemv.c:1114-1212).
        let precision = if inter.cur_frame_force_integer_mv {
            -1
        } else if inter.allow_high_precision_mv {
            1
        } else {
            0
        };
        // Compound sub-modes (blockd.h compound_ref{0,1}_mode pairings).
        const NEAREST_NEARESTMV: i32 = 17;
        const NEAR_NEARMV: i32 = 18;
        const NEAREST_NEWMV: i32 = 19;
        const NEW_NEARESTMV: i32 = 20;
        const NEAR_NEWMV: i32 = 21;
        const NEW_NEARMV: i32 = 22;
        const GLOBAL_GLOBALMV: i32 = 23;
        const NEW_NEWMV: i32 = 24;
        let (mv_row, mv_col, mv1_row, mv1_col) = if is_compound {
            // nearest/near come off the compound PAIR stack: `stack[i]` is
            // ref0's candidate, `comp_stack[i]` ref1's (this_mv / comp_mv).
            // Both are `lower_mv_precision`'d; GLOBAL_GLOBALMV skips the block.
            let mut nearest = [(0i32, 0i32); 2];
            let mut near = [(0i32, 0i32); 2];
            if mode != GLOBAL_GLOBALMV {
                nearest = [imv.stack[0], imv.comp_stack[0]];
                near = [
                    imv.stack[(1 + ref_mv_idx) as usize],
                    imv.comp_stack[(1 + ref_mv_idx) as usize],
                ];
                for m in nearest.iter_mut().chain(near.iter_mut()) {
                    aom_dsp::entropy::dv_ref::lower_mv_precision(
                        &mut m.0,
                        &mut m.1,
                        inter.allow_high_precision_mv,
                        inter.cur_frame_force_integer_mv,
                    );
                }
            }
            // Each NEW component's read_mv base defaults to the (lowered)
            // nearest and is overwritten by the RAW stack entry
            // (`ref_mv_stack[ref_mv_idx]` — `+1` for the NEAR_NEW/NEW_NEAR
            // pairings, decodemv.c:1362-1368).
            let mut ref_mv = nearest;
            let rmi = if mode == NEAR_NEWMV || mode == NEW_NEARMV {
                (1 + ref_mv_idx) as usize
            } else {
                ref_mv_idx as usize
            };
            if ep::compound_ref0_mode(mode) == NEWMV {
                ref_mv[0] = imv.stack[rmi];
            }
            if ep::compound_ref1_mode(mode) == NEWMV {
                ref_mv[1] = imv.comp_stack[rmi];
            }
            let mut mvp = [(0i32, 0i32); 2];
            match mode {
                NEAREST_NEARESTMV => mvp = nearest,
                NEAR_NEARMV => mvp = near,
                NEW_NEWMV => {
                    for i in 0..2 {
                        let [c0, c1] = &mut icdfs.nmv_comps;
                        let (dr, dc) = ep::read_mv(dec, &mut icdfs.nmv_joints, c0, c1, precision);
                        mvp[i] = (ref_mv[i].0 + dr, ref_mv[i].1 + dc);
                    }
                }
                NEAREST_NEWMV => {
                    mvp[0] = nearest[0];
                    let [c0, c1] = &mut icdfs.nmv_comps;
                    let (dr, dc) = ep::read_mv(dec, &mut icdfs.nmv_joints, c0, c1, precision);
                    mvp[1] = (ref_mv[1].0 + dr, ref_mv[1].1 + dc);
                }
                NEW_NEARESTMV => {
                    let [c0, c1] = &mut icdfs.nmv_comps;
                    let (dr, dc) = ep::read_mv(dec, &mut icdfs.nmv_joints, c0, c1, precision);
                    mvp[0] = (ref_mv[0].0 + dr, ref_mv[0].1 + dc);
                    mvp[1] = nearest[1];
                }
                NEAR_NEWMV => {
                    mvp[0] = near[0];
                    let [c0, c1] = &mut icdfs.nmv_comps;
                    let (dr, dc) = ep::read_mv(dec, &mut icdfs.nmv_joints, c0, c1, precision);
                    mvp[1] = (ref_mv[1].0 + dr, ref_mv[1].1 + dc);
                }
                NEW_NEARMV => {
                    let [c0, c1] = &mut icdfs.nmv_comps;
                    let (dr, dc) = ep::read_mv(dec, &mut icdfs.nmv_joints, c0, c1, precision);
                    mvp[0] = (ref_mv[0].0 + dr, ref_mv[0].1 + dc);
                    mvp[1] = near[1];
                }
                GLOBAL_GLOBALMV => mvp = [imv.global_mv, imv.global_mv1],
                _ => {
                    self.mark_corrupt(format!("inter: invalid compound mode {mode}"));
                    return Stage::Stop;
                }
            }
            (mvp[0].0, mvp[0].1, mvp[1].0, mvp[1].1)
        } else {
            let (r, c) = match mode {
                NEWMV => {
                    // ref_mv[0] = nearest, or stack[ref_mv_idx] when the list has >1.
                    let base = if imv.ref_mv_count > 1 {
                        imv.stack[ref_mv_idx as usize]
                    } else {
                        imv.nearest
                    };
                    let [c0, c1] = &mut icdfs.nmv_comps;
                    let (dr, dc) = ep::read_mv(dec, &mut icdfs.nmv_joints, c0, c1, precision);
                    (base.0 + dr, base.1 + dc)
                }
                NEARESTMV => imv.nearest,
                NEARMV => {
                    if ref_mv_idx > 0 {
                        imv.stack[(1 + ref_mv_idx) as usize]
                    } else {
                        imv.near
                    }
                }
                GLOBALMV => (0, 0), // identity global motion (census: all IDENTITY)
                _ => {
                    self.mark_corrupt(format!("inter: unsupported single-ref mode {mode}"));
                    return Stage::Stop;
                }
            };
            (r, c, 0, 0)
        };

        // Inter-intra (decodemv.c:1383-1407 — AFTER assign_mv, BEFORE findSamples /
        // read_motion_mode): the flag, then (when set) the mode and the optional
        // wedge flag + shape index. Coded for an interintra-allowed single-ref
        // inter block when `enable_interintra_compound`. `av1_is_wedge_used` is
        // true for EVERY interintra-allowed bsize (BLOCK_8X8..BLOCK_32X32 all have
        // 16 wedge types), so the wedge flag is always read when the mode is —
        // carried as a gate anyway, matching C.
        let mut ref1 = ref1;
        let (interintra, ii_mode, ii_use_wedge, ii_wedge_idx) =
            if inter.enable_interintra_compound && is_interintra_allowed(bsize, mode, ref0, ref1) {
                let grp = SIZE_GROUP_LOOKUP[bsize];
                // Four DISJOINT fields of `icdfs`, so the borrows do not conflict.
                let r = ep::read_interintra_info(
                    dec,
                    true,
                    &mut icdfs.interintra[grp],
                    &mut icdfs.interintra_mode[grp],
                    aom_dsp::inter::interintra::is_wedge_used(bsize),
                    &mut icdfs.wedge_interintra[bsize],
                    &mut icdfs.wedge_idx[bsize],
                );
                if r.0 != 0 {
                    // `mbmi->ref_frame[1] = INTRA_FRAME` (decodemv.c:1393). This is
                    // load-bearing beyond bookkeeping: `av1_findSamples` counts a warp
                    // sample only for a neighbour with `ref_frame[1] == NONE_FRAME`
                    // (-1), so an interintra neighbour must NOT look single-ref there.
                    // (`collect_neighbors_ref_counts` and the MV scan both gate on
                    // `> INTRA_FRAME`, so they are unaffected either way.)
                    ref1 = 0;
                }
                r
            } else {
                (0, 0, 0, 0)
            };
        let interintra = interintra != 0;

        // read_motion_mode (decodemv.c:1422 — BEFORE read_mb_interp_filter). A
        // symbol is read only when the frame allows switchable motion modes AND
        // (via motion_mode_allowed) the block is motion-variation-allowed
        // (min(bw,bh) >= 8) with >= 1 overlappable inter neighbour. The ceiling
        // selects the 2-symbol obmc_cdf (OBMC ceiling) or the 3-symbol
        // motion_mode_cdf (WARP ceiling); the resolved mode may still be SIMPLE.
        // The earlier targets read nothing (64x64 skeleton: no neighbours; 16x16
        // ratchet: BLOCK_16X4 fails the size gate). WARPED_CAUSAL is chunk 5.
        //
        // An INTER-INTRA block reads NO motion-mode symbol: C gates the whole read
        // on `mbmi->ref_frame[1] != INTRA_FRAME` (decodemv.c:1421) after seeding
        // `motion_mode = SIMPLE_TRANSLATION` (:1414), so inter-intra and
        // OBMC/WARPED_CAUSAL are mutually exclusive by construction.
        let motion_mode = if interintra {
            0 // SIMPLE_TRANSLATION — no symbol read
        } else if inter.switchable_motion_mode {
            let ceiling = self.motion_mode_ceiling(mi_row, mi_col, bsize, mode, ref0, ref1, inter);
            ep::read_motion_mode(
                dec,
                &mut icdfs.obmc[bsize],
                &mut icdfs.motion_mode[bsize],
                ceiling,
            )
        } else {
            0 // SIMPLE_TRANSLATION
        };

        // read_compound_type_info (decodemv.c:1424-1478 — AFTER motion_mode,
        // BEFORE read_mb_interp_filter): only a second-ref (compound) block
        // codes these. `comp_group_idx` selects average/dist-wtd (group 0) vs
        // masked (group 1); within group 0, `compound_idx` picks dist-wtd vs
        // plain average. Group 1 is the wedge/diffwtd masked family — the
        // mask payload (comp_type + wedge_index/sign or mask_type) feeds the
        // masked predictor below.
        let (comp_group_idx, compound_idx, comp_type, wedge_index, wedge_sign, mask_type) =
            if is_compound {
                let masked_compound_used = comp_allowed && inter.enable_masked_compound;
                let cgi_ctx = ep::get_comp_group_idx_context(
                    up_available,
                    above_dv.map_or(0, |d| d.ref_frame0),
                    above_dv.map_or(-1, |d| d.ref_frame1),
                    above_dv.map_or(0, |d| d.comp_group_idx),
                    left_available,
                    left_dv.map_or(0, |d| d.ref_frame0),
                    left_dv.map_or(-1, |d| d.ref_frame1),
                    left_dv.map_or(0, |d| d.comp_group_idx),
                ) as usize;
                // `fwd`/`bck` order-hint pairing follows C's buffer lookup
                // (pred_common.h:102): `bck_buf` is ref_frame[0], `fwd_buf`
                // ref_frame[1] — so ref1's order hint is the `fwd` arg.
                let ci_ctx = ep::get_comp_index_context(
                    inter.enable_order_hint,
                    inter.order_hint_bits_minus_1,
                    inter.order_hint,
                    inter.ref_order_hints[ref1 as usize],
                    inter.ref_order_hints[ref0 as usize],
                    up_available,
                    above_dv.is_some_and(|d| d.ref_frame1 > 0),
                    above_dv.map_or(0, |d| d.compound_idx),
                    above_dv.map_or(0, |d| d.ref_frame0),
                    left_available,
                    left_dv.is_some_and(|d| d.ref_frame1 > 0),
                    left_dv.map_or(0, |d| d.compound_idx),
                    left_dv.map_or(0, |d| d.ref_frame0),
                ) as usize;
                ep::read_compound_type_info(
                    dec,
                    masked_compound_used,
                    &mut icdfs.comp_group_idx[cgi_ctx],
                    inter.enable_dist_wtd_comp,
                    &mut icdfs.compound_idx[ci_ctx],
                    comp_allowed && aom_dsp::inter::interintra::is_wedge_used(bsize),
                    &mut icdfs.compound_type[bsize],
                    &mut icdfs.wedge_idx[bsize],
                )
            } else {
                // Non-compound defaults (C seeds comp_group_idx=0, compound_idx=1).
                (0, 1, 0, 0, 0, 0)
            };
        // Masked compound (`comp_group_idx == 1`): `comp_type` picks the wedge
        // codebook mask (`COMPOUND_WEDGE` = 2, with wedge_index/wedge_sign) or
        // the diff-weighted `seg_mask` (`COMPOUND_DIFFWTD` = 3, with mask_type).
        let masked: Option<aom_dsp::inter::MaskedCompound> = if is_compound && comp_group_idx != 0 {
            const COMPOUND_WEDGE: i32 = 2;
            Some(if comp_type == COMPOUND_WEDGE {
                aom_dsp::inter::MaskedCompound::Wedge {
                    index: wedge_index as usize,
                    sign: wedge_sign as usize,
                }
            } else {
                aom_dsp::inter::MaskedCompound::Diffwtd {
                    mask_type: if mask_type != 0 {
                        aom_dsp::inter::compound::DiffwtdMaskType::Diffwtd38Inv
                    } else {
                        aom_dsp::inter::compound::DiffwtdMaskType::Diffwtd38
                    },
                }
            })
        } else {
            None
        };
        // `av1_dist_wtd_comp_weight_assign` (reconinter.c:669): compound_idx==0
        // -> distance-weighted offsets from the pair's order-hint distances;
        // compound_idx==1 (or !is_compound) -> the plain 8/8 average.
        let comp_weights = if is_compound {
            aom_dsp::inter::compound::dist_wtd_comp_weight_assign(
                inter.enable_order_hint,
                inter.order_hint_bits_minus_1,
                inter.order_hint,
                inter.ref_order_hints[ref1 as usize],
                inter.ref_order_hints[ref0 as usize],
                compound_idx != 0,
                is_compound,
            )
        } else {
            aom_dsp::inter::compound::DistWtdWeights {
                fwd_offset: 8,
                bck_offset: 8,
                use_dist_wtd_comp_avg: false,
            }
        };
        // WARPED_CAUSAL (chunk 5): gather the warp samples (av1_findSamples),
        // select (av1_selectSamples when num_proj_ref > 1), derive the local
        // AFFINE model (av1_find_projection). C: decodemv.c:1484-1503. An invalid
        // model marks `wm.invalid` -> MC falls back to translational (allow_warp ->
        // global(identity) -> TRANSLATION_PRED). Ordering vs read_mb_interp_filter is
        // moot: find_projection reads NO entropy symbols. `warp_luma` (a usable local
        // model) drives the affine MC below; luma always passes the per-plane >= 8
        // gate (motion-variation requires min dim >= 8), chroma is re-gated at its MC.
        let warp_params: Option<aom_dsp::inter::warp::WarpedMotionParams> = if motion_mode == 2 {
            let warp_grid = MiDvGrid {
                mi_dv: &self.mi_dv,
                cols: cfg.mi_cols,
                rows: cfg.mi_rows,
                mi_row,
                mi_col,
            };
            let mut ws = find_samples(
                &warp_grid,
                &dv_tile,
                mib_size,
                cfg.mi_rows,
                cfg.mi_cols,
                mi_row,
                mi_col,
                MI_SIZE_WIDE[bsize],
                MI_SIZE_HIGH[bsize],
                partition,
                ref0,
                up_available,
                left_available,
            );
            if ws.np > 1 {
                select_samples(
                    &mut ws,
                    mv_row,
                    mv_col,
                    BLOCK_SIZE_WIDE[bsize],
                    BLOCK_SIZE_HIGH[bsize],
                );
            }
            let mut wm = aom_dsp::inter::warp::WarpedMotionParams {
                wmtype: aom_dsp::inter::warp::AFFINE,
                ..Default::default()
            };
            let invalid = aom_dsp::inter::warp::find_projection(
                ws.np,
                &ws.pts,
                &ws.pts_inref,
                BLOCK_SIZE_WIDE[bsize],
                BLOCK_SIZE_HIGH[bsize],
                mv_row,
                mv_col,
                &mut wm,
                mi_row,
                mi_col,
            ) != 0;
            wm.invalid = invalid as u8;
            Some(wm)
        } else {
            None
        };
        let warp_luma = warp_params.filter(|w| w.invalid == 0);

        // read_mb_interp_filter (decodemv.c:1481 — AFTER motion_mode). av1_is_interp_needed
        // (reconinter.h:420) gates on motion_mode != WARPED_CAUSAL: a WARPED_CAUSAL block
        // reads NO interp symbol (set_default_interp_filters). WARPED_CAUSAL is guarded off
        // above, so interp_needed holds here; the switchable per-direction neighbour-context
        // read (chunk 6) follows.
        let interp_needed = motion_mode != 2; // 2 = WARPED_CAUSAL
        let is_switchable = inter.interp_filter == SWITCHABLE;
        // av1_extract_interp_filter: dir 0 = y_filter (vertical), dir 1 = x_filter.
        let (filter_y, filter_x) = if !interp_needed {
            // set_default_interp_filters -> av1_broadcast_interp_filter(
            // av1_unswitchable_filter(frame_filter)): EIGHTTAP_REGULAR (0) for a
            // SWITCHABLE frame, else the frame's fixed filter. No symbol is read.
            let f = if is_switchable {
                0
            } else {
                inter.interp_filter as usize
            };
            (f, f)
        } else if is_switchable {
            // A switchable frame reads a per-direction filter on the CDF context
            // selected from the neighbour filters
            // (av1_get_pred_context_switchable_interp). The neighbour projections
            // are Some((ref_frame0, ref_frame1, y_filter, x_filter)) when the edge
            // is available (C's xd->mi[-1] / xd->mi[-mi_stride]); get_ref_filter_type
            // returns the neighbour's coded filter only when it references the
            // current block's ref, else SWITCHABLE_FILTERS. With no neighbours the
            // context is (dir==0 ? 3 : 11), matching the single-block skeleton.
            let above_flt = above_dv
                .zip(above_if)
                .map(|(d, (yf, xf))| (d.ref_frame0, d.ref_frame1, yf as usize, xf as usize));
            let left_flt = left_dv
                .zip(left_if)
                .map(|(d, (yf, xf))| (d.ref_frame0, d.ref_frame1, yf as usize, xf as usize));
            let cur_is_compound = ref1 > 0; // ref_frame[1] > INTRA_FRAME
            let ctx0 = ep::get_pred_context_switchable_interp(
                0,
                ref0,
                cur_is_compound,
                above_flt,
                left_flt,
            ) as usize;
            let ctx1 = ep::get_pred_context_switchable_interp(
                1,
                ref0,
                cur_is_compound,
                above_flt,
                left_flt,
            ) as usize;
            // ctx0 (dir 0) and ctx1 (dir 1) always differ by the 8-wide dir offset
            // -> distinct CDF rows, no aliasing. Non-dual reads only ctx0.
            let mut c0 = icdfs.switchable_interp[ctx0];
            let mut c1 = icdfs.switchable_interp[ctx1];
            let (f0, f1) = ep::read_mb_interp_filter(
                dec,
                &mut c0,
                &mut c1,
                true,
                true,
                inter.enable_dual_filter,
            );
            icdfs.switchable_interp[ctx0] = c0;
            icdfs.switchable_interp[ctx1] = c1;
            (f0 as usize, f1 as usize)
        } else {
            // av1_broadcast_interp_filter(frame_filter): the fixed frame filter.
            (inter.interp_filter as usize, inter.interp_filter as usize)
        };

        // The subpel MC kernels in `aom_dsp::inter`/`convolve` implement the
        // 8-tap/4-tap REGULAR/SMOOTH/SHARP filters (0/1/2) — BILINEAR (3) and any
        // out-of-range value are not in the decode envelope. A malformed frame
        // header can code `interp_filter == BILINEAR` (a valid AV1 value the port
        // does not yet implement), which would reach the kernel lookup and panic.
        // Reject it here instead. Byte-inert on in-envelope streams (resolved
        // filters are always 0/1/2 there).
        if filter_y > 3 || filter_x > 3 {
            self.mark_corrupt(format!(
                "inter: out-of-range interp filter (y={filter_y}, x={filter_x})"
            ));
            return Stage::Stop;
        }
        if filter_y > 2 || filter_x > 2 {
            self.mark_unsupported(
                "inter: interp filter BILINEAR not supported in this decode envelope",
            );
            return Stage::Stop;
        }

        // tx_size (decodeframe.c:1179-1198, inter path): TX_MODE_SELECT + a
        // signalling block (bsize > BLOCK_4X4) + !skip -> the inter var-tx
        // quadtree (read_tx_size_vartx over txfm_partition_cdf; it stamps the
        // txfm-context arrays itself). Else the tx_mode fallback (TX_MODE_LARGEST
        // for the 64x64 / 16x16 / 64x66 targets — a single per-block tx, no
        // symbol). Every block in the 16x18 OBMC target is Select + skip=0 and
        // resolves to a UNIFORM leaf tiling (the strips stay TX_4X16; the OBMC
        // BLOCK_16X8 splits to 2x TX_8X8) — a non-uniform partition would need the
        // reconstruction-phase leaf walk (collect_vartx_leaves) and is guarded.
        let a_off = mi_col as usize;
        let l_off = (mi_row & 31) as usize;
        let bw4 = MI_SIZE_WIDE[bsize] as usize;
        let bh4 = MI_SIZE_HIGH[bsize] as usize;
        let mut vartx_leaf_grid: Vec<u8> = Vec::new();
        let mut vartx_non_uniform = false;
        let tx_size = if cfg.tx_mode == TxMode::Select && bsize > 0 && skip == 0 {
            let max_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
            let bw_u = TX_SIZE_WIDE_UNIT[max_tx] as i32;
            let bh_u = TX_SIZE_HIGH_UNIT[max_tx] as i32;
            let width_u = MI_SIZE_WIDE[bsize];
            let height_u = MI_SIZE_HIGH[bsize];
            let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
            let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
            let mut vartx_tx = max_tx;
            let mut first_leaf = -1i32;
            let mut non_uniform = false;
            vartx_leaf_grid = vec![0u8; bw4 * bh4];
            let mut idy = 0;
            while idy < height_u {
                let mut idx = 0;
                while idx < width_u {
                    read_tx_size_vartx(
                        dec,
                        &mut self.txfm_partition,
                        &mut self.above_t[a_off..],
                        &mut self.left_t[l_off..],
                        bsize,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                        max_tx,
                        0,
                        idy,
                        idx,
                        &mut vartx_tx,
                        &mut first_leaf,
                        &mut non_uniform,
                        &mut vartx_leaf_grid,
                        bw4,
                        bh4,
                    );
                    idx += bw_u;
                }
                idy += bh_u;
            }
            vartx_non_uniform = non_uniform;
            vartx_tx
        } else {
            // read_tx_size fallback (decodeframe.c:1179-1198): a SKIP inter block
            // (allow_select = !skip = false) or a non-SELECT frame reads NO tx-size
            // symbol and derives `tx_size_from_tx_mode`, then MUST stamp the
            // txfm-context arrays via `set_txfm_ctxs(tx, bw, bh, skip && is_inter)`.
            // The var-tx quadtree above stamps itself; this else-arm did not, so a
            // skip block left `above_t`/`left_t` at the init 64. A later var-tx
            // block whose above/left neighbour is that skip block then read a wrong
            // `txfm_partition` context whenever the true stamp (block width/height
            // px) straddles the tx dim — the q63 F1 desync (mi(16,0) skip
            // BLOCK_16X64 → above_t 16 in C vs 64 in the port; mi(32,0)'s TX_32X64
            // read saw above=1 in C, 0 in the port).
            let ts = tx_size_from_tx_mode(bsize, cfg.tx_mode);
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                ts,
                bw4,
                bh4,
                skip != 0,
            );
            ts
        };
        // The leaf grid drives the non-uniform reconstruction walk
        // (collect_vartx_leaves); every block in this target is uniform (guarded),
        // so the uniform residual loop below tiles with `tx_size` = the read size.
        let _ = &vartx_leaf_grid;
        if vartx_non_uniform {
            self.mark_unsupported(
                "inter: non-uniform var-tx not supported in this decode envelope",
            );
            return Stage::Stop;
        }

        // --- parse_decode_block tail (decodeframe.c:1219): a SKIP block resets
        // its entropy-context footprint to zero (`av1_reset_entropy_context`,
        // blockd.c:58) — plane 0 always, chroma planes when this block is the
        // chroma reference, each over its own plane_bsize footprint. This is NOT
        // intra-specific: C runs it for every skipped block before
        // `decode_token_recon_block`, and a skip block reads (and therefore
        // stamps) no coefficients, so without the reset the footprint keeps the
        // stale culs of whatever block last occupied those context cells.
        //
        // The port had this only on the intra path. It stayed invisible while the
        // probe pinned before any block could read across a skipped inter
        // neighbour's stale cells; the first real victim is mi(44,18) (the first
        // intra-in-inter block), whose LEFT neighbour mi(44,16) is a skipped inter
        // block: its stale non-zero culs flip mi(44,18)'s `txb_skip_ctx`, so all
        // three of its txbs still decode all-zero (the same symbol VALUES C reads)
        // but off a different `txb_skip_cdf` row — the arithmetic decoder drifts
        // and the next block's `skip_txfm` reads 0 where C reads 1.
        if skip != 0 {
            let a0 = mi_col as usize;
            let l0 = (mi_row & 31) as usize;
            self.above_e[0][a0..a0 + bw4].fill(0);
            self.left_e[0][l0..l0 + bh4].fill(0);
            let reset_chroma =
                !cfg.monochrome && is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y);
            if reset_chroma {
                // Same chroma-reference origin shift the coefficient loop uses
                // (setup_pred_plane's odd-position adjustment).
                let adj_row = if ss_y != 0 && (mi_row & 1) != 0 && MI_SIZE_HIGH[bsize] == 1 {
                    mi_row - 1
                } else {
                    mi_row
                };
                let adj_col = if ss_x != 0 && (mi_col & 1) != 0 && MI_SIZE_WIDE[bsize] == 1 {
                    mi_col - 1
                } else {
                    mi_col
                };
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let uw = MI_SIZE_WIDE[plane_bsize] as usize;
                let uh = MI_SIZE_HIGH[plane_bsize] as usize;
                let uv_a_base = (adj_col >> ss_x) as usize;
                let uv_l_base = ((adj_row & 31) >> ss_y) as usize;
                for plane in 1..=2 {
                    self.above_e[plane][uv_a_base..uv_a_base + uw].fill(0);
                    self.left_e[plane][uv_l_base..uv_l_base + uh].fill(0);
                }
            }
        }

        // Pack the mutually exclusive families: a compound block carries its
        // second ref/MV + blend as one `Compound` (masked xor weighted — the
        // comp_weights a masked block computed are dropped unused), and an
        // inter-intra blend only exists on a single-ref block.
        Stage::Continue(InterModeInfo {
            ref0,
            mode,
            mv_row,
            mv_col,
            compound: is_compound.then_some(Compound {
                ref1,
                mv1_row,
                mv1_col,
                comp_group_idx,
                compound_idx,
                kind: match masked {
                    Some(mask) => CompoundKind::Masked(mask),
                    None => CompoundKind::Weighted(comp_weights),
                },
            }),
            interintra: interintra.then_some(InterIntra {
                mode: ii_mode,
                use_wedge: ii_use_wedge != 0,
                wedge_idx: ii_wedge_idx,
            }),
            motion_mode,
            warp_luma,
            filter_y,
            filter_x,
            tx_size,
        })
    }

    /// `dec_build_inter_predictors` + `predict_inter_block` (decodeframe.c):
    /// the per-plane motion-compensation dispatch — single or compound
    /// predictor (masked / distance-weighted when coded), the local-warp and
    /// scaled-reference variants, then the inter-intra luma/chroma blend and
    /// the OBMC above/left feather for an OBMC_CAUSAL block. No entropy
    /// reads. Returns `Stage::Stop` on a named refusal / corrupt state; the
    /// caller then exits the block without reconstructing.
    fn predict_inter_block(
        &mut self,
        mi: &InterModeInfo,
        inter: &InterFrameCfg,
        bx: &BlockCtx,
    ) -> Stage<()> {
        use aom_dsp::entropy::partition as ep;
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            partition,
            chroma_ref,
            adj_row,
            adj_col,
            up_available,
            left_available,
            above_mi,
            left_mi,
            ..
        } = *bx;
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let &InterModeInfo {
            ref0,
            mv_row,
            mv_col,
            compound,
            interintra,
            motion_mode,
            warp_luma,
            filter_y,
            filter_x,
            ..
        } = mi;
        // --- motion compensation (predict phase; NO entropy reads) ---
        let (cmv_row, cmv_col) = clamp_mv_to_umv_border(mv_row, mv_col, mi_row, mi_col, bsize, cfg);
        // A compound block's SECOND MV clamps independently (dec_calc_subpel_params
        // runs per ref). `(0,0)` for single-ref (unused there).
        let (cmv1_row, cmv1_col) = compound.map_or((0, 0), |c| {
            clamp_mv_to_umv_border(c.mv1_row, c.mv1_col, mi_row, mi_col, bsize, cfg)
        });
        // Above bd8, nonzero-MV (sub-pel or integer-pel) motion compensation is
        // an `experimental-video` tool: without the feature the block is refused
        // by name; with it, `build_inter_predictor` dispatches to the u16
        // highbd convolve kernels (u8 scratch would truncate >8-bit samples).
        // Fail-loud, never corrupt. (Compound is feature-gated earlier — this
        // arm only ever sees single-ref.)
        #[cfg(not(feature = "experimental-video"))]
        if cfg.bd > 8 && (cmv_row != 0 || cmv_col != 0) {
            self.mark_unsupported("inter: sub/nonzero-pel MC above bd8 not yet supported");
            return Stage::Stop;
        }
        let bw_px = (MI_SIZE_WIDE[bsize] * 4) as usize;
        let bh_px = (MI_SIZE_HIGH[bsize] * 4) as usize;
        // Bind the block's coded reference (ref0 was range-checked at read).
        let Some(last) = inter.refs[(ref0 - 1) as usize] else {
            self.mark_corrupt(format!(
                "inter: block references unavailable ref {ref0} (no stored frame)"
            ));
            return Stage::Stop;
        };
        // `xd->block_ref_scale_factors[0]` — the bound ref's luma scale factors.
        let sf = &inter.ref_sf[(ref0 - 1) as usize];
        // Compound's second reference + its scale factors (`block_ref_scale_factors[1]`).
        let sf1 = compound.map_or(sf, |c| &inter.ref_sf[(c.ref1 - 1) as usize]);
        let blk_x = (mi_col * 4) as usize;
        let blk_y = (mi_row * 4) as usize;
        let dst_off = blk_y * self.stride + blk_x;
        // Masked compound: a `COMPOUND_DIFFWTD` `seg_mask` is built on luma
        // (`!conv_params.plane`, reconinter.c:655) and reused subsampled by the
        // chroma planes — keep one luma-resolution scratch across the block's
        // luma + both chroma masked blends. Wedge refetches the codebook mask
        // per plane and never touches it.
        let mut seg_mask = if compound.is_some_and(|c| matches!(c.kind, CompoundKind::Masked(_))) {
            vec![0u8; bw_px * bh_px]
        } else {
            Vec::new()
        };
        // Luma MC: WARPED_CAUSAL affine warp (av1_warp_plane) with a valid local
        // model; else translational. OBMC uses translational here (its overlap
        // blend runs after, below). Luma block is always >= 8 -> passes
        // av1_init_warp_params' per-plane size gate. `av1_allow_warp` also gates
        // on `!av1_is_scaled(sf)` — a scaled ref falls back to translational.
        // A COMPOUND block is always SIMPLE_TRANSLATION (motion_variation is
        // single-ref only), so it takes the plain two-ref translational arm.
        if let Some(compound) = compound {
            let Some(bck) = inter.refs[(compound.ref1 - 1) as usize] else {
                self.mark_corrupt(format!(
                    "inter: block references unavailable ref {} (no stored frame)",
                    compound.ref1
                ));
                return Stage::Stop;
            };
            let refs = [
                aom_dsp::inter::CompoundRefPlane {
                    plane: &last.y,
                    stride: last.stride,
                    w: last.width,
                    h: last.height,
                    sf,
                    mv: (cmv_row, cmv_col),
                    raw_mv: (mv_row, mv_col),
                },
                aom_dsp::inter::CompoundRefPlane {
                    plane: &bck.y,
                    stride: bck.stride,
                    w: bck.width,
                    h: bck.height,
                    sf: sf1,
                    mv: (cmv1_row, cmv1_col),
                    raw_mv: (compound.mv1_row, compound.mv1_col),
                },
            ];
            self.recon.with_wide_rect(
                dst_off,
                self.stride,
                bw_px,
                bh_px,
                &mut self.wide_rect,
                |dst, stride| {
                    match compound.kind {
                        // Masked (wedge/diffwtd): each ref to its own d16
                        // buffer, then blend via the luma-res mask.
                        CompoundKind::Masked(comp) => {
                            aom_dsp::inter::build_masked_compound_inter_predictor(
                                refs,
                                dst,
                                0,
                                stride,
                                blk_x,
                                blk_y,
                                bw_px,
                                bh_px,
                                0,
                                0,
                                filter_x,
                                filter_y,
                                cfg.bd as u32,
                                comp,
                                bsize,
                                true,
                                &mut seg_mask,
                            );
                        }
                        CompoundKind::Weighted(weights) => {
                            aom_dsp::inter::build_compound_inter_predictor(
                                refs,
                                dst,
                                0,
                                stride,
                                blk_x,
                                blk_y,
                                bw_px,
                                bh_px,
                                0,
                                0,
                                filter_x,
                                filter_y,
                                cfg.bd as u32,
                                weights,
                            );
                        }
                    }
                },
            );
        } else if let Some(wm) = warp_luma.filter(|_| !sf.is_scaled()) {
            self.recon.with_wide_rect(
                dst_off,
                self.stride,
                bw_px,
                bh_px,
                &mut self.wide_rect,
                |dst, stride| {
                    if cfg.bd > 8 {
                        // is_cur_buf_hbd: the general u16 affine kernel
                        // (av1_highbd_warp_affine_c) — the bd8 `warp_affine`
                        // specialization would round-clip at 8 bits.
                        let (round_0, round_1) = aom_dsp::inter::single_ref_rounds(cfg.bd as u32);
                        let cp = aom_dsp::inter::warp::WarpConvolveParams {
                            round_0,
                            round_1,
                            is_compound: false,
                            do_average: false,
                            use_dist_wtd_comp_avg: false,
                            fwd_offset: 0,
                            bck_offset: 0,
                        };
                        let mut dst16 = vec![0u16; bw_px * bh_px];
                        aom_dsp::inter::warp::highbd_warp_affine(
                            &wm.wmmat,
                            &last.y,
                            last.width,
                            last.height,
                            last.stride,
                            dst,
                            stride,
                            &mut dst16,
                            bw_px,
                            blk_x as i32,
                            blk_y as i32,
                            bw_px,
                            bh_px,
                            0,
                            0,
                            cfg.bd as u32,
                            &cp,
                            wm.alpha,
                            wm.beta,
                            wm.gamma,
                            wm.delta,
                        );
                    } else {
                        aom_dsp::inter::warp::warp_affine(
                            &wm.wmmat,
                            &last.y,
                            last.width,
                            last.height,
                            last.stride,
                            dst,
                            0,
                            stride,
                            blk_x as i32,
                            blk_y as i32,
                            bw_px,
                            bh_px,
                            0,
                            0,
                            wm.alpha,
                            wm.beta,
                            wm.gamma,
                            wm.delta,
                        );
                    }
                },
            );
        } else {
            self.recon.with_wide_rect(
                dst_off,
                self.stride,
                bw_px,
                bh_px,
                &mut self.wide_rect,
                |dst, stride| {
                    aom_dsp::inter::build_inter_predictor(
                        &last.y,
                        last.stride,
                        last.width,
                        last.height,
                        dst,
                        0,
                        stride,
                        blk_x,
                        blk_y,
                        bw_px,
                        bh_px,
                        cmv_row,
                        cmv_col,
                        mv_row,
                        mv_col,
                        0,
                        0,
                        filter_x,
                        filter_y,
                        cfg.bd as u32,
                        sf,
                    );
                },
            );
        }
        // Chroma prediction only at the chroma-reference block (sub-8x8 members
        // share one chroma block, coded on the group's bottom/right member); the
        // caller computed `chroma_ref`/`adj_*` (setup_pred_plane's odd-position
        // shared-group origin shift) since recon needs the same geometry.
        if let Some(compound) = compound.filter(|_| chroma_ref) {
            // Compound chroma: a compound block is always min-dim >= 8, so the
            // sub-8x8 sharing path can never apply — one whole-block combine
            // per chroma plane. Each ref's MV clamps against the CHROMA dims
            // (clamp_mv_to_umv_border_plane, as the single-ref path does).
            let Some(bck) = inter.refs[(compound.ref1 - 1) as usize] else {
                self.mark_corrupt(format!(
                    "inter: block references unavailable ref {} (no stored frame)",
                    compound.ref1
                ));
                return Stage::Stop;
            };
            let bw_uv = bw_px >> ss_x;
            let bh_uv = bh_px >> ss_y;
            let uv_org_x = ((adj_col * 4) >> ss_x) as usize;
            let uv_org_y = ((adj_row * 4) >> ss_y) as usize;
            let (cmv0r_uv, cmv0c_uv) = clamp_mv_to_umv_border_plane(
                mv_row,
                mv_col,
                mi_row,
                mi_col,
                bsize,
                bw_uv as i32,
                bh_uv as i32,
                ss_x,
                ss_y,
                cfg,
            );
            let (cmv1r_uv, cmv1c_uv) = clamp_mv_to_umv_border_plane(
                compound.mv1_row,
                compound.mv1_col,
                mi_row,
                mi_col,
                bsize,
                bw_uv as i32,
                bh_uv as i32,
                ss_x,
                ss_y,
                cfg,
            );
            let doff = uv_org_y * self.stride_uv + uv_org_x;
            for (dst_plane, (p0, p1)) in [
                (&mut self.recon_u, (&last.u, &bck.u)),
                (&mut self.recon_v, (&last.v, &bck.v)),
            ] {
                let refs = [
                    aom_dsp::inter::CompoundRefPlane {
                        plane: p0,
                        stride: last.stride_uv,
                        w: last.width_uv,
                        h: last.height_uv,
                        sf,
                        mv: (cmv0r_uv, cmv0c_uv),
                        raw_mv: (mv_row, mv_col),
                    },
                    aom_dsp::inter::CompoundRefPlane {
                        plane: p1,
                        stride: bck.stride_uv,
                        w: bck.width_uv,
                        h: bck.height_uv,
                        sf: sf1,
                        mv: (cmv1r_uv, cmv1c_uv),
                        raw_mv: (compound.mv1_row, compound.mv1_col),
                    },
                ];
                dst_plane.with_wide_rect(
                    doff,
                    self.stride_uv,
                    bw_uv,
                    bh_uv,
                    &mut self.wide_rect,
                    |dst, stride| {
                        match compound.kind {
                            // Chroma masked: chroma dims + subsampling; the
                            // mask stays luma-resolution (`sb_type`/`mask_stride`
                            // from `bsize`, seg_mask already built on luma).
                            CompoundKind::Masked(comp) => {
                                aom_dsp::inter::build_masked_compound_inter_predictor(
                                    refs,
                                    dst,
                                    0,
                                    stride,
                                    uv_org_x,
                                    uv_org_y,
                                    bw_uv,
                                    bh_uv,
                                    ss_x,
                                    ss_y,
                                    filter_x,
                                    filter_y,
                                    cfg.bd as u32,
                                    comp,
                                    bsize,
                                    false,
                                    &mut seg_mask,
                                );
                            }
                            CompoundKind::Weighted(weights) => {
                                aom_dsp::inter::build_compound_inter_predictor(
                                    refs,
                                    dst,
                                    0,
                                    stride,
                                    uv_org_x,
                                    uv_org_y,
                                    bw_uv,
                                    bh_uv,
                                    ss_x,
                                    ss_y,
                                    filter_x,
                                    filter_y,
                                    cfg.bd as u32,
                                    weights,
                                );
                            }
                        }
                    },
                );
            }
        } else if chroma_ref {
            let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
            let uv_org_x = ((adj_col * 4) >> ss_x) as usize;
            let uv_org_y = ((adj_row * 4) >> ss_y) as usize;
            // is_sub8x8_inter (chroma): a luma side == 4 on a subsampled axis means
            // the chroma covers > 1 luma block; predict per covered luma block using
            // its OWN MV (build_inter_predictors_sub8x8). Every covered block is
            // inter here (all-inter frame). Otherwise a single whole-block chroma MC.
            let is_sub4_x = BLOCK_SIZE_WIDE[bsize] == 4 && ss_x != 0;
            let is_sub4_y = BLOCK_SIZE_HIGH[bsize] == 4 && ss_y != 0;
            if is_sub4_x || is_sub4_y {
                let b4_w = (BLOCK_SIZE_WIDE[bsize] >> ss_x) as usize;
                let b4_h = (BLOCK_SIZE_HIGH[bsize] >> ss_y) as usize;
                let b8_w = BLOCK_SIZE_WIDE[plane_bsize] as usize;
                let b8_h = BLOCK_SIZE_HIGH[plane_bsize] as usize;
                let row_start: i32 = if is_sub4_y { -1 } else { 0 };
                let col_start: i32 = if is_sub4_x { -1 } else { 0 };
                let cols = cfg.mi_cols;
                let mut y = 0usize;
                let mut row = row_start;
                while y < b8_h {
                    let mut x = 0usize;
                    let mut col = col_start;
                    while x < b8_w {
                        // The covered sub-block's own MV: xd->mi[row*stride+col].
                        // The current block (row==0 && col==0) is not yet stamped
                        // into `mi_dv` (stamp is at the end of this fn), so use the
                        // just-decoded MV; neighbours come from the grid.
                        let (smv_r, smv_c) = if row == 0 && col == 0 {
                            (mv_row, mv_col)
                        } else {
                            let d = DvNbr::from_packed(
                                self.mi_dv[((mi_row + row) * cols + (mi_col + col)) as usize],
                            );
                            (d.mv0_row, d.mv0_col)
                        };
                        // Per-plane clamp for this covered sub-block (C's sub8x8
                        // predictor runs dec_calc_subpel_params per b4 with the
                        // block's shared luma edges + the b4 dims). `(smv_r,
                        // smv_c)` stays the RAW coded MV — the scaled-MC arm
                        // reads it instead.
                        let (csmv_r, csmv_c) = clamp_mv_to_umv_border_plane(
                            smv_r,
                            smv_c,
                            mi_row,
                            mi_col,
                            bsize,
                            b4_w as i32,
                            b4_h as i32,
                            ss_x,
                            ss_y,
                            cfg,
                        );
                        let bxu = uv_org_x + x;
                        let byu = uv_org_y + y;
                        let doff = byu * self.stride_uv + bxu;
                        for (dst, src) in
                            [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)]
                        {
                            dst.with_wide_rect(
                                doff,
                                self.stride_uv,
                                b4_w,
                                b4_h,
                                &mut self.wide_rect,
                                |d, stride| {
                                    aom_dsp::inter::build_inter_predictor(
                                        src,
                                        last.stride_uv,
                                        last.width_uv,
                                        last.height_uv,
                                        d,
                                        0,
                                        stride,
                                        bxu,
                                        byu,
                                        b4_w,
                                        b4_h,
                                        csmv_r,
                                        csmv_c,
                                        smv_r,
                                        smv_c,
                                        ss_x,
                                        ss_y,
                                        filter_x,
                                        filter_y,
                                        cfg.bd as u32,
                                        sf,
                                    );
                                },
                            );
                        }
                        x += b4_w;
                        col += 1;
                    }
                    y += b4_h;
                    row += 1;
                }
            } else {
                let bw_uv = bw_px >> ss_x;
                let bh_uv = bh_px >> ss_y;
                // Per-plane MV clamp (clamp_mv_to_umv_border_sb with the CHROMA
                // subsampling): C re-clamps the RAW block MV against chroma-scaled
                // edges + the chroma prediction dims (decodeframe.c:625). Reusing
                // the luma clamp (`cmv_*`) is wrong once the clamp fires — the
                // chroma bounds differ (edges * (1<<(1-ss)), spel from `bw_uv`).
                let (cmv_row_uv, cmv_col_uv) = clamp_mv_to_umv_border_plane(
                    mv_row,
                    mv_col,
                    mi_row,
                    mi_col,
                    bsize,
                    bw_uv as i32,
                    bh_uv as i32,
                    ss_x,
                    ss_y,
                    cfg,
                );
                let doff = uv_org_y * self.stride_uv + uv_org_x;
                // Chroma warp only when the (subsampled) plane block is >= 8 in both
                // dims (av1_init_warp_params); otherwise translational. Same
                // `av1_allow_warp` scale gate as luma.
                let warp_chroma = warp_luma.filter(|_| bw_uv >= 8 && bh_uv >= 8 && !sf.is_scaled());
                if let Some(wm) = warp_chroma {
                    for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                        dst.with_wide_rect(
                            doff,
                            self.stride_uv,
                            bw_uv,
                            bh_uv,
                            &mut self.wide_rect,
                            |d, stride| {
                                if cfg.bd > 8 {
                                    let (round_0, round_1) =
                                        aom_dsp::inter::single_ref_rounds(cfg.bd as u32);
                                    let cp = aom_dsp::inter::warp::WarpConvolveParams {
                                        round_0,
                                        round_1,
                                        is_compound: false,
                                        do_average: false,
                                        use_dist_wtd_comp_avg: false,
                                        fwd_offset: 0,
                                        bck_offset: 0,
                                    };
                                    let mut dst16 = vec![0u16; bw_uv * bh_uv];
                                    aom_dsp::inter::warp::highbd_warp_affine(
                                        &wm.wmmat,
                                        src,
                                        last.width_uv,
                                        last.height_uv,
                                        last.stride_uv,
                                        d,
                                        stride,
                                        &mut dst16,
                                        bw_uv,
                                        uv_org_x as i32,
                                        uv_org_y as i32,
                                        bw_uv,
                                        bh_uv,
                                        ss_x,
                                        ss_y,
                                        cfg.bd as u32,
                                        &cp,
                                        wm.alpha,
                                        wm.beta,
                                        wm.gamma,
                                        wm.delta,
                                    );
                                } else {
                                    aom_dsp::inter::warp::warp_affine(
                                        &wm.wmmat,
                                        src,
                                        last.width_uv,
                                        last.height_uv,
                                        last.stride_uv,
                                        d,
                                        0,
                                        stride,
                                        uv_org_x as i32,
                                        uv_org_y as i32,
                                        bw_uv,
                                        bh_uv,
                                        ss_x,
                                        ss_y,
                                        wm.alpha,
                                        wm.beta,
                                        wm.gamma,
                                        wm.delta,
                                    );
                                }
                            },
                        );
                    }
                } else {
                    // Translational chroma MC uses the PER-PLANE clamped MV
                    // (cmv_*_uv), not the luma-clamped cmv_* — see the
                    // clamp_mv_to_umv_border_plane call above.
                    for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                        dst.with_wide_rect(
                            doff,
                            self.stride_uv,
                            bw_uv,
                            bh_uv,
                            &mut self.wide_rect,
                            |d, stride| {
                                aom_dsp::inter::build_inter_predictor(
                                    src,
                                    last.stride_uv,
                                    last.width_uv,
                                    last.height_uv,
                                    d,
                                    0,
                                    stride,
                                    uv_org_x,
                                    uv_org_y,
                                    bw_uv,
                                    bh_uv,
                                    cmv_row_uv,
                                    cmv_col_uv,
                                    mv_row,
                                    mv_col,
                                    ss_x,
                                    ss_y,
                                    filter_x,
                                    filter_y,
                                    cfg.bd as u32,
                                    sf,
                                );
                            },
                        );
                    }
                }
            }
        }

        // --- INTER-INTRA (av1_build_interintra_predictor, reconinter.c:1162): for
        // each plane, build an intra predictor over the WHOLE plane block and blend
        // it onto that plane's freshly-built inter predictor.
        //
        // C runs this inside `dec_build_inter_predictors`' plane loop, immediately
        // after each plane's inter prediction lands in `dst` (decodeframe.c:694-702).
        // Doing all planes here instead is equivalent: the luma blend writes only
        // inside the luma block and chroma MC reads the REFERENCE frame, never the
        // current recon, so no plane's inputs depend on another plane's blend.
        //
        // Two conventions must not be crossed (they differ between the arms, see
        // `combine_interintra`): the SMOOTH mask is prebuilt at PLANE resolution and
        // blended 1:1, while the WEDGE mask is at LUMA resolution and box-averaged
        // down by `subw`/`subh` inside the blend.
        if let Some(ii) = interintra {
            let ii_intra_mode =
                aom_dsp::inter::interintra::INTERINTRA_TO_INTRA_MODE[ii.mode as usize];
            // `get_intra_edge_filter_type` (reconintra.c:974). Inert for every
            // inter-intra block — the four modes are DC/V/H/SMOOTH and the angle
            // delta is forced to 0 (decodemv.c:1395), so V/H predict at exactly
            // 90/180 where the edge filter is skipped — but derived faithfully.
            let is_smooth = |m: Option<MiNbrKf>| {
                m.is_some_and(|n| (SMOOTH_PRED..=SMOOTH_H_PRED).contains(&n.y_mode))
            };
            let filt_type = (is_smooth(above_mi) || is_smooth(left_mi)) as i32;

            // Luma.
            let ii_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
            let (n_top, n_tr, n_left, n_bl) = ep::intra_avail(
                self.st.sb_size,
                bsize,
                mi_row,
                mi_col,
                up_available,
                left_available,
                self.tile.mi_col_end,
                self.tile.mi_row_end,
                partition,
                ii_tx,
                0,
                0,
                0,
                0,
                BLOCK_SIZE_WIDE[bsize],
                BLOCK_SIZE_HIGH[bsize],
                cfg.mi_cols,
                cfg.mi_rows,
                ii_intra_mode,
                0,
                false,
            );
            let mut intra_pred = vec![0u16; bw_px * bh_px];
            let n_top_u = usize::try_from(n_top).expect("n_top_px must be non-negative");
            let n_left_u = usize::try_from(n_left).expect("n_left_px must be non-negative");
            with_wide_apron(
                &self.recon,
                dst_off,
                self.stride,
                n_top_u,
                n_tr,
                n_left_u,
                n_bl,
                &mut self.wide_apron,
                |src, roff, rstride| {
                    predict_intra_high(
                        src,
                        roff,
                        rstride,
                        &mut intra_pred,
                        bw_px,
                        ii_intra_mode,
                        0,
                        false,
                        0,
                        cfg.disable_edge_filter,
                        filt_type,
                        ii_tx,
                        n_top_u,
                        n_tr,
                        n_left_u,
                        n_bl,
                        cfg.bd,
                    );
                },
            );
            // C blends in place (`comppred == interpred`). Every output pixel is a
            // function of the SAME position in both sources, so lifting the inter
            // predictor into a scratch first is numerically identical and lets the
            // destination be borrowed mutably.
            let mut inter_pred = vec![0u16; bw_px * bh_px];
            for r in 0..bh_px {
                let s = dst_off + r * self.stride;
                self.recon
                    .copy_row_wide(s, &mut inter_pred[r * bw_px..(r + 1) * bw_px]);
            }
            self.recon.with_wide_rect(
                dst_off,
                self.stride,
                bw_px,
                bh_px,
                &mut self.wide_rect,
                |dst, stride| {
                    aom_dsp::inter::interintra::combine_interintra(
                        ii.mode as usize,
                        ii.use_wedge,
                        ii.wedge_idx as usize,
                        bsize,
                        bsize,
                        dst,
                        stride,
                        &inter_pred,
                        bw_px,
                        &intra_pred,
                        bw_px,
                    );
                },
            );

            // Chroma. `is_chroma_ref` is always true for an inter-intra bsize
            // (>= BLOCK_8X8), so C's `plane && !is_chroma_ref` break never fires.
            if chroma_ref {
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let uv_tx = MAX_TXSIZE_RECT_LOOKUP[plane_bsize];
                let uw = BLOCK_SIZE_WIDE[plane_bsize] as usize;
                let uh = BLOCK_SIZE_HIGH[plane_bsize] as usize;
                let wpx = ((MI_SIZE_WIDE[bsize] * 4) >> ss_x).max(4);
                let hpx = ((MI_SIZE_HIGH[bsize] * 4) >> ss_y).max(4);
                let bsize_uv = scale_chroma_bsize(bsize, ss_x, ss_y);
                let up_uv = adj_row > self.tile.mi_row_start;
                let left_uv = adj_col > self.tile.mi_col_start;
                let uv_org_x = ((adj_col * 4) >> ss_x) as usize;
                let uv_org_y = ((adj_row * 4) >> ss_y) as usize;
                let uv_off = uv_org_y * self.stride_uv + uv_org_x;
                let (n_top, n_tr, n_left, n_bl) = ep::intra_avail(
                    self.st.sb_size,
                    bsize_uv,
                    adj_row,
                    adj_col,
                    up_uv,
                    left_uv,
                    self.tile.mi_col_end,
                    self.tile.mi_row_end,
                    partition,
                    uv_tx,
                    ss_x as i32,
                    ss_y as i32,
                    0,
                    0,
                    wpx,
                    hpx,
                    cfg.mi_cols,
                    cfg.mi_rows,
                    ii_intra_mode,
                    0,
                    false,
                );
                let mut intra_uv = vec![0u16; uw * uh];
                let mut inter_uv = vec![0u16; uw * uh];
                let n_top_u = usize::try_from(n_top).expect("n_top_px must be non-negative");
                let n_left_u = usize::try_from(n_left).expect("n_left_px must be non-negative");
                for plane in 1..=2 {
                    let src_plane: &ReconPlane = if plane == 1 {
                        &self.recon_u
                    } else {
                        &self.recon_v
                    };
                    with_wide_apron(
                        src_plane,
                        uv_off,
                        self.stride_uv,
                        n_top_u,
                        n_tr,
                        n_left_u,
                        n_bl,
                        &mut self.wide_apron,
                        |src, roff, rstride| {
                            predict_intra_high(
                                src,
                                roff,
                                rstride,
                                &mut intra_uv,
                                uw,
                                ii_intra_mode,
                                0,
                                false,
                                0,
                                cfg.disable_edge_filter,
                                filt_type,
                                uv_tx,
                                n_top_u,
                                n_tr,
                                n_left_u,
                                n_bl,
                                cfg.bd,
                            );
                        },
                    );
                    for r in 0..uh {
                        let s = uv_off + r * self.stride_uv;
                        src_plane.copy_row_wide(s, &mut inter_uv[r * uw..(r + 1) * uw]);
                    }
                    let dst_plane: &mut ReconPlane = if plane == 1 {
                        &mut self.recon_u
                    } else {
                        &mut self.recon_v
                    };
                    dst_plane.with_wide_rect(
                        uv_off,
                        self.stride_uv,
                        uw,
                        uh,
                        &mut self.wide_rect,
                        |dst, stride| {
                            aom_dsp::inter::interintra::combine_interintra(
                                ii.mode as usize,
                                ii.use_wedge,
                                ii.wedge_idx as usize,
                                bsize,
                                plane_bsize,
                                dst,
                                stride,
                                &inter_uv,
                                uw,
                                &intra_uv,
                                uw,
                            );
                        },
                    );
                }
            }
        }

        // --- OBMC (predict_inter_block, decodeframe.c:878): after the block's own
        // predictor is built (luma + chroma above) and BEFORE the residual add,
        // an OBMC_CAUSAL block feather-blends its predictor with predictors from
        // its overlappable above/left neighbours' motion. Non-switchable frame ->
        // every neighbour shares the frame filter (filter_x == filter_y); a
        // switchable frame would feed each neighbour's own stored filter (chunk 6).
        if motion_mode == 1 {
            self.obmc_above_blend(mi_row, mi_col, bsize, inter);
            self.obmc_left_blend(mi_row, mi_col, bsize, inter);
        }
        Stage::Continue(())
    }

    /// `decode_token_recon_block`'s inter path: read the residual
    /// coefficients (luma txbs then, at the chroma-reference block, the U/V
    /// txbs) and add them onto the predictor built by
    /// [`Self::predict_inter_block`], persist the adapted inter mode-info
    /// CDFs, stamp the neighbour grids (`mi`/`dv`/interp/`mvs`), and push the
    /// block record the loop-filter / output structures consume.
    fn recon_inter_block(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        icdfs: InterCdfs,
        mi: InterModeInfo,
        inter: &InterFrameCfg,
        bx: &BlockCtx,
    ) {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            partition,
            skip,
            cdef_strength,
            chroma_ref,
            adj_row,
            adj_col,
            ..
        } = *bx;
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let InterModeInfo {
            ref0,
            mode,
            mv_row,
            mv_col,
            compound,
            filter_y,
            filter_x,
            tx_size,
            ..
        } = mi;
        // A single-ref block's neighbour-grid stamp carries the C defaults
        // (`ref_frame[1]` = the inter-intra rewrite or NONE, zero MV, the
        // non-coded ctx seeds 0/1).
        let ref1 = mi.ref1();
        let (mv1_row, mv1_col, comp_group_idx, compound_idx) = compound.map_or((0, 0, 0, 1), |c| {
            (c.mv1_row, c.mv1_col, c.comp_group_idx, c.compound_idx)
        });
        // --- reconstruction: read residual coefficients + ADD onto the MC
        // prediction (decode_token_recon_block inter path). Skip blocks read no
        // coeffs. tx_size is the (uniform) var-tx / LARGEST per-block luma tx
        // tiling the block, then (at the chroma reference) one U and one V tx, in
        // that plane order — all within the single <=64px 64x64 chunk the target's
        // blocks occupy.
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let mut luma_tt0 = 0usize; // co-located luma tx-type for inter chroma
        if skip == 0 {
            // av1_read_tx_type gate: !skip && qindex(seg,base) > 0. Segmentation
            // is off in this envelope (asserted above) so segment_id == 0.
            let signal_gate = av1_get_qindex(&cfg.seg, 0, cfg.base_qindex) > 0;
            let max_blocks_wide = max_block_units(BLOCK_SIZE_WIDE[bsize], mb_to_right_edge);
            let max_blocks_high = max_block_units(BLOCK_SIZE_HIGH[bsize], mb_to_bottom_edge);
            // Luma plane.
            let txw = TX_SIZE_WIDE_UNIT[tx_size];
            let txh = TX_SIZE_HIGH_UNIT[tx_size];
            let larea = txb_wide(tx_size) * txb_high(tx_size);
            let mut tcoeff = vec![0i32; larea];
            let mut blk_row = 0usize;
            while blk_row < max_blocks_high {
                let mut blk_col = 0usize;
                while blk_col < max_blocks_wide {
                    let a0 = mi_col as usize + blk_col;
                    let l0 = (mi_row & 31) as usize + blk_row;
                    let (tsc, dsc) = get_txb_ctx(
                        bsize,
                        tx_size,
                        0,
                        &self.above_e[0][a0..],
                        &self.left_e[0][l0..],
                    );
                    // is_inter block -> av1_read_tx_type selects the inter ext-tx CDF.
                    let ext = inter_ext_tx_cdf(&mut cdfs.inter_ext_tx, tx_size, cfg.reduced_tx_set);
                    let (eob, tt) = read_coeffs_txb_full(
                        dec,
                        &mut cdfs.coeff,
                        ext,
                        &mut tcoeff,
                        tx_size,
                        0,
                        tsc as usize,
                        dsc as usize,
                        true,
                        true,
                        cfg.reduced_tx_set,
                        signal_gate,
                        0,
                    );
                    if blk_row == 0 && blk_col == 0 {
                        luma_tt0 = tt;
                    }
                    let cul = txb_entropy_context(&tcoeff, tx_size, tt, eob) as i8;
                    self.set_entropy_ctx(
                        0,
                        cul,
                        mi_col as usize,
                        (mi_row & 31) as usize,
                        blk_row,
                        blk_col,
                        txw,
                        txh,
                        max_blocks_wide,
                        max_blocks_high,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                    );
                    if eob > 0 {
                        let off = ((mi_row * 4) as usize + blk_row * 4) * self.stride
                            + (mi_col * 4) as usize
                            + blk_col * 4;
                        let iqm = qm::iqmatrix(self.block_qm_level[0], 0, tx_size, tt);
                        let dequant = self.dequants[0];
                        match &mut self.recon {
                            ReconPlane::HighBd(p) => reconstruct_txb_into(
                                &mut p[off..],
                                self.stride,
                                tx_size,
                                tt,
                                &tcoeff,
                                dequant,
                                iqm,
                                cfg.bd,
                                &mut self.recon_scratch,
                            ),
                            ReconPlane::LowBd(p) => reconstruct_txb_u8_into(
                                &mut p[off..],
                                self.stride,
                                tx_size,
                                tt,
                                &tcoeff,
                                dequant,
                                iqm,
                                &mut self.recon_scratch,
                            ),
                        }
                    }
                    blk_col += txw;
                }
                blk_row += txh;
            }
            // Chroma planes (only at the chroma reference block).
            if chroma_ref {
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let uv_tx = max_uv_txsize(bsize, ss_x, ss_y);
                let uv_txw = TX_SIZE_WIDE_UNIT[uv_tx];
                let uv_txh = TX_SIZE_HIGH_UNIT[uv_tx];
                let uv_area = txb_wide(uv_tx) * txb_high(uv_tx);
                let uv_a_base = (adj_col >> ss_x) as usize;
                let uv_l_base = ((adj_row & 31) >> ss_y) as usize;
                let uv_org_x = ((adj_col * 4) >> ss_x) as usize;
                let uv_org_y = ((adj_row * 4) >> ss_y) as usize;
                let blocks_wide_uv =
                    max_block_units_ss(BLOCK_SIZE_WIDE[plane_bsize], mb_to_right_edge, ss_x);
                let blocks_high_uv =
                    max_block_units_ss(BLOCK_SIZE_HIGH[plane_bsize], mb_to_bottom_edge, ss_y);
                // Inter chroma tx-type = the CO-LOCATED luma tx-type
                // (av1_get_tx_type is_inter branch), validated against the uv inter
                // ext-tx set and demoted to DCT_DCT when unused. For this uniform
                // single-tx block the co-location is the block's own luma tx-type.
                let tt_uv = if ext_tx_derive(uv_tx, true, cfg.reduced_tx_set, luma_tt0, false, 0, 0)
                    .used
                    == 1
                {
                    luma_tt0
                } else {
                    0
                };
                let mut tcoeff_uv = vec![0i32; uv_area];
                let mut no_ext: [u16; 0] = [];
                for plane in 1..=2usize {
                    let mut blk_row = 0usize;
                    while blk_row < blocks_high_uv {
                        let mut blk_col = 0usize;
                        while blk_col < blocks_wide_uv {
                            let (tsc, dsc) = get_txb_ctx(
                                plane_bsize,
                                uv_tx,
                                plane,
                                &self.above_e[plane][uv_a_base + blk_col..],
                                &self.left_e[plane][uv_l_base + blk_row..],
                            );
                            let (eob, _tt) = read_coeffs_txb_full(
                                dec,
                                &mut cdfs.coeff,
                                &mut no_ext,
                                &mut tcoeff_uv,
                                uv_tx,
                                1,
                                tsc as usize,
                                dsc as usize,
                                true,
                                false,
                                cfg.reduced_tx_set,
                                false,
                                tt_uv,
                            );
                            let cul = txb_entropy_context(&tcoeff_uv, uv_tx, tt_uv, eob) as i8;
                            self.set_entropy_ctx(
                                plane,
                                cul,
                                uv_a_base,
                                uv_l_base,
                                blk_row,
                                blk_col,
                                uv_txw,
                                uv_txh,
                                blocks_wide_uv,
                                blocks_high_uv,
                                mb_to_right_edge,
                                mb_to_bottom_edge,
                            );
                            if eob > 0 {
                                let off = (uv_org_y + blk_row * 4) * self.stride_uv
                                    + uv_org_x
                                    + blk_col * 4;
                                let iqm =
                                    qm::iqmatrix(self.block_qm_level[plane], plane, uv_tx, tt_uv);
                                let dst = if plane == 1 {
                                    &mut self.recon_u
                                } else {
                                    &mut self.recon_v
                                };
                                let dequant = self.dequants[plane];
                                match dst {
                                    ReconPlane::HighBd(p) => reconstruct_txb_into(
                                        &mut p[off..],
                                        self.stride_uv,
                                        uv_tx,
                                        tt_uv,
                                        &tcoeff_uv,
                                        dequant,
                                        iqm,
                                        cfg.bd,
                                        &mut self.recon_scratch,
                                    ),
                                    ReconPlane::LowBd(p) => reconstruct_txb_u8_into(
                                        &mut p[off..],
                                        self.stride_uv,
                                        uv_tx,
                                        tt_uv,
                                        &tcoeff_uv,
                                        dequant,
                                        iqm,
                                        &mut self.recon_scratch,
                                    ),
                                }
                            }
                            blk_col += uv_txw;
                        }
                        blk_row += uv_txh;
                    }
                }
            }
        }
        // Persist the adapted inter mode-info CDFs for the next inter block.
        self.inter_cdfs = icdfs;

        // Stamp the neighbour grids (inert for a single block; needed for the
        // ratchet's spatial scan + skip/interp contexts).
        self.stamp_mi(
            mi_row,
            mi_col,
            bsize,
            MiNbrKf {
                y_mode: 0,
                skip_txfm: skip,
            },
        );
        self.stamp_dv(
            mi_row,
            mi_col,
            bsize,
            DvNbr {
                bsize,
                ref_frame0: ref0,
                ref_frame1: ref1,
                use_intrabc: false,
                mode,
                mv0_row: mv_row,
                mv0_col: mv_col,
                mv1_row,
                mv1_col,
                compound_idx,
                comp_group_idx,
            },
        );
        // Stamp the block's coded interp filters so later switchable inter blocks'
        // av1_get_pred_context_switchable_interp neighbour reads see them.
        self.stamp_interp(mi_row, mi_col, bsize, (filter_y as u8, filter_x as u8));
        if dbg_blocks() {
            aom_dsp::trace_out!(
                "BLK mi({mi_row},{mi_col}) bs={bsize} inter ref={ref0},{ref1} mode={mode} skip={skip} mv=({mv_row},{mv_col}),({mv1_row},{mv1_col}) f=({filter_y},{filter_x})"
            );
        }
        // (av1_copy_frame_mvs) for LATER frames' temporal projection.
        self.stamp_frame_mvs(
            inter,
            mi_row,
            mi_col,
            bsize,
            [ref0, ref1],
            [(mv_row, mv_col), (mv1_row, mv1_col)],
        );
        // Minimal per-block record for the post-filter / output structures. The
        // inter block carries no intra fields; `skip`/`current_qindex` are what
        // the deblock reads (the mode-info's inter-ness lives in the DV grid).
        let info = MbModeInfoKf {
            segment_id: 0,
            skip,
            // The per-64x64-unit CDEF strength `read_cdef` returned for THIS block
            // (-1 = skip / unit already coded). The frame CDEF walk (apply_cdef)
            // stamps it on every covered 64x64 unit, exactly as for intra blocks.
            cdef_strength,
            current_qindex: cfg.base_qindex,
            delta_lf: [0; 4],
            delta_lf_from_base: 0,
            use_intrabc: 0,
            dv_row: 0,
            dv_col: 0,
            y_mode: 0,
            angle_delta_y: 0,
            uv_mode: 0,
            cfl_alpha_idx: 0,
            cfl_joint_sign: 0,
            angle_delta_uv: 0,
            palette_size: [0, 0],
            palette_colors: [0; 24],
            use_filter_intra: 0,
            filter_intra_mode: 0,
        };
        self.tree.push(partition as i8);
        self.blocks.push(DecodedBlockKf {
            mi_row,
            mi_col,
            bsize,
            partition,
            info,
            tx_size,
            txbs: Vec::new(),
            txbs_uv: Vec::new(),
            // Inter block: carry ref_frame[0] + mode for the loop-filter grid
            // (build_lf_inputs derives is_inter/ref/mode_lf from this).
            inter_lf: Some((ref0, mode)),
        });
    }

    /// One leaf block: `parse_decode_block` (mode info + tx sizing + skip
    /// entropy-reset) followed by the intra `decode_token_recon_block` txb loop.
    /// The INTER-frame single-block mode-info + motion-compensation path,
    /// mirroring `read_inter_frame_mode_info` + `read_inter_block_mode_info`
    /// (decodemv.c) then `dec_build_inter_predictors` (decodeframe.c). The
    /// walking-skeleton envelope was single LAST reference, `SINGLE_REFERENCE`,
    /// `SIMPLE_TRANSLATION`, `skip = 1`; chunk 4 adds `OBMC_CAUSAL` (motion_mode
    /// read + above/left neighbour feather-blend) and inter var-tx
    /// (`TX_MODE_SELECT`). The pre-mode reads that are no-ops in this envelope
    /// (segment_id, skip_mode, cdef-for-skip, delta-q) are asserted off.
    pub(crate) fn decode_block_inter(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
        inter: &InterFrameCfg,
    ) {
        use aom_dsp::entropy::partition as ep;

        let cfg = self.cfg;

        if dbg_blocks() {
            aom_dsp::trace_out!(
                "ENTER mi({mi_row},{mi_col}) bs={bsize} tellq={}",
                dec.tell_frac() as i32
            );
        }

        // Envelope invariants (STEP-0 census): these pre-mode reads are inert.
        // Out-of-envelope inter features are rejected as a clean error (not a
        // panic) so a malformed / unsupported inter frame from untrusted input
        // returns `Err` instead of aborting the decode.
        if cfg.seg.enabled {
            self.mark_unsupported("inter: segmentation not supported in this decode envelope");
            return;
        }
        if inter.skip_mode_present {
            self.mark_unsupported("inter: skip_mode not supported in this decode envelope");
            return;
        }
        if cfg.delta_q_present {
            self.mark_unsupported("inter: delta-q not supported in this decode envelope");
            return;
        }
        // tx_mode is TX_MODE_SELECT for the OBMC target (av1-1-b8-01-size-16x18):
        // inter blocks code a var-tx quadtree (read_tx_size_vartx). The earlier
        // walking-skeleton / ratchet targets were TX_MODE_LARGEST; both are
        // handled below (the var-tx read collapses to the single largest tx when
        // the frame codes LARGEST).
        if !matches!(cfg.tx_mode, TxMode::Largest | TxMode::Select) {
            self.mark_unsupported("inter: tx_mode ONLY_4X4 not supported in this decode envelope");
            return;
        }

        // Neighbour projections + shared per-block values, bundled once.
        let mut bx = self.block_ctx(mi_row, mi_col, bsize, partition);
        let dv_inter = |d: DvNbr| d.use_intrabc || d.ref_frame0 > 0;

        // Snapshot the tile's persistent inter CDFs; every read below adapts this
        // local copy (via `read_symbol`/`update_cdf` when `dec.allow_update_cdf`),
        // and it is persisted back to `self.inter_cdfs` at the end of the block so
        // the next inter block sees the adapted state (matching C's in-place
        // `tile_data->tctx` adaptation).
        let mut icdfs = self.inter_cdfs;

        // --- read_inter_frame_mode_info pre-mode reads ---
        // segment_id (seg off -> 0); skip_mode (allowed off -> 0): no reads.
        // read_skip_txfm.
        let skip_ctx = ep::skip_txfm_context(
            bx.above_mi.map_or(0, |m| m.skip_txfm),
            bx.left_mi.map_or(0, |m| m.skip_txfm),
        ) as usize;
        bx.skip = ep::read_skip(dec, &mut cdfs.skip[skip_ctx], false);
        // read_cdef (decodemv.c, ordered after read_skip, before read_delta_q /
        // read_is_inter): the FIRST non-skip block in each 64x64 CDEF unit reads
        // that unit's `cdef_bits`-wide strength literal; skip blocks and
        // already-read units read nothing. This is a real entropy read on any
        // CDEF-enabled frame (`cdef_bits > 0`) — omitting it desyncs the
        // arithmetic decoder for every following symbol. (The earlier envelope
        // targets — 16x18, 64x66 — had `cdef_bits == 0` or a skip mi(0,0), so the
        // gap was inert; `av1-1-b8-01-size-16x66` frame 1 has `cdef_bits == 1`
        // with a non-skip NEWMV mi(0,0), so the missing read shifted its `read_mv`
        // and mv desynced to (0,-15) vs C's (-1,-7).)
        let coded_lossless = self.st.coded_lossless;
        let allow_intrabc = self.st.allow_intrabc;
        let mib_size = self.st.mib_size;
        let sb_size = self.st.sb_size;
        let cdef_bits = self.st.cdef_bits;
        bx.cdef_strength = ep::read_cdef(
            dec,
            coded_lossless,
            allow_intrabc,
            mi_row,
            mi_col,
            mib_size,
            sb_size,
            bx.skip,
            &mut self.st.cdef_transmitted,
            cdef_bits,
        );
        // read_delta_q_params: delta_q_present off -> no read.

        // read_is_inter_block.
        let ii_ctx = ep::get_intra_inter_context(
            bx.up_available,
            bx.above_dv.is_some_and(dv_inter),
            bx.left_available,
            bx.left_dv.is_some_and(dv_inter),
        ) as usize;
        let is_inter = ep::read_is_inter(dec, &mut icdfs.intra_inter[ii_ctx], false, false);

        // --- is_inter == 0: an INTRA-coded block inside this inter frame
        // (`read_inter_frame_mode_info`'s else-arm, decodemv.c:1550). Everything
        // from here on is the EXISTING byte-exact KEY-frame intra decode: the
        // mode-info tail (shared with `read_mb_modes_kf_fc`, differing only in
        // the Y-mode CDF) then `decode_intra_block_body` (shared verbatim — C
        // installs one frame-type-independent intra recon visitor pair,
        // decodeframe.c:2756/:2761). The block reads `ref_frame = [INTRA_FRAME,
        // NONE_FRAME]`, so `is_inter_block` is false for it and every downstream
        // gate (tx-size's `inter_block_tx`, `av1_read_tx_type`'s `inter_block`,
        // `decode_token_recon_block`'s branch) takes its intra arm unchanged.
        if is_inter == 0 {
            self.decode_intra_in_inter_block(dec, cdfs, icdfs, &bx);
            return;
        }

        // --- read_inter_block_mode_info + parse_decode_block tail ---
        let Stage::Continue(mi) = self.read_inter_mode_info(dec, &mut icdfs, inter, &bx) else {
            return;
        };
        // --- motion compensation (predict phase; NO entropy reads) ---
        if let Stage::Stop = self.predict_inter_block(&mi, inter, &bx) {
            return;
        }
        // --- reconstruction: residual coefficients + CDF/grid writes ---
        self.recon_inter_block(dec, cdfs, icdfs, mi, inter, &bx);
    }
}

/// `clamp_mv_to_umv_border_sb` (reconinter.h:343), the general **per-plane** form.
/// C applies this once per plane in `dec_calc_subpel_params` (decodeframe.c:611/625)
/// with the plane's `subsampling_x`/`subsampling_y` and the plane prediction dims
/// `bw`/`bh` (`= inter_pred_params->block_{width,height}` = `xd->plane[p].{width,
/// height}`, plane pixels): it scales the MV **and** the (luma-domain 1/8-pel)
/// block edges by `1 << (1 - ss)` before clamping, using a `spel` margin computed
/// from the plane dims. The returned `MV mv_q4` is in the plane's q4 (1/16-plane-
/// pel) grid.
///
/// This port keeps the MV in 1/8-pel-luma units and lets
/// [`aom_dsp::inter::build_inter_predictor`] re-apply the `1 << (1 - ss)` plane scaling,
/// so this returns `mv_q4 / (1 << (1 - ss))` (division is exact: for `ss = 0` the
/// q4 value is even — the edge/spel terms are multiples of 16 — and for `ss = 1`
/// the scale is 1). `mb_to_*` are the CURRENT block's luma edges (mi_row/mi_col/
/// bsize), shared by every plane; `bw`/`bh` are this plane's prediction dims.
#[allow(clippy::too_many_arguments)]
fn clamp_mv_to_umv_border_plane(
    mv_row: i32,
    mv_col: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    bw: i32,
    bh: i32,
    ss_x: usize,
    ss_y: usize,
    cfg: &KfTileConfig,
) -> (i32, i32) {
    const AOM_INTERP_EXTEND: i32 = 4;
    const SUBPEL_BITS: i32 = 4;
    const SUBPEL_SHIFTS: i32 = 16;
    let sx = 1i32 << (1 - ss_x as i32);
    let sy = 1i32 << (1 - ss_y as i32);
    let mb_to_left = -(mi_col * 4 * 8);
    let mb_to_right = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 4 * 8;
    let mb_to_top = -(mi_row * 4 * 8);
    let mb_to_bottom = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 4 * 8;
    let spel_left = (AOM_INTERP_EXTEND + bw) << SUBPEL_BITS;
    let spel_right = spel_left - SUBPEL_SHIFTS;
    let spel_top = (AOM_INTERP_EXTEND + bh) << SUBPEL_BITS;
    let spel_bottom = spel_top - SUBPEL_SHIFTS;
    let col_min = mb_to_left * sx - spel_left;
    let col_max = mb_to_right * sx + spel_right;
    let row_min = mb_to_top * sy - spel_top;
    let row_max = mb_to_bottom * sy + spel_bottom;
    let cq4 = (mv_col * sx).clamp(col_min, col_max);
    let rq4 = (mv_row * sy).clamp(row_min, row_max);
    (rq4 / sy, cq4 / sx)
}

/// `clamp_mv_to_umv_border_sb` (reconinter.h:343) in the LUMA (`ss = 0`) domain —
/// a thin wrapper over [`clamp_mv_to_umv_border_plane`] with the luma plane dims
/// (`block_size_{wide,high}[bsize]`). Returns a 1/8-pel MV. The q4 limits are
/// always even, so the luma clamp is exact in 1/8-pel and
/// [`aom_dsp::inter::build_inter_predictor`] rescales it per plane.
fn clamp_mv_to_umv_border(
    mv_row: i32,
    mv_col: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    cfg: &KfTileConfig,
) -> (i32, i32) {
    let bw = MI_SIZE_WIDE[bsize] * 4;
    let bh = MI_SIZE_HIGH[bsize] * 4;
    clamp_mv_to_umv_border_plane(mv_row, mv_col, mi_row, mi_col, bsize, bw, bh, 0, 0, cfg)
}

// ===================================================================
// OBMC (chunk 4) — motion_mode ceiling + overlapped-block MC helpers.
// ===================================================================

/// `is_motion_variation_allowed_bsize` (blockd.h:1460): OBMC/warp are allowed
/// only for blocks whose smaller side is `>= 8` px.
fn is_motion_variation_allowed_bsize(bsize: usize) -> bool {
    BLOCK_SIZE_WIDE[bsize].min(BLOCK_SIZE_HIGH[bsize]) >= 8
}

/// `is_interintra_allowed` (blockd.h:1430): the block can carry inter-intra
/// (bsize BLOCK_8X8..=BLOCK_32X32, a single-ref inter mode, single reference).
/// The flag is coded (and hence must be read for entropy sync) whenever this
/// holds AND `enable_interintra_compound`. BLOCK_8X8=3, BLOCK_32X32=9;
/// SINGLE_INTER_MODE_START=NEARESTMV=13, SINGLE_INTER_MODE_END=NEAREST_NEARESTMV=17.
fn is_interintra_allowed(bsize: usize, mode: i32, ref0: i32, ref1: i32) -> bool {
    (3..=9).contains(&bsize) && (13..17).contains(&mode) && ref0 > 0 && ref1 <= 0
}

/// `size_group_lookup[bsize]` (common_data.h): selects `interintra_cdf[group]`.
const SIZE_GROUP_LOOKUP: [usize; 22] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2,
];

/// `max_neighbor_obmc[log2(mi_width)]` (blockd.h:1471): the OBMC neighbour cap.
const MAX_NEIGHBOR_OBMC: [i32; 6] = [0, 1, 2, 3, 4, 4];

/// `mi_size_wide_log2[bsize]`: log2 of the block's mi width (a power of two).
fn mi_size_wide_log2(bsize: usize) -> usize {
    MI_SIZE_WIDE[bsize].trailing_zeros() as usize
}

/// `mi_size_high_log2[bsize]`: log2 of the block's mi height (a power of two).
fn mi_size_high_log2(bsize: usize) -> usize {
    MI_SIZE_HIGH[bsize].trailing_zeros() as usize
}

/// `av1_skip_u4x4_pred_in_obmc` (reconinter.c:818): for a plane whose subsampled
/// block size is `<= 8x8` in one dimension, OBMC blends only from the LEFT (skip
/// ABOVE). `dir` is 0 = above, 1 = left. (`DISABLE_CHROMA_U8X8_OBMC == 0`, so the
/// one-sided form.)
fn skip_u4x4_pred_in_obmc(bsize: usize, ss_x: usize, ss_y: usize, dir: i32) -> bool {
    // BLOCK_4X4 = 0, BLOCK_4X8 = 1, BLOCK_8X4 = 2 (enums.h).
    let bsize_plane = get_plane_block_size(bsize, ss_x, ss_y);
    matches!(bsize_plane, 0..=2) && dir == 0
}

/// `clamp_mv_to_umv_border_sb` (reconinter.h:343) with the caller's explicit
/// `mb_to_*_edge` (1/8-pel LUMA) and the PREDICTION block dims `bw`/`bh` (px, in
/// the target PLANE's resolution) — the OBMC-neighbour form (the edges are the
/// OBMC-adjusted values, the dims are the narrow overlap strip, not the coding
/// block). `ss_x`/`ss_y` select the plane: the luma edges are scaled by
/// `1 << (1 - ss)` and the `spel_*` bounds use the subsampled `bw`/`bh`, exactly
/// as C's `clamp_mv_to_umv_border_sb`. Returns the (possibly clamped) MV in
/// 1/8-pel LUMA units ([`aom_dsp::inter::build_inter_predictor`] rescales per plane).
#[allow(clippy::too_many_arguments)]
fn clamp_mv_umv_border_px(
    mv_row: i32,
    mv_col: i32,
    bw: i32,
    bh: i32,
    mb_to_left: i32,
    mb_to_right: i32,
    mb_to_top: i32,
    mb_to_bottom: i32,
    ss_x: usize,
    ss_y: usize,
) -> (i32, i32) {
    const AOM_INTERP_EXTEND: i32 = 4;
    const SUBPEL_BITS: i32 = 4;
    const SUBPEL_SHIFTS: i32 = 16;
    let sx = 1i32 << (1 - ss_x as i32);
    let sy = 1i32 << (1 - ss_y as i32);
    let spel_left = (AOM_INTERP_EXTEND + bw) << SUBPEL_BITS;
    let spel_right = spel_left - SUBPEL_SHIFTS;
    let spel_top = (AOM_INTERP_EXTEND + bh) << SUBPEL_BITS;
    let spel_bottom = spel_top - SUBPEL_SHIFTS;
    let col_min = mb_to_left * sx - spel_left;
    let col_max = mb_to_right * sx + spel_right;
    let row_min = mb_to_top * sy - spel_top;
    let row_max = mb_to_bottom * sy + spel_bottom;
    let cq4 = (mv_col * sx).clamp(col_min, col_max);
    let rq4 = (mv_row * sy).clamp(row_min, row_max);
    (rq4 / sy, cq4 / sx)
}
