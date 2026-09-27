//! The intra block path — `decode_intra_block_body` (`parse_decode_block`'s
//! intra arm + `decode_token_recon_block`'s per-txb read→predict→recon loops,
//! decodeframe.c). Used by `decode_block` for key frames and by
//! `decode_intra_in_inter_block` ([`super::inter`]) for intra blocks inside an
//! inter frame — C shares the same visitor pair for both (decodeframe.c:2756,
//! :2761).
use super::*;

/// The `av1_visit_palette` token decode's product — the per-pixel
/// colour-index maps for the luma and (shared) chroma palettes. Empty `Vec`s
/// when the block isn't palette-coded; the recon loops only index them under
/// `palette_size[..] > 0` gates, so the maps and the mode are never
/// inconsistent (`PaletteMaps` can't express "palette flag set, no map").
struct PaletteMaps {
    y: Vec<u8>,
    uv: Vec<u8>,
    /// The chroma map's row stride in pixels — 0 when there is no chroma
    /// palette (the luma map's stride is the block width, re-derived where
    /// used).
    uv_wpx: usize,
}

/// `read_tx_size`'s product for an intra block: the block transform size plus,
/// for an intrabc var-tx block, the per-4x4 leaf grid (`mbmi->inter_tx_size[]`)
/// and the flag recording that the quadtree was actually READ — the gate that
/// selects the leaf walk over the uniform raster (KB-29: equal leaf sizes do
/// NOT imply raster order).
struct IntraTxLayout {
    tx_size: usize,
    vartx_leaf_grid: Vec<u8>,
    vartx_quadtree_read: bool,
}

/// One 64x64 chunk of `decode_token_recon_block`'s unit-grid walk: the chunk
/// origin and step in 4x4 units plus the block's (frame-edge-clamped) unit
/// extent. Bundles the walk's geometry so the per-chunk arms take `(bx, g)`
/// rather than eight positional ints.
#[derive(Clone, Copy)]
struct ChunkGeom {
    row: usize,
    col: usize,
    /// `mu_w`/`mu_h` — the chunk step, `min(block extent, 64x64 units)`.
    step_w: usize,
    step_h: usize,
    /// `max_blocks_wide`/`max_blocks_high` — the block's extent in units.
    blocks_wide: usize,
    blocks_high: usize,
}

/// The uniform luma raster's per-block scratch — the coeff plane plus the
/// u16/u8 prediction scratch, allocated once per block at the block's tx size
/// and re-zeroed per txb. `scratch8` is populated only on the lowbd path
/// (`Vec::new()` at high bit depth — the lowbd arm is inexpressible rather
/// than gated at every use).
struct LumaScratch {
    tcoeff: Vec<i32>,
    scratch: Vec<u16>,
    scratch8: Vec<u8>,
}

impl<'c> TileKf<'c> {
    /// `av1_visit_palette(..., av1_decode_palette_tokens)` (decodeframe.c):
    /// the colour-index MAP tokens — a SEPARATE step from the mode-info
    /// palette flags/size/colours (it needs `av1_get_block_dimensions`' block
    /// geometry, which the mode-info driver doesn't carry). Y decodes iff
    /// `palette_size[0] > 0`; chroma (ONE shared map, indexed by BOTH U and V
    /// during reconstruction) iff `palette_size[1] > 0` — gated on
    /// `plane == 0 || is_chroma_ref` like `av1_visit_palette` itself (a
    /// non-chroma-reference block can never reach here with
    /// `palette_size[1] > 0`: `read_mb_modes_kf_fc`'s uv_dc_pred gate already
    /// requires `is_chroma_ref`).
    fn read_intra_color_maps(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        bx: &BlockCtx,
        info: &MbModeInfoKf,
    ) -> PaletteMaps {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            ..
        } = *bx;
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        // av1_visit_palette(..., av1_decode_palette_tokens) (decodeframe.c): the
        // colour-index MAP tokens — a SEPARATE step from the mode-info flags/size/
        // colours just read above (needs av1_get_block_dimensions' block-geometry
        // inputs, which the mode-info driver doesn't carry). Y decodes iff
        // palette_size[0]>0; chroma (ONE shared map, indexed by BOTH U and V during
        // reconstruction) iff palette_size[1]>0 — gated on `plane==0 || is_chroma_ref`
        // like av1_visit_palette itself (a non-chroma-reference block can never reach
        // here with palette_size[1]>0, since read_mb_modes_kf_fc's uv_dc_pred gate
        // already requires is_chroma_ref).
        let pal_mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let pal_mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let mut color_map_y: Vec<u8> = Vec::new();
        let mut color_map_uv: Vec<u8> = Vec::new();
        let mut uv_map_wpx = 0usize;
        if info.palette_size[0] > 0 {
            let (wpx, hpx, rows, cols) = get_block_dimensions(
                bsize,
                0,
                ss_x,
                ss_y,
                pal_mb_to_right_edge,
                pal_mb_to_bottom_edge,
            );
            // palette_{y,uv}_color_index_cdf[n - PALETTE_MIN_SIZE] (PALETTE_MIN_SIZE=2).
            let n = info.palette_size[0];
            color_map_y = decode_color_map_tokens(
                dec,
                n,
                wpx,
                hpx,
                rows,
                cols,
                &mut cdfs.palette_y_color_index[(n - 2) as usize],
            );
        }
        if info.palette_size[1] > 0 {
            let (wpx, hpx, rows, cols) = get_block_dimensions(
                bsize,
                1,
                ss_x,
                ss_y,
                pal_mb_to_right_edge,
                pal_mb_to_bottom_edge,
            );
            let n = info.palette_size[1];
            color_map_uv = decode_color_map_tokens(
                dec,
                n,
                wpx,
                hpx,
                rows,
                cols,
                &mut cdfs.palette_uv_color_index[(n - 2) as usize],
            );
            uv_map_wpx = wpx;
        }
        PaletteMaps {
            y: color_map_y,
            uv: color_map_uv,
            uv_wpx: uv_map_wpx,
        }
    }
    /// `read_tx_size` for an intra block (decodeframe.c) + the txfm-context
    /// stamp, in C's statement order (after mode info / palette tokens, before
    /// the skip reset): the lossless TX_4X4 preemption, the intrabc var-tx
    /// quadtree (`read_tx_size_vartx` — which stamps the txfm contexts itself),
    /// the intra `tx_size_cdf` depth read under TX_MODE_SELECT, or the tx_mode
    /// fallback for non-signalling blocks.
    fn read_intra_tx_layout(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        bx: &BlockCtx,
        info: &MbModeInfoKf,
    ) -> IntraTxLayout {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            up_available,
            left_available,
            ..
        } = *bx;
        let cfg = self.cfg;
        // --- parse_decode_block: the block's transform size (read_tx_size) +
        // txfm-context stamp, in the C statement order (after the mode info /
        // palette tokens, before the skip entropy reset) ---
        let bw = MI_SIZE_WIDE[bsize] as usize;
        let bh = MI_SIZE_HIGH[bsize] as usize;
        // read_tx_size (decodeframe.c), intra: the xd->lossless[segment_id]
        // TX_4X4 preemption is off in this scope (coded_lossless rejected;
        // lossless SEGMENTS asserted away in TileKf::new); a signalling block
        // (bsize > BLOCK_4X4) under TX_MODE_SELECT codes its tx-size depth —
        // intra blocks code it even when skip_txfm is set (`!is_inter ||
        // allow_select_inter` is true for intra) — else the tx_mode fallback.
        let a_off = mi_col as usize;
        let l_off = (mi_row & 31) as usize;
        // Var-tx leaf grid (C's `mbmi->inter_tx_size[]`), block-relative per-4x4,
        // filled by the size-read phase for an intrabc var-tx block. When the
        // partition is non-uniform the reconstruction phase walks it
        // (`collect_vartx_leaves`) instead of tiling with a single tx size.
        let bw4 = MI_SIZE_WIDE[bsize] as usize;
        let bh4 = MI_SIZE_HIGH[bsize] as usize;
        let mut vartx_leaf_grid: Vec<u8> = Vec::new();
        // Set when the intrabc var-tx quadtree was actually READ for this
        // block — the gate that selects the leaf walk (see `do_uniform` below).
        let mut vartx_quadtree_read = false;
        let tx_size = if self.st.coded_lossless {
            // read_tx_size (decodeframe.c): xd->lossless[segment_id] preempts to
            // TX_4X4 before any tx-size symbol / block_signals_txsize test. The
            // else-arm of read_block_tx_size then stamps set_txfm_ctxs(TX_4X4, ...,
            // skip && is_inter_block); is_inter_block on a KEY frame is
            // use_intrabc. The var-tx quadtree is gated on !lossless in C, so it
            // never runs here.
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                TX_4X4_IDX,
                bw,
                bh,
                info.skip != 0 && info.use_intrabc != 0,
            );
            TX_4X4_IDX
        } else if info.use_intrabc != 0 {
            // Intrabc is `is_inter_block`, so `inter_block_tx` is set and the tx
            // size follows the INTER path (decodeframe.c:1179-1198):
            //  - TX_MODE_SELECT && block_signals_txsize && !skip  → the var-tx
            //    quadtree (`read_tx_size_vartx`, `txfm_partition_cdf` — NOT the
            //    intra `tx_size_cdf`). It stamps the txfm-context arrays itself
            //    via `txfm_partition_update`, so no `set_txfm_ctxs` here.
            //  - otherwise → `read_tx_size(is_inter=true, allow_select=!skip)`:
            //    with `is_inter` the `read_selected` arm needs
            //    `allow_select && select && signalling`, which is exactly the
            //    var-tx case above, so this always resolves to the fallback
            //    (`tx_size_from_tx_mode` when signalling, else max rect) — no
            //    tx-size symbol — then `set_txfm_ctxs(skip && is_inter = skip)`.
            if cfg.tx_mode == TxMode::Select && bsize > 0 && info.skip == 0 {
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
                // `read_tx_size_vartx` reports whether the leaf SIZES differ;
                // that is not what selects the walk (see `do_uniform` below),
                // so it is deliberately unused here.
                let _ = non_uniform;
                vartx_quadtree_read = true;
                vartx_tx
            } else {
                let ts = if bsize > 0 {
                    tx_size_from_tx_mode(bsize, cfg.tx_mode)
                } else {
                    MAX_TXSIZE_RECT_LOOKUP[bsize]
                };
                set_txfm_ctxs(
                    &mut self.above_t[a_off..],
                    &mut self.left_t[l_off..],
                    ts,
                    bw,
                    bh,
                    info.skip != 0,
                );
                ts
            }
        } else if bsize > 0 {
            // Intra, block_signals_txsize: read_tx_size(is_inter=false,
            // allow_select=!skip). `!is_inter` makes `(!is_inter || …)` true, so a
            // signalling block under TX_MODE_SELECT codes its tx-size depth
            // (intra codes it even when skip_txfm is set) — else the tx_mode
            // fallback. set_txfm_ctxs' skip arg is `skip && is_inter` = 0.
            let tx_size = if cfg.tx_mode == TxMode::Select {
                let cat = bsize_to_tx_size_cat(bsize) as usize;
                // get_tx_size_context (blockd.h) overrides the txfm-context-array
                // term with the NEIGHBOUR'S BLOCK size whenever that neighbour is
                // `is_inter_block` — which is `use_intrabc || ref_frame[0] >
                // INTRA_FRAME` (blockd.h:372). On a KEY frame only intrabc
                // qualifies (every stamp carries ref_frame0 == INTRA_FRAME), so
                // this is behaviour-identical there; inside an INTER frame an
                // intra block's neighbours are usually genuine inter blocks, and
                // missing them picks the wrong `tx_size_cdf` row — the same
                // symbol on different probabilities, which drifts the arithmetic
                // decoder and desyncs a few reads later.
                let is_inter_nbr = |d: &DvNbr| d.use_intrabc || d.ref_frame0 > 0;
                let above_inter_bsize = up_available
                    .then(|| {
                        DvNbr::from_packed(
                            self.mi_dv[((mi_row - 1) * cfg.mi_cols + mi_col) as usize],
                        )
                    })
                    .filter(is_inter_nbr)
                    .map(|d| d.bsize);
                let left_inter_bsize = left_available
                    .then(|| {
                        DvNbr::from_packed(self.mi_dv[(mi_row * cfg.mi_cols + mi_col - 1) as usize])
                    })
                    .filter(is_inter_nbr)
                    .map(|d| d.bsize);
                let ctx = get_tx_size_context(
                    bsize,
                    self.above_t[a_off],
                    self.left_t[l_off],
                    up_available,
                    left_available,
                    above_inter_bsize,
                    left_inter_bsize,
                );
                let depth = read_selected_tx_size(
                    dec,
                    &mut cdfs.tx_size[cat][ctx],
                    bsize,
                    bsize_to_max_depth(bsize),
                );
                depth_to_tx_size(depth, bsize)
            } else {
                tx_size_from_tx_mode(bsize, cfg.tx_mode)
            };
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                tx_size,
                bw,
                bh,
                false,
            );
            tx_size
        } else {
            // Intra, non-signalling (BLOCK_4X4): max rect (TX_4X4), no symbol.
            let tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                tx_size,
                bw,
                bh,
                false,
            );
            tx_size
        };
        IntraTxLayout {
            tx_size,
            vartx_leaf_grid,
            vartx_quadtree_read,
        }
    }

    /// `av1_reset_entropy_context`'s intra arm (parse_decode_block tail):
    /// skip blocks zero their entropy contexts — plane 0 always, over the
    /// block's mi footprint; the chroma planes when this block is the chroma
    /// reference, over the chroma plane-bsize footprint from the adjusted
    /// context bases (the shared-chroma group's `adj_row`/`adj_col` origin).
    fn reset_intra_skip_ctx(&mut self, bx: &BlockCtx, info: &MbModeInfoKf) {
        if info.skip == 0 {
            return;
        }
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let bw = MI_SIZE_WIDE[bx.bsize] as usize;
        let bh = MI_SIZE_HIGH[bx.bsize] as usize;
        let a0 = bx.mi_col as usize;
        self.above_e[0][a0..a0 + bw].fill(0);
        let l0 = (bx.mi_row & 31) as usize;
        self.left_e[0][l0..l0 + bh].fill(0);
        if !cfg.monochrome && bx.chroma_ref {
            let plane_bsize = get_plane_block_size(bx.bsize, ss_x, ss_y);
            let (uw, uh) = (
                MI_SIZE_WIDE[plane_bsize] as usize,
                MI_SIZE_HIGH[plane_bsize] as usize,
            );
            let uv_a_base = (bx.adj_col >> ss_x) as usize;
            let uv_l_base = ((bx.adj_row & 31) >> ss_y) as usize;
            for plane in 1..=2 {
                self.above_e[plane][uv_a_base..uv_a_base + uw].fill(0);
                self.left_e[plane][uv_l_base..uv_l_base + uh].fill(0);
            }
        }
    }

    /// `decode_reconstruct_tx`'s non-uniform arm (decodeframe.c): an intrabc
    /// var-tx block walks the leaf grid in DFS order — each leaf reads its own
    /// coeffs + inter ext-tx type, then copy-reconstructs at the leaf size.
    ///
    /// The gate is "was the quadtree read", NOT "are the leaf sizes distinct"
    /// (KB-29): a BLOCK_16X8 split all the way to TX_4X4 has eight same-size
    /// leaves whose DFS order is not raster, so the per-txb `txb_skip_ctx`
    /// sequence differs from byte one of the third txb.
    fn recon_intra_vartx_luma(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        bx: &BlockCtx,
        info: &MbModeInfoKf,
        tx: &IntraTxLayout,
        signal_gate: bool,
        txbs: &mut Vec<(usize, usize)>,
    ) {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            chroma_ref,
            ..
        } = *bx;
        let cfg = self.cfg;
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let max_blocks_wide = max_block_units(BLOCK_SIZE_WIDE[bsize], mb_to_right_edge);
        let max_blocks_high = max_block_units(BLOCK_SIZE_HIGH[bsize], mb_to_bottom_edge);
        let bw4 = MI_SIZE_WIDE[bsize] as usize;
        let vartx_leaf_grid = &tx.vartx_leaf_grid;
        let max_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
        let bw_mt = TX_SIZE_WIDE_UNIT[max_tx];
        let bh_mt = TX_SIZE_HIGH_UNIT[max_tx];
        let mut leaves: Vec<(usize, usize, usize)> = Vec::new();
        let mut r = 0;
        while r < max_blocks_high {
            let mut c = 0;
            while c < max_blocks_wide {
                collect_vartx_leaves(
                    vartx_leaf_grid,
                    bw4,
                    r,
                    c,
                    max_tx,
                    max_blocks_high,
                    max_blocks_wide,
                    &mut leaves,
                );
                c += bw_mt;
            }
            r += bh_mt;
        }
        let mut nu_tcoeff = vec![0i32; txb_wide(max_tx) * txb_high(max_tx)];
        let mut nu_scratch = vec![0u16; TX_SIZE_WIDE[max_tx] * TX_SIZE_HIGH[max_tx]];
        let mut nu_scratch8 = if self.recon.is_lowbd() {
            vec![0u8; TX_SIZE_WIDE[max_tx] * TX_SIZE_HIGH[max_tx]]
        } else {
            Vec::new()
        };
        for &(blk_row, blk_col, cur_tx) in &leaves {
            let (ltxw, ltxh) = (TX_SIZE_WIDE_UNIT[cur_tx], TX_SIZE_HIGH_UNIT[cur_tx]);
            let (ltxwpx, ltxhpx) = (TX_SIZE_WIDE[cur_tx], TX_SIZE_HIGH[cur_tx]);
            let larea = txb_wide(cur_tx) * txb_high(cur_tx);
            let (eob, tx_type) = if info.skip == 0 {
                let a0 = mi_col as usize + blk_col;
                let l0 = (mi_row & 31) as usize + blk_row;
                let (tsc, dsc) = get_txb_ctx(
                    bsize,
                    cur_tx,
                    0,
                    &self.above_e[0][a0..],
                    &self.left_e[0][l0..],
                );
                let ext = inter_ext_tx_cdf(&mut cdfs.inter_ext_tx, cur_tx, cfg.reduced_tx_set);
                let (eob, tt) = read_coeffs_txb_full(
                    dec,
                    &mut cdfs.coeff,
                    ext,
                    &mut nu_tcoeff[..larea],
                    cur_tx,
                    0,
                    tsc as usize,
                    dsc as usize,
                    true,
                    true,
                    cfg.reduced_tx_set,
                    signal_gate,
                    0,
                );
                let cul = txb_entropy_context(&nu_tcoeff[..larea], cur_tx, tt, eob) as i8;
                self.set_entropy_ctx(
                    0,
                    cul,
                    mi_col as usize,
                    (mi_row & 31) as usize,
                    blk_row,
                    blk_col,
                    ltxw,
                    ltxh,
                    max_blocks_wide,
                    max_blocks_high,
                    mb_to_right_edge,
                    mb_to_bottom_edge,
                );
                (eob, tt)
            } else {
                (0, 0)
            };
            // Luma tx_type_map stamp (top-left + 64-level), per leaf, so the
            // chroma co-location reads the right leaf's tx-type.
            if !self.luma_tt.is_empty() {
                let cols = cfg.mi_cols as usize;
                let r0 = mi_row as usize + blk_row;
                let c0 = mi_col as usize + blk_col;
                self.luma_tt[r0 * cols + c0] = tx_type as u8;
                if ltxw == 16 || ltxh == 16 {
                    let rmax = ltxh.min(max_blocks_high - blk_row);
                    let cmax = ltxw.min(max_blocks_wide - blk_col);
                    let mut idy = 0;
                    while idy < rmax {
                        let mut idx = 0;
                        while idx < cmax {
                            self.luma_tt[(r0 + idy) * cols + c0 + idx] = tx_type as u8;
                            idx += 4;
                        }
                        idy += 4;
                    }
                }
            }
            let off = ((mi_row * 4) as usize + blk_row * 4) * self.stride
                + (mi_col * 4) as usize
                + blk_col * 4;
            let src = (off as i32 + (info.dv_row >> 3) * self.stride as i32 + (info.dv_col >> 3))
                as usize;
            match &mut self.recon {
                ReconPlane::HighBd(p) => {
                    for r in 0..ltxhpx {
                        let s = src + r * self.stride;
                        nu_scratch[r * ltxwpx..(r + 1) * ltxwpx].copy_from_slice(&p[s..s + ltxwpx]);
                    }
                    for r in 0..ltxhpx {
                        let d = off + r * self.stride;
                        p[d..d + ltxwpx].copy_from_slice(&nu_scratch[r * ltxwpx..(r + 1) * ltxwpx]);
                    }
                }
                ReconPlane::LowBd(p) => {
                    for r in 0..ltxhpx {
                        let s = src + r * self.stride;
                        nu_scratch8[r * ltxwpx..(r + 1) * ltxwpx]
                            .copy_from_slice(&p[s..s + ltxwpx]);
                    }
                    for r in 0..ltxhpx {
                        let d = off + r * self.stride;
                        p[d..d + ltxwpx]
                            .copy_from_slice(&nu_scratch8[r * ltxwpx..(r + 1) * ltxwpx]);
                    }
                }
            }
            if info.skip == 0 && eob > 0 {
                let iqm = qm::iqmatrix(self.block_qm_level[0], 0, cur_tx, tx_type);
                let dequant = self.dequants[0];
                match &mut self.recon {
                    ReconPlane::HighBd(p) => reconstruct_txb_into(
                        &mut p[off..],
                        self.stride,
                        cur_tx,
                        tx_type,
                        &nu_tcoeff[..larea],
                        dequant,
                        iqm,
                        cfg.bd,
                        &mut self.recon_scratch,
                    ),
                    ReconPlane::LowBd(p) => reconstruct_txb_u8_into(
                        &mut p[off..],
                        self.stride,
                        cur_tx,
                        tx_type,
                        &nu_tcoeff[..larea],
                        dequant,
                        iqm,
                        &mut self.recon_scratch,
                    ),
                }
            }
            // (4) CfL luma store — the same
            // `predict_and_reconstruct_intra_block` tail the uniform loop
            // runs (`store_cfl_required`): a NON-chroma-reference block
            // always stores, because a later member of its shared chroma
            // group may pick `UV_CFL_PRED` over a footprint that contains
            // this block. `is_inter_block(mbmi)` is TRUE for intrabc
            // (blockd.h:372), which is exactly why C's `encode_superblock`
            // has the mirrored `cfl_store_block` call on its inter path —
            // the encoder-side twin of this was KB-15 root #4. Omitting it
            // here left the CfL predictor reading a STALE luma buffer, so
            // the sibling's chroma reconstruction diverged from the C
            // decoder's while luma matched exactly (KB-29's leaf-arm
            // residual: 253 of 9604 U samples, first at chroma (80,44)).
            if !cfg.monochrome && (!chroma_ref || info.uv_mode == UV_CFL_PRED) {
                let block_off = (mi_row * 4) as usize * self.stride + (mi_col * 4) as usize;
                cfl_store_tx_any(
                    &mut self.cfl,
                    &self.recon,
                    block_off,
                    self.stride,
                    blk_row as i32,
                    blk_col as i32,
                    cur_tx,
                    bsize,
                    mi_row,
                    mi_col,
                    &mut self.wide_rect,
                );
            }
            txbs.push((eob, tx_type));
        }
    }

    /// `decode_reconstruct_tx`'s uniform arm for one 64x64 chunk
    /// (decodeframe.c:933-962): the luma txb raster — each unit's
    /// `read_coeffs_tx_intra_block` (coeffs + `av1_read_tx_type`) then
    /// `predict_and_reconstruct_intra_block` (intra predict + dequant +
    /// inverse transform + optional CfL store), in raster order. Runs only
    /// when `do_uniform` (the non-uniform intrabc arm consumed plane 0's
    /// leaves already).
    fn recon_intra_luma_chunk(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        bx: &BlockCtx,
        info: &MbModeInfoKf,
        palette: &PaletteMaps,
        tx: &IntraTxLayout,
        g: &ChunkGeom,
        buf: &mut LumaScratch,
        filt_type: i32,
        signal_gate: bool,
        txbs: &mut Vec<(usize, usize)>,
    ) {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            partition,
            chroma_ref,
            up_available,
            left_available,
            ..
        } = *bx;
        let ChunkGeom {
            row: chunk_row,
            col: chunk_col,
            step_w: mu_w,
            step_h: mu_h,
            blocks_wide: max_blocks_wide,
            blocks_high: max_blocks_high,
        } = *g;
        let cfg = self.cfg;
        let tx_size = tx.tx_size;
        let (txw, txh) = (TX_SIZE_WIDE_UNIT[tx_size], TX_SIZE_HIGH_UNIT[tx_size]);
        let (txwpx, txhpx) = (TX_SIZE_WIDE[tx_size], TX_SIZE_HIGH[tx_size]);
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let color_map_y = &palette.y;
        let LumaScratch {
            tcoeff,
            scratch,
            scratch8,
        } = buf;
        let luma_row_end = (chunk_row + mu_h).min(max_blocks_high);
        let luma_col_end = (chunk_col + mu_w).min(max_blocks_wide);
        let mut blk_row = chunk_row;
        while blk_row < luma_row_end {
            let mut blk_col = chunk_col;
            while blk_col < luma_col_end {
                // (1) coefficients — read_coeffs_tx_intra_block (skipped blocks
                // code nothing; their contexts stay at the reset zeros).
                let (eob, tx_type) = if info.skip == 0 {
                    let a0 = mi_col as usize + blk_col;
                    let l0 = (mi_row & 31) as usize + blk_row;
                    let (tsc, dsc) = get_txb_ctx(
                        bsize,
                        tx_size,
                        0,
                        &self.above_e[0][a0..],
                        &self.left_e[0][l0..],
                    );
                    // Intrabc is is_inter_block, so av1_read_tx_type selects the
                    // tx-type CDF from inter_ext_tx_cdf (and maps the symbol with
                    // the inter set type); a normal intra block uses the intra
                    // ext-tx sets keyed on (square tx size, intra direction).
                    let ext = if info.use_intrabc != 0 {
                        inter_ext_tx_cdf(&mut cdfs.inter_ext_tx, tx_size, cfg.reduced_tx_set)
                    } else {
                        intra_ext_tx_cdf(
                            &mut cdfs.ext_tx_1ddct,
                            &mut cdfs.ext_tx_dtt4,
                            tx_size,
                            cfg.reduced_tx_set,
                            info.use_filter_intra != 0,
                            info.filter_intra_mode as usize,
                            info.y_mode as usize,
                        )
                    };
                    let (eob, tt) = read_coeffs_txb_full(
                        dec,
                        &mut cdfs.coeff,
                        ext,
                        tcoeff,
                        tx_size,
                        0,
                        tsc as usize,
                        dsc as usize,
                        true,
                        info.use_intrabc != 0,
                        cfg.reduced_tx_set,
                        signal_gate,
                        0,
                    );
                    let cul = txb_entropy_context(tcoeff, tx_size, tt, eob) as i8;
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
                    (eob, tt)
                } else {
                    (0, 0)
                };

                // Record the luma tx-type in `cm->tx_type_map` (mi granularity)
                // exactly as C's `update_txk_array`: the txb's TOP-LEFT mi cell
                // only; a 64-level transform (a 64px side => 16 mi units)
                // additionally stamps every 16x16 (4-mi) unit of its footprint.
                // Cells left unwritten stay DCT_DCT (the zeroed map). Colour
                // intrabc chroma reads the co-located luma tx-type from here;
                // empty (skipped) on monochrome / non-intrabc frames. Skip blocks
                // stamp DCT_DCT (0), matching C.
                if !self.luma_tt.is_empty() {
                    let cols = cfg.mi_cols as usize;
                    let r0 = mi_row as usize + blk_row;
                    let c0 = mi_col as usize + blk_col;
                    self.luma_tt[r0 * cols + c0] = tx_type as u8;
                    if txw == 16 || txh == 16 {
                        let rmax = txh.min(max_blocks_high - blk_row);
                        let cmax = txw.min(max_blocks_wide - blk_col);
                        let mut idy = 0;
                        while idy < rmax {
                            let mut idx = 0;
                            while idx < cmax {
                                self.luma_tt[(r0 + idy) * cols + c0 + idx] = tx_type as u8;
                                idx += 4;
                            }
                            idy += 4;
                        }
                    }
                }

                // (2) intra prediction into the reconstruction plane.
                let (n_top, n_tr, n_left, n_bl) = intra_avail(
                    self.st.sb_size,
                    bsize,
                    mi_row,
                    mi_col,
                    up_available,
                    left_available,
                    self.tile.mi_col_end,
                    self.tile.mi_row_end,
                    partition,
                    tx_size,
                    0,
                    0,
                    blk_row as i32,
                    blk_col as i32,
                    BLOCK_SIZE_WIDE[bsize],
                    BLOCK_SIZE_HIGH[bsize],
                    cfg.mi_cols,
                    cfg.mi_rows,
                    info.y_mode as usize,
                    info.angle_delta_y * ANGLE_STEP,
                    info.use_filter_intra != 0,
                );
                let off = ((mi_row * 4) as usize + blk_row * 4) * self.stride
                    + (mi_col * 4) as usize
                    + blk_col * 4;
                if info.use_intrabc != 0 {
                    // Intra block copy, luma: an integer block copy from the
                    // DV-referenced region of the SAME reconstruction plane. The
                    // DV is read at MV_SUBPEL_NONE (full-pel) and validated by
                    // av1_is_dv_valid to reference only already-decoded pixels, so
                    // the source (off shifted by dv/8) is always reconstructed and
                    // never overlaps this block's pending tx units. Luma needs no
                    // interpolation (av1_dc_128... the intrabc convolve collapses
                    // to a copy at integer positions).
                    let src = (off as i32
                        + (info.dv_row >> 3) * self.stride as i32
                        + (info.dv_col >> 3)) as usize;
                    match &self.recon {
                        ReconPlane::HighBd(p) => {
                            for r in 0..txhpx {
                                let s = src + r * self.stride;
                                scratch[r * txwpx..(r + 1) * txwpx]
                                    .copy_from_slice(&p[s..s + txwpx]);
                            }
                        }
                        ReconPlane::LowBd(p) => {
                            for r in 0..txhpx {
                                let s = src + r * self.stride;
                                scratch8[r * txwpx..(r + 1) * txwpx]
                                    .copy_from_slice(&p[s..s + txwpx]);
                            }
                        }
                    }
                } else if info.palette_size[0] > 0 {
                    // av1_predict_intra_block's palette branch (reconintra.c): pixels
                    // come directly from the colour-index map + palette LUT — no
                    // directional/DC prediction math (the surrounding residual
                    // add below is unaffected: palette replaces PREDICTION only).
                    // The map covers the whole coding block (BLOCK_SIZE_WIDE[bsize]
                    // stride); this tx block reads its (blk_col*4, blk_row*4)
                    // pixel sub-rectangle.
                    let map_w = BLOCK_SIZE_WIDE[bsize] as usize;
                    let (x0, y0) = (blk_col * 4, blk_row * 4);
                    let lowbd = self.recon.is_lowbd();
                    for r in 0..txhpx {
                        for c in 0..txwpx {
                            let idx = color_map_y[(y0 + r) * map_w + x0 + c] as usize;
                            if lowbd {
                                // bd8 palette colours are <= 255 on conformant
                                // input; hostile input truncates like the C
                                // lowbd (uint8_t) store (see `plane`).
                                scratch8[r * txwpx + c] = info.palette_colors[idx] as u8;
                            } else {
                                scratch[r * txwpx + c] = info.palette_colors[idx];
                            }
                        }
                    }
                } else {
                    let n_top_u = usize::try_from(n_top).expect("n_top_px must be non-negative");
                    let n_left_u = usize::try_from(n_left).expect("n_left_px must be non-negative");
                    match &self.recon {
                        ReconPlane::HighBd(p) => predict_intra_high(
                            p,
                            off,
                            self.stride,
                            scratch,
                            txwpx,
                            info.y_mode as usize,
                            info.angle_delta_y * ANGLE_STEP,
                            info.use_filter_intra != 0,
                            info.filter_intra_mode as usize,
                            cfg.disable_edge_filter,
                            filt_type,
                            tx_size,
                            n_top_u,
                            n_tr,
                            n_left_u,
                            n_bl,
                            cfg.bd,
                        ),
                        // bd8 lowbd: the byte-identity-proven u8 intra family
                        // reads the u8 plane directly (Phase B).
                        ReconPlane::LowBd(p) => predict_intra_u8(
                            p,
                            off,
                            self.stride,
                            scratch8,
                            txwpx,
                            info.y_mode as usize,
                            info.angle_delta_y * ANGLE_STEP,
                            info.use_filter_intra != 0,
                            info.filter_intra_mode as usize,
                            cfg.disable_edge_filter,
                            filt_type,
                            tx_size,
                            n_top_u,
                            n_tr,
                            n_left_u,
                            n_bl,
                        ),
                    }
                }
                match &mut self.recon {
                    ReconPlane::HighBd(p) => {
                        for r in 0..txhpx {
                            let d = off + r * self.stride;
                            p[d..d + txwpx].copy_from_slice(&scratch[r * txwpx..(r + 1) * txwpx]);
                        }
                    }
                    ReconPlane::LowBd(p) => {
                        for r in 0..txhpx {
                            let d = off + r * self.stride;
                            p[d..d + txwpx].copy_from_slice(&scratch8[r * txwpx..(r + 1) * txwpx]);
                        }
                    }
                }

                // (3) dequant + inverse transform + add (only when residual
                // exists) — the block-effective luma dequant row.
                if info.skip == 0 && eob > 0 {
                    let dequant = self.dequants[0];
                    if self.st.coded_lossless {
                        // lossless: TX_4X4 + WHT with the qindex-0 dequant.
                        match &mut self.recon {
                            ReconPlane::HighBd(p) => reconstruct_txb_wht(
                                &mut p[off..],
                                self.stride,
                                tcoeff,
                                dequant,
                                eob,
                                cfg.bd,
                            ),
                            ReconPlane::LowBd(p) => reconstruct_txb_wht_u8(
                                &mut p[off..],
                                self.stride,
                                tcoeff,
                                dequant,
                                eob,
                            ),
                        }
                    } else {
                        let iqm = qm::iqmatrix(self.block_qm_level[0], 0, tx_size, tx_type);
                        match &mut self.recon {
                            ReconPlane::HighBd(p) => reconstruct_txb_into(
                                &mut p[off..],
                                self.stride,
                                tx_size,
                                tx_type,
                                tcoeff,
                                dequant,
                                iqm,
                                cfg.bd,
                                &mut self.recon_scratch,
                            ),
                            ReconPlane::LowBd(p) => reconstruct_txb_u8_into(
                                &mut p[off..],
                                self.stride,
                                tx_size,
                                tx_type,
                                tcoeff,
                                dequant,
                                iqm,
                                &mut self.recon_scratch,
                            ),
                        }
                    }
                }
                // (4) CfL luma store (predict_and_reconstruct_intra_block tail,
                // store_cfl_required): non-chroma-reference blocks always store
                // (a later group member may pick CfL); the chroma-reference
                // block stores only when it actually uses CfL. Runs for skip
                // blocks too (their reconstruction is the prediction).
                if !cfg.monochrome && (!chroma_ref || info.uv_mode == UV_CFL_PRED) {
                    let block_off = (mi_row * 4) as usize * self.stride + (mi_col * 4) as usize;
                    cfl_store_tx_any(
                        &mut self.cfl,
                        &self.recon,
                        block_off,
                        self.stride,
                        blk_row as i32,
                        blk_col as i32,
                        tx_size,
                        bsize,
                        mi_row,
                        mi_col,
                        &mut self.wide_rect,
                    );
                }
                txbs.push((eob, tx_type));
                blk_col += txw;
            }
            blk_row += txh;
        }
    }

    /// `decode_token_recon_block`'s chroma arm for one 64x64 chunk
    /// (decodeframe.c:963-995): only the chroma-reference block of a shared
    /// group decodes chroma, covering the merged area from the adjusted
    /// plane origin — and it runs after ALL of the chunk's luma, so the
    /// block's own luma is already in the CfL store.
    ///
    /// `Stage::Stop` on the impossible `plane_bsize` — the second guard on
    /// the corrupt-frame condition `decode_mbmi_block` rejects (crafted
    /// stream; byte-inert on conformant input).
    fn recon_intra_chroma_chunk(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        bx: &BlockCtx,
        info: &MbModeInfoKf,
        palette: &PaletteMaps,
        g: &ChunkGeom,
        txbs_uv: &mut Vec<(usize, usize)>,
    ) -> Stage<()> {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            partition,
            adj_row,
            adj_col,
            ..
        } = *bx;
        let ChunkGeom {
            row: chunk_row,
            col: chunk_col,
            step_w: mu_w,
            step_h: mu_h,
            blocks_wide: max_blocks_wide,
            blocks_high: max_blocks_high,
        } = *g;
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let uv_a_base = (adj_col >> ss_x) as usize;
        let uv_l_base = ((adj_row & 31) >> ss_y) as usize;
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let color_map_uv = &palette.uv;
        let uv_map_wpx = palette.uv_wpx;
        let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
        // Second (deeper) guard on the same corrupt-frame condition
        // `decode_mbmi_block` rejects — see the AOM_CODEC_CORRUPT_FRAME
        // note in `decode_block`. This one also covers the sub-8x8
        // shapes C's `bsize >= BLOCK_8X8` gate exempts but which still
        // index MAX_TXSIZE_RECT_LOOKUP / the chroma tiling below:
        // BLOCK_4X8 at 4:2:2 IS chroma-reference at odd mi_col yet has
        // no valid chroma size. `mark_corrupt`, not `assert_ne!` —
        // reachable only from a crafted bitstream, where a panic is a
        // a decoder panic. Byte-inert on conformant streams.
        if plane_bsize == 255 {
            self.mark_corrupt(format!(
                "corrupt frame: invalid chroma block size — luma bsize {bsize} \
                 has no valid chroma plane size at subsampling ({ss_x},{ss_y})"
            ));
            return Stage::Stop;
        }
        // av1_get_tx_size(plane > 0): lossless forces TX_4X4 (the chroma txb
        // loop tiles + reads coeffs at 4x4), else the max rect uv tx size.
        let uv_tx = if self.st.coded_lossless {
            TX_4X4_IDX
        } else {
            max_uv_txsize(bsize, ss_x, ss_y)
        };
        let (uv_txw, uv_txh) = (TX_SIZE_WIDE_UNIT[uv_tx], TX_SIZE_HIGH_UNIT[uv_tx]);
        let (uv_txwpx, uv_txhpx) = (TX_SIZE_WIDE[uv_tx], TX_SIZE_HIGH[uv_tx]);
        // unit_width/height: THIS 64x64 chunk's luma extent (clamped to the
        // block), ceil-scaled to chroma units (decodeframe.c:944-947).
        let unit_width =
            round_power_of_two((chunk_col + mu_w).min(max_blocks_wide) as i32, ss_x) as usize;
        let unit_height =
            round_power_of_two((chunk_row + mu_h).min(max_blocks_high) as i32, ss_y) as usize;
        // av1_set_entropy_contexts' frame-edge clip uses the CHROMA plane
        // block's in-frame extent.
        let blocks_wide_uv =
            max_block_units_ss(BLOCK_SIZE_WIDE[plane_bsize], mb_to_right_edge, ss_x);
        let blocks_high_uv =
            max_block_units_ss(BLOCK_SIZE_HIGH[plane_bsize], mb_to_bottom_edge, ss_y);
        // Prediction geometry: pd->width/height (chroma px, min 4), the
        // scaled block size the has_top_right/bottom_left walk sees, and
        // the chroma availability (equal to group-origin availability).
        let wpx = ((MI_SIZE_WIDE[bsize] * 4) >> ss_x).max(4);
        let hpx = ((MI_SIZE_HIGH[bsize] * 4) >> ss_y).max(4);
        let bsize_uv = scale_chroma_bsize(bsize, ss_x, ss_y);
        // set_mi_row_col's chroma_up_available/chroma_left_available:
        // equal to the luma up_available/left_available EXCEPT the
        // sub-8x8-odd-position group case, where it's `(mi_row/col - 1) >
        // tile->mi_row/col_start` — exactly `adj_row/col >
        // tile.mi_row/col_start` in both cases (adj_row/adj_col already
        // encode the "-1 when sub-8x8 odd position" shift above).
        let up_uv = adj_row > self.tile.mi_row_start;
        let left_uv = adj_col > self.tile.mi_col_start;
        // get_filt_type(xd, plane > 0): smoothness of the chroma
        // above/left neighbours — the bottom-right-most mi of the
        // neighbouring chroma region (set_mi_row_col's chroma_above_mbmi /
        // chroma_left_mbmi), read from the uv-mode grid.
        let cols = cfg.mi_cols;
        let base_row = mi_row - (mi_row & ss_y as i32);
        let base_col = mi_col - (mi_col & ss_x as i32);
        let uv_smooth = |m: i8| (9..=11).contains(&m);
        let ab_sm = up_uv
            && uv_smooth(self.mi_uv[((base_row - 1) * cols + base_col + ss_x as i32) as usize]);
        let le_sm = left_uv
            && uv_smooth(self.mi_uv[((base_row + ss_y as i32) * cols + base_col - 1) as usize]);
        let filt_type_uv = (ab_sm || le_sm) as i32;
        // The block origin in the chroma planes.
        let uv_org =
            ((adj_row * 4) >> ss_y) as usize * self.stride_uv + ((adj_col * 4) >> ss_x) as usize;
        // Chroma transform types are not coded: the UV intra mode implies
        // the type, demoted to DCT_DCT outside the block's ext-tx set
        // (av1_get_tx_type, PLANE_TYPE_UV intra).
        let tt_uv = uv_tx_type(info.uv_mode, uv_tx, cfg.reduced_tx_set);
        let mode_uv = get_uv_mode(info.uv_mode as usize) as usize;
        let uv_area = txb_wide(uv_tx) * txb_high(uv_tx);
        let mut tcoeff_uv = vec![0i32; uv_area];
        let mut scratch_uv = vec![0u16; uv_txwpx * uv_txhpx];
        // bd8 lowbd non-CfL chroma predicts through a u8 scratch +
        // the *_u8 kernels; a CfL block keeps the u16 scratch (the
        // CfL AC add is a u16 kernel) via widen/narrow delegation.
        let use_cfl = info.uv_mode == UV_CFL_PRED && info.use_intrabc == 0;
        let lowbd_uv = self.recon_u.is_lowbd();
        let mut scratch8_uv = if lowbd_uv && !use_cfl {
            vec![0u8; uv_txwpx * uv_txhpx]
        } else {
            Vec::new()
        };
        let mut no_ext: [u16; 0] = [];

        for plane in 1..=2usize {
            let mut blk_row = chunk_row >> ss_y;
            while blk_row < unit_height {
                let mut blk_col = chunk_col >> ss_x;
                while blk_col < unit_width {
                    // Chroma tx-type: for intrabc (is_inter) it is the
                    // CO-LOCATED luma tx-type (av1_get_tx_type inter branch),
                    // read from the luma tx_type_map at (mi_row+(blk_row<<ss_y),
                    // mi_col+(blk_col<<ss_x)) — the block's OWN mi origin, not
                    // the shared-group base — then re-validated against the
                    // inter ext-tx set for uv_tx and demoted to DCT_DCT if
                    // unused. Ordinary intra chroma uses the block-level
                    // uv_tx_type computed above.
                    let tt_uv_eff = if info.use_intrabc != 0 {
                        let lr = mi_row as usize + (blk_row << ss_y);
                        let lc = mi_col as usize + (blk_col << ss_x);
                        let luma_tt = self.luma_tt[lr * cfg.mi_cols as usize + lc] as usize;
                        if ext_tx_derive(uv_tx, true, cfg.reduced_tx_set, luma_tt, false, 0, 0).used
                            == 1
                        {
                            luma_tt
                        } else {
                            0
                        }
                    } else {
                        tt_uv
                    };
                    // (1) chroma coefficients (read_coeffs_tx_intra_block).
                    let eob = if info.skip == 0 {
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
                            &mut no_ext, // plane_type 1: no tx_type symbol
                            &mut tcoeff_uv,
                            uv_tx,
                            1,
                            tsc as usize,
                            dsc as usize,
                            true,
                            false,
                            cfg.reduced_tx_set,
                            false,
                            tt_uv_eff,
                        );
                        let cul = txb_entropy_context(&tcoeff_uv, uv_tx, tt_uv_eff, eob) as i8;
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
                        eob
                    } else {
                        0
                    };

                    // (2) chroma intra prediction (av1_predict_intra_block_facade):
                    // ordinary intra with mode = get_uv_mode(uv_mode) — DC for
                    // CfL — then the CfL AC contribution on top.
                    let (n_top, n_tr, n_left, n_bl) = intra_avail(
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
                        blk_row as i32,
                        blk_col as i32,
                        wpx,
                        hpx,
                        cfg.mi_cols,
                        cfg.mi_rows,
                        mode_uv,
                        info.angle_delta_uv * ANGLE_STEP,
                        false,
                    );
                    let off_uv = uv_org + (blk_row * 4) * self.stride_uv + blk_col * 4;
                    if info.use_intrabc != 0 {
                        // Intra block copy, chroma: reuse the luma DV, scaled
                        // by subsampling. mv_q4 = dv << (1-ss) is in 1/16
                        // chroma-pel; the integer chroma-pixel offset is
                        // mv_q4>>4 and the 2-tap intrabc bilinear fires when
                        // mv_q4&15 == 8 (only when the integer-luma-pel DV is
                        // odd on a subsampled axis — 4:4:4 is always a copy).
                        // Source is this block's own chroma recon plane, which
                        // DV validity keeps already-decoded.
                        let mvq4_row = info.dv_row << (1 - ss_y as i32);
                        let mvq4_col = info.dv_col << (1 - ss_x as i32);
                        let src = (off_uv as isize
                            + (mvq4_row >> 4) as isize * self.stride_uv as isize
                            + (mvq4_col >> 4) as isize) as usize;
                        let plane_recon = if plane == 1 {
                            &self.recon_u
                        } else {
                            &self.recon_v
                        };
                        match plane_recon {
                            ReconPlane::HighBd(p) => intrabc_chroma_predict(
                                p,
                                src,
                                self.stride_uv,
                                &mut scratch_uv,
                                uv_txwpx,
                                uv_txwpx,
                                uv_txhpx,
                                mvq4_col & 15,
                                mvq4_row & 15,
                                cfg.bd,
                            ),
                            ReconPlane::LowBd(p) => intrabc_chroma_predict_u8(
                                p,
                                src,
                                self.stride_uv,
                                &mut scratch8_uv,
                                uv_txwpx,
                                uv_txwpx,
                                uv_txhpx,
                                mvq4_col & 15,
                                mvq4_row & 15,
                            ),
                        }
                    } else if info.palette_size[1] > 0 {
                        // av1_predict_intra_block's palette branch, chroma: ONE
                        // shared colour-index map for U and V (uv_map_wpx-strided,
                        // from av1_get_block_dimensions(bsize, plane=1, ...)),
                        // looked up against palette_colors[plane * PALETTE_MAX_SIZE]
                        // (plane 1 = U, plane 2 = V — the palette_colors offset
                        // matches this loop's own `plane` var directly).
                        let (x0, y0) = (blk_col * 4, blk_row * 4);
                        let pal_base = plane * 8;
                        for r in 0..uv_txhpx {
                            for c in 0..uv_txwpx {
                                let idx = color_map_uv[(y0 + r) * uv_map_wpx + x0 + c] as usize;
                                if lowbd_uv {
                                    scratch8_uv[r * uv_txwpx + c] =
                                        info.palette_colors[pal_base + idx] as u8;
                                } else {
                                    scratch_uv[r * uv_txwpx + c] =
                                        info.palette_colors[pal_base + idx];
                                }
                            }
                        }
                    } else {
                        let plane_recon = if plane == 1 {
                            &self.recon_u
                        } else {
                            &self.recon_v
                        };
                        let n_top_u =
                            usize::try_from(n_top).expect("n_top_px must be non-negative");
                        let n_left_u =
                            usize::try_from(n_left).expect("n_left_px must be non-negative");
                        match plane_recon {
                            // A CfL block's DC prediction must land in the
                            // u16 scratch for the (u16) CfL AC add — the
                            // lowbd plane delegates via the widened apron
                            // (byte-identical); a non-CfL lowbd block
                            // predicts directly through the u8 family.
                            ReconPlane::LowBd(p) if !use_cfl => predict_intra_u8(
                                p,
                                off_uv,
                                self.stride_uv,
                                &mut scratch8_uv,
                                uv_txwpx,
                                mode_uv,
                                info.angle_delta_uv * ANGLE_STEP,
                                false,
                                0,
                                cfg.disable_edge_filter,
                                filt_type_uv,
                                uv_tx,
                                n_top_u,
                                n_tr,
                                n_left_u,
                                n_bl,
                            ),
                            plane_recon => with_wide_apron(
                                plane_recon,
                                off_uv,
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
                                        &mut scratch_uv,
                                        uv_txwpx,
                                        mode_uv,
                                        info.angle_delta_uv * ANGLE_STEP,
                                        false,
                                        0,
                                        cfg.disable_edge_filter,
                                        filt_type_uv,
                                        uv_tx,
                                        n_top_u,
                                        n_tr,
                                        n_left_u,
                                        n_bl,
                                        cfg.bd,
                                    );
                                },
                            ),
                        }
                    }
                    if info.uv_mode == UV_CFL_PRED && info.use_intrabc == 0 {
                        cfl_predict_block(
                            &mut self.cfl,
                            &mut scratch_uv,
                            0,
                            uv_txwpx,
                            uv_tx,
                            plane,
                            info.cfl_alpha_idx,
                            info.cfl_joint_sign,
                            cfg.bd,
                        );
                    }
                    {
                        let plane_recon = if plane == 1 {
                            &mut self.recon_u
                        } else {
                            &mut self.recon_v
                        };
                        match plane_recon {
                            ReconPlane::HighBd(p) => {
                                for r in 0..uv_txhpx {
                                    let d = off_uv + r * self.stride_uv;
                                    p[d..d + uv_txwpx].copy_from_slice(
                                        &scratch_uv[r * uv_txwpx..(r + 1) * uv_txwpx],
                                    );
                                }
                            }
                            ReconPlane::LowBd(_) if use_cfl => {
                                // CfL prediction lives in the u16 scratch;
                                // narrow-store it (bit-exact — clamped <= 255).
                                for r in 0..uv_txhpx {
                                    let d = off_uv + r * self.stride_uv;
                                    plane_recon.store_row(
                                        d,
                                        &scratch_uv[r * uv_txwpx..(r + 1) * uv_txwpx],
                                    );
                                }
                            }
                            ReconPlane::LowBd(p) => {
                                for r in 0..uv_txhpx {
                                    let d = off_uv + r * self.stride_uv;
                                    p[d..d + uv_txwpx].copy_from_slice(
                                        &scratch8_uv[r * uv_txwpx..(r + 1) * uv_txwpx],
                                    );
                                }
                            }
                        }
                        // (3) dequant + inverse transform + add — the
                        // block-effective dequant row of this plane.
                        if info.skip == 0 && eob > 0 {
                            let dequant = self.dequants[plane];
                            if self.st.coded_lossless {
                                // lossless: TX_4X4 + WHT, this plane's qindex-0 dequant.
                                match plane_recon {
                                    ReconPlane::HighBd(p) => reconstruct_txb_wht(
                                        &mut p[off_uv..],
                                        self.stride_uv,
                                        &tcoeff_uv,
                                        dequant,
                                        eob,
                                        cfg.bd,
                                    ),
                                    ReconPlane::LowBd(p) => reconstruct_txb_wht_u8(
                                        &mut p[off_uv..],
                                        self.stride_uv,
                                        &tcoeff_uv,
                                        dequant,
                                        eob,
                                    ),
                                }
                            } else {
                                let iqm = qm::iqmatrix(
                                    self.block_qm_level[plane],
                                    plane,
                                    uv_tx,
                                    tt_uv_eff,
                                );
                                match plane_recon {
                                    ReconPlane::HighBd(p) => reconstruct_txb_into(
                                        &mut p[off_uv..],
                                        self.stride_uv,
                                        uv_tx,
                                        tt_uv_eff,
                                        &tcoeff_uv,
                                        dequant,
                                        iqm,
                                        cfg.bd,
                                        &mut self.recon_scratch,
                                    ),
                                    ReconPlane::LowBd(p) => reconstruct_txb_u8_into(
                                        &mut p[off_uv..],
                                        self.stride_uv,
                                        uv_tx,
                                        tt_uv_eff,
                                        &tcoeff_uv,
                                        dequant,
                                        iqm,
                                        &mut self.recon_scratch,
                                    ),
                                }
                            }
                        }
                    }
                    txbs_uv.push(if eob > 0 { (eob, tt_uv_eff) } else { (0, 0) });
                    blk_col += uv_txw;
                }
                blk_row += uv_txh;
            }
        }
        Stage::Continue(())
    }

    /// `decode_token_recon_block`'s intra arm (decodeframe.c:892-995): the
    /// per-txb coefficient-read -> predict -> reconstruct walk — the intrabc
    /// var-tx leaf DFS (`recon_intra_vartx_luma`), else the uniform raster —
    /// with chroma interleaved per 64x64 chunk (`recon_intra_chroma_chunk`;
    /// a >64x64 block decodes L,U,V of each chunk in turn, which the
    /// arithmetic decoder's symbol order requires).
    ///
    /// `Stage::Stop` only via the impossible-chroma-geometry `mark_corrupt`
    /// in the chunk arm (crafted stream; byte-inert on conformant input).
    fn recon_intra_block(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        bx: &BlockCtx,
        info: &MbModeInfoKf,
        palette: &PaletteMaps,
        tx: &IntraTxLayout,
    ) -> Stage<(Vec<(usize, usize)>, Vec<(usize, usize)>)> {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            chroma_ref,
            above_mi: above,
            left_mi: left,
            ..
        } = *bx;
        let tx_size = tx.tx_size;
        let vartx_quadtree_read = tx.vartx_quadtree_read;
        let cfg = self.cfg;
        // --- decode_token_recon_block (intra): per-txb read -> predict -> recon ---
        let (txwpx, txhpx) = (TX_SIZE_WIDE[tx_size], TX_SIZE_HIGH[tx_size]);
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let max_blocks_wide = max_block_units(BLOCK_SIZE_WIDE[bsize], mb_to_right_edge);
        let max_blocks_high = max_block_units(BLOCK_SIZE_HIGH[bsize], mb_to_bottom_edge);
        // get_filt_type (reconintra.c), luma: 1 when the above or left neighbour
        // block (the same xd->above_mbmi/left_mbmi the mode contexts read) has a
        // smooth y mode.
        let is_smooth = |m: Option<MiNbrKf>| {
            m.is_some_and(|n| (SMOOTH_PRED..=SMOOTH_H_PRED).contains(&n.y_mode))
        };
        let filt_type = (is_smooth(above) || is_smooth(left)) as i32;
        // av1_read_tx_type gate: !skip_txfm && !seg-SKIP && qindex > 0 —
        // xd->qindex[segment_id] = av1_get_qindex over the FRAME base qindex
        // (decodeframe.c:5165, NOT the delta-q carry). skip == 0 already
        // implies the segment's SEG_LVL_SKIP feature is inactive (read_skip
        // returns a forced 1 when it is active).
        let signal_gate = info.skip == 0
            && av1_get_qindex(&cfg.seg, info.segment_id as usize, cfg.base_qindex) > 0;
        let area = txb_wide(tx_size) * txb_high(tx_size);
        let mut buf = LumaScratch {
            tcoeff: vec![0i32; area],
            scratch: vec![0u16; txwpx * txhpx],
            // bd8 lowbd: prediction goes through a u8 scratch + the *_u8
            // kernels (Phase B — the widen/narrow delegation is gone on this
            // path).
            scratch8: if self.recon.is_lowbd() {
                vec![0u8; txwpx * txhpx]
            } else {
                Vec::new()
            },
        };
        let mut txbs = Vec::new();

        // Intrabc var-tx: the reconstruction phase (`decode_reconstruct_tx`,
        // decodeframe.c) walks the per-leaf partition rather than tiling the
        // block with one scalar tx size. Each leaf reads its own coeffs +
        // tx_type (inter ext-tx at the leaf size) in DFS order, then the
        // integer block copy + inverse transform, all at the leaf's size.
        //
        // The gate is "was the quadtree read", NOT "are the leaf sizes
        // distinct" (KB-29). Equal leaf sizes do NOT imply the raster fast
        // loop is equivalent: a BLOCK_16X8 split all the way to TX_4X4 has
        // eight same-size leaves whose DFS order is
        // (0,0)(0,1)(1,0)(1,1)(0,2)(0,3)(1,2)(1,3) — not raster — so the
        // per-txb `txb_skip_ctx` sequence differs and the arithmetic decode
        // desyncs from byte one of the third txb.
        //
        // Coded-lossless is the OPPOSITE case: no quadtree is signalled AND
        // C's `get_vartx_max_txsize` (blockd.h:1452) collapses the root to
        // TX_4X4, so `decode_reconstruct_tx` degenerates to a flat 4x4 raster
        // — exactly what `do_uniform` produces. (KB-65: the writer
        // side had been emitting the DFS order at lossless, which every
        // conforming decoder — including this one pre-fix — reads as raster.)
        let do_uniform = !(info.use_intrabc != 0 && vartx_quadtree_read);
        if !do_uniform {
            self.recon_intra_vartx_luma(dec, cdfs, bx, info, tx, signal_gate, &mut txbs);
        }

        // decode_token_recon_block (decodeframe.c:929-962): iterate the block in
        // 64x64 chunks (max_unit_bsize = BLOCK_64X64) and, within each chunk, do
        // plane 0's txbs then plane 1/2's txbs. For blocks larger than 64x64 this
        // interleaves luma/chroma per 64-unit (a 128-wide block decodes L,U,V of
        // its first 64x64, THEN L,U,V of the next), which the arithmetic decoder
        // requires; for <=64x64 blocks there is exactly one chunk, so the order
        // is identical to the previous plane-major-over-the-whole-block walk.
        let mut txbs_uv = Vec::new();
        let mu_w = max_blocks_wide.min(MI_SIZE_WIDE[BLOCK_64X64] as usize);
        let mu_h = max_blocks_high.min(MI_SIZE_HIGH[BLOCK_64X64] as usize);
        // The chunk walk runs for EVERY block, including the non-uniform
        // intrabc var-tx case handled above: that arm reads plane 0's leaves
        // only, and `write_tokens_b`'s inter arm still writes U and V for the
        // block (`write_inter_txb_coeff(plane)` — bitstream.c:1463-1468, with
        // `break` only on `!is_chroma_ref`). Gating the whole walk on
        // `do_uniform` left those chroma `all_zero` symbols unread, desyncing
        // the decode of any conformant stream with a split intrabc var-tx
        // block (KB-29's decoder half). Only the LUMA sub-loop is skipped when
        // the non-uniform arm has already consumed it.
        //
        // ORDERING CAVEAT, stated rather than assumed: for a >64x64 non-uniform
        // intrabc block (>1 chunk) the luma arm above reads ALL chunks' leaves
        // before this loop reads any chroma, where C interleaves L,U,V per
        // 64x64 chunk. That case is not reachable from this port's encoder
        // (`rd_pick_intrabc_mode_sb` is offered per leaf and the var-tx root
        // walk is per TX_64X64 unit), and it was 100% wrong before (no chroma
        // at all), so this is strictly closer; a multi-chunk non-uniform
        // intrabc block still needs the leaf read folded into this loop.
        let mut chunk_row = 0usize;
        while chunk_row < max_blocks_high {
            let mut chunk_col = 0usize;
            while chunk_col < max_blocks_wide {
                let g = ChunkGeom {
                    row: chunk_row,
                    col: chunk_col,
                    step_w: mu_w,
                    step_h: mu_h,
                    blocks_wide: max_blocks_wide,
                    blocks_high: max_blocks_high,
                };
                if do_uniform {
                    self.recon_intra_luma_chunk(
                        dec,
                        cdfs,
                        bx,
                        info,
                        palette,
                        tx,
                        &g,
                        &mut buf,
                        filt_type,
                        signal_gate,
                        &mut txbs,
                    );
                }

                // --- decode_token_recon_block, planes 1..2: the chroma txb loop of the
                // (single, <=64x64) 64x64 chunk — runs after ALL of plane 0, so the
                // block's own luma is already in the CfL store. Only the
                // chroma-reference block of a shared group decodes chroma, covering
                // the merged area from the adjusted plane origin. ---
                if !cfg.monochrome && chroma_ref {
                    let Stage::Continue(()) = self.recon_intra_chroma_chunk(
                        dec,
                        cdfs,
                        bx,
                        info,
                        palette,
                        &g,
                        &mut txbs_uv,
                    ) else {
                        return Stage::Stop;
                    };
                }
                chunk_col += mu_w;
            }
            chunk_row += mu_h;
        }
        Stage::Continue((txbs, txbs_uv))
    }

    /// Everything an intra block does AFTER its mode info is read: the palette
    /// colour-map tokens, the tx-size read + txfm-context stamp, the skip
    /// entropy reset, the `decode_token_recon_block` per-txb read -> predict ->
    /// reconstruct walk, and the neighbour-grid stamps.
    ///
    /// Shared VERBATIM by both intra paths, because C shares it too: the
    /// decoder installs `read_coeffs_tx_intra_block` /
    /// `predict_and_reconstruct_intra_block` once (decodeframe.c:2756/:2761) with
    /// NO frame-type dependence, and `decode_token_recon_block` branches only on
    /// `!is_inter_block(mbmi)` (:920). An intra block in an INTER frame has
    /// `ref_frame[0] == INTRA_FRAME` and `use_intrabc == 0`, so it takes these
    /// exact arms:
    ///
    ///  * tx size: `inter_block_tx` is 0 (decodeframe.c:1179), so the var-tx
    ///    quadtree is skipped and `read_tx_size(.., is_inter = 0, ..)` runs — and
    ///    `(!is_inter || allow_select_inter)` at :1150 is unconditionally true, so
    ///    it is `read_selected_tx_size` exactly as on a KEY frame. `set_txfm_ctxs`
    ///    gets `skip_txfm && is_inter_block(mbmi)` = 0 (:1197).
    ///  * tx type: `av1_read_tx_type`'s `inter_block` is 0, so it reads
    ///    `intra_ext_tx_cdf[eset][square_tx_size][intra_dir]` (decodemv.c:665),
    ///    with `intra_dir` = `fimode_to_intradir[filter_intra_mode]` when
    ///    filter-intra is on, else the Y mode (:660-664).
    ///
    /// So the intra-in-inter block needs no special-casing here at all — which is
    /// the point of sharing this body rather than transcribing a second copy.
    pub(crate) fn decode_intra_block_body(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        info: MbModeInfoKf,
        bx: &BlockCtx,
    ) {
        let BlockCtx {
            mi_row,
            mi_col,
            bsize,
            partition,
            ..
        } = *bx;
        let cfg = self.cfg;
        // Stamp this block's palette facts over its footprint — matches stamp_mi's
        // placement, right after the mode-info read (subsequent blocks' above/left
        // palette-cache lookups must see it).
        self.stamp_palette(
            mi_row,
            mi_col,
            bsize,
            PaletteNbrKf {
                size: info.palette_size,
                colors: info.palette_colors,
            },
        );
        // set_segment_id (read_intra_segment_id, decodemv.c): stamp the
        // block's resolved id over its frame-cropped mi footprint. The C
        // stamps between the segment read and the rest of the mode info;
        // nothing in between reads the map, so stamping here is equivalent.
        if cfg.seg.enabled {
            let x_mis = MI_SIZE_WIDE[bsize].min(cfg.mi_cols - mi_col);
            let y_mis = MI_SIZE_HIGH[bsize].min(cfg.mi_rows - mi_row);
            for r in 0..y_mis {
                let base = ((mi_row + r) * cfg.mi_cols + mi_col) as usize;
                self.seg_map[base..base + x_mis as usize].fill(info.segment_id as u8);
            }
        }
        let palette = self.read_intra_color_maps(dec, cdfs, bx, &info);

        let tx = self.read_intra_tx_layout(dec, cdfs, bx, &info);

        // parse_decode_block (decodeframe.c): with delta-q present, every
        // block's dequant is recomputed from the running current_base_qindex
        // (already advanced by this block's SB-level delta read inside the
        // mode-info decode — mbmi->current_qindex == the carry); the C
        // refills all MAX_SEGMENTS seg_dequant_QTX rows via
        // av1_get_qindex(seg, i, carry) and the txb read consumes row
        // [mbmi->segment_id] — computed here directly for the block's
        // segment. Without delta-q the frame-level rows come from
        // setup_segmentation_dequant (xd->qindex[i] = av1_get_qindex(seg, i,
        // base_qindex)) — the same formula on the never-moved carry. The
        // per-plane dc/ac deltas fold in through av1_{dc,ac}_quant_QTX.
        if cfg.delta_q_present || cfg.seg.enabled {
            debug_assert!(
                !cfg.delta_q_present || self.st.current_base_qindex == info.current_qindex
            );
            let eff_qindex = av1_get_qindex(
                &cfg.seg,
                info.segment_id as usize,
                self.st.current_base_qindex,
            );
            self.dequants = plane_dequants(cfg, eff_qindex);
        }
        // QM level per plane for this block: only the segment's lossless status
        // varies it (qmatrix_level_* is frame-constant), so recompute only when
        // the frame uses QM. Non-QM frames keep the flat init and the pre-QM
        // (flat-dequant) path byte-for-byte.
        if cfg.using_qmatrix {
            self.block_qm_level = frame_qm_levels(cfg, info.segment_id as usize);
        }

        // --- parse_decode_block tail: skip blocks reset their entropy context
        // (av1_reset_entropy_context) ---
        self.reset_intra_skip_ctx(bx, &info);

        let Stage::Continue((txbs, txbs_uv)) =
            self.recon_intra_block(dec, cdfs, bx, &info, &palette, &tx)
        else {
            return;
        };

        self.stamp_mi(
            mi_row,
            mi_col,
            bsize,
            MiNbrKf {
                y_mode: info.y_mode,
                skip_txfm: info.skip,
            },
        );
        // Block-vector grid stamp (intrabc projection of xd->mi): on a KEY frame
        // ref_frame[0] is always INTRA_FRAME and ref_frame[1] NONE_FRAME; only
        // use_intrabc + the block's own DV (mv[0]) and bsize are consulted by the
        // next block's av1_find_mv_refs / get_tx_size_context. `mode` is read only
        // by is_global_mv_block (never a match on KEY frames), so it stays 0.
        self.stamp_dv(
            mi_row,
            mi_col,
            bsize,
            DvNbr {
                bsize,
                ref_frame0: 0,
                ref_frame1: -1,
                use_intrabc: info.use_intrabc != 0,
                mode: 0,
                mv0_row: info.dv_row,
                mv0_col: info.dv_col,
                mv1_row: 0,
                mv1_col: 0,
                compound_idx: 0,
                comp_group_idx: 0,
            },
        );
        // The uv-mode grid stamp: non-chroma-reference blocks carry UV_DC_PRED
        // (read_intra_frame_mode_info's else-branch), which the tail returns.
        {
            let x_mis = MI_SIZE_WIDE[bsize].min(cfg.mi_cols - mi_col);
            let y_mis = MI_SIZE_HIGH[bsize].min(cfg.mi_rows - mi_row);
            for r in 0..y_mis {
                let base = ((mi_row + r) * cfg.mi_cols + mi_col) as usize;
                self.mi_uv[base..base + x_mis as usize].fill(info.uv_mode as i8);
            }
        }
        self.blocks.push(DecodedBlockKf {
            mi_row,
            mi_col,
            bsize,
            partition,
            info,
            tx_size: tx.tx_size,
            txbs,
            txbs_uv,
            // Intra/intrabc block: the loop-filter grid derives ref/is_inter/
            // mode from `info` (ref0 = INTRA_FRAME, is_inter = use_intrabc).
            inter_lf: None,
        });
    }
}
