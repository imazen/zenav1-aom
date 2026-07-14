//! `av1_rd_pick_partition` with REAL leaves — the NONE-vs-SPLIT recursion
//! (partition_search.c:5653) wired to [`crate::rd_pick::rd_pick_intra_mode_sb`]
//! leaf evaluation and the winner-subtree DRY_RUN [`crate::encode_sb`]
//! propagation, over live tile contexts, recon planes, and the mi-grid mode
//! state. This makes the partition skeleton REAL on its two ported types:
//! sibling blocks see each other's winner pixels (recon), entropy/partition/
//! txfm contexts (the dry-run stamps), and mi-grid neighbour modes.
//!
//! # The per-node C sequence modelled (partition_search.c:5653-6046)
//!
//! 1. `init_partition_search_state_params` — geometry + `partition_cost =
//!    mode_costs->partition_cost[pl_ctx_idx]` (`partition_plane_context`
//!    over the LIVE partition-context arrays) + stage gates.
//! 2. `av1_set_offsets` + `setup_block_rdmult` (rdmult CONSTANT across the
//!    recursion: frame RDMULT at GOOD/KEY/NO_AQ; at ALLINTRA the per-SB
//!    `intra_sb_rdmult_modifier` fold is applied ONCE at the SB root and
//!    stays constant below it — the caller passes the folded value) +
//!    `av1_rd_cost_update(rdmult, &best_rdc)`.
//! 3. `av1_save_context` (:5754) — per-plane above/left ENTROPY contexts +
//!    partition contexts + txfm contexts over the node extent. Pixels are
//!    NOT saved: winner pixels propagate via the dry-run encode alone.
//! 4. NONE stage (`none_partition_search`:4399): `pick_sb_modes` (the leaf
//!    — [`leaf_pick_sb_modes`]) with `best_remain = best_rdc - pt_cost`;
//!    strict-< best update; `av1_restore_context` at the stage tail (:4492).
//! 5. SPLIT stage (`split_partition_search`:4512): 4 recursive children
//!    (`pc_index` = child idx) with running `sum_rdc` budgets; the stage
//!    tail restores context when `bsize <= max_partition_size || bsize ==
//!    sb_size` (:4645-4647; always true in this envelope).
//! 6. The winner encode (:5998-6026): at the SB root the C emits
//!    OUTPUT_ENABLED (adds pack-stage CDF/token state; contexts/pixels are
//!    stamped identically) — modelled as the same DRY_RUN walk; below SB,
//!    `should_do_dry_run_encode_for_current_block` (:5556) gates the
//!    DRY_RUN winner-subtree walk: `bsize > max_partition -> false`;
//!    `pc_index != 3 -> true`; a 4th child re-encodes only when `bsize ==
//!    max_partition != sub_sb_size` (its data is otherwise re-created by
//!    the nearest `index != 3` ancestor's own dry-run before any reader).
//!
//! # mi-grid neighbour semantics (verified)
//!
//! The encoder's mi alloc granularity is `mi_alloc_bsize =
//! default_min_partition_size = BLOCK_4X4` (`enc_set_mb_mi`,
//! encoder_utils.h:93-99) — every 4x4 cell owns a struct; `set_mi_offsets`
//! repoints the ORIGIN grid cell at pick time and `av1_update_state`'s grid
//! fill (encode_b, ANY dry_run) points the block's cells at the winner. The
//! leaf search reads neighbour WINNER modes through `xd->above/left_mbmi`
//! (the cells at `(mi_row-1, mi_col)` / `(mi_row, mi_col-1)`), which the
//! dry-run discipline keeps coherent for every reader (any never-re-encoded
//! `index == 3` subtree has no reader before its ancestor's walk). The
//! [`ModeGrid`] models the mode byte per cell — the only mi-grid field the
//! KEY intra leaf reads (neighbour `skip_txfm` is 0 for every KEY intra
//! block so the skip context is constantly 0; `is_inter` is always false so
//! `get_tx_size_context` takes the txfm-context bytes unmodified).
//!
//! # Leaf inputs derived live (pick_sb_modes, :850-975)
//!
//! - `x->source_variance = av1_get_perpixel_variance_facade(...)`
//!   ([`perpixel_variance_y`]: plane-0 variance vs the flat
//!   `AV1_[HIGH_]VAR_OFFS` buffer `>> num_pels_log2`, encodeframe.c:190).
//! - the HOG `directional_mode_skip_mask` (`intra_pruning_with_hog = 1` at
//!   speed 0 BOTH usages; threshold `thresh[0] = -1.2`,
//!   intra_mode_search.c:1505).
//! - above/left neighbour Y modes from the [`ModeGrid`].
//! - `skip_ctx = 0` (KEY intra invariant, asserted); `tx_size_ctx =
//!   get_tx_size_context` over the live txfm-context arrays.
//! - the entropy-context slices at the block position (what
//!   `av1_get_entropy_contexts` copies).
//!
//! # Scope
//!
//! NONE + SPLIT (2 of 10 partition types), KEY intra, interior SBs,
//! sb_size <= 64, no segmentation. MISSING: rect/AB/4-way stages, the
//! edge-block partition-cost override, the SB-level must-find retry,
//! `log_sub_block_var` (the ALLINTRA SB-root rdmult modifier input — the
//! folded rdmult is an input here), and the OUTPUT pack-stage state at the
//! SB root.
//!
//! # RECT stage survey (next chunk; all verified in source)
//!
//! **THE KEY-FRAME RECT STAGE IS NN-FREE.** Every NN prune around HORZ/VERT
//! is `!frame_is_intra_only(cm)`-gated and therefore DEAD in the one-KEY-
//! frame envelope, for BOTH usages:
//! - `av1_ml_prune_rect_partition` (the 9-feature rect NN,
//!   partition_strategy.c:1124): gate at partition_search.c:4336 requires
//!   `!frame_is_intra_only` (also `!ml_early_term_after_part_split_level`,
//!   which is 1 sub-720p — either kills it);
//! - `av1_ml_early_term_after_split` (partition_strategy.c:1017): gate at
//!   partition_search.c:4323 requires `!frame_is_intra_only`;
//! - `simple_motion_search_prune_rect` (sf = 1 both usages):
//!   partition_strategy.c:692 requires `!frame_is_intra_only`;
//! - `prune_rect_part_using_none_pred_mode`: sf = 0 at speed 0 both usages.
//!
//! The one LIVE usage-differing knob: `less_rectangular_check_level`
//! (ALLINTRA 1 / GOOD 0) — pure integer logic in the SPLIT stage's ELSE arm
//! (partition_search.c:4630-4640): when SPLIT did NOT beat best and
//! (`level == 2 || idx <= 2`), `do_rectangular_split &= !(none_rd > 0 &&
//! none_rd < sum_rdc.rdcost)` — requires tracking `none_rd` (the NONE
//! leaf's rdcost before pt_cost? none_partition_search:4474 stores
//! `this_rdc.rdcost` AFTER pt_cost — re-verify the exact stored value when
//! porting) and the split stage's final `sum_rdc`.
//!
//! `rectangular_partition_search` (partition_search.c:3520-3648), per type
//! i in {HORZ, VERT} (`start/end_type` full range at speed 0):
//! 1. `is_rect_part_allowed` (:3506): `!terminate &&
//!    partition_rect_allowed[i] && !prune_rect_part[i] &&
//!    (do_rectangular_split || active_edge)` — with
//!    `partition_rect_allowed` from init: `has_cols/has_rows` + rect
//!    enabled + subsize valid for chroma (`get_plane_block_size(subsize)
//!    != BLOCK_INVALID` — the 4:2:2 tall-block guard).
//! 2. `sum_rdc = {rate: partition_cost[type], rdcost: RDCOST(rate, 0)}`.
//! 3. sub-block 0 at (mi_row, mi_col): `rd_pick_rect_partition` (:3471) =
//!    `best_remain = best - sum` -> `pick_sb_modes(partition_type)` ->
//!    `av1_rd_cost_update` -> accumulate (INT_MAX -> rdcost MAX); records
//!    `rect_part_rd[i][0]` (an AB-stage input).
//! 4. If `sum < best && is_not_edge` (has_rows for HORZ / has_cols VERT):
//!    `av1_update_state + encode_superblock(DRY_RUN_NORMAL)` of sub 0 —
//!    the MID-STAGE propagation (sub 1 reads sub 0's pixels+contexts;
//!    note: encode_superblock DIRECTLY, no encode_b — no partition-ctx
//!    stamp, no rdmult save) + `is_rect_ctx_is_ready[i]` bookkeeping
//!    (palette/CfL-free sub-0 winners feed the AB stage's
//!    `reuse_prev_rd_results_for_part_ab = 1`). Then sub-block 1 at the
//!    edge position (mi_row_edge/mi_col_edge = +hbs).
//! 5. Best update: `sum < best` -> rdcost recompute -> strict-< ->
//!    `partitioning = HORZ/VERT`.
//! 6. `av1_restore_context` at EACH type's loop tail (:3644) — HORZ's
//!    sub-0 encode debris is restored before VERT evaluates.
//!
//! Port plan: extend [`SbTree`] with `Horz([LeafWinner; 2])`/`Vert(..)` +
//! the encode_sb HORZ/VERT walk arms (partition_search.c:1640-1660), add
//! the stage between NONE and SPLIT result handling in
//! [`rd_pick_partition_real`] (C order: NONE -> SPLIT -> rect), track
//! `none_rd`/`split_rd[4]` per node for the less_rect arm, and sweep BOTH
//! usages in the diff (less_rect ON under ALLINTRA).

use crate::encode_sb::{
    encode_sb_dry, LeafEncodeOut, LeafWinner, SbEncodeEnv, SbTree, TileCtxState,
};
use crate::hog::prune_intra_mode_with_hog_y;
use crate::intra_rd::{Block4x4VarInfo, IntraSbyGates, IntraSbySearchCfg};
use crate::intra_uv_rd::{chroma_plane_offset, is_chroma_reference, UvLoopPolicy, UvRdEnv};
use crate::mode_costs::{CflCosts, IntraModeCosts};
use crate::partition::{rd_cost_update, rd_stats_subtraction, split_subsize, PartRdStats};
use crate::rd_pick::{rd_pick_intra_mode_sb, RdPickUvArgs, RdPickUvOutcome, ReencodeParams};
use crate::mode_costs::TxSizeCosts;
use crate::tx_search::{TxTypeSearchPolicy, TxfmYrdEnv, MI_SIZE_HIGH_B, MI_SIZE_WIDE_B};
use aom_dist::highbd_variance;
use aom_entropy::partition::{
    get_plane_block_size, get_tx_size_context, partition_plane_context,
};
use aom_intra::cfl::CflCtx;
use aom_txb::TxTypeCosts;

/// `num_pels_log2_lookup[BLOCK_SIZES_ALL]` (common_data.h).
const NUM_PELS_LOG2: [u32; 22] =
    [4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10];

/// `av1_get_perpixel_variance(_facade)` for plane 0 (encodeframe.c:190):
/// block variance against the flat `AV1_[HIGH_]VAR_OFFS` buffer (128 <<
/// (bd-8)), `ROUND_POWER_OF_TWO`-normalized by the pel count. Composes the
/// bit-exact [`aom_dist::highbd_variance`] (the `aom_highbd_<bd>_variance`
/// family; the bd-8 variant is numerically the lowbd kernel `aomenc` uses
/// for 8-bit sources).
pub fn perpixel_variance_y(src: &[u16], off: usize, stride: usize, bsize: usize, bd: u8) -> u32 {
    const BLK_W: [usize; 22] =
        [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
    const BLK_H: [usize; 22] =
        [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
    let (w, h) = (BLK_W[bsize], BLK_H[bsize]);
    let offs = vec![128u16 << (bd - 8); w];
    let (var, _sse) = highbd_variance(&src[off..], stride, &offs, 0, w, h, bd);
    let bits = NUM_PELS_LOG2[bsize];
    (var + (1 << (bits - 1))) >> bits
}

/// The per-mi-cell winner Y mode state (`av1_update_state`'s mi-grid fill;
/// module docs). `stride` = frame `mi_cols`.
pub struct ModeGrid {
    pub modes: Vec<u8>,
    pub stride: usize,
}

impl ModeGrid {
    /// All-DC grid (harness seeds real neighbour history where relevant).
    pub fn dc(mi_rows: usize, mi_cols: usize) -> Self {
        ModeGrid { modes: vec![0; mi_rows * mi_cols], stride: mi_cols }
    }
    fn stamp(&mut self, mi_row: i32, mi_col: i32, bsize: usize, mode: u8, mi_rows: i32, mi_cols: i32) {
        let rows = (MI_SIZE_HIGH_B[bsize] as i32).min(mi_rows - mi_row) as usize;
        let cols = (MI_SIZE_WIDE_B[bsize] as i32).min(mi_cols - mi_col) as usize;
        for r in 0..rows {
            let base = (mi_row as usize + r) * self.stride + mi_col as usize;
            self.modes[base..base + cols].fill(mode);
        }
    }
    fn at(&self, mi_row: i32, mi_col: i32) -> u8 {
        self.modes[mi_row as usize * self.stride + mi_col as usize]
    }
}

/// The frame-level leaf-search configuration (`pick_sb_modes` +
/// `av1_rd_pick_intra_mode_sb` inputs shared across leaves).
pub struct PickFrameCfg<'a> {
    pub mode_costs: &'a IntraModeCosts,
    /// KEY-frame Y mode costs `y_mode_costs[above_ctx][left_ctx][mode]`
    /// selected per leaf via `intra_mode_context[neighbour mode]`.
    pub tx_size_costs: &'a TxSizeCosts,
    pub skip_costs: &'a [[i32; 2]; 3],
    pub tx_type_costs_y: &'a TxTypeCosts,
    pub pol: &'a TxTypeSearchPolicy,
    pub uv_lp: &'a UvLoopPolicy,
    pub intra_uv_mode_cost: &'a [[[i32; 14]; 13]; 2],
    pub cfl_costs: &'a CflCosts,
    /// `mode_costs->partition_cost[pl_ctx][..]` rows.
    pub partition_costs: &'a [[i32; 10]; 20],
    /// `oxcf.mode == ALLINTRA` (the leaf variance-factor arm; the caller
    /// also pre-folds the SB rdmult modifier into `SbEncodeEnv::rdmult`).
    pub allintra: bool,
    pub speed: i32,
    pub qindex: i32,
    pub enable_filter_intra: bool,
    pub enable_tx64: bool,
    pub enable_rect_tx: bool,
    /// sf `intra_sf.intra_pruning_with_hog` (1 at speed 0, both usages;
    /// KEY-frame threshold row -1.2).
    pub intra_pruning_with_hog: bool,
    /// `x->sb_enc.max_partition_size` (default `BLOCK_128X128` at KEY —
    /// the dry-run gate + split-restore inputs).
    pub max_partition_size: usize,
    /// `x->sb_enc.min_partition_size` (default `BLOCK_4X4` at KEY;
    /// `aomenc --min-partition-size` raises it — the
    /// `av1_prune_partitions_by_max_min_bsize` clamp,
    /// partition_strategy.c:1837).
    pub min_partition_size: usize,
}

/// One leaf evaluation's differential-visibility record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafVisit {
    pub mi_row: i32,
    pub mi_col: i32,
    pub bsize: usize,
    /// The `best_remain` budget the leaf received (rdcost).
    pub budget: i64,
    /// The leaf's returned (rate, dist, rdcost) after the INT_MAX
    /// normalization.
    pub rate: i32,
    pub dist: i64,
    pub rdcost: i64,
}

/// `pick_sb_modes` (partition_search.c:850) for a KEY intra leaf: derive the
/// per-leaf inputs live (module docs) and run the whole-block
/// [`rd_pick_intra_mode_sb`]. Returns the normalized rd stats + the winner
/// as an [`LeafWinner`] (None when `rate == INT_MAX`).
#[allow(clippy::too_many_arguments)]
fn leaf_pick_sb_modes(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &TileCtxState,
    grid: &ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    best_remain: &PartRdStats,
) -> (PartRdStats, Option<LeafWinner>) {
    // av1_rd_cost_update(x->rdmult, &best_rd) on entry (pick_sb_modes:927).
    let mut best_rd = *best_remain;
    rd_cost_update(env.rdmult, &mut best_rd);

    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_h = MI_SIZE_HIGH_B[bsize];
    let up_available = mi_row > env.tile_row_start;
    let left_available = mi_col > env.tile_col_start;
    let is_chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;

    // x->source_variance (pick_sb_modes:919).
    let source_variance = perpixel_variance_y(env.src_y, ref_off_y, env.stride, bsize, env.bd);

    // The HOG directional prune mask (av1_rd_pick_intra_sby_mode preamble).
    let mut skip_mask = [false; 13];
    if cfg.intra_pruning_with_hog {
        // Interior blocks: mb_to_*_edge large positive.
        let mb_right = (env.mi_cols - mi_w as i32 - mi_col) * 4 * 8;
        let mb_bottom = (env.mi_rows - mi_h as i32 - mi_row) * 4 * 8;
        prune_intra_mode_with_hog_y(
            env.src_y, ref_off_y, env.stride, bsize, mb_right, mb_bottom, -1.2, &mut skip_mask,
        );
    }
    let gates = IntraSbyGates::speed0(skip_mask);

    // Neighbour winner modes (module docs: the mi-grid reads).
    let above_mode = if up_available { Some(i32::from(grid.at(mi_row - 1, mi_col))) } else { None };
    let left_mode = if left_available { Some(i32::from(grid.at(mi_row, mi_col - 1))) } else { None };

    // skip ctx: every KEY intra neighbour has skip_txfm == 0 => ctx 0.
    let skip_ctx = 0usize;
    let tx_size_ctx = get_tx_size_context(
        bsize,
        tile.above_tctx[a0],
        tile.left_tctx[l0],
        up_available,
        left_available,
        None,
        None,
    );

    let above_y: Vec<i8> = tile.above_ectx[0][a0..a0 + mi_w].to_vec();
    let left_y: Vec<i8> = tile.left_ectx[0][l0..l0 + mi_h].to_vec();
    let mut y_env = TxfmYrdEnv {
        sb_size: env.sb_size,
        bsize,
        mi_row,
        mi_col,
        up_available,
        left_available,
        tile_col_end: env.tile_col_end,
        tile_row_end: env.tile_row_end,
        partition: 0, // PARTITION_NONE (pick_sb_modes' partition arg)
        mi_cols: env.mi_cols,
        mi_rows: env.mi_rows,
        ref_off: ref_off_y,
        ref_stride: env.stride,
        src: env.src_y,
        src_off: ref_off_y,
        src_stride: env.stride,
        disable_edge_filter: env.disable_edge_filter,
        filter_type: env.filter_type,
        mode: 0,
        angle_delta: 0,
        use_filter_intra: false,
        filter_intra_mode: 0,
        lossless: env.lossless,
        reduced_tx_set_used: env.reduced_tx_set_used,
        bd: env.bd,
        rows: env.rows_y,
        rdmult: env.rdmult,
        coeff_costs: env.coeff_costs_y,
        tx_type_costs: cfg.tx_type_costs_y,
        skip_costs: cfg.skip_costs,
        skip_ctx,
        tx_size_costs: cfg.tx_size_costs,
        tx_size_ctx,
        tx_mode_is_select: true,
        above_ctx: &above_y,
        left_ctx: &left_y,
    };
    let sby_cfg = IntraSbySearchCfg {
        gates: &gates,
        top_intra_model_count_allowed: 4,
        adapt_top_model_rd_count_using_neighbors: false,
        above_mode,
        left_mode,
        qindex: cfg.qindex,
        mode_costs: cfg.mode_costs,
        try_palette: false,
        palette_bsize_ctx: 0,
        palette_mode_ctx: 0,
        enable_filter_intra: cfg.enable_filter_intra,
        allow_intrabc: false,
        pol: cfg.pol,
        source_variance,
        enable_tx64: cfg.enable_tx64,
        enable_rect_tx: cfg.enable_rect_tx,
        allintra: cfg.allintra,
        speed: cfg.speed,
        mb_to_right_edge: (env.mi_cols - mi_w as i32 - mi_col) * 4 * 8,
        mb_to_bottom_edge: (env.mi_rows - mi_h as i32 - mi_row) * 4 * 8,
    };
    let mut var_cache = Block4x4VarInfo::sb_cache(env.sb_size);

    // Chroma args (num_planes > 1).
    let ref_off_uv =
        chroma_plane_offset(env.base_uv, env.stride, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    let mut chroma_up_available = up_available;
    let mut chroma_left_available = left_available;
    if env.ss_x != 0 && mi_w < 2 {
        chroma_left_available = (mi_col - 1) > env.tile_col_start;
    }
    if env.ss_y != 0 && mi_h < 2 {
        chroma_up_available = (mi_row - 1) > env.tile_row_start;
    }
    let plane_bsize = get_plane_block_size(bsize, env.ss_x, env.ss_y);
    let (pmw, pmh) = (MI_SIZE_WIDE_B[plane_bsize], MI_SIZE_HIGH_B[plane_bsize]);
    let au = (mi_col >> env.ss_x) as usize;
    let lu = ((mi_row & 31) >> env.ss_y) as usize;
    let above_u: Vec<i8> = tile.above_ectx[1][au..au + pmw].to_vec();
    let left_u: Vec<i8> = tile.left_ectx[1][lu..lu + pmh].to_vec();
    let above_v: Vec<i8> = tile.above_ectx[2][au..au + pmw].to_vec();
    let left_v: Vec<i8> = tile.left_ectx[2][lu..lu + pmh].to_vec();
    const BLK_W: [usize; 22] =
        [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
    const BLK_H: [usize; 22] =
        [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
    let cfl_allowed = !env.lossless && BLK_W[bsize] <= 32 && BLK_H[bsize] <= 32;
    let mut uv_env = UvRdEnv {
        sb_size: env.sb_size,
        bsize,
        mi_row,
        mi_col,
        chroma_up_available,
        chroma_left_available,
        tile_col_end: env.tile_col_end,
        tile_row_end: env.tile_row_end,
        partition: 0,
        mi_cols: env.mi_cols,
        mi_rows: env.mi_rows,
        ss_x: env.ss_x,
        ss_y: env.ss_y,
        ref_off: [ref_off_uv, ref_off_uv],
        ref_stride: env.stride,
        src_u: env.src_u,
        src_v: env.src_v,
        src_off: [ref_off_uv, ref_off_uv],
        src_stride: env.stride,
        disable_edge_filter: env.disable_edge_filter,
        filter_type: env.filter_type,
        luma_mode: 0,
        luma_use_fi: false,
        luma_fi_mode: 0,
        lossless: env.lossless,
        reduced_tx_set_used: env.reduced_tx_set_used,
        bd: env.bd,
        rows_u: env.rows_u,
        rows_v: env.rows_v,
        rdmult: env.rdmult,
        coeff_costs: env.coeff_costs_uv,
        tx_type_costs: env.tx_type_costs,
        above_ctx: [&above_u, &above_v],
        left_ctx: [&left_u, &left_v],
    };

    let re = ReencodeParams {
        sharpness: env.sharpness,
        enable_optimize_b: env.enable_optimize_b,
    };
    let outcome = {
        let uv_args = if env.monochrome {
            None
        } else {
            Some(RdPickUvArgs {
                env: &mut uv_env,
                recon_u,
                recon_v,
                cfl,
                is_chroma_ref,
                cfl_allowed,
                intra_uv_mode_cost: cfg.intra_uv_mode_cost,
                costs: cfg.mode_costs,
                cfl_costs: cfg.cfl_costs,
                lp: cfg.uv_lp,
            })
        };
        rd_pick_intra_mode_sb(
            &mut y_env,
            recon_y,
            &sby_cfg,
            &mut var_cache,
            best_rd.rdcost,
            env.coeff_costs_y,
            re,
            uv_args,
        )
    };

    match outcome.best {
        None => {
            // rd_cost->rate == INT_MAX -> rdcost = INT64_MAX
            // (pick_sb_modes:969-970).
            (PartRdStats::invalid(), None)
        }
        Some(best) => {
            let stats = PartRdStats { rate: best.rate, dist: best.dist, rdcost: best.rdcost };
            let (uv_mode, angle_delta_uv, cfl_alpha_idx, cfl_alpha_signs) = match &best.uv {
                RdPickUvOutcome::Searched(w, _) => (
                    w.uv_mode,
                    w.angle_delta_uv,
                    i32::from(w.cfl_alpha_idx),
                    i32::from(w.cfl_alpha_signs),
                ),
                // !chroma_ref / monochrome: the uv mbmi fields are dead
                // state (nothing reads them — store_cfl_required only reads
                // uv_mode on chroma-ref blocks; packing is chroma-ref-gated).
                _ => (0, 0, 0, 0),
            };
            let winner = LeafWinner {
                bsize,
                mode: best.y.mode,
                angle_delta_y: best.y.angle_delta,
                use_filter_intra: best.y.use_filter_intra,
                filter_intra_mode: best.y.filter_intra_mode,
                tx_size: best.y.tx_size,
                uv_mode,
                angle_delta_uv,
                cfl_alpha_idx,
                cfl_alpha_signs,
                tx_type_map: best.tx_type_map,
                skip_txfm: false,
            };
            (stats, Some(winner))
        }
    }
}

/// `should_do_dry_run_encode_for_current_block` (partition_search.c:5556).
pub fn should_do_dry_run_encode(
    sb_size: usize,
    max_partition_size: usize,
    pc_index: usize,
    bsize: usize,
) -> bool {
    if bsize > max_partition_size {
        return false;
    }
    if pc_index != 3 {
        return true;
    }
    let sub_sb_size = split_subsize(sb_size);
    bsize == max_partition_size && sub_sb_size != max_partition_size
}

/// The saved node context (`av1_save_context`, encodeframe_utils.c:579):
/// per-plane above/left entropy + partition + txfm context slices over the
/// node extent.
struct SavedCtx {
    above_e: [Vec<i8>; 3],
    left_e: [Vec<i8>; 3],
    above_p: Vec<i8>,
    left_p: Vec<i8>,
    above_t: Vec<u8>,
    left_t: Vec<u8>,
}

fn save_context(tile: &TileCtxState, mi_row: i32, mi_col: i32, bsize: usize, ss_x: usize, ss_y: usize) -> SavedCtx {
    let w = MI_SIZE_WIDE_B[bsize];
    let h = MI_SIZE_HIGH_B[bsize];
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;
    SavedCtx {
        above_e: [
            tile.above_ectx[0][a0..a0 + w].to_vec(),
            tile.above_ectx[1][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].to_vec(),
            tile.above_ectx[2][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].to_vec(),
        ],
        left_e: [
            tile.left_ectx[0][l0..l0 + h].to_vec(),
            tile.left_ectx[1][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].to_vec(),
            tile.left_ectx[2][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].to_vec(),
        ],
        above_p: tile.above_pctx[a0..a0 + w].to_vec(),
        left_p: tile.left_pctx[l0..l0 + h].to_vec(),
        above_t: tile.above_tctx[a0..a0 + w].to_vec(),
        left_t: tile.left_tctx[l0..l0 + h].to_vec(),
    }
}

fn restore_context(
    tile: &mut TileCtxState,
    saved: &SavedCtx,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
) {
    let w = MI_SIZE_WIDE_B[bsize];
    let h = MI_SIZE_HIGH_B[bsize];
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;
    tile.above_ectx[0][a0..a0 + w].copy_from_slice(&saved.above_e[0]);
    tile.above_ectx[1][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].copy_from_slice(&saved.above_e[1]);
    tile.above_ectx[2][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].copy_from_slice(&saved.above_e[2]);
    tile.left_ectx[0][l0..l0 + h].copy_from_slice(&saved.left_e[0]);
    tile.left_ectx[1][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].copy_from_slice(&saved.left_e[1]);
    tile.left_ectx[2][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].copy_from_slice(&saved.left_e[2]);
    tile.above_pctx[a0..a0 + w].copy_from_slice(&saved.above_p);
    tile.left_pctx[l0..l0 + h].copy_from_slice(&saved.left_p);
    tile.above_tctx[a0..a0 + w].copy_from_slice(&saved.above_t);
    tile.left_tctx[l0..l0 + h].copy_from_slice(&saved.left_t);
}

/// `av1_rd_pick_partition` with REAL leaves, NONE + SPLIT — see the module
/// docs for the exact C sequence. Returns `(winner tree, best stats,
/// found)`; when found and the dry-run gate passes, the winner subtree has
/// been re-encoded (recons + contexts + mode grid stamped) exactly as the C
/// leaves it for siblings.
#[allow(clippy::too_many_arguments)]
pub fn rd_pick_partition_real(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &mut TileCtxState,
    grid: &mut ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mut best_rdc: PartRdStats,
    pc_index: usize,
    visits: &mut Vec<LeafVisit>,
) -> (Option<SbTree>, PartRdStats, bool) {
    if best_rdc.rdcost < 0 {
        return (None, PartRdStats::invalid(), false);
    }
    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_step = (mi_w / 2) as i32;
    let bsize_at_least_8x8 = bsize >= 3;
    let has_rows = mi_row + mi_step < env.mi_rows;
    let has_cols = mi_col + mi_step < env.mi_cols;
    let mut partition_none_allowed = has_rows && has_cols;
    let mut do_square_split = bsize_at_least_8x8;
    // av1_prune_partitions_by_max_min_bsize (partition_strategy.c:1837).
    const BLK_1D: [usize; 22] =
        [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
    if BLK_1D[bsize] > BLK_1D[cfg.max_partition_size] {
        // Larger than max: square split only.
        partition_none_allowed = false;
        do_square_split = bsize_at_least_8x8;
    } else if BLK_1D[bsize] <= BLK_1D[cfg.min_partition_size] {
        if has_rows && has_cols {
            do_square_split = false;
        }
        partition_none_allowed = !do_square_split;
    }

    // partition_cost[pl_ctx_idx] over the LIVE partition contexts.
    let pl_ctx = if bsize_at_least_8x8 {
        partition_plane_context(
            &tile.above_pctx,
            &tile.left_pctx,
            mi_row as usize,
            mi_col as usize,
            bsize,
        ) as usize
    } else {
        0
    };
    let partition_cost = &cfg.partition_costs[pl_ctx];

    // av1_rd_cost_update(x->rdmult, &best_rdc) (:5744).
    rd_cost_update(env.rdmult, &mut best_rdc);

    // av1_save_context (:5754).
    let saved = save_context(tile, mi_row, mi_col, bsize, env.ss_x, env.ss_y);

    let mut found = false;
    let mut best_tree: Option<SbTree> = None;

    // ---- PARTITION_NONE stage ----
    if partition_none_allowed {
        let mut pt_cost = 0i32;
        if bsize_at_least_8x8 {
            pt_cost = if partition_cost[0] < i32::MAX { partition_cost[0] } else { 0 };
        }
        let mut partition_rdcost = PartRdStats::init();
        partition_rdcost.rate = pt_cost;
        rd_cost_update(env.rdmult, &mut partition_rdcost);
        let best_remain = rd_stats_subtraction(env.rdmult, &best_rdc, &partition_rdcost);

        let (mut this_rdc, winner) = leaf_pick_sb_modes(
            env, cfg, tile, grid, recon_y, recon_u, recon_v, cfl, mi_row, mi_col, bsize,
            &best_remain,
        );
        visits.push(LeafVisit {
            mi_row,
            mi_col,
            bsize,
            budget: best_remain.rdcost,
            rate: this_rdc.rate,
            dist: this_rdc.dist,
            rdcost: this_rdc.rdcost,
        });
        // (pick_sb_modes normalized INT_MAX already; av1_rd_cost_update at
        // the stage is folded into the leaf's returned rdcost.)
        if this_rdc.rate != i32::MAX {
            if bsize_at_least_8x8 {
                this_rdc.rate += pt_cost;
                this_rdc.rdcost = crate::rd::rdcost(env.rdmult, this_rdc.rate, this_rdc.dist);
            }
            if this_rdc.rdcost < best_rdc.rdcost {
                best_rdc = this_rdc;
                found = true;
                best_tree = Some(SbTree::Leaf(winner.expect("valid rate has a winner")));
            }
        }
        // av1_restore_context at the NONE-stage tail (:4492).
        restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    }

    // ---- PARTITION_SPLIT stage ----
    if do_square_split {
        let subsize = split_subsize(bsize);
        let mut sum_rdc = PartRdStats::init();
        sum_rdc.rate = partition_cost[3];
        sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, 0);

        let mut children: Vec<Option<SbTree>> = Vec::new();
        let mut idx = 0usize;
        while idx < 4 && sum_rdc.rdcost < best_rdc.rdcost {
            let y = mi_row + ((idx as i32) >> 1) * mi_step;
            let x = mi_col + ((idx as i32) & 1) * mi_step;
            if y >= env.mi_rows || x >= env.mi_cols {
                children.push(None);
                idx += 1;
                continue;
            }
            let best_remain = rd_stats_subtraction(env.rdmult, &best_rdc, &sum_rdc);
            let (child_tree, child_rdc, child_found) = rd_pick_partition_real(
                env, cfg, tile, grid, recon_y, recon_u, recon_v, cfl, y, x, subsize,
                best_remain, idx, visits,
            );
            if !child_found {
                sum_rdc = PartRdStats::invalid();
                children.push(child_tree);
                break;
            }
            sum_rdc.rate += child_rdc.rate;
            sum_rdc.dist += child_rdc.dist;
            rd_cost_update(env.rdmult, &mut sum_rdc);
            children.push(child_tree);
            idx += 1;
        }
        let reached_last_index = idx == 4;

        if reached_last_index && sum_rdc.rdcost < best_rdc.rdcost {
            // split_partition_penalty_level = 0 => factor 1.0.
            sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, sum_rdc.dist);
            if sum_rdc.rdcost < best_rdc.rdcost {
                best_rdc = sum_rdc;
                found = true;
                let kids: Vec<SbTree> = children
                    .into_iter()
                    .map(|t| t.expect("found split has 4 found children"))
                    .collect();
                best_tree = Some(SbTree::Split(Box::new(
                    <[SbTree; 4]>::try_from(kids).ok().unwrap(),
                )));
            }
        }
        // The SPLIT-stage restore (:4645-4647): gated `bsize <=
        // max_partition_size || bsize == sb_size` — always true here.
        debug_assert!(bsize <= cfg.max_partition_size || bsize == env.sb_size);
        restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    }

    // ---- the winner encode (:5998-6026) ----
    if found {
        let tree = best_tree.as_mut().expect("found implies a tree");
        let do_encode = if bsize == env.sb_size {
            // The C emits OUTPUT_ENABLED at the SB root (pack-stage adds;
            // contexts/pixels identical) — modelled as the same DRY_RUN walk.
            true
        } else {
            should_do_dry_run_encode(env.sb_size, cfg.max_partition_size, pc_index, bsize)
        };
        if do_encode {
            let mut leaves: Vec<LeafEncodeOut> = Vec::new();
            encode_sb_dry(
                env, tile, recon_y, recon_u, recon_v, cfl, tree, mi_row, mi_col, bsize,
                &mut leaves,
            );
            // av1_update_state's mi-grid fill: leaf footprints are disjoint,
            // so stamping from the walk's leaf list is order-equivalent.
            stamp_grid_from_tree(grid, tree, mi_row, mi_col, bsize, env.mi_rows, env.mi_cols);
        }
    }

    if found {
        (best_tree, best_rdc, true)
    } else {
        (None, best_rdc, false)
    }
}

fn stamp_grid_from_tree(
    grid: &mut ModeGrid,
    tree: &SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    match tree {
        SbTree::Leaf(w) => {
            grid.stamp(mi_row, mi_col, bsize, w.mode as u8, mi_rows, mi_cols);
        }
        SbTree::Split(kids) => {
            let sub = split_subsize(bsize);
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            for (idx, child) in kids.iter().enumerate() {
                stamp_grid_from_tree(
                    grid,
                    child,
                    mi_row + ((idx as i32) >> 1) * hbs,
                    mi_col + ((idx as i32) & 1) * hbs,
                    sub,
                    mi_rows,
                    mi_cols,
                );
            }
        }
    }
}
