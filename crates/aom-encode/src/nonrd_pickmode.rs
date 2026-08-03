//! KB-12 — speed >= 8 nonrd pickmode (allintra KEY): `av1_nonrd_pick_intra_mode`
//! (nonrd_pickmode.c:1582) + its `av1_block_yrd` Hadamard estimator
//! (nonrd_opt.c:126) + the LP kernel set they stand on.
//!
//! STATUS (2026-07-17): LANDED — compiled + gated. Speed 9 byte-matches real
//! `aomenc --cpu-used=9` 64/64 (canon) + noise; speed 8 60/64 (canon) + noise,
//! with 4 `diag` estimate-arm V/H near-ties pinned open (KB-12 in CLAUDE.md).
//! Every function carries its exact C provenance; the remaining `// HANDOFF:`
//! marks are genuine out-of-envelope work (lossless TX_4X4, screen-content
//! palette). See CLAUDE.md KB-12 for the full state, gate names, and the pinned
//! near-tie's next step.
//!
//! STATUS (2026-07-30, KB-20): the HBD estimate arm is ported — bd10/bd12 x
//! `--cpu-used` {8,9} are byte-identical to real aomenc (24 cells,
//! `config_permutations::speed_nonrd_hbd_byte_identity`). Before this the arm
//! was a hard `assert!(env.bd == 8)`, so every such encode PANICKED. See the
//! "Bit depth" section below.
//!
//! STATUS (2026-08-02, KB-12 + KB-34): the "leaf-mode near-tie" was
//! `aom_hadamard_lp_8x8`'s dropped trailing transpose and speed 8 is now
//! **64/64** (see [`hadamard_lp_8x8`]); and the arm codes a **NON-SQUARE
//! leaf** — `nonrd_pick_intra_mode` runs C's real
//! `av1_foreach_transformed_block_in_plane` walk instead of one inlined visit,
//! so `--cpu-used 9` no longer refuses the ordinary frames the KEY variance
//! partitioner stamps a HORZ/VERT pair on (a 100x100 thumbnail is enough — the
//! refusal's own "only at 108 MP" measurement was wrong by four orders of
//! magnitude). Gate: `aom-bench/tests/kb34_nonsquare_nonrd_leaf.rs`.
//!
//! ## The chroma answer (the KB-11 flagged unknown — RESOLVED)
//! `av1_nonrd_pick_intra_mode` is Y-only and hard-sets
//! `mi->uv_mode = UV_DC_PRED` (nonrd_pickmode.c:1735, comment "Keep DC for UV
//! since mode test is based on Y channel only"). There is NO uv mode search
//! and NO uv rate/dist estimate on the estimate arm; chroma is coded as DC by
//! the ordinary leaf encode (`encode_superblock` — port: `encode_b_intra_dry`
//! consuming `LeafWinner::uv_mode = 0`). The full-RD arm
//! (`av1_rd_pick_intra_mode_sb` via `hybrid_intra_mode_search`,
//! partition_search.c:755-772) picks uv with the EXISTING ported machinery
//! (`leaf_pick_sb_modes`). Palette/CfL: `init_mbmi_nonrd` (nonrd_opt.h:516)
//! zeroes palette sizes + filter_intra; CfL is never a candidate (uv fixed
//! DC), so `cfl_alpha_* = 0` on estimate leaves.
//!
//! ## Speed-8 allintra sf deltas (speed_features.c:577-590, verified)
//! - `hybrid_intra_pickmode = 2` → full-RD arm for `bsize < BLOCK_16X16 &&
//!   source_variance >= var_thresh[1] = 101` (partition_search.c:762-766;
//!   `var_thresh = {0, 101, 201}`, index `hybrid_intra_pickmode - 1`).
//! - `use_nonrd_pick_mode = 1` → `encode_nonrd_sb` + `av1_nonrd_use_partition`.
//! - `nonrd_check_partition_merge_mode = 1` (framesize-dependent :157-160
//!   raises it to 2 below 480p) — `try_merge` is `!frame_is_intra_only`-gated
//!   (partition_search.c:3089) → INERT on KEY.
//! - `var_part_split_threshold_shift = 8` — read only under
//!   `force_large_partition_blocks_intra`, which needs speed>=8 AND 720p+
//!   (:161-163) → inert on the canon grid (<720p).
//! - `prune_palette_search_nonrd = 1` — palette arm still needs
//!   `enable_palette && av1_allow_palette(allow_screen_content_tools, bsize)`;
//!   canon grid runs `allow_screen_content_tools = 0` → dead (guarded below).
//! - `intra_y_mode_bsize_mask_nrd[...]` = INTRA_DC (>=32x32) / INTRA_DC_H_V —
//!   consumed ONLY by `is_prune_intra_mode` (nonrd_opt.c:570) on the INTER
//!   frame path → INERT on KEY (verified: `av1_nonrd_pick_intra_mode` loops
//!   `intra_mode_list` directly with no mask check).
//!
//! ## Speed-9 allintra sf deltas (speed_features.c:592-607 + :166-177, verified)
//! - `hybrid_intra_pickmode = 0` → the full-RD arm DIES; every leaf uses the
//!   estimate loop below.
//! - `nonrd_check_partition_merge_mode = 0` (still KEY-inert either way).
//! - `var_part_split_threshold_shift = 7` (still force_large-gated → inert
//!   <720p).
//! - `vbp_prune_16x16_split_using_min_max_sub_blk_var = true` → LIVE in the
//!   KEY VBP tree: the 16x16 force-split rule (var_based_part.c:1804-1809)
//!   becomes `get_part_eval_based_on_sub_blk_var(vtemp, thresholds[3])`
//!   (:1530): max/min over the four 8x8 sub-variances; `(max - min) >
//!   (threshold16 << 2)` → ONLY_SPLIT else ONLY_NONE (instead of
//!   unconditional ONLY_SPLIT). Port: thread the existing
//!   `vbp_prune_16x16_split_using_min_max_sub_blk_var` param of
//!   [`crate::var_part::choose_var_based_partitioning_key`] as
//!   `speed >= 9` — the param already exists (passed `false` today) but
//!   HANDOFF: verify var_part.rs implements the ONLY_NONE arm (3-state
//!   PART_EVAL semantics), not just a bool force-split.
//! - `prune_h_pred_using_best_mode_so_far = true` → estimate-loop prune (live).
//! - `enable_intra_mode_pruning_using_neighbors = true` → estimate-loop prune
//!   (live).
//! - `prune_intra_mode_using_best_sad_so_far = true` → per-mode SAD prune in
//!   `av1_estimate_block_intra` (live; `bsize == tx_bsize` always holds for
//!   our square single-txb leaves).
//! - `coeff_cost_upd_level = mode_cost_upd_level = INTERNAL_COST_UPD_SBROW`
//!   (framesize-independent :593-594) then **INTERNAL_COST_UPD_OFF for <4k**
//!   (framesize-DEPENDENT :166-177, runs later and wins) → on the whole canon
//!   grid the per-SB `derive_real_costs` refresh STOPS at speed 9: every SB
//!   uses the FRAME-INIT cost tables (visible on 128² cells = 4 SBs;
//!   64² = 1 SB is inert). Port: gate the `derive_real_costs` call in
//!   `pack::pack_tile` on `speed <= 8`.
//! - SB size: `av1_select_sb_size` (encoder_utils.c:958) already returns
//!   BLOCK_64X64 for `speed >= 1 && <= 480p` — the speed-9 allintra <4k rule
//!   (:1035-1037) adds nothing on the canon grid.
//!
//! ## Structural notes (encode side)
//! - `pick_sb_modes_nonrd` (partition_search.c:2254): recomputes
//!   `x->source_variance = av1_get_perpixel_variance_facade(bsize)` per leaf
//!   whenever `bsize < sb_size` OR it is UINT_MAX (:2306-2311); the SB-level
//!   value `choose_var_based_partitioning` computes (var_based_part.c:1724-1731,
//!   gated `use_nonrd_pick_mode && source_sad_nonrd > kLowSad`; KEY inits
//!   source_sad_nonrd = kMedSad, encodeframe.c:1289) is the SAME
//!   perpixel-variance of the same pixels → per-leaf recompute is exact.
//!   `get_force_zeromv_skip_flag_for_blk` (:2182) returns
//!   `force_zeromv_skip_for_sb` when < 2 — 0 on KEY → no gating.
//! - `encode_b_nonrd` (partition_search.c:2089): set_offsets_without_segment_id
//!   → `setup_block_rdmult(.., NO_AQ, NULL)` (identity here: aq NONE +
//!   VBP leaves the ALLINTRA sb modifier at 128, the KB-11 fact) →
//!   `av1_update_state` → `if (!is_inter_block) mi->skip_txfm = 0` →
//!   `encode_superblock(dry_run=0)` → cb_offsets/update_stats. In this port's
//!   split architecture the recon+context walk is `encode_b_intra_dry` and the
//!   bit-writing is `pack_sb` over the finished tree — the SAME split already
//!   proven byte-exact for speeds 0-7 (the symbol stream determines the CDF
//!   adaptation, and the nonrd walk's symbol stream is the same tree replay).
//! - rd costs of the walk are DECISION-INERT (`dummy_cost` is invalid and
//!   never compared — av1_nonrd_use_partition:2983); only the per-leaf
//!   estimate numerics decide `best_mode`, which is why `av1_block_yrd` must
//!   be bit-exact.
//!
//! ## Bit depth (KB-20 — the hbd arm, PORTED 2026-07-30)
//! `av1_block_yrd` branches on `use_hbd = is_cur_buf_hbd(xd)` (blockd.h — the
//! buffer's `YV12_FLAG_HIGHBITDEPTH`, which this project's encode path sets
//! exactly when `bd > 8`; `dec_shim.c` passes
//! `bd > 8 ? AOM_CODEC_USE_HIGHBITDEPTH : 0`):
//! - **lowbd** ([`block_yrd_lowbd`]): `aom_hadamard_lp_8x8/16x16`,
//!   `aom_fdct4x4_lp`, `av1_quantize_lp`, `aom_satd_lp`, `av1_block_error_lp`,
//!   over the `*_lp_*_transpose` scans (i16 throughout). Every one of these is
//!   locked against the exported C symbol by
//!   `tests/nonrd_block_yrd_lp_diff.rs` — added 2026-08-02, because its
//!   absence WAS KB-12 (see [`hadamard_lp_8x8`]).
//! - **hbd** ([`block_yrd_hbd`]): `aom_hadamard_8x8/16x16`, `aom_fdct4x4`,
//!   `av1_quantize_fp`, `aom_satd`, `av1_highbd_block_error`, over
//!   `default_scan_8x8_transpose`/`av1_default_iscan_8x8_transpose` and
//!   `default_scan_fp_16x16_transpose`/`av1_default_iscan_fp_16x16_transpose`
//!   (`tran_low_t` = i32 throughout; nonrd_opt.c:199-262). Note the 16x16 lp
//!   and fp Hadamards produce DIFFERENT coefficient orders (the fp one carries
//!   `aom_hadamard_16x16_c`'s extra AVX2-matching column shift), which is why
//!   they need separate scan tables.
//!
//! `is_tx_8x8_dual_applicable` is force-zeroed for hbd (nonrd_opt.c:176) and is
//! unreachable from the intra estimate arm anyway (see
//! [`hadamard_lp_8x8_dual`]), so the two arms visit the same txb grid.
//!
//! The old handoff assert named only "av1_quantize_fp + fp scans". THREE things
//! were actually bd8-specific, and the two it did not name are the interesting
//! ones:
//!
//! 2. the speed-9 SAD prune in `av1_estimate_block_intra` (nonrd_opt.c:629)
//!    calls `cpi->ppi->fn_ptr[bsize].sdf`, which `highbd_set_var_fns`
//!    (encoder_utils.h:158 `MAKE_BFP_SAD_WRAPPER`) binds to
//!    `aom_highbd_sadWxH_bits{8,10,12}` — the raw SAD `>> (bd - 8)`. See
//!    [`nonrd_pick_intra_mode`];
//! 3. **`av1_quantize_fp` is ISA-conditional in this regime.** Every SIMD tier
//!    is a 16-bit kernel; on `_c`/NEON `aom_hadamard_16x16` reaches +-65534, so
//!    the tiers stop agreeing with `av1_quantize_fp_c` — and with each other —
//!    exactly here. [`quantize_fp_dispatched`] carries the measurement and the
//!    model;
//! 4. **`aom_hadamard_16x16` is ISA-conditional too, and it comes FIRST.** Its
//!    4-way combine is `int32` in `_c` and NEON but `int16` (wrapping) in AVX2
//!    and SSE2. At bd8 the documented range fits `int16` and every tier agrees;
//!    at bd10/bd12 the x86 tiers wrap where `_c`/NEON do not, which changes the
//!    coefficients BEFORE quantization, satd and block-error ever see them.
//!    [`hadamard_16x16_dispatched`] carries that model. (Found by the first x86
//!    CI run of the KB-20 gate: the two `quantize_fp_dispatched` unit teeth
//!    passed there while the byte gate failed, which localises the divergence
//!    upstream of the quantizer.) `aom_hadamard_8x8`, `aom_satd` and
//!    `av1_highbd_block_error` were checked the same way and are **not**
//!    ISA-conditional: the 8x8 Hadamard is `int16_t` in every tier, and satd /
//!    block-error are 32-/64-bit in every tier.
//!
//! Everything else on the arm was already bd-parameterised (predict, subtract,
//! the cost tables, `rdmult`), which is the same shape the deltaq-mode-3
//! landing found for `av1_set_mb_wiener_variance` (PARITY.md section A).
//!
//! Still out of envelope at every bit depth: lossless TX_4X4 (both arms) and
//! the screen-content palette arm — which is now the ONLY refusal a bd8
//! `--cpu-used` 8/9 encode can hit, and it accounts for all 136 refusing rows
//! of KB-34's 1,264-cell sweep (libaom's own screen-content detection firing on
//! a smooth synthetic gradient and on low-quantizer real content).

use crate::encode_sb::SbEncodeEnv;
use crate::partition::PartRdStats;
use aom_dsp::dist::highbd_subtract_block;
use aom_dsp::intra::predict_intra_high;

/// `MI_SIZE_WIDE`/`HIGH` for the square sizes used here (port-wide numbering:
/// BLOCK_8X8=3, BLOCK_16X16=6, BLOCK_32X32=9, BLOCK_64X64=12).
const MI_W: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// `intra_mode_context[]` (av1_common_int.h) — KF y-mode cost context per
/// neighbour PREDICTION_MODE. HANDOFF: dedupe with the copy the full-RD leaf
/// uses (intra_rd.rs derives above_ctx/left_ctx somewhere — same table).
const INTRA_MODE_CONTEXT: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `intra_mode_list[]` (nonrd_opt.h:121): DC, V, H, SMOOTH.
const INTRA_MODE_LIST: [usize; 4] = [0, 1, 2, 9];

/// `AV1_PROB_COST_SHIFT` (av1/encoder/cost.h).
const AV1_PROB_COST_SHIFT: i32 = 9;

// ---------------------------------------------------------------------------
// LP kernels (aom_dsp/avg.c, aom_dsp/fwd_txfm.c, av1/encoder/av1_quantize.c,
// av1/encoder/rdopt.c) — C-scalar ports, all wrapping-i16 where C wraps.
// ---------------------------------------------------------------------------

/// `get_msb` (aom_dsp/bitops.h): index of the highest set bit; UB at 0 in C,
/// callers always pass `n >= 1`.
#[inline]
fn get_msb(n: u32) -> i32 {
    debug_assert!(n != 0);
    31 - n.leading_zeros() as i32
}

/// `hadamard_col8` (aom_dsp/avg.c:149). C does the arithmetic in int16_t —
/// intermediate sums are allowed to wrap (the dynamic-range comments bound
/// REAL inputs away from wrap, but bit-exactness demands wrapping semantics).
#[inline]
fn hadamard_col8(src: &[i16], stride: usize, coeff: &mut [i16; 8]) {
    let b0 = src[0].wrapping_add(src[stride]);
    let b1 = src[0].wrapping_sub(src[stride]);
    let b2 = src[2 * stride].wrapping_add(src[3 * stride]);
    let b3 = src[2 * stride].wrapping_sub(src[3 * stride]);
    let b4 = src[4 * stride].wrapping_add(src[5 * stride]);
    let b5 = src[4 * stride].wrapping_sub(src[5 * stride]);
    let b6 = src[6 * stride].wrapping_add(src[7 * stride]);
    let b7 = src[6 * stride].wrapping_sub(src[7 * stride]);

    let c0 = b0.wrapping_add(b2);
    let c1 = b1.wrapping_add(b3);
    let c2 = b0.wrapping_sub(b2);
    let c3 = b1.wrapping_sub(b3);
    let c4 = b4.wrapping_add(b6);
    let c5 = b5.wrapping_add(b7);
    let c6 = b4.wrapping_sub(b6);
    let c7 = b5.wrapping_sub(b7);

    coeff[0] = c0.wrapping_add(c4);
    coeff[7] = c1.wrapping_add(c5);
    coeff[3] = c2.wrapping_add(c6);
    coeff[4] = c3.wrapping_add(c7);
    coeff[2] = c0.wrapping_sub(c4);
    coeff[6] = c1.wrapping_sub(c5);
    coeff[1] = c2.wrapping_sub(c6);
    coeff[5] = c3.wrapping_sub(c7);
}

/// `aom_hadamard_lp_8x8_c` (aom_dsp/avg.c:209): 8x8 2D Hadamard, int16 out.
///
/// **The trailing transpose is part of the kernel** (avg.c:232-236, *"Extra
/// transpose to match SSE2 behavior (i.e., aom_hadamard_lp_8x8_sse2)"*): C
/// writes `coeff[i * 8 + j] = buffer2[j * 8 + i]`, so the exported coefficient
/// order is `buffer2` TRANSPOSED, not `buffer2`. Omitting it (KB-12's root)
/// leaves every order-invariant quantity — `aom_satd_lp`,
/// `av1_block_error_lp`, `eob == 0` — correct while moving `eob` itself, which
/// is the only scan-order-sensitive output of the estimate arm.
///
/// `_c` and `_neon` agree here (verified in
/// `tests/nonrd_block_yrd_lp_diff.rs`), so unlike `aom_hadamard_16x16`
/// (LIBAOM_UPSTREAM_NOTES A1 / KB-20 root #4) there is nothing
/// ISA-conditional to model.
pub fn hadamard_lp_8x8(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    let mut buffer = [0i16; 64];
    let mut buffer2 = [0i16; 64];
    for idx in 0..8 {
        let mut col = [0i16; 8];
        hadamard_col8(&src_diff[idx..], src_stride, &mut col);
        buffer[idx * 8..idx * 8 + 8].copy_from_slice(&col);
    }
    for idx in 0..8 {
        let mut col = [0i16; 8];
        hadamard_col8(&buffer[idx..], 8, &mut col);
        buffer2[idx * 8..idx * 8 + 8].copy_from_slice(&col);
    }
    // avg.c:232-236 — the extra transpose.
    for i in 0..8 {
        for j in 0..8 {
            coeff[i * 8 + j] = buffer2[j * 8 + i];
        }
    }
}

/// `aom_hadamard_lp_8x8_dual_c` (avg.c:240): two adjacent 8x8s. UNREACHABLE
/// from the intra estimate arm: `is_tx_8x8_dual_applicable` (nonrd_opt.c:167)
/// wants `tx_size == TX_8X8 && block_size_wide[bsize] >= 16`, and
/// `av1_block_yrd` is always called from `av1_estimate_block_intra` with
/// `bsize_tx = txsize_to_bsize[tx_size]` (nonrd_opt.c:658) — the TX block's own
/// size, 8 px wide whenever the tx is TX_8X8, at a NON-square leaf as much as a
/// square one. Kept for completeness / the inter path.
pub fn hadamard_lp_8x8_dual(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    for i in 0..2 {
        hadamard_lp_8x8(&src_diff[i * 8..], src_stride, &mut coeff[i * 64..]);
    }
}

/// `aom_hadamard_lp_16x16_c` (avg.c:291): four 8x8 stages + a cross-combine
/// with `>> 1` normalization. int16 wrapping.
///
/// No trailing transpose of its own — it inherits [`hadamard_lp_8x8`]'s, per
/// 64-coefficient quadrant, and the elementwise 4-way combine below commutes
/// with that. (Nor does it carry `aom_hadamard_16x16_c`'s extra AVX2-matching
/// column shift; that difference is why the lp and fp 16x16 scans are distinct
/// tables.)
///
/// **Note on the shift order, measured-inert rather than assumed.** C writes
/// `int16_t b0 = (a0 + a1) >> 1`, where the sum promotes to `int` and only the
/// *result* narrows — shift-then-truncate. `_mm_srai_epi16` in
/// `aom_hadamard_lp_16x16_sse2` (avg_intrin_sse2.c:442) and its AVX2 twin
/// truncate first, which is what the `wrapping_add(..) >> 1` below spells. The
/// two agree unless `|a0 + a1|` exceeds `i16::MAX`, and on this call site it
/// cannot: `block_yrd_lowbd` runs only at bd8, where the residual is 9-bit by
/// construction (`src - pred`, both u8) and the 8x8 stage peaks at
/// `255 * 64 = 16320`, so `|a0 + a1| <= 32640 < 32768`.
/// `nonrd_block_yrd_lp_diff::lp_hadamard_tiers_agree_over_the_reachable_range`
/// asserts both halves of that: the tiers agree, and the grid really does
/// drive `|coeff|` above 16000.
pub fn hadamard_lp_16x16(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    for idx in 0..4 {
        let src_off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        hadamard_lp_8x8(&src_diff[src_off..], src_stride, &mut coeff[idx * 64..]);
    }
    for idx in 0..64 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 64];
        let a2 = coeff[idx + 128];
        let a3 = coeff[idx + 192];

        let b0 = a0.wrapping_add(a1) >> 1;
        let b1 = a0.wrapping_sub(a1) >> 1;
        let b2 = a2.wrapping_add(a3) >> 1;
        let b3 = a2.wrapping_sub(a3) >> 1;

        coeff[idx] = b0.wrapping_add(b2);
        coeff[idx + 64] = b1.wrapping_add(b3);
        coeff[idx + 128] = b0.wrapping_sub(b2);
        coeff[idx + 192] = b1.wrapping_sub(b3);
    }
}

/// `aom_fdct4x4_lp_c` (aom_dsp/fwd_txfm.c:85). Reachable only at lossless
/// (TX_4X4) — outside the canon envelope, ported for completeness.
pub fn fdct4x4_lp(input: &[i16], output: &mut [i16], stride: usize) {
    // cospi constants (aom_dsp/txfm_common.h).
    const COSPI_16_64: i32 = 11585;
    const COSPI_24_64: i32 = 6270;
    const COSPI_8_64: i32 = 15137;
    const DCT_CONST_BITS: i32 = 14;
    #[inline]
    fn fdct_round_shift(v: i32) -> i32 {
        (v + (1 << (DCT_CONST_BITS - 1))) >> DCT_CONST_BITS
    }
    let mut intermediate = [0i16; 16];
    for pass in 0..2 {
        for i in 0..4 {
            let mut in_high = [0i32; 4];
            if pass == 0 {
                in_high[0] = i32::from(input[i]) * 16;
                in_high[1] = i32::from(input[stride + i]) * 16;
                in_high[2] = i32::from(input[2 * stride + i]) * 16;
                in_high[3] = i32::from(input[3 * stride + i]) * 16;
                if i == 0 && in_high[0] != 0 {
                    in_high[0] += 1;
                }
            } else {
                in_high[0] = i32::from(intermediate[i]);
                in_high[1] = i32::from(intermediate[4 + i]);
                in_high[2] = i32::from(intermediate[8 + i]);
                in_high[3] = i32::from(intermediate[12 + i]);
            }
            let step0 = in_high[0] + in_high[3];
            let step1 = in_high[1] + in_high[2];
            let step2 = in_high[1] - in_high[2];
            let step3 = in_high[0] - in_high[3];
            let t0 = fdct_round_shift((step0 + step1) * COSPI_16_64) as i16;
            let t2 = fdct_round_shift((step0 - step1) * COSPI_16_64) as i16;
            let t1 = fdct_round_shift(step2 * COSPI_24_64 + step3 * COSPI_8_64) as i16;
            let t3 = fdct_round_shift(-step2 * COSPI_8_64 + step3 * COSPI_24_64) as i16;
            if pass == 0 {
                intermediate[i * 4] = t0;
                intermediate[i * 4 + 1] = t1;
                intermediate[i * 4 + 2] = t2;
                intermediate[i * 4 + 3] = t3;
            } else {
                output[i] = t0;
                output[4 + i] = t1;
                output[8 + i] = t2;
                output[12 + i] = t3;
            }
        }
    }
    // C post-pass: output[j] = (output[j] + 1) >> 2 (fwd_txfm.c:150-ish).
    // HANDOFF: verify the final rounding loop of aom_fdct4x4_lp_c —
    // read past line 145; the fdct4x4 (non-lp) does
    // `(out + 1) >> 2`; confirm the lp variant matches before ANY lossless
    // speed-8 use. (Unreachable on the canon grid.)
    for v in output[..16].iter_mut() {
        *v = (*v + 1) >> 2;
    }
}

/// `av1_quantize_lp_c` (av1/encoder/av1_quantize.c:214): the low-precision FP
/// quantizer. `scan` orders the eob computation; qcoeff/dqcoeff are written at
/// RAW (`rc`) positions. round/quant/dequant use row lane `[rc != 0]`.
#[allow(clippy::too_many_arguments)]
pub fn quantize_lp(
    coeff: &[i16],
    n_coeffs: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 8],
    scan: &[i16],
) -> u16 {
    let mut eob: i32 = -1;
    qcoeff[..n_coeffs].fill(0);
    dqcoeff[..n_coeffs].fill(0);
    for (i, &sc) in scan[..n_coeffs].iter().enumerate() {
        let rc = sc as usize;
        let c = i32::from(coeff[rc]);
        let coeff_sign = c >> 31; // AOMSIGN
        let abs_coeff = (c ^ coeff_sign) - coeff_sign;
        let lane = usize::from(rc != 0);
        let mut tmp =
            (abs_coeff + i32::from(round_fp[lane])).clamp(i16::MIN as i32, i16::MAX as i32);
        tmp = (tmp * i32::from(quant_fp[lane])) >> 16;
        qcoeff[rc] = ((tmp ^ coeff_sign) - coeff_sign) as i16;
        dqcoeff[rc] = qcoeff[rc].wrapping_mul(dequant[lane]);
        if tmp != 0 {
            eob = i as i32;
        }
    }
    (eob + 1) as u16
}

/// `aom_satd_lp_c` (avg.c:520).
pub fn satd_lp(coeff: &[i16], length: usize) -> i32 {
    coeff[..length].iter().map(|&c| i32::from(c).abs()).sum()
}

/// `av1_block_error_lp_c` (rdopt.c:907).
pub fn block_error_lp(coeff: &[i16], dqcoeff: &[i16], block_size: usize) -> i64 {
    let mut error: i64 = 0;
    for i in 0..block_size {
        let diff = i64::from(coeff[i]) - i64::from(dqcoeff[i]);
        error += diff * diff;
    }
    error
}

// ---------------------------------------------------------------------------
// Transposed scan orders (nonrd_opt.h:212-300) — used ONLY with the lp
// Hadamard outputs (whose coefficient order is the C transposed layout).
// ---------------------------------------------------------------------------

/// `default_scan_8x8_transpose` (nonrd_opt.h:212).
pub const DEFAULT_SCAN_8X8_TRANSPOSE: [i16; 64] = [
    0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34,
    41, 48, 56, 49, 42, 35, 28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 31,
    38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
];

/// `default_scan_lp_16x16_transpose` (nonrd_opt.h:238).
pub const DEFAULT_SCAN_LP_16X16_TRANSPOSE: [i16; 256] = [
    0, 8, 2, 4, 10, 16, 24, 18, 12, 6, 64, 14, 20, 26, 32, 40, 34, 28, 22, 72, 66, 68, 74, 80, 30,
    36, 42, 48, 56, 50, 44, 38, 88, 82, 76, 70, 128, 78, 84, 90, 96, 46, 52, 58, 1, 9, 3, 60, 54,
    104, 98, 92, 86, 136, 130, 132, 138, 144, 94, 100, 106, 112, 62, 5, 11, 17, 25, 19, 13, 7, 120,
    114, 108, 102, 152, 146, 140, 134, 192, 142, 148, 154, 160, 110, 116, 122, 65, 15, 21, 27, 33,
    41, 35, 29, 23, 73, 67, 124, 118, 168, 162, 156, 150, 200, 194, 196, 202, 208, 158, 164, 170,
    176, 126, 69, 75, 81, 31, 37, 43, 49, 57, 51, 45, 39, 89, 83, 77, 71, 184, 178, 172, 166, 216,
    210, 204, 198, 206, 212, 218, 224, 174, 180, 186, 129, 79, 85, 91, 97, 47, 53, 59, 61, 55, 105,
    99, 93, 87, 137, 131, 188, 182, 232, 226, 220, 214, 222, 228, 234, 240, 190, 133, 139, 145, 95,
    101, 107, 113, 63, 121, 115, 109, 103, 153, 147, 141, 135, 248, 242, 236, 230, 238, 244, 250,
    193, 143, 149, 155, 161, 111, 117, 123, 125, 119, 169, 163, 157, 151, 201, 195, 252, 246, 254,
    197, 203, 209, 159, 165, 171, 177, 127, 185, 179, 173, 167, 217, 211, 205, 199, 207, 213, 219,
    225, 175, 181, 187, 189, 183, 233, 227, 221, 215, 223, 229, 235, 241, 191, 249, 243, 237, 231,
    239, 245, 251, 253, 247, 255,
];

// NOTE: `av1_quantize_lp_c` ignores its iscan argument entirely ((void)iscan,
// av1_quantize.c:219), so the lowbd arm above needs only the forward scans.
// The fp (hbd) quantizer DOES consume iscan (the SIMD tiers derive the EOB from
// it while walking raster order), so the hbd arm below carries both halves of
// each pair — and they MUST be used together, exactly as the C comments at
// nonrd_opt.h:225/292/321 require.

/// `av1_default_iscan_8x8_transpose` (nonrd_opt.h:227) — the inverse of
/// [`DEFAULT_SCAN_8X8_TRANSPOSE`].
pub const AV1_DEFAULT_ISCAN_8X8_TRANSPOSE: [i16; 64] = [
    0, 2, 3, 9, 10, 20, 21, 35, 1, 4, 8, 11, 19, 22, 34, 36, 5, 7, 12, 18, 23, 33, 37, 48, 6, 13,
    17, 24, 32, 38, 47, 49, 14, 16, 25, 31, 39, 46, 50, 57, 15, 26, 30, 40, 45, 51, 56, 58, 27, 29,
    41, 44, 52, 55, 59, 62, 28, 42, 43, 53, 54, 60, 61, 63,
];

/// `default_scan_fp_16x16_transpose` (nonrd_opt.h:265) — the 16x16 scan for the
/// **fp** (hbd) Hadamard, whose output order differs from the lp one by
/// `aom_hadamard_16x16_c`'s extra AVX2-matching column shift.
pub const DEFAULT_SCAN_FP_16X16_TRANSPOSE: [i16; 256] = [
    0, 4, 2, 8, 6, 16, 20, 18, 12, 10, 64, 14, 24, 22, 32, 36, 34, 28, 26, 68, 66, 72, 70, 80, 30,
    40, 38, 48, 52, 50, 44, 42, 84, 82, 76, 74, 128, 78, 88, 86, 96, 46, 56, 54, 1, 5, 3, 60, 58,
    100, 98, 92, 90, 132, 130, 136, 134, 144, 94, 104, 102, 112, 62, 9, 7, 17, 21, 19, 13, 11, 116,
    114, 108, 106, 148, 146, 140, 138, 192, 142, 152, 150, 160, 110, 120, 118, 65, 15, 25, 23, 33,
    37, 35, 29, 27, 69, 67, 124, 122, 164, 162, 156, 154, 196, 194, 200, 198, 208, 158, 168, 166,
    176, 126, 73, 71, 81, 31, 41, 39, 49, 53, 51, 45, 43, 85, 83, 77, 75, 180, 178, 172, 170, 212,
    210, 204, 202, 206, 216, 214, 224, 174, 184, 182, 129, 79, 89, 87, 97, 47, 57, 55, 61, 59, 101,
    99, 93, 91, 133, 131, 188, 186, 228, 226, 220, 218, 222, 232, 230, 240, 190, 137, 135, 145, 95,
    105, 103, 113, 63, 117, 115, 109, 107, 149, 147, 141, 139, 244, 242, 236, 234, 238, 248, 246,
    193, 143, 153, 151, 161, 111, 121, 119, 125, 123, 165, 163, 157, 155, 197, 195, 252, 250, 254,
    201, 199, 209, 159, 169, 167, 177, 127, 181, 179, 173, 171, 213, 211, 205, 203, 207, 217, 215,
    225, 175, 185, 183, 189, 187, 229, 227, 221, 219, 223, 233, 231, 241, 191, 245, 243, 237, 235,
    239, 249, 247, 253, 251, 255,
];

/// `av1_default_iscan_fp_16x16_transpose` (nonrd_opt.h:323) — the inverse of
/// [`DEFAULT_SCAN_FP_16X16_TRANSPOSE`].
pub const AV1_DEFAULT_ISCAN_FP_16X16_TRANSPOSE: [i16; 256] = [
    0, 44, 2, 46, 1, 45, 4, 64, 3, 63, 9, 69, 8, 68, 11, 87, 5, 65, 7, 67, 6, 66, 13, 89, 12, 88,
    18, 94, 17, 93, 24, 116, 14, 90, 16, 92, 15, 91, 26, 118, 25, 117, 31, 123, 30, 122, 41, 148,
    27, 119, 29, 121, 28, 120, 43, 150, 42, 149, 48, 152, 47, 151, 62, 177, 10, 86, 20, 96, 19, 95,
    22, 114, 21, 113, 35, 127, 34, 126, 37, 144, 23, 115, 33, 125, 32, 124, 39, 146, 38, 145, 52,
    156, 51, 155, 58, 173, 40, 147, 50, 154, 49, 153, 60, 175, 59, 174, 73, 181, 72, 180, 83, 198,
    61, 176, 71, 179, 70, 178, 85, 200, 84, 199, 98, 202, 97, 201, 112, 219, 36, 143, 54, 158, 53,
    157, 56, 171, 55, 170, 77, 185, 76, 184, 79, 194, 57, 172, 75, 183, 74, 182, 81, 196, 80, 195,
    102, 206, 101, 205, 108, 215, 82, 197, 100, 204, 99, 203, 110, 217, 109, 216, 131, 223, 130,
    222, 140, 232, 111, 218, 129, 221, 128, 220, 142, 234, 141, 233, 160, 236, 159, 235, 169, 245,
    78, 193, 104, 208, 103, 207, 106, 213, 105, 212, 135, 227, 134, 226, 136, 228, 107, 214, 133,
    225, 132, 224, 138, 230, 137, 229, 164, 240, 163, 239, 165, 241, 139, 231, 162, 238, 161, 237,
    167, 243, 166, 242, 189, 249, 188, 248, 190, 250, 168, 244, 187, 247, 186, 246, 192, 252, 191,
    251, 210, 254, 209, 253, 211, 255,
];

// ---------------------------------------------------------------------------
// av1_block_yrd (nonrd_opt.c:126) — lowbd arm.
// ---------------------------------------------------------------------------

/// One txb's Hadamard-estimate RD, `av1_block_yrd` with `use_hbd == 0`.
///
/// `diff` is the residual for the WHOLE txb (`bsize_tx`), stride `bw` (4 *
/// mi-width of the txb bsize) — the caller has already run
/// `aom_subtract_block` (here: [`highbd_subtract_block`] on the 8-bit-valued
/// u16 planes, identical arithmetic).
///
/// `tx_size` is the CLAMPED loop size (`AOMMIN(mi->tx_size, TX_16X16)`,
/// nonrd_opt.c:660): 0=4x4, 1=8x8, 2=16x16 sub-blocks over the txb.
/// Returns `(rate, dist, skippable)` where rate is the pre-shift SATD
/// accumulation already folded per C (`rate <<= 2 + AV1_PROB_COST_SHIFT;
/// rate += eob_cost << AV1_PROB_COST_SHIFT`).
///
/// `max_blocks_wide/high`: the C edge clamps (`num_4x4 + (mb_to_edge >> 5)`
/// when negative) — pass the full mi counts for interior leaves.
#[allow(clippy::too_many_arguments)]
pub fn block_yrd_lowbd(
    diff: &[i16],
    bw4: usize, // num_4x4_w of the txb bsize (diff stride = 4 * bw4)
    bh4: usize, // num_4x4_h
    max_blocks_wide: usize,
    max_blocks_high: usize,
    tx_size: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
) -> (i32, i64, bool) {
    debug_assert!(tx_size <= 2, "clamped to <= TX_16X16 (nonrd_opt.c:660)");
    let diff_stride = 4 * bw4;
    let block_step = 1usize << tx_size;
    let step = 1usize << (tx_size << 1); // 4x4 units per sub-block
    let _ = bh4;

    let mut rate: i32 = 0;
    let mut dist: i64 = 0;
    let mut eob_cost: i32 = 0;
    let mut temp_skippable = true;

    let mut coeff = [0i16; 256];
    let mut qcoeff = [0i16; 256];
    let mut dqcoeff = [0i16; 256];

    let mut r = 0usize;
    while r < max_blocks_high {
        let mut c = 0usize;
        while c < max_blocks_wide {
            let src_diff = &diff[(r * diff_stride + c) * 4..];
            let eob: u16 = match tx_size {
                2 => {
                    hadamard_lp_16x16(src_diff, diff_stride, &mut coeff);
                    quantize_lp(
                        &coeff,
                        256,
                        round_fp,
                        quant_fp,
                        &mut qcoeff,
                        &mut dqcoeff,
                        dequant,
                        &DEFAULT_SCAN_LP_16X16_TRANSPOSE,
                    )
                }
                1 => {
                    hadamard_lp_8x8(src_diff, diff_stride, &mut coeff);
                    quantize_lp(
                        &coeff,
                        64,
                        round_fp,
                        quant_fp,
                        &mut qcoeff,
                        &mut dqcoeff,
                        dequant,
                        &DEFAULT_SCAN_8X8_TRANSPOSE,
                    )
                }
                _ => {
                    // TX_4X4: aom_fdct4x4_lp + the NORMAL default 4x4 scan
                    // (av1_scan_orders[TX_4X4][DCT_DCT] — no transpose,
                    // nonrd_opt.c:252 comment). Lossless-only.
                    // HANDOFF: wire av1_scan_orders[TX_4X4][DCT_DCT].scan from
                    // aom-entropy's scan tables (default_scan_4x4) if a
                    // lossless speed-8 envelope ever opens.
                    unimplemented!("TX_4X4 block_yrd (lossless) — out of canon envelope")
                }
            };
            // update_yrd_loop_vars (nonrd_opt.c:43).
            let ncoeffs = eob as usize;
            let is_txfm_skip = ncoeffs == 0;
            temp_skippable &= is_txfm_skip;
            // x->txfm_search_info.blk_skip[r * num_blk_skip_w + c] write:
            // decision-inert for KEY intra (consumed by the inter var-tx
            // path only) — not modelled. HANDOFF: verify nothing on the
            // allintra pack path reads blk_skip (speeds 0-7 never set it
            // from here; the full-RD arm has its own).
            eob_cost += get_msb(ncoeffs as u32 + 1);
            if ncoeffs == 1 {
                rate += i32::from(qcoeff[0]).abs();
            } else if ncoeffs > 1 {
                rate += satd_lp(&qcoeff, step << 4);
            }
            dist += block_error_lp(&coeff, &dqcoeff, step << 4) >> 2;
            c += block_step;
        }
        r += block_step;
    }

    // (nonrd_opt.c:322-336): this_rdc->sse is INT64_MAX from the caller's
    // av1_invalid_rd_stats → the `sse < INT64_MAX` skippable-dist arm never
    // fires on the intra estimate path; rate gets the final shifts.
    let rate = (rate << (2 + AV1_PROB_COST_SHIFT)) + (eob_cost << AV1_PROB_COST_SHIFT);
    (rate, dist, temp_skippable)
}

// ---------------------------------------------------------------------------
// aom_hadamard_16x16 AS DISPATCHED — the ISA-conditional kernel (KB-20 root #4,
// found 2026-07-30 by the FIRST x86 CI run of the KB-20 gate)
// ---------------------------------------------------------------------------

/// `aom_hadamard_16x16` **as the reference build actually dispatches it** —
/// deliberately NOT always `aom_hadamard_16x16_c`.
///
/// ## Why this exists (KB-20 root #4)
/// The 8x8 stage is `int16_t` in EVERY tier (`hadamard_col8`, `aom/dsp/avg.c:149`;
/// `hadamard_8x8_sse2`; `hadamard8x8_one_pass` NEON), so all tiers agree there.
/// The 16x16 **4-way combine** is where they split, and the split is by ISA:
///
/// * `aom_hadamard_16x16_c` (`aom_dsp/avg.c:249`) combines in `tran_low_t`
///   (**int32**): `b0 = (a0 + a1) >> 1` then `coeff[0] = b0 + b2`, no narrowing;
/// * `aom_hadamard_16x16_neon` (`aom_dsp/arm/hadamard_neon.c:188`) combines in
///   `int32x4_t` — `vhaddq_s32` / `vaddq_s32` — i.e. **identical to `_c`**;
/// * `aom_hadamard_16x16_avx2` (`aom_dsp/x86/avg_intrin_avx2.c:144`) and
///   `aom_hadamard_16x16_sse2` (`aom_dsp/x86/avg_intrin_sse2.c:442`) combine in
///   **int16** — `_mm256_add_epi16` / `_mm_add_epi16` then `_mm{,256}_srai_epi16`
///   — so the sum **WRAPS**, and only the final `store_tran_low` sign-extends
///   the wrapped `int16` back out to `tran_low_t`.
///
/// libaom's own comments bound the input at `src_diff` 9-bit ([-255, 255]) and
/// the output at "16 bit, [-32640, 32640]", which fits `int16` — so at **bd8**
/// every tier agrees and the divergence is invisible. The **hbd** nonrd estimate
/// feeds an 11-/13-bit residual instead: the 8x8 stage already wraps in `int16`
/// (all tiers, identically), and the combine then reaches **±65534**, where the
/// x86 tiers wrap and `_c`/NEON do not.
///
/// The output ORDER is the same in all four (C's trailing "extra shift to match
/// AVX2" loop reproduces the lane permutation AVX2's `store_tran_low` performs
/// for free, and SSE2 gets there via `store_tran_low_offset_4`), so only the
/// arithmetic width is ISA-conditional.
///
/// ## Why the x86 arm is tier-INDEPENDENT (unlike [`quantize_fp_dispatched`])
/// AVX2 and SSE2 make the same `int16` choice here, and SSE2 is baseline on
/// x86-64 — so on x86-64 this model holds whichever tier RTCD picks. (A 32-bit
/// x86 build on a pre-SSE2 CPU would run `_c`; not a configuration this project
/// tests, and not reachable on any CI leg.)
///
/// **NOT locally measurable on an aarch64 box** — this arm rests on the C source
/// above plus the x86 CI leg of `config_permutations::speed_nonrd_hbd_byte_identity`.
#[inline]
fn hadamard_16x16_dispatched(src: &[i16], src_stride: usize) -> [i32; 256] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // aom_hadamard_16x16_avx2 / _sse2: the combine runs in int16 and wraps.
        let mut t = [0i16; 256];
        for idx in 0..4 {
            let off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
            // aom_hadamard_8x8_c's output is int16_t-valued by construction
            // (`buffer2` is int16_t), so this narrow is exact — it is the
            // `int16_t *t_coeff` staging buffer both x86 tiers write into.
            let sub = aom_dsp::dist::hadamard::hadamard_8x8(&src[off..], src_stride);
            for (d, &s) in t[idx * 64..idx * 64 + 64].iter_mut().zip(sub.iter()) {
                *d = s as i16;
            }
        }
        let mut coeff = [0i32; 256];
        for idx in 0..64 {
            let a0 = t[idx];
            let a1 = t[idx + 64];
            let a2 = t[idx + 128];
            let a3 = t[idx + 192];
            // _mm256_add_epi16 / _mm256_sub_epi16 WRAP, then _mm256_srai_epi16
            // shifts the already-wrapped int16 (C shifts the un-wrapped int32).
            let b0 = a0.wrapping_add(a1) >> 1;
            let b1 = a0.wrapping_sub(a1) >> 1;
            let b2 = a2.wrapping_add(a3) >> 1;
            let b3 = a2.wrapping_sub(a3) >> 1;
            // store_tran_low sign-extends the wrapped int16 to tran_low_t.
            coeff[idx] = i32::from(b0.wrapping_add(b2));
            coeff[idx + 64] = i32::from(b1.wrapping_add(b3));
            coeff[idx + 128] = i32::from(b0.wrapping_sub(b2));
            coeff[idx + 192] = i32::from(b1.wrapping_sub(b3));
        }
        // Same lane permutation the `_c` port applies (AVX2 gets it from
        // store_tran_low's 128-bit-lane unpack; SSE2 from store_tran_low_offset_4).
        for i in 0..16 {
            for j in 0..4 {
                coeff.swap(i * 16 + 4 + j, i * 16 + 8 + j);
            }
        }
        coeff
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        // `_c` == NEON (both combine in 32 bits). MEASURED on aarch64: the
        // 24-cell KB-20 byte-identity gate is green with this arm.
        aom_dsp::dist::hadamard::hadamard_16x16(src, src_stride)
    }
}

/// Test hook: the `_c`/NEON model of `aom_hadamard_16x16`, and the
/// as-dispatched one, side by side. Lets the KB-20 unit gate assert BOTH that
/// they agree at bd8 residual magnitude (where libaom's own cross-tier tests
/// live) and that the ISA split is real where the port claims it is.
#[doc(hidden)]
pub fn hadamard_16x16_models(src: &[i16], src_stride: usize) -> ([i32; 256], [i32; 256]) {
    (
        aom_dsp::dist::hadamard::hadamard_16x16(src, src_stride),
        hadamard_16x16_dispatched(src, src_stride),
    )
}

// ---------------------------------------------------------------------------
// av1_quantize_fp AS DISPATCHED — the ISA-conditional kernel (KB-20 root #3)
// ---------------------------------------------------------------------------

/// `av1_quantize_fp` (`log_scale = 0`) **as the reference build actually
/// dispatches it** — deliberately NOT `av1_quantize_fp_c`.
///
/// ## Why this exists (MEASURED 2026-07-30, KB-20)
/// The hbd `av1_block_yrd` arm feeds `av1_quantize_fp` the output of
/// `aom_hadamard_16x16`, whose 4-way combine stage can reach **±65534** —
/// outside `int16`. Every SIMD tier of `av1_quantize_fp` is a 16-bit kernel
/// that narrows `tran_low_t` on load and multiplies `dqcoeff` in 16 bits, so
/// outside the `int16` range the tiers stop agreeing with `av1_quantize_fp_c`
/// **and with each other**:
///
/// * `av1_quantize_fp_neon` (`av1/encoder/arm/quantize_neon.c:57`) loads via
///   `load_tran_low_to_s16q` = `vmovn_s32` — a **TRUNCATING** narrow — with no
///   per-coefficient dequant threshold, `vqaddq_s16` rounding, `vqdmulhq_s16`
///   >> 1, and `vmulq_s16` for `dqcoeff`;
/// * `av1_quantize_fp_avx2` (`av1/encoder/x86/av1_quantize_avx2.c:15`) loads
///   via `_mm_packs_epi32` — a **SATURATING** narrow — gates a whole 16-lane
///   group on `abs > (dequant >> 1) - 1`, and uses `_mm256_mulhi_epi16` /
///   `_mm256_mullo_epi16`;
/// * `av1_quantize_fp_c` narrows nothing (`int64_t abs_coeff`, 32-bit
///   `dqcoeff`).
///
/// All three agree while every coefficient AND every `qcoeff * dequant` fits
/// in `int16` — which is why the 8-bit `_lp` path, and every other
/// `av1_quantize_fp` call site in this port, never noticed. The hbd nonrd
/// estimate is the one place that leaves that range routinely.
///
/// Measured on this aarch64 reference build over bd{10,12} x cq{12,32,63} x
/// cpu-used{8,9}: the NEON model is **12/12 byte-identical** to real aomenc;
/// the saturating (x86) model is 9/12; `av1_quantize_fp_c` is 9/12. The shipped
/// gate then widened to 24 cells (cq{5,12,20,32,48,63}), still 24/24.
///
/// So this is ISA-conditional because **libaom's own encoder is**, exactly like
/// the aarch64 `-ffp-contract` note in `reference/BUILD_CONFIG.md`. The
/// non-x86/non-arm arm falls back to the `_c` semantics, which is what a build
/// with no SIMD tier would run.
///
/// ## x86 addendum (KB-20 root #4, 2026-07-30)
/// On x86 the `±65534` premise above **does not hold**, and the reason is
/// upstream of this function: [`hadamard_16x16_dispatched`] shows that
/// `aom_hadamard_16x16_{avx2,sse2}` already wrap their combine to `int16`, so
/// every coefficient reaching `av1_quantize_fp` on x86 is `int16`-valued and the
/// `_mm_packs_epi32` narrow modelled below is **inert**. That is exactly what
/// the first x86 CI run reported: both unit teeth
/// (`quantize_fp_dispatched_reduces_to_c_inside_int16` and
/// `..._differs_from_c_outside_int16`) PASSED while the 24-cell byte-identity
/// gate failed — i.e. this function was right and its INPUT was wrong. The
/// AVX2-vs-SSE2 threshold difference noted below is therefore also inert in this
/// regime (it only ever mattered for the out-of-range coefficients x86 does not
/// produce here), though the model is kept faithful to AVX2 regardless.
///
/// `iscan` must be the inverse of the scan `av1_block_yrd` selected for this
/// transform (the SIMD tiers derive the EOB from `iscan` in raster order, not
/// from the forward scan).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn quantize_fp_dispatched(
    coeff: &[i32],
    round_fp: &[i16; 2],
    quant_fp: &[i16; 2],
    dequant: &[i16; 2],
    scan: &[i16],
    iscan: &[i16],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let _ = scan;
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        // av1_quantize_fp_neon (av1/encoder/arm/quantize_neon.c:76) + its
        // quantize_fp_8 lane kernel (:57). No dequant threshold; the EOB is the
        // max `iscan` over lanes whose intermediate is > 0.
        let mut eob_max: i32 = -1;
        for (rc, &c) in coeff.iter().enumerate() {
            let lane = usize::from(rc != 0);
            // load_tran_low_to_s16q: vmovn_s32 truncates to the low 16 bits.
            let c16 = c as i16;
            let sign = c16 >> 15; // vshrq_n_s16(v_coeff, 15)
            let abs = c16.saturating_abs(); // vabsq_s16 saturates at INT16_MIN
            let tmp = abs.saturating_add(round_fp[lane]); // vqaddq_s16
            // vshrq_n_s16(vqdmulhq_s16(tmp, quant), 1)
            //   = (sat_i16((2 * tmp * quant) >> 16)) >> 1
            let dmulh = ((i32::from(tmp) * i32::from(quant_fp[lane])) >> 15)
                .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            let tmp2 = dmulh >> 1;
            let q = (tmp2 ^ sign).wrapping_sub(sign);
            qcoeff[rc] = i32::from(q);
            // vmulq_s16: the dequantized value is computed in 16 bits and WRAPS.
            dqcoeff[rc] = i32::from(q.wrapping_mul(dequant[lane]));
            if tmp2 > 0 {
                eob_max = eob_max.max(i32::from(iscan[rc]));
            }
        }
        (eob_max + 1) as u16
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // av1_quantize_fp_avx2 (av1/encoder/x86/av1_quantize_avx2.c:224) + its
        // quantize_fp_16 lane kernel (:194) and init_qp threshold (:30).
        // CORRECTED 2026-07-31 (this note previously named two SSE2/AVX2
        // differences as the first thing to check on an x86 gate failure; BOTH
        // were wrong, i.e. it pointed at a non-difference. Re-read from
        // upstream source):
        //   * The thresholds are the SAME integer predicate. SSE2 computes
        //     `thr = dequant >> 1` (av1_quantize_sse2.c:162-163) and masks with
        //     `qcoeff > thr || qcoeff == thr` (:91-92), i.e. `>= thr`; AVX2
        //     computes `thr = (dequant >> 1) - 1` and uses `>`, which is the
        //     same set. libaom says exactly why: "Subtracting 1 here eliminates
        //     a _mm256_cmpeq_epi16() instruction when calculating the zbin
        //     mask" (av1_quantize_avx2.c:52-53).
        //   * SSE2 does NOT gate per 8 lanes. It ORs two 8-lane masks into one
        //     `nzflag` (`_mm_movemask_epi8(mask0) | _mm_movemask_epi8(mask1)`,
        //     av1_quantize_sse2.c:95) and branches once — a 16-coefficient
        //     gate, same granularity as AVX2 and as the loop below.
        // The difference that IS real between the two x86 tiers is the eob
        // SOURCE: SSE2 scans the DEQUANTIZED value (`coeff0 = qcoeff0 *
        // dequant0`, then "Scan for eob", :111-121) while AVX2 tests the
        // QUANTIZED magnitude (`nz_mask = abs_q > 0`, av1_quantize_avx2.c:212)
        // — as does `_c`. So a `q * dequant` product that wraps to exactly 0 in
        // int16 gives SSE2 a different eob. That, not the threshold, is what to
        // check if the KB-20 byte-identity gate fails on an SSE2-only x86 host.
        // Catalogued as entry A2 in docs/LIBAOM_UPSTREAM_NOTES.md.
        let thr = [(dequant[0] >> 1) - 1, (dequant[1] >> 1) - 1];
        let mut eob_max: i32 = -1;
        for base in (0..coeff.len()).step_by(16) {
            let hi = (base + 16).min(coeff.len());
            // The whole 16-lane group is discarded when no lane clears the
            // threshold (`if (nzflag) ... else write_zero`).
            let mut nzflag = false;
            for rc in base..hi {
                let lane = usize::from(rc != 0);
                let c16 = saturating_narrow_i32(coeff[rc]);
                if c16.wrapping_abs() > thr[lane] {
                    nzflag = true;
                    break;
                }
            }
            if !nzflag {
                qcoeff[base..hi].fill(0);
                dqcoeff[base..hi].fill(0);
                continue;
            }
            for rc in base..hi {
                let lane = usize::from(rc != 0);
                let c16 = saturating_narrow_i32(coeff[rc]);
                // _mm256_abs_epi16 does NOT saturate: abs(INT16_MIN) is itself.
                let abs = c16.wrapping_abs();
                let tmp = abs.saturating_add(round_fp[lane]); // _mm256_adds_epi16
                // _mm256_mulhi_epi16
                let absq = ((i32::from(tmp) * i32::from(quant_fp[lane])) >> 16) as i16;
                // _mm256_sign_epi16(absq, coeff): zero when the coefficient is 0.
                let q = match c16.signum() {
                    1 => absq,
                    -1 => absq.wrapping_neg(),
                    _ => 0,
                };
                qcoeff[rc] = i32::from(q);
                // _mm256_mullo_epi16: 16-bit, wrapping.
                dqcoeff[rc] = i32::from(q.wrapping_mul(dequant[lane]));
                if absq > 0 {
                    eob_max = eob_max.max(i32::from(iscan[rc]));
                }
            }
        }
        (eob_max + 1) as u16
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "x86",
        target_arch = "x86_64"
    )))]
    {
        // No SIMD tier for this ISA in libaom -> av1_quantize_fp_c semantics.
        // The port's own dispatch is bit-identical to `_c` at every tier.
        aom_dsp::quant::simd::av1_quantize_fp_no_qmatrix_dispatch(
            quant_fp, dequant, round_fp, 0, scan, iscan, coeff, qcoeff, dqcoeff,
        )
    }
}

/// `_mm_packs_epi32` on one lane: signed saturating 32 -> 16 narrow.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn saturating_narrow_i32(v: i32) -> i16 {
    v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// One txb's Hadamard-estimate RD, `av1_block_yrd` with `use_hbd == 1`
/// (nonrd_opt.c:199-215 + `update_yrd_loop_vars_hbd`, :92) — the KB-20 arm.
///
/// Structurally identical to [`block_yrd_lowbd`]; every kernel is the 32-bit
/// (`tran_low_t`) sibling of the `_lp` one, and the 16x16 scan pair is the
/// **fp** one because the fp and lp 16x16 Hadamards emit different coefficient
/// orders. `bd` feeds `av1_highbd_block_error`'s `2*(bd-8)` rounding shift —
/// the only place the bit depth enters the arithmetic.
///
/// Same contract as [`block_yrd_lowbd`]: `diff` is the whole txb's residual at
/// stride `4 * bw4`, `tx_size` is the CLAMPED loop size (`AOMMIN(mi->tx_size,
/// TX_16X16)`), and the returned rate already carries C's final shifts.
#[allow(clippy::too_many_arguments)]
pub fn block_yrd_hbd(
    diff: &[i16],
    bw4: usize, // num_4x4_w of the txb bsize (diff stride = 4 * bw4)
    bh4: usize, // num_4x4_h
    max_blocks_wide: usize,
    max_blocks_high: usize,
    tx_size: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
    bd: u8,
) -> (i32, i64, bool) {
    debug_assert!(tx_size <= 2, "clamped to <= TX_16X16 (nonrd_opt.c:660)");
    debug_assert!(bd > 8, "the hbd arm runs only when is_cur_buf_hbd(xd)");
    let diff_stride = 4 * bw4;
    let block_step = 1usize << tx_size;
    let step = 1usize << (tx_size << 1); // 4x4 units per sub-block
    let _ = bh4;

    // `av1_quantize_fp`'s `[rc != 0]` lanes: index 0 = DC, index 1 = AC.
    let q2 = [quant_fp[0], quant_fp[1]];
    let r2 = [round_fp[0], round_fp[1]];
    let d2 = [dequant[0], dequant[1]];

    let mut rate: i32 = 0;
    let mut dist: i64 = 0;
    let mut eob_cost: i32 = 0;
    let mut temp_skippable = true;

    let mut coeff = [0i32; 256];
    let mut qcoeff = [0i32; 256];
    let mut dqcoeff = [0i32; 256];

    let mut r = 0usize;
    while r < max_blocks_high {
        let mut c = 0usize;
        while c < max_blocks_wide {
            let src_diff = &diff[(r * diff_stride + c) * 4..];
            let n = step << 4; // coefficients in this sub-block
            let eob: u16 = match tx_size {
                2 => {
                    coeff.copy_from_slice(&hadamard_16x16_dispatched(src_diff, diff_stride));
                    quantize_fp_dispatched(
                        &coeff[..256],
                        &r2,
                        &q2,
                        &d2,
                        &DEFAULT_SCAN_FP_16X16_TRANSPOSE,
                        &AV1_DEFAULT_ISCAN_FP_16X16_TRANSPOSE,
                        &mut qcoeff[..256],
                        &mut dqcoeff[..256],
                    )
                }
                1 => {
                    coeff[..64].copy_from_slice(&aom_dsp::dist::hadamard::hadamard_8x8(
                        src_diff,
                        diff_stride,
                    ));
                    quantize_fp_dispatched(
                        &coeff[..64],
                        &r2,
                        &q2,
                        &d2,
                        &DEFAULT_SCAN_8X8_TRANSPOSE,
                        &AV1_DEFAULT_ISCAN_8X8_TRANSPOSE,
                        &mut qcoeff[..64],
                        &mut dqcoeff[..64],
                    )
                }
                _ => {
                    // TX_4X4: aom_fdct4x4 + av1_quantize_fp over the NORMAL
                    // av1_scan_orders[TX_4X4][DCT_DCT] pair (no transpose,
                    // nonrd_opt.c:250). Lossless-only — the same envelope hole
                    // the lowbd arm has.
                    unimplemented!("TX_4X4 block_yrd hbd (lossless) — out of canon envelope")
                }
            };
            // update_yrd_loop_vars_hbd (nonrd_opt.c:92).
            let ncoeffs = eob as usize;
            temp_skippable &= ncoeffs == 0;
            eob_cost += get_msb(ncoeffs as u32 + 1);
            if ncoeffs == 1 {
                rate += qcoeff[0].abs();
            } else if ncoeffs > 1 {
                rate += aom_dsp::dist::hadamard::satd(&qcoeff[..n]);
            }
            dist += aom_dsp::dist::highbd_block_error(&coeff[..n], &dqcoeff[..n], bd).0 >> 2;
            c += block_step;
        }
        r += block_step;
    }

    // Same tail as the lowbd arm: `this_rdc->sse` is INT64_MAX from
    // av1_invalid_rd_stats, so the skippable-dist arm never fires here.
    let rate = (rate << (2 + AV1_PROB_COST_SHIFT)) + (eob_cost << AV1_PROB_COST_SHIFT);
    (rate, dist, temp_skippable)
}

// ---------------------------------------------------------------------------
// av1_nonrd_pick_intra_mode (nonrd_pickmode.c:1582) — the estimate arm.
// ---------------------------------------------------------------------------

/// Per-leaf inputs the estimate arm needs beyond [`SbEncodeEnv`].
pub struct NonrdIntraLeafCtx<'a> {
    /// `y_mode_costs[above_ctx][left_ctx]` KF table (13 modes) — from
    /// `mode_costs.y_mode_costs[intra_mode_context[A]][intra_mode_context[L]]`.
    pub bmode_costs: &'a [i32; 13],
    /// `skip_txfm_cost[skip_ctx]` — skip_ctx is 0 on the KEY intra path
    /// (every neighbour mi carries skip_txfm 0; the leaf_pick_sb_modes
    /// invariant, verified 64/64 across speeds 0-7).
    pub skip_cost: &'a [i32; 2],
    /// Above/left neighbour Y modes (A/L; DC=0 when unavailable) + their
    /// availability — the neighbour-prune inputs.
    pub above_mode: usize,
    pub left_mode: usize,
    pub up_available: bool,
    pub left_available: bool,
    /// x->source_variance for THIS leaf (perpixel_variance_y at leaf bsize).
    pub source_variance: u32,
    /// intra_avail geometry.
    pub partition: usize,
    /// Speed-9 sf gates (all false at speed 8).
    pub prune_h_pred_using_best_mode_so_far: bool,
    pub enable_intra_mode_pruning_using_neighbors: bool,
    pub prune_intra_mode_using_best_sad_so_far: bool,
    /// `cm->features.allow_screen_content_tools` — one of the FOUR inputs to
    /// C's `try_palette` (nonrd_pickmode.c:1701-1710), not the whole of it.
    /// See [`nonrd_palette_arm_is_live`].
    pub allow_screen_content_tools: bool,
    /// `cpi->oxcf.tool_cfg.enable_palette` (`--enable-palette`, C default 1).
    /// The C oracle's `shim_encode_av1_kf` passes **0**, which is why every
    /// canon cell has the palette arm dead by this term alone.
    pub enable_palette: bool,
    /// `cpi->sf.rt_sf.prune_palette_search_nonrd` — 1 at allintra speed >= 8
    /// (speed_features.c:582), and the estimate arm has no other dispatch.
    pub prune_palette_search_nonrd: i32,
    /// Edge filter type for directional prediction (V/H are directional):
    /// `get_intra_edge_filter_type(xd, 0)` — smooth above/left neighbour.
    pub luma_edge_filter_type: i32,
}

/// The estimate-arm result: the winner Y mode + the ctx snapshot fields
/// `store_coding_context_nonrd` (nonrd_opt.h:576-597) preserves that the
/// encode consumes. uv is ALWAYS DC (the chroma answer).
pub struct NonrdIntraPick {
    pub mode: usize,
    /// `mi->tx_size` = `AOMMIN(max_txsize_lookup[bsize],
    /// tx_mode_to_biggest_tx_size[TX_MODE_SELECT])` (nonrd_pickmode.c:1591) —
    /// the max square tx for the leaf (TX_64X64 cap).
    pub tx_size: usize,
    pub rd: PartRdStats,
}

/// `mi->tx_size` for an estimate-arm leaf: `AOMMIN(max_txsize_lookup[bsize],
/// tx_mode_to_biggest_tx_size[tx_mode_search_type])` (nonrd_pickmode.c:1591),
/// at EVERY `bsize` — square or not (KB-32's non-square refusal lived here
/// until 2026-08-02; see [`nonrd_pick_intra_mode`]'s per-txb walk).
/// (tx_mode_to_biggest_tx_size[TX_MODE_SELECT] = TX_64X64; allintra speed 8/9
/// keeps tx_size_search_method != USE_LARGEST_TX_SIZE → cm tx_mode =
/// TX_MODE_SELECT — HANDOFF: re-verify select_tx_mode at speed 8/9 allintra
/// (av1/encoder/encodeframe_utils/rdopt_utils select_tx_mode); if it were
/// TX_MODE_LARGEST the biggest is TX_64X64 anyway, same value — only
/// ONLY_4X4/lossless differs and that's out of envelope).
/// `b_width_log2_lookup[bsize]` / `b_height_log2_lookup[bsize]`
/// (nonrd_opt.h:114-119) = `log2(dimension) - 2`. Derived from the dimension
/// rather than re-typed, per the KB-34 lesson about re-typing tables; the two
/// agree on every one of the 16 `BLOCK_SIZES` entries C's table covers (it is
/// `BLOCK_SIZES`, not `BLOCK_SIZES_ALL` — C would read past its end on an
/// extended shape, which this form cannot).
fn b_log2(dim: usize) -> u32 {
    debug_assert!(dim.is_power_of_two() && dim >= 4);
    dim.trailing_zeros() - 2
}

/// C's `try_palette` for the nonrd estimate arm, verbatim
/// (nonrd_pickmode.c:1698-1710 + `av1_allow_palette`, blockd.h:1503-1510).
///
/// **Why this is a function and not an inline frame-level test.** Until
/// 2026-08-03 the port refused the whole leaf on `allow_screen_content_tools`
/// alone — one of the four terms — which is the KB-34 failure exactly: a
/// refusal that is *correct but overbroad*. It made `--cpu-used 8` refuse to
/// encode a plain smooth gradient at every size and every quantizer (136 of the
/// 2,012 rows of `benchmarks/nonsquare_leaf_reach_2026-08-02.tsv`), on cells
/// where `cpi->oxcf.tool_cfg.enable_palette` is **0** — i.e. where C provably
/// never enters the palette search at all. Speed 9 was unaffected for an
/// unrelated reason (`av1_set_screen_content_options`, encoder.c:2466-2470,
/// turns screen-content detection OFF when
/// `use_nonrd_pick_mode && !hybrid_intra_pickmode`, which is the speed-9
/// combination but not the speed-8 one), so the shape looked like a speed axis
/// and was not.
///
/// The four terms, in C's order:
/// 1. `cpi->oxcf.tool_cfg.enable_palette` (`--enable-palette`);
/// 2. `av1_allow_palette(allow_screen_content_tools, bsize)` — the frame flag
///    AND `block_size_wide <= 64` AND `block_size_high <= 64` AND
///    **ordinal** `bsize >= BLOCK_8X8` (an ordinal compare on the
///    `BLOCK_SIZES_ALL` index, so the extended 4:1 shapes at indices 16..21
///    satisfy it regardless of having a 4-px side);
/// 3. with `prune_palette_search_nonrd > 0` (1 at every speed that dispatches
///    this arm): **ordinal** `bsize <= BLOCK_16X16` and `source_variance > 200`;
/// 4. and, when the per-mode SAD prune ran, `best_sad_norm > thresh_sad`.
#[allow(clippy::too_many_arguments)]
pub fn nonrd_palette_arm_is_live(
    enable_palette: bool,
    allow_screen_content_tools: bool,
    bsize: usize,
    prune_palette_search_nonrd: i32,
    prune_mode_based_on_sad: bool,
    best_sad_norm: u32,
    source_variance: u32,
) -> bool {
    // av1_allow_palette (blockd.h:1503-1510). MAX_PALETTE_BLOCK_{WIDTH,HEIGHT}
    // are both 64 (blockd.h). BLOCK_8X8 is ordinal index 3.
    let allow_palette = allow_screen_content_tools
        && MI_W[bsize] * 4 <= 64
        && MI_H[bsize] * 4 <= 64
        && bsize >= 3;
    let mut try_palette = enable_palette && allow_palette;
    if prune_palette_search_nonrd > 0 {
        let thresh_sad = if prune_palette_search_nonrd > 1 { 100 } else { 20 };
        // BLOCK_16X16 is ordinal index 6.
        let prune = (!prune_mode_based_on_sad || best_sad_norm > thresh_sad)
            && bsize <= 6
            && source_variance > 200;
        try_palette &= prune;
    }
    try_palette
}

pub fn nonrd_leaf_tx_size(bsize: usize) -> usize {
    // `max_txsize_lookup[]` verbatim (common_data.h:105-124), BLOCK_SIZES_ALL
    // order — the square tx of the SHORT side, so it is smaller than the block
    // for every non-square `bsize`. `AOMMIN(.., tx_mode_to_biggest_tx_size[
    // TX_MODE_SELECT] = TX_64X64)` is the identity on this table (its maximum
    // IS TX_64X64), so no cap is applied.
    const MAX_TXSIZE_LOOKUP: [usize; 22] = [
        0, // 4X4                         TX_4X4
        0, 0, 1, // 4X8,     8X4,     8X8      TX_4X4,   TX_4X4,   TX_8X8
        1, 1, 2, // 8X16,    16X8,    16X16    TX_8X8,   TX_8X8,   TX_16X16
        2, 2, 3, // 16X32,   32X16,   32X32    TX_16X16, TX_16X16, TX_32X32
        3, 3, // 32X64,   64X32             TX_32X32, TX_32X32
        4, // 64X64                       TX_64X64
        4, 4, 4, // 64X128,  128X64,  128X128  TX_64X64 x3
        0, 0, 1, // 4X16,    16X4,    8X32     TX_4X4,   TX_4X4,   TX_8X8
        1, 2, 2, // 32X8,    16X64,   64X16    TX_8X8,   TX_16X16, TX_16X16
    ];
    MAX_TXSIZE_LOOKUP[bsize]
}

/// `txsize_to_bsize[]` (common_data.h:280-284) for the five SQUARE tx sizes —
/// the only ones [`nonrd_leaf_tx_size`] can return, because
/// `max_txsize_lookup` is square-valued by construction.
const TXSIZE_TO_BSIZE: [usize; 5] = [
    0,  // TX_4X4   -> BLOCK_4X4
    3,  // TX_8X8   -> BLOCK_8X8
    6,  // TX_16X16 -> BLOCK_16X16
    9,  // TX_32X32 -> BLOCK_32X32
    12, // TX_64X64 -> BLOCK_64X64
];

/// `mi_size_wide[BLOCK_64X64]` — `av1_foreach_transformed_block_in_plane`'s
/// `mu_blocks_wide/high` unit (encodemb.c:551-556, luma: `max_unit_bsize` is
/// BLOCK_64X64 at subsampling 0). Only bites for a leaf wider/taller than
/// 64 px, which the KEY VBP tree never stamps; carried so the walk is C's.
const MU_UNIT_4X4: usize = 16;

// ---------------------------------------------------------------------------
// Test instrumentation: how often the estimate arm codes a MULTI-txb leaf.
// ---------------------------------------------------------------------------
//
// A gate that encodes a cell "which reaches a non-square leaf" is vacuous
// unless it can show the cell reached one (playbook §2/§8: derive coverage
// from artefacts, not from names). Before 2026-08-02 the evidence was a panic;
// now that the arm codes these leaves the counter is what is left. Indexed by
// `bsize`, bumped once per leaf on the multi-txb path only, so a square leaf
// pays nothing.
//
// THREAD-LOCAL, deliberately. A process-global counter would be read by
// whichever test happened to finish its encode first — `cargo test` runs the
// tests in one binary CONCURRENTLY, so a shared counter turns
// "reset / encode / read" into a race and the non-vacuity assertions into
// coin flips. The encode is single-threaded, so per-thread is exactly the
// right granularity and costs no atomics.
thread_local! {
    static MULTI_TXB_LEAVES: std::cell::Cell<[u64; 22]> = const {
        std::cell::Cell::new([0; 22])
    };
}

/// Per-`bsize` count of leaves the estimate arm has coded with more than one
/// txb since the last [`reset_multi_txb_leaf_counts`] **on this thread**.
/// Test instrumentation.
pub fn multi_txb_leaf_counts() -> [u64; 22] {
    MULTI_TXB_LEAVES.with(std::cell::Cell::get)
}

/// Zero this thread's [`multi_txb_leaf_counts`] counters. Test instrumentation.
pub fn reset_multi_txb_leaf_counts() {
    MULTI_TXB_LEAVES.with(|c| c.set([0; 22]));
}

// ---------------------------------------------------------------------------
// Test instrumentation: how close the estimate arm gets to C's palette arm.
// ---------------------------------------------------------------------------
//
// The refusal below fires on C's `try_palette`, which is a conjunction of four
// terms. A gate asserting "the port did not refuse" is vacuous if the FIRST
// term (`--enable-palette`) is off, and every canon cell has it off — so a
// three-number breakdown is recorded per leaf instead of a bare pass/fail:
//
//   [0] leaves where `enable_palette && av1_allow_palette(..)` both held, i.e.
//       the speed prune (term 3/4) is the only thing left;
//   [1] of those, how many had ordinal `bsize <= BLOCK_16X16`;
//   [2] of those, how many ALSO had `source_variance > 200` (= `try_palette`,
//       modulo the SAD term — which is the one that can only ever subtract).
//
// Thread-local for the same reason as `MULTI_TXB_LEAVES` above.
thread_local! {
    static PALETTE_GATE_REACH: std::cell::Cell<[u64; 3]> = const {
        std::cell::Cell::new([0; 3])
    };
}

/// `[allow_palette_held, .. and bsize <= BLOCK_16X16, .. and variance > 200]`
/// over the estimate-arm leaves coded since the last
/// [`reset_palette_gate_reach`] **on this thread**. Test instrumentation; see
/// [`nonrd_palette_arm_is_live`].
pub fn palette_gate_reach() -> [u64; 3] {
    PALETTE_GATE_REACH.with(std::cell::Cell::get)
}

/// Zero this thread's [`palette_gate_reach`] counters. Test instrumentation.
pub fn reset_palette_gate_reach() {
    PALETTE_GATE_REACH.with(|c| c.set([0; 3]));
}

#[inline]
fn note_palette_gate_reach(allow_palette: bool, bsize: usize, source_variance: u32) {
    if !allow_palette {
        return;
    }
    PALETTE_GATE_REACH.with(|c| {
        let mut v = c.get();
        v[0] += 1;
        if bsize <= 6 {
            v[1] += 1;
            if source_variance > 200 {
                v[2] += 1;
            }
        }
        c.set(v);
    });
}

#[inline]
fn note_multi_txb_leaf(bsize: usize) {
    MULTI_TXB_LEAVES.with(|c| {
        let mut v = c.get();
        v[bsize] += 1;
        c.set(v);
    });
}

/// `should_prune_intra_modes_using_neighbors` (nonrd_pickmode.c:1566).
fn should_prune_intra_modes_using_neighbors(
    enable: bool,
    this_mode: usize,
    above_mode: usize,
    left_mode: usize,
    up_available: bool,
    left_available: bool,
) -> bool {
    if !enable {
        return false;
    }
    if this_mode == 0 {
        return false; // DC never pruned
    }
    up_available && this_mode != above_mode && left_available && this_mode != left_mode
}

/// `av1_nonrd_pick_intra_mode` (nonrd_pickmode.c:1582), Y estimate loop.
///
/// The prediction step (`av1_estimate_block_intra` → `av1_predict_intra_block
/// _facade`) writes INTO the recon plane at the leaf position (the C facade's
/// dst IS pd->dst), scribbling only inside the block — the winner encode
/// (`encode_b_intra_dry`) re-predicts + adds residual afterwards, exactly like
/// C's encode_superblock.
///
/// **The txb walk** is `av1_foreach_transformed_block_in_plane` (encodemb.c:536)
/// inlined. `mi->tx_size` is `max_txsize_lookup[bsize]`, the max square tx of
/// the SHORT side, so `txsize_to_bsize[mi->tx_size] == bsize` — one visit at
/// `(0,0)`, C's `plane_bsize == tx_bsize` early return (`:546-549`) — for every
/// SQUARE leaf, and TWO visits for the four non-square leaves the KEY VBP tree
/// can stamp (BLOCK_16X8 / 8X16 / 32X16 / 16X32). Until 2026-08-02 the
/// non-square case was a named refusal (KB-32); everything about the walk below
/// is byte-inert at a square leaf, where it degenerates to the single visit the
/// refusal's invariant described.
///
/// Three details of C's the walk carries that a naive "loop the txbs" does not:
///
/// * each visit predicts into `pd->dst` **before** the next visit reads its
///   neighbours out of that same buffer (`av1_predict_intra_block_facade`'s
///   `ref == dst`, reconintra.c:1622) — so txb 1 of a BLOCK_8X16 predicts from
///   txb 0's *prediction*, there being no residual on this arm;
/// * `av1_block_yrd` is handed `bsize_tx = txsize_to_bsize[tx_size]`
///   (nonrd_opt.c:658), so its `num_4x4_w/h` are the TXB's — but
///   `xd->mb_to_right_edge` is still the LEAF's, so the frame-edge clamp at
///   nonrd_opt.c:141-144 subtracts the leaf's overhang from each txb's extent
///   (and can clamp a txb to zero rows, which C then codes as rate 0 / dist 0
///   / skippable). Reproduced exactly;
/// * `args->skippable` is **assigned**, not accumulated: `av1_block_yrd` ends
///   `this_rdc->skip_txfm = *skippable = temp_skippable` (nonrd_opt.c:327), and
///   `temp_skippable` restarts at 1 in each call — so a multi-txb leaf's
///   skippable flag is the LAST txb's, not the AND. Rate and dist DO accumulate
///   (`args->rdc->rate += ...`, nonrd_opt.c:667-668).
#[allow(clippy::too_many_arguments)]
pub fn nonrd_pick_intra_mode(
    env: &SbEncodeEnv,
    lctx: &NonrdIntraLeafCtx,
    recon_y: &mut [u16],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    rdmult: i32,
) -> NonrdIntraPick {
    // `use_hbd = is_cur_buf_hbd(xd)` — the buffer's YV12_FLAG_HIGHBITDEPTH,
    // which this project's encode path raises exactly when `bd > 8`.
    let use_hbd = env.bd > 8;
    let mi_w = MI_W[bsize];
    let mi_h = MI_H[bsize];
    let bw = mi_w * 4;
    let bh = mi_h * 4;
    let tx_size_full = nonrd_leaf_tx_size(bsize); // mi->tx_size (signalled)
    let tx_clamped = tx_size_full.min(2); // AOMMIN(tx_size, TX_16X16) for block_yrd
    // `tx_bsize = txsize_to_bsize[mi->tx_size]` (nonrd_pickmode.c:1594) — the
    // per-visit block C hands `av1_block_yrd` as `bsize_tx`.
    let tx_bsize = TXSIZE_TO_BSIZE[tx_size_full];
    let tx_w4 = MI_W[tx_bsize];
    let tx_h4 = MI_H[tx_bsize];
    let tx_bw = tx_w4 * 4;
    let tx_bh = tx_h4 * 4;
    // C's `plane_bsize == tx_bsize` early return (encodemb.c:546-549).
    let single_txb = bsize == tx_bsize;
    if !single_txb {
        note_multi_txb_leaf(bsize);
    }

    // Edge clamps. `mb_to_right_edge = (mi_cols - mi_w - mi_col) * 4 * 8` (in
    // 1/8 pel) is the LEAF's, and BOTH clamps below read it — they differ only
    // in what they add it to:
    //
    // * the WALK's (`max_block_wide(xd, plane_bsize, 0)`,
    //   av1_common_int.h:1567) adds it to the LEAF's width, which reduces to
    //   `mi_cols - mi_col` exactly (the >>3 and >>2 are both exact here since
    //   the quantity is a multiple of 32). It decides which txbs are visited;
    // * `av1_block_yrd`'s (nonrd_opt.c:141-144) adds `mb_to_right_edge >> 5` to
    //   the **TXB's** `num_4x4_w`. At a square leaf the two are the same
    //   number; at a non-square one they are not, and this is the one C uses
    //   inside the transform loop.
    let mb_right = (env.mi_cols - mi_w as i32 - mi_col) * 32;
    let mb_bottom = (env.mi_rows - mi_h as i32 - mi_row) * 32;
    let max_blocks_wide = (tx_w4 as i32 + if mb_right >= 0 { 0 } else { mb_right >> 5 }).max(0);
    let max_blocks_high = (tx_h4 as i32 + if mb_bottom >= 0 { 0 } else { mb_bottom >> 5 }).max(0);
    let (max_blocks_wide, max_blocks_high) = (max_blocks_wide as usize, max_blocks_high as usize);
    let walk_blocks_wide = if mb_right >= 0 {
        mi_w
    } else {
        (env.mi_cols - mi_col).max(0) as usize
    };
    let walk_blocks_high = if mb_bottom >= 0 {
        mi_h
    } else {
        (env.mi_rows - mi_row).max(0) as usize
    };
    // `mu_blocks_wide/high` (encodemb.c:553-556).
    let mu_w = MU_UNIT_4X4.min(walk_blocks_wide);
    let mu_h = MU_UNIT_4X4.min(walk_blocks_high);

    let ref_off = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
    let src_off = ref_off; // src and recon share layout in this port

    let above_ctx = INTRA_MODE_CONTEXT[lctx.above_mode.min(12)];
    let left_ctx = INTRA_MODE_CONTEXT[lctx.left_mode.min(12)];
    let _ = (above_ctx, left_ctx); // bmode_costs already row-selected by caller
    // HANDOFF: caller must select bmode_costs with the SAME ctx pair —
    // the row selection is the caller's so it can reuse leaf_pick_sb_modes'
    // existing neighbour reads; assert parity there.

    let mut best_rdc = PartRdStats::invalid();
    let mut best_mode = 0usize; // DC_PRED
    let mut best_sad = u32::MAX;
    // `sf.rt_sf.prune_intra_mode_using_best_sad_so_far && bsize == tx_bsize`
    // (nonrd_pickmode.c:1599-1601) — C gates the SAD prune on the leaf BEING
    // one txb, so it is structurally unreachable on the multi-txb walk.
    let prune_mode_based_on_sad = lctx.prune_intra_mode_using_best_sad_so_far && single_txb;
    let allow_skip_nondc = true; // flat_blocks_screen is REALTIME+SCREEN only → const true (ALLINTRA)

    // The txb visit list, `av1_foreach_transformed_block_in_plane` (encodemb.c:
    // 536-585) evaluated once — it does not depend on the mode. 4 is the true
    // maximum over BLOCK_SIZES_ALL (`max_txsize_lookup` is never more than one
    // "step" below the block on either axis); the array is 8 with an assert so
    // an unexpected shape fails loudly instead of truncating.
    let mut visits = [(0usize, 0usize); 8];
    let mut n_visits = 0usize;
    if single_txb {
        visits[0] = (0, 0);
        n_visits = 1;
    } else {
        let mut chunk_r = 0usize;
        while chunk_r < walk_blocks_high {
            let unit_h = (mu_h + chunk_r).min(walk_blocks_high);
            let mut chunk_c = 0usize;
            while chunk_c < walk_blocks_wide {
                let unit_w = (mu_w + chunk_c).min(walk_blocks_wide);
                let mut blk_row = chunk_r;
                while blk_row < unit_h {
                    let mut blk_col = chunk_c;
                    while blk_col < unit_w {
                        assert!(
                            n_visits < visits.len(),
                            "nonrd estimate arm: bsize {bsize} / tx_size {tx_size_full} needs \
                             more than {} txb visits — widen `visits`",
                            visits.len()
                        );
                        visits[n_visits] = (blk_row, blk_col);
                        n_visits += 1;
                        blk_col += tx_w4;
                    }
                    blk_row += tx_h4;
                }
                chunk_c += mu_w;
            }
            chunk_r += mu_h;
        }
        // encodemb.c:584 `assert(i >= 1)`.
        assert!(n_visits >= 1, "the txb walk visited nothing at bsize {bsize}");
    }
    let visits = &visits[..n_visits];

    let mut diff = vec![0i16; tx_bw * tx_bh];
    let mut pred = vec![0u16; tx_bw * tx_bh];

    for &this_mode in INTRA_MODE_LIST.iter() {
        // Force DC for spatially flat block at top-left, bsize >= 32x32
        // (nonrd_pickmode.c:1636-1640) — LIVE on the flat canon cells.
        if lctx.source_variance == 0 && mi_col == 0 && mi_row == 0 && bsize >= 9 && this_mode > 0 {
            continue;
        }
        // prune_h_pred_using_best_mode_so_far (:1648-1650), speed 9.
        if lctx.prune_h_pred_using_best_mode_so_far
            && this_mode == 2
            && best_mode == 1
            && allow_skip_nondc
        {
            continue;
        }
        if should_prune_intra_modes_using_neighbors(
            lctx.enable_intra_mode_pruning_using_neighbors,
            this_mode,
            lctx.above_mode,
            lctx.left_mode,
            lctx.up_available,
            lctx.left_available,
        ) {
            // (:1656-1668), speed 9.
            if (this_mode == 1 || this_mode == 2) && lctx.source_variance <= 50 && allow_skip_nondc
            {
                continue;
            }
            if best_mode == 0 && this_mode == 9 && allow_skip_nondc {
                continue;
            }
        }

        // --- av1_foreach_transformed_block_in_plane(.., av1_estimate_block_
        //     intra, &args) (nonrd_pickmode.c:1671) ---
        // `args.rdc` starts at rate 0 / dist 0 (:1665) and ACCUMULATES over the
        // visits; `args.skippable` starts at 1 (:1667) but each `av1_block_yrd`
        // ASSIGNS it (nonrd_opt.c:327), so the leaf's flag is the last txb's.
        let mut acc_rate: i32 = 0;
        let mut acc_dist: i64 = 0;
        let mut skippable = true;
        let mut sad_pruned = false;
        for &(blk_row, blk_col) in visits {
            let txb_off = ref_off + (blk_row * env.stride + blk_col) * 4;
            let txb_src = src_off + (blk_row * env.stride + blk_col) * 4;
            // Predict with the leaf's SIGNALLED tx_size (prediction granularity
            // is mi->tx_size, NOT the clamped block_yrd loop size).
            let (n_top, n_topright, n_left, n_bottomleft) = aom_dsp::entropy::partition::intra_avail(
                env.sb_size,
                bsize,
                mi_row,
                mi_col,
                lctx.up_available,
                lctx.left_available,
                env.tile_col_end,
                env.tile_row_end,
                lctx.partition,
                tx_size_full,
                0,
                0,
                blk_row as i32,
                blk_col as i32,
                bw as i32,
                bh as i32,
                env.mi_cols,
                env.mi_rows,
                this_mode,
                0,     // angle_delta * ANGLE_STEP
                false, // use_filter_intra
            );
            predict_intra_high(
                recon_y,
                txb_off,
                env.stride,
                &mut pred,
                tx_bw,
                this_mode,
                0,
                false,
                0,
                env.disable_edge_filter,
                lctx.luma_edge_filter_type,
                tx_size_full,
                n_top as usize,
                n_topright,
                n_left as usize,
                n_bottomleft,
                i32::from(env.bd),
            );
            // Facade writes prediction into the recon plane (dst) — mirror that.
            // It must land BEFORE the next txb's `intra_avail`/`predict` reads its
            // above/left neighbours out of the same buffer.
            for r in 0..tx_bh {
                recon_y[txb_off + r * env.stride..txb_off + r * env.stride + tx_bw]
                    .copy_from_slice(&pred[r * tx_bw..r * tx_bw + tx_bw]);
            }

            // Speed-9 SAD prune (av1_estimate_block_intra, nonrd_opt.c:629-648).
            // `prune_mode_based_on_sad` implies `bsize == tx_bsize`, so `sdf` runs
            // over the whole leaf and this is the only visit.
            if prune_mode_based_on_sad {
                // Enforced, not argued: `pred` is tx-sized, so reading it over
                // the LEAF's bw x bh below is only sound when they are the same
                // block. C's own gate guarantees that; this is the tripwire if
                // it ever stops.
                debug_assert!(single_txb, "the SAD prune requires bsize == tx_bsize");
                let mut this_sad: u32 = 0;
                for r in 0..bh {
                    for c in 0..bw {
                        let s = env.src_y[src_off + r * env.stride + c] as i32;
                        let p = pred[r * bw + c] as i32;
                        this_sad += (s - p).unsigned_abs();
                    }
                }
                // KB-20's SECOND bd8-specific step — NOT inside av1_block_yrd, and
                // NOT named by the old handoff assert. `fn_ptr[bsize].sdf` is bound
                // by `highbd_set_var_fns` to `aom_highbd_sadWxH_bits{8,10,12}`,
                // whose `MAKE_BFP_SAD_WRAPPER` bodies (encoder_utils.h:158) return
                // the raw SAD `>> 0` / `>> 2` / `>> 4` respectively — i.e. `bd - 8`,
                // NOT `2 * (bd - 8)`: a SAD is linear in pixel magnitude, so the
                // 10-bit range is 4x the 8-bit one and the shift is 2. The raw sum
                // above is the `_bits8` value; normalise it to this bit depth.
                //
                // HONESTY NOTE (measured 2026-07-30): this normalisation is
                // source-derived, and on the 24-cell KB-20 gate it is
                // decision-INERT — deleting it leaves every cell byte-identical.
                // That is expected: the prune is the RATIO test
                // `this_sad > best_sad + (best_sad >> 4)`, so shifting both sides
                // equally only changes the rounding. What the gate DOES witness is
                // getting the shift WRONG: `2 * (bd - 8)` over-shifts far enough to
                // destroy the ratio and diverged on 1 of the 12 cells first
                // measured. Keep the correct form; do not read the inertness as
                // permission to drop it.
                this_sad >>= u32::from(env.bd) - 8;
                let sad_threshold = if best_sad != u32::MAX {
                    best_sad + (best_sad >> 4)
                } else {
                    u32::MAX
                };
                if this_sad > sad_threshold {
                    // rate INT_MAX → the caller-side `if (this_rdc.rate == INT_MAX)
                    // continue` (:1674).
                    sad_pruned = true;
                    break;
                }
                if this_sad < best_sad {
                    best_sad = this_sad;
                }
            }

            // av1_subtract_block over the txb (`bsize_tx` dims, nonrd_opt.c:657 →
            // :157-160, from `p->src.buf` / `pd->dst.buf` offset by
            // `4 * (row * stride + col)`, :653-654).
            highbd_subtract_block(
                tx_bh,
                tx_bw,
                &mut diff,
                tx_bw,
                &env.src_y[txb_src..],
                env.stride,
                &pred,
                tx_bw,
            );
            let (rate_yrd, dist_yrd, txb_skippable) = if use_hbd {
                block_yrd_hbd(
                    &diff,
                    tx_w4,
                    tx_h4,
                    max_blocks_wide,
                    max_blocks_high,
                    tx_clamped,
                    env.rows_y.round_fp,
                    env.rows_y.quant_fp,
                    env.rows_y.dequant,
                    env.bd,
                )
            } else {
                block_yrd_lowbd(
                    &diff,
                    tx_w4,
                    tx_h4,
                    max_blocks_wide,
                    max_blocks_high,
                    tx_clamped,
                    env.rows_y.round_fp,
                    env.rows_y.quant_fp,
                    env.rows_y.dequant,
                )
            };
            // `args->rdc->rate += this_rdc.rate; args->rdc->dist += this_rdc.dist`
            // (nonrd_opt.c:667-668) — `int` / `int64_t` adds in C.
            acc_rate = acc_rate.wrapping_add(rate_yrd);
            acc_dist = acc_dist.wrapping_add(dist_yrd);
            skippable = txb_skippable;
        }
        if sad_pruned {
            continue;
        }
        let (rate_yrd, dist_yrd) = (acc_rate, acc_dist);

        // (:1676-1687): skip-cost fold (skip_ctx 0 on KEY intra — module docs)
        // + the KF y-mode cost.
        let mut rate = if skippable {
            lctx.skip_cost[1] // '=' — clobbers the SATD rate (C :1678)
        } else {
            rate_yrd + lctx.skip_cost[0]
        };
        rate += lctx.bmode_costs[this_mode];
        let rdc = crate::rd::rdcost(rdmult, rate, dist_yrd);
        if rdc < best_rdc.rdcost {
            best_rdc = PartRdStats {
                rate,
                dist: dist_yrd,
                rdcost: rdc,
            };
            best_mode = this_mode;
        }
        // flat_blocks_screen / allow_skip_nondc mutation: dead at ALLINTRA
        // (cpi->oxcf.mode == REALTIME gate, :1620-1623).
    }

    // Palette arm (nonrd_pickmode.c:1698-1731). `av1_search_palette_mode_luma`
    // is not ported for this arm, so the port REFUSES rather than silently
    // coding a different winner — but it must refuse on C's own predicate, not
    // on a superset of it. See [`nonrd_palette_arm_is_live`] for the four
    // terms and for what the old frame-level form got wrong.
    let best_sad_norm = best_sad >> (b_log2(bw) + b_log2(bh));
    note_palette_gate_reach(
        lctx.enable_palette && lctx.allow_screen_content_tools && bw <= 64 && bh <= 64 && bsize >= 3,
        bsize,
        lctx.source_variance,
    );
    assert!(
        !nonrd_palette_arm_is_live(
            lctx.enable_palette,
            lctx.allow_screen_content_tools,
            bsize,
            lctx.prune_palette_search_nonrd,
            prune_mode_based_on_sad,
            best_sad_norm,
            lctx.source_variance,
        ),
        "HANDOFF: av1_search_palette_mode_luma (intra_mode_search.c:1122) is not ported for \
         the nonrd estimate arm, and C's try_palette is TRUE at bsize {bsize} \
         (source_variance {}, best_sad_norm {best_sad_norm}) — the palette RD search would \
         run here and can win the leaf. Wire it through `palette_search::\
         rd_pick_palette_intra_sby` before encoding this configuration.",
        lctx.source_variance,
    );

    // mi->mode = best_mode; mi->uv_mode = UV_DC_PRED (:1734-1735) — the
    // chroma answer. store_coding_context_nonrd's ctx->mic snapshot maps to
    // the LeafWinner the caller builds from this pick.
    NonrdIntraPick {
        mode: best_mode,
        tx_size: tx_size_full,
        rd: best_rdc,
    }
}

/// `hybrid_intra_mode_search` (partition_search.c:755): the speed-8 dispatch.
/// `hybrid_intra_pickmode`: 2 at speed 8, 0 at speed 9 (allintra).
/// Returns true → run the full-RD leaf (`leaf_pick_sb_modes`); false → the
/// estimate arm above. `var_thresh = {0, 101, 201}[hybrid - 1]`.
pub fn hybrid_use_rdopt(hybrid_intra_pickmode: i32, bsize: usize, source_variance: u32) -> bool {
    debug_assert!((0..=3).contains(&hybrid_intra_pickmode));
    if hybrid_intra_pickmode == 0 {
        return false;
    }
    // bsize < BLOCK_16X16 (port numbering: < 6 — 8x8 and the sub-8x8 rects;
    // the KEY VBP tree stamps nothing below 8x8).
    if bsize >= 6 {
        return false;
    }
    let var_thresh: [u32; 3] = [0, 101, 201];
    source_variance >= var_thresh[(hybrid_intra_pickmode - 1) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// quantize_lp against hand-computed values (C:214 semantics: round/quant
    /// lane [rc!=0], eob over the scan order, dq = q * dequant).
    #[test]
    fn quantize_lp_basic() {
        let mut coeff = [0i16; 64];
        coeff[0] = 100; // DC
        coeff[8] = -40; // first AC in the transposed scan (scan[1] = 8)
        let round_fp = [48i16, 24, 24, 24, 24, 24, 24, 24];
        let quant_fp = [2048i16, 1024, 1024, 1024, 1024, 1024, 1024, 1024];
        let dequant = [32i16, 64, 64, 64, 64, 64, 64, 64];
        let mut q = [0i16; 64];
        let mut dq = [0i16; 64];
        let eob = quantize_lp(
            &coeff,
            64,
            &round_fp,
            &quant_fp,
            &mut q,
            &mut dq,
            &dequant,
            &DEFAULT_SCAN_8X8_TRANSPOSE,
        );
        // DC: (100+48)*2048 >> 16 = 4; dq = 4*32 = 128.
        assert_eq!(q[0], 4);
        assert_eq!(dq[0], 128);
        // AC at rc=8: (40+24)*1024 >> 16 = 1, negative → -1; dq = -64.
        assert_eq!(q[8], -1);
        assert_eq!(dq[8], -64);
        // scan[1] == 8 → eob index 1 → eob = 2.
        assert_eq!(eob, 2);
    }

    /// Hadamard lp 8x8: DC-only input → coeff[0] = 64 * v (sum), others 0.
    ///
    /// **This test is transpose-BLIND and that is why it is not the gate.**
    /// A flat input puts all the energy at coefficient 0, which is the one
    /// fixed point of the transpose — so it passed for the whole time
    /// `hadamard_lp_8x8` was missing C's trailing transpose (KB-12). The
    /// kernel's real lock is `tests/nonrd_block_yrd_lp_diff.rs`, against the
    /// exported `aom_hadamard_lp_8x8_c`. Kept as a cheap smoke test only.
    #[test]
    fn hadamard_lp_8x8_flat() {
        let src = [3i16; 64];
        let mut coeff = [0i16; 64];
        hadamard_lp_8x8(&src, 8, &mut coeff);
        assert_eq!(coeff[0], 64 * 3);
        assert!(coeff[1..].iter().all(|&c| c == 0));
    }

    /// Hadamard lp 16x16: flat input → DC = 256*v/4 (the >>1 stages halve
    /// twice), others 0. Transpose-blind for the same reason as the 8x8 one.
    #[test]
    fn hadamard_lp_16x16_flat() {
        let src = [2i16; 256];
        let mut coeff = [0i16; 256];
        hadamard_lp_16x16(&src, 16, &mut coeff);
        // per-8x8 DC = 128; combine: b0 = (128+128)>>1 = 128; c0 = 128+128 = 256.
        assert_eq!(coeff[0], 256);
        assert!(coeff[1..].iter().all(|&c| c == 0));
    }

    /// A GOLDEN vector that sees the transpose, so this crate's own test suite
    /// (which cannot link the C oracle) is not left guarding the lowbd estimate
    /// kernel with flat inputs alone. Two impulses — `src[1]` at (row 0, col 1)
    /// and `src[8]` at (row 1, col 0) — make the output asymmetric, and the
    /// expected values are `aom_hadamard_lp_8x8_c`'s, captured 2026-08-02 from
    /// the reference build (the exhaustive lock is
    /// `tests/nonrd_block_yrd_lp_diff.rs::lp_hadamard_matches_c`).
    #[test]
    fn hadamard_lp_8x8_golden_asymmetric_impulse() {
        let mut src = [0i16; 64];
        src[1] = 100;
        src[8] = -40;
        #[rustfmt::skip]
        const WANT: [i16; 64] = [
            60, 60, 60, 60, 140, 140, 140, 140,
            60, 60, 60, 60, 140, 140, 140, 140,
            60, 60, 60, 60, 140, 140, 140, 140,
            60, 60, 60, 60, 140, 140, 140, 140,
            -140, -140, -140, -140, -60, -60, -60, -60,
            -140, -140, -140, -140, -60, -60, -60, -60,
            -140, -140, -140, -140, -60, -60, -60, -60,
            -140, -140, -140, -140, -60, -60, -60, -60,
        ];
        let mut coeff = [0i16; 64];
        hadamard_lp_8x8(&src, 8, &mut coeff);
        assert_eq!(
            coeff, WANT,
            "aom_hadamard_lp_8x8_c golden vector. If the values are a TRANSPOSE \
             of the expected ones, the trailing transpose at aom_dsp/avg.c:232-236 \
             was dropped again — that was KB-12's root"
        );
        // Non-vacuity: the golden output must genuinely differ from its own
        // transpose, or it could not witness the defect it exists for.
        let transposed: Vec<i16> = (0..64).map(|i| WANT[(i % 8) * 8 + i / 8]).collect();
        assert_ne!(WANT.to_vec(), transposed, "the golden vector is transpose-blind");
    }

    /// The speed axis, 0..9: which leaves reach the LOWBD estimate arm the
    /// KB-12 fix lives in. `hybrid_intra_pickmode` is 2 at speed 8 and 0 at
    /// speed >= 9 (speed_features.c:578 / :597, resolved in
    /// `partition_pick::nonrd_leaf_pick`); speeds 0..7 never run the nonrd
    /// pickmode at all. Both live arms must reach the estimate kernel, or the
    /// e2e speed-8/9 gates are covering one arm twice.
    #[test]
    fn estimate_arm_is_reachable_from_both_hybrid_arms() {
        // (speed, hybrid_intra_pickmode) as the port resolves it.
        for speed in 0..=9i32 {
            let hybrid = if speed >= 9 { 0 } else { 2 };
            if speed < 8 {
                continue; // no nonrd pickmode below speed 8 (speed_features.c:578)
            }
            // BLOCK_8X8 (3) with high source variance is the ONLY shape whose
            // arm depends on the speed: full-RD at 8, estimate at 9.
            assert_eq!(
                hybrid_use_rdopt(hybrid, 3, 5000),
                speed == 8,
                "speed {speed}: BLOCK_8X8 high-variance arm"
            );
            // Every leaf at or above BLOCK_16X16 takes the estimate arm at both
            // speeds, so the LOWBD kernels are live at 8 AND 9.
            for bsize in [6usize, 9, 12] {
                assert!(
                    !hybrid_use_rdopt(hybrid, bsize, 5000),
                    "speed {speed}: bsize {bsize} must take the estimate arm"
                );
            }
            // Low-variance 8x8 takes the estimate arm at both speeds too.
            assert!(!hybrid_use_rdopt(hybrid, 3, 100), "speed {speed}: 8x8 var 100");
        }
    }

    /// **KB-34 — `nonrd_leaf_tx_size` IS `max_txsize_lookup`, derived rather
    /// than re-typed.** The table it returns is a transcription of
    /// `common_data.h:105-124`, and a second transcription would prove nothing
    /// about the first. What `max_txsize_lookup` actually *means* is "the
    /// largest SQUARE transform that fits inside the block on both axes", so
    /// that is what is checked here, against `MI_W`/`MI_H` — which the walk
    /// itself uses and which the e2e byte gates therefore also exercise.
    #[test]
    fn nonrd_leaf_tx_size_is_the_largest_square_tx_that_fits() {
        for bsize in 0..22usize {
            let (w, h) = (MI_W[bsize] * 4, MI_H[bsize] * 4);
            let tx = nonrd_leaf_tx_size(bsize);
            let side = 4usize << tx;
            assert!(
                side <= w && side <= h,
                "bsize {bsize} ({w}x{h}): TX side {side} does not fit"
            );
            // ... and it is the LARGEST such square, up to TX_64X64.
            let bigger = 2 * side;
            assert!(
                tx == 4 || bigger > w || bigger > h,
                "bsize {bsize} ({w}x{h}): TX side {side} but {bigger} also fits"
            );
            // Its bsize is the square this walk splits the leaf into.
            let tx_b = TXSIZE_TO_BSIZE[tx];
            assert_eq!((MI_W[tx_b] * 4, MI_H[tx_b] * 4), (side, side));
            // Single-txb is exactly "square AND at most 64 px" — square is
            // necessary but not sufficient, because `max_txsize_lookup` tops
            // out at TX_64X64, so BLOCK_128X128 is four txbs. The
            // pre-2026-08-02 refusal described the invariant as "square", which
            // was right about every leaf the KEY VBP tree stamps and wrong in
            // general; asserted here rather than assumed either way.
            assert_eq!(
                tx_b == bsize,
                w == h && w <= 64,
                "bsize {bsize} ({w}x{h}): single-txb must mean square and <= 64 px"
            );
        }
    }

    /// The KEY variance partitioner can stamp exactly four non-square leaves,
    /// and each one splits into exactly TWO txbs. `set_vt_partitioning` returns
    /// 0 for `bsize > BLOCK_32X32` on an intra frame (var_based_part.c:205-209)
    /// so 64X32/32X64 never reach the estimate arm, and offers only
    /// NONE-or-split at `bsize == bsize_min` (:186-199) so nothing below 8x8
    /// does either. `kb34_nonsquare_nonrd_leaf.rs` asserts the same set from
    /// the other end — over what real encodes actually produced.
    #[test]
    fn kb34_key_rect_leaves_are_two_txbs_each() {
        for bsize in [4usize, 5, 7, 8] {
            let tx_b = TXSIZE_TO_BSIZE[nonrd_leaf_tx_size(bsize)];
            let n = (MI_W[bsize] / MI_W[tx_b]) * (MI_H[bsize] / MI_H[tx_b]);
            assert_eq!(n, 2, "bsize {bsize} splits into {n} txbs, expected 2");
        }
    }

    #[test]
    fn hybrid_gate_matches_source() {
        // speed 8: hybrid=2 → threshold 101, only below 16x16.
        assert!(hybrid_use_rdopt(2, 3, 101));
        assert!(!hybrid_use_rdopt(2, 3, 100));
        assert!(!hybrid_use_rdopt(2, 6, 5000)); // 16x16: estimate arm
        assert!(!hybrid_use_rdopt(2, 9, 5000));
        // speed 9: hybrid=0 → never.
        assert!(!hybrid_use_rdopt(0, 3, 5000));
    }

    /// **KB-35 unit lock — the nonrd palette refusal is C's `try_palette`,
    /// term for term, not the frame flag alone.**
    ///
    /// Until 2026-08-03 the guard read `!allow_screen_content_tools`, one of
    /// the four terms, and refused the whole leaf on it. The rows below are
    /// the ones that separates the two forms: every `enable_palette = false`
    /// row is a cell the old guard refused and C provably never searches
    /// (`shim_encode_av1_kf` passes `--enable-palette=0`), and every
    /// `bsize`/variance row is a cell the frame flag alone cannot classify.
    #[test]
    fn palette_arm_liveness_is_c_try_palette() {
        // sad terms held constant and INERT here (prune off) so the other
        // three vary alone.
        let live = |ep, asct, bsize, var| {
            nonrd_palette_arm_is_live(ep, asct, bsize, 1, false, u32::MAX, var)
        };
        // Term 1 — `--enable-palette=0` kills it even with the frame flag on,
        // a 16x16 leaf and a high variance. This is the whole KB-35 class.
        assert!(!live(false, true, 6, 5000));
        assert!(live(true, true, 6, 5000));
        // Term 2 — `av1_allow_palette`: the frame flag, then the size bounds.
        assert!(!live(true, false, 6, 5000));
        // BLOCK_4X4 (0), BLOCK_4X8 (1), BLOCK_8X4 (2) are all < BLOCK_8X8 (3).
        for bsize in 0..3 {
            assert!(!live(true, true, bsize, 5000), "bsize {bsize} is < BLOCK_8X8");
        }
        assert!(live(true, true, 3, 5000), "BLOCK_8X8 is allowed");
        // ORDINAL, not dimensional: BLOCK_4X16 (16) and BLOCK_16X4 (17) have a
        // 4-px side and still satisfy C's `sb_type >= BLOCK_8X8` — they die on
        // the `bsize <= BLOCK_16X16` prune instead, which is also ordinal.
        assert!(!live(true, true, 16, 5000));
        assert!(!live(true, true, 17, 5000));
        // Term 3 — the speed prune's two halves, each alone.
        assert!(!live(true, true, 7, 5000), "BLOCK_16X32 is > BLOCK_16X16");
        assert!(!live(true, true, 6, 200), "variance must be STRICTLY > 200");
        assert!(live(true, true, 6, 201));
        // ... and the prune only applies at `prune_palette_search_nonrd > 0`.
        assert!(nonrd_palette_arm_is_live(true, true, 9, 0, false, u32::MAX, 0));
        assert!(!nonrd_palette_arm_is_live(true, true, 9, 1, false, u32::MAX, 0));
        // Term 4 — the SAD term, and its threshold's level dependence
        // (`> 1 ? 100 : 20`). Only bites when the per-mode SAD prune ran.
        assert!(!nonrd_palette_arm_is_live(true, true, 6, 1, true, 20, 5000));
        assert!(nonrd_palette_arm_is_live(true, true, 6, 1, true, 21, 5000));
        assert!(!nonrd_palette_arm_is_live(true, true, 6, 2, true, 100, 5000));
        assert!(nonrd_palette_arm_is_live(true, true, 6, 2, true, 101, 5000));
    }

    /// `b_log2` must reproduce `b_width_log2_lookup` / `b_height_log2_lookup`
    /// (nonrd_opt.h:114-119) on every one of the 16 `BLOCK_SIZES` entries —
    /// derived from the dimension rather than re-typed, so the check is
    /// against C's literal table.
    #[test]
    fn b_log2_matches_c_lookup_tables() {
        const B_WIDTH_LOG2: [u32; 16] = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5];
        const B_HEIGHT_LOG2: [u32; 16] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5];
        for bsize in 0..16usize {
            assert_eq!(b_log2(MI_W[bsize] * 4), B_WIDTH_LOG2[bsize], "width {bsize}");
            assert_eq!(b_log2(MI_H[bsize] * 4), B_HEIGHT_LOG2[bsize], "height {bsize}");
        }
    }
}
