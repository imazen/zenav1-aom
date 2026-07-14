//! `av1_fill_coeff_costs` (libaom `av1/encoder/rd.c`), per (`txs_ctx`, `plane`):
//! assemble the `LV_MAP_COEFF_COST` tables from a frame's coefficient CDFs, so
//! `cost_coeffs_txb` runs from real adaptive CDFs. Uses the bit-exact
//! `cost_tokens_from_cdf`; the new logic verified here is the `base_cost[4..7]`
//! trellis-diff and the `lps_cost` cumulation + diff fixups.

use crate::cost_tokens_from_cdf;
use crate::cost::CoeffCostTables;

const COST_LIT1: i32 = 1 << 9;
const COEFF_BASE_RANGE: usize = 12;

/// Owned `LV_MAP_COEFF_COST` for one (`txs_ctx`, `plane`), flat row-major with
/// the same strides `cost_coeffs_txb` / `CoeffCostTables` expect.
pub struct LvMapCoeffCost {
    /// `txb_skip[13][2]`
    pub txb_skip: Vec<i32>,
    /// `base_eob[4][3]`
    pub base_eob: Vec<i32>,
    /// `base[42][8]`
    pub base: Vec<i32>,
    /// `eob_extra[9][2]`
    pub eob_extra: Vec<i32>,
    /// `dc_sign[3][2]`
    pub dc_sign: Vec<i32>,
    /// `lps[21][26]`
    pub lps: Vec<i32>,
}

impl LvMapCoeffCost {
    /// Borrow as `CoeffCostTables` for `cost_coeffs_txb` (caller supplies eob).
    pub fn tables<'a>(&'a self, eob: &'a [i32]) -> CoeffCostTables<'a> {
        CoeffCostTables {
            txb_skip: &self.txb_skip,
            base_eob: &self.base_eob,
            base: &self.base,
            eob_extra: &self.eob_extra,
            dc_sign: &self.dc_sign,
            lps: &self.lps,
            eob,
        }
    }
}

/// Fill the tables from the 6 coeff CDF groups already selected for this
/// (`txs_ctx`, `plane`): `txb_skip_cdf[13][3]`, `base_eob_cdf[4][4]`,
/// `base_cdf[42][5]`, `eob_extra_cdf[9][3]`, `dc_sign_cdf[3][3]`,
/// `br_cdf[21][5]` (each `[ctx][CDF_SIZE(n)]`).
#[allow(clippy::too_many_arguments)]
pub fn fill_lv_map_coeff_cost(
    txb_skip_cdf: &[u16],
    base_eob_cdf: &[u16],
    base_cdf: &[u16],
    eob_extra_cdf: &[u16],
    dc_sign_cdf: &[u16],
    br_cdf: &[u16],
) -> LvMapCoeffCost {
    let mut txb_skip = vec![0i32; 13 * 2];
    for ctx in 0..13 {
        cost_tokens_from_cdf(&mut txb_skip[ctx * 2..ctx * 2 + 2], &txb_skip_cdf[ctx * 3..ctx * 3 + 3], None);
    }
    let mut base_eob = vec![0i32; 4 * 3];
    for ctx in 0..4 {
        cost_tokens_from_cdf(&mut base_eob[ctx * 3..ctx * 3 + 3], &base_eob_cdf[ctx * 4..ctx * 4 + 4], None);
    }
    let mut base = vec![0i32; 42 * 8];
    for ctx in 0..42 {
        // cost_tokens fills [0..3]; leave [4..7] for the fixup.
        let mut tmp = [0i32; 4];
        cost_tokens_from_cdf(&mut tmp, &base_cdf[ctx * 5..ctx * 5 + 5], None);
        base[ctx * 8..ctx * 8 + 4].copy_from_slice(&tmp);
    }
    for ctx in 0..42 {
        let b = ctx * 8;
        base[b + 4] = 0;
        base[b + 5] = base[b + 1] + COST_LIT1 - base[b];
        base[b + 6] = base[b + 2] - base[b + 1];
        base[b + 7] = base[b + 3] - base[b + 2];
    }
    let mut eob_extra = vec![0i32; 9 * 2];
    for ctx in 0..9 {
        cost_tokens_from_cdf(&mut eob_extra[ctx * 2..ctx * 2 + 2], &eob_extra_cdf[ctx * 3..ctx * 3 + 3], None);
    }
    let mut dc_sign = vec![0i32; 3 * 2];
    for ctx in 0..3 {
        cost_tokens_from_cdf(&mut dc_sign[ctx * 2..ctx * 2 + 2], &dc_sign_cdf[ctx * 3..ctx * 3 + 3], None);
    }
    let stride = (COEFF_BASE_RANGE + 1) * 2; // 26
    let mut lps = vec![0i32; 21 * stride];
    for ctx in 0..21 {
        let base_off = ctx * stride;
        let mut br_rate = [0i32; 4];
        cost_tokens_from_cdf(&mut br_rate, &br_cdf[ctx * 5..ctx * 5 + 5], None);
        let mut prev_cost = 0;
        let mut i = 0;
        while i < COEFF_BASE_RANGE {
            for j in 0..3 {
                // BR_CDF_SIZE - 1
                lps[base_off + i + j] = prev_cost + br_rate[j];
            }
            prev_cost += br_rate[3];
            i += 3;
        }
        lps[base_off + i] = prev_cost;
        lps[base_off + COEFF_BASE_RANGE + 1] = lps[base_off];
        for i in 1..=COEFF_BASE_RANGE {
            lps[base_off + i + COEFF_BASE_RANGE + 1] = lps[base_off + i] - lps[base_off + i - 1];
        }
    }
    LvMapCoeffCost { txb_skip, base_eob, base, eob_extra, dc_sign, lps }
}

/// [`fill_lv_map_coeff_cost`] sourced directly from a live coefficient-CDF
/// arena (the same flat `[u16; CDF_ARENA_LEN]` layout
/// [`crate::write_coeffs_txb`] reads/adapts, e.g. `KfFrameContext::coeff`) —
/// the region offsets/strides mirror `write.rs`'s `A_TXB_SKIP` / `A_BASE_EOB`
/// / `A_BASE` / `A_EOB_EXTRA` / `A_DC_SIGN` / `A_BR` exactly, so this slices
/// the SAME bytes the entropy coder is adapting, real per-(`txs_ctx`,
/// `plane_type`) cost tables (the encoder-gate wiring `av1_fill_coeff_costs`
/// needs, vs. the synthetic-but-valid random tables used for pack-glue-only
/// verification). `txs_ctx` is [`crate::txsize_entropy_ctx`]`(tx_size)`
/// (0..=4); `plane_type` is 0 (luma) or 1 (chroma).
pub fn fill_lv_map_coeff_cost_from_arena(arena: &[u16], txs_ctx: usize, plane_type: usize) -> LvMapCoeffCost {
    use crate::write::{A_BASE, A_BASE_EOB, A_BR, A_DC_SIGN, A_EOB_EXTRA, A_TXB_SKIP};
    debug_assert!(txs_ctx < 5 && plane_type < 2);
    let pt = txs_ctx * 2 + plane_type;
    let txb_skip_cdf = &arena[A_TXB_SKIP + txs_ctx * 13 * 3..A_TXB_SKIP + txs_ctx * 13 * 3 + 13 * 3];
    let base_eob_cdf = &arena[A_BASE_EOB + pt * 4 * 4..A_BASE_EOB + pt * 4 * 4 + 4 * 4];
    let base_cdf = &arena[A_BASE + pt * 42 * 5..A_BASE + pt * 42 * 5 + 42 * 5];
    let eob_extra_cdf = &arena[A_EOB_EXTRA + pt * 9 * 3..A_EOB_EXTRA + pt * 9 * 3 + 9 * 3];
    let dc_sign_cdf = &arena[A_DC_SIGN + plane_type * 3 * 3..A_DC_SIGN + plane_type * 3 * 3 + 3 * 3];
    let br_cdf = &arena[A_BR + pt * 21 * 5..A_BR + pt * 21 * 5 + 21 * 5];
    fill_lv_map_coeff_cost(txb_skip_cdf, base_eob_cdf, base_cdf, eob_extra_cdf, dc_sign_cdf, br_cdf)
}
