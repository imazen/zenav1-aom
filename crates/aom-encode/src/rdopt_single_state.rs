//! The SINGLE-REFERENCE STATE table of libaom's inter RD brain
//! (`av1/encoder/rdopt.c`): the per-direction, per-mode ranking of reference
//! frames that the compound search uses to decide which compound pairs are
//! worth evaluating at all, plus the small initialisers around it.
//!
//! Tier 1c throughout (every function is `static`; the oracle is libaom's own
//! rdopt.c compiled into the shim archive — see
//! `crates/aom-sys-ref/shim/rdopt_shim.c`). Gate:
//! `crates/aom-encode/tests/rdopt_single_state_diff.rs`.
//!
//! | Rust | C (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | [`SingleStates::init`] | `init_single_inter_mode_search_state` `:4465` |
//! | [`SingleStates::collect`] | `collect_single_states` `:4813` |
//! | [`SingleStates::analyze`] | `analyze_single_states` `:4859` |
//! | [`SingleStates::candidates`] | `compound_skip_get_candidates` `:4948` |
//! | [`skip_repeated_mv`] | `:1238` |
//! | [`init_comp_avg_est_rd`] | `:516` |
//! | [`init_top_tx_no_split_rd_for_inter_modes`] | `:5940` |
//! | [`inter_modes_info_push`] | `:468` |
//! | [`increase_motion_mode_rd`] | `:1442` |
//! | [`skip_interp_filter_search`] | `:6060` |
//!
//! # How the table is used
//!
//! Each single-reference candidate that finishes RD calls [`SingleStates::collect`],
//! which inserts it into two insertion-sorted lists per (direction, mode): one
//! keyed on the real RD and one on the modelled RD. After the single-reference
//! pass, [`SingleStates::analyze`] marks entries more than `prune_factor / 8`
//! worse than the best NEWMV-or-GLOBALMV entry invalid, then merges the two
//! lists into `single_rd_order` — simple-RD order first, modelled-RD order
//! filling in. The compound pass then only evaluates pairs whose halves appear
//! in the first [`SingleStates::candidates`] entries of that order.
//!
//! # Translation notes
//!
//! - **The two parallel lists are one type used twice.** C spells
//!   `single_state` / `single_state_modelled` (and their two counts) as four
//!   separate members and then writes the same insertion sort and the same
//!   prune loop twice over each. Here they are two [`StateList`]s and the
//!   shared code is written once, which is why a divergence between the two
//!   copies is not expressible.
//! - **`valid` is a `bool`.** C stores it as `int` and only ever assigns 0/1.
//! - **`ref_frame` is an `Option<i32>`** where C uses `NONE_FRAME == -1`.

use crate::rdopt_mv::{Mv, PredMode, RefMvRow, get_drl_refmv_count};

/// `SINGLE_INTER_MODE_NUM` (`enums.h:359`): NEARESTMV, NEARMV, GLOBALMV, NEWMV.
pub const SINGLE_INTER_MODE_NUM: usize = 4;
/// `FWD_REFS`.
pub const FWD_REFS: usize = 4;
/// `INTER_OFFSET(mode)` (`entropymode.h:28`).
pub fn inter_offset(mode: PredMode) -> usize {
    (mode.to_i32() - PredMode::NearestMv.to_i32()) as usize
}

/// One entry of a `SingleInterModeState` list (rdopt.c:305).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SingleState {
    /// `rd`; `i64::MAX` means "not measured".
    pub rd: i64,
    /// `ref_frame`; `None` is C's `NONE_FRAME`.
    pub ref_frame: Option<i32>,
    /// `valid`.
    pub valid: bool,
}

impl Default for SingleState {
    fn default() -> Self {
        Self {
            rd: i64::MAX,
            ref_frame: None,
            valid: false,
        }
    }
}

/// One `[dir][mode]` insertion-sorted list plus its count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct StateList {
    /// The entries, sorted ascending by `rd`.
    pub entries: [SingleState; FWD_REFS],
    /// How many of them are populated.
    pub count: usize,
}

impl StateList {
    /// C's insertion sort: shift entries with a LARGER rd up by one and drop
    /// the new one into the hole. Equal RDs keep insertion order, because the
    /// shift condition is strictly greater.
    ///
    /// # Capacity
    ///
    /// C has NO bound check here: `single_state[dir][mode]` is `FWD_REFS`
    /// entries and `single_state_cnt` is incremented unconditionally, so a
    /// caller that filed more than four candidates into one (direction, mode)
    /// would write past the array. The encoder cannot: it collects each
    /// (direction, mode, reference) triple at most once, and each direction
    /// has at most `FWD_REFS` references. The port asserts that contract
    /// rather than reproducing the overflow.
    fn insert(&mut self, state: SingleState) {
        assert!(
            self.count < FWD_REFS,
            "more than FWD_REFS candidates filed into one (direction, mode)              list — C would write past single_state[dir][mode] here"
        );
        let mut j = self.count;
        while j > 0 && self.entries[j - 1].rd > state.rd {
            self.entries[j] = self.entries[j - 1];
            j -= 1;
        }
        self.entries[j] = state;
        self.count += 1;
    }
}

/// The single-reference half of `InterModeSearchState` (rdopt.c:311).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SingleStates {
    /// `single_state` / `single_state_cnt` — ranked by the real RD.
    pub simple: [[StateList; SINGLE_INTER_MODE_NUM]; 2],
    /// `single_state_modelled` / `_cnt` — ranked by the modelled RD.
    pub modelled: [[StateList; SINGLE_INTER_MODE_NUM]; 2],
    /// `single_rd_order[dir][mode][i]`; `None` is `NONE_FRAME`.
    pub order: [[[Option<i32>; FWD_REFS]; SINGLE_INTER_MODE_NUM]; 2],
}

impl SingleStates {
    /// `init_single_inter_mode_search_state` (rdopt.c:4465), for the fields
    /// this type holds.
    ///
    /// C also clears `best_single_rd` / `best_single_mode`, which live
    /// elsewhere in the port ([`crate::rdopt_skip::update_best_single_mode`]'s
    /// two arrays) and are initialised by their owner.
    pub fn init() -> Self {
        Self::default()
    }

    /// `collect_single_states` (rdopt.c:4813): file one finished
    /// single-reference candidate into both lists.
    ///
    /// `simple_rd` / `modelled_rd` are `search_state->{simple,modelled}_rd
    /// [this_mode][*][ref_frame]`; only the first `ref_set` of them are read,
    /// where `ref_set` is [`get_drl_refmv_count`].
    pub fn collect(
        &mut self,
        this_mode: PredMode,
        ref_frame: i32,
        row: &RefMvRow,
        simple_rd: &[i64],
        modelled_rd: &[i64],
    ) {
        // GOLDEN_FRAME and below are the forward references.
        let dir = usize::from(ref_frame > crate::rdopt_mv::GOLDEN_FRAME);
        let mode_offset = inter_offset(this_mode);
        let ref_set = get_drl_refmv_count(this_mode, row.count);

        let best = |v: &[i64]| v[..ref_set].iter().copied().min().unwrap_or(i64::MAX);
        self.simple[dir][mode_offset].insert(SingleState {
            rd: best(simple_rd),
            ref_frame: Some(ref_frame),
            valid: true,
        });
        self.modelled[dir][mode_offset].insert(SingleState {
            rd: best(modelled_rd),
            ref_frame: Some(ref_frame),
            valid: true,
        });
    }

    /// `analyze_single_states` (rdopt.c:4859): invalidate the far-from-best
    /// entries, then merge the two rankings into `order`.
    ///
    /// `prune_level` is `sf.inter_sf.prune_comp_search_by_single_result`; C
    /// asserts it is at least 1.
    pub fn analyze(&mut self, prune_level: i32) {
        debug_assert!(prune_level >= 1);
        // `>> 3` then `* prune_factor` is C's; the shift comes FIRST, so this
        // is not `rd * factor / 8` and rounds differently.
        let prune_factor = if prune_level >= 2 { 6 } else { 5 };
        for dir in 0..2 {
            for lists in [&mut self.simple[dir], &mut self.modelled[dir]] {
                // The yardstick is the best of NEWMV and GLOBALMV, because
                // NEARESTMV/NEARMV can legitimately differ in MV and so are
                // not comparable; each mode's own best is always kept.
                let best_rd = lists[inter_offset(PredMode::NewMv)].entries[0]
                    .rd
                    .min(lists[inter_offset(PredMode::GlobalMv)].entries[0].rd);
                for list in lists.iter_mut() {
                    // `1..count` and not `1..`: C's loop starts at 1 (each
                    // mode's own best is always kept) and stops at the count,
                    // and a count of 0 must make it empty rather than panic.
                    let count = list.count;
                    for e in list.entries.iter_mut().take(count).skip(1) {
                        if e.rd != i64::MAX && (e.rd >> 3) * prune_factor > best_rd {
                            e.valid = false;
                        }
                    }
                }
            }
        }

        for dir in 0..2 {
            for mode in 0..SINGLE_INTER_MODE_NUM {
                let simple = self.simple[dir][mode];
                let modelled = self.modelled[dir][mode];
                let max_candidates = simple.count.max(modelled.count);
                let order = &mut self.order[dir][mode];
                let mut count = 0;

                for e in simple.entries[..simple.count].iter() {
                    if e.rd == i64::MAX {
                        break;
                    }
                    if e.valid {
                        order[count] = e.ref_frame;
                        count += 1;
                    }
                }
                if count >= max_candidates {
                    continue;
                }
                for e in modelled.entries[..modelled.count].iter() {
                    if count >= max_candidates {
                        break;
                    }
                    if e.rd == i64::MAX {
                        break;
                    }
                    if !e.valid {
                        continue;
                    }
                    if order[..count].contains(&e.ref_frame) {
                        continue;
                    }
                    // A reference the simple-RD pass invalidated stays out,
                    // even though the modelled pass kept it. C finds the FIRST
                    // matching entry and takes its `valid`; a reference absent
                    // from the simple list is admitted.
                    let simple_valid = simple.entries[..simple.count]
                        .iter()
                        .find(|s| s.ref_frame == e.ref_frame)
                        .is_none_or(|s| s.valid);
                    if simple_valid {
                        order[count] = e.ref_frame;
                        count += 1;
                    }
                }
            }
        }
    }

    /// `compound_skip_get_candidates` (rdopt.c:4948): how many entries of
    /// `single_rd_order[dir][mode]` the compound search may use.
    pub fn candidates(&self, prune_level: i32, dir: usize, mode: PredMode) -> usize {
        let mode_offset = inter_offset(mode);
        let max_candidates = self.order[dir][mode_offset]
            .iter()
            .position(Option::is_none)
            .unwrap_or(FWD_REFS);

        let mut candidates = max_candidates;
        if prune_level >= 2 {
            candidates = candidates.min(2);
        }
        if prune_level >= 3 {
            let s = &self.simple[dir][mode_offset].entries[0];
            let m = &self.modelled[dir][mode_offset].entries[0];
            if s.rd != i64::MAX && m.rd != i64::MAX && s.ref_frame == m.ref_frame {
                candidates = 1;
            }
            if matches!(mode, PredMode::NearMv | PredMode::GlobalMv) {
                candidates = 1;
            }
        }
        if prune_level >= 4 {
            candidates = candidates.min(1);
        }
        candidates
    }
}

/// `compound_skip_by_single_states` (rdopt.c:4982): skip a compound pair
/// whose halves both behave exactly like their single-reference searches did,
/// and whose reference did not make that mode's shortlist.
///
/// The gate has three parts, all of which must hold for a half before it can
/// veto: the reference was actually SEARCHED as a single reference in that
/// mode; the compound MV it would use MATCHES the single one at every DRL
/// index (so the compound evaluation would learn nothing new); and the
/// reference is outside the first [`SingleStates::candidates`] entries of that
/// mode's ranking. Only NEARESTMV / NEARMV halves are checked — a NEWMV half
/// re-searches and can differ.
///
/// `rows` supplies the three `mbmi_ext` rows C reads: `rows.compound` for the
/// pair and `rows.single[i]` for each half on its own. `global_mvs` is indexed
/// by reference frame.
pub fn compound_skip_by_single_states(
    states: &SingleStates,
    prune_level: i32,
    this_mode: PredMode,
    refs: [i32; 2],
    rows: &CompoundRows<'_>,
    global_mvs: &[Mv; 8],
) -> bool {
    use crate::rdopt_mv::{compound_ref0_mode, compound_ref1_mode, get_this_mv};
    let Some(mode0) = compound_ref0_mode(this_mode) else {
        return false;
    };
    let Some(mode1) = compound_ref1_mode(this_mode) else {
        return false;
    };
    let modes = [mode0, mode1];
    let dirs = [
        usize::from(refs[0] > crate::rdopt_mv::GOLDEN_FRAME),
        usize::from(refs[1] > crate::rdopt_mv::GOLDEN_FRAME),
    ];

    let mut ref_searched = [false; 2];
    for i in 0..2 {
        let list = &states.simple[dirs[i]][inter_offset(modes[i])];
        ref_searched[i] = list.entries[..list.count]
            .iter()
            .any(|e| e.ref_frame == Some(refs[i]));
    }

    // NOTE the DRL count comes from the COMPOUND row, and is then used to walk
    // the SINGLE rows' stacks as well.
    let ref_set = get_drl_refmv_count(this_mode, rows.compound.count);

    let mut ref_mv_match = [true; 2];
    for i in 0..2 {
        if !ref_searched[i] || !matches!(modes[i], PredMode::NearestMv | PredMode::NearMv) {
            continue;
        }
        for ref_mv_idx in 0..ref_set {
            // `skip_repeated_ref_mv = 0`, so neither call can decline.
            let single_mv = get_this_mv(
                modes[i],
                0,
                ref_mv_idx,
                false,
                rows.single[i],
                global_mvs[refs[i] as usize],
            );
            let comp_mv = get_this_mv(
                this_mode,
                i,
                ref_mv_idx,
                false,
                rows.compound,
                global_mvs[refs[i] as usize],
            );
            if single_mv != comp_mv {
                ref_mv_match[i] = false;
                break;
            }
        }
    }

    for i in 0..2 {
        if !ref_searched[i] || !ref_mv_match[i] {
            continue;
        }
        let candidates = states.candidates(prune_level, dirs[i], modes[i]);
        let order = &states.order[dirs[i]][inter_offset(modes[i])];
        if !order[..candidates].contains(&Some(refs[i])) {
            return true;
        }
    }
    false
}

/// The three `mbmi_ext` rows [`compound_skip_by_single_states`] reads.
pub struct CompoundRows<'a> {
    /// The row `av1_ref_frame_type([rf0, rf1])` selects.
    pub compound: &'a RefMvRow,
    /// The rows `rf0` and `rf1` select on their own.
    pub single: [&'a RefMvRow; 2],
}

/// `skip_repeated_mv` (rdopt.c:1238): drop a single-reference candidate whose
/// motion vector is identical to a CHEAPER mode's, carrying that mode's
/// modelled RD across so the caller still sees a number.
///
/// Returns `true` to skip. `modelled_rd` is
/// `search_state->modelled_rd[*][0][ref_frame[0]]`, in/out — the carry-across
/// write is the whole reason this cannot be a pure predicate.
///
/// `gm_wmtype_is_translational` is `cm->global_motion[rf0].wmtype <= TRANSLATION`.
pub fn skip_repeated_mv(
    this_mode: PredMode,
    is_comp_pred: bool,
    ref_mv_count: usize,
    gm_wmtype_is_translational: bool,
    mode_context: i32,
    costs: &crate::inter_costs::InterModeCosts,
    modelled_rd: &mut [i64; 25],
) -> bool {
    if is_comp_pred {
        return false;
    }
    let compare_mode = match this_mode {
        PredMode::NearMv => {
            if ref_mv_count == 0 {
                // NEARMV has the same MV as NEARESTMV.
                Some(PredMode::NearestMv)
            } else if ref_mv_count == 1 && gm_wmtype_is_translational {
                // ...and the same as GLOBALMV.
                Some(PredMode::GlobalMv)
            } else {
                None
            }
        }
        PredMode::GlobalMv => {
            if ref_mv_count == 0 && gm_wmtype_is_translational {
                Some(PredMode::NearestMv)
            } else if ref_mv_count == 1 {
                Some(PredMode::NearMv)
            } else {
                None
            }
        }
        _ => None,
    };
    let Some(compare_mode) = compare_mode else {
        return false;
    };
    // `modelled_rd != INT64_MAX` is how C asks "was that mode searched?".
    if modelled_rd[compare_mode.to_i32() as usize] == i64::MAX {
        return false;
    }
    let compare_cost = crate::inter_costs::cost_mv_ref(costs, compare_mode.to_i32(), mode_context);
    let this_cost = crate::inter_costs::cost_mv_ref(costs, this_mode.to_i32(), mode_context);
    if this_cost <= compare_cost {
        return false;
    }
    modelled_rd[this_mode.to_i32() as usize] = modelled_rd[compare_mode.to_i32() as usize];
    true
}

/// `init_comp_avg_est_rd` (rdopt.c:516): reset the compound-average estimated
/// RD ring, but ONLY when the speed feature that reads it is on — at level 0
/// the buffer is left alone, which is observable.
pub fn init_comp_avg_est_rd(buf: &mut [i64], skip_cmp_using_top_cmp_avg_est_rd_lvl: i32) {
    if skip_cmp_using_top_cmp_avg_est_rd_lvl == 0 {
        return;
    }
    buf.fill(i64::MAX);
}

/// `init_top_tx_no_split_rd_for_inter_modes` (rdopt.c:5940). Same
/// level-gated shape as [`init_comp_avg_est_rd`].
pub fn init_top_tx_no_split_rd_for_inter_modes(
    buf: &mut [i64],
    prune_inter_tx_split_rd_eval_lvl: i32,
) {
    if prune_inter_tx_split_rd_eval_lvl == 0 {
        return;
    }
    buf.fill(i64::MAX);
}

/// One candidate recorded by [`inter_modes_info_push`] for the later full
/// transform search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterModeInfoEntry {
    /// `mode_rate_arr[i]`.
    pub mode_rate: i32,
    /// `sse_arr[i]`.
    pub sse: i64,
    /// `est_rd_arr[i]` — the key [`crate::rdopt_skip::inter_modes_info_sort`]
    /// ranks on.
    pub est_rd: i64,
}

/// `MAX_INTER_MODES` (`encoder.h`).
pub const MAX_INTER_MODES: usize = 1024;

/// `inter_modes_info_push` (rdopt.c:468): append one candidate.
///
/// C's `mbmi_arr` / `rd_cost*_arr` copies are not modelled here — the port
/// carries those in its own candidate type — so this is the scalar columns
/// plus the count, which are what the sort and the later re-scoring read.
/// C asserts `num < MAX_INTER_MODES`; the port returns `false` instead of
/// writing past the end.
pub fn inter_modes_info_push(
    list: &mut Vec<InterModeInfoEntry>,
    entry: InterModeInfoEntry,
) -> bool {
    if list.len() >= MAX_INTER_MODES {
        return false;
    }
    list.push(entry);
    true
}

/// `MOTION_MODE` (`enums.h`).
pub const SIMPLE_TRANSLATION: i32 = 0;
/// `OBMC_CAUSAL`.
pub const OBMC_CAUSAL: i32 = 1;
/// `WARPED_CAUSAL`.
pub const WARPED_CAUSAL: i32 = 2;

/// `increase_motion_mode_rd` (rdopt.c:1442): bias warp and OBMC RD upward, to
/// trade a little compression for cheaper decoding.
///
/// Both RDs are in/out and BOTH are scaled — C biases the incumbent as well as
/// the challenger, so this is not "penalise the candidate", it is "compare
/// both at their decode-cost-adjusted values". Either being `INT64_MAX` makes
/// the whole call a no-op.
///
/// The scale factors are percentages: an `int` one for warp and a `float` one
/// for OBMC, each divided by 100.0 into an `f64`. That asymmetry is C's.
pub fn increase_motion_mode_rd(
    best_motion_mode: i32,
    this_motion_mode: i32,
    best_scaled_rd: &mut i64,
    this_scaled_rd: &mut i64,
    rd_warp_bias_scale_pct: i32,
    rd_obmc_bias_scale_pct: f32,
) {
    if *best_scaled_rd == i64::MAX || *this_scaled_rd == i64::MAX {
        return;
    }
    let warp = f64::from(rd_warp_bias_scale_pct) / 100.0;
    // C: `rd_obmc_bias_scale_pct / 100.0` where the numerator is a float and
    // the denominator a double, so the float is promoted first.
    let obmc = f64::from(rd_obmc_bias_scale_pct) / 100.0;
    let bias = |mode: i32, rd: &mut i64| {
        let scale = if mode == WARPED_CAUSAL {
            warp
        } else if mode == OBMC_CAUSAL {
            obmc
        } else {
            return;
        };
        *rd += (scale * *rd as f64) as i64;
    };
    bias(best_motion_mode, best_scaled_rd);
    bias(this_motion_mode, this_scaled_rd);
}

/// `MODE` (`enc_enums.h:270`).
pub const MODE_GOOD: i32 = 0;
/// `REALTIME`.
pub const MODE_REALTIME: i32 = 1;
/// `SINGLE_REFERENCE` (`enums.h`).
pub const SINGLE_REFERENCE: i32 = 0;

/// `skip_interp_filter_search` (rdopt.c:6060).
///
/// The two encoding modes ask different questions: REALTIME skips whenever the
/// frame is single-reference and either interp speed feature is on, GOOD skips
/// per-candidate for single-prediction modes only. ALLINTRA never gets here.
pub fn skip_interp_filter_search(
    encoding_mode: i32,
    reference_mode: i32,
    sf_skip_interp_filter_search: bool,
    winner_mode_ifs: bool,
    is_single_pred: bool,
) -> bool {
    if encoding_mode == MODE_REALTIME {
        return reference_mode == SINGLE_REFERENCE
            && (sf_skip_interp_filter_search || winner_mode_ifs);
    }
    if encoding_mode == MODE_GOOD {
        return sf_skip_interp_filter_search && is_single_pred;
    }
    false
}

/// Unused import guard: [`Mv`] is re-exported for callers assembling a
/// [`RefMvRow`] to pass to [`SingleStates::collect`].
pub type CollectMv = Mv;
