//! Full-pel inter motion-search kernels that live outside the diamond/mesh
//! core — port of libaom v3.14.1 `av1/encoder/mcomp.c`.
//!
//! The NSTEP diamond and the mesh (`av1_full_pixel_search`) already live in
//! [`crate::intrabc_search`], retargeted from the current frame to a reference.
//! This module adds the two exported full-pel kernels that sit beside them:
//!
//! | Rust | C |
//! |---|---|
//! | [`refining_search_8p`] | `av1_refining_search_8p_c` (mcomp.c:1696) |
//! | [`vector_match`] | `av1_vector_match` (mcomp.c:2276) |
//!
//! # Differential coverage
//! `tests/inter_fullpel_diff.rs`, tier 1 against the real exported C.

use crate::intrabc_search::{DvCosts, FullMvLimits, mvsad_err_cost, sad_wxh};

/// `SEARCH_RANGE_8P` (`mcomp_structs.h:32`).
const SEARCH_RANGE_8P: i32 = 3;
/// `SEARCH_GRID_STRIDE_8P` (`mcomp_structs.h:33`).
const SEARCH_GRID_STRIDE_8P: i32 = 2 * SEARCH_RANGE_8P + 1;
/// `SEARCH_GRID_CENTER_8P` (`mcomp_structs.h:34`).
const SEARCH_GRID_CENTER_8P: i32 = SEARCH_RANGE_8P * SEARCH_GRID_STRIDE_8P + SEARCH_RANGE_8P;

/// The 8 neighbours `av1_refining_search_8p_c` probes, in C's order — cardinals
/// first, then diagonals. The order is load-bearing: ties are broken by
/// first-seen, so a reordering changes which MV wins.
const NEIGHBORS_8P: [((i32, i32), i32); 8] = [
    ((-1, 0), -SEARCH_GRID_STRIDE_8P),
    ((0, -1), -1),
    ((0, 1), 1),
    ((1, 0), SEARCH_GRID_STRIDE_8P),
    ((-1, -1), -SEARCH_GRID_STRIDE_8P - 1),
    ((1, -1), SEARCH_GRID_STRIDE_8P - 1),
    ((-1, 1), -SEARCH_GRID_STRIDE_8P + 1),
    ((1, 1), SEARCH_GRID_STRIDE_8P + 1),
];

/// Inputs to [`refining_search_8p`]. Planes are `u16` bd8 (`0..=255`), matching
/// the rest of the port. `ref_origin` is the reference `buf_2d` origin for a
/// zero full-pel MV; `get_buf_from_fullmv` offsets it by `mv` directly.
pub struct RefineSearch8pParams<'a> {
    /// Source block plane.
    pub src: &'a [u16],
    /// Index of the block's top-left sample in `src`.
    pub src_off: usize,
    pub src_stride: usize,
    /// Border-extended reference plane.
    pub refb: &'a [u16],
    /// Index in `refb` of the zero-MV block origin.
    pub ref_origin: usize,
    pub ref_stride: usize,
    pub w: usize,
    pub h: usize,
    pub limits: FullMvLimits,
    /// MV entropy cost tables (`mv_cost_params->mvjcost` / `mvcost`).
    pub dv: &'a DvCosts,
    /// `mv_cost_params->full_ref_mv` — the FULL-PEL predicted MV the cost is
    /// measured against.
    pub full_ref_mv: (i32, i32),
    /// `mv_cost_params->sad_per_bit`.
    pub sad_per_bit: i32,
}

/// What [`refining_search_8p`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefineSearch8pResult {
    /// C's `*best_mv`.
    pub best_mv: (i32, i32),
    /// C's return value: the winning SAD **plus** its MV cost.
    pub best_sad: u32,
}

/// `av1_refining_search_8p_c` (mcomp.c:1696): up to `SEARCH_RANGE_8P` rounds of
/// 8-neighbour full-pel refinement around `start_mv`, on the SAD metric.
///
/// Three shapes are reproduced from C. **Only the first changes results** —
/// the other two are short-circuits, and this port says so rather than
/// asserting a significance it could not demonstrate:
///
/// * **The neighbour probe ORDER is load-bearing.** Acceptance uses a strict
///   `<`, so a unique minimum is order-independent, but on an exact tie the
///   first-probed candidate wins. `refining_search_8p_breaks_ties_like_c`
///   constructs such a tie, and swapping two entries of [`NEIGHBORS_8P`] fails
///   it (verified by mutation).
/// * The visited-grid and the doubled improvement test are **optimisations, not
///   semantics.** The 7x7 grid coordinate is in bijection with the candidate MV
///   over the reachable ±3 window, so skipping a visited cell only skips an MV
///   that was already scored and lost — and since `best_sad` is monotonically
///   non-increasing it would lose again. Likewise, C's `sad < best_sad` test
///   *before* adding the MV cost cannot change the outcome, because MV costs
///   are non-negative. Both were mutation-tested: deleting either leaves the
///   differential green. They are kept because they are what C does and what
///   its cost profile assumes, not because dropping them would diverge.
///
/// `second_pred`/`mask` are not modelled (the compound arms of
/// `get_mvpred_compound_sad`); this is the single-reference `ms_params->sdf`
/// path.
#[must_use]
pub fn refining_search_8p(p: &RefineSearch8pParams, start_mv: (i32, i32)) -> RefineSearch8pResult {
    let n = (SEARCH_GRID_STRIDE_8P * SEARCH_GRID_STRIDE_8P) as usize;
    let mut visited = vec![0u8; n];
    let mut grid_center = SEARCH_GRID_CENTER_8P;

    let mut best_mv = (
        start_mv.0.clamp(p.limits.row_min, p.limits.row_max),
        start_mv.1.clamp(p.limits.col_min, p.limits.col_max),
    );

    let sad_at = |mv: (i32, i32)| -> u32 {
        let off = (p.ref_origin as isize + mv.0 as isize * p.ref_stride as isize + mv.1 as isize)
            as usize;
        sad_wxh(
            p.src,
            p.src_off,
            p.src_stride,
            p.refb,
            off,
            p.ref_stride,
            p.w,
            p.h,
        )
    };
    let cost_at = |mv: (i32, i32)| -> i32 {
        mvsad_err_cost(
            mv.0 - p.full_ref_mv.0,
            mv.1 - p.full_ref_mv.1,
            p.dv,
            p.sad_per_bit,
        )
    };

    let mut best_sad = sad_at(best_mv).wrapping_add(cost_at(best_mv) as u32);
    visited[grid_center as usize] = 1;

    for _ in 0..SEARCH_RANGE_8P {
        let mut best_site: Option<usize> = None;
        for (j, &(coord, coord_offset)) in NEIGHBORS_8P.iter().enumerate() {
            let grid_coord = grid_center + coord_offset;
            if visited[grid_coord as usize] == 1 {
                continue;
            }
            let mv = (best_mv.0 + coord.0, best_mv.1 + coord.1);
            visited[grid_coord as usize] = 1;
            let in_range = mv.1 >= p.limits.col_min
                && mv.1 <= p.limits.col_max
                && mv.0 >= p.limits.row_min
                && mv.0 <= p.limits.row_max;
            if !in_range {
                continue;
            }
            let sad = sad_at(mv);
            // C's double test: reject on the raw SAD first, then again after
            // the MV cost is added.
            if sad < best_sad {
                let sad = sad.wrapping_add(cost_at(mv) as u32);
                if sad < best_sad {
                    best_sad = sad;
                    best_site = Some(j);
                }
            }
        }
        match best_site {
            None => break,
            Some(j) => {
                let (coord, coord_offset) = NEIGHBORS_8P[j];
                best_mv = (best_mv.0 + coord.0, best_mv.1 + coord.1);
                grid_center += coord_offset;
            }
        }
    }

    RefineSearch8pResult { best_mv, best_sad }
}

/// `av1_vector_match` (mcomp.c:2276): the 1-D projection search
/// `av1_int_pro_motion_estimation` runs on row/column integral projections.
///
/// `ref` is `search_size_top + search_size_bottom + 1` positions long (each a
/// `4 << bwl`-wide window); `src` is one `4 << bwl`-wide vector. Returns
/// `(offset_relative_to_search_size_top, best_sad)`.
///
/// **`bwl` must be in `{2, 3, 4, 5}`.** That is `aom_vector_var`'s documented
/// contract (`aom_dsp/avg.c:557`) and `aom_vector_var_neon` asserts it; at
/// `bwl < 2` the NEON kernel's `width -= 8` underflows its `while (width != 0)`
/// loop and reads past the buffer. This port does not widen the contract.
///
/// `full_search` scans every position; otherwise C runs a coarse-to-fine
/// cascade — stride 16, then ±8, ±4, ±2, ±1. **The cascade is not a bisection
/// and does not always find the stride-16 minimum's true neighbourhood**: each
/// refinement round recentres on `center` but reads `offset`, which is only
/// updated at the end of the round, so a round can probe around the previous
/// round's centre. That is upstream's shape and this port keeps it.
#[must_use]
pub fn vector_match(
    reff: &[i16],
    src: &[i16],
    bwl: i32,
    search_size_top: i32,
    search_size_bottom: i32,
    full_search: bool,
) -> (i32, i32) {
    let bw = search_size_top + search_size_bottom;
    let mut best_sad = i32::MAX;
    let mut offset = 0i32;

    let var_at = |d: i32| -> i32 {
        let start = d as usize;
        let width = 4usize << bwl;
        aom_dsp::dist::vector_var(&reff[start..start + width], src, bwl)
    };

    if full_search {
        for d in 0..=bw {
            let this_sad = var_at(d);
            if this_sad < best_sad {
                best_sad = this_sad;
                offset = d;
            }
        }
        return (offset - search_size_top, best_sad);
    }

    let mut d = 0i32;
    while d <= bw {
        let this_sad = var_at(d);
        if this_sad < best_sad {
            best_sad = this_sad;
            offset = d;
        }
        d += 16;
    }
    let mut center = offset;

    // Each round reads `offset` (fixed for the whole round) and writes
    // `center`; `offset = center` only at the end. Faithful to C.
    for &(lo, step) in &[(-8i32, 16i32), (-4, 8), (-2, 4), (-1, 2)] {
        let mut d = lo;
        while d <= -lo {
            let this_pos = offset + d;
            if this_pos >= 0 && this_pos <= bw {
                let this_sad = var_at(this_pos);
                if this_sad < best_sad {
                    best_sad = this_sad;
                    center = this_pos;
                }
            }
            d += step;
        }
        offset = center;
    }

    (center - search_size_top, best_sad)
}

// ===================================================================
// The OBMC full-pel motion search — mcomp.c
//
// OBMC scores a candidate against a per-pixel WEIGHTED target (`wsrc`) with a
// per-pixel `mask`, not against the source block, so it needs its own copy of
// the diamond search rather than a mode flag on the existing one. The two
// copies also differ structurally — see `obmc_full_pixel_diamond`.
// ===================================================================

use crate::intrabc_search::SiteCfg;

/// Inputs to the OBMC full-pel search. Planes are `u16` bd8 (`0..=255`) to
/// match the rest of the port; `wsrc` and `mask` are the tight `w*h` buffers
/// `calc_target_weighted_pred` produces, both at 1/4096 precision.
pub struct ObmcFullPelParams<'a> {
    /// Border-extended reference plane.
    pub refb: &'a [u16],
    /// Index in `refb` of the zero-MV block origin.
    pub ref_origin: usize,
    pub ref_stride: usize,
    pub w: usize,
    pub h: usize,
    /// `ms_buffers->wsrc`
    pub wsrc: &'a [i32],
    /// `ms_buffers->obmc_mask`
    pub obmc_mask: &'a [i32],
    pub limits: FullMvLimits,
    /// MV entropy cost tables.
    pub dv: &'a DvCosts,
    /// `mv_cost_params->full_ref_mv`.
    pub full_ref_mv: (i32, i32),
    /// `mv_cost_params->sad_per_bit`.
    pub sad_per_bit: i32,
    /// `mv_cost_params->error_per_bit`.
    pub error_per_bit: i32,
}

impl ObmcFullPelParams<'_> {
    /// A tight `w*h` u8 copy of the reference window at `mv`, which is what the
    /// OBMC kernels take.
    fn window(&self, mv: (i32, i32)) -> Vec<u8> {
        let off = (self.ref_origin as isize
            + mv.0 as isize * self.ref_stride as isize
            + mv.1 as isize) as usize;
        let mut v = vec![0u8; self.w * self.h];
        for r in 0..self.h {
            for c in 0..self.w {
                v[r * self.w + c] = self.refb[off + r * self.ref_stride + c] as u8;
            }
        }
        v
    }

    /// `fn_ptr->osdf` — `aom_obmc_sad{W}x{H}`.
    fn osdf(&self, mv: (i32, i32)) -> u32 {
        let win = self.window(mv);
        aom_dsp::dist::obmc_sad(&win, self.w, self.wsrc, self.obmc_mask, self.w, self.h)
    }

    /// `vfp->ovf` — `aom_obmc_variance{W}x{H}`.
    fn ovf(&self, mv: (i32, i32)) -> u32 {
        let win = self.window(mv);
        aom_dsp::dist::obmc::obmc_variance(
            &win,
            0,
            self.w,
            self.wsrc,
            self.obmc_mask,
            self.w,
            self.h,
        )
        .0
    }

    /// `mvsad_err_cost_` at `MV_COST_ENTROPY`.
    fn sad_cost(&self, mv: (i32, i32)) -> i32 {
        mvsad_err_cost(
            mv.0 - self.full_ref_mv.0,
            mv.1 - self.full_ref_mv.1,
            self.dv,
            self.sad_per_bit,
        )
    }

    /// `mv_err_cost_` at `MV_COST_ENTROPY`, on the full-pel MV promoted to
    /// 1/8-pel (`get_mv_from_fullmv`).
    fn var_cost(&self, mv: (i32, i32)) -> i32 {
        crate::intrabc_search::mv_err_cost(
            mv.0 * 8 - self.full_ref_mv.0 * 8,
            mv.1 * 8 - self.full_ref_mv.1 * 8,
            self.dv,
            self.error_per_bit,
        )
    }

    fn in_range(&self, mv: (i32, i32)) -> bool {
        mv.1 >= self.limits.col_min
            && mv.1 <= self.limits.col_max
            && mv.0 >= self.limits.row_min
            && mv.0 <= self.limits.row_max
    }

    fn clamp(&self, mv: (i32, i32)) -> (i32, i32) {
        (
            mv.0.clamp(self.limits.row_min, self.limits.row_max),
            mv.1.clamp(self.limits.col_min, self.limits.col_max),
        )
    }
}

/// `get_obmc_mvpred_var` (mcomp.c:756): the OBMC variance at `mv` plus the
/// subpel-metric MV cost. C's return type is `int`.
#[must_use]
pub fn get_obmc_mvpred_var(p: &ObmcFullPelParams, mv: (i32, i32)) -> i32 {
    (p.ovf(mv) as i32).wrapping_add(p.var_cost(mv))
}

/// `obmc_diamond_search_sad` (mcomp.c:2103): one diamond pass on the OBMC SAD
/// metric. Returns `(best_sad, best_mv, num00)`.
///
/// This is **not** the single-reference `diamond_search_sad` with a different
/// metric. It has no all-sites-in-range fast path, no radius-repeat step
/// skipping, and its `num00` counts the leading steps during which the search
/// never left the start position (C spells that `best_address == init_ref`,
/// which is the same predicate because the site offset is
/// `row * stride + col`, a bijection over the reachable window).
fn obmc_diamond_search_sad(
    p: &ObmcFullPelParams,
    start_mv: (i32, i32),
    search_step: usize,
    cfg: SiteCfg,
) -> (u32, (i32, i32), i32) {
    let radii = cfg.radii();
    let tot_steps = radii.len() - search_step;
    let start_mv = p.clamp(start_mv);
    let mut best_mv = start_mv;
    let mut num00 = 0i32;
    let mut best_sad = p.osdf(start_mv).wrapping_add(p.sad_cost(start_mv) as u32);

    for step in (0..tot_steps).rev() {
        let radius = radii[step];
        let (site, num_searches) = cfg.stage_sites(radius);
        let mut best_site = 0usize;
        // Indexed rather than iterated: `best_site` is the winning INDEX, which
        // the step update then reads back out of `site`.
        #[allow(clippy::needless_range_loop)]
        for idx in 1..=num_searches {
            let (dr, dc) = site[idx];
            let mv = (best_mv.0 + dr, best_mv.1 + dc);
            if p.in_range(mv) {
                let sad = p.osdf(mv);
                if sad < best_sad {
                    let sad = sad.wrapping_add(p.sad_cost(mv) as u32);
                    if sad < best_sad {
                        best_sad = sad;
                        best_site = idx;
                    }
                }
            }
        }
        if best_site != 0 {
            best_mv = (best_mv.0 + site[best_site].0, best_mv.1 + site[best_site].1);
        } else if best_mv == start_mv {
            num00 += 1;
        }
    }
    (best_sad, best_mv, num00)
}

/// `obmc_full_pixel_diamond` (mcomp.c:2160).
///
/// The step loop differs from the single-reference `full_pixel_diamond`: `n` is
/// seeded from the FIRST diamond's `num00` output, and inside the loop a
/// separate `num00` counter *skips* iterations rather than advancing `n`. This
/// port keeps C's shape.
///
/// **Not verified by the differential.** A mutation replacing this loop with
/// the single-reference shape (advance `n` by `num00`, search every iteration)
/// left `tests/inter_fullpel_diff.rs` green. That is consistent with the skip
/// being a speed heuristic in the regime the sweep reaches — `num00` counts
/// leading steps at which the diamond never left the start, and re-running
/// those step sizes from the same start finds the same MV — but it is not a
/// proof of equivalence, so the difference is recorded as untested rather than
/// claimed. The `num00` predicate itself IS pinned: counting every non-move
/// instead of only the leading ones fails the differential.
fn obmc_full_pixel_diamond(
    p: &ObmcFullPelParams,
    start_mv: (i32, i32),
    step_param: usize,
    cfg: SiteCfg,
) -> (i32, (i32, i32)) {
    let (sad0, tmp_mv, mut n) = obmc_diamond_search_sad(p, start_mv, step_param, cfg);
    let mut bestsme = if sad0 < u32::MAX {
        get_obmc_mvpred_var(p, tmp_mv)
    } else {
        i32::MAX
    };
    let mut best_mv = tmp_mv;

    let further_steps = cfg.radii().len() as i32 - 1 - step_param as i32;
    let mut num00 = 0i32;
    while n < further_steps {
        n += 1;
        if num00 != 0 {
            num00 -= 1;
        } else {
            let (thissad, tmp_mv, n00) =
                obmc_diamond_search_sad(p, start_mv, step_param + n as usize, cfg);
            num00 = n00;
            let thissme = if thissad < u32::MAX {
                get_obmc_mvpred_var(p, tmp_mv)
            } else {
                i32::MAX
            };
            if thissme < bestsme {
                bestsme = thissme;
                best_mv = tmp_mv;
            }
        }
    }
    (bestsme, best_mv)
}

/// `obmc_refining_search_sad` (mcomp.c:2064): up to 8 rounds of 4-neighbour
/// refinement on the OBMC SAD metric. Returns `(best_sad, best_mv)`.
fn obmc_refining_search_sad(p: &ObmcFullPelParams, start_mv: (i32, i32)) -> (u32, (i32, i32)) {
    const NEIGHBORS: [(i32, i32); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];
    const K_SEARCH_RANGE: usize = 8;
    let mut best_mv = start_mv;
    let mut best_sad = p.osdf(best_mv).wrapping_add(p.sad_cost(best_mv) as u32);
    for _ in 0..K_SEARCH_RANGE {
        let mut best_site: Option<usize> = None;
        for (j, &(dr, dc)) in NEIGHBORS.iter().enumerate() {
            let mv = (best_mv.0 + dr, best_mv.1 + dc);
            if p.in_range(mv) {
                let sad = p.osdf(mv);
                if sad < best_sad {
                    let sad = sad.wrapping_add(p.sad_cost(mv) as u32);
                    if sad < best_sad {
                        best_sad = sad;
                        best_site = Some(j);
                    }
                }
            }
        }
        match best_site {
            None => break,
            Some(j) => best_mv = (best_mv.0 + NEIGHBORS[j].0, best_mv.1 + NEIGHBORS[j].1),
        }
    }
    (best_sad, best_mv)
}

/// `av1_obmc_full_pixel_search` (mcomp.c:2202).
///
/// `fast_obmc_search` picks between the diamond (false) and a clamped
/// 4-neighbour refinement whose SAD result is then **re-scored** with the
/// variance metric (true). The two arms therefore return values on different
/// metrics unless the refinement runs, which is why C re-scores.
///
/// The search-site configuration is NSTEP, which is what
/// `av1_set_mv_search_method` installs for the OBMC search on the GOOD ladder.
/// Returns `(bestsme, best_mv)`.
#[must_use]
pub fn obmc_full_pixel_search(
    p: &ObmcFullPelParams,
    start_mv: (i32, i32),
    step_param: usize,
    fast_obmc_search: bool,
) -> (i32, (i32, i32)) {
    let cfg = SiteCfg::Nstep;
    if !fast_obmc_search {
        obmc_full_pixel_diamond(p, start_mv, step_param, cfg)
    } else {
        let best_mv = p.clamp(start_mv);
        let (thissme, best_mv) = obmc_refining_search_sad(p, best_mv);
        let thissme = if thissme < u32::MAX {
            get_obmc_mvpred_var(p, best_mv)
        } else {
            i32::MAX
        };
        (thissme, best_mv)
    }
}
