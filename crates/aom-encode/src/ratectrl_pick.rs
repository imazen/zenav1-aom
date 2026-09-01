//! The qindex decision's outer layers — `av1_rc_pick_q_and_bounds` and the two
//! paths below it that [`crate::ratectrl`]'s `AOM_Q` leaf does not cover.
//!
//! [`crate::ratectrl::pick_q_and_bounds_q_mode`] is the leaf every `AOM_Q`
//! frame ends at. This module is everything above and beside it: the general
//! `rc_pick_q_and_bounds` (which is what `AOM_Q` reaches, and which routes to
//! the leaf), the one-pass no-stats path an `ARF_UPDATE` frame takes even
//! under `AOM_Q`, and the exported dispatcher that chooses between them.
//!
//! | Rust | C (`av1/encoder/ratectrl.c`) |
//! |---|---|
//! | [`active_worst_quality_no_stats_vbr`] | `calc_active_worst_quality_no_stats_vbr` (:1225, static) |
//! | [`adjust_active_best_and_worst_quality`] | `adjust_active_best_and_worst_quality` (:1921, static) |
//! | [`get_q`] | `get_q` (:2005, static) |
//! | [`pick_q_and_bounds_no_stats`] | `rc_pick_q_and_bounds_no_stats` (:1588, static) |
//! | [`pick_q_and_bounds`] | `rc_pick_q_and_bounds` (:2188, static) |
//! | [`PickQRoute::of`] / [`pick_q_and_bounds_dispatch`] | `av1_rc_pick_q_and_bounds` (:2350) |
//!
//! # Build-config facts this module is written against, all checked
//! * `CONFIG_FPMT_TEST` is **0** in `upstream/build/config/aom_config.h:40`, so
//!   every `simulate_parallel_frame` read resolves to the plain field and the
//!   `temp_*` shadow copies do not exist. The `#else` arm is the one that
//!   compiles — including `adjust_active_best_and_worst_quality`'s, which is
//!   `extend_minq / 4` and NOT the `/1` and `/2` split the `#if` arm uses.
//! * `STRICT_RC` is commented out (`ratectrl.h:55`), so the `#ifndef STRICT_RC`
//!   qdelta block DOES compile and the two `#ifdef STRICT_RC` overrides do not.
//! * `USE_UNRESTRICTED_Q_IN_CQ_MODE` is **0** (`ratectrl.c:42`), so
//!   `rc_pick_q_and_bounds_no_stats_cq` is dead code and the `AOM_CQ` arm of
//!   the dispatcher never fires.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/ratectrl_pick_diff.rs`. **Tier 1** for the
//! exported `av1_rc_pick_q_and_bounds`, **tier 1c** for the four statics.

use crate::rate_model::{compute_qdelta, convert_qindex_to_q};
use crate::ratectrl::{
    FrameQParams, FrameUpdateType, PickedQ, QBounds, RcMode, RcState,
    active_best_quality as leaf_active_best_quality, active_cq_level, gf_active_quality,
    gf_group_pyramid_level, intra_q_and_bounds, kf_active_quality, pick_q_and_bounds_q_mode,
};
use crate::ratectrl_rate::{compute_qdelta_by_rate, frame_type_qdelta, regulate_q};

/// `FIXED_GF_INTERVAL` (encoder/ratectrl.h:46).
pub const FIXED_GF_INTERVAL: usize = 16;

/// `MAX_ARF_LAYERS` (ratectrl.h:54).
const MAX_ARF_LAYERS: i32 = 6;

/// `STATIC_MOTION_THRESH` (ratectrl.c:1814).
const STATIC_MOTION_THRESH: i32 = 95;
/// `STATIC_KF_GROUP_THRESH` (ratectrl.h:38).
const STATIC_KF_GROUP_THRESH: i32 = 99;

/// `delta_rate[FIXED_GF_INTERVAL]` from `rc_pick_q_and_bounds_no_stats`
/// (ratectrl.c:1694).
///
/// **The array is SIXTEEN long and C supplies only EIGHT initialisers**, so
/// entries 8..15 are zero — and the index is
/// `frame_number % FIXED_GF_INTERVAL`, which reaches all sixteen. Half of all
/// leaf frames therefore get a rate factor of 0.0, which drives
/// `av1_compute_qdelta` to `best_quality`. An eight-entry table indexed `% 8`
/// is a plausible-looking transcription and is wrong on every frame whose
/// number mod 16 is 8 or more.
const DELTA_RATE: [f64; FIXED_GF_INTERVAL] = [
    0.50, 1.0, 0.85, 1.0, 0.70, 1.0, 0.85, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// C's `clamp` macro on `int` — `value < low ? low : value > high ? high :
/// value`. NOT Rust's `clamp`, which panics when `low > high`; several call
/// sites here can be handed a reversed range by a hostile config, and C
/// silently returns `high`.
fn clamp_c(value: i32, low: i32, high: i32) -> i32 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

/// `calc_active_worst_quality_no_stats_vbr` (ratectrl.c:1225).
///
/// `refresh_bwd_ref` is `cpi->refresh_frame.bwd_ref_frame`, which this
/// function reads and `get_active_best_quality` does not — the two
/// "is this a reference-refreshing frame" tests are NOT the same predicate.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn active_worst_quality_no_stats_vbr(
    is_key_frame: bool,
    frame_number: u32,
    last_q_key: i32,
    last_q_inter: i32,
    is_src_frame_alt_ref: bool,
    refresh_golden: bool,
    refresh_bwd_ref: bool,
    refresh_alt_ref: bool,
    worst_quality: i32,
) -> i32 {
    let active_worst_quality = if is_key_frame {
        if frame_number == 0 {
            worst_quality
        } else {
            last_q_key * 2
        }
    } else if !is_src_frame_alt_ref && (refresh_golden || refresh_bwd_ref || refresh_alt_ref) {
        if frame_number == 1 {
            last_q_key * 5 / 4
        } else {
            last_q_inter
        }
    } else if frame_number == 1 {
        last_q_key * 2
    } else {
        last_q_inter * 2
    };
    active_worst_quality.min(worst_quality)
}

/// The two-pass extension amounts `adjust_active_best_and_worst_quality`
/// applies under a non-`AOM_Q` mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TwoPassExtend {
    /// `cpi->ppi->twopass.extend_minq`.
    pub extend_minq: i32,
    /// `cpi->ppi->twopass.extend_maxq`.
    pub extend_maxq: i32,
}

/// `adjust_active_best_and_worst_quality` (ratectrl.c:1921).
///
/// Three things happen, in this order, and the order matters because each
/// reads the previous one's output:
/// 1. under a non-`AOM_Q` mode, widen by the two-pass extension — with
///    `CONFIG_FPMT_TEST == 0` that is `best -= extend_minq / 4` and
///    `worst += extend_maxq`, with NO `is_intrl_arf_boost` split;
/// 2. unless this is a static forced key frame, push `active_worst` out by
///    this frame's pyramid-layer qdelta (never below `active_best`);
/// 3. for a downscaled non-KF/GF/ARF frame, pull `active_best` down by the
///    1/2-rate qdelta, then clamp both into `[best_quality, worst_quality]`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn adjust_active_best_and_worst_quality(
    bounds: QBounds,
    rc_mode: RcMode,
    two_pass: TwoPassExtend,
    intra_only: bool,
    this_key_frame_forced: bool,
    last_kfgroup_zeromotion_pct: i32,
    update_type: FrameUpdateType,
    gf_group_is_key_frame: bool,
    layer_depth: i32,
    is_screen_content_type: bool,
    bit_depth: u8,
    best_quality: i32,
    worst_quality: i32,
    frame_scaled: bool,
    frame_is_kf_gf_arf: bool,
    frame_type_is_key: bool,
) -> QBounds {
    let mut active_best_quality = bounds.active_best;
    let mut active_worst_quality = bounds.active_worst;

    // Extend to max or min Q if the undershoot or overshoot is outside the
    // permitted range. CONFIG_FPMT_TEST is 0, so this is the `#else` arm.
    if rc_mode != RcMode::Q {
        active_best_quality -= two_pass.extend_minq / 4;
        active_worst_quality += two_pass.extend_maxq;
    }

    // STRICT_RC is not defined, so this block compiles. Static forced key
    // frames are dealt with elsewhere.
    if !intra_only || !this_key_frame_forced || last_kfgroup_zeromotion_pct < STATIC_MOTION_THRESH {
        let qdelta = frame_type_qdelta(
            update_type,
            gf_group_is_key_frame,
            layer_depth,
            is_screen_content_type,
            active_worst_quality,
            bit_depth,
            best_quality,
            worst_quality,
        );
        active_worst_quality = (active_worst_quality + qdelta).max(active_best_quality);
    }

    // Modify active_best_quality for downscaled normal frames.
    if frame_scaled && !frame_is_kf_gf_arf {
        let qdelta = compute_qdelta_by_rate(
            frame_type_is_key,
            is_screen_content_type,
            active_best_quality,
            2.0,
            bit_depth,
            best_quality,
            worst_quality,
        );
        active_best_quality = (active_best_quality + qdelta).max(best_quality);
    }

    let active_best_quality = clamp_c(active_best_quality, best_quality, worst_quality);
    let active_worst_quality = clamp_c(active_worst_quality, active_best_quality, worst_quality);
    QBounds {
        active_best: active_best_quality,
        active_worst: active_worst_quality,
    }
}

/// `get_q` (ratectrl.c:2005): pick a Q from the permitted range that should
/// land near the target bit count.
///
/// Under `AOM_Q` this is the identity on `active_best_quality`, which is why
/// the fixed-Q envelope never consults a bitrate.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn get_q(
    rc_mode: RcMode,
    intra_only: bool,
    this_key_frame_forced: bool,
    kf_zeromotion_pct: i32,
    last_kfgroup_zeromotion_pct: i32,
    frames_to_key: i32,
    last_boosted_qindex: i32,
    last_kf_qindex: i32,
    active_worst_quality: i32,
    active_best_quality: i32,
    this_frame_target: i32,
    max_frame_bandwidth: i32,
    width: i32,
    height: i32,
    is_key_frame: bool,
    is_screen_content_type: bool,
    correction_factor: f64,
    bit_depth: u8,
) -> i32 {
    if rc_mode == RcMode::Q
        || (intra_only
            && !this_key_frame_forced
            && kf_zeromotion_pct >= STATIC_KF_GROUP_THRESH
            && frames_to_key > 1)
    {
        return active_best_quality;
    }
    if intra_only && this_key_frame_forced {
        // If static since the last kf, use the better of the last boosted and
        // the last kf q.
        let q = if last_kfgroup_zeromotion_pct >= STATIC_MOTION_THRESH {
            last_kf_qindex.min(last_boosted_qindex)
        } else {
            last_boosted_qindex.min((active_best_quality + active_worst_quality) / 2)
        };
        return clamp_c(q, active_best_quality, active_worst_quality);
    }
    let mut q = regulate_q(
        this_frame_target,
        active_best_quality,
        active_worst_quality,
        width,
        height,
        is_key_frame,
        is_screen_content_type,
        correction_factor,
        bit_depth,
    );
    if q > active_worst_quality {
        // Special case when targeting the max allowed rate.
        if this_frame_target < max_frame_bandwidth {
            q = active_worst_quality;
        }
    }
    q.max(active_best_quality)
}

/// `rc_pick_q_and_bounds` (ratectrl.c:2188): the general path, which routes
/// straight to the `AOM_Q` leaf and otherwise runs the layered GF/ARF logic.
///
/// `active_best_quality_by_layer[pyramid_level - 1]` is
/// `p_rc->active_best_quality[]`, the previous frame at each pyramid level.
///
/// `frame_type_is_key` is `cm->current_frame.frame_type == KEY_FRAME`;
/// `gf_group_is_key_frame` is `gf_group->frame_type[gf_index] == KEY_FRAME`.
/// They are DIFFERENT reads in C — the first feeds the downscale qdelta, the
/// second feeds `frame_type_qdelta`'s rate model — and conflating them was a
/// real defect this function's differential caught.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn pick_q_and_bounds(
    p: &FrameQParams,
    rc: &RcState,
    rc_mode: RcMode,
    configured_cq_level: i32,
    active_worst_in: i32,
    intra_only: bool,
    update_type: FrameUpdateType,
    layer_depth: i32,
    total_actual_bits: i64,
    total_target_bits: i64,
    active_best_quality_by_layer: &[i32; MAX_ARF_LAYERS as usize],
    two_pass: TwoPassExtend,
    is_src_frame_alt_ref: bool,
    this_frame_target: i32,
    max_frame_bandwidth: i32,
    correction_factor: f64,
    frame_scaled: bool,
    frame_type_is_key: bool,
    gf_group_is_key_frame: bool,
) -> PickedQ {
    let cq_level = active_cq_level(
        configured_cq_level,
        rc_mode,
        intra_only,
        rc.frames_to_key,
        p.superres_mode,
        p.superres_denom,
        total_actual_bits,
        total_target_bits,
    );

    if rc_mode == RcMode::Q {
        return pick_q_and_bounds_q_mode(
            p,
            rc,
            rc_mode,
            configured_cq_level,
            active_worst_in,
            intra_only,
            update_type,
            layer_depth,
            total_actual_bits,
            total_target_bits,
        );
    }

    let luts = p.minq_luts();
    let is_intrl_arf_boost = update_type == FrameUpdateType::IntnlArf;
    let mut active_worst_quality = active_worst_in;
    let mut active_best_quality;

    if intra_only {
        let b = intra_q_and_bounds(p, rc, &luts, rc_mode, cq_level, active_worst_quality);
        active_best_quality = b.active_best;
        active_worst_quality = b.active_worst;
    } else {
        // Active best quality limited by the previous layer.
        let pyramid_level = gf_group_pyramid_level(layer_depth);
        if pyramid_level <= 1 || pyramid_level > MAX_ARF_LAYERS {
            active_best_quality = leaf_active_best_quality(
                p,
                rc,
                &luts,
                rc_mode,
                active_worst_quality,
                cq_level,
                update_type,
                layer_depth,
            );
        } else {
            active_best_quality = active_best_quality_by_layer
                [usize::try_from(pyramid_level - 1).expect("pyramid_level >= 2 here")]
                + 1;
            active_best_quality = active_best_quality.min(active_worst_quality);
            // STRICT_RC undefined, so the divisor is 2, not 16.
            active_best_quality += (active_worst_quality - active_best_quality) / 2;
        }

        // For alt-ref and GF frames (internal ARFs included) adjust the worst
        // allowed quality too, so hard sections do not clamp ARF and leaf
        // frames at the same Q — the TPL model assumes Q drops per ARF level.
        if !is_src_frame_alt_ref && (p.refresh_golden || p.refresh_alt_ref || is_intrl_arf_boost) {
            active_worst_quality = (active_best_quality + (3 * active_worst_quality) + 2) / 4;
        }
    }

    let adjusted = adjust_active_best_and_worst_quality(
        QBounds {
            active_best: active_best_quality,
            active_worst: active_worst_quality,
        },
        rc_mode,
        two_pass,
        intra_only,
        rc.this_key_frame_forced,
        rc.last_kfgroup_zeromotion_pct,
        update_type,
        // frame_type_qdelta reads gf_group->frame_type[gf_index], NOT
        // cm->current_frame.frame_type. The two differ on, among others, an
        // internal-ARF frame in a key-frame-typed gf slot, and passing the
        // wrong one shifts the qdelta by a whole rate ratio.
        gf_group_is_key_frame,
        layer_depth,
        p.screen_content,
        p.bit_depth,
        rc.best_quality,
        rc.worst_quality,
        frame_scaled,
        // frame_is_kf_gf_arf(cpi)
        intra_only || matches!(update_type, FrameUpdateType::Arf | FrameUpdateType::Gf),
        frame_type_is_key,
    );

    let q = get_q(
        rc_mode,
        intra_only,
        rc.this_key_frame_forced,
        rc.kf_zeromotion_pct,
        rc.last_kfgroup_zeromotion_pct,
        rc.frames_to_key,
        rc.last_boosted_qindex,
        rc.last_kf_qindex,
        adjusted.active_worst,
        adjusted.active_best,
        this_frame_target,
        max_frame_bandwidth,
        p.coded_width,
        p.coded_height,
        frame_type_is_key,
        p.screen_content,
        correction_factor,
        p.bit_depth,
    );

    let mut active_worst_quality = adjusted.active_worst;
    // Special case when targeting the max allowed rate.
    if this_frame_target >= max_frame_bandwidth && q > active_worst_quality {
        active_worst_quality = q;
    }
    PickedQ {
        q,
        bottom_index: adjusted.active_best,
        top_index: active_worst_quality,
    }
}

/// `rc_pick_q_and_bounds_no_stats` (ratectrl.c:1588): the one-pass,
/// no-lookahead path.
///
/// Reachable under `AOM_Q` only for an `ARF_UPDATE` frame (see
/// [`PickQRoute`]), and under `AOM_VBR` for every frame.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn pick_q_and_bounds_no_stats(
    p: &FrameQParams,
    rc: &RcState,
    rc_mode: RcMode,
    configured_cq_level: i32,
    intra_only: bool,
    frame_number: u32,
    width: i32,
    height: i32,
    last_q_key: i32,
    last_q_inter: i32,
    avg_frame_qindex_key: i32,
    is_src_frame_alt_ref: bool,
    refresh_bwd_ref: bool,
    this_frame_target: i32,
    max_frame_bandwidth: i32,
    correction_factor: f64,
    total_actual_bits: i64,
    total_target_bits: i64,
    frame_type_is_key: bool,
) -> PickedQ {
    let bit_depth = p.bit_depth;
    let luts = p.minq_luts();
    let cq_level = active_cq_level(
        configured_cq_level,
        rc_mode,
        intra_only,
        rc.frames_to_key,
        p.superres_mode,
        p.superres_denom,
        total_actual_bits,
        total_target_bits,
    );

    let mut active_worst_quality = active_worst_quality_no_stats_vbr(
        frame_type_is_key,
        frame_number,
        last_q_key,
        last_q_inter,
        is_src_frame_alt_ref,
        p.refresh_golden,
        refresh_bwd_ref,
        p.refresh_alt_ref,
        rc.worst_quality,
    );
    let mut active_best_quality;

    let qdelta_to = |qindex: i32, ratio: f64| {
        let q_val = convert_qindex_to_q(qindex, bit_depth);
        compute_qdelta(
            q_val,
            q_val * ratio,
            bit_depth,
            rc.best_quality,
            rc.worst_quality,
        )
    };

    if intra_only {
        if rc_mode == RcMode::Q {
            active_best_quality = (cq_level + qdelta_to(cq_level, 0.25)).max(rc.best_quality);
        } else if rc.this_key_frame_forced {
            let qindex = rc.last_boosted_qindex;
            active_best_quality = (qindex + qdelta_to(qindex, 0.75)).max(rc.best_quality);
        } else {
            // Not the first frame of a one-pass encode, and kf_boost is set.
            let mut q_adj_factor = 1.0f64;
            active_best_quality =
                kf_active_quality(&luts, rc.kf_boost, avg_frame_qindex_key, p.rtc_mode);
            // Allow a somewhat lower kf minq with small image formats.
            if (width * height) <= (352 * 288) {
                q_adj_factor -= 0.25;
            }
            let q_val = convert_qindex_to_q(active_best_quality, bit_depth);
            active_best_quality += compute_qdelta(
                q_val,
                q_val * q_adj_factor,
                bit_depth,
                rc.best_quality,
                rc.worst_quality,
            );
        }
    } else if !is_src_frame_alt_ref && (p.refresh_golden || p.refresh_alt_ref) {
        // The lower of active_worst_quality and the recent average Q, unless
        // the last frame was a key frame.
        let mut q = if rc.frames_since_key > 1 && rc.avg_frame_qindex_inter < active_worst_quality {
            rc.avg_frame_qindex_inter
        } else {
            avg_frame_qindex_key
        };
        if rc_mode == RcMode::Cq {
            if q < cq_level {
                q = cq_level;
            }
            active_best_quality = gf_active_quality(
                &luts,
                rc.gfu_boost,
                rc.gfu_boost_average,
                q,
                p.res_idx(),
                p.rtc_mode,
            );
            active_best_quality = active_best_quality * 15 / 16;
        } else if rc_mode == RcMode::Q {
            // An alt-ref frame gets a deeper drop than a golden one.
            let ratio = if p.refresh_alt_ref { 0.40 } else { 0.50 };
            active_best_quality = (cq_level + qdelta_to(cq_level, ratio)).max(rc.best_quality);
        } else {
            active_best_quality = gf_active_quality(
                &luts,
                rc.gfu_boost,
                rc.gfu_boost_average,
                q,
                p.res_idx(),
                p.rtc_mode,
            );
        }
    } else if rc_mode == RcMode::Q {
        // The sixteen-entry delta_rate table with eight zero tail entries —
        // see DELTA_RATE.
        let ratio = DELTA_RATE[frame_number as usize % FIXED_GF_INTERVAL];
        active_best_quality = (cq_level + qdelta_to(cq_level, ratio)).max(rc.best_quality);
    } else {
        active_best_quality = if frame_number > 1 {
            luts.inter[usize::try_from(rc.avg_frame_qindex_inter).expect("qindex")]
        } else {
            luts.inter[usize::try_from(avg_frame_qindex_key).expect("qindex")]
        };
        if rc_mode == RcMode::Cq && active_best_quality < cq_level {
            active_best_quality = cq_level;
        }
    }

    // Clip the active best and worst values to the limits.
    active_best_quality = clamp_c(active_best_quality, rc.best_quality, rc.worst_quality);
    active_worst_quality = clamp_c(active_worst_quality, active_best_quality, rc.worst_quality);

    let bottom_index = active_best_quality;
    // Limit the Q range for the adaptive loop.
    let qdelta = if frame_type_is_key && !rc.this_key_frame_forced && frame_number != 0 {
        compute_qdelta_by_rate(
            frame_type_is_key,
            p.screen_content,
            active_worst_quality,
            2.0,
            bit_depth,
            rc.best_quality,
            rc.worst_quality,
        )
    } else if !is_src_frame_alt_ref && (p.refresh_golden || p.refresh_alt_ref) {
        compute_qdelta_by_rate(
            frame_type_is_key,
            p.screen_content,
            active_worst_quality,
            1.75,
            bit_depth,
            rc.best_quality,
            rc.worst_quality,
        )
    } else {
        0
    };
    let mut top_index = (active_worst_quality + qdelta).max(bottom_index);

    let q = if rc_mode == RcMode::Q {
        active_best_quality
    } else if frame_type_is_key && rc.this_key_frame_forced {
        rc.last_boosted_qindex
    } else {
        let mut q = regulate_q(
            this_frame_target,
            active_best_quality,
            active_worst_quality,
            width,
            height,
            frame_type_is_key,
            p.screen_content,
            correction_factor,
            bit_depth,
        );
        if q > top_index {
            // Special case when targeting the max allowed rate.
            if this_frame_target >= max_frame_bandwidth {
                top_index = q;
            } else {
                q = top_index;
            }
        }
        q
    };

    PickedQ {
        q,
        bottom_index,
        top_index,
    }
}

/// Which of `av1_rc_pick_q_and_bounds`'s three leaves a frame takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickQRoute {
    /// `rc_pick_q_and_bounds_no_stats_cbr` — not ported (CBR).
    NoStatsCbr,
    /// `rc_pick_q_and_bounds_no_stats`.
    NoStats,
    /// `rc_pick_q_and_bounds`, which itself routes to the `AOM_Q` leaf.
    General,
}

impl PickQRoute {
    /// The routing in `av1_rc_pick_q_and_bounds` (ratectrl.c:2350).
    ///
    /// The `AOM_CQ` leaf is behind `USE_UNRESTRICTED_Q_IN_CQ_MODE`, which is
    /// `0`, so an `AOM_CQ` no-stats frame falls through to [`Self::NoStats`]
    /// rather than to a CQ-specific function.
    ///
    /// Note the condition: an `ARF_UPDATE` frame takes the no-stats leaf even
    /// under `AOM_Q`. That is why the fixed-Q envelope's first target is
    /// `--lag-in-frames=0`, which has no ARF frames at all.
    #[must_use]
    pub fn of(rc_mode: RcMode, update_type: FrameUpdateType, has_no_stats_stage: bool) -> Self {
        if (rc_mode != RcMode::Q || update_type == FrameUpdateType::Arf) && has_no_stats_stage {
            if rc_mode == RcMode::Cbr {
                Self::NoStatsCbr
            } else {
                Self::NoStats
            }
        } else {
            Self::General
        }
    }
}

/// The tail of `av1_rc_pick_q_and_bounds`: `if (update_type == ARF_UPDATE)
/// p_rc->arf_q = q`.
///
/// Returned rather than written, so a caller that forgets it is a compile
/// error rather than a silently stale `arf_q` on the next internal-ARF frame.
#[must_use]
pub fn arf_q_after_pick(update_type: FrameUpdateType, q: i32, current_arf_q: i32) -> i32 {
    if update_type == FrameUpdateType::Arf {
        q
    } else {
        current_arf_q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_rate_tail_is_zero_not_a_repeat_of_the_head() {
        // The half of the table C never initialises. An 8-entry transcription
        // indexed `% 8` would put 0.50 at index 8; C puts 0.0.
        assert_eq!(DELTA_RATE.len(), FIXED_GF_INTERVAL);
        assert_eq!(DELTA_RATE[0], 0.50);
        assert_eq!(DELTA_RATE[7], 1.0);
        for (i, &v) in DELTA_RATE.iter().enumerate().skip(8) {
            assert_eq!(v, 0.0, "delta_rate[{i}]");
        }
    }

    #[test]
    fn arf_update_takes_the_no_stats_leaf_even_under_q() {
        assert_eq!(
            PickQRoute::of(RcMode::Q, FrameUpdateType::Arf, true),
            PickQRoute::NoStats
        );
        assert_eq!(
            PickQRoute::of(RcMode::Q, FrameUpdateType::Lf, true),
            PickQRoute::General
        );
        // Without a no-stats stage (i.e. two-pass) everything is General.
        assert_eq!(
            PickQRoute::of(RcMode::Vbr, FrameUpdateType::Arf, false),
            PickQRoute::General
        );
        assert_eq!(
            PickQRoute::of(RcMode::Cbr, FrameUpdateType::Lf, true),
            PickQRoute::NoStatsCbr
        );
    }

    #[test]
    fn get_q_under_aom_q_is_the_identity_on_active_best() {
        for best in [0, 1, 128, 255] {
            assert_eq!(
                get_q(
                    RcMode::Q,
                    false,
                    false,
                    0,
                    0,
                    10,
                    100,
                    100,
                    200,
                    best,
                    1000,
                    2000,
                    352,
                    288,
                    false,
                    false,
                    1.0,
                    8
                ),
                best
            );
        }
    }

    #[test]
    fn arf_q_is_only_written_on_an_arf_update() {
        assert_eq!(arf_q_after_pick(FrameUpdateType::Arf, 77, 12), 77);
        for ty in [
            FrameUpdateType::Kf,
            FrameUpdateType::Lf,
            FrameUpdateType::Gf,
            FrameUpdateType::Overlay,
            FrameUpdateType::IntnlOverlay,
            FrameUpdateType::IntnlArf,
        ] {
            assert_eq!(arf_q_after_pick(ty, 77, 12), 12, "{ty:?}");
        }
    }
}
