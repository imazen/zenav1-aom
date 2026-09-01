//! Multi-frame fixed-Q rate control — the `AOM_Q` arms of
//! `av1/encoder/ratectrl.c`.
//!
//! [`crate::rc`] answers "what `base_qindex` does a LONE key frame get?" and
//! [`crate::rate_model`] holds the qindex↔Q conversions underneath it. This
//! module is the layer between them and a real GOP: the minq lookup tables, the
//! boost-interpolated active-quality curves they feed, and the
//! `rc_pick_q_and_bounds_q_mode` chain that turns a GF-group entry into the
//! frame's qindex. Every coded frame's `base_qindex` comes out of here, so it
//! is bitstream-visible.
//!
//! | Rust | C (`av1/encoder/ratectrl.c`) |
//! |---|---|
//! | [`minq_index`] | `get_minq_index` (:132, static) |
//! | [`MinqLuts::new`] | `init_minq_luts` (:155) / `rc_init_minq_luts` (:181) / `av1_rc_init_minq_luts` (:192) |
//! | [`active_quality`] | `get_active_quality` (:1156, static) |
//! | [`kf_active_quality`] | `get_kf_active_quality` (:1173, static) |
//! | [`gf_active_quality`] | `get_gf_active_quality` / `_no_rc` (:1188, :1213, static) |
//! | [`gf_high_motion_quality`] | `get_gf_high_motion_quality` (:1219, static) |
//! | [`gf_group_pyramid_level`] | `gf_group_pyramid_level` (:1535, static) |
//! | [`active_cq_level`] | `get_active_cq_level` (:1539, static) |
//! | [`intra_q_and_bounds`] | `get_intra_q_and_bounds` (:1815, static) |
//! | [`active_best_quality`] | `get_active_best_quality` (:2057, static) |
//! | [`pick_q_and_bounds_q_mode`] | `rc_pick_q_and_bounds_q_mode` (:2133, static) |
//! | [`default_max_gf_interval`] | `get_default_max_gf_interval` (:452, static) |
//!
//! # Scope: the `AOM_Q` arms only
//! The encode target is `--end-usage=q` (INTER-ENCODE-ROADMAP §3). Everything
//! keyed on `rc_cfg.mode == AOM_CBR` / `AOM_VBR` / `AOM_CQ` — `adjust_q_cbr`,
//! the `calc_*_target_size_one_pass_{cbr,vbr}` family, `vbr_rate_correction`,
//! `rc_pick_q_and_bounds_no_stats_cbr`, the drop-frame machinery — is out of
//! this module by design, not by omission. Where a ported function has a
//! non-`AOM_Q` arm the arm is kept (it costs nothing and the differential
//! sweeps it), and where it is absent the doc says so.
//!
//! Two-pass state is likewise absent: `--passes=1` means
//! `is_stat_consumption_stage_twopass` is false, which erases the
//! `kf_zeromotion_pct` / `last_kfgroup_zeromotion_pct` arms of
//! [`intra_q_and_bounds`]. Those arms ARE ported and are swept by the
//! differential, because they are pure arithmetic once their inputs are given.
//!
//! # Findings a later reader is likely to "correct" back
//! 1. **`p_rc->arf_boost_factor` is a `float`, not a `double`.** C declares it
//!    `float_t` (`encoder/ratectrl.h:383`), which is `float` wherever
//!    `FLT_EVAL_METHOD == 0` — aarch64 and x86-64 both. So
//!    `min_boost - (int)(boost * arf_boost_factor)` in [`active_best_quality`]
//!    is a SINGLE-precision multiply. An `f64` transcription of that one line
//!    is what the first run of `active_best_quality_matches_c` caught, and
//!    reverting to `f64` still fails it.
//! 2. **`get_active_quality`'s rounding is C integer division, not Euclidean.**
//!    `(offset * qdiff + gap/2) / gap` truncates toward zero, so a negative
//!    `qdiff` rounds the other way. Measured across all twelve real minq
//!    tables, `qdiff >= 0` in all 6,144 cells — so a Euclidean transcription
//!    passes every table-driven sweep. The differential therefore includes a
//!    synthetic-array arm with the curves swapped, which is the only thing
//!    that makes that division observable.
//! 3. **`ASSIGN_MINQ_TABLE_2`'s parameters are named in the opposite order to
//!    its body.** It is declared `(bit_depth, name, res_idx, mode_idx)` and
//!    indexes `name##_<bd>[mode_idx][res_idx]`; every call site passes
//!    `(res_idx > 1, rtc_mode)`. Reading the declaration transposes the tables.
//! 4. **`get_default_max_gf_interval` undoes its own `AOMMIN`**: it takes
//!    `AOMMIN(MAX_GF_INTERVAL, framerate*0.75)` and two lines later
//!    `AOMMAX(MAX_GF_INTERVAL, interval)`, so the framerate term can only
//!    reach the result through `min_gf_interval`. Reproduced verbatim.
//! 5. **`get_active_best_quality`'s `res_idx` is three-valued, but the minq
//!    tables' is two-valued.** `gfboost_thresh` is indexed by the full
//!    `0/1/2`, the tables by `res_idx > 1`. Since `gfboost_thresh[0] ==
//!    gfboost_thresh[1] == 4000`, collapsing the former is an INERT change —
//!    it passes the differential. It is still wrong, and it is why the bite
//!    proof for that line flips index 0 against index 2 instead.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/ratectrl_q_diff.rs`. Tier 1c: the oracle is
//! libaom's own `ratectrl.c`, compiled verbatim into
//! `crates/aom-sys-ref/shim/ratectrl_shim.c` with its 31 exported symbols
//! renamed out of the way, because 20 of the functions above are file-static
//! and have no address the archive can hand out. The TU is proved equivalent
//! to the archive's copy by `ratectrl_shim_tu_matches_archive`, which compares
//! re-exported `av1_convert_qindex_to_q` / `av1_find_qindex` /
//! `av1_compute_qdelta` / `av1_rc_bits_per_mb` /
//! `av1_rc_get_default_min_gf_interval` from this TU against the archive's.

use crate::rate_model::{compute_qdelta, convert_qindex_to_q, find_qindex};

/// `QINDEX_RANGE` (av1/common/quant_common.h:27) — `MAXQ - MINQ + 1`.
pub const QINDEX_RANGE: usize = 256;
/// `MINQ` (quant_common.h:25).
pub const MINQ: i32 = 0;
/// `MAXQ` (quant_common.h:26).
pub const MAXQ: i32 = 255;

/// `MAX_GF_INTERVAL` (ratectrl.h:45). Its sibling `MIN_GF_INTERVAL` is used
/// only by `av1_rc_get_default_min_gf_interval`, which lives in
/// [`crate::rate_model`].
const MAX_GF_INTERVAL: i32 = 32;

/// `SCALE_NUMERATOR` (av1/common/scale.h:22) — the superres denominator that
/// means "not scaled".
pub const SCALE_NUMERATOR: i32 = 8;
/// `SUPERRES_QADJ_PER_DENOM_KEYFRAME_SOLO` (ratectrl.c:55).
const SUPERRES_QADJ_PER_DENOM_KEYFRAME_SOLO: i32 = 0;
/// `SUPERRES_QADJ_PER_DENOM_KEYFRAME` (ratectrl.c:56).
const SUPERRES_QADJ_PER_DENOM_KEYFRAME: i32 = 2;
/// `SUPERRES_QADJ_PER_DENOM_ARFFRAME` (ratectrl.c:57).
const SUPERRES_QADJ_PER_DENOM_ARFFRAME: i32 = 0;
/// `STATIC_KF_GROUP_THRESH` (ratectrl.h:38).
const STATIC_KF_GROUP_THRESH: i32 = 99;
/// `STATIC_MOTION_THRESH` (ratectrl.c:1814).
const STATIC_MOTION_THRESH: i32 = 95;

/// The GF/ARF boost thresholds `get_gf_active_quality_no_rc` interpolates
/// between when the boost average is below/above `gfboost_thresh`
/// (`gf_low_1` / `gf_high_1` / `gf_low_2` / `gf_high_2`, ratectrl.c:110-113).
const GF_HIGH_1: i32 = 2875;
const GF_LOW_1: i32 = 562;
const GF_HIGH_2: i32 = 4994;
const GF_LOW_2: i32 = 100;
/// `kf_high` / `kf_low` (ratectrl.c:115-116).
const KF_HIGH: i32 = 8000;
const KF_LOW: i32 = 553;
/// The real-time variants (ratectrl.c:118-121).
const GF_HIGH_RTC: i32 = 2400;
const GF_LOW_RTC: i32 = 300;
const KF_HIGH_RTC: i32 = 5000;
const KF_LOW_RTC: i32 = 400;
/// `gfboost_thresh[3]` (ratectrl.c:1171), indexed by the *three*-valued
/// `res_idx` (unlike the minq tables, which collapse it to `res_idx > 1`).
const GFBOOST_THRESH: [i32; 3] = [4000, 4000, 3000];

/// `x1[MODE_NUM][RES_NUM][5]` (ratectrl.c:145): the linear coefficient of the
/// minq polynomial, per `[rtc_mode][hi_res]`, for the five curves
/// `[kf_low, kf_high, arfgf_low, arfgf_high, inter]`.
const X1: [[[f64; 5]; 2]; 2] = [
    [
        [0.1771, 0.379, 0.3279, 0.6634, 1.385],
        [0.1917, 0.3760, 0.3457, 0.6916, 1.1482],
    ],
    [
        [0.15, 0.45, 0.30, 0.55, 0.90],
        [0.15, 0.45, 0.30, 0.55, 0.90],
    ],
];

/// `get_minq_index` (ratectrl.c:132): invert a 3rd-order polynomial fit of
/// "real maxq vs minq" back into a qindex.
///
/// The `<= 2.0` early return is C's own comment: "special case handling to deal
/// with the step from q2.0 down to lossless mode represented by q 1.0".
///
/// # Panics
/// Panics if `bit_depth` is not 8, 10 or 12 (via [`convert_qindex_to_q`]).
#[must_use]
pub fn minq_index(maxq: f64, x3: f64, x2: f64, x1: f64, bit_depth: u8) -> i32 {
    let minqtarget = (((x3 * maxq + x2) * maxq + x1) * maxq).min(maxq);
    if minqtarget <= 2.0 {
        return 0;
    }
    find_qindex(minqtarget, bit_depth, MINQ, MAXQ)
}

/// The five minq lookup tables for one `(bit_depth, rtc_mode, hi_res)` cell.
///
/// C keeps twelve `[MODE_NUM][RES_NUM][QINDEX_RANGE]` file-static arrays (five
/// curves plus `rtc_minq`, times three bit depths) filled once by
/// `av1_rc_init_minq_luts` behind an `aom_once`. The port builds the one cell a
/// frame actually indexes; C's `ASSIGN_MINQ_TABLE_2` selects exactly that cell
/// with `name##_<bd>[rtc_mode][res_idx > 1]`, so nothing else is ever read.
///
/// Note the argument order of the C macro: it is declared `(res_idx, mode_idx)`
/// but its body indexes `[mode_idx][res_idx]`, and every call site passes
/// `(res_idx > 1, rtc_mode)` — so the FIRST table subscript is `rtc_mode`.
/// Reading the declaration instead of the body transposes the tables.
#[derive(Clone, Debug, PartialEq)]
pub struct MinqLuts {
    /// `kf_low_motion_minq_*`.
    pub kf_low: [i32; QINDEX_RANGE],
    /// `kf_high_motion_minq_*`.
    pub kf_high: [i32; QINDEX_RANGE],
    /// `arfgf_low_motion_minq_*`.
    pub arfgf_low: [i32; QINDEX_RANGE],
    /// `arfgf_high_motion_minq_*`.
    pub arfgf_high: [i32; QINDEX_RANGE],
    /// `inter_minq_*`.
    pub inter: [i32; QINDEX_RANGE],
    /// `rtc_minq_*`. C fills this from a fixed `x1 = 0.70` independent of the
    /// `[mode][res]` cell, so it is the same array for every cell of a bit
    /// depth. Kept here because `rc_pick_q_and_bounds_no_stats_cbr` reads it.
    pub rtc: [i32; QINDEX_RANGE],
}

impl MinqLuts {
    /// `init_minq_luts` (ratectrl.c:155) restricted to one `[rtc_mode][hi_res]`
    /// cell — the cell `ASSIGN_MINQ_TABLE_2` would select.
    ///
    /// `hi_res` is C's `res_idx > 1`, i.e. the shorter frame side is >= 608.
    ///
    /// # Panics
    /// Panics if `bit_depth` is not 8, 10 or 12.
    #[must_use]
    pub fn new(bit_depth: u8, rtc_mode: bool, hi_res: bool) -> Self {
        let x1 = X1[usize::from(rtc_mode)][usize::from(hi_res)];
        let mut luts = Self {
            kf_low: [0; QINDEX_RANGE],
            kf_high: [0; QINDEX_RANGE],
            arfgf_low: [0; QINDEX_RANGE],
            arfgf_high: [0; QINDEX_RANGE],
            inter: [0; QINDEX_RANGE],
            rtc: [0; QINDEX_RANGE],
        };
        for i in 0..QINDEX_RANGE {
            let maxq = convert_qindex_to_q(i as i32, bit_depth);
            luts.kf_low[i] = minq_index(maxq, 0.000001, -0.0004, x1[0], bit_depth);
            luts.kf_high[i] = minq_index(maxq, 0.0000021, -0.00125, x1[1], bit_depth);
            luts.arfgf_low[i] = minq_index(maxq, 0.0000015, -0.0009, x1[2], bit_depth);
            luts.arfgf_high[i] = minq_index(maxq, 0.0000021, -0.00125, x1[3], bit_depth);
            luts.inter[i] = minq_index(maxq, 0.00000271, -0.00113, x1[4], bit_depth);
            luts.rtc[i] = minq_index(maxq, 0.00000271, -0.00113, 0.70, bit_depth);
        }
        luts
    }
}

/// `get_active_quality` (ratectrl.c:1156): pick between the low-motion and
/// high-motion minq curves, interpolating linearly in `gfu_boost` between them.
///
/// The interpolation rounds with `(offset * qdiff + gap/2) / gap` — C integer
/// division, which truncates toward zero, so a NEGATIVE `qdiff` (high-motion
/// minq below low-motion minq) rounds the other way. `i32` division matches
/// that; Euclidean division would not.
///
/// # Panics
/// Panics if `q` is outside `0..QINDEX_RANGE`, or if `low == high` while
/// `gfu_boost` is between them (C would divide by zero).
#[must_use]
pub fn active_quality(
    q: i32,
    gfu_boost: i32,
    low: i32,
    high: i32,
    low_motion_minq: &[i32; QINDEX_RANGE],
    high_motion_minq: &[i32; QINDEX_RANGE],
) -> i32 {
    let q = usize::try_from(q).expect("q must be a qindex");
    if gfu_boost > high {
        low_motion_minq[q]
    } else if gfu_boost < low {
        high_motion_minq[q]
    } else {
        let gap = high - low;
        assert!(gap != 0, "get_active_quality divides by high - low");
        let offset = high - gfu_boost;
        let qdiff = high_motion_minq[q] - low_motion_minq[q];
        let adjustment = ((offset * qdiff) + (gap >> 1)) / gap;
        low_motion_minq[q] + adjustment
    }
}

/// `get_kf_active_quality` (ratectrl.c:1173).
#[must_use]
pub fn kf_active_quality(luts: &MinqLuts, kf_boost: i32, q: i32, rtc_mode: bool) -> i32 {
    let (low, high) = if rtc_mode {
        (KF_LOW_RTC, KF_HIGH_RTC)
    } else {
        (KF_LOW, KF_HIGH)
    };
    active_quality(q, kf_boost, low, high, &luts.kf_low, &luts.kf_high)
}

/// `get_gf_active_quality` (ratectrl.c:1213), which is a straight call through
/// to `get_gf_active_quality_no_rc` (:1188).
///
/// `res_idx` is the THREE-valued resolution index (`0` below 480p, `1` at 480p,
/// `2` at 608p and above) because `gfboost_thresh` has three entries — unlike
/// the minq tables, which use the collapsed `res_idx > 1`. Passing the
/// collapsed form here silently picks the 480p threshold for sub-480p frames.
///
/// # Panics
/// Panics if `res_idx` is outside `0..=2`.
#[must_use]
pub fn gf_active_quality(
    luts: &MinqLuts,
    gfu_boost: i32,
    gfu_boost_average: i32,
    q: i32,
    res_idx: usize,
    rtc_mode: bool,
) -> i32 {
    let (low, high) = if rtc_mode {
        (GF_LOW_RTC, GF_HIGH_RTC)
    } else if gfu_boost_average < GFBOOST_THRESH[res_idx] {
        (GF_LOW_1, GF_HIGH_1)
    } else {
        (GF_LOW_2, GF_HIGH_2)
    };
    active_quality(q, gfu_boost, low, high, &luts.arfgf_low, &luts.arfgf_high)
}

/// `get_gf_high_motion_quality` (ratectrl.c:1219): the arfgf high-motion curve
/// read directly, with no boost interpolation.
///
/// # Panics
/// Panics if `q` is outside `0..QINDEX_RANGE`.
#[must_use]
pub fn gf_high_motion_quality(luts: &MinqLuts, q: i32) -> i32 {
    luts.arfgf_high[usize::try_from(q).expect("q must be a qindex")]
}

/// `get_default_max_gf_interval` (ratectrl.c:452).
///
/// Note the `AOMMAX(MAX_GF_INTERVAL, interval)` on the line after the
/// `AOMMIN(MAX_GF_INTERVAL, ...)`: the min is immediately undone, so the result
/// is always at least `MAX_GF_INTERVAL` (32) and the framerate term can only
/// matter through `min_gf_interval`. Reproduced verbatim; libaom carries the
/// same shape.
#[must_use]
pub fn default_max_gf_interval(framerate: f64, min_gf_interval: i32) -> i32 {
    let mut interval = MAX_GF_INTERVAL.min((framerate * 0.75) as i32);
    interval += interval & 0x01; // Round to even value.
    interval = interval.max(MAX_GF_INTERVAL);
    interval.max(min_gf_interval)
}

/// `gf_group_pyramid_level` (ratectrl.c:1535).
#[must_use]
pub fn gf_group_pyramid_level(layer_depth: i32) -> i32 {
    layer_depth
}

/// `aom_superres_mode` (aom/aomcx.h), as far as `get_active_cq_level` reads it.
/// Discriminants match C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuperresMode {
    /// `AOM_SUPERRES_NONE`.
    None = 0,
    /// `AOM_SUPERRES_FIXED`.
    Fixed = 1,
    /// `AOM_SUPERRES_RANDOM`.
    Random = 2,
    /// `AOM_SUPERRES_QTHRESH`.
    QThresh = 3,
    /// `AOM_SUPERRES_AUTO`.
    Auto = 4,
}

impl SuperresMode {
    /// The two modes whose qindex adjustment `get_active_cq_level` and
    /// `get_intra_q_and_bounds` apply.
    #[must_use]
    pub fn adjusts_q(self) -> bool {
        matches!(self, Self::QThresh | Self::Auto)
    }
}

/// `aom_rc_mode` (aom/aom_encoder.h). Discriminants match C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RcMode {
    /// `AOM_VBR`.
    Vbr = 0,
    /// `AOM_CBR`.
    Cbr = 1,
    /// `AOM_CQ`.
    Cq = 2,
    /// `AOM_Q` — the port's encode target.
    Q = 3,
}

/// `get_active_cq_level` (ratectrl.c:1539): the configured `cq_level` after the
/// superres and (CQ-only) undershoot adjustments.
///
/// `total_actual_bits` / `total_target_bits` are `p_rc`'s running totals and are
/// read only in the `AOM_CQ` arm.
// The parameter list is exactly C's read set at this call; grouping it behind
// a context struct would rename the same fields without removing any.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn active_cq_level(
    cq_level: i32,
    rc_mode: RcMode,
    intra_only: bool,
    frames_to_key: i32,
    superres_mode: SuperresMode,
    superres_denom: i32,
    total_actual_bits: i64,
    total_target_bits: i64,
) -> i32 {
    const CQ_ADJUST_THRESHOLD: f64 = 0.1;
    let mut active_cq_level = cq_level;
    if matches!(rc_mode, RcMode::Cq | RcMode::Q)
        && superres_mode.adjusts_q()
        && superres_denom != SCALE_NUMERATOR
    {
        let mult = if intra_only && frames_to_key <= 1 {
            0
        } else if intra_only {
            SUPERRES_QADJ_PER_DENOM_KEYFRAME
        } else {
            SUPERRES_QADJ_PER_DENOM_ARFFRAME
        };
        // The SOLO constant names the `intra_only && frames_to_key <= 1` case;
        // it is 0, which is the literal C writes there.
        debug_assert_eq!(SUPERRES_QADJ_PER_DENOM_KEYFRAME_SOLO, 0);
        active_cq_level = (active_cq_level - ((superres_denom - SCALE_NUMERATOR) * mult)).max(0);
    }
    if rc_mode == RcMode::Cq && total_target_bits > 0 {
        let x = total_actual_bits as f64 / total_target_bits as f64;
        if x < CQ_ADJUST_THRESHOLD {
            active_cq_level = (f64::from(active_cq_level) * x / CQ_ADJUST_THRESHOLD) as i32;
        }
    }
    active_cq_level
}

/// The `[active_best, active_worst]` qindex window a frame is picked from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QBounds {
    /// `active_best_quality` — the best (lowest) qindex allowed.
    pub active_best: i32,
    /// `active_worst_quality` — the worst (highest) qindex allowed.
    pub active_worst: i32,
}

/// The frame-independent rate-control state `get_intra_q_and_bounds` and
/// `get_active_best_quality` read out of `PRIMARY_RATE_CONTROL` and
/// `RATE_CONTROL`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RcState {
    /// `p_rc->kf_boost`.
    pub kf_boost: i32,
    /// `p_rc->gfu_boost`.
    pub gfu_boost: i32,
    /// `p_rc->gfu_boost_average`.
    pub gfu_boost_average: i32,
    /// `p_rc->arf_boost_factor`. **`float`, not `double`** — C declares it
    /// `float_t` (ratectrl.h:383), which is `float` wherever
    /// `FLT_EVAL_METHOD == 0` (aarch64 and x86-64 both). `boost *
    /// arf_boost_factor` in [`active_best_quality`] is therefore SINGLE
    /// precision, and an `f64` transcription picks a different qindex on
    /// roughly one GF/ARF frame in a thousand.
    pub arf_boost_factor: f32,
    /// `p_rc->arf_q`.
    pub arf_q: i32,
    /// `p_rc->avg_frame_qindex[INTER_FRAME]`.
    pub avg_frame_qindex_inter: i32,
    /// `p_rc->this_key_frame_forced`.
    pub this_key_frame_forced: bool,
    /// `p_rc->last_boosted_qindex`.
    pub last_boosted_qindex: i32,
    /// `p_rc->last_kf_qindex`.
    pub last_kf_qindex: i32,
    /// `rc->frames_to_key`.
    pub frames_to_key: i32,
    /// `rc->frames_since_key`.
    pub frames_since_key: i32,
    /// `rc->best_quality` — the `--min-q` clamp, as a qindex.
    pub best_quality: i32,
    /// `rc->worst_quality` — the `--max-q` clamp, as a qindex.
    pub worst_quality: i32,
    /// `cpi->ppi->twopass.kf_zeromotion_pct`, read only when `two_pass`.
    pub kf_zeromotion_pct: i32,
    /// `cpi->ppi->twopass.last_kfgroup_zeromotion_pct`, ditto.
    pub last_kfgroup_zeromotion_pct: i32,
    /// `is_stat_consumption_stage_twopass(cpi)` — `oxcf.pass >= AOM_RC_SECOND_PASS`.
    pub two_pass: bool,
}

/// The per-frame inputs both q-mode arms need.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameQParams {
    /// `cm->seq_params->bit_depth`.
    pub bit_depth: u8,
    /// `cm->width` / `cm->height` — the CODED frame size, which is what
    /// `res_idx` is derived from.
    pub coded_width: i32,
    /// See [`FrameQParams::coded_width`].
    pub coded_height: i32,
    /// The `width` / `height` arguments, which reach only the
    /// `<= 352 * 288` small-format test in [`intra_q_and_bounds`].
    pub width: i32,
    /// See [`FrameQParams::width`].
    pub height: i32,
    /// `cpi->oxcf.mode == REALTIME`.
    pub rtc_mode: bool,
    /// `cpi->is_screen_content_type`.
    pub screen_content: bool,
    /// `cpi->superres_mode`.
    pub superres_mode: SuperresMode,
    /// `cm->superres_scale_denominator`.
    pub superres_denom: i32,
    /// `cm->tiles.large_scale`.
    pub large_scale: bool,
    /// `cpi->refresh_frame.golden_frame`.
    pub refresh_golden: bool,
    /// `cpi->refresh_frame.alt_ref_frame`.
    pub refresh_alt_ref: bool,
}

impl FrameQParams {
    /// C's `res_idx`: `0` below 480p, `1` at 480p, `2` at 608p and above, on
    /// the SHORTER side of the coded frame (`AOMMIN(cm->width, cm->height)`).
    #[must_use]
    pub fn res_idx(&self) -> usize {
        let min_side = self.coded_width.min(self.coded_height);
        usize::from(min_side >= 480) + usize::from(min_side >= 608)
    }

    /// The minq table cell this frame indexes — `res_idx > 1` collapsed with
    /// `rtc_mode`, exactly what `ASSIGN_MINQ_TABLE_2` selects.
    #[must_use]
    pub fn minq_luts(&self) -> MinqLuts {
        MinqLuts::new(self.bit_depth, self.rtc_mode, self.res_idx() > 1)
    }
}

/// `get_intra_q_and_bounds` (ratectrl.c:1815): the qindex window for a KEY or
/// INTRA_ONLY frame.
///
/// `active_worst_in` is the incoming `*active_worst` (C reads it before writing
/// it back). `rc_mode` selects the first branch; `luts` must be the cell
/// [`FrameQParams::minq_luts`] returns for `p`.
#[must_use]
pub fn intra_q_and_bounds(
    p: &FrameQParams,
    rc: &RcState,
    luts: &MinqLuts,
    rc_mode: RcMode,
    cq_level: i32,
    active_worst_in: i32,
) -> QBounds {
    let bit_depth = p.bit_depth;
    let mut active_worst_quality = active_worst_in;
    let active_best_quality;

    if rc.frames_to_key <= 1 && rc_mode == RcMode::Q {
        // The next frame is also a key frame, or this is the only frame in the
        // sequence: use cq_level directly.
        active_best_quality = cq_level;
        active_worst_quality = cq_level;
    } else if rc.this_key_frame_forced {
        // Forced key frame at the maximum key-frame interval: pin Q to a range
        // around the ambient Q to reduce popping.
        if rc.two_pass && rc.last_kfgroup_zeromotion_pct >= STATIC_MOTION_THRESH {
            let qindex = rc.last_kf_qindex.min(rc.last_boosted_qindex);
            active_best_quality = qindex;
            let last_boosted_q = convert_qindex_to_q(qindex, bit_depth);
            let delta_qindex = compute_qdelta(
                last_boosted_q,
                last_boosted_q * 1.25,
                bit_depth,
                rc.best_quality,
                rc.worst_quality,
            );
            active_worst_quality = (qindex + delta_qindex).min(active_worst_quality);
        } else {
            let qindex = rc.last_boosted_qindex;
            let last_boosted_q = convert_qindex_to_q(qindex, bit_depth);
            let delta_qindex = compute_qdelta(
                last_boosted_q,
                last_boosted_q * 0.50,
                bit_depth,
                rc.best_quality,
                rc.worst_quality,
            );
            active_best_quality = (qindex + delta_qindex).max(rc.best_quality);
        }
    } else {
        // Not a forced key frame.
        let mut q_adj_factor = 1.0f64;
        // Baseline from active_worst_quality and the kf boost.
        let mut abq = kf_active_quality(luts, rc.kf_boost, active_worst_quality, p.rtc_mode);
        if p.screen_content {
            abq /= 2;
        }
        if rc.two_pass && rc.kf_zeromotion_pct >= STATIC_KF_GROUP_THRESH {
            abq /= 3;
        }
        // Allow a somewhat lower kf minq with small image formats.
        if (p.width * p.height) <= (352 * 288) {
            q_adj_factor -= 0.25;
        }
        // Further adjustment from the kf zero-motion measure.
        if rc.two_pass {
            q_adj_factor += 0.05 - (0.001 * f64::from(rc.kf_zeromotion_pct));
        }
        // Convert the adjustment factor into a qindex delta.
        let q_val = convert_qindex_to_q(abq, bit_depth);
        abq += compute_qdelta(
            q_val,
            q_val * q_adj_factor,
            bit_depth,
            rc.best_quality,
            rc.worst_quality,
        );
        // Under AOM_Q with superres on, active_best is used directly as q.
        if rc_mode == RcMode::Q
            && p.superres_mode.adjusts_q()
            && p.superres_denom != SCALE_NUMERATOR
        {
            abq = (abq - ((p.superres_denom - SCALE_NUMERATOR) * SUPERRES_QADJ_PER_DENOM_KEYFRAME))
                .max(0);
        }
        active_best_quality = abq;
    }

    QBounds {
        active_best: active_best_quality,
        active_worst: active_worst_quality,
    }
}

/// `FRAME_UPDATE_TYPE` — re-exported from [`crate::ref_gop`], which owns it.
pub use crate::ref_gop::FrameUpdateType;

/// `get_active_best_quality` (ratectrl.c:2057): the `active_best_quality` for
/// an INTER frame.
///
/// `is_src_frame_alt_ref` is not read here (it belongs to
/// `adjust_active_best_and_worst_quality`). `luts` must be
/// [`FrameQParams::minq_luts`] for `p`.
///
/// # Panics
/// Panics if `active_worst_quality` is outside `0..QINDEX_RANGE`.
// The parameter list is exactly C's read set at this call; grouping it behind
// a context struct would rename the same fields without removing any.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn active_best_quality(
    p: &FrameQParams,
    rc: &RcState,
    luts: &MinqLuts,
    rc_mode: RcMode,
    active_worst_quality: i32,
    cq_level: i32,
    update_type: FrameUpdateType,
    layer_depth: i32,
) -> i32 {
    let is_intrl_arf_boost = update_type == FrameUpdateType::IntnlArf;
    let mut is_leaf_frame = !matches!(
        update_type,
        FrameUpdateType::Arf | FrameUpdateType::Gf | FrameUpdateType::IntnlArf
    );
    // libaom's own hack for the lightfield (large-scale-tile) setting: there,
    // "leaf" is decided from the refresh flags instead of the update type.
    if p.large_scale {
        is_leaf_frame = !(p.refresh_golden || p.refresh_alt_ref || is_intrl_arf_boost);
    }
    let is_overlay_frame = matches!(
        update_type,
        FrameUpdateType::Overlay | FrameUpdateType::IntnlOverlay
    );

    if is_leaf_frame || is_overlay_frame {
        if rc_mode == RcMode::Q {
            return cq_level;
        }
        let idx = usize::try_from(active_worst_quality).expect("active_worst must be a qindex");
        let abq = luts.inter[idx];
        // Constrained quality must not fall below the cq level.
        return if rc_mode == RcMode::Cq && abq < cq_level {
            cq_level
        } else {
            abq
        };
    }

    // Neither leaf nor overlay: a GF / ARF / internal-ARF frame.
    let mut q = active_worst_quality;
    // Use the lower of active_worst_quality and the recent average Q as the
    // basis for the GF/ARF best-Q limit, unless the last frame was a key frame.
    if rc.frames_since_key > 1 && rc.avg_frame_qindex_inter < active_worst_quality {
        q = rc.avg_frame_qindex_inter;
    }
    if rc_mode == RcMode::Cq && q < cq_level {
        q = cq_level;
    }
    let mut abq = gf_active_quality(
        luts,
        rc.gfu_boost,
        rc.gfu_boost_average,
        q,
        p.res_idx(),
        p.rtc_mode,
    );
    // Constrained quality uses a slightly lower active best.
    if rc_mode == RcMode::Cq {
        abq = abq * 15 / 16;
    }
    let min_boost = gf_high_motion_quality(luts, q);
    let boost = min_boost - abq;
    // C: `min_boost - (int)(boost * p_rc->arf_boost_factor)`. `boost` is int
    // and the factor is `float_t`, so the usual arithmetic conversions make
    // this a SINGLE-precision multiply truncated toward zero.
    abq = min_boost - ((boost as f32) * rc.arf_boost_factor) as i32;
    if !is_intrl_arf_boost {
        return abq;
    }

    if matches!(rc_mode, RcMode::Q | RcMode::Cq) {
        abq = rc.arf_q;
    }
    // Halve the gap to active_worst once per pyramid level below the top.
    let mut this_height = gf_group_pyramid_level(layer_depth);
    while this_height > 1 {
        abq = (abq + active_worst_quality + 1) / 2;
        this_height -= 1;
    }
    abq
}

/// The output of [`pick_q_and_bounds_q_mode`]: the picked qindex plus the
/// `[bottom_index, top_index]` bounds C writes through pointers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PickedQ {
    /// The chosen qindex — under `AOM_Q` this is always `bottom_index`.
    pub q: i32,
    /// `*bottom_index`, the best-quality bound.
    pub bottom_index: i32,
    /// `*top_index`, the worst-quality bound.
    pub top_index: i32,
}

/// `rc_pick_q_and_bounds_q_mode` (ratectrl.c:2133): the whole `AOM_Q` qindex
/// decision for one frame.
///
/// `intra_only` is C's `frame_is_intra_only(cm)` — `KEY_FRAME ||
/// INTRA_ONLY_FRAME`. `active_worst_in` is `rc->active_worst_quality`.
///
/// The `cq_level` passed to the two arms is [`active_cq_level`]'s output, not
/// the raw configured value — the superres adjustment happens once, here.
// The parameter list is exactly C's read set at this call; grouping it behind
// a context struct would rename the same fields without removing any.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn pick_q_and_bounds_q_mode(
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
) -> PickedQ {
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

    let (mut active_best_q, active_worst_quality) = if intra_only {
        let b = intra_q_and_bounds(p, rc, &luts, rc_mode, cq_level, active_worst_in);
        (b.active_best, b.active_worst)
    } else {
        (
            active_best_quality(
                p,
                rc,
                &luts,
                rc_mode,
                active_worst_in,
                cq_level,
                update_type,
                layer_depth,
            ),
            active_worst_in,
        )
    };

    if cq_level > 0 {
        active_best_q = active_best_q.max(1);
    }

    // C's `clamp` macro, NOT Rust's `clamp` — the latter panics when
    // `best_quality > worst_quality`, a config aomenc rejects but this function
    // does not itself validate.
    let clamp_c = |v: i32, lo: i32, hi: i32| {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    };
    let top_index = clamp_c(active_worst_quality, rc.best_quality, rc.worst_quality);
    let bottom_index = clamp_c(active_best_q, rc.best_quality, rc.worst_quality);

    PickedQ {
        q: bottom_index,
        bottom_index,
        top_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minq_index_is_zero_below_the_lossless_step() {
        // minqtarget <= 2.0 is the lossless special case; a maxq of 0 makes the
        // polynomial 0 for every coefficient set.
        assert_eq!(minq_index(0.0, 0.000001, -0.0004, 0.1771, 8), 0);
    }

    #[test]
    fn minq_luts_are_monotonic_and_bounded() {
        for bd in [8u8, 10, 12] {
            for rtc in [false, true] {
                for hi in [false, true] {
                    let l = MinqLuts::new(bd, rtc, hi);
                    for t in [&l.kf_low, &l.kf_high, &l.arfgf_low, &l.arfgf_high, &l.inter] {
                        assert!(t.iter().all(|&v| (0..=MAXQ).contains(&v)));
                        assert!(
                            t.windows(2).all(|w| w[0] <= w[1]),
                            "minq curves are monotonic in qindex (bd={bd} rtc={rtc} hi={hi})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn res_idx_has_three_steps_on_the_shorter_side() {
        let mk = |w, h| FrameQParams {
            bit_depth: 8,
            coded_width: w,
            coded_height: h,
            width: w,
            height: h,
            rtc_mode: false,
            screen_content: false,
            superres_mode: SuperresMode::None,
            superres_denom: SCALE_NUMERATOR,
            large_scale: false,
            refresh_golden: false,
            refresh_alt_ref: false,
        };
        assert_eq!(mk(1920, 479).res_idx(), 0);
        assert_eq!(mk(1920, 480).res_idx(), 1);
        assert_eq!(mk(1920, 607).res_idx(), 1);
        assert_eq!(mk(1920, 608).res_idx(), 2);
        // The SHORTER side decides, so a tall narrow frame is low-res.
        assert_eq!(mk(300, 2000).res_idx(), 0);
    }

    #[test]
    fn active_quality_interpolates_and_saturates() {
        let mut low = [0i32; QINDEX_RANGE];
        let mut high = [0i32; QINDEX_RANGE];
        low[10] = 100;
        high[10] = 200;
        // Above `high` the low-motion curve wins, below `low` the high-motion.
        assert_eq!(active_quality(10, 9000, 553, 8000, &low, &high), 100);
        assert_eq!(active_quality(10, 1, 553, 8000, &low, &high), 200);
        // Exactly at the endpoints C takes the interpolating branch.
        assert_eq!(active_quality(10, 8000, 553, 8000, &low, &high), 100);
        assert_eq!(active_quality(10, 553, 553, 8000, &low, &high), 200);
    }

    #[test]
    fn q_mode_leaf_inter_frame_is_the_cq_level() {
        // The lag=0 low-delay P case that crate::rc::base_qindex_lowdelay_p_from_cq
        // documents: an LF_UPDATE leaf under AOM_Q returns cq_level verbatim.
        let p = FrameQParams {
            bit_depth: 8,
            coded_width: 352,
            coded_height: 288,
            width: 352,
            height: 288,
            rtc_mode: false,
            screen_content: false,
            superres_mode: SuperresMode::None,
            superres_denom: SCALE_NUMERATOR,
            large_scale: false,
            refresh_golden: false,
            refresh_alt_ref: false,
        };
        let rc = RcState {
            best_quality: 0,
            worst_quality: 255,
            ..Default::default()
        };
        for cq in 1..=255 {
            let got = pick_q_and_bounds_q_mode(
                &p,
                &rc,
                RcMode::Q,
                cq,
                255,
                false,
                FrameUpdateType::Lf,
                0,
                0,
                0,
            );
            assert_eq!(got.q, cq);
            assert_eq!(got.bottom_index, cq);
        }
    }
}
