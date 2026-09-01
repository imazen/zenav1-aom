//! The ref-MV / DRL layer of libaom's inter RD brain (`av1/encoder/rdopt.c`).
//!
//! Everything here is `static` in C — `nm -g upstream/build/libaom.a` reports
//! ten exported symbols for the whole of rdopt.c, and none of these is one of
//! them. The oracle is `crates/aom-sys-ref/shim/rdopt_shim.c`, which compiles
//! libaom's own rdopt.c into the shim archive and exposes flat wrappers around
//! the statics; the gate is `crates/aom-encode/tests/rdopt_mv_diff.rs`. Read
//! the shim's header for why that is the real C rather than a transcription of
//! it, and for the measurement that backs the claim.
//!
//! # What is here
//!
//! The functions that turn a populated ref-MV stack into the candidate motion
//! vectors an inter mode is evaluated at, plus the DRL rate and the small
//! pruning predicates that gate the ref-MV loop:
//!
//! | Rust | C (`av1/encoder/rdopt.c`) |
//! |---|---|
//! | [`ref_frame_type`] | `av1_ref_frame_type` (`mvref_common.h:113`) |
//! | [`compound_ref0_mode`] / [`compound_ref1_mode`] | `blockd.h:85` / `:118` |
//! | [`get_single_mode`] | `:989` |
//! | [`check_repeat_ref_mv`] | `:1993` |
//! | [`get_this_mv`] | `:2030` |
//! | [`build_cur_mv`] | `:2110` |
//! | [`clamp_mv2`] | `:1227` |
//! | [`clamp_and_check_mv`] | `:1293` |
//! | [`get_drl_cost`] | `:2139` |
//! | [`get_drl_refmv_count`] | `:2182` |
//! | [`is_single_newmv_valid`] | `:2168` |
//! | [`prune_ref_mv_idx_using_qindex`] | `:2199` |
//! | [`skip_nearest_near_mv_using_refmv_weight`] | `:2069` |
//! | [`conditional_skipintra`] | `:941` |
//! | [`IdxMask`] | `mask_set_bit` `:2347` / `mask_check_bit` `:2349` |
//!
//! # Translation notes (where this is not a transliteration)
//!
//! - **`PredMode` is an enum, not an `int`.** C passes `PREDICTION_MODE`
//!   around as a small integer and uses `MB_MODE_COUNT` (25) as "no such
//!   mode"; `compound_ref1_mode` returns it for every single-reference mode.
//!   Here that sentinel is `Option::None`, so a caller cannot index a LUT with
//!   it by accident. The numeric values are unchanged (`#[repr(i32)]`) because
//!   they are the bitstream's.
//! - **`INVALID_MV` stays a value, not an `Option`.** C's `get_this_mv` writes
//!   `INVALID_MV` into `*this_mv` *and returns 1* for a NEWMV component,
//!   because `build_cur_mv` overwrites it from the stack immediately after.
//!   Modelling that as `None` would conflate it with C's `return 0`, which is
//!   a different outcome (skip the candidate). So [`Mv::INVALID`] is a real
//!   `Mv`, and the `return 0` path is the `Option`.
//! - **`build_cur_mv` takes `&mut [Mv; 2]`.** C's `cur_mv` is a caller-owned
//!   array that is PARTIALLY written on the failing path — the second slot
//!   keeps whatever the caller had. Returning a fresh array would silently
//!   change that, so the in/out buffer is kept and the boolean is separate.
//! - **Integer widths.** `weight` is `uint16_t` in C and `u16` here;
//!   `ref_mv_count` is `uint8_t` in the struct but is compared as `int`
//!   throughout, so it is `usize` here and every comparison is done on values
//!   that cannot go negative. `get_drl_cost` accumulates in `int`, kept `i32`.
//!   The `mb_to_*_edge` fields are `int` 1/8-pel distances that go negative,
//!   and the `LEFT_TOP_MARGIN` arithmetic in `clamp_mv2` is done in `i32`
//!   before the `i16` MV is clamped into range — same as C, which clamps an
//!   `int16_t` field against `int` limits.

/// `INTRA_FRAME` (`av1/common/enums.h`) — also the "no second reference"
/// marker in `ref_frame[1]`; `NONE_FRAME` is `-1`.
pub const INTRA_FRAME: i32 = 0;
/// `LAST_FRAME`.
pub const LAST_FRAME: i32 = 1;
/// `LAST2_FRAME`.
pub const LAST2_FRAME: i32 = 2;
/// `LAST3_FRAME`.
pub const LAST3_FRAME: i32 = 3;
/// `GOLDEN_FRAME`.
pub const GOLDEN_FRAME: i32 = 4;
/// `BWDREF_FRAME`.
pub const BWDREF_FRAME: i32 = 5;
/// `ALTREF2_FRAME`.
pub const ALTREF2_FRAME: i32 = 6;
/// `ALTREF_FRAME`.
pub const ALTREF_FRAME: i32 = 7;
/// `REF_FRAMES`.
pub const REF_FRAMES: usize = 8;
/// `FWD_REFS` = `GOLDEN_FRAME - LAST_FRAME + 1`.
pub const FWD_REFS: i32 = 4;
/// `TOTAL_UNIDIR_COMP_REFS` (`enums.h:585`).
pub const TOTAL_UNIDIR_COMP_REFS: usize = 9;
/// `MODE_CTX_REF_FRAMES` = `REF_FRAMES + TOTAL_COMP_REFS` = 8 + (4*3 + 9).
pub const MODE_CTX_REF_FRAMES: usize = 29;
/// `MAX_REF_MV_STACK_SIZE` (`enums.h:510`).
pub const MAX_REF_MV_STACK_SIZE: usize = 8;
/// `USABLE_REF_MV_STACK_SIZE` (`enums.h:511`).
pub const USABLE_REF_MV_STACK_SIZE: usize = 4;
/// `REF_CAT_LEVEL` (`enums.h:512`) — the ref-MV weight that marks a candidate
/// as having come from a *nearest* spatial neighbour.
pub const REF_CAT_LEVEL: u16 = 640;
/// `MAX_REF_MV_SEARCH` (`av1/encoder/rdopt_utils.h:25`).
pub const MAX_REF_MV_SEARCH: usize = 3;
/// `DRL_MODE_CONTEXTS`.
pub const DRL_MODE_CONTEXTS: usize = 3;
/// `QINDEX_RANGE`.
pub const QINDEX_RANGE: i32 = 256;
/// `AOM_BORDER_IN_PIXELS` (`aom_scale/yv12config.h`).
const AOM_BORDER_IN_PIXELS: i32 = 288;
/// `AOM_INTERP_EXTEND` (`av1/common/filter.h`).
const AOM_INTERP_EXTEND: i32 = 4;
/// `LEFT_TOP_MARGIN` / `RIGHT_BOTTOM_MARGIN` (rdopt.c:1223-1224) — identical
/// expressions in C, kept as one constant here.
const MV_BORDER_MARGIN: i32 = (AOM_BORDER_IN_PIXELS - AOM_INTERP_EXTEND) << 3;

/// A motion vector in 1/8-pel units — C's `MV { int16_t row, col; }`.
///
/// C also passes this through an `int_mv` union whose `as_int` is compared for
/// equality; that is exactly `PartialEq` on the pair, so the union is not
/// reproduced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Mv {
    /// Vertical component, 1/8-pel.
    pub row: i16,
    /// Horizontal component, 1/8-pel.
    pub col: i16,
}

impl Mv {
    /// `INVALID_MV` (`av1/common/mv.h:26`), i.e. `as_int == 0x80008000`, which
    /// is `row == col == INVALID_MV_ROW_COL == -32768`.
    pub const INVALID: Self = Self {
        row: i16::MIN,
        col: i16::MIN,
    };

    /// Construct from a `(row, col)` pair.
    pub const fn new(row: i16, col: i16) -> Self {
        Self { row, col }
    }
}

/// `PREDICTION_MODE` (`av1/common/enums.h`). The discriminants are the
/// bitstream's own numbering and are load-bearing (they index every mode LUT
/// and CDF in the format), so they are pinned with `#[repr(i32)]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
#[allow(missing_docs)]
pub enum PredMode {
    DcPred = 0,
    VPred = 1,
    HPred = 2,
    D45Pred = 3,
    D135Pred = 4,
    D113Pred = 5,
    D157Pred = 6,
    D203Pred = 7,
    D67Pred = 8,
    SmoothPred = 9,
    SmoothVPred = 10,
    SmoothHPred = 11,
    PaethPred = 12,
    NearestMv = 13,
    NearMv = 14,
    GlobalMv = 15,
    NewMv = 16,
    NearestNearestMv = 17,
    NearNearMv = 18,
    NearestNewMv = 19,
    NewNearestMv = 20,
    NearNewMv = 21,
    NewNearMv = 22,
    GlobalGlobalMv = 23,
    NewNewMv = 24,
}

/// `MB_MODE_COUNT` — C's "not a mode" sentinel, which this port spells
/// `Option::None`. Exposed because the C ABI at the shim boundary still uses
/// the integer.
pub const MB_MODE_COUNT: i32 = 25;

impl PredMode {
    /// The numeric `PREDICTION_MODE` C uses.
    pub const fn to_i32(self) -> i32 {
        self as i32
    }

    /// Parse a `PREDICTION_MODE`. `MB_MODE_COUNT` and anything out of range
    /// are `None`; C would index a LUT out of bounds.
    pub const fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            0 => Self::DcPred,
            1 => Self::VPred,
            2 => Self::HPred,
            3 => Self::D45Pred,
            4 => Self::D135Pred,
            5 => Self::D113Pred,
            6 => Self::D157Pred,
            7 => Self::D203Pred,
            8 => Self::D67Pred,
            9 => Self::SmoothPred,
            10 => Self::SmoothVPred,
            11 => Self::SmoothHPred,
            12 => Self::PaethPred,
            13 => Self::NearestMv,
            14 => Self::NearMv,
            15 => Self::GlobalMv,
            16 => Self::NewMv,
            17 => Self::NearestNearestMv,
            18 => Self::NearNearMv,
            19 => Self::NearestNewMv,
            20 => Self::NewNearestMv,
            21 => Self::NearNewMv,
            22 => Self::NewNearMv,
            23 => Self::GlobalGlobalMv,
            24 => Self::NewNewMv,
            _ => return None,
        })
    }

    /// `is_inter_singleref_mode` (`blockd.h`): `NEARESTMV..=NEWMV`.
    pub const fn is_inter_singleref(self) -> bool {
        (self as i32) >= Self::NearestMv as i32 && (self as i32) <= Self::NewMv as i32
    }

    /// `is_inter_compound_mode` (`blockd.h`): `NEAREST_NEARESTMV..=NEW_NEWMV`.
    pub const fn is_inter_compound(self) -> bool {
        (self as i32) >= Self::NearestNearestMv as i32 && (self as i32) <= Self::NewNewMv as i32
    }

    /// `is_inter_mode` (`blockd.h`): any of the twelve inter modes.
    pub const fn is_inter(self) -> bool {
        self.is_inter_singleref() || self.is_inter_compound()
    }

    /// `have_nearmv_in_inter_mode` (`blockd.h:151`).
    pub const fn have_nearmv(self) -> bool {
        matches!(
            self,
            Self::NearMv | Self::NearNearMv | Self::NearNewMv | Self::NewNearMv
        )
    }

    /// `have_newmv_in_inter_mode` (`blockd.h:156`).
    pub const fn have_newmv(self) -> bool {
        matches!(
            self,
            Self::NewMv
                | Self::NewNewMv
                | Self::NearestNewMv
                | Self::NewNearestMv
                | Self::NearNewMv
                | Self::NewNearMv
        )
    }
}

/// `compound_ref0_mode` (`blockd.h:85`) — the single-reference mode the first
/// reference of `mode` behaves as. Total over the twelve inter modes; C's LUT
/// also maps the thirteen intra modes to themselves, which no caller uses (its
/// own assert forbids them), so those are `None` here.
pub const fn compound_ref0_mode(mode: PredMode) -> Option<PredMode> {
    use PredMode::*;
    Some(match mode {
        NearestMv | NearestNearestMv | NearestNewMv => NearestMv,
        NearMv | NearNearMv | NearNewMv => NearMv,
        GlobalMv | GlobalGlobalMv => GlobalMv,
        NewMv | NewNearestMv | NewNearMv | NewNewMv => NewMv,
        _ => return None,
    })
}

/// `compound_ref1_mode` (`blockd.h:118`) — the single-reference mode the
/// SECOND reference behaves as. `None` is C's `MB_MODE_COUNT`, returned for
/// every non-compound mode.
pub const fn compound_ref1_mode(mode: PredMode) -> Option<PredMode> {
    use PredMode::*;
    Some(match mode {
        NearestNearestMv | NewNearestMv => NearestMv,
        NearNearMv | NewNearMv => NearMv,
        NearestNewMv | NearNewMv | NewNewMv => NewMv,
        GlobalGlobalMv => GlobalMv,
        _ => return None,
    })
}

/// `get_single_mode` (rdopt.c:989): the per-reference-slot single mode.
pub const fn get_single_mode(this_mode: PredMode, ref_idx: usize) -> Option<PredMode> {
    if ref_idx != 0 {
        compound_ref1_mode(this_mode)
    } else {
        compound_ref0_mode(this_mode)
    }
}

/// `get_uni_comp_ref_idx` (`mvref_common.h:99`): the index of a unidirectional
/// compound reference pair, or `None` for single-reference and bidirectional
/// pairs.
pub fn uni_comp_ref_idx(rf: [i32; 2]) -> Option<usize> {
    // `comp_ref0` / `comp_ref1` (blockd.h:385 / :401), as one table.
    const UNIDIR: [[i32; 2]; TOTAL_UNIDIR_COMP_REFS] = [
        [LAST_FRAME, LAST2_FRAME],
        [LAST_FRAME, LAST3_FRAME],
        [LAST_FRAME, GOLDEN_FRAME],
        [BWDREF_FRAME, ALTREF_FRAME],
        [LAST2_FRAME, LAST3_FRAME],
        [LAST2_FRAME, GOLDEN_FRAME],
        [LAST3_FRAME, GOLDEN_FRAME],
        [BWDREF_FRAME, ALTREF2_FRAME],
        [ALTREF2_FRAME, ALTREF_FRAME],
    ];
    if rf[1] <= INTRA_FRAME {
        return None;
    }
    if rf[0] < BWDREF_FRAME && rf[1] >= BWDREF_FRAME {
        return None;
    }
    UNIDIR.iter().position(|pair| *pair == rf)
}

/// `av1_ref_frame_type` (`mvref_common.h:113`): the `mbmi_ext` row a reference
/// pair selects. Single reference maps to `rf[0]`; compound pairs map into the
/// `REF_FRAMES..MODE_CTX_REF_FRAMES` range.
pub fn ref_frame_type(rf: [i32; 2]) -> usize {
    if rf[1] > INTRA_FRAME {
        let row = match uni_comp_ref_idx(rf) {
            Some(idx) => REF_FRAMES as i32 + FWD_REFS * 3 + idx as i32,
            None => REF_FRAMES as i32 + (rf[0] - LAST_FRAME) + (rf[1] - BWDREF_FRAME) * FWD_REFS,
        };
        debug_assert!((0..MODE_CTX_REF_FRAMES as i32).contains(&row));
        row as usize
    } else {
        debug_assert!(rf[0] >= 0);
        rf[0] as usize
    }
}

/// One `mbmi_ext` ref-MV row: `ref_mv_stack[t]` + `weight[t]` +
/// `ref_mv_count[t]` for a single `t = ref_frame_type(rf)`.
///
/// C keeps all `MODE_CTX_REF_FRAMES` rows in one struct and indexes them; the
/// helpers here each read exactly one row, so the row is the argument. That
/// removes the "which index does this function use" question the C reader has
/// to answer at every call site.
#[derive(Clone, Debug)]
pub struct RefMvRow {
    /// `ref_mv_count[t]` — how many of the eight stack slots are populated.
    pub count: usize,
    /// `ref_mv_stack[t][i].this_mv`.
    pub this_mv: [Mv; MAX_REF_MV_STACK_SIZE],
    /// `ref_mv_stack[t][i].comp_mv` (only meaningful for a compound row).
    pub comp_mv: [Mv; MAX_REF_MV_STACK_SIZE],
    /// `weight[t][i]`, compared against [`REF_CAT_LEVEL`].
    pub weight: [u16; MAX_REF_MV_STACK_SIZE],
}

impl Default for RefMvRow {
    fn default() -> Self {
        Self {
            count: 0,
            this_mv: [Mv::default(); MAX_REF_MV_STACK_SIZE],
            comp_mv: [Mv::default(); MAX_REF_MV_STACK_SIZE],
            weight: [0; MAX_REF_MV_STACK_SIZE],
        }
    }
}

impl RefMvRow {
    /// `ref_mv_stack[t][i].this_mv` or `.comp_mv`, picked by the reference
    /// slot — C spells this as an `if (ref_idx == 0)` at four call sites.
    pub fn stack_mv(&self, ref_idx: usize, i: usize) -> Mv {
        if ref_idx == 0 {
            self.this_mv[i]
        } else {
            self.comp_mv[i]
        }
    }
}

/// The block-edge distances `clamp_mv2` reads out of `MACROBLOCKD`, in 1/8-pel
/// units. `left`/`top` are negative, `right`/`bottom` positive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockEdges {
    /// `xd->mb_to_left_edge`.
    pub left: i32,
    /// `xd->mb_to_right_edge`.
    pub right: i32,
    /// `xd->mb_to_top_edge`.
    pub top: i32,
    /// `xd->mb_to_bottom_edge`.
    pub bottom: i32,
}

/// `GET_MV_RAWPEL` (`av1/common/mv.h:28`): a 1/8-pel component to full pel.
///
/// This is NOT `>> 3`. C is `((x) + 3 + ((x) >= 0)) >> 3` — a round-to-nearest
/// with the tie broken towards positive infinity for non-negative inputs and
/// towards negative infinity for negative ones. A plain arithmetic shift
/// differs on 4 of every 8 values and shows up as a wrong `av1_is_fullmv_in_range`
/// verdict at the search-window edge, which was measured against the C oracle
/// rather than reasoned about.
pub const fn get_mv_rawpel(v: i16) -> i32 {
    let x = v as i32;
    (x + 3 + (x >= 0) as i32) >> 3
}

/// `FullMvLimits` (`av1/common/mv.h`) — `x->mv_limits`, in FULL-pel units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FullMvLimits {
    /// `col_min`.
    pub col_min: i32,
    /// `col_max`.
    pub col_max: i32,
    /// `row_min`.
    pub row_min: i32,
    /// `row_max`.
    pub row_max: i32,
}

impl FullMvLimits {
    /// `av1_is_fullmv_in_range` (`mcomp.h:268`).
    pub const fn contains(&self, row: i32, col: i32) -> bool {
        col >= self.col_min && col <= self.col_max && row >= self.row_min && row <= self.row_max
    }
}

/// `clamp_mv2` (rdopt.c:1227): clamp a subpel MV into the block's own extended
/// search window (the frame edge plus the interpolation border).
pub fn clamp_mv2(mv: Mv, edges: BlockEdges) -> Mv {
    let col_min = edges.left - MV_BORDER_MARGIN;
    let col_max = edges.right + MV_BORDER_MARGIN;
    let row_min = edges.top - MV_BORDER_MARGIN;
    let row_max = edges.bottom + MV_BORDER_MARGIN;
    // C clamps the int16_t fields against int limits, so the comparison and
    // the clamp both happen in `int`; the result is then stored back into an
    // int16_t. Widening to i32 here reproduces that exactly, and the saturating
    // cast never fires because the limits are themselves in i16 range whenever
    // the input is.
    Mv {
        row: i32::from(mv.row).clamp(row_min, row_max) as i16,
        col: i32::from(mv.col).clamp(col_min, col_max) as i16,
    }
}

/// `integer_mv_precision` (`av1/common/mv.h:199`): round one 1/8-pel component
/// to the nearest whole pel, ties (`|mod| == 4`) TOWARDS zero.
///
/// C does this in `int16_t`, and `mv->row -= mod` followed by `+= 8` can leave
/// the `int16_t` range for a component within 7 of `i16::MAX` — signed
/// overflow, which is UB in C and in practice wraps. `wrapping_*` reproduces
/// the wrap rather than panicking in a debug build; no caller can reach it,
/// because every MV here has already been clamped into the block window.
fn integer_mv_component(v: i16) -> i16 {
    // Rust's `%` truncates towards zero exactly as C's does, so `mod` carries
    // the sign of `v` in both languages.
    let m = v % 8;
    if m == 0 {
        return v;
    }
    let rounded = v.wrapping_sub(m);
    if m.abs() > 4 {
        if m > 0 {
            rounded.wrapping_add(8)
        } else {
            rounded.wrapping_sub(8)
        }
    } else {
        rounded
    }
}

/// `lower_mv_precision` (`mvref_common.h:88`): force whole-pel, else drop odd
/// 1/8-pel components (which 1/4-pel cannot represent) one step towards zero.
fn lower_mv_precision(mv: Mv, allow_hp: bool, is_integer: bool) -> Mv {
    if is_integer {
        return Mv {
            row: integer_mv_component(mv.row),
            col: integer_mv_component(mv.col),
        };
    }
    if allow_hp {
        return mv;
    }
    let drop_odd = |v: i16| {
        if v & 1 != 0 {
            v + if v > 0 { -1 } else { 1 }
        } else {
            v
        }
    };
    Mv {
        row: drop_odd(mv.row),
        col: drop_odd(mv.col),
    }
}

/// `clamp_and_check_mv` (rdopt.c:1293): precision-lower, clamp into the block
/// window, and report whether the result is still inside the full-pel search
/// limits. Returns the clamped MV either way — C writes `*out_mv`
/// unconditionally and only the boolean tells the caller to drop it.
pub fn clamp_and_check_mv(
    in_mv: Mv,
    allow_high_precision_mv: bool,
    cur_frame_force_integer_mv: bool,
    edges: BlockEdges,
    limits: FullMvLimits,
) -> (Mv, bool) {
    let lowered = lower_mv_precision(in_mv, allow_high_precision_mv, cur_frame_force_integer_mv);
    let out = clamp_mv2(lowered, edges);
    let ok = limits.contains(get_mv_rawpel(out.row), get_mv_rawpel(out.col));
    (out, ok)
}

/// `check_repeat_ref_mv` (rdopt.c:1993): is this single mode's MV already
/// covered by a cheaper mode for the same reference?
///
/// `global_mv` is `mbmi_ext->global_mvs[ref_frame[ref_idx]]`, resolved by the
/// caller — C indexes it inside, but the row is the only thing the function
/// needs from the wider struct.
pub fn check_repeat_ref_mv(
    row: &RefMvRow,
    ref_idx: usize,
    single_mode: PredMode,
    global_mv: Mv,
) -> bool {
    debug_assert_ne!(single_mode, PredMode::NewMv);
    match single_mode {
        PredMode::NearestMv => false,
        // ref_mv_count 0: NEARESTMV and NEARMV both collapse to GLOBALMV.
        // ref_mv_count 1: NEARMV alone collapses to GLOBALMV.
        PredMode::NearMv => row.count < 2,
        PredMode::GlobalMv => match row.count {
            0 => true,
            1 => false,
            n => row
                .this_mv
                .iter()
                .zip(row.comp_mv.iter())
                .take(USABLE_REF_MV_STACK_SIZE.min(n))
                .any(|(this, comp)| *(if ref_idx == 0 { this } else { comp }) == global_mv),
        },
        _ => false,
    }
}

/// `get_this_mv` (rdopt.c:2030): the MV one reference slot of `this_mode`
/// predicts from. `None` is C's `return 0` — the candidate is a repeat of a
/// cheaper one and the caller must drop it.
///
/// A NEWMV slot yields [`Mv::INVALID`] and still succeeds, exactly as C does;
/// [`build_cur_mv`] overwrites it from the stack.
pub fn get_this_mv(
    this_mode: PredMode,
    ref_idx: usize,
    ref_mv_idx: usize,
    skip_repeated_ref_mv: bool,
    row: &RefMvRow,
    global_mv: Mv,
) -> Option<Mv> {
    let single_mode = get_single_mode(this_mode, ref_idx)?;
    debug_assert!(single_mode.is_inter_singleref());
    match single_mode {
        PredMode::NewMv => Some(Mv::INVALID),
        PredMode::GlobalMv => {
            if skip_repeated_ref_mv && check_repeat_ref_mv(row, ref_idx, single_mode, global_mv) {
                return None;
            }
            Some(global_mv)
        }
        PredMode::NearestMv | PredMode::NearMv => {
            let offset = if single_mode == PredMode::NearestMv {
                0
            } else {
                ref_mv_idx + 1
            };
            if offset < row.count {
                Some(row.stack_mv(ref_idx, offset))
            } else {
                if skip_repeated_ref_mv && check_repeat_ref_mv(row, ref_idx, single_mode, global_mv)
                {
                    return None;
                }
                Some(global_mv)
            }
        }
        // C asserts NEARMV || NEARESTMV here, so anything else is a caller bug.
        _ => None,
    }
}

/// `build_cur_mv` (rdopt.c:2110): fill both reference slots' motion vectors
/// for a non-NEWMV evaluation of `this_mode`.
///
/// `cur_mv` is in/out because C's is: on the `get_this_mv` failure path the
/// later slot keeps the caller's value, and on the clamp-failure path both
/// slots ARE written before `false` is returned.
#[allow(clippy::too_many_arguments)]
pub fn build_cur_mv(
    cur_mv: &mut [Mv; 2],
    this_mode: PredMode,
    is_comp_pred: bool,
    ref_mv_idx: usize,
    skip_repeated_ref_mv: bool,
    row: &RefMvRow,
    global_mvs: [Mv; 2],
    allow_high_precision_mv: bool,
    cur_frame_force_integer_mv: bool,
    edges: BlockEdges,
    limits: FullMvLimits,
) -> bool {
    let mut ret = true;
    for i in 0..(usize::from(is_comp_pred) + 1) {
        let Some(this_mv) = get_this_mv(
            this_mode,
            i,
            ref_mv_idx,
            skip_repeated_ref_mv,
            row,
            global_mvs[i],
        ) else {
            return false;
        };
        // C reads `ret = get_this_mv(...)` — an ASSIGNMENT, not an `&=`. So a
        // clamp failure recorded for the first reference is DISCARDED when the
        // second reference's `get_this_mv` succeeds, and the returned flag
        // describes only the last slot that went through `clamp_and_check_mv`.
        // Reachable and observed: NEAREST_NEWMV with a first-slot MV outside
        // the search window returns 1 from C because slot 1 is NEWMV and
        // resets the flag. Reproduced deliberately; "fixing" it fails the
        // differential.
        ret = true;
        if get_single_mode(this_mode, i) == Some(PredMode::NewMv) {
            cur_mv[i] = row.stack_mv(i, ref_mv_idx);
        } else {
            let (clamped, ok) = clamp_and_check_mv(
                this_mv,
                allow_high_precision_mv,
                cur_frame_force_integer_mv,
                edges,
                limits,
            );
            cur_mv[i] = clamped;
            ret &= ok;
        }
    }
    ret
}

/// `av1_drl_ctx` (`mvref_common.h:185`): the DRL CDF context from the two
/// candidate weights straddling `ref_idx`.
///
/// C's fourth combination (`w[i] < LEVEL && w[i+1] >= LEVEL`) falls off the end
/// of its `if` chain and returns 0; it is spelled out here rather than left
/// implicit.
pub fn drl_ctx(weight: &[u16; MAX_REF_MV_STACK_SIZE], ref_idx: usize) -> usize {
    match (
        weight[ref_idx] >= REF_CAT_LEVEL,
        weight[ref_idx + 1] >= REF_CAT_LEVEL,
    ) {
        (true, true) => 0,
        (true, false) => 1,
        (false, false) => 2,
        (false, true) => 0,
    }
}

/// `get_drl_cost` (rdopt.c:2139): the rate of signalling `ref_mv_idx`.
///
/// NEWMV modes scan indices 0..2; NEAR modes scan 1..3 and code
/// `ref_mv_idx == idx - 1` (the NEARESTMV slot is implicit). Every other mode
/// costs nothing — it has no DRL index in the bitstream.
pub fn get_drl_cost(
    mode: PredMode,
    ref_mv_idx: usize,
    row: &RefMvRow,
    drl_mode_cost0: &[[i32; 2]; DRL_MODE_CONTEXTS],
) -> i32 {
    let (range, coded_offset) = if matches!(mode, PredMode::NewMv | PredMode::NewNewMv) {
        (0..2, 0)
    } else if mode.have_nearmv() {
        (1..3, 1)
    } else {
        return 0;
    };
    let mut cost = 0;
    for idx in range {
        if row.count > idx + 1 {
            let ctx = drl_ctx(&row.weight, idx);
            cost += drl_mode_cost0[ctx][usize::from(ref_mv_idx != idx - coded_offset)];
            if ref_mv_idx == idx - coded_offset {
                return cost;
            }
        }
    }
    cost
}

/// `get_drl_refmv_count` (rdopt.c:2182): how many DRL indices this mode is
/// allowed to search.
pub fn get_drl_refmv_count(mode: PredMode, ref_mv_count: usize) -> usize {
    let has_nearmv = usize::from(mode.have_nearmv());
    let only_newmv = matches!(mode, PredMode::NewMv | PredMode::NewNewMv);
    let has_drl = (has_nearmv == 1 && ref_mv_count > 2) || (only_newmv && ref_mv_count > 1);
    if has_drl {
        MAX_REF_MV_SEARCH.min(ref_mv_count - has_nearmv)
    } else {
        1
    }
}

/// `is_single_newmv_valid` (rdopt.c:2168): both compound halves that use NEWMV
/// must have found a valid single-reference motion vector at this DRL index.
///
/// `single_newmv_valid` is `args->single_newmv_valid[ref_mv_idx]`, i.e. the
/// row for the index under test, indexed by reference frame.
pub fn is_single_newmv_valid(
    this_mode: PredMode,
    ref_frame: [i32; 2],
    single_newmv_valid: &[bool; REF_FRAMES],
) -> bool {
    (0..2).all(|ref_idx| {
        if get_single_mode(this_mode, ref_idx) != Some(PredMode::NewMv) {
            return true;
        }
        let r = ref_frame[ref_idx];
        // C indexes `[ref]` with `mbmi->ref_frame[ref_idx]`, which is
        // NONE_FRAME (-1) for an absent second reference — but it only gets
        // there when that slot's single mode is NEWMV, which never happens for
        // a single-reference mode. Reject rather than reproduce the UB.
        (0..REF_FRAMES as i32).contains(&r) && single_newmv_valid[r as usize]
    })
}

/// `prune_ref_mv_idx_using_qindex` (rdopt.c:2199).
///
/// `reduce_inter_modes >= 3` prunes unconditionally; at exactly 2 the cut-off
/// rises with the quantiser (q 0–85 keeps only index 0, 86–170 keeps 0–1,
/// 171–255 keeps everything). C asserts `reduce_inter_modes == 2` on the
/// second path, so anything below 2 is a caller bug.
pub fn prune_ref_mv_idx_using_qindex(
    reduce_inter_modes: i32,
    qindex: i32,
    ref_mv_idx: i32,
) -> bool {
    if reduce_inter_modes >= 3 {
        return true;
    }
    debug_assert_eq!(reduce_inter_modes, 2);
    let min_prune_ref_mv_idx = (qindex * 3 / QINDEX_RANGE) + 1;
    ref_mv_idx >= min_prune_ref_mv_idx
}

/// `skip_nearest_near_mv_using_refmv_weight` (rdopt.c:2069): prune NEARESTMV /
/// NEARMV when few of the ref-MV candidates came from *nearest* neighbours.
pub fn skip_nearest_near_mv_using_refmv_weight(
    this_mode: PredMode,
    best_mode: Option<PredMode>,
    left_available: bool,
    up_available: bool,
    row: &RefMvRow,
) -> bool {
    if !matches!(this_mode, PredMode::NearestMv | PredMode::NearMv) {
        return false;
    }
    // Never prune before a valid inter mode has been found.
    if !best_mode.is_some_and(PredMode::is_inter) {
        return false;
    }
    if !left_available || !up_available {
        return false;
    }
    let count = MAX_REF_MV_SEARCH.min(row.count);
    if count == 0 {
        return false;
    }
    if this_mode == PredMode::NearestMv && row.weight[0] >= REF_CAT_LEVEL {
        return false;
    }
    let nearest_refmv_count = row.weight[..count]
        .iter()
        .filter(|&&w| w >= REF_CAT_LEVEL)
        .count();
    let prune_thresh = 1 + usize::from(count >= 2);
    nearest_refmv_count < prune_thresh
}

/// `conditional_skipintra` (rdopt.c:941): the four directional intra modes that
/// are only worth evaluating when a related direction already won.
pub fn conditional_skipintra(mode: PredMode, best_intra_mode: PredMode) -> bool {
    use PredMode::*;
    match mode {
        D113Pred => best_intra_mode != VPred && best_intra_mode != D135Pred,
        D67Pred => best_intra_mode != VPred && best_intra_mode != D45Pred,
        D203Pred => best_intra_mode != HPred && best_intra_mode != D45Pred,
        D157Pred => best_intra_mode != HPred && best_intra_mode != D135Pred,
        _ => false,
    }
}

/// `mask_set_bit` (rdopt.c:2347) / `mask_check_bit` (`:2349`): the bitmask of
/// ref-MV indices `ref_mv_idx_to_search` returns.
///
/// C passes a bare `int` and two free functions; naming the type keeps the
/// "which int is a mask" question from arising at the call sites.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IdxMask(pub i32);

impl IdxMask {
    /// An empty mask — no index selected.
    pub const EMPTY: Self = Self(0);

    /// `mask_set_bit`.
    pub fn set(&mut self, index: usize) {
        self.0 |= 1 << index;
    }

    /// `mask_check_bit`.
    pub const fn get(self, index: usize) -> bool {
        (self.0 >> index) & 1 != 0
    }
}
