//! Reference-buffer and GOP management — the port of `av1/encoder/encode_strategy.c`'s
//! decision layer.
//!
//! This is the layer that decides, for every coded frame, **which of the eight
//! reference-buffer slots the frame refreshes** and **which slot each of the
//! seven named reference types (`LAST_FRAME`..`ALTREF_FRAME`) points at**. Both
//! answers are written into the frame header (`refresh_frame_flags`,
//! `ref_frame_idx[]`), so both are bitstream-visible: an inter GOP cannot be
//! byte-exact without them.
//!
//! | Rust | C (`av1/encoder/encode_strategy.c`) |
//! |---|---|
//! | [`RefreshFrameInfo::set`] | `set_refresh_frame_flags` (:47, static) |
//! | [`configure_buffer_updates`] | `av1_configure_buffer_updates` (:55) |
//! | [`additional_frame_flags`] | `set_additional_frame_flags` (:128, static) |
//! | [`apply_ext_overrides`] | `set_ext_overrides` (:141, static) |
//! | [`choose_primary_ref_frame`] | `choose_primary_ref_frame` (:168, static) |
//! | [`update_frame_flags`] | `update_frame_flags` (:427, static) |
//! | [`refresh_ref_frame_map`] | `av1_get_refresh_ref_frame_map` (:515) |
//! | [`free_ref_map_index`] | `get_free_ref_map_index` (:525, static) |
//! | [`refresh_idx`] | `get_refresh_idx` (:531, static) |
//! | [`calc_refresh_idx_for_intnl_arf`] | `av1_calc_refresh_idx_for_intnl_arf` (:594) |
//! | [`new_fb_map_idx_rc`] | `get_new_fb_map_idx_rc` (:614, static) |
//! | [`get_refresh_frame_flags`] | `av1_get_refresh_frame_flags` (:619) |
//! | [`get_ref_frames`] | `av1_get_ref_frames` (:1007) |
//! | (absorbed) | `is_in_ref_map`, `add_ref_to_slot`, `set_unmapped_ref`, `compare_map_idx_pair_asc` |
//!
//! # What is deliberately not here, and why
//! * `av1_encode_strategy`, `denoise_and_encode`, `choose_frame_source`,
//!   `allow_show_existing`, `is_forced_keyframe_pending` — frame-source
//!   selection driven by the `lookahead_ctx` ring buffer, plus the
//!   temporal-filter / TPL pipeline. The port feeds frames through Rust
//!   ownership and has no `struct lookahead_ctx` to reproduce; these are
//!   orchestration, not bitstream semantics.
//! * `adjust_frame_rate` — mutates `cpi->time_stamps` from the caller's
//!   timestamps; its only *arithmetic* output is the framerate that
//!   `av1_rc_update_framerate` consumes, which is ported separately.
//! * `dump_one_image` / `dump_ref_frame_images` — behind
//!   `#define DUMP_REF_FRAME_IMAGES 0`, so they are not even compiled.
//! * `compare_map_idx_pair_asc` — a `qsort` comparator; expressed here as the
//!   sort key. See [`get_ref_frames`] for why the non-stable `qsort` is not a
//!   divergence risk.
//! * The `use_ducky_encode` arms of `av1_get_refresh_frame_flags` and
//!   `av1_get_ref_frames` — reachable only through the `DuckyEncode` test API,
//!   which the port does not expose.
//! * The `rtc_ref.set_ref_frame_config` / `use_rtc_reference_structure_one_layer`
//!   arm of `av1_get_refresh_frame_flags` — SVC / real-time reference
//!   structures, outside the `--end-usage=q` envelope (INTER-ENCODE-GAPMAP §2.1).
//!
//! # Differential coverage and evidence tier, per function
//! **Tier 1** — `crates/aom-encode/tests/ref_gop_diff.rs` drives the REAL
//! exported C symbol through `crates/aom-sys-ref/shim/refgop_shim.c`:
//! [`configure_buffer_updates`] (`av1_configure_buffer_updates`, which also
//! covers the static `set_refresh_frame_flags`), [`refresh_ref_frame_map`],
//! [`calc_refresh_idx_for_intnl_arf`] (covering the statics
//! `get_free_ref_map_index` and `get_refresh_idx`, i.e. [`free_ref_map_index`]
//! and [`refresh_idx`]), [`get_refresh_frame_flags`] (both the default and the
//! external-override arms, plus `get_new_fb_map_idx_rc`), and
//! [`get_ref_frames`] with [`get_ref_frames_from_ext_map`] (covering the
//! statics `is_in_ref_map`, `add_ref_to_slot`, `set_unmapped_ref` and the
//! `compare_map_idx_pair_asc` ordering).
//!
//! **Tier 4** — hand-derived from the C source, with unit tests in this file
//! only, because the C function is `static` with no exported symbol and its
//! only caller is `av1_encode_strategy` (driving that means driving the whole
//! encoder): [`additional_frame_flags`] (`set_additional_frame_flags`),
//! [`update_frame_flags`], [`apply_ext_overrides`] (`set_ext_overrides`) and
//! [`choose_primary_ref_frame`]. These four are the ones to re-verify first if
//! a frame-header byte ever disagrees.
//!
//! # C behaviour reproduced deliberately
//! `av1_get_refresh_frame_flags` ends in `return 1 << refresh_idx` where
//! `get_refresh_idx` can return `-1`; C's own `assert(0 && "No valid refresh
//! index found")` says that state should not arise, but with `NDEBUG` the shift
//! is UB and this ISA yields `INT_MIN`. [`get_refresh_frame_flags`] returns
//! `None` there. The differential asserts exactly that correspondence rather
//! than skipping the cell.

/// `REF_FRAMES` (av1/common/enums.h:535): the number of reference-buffer slots.
pub const REF_FRAMES: usize = 8;

/// `INTER_REFS_PER_FRAME` (enums.h): the number of *named* reference types,
/// `LAST_FRAME..=ALTREF_FRAME`. One fewer than [`REF_FRAMES`] — which is the
/// whole reason `set_unmapped_ref` exists.
pub const INTER_REFS_PER_FRAME: usize = 7;

/// `SELECT_ALL_BUF_SLOTS` (enums.h:573): refresh every slot.
pub const SELECT_ALL_BUF_SLOTS: u32 = 0xFF;

/// `MAX_ARF_LAYERS` (encoder/ratectrl.h:54), the initial `min_level` sentinel in
/// `av1_get_ref_frames`.
const MAX_ARF_LAYERS: i32 = 6;

/// `LOW_LEVEL_FRAMES_TR` (encode_strategy.c:982): above this many lowest-level
/// references, `set_unmapped_ref` may leave a lowest-level frame unmapped.
const LOW_LEVEL_FRAMES_TR: i32 = 5;

/// `FRAME_TYPE` (av1/common/enums.h) in full. Discriminants match C.
///
/// Distinct from [`crate::rd::FrameType`], which is the two-valued reduction
/// (`Key` / `NonKey`) that the RD multiplier needs; this one keeps the
/// `INTRA_ONLY` and `S_FRAME` cases apart because both change the refresh mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    /// `KEY_FRAME`.
    Key = 0,
    /// `INTER_FRAME`.
    Inter = 1,
    /// `INTRA_ONLY_FRAME`.
    IntraOnly = 2,
    /// `S_FRAME`.
    Switch = 3,
}

/// `FRAME_UPDATE_TYPE` (encoder/ratectrl.h:84). Discriminants match C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameUpdateType {
    /// `KF_UPDATE`.
    Kf = 0,
    /// `LF_UPDATE` — a leaf (non-reference-pyramid) frame.
    Lf = 1,
    /// `GF_UPDATE`.
    Gf = 2,
    /// `ARF_UPDATE`.
    Arf = 3,
    /// `OVERLAY_UPDATE`.
    Overlay = 4,
    /// `INTNL_OVERLAY_UPDATE`.
    IntnlOverlay = 5,
    /// `INTNL_ARF_UPDATE`.
    IntnlArf = 6,
}

/// `REFBUF_STATE` (encoder/ratectrl.h:95). Discriminants match C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefbufState {
    /// `REFBUF_RESET` — clear the whole reference buffer.
    Reset = 0,
    /// `REFBUF_UPDATE` — refresh selected slots.
    Update = 1,
}

/// `RefreshFrameInfo` (encoder/encoder.h): which *named* buffers this frame
/// refreshes, as the encoder's legacy per-name view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefreshFrameInfo {
    /// `refresh_frame->golden_frame`.
    pub golden_frame: bool,
    /// `refresh_frame->bwd_ref_frame`.
    pub bwd_ref_frame: bool,
    /// `refresh_frame->alt_ref_frame`.
    pub alt_ref_frame: bool,
}

impl RefreshFrameInfo {
    /// `set_refresh_frame_flags` (encode_strategy.c:47).
    pub fn set(&mut self, refresh_gf: bool, refresh_bwdref: bool, refresh_arf: bool) {
        self.golden_frame = refresh_gf;
        self.bwd_ref_frame = refresh_bwdref;
        self.alt_ref_frame = refresh_arf;
    }
}

/// The result of [`configure_buffer_updates`]: C writes the refresh flags
/// through a pointer and `cpi->rc.is_src_frame_alt_ref` as a side effect, so
/// the port returns both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferUpdates {
    /// The named refresh flags (`*refresh_frame`).
    pub refresh: RefreshFrameInfo,
    /// `cpi->rc.is_src_frame_alt_ref` — set for the two overlay update types.
    pub is_src_frame_alt_ref: bool,
}

/// `ExtRefreshFrameFlagsInfo` (encoder/encoder.h) — the externally supplied
/// refresh overrides (`AV1E_SET_..` / `av1_update_reference()`).
///
/// C carries `update_pending` inside the struct; here the whole struct is
/// wrapped in an [`Option`], so `Some` *is* `update_pending`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtRefreshFrameFlags {
    /// `ext_refresh_frame_flags->last_frame`.
    pub last_frame: bool,
    /// `->golden_frame`.
    pub golden_frame: bool,
    /// `->bwd_ref_frame`.
    pub bwd_ref_frame: bool,
    /// `->alt_ref_frame`.
    pub alt_ref_frame: bool,
    /// `->alt2_ref_frame`.
    pub alt2_ref_frame: bool,
}

impl ExtRefreshFrameFlags {
    /// `is_frame_droppable` (encode_strategy.h:99), the `update_pending` arm:
    /// a frame that refreshes nothing at all is droppable.
    ///
    /// The `rtc_ref->set_ref_frame_config` arm is SVC-only and not ported.
    #[must_use]
    pub fn is_frame_droppable(&self) -> bool {
        !(self.alt_ref_frame
            || self.alt2_ref_frame
            || self.bwd_ref_frame
            || self.golden_frame
            || self.last_frame)
    }
}

/// `av1_configure_buffer_updates` (encode_strategy.c:55).
///
/// `ext_refresh` is `Some` exactly when C's
/// `ext_refresh_frame_flags->update_pending && !is_stat_generation_stage(cpi)`
/// holds. C additionally rewrites `gf_group->update_type[gf_frame_index]` in
/// that arm; that write is returned as the second tuple element rather than
/// performed through a pointer, and is `None` when C leaves the entry alone.
///
/// Note the ordering C uses and this reproduces: the external override replaces
/// the update-type-derived flags wholesale, and `force_refresh_all` then
/// overrides *both*.
#[must_use]
pub fn configure_buffer_updates(
    update_type: FrameUpdateType,
    refbuf_state: RefbufState,
    force_refresh_all: bool,
    ext_refresh: Option<ExtRefreshFrameFlags>,
) -> (BufferUpdates, Option<FrameUpdateType>) {
    let mut refresh = RefreshFrameInfo::default();
    let mut is_src_frame_alt_ref = false;

    match update_type {
        FrameUpdateType::Kf => refresh.set(true, true, true),
        FrameUpdateType::Lf => refresh.set(false, false, false),
        FrameUpdateType::Gf => refresh.set(true, false, false),
        FrameUpdateType::Overlay => {
            if refbuf_state == RefbufState::Reset {
                refresh.set(true, true, true);
            } else {
                refresh.set(true, false, false);
            }
            is_src_frame_alt_ref = true;
        }
        FrameUpdateType::Arf => {
            // NOTE: BWDREF does not get updated along with ALTREF_FRAME.
            if refbuf_state == RefbufState::Reset {
                refresh.set(true, true, true);
            } else {
                refresh.set(false, false, true);
            }
        }
        FrameUpdateType::IntnlOverlay => {
            refresh.set(false, false, false);
            is_src_frame_alt_ref = true;
        }
        FrameUpdateType::IntnlArf => refresh.set(false, true, false),
    }

    let mut new_update_type = None;
    if let Some(ext) = ext_refresh {
        refresh.set(ext.golden_frame, ext.bwd_ref_frame, ext.alt_ref_frame);
        // C writes these in sequence, so the LAST matching flag wins.
        if ext.golden_frame {
            new_update_type = Some(FrameUpdateType::Gf);
        }
        if ext.alt_ref_frame {
            new_update_type = Some(FrameUpdateType::Arf);
        }
        if ext.bwd_ref_frame {
            new_update_type = Some(FrameUpdateType::IntnlArf);
        }
    }

    if force_refresh_all {
        refresh.set(true, true, true);
    }

    (
        BufferUpdates {
            refresh,
            is_src_frame_alt_ref,
        },
        new_update_type,
    )
}

/// `FRAMETYPE_FLAGS` (encoder/encoder.h:110) — the caller-visible frame flags.
pub mod frame_flags {
    /// `FRAMEFLAGS_KEY`.
    pub const KEY: u32 = 1 << 0;
    /// `FRAMEFLAGS_GOLDEN`.
    pub const GOLDEN: u32 = 1 << 1;
    /// `FRAMEFLAGS_BWDREF`.
    pub const BWDREF: u32 = 1 << 2;
    /// `FRAMEFLAGS_ALTREF`.
    pub const ALTREF: u32 = 1 << 3;
    /// `FRAMEFLAGS_INTRAONLY`.
    pub const INTRAONLY: u32 = 1 << 4;
    /// `FRAMEFLAGS_SWITCH`.
    pub const SWITCH: u32 = 1 << 5;
    /// `FRAMEFLAGS_ERROR_RESILIENT`.
    pub const ERROR_RESILIENT: u32 = 1 << 6;
}

/// `set_additional_frame_flags` (encode_strategy.c:128): the bits this frame's
/// *type* contributes, to be OR-ed into the caller's flags.
///
/// C takes `frame_flags` by pointer and only ever sets bits; returning the
/// set-mask instead makes that explicit and keeps the function pure.
/// `frame_is_intra_only` is `KEY_FRAME || INTRA_ONLY_FRAME`;
/// `frame_is_sframe` is `S_FRAME` (av1/common/av1_common_int.h).
#[must_use]
pub fn additional_frame_flags(frame_type: FrameType, error_resilient_mode: bool) -> u32 {
    let mut set = 0;
    if matches!(frame_type, FrameType::Key | FrameType::IntraOnly) {
        set |= frame_flags::INTRAONLY;
    }
    if frame_type == FrameType::Switch {
        set |= frame_flags::SWITCH;
    }
    if error_resilient_mode {
        set |= frame_flags::ERROR_RESILIENT;
    }
    set
}

/// `update_frame_flags` (encode_strategy.c:427): rewrite the four
/// refresh-describing bits to match what was actually coded.
///
/// `show_existing` is C's `encode_show_existing_frame(cm)`; on that path every
/// one of the four bits is cleared and nothing else is consulted.
#[must_use]
pub fn update_frame_flags(
    frame_flags: u32,
    refresh: RefreshFrameInfo,
    frame_type: FrameType,
    show_existing: bool,
) -> u32 {
    let mut f = frame_flags;
    if show_existing {
        return f & !(frame_flags::GOLDEN
            | frame_flags::BWDREF
            | frame_flags::ALTREF
            | frame_flags::KEY);
    }
    for (bit, on) in [
        (frame_flags::GOLDEN, refresh.golden_frame),
        (frame_flags::ALTREF, refresh.alt_ref_frame),
        (frame_flags::BWDREF, refresh.bwd_ref_frame),
        (frame_flags::KEY, frame_type == FrameType::Key),
    ] {
        if on {
            f |= bit;
        } else {
            f &= !bit;
        }
    }
    f
}

/// The `S_FRAME` / error-resilience overrides of `set_ext_overrides`
/// (encode_strategy.c:141), as a pure function of the external flags.
///
/// The rest of that C function writes `cm->features.refresh_frame_context` and
/// `cm->features.allow_ref_frame_mvs` straight from the external flags — plain
/// assignment with no arithmetic, so it is left to the caller's frame-header
/// assembly rather than duplicated here.
///
/// Returns `(frame_type, error_resilient_mode)`.
#[must_use]
pub fn apply_ext_overrides(
    frame_type: FrameType,
    use_s_frame: bool,
    use_error_resilient: bool,
) -> (FrameType, bool) {
    let frame_type = if use_s_frame {
        FrameType::Switch
    } else {
        frame_type
    };
    // A keyframe is already error resilient, and error_resilient_mode on a
    // keyframe interferes with show_existing_frame when forward keyframes are
    // enabled. S-frames must be error-resilient for bitstream conformance.
    let error_resilient_mode =
        (use_error_resilient && frame_type != FrameType::Key) || frame_type == FrameType::Switch;
    (frame_type, error_resilient_mode)
}

/// `PRIMARY_REF_NONE` (av1/common/av1_common_int.h:66).
pub const PRIMARY_REF_NONE: i32 = 7;

/// `choose_primary_ref_frame` (encode_strategy.c:168) — the non-SVC,
/// non-ducky, non-large-scale-tile path.
///
/// `ref_frame_map_idx[i]` is `get_ref_frame_map_idx(cm, LAST_FRAME + i)` for the
/// seven named references, i.e. the `remapped_ref_idx` this frame will use.
/// `wanted_fb` is `cpi->ppi->fb_of_context_type[get_current_frame_ref_type(cpi)]`.
///
/// C scans `LAST_FRAME..=ALTREF_FRAME` **without breaking**, so the LAST match
/// wins, not the first. `rposition` expresses that directly.
#[must_use]
pub fn choose_primary_ref_frame(
    frame_type: FrameType,
    error_resilient_mode: bool,
    use_primary_ref_none: bool,
    ref_frame_map_idx: &[i32; INTER_REFS_PER_FRAME],
    wanted_fb: i32,
) -> i32 {
    let intra_only = matches!(frame_type, FrameType::Key | FrameType::IntraOnly);
    if intra_only || error_resilient_mode || use_primary_ref_none {
        return PRIMARY_REF_NONE;
    }
    match ref_frame_map_idx.iter().rposition(|&idx| idx == wanted_fb) {
        Some(i) => i as i32,
        None => PRIMARY_REF_NONE,
    }
}

/// `av1_get_refresh_ref_frame_map` (encode_strategy.c:515): the index of the
/// lowest set bit of `refresh_frame_flags`.
///
/// C returns `INVALID_IDX` (-1) when no bit is set; the port returns `None`.
#[must_use]
pub fn refresh_ref_frame_map(refresh_frame_flags: u32) -> Option<usize> {
    (0..REF_FRAMES).find(|&i| (refresh_frame_flags >> i) & 1 == 1)
}

/// `RefFrameMapPair` (encoder/encoder.h:3970): the display order and pyramid
/// level of whatever currently occupies one reference-buffer slot.
///
/// An empty slot is `disp_order == -1` (C `memset`s the array to `-1`, which
/// gives `pyr_level == -1` too; nothing reads `pyr_level` of an empty slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefFrameMapPair {
    /// `pyr_level` — the frame's level in the reference pyramid, 1 = top.
    pub pyr_level: i32,
    /// `disp_order` — display order, or `-1` for an empty slot.
    pub disp_order: i32,
}

impl RefFrameMapPair {
    /// An empty slot, C's all-`-1` `memset`.
    pub const EMPTY: Self = Self {
        pyr_level: -1,
        disp_order: -1,
    };

    /// Whether this slot holds a frame.
    #[must_use]
    pub fn is_occupied(&self) -> bool {
        self.disp_order != -1
    }
}

impl Default for RefFrameMapPair {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// `get_free_ref_map_index` (encode_strategy.c:525): the first empty slot, or
/// `None` for C's `INVALID_IDX`.
#[must_use]
pub fn free_ref_map_index(pairs: &[RefFrameMapPair; REF_FRAMES]) -> Option<usize> {
    pairs.iter().position(|p| !p.is_occupied())
}

/// `get_refresh_idx` (encode_strategy.c:531): pick the reference-buffer slot to
/// overwrite when no slot is free.
///
/// `skip_frame_refresh` is `gf_group->skip_frame_refresh[gf_index]`, a list of
/// display orders terminated by `INVALID_IDX` (-1); it is consulted only when
/// `enable_refresh_skip`.
///
/// C returns `-1` from the (assert-guarded, `NDEBUG`-erased) fall-through where
/// neither an oldest non-top-level frame nor an oldest top-level frame was
/// found — and its caller would then evaluate `1 << -1`. The port returns
/// `None` there instead of reproducing that UB.
#[must_use]
pub fn refresh_idx(
    pairs: &[RefFrameMapPair; REF_FRAMES],
    update_arf: bool,
    skip_frame_refresh: &[i32; REF_FRAMES],
    enable_refresh_skip: bool,
    cur_frame_disp: i32,
) -> Option<usize> {
    let mut arf_count = 0i32;
    let mut oldest_arf: Option<(i32, usize)> = None;
    let mut oldest: Option<(i32, usize)> = None;

    for (map_idx, ref_pair) in pairs.iter().enumerate() {
        if !ref_pair.is_occupied() {
            continue;
        }
        let frame_order = ref_pair.disp_order;
        // Keep future frames and the three closest previous frames in output order.
        if frame_order > cur_frame_disp - 3 {
            continue;
        }
        if enable_refresh_skip
            && skip_frame_refresh
                .iter()
                .take_while(|&&s| s != -1)
                .any(|&s| s == frame_order)
        {
            continue;
        }

        if ref_pair.pyr_level == 1 {
            // Track the oldest level-1 frame; if more than two are in the
            // reference list, the oldest is the one to discard.
            if oldest_arf.is_none_or(|(order, _)| frame_order < order) {
                oldest_arf = Some((frame_order, map_idx));
            }
            arf_count += 1;
            continue;
        }

        if oldest.is_none_or(|(order, _)| frame_order < order) {
            oldest = Some((frame_order, map_idx));
        }
    }

    if update_arf && arf_count > 2 {
        return oldest_arf.map(|(_, idx)| idx);
    }
    oldest.or(oldest_arf).map(|(_, idx)| idx)
}

/// `av1_calc_refresh_idx_for_intnl_arf` (encode_strategy.c:594): the slot an
/// `INTNL_ARF_UPDATE` frame writes into — a free slot if there is one,
/// otherwise [`refresh_idx`] with `update_arf = false`.
#[must_use]
pub fn calc_refresh_idx_for_intnl_arf(
    pairs: &[RefFrameMapPair; REF_FRAMES],
    skip_frame_refresh: &[i32; REF_FRAMES],
    enable_refresh_skip: bool,
    cur_frame_disp: i32,
) -> Option<usize> {
    free_ref_map_index(pairs).or_else(|| {
        refresh_idx(
            pairs,
            false,
            skip_frame_refresh,
            enable_refresh_skip,
            cur_frame_disp,
        )
    })
}

/// `get_new_fb_map_idx_rc` (encode_strategy.c:614): the external-rate-control
/// refresh mask — a single bit, or 0 when the slot is `INVALID_IDX`.
#[must_use]
pub fn new_fb_map_idx_rc(new_fb_map_idx: Option<usize>) -> u32 {
    match new_fb_map_idx {
        Some(idx) => 1 << idx,
        None => 0,
    }
}

/// `av1_get_refresh_frame_flags` (encode_strategy.c:619): the eight-bit
/// `refresh_frame_flags` mask this frame writes into its header.
///
/// The external-override arm needs the frame's current named→slot mapping;
/// `ext` carries it as `(flags, ref_frame_map_idx)` where `ref_frame_map_idx[i]`
/// is `get_ref_frame_map_idx(cm, LAST_FRAME + i)` and `-1` is `INVALID_IDX`.
/// Slot 7 of that array is `EXTREF_FRAME`, which the override arm reads.
///
/// Returns `None` only where C would evaluate `1 << -1` (see [`refresh_idx`]).
// The nine parameters are exactly what C reads out of `cpi`, `frame_params`
// and `gf_group` at this call; collapsing them into a context struct would move
// the same fields behind a name without making any of them optional.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn get_refresh_frame_flags(
    pairs: &[RefFrameMapPair; REF_FRAMES],
    refbuf_state: RefbufState,
    frame_type: FrameType,
    show_existing_frame: bool,
    update_type: FrameUpdateType,
    skip_frame_refresh: &[i32; REF_FRAMES],
    enable_refresh_skip: bool,
    cur_disp_order: i32,
    ext: Option<(ExtRefreshFrameFlags, &[i32; REF_FRAMES])>,
) -> Option<u32> {
    if refbuf_state == RefbufState::Reset {
        return Some(SELECT_ALL_BUF_SLOTS);
    }
    // Switch frames and shown key-frames overwrite all reference slots.
    if frame_type == FrameType::Switch {
        return Some(SELECT_ALL_BUF_SLOTS);
    }
    // show_existing_frames don't send refresh_frame_flags at all.
    if show_existing_frame {
        return Some(0);
    }
    if ext.is_some_and(|(flags, _)| flags.is_frame_droppable()) {
        return Some(0);
    }

    if let Some((flags, map_idx)) = ext {
        // Replicate the legacy per-name refresh flags: each external flag sets
        // the bit of the slot its named reference currently points at.
        let mut mask = 0u32;
        let mut set = |named: usize, on: bool| {
            let idx = map_idx[named];
            if idx != -1 {
                mask |= u32::from(on) << idx;
            }
        };
        set(LAST_SLOT, flags.last_frame);
        set(EXTREF_SLOT, flags.bwd_ref_frame);
        set(ALTREF2_SLOT, flags.alt2_ref_frame);
        if update_type == FrameUpdateType::Overlay {
            // On an overlay the golden flag lands on the ALTREF slot.
            set(ALTREF_SLOT, flags.golden_frame);
        } else {
            set(GOLDEN_SLOT, flags.golden_frame);
            set(ALTREF_SLOT, flags.alt_ref_frame);
        }
        return Some(mask);
    }

    // No refresh necessary for these frame types.
    if matches!(
        update_type,
        FrameUpdateType::Overlay | FrameUpdateType::IntnlOverlay
    ) {
        return Some(0);
    }
    // If there is an open slot, refresh that instead of replacing a reference.
    if let Some(free) = free_ref_map_index(pairs) {
        return Some(1 << free);
    }
    refresh_idx(
        pairs,
        update_type == FrameUpdateType::Arf,
        skip_frame_refresh,
        enable_refresh_skip,
        cur_disp_order,
    )
    .map(|idx| 1 << idx)
}

/// `remapped_ref_idx` slot for `LAST_FRAME`.
pub const LAST_SLOT: usize = 0;
/// `remapped_ref_idx` slot for `LAST2_FRAME`.
pub const LAST2_SLOT: usize = 1;
/// `remapped_ref_idx` slot for `LAST3_FRAME`.
pub const LAST3_SLOT: usize = 2;
/// `remapped_ref_idx` slot for `GOLDEN_FRAME`.
pub const GOLDEN_SLOT: usize = 3;
/// `remapped_ref_idx` slot for `BWDREF_FRAME`.
pub const BWDREF_SLOT: usize = 4;
/// `remapped_ref_idx` slot for `ALTREF2_FRAME`.
pub const ALTREF2_SLOT: usize = 5;
/// `remapped_ref_idx` slot for `ALTREF_FRAME`.
pub const ALTREF_SLOT: usize = 6;
/// `remapped_ref_idx` slot for `EXTREF_FRAME` — never assigned a buffer by
/// [`get_ref_frames`], only zero-filled by its trailing loop.
pub const EXTREF_SLOT: usize = 7;

/// Which reference buffer the frame-parallel exclusion in [`get_ref_frames`]
/// must not use.
///
/// C reaches this only when `frame_parallel_level[gf_index] == 2`, the previous
/// gf entry is a level-1 `INTNL_ARF_UPDATE`, and the encode is not one-pass RT;
/// `None` covers every other case. Which of the two discriminants applies is
/// C's `is_parallel_encode` ternary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelSkip {
    /// `is_parallel_encode`: skip the buffer whose map index is
    /// `cpi->ref_idx_to_skip`.
    MapIdx(i32),
    /// otherwise: skip the buffer whose display order is
    /// `gf_group->skip_frame_as_ref[gf_index]`.
    DispOrder(i32),
}

/// One entry of C's `RefBufMapData` (encode_strategy.c:944).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RefBufMapData {
    map_idx: usize,
    disp_order: i32,
    pyr_level: i32,
    used: bool,
}

/// `set_unmapped_ref` (encode_strategy.c:985): with eight buffers and only
/// seven named slots, decide which buffer to leave unmapped — the farthest in
/// display order from the current frame, preferring not to drop a lowest-level
/// reference while fewer than [`LOW_LEVEL_FRAMES_TR`] of them exist.
///
/// Returns the index to mark used, or `None`. `None` is where C asserts
/// (`unmapped_idx >= 0`) and, with `NDEBUG`, would write `buffer_map[-1]`; the
/// port declines rather than reproducing that. C also returns early when
/// `n_bufs <= ALTREF_FRAME` (7) — i.e. when every buffer fits in a named slot.
fn unmapped_ref(
    buffer_map: &[RefBufMapData],
    n_min_level_refs: i32,
    min_level: i32,
    cur_frame_disp: i32,
) -> Option<usize> {
    if buffer_map.len() <= INTER_REFS_PER_FRAME {
        return None;
    }
    let mut max_dist = 0;
    let mut unmapped = None;
    for (i, buf) in buffer_map.iter().enumerate() {
        if buf.used {
            continue;
        }
        if buf.pyr_level != min_level || n_min_level_refs >= LOW_LEVEL_FRAMES_TR {
            let dist = (cur_frame_disp - buf.disp_order).abs();
            if dist > max_dist {
                max_dist = dist;
                unmapped = Some(i);
            }
        }
    }
    unmapped
}

/// `av1_get_ref_frames` (encode_strategy.c:1007): map the eight reference
/// buffers onto the seven named reference types plus `EXTREF`.
///
/// Returns `remapped_ref_idx`: entry `i` is the buffer index that reference
/// type `LAST_FRAME + i` points at (see the [`LAST_SLOT`]..[`EXTREF_SLOT`]
/// constants). Every entry is a valid buffer index — C's trailing loop fills
/// anything still `INVALID_IDX` with 0.
///
/// # On C's `qsort`
/// C sorts `buffer_map` by display order with a comparator that returns 0 for
/// equal keys, and `qsort` is not stable — but the build step above it skips
/// any buffer whose display order is already present (`is_in_ref_map`), so no
/// two entries can compare equal and the ordering is total. `sort_by_key` is
/// therefore not merely equivalent-in-practice; it is the same permutation.
#[must_use]
pub fn get_ref_frames(
    pairs: &[RefFrameMapPair; REF_FRAMES],
    cur_frame_disp: i32,
    parallel_skip: Option<ParallelSkip>,
) -> [i32; REF_FRAMES] {
    let mut remapped: [i32; REF_FRAMES] = [-1; REF_FRAMES];

    // Collect the occupied buffers, deduplicated on display order
    // (`is_in_ref_map`), tracking the pyramid-level extremes as we go.
    let mut buffer_map: Vec<RefBufMapData> = Vec::with_capacity(REF_FRAMES);
    let mut min_level = MAX_ARF_LAYERS;
    let mut max_level = 0;
    for (map_idx, ref_pair) in pairs.iter().enumerate() {
        if !ref_pair.is_occupied() {
            continue;
        }
        let frame_order = ref_pair.disp_order;
        if buffer_map.iter().any(|b| b.disp_order == frame_order) {
            continue;
        }
        min_level = min_level.min(ref_pair.pyr_level);
        max_level = max_level.max(ref_pair.pyr_level);
        buffer_map.push(RefBufMapData {
            map_idx,
            disp_order: frame_order,
            pyr_level: ref_pair.pyr_level,
            used: false,
        });
    }
    let _ = max_level; // C tracks it here and never reads it.

    buffer_map.sort_by_key(|b| b.disp_order);
    let n_bufs = buffer_map.len();

    let mut n_min_level_refs = 0i32;
    let mut closest_past_ref: Option<usize> = None;
    let mut golden_idx: Option<usize> = None;
    let mut altref_idx: Option<usize> = None;
    let mut skip_ref_unmapping = false;

    // Walk newest-first: find GOLDEN and ALTREF, apply the parallel-encode
    // exclusion, and locate the past/future boundary.
    for i in (0..n_bufs).rev() {
        if buffer_map[i].pyr_level == min_level {
            n_min_level_refs += 1;
            if buffer_map[i].disp_order < cur_frame_disp
                && golden_idx.is_none()
                && remapped[GOLDEN_SLOT] == -1
            {
                golden_idx = Some(i);
            } else if buffer_map[i].disp_order > cur_frame_disp
                && altref_idx.is_none()
                && remapped[ALTREF_SLOT] == -1
            {
                altref_idx = Some(i);
            }
        } else if buffer_map[i].disp_order == cur_frame_disp {
            // This is the show_existing_frame; map it to BWDREF.
            remapped[BWDREF_SLOT] = buffer_map[i].map_idx as i32;
            buffer_map[i].used = true;
        }

        if let Some(skip) = parallel_skip {
            buffer_map[i].used = match skip {
                ParallelSkip::MapIdx(idx) => buffer_map[i].map_idx as i32 == idx,
                ParallelSkip::DispOrder(order) => buffer_map[i].disp_order == order,
            };
            // A buffer excluded here must stay unmapped, so do not additionally
            // drop one in set_unmapped_ref.
            if buffer_map[i].used {
                skip_ref_unmapping = true;
            }
        }

        if buffer_map[i].disp_order < cur_frame_disp && closest_past_ref.is_none() {
            closest_past_ref = Some(i);
        }
    }

    // Only map GOLDEN/ALTREF by pyramid level if the levels actually differ.
    if n_min_level_refs < n_bufs as i32 {
        if let Some(i) = golden_idx {
            remapped[GOLDEN_SLOT] = buffer_map[i].map_idx as i32;
            buffer_map[i].used = true;
        }
        if let Some(i) = altref_idx {
            remapped[ALTREF_SLOT] = buffer_map[i].map_idx as i32;
            buffer_map[i].used = true;
        }
    }

    if !skip_ref_unmapping
        && let Some(i) = unmapped_ref(&buffer_map, n_min_level_refs, min_level, cur_frame_disp)
    {
        buffer_map[i].used = true;
    }

    // LAST/LAST2/LAST3: the nearest unused past frames, newest first.
    for slot in [LAST_SLOT, LAST2_SLOT, LAST3_SLOT] {
        if remapped[slot] != -1 {
            continue;
        }
        let pick = (0..n_bufs)
            .rev()
            .filter(|&i| !buffer_map[i].used && buffer_map[i].disp_order < cur_frame_disp)
            .max_by_key(|&i| buffer_map[i].disp_order);
        let Some(i) = pick else { break };
        remapped[slot] = buffer_map[i].map_idx as i32;
        buffer_map[i].used = true;
    }

    // BWDREF/ALTREF2/ALTREF: the nearest unused future frames, oldest first.
    // C's `frame < REF_FRAMES` bound stops at ALTREF_FRAME; EXTREF is never
    // assigned a buffer here, only zero-filled by the trailing loop.
    for slot in [BWDREF_SLOT, ALTREF2_SLOT, ALTREF_SLOT] {
        if remapped[slot] != -1 {
            continue;
        }
        let pick = (0..n_bufs)
            .rev()
            .filter(|&i| !buffer_map[i].used && buffer_map[i].disp_order > cur_frame_disp)
            .min_by_key(|&i| buffer_map[i].disp_order);
        let Some(i) = pick else { break };
        remapped[slot] = buffer_map[i].map_idx as i32;
        buffer_map[i].used = true;
    }

    // Remaining past frames: walk down from the past/future boundary.
    // C starts `buf_map_idx = closest_past_ref`, which is -1 (i.e. "stop
    // immediately") when every buffer is a future frame.
    let mut cursor: i64 = closest_past_ref.map_or(-1, |i| i as i64);
    for entry in &mut remapped[LAST_SLOT..=ALTREF_SLOT] {
        if *entry != -1 {
            continue;
        }
        while cursor >= 0 && buffer_map[cursor as usize].used {
            cursor -= 1;
        }
        if cursor < 0 {
            break;
        }
        let i = cursor as usize;
        *entry = buffer_map[i].map_idx as i32;
        buffer_map[i].used = true;
    }

    // Remaining future frames: walk down from the newest, stopping at the
    // boundary. Named slots are filled ALTREF first, back to LAST.
    // C's loop bound is `buf_map_idx > closest_past_ref`, and closest_past_ref
    // is -1 when there is no past frame, so the walk then reaches index 0.
    let past_boundary: i64 = closest_past_ref.map_or(-1, |i| i as i64);
    let mut cursor: i64 = n_bufs as i64 - 1;
    for slot in (LAST_SLOT..=ALTREF_SLOT).rev() {
        if remapped[slot] != -1 {
            continue;
        }
        while cursor > past_boundary && buffer_map[cursor as usize].used {
            cursor -= 1;
        }
        if cursor < 0 {
            break;
        }
        let i = cursor as usize;
        if buffer_map[i].used {
            break;
        }
        remapped[slot] = buffer_map[i].map_idx as i32;
        buffer_map[i].used = true;
    }

    // Anything still unmapped points at buffer 0 (only happens for the first
    // seven frames of a sequence).
    for entry in &mut remapped {
        if *entry == -1 {
            *entry = 0;
        }
    }
    remapped
}

/// `av1_get_ref_frames`'s `use_ext_ref_frame_map` early return
/// (encode_strategy.c:1017): take the named mapping straight from the external
/// rate controller's GOP decision, then zero-fill anything it left unset.
///
/// `ref_frame_list[i]` is C's `gf_group->ref_frame_list[gf_index][LAST_FRAME + i]`,
/// with `-1` for `INVALID_IDX`.
#[must_use]
pub fn get_ref_frames_from_ext_map(
    ref_frame_list: &[i32; INTER_REFS_PER_FRAME],
) -> [i32; REF_FRAMES] {
    let mut remapped = [0i32; REF_FRAMES];
    for (slot, &idx) in ref_frame_list.iter().enumerate() {
        if idx != -1 {
            remapped[slot] = idx;
        }
    }
    remapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_ref_frame_map_is_the_lowest_set_bit() {
        assert_eq!(refresh_ref_frame_map(0), None);
        assert_eq!(refresh_ref_frame_map(1), Some(0));
        assert_eq!(refresh_ref_frame_map(0b1010_0000), Some(5));
        assert_eq!(refresh_ref_frame_map(SELECT_ALL_BUF_SLOTS), Some(0));
    }

    #[test]
    fn configure_buffer_updates_matches_the_c_switch() {
        let none = RefbufState::Update;
        let cases = [
            (FrameUpdateType::Kf, (true, true, true), false),
            (FrameUpdateType::Lf, (false, false, false), false),
            (FrameUpdateType::Gf, (true, false, false), false),
            (FrameUpdateType::Arf, (false, false, true), false),
            (FrameUpdateType::Overlay, (true, false, false), true),
            (FrameUpdateType::IntnlOverlay, (false, false, false), true),
            (FrameUpdateType::IntnlArf, (false, true, false), false),
        ];
        for (ty, (g, b, a), alt_ref) in cases {
            let (out, retype) = configure_buffer_updates(ty, none, false, None);
            assert_eq!(
                (
                    out.refresh.golden_frame,
                    out.refresh.bwd_ref_frame,
                    out.refresh.alt_ref_frame
                ),
                (g, b, a),
                "{ty:?}"
            );
            assert_eq!(out.is_src_frame_alt_ref, alt_ref, "{ty:?}");
            assert_eq!(retype, None);
        }
        // REFBUF_RESET turns both OVERLAY and ARF into refresh-everything.
        for ty in [FrameUpdateType::Overlay, FrameUpdateType::Arf] {
            let (out, _) = configure_buffer_updates(ty, RefbufState::Reset, false, None);
            assert_eq!(
                out.refresh,
                RefreshFrameInfo {
                    golden_frame: true,
                    bwd_ref_frame: true,
                    alt_ref_frame: true
                }
            );
        }
        // force_refresh_all wins over the external override.
        let ext = ExtRefreshFrameFlags::default();
        let (out, _) = configure_buffer_updates(FrameUpdateType::Lf, none, true, Some(ext));
        assert!(out.refresh.golden_frame && out.refresh.bwd_ref_frame && out.refresh.alt_ref_frame);
    }

    #[test]
    fn ext_override_update_type_takes_the_last_matching_flag() {
        let ext = ExtRefreshFrameFlags {
            golden_frame: true,
            alt_ref_frame: true,
            bwd_ref_frame: true,
            ..Default::default()
        };
        let (_, retype) =
            configure_buffer_updates(FrameUpdateType::Lf, RefbufState::Update, false, Some(ext));
        assert_eq!(retype, Some(FrameUpdateType::IntnlArf));
    }

    #[test]
    fn update_frame_flags_clears_all_four_on_show_existing() {
        let all = frame_flags::GOLDEN
            | frame_flags::BWDREF
            | frame_flags::ALTREF
            | frame_flags::KEY
            | frame_flags::SWITCH;
        let out = update_frame_flags(all, RefreshFrameInfo::default(), FrameType::Key, true);
        assert_eq!(out, frame_flags::SWITCH);
    }

    #[test]
    fn choose_primary_ref_frame_takes_the_last_match_not_the_first() {
        // Two named references point at buffer 3; C's non-breaking loop keeps
        // the LAST one (ALTREF, slot 6).
        let map = [3, 1, 2, 3, 4, 5, 3];
        assert_eq!(
            choose_primary_ref_frame(FrameType::Inter, false, false, &map, 3),
            6
        );
        assert_eq!(
            choose_primary_ref_frame(FrameType::Inter, false, false, &map, 9),
            PRIMARY_REF_NONE
        );
        assert_eq!(
            choose_primary_ref_frame(FrameType::Key, false, false, &map, 3),
            PRIMARY_REF_NONE
        );
        assert_eq!(
            choose_primary_ref_frame(FrameType::Inter, true, false, &map, 3),
            PRIMARY_REF_NONE
        );
    }

    #[test]
    fn free_slot_beats_replacement_in_the_refresh_mask() {
        let mut pairs = [RefFrameMapPair::EMPTY; REF_FRAMES];
        for (i, p) in pairs.iter_mut().enumerate().take(5) {
            *p = RefFrameMapPair {
                pyr_level: 2,
                disp_order: i as i32,
            };
        }
        let skip = [-1; REF_FRAMES];
        let mask = get_refresh_frame_flags(
            &pairs,
            RefbufState::Update,
            FrameType::Inter,
            false,
            FrameUpdateType::Lf,
            &skip,
            true,
            10,
            None,
        );
        assert_eq!(mask, Some(1 << 5));
    }

    #[test]
    fn every_named_slot_is_filled_with_a_real_buffer() {
        // Eight occupied buffers, seven named slots: one must go unmapped, and
        // every returned entry must still be a valid buffer index.
        let mut pairs = [RefFrameMapPair::EMPTY; REF_FRAMES];
        for (i, p) in pairs.iter_mut().enumerate() {
            *p = RefFrameMapPair {
                pyr_level: if i == 0 { 1 } else { 2 },
                disp_order: i as i32,
            };
        }
        let remapped = get_ref_frames(&pairs, 20, None);
        assert!(remapped.iter().all(|&i| (0..8).contains(&i)));
    }
}
