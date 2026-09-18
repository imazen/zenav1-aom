//! Loop-filter-level RD search — port of `av1/encoder/picklpf.c`'s
//! `av1_pick_filter_level`, for the envelope this port targets: a single
//! shown KEY frame (`frame_is_intra_only`), ALLINTRA (usage=2) or GOOD
//! (usage=0) one-pass encode, `method == LPF_PICK_FROM_FULL_IMAGE` (the
//! speed-0 default for BOTH usages — `av1/encoder/speed_features.c`'s
//! `init_lpf_sf` sets this; `set_allintra_speed_feature_framesize_independent`
//! only overrides `lpf_pick` at `speed >= 4` (`NON_DUAL`) / `speed >= 6`
//! (`FROM_Q`), never at speed 0; `set_good_speed_features_framesize_independent`
//! likewise only touches it at `speed >= 5`/`speed >= 7` — verified by
//! grepping every `lpf_pick =` assignment site in speed_features.c). The
//! `LPF_PICK_FROM_FULL_IMAGE_NON_DUAL` variant (allintra speed 4/5) is handled
//! via the `non_dual` flag on [`pick_filter_level`] (it skips the two Y
//! single-direction refine passes — picklpf.c:376); `FROM_Q` (speed>=6) is out
//! of the current cpu-used 0..=5 gate scope.
//!
//! # Envelope simplifications (each individually verified against the C, not
//! assumed)
//!
//! - **`last_frame_filter_level` is always `[0,0,0,0]`.** `av1_pick_filter_level`
//!   only populates it `if !frame_is_intra_only(cm)` — always false for our
//!   single-KEY-frame envelope — so every [`search_filter_level`] search
//!   starts at `filt_mid = 0`, hence `filter_step = 4` (`0 < 16`) always.
//! - **`max_filter_level` is always [`MAX_LOOP_FILTER`] (63).**
//!   `get_max_filter_level` only shrinks it `if
//!   is_stat_consumption_stage_twopass(cpi)` — never true for a one-pass
//!   `--cq-level` encode.
//! - **`use_coarse_filter_level_search` is always `0`** (`init_lpf_sf`
//!   default; neither the allintra nor good speed-feature setter touches it
//!   below `speed >= 5`), so `min_filter_step_thesh = 0` — the search always
//!   runs to `filter_step == 0`, never stopping early at `2`.
//! - **`skip_loop_filter_using_filt_error` / `adaptive_luma_loop_filter_skip`
//!   are always `0`** (dead) at speed 0, both usages — grepped every
//!   assignment site in speed_features.c; the only non-zero writes are in
//!   `set_good_speed_features_lc_dec_framesize_{dependent,independent}`
//!   (a "low complexity decode" variant this envelope never reaches) or
//!   `speed >= 5`/`speed >= 6` blocks. The corresponding C branches
//!   (`av1_pick_filter_level`'s zero/best-filter-SSE reset logic) are
//!   therefore omitted here, not reachable in this envelope.
//! - **`loopfilter_control == LOOPFILTER_ALL`** (default; `disable_filter_rt_screen`
//!   is always 0 — no cyclic-refresh screen-content AQ in this envelope), so
//!   the early `[0,0]` return never fires.
//! - **`cm->features.tx_mode != ONLY_4X4` always holds where this result is
//!   used.** The frame header only calls `encode_loopfilter` (hence only
//!   needs this derivation) `if !coded_lossless`
//!   (`write_frame_header_obu`); `ONLY_4X4` only arises together with
//!   lossless / very old profiles, so the bias-halving branch in
//!   [`search_filter_level`] is applied unconditionally.
//! - **`mode_ref_delta_enabled` is `true` with the textbook default deltas**
//!   `ref_deltas = [1,0,0,0,-1,0,-1,-1]` / `mode_deltas = [0,0]`
//!   (`av1_setup_past_independence` → `set_default_lf_deltas`, called for
//!   every KEY frame) — VERIFIED against the real parsed header (both a
//!   flat and a textured case show `mode_ref_delta_enabled: true` with
//!   exactly these deltas). Since every block in this envelope is intra
//!   (`ref0 == INTRA_FRAME == 0`, `mode_lf == 0` — `MODE_LF_LUT` maps every
//!   intra mode to 0), the ONLY `(ref, mode)` cell of the per-level table
//!   ever read is `[0][0]`, which resolves to `clamp(base_level +
//!   ref_deltas[0] * scale, 0, 63)` — a uniform `+1` (or `+2` once
//!   `base_level >= 32`, since `scale = 1 << (base_level >> 5)`) added to
//!   EVERY block's level, not a per-block-varying adjustment. Getting this
//!   wrong (e.g. assuming `mode_ref_delta_enabled == false`, a reasonable
//!   but WRONG first guess) would silently pick a different filter level
//!   than real aomenc.
//! - **`sharpness_level`**: ALLINTRA → `cpi->oxcf.algo_cfg.sharpness` (this
//!   port's callers never set `--sharpness`, so 0 in every case actually
//!   exercised); GOOD/other → 0 unconditionally (only nonzero for
//!   `AOM_TUNE_IQ`/`AOM_TUNE_SSIMULACRA2`, and this port's tune is always
//!   `TuneMetric::Psnr` — `rd.rs`).
//!
//! Trial deblocking reuses the ALREADY bit-exact
//! [`aom_dsp::loopfilter::frame::loop_filter_frame`] verbatim (never
//! reimplemented) — a fresh clone of the plane under test is filtered at
//! each candidate level and compared against the source via [`sse_plane`].

use crate::encode_sb::{LeafWinner, SbTree};
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B};
use aom_dsp::loopfilter::frame::{
    LfFrameBuf, LfFrameBufU8, LfMi, LfMiGrid, LfParams, MAX_LOOP_FILTER,
    loop_filter_frame_opt, loop_filter_frame_u8_opt,
};

/// `av1_set_default_ref_deltas`/`av1_set_default_mode_deltas`
/// (`av1/common/entropymode.c`) — the KEY-frame reset values, applied via
/// `av1_setup_past_independence` -> `set_default_lf_deltas` (which ALSO sets
/// `mode_ref_delta_enabled = 1`, verified against the real parsed header —
/// see module docs).
const DEFAULT_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const DEFAULT_MODE_DELTAS: [i8; 2] = [0, 0];

/// `aom_get_{y,u,v}_sse` / `highbd_get_sse` (`aom_dsp/psnr.c`): sum of
/// squared per-pixel differences over `w x h`, u16 samples at every bit
/// depth (bd==8 values fit in 0..255 and are numerically identical whether
/// the C reads them as u8 or u16 — the same insight
/// `aom-loopfilter/src/frame.rs`'s own module docs already rely on). The C
/// tiles this sum in 16x16 blocks via `aom_sse`/`aom_highbd_sse` (RTCD/SIMD
/// dispatch) for performance; summing non-negative i64 terms is associative
/// with no overflow risk at any realistic frame size (even 8192x8192 x
/// 65535^2 ~= 2.9e14, far under `i64::MAX`), so this straight row-major sum
/// reproduces the bit-identical i64 total regardless of tiling order — no
/// need to replicate the tiling itself.
pub fn sse_plane(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    let mut total: i64 = 0;
    for row in 0..h {
        let ar = &a[row * a_stride..row * a_stride + w];
        let br = &b[row * b_stride..row * b_stride + w];
        for (av, bv) in ar.iter().zip(br) {
            let diff = i64::from(*av) - i64::from(*bv);
            total += diff * diff;
        }
    }
    total
}

/// The pixel type a [`pick_filter_level`] trial runs over. `u16` is the
/// general path (all bit depths); `u8` is the bd8 lowbd path, where the
/// per-trial u16-copy + narrow + u8-filter + widen round trip inside
/// [`loop_filter_frame_opt`] collapses to a u8 copy + filter — same trials,
/// same SSE values (bd8 samples are `< 256` on both representations, and
/// `diff * diff` is representation-blind), so the picked levels are
/// bit-identical.
pub trait LfTrialPixel: Copy + Send + Sync + Sized {
    /// One `try_filter_frame` trial (`picklpf.c`): copy `recon_*[plane]`
    /// into `scratch`, run the loop-filter walk at `p`, return SSE vs
    /// `src_*[plane]` over the crop dims. The `LfParams`/`LfMiGrid` setup
    /// is pixel-independent — [`LfSearchFrame::try_filter_plane`] builds it
    /// once per trial and delegates only the typed copy/filter/SSE here.
    fn filter_trial(
        f: &LfSearchFrame<'_, Self>,
        plane: usize,
        p: &LfParams,
        grid: &LfMiGrid<'_>,
        scratch: &mut Vec<Self>,
    ) -> i64;
}

/// The frame state one [`pick_filter_level`] call searches over: this
/// port's OWN reconstruction (pre-deblock) + the original source, both
/// planes sharing one `stride` (matching [`crate::encode_sb::SbEncodeEnv`]'s
/// own contract), plus the mi grid the filter reads. `P` is `u16` for the
/// general path and `u8` for the bd8 lowbd staging ([`LfTrialPixel`]).
#[derive(Clone, Copy)]
pub struct LfSearchFrame<'a, P: LfTrialPixel = u16> {
    pub recon_y: &'a [P],
    pub recon_u: &'a [P],
    pub recon_v: &'a [P],
    pub src_y: &'a [P],
    pub src_u: &'a [P],
    pub src_v: &'a [P],
    pub stride: usize,
    /// Luma CROP dims (the coded frame size — SSE is measured over crop
    /// dims, matching `aom_get_y_sse`'s use of `y_crop_width/height`, NOT
    /// the mi-aligned dims).
    pub crop_width: u32,
    pub crop_height: u32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: i32,
    pub monochrome: bool,
    /// The mi grid (luma mi units; `aom_dsp::loopfilter::frame` derives chroma
    /// positions from `ss_x`/`ss_y` internally — one grid serves all 3
    /// planes).
    pub mi: &'a [LfMi],
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// `delta_q_info.delta_lf_present_flag` (`--delta-lf-mode=1`): when true the
    /// trial deblock reads each mi's `delta_lf_from_base` via `get_filter_level`
    /// (av1_loopfilter.c:73-88), so the derived frame filter_level depends on
    /// the per-SB delta-lf. `false` (default) = the uniform-level envelope. The
    /// per-SB `delta_lf_from_base` must be stamped into `mi` (see
    /// [`stamp_lf_delta_lf`]).
    pub delta_lf_present: bool,
}

/// The derived per-frame loop-filter state (`struct loopfilter`'s
/// search-relevant fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopFilterLevels {
    /// `[vert, horz]` Y-plane levels.
    pub filter_level: [i32; 2],
    pub filter_level_u: i32,
    pub filter_level_v: i32,
    pub sharpness: i32,
}

/// Owning u8 staging of an [`LfSearchFrame<u16>`]'s six planes — the bd8
/// codec-invariant narrow ([`aom_dsp::lowbd::narrow_u16_to_u8`]), allocated
/// ONCE per pick so each trial filters u8 directly instead of paying a
/// per-trial u16 copy + narrow + u8 filter + widen + u16 SSE round trip.
/// [`LfSearchFrame::as_lowbd`] re-borrows these as the `u8` search frame.
pub struct LfSearchFrameU8Bufs {
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
    pub src_y: Vec<u8>,
    pub src_u: Vec<u8>,
    pub src_v: Vec<u8>,
}

impl LfSearchFrame<'_, u16> {
    /// Stage the six planes u8 (bd8 narrow; `min(255)` inside
    /// [`aom_dsp::lowbd::narrow_u16_to_u8`] is a safety net, bd8 samples are
    /// already `< 256`). Callers hold the buffers, then [`Self::as_lowbd`].
    pub fn stage_lowbd(&self) -> LfSearchFrameU8Bufs {
        LfSearchFrameU8Bufs {
            recon_y: aom_dsp::lowbd::narrow_u16_to_u8(self.recon_y),
            recon_u: aom_dsp::lowbd::narrow_u16_to_u8(self.recon_u),
            recon_v: aom_dsp::lowbd::narrow_u16_to_u8(self.recon_v),
            src_y: aom_dsp::lowbd::narrow_u16_to_u8(self.src_y),
            src_u: aom_dsp::lowbd::narrow_u16_to_u8(self.src_u),
            src_v: aom_dsp::lowbd::narrow_u16_to_u8(self.src_v),
        }
    }

    /// Borrow `bufs` ([`Self::stage_lowbd`]'s output) as the u8 search
    /// frame — the scalar fields carry over unchanged.
    pub fn as_lowbd<'b>(&'b self, bufs: &'b LfSearchFrameU8Bufs) -> LfSearchFrame<'b, u8> {
        LfSearchFrame {
            recon_y: &bufs.recon_y,
            recon_u: &bufs.recon_u,
            recon_v: &bufs.recon_v,
            src_y: &bufs.src_y,
            src_u: &bufs.src_u,
            src_v: &bufs.src_v,
            stride: self.stride,
            crop_width: self.crop_width,
            crop_height: self.crop_height,
            ss_x: self.ss_x,
            ss_y: self.ss_y,
            bd: self.bd,
            monochrome: self.monochrome,
            mi: self.mi,
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            delta_lf_present: self.delta_lf_present,
        }
    }
}

/// One loop-filter trial (`try_filter_frame`, picklpf.c, MINUS the
/// MT/YV12-buffer plumbing): clone only the plane under test (the other two
/// are never dereferenced — see the `plane_start`/`plane_end` gating note
/// below), apply [`loop_filter_frame_opt`] at the given levels, return the SSE
/// vs source. This port always starts each trial from the untouched
/// original reconstruction (no persistent "restore the unfiltered frame"
/// dance is needed since nothing here mutates shared state between trials).
///
/// Passing a same-sized real slice for the OTHER two planes (rather than a
/// tiny dummy) would also be correct but wasteful; a 1-element dummy is
/// safe because [`loop_filter_frame_opt`]'s `plane_start`/`plane_end` gate
/// (`planes_to_lf`) skips the entire per-plane loop body — including the
/// `match plane { .. }` buffer selection — for any plane outside
/// `[plane_start, plane_end)`, so the dummy is provably never indexed.
impl<P: LfTrialPixel> LfSearchFrame<'_, P> {
    #[allow(clippy::too_many_arguments)]
    fn try_filter_plane(
        &self,
        plane: usize,
        filter_level: [i32; 2],
        filter_level_u: i32,
        filter_level_v: i32,
        sharpness: i32,
        scratch: &mut Vec<P>,
    ) -> i64 {
        let p = LfParams {
            filter_level,
            filter_level_u,
            filter_level_v,
            sharpness,
            mode_ref_delta_enabled: true,
            ref_deltas: DEFAULT_REF_DELTAS,
            mode_deltas: DEFAULT_MODE_DELTAS,
            delta_lf_present: self.delta_lf_present,
            delta_lf_multi: false,
            lossless: [false; 8],
            seg: Default::default(),
        };
        let grid = LfMiGrid {
            mi: self.mi,
            stride: self.mi_cols as usize,
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
        };
        P::filter_trial(self, plane, &p, &grid, scratch)
    }
}

/// The u16 trial — every bit depth. At bd8 [`loop_filter_frame_opt`] still
/// narrows to u8, filters, and widens back internally; the lowbd path below
/// stages that narrow once per pick instead of once per trial.
impl LfTrialPixel for u16 {
    fn filter_trial(
        f: &LfSearchFrame<'_, u16>,
        plane: usize,
        p: &LfParams,
        grid: &LfMiGrid<'_>,
        scratch: &mut Vec<u16>,
    ) -> i64 {
        let uv_w = ((f.crop_width + f.ss_x as u32) >> f.ss_x) as usize;
        let uv_h = ((f.crop_height + f.ss_y as u32) >> f.ss_y) as usize;
        // Two separate dummy buffers (never dereferenced -- see the doc
        // comment above) since `LfFrameBuf` needs distinct mutable refs for
        // its 3 plane fields even when only one plane is under test.
        let mut dummy_a = [0u16; 1];
        let mut dummy_b = [0u16; 1];
        let (recon, src, plane_start) = match plane {
            0 => (f.recon_y, f.src_y, 0usize),
            1 => (f.recon_u, f.src_u, 1),
            _ => (f.recon_v, f.src_v, 2),
        };
        // C's try_filter_frame filters cur_frame->buf in place and
        // restores from a saved copy; we keep the copy semantics but the
        // buffer is REUSED across trials (C's pick_lf_lvl_frame_buffer).
        scratch.clear();
        scratch.extend_from_slice(recon);
        let (y, u, v) = match plane {
            0 => (scratch.as_mut_slice(), dummy_a.as_mut_slice(), dummy_b.as_mut_slice()),
            1 => (dummy_a.as_mut_slice(), scratch.as_mut_slice(), dummy_b.as_mut_slice()),
            _ => (dummy_a.as_mut_slice(), dummy_b.as_mut_slice(), scratch.as_mut_slice()),
        };
        let mut buf = LfFrameBuf {
            y,
            y_stride: f.stride,
            u,
            v,
            uv_stride: f.stride,
            crop_width: f.crop_width,
            crop_height: f.crop_height,
            ss_x: f.ss_x,
            ss_y: f.ss_y,
            bd: f.bd,
        };
        loop_filter_frame_opt(&mut buf, grid, p, plane_start, plane_start + 1);
        let (dst, w, h) = match plane {
            0 => (&buf.y[..], f.crop_width as usize, f.crop_height as usize),
            1 => (&buf.u[..], uv_w, uv_h),
            _ => (&buf.v[..], uv_w, uv_h),
        };
        sse_plane(src, f.stride, dst, f.stride, w, h)
    }
}

/// The bd8 u8 trial — same search, half the bytes moved per trial: the
/// staged-u8 plane copies straight into `scratch` and
/// [`loop_filter_frame_u8_opt`] filters it natively (the identical walk
/// `loop_filter_frame_opt`'s bd8 arm reaches through its own narrow/widen),
/// and the SSE runs u8 x u8 through [`aom_dsp::dist::sse`].
impl LfTrialPixel for u8 {
    fn filter_trial(
        f: &LfSearchFrame<'_, u8>,
        plane: usize,
        p: &LfParams,
        grid: &LfMiGrid<'_>,
        scratch: &mut Vec<u8>,
    ) -> i64 {
        let uv_w = ((f.crop_width + f.ss_x as u32) >> f.ss_x) as usize;
        let uv_h = ((f.crop_height + f.ss_y as u32) >> f.ss_y) as usize;
        let mut dummy_a = [0u8; 1];
        let mut dummy_b = [0u8; 1];
        let (recon, src, plane_start) = match plane {
            0 => (f.recon_y, f.src_y, 0usize),
            1 => (f.recon_u, f.src_u, 1),
            _ => (f.recon_v, f.src_v, 2),
        };
        scratch.clear();
        scratch.extend_from_slice(recon);
        let (y, u, v) = match plane {
            0 => (scratch.as_mut_slice(), dummy_a.as_mut_slice(), dummy_b.as_mut_slice()),
            1 => (dummy_a.as_mut_slice(), scratch.as_mut_slice(), dummy_b.as_mut_slice()),
            _ => (dummy_a.as_mut_slice(), dummy_b.as_mut_slice(), scratch.as_mut_slice()),
        };
        let mut buf = LfFrameBufU8 {
            y,
            y_stride: f.stride,
            u,
            v,
            uv_stride: f.stride,
            crop_width: f.crop_width,
            crop_height: f.crop_height,
            ss_x: f.ss_x,
            ss_y: f.ss_y,
        };
        loop_filter_frame_u8_opt(&mut buf, grid, p, plane_start, plane_start + 1);
        let (dst, w, h) = match plane {
            0 => (&buf.y[..], f.crop_width as usize, f.crop_height as usize),
            1 => (&buf.u[..], uv_w, uv_h),
            _ => (&buf.v[..], uv_w, uv_h),
        };
        aom_dsp::dist::sse(src, f.stride, dst, f.stride, w, h)
    }
}

/// `filter_level`/`filter_level_u`/`filter_level_v` for one trial candidate
/// (`try_filter_frame`'s `filter_level[2]` construction, picklpf.c:71-84).
/// `dir`: 2 = combined Y search (both directions equal the candidate), 0 =
/// vertical-only (horizontal held at `held_horiz`), 1 = horizontal-only
/// (vertical held at `held_vert`); irrelevant for chroma (`plane != 0`).
fn filt_pair(
    plane: usize,
    dir: usize,
    level: i32,
    held_vert: i32,
    held_horiz: i32,
) -> ([i32; 2], i32, i32) {
    if plane == 0 {
        let fl = match dir {
            0 => [level, held_horiz],
            1 => [held_vert, level],
            _ => [level, level], // dir == 2: combined
        };
        (fl, 0, 0)
    } else if plane == 1 {
        ([0, 0], level, 0)
    } else {
        ([0, 0], 0, level)
    }
}

/// `search_filter_level` (picklpf.c) for this envelope: `last_frame_filter_level`
/// is always 0 (KEY / intra-only — module docs), so `filt_mid` always starts
/// at 0, `filter_step` always starts at 4, and the search always runs to
/// `filter_step == 0` (`min_filter_step_thesh == 0`, `use_coarse_filter_level_search
/// == 0` default). `dir` per [`filt_pair`]'s doc; `held_vert`/`held_horiz`
/// are the ALREADY-COMMITTED other-axis levels from a prior stage of
/// [`pick_filter_level`] (only one is read per `dir`).
#[allow(clippy::too_many_arguments)]
fn search_filter_level<P: LfTrialPixel>(
    f: &LfSearchFrame<P>,
    plane: usize,
    dir: usize,
    held_vert: i32,
    held_horiz: i32,
    sharpness: i32,
    scratch: &mut Vec<P>,
) -> i32 {
    search_filter_level_impl(f, plane, dir, held_vert, held_horiz, sharpness, scratch, false)
}

/// The step search itself is sequential (each iteration's direction/step
/// depends on the last), but the `filt_low`/`filt_high` TRIALS inside one
/// iteration are independent — `par` runs that pair on a scoped worker with
/// its own trial scratch. Results are per-level deterministic, so the
/// parallel pre-fill is bit-identical to the serial `trial()` calls.
#[allow(clippy::too_many_arguments)]
fn search_filter_level_impl<P: LfTrialPixel>(
    f: &LfSearchFrame<P>,
    plane: usize,
    dir: usize,
    held_vert: i32,
    held_horiz: i32,
    sharpness: i32,
    scratch: &mut Vec<P>,
    par: bool,
) -> i32 {
    const MIN_FILTER_LEVEL: i32 = 0;
    let max_filter_level = MAX_LOOP_FILTER; // one-pass envelope: always 63 (module docs)
    let mut filt_mid = 0i32; // last_frame_filter_level always 0 in this envelope
    let mut filter_step = if filt_mid < 16 { 4 } else { filt_mid / 4 };
    let mut filt_direction = 0i32;
    let mut ss_err = [-1i64; (MAX_LOOP_FILTER + 1) as usize];

    let trial = |level: i32,
                 ss_err: &mut [i64; (MAX_LOOP_FILTER + 1) as usize],
                 scratch: &mut Vec<P>|
     -> i64 {
        if ss_err[level as usize] < 0 {
            let (fl, flu, flv) = filt_pair(plane, dir, level, held_vert, held_horiz);
            ss_err[level as usize] =
                f.try_filter_plane(plane, fl, flu, flv, sharpness, scratch);
        }
        ss_err[level as usize]
    };

    let mut best_err = trial(filt_mid, &mut ss_err, scratch);
    let mut filt_best = filt_mid;

    while filter_step > 0 {
        let filt_high = (filt_mid + filter_step).min(max_filter_level);
        let filt_low = (filt_mid - filter_step).max(MIN_FILTER_LEVEL);

        // Bias against raising the loop filter in favor of lowering it.
        // (one-pass envelope: no twopass section_intra_rating scaling.)
        let mut bias = (best_err >> (15 - (filt_mid / 8))) * i64::from(filter_step);
        // cm.features.tx_mode != ONLY_4X4 always holds where this result is
        // used (module docs) -- unconditional halving.
        bias >>= 1;

        if par {
            let need_low =
                filt_direction <= 0 && filt_low != filt_mid && ss_err[filt_low as usize] < 0;
            let need_high =
                filt_direction >= 0 && filt_high != filt_mid && ss_err[filt_high as usize] < 0;
            if need_low && need_high {
                let (lo, hi) = (
                    filt_pair(plane, dir, filt_low, held_vert, held_horiz),
                    filt_pair(plane, dir, filt_high, held_vert, held_horiz),
                );
                let mut scratch2 = Vec::new();
                let (l, h) = aom_dsp::par::join(
                    || f.try_filter_plane(plane, lo.0, lo.1, lo.2, sharpness, scratch),
                    || f.try_filter_plane(plane, hi.0, hi.1, hi.2, sharpness, &mut scratch2),
                );
                ss_err[filt_low as usize] = l;
                ss_err[filt_high as usize] = h;
            }
        }

        if filt_direction <= 0 && filt_low != filt_mid {
            let e = trial(filt_low, &mut ss_err, scratch);
            if e < best_err + bias {
                if e < best_err {
                    best_err = e;
                }
                filt_best = filt_low;
            }
        }
        if filt_direction >= 0 && filt_high != filt_mid {
            let e = trial(filt_high, &mut ss_err, scratch);
            if e < best_err - bias {
                best_err = e;
                filt_best = filt_high;
            }
        }

        if filt_best == filt_mid {
            filter_step /= 2;
            filt_direction = 0;
        } else {
            filt_direction = if filt_best < filt_mid { -1 } else { 1 };
            filt_mid = filt_best;
        }
    }
    filt_best
}

/// `av1_pick_filter_level` (picklpf.c) for this port's envelope — see the
/// module docs for every simplification and its justification. `allintra`
/// selects the ALLINTRA-vs-GOOD sharpness derivation (`sharpness` is this
/// port's `--sharpness`-equivalent input, 0 for every case actually
/// exercised so far).
/// The frame LF `sharpness_level` derivation (`av1_pick_filter_level`,
/// picklpf.c:220-247), shared by every pick method:
///
/// - `lf->sharpness_level = oxcf.algo_cfg.sharpness` when the encode mode is
///   ALLINTRA **or** the tuning is IQ / SSIMULACRA2 (:226-230); every other
///   mode forces 0 (frames serve as references).
/// - `--enable-adaptive-sharpness` (:232-247) then caps it by base qindex —
///   "lf sharpness 1 is visually closer to 7 than to 0", so the cap picks
///   only {7, 1, 0}: `qindex <= 112 -> 7`, `<= 160 -> 1`, else `0`;
///   `sharpness_level = min(sharpness_level, cap)`.
///
/// Returns the final `sharpness_level` for the frame header + LF limits.
pub fn frame_lf_sharpness(
    allintra: bool,
    iq_or_ssimulacra2_tune: bool,
    sharpness_cfg: i32,
    enable_adaptive_sharpness: bool,
    base_qindex: i32,
) -> i32 {
    let mut level = if allintra || iq_or_ssimulacra2_tune {
        sharpness_cfg
    } else {
        0
    };
    if enable_adaptive_sharpness {
        let max_lf_sharpness = if base_qindex <= 112 {
            7
        } else if base_qindex <= 160 {
            1
        } else {
            0
        };
        level = level.min(max_lf_sharpness);
    }
    level
}

pub fn pick_filter_level<P: LfTrialPixel>(
    f: &LfSearchFrame<P>,
    allintra: bool,
    sharpness_cfg: i32,
    non_dual: bool,
) -> LoopFilterLevels {
    let sharpness = if allintra { sharpness_cfg } else { 0 };

    // Y: combined search first (both directions equal), then -- for the DUAL
    // methods (`LPF_PICK_FROM_FULL_IMAGE`, the speed 0..=3 allintra default) --
    // refine vertical alone (horizontal held at the combined winner) then
    // horizontal alone (vertical held at the just-refined value): exactly
    // av1_pick_filter_level's dir=2 -> dir=0 -> dir=1 sequence (picklpf.c:373-383).
    // `LPF_PICK_FROM_FULL_IMAGE_NON_DUAL` (allintra speed>=4, speed_features.c:496)
    // SKIPS the two refine passes (`method != ..._NON_DUAL` guard, picklpf.c:376)
    // and leaves both luma levels equal to the combined dir=2 winner.
    // C's `pick_lf_lvl_frame_buffer`: one scratch the per-trial filtered
    // copies reuse — the port's `to_vec`-per-trial paid a full-plane malloc
    // on every level probe.
    let mut scratch = Vec::new();
    let combined = search_filter_level(f, 0, 2, 0, 0, sharpness, &mut scratch);
    let mut filter_level = [combined, combined];
    if !non_dual {
        filter_level[0] =
            search_filter_level(f, 0, 0, 0, filter_level[1], sharpness, &mut scratch);
        filter_level[1] =
            search_filter_level(f, 0, 1, filter_level[0], 0, sharpness, &mut scratch);
    }

    let (filter_level_u, filter_level_v) = if f.monochrome {
        (0, 0)
    } else {
        (
            search_filter_level(f, 1, 0, 0, 0, sharpness, &mut scratch),
            search_filter_level(f, 2, 0, 0, 0, sharpness, &mut scratch),
        )
    };

    LoopFilterLevels {
        filter_level,
        filter_level_u,
        filter_level_v,
        sharpness,
    }
}

/// `pick_filter_level` with the independent passes on workers: the luma
/// chain (combined → vert refine → horiz refine) is sequentially dependent,
/// but the U and V searches depend on nothing in it — three disjoint
/// `search_filter_level` walks, each with its own trial scratch. The result
/// is identical to the serial pick.
pub fn pick_filter_level_mt<P: LfTrialPixel>(
    f: &LfSearchFrame<P>,
    allintra: bool,
    sharpness_cfg: i32,
    non_dual: bool,
    workers: usize,
) -> LoopFilterLevels {
    if workers <= 1 || f.monochrome {
        return pick_filter_level(f, allintra, sharpness_cfg, non_dual);
    }
    let sharpness = if allintra { sharpness_cfg } else { 0 };
    // Y(2)+U(1)+V(1) = 4-way under `par` (std scope by default, the host's
    // rayon pool under the `rayon` feature). The luma chain is the long
    // pole — its low/high trial pairs run on a second worker.
    let (level_y, (level_u, level_v)) = aom_dsp::par::join(
        || {
            let mut scratch = Vec::new();
            let combined =
                search_filter_level_impl(f, 0, 2, 0, 0, sharpness, &mut scratch, true);
            let mut fl = [combined, combined];
            if !non_dual {
                fl[0] =
                    search_filter_level_impl(f, 0, 0, 0, fl[1], sharpness, &mut scratch, true);
                fl[1] =
                    search_filter_level_impl(f, 0, 1, fl[0], 0, sharpness, &mut scratch, true);
            }
            fl
        },
        || {
            aom_dsp::par::join(
                || {
                    let mut scratch = Vec::new();
                    search_filter_level(f, 1, 0, 0, 0, sharpness, &mut scratch)
                },
                || {
                    let mut scratch = Vec::new();
                    search_filter_level(f, 2, 0, 0, 0, sharpness, &mut scratch)
                },
            )
        },
    );
    LoopFilterLevels {
        filter_level: level_y,
        filter_level_u: level_u,
        filter_level_v: level_v,
        sharpness,
    }
}

/// `av1_pick_filter_level`'s `method >= LPF_PICK_FROM_Q` arm (picklpf.c:
/// 266-330) for this port's envelope — a one-pass shown KEY frame (no SVC
/// temporal layers, no rt screen / cyclic-refresh paths, no
/// `use_fast_fixed_part`). The level is a CLOSED-FORM function of the AC
/// quantizer — no reconstruction search at all:
///
/// - `q = av1_ac_quant_QTX(base_qindex, 0, bit_depth)` (:269)
/// - bd8 KEY: `filt_guess = ROUND_POWER_OF_TWO(q * 17563 - 421574, 18)`
///   (:303-305; the `q * 0.06699 - 1.60817` linear fit)
/// - bd10:    `filt_guess = ROUND_POWER_OF_TWO(q * 20723 + 4060632, 20)` (:309)
/// - bd12:    `filt_guess = ROUND_POWER_OF_TWO(q * 20723 + 16242526, 22)` (:311)
/// - bd10/bd12 KEY frames then subtract 4 (:320-322)
/// - clamp to `[0, MAX_LOOP_FILTER]`; **all four** levels (both Y
///   directions, U, V) get the SAME clamped value (:324-327 — C's
///   "retrain the model for Y, U, V" TODO)
///
/// The `inter_frame_multiplier` arm (:274-296) is non-KEY-only; the
/// LOOPFILTER_SELECTIVELY tail (:328) is `!frame_is_intra_only`-gated. The
/// sharpness derivation is method-independent (picklpf.c:225-230): allintra
/// keeps the CLI sharpness, other modes force 0 — same as
/// [`pick_filter_level`] (`enable_adaptive_sharpness` is default-off and out
/// of this envelope). `ROUND_POWER_OF_TWO` on the (possibly negative at tiny
/// q) bd8 KEY numerator uses C's arithmetic right shift — matched by Rust's
/// signed `>>`.
///
/// Consumer: `lpf_pick = LPF_PICK_FROM_Q` is allintra **speed >= 6**
/// (speed_features.c:559). No speed 0..=5 caller exists, so landing this
/// building block moves no existing byte gate; the speed-6 flip will route
/// the e2e harness LF derivation here (mirroring the `non_dual` flag's
/// wiring for speeds 4/5). Validated against REAL `aomenc --cpu-used=6`
/// header LF levels by `speed6_prep_lf_from_q_matches_real_aomenc`
/// (encoder_gate_e2e_byte_match.rs).
pub fn pick_filter_level_from_q(
    base_qindex: i32,
    bit_depth: u8,
    allintra: bool,
    sharpness_cfg: i32,
) -> LoopFilterLevels {
    let q = i32::from(aom_dsp::quant::av1_ac_quant_qtx(base_qindex, 0, bit_depth));
    // ROUND_POWER_OF_TWO(value, n) = ((value) + ((1 << (n)) >> 1)) >> (n),
    // arithmetic shift (aom_dsp/aom_dsp_common.h) — i32 `>>` matches.
    let rpot = |value: i32, n: i32| (value + ((1 << n) >> 1)) >> n;
    let filt_guess = match bit_depth {
        8 => rpot(q * 17563 - 421574, 18), // KEY-frame fit (:303-305)
        10 => rpot(q * 20723 + 4060632, 20) - 4, // :309 + the KEY -4 (:320-322)
        12 => rpot(q * 20723 + 16242526, 22) - 4, // :311 + the KEY -4
        _ => unreachable!("bit_depth is 8/10/12"),
    };
    let level = filt_guess.clamp(0, MAX_LOOP_FILTER);
    LoopFilterLevels {
        filter_level: [level, level],
        filter_level_u: level,
        filter_level_v: level,
        sharpness: if allintra { sharpness_cfg } else { 0 },
    }
}

// ---- mi-grid construction from this port's OWN picked+packed trees -------

/// Build the [`LfMi`] grid the loop filter reads, from this port's OWN
/// picked+packed [`SbTree`]s (a post-hoc walk over [`crate::pack::pack_tile`]'s
/// return value, deliberately NOT a `pack_sb` modification — mirrors the
/// existing `stamp_grid_from_tree`/`ModeGrid` pattern in
/// `partition_pick.rs`, duplicated here for a different per-cell payload,
/// consistent with that pattern already existing 3x in this crate). All-intra
/// KEY envelope: `segment_id`/`ref0`/`mode_lf`/`is_inter`/`delta_lf*` are
/// frame constants (no segmentation; every block is intra, `ref0 ==
/// INTRA_FRAME == 0`; every intra mode maps to `mode_lf == 0` via
/// `MODE_LF_LUT`; no delta-lf signaling in this envelope).
pub fn build_lf_mi_grid(
    trees: &[SbTree],
    mi_rows: i32,
    mi_cols: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
) -> Vec<LfMi> {
    let mut mi = vec![LfMi::default(); mi_rows as usize * mi_cols as usize];
    let stride = mi_cols as usize;
    for (idx, tree) in trees.iter().enumerate() {
        let r = idx as i32 / n_sb_cols;
        let c = idx as i32 % n_sb_cols;
        stamp_lf_tree(
            &mut mi,
            stride,
            tree,
            r * sb_mi,
            c * sb_mi,
            sb_size,
            mi_rows,
            mi_cols,
        );
    }
    mi
}

/// `build_lf_mi_grid` threaded by superblock rows: each SB stamps only cells
/// inside its own `sb_mi` block, so contiguous SB-row bands own disjoint `mi`
/// rows and the result is identical to the serial build.
#[allow(clippy::too_many_arguments)]
pub fn build_lf_mi_grid_mt(
    trees: &[SbTree],
    mi_rows: i32,
    mi_cols: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
    workers: usize,
) -> Vec<LfMi> {
    let n_sb_rows = trees.len() as i32 / n_sb_cols;
    if workers <= 1 || n_sb_rows < 2 {
        return build_lf_mi_grid(trees, mi_rows, mi_cols, n_sb_cols, sb_mi, sb_size);
    }
    let mut mi = vec![LfMi::default(); mi_rows as usize * mi_cols as usize];
    let stride = mi_cols as usize;
    let workers = workers.min(n_sb_rows as usize);
    let (per, rem) = (n_sb_rows as usize / workers, n_sb_rows as usize % workers);
    // One `&mut` band per SB row; workers claim whole rows.
    let mut bands: Vec<&mut [LfMi]> = Vec::with_capacity(n_sb_rows as usize);
    let mut rest = mi.as_mut_slice();
    for _ in 0..n_sb_rows {
        let n = (sb_mi as usize * stride).min(rest.len());
        let (b, r) = rest.split_at_mut(n);
        bands.push(b);
        rest = r;
    }
    let bands = std::sync::Mutex::new(
        bands
            .into_iter()
            .map(Some)
            .chain(std::iter::repeat_with(|| None))
            .take(n_sb_rows as usize)
            .collect::<Vec<_>>(),
    );
    let bands = &bands;
    aom_dsp::par::map_workers(workers, |w| {
        // Worker w's contiguous chunk: per + (w < rem) — same static
        // assignment as the serial build, so the grid is identical.
        let start = w * per + w.min(rem);
        let take = per + usize::from(w < rem);
        for sb_r_i in start..start + take {
            let band = &mut *{
                let mut g = bands.lock().unwrap_or_else(|p| p.into_inner());
                g[sb_r_i].take().expect("one band per SB row")
            };
            let band_rows = (band.len() / stride) as i32;
            for c in 0..n_sb_cols {
                let tree = &trees[sb_r_i * n_sb_cols as usize + c as usize];
                stamp_lf_tree(
                    band,
                    stride,
                    tree,
                    0,
                    c * sb_mi,
                    sb_size,
                    band_rows,
                    mi_cols,
                );
            }
        }
    });
    mi
}

/// `--delta-lf-mode=1`: overwrite each SB's `delta_lf_from_base` in an already-
/// built [`build_lf_mi_grid`] grid from the per-SB delta-lf values (SB raster
/// order, matching `setup_delta_q`'s stamp of `mi_alloc[..].delta_lf_from_base`
/// over the whole SB, encodeframe.c:388-398). `get_filter_level` reads
/// `delta_lf_from_base` (single mode); the multi `delta_lf[]` is only read when
/// `delta_lf_multi`, off in this envelope. `LfMi.delta_lf_from_base` is `i8`
/// (values are already clamped to [-63, 63]).
pub fn stamp_lf_delta_lf(
    mi: &mut [LfMi],
    delta_lf_per_sb: &[i32],
    mi_rows: i32,
    mi_cols: i32,
    n_sb_cols: i32,
    sb_mi: i32,
) {
    let stride = mi_cols as usize;
    for (idx, &dlf) in delta_lf_per_sb.iter().enumerate() {
        let sb_r = (idx as i32 / n_sb_cols) * sb_mi;
        let sb_c = (idx as i32 % n_sb_cols) * sb_mi;
        let rows = sb_mi.min(mi_rows - sb_r) as usize;
        let cols = sb_mi.min(mi_cols - sb_c) as usize;
        let dlf8 = dlf as i8;
        for r in 0..rows {
            let base = (sb_r as usize + r) * stride + sb_c as usize;
            for cell in &mut mi[base..base + cols] {
                cell.delta_lf_from_base = dlf8;
                cell.delta_lf = [dlf8; 4]; // FRAME_LF_COUNT
            }
        }
    }
}

fn lf_cell(bsize: usize, w: &LeafWinner) -> LfMi {
    LfMi {
        bsize: bsize as u8,
        tx_size: w.tx_size as u8,
        segment_id: 0,
        ref0: 0,
        mode_lf: 0,
        is_inter: false,
        skip_txfm: w.skip_txfm,
        delta_lf_from_base: 0,
        delta_lf: [0; 4],
    }
}

#[allow(clippy::too_many_arguments)]
fn stamp_lf(
    mi: &mut [LfMi],
    stride: usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    w: &LeafWinner,
    mi_rows: i32,
    mi_cols: i32,
) {
    let rows = (MI_SIZE_HIGH_B[bsize] as i32).min(mi_rows - mi_row) as usize;
    let cols = (MI_SIZE_WIDE_B[bsize] as i32).min(mi_cols - mi_col) as usize;
    let cell = lf_cell(bsize, w);
    for r in 0..rows {
        let base = (mi_row as usize + r) * stride + mi_col as usize;
        mi[base..base + cols].fill(cell);
    }
}

/// PARTITION_HORZ/VERT/HORZ_4/VERT_4/SPLIT constants matching
/// `aom_dsp::entropy::partition::get_partition_subsize`'s expected values
/// (duplicated per-file, matching this crate's established convention —
/// see e.g. `pack.rs`'s own copies).
const PARTITION_HORZ: i32 = 1;
const PARTITION_VERT: i32 = 2;
const PARTITION_HORZ_A: i32 = 4;
const PARTITION_HORZ_B: i32 = 5;
const PARTITION_VERT_A: i32 = 6;
const PARTITION_VERT_B: i32 = 7;
const PARTITION_HORZ_4: i32 = 8;
const PARTITION_VERT_4: i32 = 9;

/// Mirrors `partition_pick.rs`'s private `stamp_grid_from_tree` /
/// `pack.rs`'s `pack_sb` recursion exactly (frame-bound gating on every
/// rect/4-way second+ sub-block), stamping [`LfMi`] cells instead of a mode
/// byte.
#[allow(clippy::too_many_arguments)]
fn stamp_lf_tree(
    mi: &mut [LfMi],
    stride: usize,
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
        SbTree::Leaf(w) => stamp_lf(mi, stride, mi_row, mi_col, bsize, w, mi_rows, mi_cols),
        SbTree::Split(kids) => {
            let sub = crate::partition::split_subsize(bsize);
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            for (idx, child) in kids.iter().enumerate() {
                stamp_lf_tree(
                    mi,
                    stride,
                    child,
                    mi_row + ((idx as i32) >> 1) * hbs,
                    mi_col + ((idx as i32) & 1) * hbs,
                    sub,
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Horz(subs) => {
            let sub = aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, sub, &subs[0], mi_rows, mi_cols);
            if mi_row + hbs < mi_rows {
                stamp_lf(
                    mi,
                    stride,
                    mi_row + hbs,
                    mi_col,
                    sub,
                    &subs[1],
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Vert(subs) => {
            let sub = aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, sub, &subs[0], mi_rows, mi_cols);
            if mi_col + hbs < mi_cols {
                stamp_lf(
                    mi,
                    stride,
                    mi_row,
                    mi_col + hbs,
                    sub,
                    &subs[1],
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Horz4(subs) => {
            let sub =
                aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_4) as usize;
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, w) in subs.iter().enumerate() {
                let this_mi_row = mi_row + (i as i32) * quarter_step;
                if i > 0 && this_mi_row >= mi_rows {
                    break;
                }
                // A strip past the frame edge is absent (`rd_pick_4partition` broke
                // out before it, C partition_search.c:3948); every in-frame strip won.
                let w = w.as_ref().expect("in-frame 4-way strip carries a winner");
                stamp_lf(mi, stride, this_mi_row, mi_col, sub, w, mi_rows, mi_cols);
            }
        }
        SbTree::Vert4(subs) => {
            let sub =
                aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_4) as usize;
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, w) in subs.iter().enumerate() {
                let this_mi_col = mi_col + (i as i32) * quarter_step;
                if i > 0 && this_mi_col >= mi_cols {
                    break;
                }
                // A strip past the frame edge is absent (`rd_pick_4partition` broke
                // out before it, C partition_search.c:3948); every in-frame strip won.
                let w = w.as_ref().expect("in-frame 4-way strip carries a winner");
                stamp_lf(mi, stride, mi_row, this_mi_col, sub, w, mi_rows, mi_cols);
            }
        }
        SbTree::HorzA(subs) => {
            // AB is interior-only (module docs on encode_sb.rs's SbTree::
            // HorzA) -- no frame-bound gating on any of the 3 sub-blocks.
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize =
                aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_A) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(
                mi, stride, mi_row, mi_col, bsize2, &subs[0], mi_rows, mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row,
                mi_col + hbs,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col,
                subsize,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
        SbTree::HorzB(subs) => {
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize =
                aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_B) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(
                mi, stride, mi_row, mi_col, subsize, &subs[0], mi_rows, mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertA(subs) => {
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize =
                aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_A) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(
                mi, stride, mi_row, mi_col, bsize2, &subs[0], mi_rows, mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row,
                mi_col + hbs,
                subsize,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertB(subs) => {
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize =
                aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_B) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(
                mi, stride, mi_row, mi_col, subsize, &subs[0], mi_rows, mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row,
                mi_col + hbs,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
        // Off-frame placeholder — unreachable past the entry frame-bound guard.
        SbTree::Absent => {}
    }
}
