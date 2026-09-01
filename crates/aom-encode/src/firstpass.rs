//! The two-pass encoder's first pass — port of `av1/encoder/firstpass.c`.
//!
//! The first pass runs a cheap 16x16-block analysis over every frame and
//! writes one [`FirstpassStats`] record per frame. Pass 2 reads those records
//! back to choose the GOP structure and the per-frame bit allocation, so
//! every number here is an input to a rate-control decision, not a
//! diagnostic.
//!
//! | Rust | C (`av1/encoder/firstpass.c`) |
//! |---|---|
//! | [`FirstpassStats::zero`] | `av1_twopass_zero_stats` (:91) |
//! | [`FirstpassStats::accumulate`] | `av1_accumulate_stats` (:123) |
//! | [`FirstpassStats::normalize`] | `normalize_firstpass_stats` (:874, static) |
//! | [`get_unit_rows`] / [`get_unit_cols`] | `get_unit_rows` (:153) / `get_unit_cols` (:163), static |
//! | [`get_num_mbs`] | `get_num_mbs` (:174, static) |
//! | [`get_unit_rows_in_tile`] / [`get_unit_cols_in_tile`] | `av1_get_unit_rows_in_tile` (:1105) / `_cols_` (:1114) |
//! | [`get_search_range`] | `get_search_range` (:257, static) |
//! | [`find_fp_qindex`] | `find_fp_qindex` (:368, static) |
//! | [`raw_motion_error_stdev`] | `raw_motion_error_stdev` (:372, static) |
//! | [`calc_wavelet_energy`] | `calc_wavelet_energy` (:395, static) |
//! | [`FrameStats::accumulate_frame_stats`] | `accumulate_frame_stats` (:1043, static) |
//! | [`FrameStats::accumulate_mv_stats`] | `accumulate_mv_stats` (:635, static) |
//! | [`FirstpassInfo`] | `FIRSTPASS_INFO` + its seven `av1_firstpass_info_*` entry points |
//!
//! # Not ported, and why
//! - `output_stats` (:61) writes an `aom_codec_cx_pkt` into the encoder's
//!   output packet list, and `av1_end_first_pass` (:192) is a one-line call to
//!   it. Packet plumbing, no arithmetic.
//! - `print_reconstruction_frame` (:1024) is debug-only.
//! - `setup_firstpass_data` (:1081) and `av1_free_firstpass_data` (:1098) are
//!   the `aom_calloc`/`aom_free` pair for the per-MB stats arrays; Rust owns
//!   those by value.
//! - `av1_first_pass_row` (:1152), `first_pass_tile` (:1124) and
//!   `first_pass_tiles` (:1137) shard the block walk across row/tile workers.
//!   The oracle is built `CONFIG_MULTITHREAD=0` and this port is
//!   single-threaded by construction.
//! - `av1_get_first_pass_search_site_config` (:266) selects a preallocated
//!   `search_site_config` out of the encoder's table by stride; the table is
//!   allocation state, not a computation.
//!
//! **Still missing from this module**, and named rather than glossed: the
//! per-block first pass itself — `firstpass_intra_prediction`,
//! `firstpass_inter_prediction`, `first_pass_motion_search`,
//! `first_pass_intra_pred_and_calc_diff`,
//! `first_pass_predict_intra_block_for_luma_plane`,
//! `get_prediction_error` / `highbd_get_prediction_error` /
//! `get_prediction_error_bitdepth`, `get_block_variance_fn` /
//! `highbd_get_block_variance_fn`, `get_bsize`, `update_firstpass_stats`,
//! `av1_first_pass` and `av1_noop_first_pass_frame`. Those need the encoder's
//! frame buffers and motion search wired up; this module is the arithmetic
//! layer underneath them.
//!
//! # Differential coverage
//! `crates/aom-encode/tests/firstpass_diff.rs`. The eleven exported entry
//! points are **tier 1** (bound straight to the archive's symbols); the
//! statics are **tier 1c** through `shim/fp_shim.c`, which compiles
//! firstpass.c verbatim.

/// `FIRST_PASS_Q` (firstpass.c:51) — the fixed Q the first pass analyses at.
pub const FIRST_PASS_Q: f64 = 10.0;

/// `QINDEX_RANGE` (`av1/common/quant_common.h`).
const QINDEX_RANGE: i32 = 256;

/// `MAX_FULL_PEL_VAL` (`encoder/mcomp_structs.h:22`) —
/// `(1 << (MAX_MVSEARCH_STEPS - 1)) - 1`, i.e. 1023.
const MAX_FULL_PEL_VAL: i32 = 1023;

/// `MI_SIZE` (`av1/common/enums.h`).
const MI_SIZE: i32 = 4;

/// `INVALID_ROW` (firstpass.c:469) — the sentinel
/// `FrameStats::image_data_start_row` carries until a block with real image
/// data is seen.
pub const INVALID_ROW: i32 = -1;

/// `FIRSTPASS_STATS` (firstpass.h:41) — one frame's first-pass record.
///
/// Field order and widths are C's, and the layout is asserted against C's own
/// struct by `firstpass_stats_layout_matches_c`, because these records are
/// also what a `--pass=1` run writes to disk for a later `--pass=2` to read.
///
/// Everything is `f64` except `is_flash`, which C declares `int64_t` even
/// though it only ever holds 0 or 1 — the width is part of the on-disk
/// layout, so it is not narrowed to a `bool` here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FirstpassStats {
    /// Frame number.
    pub frame: f64,
    /// Weight assigned to this frame's contribution.
    pub weight: f64,
    /// Intra prediction error.
    pub intra_error: f64,
    /// Average wavelet energy over the frame.
    pub frame_avg_wavelet_energy: f64,
    /// Best of intra and last-frame inter error.
    pub coded_error: f64,
    /// Best of intra and golden-frame inter error.
    pub sr_coded_error: f64,
    /// Best coded error against the long-term reference.
    pub lt_coded_error: f64,
    /// Fraction of blocks that chose inter.
    pub pcnt_inter: f64,
    /// Fraction of blocks with a non-zero motion vector.
    pub pcnt_motion: f64,
    /// Fraction of blocks that chose the second reference.
    pub pcnt_second_ref: f64,
    /// Fraction of blocks where inter and intra were both low and close.
    pub pcnt_neutral: f64,
    /// Fraction of blocks whose intra error was negligible.
    pub intra_skip_pct: f64,
    /// Rows of inactive (letterbox) area.
    pub inactive_zone_rows: f64,
    /// Columns of inactive (pillarbox) area.
    pub inactive_zone_cols: f64,
    /// Mean motion-vector row.
    pub mvr: f64,
    /// Mean absolute motion-vector row.
    pub mvr_abs: f64,
    /// Mean motion-vector column.
    pub mvc: f64,
    /// Mean absolute motion-vector column.
    pub mvc_abs: f64,
    /// Motion-vector row variance.
    pub mvrv: f64,
    /// Motion-vector column variance.
    pub mvcv: f64,
    /// Net inward/outward motion.
    pub mv_in_out_count: f64,
    /// Count of distinct non-zero motion vectors.
    pub new_mv_count: f64,
    /// Duration this record covers.
    pub duration: f64,
    /// Number of frames folded into this record.
    pub count: f64,
    /// Standard deviation of the zero-MV motion error.
    pub raw_error_stdev: f64,
    /// Whether this frame is a flash. `int64_t` in C.
    pub is_flash: i64,
    /// Estimated noise variance.
    pub noise_var: f64,
    /// Correlation coefficient with the previous frame.
    pub cor_coeff: f64,
    /// `log1p(intra_error)`, accumulated separately.
    pub log_intra_error: f64,
    /// `log1p(coded_error)`, accumulated separately.
    pub log_coded_error: f64,
}

impl FirstpassStats {
    /// `av1_twopass_zero_stats` (firstpass.c:91) — the identity record for
    /// accumulation.
    ///
    /// **Not a zeroed struct.** `duration` starts at 1.0 and `cor_coeff` at
    /// 1.0, and `raw_error_stdev` is *not written at all* — C leaves whatever
    /// was in the struct. That last one is why this is a method on an
    /// existing value rather than an associated constructor: reproducing "all
    /// fields but one" requires knowing what the one was.
    pub fn zero(&mut self) {
        self.frame = 0.0;
        self.weight = 0.0;
        self.intra_error = 0.0;
        self.frame_avg_wavelet_energy = 0.0;
        self.coded_error = 0.0;
        self.log_intra_error = 0.0;
        self.log_coded_error = 0.0;
        self.sr_coded_error = 0.0;
        self.lt_coded_error = 0.0;
        self.pcnt_inter = 0.0;
        self.pcnt_motion = 0.0;
        self.pcnt_second_ref = 0.0;
        self.pcnt_neutral = 0.0;
        self.intra_skip_pct = 0.0;
        self.inactive_zone_rows = 0.0;
        self.inactive_zone_cols = 0.0;
        self.mvr = 0.0;
        self.mvr_abs = 0.0;
        self.mvc = 0.0;
        self.mvc_abs = 0.0;
        self.mvrv = 0.0;
        self.mvcv = 0.0;
        self.mv_in_out_count = 0.0;
        self.new_mv_count = 0.0;
        self.count = 0.0;
        self.duration = 1.0;
        self.is_flash = 0;
        self.noise_var = 0.0;
        self.cor_coeff = 1.0;
        // raw_error_stdev is deliberately untouched — C does not write it.
    }

    /// `av1_accumulate_stats` (firstpass.c:123) — fold one frame's record
    /// into a running total.
    ///
    /// Three fields are **not** accumulated — `raw_error_stdev`, `is_flash`,
    /// `noise_var` and `cor_coeff` — because they are per-frame properties
    /// with no meaningful sum. And two are not simple sums: `log_intra_error`
    /// and `log_coded_error` accumulate `log1p` of the *frame's* error, not
    /// the log of the running total, so the totals are sums of logs rather
    /// than the log of a sum. A port that recomputed them from the summed
    /// `coded_error` at the end would give a different (and much larger)
    /// number.
    pub fn accumulate(&mut self, frame: &Self) {
        self.frame += frame.frame;
        self.weight += frame.weight;
        self.intra_error += frame.intra_error;
        self.log_intra_error += frame.intra_error.ln_1p();
        self.log_coded_error += frame.coded_error.ln_1p();
        self.frame_avg_wavelet_energy += frame.frame_avg_wavelet_energy;
        self.coded_error += frame.coded_error;
        self.sr_coded_error += frame.sr_coded_error;
        self.lt_coded_error += frame.lt_coded_error;
        self.pcnt_inter += frame.pcnt_inter;
        self.pcnt_motion += frame.pcnt_motion;
        self.pcnt_second_ref += frame.pcnt_second_ref;
        self.pcnt_neutral += frame.pcnt_neutral;
        self.intra_skip_pct += frame.intra_skip_pct;
        self.inactive_zone_rows += frame.inactive_zone_rows;
        self.inactive_zone_cols += frame.inactive_zone_cols;
        self.mvr += frame.mvr;
        self.mvr_abs += frame.mvr_abs;
        self.mvc += frame.mvc;
        self.mvc_abs += frame.mvc_abs;
        self.mvrv += frame.mvrv;
        self.mvcv += frame.mvcv;
        self.mv_in_out_count += frame.mv_in_out_count;
        self.new_mv_count += frame.new_mv_count;
        self.count += frame.count;
        self.duration += frame.duration;
    }

    /// `normalize_firstpass_stats` (firstpass.c:874, static) — divide the
    /// frame's raw sums down to per-macroblock and per-pixel units.
    ///
    /// The three normalizers are different on purpose: errors and
    /// `new_mv_count` divide by the 16x16 macroblock count, MV means divide
    /// by the frame dimension in the matching axis, and MV *variances* divide
    /// by its square. Note `log_coded_error` / `log_intra_error` are
    /// RECOMPUTED here from the already-normalized errors, not divided — so
    /// they are `log1p(error / num_mbs)`, which is what
    /// [`Self::accumulate`] then sums.
    pub fn normalize(&mut self, num_mbs_16x16: f64, f_w: f64, f_h: f64) {
        self.coded_error /= num_mbs_16x16;
        self.sr_coded_error /= num_mbs_16x16;
        self.lt_coded_error /= num_mbs_16x16;
        self.intra_error /= num_mbs_16x16;
        self.frame_avg_wavelet_energy /= num_mbs_16x16;
        self.log_coded_error = self.coded_error.ln_1p();
        self.log_intra_error = self.intra_error.ln_1p();
        self.mvr /= f_h;
        self.mvr_abs /= f_h;
        self.mvc /= f_w;
        self.mvc_abs /= f_w;
        self.mvrv /= f_h * f_h;
        self.mvcv /= f_w * f_w;
        self.new_mv_count /= num_mbs_16x16;
    }
}

/// `FRAME_STATS` (firstpass.h:479) — the per-block accumulator the first pass
/// fills, before it is condensed into a [`FirstpassStats`].
///
/// Layout is C's and is asserted against it, because
/// `accumulate_frame_stats` sums an array of these.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameStats {
    /// Intra prediction error.
    pub intra_error: i64,
    /// Wavelet energy.
    pub frame_avg_wavelet_energy: i64,
    /// Best of intra and last-frame inter error.
    pub coded_error: i64,
    /// Best of intra and golden-frame inter error.
    pub sr_coded_error: i64,
    /// Best coded error against the long-term reference.
    pub lt_coded_error: i64,
    /// Count of motion vectors.
    pub mv_count: i32,
    /// Count of blocks that chose inter.
    pub inter_count: i32,
    /// Count of blocks that chose the second reference.
    pub second_ref_count: i32,
    /// Count of neutral blocks. `double` in C, despite being a count.
    pub neutral_count: f64,
    /// Count of blocks whose intra error was negligible.
    pub intra_skip_count: i32,
    /// First row with real image data, or [`INVALID_ROW`].
    pub image_data_start_row: i32,
    /// Count of distinct non-zero motion vectors.
    pub new_mv_count: i32,
    /// Net inward motion, +1 per inward component, -1 per outward.
    pub sum_in_vectors: i32,
    /// Sum of MV rows.
    pub sum_mvr: i32,
    /// Sum of MV columns.
    pub sum_mvc: i32,
    /// Sum of |MV row|.
    pub sum_mvr_abs: i32,
    /// Sum of |MV column|.
    pub sum_mvc_abs: i32,
    /// Sum of MV row squares.
    pub sum_mvrs: i64,
    /// Sum of MV column squares.
    pub sum_mvcs: i64,
    /// Intra weighting factor.
    pub intra_factor: f64,
    /// Brightness weighting factor.
    pub brightness_factor: f64,
}

impl FrameStats {
    /// `accumulate_frame_stats` (firstpass.c:1043, static) — sum the per-block
    /// accumulators into one frame-level record.
    ///
    /// Two things this is not: it is not a field-wise sum (the result's
    /// `image_data_start_row` is the FIRST valid row seen in raster order,
    /// not a sum), and the traversal order is load-bearing for exactly that
    /// reason — `j` outer over rows, `i` inner over columns.
    ///
    /// Every other field sums, including `intra_factor` and
    /// `brightness_factor`, which the caller divides down afterwards.
    #[must_use]
    pub fn accumulate_frame_stats(mb_stats: &[Self], mb_rows: usize, mb_cols: usize) -> Self {
        let mut stats = Self {
            image_data_start_row: INVALID_ROW,
            ..Self::default()
        };
        for j in 0..mb_rows {
            for i in 0..mb_cols {
                let mb = mb_stats[j * mb_cols + i];
                stats.brightness_factor += mb.brightness_factor;
                stats.coded_error += mb.coded_error;
                stats.frame_avg_wavelet_energy += mb.frame_avg_wavelet_energy;
                if stats.image_data_start_row == INVALID_ROW
                    && mb.image_data_start_row != INVALID_ROW
                {
                    stats.image_data_start_row = mb.image_data_start_row;
                }
                stats.inter_count += mb.inter_count;
                stats.intra_error += mb.intra_error;
                stats.intra_factor += mb.intra_factor;
                stats.intra_skip_count += mb.intra_skip_count;
                stats.mv_count += mb.mv_count;
                stats.neutral_count += mb.neutral_count;
                stats.new_mv_count += mb.new_mv_count;
                stats.second_ref_count += mb.second_ref_count;
                stats.sr_coded_error += mb.sr_coded_error;
                stats.lt_coded_error += mb.lt_coded_error;
                stats.sum_in_vectors += mb.sum_in_vectors;
                stats.sum_mvc += mb.sum_mvc;
                stats.sum_mvc_abs += mb.sum_mvc_abs;
                stats.sum_mvcs += mb.sum_mvcs;
                stats.sum_mvr += mb.sum_mvr;
                stats.sum_mvr_abs += mb.sum_mvr_abs;
                stats.sum_mvrs += mb.sum_mvrs;
            }
        }
        stats
    }

    /// `accumulate_mv_stats` (firstpass.c:635, static) — fold one block's
    /// motion vector into the frame's motion statistics.
    ///
    /// A zero MV contributes nothing at all (C returns immediately), so
    /// `mv_count` counts *moving* blocks, not blocks. `new_mv_count` counts
    /// runs: it increments only when this MV differs from the last non-zero
    /// one, which makes it a measure of motion coherence rather than of
    /// distinct vectors.
    ///
    /// `sum_in_vectors` is the inward/outward balance. A block in the top
    /// half with a downward MV is moving *inwards*, so it decrements;
    /// mirrored for the bottom half. Blocks exactly on the mid-row or
    /// mid-column (`mb_row == mb_rows / 2`) contribute nothing — the `if /
    /// else if` has no `else`, and that gap is deliberate, not an oversight
    /// to tidy.
    ///
    /// Note the two MVs: the sign tests use the *full-pel* `mv`, while the
    /// zero test, the equality test and `last_non_zero_mv` use the sub-pel
    /// `best_mv`. They can disagree — a sub-pel MV under half a pixel is
    /// non-zero as `best_mv` and zero as `mv` — which is why both are
    /// parameters.
    pub fn accumulate_mv_stats(
        &mut self,
        best_mv: (i16, i16),
        mv: (i16, i16),
        mb_row: i32,
        mb_col: i32,
        mb_rows: i32,
        mb_cols: i32,
        last_non_zero_mv: &mut (i16, i16),
    ) {
        if best_mv == (0, 0) {
            return;
        }
        self.mv_count += 1;
        if best_mv != *last_non_zero_mv {
            self.new_mv_count += 1;
        }
        *last_non_zero_mv = best_mv;

        let (mv_row, mv_col) = mv;
        // Row: inward is negative in the top half, positive in the bottom.
        if mb_row < mb_rows / 2 {
            self.sum_in_vectors -= (mv_row > 0) as i32;
            self.sum_in_vectors += (mv_row < 0) as i32;
        } else if mb_row > mb_rows / 2 {
            self.sum_in_vectors += (mv_row > 0) as i32;
            self.sum_in_vectors -= (mv_row < 0) as i32;
        }
        // Column: the same, mirrored about the mid-column.
        if mb_col < mb_cols / 2 {
            self.sum_in_vectors -= (mv_col > 0) as i32;
            self.sum_in_vectors += (mv_col < 0) as i32;
        } else if mb_col > mb_cols / 2 {
            self.sum_in_vectors += (mv_col > 0) as i32;
            self.sum_in_vectors -= (mv_col < 0) as i32;
        }
    }
}

/// `mi_size_high_log2` / `mi_size_wide_log2` for a `BLOCK_SIZE`, as
/// `(wide_log2, high_log2)`.
///
/// The first pass only ever uses `BLOCK_8X8` or `BLOCK_16X16`
/// (`get_fp_block_size`, firstpass.h:554, picks between them on
/// `is_screen_content_type`), but the three helpers below index the table
/// with whatever they are given, so the full square set is covered and
/// anything else is rejected rather than silently mapped.
fn mi_size_log2(fp_block_size: i32) -> (u32, u32) {
    // BLOCK_4X4=0, 4X8=1, 8X4=2, 8X8=3, 8X16=4, 16X8=5, 16X16=6, 16X32=7,
    // 32X16=8, 32X32=9, 32X64=10, 64X32=11, 64X64=12, 64X128=13, 128X64=14,
    // 128X128=15 (av1/common/enums.h).
    match fp_block_size {
        0 => (0, 0),
        1 => (0, 1),
        2 => (1, 0),
        3 => (1, 1),
        4 => (1, 2),
        5 => (2, 1),
        6 => (2, 2),
        7 => (2, 3),
        8 => (3, 2),
        9 => (3, 3),
        10 => (3, 4),
        11 => (4, 3),
        12 => (4, 4),
        13 => (4, 5),
        14 => (5, 4),
        15 => (5, 5),
        _ => panic!("fp_block_size {fp_block_size} is not a BLOCK_SIZE"),
    }
}

/// `mi_size_{wide,high}_log2[BLOCK_16X16]` — 2 on both axes.
const MB_MI_LOG2: u32 = 2;

/// `get_unit_rows` (firstpass.c:153, static) — rescale a 16x16-macroblock row
/// count to first-pass block units.
///
/// The two arms are a shift each way, not a divide: a block taller than 16
/// covers several macroblock rows, a shorter one splits each into several.
/// The `>` in the test means the equal case (a 16-tall block) takes the
/// *left*-shift arm with a zero shift, which is the identity either way.
#[must_use]
pub fn get_unit_rows(fp_block_size: i32, mb_rows: i32) -> i32 {
    let (_, height_mi_log2) = mi_size_log2(fp_block_size);
    if height_mi_log2 > MB_MI_LOG2 {
        mb_rows >> (height_mi_log2 - MB_MI_LOG2)
    } else {
        mb_rows << (MB_MI_LOG2 - height_mi_log2)
    }
}

/// `get_unit_cols` (firstpass.c:163, static) — the column counterpart of
/// [`get_unit_rows`].
#[must_use]
pub fn get_unit_cols(fp_block_size: i32, mb_cols: i32) -> i32 {
    let (width_mi_log2, _) = mi_size_log2(fp_block_size);
    if width_mi_log2 > MB_MI_LOG2 {
        mb_cols >> (width_mi_log2 - MB_MI_LOG2)
    } else {
        mb_cols << (MB_MI_LOG2 - width_mi_log2)
    }
}

/// `get_num_mbs` (firstpass.c:174, static) — rescale a 16x16-macroblock
/// *count* (an area) to first-pass block units.
///
/// The shift is the sum of both axes' shifts, because this is an area, where
/// [`get_unit_rows`] and [`get_unit_cols`] each shift one axis. C asserts the
/// block is square and this reproduces that assert rather than the
/// rectangular behaviour, which upstream flags as unsupported.
///
/// # Panics
/// If `fp_block_size` is not square. C's `assert` is compiled out in the
/// Release oracle, where a rectangular block silently takes the branch chosen
/// by the *width* alone and shifts by the sum — so this panics rather than
/// reproducing an arm upstream calls unsupported.
#[must_use]
pub fn get_num_mbs(fp_block_size: i32, num_mbs_16x16: i32) -> i32 {
    let (width_mi_log2, height_mi_log2) = mi_size_log2(fp_block_size);
    assert_eq!(
        width_mi_log2, height_mi_log2,
        "get_num_mbs assumes a square first-pass block (firstpass.c:180)"
    );
    if width_mi_log2 > MB_MI_LOG2 {
        num_mbs_16x16 >> ((width_mi_log2 - MB_MI_LOG2) + (height_mi_log2 - MB_MI_LOG2))
    } else {
        num_mbs_16x16 << ((MB_MI_LOG2 - width_mi_log2) + (MB_MI_LOG2 - height_mi_log2))
    }
}

/// `av1_get_unit_rows_in_tile` (firstpass.c:1105) — first-pass block rows in
/// a tile.
///
/// `CEIL_POWER_OF_TWO(mi_rows, unit_height_log2)`, i.e. rounded UP — a tile
/// whose mi height is not a multiple of the block height still gets a partial
/// last row of units.
#[must_use]
pub fn get_unit_rows_in_tile(mi_row_start: i32, mi_row_end: i32, fp_block_size: i32) -> i32 {
    let (_, unit_height_log2) = mi_size_log2(fp_block_size);
    let mi_rows = mi_row_end - mi_row_start;
    (mi_rows + (1 << unit_height_log2) - 1) >> unit_height_log2
}

/// `av1_get_unit_cols_in_tile` (firstpass.c:1114) — the column counterpart.
#[must_use]
pub fn get_unit_cols_in_tile(mi_col_start: i32, mi_col_end: i32, fp_block_size: i32) -> i32 {
    let (unit_width_log2, _) = mi_size_log2(fp_block_size);
    let mi_cols = mi_col_end - mi_col_start;
    (mi_cols + (1 << unit_width_log2) - 1) >> unit_width_log2
}

/// `get_search_range` (firstpass.c:257, static) — how many motion-search step
/// levels the frame size supports.
///
/// The smaller frame dimension is floored at one mi (4 pixels) and then
/// doubled until it reaches [`MAX_FULL_PEL_VAL`]; the number of doublings is
/// the search range. So a small frame gets a *larger* range — the search is
/// bounded by the frame, and a tiny frame needs more halvings to get there.
#[must_use]
pub fn get_search_range(width: i32, height: i32) -> i32 {
    let mut sr = 0;
    let dim = width.min(height).max(MI_SIZE);
    while (dim << sr) < MAX_FULL_PEL_VAL {
        sr += 1;
    }
    sr
}

/// `find_fp_qindex` (firstpass.c:368, static) — the qindex the first pass
/// analyses at, for a given bit depth.
///
/// A [`FIRST_PASS_Q`] lookup through `av1_find_qindex` over the whole qindex
/// range. It is bit-depth dependent because the Q-to-qindex table is.
#[must_use]
pub fn find_fp_qindex(bit_depth: u8) -> i32 {
    crate::rate_model::find_qindex(FIRST_PASS_Q, bit_depth, 0, QINDEX_RANGE - 1)
}

/// `raw_motion_error_stdev` (firstpass.c:372, static) — the standard
/// deviation of the zero-MV prediction error across a frame's blocks.
///
/// Pass 2 uses this to tell a real scene cut from a flash: a frame that
/// predicts badly *everywhere* is a cut, one that predicts badly in patches
/// is not.
///
/// Returns 0 for an empty list (C's early exit — not NaN from a 0/0). The sum
/// is accumulated in `i64` and only then converted, so a long list of large
/// errors does not lose precision the way an `f64` running sum would; the
/// deviation sum, by contrast, is `f64` from the start, exactly as C has it.
#[must_use]
pub fn raw_motion_error_stdev(raw_motion_err_list: &[i32]) -> f64 {
    let count = raw_motion_err_list.len();
    if count == 0 {
        return 0.0;
    }
    let sum_raw_err: i64 = raw_motion_err_list.iter().map(|&e| i64::from(e)).sum();
    let raw_err_avg = sum_raw_err as f64 / count as f64;
    let mut raw_err_stdev = 0.0f64;
    for &e in raw_motion_err_list {
        let d = f64::from(e) - raw_err_avg;
        raw_err_stdev += d * d;
    }
    (raw_err_stdev / count as f64).sqrt()
}

/// `calc_wavelet_energy` (firstpass.c:395, static) — whether the first pass
/// computes the DWT energy term at all.
///
/// Only `DELTA_Q_PERCEPTUAL` needs it; every other delta-q mode leaves
/// `frame_avg_wavelet_energy` at its invalid sentinel (see
/// `is_fp_wavelet_energy_invalid`, firstpass.h:548, which tests `< 0`).
#[must_use]
pub fn calc_wavelet_energy(deltaq_mode: i32) -> bool {
    /// `DELTA_Q_PERCEPTUAL` (`encoder/enc_enums.h`).
    const DELTA_Q_PERCEPTUAL: i32 = 2;
    deltaq_mode == DELTA_Q_PERCEPTUAL
}

/// `FIRSTPASS_INFO_STATS_PAST_MIN` (firstpass.h:183) — how many already-coded
/// frames the pass-2 window keeps behind the cursor.
pub const FIRSTPASS_INFO_STATS_PAST_MIN: usize = 1;

/// `MAX_LAP_BUFFERS` (`encoder/lookahead.h`).
pub const MAX_LAP_BUFFERS: usize = 48;

/// `FIRSTPASS_INFO_STATIC_BUF_SIZE` (firstpass.h:185).
pub const FIRSTPASS_INFO_STATIC_BUF_SIZE: usize = MAX_LAP_BUFFERS + FIRSTPASS_INFO_STATS_PAST_MIN;

/// `aom_codec_err_t`'s two values this module produces: `AOM_CODEC_OK` and
/// `AOM_CODEC_ERROR`.
///
/// C returns the bare enum, and every caller compares it against
/// `AOM_CODEC_OK`; that is a `Result` with no payload, so the ring-buffer
/// entry points below return `Result<(), FirstpassInfoError>` and the
/// differential maps it back to C's integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirstpassInfoError {
    /// `AOM_CODEC_ERROR` — the operation could not be performed: the buffer
    /// is full, empty, or the cursor is already at the newest entry.
    Failed,
}

/// `FIRSTPASS_INFO` (firstpass.h:187) — the sliding window of first-pass
/// records pass 2 walks.
///
/// A circular buffer with **three** cursors, which is what makes it more than
/// a `VecDeque`: `start_index` is the oldest retained record, `cur_index` is
/// the frame pass 2 is currently deciding for, and the two counts split the
/// window into what is behind the cursor (`past_stats_count`, kept so
/// pass 2 can look back) and what is ahead (`future_stats_count`, the
/// lookahead). Popping and advancing the cursor are separate operations
/// precisely so the past window can be trimmed independently of progress.
///
/// C's `stats_buf` either points at the inline `static_stats_buf` or at a
/// caller-provided array; that pointer is allocation bookkeeping, so this
/// holds one `Vec` and records which case it was constructed for.
#[derive(Clone, Debug)]
pub struct FirstpassInfo {
    /// The ring itself. Length is C's `stats_buf_size`.
    stats_buf: Vec<FirstpassStats>,
    /// Index of the oldest retained record.
    start_index: usize,
    /// Number of records currently retained.
    stats_count: usize,
    /// Index of the record pass 2 is deciding for.
    cur_index: usize,
    /// Records at or after `cur_index`.
    future_stats_count: usize,
    /// Records before `cur_index`.
    past_stats_count: usize,
    /// Running accumulation of every record ever pushed.
    total_stats: FirstpassStats,
}

impl FirstpassInfo {
    /// `av1_firstpass_info_init` (firstpass.c:1544) — the internal-buffer
    /// form, `ext_stats_buf == NULL`.
    ///
    /// The window starts empty; records arrive through [`Self::push`].
    #[must_use]
    pub fn new_internal() -> Self {
        Self {
            stats_buf: vec![FirstpassStats::default(); FIRSTPASS_INFO_STATIC_BUF_SIZE],
            start_index: 0,
            stats_count: 0,
            cur_index: 0,
            future_stats_count: 0,
            past_stats_count: 0,
            total_stats: FirstpassStats::default(),
        }
    }

    /// `av1_firstpass_info_init` (firstpass.c:1544) — the external-buffer
    /// form, where the caller hands over an already-populated stats array.
    ///
    /// The whole array counts as *future*: `stats_count` and
    /// `future_stats_count` are both the buffer size, and `total_stats` is
    /// the accumulation of every entry. This is the `--pass=2` entry point,
    /// where the first pass's output file has already been read in.
    #[must_use]
    pub fn new_external(ext_stats_buf: Vec<FirstpassStats>) -> Self {
        let mut total_stats = FirstpassStats::default();
        for s in &ext_stats_buf {
            total_stats.accumulate(s);
        }
        Self {
            start_index: 0,
            cur_index: 0,
            stats_count: ext_stats_buf.len(),
            future_stats_count: ext_stats_buf.len(),
            past_stats_count: 0,
            stats_buf: ext_stats_buf,
            total_stats,
        }
    }

    /// The running accumulation (`FIRSTPASS_INFO::total_stats`).
    #[must_use]
    pub fn total_stats(&self) -> &FirstpassStats {
        &self.total_stats
    }

    /// `FIRSTPASS_INFO::stats_count`.
    #[must_use]
    pub fn stats_count(&self) -> usize {
        self.stats_count
    }

    /// `FIRSTPASS_INFO::future_stats_count`.
    #[must_use]
    pub fn future_stats_count(&self) -> usize {
        self.future_stats_count
    }

    /// `FIRSTPASS_INFO::past_stats_count`.
    #[must_use]
    pub fn past_stats_count(&self) -> usize {
        self.past_stats_count
    }

    /// `FIRSTPASS_INFO::cur_index`.
    #[must_use]
    pub fn cur_index(&self) -> usize {
        self.cur_index
    }

    /// `FIRSTPASS_INFO::start_index`.
    #[must_use]
    pub fn start_index(&self) -> usize {
        self.start_index
    }

    /// `av1_firstpass_info_move_cur_index` (firstpass.c:1581) — advance the
    /// cursor one frame.
    ///
    /// **Fails when only one future record remains**, not when zero do: the
    /// test is `future_stats_count > 1`, so the cursor never leaves the
    /// window. Pass 2 relies on that — `peek(0)` is always valid.
    pub fn move_cur_index(&mut self) -> Result<(), FirstpassInfoError> {
        debug_assert_eq!(
            self.future_stats_count + self.past_stats_count,
            self.stats_count
        );
        if self.future_stats_count > 1 {
            self.cur_index = (self.cur_index + 1) % self.stats_buf.len();
            self.future_stats_count -= 1;
            self.past_stats_count += 1;
            Ok(())
        } else {
            Err(FirstpassInfoError::Failed)
        }
    }

    /// `av1_firstpass_info_pop` (firstpass.c:1597) — drop the oldest record.
    ///
    /// Only *past* records can be popped, so the window never loses a frame
    /// pass 2 has not decided yet. Note `total_stats` is NOT decremented —
    /// it is the accumulation of everything ever pushed, not of what is
    /// currently retained.
    pub fn pop(&mut self) -> Result<(), FirstpassInfoError> {
        if self.stats_count > 0 && self.past_stats_count > 0 {
            self.start_index = (self.start_index + 1) % self.stats_buf.len();
            self.stats_count -= 1;
            self.past_stats_count -= 1;
            Ok(())
        } else {
            Err(FirstpassInfoError::Failed)
        }
    }

    /// `av1_firstpass_info_move_cur_index_and_pop` (firstpass.c:1610).
    ///
    /// The two in sequence, aborting after the first failure — so a failed
    /// advance leaves the window untouched, but a successful advance followed
    /// by a failed pop does not roll the advance back. That asymmetry is C's.
    pub fn move_cur_index_and_pop(&mut self) -> Result<(), FirstpassInfoError> {
        self.move_cur_index()?;
        self.pop()
    }

    /// `av1_firstpass_info_push` (firstpass.c:1618) — append one record.
    ///
    /// Fails when the ring is full rather than overwriting. The new record
    /// lands `stats_count` slots after `start_index`, so it is always at the
    /// future end, and `total_stats` accumulates it immediately.
    pub fn push(&mut self, input_stats: &FirstpassStats) -> Result<(), FirstpassInfoError> {
        if self.stats_count < self.stats_buf.len() {
            let next_index = (self.start_index + self.stats_count) % self.stats_buf.len();
            self.stats_buf[next_index] = *input_stats;
            self.stats_count += 1;
            self.future_stats_count += 1;
            self.total_stats.accumulate(input_stats);
            Ok(())
        } else {
            Err(FirstpassInfoError::Failed)
        }
    }

    /// `av1_firstpass_info_peek` (firstpass.c:1634) — the record
    /// `offset_from_cur` frames from the cursor, or `None` outside the
    /// window.
    ///
    /// The window is `[-past_stats_count, future_stats_count)` — asymmetric,
    /// because the cursor sits ON the first future record.
    ///
    /// # Divergence from C, deliberate
    /// C computes `(cur_index + offset_from_cur) % stats_buf_size`. In C, `%`
    /// on a negative left operand yields a negative result, so once the ring
    /// has wrapped (`cur_index < start_index`) a legal negative offset can
    /// produce a **negative index**, and C reads `stats_buf[-k]` — out of
    /// bounds, undefined behaviour. This uses `rem_euclid`, which returns the
    /// slot C's arithmetic was clearly reaching for. The two agree wherever
    /// C is defined; the differential is bounded to that region and says so.
    #[must_use]
    pub fn peek(&self, offset_from_cur: i32) -> Option<&FirstpassStats> {
        let past = i32::try_from(self.past_stats_count).ok()?;
        let future = i32::try_from(self.future_stats_count).ok()?;
        if offset_from_cur >= -past && offset_from_cur < future {
            let size = i32::try_from(self.stats_buf.len()).ok()?;
            let index = (i32::try_from(self.cur_index).ok()? + offset_from_cur).rem_euclid(size);
            self.stats_buf.get(index as usize)
        } else {
            None
        }
    }

    /// `av1_firstpass_info_future_count` (firstpass.c:1646) — how many
    /// records remain at or after `cur_index + offset_from_cur`.
    ///
    /// Returns 0 past the end rather than a negative count, and — note — does
    /// **not** clamp a negative offset: `future_count(-3)` is
    /// `future_stats_count + 3`, counting records that are behind the cursor
    /// as if they were ahead of it. Every caller passes a non-negative
    /// offset, so this is reproduced rather than corrected.
    #[must_use]
    pub fn future_count(&self, offset_from_cur: i32) -> i32 {
        let future = i32::try_from(self.future_stats_count).unwrap_or(i32::MAX);
        if offset_from_cur < future {
            future - offset_from_cur
        } else {
            0
        }
    }
}

// ===========================================================================
// The per-block helpers: block-size selection and the prediction-error MSE.
// ===========================================================================

/// `get_bsize` (firstpass.c:335, static) — the block size to analyse the
/// first-pass unit at `(unit_row, unit_col)` with.
///
/// A unit whose second half falls outside the frame is analysed at half size
/// on that axis, so an edge unit measures only real pixels instead of
/// averaging in the replicated border. Both halves out of frame gives the
/// quarter-size `PARTITION_SPLIT` subsize.
///
/// The test is `unit_width * unit_col + unit_width / 2 >= mi_cols`, i.e. the
/// unit is "half width" when its **midpoint** is outside — not when its right
/// edge is. A unit straddling the boundary by one mi is still analysed full
/// width.
///
/// C derives the square-size index from `AOMMAX(block_size_wide,
/// block_size_high)`, which for a rectangular `fp_block_size` is NOT
/// `get_sqr_bsize_idx` — the shared
/// [`aom_dsp::entropy::partition::get_partition_subsize`] would return
/// `BLOCK_INVALID` there. It cannot happen (`get_fp_block_size`,
/// firstpass.h:554, returns only `BLOCK_8X8` or `BLOCK_16X16`), but the
/// mapping is written C's way rather than delegated, so the two do not
/// silently disagree if a rectangular size ever arrives.
///
/// # Panics
/// If `fp_block_size`'s larger dimension is not one of 4/8/16/32/64/128 —
/// C's `default:` arm asserts and then falls through with
/// `square_block_size` left at 0, which would index the wrong table row.
#[must_use]
pub fn get_bsize(
    mi_rows: i32,
    mi_cols: i32,
    fp_block_size: i32,
    unit_row: i32,
    unit_col: i32,
) -> i32 {
    let (w_log2, h_log2) = mi_size_log2(fp_block_size);
    let unit_width = 1i32 << w_log2;
    let unit_height = 1i32 << h_log2;
    let is_half_width = unit_width * unit_col + unit_width / 2 >= mi_cols;
    let is_half_height = unit_height * unit_row + unit_height / 2 >= mi_rows;

    // `block_size_wide` / `block_size_high` are the mi dimensions times
    // MI_SIZE, so the max dimension is `4 << max(w_log2, h_log2)`.
    let max_dimension = MI_SIZE << w_log2.max(h_log2);
    let square_block_size = match max_dimension {
        4 => 0,
        8 => 1,
        16 => 2,
        32 => 3,
        64 => 4,
        128 => 5,
        other => panic!("first pass block size {other} is not supported (firstpass.c:356)"),
    };
    // PARTITION_HORZ = 1, PARTITION_VERT = 2, PARTITION_SPLIT = 3.
    // The square-size index is what `get_partition_subsize` derives
    // internally, so it is fed the square BLOCK_SIZE of that index rather
    // than `fp_block_size` itself.
    const SQUARE_BSIZE: [i32; 6] = [0, 3, 6, 9, 12, 15];
    let sq = SQUARE_BSIZE[square_block_size] as usize;
    if is_half_width && is_half_height {
        aom_dsp::entropy::partition::get_partition_subsize(sq, 3)
    } else if is_half_width {
        aom_dsp::entropy::partition::get_partition_subsize(sq, 2)
    } else if is_half_height {
        aom_dsp::entropy::partition::get_partition_subsize(sq, 1)
    } else {
        fp_block_size
    }
}

/// The `(width, height)` the MSE kernel for `bsize` measures over —
/// `get_block_variance_fn` (firstpass.c:198, static) resolved to dimensions
/// instead of a function pointer.
///
/// C returns one of four `aom_mseWxH` kernels and **falls through to 16x16
/// for every size other than 8x8 / 16x8 / 8x16**, including sizes larger than
/// 16x16. That default is not a safety net: `get_bsize` can only produce
/// those four for a `BLOCK_16X16` first pass, but if it ever produced
/// `BLOCK_4X4` the measurement would silently cover 16x16 pixels. Reproduced
/// as written.
#[must_use]
fn block_variance_dims(bsize: i32) -> (usize, usize) {
    match bsize {
        3 => (8, 8),   // BLOCK_8X8
        5 => (16, 8),  // BLOCK_16X8
        4 => (8, 16),  // BLOCK_8X16
        _ => (16, 16), // C's `default: return aom_mse16x16`
    }
}

/// `get_prediction_error` (firstpass.c:207, static) — the sum of squared
/// error between a source block and its prediction, lowbd.
///
/// C's `aom_mseWxH` returns the `sse` its `variance()` computed; the mean is
/// never taken despite the name. `src`/`ref` are `buf_2d`s, i.e. a pointer
/// plus a stride, so they are slices plus strides here.
#[must_use]
pub fn get_prediction_error(
    bsize: i32,
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
) -> u32 {
    let (w, h) = block_variance_dims(bsize);
    aom_dsp::dist::variance(src, src_stride, reference, ref_stride, w, h).1
}

/// `highbd_get_prediction_error` (firstpass.c:244, static) — the highbd
/// counterpart.
///
/// The bit depth selects between three kernel families that differ by more
/// than the input width: `aom_highbd_10_mse*` rounds `sse` down by 4 bits and
/// `aom_highbd_12_mse*` by 8, so the returned error is in 8-bit units at
/// every depth. A port that only widened the pixels would be off by 16x or
/// 256x.
///
/// C's `switch (bd)` has `default:` on the **8-bit** arm, so any depth other
/// than 10 or 12 takes it — including nonsense values. Reproduced.
///
/// # Contract: a bd-8 highbd plane holds 8-bit samples
/// MEASURED on this build: with 16-bit samples in `0..=1023` at `bd == 8`,
/// C's `aom_highbd_8_mse16x16` returns 2_741_760 where the scalar definition
/// (and this function) give 42_944_000 — its kernel accumulates in a width
/// that assumes 8-bit samples. At `bd == 10` and `bd == 12` the same inputs
/// agree exactly. The encoder cannot reach the divergent input (a highbd
/// plane at bit depth 8 holds 8-bit values), so this is a contract on the
/// caller, not a divergence to reconcile — see
/// `highbd_get_prediction_error_matches_c`, which bounds its sweep by it and
/// says so.
#[must_use]
pub fn highbd_get_prediction_error(
    bsize: i32,
    src: &[u16],
    src_stride: usize,
    reference: &[u16],
    ref_stride: usize,
    bd: u8,
) -> u32 {
    let (w, h) = block_variance_dims(bsize);
    let bd = if bd == 10 || bd == 12 { bd } else { 8 };
    aom_dsp::dist::highbd_variance(src, src_stride, reference, ref_stride, w, h, bd).1
}

/// `get_prediction_error_bitdepth` (firstpass.c:618, static) — the dispatch
/// between the two above.
///
/// `is_high_bitdepth` and `bitdepth` are separate arguments in C and this
/// keeps them separate: a highbd *buffer* at bit depth 8 is a real
/// configuration (`aom_highbd_8_mse*` exists precisely for it), so the flag
/// is not derivable from the depth.
#[must_use]
pub fn get_prediction_error_bitdepth(
    bitdepth: u8,
    bsize: i32,
    src: HighbdOrLowbd<'_>,
    reference: HighbdOrLowbd<'_>,
    src_stride: usize,
    ref_stride: usize,
) -> u32 {
    match (src, reference) {
        (HighbdOrLowbd::Highbd(s), HighbdOrLowbd::Highbd(r)) => {
            highbd_get_prediction_error(bsize, s, src_stride, r, ref_stride, bitdepth)
        }
        (HighbdOrLowbd::Lowbd(s), HighbdOrLowbd::Lowbd(r)) => {
            get_prediction_error(bsize, s, src_stride, r, ref_stride)
        }
        _ => panic!("source and reference must have the same pixel width"),
    }
}

/// A `buf_2d`'s pixels at one of the two widths the encoder uses. C carries
/// this as a `uint8_t *` plus a flag and casts; the pair is an enum here so
/// the mismatched case is a compile-time-visible arm rather than a silent
/// reinterpretation.
#[derive(Clone, Copy, Debug)]
pub enum HighbdOrLowbd<'a> {
    /// An 8-bit plane.
    Lowbd(&'a [u8]),
    /// A 10/12-bit plane (also used at bit depth 8 in a highbd build).
    Highbd(&'a [u16]),
}
