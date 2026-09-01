/* Oracle shims for the FILE-STATIC error/boost model of
 * av1/encoder/pass2_strategy.c -- the 2-pass bit allocator.
 *
 * WHY THIS FILE PULLS IN A libaom .c
 * ----------------------------------
 * `nm -g` on the object file reports SEVEN exported symbols for a 4781-line
 * file (av1_calc_arf_boost, av1_get_second_pass_params, av1_gop_bit_allocation,
 * av1_init_second_pass, av1_init_single_pass_lap, av1_setup_target_rate,
 * av1_twopass_postencode_update). Every piece of arithmetic they are built out
 * of -- the modified-error model, the decay rates, the frame-boost curves, the
 * boost-bit split -- is static.
 *
 * EVIDENCE TIER 1c -- the real C source compiled verbatim, with its seven
 * exported symbols renamed out of the way. Same technique and same
 * justification as shim/rdopt_shim.c; read that file's header. The
 * second-compilation gap is measured by `shim_p2_tu_qbpm_enumerator` and
 * `shim_p2_firstpass_stats_size`, checked in tests/pass2_model_diff.rs.
 *
 * FLAGS. build.rs compiles this TU with libaom's own Release flags
 * (`-O3 -DNDEBUG`, plus the oracle-wide `-ffp-contract=off`). The FP flag is
 * load-bearing here in a way it is not for the integer shims: this file is
 * almost entirely `double` arithmetic, and without it clang would contract
 * multiply-accumulates on aarch64 and not on x86, so the oracle would mean two
 * different things per host.
 *
 * FIRSTPASS_STATS CROSSES AS A FLAT ARRAY of its 29 `double` members in
 * declaration order, plus `is_flash` as a separate int64_t. The shim assigns
 * them by NAME below, so the mapping is explicit and does not depend on the
 * struct's layout matching any Rust mirror.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

/* --- Rename pass2_strategy.c's seven exported symbols. --- */
#define av1_calc_arf_boost shim_p2_calc_arf_boost
#define av1_get_second_pass_params shim_p2_get_second_pass_params
#define av1_gop_bit_allocation shim_p2_gop_bit_allocation
#define av1_init_second_pass shim_p2_init_second_pass
#define av1_init_single_pass_lap shim_p2_init_single_pass_lap
#define av1_setup_target_rate shim_p2_setup_target_rate
#define av1_twopass_postencode_update shim_p2_twopass_postencode_update

/* --- libaom's own 2-pass strategy, unmodified. --- */
#include "av1/encoder/pass2_strategy.c"

/* Field order of the flat array the Rust side sends. Kept in ONE place. */
static void shim_p2_fill_stats(FIRSTPASS_STATS *s, const double *f,
                               int64_t is_flash) {
  memset(s, 0, sizeof(*s));
  s->frame = f[0];
  s->weight = f[1];
  s->intra_error = f[2];
  s->frame_avg_wavelet_energy = f[3];
  s->coded_error = f[4];
  s->sr_coded_error = f[5];
  s->lt_coded_error = f[6];
  s->pcnt_inter = f[7];
  s->pcnt_motion = f[8];
  s->pcnt_second_ref = f[9];
  s->pcnt_neutral = f[10];
  s->intra_skip_pct = f[11];
  s->inactive_zone_rows = f[12];
  s->inactive_zone_cols = f[13];
  s->MVr = f[14];
  s->mvr_abs = f[15];
  s->MVc = f[16];
  s->mvc_abs = f[17];
  s->MVrv = f[18];
  s->MVcv = f[19];
  s->mv_in_out_count = f[20];
  s->new_mv_count = f[21];
  s->duration = f[22];
  s->count = f[23];
  s->raw_error_stdev = f[24];
  s->noise_var = f[25];
  s->cor_coeff = f[26];
  s->log_intra_error = f[27];
  s->log_coded_error = f[28];
  s->is_flash = is_flash;
}

int shim_p2_firstpass_stats_doubles(void) { return 29; }
size_t shim_p2_firstpass_stats_size(void) { return sizeof(FIRSTPASS_STATS); }

static void shim_p2_fill_frame_info(FRAME_INFO *fi, int frame_width,
                                    int frame_height, int mb_rows, int mb_cols,
                                    int num_mbs, int bit_depth) {
  memset(fi, 0, sizeof(*fi));
  fi->frame_width = frame_width;
  fi->frame_height = frame_height;
  fi->mb_rows = mb_rows;
  fi->mb_cols = mb_cols;
  fi->num_mbs = num_mbs;
  fi->bit_depth = (aom_bit_depth_t)bit_depth;
}

double shim_p2_calculate_active_area(int frame_width, int frame_height,
                                     int mb_rows, int mb_cols, int num_mbs,
                                     int bit_depth, const double *stats,
                                     int64_t is_flash) {
  FRAME_INFO fi;
  FIRSTPASS_STATS s;
  shim_p2_fill_frame_info(&fi, frame_width, frame_height, mb_rows, mb_cols,
                          num_mbs, bit_depth);
  shim_p2_fill_stats(&s, stats, is_flash);
  return calculate_active_area(&fi, &s);
}

double shim_p2_calculate_modified_err_new(int frame_width, int frame_height,
                                          int mb_rows, int mb_cols,
                                          int num_mbs, int bit_depth,
                                          int have_total,
                                          const double *total_stats,
                                          const double *this_stats, int vbrbias,
                                          double err_min, double err_max) {
  FRAME_INFO fi;
  FIRSTPASS_STATS total, s;
  shim_p2_fill_frame_info(&fi, frame_width, frame_height, mb_rows, mb_cols,
                          num_mbs, bit_depth);
  shim_p2_fill_stats(&total, total_stats, 0);
  shim_p2_fill_stats(&s, this_stats, 0);
  return calculate_modified_err_new(&fi, have_total ? &total : NULL, &s,
                                    vbrbias, err_min, err_max);
}

int shim_p2_frame_max_bits(int64_t avg_frame_bandwidth,
                           int64_t max_frame_bandwidth, int vbrmax_section) {
  RATE_CONTROL *rc = (RATE_CONTROL *)calloc(1, sizeof(*rc));
  AV1EncoderConfig *oxcf = (AV1EncoderConfig *)calloc(1, sizeof(*oxcf));
  if (!rc || !oxcf) {
    free(rc);
    free(oxcf);
    return -1;
  }
  rc->avg_frame_bandwidth = avg_frame_bandwidth;
  rc->max_frame_bandwidth = max_frame_bandwidth;
  oxcf->rc_cfg.vbrmax_section = vbrmax_section;
  const int r = frame_max_bits(rc, oxcf);
  free(rc);
  free(oxcf);
  return r;
}

double shim_p2_calc_correction_factor(double err_per_mb, int q) {
  return calc_correction_factor(err_per_mb, q);
}

int shim_p2_qbpm_enumerator(int rate_err_tol) {
  return qbpm_enumerator(rate_err_tol);
}

int shim_p2_tu_qbpm_enumerator(int rate_err_tol) {
  /* Same function, reached through the TU's own copy -- see the header. */
  return qbpm_enumerator(rate_err_tol);
}

double shim_p2_get_sr_decay_rate(const double *stats) {
  FIRSTPASS_STATS s;
  shim_p2_fill_stats(&s, stats, 0);
  return get_sr_decay_rate(&s);
}

double shim_p2_get_zero_motion_factor(const double *stats) {
  FIRSTPASS_STATS s;
  shim_p2_fill_stats(&s, stats, 0);
  return get_zero_motion_factor(&s);
}

double shim_p2_get_prediction_decay_rate(const double *stats) {
  FIRSTPASS_STATS s;
  shim_p2_fill_stats(&s, stats, 0);
  return get_prediction_decay_rate(&s);
}

double shim_p2_baseline_err_per_mb(int frame_width, int frame_height) {
  FRAME_INFO fi;
  shim_p2_fill_frame_info(&fi, frame_width, frame_height, 0, 0, 0, 8);
  return baseline_err_per_mb(&fi);
}

double shim_p2_calc_frame_boost(int avg_frame_qindex_inter, int frame_width,
                                int frame_height, int mb_rows, int mb_cols,
                                int num_mbs, int bit_depth,
                                const double *stats, double this_frame_mv_in_out,
                                double max_boost, int scale_max_boost) {
  PRIMARY_RATE_CONTROL *p_rc =
      (PRIMARY_RATE_CONTROL *)calloc(1, sizeof(*p_rc));
  if (!p_rc) return 0.0;
  FRAME_INFO fi;
  FIRSTPASS_STATS s;
  p_rc->avg_frame_qindex[INTER_FRAME] = avg_frame_qindex_inter;
  shim_p2_fill_frame_info(&fi, frame_width, frame_height, mb_rows, mb_cols,
                          num_mbs, bit_depth);
  shim_p2_fill_stats(&s, stats, 0);
  const double r = calc_frame_boost(p_rc, &fi, &s, this_frame_mv_in_out,
                                    max_boost, scale_max_boost != 0);
  free(p_rc);
  return r;
}

double shim_p2_calc_kf_frame_boost(int avg_frame_qindex_inter, int frame_width,
                                   int frame_height, int mb_rows, int mb_cols,
                                   int num_mbs, int bit_depth,
                                   const double *stats, double *sr_accumulator,
                                   double max_boost) {
  PRIMARY_RATE_CONTROL *p_rc =
      (PRIMARY_RATE_CONTROL *)calloc(1, sizeof(*p_rc));
  if (!p_rc) return 0.0;
  FRAME_INFO fi;
  FIRSTPASS_STATS s;
  p_rc->avg_frame_qindex[INTER_FRAME] = avg_frame_qindex_inter;
  shim_p2_fill_frame_info(&fi, frame_width, frame_height, mb_rows, mb_cols,
                          num_mbs, bit_depth);
  shim_p2_fill_stats(&s, stats, 0);
  const double r = calc_kf_frame_boost(p_rc, &fi, &s, sr_accumulator, max_boost);
  free(p_rc);
  return r;
}

int shim_p2_calculate_boost_bits(int frame_count, int boost,
                                 int64_t total_group_bits) {
  return calculate_boost_bits(frame_count, boost, total_group_bits);
}

int shim_p2_calculate_boost_factor(int frame_count, int bits,
                                   int64_t total_group_bits) {
  return calculate_boost_factor(frame_count, bits, total_group_bits);
}

int shim_p2_get_projected_gfu_boost(int baseline_gf_interval, int gfu_boost,
                                    int frames_to_project,
                                    int num_stats_used_for_gfu_boost) {
  PRIMARY_RATE_CONTROL *p_rc =
      (PRIMARY_RATE_CONTROL *)calloc(1, sizeof(*p_rc));
  if (!p_rc) return -1;
  p_rc->baseline_gf_interval = baseline_gf_interval;
  const int r = get_projected_gfu_boost(p_rc, gfu_boost, frames_to_project,
                                        num_stats_used_for_gfu_boost);
  free(p_rc);
  return r;
}

int shim_p2_is_almost_static(double gf_zero_motion, int kf_zero_motion,
                             int is_lap_enabled) {
  return is_almost_static(gf_zero_motion, kf_zero_motion, is_lap_enabled);
}

/* ======================================================================== *
 * The GF_GROUP_STATS accumulator cluster.
 *
 * GF_GROUP_STATS crosses as its 17 `double` members in declaration order plus
 * `non_zero_stdev_count` as an int, assigned by NAME below for the same
 * reason FIRSTPASS_STATS is.
 * ======================================================================== */
static void shim_p2_load_gf(GF_GROUP_STATS *g, const double *d, int nz_count) {
  memset(g, 0, sizeof(*g));
  g->gf_group_err = d[0];
  g->gf_group_raw_error = d[1];
  g->gf_group_skip_pct = d[2];
  g->gf_group_inactive_zone_rows = d[3];
  g->mv_ratio_accumulator = d[4];
  g->decay_accumulator = d[5];
  g->zero_motion_accumulator = d[6];
  g->loop_decay_rate = d[7];
  g->last_loop_decay_rate = d[8];
  g->this_frame_mv_in_out = d[9];
  g->mv_in_out_accumulator = d[10];
  g->abs_mv_in_out_accumulator = d[11];
  g->avg_sr_coded_error = d[12];
  g->avg_pcnt_second_ref = d[13];
  g->avg_new_mv_count = d[14];
  g->avg_wavelet_energy = d[15];
  g->avg_raw_err_stdev = d[16];
  g->non_zero_stdev_count = nz_count;
}

static void shim_p2_store_gf(const GF_GROUP_STATS *g, double *d,
                             int *nz_count) {
  d[0] = g->gf_group_err;
  d[1] = g->gf_group_raw_error;
  d[2] = g->gf_group_skip_pct;
  d[3] = g->gf_group_inactive_zone_rows;
  d[4] = g->mv_ratio_accumulator;
  d[5] = g->decay_accumulator;
  d[6] = g->zero_motion_accumulator;
  d[7] = g->loop_decay_rate;
  d[8] = g->last_loop_decay_rate;
  d[9] = g->this_frame_mv_in_out;
  d[10] = g->mv_in_out_accumulator;
  d[11] = g->abs_mv_in_out_accumulator;
  d[12] = g->avg_sr_coded_error;
  d[13] = g->avg_pcnt_second_ref;
  d[14] = g->avg_new_mv_count;
  d[15] = g->avg_wavelet_energy;
  d[16] = g->avg_raw_err_stdev;
  *nz_count = g->non_zero_stdev_count;
}

int shim_p2_gf_group_stats_doubles(void) { return 17; }

void shim_p2_init_gf_stats(double *out, int *nz_count) {
  GF_GROUP_STATS g;
  init_gf_stats(&g);
  shim_p2_store_gf(&g, out, nz_count);
}

void shim_p2_accumulate_frame_motion_stats(const double *stats, double *gf,
                                           int *nz_count, double f_w,
                                           double f_h) {
  FIRSTPASS_STATS s;
  GF_GROUP_STATS g;
  shim_p2_fill_stats(&s, stats, 0);
  shim_p2_load_gf(&g, gf, *nz_count);
  accumulate_frame_motion_stats(&s, &g, f_w, f_h);
  shim_p2_store_gf(&g, gf, nz_count);
}

void shim_p2_accumulate_this_frame_stats(const double *stats,
                                         double mod_frame_err, double *gf,
                                         int *nz_count) {
  FIRSTPASS_STATS s;
  GF_GROUP_STATS g;
  shim_p2_fill_stats(&s, stats, 0);
  shim_p2_load_gf(&g, gf, *nz_count);
  accumulate_this_frame_stats(&s, mod_frame_err, &g);
  shim_p2_store_gf(&g, gf, nz_count);
}

void shim_p2_accumulate_next_frame_stats(const double *stats,
                                         int flash_detected,
                                         int frames_since_key, int cur_idx,
                                         double *gf, int *nz_count, int f_w,
                                         int f_h) {
  FIRSTPASS_STATS s;
  GF_GROUP_STATS g;
  shim_p2_fill_stats(&s, stats, 0);
  shim_p2_load_gf(&g, gf, *nz_count);
  accumulate_next_frame_stats(&s, flash_detected, frames_since_key, cur_idx, &g,
                             f_w, f_h);
  shim_p2_store_gf(&g, gf, nz_count);
}

void shim_p2_average_gf_stats(int total_frame, double *gf, int *nz_count) {
  GF_GROUP_STATS g;
  shim_p2_load_gf(&g, gf, *nz_count);
  average_gf_stats(total_frame, &g);
  shim_p2_store_gf(&g, gf, nz_count);
}

/* calculate_section_intra_ratio walks a run of FIRSTPASS_STATS; the shim
 * rebuilds that run from `count` flat records. */
int shim_p2_calculate_section_intra_ratio(const double *stats_flat, int count,
                                          int section_length) {
  if (count <= 0) return calculate_section_intra_ratio(NULL, NULL, section_length);
  FIRSTPASS_STATS *arr =
      (FIRSTPASS_STATS *)calloc((size_t)count, sizeof(FIRSTPASS_STATS));
  if (!arr) return -1;
  for (int i = 0; i < count; ++i) shim_p2_fill_stats(&arr[i], stats_flat + i * 29, 0);
  const int r = calculate_section_intra_ratio(arr, arr + count, section_length);
  free(arr);
  return r;
}

double shim_p2_get_second_ref_usage_thresh(int frame_count_so_far) {
  return get_second_ref_usage_thresh(frame_count_so_far);
}

/* detect_flash reads a stats run through read_frame_stats, which bounds-checks
 * against the buffer's start and end. The shim builds that run and positions
 * `stats_in` at `cur`, so `offset` is exercised in both directions. */
int shim_p2_detect_flash(const double *stats_flat, int count, int cur,
                         int offset) {
  TWO_PASS p;
  TWO_PASS_FRAME pf;
  STATS_BUFFER_CTX ctx;
  memset(&p, 0, sizeof(p));
  memset(&pf, 0, sizeof(pf));
  memset(&ctx, 0, sizeof(ctx));
  FIRSTPASS_STATS *arr =
      (FIRSTPASS_STATS *)calloc((size_t)(count > 0 ? count : 1),
                                sizeof(FIRSTPASS_STATS));
  if (!arr) return -1;
  for (int i = 0; i < count; ++i) shim_p2_fill_stats(&arr[i], stats_flat + i * 29, 0);
  ctx.stats_in_start = arr;
  ctx.stats_in_end = arr + count;
  p.stats_buf_ctx = &ctx;
  pf.stats_in = arr + cur;
  const int r = detect_flash(&p, &pf, offset);
  free(arr);
  return r;
}

int shim_p2_read_frame_stats_in_range(int count, int cur, int offset) {
  /* Returns 1 when read_frame_stats would return non-NULL. Lets the port's
   * bounds logic be checked without a pointer crossing the boundary. */
  TWO_PASS p;
  TWO_PASS_FRAME pf;
  STATS_BUFFER_CTX ctx;
  memset(&p, 0, sizeof(p));
  memset(&pf, 0, sizeof(pf));
  memset(&ctx, 0, sizeof(ctx));
  FIRSTPASS_STATS *arr =
      (FIRSTPASS_STATS *)calloc((size_t)(count > 0 ? count : 1),
                                sizeof(FIRSTPASS_STATS));
  if (!arr) return -1;
  ctx.stats_in_start = arr;
  ctx.stats_in_end = arr + count;
  p.stats_buf_ctx = &ctx;
  pf.stats_in = arr + cur;
  const int r = read_frame_stats(&p, &pf, offset) != NULL;
  free(arr);
  return r;
}

/* ======================================================================== *
 * The region-analysis cluster (pass2_strategy.c:1125-1460, :3677-3730).
 *
 * REGIONS crosses as five doubles + three ints per entry, assigned by name.
 * A stats RUN crosses as `count` records of 29 doubles, plus a separate
 * `is_flash` byte per record -- the region code branches on is_flash all over,
 * and it is an int64_t rather than a double so it cannot ride in the array.
 * ======================================================================== */
static FIRSTPASS_STATS *shim_p2_alloc_run(const double *flat,
                                          const int8_t *is_flash, int count) {
  FIRSTPASS_STATS *arr =
      (FIRSTPASS_STATS *)calloc((size_t)(count > 0 ? count : 1),
                                sizeof(FIRSTPASS_STATS));
  if (!arr) return NULL;
  for (int i = 0; i < count; ++i) {
    shim_p2_fill_stats(&arr[i], flat + i * 29, is_flash ? is_flash[i] : 0);
  }
  return arr;
}

static void shim_p2_load_regions(REGIONS *r, const int *starts, const int *lasts,
                                 const int *types, const double *dbl,
                                 int count) {
  for (int k = 0; k < count; ++k) {
    r[k].start = starts[k];
    r[k].last = lasts[k];
    r[k].type = (REGION_TYPES)types[k];
    r[k].avg_noise_var = dbl[k * 5 + 0];
    r[k].avg_cor_coeff = dbl[k * 5 + 1];
    r[k].avg_sr_fr_ratio = dbl[k * 5 + 2];
    r[k].avg_intra_err = dbl[k * 5 + 3];
    r[k].avg_coded_err = dbl[k * 5 + 4];
  }
}

static void shim_p2_store_regions(const REGIONS *r, int *starts, int *lasts,
                                  int *types, double *dbl, int count) {
  for (int k = 0; k < count; ++k) {
    starts[k] = r[k].start;
    lasts[k] = r[k].last;
    types[k] = (int)r[k].type;
    dbl[k * 5 + 0] = r[k].avg_noise_var;
    dbl[k * 5 + 1] = r[k].avg_cor_coeff;
    dbl[k * 5 + 2] = r[k].avg_sr_fr_ratio;
    dbl[k * 5 + 3] = r[k].avg_intra_err;
    dbl[k * 5 + 4] = r[k].avg_coded_err;
  }
}

/* The region array libaom sizes as MAX_FIRSTPASS_ANALYSIS_FRAMES; the shim
 * allocates the caller's capacity, which the tests keep well above it. */
int shim_p2_smooth_filter_stats(const double *flat, const int8_t *is_flash,
                                int count, int start_idx, int last_idx,
                                double *filt_intra, double *filt_coded) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash, count);
  if (!arr) return -1;
  smooth_filter_stats(arr, start_idx, last_idx, filt_intra, filt_coded);
  free(arr);
  return 0;
}

void shim_p2_get_gradient(const double *values, int start, int last,
                          double *grad) {
  get_gradient(values, start, last, grad);
}

int shim_p2_analyze_region(const double *flat, const int8_t *is_flash,
                           int count, int k, int cap, int *starts, int *lasts,
                           int *types, double *dbl) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash, count);
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!arr || !r) {
    free(arr);
    free(r);
    return -1;
  }
  shim_p2_load_regions(r, starts, lasts, types, dbl, cap);
  analyze_region(arr, k, r);
  shim_p2_store_regions(r, starts, lasts, types, dbl, cap);
  free(arr);
  free(r);
  return 0;
}

int shim_p2_get_region_stats(const double *flat, const int8_t *is_flash,
                             int count, int num_regions, int cap, int *starts,
                             int *lasts, int *types, double *dbl) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash, count);
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!arr || !r) {
    free(arr);
    free(r);
    return -1;
  }
  shim_p2_load_regions(r, starts, lasts, types, dbl, cap);
  get_region_stats(arr, r, num_regions);
  shim_p2_store_regions(r, starts, lasts, types, dbl, cap);
  free(arr);
  free(r);
  return 0;
}

int shim_p2_find_stable_regions(const double *flat, const int8_t *is_flash,
                                int count, const double *grad_coded,
                                int this_start, int this_last, int cap,
                                int *starts, int *lasts, int *types,
                                double *dbl) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash, count);
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!arr || !r) {
    free(arr);
    free(r);
    return -1;
  }
  const int n = find_stable_regions(arr, grad_coded, this_start, this_last, r);
  shim_p2_store_regions(r, starts, lasts, types, dbl, cap);
  free(arr);
  free(r);
  return n;
}

int shim_p2_remove_region(int merge, int cap, int *starts, int *lasts,
                          int *types, double *dbl, int *num_regions,
                          int *next_region) {
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!r) return -1;
  shim_p2_load_regions(r, starts, lasts, types, dbl, cap);
  remove_region(merge, r, num_regions, next_region);
  shim_p2_store_regions(r, starts, lasts, types, dbl, cap);
  free(r);
  return 0;
}

int shim_p2_insert_region(int start, int last, int type, int cap, int *starts,
                          int *lasts, int *types, double *dbl,
                          int *num_regions, int *cur_region_idx) {
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!r) return -1;
  shim_p2_load_regions(r, starts, lasts, types, dbl, cap);
  insert_region(start, last, (REGION_TYPES)type, r, num_regions,
                cur_region_idx);
  shim_p2_store_regions(r, starts, lasts, types, dbl, cap);
  free(r);
  return 0;
}

int shim_p2_cleanup_regions(int cap, int *starts, int *lasts, int *types,
                            double *dbl, int *num_regions) {
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!r) return -1;
  shim_p2_load_regions(r, starts, lasts, types, dbl, cap);
  cleanup_regions(r, num_regions);
  shim_p2_store_regions(r, starts, lasts, types, dbl, cap);
  free(r);
  return 0;
}

int shim_p2_remove_short_regions(int cap, int *starts, int *lasts, int *types,
                                 double *dbl, int *num_regions, int type,
                                 int length) {
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!r) return -1;
  shim_p2_load_regions(r, starts, lasts, types, dbl, cap);
  remove_short_regions(r, num_regions, (REGION_TYPES)type, length);
  shim_p2_store_regions(r, starts, lasts, types, dbl, cap);
  free(r);
  return 0;
}

int shim_p2_find_regions_index(int cap, const int *starts, const int *lasts,
                               const int *types, const double *dbl,
                               int num_regions, int frame_idx) {
  REGIONS *r = (REGIONS *)calloc((size_t)cap, sizeof(REGIONS));
  if (!r) return -2;
  shim_p2_load_regions(r, (int *)starts, (int *)lasts, (int *)types,
                       (double *)dbl, cap);
  const int k = find_regions_index(r, num_regions, frame_idx);
  free(r);
  return k;
}

int shim_p2_mark_flashes(const double *flat, const int8_t *is_flash_in,
                         int count, int8_t *is_flash_out) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash_in, count);
  if (!arr) return -1;
  mark_flashes(arr, arr + count);
  for (int i = 0; i < count; ++i) is_flash_out[i] = (int8_t)arr[i].is_flash;
  free(arr);
  return 0;
}

int shim_p2_smooth_filter_noise(const double *flat, const int8_t *is_flash,
                                int count, double *noise_out) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash, count);
  if (!arr) return -1;
  const int r = smooth_filter_noise(arr, arr + count);
  for (int i = 0; i < count; ++i) noise_out[i] = arr[i].noise_var;
  free(arr);
  return r;
}

int shim_p2_region_types(int *stable, int *high_var, int *scenecut,
                         int *blending) {
  *stable = (int)STABLE_REGION;
  *high_var = (int)HIGH_VAR_REGION;
  *scenecut = (int)SCENECUT_REGION;
  *blending = (int)BLENDING_REGION;
  return 0;
}

int shim_p2_half_win(void) { return HALF_WIN; }
int shim_p2_half_filt_len(void) { return HALF_FILT_LEN; }

/* ======================================================================== *
 * The scenecut / noise-model helpers.
 * ======================================================================== */
int shim_p2_find_qindex_by_rate_with_correction(uint64_t desired_bits_per_mb,
                                                int bit_depth,
                                                double error_per_mb,
                                                double group_weight_factor,
                                                int rate_err_tol,
                                                int best_qindex,
                                                int worst_qindex) {
  return find_qindex_by_rate_with_correction(
      desired_bits_per_mb, (aom_bit_depth_t)bit_depth, error_per_mb,
      group_weight_factor, rate_err_tol, best_qindex, worst_qindex);
}

int shim_p2_slide_transition(const double *this_flat, const double *last_flat,
                             const double *next_flat) {
  FIRSTPASS_STATS a, b, c;
  shim_p2_fill_stats(&a, this_flat, 0);
  shim_p2_fill_stats(&b, last_flat, 0);
  shim_p2_fill_stats(&c, next_flat, 0);
  return slide_transition(&a, &b, &c);
}

/* estimate_noise and estimate_coeff both rewrite the run in place; the shim
 * returns the two fields they touch. estimate_noise's error_info is only
 * dereferenced when smooth_filter_noise fails to allocate. */
int shim_p2_estimate_noise(const double *flat, const int8_t *is_flash,
                           int count, double *noise_out) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash, count);
  struct aom_internal_error_info *err =
      (struct aom_internal_error_info *)calloc(1, sizeof(*err));
  if (!arr || !err) {
    free(arr);
    free(err);
    return -1;
  }
  estimate_noise(arr, arr + count, err);
  for (int i = 0; i < count; ++i) noise_out[i] = arr[i].noise_var;
  free(arr);
  free(err);
  return 0;
}

int shim_p2_estimate_coeff(const double *flat, const int8_t *is_flash,
                           int count, const double *noise_in,
                           double *cor_out) {
  FIRSTPASS_STATS *arr = shim_p2_alloc_run(flat, is_flash, count);
  if (!arr) return -1;
  /* estimate_coeff reads noise_var, which estimate_noise writes; the caller
   * supplies it so the two can be driven independently. */
  for (int i = 0; i < count; ++i) arr[i].noise_var = noise_in[i];
  estimate_coeff(arr, arr + count);
  for (int i = 0; i < count; ++i) cor_out[i] = arr[i].cor_coeff;
  free(arr);
  return 0;
}

int shim_p2_very_low_ii_x1000(void) { return (int)(VERY_LOW_II * 1000); }
double shim_p2_very_low_ii(void) { return VERY_LOW_II; }
double shim_p2_error_spike(void) { return ERROR_SPIKE; }
