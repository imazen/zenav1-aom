/* Oracle shim for firstpass.c's FIRSTPASS_INFO ring buffer. **Tier 1.**
 *
 * All seven `av1_firstpass_info_*` entry points are exported `T` symbols, so
 * this file calls the ARCHIVE's copies — it deliberately does NOT include
 * firstpass.c (that is fp_shim.c's job, for the file-statics). Keeping the
 * two in separate TUs is what lets these stay tier 1.
 *
 * The buffer is stateful, so the shim hands out an opaque handle and the Rust
 * side drives it operation by operation, comparing the full cursor state
 * after each one. That is a stronger gate than replaying a fixed script: an
 * off-by-one in `start_index` shows up on the operation that caused it, not
 * several operations later when it finally changes an answer.
 *
 * `FIRSTPASS_INFO` is ~12 KB (49 inline FIRSTPASS_STATS plus a total) and is
 * heap-allocated for the same reason rdopt_shim.c heap-allocates MACROBLOCK.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"
#include "av1/encoder/firstpass.h"

typedef struct {
  FIRSTPASS_INFO info;
  /* Backing store for the external-buffer form of av1_firstpass_info_init;
   * NULL for the internal form. */
  FIRSTPASS_STATS *ext_buf;
} ShimFpi;

/* Returns NULL on allocation failure; `*out_err` gets the aom_codec_err_t
 * av1_firstpass_info_init returned. */
void *shim_fpi_new(int use_external, const FIRSTPASS_STATS *ext_stats,
                   int ext_size, int *out_err) {
  ShimFpi *h = (ShimFpi *)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (use_external) {
    if (ext_size > 0) {
      h->ext_buf = (FIRSTPASS_STATS *)calloc((size_t)ext_size, sizeof(*h->ext_buf));
      if (!h->ext_buf) {
        free(h);
        return NULL;
      }
      memcpy(h->ext_buf, ext_stats, (size_t)ext_size * sizeof(*h->ext_buf));
    }
    *out_err = (int)av1_firstpass_info_init(&h->info, h->ext_buf, ext_size);
  } else {
    *out_err = (int)av1_firstpass_info_init(&h->info, NULL, 0);
  }
  return h;
}

void shim_fpi_free(void *handle) {
  if (!handle) return;
  ShimFpi *h = (ShimFpi *)handle;
  free(h->ext_buf);
  free(h);
}

int shim_fpi_push(void *handle, const FIRSTPASS_STATS *stats) {
  return (int)av1_firstpass_info_push(&((ShimFpi *)handle)->info, stats);
}

int shim_fpi_pop(void *handle) {
  return (int)av1_firstpass_info_pop(&((ShimFpi *)handle)->info);
}

int shim_fpi_move_cur_index(void *handle) {
  return (int)av1_firstpass_info_move_cur_index(&((ShimFpi *)handle)->info);
}

int shim_fpi_move_cur_index_and_pop(void *handle) {
  return (int)av1_firstpass_info_move_cur_index_and_pop(
      &((ShimFpi *)handle)->info);
}

/* Returns 1 and fills `out` when the peek is in-window, 0 otherwise. */
int shim_fpi_peek(void *handle, int offset_from_cur, FIRSTPASS_STATS *out) {
  const FIRSTPASS_STATS *s =
      av1_firstpass_info_peek(&((ShimFpi *)handle)->info, offset_from_cur);
  if (!s) return 0;
  *out = *s;
  return 1;
}

int shim_fpi_future_count(void *handle, int offset_from_cur) {
  return av1_firstpass_info_future_count(&((ShimFpi *)handle)->info,
                                         offset_from_cur);
}

void shim_fpi_state(void *handle, int *start_index, int *stats_count,
                    int *cur_index, int *future_stats_count,
                    int *past_stats_count, int *stats_buf_size,
                    FIRSTPASS_STATS *total_stats) {
  const FIRSTPASS_INFO *i = &((ShimFpi *)handle)->info;
  *start_index = i->start_index;
  *stats_count = i->stats_count;
  *cur_index = i->cur_index;
  *future_stats_count = i->future_stats_count;
  *past_stats_count = i->past_stats_count;
  *stats_buf_size = i->stats_buf_size;
  *total_stats = i->total_stats;
}

/* AOM_CODEC_OK, so the Rust side does not hard-code an enum value. */
int shim_fpi_codec_ok(void) { return (int)AOM_CODEC_OK; }

/* FIRSTPASS_INFO_STATIC_BUF_SIZE, likewise. */
int shim_fpi_static_buf_size(void) { return FIRSTPASS_INFO_STATIC_BUF_SIZE; }
