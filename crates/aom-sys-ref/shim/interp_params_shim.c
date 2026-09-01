/* Oracle shim for av1/common/filter.h's block-size-dependent filter selection.
 *
 * `av1_get_interp_filter_params_with_block_size` (filter.h:249) is a `static
 * inline` in a header, so it has no address; this TU compiles it and reports
 * what it returns. Both the CHOICE of table and the table's CONTENTS cross the
 * boundary, so the two are gated together — a port with the right selector and
 * a mistyped coefficient fails exactly as loudly as one with the wrong
 * selector.
 *
 * Why it matters: on an axis of 4 pixels or fewer the encoder switches to
 * `av1_interp_4tap`, which is a different set of coefficients at the SAME
 * eight taps (filter.h:240-246 declares them `SUBPEL_TAPS`), and MULTITAP_SHARP
 * collapses onto the REGULAR entry there. Every chroma plane of an 8x8 luma
 * block takes that path.
 */
#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "av1/common/filter.h"

/* `out` receives SUBPEL_SHIFTS * SUBPEL_TAPS int16 coefficients, row-major.
 * Returns the filter's `taps`. */
int shim_ifp_table(int interp_filter, int w, int16_t *out) {
  const InterpFilterParams *p = av1_get_interp_filter_params_with_block_size(
      (InterpFilter)interp_filter, w);
  memcpy(out, p->filter_ptr,
         (size_t)SUBPEL_SHIFTS * SUBPEL_TAPS * sizeof(int16_t));
  return p->taps;
}

int shim_ifp_subpel_shifts(void) { return SUBPEL_SHIFTS; }
int shim_ifp_subpel_taps(void) { return SUBPEL_TAPS; }
