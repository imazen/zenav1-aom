// Differential shim for the interintra blend + wedge codebook.
#include <string.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/blend.h"
#include "av1/common/common_data.h"
#include "av1/common/reconinter.h"
#include "av1/common/blockd.h"

// aom_blend_a64_mask_c: dst = round(mask*src0 + (64-mask)*src1, 6), with the
// mask box-subsampled by (subw, subh) for the chroma plane.
void shim_blend_a64_mask(uint8_t *dst, uint32_t ds, const uint8_t *s0,
                         uint32_t s0s, const uint8_t *s1, uint32_t s1s,
                         const uint8_t *mask, uint32_t ms, int w, int h,
                         int subw, int subh) {
  aom_blend_a64_mask_c(dst, ds, s0, s0s, s1, s1s, mask, ms, w, h, subw, subh);
}

// Copy the baked wedge mask for sign 0, wedge `index`
// (block_size_wide[bsize] * block_size_high[bsize] bytes, stride bw) into out.
// Returns 0 if the bsize has no wedge types, -1 if the mask lookup yields NULL.
//
// v3.15 precomputes the wedge masks at codegen time (tools/gen_wedge_masks_data.py
// -> av1/common/wedge_masks_data.inc): `wedge_mask_buf` is `static const` data and
// `av1_wedge_params_lookup[].masks[][]` holds BYTE OFFSETS into it, read back only
// through `av1_get_contiguous_soft_mask` (reconinter.c, now out of line).
// `av1_init_wedge_masks()` is an empty function. The v3.14.1 hazard this shim used
// to guard (an unsynchronised `aom_once` under CONFIG_MULTITHREAD=0 leaving every
// mask entry NULL while an init was in flight, which SIGSEGV'd in memmove with no
// attribution) no longer exists; the NULL check stays so that a lookup that cannot
// resolve still names itself.
int shim_ii_wedge_mask(int bsize, int index, uint8_t *out) {
  av1_init_wedge_masks();
  if (av1_wedge_params_lookup[bsize].wedge_types == 0) return 0;
  const uint8_t *m = av1_get_contiguous_soft_mask((int8_t)index, 0,
                                                  (BLOCK_SIZE)bsize);
  if (m == NULL) return -1;
  int bw = block_size_wide[bsize];
  int bh = block_size_high[bsize];
  memcpy(out, m, (size_t)bw * (size_t)bh);
  return 1;
}
