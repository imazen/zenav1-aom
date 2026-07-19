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

// Init the compound wedge codebook and copy the baked mask masks[0][index]
// (block_size_wide[bsize] * block_size_high[bsize] bytes, stride bw) into out.
// Returns 0 if the bsize has no wedge types.
int shim_ii_wedge_mask(int bsize, int index, uint8_t *out) {
  av1_init_wedge_masks();
  if (av1_wedge_params_lookup[bsize].wedge_types == 0) return 0;
  const uint8_t *m = av1_get_contiguous_soft_mask((int8_t)index, 0,
                                                  (BLOCK_SIZE)bsize);
  int bw = block_size_wide[bsize];
  int bh = block_size_high[bsize];
  memcpy(out, m, (size_t)bw * (size_t)bh);
  return 1;
}
