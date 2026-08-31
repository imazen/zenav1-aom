/* A probe that lets a Rust test observe the ORACLE's own view of its RTCD
 * dispatch table. Oracle use only.
 *
 * Why this exists: on this aarch64 oracle most RTCD names are `#define`d
 * straight to their NEON implementations (see config/aom_dsp_rtcd.h), so no
 * function pointer exists and a shim that forgets `ref_init()` cannot fault.
 * On x86 the same names are real `RTCD_EXTERN` pointers, null until
 * `aom_dsp_rtcd()` / `av1_rtcd()` run — so the identical shim segfaults there
 * and passes here. `nm` on the archive already shows the split (`C` common
 * symbols vs absent names), and this probe lets a test assert the invariant at
 * runtime on whichever ISA it is built for.
 *
 * `shim_rtcd_ptr_names` reports which of the kernels this repo's inter shims
 * reach are pointers in THIS build; `shim_rtcd_ptrs_initialised` reports
 * whether each is non-NULL right now.
 */
#include <stdint.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

/* The dispatched kernels reachable from the inter-encode shims, one level
 * below the entry point each shim actually calls:
 *   aom_sad32x32        <- av1_segmented_frame_error, warp_error
 *   av1_warp_affine     <- warp_plane <- av1_refine_integerized_param
 *   aom_vector_var      <- av1_vector_match
 *   aom_variance16x16   <- the full-pel/subpel searches' vfp->vf default
 *   aom_sad16x16        <- ms_params->sdf default
 * Names that are `#define`d on this build report as "not a pointer here",
 * which is information, not a pass.
 */
int shim_rtcd_probe(int which, int *out_is_pointer) {
  const void *p = NULL;
  int is_ptr = 0;
#define PROBE(idx, name)                 \
  case idx:                              \
    is_ptr = 1;                          \
    p = (const void *)(uintptr_t)(name); \
    break;
  switch (which) {
#ifdef aom_sad32x32
    case 0: is_ptr = 0; break;
#else
    PROBE(0, aom_sad32x32)
#endif
#ifdef av1_warp_affine
    case 1: is_ptr = 0; break;
#else
    PROBE(1, av1_warp_affine)
#endif
#ifdef aom_vector_var
    case 2: is_ptr = 0; break;
#else
    PROBE(2, aom_vector_var)
#endif
#ifdef aom_variance16x16
    case 3: is_ptr = 0; break;
#else
    PROBE(3, aom_variance16x16)
#endif
#ifdef aom_sad16x16
    case 4: is_ptr = 0; break;
#else
    PROBE(4, aom_sad16x16)
#endif
/* The four below are `#define`d to their NEON implementations on aarch64 and
 * are real pointers on x86. They are the ones the comp-pred / OBMC / wedge
 * shims reach, i.e. exactly the ones this host cannot observe. Listing them
 * makes the split visible in the test's output instead of implicit. */
#ifdef aom_upsampled_pred
    case 5: is_ptr = 0; break;
#else
    PROBE(5, aom_upsampled_pred)
#endif
#ifdef aom_comp_mask_pred
    case 6: is_ptr = 0; break;
#else
    PROBE(6, aom_comp_mask_pred)
#endif
#ifdef aom_convolve_copy
    case 7: is_ptr = 0; break;
#else
    PROBE(7, aom_convolve_copy)
#endif
#ifdef aom_highbd_upsampled_pred
    case 8: is_ptr = 0; break;
#else
    PROBE(8, aom_highbd_upsampled_pred)
#endif
    default: return -1;
  }
#undef PROBE
  *out_is_pointer = is_ptr;
  return (is_ptr && p != NULL) ? 1 : 0;
}
