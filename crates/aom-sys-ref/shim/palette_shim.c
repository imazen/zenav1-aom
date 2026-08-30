/* Oracle shims for the palette k-means kernels — the DISPATCHED entry points
 * (`av1_calc_indices_dim{1,2}` are rtcd-specialised sse2/avx2/neon; the
 * `av1_k_means_dim{1,2}` templates call them), so a differential against these
 * measures the kernel real aomenc runs, not the `_c` template the port was
 * transcribed from. KB-41: palette search fidelity on screen-detected content.
 * Append-only; every other shim is untouched. Requires ref_init() (rtcd). */
#include <stdint.h>
#include "config/av1_rtcd.h"
#include "av1/encoder/palette.h"

int64_t shim_calc_indices(const int16_t *data, const int16_t *centroids,
                          uint8_t *indices, int n, int k, int dim) {
  int64_t total_dist = 0;
  if (dim == 1)
    av1_calc_indices_dim1(data, centroids, indices, &total_dist, n, k);
  else
    av1_calc_indices_dim2(data, centroids, indices, &total_dist, n, k);
  return total_dist;
}

void shim_k_means(const int16_t *data, int16_t *centroids, uint8_t *indices,
                  int n, int k, int dim, int max_itr) {
  if (dim == 1)
    av1_k_means_dim1(data, centroids, indices, n, k, max_itr);
  else
    av1_k_means_dim2(data, centroids, indices, n, k, max_itr);
}
