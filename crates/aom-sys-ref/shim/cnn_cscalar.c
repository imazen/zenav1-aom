/* A SCALAR-BOUND copy of libaom's CNN inference engine (av1/encoder/cnn.c),
 * compiled into libaom_shim.a alongside the dispatched copy in libaom.a.
 *
 * WHY THIS FILE EXISTS (KB-ARM-FLOAT root #2).
 * `cnn_partition_cnn_diff` asserts the Rust cascade is a bit-exact
 * transcription of `av1_cnn_convolve_no_maxpool_padding_valid_c` + the layer
 * wiring, so the oracle it compares against must genuinely be the C-scalar
 * engine. rd_shim.c used to obtain that by swapping libaom's runtime RTCD
 * FUNCTION POINTER to the `_c` variant, which only exists on x86-64. On
 * aarch64 NEON is baseline, so the generated config/av1_rtcd.h binds the same
 * primitive at COMPILE TIME:
 *
 *     #define av1_cnn_convolve_no_maxpool_padding_valid \
 *             av1_cnn_convolve_no_maxpool_padding_valid_neon
 *
 * There is no pointer to swap, so `av1_cnn_predict_c` in libaom.a calls the
 * NEON convolve and the "C-scalar" oracle was not scalar at all (residual
 * 1 ULP after the -ffp-contract=off fix).
 *
 * HOW. Include av1_rtcd.h first so its declarations are processed verbatim,
 * then rebind the ONE RTCD-dispatched primitive the CNN engine uses to its
 * `_c` variant with a macro that the compiler applies to cnn.c's call site,
 * and rename every symbol cnn.c exports so this copy links next to libaom.a's
 * instead of colliding with it. The rebinding is uniform across targets: on
 * x86-64 the RTCD name is a pointer object (the `#undef` is a no-op and the
 * macro simply rewrites the call), on aarch64 it is a macro that gets
 * redirected. The oracle therefore means exactly the same thing on every host,
 * which is the whole point of pinning an oracle (cf. -ffp-contract=off,
 * CONFIG_MULTITHREAD=0).
 *
 * FLOAT FAITHFULNESS. build.rs compiles this TU with libaom's own Release
 * flags (-O3 -DNDEBUG) rather than the shims' default -O2, so it is the same
 * source under the same optimisation settings as libaom.a's copy. (Absent
 * fast-math a C compiler may not reassociate float arithmetic, so the
 * optimisation level cannot change the values anyway — matching the flags just
 * removes the question.)
 *
 * NOTE this is the ONLY shim that pulls a libaom .c into the shim archive.
 * Everything else in shim/ calls the real exported functions, per the evidence
 * hierarchy in CLAUDE.md. This file does not transcribe or reimplement
 * anything: it is libaom's own source, compiled with one RTCD binding pinned.
 */

/* Processed with its normal per-target contents BEFORE any rebinding below. */
#include "config/aom_config.h"
#include "config/av1_rtcd.h"

/* --- 1. Rebind the CNN's only SIMD-dispatched primitive to the scalar one. --- */
#undef av1_cnn_convolve_no_maxpool_padding_valid
#define av1_cnn_convolve_no_maxpool_padding_valid \
  av1_cnn_convolve_no_maxpool_padding_valid_c

/* --- 2. Rename every symbol av1/encoder/cnn.c exports. ---
 * Macro expansion is rescanned, so the rebinding above resolves through to the
 * renamed local definition:
 *   av1_cnn_convolve_no_maxpool_padding_valid
 *     -> av1_cnn_convolve_no_maxpool_padding_valid_c
 *     -> shim_cscalar_cnn_convolve_no_maxpool_padding_valid_c
 * The engine entry point is renamed straight to its public shim name.
 */
#define av1_find_cnn_layer_output_size shim_cscalar_find_cnn_layer_output_size
#define av1_find_cnn_output_size shim_cscalar_find_cnn_output_size
#define av1_cnn_add_c shim_cscalar_cnn_add_c
#define av1_cnn_activate_c shim_cscalar_cnn_activate_c
#define av1_cnn_convolve_no_maxpool_padding_valid_c \
  shim_cscalar_cnn_convolve_no_maxpool_padding_valid_c
#define av1_cnn_batchnorm_c shim_cscalar_cnn_batchnorm_c
#define av1_cnn_deconvolve_c shim_cscalar_cnn_deconvolve_c
#define av1_cnn_predict_c shim_cscalar_cnn_predict_c
#define av1_cnn_predict_img_multi_out shim_cnn_predict_img_multi_out_cscalar
#define av1_cnn_predict_img_multi_out_highbd \
  shim_cnn_predict_img_multi_out_highbd_cscalar

/* --- 3. libaom's own engine, unmodified. --- */
#include "av1/encoder/cnn.c"
