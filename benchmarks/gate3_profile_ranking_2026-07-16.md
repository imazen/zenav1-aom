# Gate-3 Phase-2 hot-kernel ranking (callgrind instruction counts)

Instruction-count profiles of the PORT side of two representative Gate-3
cells. Ir (instructions retired under callgrind) is load-tolerant — this
ranking was measured on a busy box and is valid; WALL-CLOCK ratios are a
separate, quiet-box measurement (`benchmarks/gate3_baseline_*.csv`, pending).

Per-function Ir below aggregates the function's own lines PLUS the
std-library lines callgrind attributes to it via inlining (cmp.rs `clamp`,
slice index, iter/range) — those are part of the function's real cost.

## Encode: `enc_s0_128_cq32` (128x128 real content, bd8 4:2:0, allintra speed-0)

Command: `gate3_profile enc port enc_s0_128_cq32 6` (7 port encodes + 1 C
encode from setup byte-verify; the C encode is ~4% of total Ir).
Total: 155,210,572,291 Ir.

| rank | kernel cluster | ~Ir share | SIMD in C? | plan |
|---|---|---|---|---|
| 1 | transforms total: `av1_inv_txfm2d_add` ~13%, `av1_fwd_txfm2d` ~9%, 1-D kernels (idct/iadst/fdct/fadst 4..32) ~11% | **~33%** | yes (avx2/sse4) | magetypes SIMD, bit-identical |
| 2 | aom-txb coeff kernels: `optimize_txb_core` 5.5%, `get_lower_levels_ctx` ~4.1%, `get_nz_mag` ~4.6%, `txb_init_levels` ~2.8%, `two_coeff_cost_simple` 1.1% | **~18%** | levels/contexts yes; the trellis loop itself is scalar in C too | SIMD init_levels + nz_mag + levels_ctx; trellis core stays scalar (C pays the same) |
| 3 | `av1_quantize_fp_no_qmatrix` | **~5%** | yes | magetypes SIMD (lane-pure) |
| 4 | `highbd_variance` | **~3%** | yes | magetypes SIMD |
| 5 | malloc/calloc/free/memset | ~3.5% | n/a | allocation-reuse pass, later |
| 6 | intra predictors (`z2_high` + family) | ~2% | yes | after the above |

`search_tx_type_intra` (driver, 0.9%) and the rest are control flow.

## Decode: `dec_352x288_q32` (real conformance vector, CDEF + LR coded)

Command: `gate3_profile dec port dec_352x288_q32 30`.
Total: 3,656,695,999 Ir.

| rank | kernel cluster | ~Ir share | SIMD in C? | plan |
|---|---|---|---|---|
| 1 | `cdef_filter_block_16` (+ `cdef_find_dir` 1.7%) | **~27%** | yes | magetypes SIMD |
| 2 | `wiener_convolve_add_src` (loop restoration) | **~13%** | yes | magetypes SIMD |
| 3 | `av1_inv_txfm2d_add` + 1-D kernels | **~12%** | yes | shared with encode item 1 |
| 4 | deblock (`filter4/6/8/14`, `lpf_*`, `set_lpf_parameters`) | **~9.5%** | yes | magetypes SIMD |
| 5 | od_ec entropy decode (`read_symbol`/`decode_cdf_q15`) | ~6.5% | no (serial) | NOT SIMD-able; C pays it too |
| 6 | txb read contexts (`read_txb_body`, `get_nz_mag`, `get_lower_levels_ctx`, `dequant_txb`) | ~5% | partly | shared with encode item 2 |
| 7 | memset/memcpy (buffer zeroing) | ~4.7% | n/a | allocation-reuse pass, later |

Note: CDEF/LR shares depend on the stream (this vector codes both). The
allintra-default (zenavif) encode path has CDEF/LR off, but decode must take
whatever the stream carries, and conformance vectors carry them.

## Cross-track SIMD order (breadth-first, per-kernel landings)

1. `av1_quantize_fp` family (prove the dispatch pattern end-to-end; lane-pure)
2. inverse transform stack (both tracks' biggest shared block)
3. forward transform stack
4. txb trio: `txb_init_levels`, `get_nz_mag`, `get_lower_levels_ctx`
5. CDEF filter (decode #1)
6. deblock filters
7. `highbd_variance` + SAD/SATD family
8. wiener convolve

Float decision helpers (`av1_nn_predict`, HOG, softmax, variance-factor)
stay SCALAR per the parity rules — decision-side, cheap, and float SIMD
reassociation would shift RD decisions.
