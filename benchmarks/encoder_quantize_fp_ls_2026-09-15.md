# quantize_fp — const-generic `LS` specialization, mirroring C's three kernels (2026-09-15)

## Mechanism

`crates/aom-dsp/src/quant/simd.rs` `quantize_fp_impl_v3` carried `log_scale` as
a runtime parameter, so LLVM inlined the quant body three ways under a runtime
branch at every 16-coefficient chunk (kernel disassembly: 365 instructions vs
the C kernel's 126). C ships three log_scale-specialized kernels
(`av1_quantize_fp_avx2`, `_32x32`, `_64x64`) — `const int log_scale` per body.

The v3 dispatcher now matches `log_scale` once per call and monomorphizes
`quantize_fp_v3_ls::<LS>` for `LS in {0, 1, 2}` — with `LS` a literal the
`rt`/`qt` param math, the per-chunk selects and the variable-count threshold
shift all const-fold to exactly the arm C compiles. Call sites and the
`n % 16`/length preflight are unchanged; unroutable shapes still take the
scalar recipe.

## Measurement

Callgrind 196x196 cq27 `--cpu-used 3`, 5 reps, identical call counts:

- `quantize_fp_impl_v3` self Ir: **172.3M → 165.4M (−4 %)**
- cell total: −0.08 %

Below band resolution — landed on the strict-Ir precedent (real instruction
reduction, wall-unmeasurable at this size).

## Correctness

- `quantize_fp_avx2_diff` — real exported `av1_quantize_fp_avx2` oracle — green.
- Shipping cell byte-identical: 40,237 B.
