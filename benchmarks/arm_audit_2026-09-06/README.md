# ARM DSP audit, 2026-09-06

The initial 87-cell tier sweep is complete. Additional dispatched quantization, SAD, and high-bit-depth intra cells correct the scalar-only entry points in the original benchmark. Coverage is the existing standalone DSP benchmark's transform, CDEF, loopfilter, distortion, quantization, and intra-prediction cells; this is not a full encoder or decoder timing.

Apple M4 Pro, 24 GiB RAM, macOS Darwin 25.5, rustc 1.98 / LLVM 22. Production source `45c53ddb`; formatting commit `a95306fe`, benchmark rewrite `f8b88099`. No target-cpu=native. Command: `CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR=/Users/lilith/tmp nice -n 19 /usr/bin/time -l cargo bench --locked -p zenav1-aom-dsp-bench --bench dsp_kernels -- --format=llm`.

The standalone benchmark now enables archmage/testable_dispatch explicitly. Its old ARM comparison instructions could not disable baseline NEON without that feature. Each kernel/size is now its own interleaved native-versus-forced-scalar comparison, with token switching outside the measured region. The scalar compiler remains free to auto-vectorize. The benchmark rejects AOM_FORCE_SCALAR to avoid overriding its paired setup.

Each sample processes the pre-existing WORK_PX=65536 work budget with a 256 KiB cap on the cycled working set. These are hot-kernel measurements and do not establish end-to-end codec speedups or memory scaling.

Baseline strict Clippy reports 130 errors in the existing zenav1-aom-dsp library, including unused imports, range-loop suggestions, and newer lint recommendations. They predate the benchmark rewrite and are not suppressed. The benchmark's validation will be recorded after the measurement finishes.


## Initial sweep and corrected dispatch coverage

The initial sweep contains 27 inverse-transform cells (24 significant SIMD wins, three ties), 17 forward-transform cells (11 wins, six ties), three CDEF cells (three wins), and six loopfilter cells (six wins). The other 34 cells are scalar-only controls: 15 distortion, four quantization, and 15 8-bit intra-prediction cells. Their source entry points never consult the SIMD token, so their ties say nothing about NEON performance. The per-family logs retain their original labels; the benchmark now labels them `*_scalar_control`.

`dist::simd::sad_simd` is measured separately against both its forced-scalar variant and `dist::sad`, with exact result checks at all five sizes. Quantization now also measures `quant::simd::av1_quantize_fp_no_qmatrix_dispatch`; this is used by the encoder at `crates/aom-encode/src/lib.rs` and `nonrd_pickmode.rs`. Intra dispatch is measured through `intra::predict_highbd` at bit depth 8 with u16 storage, against its own fallback. These are distinct from the original u8 scalar control; do not compare their timings as before/after production changes.

The four dispatched quantizer cells passed exact EOB/qcoeff/dqcoeff comparisons. Paired scalar-over-SIMD time increases were 225.55%, 272.98%, 309.74%, and 326.00% at 16, 64, 256, and 1024 coefficients. The 1024-coefficient cell had 27% CV; its paired interval remained positive. No encoder throughput improvement is claimed: the production dispatch already existed.

The SAD dispatcher disassembly is retained in `sad_dispatch_arm.s`. Both token branches contain `uabd.16b` and `uabdl.8h` vector loops, explaining why toggling NEON does not remove vectorization here. All five paired SAD cells tie; the separate scalar helper is slower in all five. This is generated-code evidence for this binary, not a claim about other CPUs or compilers.

The 15 dispatched intra cells expose four regressions: forced scalar is 7.39% faster for Paeth 4x4, 6.10% for smooth 4x4, 13.42% for smooth-V 4x4, and 10.75% for smooth-V 32x32. Five larger Paeth/smooth cells favor SIMD; the six V/H copy controls tie (the smooth-V 32x32 regression is counted separately). These measurements motivate codegen follow-up; no fallback policy has been changed based on this single bit-depth/shape fixture.

Validation: seven existing DSP integration binaries passed: forward/inverse transforms, quantization, CDEF, intra predictors, loopfilter and SAD. One pre-existing SAD performance test is ignored. The intra differential covers all modes, widths/heights 4/8/16/32/64, bit depths 8/10/12, and both tight and padded strides across token permutations. Full log: `aom_simd_parity_2026-09-06.log`.
