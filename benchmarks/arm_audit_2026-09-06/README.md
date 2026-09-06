# ARM DSP audit, 2026-09-06

The full tier sweep is in progress. Coverage is the existing standalone DSP benchmark's transform, CDEF, loopfilter, distortion, quantization, and intra-prediction cells; this is not a full encoder or decoder timing.

Apple M4 Pro, 24 GiB RAM, macOS Darwin 25.5, rustc 1.98 / LLVM 22. Production source `45c53ddb`; formatting commit `a95306fe`, benchmark rewrite `f8b88099`. No target-cpu=native. Command: `CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR=/Users/lilith/tmp nice -n 19 /usr/bin/time -l cargo bench --locked -p zenav1-aom-dsp-bench --bench dsp_kernels -- --format=llm`.

The standalone benchmark now enables archmage/testable_dispatch explicitly. Its old ARM comparison instructions could not disable baseline NEON without that feature. Each kernel/size is now its own interleaved native-versus-forced-scalar comparison, with token switching outside the measured region. The scalar compiler remains free to auto-vectorize. The benchmark rejects AOM_FORCE_SCALAR to avoid overriding its paired setup.

Each sample processes the pre-existing WORK_PX=65536 work budget with a 256 KiB cap on the cycled working set. These are hot-kernel measurements and do not establish end-to-end codec speedups or memory scaling.

Baseline strict Clippy reports 130 errors in the existing zenav1-aom-dsp library, including unused imports, range-loop suggestions, and newer lint recommendations. They predate the benchmark rewrite and are not suppressed. The benchmark's validation will be recorded after the measurement finishes.
