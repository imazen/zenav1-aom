# Reference oracle build config

- Source: libaom, tag **v3.14.1**, git `03087864cf4bea6abb0d28f95cf7843511413d8f`
  — the pinned **`upstream/`** git submodule (canonical). The gitignored
  `reference/libaom` clone remains as a fallback.
- Toolchain: gcc 15.2.0 / clang 21.1.8 / nasm 3.01, cmake 4.2.3
- CMake:
  ```
  -DCMAKE_BUILD_TYPE=Release
  -DCONFIG_MULTITHREAD=0     # single-thread → deterministic encoder output target
  -DENABLE_TESTS=1 -DENABLE_EXAMPLES=1 -DENABLE_TOOLS=1
  -DCONFIG_AV1_DECODER=1 -DCONFIG_AV1_ENCODER=1
  ```
- C flags, pinned on EVERY oracle TU (libaom via `CMAKE_C_FLAGS`, and the
  `aom-sys-ref/shim/*.c` compile line):
  ```
  -ffp-contract=off          # no a*b+c -> FMA fusion; see below
  ```
  This is part of the oracle's *definition*, not a tuning knob. Clang defaults
  to `-ffp-contract=on` for C. On **aarch64** `fmadd` is baseline, so the default
  fuses and rounds once; on **x86-64** no libaom TU enables FMA (`-mavx2` only,
  `cmake/aom_optimization.cmake` + `av1/av1.cmake`), so nothing can fuse and the
  flag is a no-op. Rust never contracts, so without this pin "bit-exact vs
  libaom" would mean different things on different hosts. Pinning it makes the
  oracle strictly IEEE-per-operation and host-independent — same class of
  decision as `CONFIG_MULTITHREAD=0`. (CLAUDE.md KB-ARM-FLOAT root #1.)
  Note the consequence: a *production* aarch64 libaom build does contract, so it
  can differ from this oracle (and from x86-64 libaom) by a few ULP in the
  NN/curve-fit/denoise float kernels.
- Shim TUs otherwise compile at `-O2`, with ONE documented exception:
  `shim/cnn_cscalar.c` uses `-O3 -DNDEBUG` (libaom's own Release flags). It is
  the only shim that pulls a libaom `.c` — `av1/encoder/cnn.c` — into
  `libaom_shim.a`, to obtain a **scalar-bound copy of the CNN inference
  engine**: the CNN's one RTCD-dispatched primitive
  (`av1_cnn_convolve_no_maxpool_padding_valid`) is rebound to its `_c` variant
  and every export is renamed `shim_cscalar_*`, so it links beside libaom.a's
  dispatched copy. The former mechanism — swapping libaom's runtime RTCD
  function *pointer* — only exists on x86-64; on aarch64 NEON is baseline and
  the generated `config/av1_rtcd.h` binds the primitive with a compile-time
  `#define ..._neon`, so the "C-scalar" CNN oracle silently was not scalar
  (CLAUDE.md KB-ARM-FLOAT root #2). Matching libaom's Release flags on this one
  TU makes it the same source under the same settings as the copy inside
  libaom.a. Per-TU flags live in `extra_shim_cflags()` in
  `crates/aom-sys-ref/build.rs`.
- Artifacts: `upstream/build/{libaom.a, aomenc, aomdec}`, built automatically by
  `crates/aom-sys-ref/build.rs` (cached by the submodule SHA).
- `CONFIG_COEFFICIENT_RANGE_CHECKING = 0`, `DO_RANGE_CHECK_CLAMP` off (default),
  so transform range-check functions are no-ops. This is the definition against
  which aom-rs bit-exactness is measured.

Build: cargo-driven — `cargo test` (or `cargo build -p aom-sys-ref`) builds the
oracle from `upstream/` into `upstream/build/` automatically, once, cached by the
submodule SHA. If `upstream/` is empty, build.rs auto-runs
`git submodule update --init upstream`. `bash reference/build.sh` remains a
fallback that clones + builds `reference/libaom`.
