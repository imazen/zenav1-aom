# Smooth-V codegen captures

Apple M4 Pro; rustc 1.98 / LLVM 22. Disassembled with `otool -tvV -p <smooth_v symbol> target/release/deps/dsp_kernels-4b99c028063fb4cd`.

- `/Users/lilith/tmp/arm-all-2026-09-06/aom-smooth_v-before.s`: SHA-256 `53ce30a0283623113f6728ba451a40118265e422c5d40508a275210b5f3c5890`
- `/Users/lilith/tmp/arm-all-2026-09-06/aom-smooth_v-direct-store.s`: SHA-256 `40b76c59a7dff667cae3157096a08787b21de98944fb12b561df2025c51d0a49`

No cloud/NAS mirror made. Before: benchmark commit `257d455f` / production `45c53ddb`. After: direct-store experiment described in README.md. Both have two memcpy call sites because partial-vector paths remain; full vectors now store directly. Total function assembly grows from 829 to 865 lines.
