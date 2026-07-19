# zenav1-aom-dsp

The consolidated DSP + entropy kernels for the pure-Rust, bit-exact
[libaom](https://aomedia.googlesource.com/aom) v3.14.1 port: transform, quant,
txb, cdef, restore, intra, loopfilter, dist, inter, convolve, recon, dispatch,
and the MSAC range coder — each a module (`aom_dsp::transform`, `aom_dsp::entropy`,
…). Runtime SIMD dispatch via [archmage](https://github.com/imazen/archmage);
`#![forbid(unsafe_code)]`.

Most consumers want the [`zenav1-aom`](https://crates.io/crates/zenav1-aom) facade
instead. Which module ports which libaom file, and how to run each differential
gate, is in [PORTING.md](https://github.com/imazen/zenav1-aom/blob/main/PORTING.md).

## License

Dual-licensed: AGPL-3.0-only OR the Imazen commercial license
(`LicenseRef-Imazen-Commercial`). See
[LICENSE-AGPL3](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-AGPL3) and
[LICENSE-COMMERCIAL](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-COMMERCIAL).
Derived from upstream libaom (BSD-2-Clause + AOM Patent License 1.0).
