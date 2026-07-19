# zenav1-aom-decode

The AV1 decoder for the pure-Rust, bit-exact
[libaom](https://aomedia.googlesource.com/aom) v3.14.1 port: the tile
reconstruction driver — partition walk, per-leaf mode-info/coeff decode, intra
predict, inverse transform, and the deblock/CDEF/restoration frame walk — over the
[`zenav1-aom-dsp`](https://crates.io/crates/zenav1-aom-dsp) kernels. Intra decode
is bit-identical to C across the AV1 intra conformance scope; inter-frame decode is
in progress. `#![forbid(unsafe_code)]`, no C toolchain, no `build.rs`.

Most consumers want the [`zenav1-aom`](https://crates.io/crates/zenav1-aom) facade
(`features = ["decode"]` for a decode-only stack). Status is in
[STATUS.md](https://github.com/imazen/zenav1-aom/blob/main/STATUS.md); the C→Rust
map is in [PORTING.md](https://github.com/imazen/zenav1-aom/blob/main/PORTING.md).

## License

Dual-licensed: AGPL-3.0-only OR the Imazen commercial license
(`LicenseRef-Imazen-Commercial`). See
[LICENSE-AGPL3](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-AGPL3) and
[LICENSE-COMMERCIAL](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-COMMERCIAL).
Derived from upstream libaom (BSD-2-Clause + AOM Patent License 1.0).
