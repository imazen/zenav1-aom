# zenav1-aom

Pure-Rust, bit-exact reimplementation of [libaom](https://aomedia.googlesource.com/aom)
v3.14.1 (the AV1 reference codec). This is the facade crate: it re-exports the DSP
kernels ([`zenav1-aom-dsp`](https://crates.io/crates/zenav1-aom-dsp)) plus the
feature-gated decoder ([`zenav1-aom-decode`](https://crates.io/crates/zenav1-aom-decode))
and encoder ([`zenav1-aom-encode`](https://crates.io/crates/zenav1-aom-encode)) as a
single dependency. `default-features = false, features = ["decode"]` builds a
decode-only stack.

`#![forbid(unsafe_code)]`. Depends on no C toolchain and no `build.rs`.

Status, the crate map, and the fresh-box test flow are in the
[project README](https://github.com/imazen/zenav1-aom#readme); the C→Rust
auditability map is in [PORTING.md](https://github.com/imazen/zenav1-aom/blob/main/PORTING.md).

## License

Dual-licensed: AGPL-3.0-only OR the Imazen commercial license
(`LicenseRef-Imazen-Commercial`). See
[LICENSE-AGPL3](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-AGPL3) and
[LICENSE-COMMERCIAL](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-COMMERCIAL).
Derived from upstream libaom (BSD-2-Clause + AOM Patent License 1.0).
