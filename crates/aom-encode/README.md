# zenav1-aom-encode

The AV1 encoder for the pure-Rust, bit-exact
[libaom](https://aomedia.googlesource.com/aom) v3.14.1 port: the RD
partition/mode/tx search, forward transform + quantize + entropy coding, and
bitstream pack, over the [`zenav1-aom-dsp`](https://crates.io/crates/zenav1-aom-dsp)
kernels. The all-intra (usage=2) KEY-frame path byte-matches real `aomenc` across
`--cpu-used 0..9`; inter-frame encode is an early skeleton. `#![forbid(unsafe_code)]`,
no C toolchain, no `build.rs`.

Most consumers want the [`zenav1-aom`](https://crates.io/crates/zenav1-aom) facade.
Status is in [STATUS.md](https://github.com/imazen/zenav1-aom/blob/main/STATUS.md)
and the stills-parity ledger [PARITY.md](https://github.com/imazen/zenav1-aom/blob/main/PARITY.md);
the C→Rust map is in [PORTING.md](https://github.com/imazen/zenav1-aom/blob/main/PORTING.md).

## License

Dual-licensed: AGPL-3.0-only OR the Imazen commercial license
(`LicenseRef-Imazen-Commercial`). See
[LICENSE-AGPL3](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-AGPL3) and
[LICENSE-COMMERCIAL](https://github.com/imazen/zenav1-aom/blob/main/LICENSE-COMMERCIAL).
Derived from upstream libaom (BSD-2-Clause + AOM Patent License 1.0).
