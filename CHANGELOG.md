# Changelog

## Workspace

### [Unreleased]

### Changed

- **Consolidated the 13 DSP/entropy kernel crates into one `zenav1-aom-dsp`**
  (transform, quant, txb, cdef, restore, intra, loopfilter, dist, inter,
  convolve, recon, dispatch, entropy) — each is now a module, e.g.
  `aom_dsp::transform`, `aom_dsp::entropy`. Shrinks the release surface from 12
  publishable sub-crates to one. Byte-exactness unchanged (pure namespacing —
  only module paths moved); the differential gates stay green. (GitHub #2;
  20324ad, cf0541e, a9a995e, be7586b, c63c3f9, c51fdce, e57c31e)
- **Renamed every crate to the `zenav1-aom-*` prefix** (`zenav1-aom-dsp`,
  `zenav1-aom-decode`, `zenav1-aom-encode`, `zenav1-aom-sys-ref`,
  `zenav1-aom-bench`). Short `[lib] name`s (`aom_dsp`, `aom_decode`, …) are
  retained so interior `use aom_dsp::…` does not churn; only package names, dep
  keys, and CI/justfile `-p` args changed. (GitHub #3 Phase 2; 52be170)
- Publish flags corrected: `zenav1-aom-sys-ref` is now `publish = false` (was
  wrongly publish=default); `zenav1-aom-decode` / `zenav1-aom-encode` are now
  publishable (the facade re-exports them). End state: 4 publishable
  (`zenav1-aom`, `-dsp`, `-decode`, `-encode`) + 2 dev-only (`-sys-ref`,
  `-bench`). (52be170)
- Relicensed to `AGPL-3.0-only OR LicenseRef-Imazen-Commercial` — the standard
  Imazen dual license (LICENSE-AGPL3 + LICENSE-COMMERCIAL added). Upstream
  libaom LICENSE (BSD-2-Clause) and PATENTS (AOM Patent License 1.0) restored
  at the repo root; they continue to cover the upstream work this port derives
  from. We will release this port under MIT or the original upstream license
  if Imazen's 2026 AI + server costs are covered. (527852efc15a)

### Added

- **`zenav1-aom` facade crate** re-exporting `dsp` plus feature-gated `decode` /
  `encode` (both default). `default-features = false, features = ["decode"]`
  builds a decode-only stack (the encoder crate is never compiled) for
  size-sensitive / wasm consumers. (GitHub #2; 52be170)
- README.md (project summary, status pointers, licensing) and this changelog.
  (527852efc15a)
