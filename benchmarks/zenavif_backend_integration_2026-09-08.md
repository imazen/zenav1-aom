# Clause (1): zenavif backend selection — MEASURED 2026-09-08

The standing goal's first clause is *"a backend zenavif can select by default for
still images."* This is the first time it has been measured rather than asserted.
Headline: **the seam works against today's `main`; "by default" is blocked on
encode time and on nothing else.**

## What was measured

zenavif at `~/work/zen/zenavif`, branch `main`, `7b058bb`. Its `Cargo.toml`
pins `zenav1-aom-encode` at rev `a7b1ab13` — **34 commits behind** this repo's
`7b86ffc`, i.e. behind every landing of the last two days (the six zen
contracts, the sample-range fix KB-51, `AllocMode` KB-52, the cancellation
work). The question is whether the seam still builds and passes at HEAD.

Composed with the mechanism zenavif's own `Cargo.toml` documents for in-flight
work (its lines 145-148), so nothing in that repo was modified:

```
cargo <cmd> --features zenav1-aom-encode,encode \
  --config 'patch."https://github.com/imazen/zenav1-aom".zenav1-aom-encode.path="/home/lilith/work/zen/zenav1-aom/crates/aom-encode"'
```

| leg | result |
|---|---|
| `cargo check`, feature on, against `7b86ffc` | **clean** (7 pre-existing warnings in this repo, 0 errors) |
| `tests/aom_encode_backend.rs` | **22 passed / 0 failed** |
| `--lib` (whole crate, feature on) | 179 passed / **2 failed** / 1 ignored |

The 2 lib failures are `decode_av1::tests::decode_gain_map_from_avif_test_file`
and `::raw_obu_422_matches_the_container_path`. Both **reproduce without the
patch**, both are on the DECODE path this change does not touch, and both are
missing-fixture panics (*"No such file or directory ... (run: just
download-vectors)"* / `just download-linku`) rather than assertion failures. So
they are zenavif's own provisioning, not a regression from advancing the pin.

The 22 encode-backend tests include the ones that would catch a real break:
`aom_backend_flat_content_decodes_exactly_on_an_independent_decoder`,
`aom_cq0_encodes_and_reconstructs_the_coded_planes_exactly`,
`aom_backend_every_speed_decodes`, `aom_backend_encodes_10_and_12_bit_that_decode`,
plus four `gate_can_fail_*` anti-vacuity tests.

## The API-break risk that did not materialize, and why

This session added four variants to `KeyFrameError` (`Cancelled`, `LimitExceeded`,
`SampleRange`, `AllocFailed`) and a new `encode_key_frame_with` entry point. A
consumer that matched exhaustively on that enum would have broken. zenavif does
not: `encoder_aom.rs`'s `encode_key_frame_checked` formats it —
`.map_err(|e| at!(Error::Encode(format!("zenav1-aom key-frame encode: {e}"))))` —
so the additions are source-compatible. **That is luck, not design**, and it has
a cost: every refusal collapses into one `Error::Encode(String)`, so the
`category()` / `is_transient()` work this session landed is invisible at the
seam. Consuming it is a zenavif-side change (see "What remains" below).

## What "by default" would take, and what actually blocks it

Two flips, both in zenavif, neither of them integration work:

1. `Cargo.toml:557` — `default = ["avx512"]` would have to include
   `zenav1-aom-encode`.
2. `encoder.rs:230` — `#[default]` sits on `Av1Backend::Zenravif` (the rav1e
   fork, *"default, production-proven"*). It would have to move.

Neither is blocked by the seam, by the API, or by the supported-format scope.
Both are blocked by the same thing: **`benchmarks/encode_perf_vs_libaom_2026-09-08.md`
measured this encoder at 3.24x–4.03x libaom over byte-identical cells**, against
Gate 3's ≤1.5x bar. Nothing that is 4x the incumbent becomes a library's default.

**So clause (1) reduces to clause (4).** The remaining work on "selectable by
default" is encode-time work, not wiring. That reduction is the finding here; it
was not established before, and it retires "integrate with zenavif" as a
separate work item.

## What remains, stated plainly

* **Advance the pin.** `a7b1ab13` -> `7b86ffc` in zenavif's `Cargo.toml`, with
  the pin-history comment extended in the style that block already uses. Proven
  safe by the table above. Not done here — different repository, and the user
  has not authorized a zenavif commit.
* **Consume the error contract.** Map `KeyFrameError::category()` onto zenavif's
  `zencodec::CategorizedError` instead of `Error::Encode(String)`; thread
  `EncodeConfig`'s `limits` / `stop` through. The backend's module doc still says
  *"`encode_key_frame` takes no stop token"* (`encoder_aom.rs:665`), which
  `encode_key_frame_with` made stale.
* **Encode time.** Clause (4). The blocker.

## Reproduce

```
cd ~/work/zen/zenavif
cargo test --release --features zenav1-aom-encode,encode --test aom_encode_backend \
  --config 'patch."https://github.com/imazen/zenav1-aom".zenav1-aom-encode.path="/home/lilith/work/zen/zenav1-aom/crates/aom-encode"'
```
