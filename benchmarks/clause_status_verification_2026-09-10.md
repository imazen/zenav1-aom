# The clause table re-verified at HEAD, with counts — not cited, run

**2026-09-10, no code changed.** Every clause of the standing goal except (4) is
claimed met or capped in `CLAUDE.md`. Those claims were carried forward from the
sessions that established them; this re-runs their gates at HEAD and records
what they actually print, so the evidence is a measurement rather than a
citation.

`cargo nextest run --cargo-profile test-fast --workspace` over the contract
gates: **25 tests run, 25 passed, 0 failed.**

## (5) match the RD of C — byte identity is the evidence

```
self_contained_key_frame_byte_matches_real_aomenc: 427/427 cells byte-exact
coded_lossless_reconstructs_the_source_exactly:    248/248 cells decode to the
                                                   source exactly, on BOTH decoders
```

Byte identity is the strongest available RD evidence: the same bytes are the
same rate-distortion POINT, not merely a similar one. 427/427 holds across
cq 0..63, {mono, 4:2:0, 4:2:2, 4:4:4} x bd {8, 10, 12}, 16x16..8320x64, SB64 and
SB128, single- and multi-tile, `--cpu-used` 0..9, all four
(CDEF, loop-restoration) combinations.

Separately, `bd12_dispatch_tier_agreement::bd12_small_grid_byte_matches_real_aomenc`
passes — 18 bd12 cells byte-identical to real aomenc — and because
`just gate-landing` runs the whole suite twice (default dispatch and
`AOM_FORCE_SCALAR=1`), a byte-identity assertion IS a tier-agreement assertion.
Both legs were green this session.

## (2) a support contract that never lies

```
format surface:      72 ok, 0 refused, of 72
knob ranges:         90 ok, 0 refused, of 90
documented refusals:  8 pinned
```

The 8 refusals are pinned in BOTH directions: each must refuse, and the support
query must refuse whatever the encoder refuses. A query that accepted something
the encoder rejects — or vice versa — fails the gate.

## (3) no panics or refusals on inputs a caller can produce

```
encoder fuzz sweep: 600 inputs, 178 encoded, 245 unsupported, 117 plane-size,
                    60 sample-range, 0 limited, 0 panics
```

The reach split is the part that matters: 178 of 600 inputs reach a real encode,
so the sweep is testing the encoder and not just `validate_configuration`. Every
refusal class is reached, so none of them can be deleted later as unreachable.

`refusal_census::screen_shaped_tiny_cells_encode_rather_than_refuse` also passes
— the screen-content class the unported SCM trial was once thought to refuse
encodes instead, which is what moved that item from "refusal" (must close) to
"divergence" (capped).

## (1) a backend zenavif can select

In-repo half: `standalone_avif_parse_readback::standalone_stream_survives_avif_mux_and_zenavif_parse_readback`
passes — the port's own stream round-trips through a real AVIF container and
back through the independently-maintained `zenavif-parse`. The zenavif-side
gates (`aom_encode_backend.rs` 24/24 etc.) live in that repo and are not
runnable here.

## (4) encode time within libaom — THE ONE OPEN CLAUSE

**1.9620x** at 1024x1024 cq27 `--cpu-used 3`
(`encoder_clause4_shipping_preset_2026-09-10.md`). The bar is 1.5x. Closing it
needs **715.9 ms of the 1490.6 ms gap (48 %)**, which is several more sessions
of the kind of work this cycle did — not one landing.

## What this document is not

It is a VERIFICATION, not new capability. Nothing here moves a clause; it
converts four clause-table rows from "recorded as met" into "measured green at
HEAD, with the counts printed".
