# The shipping path had no palette and no IntraBC — measured, wired, gated

**2026-09-10.** `aom_encode::key_frame::encode_key_frame` — the self-contained
entry zenavif calls — built its `PickFrameCfg` with `palette_costs: None` and
`intrabc: None`. It therefore ran **neither screen-content RD search on any
frame**, including frames whose header it writes with
`allow_screen_content_tools = 1` from its own ported detector.

The machinery was not missing. `palette_search.rs` and `intrabc_search.rs` have
been ported and byte-gated for months, and `aom-bench`'s `port_encode_with`
reaches both through `ToggleKnobs`. Only the shell's wiring was absent.

## Why no gate could see it

Every byte gate in `self_contained_key_frame.rs` drives `shim_encode_av1_kf`,
which hardcodes `enable_palette = 0, enable_intrabc = 0`
(`crates/aom-sys-ref/shim/dec_shim.c:612`, and the same in its `_sb128` and
`_tiles` twins). So the much-quoted **427/427 byte-identical** is a parity claim
against a **palette-disabled libaom**, and it was blind to both tools *by
construction* — not because they diverged, but because neither side had them.

aomenc's real ALLINTRA defaults turn both ON, gated only on the screen flag.

**This is the KB-42 shape on a new axis.** A gate can be green, exhaustive and
honest about what it measures, and still say nothing about a feature — because
the ORACLE was configured out of the question. Check what the oracle is
configured with before reading a parity count as coverage.

## What the gap was worth

libaom on **both** sides, so the number is the value of the TOOLS and not of
this port's RD (`crates/aom-bench/examples/screen_tools_gap.rs`, the committed
census-gated `Content::Screen` generator):

| cell | palette+IntraBC off | both on | gain |
|---|---:|---:|---:|
| 1024x768 cq20 s3 | 115,637 | 31,132 | **−73.1 %** |
| 1024x768 cq32 s3 | 76,839 | 24,168 | **−68.6 %** |
| 512x384 cq20 s3 | 22,698 | 6,898 | **−69.6 %** |
| 512x384 cq32 s6 | 12,864 | 5,337 | **−58.5 %** |

So the shipping path was leaving roughly **two thirds of the bytes** on the
table on UI content, silently, on every frame its own detector called screen.

**Read the sign carefully at the low-rate end.** At cq44–55 palette ALONE is
often *worse* (+21 % to +83 %): a palette block costs a colour table and a
per-pixel index map, and at a coarse quantizer ordinary transform coding is
cheaper. IntraBC is what carries those cells. "Turn palette on" is not
uniformly a win, and a table that reported only the aggressive-quality rows
would misrepresent it.

## The bug the matched oracle found

`uncompressed_header` SKIPS `loop_filter_params`, `cdef_params` and `lr_params`
entirely when `allow_intrabc` is set — and the three sub-header structs each
carry their own copy of that bit. `derive_frame_header` hardcoded all three to
`false`, and nothing propagated the post-search flip.

So the writer emitted **three syntax elements the decoder never reads**,
desynchronising the tile group that follows in the same `OBU_FRAME`. That is a
**corrupt stream**, not a larger one.

Its signature is worth keeping: a **constant +3 bytes over a byte-IDENTICAL
tile payload**. Sign-random, size-varying deltas are RD divergences; a constant
delta with an identical payload is a header-length bug. The common suffix was
31,114 of 31,132 bytes on the 1024x768 cell — every coded symbol agreed.

Fixed by propagating `p.allow_intrabc` into all three. 51/54 → **54/54**.

## What is gated now

`screen_content_tools_byte_match_real_aomenc` drives
`shim_encode_av1_kf_screen_content` with the **same two knobs the port is
given**, so a divergence is a divergence in the ported search rather than a
configuration mismatch. 54 cells (3 sizes × 3 quantizers × 3 speeds ×
checkerboard/UI), each asserting:

1. **byte-identity** to the matched oracle;
2. **non-vacuity** — the oracle's tools-ON stream must DIFFER from its own
   tools-OFF stream, so a cell the detector calls photographic cannot pass the
   way a deleted test passes;
3. a **decode round-trip** against the real C decoder plus this port's own, with
   identical pixels. KB-29 and KB-33 are the standing precedent that an IntraBC
   stream can be wrong in a way only a decoder sees — and this landing produced
   exactly that bug, so the leg is not ceremonial.

Wider probe (`screen_tools_gap`, the TSV beside this file): **24/24 cells
byte-identical** to libaom with both tools on, 256x256 through 1024x768.

## The 427-cell gate now sets both knobs false

That is **matching, not weakening**. On every detector-negative cell the two
lines are inert; on the `chk` checkerboard cells they are what keeps it a parity
test instead of a tools-on-vs-tools-off size comparison. The screen envelope has
its own gate with its own matched oracle, which is the right split: one gate per
oracle configuration.

## One divergence, measured and bounded

**IntraBC declines at coded-lossless.** C runs it there, dispatching the coeff
arm to `av1_pick_uniform_tx_size_type_yrd` instead of the recursive var-tx one
(`tx_search.c:3824`: `tx_mode_search_type == TX_MODE_SELECT && !xd->lossless[..]`).
The port has no INTER uniform-tx arm, so its IntraBC coeff path is var-tx-only
and fired a `lossless forces TX_4X4` assertion — a crossing nothing in the tree
had ever reached, because the shell never ran IntraBC and the bench never
crossed IntraBC with cq 0.

Declining the SEARCH is the bounded choice: a lossless frame reconstructs to the
source either way, so this can only cost SIZE on cq-0 screen content, never a
pixel — and it is a divergence rather than a refusal, so no caller-reachable
configuration is rejected. Closing it means porting an inter uniform-tx arm.

## Still open, named

* **Tools OFF on screen content** diverges slightly from a tools-off libaom
  (e.g. 512x384 cq20 s3: 22,730 vs 22,698). Small, sign-mixed, and it is NOT the
  default configuration any more. Not localized.
* `av1_determine_sc_tools_with_encoding` (PARITY C3) is still unported. Its two
  trial encodes turn screen tools ON when the detector said off — and now that
  the shell HAS palette, porting it would no longer be vacuous, which it would
  have been before this landing.
