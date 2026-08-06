# The decoder's crafted-bitstream panic surface (2026-08-06)

This decoder ships into zenavif's untrusted AVIF path. A panic there is a
denial of service, so a `panic!` / `assert!` / `unwrap` reachable from a
crafted bitstream is a **defect**, not a style question. This is the audit,
what it found, and what the numbers actually support.

Provenance, box, exact commands and what is *not* measured:
[`decoder_panic_surface_2026-08-06.meta`](decoder_panic_surface_2026-08-06.meta).

---

## 1. What the inventory looked like

Static scan of non-`#[cfg(test)]` code in the four shipping crates
(`~/tmp/panics/scan2.py`, counting `unwrap( expect( panic! unreachable! todo!
unimplemented! assert!*`, excluding `debug_assert*`):

| crate | assert-family | panic!/unreachable! | unwrap/expect |
|---|---:|---:|---:|
| aom-decode | 12 | 1 | 10 |
| aom-dsp | 24 | 22 | 35 |
| aom-encode | 55 | 18 | 53 |

The encoder's surface is out of scope here: it is a **trusted-input** API, and
none of its panics are reachable from a bitstream. Of the decode-reachable
sites, most are one of two legitimate shapes the audit deliberately did NOT
churn:

* **fixed-array `try_into().unwrap()`** — `buf[i..i+8].try_into().unwrap()`
  after a proven-in-range slice. This is the bounds-check-elimination idiom the
  hot kernels are built on; it compiles to nothing.
* **provable invariants named with a reason** — e.g.
  `unreachable!("invalid partition type {p}")` where `p` comes from
  `read_partition` on a CDF whose alphabet is the 10 partition types the match
  already covers exhaustively. Replacing those with error paths adds dead code
  and hides which facts the decoder actually relies on.

Several attacker-reachable classes had already been converted before this pass
(the `mark_corrupt` poison channel): unsupported interp filter, out-of-range
`segment_id`, intraBC DV validity, film-grain point counts, the frame-dimension
DoS ceiling. Those are the committed POCs under `fuzz/regression/`.

## 2. The discovery mechanism was green for the wrong reason

`tests/fuzz_sweep.rs` mutates the committed seeds and asserts no public decode
entry panics. It was green — but its mutation ops were bit flips, truncation,
length-field corruption, splices and byte insert/delete. Those leave a
**mostly-valid arithmetic stream**: the symbol decoder stays near the states a
real encoder produced, so the partition / tx-size / palette / coefficient
machinery is barely exercised on hostile values.

Two ops were added:

* **hostile tile payload** — keep the leading header bytes, replace the whole
  tail with PRNG bytes, so `OdEcDec` walks arbitrary symbol sequences.
* **payload extension** — append PRNG bytes so the range decoder keeps finding
  readable bytes instead of latching `OD_EC_LOTS_OF_BITS` at the real tile end.

And, because "no panic" over inputs the OBU parser rejected is not evidence
about the decoder, `probe()` now classifies every input as decoded / deep-err /
shallow-err, prints the histogram, and **fails** below `MIN_DEEP_REACH_PPM`.

Measured reach on the current corpus + mutator (seed 101, 500 000 iterations):

| outcome | count | share |
|---|---:|---:|
| decoded a full frame | 50 208 | 10.0 % |
| rejected at frame/tile level | 201 149 | 40.2 % |
| rejected at/before the header parse | 248 643 | 49.7 % |

**50.3 % deep reach.** The floor is pinned at 10 % — five times under the
measurement, so ordinary corpus churn cannot trip it, but a mutator or seed
change that collapses the sweep into a header-parser test will.

## 3. What the sweep found: nothing

| mutator | seeds | iterations | distinct panics |
|---|---:|---:|---:|
| pre-existing | 1, 2, 3, 7, 11 | 5 × 250 000 | 0 |
| widened | 7, 101 (+ the run below) | see §5 | 0 |

The mutation-reachable panic surface is clean. That is a real result and it
bounds what fuzzing can still contribute here — the remaining sites need
reasoning, not iterations.

## 4. What reasoning found: the chroma `BLOCK_INVALID` asserts

`av1_ss_size_lookup` (common_data.c:17) has no valid chroma plane size for some
luma shapes. libaom's `decode_mbmi_block` (decodeframe.c:393-401) treats that
as **untrusted-input rejection**, not as an invariant:

```c
  if (bsize >= BLOCK_8X8 && (ss_x || ss_y)) {
    const BLOCK_SIZE uv_subsize = av1_ss_size_lookup[bsize][ss_x][ss_y];
    if (uv_subsize == BLOCK_INVALID)
      aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                         "Invalid block size.");
  }
```

The port had `assert_ne!` at the same two places, justified in a comment by
*"the roundtrip never produces them"*. **What our own encoder emits says
nothing about what a crafted bitstream can reach** — that is the wrong warrant
for a decoder, and it is the one the C oracle explicitly declines to rely on.

Converted (see the commit for detail):

| site | was | now |
|---|---|---|
| `aom-decode` `decode_block` | `assert_ne!` | `mark_corrupt` + unwind, C's check at C's place |
| `aom-decode` chroma txb loop | `assert_ne!` | `mark_corrupt` + unwind (also covers the sub-8x8 shapes C's `>= BLOCK_8X8` gate exempts) |
| `aom-decode::max_uv_txsize` | `debug_assert_ne!` | named panic + `# Panics` |
| `aom-dsp` loopfilter `max_uv_txsize` | `debug_assert_ne!` | named panic + `# Panics` |

The two `debug_assert`s are the more insidious shape: **compiled out in
release**, where the same input then died as a bare `MAX_TXSIZE_RECT_LOOKUP`
"index out of bounds" — a panic either way, with the diagnostic removed exactly
in the build that ships. Those two stay panics (their callers own the
contract), but they are now *named* panics that say what the caller did wrong,
at the cost of the bounds check that was already there.

**Byte-inert on conformant streams.** `decode_partition` already turns the only
reachable case away one frame up, and the new
`block_invalid_chroma_sizes_occur_only_at_422_and_440` test pins why: of the
88 (bsize, ss_x, ss_y) combinations, **16 are `BLOCK_INVALID` and all 16 sit in
the ss=(1,0) (4:2:2) and ss=(0,1) (4:4:0) columns**. 4:4:0 cannot be coded by a
sequence header, so 4:2:2 is the only reachable hole — which is what makes the
existing 4:2:2-scoped guard sufficient, and what a future table edit would
silently invalidate without that test.

## 5. An error message that hedged between two different failures

```
corrupt frame header (bit-reader error / out-of-range syntax value)
```

`ReadBitBuffer::error` was one bool for two conditions: the reader ran past the
end of the payload, and a header parser found a readable-but-out-of-spec value.
Those are **different failures for a consumer** — a short file versus a corrupt
one — and `DecodeError` already distinguishes `Truncated` from `Malformed`, so
the seam was collapsing information the type system was carrying.

`mark_syntax_error(field)` now records *which* field was out of range; the
frame-header site returns `Truncated(..)` for an overread and
`Malformed("frame header syntax out of range: <field>")` otherwise. The split is
measured live, not asserted: 180 of 182 prefixes of a decoding seed report
`truncated`, and the committed film-grain POC reports `malformed` naming
`num_y_points`.

## 6. Verification

Behaviour-preservation is proven by the suite, not by inspection:

```
cargo test --profile test-fast -p zenav1-aom-decode -p zenav1-aom-dsp -j 4
  => 459 passed / 0 failed
```

including `tests/real_bitstream.rs` (the **live C-oracle differential** — 15
tests, ~200 s, byte-identity against in-process libaom v3.14.1 across 4:2:0 /
4:4:4 / 4:2:2 / monochrome at 8, 10 and 12 bits) and
`tests/conformance_corpus.rs`.

## 7. What this does NOT establish

* **The encoder's panic surface is untouched.** 126 sites in `aom-encode`,
  reviewed only far enough to confirm none is bitstream-reachable
  (`set_q_index`, `av1_build_quantizer`, `fill_tx_type_costs` and the rest of
  the flagged sites are encoder-only entry points).
* **No claim that the decoder is panic-free.** The claim is narrower and
  measurable: no panic escapes the two public decode entry points over the
  iteration counts in §3, at the deep-reach fraction in §2, and the four sites
  in §4 no longer panic on the condition the C oracle calls corrupt.
* **A separate finding, not fixed here.** `decode_partition`'s 4:2:2 guard
  fires on *any* subsize whose chroma size is `BLOCK_INVALID`, while C gates
  its equivalent check on `bsize >= BLOCK_8X8`. A conformant 4:2:2 stream
  containing a `BLOCK_4X8` luma block (`PARTITION_VERT` on an 8x8) would
  therefore be **rejected by the port and accepted by libaom**. That is an
  over-rejection, not a panic or a corruption, and the conformance corpus has
  no 4:2:2 vector that reaches it — so it is recorded here rather than changed
  on inspection. Fixing it means moving the check to `decode_block` with C's
  size gate and proving the widened acceptance decodes byte-identical to the
  oracle, which needs a 4:2:2 sub-8x8 stream the corpus does not have.
