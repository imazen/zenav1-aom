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

Measured reach in the configuration **CI actually runs** — default seed,
60 000 iterations, final tree:

| outcome | count | share |
|---|---:|---:|
| decoded a full frame | 5 934 | 9.9 % |
| rejected at frame/tile level | 24 118 | 40.2 % |
| rejected at/before the header parse | 29 948 | 49.9 % |

**50.09 % deep reach**, one input in ten decoding a whole frame. The floor is
pinned at 10 % — five times under the measurement, so ordinary corpus churn
cannot trip it, but a mutator or seed change that collapses the sweep into a
header-parser test will. The run is deterministic in `(seed, iterations)` — the
PRNG is seeded and the decoder is bit-exact — so the histogram reproduces on any
box, which is what makes a hard floor safe here rather than flaky. The 500 000-
iteration seed-101 run on the final tree measures **50.35 %** (50 208 decoded /
201 534 deep / 248 258 shallow), so the CI-sized sample is not flattering the
number.

## 3. What the sweep found: nothing

| mutator | code under test | seeds | iterations | distinct panics |
|---|---|---|---:|---:|
| pre-existing | pre-conversion | 1, 2, 3, 7, 11 | 1 250 000 | 0 |
| widened | pre-conversion | 7, 101, 101 | 550 000 | 0 |
| widened | post-conversion | default, 101, 202 | 1 060 000 | 0 |

Every run under `--profile test-fast`, which keeps `debug-assertions` and
`overflow-checks` ON — so these also cover debug-only arithmetic-overflow
panics, not just explicit ones. **2.86 M mutated inputs, no panic.**

A side result worth keeping: the seed-101 500 000-iteration run decoded
**exactly 50 208 full frames before the conversions and exactly 50 208 after**.
The guards changed which failures are reported and how, and moved nothing that
decodes. Seed 202 on the final tree: 49 906 decoded, 50.17 % deep, 0 panics.

The mutation-reachable panic surface is clean. That is a real result, and it
bounds what fuzzing can still contribute here: the remaining sites need
reasoning, not iterations. The next two sections are that reasoning.

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
in the build that ships. Those two stay panics — their callers own the contract
— but they are now *named* panics that say what the caller did wrong, and they
cost nothing: the array bounds check they route through was already being paid.

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
cargo test --profile test-fast -p zenav1-aom-decode -p zenav1-aom-dsp   459 passed / 0 failed
cargo test --profile test-fast --workspace                              973 passed / 0 failed
AOM_FORCE_SCALAR=1  (decode + dsp, +1 new test)                         460 passed / 0 failed
cargo check -p zenav1-aom-decode --features whereat --all-targets       clean
```

The workspace run includes `tests/real_bitstream.rs` — the **live C-oracle
differential**, 15 tests asserting byte identity against in-process libaom
v3.14.1 across 4:2:0 / 4:4:4 / 4:2:2 / monochrome at 8, 10 and 12 bits — plus
`tests/conformance_corpus.rs` and the encoder's e2e byte gates (which consume
the changed `aom-dsp`).

One more reachability question got an exhaustive answer rather than an
argument. `av1_inv_txfm2d_add_{into,u8_into}` assert on `(tx_type, tx_size)`
pairs with no kernel — `TXFM_TYPE_LS` has holes because only DCT is defined at
64 points — and the decoder feeds both arguments from the bitstream. The new
`aom-dsp/tests/inv_txfm_decodable_pairs.rs` enumerates every `(tx_size,
is_inter, reduced)` state the decoder can be in, takes every `tx_type` that
state's ext-tx set marks decodable, and requires a kernel for each:

```
314 decodable (tx_size, tx_type, is_inter, reduced) selections, all with kernels;
111 of the 304 (tx_size, tx_type) pairs have NO kernel and none is selectable
```

So that assert is unreachable from a bitstream — and the 111 kernel-less pairs
make the constraint bite, so the test is not passing vacuously. It is left as
an `assert!` (a provable invariant, named) with the proof now executable.

## 7. One thing this audit got wrong, and how

Mid-audit I concluded the port **over-rejects** 4:2:2: `decode_partition`'s
guard fires on any `subsize` whose chroma size is `BLOCK_INVALID`, whereas the
`decode_mbmi_block` check quoted in §4 is gated on `bsize >= BLOCK_8X8` — so a
conformant 4:2:2 stream with a `BLOCK_4X8` block (`PARTITION_VERT` on an 8x8)
would be rejected by us and accepted by libaom. That reasoning was sound and
the conclusion was false, because libaom has a **second** check I had not read.
`decodeframe.c:1359-1371`, in `decode_partition`, verbatim:

```c
  // Check the bitstream is conformant: if there is subsampling on the
  // chroma planes, subsize must subsample to a valid block size.
  const struct macroblockd_plane *const pd_u = &xd->plane[1];
  if (get_plane_block_size(subsize, pd_u->subsampling_x, pd_u->subsampling_y) ==
      BLOCK_INVALID) {
    aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                       "Block size %dx%d invalid with this subsampling mode", ...);
  }
```

Ungated on size, on `subsize`, in the same place as the port's. **The port
matches C; there is no over-rejection.** libaom makes the `BLOCK_INVALID` call
in three separate places — invalid `subsize`, invalid chroma `subsize` (both in
`decode_partition`), and invalid chroma `bsize` at `>= BLOCK_8X8` (in
`decode_mbmi_block`). The port already had the first two as rejections; this
work converts the third from an `assert_ne!`, and adds the deeper txb-loop
guard as defence in depth.

Recorded because the near-miss is the point: reasoning from *one* citation in
the C reference produced a confident, wrong conclusion about a conformance
divergence, and the only thing that caught it was going back and reading the
surrounding function. The oracle is the whole file, not the line you found
first — which is also why every claim in §6 is a suite result rather than an
argument.

## 8. What this does NOT establish

* **The encoder's panic surface is untouched.** 126 sites in `aom-encode`,
  reviewed only far enough to confirm none is bitstream-reachable
  (`set_q_index`, `av1_build_quantizer`, `fill_tx_type_costs` and the rest of
  the flagged sites are encoder-only entry points).
* **No claim that the decoder is panic-free.** The claim is narrower and
  measurable: no panic escapes the two public decode entry points over the
  iteration counts in §3, at the deep-reach fraction in §2, and the four sites
  in §4 no longer panic on the condition the C oracle calls corrupt.
* **No timing claim of any kind.** Everything ran under `nice -n 19`, which on
  Darwin is background QoS; see the `.meta`.
