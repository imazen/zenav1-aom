# KB-PERF-15 — `txb_init_levels` computed eight lanes in parallel and stored them one byte at a time

**Landed 2026-09-09.** Byte-identical output; **−0.30 %** at 512x512 and
**−0.32 %** at 1024x1024 (pooled over two independent bands — see the honesty
note below).

## How it was found — the instrument mattered more than the search

The memory class had been the 12x-vs-C outlier since the 1 MP re-profile, but it
was **unattributable**: release builds omit frame pointers, so `perf`'s inverted
callgraph could not name a single `memset` caller. Rebuilding the profiling
driver with `RUSTFLAGS="-C force-frame-pointer=yes"` and recording
`--call-graph fp` fixed that in one step, and the top caller was not what the
flat profile suggested:

    1.55%  __memset_avx512_unaligned_erms
            |--0.26%--0
            |          --0.20%--aom_dsp::txb::simd::__arcane_txb_init_levels_impl_v3
            |--0.11%--aom_dsp::transform::simd::try_inv_row_pass
            |--0.07%--aom_dsp::transform::simd::try_fwd_col_pass
            |--0.06%--aom_dsp::intra::__arcane_assemble_dir_edges_v4

**Frame pointers cost ~nothing on a profiling build and turn an unattributable
stage into a ranked list.** Do this before deciding a memory class is a
"programme" rather than a lever.

## The defect, and why the aggregate class ratio hid it

Checking that symbol against C like-for-like:

| | port | C |
|---|---:|---:|
| `txb_init_levels_impl_v3` | 183 ms | `av1_txb_init_levels_avx2` 41 ms |

**4.5x, +142 ms.** The 1 MP class table records the whole txb/trellis class at
**ratio 1.32 — the best of any class** — because `optimize_txb_core` is *faster*
than libaom's trellis and offsets this. **An aggregate ratio near parity does not
mean every member is near parity**; the class table is a ranking tool, not a
verdict on its contents.

Both sides are already SIMD, so this is not a missing-vector-path case. The port
computes `abs127` across eight `i32` lanes and then does:

    let arr = ...to_array();
    for (k, v) in arr.into_iter().enumerate() { out[c * 8 + k] = v as u8; }

— a stack round-trip and **eight bounds-checked single-byte stores** per vector.
libaom narrows with `_mm256_packs_epi32` + `_mm256_packus_epi16` and stores
8/16/32 bytes at once.

## What was and was not available

**magetypes 0.9.28 has no i32→u8 narrowing primitive.** Checked directly against
the crate source rather than assumed: it exposes `bitcast_*` and the `store*`
family, and no `narrow` / `pack` / `shrink` / `saturat*` of any kind — the same
vocabulary gap as the missing gather (KB-PERF-8) and the missing shuffle
(KB-PERF-10). So libaom's pack shape is not expressible here.

What *is* available is to build the eight bytes and store them as one run, which
is what this does — one 8-byte `copy_from_slice` per vector instead of eight byte
stores. In the `height == 4` branch the same rewrite also **absorbs the two
4-byte pad fills**, because `stride == height + TX_PAD_HOR == 8` exactly, so each
column's four levels and four pad zeros are one 8-byte run.

## Correctness

No arithmetic changed — the `abs127` lanes and the `as u8` narrowing are
untouched, and the module's existing bit-exactness argument (values are already
in `0..=127`) still carries the narrowing. Byte-identical output verified
directly on two cells before any timing: **39,694 B at 1024x1024 and 10,912 B at
512x512**. The 16-test txb differential suite against the real exported C is
green.

## Measurement, and an honest note on pooling

Two sha256-distinct binaries from one tree, arms **rotated** each round, a
same-binary null (`baseB`) in every band.

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 512x512 cq27 s0 | 3622.4 ms | 3613.3 ms | **−0.30 %** | 19/24 | 0.0066 | +0.00 %, 12/24, p=1.00 |
| 1024x1024, band A (n=16) | 10117.1 | 10064.2 | −0.49 % | 12/16 | 0.0768 | +0.01 %, 8/16 |
| 1024x1024, band B (n=30) | 10108.7 | 10075.1 | −0.25 % | 20/30 | 0.0987 | +0.00 %, 14/30 |
| **1024x1024, A+B pooled (n=46)** | | | **−0.32 %** | **32/46** | **0.0114** | +0.00 %, 22/46, p=0.88 |

**Neither 1 MP band reached p < 0.05 on its own**, and that is stated rather than
buried. Band A was underpowered at n=16 (the `new` arm's spread is consistently
~2 % against base's ~1 %, which is why); band B was run to n=30 for that reason
and still did not clear the bar alone. The two are independent replicates of the
same comparison with the same two binaries, so pooling them is meta-analysis
rather than p-hacking — and the check that matters is that **the pooled median
(−0.32 %) sits BETWEEN the two individual medians (−0.49 %, −0.25 %)**, so this
is not a favourable band being selected. The pooled null is flat at +0.00 %,
22/46, p=0.88.

The 512x512 band is significant on its own and agrees in both sign and magnitude.

## Not covered

* **The narrowing is still scalar.** A real `pack` would need a magetypes
  primitive that does not exist in 0.9.28; adding one (or moving to a version
  that has it) would let this kernel take libaom's shape and is the follow-up
  worth more than this landing.
* The other `memset` callers the frame-pointer profile named —
  `try_inv_row_pass` 0.11 %, `try_fwd_col_pass` 0.07 %,
  `assemble_dir_edges_v4` 0.06 %, `quantize_fp_impl_v3` 0.05 % — are untouched
  and are now a ranked list rather than an opaque class.
* One box, one content class, one speed, `--cpu-used 0`, x86-64.

## Gate status

`just gate-landing` green in full — `test-next` 1503/1503,
`test-next-scalar` 1503/1503, `census-gate` 4/4, `test-whereat` 4/4, all exit 0.
