# What magetypes actually offers (0.9.29) — checked, not remembered

Written 2026-09-09 because **three KB-PERF entries and one commit message state
this wrongly**, in two opposite directions, and each restatement made the wrong
version look better established. Check here, and re-check the crate, before
concluding a kernel shape is unavailable.

## The rule that produced the errors

Every wrong claim came from grepping the crate for the names I expected —
`shuffle`, `permute`, `pack`, `gather` — and concluding from their absence.
The lane-movement primitives are there under **different names**
(`interleave_*`, `transpose_*`), and I never grepped for those.
**Grep for the OPERATION's shape, not for one vendor's spelling of it**, and
grep the version in `Cargo.lock` rather than the newest on crates.io
(KB-PERF-7 records that failure separately).

## INTEGER lanes — what is and is not there

| want | available? |
|---|---|
| `blend` (lane select) | **yes** |
| `madd_adjacent` (`lane k = a[2k]b[2k] + a[2k+1]b[2k+1]`) | **yes, 0.9.29** (not 0.9.28) |
| `narrow_saturating_*` (i32→i16/u16, i16→i8/u8) | **yes, 0.9.29** (not 0.9.28) |
| `widen_low` / `widen_high` | **yes, 0.9.29** |
| `shl_uniform`, `shr_arithmetic_uniform`, `shr_logical_uniform` (runtime shift) | **yes, 0.9.29** |
| `saturating_add/sub`, `abs_diff`, `sum_abs_diff`, `reduce_add_u32` | **yes, 0.9.29** |
| shuffle / permute / swizzle | **NO** |
| interleave / unpack | **NO for integer lanes** |
| transpose | **NO for integer lanes** |
| gather | **NO**, in either version |

## FLOAT lanes — the part the earlier claims missed

`f32x4` and `f32x8` **do** have `interleave_lo`, `interleave_hi`, `interleave`,
`transpose_4x4`, `transpose_8x8` (+ `_copy` / `_repr` forms), with an x86_v3
specialisation. So the blanket statement *"magetypes has no shuffle, permute or
unpack — only `blend`"* (KB-PERF-10, repeated in KB-PERF-15 and the 0.9.29 bump
record) is **wrong as written**. It is right only of integer lanes.

**Whether those are usable for integers by bitcasting is UNTESTED.** Lane
movement is not arithmetic, and `bitcast_f32x8` exists, so it is plausibly
bit-preserving — but it would need its own differential across every tier
(the generic fallback round-trips through `f32` arrays), and it moves **32-bit**
lanes, so it does not supply the **i16** transpose the Hadamard wants anyway.

## Consequences for the standing entries — corrections

* **KB-PERF-10 (Hadamard)** — its conclusion **stands**: an in-register **i16**
  8x8 transpose is not expressible, so the two-pass Hadamard cannot be
  vectorised end to end. Its *stated reason* ("no shuffle/permute/unpack at all,
  only blend") is too broad; the accurate reason is "no INTEGER lane-movement
  primitive, and the float `transpose_8x8` is 32-bit-lane".
* **KB-PERF-15 (`txb_init_levels`)** — its 0.9.28 conclusion stands, but
  **0.9.29 adds `narrow_saturating_i32_to_i16` / `_i16_to_u8`**, so its named
  follow-up (a real pack instead of building an 8-byte run) is **now
  available**. This is the most concrete unblocked item.
* **The 0.9.29 bump record and commit message are WRONG about wiener.** They say
  `madd_adjacent` unblocks `wiener_impl` (the 7.6x, ~167 ms). It does not:
  `madd_adjacent` pairs **adjacent lanes**, so a convolution whose lanes are
  output columns needs `src[j], src[j+1]` interleaved *per lane* — which libaom
  builds with `_mm256_unpacklo_epi16` and which **has no integer equivalent
  here**. Both convolution passes have this shape, and offsetting the loads does
  not avoid it (it yields partial sums at stride 2 that then need a shuffle to
  recombine). **Do not start that work expecting it to close.**
* **KB-PERF-8 (SGR gather)** — stands unchanged; `gather` is absent in both
  versions. Its blend-chain shift select, however, can now use
  `shr_logical_uniform`.

## Still genuinely blocked on absent primitives

Wiener's i16 convolution, `acc_stat_line`'s i16 fold, and the full-SIMD
Hadamard — all three need integer interleave/unpack, which does not exist at
0.9.29. Closing them means a magetypes contribution, not a version bump.
