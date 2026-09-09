# Wiener, costed in uops before writing any code — and the OBVIOUS madd version is a near-wash

**2026-09-09. No code changed.** This corrects the estimate in
`encoder_wiener_disasm_2026-09-09.md` (*"expect roughly −0.5 %"*) and, more
usefully, identifies the shape that would measure NULL — which is the trap two
of this session's three kernels already fell into.

## First, a concern I raised and then refuted: there is NO range obligation

The disassembly record said an i16 rewrite *"introduces a range obligation at
bd10/12 that has to be derived"*. **That is wrong, and the reason matters:**
`_mm256_madd_epi16` takes i16 inputs and accumulates in **i32**. Nothing is
narrowed.

* horizontal inputs are samples `<= (1 << bd) - 1 <= 4095` — inside i16 by a
  factor of 8 — and taps are already `i16`;
* vertical inputs are the intermediate, clamped to
  `(1 << (bd + 1 + FILTER_BITS - round_0)) - 1`. `conv_params_wiener` raises
  `round_0` exactly when `bd + FILTER_BITS - round_0 + 2 > 16`, which pins that
  ceiling at **32767 — i16::MAX exactly** — at every bit depth;
* four madd results sum to at most ~33.5 M, comfortably inside i32.

So the kernel is sound at bd 8/10/12 with **no runtime gate and no audit**. The
earlier caveat conflated "use i16 lanes" with "narrow the accumulator".

## The costing, per 8 output columns of the horizontal pass

Read off the shipped binary plus Intel uop counts (`vpmulld` 2 uops / ~10 cy,
`vpmaddwd` 1 uop / ~5 cy, `vpmovzxwd`-from-memory 1, unpack 1, `vperm2i128` 1):

| variant | instructions | **uops per 8 columns** |
|---|---|---:|
| **today** — 8x `vpmovzxwd` + 8x `vpmulld` + 8x `vpaddd` | 24 | **32** |
| **naive 128-bit madd** — 8 loads + 8 unpack + 4 `vinserti128` + 4 madd + 3 add | 27 | **27** |
| **256-bit, 16 columns at a time** — per 16 cols: 8 loads + 8 unpack + 8 `vperm2i128` + 8 madd + 6 add | 38 / 16 cols | **19** |

**The naive version — swap `vpmulld` for `vpmaddwd`, keep 8 columns — is a 16 %
uop reduction.** That is at or under the resolution of a 24-round band on this
box (its nulls run 0.04–0.15 %), so it would very likely measure NULL. It is
exactly the shape someone reading "use madd like libaom does" would write.

**The reason it fails is the same one that sank the edge filter**: the
per-iteration setup (the unpack + lane-fix to build interleaved pairs) is fixed
cost, and 8 columns is too little to amortise it. Going to 16 columns spreads
the same setup over twice the work and takes the reduction to **~40 %**, which
is worth roughly **−0.3 to −0.4 %** at the shipping preset — not the −0.5 %
previously guessed, and only for the 16-wide shape.

## Why 16 columns costs a permute, and why libaom does not pay it

The pairs for consecutive output columns come from two source loads one sample
apart, interleaved: `unpacklo(s0, s1)` gives `(src[0],src[1]), (src[1],src[2])
…`. On 256-bit registers AVX2's unpack works **within each 128-bit half**, so
`unpacklo` yields columns 0-3 AND 8-11, `unpackhi` yields 4-7 and 12-15 — the
columns arrive interleaved between halves and need a `vperm2i128` pair to
straighten.

**libaom does not pay this because its source is `u8`**: a 256-bit load holds 32
samples, so after widening to i16 pairs it gets 16 output columns per madd with
the halves already in the right place. The port holds planes as `u16` at every
bit depth, so it gets half the samples per load and must fix the lanes.

That is the **u16-at-bd8 structural root** the lane-width audit named, showing up
here as a concrete instruction rather than as a general claim: the port can close
most of wiener's 6.60x with the 16-wide madd shape, but the last factor of ~2 is
the plane representation and is not reachable from inside this kernel.

## Handoff, revised

1. Implement the **16-column** shape only. Do not implement the 8-column one; it
   is a documented near-wash.
2. Requires `w >= 16`; the existing overlap-back tail (`x0 = xs.min(w - 8)`)
   generalises to `min(w - 16)` with an 8-wide path retained for `w == 8`.
3. **No bound to derive and no runtime gate** — see above. That is a real
   simplification over what the previous record claimed.
4. Same pattern as KB-PERF-21: magetypes `_v3` body, raw `core::arch` inside,
   `safe_unaligned_simd` loads, `_scalar` tier declining to the untouched scalar
   core, `cfg(target_arch = "x86_64")`.
5. Expect **−0.3 to −0.4 %**, and size the band for it: at n=24 this box's null
   is 0.04–0.15 %, so n=24 is marginal and n=48 is safer.
