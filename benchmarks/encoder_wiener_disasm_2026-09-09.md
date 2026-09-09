# The wiener convolve's lever, read off the disassembly: `vpmulld` where libaom has `vpmaddwd`

**2026-09-09. No code changed.** The last of the three levers the intrinsics
unblock (`encoder_intrinsics_unblock_2026-09-09.md`), scoped but **not
attempted** — this records what it is, so the next session starts from a
measurement rather than a hypothesis.

## The pairing, verified (two earlier audit rows were asymmetric, this one is not)

| | port | C | ratio | gap |
|---|---:|---:|---:|---:|
| wiener convolve | `restore::wiener::wiener_impl_v3` **40.9 ms** | `av1_wiener_convolve_add_src_avx2` **6.2 ms** | **6.60x** | **+34.7** |

**The worst ratio in the whole audit**, and the pairing is one symbol against one
symbol.

## What the shipped binary actually executes

`objdump` of `wiener_impl_v3`, 655 instructions:

| mnemonic | count | what it is |
|---|---:|---|
| `mov` / `lea` / `cmp` / `ja` | 108 / 71 / 53 / 40 | scalar addressing and bounds checks |
| `vpbroadcastd` | 51 | tap and constant splats |
| `vmovd` | 25 | |
| `vpaddd` | 18 | the accumulate |
| **`vpmulld`** | **16** | **the tap multiply — 8 horizontal + 8 vertical** |
| `vpmovzxwd` | 16 | the u16 -> i32 widen |
| `vpblendd` | 16 | |
| **`vpmaddwd`** | **0** | **libaom's primary instruction, absent here** |

**Two things follow, and the first corrects a comment in the source.**

* The `widen` closure's claim — *"LLVM: vpmovzxwd instead of 8 checked scalar
  loads"* — is **TRUE**: 16 of them, exactly one per widen site. That is not the
  problem.
* **The problem is `vpmulld`.** It is a 32-bit lane multiply: ~10 cycles latency,
  2 uops on Intel, one multiply per lane. libaom uses `vpmaddwd`: ~5 cycles,
  1 uop, and **two multiply-accumulates per lane pair**. So per tap-pair libaom
  does in one cheap instruction what the port does in two expensive ones plus an
  add — before counting that its lanes are `i16`, so it carries **16 output
  columns per register against the port's 8**.

## Why it was recorded as blocked, and why it is not

`CLAUDE.md` states magetypes' `madd_adjacent` does not unblock this, and the
reasoning is correct as far as it goes: `madd_adjacent` pairs ADJACENT LANES, so
a straight `src[0..16]` load pairs `(src[0],src[1])`, `(src[2],src[3])` — the
taps for output columns 0, 2, 4 … only.

libaom builds the right layout with **`_mm256_unpacklo_epi16(s0, s1)`** on two
source loads one sample apart, giving `(src[0],src[1]), (src[1],src[2]),
(src[2],src[3]) …` — the pairs for CONSECUTIVE output columns. That instruction
is now available (KB-PERF-21 is the precedent: raw `core::arch` intrinsics
compile inside a magetypes `_v3` body with `#![forbid(unsafe_code)]` in force).

So the shape is: two shifted `u16` loads -> `unpacklo`/`unpackhi` -> four
`vpmaddwd` against broadcast tap PAIRS -> three `vpaddd`, replacing eight
`vpmovzxwd` + eight `vpmulld` + seven `vpaddd` per pass.

## Why this session stopped short of doing it

Honest scoping rather than a blocker:

* it is **two** passes (horizontal into a strided `temp`, then vertical), each
  with bd-dependent rounding (`round_0`/`round_1`, the 15-arm `shr_round_by!`),
  a clamp to `1 << (bd + 1 + FILTER_BITS - round_0)`, and an overlap-back tail
  for `w` not a multiple of 8;
* it is on the **loop-restoration path, which is byte-gated**, and unlike the
  Hadamard — whose exactness argument is "wrapping adds and a lane permutation,
  no bound" — an i16 rewrite here introduces a **range obligation** at bd10/12
  that has to be derived, not asserted;
* the two landings this session succeeded because they were small enough to
  verify carefully. This one is not, and a rushed loop-restoration change is how
  a silent bitstream defect ships.

## Concrete handoff

1. Reuse KB-PERF-21's pattern verbatim: magetypes `_v3` body, raw intrinsics
   inside, `safe_unaligned_simd` reference loads, an `_scalar` tier that declines
   to the untouched scalar core, `cfg(target_arch = "x86_64")` so aarch64/wasm
   are unaffected.
2. Derive the i16 bound the way `xtask/audit_i16_fwd.py` does — exhaustively over
   the tap set, not by inequality — and gate on it at runtime like
   `try_fwd_col_pass` does, so bd12 declines rather than diverges.
3. The bite proof must be asymmetric: perturb one tap pair and expect
   `restore` differentials against the real exported C to fail while the rest
   stay green.
4. Expect roughly **−0.5 %** at the shipping preset if it converts at the ~70 %
   rate KB-PERF-21 achieved.
