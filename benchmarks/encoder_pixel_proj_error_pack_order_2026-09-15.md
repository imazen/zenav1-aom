# pixel_proj_error — packs-order restructure + truncating narrow (2026-09-15)

## Mechanism

`crates/aom-dsp/src/restore/pick.rs` `pixel_proj_error_impl_v3` mirrored C's
AVX2 shape but cost ~1.9x per call. Two causes, both in the i32→i16 narrowing:

1. **`permute4x64(packs_epi32(..))` merges into a ~9-insn expansion.**
   `core_arch`'s `_mm256_packs_epi32` is spelled `simd_imax(simd_imin())` +
   shuffle, not the `x86.avx2.packssdw` intrinsic. When the pack result feeds
   another shuffle (`permute4x64` for lane fixup), LLVM merges the two into a
   clamp+`pshuflw`/`pshufhw`/`shufps`/`permpd` chain per pack — ~9 insns where
   C emits `vpackssdw`+`vpermq` (2). The kernel has 2–3 packs per inner-loop
   iteration.

   Fix: keep every pack in `vpackssdw` lane order and permute the *other*
   operand. `packs(f_lo, f_hi)` now feeds `sub_epi16`/`unpack` directly —
   non-shuffle consumers, so no merge — while `d0`/`s0`/`u0` are permuted once
   per chunk into packs order (`[px0..3, px8..11, px4..7, px12..15]`). The
   `unpacklo/hi` pair restores i32-lane pixel order for the madds (`v0` = px
   0..7, `v1` = px 8..15 in order), so the final `packs(vr0, vr1)` lands back
   in packs order to match `d0p`/`s0p`. The accumulating `madd(e0,e0)` sum is
   lane-order-invariant anyway.

2. **LLVM keeps the (dead) clamps next to the `vpackssdw` it selects** —
   4 `vpmaxsd`/`vpminsd` per pack even after the shuffle folds (the same
   residue is visible in `txb_init_levels` and every other `packs_epi32` site;
   C's clang-compiled kernels emit bare `vpackssdw`, mem-folded).

   Fix: the packs on this kernel's documented domain never saturate — `|flt| <
   2^15` is C's own assert (pickrst.c:244-245) and `|vr| < 2^14` on the
   reachable domain (`|xq| <= 96`, `|f - u| < 2^17` → `|v| < 2^25`) — so a
   *truncating* narrow is bit-identical. `tpack!` spells it as 2 `vpshufb` +
   `vpunpcklqdq` (packs-order output); LLVM lowers it to
   `vpshuflw`/`vpshufhw`/`vshufps` with memory-folded loads and no broadcasts.
   Side benefit LLVM found itself: with only the low 16 bits consumed, the
   `(v + rounding) >> SHIFT` shifts now lower as `vpsrld` — `sar`/`shr` differ
   only in bits the trunc discards.

Both changes apply to all four arms (lowbd 1f/2f, highbd 1f/2f). Bit-exactness
is unchanged on every input the encoder can produce; the doc comment's stated
bounds are the argument, and the real-C differential exercises the domain.

## Measurement

Callgrind, 196x196 cq27 `--cpu-used 3`, 5 reps, same-protocol A/B against the
pre-change profile, identical call arcs (138+144+66+42 = 390 calls) both sides:

| | pixel_proj_error_impl_v3 inclusive Ir |
|---|---|
| base (`cg_ppe_port.out`) | 63,355,410 |
| packs-order restructure (`cg_ppe_port_new.out`) | 56,581,740 |
| + `tpack!` trunc narrow (`cg_ppe_port_tpack.out`) | 54,746,910 |

**−13.6 % kernel inclusive** (−10.7 % restructure, −3.2 % tpack), ~1.72M
Ir/rep on a ~1.81G Ir/rep cell → ~0.10 % of encode. C's
`av1_lowbd_pixel_proj_error_avx2` sits at ~32.9M for the equivalent calls, so
the per-call gap narrows from ~1.9x to ~1.6x — the residual is the intrinsic
safe-Rust SIMD lowering floor (no bare `vpackssdw` without `core_arch`'s
clamp spelling; `forbid(unsafe_code)` rules out `asm!`).

Below the ~0.25 % same-binary band resolution — landed on the
`txb_init_levels`/`acc_stat`/`boxsum_horz` precedent (strict instruction-count
+ structural reduction, wall-unmeasurable at this size).

## Correctness

- `pick_diff` 6/6 including `pixel_proj_error_matches_c` (real-C oracle,
  both bit-depths, all ep classes, odd sizes with scalar tails).
- `kernels_diff` `apply_selfguided_matches_c` / `wiener_convolve_matches_c`
  green.
- Shipping cell byte-identical: 1024x1024 cq27 s3 → 40,237 B, ~2,371 ms.
