# wiener convolve v3 re-mirrored onto the real `av1_highbd_wiener_convolve_add_src_avx2` — 2026-09-14

Cell: 1024x1024 cq27 `--cpu-used 3` (the preset zenavif ships), eprof_x86 `port`,
reps=1 per round, `nice -n 19`, rotated base/new/baseB. Band TSV:
`benchmarks/encoder_wiener_hbd_avx2_mirror_2026-09-14.band1024s3.tsv`.
base = `90d356c`; new = working tree.

## Result

```
base     median   2399.24 ms  min   2385.59  n=24  bytes=['40237']
baseB    median   2392.82 ms  min   2384.47  n=24  bytes=['40237']
new      median   2393.06 ms  min   2379.06  n=24  bytes=['40237']
new vs base   : paired median -0.214 %   14/24 rounds faster   p=0.5413
null (baseB vs base): paired median -0.174 %   16/24 rounds faster   p=0.1516
```

**Null on wall** — does not clear the same-binary null and p is not
significant. Landed anyway on the `txb_init_levels` precedent (`ad9ae21`):
the kernel-level instruction win is real and large, byte-identical output
(40,237 B every round, both arms), and the new body is oracled against the
REAL exported C-avx2 symbol rather than a generic formulation. Shipping-cell
wall ≈ 2393 ms vs C ≈ 1604 ms ≈ **1.49x** — still at the Gate-3 bar.

## Mechanism (kernel Ir, 196x196 cq27 s3, 5 reps)

| kernel | base | new | C |
|---|---:|---:|---:|
| wiener convolve | 36.3M/rep (`wiener_pass_madd_v3`) | **10.2M/rep** (`wiener_impl_v3`) | 7.5M/rep (`av1_highbd_wiener_convolve_add_src_avx2`) |

−72% vs base; vs C 4.8x -> **1.35x**.

The old body was a generic `i32x8` formulation: 9 bounds-checked loads +
8 unpack + 8 permute + 8 madd per 16 columns. The new `#[arcane]`
`wiener_impl_v3` mirrors the real highbd AVX2 kernel:

* horizontal: eight shifted-window loads `madd`ed directly against
  `[f(2k) f(2k+1)]` pairs — LLVM fused all eight loads into `vpmaddwd`
  memory operands; no unpack/permute at all;
* vertical: eight 16-column row loads + `unpacklo/hi_epi16` row-pair
  interleave + 8 madds + i32 even/odd sums; the `packs_epi32` lane
  permutation is self-consistent across `temp` so the dst store lands in
  natural order with no fixup;
* centre-tap `(in[c+3] << 7)` folded into `tap[3] += 1 << FILTER_BITS`,
  add-src offset folded into the rounding constant — same two folds C does.

Bounds-check structure: one contract preflight per call (window outside the
slices routes to the scalar port — same values, same panic-on-real-OOB), one
checked row view per row (`x0 <= w-16` makes every per-tile window statically
provable), one statically-sized `[u16; 7*128+16]` tstrip per vertical tile
(literal-offset row loads elide to bare `vmovdqu`). Remaining ~35% Ir excess
over C is the per-row view construction (~2 branches/row vs C's pointer
increments) plus per-call dispatch — the loop bodies are instruction-identical
to C's.

## Semantics that intentionally differ from C-scalar

* `madd_epi16` reads u16 samples as SIGNED i16: for adversarial values
  >= 32768 (outside every valid bit depth) this kernel computes what
  C-avx2 computes, which differs from the unsigned-widening scalar body.
  On the reachable domain (<= (1<<bd)-1) all tiers are identical — the
  direct C-avx2 differential covers the full-u16 adversarial domain.
* C-avx2 stores 16 columns unconditionally — for `w % 16 == 8` it writes 8
  columns past `w` into the next unit's dst region. The port keeps its
  no-overwrite contract via overlap-back tails (`x0 = min(xs, w-16)`);
  outputs are pure functions of the window so recomputed columns store
  identical values. `kernels_diff` asserts whole-buffer no-overwrite.

## Oracle

New `wiener_avx2_diff.rs` compares the v3 tier against the REAL exported
`av1_highbd_wiener_convolve_add_src_avx2` (via `shim_wiener_convolve_hbd_avx2`)
over bd 8/10/12, widths 8..=128 step 8 (including `w % 16 == 8` tails), odd
heights, random valid symmetric filters, chroma `tap0 == 0`, and full-u16
adversarial pixels — logical `w x h` region only, because of C's tail stores.
Existing scalar-C (`kernels_diff::wiener_convolve_matches_c`) and
tier-agreement (`wiener_simd_diff`) tests stay green; `AOM_FORCE_SCALAR` leg
unchanged.
