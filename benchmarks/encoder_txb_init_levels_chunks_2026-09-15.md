# txb_init_levels v3 — as_chunks bounds-check removal — 2026-09-15

## Change

`crates/aom-dsp/src/txb/simd.rs::txb_init_levels_impl_v3` — every load and
store carved through `as_chunks{,_mut}` fixed-size views instead of
`slice[a..a+N].try_into()`: output columns become `&mut [[u8; stride]]`
(stride 8/12/20/36 by height), coeff input `&[[i32; 8|16|32]]`, tail pad a
`chunks_mut::<16>` zero pass. One up-front length preflight keeps the contract
(short buffers still reach the scalar path, which panics on them identically).
Companion fix: `aom-sys-ref/build.rs` `env!` -> `env::var` for
`CARGO_MANIFEST_DIR` (stale worktree-keyed build-script binary under a shared
target dir) + `perf_arms.sh` gives the base arm its own `CARGO_TARGET_DIR`.

## Paired callgrind, 196x196 cq27 cpu-used=3 (port arm, 10 reps)

| symbol | before | after | C `av1_txb_init_levels_avx2` |
|---|---:|---:|---:|
| `txb_init_levels_impl_v3` | 315.9M Ir / 274 per call | 178.8M Ir / 155 per call | 31.4M Ir / 112 per call (2 reps) |
| total | 17,491M | 17,126M | — |

−2.1 % total Ir; the residual 155 vs 112 is preamble (dispatch + preflight +
pad pass) amortized over small blocks.

## Band, 1024x1024 cq27 cpu-used=3 — WALL NULL

| arm | median | min | n | bytes |
|---|---:|---:|---:|---|
| base (`18c918d`) | 2445.36 ms | 2430.87 | 24 | 40237 |
| baseB (null) | 2446.32 ms | 2432.38 | 24 | 40237 |
| new | 2445.89 ms | 2434.83 | 24 | 40237 |

new vs base: paired median **+0.010 %**, 12/24 faster, **p=1**.0 — no wall
movement at the shipping cell (the kernel's share is smaller at 1024x1024 than
at the 196 profile cell; its memory pattern is unchanged). Landed as a strict
instruction-count/bit-identical reduction, same standard as quantize_fp and
the memset eliminations. Full data:
`encoder_txb_init_levels_chunks_2026-09-15.band1024s3.tsv`.
