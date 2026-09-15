# sum_squares_2d_i16 v3 re-mirrored onto the real `aom_sum_squares_2d_i16_avx2` dispatcher — 2026-09-15

Cell: 1024x1024 cq27 `--cpu-used 3` (the preset zenavif ships), eprof_x86 `port`,
reps=1 per round, `nice -n 19`, rotated base/new/baseB. Band TSV:
`benchmarks/encoder_sum_squares_avx2_mirror_2026-09-15.band1024s3.tsv`.
base = `ad9ae21`; new = working tree.

## Result

```
base     median   2405.47 ms  min   2395.37  n=24  bytes=['40237']
baseB    median   2403.19 ms  min   2386.33  n=24  bytes=['40237']
new      median   2389.26 ms  min   2380.45  n=24  bytes=['40237']
new vs base   : paired median -0.699 %   22/24 rounds faster   p=3.588e-05
null (baseB vs base): paired median -0.135 %   18/24 rounds faster   p=0.02266
```

Clears the same-binary null and p < 0.05 — a real landing. Byte-identical
(40,237 B every round, both arms). Shipping-cell ratio: 2389.26 / ~1591 C ≈
**1.50x** — at the Gate-3 bar.

## Mechanism

Paired callgrind (196x196 cq27 s3, 10 reps) put `sum_squares_2d_i16` at ~233M Ir
against C's whole family (`aom_sum_squares_2d_i16_{4x4,4xn,nxn}_sse2` +
dispatcher) at ~47.5M — the port ran an autoversioned scalar loop with a serial
u64 accumulator and per-element bounds checks, ~330 Ir/call vs C's ~67.

New `#[arcane]` `sum_squares_2d_i16_impl_v3` (`dist/simd.rs`) mirrors the real
x86 dispatcher arm-for-arm: `4x4` and `4xn` SSE2 (`loadl`/`loadh` 8-byte rows),
`nxn_sse2` at width 8 (16-byte row loads, i32 `madd_epi16` accumulation
per 4-row group, `punpck` + i64 combine), `nxn_avx2` at width %16 (unaligned
32-byte loads, `vpmaddwd` + `vphaddd`-folded accumulation, sign-extended i64
total). C's scalar `u64` result is EXACT; the x86 kernels WRAP at i32 — the v3
mirror reproduces whichever semantics the C dispatcher would select per shape.

## Domain contract — the w=8 arm is alignment-gated

C's `nxn_sse2` rows go through `xx_load_128` == `movdqa` — an ALIGNED load.
Its real callers always satisfy `src` 16-byte-aligned + `stride % 8 == 0`
(`av1_pixel_diff_dist` offsets `src_diff` by `<< MI_SIZE_LOG2` on even blk_col
with power-of-two stride). An arbitrary `&[i16]` slice can violate it, and the
port cannot fault — so the v3 arm checks both conditions and falls to the
exact scalar loop otherwise (== C-c's result, which is also what the C
dispatcher computes for unroutable shapes). Verified unreachable-in-production:
C's own caller has the identical offset structure.

## Oracle

`sum_squares_diff` now oracles against the REAL exported
`aom_sum_squares_2d_i16_avx2` symbol whenever the v3 tier is live (scalar C
otherwise), across every dispatch arm — `(4,{4,8,16,64})`, `(8,{4,8,32})`,
`(16,{4,16})`, `(32,{8,32})`, `64x64`, `128x4`, and non-multiple fallbacks
`(16,5)`, `(12,8)`, `(6,10)` — with `i16::MIN` salting to exercise the
per-arm wrapping differences (4x4 sign-extends the i32 total; 4xn wraps across
all groups; nxn wraps within each 4-row group). The C-side calls use a
`#[repr(align(16))]` buffer and `stride % 8 == 0` on the w=8 arm — the movdqa
contract C's own callers satisfy. Misaligned / odd-stride w=8 inputs are
asserted to take the scalar fallback (== C-c).

## Numbers at the profile cell (196x196 cq27 s3, 10 reps)

| kernel | before | after | C |
|---|---:|---:|---:|
| sum_squares family (incl) | ~233M | ~99.8M | ~47.5M |
| — impl body | ~233M | 82.1M | ~32M kernels |
| — dispatcher | — | 17.7M (~25 Ir/call summon) | 15.3M |

708k calls both sides. Total Ir: 17.13G -> 17.00G (-0.8 %).

## Test-harness fix alongside (same commit)

Five encode `all`-harness differentials (`encode_block_coeffs_diff`,
`encode_block_full_diff`, `qm_forward_block_diff`, `xform_quant_diff`,
`xform_quant_optimize_diff`) oracle their `Fp`+no-qmatrix quantize step against
`ref_quantize_fp` — C-SCALAR — and legitimately failed once the port's v3
quantizer (`d719fd1`) mirrored C-avx2. Each now picks `ref_quantize_fp_avx2`
(the real kernel) when X64V3 is live, C-scalar otherwise, guarded by
`archmage::testing::lock_token_testing()`; `archmage` added to aom-encode
dev-deps with `testable_dispatch`. 652/652 green including all byte gates.
