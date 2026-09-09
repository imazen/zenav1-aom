# Ranking at HEAD after the 2026-09-09 cycle (KB-PERF-9..15 + the 0.9.29 bump)

Re-taken because §14 requires it: eight landings moved the table, and the
opening table in `encoder_x86_reprofile_1024_2026-09-09.md` is now stale for
ranking purposes. Same cell, same method (`perf record -F 499 -g` on
`examples/eprof_x86`, 1024x1024 cq27 `--cpu-used 0`, symbol shares x each arm's
own wall).

    port 10128 ms    C 4222 ms    ratio 2.399    gap 5906 ms
    (session start: 10582 / 4212 / 2.513 / 6370)

| class | port | C | ratio | gap | % of gap | was |
|---|---:|---:|---:|---:|---:|---:|
| **transform** | 2991 | 728 | 4.11 | **+2263** | **38.3 %** | +2440 |
| rd-driver | 1379 | 429 | 3.21 | +950 | 16.1 % | +1023 |
| loop-restoration | 1179 | 278 | 4.24 | +901 | 15.3 % | +900 |
| intra-pred | 898 | 426 | 2.11 | +473 | 8.0 % | +514 |
| txb/trellis | 1944 | 1480 | **1.31** | +464 | 7.9 % | +483 |
| memory | 518 | 70 | 7.43 | +448 | 7.6 % | +486 |
| quantize | 536 | 152 | 3.52 | +384 | 6.5 % | +340 |
| distortion | 407 | 174 | 2.35 | +234 | 4.0 % | +295 |
| entropy/pack | 72 | 12 | 5.87 | +60 | 1.0 % | +60 |

**Quote the ratio with its C median.** 2.399 here against a C arm of 4222 ms;
the C arm ranged 4140.9–4222.3 across this session's bands, ~2 %, so ratio
differences under ~0.02 between records are not resolvable. The paired
port-vs-port deltas in each KB-PERF entry are the solid numbers.

## What did NOT move, and why that is the useful part

**loop-restoration is unchanged at +901 ms** — nothing this cycle touched it,
and its largest item is blocked: `wiener_impl_v3` **192.9 ms vs
`av1_wiener_convolve_add_src_avx2` 25.5 ms (7.6x, the worst single ratio
anywhere)**, because libaom convolves in i16 with `madd_epi16` while the port
uses `i32x8` with a multiply and an add per tap. Closing it needs an **integer
interleave**, which magetypes does not have at 0.9.29 — see
`docs/MAGETYPES_VOCABULARY.md`, and note that the 0.9.29 bump commit's claim to
unblock this is **wrong and corrected there**.

**txb/trellis stays the best class at ratio 1.31** — `optimize_txb_core` is
genuinely faster than libaom's trellis. It also still contains
`txb_init_levels` at ~4x, which is what KB-PERF-15 addressed: **an aggregate
class ratio near parity says nothing about its members.**

## Ranked, with what each actually needs

1. **transform +2263 ms (38 %)** — still the largest, and **no longer an
   unscoped programme: `benchmarks/encoder_txfm_size_census_2026-09-09.md`
   measures where inside it to start.** 4x4 is **50.7 %** of all forward
   transforms and 4x4+8x8 is **73 %**, while everything with both dims >= 32 is
   together **under 1 %** — so the big kernels, which a whole-transform rewrite
   instinctively starts with, are worth almost nothing. For a 16-coefficient
   transform the lever is **per-call driver overhead** (config derivation, the
   scratch clear/resize, two pass dispatches, the intermediate `buf`
   round-trip), not lane width — which also means KB-PERF-3's `fadst4` i16
   rejection does not block a fused 4x4, so it can serve all four DCT/ADST
   combinations (81 % of types) rather than DCT-only. **8x8 was then built
   and measured +7.07 % (0/16 rounds) and REVERTED** — the census ranks by
   call count and cannot see that 8x8's generic path is already FULL 8-wide
   SIMD on both passes, where 4x4's row pass declines and its column pass is
   half-filled. The usable rule is the conjunction: fuse where the call count
   is high AND the generic SIMD declines. Do not re-attempt 8x8 scalar fusion. Lane width itself was
   measured NULL at three frame sizes (clause-4 row); do not rebuild that.
2. **rd-driver +950 ms (16 %)** — beware: its two biggest apparent items
   (`intra_model_rd_y`, `txfm_rd_in_plane_intra`) are **inlining sinks**, not
   gaps (KB-PERF-10). Anything costed off them is costed off C's dispatch
   overhead.
3. **loop-restoration +901 ms (15 %)** — blocked on integer interleave as above.
4. **memory +448 ms at 7.43x** — now *attributable*: profile with
   `RUSTFLAGS="-C force-frame-pointer=yes"` and `--call-graph fp`, which turns
   this from an opaque class into a ranked caller list (KB-PERF-15). Remaining
   named callers: `try_inv_row_pass`, `try_fwd_col_pass`,
   `assemble_dir_edges_v4`, `quantize_fp_impl_v3`.
5. **quantize +384 ms at 3.52x** — **examined 2026-09-09 and it is NOT the
   cheap lever this line originally called it; see
   `encoder_quantize_investigation_2026-09-09.md`.** Two candidate levers were
   refuted (the `iscan` load already compiles to `vpmovsxwd`; the driver has no
   hot loop, only per-call overhead) and the real one — libaom's i16, 16-lane
   kernel — is **blocked on a missing magetypes `mul_high`**, the same class of
   blocker as wiener's interleave.

## Method notes worth carrying

* Frame pointers cost ~nothing on a profiling build and are the difference
  between "a programme" and "a ranked list". Do it first.
* Check whether a symbol is a closure or an inlining sink before costing a
  lever off it — this cycle produced three false levers that way.
* `just gate-landing` is ~11 min; a 1 MP band is ~5–15 min. **Do not run them
  concurrently** — that is what exhausted the disk quota mid-session and cost
  hours (the failure presents as every shell command exiting 1 with empty
  output; diagnose with the Write tool, which reports `EDQUOT`).
